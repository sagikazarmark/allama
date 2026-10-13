use allama::{
    CacheRetention, ContentPart, ImageDetail, PromptCache, ReasoningConfig, ReasoningEffort,
    Refusal, SystemPrompt, Text, ToolChoice, ToolDefinition, UnsupportedPolicy, WarningKind,
};
use serde_json::json;

use super::*;
use crate::responses::decode::TEST_NAMESPACE;
use crate::{SamplingSupport, StreamingSupport};

/// The OpenAI feature set under the test namespace.
fn full_quirks() -> ResponsesQuirks {
    let mut quirks = ResponsesQuirks::openai();
    quirks.namespace = TEST_NAMESPACE.into();
    quirks
}

/// Model rules shaped like OpenAI's families, so the encoder's handling of
/// each rule is exercised (the real table lives in `allama-openai`).
fn rules(model: &str) -> ResponsesModelRules {
    const GPT5: &[ReasoningEffort] = &[
        ReasoningEffort::Minimal,
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
    ];
    const O_SERIES: &[ReasoningEffort] = &[
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
    ];
    let reasoning = ResponsesModelRules::new().reasoning(CapabilitySupport::Supported);
    match model {
        "gpt-5-nano" | "gpt-5-mini" => reasoning
            .effort_levels(GPT5)
            .sampling(SamplingSupport::Rejected),
        "o4-mini" => reasoning
            .effort_levels(O_SERIES)
            .sampling(SamplingSupport::Rejected),
        "gpt-5.1" => reasoning.sampling(SamplingSupport::WithoutReasoning),
        "gpt-4.1-nano" | "gpt-4.1" => {
            ResponsesModelRules::new().reasoning(CapabilitySupport::Unsupported)
        }
        _ => ResponsesModelRules::new(),
    }
}

fn encode_for(
    request: &ChatRequest,
    stream: bool,
    options: &ResponsesWireOptions,
    quirks: &ResponsesQuirks,
) -> Result<EncodedRequest> {
    let rules = rules(&request.model);
    encode_request(
        request,
        stream,
        options,
        Target {
            provider: &std::borrow::Cow::Borrowed("test"),
            quirks,
            rules: &rules,
        },
    )
}

/// The body as it appears on the wire.
fn wire(encoded: &EncodedRequest) -> Value {
    let wire = serde_json::to_string(&encoded.body).unwrap();
    serde_json::from_str(&wire).unwrap()
}

fn encode_with(
    request: &ChatRequest,
    options: &ResponsesWireOptions,
    quirks: &ResponsesQuirks,
) -> (Value, Vec<Warning>) {
    let encoded = encode_for(request, false, options, quirks).unwrap();
    (wire(&encoded), encoded.warnings)
}

fn encode(request: &ChatRequest) -> (Value, Vec<Warning>) {
    encode_with(request, &ResponsesWireOptions::default(), &full_quirks())
}

fn encode_options(request: &ChatRequest, options: ResponsesWireOptions) -> (Value, Vec<Warning>) {
    encode_with(request, &options, &full_quirks())
}

fn encode_err_with(request: &ChatRequest, quirks: &ResponsesQuirks) -> Error {
    encode_for(request, false, &ResponsesWireOptions::default(), quirks).unwrap_err()
}

fn encode_err(request: &ChatRequest) -> Error {
    encode_err_with(request, &full_quirks())
}

fn kinds(warnings: &[Warning]) -> Vec<WarningKind> {
    warnings.iter().map(|w| w.kind).collect()
}

fn own(pairs: &[(&str, Value)]) -> ProviderData {
    data_in(TEST_NAMESPACE, pairs)
}

fn data_in(namespace: &str, pairs: &[(&str, Value)]) -> ProviderData {
    let mut data = ProviderData::new();
    for (key, value) in pairs {
        data.insert_raw(namespace, *key, value.clone());
    }
    data
}

#[test]
fn minimal_request_is_stateless() {
    let (body, warnings) = encode(&ChatRequest::new("gpt-4.1-nano").user("hi"));
    assert!(warnings.is_empty());
    assert_eq!(
        body,
        json!({
            "model": "gpt-4.1-nano",
            "store": false,
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}]
        })
    );
    let streamed = encode_for(
        &ChatRequest::new("m").user("hi"),
        true,
        &ResponsesWireOptions::default(),
        &full_quirks(),
    )
    .unwrap();
    assert!(streamed.body.stream);
}

#[test]
fn system_prompts_become_role_messages() {
    let request = ChatRequest::new("gpt-5-mini")
        .system("Be brief.")
        .system(SystemPrompt::new("Never lie.").with_option(SystemOptions::developer()))
        .system("")
        .user("hi");
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty());
    let input = body["input"].as_array().unwrap();
    assert_eq!(input.len(), 3);
    assert_eq!(input[0]["role"], "system");
    assert_eq!(input[0]["content"][0]["text"], "Be brief.");
    assert_eq!(input[1]["role"], "developer");
    assert_eq!(input[2]["role"], "user");
    assert!(body.get("instructions").is_none());
}

#[test]
fn user_parts_map_to_input_content() {
    let request = ChatRequest::new("gpt-4.1").user(vec![
        ContentPart::text("look"),
        ContentPart::Image(Image::url("https://x/y.png").with_detail(ImageDetail::High)),
        ContentPart::image_base64("image/png", "AAAA"),
        ContentPart::file_base64("application/pdf", "JVBE"),
        ContentPart::File(File::base64("text/plain", "aGk=").with_filename("notes.txt")),
        ContentPart::file_url("application/pdf", "https://x/doc.pdf"),
        ContentPart::Other(
            OtherContent::new(
                "input_text",
                serde_json::Map::from_iter([("text".into(), json!("raw"))]),
            )
            .with_origin(TEST_NAMESPACE),
        ),
    ]);
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["input"][0]["content"],
        json!([
            {"type": "input_text", "text": "look"},
            {"type": "input_image", "image_url": "https://x/y.png", "detail": "high"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
            {"type": "input_file", "filename": "document.pdf", "file_data": "data:application/pdf;base64,JVBE"},
            {"type": "input_file", "filename": "notes.txt", "file_data": "data:text/plain;base64,aGk="},
            {"type": "input_file", "file_url": "https://x/doc.pdf"},
            {"type": "input_text", "text": "raw"}
        ])
    );
}

#[test]
fn unsupported_user_content_fails() {
    let audio =
        ChatRequest::new("gpt-4.1").user(vec![ContentPart::file_base64("audio/wav", "AAAA")]);
    assert!(matches!(encode_err(&audio), Error::Unsupported { .. }));
}

#[test]
fn foreign_parts_are_dropped_with_a_warning() {
    let foreign = || {
        ContentPart::Other(OtherContent::new("document", ExtraMap::new()).with_origin("anthropic"))
    };
    let request = ChatRequest::new("gpt-4.1")
        .user(vec![ContentPart::text("look"), foreign()])
        .message(Message::assistant_blocks(vec![ContentBlock::tool_call(
            "c1", "lookup", "{}",
        )]))
        .message(Message::tool_result(
            "c1",
            "lookup",
            vec![ContentPart::text("found"), foreign()],
        ));
    let (body, warnings) = encode(&request);
    assert_eq!(
        body["input"][0]["content"],
        json!([{"type": "input_text", "text": "look"}])
    );
    assert_eq!(
        body["input"][2]["output"],
        json!([{"type": "input_text", "text": "found"}])
    );
    // Both drops share a code: one warning per request.
    assert_eq!(kinds(&warnings), vec![WarningKind::Dropped]);
}

#[test]
fn assistant_turn_replays_items_in_order() {
    let reasoning = Reasoning::new("Thinking about Paris.").with_provider_data(own(&[
        ("item_id", json!("rs_1")),
        ("encrypted_content", json!("gAAA")),
    ]));
    let call = ToolCall::new("call_1", "get_weather", r#"{"city":"Paris"}"#)
        .with_provider_data(own(&[("item_id", json!("fc_1"))]));
    let hosted = OtherContent::new(
        "web_search_call",
        serde_json::Map::from_iter([
            ("id".into(), json!("ws_1")),
            ("status".into(), json!("completed")),
        ]),
    )
    .with_origin(TEST_NAMESPACE);
    let request = ChatRequest::new("gpt-5-nano").extend_messages([
        Message::user("Weather?"),
        Message::assistant_blocks(vec![
            ContentBlock::Reasoning(reasoning),
            ContentBlock::text("Let me check."),
            ContentBlock::Text(
                Text::new("Note.")
                    .with_provider_data(own(&[("annotations", json!([{"type": "url_citation"}]))])),
            ),
            ContentBlock::ToolCall(call),
            ContentBlock::Other(hosted),
            ContentBlock::Refusal(Refusal::new("No.")),
            // A refusal an earlier version stored as flagged text.
            ContentBlock::Text(
                Text::new("Still no.").with_provider_data(own(&[("refusal", json!(true))])),
            ),
        ]),
        Message::tool_result("call_1", "get_weather", "Sunny"),
    ]);
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["input"],
        json!([
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Weather?"}]},
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAA",
             "summary": [{"type": "summary_text", "text": "Thinking about Paris."}]},
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "Let me check.", "annotations": []},
                {"type": "output_text", "text": "Note.", "annotations": [{"type": "url_citation"}]}
            ]},
            {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_weather",
             "arguments": "{\"city\":\"Paris\"}"},
            {"type": "web_search_call", "id": "ws_1", "status": "completed"},
            {"type": "message", "role": "assistant", "content": [
                {"type": "refusal", "refusal": "No."},
                {"type": "refusal", "refusal": "Still no."}
            ]},
            {"type": "function_call_output", "call_id": "call_1", "output": "Sunny"}
        ])
    );
}

#[test]
fn redacted_reasoning_replays_with_empty_summary_and_phase_splits_messages() {
    let request =
        ChatRequest::new("gpt-5-nano").extend_messages([Message::assistant_blocks(vec![
            ContentBlock::Text(
                Text::new("Working on it.")
                    .with_provider_data(own(&[("phase", json!("commentary"))])),
            ),
            ContentBlock::Reasoning(
                Reasoning::new("").with_provider_data(own(&[("encrypted_content", json!("gBBB"))])),
            ),
            ContentBlock::Text(
                Text::new("Done.").with_provider_data(own(&[("phase", json!("final_answer"))])),
            ),
        ])]);
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty());
    assert_eq!(body["input"][0]["phase"], "commentary");
    assert_eq!(
        body["input"][1],
        json!({"type": "reasoning", "encrypted_content": "gBBB", "summary": []})
    );
    assert_eq!(body["input"][2]["phase"], "final_answer");
}

#[test]
fn raw_reasoning_content_replays_without_encrypted_content() {
    let raw = json!([{"type": "reasoning_text", "text": "Let me think."}]);
    let request =
        ChatRequest::new("gpt-oss-20b").extend_messages([Message::assistant_blocks(vec![
            // Decoded from a vLLM-style item: text from raw content, no summary.
            ContentBlock::Reasoning(Reasoning::new("Let me think.").with_provider_data(own(&[
                ("item_id", json!("rs_1")),
                ("content", raw.clone()),
                (TEXT_SOURCE_KEY, json!("content")),
            ]))),
            // Summary text plus raw content (both kept).
            ContentBlock::Reasoning(Reasoning::new("Summary.").with_provider_data(own(&[
                ("content", raw.clone()),
                ("encrypted_content", json!("gAAA")),
            ]))),
        ])]);
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["input"][0],
        json!({"type": "reasoning", "id": "rs_1", "summary": [], "content": raw})
    );
    assert_eq!(
        body["input"][1],
        json!({"type": "reasoning", "encrypted_content": "gAAA",
               "summary": [{"type": "summary_text", "text": "Summary."}], "content": raw})
    );
}

#[test]
fn foreign_and_stateless_reasoning_is_dropped_with_warnings() {
    let anthropic = Reasoning::new("thoughts").with_provider_data(ProviderData::new().with_raw(
        "anthropic",
        "signature",
        json!("sig"),
    ));
    let request =
        ChatRequest::new("gpt-5-nano").extend_messages([Message::assistant_blocks(vec![
            ContentBlock::Reasoning(anthropic),
            ContentBlock::Reasoning(Reasoning::new("plain text reasoning")),
            ContentBlock::Other(
                OtherContent::new("server_tool_use", ExtraMap::new()).with_origin("anthropic"),
            ),
            ContentBlock::ToolCall(ToolCall::new("c", "f", "{}").with_provider_data(
                ProviderData::new().with_raw("google-ai", "thought_signature", json!("x")),
            )),
        ])]);
    let (body, warnings) = encode(&request);
    assert_eq!(kinds(&warnings), vec![WarningKind::Dropped; 4]);
    assert!(
        warnings[0].message.contains("anthropic"),
        "{}",
        warnings[0].message
    );
    assert!(
        warnings[1].message.contains("encrypted or raw content"),
        "{}",
        warnings[1].message
    );
    assert!(warnings[3].message.contains("google-ai"));
    // Only the tool call (without its foreign state) is replayed.
    assert_eq!(
        body["input"],
        json!([{"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"}])
    );
}

#[test]
fn core_provider_data_is_not_reported_as_foreign() {
    let core = ProviderData::new().with_raw("allama", "note", json!(1));
    let request = ChatRequest::new("gpt-5-nano").extend_messages([
        Message::user(vec![ContentPart::Text(
            allama::Text::new("hi").with_provider_data(core.clone()),
        )]),
        Message::assistant_blocks(vec![ContentBlock::ToolCall(
            ToolCall::new("c", "f", "{}").with_provider_data(core),
        )]),
    ]);
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(!body.to_string().contains("note"));
}

#[test]
fn assistant_images_cannot_be_replayed() {
    let request = ChatRequest::new("gpt-5").extend_messages([Message::assistant_blocks(vec![
        ContentBlock::Image(Image::url("https://x/y.png")),
    ])]);
    assert!(matches!(encode_err(&request), Error::Unsupported { .. }));
}

#[test]
fn malformed_tool_arguments_are_replayed_verbatim() {
    let request = ChatRequest::new("gpt-4.1").extend_messages([Message::assistant_blocks(vec![
        ContentBlock::tool_call("c", "f", r#"{"city": "Par"#),
    ])]);
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty());
    assert_eq!(body["input"][0]["arguments"], r#"{"city": "Par"#);
}

#[test]
fn tool_results_support_objects_parts_and_error_hint() {
    let request = ChatRequest::new("gpt-4.1").extend_messages([
        Message::tool_result(
            "c1",
            "f",
            serde_json::Map::from_iter([("ok".into(), json!(true))]),
        ),
        Message::tool_result(
            "c2",
            "f",
            vec![
                ContentPart::text("chart:"),
                ContentPart::image_base64("image/png", "AAAA"),
            ],
        ),
        Message::tool_error("c3", "f", "boom"),
    ]);
    let (body, warnings) = encode(&request);
    assert_eq!(body["input"][0]["output"], r#"{"ok":true}"#);
    assert_eq!(
        body["input"][1]["output"],
        json!([
            {"type": "input_text", "text": "chart:"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
        ])
    );
    assert_eq!(body["input"][2]["output"], "boom");
    assert_eq!(kinds(&warnings), vec![WarningKind::Approximated]);
    assert_eq!(
        warnings[0].code.as_deref(),
        Some("approximated.tool_result_error")
    );
    assert_eq!(
        warnings[0].subject,
        Subject::Content {
            message: 2,
            block: None
        }
    );

    assert_eq!(
        warnings[0].message,
        "the Responses API has no tool error flag; the error text was sent as the tool result; the model still reads the error text; to send your own encoding to this endpoint, use `push_tool_result`"
    );

    // Sent under a strict policy too: the model still reads the error text.
    let strict = request.unsupported_policy(UnsupportedPolicy::Error);
    let (body, warnings) = encode(&strict);
    assert_eq!(body["input"][2]["output"], "boom");
    assert_eq!(kinds(&warnings), vec![WarningKind::Approximated]);
}

#[test]
fn message_names_are_hints() {
    let mut message = Message::user("hi");
    message.provider_data_mut().insert_raw(
        TEST_NAMESPACE,
        crate::PARTICIPANT_NAME_KEY,
        "alice".into(),
    );
    let (_, warnings) = encode(&ChatRequest::new("gpt-4.1").message(message.clone()));
    assert_eq!(kinds(&warnings), vec![WarningKind::Unsupported]);
}

#[test]
fn tools_send_strict_explicitly_and_append_hosted_tools() {
    let schema = json!({"type": "object", "properties": {"q": {"type": "string"}, "n": {"type": "integer"}}, "required": ["q"]});
    let mut extra = ToolDefinition::new("extra", json!({"type": "object"}));
    extra
        .provider_data
        .insert_raw(TEST_NAMESPACE, "defer_loading", json!(true));
    let request = ChatRequest::new("gpt-4.1").user("hi").tools([
        ToolDefinition::new("loose", schema.clone()).description("Loose"),
        ToolDefinition::new("tight", schema).strict(true),
        extra,
    ]);
    let (body, warnings) = encode_options(
        &request,
        ResponsesWireOptions {
            hosted_tools: vec![serde_json::Map::from_iter([(
                "type".into(),
                json!("web_search"),
            )])],
            ..Default::default()
        },
    );
    assert!(warnings.is_empty());
    let tools = body["tools"].as_array().unwrap();
    assert_eq!(tools[0]["strict"], false);
    assert_eq!(tools[0]["description"], "Loose");
    assert_eq!(tools[0]["parameters"]["required"], json!(["q"]));
    assert_eq!(tools[1]["strict"], true);
    assert_eq!(tools[1]["parameters"]["additionalProperties"], false);
    assert_eq!(tools[2]["defer_loading"], true);
    assert_eq!(tools[3], json!({"type": "web_search"}));
}

#[test]
fn client_tools_follow_hosted_tools() {
    let tool = |value: Value| -> ExtraMap { serde_json::from_value(value).unwrap() };
    let (body, warnings) = encode_options(
        &ChatRequest::new("gpt-4.1")
            .user("hi")
            .tools([ToolDefinition::new("f", json!({"type": "object"}))]),
        ResponsesWireOptions {
            hosted_tools: vec![tool(json!({"type": "web_search"}))],
            client_tools: vec![
                tool(json!({"type": "custom", "name": "sql"})),
                tool(json!({"type": "local_shell"})),
            ],
            ..Default::default()
        },
    );
    assert!(warnings.is_empty());
    let types: Vec<&str> = body["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["type"].as_str().unwrap())
        .collect();
    assert_eq!(types, ["function", "web_search", "custom", "local_shell"]);
}

#[test]
fn client_tool_outputs_are_input_items() {
    use crate::{ClientToolCall, ShellCommandOutput};
    let item = |type_name: &str, data: Value| {
        ContentBlock::Other(
            OtherContent::new(type_name, serde_json::from_value(data).unwrap())
                .with_origin(TEST_NAMESPACE),
        )
    };
    let turn = vec![
        item(
            "custom_tool_call",
            json!({"id": "ctc_1", "call_id": "call_1", "name": "sql", "input": "SELECT 1", "status": "completed"}),
        ),
        item(
            "local_shell_call",
            json!({"id": "lsh_1", "call_id": "call_2", "status": "completed",
                   "action": {"type": "exec", "command": ["ls"], "env": {}}}),
        ),
        item(
            "shell_call",
            json!({"id": "sh_1", "call_id": "call_3", "status": "completed",
                   "action": {"commands": ["pwd"], "timeout_ms": null, "max_output_length": null}}),
        ),
    ];
    let mut answers = vec![ContentPart::text("Results:")];
    for call in ClientToolCall::pending(&turn) {
        answers.push(match call {
            ClientToolCall::Custom(call) => call.output("1"),
            ClientToolCall::LocalShell(call) => call.output("a.txt"),
            ClientToolCall::Shell(call) => call.output([ShellCommandOutput::exited("/\n", "", 0)]),
            other => panic!("unexpected {other:?}"),
        });
    }
    answers.push(ContentPart::text("Go on."));
    let request = ChatRequest::new("gpt-4.1")
        .user("hi")
        .message(Message::assistant_blocks(turn))
        .message(Message::user(answers));
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["input"],
        json!([
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1", "name": "sql", "input": "SELECT 1", "status": "completed"},
            {"type": "local_shell_call", "id": "lsh_1", "call_id": "call_2", "status": "completed",
             "action": {"type": "exec", "command": ["ls"], "env": {}}},
            {"type": "shell_call", "id": "sh_1", "call_id": "call_3", "status": "completed",
             "action": {"commands": ["pwd"], "timeout_ms": null, "max_output_length": null}},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Results:"}]},
            {"type": "custom_tool_call_output", "call_id": "call_1", "output": "1"},
            {"type": "local_shell_call_output", "id": "call_2", "output": "a.txt"},
            {"type": "shell_call_output", "call_id": "call_3", "output": [
                {"stdout": "/\n", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}
            ]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Go on."}]}
        ])
    );

    // A user message holding only answers sends no empty message.
    let only = ChatRequest::new("gpt-4.1").user(vec![ContentPart::Other(
        OtherContent::new(
            "custom_tool_call_output",
            serde_json::from_value(json!({"call_id": "call_1", "output": "1"})).unwrap(),
        )
        .with_origin(TEST_NAMESPACE),
    )]);
    assert_eq!(
        encode(&only).0["input"],
        json!([{"type": "custom_tool_call_output", "call_id": "call_1", "output": "1"}])
    );

    // Another endpoint's answer is not sent: it is dropped with a warning,
    // under every policy.
    for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
        let mut foreign = ChatRequest::new("gpt-4.1").user(vec![
            ContentPart::text("Go on."),
            ContentPart::Other(
                OtherContent::new("custom_tool_call_output", ExtraMap::new()).with_origin("other"),
            ),
        ]);
        foreign.unsupported_policy = Some(policy);
        let (body, warnings) = encode(&foreign);
        assert_eq!(
            body["input"],
            json!([{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Go on."}]}]),
            "{policy:?}"
        );
        assert_eq!(kinds(&warnings), vec![WarningKind::Dropped], "{policy:?}");
        assert_eq!(
            warnings[0].subject,
            Subject::Content {
                message: 0,
                block: Some(1)
            }
        );
    }
}

/// Server-side continuation keeps the call on the server: a request whose
/// messages hold only the answer sends it.
#[test]
fn a_continued_request_sends_an_answer_without_its_call() {
    let request =
        ChatRequest::new("gpt-4.1").message(Message::user_parts(vec![ContentPart::Other(
            OtherContent::new(
                "custom_tool_call_output",
                serde_json::from_value(json!({"call_id": "call_1", "output": "42"})).unwrap(),
            )
            .with_origin(TEST_NAMESPACE),
        )]));
    let (body, warnings) = encode_options(
        &request,
        ResponsesWireOptions {
            store: Some(true),
            previous_response_id: Some("resp_1".into()),
            ..ResponsesWireOptions::default()
        },
    );
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(body["previous_response_id"], "resp_1");
    assert_eq!(
        body["input"],
        json!([{"type": "custom_tool_call_output", "call_id": "call_1", "output": "42"}])
    );
}

/// A custom call decoded from Chat Completions has no item `id` or
/// `status`; it is replayed as it is.
#[test]
fn a_chat_decoded_custom_call_is_replayed() {
    let call = OtherContent::new(
        "custom_tool_call",
        serde_json::from_value(json!({"call_id": "call_1", "name": "sql", "input": "SELECT 1"}))
            .unwrap(),
    )
    .with_origin(TEST_NAMESPACE);
    let answer = crate::CustomToolCall::from_content(&call)
        .unwrap()
        .output("1");
    let request = ChatRequest::new("gpt-4.1")
        .user("Count.")
        .message(Message::assistant_blocks(vec![ContentBlock::Other(call)]))
        .message(Message::user(vec![answer]));
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["input"].as_array().unwrap()[1..],
        [
            json!({"type": "custom_tool_call", "call_id": "call_1", "name": "sql", "input": "SELECT 1"}),
            json!({"type": "custom_tool_call_output", "call_id": "call_1", "output": "1"})
        ]
    );
}

/// A client tool call that is in the request but not replayed (here, one
/// from another endpoint) takes its answer with it.
#[test]
fn an_answer_is_dropped_with_its_unreplayed_call() {
    let data = |value: Value| serde_json::from_value::<ExtraMap>(value).unwrap();
    let request = ChatRequest::new("gpt-4.1")
        .user("Count.")
        .message(Message::assistant_blocks(vec![
            ContentBlock::text("Counting."),
            ContentBlock::Other(
                OtherContent::new(
                    "custom_tool_call",
                    data(json!({"call_id": "call_1", "name": "sql", "input": "SELECT 1"})),
                )
                .with_origin("other"),
            ),
        ]))
        .message(Message::user_parts(vec![
            ContentPart::Other(
                OtherContent::new(
                    "custom_tool_call_output",
                    data(json!({"call_id": "call_1", "output": "1"})),
                )
                .with_origin(TEST_NAMESPACE),
            ),
            ContentPart::text("Well?"),
        ]));
    let (body, warnings) = encode(&request);
    assert_eq!(
        body["input"],
        json!([
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Count."}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Counting.", "annotations": []}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Well?"}]}
        ])
    );
    let subjects: Vec<&Subject> = warnings.iter().map(|w| &w.subject).collect();
    assert_eq!(
        subjects,
        [
            &Subject::Content {
                message: 1,
                block: Some(1)
            },
            &Subject::Content {
                message: 2,
                block: Some(0)
            }
        ]
    );
    assert!(
        warnings[1].message.contains("answers was not replayed"),
        "{warnings:?}"
    );
}

/// A provider tool call of this namespace without the item it was decoded
/// from cannot be replayed, nor can its result: both drops recur with the
/// history, as content replay state.
#[test]
fn own_provider_tool_calls_without_their_item_are_dropped_as_content_replay() {
    let call =
        allama::ProviderToolCall::new("ws_1", "web_search", "{}").with_origin(TEST_NAMESPACE);
    let mut result = allama::ProviderToolResult::new("ws_1", json!({"status": "completed"}));
    result.origin = Some(TEST_NAMESPACE.into());
    let request = ChatRequest::new("gpt-5")
        .user("Search.")
        .message(Message::assistant_blocks(vec![
            ContentBlock::ProviderToolCall(call),
            ContentBlock::ProviderToolResult(result),
            ContentBlock::text("Found it."),
        ]));
    let (_, warnings) = encode(&request);
    let dropped: Vec<_> = warnings
        .iter()
        .map(|w| {
            (
                w.kind,
                w.subject.clone(),
                w.message.as_str(),
                w.code.as_deref(),
                w.replay_state(),
            )
        })
        .collect();
    let block = |block| Subject::Content {
        message: 1,
        block: Some(block),
    };
    assert_eq!(
        dropped,
        [
            (
                WarningKind::Dropped,
                block(0),
                "the Responses API cannot replay provider tool call `web_search`; it was dropped",
                Some("dropped.content_replay"),
                Some(ReplayState::Content),
            ),
            // The orphaned result's drop shares the code, and the message:
            // one warning per code per request.
        ]
    );
}

/// The caller's own raw part (no origin) is sent verbatim.
#[test]
fn raw_parts_without_an_origin_are_sent() {
    let raw = OtherContent::new(
        "input_text",
        serde_json::from_value(json!({"text": "raw"})).unwrap(),
    );
    let request = ChatRequest::new("gpt-4.1").user(vec![ContentPart::Other(raw)]);
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["input"][0]["content"],
        json!([{"type": "input_text", "text": "raw"}])
    );
}

#[test]
fn tool_choice_variants() {
    let choice = |c: ToolChoice| {
        encode(&ChatRequest::new("m").user("hi").tool_choice(c)).0["tool_choice"].clone()
    };
    assert_eq!(choice(ToolChoice::Auto), "auto");
    assert_eq!(choice(ToolChoice::Disabled), "none");
    assert_eq!(choice(ToolChoice::Required), "required");
    assert_eq!(
        choice(ToolChoice::Specific { name: "f".into() }),
        json!({"type": "function", "name": "f"})
    );
}

#[test]
fn response_formats_map_to_text_format() {
    let format = |f: ResponseFormat| {
        encode(&ChatRequest::new("m").user("hi").response_format(f)).0["text"].clone()
    };
    assert_eq!(
        format(ResponseFormat::Json),
        json!({"format": {"type": "json_object"}})
    );
    assert_eq!(
        format(ResponseFormat::Text),
        json!({"format": {"type": "text"}})
    );
    let schema = json!({"type": "object", "properties": {"a": {"type": "string"}}});
    let strict = format(ResponseFormat::json_schema(schema.clone()).name("answer"));
    assert_eq!(strict["format"]["name"], "answer");
    assert_eq!(strict["format"]["strict"], true);
    assert_eq!(strict["format"]["schema"]["required"], json!(["a"]));
    // `strict: None` (from a stored request) does not require it either.
    let mut unset = ResponseFormat::json_schema(schema.clone());
    if let ResponseFormat::JsonSchema { strict, .. } = &mut unset {
        *strict = None;
    }
    let loose = format(unset);
    assert_eq!(
        loose,
        json!({"format": {"type": "json_schema", "name": "response", "schema": schema, "strict": false}})
    );
}

/// `json_schema` is strict unless relaxed with `non_strict`, which sends
/// the schema as written with `strict: false`.
#[test]
fn json_schema_is_strict_unless_relaxed() {
    let schema = json!({"type": "object", "properties": {"a": {"type": "string"}}});
    let (body, warnings) = encode(
        &ChatRequest::new("m")
            .user("hi")
            .response_format(ResponseFormat::json_schema(schema.clone())),
    );
    let format = &body["text"]["format"];
    assert_eq!(format["strict"], true);
    assert_eq!(format["schema"]["required"], json!(["a"]));
    assert!(warnings.is_empty(), "{warnings:?}");

    let (body, warnings) = encode(
        &ChatRequest::new("m")
            .user("hi")
            .response_format(ResponseFormat::json_schema(schema.clone()).non_strict()),
    );
    let format = &body["text"]["format"];
    assert_eq!(format["strict"], false);
    assert_eq!(format["schema"], schema);
    assert!(warnings.is_empty(), "{warnings:?}");
}

#[test]
fn reasoning_effort_maps_to_supported_levels() {
    let effort = |model: &str, effort: ReasoningEffort| {
        let (body, warnings) = encode(&ChatRequest::new(model).user("hi").reasoning(effort));
        (body["reasoning"].clone(), kinds(&warnings))
    };
    assert_eq!(
        effort("gpt-5-nano", ReasoningEffort::Low),
        (json!({"effort": "low"}), vec![])
    );
    assert_eq!(
        effort("gpt-5-nano", ReasoningEffort::None),
        (
            json!({"effort": "minimal"}),
            vec![WarningKind::Approximated]
        )
    );
    assert_eq!(
        effort("gpt-5-nano", ReasoningEffort::XHigh),
        (json!({"effort": "high"}), vec![WarningKind::Approximated])
    );
    assert_eq!(
        effort("o4-mini", ReasoningEffort::Minimal),
        (json!({"effort": "low"}), vec![WarningKind::Approximated])
    );
    // Unclassified levels pass through.
    assert_eq!(
        effort("gpt-5.1", ReasoningEffort::None),
        (json!({"effort": "none"}), vec![])
    );
    // Non-reasoning models: the effort is an unsupported hint, except `None`.
    assert_eq!(
        effort("gpt-4.1-nano", ReasoningEffort::Low),
        (Value::Null, vec![WarningKind::Unsupported])
    );
    assert_eq!(
        effort("gpt-4.1-nano", ReasoningEffort::None),
        (Value::Null, vec![])
    );
    let strict = ChatRequest::new("gpt-4.1-nano")
        .user("hi")
        .reasoning(ReasoningEffort::Low)
        .unsupported_policy(UnsupportedPolicy::Error);
    assert!(matches!(encode_err(&strict), Error::Unsupported { .. }));
}

#[test]
fn reasoning_output_maps_to_summary() {
    let summary = |output: ReasoningOutput| {
        let request = ChatRequest::new("gpt-5-mini")
            .user("hi")
            .reasoning(ReasoningConfig::new().output(output));
        let (body, warnings) = encode(&request);
        (body["reasoning"].clone(), kinds(&warnings))
    };
    assert_eq!(
        summary(ReasoningOutput::Summary),
        (json!({"summary": "auto"}), vec![])
    );
    assert_eq!(
        summary(ReasoningOutput::Full),
        (
            json!({"summary": "detailed"}),
            vec![WarningKind::Approximated]
        )
    );
    assert_eq!(summary(ReasoningOutput::Omitted), (Value::Null, vec![]));

    let raw = ChatRequest::new("gpt-5-mini")
        .user("hi")
        .reasoning(ReasoningEffort::Low);
    let (body, warnings) = encode_options(
        &raw,
        ResponsesWireOptions {
            reasoning_effort: Some("ultra".into()),
            ..Default::default()
        },
    );
    assert_eq!(body["reasoning"], json!({"effort": "ultra"}));
    assert_eq!(kinds(&warnings), vec![WarningKind::Approximated]);
}

#[test]
fn sampling_parameters_follow_the_model() {
    let sampled = |model: &str| {
        ChatRequest::new(model)
            .user("hi")
            .temperature(0.5)
            .top_p(0.9)
            .frequency_penalty(0.1)
            .presence_penalty(0.2)
    };
    let (body, warnings) = encode(&sampled("gpt-4.1-nano"));
    assert!(warnings.is_empty());
    assert_eq!(body["temperature"], 0.5);
    assert_eq!(body["presence_penalty"], 0.2);

    let (body, warnings) = encode(&sampled("gpt-5-nano"));
    assert_eq!(kinds(&warnings), vec![WarningKind::Unsupported; 4]);
    for field in [
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
    ] {
        assert!(body.get(field).is_none(), "{field} was sent");
    }

    // GPT-5.1 accepts sampling only without reasoning.
    let (body, warnings) = encode(&sampled("gpt-5.1"));
    assert!(warnings.is_empty());
    assert_eq!(body["top_p"], 0.9);
    let (body, _) = encode(&sampled("gpt-5.1").reasoning(ReasoningEffort::None));
    assert!(body.get("temperature").is_some());

    // "When reasoning effort is not none, remove temperature, top_p, and
    // top_logprobs" (<https://developers.openai.com/api/docs/guides/latest-model>).
    let (body, warnings) = encode_options(
        &ChatRequest::new("gpt-5-mini").user("hi"),
        ResponsesWireOptions {
            extra: serde_json::Map::from_iter([("top_logprobs".into(), json!(2))]),
            ..Default::default()
        },
    );
    assert!(body.get("top_logprobs").is_none());
    assert_eq!(
        warnings[0].code.as_deref(),
        Some("unsupported.top_logprobs")
    );
    let (body, warnings) = encode(&sampled("gpt-5.1").reasoning(ReasoningEffort::Low));
    assert!(body.get("temperature").is_none());
    assert_eq!(warnings.len(), 4);

    let strict = sampled("gpt-5-nano").unsupported_policy(UnsupportedPolicy::Error);
    assert!(matches!(encode_err(&strict), Error::Unsupported { .. }));
}

#[test]
fn seed_and_stop_are_hints() {
    let request = ChatRequest::new("gpt-4.1").user("hi").seed(1).stop(["END"]);
    let (body, warnings) = encode(&request);
    assert_eq!(kinds(&warnings), vec![WarningKind::Unsupported; 2]);
    assert!(body.get("seed").is_none() && body.get("stop").is_none());
    assert!(matches!(
        encode_err(&request.unsupported_policy(UnsupportedPolicy::Error)),
        Error::Unsupported { .. }
    ));
}

#[test]
fn options_map_and_extra_is_guarded() {
    let request = ChatRequest::new("gpt-4.1")
        .user("hi")
        .cache(PromptCache::new().key("k").retention(CacheRetention::Long));
    let (body, warnings) = encode_options(
        &request,
        ResponsesWireOptions {
            service_tier: Some("flex".into()),
            safety_identifier: Some("hash".into()),
            truncation: Some("auto".into()),
            verbosity: Some("low".into()),
            extra: serde_json::Map::from_iter([("top_logprobs".into(), json!(2))]),
            ..Default::default()
        },
    );
    assert!(warnings.is_empty());
    assert_eq!(body["service_tier"], "flex");
    assert_eq!(body["safety_identifier"], "hash");
    assert_eq!(body["truncation"], "auto");
    assert_eq!(body["text"], json!({"verbosity": "low"}));
    assert_eq!(body["top_logprobs"], 2);
    assert_eq!(body["prompt_cache_key"], "k");
    assert_eq!(body["prompt_cache_retention"], "24h");

    for key in ["previous_response_id", "store", "background", "input"] {
        let options = ResponsesWireOptions {
            extra: serde_json::Map::from_iter([(key.into(), json!(true))]),
            ..Default::default()
        };
        let err = encode_for(
            &ChatRequest::new("m").user("hi"),
            false,
            &options,
            &full_quirks(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }), "{key}");
        assert_eq!(err.context().unwrap().provider.as_deref(), Some("test"));
    }
}

#[test]
fn default_filenames() {
    assert_eq!(default_filename("application/pdf"), "document.pdf");
    assert_eq!(default_filename("text/csv"), "document.csv");
    assert_eq!(default_filename("application/x-unknown"), "file");
}

/// A turn decoded under one namespace, replayed to an endpoint with another.
fn turn_minted_in(namespace: &str) -> ChatRequest {
    let pairs = |pairs: &[(&str, Value)]| data_in(namespace, pairs);
    ChatRequest::new("m").extend_messages([
        Message::user("Weather?"),
        Message::assistant_blocks(vec![
            ContentBlock::Reasoning(Reasoning::new("Thinking.").with_provider_data(pairs(&[
                ("item_id", json!("rs_1")),
                ("encrypted_content", json!("gAAA")),
            ]))),
            ContentBlock::ToolCall(
                ToolCall::new("call_1", "get_weather", "{}")
                    .with_provider_data(pairs(&[("item_id", json!("fc_1"))])),
            ),
            ContentBlock::Other(
                OtherContent::new("web_search_call", ExtraMap::new()).with_origin(namespace),
            ),
        ]),
        Message::tool_result("call_1", "get_weather", "Sunny"),
    ])
}

#[test]
fn namespaces_are_isolated_in_both_directions() {
    let openai = ResponsesQuirks::openai();
    let endpoint = full_quirks();
    for (minted, replayed_to) in [(TEST_NAMESPACE, &openai), (TEST_NAMESPACE, &endpoint)]
        .into_iter()
        .chain([("openai", &endpoint), ("openai", &openai)])
    {
        let (body, warnings) = encode_with(
            &turn_minted_in(minted),
            &ResponsesWireOptions::default(),
            replayed_to,
        );
        let input = body["input"].as_array().unwrap();
        if minted == replayed_to.namespace {
            assert!(warnings.is_empty(), "{minted}: {warnings:?}");
            assert_eq!(input[1]["type"], "reasoning");
            assert_eq!(input[2]["id"], "fc_1");
            assert_eq!(input[3]["type"], "web_search_call");
            continue;
        }
        // Reasoning, the item id, and the hosted item are not replayed; the
        // tool call itself is portable.
        assert_eq!(
            kinds(&warnings),
            vec![WarningKind::Dropped; 3],
            "{minted} -> {}",
            replayed_to.namespace
        );
        assert!(
            warnings[0].message.contains(minted),
            "{}",
            warnings[0].message
        );
        assert_eq!(
            input
                .iter()
                .map(|i| i["type"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["message", "function_call", "function_call_output"]
        );
        assert!(input[1].get("id").is_none());
    }
}

#[test]
fn reasoning_replay_can_be_disabled() {
    let mut quirks = full_quirks();
    quirks.reasoning_replay = ReasoningReplay::Disabled;
    let (body, warnings) = encode_with(
        &turn_minted_in(TEST_NAMESPACE),
        &ResponsesWireOptions::default(),
        &quirks,
    );
    assert_eq!(kinds(&warnings), vec![WarningKind::Dropped]);
    assert!(warnings[0].message.contains("takes no reasoning back"));
    assert_eq!(body["input"][1]["type"], "function_call");
    assert_eq!(body["input"][1]["id"], "fc_1");
}

#[test]
fn streaming_support_is_enforced_before_sending() {
    let request = ChatRequest::new("m").user("hi");
    let mut quirks = full_quirks();
    let attempt = |quirks: &ResponsesQuirks, stream: bool| {
        encode_for(&request, stream, &ResponsesWireOptions::default(), quirks)
    };
    assert!(attempt(&quirks, true).is_ok() && attempt(&quirks, false).is_ok());

    quirks.streaming = StreamingSupport::NonStreamingOnly;
    assert!(attempt(&quirks, false).is_ok());
    let err = attempt(&quirks, true).unwrap_err();
    assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
    assert!(err.to_string().contains("streaming"), "{err}");

    quirks.streaming = StreamingSupport::StreamOnly;
    assert!(attempt(&quirks, true).is_ok());
    let err = attempt(&quirks, false).unwrap_err();
    assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
    assert_eq!(err.context().unwrap().provider.as_deref(), Some("test"));
}

#[test]
fn strict_is_omitted_and_reported_without_strict_support() {
    let schema = json!({"type": "object", "properties": {"q": {"type": "string"}, "n": {"type": "integer"}}, "required": ["q"]});
    let mut quirks = full_quirks();
    quirks.supports_strict = false;
    let request = ChatRequest::new("m")
        .user("hi")
        .tools([
            ToolDefinition::new("loose", schema.clone()),
            ToolDefinition::new("tight", schema.clone()).strict(true),
        ])
        .response_format(ResponseFormat::json_schema(schema.clone()));
    let (body, warnings) = encode_with(&request, &ResponsesWireOptions::default(), &quirks);
    assert_eq!(kinds(&warnings), vec![WarningKind::Unsupported; 2]);
    assert!(
        warnings[0].message.contains("tight"),
        "{}",
        warnings[0].message
    );
    for tool in body["tools"].as_array().unwrap() {
        assert!(tool.get("strict").is_none(), "{tool}");
        assert_eq!(tool["parameters"], schema, "schema is sent as written");
    }
    assert_eq!(
        body["text"]["format"],
        json!({"type": "json_schema", "name": "response", "schema": schema})
    );

    // A loose request needs no report.
    let loose = ChatRequest::new("m")
        .user("hi")
        .tools([ToolDefinition::new("loose", schema)]);
    let (_, warnings) = encode_with(&loose, &ResponsesWireOptions::default(), &quirks);
    assert!(warnings.is_empty());

    let strict = request.unsupported_policy(UnsupportedPolicy::Error);
    assert!(matches!(
        encode_err_with(&strict, &quirks),
        Error::Unsupported { .. }
    ));
}

#[test]
fn function_call_status_null_is_sent_when_required() {
    let request = ChatRequest::new("m").extend_messages([Message::assistant_blocks(vec![
        ContentBlock::tool_call("c", "f", "{}"),
    ])]);
    let (body, _) = encode(&request);
    assert!(body["input"][0].get("status").is_none());

    let mut quirks = full_quirks();
    quirks.function_call_status_null = true;
    let (body, warnings) = encode_with(&request, &ResponsesWireOptions::default(), &quirks);
    assert!(warnings.is_empty());
    assert_eq!(
        body["input"][0],
        json!({"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}", "status": null})
    );
}

#[test]
fn function_calls_are_grouped_after_the_turn_without_interleaving() {
    let request = ChatRequest::new("m").extend_messages([
        Message::user("Weather in Paris and Rome?"),
        Message::assistant_blocks(vec![
            ContentBlock::text("Checking Paris."),
            ContentBlock::tool_call("c1", "get_weather", r#"{"city":"Paris"}"#),
            ContentBlock::text("And Rome."),
            ContentBlock::tool_call("c2", "get_weather", r#"{"city":"Rome"}"#),
        ]),
        Message::tool_result("c1", "get_weather", "Sunny"),
        Message::tool_result("c2", "get_weather", "Rainy"),
    ]);
    let types = |body: &Value| {
        body["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i["type"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>()
    };
    let (interleaved, _) = encode(&request);
    assert_eq!(
        types(&interleaved),
        [
            "message",
            "message",
            "function_call",
            "message",
            "function_call",
            "function_call_output",
            "function_call_output"
        ]
    );

    let mut quirks = full_quirks();
    quirks.interleaved_function_calls = false;
    let (grouped, warnings) = encode_with(&request, &ResponsesWireOptions::default(), &quirks);
    assert!(warnings.is_empty());
    assert_eq!(
        types(&grouped),
        [
            "message",
            "message",
            "function_call",
            "function_call",
            "function_call_output",
            "function_call_output"
        ]
    );
    // The turn's text stays in order in one message, then the calls in order.
    assert_eq!(grouped["input"][1]["content"][0]["text"], "Checking Paris.");
    assert_eq!(grouped["input"][1]["content"][1]["text"], "And Rome.");
    assert_eq!(grouped["input"][2]["call_id"], "c1");
    assert_eq!(grouped["input"][3]["call_id"], "c2");
}

#[test]
fn json_schema_fails_without_support() {
    let mut quirks = full_quirks();
    quirks.supports_json_schema = false;
    let request = ChatRequest::new("m")
        .user("hi")
        .response_format(ResponseFormat::json_schema(json!({"type": "object"})).non_strict());
    let err = encode_err_with(&request, &quirks);
    assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
    assert!(err.to_string().contains("response_format"), "{err}");
    // JSON mode is unaffected.
    let json_mode = ChatRequest::new("m")
        .user("hi")
        .response_format(ResponseFormat::Json);
    let (body, _) = encode_with(&json_mode, &ResponsesWireOptions::default(), &quirks);
    assert_eq!(body["text"]["format"]["type"], "json_object");
}

#[test]
fn developer_prompts_fall_back_to_system() {
    let quirks = ResponsesQuirks::new(TEST_NAMESPACE);
    assert!(!quirks.supports_developer_role);
    let request = ChatRequest::new("m")
        .system(SystemPrompt::new("Never lie.").with_option(SystemOptions::developer()))
        .user("hi");
    let (body, warnings) = encode_with(&request, &ResponsesWireOptions::default(), &quirks);
    assert_eq!(body["input"][0]["role"], "system");
    assert_eq!(kinds(&warnings), vec![WarningKind::Unsupported]);
    assert_eq!(warnings[0].subject, Subject::System { index: 0 });
    assert!(matches!(
        encode_err_with(
            &request.unsupported_policy(UnsupportedPolicy::Error),
            &quirks
        ),
        Error::Unsupported { .. }
    ));
}

/// `true` is permission: an endpoint without the field drops it silently.
/// `false` cannot be honoured there: a hint.
#[test]
fn parallel_tool_calls_true_is_never_a_degradation() {
    let mut quirks = ResponsesQuirks::new(TEST_NAMESPACE);
    quirks.supports_parallel_tool_calls = false;
    let tools = || {
        ChatRequest::new("m")
            .user("hi")
            .tools([ToolDefinition::new("f", json!({"type": "object"}))])
    };

    let request = tools().parallel_tool_calls(true);
    let (body, warnings) = encode_with(&request, &ResponsesWireOptions::default(), &quirks);
    assert!(body.get("parallel_tool_calls").is_none());
    assert!(warnings.is_empty(), "{warnings:?}");
    let strict = request.unsupported_policy(UnsupportedPolicy::Error);
    assert!(encode_for(&strict, false, &ResponsesWireOptions::default(), &quirks).is_ok());

    let request = tools().parallel_tool_calls(false);
    let (body, warnings) = encode_with(&request, &ResponsesWireOptions::default(), &quirks);
    assert!(body.get("parallel_tool_calls").is_none());
    assert_eq!(kinds(&warnings), vec![WarningKind::Unsupported]);
    assert_eq!(
        warnings[0].message,
        "this endpoint has no `parallel_tool_calls` field; `false` was not sent, so the model may still call several tools at once"
    );
    let err = encode_err_with(
        &request.unsupported_policy(UnsupportedPolicy::Error),
        &quirks,
    );
    assert_eq!(
        err.unsupported_field(),
        Some(RequestField::ParallelToolCalls)
    );
}

#[test]
fn unsupported_tool_settings_and_cache_keys() {
    let mut quirks = ResponsesQuirks::new(TEST_NAMESPACE);
    quirks.supports_parallel_tool_calls = false;
    quirks.supports_tool_choice = false;
    let request = ChatRequest::new("m")
        .user("hi")
        .tools([ToolDefinition::new("f", json!({"type": "object"}))])
        .tool_choice(ToolChoice::Auto)
        .parallel_tool_calls(false)
        .cache(PromptCache::new().key("k"));
    let (body, warnings) = encode_with(&request, &ResponsesWireOptions::default(), &quirks);
    assert_eq!(kinds(&warnings), vec![WarningKind::Unsupported; 2]);
    assert_eq!(
        warnings[0].subject,
        Subject::Field {
            field: RequestField::ParallelToolCalls
        }
    );
    assert_eq!(
        warnings[1].subject,
        Subject::Field {
            field: RequestField::Cache
        }
    );
    for field in ["tool_choice", "parallel_tool_calls", "prompt_cache_key"] {
        assert!(body.get(field).is_none(), "{field} was sent");
    }

    // Forcing or disabling tools is semantic.
    for choice in [
        ToolChoice::Required,
        ToolChoice::Disabled,
        ToolChoice::Specific { name: "f".into() },
    ] {
        let forced = ChatRequest::new("m").user("hi").tool_choice(choice);
        assert!(matches!(
            encode_err_with(&forced, &quirks),
            Error::Unsupported { .. }
        ));
    }
}

#[test]
fn caller_warnings_come_first() {
    let warning = Warning::new(
        WarningKind::Dropped,
        Subject::Option {
            namespace: "openai-compat".into(),
        },
        "ignored",
    );
    let (_, warnings) = encode_options(
        &ChatRequest::new("m").user("hi").seed(1),
        ResponsesWireOptions {
            warnings: vec![warning],
            ..Default::default()
        },
    );
    assert_eq!(
        kinds(&warnings),
        vec![WarningKind::Dropped, WarningKind::Unsupported]
    );
    assert_eq!(warnings[0].provider.as_deref(), Some("test"));
}

#[test]
fn empty_turns_are_skipped_without_a_warning() {
    let baseline = ChatRequest::new("gpt-4.1")
        .user("one")
        .assistant("answer")
        .user("two");
    let with_empty = ChatRequest::new("gpt-4.1")
        .user("one")
        .message(Message::user(""))
        .message(Message::assistant_blocks(vec![]))
        .assistant("answer")
        .message(Message::user_parts(vec![]))
        .message(Message::user_parts(vec![ContentPart::text("")]))
        .user("two");
    let (expected, _) = encode(&baseline);
    let (body, warnings) = encode(&with_empty);
    assert_eq!(body, expected);
    assert!(warnings.is_empty(), "{warnings:?}");
}

#[test]
fn a_reply_left_last_by_skipped_turns_is_dropped_under_every_policy() {
    for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
        let mut request = ChatRequest::new("gpt-4.1")
            .user("one")
            .assistant("answer")
            .message(Message::user(""));
        request.unsupported_policy = Some(policy);
        let (body, warnings) = encode(&request);
        let (expected, _) = encode(&ChatRequest::new("gpt-4.1").user("one"));
        assert_eq!(body, expected, "{policy:?}");
        assert_eq!(kinds(&warnings), [WarningKind::Dropped]);
        assert_eq!(
            warnings[0].subject,
            Subject::Content {
                message: 1,
                block: None
            }
        );
    }
}

#[test]
fn a_conversation_with_nothing_to_send_is_refused() {
    let err = encode_err(&ChatRequest::new("gpt-4.1").message(Message::user("")));
    assert!(matches!(err, Error::InvalidRequest { .. }), "{err}");
}

/// Replay state stays in a stored conversation, so a conversation that
/// moved here from another provider drops it on every turn: each drop is
/// coded, so its warning event goes out at `WARN` once per model, not once
/// per turn, and points to `retain_provider_state`.
#[test]
fn a_long_foreign_reasoning_history_drops_with_codes() {
    let mut request = ChatRequest::new("gpt-5-mini");
    for turn in 0..10 {
        request = request
            .user(format!("question {turn}"))
            .message(Message::assistant_blocks(vec![
                ContentBlock::Reasoning(Reasoning::new("thoughts").with_provider_data(
                    ProviderData::new().with_raw("anthropic", "signature", json!("opaque")),
                )),
                ContentBlock::text("answer"),
            ]));
    }
    let (_, warnings) = encode_with(
        &request.user("next"),
        &ResponsesWireOptions::default(),
        &full_quirks(),
    );
    // One warning per code per request, listing the ten messages.
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(matches!(&warnings[0].subject, Subject::Messages { messages } if messages.len() == 10));
    for warning in &warnings {
        assert_eq!(warning.code.as_deref(), Some("dropped.foreign_reasoning"));
        assert!(
            warning.message.contains("retain_provider_state"),
            "{}",
            warning.message
        );
    }
}

#[test]
fn raw_tool_choice_forces_a_hosted_tool() {
    let options = ResponsesWireOptions {
        tool_choice: Some(json!({"type": "web_search"})),
        hosted_tools: vec![json!({"type": "web_search"}).as_object().unwrap().clone()],
        ..Default::default()
    };
    let encoded = encode_for(
        &ChatRequest::new("m").user("news?"),
        false,
        &options,
        &full_quirks(),
    )
    .unwrap();
    let body = serde_json::to_value(&encoded.body).unwrap();
    assert_eq!(body["tool_choice"], json!({"type": "web_search"}));
    assert!(encoded.warnings.is_empty(), "{:?}", encoded.warnings);

    // It overrides a portable choice, with a warning.
    let request = ChatRequest::new("m")
        .user("news?")
        .tools([ToolDefinition::new("f", json!({"type": "object"}))])
        .tool_choice(ToolChoice::Required);
    let encoded = encode_for(&request, false, &options, &full_quirks()).unwrap();
    let body = serde_json::to_value(&encoded.body).unwrap();
    assert_eq!(body["tool_choice"], json!({"type": "web_search"}));
    assert_eq!(
        encoded.warnings.iter().map(|w| w.kind).collect::<Vec<_>>(),
        [WarningKind::Approximated]
    );
}

#[test]
fn extra_cannot_set_owned_fields_in_any_case() {
    for key in RESERVED_EXTRA_FIELDS
        .iter()
        .copied()
        .chain(["Tool_Choice", "MODEL"])
    {
        let options = ResponsesWireOptions {
            extra: serde_json::Map::from_iter([(key.into(), json!(1))]),
            ..Default::default()
        };
        let err = encode_for(
            &ChatRequest::new("m").user("hi"),
            false,
            &options,
            &full_quirks(),
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::InvalidRequest { .. }),
            "{key}: {err:?}"
        );
    }
}

#[test]
fn every_body_field_is_reserved() {
    // A field added to the body must be reserved too, or `extra` could send
    // it twice.
    let options = ResponsesWireOptions {
        service_tier: Some("flex".into()),
        metadata: Some(serde_json::Map::new()),
        safety_identifier: Some("s".into()),
        user: Some("u".into()),
        truncation: Some("auto".into()),
        verbosity: Some("low".into()),
        store: Some(true),
        previous_response_id: Some("resp_1".into()),
        background: Some(true),
        ..Default::default()
    };
    let request = ChatRequest::new("gpt-4.1")
        .system("s")
        .user("hi")
        .temperature(0.5)
        .top_p(0.9)
        .frequency_penalty(0.1)
        .presence_penalty(0.1)
        .max_output_tokens(10)
        .parallel_tool_calls(true)
        .tools([ToolDefinition::new("f", json!({"type": "object"}))])
        .tool_choice(ToolChoice::Auto)
        .cache(PromptCache::new().key("k").retention(CacheRetention::Long));
    let encoded = encode_for(&request, false, &options, &full_quirks()).unwrap();
    let body = serde_json::to_value(&encoded.body).unwrap();
    for key in body.as_object().unwrap().keys() {
        assert!(
            RESERVED_EXTRA_FIELDS.contains(&key.as_str()),
            "{key} is not reserved"
        );
    }
}

/// A `call_id` over OpenAI's 64 characters is mapped, the same way for the
/// `function_call` and its `function_call_output`; shorter ones are kept.
#[test]
fn call_ids_over_the_limit_are_mapped_consistently() {
    let long = format!("vendor:{}:call:0", "r".repeat(60));
    let request = ChatRequest::new("m").extend_messages([
        Message::user("q"),
        Message::assistant_blocks(vec![
            ContentBlock::tool_call(long.clone(), "f", "{}"),
            ContentBlock::tool_call("call_1", "f", "{}"),
        ]),
        Message::tool_result("call_1", "f", "b"),
        Message::tool_result(long.clone(), "f", "a"),
    ]);
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty(), "{warnings:?}");
    let ids: Vec<(&str, &str)> = body["input"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|item| Some((item["type"].as_str()?, item["call_id"].as_str()?)))
        .collect();
    let mapped = ids[0].1;
    assert_eq!(mapped.chars().count(), 64);
    assert_ne!(mapped, long);
    assert_eq!(
        ids,
        [
            ("function_call", mapped),
            ("function_call", "call_1"),
            ("function_call_output", "call_1"),
            ("function_call_output", mapped),
        ]
    );
}

/// OpenAI's Responses API rejects a non-object root in a `json_schema`
/// text format or a function's parameters, strict or not (HTTP 400,
/// verified live), so the OpenAI dialect refuses it before sending.
#[test]
fn a_non_object_root_is_refused_where_the_endpoint_requires_one() {
    let label = json!({"type": "string", "enum": ["billing", "bug"]});
    let options = ResponsesWireOptions::default();
    let quirks = full_quirks();
    for format in [
        ResponseFormat::json_schema(label.clone()),
        ResponseFormat::json_schema(label.clone()).non_strict(),
    ] {
        let request = ChatRequest::new("gpt-4.1")
            .user("x")
            .response_format(format);
        let err = encode_for(&request, false, &options, &quirks).unwrap_err();
        assert_eq!(
            err.unsupported_field(),
            Some(RequestField::ResponseFormat),
            "{err:?}"
        );
    }
    let tool = ChatRequest::new("gpt-4.1")
        .user("x")
        .tools([ToolDefinition::new("label", label.clone())]);
    let err = encode_for(&tool, false, &options, &quirks).unwrap_err();
    assert_eq!(
        err.unsupported_field(),
        Some(RequestField::Tools),
        "{err:?}"
    );

    let mut generic = ResponsesQuirks::new(TEST_NAMESPACE);
    generic.supports_strict = false;
    let (body, _) = encode_with(&tool, &options, &generic);
    assert_eq!(body["tools"][0]["parameters"], label);
}
