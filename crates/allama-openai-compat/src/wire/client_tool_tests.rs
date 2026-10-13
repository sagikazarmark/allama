//! Client tool content on Chat Completions (ADR 0014): custom tool calls
//! round-trip in the shape the Responses API uses, their answers are lifted
//! into `tool` messages, and the Responses-only items are dropped in pairs.

use allama::{
    ContentBlock, ContentPart, Message, OtherContent, ReplayState, UnsupportedPolicy, WarningKind,
};
use serde_json::{Value, json};

use super::*;

fn other(type_name: &str, data: Value) -> OtherContent {
    OtherContent::new(type_name, data.as_object().cloned().unwrap()).with_origin("openai")
}

fn custom_call(call_id: &str) -> ContentBlock {
    ContentBlock::Other(other(
        "custom_tool_call",
        json!({"call_id": call_id, "name": "sql", "input": "SELECT 1"}),
    ))
}

fn custom_answer(call_id: &str, output: Value) -> ContentPart {
    ContentPart::Other(other(
        "custom_tool_call_output",
        json!({"call_id": call_id, "output": output}),
    ))
}

fn encode(request: &ChatRequest) -> (Value, Vec<Warning>) {
    encode_with(request, &WireOptions::default())
}

fn encode_with(request: &ChatRequest, options: &WireOptions) -> (Value, Vec<Warning>) {
    let (body, warnings) =
        to_chat_completion_request(request, false, options, &CompatQuirks::openai(), "openai")
            .unwrap();
    (serde_json::to_value(body).unwrap(), warnings)
}

fn subjects(warnings: &[Warning]) -> Vec<(WarningKind, Subject)> {
    warnings
        .iter()
        .map(|w| (w.kind, w.subject.clone()))
        .collect()
}

fn part(message: usize, block: usize) -> Subject {
    Subject::Content {
        message,
        block: Some(block),
    }
}

#[test]
fn a_custom_call_is_replayed_and_its_answer_lifted_into_a_tool_message() {
    let request = ChatRequest::new("gpt-5")
        .user("Count the users.")
        .message(Message::assistant_blocks(vec![
            ContentBlock::text("Querying."),
            custom_call("call_1"),
        ]))
        .message(Message::user_parts(vec![custom_answer(
            "call_1",
            json!("42"),
        )]));
    for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
        let mut request = request.clone();
        request.unsupported_policy = Some(policy);
        let (body, warnings) = encode(&request);
        assert_eq!(
            body["messages"],
            json!([
                {"role": "user", "content": "Count the users."},
                {"role": "assistant", "content": "Querying.", "tool_calls": [
                    {"id": "call_1", "type": "custom", "custom": {"name": "sql", "input": "SELECT 1"}}
                ]},
                // The user message emptied by lifting is skipped.
                {"role": "tool", "content": "42", "tool_call_id": "call_1"}
            ]),
            "{policy:?}"
        );
        assert!(warnings.is_empty(), "{policy:?}: {warnings:?}");
    }
}

#[test]
fn lifted_answers_follow_the_tool_messages_of_the_turn() {
    let request = ChatRequest::new("gpt-5")
        .user("Look up rust, then count the users.")
        .message(Message::assistant_blocks(vec![
            ContentBlock::tool_call("c1", "lookup", r#"{"q":"rust"}"#),
            custom_call("call_2"),
        ]))
        .message(Message::tool_result("c1", "lookup", "Rust is a language."))
        .message(Message::user_parts(vec![
            custom_answer("call_2", json!("42")),
            ContentPart::text("Now summarize."),
        ]));
    let (body, warnings) = encode(&request);
    assert_eq!(
        body["messages"],
        json!([
            {"role": "user", "content": "Look up rust, then count the users."},
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": "{\"q\":\"rust\"}"}},
                {"id": "call_2", "type": "custom", "custom": {"name": "sql", "input": "SELECT 1"}}
            ]},
            {"role": "tool", "content": "Rust is a language.", "tool_call_id": "c1"},
            {"role": "tool", "content": "42", "tool_call_id": "call_2"},
            {"role": "user", "content": [{"type": "text", "text": "Now summarize."}]}
        ])
    );
    assert!(warnings.is_empty(), "{warnings:?}");
}

#[test]
fn an_answer_that_cannot_be_lifted_in_place_is_dropped_never_reordered() {
    // Behind other content of its message.
    let request = ChatRequest::new("gpt-5")
        .user("Count the users.")
        .message(Message::assistant_blocks(vec![custom_call("call_1")]))
        .message(Message::user_parts(vec![
            ContentPart::text("Here you go:"),
            custom_answer("call_1", json!("42")),
        ]));
    let (body, warnings) = encode(&request);
    assert_eq!(body["messages"].as_array().unwrap().len(), 3);
    assert_eq!(
        body["messages"][2],
        json!({"role": "user", "content": [{"type": "text", "text": "Here you go:"}]})
    );
    assert_eq!(subjects(&warnings), [(WarningKind::Dropped, part(2, 1))]);

    // In a later user message than the one right after the turn.
    let request = ChatRequest::new("gpt-5")
        .user("Count the users.")
        .message(Message::assistant_blocks(vec![custom_call("call_1")]))
        .user("Hold on.")
        .message(Message::user_parts(vec![custom_answer(
            "call_1",
            json!("42"),
        )]));
    let (body, warnings) = encode(&request);
    let roles: Vec<&str> = body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, ["user", "assistant", "user"]);
    assert_eq!(subjects(&warnings), [(WarningKind::Dropped, part(3, 0))]);
}

/// Chat Completions has no server-side continuation: an answer whose call
/// is not in the request is dropped, never sent as message content.
#[test]
fn an_answer_without_its_call_is_dropped() {
    for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
        let mut request = ChatRequest::new("gpt-5").message(Message::user_parts(vec![
            custom_answer("call_9", json!("42")),
            ContentPart::text("And now?"),
        ]));
        request.unsupported_policy = Some(policy);
        let (body, warnings) = encode(&request);
        assert_eq!(
            body["messages"],
            json!([{"role": "user", "content": [{"type": "text", "text": "And now?"}]}]),
            "{policy:?}"
        );
        assert_eq!(subjects(&warnings), [(WarningKind::Dropped, part(0, 0))]);
    }
}

#[test]
fn a_list_output_becomes_text_and_its_images_are_dropped() {
    let request = ChatRequest::new("gpt-5")
        .user("Render the chart.")
        .message(Message::assistant_blocks(vec![custom_call("call_1")]))
        .message(Message::user_parts(vec![custom_answer(
            "call_1",
            json!([
                {"type": "input_text", "text": "Rendered "},
                {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                {"type": "input_text", "text": "the chart."}
            ]),
        )]));
    let (body, warnings) = encode(&request);
    assert_eq!(
        body["messages"][2],
        json!({"role": "tool", "content": "Rendered the chart.", "tool_call_id": "call_1"})
    );
    // The warning names the caller's message and part.
    assert_eq!(subjects(&warnings), [(WarningKind::Dropped, part(2, 0))]);
    assert!(warnings[0].message.contains("input_image"), "{warnings:?}");
}

/// A Responses-recorded local shell call and its output: Chat Completions
/// cannot express the call, so the pair is dropped.
#[test]
fn responses_only_calls_and_their_outputs_are_dropped_as_a_pair() {
    let shell_call = ContentBlock::Other(other(
        "local_shell_call",
        json!({"id": "lsh_1", "call_id": "call_3", "status": "completed",
               "action": {"type": "exec", "command": ["ls"]}}),
    ));
    let shell_output = ContentPart::Other(other(
        "local_shell_call_output",
        json!({"id": "call_3", "output": "a.txt"}),
    ));
    for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
        let mut request = ChatRequest::new("gpt-5")
            .user("List the files.")
            .message(Message::assistant_blocks(vec![
                ContentBlock::text("Listing."),
                shell_call.clone(),
            ]))
            .message(Message::user_parts(vec![shell_output.clone()]))
            .user("What did you find?");
        request.unsupported_policy = Some(policy);
        let (body, warnings) = encode(&request);
        assert_eq!(
            body["messages"],
            json!([
                {"role": "user", "content": "List the files."},
                {"role": "assistant", "content": "Listing."},
                {"role": "user", "content": "What did you find?"}
            ]),
            "{policy:?}"
        );
        assert_eq!(
            subjects(&warnings),
            [
                (WarningKind::Dropped, part(1, 1)),
                (WarningKind::Dropped, part(2, 0))
            ],
            "{policy:?}"
        );
        assert!(
            warnings[1].message.contains("answers cannot be replayed"),
            "{warnings:?}"
        );
        // Content of this namespace the endpoint cannot take back is
        // replay state that recurs with the history.
        assert_eq!(
            warnings[0].message,
            "Chat Completions cannot replay `local_shell_call` content; it was dropped"
        );
        assert_eq!(warnings[0].code.as_deref(), Some("dropped.content_replay"));
        assert_eq!(warnings[0].replay_state(), Some(ReplayState::Content));
        assert_eq!(warnings[0].provider.as_deref(), Some("openai"));
    }
}

#[test]
fn own_parts_are_sent_only_for_chat_content_types() {
    let audio = other(
        "input_audio",
        json!({"input_audio": {"data": "AAAA", "format": "wav"}}),
    );
    let request = ChatRequest::new("gpt-5").message(Message::user_parts(vec![
        ContentPart::text("Listen."),
        ContentPart::Other(audio),
        // Responses-only input items and unknown types.
        ContentPart::Other(other("computer_call_output", json!({"call_id": "c"}))),
        ContentPart::Other(other("mcp_approval_response", json!({"approve": true}))),
        ContentPart::Other(other("program_output", json!({"output": "x"}))),
        // Another provider's part.
        ContentPart::Other(
            OtherContent::new("search_result", serde_json::Map::new()).with_origin("anthropic"),
        ),
        // The caller's raw part, with no origin, is sent verbatim.
        ContentPart::Other(OtherContent::new(
            "video_url",
            json!({"video_url": {"url": "https://example.com/v.mp4"}})
                .as_object()
                .cloned()
                .unwrap(),
        )),
    ]));
    let (body, warnings) = encode(&request);
    assert_eq!(
        body["messages"][0]["content"],
        json!([
            {"type": "text", "text": "Listen."},
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
            {"type": "video_url", "video_url": {"url": "https://example.com/v.mp4"}}
        ])
    );
    assert_eq!(
        subjects(&warnings),
        [
            (WarningKind::Dropped, part(0, 2)),
            (WarningKind::Dropped, part(0, 3)),
            (WarningKind::Dropped, part(0, 4)),
            (WarningKind::Dropped, part(0, 5)),
        ]
    );
}

#[test]
fn foreign_tool_result_parts_are_dropped() {
    let request = ChatRequest::new("gpt-5")
        .user("Look up rust.")
        .message(Message::assistant_blocks(vec![ContentBlock::tool_call(
            "c1", "lookup", "{}",
        )]))
        .message(Message::tool_result(
            "c1",
            "lookup",
            vec![
                ContentPart::text("Rust is a language."),
                ContentPart::Other(
                    OtherContent::new("search_result", serde_json::Map::new())
                        .with_origin("anthropic"),
                ),
            ],
        ));
    let (body, warnings) = encode(&request);
    assert_eq!(body["messages"][2]["content"], "Rust is a language.");
    assert_eq!(subjects(&warnings), [(WarningKind::Dropped, part(2, 1))]);

    // A raw part is sent with the text, as a list of parts.
    let request = ChatRequest::new("gpt-5")
        .user("Look up rust.")
        .message(Message::assistant_blocks(vec![ContentBlock::tool_call(
            "c1", "lookup", "{}",
        )]))
        .message(Message::tool_result(
            "c1",
            "lookup",
            vec![
                ContentPart::text("Rust is a language."),
                ContentPart::Other(OtherContent::new(
                    "text",
                    json!({"text": " Fast."}).as_object().cloned().unwrap(),
                )),
            ],
        ));
    let (body, _) = encode(&request);
    assert_eq!(
        body["messages"][2]["content"],
        json!([
            {"type": "text", "text": "Rust is a language."},
            {"type": "text", "text": " Fast."}
        ])
    );
}

#[test]
fn client_tool_definitions_follow_the_function_tools() {
    let sql = json!({"type": "custom", "custom": {
        "name": "sql",
        "description": "Run a read-only SQLite query.",
        "format": {"type": "grammar", "grammar": {"syntax": "regex", "definition": "SELECT .+"}}
    }});
    let options = WireOptions {
        client_tools: vec![sql.as_object().cloned().unwrap()],
        ..WireOptions::default()
    };
    let request =
        ChatRequest::new("gpt-5")
            .user("Count the users.")
            .tools([allama::ToolDefinition::new(
                "lookup",
                json!({"type": "object", "properties": {}}),
            )]);
    let (body, warnings) = encode_with(&request, &options);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(body["tools"][0]["type"], "function");
    assert_eq!(body["tools"][1], sql);

    // Definitions alone are tools too.
    let (body, _) = encode_with(&ChatRequest::new("gpt-5").user("Hi."), &options);
    assert_eq!(body["tools"].as_array().unwrap().len(), 1);

    // A definition without a type is refused.
    let options = WireOptions {
        client_tools: vec![
            json!({"custom": {"name": "sql"}})
                .as_object()
                .cloned()
                .unwrap(),
        ],
        ..WireOptions::default()
    };
    let err = to_chat_completion_request(
        &ChatRequest::new("gpt-5").user("Hi."),
        false,
        &options,
        &CompatQuirks::openai(),
        "openai",
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidRequest { .. }), "{err}");
}

fn context() -> ResponseContext {
    ResponseContext::new("openai", "openai")
}

/// A turn that called only a custom tool, as the SDK types describe it
/// (`ChatCompletionMessageCustomToolCall`).
fn custom_only_response(finish_reason: &str) -> ChatCompletionResponse {
    serde_json::from_value(json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1_790_906_495,
        "model": "gpt-5-2025-08-07",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "refusal": null,
                "tool_calls": [{
                    "id": "call_Rk3XbQ1mZc7Vn2Pq8Ld5Ws0Y",
                    "type": "custom",
                    "custom": {"name": "sql", "input": "SELECT count(*) FROM users"}
                }]
            },
            "finish_reason": finish_reason
        }],
        "usage": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100}
    }))
    .unwrap()
}

#[test]
fn a_custom_only_turn_decodes_to_the_shared_shape_and_waits_for_the_client() {
    for raw in ["tool_calls", "stop"] {
        let response = from_api_response(custom_only_response(raw), context()).unwrap();
        assert_eq!(
            response.finish_reason,
            Some(FinishReason::ToolCalls),
            "{raw}"
        );
        assert!(
            response.tool_calls().next().is_none(),
            "not a function call"
        );
        let pending = crate::ClientToolCall::pending(&response.content);
        let [crate::ClientToolCall::Custom(call)] = pending.as_slice() else {
            panic!("expected one custom call, got {pending:?}")
        };
        assert_eq!(call.call_id, "call_Rk3XbQ1mZc7Vn2Pq8Ld5Ws0Y");
        assert_eq!(call.name, "sql");
        assert_eq!(call.input, "SELECT count(*) FROM users");
        assert_eq!(call.origin.as_deref(), Some("openai"));
        let ContentBlock::Other(other) = &response.content[0] else {
            panic!("expected an Other block")
        };
        assert_eq!(other.type_name, "custom_tool_call");
        assert_eq!(
            Value::Object(other.data.clone()),
            json!({"call_id": "call_Rk3XbQ1mZc7Vn2Pq8Ld5Ws0Y", "name": "sql",
                   "input": "SELECT count(*) FROM users"})
        );
    }
}

#[test]
fn a_decoded_custom_call_round_trips() {
    let response = from_api_response(custom_only_response("tool_calls"), context()).unwrap();
    let crate::ClientToolCall::Custom(call) =
        crate::ClientToolCall::pending(&response.content).remove(0)
    else {
        panic!("expected a custom call")
    };
    let mut request = ChatRequest::new("gpt-5").user("How many users?");
    request.push_message(response.to_assistant_message());
    request.push_message(Message::user(vec![call.output("42")]));
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["messages"][1]["tool_calls"],
        json!([{"id": "call_Rk3XbQ1mZc7Vn2Pq8Ld5Ws0Y", "type": "custom",
                "custom": {"name": "sql", "input": "SELECT count(*) FROM users"}}])
    );
    assert_eq!(
        body["messages"][2],
        json!({"role": "tool", "content": "42", "tool_call_id": "call_Rk3XbQ1mZc7Vn2Pq8Ld5Ws0Y"})
    );
}

#[test]
fn a_tool_call_with_neither_function_nor_custom_is_an_unexpected_response() {
    let response: ChatCompletionResponse = serde_json::from_value(json!({
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "tool_calls": [{"id": "call_1", "type": "mystery"}]
        }}]
    }))
    .unwrap();
    let err = from_api_response(response, context()).unwrap_err();
    assert!(matches!(err, Error::UnexpectedResponse { .. }), "{err}");
}
