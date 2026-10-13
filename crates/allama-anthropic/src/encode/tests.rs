use super::*;
use allama::{
    AssistantMessage, CacheBreakpoint, CacheRetention, ContentBlock, ContentPart, File, Image,
    ImageDetail, Message, OtherContent, PromptCache, ProviderData, Reasoning, ReasoningConfig,
    ReasoningEffort, ReasoningOutput, SystemPrompt, Text, ToolCall, ToolDefinition, ToolMessage,
    UnsupportedPolicy, WarningKind,
};
use serde_json::{Value, json};

const HAIKU: &str = "claude-haiku-4-5-20251001";
const SONNET_55: &str = "claude-sonnet-5-5";

fn encode(request: &ChatRequest) -> (Value, Vec<Warning>) {
    let encoded = encode_request(request, false).expect("request should encode");
    (
        serde_json::to_value(&encoded.body).unwrap(),
        encoded.warnings,
    )
}

fn encode_err(request: &ChatRequest) -> Error {
    encode_request(request, false).expect_err("request should fail to encode")
}

/// Encode with the caller's capability `answers`, as a provider with a
/// capability resolver does.
fn encode_answering(
    request: &ChatRequest,
    answers: &[(allama::ChatCapability, CapabilitySupport)],
) -> Result<EncodedRequest> {
    let profile = ModelProfile::for_model(&request.model).with_overrides(|capability| {
        answers
            .iter()
            .find(|(asked, _)| *asked == capability)
            .map(|(_, answer)| *answer)
    });
    encode_request_for(request, false, profile)
}

fn kinds(warnings: &[Warning]) -> Vec<WarningKind> {
    warnings.iter().map(|w| w.kind).collect()
}

fn weather_tool() -> ToolDefinition {
    ToolDefinition::new(
        "get_weather",
        json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
    )
    .description("Get the weather")
}

fn signed_reasoning(text: &str, signature: &str) -> ContentBlock {
    ContentBlock::Reasoning(
        Reasoning::new(text).with_provider_data(ProviderData::new().with_raw(
            "anthropic",
            "signature",
            json!(signature),
        )),
    )
}

// --- Messages ---

#[test]
fn minimal_request_shape() {
    let (body, warnings) = encode(&ChatRequest::new(HAIKU).user("Hello!"));
    assert_eq!(
        body,
        json!({
            "model": HAIKU,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}],
            "stream": false
        })
    );
    assert!(warnings.is_empty());
}

#[test]
fn system_prompts_are_hoisted_in_order_and_empty_ones_skipped() {
    let (body, _) = encode(
        &ChatRequest::new(HAIKU)
            .system("First")
            .system("")
            .system("Second")
            .user("hi"),
    );
    assert_eq!(
        body["system"],
        json!([{"type": "text", "text": "First"}, {"type": "text", "text": "Second"}])
    );
}

#[test]
fn anthropic_options_on_system_prompts_are_reported() {
    let mut prompt = SystemPrompt::new("Be brief");
    prompt.options.insert_json("anthropic", json!({"x": 1}));
    let (_, warnings) = encode(&ChatRequest::new(HAIKU).system(prompt).user("hi"));
    assert_eq!(kinds(&warnings), vec![WarningKind::Dropped]);
    assert_eq!(warnings[0].subject, Subject::System { index: 0 });
}

#[test]
fn tool_results_are_grouped_into_one_user_message() {
    let request = ChatRequest::new(HAIKU)
        .user("Weather in Paris and Tokyo?")
        .message(Message::assistant_blocks(vec![
            ContentBlock::tool_call("t1", "get_weather", r#"{"city":"Paris"}"#),
            ContentBlock::tool_call("t2", "get_weather", r#"{"city":"Tokyo"}"#),
        ]))
        .message(Message::tool_result("t1", "get_weather", "Sunny"))
        .message(Message::tool_error("t2", "get_weather", "unavailable"))
        .user("Thanks");
    let (body, _) = encode(&request);
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(
        messages[1]["content"][0],
        json!({"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Paris"}})
    );
    assert_eq!(messages[2]["role"], "user");
    assert_eq!(
        messages[2]["content"],
        json!([
            {"type": "tool_result", "tool_use_id": "t1", "content": "Sunny"},
            {"type": "tool_result", "tool_use_id": "t2", "content": "unavailable", "is_error": true},
            {"type": "text", "text": "Thanks"}
        ])
    );
}

/// Anthropic refuses a `tool_use.id` outside `^[a-zA-Z0-9_-]+$` (verified
/// live: `gemini:abc:call:0` got a 400). Foreign ids that break it are
/// mapped, the same way for the call and its result; valid ones are kept.
#[test]
fn foreign_tool_call_ids_are_mapped_to_the_accepted_alphabet() {
    let request = ChatRequest::new(HAIKU)
        .user("Weather in Paris and Tokyo?")
        .message(Message::assistant_blocks(vec![
            ContentBlock::tool_call("vendor:resp:call:0", "get_weather", r#"{"city":"Paris"}"#),
            ContentBlock::tool_call("vendor:resp:call:1", "get_weather", r#"{"city":"Tokyo"}"#),
            ContentBlock::tool_call("call_ok-1", "get_weather", r#"{"city":"Rome"}"#),
        ]))
        .message(Message::tool_result(
            "vendor:resp:call:1",
            "get_weather",
            "Rain",
        ))
        .message(Message::tool_result(
            "vendor:resp:call:0",
            "get_weather",
            "Sunny",
        ))
        .message(Message::tool_result("call_ok-1", "get_weather", "Hot"));
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty(), "{warnings:?}");
    let calls: Vec<&str> = body["messages"][1]["content"]
        .as_array()
        .unwrap()
        .iter()
        .map(|block| block["id"].as_str().unwrap())
        .collect();
    let results: Vec<&str> = body["messages"][2]["content"]
        .as_array()
        .unwrap()
        .iter()
        .map(|block| block["tool_use_id"].as_str().unwrap())
        .collect();
    for id in calls.iter().chain(&results) {
        assert!(
            !id.is_empty()
                && id
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
            "{id}"
        );
    }
    assert_ne!(calls[0], calls[1]);
    assert_eq!(calls[2], "call_ok-1");
    assert_eq!(results, [calls[1], calls[0], calls[2]]);
}

#[test]
fn a_validated_tool_run_followed_by_user_text_keeps_results_first() {
    // `history::validate` accepts a user message right after the run of tool
    // results, with the results in any order. Anthropic needs every
    // `tool_result` in the next user turn, before any text: the merge of the
    // run and the user message must keep that order.
    let request = ChatRequest::new(HAIKU)
        .user("Weather in Paris and Tokyo?")
        .message(Message::assistant_blocks(vec![
            ContentBlock::text("Checking both."),
            ContentBlock::tool_call("t1", "get_weather", r#"{"city":"Paris"}"#),
            ContentBlock::tool_call("t2", "get_weather", r#"{"city":"Tokyo"}"#),
        ]))
        .message(Message::tool_result("t2", "get_weather", "Rain"))
        .message(Message::tool_result("t1", "get_weather", "Sunny"))
        .user(vec![
            ContentPart::text("Thanks."),
            ContentPart::text("And Rome?"),
        ]);
    assert_eq!(
        allama::history::validate(&request.messages, allama::history::ValidationMode::Sendable),
        Ok(())
    );

    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty(), "{warnings:?}");
    let messages = body["messages"].as_array().unwrap();
    let roles: Vec<&str> = messages
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, ["user", "assistant", "user"]);
    assert_eq!(
        messages[2]["content"],
        json!([
            {"type": "tool_result", "tool_use_id": "t2", "content": "Rain"},
            {"type": "tool_result", "tool_use_id": "t1", "content": "Sunny"},
            {"type": "text", "text": "Thanks."},
            {"type": "text", "text": "And Rome?"}
        ])
    );
}

#[test]
fn structured_and_multipart_tool_results() {
    let mut object = ExtraMap::new();
    object.insert("temp".into(), json!(18));
    let request = ChatRequest::new(HAIKU)
        .user("go")
        .message(Message::assistant_blocks(vec![
            ContentBlock::tool_call("t1", "a", "{}"),
            ContentBlock::tool_call("t2", "b", "{}"),
        ]))
        .message(Message::tool_result(
            "t1",
            "a",
            ToolResultContent::Object(object),
        ))
        .message(Message::tool_result(
            "t2",
            "b",
            ToolResultContent::Parts(vec![
                ContentPart::text("chart:"),
                ContentPart::image_base64("image/png", "AAAA"),
            ]),
        ));
    let (body, _) = encode(&request);
    let results = &body["messages"][2]["content"];
    assert_eq!(results[0]["content"], json!(r#"{"temp":18}"#));
    assert_eq!(
        results[1]["content"],
        json!([
            {"type": "text", "text": "chart:"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
        ])
    );
}

#[test]
fn consecutive_user_messages_are_merged() {
    let (body, _) = encode(
        &ChatRequest::new(HAIKU)
            .user("one")
            .user("two")
            .assistant("ok"),
    );
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["content"].as_array().unwrap().len(), 2);
}

#[test]
fn images_and_detail_hint() {
    let request = ChatRequest::new(HAIKU).user(vec![
        ContentPart::text("look"),
        ContentPart::Image(Image::url("https://example.com/a.png").with_detail(ImageDetail::High)),
    ]);
    let (body, warnings) = encode(&request);
    assert_eq!(
        body["messages"][0]["content"][1],
        json!({"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}})
    );
    assert_eq!(kinds(&warnings), vec![WarningKind::Unsupported]);
    assert_eq!(
        warnings[0].subject,
        Subject::Content {
            message: 0,
            block: Some(1)
        }
    );

    let strict = request.unsupported_policy(UnsupportedPolicy::Error);
    assert!(matches!(encode_err(&strict), Error::Unsupported { .. }));
}

#[test]
fn file_parts_map_to_documents() {
    let request = ChatRequest::new(HAIKU).user(vec![
        ContentPart::File(File::base64("application/pdf", "JVBERi0=").with_filename("a.pdf")),
        ContentPart::file_url("application/pdf", "https://example.com/a.pdf"),
        // "hello world" in base64
        ContentPart::file_base64("text/plain", "aGVsbG8gd29ybGQ="),
    ]);
    let (body, _) = encode(&request);
    assert_eq!(
        body["messages"][0]["content"],
        json!([
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="}, "title": "a.pdf"},
            {"type": "document", "source": {"type": "url", "url": "https://example.com/a.pdf"}},
            {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "hello world"}}
        ])
    );
}

#[test]
fn unsupported_file_types_fail() {
    let request = ChatRequest::new(HAIKU).user(vec![ContentPart::file_base64("audio/wav", "AAAA")]);
    assert!(matches!(encode_err(&request), Error::Unsupported { .. }));
    let request = ChatRequest::new(HAIKU).user(vec![ContentPart::file_url(
        "text/plain",
        "https://example.com/a.txt",
    )]);
    assert!(matches!(encode_err(&request), Error::Unsupported { .. }));
}

#[test]
fn foreign_parts_are_dropped_and_own_and_raw_parts_pass_through() {
    let mut data = ExtraMap::new();
    data.insert("content".into(), json!("x"));
    let foreign =
        || ContentPart::Other(OtherContent::new("input_audio", data.clone()).with_origin("openai"));
    for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
        // A user part and a tool result part from another provider are
        // dropped with a warning, under every policy.
        let mut request = ChatRequest::new(HAIKU)
            .message(Message::user_parts(vec![
                ContentPart::text("Hi."),
                foreign(),
            ]))
            .message(Message::assistant_blocks(vec![ContentBlock::tool_call(
                "toolu_1", "lookup", "{}",
            )]))
            .message(Message::tool_result(
                "toolu_1",
                "lookup",
                vec![ContentPart::text("found"), foreign()],
            ));
        request.unsupported_policy = Some(policy);
        let (body, warnings) = encode(&request);
        assert_eq!(
            body["messages"][0]["content"],
            json!([{"type": "text", "text": "Hi."}]),
            "{policy:?}"
        );
        assert_eq!(
            body["messages"][2]["content"][0]["content"],
            json!([{"type": "text", "text": "found"}]),
            "{policy:?}"
        );
        // One warning per code per request, listing both messages.
        assert_eq!(kinds(&warnings), vec![WarningKind::Dropped], "{policy:?}");
        assert_eq!(
            warnings[0].subject,
            Subject::Messages {
                messages: vec![0, 2]
            }
        );
    }

    // Anthropic's own part, and the caller's raw part (no origin), are sent
    // verbatim.
    let native = OtherContent::new("search_result", data.clone()).with_origin("anthropic");
    let raw = OtherContent::new("search_result", data);
    let (body, warnings) = encode(
        &ChatRequest::new(HAIKU).user(vec![ContentPart::Other(native), ContentPart::Other(raw)]),
    );
    assert_eq!(
        body["messages"][0]["content"],
        json!([
            {"type": "search_result", "content": "x"},
            {"type": "search_result", "content": "x"}
        ])
    );
    assert!(warnings.is_empty(), "{warnings:?}");
}

/// A refusal another provider wrote (OpenAI's `refusal`) has no Anthropic
/// block: it is replayed as the assistant text it was, without a warning.
#[test]
fn refusals_are_replayed_as_text() {
    let request = ChatRequest::new(HAIKU)
        .user("Help me.")
        .message(Message::assistant_blocks(vec![ContentBlock::Refusal(
            allama::Refusal::new("I can't help with that."),
        )]))
        .user("Why not?");
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["messages"][1],
        json!({"role": "assistant", "content": [{"type": "text", "text": "I can't help with that."}]})
    );
}

#[test]
fn assistant_images_cannot_be_replayed() {
    let request = ChatRequest::new(HAIKU)
        .user("draw")
        .message(Message::assistant_blocks(vec![ContentBlock::Image(
            Image::url("https://example.com/a.png"),
        )]));
    assert!(matches!(encode_err(&request), Error::Unsupported { .. }));
}

// --- Replay of opaque state ---

#[test]
fn thinking_is_replayed_with_signature() {
    let request = ChatRequest::new(HAIKU)
        .user("q")
        .message(Message::assistant_blocks(vec![
            signed_reasoning("Let me think", "sig_1"),
            ContentBlock::text("Answer"),
        ]));
    let (body, warnings) = encode(&request);
    assert_eq!(
        body["messages"][1]["content"],
        json!([
            {"type": "thinking", "thinking": "Let me think", "signature": "sig_1"},
            {"type": "text", "text": "Answer"}
        ])
    );
    assert!(warnings.is_empty());
}

#[test]
fn redacted_thinking_is_replayed() {
    let block = ContentBlock::Reasoning(Reasoning::new("").with_provider_data(
        ProviderData::new().with_raw("anthropic", "redacted_data", json!("opaque")),
    ));
    let request = ChatRequest::new(HAIKU)
        .user("q")
        .message(Message::assistant_blocks(vec![
            block,
            ContentBlock::text("ok"),
        ]));
    let (body, _) = encode(&request);
    assert_eq!(
        body["messages"][1]["content"][0],
        json!({"type": "redacted_thinking", "data": "opaque"})
    );
}

#[test]
fn foreign_and_unsigned_reasoning_is_dropped_with_warning() {
    let foreign = ContentBlock::Reasoning(Reasoning::new("thoughts").with_provider_data(
        ProviderData::new().with_raw("google-ai", "thought_signature", json!("g")),
    ));
    let unsigned = ContentBlock::reasoning("plain thoughts");
    let request = ChatRequest::new(HAIKU)
        .user("q")
        .message(Message::assistant_blocks(vec![
            foreign,
            unsigned,
            ContentBlock::text("answer"),
        ]));
    let (body, warnings) = encode(&request);
    assert_eq!(
        body["messages"][1]["content"],
        json!([{"type": "text", "text": "answer"}])
    );
    assert_eq!(
        kinds(&warnings),
        vec![WarningKind::Dropped, WarningKind::Dropped]
    );
    assert!(warnings[0].message.contains("google-ai"));
    assert_eq!(
        warnings[0].subject,
        Subject::Content {
            message: 1,
            block: Some(0)
        }
    );
}

#[test]
fn assistant_with_only_foreign_content_is_skipped() {
    let foreign = ContentBlock::Reasoning(Reasoning::new("t").with_provider_data(
        ProviderData::new().with_raw("openai", "encrypted_content", json!("e")),
    ));
    let request = ChatRequest::new(HAIKU)
        .user("one")
        .message(Message::assistant_blocks(vec![foreign]))
        .user("two");
    let (body, warnings) = encode(&request);
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1, "user turns merge after the skip");
    // The dropped block is reported; skipping the emptied turn loses nothing more.
    assert_eq!(kinds(&warnings), vec![WarningKind::Dropped]);
    assert_eq!(
        warnings[0].subject,
        Subject::Content {
            message: 1,
            block: Some(0)
        }
    );
}

#[test]
fn empty_turns_are_skipped_and_the_rest_keep_alternating() {
    let baseline = ChatRequest::new(HAIKU)
        .user("one")
        .assistant("answer")
        .user("two");
    let with_empty = ChatRequest::new(HAIKU)
        .user("one")
        .message(Message::user(""))
        .message(Message::assistant_blocks(vec![]))
        .assistant("answer")
        .message(Message::user_parts(vec![]))
        .message(Message::assistant_blocks(vec![]))
        .user("two");
    let (expected, _) = encode(&baseline);
    let (body, warnings) = encode(&with_empty);
    assert_eq!(body, expected);
    assert!(warnings.is_empty(), "{warnings:?}");
    let roles: Vec<&str> = body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, ["user", "assistant", "user"]);
}

#[test]
fn a_cache_breakpoint_on_a_skipped_turn_is_reported() {
    let request = ChatRequest::new(HAIKU)
        .user("one")
        .message(Message::user("").with_cache(CacheBreakpoint::new()))
        .user("two");
    let (body, warnings) = encode(&request);
    assert_eq!(body["messages"].as_array().unwrap().len(), 1);
    assert_eq!(kinds(&warnings), vec![WarningKind::Dropped]);
    assert!(warnings[0].message.contains("cache breakpoint"));
}

#[test]
fn an_empty_tool_result_is_sent() {
    let request = ChatRequest::new(HAIKU)
        .user("Weather?")
        .message(Message::assistant_blocks(vec![ContentBlock::tool_call(
            "c1",
            "get_weather",
            "{}",
        )]))
        .message(Message::tool_result("c1", "get_weather", ""));
    let (body, _) = encode(&request);
    assert_eq!(
        body["messages"][2]["content"][0],
        json!({"type": "tool_result", "tool_use_id": "c1", "content": ""})
    );
}

#[test]
fn a_reply_left_last_by_skipped_turns_is_dropped_under_every_policy() {
    for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
        let mut request = ChatRequest::new(HAIKU)
            .user("one")
            .assistant("answer")
            .message(Message::user(""));
        request.unsupported_policy = Some(policy);
        let (body, warnings) = encode(&request);
        assert_eq!(
            body["messages"],
            json!([{"role": "user", "content": [{"type": "text", "text": "one"}]}]),
            "{policy:?}"
        );
        assert_eq!(kinds(&warnings), vec![WarningKind::Dropped]);
        assert_eq!(
            warnings[0].subject,
            Subject::Content {
                message: 1,
                block: None
            }
        );
    }
}

/// A prefill is sent on every model, with structured output too: whether
/// the model takes it is the API's answer, not the model id's (ADR 0015).
/// Claude 4.6 and later generations refuse it, and so do earlier ones with
/// an output format; `error.rs` maps those refusals.
#[test]
fn a_prefill_is_sent_last() {
    let format = ResponseFormat::json_schema(
        json!({"type": "object", "properties": {"a": {"type": "string"}}}),
    )
    .non_strict();
    for model in [
        HAIKU,
        "claude-sonnet-4-6",
        "claude-opus-5-5",
        "my-proxy-model",
    ] {
        let request = ChatRequest::new(model).user("one").assistant("{");
        for request in [request.clone(), request.response_format(format.clone())] {
            let (body, warnings) = encode(&request);
            assert_eq!(body["messages"][1]["role"], "assistant", "{model}");
            assert_eq!(body["messages"][1]["content"][0]["text"], "{", "{model}");
            assert!(warnings.is_empty(), "{model}: {warnings:?}");
        }
    }
}

#[test]
fn a_conversation_with_nothing_to_send_is_refused() {
    let err = encode_err(&ChatRequest::new(HAIKU).message(Message::user("")));
    assert!(matches!(err, Error::InvalidRequest { .. }), "{err}");
    assert_eq!(err.message_origin(), allama::MessageOrigin::Authored);
}

/// A custom tool call recorded by the OpenAI Responses API and its answer,
/// after `retain_provider_state` moved the conversation to Anthropic: the
/// answer's user message is left empty, and the reply before it would end
/// the conversation.
#[test]
fn a_moved_client_tool_turn_ends_on_the_question_after_retain() {
    let object = |data: Value| data.as_object().cloned().unwrap();
    let mut history = allama::history::History::new(vec![
        Message::user("Run the formatter."),
        Message::assistant_blocks(vec![
            ContentBlock::text("Running it."),
            ContentBlock::Other(
                OtherContent::new(
                    "custom_tool_call",
                    object(json!({"call_id": "call_1", "name": "fmt", "input": "src/"})),
                )
                .with_origin("openai"),
            ),
        ]),
        Message::user_parts(vec![ContentPart::Other(
            OtherContent::new(
                "custom_tool_call_output",
                object(json!({"call_id": "call_1", "output": "ok"})),
            )
            .with_origin("openai"),
        )]),
    ]);
    history.retain_provider_state(&["anthropic"]);
    assert_eq!(history.messages.len(), 3, "no message is removed");

    for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
        let mut request = history.clone().into_request(HAIKU);
        request.unsupported_policy = Some(policy);
        let (body, warnings) = encode(&request);
        assert_eq!(
            body["messages"],
            json!([{"role": "user", "content": [{"type": "text", "text": "Run the formatter."}]}]),
            "{policy:?}"
        );
        assert_eq!(kinds(&warnings), vec![WarningKind::Dropped], "{policy:?}");
        assert_eq!(
            warnings[0].subject,
            Subject::Content {
                message: 1,
                block: None
            }
        );
    }
}

/// An OpenAI custom tool call, as both OpenAI APIs decode it.
fn openai_custom_call(call_id: &str) -> ContentBlock {
    let data = json!({"call_id": call_id, "name": "fmt", "input": "src/"});
    ContentBlock::Other(
        OtherContent::new("custom_tool_call", data.as_object().cloned().unwrap())
            .with_origin("openai"),
    )
}

/// The answer to [`openai_custom_call`].
fn openai_custom_answer(call_id: &str) -> ContentPart {
    let data = json!({"call_id": call_id, "output": "ok"});
    ContentPart::Other(
        OtherContent::new(
            "custom_tool_call_output",
            data.as_object().cloned().unwrap(),
        )
        .with_origin("openai"),
    )
}

/// The same conversation sent as is, as a failover does: the foreign call
/// and its answer are dropped with warnings, the emptied user message is
/// skipped, and the reply it leaves last is dropped. Neither drop depends on
/// the policy, so the request succeeds under `Error` too.
#[test]
fn a_moved_client_tool_turn_ends_on_the_question() {
    for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
        let mut request = ChatRequest::new(HAIKU)
            .user("Run the formatter.")
            .message(Message::assistant_blocks(vec![
                ContentBlock::text("Running it."),
                openai_custom_call("call_1"),
            ]))
            .message(Message::user_parts(vec![openai_custom_answer("call_1")]));
        request.unsupported_policy = Some(policy);
        let (body, warnings) = encode(&request);
        assert_eq!(
            body["messages"],
            json!([{"role": "user", "content": [{"type": "text", "text": "Run the formatter."}]}]),
            "{policy:?}"
        );
        let subjects: Vec<&Subject> = warnings.iter().map(|w| &w.subject).collect();
        assert_eq!(
            subjects,
            [
                &Subject::Messages {
                    messages: vec![1, 2]
                },
                &Subject::Content {
                    message: 1,
                    block: None
                },
            ],
            "{policy:?}"
        );
        assert!(
            warnings.iter().all(|w| w.kind == WarningKind::Dropped),
            "{warnings:?}"
        );
    }
}

/// A turn that called a function tool and a custom tool keeps its function
/// call: the conversation ends on the `tool_result` turn, and the dropped
/// custom answer leaves no user turn behind.
#[test]
fn a_kept_tool_call_keeps_its_turn_when_the_custom_answer_is_dropped() {
    for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
        let mut request = ChatRequest::new(HAIKU)
            .user("Look up rust and format src/.")
            .message(Message::assistant_blocks(vec![
                ContentBlock::tool_call("c1", "lookup", r#"{"q":"rust"}"#),
                openai_custom_call("call_2"),
            ]))
            .message(Message::tool_result("c1", "lookup", "Rust is a language."))
            .message(Message::user_parts(vec![openai_custom_answer("call_2")]));
        request.unsupported_policy = Some(policy);
        let (body, warnings) = encode(&request);
        assert_eq!(
            body["messages"],
            json!([
                {"role": "user", "content": [{"type": "text", "text": "Look up rust and format src/."}]},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "c1", "name": "lookup", "input": {"q": "rust"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "c1", "content": "Rust is a language."}
                ]}
            ]),
            "{policy:?}"
        );
        // Both drops share a code: one warning per request.
        assert_eq!(kinds(&warnings), vec![WarningKind::Dropped], "{policy:?}");
    }
}

#[test]
fn other_blocks_replay_only_to_anthropic() {
    let mut data = ExtraMap::new();
    data.insert("id".into(), json!("srvtoolu_1"));
    data.insert("name".into(), json!("web_search"));
    data.insert("input".into(), json!({"query": "rust"}));
    let native = OtherContent::new("server_tool_use", data.clone()).with_origin("anthropic");
    let foreign = OtherContent::new("web_search_call", data).with_origin("openai");
    let request = ChatRequest::new(HAIKU)
        .user("q")
        .message(Message::assistant_blocks(vec![
            ContentBlock::Other(native),
            ContentBlock::Other(foreign),
            ContentBlock::text("done"),
        ]));
    let (body, warnings) = encode(&request);
    assert_eq!(
        body["messages"][1]["content"][0],
        json!({"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "rust"}})
    );
    assert_eq!(body["messages"][1]["content"].as_array().unwrap().len(), 2);
    assert_eq!(kinds(&warnings), vec![WarningKind::Dropped]);
}

#[test]
fn text_citations_are_replayed() {
    let text = Text::new("Rust is fast.").with_provider_data(ProviderData::new().with_raw(
        "anthropic",
        "citations",
        json!([{"type": "char_location", "cited_text": "fast"}]),
    ));
    let request = ChatRequest::new(HAIKU)
        .user("q")
        .message(Message::assistant_blocks(vec![ContentBlock::Text(text)]));
    let (body, _) = encode(&request);
    assert_eq!(
        body["messages"][1]["content"][0]["citations"][0]["cited_text"],
        "fast"
    );
}

#[test]
fn invalid_tool_arguments_become_empty_object_with_warning() {
    let request = ChatRequest::new(HAIKU)
        .user("q")
        .message(Message::assistant_blocks(vec![
            ContentBlock::tool_call("t1", "search", r#"{"q": "unterminated"#),
            ContentBlock::tool_call("t2", "noargs", ""),
            ContentBlock::tool_call("t3", "array", "[1,2]"),
        ]));
    let (body, warnings) = encode(&request);
    let content = &body["messages"][1]["content"];
    assert_eq!(content[0]["input"], json!({}));
    assert_eq!(content[1]["input"], json!({}));
    assert_eq!(content[2]["input"], json!({}));
    // Replayed on every turn, so coded, and one warning per code per
    // request: the first call's.
    assert_eq!(kinds(&warnings), vec![WarningKind::Approximated]);
    assert_eq!(
        warnings[0].code.as_deref(),
        Some("approximated.tool_call_arguments")
    );
    assert_eq!(
        warnings[0].message,
        "the arguments of tool call `search` are not valid JSON; they were sent as `{}`"
    );

    let strict = request.unsupported_policy(UnsupportedPolicy::Error);
    let err = encode_err(&strict);
    assert!(matches!(err, Error::Unsupported { .. }));
    assert!(!err.to_string().contains("sent as"), "{err}");
}

#[test]
fn programmatic_tool_caller_is_replayed() {
    let call = ToolCall::new("t1", "f", "{}").with_provider_data(ProviderData::new().with_raw(
        "anthropic",
        "caller",
        json!({"type": "code_execution_20260120", "tool_id": "srvtoolu_1"}),
    ));
    let request = ChatRequest::new(HAIKU)
        .user("q")
        .message(Message::assistant_blocks(vec![ContentBlock::ToolCall(
            call,
        )]));
    let (body, _) = encode(&request);
    assert_eq!(
        body["messages"][1]["content"][0]["caller"]["tool_id"],
        "srvtoolu_1"
    );
}

#[test]
fn computer_toolset_name_is_replayed_and_repeated_on_the_result() {
    // The shapes of Anthropic's computer-use tool documentation: a toolset
    // member's `tool_use` names the toolset, and its `tool_result` must too.
    let member = |id: &str, name: &str, input: &str| {
        ContentBlock::ToolCall(ToolCall::new(id, name, input).with_provider_data(
            ProviderData::new().with_raw("anthropic", "toolset_name", json!("computer")),
        ))
    };
    let request = ChatRequest::new(SONNET_55)
        .user("Search for pictures of cats")
        .message(Message::assistant_blocks(vec![
            member("toolu_1", "left_click", r#"{"coordinate":[640,60]}"#),
            member("toolu_2", "screenshot", "{}"),
            ContentBlock::tool_call("toolu_3", "get_weather", r#"{"city":"Paris"}"#),
        ]))
        .message(Message::tool_result("toolu_1", "left_click", "OK"))
        .message(Message::tool_error(
            "toolu_2",
            "screenshot",
            vec![ContentPart::image_base64("image/png", "iVBORw0KGgo=")],
        ))
        .message(Message::tool_result("toolu_3", "get_weather", "Sunny"))
        .with_options_json(
            "anthropic",
            json!({"server_tools": [{"type": "computer_toolset_20260801"}]}),
        );
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["tools"],
        json!([{"type": "computer_toolset_20260801"}])
    );
    assert_eq!(
        body["messages"][1]["content"][0],
        json!({"type": "tool_use", "id": "toolu_1", "name": "left_click", "toolset_name": "computer", "input": {"coordinate": [640, 60]}})
    );
    assert_eq!(
        body["messages"][1]["content"][2],
        json!({"type": "tool_use", "id": "toolu_3", "name": "get_weather", "input": {"city": "Paris"}})
    );
    let results = &body["messages"][2]["content"];
    assert_eq!(
        results[0],
        json!({"type": "tool_result", "tool_use_id": "toolu_1", "toolset_name": "computer", "content": "OK"})
    );
    assert_eq!(results[1]["toolset_name"], "computer");
    assert_eq!(results[1]["is_error"], true);
    assert_eq!(results[1]["content"][0]["type"], "image");
    assert_eq!(
        results[2],
        json!({"type": "tool_result", "tool_use_id": "toolu_3", "content": "Sunny"}),
        "a function tool's result names no toolset"
    );
}

#[test]
fn replayed_tool_use_fields_cannot_replace_the_call() {
    let call = ToolCall::new("t1", "f", r#"{"a":1}"#).with_provider_data(
        ProviderData::new()
            .with_raw("anthropic", "type", json!("server_tool_use"))
            .with_raw("anthropic", "id", json!("other"))
            .with_raw("anthropic", "name", json!("g"))
            .with_raw("anthropic", "input", json!({}))
            .with_raw("anthropic", "future_field", json!(true)),
    );
    let request = ChatRequest::new(HAIKU)
        .user("q")
        .message(Message::assistant_blocks(vec![ContentBlock::ToolCall(
            call,
        )]));
    let (body, _) = encode(&request);
    assert_eq!(
        body["messages"][1]["content"][0],
        json!({"type": "tool_use", "id": "t1", "name": "f", "input": {"a": 1}, "future_field": true})
    );
}

// --- Tools ---

#[test]
fn tools_map_with_strict_and_provider_extras() {
    let mut tool = weather_tool().strict(true);
    tool.provider_data
        .insert_raw("anthropic", "eager_input_streaming", json!(true));
    let (body, warnings) = encode(&ChatRequest::new(HAIKU).user("q").tools([tool]));
    assert_eq!(
        body["tools"][0],
        json!({
            "name": "get_weather",
            "description": "Get the weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": false
            },
            "strict": true,
            "eager_input_streaming": true
        })
    );
    assert!(warnings.is_empty());
    assert!(body.get("tool_choice").is_none());
}

#[test]
fn non_strict_tools_keep_their_schema() {
    let (body, _) = encode(&ChatRequest::new(HAIKU).user("q").tools([weather_tool()]));
    assert!(
        body["tools"][0]["input_schema"]
            .get("additionalProperties")
            .is_none()
    );
    assert!(body["tools"][0].get("strict").is_none());
}

fn map_tool() -> ToolDefinition {
    ToolDefinition::new(
        "tag",
        json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"labels": {"type": "object", "additionalProperties": {"type": "string"}}},
            "required": ["labels"]
        }),
    )
    .strict(true)
}

#[test]
fn strict_tools_strict_mode_cannot_express_degrade_under_warn() {
    let recursive = ToolDefinition::new(
        "tree",
        json!({"type": "object", "properties": {"children": {"type": "array", "items": {"$ref": "#"}}}}),
    )
    .strict(true);
    for tool in [map_tool(), recursive] {
        let (body, warnings) = encode(&ChatRequest::new(HAIKU).user("q").tools([tool.clone()]));
        let sent = &body["tools"][0];
        assert!(sent.get("strict").is_none(), "{sent}");
        let mut expected = tool.parameters.clone();
        expected.as_object_mut().unwrap().remove("$schema");
        assert_eq!(sent["input_schema"], expected);
        assert_eq!(kinds(&warnings), vec![WarningKind::Unsupported]);
        assert_eq!(
            warnings[0].subject,
            Subject::Field {
                field: RequestField::Tools
            }
        );
        assert!(
            warnings[0]
                .message
                .ends_with("; it was sent without `strict`"),
            "{}",
            warnings[0]
        );
    }
}

#[test]
fn strict_tools_strict_mode_cannot_express_fail_under_error() {
    let err = encode_err(
        &ChatRequest::new(HAIKU)
            .user("q")
            .tools([map_tool()])
            .unsupported_policy(UnsupportedPolicy::Error),
    );
    assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
}

#[test]
fn a_degraded_strict_tool_reports_only_why() {
    // The rewrite would strip `pattern` with an `Approximated` warning, but
    // the map makes the tool not representable: only the `Unsupported`
    // warning remains, as the core strict policy requires of a dialect.
    let tool = ToolDefinition::new(
        "tag",
        json!({
            "type": "object",
            "properties": {
                "code": {"type": "string", "maxLength": 3},
                "labels": {"type": "object", "additionalProperties": {"type": "string"}}
            }
        }),
    )
    .strict(true);
    let (_, warnings) = encode(&ChatRequest::new(HAIKU).user("q").tools([tool]));
    assert_eq!(
        kinds(&warnings),
        vec![WarningKind::Unsupported],
        "{warnings:?}"
    );
    assert!(
        warnings[0]
            .message
            .starts_with("tool `tag` is not strict-representable: map-typed"),
        "{}",
        warnings[0]
    );
}

#[test]
fn malformed_strict_schemas_fail_under_every_policy() {
    let broken = json!({
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/Missing", "title": "A"}}
    });
    let tool = ToolDefinition::new("broken", broken.clone()).strict(true);
    let format = ResponseFormat::json_schema(broken);
    for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
        let err = encode_err(
            &ChatRequest::new(HAIKU)
                .user("q")
                .tools([tool.clone()])
                .unsupported_policy(policy),
        );
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
        assert!(err.to_string().contains("#/$defs/Missing"), "{err}");
        let err = encode_err(
            &ChatRequest::new(SONNET_55)
                .user("q")
                .response_format(format.clone())
                .unsupported_policy(policy),
        );
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
        assert_eq!(err.unsupported_field(), None);
    }
}

/// Strict tools are sent strict on every model, older generations
/// included (ADR 0015): the API decides.
#[test]
fn strict_tools_are_sent_strict_whatever_the_model() {
    let tool = ToolDefinition::new("f", json!({"type": "object"})).strict(true);
    for model in ["claude-sonnet-4-20250514", "claude-3-5-haiku-latest"] {
        let (body, warnings) = encode(&ChatRequest::new(model).user("q").tools([tool.clone()]));
        assert_eq!(body["tools"][0]["strict"], true, "{model}");
        assert!(warnings.is_empty(), "{model}: {warnings:?}");
    }
}

/// A resolver that answers structured output `Unsupported` relaxes strict
/// tools, as the built-in answers no longer do.
#[test]
fn strict_tools_answered_unsupported_drop_the_schema_uri() {
    let tool = ToolDefinition::new(
        "f",
        json!({"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"}),
    )
    .strict(true);
    let encoded = encode_answering(
        &ChatRequest::new("claude-sonnet-4-20250514")
            .user("q")
            .tools([tool]),
        &[(
            allama::ChatCapability::StructuredOutput,
            CapabilitySupport::Unsupported,
        )],
    )
    .unwrap();
    let body = serde_json::to_value(&encoded.body).unwrap();
    assert_eq!(body["tools"][0]["input_schema"], json!({"type": "object"}));
    assert!(body["tools"][0].get("strict").is_none());
    assert_eq!(kinds(&encoded.warnings), vec![WarningKind::Unsupported]);
}

#[test]
fn tool_choice_mapping() {
    let base = ChatRequest::new(HAIKU).user("q").tools([weather_tool()]);
    let choice =
        |c: allama::ToolChoice| encode(&base.clone().tool_choice(c)).0["tool_choice"].clone();
    assert_eq!(choice(allama::ToolChoice::Auto), json!({"type": "auto"}));
    assert_eq!(choice(allama::ToolChoice::Required), json!({"type": "any"}));
    assert_eq!(
        choice(allama::ToolChoice::Specific {
            name: "get_weather".into()
        }),
        json!({"type": "tool", "name": "get_weather"})
    );
    assert_eq!(
        choice(allama::ToolChoice::Disabled),
        json!({"type": "none"})
    );
    let (body, _) = encode(&base.clone().tool_choice(allama::ToolChoice::Disabled));
    assert!(
        body.get("tools").is_some(),
        "tools stay defined for history"
    );
}

/// Forced tool choice is sent to every model, those documented to reject
/// it included: the API's refusal is mapped (ADR 0015). Only a resolver
/// answering `Unsupported` refuses it before sending.
#[test]
fn forced_tool_choice_is_sent_whatever_the_model() {
    for model in [SONNET_55, "claude-opus-5-5", "claude-fable-5-1"] {
        let request = ChatRequest::new(model)
            .user("q")
            .tools([weather_tool()])
            .tool_choice(allama::ToolChoice::Required);
        let (body, warnings) = encode(&request);
        assert_eq!(body["tool_choice"], json!({"type": "any"}), "{model}");
        assert!(warnings.is_empty(), "{model}: {warnings:?}");

        let err = encode_answering(
            &request,
            &[(
                allama::ChatCapability::ForcedToolChoice,
                CapabilitySupport::Unsupported,
            )],
        )
        .unwrap_err();
        assert_eq!(err.unsupported_field(), Some(RequestField::ToolChoice));
    }
}

#[test]
fn parallel_tool_calls_false_disables_parallel_use() {
    let base = ChatRequest::new(HAIKU).user("q").tools([weather_tool()]);
    let (body, warnings) = encode(&base.clone().parallel_tool_calls(false));
    assert_eq!(
        body["tool_choice"],
        json!({"type": "auto", "disable_parallel_tool_use": true})
    );
    assert!(warnings.is_empty());

    let (body, _) = encode(
        &base
            .clone()
            .parallel_tool_calls(false)
            .tool_choice(allama::ToolChoice::Required),
    );
    assert_eq!(
        body["tool_choice"],
        json!({"type": "any", "disable_parallel_tool_use": true})
    );

    let (body, _) = encode(&base.clone().parallel_tool_calls(true));
    assert!(body.get("tool_choice").is_none());

    let (body, _) = encode(&ChatRequest::new(HAIKU).user("q").parallel_tool_calls(false));
    assert!(
        body.get("tool_choice").is_none(),
        "no tools, nothing to control"
    );
}

// --- Hints and policy ---

#[test]
fn unsupported_hints_warn_and_strict_policy_rejects() {
    let request = ChatRequest::new(HAIKU)
        .user("q")
        .seed(7)
        .frequency_penalty(0.5)
        .presence_penalty(0.5);
    let (body, warnings) = encode(&request);
    assert!(body.get("seed").is_none());
    assert_eq!(
        kinds(&warnings),
        vec![WarningKind::Unsupported; 3],
        "{warnings:?}"
    );
    assert_eq!(
        warnings[0].subject,
        Subject::Field {
            field: RequestField::Seed
        }
    );
    let err = encode_err(&request.unsupported_policy(UnsupportedPolicy::Error));
    assert!(matches!(err, Error::Unsupported { .. }));
}

/// Sampling parameters are sent to every model, those documented to
/// reject them included; the API's refusal is mapped (ADR 0015).
#[test]
fn sampling_parameters_are_sent_whatever_the_model() {
    for model in [HAIKU, "claude-opus-4-7", SONNET_55, "claude-mythos-preview"] {
        let request = ChatRequest::new(model)
            .user("q")
            .temperature(0.2)
            .top_p(0.9)
            .top_k(5);
        let (body, warnings) = encode(&request);
        assert!(body["temperature"].as_f64().is_some(), "{model}");
        assert!(body["top_p"].as_f64().is_some(), "{model}");
        assert_eq!(body["top_k"], 5, "{model}");
        assert!(warnings.is_empty(), "{model}: {warnings:?}");
    }
}

#[test]
fn provider_options_are_mapped() {
    let mut metadata = ExtraMap::new();
    metadata.insert("user_id".into(), json!("u1"));
    let request = ChatRequest::new(HAIKU)
        .user("q")
        .stop(["END"])
        .max_output_tokens(100)
        .with_option(
            ChatRequestOptions::new()
                .metadata(metadata)
                .service_tier("standard_only")
                .beta("beta-1")
                .beta("beta-2"),
        );
    let encoded = encode_request(&request, true).unwrap();
    let body = serde_json::to_value(&encoded.body).unwrap();
    assert_eq!(body["metadata"], json!({"user_id": "u1"}));
    assert_eq!(body["service_tier"], "standard_only");
    assert_eq!(body["stop_sequences"], json!(["END"]));
    assert_eq!(body["max_tokens"], 100);
    assert_eq!(body["stream"], true);
    assert_eq!(encoded.betas, vec!["beta-1", "beta-2"]);
}

#[test]
fn options_given_as_json_are_resolved() {
    let request = ChatRequest::new(HAIKU)
        .user("q")
        .with_options_json("anthropic", json!({"service_tier": "auto"}));
    let (body, _) = encode(&request);
    assert_eq!(body["service_tier"], "auto");

    // `top_k` is portable (`ChatRequest::top_k`), not an option.
    for field in ["topk", "top_k"] {
        let bad = ChatRequest::new(HAIKU)
            .user("q")
            .with_options_json("anthropic", json!({field: 3}));
        assert!(
            matches!(encode_err(&bad), Error::InvalidRequest { .. }),
            "{field}"
        );
    }
}

// --- Structured output ---

#[test]
fn json_schema_maps_to_output_config_format() {
    let request =
        ChatRequest::new(SONNET_55)
            .user("q")
            .response_format(ResponseFormat::json_schema(json!({
                "type": "object",
                "properties": {"greeting": {"type": "string", "maxLength": 20}},
                "required": ["greeting"]
            })));
    let (body, warnings) = encode(&request);
    assert_eq!(
        body["output_config"],
        json!({"format": {"type": "json_schema", "schema": {
            "type": "object",
            "properties": {"greeting": {"type": "string"}},
            "required": ["greeting"],
            "additionalProperties": false
        }}})
    );
    assert_eq!(kinds(&warnings), vec![WarningKind::Approximated]);
}

/// `output_config.format` is always constrained, so `non_strict` (which
/// does not require constraining) changes neither the wire nor the
/// warnings.
#[test]
fn json_schema_is_constrained_strict_or_not() {
    let schema = json!({
        "type": "object",
        "properties": {"greeting": {"type": "string"}},
        "required": ["greeting"]
    });
    let request = |format| {
        ChatRequest::new(SONNET_55)
            .user("q")
            .response_format(format)
    };
    let (strict, strict_warnings) = encode(&request(ResponseFormat::json_schema(schema.clone())));
    let (relaxed, relaxed_warnings) =
        encode(&request(ResponseFormat::json_schema(schema).non_strict()));
    assert_eq!(strict["output_config"], relaxed["output_config"]);
    assert_eq!(
        strict["output_config"]["format"]["schema"]["additionalProperties"],
        false
    );
    assert!(strict_warnings.is_empty(), "{strict_warnings:?}");
    assert!(relaxed_warnings.is_empty(), "{relaxed_warnings:?}");
}

#[test]
fn schemaless_json_and_unsupported_models_fail() {
    let err = encode_err(
        &ChatRequest::new(SONNET_55)
            .user("q")
            .response_format(ResponseFormat::Json),
    );
    assert!(matches!(err, Error::Unsupported { .. }));
    // Sent to every model, older generations included (ADR 0015); refused
    // only where a resolver answers `Unsupported`.
    let schema = ChatRequest::new("claude-sonnet-4-20250514")
        .user("q")
        .response_format(ResponseFormat::json_schema(json!({"type": "object"})));
    let (body, _) = encode(&schema);
    assert_eq!(body["output_config"]["format"]["type"], "json_schema");
    let err = encode_answering(
        &schema,
        &[(
            allama::ChatCapability::StructuredOutput,
            CapabilitySupport::Unsupported,
        )],
    )
    .unwrap_err();
    assert_eq!(err.unsupported_field(), Some(RequestField::ResponseFormat));
    let (body, _) = encode(
        &ChatRequest::new(HAIKU)
            .user("q")
            .response_format(ResponseFormat::Text),
    );
    assert!(body.get("output_config").is_none());
}

#[test]
fn json_schemas_strict_mode_cannot_express_fail_whatever_the_policy() {
    let request = ChatRequest::new(SONNET_55).user("q").response_format(
        ResponseFormat::json_schema(json!({
            "type": "object",
            "properties": {"labels": {"type": "object", "additionalProperties": {"type": "string"}}}
        })),
    );
    let err = encode_err(&request);
    assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
    assert!(err.to_string().contains("map-typed"), "{err}");
}

// --- Reasoning ---

#[test]
fn budget_models_map_effort_to_budget_tokens() {
    let (body, warnings) = encode(
        &ChatRequest::new(HAIKU)
            .user("q")
            .max_output_tokens(16000)
            .reasoning(ReasoningEffort::Low),
    );
    assert_eq!(
        body["thinking"],
        json!({"type": "enabled", "budget_tokens": 2048})
    );
    assert_eq!(body["max_tokens"], 16000);
    assert!(body.get("output_config").is_none());
    assert!(warnings.is_empty());
}

// Budget rules (ADR 0003, principle 4): Anthropic requires budget_tokens >=
// 1024 and < max_tokens, except with interleaved thinking, where the budget
// may exceed max_tokens.
// https://platform.claude.com/docs/en/build-with-claude/extended-thinking#budget-rules-and-tuning

const SONNET_45: &str = "claude-sonnet-4-5-20250929";

fn reasoning_field() -> Subject {
    Subject::Field {
        field: RequestField::Reasoning,
    }
}

fn interleaved() -> ChatRequestOptions {
    ChatRequestOptions::new().beta("interleaved-thinking-2025-05-14")
}

#[test]
fn budget_raises_the_default_max_tokens() {
    // max_output_tokens unset: the default is the adapter's, so it grows.
    let (body, warnings) = encode(
        &ChatRequest::new(HAIKU)
            .user("q")
            .reasoning(ReasoningEffort::High),
    );
    assert_eq!(body["thinking"]["budget_tokens"], 8192);
    assert_eq!(body["max_tokens"], 8192 + 4096);
    assert!(warnings.is_empty());

    let (body, warnings) = encode(
        &ChatRequest::new(HAIKU)
            .user("q")
            .with_option(ChatRequestOptions::new().budget_tokens(10_000)),
    );
    assert_eq!(body["thinking"]["budget_tokens"], 10_000);
    assert_eq!(body["max_tokens"], 10_000 + 4096);
    assert!(warnings.is_empty());
}

#[test]
fn effort_budget_is_lowered_to_fit_an_explicit_cap() {
    let request = ChatRequest::new(HAIKU)
        .user("q")
        .max_output_tokens(6000)
        .reasoning(ReasoningEffort::High);
    let (body, warnings) = encode(&request);
    assert_eq!(
        body["thinking"],
        json!({"type": "enabled", "budget_tokens": 3000})
    );
    assert_eq!(body["max_tokens"], 6000, "a caller-set cap is never raised");
    assert_eq!(kinds(&warnings), vec![WarningKind::Approximated]);
    assert_eq!(warnings[0].subject, reasoning_field());
    assert!(
        warnings[0]
            .message
            .contains("3000 tokens was used instead of 8192"),
        "{warnings:?}"
    );

    // Lowering the budget is an approximation, never refused.
    let encoded = encode_request(&request.unsupported_policy(UnsupportedPolicy::Error), false)
        .expect("a lowered budget is not refused");
    assert_eq!(
        serde_json::to_value(&encoded.body).unwrap()["thinking"]["budget_tokens"],
        3000
    );
}

#[test]
fn effort_budget_leaves_room_for_the_answer() {
    // A cap just above the budget used to leave one token for the answer.
    let request = ChatRequest::new(HAIKU)
        .user("q")
        .max_output_tokens(2049)
        .reasoning(ReasoningEffort::Low);
    let (body, warnings) = encode(&request);
    assert_eq!(body["thinking"]["budget_tokens"], 1024);
    assert_eq!(kinds(&warnings), vec![WarningKind::Approximated]);
    assert!(
        warnings[0]
            .message
            .contains("1024 tokens was used instead of 2048"),
        "{warnings:?}"
    );

    // Half the cap exactly: nothing was lowered (the recorded fixture).
    let (body, warnings) = encode(
        &ChatRequest::new(HAIKU)
            .user("q")
            .max_output_tokens(8192)
            .reasoning(ReasoningEffort::Medium),
    );
    assert_eq!(body["thinking"]["budget_tokens"], 4096);
    assert!(warnings.is_empty(), "{warnings:?}");
}

#[test]
fn minimal_effort_thinks_under_a_small_cap() {
    let (body, warnings) = encode(
        &ChatRequest::new(HAIKU)
            .user("q")
            .max_output_tokens(1500)
            .reasoning(ReasoningEffort::Minimal),
    );
    assert_eq!(
        body["thinking"],
        json!({"type": "enabled", "budget_tokens": 1024})
    );
    assert_eq!(body["max_tokens"], 1500);
    assert!(warnings.is_empty(), "{warnings:?}");
}

#[test]
fn strict_policy_never_refuses_a_cap_between_the_budget_and_twice_it() {
    for effort in [
        ReasoningEffort::Minimal,
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
        ReasoningEffort::XHigh,
    ] {
        let budget = model::budget_for_effort(effort);
        for cap in (budget + 1)..(2 * budget) {
            let request = ChatRequest::new(HAIKU)
                .user("q")
                .max_output_tokens(cap)
                .reasoning(effort)
                .unsupported_policy(UnsupportedPolicy::Error);
            let encoded = encode_request(&request, false);
            assert!(encoded.is_ok(), "{effort:?} at {cap}: {encoded:?}");
        }
    }
}

#[test]
fn effort_budget_is_monotone_in_the_cap() {
    for effort in [
        ReasoningEffort::Minimal,
        ReasoningEffort::Low,
        ReasoningEffort::Medium,
        ReasoningEffort::High,
        ReasoningEffort::XHigh,
    ] {
        let mut previous = 0;
        for cap in 1000..=40_000 {
            let (body, warnings) = encode(
                &ChatRequest::new(HAIKU)
                    .user("q")
                    .max_output_tokens(cap)
                    .reasoning(effort),
            );
            assert_eq!(body["max_tokens"], cap);
            let Some(budget) = body["thinking"]["budget_tokens"].as_u64() else {
                assert!(
                    cap <= MIN_THINKING_BUDGET,
                    "{effort:?}: thinking dropped at {cap}"
                );
                assert_eq!(kinds(&warnings), vec![WarningKind::Unsupported]);
                continue;
            };
            let budget = u32::try_from(budget).unwrap();
            assert!(cap > MIN_THINKING_BUDGET, "{effort:?}: thinking at {cap}");
            assert!(budget < cap, "{effort:?} at {cap}: {budget}");
            assert!(
                budget >= previous,
                "{effort:?} at {cap}: {budget} < {previous}"
            );
            previous = budget;
        }
    }
}

#[test]
fn effort_budget_without_room_drops_thinking() {
    // No budget at or above the 1024-token minimum fits below the cap.
    for cap in [1000, 1024] {
        let request = ChatRequest::new(HAIKU)
            .user("q")
            .max_output_tokens(cap)
            .reasoning(ReasoningEffort::Medium);
        let (body, warnings) = encode(&request);
        assert!(body.get("thinking").is_none());
        assert_eq!(body["max_tokens"], cap);
        assert_eq!(kinds(&warnings), vec![WarningKind::Unsupported]);
        assert_eq!(warnings[0].subject, reasoning_field());
        assert!(
            warnings[0]
                .message
                .starts_with("thinking needs `max_output_tokens` above 1024"),
            "{warnings:?}"
        );

        let err = encode_err(&request.unsupported_policy(UnsupportedPolicy::Error));
        assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
        assert_eq!(err.unsupported_field(), Some(RequestField::Reasoning));
        assert!(!err.to_string().contains("was not sent"), "{err}");
        assert!(
            err.to_string()
                .ends_with("raise `max_output_tokens` above 1024 to enable thinking"),
            "the refusal keeps the remedy: {err}"
        );
    }
}

#[test]
fn dropped_thinking_allows_forced_tool_choice() {
    // Thinking is off once dropped, so nothing conflicts with forcing a tool.
    let (body, warnings) = encode(
        &ChatRequest::new(HAIKU)
            .user("q")
            .max_output_tokens(1000)
            .tools([weather_tool()])
            .tool_choice(allama::ToolChoice::Required)
            .reasoning(ReasoningEffort::Low),
    );
    assert!(body.get("thinking").is_none());
    assert_eq!(body["tool_choice"]["type"], "any");
    assert_eq!(kinds(&warnings), vec![WarningKind::Unsupported]);
}

#[test]
fn budget_option_contradicting_an_explicit_cap_is_invalid() {
    for model in [HAIKU, "claude-sonnet-4-6"] {
        let request = ChatRequest::new(model)
            .user("q")
            .max_output_tokens(1000)
            .with_option(ChatRequestOptions::new().budget_tokens(2048));
        let err = encode_err(&request);
        assert!(
            matches!(err, Error::InvalidRequest { .. }),
            "{model}: {err:?}"
        );
        assert!(err.to_string().contains("2048"), "{err}");

        // Equal is not less.
        let request = ChatRequest::new(model)
            .user("q")
            .max_output_tokens(2048)
            .with_option(ChatRequestOptions::new().budget_tokens(2048));
        assert!(matches!(encode_err(&request), Error::InvalidRequest { .. }));
    }
}

#[test]
fn interleaved_thinking_with_tools_neither_clamps_nor_raises() {
    // The budget spans every thinking block of the turn.
    let base = || {
        ChatRequest::new(SONNET_45)
            .user("q")
            .tools([weather_tool()])
    };

    let (body, warnings) = encode(
        &base()
            .max_output_tokens(4000)
            .reasoning(ReasoningEffort::High)
            .with_option(interleaved()),
    );
    assert_eq!(body["thinking"]["budget_tokens"], 8192);
    assert_eq!(body["max_tokens"], 4000);
    assert!(warnings.is_empty());

    let (body, warnings) = encode(
        &base()
            .max_output_tokens(1000)
            .with_option(interleaved().budget_tokens(2048)),
    );
    assert_eq!(body["thinking"]["budget_tokens"], 2048);
    assert_eq!(body["max_tokens"], 1000);
    assert!(warnings.is_empty());

    let (body, _) = encode(
        &base()
            .reasoning(ReasoningEffort::High)
            .with_option(interleaved()),
    );
    assert_eq!(body["thinking"]["budget_tokens"], 8192);
    assert_eq!(body["max_tokens"], DEFAULT_MAX_TOKENS);
}

#[test]
fn interleaved_exception_needs_tools_and_a_model_that_interleaves() {
    // Without tools there is nothing to interleave with.
    let (body, warnings) = encode(
        &ChatRequest::new(SONNET_45)
            .user("q")
            .max_output_tokens(4000)
            .reasoning(ReasoningEffort::High)
            .with_option(interleaved()),
    );
    assert_eq!(body["thinking"]["budget_tokens"], 2000);
    assert_eq!(kinds(&warnings), vec![WarningKind::Approximated]);

    // Haiku 4.5 ignores the beta; Opus 4.6 interleaves only adaptively.
    for model in [HAIKU, "claude-opus-4-6"] {
        let request = ChatRequest::new(model)
            .user("q")
            .tools([weather_tool()])
            .max_output_tokens(1000)
            .with_option(interleaved().budget_tokens(2048));
        assert!(
            matches!(encode_err(&request), Error::InvalidRequest { .. }),
            "{model}"
        );
    }
}

#[test]
fn adaptive_models_map_effort_and_display() {
    let (body, warnings) = encode(
        &ChatRequest::new(SONNET_55).user("q").reasoning(
            ReasoningConfig::new()
                .effort(ReasoningEffort::Medium)
                .output(ReasoningOutput::Summary),
        ),
    );
    assert_eq!(
        body["thinking"],
        json!({"type": "adaptive", "display": "summarized"})
    );
    assert_eq!(body["output_config"], json!({"effort": "medium"}));
    assert!(warnings.is_empty());

    let (body, warnings) = encode(
        &ChatRequest::new("claude-opus-4-6").user("q").reasoning(
            ReasoningConfig::new()
                .effort(ReasoningEffort::XHigh)
                .output(ReasoningOutput::Full),
        ),
    );
    assert_eq!(body["output_config"]["effort"], "max");
    assert_eq!(body["thinking"]["display"], "summarized");
    assert_eq!(
        kinds(&warnings),
        vec![WarningKind::Approximated, WarningKind::Approximated]
    );
}

/// Claude Haiku 5.5, and a later version of any family, takes an effort in
/// the adaptive form, never a budget, which it refuses.
#[test]
fn newer_versions_map_effort_to_the_adaptive_form() {
    for model in [
        "claude-haiku-5-5",
        "claude-haiku-6",
        "claude-sonnet-6",
        "claude-opus-6",
        "claude-fable-6",
        "claude-mythos-6",
    ] {
        let (body, warnings) = encode(
            &ChatRequest::new(model)
                .user("q")
                .reasoning(ReasoningEffort::Low),
        );
        assert_eq!(body["thinking"], json!({"type": "adaptive"}), "{model}");
        assert_eq!(body["output_config"], json!({"effort": "low"}), "{model}");
        assert!(warnings.is_empty(), "{model}: {warnings:?}");
    }
    // Every level is sent as asked on Haiku 5.5.
    let (body, warnings) = encode(
        &ChatRequest::new("claude-haiku-5-5")
            .user("q")
            .reasoning(ReasoningEffort::XHigh),
    );
    assert_eq!(body["output_config"]["effort"], "xhigh");
    assert!(warnings.is_empty(), "{warnings:?}");
}

/// The output setting alone never turns thinking on. Where the model thinks
/// anyway it goes out as `display`; elsewhere it has no effect and is an
/// unsupported hint.
#[test]
fn reasoning_output_alone_never_turns_thinking_on() {
    let output_only = |model: &str| {
        ChatRequest::new(model)
            .user("q")
            .reasoning(ReasoningConfig::new().output(ReasoningOutput::Omitted))
    };
    for model in [
        HAIKU,
        "claude-sonnet-4-5",
        "claude-opus-4-6",
        "claude-opus-4-8",
        "my-gateway-alias",
    ] {
        let (body, warnings) = encode(&output_only(model));
        assert!(body.get("thinking").is_none(), "{model}: {body}");
        assert!(body.get("output_config").is_none(), "{model}: {body}");
        assert_eq!(body["max_tokens"], DEFAULT_MAX_TOKENS, "{model}");
        assert_eq!(kinds(&warnings), vec![WarningKind::Unsupported], "{model}");
        assert!(
            warnings[0].message.contains("set a reasoning effort"),
            "{model}"
        );
        assert!(
            matches!(
                encode_err(&output_only(model).unsupported_policy(UnsupportedPolicy::Error)),
                Error::Unsupported { .. }
            ),
            "{model}"
        );
    }

    // Thinking is on by default: only its display is set, with no effort.
    for model in [
        SONNET_55,
        "claude-opus-5-5",
        "claude-fable-5-1",
        "claude-haiku-5-5",
    ] {
        let (body, warnings) = encode(&output_only(model));
        assert_eq!(
            body["thinking"],
            json!({"type": "adaptive", "display": "omitted"}),
            "{model}"
        );
        assert!(body.get("output_config").is_none(), "{model}");
        assert!(warnings.is_empty(), "{model}: {warnings:?}");
    }

    // An empty config asks for nothing.
    for model in [HAIKU, "claude-opus-4-6", SONNET_55] {
        let (body, warnings) = encode(
            &ChatRequest::new(model)
                .user("q")
                .reasoning(ReasoningConfig::default()),
        );
        assert!(body.get("thinking").is_none(), "{model}: {body}");
        assert!(warnings.is_empty(), "{model}: {warnings:?}");
    }
}

#[test]
fn effort_none_disables_thinking_per_model() {
    let none = |model: &str| {
        encode(
            &ChatRequest::new(model)
                .user("q")
                .reasoning(ReasoningEffort::None),
        )
    };

    let (body, warnings) = none(HAIKU);
    assert!(body.get("thinking").is_none());
    assert!(warnings.is_empty());

    let (body, _) = none("claude-opus-4-8");
    assert_eq!(body["thinking"], json!({"type": "disabled"}));

    let (body, _) = none(SONNET_55);
    assert_eq!(body["thinking"], json!({"type": "between_tools"}));

    // Models documented as always thinking get the adaptive off value too,
    // not a rewritten effort: the API's refusal is mapped (ADR 0015).
    for model in [
        "claude-opus-5-5",
        "claude-fable-5-1",
        "claude-mythos-preview",
    ] {
        let (body, warnings) = none(model);
        assert_eq!(body["thinking"], json!({"type": "disabled"}), "{model}");
        assert!(body.get("output_config").is_none(), "{model}");
        assert!(warnings.is_empty(), "{model}: {warnings:?}");
    }
}

/// Reasoning is sent to older generations too (ADR 0015), in their wire
/// format; only a resolver answering `Unsupported` drops it, as a hint.
#[test]
fn reasoning_answered_unsupported_is_a_hint() {
    let request = ChatRequest::new("claude-3-5-haiku-latest")
        .user("q")
        .reasoning(ReasoningEffort::High);
    let (body, warnings) = encode(&request);
    assert_eq!(body["thinking"]["type"], "enabled");
    assert!(warnings.is_empty(), "{warnings:?}");

    let unsupported = [(
        allama::ChatCapability::ReasoningConfig,
        CapabilitySupport::Unsupported,
    )];
    let encoded = encode_answering(&request, &unsupported).unwrap();
    let body = serde_json::to_value(&encoded.body).unwrap();
    assert!(body.get("thinking").is_none());
    assert_eq!(kinds(&encoded.warnings), vec![WarningKind::Unsupported]);

    let strict = request.unsupported_policy(UnsupportedPolicy::Error);
    assert!(matches!(
        encode_answering(&strict, &unsupported),
        Err(Error::Unsupported { .. })
    ));

    let off = ChatRequest::new("claude-3-5-haiku-latest")
        .user("q")
        .reasoning(ReasoningEffort::None);
    let encoded = encode_answering(&off, &unsupported).unwrap();
    assert!(encoded.body.thinking.is_none());
    assert!(encoded.warnings.is_empty());
}

/// An explicit `budget_tokens` is sent to every model, those that take
/// only adaptive thinking included; the API's refusal is mapped (ADR 0015).
#[test]
fn budget_tokens_option_is_sent_whatever_the_model() {
    for model in [SONNET_55, "claude-opus-4-7", "my-proxy-model"] {
        let (body, warnings) = encode(
            &ChatRequest::new(model)
                .user("q")
                .with_option(ChatRequestOptions::new().budget_tokens(2048)),
        );
        assert_eq!(
            body["thinking"],
            json!({"type": "enabled", "budget_tokens": 2048}),
            "{model}"
        );
        assert!(warnings.is_empty(), "{model}: {warnings:?}");
    }

    let request = ChatRequest::new(HAIKU)
        .user("q")
        .with_option(ChatRequestOptions::new().budget_tokens(512));
    assert!(matches!(encode_err(&request), Error::InvalidRequest { .. }));
}

#[test]
fn budget_tokens_option_overrides_effort_with_warning() {
    let (body, warnings) = encode(
        &ChatRequest::new(HAIKU)
            .user("q")
            .max_output_tokens(20000)
            .reasoning(ReasoningEffort::Low)
            .with_option(ChatRequestOptions::new().budget_tokens(5000)),
    );
    assert_eq!(body["thinking"]["budget_tokens"], 5000);
    assert_eq!(kinds(&warnings), vec![WarningKind::Approximated]);
}

/// "Forced tool use (`tool_choice: {"type": "any"}` or `{"type": "tool",
/// ...}`) is incompatible with manual extended thinking but works with
/// adaptive thinking"
/// (<https://platform.claude.com/docs/en/build-with-claude/thinking#limits-and-feature-compatibility>):
/// a protocol fact, refused before sending. The models that reject forced
/// tool use on every request are left to the API (ADR 0015).
#[test]
fn forced_tool_choice_with_thinking_is_unsupported() {
    let forced = |model: &str| {
        ChatRequest::new(model)
            .user("q")
            .tools([weather_tool()])
            .tool_choice(allama::ToolChoice::Required)
            .reasoning(ReasoningEffort::Low)
    };
    // Manual thinking: refused.
    assert!(matches!(
        encode_err(&forced(HAIKU)),
        Error::Unsupported { .. }
    ));
    let manual = ChatRequest::new("claude-sonnet-4-6")
        .user("q")
        .tools([weather_tool()])
        .tool_choice(allama::ToolChoice::Required)
        .with_option(ChatRequestOptions::new().budget_tokens(2048));
    assert!(matches!(encode_err(&manual), Error::Unsupported { .. }));
    // Adaptive thinking: sent.
    for model in ["claude-sonnet-4-6", "claude-opus-4-7", "claude-sonnet-5"] {
        let (body, _) = encode(&forced(model));
        assert_eq!(body["thinking"]["type"], "adaptive", "{model}");
        assert_eq!(body["tool_choice"]["type"], "any", "{model}");
    }
    // The models documented to reject it always: sent.
    for model in ["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1"] {
        let (body, _) = encode(&forced(model));
        assert_eq!(body["tool_choice"]["type"], "any", "{model}");
    }
}

/// While thinking is on (manual or adaptive), sampling takes `temperature`
/// 1 only, no `top_k`, and `top_p` between 0.95 and 1: "`temperature` and
/// `top_k` are incompatible with thinking, and `top_p` is allowed at values
/// between 0.95 and 1", which the API documents for thinking as a whole
/// (<https://platform.claude.com/docs/en/build-with-claude/thinking#limits-and-feature-compatibility>).
#[test]
fn sampling_is_restricted_while_thinking() {
    let sampled = |model: &str, temperature: f64, top_p: f64| {
        ChatRequest::new(model)
            .user("q")
            .temperature(temperature)
            .top_p(top_p)
            .reasoning(ReasoningEffort::High)
    };
    // Adaptive (Sonnet 4.6) and manual (Sonnet 4.5) thinking alike.
    for model in ["claude-sonnet-4-6", "claude-sonnet-4-5"] {
        let (body, warnings) = encode(&sampled(model, 0.3, 0.5).top_k(5));
        assert!(body["thinking"]["type"].is_string(), "{model}");
        assert!(body.get("temperature").is_none(), "{model}");
        assert!(body.get("top_p").is_none(), "{model}");
        assert!(body.get("top_k").is_none(), "{model}");
        assert_eq!(
            kinds(&warnings),
            vec![WarningKind::Unsupported; 3],
            "{model}"
        );

        let (body, warnings) = encode(&sampled(model, 1.0, 0.97));
        assert_eq!(body["temperature"], 1.0, "{model}");
        assert!(body["top_p"].as_f64().is_some(), "{model}");
        assert!(warnings.is_empty(), "{model}: {warnings:?}");

        let strict = sampled(model, 0.3, 0.97).unsupported_policy(UnsupportedPolicy::Error);
        assert!(matches!(encode_err(&strict), Error::Unsupported { .. }));
    }
    // Without thinking, nothing is restricted.
    let plain = ChatRequest::new("claude-sonnet-4-5")
        .user("q")
        .temperature(0.3)
        .top_p(0.5);
    let (body, warnings) = encode(&plain);
    assert!(body["temperature"].as_f64().is_some() && body["top_p"].as_f64().is_some());
    assert!(warnings.is_empty());
}

/// `temperature` and `top_p` are sent together to every model, those
/// documented to reject the pair included (Opus 4.1, Haiku 4.5); the API's
/// refusal is mapped (ADR 0015).
#[test]
fn temperature_and_top_p_are_sent_together() {
    for model in ["claude-opus-4-1", "claude-haiku-4-5-20251001"] {
        let request = ChatRequest::new(model)
            .user("q")
            .temperature(0.3)
            .top_p(0.9);
        let (body, warnings) = encode(&request);
        assert!(body["temperature"].as_f64().is_some(), "{model}");
        assert!(body["top_p"].as_f64().is_some(), "{model}");
        assert!(warnings.is_empty(), "{model}: {warnings:?}");
    }
}

// --- Caching ---

#[test]
fn cache_breakpoints_are_placed_on_the_right_blocks() {
    let request = ChatRequest::new(HAIKU)
        .system(SystemPrompt::new("Long instructions").with_cache(CacheBreakpoint::new()))
        .tools([weather_tool().with_cache(CacheBreakpoint::with_retention(CacheRetention::Long))])
        .message(
            Message::user(vec![ContentPart::text("a"), ContentPart::text("b")])
                .with_cache(CacheBreakpoint::new()),
        )
        .message(
            Message::assistant_blocks(vec![
                ContentBlock::text("answer"),
                signed_reasoning("late thought", "sig"),
            ])
            .with_cache(CacheBreakpoint::new()),
        );
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["tools"][0]["cache_control"],
        json!({"type": "ephemeral", "ttl": "1h"})
    );
    assert_eq!(
        body["system"][0]["cache_control"],
        json!({"type": "ephemeral"})
    );
    let user = &body["messages"][0]["content"];
    assert!(user[0].get("cache_control").is_none());
    assert_eq!(user[1]["cache_control"], json!({"type": "ephemeral"}));
    let assistant = &body["messages"][1]["content"];
    assert_eq!(
        assistant[0]["cache_control"],
        json!({"type": "ephemeral"}),
        "thinking blocks cannot be cached; the breakpoint moves to the last cacheable block"
    );
    assert!(assistant[1].get("cache_control").is_none());
}

#[test]
fn tool_message_breakpoint_lands_on_its_tool_result() {
    let mut tool = ToolMessage::new("t1", "get_weather", "Sunny");
    tool.cache = Some(CacheBreakpoint::new());
    let request = ChatRequest::new(HAIKU)
        .user("q")
        .message(Message::assistant_blocks(vec![ContentBlock::tool_call(
            "t1",
            "get_weather",
            "{}",
        )]))
        .message(tool);
    let (body, _) = encode(&request);
    assert_eq!(
        body["messages"][2]["content"][0]["cache_control"],
        json!({"type": "ephemeral"})
    );
}

#[test]
fn excess_breakpoints_are_dropped_with_warnings() {
    let mut request =
        ChatRequest::new(HAIKU).system(SystemPrompt::new("sys").with_cache(CacheBreakpoint::new()));
    for i in 0..5 {
        request =
            request.message(Message::user(format!("m{i}")).with_cache(CacheBreakpoint::new()));
        request = request.message(Message::assistant("ok"));
    }
    let (body, warnings) = encode(&request);
    let count = body.to_string().matches("cache_control").count();
    assert_eq!(count, 4);
    assert_eq!(
        body["system"][0]["cache_control"],
        json!({"type": "ephemeral"})
    );
    // The two oldest message breakpoints are dropped.
    assert!(
        body["messages"][0]["content"][0]
            .get("cache_control")
            .is_none()
    );
    assert!(
        body["messages"][2]["content"][0]
            .get("cache_control")
            .is_none()
    );
    assert_eq!(
        body["messages"][8]["content"][0]["cache_control"],
        json!({"type": "ephemeral"})
    );
    assert_eq!(
        kinds(&warnings),
        vec![WarningKind::Dropped, WarningKind::Dropped]
    );
}

#[test]
fn automatic_caching_uses_top_level_cache_control_and_a_slot() {
    let mut request =
        ChatRequest::new(HAIKU).cache(PromptCache::automatic().retention(CacheRetention::Long));
    for i in 0..4 {
        request = request.message(
            Message::user(format!("m{i}"))
                .with_cache(CacheBreakpoint::with_retention(CacheRetention::Long)),
        );
        request = request.message(Message::assistant("ok"));
    }
    let (body, warnings) = encode(&request);
    assert_eq!(
        body["cache_control"],
        json!({"type": "ephemeral", "ttl": "1h"})
    );
    let explicit = body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["content"][0].get("cache_control").is_some())
        .count();
    assert_eq!(explicit, 3);
    assert_eq!(kinds(&warnings), vec![WarningKind::Dropped]);
}

/// A key set for another provider's routing never buys
/// Anthropic cache writes.
#[test]
fn prompt_cache_key_is_an_unsupported_hint_and_turns_no_caching_on() {
    let request = ChatRequest::new(HAIKU)
        .user("q")
        .cache(PromptCache::new().key("session-1"));
    let (body, warnings) = encode(&request);
    assert!(body.get("cache_control").is_none(), "{body}");
    assert_eq!(kinds(&warnings), vec![WarningKind::Unsupported]);
    assert!(matches!(
        encode_err(&request.unsupported_policy(UnsupportedPolicy::Error)),
        Error::Unsupported { .. }
    ));
}

/// A retention alone is not the intent to cache automatically.
#[test]
fn prompt_cache_retention_alone_turns_no_caching_on() {
    let request = ChatRequest::new(HAIKU)
        .user("q")
        .cache(PromptCache::new().retention(CacheRetention::Long));
    let (body, warnings) = encode(&request);
    assert!(body.get("cache_control").is_none(), "{body}");
    assert_eq!(kinds(&warnings), vec![WarningKind::Unsupported]);
    assert!(warnings[0].message.contains("PromptCache::automatic"));
    assert!(matches!(
        encode_err(&request.unsupported_policy(UnsupportedPolicy::Error)),
        Error::Unsupported { .. }
    ));
}

/// Without automatic caching every breakpoint slot is the caller's.
#[test]
fn breakpoints_keep_all_four_slots_without_automatic_caching() {
    let mut request = ChatRequest::new(HAIKU).cache(PromptCache::new().key("k"));
    for i in 0..4 {
        request = request
            .message(Message::user(format!("m{i}")).with_cache(CacheBreakpoint::new()))
            .message(Message::assistant("ok"));
    }
    let (body, warnings) = encode(&request);
    assert!(body.get("cache_control").is_none());
    let explicit = body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["content"][0].get("cache_control").is_some())
        .count();
    assert_eq!(explicit, 4);
    // Only the key's hint.
    assert_eq!(kinds(&warnings), vec![WarningKind::Unsupported]);
}

#[test]
fn automatic_caching_with_the_default_retention() {
    let request = ChatRequest::new(HAIKU)
        .user("q")
        .cache(PromptCache::automatic());
    let (body, warnings) = encode(&request);
    assert_eq!(body["cache_control"], json!({"type": "ephemeral"}));
    assert!(warnings.is_empty(), "{warnings:?}");
}

#[test]
fn long_retention_after_short_is_shortened() {
    let request = ChatRequest::new(HAIKU)
        .system(SystemPrompt::new("sys").with_cache(CacheBreakpoint::new()))
        .message(
            Message::user("q").with_cache(CacheBreakpoint::with_retention(CacheRetention::Long)),
        )
        .cache(PromptCache::automatic().retention(CacheRetention::Long));
    let (body, warnings) = encode(&request);
    assert_eq!(
        body["messages"][0]["content"][0]["cache_control"],
        json!({"type": "ephemeral"})
    );
    assert_eq!(body["cache_control"], json!({"type": "ephemeral"}));
    assert_eq!(
        kinds(&warnings),
        vec![WarningKind::Approximated, WarningKind::Approximated]
    );
}

#[test]
fn base64_decoder() {
    assert_eq!(decode_base64("aGVsbG8gd29ybGQ=").unwrap(), b"hello world");
    assert_eq!(decode_base64("aGk").unwrap(), b"hi");
    assert!(decode_base64("!!").is_none());
}

#[test]
fn assistant_message_struct_is_supported() {
    let message = AssistantMessage::new(vec![ContentBlock::text("x")]);
    let (body, _) = encode(&ChatRequest::new(HAIKU).user("q").message(message));
    assert_eq!(body["messages"][1]["role"], "assistant");
}

#[test]
fn server_tools_are_sent_after_function_tools() {
    let request = ChatRequest::new("claude-haiku-4-5")
        .user("Latest Rust release?")
        .tools([weather_tool()])
        .with_options_json(
            "anthropic",
            json!({"server_tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 2}]}),
        );
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(body["tools"][0]["name"], "get_weather");
    assert_eq!(
        body["tools"][1],
        json!({"type": "web_search_20250305", "name": "web_search", "max_uses": 2})
    );

    let nameless = ChatRequest::new("claude-haiku-4-5")
        .user("hi")
        .with_options_json(
            "anthropic",
            json!({"server_tools": [{"type": "web_search_20250305"}]}),
        );
    assert!(matches!(encode_err(&nameless), Error::Unsupported { .. }));
}

#[test]
fn computer_toolset_is_declared_without_a_name() {
    let toolset = json!({
        "type": "computer_toolset_20260801",
        "configs": {"zoom": {"enabled": false}},
        "cache_control": {"type": "ephemeral"}
    });
    let request = ChatRequest::new(SONNET_55)
        .user("Open the browser")
        .tools([weather_tool()])
        .with_options_json("anthropic", json!({"server_tools": [toolset.clone()]}));
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(body["tools"][1], toolset, "sent verbatim, with no `name`");

    // A name other than a string is still refused.
    let bad = ChatRequest::new(SONNET_55).user("hi").with_options_json(
        "anthropic",
        json!({"server_tools": [{"type": "computer_toolset_20260801", "name": 7}]}),
    );
    assert!(matches!(encode_err(&bad), Error::Unsupported { .. }));
}

#[test]
fn provider_tool_blocks_replay_only_to_anthropic() {
    let call = allama::ProviderToolCall::new("srvtoolu_1", "web_search", r#"{"query":"rust"}"#)
        .with_origin("anthropic")
        .with_provider_data(allama::ProviderData::new().with_raw(
            "anthropic",
            "type",
            json!("server_tool_use"),
        ));
    let result = allama::ProviderToolResult::new(
        "srvtoolu_1",
        json!([{"type": "web_search_result", "url": "https://rust-lang.org"}]),
    )
    .with_origin("anthropic")
    .with_provider_data(allama::ProviderData::new().with_raw(
        "anthropic",
        "type",
        json!("web_search_tool_result"),
    ));
    let foreign = allama::ProviderToolCall::new("ws_1", "web_search", "{}").with_origin("openai");
    let request = ChatRequest::new("claude-haiku-4-5")
        .user("Latest Rust release?")
        .message(Message::Assistant(allama::AssistantMessage::new(vec![
            call.into(),
            result.into(),
            foreign.into(),
            ContentBlock::text("Rust 1.99."),
        ])))
        .user("Thanks");
    let (body, warnings) = encode(&request);
    assert_eq!(
        body["messages"][1]["content"][0],
        json!({"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "rust"}})
    );
    assert_eq!(
        body["messages"][1]["content"][1],
        json!({"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [{"type": "web_search_result", "url": "https://rust-lang.org"}]})
    );
    assert_eq!(body["messages"][1]["content"][2]["type"], "text");
    assert_eq!(kinds(&warnings), vec![WarningKind::Dropped]);
}

fn anthropic_call(id: &str, name: &str) -> allama::ProviderToolCall {
    allama::ProviderToolCall::new(id, name, "{}").with_origin("anthropic")
}

/// A result as a hand-built or older history holds it: no wire `type`.
fn untyped_result(id: &str) -> allama::ProviderToolResult {
    allama::ProviderToolResult::new(id, json!({"type": "opaque"})).with_origin("anthropic")
}

fn assistant_turn(blocks: Vec<ContentBlock>) -> Message {
    Message::Assistant(allama::AssistantMessage::new(blocks))
}

fn sent_types(body: &Value) -> Vec<String> {
    body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|message| message["content"].as_array().unwrap())
        .map(|block| block["type"].as_str().unwrap().to_owned())
        .collect()
}

#[test]
fn an_untyped_provider_tool_result_takes_the_type_its_call_fixes() {
    let mcp_call = anthropic_call("mcptoolu_1", "echo").with_provider_data(
        ProviderData::new().with_raw("anthropic", "type", json!("mcp_tool_use")),
    );
    let request = ChatRequest::new(HAIKU)
        .user("Fetch it, then echo it")
        .message(assistant_turn(vec![
            anthropic_call("srvtoolu_1", "web_fetch").into(),
            untyped_result("srvtoolu_1").into(),
            mcp_call.into(),
            untyped_result("mcptoolu_1").into(),
            // Programmatic tool calling: the call waits on a client tool,
            // and its result arrives in a later response.
            anthropic_call("srvtoolu_2", "code_execution").into(),
            ContentBlock::tool_call("toolu_1", "get_weather", "{}"),
        ]))
        .message(Message::tool_result("toolu_1", "get_weather", "sunny"))
        .message(assistant_turn(vec![
            untyped_result("srvtoolu_2").into(),
            ContentBlock::text("Done."),
        ]))
        .user("Thanks");
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["messages"][1]["content"][1],
        json!({"type": "web_fetch_tool_result", "tool_use_id": "srvtoolu_1", "content": {"type": "opaque"}})
    );
    assert_eq!(body["messages"][1]["content"][3]["type"], "mcp_tool_result");
    assert_eq!(
        body["messages"][3]["content"][0],
        json!({"type": "code_execution_tool_result", "tool_use_id": "srvtoolu_2", "content": {"type": "opaque"}})
    );
}

#[test]
fn an_untyped_result_of_an_undocumented_tool_is_dropped_with_its_call() {
    for split in [false, true] {
        // The call and its result, in one turn or in two.
        let call: ContentBlock = anthropic_call("srvtoolu_1", "future_tool").into();
        let result: ContentBlock = untyped_result("srvtoolu_1").into();
        let mut request = ChatRequest::new(HAIKU).user("Go");
        request = if split {
            request
                .message(assistant_turn(vec![call, ContentBlock::text("Working.")]))
                .user("Go on")
                .message(assistant_turn(vec![result, ContentBlock::text("Done.")]))
        } else {
            request.message(assistant_turn(vec![
                call,
                result,
                ContentBlock::text("Done."),
            ]))
        };
        let (body, warnings) = encode(&request.user("Thanks"));
        let types = sent_types(&body);
        assert!(!types.contains(&"server_tool_use".to_owned()), "{types:?}");
        assert!(
            !types.iter().any(|t| t.ends_with("_tool_result")),
            "{types:?}"
        );

        // One warning for the pair, which does not call it foreign.
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        let warning = &warnings[0];
        assert_eq!(warning.kind, WarningKind::Dropped);
        assert_eq!(warning.code.as_deref(), Some("dropped.content_replay"));
        assert!(
            warning.message.contains("`future_tool`"),
            "{}",
            warning.message
        );
        assert!(
            warning.message.contains("`srvtoolu_1`"),
            "{}",
            warning.message
        );
        assert!(!warning.message.contains("retain_provider_state"));
        let expected = if split {
            Subject::Messages {
                messages: vec![1, 3],
            }
        } else {
            Subject::Content {
                message: 1,
                block: Some(0),
            }
        };
        assert_eq!(warning.subject, expected);
    }
}

#[test]
fn an_untyped_result_without_a_call_is_dropped() {
    let request = ChatRequest::new(HAIKU)
        .user("Go")
        .message(assistant_turn(vec![
            untyped_result("srvtoolu_1").into(),
            ContentBlock::text("Done."),
        ]))
        .user("Thanks");
    let (body, warnings) = encode(&request);
    assert_eq!(sent_types(&body), vec!["text", "text", "text"]);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].code.as_deref(), Some("dropped.content_replay"));
    assert_eq!(
        warnings[0].subject,
        Subject::Content {
            message: 1,
            block: Some(0)
        }
    );
}

/// A paused turn (`pause_turn`), or one waiting on client tools, ends with
/// a call whose result has not arrived: it is sent as Anthropic returned
/// it, whatever the tool.
#[test]
fn a_provider_tool_call_without_a_result_is_sent() {
    for name in ["web_search", "future_tool"] {
        let request = ChatRequest::new(HAIKU)
            .user("Go")
            .message(assistant_turn(vec![
                anthropic_call("srvtoolu_1", name).into(),
            ]));
        let (body, warnings) = encode(&request);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(
            body["messages"][1]["content"],
            json!([{"type": "server_tool_use", "id": "srvtoolu_1", "name": name, "input": {}}])
        );
    }
}

/// Another provider's half of a pair is foreign content; Anthropic's half
/// goes with it, reported as its own content it cannot take back.
#[test]
fn a_provider_tool_pair_with_a_foreign_half_is_dropped_whole() {
    let foreign_result =
        allama::ProviderToolResult::new("srvtoolu_1", json!({})).with_origin("openai");
    let foreign_call =
        allama::ProviderToolCall::new("srvtoolu_2", "web_search", "{}").with_origin("openai");
    let request = ChatRequest::new(HAIKU)
        .user("Go")
        .message(assistant_turn(vec![
            anthropic_call("srvtoolu_1", "web_search").into(),
            foreign_result.into(),
            foreign_call.into(),
            untyped_result("srvtoolu_2").into(),
            ContentBlock::text("Done."),
        ]))
        .user("Thanks");
    let (body, warnings) = encode(&request);
    assert_eq!(sent_types(&body), vec!["text", "text", "text"]);
    let mut codes: Vec<_> = warnings.iter().filter_map(|w| w.code.as_deref()).collect();
    codes.sort_unstable();
    assert_eq!(
        codes,
        vec!["dropped.content_replay", "dropped.foreign_content"],
        "{warnings:?}"
    );
}

/// Replay state stays in a stored conversation, so a conversation that
/// moved here from another provider drops it on every turn: each drop is
/// coded, so its warning event goes out at `WARN` once per model, not once
/// per turn, and points to `retain_provider_state`.
#[test]
fn a_long_foreign_reasoning_history_drops_with_codes() {
    let mut request = ChatRequest::new(HAIKU);
    for turn in 0..10 {
        request = request
            .user(format!("question {turn}"))
            .message(Message::assistant_blocks(vec![
                ContentBlock::Reasoning(Reasoning::new("thoughts").with_provider_data(
                    ProviderData::new().with_raw("openai", "signature", json!("opaque")),
                )),
                ContentBlock::text("answer"),
            ]));
    }
    let (_, warnings) = encode(&request.user("next"));
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
fn extra_fields_are_merged_into_the_body() {
    let options = ChatRequestOptions::new()
        .extra("container", "container_123")
        .extra("context_management", json!({"edits": []}));
    let (body, warnings) = encode(&ChatRequest::new(HAIKU).user("q").with_option(options));
    assert_eq!(body["container"], "container_123");
    assert_eq!(body["context_management"], json!({"edits": []}));
    assert_eq!(body["model"], HAIKU);
    assert!(warnings.is_empty(), "{warnings:?}");

    // The JSON form nests them under `extra`.
    let (body, _) = encode(
        &ChatRequest::new(HAIKU)
            .user("q")
            .with_options_json("anthropic", json!({"extra": {"inference_geo": "us"}})),
    );
    assert_eq!(body["inference_geo"], "us");
}

#[test]
fn extra_cannot_set_fields_the_adapter_owns() {
    for key in RESERVED_FIELDS.iter().copied().chain(["Model", "STREAM"]) {
        let request = ChatRequest::new(HAIKU)
            .user("q")
            .with_option(ChatRequestOptions::new().extra(key, "x"));
        let err = encode_err(&request);
        assert!(
            matches!(err, Error::InvalidRequest { .. }),
            "{key}: {err:?}"
        );
        assert!(err.to_string().contains(key), "{key}: {err}");
    }
}

#[test]
fn every_body_field_is_reserved() {
    // A field added to the body must be added to `RESERVED_FIELDS`, or
    // `extra` could send it twice.
    let options = ChatRequestOptions::new()
        .metadata(ExtraMap::from_iter([("user_id".into(), json!("u"))]))
        .service_tier("auto");
    let request = ChatRequest::new(HAIKU)
        .system("s")
        .user("q")
        .temperature(0.5)
        .top_p(0.9)
        .top_k(1)
        .stop(["x"])
        .tools([ToolDefinition::new("f", json!({"type": "object"}))])
        .tool_choice(allama::ToolChoice::Auto)
        .response_format(allama::ResponseFormat::json_schema(json!({
            "type": "object", "properties": {}, "additionalProperties": false
        })))
        .cache(PromptCache::automatic())
        .with_option(options);
    let encoded = encode_request(&request, true).unwrap();
    let body = serde_json::to_value(&encoded.body).unwrap();
    for key in body.as_object().unwrap().keys() {
        assert!(
            RESERVED_FIELDS.contains(&key.as_str()),
            "{key} is not reserved"
        );
    }
}

// --- Files, documents and provider data ---

/// An Anthropic Files API id is sent as a `file` source, on images and
/// documents alike.
#[test]
fn anthropic_provider_files_map_to_file_sources() {
    let request = ChatRequest::new(HAIKU).user(vec![
        ContentPart::Image(Image::provider_file(NAMESPACE, "file_img")),
        ContentPart::File(File::provider_file(
            "application/pdf",
            NAMESPACE,
            "file_pdf",
        )),
        ContentPart::File(File::provider_file("text/plain", NAMESPACE, "file_txt")),
    ]);
    let (body, warnings) = encode(&request);
    let content = &body["messages"][0]["content"];
    assert_eq!(
        content[0],
        json!({"type": "image", "source": {"type": "file", "file_id": "file_img"}})
    );
    assert_eq!(
        content[1],
        json!({"type": "document", "source": {"type": "file", "file_id": "file_pdf"}})
    );
    assert_eq!(content[2]["source"]["file_id"], "file_txt");
    assert!(warnings.is_empty(), "{warnings:?}");
}

/// A foreign file id never reaches the encoder through `prepare`; the
/// encoder refuses one too.
#[test]
fn foreign_provider_files_are_refused() {
    let request = ChatRequest::new(HAIKU).user(vec![ContentPart::Image(Image::provider_file(
        "openai", "file-1",
    ))]);
    assert!(matches!(encode_err(&request), Error::Unsupported { .. }));
}

/// The file's own provider data reaches the document block, so
/// document citations can be turned on.
#[test]
fn file_provider_data_enables_document_citations() {
    let file = File::base64("application/pdf", "JVBE")
        .with_filename("report.pdf")
        .with_provider_data(
            ProviderData::new()
                .with_raw(NAMESPACE, "citations", json!({"enabled": true}))
                .with_raw(NAMESPACE, "context", json!("Q3 figures")),
        );
    let (body, warnings) = encode(&ChatRequest::new(HAIKU).user(vec![
        ContentPart::File(file),
        ContentPart::text("Summarize, with citations."),
    ]));
    assert_eq!(
        body["messages"][0]["content"][0],
        json!({
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBE"},
            "title": "report.pdf",
            "citations": {"enabled": true},
            "context": "Q3 figures"
        })
    );
    assert!(warnings.is_empty(), "{warnings:?}");
}

#[test]
fn file_provider_data_cannot_set_document_fields_the_adapter_owns() {
    let file = File::base64("application/pdf", "JVBE")
        .with_provider_data(ProviderData::new().with_raw(NAMESPACE, "source", json!({})));
    let err = encode_err(&ChatRequest::new(HAIKU).user(vec![ContentPart::File(file)]));
    assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
    assert!(err.message().contains("`source`"), "{err}");
}

/// Tool provider data cannot send a key the definition already
/// sets, which produced `{"name":"f",…,"name":"g"}`.
#[test]
fn tool_provider_data_cannot_set_fields_the_adapter_owns() {
    for key in [
        "name",
        "input_schema",
        "strict",
        "description",
        "cache_control",
    ] {
        let mut tool = ToolDefinition::new("f", json!({"type": "object"}));
        tool.provider_data = ProviderData::new().with_raw(NAMESPACE, key, json!("g"));
        let err = encode_err(&ChatRequest::new(HAIKU).user("q").tools([tool]));
        assert!(
            matches!(err, Error::InvalidRequest { .. }),
            "{key}: {err:?}"
        );
    }
    let mut tool = ToolDefinition::new("f", json!({"type": "object"}));
    tool.provider_data = ProviderData::new().with_raw(NAMESPACE, "defer_loading", json!(true));
    let (body, _) = encode(&ChatRequest::new(HAIKU).user("q").tools([tool]));
    assert_eq!(body["tools"][0]["defer_loading"], true);
}

/// A forced tool choice with no tool to call is a contradiction
/// the API refuses; it is refused before sending.
#[test]
fn forced_tool_choice_without_tools_is_invalid() {
    for choice in [
        allama::ToolChoice::Required,
        allama::ToolChoice::Specific { name: "f".into() },
    ] {
        let err = encode_err(&ChatRequest::new(HAIKU).user("q").tool_choice(choice));
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
    }
}

/// A media type with parameters is the media type.
#[test]
fn media_type_parameters_are_ignored() {
    let request = ChatRequest::new(HAIKU).user(vec![
        ContentPart::file_base64("text/plain; charset=utf-8", "aGk="),
        ContentPart::image_base64("image/png; name=x.png", "iVBORw=="),
    ]);
    let (body, warnings) = encode(&request);
    let content = &body["messages"][0]["content"];
    assert_eq!(
        content[0]["source"],
        json!({"type": "text", "media_type": "text/plain", "data": "hi"})
    );
    assert_eq!(content[1]["source"]["media_type"], "image/png");
    assert!(warnings.is_empty(), "{warnings:?}");
}

/// Provider data in another provider's namespace is reported as
/// dropped, on messages, parts and blocks alike.
#[test]
fn foreign_provider_data_is_reported_as_dropped() {
    let foreign = || ProviderData::new().with_raw("openai", "annotations", json!([]));
    let mut user = Message::user(vec![ContentPart::Text(
        Text::new("q").with_provider_data(foreign()),
    )]);
    if let Message::User(user) = &mut user {
        user.provider_data = foreign();
    }
    let request = ChatRequest::new(HAIKU)
        .message(user)
        .message(Message::assistant_blocks(vec![ContentBlock::Text(
            Text::new("a").with_provider_data(foreign()),
        )]))
        .user("again");
    let (_, warnings) = encode(&request);
    assert!(!warnings.is_empty());
    for warning in &warnings {
        assert_eq!(warning.kind, WarningKind::Dropped);
        assert_eq!(
            warning.code.as_deref(),
            Some("dropped.foreign_provider_data")
        );
    }
}

/// `max` is reachable on models that also have `xhigh`,
/// through the effort option, which replaces only the level.
#[test]
fn effort_option_replaces_the_mapped_level() {
    let request = ChatRequest::new("claude-opus-4-8")
        .user("q")
        .reasoning(ReasoningEffort::XHigh)
        .with_option(ChatRequestOptions::new().effort(crate::EffortLevel::Max));
    let (body, warnings) = encode(&request);
    assert_eq!(body["output_config"], json!({"effort": "max"}));
    assert_eq!(body["thinking"], json!({"type": "adaptive"}));
    assert!(warnings.is_empty(), "{warnings:?}");

    // Alone, it sets the effort and leaves thinking at the model default.
    let request = ChatRequest::new("claude-opus-4-8")
        .user("q")
        .with_option(ChatRequestOptions::new().effort(crate::EffortLevel::Low));
    let (body, _) = encode(&request);
    assert_eq!(body["output_config"], json!({"effort": "low"}));
    assert!(body.get("thinking").is_none());

    // The JSON form names the level as the API does.
    let json = ChatRequest::new("claude-opus-4-8")
        .user("q")
        .with_options_json("anthropic", json!({"effort": "xhigh"}));
    assert_eq!(encode(&json).0["output_config"]["effort"], "xhigh");
}
