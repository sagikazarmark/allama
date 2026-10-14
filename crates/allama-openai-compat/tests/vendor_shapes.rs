//! Response shapes of OpenAI-compatible endpoints that differ from OpenAI's
//! (`fixtures/vendor_shapes`, see its README for the sources): parallel tool
//! calls without `index`, Gemini thought signatures, Mistral `thinking`
//! parts, OpenRouter `reasoning_details` and usage counters, reasoning
//! after text, tool calls without ids, and in-band errors in their several
//! shapes.
//!
//! Regenerate the expected files with `ALLAMA_UPDATE_FIXTURES=1` and review
//! the diff.

use std::path::PathBuf;

use allama::{
    ChatProvider, ChatRequest, ChatResponse, ChatStreamExt, ContentBlock, Error, FinishReason,
    Message, StreamErrorKind, ToolCall, WarningKind,
};
use allama_conformance::{
    FixtureDir, assert_response_fixture_eq, assert_stream_fixture_eq, load_text_fixture,
};
use allama_http::{ReplayResponse, ReplayTransport};
use allama_openai_compat::{CompatQuirks, Provider, ReasoningReplayField};
use serde_json::{Value, json};

const NAMESPACE: &str = "openai-compat:vendor";

fn fixtures() -> FixtureDir {
    FixtureDir::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/vendor_shapes"))
}

fn fixture(name: &str) -> String {
    load_text_fixture(&fixtures(), name)
}

fn provider(quirks: CompatQuirks, responses: Vec<ReplayResponse>) -> (Provider, ReplayTransport) {
    let transport = ReplayTransport::new(responses);
    let provider = Provider::builder()
        .base_url("http://upstream/v1")
        .provider_name("vendor")
        .quirks(quirks)
        .http_transport(transport.clone())
        .build()
        .unwrap();
    (provider, transport)
}

fn quirks() -> CompatQuirks {
    CompatQuirks::new(NAMESPACE)
}

async fn streamed(name: &str) -> allama::Result<ChatResponse> {
    let (provider, _) = provider(quirks(), vec![ReplayResponse::sse(fixture(name))]);
    provider
        .chat_stream(&ChatRequest::new("m").user("q"))
        .await?
        .collect_response()
        .await
}

async fn unary(name: &str) -> allama::Result<ChatResponse> {
    let (provider, _) = provider(quirks(), vec![ReplayResponse::json(200, fixture(name))]);
    provider.chat(&ChatRequest::new("m").user("q")).await
}

const STREAMS: &[&str] = &[
    "copilot_reasoning_opaque",
    "gemini_parallel_no_index",
    "gemini3_thought_signature",
    "mistral_thinking",
    "openrouter_reasoning_details",
    "reasoning_after_text",
];

const UNARY: &[&str] = &[
    "copilot_reasoning_opaque",
    "mistral_thinking",
    "openrouter_reasoning_details",
    "openrouter_usage_cost",
    "tool_call_without_id",
];

#[tokio::test]
async fn shapes_match_fixtures() {
    for name in STREAMS {
        let (provider, _) = provider(
            quirks(),
            vec![ReplayResponse::sse(fixture(&format!("{name}.sse")))],
        );
        let stream = provider
            .chat_stream(&ChatRequest::new("m").user("q"))
            .await
            .unwrap();
        assert_stream_fixture_eq(
            stream,
            &fixtures(),
            &format!("{name}.events.json"),
            &format!("{name}.response.json"),
        )
        .await;
    }
    for name in UNARY {
        let response = unary(&format!("{name}.json")).await.unwrap();
        assert_response_fixture_eq(&response, &fixtures(), &format!("{name}.expected.json"));
    }
}

fn calls(response: &ChatResponse) -> Vec<(&str, &str, &str)> {
    response
        .tool_calls()
        .map(|call| {
            (
                call.id.as_str(),
                call.name.as_str(),
                call.arguments.as_str(),
            )
        })
        .collect()
}

/// Gemini's OpenAI-compatible endpoint sends parallel calls
/// without `index` (recorded); each id starts a call of its own instead of
/// merging into one invalid call.
#[tokio::test]
async fn parallel_calls_without_index_stay_apart() {
    let response = streamed("gemini_parallel_no_index.sse").await.unwrap();
    assert_eq!(
        calls(&response),
        [
            (
                "function-call-7709560096072903355",
                "get_weather",
                r#"{"city":"Paris"}"#
            ),
            (
                "function-call-7709560096072902848",
                "get_weather",
                r#"{"city":"Tokyo"}"#
            ),
        ]
    );
    assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));

    // The same without `index`, sent fragment by fragment: fragments
    // without an id continue the last call.
    let sse = [
        r#"{"id":"r","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{\"x\""}}]}}]}"#,
        r#"{"id":"r","choices":[{"index":0,"delta":{"tool_calls":[{"function":{"arguments":":1}"}}]}}]}"#,
        r#"{"id":"r","choices":[{"index":0,"delta":{"tool_calls":[{"id":"b","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}"#,
        r#"{"id":"r","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c","type":"function","function":{"name":"g","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#,
        "[DONE]",
    ]
    .map(|line| format!("data: {line}\n\n"))
    .concat();
    let (provider, _) = provider(quirks(), vec![ReplayResponse::sse(sse)]);
    let response = provider
        .chat_stream(&ChatRequest::new("m").user("q"))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();
    assert_eq!(
        calls(&response),
        [("a", "f", r#"{"x":1}"#), ("b", "f", "{}"), ("c", "g", "{}")]
    );
}

/// Gemini 3's thought signature (`extra_content`, recorded)
/// stays on the call and goes back with it; the turn ends `stop` with calls,
/// which is `ToolCalls`.
#[tokio::test]
async fn thought_signatures_round_trip_and_stop_with_calls_is_tool_calls() {
    let response = streamed("gemini3_thought_signature.sse").await.unwrap();
    assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
    assert_eq!(response.finish_reason_raw.as_deref(), Some("stop"));
    let first = response.first_tool_call().unwrap();
    let signature = first
        .provider_data
        .get_raw(NAMESPACE, "extra_content")
        .and_then(|extra| extra.pointer("/google/thought_signature"))
        .and_then(Value::as_str)
        .unwrap();
    assert!(signature.starts_with("Eu4ECusE"), "{signature}");
    assert_eq!(response.tool_calls().count(), 2);

    // The next turn sends it back, on the call it came with.
    let (provider, transport) = provider(
        quirks(),
        vec![ReplayResponse::json(
            200,
            r#"{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#,
        )],
    );
    let mut request = ChatRequest::new("m").user("q");
    request.push_message(response.to_assistant_message());
    for call in response.tool_calls() {
        request.push_message(Message::tool_result(
            call.id.clone(),
            call.name.clone(),
            "sunny",
        ));
    }
    let replayed = provider.chat(&request).await.unwrap();
    assert!(replayed.warnings.is_empty(), "{:?}", replayed.warnings);
    let body: Value = serde_json::from_str(transport.requests()[0].body_text().unwrap()).unwrap();
    let calls = &body["messages"][1]["tool_calls"];
    assert_eq!(
        calls[0]["extra_content"]["google"]["thought_signature"],
        signature
    );
    assert!(calls[1].get("extra_content").is_none(), "{calls}");

    // Another provider's tool call carries nothing of this one's.
    let foreign = ToolCall::new("x", "f", "{}").with_provider_data(
        allama::ProviderData::new().with_raw("openai", "item_id", json!("fc_1")),
    );
    let mut request = ChatRequest::new("m").user("q");
    request.push_message(Message::assistant_blocks(vec![foreign.into()]));
    request.push_message(Message::tool_result("x", "f", "r"));
    let validation = provider.validate_chat_request(&request).unwrap();
    assert!(
        validation
            .warnings()
            .iter()
            .all(|w| w.kind == WarningKind::Dropped)
    );
}

/// Mistral's `thinking` parts are reasoning, in place, in both
/// modes; streamed array content no longer fails the stream.
#[tokio::test]
async fn mistral_thinking_parts_are_reasoning() {
    for response in [
        streamed("mistral_thinking.sse").await.unwrap(),
        unary("mistral_thinking.json").await.unwrap(),
    ] {
        assert_eq!(
            response.reasoning_text().as_deref(),
            Some("The user asks for a greeting.")
        );
        assert_eq!(response.text().as_deref(), Some("Hello there!"));
        assert!(matches!(response.content[0], ContentBlock::Reasoning(_)));
        assert_eq!(
            response.content[0]
                .provider_data()
                .get_raw(NAMESPACE, "reasoning_field"),
            Some(&json!("thinking"))
        );
        assert!(response.warnings.is_empty(), "{:?}", response.warnings);
    }
}

/// OpenRouter's `reasoning_details` are kept on the
/// reasoning block (fragments merged when streamed) and go back as they
/// came under `ReasoningReplayField::ReasoningDetails`.
#[tokio::test]
async fn reasoning_details_are_kept_and_replayed() {
    let streamed = streamed("openrouter_reasoning_details.sse").await.unwrap();
    let details = streamed.content[0]
        .provider_data()
        .get_raw(NAMESPACE, "reasoning_details")
        .unwrap()
        .clone();
    assert_eq!(
        details,
        json!([{"type": "reasoning.text", "text": "Weather needs a tool.",
                "signature": "EqQBCkgIBxABGAIiQK", "format": "anthropic-claude-v1", "index": 0}])
    );
    assert_eq!(
        streamed.reasoning_text().as_deref(),
        Some("Weather needs a tool.")
    );

    let unary = unary("openrouter_reasoning_details.json").await.unwrap();
    assert_eq!(unary.finish_reason, Some(FinishReason::ToolCalls));
    let mut replay = quirks();
    replay.reasoning_replay = ReasoningReplayField::ReasoningDetails;
    let (provider, transport) = provider(
        replay,
        vec![ReplayResponse::json(
            200,
            r#"{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#,
        )],
    );
    let mut request = ChatRequest::new("m").user("q");
    request.push_message(unary.to_assistant_message());
    request.push_message(Message::tool_result(
        "tool_get_weather_1",
        "get_weather",
        "sunny",
    ));
    // Reasoning without details (written by hand) cannot be replayed.
    request.push_message(Message::user("and?"));
    request.push_message(Message::assistant_blocks(vec![ContentBlock::reasoning(
        "hand-written",
    )]));
    request.push_message(Message::user("go on"));
    let response = provider.chat(&request).await.unwrap();
    let body: Value = serde_json::from_str(transport.requests()[0].body_text().unwrap()).unwrap();
    assert_eq!(
        body["messages"][1]["reasoning_details"],
        json!([
            {"type": "reasoning.text", "text": "Looking up the weather.", "format": "google-gemini-v1", "index": 0},
            {"type": "reasoning.encrypted", "data": "CiQB0e2Kb1x", "id": "tool_get_weather_1", "format": "google-gemini-v1", "index": 1}
        ])
    );
    assert!(body["messages"][1].get("reasoning").is_none());
    let codes: Vec<_> = response
        .warnings
        .iter()
        .filter_map(|w| w.code.as_deref())
        .collect();
    assert_eq!(
        codes,
        ["dropped.reasoning_replay"],
        "{:?}",
        response.warnings
    );
}

/// GitHub Copilot's `reasoning_text` is reasoning, with its
/// `reasoning_opaque` signature on the block. A signature that arrives
/// after the reasoning closed (Copilot sends a new one before each tool
/// call) is a block of its own, in place. Under
/// `ReasoningReplayField::ReasoningText` the turn's text goes back joined,
/// with its last signature; reasoning without a signature does not.
#[tokio::test]
async fn reasoning_text_and_signature_are_kept_and_replayed() {
    let signature = |block: &ContentBlock| {
        block
            .provider_data()
            .get_raw(NAMESPACE, "reasoning_opaque")
            .cloned()
    };
    let streamed = streamed("copilot_reasoning_opaque.sse").await.unwrap();
    assert_eq!(streamed.content.len(), 4, "{:?}", streamed.content);
    assert!(matches!(&streamed.content[0], ContentBlock::Reasoning(r)
        if r.text == "The user wants the weather in Paris."));
    assert_eq!(signature(&streamed.content[0]), Some(json!("c2lnLTE=")));
    assert_eq!(streamed.content[1].as_text(), Some("Let me check."));
    assert!(matches!(&streamed.content[2], ContentBlock::Reasoning(r) if r.is_opaque()));
    assert_eq!(signature(&streamed.content[2]), Some(json!("c2lnLTI=")));
    assert!(matches!(streamed.content[3], ContentBlock::ToolCall(_)));
    assert_eq!(streamed.finish_reason, Some(FinishReason::ToolCalls));

    let unary = unary("copilot_reasoning_opaque.json").await.unwrap();
    assert_eq!(
        unary.reasoning_text().as_deref(),
        Some("The user wants the weather in Paris.")
    );
    assert_eq!(signature(&unary.content[0]), Some(json!("c2lnLTE=")));
    // Modeled, so not kept again as unmodeled message fields.
    assert!(
        unary
            .provider_data
            .get_raw(NAMESPACE, "reasoning_opaque")
            .is_none()
    );

    let mut replay = quirks();
    replay.reasoning_replay = ReasoningReplayField::ReasoningText;
    let (provider, transport) = provider(
        replay,
        vec![ReplayResponse::json(
            200,
            r#"{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#,
        )],
    );
    let mut request = ChatRequest::new("m").user("q");
    request.push_message(streamed.to_assistant_message());
    request.push_message(Message::tool_result("toolu_01", "get_weather", "sunny"));
    // Reasoning without a signature (written by hand) cannot be replayed.
    request.push_message(Message::user("and?"));
    request.push_message(Message::assistant_blocks(vec![ContentBlock::reasoning(
        "hand-written",
    )]));
    request.push_message(Message::user("go on"));
    let response = provider.chat(&request).await.unwrap();
    let body: Value = serde_json::from_str(transport.requests()[0].body_text().unwrap()).unwrap();
    let assistant = &body["messages"][1];
    assert_eq!(
        assistant["reasoning_text"],
        "The user wants the weather in Paris."
    );
    assert_eq!(assistant["reasoning_opaque"], "c2lnLTI=");
    assert_eq!(assistant["content"], "Let me check.");
    let hand_written = &body["messages"][4];
    assert!(
        hand_written.get("reasoning_text").is_none(),
        "{hand_written}"
    );
    let codes: Vec<_> = response
        .warnings
        .iter()
        .filter_map(|w| w.code.as_deref())
        .collect();
    assert_eq!(
        codes,
        ["dropped.reasoning_replay"],
        "{:?}",
        response.warnings
    );
}

/// Reasoning the endpoint returned is dropped without a
/// warning when the endpoint takes none back; other reasoning still warns.
#[tokio::test]
async fn own_reasoning_drops_silently() {
    let streamed = streamed("reasoning_after_text.sse").await.unwrap();
    let (provider, _) = provider(quirks(), Vec::new());
    let mut request = ChatRequest::new("m").user("q");
    request.push_message(streamed.to_assistant_message());
    request.push_message(Message::user("again"));
    let validation = provider.validate_chat_request(&request).unwrap();
    assert!(
        validation.warnings().is_empty(),
        "{:?}",
        validation.warnings()
    );

    request.push_message(Message::assistant_blocks(vec![ContentBlock::reasoning(
        "from nowhere",
    )]));
    request.push_message(Message::user("more"));
    let validation = provider.validate_chat_request(&request).unwrap();
    assert_eq!(
        validation.warnings().len(),
        1,
        "{:?}",
        validation.warnings()
    );
}

/// Reasoning that resumes after text keeps its place.
#[tokio::test]
async fn reasoning_after_text_keeps_its_order() {
    let response = streamed("reasoning_after_text.sse").await.unwrap();
    let kinds: Vec<_> = response
        .content
        .iter()
        .map(|block| match block {
            ContentBlock::Reasoning(r) => format!("reasoning:{}", r.text),
            ContentBlock::Text(t) => format!("text:{}", t.text),
            other => format!("{other:?}"),
        })
        .collect();
    assert_eq!(
        kinds,
        [
            "reasoning:First thought.",
            "text:Partial answer.",
            "reasoning:Second thought.",
            "text:Final answer."
        ]
    );
}

/// Vendor usage counters and choice fields are provider
/// data; a `null` counter decodes as zero.
#[tokio::test]
async fn vendor_usage_and_choice_fields_are_provider_data() {
    let response = unary("openrouter_usage_cost.json").await.unwrap();
    let data = response.provider_data.namespace(NAMESPACE).unwrap();
    assert_eq!(
        data["usage"],
        json!({"cost": 0.000_002_1, "is_byok": false})
    );
    assert_eq!(data["native_finish_reason"], "stop");
    assert_eq!(data["logprobs"]["content"][0]["token"], "Hi");
    assert_eq!(data["provider"], "OpenAI");
    let usage = response.usage.unwrap();
    assert_eq!(usage.output_tokens, Some(0));
    assert!(usage.tool_requests.is_empty());

    let streamed = streamed("openrouter_reasoning_details.sse").await.unwrap();
    let data = streamed.provider_data.namespace(NAMESPACE).unwrap();
    assert_eq!(data["usage"]["cost"], json!(0.00096));
    assert_eq!(data["native_finish_reason"], "tool_use");
}

/// A unary tool call without an id (or with an empty one) gets one
/// made up, as streamed calls do, instead of failing the decode.
#[tokio::test]
async fn unary_tool_calls_without_ids_get_one() {
    let response = unary("tool_call_without_id.json").await.unwrap();
    let ids: Vec<_> = response.tool_calls().map(|call| call.id.clone()).collect();
    assert_eq!(ids, ["call_chatcmpl_x9_0", "call_chatcmpl_x9_1"]);
}

fn error_stream(lines: &[&str]) -> ReplayResponse {
    ReplayResponse::sse(
        lines
            .iter()
            .map(|line| format!("data: {line}\n\n"))
            .collect::<String>(),
    )
}

async fn stream_error(lines: &[&str]) -> Error {
    let (provider, _) = provider(quirks(), vec![error_stream(lines)]);
    provider
        .chat_stream(&ChatRequest::new("m").user("q"))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap_err()
}

const START: &str = r#"{"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"x"}}]}"#;

/// In-band stream errors in their other shapes are typed:
/// a numeric code is a status, a bare string and vLLM's legacy envelope are
/// errors, not decode failures or a successful partial response.
#[tokio::test]
async fn in_band_stream_errors_in_every_shape_are_typed() {
    let err = stream_error(&[
        START,
        r#"{"error":{"message":"Rate limit exceeded","code":429}}"#,
    ])
    .await;
    assert!(matches!(err, Error::RateLimited { .. }), "{err:?}");
    assert_eq!(err.upstream_message(), Some("Rate limit exceeded"));

    let err = stream_error(&[START, r#"{"error":"upstream connection reset"}"#]).await;
    assert!(matches!(err, Error::Provider { .. }), "{err:?}");
    assert_eq!(err.upstream_message(), Some("upstream connection reset"));

    let err = stream_error(&[
        START,
        r#"{"object":"error","message":"This model's maximum context length is 4096 tokens.","type":"BadRequestError","param":null,"code":400}"#,
        "[DONE]",
    ])
    .await;
    assert!(
        matches!(
            err,
            Error::ContextLengthExceeded { .. } | Error::InvalidRequest { .. }
        ),
        "{err:?}"
    );
    assert!(!err.is_retryable());

    // OpenRouter ends a failed stream with an error and `finish_reason: "error"`.
    let err = stream_error(&[
        START,
        r#"{"id":"c1","object":"chat.completion.chunk","error":{"code":502,"message":"Provider returned error"},"choices":[{"index":0,"delta":{"content":""},"finish_reason":"error"}]}"#,
    ])
    .await;
    assert!(matches!(err, Error::Provider { .. }), "{err:?}");
    assert!(err.is_retryable());

    // `[DONE]` with nothing before it is no response, not a stream that
    // missed its `[DONE]`.
    let err = stream_error(&["[DONE]"]).await;
    assert!(
        matches!(
            err,
            Error::Stream {
                kind: StreamErrorKind::Incomplete,
                ..
            }
        ),
        "{err:?}"
    );
    assert!(err.to_string().contains("before any chunk"), "{err}");
}

/// An HTTP 200 whose body is an error envelope (OpenRouter) is
/// that error, not a malformed response; Gemini's array envelope is read.
#[tokio::test]
async fn error_bodies_with_a_success_status_are_typed() {
    let (provider, _) = provider(
        quirks(),
        vec![ReplayResponse::json(
            200,
            r#"{"error":{"message":"Provider returned error","code":502,"metadata":{"provider_name":"X"}},"user_id":"u"}"#,
        )],
    );
    let err = provider
        .chat(&ChatRequest::new("m").user("q"))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Provider { .. }), "{err:?}");
    assert!(err.is_retryable());
    assert_eq!(err.upstream_message(), Some("Provider returned error"));

    let gemini = quirks().map_http_error(
        429,
        &allama_http::Headers::new(),
        r#"[{"error":{"code":429,"message":"Resource has been exhausted (e.g. check quota).","status":"RESOURCE_EXHAUSTED"}}]"#,
    );
    assert_eq!(
        gemini.upstream_message(),
        Some("Resource has been exhausted (e.g. check quota).")
    );
}

/// Gemini's OpenAI-compatible endpoint wraps its
/// error envelope in an array on error statuses too; the call reads it.
#[tokio::test]
async fn array_error_envelopes_on_error_statuses_are_typed() {
    let (provider, _) = provider(
        quirks(),
        vec![ReplayResponse::json(
            400,
            r#"[{"error":{"code":400,"message":"Invalid JSON payload received.","status":"INVALID_ARGUMENT"}}]"#,
        )],
    );
    let err = provider
        .chat(&ChatRequest::new("m").user("q"))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
    assert_eq!(err.status(), Some(400));
    assert_eq!(
        err.upstream_message(),
        Some("Invalid JSON payload received.")
    );
}
