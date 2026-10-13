//! Fixture-based conformance tests.
//!
//! `response_raw.json`, `stream.sse`, `response_cache_raw.json`,
//! `stream_cache.sse`, `response_server_tools_raw.json` and
//! `stream_server_tools.sse` are recorded from the live API
//! (claude-haiku-4-5-20251001). `response_computer_toolset_raw.json` is
//! written from the shapes in Anthropic's computer-use tool documentation
//! (<https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool>);
//! the gated `anthropic_computer_toolset_round_trip` e2e test checks them
//! live. Set `ALLAMA_UPDATE_FIXTURES=1` to rewrite the expected files, then
//! review every value before committing.

use std::path::PathBuf;

use allama::ChatProvider;
use allama::ValidationCoverage;
use allama::{
    CacheBreakpoint, ChatRequest, ChatResponse, ChatStream, ChatStreamExt, ContentBlock,
    ContentPart, Message, PromptCache, ReasoningConfig, ReasoningEffort, ReasoningOutput, Refusal,
    ResponseFormat, SystemPrompt, Text, ToolCall, ToolChoice, ToolDefinition,
};
use allama_conformance::capabilities::UNKNOWN_MODEL;
use allama_conformance::probe::ProbeOutcome;
use allama_conformance::stream_end::StreamEnds;
use allama_conformance::stream_errors::ErrorKindProbe;
use allama_conformance::{
    AdapterUnderTest, Contract, FixtureDir, TransportOutcome, assert_chat_adapter,
    assert_error_fixture_eq, assert_json_fixture_eq, assert_partial_stream_fixture_eq,
    assert_response_fixture_eq, assert_stream_finish_error_fixture_eq, assert_stream_fixture_eq,
    assert_usage_contract, load_json_fixture, load_text_fixture,
};
use allama_http::{CredentialPlacement, CredentialSlot, Headers, ReplayResponse, ReplayTransport};
use futures_util::StreamExt;
use serde_json::{Value, json};

use crate::ChatRequestOptions;

fn headers(pairs: &[(&str, &str)]) -> Headers {
    pairs.iter().copied().collect()
}

fn fixtures() -> FixtureDir {
    FixtureDir::new(fixtures_path())
}

fn fixtures_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

fn updating() -> bool {
    allama_conformance::update_fixtures()
}

fn write_fixture(name: &str, value: &Value) {
    let text = serde_json::to_string_pretty(value).unwrap() + "\n";
    std::fs::write(fixtures_path().join(name), text).unwrap();
}

fn decode_fixture(name: &str) -> ChatResponse {
    let raw = load_json_fixture(&fixtures(), name);
    crate::decode::conformance_from_api_response_value(raw).unwrap()
}

fn stream_fixture(name: &str) -> ChatStream {
    crate::conformance_stream_from_sse_text(&load_text_fixture(&fixtures(), name))
}

/// Regenerate stream fixtures (events, then the collected response).
async fn update_stream_fixtures(sse: &str, events_name: &str, response_name: Option<&str>) {
    let mut stream = stream_fixture(sse);
    let mut collector = allama::chat::stream::StreamCollector::new();
    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        events.push(match &item {
            Ok(event) => {
                collector.push_ref(event).unwrap();
                serde_json::to_value(event).unwrap()
            }
            Err(err) => json!({"error": serde_json::to_value(err.as_log()).unwrap()}),
        });
    }
    write_fixture(events_name, &Value::Array(events));
    if let Some(response_name) = response_name {
        let response = collector.finish().unwrap();
        write_fixture(response_name, &serde_json::to_value(&response).unwrap());
    }
}

fn request_fixture() -> ChatRequest {
    ChatRequest::new("claude-haiku-4-5-20251001")
        .system(
            SystemPrompt::new("You are a careful assistant.").with_cache(CacheBreakpoint::new()),
        )
        .message(Message::user("Find the answer"))
        .tools(vec![
            ToolDefinition::new(
                "search",
                json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}),
            )
            .description("Search docs")
            .strict(true),
        ])
        .tool_choice(ToolChoice::Auto)
        .parallel_tool_calls(false)
        .stop(["END"])
        .max_output_tokens(8192)
        .reasoning(ReasoningEffort::Medium)
        .cache(PromptCache::automatic())
        .with_option(
            ChatRequestOptions::new()
                .metadata(serde_json::Map::from_iter([(
                    "user_id".into(),
                    json!("user-anthropic-123"),
                )]))
                .service_tier("standard_only"),
        )
}

#[test]
fn request_fixture_matches() {
    let actual = crate::encode::conformance_to_api_request(&request_fixture(), false).unwrap();
    if updating() {
        write_fixture("request.json", &actual);
    }
    assert_json_fixture_eq(&actual, &fixtures(), "request.json");
}

/// Encode `request` and compare its body and warnings with fixture `name`.
fn assert_request_case(name: &str, request: &ChatRequest) {
    let encoded = crate::encode::encode_request(request, false).unwrap();
    let actual = json!({
        "body": serde_json::to_value(&encoded.body).unwrap(),
        "warnings": serde_json::to_value(&encoded.warnings).unwrap(),
    });
    if updating() {
        write_fixture(name, &actual);
    }
    assert_json_fixture_eq(&actual, &fixtures(), name);
}

/// Empty text, from an empty user message or part, or a replayed unary
/// `Text("")` or refusal, is never sent (confirmed live: 400 "text content
/// blocks must be non-empty").
#[test]
fn request_without_empty_text_fixture_matches() {
    let request = ChatRequest::new("claude-haiku-4-5-20251001")
        .user("What is the weather in Paris?")
        .message(Message::assistant_blocks(vec![
            ContentBlock::Text(Text::new("")),
            ContentBlock::ToolCall(ToolCall::new(
                "toolu_01",
                "get_weather",
                r#"{"city":"Paris"}"#,
            )),
        ]))
        .message(Message::tool_result("toolu_01", "get_weather", "Sunny"))
        .message(Message::assistant_blocks(vec![
            ContentBlock::Text(Text::new("It is sunny.")),
            ContentBlock::Refusal(Refusal::new("")),
        ]))
        .user(vec![ContentPart::text(""), ContentPart::text("And Rome?")])
        .tools([weather_tool()]);
    assert_request_case("request_empty_text.json", &request);
}

/// A user message merged into a tool result run follows the results
/// (confirmed live: a `text` before the `tool_result` is a 400, "tool_use
/// ids were found without tool_result blocks immediately after").
#[test]
fn request_tool_result_order_fixture_matches() {
    let request = ChatRequest::new("claude-haiku-4-5-20251001")
        .user("Weather in Paris and Rome?")
        .message(Message::assistant_blocks(vec![
            ContentBlock::ToolCall(ToolCall::new(
                "toolu_01",
                "get_weather",
                r#"{"city":"Paris"}"#,
            )),
            ContentBlock::ToolCall(ToolCall::new(
                "toolu_02",
                "get_weather",
                r#"{"city":"Rome"}"#,
            )),
        ]))
        .message(Message::tool_result("toolu_01", "get_weather", "Sunny"))
        .user("(Rome may be slow to answer.)")
        .message(Message::tool_result("toolu_02", "get_weather", "Rain"))
        .tools([weather_tool()]);
    assert_request_case("request_tool_result_order.json", &request);
}

/// Asking only to hide reasoning never turns thinking on. Claude Haiku 4.5
/// thinks only when asked: nothing is sent, with an `Unsupported` warning.
/// Claude Sonnet 5.5 thinks by default: only the display is set.
#[test]
fn request_output_only_reasoning_fixture_matches() {
    let request = |model: &str| {
        ChatRequest::new(model)
            .user("Hello")
            .reasoning(ReasoningConfig::new().output(ReasoningOutput::Omitted))
    };
    assert_request_case(
        "request_output_only_reasoning.json",
        &request("claude-haiku-4-5-20251001"),
    );
    assert_request_case(
        "request_output_only_reasoning_thinking_by_default.json",
        &request("claude-sonnet-5-5"),
    );
}

fn weather_tool() -> ToolDefinition {
    ToolDefinition::new(
        "get_weather",
        json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
    )
}

#[test]
fn response_fixture_matches() {
    let response = decode_fixture("response_raw.json");
    if updating() {
        write_fixture(
            "response_expected.json",
            &serde_json::to_value(&response).unwrap(),
        );
    }
    assert_response_fixture_eq(&response, &fixtures(), "response_expected.json");
    assert_usage_contract(response.usage.as_ref().unwrap());
}

#[test]
fn recorded_response_replays_thinking_and_tool_use() {
    let raw = load_json_fixture(&fixtures(), "response_raw.json");
    let response = decode_fixture("response_raw.json");
    let request = request_fixture()
        .message(response.to_assistant_message())
        .message(Message::tool_result(
            "toolu_01E1R6AMsEMZ58RwQnHCAaB1",
            "get_weather",
            "Sunny",
        ));
    let encoded = crate::encode::encode_request(&request, false).unwrap();
    assert!(encoded.warnings.is_empty(), "{:?}", encoded.warnings);
    let body = serde_json::to_value(&encoded.body).unwrap();
    let replayed = &body["messages"][1]["content"];
    // Thinking (with its signature) replays verbatim; the direct `caller`
    // carries no state and is omitted.
    assert_eq!(replayed[0], raw["content"][0]);
    let mut tool_use = raw["content"][1].clone();
    tool_use.as_object_mut().unwrap().remove("caller");
    assert_eq!(replayed[1], tool_use);
}

#[test]
fn cache_response_fixture_matches() {
    let response = decode_fixture("response_cache_raw.json");
    if updating() {
        write_fixture(
            "response_cache_expected.json",
            &serde_json::to_value(&response).unwrap(),
        );
    }
    assert_response_fixture_eq(&response, &fixtures(), "response_cache_expected.json");
    let usage = response.usage.unwrap();
    assert_eq!(usage.cache_write_input_tokens, Some(7201));
    assert_eq!(usage.input_tokens, Some(7213));
    assert_usage_contract(&usage);
}

#[test]
fn server_tools_response_fixture_matches_and_replays_verbatim() {
    let raw = load_json_fixture(&fixtures(), "response_server_tools_raw.json");
    let response = decode_fixture("response_server_tools_raw.json");
    if updating() {
        write_fixture(
            "response_server_tools_expected.json",
            &serde_json::to_value(&response).unwrap(),
        );
    }
    assert_response_fixture_eq(
        &response,
        &fixtures(),
        "response_server_tools_expected.json",
    );

    let request = ChatRequest::new("claude-haiku-4-5-20251001")
        .user("What is the latest stable Rust version?")
        .message(response.to_assistant_message())
        .user("Thanks!");
    let encoded = crate::encode::encode_request(&request, false).unwrap();
    assert!(encoded.warnings.is_empty(), "{:?}", encoded.warnings);
    let body = serde_json::to_value(&encoded.body).unwrap();
    assert_eq!(
        body["messages"][1]["content"], raw["content"],
        "every block, including server tool blocks and citations, replays verbatim"
    );
}

/// `computer_toolset_20260801`: each member call names its toolset, the
/// calls replay verbatim, and each `tool_result` repeats the toolset name.
#[test]
fn computer_toolset_response_fixture_matches_and_round_trips() {
    let raw = load_json_fixture(&fixtures(), "response_computer_toolset_raw.json");
    let response = decode_fixture("response_computer_toolset_raw.json");
    if updating() {
        write_fixture(
            "response_computer_toolset_expected.json",
            &serde_json::to_value(&response).unwrap(),
        );
    }
    assert_response_fixture_eq(
        &response,
        &fixtures(),
        "response_computer_toolset_expected.json",
    );
    let calls: Vec<_> = response.tool_calls().cloned().collect();
    assert_eq!(calls.len(), 3);
    for call in &calls {
        assert_eq!(
            call.provider_data.get_raw(crate::NAMESPACE, "toolset_name"),
            Some(&json!("computer")),
            "{call:?}"
        );
    }

    let mut request = ChatRequest::new("claude-sonnet-5-5")
        .user("Search for pictures of cats.")
        .with_option(
            ChatRequestOptions::new().server_tool(
                json!({"type": "computer_toolset_20260801"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ),
        );
    request.push_message(response.to_assistant_message());
    request.push_tool_result(&calls[0], "OK");
    request.push_tool_result(&calls[1], "OK");
    request.push_message(Message::tool_result(
        calls[2].id.clone(),
        calls[2].name.clone(),
        vec![allama::ContentPart::image_base64(
            "image/png",
            "iVBORw0KGgo=",
        )],
    ));
    let encoded = crate::encode::encode_request(&request, false).unwrap();
    assert!(encoded.warnings.is_empty(), "{:?}", encoded.warnings);
    let body = serde_json::to_value(&encoded.body).unwrap();
    assert_eq!(
        body["tools"],
        json!([{"type": "computer_toolset_20260801"}])
    );
    assert_eq!(
        body["messages"][1]["content"], raw["content"],
        "the member calls replay verbatim, `toolset_name` included"
    );
    let results = body["messages"][2]["content"].as_array().unwrap();
    assert_eq!(results.len(), 3);
    for (result, call) in results.iter().zip(&calls) {
        assert_eq!(result["type"], "tool_result");
        assert_eq!(result["tool_use_id"], call.id.as_str());
        assert_eq!(result["toolset_name"], "computer");
    }
    assert_eq!(results[2]["content"][0]["type"], "image");
}

#[tokio::test]
async fn stream_fixture_matches() {
    if updating() {
        update_stream_fixtures(
            "stream.sse",
            "stream_events.json",
            Some("stream_response_expected.json"),
        )
        .await;
    }
    assert_stream_fixture_eq(
        stream_fixture("stream.sse"),
        &fixtures(),
        "stream_events.json",
        "stream_response_expected.json",
    )
    .await;
}

#[tokio::test]
async fn cache_stream_fixture_matches() {
    if updating() {
        update_stream_fixtures(
            "stream_cache.sse",
            "stream_cache_events.json",
            Some("stream_cache_response_expected.json"),
        )
        .await;
    }
    assert_stream_fixture_eq(
        stream_fixture("stream_cache.sse"),
        &fixtures(),
        "stream_cache_events.json",
        "stream_cache_response_expected.json",
    )
    .await;
    let response = stream_fixture("stream_cache.sse")
        .collect_response()
        .await
        .unwrap();
    let usage = response.usage.unwrap();
    assert_eq!(usage.cache_read_input_tokens, Some(7201));
    assert_eq!(usage.input_tokens, Some(7213));
    assert_eq!(usage.output_tokens, Some(6));
    assert_usage_contract(&usage);
}

#[tokio::test]
async fn server_tools_stream_fixture_matches() {
    if updating() {
        update_stream_fixtures(
            "stream_server_tools.sse",
            "stream_server_tools_events.json",
            Some("stream_server_tools_response_expected.json"),
        )
        .await;
    }
    assert_stream_fixture_eq(
        stream_fixture("stream_server_tools.sse"),
        &fixtures(),
        "stream_server_tools_events.json",
        "stream_server_tools_response_expected.json",
    )
    .await;
}

#[tokio::test]
async fn streamed_and_unary_server_tool_blocks_agree() {
    let streamed = stream_fixture("stream_server_tools.sse")
        .collect_response()
        .await
        .unwrap();
    let types: Vec<&str> = streamed.content.iter().map(|b| b.type_name()).collect();
    assert_eq!(
        types,
        vec!["provider_tool_call", "provider_tool_result", "text", "text"]
    );
    let allama::ContentBlock::ProviderToolCall(call) = &streamed.content[0] else {
        panic!("expected a provider tool call");
    };
    let input: serde_json::Value = serde_json::from_str(&call.arguments).unwrap();
    assert!(input["query"].is_string(), "{input:?}");
    assert_eq!(
        streamed.content[1]
            .as_provider_tool_result()
            .map(|r| &r.call_id),
        Some(&call.id),
        "the result answers the streamed call"
    );
}

#[tokio::test]
async fn truncated_stream_fixture_reports_strict_error_and_partial_recovery() {
    if updating() {
        update_stream_fixtures("stream_truncated.sse", "stream_truncated_events.json", None).await;
        let mut stream = stream_fixture("stream_truncated.sse");
        let mut collector = allama::chat::stream::StreamCollector::new();
        let mut error = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(event) => {
                    collector.push(event).unwrap();
                }
                Err(err) => error = Some(err),
            }
        }
        write_fixture(
            "stream_truncated_error.json",
            &serde_json::to_value(error.unwrap().as_log()).unwrap(),
        );
        let partial = collector.finish_partial().unwrap();
        write_fixture(
            "stream_truncated_response_expected.json",
            &serde_json::to_value(&partial.response).unwrap(),
        );
        write_fixture(
            "stream_truncated_completeness.json",
            &serde_json::to_value(&partial.completeness).unwrap(),
        );
    }

    assert_stream_finish_error_fixture_eq(
        stream_fixture("stream_truncated.sse"),
        &fixtures(),
        "stream_truncated_events.json",
        "stream_truncated_error.json",
    )
    .await;

    assert_partial_stream_fixture_eq(
        stream_fixture("stream_truncated.sse"),
        &fixtures(),
        "stream_truncated_events.json",
        "stream_truncated_response_expected.json",
        "stream_truncated_completeness.json",
    )
    .await;
}

#[test]
fn error_fixtures_match() {
    let body = |error_type: &str, message: &str| {
        json!({"type": "error", "error": {"type": error_type, "message": message}}).to_string()
    };
    let cases = [
        (
            "error_auth.json",
            crate::error::map_http_error(
                401,
                &headers(&[("request-id", "req_011CauthExample")]),
                &body("authentication_error", "invalid x-api-key"),
                None,
            ),
        ),
        (
            "error_rate_limited.json",
            crate::error::map_http_error(
                429,
                &headers(&[("request-id", "req-anthropic-429"), ("retry-after", "3")]),
                &body("rate_limit_error", "too many requests"),
                None,
            ),
        ),
        (
            "error_quota_exhausted.json",
            crate::error::map_http_error(
                429,
                &Headers::new(),
                &json!({
                    "type": "error",
                    "error": {
                        "type": "rate_limit_error",
                        "message": "You have reached your organization's spend limit.",
                        "details": {"error_code": "enforced_spend_limit_reached"}
                    },
                    "request_id": "req_011CquotaExample"
                })
                .to_string(),
                None,
            ),
        ),
        (
            "error_context_length.json",
            crate::error::map_http_error(
                400,
                &Headers::new(),
                &body(
                    "invalid_request_error",
                    "prompt is too long: 208310 tokens > 200000 maximum",
                ),
                None,
            ),
        ),
        (
            "error_model_not_found.json",
            crate::error::map_http_error(
                404,
                &Headers::new(),
                &body("not_found_error", "model: claude-does-not-exist"),
                None,
            ),
        ),
        (
            "error_timeout.json",
            crate::error::conformance_map_timeout_transport_error(),
        ),
    ];
    for (name, error) in cases {
        if updating() {
            write_fixture(name, &serde_json::to_value(error.as_log()).unwrap());
        }
        assert_error_fixture_eq(&error, &fixtures(), name);
    }
}

#[test]
fn schemaless_json_response_format_is_rejected() {
    let request = ChatRequest::new("claude-sonnet-5-5")
        .message(Message::user("Return JSON"))
        .response_format(ResponseFormat::Json);
    let err = crate::encode::encode_request(&request, false).unwrap_err();
    assert!(
        matches!(&err, allama::Error::Unsupported { message, .. } if message.contains("response_format"))
    );
}

#[test]
fn stream_events_carry_provider_identity() {
    let events = load_json_fixture(&fixtures(), "stream_events.json");
    let metadata: Vec<&Value> = events
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["event"] == "response_metadata")
        .collect();
    assert!(!metadata.is_empty());
    assert!(metadata.iter().all(|m| m["provider"] == "anthropic"));
}

/// Anthropic's envelope for each probed kind, told apart by error type and
/// spend-limit code, with the body request id and the upstream text after
/// its message.
fn error_envelope(kind: ErrorKindProbe, upstream: &str) -> Option<String> {
    let (error_type, message, code) = match kind {
        ErrorKindProbe::RateLimited => ("rate_limit_error", "Number of requests exceeded", None),
        ErrorKindProbe::Overloaded => ("overloaded_error", "Overloaded", None),
        ErrorKindProbe::InvalidRequest => {
            ("invalid_request_error", "messages: Field required", None)
        }
        ErrorKindProbe::Auth => ("authentication_error", "invalid x-api-key", None),
        ErrorKindProbe::QuotaExhausted => (
            "rate_limit_error",
            "Request rejected",
            Some("enforced_spend_limit_reached"),
        ),
        ErrorKindProbe::ContextLength => (
            "invalid_request_error",
            "prompt is too long: 300000 tokens > 200000 maximum",
            None,
        ),
        // https://docs.claude.com/en/api/errors: `not_found_error` (404) and
        // `timeout_error` (504).
        ErrorKindProbe::ModelNotFound => ("not_found_error", "model: claude-nope", None),
        ErrorKindProbe::Timeout => ("timeout_error", "Request timed out", None),
        // A field the model refuses (see `error::upstream_refusal`).
        ErrorKindProbe::Unsupported => (
            "invalid_request_error",
            "`temperature` may only be set to 1 when thinking is enabled.",
            None,
        ),
        _ => return None,
    };
    let message = format!("{message} {upstream}");
    let mut error = json!({"type": error_type, "message": message.trim_end()});
    if let Some(code) = code {
        error["details"] = json!({"error_code": code});
    }
    Some(json!({"type": "error", "error": error, "request_id": "req_probe"}).to_string())
}

/// An `error` event after the stream started.
fn error_stream(envelope: &str) -> String {
    format!(
        "event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"msg_1\",\"model\":\"claude-haiku-4-5\",\"usage\":{{\"input_tokens\":1}}}}}}\n\n\
         event: error\ndata: {envelope}\n\n"
    )
}

/// Current, previous-generation, and thinking-only model ids.
const MODELS: &[&str] = &[
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5",
    "claude-opus-4-1",
    "claude-3-5-haiku-latest",
    "claude-3-7-sonnet-latest",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
];

fn ok() -> ReplayResponse {
    ReplayResponse::json(
        200,
        json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn", "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })
        .to_string(),
    )
}

fn provider(transport: &ReplayTransport) -> crate::Provider {
    crate::Provider::builder()
        .api_key("test-key")
        .http_transport(transport.clone())
        .build()
        .unwrap()
}

fn bodies(transport: &ReplayTransport) -> Vec<Value> {
    transport
        .requests()
        .iter()
        .filter_map(|r| r.body.as_ref())
        .map(|body| serde_json::from_slice::<Value>(body).expect("JSON request body"))
        .collect()
}

async fn probe(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
    let transport = ReplayTransport::new([ok()]);
    let result = provider(&transport).chat(&request).await;
    ProbeOutcome::new(result, bodies(&transport))
}

/// Answers with a captured stream (`fixtures/stream.sse`), whose
/// `message_start` and `message_delta` carry usage.
async fn stream_probe(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
    let transport =
        ReplayTransport::new([ReplayResponse::sse(include_str!("../fixtures/stream.sse"))]);
    let result = match provider(&transport).chat_stream(&request).await {
        Ok(stream) => stream.collect_response().await,
        Err(err) => Err(err),
    };
    ProbeOutcome::new(result, bodies(&transport))
}

/// Requests whose encoding depends on the model: manual thinking with a
/// forced tool choice, and thinking budgets that do not fit an explicit cap
/// (scaled, dropped, or contradicting it; ADR 0003, principle 4).
fn validation_requests(model: &str) -> Vec<ChatRequest> {
    let tool = ToolDefinition::new("t", json!({"type": "object"}));
    let mut requests = vec![
        ChatRequest::new(model)
            .user("Hi")
            .tools([tool])
            .tool_choice(ToolChoice::Required)
            .with_option(ChatRequestOptions::new().budget_tokens(1024)),
    ];
    for cap in [4000, 1500] {
        let request = ChatRequest::new(model)
            .user("Hi")
            .max_output_tokens(cap)
            .reasoning(ReasoningEffort::High);
        requests.push(request.clone());
        requests.push(request.unsupported_policy(allama::UnsupportedPolicy::Error));
    }
    requests.push(
        ChatRequest::new(model)
            .user("Hi")
            .max_output_tokens(1000)
            .with_option(ChatRequestOptions::new().budget_tokens(2048)),
    );
    requests
}

/// One call with transport options attached, on `chat` or `chat_stream`.
async fn send(options: allama::RequestOptions, streaming: bool) -> TransportOutcome {
    let transport = ReplayTransport::default();
    let mut request = ChatRequest::new("claude-haiku-4-5").user("Hi");
    request.options = options;
    let provider = provider(&transport);
    let result = if streaming {
        provider.chat_stream(&request).await.map(drop)
    } else {
        provider.chat(&request).await.map(drop)
    };
    TransportOutcome::new(result, transport.requests())
}

/// Every contract, for every model kind and for one Anthropic does not
/// know. Validation is a dry run of the encoding (`Encoding` coverage).
#[tokio::test]
async fn the_adapter_passes_every_contract() {
    let anthropic = provider(&ReplayTransport::new([]));
    for model in MODELS.iter().chain(&[UNKNOWN_MODEL]) {
        let validation = anthropic
            .validate_chat_request(&ChatRequest::new(*model).user("Hi"))
            .unwrap();
        assert_eq!(validation.coverage(), ValidationCoverage::Encoding);
    }
    // `message_stop` is the terminal marker, so a body that closes before it
    // is `Incomplete`, and one that reaches it without a `message_delta`
    // stop reason collects with the missing-finish warning.
    let start = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-haiku-4-5\",\"usage\":{\"input_tokens\":1}}}\n\n\
         event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
         event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n\
         event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n";
    let ends = StreamEnds::new(start).terminated_without_finish(format!(
        "{start}event: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n"
    ));
    let slot = CredentialSlot::new(
        crate::DEFAULT_CREDENTIAL_SCOPE,
        CredentialPlacement::Header("x-api-key".into()),
    );
    let under_test = AdapterUnderTest::new(
        MODELS.iter().copied(),
        crate::NAMESPACE,
        probe,
        stream_probe,
    )
    .validation_requests(
        MODELS
            .iter()
            .chain(&[UNKNOWN_MODEL])
            .flat_map(|model| validation_requests(model)),
    )
    // No model is answered `Unsupported` for structured output or
    // reasoning replay on its id (ADR 0015): every model sends strict tools
    // strict and replays reasoning, and the API decides. The resolver path
    // that relaxes or drops them is covered by the encoder's and core
    // preflight's own tests.
    .skip([
        Contract::StrictNotHonouredWarns,
        Contract::UnreplayableReasoningDropped,
    ])
    .error_mapping(|status, headers, body| {
        crate::error::map_http_error(status, &headers.iter().copied().collect(), &body, None)
    })
    .error_envelope(error_envelope)
    .stream_body(error_stream)
    .stream_replay(|body| async move {
        let transport = ReplayTransport::new([ReplayResponse::sse(body)]);
        provider(&transport)
            .chat_stream(&ChatRequest::new("claude-haiku-4-5").user("hi"))
            .await
    })
    .stream_ends(ends)
    .sse_captures(
        ["stream.sse", "stream_cache.sse", "stream_server_tools.sse"]
            .map(|name| load_text_fixture(&fixtures(), name)),
    )
    .transport_options(
        slot,
        |options| send(options, false),
        |options| send(options, true),
    )
    .malformed_response(
        |body| async move {
            let answer = ReplayResponse::new(200, body).header("request-id", "req_1");
            provider(&ReplayTransport::new([answer]))
                .chat(&ChatRequest::new("claude-haiku-4-5").user("hi"))
                .await
        },
        Some("req_1"),
    )
    // The decoding contracts: each pair is one response as `chat` and as
    // `chat_stream` receive it.
    .unary_replay(|body| async move {
        provider(&ReplayTransport::new([ReplayResponse::json(200, body)]))
            .chat(&ChatRequest::new("claude-haiku-4-5").user("hi"))
            .await
    })
    .response_captures([capture("agree_text")])
    .tool_call_capture(
        capture("agree_tool_calls")
            .wire_tool_call_ids(["toolu_01WeatherParisA1", "toolu_01TimeParisB2"]),
    )
    .length_capture(capture("agree_length"));
    #[cfg(feature = "reqwest")]
    let under_test = under_test.redirect_not_followed(|base_url, streaming| async move {
        let provider = crate::Provider::builder()
            .api_key("test-key")
            .base_url(base_url)
            .build()?;
        let request = ChatRequest::new("claude-haiku-4-5").user("hi");
        if streaming {
            provider.chat_stream(&request).await.map(drop)
        } else {
            provider.chat(&request).await.map(drop)
        }
    });
    // Without the default client there is no redirect policy to check.
    #[cfg(not(feature = "reqwest"))]
    let under_test = under_test.skip([Contract::RedirectNotFollowed]);
    assert_chat_adapter(&anthropic, &under_test).await;
}

/// `fixtures/<name>.json` and `fixtures/<name>.sse`: one response as the
/// Messages API sends it unary and streamed.
fn capture(name: &str) -> allama_conformance::ResponseCapture {
    allama_conformance::ResponseCapture::new(
        load_text_fixture(&fixtures(), &format!("{name}.json")),
        load_text_fixture(&fixtures(), &format!("{name}.sse")),
    )
}
