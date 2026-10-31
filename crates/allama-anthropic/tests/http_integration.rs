// Sends through the default `reqwest` transport.
#![cfg(feature = "reqwest")]

use std::path::PathBuf;
use std::time::Duration;

use allama::{
    CacheBreakpoint, CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatStreamExt,
    ContentPart, Error, FinishReason, Message, RequestField, ResponseFormat, StreamEvent,
    SystemPrompt, ToolChoice, ToolDefinition, UnsupportedStage, WarningKind,
};
use allama_anthropic::{ChatRequestOptions, Provider};
use allama_conformance::contract::{
    assert_auth_error_contract, assert_chat_success_contract,
    assert_context_length_exceeded_contract, assert_overloaded_error_contract,
    assert_quota_exhausted_contract, assert_rate_limited_error_contract,
    assert_streaming_chat_contract, assert_strict_policy_rejects,
    assert_strict_policy_rejects_stream, assert_structured_output_contract,
    assert_structured_output_contract_stream, assert_tool_call_contract,
    assert_unsupported_hint_warns, assert_unsupported_hint_warns_stream, collect_chat_stream,
};
use allama_conformance::{
    FixtureDir, MockHttpResponse, TestHttpServer, assert_partial_stream_fixture_eq,
    assert_response_fixture_eq, assert_stream_finish_error_fixture_eq, assert_stream_fixture_eq,
    load_json_fixture, load_text_fixture,
};
use allama_http::http::{HeaderValue, StatusCode};
use futures_util::StreamExt;
use serde_json::{Value, json};

const MODEL: &str = "claude-haiku-4-5-20251001";

fn fixtures() -> FixtureDir {
    FixtureDir::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"))
}

fn provider(server: &TestHttpServer) -> Provider {
    Provider::builder()
        .api_key("anthropic-test-key")
        .base_url(format!("{}/", server.url()))
        .build()
        .unwrap()
}

fn error_body(error_type: &str, message: &str) -> Value {
    json!({"type": "error", "error": {"type": error_type, "message": message}})
}

fn text_response() -> Value {
    json!({
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": 10, "output_tokens": 3}
    })
}

#[tokio::test]
async fn chat_posts_expected_request_and_parses_response() {
    let fixtures = fixtures();
    let raw = load_json_fixture(&fixtures, "response_raw.json");
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(StatusCode::OK, &raw)]).await;
    let provider = provider(&server);

    let response = provider
        .chat(
            &ChatRequest::new(MODEL)
                .user("Hello from the transport integration test.")
                .with_option(
                    ChatRequestOptions::new()
                        .beta("integration-beta")
                        .beta("other-beta"),
                ),
        )
        .await
        .unwrap();

    assert_response_fixture_eq(&response, &fixtures, "response_expected.json");

    let requests = server.recorded_requests().await;
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    let body = request.body_json();
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v1/messages");
    assert_eq!(request.header("x-api-key"), Some("anthropic-test-key"));
    assert_eq!(request.header("anthropic-version"), Some("2023-06-01"));
    assert_eq!(
        request.header("anthropic-beta"),
        Some("integration-beta,other-beta")
    );
    assert_eq!(request.header("content-type"), Some("application/json"));
    assert_eq!(body["model"], MODEL);
    assert_eq!(body["messages"][0]["role"], "user");
    assert_eq!(body["stream"], false);
}

#[tokio::test]
async fn chat_reads_request_id_header() {
    let server =
        TestHttpServer::spawn([
            MockHttpResponse::json_value(StatusCode::OK, &text_response())
                .header("request-id", HeaderValue::from_static("req_011Cabc")),
        ])
        .await;
    let response = provider(&server)
        .chat(&ChatRequest::new(MODEL).user("hi"))
        .await
        .unwrap();
    assert_eq!(response.request_id.as_deref(), Some("req_011Cabc"));
    assert_eq!(response.provider.as_deref(), Some("anthropic"));
}

#[tokio::test]
async fn chat_success_and_tool_call_contracts() {
    let tool_response = json!({
        "id": "msg_2",
        "model": MODEL,
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 3}
    });
    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(StatusCode::OK, &text_response()),
        MockHttpResponse::json_value(StatusCode::OK, &tool_response),
    ])
    .await;
    let provider = provider(&server);
    assert_chat_success_contract(&provider, &ChatRequest::new(MODEL).user("hi")).await;
    let tool = ToolDefinition::new(
        "get_weather",
        json!({"type": "object", "properties": {"city": {"type": "string"}}}),
    );
    assert_tool_call_contract(
        &provider,
        &ChatRequest::new(MODEL).user("weather?").tools([tool]),
    )
    .await;
}

#[tokio::test]
async fn chat_stream_uses_stream_mode_and_collects_fixture() {
    let fixtures = fixtures();
    let server = TestHttpServer::spawn([MockHttpResponse::sse(load_text_fixture(
        &fixtures,
        "stream.sse",
    ))])
    .await;

    let stream = provider(&server)
        .chat_stream(&ChatRequest::new(MODEL).user("Stream this."))
        .await
        .unwrap();

    assert_stream_fixture_eq(
        stream,
        &fixtures,
        "stream_events.json",
        "stream_response_expected.json",
    )
    .await;

    let requests = server.recorded_requests().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].path, "/v1/messages");
    assert_eq!(requests[0].body_json()["stream"], true);
}

#[tokio::test]
async fn chat_stream_contract_and_request_id() {
    let fixtures = fixtures();
    let sse = load_text_fixture(&fixtures, "stream_cache.sse");
    let server = TestHttpServer::spawn([
        MockHttpResponse::sse(sse.clone()),
        MockHttpResponse::sse(sse).header("request-id", HeaderValue::from_static("req_stream")),
    ])
    .await;
    let provider = provider(&server);
    assert_streaming_chat_contract(&provider, &ChatRequest::new(MODEL).user("hi")).await;

    let response = provider
        .chat_stream(&ChatRequest::new(MODEL).user("hi"))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();
    assert_eq!(response.request_id.as_deref(), Some("req_stream"));
    assert_eq!(response.provider.as_deref(), Some("anthropic"));
}

#[tokio::test]
async fn chat_stream_clean_eof_reports_incomplete_stream() {
    let fixtures = fixtures();
    let truncated = load_text_fixture(&fixtures, "stream_truncated.sse");
    let server = TestHttpServer::spawn([
        MockHttpResponse::sse(truncated.clone()),
        MockHttpResponse::sse(truncated),
    ])
    .await;
    let provider = provider(&server);

    let stream = provider
        .chat_stream(&ChatRequest::new(MODEL).user("Stream this."))
        .await
        .unwrap();
    assert_stream_finish_error_fixture_eq(
        stream,
        &fixtures,
        "stream_truncated_events.json",
        "stream_truncated_error.json",
    )
    .await;

    let stream = provider
        .chat_stream(&ChatRequest::new(MODEL).user("Stream this."))
        .await
        .unwrap();
    assert_partial_stream_fixture_eq(
        stream,
        &fixtures,
        "stream_truncated_events.json",
        "stream_truncated_response_expected.json",
        "stream_truncated_completeness.json",
    )
    .await;
}

#[tokio::test]
async fn chat_stream_mid_stream_overload_is_typed() {
    let sse = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"x\",\"usage\":{\"input_tokens\":1}}}\n\n\
event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n";
    let server = TestHttpServer::spawn([
        MockHttpResponse::sse(sse).header("request-id", HeaderValue::from_static("req_ov"))
    ])
    .await;
    let mut stream = provider(&server)
        .chat_stream(&ChatRequest::new(MODEL).user("hi"))
        .await
        .unwrap();
    let mut last = None;
    while let Some(item) = stream.next().await {
        last = Some(item);
    }
    let err = last.unwrap().unwrap_err();
    assert!(matches!(err, Error::Overloaded { .. }), "{err:?}");
    assert_eq!(err.request_id(), Some("req_ov"));
    assert!(err.is_retryable());
}

#[tokio::test]
async fn chat_maps_non_success_status_to_typed_error() {
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(
        StatusCode::TOO_MANY_REQUESTS,
        &error_body("rate_limit_error", "too many requests"),
    )
    .header("request-id", HeaderValue::from_static("req-anthropic-429"))
    .header("retry-after", HeaderValue::from_static("3"))])
    .await;

    let err = provider(&server)
        .chat(&ChatRequest::new(MODEL).user("Trigger an error."))
        .await
        .unwrap_err();
    assert!(matches!(&err, Error::RateLimited { message, .. } if message == "HTTP 429"));
    assert_eq!(err.upstream_message(), Some("too many requests"));
    assert_eq!(err.retry_after(), Some(Duration::from_secs(3)));
    assert_eq!(err.request_id(), Some("req-anthropic-429"));
    assert_eq!(err.status(), Some(429));
}

#[tokio::test]
async fn server_retry_directive_is_surfaced_and_followed() {
    use allama::wrappers::{RetryPolicy, RetryingProvider};

    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(
            StatusCode::CONFLICT,
            &error_body("invalid_request_error", "conflict"),
        )
        .header("request-id", HeaderValue::from_static("req_409"))
        .header("x-should-retry", HeaderValue::from_static("true")),
        MockHttpResponse::json_value(StatusCode::OK, &text_response()),
    ])
    .await;
    let err = provider(&server)
        .chat(&ChatRequest::new(MODEL).user("x"))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
    assert_eq!(err.server_retry_directive(), Some(true));
    assert_eq!(err.request_id(), Some("req_409"));

    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(
            StatusCode::CONFLICT,
            &error_body("invalid_request_error", "conflict"),
        )
        .header("x-should-retry", HeaderValue::from_static("true")),
        MockHttpResponse::json_value(StatusCode::OK, &text_response()),
    ])
    .await;
    let retrying = RetryingProvider::new(
        provider(&server),
        RetryPolicy::new().base_delay(Duration::ZERO),
    );
    let response = retrying
        .chat(&ChatRequest::new(MODEL).user("x"))
        .await
        .unwrap();
    assert_eq!(response.text().as_deref(), Some("Hello!"));
    assert_eq!(server.recorded_requests().await.len(), 2);
}

#[tokio::test]
async fn retry_after_accepts_http_dates_and_ignores_garbage() {
    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(
            StatusCode::TOO_MANY_REQUESTS,
            &error_body("rate_limit_error", "slow"),
        )
        .header(
            "retry-after",
            HeaderValue::from_static("Thu, 01 Jan 1970 00:00:00 GMT"),
        ),
        MockHttpResponse::json_value(
            StatusCode::TOO_MANY_REQUESTS,
            &error_body("rate_limit_error", "slow"),
        )
        .header("retry-after", HeaderValue::from_static("-1")),
    ])
    .await;
    let provider = provider(&server);
    let err = provider
        .chat(&ChatRequest::new(MODEL).user("x"))
        .await
        .unwrap_err();
    assert_eq!(
        err.retry_after(),
        Some(Duration::ZERO),
        "past dates clamp to zero"
    );
    let err = provider
        .chat(&ChatRequest::new(MODEL).user("x"))
        .await
        .unwrap_err();
    assert_eq!(err.retry_after(), None);
}

#[tokio::test]
async fn error_contracts() {
    let quota = json!({
        "type": "error",
        "error": {
            "type": "rate_limit_error",
            "message": "You have reached your organization's spend limit.",
            "details": {"error_code": "enforced_spend_limit_reached"}
        }
    });
    let failures = [
        MockHttpResponse::json_value(
            StatusCode::UNAUTHORIZED,
            &error_body("authentication_error", "invalid x-api-key"),
        ),
        MockHttpResponse::json_value(
            StatusCode::TOO_MANY_REQUESTS,
            &error_body("rate_limit_error", "slow down"),
        ),
        MockHttpResponse::json_value(
            StatusCode::from_u16(529).unwrap(),
            &error_body("overloaded_error", "Overloaded"),
        ),
        MockHttpResponse::json_value(
            StatusCode::BAD_REQUEST,
            &error_body(
                "invalid_request_error",
                "prompt is too long: 300000 tokens > 200000 maximum",
            ),
        ),
        MockHttpResponse::json_value(StatusCode::TOO_MANY_REQUESTS, &quota),
    ];
    // Each failure answers `chat`, then `chat_stream`.
    let server = TestHttpServer::spawn(
        failures
            .into_iter()
            .flat_map(|failure| [failure.clone(), failure]),
    )
    .await;
    let provider = provider(&server);
    let request = ChatRequest::new(MODEL).user("hi");
    let stream = |request| collect_chat_stream(&provider, request);
    assert_auth_error_contract(&provider, &request, stream).await;
    assert_rate_limited_error_contract(&provider, &request, stream).await;
    assert_overloaded_error_contract(&provider, &request, stream).await;
    assert_context_length_exceeded_contract(&provider, &request, stream).await;
    assert_quota_exhausted_contract(&provider, &request, stream).await;
}

#[tokio::test]
async fn structured_output_contracts() {
    let answer = r#"{"city":"Paris"}"#;
    let mut response = text_response();
    response["content"] = json!([{"type": "text", "text": answer}]);
    let delta = json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": answer}});
    let sse = format!(
        "event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"msg_1\",\"model\":\"{MODEL}\",\"usage\":{{\"input_tokens\":10}}}}}}\n\n\
         event: content_block_start\ndata: {{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{{\"type\":\"text\",\"text\":\"\"}}}}\n\n\
         event: content_block_delta\ndata: {delta}\n\n\
         event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":0}}\n\n\
         event: message_delta\ndata: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"end_turn\"}},\"usage\":{{\"output_tokens\":5}}}}\n\n\
         event: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n"
    );
    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(StatusCode::OK, &response),
        MockHttpResponse::sse(sse),
    ])
    .await;
    let provider = provider(&server);
    let request = ChatRequest::new(MODEL).user("Which city?").response_format(
        ResponseFormat::json_schema(json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": false
        }))
        .name("city"),
    );
    assert_structured_output_contract(&provider, &request).await;
    assert_structured_output_contract_stream(&provider, &request).await;
}

#[tokio::test]
async fn unsupported_hints_warn_and_strict_policy_rejects_before_sending() {
    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(StatusCode::OK, &text_response()),
        MockHttpResponse::sse(load_text_fixture(&fixtures(), "stream.sse")),
    ])
    .await;
    let provider = provider(&server);
    let request = ChatRequest::new(MODEL).user("hi").seed(42);
    assert_unsupported_hint_warns(&provider, &request).await;
    assert_strict_policy_rejects(&provider, &request).await;
    assert_unsupported_hint_warns_stream(&provider, &request).await;
    assert_strict_policy_rejects_stream(&provider, &request).await;
    let requests = server.recorded_requests().await;
    assert_eq!(
        requests.len(),
        2,
        "the strict requests never reached the server"
    );
    assert!(requests.iter().all(|r| r.body_json().get("seed").is_none()));
}

#[tokio::test]
async fn streaming_emits_request_warnings() {
    let fixtures = fixtures();
    let server = TestHttpServer::spawn([MockHttpResponse::sse(load_text_fixture(
        &fixtures,
        "stream_cache.sse",
    ))])
    .await;
    let mut stream = provider(&server)
        .chat_stream(&ChatRequest::new(MODEL).user("hi").presence_penalty(0.4))
        .await
        .unwrap();
    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        events.push(item.unwrap());
    }
    assert!(matches!(events[0], StreamEvent::ResponseStart { .. }));
    assert!(matches!(
        &events[1],
        StreamEvent::Warning { warning, .. } if warning.kind == WarningKind::Unsupported
    ));
}

/// A forced tool choice is sent to a model documented to reject it, and
/// the API's refusal ("tool_choice: type \"tool\" and \"any\" are not
/// supported for this model.",
/// <https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide#forced-tool-use>)
/// is the provider's own refusal of the field (ADR 0015).
#[tokio::test]
async fn forced_tool_choice_is_refused_by_the_api_not_the_adapter() {
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(
        StatusCode::BAD_REQUEST,
        &error_body(
            "invalid_request_error",
            "tool_choice: type \"tool\" and \"any\" are not supported for this model.",
        ),
    )])
    .await;
    let tool = ToolDefinition::new("f", json!({"type": "object", "properties": {}}));
    let request = ChatRequest::new("claude-sonnet-5-5")
        .user("hi")
        .tools([tool])
        .tool_choice(ToolChoice::Specific { name: "f".into() });
    let err = provider(&server).chat(&request).await.unwrap_err();
    assert_eq!(err.unsupported_field(), Some(RequestField::ToolChoice));
    assert_eq!(err.unsupported_stage(), Some(UnsupportedStage::Provider));
    let sent = server.recorded_requests().await;
    assert_eq!(sent.len(), 1);
    assert_eq!(
        sent[0].body_json()["tool_choice"],
        json!({"type": "tool", "name": "f"})
    );
}

/// A sampling parameter is sent to a model that refuses it, and the API's
/// refusal (recorded from `claude-sonnet-5-5`: "`temperature` is deprecated
/// for this model.") is the provider's own refusal of that field, so a
/// fallback fails over to its other side.
#[tokio::test]
async fn a_deprecated_sampling_parameter_is_refused_by_the_api() {
    let refusal = |parameter: &str| {
        MockHttpResponse::json_value(
            StatusCode::BAD_REQUEST,
            &json!({
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": format!("`{parameter}` is deprecated for this model.")
                },
                "request_id": "req_011CfzPcSSaAQ7Wd46DMkbkW"
            }),
        )
    };
    let server = TestHttpServer::spawn([
        refusal("temperature"),
        refusal("top_p"),
        refusal("top_k"),
        refusal("temperature"),
        MockHttpResponse::json_value(StatusCode::OK, &text_response()),
    ])
    .await;
    let provider = provider(&server);
    let request = ChatRequest::new("claude-sonnet-5-5").user("hi");
    for (sampled, field) in [
        (request.clone().temperature(0.2), RequestField::Temperature),
        (request.clone().top_p(0.5), RequestField::TopP),
        (request.clone().top_k(5), RequestField::TopK),
    ] {
        let err = provider.chat(&sampled).await.unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
        assert_eq!(err.unsupported_field(), Some(field), "{err:?}");
        assert_eq!(err.unsupported_stage(), Some(UnsupportedStage::Provider));
        assert_eq!(err.request_id(), Some("req_011CfzPcSSaAQ7Wd46DMkbkW"));
    }

    let pair = allama::wrappers::FallbackProvider::new(provider.clone(), provider);
    let response = pair.chat(&request.temperature(0.2)).await.unwrap();
    assert_eq!(response.text().as_deref(), Some("Hello!"));

    let sent = server.recorded_requests().await;
    assert_eq!(sent.len(), 5);
    assert_eq!(sent[0].body_json()["temperature"], json!(0.2));
    assert_eq!(sent[1].body_json()["top_p"], json!(0.5));
    assert_eq!(sent[2].body_json()["top_k"], json!(5));
}

/// A prefill (an assistant message sent last) is sent as written, and the
/// API's refusal of it (here, a prefill with an output format, worded as
/// `claude-haiku-4-5` words it) is the provider's own refusal of that
/// message, on both paths: the turn contract's "sent, or refused with
/// `Unsupported`", never dropped.
#[tokio::test]
async fn a_refused_prefill_names_the_assistant_message_sent_last() {
    let refusal = || {
        MockHttpResponse::json_value(
            StatusCode::BAD_REQUEST,
            &error_body(
                "invalid_request_error",
                "Your API request included an `assistant` message in the final position, which \
                 would pre-fill the `assistant` response. When using output format, \
                 pre-filling the `assistant` response is not supported.",
            ),
        )
    };
    let server = TestHttpServer::spawn([refusal(), refusal()]).await;
    let provider = provider(&server);
    let request = ChatRequest::new(MODEL)
        .user("Which city?")
        .assistant("{")
        .response_format(
            ResponseFormat::json_schema(
                json!({"type": "object", "properties": {"city": {"type": "string"}}}),
            )
            .name("city")
            .non_strict(),
        );
    let prefill = allama::Subject::Content {
        message: 1,
        block: None,
    };
    let err = provider.chat(&request).await.unwrap_err();
    assert_eq!(err.unsupported_subject(), Some(&prefill), "{err:?}");
    assert_eq!(err.unsupported_stage(), Some(UnsupportedStage::Provider));
    let Err(err) = provider.chat_stream(&request).await else {
        panic!("the stream should be refused");
    };
    assert_eq!(err.unsupported_subject(), Some(&prefill), "{err:?}");
    let sent = server.recorded_requests().await;
    assert_eq!(sent.len(), 2);
    for request in &sent {
        let body = request.body_json();
        assert_eq!(body["messages"][1]["role"], "assistant");
        assert_eq!(body["messages"][1]["content"][0]["text"], "{");
    }
}

/// A resolver that answers forced tool choice `Unsupported` refuses it
/// before sending, as the built-in answers no longer do.
#[tokio::test]
async fn forced_tool_choice_answered_unsupported_is_refused_without_a_request() {
    let server = TestHttpServer::spawn(Vec::<MockHttpResponse>::new()).await;
    let tool = ToolDefinition::new("f", json!({"type": "object", "properties": {}}));
    let err = provider(&server)
        .with_chat_capabilities(|_: &str, capability| {
            (capability == ChatCapability::ForcedToolChoice)
                .then_some(CapabilitySupport::Unsupported)
        })
        .chat(
            &ChatRequest::new("claude-sonnet-5-5")
                .user("hi")
                .tools([tool])
                .tool_choice(ToolChoice::Specific { name: "f".into() }),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
    assert!(server.recorded_requests().await.is_empty());
}

#[tokio::test]
async fn caching_files_and_parallel_control_reach_the_wire() {
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(
        StatusCode::OK,
        &text_response(),
    )])
    .await;
    let tool = ToolDefinition::new("f", json!({"type": "object", "properties": {}}));
    let request = ChatRequest::new(MODEL)
        .system(SystemPrompt::new("Long system prompt").with_cache(CacheBreakpoint::new()))
        .message(Message::user(vec![
            ContentPart::text("Summarize"),
            ContentPart::file_base64("application/pdf", "JVBERi0xLjQK"),
        ]))
        .tools([tool])
        .parallel_tool_calls(false);
    let response = provider(&server).chat(&request).await.unwrap();
    assert_eq!(response.finish_reason, Some(FinishReason::Stop));
    let body = server.recorded_requests().await[0].body_json();
    assert_eq!(
        body["system"][0]["cache_control"],
        json!({"type": "ephemeral"})
    );
    assert_eq!(body["messages"][0]["content"][1]["type"], "document");
    assert_eq!(body["tool_choice"]["disable_parallel_tool_use"], true);
}

#[tokio::test]
async fn raw_response_is_kept_only_when_requested() {
    let raw = load_json_fixture(&fixtures(), "response_raw.json");
    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(StatusCode::OK, &raw),
        MockHttpResponse::json_value(StatusCode::OK, &raw),
    ])
    .await;
    let provider = provider(&server);
    let request = ChatRequest::new(MODEL).user("hi");

    let plain = provider.chat(&request).await.unwrap();
    assert!(
        plain
            .provider_data
            .get::<allama_anthropic::RawResponse>()
            .unwrap()
            .is_none()
    );

    let with_raw = provider
        .chat(&request.with_option(ChatRequestOptions::new().raw_response(true)))
        .await
        .unwrap();
    let kept = with_raw
        .provider_data
        .get::<allama_anthropic::RawResponse>()
        .unwrap()
        .unwrap();
    assert_eq!(kept.0, raw);
    let body = server.recorded_requests().await[1].body_json();
    assert!(body.get("raw_response").is_none(), "the option is not sent");
}

/// The adapter runs end to end over an in-memory transport, recorded
/// wire data in, request out, with no server.
#[tokio::test]
async fn replay_transport_runs_the_adapter_without_a_server() {
    use allama_http::{ReplayResponse, ReplayTransport};

    let raw = load_text_fixture(&fixtures(), "response_raw.json");
    let sse = load_text_fixture(&fixtures(), "stream.sse");
    let transport = ReplayTransport::new([
        ReplayResponse::json(StatusCode::OK, raw)
            .header("request-id", HeaderValue::from_static("req_replay")),
        ReplayResponse::sse(sse),
    ]);
    let provider = Provider::builder()
        .api_key("replay-key")
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let request = ChatRequest::new(MODEL).user("hi");

    let response = provider.chat(&request).await.unwrap();
    assert_eq!(response.request_id.as_deref(), Some("req_replay"));
    assert!(response.has_tool_calls());
    let streamed = provider
        .chat_stream(&request)
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();
    assert!(!streamed.content.is_empty());

    let sent = transport.requests();
    assert_eq!(
        sent[0].uri.to_string(),
        "https://api.anthropic.com/v1/messages"
    );
    assert_eq!(
        sent[0].headers.get("x-api-key"),
        Some(&HeaderValue::from_static("replay-key"))
    );
    assert!(sent[0].timeout.is_some(), "unary requests are bounded");
    assert!(sent[1].timeout.is_none(), "streams are not");
    let body: Value = serde_json::from_slice(sent[1].body.as_ref()).unwrap();
    assert_eq!(body["stream"], true);
}

/// A stream whose connection stalls after its first event fails with a
/// retryable timeout instead of hanging.
#[tokio::test]
async fn stalled_stream_times_out() {
    use allama_anthropic::StreamTimeouts;
    use allama_http::{ReplayResponse, ReplayTransport};

    let sse = load_text_fixture(&fixtures(), "stream.sse");
    let first_event = sse.split_inclusive("\n\n").next().unwrap();
    let transport = ReplayTransport::new([ReplayResponse::sse(first_event).then_stall()]);
    let provider = Provider::builder()
        .api_key("replay-key")
        .http_transport(transport)
        .stream_timeouts(StreamTimeouts::default().idle(Some(Duration::from_millis(30))))
        .build()
        .unwrap();
    let err = provider
        .chat_stream(&ChatRequest::new(MODEL).user("hi"))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Timeout { .. }), "{err:?}");
    assert!(err.is_retryable());
    assert!(
        err.to_string()
            .contains("wait for the next body chunk exceeded 30ms"),
        "{err}"
    );
}

/// An event past the configured size cap fails the stream with a stream
/// protocol error, which retrying would only repeat.
#[tokio::test]
async fn stream_event_past_the_size_cap_fails() {
    use allama::StreamErrorKind;
    use allama_http::{ReplayResponse, ReplayTransport};

    let sse = load_text_fixture(&fixtures(), "stream.sse");
    let transport = ReplayTransport::new([ReplayResponse::sse(sse)]);
    let provider = Provider::builder()
        .api_key("replay-key")
        .http_transport(transport)
        .max_stream_event_size(64)
        .build()
        .unwrap();
    let err = provider
        .chat_stream(&ChatRequest::new(MODEL).user("hi"))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap_err();
    assert!(
        matches!(
            err,
            Error::Stream {
                kind: StreamErrorKind::Protocol,
                ..
            }
        ),
        "{err:?}"
    );
    assert!(!err.is_retryable());
    assert!(
        err.to_string()
            .contains("an SSE event passed the size cap of 64 bytes"),
        "{err}"
    );
    assert_eq!(
        err.context().and_then(|c| c.provider.as_deref()),
        Some("anthropic")
    );
}

/// A stream line that is not UTF-8 fails the stream with a stream decode
/// error, not a retryable transport failure: nothing replaces the bytes,
/// and asking again streams the same ones.
#[tokio::test]
async fn stream_invalid_utf8_fails_without_a_retry() {
    use allama::StreamErrorKind;
    use allama::wrappers::{RetryPolicy, RetryingProvider};
    use allama_http::{ReplayResponse, ReplayTransport};

    let sse = load_text_fixture(&fixtures(), "stream.sse");
    let message_start = &sse[..sse.find("\n\n").unwrap() + 2];
    let garbled = |body: &[u8]| {
        let mut bytes = message_start.as_bytes().to_vec();
        bytes.extend_from_slice(body);
        ReplayResponse::new(StatusCode::OK, bytes).header(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        )
    };
    let block_start = b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\
        \"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n";
    let delta = |text: &[u8]| {
        let mut event = b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\
            \"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\""
            .to_vec();
        event.extend_from_slice(text);
        event.extend_from_slice(b"\"}}\n\n");
        event
    };

    // Mid-stream: the text before the bad line arrives, then the error.
    let mut body = block_start.to_vec();
    body.extend(delta(b"Hel"));
    body.extend(delta(b"caf\xC3"));
    let transport = ReplayTransport::new([garbled(&body)]);
    let provider = Provider::builder()
        .api_key("replay-key")
        .http_transport(transport)
        .build()
        .unwrap();
    let mut stream = provider
        .chat_stream(&ChatRequest::new(MODEL).user("hi"))
        .await
        .unwrap();
    let mut text = String::new();
    let err = loop {
        match stream.next().await.expect("the stream ends with an error") {
            Ok(StreamEvent::TextDelta { text: delta, .. }) => text.push_str(&delta),
            Ok(_) => {}
            Err(err) => break err,
        }
    };
    assert_eq!(text, "Hel");
    assert!(
        matches!(
            err,
            Error::Stream {
                kind: StreamErrorKind::Decode,
                ..
            }
        ),
        "{err:?}"
    );
    assert!(!err.is_retryable());
    assert!(
        err.to_string().contains("invalid UTF-8 in the SSE stream"),
        "{err}"
    );
    assert!(err.as_log().to_string().contains("invalid UTF-8"));
    assert_eq!(
        err.context().and_then(|c| c.provider.as_deref()),
        Some("anthropic")
    );

    // Before any output: the retry wrapper does not ask again.
    let transport = ReplayTransport::new([garbled(b"data: \xFF\n\n"), garbled(&body)]);
    let retrying = RetryingProvider::new(
        Provider::builder()
            .api_key("replay-key")
            .http_transport(transport.clone())
            .build()
            .unwrap(),
        RetryPolicy::new().base_delay(Duration::ZERO).jitter(false),
    );
    let result = match retrying
        .chat_stream(&ChatRequest::new(MODEL).user("hi"))
        .await
    {
        Ok(stream) => stream.collect_response().await,
        Err(err) => Err(err),
    };
    let err = result.unwrap_err();
    assert!(
        matches!(
            err,
            Error::Stream {
                kind: StreamErrorKind::Decode,
                ..
            }
        ),
        "{err:?}"
    );
    assert_eq!(transport.requests().len(), 1);
}

/// An endpoint that answers `307` to another host never gets the key or
/// the prompt sent there, on a call or a stream; the redirect is an
/// unexpected response naming the host.
#[tokio::test]
async fn a_redirect_to_another_host_is_never_followed() {
    for stream in [false, true] {
        let target = TestHttpServer::spawn([MockHttpResponse::json(StatusCode::OK, "{}")]).await;
        let location = format!(
            "{}/v1/messages",
            target.url().replace("127.0.0.1", "localhost")
        );
        let origin =
            TestHttpServer::spawn([MockHttpResponse::text(StatusCode::TEMPORARY_REDIRECT, "")
                .header("location", HeaderValue::try_from(location).unwrap())])
            .await;
        let provider = Provider::builder()
            .api_key("SECRET-KEY-123")
            .base_url(origin.url())
            .build()
            .unwrap();
        let request = ChatRequest::new(MODEL).user("hi");
        let err = if stream {
            provider.chat_stream(&request).await.err().unwrap()
        } else {
            provider.chat(&request).await.unwrap_err()
        };
        assert!(matches!(err, Error::UnexpectedResponse { .. }), "{err:?}");
        assert_eq!(err.status(), Some(307));
        assert!(err.to_string().contains("`localhost`"), "{err}");
        assert!(!err.is_retryable());
        assert!(target.recorded_requests().await.is_empty());
    }
}
