//! The Responses provider against a local HTTP server.

// Sends through the default `reqwest` transport.
#![cfg(feature = "reqwest")]

use std::time::Duration;

use allama::prelude::*;
use allama::{ContentPart, Image, ImageDetail, RequestField, StreamErrorKind, WarningKind};
use allama_conformance::{MockHttpResponse, TestHttpServer, contract};
use allama_http::http::{HeaderValue, StatusCode};
use allama_openai::ResponsesProvider;
use serde_json::json;

fn ok_body() -> serde_json::Value {
    json!({
        "id": "resp_1", "object": "response", "status": "completed", "model": "gpt-4.1-2025-04-14",
        "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "red", "annotations": []}]}],
        "usage": {"input_tokens": 12, "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens": 1, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 13}
    })
}

fn provider(server: &TestHttpServer) -> ResponsesProvider {
    ResponsesProvider::new_with_base_url("k", server.url()).unwrap()
}

fn api_error(status: u16, code: &str, kind: &str, message: &str) -> MockHttpResponse {
    MockHttpResponse::json_value(
        StatusCode::from_u16(status).unwrap(),
        &json!({"error": {"message": message, "type": kind, "code": code, "param": null}}),
    )
    .header("x-request-id", HeaderValue::from_static("req_err"))
    .header("retry-after-ms", HeaderValue::from_static("1500"))
}

#[tokio::test]
async fn responses_image_and_file_request_shape() {
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(StatusCode::OK, &ok_body())
        .header("x-request-id", HeaderValue::from_static("req_ok"))])
    .await;
    let response = provider(&server)
        .chat(&ChatRequest::new("gpt-4.1").user(vec![
            ContentPart::text("What is this?"),
            ContentPart::Image(Image::base64("image/png", "AAAA").with_detail(ImageDetail::Low)),
            ContentPart::file_base64("application/pdf", "JVBE"),
        ]))
        .await
        .unwrap();
    assert_eq!(response.text().as_deref(), Some("red"));
    assert_eq!(response.request_id.as_deref(), Some("req_ok"));
    assert_eq!(response.provider.as_deref(), Some("openai"));

    let sent = &server.recorded_requests().await[0];
    assert_eq!(sent.path, "/responses");
    assert_eq!(sent.header("authorization"), Some("Bearer k"));
    assert_eq!(sent.header("content-type"), Some("application/json"));
    let body = sent.body_json();
    assert_eq!(body["store"], false);
    assert!(body.get("stream").is_none());
    assert_eq!(
        body["input"][0]["content"],
        json!([
            {"type": "input_text", "text": "What is this?"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "low"},
            {"type": "input_file", "filename": "document.pdf", "file_data": "data:application/pdf;base64,JVBE"}
        ])
    );
}

#[tokio::test]
async fn responses_http_errors_are_typed() {
    let failures = [
        api_error(
            401,
            "invalid_api_key",
            "invalid_request_error",
            "Incorrect API key provided",
        ),
        api_error(429, "rate_limit_exceeded", "requests", "Rate limit reached"),
        api_error(
            429,
            "insufficient_quota",
            "insufficient_quota",
            "You exceeded your current quota",
        ),
        api_error(
            400,
            "context_length_exceeded",
            "invalid_request_error",
            "Your input exceeds the context window of this model.",
        ),
        MockHttpResponse::text(StatusCode::SERVICE_UNAVAILABLE, "upstream unavailable"),
    ];
    // The contracts call `chat` and then `chat_stream`, so their failures
    // (all but the rate limit) answer twice.
    let server = TestHttpServer::spawn(
        failures
            .into_iter()
            .enumerate()
            .flat_map(|(i, failure)| std::iter::repeat_n(failure, if i == 1 { 1 } else { 2 })),
    )
    .await;
    let provider = provider(&server);
    let request = ChatRequest::new("gpt-4.1").user("hi");
    let stream = |request| contract::collect_chat_stream(&provider, request);

    contract::assert_auth_error_contract(&provider, &request, stream).await;
    let rate = provider.chat(&request).await.unwrap_err();
    assert!(matches!(rate, Error::RateLimited { .. }));
    assert_eq!(rate.retry_after(), Some(Duration::from_millis(1500)));
    assert_eq!(rate.request_id(), Some("req_err"));
    assert_eq!(rate.status(), Some(429));
    contract::assert_quota_exhausted_contract(&provider, &request, stream).await;
    contract::assert_context_length_exceeded_contract(&provider, &request, stream).await;
    contract::assert_overloaded_error_contract(&provider, &request, stream).await;
}

#[tokio::test]
async fn responses_hints_warn_and_strict_policy_fails_before_sending() {
    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(StatusCode::OK, &ok_body()),
        MockHttpResponse::sse(include_str!("../fixtures/responses/text_stream.sse")),
    ])
    .await;
    let provider = provider(&server);
    let request = ChatRequest::new("gpt-4.1").user("hi").seed(7);
    contract::assert_unsupported_hint_warns(&provider, &request).await;
    contract::assert_strict_policy_rejects(&provider, &request).await;
    contract::assert_unsupported_hint_warns_stream(&provider, &request).await;
    contract::assert_strict_policy_rejects_stream(&provider, &request).await;
    let typo = ChatRequest::new("gpt-4.1")
        .user("hi")
        .with_options_json("openai-responses", json!({"servce_tier": "flex"}));
    assert!(matches!(
        provider.chat(&typo).await.unwrap_err(),
        Error::InvalidRequest { .. }
    ));
    let sent = server.recorded_requests().await;
    assert_eq!(sent.len(), 2, "only the warning requests may be sent");
    assert!(sent.iter().all(|r| r.body_json().get("seed").is_none()));
}

/// Sampling is sent to a reasoning model as asked (ADR 0015): OpenAI's
/// rejection, not a table in this crate, decides.
#[tokio::test]
async fn responses_reasoning_model_gets_sampling_as_asked() {
    let server =
        TestHttpServer::spawn([MockHttpResponse::json_value(StatusCode::OK, &ok_body())]).await;
    let response = provider(&server)
        .chat(&ChatRequest::new("gpt-5-nano").user("hi").temperature(0.2))
        .await
        .unwrap();
    assert!(response.warnings.is_empty(), "{:?}", response.warnings);
    let body = server.recorded_requests().await[0].body_json();
    assert!(body["temperature"].is_number());
}

/// The Responses API refuses a sampling parameter on a reasoning model with
/// `code: null` (the live reply for `temperature`), over HTTP and in a
/// stream's `error` event: both are the provider's refusal of the field.
#[tokio::test]
async fn responses_refusals_without_a_code_are_unsupported() {
    let message = "Unsupported parameter: 'temperature' is not supported with this model.";
    let refusal = json!({"error": {
        "message": message,
        "type": "invalid_request_error",
        "param": "temperature",
        "code": null
    }});
    let event = json!({"type": "error", "sequence_number": 1, "message": message,
                       "param": "temperature", "code": null});
    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(StatusCode::BAD_REQUEST, &refusal).header("x-request-id", HeaderValue::from_static("req_err")),
        MockHttpResponse::sse(format!(
            "event: response.created\n\
             data: {{\"type\":\"response.created\",\"response\":{{\"id\":\"resp_1\",\"status\":\"in_progress\",\"output\":[]}}}}\n\n\
             event: error\ndata: {event}\n\n"
        )),
    ])
    .await;
    let provider = provider(&server);
    let request = ChatRequest::new("gpt-5-nano").user("hi").temperature(0.2);

    let err = provider.chat(&request).await.unwrap_err();
    assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
    assert_eq!(err.unsupported_field(), Some(RequestField::Temperature));
    assert_eq!(
        err.unsupported_stage(),
        Some(allama::UnsupportedStage::Provider)
    );
    assert_eq!(err.status(), Some(400));
    assert_eq!(err.request_id(), Some("req_err"));
    assert_eq!(err.upstream_message(), Some(message));
    assert!(!err.is_retryable());

    let err = provider
        .chat_stream(&request)
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
    assert_eq!(err.unsupported_field(), Some(RequestField::Temperature));
    assert_eq!(
        err.unsupported_stage(),
        Some(allama::UnsupportedStage::Provider)
    );
}

/// Effort levels are wire format: a model with documented levels gets the
/// nearest one (with an `Approximated` warning). A model not known to
/// reason gets its effort as asked (ADR 0015).
#[tokio::test]
async fn reasoning_effort_maps_to_supported_levels() {
    let cases = [
        (
            "gpt-5-nano",
            ReasoningEffort::Low,
            json!({"effort": "low"}),
            vec![],
        ),
        (
            "gpt-5-nano",
            ReasoningEffort::None,
            json!({"effort": "minimal"}),
            vec![WarningKind::Approximated],
        ),
        (
            "gpt-5-nano",
            ReasoningEffort::XHigh,
            json!({"effort": "high"}),
            vec![WarningKind::Approximated],
        ),
        (
            "o4-mini",
            ReasoningEffort::Minimal,
            json!({"effort": "low"}),
            vec![WarningKind::Approximated],
        ),
        (
            "gpt-5.1",
            ReasoningEffort::None,
            json!({"effort": "none"}),
            vec![],
        ),
        (
            "gpt-4.1-nano",
            ReasoningEffort::Low,
            json!({"effort": "low"}),
            vec![],
        ),
        (
            "gpt-4.1-nano",
            ReasoningEffort::None,
            json!({"effort": "none"}),
            vec![],
        ),
    ];
    let server = TestHttpServer::spawn(
        cases
            .iter()
            .map(|_| MockHttpResponse::json_value(StatusCode::OK, &ok_body()))
            .collect::<Vec<_>>(),
    )
    .await;
    let provider = provider(&server);
    for (model, effort, _, warnings) in &cases {
        let response = provider
            .chat(&ChatRequest::new(*model).user("hi").reasoning(*effort))
            .await
            .unwrap();
        let kinds: Vec<_> = response.warnings.iter().map(|w| w.kind).collect();
        assert_eq!(&kinds, warnings, "{model} {effort:?}");
    }
    let sent = server.recorded_requests().await;
    for ((model, effort, expected, _), request) in cases.iter().zip(&sent) {
        assert_eq!(
            request.body_json()["reasoning"],
            *expected,
            "{model} {effort:?}"
        );
    }
}

#[tokio::test]
async fn tool_choice_variants() {
    let choices = [
        (ToolChoice::Auto, json!("auto")),
        (ToolChoice::Disabled, json!("none")),
        (ToolChoice::Required, json!("required")),
        (
            ToolChoice::Specific {
                name: "get_weather".into(),
            },
            json!({"type": "function", "name": "get_weather"}),
        ),
    ];
    let server = TestHttpServer::spawn(
        choices
            .iter()
            .map(|_| MockHttpResponse::json_value(StatusCode::OK, &ok_body()))
            .collect::<Vec<_>>(),
    )
    .await;
    let provider = provider(&server);
    let tool = ToolDefinition::new("get_weather", json!({"type": "object"}));
    for (choice, _) in &choices {
        let request = ChatRequest::new("gpt-4.1")
            .user("hi")
            .tools([tool.clone()])
            .tool_choice(choice.clone());
        let response = provider.chat(&request).await.unwrap();
        assert!(response.warnings.is_empty(), "{:?}", response.warnings);
    }
    let sent = server.recorded_requests().await;
    for ((_, expected), request) in choices.iter().zip(&sent) {
        assert_eq!(request.body_json()["tool_choice"], *expected);
    }
}

#[tokio::test]
async fn responses_mid_stream_error_is_typed() {
    let server = TestHttpServer::spawn([MockHttpResponse::sse(concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"status\":\"in_progress\",\"output\":[]}}\n\n",
        "event: error\n",
        "data: {\"type\":\"error\",\"code\":\"insufficient_quota\",\"message\":\"You exceeded your current quota\",\"param\":null}\n\n",
    ))
    .header("x-request-id", HeaderValue::from_static("req_sse"))])
    .await;
    let err = provider(&server)
        .chat_stream(&ChatRequest::new("gpt-4.1").user("hi"))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap_err();
    assert!(matches!(err, Error::QuotaExhausted { .. }), "{err:?}");
    assert_eq!(err.request_id(), Some("req_sse"));
    let body = server.recorded_requests().await[0].body_json();
    assert_eq!(body["stream"], true);
    assert_eq!(body["store"], false);
}

#[tokio::test]
async fn responses_clean_eof_without_terminal_event_is_incomplete() {
    let server = TestHttpServer::spawn([MockHttpResponse::sse(concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"status\":\"in_progress\",\"output\":[]}}\n\n",
    ))])
    .await;
    let err = provider(&server)
        .chat_stream(&ChatRequest::new("gpt-4.1").user("hi"))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Stream {
            kind: StreamErrorKind::Incomplete,
            ..
        }
    ));
    assert!(err.is_retryable());
}

/// The adapter runs end to end over an in-memory transport.
#[tokio::test]
async fn replay_transport_runs_the_adapter_without_a_server() {
    use allama::ChatStreamExt;
    use allama_http::{ReplayResponse, ReplayTransport};

    let sse = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/responses/citations_stream.sse"),
    )
    .unwrap();
    let transport = ReplayTransport::new([ReplayResponse::sse(sse)]);
    let provider = allama_openai::ResponsesProvider::builder()
        .api_key("replay-key")
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let response = provider
        .chat_stream(&allama::ChatRequest::new("gpt-4.1-mini").user("Latest Rust release?"))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();
    assert!(response.citations().count() > 0);
    let sent = &transport.requests()[0];
    assert_eq!(sent.uri.to_string(), "https://api.openai.com/v1/responses");
    assert_eq!(
        sent.headers.get("authorization"),
        Some(&HeaderValue::from_static("Bearer replay-key"))
    );
}

/// The builder's size cap reaches the decoder: an event past it fails the
/// stream with a stream protocol error, which retrying would only repeat.
#[tokio::test]
async fn stream_event_past_the_size_cap_fails() {
    use allama::ChatStreamExt;
    use allama_http::{ReplayResponse, ReplayTransport};

    let sse = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/responses/citations_stream.sse"),
    )
    .unwrap();
    let provider = allama_openai::ResponsesProvider::builder()
        .api_key("replay-key")
        .http_transport(ReplayTransport::new([ReplayResponse::sse(sse)]))
        .max_stream_event_size(128)
        .build()
        .unwrap();
    let err = provider
        .chat_stream(&allama::ChatRequest::new("gpt-4.1-mini").user("Latest Rust release?"))
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
            .contains("an SSE event passed the size cap of 128 bytes"),
        "{err}"
    );
}

/// Issue 22: a background stream that breaks off resumes from where it
/// stopped, and the two halves assemble into the uninterrupted response.
/// `background_stream.sse` is the recorded replay of a background response
/// (`GET /v1/responses/{id}?stream=true`, gpt-4.1-nano).
#[tokio::test]
async fn interrupted_background_streams_resume_without_repeating_events() {
    use allama::ChatStreamExt;
    use allama_http::{ReplayResponse, ReplayTransport};
    use allama_openai::StreamPosition;
    use futures_util::StreamExt;

    let sse = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/responses/background_stream.sse"),
    )
    .unwrap();
    let cut: String = sse.split_inclusive("\n\n").take(20).collect();
    let transport = ReplayTransport::new([
        ReplayResponse::sse(&sse),
        ReplayResponse::sse(cut),
        ReplayResponse::sse(&sse),
    ]);
    let provider = allama_openai::ResponsesProvider::builder()
        .api_key("replay-key")
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let request = allama::ChatRequest::new("gpt-4.1-nano").user("Count from 1 to 30.");

    let whole = provider
        .chat_stream(&request)
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();

    let first: Vec<_> = provider
        .chat_stream(&request)
        .await
        .unwrap()
        .collect()
        .await;
    let error = first.last().unwrap().as_ref().unwrap_err();
    let position = StreamPosition::from_error(error).expect("the error records where it stopped");
    assert_eq!(Some(position.response_id.as_str()), whole.id.as_deref());

    let rest: Vec<_> = provider
        .resume_stream(&position.response_id, Some(position.sequence_number))
        .await
        .unwrap()
        .collect()
        .await;
    let stitched = futures_util::stream::iter(first.into_iter().filter(Result::is_ok).chain(rest))
        .collect_response()
        .await
        .unwrap();
    assert_eq!(stitched.text(), whole.text());
    assert_eq!(stitched.content, whole.content);

    let resumed = &transport.requests()[2];
    assert_eq!(
        resumed.uri.to_string(),
        format!(
            "https://api.openai.com/v1/responses/{}?stream=true",
            position.response_id
        )
    );
    assert_eq!(resumed.method, allama_http::http::Method::GET);
}

/// The provider-specific continuation methods stay reachable
/// through wrappers and type erasure.
#[tokio::test]
async fn retrieve_response_through_wrappers_and_erasure() {
    let server =
        TestHttpServer::spawn([MockHttpResponse::json_value(StatusCode::OK, &ok_body())]).await;
    let erased = allama::DynChatProvider::new(allama::wrappers::RetryingProvider::new(
        provider(&server),
        allama::wrappers::RetryPolicy::default(),
    ));

    let responses = erased
        .downcast_ref::<ResponsesProvider>()
        .expect("the Responses provider is reachable");
    // The compat provider it is built on is an implementation detail.
    assert!(
        erased
            .downcast_ref::<allama_openai_compat::ResponsesProvider>()
            .is_none()
    );
    let response = responses.retrieve_response("resp_1").await.unwrap();
    assert_eq!(response.text().as_deref(), Some("red"));

    let sent = &server.recorded_requests().await[0];
    assert_eq!(sent.method, "GET");
    assert_eq!(sent.path, "/responses/resp_1");
}

#[tokio::test]
async fn responses_http_errors_carry_the_server_retry_directive() {
    let server = TestHttpServer::spawn([api_error(
        500,
        "server_error",
        "server_error",
        "The server had an error",
    )
    .header("x-should-retry", HeaderValue::from_static("false"))])
    .await;
    let err = provider(&server)
        .chat(&ChatRequest::new("gpt-4.1").user("hi"))
        .await
        .unwrap_err();
    assert!(err.is_retryable(), "{err:?}");
    assert_eq!(err.server_retry_directive(), Some(false));
    assert_eq!(err.request_id(), Some("req_err"));
    assert!(
        !allama::wrappers::RetryPolicy::new().should_retry(&err),
        "the default policy follows the directive"
    );
}
