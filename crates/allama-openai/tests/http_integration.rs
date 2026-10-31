// Sends through the default `reqwest` transport.
#![cfg(feature = "reqwest")]

use allama::prelude::*;
use allama::{ContentPart, Image, ImageDetail, StreamErrorKind};
use allama_conformance::{MockHttpResponse, TestHttpServer};
use allama_http::http::{HeaderValue, StatusCode};
use allama_openai::Provider;
use serde_json::json;

#[tokio::test]
async fn vision_and_pdf_request_shape() {
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(
        StatusCode::OK,
        &json!({"id": "c", "model": "gpt-4.1", "choices": [{"index": 0, "finish_reason": "stop", "message": {"content": "red"}}]}),
    )])
    .await;
    let provider = Provider::new_with_base_url("k", server.url()).unwrap();
    provider
        .chat(&ChatRequest::new("gpt-4.1").user(vec![
            ContentPart::text("What is this?"),
            ContentPart::Image(Image::base64("image/png", "AAAA").with_detail(ImageDetail::Low)),
            ContentPart::file_base64("application/pdf", "JVBE"),
        ]))
        .await
        .unwrap();
    let body = server.recorded_requests().await[0].body_json();
    let parts = &body["messages"][0]["content"];
    assert_eq!(
        parts[1],
        json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "low"}})
    );
    assert_eq!(parts[2]["type"], "file");
    assert!(body.get("max_tokens").is_none());
}

#[tokio::test]
async fn clean_eof_without_done_is_incomplete() {
    let server = TestHttpServer::spawn([MockHttpResponse::sse(
        "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n",
    )])
    .await;
    let provider = Provider::new_with_base_url("k", server.url()).unwrap();
    let err = provider
        .chat_stream(&ChatRequest::new("gpt-4o").user("hi"))
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

#[tokio::test]
async fn embed_maps_401_to_auth() {
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(
        StatusCode::UNAUTHORIZED,
        &json!({"error": {"message": "bad key", "type": "invalid_request_error", "code": "invalid_api_key"}}),
    )])
    .await;
    let provider = Provider::new_with_base_url("k", server.url()).unwrap();
    let err = provider
        .embed(&EmbeddingRequest::new("text-embedding-3-small").input("x"))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Auth { .. }));
}

#[tokio::test]
async fn raw_responses_from_both_apis() {
    let completion = json!({"id": "c", "model": "gpt-4.1", "system_fingerprint": "fp_1",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"content": "red"}}]});
    let response = json!({"id": "resp_1", "object": "response", "status": "completed", "model": "gpt-4.1",
        "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "red", "annotations": []}]}]});
    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(StatusCode::OK, &completion),
        MockHttpResponse::json_value(StatusCode::OK, &response),
    ])
    .await;

    let chat = Provider::new_with_base_url("k", server.url())
        .unwrap()
        .chat(
            &ChatRequest::new("gpt-4.1")
                .user("Color?")
                .with_options_json("openai", json!({"raw_response": true})),
        )
        .await
        .unwrap();
    let raw = chat
        .provider_data
        .get::<allama_openai::RawResponse>()
        .unwrap()
        .unwrap();
    assert_eq!(raw.0["system_fingerprint"], "fp_1");

    let responses = allama_openai::ResponsesProvider::new_with_base_url("k", server.url())
        .unwrap()
        .chat(
            &ChatRequest::new("gpt-4.1")
                .user("Color?")
                .with_options_json("openai-responses", json!({"raw_response": true})),
        )
        .await
        .unwrap();
    assert_eq!(
        responses
            .provider_data
            .get::<allama_openai::RawResponse>()
            .unwrap()
            .unwrap()
            .0,
        response
    );
    let requests = server.recorded_requests().await;
    assert!(
        requests
            .iter()
            .all(|r| r.body_json().get("raw_response").is_none())
    );
}

fn chat_ok_body() -> serde_json::Value {
    json!({"id": "c", "model": "gpt-4.1", "choices": [{"index": 0, "finish_reason": "stop", "message": {"content": "ok"}}]})
}

/// `x-should-retry: true` makes the default retry policy retry a 409, which
/// it would not retry by status.
#[tokio::test]
async fn server_retry_directive_true_retries_a_conflict() {
    use std::time::Duration;

    use allama::wrappers::{RetryPolicy, RetryingProvider};

    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(
            StatusCode::CONFLICT,
            &json!({"error": {"message": "conflict", "type": "invalid_request_error"}}),
        )
        .header("x-should-retry", HeaderValue::from_static("true"))
        .header("x-request-id", HeaderValue::from_static("req_409")),
        MockHttpResponse::json_value(StatusCode::OK, &chat_ok_body()),
    ])
    .await;
    let provider = Provider::new_with_base_url("k", server.url()).unwrap();
    let request = ChatRequest::new("gpt-4.1").user("hi");

    let err = provider.chat(&request).await.unwrap_err();
    assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
    assert_eq!(err.server_retry_directive(), Some(true));
    assert_eq!(err.request_id(), Some("req_409"));
    assert!(!err.is_retryable());

    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(
            StatusCode::CONFLICT,
            &json!({"error": {"message": "conflict", "type": "invalid_request_error"}}),
        )
        .header("x-should-retry", HeaderValue::from_static("true")),
        MockHttpResponse::json_value(StatusCode::OK, &chat_ok_body()),
    ])
    .await;
    let retrying = RetryingProvider::new(
        Provider::new_with_base_url("k", server.url()).unwrap(),
        RetryPolicy::new().base_delay(Duration::ZERO),
    );
    let response = retrying.chat(&request).await.unwrap();
    assert_eq!(response.text().as_deref(), Some("ok"));
    assert_eq!(server.recorded_requests().await.len(), 2);
}

/// `x-should-retry: false` stops the default retry policy on a 429, even
/// with a retry hint it would wait.
#[tokio::test]
async fn server_retry_directive_false_stops_a_rate_limit() {
    use std::time::Duration;

    use allama::wrappers::{RetryPolicy, RetryingProvider};

    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(
            StatusCode::TOO_MANY_REQUESTS,
            &json!({"error": {"message": "slow down", "type": "requests", "code": "rate_limit_exceeded"}}),
        )
        .header("x-should-retry", HeaderValue::from_static("false"))
        .header("retry-after-ms", HeaderValue::from_static("1")),
        MockHttpResponse::json_value(StatusCode::OK, &chat_ok_body()),
    ])
    .await;
    let retrying = RetryingProvider::new(
        Provider::new_with_base_url("k", server.url()).unwrap(),
        RetryPolicy::new().base_delay(Duration::ZERO),
    );
    let err = retrying
        .chat(&ChatRequest::new("gpt-4.1").user("hi"))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::RateLimited { .. }), "{err:?}");
    assert_eq!(err.server_retry_directive(), Some(false));
    assert_eq!(err.retry_after(), Some(Duration::from_millis(1)));
    assert_eq!(server.recorded_requests().await.len(), 1);
}
