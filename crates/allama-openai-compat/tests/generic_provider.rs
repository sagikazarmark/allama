//! The generic provider end to end, as a custom endpoint uses it: request
//! translation with typed options, stream and response decoding, warnings,
//! request ids and error mapping.

// Sends through the default `reqwest` transport (a local test server).
#![cfg(feature = "reqwest")]

use allama::prelude::*;
use allama::{ContentPart, FinishReason, UnsupportedPolicy, WarningKind};
use allama_conformance::{MockHttpResponse, TestHttpServer};
use allama_http::http::{HeaderValue, StatusCode};
use allama_openai_compat::{ChatRequestOptions, CompatQuirks, Provider, SeedField};
use serde_json::json;

fn request() -> ChatRequest {
    ChatRequest::new("custom-chat")
        .system("You are a custom provider.")
        .user(vec![
            ContentPart::text("Look at this image and search the docs."),
            ContentPart::image_url("https://example.com/cat.png"),
        ])
        .tools(vec![
            ToolDefinition::new(
                "search_docs",
                json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}),
            )
            .description("Search internal docs")
            .strict(true),
        ])
        .tool_choice(ToolChoice::Required)
        .seed(123)
        .parallel_tool_calls(true)
}

#[tokio::test]
async fn generic_provider_streams_with_typed_options() {
    let server = TestHttpServer::spawn([MockHttpResponse::sse(
        [
            r#"{"id":"c","choices":[{"index":0,"delta":{"content":"Hi"}}]}"#,
            r#"{"id":"c","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            "[DONE]",
        ]
        .map(|data| format!("data: {data}\n\n"))
        .concat(),
    )])
    .await;
    let provider = Provider::builder()
        .base_url(server.url())
        .bearer_token("token")
        .provider_name("custom")
        .build()
        .unwrap();
    let options = ChatRequestOptions::new()
        .user("user-42")
        .service_tier("priority")
        .extra("top_k", 5);
    let response = provider
        .chat_stream(&request().with_option(options))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();
    assert_eq!(response.text().as_deref(), Some("Hi"));
    assert_eq!(response.finish_reason, Some(FinishReason::Stop));
    // A custom endpoint's strict mode is not verified.
    let kinds: Vec<_> = response.warnings.iter().map(|w| w.kind).collect();
    assert_eq!(kinds, [WarningKind::Approximated]);

    let body = server.recorded_requests().await[0].body_json();
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][1]["content"][1]["type"], "image_url");
    assert_eq!(body["tools"][0]["function"]["strict"], true);
    assert_eq!(body["tool_choice"], "required");
    assert_eq!(body["seed"], 123);
    assert_eq!(body["user"], "user-42");
    assert_eq!(body["service_tier"], "priority");
    assert_eq!(body["top_k"], 5);
    assert_eq!(body["stream_options"]["include_usage"], true);
}

#[tokio::test]
async fn generic_provider_honors_options_warnings_and_request_ids() {
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(
        StatusCode::OK,
        &json!({
            "id": "chatcmpl-1",
            "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hello"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
        }),
    )
    .header("x-request-id", HeaderValue::from_static("req_abc"))])
    .await;
    let mut quirks = CompatQuirks::new("openai-compat:custom");
    quirks.seed_field = SeedField::Unsupported;
    let provider = Provider::builder()
        .base_url(server.url())
        .bearer_token("token")
        .provider_name("custom")
        .quirks(quirks)
        .build()
        .unwrap();

    let request = ChatRequest::new("m").user("hi").seed(9).with_options_json(
        "openai-compat",
        json!({"service_tier": "flex", "extra": {"top_k": 3}}),
    );
    let response = provider.chat(&request).await.unwrap();
    assert_eq!(response.text().as_deref(), Some("hello"));
    assert_eq!(response.provider.as_deref(), Some("custom"));
    assert_eq!(response.request_id.as_deref(), Some("req_abc"));
    assert_eq!(response.warnings.len(), 1);
    assert_eq!(response.warnings[0].kind, WarningKind::Unsupported);

    let sent = &server.recorded_requests().await[0];
    let body = sent.body_json();
    assert_eq!(body["service_tier"], "flex");
    assert_eq!(body["top_k"], 3);
    assert!(body.get("seed").is_none());
    assert_eq!(sent.header("authorization"), Some("Bearer token"));

    // Strict policy rejects before sending; typed options take precedence.
    let strict = request.clone().unsupported_policy(UnsupportedPolicy::Error);
    assert!(matches!(
        provider.chat(&strict).await.unwrap_err(),
        Error::Unsupported { .. }
    ));
    assert_eq!(server.recorded_requests().await.len(), 1);
    let typed = ChatRequest::new("m").with_option(ChatRequestOptions::new().user("u"));
    let bad = typed.with_options_json("openai-compat", json!({"usr": "typo"}));
    assert_eq!(
        bad.resolve_option::<ChatRequestOptions>()
            .unwrap()
            .user
            .as_deref(),
        Some("u")
    );
    let typo = ChatRequest::new("m").with_options_json("openai-compat", json!({"usr": "typo"}));
    assert!(matches!(
        provider.chat(&typo).await.unwrap_err(),
        Error::InvalidRequest { .. }
    ));
}

#[tokio::test]
async fn generic_provider_maps_quota_errors() {
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(
        StatusCode::TOO_MANY_REQUESTS,
        &json!({"error": {"message": "You exceeded your current quota", "type": "insufficient_quota", "code": "insufficient_quota"}}),
    )])
    .await;
    let provider = Provider::builder()
        .base_url(server.url())
        .bearer_token("t")
        .build()
        .unwrap();
    let err = provider
        .chat(&ChatRequest::new("m").user("hi"))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::QuotaExhausted { .. }));
    assert!(!err.is_retryable());
}
