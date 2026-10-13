//! Provider options given as the typed struct or as JSON produce identical
//! requests, and survive serializing the request.

// Sends through the default `reqwest` transport.
#![cfg(feature = "reqwest")]

use allama::{ChatProvider, ChatRequest};
use allama_anthropic::{ChatRequestOptions, Provider};
use allama_conformance::{MockHttpResponse, TestHttpServer};
use serde_json::{Value, json};

fn ok() -> MockHttpResponse {
    MockHttpResponse::json_value(
        200,
        &json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
            "stop_sequence": null, "usage": {"input_tokens": 3, "output_tokens": 1}
        }),
    )
}

fn base() -> ChatRequest {
    // `top_k` is portable: it reaches the wire without an option.
    ChatRequest::new("claude-haiku-4-5-20251001")
        .user("hi")
        .top_k(20)
}

#[tokio::test]
async fn typed_json_and_serialized_options_send_the_same_body() {
    let typed = base().with_option(
        ChatRequestOptions::new()
            .service_tier("standard_only")
            .extra("container", "container_1"),
    );
    let json = base().with_options_json(
        "anthropic",
        json!({"service_tier": "standard_only", "extra": {"container": "container_1"}}),
    );
    let reloaded: ChatRequest =
        serde_json::from_str(&serde_json::to_string(&typed).unwrap()).unwrap();
    assert_eq!(
        reloaded, json,
        "a serialized typed option reloads as its JSON form"
    );

    let server = TestHttpServer::spawn([ok(), ok(), ok()]).await;
    let provider = Provider::new_with_base_url("test-key", server.url()).unwrap();
    for request in [&typed, &json, &reloaded] {
        provider.chat(request).await.unwrap();
    }
    let bodies: Vec<Value> = server
        .recorded_requests()
        .await
        .iter()
        .map(|r| r.body_json())
        .collect();
    assert_eq!(bodies[0]["top_k"], 20);
    assert_eq!(bodies[0]["service_tier"], "standard_only");
    assert_eq!(bodies[0]["container"], "container_1");
    assert_eq!(bodies[0], bodies[1]);
    assert_eq!(bodies[0], bodies[2]);
}

#[tokio::test]
async fn unknown_json_fields_fail_before_sending() {
    let server = TestHttpServer::spawn([]).await;
    let provider = Provider::new_with_base_url("test-key", server.url()).unwrap();
    let typo = base().with_options_json("anthropic", json!({"topk": 20}));
    assert!(matches!(
        provider.chat(&typo).await.unwrap_err(),
        allama::Error::InvalidRequest { .. }
    ));
    assert!(server.recorded_requests().await.is_empty());
}
