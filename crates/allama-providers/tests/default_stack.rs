//! The default stack against a real adapter and HTTP responses: the retry
//! rules a provider from a lookup follows.
#![cfg(feature = "openai-compat")]

use allama::prelude::*;
use allama_conformance::{MockHttpResponse, TestHttpServer};
use allama_openai_compat::providers::Vllm;
use allama_providers::{ProviderSettings, Registry};
use serde_json::json;

fn completion() -> MockHttpResponse {
    MockHttpResponse::json_value(
        200,
        &json!({
            "id": "c1", "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        }),
    )
}

fn error(status: u16) -> MockHttpResponse {
    MockHttpResponse::json_value(
        status,
        &json!({"error": {"message": "try again", "type": "conflict"}}),
    )
}

/// A provider for `server`, registered the way an application registers
/// its own, and looked up with the default settings.
fn provider(server: &TestHttpServer) -> DynChatProvider {
    let url = server.url().to_owned();
    let mut registry = Registry::new();
    registry.register_chat("local", move || Vllm::builder(&url, None)?.build());
    let (provider, _) = registry
        .chat_from_model_with("local:m", &ProviderSettings::default())
        .unwrap();
    provider
}

fn request() -> ChatRequest {
    ChatRequest::new("m").user("hi")
}

#[tokio::test]
async fn a_transient_error_is_retried() {
    let server = TestHttpServer::spawn([error(503), completion()]).await;

    let response = provider(&server).chat(&request()).await.unwrap();

    assert_eq!(response.text_or_empty(), "ok");
    assert_eq!(server.recorded_requests().await.len(), 2);
}

/// `409` is not retried by default, unlike in the official OpenAI and
/// Anthropic SDKs; `x-should-retry: true` retries it.
#[tokio::test]
async fn x_should_retry_true_retries_a_conflict() {
    let server = TestHttpServer::spawn([error(409)]).await;
    let err = provider(&server).chat(&request()).await.unwrap_err();
    assert_eq!(err.status(), Some(409));
    assert_eq!(server.recorded_requests().await.len(), 1);

    let server =
        TestHttpServer::spawn([error(409).header("x-should-retry", "true"), completion()]).await;
    let response = provider(&server).chat(&request()).await.unwrap();
    assert_eq!(response.text_or_empty(), "ok");
    assert_eq!(server.recorded_requests().await.len(), 2);
}

#[tokio::test]
async fn x_should_retry_false_stops_a_rate_limit() {
    let server =
        TestHttpServer::spawn([error(429).header("x-should-retry", "false"), completion()]).await;

    let err = provider(&server).chat(&request()).await.unwrap_err();

    assert!(matches!(err, Error::RateLimited { .. }), "{err:?}");
    assert_eq!(server.recorded_requests().await.len(), 1);
}

/// An adapter built outside any registry gets the same stack from `wrap`.
#[tokio::test]
async fn a_wrapped_adapter_retries_a_transient_error() {
    let server = TestHttpServer::spawn([error(503), completion()]).await;
    let adapter = Vllm::builder(server.url(), None).unwrap().build().unwrap();
    let provider = allama_providers::chat::wrap(adapter, &ProviderSettings::default()).unwrap();

    let response = provider.chat(&request()).await.unwrap();

    assert_eq!(response.text_or_empty(), "ok");
    assert_eq!(server.recorded_requests().await.len(), 2);
}

/// A provider looked up by model address answers which namespaces it reads
/// through the default stack, so a stored conversation can be moved to it
/// without knowing its namespace, which differs from its name.
#[tokio::test]
async fn a_provider_from_a_model_address_keeps_its_own_state() {
    use allama::history::retain_provider_state_for;
    use allama::{ProviderData, Reasoning};
    use allama_openai_compat::providers::Vllm;

    let server = TestHttpServer::spawn([]).await;
    let provider = provider(&server);
    assert_eq!(provider.provider_name(), Vllm::PROVIDER_NAME);
    assert!(provider.reads_namespace(Vllm::NAMESPACE));
    assert!(!provider.reads_namespace(Vllm::PROVIDER_NAME));

    let signed =
        |namespace: &str| {
            ContentBlock::from(Reasoning::new("…").with_provider_data(
                ProviderData::new().with_raw(namespace, "signature", json!("sig")),
            ))
        };
    let mut messages = vec![
        Message::user("Hi"),
        Message::assistant_blocks(vec![
            signed(Vllm::NAMESPACE),
            signed("anthropic"),
            ContentBlock::text("Hello"),
        ]),
    ];
    let warnings = retain_provider_state_for(&mut messages, &provider);

    assert_eq!(warnings.len(), 1, "{warnings:?}");
    let kept = &messages[1].as_assistant().unwrap().content;
    assert_eq!(kept.len(), 2);
    assert!(kept[0].provider_data().namespace(Vllm::NAMESPACE).is_some());
}
