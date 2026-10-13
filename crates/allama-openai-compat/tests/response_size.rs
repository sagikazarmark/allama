//! A success body past the builder's
//! `max_response_size` fails the call as an unexpected response, which is
//! not retried, on every non-streaming call path.

use allama::prelude::*;
use allama::{EmbeddingProvider, EmbeddingRequest};
use allama_http::{ReplayResponse, ReplayTransport};
use allama_openai_compat::{Provider, ResponsesProvider};

const CHAT: &str = r#"{"id":"c1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#;
const EMBEDDING: &str = r#"{"object":"list","model":"m","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]}]}"#;
const RESPONSE: &str = r#"{"id":"resp_1","object":"response","status":"completed","model":"m","output":[{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"ok","annotations":[]}]}]}"#;

fn compat(body: &'static str, cap: usize) -> Provider {
    Provider::builder()
        .base_url("http://upstream/v1")
        .max_response_size(cap)
        .http_transport(ReplayTransport::new([ReplayResponse::json(200, body)]))
        .build()
        .unwrap()
}

fn responses(cap: usize) -> ResponsesProvider {
    ResponsesProvider::builder()
        .base_url("http://upstream/v1")
        .max_response_size(cap)
        .http_transport(ReplayTransport::new([ReplayResponse::json(200, RESPONSE)]))
        .build()
        .unwrap()
}

fn assert_too_large(err: &allama::Error) {
    assert!(
        matches!(err, allama::Error::UnexpectedResponse { .. }),
        "{err:?}"
    );
    assert!(!err.is_retryable(), "{err:?}");
}

#[tokio::test]
async fn chat_completions_bodies_are_capped() {
    let request = ChatRequest::new("m").user("hi");
    assert!(compat(CHAT, CHAT.len()).chat(&request).await.is_ok());
    let err = compat(CHAT, CHAT.len() - 1)
        .chat(&request)
        .await
        .unwrap_err();
    assert_too_large(&err);
}

#[tokio::test]
async fn embedding_bodies_are_capped() {
    let request = EmbeddingRequest::new("m").input("hi");
    assert!(
        compat(EMBEDDING, EMBEDDING.len())
            .embed(&request)
            .await
            .is_ok()
    );
    let err = compat(EMBEDDING, EMBEDDING.len() - 1)
        .embed(&request)
        .await
        .unwrap_err();
    assert_too_large(&err);
}

#[tokio::test]
async fn responses_bodies_are_capped() {
    let request = ChatRequest::new("m").user("hi");
    assert!(responses(RESPONSE.len()).chat(&request).await.is_ok());
    let err = responses(RESPONSE.len() - 1)
        .chat(&request)
        .await
        .unwrap_err();
    assert_too_large(&err);
}
