//! Extraction refusals through the Anthropic adapter: a `refusal`
//! stop reason fails extraction before parsing, on both paths, with the
//! explanation from `stop_details` as its text, and is never retried.

// The target type is only deserialized into, never read.
#![allow(dead_code)]

use allama::extract::{ExtractError, ExtractExt, Extractor, RefusalSource};
use allama::prelude::*;
use allama::schemars::{self, JsonSchema};
use allama::{Error, FinishReason};
use allama_anthropic::Provider;
use allama_http::{ReplayResponse, ReplayTransport};
use serde::Deserialize;
use serde_json::json;

const MODEL: &str = "claude-haiku-4-5-20251001";

/// The explanation in the `stop_details` of a refusal, as the decoder's
/// tests record the shape (`{"type": "refusal", "category", "explanation"}`).
const EXPLANATION: &str = "This request may relate to cyber attacks.";

#[derive(Debug, Deserialize, JsonSchema)]
struct Invoice {
    total: f64,
}

fn provider(response: ReplayResponse) -> (Provider, ReplayTransport) {
    let transport = ReplayTransport::new([response]);
    let provider = Provider::builder()
        .api_key("k")
        .http_transport(transport.clone())
        .build()
        .unwrap();
    (provider, transport)
}

fn stop_details() -> serde_json::Value {
    json!({"type": "refusal", "category": "cyber", "explanation": EXPLANATION})
}

fn assert_refused(error: Error) {
    let Error::Extract { failure, .. } = error else {
        panic!("an extraction failure expected: {error:?}");
    };
    assert!(
        matches!(
            failure.error(),
            ExtractError::Refused {
                finish_reason: Some(FinishReason::Refusal),
                text: Some(text),
                source: RefusalSource::Provider,
                ..
            } if text == EXPLANATION
        ),
        "{failure:?}"
    );
    assert_eq!(
        failure.to_string(),
        format!("anthropic: the model refused: {EXPLANATION}")
    );
}

#[tokio::test]
async fn a_refusal_fails_extraction_once() {
    let (provider, transport) = provider(ReplayResponse::json(
        200,
        json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": MODEL,
            "content": [], "stop_reason": "refusal", "stop_sequence": null,
            "stop_details": stop_details(),
            "usage": {"input_tokens": 12, "output_tokens": 0}
        })
        .to_string(),
    ));
    let error = Extractor::new()
        .retries(2)
        .extract::<_, Invoice>(&provider, &ChatRequest::new(MODEL).user("total?"))
        .await
        .unwrap_err();
    assert_eq!(transport.requests().len(), 1, "a refusal is never retried");
    assert_refused(error);
}

#[tokio::test]
async fn a_streamed_refusal_fails_extraction() {
    let delta = json!({"type": "message_delta",
        "delta": {"stop_reason": "refusal", "stop_sequence": null, "stop_details": stop_details()},
        "usage": {"output_tokens": 0}});
    let sse = format!(
        "event: message_start\ndata: {}\n\nevent: message_delta\ndata: {delta}\n\nevent: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n",
        json!({"type": "message_start", "message": {"id": "msg_1", "model": MODEL, "usage": {"input_tokens": 12}}})
    );
    let (provider, _) = provider(ReplayResponse::sse(sse));
    let mut stream = provider
        .extract_stream::<Invoice>(&ChatRequest::new(MODEL).user("total?"))
        .await
        .unwrap();
    while stream.next_event().await.is_some() {}
    assert_refused(stream.finish().unwrap_err());
}

/// An extraction from a conversation that ends on an assistant message, a
/// prefill, is refused by the API with structured output. The refusal
/// names that message, not the response format, so `Auto` does not fall
/// back to a forced tool call that would send the prefill again.
#[tokio::test]
async fn a_refused_prefill_fails_extraction_without_a_fallback() {
    let (provider, transport) = provider(ReplayResponse::json(
        400,
        json!({"type": "error", "error": {"type": "invalid_request_error", "message":
            "Your API request included an `assistant` message in the final position, which \
             would pre-fill the `assistant` response. When using output format, pre-filling \
             the `assistant` response is not supported."}})
        .to_string(),
    ));
    let request = ChatRequest::new(MODEL)
        .user("The invoice total is 12.")
        .assistant("Noted.");
    let error = provider.extract::<Invoice>(&request).await.unwrap_err();
    assert_eq!(transport.requests().len(), 1, "no fallback was tried");
    assert_eq!(
        error.unsupported_subject(),
        Some(&allama::Subject::Content {
            message: 1,
            block: None
        }),
        "{error:?}"
    );
}
