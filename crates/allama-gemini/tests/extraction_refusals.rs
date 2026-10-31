//! Extraction refusals through the Gemini adapter: a safety finish
//! fails extraction before parsing, on both paths, with the
//! `finishMessage` as its text, and is never retried.

// The target type is only deserialized into, never read.
#![allow(dead_code)]

use allama::chat::extract::{ChatExtractError, ChatExtractExt, ChatExtractor, RefusalSource};
use allama::prelude::*;
use allama::schemars::{self, JsonSchema};
use allama::{Error, FinishReason};
use allama_gemini::Provider;
use allama_http::http::StatusCode;
use allama_http::{ReplayResponse, ReplayTransport};
use serde::Deserialize;
use serde_json::json;

const MODEL: &str = "gemini-2.5-flash";

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

fn assert_refused(error: Error, explanation: &str) {
    let Error::Extract { failure, .. } = error else {
        panic!("an extraction failure expected: {error:?}");
    };
    assert!(
        matches!(
            failure.error(),
            ChatExtractError::Refused {
                finish_reason: Some(FinishReason::ContentFilter),
                text: Some(text),
                source: RefusalSource::Provider,
                ..
            } if text == explanation
        ),
        "{failure:?}"
    );
}

#[tokio::test]
async fn a_safety_finish_fails_extraction_once() {
    let explanation = "The response was blocked for safety.";
    let (provider, transport) = provider(ReplayResponse::json(
        StatusCode::OK,
        json!({
            "candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "SAFETY",
                "finishMessage": explanation, "index": 0}],
            "usageMetadata": {"promptTokenCount": 9, "totalTokenCount": 9},
            "modelVersion": MODEL
        })
        .to_string(),
    ));
    let error = ChatExtractor::new()
        .retries(2)
        .extract::<_, Invoice>(&provider, &ChatRequest::new(MODEL).user("total?"))
        .await
        .unwrap_err();
    assert_eq!(transport.requests().len(), 1, "a refusal is never retried");
    assert_refused(error, explanation);
}

/// The recorded stream (`fixtures/stream.sse`) ends with `RECITATION` and
/// Gemini's own `finishMessage`.
#[tokio::test]
async fn a_streamed_recitation_finish_fails_extraction() {
    let (provider, _) = provider(ReplayResponse::sse(include_str!("../fixtures/stream.sse")));
    let mut stream = provider
        .extract_stream::<Invoice>(&ChatRequest::new(MODEL).user("total?"))
        .await
        .unwrap();
    while stream.next_event().await.is_some() {}
    assert_refused(
        stream.finish().unwrap_err(),
        "The generated content was filtered because it may contain material that resembles \
         existing copyrighted works. Try rephrasing the prompt. If you think this was an error, \
         [send feedback](https://ai.google.dev/gemini-api/docs/troubleshooting).",
    );
}
