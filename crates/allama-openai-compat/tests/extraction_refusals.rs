//! Extraction refusals through both dialects: a refusal the model
//! writes (Chat Completions `message.refusal` / `delta.refusal`, the
//! Responses `refusal` part) finishes with `Refusal`, which fails
//! extraction before parsing, on both paths, and is never retried. The
//! shapes are OpenAI's documented refusal fields
//! (<https://platform.openai.com/docs/guides/structured-outputs#refusals>).

// The target type is only deserialized into, never read.
#![allow(dead_code)]

use allama::chat::extract::{
    ChatExtractError, ChatExtractExt, ChatExtractionMode, ChatExtractor, RefusalSource,
};
use allama::prelude::*;
use allama::schemars::{self, JsonSchema};
use allama::{Error, FinishReason};
use allama_http::{ReplayResponse, ReplayTransport};
use allama_openai_compat::{Provider, ResponsesProvider};
use serde::Deserialize;
use serde_json::json;

const REFUSAL: &str = "I'm sorry, I can't help with that.";

#[derive(Debug, Deserialize, JsonSchema)]
struct Invoice {
    total: f64,
}

/// Native mode: a generic endpoint reports no structured-output support,
/// so `Auto` has nothing to pick.
fn request() -> ChatRequest {
    ChatRequest::new("m")
        .user("total?")
        .with_option(ChatExtractionMode::Native)
}

fn chat_completions(response: ReplayResponse) -> (Provider, ReplayTransport) {
    let transport = ReplayTransport::new([response]);
    let provider = Provider::builder()
        .base_url("http://upstream/v1")
        .provider_name("vendor")
        .http_transport(transport.clone())
        .build()
        .unwrap();
    (provider, transport)
}

fn responses(response: ReplayResponse) -> (ResponsesProvider, ReplayTransport) {
    let transport = ReplayTransport::new([response]);
    let provider = ResponsesProvider::builder()
        .base_url("http://upstream/v1")
        .provider_name("vendor")
        .http_transport(transport.clone())
        .build()
        .unwrap();
    (provider, transport)
}

/// The refusal fails extraction with the text the model wrote, read from
/// its `Refusal` block.
fn assert_refused(error: Error) {
    let Error::Extract { failure, .. } = error else {
        panic!("an extraction failure expected: {error:?}");
    };
    assert!(
        matches!(
            failure.error(),
            ChatExtractError::Refused {
                finish_reason: Some(FinishReason::Refusal),
                source: RefusalSource::Model,
                text: Some(text),
                ..
            } if text == REFUSAL
        ),
        "{failure:?}"
    );
}

#[tokio::test]
async fn chat_completions_refusals_fail_extraction_once() {
    let (provider, transport) = chat_completions(ReplayResponse::json(
        200,
        json!({
            "id": "c1", "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": null, "refusal": REFUSAL}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14}
        })
        .to_string(),
    ));
    let error = ChatExtractor::new()
        .retries(2)
        .extract::<_, Invoice>(&provider, &request())
        .await
        .unwrap_err();
    assert_eq!(transport.requests().len(), 1, "a refusal is never retried");
    assert_refused(error);

    let chunk = |delta: serde_json::Value, finish: Option<&str>| json!({"id": "c1", "model": "m", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
    let sse = format!(
        "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
        chunk(json!({"role": "assistant", "refusal": REFUSAL}), None),
        chunk(json!({}), Some("stop")),
    );
    let (provider, _) = chat_completions(ReplayResponse::sse(sse));
    let mut stream = provider
        .extract_stream::<Invoice>(&request())
        .await
        .unwrap();
    while stream.next_event().await.is_some() {}
    assert_refused(stream.finish().unwrap_err());
}

#[tokio::test]
async fn responses_refusals_fail_extraction_once() {
    let message = json!({"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
        "content": [{"type": "refusal", "refusal": REFUSAL}]});
    let (provider, transport) = responses(ReplayResponse::json(
        200,
        json!({"id": "resp_1", "model": "m", "status": "completed", "output": [message]})
            .to_string(),
    ));
    let error = ChatExtractor::new()
        .retries(2)
        .extract::<_, Invoice>(&provider, &request())
        .await
        .unwrap_err();
    assert_eq!(transport.requests().len(), 1, "a refusal is never retried");
    assert_refused(error);

    let events = [
        json!({"type": "response.output_item.added", "output_index": 0,
            "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": []}}),
        json!({"type": "response.content_part.added", "output_index": 0, "content_index": 0,
            "part": {"type": "refusal", "refusal": ""}}),
        json!({"type": "response.refusal.delta", "output_index": 0, "content_index": 0,
            "delta": REFUSAL}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": message}),
        json!({"type": "response.completed",
            "response": {"id": "resp_1", "model": "m", "status": "completed", "output": [message]}}),
    ];
    let sse: String = events
        .iter()
        .map(|event| {
            format!(
                "event: {}\ndata: {event}\n\n",
                event["type"].as_str().unwrap()
            )
        })
        .collect();
    let (provider, _) = responses(ReplayResponse::sse(sse));
    let mut stream = provider
        .extract_stream::<Invoice>(&request())
        .await
        .unwrap();
    while stream.next_event().await.is_some() {}
    assert_refused(stream.finish().unwrap_err());
}
