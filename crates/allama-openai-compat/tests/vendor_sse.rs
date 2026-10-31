//! Vendor-quirk SSE corpus (`fixtures/vendor_sse`): named events, comments
//! and `retry:` lines, an empty first chunk, and citations in top-level
//! chunk fields. Each fixture pins the decoded events and the collected
//! response, and runs the unknown-frame contract (C1).
//!
//! Regenerate the expected files with `ALLAMA_UPDATE_FIXTURES=1` and review
//! the diff.

use std::path::PathBuf;

use allama::chat::stream::StreamCollector;
use allama::{
    ChatProvider, ChatRequest, ChatStream, ChatStreamExt, Error, StreamEvent, WarningKind,
};
use allama_conformance::stream_frames::{StreamFraming, assert_unknown_stream_frames_surfaced};
use allama_conformance::{FixtureDir, assert_stream_fixture_eq, load_text_fixture};
use allama_http::http::StatusCode;
use allama_http::{ReplayResponse, ReplayTransport};
use allama_openai_compat::{CompatQuirks, Provider};
use futures_util::StreamExt;
use serde_json::json;

const NAMESPACE: &str = "openai-compat:vendor";

const CORPUS: &[&str] = &[
    "named_event_first",
    "comments_and_retry",
    "empty_first_chunk",
    "citations_tail",
];

fn fixtures() -> FixtureDir {
    FixtureDir::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/vendor_sse"))
}

fn sse(name: &str) -> String {
    load_text_fixture(&fixtures(), &format!("{name}.sse"))
}

fn provider_with(quirks: Option<CompatQuirks>, body: impl Into<String>) -> Provider {
    let mut builder = Provider::builder()
        .base_url("http://upstream/v1")
        .provider_name("vendor")
        .http_transport(ReplayTransport::new([ReplayResponse::sse(body.into())]));
    if let Some(quirks) = quirks {
        builder = builder.quirks(quirks);
    }
    builder.build().unwrap()
}

async fn stream(body: impl Into<String>) -> allama::Result<ChatStream> {
    provider_with(None, body)
        .chat_stream(&ChatRequest::new("m").user("q"))
        .await
}

#[tokio::test]
async fn corpus_matches_fixtures() {
    for name in CORPUS {
        assert_stream_fixture_eq(
            stream(sse(name)).await.unwrap(),
            &fixtures(),
            &format!("{name}.events.json"),
            &format!("{name}.response.json"),
        )
        .await;
    }
}

#[tokio::test]
async fn corpus_surfaces_unknown_frames() {
    for name in CORPUS {
        assert_unknown_stream_frames_surfaced(&sse(name), &StreamFraming::sse(), stream).await;
    }
}

/// A named vendor event is not decoded as a Chat Completions chunk, so it
/// does not become the response start.
#[tokio::test]
async fn named_event_does_not_become_the_response_start() {
    let events: Vec<_> = stream(sse("named_event_first"))
        .await
        .unwrap()
        .collect()
        .await;
    let first = events[0].as_ref().unwrap();
    assert!(
        matches!(first, StreamEvent::ResponseStart { id: Some(id), model: Some(model), .. } if id == "c1" && model == "m-large"),
        "{first:?}"
    );
}

/// The data of a named vendor event reaches the response, once per event
/// type, with one warning per event type.
#[tokio::test]
async fn named_event_data_is_surfaced_with_one_warning() {
    let response = stream(sse("named_event_first"))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();
    assert_eq!(response.id.as_deref(), Some("c1"));
    assert_eq!(response.text().as_deref(), Some("Rust is fast [1]."));
    assert_eq!(
        response.provider_data.get_raw(NAMESPACE, "search_results"),
        Some(&json!([
            {"results": [{"url": "https://a.example", "title": "A"}, {"url": "https://b.example", "title": "B"}]},
            {"results": [{"url": "https://c.example", "title": "C"}]},
        ]))
    );
    let warnings: Vec<_> = response
        .warnings
        .iter()
        .filter(|w| w.message.contains("search_results"))
        .collect();
    assert_eq!(warnings.len(), 1, "{:?}", response.warnings);
    assert_eq!(warnings[0].kind, WarningKind::Other);
    assert_eq!(warnings[0].provider.as_deref(), Some("vendor"));
}

/// Top-level vendor fields (`citations`, `search_results`) are reachable
/// without re-parsing a raw response; repeats are not re-emitted.
#[tokio::test]
async fn top_level_chunk_fields_are_surfaced_once() {
    let events: Vec<_> = stream(sse("citations_tail"))
        .await
        .unwrap()
        .map(Result::unwrap)
        .collect()
        .await;
    let surfaced: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::ResponseProviderData { provider_data, .. } => Some(
                provider_data
                    .namespace(NAMESPACE)
                    .unwrap()
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .collect();
    // Perplexity's `usage.search_context_size` is a vendor counter, kept
    // as provider data.
    assert_eq!(
        surfaced,
        [vec!["citations"], vec!["search_results"], vec!["usage"]]
    );

    let mut collector = StreamCollector::new();
    for event in events {
        collector.push(event).unwrap();
    }
    let response = collector.finish().unwrap();
    assert_eq!(
        response.provider_data.get_raw(NAMESPACE, "citations"),
        Some(&json!(["https://a.example", "https://b.example"]))
    );
    assert_eq!(
        response
            .provider_data
            .get_raw(NAMESPACE, "search_results")
            .unwrap()[0]["url"],
        "https://a.example"
    );
    // Object tags and timestamps are not kept.
    assert!(
        response
            .provider_data
            .get_raw(NAMESPACE, "object")
            .is_none()
    );
    assert!(
        response
            .provider_data
            .get_raw(NAMESPACE, "created")
            .is_none()
    );
    assert!(response.warnings.is_empty(), "{:?}", response.warnings);
}

/// Azure's first chunk has no choices and empty id and model: the response
/// still gets the real ones.
#[tokio::test]
async fn empty_first_chunk_does_not_blank_the_id_and_model() {
    let response = stream(sse("empty_first_chunk"))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();
    assert_eq!(response.id.as_deref(), Some("chatcmpl-A1"));
    assert_eq!(response.model.as_deref(), Some("gpt-4o-2024-08-06"));
    assert_eq!(response.text().as_deref(), Some("Hi"));
    assert_eq!(
        response
            .provider_data
            .get_raw(NAMESPACE, "system_fingerprint"),
        Some(&json!("fp_1"))
    );
    assert!(
        response
            .provider_data
            .get_raw(NAMESPACE, "prompt_filter_results")
            .is_some()
    );
}

/// Unary responses keep top-level vendor fields the same way.
#[tokio::test]
async fn unary_responses_keep_top_level_vendor_fields() {
    let provider = Provider::builder()
        .base_url("http://upstream/v1")
        .provider_name("vendor")
        .http_transport(ReplayTransport::new([ReplayResponse::json(
            StatusCode::OK,
            json!({
                "id": "p1", "object": "chat.completion", "created": 1_727_000_000, "model": "sonar",
                "citations": ["https://a.example"],
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok [1]"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                "nothing": null
            })
            .to_string(),
        )]))
        .build()
        .unwrap();
    let response = provider
        .chat(&ChatRequest::new("sonar").user("q"))
        .await
        .unwrap();
    assert_eq!(
        response.provider_data.namespace(NAMESPACE).cloned(),
        Some(serde_json::from_value(json!({"citations": ["https://a.example"]})).unwrap())
    );
}

/// `event: error` frames are typed errors, classified with the dialect's
/// extra quota codes.
#[tokio::test]
async fn error_events_are_typed_errors() {
    let body = concat!(
        "data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"}}]}\n\n",
        "event: error\n",
        "data: {\"error\":{\"message\":\"monthly quota used\",\"code\":\"quota_exceeded\"}}\n\n",
    );
    let mut quirks = CompatQuirks::new(NAMESPACE);
    quirks.quota_error_codes.push("quota_exceeded".into());
    let err = provider_with(Some(quirks), body)
        .chat_stream(&ChatRequest::new("m").user("q"))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap_err();
    assert!(matches!(err, Error::QuotaExhausted { .. }), "{err:?}");
    assert_eq!(err.context().unwrap().provider.as_deref(), Some("vendor"));

    // A bare error object, without the `error` envelope.
    let body = "event: error\ndata: {\"message\":\"overloaded\",\"code\":\"overloaded\"}\n\n";
    let err = stream(body)
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Overloaded { .. }), "{err:?}");
}

/// HTTP errors honor the dialect's extra quota codes too.
#[tokio::test]
async fn http_errors_use_dialect_quota_codes() {
    let mut quirks = CompatQuirks::new(NAMESPACE);
    quirks.quota_error_codes.push("quota_exceeded".into());
    let provider = Provider::builder()
        .base_url("http://upstream/v1")
        .provider_name("vendor")
        .quirks(quirks)
        .http_transport(ReplayTransport::new([ReplayResponse::json(
            StatusCode::TOO_MANY_REQUESTS,
            json!({"error": {"message": "monthly quota used", "code": "quota_exceeded"}})
                .to_string(),
        )]))
        .build()
        .unwrap();
    let err = provider
        .chat(&ChatRequest::new("m").user("q"))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::QuotaExhausted { .. }), "{err:?}");
    assert!(!err.is_retryable());
}
