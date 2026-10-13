//! A stream event past the provider's size cap fails the stream with a
//! stream protocol error, not a retryable transport failure.

use allama::prelude::*;
use allama::{ChatStreamExt, StreamErrorKind, StreamEvent};
use allama_http::{ReplayResponse, ReplayTransport};
use allama_openai_compat::{Provider, ResponsesProvider};
use futures_util::StreamExt;

/// One small Chat Completions chunk, then one whose data passes 256 bytes.
fn chat_sse() -> String {
    let chunk = |text: &str| {
        format!(
            "data: {{\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\
             \"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{text}\"}},\"finish_reason\":null}}]}}\n\n"
        )
    };
    format!(
        "{}{}data: [DONE]\n\n",
        chunk("Hel"),
        chunk(&"x".repeat(300))
    )
}

/// One Responses event, then one whose data passes 512 bytes.
fn responses_sse() -> String {
    let created = "event: response.created\ndata: {\"type\":\"response.created\",\
        \"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\
        \"status\":\"in_progress\",\"model\":\"m\",\"output\":[]}}\n\n";
    let delta = format!(
        "event: response.output_text.delta\ndata: {{\"type\":\"response.output_text.delta\",\
         \"sequence_number\":1,\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\
         \"delta\":\"{}\"}}\n\n",
        "x".repeat(600)
    );
    format!("{created}{delta}")
}

fn assert_too_large(err: &Error, limit: usize) {
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
        err.to_string().contains(&format!(
            "an SSE event passed the size cap of {limit} bytes"
        )),
        "{err}"
    );
    assert_eq!(
        err.context().and_then(|c| c.provider.as_deref()),
        Some("capped")
    );
    // Written by the decoder: the redacted log keeps it.
    assert!(
        err.as_log().to_string().contains("size cap"),
        "{}",
        err.as_log()
    );
}

#[tokio::test]
async fn chat_stream_event_past_the_cap_fails_after_the_events_before_it() {
    let provider = Provider::builder()
        .base_url("http://upstream/v1")
        .provider_name("capped")
        .http_transport(ReplayTransport::new([ReplayResponse::sse(chat_sse())]))
        .max_stream_event_size(256)
        .build()
        .unwrap();
    let mut stream = provider
        .chat_stream(&ChatRequest::new("m").user("hi"))
        .await
        .unwrap();
    let mut text = String::new();
    let err = loop {
        match stream.next().await.expect("the stream ends with an error") {
            Ok(StreamEvent::TextDelta { text: delta, .. }) => text.push_str(&delta),
            Ok(_) => {}
            Err(err) => break err,
        }
    };
    assert_eq!(text, "Hel");
    assert_too_large(&err, 256);
}

#[tokio::test]
async fn the_default_cap_admits_the_same_stream() {
    let provider = Provider::builder()
        .base_url("http://upstream/v1")
        .http_transport(ReplayTransport::new([ReplayResponse::sse(chat_sse())]))
        .build()
        .unwrap();
    let response = provider
        .chat_stream(&ChatRequest::new("m").user("hi"))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();
    assert_eq!(response.text_or_empty().len(), 303);
}

#[tokio::test]
async fn responses_stream_event_past_the_cap_fails() {
    let provider = ResponsesProvider::builder()
        .base_url("http://upstream/v1")
        .provider_name("capped")
        .http_transport(ReplayTransport::new([ReplayResponse::sse(responses_sse())]))
        .max_stream_event_size(512)
        .build()
        .unwrap();
    let err = provider
        .chat_stream(&ChatRequest::new("m").user("hi"))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap_err();
    assert_too_large(&err, 512);
}
