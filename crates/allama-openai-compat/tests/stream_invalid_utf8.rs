//! A stream line that is not UTF-8 fails the stream with a stream decode
//! error, which is not retryable: the decoder does not replace the bytes
//! with U+FFFD, as the WHATWG standard would, and asking again streams the
//! same bytes.

use std::time::Duration;

use allama::prelude::*;
use allama::wrappers::{RetryPolicy, RetryingProvider};
use allama::{ChatStreamExt, StreamErrorKind, StreamEvent};
use allama_http::{ReplayResponse, ReplayTransport};
use allama_openai_compat::{Provider, ResponsesProvider};
use futures_util::StreamExt;

/// A Chat Completions chunk with `text` as its content delta.
fn chat_chunk(text: &[u8]) -> Vec<u8> {
    let mut chunk = br#"data: {"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"content":""#.to_vec();
    chunk.extend_from_slice(text);
    chunk.extend_from_slice(br#""},"finish_reason":null}]}"#);
    chunk.extend_from_slice(b"\n\n");
    chunk
}

/// `Hel`, then a chunk that ends in the middle of a two-byte sequence
/// (`caf` and the first byte of `é`).
fn chat_body() -> Vec<u8> {
    let mut body = chat_chunk(b"Hel");
    body.extend(chat_chunk(b"caf\xC3"));
    body.extend_from_slice(b"data: [DONE]\n\n");
    body
}

/// A Responses stream: `response.created`, then a text delta with a byte
/// that is never valid UTF-8.
fn responses_body() -> Vec<u8> {
    let mut body = b"event: response.created\ndata: {\"type\":\"response.created\",\
        \"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\
        \"status\":\"in_progress\",\"model\":\"m\",\"output\":[]}}\n\n"
        .to_vec();
    body.extend_from_slice(
        b"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\
          \"sequence_number\":1,\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\
          \"delta\":\"\xFF\"}\n\n",
    );
    body
}

fn sse(body: Vec<u8>) -> ReplayResponse {
    ReplayResponse::new(200, body).header("content-type", "text/event-stream")
}

fn assert_invalid_utf8(err: &Error, provider: &str) {
    assert!(
        matches!(
            err,
            Error::Stream {
                kind: StreamErrorKind::Decode,
                ..
            }
        ),
        "{err:?}"
    );
    assert!(!err.is_retryable());
    assert!(
        err.to_string().contains("invalid UTF-8 in the SSE stream"),
        "{err}"
    );
    assert_eq!(
        err.context().and_then(|c| c.provider.as_deref()),
        Some(provider)
    );
    // Written by the decoder: the redacted log keeps it.
    assert!(
        err.as_log().to_string().contains("invalid UTF-8"),
        "{}",
        err.as_log()
    );
}

fn chat_provider(responses: impl IntoIterator<Item = ReplayResponse>) -> Provider {
    Provider::builder()
        .base_url("http://upstream/v1")
        .provider_name("garbled")
        .http_transport(ReplayTransport::new(responses))
        .build()
        .unwrap()
}

#[tokio::test]
async fn chat_stream_invalid_utf8_fails_after_the_events_before_it() {
    let provider = chat_provider([sse(chat_body())]);
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
    // Nothing replaced the byte: the text stops before the bad chunk.
    assert_eq!(text, "Hel");
    assert_invalid_utf8(&err, "garbled");
}

#[tokio::test]
async fn chat_stream_invalid_utf8_is_not_retried() {
    let transport = ReplayTransport::new([sse(chat_chunk(b"\xFF")), sse(chat_body())]);
    let provider = Provider::builder()
        .base_url("http://upstream/v1")
        .provider_name("garbled")
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let retrying = RetryingProvider::new(
        provider,
        RetryPolicy::new().base_delay(Duration::ZERO).jitter(false),
    );
    let result = match retrying
        .chat_stream(&ChatRequest::new("m").user("hi"))
        .await
    {
        Ok(stream) => stream.collect_response().await,
        Err(err) => Err(err),
    };
    let err = result.unwrap_err();
    assert_invalid_utf8(&err, "garbled");
    assert_eq!(
        transport.requests().len(),
        1,
        "asked again for the same bytes"
    );
}

#[tokio::test]
async fn responses_stream_invalid_utf8_fails() {
    let provider = ResponsesProvider::builder()
        .base_url("http://upstream/v1")
        .provider_name("garbled")
        .http_transport(ReplayTransport::new([sse(responses_body())]))
        .build()
        .unwrap();
    let err = provider
        .chat_stream(&ChatRequest::new("m").user("hi"))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap_err();
    assert_invalid_utf8(&err, "garbled");
}
