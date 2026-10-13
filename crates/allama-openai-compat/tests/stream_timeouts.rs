//! Streams that stall fail with `Error::Timeout` instead of hanging.

use std::time::Duration;

use allama::prelude::*;
use allama::{ChatStreamExt, StreamEvent};
use allama_http::{
    HttpRequest, HttpTransport, ReplayResponse, ReplayTransport, SendFuture, StreamTimeouts,
};
use allama_openai_compat::{Provider, ResponsesProvider};
use futures_util::StreamExt;

const SHORT: Duration = Duration::from_millis(30);

/// One Chat Completions chunk, then silence.
const CHAT_START: &str = "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\
\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n";

/// One Responses event, then silence.
const RESPONSES_START: &str = "event: response.created\ndata: {\"type\":\"response.created\",\
\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"in_progress\",\
\"model\":\"m\",\"output\":[]}}\n\n";

/// A transport whose responses never arrive.
struct NoHeaders;

impl HttpTransport for NoHeaders {
    fn send(&self, _request: HttpRequest) -> SendFuture<'_> {
        Box::pin(std::future::pending())
    }
}

fn chat(transport: impl HttpTransport, timeouts: StreamTimeouts) -> Provider {
    Provider::builder()
        .base_url("http://upstream/v1")
        .provider_name("stally")
        .http_transport(transport)
        .stream_timeouts(timeouts)
        .build()
        .unwrap()
}

fn responses(transport: impl HttpTransport, timeouts: StreamTimeouts) -> ResponsesProvider {
    ResponsesProvider::builder()
        .base_url("http://upstream/v1")
        .provider_name("stally")
        .http_transport(transport)
        .stream_timeouts(timeouts)
        .build()
        .unwrap()
}

fn request() -> ChatRequest {
    ChatRequest::new("m").user("hi")
}

fn assert_timeout(err: &Error, message: &str) {
    assert!(matches!(err, Error::Timeout { .. }), "{err:?}");
    assert!(err.is_retryable());
    assert!(err.to_string().contains(message), "{err}");
    assert_eq!(
        err.context().and_then(|c| c.provider.as_deref()),
        Some("stally")
    );
}

#[tokio::test]
async fn chat_stream_that_stalls_mid_body_times_out() {
    let transport = ReplayTransport::new([ReplayResponse::sse(CHAT_START).then_stall()]);
    let provider = chat(transport, StreamTimeouts::default().idle(Some(SHORT)));
    let mut stream = provider.chat_stream(&request()).await.unwrap();

    let mut text = String::new();
    let err = loop {
        match stream.next().await.expect("the stream ends with an error") {
            Ok(StreamEvent::TextDelta { text: delta, .. }) => text.push_str(&delta),
            Ok(_) => {}
            Err(err) => break err,
        }
    };
    assert_eq!(text, "Hel");
    assert_timeout(&err, "wait for the next body chunk exceeded 30ms");
}

#[tokio::test]
async fn chat_stream_without_headers_times_out() {
    let provider = chat(NoHeaders, StreamTimeouts::default().setup(Some(SHORT)));
    let Err(err) = provider.chat_stream(&request()).await else {
        panic!("expected a setup timeout");
    };
    assert_timeout(&err, "stream setup exceeded 30ms");
}

#[tokio::test]
async fn responses_stream_that_stalls_times_out() {
    let transport = ReplayTransport::new([ReplayResponse::sse(RESPONSES_START).then_stall()]);
    let provider = responses(transport, StreamTimeouts::default().idle(Some(SHORT)));
    let err = provider
        .chat_stream(&request())
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap_err();
    assert_timeout(&err, "wait for the next body chunk exceeded 30ms");

    let provider = responses(NoHeaders, StreamTimeouts::default().setup(Some(SHORT)));
    let Err(err) = provider.chat_stream(&request()).await else {
        panic!("expected a setup timeout");
    };
    assert_timeout(&err, "stream setup exceeded 30ms");
}

#[tokio::test]
async fn unbounded_streams_keep_waiting() {
    let transport = ReplayTransport::new([ReplayResponse::sse(CHAT_START).then_stall()]);
    let provider = chat(transport, StreamTimeouts::unbounded());
    let mut stream = provider.chat_stream(&request()).await.unwrap();
    while let Ok(Some(event)) =
        tokio::time::timeout(Duration::from_millis(100), stream.next()).await
    {
        assert!(event.is_ok(), "{event:?}");
    }
}
