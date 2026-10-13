//! Log every chat call as a JSON line, streams included.
//!
//! `RecordingProvider` hands a `ChatExchange` (request, response or error,
//! and the stream transcript) to a sink when each call's outcome is known.
//! `TimeoutProvider` bounds the calls; a stalled stream is recorded with its
//! partial output and the timeout.

use std::time::Duration;

use allama::mock::{MockStreamEvent, MockStreamingProvider};
use allama::prelude::*;
use allama::wrappers::{RecordingProvider, TimeoutPolicy, TimeoutProvider};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut stalled = MockStreamEvent::text_response("Hello");
    // Pause before `ResponseStop` for longer than the idle bound.
    stalled.insert(
        stalled.len() - 1,
        MockStreamEvent::delayed(Duration::from_secs(5)),
    );
    let provider =
        MockStreamingProvider::new([MockStreamEvent::text_response("Hi there"), stalled]);

    let provider = RecordingProvider::new(
        TimeoutProvider::new(
            provider,
            TimeoutPolicy::new().stream_idle(Duration::from_millis(100)),
        ),
        |exchange| {
            println!(
                "{}",
                serde_json::to_string(&exchange).expect("exchanges serialize")
            );
        },
    );

    let request = ChatRequest::new("demo-model").user("Say hello");
    let response = provider
        .chat_stream(&request)
        .await?
        .collect_response()
        .await?;
    eprintln!("first: {}", response.text_or_empty());

    let error = provider
        .chat_stream(&request)
        .await?
        .collect_response()
        .await
        .unwrap_err();
    eprintln!("second: {error}");
    Ok(())
}
