//! OpenTelemetry GenAI spans with content capture, printed to the console.
//!
//! `TracingProvider` records spans, not events, so a `fmt` subscriber
//! prints a successful call only when it is told to print spans as they
//! close (`FmtSpan::CLOSE`), with every attribute recorded on them.

use allama::mock::MockStreamingProvider;
use allama::prelude::*;
use allama::wrappers::{TracingContentConfig, TracingProvider};
use tracing_subscriber::FmtSubscriber;
use tracing_subscriber::fmt::format::FmtSpan;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = tracing::subscriber::set_global_default(
        FmtSubscriber::builder()
            // One line per call, when its span closes.
            .with_span_events(FmtSpan::CLOSE)
            .with_target(false)
            .without_time()
            .finish(),
    );

    let provider = TracingProvider::new(MockStreamingProvider::build(|b| {
        b.text("Traced hello from allama.")
            .text("Traced hello from allama.")
    }))
    .with_content_capture(TracingContentConfig::all());

    let request = ChatRequest::new("demo-model")
        .system("You are concise.")
        .user("Say hello");
    let response = provider.chat(&request).await?;
    println!("traced chat text: {}", response.text_or_empty());

    let streamed = provider
        .chat_stream(&request)
        .await?
        .collect_response()
        .await?;
    println!("traced stream text: {}", streamed.text_or_empty());
    Ok(())
}
