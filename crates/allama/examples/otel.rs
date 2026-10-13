//! Export allama's spans to OpenTelemetry.
//!
//! [`TracingProvider`] records one span per call through `tracing`, named
//! `{operation} {model}` and following the OpenTelemetry GenAI conventions.
//! The `tracing-opentelemetry` layer (0.32 or later) turns those spans into
//! OpenTelemetry spans and hands them to the tracer provider's exporter.
//!
//! This example exports to an in-memory exporter, so it runs anywhere, and
//! checks that the `chat demo-model` span arrives. In an application, build
//! the tracer provider with an OTLP exporter instead (the
//! `opentelemetry-otlp` crate, at the same version as `opentelemetry`) and
//! keep the rest:
//!
//! ```text
//! let exporter = opentelemetry_otlp::SpanExporter::builder().with_tonic().build()?;
//! let tracer_provider = SdkTracerProvider::builder().with_batch_exporter(exporter).build();
//! ```

use allama::mock::MockProvider;
use allama::prelude::*;
use allama::wrappers::TracingProvider;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // The exporter and the subscriber, once at startup.
    let exporter = InMemorySpanExporter::default();
    let tracer_provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    // The layer adds code location, thread, target and timing attributes of
    // its own; turning them off (optional) leaves the ones allama records.
    let layer = tracing_opentelemetry::layer()
        .with_tracer(tracer_provider.tracer("my-app"))
        .with_location(false)
        .with_threads(false)
        .with_target(false)
        .with_tracked_inactivity(false);
    tracing_subscriber::registry().with(layer).init();

    // Any provider, wrapped in `TracingProvider`.
    let provider = TracingProvider::new(MockProvider::with_text("Hello from a traced call."));
    let request = ChatRequest::new("demo-model").user("Say hello");
    let response = provider.chat(&request).await?;
    println!("{}", response.text_or_empty());

    // Flush before reading the spans, and before the process exits.
    tracer_provider.force_flush()?;
    let spans = exporter.get_finished_spans()?;
    let chat = spans
        .iter()
        .find(|span| span.name == "chat demo-model")
        .ok_or_else(|| anyhow::anyhow!("no `chat demo-model` span was exported"))?;
    println!("exported span `{}` ({:?})", chat.name, chat.span_kind);
    for attribute in &chat.attributes {
        println!("  {} = {}", attribute.key, attribute.value);
    }
    tracer_provider.shutdown()?;
    Ok(())
}
