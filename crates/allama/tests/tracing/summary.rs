//! The call-scope summary on spans: each key recorded once, the counting
//! rules across retry layers and fallback chains, and
//! `allama.response.provider` naming the serving provider. How facts add up
//! in a call scope is unit-tested in `wrappers/call_scope.rs`.

use std::time::Duration;

use allama::chat::stream::StreamBlockType;
use allama::mock::{
    MockEmbeddingProvider, MockProvider, MockResponse, MockStreamEvent, MockStreamingProvider,
};
use allama::wrappers::{FallbackProvider, RetryPolicy, RetryingProvider, TracingProvider};
use allama::{
    ChatProvider, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, Error, StreamEvent,
};
use futures_util::{FutureExt, StreamExt};
use opentelemetry::Value;

use super::otel::{ExportedSpan, export};
use super::{Call, request};

/// The keys of the resilience summary and `allama.response.provider`.
const SUMMARY_KEYS: &[&str] = &[
    "allama.response.provider",
    "allama.attempts",
    "allama.retry.total_delay_ms",
    "allama.retry.last_delay_ms",
    "allama.retry.server_directive",
    "allama.retry.budget_denied",
    "allama.retry.give_up_reason",
    "allama.fallback.used",
    "allama.fallback.attempts",
    "allama.fallback.error_type",
];

/// The summary keys a span exported, with their values, in
/// [`SUMMARY_KEYS`] order.
fn summary(span: ExportedSpan<'_>) -> Vec<(&'static str, Value)> {
    SUMMARY_KEYS
        .iter()
        .filter_map(|key| span.get(key).map(|value| (*key, value.clone())))
        .collect()
}

/// No backoff or jitter, so each delay is exactly its retry hint.
fn exact_delays(max_attempts: usize) -> RetryPolicy {
    RetryPolicy::new()
        .max_attempts(max_attempts)
        .base_delay(Duration::ZERO)
        .max_delay(Duration::from_secs(1))
        .jitter(false)
}

fn hinted(error: Error, millis: u64) -> Error {
    error.with_retry_after(Duration::from_millis(millis))
}

/// Retry layers on both sides of a failover: every key is recorded once
/// (the harness fails otherwise) and summarizes both layers.
#[tokio::test]
async fn retry_on_both_sides_and_failover_summarize_once() {
    for call in Call::ALL {
        let primary = RetryingProvider::new(
            MockProvider::new([
                hinted(Error::overloaded("busy"), 7),
                hinted(Error::timeout("slow"), 7),
            ])
            .with_provider_name("anthropic"),
            exact_delays(2),
        );
        let backup = RetryingProvider::new(
            MockProvider::build(|b| {
                b.error(hinted(Error::rate_limited("slow down"), 3))
                    .text("ok")
            })
            .with_provider_name("xai"),
            exact_delays(2),
        );
        let provider = TracingProvider::new(FallbackProvider::new(primary, backup));
        let (result, exported) = export(call.run(&provider)).await;
        assert_eq!(result.unwrap().text().as_deref(), Some("ok"), "{call:?}");
        let chat = exported.span("chat");
        assert_eq!(
            summary(chat),
            [
                ("allama.response.provider", Value::from("x_ai")),
                ("allama.attempts", Value::I64(4)),
                ("allama.retry.total_delay_ms", Value::I64(10)),
                ("allama.retry.last_delay_ms", Value::I64(3)),
                ("allama.retry.give_up_reason", Value::from("max_attempts")),
                ("allama.fallback.used", Value::Bool(true)),
                ("allama.fallback.attempts", Value::I64(1)),
                ("allama.fallback.error_type", Value::from("timeout")),
            ],
            "{call:?}"
        );
        assert_eq!(
            chat.value("gen_ai.provider.name"),
            &Value::from("anthropic"),
            "{call:?}"
        );
    }
}

/// `Fallback(Retry(a), Fallback(Retry(b), c))` served by `c`: attempts add
/// up across the chain, each failed provider counts once, and the innermost
/// serving provider is the response provider.
#[tokio::test]
async fn nested_chain_with_retries_summarizes_once() {
    let retried = |name: &'static str, error: fn(&'static str) -> Error, millis| {
        RetryingProvider::new(
            MockProvider::new([
                hinted(error("first"), millis),
                hinted(error("second"), millis),
            ])
            .with_provider_name(name),
            exact_delays(2),
        )
    };
    for call in Call::ALL {
        let provider = TracingProvider::new(FallbackProvider::new(
            retried("first", Error::overloaded, 2),
            FallbackProvider::new(
                retried("second", Error::timeout, 5),
                MockProvider::with_text("third").with_provider_name("third"),
            ),
        ));
        let (result, exported) = export(call.run(&provider)).await;
        assert_eq!(result.unwrap().text().as_deref(), Some("third"), "{call:?}");
        let chat = exported.span("chat");
        assert_eq!(
            summary(chat),
            [
                ("allama.response.provider", Value::from("third")),
                ("allama.attempts", Value::I64(5)),
                ("allama.retry.total_delay_ms", Value::I64(7)),
                ("allama.retry.last_delay_ms", Value::I64(5)),
                ("allama.retry.give_up_reason", Value::from("max_attempts")),
                ("allama.fallback.used", Value::Bool(true)),
                ("allama.fallback.attempts", Value::I64(2)),
                ("allama.fallback.error_type", Value::from("timeout")),
            ],
            "{call:?}"
        );
        assert_eq!(chat.value("gen_ai.provider.name"), &Value::from("first"));
    }
}

/// `Fallback(a, Fallback(b, c))` served by `c` attributes an embedding span
/// to `c`: the inner fallback names `c` on the response, and the outer one
/// keeps it.
#[tokio::test]
async fn nested_embedding_chain_reports_the_innermost_serving_provider() {
    let failing = |name| {
        MockEmbeddingProvider::with_error(Error::overloaded("down")).with_provider_name(name)
    };
    let provider = TracingProvider::new(FallbackProvider::new(
        failing("first"),
        FallbackProvider::new(
            failing("second"),
            MockEmbeddingProvider::new([Ok(EmbeddingResponse::new(vec![vec![0.5]]))])
                .with_provider_name("mistral"),
        ),
    ));
    let (result, exported) =
        export(async { provider.embed(&EmbeddingRequest::new("e").input("x")).await }).await;
    result.unwrap();
    let embeddings = exported.span("embeddings");
    assert_eq!(
        summary(embeddings),
        [
            ("allama.response.provider", Value::from("mistral_ai")),
            ("allama.attempts", Value::I64(3)),
            ("allama.fallback.used", Value::Bool(true)),
            ("allama.fallback.attempts", Value::I64(2)),
            ("allama.fallback.error_type", Value::from("overloaded")),
        ]
    );
    assert_eq!(
        embeddings.value("gen_ai.provider.name"),
        &Value::from("first")
    );
}

/// The span names the provider the embedding response names, ahead of the
/// addressed one: here a gateway that reports the upstream it routed to.
#[tokio::test]
async fn an_embedding_span_prefers_the_provider_the_response_names() {
    let provider = TracingProvider::new(
        MockEmbeddingProvider::new([Ok(
            EmbeddingResponse::new(vec![vec![0.5]]).provider("mistral")
        )])
        .with_provider_name("gateway"),
    );
    let (result, exported) =
        export(async { provider.embed(&EmbeddingRequest::new("e").input("x")).await }).await;
    assert_eq!(result.unwrap().provider.as_deref(), Some("mistral"));
    let embeddings = exported.span("embeddings");
    assert_eq!(
        embeddings.value("allama.response.provider"),
        &Value::from("mistral_ai")
    );
    assert_eq!(
        embeddings.value("gen_ai.provider.name"),
        &Value::from("gateway")
    );
}

/// A provider that served still served when the stream was cut short
/// afterwards: by the consumer, or by an error after the first output.
#[tokio::test]
async fn a_stream_cut_short_after_failover_reports_the_serving_provider() {
    let provider = |tail: MockStreamEvent| {
        TracingProvider::new(FallbackProvider::new(
            MockProvider::with_error(Error::overloaded("HTTP 503").with_status(503))
                .with_provider_name("anthropic"),
            MockStreamingProvider::with_stream([
                MockStreamEvent::Event(StreamEvent::response_start(None, None)),
                MockStreamEvent::Event(StreamEvent::block_start(0, StreamBlockType::Text)),
                MockStreamEvent::Event(StreamEvent::text_delta(0, "first")),
                tail,
            ])
            .with_provider_name("xai"),
        ))
    };

    // Dropped after the fallback's first token.
    let dropped = provider(MockStreamEvent::Event(StreamEvent::text_delta(
        0,
        " never read",
    )));
    let ((), exported) = export(async {
        let mut stream = dropped.chat_stream(&request()).await.unwrap();
        while let Some(event) = stream.next().await {
            if matches!(event.unwrap(), StreamEvent::TextDelta { .. }) {
                break;
            }
        }
        drop(stream);
    })
    .await;
    let chat = exported.span("chat");
    assert_eq!(chat.value("allama.response.cancelled"), &Value::Bool(true));
    assert_eq!(chat.value("allama.response.provider"), &Value::from("x_ai"));
    assert_eq!(chat.value("allama.attempts"), &Value::I64(2));
    assert_eq!(chat.value("allama.fallback.attempts"), &Value::I64(1));
    assert_eq!(
        chat.value("allama.fallback.error_type"),
        &Value::from("overloaded")
    );
    assert_eq!(
        chat.value("gen_ai.provider.name"),
        &Value::from("anthropic")
    );

    // Failed after the fallback's first token.
    let failed = provider(MockStreamEvent::Error(Error::timeout("stalled")));
    let ((), exported) = export(async {
        let mut stream = failed.chat_stream(&request()).await.unwrap();
        while stream.next().await.is_some() {}
    })
    .await;
    let chat = exported.span("chat");
    assert_eq!(chat.value("error.type"), &Value::from("timeout"));
    assert_eq!(chat.value("allama.response.provider"), &Value::from("x_ai"));
    assert!(!chat.has("allama.response.cancelled"));
}

/// A call dropped before any provider served records the last fallback side
/// tried and the attempts made so far, unary and streaming.
#[tokio::test]
async fn a_call_dropped_before_serving_reports_the_last_side_tried() {
    let provider = || {
        TracingProvider::new(FallbackProvider::new(
            MockProvider::with_error(Error::overloaded("down")).with_provider_name("anthropic"),
            MockProvider::new([MockResponse::delayed(
                Duration::from_secs(30),
                MockResponse::text("too late"),
            )])
            .with_provider_name("xai"),
        ))
    };
    let unary = provider();
    let streaming = provider();
    let ((), exported) = export(async {
        // Polled once: the primary fails, the fallback waits; then dropped.
        assert!(unary.chat(&request()).now_or_never().is_none());
        assert!(streaming.chat_stream(&request()).now_or_never().is_none());
    })
    .await;
    let spans: Vec<_> = exported.spans_for("chat").collect();
    assert_eq!(spans.len(), 2);
    for chat in spans {
        assert_eq!(
            summary(chat),
            [
                ("allama.response.provider", Value::from("x_ai")),
                ("allama.attempts", Value::I64(2)),
                ("allama.fallback.used", Value::Bool(true)),
                ("allama.fallback.attempts", Value::I64(1)),
                ("allama.fallback.error_type", Value::from("overloaded")),
            ],
            "{}",
            chat.name()
        );
        assert_eq!(chat.value("allama.response.cancelled"), &Value::Bool(true));
    }
}

/// In a nested chain, the last side tried is the innermost one started.
#[tokio::test]
async fn a_nested_chain_dropped_before_serving_reports_the_innermost_side_tried() {
    let provider = TracingProvider::new(FallbackProvider::new(
        MockProvider::with_error(Error::overloaded("down")).with_provider_name("anthropic"),
        FallbackProvider::new(
            MockProvider::with_error(Error::overloaded("down")).with_provider_name("openai"),
            MockProvider::new([MockResponse::delayed(
                Duration::from_secs(30),
                MockResponse::text("too late"),
            )])
            .with_provider_name("gemini"),
        ),
    ));
    let ((), exported) = export(async {
        assert!(provider.chat(&request()).now_or_never().is_none());
    })
    .await;
    let chat = exported.span("chat");
    assert_eq!(
        chat.value("allama.response.provider"),
        &Value::from("gcp.gemini")
    );
    assert_eq!(chat.value("allama.fallback.attempts"), &Value::I64(2));
}

/// A `TracingProvider` nested on a fallback side keeps its retries on its
/// own span; the outer span sees the fallback's facts and the inner call as
/// one attempt.
#[tokio::test]
async fn nested_tracing_keeps_inner_facts_on_the_inner_span() {
    let inner = TracingProvider::new(RetryingProvider::new(
        MockProvider::new([
            hinted(Error::timeout("slow"), 1),
            hinted(Error::timeout("slow"), 1),
        ]),
        exact_delays(2),
    ));
    let provider = TracingProvider::new(FallbackProvider::new(
        inner,
        MockProvider::with_text("backup").with_provider_name("backup"),
    ));
    let (result, exported) = export(async { provider.chat(&request()).await }).await;
    result.unwrap();
    // The inner span ends first.
    let [inner, outer] = exported.spans_for("chat").collect::<Vec<_>>()[..] else {
        panic!("expected two chat spans");
    };
    assert_eq!(
        summary(inner),
        [
            ("allama.response.provider", Value::from("mock")),
            ("allama.attempts", Value::I64(2)),
            ("allama.retry.total_delay_ms", Value::I64(1)),
            ("allama.retry.last_delay_ms", Value::I64(1)),
            ("allama.retry.give_up_reason", Value::from("max_attempts")),
        ]
    );
    assert_eq!(
        summary(outer),
        [
            ("allama.response.provider", Value::from("backup")),
            ("allama.attempts", Value::I64(2)),
            ("allama.fallback.used", Value::Bool(true)),
            ("allama.fallback.attempts", Value::I64(1)),
            ("allama.fallback.error_type", Value::from("timeout")),
        ]
    );
}

/// Each give-up reason, from the retry layer that gave up.
#[tokio::test]
async fn give_up_reasons() {
    let cases = [
        (
            RetryPolicy::new().max_attempts(1),
            Error::overloaded("busy"),
            "max_attempts",
        ),
        (
            RetryPolicy::new().max_retry_after(Duration::from_millis(10)),
            hinted(Error::rate_limited("slow down"), 30_000),
            "hint_over_ceiling",
        ),
        (
            RetryPolicy::new().budget(allama::wrappers::RetryBudget::new().capacity(0)),
            Error::overloaded("busy"),
            "budget",
        ),
        (
            RetryPolicy::new(),
            Error::invalid_request("bad"),
            "not_retryable",
        ),
    ];
    for (policy, error, reason) in cases {
        let provider = TracingProvider::new(RetryingProvider::new(
            MockProvider::with_error(error),
            policy,
        ));
        let (result, exported) = export(async { provider.chat(&request()).await }).await;
        assert!(result.is_err(), "{reason}");
        let chat = exported.span("chat");
        assert_eq!(
            chat.value("allama.retry.give_up_reason"),
            &Value::from(reason)
        );
        assert_eq!(chat.value("allama.attempts"), &Value::I64(1), "{reason}");
        assert_eq!(chat.has("allama.retry.budget_denied"), reason == "budget");
    }
}

/// A call through no resilience wrapper still records one attempt and the
/// provider that served it.
#[tokio::test]
async fn a_plain_call_records_one_attempt() {
    let provider = TracingProvider::new(
        MockEmbeddingProvider::with_vectors(vec![vec![0.1]]).with_provider_name("google-vertex"),
    );
    let (result, exported) =
        export(async { provider.embed(&EmbeddingRequest::new("e").input("x")).await }).await;
    result.unwrap();
    assert_eq!(
        summary(exported.span("embeddings")),
        [
            ("allama.response.provider", Value::from("gcp.vertex_ai")),
            ("allama.attempts", Value::I64(1)),
        ]
    );
}
