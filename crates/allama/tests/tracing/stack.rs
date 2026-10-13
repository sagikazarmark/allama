//! The recommended interactive stack,
//! `Tracing(Timeout{stream_setup}(Fallback(Retry(Timeout(p)), ..)))`: long
//! rate limits fail over at once, and the outer deadline is a timeout on the
//! span, not a cancellation.

use std::time::Duration;

use allama::mock::{MockProvider, MockStreamingProvider};
use allama::wrappers::{
    FallbackProvider, RetryPolicy, RetryingProvider, TimeoutPolicy, TimeoutProvider,
    TracingProvider,
};
use allama::{ChatProvider, ChatStreamExt, Error};
use opentelemetry::Value;
use opentelemetry::trace::Status;

use super::otel::export;
use super::request;

/// The interactive stack the guides recommend: a deadline on the first
/// output inside tracing, around a fallback whose primary retries a bounded
/// provider and waits out a `Retry-After` of at most 5 s.
pub(super) fn recommended_stack<P, F>(
    primary: P,
    fallback: F,
    first_output: Duration,
) -> impl ChatProvider
where
    P: ChatProvider + Send + Sync + 'static,
    P::Stream: Send + 'static,
    F: ChatProvider + Send + Sync + 'static,
    F::Stream: Send + 'static,
{
    TracingProvider::new(TimeoutProvider::new(
        FallbackProvider::new(
            RetryingProvider::new(
                TimeoutProvider::new(
                    primary,
                    TimeoutPolicy::new()
                        .stream_setup(Duration::from_secs(60))
                        .stream_idle(Duration::from_secs(60)),
                ),
                RetryPolicy::new().max_retry_after(Duration::from_secs(5)),
            ),
            fallback,
        ),
        TimeoutPolicy::new().stream_setup(first_output),
    ))
}

fn rate_limited(hint: Duration) -> Error {
    Error::rate_limited("HTTP 429")
        .with_status(429)
        .with_retry_after(hint)
}

/// A 429 asking for 30 s fails over within a second, unary and streaming.
#[tokio::test]
async fn a_long_rate_limit_fails_over_at_once() {
    let provider = recommended_stack(
        MockStreamingProvider::build(|b| {
            b.setup_error(rate_limited(Duration::from_secs(30)))
                .setup_error(rate_limited(Duration::from_secs(30)))
        })
        .with_provider_name("anthropic"),
        MockProvider::build(|b| b.text("unary backup").text("streamed backup"))
            .with_provider_name("openai"),
        Duration::from_secs(60),
    );
    let started = std::time::Instant::now();
    let ((unary, streamed), exported) = export(async {
        let unary = provider.chat(&request()).await;
        let streamed = provider
            .chat_stream(&request())
            .await
            .unwrap()
            .collect_response()
            .await;
        (unary, streamed)
    })
    .await;
    assert!(started.elapsed() < Duration::from_secs(1));
    assert_eq!(unary.unwrap().text().as_deref(), Some("unary backup"));
    assert_eq!(streamed.unwrap().text().as_deref(), Some("streamed backup"));
    for chat in exported.spans_for("chat") {
        assert_eq!(
            chat.value("allama.retry.give_up_reason"),
            &Value::from("hint_over_ceiling")
        );
        assert_eq!(chat.value("allama.fallback.used"), &Value::Bool(true));
        assert_eq!(
            chat.value("allama.response.provider"),
            &Value::from("openai")
        );
    }
}

/// The outer deadline expires while the primary backs off before its retry:
/// the call fails with `Error::Timeout` before the commit point, and the
/// span records a failed call, not a cancelled one. Scaled down, because
/// `futures-timer` does not follow tokio's paused clock.
#[tokio::test]
async fn the_outer_deadline_expiring_during_backoff_is_a_timeout() {
    let deadline = Duration::from_millis(100);
    let provider = recommended_stack(
        // A hint within the ceiling, so the primary waits it out.
        MockStreamingProvider::build(|b| {
            b.setup_error(rate_limited(Duration::from_millis(400)))
                .text("too late")
        }),
        MockProvider::with_text("never asked"),
        deadline,
    );
    let started = std::time::Instant::now();
    let (result, exported) = export(async { provider.chat_stream(&request()).await }).await;
    let elapsed = started.elapsed();
    let Err(error) = result else {
        panic!("the deadline should end the call before its first output");
    };
    assert!(matches!(error, Error::Timeout { .. }), "{error}");
    assert!(
        elapsed >= deadline && elapsed < Duration::from_millis(400),
        "{elapsed:?}"
    );

    let chat = exported.span("chat");
    assert_eq!(chat.value("error.type"), &Value::from("timeout"));
    assert!(
        matches!(chat.status(), Status::Error { .. }),
        "{:?}",
        chat.status()
    );
    assert!(!chat.has("allama.response.cancelled"));
    // The primary was waiting for its retry: one attempt, one backoff begun.
    assert_eq!(chat.value("allama.attempts"), &Value::I64(1));
}

/// The span records the request `TracingProvider`
/// was given. A redaction transform inside tracing runs after the span
/// recorded the caller's text, so it belongs outside.
#[tokio::test]
async fn a_redaction_transform_goes_outside_tracing() {
    use allama::ChatRequest;
    use allama::wrappers::{MapRequestProvider, TracingContentConfig};

    fn redact(request: &mut ChatRequest) {
        for message in &mut request.messages {
            if let allama::Message::User(user) = message {
                user.content = "[email]".into();
            }
        }
    }
    let request = ChatRequest::new("m").user("mail jane@example.com");
    let inner_content = |key: &str, exported: &super::otel::Exported| {
        exported.span("chat").value(key).as_str().into_owned()
    };

    // Outside tracing: the span shows what was sent.
    let outside = MapRequestProvider::new(
        TracingProvider::new(MockProvider::with_text("ok"))
            .with_content_capture(TracingContentConfig::all()),
        redact,
    );
    let (_, exported) = export(async { outside.chat(&request).await }).await;
    let recorded = inner_content("gen_ai.input.messages", &exported);
    assert!(recorded.contains("[email]"), "{recorded}");
    assert!(!recorded.contains("jane@example.com"), "{recorded}");

    // Inside tracing: the span shows the caller's request, unredacted.
    let inside = TracingProvider::new(MapRequestProvider::new(
        MockProvider::with_text("ok"),
        redact,
    ))
    .with_content_capture(TracingContentConfig::all());
    let (_, exported) = export(async { inside.chat(&request).await }).await;
    let recorded = inner_content("gen_ai.input.messages", &exported);
    assert!(recorded.contains("jane@example.com"), "{recorded}");
}
