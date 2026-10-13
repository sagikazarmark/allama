//! The server retry directive (`x-should-retry`, surfaced as
//! `Error::server_retry_directive`) under `RetryingProvider`.
//!
//! The default classifier follows the directive in both directions, but the
//! directive never lifts the attempt limit, the ADR 0011 give-up rule, or the
//! rule against retrying a stream after its first output. It stops retries
//! only: fallback eligibility is unchanged. A custom classifier owns the
//! decision.

#![cfg(feature = "mock")]

use std::time::Duration;

use allama::mock::{ChatResponseBuilder, MockProvider, MockStreamEvent, MockStreamingProvider};
use allama::wrappers::{FallbackProvider, RetryPolicy, RetryingProvider};
use allama::{ChatProvider, ChatRequest, ChatStreamExt, Error, StreamEvent};

fn request() -> ChatRequest {
    ChatRequest::new("m").user("hi")
}

/// No backoff, and room for retry hints up to a second.
fn policy() -> RetryPolicy {
    RetryPolicy::new()
        .base_delay(Duration::ZERO)
        .max_retry_after(Duration::from_secs(1))
}

/// A `409 Conflict`: not retryable by status.
fn conflict(directive: Option<bool>) -> Error {
    Error::invalid_request("HTTP 409")
        .authored()
        .with_status(409)
        .with_server_retry_directive_opt(directive)
}

/// A `429` with a retry hint the policy would wait.
fn rate_limited(hint: Duration, directive: Option<bool>) -> Error {
    Error::rate_limited("HTTP 429")
        .authored()
        .with_status(429)
        .with_retry_after(hint)
        .with_server_retry_directive_opt(directive)
}

#[tokio::test]
async fn directive_true_retries_a_conflict() {
    let mock = MockProvider::build(|b| b.error(conflict(Some(true))).text("ok"));
    let wrapper = RetryingProvider::new(mock.clone(), policy());
    let response = wrapper.chat(&request()).await.unwrap();
    assert_eq!(response.text().as_deref(), Some("ok"));
    assert_eq!(mock.call_count(), 2);

    // Without the directive the status decides: a 409 is not retried.
    let mock = MockProvider::build(|b| b.error(conflict(None)).text("ok"));
    let wrapper = RetryingProvider::new(mock.clone(), policy());
    assert!(wrapper.chat(&request()).await.is_err());
    assert_eq!(mock.call_count(), 1);
}

#[tokio::test]
async fn directive_false_stops_a_rate_limit_with_a_retry_hint() {
    let hint = Duration::from_millis(1);
    let mock = MockProvider::build(|b| b.error(rate_limited(hint, Some(false))).text("ok"));
    let wrapper = RetryingProvider::new(mock.clone(), policy());
    let error = wrapper.chat(&request()).await.unwrap_err();
    assert!(matches!(error, Error::RateLimited { .. }));
    assert_eq!(error.retry_after(), Some(hint));
    assert_eq!(mock.call_count(), 1);

    // The same error without the directive is retried after its hint.
    let mock = MockProvider::build(|b| b.error(rate_limited(hint, None)).text("ok"));
    let wrapper = RetryingProvider::new(mock.clone(), policy());
    assert!(wrapper.chat(&request()).await.is_ok());
    assert_eq!(mock.call_count(), 2);
}

#[tokio::test]
async fn directive_false_stops_a_stream_before_output() {
    let mock = MockStreamingProvider::build(|b| {
        b.stream([
            MockStreamEvent::Event(StreamEvent::response_start(None, None)),
            MockStreamEvent::Error(
                Error::overloaded("HTTP 503").with_server_retry_directive(false),
            ),
        ])
        .response(ChatResponseBuilder::new().text("never").build())
    });
    let wrapper = RetryingProvider::new(mock.clone(), policy());
    assert!(wrapper.chat_stream(&request()).await.is_err());
    assert_eq!(mock.call_count(), 1);
}

#[tokio::test]
async fn directive_true_keeps_the_attempt_limit() {
    let mock = MockProvider::new([
        conflict(Some(true)),
        conflict(Some(true)),
        conflict(Some(true)),
    ]);
    let wrapper = RetryingProvider::new(mock.clone(), policy().max_attempts(2));
    assert!(wrapper.chat(&request()).await.is_err());
    assert_eq!(mock.call_count(), 2);
}

/// ADR 0011: a hint longer than `max_retry_after` gives up at once, directive or
/// not, and the error keeps its hint.
#[tokio::test]
async fn directive_true_keeps_the_give_up_rule() {
    let hint = Duration::from_secs(30);
    let mock = MockProvider::build(|b| b.error(rate_limited(hint, Some(true))).text("too early"));
    let wrapper = RetryingProvider::new(mock.clone(), policy());
    let error = wrapper.chat(&request()).await.unwrap_err();
    assert_eq!(error.retry_after(), Some(hint));
    assert_eq!(mock.call_count(), 1);
}

#[tokio::test]
async fn directive_true_never_retries_after_output() {
    let mock = MockStreamingProvider::build(|b| {
        b.stream([
            MockStreamEvent::Event(StreamEvent::response_start(None, None)),
            MockStreamEvent::Event(StreamEvent::text_delta(0, "partial")),
            MockStreamEvent::Error(conflict(Some(true))),
        ])
        .response(ChatResponseBuilder::new().text("never").build())
    });
    let wrapper = RetryingProvider::new(mock.clone(), policy());
    let result = wrapper
        .chat_stream(&request())
        .await
        .unwrap()
        .collect_response()
        .await;
    assert!(result.is_err());
    assert_eq!(mock.call_count(), 1);
}

/// F12: a rate limit the retry policy did not retry is still
/// fallback-eligible.
#[tokio::test]
async fn directive_false_rate_limit_still_fails_over() {
    let primary = MockProvider::new([rate_limited(Duration::from_millis(1), Some(false))])
        .with_provider_name("primary");
    let fallback = MockProvider::with_text("backup").with_provider_name("backup");
    let wrapper = FallbackProvider::new(
        RetryingProvider::new(primary.clone(), policy()),
        fallback.clone(),
    );
    let response = wrapper.chat(&request()).await.unwrap();
    assert_eq!(response.text().as_deref(), Some("backup"));
    assert_eq!(primary.call_count(), 1);
    assert_eq!(fallback.call_count(), 1);
}

#[tokio::test]
async fn custom_classifiers_own_the_decision() {
    // Not overridden by a `true` directive...
    let mock = MockProvider::build(|b| b.error(conflict(Some(true))).text("ok"));
    let wrapper = RetryingProvider::new(mock.clone(), policy().retry_if(|_| false));
    assert!(wrapper.chat(&request()).await.is_err());
    assert_eq!(mock.call_count(), 1);

    // ...nor by a `false` one.
    let mock = MockProvider::build(|b| b.error(conflict(Some(false))).text("ok"));
    let wrapper = RetryingProvider::new(mock.clone(), policy().retry_if(|_| true));
    assert!(wrapper.chat(&request()).await.is_ok());
    assert_eq!(mock.call_count(), 2);

    // A classifier can read the directive to follow it.
    let follows = policy().retry_if(|error: &Error| error.server_retry_directive() == Some(true));
    assert!(follows.should_retry(&conflict(Some(true))));
    assert!(!follows.should_retry(&conflict(None)));
}

#[test]
fn default_classifier_follows_the_directive_ahead_of_kind_and_status() {
    let policy = RetryPolicy::new();
    assert!(policy.should_retry(&conflict(Some(true))));
    assert!(!policy.should_retry(&conflict(None)));
    assert!(!policy.should_retry(&rate_limited(Duration::ZERO, Some(false))));
    assert!(policy.should_retry(&rate_limited(Duration::ZERO, None)));
    // A failed failover is its last error, directive included.
    let fallback = conflict(Some(true)).with_fallback_from(Error::timeout("p"));
    assert!(policy.should_retry(&fallback));
}
