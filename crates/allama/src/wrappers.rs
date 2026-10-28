//! Composable provider wrappers.
//!
//! Each wrapper implements [`ChatProvider`](crate::ChatProvider),
//! [`EmbeddingProvider`](crate::EmbeddingProvider), and
//! [`DecisionProvider`](crate::DecisionProvider) over an inner provider and
//! composes with the others. Recommended order, outermost first:
//!
//! ```rust,ignore
//! TracingProvider::new(TimeoutProvider::new(
//!     FallbackProvider::new(
//!         RetryingProvider::new(TimeoutProvider::new(primary, per_attempt), RetryPolicy::default()),
//!         RetryingProvider::new(TimeoutProvider::new(backup, per_attempt), RetryPolicy::default()),
//!     ),
//!     deadline,
//! ))
//! ```
//!
//! - Tracing outermost, so one span covers retries and failover.
//! - A deadline on the whole call ([`TimeoutProvider`]) just inside
//!   tracing: a [`request`](TimeoutPolicy::request) bound, which also
//!   bounds stream setup unless [`stream_setup`](TimeoutPolicy::stream_setup)
//!   sets a bound of its own. Retry and fallback hand a stream over only at
//!   its first output, so stream setup here is the wait for that output
//!   across every attempt and backoff, the deadline a server needs before
//!   it sends a status. Not [`stream_total`](TimeoutPolicy::stream_total) in an
//!   interactive stack: it would cut off long answers after they start.
//!   Inside tracing, because a deadline outside it drops the traced call,
//!   whose span then records a cancellation instead of a timeout.
//! - Fallback around retry, so each backend gets its own retries.
//!   [`RetryingProvider`] waits out a provider's `Retry-After` hint up to
//!   its [`max_retry_after`](RetryPolicy::max_retry_after) and gives up on
//!   longer ones, so the rate limit reaches the fallback instead of an early
//!   retry. The default, 120 s, suits batch work; an interactive server sets
//!   a few seconds. Each `RetryingProvider` owns a [`RetryBudget`] that
//!   bounds its retries across calls during an outage, so each side drains
//!   its own; share one across both sides explicitly with
//!   [`RetryPolicy::budget`].
//! - Timeout ([`TimeoutProvider`]) inside retry, so each attempt is bounded
//!   and a timed-out attempt (including a stream that stalls before its
//!   first output) is retried.
//! - Request transforms ([`MapRequestProvider`]) whose result the span
//!   should show, such as redaction or sanitization, **outside** tracing:
//!   `MapRequestProvider::new(TracingProvider::new(p), redact)`. A span
//!   records the request `TracingProvider` is given, so inside it, the span
//!   shows the caller's request, before the transform: captured content
//!   would hold the text the transform removes.
//! - Model mapping ([`MapModelProvider`]) around the one provider whose
//!   catalog it targets, typically a fallback side:
//!   `FallbackProvider::new(primary, MapModelProvider::new(backup, map))`.
//! - Recording ([`RecordingProvider`]) outermost to log what the application
//!   saw, or around a single provider to log every attempt.
//!
//! `TimeoutProvider` adds workload-specific bounds on top of the transport
//! defaults of the HTTP provider crates (a connect timeout, a whole-call
//! timeout for unary requests, and generous setup and idle bounds for
//! streams, all in `allama-http`). Both fail with
//! [`Error::Timeout`](crate::Error::Timeout) phrased `"<phase> exceeded
//! <limit>"`; whichever bound is shorter fires first. Tighten with a
//! `TimeoutProvider`; loosen or disable the transport bounds in the
//! provider's builder (`timeout`, `stream_timeouts`).
//!
//! # Constructing wrappers
//!
//! Every wrapper is built the same way: `new` takes the wrapped provider and
//! the configuration that is the wrapper's purpose, and optional behaviour
//! is a chained setter. So [`RetryingProvider::new(inner, policy)`](RetryingProvider::new)
//! and [`TimeoutProvider::new(inner, policy)`](TimeoutProvider::new) read
//! alike (`RetryPolicy::default()` is the retry default),
//! [`FallbackProvider::new(primary, fallback)`](FallbackProvider::new) takes
//! both sides, and a wrapper with nothing it must be told
//! ([`ObserveProvider`], `TracingProvider`) takes only the provider and
//! offers its hooks and options as setters. A setter that registers a
//! callback the wrapper runs as calls happen is named `on_*`
//! ([`ObserveProvider::on_outcome`], `TracingProvider::on_span`); one that
//! sets configuration is named `with_*` or after the setting.
//!
//! # Writing a wrapper
//!
//! A wrapper is transparent: it changes how calls run, not who answers them.
//! Write the operations (`chat`, `chat_stream`, `embed`, `decide`) and
//! forward every question about the provider with
//! [`forward_provider_identity!`], [`forward_chat_questions!`],
//! [`forward_embedding_questions!`], [`forward_decision_questions!`] and
//! [`forward_extraction_questions!`],
//! naming the field that holds the wrapped provider. A question added to a
//! provider trait later is added to its macro in the same release, so a
//! wrapper that uses them forwards it by being recompiled. They forward:
//!
//! - [`provider_name`](crate::ProviderIdentity::provider_name) and
//!   [`downcast_any`](crate::ProviderIdentity::downcast_any)
//!   (`self.inner.downcast_any(id)`), so callers can still reach the
//!   provider behind it with
//!   [`downcast_ref`](crate::ProviderIdentityExt::downcast_ref);
//! - the capability answers (`chat_capability`, `embedding_capability`,
//!   `decision_capability`);
//! - the limits ([`embedding_limits`](crate::EmbeddingProvider::embedding_limits),
//!   [`decision_limits`](crate::DecisionProvider::decision_limits)) and
//!   request validation (`validate_chat_request`,
//!   `validate_chat_stream_request`, `validate_embedding_request`,
//!   `validate_decision_request`), whose defaults are right for providers
//!   but wrong for wrappers: a wrapper that does not forward them reports
//!   the default `Capabilities` coverage instead of the adapter's dry run.
//!
//! A wrapper that adds warnings of its own builds on
//! [`chat::prepare`](crate::adapter::chat::prepare) and passes the inner response
//! through [`Prepared::finish`](crate::adapter::chat::Prepared::finish) (or
//! `finish_stream`): warnings the inner provider already emitted as warning
//! events are not emitted again, so `finish` is safe to call at every layer.
//! Give a warning that recurs with the request's shape a
//! [code](crate::Warning::with_code), so its event goes out at `WARN` once
//! rather than on every call. A response cache or replay layer belongs
//! outside such a wrapper: the warnings of a deserialized response are
//! unmarked, so a wrapper's `finish` emits them again on every hit.
//!
//! Where a wrapper's annotations go:
//!
//! - what it changed is a coded warning of the matching kind: a redaction
//!   wrapper that rewrites the prompt adds an
//!   [`Approximated`](crate::WarningKind::Approximated) warning with code
//!   `approximated.redacted_content`;
//! - pure observations (timings, cache hits, what a lookup found) go to
//!   `tracing` or the application's own channel;
//! - it never writes provider data: not under
//!   [`ALLAMA_NAMESPACE`](crate::provider_data::ALLAMA_NAMESPACE), which
//!   is reserved for conventions allama publishes, not under a provider's
//!   namespace, whose state that provider reads back on replay, and not
//!   on blocks. [`FallbackAttempts`] is the one reserved exception;
//! - an annotation that is not a degradation has no portable place on the
//!   response yet;
//! - a wrapper that rewrites or annotates responses beyond adding warnings
//!   is not transparent: it runs the identity contract
//!   (`allama_conformance::assert_transparent_chat_wrapper`), not the data
//!   path one.
//!
//! [`RetryingProvider`] runs each attempt in an attempt scope and commits
//! the attempt the caller gets, so the warnings of discarded attempts are
//! not logged. The scope is internal in this release: a third-party wrapper
//! that calls its inner provider more than once for one call (retries,
//! hedging) logs the warnings of every attempt. [`ObserveProvider`] hands
//! every warning to a callback, for metrics.
//!
//! The name and capability methods are required, so the compiler catches a
//! wrapper that forgets them; `allama_conformance::assert_wrapper` (and its
//! per-trait variants) checks all of them, and that the provider's
//! responses, stream events and errors pass through unchanged. Wrappers that combine
//! providers ([`FallbackProvider`]), map models ([`MapModelProvider`]), or
//! transform requests ([`MapRequestProvider`]) document how they answer.
//!
//! [`RetryingProvider`] and [`FallbackProvider`] report what they did
//! through [`call_scope`] rather than recording span fields, so
//! `TracingProvider` (feature `tracing`) puts it in the call's summary. A
//! third-party resilience wrapper (hedging, a circuit breaker) appears there
//! as the attempts made under it ([What is public](call_scope#what-is-public)).
//!
//! Middleware is a wrapper too. [`MapRequestProvider`] covers a synchronous
//! request rewrite; for an async request hook (a memory or retrieval
//! lookup) or for watching responses and streams, write a wrapper. The
//! [`middleware` example]
//! is the recipe: it awaits a lookup before each call,
//! observes responses and errors, observes streams through a boxed
//! [`ChatStream`](crate::ChatStream), forwards everything above, and runs
//! `assert_wrapper` in its tests. Every provider's stream is `'static`
//! ([`ChatProvider::Stream`](crate::ChatProvider::Stream)), so a wrapper
//! boxes the inner stream with no extra bound.
//!
#![doc = concat!(
    "[`middleware` example]: https://github.com/sagikazarmark/allama/blob/v",
    env!("CARGO_PKG_VERSION"),
    "/crates/allama-examples/examples/middleware.rs",
)]

pub mod call_scope;
mod capability_override;
mod fallback;
mod forward;
mod map_model;
mod map_request;
mod observe;
mod recording;
mod retry;
mod timeout;
#[cfg(feature = "tracing")]
mod tracing;
mod unsupported_policy;

#[doc(inline)]
pub use crate::{
    __forward_chat_questions as forward_chat_questions,
    __forward_decision_questions as forward_decision_questions,
    __forward_embedding_questions as forward_embedding_questions,
    __forward_extraction_questions as forward_extraction_questions,
    __forward_provider_identity as forward_provider_identity,
};
pub use capability_override::CapabilityOverrideProvider;
pub use fallback::{FallbackAttempt, FallbackAttempts, FallbackProvider, should_fallback};
pub use map_model::MapModelProvider;
pub use map_request::MapRequestProvider;
pub use observe::{CallKind, CallOutcome, ObserveProvider, ObserveStream};
pub use recording::{ChatExchange, RecordingProvider, RecordingStream};
pub use retry::{RetryBudget, RetryPermit, RetryPolicy, RetryingProvider};
pub use timeout::{TimeoutPolicy, TimeoutProvider, TimeoutStream};
#[cfg(feature = "tracing")]
pub use tracing::{TracedRequest, TracingContentConfig, TracingProvider, otel_genai_provider_name};
pub use unsupported_policy::UnsupportedPolicyProvider;

/// `otel_genai_provider_name`: the provider name attempt events carry,
/// also without the `tracing` feature.
#[cfg(any(feature = "tracing", feature = "warning-events"))]
fn semconv_provider_name(provider_name: &str) -> &str {
    match provider_name {
        "azure" => "azure.ai.openai",
        "gemini" | "google-ai" => "gcp.gemini",
        "google-vertex" => "gcp.vertex_ai",
        "mistral" => "mistral_ai",
        "xai" => "x_ai",
        other => other,
    }
}

/// A custom error classifier, set with [`RetryPolicy::retry_if`] or
/// [`FallbackProvider::fallback_if`].
type Classifier = std::sync::Arc<dyn Fn(&crate::Error) -> bool + Send + Sync>;
