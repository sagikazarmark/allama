//! The call scope: what happened during one call, for `TracingProvider`'s
//! resilience summary.
//!
//! A wrapper that retries or fails over learns things about a call that no
//! single response carries: how many attempts were made, how long it
//! waited, why it gave up, which provider finally served. It reports each
//! as a [`CallFact`] with [`record`]. The facts go to the innermost call
//! scope around the current poll, which accumulates them into one summary
//! per call.
//!
//! `TracingProvider` (feature `tracing`) opens a scope for each call and
//! records the summary on the call's span once, when the call completes or
//! is dropped. Without a scope (no `TracingProvider` around the call, or the
//! feature off) [`record`] does nothing, so reporting is unconditional.
//!
//! # What is public
//!
//! One fact is reported from outside allama: the server a request was sent
//! to, [`CallFact::server`], which an HTTP transport records as it sends
//! each request (`allama-http`'s reqwest transport does; a custom transport
//! records it the same way). The other facts (attempts, backoff, retry
//! directives, budget denials, give-ups, failovers and the serving
//! provider) are reported by [`RetryingProvider`](super::RetryingProvider)
//! and [`FallbackProvider`](super::FallbackProvider) only. A third-party
//! retry, hedging or failover wrapper appears in the summary as the
//! attempts made under it. Reporting those facts from outside allama can
//! be added later without a breaking change.
//!
//! ```
//! use allama::wrappers::call_scope::{self, CallFact};
//!
//! /// Called by a transport just before it sends a request.
//! fn sending_to(host: &str, port: Option<u16>) {
//!     call_scope::record(CallFact::server(host.to_owned(), port));
//! }
//! # sending_to("api.example.com", Some(443));
//! ```
//!
//! # Counting
//!
//! - **Attempts.** A scope counts the attempts reported inside it. A
//!   fallback wrapper runs each provider it tries in a side of its own:
//!   the side adds the attempts reported inside it, or one when nothing
//!   inside reported any. `Fallback(Retry(a), b)` with `a` tried three times
//!   counts 3 + 1, and `Fallback(a, b)` with `a` failing counts 2. A call
//!   with no attempt reported at all counts one.
//! - **Failed providers.** A side that fails counts as one failed provider,
//!   unless sides nested in it already counted theirs, so a chain
//!   `Fallback(a, Fallback(b, c))` where every provider fails counts 3, and
//!   a retried provider counts once.
//! - **Serving provider.** The first serving provider recorded wins.
//!   Wrappers record it as their side completes, innermost first, so in
//!   `Fallback(a, Fallback(b, c))` served by `c`, the outer wrapper's report
//!   of `b` is ignored.
//! - **Tried provider.** The provider of the last side started wins: in
//!   `Fallback(a, Fallback(b, c))`, `c` once the inner wrapper tries it.
//! - **Everything else** keeps the last value reported (the last delay, the
//!   server of the last request sent, the last server directive, the last give-up reason, the error type of the
//!   last failover), or whether any was reported (a budget denial, a
//!   failover). Delays also add up.
//!
//! # Nesting and tasks
//!
//! Scopes nest: facts reach the innermost scope only. A `TracingProvider`
//! nested inside another wrapper keeps the facts reported inside it on its
//! own span; the outer span sees the outer wrappers' facts, and the inner
//! call as one attempt.
//!
//! The scope is a thread-local value installed while the call's future (or
//! its stream) is polled, so it follows the call across threads of a
//! multi-threaded runtime. Work the call spawns onto another task is polled
//! outside the scope, and what it reports is not seen.

use std::borrow::Cow;
use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll};
use std::time::Duration;

use crate::Error;

/// Report `fact` to the current call scope.
///
/// Does nothing outside a scope. See the [module docs](self) for how facts
/// combine.
pub fn record(fact: CallFact) {
    if let Some(scope) = current() {
        scope.apply(fact);
    }
}

/// Run `side` as one side of a fallback: a provider tried as an
/// alternative to another.
///
/// Facts reported inside `side` go to a scope of its own. When it completes
/// (or is dropped), they are added to the enclosing scope, with the side
/// counted as one attempt when nothing inside reported attempts, and, when
/// it fails, as one failed provider when nothing inside counted failed
/// providers. Outside a scope it only runs `side`.
///
/// For a stream, run the attempt up to its commit point inside the side,
/// so a stream that fails before its first output counts as a failed side.
pub(crate) fn side<F, T, E>(side: F) -> impl Future<Output = Result<T, E>>
where
    F: Future<Output = Result<T, E>>,
{
    Nested::new(side, Nesting::Side)
}

/// Something learned about the call in progress, reported with [`record`].
///
/// Facts are opaque: they are read back only as the summary of the call
/// (see `TracingProvider`'s resilience fields). The one a crate outside
/// allama builds is [`server`](Self::server); see
/// [What is public](self#what-is-public).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct CallFact {
    kind: FactKind,
}

impl CallFact {
    /// One physical attempt: a request sent to a provider.
    ///
    /// Only the tests report attempts one by one:
    /// [`RetryingProvider`](super::RetryingProvider) runs each attempt in a
    /// nested scope ([`attempt`](self::attempt)), and a [`side`] counts as
    /// one attempt when nothing inside it reported any.
    #[cfg(test)]
    pub(crate) fn attempt() -> Self {
        Self {
            kind: FactKind::Attempt,
        }
    }

    /// A backoff waited before the next attempt.
    #[must_use]
    pub(crate) fn delay(delay: Duration) -> Self {
        Self {
            kind: FactKind::Delay(delay),
        }
    }

    /// The provider that served the call, by its
    /// [`provider_name`](crate::ProviderIdentity::provider_name). Record it
    /// when a side completes with a response, or a stream reaches its first
    /// output; the first one recorded wins.
    #[must_use]
    pub(crate) fn served_by(provider: impl Into<Cow<'static, str>>) -> Self {
        Self {
            kind: FactKind::ServedBy(provider.into()),
        }
    }

    /// A fallback side for `provider`, by its
    /// [`provider_name`](crate::ProviderIdentity::provider_name), started.
    /// Record it as the side starts; the last one recorded wins.
    #[must_use]
    pub(crate) fn tried(provider: impl Into<Cow<'static, str>>) -> Self {
        Self {
            kind: FactKind::Tried(provider.into()),
        }
    }

    /// A request was sent to the server at `address` (a host name or IP
    /// address) and `port`: what `TracingProvider` records as the
    /// semantic conventions' `server.address` and `server.port`.
    ///
    /// An HTTP transport records it as it sends each request, so the last
    /// one recorded names the endpoint of the attempt that ended the call,
    /// the serving side after a failover. allama-http's reqwest transport
    /// records it; a custom transport records it the same way.
    #[must_use]
    pub fn server(address: impl Into<Cow<'static, str>>, port: Option<u16>) -> Self {
        Self {
            kind: FactKind::Server(address.into(), port),
        }
    }

    /// The server's retry directive (see [`Error::server_retry_directive`])
    /// decided whether to retry: `true` to retry, `false` not to.
    #[must_use]
    pub(crate) fn server_directive(retry: bool) -> Self {
        Self {
            kind: FactKind::ServerDirective(retry),
        }
    }

    /// A retry budget refused a retry the policy allowed.
    #[must_use]
    pub(crate) fn budget_denied() -> Self {
        Self {
            kind: FactKind::BudgetDenied,
        }
    }

    /// A retry layer gave up and returned the error.
    #[must_use]
    pub(crate) fn give_up(reason: GiveUpReason) -> Self {
        Self {
            kind: FactKind::GiveUp(reason),
        }
    }

    /// The call fails over to another provider after `error`.
    ///
    /// The fact keeps the error's [kind](Error::kind_name). A nested chain
    /// that failed returns its last side's error as itself (see
    /// [`Error::fallback_from`]), so this is that error's kind.
    #[must_use]
    pub(crate) fn failover(error: &Error) -> Self {
        Self {
            kind: FactKind::Failover(error.kind_name()),
        }
    }
}

/// Why a retry layer gave up and returned the error (see
/// [`CallFact::give_up`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub(crate) enum GiveUpReason {
    /// The attempt limit was reached.
    MaxAttempts,
    /// The provider asked to wait longer than the policy's hint ceiling
    /// ([`RetryPolicy::max_retry_after`](super::RetryPolicy::max_retry_after)).
    HintOverCeiling,
    /// A retry budget refused the retry.
    Budget,
    /// The classifier did not accept the error for a retry.
    NotRetryable,
}

impl GiveUpReason {
    /// The reason as recorded in `allama.retry.give_up_reason`:
    /// `max_attempts`, `hint_over_ceiling`, `budget`, or `not_retryable`.
    #[must_use]
    #[cfg_attr(not(feature = "tracing"), allow(dead_code))]
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::MaxAttempts => "max_attempts",
            Self::HintOverCeiling => "hint_over_ceiling",
            Self::Budget => "budget",
            Self::NotRetryable => "not_retryable",
        }
    }
}

/// One call's accumulator. The scope that a `TracingProvider` opens is the
/// root; [`side`] and the retry wrapper's attempts open nested ones.
#[derive(Debug, Default)]
pub(crate) struct Scope {
    summary: Mutex<Summary>,
}

impl Scope {
    /// A new, empty scope.
    pub(crate) fn new() -> Arc<Self> {
        Arc::default()
    }

    /// A copy of what was reported so far. The lock is released before the
    /// caller reads it, so recording the summary never holds it.
    #[cfg_attr(not(feature = "tracing"), allow(dead_code))]
    pub(crate) fn summary(&self) -> Summary {
        self.lock().clone()
    }

    fn apply(&self, fact: CallFact) {
        self.lock().apply(fact);
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Summary> {
        // Nothing panics while holding the lock, but a poisoned lock still
        // holds a consistent summary.
        self.summary.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// What the facts reported in a scope add up to.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(not(feature = "tracing"), allow(dead_code))]
pub(crate) struct Summary {
    /// Physical attempts reported.
    pub(crate) attempts: usize,
    /// The sum of every backoff waited.
    pub(crate) total_delay: Duration,
    /// The last backoff waited.
    pub(crate) last_delay: Option<Duration>,
    /// The last server retry directive that decided.
    pub(crate) server_directive: Option<bool>,
    /// Whether a budget denied a retry.
    pub(crate) budget_denied: bool,
    /// Why the last retry layer to give up gave up.
    pub(crate) give_up: Option<GiveUpReason>,
    /// Whether the call failed over.
    pub(crate) failed_over: bool,
    /// The error type of the last failure that triggered a failover.
    pub(crate) failover_error: Option<&'static str>,
    /// Providers that failed, each counted once (see the module docs).
    pub(crate) failed_providers: usize,
    /// The first serving provider reported.
    pub(crate) served_by: Option<Cow<'static, str>>,
    /// The provider of the last fallback side started.
    pub(crate) tried: Option<Cow<'static, str>>,
    /// The server the last request was sent to, and its port.
    pub(crate) server: Option<(Cow<'static, str>, Option<u16>)>,
}

impl Summary {
    /// Attempts as a call reports them: at least one.
    #[cfg_attr(not(feature = "tracing"), allow(dead_code))]
    pub(crate) fn call_attempts(&self) -> usize {
        self.attempts.max(1)
    }

    fn apply(&mut self, fact: CallFact) {
        match fact.kind {
            FactKind::Attempt => self.attempts += 1,
            FactKind::Delay(delay) => {
                self.total_delay = self.total_delay.saturating_add(delay);
                self.last_delay = Some(delay);
            }
            FactKind::ServedBy(provider) => {
                self.served_by.get_or_insert(provider);
            }
            FactKind::Tried(provider) => self.tried = Some(provider),
            FactKind::ServerDirective(retry) => self.server_directive = Some(retry),
            FactKind::Server(address, port) => self.server = Some((address, port)),
            FactKind::BudgetDenied => self.budget_denied = true,
            FactKind::GiveUp(reason) => self.give_up = Some(reason),
            FactKind::Failover(error_type) => {
                self.failed_over = true;
                self.failover_error = Some(error_type);
            }
        }
    }

    /// Add what a nested scope reported, after it ended. `failed` is whether
    /// it ended in an error. A nested scope ran after every fact already
    /// here, so its last values replace ours, except the serving provider,
    /// where the first wins.
    fn absorb(&mut self, nested: Summary, nesting: Nesting, failed: bool) {
        self.attempts += nested.attempts.max(1);
        self.failed_providers = match nesting {
            Nesting::Side if failed => self.failed_providers + nested.failed_providers.max(1),
            Nesting::Side => self.failed_providers + nested.failed_providers,
            // Every attempt of a retry tries the same providers again.
            Nesting::Attempt => self.failed_providers.max(nested.failed_providers),
        };
        self.total_delay = self.total_delay.saturating_add(nested.total_delay);
        self.last_delay = nested.last_delay.or(self.last_delay);
        self.server_directive = nested.server_directive.or(self.server_directive);
        self.budget_denied |= nested.budget_denied;
        self.give_up = nested.give_up.or(self.give_up);
        self.failed_over |= nested.failed_over;
        self.failover_error = nested.failover_error.or(self.failover_error);
        self.server = nested.server.or(self.server.take());
        self.tried = nested.tried.or(self.tried.take());
        if self.served_by.is_none() {
            self.served_by = nested.served_by;
        }
    }
}

/// Install `scope` for the duration of each poll of `future`.
#[cfg_attr(not(feature = "tracing"), allow(dead_code))]
pub(crate) fn within<F: Future>(scope: &Arc<Scope>, future: F) -> Within<F> {
    Within {
        scope: Arc::clone(scope),
        future: Box::pin(future),
    }
}

/// Install `scope` as the current one until the guard drops, which restores
/// the scope that was current before, also when the poll panics.
#[cfg_attr(not(feature = "tracing"), allow(dead_code))]
pub(crate) fn enter(scope: &Arc<Scope>) -> Entered {
    let previous = CURRENT
        .try_with(|current| current.replace(Some(Arc::clone(scope))))
        .ok()
        .flatten();
    Entered { previous }
}

/// Run one attempt of a retry layer: like a [`side`], it counts as one
/// attempt when nothing inside reported any, but a failed attempt is not a
/// failed provider.
pub(crate) fn attempt<F, T, E>(attempt: F) -> impl Future<Output = Result<T, E>>
where
    F: Future<Output = Result<T, E>>,
{
    Nested::new(attempt, Nesting::Attempt)
}

/// A future polled with a scope installed (see [`within`]).
///
/// The future is boxed, here and in [`Nested`]: wrappers nest these once per
/// layer, and a future held inline would be copied into every layer's state.
pub(crate) struct Within<F> {
    scope: Arc<Scope>,
    future: Pin<Box<F>>,
}

impl<F: Future> Future for Within<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<F::Output> {
        let this = self.get_mut();
        let _entered = enter(&this.scope);
        this.future.as_mut().poll(cx)
    }
}

/// Restores the previous scope when dropped (see [`enter`]).
pub(crate) struct Entered {
    previous: Option<Arc<Scope>>,
}

impl Drop for Entered {
    fn drop(&mut self) {
        let previous = self.previous.take();
        let _ = CURRENT.try_with(|current| current.set(previous));
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FactKind {
    #[cfg_attr(not(test), allow(dead_code))]
    Attempt,
    Delay(Duration),
    ServedBy(Cow<'static, str>),
    Tried(Cow<'static, str>),
    Server(Cow<'static, str>, Option<u16>),
    ServerDirective(bool),
    BudgetDenied,
    GiveUp(GiveUpReason),
    Failover(&'static str),
}

/// How a nested scope adds to the one around it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Nesting {
    /// A provider tried as an alternative ([`side`]).
    Side,
    /// One attempt of a retry layer ([`attempt`]).
    Attempt,
}

/// A future run in a scope of its own, nested in the scope current when it
/// is first polled (see [`side`] and [`attempt`]).
struct Nested<F> {
    future: Pin<Box<F>>,
    nesting: Nesting,
    started: bool,
    /// `None` outside a scope, and once the future completed.
    absorb: Option<Absorb>,
}

impl<F> Nested<F> {
    fn new(future: F, nesting: Nesting) -> Self {
        Self {
            future: Box::pin(future),
            nesting,
            started: false,
            absorb: None,
        }
    }
}

impl<F, T, E> Future for Nested<F>
where
    F: Future<Output = Result<T, E>>,
{
    type Output = Result<T, E>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if !this.started {
            this.started = true;
            this.absorb = current().map(|parent| Absorb {
                parent,
                nested: Scope::new(),
                nesting: this.nesting,
                failed: false,
            });
        }
        let poll = match &this.absorb {
            Some(absorb) => {
                let _entered = enter(&absorb.nested);
                this.future.as_mut().poll(cx)
            }
            None => this.future.as_mut().poll(cx),
        };
        if let Poll::Ready(output) = &poll
            && let Some(mut absorb) = this.absorb.take()
        {
            // Dropping it adds the nested scope to its parent.
            absorb.failed = output.is_err();
        }
        poll
    }
}

/// Adds a nested scope to its parent when dropped: after the nested future
/// completed, or when it was dropped before.
struct Absorb {
    parent: Arc<Scope>,
    nested: Arc<Scope>,
    nesting: Nesting,
    failed: bool,
}

impl Drop for Absorb {
    fn drop(&mut self) {
        let nested = std::mem::take(&mut *self.nested.lock());
        self.parent.lock().absorb(nested, self.nesting, self.failed);
    }
}

thread_local! {
    /// The innermost call scope of the future being polled on this thread.
    static CURRENT: Cell<Option<Arc<Scope>>> = const { Cell::new(None) };
}

/// The current scope, if any.
fn current() -> Option<Arc<Scope>> {
    CURRENT
        .try_with(|current| {
            let scope = current.take();
            let copy = scope.clone();
            current.set(scope);
            copy
        })
        .ok()
        .flatten()
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::pin::pin;
    use std::task::Waker;

    use super::*;

    /// The server of the last request sent wins,
    /// across a fallback's sides.
    #[test]
    fn the_last_server_reported_wins() {
        let mut summary = Summary::default();
        summary.apply(CallFact::server("a.internal", Some(1)));
        let mut side = Summary::default();
        side.apply(CallFact::server("b.internal", None));
        summary.absorb(side, Nesting::Side, false);
        assert_eq!(summary.server, Some(("b.internal".into(), None)));
        // A side that sent nothing keeps what was sent before.
        summary.absorb(Summary::default(), Nesting::Side, true);
        assert_eq!(summary.server, Some(("b.internal".into(), None)));
    }

    /// The last side tried wins, and a nested side that tried none keeps
    /// what was tried before.
    #[test]
    fn the_last_side_tried_wins() {
        let mut summary = Summary::default();
        summary.apply(CallFact::tried("a"));
        let mut side = Summary::default();
        side.apply(CallFact::tried("b"));
        side.apply(CallFact::tried("c"));
        summary.absorb(side, Nesting::Side, true);
        assert_eq!(summary.tried.as_deref(), Some("c"));
        summary.absorb(Summary::default(), Nesting::Attempt, true);
        assert_eq!(summary.tried.as_deref(), Some("c"));
    }

    type Outcome = Result<(), ()>;

    /// Run `future` inside a new root scope and return what it reported.
    async fn summarize<F: Future>(future: F) -> (F::Output, Summary) {
        let scope = Scope::new();
        let output = within(&scope, future).await;
        (output, scope.summary())
    }

    fn served_by(summary: &Summary) -> Option<&str> {
        summary.served_by.as_deref()
    }

    #[tokio::test]
    async fn record_outside_a_scope_does_nothing() {
        record(CallFact::attempt());
        assert!(current().is_none());
        let output: Outcome = side(async { Ok(()) }).await;
        assert!(output.is_ok());
    }

    #[tokio::test]
    async fn facts_reach_the_current_scope() {
        let ((), summary) = summarize(async {
            record(CallFact::attempt());
            record(CallFact::delay(Duration::from_millis(5)));
            record(CallFact::delay(Duration::from_millis(7)));
            record(CallFact::server_directive(true));
            record(CallFact::server_directive(false));
            record(CallFact::give_up(GiveUpReason::HintOverCeiling));
            record(CallFact::give_up(GiveUpReason::MaxAttempts));
            record(CallFact::budget_denied());
            record(CallFact::attempt());
        })
        .await;
        assert_eq!(
            summary,
            Summary {
                attempts: 2,
                total_delay: Duration::from_millis(12),
                last_delay: Some(Duration::from_millis(7)),
                server_directive: Some(false),
                budget_denied: true,
                give_up: Some(GiveUpReason::MaxAttempts),
                ..Summary::default()
            }
        );
    }

    #[tokio::test]
    async fn a_call_without_attempts_counts_one() {
        let ((), summary) = summarize(async {}).await;
        assert_eq!(summary.attempts, 0);
        assert_eq!(summary.call_attempts(), 1);
    }

    #[tokio::test]
    async fn the_first_serving_provider_wins() {
        let ((), summary) = summarize(async {
            record(CallFact::served_by("inner"));
            record(CallFact::served_by("outer"));
        })
        .await;
        assert_eq!(served_by(&summary), Some("inner"));
    }

    /// A side without reported attempts is one attempt; one with reported
    /// attempts adds them.
    #[tokio::test]
    async fn sides_count_attempts() {
        let (_, summary) = summarize(async {
            let first: Outcome = side(async {
                for _ in 0..3 {
                    record(CallFact::attempt());
                }
                Err(())
            })
            .await;
            let second: Outcome = side(async { Ok(()) }).await;
            (first, second)
        })
        .await;
        assert_eq!(summary.attempts, 4);
        assert_eq!(summary.failed_providers, 1);
    }

    /// `Fallback(a, Fallback(b, c))`: the failed providers of a nested chain
    /// count once each, and the innermost serving provider wins.
    #[tokio::test]
    async fn nested_sides_count_each_failed_provider_once() {
        async fn chain(c_serves: bool) -> Summary {
            let (_, summary) = summarize(async {
                let _: Outcome = side(async { Err(()) }).await;
                record(CallFact::failover(&Error::overloaded("a")));
                let inner: Outcome = side(async {
                    let _: Outcome = side(async { Err(()) }).await;
                    record(CallFact::failover(&Error::timeout("b")));
                    let c: Outcome = side(async { if c_serves { Ok(()) } else { Err(()) } }).await;
                    if c.is_ok() {
                        record(CallFact::served_by("c"));
                    }
                    c
                })
                .await;
                if inner.is_ok() {
                    record(CallFact::served_by("b"));
                }
            })
            .await;
            summary
        }

        let served = chain(true).await;
        assert_eq!(served.attempts, 3);
        assert_eq!(served.failed_providers, 2);
        assert_eq!(served_by(&served), Some("c"));
        assert!(served.failed_over);
        assert_eq!(served.failover_error, Some("timeout"));

        let failed = chain(false).await;
        assert_eq!(failed.attempts, 3);
        assert_eq!(failed.failed_providers, 3);
        assert_eq!(served_by(&failed), None);
    }

    /// Retry attempts count like sides, but a failed attempt is not a failed
    /// provider; attempts that each fail over count each provider once.
    #[tokio::test]
    async fn retry_attempts_count_attempts_not_failed_providers() {
        let (_, summary) = summarize(async {
            for _ in 0..2 {
                let _: Outcome = attempt(async {
                    let _: Outcome = side(async { Err(()) }).await;
                    side(async { Err(()) }).await
                })
                .await;
            }
            let plain: Outcome = attempt(async { Err(()) }).await;
            plain
        })
        .await;
        assert_eq!(summary.attempts, 5);
        assert_eq!(summary.failed_providers, 2);
    }

    #[tokio::test]
    async fn nested_last_values_replace_earlier_ones() {
        let (_, summary) = summarize(async {
            record(CallFact::delay(Duration::from_millis(1)));
            record(CallFact::give_up(GiveUpReason::NotRetryable));
            let _: Outcome = side(async {
                record(CallFact::delay(Duration::from_millis(4)));
                record(CallFact::give_up(GiveUpReason::Budget));
                record(CallFact::budget_denied());
                Err(())
            })
            .await;
        })
        .await;
        assert_eq!(summary.total_delay, Duration::from_millis(5));
        assert_eq!(summary.last_delay, Some(Duration::from_millis(4)));
        assert_eq!(summary.give_up, Some(GiveUpReason::Budget));
        assert!(summary.budget_denied);
    }

    /// A failover's error type is that of the error that ended a nested
    /// chain, whatever it fell back from.
    #[test]
    fn failover_reads_the_error_that_ended_a_chain() {
        let chain = Error::rate_limited("b").with_fallback_from(Error::timeout("a"));
        assert_eq!(
            CallFact::failover(&chain),
            CallFact::failover(&Error::rate_limited("c"))
        );
    }

    #[test]
    fn give_up_reasons_have_stable_names() {
        let names = [
            GiveUpReason::MaxAttempts,
            GiveUpReason::HintOverCeiling,
            GiveUpReason::Budget,
            GiveUpReason::NotRetryable,
        ]
        .map(GiveUpReason::as_str);
        assert_eq!(
            names,
            [
                "max_attempts",
                "hint_over_ceiling",
                "budget",
                "not_retryable"
            ]
        );
    }

    /// A side that fails after waiting counts as failed when it completes,
    /// not when it first waits.
    #[tokio::test]
    async fn a_side_that_waits_then_fails_counts_as_failed() {
        let (_, summary) = summarize(async {
            let _: Outcome = side(async {
                YieldOnce(false).await;
                Err(())
            })
            .await;
        })
        .await;
        assert_eq!(summary.attempts, 1);
        assert_eq!(summary.failed_providers, 1);
    }

    /// A side dropped before it completes still adds its attempts.
    #[test]
    fn a_dropped_side_still_counts() {
        let root = Scope::new();
        {
            let mut call = pin!(within(&root, async {
                let _: Outcome = side(async {
                    record(CallFact::attempt());
                    record(CallFact::attempt());
                    std::future::pending().await
                })
                .await;
            }));
            let mut cx = Context::from_waker(Waker::noop());
            assert!(call.as_mut().poll(&mut cx).is_pending());
        }
        assert_eq!(root.summary().attempts, 2);
    }

    /// The previous scope comes back after nesting and after a panicking
    /// poll.
    #[test]
    fn the_previous_scope_is_restored() {
        let outer = Scope::new();
        let inner = Scope::new();
        {
            let _outer = enter(&outer);
            {
                let _inner = enter(&inner);
                record(CallFact::attempt());
            }
            record(CallFact::budget_denied());

            let panicked = catch_unwind(AssertUnwindSafe(|| {
                let _inner = enter(&inner);
                panic!("poll panicked");
            }));
            assert!(panicked.is_err());
            record(CallFact::attempt());
        }
        assert!(current().is_none());
        assert_eq!(inner.summary().attempts, 1);
        let outer = outer.summary();
        assert_eq!(outer.attempts, 1);
        assert!(outer.budget_denied);
    }

    /// Work spawned onto another task runs outside the scope.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spawned_work_is_not_seen() {
        let ((), summary) = summarize(async {
            tokio::spawn(async { record(CallFact::attempt()) })
                .await
                .unwrap();
        })
        .await;
        assert_eq!(summary.attempts, 0);
    }

    /// The scope follows the call between polls, whichever thread polls it.
    #[test]
    fn the_scope_is_installed_only_while_polling() {
        let root = Scope::new();
        let mut call = pin!(within(&root, async {
            record(CallFact::attempt());
            YieldOnce(false).await;
            record(CallFact::attempt());
        }));
        let mut cx = Context::from_waker(Waker::noop());
        assert!(call.as_mut().poll(&mut cx).is_pending());
        assert!(current().is_none());
        record(CallFact::budget_denied());
        assert!(call.as_mut().poll(&mut cx).is_ready());
        let summary = root.summary();
        assert_eq!(summary.attempts, 2);
        assert!(!summary.budget_denied);
    }

    /// Waits once, then completes.
    struct YieldOnce(bool);

    impl Future for YieldOnce {
        type Output = ();

        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
            if self.0 {
                Poll::Ready(())
            } else {
                self.0 = true;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }

    /// A retry wrapper outside `RetryingProvider`: tries its provider up to
    /// three times and reports each attempt, backoff, and give-up.
    struct ThreeTries<P>(P);

    impl<P: crate::ChatProvider> ThreeTries<P> {
        async fn chat(&self, request: &crate::ChatRequest) -> crate::Result<crate::ChatResponse> {
            let mut attempt = 1;
            loop {
                record(CallFact::attempt());
                match self.0.chat(request).await {
                    Ok(response) => return Ok(response),
                    Err(error) if attempt == 3 => {
                        record(CallFact::give_up(GiveUpReason::MaxAttempts));
                        return Err(error);
                    }
                    Err(_) => {
                        record(CallFact::delay(Duration::from_millis(2)));
                        attempt += 1;
                    }
                }
            }
        }
    }

    /// A fallback wrapper outside `FallbackProvider`: runs each provider in
    /// a [`side`].
    async fn or_else<P: crate::ChatProvider>(
        first: &ThreeTries<P>,
        backup: &crate::mock::MockProvider,
        request: &crate::ChatRequest,
    ) -> crate::Result<crate::ChatResponse> {
        use crate::{ChatProvider, ProviderIdentity};

        let error = match side(first.chat(request)).await {
            Ok(response) => {
                record(CallFact::served_by(first.0.provider_name().to_owned()));
                return Ok(response);
            }
            Err(error) => error,
        };
        record(CallFact::failover(&error));
        let response = side(backup.chat(request)).await?;
        record(CallFact::served_by(backup.provider_name().to_owned()));
        Ok(response)
    }

    /// Wrappers that report through the call scope add up in one summary:
    /// a retry layer alone, and under a fallback that counts sides (three
    /// attempts, plus one).
    #[tokio::test]
    async fn wrappers_report_into_one_summary() {
        use crate::mock::MockProvider;

        let request = crate::ChatRequest::new("m").user("q");
        let retry = ThreeTries(MockProvider::build(|b| {
            b.error(Error::overloaded("busy"))
                .error(Error::overloaded("busy"))
                .text("ok")
        }));
        let (result, summary) = summarize(retry.chat(&request)).await;
        result.unwrap();
        assert_eq!(
            summary,
            Summary {
                attempts: 3,
                total_delay: Duration::from_millis(4),
                last_delay: Some(Duration::from_millis(2)),
                ..Summary::default()
            }
        );

        let retry = ThreeTries(MockProvider::new([
            Error::overloaded("busy"),
            Error::overloaded("busy"),
            Error::rate_limited("slow down"),
        ]));
        let backup = MockProvider::with_text("backup").with_provider_name("backup");
        let (result, summary) = summarize(or_else(&retry, &backup, &request)).await;
        result.unwrap();
        assert_eq!(
            summary,
            Summary {
                attempts: 4,
                total_delay: Duration::from_millis(4),
                last_delay: Some(Duration::from_millis(2)),
                give_up: Some(GiveUpReason::MaxAttempts),
                failed_over: true,
                failover_error: Some("rate_limited"),
                failed_providers: 1,
                served_by: Some("backup".into()),
                ..Summary::default()
            }
        );
    }
}
