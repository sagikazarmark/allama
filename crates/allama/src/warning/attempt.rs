//! Attempt scopes: warning events held back until an attempt commits.

use std::cell::RefCell;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll};

use futures_core::Stream;

#[cfg(feature = "warning-events")]
use super::events::{self, Pending};

/// One attempt of a call that may be retried or abandoned: the warning
/// events [`emit`](super::emit) sends while it runs wait until the wrapper
/// decides whether the attempt counts.
///
/// A retry or hedging wrapper creates a scope per attempt, runs the attempt
/// through [`instrument`](Self::instrument) (or
/// [`instrument_stream`](Self::instrument_stream)), and then decides:
///
/// - [`commit`](Self::commit) when the attempt's result is what the caller
///   gets: a response, a stream at its first output, or the error of the
///   last attempt. The held events go out.
/// - [`discard`](Self::discard) when the attempt is retried or loses a
///   hedge. The held events are dropped, so a retried call logs each
///   warning once.
///
/// A scope dropped undecided (a wrapper returned early, or its future was
/// dropped) flushes as a commit does, so no event is lost silently. After the
/// decision, events the instrumented future or stream still emits go
/// straight out (after a commit) or are dropped (after a discard).
/// [`RetryingProvider`](crate::wrappers::RetryingProvider) runs every
/// attempt in one.
///
/// # Nesting
///
/// Scopes nest: an event goes to the nearest scope being polled, and a scope
/// that commits inside another hands its events to the outer one instead of
/// to `tracing`. Only the outermost scope's flush decides each event's level
/// and uses up its deduplication key, so in `Retry(Fallback(Retry(a), ..))`
/// a discarded outer attempt also discards what its inner retries committed.
///
/// The scope follows the instrumented future across threads: it is installed
/// on the polling thread for each poll, and the buffer it fills is shared
/// with this handle, so the decision can be made after
/// [`instrument`](Self::instrument) took the future. Work spawned onto
/// another task is outside the scope. Without the `warning-events` feature
/// a scope holds nothing.
#[must_use = "an attempt scope holds the attempt's warning events until it is committed or discarded"]
pub(crate) struct AttemptScope {
    buffer: Arc<Buffer>,
}

impl AttemptScope {
    /// A scope for one attempt, open until [`commit`](Self::commit) or
    /// [`discard`](Self::discard).
    pub(crate) fn new() -> Self {
        Self {
            buffer: Arc::new(Buffer::default()),
        }
    }

    /// Run `future` in the scope: the scope is installed for each poll.
    pub(crate) fn instrument<F: Future>(&self, future: F) -> Instrumented<F> {
        Instrumented {
            inner: Box::pin(future),
            buffer: Arc::clone(&self.buffer),
        }
    }

    /// Run `stream` in the scope: the scope is installed for each poll.
    ///
    /// Instrument an attempt's stream as well as its future when something
    /// polls the stream before the attempt is decided, such as
    /// [`until_committed`](crate::chat::stream::until_committed) waiting for its
    /// first output.
    pub(crate) fn instrument_stream<S: Stream>(&self, stream: S) -> Instrumented<S> {
        Instrumented {
            inner: Box::pin(stream),
            buffer: Arc::clone(&self.buffer),
        }
    }

    /// The attempt counts: its held events go out, to the enclosing scope
    /// when there is one, and later events pass straight through.
    pub(crate) fn commit(self) {
        // `Drop` flushes an undecided scope.
        drop(self);
    }

    /// The attempt does not count: its held events are dropped, and so are
    /// later ones.
    pub(crate) fn discard(self) {
        drop(self.buffer.decide(Decision::Discarded));
    }
}

impl Default for AttemptScope {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for AttemptScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.buffer.lock();
        f.debug_struct("AttemptScope")
            .field("decision", &state.decision)
            .field("held", &state.held())
            .finish()
    }
}

impl Drop for AttemptScope {
    fn drop(&mut self) {
        let held = self.buffer.decide(Decision::Committed);
        flush(held);
    }
}

/// A future or stream run in an [`AttemptScope`] (see
/// [`AttemptScope::instrument`]).
#[must_use = "futures and streams do nothing unless polled"]
pub(crate) struct Instrumented<T> {
    inner: Pin<Box<T>>,
    buffer: Arc<Buffer>,
}

impl<T> fmt::Debug for Instrumented<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Instrumented").finish_non_exhaustive()
    }
}

impl<F: Future> Future for Instrumented<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<F::Output> {
        let this = self.get_mut();
        let _entered = enter(&this.buffer);
        this.inner.as_mut().poll(cx)
    }
}

impl<S: Stream> Stream for Instrumented<S> {
    type Item = S::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<S::Item>> {
        let this = self.get_mut();
        let _entered = enter(&this.buffer);
        this.inner.as_mut().poll_next(cx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

/// Hold `pending` in the nearest open scope being polled on this thread, or
/// hand it back to go out now. A committed scope passes it on to the next
/// one out; a discarded scope drops it.
#[cfg(feature = "warning-events")]
pub(super) fn hold(pending: Pending) -> Option<Pending> {
    let mut pending = Some(pending);
    // Dropped after the locks are released: dropping holds a span's last
    // reference, which calls into the subscriber.
    let mut discarded = None;
    let _ = STACK.try_with(|stack| {
        for buffer in stack.borrow().iter().rev() {
            let mut state = buffer.lock();
            match state.decision {
                Decision::Open => {
                    state.events.extend(pending.take());
                    return;
                }
                Decision::Discarded => {
                    discarded = pending.take();
                    return;
                }
                Decision::Committed => {}
            }
        }
    });
    drop(discarded);
    pending
}

/// Send the events a scope held to the enclosing scope, or out.
fn flush(held: Held) {
    #[cfg(feature = "warning-events")]
    for pending in held {
        if let Some(pending) = hold(pending) {
            events::fire(&pending);
        }
    }
    #[cfg(not(feature = "warning-events"))]
    let _ = held;
}

/// What a scope holds, shared by its handle and what it instruments.
#[derive(Default)]
struct Buffer {
    state: Mutex<State>,
}

impl Buffer {
    fn lock(&self) -> MutexGuard<'_, State> {
        // Nothing panics while holding the lock; a poisoned buffer is still
        // consistent.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Decide an open scope and take what it held; nothing when it was
    /// decided already.
    fn decide(&self, decision: Decision) -> Held {
        let mut state = self.lock();
        if state.decision != Decision::Open {
            return Held::default();
        }
        state.decision = decision;
        std::mem::take(&mut state.events)
    }
}

#[derive(Default)]
struct State {
    decision: Decision,
    events: Held,
}

impl State {
    fn held(&self) -> usize {
        self.events.len()
    }
}

#[cfg(feature = "warning-events")]
type Held = Vec<Pending>;
#[cfg(not(feature = "warning-events"))]
type Held = Vec<std::convert::Infallible>;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Decision {
    #[default]
    Open,
    Committed,
    Discarded,
}

thread_local! {
    /// The scopes of the futures being polled on this thread, innermost
    /// last.
    static STACK: RefCell<Vec<Arc<Buffer>>> = const { RefCell::new(Vec::new()) };
}

/// Install `buffer` as the innermost scope until the guard drops. Without
/// the `warning-events` feature nothing is held, so nothing is installed.
fn enter(buffer: &Arc<Buffer>) -> Entered {
    #[cfg(feature = "warning-events")]
    let entered = STACK
        .try_with(|stack| stack.borrow_mut().push(Arc::clone(buffer)))
        .is_ok();
    #[cfg(not(feature = "warning-events"))]
    let entered = {
        let _ = buffer;
        false
    };
    Entered { entered }
}

/// Removes the scope [`enter`] installed, also when the poll panics.
struct Entered {
    entered: bool,
}

impl Drop for Entered {
    fn drop(&mut self) {
        if self.entered {
            let _ = STACK.try_with(|stack| stack.borrow_mut().pop());
        }
    }
}

#[cfg(all(test, feature = "warning-events"))]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::mock::{MockProvider, MockResponse};
    use crate::warning::capture::{Captured, captured, captured_on_threads};
    use crate::wrappers::{FallbackProvider, RetryPolicy, RetryingProvider};
    use crate::{
        CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, Error,
        ProviderIdentity, ReasoningEffort, Result, Subject, Warning, WarningKind,
    };

    fn coded(text: &'static str) -> Warning {
        Warning::new(WarningKind::Other, Subject::Response, text).with_code("other.scope_test")
    }

    /// Emit `warning` (a fresh copy) for `model`.
    fn emit(model: &str, warning: &Warning) {
        let mut warning = warning.clone();
        crate::warning::emit("p", model, &mut warning);
    }

    fn for_model<'a>(events: &'a [Captured], model: &str) -> Vec<&'a Captured> {
        events
            .iter()
            .filter(|event| event.model.as_deref() == Some(model))
            .collect()
    }

    fn messages<'a>(events: &[&'a Captured]) -> Vec<&'a str> {
        events.iter().map(|event| event.message.as_str()).collect()
    }

    /// Count the events at each level: `(warn, debug)`.
    fn levels(events: &[&Captured]) -> (usize, usize) {
        let count = |level| events.iter().filter(|e| e.level == level).count();
        (count(tracing::Level::WARN), count(tracing::Level::DEBUG))
    }

    #[test]
    fn a_scope_holds_events_until_it_is_decided() {
        let (held_until_commit, events) = captured(async {
            // Discarded: nothing goes out, also what is emitted afterwards.
            let scope = AttemptScope::new();
            scope
                .instrument(async { emit("scope-discard", &coded("held")) })
                .await;
            let mut late = scope.instrument_stream(futures_util::stream::once(async {
                emit("scope-discard", &coded("late"));
            }));
            scope.discard();
            futures_util::StreamExt::next(&mut late).await;

            // Committed: held events go out, later ones pass straight through.
            let scope = AttemptScope::new();
            scope
                .instrument(async { emit("scope-commit", &coded("first")) })
                .await;
            let held = scope.buffer.lock().held();
            let late = scope.instrument(async { emit("scope-commit", &coded("after")) });
            scope.commit();
            late.await;

            // Dropped undecided: flushed.
            let scope = AttemptScope::new();
            scope
                .instrument(async { emit("scope-drop", &coded("dropped")) })
                .await;
            drop(scope);
            held
        });
        assert_eq!(held_until_commit, 1, "held until the commit");
        assert!(for_model(&events, "scope-discard").is_empty());
        assert_eq!(
            messages(&for_model(&events, "scope-commit")),
            ["first", "after"]
        );
        assert_eq!(messages(&for_model(&events, "scope-drop")), ["dropped"]);
    }

    /// An inner scope that commits flushes into the outer one; a discarded
    /// outer attempt drops it, without using up the dedupe key.
    #[test]
    fn nested_scopes_flush_into_the_enclosing_one() {
        let ((), events) = captured(async {
            let outer = AttemptScope::new();
            outer
                .instrument(async {
                    let inner = AttemptScope::new();
                    inner
                        .instrument(async { emit("nested", &coded("inner")) })
                        .await;
                    inner.commit();
                })
                .await;
            outer.discard();

            let outer = AttemptScope::new();
            outer
                .instrument(async {
                    let inner = AttemptScope::new();
                    inner
                        .instrument(async { emit("nested", &coded("inner")) })
                        .await;
                    inner.commit();
                })
                .await;
            outer.commit();
        });
        let nested = for_model(&events, "nested");
        // One event: the discarded attempt's never went out, and its key was
        // not spent, so the committed one is the first `WARN`.
        assert_eq!(levels(&nested), (1, 0), "{nested:?}");
        assert!(nested[0].repeats);
    }

    /// A retry wrapper outside `RetryingProvider`: tries up to three times,
    /// each attempt in an attempt scope, yielding to the runtime between
    /// attempts so the call can move to another worker thread.
    struct ThreeTries<P>(P);

    impl<P: ProviderIdentity> ProviderIdentity for ThreeTries<P> {
        fn provider_name(&self) -> &str {
            self.0.provider_name()
        }

        fn reads_namespace(&self, namespace: &str) -> bool {
            self.0.reads_namespace(namespace)
        }
    }

    impl<P: ChatProvider> ChatProvider for ThreeTries<P> {
        type Stream = P::Stream;

        async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
            let mut attempt = 1;
            loop {
                let scope = AttemptScope::new();
                let result = scope.instrument(self.0.chat(request)).await;
                match result {
                    Err(error) if error.is_retryable() && attempt < 3 => {
                        scope.discard();
                        attempt += 1;
                        tokio::task::yield_now().await;
                    }
                    result => {
                        scope.commit();
                        return result;
                    }
                }
            }
        }

        async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
            self.0.chat_stream(request).await
        }

        fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
            self.0.chat_capability(model, capability)
        }
    }

    /// A provider that fails twice, each time after a delay that lets the
    /// runtime move the call between threads, then answers.
    fn flaky(name: &'static str) -> MockProvider {
        let delay = Duration::from_millis(2);
        MockProvider::new([
            MockResponse::delayed(delay, Error::timeout("first")),
            MockResponse::delayed(delay, Error::timeout("second")),
            MockResponse::delayed(delay, ChatResponse::default()),
        ])
        .with_provider_name(name)
        .with_chat_capability(
            ChatCapability::ReasoningConfig,
            CapabilitySupport::Unsupported,
        )
    }

    fn request(model: &str) -> ChatRequest {
        ChatRequest::new(model)
            .user("q")
            .reasoning(ReasoningEffort::High)
    }

    fn fast() -> RetryPolicy {
        RetryPolicy::new()
            .base_delay(Duration::ZERO)
            .max_delay(Duration::ZERO)
            .no_budget()
    }

    /// On a multi-thread runtime the scope is installed on whichever thread
    /// polls the attempt, so a retry wrapper built on `AttemptScope`, and
    /// retries nested inside retries, emit each warning once.
    #[test]
    fn retries_emit_once_on_a_multi_thread_runtime() {
        let ((), events) = captured_on_threads("mt-", async {
            let mut calls = Vec::new();
            for call in 0..16 {
                // A scoped retry wrapper on its own.
                calls.push(tokio::spawn(async move {
                    let provider = ThreeTries(flaky("three-tries"));
                    provider.chat(&request(&format!("mt-tries-{call}"))).await
                }));
                // Retry inside retry: an inner attempt that commits flushes
                // into the outer attempt, which commits once.
                calls.push(tokio::spawn(async move {
                    let provider = ThreeTries(RetryingProvider::new(
                        flaky("nested"),
                        fast().max_attempts(2),
                    ));
                    provider.chat(&request(&format!("mt-nested-{call}"))).await
                }));
                // `Retry(Fallback(Retry(a), Retry(b)))`, where `a` never answers.
                calls.push(tokio::spawn(async move {
                    let a = MockProvider::new([
                        MockResponse::delayed(Duration::from_millis(1), Error::timeout("a")),
                        MockResponse::delayed(Duration::from_millis(1), Error::timeout("a")),
                    ])
                    .with_provider_name("a");
                    let stack = RetryingProvider::new(
                        FallbackProvider::new(
                            RetryingProvider::new(a, fast().max_attempts(2)),
                            RetryingProvider::new(flaky("b"), fast()),
                        ),
                        fast(),
                    );
                    stack.chat(&request(&format!("mt-chain-{call}"))).await
                }));
            }
            for call in calls {
                call.await.expect("no panic").expect("a response");
            }
        });

        for call in 0..16 {
            for model in [format!("mt-tries-{call}"), format!("mt-nested-{call}")] {
                let model_events = for_model(&events, &model);
                let hints = model_events
                    .iter()
                    .filter(|e| e.target == "allama::warning::unsupported")
                    .count();
                assert_eq!(hints, 1, "{model}: {model_events:?}");
            }
            // `b`'s hint once, and the failover once.
            let chain = for_model(&events, &format!("mt-chain-{call}"));
            let mut targets: Vec<_> = chain.iter().map(|e| e.target.as_str()).collect();
            targets.sort_unstable();
            assert_eq!(
                targets,
                ["allama::warning::fallback", "allama::warning::unsupported"],
                "{chain:?}"
            );
        }
        // Each model's first warning is its first `WARN`.
        assert_eq!(levels(&for_model(&events, "mt-tries-0")), (1, 0));
    }
}
