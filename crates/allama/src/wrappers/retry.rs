use std::collections::hash_map::RandomState;
use std::fmt;
use std::future::Future;
use std::hash::{BuildHasher, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use futures_timer::Delay;
// `std::time::SystemTime::now` panics on `wasm32-unknown-unknown`.
use web_time::{SystemTime, UNIX_EPOCH};

mod budget;

pub use budget::{RetryBudget, RetryPermit};

use super::Classifier;
use super::call_scope::{self, CallFact, GiveUpReason};
use crate::chat::stream::until_committed;
use crate::warning::AttemptScope;
use crate::{
    ChatProvider, ChatRequest, ChatResponse, ChatStream, DecisionProvider, DecisionRequest,
    DecisionResponse, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, Error,
    ExtractionProvider, ExtractionRequest, ExtractionResponse, ProviderIdentity, Result,
};

/// Retries transient errors against the same provider (chat, embeddings, and
/// decisions).
///
/// Unary calls are retried until they succeed or the [`RetryPolicy`] gives
/// up. Streams are retried while nothing has reached the caller: an error
/// before the first model-output event (see
/// [`StreamEvent::is_output`](crate::StreamEvent::is_output)) triggers a
/// retry; errors after that point are passed through, because output has
/// already been committed. The same policy, including the `Retry-After`
/// rule, applies to both.
///
/// Each attempt, backoff, server retry directive that decided, budget
/// denial, and give-up is reported to the call scope (see [`call_scope`]),
/// which a `TracingProvider` around the wrapper summarizes on its span.
/// Each retry is also an `allama.retry` event at `INFO`, target
/// `allama::retry` (features `warning-events` or `tracing`), and a retry the
/// budget denies an `allama.retry.budget_denied` event at `WARN`.
///
/// Each attempt runs in an attempt scope: the warning events of an
/// attempt that is retried are dropped, and those of the attempt the caller
/// gets go out, so a retried call logs each warning once. For a stream that
/// is the attempt that reached its first output.
///
/// The wrapper has no total deadline. Wrap it in a
/// [`TimeoutProvider`](crate::wrappers::TimeoutProvider) to bound a call
/// including every retry and backoff.
///
/// # Retry budget
///
/// Each wrapper owns a [`RetryBudget`] by default, with the AWS SDK
/// defaults, so retries across all the calls made through it stay bounded
/// while a provider is down. This matches the AWS SDKs, where each client
/// has its own retry quota. The budget is created by [`new`](Self::new),
/// from the policy's budget setting ([`RetryPolicy::make_budget`]):
///
/// - clones of the wrapper share its budget, as clones of an AWS client
///   share its token bucket;
/// - a new wrapper gets a budget of its own, also around a clone of the
///   same inner provider;
/// - to share one budget across wrappers (both sides of a fallback, say),
///   pass it explicitly with [`RetryPolicy::budget`];
/// - [`RetryPolicy::no_budget`] opts out, and [`budget`](Self::budget)
///   reads the budget, for example its [`available`](RetryBudget::available)
///   tokens.
///
/// Two caveats:
///
/// - **A wrapper built per request has no protection.** It gets a fresh,
///   full budget each time, so nothing carries over between calls. Build
///   the wrapper once and share it (or share a budget explicitly).
/// - **Refill is slow at low volume.** A drained default budget regains one
///   transient retry per 14 calls that succeed on their first attempt, so a
///   service with little traffic retries less for a while after an outage.
///
/// ```
/// # #[cfg(feature = "mock")] {
/// use allama::mock::MockProvider;
/// use allama::wrappers::{RetryBudget, RetryPolicy, RetryingProvider};
///
/// let wrapper = RetryingProvider::new(MockProvider::with_text("hi"), RetryPolicy::default());
/// let budget = wrapper.budget().expect("a budget by default");
/// assert_eq!(budget.available(), RetryBudget::DEFAULT_CAPACITY);
///
/// // A clone draws on the same tokens.
/// let clone = wrapper.clone();
/// assert_eq!(clone.budget().unwrap().available(), budget.available());
/// # }
/// ```
#[derive(Clone)]
pub struct RetryingProvider<T> {
    inner: T,
    policy: RetryPolicy,
    /// What this wrapper draws on, from the policy's budget setting.
    budget: Option<RetryBudget>,
}

impl<T> fmt::Debug for RetryingProvider<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RetryingProvider")
            .field("policy", &self.policy)
            .field("budget", &self.budget)
            .finish_non_exhaustive()
    }
}

impl<T> RetryingProvider<T> {
    /// Wrap `inner` with `policy`, and the budget the policy's budget
    /// setting gives: a new one of its own by default
    /// ([`RetryPolicy::budget`] shares one). `RetryPolicy::default()` is
    /// three attempts with jittered exponential backoff.
    #[must_use]
    pub fn new(inner: T, policy: RetryPolicy) -> Self {
        Self {
            inner,
            budget: policy.make_budget(),
            policy,
        }
    }

    /// Borrow the policy.
    #[must_use]
    pub fn policy(&self) -> &RetryPolicy {
        &self.policy
    }

    /// The budget this wrapper draws on; `None` when the policy opted out
    /// with [`RetryPolicy::no_budget`].
    #[must_use]
    pub fn budget(&self) -> Option<&RetryBudget> {
        self.budget.as_ref()
    }

    /// Borrow the wrapped provider.
    #[must_use]
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Unwrap the provider.
    #[must_use]
    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<T> ChatProvider for RetryingProvider<T>
where
    T: ChatProvider,
{
    type Stream = ChatStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        self.retry(|| self.inner.chat(request)).await
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        let mut attempts = Attempts::first();
        loop {
            // An attempt lasts until the stream's commit point, so the
            // stream is polled in the attempt's scope too.
            let scope = AttemptScope::new();
            let attempt = scope.instrument(call_scope::attempt(async {
                let stream = self.inner.chat_stream(request).await?;
                until_committed(Box::pin(scope.instrument_stream(stream))).await
            }));
            match attempt.await {
                Ok(prefixed) => {
                    scope.commit();
                    return Ok(self.succeeded(attempts, Box::pin(prefixed)));
                }
                Err(error) => match self.next_delay(error, &mut attempts) {
                    Ok(delay) => {
                        scope.discard();
                        wait(delay).await;
                    }
                    Err(error) => {
                        scope.commit();
                        return Err(error);
                    }
                },
            }
        }
    }

    super::forward_chat_questions!(inner);
}

impl<T: ProviderIdentity> RetryingProvider<T> {
    /// Run `call` until it succeeds or the policy gives up, each attempt in
    /// a call-scope attempt and an [`AttemptScope`]: the warning events of
    /// an attempt that is retried are dropped, and those of the attempt the
    /// caller gets (its response, or the last error) go out.
    async fn retry<R, F>(&self, mut call: impl FnMut() -> F) -> Result<R>
    where
        F: Future<Output = Result<R>>,
    {
        let mut attempts = Attempts::first();
        loop {
            let scope = AttemptScope::new();
            match scope.instrument(call_scope::attempt(call())).await {
                Ok(response) => {
                    scope.commit();
                    return Ok(self.succeeded(attempts, response));
                }
                Err(error) => match self.next_delay(error, &mut attempts) {
                    Ok(delay) => {
                        scope.discard();
                        wait(delay).await;
                    }
                    Err(error) => {
                        scope.commit();
                        return Err(error);
                    }
                },
            }
        }
    }

    /// How long to wait before the next attempt, or the error unchanged when
    /// the policy gives up or its budget denies the retry. Each decision is
    /// reported to the call scope (see [`call_scope`]), and a retry as an
    /// `allama.retry` event.
    fn next_delay(&self, error: Error, attempts: &mut Attempts) -> Result<Duration> {
        let attempt = attempts.number;
        if let Some(directive) = self.policy.applied_server_directive(&error, attempt) {
            call_scope::record(CallFact::server_directive(directive));
        }
        let delay = match self.policy.retry_decision(&error, attempt) {
            Ok(delay) => delay,
            Err(reason) => {
                call_scope::record(CallFact::give_up(reason));
                return Err(error);
            }
        };
        if let Some(budget) = &self.budget {
            let Some(permit) = budget.try_acquire(&error) else {
                self.budget_denied(&error, attempt);
                return Err(error);
            };
            attempts.permit = Some(permit);
        }
        attempts.number += 1;
        call_scope::record(CallFact::delay(delay));
        // OpenTelemetry integers are `i64`; `u64` values export as strings.
        #[cfg(any(feature = "tracing", feature = "warning-events"))]
        tracing::event!(
            target: "allama::retry",
            tracing::Level::INFO,
            "allama.attempt.provider" = super::semconv_provider_name(self.inner.provider_name()),
            "error.type" = error.kind_name(),
            "allama.retry.attempt" = i64::try_from(attempt).unwrap_or(i64::MAX),
            "allama.retry.max_attempts" =
                i64::try_from(self.policy.max_attempts.max(1)).unwrap_or(i64::MAX),
            "allama.retry.delay_ms" = i64::try_from(delay.as_millis()).unwrap_or(i64::MAX),
            "allama.retry"
        );
        Ok(delay)
    }

    /// A call succeeded after `attempts`: refund the budget per its rules.
    fn succeeded<R>(&self, attempts: Attempts, response: R) -> R {
        if let Some(budget) = &self.budget {
            budget.record_success(attempts.permit);
        }
        response
    }

    /// Report that the budget denied the retry after attempt number
    /// `attempt` failed with `error`: to the call scope, and as an
    /// `allama.retry.budget_denied` event in the current span.
    fn budget_denied(&self, error: &Error, attempt: usize) {
        call_scope::record(CallFact::budget_denied());
        call_scope::record(CallFact::give_up(GiveUpReason::Budget));
        #[cfg(any(feature = "tracing", feature = "warning-events"))]
        tracing::event!(
            target: "allama::retry",
            tracing::Level::WARN,
            "allama.attempt.provider" = super::semconv_provider_name(self.inner.provider_name()),
            "error.type" = error.kind_name(),
            "allama.retry.attempt" = i64::try_from(attempt).unwrap_or(i64::MAX),
            "allama.retry.budget_denied"
        );
        #[cfg(not(any(feature = "tracing", feature = "warning-events")))]
        let _ = (error, attempt);
    }
}

impl<T: EmbeddingProvider> EmbeddingProvider for RetryingProvider<T> {
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        self.retry(|| self.inner.embed(request)).await
    }

    super::forward_embedding_questions!(inner);
}

impl<T: DecisionProvider> DecisionProvider for RetryingProvider<T> {
    async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse> {
        self.retry(|| self.inner.decide(request)).await
    }

    super::forward_decision_questions!(inner);
}

impl<T: ExtractionProvider> ExtractionProvider for RetryingProvider<T> {
    async fn extract_values(&self, request: &ExtractionRequest) -> Result<ExtractionResponse> {
        self.retry(|| self.inner.extract_values(request)).await
    }

    super::forward_extraction_questions!(inner);
}

impl<T: ProviderIdentity> ProviderIdentity for RetryingProvider<T> {
    super::forward_provider_identity!(inner);
}

/// When and how long [`RetryingProvider`] waits before trying again.
///
/// An error is retried when the classifier accepts it and fewer than
/// `max_attempts` attempts have been made. The default classifier follows
/// the server's retry directive ([`Error::server_retry_directive`], from
/// `x-should-retry`) when there is one, in both directions: `true` retries
/// even a `409`, `false` stops even a `429` with a retry hint. Without a
/// directive it asks [`Error::is_retryable`]. The directive only replaces
/// that classification: the attempt limit, the retry hint rule below, and
/// the rule against retrying a stream after its first output still hold, and
/// fallback is unaffected: a rate limit the policy did not retry is still
/// fallback-eligible. A classifier set with [`retry_if`](Self::retry_if)
/// replaces the default and is not overridden by the directive.
///
/// The wait before the retry is chosen as follows:
///
/// - **The provider sent a positive retry hint** ([`Error::retry_after`],
///   from `Retry-After` and similar headers) and `respect_retry_after` is on:
///   - a hint up to `max_retry_after` (the hint ceiling, 120 s by default)
///     is a floor: the wrapper waits the longer of the hint and the
///     exponential backoff below, plus, with `jitter` on, a random extra of
///     up to `base_delay`, clamped so it does not take the wait past
///     `max_retry_after`. It never waits less than the hint. The extra
///     spreads out clients that all got the same hint when a rate limit
///     resets;
///   - a hint longer than `max_retry_after` makes the policy **give up at
///     once**. The error is returned unchanged, `retry_after` included, so a
///     [`FallbackProvider`](crate::wrappers::FallbackProvider) around the
///     wrapper (rate limits are fallback-eligible) or the caller decides what
///     to do. Waiting less than the provider asked for would only spend an
///     attempt on a request the provider already said it would reject.
/// - **Otherwise**, exponential backoff: `base_delay` doubled per retry,
///   capped at `max_delay`. With `jitter` on (the default) the wait is drawn
///   with *equal jitter*: uniformly between half the backoff and the full
///   backoff. Keeping a floor of half the backoff spreads out clients that
///   failed together without spending the few attempts on near-immediate
///   retries, as *full jitter* (anywhere from zero) can. A zero hint
///   (`Retry-After: 0`, or an HTTP date already past, which clock skew can
///   produce) counts as no hint and gets this backoff, so it never makes the
///   wrapper retry at once.
///
/// `max_delay` caps only the policy's own backoff; how long a provider may
/// make the wrapper wait is `max_retry_after`, a separate bound as in
/// openai-python. The default ceiling, 120 s, is openai-python's
/// `MAX_RETRY_AFTER_DELAY` (since v2.52.0) and suits batch work that would
/// rather wait than fail. An interactive server, which has a user waiting
/// and usually a fallback, sets a few seconds, so a long rate limit reaches
/// the fallback or the client (as `429` with `Retry-After`) at once:
///
/// ```
/// use std::time::Duration;
///
/// use allama::wrappers::RetryPolicy;
///
/// let interactive = RetryPolicy::new().max_retry_after(Duration::from_secs(5));
/// assert_eq!(interactive.max_retry_after, Duration::from_secs(5));
/// ```
///
/// A retry the rules above allow also needs tokens from a [`RetryBudget`];
/// when it runs dry, the error is returned unchanged. Not even a server
/// retry directive overrides a denial. By default each wrapper built with
/// the policy owns a budget; [`budget`](Self::budget) shares one across
/// wrappers explicitly, and
/// [`no_budget`](Self::no_budget) turns budgets off, so nothing limits
/// retries across calls. Clones of a policy share a budget only when it was
/// set explicitly. There is no total deadline; wrap the
/// [`RetryingProvider`] in a
/// [`TimeoutProvider`](crate::wrappers::TimeoutProvider) for one.
///
/// ```
/// use std::time::Duration;
///
/// use allama::Error;
/// use allama::wrappers::RetryPolicy;
///
/// let policy = RetryPolicy::new()
///     .max_attempts(5)
///     .base_delay(Duration::from_millis(100))
///     .max_delay(Duration::from_secs(10))
///     .retry_if(|error: &Error| error.is_retryable() || error.is_auth());
/// assert_eq!(policy.max_attempts, 5);
/// ```
#[derive(Clone)]
#[non_exhaustive]
pub struct RetryPolicy {
    /// Maximum attempts including the first. `0` is treated as `1`.
    pub max_attempts: usize,
    /// Backoff before the first retry; it doubles with each further retry.
    pub base_delay: Duration,
    /// Upper bound on the exponential backoff. It does not limit a wait the
    /// provider asked for; [`max_retry_after`](Self::max_retry_after) does.
    pub max_delay: Duration,
    /// The hint ceiling: the longest retry hint the policy waits out. A
    /// longer hint makes it give up instead of waiting.
    pub max_retry_after: Duration,
    /// Honour provider retry hints ([`Error::retry_after`]) as the shortest
    /// wait, and give up on hints longer than `max_retry_after`.
    pub respect_retry_after: bool,
    /// Randomize the wait: equal jitter on exponential backoff, and up to
    /// `base_delay` added to a retry hint (never beyond `max_retry_after`).
    pub jitter: bool,
    /// Which retry budget a wrapper built with the policy draws on: its own
    /// (the default), a shared one, or none. Set by [`budget`](Self::budget)
    /// and [`no_budget`](Self::no_budget), read by
    /// [`make_budget`](Self::make_budget).
    budget: BudgetSetting,
    /// `None` is the default classifier (see [`RetryPolicy::should_retry`]).
    retry_if: Option<Classifier>,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(250),
            max_delay: Duration::from_secs(5),
            max_retry_after: Duration::from_secs(120),
            respect_retry_after: true,
            jitter: true,
            budget: BudgetSetting::PerWrapper,
            retry_if: None,
        }
    }
}

impl fmt::Debug for RetryPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RetryPolicy")
            .field("max_attempts", &self.max_attempts)
            .field("base_delay", &self.base_delay)
            .field("max_delay", &self.max_delay)
            .field("max_retry_after", &self.max_retry_after)
            .field("respect_retry_after", &self.respect_retry_after)
            .field("jitter", &self.jitter)
            .field("budget", &self.budget)
            .finish_non_exhaustive()
    }
}

impl RetryPolicy {
    /// The default policy: 3 attempts, 250 ms base delay, backoff capped at
    /// 5 s, retry hints up to 120 s honoured, jitter on, a budget per
    /// wrapper, and the default classifier: the server's retry directive,
    /// else [`Error::is_retryable`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum attempts, including the first. `0` is treated as
    /// `1`.
    #[must_use]
    pub fn max_attempts(mut self, max_attempts: usize) -> Self {
        self.max_attempts = max_attempts;
        self
    }

    /// Set the backoff before the first retry.
    #[must_use]
    pub fn base_delay(mut self, delay: Duration) -> Self {
        self.base_delay = delay;
        self
    }

    /// Cap the exponential backoff. Retry hints are bounded by
    /// [`max_retry_after`](Self::max_retry_after) instead.
    #[must_use]
    pub fn max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Set the hint ceiling: the longest retry hint the policy waits out.
    /// Longer hints make it give up and return the error, hint included.
    ///
    /// The default, 120 s, suits batch callers. Interactive servers set a
    /// few seconds, so a long rate limit fails over or reaches the client
    /// at once instead of holding the request.
    #[must_use]
    pub fn max_retry_after(mut self, ceiling: Duration) -> Self {
        self.max_retry_after = ceiling;
        self
    }

    /// Honour (`true`, the default) or ignore provider retry hints.
    #[must_use]
    pub fn respect_retry_after(mut self, respect: bool) -> Self {
        self.respect_retry_after = respect;
        self
    }

    /// Turn jitter on (the default) or off. Off makes delays deterministic,
    /// which tests rely on: the exponential backoff, or the longer of a
    /// retry hint and that backoff.
    ///
    /// Jitter spreads out clients that failed together, but it is drawn from
    /// a hash of `RandomState`, the wall clock, and a call counter, not from
    /// an entropy source. On Cloudflare Workers all three may repeat across
    /// isolates: `wasm32-unknown-unknown` gives `RandomState` no entropy,
    /// the clock does not advance while code runs, and isolates start fresh.
    /// Isolates that fail together may then retry together. This is
    /// unverified. If it matters, turn jitter off and randomize the delays
    /// from [`retry_delay`](Self::retry_delay) yourself, from a real entropy
    /// source (on Workers, `crypto.getRandomValues`), in a custom retry
    /// wrapper, which can still own a [`RetryBudget`] from
    /// [`make_budget`](Self::make_budget) ([`RetryBudget::try_acquire`],
    /// [`RetryBudget::record_success`]).
    #[must_use]
    pub fn jitter(mut self, jitter: bool) -> Self {
        self.jitter = jitter;
        self
    }

    /// Draw every retry from `budget` (see [`RetryBudget`]), shared with
    /// every wrapper built with this policy or a clone of it, and with every
    /// policy given a clone of `budget`.
    #[must_use]
    pub fn budget(mut self, budget: RetryBudget) -> Self {
        self.budget = BudgetSetting::Shared(budget);
        self
    }

    /// Use no retry budget: nothing limits retries across calls beyond
    /// `max_attempts`.
    #[must_use]
    pub fn no_budget(mut self) -> Self {
        self.budget = BudgetSetting::Off;
        self
    }

    /// The budget a wrapper built with this policy draws on: a new budget
    /// with the AWS defaults (the default), the one shared with
    /// [`budget`](Self::budget), or `None` after
    /// [`no_budget`](Self::no_budget).
    ///
    /// [`RetryingProvider`] calls it once when it is built; a custom retry
    /// wrapper calls it the same way, so it owns a default budget too, and
    /// keeps the result for its lifetime (calling it per request would give
    /// every call a fresh, full budget).
    #[must_use]
    pub fn make_budget(&self) -> Option<RetryBudget> {
        match &self.budget {
            BudgetSetting::PerWrapper => Some(RetryBudget::new()),
            BudgetSetting::Shared(budget) => Some(budget.clone()),
            BudgetSetting::Off => None,
        }
    }

    /// Retry only the errors `classify` accepts.
    ///
    /// It replaces the default classifier, server retry directive included:
    /// the policy never overrides `classify`. Read the directive with
    /// [`Error::server_retry_directive`] to follow it.
    #[must_use]
    pub fn retry_if(mut self, classify: impl Fn(&Error) -> bool + Send + Sync + 'static) -> Self {
        self.retry_if = Some(Arc::new(classify));
        self
    }

    /// Whether the classifier accepts `error` for a retry.
    ///
    /// The default classifier follows the server's retry directive
    /// ([`Error::server_retry_directive`]) when there is one, and
    /// [`Error::is_retryable`] otherwise.
    #[must_use]
    pub fn should_retry(&self, error: &Error) -> bool {
        match &self.retry_if {
            Some(classify) => classify(error),
            None => error
                .server_retry_directive()
                .unwrap_or_else(|| error.is_retryable()),
        }
    }

    /// The server retry directive that decided whether to retry after
    /// attempt number `attempt` failed with `error`: `Some` only when the
    /// default classifier was consulted and the error carries one.
    fn applied_server_directive(&self, error: &Error, attempt: usize) -> Option<bool> {
        if self.retry_if.is_some() || attempt >= self.max_attempts.max(1) {
            return None;
        }
        error.server_retry_directive()
    }

    /// How long to wait before the next attempt, after attempt number
    /// `attempt` (counting from 1) failed with `error`; `None` when the
    /// policy gives up and the error should be returned as is.
    ///
    /// Applies every rule of the policy except the budget: the attempt
    /// limit, the classifier, retry hints, and backoff with jitter. With
    /// jitter on, repeated calls return different delays. Custom retry
    /// wrappers can use it to follow the same rules, and draw on the budget
    /// from [`make_budget`](Self::make_budget) with
    /// [`RetryBudget::try_acquire`] when it returns a delay.
    ///
    /// With a positive retry hint `hint <= max_retry_after`, the delay is at
    /// least `max(hint, backoff)`, exactly that with jitter off, and the
    /// jitter never takes it past `max_retry_after` (it only trims the
    /// extra, never the floor); a longer hint returns `None`. A zero hint is
    /// ignored. Without a hint the delay is at most `max_delay`.
    #[must_use]
    pub fn retry_delay(&self, error: &Error, attempt: usize) -> Option<Duration> {
        self.retry_decision(error, attempt).ok()
    }

    /// [`retry_delay`](Self::retry_delay), with the reason when the policy
    /// gives up.
    fn retry_decision(&self, error: &Error, attempt: usize) -> Result<Duration, GiveUpReason> {
        if attempt >= self.max_attempts.max(1) {
            return Err(GiveUpReason::MaxAttempts);
        }
        if !self.should_retry(error) {
            return Err(GiveUpReason::NotRetryable);
        }
        let backoff = exponential_backoff(self.base_delay, self.max_delay, attempt);
        let hint = error
            .retry_after()
            .filter(|hint| self.respect_retry_after && !hint.is_zero());
        if let Some(hint) = hint {
            if hint > self.max_retry_after {
                return Err(GiveUpReason::HintOverCeiling);
            }
            let floor = hint.max(backoff);
            return Ok(if self.jitter {
                hint_jitter(floor, self.base_delay, self.max_retry_after, random_unit())
            } else {
                floor
            });
        }
        Ok(if self.jitter {
            equal_jitter(backoff, random_unit())
        } else {
            backoff
        })
    }
}

/// Which [`RetryBudget`] a wrapper built with a [`RetryPolicy`] draws on
/// (see [`RetryPolicy::make_budget`]).
#[derive(Debug, Clone)]
enum BudgetSetting {
    /// Each wrapper creates its own budget with the AWS defaults, when it is
    /// built. Clones of the wrapper share it; clones of the policy do not.
    PerWrapper,
    /// Every wrapper draws on this budget, shared through its clones
    /// ([`RetryPolicy::budget`]).
    Shared(RetryBudget),
    /// No budget ([`RetryPolicy::no_budget`]).
    Off,
}

/// Wait `delay` before the next attempt.
async fn wait(delay: Duration) {
    if !delay.is_zero() {
        Delay::new(delay).await;
    }
}

/// How far a call has got: the number of the current attempt, from 1, and
/// the budget permit of the retry that started it.
struct Attempts {
    number: usize,
    permit: Option<RetryPermit>,
}

impl Attempts {
    fn first() -> Self {
        Self {
            number: 1,
            permit: None,
        }
    }
}

/// `base * 2^(retry_index - 1)`, capped at `max`.
fn exponential_backoff(base: Duration, max: Duration, retry_index: usize) -> Duration {
    let exponent = retry_index.saturating_sub(1).min(20) as u32;
    base.saturating_mul(1 << exponent).min(max)
}

/// A delay drawn from `[backoff / 2, backoff]`, placed by `unit` in `[0, 1)`.
fn equal_jitter(backoff: Duration, unit: f64) -> Duration {
    let half = backoff / 2;
    half + (backoff - half).mul_f64(unit)
}

/// `floor` plus up to `spread`, placed by `unit` in `[0, 1)`, capped at `max`.
///
/// The cap only trims the extra, never the floor: a `floor` above `max` (a
/// backoff longer than the hint ceiling) is returned as is.
fn hint_jitter(floor: Duration, spread: Duration, max: Duration, unit: f64) -> Duration {
    floor
        .saturating_add(spread.mul_f64(unit))
        .min(max)
        .max(floor)
}

/// A number in `[0, 1)` that differs between calls, processes, and
/// machines.
///
/// Jitter needs spread, not cryptographic randomness, and must work on
/// `wasm32-unknown-unknown` without `getrandom`. `RandomState` is seeded per
/// process where the platform has entropy (and from addresses where it does
/// not); the wall clock and a call counter vary it further.
fn random_unit() -> f64 {
    static CALLS: AtomicU64 = AtomicU64::new(0);
    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u64(CALLS.fetch_add(1, Ordering::Relaxed));
    if let Ok(now) = SystemTime::now().duration_since(UNIX_EPOCH) {
        hasher.write_u128(now.as_nanos());
    }
    // The top 53 bits fill an f64 mantissa exactly.
    (hasher.finish() >> 11) as f64 / (1u64 << 53) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::{ChatResponseBuilder, MockProvider, MockStreamEvent, MockStreamingProvider};
    use crate::{ChatStreamExt, StreamEvent};

    fn fast() -> RetryPolicy {
        RetryPolicy::new()
            .base_delay(Duration::ZERO)
            .max_delay(Duration::ZERO)
    }

    fn request() -> ChatRequest {
        ChatRequest::new("m").user("hi")
    }

    fn rate_limited(hint: Duration) -> Error {
        Error::rate_limited("slow down").with_retry_after(hint)
    }

    #[tokio::test]
    async fn retries_until_success() {
        let mock = MockProvider::build(|b| b.error(Error::timeout("slow")).text("ok"));
        let wrapper = RetryingProvider::new(mock.clone(), fast());
        assert_eq!(
            wrapper.chat(&request()).await.unwrap().text().as_deref(),
            Some("ok")
        );
        assert_eq!(mock.call_count(), 2);
    }

    #[tokio::test]
    async fn does_not_retry_permanent_errors() {
        for error in [Error::auth("bad"), Error::quota_exhausted("no credit")] {
            let mock = MockProvider::with_error(error);
            let wrapper = RetryingProvider::new(mock.clone(), fast());
            assert!(wrapper.chat(&request()).await.is_err());
            assert_eq!(mock.call_count(), 1);
        }
    }

    /// A `501` is answered the same way every time: one attempt, and a
    /// fallback provider still fails over on it, as on any provider error.
    #[tokio::test]
    async fn a_501_is_tried_once_and_fails_over() {
        let not_implemented = || Error::provider("HTTP 501").with_status(501);
        let mock = MockProvider::build(|b| b.error(not_implemented()).text("ok"));
        let wrapper = RetryingProvider::new(mock.clone(), fast());
        let err = wrapper.chat(&request()).await.unwrap_err();
        assert_eq!(err.status(), Some(501));
        assert_eq!(mock.call_count(), 1);

        let backup = MockProvider::with_text("backup");
        let primary = MockProvider::with_error(not_implemented());
        let wrapper = crate::wrappers::FallbackProvider::new(
            RetryingProvider::new(primary.clone(), fast()),
            backup.clone(),
        );
        let response = wrapper.chat(&request()).await.unwrap();
        assert_eq!(response.text().as_deref(), Some("backup"));
        assert_eq!(primary.call_count(), 1);
        assert_eq!(backup.call_count(), 1);
    }

    #[tokio::test]
    async fn stops_after_max_attempts() {
        let mock = MockProvider::new([Error::timeout("one"), Error::timeout("two")]);
        let wrapper = RetryingProvider::new(mock.clone(), fast().max_attempts(2));
        let err = wrapper.chat(&request()).await.unwrap_err();
        assert_eq!(err.message(), "two");
        assert_eq!(mock.call_count(), 2);
    }

    #[tokio::test]
    async fn custom_classifier_decides_what_is_retried() {
        let mock = MockProvider::build(|b| b.error(Error::auth("rotating")).text("ok"));
        let wrapper = RetryingProvider::new(
            mock.clone(),
            fast().retry_if(|error: &Error| error.is_auth()),
        );
        assert!(wrapper.chat(&request()).await.is_ok());
        assert_eq!(mock.call_count(), 2);

        let mock = MockProvider::new([Error::timeout("slow")]);
        let wrapper = RetryingProvider::new(mock.clone(), fast().retry_if(|_| false));
        assert!(wrapper.chat(&request()).await.is_err());
        assert_eq!(mock.call_count(), 1);
    }

    #[test]
    fn retry_hint_without_jitter_waits_the_longer_of_hint_and_backoff() {
        let policy = RetryPolicy::new()
            .max_attempts(10)
            .base_delay(Duration::from_millis(100))
            .max_delay(Duration::from_secs(1))
            .jitter(false);
        let hint = Duration::from_millis(300);
        // Backoff 100 ms, 200 ms: the hint is longer. 400 ms, 800 ms: backoff.
        let delays: Vec<_> = (1..=4)
            .map(|attempt| policy.retry_delay(&rate_limited(hint), attempt).unwrap())
            .collect();
        assert_eq!(delays, [300, 300, 400, 800].map(Duration::from_millis));
        assert_eq!(
            policy.retry_delay(&rate_limited(Duration::from_secs(1)), 1),
            Some(Duration::from_secs(1))
        );
    }

    #[test]
    fn zero_retry_hint_is_treated_as_no_hint() {
        let policy = RetryPolicy::new()
            .max_attempts(10)
            .base_delay(Duration::from_millis(100))
            .max_delay(Duration::from_secs(1))
            .jitter(false);
        for attempt in 1..=3 {
            assert_eq!(
                policy.retry_delay(&rate_limited(Duration::ZERO), attempt),
                policy.retry_delay(&Error::rate_limited("no hint"), attempt),
            );
        }
        assert_eq!(
            policy.retry_delay(&rate_limited(Duration::ZERO), 2),
            Some(Duration::from_millis(200))
        );

        // With jitter on it gets the jittered backoff, never zero.
        let jittered = policy.clone().jitter(true);
        for _ in 0..64 {
            let delay = jittered
                .retry_delay(&rate_limited(Duration::ZERO), 2)
                .unwrap();
            assert!(
                (Duration::from_millis(100)..=Duration::from_millis(200)).contains(&delay),
                "{delay:?} outside equal-jitter range"
            );
        }
    }

    #[test]
    fn past_retry_after_date_is_treated_as_no_hint() {
        let policy = RetryPolicy::new()
            .base_delay(Duration::from_millis(100))
            .max_delay(Duration::from_secs(1))
            .jitter(false);
        // What `allama_http::parse_retry_after` gives for a past date.
        let hint = Some(Duration::ZERO);
        let error = Error::rate_limited("slow down").with_retry_after_opt(hint);
        assert_eq!(
            policy.retry_delay(&error, 1),
            Some(Duration::from_millis(100))
        );
    }

    /// Property: with jitter on, a positive hint `h <= max_retry_after`
    /// gives a delay in `[max(h, backoff), max_retry_after]`, over hints
    /// below and above the backoff cap, attempts, and the policy's own
    /// random draws.
    #[test]
    fn jittered_retry_hint_stays_between_floor_and_the_hint_ceiling() {
        let max_delay = Duration::from_millis(400);
        let ceiling = Duration::from_millis(1_000);
        let policy = RetryPolicy::new()
            .max_attempts(10)
            .base_delay(Duration::from_millis(100))
            .max_delay(max_delay)
            .max_retry_after(ceiling);
        let mut spread = false;
        for hint_ms in [
            1, 50, 99, 100, 250, 399, 400, 401, 640, 899, 900, 950, 999, 1_000,
        ] {
            let hint = Duration::from_millis(hint_ms);
            for attempt in 1..=6 {
                let backoff = exponential_backoff(policy.base_delay, max_delay, attempt);
                let floor = hint.max(backoff);
                let delays: Vec<_> = (0..32)
                    .map(|_| policy.retry_delay(&rate_limited(hint), attempt).unwrap())
                    .collect();
                for delay in &delays {
                    assert!(
                        (floor..=ceiling).contains(delay),
                        "hint {hint:?}, attempt {attempt}: {delay:?} outside [{floor:?}, {ceiling:?}]"
                    );
                    assert!(*delay >= hint);
                }
                spread |= delays.iter().any(|delay| *delay != delays[0]);
            }
        }
        assert!(spread, "jitter produced identical delays");
    }

    #[test]
    fn hint_jitter_adds_up_to_spread_within_max() {
        let base = Duration::from_millis(100);
        let max = Duration::from_millis(1_000);
        let floor = Duration::from_millis(700);
        assert_eq!(hint_jitter(floor, base, max, 0.0), floor);
        assert_eq!(
            hint_jitter(floor, base, max, 0.5),
            Duration::from_millis(750)
        );
        // Near the ceiling the extra is trimmed, never the floor.
        let floor = Duration::from_millis(950);
        assert_eq!(hint_jitter(floor, base, max, 0.999_999), max);
        assert_eq!(hint_jitter(max, base, max, 0.7), max);
        // A backoff floor above the ceiling is kept.
        let above = Duration::from_millis(1_200);
        assert_eq!(hint_jitter(above, base, max, 0.7), above);
        for unit in (0..1_000).map(|step| f64::from(step) / 1_000.0) {
            let delay = hint_jitter(floor, base, max, unit);
            assert!((floor..=max).contains(&delay), "{delay:?} at {unit}");
        }
    }

    #[test]
    fn retry_hint_beyond_the_hint_ceiling_gives_up() {
        let policy = RetryPolicy::new()
            .base_delay(Duration::from_millis(100))
            .max_retry_after(Duration::from_secs(1));
        assert_eq!(
            policy.retry_delay(&rate_limited(Duration::from_secs(5)), 1),
            None
        );
        assert_eq!(
            policy.retry_decision(&rate_limited(Duration::from_secs(5)), 1),
            Err(GiveUpReason::HintOverCeiling)
        );
        // The default ceiling is 120 s.
        let default = RetryPolicy::new().jitter(false);
        assert_eq!(default.max_retry_after, Duration::from_secs(120));
        assert_eq!(
            default.retry_delay(&rate_limited(Duration::from_secs(120)), 1),
            Some(Duration::from_secs(120))
        );
        assert_eq!(
            default.retry_delay(&rate_limited(Duration::from_secs(121)), 1),
            None
        );
    }

    /// `max_delay` caps the backoff only: a hint above it, within the hint
    /// ceiling, is waited out.
    #[test]
    fn max_delay_caps_backoff_not_retry_hints() {
        let policy = RetryPolicy::new().max_attempts(20).jitter(false);
        assert_eq!(policy.max_delay, Duration::from_secs(5));
        assert_eq!(
            policy.retry_delay(&rate_limited(Duration::from_secs(30)), 1),
            Some(Duration::from_secs(30))
        );
        assert_eq!(
            policy.retry_delay(&Error::timeout("x"), 10),
            Some(Duration::from_secs(5))
        );
    }

    #[test]
    fn ignored_retry_hint_falls_back_to_backoff() {
        let policy = RetryPolicy::new()
            .base_delay(Duration::from_millis(100))
            .max_delay(Duration::from_secs(1))
            .respect_retry_after(false)
            .jitter(false);
        assert_eq!(
            policy.retry_delay(&rate_limited(Duration::from_secs(5)), 1),
            Some(Duration::from_millis(100))
        );
    }

    #[test]
    fn backoff_doubles_up_to_max_delay() {
        let policy = RetryPolicy::new()
            .max_attempts(10)
            .base_delay(Duration::from_millis(100))
            .max_delay(Duration::from_millis(500))
            .jitter(false);
        let timeout = Error::timeout("x");
        let delays: Vec<_> = (1..=5)
            .map(|attempt| policy.retry_delay(&timeout, attempt).unwrap())
            .collect();
        assert_eq!(delays, [100, 200, 400, 500, 500].map(Duration::from_millis));
    }

    #[test]
    fn retry_delay_respects_attempts_and_classifier() {
        let policy = fast().max_attempts(2);
        assert_eq!(
            policy.retry_delay(&Error::timeout("x"), 1),
            Some(Duration::ZERO)
        );
        assert_eq!(policy.retry_delay(&Error::timeout("x"), 2), None);
        assert_eq!(policy.retry_delay(&Error::auth("x"), 1), None);
        assert_eq!(
            fast().max_attempts(0).retry_delay(&Error::timeout("x"), 1),
            None
        );
    }

    #[test]
    fn jittered_backoff_stays_between_half_and_full() {
        let policy = RetryPolicy::new()
            .max_attempts(10)
            .base_delay(Duration::from_millis(100))
            .max_delay(Duration::from_secs(1));
        let timeout = Error::timeout("x");
        let delays: Vec<_> = (0..64)
            .map(|_| policy.retry_delay(&timeout, 3).unwrap())
            .collect();
        for delay in &delays {
            assert!(
                (Duration::from_millis(200)..=Duration::from_millis(400)).contains(delay),
                "{delay:?} outside equal-jitter range"
            );
        }
        assert!(
            delays.iter().any(|delay| *delay != delays[0]),
            "jitter produced identical delays"
        );
    }

    #[test]
    fn equal_jitter_spans_half_to_full_backoff() {
        let backoff = Duration::from_millis(400);
        assert_eq!(equal_jitter(backoff, 0.0), Duration::from_millis(200));
        assert_eq!(equal_jitter(backoff, 0.5), Duration::from_millis(300));
        assert!(equal_jitter(backoff, 0.999_999) <= backoff);
        assert_eq!(equal_jitter(Duration::ZERO, 0.7), Duration::ZERO);
    }

    #[tokio::test]
    async fn long_retry_hint_returns_the_error_without_waiting() {
        let mock =
            MockProvider::build(|b| b.error(rate_limited(Duration::from_secs(30))).text("ok"));
        let wrapper = RetryingProvider::new(
            mock.clone(),
            RetryPolicy::new().max_retry_after(Duration::from_millis(50)),
        );
        let started = web_time::Instant::now();
        let error = wrapper.chat(&request()).await.unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(1));
        assert_eq!(mock.call_count(), 1);
        assert_eq!(error.kind_name(), "rate_limited");
        assert_eq!(error.retry_after(), Some(Duration::from_secs(30)));
    }

    #[tokio::test]
    async fn zero_retry_hint_still_backs_off() {
        let base = Duration::from_millis(40);
        let mock = MockProvider::build(|b| b.error(rate_limited(Duration::ZERO)).text("ok"));
        let wrapper = RetryingProvider::new(mock.clone(), RetryPolicy::new().base_delay(base));
        let started = web_time::Instant::now();
        assert!(wrapper.chat(&request()).await.is_ok());
        // Equal jitter waits at least half the backoff.
        assert!(started.elapsed() >= base / 2);
        assert_eq!(mock.call_count(), 2);
    }

    /// Under the default policy's shape (a hint ceiling far above the
    /// backoff cap) a hint longer than `max_delay` is waited out. Scaled
    /// down, because `futures-timer` does not follow tokio's paused clock.
    #[tokio::test]
    async fn a_hint_above_the_backoff_cap_is_waited_out() {
        let hint = Duration::from_millis(120);
        let mock = MockProvider::build(|b| b.error(rate_limited(hint)).text("ok"));
        let wrapper = RetryingProvider::new(
            mock.clone(),
            RetryPolicy::new()
                .base_delay(Duration::from_millis(5))
                .max_delay(Duration::from_millis(20))
                .max_retry_after(Duration::from_millis(500)),
        );
        let started = web_time::Instant::now();
        assert!(wrapper.chat(&request()).await.is_ok());
        assert!(started.elapsed() >= hint);
        assert_eq!(mock.call_count(), 2);
    }

    #[tokio::test]
    async fn short_retry_hint_is_waited_before_retrying() {
        let hint = Duration::from_millis(60);
        let mock = MockProvider::build(|b| b.error(rate_limited(hint)).text("ok"));
        let wrapper = RetryingProvider::new(mock.clone(), RetryPolicy::new());
        let started = web_time::Instant::now();
        assert!(wrapper.chat(&request()).await.is_ok());
        assert!(started.elapsed() >= hint);
        assert_eq!(mock.call_count(), 2);
    }

    #[tokio::test]
    async fn stream_gives_up_on_long_retry_hint() {
        let mock = MockStreamingProvider::build(|b| {
            b.stream([
                MockStreamEvent::Event(StreamEvent::ResponseStart {
                    id: None,
                    model: None,
                }),
                MockStreamEvent::Error(rate_limited(Duration::from_secs(30))),
            ])
            .response(ChatResponseBuilder::new().text("too early").build())
        });
        let wrapper = RetryingProvider::new(
            mock.clone(),
            RetryPolicy::new().max_retry_after(Duration::from_millis(50)),
        );
        let Err(error) = wrapper.chat_stream(&request()).await else {
            panic!("stream should give up before output");
        };
        assert_eq!(error.retry_after(), Some(Duration::from_secs(30)));
        assert_eq!(mock.call_count(), 1);
    }

    #[tokio::test]
    async fn stream_retries_before_commit_only() {
        let mock = MockStreamingProvider::build(|b| {
            b.stream([
                MockStreamEvent::Event(StreamEvent::ResponseStart {
                    id: None,
                    model: None,
                }),
                MockStreamEvent::Error(Error::overloaded("busy")),
            ])
            .response(ChatResponseBuilder::new().text("second").build())
        });
        let wrapper = RetryingProvider::new(mock.clone(), fast());
        let response = wrapper
            .chat_stream(&request())
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        assert_eq!(response.text().as_deref(), Some("second"));
        assert_eq!(mock.call_count(), 2);

        let committed = MockStreamingProvider::build(|b| {
            b.stream([
                MockStreamEvent::Event(StreamEvent::ResponseStart {
                    id: None,
                    model: None,
                }),
                MockStreamEvent::Event(StreamEvent::TextDelta {
                    index: 0,
                    text: "partial".into(),
                }),
                MockStreamEvent::Error(Error::overloaded("busy")),
            ])
            .response(ChatResponseBuilder::new().text("never").build())
        });
        let wrapper = RetryingProvider::new(committed.clone(), fast());
        let result = wrapper
            .chat_stream(&request())
            .await
            .unwrap()
            .collect_response()
            .await;
        assert!(result.is_err());
        assert_eq!(committed.call_count(), 1);
    }

    /// A streamed refusal is output: once the caller may have seen the
    /// model decline, a later error is not retried.
    #[tokio::test]
    async fn a_streamed_refusal_is_a_commit_point() {
        let refused = MockStreamingProvider::build(|b| {
            b.stream([
                MockStreamEvent::Event(StreamEvent::response_start(None, None)),
                MockStreamEvent::Event(StreamEvent::block_start(
                    0,
                    crate::chat::stream::StreamBlockType::Refusal,
                )),
                MockStreamEvent::Event(StreamEvent::refusal_delta(0, "I can't")),
                MockStreamEvent::Error(Error::overloaded("busy")),
            ])
            .response(ChatResponseBuilder::new().text("never").build())
        });
        let wrapper = RetryingProvider::new(refused.clone(), fast());
        let result = wrapper
            .chat_stream(&request())
            .await
            .unwrap()
            .collect_response()
            .await;
        assert!(result.is_err());
        assert_eq!(refused.call_count(), 1);
    }

    #[test]
    fn debug_lists_settings() {
        let debug = format!("{:?}", RetryPolicy::new().jitter(false));
        assert!(debug.contains("max_attempts: 3"), "{debug}");
        assert!(debug.contains("jitter: false"), "{debug}");
        assert!(debug.contains("max_retry_after: 120s"), "{debug}");
    }
}
