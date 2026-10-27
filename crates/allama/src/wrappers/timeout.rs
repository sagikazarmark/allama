use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use futures_core::Stream;
use futures_timer::Delay;
use futures_util::future::{Either, select};

use crate::{
    ChatProvider, ChatRequest, ChatResponse, DecisionProvider, DecisionRequest, DecisionResponse,
    EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, Error, ProviderIdentity, Result,
    StreamEvent,
};

/// Time bounds enforced by [`TimeoutProvider`]. Each bound is optional.
/// An unset [`stream_setup`](Self::stream_setup) takes the
/// [`request`](Self::request) bound, so one `request` bound covers a call
/// whether it is unary or streamed; any other unset bound leaves its phase
/// unbounded.
///
/// # The bounds
///
/// | Bound | Method | What it bounds |
/// |---|---|---|
/// | request | [`request`](Self::request) | a whole unary call (`chat`, `embed`, `decide`), and stream setup unless that has a bound of its own |
/// | setup | [`stream_setup`](Self::stream_setup) | opening a stream: the `chat_stream` call, which for an HTTP provider ends at the response headers, and through retry and fallback at the first output; unset, the request bound |
/// | idle | [`stream_idle`](Self::stream_idle) | each wait for the next stream event, the first one included |
/// | total | [`stream_total`](Self::stream_total) | a whole stream, from the `chat_stream` call to its last event |
///
/// Whichever stream bound expires first ends the call, with
/// [`Error::Timeout`]. Without a total bound, a stream that keeps producing
/// events is never cut off; with one, a long answer is cut off too.
///
/// A bound of [`Duration::MAX`] is no bound. It is how a policy bounds
/// unary calls but not stream setup:
/// `TimeoutPolicy::new().request(d).stream_setup(Duration::MAX)`.
///
/// These are portable bounds on decoded calls and events. The provider
/// builders apply transport timeouts of their own first, to catch dead
/// connections: `timeout` for a whole non-streaming call and
/// `stream_timeouts` for setup and idle.
///
/// # Attempt timeout vs deadline
///
/// Where the [`TimeoutProvider`] sits decides what the policy bounds:
///
/// - **An attempt timeout** sits inside the retries
///   (`RetryingProvider::new(TimeoutProvider::new(p, policy), RetryPolicy::default())`, or
///   `allama_providers::ProviderSettings::attempt_timeout`) and bounds one
///   attempt: a stalled attempt fails with [`Error::Timeout`] and is
///   retried. A `stream_total` there restarts with every attempt.
/// - **A deadline** sits outside retries and fallback
///   (`TimeoutProvider::new(FallbackProvider::new(..), policy)`) and bounds
///   the whole call, across every attempt, backoff and failover. Retry and
///   fallback hand a stream over only at its
///   [commit point](crate::chat::stream#the-commit-point), so `stream_setup` there
///   is the wait for the first output: an interactive server sets it (or
///   `request` for unary calls), never `stream_total`, which would cut off
///   a long answer after it started.
///
/// ```
/// use std::time::Duration;
///
/// use allama::wrappers::TimeoutPolicy;
///
/// // Catch stalls quickly, and cap runaway generations at five minutes.
/// let policy = TimeoutPolicy::new()
///     .stream_idle(Duration::from_secs(60))
///     .stream_total(Duration::from_secs(300));
/// assert_eq!(policy.stream_total, Some(Duration::from_secs(300)));
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct TimeoutPolicy {
    /// Bound on a whole unary call (`chat`, `embed`, `decide`), and on
    /// opening a stream while `stream_setup` is `None`.
    pub request: Option<Duration>,
    /// Bound on opening a stream: the `chat_stream` call itself, never a
    /// unary call. Inside the retries that ends at an HTTP provider's
    /// response headers, outside them at the first output (see
    /// [`stream_setup`](Self::stream_setup)). `None` takes the `request`
    /// bound.
    pub stream_setup: Option<Duration>,
    /// Bound on the wait for each stream item, the first one included.
    /// Only time spent waiting on the provider counts: the clock for an
    /// item starts when the consumer polls for it and it is not ready, not
    /// when the stream opened or the previous item was handed out.
    pub stream_idle: Option<Duration>,
    /// Bound on a whole stream, measured from the `chat_stream` call:
    /// setup, every event, and the waits between them. It also bounds a
    /// stream that keeps producing events.
    pub stream_total: Option<Duration>,
}

impl TimeoutPolicy {
    /// A policy with every bound unset.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Bound unary calls (`chat`, `embed` and `decide`), and stream setup
    /// too unless [`stream_setup`](Self::stream_setup) sets a bound of its
    /// own.
    ///
    /// One `request` bound is a deadline for a server that makes both kinds
    /// of call: an extraction or a tool loop that calls `chat`, and a reply
    /// it streams. A stream's setup never takes longer than the same call
    /// made unary, so the inherited bound is never tighter than the one
    /// the caller set.
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use allama::wrappers::TimeoutPolicy;
    ///
    /// // A unary call, and a stream's wait for its first output.
    /// let deadline = TimeoutPolicy::new().request(Duration::from_secs(60));
    /// # let _ = deadline;
    /// ```
    #[must_use]
    pub fn request(mut self, timeout: Duration) -> Self {
        self.request = Some(timeout);
        self
    }

    /// Bound stream setup: the `chat_stream` call, until it returns a
    /// stream. Unset, it takes the [`request`](Self::request) bound; set
    /// it for a different one, or to [`Duration::MAX`] for none.
    ///
    /// When `chat_stream` returns depends on what the [`TimeoutProvider`]
    /// wraps, so the same bound means different waits at different layers
    /// (see [attempt timeout vs deadline](Self#attempt-timeout-vs-deadline)):
    ///
    /// - **As an attempt timeout**, around an adapter and inside the
    ///   retries, `chat_stream` returns once the stream is open, which for
    ///   an HTTP provider is when the response headers arrive: it bounds
    ///   the wait for the headers of one attempt.
    /// - **As a deadline**, around [`RetryingProvider`](super::RetryingProvider)
    ///   or [`FallbackProvider`](super::FallbackProvider), `chat_stream`
    ///   returns only at the stream's
    ///   [commit point](crate::chat::stream#the-commit-point), its first
    ///   output: it bounds the wait for the first output, across every
    ///   attempt, backoff and failover.
    ///
    /// Neither bounds the answer once it has started; that is
    /// [`stream_idle`](Self::stream_idle) and
    /// [`stream_total`](Self::stream_total).
    #[must_use]
    pub fn stream_setup(mut self, timeout: Duration) -> Self {
        self.stream_setup = Some(timeout);
        self
    }

    /// Bound the gap before each stream item.
    #[must_use]
    pub fn stream_idle(mut self, timeout: Duration) -> Self {
        self.stream_idle = Some(timeout);
        self
    }

    /// Bound a whole stream, from the `chat_stream` call to its last event.
    ///
    /// When the bound expires the stream yields [`Error::Timeout`] and ends,
    /// even after output has started. Like any error after output, it is
    /// then passed through by [`RetryingProvider`](super::RetryingProvider)
    /// and [`FallbackProvider`](super::FallbackProvider), not retried.
    #[must_use]
    pub fn stream_total(mut self, timeout: Duration) -> Self {
        self.stream_total = Some(timeout);
        self
    }

    /// The bounds [`TimeoutProvider`] applies: an unset setup bound takes
    /// the request bound, and [`Duration::MAX`] is no bound.
    fn effective(self) -> Self {
        let bound = |limit: Option<Duration>| limit.filter(|limit| *limit != Duration::MAX);
        Self {
            request: bound(self.request),
            stream_setup: bound(self.stream_setup.or(self.request)),
            stream_idle: bound(self.stream_idle),
            stream_total: bound(self.stream_total),
        }
    }
}

/// Fails calls that exceed a [`TimeoutPolicy`] with [`Error::Timeout`].
///
/// This is a portable bound on top of whatever timeouts the provider's
/// transport applies. Expiry drops the inner future or stream (cancelling
/// the request) and surfaces as a retryable, fallback-eligible
/// [`Error::Timeout`], so it composes with the other wrappers:
///
/// - `RetryingProvider::new(TimeoutProvider::new(p, policy), RetryPolicy::default())` bounds each
///   attempt; a stream that stalls before its first output is retried.
/// - `TimeoutProvider::new(RetryingProvider::new(p, RetryPolicy::default()), policy)` bounds unary
///   calls and a stream's wait for its first output (`request`), or whole
///   streams (`stream_total`), including all retries and backoff.
///
/// Keep it inside `TracingProvider` (feature `tracing`). Outside it, as in
/// `Timeout(Tracing(p))`, an expiry drops the traced call, so its span
/// records a cancellation instead of `error.type = timeout`.
///
/// Timers come from `futures-timer`, so the wrapper works on any async
/// runtime; on `wasm32-unknown-unknown` (browsers, Workers) they use the
/// host's `setTimeout`, with no feature to enable.
#[derive(Clone)]
pub struct TimeoutProvider<T> {
    inner: T,
    policy: TimeoutPolicy,
}

impl<T> TimeoutProvider<T> {
    /// Wrap `inner` with `policy`.
    #[must_use]
    pub fn new(inner: T, policy: TimeoutPolicy) -> Self {
        Self { inner, policy }
    }

    /// Borrow the policy.
    #[must_use]
    pub fn policy(&self) -> &TimeoutPolicy {
        &self.policy
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

impl<T: ProviderIdentity> TimeoutProvider<T> {
    async fn bounded<O>(
        &self,
        what: &str,
        limit: Option<Duration>,
        future: impl Future<Output = Result<O>>,
    ) -> Result<O> {
        let Some(limit) = limit else {
            return future.await;
        };
        let future = std::pin::pin!(future);
        match select(future, Delay::new(limit)).await {
            Either::Left((result, _)) => result,
            Either::Right(((), _)) => Err(expired(self.inner.provider_name(), what, limit)),
        }
    }
}

/// What the total stream bound limits, in its error message.
const TOTAL: &str = "stream duration";

fn expired(provider: &str, what: &str, limit: Duration) -> Error {
    Error::timeout(format!("{what} exceeded {limit:?}"))
        .authored()
        .with_provider(provider)
}

impl<T> ChatProvider for TimeoutProvider<T>
where
    T: ChatProvider,
{
    type Stream = TimeoutStream<T::Stream>;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        self.bounded(
            "chat request",
            self.policy.effective().request,
            self.inner.chat(request),
        )
        .await
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
        let policy = self.policy.effective();
        // The total bound starts with the call, so it covers setup too.
        let Some(total) = policy.stream_total else {
            let stream = self
                .bounded(
                    "stream setup",
                    policy.stream_setup,
                    self.inner.chat_stream(request),
                )
                .await?;
            return Ok(TimeoutStream::new(
                stream,
                policy.stream_idle,
                None,
                self.inner.provider_name().to_owned(),
            ));
        };
        let mut deadline = Delay::new(total);
        let setup = std::pin::pin!(self.bounded(
            "stream setup",
            policy.stream_setup,
            self.inner.chat_stream(request),
        ));
        let stream = match select(setup, &mut deadline).await {
            Either::Left((result, _)) => result?,
            Either::Right(((), _)) => {
                return Err(expired(self.inner.provider_name(), TOTAL, total));
            }
        };
        Ok(TimeoutStream::new(
            stream,
            policy.stream_idle,
            Some((deadline, total)),
            self.inner.provider_name().to_owned(),
        ))
    }

    super::forward_chat_questions!(inner);
}

impl<T: EmbeddingProvider> EmbeddingProvider for TimeoutProvider<T> {
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        self.bounded(
            "embedding request",
            self.policy.effective().request,
            self.inner.embed(request),
        )
        .await
    }

    super::forward_embedding_questions!(inner);
}

impl<T: DecisionProvider> DecisionProvider for TimeoutProvider<T> {
    async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse> {
        self.bounded(
            "decision request",
            self.policy.effective().request,
            self.inner.decide(request),
        )
        .await
    }

    super::forward_decision_questions!(inner);
}

impl<T: ProviderIdentity> ProviderIdentity for TimeoutProvider<T> {
    super::forward_provider_identity!(inner);
}

impl<T> fmt::Debug for TimeoutProvider<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TimeoutProvider")
            .field("policy", &self.policy)
            .finish_non_exhaustive()
    }
}

/// A stream that fails with [`Error::Timeout`] when no item arrives within
/// the idle bound, or when the total bound expires. Returned by
/// [`TimeoutProvider::chat_stream`].
///
/// The idle bound measures how long the provider keeps the consumer
/// waiting: for each item, the first one included, the clock starts when
/// the consumer polls and the item is not ready yet. Time the consumer
/// spends before its first poll or between polls does not count.
///
/// The total bound runs from the `chat_stream` call and counts everything,
/// the consumer's time included; it is checked before each item, so it also
/// cuts off a stream whose items are always ready.
///
/// After the timeout error the stream ends; the inner stream is dropped.
pub struct TimeoutStream<S> {
    inner: Option<Pin<Box<S>>>,
    idle: Option<Duration>,
    timer: Option<Delay>,
    /// The idle timer is running: set on the first `Pending` of each wait
    /// (the first one included), cleared when an item is handed out.
    armed: bool,
    /// The total bound's timer, started by the `chat_stream` call, and the
    /// bound itself for the error message.
    deadline: Option<(Delay, Duration)>,
    provider: String,
}

impl<S> TimeoutStream<S> {
    fn new(
        inner: S,
        idle: Option<Duration>,
        deadline: Option<(Delay, Duration)>,
        provider: String,
    ) -> Self {
        Self {
            inner: Some(Box::pin(inner)),
            idle,
            timer: idle.map(Delay::new),
            // Disarmed until the consumer first waits: the time before its
            // first poll is its own.
            armed: false,
            deadline,
            provider,
        }
    }
}

impl<S> fmt::Debug for TimeoutStream<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TimeoutStream")
            .field("provider", &self.provider)
            .field("idle", &self.idle)
            .field("total", &self.deadline.as_ref().map(|(_, total)| *total))
            .field("ended", &self.inner.is_none())
            .finish_non_exhaustive()
    }
}

impl<S> Stream for TimeoutStream<S>
where
    S: Stream<Item = Result<StreamEvent>>,
{
    type Item = Result<StreamEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        let Some(inner) = this.inner.as_mut() else {
            return Poll::Ready(None);
        };
        if let Some((deadline, total)) = this.deadline.as_mut()
            && Pin::new(deadline).poll(cx).is_ready()
        {
            let total = *total;
            this.inner = None;
            return Poll::Ready(Some(Err(expired(&this.provider, TOTAL, total))));
        }
        match inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(item)) => {
                // The consumer has the item; its processing time is not idle.
                this.armed = false;
                Poll::Ready(Some(item))
            }
            Poll::Ready(None) => {
                this.inner = None;
                Poll::Ready(None)
            }
            Poll::Pending => {
                let (Some(timer), Some(idle)) = (this.timer.as_mut(), this.idle) else {
                    return Poll::Pending;
                };
                if !this.armed {
                    timer.reset(idle);
                    this.armed = true;
                }
                if Pin::new(timer).poll(cx).is_pending() {
                    return Poll::Pending;
                }
                this.inner = None;
                Poll::Ready(Some(Err(expired(
                    &this.provider,
                    "wait for the next stream event",
                    idle,
                ))))
            }
        }
    }
}

impl<T: crate::ExtractionProvider> crate::ExtractionProvider for TimeoutProvider<T> {
    async fn extract_values(
        &self,
        request: &crate::ExtractionRequest,
    ) -> Result<crate::ExtractionResponse> {
        self.bounded(
            "extraction request",
            self.policy.effective().request,
            self.inner.extract_values(request),
        )
        .await
    }

    super::forward_extraction_questions!(inner);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::{MockEmbeddingProvider, MockProvider};
    use crate::wrappers::RetryingProvider;
    use crate::{CapabilitySupport, ChatCapability, DecisionCapability, DecisionLimits};
    use crate::{ChatStream, ChatStreamExt, ProviderIdentityExt};
    use futures_util::StreamExt;

    const SHORT: Duration = Duration::from_millis(20);

    /// A provider whose calls take `delay`, and whose streams pause for
    /// `gap` before each event.
    struct Slow {
        delay: Duration,
        gap: Duration,
    }

    impl ProviderIdentity for Slow {
        fn provider_name(&self) -> &str {
            "slow"
        }

        fn reads_namespace(&self, namespace: &str) -> bool {
            namespace == "slow"
        }
    }

    impl DecisionProvider for Slow {
        async fn decide(&self, _request: &DecisionRequest) -> Result<DecisionResponse> {
            Delay::new(self.delay).await;
            Ok(DecisionResponse::default())
        }

        fn decision_capability(
            &self,
            _model: &str,
            _capability: DecisionCapability,
        ) -> CapabilitySupport {
            CapabilitySupport::Unknown
        }

        fn decision_limits(&self, _model: &str) -> DecisionLimits {
            DecisionLimits::new().max_questions(3)
        }
    }

    impl ChatProvider for Slow {
        type Stream = ChatStream;

        async fn chat(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            Delay::new(self.delay).await;
            Ok(ChatResponse::default())
        }

        async fn chat_stream(&self, _request: &ChatRequest) -> Result<ChatStream> {
            Delay::new(self.delay).await;
            let gap = self.gap;
            let events = ChatResponse::new(vec![crate::ContentBlock::text("hi")])
                .into_stream_events()
                .collect::<Vec<_>>();
            Ok(Box::pin(futures_util::stream::iter(events).then(
                move |event| async move {
                    Delay::new(gap).await;
                    Ok(event)
                },
            )))
        }

        fn chat_capability(&self, _model: &str, _capability: ChatCapability) -> CapabilitySupport {
            CapabilitySupport::Unknown
        }
    }

    fn request() -> ChatRequest {
        ChatRequest::new("m").user("hi")
    }

    #[tokio::test]
    async fn unary_calls_time_out() {
        let slow = Slow {
            delay: Duration::from_secs(5),
            gap: Duration::ZERO,
        };
        let provider = TimeoutProvider::new(slow, TimeoutPolicy::new().request(SHORT));
        let err = provider.chat(&request()).await.unwrap_err();
        assert!(matches!(err, Error::Timeout { .. }), "{err}");
        assert_eq!(err.context().unwrap().provider.as_deref(), Some("slow"));
        assert!(err.is_retryable());
        // The wrapper wrote the message, so the redacted log keeps it.
        assert_eq!(
            err.as_log().to_string(),
            r#"kind=timeout provider=slow message="chat request exceeded 20ms""#
        );
    }

    #[tokio::test]
    async fn fast_calls_and_unset_bounds_pass() {
        let provider = TimeoutProvider::new(
            MockProvider::with_text("ok"),
            TimeoutPolicy::new().request(Duration::from_secs(5)),
        );
        assert!(provider.chat(&request()).await.is_ok());
        assert!(provider.downcast_ref::<MockProvider>().is_some());

        let unbounded = TimeoutProvider::new(
            Slow {
                delay: SHORT,
                gap: SHORT,
            },
            TimeoutPolicy::new(),
        );
        let response = unbounded
            .chat_stream(&request())
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        assert_eq!(response.text().as_deref(), Some("hi"));
    }

    #[tokio::test]
    async fn stream_setup_times_out() {
        let slow = Slow {
            delay: Duration::from_secs(5),
            gap: Duration::ZERO,
        };
        let provider = TimeoutProvider::new(slow, TimeoutPolicy::new().stream_setup(SHORT));
        let err = provider.chat_stream(&request()).await.err().unwrap();
        assert!(matches!(err, Error::Timeout { .. }), "{err}");
    }

    /// A `request` bound alone bounds `chat_stream` too: a probe once saw
    /// `chat` time out at 20 ms and `chat_stream` return after 300 ms.
    #[tokio::test]
    async fn an_unset_stream_setup_takes_the_request_bound() {
        let slow = Slow {
            delay: Duration::from_millis(300),
            gap: Duration::ZERO,
        };
        let provider = TimeoutProvider::new(slow, TimeoutPolicy::new().request(SHORT));

        let err = provider.chat(&request()).await.unwrap_err();
        assert!(matches!(err, Error::Timeout { .. }), "{err}");

        let started = web_time::Instant::now();
        let err = provider.chat_stream(&request()).await.err().unwrap();
        assert!(matches!(err, Error::Timeout { .. }), "{err}");
        assert_eq!(err.message(), "stream setup exceeded 20ms");
        assert!(
            started.elapsed() < Duration::from_millis(200),
            "{:?}",
            started.elapsed()
        );

        // The field keeps what was set; the provider applies the rule.
        assert_eq!(provider.policy().stream_setup, None);
    }

    #[tokio::test]
    async fn a_stream_setup_bound_overrides_the_request_bound() {
        let slow = || Slow {
            delay: Duration::from_millis(60),
            gap: Duration::ZERO,
        };
        // Longer than the request bound: the stream opens.
        let provider = TimeoutProvider::new(
            slow(),
            TimeoutPolicy::new()
                .request(SHORT)
                .stream_setup(Duration::from_secs(5)),
        );
        assert!(provider.chat_stream(&request()).await.is_ok());
        assert!(provider.chat(&request()).await.is_err());

        // `Duration::MAX` is no bound: unary calls bounded, setup not.
        let provider = TimeoutProvider::new(
            slow(),
            TimeoutPolicy::new()
                .request(SHORT)
                .stream_setup(Duration::MAX),
        );
        assert!(provider.chat_stream(&request()).await.is_ok());
        assert!(provider.chat(&request()).await.is_err());
    }

    #[tokio::test]
    async fn a_max_bound_is_no_bound() {
        let provider = TimeoutProvider::new(
            Slow {
                delay: SHORT,
                gap: SHORT,
            },
            TimeoutPolicy::new()
                .request(Duration::MAX)
                .stream_idle(Duration::MAX)
                .stream_total(Duration::MAX),
        );
        assert!(provider.chat(&request()).await.is_ok());
        let response = provider
            .chat_stream(&request())
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        assert_eq!(response.text().as_deref(), Some("hi"));
    }

    #[tokio::test]
    async fn idle_streams_time_out_and_end() {
        let slow = Slow {
            delay: Duration::ZERO,
            gap: Duration::from_secs(5),
        };
        let provider = TimeoutProvider::new(slow, TimeoutPolicy::new().stream_idle(SHORT));
        let items: Vec<_> = provider
            .chat_stream(&request())
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(items.len(), 1);
        assert!(matches!(items[0], Err(Error::Timeout { .. })));
    }

    #[tokio::test]
    async fn idle_bound_resets_on_each_event() {
        let slow = Slow {
            delay: Duration::ZERO,
            gap: Duration::from_millis(10),
        };
        let provider = TimeoutProvider::new(
            slow,
            TimeoutPolicy::new().stream_idle(Duration::from_millis(200)),
        );
        let response = provider
            .chat_stream(&request())
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        assert_eq!(response.text().as_deref(), Some("hi"));
    }

    /// The idle bound measures the provider, not the consumer: time spent
    /// between polls does not count, even against a stream that only makes
    /// progress while polled.
    #[tokio::test]
    async fn slow_consumers_do_not_trip_the_idle_bound() {
        let fast = Slow {
            delay: Duration::ZERO,
            gap: SHORT,
        };
        let provider = TimeoutProvider::new(
            fast,
            TimeoutPolicy::new().stream_idle(Duration::from_millis(100)),
        );
        let mut stream = provider.chat_stream(&request()).await.unwrap();
        let mut events = 0;
        while let Some(item) = stream.next().await {
            item.unwrap();
            events += 1;
            Delay::new(Duration::from_millis(150)).await;
        }
        assert!(events > 1, "{events}");
    }

    /// A stall after an item still trips the bound, measured from the poll
    /// that found nothing ready.
    #[tokio::test]
    async fn a_stall_after_an_item_times_out_and_ends() {
        let first = StreamEvent::response_start(None, None);
        let stream = futures_util::stream::iter([Ok(first)]).chain(futures_util::stream::pending());
        let mut stream = TimeoutStream::new(stream, Some(SHORT), None, "stall".into());
        assert!(stream.next().await.unwrap().is_ok());
        Delay::new(SHORT * 3).await;
        let started = web_time::Instant::now();
        let err = stream.next().await.unwrap().unwrap_err();
        assert!(matches!(err, Error::Timeout { .. }), "{err}");
        assert!(started.elapsed() >= SHORT, "{:?}", started.elapsed());
        assert!(stream.next().await.is_none());
    }

    /// The idle bound starts at the first wait, not when the stream opens:
    /// a consumer that polls late is not timed out for it.
    #[tokio::test]
    async fn a_late_first_poll_does_not_trip_the_idle_bound() {
        let slow = Slow {
            delay: Duration::ZERO,
            gap: Duration::from_millis(10),
        };
        let provider = TimeoutProvider::new(
            slow,
            TimeoutPolicy::new().stream_idle(Duration::from_millis(50)),
        );
        let stream = provider.chat_stream(&request()).await.unwrap();
        // The first event is not ready when first polled: a timer started
        // at creation would have expired by then.
        Delay::new(Duration::from_millis(100)).await;
        let response = stream.collect_response().await.unwrap();
        assert_eq!(response.text().as_deref(), Some("hi"));
    }

    /// A first item that never comes still trips the bound, measured from
    /// the first poll.
    #[tokio::test]
    async fn a_stalled_first_item_times_out() {
        let stream = futures_util::stream::pending::<Result<StreamEvent>>();
        let mut stream = TimeoutStream::new(
            stream,
            Some(Duration::from_millis(50)),
            None,
            "stall".into(),
        );
        Delay::new(Duration::from_millis(100)).await;
        let started = web_time::Instant::now();
        let err = stream.next().await.unwrap().unwrap_err();
        assert!(matches!(err, Error::Timeout { .. }), "{err}");
        assert!(
            started.elapsed() >= Duration::from_millis(50),
            "{:?}",
            started.elapsed()
        );
        assert!(stream.next().await.is_none());
    }

    /// A provider whose streams never end: `ResponseStart`, then a text
    /// delta every `gap` (always ready when `gap` is zero).
    #[derive(Clone, Default)]
    struct Endless {
        gap: Duration,
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl ProviderIdentity for Endless {
        fn provider_name(&self) -> &str {
            "endless"
        }

        fn reads_namespace(&self, namespace: &str) -> bool {
            namespace == "endless"
        }
    }

    impl ChatProvider for Endless {
        type Stream = ChatStream;

        async fn chat(&self, _: &ChatRequest) -> Result<ChatResponse> {
            unreachable!()
        }

        async fn chat_stream(&self, _: &ChatRequest) -> Result<ChatStream> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let gap = self.gap;
            let start = futures_util::stream::iter([Ok(StreamEvent::response_start(None, None))]);
            let deltas = futures_util::stream::repeat(()).then(move |()| async move {
                if !gap.is_zero() {
                    Delay::new(gap).await;
                }
                Ok(StreamEvent::text_delta(0, "token"))
            });
            Ok(Box::pin(start.chain(deltas)))
        }

        fn chat_capability(&self, _: &str, _: ChatCapability) -> CapabilitySupport {
            CapabilitySupport::Unknown
        }
    }

    /// Drain `stream` into its items, stopping after the first error.
    async fn items_until_error(
        mut stream: impl Stream<Item = Result<StreamEvent>> + Unpin,
    ) -> (usize, Option<Error>, bool) {
        let mut ok = 0;
        while let Some(item) = stream.next().await {
            match item {
                Ok(_) => ok += 1,
                Err(error) => {
                    let ended = stream.next().await.is_none();
                    return (ok, Some(error), ended);
                }
            }
        }
        (ok, None, true)
    }

    #[tokio::test]
    async fn a_steady_stream_is_cut_off_at_the_total_bound_then_ends() {
        for gap in [Duration::ZERO, Duration::from_millis(5)] {
            let provider = TimeoutProvider::new(
                Endless {
                    gap,
                    ..Endless::default()
                },
                TimeoutPolicy::new().stream_total(Duration::from_millis(60)),
            );
            let started = web_time::Instant::now();
            let stream = provider.chat_stream(&request()).await.unwrap();
            let (ok, error, ended) = items_until_error(stream).await;
            let error = error.expect("the total bound ends an endless stream");
            assert!(matches!(error, Error::Timeout { .. }), "{error}");
            assert_eq!(error.message(), "stream duration exceeded 60ms");
            assert_eq!(
                error.context().unwrap().provider.as_deref(),
                Some("endless")
            );
            assert!(ok > 1, "{gap:?}: {ok} events before the bound");
            assert!(ended, "{gap:?}: the stream ends after the timeout");
            assert!(started.elapsed() >= Duration::from_millis(60));
        }
    }

    #[tokio::test]
    async fn the_total_bound_covers_stream_setup() {
        let slow = Slow {
            delay: Duration::from_secs(5),
            gap: Duration::ZERO,
        };
        let provider = TimeoutProvider::new(
            slow,
            TimeoutPolicy::new()
                .stream_setup(Duration::from_secs(1))
                .stream_total(SHORT),
        );
        let err = provider.chat_stream(&request()).await.err().unwrap();
        assert_eq!(err.message(), "stream duration exceeded 20ms");

        // A shorter setup bound still fires first.
        let slow = Slow {
            delay: Duration::from_secs(5),
            gap: Duration::ZERO,
        };
        let provider = TimeoutProvider::new(
            slow,
            TimeoutPolicy::new()
                .stream_setup(SHORT)
                .stream_total(Duration::from_secs(1)),
        );
        let err = provider.chat_stream(&request()).await.err().unwrap();
        assert_eq!(err.message(), "stream setup exceeded 20ms");
    }

    #[tokio::test]
    async fn the_idle_or_total_bound_whichever_expires_first_wins() {
        // Steady events keep the idle bound satisfied; the total bound ends
        // the stream.
        let provider = TimeoutProvider::new(
            Endless {
                gap: Duration::from_millis(5),
                ..Endless::default()
            },
            TimeoutPolicy::new()
                .stream_idle(Duration::from_millis(50))
                .stream_total(Duration::from_millis(80)),
        );
        let stream = provider.chat_stream(&request()).await.unwrap();
        let (_, error, _) = items_until_error(stream).await;
        assert_eq!(error.unwrap().message(), "stream duration exceeded 80ms");

        // A stall trips the shorter idle bound long before the total one.
        let first = StreamEvent::response_start(None, None);
        let stalled =
            futures_util::stream::iter([Ok(first)]).chain(futures_util::stream::pending());
        let stream = TimeoutStream::new(
            stalled,
            Some(SHORT),
            Some((Delay::new(Duration::from_secs(5)), Duration::from_secs(5))),
            "stall".into(),
        );
        let (ok, error, ended) = items_until_error(stream).await;
        assert_eq!(ok, 1);
        assert_eq!(
            error.unwrap().message(),
            "wait for the next stream event exceeded 20ms"
        );
        assert!(ended);
    }

    /// Once output has started, a total-bound timeout is passed through:
    /// retry does not start the stream again.
    #[tokio::test]
    async fn retry_passes_a_total_timeout_after_output_through() {
        let endless = Endless {
            gap: Duration::from_millis(5),
            ..Endless::default()
        };
        let provider = RetryingProvider::new(
            TimeoutProvider::new(
                endless.clone(),
                TimeoutPolicy::new().stream_total(Duration::from_millis(40)),
            ),
            crate::wrappers::RetryPolicy::default(),
        );
        let stream = provider.chat_stream(&request()).await.unwrap();
        let (ok, error, _) = items_until_error(stream).await;
        assert!(ok > 1);
        assert!(matches!(error, Some(Error::Timeout { .. })), "{error:?}");
        assert_eq!(endless.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn retry_retries_an_uncommitted_stall() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        // First stream stalls before any event; the retry gets a fast one.
        struct StallOnce(Arc<AtomicUsize>);
        impl ProviderIdentity for StallOnce {
            fn provider_name(&self) -> &str {
                "stall-once"
            }

            fn reads_namespace(&self, namespace: &str) -> bool {
                namespace == "stall-once"
            }
        }
        impl ChatProvider for StallOnce {
            type Stream = ChatStream;
            async fn chat(&self, _: &ChatRequest) -> Result<ChatResponse> {
                unreachable!()
            }
            async fn chat_stream(&self, _: &ChatRequest) -> Result<ChatStream> {
                if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
                    return Ok(Box::pin(futures_util::stream::pending()));
                }
                Ok(Box::pin(
                    ChatResponse::new(vec![crate::ContentBlock::text("second")]).into_stream(),
                ))
            }
            fn chat_capability(&self, _: &str, _: ChatCapability) -> CapabilitySupport {
                CapabilitySupport::Unknown
            }
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let provider = RetryingProvider::new(
            TimeoutProvider::new(
                StallOnce(calls.clone()),
                TimeoutPolicy::new().stream_idle(SHORT),
            ),
            crate::wrappers::RetryPolicy::default(),
        );
        let response = provider
            .chat_stream(&request())
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        assert_eq!(response.text().as_deref(), Some("second"));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn decisions_are_bounded_by_the_request_timeout() {
        let provider = TimeoutProvider::new(
            Slow {
                delay: Duration::from_millis(200),
                gap: Duration::ZERO,
            },
            TimeoutPolicy::new().request(Duration::from_millis(10)),
        );
        let request =
            DecisionRequest::new("d", "x").question("q", crate::decision::Question::yes_no("?"));
        let err = provider.decide(&request).await.unwrap_err();
        assert!(matches!(err, Error::Timeout { .. }), "{err:?}");
        assert_eq!(provider.decision_limits("d").max_questions, Some(3));
    }

    #[tokio::test]
    async fn embeddings_are_bounded_by_the_request_timeout() {
        let provider = TimeoutProvider::new(
            MockEmbeddingProvider::with_vectors(vec![vec![1.0]]),
            TimeoutPolicy::new().request(Duration::from_secs(5)),
        );
        assert!(
            provider
                .embed(&EmbeddingRequest::new("e").inputs(["a"]))
                .await
                .is_ok()
        );
    }
}
