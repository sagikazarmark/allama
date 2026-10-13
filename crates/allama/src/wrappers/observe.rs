use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use futures_core::Stream;
use web_time::Instant;

use crate::chat::stream::StreamCollector;
use crate::{
    ChatProvider, ChatRequest, ChatResponse, DecisionProvider, DecisionRequest, DecisionResponse,
    EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, Error, ProviderIdentity, Result,
    StreamEvent, Usage, Warning,
};

/// Hands every warning a call reports to a callback: for metrics, alerts or
/// an application's own log, alongside (or instead of) the warning events.
///
/// The callback gets the requested model and each warning of a successful
/// chat response, of each `StreamEvent::Warning` as the stream yields it, and
/// of embedding and decision responses. Unlike the
/// [warning events](crate::warning#warning-events), it sees every
/// occurrence: nothing is deduplicated, and a warning emitted already is
/// still observed. A call that fails reports no response, so its warnings
/// are not observed.
///
/// ```
/// # #[cfg(feature = "mock")] {
/// use std::sync::Arc;
/// use std::sync::atomic::{AtomicUsize, Ordering};
///
/// use allama::mock::MockProvider;
/// use allama::wrappers::ObserveProvider;
/// use allama::WarningKind;
///
/// let failovers = Arc::new(AtomicUsize::new(0));
/// let counter = failovers.clone();
/// let provider = ObserveProvider::new(MockProvider::with_text("hi")).on_warning(move |_model, warning| {
///     if warning.kind == WarningKind::Fallback {
///         counter.fetch_add(1, Ordering::Relaxed);
///     }
/// });
/// # let _ = provider; }
/// ```
///
/// Put it where the warnings should be counted: outermost to see what the
/// application gets, failover warnings included. Everything else is
/// forwarded unchanged: it is a transparent wrapper. Whole responses are
/// recorded with [`RecordingProvider`](super::RecordingProvider).
///
/// # Latency, usage and outcomes
///
/// [`on_outcome`](Self::on_outcome) hands over one [`CallOutcome`] per
/// call: its usage, its error, its duration, and for a stream the time to
/// the first output, for a metrics endpoint the service exposes itself
/// (Prometheus pull) without an OpenTelemetry Collector:
///
/// ```
/// # #[cfg(feature = "mock")] {
/// use allama::mock::MockProvider;
/// use allama::wrappers::ObserveProvider;
///
/// let provider = ObserveProvider::new(MockProvider::with_text("hi")).on_outcome(|outcome| {
///     let kind = outcome.error.map_or("ok", |error| error.kind_name());
///     let tokens = outcome.usage.and_then(|usage| usage.total()).unwrap_or(0);
///     println!("{} {kind} {tokens} tokens in {:?}", outcome.model, outcome.duration);
/// });
/// # let _ = provider; }
/// ```
///
/// With a Collector, `TracingProvider` (feature `tracing`) records the same
/// and more on one span per call, attempts and failovers included. A
/// request id header is not an
/// observation but a change to the request:
/// [`MapRequestProvider`](super::MapRequestProvider) adds it to the
/// request's `allama_http::HttpRequestOptions`, which every HTTP adapter
/// sends (see that type's "From a wrapper"). Together:
/// `TracingProvider::new(MapRequestProvider::new(provider, tag))`.
///
/// # Which model, which provider
///
/// The model the callback gets is the one this layer received, the
/// request's model. A wrapper below it may send another: a
/// [`MapModelProvider`](super::MapModelProvider) maps it, and a
/// [`FallbackProvider`](super::FallbackProvider) may serve the call from
/// its second side. The warning's [`provider`](Warning::provider) names the
/// provider that reported it; outside the callback, the response's
/// `provider` and `model` name the one that served. An `ObserveProvider`
/// on each side of a fallback sees that side's model, but misses the
/// failover warning, which the fallback adds above its sides: observe
/// failovers outermost.
#[derive(Clone)]
pub struct ObserveProvider<P> {
    inner: P,
    on_warning: Option<WarningObserver>,
    on_outcome: Option<OutcomeObserver>,
}

/// The callback of [`ObserveProvider::on_warning`].
type WarningObserver = Arc<dyn Fn(&str, &Warning) + Send + Sync>;

/// The callback of [`ObserveProvider::on_outcome`].
type OutcomeObserver = Arc<dyn Fn(&CallOutcome<'_>) + Send + Sync>;

/// What one call observed by an [`ObserveProvider`] came to, handed to its
/// [`on_outcome`](ObserveProvider::on_outcome) callback once the call
/// ends: a unary call when it returns, a stream when it ends, fails, or is
/// dropped.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct CallOutcome<'a> {
    /// Which operation the call was.
    pub kind: CallKind,
    /// The model the request asked for, at this layer.
    pub model: &'a str,
    /// The provider that served the call, as the response reports it;
    /// `None` when it failed or reported none.
    pub provider: Option<&'a str>,
    /// The usage the call reported. For a stream that failed or was
    /// dropped, the usage it reported before.
    pub usage: Option<&'a Usage>,
    /// The error the call ended with, if it failed. A stream that failed
    /// mid-way reports the error it ended with.
    pub error: Option<&'a Error>,
    /// From the call to its end: for a stream, to its last event.
    pub duration: Duration,
    /// For a stream, from the call to its first model output
    /// ([`StreamEvent::is_output`]); `None` without output.
    pub time_to_first_output: Option<Duration>,
    /// The caller dropped the stream before it ended.
    pub cancelled: bool,
}

/// The operation a [`CallOutcome`] is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CallKind {
    /// `chat`.
    Chat,
    /// `chat_stream`.
    ChatStream,
    /// `embed`.
    Embedding,
    /// `decide`.
    Decision,
}

impl<'a> CallOutcome<'a> {
    fn unary<T>(
        kind: CallKind,
        model: &'a str,
        started: Instant,
        result: &'a Result<T>,
        parts: impl FnOnce(&'a T) -> (Option<&'a str>, Option<&'a Usage>),
    ) -> Self {
        let (provider, usage, error) = match result {
            Ok(response) => {
                let (provider, usage) = parts(response);
                (provider, usage, None)
            }
            Err(error) => (None, None, Some(error)),
        };
        Self {
            kind,
            model,
            provider,
            usage,
            error,
            duration: started.elapsed(),
            time_to_first_output: None,
            cancelled: false,
        }
    }
}

impl<P> ObserveProvider<P> {
    /// Wrap `inner`, observing nothing yet.
    #[must_use]
    pub fn new(inner: P) -> Self {
        Self {
            inner,
            on_warning: None,
            on_outcome: None,
        }
    }

    /// Call `observe` with each call's [`CallOutcome`] once the call ends,
    /// replacing any earlier callback.
    #[must_use]
    pub fn on_outcome(
        mut self,
        observe: impl Fn(&CallOutcome<'_>) + Send + Sync + 'static,
    ) -> Self {
        self.on_outcome = Some(Arc::new(observe));
        self
    }

    fn outcome(&self, outcome: &CallOutcome<'_>) {
        if let Some(observe) = &self.on_outcome {
            observe(outcome);
        }
    }

    /// Call `observe` with the requested model and each warning a call
    /// reports, replacing any earlier callback.
    #[must_use]
    pub fn on_warning(mut self, observe: impl Fn(&str, &Warning) + Send + Sync + 'static) -> Self {
        self.on_warning = Some(Arc::new(observe));
        self
    }

    /// Borrow the wrapped provider.
    #[must_use]
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Unwrap into the wrapped provider.
    #[must_use]
    pub fn into_inner(self) -> P {
        self.inner
    }

    fn observe(&self, model: &str, warnings: &[Warning]) {
        if let Some(observe) = &self.on_warning {
            for warning in warnings {
                observe(model, warning);
            }
        }
    }
}

impl<P: ChatProvider> ChatProvider for ObserveProvider<P> {
    type Stream = ObserveStream<P::Stream>;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let started = Instant::now();
        let result = self.inner.chat(request).await;
        self.outcome(&CallOutcome::unary(
            CallKind::Chat,
            &request.model,
            started,
            &result,
            |response| (response.provider.as_deref(), response.usage.as_ref()),
        ));
        let response = result?;
        self.observe(&request.model, &response.warnings);
        Ok(response)
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
        let started = Instant::now();
        let stream = match self.inner.chat_stream(request).await {
            Ok(stream) => stream,
            Err(error) => {
                self.outcome(&CallOutcome::unary(
                    CallKind::ChatStream,
                    &request.model,
                    started,
                    &Err::<(), _>(error.clone()),
                    |()| (None, None),
                ));
                return Err(error);
            }
        };
        Ok(ObserveStream {
            inner: Box::pin(stream),
            model: request.model.clone(),
            on_warning: self.on_warning.clone(),
            outcome: self.on_outcome.clone().map(|observe| StreamOutcome {
                observe,
                started,
                first_output: None,
                collector: StreamCollector::metadata_only(),
                error: None,
            }),
        })
    }

    super::forward_chat_questions!(inner);
}

impl<P: EmbeddingProvider> EmbeddingProvider for ObserveProvider<P> {
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        let started = Instant::now();
        let result = self.inner.embed(request).await;
        self.outcome(&CallOutcome::unary(
            CallKind::Embedding,
            &request.model,
            started,
            &result,
            |response| (response.provider.as_deref(), response.usage.as_ref()),
        ));
        let response = result?;
        self.observe(&request.model, &response.warnings);
        Ok(response)
    }

    super::forward_embedding_questions!(inner);
}

impl<P: DecisionProvider> DecisionProvider for ObserveProvider<P> {
    async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse> {
        let started = Instant::now();
        let result = self.inner.decide(request).await;
        self.outcome(&CallOutcome::unary(
            CallKind::Decision,
            &request.model,
            started,
            &result,
            |response| (response.provider.as_deref(), response.usage.as_ref()),
        ));
        let response = result?;
        self.observe(&request.model, &response.warnings);
        Ok(response)
    }

    super::forward_decision_questions!(inner);
}

impl<P: ProviderIdentity> ProviderIdentity for ObserveProvider<P> {
    super::forward_provider_identity!(inner);
}

impl<P> fmt::Debug for ObserveProvider<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ObserveProvider")
            .field("on_warning", &self.on_warning.is_some())
            .field("on_outcome", &self.on_outcome.is_some())
            .finish_non_exhaustive()
    }
}

/// The stream of an [`ObserveProvider`]: the inner stream, with each
/// warning event observed as it passes.
pub struct ObserveStream<S> {
    inner: Pin<Box<S>>,
    model: String,
    on_warning: Option<WarningObserver>,
    /// What the outcome callback needs, until the stream reports it.
    outcome: Option<StreamOutcome>,
}

/// A stream's outcome so far.
struct StreamOutcome {
    observe: OutcomeObserver,
    started: Instant,
    first_output: Option<Duration>,
    /// Folds the stream's usage and serving provider.
    collector: StreamCollector,
    error: Option<Error>,
}

impl StreamOutcome {
    fn see(&mut self, event: &StreamEvent) {
        if self.first_output.is_none() && event.is_output() {
            self.first_output = Some(self.started.elapsed());
        }
        // A protocol error belongs to the consumer; the outcome only
        // folds what it can.
        let _ = self.collector.push_ref(event);
    }

    fn report(self, model: &str, cancelled: bool) {
        let duration = self.started.elapsed();
        let collected = self.collector.finish_partial().ok();
        let response = collected.as_ref().map(|collected| &collected.response);
        (self.observe)(&CallOutcome {
            kind: CallKind::ChatStream,
            model,
            provider: response.and_then(|response| response.provider.as_deref()),
            usage: response.and_then(|response| response.usage.as_ref()),
            error: self.error.as_ref(),
            duration,
            time_to_first_output: self.first_output,
            cancelled,
        });
    }
}

impl<S> Drop for ObserveStream<S> {
    fn drop(&mut self) {
        if let Some(outcome) = self.outcome.take() {
            outcome.report(&self.model, true);
        }
    }
}

impl<S> fmt::Debug for ObserveStream<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ObserveStream")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl<S: Stream<Item = Result<StreamEvent>>> Stream for ObserveStream<S> {
    type Item = Result<StreamEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        let poll = this.inner.as_mut().poll_next(cx);
        if let Poll::Ready(Some(Ok(StreamEvent::Warning { warning }))) = &poll
            && let Some(observe) = &this.on_warning
        {
            observe(&this.model, warning);
        }
        match &poll {
            Poll::Ready(Some(Ok(event))) => {
                if let Some(outcome) = &mut this.outcome {
                    outcome.see(event);
                }
            }
            // A chat stream ends after its first error.
            Poll::Ready(Some(Err(error))) => {
                if let Some(mut outcome) = this.outcome.take() {
                    outcome.error = Some(error.clone());
                    outcome.report(&this.model, false);
                }
            }
            Poll::Ready(None) => {
                if let Some(outcome) = this.outcome.take() {
                    outcome.report(&this.model, false);
                }
            }
            Poll::Pending => {}
        }
        poll
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use futures_util::StreamExt;

    use super::*;
    use crate::mock::{MockDecisionProvider, MockEmbeddingProvider, MockProvider};
    use crate::{CapabilitySupport, ChatCapability};
    use crate::{ReasoningEffort, Subject, WarningKind};

    type Seen = Arc<Mutex<Vec<(String, String)>>>;

    fn observed<P>(inner: P) -> (ObserveProvider<P>, Seen) {
        let seen = Seen::default();
        let sink = seen.clone();
        let provider = ObserveProvider::new(inner).on_warning(move |model, warning| {
            sink.lock()
                .unwrap()
                .push((model.to_owned(), warning.message.clone()));
        });
        (provider, seen)
    }

    fn no_reasoning() -> MockProvider {
        MockProvider::build(|b| b.text("a").text("b")).with_chat_capability(
            ChatCapability::ReasoningConfig,
            CapabilitySupport::Unsupported,
        )
    }

    fn reasoning_request() -> ChatRequest {
        ChatRequest::new("m")
            .user("q")
            .reasoning(ReasoningEffort::High)
    }

    #[tokio::test]
    async fn chat_warnings_are_observed_on_both_paths() {
        let (provider, seen) = observed(no_reasoning());
        provider.chat(&reasoning_request()).await.unwrap();
        let events: Vec<_> = provider
            .chat_stream(&reasoning_request())
            .await
            .unwrap()
            .collect()
            .await;
        assert!(events.iter().all(Result::is_ok));
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 2, "{seen:?}");
        assert!(seen.iter().all(|(model, _)| model == "m"));
        assert_eq!(seen[0], seen[1]);
    }

    /// Observers see every occurrence, also of a warning emitted already.
    #[tokio::test]
    async fn every_occurrence_is_observed() {
        let (provider, seen) = observed(MockProvider::empty());
        let mut warning = Warning::new(WarningKind::Other, Subject::Response, "again");
        crate::warning::emit("p", "m", &mut warning);
        for _ in 0..3 {
            let inner = ChatResponse::default().warning(warning.clone());
            provider.inner().push_response(inner);
        }
        for _ in 0..3 {
            provider
                .chat(&ChatRequest::new("m").user("q"))
                .await
                .unwrap();
        }
        assert_eq!(seen.lock().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn embedding_and_decision_warnings_are_observed() {
        let hint = || Warning::new(WarningKind::Other, Subject::Response, "note");
        let (embedder, seen) = observed(MockEmbeddingProvider::new([Ok(EmbeddingResponse::new(
            vec![vec![1.0]],
        )
        .warning(hint()))]));
        embedder
            .embed(&EmbeddingRequest::new("e").input("x"))
            .await
            .unwrap();
        assert_eq!(*seen.lock().unwrap(), [("e".to_owned(), "note".to_owned())]);

        let decision = DecisionResponse::default().warning(hint());
        let (decider, seen) = observed(MockDecisionProvider::with_response(decision));
        let request = DecisionRequest::new("d", "context")
            .question("q", crate::decision::Question::yes_no("?"));
        decider.decide(&request).await.unwrap();
        assert_eq!(*seen.lock().unwrap(), [("d".to_owned(), "note".to_owned())]);
    }

    #[test]
    fn without_a_callback_nothing_happens() {
        let provider = ObserveProvider::new(MockProvider::with_text("a"));
        provider.observe(
            "m",
            &[Warning::new(WarningKind::Other, Subject::Response, "x")],
        );
        assert!(format!("{provider:?}").contains("on_warning: false"));
    }

    type Outcomes = Arc<
        Mutex<
            Vec<(
                CallKind,
                Option<String>,
                Option<u64>,
                Option<&'static str>,
                bool,
                bool,
            )>,
        >,
    >;

    /// Records `(kind, provider, total tokens, error kind, first output
    /// seen, cancelled)` per call.
    fn outcomes<P>(inner: P) -> (ObserveProvider<P>, Outcomes) {
        let seen = Outcomes::default();
        let sink = seen.clone();
        let provider = ObserveProvider::new(inner).on_outcome(move |outcome| {
            sink.lock().unwrap().push((
                outcome.kind,
                outcome.provider.map(str::to_owned),
                outcome.usage.and_then(Usage::total),
                outcome.error.map(Error::kind_name),
                outcome.time_to_first_output.is_some(),
                outcome.cancelled,
            ));
        });
        (provider, seen)
    }

    fn with_usage() -> ChatResponse {
        let mut response = crate::mock::ChatResponseBuilder::new()
            .text("hi")
            .usage(3, 2)
            .build();
        response.provider = Some("mock".into());
        response
    }

    /// One outcome per call, for metrics without a
    /// Collector.
    #[tokio::test]
    async fn every_call_reports_one_outcome() {
        let (provider, seen) = outcomes(MockProvider::with_response(with_usage()));
        provider
            .chat(&ChatRequest::new("m").user("q"))
            .await
            .unwrap();
        let (provider_err, seen_err) =
            outcomes(MockProvider::with_error(Error::rate_limited("slow down")));
        provider_err
            .chat(&ChatRequest::new("m").user("q"))
            .await
            .unwrap_err();
        assert_eq!(
            *seen.lock().unwrap(),
            [(
                CallKind::Chat,
                Some("mock".into()),
                Some(5),
                None,
                false,
                false
            )]
        );
        assert_eq!(
            *seen_err.lock().unwrap(),
            [(
                CallKind::Chat,
                None,
                None,
                Some("rate_limited"),
                false,
                false
            )]
        );

        let (embedding, seen) = outcomes(MockEmbeddingProvider::with_vectors(vec![vec![0.1]]));
        embedding
            .embed(&EmbeddingRequest::new("e").inputs(["a"]))
            .await
            .unwrap();
        assert_eq!(seen.lock().unwrap()[0].0, CallKind::Embedding);
    }

    #[tokio::test]
    async fn a_stream_reports_its_outcome_when_it_ends_or_is_dropped() {
        let (provider, seen) = outcomes(MockProvider::with_response(with_usage()));
        let request = ChatRequest::new("m").user("q");
        let mut stream = provider.chat_stream(&request).await.unwrap();
        while stream.next().await.is_some() {}
        assert_eq!(
            *seen.lock().unwrap(),
            [(
                CallKind::ChatStream,
                Some("mock".into()),
                Some(5),
                None,
                true,
                false
            )]
        );

        let (provider, seen) = outcomes(MockProvider::with_response(with_usage()));
        let mut stream = provider.chat_stream(&request).await.unwrap();
        stream.next().await;
        drop(stream);
        let outcomes = seen.lock().unwrap();
        assert_eq!(outcomes.len(), 1);
        assert!(outcomes[0].5, "a dropped stream is cancelled");
    }
}
