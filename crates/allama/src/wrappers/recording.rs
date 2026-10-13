use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures_core::Stream;
use serde::{Deserialize, Serialize};

use crate::chat::stream::{StreamCollector, StreamTranscript};
use crate::{
    ChatProvider, ChatRequest, ChatResponse, DecisionProvider, DecisionRequest, DecisionResponse,
    EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, Error, ProviderIdentity, Result,
    StreamEvent, Warning,
};

/// The record of one chat call: what was asked, and what happened.
///
/// Produced by [`RecordingProvider`] for every `chat` and `chat_stream`
/// call. Requests, responses, errors and transcripts are all serializable,
/// so an exchange can be logged as JSON, stored as a fixture, or compared
/// across providers. It is an observability artifact, not a conversation
/// store.
///
/// - Unary calls set `response` or `error`.
/// - Streams also set `stream`, the transcript of every event that reached
///   the caller. `response` is set when the stream completed; otherwise
///   `error` holds the setup error, the error that ended the stream, or the
///   incomplete-stream error when the caller stopped reading early.
///
/// To replay one in a test, script a mock from it: the `allama::mock`
/// docs (feature `mock`) have the recipe, under "Replaying a recorded
/// call".
///
/// # Stability
///
/// An exchange is a same-version artifact. It serializes the request,
/// response, error and stream event types of the allama version that
/// recorded it, and carries no format version, so a record from another
/// version may not load (a stream event kind this version does not know)
/// or may load with fields left out (a request field it does not know).
/// Re-record fixtures after upgrading allama. Keep conversations that must
/// outlive an upgrade in a [`History`](crate::history::History), which is
/// versioned, and a call's envelope in a
/// [`TurnEnvelope`](crate::history::TurnEnvelope).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ChatExchange {
    /// Identity of the wrapped provider (see
    /// [`ProviderIdentity::provider_name`]); the provider that actually
    /// served the call is in [`ChatResponse::provider`].
    pub provider: String,
    /// The request, as the wrapped provider received it.
    pub request: ChatRequest,
    /// The response, when the call succeeded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<ChatResponse>,
    /// The error, when the call failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<Error>,
    /// The stream transcript, for streaming calls that opened a stream.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<StreamTranscript>,
}

impl ChatExchange {
    /// Returns `true` when the call produced a complete response.
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.response.is_some()
    }
}

/// Hands a [`ChatExchange`] to a sink for every chat call.
///
/// The sink runs once per call, when the outcome is known:
///
/// - unary calls: when `chat` returns;
/// - streams: when the stream ends, fails, or is dropped by the caller, so
///   abandoned streams are recorded too. Stream setup errors are recorded
///   immediately.
///
/// The sink runs inline, so keep it cheap (send the record to a channel,
/// append to a buffer). Stream events are recorded as they pass through,
/// which keeps a copy of every event until the stream ends.
///
/// ```
/// # #[cfg(feature = "mock")] {
/// use std::sync::{Arc, Mutex};
/// use allama::mock::MockProvider;
/// use allama::wrappers::{ChatExchange, RecordingProvider};
///
/// let log: Arc<Mutex<Vec<ChatExchange>>> = Arc::default();
/// let sink = log.clone();
/// let provider = RecordingProvider::new(MockProvider::with_text("hi"), move |exchange| {
///     sink.lock().unwrap().push(exchange);
/// });
/// # let _ = provider; }
/// ```
///
/// Embeddings and decisions pass through unrecorded. Place the wrapper where
/// the records should be taken: outermost to record what the application saw,
/// around a single provider (inside fallback or retry) to record every
/// attempt.
///
/// The recorded request keeps the messages' and system prompts'
/// [application data](crate::history#application-data), so an exchange
/// replayed as a fixture reproduces the application's request exactly. Call
/// [`strip_app_data`](Self::strip_app_data) to leave it out of the records.
///
/// The warnings in a record are marked as emitted (see
/// [warning events](crate::warning#warning-events)): a record replayed in
/// memory through a wrapper built on [`chat::prepare`](crate::adapter::chat::prepare)
/// does not log them again. A record that was serialized loses the mark, as
/// any deserialized warning does.
pub struct RecordingProvider<P, F> {
    inner: P,
    sink: Arc<F>,
    strip_app_data: bool,
}

impl<P, F> RecordingProvider<P, F>
where
    F: Fn(ChatExchange) + Send + Sync,
{
    /// Wrap `inner`, handing every exchange to `sink`.
    #[must_use]
    pub fn new(inner: P, sink: F) -> Self {
        Self {
            inner,
            sink: Arc::new(sink),
            strip_app_data: false,
        }
    }

    /// Leave [application data](crate::history#application-data) out of
    /// recorded requests, for example when records go to a log that must not
    /// hold it. The wrapped provider still receives the request unchanged.
    #[must_use]
    pub fn strip_app_data(mut self) -> Self {
        self.strip_app_data = true;
        self
    }

    /// Borrow the wrapped provider.
    #[must_use]
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Unwrap the provider.
    #[must_use]
    pub fn into_inner(self) -> P {
        self.inner
    }
}

impl<P, F> RecordingProvider<P, F> {
    /// The request as it goes into the record.
    fn recorded(&self, request: &ChatRequest) -> ChatRequest {
        let mut request = request.clone();
        if self.strip_app_data {
            for prompt in &mut request.system {
                prompt.app_data.clear();
            }
            for message in &mut request.messages {
                message.app_data_mut().clear();
            }
        }
        request
    }
}

impl<P, F> ChatProvider for RecordingProvider<P, F>
where
    P: ChatProvider,
    F: Fn(ChatExchange) + Send + Sync + 'static,
{
    type Stream = RecordingStream<P::Stream, F>;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let result = self.inner.chat(request).await;
        let (response, error) = match &result {
            Ok(response) => (Some(recorded_response(response.clone())), None),
            Err(error) => (None, Some(error.clone())),
        };
        (self.sink)(ChatExchange {
            provider: self.inner.provider_name().to_owned(),
            request: self.recorded(request),
            response,
            error,
            stream: None,
        });
        result
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
        let exchange = ChatExchange {
            provider: self.inner.provider_name().to_owned(),
            request: self.recorded(request),
            response: None,
            error: None,
            stream: None,
        };
        match self.inner.chat_stream(request).await {
            Ok(stream) => Ok(RecordingStream {
                inner: Box::pin(stream),
                transcript: StreamTranscript::new(),
                pending: Some((exchange, self.sink.clone())),
            }),
            Err(error) => {
                (self.sink)(ChatExchange {
                    error: Some(error.clone()),
                    ..exchange
                });
                Err(error)
            }
        }
    }

    super::forward_chat_questions!(inner);
}

impl<P, F> EmbeddingProvider for RecordingProvider<P, F>
where
    P: EmbeddingProvider,
    F: Fn(ChatExchange) + Send + Sync,
{
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        self.inner.embed(request).await
    }

    super::forward_embedding_questions!(inner);
}

impl<P, F> DecisionProvider for RecordingProvider<P, F>
where
    P: DecisionProvider,
    F: Fn(ChatExchange) + Send + Sync,
{
    async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse> {
        self.inner.decide(request).await
    }

    super::forward_decision_questions!(inner);
}

impl<P, F> ProviderIdentity for RecordingProvider<P, F>
where
    P: ProviderIdentity,
    F: Send + Sync,
{
    super::forward_provider_identity!(inner);
}

/// Clones share one sink: every clone hands its exchanges to the same
/// function, which need not be `Clone` itself.
impl<P: Clone, F> Clone for RecordingProvider<P, F> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            sink: Arc::clone(&self.sink),
            strip_app_data: self.strip_app_data,
        }
    }
}

impl<P, F> fmt::Debug for RecordingProvider<P, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RecordingProvider").finish_non_exhaustive()
    }
}

/// A stream that records its events and reports the exchange when it ends
/// or is dropped. Returned by [`RecordingProvider::chat_stream`].
pub struct RecordingStream<S, F>
where
    F: Fn(ChatExchange) + Send + Sync,
{
    inner: Pin<Box<S>>,
    transcript: StreamTranscript,
    /// The exchange waiting for its outcome; taken when reported.
    pending: Option<(ChatExchange, Arc<F>)>,
}

impl<S, F> RecordingStream<S, F>
where
    F: Fn(ChatExchange) + Send + Sync,
{
    fn report(&mut self) {
        let Some((mut exchange, sink)) = self.pending.take() else {
            return;
        };
        let transcript = std::mem::take(&mut self.transcript);
        let mut collector = StreamCollector::new();
        let outcome = transcript
            .events
            .iter()
            .try_for_each(|event| collector.push_ref(event).map(drop))
            .and_then(|()| match &transcript.terminal_error {
                Some(error) => Err(error.clone()),
                None => collector.finish(),
            });
        match outcome {
            Ok(response) => exchange.response = Some(recorded_response(response)),
            Err(error) => exchange.error = Some(error),
        }
        let mut transcript = transcript;
        transcript
            .events
            .iter_mut()
            .for_each(crate::chat::transcript::mark_warning_emitted);
        exchange.stream = Some(transcript);
        sink(exchange);
    }
}

impl<S, F> fmt::Debug for RecordingStream<S, F>
where
    F: Fn(ChatExchange) + Send + Sync,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RecordingStream")
            .field("recorded_events", &self.transcript.events.len())
            .field("reported", &self.pending.is_none())
            .finish_non_exhaustive()
    }
}

impl<S, F> Drop for RecordingStream<S, F>
where
    F: Fn(ChatExchange) + Send + Sync,
{
    fn drop(&mut self) {
        self.report();
    }
}

impl<S, F> Stream for RecordingStream<S, F>
where
    S: Stream<Item = Result<StreamEvent>>,
    F: Fn(ChatExchange) + Send + Sync,
{
    type Item = Result<StreamEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        let poll = this.inner.as_mut().poll_next(cx);
        match &poll {
            Poll::Ready(Some(item)) => {
                if this.pending.is_some() {
                    this.transcript.record(item.clone());
                }
                if item.is_err() {
                    this.report();
                }
            }
            Poll::Ready(None) => this.report(),
            Poll::Pending => {}
        }
        poll
    }
}

/// A response as it goes into the record: its warnings marked as emitted,
/// so a sink that replays the record through a wrapper built on
/// [`chat::prepare`](crate::adapter::chat::prepare) does not log them again.
fn recorded_response(mut response: ChatResponse) -> ChatResponse {
    response.warnings.iter_mut().for_each(Warning::mark_emitted);
    response
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::mock::{MockProvider, MockStreamEvent, MockStreamingProvider};
    use crate::{CapabilitySupport, ChatCapability};
    use crate::{ChatStreamExt, ProviderIdentityExt, StreamErrorKind};
    use futures_util::StreamExt;

    type Log = Arc<Mutex<Vec<ChatExchange>>>;

    fn recording<P>(inner: P) -> (RecordingProvider<P, impl Fn(ChatExchange)>, Log)
    where
        P: ChatProvider,
    {
        let log: Log = Arc::default();
        let sink = log.clone();
        let provider =
            RecordingProvider::new(inner, move |e: ChatExchange| sink.lock().unwrap().push(e));
        (provider, log)
    }

    fn request() -> ChatRequest {
        ChatRequest::new("m").user("hi")
    }

    #[tokio::test]
    async fn clones_share_one_sink() {
        let (provider, log) = recording(MockProvider::with_text("ok"));
        let provider = provider.strip_app_data();
        let clone = provider.clone();
        let _ = provider.chat(&request()).await;
        let _ = clone.chat(&request()).await;
        assert_eq!(log.lock().unwrap().len(), 2);
        assert!(clone.strip_app_data);
    }

    /// A provider that reports a warning nobody emitted.
    struct Raw;

    impl ProviderIdentity for Raw {
        fn provider_name(&self) -> &str {
            "raw"
        }

        fn reads_namespace(&self, namespace: &str) -> bool {
            namespace == "raw"
        }
    }

    impl ChatProvider for Raw {
        type Stream = crate::ChatStream;

        async fn chat(&self, _: &ChatRequest) -> Result<ChatResponse> {
            Ok(ChatResponse::default().warning(Warning::new(
                crate::WarningKind::Other,
                crate::Subject::Response,
                "raw",
            )))
        }

        async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
            Ok(Box::pin(self.chat(request).await?.into_stream()))
        }

        fn chat_capability(&self, _: &str, _: ChatCapability) -> CapabilitySupport {
            CapabilitySupport::Unknown
        }
    }

    /// Records replay what happened: their warnings are marked, the
    /// caller's are left alone.
    #[tokio::test]
    async fn recorded_warnings_are_marked_emitted() {
        let (provider, log) = recording(Raw);
        let response = provider.chat(&request()).await.unwrap();
        assert!(!crate::warning::was_emitted(&response.warnings[0]));
        let _ = provider
            .chat_stream(&request())
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();

        let log = log.lock().unwrap();
        for exchange in log.iter() {
            let response = exchange.response.as_ref().unwrap();
            assert!(crate::warning::was_emitted(&response.warnings[0]));
        }
        let transcript = log[1].stream.as_ref().unwrap();
        let warnings: Vec<_> = transcript
            .events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::Warning { warning } => Some(warning),
                _ => None,
            })
            .collect();
        assert_eq!(warnings.len(), 1);
        assert!(crate::warning::was_emitted(warnings[0]));
    }

    #[tokio::test]
    async fn unary_success_and_failure_are_recorded() {
        let mock = MockProvider::build(|b| b.text("hello").error(Error::overloaded("busy")));
        let (provider, log) = recording(mock);
        provider.chat(&request()).await.unwrap();
        provider.chat(&request()).await.unwrap_err();

        let log = log.lock().unwrap();
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].provider, "mock");
        assert_eq!(log[0].request, request());
        assert_eq!(
            log[0].response.as_ref().unwrap().text().as_deref(),
            Some("hello")
        );
        assert!(log[0].is_success() && log[0].stream.is_none());
        assert!(matches!(log[1].error, Some(Error::Overloaded { .. })));

        let value = serde_json::to_value(&log[1]).unwrap();
        assert_eq!(value["error"]["type"], "overloaded");
        let back: ChatExchange = serde_json::from_value(value).unwrap();
        assert_eq!(back.request, request());
    }

    #[tokio::test]
    async fn complete_streams_record_response_and_transcript() {
        let (provider, log) = recording(MockProvider::with_text("streamed"));
        assert!(provider.downcast_ref::<MockProvider>().is_some());
        let response = provider
            .chat_stream(&request())
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();

        let log = log.lock().unwrap();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].response.as_ref(), Some(&response));
        let transcript = log[0].stream.as_ref().unwrap();
        assert_eq!(transcript.to_collected().unwrap().response, response);
    }

    #[tokio::test]
    async fn failed_and_abandoned_streams_are_recorded() {
        let failing = MockStreamingProvider::with_stream([
            MockStreamEvent::from(StreamEvent::ResponseStart {
                id: None,
                model: None,
            }),
            MockStreamEvent::from(StreamEvent::TextDelta {
                index: 0,
                text: "par".into(),
            }),
            MockStreamEvent::from(Error::overloaded("busy")),
        ]);
        let (provider, log) = recording(failing);
        let items: Vec<_> = provider
            .chat_stream(&request())
            .await
            .unwrap()
            .collect()
            .await;
        assert!(items.last().unwrap().is_err());
        {
            let log = log.lock().unwrap();
            assert_eq!(log.len(), 1, "reported once, at the error");
            assert!(matches!(log[0].error, Some(Error::Overloaded { .. })));
            let partial = log[0].stream.as_ref().unwrap().to_collected().unwrap();
            assert_eq!(partial.response.text().as_deref(), Some("par"));
        }

        let (provider, log) = recording(MockProvider::with_text("never read"));
        let mut stream = provider.chat_stream(&request()).await.unwrap();
        stream.next().await;
        drop(stream);
        let log = log.lock().unwrap();
        assert_eq!(log.len(), 1);
        assert!(matches!(
            log[0].error,
            Some(Error::Stream {
                kind: StreamErrorKind::Incomplete,
                ..
            })
        ));
        assert_eq!(log[0].stream.as_ref().unwrap().events.len(), 1);
    }

    #[tokio::test]
    async fn stream_setup_errors_are_recorded() {
        let (provider, log) = recording(MockProvider::with_error(Error::auth("bad key")));
        assert!(provider.chat_stream(&request()).await.is_err());
        let log = log.lock().unwrap();
        assert!(matches!(log[0].error, Some(Error::Auth { .. })));
        assert!(log[0].stream.is_none());
    }

    fn request_with_app_data() -> ChatRequest {
        let mut request = ChatRequest::new("m")
            .system(crate::SystemPrompt::new("sys").with_app_data("myapp.id", "s1"))
            .message(crate::Message::user("hi").with_app_data("myapp.id", "u1"));
        request.messages[0]
            .app_data_insert("myapp.turn", &serde_json::json!({"n": 1}))
            .unwrap();
        request
    }

    #[tokio::test]
    async fn app_data_is_recorded_by_default() {
        let (provider, log) = recording(MockProvider::build(|b| b.text("hello").text("again")));
        let request = request_with_app_data();
        provider.chat(&request).await.unwrap();
        provider
            .chat_stream(&request)
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();

        let log = log.lock().unwrap();
        assert_eq!(log.len(), 2);
        for exchange in log.iter() {
            assert_eq!(exchange.request, request);
            let value = serde_json::to_value(exchange).unwrap();
            assert_eq!(
                value["request"]["messages"][0]["app_data"]["myapp.id"],
                "u1"
            );
            assert_eq!(value["request"]["system"][0]["app_data"]["myapp.id"], "s1");
            let back: ChatExchange = serde_json::from_value(value).unwrap();
            assert_eq!(back.request, request);
        }
    }

    #[tokio::test]
    async fn app_data_is_stripped_on_request() {
        let mock = MockProvider::build(|b| b.text("hello").text("again"));
        let log: Log = Arc::default();
        let sink = log.clone();
        let provider = RecordingProvider::new(mock.clone(), move |e: ChatExchange| {
            sink.lock().unwrap().push(e)
        })
        .strip_app_data();
        let request = request_with_app_data();
        provider.chat(&request).await.unwrap();
        provider
            .chat_stream(&request)
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();

        let log = log.lock().unwrap();
        assert_eq!(log.len(), 2);
        for exchange in log.iter() {
            assert!(exchange.request.system[0].app_data.is_empty());
            assert!(exchange.request.messages[0].app_data().is_empty());
            assert_eq!(exchange.request.messages, [crate::Message::user("hi")]);
        }
        assert_eq!(
            mock.requests(),
            [request.clone(), request],
            "the wrapped provider receives the request unchanged"
        );
    }

    #[tokio::test]
    async fn decisions_pass_through_unrecorded() {
        use crate::decision::{Answer, Question, YesNoAnswer};
        use crate::mock::MockDecisionProvider;

        let log: Log = Arc::default();
        let sink = log.clone();
        let provider = RecordingProvider::new(
            MockDecisionProvider::with_response(DecisionResponse::new([(
                "q",
                Answer::from(YesNoAnswer::new(0.9)),
            )])),
            move |e: ChatExchange| sink.lock().unwrap().push(e),
        );
        let request = DecisionRequest::new("d", "x").question("q", Question::yes_no("?"));
        assert!(
            provider
                .decide(&request)
                .await
                .unwrap()
                .yes_no("q")
                .unwrap()
                .is_yes()
        );
        assert!(log.lock().unwrap().is_empty());
    }
}
