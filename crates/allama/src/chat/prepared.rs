use std::borrow::Cow;
use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures_core::Stream;

use crate::adapter::chat::WireSupport;
use crate::chat::stream::{MetadataBuilder, UsageMetadataMode};
use crate::prepared::{PendingWarnings, Taken};
use crate::warning;
use crate::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, ChatStream,
    ChatValidation, ContentBlock, ContentPart, Degradations, Error, MediaSource, Message,
    ProviderIdentity, Result, StreamErrorKind, StreamEvent, Subject, Validation,
    ValidationCoverage, Warning, WarningKind,
};

/// A chat request checked against a provider's capability answers, on its
/// way to being encoded and sent.
///
/// [`prepare`] builds it: it resolves the request's
/// [`UnsupportedPolicy`](crate::UnsupportedPolicy), runs
/// [`preflight`](crate::adapter::chat::preflight) and records what the request relies
/// on that the provider answered `Unknown`. An adapter then encodes
/// [`request`](Self::request), reporting through
/// [`degradations_mut`](Self::degradations_mut) (or both at once with
/// [`parts_mut`](Self::parts_mut)), and ends in one of three ways:
///
/// - [`finish`](Self::finish) or [`finish_stream`](Self::finish_stream)
///   once the provider answered: the request-side warnings join the
///   response, ahead of its own, and are emitted as warning events
///   ([`warning::emit`](crate::warning::emit));
/// - [`into_validation`](Self::into_validation) for a dry run
///   ([`prepare_dry_run`]), which never emits;
/// - dropped, on an error path (a refusal while encoding, a rate limit).
///   Its warnings are then emitted only if the request reached the
///   transport: call [`mark_sent`](Self::mark_sent) immediately before
///   sending. A request refused before sending emits nothing; the error is
///   the only signal. Dropping never panics, so `?` needs no ceremony.
///
/// The split between encoding and sending is what makes validation a dry
/// run: an adapter's `validate_chat_request` runs the same encoding on a
/// [`prepare_dry_run`] request and reports
/// [`ValidationCoverage::Encoding`].
///
/// ```
/// use allama::adapter::chat::{self, Prepared};
/// use allama::{ChatProvider, ChatRequest, ChatResponse, ChatValidation, Result, ValidationCoverage};
///
/// /// Everything before sending: the dry run covers it.
/// fn encode(prepared: &mut Prepared<'_>) -> Result<String> {
///     let (request, degradations) = prepared.parts_mut();
///     if request.seed.is_some() {
///         let message = "demo has no seed; `seed` was not sent";
///         degradations.unsupported_hint(allama::RequestField::Seed, message)?;
///     }
///     Ok(format!("{{\"model\":\"{}\"}}", request.model))
/// }
///
/// async fn chat<P: ChatProvider>(provider: &P, request: &ChatRequest) -> Result<ChatResponse> {
///     let mut prepared = chat::prepare(provider, request)?;
///     let body = encode(&mut prepared)?;
///     prepared.mark_sent();
///     // Send `body` here; a `?` on the way out still reports the warnings.
///     # let _ = body;
///     let response = ChatResponse::default();
///     Ok(prepared.finish(response))
/// }
///
/// fn validate<P: ChatProvider>(provider: &P, request: &ChatRequest) -> Result<ChatValidation> {
///     let mut prepared = chat::prepare_dry_run(provider, request)?;
///     encode(&mut prepared)?;
///     Ok(prepared.into_validation(ValidationCoverage::Encoding))
/// }
/// ```
#[must_use = "a prepared request reports its warnings through `finish`, `finish_stream` or `into_validation`"]
#[derive(Debug)]
pub struct Prepared<'a> {
    request: Cow<'a, ChatRequest>,
    pending: PendingWarnings,
}

/// Check `request` against what `provider` reports for its model, for a call
/// that will send it (see [`Prepared`]).
///
/// It first refuses a [provider file](crate::MediaSource::ProviderFile)
/// that another provider stored ([`ProviderIdentity::reads_namespace`]
/// answers `false` for its origin) with [`Error::Unsupported`](crate::Error::Unsupported),
/// under every policy: its id means nothing to `provider` (ADR 0002).
///
/// The request's [`unsupported_policy`](ChatRequest::unsupported_policy)
/// applies, and when it is unset, [`UnsupportedPolicy::Warn`](crate::UnsupportedPolicy::Warn).
///
/// # Errors
///
/// What [`preflight`](crate::adapter::chat::preflight) returns.
pub fn prepare<'a, P: ChatProvider + ?Sized>(
    provider: &P,
    request: &'a ChatRequest,
) -> Result<Prepared<'a>> {
    Prepared::new(provider, request, false)
}

/// Check `request` as [`prepare`] does, for a dry run: the result never
/// emits warning events, not even when dropped. The default
/// [`validate_chat_request`](ChatProvider::validate_chat_request) and every
/// adapter's override use it.
///
/// # Errors
///
/// What [`preflight`](crate::adapter::chat::preflight) returns.
pub fn prepare_dry_run<'a, P: ChatProvider + ?Sized>(
    provider: &P,
    request: &'a ChatRequest,
) -> Result<Prepared<'a>> {
    Prepared::new(provider, request, true)
}

impl<'a> Prepared<'a> {
    fn new<P: ChatProvider + ?Sized>(
        provider: &P,
        request: &'a ChatRequest,
        dry_run: bool,
    ) -> Result<Self> {
        let degradations = Degradations::new(
            provider.provider_name().to_owned(),
            request.unsupported_policy.unwrap_or_default(),
        );
        refuse_foreign_files(provider, request, &degradations)?;
        let mut prepared = Self {
            request: Cow::Borrowed(request),
            pending: PendingWarnings::new(request.model.clone(), degradations, dry_run),
        };
        crate::adapter::chat::preflight(
            provider,
            &mut prepared.request,
            prepared.pending.degradations_mut(),
        )?;
        Ok(prepared)
    }

    /// The request to encode: the caller's, less what preflight removed.
    #[must_use]
    pub fn request(&self) -> &ChatRequest {
        &self.request
    }

    /// The helper to report through while encoding. Report indices relative
    /// to [`request`](Self::request); they are translated to the caller's.
    pub fn degradations_mut(&mut self) -> &mut Degradations {
        self.pending.degradations_mut()
    }

    /// The request and the helper together, so an encoder can walk the
    /// messages while it reports.
    pub fn parts_mut(&mut self) -> (&ChatRequest, &mut Degradations) {
        (&self.request, self.pending.degradations_mut())
    }

    /// Whether this is a dry run ([`prepare_dry_run`]).
    #[must_use]
    pub fn is_dry_run(&self) -> bool {
        self.pending.is_dry_run()
    }

    /// Run a second preflight against `other`'s capability answers, for an
    /// adapter layered on another provider (a toolkit dialect) that checks
    /// the request against both.
    ///
    /// Content indices compose: warnings recorded afterwards still name the
    /// caller's blocks. The capabilities `other` answers `Unknown` are not
    /// recorded, because validation reports the answers of the provider the
    /// caller asked, the one passed to [`prepare`].
    ///
    /// # Errors
    ///
    /// What [`preflight`](crate::adapter::chat::preflight) returns for `other`.
    pub fn preflight_with<P: ChatProvider + ?Sized>(&mut self, other: &P) -> Result<()> {
        let degradations = self.pending.degradations_mut();
        let recorded = degradations.unknown_len();
        let result = crate::adapter::chat::preflight(other, &mut self.request, degradations);
        degradations.truncate_unknown(recorded);
        result
    }

    /// Check what a streaming call relies on: records
    /// [`Streaming`](ChatCapability::Streaming) and
    /// [`NativeStreaming`](ChatCapability::NativeStreaming) when `provider`
    /// answers them `Unknown`. Call it while encoding a stream, so
    /// `validate_chat_stream_request` reports them.
    pub fn preflight_stream<P: ChatProvider + ?Sized>(&mut self, provider: &P) {
        for capability in [ChatCapability::Streaming, ChatCapability::NativeStreaming] {
            let support = provider.chat_capability(&self.request.model, capability);
            if support == CapabilitySupport::Unknown {
                self.pending
                    .degradations_mut()
                    .record_unknown(capability, Subject::Response);
            }
        }
    }

    /// Fit the request to what `provider`'s wire format carries: refuse what
    /// it cannot carry and must not lose, and remove the rest with the
    /// warnings ADR 0003 calls for (see [`WireSupport`] for the rules), so
    /// [`request`](Self::request) holds only what the encoder can send.
    ///
    /// Call it first while encoding, so the dry run covers it. `provider`
    /// tells its own `Other` content and provider tool blocks from another
    /// provider's ([`reads_namespace`](crate::ProviderIdentity::reads_namespace)).
    /// Like every warning recorded here, the warnings name the caller's
    /// indices.
    ///
    /// ```
    /// # #[cfg(feature = "mock")] {
    /// use allama::adapter::chat::{self, WireSupport};
    /// use allama::mock::MockProvider;
    /// use allama::{ChatRequest, ContentBlock, Message, RequestField, WarningKind};
    ///
    /// const WIRE: WireSupport = WireSupport::text().field(RequestField::System);
    ///
    /// let provider = MockProvider::empty();
    /// let request = ChatRequest::new("m")
    ///     .system("Be brief.")
    ///     .user("Hi")
    ///     .message(Message::assistant_blocks(vec![
    ///         ContentBlock::reasoning("The user greets me."),
    ///         ContentBlock::text("Hello!"),
    ///     ]))
    ///     .user("How are you?")
    ///     .temperature(0.2)
    ///     .seed(7);
    /// let mut prepared = chat::prepare(&provider, &request)?;
    /// prepared.degrade_to(&provider, &WIRE)?;
    ///
    /// // What is left is what the wire format carries.
    /// let request = prepared.request();
    /// assert_eq!((request.temperature, request.seed), (None, None));
    /// let warnings = prepared.into_validation(allama::ValidationCoverage::Encoding);
    /// let kinds: Vec<_> = warnings.warnings().iter().map(|w| w.kind).collect();
    /// assert_eq!(
    ///     kinds,
    ///     [WarningKind::Unsupported, WarningKind::Unsupported, WarningKind::Dropped]
    /// );
    /// # }
    /// # Ok::<(), allama::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] with the stage
    /// [`BeforeSending`](crate::UnsupportedStage::BeforeSending) for the
    /// first semantic field or content the wire format cannot carry, or,
    /// under [`UnsupportedPolicy::Error`](crate::UnsupportedPolicy::Error),
    /// the first hint.
    pub fn degrade_to<P: ProviderIdentity + ?Sized>(
        &mut self,
        provider: &P,
        wire: &WireSupport,
    ) -> Result<()> {
        let reads_namespace = |namespace: &str| provider.reads_namespace(namespace);
        crate::chat::wire::degrade(
            wire,
            &reads_namespace,
            &mut self.request,
            self.pending.degradations_mut(),
        )
    }

    /// Record that the request is about to reach the transport. Call it
    /// immediately before sending: from then on, dropping the request
    /// emits its warnings, because the call happened even if it fails.
    pub fn mark_sent(&mut self) {
        self.pending.mark_sent();
    }

    /// Join the request-side warnings to `response`, ahead of its own, and
    /// emit them as warning events.
    ///
    /// The response's own warnings are translated to the caller's indices,
    /// so an adapter that delegated the preflighted request to another
    /// provider reports the caller's blocks. Every warning not emitted yet
    /// is emitted now (not for a dry run), so a warning the inner provider
    /// emitted is not emitted again.
    ///
    /// An abnormal finish ([`Length`](crate::FinishReason::Length),
    /// [`ContentFilter`](crate::FinishReason::ContentFilter) or
    /// [`Refusal`](crate::FinishReason::Refusal)) is reported too, as a
    /// finish event (see [Finish events](crate::chat#finish-events)),
    /// once: the response remembers it was reported.
    ///
    /// A response without a [`provider`](ChatResponse::provider) is stamped
    /// with the provider this request was prepared for.
    pub fn finish(self, mut response: ChatResponse) -> ChatResponse {
        let inner = std::mem::take(&mut response.warnings);
        let taken = self.pending.take();
        if taken.emit && !response.finish_reported {
            response.finish_reported = finish_events::report(
                taken.degradations.provider(),
                &taken.model,
                response.finish_reason.as_ref(),
                response.finish_reason_raw.as_deref(),
                response.usage.as_ref(),
            );
        }
        if response.provider.is_none() {
            response.provider = Some(taken.degradations.provider().to_owned().into());
        }
        response.warnings = taken.merge(inner);
        response
    }

    /// Join the request-side warnings to `stream`: they follow its
    /// `ResponseStart`, or come first when it does not start with one.
    ///
    /// The stream's own warnings and response-level provider data that
    /// arrive before its first other event, such as a vendor frame read
    /// before the response started, are held until that event, as
    /// [`start_first`] does: they follow the `ResponseStart` (after the
    /// request-side warnings), or come right before the event when it is
    /// something else (an error, a stream without a start). An adapter
    /// yields them as they arrive and keeps no buffer of its own.
    ///
    /// The stream's own warnings are translated to the caller's indices. As
    /// in [`finish`](Self::finish), each warning is emitted (not for a dry
    /// run) immediately before its event is yielded; a stream dropped before
    /// its first poll emits nothing. An abnormal finish is reported as a
    /// finish event when its `ResponseMetadata` event is yielded, by the
    /// innermost layer that sees it.
    ///
    /// When the stream names no provider, its first `ResponseMetadata`
    /// event is stamped with the provider this request was prepared for (a
    /// stream without one gets one before its `ResponseStop`), so the
    /// collected response is stamped as [`finish`](Self::finish) stamps
    /// one. A provider the adapter names itself is kept.
    ///
    /// A stream that ends with `ResponseStop` (its terminal marker arrived)
    /// but reported no finish reason gets a warning of kind
    /// [`Other`](crate::WarningKind::Other) with the code
    /// [`MISSING_FINISH_REASON`] before its `ResponseStop`: the response is
    /// `Ok`, with no finish reason, so
    /// [`finished_normally`](ChatResponse::finished_normally) is `false`.
    ///
    /// A stream that ends without `ResponseStop` and without an error was
    /// cut short: the result ends with a retryable
    /// [`StreamErrorKind::Incomplete`](crate::StreamErrorKind::Incomplete)
    /// error, so a caller reading the raw stream never takes a quiet end
    /// for a complete response. An adapter whose decoder sees the stream end
    /// before its terminal marker fails it with its own `Incomplete` error,
    /// which says what was missing.
    ///
    /// The result ends after its first error, as every
    /// [`ChatStream`] does: the error is yielded after the events before it,
    /// and `stream` is dropped without being polled again, so an adapter
    /// that yields more after an error never reaches the caller with it.
    pub fn finish_stream<S>(self, stream: S) -> ChatStream
    where
        S: Stream<Item = Result<StreamEvent>> + Send + 'static,
    {
        Box::pin(FinishedStream {
            inner: Some(Box::pin(start_first(stream))),
            taken: self.pending.take(),
            injected: false,
            queue: VecDeque::new(),
            saw_finish_reason: false,
            saw_metadata: false,
            saw_provider: false,
            finish_reported: false,
            stopped: false,
        })
    }

    /// End a dry run of a layer that forwards the request to another
    /// provider: this layer's warnings, then `inner`'s (the forwarded
    /// request's own dry run, from its `validate_chat_request`), translated to the
    /// caller's indices, as [`finish`](Self::finish) joins them to a
    /// response. The capabilities either answered `Unknown` are listed, this
    /// layer's first; the coverage and fallback are `inner`'s. Nothing is
    /// emitted.
    pub fn finish_validation(self, inner: ChatValidation) -> ChatValidation {
        self.pending.take().merge_validation(inner)
    }

    /// End a dry run: the warnings the call would report and what the
    /// request relies on that the provider answered `Unknown`. Nothing is
    /// emitted.
    pub fn into_validation(self, coverage: ValidationCoverage) -> ChatValidation {
        let (warnings, unknown) = self.pending.take().into_validation_parts();
        Validation::from_parts(warnings, unknown, coverage)
    }
}

/// Hold a stream's pre-start events until its start: the `Warning` and
/// `ResponseProviderData` events that arrive before its first other event
/// follow that event when it is `ResponseStart`, and come right before it
/// otherwise (an error, or a stream without a `ResponseStart`), or at the
/// end of a stream that has no other event.
///
/// A stream reads `ResponseStart` first, but an adapter can learn of a
/// warning or a vendor frame (`allama_http::framing::VendorFrames`) before the
/// response started. [`Prepared::finish_stream`] applies this rule, so an adapter
/// built on it yields such events as they arrive and holds nothing itself.
/// Call it directly on a stream that does not go through `finish_stream`,
/// such as a decoder's stream in a test or a benchmark, so it reads as the
/// adapter's callers see it.
///
/// Like every chat stream, the result ends after its first error, and
/// `stream` is not polled again.
///
/// ```
/// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
/// use allama::adapter::chat::start_first;
/// use allama::{ProviderData, StreamEvent};
/// use futures_util::{StreamExt, stream};
///
/// let vendor_frame = StreamEvent::response_provider_data(ProviderData::new());
/// let events = stream::iter([
///     Ok(vendor_frame.clone()),
///     Ok(StreamEvent::response_start(None, None)),
///     Ok(StreamEvent::ResponseStop),
/// ]);
/// let events: Vec<_> = start_first(events).map(Result::unwrap).collect().await;
/// assert_eq!(
///     events,
///     [StreamEvent::response_start(None, None), vendor_frame, StreamEvent::ResponseStop]
/// );
/// # });
/// ```
pub fn start_first<S>(stream: S) -> impl Stream<Item = Result<StreamEvent>> + Send + 'static
where
    S: Stream<Item = Result<StreamEvent>> + Send + 'static,
{
    StartFirst {
        inner: Some(Box::pin(stream)),
        held: Some(Vec::new()),
        queue: VecDeque::new(),
    }
}

/// The [code](Warning::code) of the warning a stream gets when it ended
/// with its terminal marker but reported no finish reason (see
/// [`Prepared::finish_stream`]).
///
/// It is a property of the server, so it recurs on every call to the same
/// provider and model.
pub const MISSING_FINISH_REASON: &str = "stream.missing_finish_reason";

/// A stream with the request-side warnings joined in.
struct FinishedStream {
    /// The adapter's stream; dropped once it ended or failed, so it is
    /// never polled again.
    inner: Option<ChatStream>,
    taken: Taken,
    /// Whether the request-side warnings were queued.
    injected: bool,
    queue: VecDeque<Result<StreamEvent>>,
    /// Whether a finish reason was reported, or an inner layer already
    /// warned that none was.
    saw_finish_reason: bool,
    /// Whether a `ResponseMetadata` event arrived.
    saw_metadata: bool,
    /// Whether one named the provider, or was stamped with it.
    saw_provider: bool,
    /// Whether the finish was reported as a finish event, here or below.
    finish_reported: bool,
    /// Whether `ResponseStop` arrived.
    stopped: bool,
}

impl FinishedStream {
    fn emit(&self, warning: &mut Warning) {
        if self.taken.emit {
            warning::emit(
                self.taken.degradations.provider(),
                &self.taken.model,
                warning,
            );
        }
    }

    fn inject(&mut self) {
        self.injected = true;
        let mut warnings = self.taken.degradations.warnings().to_vec();
        for warning in &mut warnings {
            self.emit(warning);
        }
        self.queue
            .extend(warnings.into_iter().map(|w| Ok(StreamEvent::warning(w))));
    }

    /// Track the finish reason, and report an abnormal finish, which
    /// `reported_below` says an inner layer already reported.
    fn observe(&mut self, event: &StreamEvent, reported_below: bool) {
        if let StreamEvent::ResponseMetadata { provider, .. } = event {
            self.saw_metadata = true;
            self.saw_provider |= provider.is_some();
        }
        match event {
            StreamEvent::ResponseMetadata {
                finish_reason: Some(reason),
                finish_reason_raw,
                usage,
                usage_mode,
                ..
            } => {
                self.saw_finish_reason = true;
                if !self.finish_reported {
                    // Only a snapshot is the response's whole usage; a delta
                    // says nothing about how the output was spent.
                    let usage = usage
                        .as_ref()
                        .filter(|_| *usage_mode == UsageMetadataMode::Snapshot);
                    self.finish_reported = reported_below
                        || (self.taken.emit
                            && finish_events::report(
                                self.taken.degradations.provider(),
                                &self.taken.model,
                                Some(reason),
                                finish_reason_raw.as_deref(),
                                usage,
                            ));
                }
            }
            StreamEvent::Warning { warning }
                if warning.code.as_deref() == Some(MISSING_FINISH_REASON) =>
            {
                self.saw_finish_reason = true;
            }
            _ => {}
        }
    }

    /// Stop polling the adapter's stream, and drop it.
    fn end(&mut self) {
        self.inner = None;
    }

    /// The error for a stream that ended without `ResponseStop` or an
    /// error.
    fn cut_short(&self) -> Error {
        Error::stream(
            StreamErrorKind::Incomplete,
            "the stream ended before its terminal marker (no `ResponseStop`)",
        )
        .with_provider(self.taken.degradations.provider().to_owned())
    }

    /// The warning for a stream that ended with its terminal marker but no
    /// finish reason.
    fn missing_finish_reason(&self) -> StreamEvent {
        let mut warning = Warning::new(
            WarningKind::Other,
            Subject::Response,
            "the stream ended without a finish reason; the response has none",
        )
        .with_provider(self.taken.degradations.provider().to_owned())
        .with_code(MISSING_FINISH_REASON);
        self.emit(&mut warning);
        StreamEvent::warning(warning)
    }
}

impl Stream for FinishedStream {
    type Item = Result<StreamEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        loop {
            if let Some(item) = this.queue.pop_front() {
                return Poll::Ready(Some(item));
            }
            let Some(inner) = this.inner.as_mut() else {
                return Poll::Ready(None);
            };
            let layer = finish_events::Layer::enter();
            let poll = inner.as_mut().poll_next(cx);
            if let Poll::Ready(Some(Ok(event))) = &poll {
                this.observe(event, layer.reported_below());
                this.stopped |= matches!(event, StreamEvent::ResponseStop);
            }
            layer.exit(this.finish_reported);
            match poll {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    this.end();
                    if !this.injected {
                        this.inject();
                    }
                    if !this.stopped {
                        this.queue.push_back(Err(this.cut_short()));
                    }
                }
                // A chat stream ends after its first error: the error
                // follows what is queued, and the adapter's stream is never
                // polled again.
                Poll::Ready(Some(Err(error))) => {
                    this.end();
                    if !this.injected {
                        this.inject();
                    }
                    this.queue.push_back(Err(error));
                }
                Poll::Ready(Some(item)) => {
                    let item = match item {
                        Ok(StreamEvent::Warning { warning }) => {
                            let mut warning = this.taken.degradations.translate(warning);
                            this.emit(&mut warning);
                            Ok(StreamEvent::Warning { warning })
                        }
                        // The adapter named no provider: this one served
                        // the stream, as `finish` stamps a response.
                        Ok(mut event @ StreamEvent::ResponseMetadata { provider: None, .. })
                            if !this.saw_provider =>
                        {
                            this.saw_provider = true;
                            if let StreamEvent::ResponseMetadata { provider, .. } = &mut event {
                                *provider =
                                    Some(this.taken.degradations.provider().to_owned().into());
                            }
                            Ok(event)
                        }
                        Ok(StreamEvent::ResponseStop)
                            if !this.saw_finish_reason || !this.saw_metadata =>
                        {
                            if !this.injected {
                                this.inject();
                            }
                            if !this.saw_finish_reason {
                                this.saw_finish_reason = true;
                                let warning = this.missing_finish_reason();
                                this.queue.push_back(Ok(warning));
                            }
                            // No metadata named the provider: add one that
                            // does.
                            if !this.saw_metadata {
                                this.saw_metadata = true;
                                let provider = this.taken.degradations.provider().to_owned();
                                this.queue.push_back(Ok(MetadataBuilder::new()
                                    .with_provider(provider)
                                    .build()));
                            }
                            Ok(StreamEvent::ResponseStop)
                        }
                        other => other,
                    };
                    if this.injected {
                        this.queue.push_back(item);
                    } else if matches!(item, Ok(StreamEvent::ResponseStart { .. })) {
                        this.queue.push_back(item);
                        this.inject();
                    } else {
                        this.inject();
                        this.queue.push_back(item);
                    }
                }
            }
        }
    }
}

/// See [`start_first`].
struct StartFirst {
    /// The stream; dropped once it ended or failed.
    inner: Option<ChatStream>,
    /// The pre-start events so far; `None` once released.
    held: Option<Vec<StreamEvent>>,
    queue: VecDeque<Result<StreamEvent>>,
}

impl StartFirst {
    /// Queue the held events.
    fn release(&mut self) {
        if let Some(held) = self.held.take() {
            self.queue.extend(held.into_iter().map(Ok));
        }
    }
}

impl Stream for StartFirst {
    type Item = Result<StreamEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        loop {
            if let Some(item) = this.queue.pop_front() {
                return Poll::Ready(Some(item));
            }
            let Some(inner) = this.inner.as_mut() else {
                return Poll::Ready(None);
            };
            match inner.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    this.inner = None;
                    this.release();
                }
                Poll::Ready(Some(Ok(event))) => match &mut this.held {
                    None => return Poll::Ready(Some(Ok(event))),
                    Some(held) => match event {
                        StreamEvent::Warning { .. } | StreamEvent::ResponseProviderData { .. } => {
                            held.push(event);
                        }
                        StreamEvent::ResponseStart { .. } => {
                            this.queue.push_back(Ok(event));
                            this.release();
                        }
                        event => {
                            this.release();
                            this.queue.push_back(Ok(event));
                        }
                    },
                },
                Poll::Ready(Some(Err(error))) => {
                    this.inner = None;
                    this.release();
                    this.queue.push_back(Err(error));
                }
            }
        }
    }
}

/// Finish events: an abnormal finish reported as a `tracing` event.
/// Refuse a provider file another provider stored: its id resolves only at
/// its origin (ADR 0002), so it is never sent elsewhere.
fn refuse_foreign_files<P: ProviderIdentity + ?Sized>(
    provider: &P,
    request: &ChatRequest,
    d: &Degradations,
) -> Result<()> {
    for (index, message) in request.messages.iter().enumerate() {
        let sources: Vec<(usize, &MediaSource)> = match message {
            Message::Assistant(m) => m
                .content
                .iter()
                .enumerate()
                .filter_map(|(block, content)| match content {
                    ContentBlock::Image(image) => Some((block, &image.source)),
                    _ => None,
                })
                .collect(),
            _ => crate::chat::wire::message_parts(message)
                .iter()
                .enumerate()
                .filter_map(|(block, part)| match part {
                    ContentPart::Image(image) => Some((block, &image.source)),
                    ContentPart::File(file) => Some((block, &file.source)),
                    _ => None,
                })
                .collect(),
        };
        for (block, source) in sources {
            if let Some((origin, _)) = source.provider_file_id()
                && !provider.reads_namespace(origin)
            {
                return Err(d.reject(
                    Subject::Content {
                        message: index,
                        block: Some(block),
                    },
                    format!(
                        "a file stored by `{origin}` cannot be sent to `{}`: its id means \
                         nothing there; send the file inline or by URL",
                        provider.provider_name()
                    ),
                ));
            }
        }
    }
    Ok(())
}

mod finish_events {
    use crate::{FinishReason, Usage};

    /// The target of every finish event.
    #[cfg(feature = "warning-events")]
    pub(crate) const TARGET: &str = "allama::finish";

    /// Report `reason` as a finish event when it is abnormal enough to watch
    /// in production (the output token limit, the context window, a content
    /// filter, a refusal),
    /// returning whether it did.
    ///
    /// At the output token limit, the message also says how much of the
    /// output went to reasoning when `usage` reports it.
    pub(super) fn report(
        provider: &str,
        model: &str,
        reason: Option<&FinishReason>,
        raw: Option<&str>,
        usage: Option<&Usage>,
    ) -> bool {
        #[cfg(feature = "warning-events")]
        {
            let message = match reason {
                Some(FinishReason::Length) => length_message(usage),
                Some(FinishReason::ContextWindow) => {
                    "the output stopped at the model's context window".to_owned()
                }
                Some(FinishReason::ContentFilter) => {
                    "the output was stopped by a content filter".to_owned()
                }
                Some(FinishReason::Refusal) => "the model declined to answer".to_owned(),
                _ => return false,
            };
            tracing::event!(
                target: TARGET,
                tracing::Level::INFO,
                allama.finish.provider = provider,
                allama.finish.model = model,
                allama.finish.reason = reason.map(FinishReason::as_str),
                allama.finish.reason_raw = raw,
                "{message}",
            );
            true
        }
        #[cfg(not(feature = "warning-events"))]
        {
            let _ = (provider, model, reason, raw, usage);
            false
        }
    }

    /// The message of a finish at the output token limit.
    #[cfg(feature = "warning-events")]
    pub(super) fn length_message(usage: Option<&Usage>) -> String {
        let share = usage.and_then(|usage| {
            crate::usage::reasoning_share(usage.output_tokens, usage.reasoning_tokens)
        });
        match share {
            Some(share) => format!("the output stopped at the output token limit; {share}"),
            None => "the output stopped at the output token limit".to_owned(),
        }
    }

    #[cfg(feature = "warning-events")]
    std::thread_local! {
        /// Whether a stream layer below the one polling reported the finish.
        static REPORTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// One stream layer's poll of the stream it wraps. A layer built on
    /// `prepare` over another one polls it synchronously, so the inner
    /// layer's report reaches the outer one through a thread-local flag,
    /// and the finish is reported once, by the innermost layer.
    pub(super) struct Layer {
        #[cfg(feature = "warning-events")]
        outer: bool,
    }

    impl Layer {
        pub(super) fn enter() -> Self {
            Self {
                #[cfg(feature = "warning-events")]
                outer: REPORTED.with(|reported| reported.replace(false)),
            }
        }

        /// Whether a layer below reported the finish.
        pub(super) fn reported_below(&self) -> bool {
            #[cfg(feature = "warning-events")]
            {
                REPORTED.with(std::cell::Cell::get)
            }
            #[cfg(not(feature = "warning-events"))]
            false
        }

        /// Tell the layer above whether this one (or one below) reported.
        pub(super) fn exit(self, reported: bool) {
            #[cfg(feature = "warning-events")]
            REPORTED.with(|flag| flag.set(self.outer || reported));
            #[cfg(not(feature = "warning-events"))]
            let _ = reported;
        }
    }
}

#[cfg(test)]
mod tests {
    use futures_util::StreamExt;

    use super::*;
    use crate::mock::MockProvider;
    use crate::{
        CapabilitySupport, ContentBlock, FinishReason, Message, ReasoningEffort, RequestField,
        WarningKind,
    };

    /// A response or stream the adapter left without a provider is stamped
    /// with the one the request was prepared for.
    #[tokio::test]
    async fn finish_and_finish_stream_stamp_the_provider() {
        let provider = MockProvider::with_text("ok").with_provider_name("gateway");
        let request = ChatRequest::new("m").user("q");

        let unstamped = ChatResponse::new(vec![ContentBlock::text("ok")]);
        let response = prepare(&provider, &request).unwrap().finish(unstamped);
        assert_eq!(response.provider.as_deref(), Some("gateway"));
        let stamped = ChatResponse::new(vec![]).provider("inner");
        let response = prepare(&provider, &request).unwrap().finish(stamped);
        assert_eq!(response.provider.as_deref(), Some("inner"));

        // With metadata that names no provider, and with no metadata.
        for events in [
            vec![
                StreamEvent::response_start(None, None),
                finish(Some(FinishReason::Stop)),
                StreamEvent::ResponseStop,
            ],
            vec![
                StreamEvent::response_start(None, None),
                StreamEvent::ResponseStop,
            ],
        ] {
            let inner = futures_util::stream::iter(events.into_iter().map(Ok));
            let response = crate::ChatStreamExt::collect_response(
                prepare(&provider, &request).unwrap().finish_stream(inner),
            )
            .await
            .unwrap();
            assert_eq!(response.provider.as_deref(), Some("gateway"));
        }
        let inner = futures_util::stream::iter([
            Ok(StreamEvent::response_start(None, None)),
            Ok(MetadataBuilder::new()
                .with_finish_reason(FinishReason::Stop)
                .with_provider("inner")
                .build()),
            Ok(StreamEvent::ResponseStop),
        ]);
        let response = crate::ChatStreamExt::collect_response(
            prepare(&provider, &request).unwrap().finish_stream(inner),
        )
        .await
        .unwrap();
        assert_eq!(response.provider.as_deref(), Some("inner"));
    }

    /// A file another provider stores is refused before sending, under
    /// every policy; one the provider reads goes on to the encoder.
    #[test]
    fn foreign_provider_files_are_refused() {
        let provider = MockProvider::with_text("ok");
        let file = |origin: &str| {
            ChatRequest::new("m").message(Message::user_parts(vec![
                ContentPart::text("Summarize it."),
                ContentPart::File(crate::File::provider_file(
                    "application/pdf",
                    origin,
                    "file-1",
                )),
            ]))
        };
        let err = prepare(&provider, &file("anthropic")).unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
        assert!(err.to_string().contains("`anthropic`"), "{err}");
        assert!(prepare_dry_run(&provider, &file("anthropic")).is_err());

        let image = ChatRequest::new("m")
            .user("Draw it.")
            .message(Message::assistant_blocks(vec![ContentBlock::Image(
                crate::Image::provider_file("openai", "file-2"),
            )]))
            .user("Again.");
        assert!(prepare(&provider, &image).is_err());

        let own = provider.provider_name().to_owned();
        assert!(prepare(&provider, &file(&own)).is_ok());
    }

    /// A model without reasoning controls or reasoning replay.
    fn plain() -> MockProvider {
        MockProvider::with_text("ok")
            .with_chat_capability(
                ChatCapability::ReasoningConfig,
                CapabilitySupport::Unsupported,
            )
            .with_chat_capability(
                ChatCapability::ReasoningReplay,
                CapabilitySupport::Unsupported,
            )
    }

    /// A request preflight changes: a reasoning hint, and a reasoning block
    /// ahead of text in message 1.
    fn request() -> ChatRequest {
        ChatRequest::new("m")
            .user("q")
            .message(Message::assistant_blocks(vec![
                ContentBlock::reasoning("think"),
                ContentBlock::text("a"),
            ]))
            .user("q2")
            .reasoning(ReasoningEffort::High)
    }

    fn inner_warning() -> Warning {
        // Block 0 of the preflighted message 1: block 1 of the caller's.
        Warning::new(
            WarningKind::Other,
            Subject::Content {
                message: 1,
                block: Some(0),
            },
            "inner",
        )
    }

    #[test]
    fn finish_puts_request_warnings_first_and_translates_the_responses() {
        let request = request();
        let prepared = prepare(&plain(), &request).unwrap();
        assert!(prepared.request().reasoning.is_none());
        let mut response = ChatResponse::default();
        response.warnings.push(inner_warning());
        let response = prepared.finish(response);
        let subjects: Vec<_> = response.warnings.iter().map(|w| &w.subject).collect();
        assert_eq!(
            subjects,
            [
                &Subject::Field {
                    field: RequestField::Reasoning
                },
                &Subject::Content {
                    message: 1,
                    block: Some(0)
                },
                &Subject::Content {
                    message: 1,
                    block: Some(1)
                },
            ]
        );
    }

    async fn collect(stream: ChatStream) -> Vec<StreamEvent> {
        stream.map(|event| event.unwrap()).collect().await
    }

    fn names(events: &[StreamEvent]) -> Vec<&'static str> {
        events.iter().map(StreamEvent::name).collect()
    }

    #[tokio::test]
    async fn finish_stream_puts_request_warnings_after_the_start() {
        let request = ChatRequest::new("m")
            .user("q")
            .reasoning(ReasoningEffort::High);
        let prepared = prepare(&plain(), &request).unwrap();
        let inner = futures_util::stream::iter([
            Ok(StreamEvent::response_start(None, None)),
            Ok(StreamEvent::warning(inner_warning())),
            Ok(finish(Some(FinishReason::Stop))),
            Ok(StreamEvent::ResponseStop),
        ]);
        let events = collect(prepared.finish_stream(inner)).await;
        assert_eq!(
            names(&events),
            [
                "response_start",
                "warning",
                "warning",
                "response_metadata",
                "response_stop"
            ]
        );

        // Without a start, they come first; with no events, they still come
        // (before the error for a stream cut short).
        for inner in [
            vec![Ok(StreamEvent::ResponseStop)],
            Vec::<Result<StreamEvent>>::new(),
        ] {
            let prepared = prepare(&plain(), &request).unwrap();
            let items: Vec<_> = prepared
                .finish_stream(futures_util::stream::iter(inner))
                .collect()
                .await;
            assert!(
                matches!(items[0], Ok(StreamEvent::Warning { .. })),
                "{items:?}"
            );
        }
    }

    /// A stream that counts the polls it receives.
    struct Counting {
        items: VecDeque<Result<StreamEvent>>,
        polls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Stream for Counting {
        type Item = Result<StreamEvent>;

        fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            self.polls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Poll::Ready(self.items.pop_front())
        }
    }

    #[tokio::test]
    async fn a_finished_stream_ends_after_its_first_error() {
        let request = ChatRequest::new("m").user("q");
        let polls = std::sync::Arc::default();
        let inner = Counting {
            items: VecDeque::from([
                Ok(StreamEvent::response_start(None, None)),
                Err(crate::Error::stream_decode("broken")),
                Ok(StreamEvent::text_delta(0, "after")),
            ]),
            polls: std::sync::Arc::clone(&polls),
        };
        let items: Vec<_> = prepare(&plain(), &request)
            .unwrap()
            .finish_stream(inner)
            .collect()
            .await;
        assert_eq!(items.len(), 2, "{items:?}");
        assert!(matches!(items[0], Ok(StreamEvent::ResponseStart { .. })));
        assert!(items[1].is_err());
        assert_eq!(polls.load(std::sync::atomic::Ordering::SeqCst), 2);

        // An error before anything else still comes after the request-side
        // warnings.
        let request = request.reasoning(ReasoningEffort::High);
        let inner = futures_util::stream::iter([
            Err(crate::Error::stream_decode("broken")),
            Ok(StreamEvent::ResponseStop),
        ]);
        let items: Vec<_> = prepare(&plain(), &request)
            .unwrap()
            .finish_stream(inner)
            .collect()
            .await;
        assert_eq!(items.len(), 2, "{items:?}");
        assert!(matches!(items[0], Ok(StreamEvent::Warning { .. })));
        assert!(items[1].is_err());
    }

    /// A stream that ends without `ResponseStop` and without an error was
    /// cut short: its raw consumer gets a terminal `Incomplete` error, not
    /// a quiet end.
    #[tokio::test]
    async fn a_quiet_end_is_an_incomplete_stream_error() {
        let request = ChatRequest::new("m").user("q");
        for inner in [
            vec![
                Ok(StreamEvent::response_start(None, None)),
                Ok(StreamEvent::text_delta(0, "par")),
            ],
            Vec::new(),
        ] {
            let items: Vec<_> = prepare(&plain(), &request)
                .unwrap()
                .finish_stream(futures_util::stream::iter(inner))
                .collect()
                .await;
            match items.last() {
                Some(Err(crate::Error::Stream {
                    kind: crate::StreamErrorKind::Incomplete,
                    ..
                })) => {}
                other => panic!("expected an Incomplete stream error, got {other:?}"),
            }
            let error = items.last().unwrap().as_ref().unwrap_err();
            assert!(error.is_retryable());
            let provider = error.context().and_then(|c| c.provider.as_deref());
            assert_eq!(provider, Some("mock"));
        }

        // An outer layer passes the inner layer's error on, adding none.
        let inner = prepare(&plain(), &request)
            .unwrap()
            .finish_stream(futures_util::stream::iter([Ok(
                StreamEvent::response_start(None, None),
            )]));
        let items: Vec<_> = prepare(&plain(), &request)
            .unwrap()
            .finish_stream(inner)
            .collect()
            .await;
        assert_eq!(items.iter().filter(|item| item.is_err()).count(), 1);

        // A stream that ends with its marker, or with an error, gets none.
        let items: Vec<_> = prepare(&plain(), &request)
            .unwrap()
            .finish_stream(futures_util::stream::iter([
                Ok(StreamEvent::response_start(None, None)),
                Ok(finish(Some(FinishReason::Stop))),
                Ok(StreamEvent::ResponseStop),
            ]))
            .collect()
            .await;
        assert!(items.iter().all(Result::is_ok), "{items:?}");
        let items: Vec<_> = prepare(&plain(), &request)
            .unwrap()
            .finish_stream(futures_util::stream::iter([Err(
                crate::Error::stream_decode("broken"),
            )]))
            .collect()
            .await;
        assert_eq!(items.len(), 1);
        assert!(matches!(
            &items[0],
            Err(crate::Error::Stream {
                kind: crate::StreamErrorKind::Decode,
                ..
            })
        ));
    }

    fn vendor_frame(key: &str) -> StreamEvent {
        StreamEvent::response_provider_data(crate::ProviderData::new().with_raw(
            "vendor",
            key,
            serde_json::json!(true),
        ))
    }

    fn stream_warning(message: &str) -> StreamEvent {
        StreamEvent::warning(Warning::new(WarningKind::Other, Subject::Response, message))
    }

    /// The events of `start_first` over `inner`, as names, with the
    /// warnings' messages and the frames' keys.
    async fn held(inner: Vec<Result<StreamEvent>>) -> Vec<String> {
        start_first(futures_util::stream::iter(inner))
            .map(|item| match item {
                Ok(StreamEvent::Warning { warning }) => format!("warning {}", warning.message),
                Ok(StreamEvent::ResponseProviderData { provider_data }) => format!(
                    "frame {}",
                    provider_data
                        .namespace("vendor")
                        .unwrap()
                        .keys()
                        .next()
                        .unwrap()
                ),
                Ok(event) => event.name().to_owned(),
                Err(_) => "error".to_owned(),
            })
            .collect()
            .await
    }

    #[tokio::test]
    async fn pre_start_events_are_held_until_the_start() {
        // With a start: they follow it, in order; later ones pass.
        let events = held(vec![
            Ok(vendor_frame("a")),
            Ok(stream_warning("w")),
            Ok(StreamEvent::response_start(None, None)),
            Ok(StreamEvent::metadata()),
            Ok(vendor_frame("b")),
            Ok(StreamEvent::ResponseStop),
        ])
        .await;
        assert_eq!(
            events,
            [
                "response_start",
                "frame a",
                "warning w",
                "response_metadata",
                "frame b",
                "response_stop"
            ]
        );

        // Without one: they come right before the first other event, or
        // at the end.
        let events = held(vec![
            Ok(vendor_frame("a")),
            Ok(StreamEvent::metadata()),
            Ok(StreamEvent::ResponseStop),
        ])
        .await;
        assert_eq!(events, ["frame a", "response_metadata", "response_stop"]);
        let events = held(vec![Ok(vendor_frame("a")), Ok(stream_warning("w"))]).await;
        assert_eq!(events, ["frame a", "warning w"]);

        // With an early error: they come before it, and the stream ends.
        let events = held(vec![
            Ok(vendor_frame("a")),
            Err(crate::Error::stream_decode("broken")),
            Ok(StreamEvent::response_start(None, None)),
        ])
        .await;
        assert_eq!(events, ["frame a", "error"]);
    }

    #[tokio::test]
    async fn finish_stream_puts_held_events_after_the_request_warnings() {
        let request = ChatRequest::new("m")
            .user("q")
            .reasoning(ReasoningEffort::High);
        let finished = |inner: Vec<Result<StreamEvent>>| {
            let prepared = prepare(&plain(), &request).unwrap();
            prepared.finish_stream(futures_util::stream::iter(inner))
        };
        let messages = |events: Vec<Result<StreamEvent>>| -> Vec<String> {
            events
                .into_iter()
                .map(|item| match item {
                    Ok(StreamEvent::Warning { warning }) if warning.message == "w" => {
                        "stream warning".to_owned()
                    }
                    Ok(event) => event.name().to_owned(),
                    Err(_) => "error".to_owned(),
                })
                .collect()
        };

        let events = finished(vec![
            Ok(stream_warning("w")),
            Ok(vendor_frame("a")),
            Ok(StreamEvent::response_start(None, None)),
            Ok(finish(Some(FinishReason::Stop))),
            Ok(StreamEvent::ResponseStop),
        ])
        .collect()
        .await;
        assert_eq!(
            messages(events),
            [
                "response_start",
                "warning",
                "stream warning",
                "response_provider_data",
                "response_metadata",
                "response_stop"
            ]
        );

        let events = finished(vec![
            Ok(stream_warning("w")),
            Err(crate::Error::stream_decode("broken")),
        ])
        .collect()
        .await;
        assert_eq!(messages(events), ["warning", "stream warning", "error"]);
    }

    #[tokio::test]
    async fn stream_warnings_name_the_callers_blocks() {
        let request = request();
        let prepared = prepare(&plain(), &request).unwrap();
        let inner = futures_util::stream::iter([
            Ok(StreamEvent::warning(inner_warning())),
            Ok(StreamEvent::ResponseStop),
        ]);
        let events = collect(prepared.finish_stream(inner)).await;
        let Some(warning) = events.iter().find_map(|event| match event {
            StreamEvent::Warning { warning } if warning.message == "inner" => Some(warning),
            _ => None,
        }) else {
            panic!("the inner warning expected: {events:?}");
        };
        assert_eq!(
            warning.subject,
            Subject::Content {
                message: 1,
                block: Some(1)
            }
        );
    }

    #[test]
    fn a_dry_run_reports_what_the_call_would() {
        let request = request();
        let validation = prepare_dry_run(&plain(), &request)
            .unwrap()
            .into_validation(ValidationCoverage::Encoding);
        let call = prepare(&plain(), &request)
            .unwrap()
            .finish(ChatResponse::default());
        assert_eq!(validation.warnings(), call.warnings);
        assert_eq!(validation.coverage(), ValidationCoverage::Encoding);
        assert!(prepare_dry_run(&plain(), &request).unwrap().is_dry_run());
    }

    fn finish(reason: Option<FinishReason>) -> StreamEvent {
        let mut metadata = StreamEvent::metadata();
        if let StreamEvent::ResponseMetadata { finish_reason, .. } = &mut metadata {
            *finish_reason = reason;
        }
        metadata
    }

    fn missing_finish_warnings(events: &[StreamEvent]) -> Vec<usize> {
        events
            .iter()
            .enumerate()
            .filter(|(_, event)| {
                matches!(event, StreamEvent::Warning { warning }
                    if warning.code.as_deref() == Some(MISSING_FINISH_REASON))
            })
            .map(|(index, _)| index)
            .collect()
    }

    #[tokio::test]
    async fn a_terminated_stream_without_a_finish_reason_is_warned_about() {
        let request = ChatRequest::new("m").user("q");
        let finished = |inner: Vec<StreamEvent>| {
            let prepared = prepare(&plain(), &request).unwrap();
            prepared.finish_stream(futures_util::stream::iter(inner.into_iter().map(Ok)))
        };

        let events = collect(finished(vec![
            StreamEvent::response_start(None, None),
            finish(None),
            StreamEvent::ResponseStop,
        ]))
        .await;
        assert_eq!(missing_finish_warnings(&events), [2]);
        let StreamEvent::Warning { warning } = &events[2] else {
            unreachable!()
        };
        assert_eq!(warning.kind, WarningKind::Other);
        assert_eq!(warning.subject, Subject::Response);
        assert_eq!(names(&events).last(), Some(&"response_stop"));

        // A reason, at any point before the end, is enough.
        let events = collect(finished(vec![
            StreamEvent::response_start(None, None),
            finish(Some(FinishReason::Length)),
            finish(None),
            StreamEvent::ResponseStop,
        ]))
        .await;
        assert!(missing_finish_warnings(&events).is_empty());

        // A stream that never ended is cut short: no warning, an error.
        let items: Vec<_> = finished(vec![StreamEvent::response_start(None, None)])
            .collect()
            .await;
        assert!(items.last().unwrap().is_err());
        let events: Vec<_> = items.into_iter().filter_map(Result::ok).collect();
        assert!(missing_finish_warnings(&events).is_empty());

        // An outer layer passes the inner layer's warning on, adding none.
        let inner = finished(vec![
            StreamEvent::response_start(None, None),
            StreamEvent::ResponseStop,
        ]);
        let outer = prepare(&plain(), &request).unwrap().finish_stream(inner);
        let response = crate::ChatStreamExt::collect_response(outer).await.unwrap();
        assert_eq!(response.finish_reason, None);
        assert!(!response.finished_normally());
        let warned = response
            .warnings
            .iter()
            .filter(|w| w.code.as_deref() == Some(MISSING_FINISH_REASON))
            .count();
        assert_eq!(warned, 1);
    }

    #[test]
    fn a_second_preflight_records_no_unknown_answers() {
        let request = ChatRequest::new("m")
            .user("q")
            .reasoning(ReasoningEffort::High);
        let known = MockProvider::empty().with_chat_capability(
            ChatCapability::ReasoningConfig,
            CapabilitySupport::Supported,
        );
        let mut prepared = prepare_dry_run(&known, &request).unwrap();
        prepared.preflight_with(&MockProvider::empty()).unwrap();
        let validation = prepared.into_validation(ValidationCoverage::Capabilities);
        assert!(validation.relied_on_unknown().is_empty());
    }

    #[cfg(feature = "warning-events")]
    mod events {
        use super::*;
        use crate::mock::{ChatResponseBuilder, MockStreamingProvider};
        use crate::warning::capture::warning_events;
        use crate::{UnsupportedPolicy, Usage};

        #[test]
        fn dry_runs_never_emit() {
            let request = request();
            let ((), events) = warning_events(async {
                let validation = plain().validate_chat_request(&request).unwrap();
                assert_eq!(validation.warnings().len(), 2);
                // A refused dry run, after a hint warning.
                let strict = request.clone().unsupported_policy(UnsupportedPolicy::Error);
                assert!(plain().validate_chat_request(&strict).is_err());
                // Dropped, even after `mark_sent`.
                let mut prepared = prepare_dry_run(&plain(), &request).unwrap();
                prepared.mark_sent();
                drop(prepared);
            });
            assert!(events.is_empty(), "{events:?}");
        }

        #[test]
        fn a_dropped_request_emits_only_once_sent() {
            let request = ChatRequest::new("m")
                .user("q")
                .reasoning(ReasoningEffort::High);
            let ((), events) = warning_events(async {
                // Refused before sending (here: dropped unsent): nothing.
                drop(prepare(&plain(), &request).unwrap());
                // Failed after sending, such as a rate limit: the hint was
                // dropped from a call that happened.
                let mut prepared = prepare(&plain(), &request).unwrap();
                prepared.mark_sent();
                drop(prepared);
            });
            assert_eq!(
                events,
                ["reasoning configuration is not supported by `mock`; `reasoning` was not sent"]
            );
        }

        #[test]
        fn calls_emit_each_warning_once() {
            let request = ChatRequest::new("m")
                .user("q")
                .reasoning(ReasoningEffort::High);
            let (warnings, events) = warning_events(async {
                let response = plain().chat(&request).await.unwrap();
                response.warnings.len()
            });
            assert_eq!(warnings, 1);
            assert_eq!(events.len(), 1, "{events:?}");
        }

        /// A streaming wrapper built on `prepare` over an adapter: the
        /// adapter emits its warnings, the wrapper only its own.
        struct Wrapper<P>(P);

        impl<P: crate::ProviderIdentity> crate::ProviderIdentity for Wrapper<P> {
            fn provider_name(&self) -> &str {
                self.0.provider_name()
            }

            fn reads_namespace(&self, namespace: &str) -> bool {
                self.0.reads_namespace(namespace)
            }
        }

        impl<P: ChatProvider> ChatProvider for Wrapper<P> {
            type Stream = ChatStream;

            async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
                let mut prepared = prepare(self, request)?;
                prepared.degradations_mut().warn(Warning::new(
                    WarningKind::Other,
                    Subject::Response,
                    "wrapper",
                ));
                prepared.mark_sent();
                let response = self.0.chat(prepared.request()).await?;
                Ok(prepared.finish(response))
            }

            async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
                let mut prepared = prepare(self, request)?;
                prepared.degradations_mut().warn(Warning::new(
                    WarningKind::Other,
                    Subject::Response,
                    "wrapper",
                ));
                prepared.mark_sent();
                let stream = self.0.chat_stream(prepared.request()).await?;
                Ok(prepared.finish_stream(stream))
            }

            fn chat_capability(
                &self,
                model: &str,
                capability: ChatCapability,
            ) -> CapabilitySupport {
                self.0.chat_capability(model, capability)
            }
        }

        /// Run `future` and return the finish events it emitted, as
        /// `reason: message`.
        fn finish_events<T>(future: impl std::future::Future<Output = T>) -> (T, Vec<String>) {
            let (output, events) = crate::warning::capture::captured(future);
            let finishes = events
                .into_iter()
                .filter(|event| event.target == super::super::finish_events::TARGET)
                .inspect(|event| assert_eq!(event.level, tracing::Level::INFO))
                .map(|event| {
                    format!(
                        "{}: {}",
                        event.finish_reason.unwrap_or_default(),
                        event.message
                    )
                })
                .collect();
            (output, finishes)
        }

        #[test]
        fn abnormal_finishes_are_reported_once_per_call() {
            let request = ChatRequest::new("m").user("q");
            let response = |reason| {
                ChatResponseBuilder::new()
                    .text("ok")
                    .finish_reason(reason)
                    .build()
            };
            let ((), events) = finish_events(async {
                for reason in [
                    FinishReason::Length,
                    FinishReason::ContentFilter,
                    FinishReason::Refusal,
                    // Not reported: a normal finish, and reasons allama does
                    // not know.
                    FinishReason::Stop,
                    FinishReason::ToolCalls,
                    FinishReason::Other("malformed_function_call".into()),
                ] {
                    // A layer built on `prepare` over the adapter: reported
                    // by the adapter alone, on both paths.
                    let unary = MockProvider::with_response(response(reason.clone()));
                    Wrapper(unary).chat(&request).await.unwrap();
                    let streaming = MockStreamingProvider::from_responses([response(reason)]);
                    let stream = Wrapper(streaming).chat_stream(&request).await.unwrap();
                    crate::ChatStreamExt::collect_response(stream)
                        .await
                        .unwrap();
                }
                // A dry run reports nothing.
                let validation = MockProvider::with_response(response(FinishReason::Length))
                    .validate_chat_request(&request)
                    .unwrap();
                assert!(validation.warnings().is_empty());
            });
            assert_eq!(
                events,
                [
                    "length: the output stopped at the output token limit",
                    "length: the output stopped at the output token limit",
                    "content_filter: the output was stopped by a content filter",
                    "content_filter: the output was stopped by a content filter",
                    "refusal: the model declined to answer",
                    "refusal: the model declined to answer",
                ]
            );
        }

        #[test]
        fn a_finish_at_the_limit_says_how_much_went_to_reasoning() {
            let request = ChatRequest::new("m").user("q");
            let response = |usage: Usage| {
                ChatResponseBuilder::new()
                    .finish_reason(FinishReason::Length)
                    .usage_value(usage)
                    .build()
            };
            let reasoned = Usage::new().output_tokens(400).reasoning_tokens(400);
            let ((), events) = finish_events(async {
                for usage in [reasoned, Usage::new().output_tokens(400)] {
                    let unary = MockProvider::with_response(response(usage.clone()));
                    Wrapper(unary).chat(&request).await.unwrap();
                    let streaming = MockStreamingProvider::from_responses([response(usage)]);
                    let stream = Wrapper(streaming).chat_stream(&request).await.unwrap();
                    crate::ChatStreamExt::collect_response(stream)
                        .await
                        .unwrap();
                }
            });
            let reasoning = "length: the output stopped at the output token limit; 400 of 400 output tokens went to reasoning";
            let plain = "length: the output stopped at the output token limit";
            assert_eq!(events, [reasoning, reasoning, plain, plain]);
        }

        #[test]
        fn layers_emit_each_warning_once() {
            let request = ChatRequest::new("m").user("q").seed(1);
            let adapter = || {
                MockStreamingProvider::from_responses([ChatResponseBuilder::new()
                    .text("ok")
                    .warning(Warning::unsupported(RequestField::Seed, "seed ignored"))
                    .build()])
            };
            let (collected, events) = warning_events(async {
                let stream = Wrapper(adapter()).chat_stream(&request).await.unwrap();
                crate::ChatStreamExt::collect_response(stream)
                    .await
                    .unwrap()
            });
            assert_eq!(collected.warnings.len(), 2);
            let mut events = events;
            events.sort();
            assert_eq!(events, ["seed ignored", "wrapper"]);

            let unary = MockProvider::with_response(
                ChatResponseBuilder::new()
                    .text("ok")
                    .warning(Warning::unsupported(RequestField::Seed, "seed ignored"))
                    .build(),
            );
            let (_, events) =
                warning_events(async { Wrapper(unary).chat(&request).await.unwrap() });
            assert_eq!(events.len(), 2, "{events:?}");
        }
    }
}
