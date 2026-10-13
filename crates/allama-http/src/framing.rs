//! Streaming for any wire framing: the chat stream driver and vendor
//! frames.
//!
//! A streamed response arrives in *frames* of its wire framing: SSE events
//! ([`sse`](crate::sse)), NDJSON lines ([`ndjson`](crate::ndjson)), or the
//! messages of another framing an adapter frames itself. What turns frames
//! into chat stream events is the same loop for all of them, and so are the
//! rules a hand-written loop gets wrong, so they live here:
//!
//! - **The driver.** [`FramedChatStream`] runs an adapter's
//!   [`ChatFrameDecoder`] over a stream of frames: it ends the stream after
//!   the first error, runs the decoder's end-of-body or failure hook once,
//!   classifies a failed read (a stream timeout stays a timeout), and never
//!   polls the body after the stream ended.
//! - **Vendor frames.** [`VendorFrames`] surfaces frames of a type an
//!   adapter does not model (`event: search_results`, an NDJSON line of an
//!   unknown `type`) as response-level provider data under the adapter's
//!   namespace, with one warning per frame type, so they are never decoded
//!   as the adapter's own and never dropped.
//!
//! Framing the bytes stays with a framer ([`SseStream`](crate::sse::SseStream),
//! [`NdjsonStream`](crate::ndjson::NdjsonStream), or your own stream of
//! frames), and decoding the frames' data with each adapter: the helpers
//! frame, drive and surface, they do not interpret.

use std::borrow::Cow;
use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::pin::Pin;
use std::task::{Context, Poll, ready};

use allama::{ProviderData, Result, StreamEvent, Subject, Warning, WarningKind};
use futures_core::{Stream, TryStream};
use serde_json::Value;

/// An adapter's decoder for one streamed response: frames of type `F` in
/// (an [`SseEvent`](crate::sse::SseEvent), an NDJSON line), chat stream
/// events out. [`FramedChatStream`] drives it.
///
/// The decoder holds what only the adapter knows: its wire format, when its
/// response starts and ends, and its error classification. The driver holds
/// the loop every streaming adapter needs, and its guarantees:
///
/// - each frame goes to [`frame`](Self::frame), in order;
/// - the stream ends after the first error, from the decoder or from the
///   body: the items after it are dropped;
/// - when the body ends, [`end`](Self::end) runs; when reading or framing
///   it fails, [`fail`](Self::fail) runs, with the failure already
///   classified. At most one of them runs, once, and only while the decoder
///   is not done (it has yielded no error and [`is_done`](Self::is_done) is
///   `false`);
/// - the stream ends once the decoder is done, without waiting for the
///   body to close, and the body is never polled after the stream ended.
///
/// A decoder yields warnings and response-level provider data as they
/// arrive, even before its `ResponseStart`:
/// [`Prepared::finish_stream`](allama::adapter::chat::Prepared::finish_stream) puts
/// them after the start ([`start_first`](allama::adapter::chat::start_first)).
///
/// An SSE adapter implements `ChatFrameDecoder<SseEvent>`, and
/// [`SseChatStream`](crate::sse::SseChatStream) drives it.
///
/// Each method returns the items of one frame as a `Vec`, and this shape is
/// deliberate and stable. An empty `Vec` does not allocate, and a frame that
/// yields events was decoded from JSON anyway, so an `out: &mut Vec<_>` sink
/// parameter would save one small allocation per frame and cost every decoder
/// a harder signature: early returns become pushes, and a decoder that
/// forgets to push loses an error silently.
///
/// ```
/// use allama::{Error, Result, StreamEvent, StreamErrorKind};
/// use allama_http::TransportError;
/// use allama_http::framing::{ChatFrameDecoder, FramedChatStream};
/// use allama_http::ndjson::NdjsonStream;
/// use futures_util::{FutureExt, StreamExt, stream};
///
/// /// `{"text": ..}` lines, then `{"done": true}`.
/// struct Lines {
///     started: bool,
///     done: bool,
/// }
///
/// impl ChatFrameDecoder<String> for Lines {
///     fn frame(&mut self, line: String) -> Vec<Result<StreamEvent>> {
///         let json: serde_json::Value = match serde_json::from_str(&line) {
///             Ok(json) => json,
///             Err(error) => return vec![Err(Error::stream_decode(error.to_string()))],
///         };
///         let mut out = Vec::new();
///         if !self.started {
///             self.started = true;
///             out.push(Ok(StreamEvent::response_start(None, None)));
///             out.push(Ok(StreamEvent::block_start(0, allama::StreamBlockType::Text)));
///         }
///         if json["done"] == true {
///             self.done = true;
///             out.push(Ok(StreamEvent::block_stop(0)));
///             out.push(Ok(StreamEvent::ResponseStop));
///         } else {
///             out.push(Ok(StreamEvent::text_delta(0, json["text"].as_str().unwrap_or_default())));
///         }
///         out
///     }
///
///     fn end(&mut self) -> Vec<Result<StreamEvent>> {
///         let error = Error::stream(StreamErrorKind::Incomplete, "the stream ended early");
///         vec![Err(error.authored().with_provider("lines"))]
///     }
///
///     fn fail(&mut self, error: Error) -> Vec<Result<StreamEvent>> {
///         vec![Err(error.with_provider("lines"))]
///     }
///
///     fn is_done(&self) -> bool {
///         self.done
///     }
/// }
///
/// let body = stream::iter([Ok::<_, TransportError>(
///     &b"{\"text\":\"Hel\"}\n{\"text\":\"lo\"}\n{\"done\":true}\n"[..],
/// )]);
/// let lines = NdjsonStream::new(body);
/// let stream = FramedChatStream::new(lines, Lines { started: false, done: false });
/// let events: Vec<_> = stream.collect().now_or_never().unwrap();
/// assert_eq!(events.len(), 6);
/// assert!(matches!(events[5], Ok(StreamEvent::ResponseStop)));
/// ```
pub trait ChatFrameDecoder<F>: Send + 'static {
    /// Decode one frame. An error among the items is the stream's last
    /// item: the driver drops the items after it and ends the stream.
    fn frame(&mut self, frame: F) -> Vec<Result<StreamEvent>>;

    /// The body ended before the decoder was done: what the stream still
    /// yields, typically a
    /// [`StreamErrorKind::Incomplete`](allama::StreamErrorKind::Incomplete)
    /// error, since the terminal frame never arrived. The stream ends after
    /// it, whatever it returns, nothing included.
    fn end(&mut self) -> Vec<Result<StreamEvent>>;

    /// Reading or framing the body failed before the decoder was done.
    /// `error` is the classified failure (the framer's error converted
    /// into an [`allama::Error`], or the driver's error map). The default
    /// yields it as it is; override it to close open blocks first, to
    /// attach the provider and the request id, or to add what the adapter
    /// knows about where to resume. The stream ends after it.
    fn fail(&mut self, error: allama::Error) -> Vec<Result<StreamEvent>> {
        vec![Err(error)]
    }

    /// Whether the decoder is done: its terminal frame arrived. The driver
    /// then ends the stream without waiting for the body to close, and
    /// calls neither [`end`](Self::end) nor [`fail`](Self::fail). The
    /// default is `false`: the stream ends with the body.
    fn is_done(&self) -> bool {
        false
    }
}

/// A chat stream decoded from a response's frames by an adapter's
/// [`ChatFrameDecoder`], with the driver guarantees the decoder trait
/// lists.
///
/// `frames` is any stream of `Result<frame, error>`: an
/// [`SseStream`](crate::sse::SseStream), an
/// [`NdjsonStream`](crate::ndjson::NdjsonStream), or a framer of your own.
/// It is unboxed and `Send` (for a `Send` body); box it into a
/// [`ChatStream`](allama::ChatStream) with
/// [`Prepared::finish_stream`](allama::adapter::chat::Prepared::finish_stream):
///
/// ```ignore
/// let lines = NdjsonStream::new(response.into_body()).max_line_size(max_stream_event_size);
/// Ok(prepared.finish_stream(FramedChatStream::new(lines, StreamDecoder::new(request_id))))
/// ```
pub struct FramedChatStream<S: TryStream, D> {
    /// The frames; `None` once the stream ended, so the body is dropped
    /// and never polled again.
    frames: Option<S>,
    map: ErrorMap<S::Error>,
    driver: Driver<D>,
}

impl<S: TryStream, D: ChatFrameDecoder<S::Ok>> FramedChatStream<S, D> {
    /// Drive `decoder` over `frames`, classifying a failure with its
    /// conversion into an [`allama::Error`]. The bundled framers' errors
    /// convert for a body whose error converts into a
    /// [`TransportError`](crate::TransportError), as an
    /// [`HttpResponse`](crate::HttpResponse) body's does
    /// ([`SseError::into_error`](crate::sse::SseError::into_error),
    /// [`NdjsonError::into_error`](crate::ndjson::NdjsonError::into_error)).
    #[must_use]
    pub fn new(frames: S, decoder: D) -> Self
    where
        S::Error: Into<allama::Error> + 'static,
    {
        Self::with_error_map(frames, decoder, Into::into)
    }

    /// Drive `decoder` over `frames`, classifying a failure with `map`, for
    /// a body whose error is not a [`TransportError`](crate::TransportError)
    /// (a platform's own byte stream). `map` must keep the rules of the
    /// framer's own classification: a timeout stays a timeout, bytes the
    /// framer cannot decode are a stream decode error, a frame past the
    /// size cap a stream protocol error, and the messages are the
    /// adapter's own.
    #[must_use]
    pub fn with_error_map<M>(frames: S, decoder: D, map: M) -> Self
    where
        S::Error: 'static,
        M: FnMut(S::Error) -> allama::Error + Send + 'static,
    {
        Self {
            frames: Some(frames),
            map: Box::new(map),
            driver: Driver::new(decoder),
        }
    }
}

// The driver never pins its fields: it polls the frames through
// `Pin::new`, which needs them `Unpin` itself.
impl<S: TryStream, D> Unpin for FramedChatStream<S, D> {}

impl<S, F, E, D> Stream for FramedChatStream<S, D>
where
    S: Stream<Item = std::result::Result<F, E>> + Unpin,
    D: ChatFrameDecoder<F>,
{
    type Item = Result<StreamEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        this.driver.poll(&mut this.frames, &mut this.map, cx)
    }
}

impl<S: TryStream, D> fmt::Debug for FramedChatStream<S, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FramedChatStream")
            .field("ended", &self.frames.is_none())
            .field("queued", &self.driver.queued())
            .finish_non_exhaustive()
    }
}

/// Surfaces vendor frames: stream frames of a type an adapter does not
/// model.
///
/// Endpoints and proxies add frames to standard streams: a named SSE event
/// (`event: search_results`), an NDJSON line of a `type` the API does not
/// define. An adapter never decodes them as its own and never drops them:
/// [`surface`](Self::surface) turns each one into events, as it arrives,
/// and [`finish`](Self::finish) completes them when the response ends.
///
/// # Events
///
/// - The first frame of a type is preceded by one [`Warning`]
///   (`WarningKind::Other`, `Subject::Response`, with the provider name).
/// - Each frame yields a
///   [`ResponseProviderData`](StreamEvent::ResponseProviderData) event
///   under the adapter's namespace and the frame type, holding that frame's
///   payload alone, as a one-element array: a live consumer sees each
///   payload once, as it arrives, in constant work per frame.
/// - [`finish`](Self::finish), called by the decoder just before its
///   `ResponseStop`, yields one more `ResponseProviderData` event holding
///   the whole array, in arrival order, of every type that arrived more
///   than once. [`ProviderData::merge`] replaces a value by key, so a
///   collector that merges the events ends with every payload of every
///   type: the collected response's provider data holds the full arrays.
///
/// Before 2026-10-07 each event carried the whole array so far, which made a
/// stream with a vendor frame per chunk quadratic in time and in what a
/// recording holds. An adapter that forgets [`finish`](Self::finish) keeps
/// only the last payload of a repeated type in its collected response.
///
/// A vendor frame never starts the response. Yield the returned events as
/// they come, also before `ResponseStart`:
/// [`Prepared::finish_stream`](allama::adapter::chat::Prepared::finish_stream) holds
/// them until the start ([`start_first`](allama::adapter::chat::start_first)).
/// `allama_conformance::stream_frames::assert_unknown_stream_frames_surfaced`
/// checks the result.
///
/// ```
/// use allama::{StreamEvent, WarningKind};
/// use allama_http::framing::VendorFrames;
///
/// let mut frames = VendorFrames::new("demo", "openai-compat:demo");
/// let first = frames.surface("search_results", r#"{"url":"https://a"}"#);
/// assert!(matches!(
///     &first[0],
///     StreamEvent::Warning { warning, .. } if warning.kind == WarningKind::Other
/// ));
/// let second = frames.surface("search_results", "not json");
/// let StreamEvent::ResponseProviderData { provider_data, .. } = &second[0] else {
///     panic!("{second:?}")
/// };
/// assert_eq!(
///     provider_data.get_raw("openai-compat:demo", "search_results"),
///     Some(&serde_json::json!(["not json"])),
/// );
///
/// // Before `ResponseStop`: the whole array of each repeated type.
/// let Some(StreamEvent::ResponseProviderData { provider_data, .. }) = frames.finish() else {
///     panic!()
/// };
/// assert_eq!(
///     provider_data.get_raw("openai-compat:demo", "search_results"),
///     Some(&serde_json::json!([{"url": "https://a"}, "not json"])),
/// );
/// ```
#[derive(Debug, Clone)]
pub struct VendorFrames {
    provider: Cow<'static, str>,
    namespace: Cow<'static, str>,
    /// Payloads so far, by frame type.
    payloads: BTreeMap<String, Vec<Value>>,
}

impl VendorFrames {
    /// Vendor frames of a `provider` (for the warnings), surfaced under its
    /// `namespace`.
    #[must_use]
    pub fn new(
        provider: impl Into<Cow<'static, str>>,
        namespace: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self {
            provider: provider.into(),
            namespace: namespace.into(),
            payloads: BTreeMap::new(),
        }
    }

    /// Surface the raw `data` of a frame of type `event`: kept as JSON when
    /// it parses, as the trimmed text otherwise. Empty data surfaces
    /// nothing.
    #[must_use]
    pub fn surface(&mut self, event: &str, data: &str) -> Vec<StreamEvent> {
        let data = data.trim();
        if data.is_empty() {
            return Vec::new();
        }
        let payload = serde_json::from_str(data).unwrap_or_else(|_| Value::String(data.to_owned()));
        self.surface_payload(event, payload)
    }

    /// Surface a payload already decoded, for a frame whose type arrives
    /// inside its data (an NDJSON line's `type`, an SSE event's JSON
    /// `type`) rather than as an SSE event name.
    #[must_use]
    pub fn surface_payload(&mut self, event: &str, payload: Value) -> Vec<StreamEvent> {
        let payloads = self.payloads.entry(event.to_owned()).or_default();
        let mut events = Vec::with_capacity(2);
        if payloads.is_empty() {
            events.push(StreamEvent::warning(
                Warning::new(
                    WarningKind::Other,
                    Subject::Response,
                    format!(
                        "unrecognized stream event `{event}`; its data is kept in the \
                         response's provider data under `{}`",
                        self.namespace
                    ),
                )
                .with_provider(self.provider.clone()),
            ));
        }
        events.push(StreamEvent::response_provider_data(
            ProviderData::new().with_raw(
                self.namespace.clone(),
                event,
                Value::Array(vec![payload.clone()]),
            ),
        ));
        payloads.push(payload);
        events
    }

    /// The event that completes the surfaced frames, for the decoder to
    /// yield just before its `ResponseStop`: one
    /// [`ResponseProviderData`](StreamEvent::ResponseProviderData) holding
    /// the whole array of every type that arrived more than once, which
    /// replaces the single payloads a collector merged. `None` when no type
    /// repeated (the events already carry every payload). The frames are
    /// taken: a second call yields nothing.
    #[must_use]
    pub fn finish(&mut self) -> Option<StreamEvent> {
        let mut provider_data = ProviderData::new();
        let mut any = false;
        for (event, payloads) in std::mem::take(&mut self.payloads) {
            if payloads.len() > 1 {
                provider_data.insert_raw(self.namespace.clone(), event, Value::Array(payloads));
                any = true;
            }
        }
        any.then(|| StreamEvent::response_provider_data(provider_data))
    }
}

/// Classifies a framer's failure.
type ErrorMap<E> = Box<dyn FnMut(E) -> allama::Error + Send>;

/// The loop of [`FramedChatStream`]: a decoder and the items it yielded
/// that the stream has not.
pub(crate) struct Driver<D> {
    decoder: D,
    queue: VecDeque<Result<StreamEvent>>,
}

impl<D> Driver<D> {
    pub(crate) fn new(decoder: D) -> Self {
        Self {
            decoder,
            queue: VecDeque::new(),
        }
    }

    pub(crate) fn queued(&self) -> usize {
        self.queue.len()
    }

    /// The next item of the stream decoded from `frames`, which is set to
    /// `None` once the stream ended.
    pub(crate) fn poll<S, F, E>(
        &mut self,
        frames: &mut Option<S>,
        map: &mut ErrorMap<E>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<StreamEvent>>>
    where
        S: Stream<Item = std::result::Result<F, E>> + Unpin,
        D: ChatFrameDecoder<F>,
    {
        loop {
            if let Some(item) = self.queue.pop_front() {
                return Poll::Ready(Some(item));
            }
            let Some(stream) = frames.as_mut() else {
                return Poll::Ready(None);
            };
            let (items, last) = match ready!(Pin::new(stream).poll_next(cx)) {
                Some(Ok(frame)) => (self.decoder.frame(frame), false),
                Some(Err(error)) => (self.decoder.fail(map(error)), true),
                None => (self.decoder.end(), true),
            };
            if self.queue(items) || last || self.decoder.is_done() {
                *frames = None;
            }
        }
    }

    /// Queue `items` up to and including the first error, and say whether
    /// there was one.
    fn queue(&mut self, items: Vec<Result<StreamEvent>>) -> bool {
        for item in items {
            let failed = item.is_err();
            self.queue.push_back(item);
            if failed {
                return true;
            }
        }
        false
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
