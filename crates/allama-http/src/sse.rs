//! Server-sent events (SSE) for streaming adapters.
//!
//! Every streaming HTTP API in the bundled providers speaks SSE, so this is
//! the framing most adapters need, first-party or not:
//!
//! - **Framing.** [`SseStream`] turns a response [`Body`](crate::Body) (or
//!   any stream of byte chunks) into [`SseEvent`]s; [`SseDecoder`] is the
//!   same decoder without the stream, for callers that feed bytes
//!   themselves. Both follow the [WHATWG event stream format]: `LF`, `CR`
//!   and `CRLF` line endings (also when a chunk boundary splits a `CRLF`),
//!   multi-line `data:` fields joined with `\n`, a leading byte order mark,
//!   comments (`: keepalive`) and `retry:` lines skipped, and an event
//!   without a trailing blank line discarded at the end of the body. Each
//!   byte is scanned once, however the chunks split the lines, and a size
//!   cap ([`SseDecoder::DEFAULT_MAX_EVENT_SIZE`] by default) bounds what a
//!   stream can make the decoder hold.
//! - **Two deliberate failures.** Where the standard decodes a stream with
//!   invalid UTF-8 lossily, these helpers fail
//!   ([`SseError::InvalidUtf8`]), and a line or an event past the size cap
//!   fails too ([`SseError::EventTooLarge`]). Neither is retryable: asking
//!   again streams the same bytes. [`SseError::into_error`] maps them, and a
//!   failed read, to the stream errors the variants name.
//! - **The driver.** [`SseChatStream`] runs an adapter's
//!   [`ChatFrameDecoder`](crate::framing::ChatFrameDecoder)`<`[`SseEvent`]`>`
//!   over the events: it is the framing-neutral
//!   [`FramedChatStream`](crate::framing::FramedChatStream) over an
//!   [`SseStream`] (a type alias), so it ends the stream after the first error, runs the
//!   decoder's end-of-body or failure hook once, classifies a failed read
//!   (a stream timeout stays a timeout), and never polls the body after
//!   the stream ended.
//! - **Vendor frames.** [`VendorFrames`] (from [`framing`](crate::framing))
//!   surfaces named events an adapter does not model
//!   (`event: search_results`) as response-level provider data under the
//!   adapter's namespace, with one warning per event type, so they are
//!   never decoded as the adapter's own and never dropped.
//!
//! Other wire framings use the same driver and vendor frames:
//! [`ndjson`](crate::ndjson) frames newline-delimited JSON, and any other
//! stream of frames drives a
//! [`ChatFrameDecoder`](crate::framing::ChatFrameDecoder).
//!
//! Decoding the events' data stays with each adapter: the helpers frame,
//! drive and surface, they do not interpret.
//!
//! ```
//! use allama::{ProviderData, StreamEvent};
//! use allama_http::sse::{SseStream, VendorFrames};
//! use futures_util::{FutureExt, StreamExt, stream};
//!
//! // A body whose second chunk starts in the middle of a `data:` line.
//! let body = stream::iter([
//!     Ok::<_, std::io::Error>(&b": keepalive\r\nevent: delta\r\ndata: {\"text\":"[..]),
//!     Ok(&b"\"Hi\"}\r\n\r\nevent: search_results\r\ndata: [1]\r\n\r\n"[..]),
//! ]);
//! let events: Vec<_> = SseStream::new(body).collect().now_or_never().unwrap();
//! let mut vendor_frames = VendorFrames::new("demo", "demo");
//! let mut out = Vec::new();
//! for event in events {
//!     let event = event.unwrap();
//!     match event.event.as_str() {
//!         "delta" => assert_eq!(event.data, r#"{"text":"Hi"}"#),
//!         name => out.extend(vendor_frames.surface(name, &event.data)),
//!     }
//! }
//! // A warning, then the data under the namespace and event type.
//! assert!(matches!(out[0], StreamEvent::Warning { .. }));
//! assert_eq!(
//!     out[1],
//!     StreamEvent::response_provider_data(ProviderData::new().with_raw(
//!         "demo",
//!         "search_results",
//!         serde_json::json!([[1]]),
//!     )),
//! );
//! ```
//!
//! [WHATWG event stream format]: https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream

use std::convert::Infallible;
use std::fmt;
use std::pin::Pin;
use std::str::Utf8Error;
use std::task::{Context, Poll, ready};

use futures_core::Stream;

/// The SSE events of a byte stream, such as an [`HttpResponse`](crate::HttpResponse)
/// body.
///
/// Yields each event once its terminating blank line arrives (see the
/// [module docs](self) for the framing rules). After the first error,
/// a failed read ([`SseError::Transport`]), a line that is not UTF-8
/// ([`SseError::InvalidUtf8`]) or an event past the size cap
/// ([`SseError::EventTooLarge`], see [`max_event_size`](Self::max_event_size)),
/// the stream ends. It polls the body only when no complete event is
/// buffered, so transport-level idle bounds
/// ([`StreamTimeouts`](crate::StreamTimeouts)) see the same waits as
/// before.
///
/// ```
/// use allama_http::sse::SseStream;
/// use futures_util::{FutureExt, StreamExt, stream};
///
/// let body = stream::iter([Ok::<_, std::io::Error>("data: {\"a\":\ndata: 1}\n\n")]);
/// let events: Vec<_> = SseStream::new(body).collect().now_or_never().unwrap();
/// let event = events[0].as_ref().unwrap();
/// assert_eq!(event.event, "message");
/// assert_eq!(event.data, "{\"a\":\n1}");
/// ```
#[derive(Debug)]
pub struct SseStream<S> {
    /// The byte stream; `None` once it ended or failed.
    body: Option<S>,
    decoder: SseDecoder,
}

impl<S> SseStream<S> {
    /// Frame `body` into SSE events, with the default size cap
    /// ([`SseDecoder::DEFAULT_MAX_EVENT_SIZE`]).
    #[must_use]
    pub fn new(body: S) -> Self {
        Self {
            body: Some(body),
            decoder: SseDecoder::new(),
        }
    }

    /// Set the size cap, in bytes: see [`SseDecoder::max_event_size`].
    /// Provider crates set it from their builder's
    /// `max_stream_event_size`.
    #[must_use]
    pub fn max_event_size(mut self, limit: usize) -> Self {
        self.decoder = self.decoder.max_event_size(limit);
        self
    }

    fn end(&mut self) {
        self.body = None;
        self.decoder = SseDecoder::new();
    }
}

impl<S, B, E> Stream for SseStream<S>
where
    S: Stream<Item = Result<B, E>> + Unpin,
    B: AsRef<[u8]>,
{
    type Item = Result<SseEvent, SseError<E>>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            match this.decoder.next_event() {
                Ok(Some(event)) => return Poll::Ready(Some(Ok(event))),
                Ok(None) => {}
                Err(error) => {
                    this.end();
                    return Poll::Ready(Some(Err(error.widen())));
                }
            }
            let Some(body) = this.body.as_mut() else {
                return Poll::Ready(None);
            };
            match ready!(Pin::new(body).poll_next(cx)) {
                Some(Ok(chunk)) => this.decoder.push(chunk.as_ref()),
                Some(Err(error)) => {
                    this.end();
                    return Poll::Ready(Some(Err(SseError::Transport(error))));
                }
                // What is left is an event without its blank line: discarded.
                None => {
                    this.end();
                    return Poll::Ready(None);
                }
            }
        }
    }
}

/// An incremental SSE decoder: push bytes, take complete events.
///
/// [`SseStream`] drives one over a byte stream; use it directly when the
/// bytes do not come as a [`Stream`] (a callback API, a test).
///
/// Decoding is linear: each byte is scanned for a line ending once, however
/// the chunks split the lines. A size cap
/// ([`max_event_size`](Self::max_event_size)) bounds the line still waiting
/// for its ending, where memory grows from chunk to chunk, and the data an
/// event accumulates.
///
/// ```
/// use allama_http::sse::SseDecoder;
///
/// let mut decoder = SseDecoder::new();
/// decoder.push(b"event: ping\r");
/// assert_eq!(decoder.next_event(), Ok(None));
/// decoder.push(b"\ndata: {}\r\n\r\n");
/// let event = decoder.next_event().unwrap().unwrap();
/// assert_eq!((event.event.as_str(), event.data.as_str()), ("ping", "{}"));
/// assert_eq!(decoder.next_event(), Ok(None));
/// ```
#[derive(Debug)]
pub struct SseDecoder {
    /// Received bytes; those before `read` are consumed.
    buffer: Vec<u8>,
    read: usize,
    /// The scanned offset: the bytes from `read` to here hold no line
    /// ending. [`next_event`](Self::next_event) looks for one from here on.
    scanned: usize,
    /// The bytes from here on hold no line ending either: the line still
    /// waiting for its ending. [`push`](Self::push) scans each chunk back
    /// to its last line ending to move it, and `next_event` looks for line
    /// endings only before it, so each byte is scanned once.
    unterminated: usize,
    /// Whether the start of the stream (and a byte order mark) is behind.
    started: bool,
    /// The last line ended with `CR`: a `LF` right after it belongs to it.
    after_cr: bool,
    max_event_size: usize,
    /// The line waiting for its ending passed the size cap: the decoder
    /// fails once the complete lines before it are taken.
    overflowing: bool,
    /// The size cap was passed: decoding cannot go on.
    overflowed: bool,
    event: PendingEvent,
    /// Bytes examined for line endings, for the linearity test.
    #[cfg(test)]
    examined: usize,
}

impl Default for SseDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SseDecoder {
    /// The default [`max_event_size`](Self::max_event_size): 64 MiB.
    ///
    /// Generous on purpose: a Responses `response.completed` event repeats
    /// every output of the response, and streamed base64 images (OpenAI
    /// partial images, Gemini inline data) can pass 16 MiB.
    pub const DEFAULT_MAX_EVENT_SIZE: usize = 64 * 1024 * 1024;

    /// A decoder at the start of a stream, with the default size cap.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            read: 0,
            scanned: 0,
            unterminated: 0,
            started: false,
            after_cr: false,
            max_event_size: Self::DEFAULT_MAX_EVENT_SIZE,
            overflowing: false,
            overflowed: false,
            event: PendingEvent::default(),
            #[cfg(test)]
            examined: 0,
        }
    }

    /// Set the size cap, in bytes (default
    /// [`DEFAULT_MAX_EVENT_SIZE`](Self::DEFAULT_MAX_EVENT_SIZE)).
    ///
    /// It bounds a line (its line ending excluded) and an event's data (its
    /// `data:` values, joined). [`push`](Self::push) checks the line still
    /// waiting for its ending before it buffers a chunk, so that line never
    /// grows past the cap, not even briefly while the buffer reallocates.
    ///
    /// Past the cap, [`next_event`](Self::next_event) fails with
    /// [`SseError::EventTooLarge`], and decoding cannot go on: the decoder
    /// drops what it holds, ignores later bytes and keeps returning the
    /// error.
    ///
    /// ```
    /// use allama_http::sse::{SseDecoder, SseError};
    ///
    /// let mut decoder = SseDecoder::new().max_event_size(16);
    /// decoder.push(b"data: 0123456789");
    /// assert_eq!(decoder.next_event(), Ok(None));
    /// // A 17th byte without a line ending.
    /// decoder.push(b"a");
    /// assert!(matches!(
    ///     decoder.next_event(),
    ///     Err(SseError::EventTooLarge { limit: 16, .. })
    /// ));
    /// ```
    #[must_use]
    pub fn max_event_size(mut self, limit: usize) -> Self {
        self.max_event_size = limit;
        self
    }

    /// Append received bytes. Chunk boundaries may fall anywhere, inside a
    /// line, a `CRLF` or a UTF-8 sequence.
    ///
    /// Bytes that would take the line still waiting for its ending past the
    /// [size cap](Self::max_event_size) are not buffered:
    /// [`next_event`](Self::next_event) returns the events completed before
    /// that line, then fails.
    pub fn push(&mut self, bytes: &[u8]) {
        if self.overflowing || self.overflowed {
            return;
        }
        if self.read > 0 {
            self.buffer.drain(..self.read);
            self.scanned = self.scanned.saturating_sub(self.read);
            self.unterminated = self.unterminated.saturating_sub(self.read);
            self.read = 0;
        }
        // Back to the chunk's last line ending: what follows it starts, or
        // continues, the line still waiting for its ending.
        let last_ending = bytes.iter().rposition(|byte| is_line_ending(*byte));
        #[cfg(test)]
        {
            self.examined += last_ending.map_or(bytes.len(), |end| bytes.len() - end);
        }
        let (unterminated, waiting) = match last_ending {
            Some(end) => (self.buffer.len() + end + 1, bytes.len() - end - 1),
            None => (
                self.unterminated,
                self.buffer.len() - self.unterminated + bytes.len(),
            ),
        };
        let bytes = if waiting > self.max_event_size {
            // Keep the complete lines; the rest is never buffered.
            self.overflowing = true;
            let Some(end) = last_ending else {
                return;
            };
            &bytes[..=end]
        } else {
            bytes
        };
        let additional = room(
            self.buffer.len(),
            self.buffer.capacity(),
            bytes.len(),
            self.max_event_size,
        );
        self.buffer.reserve_exact(additional);
        // A chunk without a line ending extends a line scanned to its end.
        if last_ending.is_none() && self.scanned == self.buffer.len() {
            self.scanned += bytes.len();
        }
        self.buffer.extend_from_slice(bytes);
        self.unterminated = unterminated.min(self.buffer.len());
    }

    /// The next complete event, or `None` until more bytes arrive.
    ///
    /// # Errors
    ///
    /// - [`SseError::InvalidUtf8`] when a line is not valid UTF-8. The line
    ///   is skipped, and decoding can go on, but the stream is corrupt:
    ///   [`SseStream`] ends there.
    /// - [`SseError::EventTooLarge`] past the
    ///   [size cap](Self::max_event_size). Decoding cannot go on.
    ///
    /// A decoder fails with no other error: it reads no transport.
    pub fn next_event(&mut self) -> Result<Option<SseEvent>, SseError<Infallible>> {
        if self.overflowed {
            return Err(self.too_large());
        }
        if !self.started {
            let pending = &self.buffer[self.read..];
            if !self.overflowing && pending.len() < BOM.len() && BOM.starts_with(pending) {
                return Ok(None);
            }
            if pending.starts_with(BOM) {
                self.read += BOM.len();
                self.scanned = self.scanned.max(self.read);
            }
            self.started = true;
        }
        loop {
            let pending = &self.buffer[self.read..];
            if self.after_cr && !pending.is_empty() {
                self.after_cr = false;
                if pending[0] == b'\n' {
                    self.read += 1;
                    self.scanned = self.scanned.max(self.read);
                    continue;
                }
            }
            // Line endings lie only between the two scanned offsets.
            let from = self.scanned.max(self.read);
            let scan = &self.buffer[from..self.unterminated.max(from)];
            let found = scan.iter().position(|byte| is_line_ending(*byte));
            #[cfg(test)]
            {
                self.examined += found.map_or(scan.len(), |offset| offset + 1);
            }
            let Some(offset) = found else {
                self.scanned = self.buffer.len();
                if self.overflowing {
                    self.overflow();
                    return Err(self.too_large());
                }
                return Ok(None);
            };
            let (start, end) = (self.read, from + offset);
            if end - start > self.max_event_size {
                self.overflow();
                return Err(self.too_large());
            }
            self.read = end + 1;
            self.scanned = self.read;
            self.after_cr = self.buffer[end] == b'\r';
            let line =
                std::str::from_utf8(&self.buffer[start..end]).map_err(SseError::InvalidUtf8)?;
            match self.event.line(line, self.max_event_size) {
                Ok(Some(event)) => return Ok(Some(event)),
                Ok(None) => {}
                Err(TooLarge) => {
                    self.overflow();
                    return Err(self.too_large());
                }
            }
        }
    }

    /// Fail for good: drop what is held.
    fn overflow(&mut self) {
        self.overflowed = true;
        self.buffer = Vec::new();
        self.read = 0;
        self.scanned = 0;
        self.unterminated = 0;
        self.event = PendingEvent::default();
    }

    fn too_large(&self) -> SseError<Infallible> {
        SseError::EventTooLarge {
            limit: self.max_event_size,
        }
    }
}

/// One SSE event.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct SseEvent {
    /// The event type: the `event:` field, or `"message"` when the event
    /// has none (as in the SSE standard).
    pub event: String,
    /// The `data:` fields, joined with `\n`. An event with no `data:`
    /// field is never dispatched; one with an empty `data:` field is, with
    /// empty data.
    pub data: String,
    /// The last event id the stream set (`id:`), if any. It carries over to
    /// later events until another `id:` field changes it.
    pub id: Option<String>,
}

impl SseEvent {
    /// An event with the given type and data, and no event id.
    ///
    /// Decoders get their events from [`SseStream`] or [`SseDecoder`]; this
    /// builds one directly, as when testing an adapter's event mapping. Set
    /// [`id`](Self::id) afterwards for an event that carries one.
    ///
    /// ```
    /// use allama_http::sse::SseEvent;
    ///
    /// let mut event = SseEvent::new("message_delta", r#"{"type":"message_delta"}"#);
    /// assert_eq!(event.event, "message_delta");
    /// assert_eq!(event.id, None);
    ///
    /// event.id = Some("42".into());
    /// ```
    #[must_use]
    pub fn new(event: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            event: event.into(),
            data: data.into(),
            id: None,
        }
    }
}

/// Why an [`SseStream`] (or an [`SseDecoder`], with `E` =
/// [`Infallible`]) failed.
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SseError<E> {
    /// Reading the body failed: the body stream's own error. Map it as you
    /// would any mid-stream transport failure (a
    /// [`TransportError`](crate::TransportError) keeps its kind, so a
    /// stream timeout stays a timeout).
    Transport(E),
    /// A line was not valid UTF-8.
    ///
    /// The [WHATWG event stream format] decodes the stream as UTF-8 and
    /// replaces invalid bytes with U+FFFD. This decoder fails instead, on
    /// purpose. A browser shows event text to a person; an adapter decodes
    /// JSON from it, and a replaced byte changes a string value (generated
    /// text, tool arguments, an opaque signature that is replayed later)
    /// without anything saying so, which allama never does silently (ADR
    /// 0003). Invalid UTF-8 means a broken server or intermediary, and
    /// asking again streams the same bytes: map it to a stream decode
    /// error ([`allama::Error::stream_decode`]), which is not retryable,
    /// not to a transport failure. A chunk boundary inside a UTF-8
    /// sequence is not an error: each line is decoded once it is complete.
    ///
    /// [WHATWG event stream format]: https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation
    InvalidUtf8(Utf8Error),
    /// A line, or an event's data, passed the decoder's size cap
    /// ([`SseDecoder::max_event_size`]). The stream is not one the caller
    /// is prepared to hold, and asking again gets the same answer: map it
    /// to a stream protocol error ([`allama::Error::stream_protocol`]),
    /// not a retryable transport failure.
    #[non_exhaustive]
    EventTooLarge {
        /// The cap, in bytes.
        limit: usize,
    },
}

impl<E: Into<crate::TransportError>> SseError<E> {
    /// The portable error for a failed SSE stream, as the variants say:
    ///
    /// - [`Transport`](Self::Transport): the body error as a
    ///   [`TransportError`](crate::TransportError), mapped by
    ///   [`TransportError::into_stream_error`](crate::TransportError::into_stream_error),
    ///   so a stream timeout stays [`allama::Error::Timeout`] and any other
    ///   failed read is a retryable transport stream error, with the
    ///   transport's text as the upstream message;
    /// - [`InvalidUtf8`](Self::InvalidUtf8): a stream decode error;
    /// - [`EventTooLarge`](Self::EventTooLarge): a stream protocol error.
    ///
    /// Neither of the last two is retryable, and every message is
    /// [authored](allama::Error::authored). Add the provider and the request
    /// id. [`SseChatStream::new`] maps failures with it.
    ///
    /// ```
    /// use allama::{Error, StreamErrorKind};
    /// use allama_http::sse::{SseError, SseStream};
    /// use allama_http::{TransportError, TransportErrorKind};
    /// use futures_util::{FutureExt, StreamExt, stream};
    ///
    /// let body = stream::iter([Ok::<_, TransportError>(&b"data: \xFF\n\n"[..])]);
    /// let failure = SseStream::new(body).next().now_or_never().flatten().unwrap().unwrap_err();
    /// let err = failure.into_error().with_provider("acme");
    /// assert!(matches!(err, Error::Stream { kind: StreamErrorKind::Decode, .. }));
    /// assert!(!err.is_retryable());
    ///
    /// let timeout = SseError::Transport(TransportError::new(TransportErrorKind::Timeout, "idle"));
    /// assert!(matches!(timeout.into_error(), Error::Timeout { .. }));
    /// ```
    #[must_use]
    pub fn into_error(self) -> allama::Error {
        match self {
            Self::Transport(error) => error.into().into_stream_error(),
            Self::InvalidUtf8(error) => {
                allama::Error::stream_decode(SseError::<Infallible>::InvalidUtf8(error).to_string())
                    .authored()
            }
            Self::EventTooLarge { limit } => allama::Error::stream_protocol(
                SseError::<Infallible>::EventTooLarge { limit }.to_string(),
            )
            .authored(),
        }
    }
}

impl SseError<Infallible> {
    /// A decoder's error as a stream's.
    fn widen<E>(self) -> SseError<E> {
        match self {
            Self::Transport(never) => match never {},
            Self::InvalidUtf8(error) => SseError::InvalidUtf8(error),
            Self::EventTooLarge { limit } => SseError::EventTooLarge { limit },
        }
    }
}

impl<E: fmt::Display> fmt::Display for SseError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(error) => error.fmt(f),
            Self::InvalidUtf8(error) => write!(f, "invalid UTF-8 in the SSE stream: {error}"),
            Self::EventTooLarge { limit } => {
                write!(f, "an SSE event passed the size cap of {limit} bytes")
            }
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for SseError<E> {
    /// A transport failure is transparent: its source is the body error's
    /// source.
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transport(error) => error.source(),
            Self::InvalidUtf8(error) => Some(error),
            Self::EventTooLarge { .. } => None,
        }
    }
}

impl<E: Into<crate::TransportError>> From<SseError<E>> for allama::Error {
    /// The error [`SseError::into_error`] gives: what
    /// [`FramedChatStream::new`](crate::framing::FramedChatStream::new)
    /// classifies a failure with.
    fn from(error: SseError<E>) -> Self {
        error.into_error()
    }
}

/// The UTF-8 byte order mark, skipped at the start of a stream.
const BOM: &[u8] = b"\xEF\xBB\xBF";

fn is_line_ending(byte: u8) -> bool {
    matches!(byte, b'\n' | b'\r')
}

/// How much to reserve so a buffer of `len` bytes and `capacity` takes
/// `additional` more: doubling, as `Vec` does, but never past `limit` for
/// the doubling alone.
fn room(len: usize, capacity: usize, additional: usize, limit: usize) -> usize {
    let needed = len + additional;
    if needed <= capacity {
        return 0;
    }
    capacity.saturating_mul(2).min(limit).max(needed) - len
}

/// An event's data passed the size cap.
struct TooLarge;

/// The fields of the event being received.
#[derive(Debug, Default)]
struct PendingEvent {
    event: String,
    /// `data:` values, each followed by `\n`: empty until a `data:` field.
    data: String,
    /// The last event id: it outlives the event.
    last_id: String,
}

impl PendingEvent {
    /// Process one line; a blank line dispatches the event. The event's
    /// data may hold up to `max_data` bytes.
    fn line(&mut self, line: &str, max_data: usize) -> Result<Option<SseEvent>, TooLarge> {
        if line.is_empty() {
            return Ok(self.dispatch());
        }
        if line.starts_with(':') {
            return Ok(None);
        }
        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };
        match field {
            "event" => value.clone_into(&mut self.event),
            "data" => {
                // The joined data: the values so far, each with its `\n`,
                // then this one.
                if self.data.len() + value.len() > max_data {
                    return Err(TooLarge);
                }
                let additional = room(
                    self.data.len(),
                    self.data.capacity(),
                    value.len() + 1,
                    max_data.saturating_add(1),
                );
                self.data.reserve_exact(additional);
                self.data.push_str(value);
                self.data.push('\n');
            }
            "id" if !value.contains('\0') => value.clone_into(&mut self.last_id),
            // `retry:` sets a reconnection delay; a streamed model response
            // is never reconnected. Unknown fields are ignored, as the
            // standard requires.
            _ => {}
        }
        Ok(None)
    }

    fn dispatch(&mut self) -> Option<SseEvent> {
        let event = std::mem::take(&mut self.event);
        let mut data = std::mem::take(&mut self.data);
        // No `data:` field, no event; otherwise drop the last `\n`.
        data.pop()?;
        Some(SseEvent {
            event: if event.is_empty() {
                "message".to_owned()
            } else {
                event
            },
            data,
            id: (!self.last_id.is_empty()).then(|| self.last_id.clone()),
        })
    }
}

mod chat;

pub use crate::framing::VendorFrames;
pub use chat::SseChatStream;

#[cfg(test)]
mod tests;
