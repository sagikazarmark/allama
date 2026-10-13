//! Newline-delimited JSON (NDJSON) for streaming adapters.
//!
//! Some APIs stream one JSON value per line instead of SSE (Ollama's native
//! API, several self-hosted servers). [`NdjsonStream`] frames a response
//! [`Body`](crate::Body) (or any stream of byte chunks) into its lines, and
//! [`NdjsonDecoder`] is the same framer without the stream, for callers that
//! feed bytes themselves. The lines drive an adapter's
//! [`ChatFrameDecoder<String>`](crate::framing::ChatFrameDecoder) through
//! the shared [`FramedChatStream`](crate::framing::FramedChatStream), the
//! same driver SSE adapters use.
//!
//! The framing rules:
//!
//! - A line ends with `LF`; a `CR` right before it belongs to the line
//!   ending (`CRLF`), also when a chunk boundary splits the two.
//! - Blank lines (empty, or only JSON whitespace) are skipped: they carry
//!   no value, and servers send them as keep-alives.
//! - A leading UTF-8 byte order mark is skipped.
//! - A last line without its `LF` is still a line when the body ends, as
//!   NDJSON allows. A body cut off in the middle of a line therefore
//!   yields a partial line, which the adapter's JSON decoding rejects.
//! - Each byte is scanned once, however the chunks split the lines, and a
//!   size cap ([`NdjsonDecoder::DEFAULT_MAX_LINE_SIZE`] by default) bounds
//!   what a stream can make the framer hold.
//!
//! Two failures end the stream, as in [`sse`](crate::sse): a line that is
//! not UTF-8 ([`NdjsonError::InvalidUtf8`]: JSON text is UTF-8, and a
//! replaced byte would change a value silently) and a line past the size cap
//! ([`NdjsonError::LineTooLarge`]). Neither is retryable: asking again
//! streams the same bytes. [`NdjsonError::into_error`] maps them, and a
//! failed read, to the stream errors the variants name.
//!
//! Decoding each line's JSON stays with the adapter: the framer splits, it
//! does not interpret.
//!
//! ```
//! use allama_http::ndjson::NdjsonStream;
//! use futures_util::{FutureExt, StreamExt, stream};
//!
//! // A `CRLF` split by a chunk boundary, a blank keep-alive line, and a last
//! // line without its line ending.
//! let body = stream::iter([
//!     Ok::<_, std::io::Error>(&b"{\"text\":\"Hel\"}\r"[..]),
//!     Ok(&b"\n\n{\"text\":\"lo\"}\n{\"done\":true}"[..]),
//! ]);
//! let lines: Vec<_> = NdjsonStream::new(body).collect().now_or_never().unwrap();
//! let lines: Vec<_> = lines.into_iter().map(Result::unwrap).collect();
//! assert_eq!(lines, [r#"{"text":"Hel"}"#, r#"{"text":"lo"}"#, r#"{"done":true}"#]);
//! ```

use std::convert::Infallible;
use std::fmt;
use std::pin::Pin;
use std::str::Utf8Error;
use std::task::{Context, Poll, ready};

use futures_core::Stream;

/// The NDJSON lines of a byte stream, such as an
/// [`HttpResponse`](crate::HttpResponse) body.
///
/// Yields each non-blank line, without its line ending, once the line is
/// complete (see the [module docs](self) for the framing rules). After the
/// first error, a failed read ([`NdjsonError::Transport`]), a line that is
/// not UTF-8 ([`NdjsonError::InvalidUtf8`]) or a line past the size cap
/// ([`NdjsonError::LineTooLarge`], see [`max_line_size`](Self::max_line_size)),
/// the stream ends. It polls the body only when no complete line is
/// buffered, so transport-level idle bounds
/// ([`StreamTimeouts`](crate::StreamTimeouts)) see the same waits as
/// before.
///
/// Drive an adapter's decoder over it with
/// [`FramedChatStream::new`](crate::framing::FramedChatStream::new), which
/// classifies a failure with [`NdjsonError::into_error`].
#[derive(Debug)]
pub struct NdjsonStream<S> {
    /// The byte stream; `None` once it ended or failed.
    body: Option<S>,
    decoder: NdjsonDecoder,
}

impl<S> NdjsonStream<S> {
    /// Frame `body` into NDJSON lines, with the default size cap
    /// ([`NdjsonDecoder::DEFAULT_MAX_LINE_SIZE`]).
    #[must_use]
    pub fn new(body: S) -> Self {
        Self {
            body: Some(body),
            decoder: NdjsonDecoder::new(),
        }
    }

    /// Set the size cap, in bytes: see [`NdjsonDecoder::max_line_size`].
    /// Provider crates set it from their builder's
    /// `max_stream_event_size`.
    #[must_use]
    pub fn max_line_size(mut self, limit: usize) -> Self {
        self.decoder = self.decoder.max_line_size(limit);
        self
    }

    fn end(&mut self) {
        self.body = None;
        self.decoder = NdjsonDecoder::new();
    }
}

impl<S, B, E> Stream for NdjsonStream<S>
where
    S: Stream<Item = Result<B, E>> + Unpin,
    B: AsRef<[u8]>,
{
    type Item = Result<String, NdjsonError<E>>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            match this.decoder.next_line() {
                Ok(Some(line)) => return Poll::Ready(Some(Ok(line))),
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
                    return Poll::Ready(Some(Err(NdjsonError::Transport(error))));
                }
                None => {
                    // What is left is a last line without its line ending.
                    let last = this.decoder.finish();
                    this.end();
                    return Poll::Ready(last.map_err(NdjsonError::widen).transpose());
                }
            }
        }
    }
}

/// An incremental NDJSON framer: push bytes, take complete lines.
///
/// [`NdjsonStream`] drives one over a byte stream; use it directly when the
/// bytes do not come as a [`Stream`] (a callback API, a test).
///
/// Framing is linear: each byte is scanned for a line ending once, however
/// the chunks split the lines. A size cap
/// ([`max_line_size`](Self::max_line_size)) bounds the line still waiting
/// for its ending, where memory grows from chunk to chunk.
///
/// ```
/// use allama_http::ndjson::NdjsonDecoder;
///
/// let mut decoder = NdjsonDecoder::new();
/// decoder.push(b"{\"a\":1}\r");
/// assert_eq!(decoder.next_line(), Ok(None));
/// decoder.push(b"\n\n{\"b\":");
/// assert_eq!(decoder.next_line(), Ok(Some(r#"{"a":1}"#.to_owned())));
/// assert_eq!(decoder.next_line(), Ok(None));
/// decoder.push(b"2}");
/// // The body ended: the last line needs no line ending.
/// assert_eq!(decoder.finish(), Ok(Some(r#"{"b":2}"#.to_owned())));
/// ```
#[derive(Debug)]
pub struct NdjsonDecoder {
    /// Received bytes; those before `read` are consumed.
    buffer: Vec<u8>,
    read: usize,
    /// The scanned offset: the bytes from `read` to here hold no `LF`.
    /// [`next_line`](Self::next_line) looks for one from here on.
    scanned: usize,
    /// The bytes from here on hold no `LF` either: the line still waiting
    /// for its ending. [`push`](Self::push) scans each chunk back to its
    /// last `LF` to move it, and `next_line` looks for `LF`s only before
    /// it, so each byte is scanned once.
    unterminated: usize,
    /// Whether the start of the stream (and a byte order mark) is behind.
    started: bool,
    max_line_size: usize,
    /// The line waiting for its ending passed the size cap: the framer
    /// fails once the complete lines before it are taken.
    overflowing: bool,
    /// The size cap was passed: framing cannot go on.
    overflowed: bool,
    /// Bytes examined for line endings, for the linearity test.
    #[cfg(test)]
    examined: usize,
}

impl Default for NdjsonDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl NdjsonDecoder {
    /// The default [`max_line_size`](Self::max_line_size): 64 MiB, as
    /// [`SseDecoder::DEFAULT_MAX_EVENT_SIZE`](crate::sse::SseDecoder::DEFAULT_MAX_EVENT_SIZE),
    /// for the same reasons: a final line can repeat the whole response,
    /// and streamed base64 images are large.
    pub const DEFAULT_MAX_LINE_SIZE: usize = 64 * 1024 * 1024;

    /// A framer at the start of a stream, with the default size cap.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            read: 0,
            scanned: 0,
            unterminated: 0,
            started: false,
            max_line_size: Self::DEFAULT_MAX_LINE_SIZE,
            overflowing: false,
            overflowed: false,
            #[cfg(test)]
            examined: 0,
        }
    }

    /// Set the size cap, in bytes (default
    /// [`DEFAULT_MAX_LINE_SIZE`](Self::DEFAULT_MAX_LINE_SIZE)).
    ///
    /// It bounds a line, its line ending excluded. [`push`](Self::push)
    /// checks the line still waiting for its ending before it buffers a
    /// chunk, so that line never grows much past the cap.
    ///
    /// Past the cap, [`next_line`](Self::next_line) fails with
    /// [`NdjsonError::LineTooLarge`], and framing cannot go on: the framer
    /// drops what it holds, ignores later bytes and keeps returning the
    /// error.
    ///
    /// ```
    /// use allama_http::ndjson::{NdjsonDecoder, NdjsonError};
    ///
    /// let mut decoder = NdjsonDecoder::new().max_line_size(8);
    /// decoder.push(b"{\"a\":12}");
    /// assert_eq!(decoder.next_line(), Ok(None));
    /// // A 9th byte without a line ending.
    /// decoder.push(b"3");
    /// assert!(matches!(
    ///     decoder.next_line(),
    ///     Err(NdjsonError::LineTooLarge { limit: 8, .. })
    /// ));
    /// ```
    #[must_use]
    pub fn max_line_size(mut self, limit: usize) -> Self {
        self.max_line_size = limit;
        self
    }

    /// Append received bytes. Chunk boundaries may fall anywhere, inside a
    /// line, a `CRLF` or a UTF-8 sequence.
    ///
    /// Bytes that would take the line still waiting for its ending past the
    /// [size cap](Self::max_line_size) are not buffered:
    /// [`next_line`](Self::next_line) returns the lines completed before
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
        // Back to the chunk's last `LF`: what follows it starts, or
        // continues, the line still waiting for its ending.
        let last_ending = bytes.iter().rposition(|byte| *byte == b'\n');
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
        // One byte of slack: the line's `CR`, which belongs to its ending.
        let bytes = if waiting > self.max_line_size.saturating_add(1) {
            // Keep the complete lines; the rest is never buffered.
            self.overflowing = true;
            let Some(end) = last_ending else {
                return;
            };
            &bytes[..=end]
        } else {
            bytes
        };
        // A chunk without a line ending extends a line scanned to its end.
        if last_ending.is_none() && self.scanned == self.buffer.len() {
            self.scanned += bytes.len();
        }
        self.buffer.extend_from_slice(bytes);
        self.unterminated = unterminated.min(self.buffer.len());
    }

    /// The next complete, non-blank line, without its line ending, or
    /// `None` until more bytes arrive.
    ///
    /// # Errors
    ///
    /// - [`NdjsonError::InvalidUtf8`] when a line is not valid UTF-8. The
    ///   line is skipped, and framing can go on, but the stream is corrupt:
    ///   [`NdjsonStream`] ends there.
    /// - [`NdjsonError::LineTooLarge`] past the
    ///   [size cap](Self::max_line_size). Framing cannot go on.
    ///
    /// A framer fails with no other error: it reads no transport.
    pub fn next_line(&mut self) -> Result<Option<String>, NdjsonError<Infallible>> {
        if self.overflowed {
            return Err(self.too_large());
        }
        if !self.started {
            let pending = &self.buffer[self.read..];
            if !self.overflowing && pending.len() < BOM.len() && BOM.starts_with(pending) {
                return Ok(None);
            }
            self.skip_bom();
        }
        loop {
            // `LF`s lie only between the two scanned offsets.
            let from = self.scanned.max(self.read);
            let scan = &self.buffer[from..self.unterminated.max(from)];
            let found = scan.iter().position(|byte| *byte == b'\n');
            #[cfg(test)]
            {
                self.examined += found.map_or(scan.len(), |offset| offset + 1);
            }
            let Some(offset) = found else {
                self.scanned = self.buffer.len();
                let waiting = strip_cr(&self.buffer[self.read..]).len();
                if self.overflowing || waiting > self.max_line_size {
                    self.overflow();
                    return Err(self.too_large());
                }
                return Ok(None);
            };
            let (start, end) = (self.read, from + offset);
            self.read = end + 1;
            self.scanned = self.read;
            if let Some(line) = self.line(start, end)? {
                return Ok(Some(line));
            }
        }
    }

    /// The last line, once the body ended: what is left without a line
    /// ending, unless it is blank. Call it after
    /// [`next_line`](Self::next_line) returned `None`; the framer is empty
    /// afterwards.
    ///
    /// # Errors
    ///
    /// As [`next_line`](Self::next_line).
    pub fn finish(&mut self) -> Result<Option<String>, NdjsonError<Infallible>> {
        if self.overflowed || self.overflowing {
            self.overflow();
            return Err(self.too_large());
        }
        if !self.started {
            self.skip_bom();
        }
        let (start, end) = (self.read, self.buffer.len());
        self.read = end;
        self.scanned = end;
        self.unterminated = end;
        self.line(start, end)
    }

    /// The line at `start..end`, its `CR` stripped: `None` when it is
    /// blank.
    fn line(
        &mut self,
        start: usize,
        end: usize,
    ) -> Result<Option<String>, NdjsonError<Infallible>> {
        let line = strip_cr(&self.buffer[start..end]);
        if line.len() > self.max_line_size {
            self.overflow();
            return Err(self.too_large());
        }
        if line.iter().all(|byte| matches!(byte, b' ' | b'\t' | b'\r')) {
            return Ok(None);
        }
        let line = std::str::from_utf8(line).map_err(NdjsonError::InvalidUtf8)?;
        Ok(Some(line.to_owned()))
    }

    fn skip_bom(&mut self) {
        if self.buffer[self.read..].starts_with(BOM) {
            self.read += BOM.len();
            self.scanned = self.scanned.max(self.read);
        }
        self.started = true;
    }

    /// Fail for good: drop what is held.
    fn overflow(&mut self) {
        self.overflowed = true;
        self.buffer = Vec::new();
        self.read = 0;
        self.scanned = 0;
        self.unterminated = 0;
    }

    fn too_large(&self) -> NdjsonError<Infallible> {
        NdjsonError::LineTooLarge {
            limit: self.max_line_size,
        }
    }
}

/// Why an [`NdjsonStream`] (or an [`NdjsonDecoder`], with `E` =
/// [`Infallible`]) failed.
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum NdjsonError<E> {
    /// Reading the body failed: the body stream's own error. Map it as you
    /// would any mid-stream transport failure (a
    /// [`TransportError`](crate::TransportError) keeps its kind, so a
    /// stream timeout stays a timeout).
    Transport(E),
    /// A line was not valid UTF-8. JSON text is UTF-8, and a replaced byte
    /// would change a value without anything saying so, which allama never
    /// does silently (ADR 0003). Invalid UTF-8 means a broken server or
    /// intermediary, and asking again streams the same bytes: map it to a
    /// stream decode error ([`allama::Error::stream_decode`]), which is not
    /// retryable. A chunk boundary inside a UTF-8 sequence is not an error:
    /// each line is decoded once it is complete.
    InvalidUtf8(Utf8Error),
    /// A line passed the framer's size cap
    /// ([`NdjsonDecoder::max_line_size`]). The stream is not one the caller
    /// is prepared to hold, and asking again gets the same answer: map it
    /// to a stream protocol error ([`allama::Error::stream_protocol`]), not
    /// a retryable transport failure.
    #[non_exhaustive]
    LineTooLarge {
        /// The cap, in bytes.
        limit: usize,
    },
}

impl<E: Into<crate::TransportError>> NdjsonError<E> {
    /// The portable error for a failed NDJSON stream, as the variants say:
    ///
    /// - [`Transport`](Self::Transport): the body error as a
    ///   [`TransportError`](crate::TransportError), mapped by
    ///   [`TransportError::into_stream_error`](crate::TransportError::into_stream_error),
    ///   so a stream timeout stays [`allama::Error::Timeout`] and any other
    ///   failed read is a retryable transport stream error;
    /// - [`InvalidUtf8`](Self::InvalidUtf8): a stream decode error;
    /// - [`LineTooLarge`](Self::LineTooLarge): a stream protocol error.
    ///
    /// Neither of the last two is retryable, and every message is
    /// [authored](allama::Error::authored). Add the provider and the request
    /// id. [`FramedChatStream::new`](crate::framing::FramedChatStream::new)
    /// maps failures with it.
    ///
    /// ```
    /// use allama::{Error, StreamErrorKind};
    /// use allama_http::TransportError;
    /// use allama_http::ndjson::NdjsonStream;
    /// use futures_util::{FutureExt, StreamExt, stream};
    ///
    /// let body = stream::iter([Ok::<_, TransportError>(&b"{\"a\":\"\xFF\"}\n"[..])]);
    /// let failure = NdjsonStream::new(body).next().now_or_never().flatten().unwrap().unwrap_err();
    /// let err = failure.into_error().with_provider("acme");
    /// assert!(matches!(err, Error::Stream { kind: StreamErrorKind::Decode, .. }));
    /// assert!(!err.is_retryable());
    /// ```
    #[must_use]
    pub fn into_error(self) -> allama::Error {
        match self {
            Self::Transport(error) => error.into().into_stream_error(),
            Self::InvalidUtf8(error) => allama::Error::stream_decode(
                NdjsonError::<Infallible>::InvalidUtf8(error).to_string(),
            )
            .authored(),
            Self::LineTooLarge { limit } => allama::Error::stream_protocol(
                NdjsonError::<Infallible>::LineTooLarge { limit }.to_string(),
            )
            .authored(),
        }
    }
}

impl<E: Into<crate::TransportError>> From<NdjsonError<E>> for allama::Error {
    /// The error [`NdjsonError::into_error`] gives: what
    /// [`FramedChatStream::new`](crate::framing::FramedChatStream::new)
    /// classifies a failure with.
    fn from(error: NdjsonError<E>) -> Self {
        error.into_error()
    }
}

impl NdjsonError<Infallible> {
    /// A framer's error as a stream's.
    fn widen<E>(self) -> NdjsonError<E> {
        match self {
            Self::Transport(never) => match never {},
            Self::InvalidUtf8(error) => NdjsonError::InvalidUtf8(error),
            Self::LineTooLarge { limit } => NdjsonError::LineTooLarge { limit },
        }
    }
}

impl<E: fmt::Display> fmt::Display for NdjsonError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(error) => error.fmt(f),
            Self::InvalidUtf8(error) => write!(f, "invalid UTF-8 in the NDJSON stream: {error}"),
            Self::LineTooLarge { limit } => {
                write!(f, "an NDJSON line passed the size cap of {limit} bytes")
            }
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for NdjsonError<E> {
    /// A transport failure is transparent: its source is the body error's
    /// source.
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transport(error) => error.source(),
            Self::InvalidUtf8(error) => Some(error),
            Self::LineTooLarge { .. } => None,
        }
    }
}

/// The UTF-8 byte order mark, skipped at the start of a stream.
const BOM: &[u8] = b"\xEF\xBB\xBF";

/// `line` without the `CR` of a `CRLF` ending.
fn strip_cr(line: &[u8]) -> &[u8] {
    line.strip_suffix(b"\r").unwrap_or(line)
}

#[cfg(test)]
mod tests;
