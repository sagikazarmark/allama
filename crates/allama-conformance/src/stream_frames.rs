//! Vendor frame contracts for streaming adapters, in any wire framing.
//!
//! Vendors add frames to standard streams: frames of a type the adapter
//! was not written for (`event: search_results` in SSE, a line of an
//! unknown `type` in NDJSON), and framing that carries no data (SSE
//! keep-alive comments and `retry:` lines, blank NDJSON lines). Servers and
//! proxies also frame the same data differently: `CRLF` line endings, or
//! one SSE event's data over several `data:` lines.
//! [`assert_unknown_stream_frames_surfaced`] (contract C1) checks that a
//! decoder neither misreads such frames as its own nor drops their data
//! silently, and that the framing does not change what it decodes.
//!
//! The contract injects frames into a captured stream body, so it needs to
//! know the body's framing: a [`StreamFraming`]. [`StreamFraming::sse`]
//! knows SSE whole. Other framings take a closure that writes a
//! vendor frame the way the adapter's wire format would carry one, since
//! only the adapter knows where its frame type goes:
//! [`StreamFraming::ndjson`], or [`StreamFraming::new`] for a framing of
//! your own. `allama_http::framing::VendorFrames` surfaces vendor frames
//! for the decoder side.
//!
//! ```no_run
//! use allama::{ChatProvider, ChatRequest};
//! use allama_conformance::stream_frames::{StreamFraming, assert_unknown_stream_frames_surfaced};
//! use serde_json::json;
//!
//! # async fn example<P: ChatProvider>(provider_for: impl Fn(String) -> P) {
//! let base = "{\"type\":\"text\",\"delta\":\"Hi\"}\n{\"type\":\"done\"}\n";
//! // Acme puts a line's type in its `type` field.
//! let framing = StreamFraming::ndjson(|frame_type, payload| {
//!     let mut line = payload.clone();
//!     line["type"] = json!(frame_type);
//!     line.to_string()
//! });
//! assert_unknown_stream_frames_surfaced(base, &framing, |body| {
//!     // A provider whose transport replays `body`.
//!     let provider = provider_for(body);
//!     async move { provider.chat_stream(&ChatRequest::new("m").user("hi")).await }
//! })
//! .await;
//! # }
//! ```

use std::fmt;
use std::future::Future;

use allama::chat::stream::StreamCollector;
use allama::{ChatResponse, StreamEvent};
use futures_core::Stream;
use futures_util::StreamExt;
use serde_json::{Value, json};

/// Contract C1: unknown stream frames are surfaced, never misdecoded.
///
/// `base` is a complete stream body the adapter decodes (a fixture),
/// framed as `framing` says, and `decode` turns a body into the adapter's
/// event stream (typically through a provider backed by a replay
/// transport). The contract injects a vendor frame of type
/// [`VENDOR_EVENT`] carrying [`VENDOR_MARKER`] (written by the framing's
/// vendor frame closure), with the framing's filler frames, before the
/// first frame, after it (with filler between every frame), and
/// mid-stream, and twice mid-stream with different data, and asserts for
/// each variant that:
///
/// - the stream is the base stream, except for added `Warning` and
///   `ResponseProviderData` events (so `ResponseStart` keeps its id and
///   model, and no frame is decoded as the adapter's own);
/// - the data of every injected frame reaches the collected response's
///   [`provider_data`](ChatResponse::provider_data): of a type that
///   repeats, every payload, not only the last one (a decoder that never
///   yields `VendorFrames::finish` keeps only the last).
///
/// It also reframes the base body with each of the framing's
/// [reframings](StreamFraming::reframing), without adding anything, and
/// asserts that each decodes to exactly the base events.
///
/// Frames are never injected after the middle of the base stream: once the
/// response has ended there is nowhere to surface them, so the base stream
/// should not end with a long tail of frames after its terminal frame.
///
/// # Panics
///
/// When a variant violates the contract, or `decode` fails.
pub async fn assert_unknown_stream_frames_surfaced<F, Fut, S>(
    base: &str,
    framing: &StreamFraming,
    mut decode: F,
) where
    F: FnMut(String) -> Fut,
    Fut: Future<Output = allama::Result<S>>,
    S: Stream<Item = allama::Result<StreamEvent>>,
{
    let base_run = Run::collect(decode(base.to_owned()).await).await;
    let base_portable = base_run.portable();
    for (label, body, markers) in framing.injected(base) {
        let run = Run::collect(decode(body.clone()).await).await;
        assert_eq!(
            run.response_start(),
            base_run.response_start(),
            "unknown_stream_frames ({label}): ResponseStart changed\n--- body ---\n{body}"
        );
        assert_eq!(
            run.portable(),
            base_portable,
            "unknown_stream_frames ({label}): injected frames changed the portable events\n--- body ---\n{body}"
        );
        let provider_data =
            serde_json::to_string(&run.response.provider_data).expect("serialize provider data");
        for marker in markers {
            assert!(
                provider_data.contains(&format!("\"{marker}\"")),
                "unknown_stream_frames ({label}): `{VENDOR_EVENT}` data `{marker}` was not \
                 surfaced in the response's provider data (a repeated frame type keeps every \
                 payload only when the decoder yields `VendorFrames::finish` before its \
                 `ResponseStop`; warnings: {:?})\n--- body ---\n{body}",
                run.response.warnings
            );
        }
    }
    for (label, reframe) in &framing.reframings {
        let body = reframe(base);
        let run = Run::collect(decode(body.clone()).await).await;
        assert_eq!(
            run.events, base_run.events,
            "unknown_stream_frames ({label}): the framing changed the decoded events\n--- body ---\n{body:?}"
        );
    }
}

/// Type of the vendor frame the contract injects.
pub const VENDOR_EVENT: &str = "x-vendor";

/// A value inside the injected frame's data, looked up in the collected
/// response's provider data.
pub const VENDOR_MARKER: &str = "allama-conformance-vendor-frame";

/// A stream body's wire framing, as contract C1 needs it: how to split a
/// body into frames and join them back, how to write a vendor frame, the
/// filler frames that carry no data, and other ways to frame the same
/// body.
///
/// [`sse`](Self::sse) and [`ndjson`](Self::ndjson) cover the bundled
/// framers (`allama_http::sse`, `allama_http::ndjson`); [`new`](Self::new)
/// describes any other text framing.
pub struct StreamFraming {
    split: Split,
    join: Join,
    vendor_frame: WriteVendorFrame,
    filler: Vec<String>,
    reframings: Vec<(&'static str, Reframe)>,
}

impl StreamFraming {
    /// A framing that splits a body into its frames with `split`, joins
    /// frames back into a body with `join`, and writes a vendor frame of a
    /// type carrying a JSON payload with `vendor_frame`, as the adapter's
    /// wire format would carry it. It has no filler and no reframings until
    /// [`filler`](Self::filler) and [`reframing`](Self::reframing) add them.
    #[must_use]
    pub fn new(
        split: impl Fn(&str) -> Vec<String> + 'static,
        join: impl Fn(&[String]) -> String + 'static,
        vendor_frame: impl Fn(&str, &Value) -> String + 'static,
    ) -> Self {
        Self {
            split: Box::new(split),
            join: Box::new(join),
            vendor_frame: Box::new(vendor_frame),
            filler: Vec::new(),
            reframings: Vec::new(),
        }
    }

    /// Server-sent events: frames separated by a blank line, a vendor frame
    /// as a named event (`event: x-vendor`), comments (`: keepalive`) and
    /// `retry:` lines as filler, and two reframings: `CRLF` line endings,
    /// and each JSON `data:` field split over several `data:` lines (after
    /// the commas outside strings), which SSE joins back with `\n`, so the
    /// data is the same JSON.
    #[must_use]
    pub fn sse() -> Self {
        Self::new(
            |body| split_frames(body, "\n\n"),
            |frames| join_frames(frames, "\n\n"),
            |frame_type, payload| format!(": vendor frame\nevent: {frame_type}\ndata: {payload}"),
        )
        .filler([": keepalive", "retry: 3000"])
        .reframing("CRLF line endings", crlf)
        .reframing("data split over several `data:` lines", multi_line_data)
    }

    /// Newline-delimited JSON: one frame per line, a vendor frame written by
    /// `vendor_frame` (one JSON line of the given type, with the payload's
    /// fields, the way the adapter's wire format names a line's type),
    /// blank lines as filler, and `CRLF` line endings as a reframing.
    ///
    /// A wire format whose lines carry no type (each line is told apart by
    /// its fields, `{"delta": ..}` or `{"done": true}`) needs no invented
    /// type field: write the payload as the line and ignore the type. The
    /// payload's fields are none of the adapter's, so the decoder must take
    /// a line it does not recognize for a vendor frame and surface it under
    /// a name of its own (`VendorFrames::surface_payload("line", json)`).
    ///
    /// ```
    /// use allama_conformance::stream_frames::StreamFraming;
    ///
    /// let untyped = StreamFraming::ndjson(|_frame_type, payload| payload.to_string());
    /// # let _ = untyped;
    /// ```
    #[must_use]
    pub fn ndjson(vendor_frame: impl Fn(&str, &Value) -> String + 'static) -> Self {
        Self::new(
            |body| split_frames(body, "\n"),
            |frames| join_frames(frames, "\n"),
            vendor_frame,
        )
        .filler(["", " "])
        .reframing("CRLF line endings", crlf)
    }

    /// Add frames that carry no data (keep-alives), which the contract
    /// injects around the vendor frame and between every frame.
    #[must_use]
    pub fn filler<S: Into<String>>(mut self, frames: impl IntoIterator<Item = S>) -> Self {
        self.filler.extend(frames.into_iter().map(Into::into));
        self
    }

    /// Add another way to frame the same body, which must decode to exactly
    /// the same events, labelled for failure messages.
    #[must_use]
    pub fn reframing(
        mut self,
        label: &'static str,
        reframe: impl Fn(&str) -> String + 'static,
    ) -> Self {
        self.reframings.push((label, Box::new(reframe)));
        self
    }

    /// The base body with a vendor frame and filler injected, labelled by
    /// position, with the markers the collected provider data must hold.
    fn injected(&self, base: &str) -> Vec<(&'static str, String, Vec<String>)> {
        let frames = (self.split)(base);
        let payload = json!({"marker": VENDOR_MARKER, "results": [{"url": "https://example.com"}]});
        let vendor = (self.vendor_frame)(VENDOR_EVENT, &payload);
        let repeated: Vec<String> = (1..=2).map(|n| format!("{VENDOR_MARKER}-{n}")).collect();
        let filler = self.filler.first().cloned();

        let mut leading = self.filler.clone();
        leading.push(vendor.clone());
        leading.extend(frames.iter().cloned());

        let mid = frames.len().div_ceil(2);
        let mut mid_stream = frames[..mid].to_vec();
        mid_stream.push(vendor.clone());
        mid_stream.extend(self.filler.iter().cloned());
        mid_stream.extend(frames[mid..].iter().cloned());

        let mut interleaved = Vec::new();
        for (position, frame) in frames.iter().enumerate() {
            interleaved.extend(filler.clone());
            interleaved.push(frame.clone());
            if position == 0 {
                interleaved.push(vendor.clone());
            }
        }

        let mut twice = frames[..mid].to_vec();
        for marker in &repeated {
            twice.push((self.vendor_frame)(
                VENDOR_EVENT,
                &json!({"marker": marker}),
            ));
        }
        twice.extend(frames[mid..].iter().cloned());

        let single = || vec![VENDOR_MARKER.to_owned()];
        vec![
            ("before the first frame", (self.join)(&leading), single()),
            ("mid-stream", (self.join)(&mid_stream), single()),
            (
                "after the first frame, filler between every frame",
                (self.join)(&interleaved),
                single(),
            ),
            (
                "twice mid-stream, with different data",
                (self.join)(&twice),
                repeated,
            ),
        ]
    }
}

impl fmt::Debug for StreamFraming {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let reframings: Vec<_> = self.reframings.iter().map(|(label, _)| label).collect();
        f.debug_struct("StreamFraming")
            .field("filler", &self.filler)
            .field("reframings", &reframings)
            .finish_non_exhaustive()
    }
}

/// Splits a body into its frames.
type Split = Box<dyn Fn(&str) -> Vec<String>>;

/// Joins frames into a body.
type Join = Box<dyn Fn(&[String]) -> String>;

/// Writes a vendor frame of a type, carrying a payload.
type WriteVendorFrame = Box<dyn Fn(&str, &Value) -> String>;

/// Frames a body differently.
type Reframe = Box<dyn Fn(&str) -> String>;

/// One decoded stream: its events (errors as their log view) and the
/// response they collect to.
struct Run {
    events: Vec<(bool, Value)>,
    response: ChatResponse,
}

impl Run {
    async fn collect<S>(stream: allama::Result<S>) -> Self
    where
        S: Stream<Item = allama::Result<StreamEvent>>,
    {
        let mut stream = Box::pin(stream.expect("unknown_stream_frames: chat_stream failed"));
        let mut collector = StreamCollector::new();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(event) => {
                    let allowed = matches!(
                        event,
                        StreamEvent::Warning { .. } | StreamEvent::ResponseProviderData { .. }
                    );
                    events.push((
                        allowed,
                        serde_json::to_value(&event).expect("serialize stream event"),
                    ));
                    collector
                        .push(event)
                        .expect("unknown_stream_frames: stream violated the event protocol");
                }
                Err(error) => events.push((
                    false,
                    json!({"error": serde_json::to_value(error.as_log()).expect("serialize error")}),
                )),
            }
        }
        let response = collector
            .finish_partial()
            .expect("unknown_stream_frames: collect response")
            .response;
        Self { events, response }
    }

    /// Events other than warnings and response-level provider data.
    fn portable(&self) -> Vec<&Value> {
        self.events
            .iter()
            .filter(|(allowed, _)| !allowed)
            .map(|(_, event)| event)
            .collect()
    }

    fn response_start(&self) -> Option<&Value> {
        self.events
            .iter()
            .map(|(_, event)| event)
            .find(|event| event["event"] == "response_start")
    }
}

/// The frames of `body`, separated by `separator` after `CRLF` endings are
/// normalized; blank frames are dropped.
fn split_frames(body: &str, separator: &str) -> Vec<String> {
    body.replace("\r\n", "\n")
        .split(separator)
        .map(|frame| frame.trim_matches('\n'))
        .filter(|frame| !frame.trim().is_empty())
        .map(str::to_owned)
        .collect()
}

/// `frames` separated by `separator`, the last one terminated by it.
fn join_frames(frames: &[String], separator: &str) -> String {
    let mut body = frames.join(separator);
    body.push_str(separator);
    body
}

/// `body` with `CRLF` line endings.
fn crlf(body: &str) -> String {
    body.replace("\r\n", "\n").replace('\n', "\r\n")
}

/// `body` with each JSON `data:` field split over several `data:` lines.
fn multi_line_data(body: &str) -> String {
    let normalized = body.replace("\r\n", "\n");
    let mut multi_line = String::with_capacity(normalized.len() * 2);
    for line in normalized.split_inclusive('\n') {
        let (content, newline) = match line.strip_suffix('\n') {
            Some(content) => (content, "\n"),
            None => (line, ""),
        };
        let data = content
            .strip_prefix("data:")
            .map(|data| data.strip_prefix(' ').unwrap_or(data))
            .filter(|data| serde_json::from_str::<Value>(data).is_ok());
        match data {
            Some(data) => {
                let parts = split_after_commas(data);
                let last = parts.len() - 1;
                for (position, part) in parts.into_iter().enumerate() {
                    multi_line.push_str("data: ");
                    multi_line.push_str(part);
                    multi_line.push_str(if position == last { newline } else { "\n" });
                }
            }
            None => multi_line.push_str(line),
        }
    }
    multi_line
}

/// JSON text split after each comma outside a string. Joined with `\n`, the
/// parts are the same JSON value.
fn split_after_commas(json: &str) -> Vec<&str> {
    let (mut parts, mut start) = (Vec::new(), 0);
    let (mut in_string, mut escaped) = (false, false);
    for (i, c) in json.char_indices() {
        match c {
            _ if escaped => escaped = false,
            '\\' if in_string => escaped = true,
            '"' => in_string = !in_string,
            ',' if !in_string => {
                parts.push(&json[start..=i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&json[start..]);
    parts
}

#[cfg(test)]
mod tests {
    use allama::chat::stream::StreamBlockType;
    use allama::{Error, StreamErrorKind};
    use allama_http::TransportError;
    use allama_http::framing::{ChatFrameDecoder, FramedChatStream, VendorFrames};
    use allama_http::ndjson::NdjsonStream;
    use futures_util::FutureExt;

    use super::*;

    /// `{"type": "text"}` and `{"type": "done"}` lines; any other type is a
    /// vendor frame. `finish` says whether it yields `VendorFrames::finish`.
    struct Lines {
        frames: VendorFrames,
        started: bool,
        done: bool,
        finish: bool,
    }

    impl ChatFrameDecoder<String> for Lines {
        fn frame(&mut self, line: String) -> Vec<allama::Result<StreamEvent>> {
            let json: Value = serde_json::from_str(&line).expect("a JSON line");
            let mut out = Vec::new();
            if !self.started {
                self.started = true;
                out.push(StreamEvent::response_start(None, None));
                out.push(StreamEvent::block_start(0, StreamBlockType::Text));
            }
            match json["type"].as_str() {
                Some("text") => out.push(StreamEvent::text_delta(0, "Hi")),
                Some("done") => {
                    self.done = true;
                    out.push(StreamEvent::block_stop(0));
                    if self.finish {
                        out.extend(self.frames.finish());
                    }
                    out.push(StreamEvent::ResponseStop);
                }
                Some(other) => out.extend(self.frames.surface_payload(other, json.clone())),
                None => panic!("untyped line {line}"),
            }
            out.into_iter().map(Ok).collect()
        }

        fn end(&mut self) -> Vec<allama::Result<StreamEvent>> {
            vec![Err(Error::stream(
                StreamErrorKind::Incomplete,
                "ended early",
            ))]
        }

        fn is_done(&self) -> bool {
            self.done
        }
    }

    async fn run_lines(finish: bool) {
        let framing = StreamFraming::ndjson(|frame_type, payload| {
            let mut line = payload.clone();
            line["type"] = json!(frame_type);
            line.to_string()
        });
        let base = "{\"type\":\"text\"}\n{\"type\":\"text\"}\n{\"type\":\"done\"}\n";
        assert_unknown_stream_frames_surfaced(base, &framing, |body| async move {
            let body = futures_util::stream::iter([Ok::<_, TransportError>(body.into_bytes())]);
            Ok(FramedChatStream::new(
                NdjsonStream::new(body),
                Lines {
                    frames: VendorFrames::new("lines", "lines"),
                    started: false,
                    done: false,
                    finish,
                },
            ))
        })
        .await;
    }

    #[tokio::test]
    async fn a_decoder_that_finishes_its_vendor_frames_passes() {
        run_lines(true).await;
    }

    /// Without `VendorFrames::finish`, a repeated frame type keeps only its
    /// last payload.
    #[tokio::test]
    async fn a_decoder_that_never_finishes_its_vendor_frames_fails() {
        let panic = std::panic::AssertUnwindSafe(run_lines(false))
            .catch_unwind()
            .await
            .expect_err("the contract catches a missing `finish`");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .unwrap_or_default();
        assert!(message.contains("twice mid-stream"), "{message}");
        assert!(message.contains(&format!("{VENDOR_MARKER}-1")), "{message}");
    }

    fn vendor_payload() -> Value {
        json!({"marker": VENDOR_MARKER, "results": [{"url": "https://example.com"}]})
    }

    #[test]
    fn sse_injects_around_the_base_frames() {
        let variants = StreamFraming::sse().injected("data: a\n\ndata: b\n\ndata: c\n\n");
        assert_eq!(
            variants[0].1,
            format!(
                ": keepalive\n\nretry: 3000\n\n: vendor frame\nevent: x-vendor\ndata: {}\n\ndata: a\n\ndata: b\n\ndata: c\n\n",
                vendor_payload()
            )
        );
        assert!(
            variants[1]
                .1
                .starts_with("data: a\n\ndata: b\n\n: vendor frame")
        );
        assert!(
            variants[2]
                .1
                .starts_with(": keepalive\n\ndata: a\n\n: vendor frame\nevent: x-vendor\n")
        );
        assert!(
            variants[2]
                .1
                .ends_with(": keepalive\n\ndata: b\n\n: keepalive\n\ndata: c\n\n")
        );
    }

    #[test]
    fn sse_reframings_keep_the_events() {
        let base = "event: delta\ndata: {\"text\":\"a, b\\\"\",\"n\":[1,2]}\n\ndata: [DONE]\n\n";
        let framing = StreamFraming::sse();
        let reframed: Vec<_> = framing
            .reframings
            .iter()
            .map(|(_, reframe)| reframe(base))
            .collect();
        assert_eq!(
            reframed[0],
            "event: delta\r\ndata: {\"text\":\"a, b\\\"\",\"n\":[1,2]}\r\n\r\ndata: [DONE]\r\n\r\n"
        );
        assert_eq!(
            reframed[1],
            "event: delta\ndata: {\"text\":\"a, b\\\"\",\ndata: \"n\":[1,\ndata: 2]}\n\ndata: [DONE]\n\n"
        );
        // Joined back, the data is the same JSON.
        let joined = "{\"text\":\"a, b\\\"\",\n\"n\":[1,\n2]}";
        assert_eq!(
            serde_json::from_str::<Value>(joined).unwrap(),
            json!({"text": "a, b\"", "n": [1, 2]})
        );
    }

    #[test]
    fn ndjson_injects_the_adapter_written_line() {
        let framing = StreamFraming::ndjson(|frame_type, payload| {
            let mut line = payload.clone();
            line["type"] = json!(frame_type);
            line.to_string()
        });
        let variants = framing.injected("{\"n\":1}\r\n\n{\"n\":2}\n{\"n\":3}");
        let mut vendor = vendor_payload();
        vendor["type"] = json!(VENDOR_EVENT);
        assert_eq!(
            variants[0].1,
            format!("\n \n{vendor}\n{{\"n\":1}}\n{{\"n\":2}}\n{{\"n\":3}}\n")
        );
        assert_eq!(
            variants[1].1,
            format!("{{\"n\":1}}\n{{\"n\":2}}\n{vendor}\n\n \n{{\"n\":3}}\n")
        );
        assert_eq!(
            variants[2].1,
            format!("\n{{\"n\":1}}\n{vendor}\n\n{{\"n\":2}}\n\n{{\"n\":3}}\n")
        );
        let (label, crlf) = &framing.reframings[0];
        assert_eq!(*label, "CRLF line endings");
        assert_eq!(crlf("{}\n{}\n"), "{}\r\n{}\r\n");
    }
}
