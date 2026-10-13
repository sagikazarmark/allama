//! How a stream ends: cut short, or complete without a finish reason.
//!
//! A stream that closes before its terminal marker (`[DONE]`,
//! `message_stop`, `response.completed`, ...) was cut short: the adapter
//! fails it with [`StreamErrorKind::Incomplete`], which is retryable, and
//! never ends it quietly. The check reads the stream as a caller does, not
//! collected (collecting reports a quiet end incomplete by itself), so its
//! last item must be the error;
//! [`Prepared::finish_stream`](allama::adapter::chat::Prepared::finish_stream) adds
//! one to a stream that ends without `ResponseStop`. A stream that ends with its marker but reported
//! no finish reason is complete: it collects to `Ok` with no finish reason
//! and one warning coded [`MISSING_FINISH_REASON`], which
//! [`Prepared::finish_stream`](allama::adapter::chat::Prepared::finish_stream) adds,
//! so [`finished_normally`](allama::ChatResponse::finished_normally) is
//! `false`. [`assert_stream_ends_classified`] checks both.
//!
//! ```
//! use allama::adapter::chat::{self, MISSING_FINISH_REASON};
//! use allama::mock::MockProvider;
//! use allama::{ChatRequest, ChatStream, Error, StreamErrorKind, StreamEvent};
//! use allama_conformance::stream_end::{StreamEnds, assert_stream_ends_classified};
//!
//! /// A toy wire format: one line per event, `end` as the terminal marker.
//! fn decode(body: &str) -> Vec<allama::Result<StreamEvent>> {
//!     let mut events = vec![Ok(StreamEvent::response_start(None, None))];
//!     if body.ends_with("end") {
//!         events.push(Ok(StreamEvent::ResponseStop));
//!     } else {
//!         events.push(Err(Error::stream(StreamErrorKind::Incomplete, "no end")));
//!     }
//!     events
//! }
//!
//! # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
//! let provider = MockProvider::empty();
//! let request = ChatRequest::new("m").user("hi");
//! assert_stream_ends_classified(
//!     StreamEnds::new("start").terminated_without_finish("start\nend"),
//!     |body| {
//!         // What the adapter's `chat_stream` does: decode, then finish.
//!         let prepared = chat::prepare(&provider, &request);
//!         async move {
//!             let stream = futures_util::stream::iter(decode(&body));
//!             Ok::<ChatStream, Error>(prepared?.finish_stream(stream))
//!         }
//!     },
//! )
//! .await;
//! # });
//! ```

use std::future::Future;

use allama::adapter::chat::MISSING_FINISH_REASON;
use allama::{ChatStreamExt, Error, StreamErrorKind, StreamEvent};
use futures_core::Stream;
use futures_util::StreamExt;

/// The stream bodies [`assert_stream_ends_classified`] decodes.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StreamEnds {
    /// A body cut short: it ends before the terminal marker.
    pub unterminated: String,
    /// A body that ends with its terminal marker without reporting a
    /// finish reason, or `None` when the wire format has no marker apart
    /// from the finish reason (Gemini), or documents that a marker alone
    /// means a normal stop (older Workers AI runtimes).
    pub terminated_without_finish: Option<String>,
}

impl StreamEnds {
    /// The ends of a wire format whose only check is the cut-short body.
    #[must_use]
    pub fn new(unterminated: impl Into<String>) -> Self {
        Self {
            unterminated: unterminated.into(),
            terminated_without_finish: None,
        }
    }

    /// Add a body that ends with the terminal marker but no finish reason.
    #[must_use]
    pub fn terminated_without_finish(mut self, body: impl Into<String>) -> Self {
        self.terminated_without_finish = Some(body.into());
        self
    }
}

/// Assert that a stream cut short ends with a
/// [`StreamErrorKind::Incomplete`] error, read item by item as a caller
/// does, and that a stream that ends with its
/// terminal marker but no finish reason collects to `Ok`, with no finish
/// reason and exactly one warning coded [`MISSING_FINISH_REASON`].
///
/// `probe` sends a request whose transport answers `200` with the given
/// body and returns the adapter's stream (`chat_stream`'s result), so the
/// stream goes through the adapter's
/// [`finish_stream`](allama::adapter::chat::Prepared::finish_stream).
///
/// # Panics
///
/// When the cut-short stream ends without an error, collects, or fails with
/// another error, or the terminated one fails, reports a finish reason, or does not carry the
/// warning once.
pub async fn assert_stream_ends_classified<F, Fut, S>(ends: StreamEnds, mut probe: F)
where
    F: FnMut(String) -> Fut,
    Fut: Future<Output = allama::Result<S>>,
    S: Stream<Item = allama::Result<StreamEvent>> + Send + 'static,
{
    let mut problems = Vec::new();
    let items = drain(probe(ends.unterminated.clone()).await).await;
    match items.last() {
        Some(Err(Error::Stream {
            kind: StreamErrorKind::Incomplete,
            ..
        })) => {}
        Some(Err(other)) => problems.push(format!(
            "a stream cut short before its terminal marker fails with {} ({other}), \
             not a stream error of kind Incomplete",
            other.kind_name()
        )),
        _ if items
            .iter()
            .any(|item| matches!(item, Ok(StreamEvent::ResponseStop))) =>
        {
            let collected = futures_util::stream::iter(items).collect_response().await;
            let finish_reason = collected.ok().and_then(|response| response.finish_reason);
            problems.push(format!(
                "a stream cut short before its terminal marker collects (finish reason \
                 {finish_reason:?}); it must fail with a stream error of kind Incomplete"
            ));
        }
        _ => problems.push(format!(
            "a stream cut short before its terminal marker ends without an error after {} \
             item(s), so a caller reading it takes it for complete; it must end with a \
             stream error of kind Incomplete (does the stream go through \
             `Prepared::finish_stream`?)",
            items.len()
        )),
    }
    if let Some(body) = ends.terminated_without_finish {
        match collect(probe(body).await).await {
            Ok(response) => {
                if let Some(reason) = &response.finish_reason {
                    problems.push(format!(
                        "a terminated stream without a finish reason reports `{reason}`"
                    ));
                }
                let warned = response
                    .warnings
                    .iter()
                    .filter(|w| w.code.as_deref() == Some(MISSING_FINISH_REASON))
                    .count();
                if warned != 1 {
                    problems.push(format!(
                        "a terminated stream without a finish reason carries {warned} \
                         `{MISSING_FINISH_REASON}` warnings, not 1 (does the stream go \
                         through `Prepared::finish_stream`?)"
                    ));
                }
            }
            Err(error) => problems.push(format!(
                "a stream that ends with its terminal marker but no finish reason fails \
                 with {} ({error}); it must collect, with a warning",
                error.kind_name()
            )),
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// Every item of the stream, or the error that kept it from starting.
async fn drain<S>(stream: allama::Result<S>) -> Vec<allama::Result<StreamEvent>>
where
    S: Stream<Item = allama::Result<StreamEvent>> + Send + 'static,
{
    match stream {
        Ok(stream) => stream.collect().await,
        Err(error) => vec![Err(error)],
    }
}

async fn collect<S>(stream: allama::Result<S>) -> allama::Result<allama::ChatResponse>
where
    S: Stream<Item = allama::Result<StreamEvent>> + Send + 'static,
{
    let stream: allama::ChatStream = Box::pin(stream?);
    stream.collect_response().await
}

#[cfg(test)]
mod tests {
    use allama::adapter::chat;
    use allama::mock::MockProvider;
    use allama::{ChatRequest, ChatStream, FinishReason};

    use super::*;

    fn probe(
        events: Vec<allama::Result<StreamEvent>>,
    ) -> impl FnMut(String) -> std::future::Ready<allama::Result<ChatStream>> {
        move |_| {
            let provider = MockProvider::empty();
            let request = ChatRequest::new("m").user("hi");
            let prepared = chat::prepare(&provider, &request).unwrap();
            std::future::ready(Ok(
                prepared.finish_stream(futures_util::stream::iter(events.clone()))
            ))
        }
    }

    fn start() -> allama::Result<StreamEvent> {
        Ok(StreamEvent::response_start(None, None))
    }

    #[tokio::test]
    #[should_panic(expected = "collects")]
    async fn a_stop_before_the_marker_fails() {
        assert_stream_ends_classified(
            StreamEnds::new("x"),
            probe(vec![start(), Ok(StreamEvent::ResponseStop)]),
        )
        .await;
    }

    /// A decoder that ends quietly, read without `finish_stream`: a caller
    /// of the raw stream would take it for complete, though collecting it
    /// reports it incomplete.
    #[tokio::test]
    #[should_panic(expected = "ends without an error")]
    async fn a_quiet_end_outside_finish_stream_fails() {
        assert_stream_ends_classified(StreamEnds::new("x"), |_| {
            std::future::ready(Ok(
                Box::pin(futures_util::stream::iter(vec![start()])) as ChatStream
            ))
        })
        .await;
    }

    /// The same decoder through `finish_stream` passes: the layer fails the
    /// stream with `Incomplete`.
    #[tokio::test]
    async fn a_quiet_end_through_finish_stream_passes() {
        assert_stream_ends_classified(StreamEnds::new("x"), probe(vec![start()])).await;
    }

    #[tokio::test]
    #[should_panic(expected = "reports `stop`")]
    async fn a_made_up_finish_reason_fails() {
        let mut metadata = StreamEvent::metadata();
        if let StreamEvent::ResponseMetadata { finish_reason, .. } = &mut metadata {
            *finish_reason = Some(FinishReason::Stop);
        }
        let probe = probe(vec![start(), Ok(metadata), Ok(StreamEvent::ResponseStop)]);
        let mut cut_short = true;
        let mut probe = probe;
        assert_stream_ends_classified(
            StreamEnds::new("x").terminated_without_finish("y"),
            move |body| {
                if std::mem::take(&mut cut_short) {
                    let error = Error::stream(StreamErrorKind::Incomplete, "cut short");
                    return std::future::ready(Ok(Box::pin(futures_util::stream::iter(vec![Err(
                        error,
                    )])) as ChatStream));
                }
                probe(body)
            },
        )
        .await;
    }
}
