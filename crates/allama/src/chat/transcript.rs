//! Stream transcripts: what a stream delivered, including how it ended.

use futures_core::Stream;
use serde::{Deserialize, Serialize};

use crate::chat::stream::{CollectedResponse, StreamCollector};
use crate::{Error, Result, StreamEvent};

/// The events a chat stream delivered, in order, and the error that ended
/// it, if any.
///
/// A [`ChatResponse`](crate::ChatResponse) is what a stream *assembles to*;
/// a transcript is what *arrived*. It keeps what a response cannot: partial
/// output followed by a failure, warnings in their original position, and
/// [`ProviderDelta`](StreamEvent::ProviderDelta) events. Transcripts are
/// serializable, so they work as debugging artifacts and as fixtures to
/// replay through a mock (see "Replaying a recorded call" in the
/// `allama::mock` docs, feature `mock`).
///
/// A transcript is a same-version artifact, like a
/// [`ChatExchange`](crate::wrappers::ChatExchange#stability): it carries no
/// format version, and one recorded by another allama version may not
/// load. Re-record it after upgrading, and keep conversations in a
/// [`History`](crate::history::History).
///
/// Capture one for every call with
/// [`RecordingProvider`](crate::wrappers::RecordingProvider), or build one
/// item by item with [`record`](Self::record).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StreamTranscript {
    /// Events in arrival order.
    pub events: Vec<StreamEvent>,
    /// The error that ended the stream.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_error: Option<Error>,
}

impl StreamTranscript {
    /// An empty transcript.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one stream item. An error becomes the terminal error.
    pub fn record(&mut self, item: Result<StreamEvent>) {
        match item {
            Ok(event) => self.events.push(event),
            Err(error) => self.terminal_error = Some(error),
        }
    }

    /// Assemble the events into a response, keeping the terminal error and
    /// completeness flags (the same result as
    /// [`with_collector`](crate::ChatStreamExt::with_collector) and
    /// [`finish_partial`](crate::chat::stream::WithCollector::finish_partial)
    /// on the original stream).
    ///
    /// # Errors
    ///
    /// Fails when the events violate the stream protocol.
    pub fn to_collected(&self) -> Result<CollectedResponse> {
        let mut collector = StreamCollector::new();
        for event in &self.events {
            collector.push_ref(event)?;
        }
        if let Some(error) = &self.terminal_error {
            collector.set_terminal_error(error.clone());
        }
        collector.finish_partial()
    }

    /// Replay the transcript as a stream: every event, then the terminal
    /// error.
    ///
    /// The replayed warnings are marked as emitted (see
    /// [warning events](crate::warning#warning-events)): they describe a
    /// call that was logged when it happened, so a wrapper that passes them
    /// through [`Prepared::finish_stream`](crate::adapter::chat::Prepared::finish_stream)
    /// does not log them again.
    pub fn replay(self) -> impl Stream<Item = Result<StreamEvent>> + Send + Unpin + 'static {
        futures_util::stream::iter(
            self.events
                .into_iter()
                .map(|mut event| {
                    mark_warning_emitted(&mut event);
                    Ok(event)
                })
                .chain(self.terminal_error.map(Err)),
        )
    }
}

/// Mark the warning `event` carries, if any, as emitted.
pub(crate) fn mark_warning_emitted(event: &mut StreamEvent) {
    if let StreamEvent::Warning { warning } = event {
        warning.mark_emitted();
    }
}

/// Record every item of `stream` into a transcript, up to the first error.
#[cfg(test)]
async fn collect_transcript<S>(stream: S) -> StreamTranscript
where
    S: Stream<Item = Result<StreamEvent>> + Send,
{
    use futures_util::StreamExt;
    let mut transcript = StreamTranscript::new();
    let mut stream = std::pin::pin!(stream);
    while let Some(item) = stream.next().await {
        let failed = item.is_err();
        transcript.record(item);
        if failed {
            break;
        }
    }
    transcript
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChatResponse, ChatStreamExt, ContentBlock};
    use futures_util::StreamExt;

    fn response() -> ChatResponse {
        ChatResponse::new(vec![ContentBlock::text("hello")]).id("r1")
    }

    #[tokio::test]
    async fn complete_streams_round_trip() {
        let transcript = collect_transcript(response().into_stream()).await;
        assert!(transcript.terminal_error.is_none());
        let collected = transcript.to_collected().unwrap();
        assert!(collected.completeness.is_complete());
        assert_eq!(collected.response, response());

        let value = serde_json::to_value(&transcript).unwrap();
        assert!(value.get("terminal_error").is_none());
        let back: StreamTranscript = serde_json::from_value(value).unwrap();
        assert_eq!(back.events, transcript.events);
        assert_eq!(back.replay().collect_response().await.unwrap(), response());
    }

    #[tokio::test]
    async fn partial_output_and_the_terminal_error_are_kept() {
        let stream = futures_util::stream::iter(vec![
            Ok(StreamEvent::ResponseStart {
                id: None,
                model: None,
            }),
            Ok(StreamEvent::TextDelta {
                index: 0,
                text: "par".into(),
            }),
            Err(Error::overloaded("busy")),
            Ok(StreamEvent::ResponseStop),
        ]);
        let transcript = collect_transcript(stream).await;
        assert_eq!(transcript.events.len(), 2, "recording stops at the error");

        let value = serde_json::to_value(&transcript).unwrap();
        assert_eq!(value["terminal_error"]["type"], "overloaded");
        let back: StreamTranscript = serde_json::from_value(value).unwrap();

        let collected = back.to_collected().unwrap();
        assert_eq!(collected.response.text().as_deref(), Some("par"));
        assert!(!collected.completeness.is_complete());
        assert!(matches!(
            collected.terminal_error,
            Some(Error::Overloaded { .. })
        ));

        let replayed: Vec<_> = back.replay().collect().await;
        assert_eq!(replayed.len(), 3);
        assert!(replayed[2].is_err());
    }

    /// A replayed warning was logged when the call happened: it is marked,
    /// though the deserialized transcript's warning was not.
    #[tokio::test]
    async fn replayed_warnings_are_marked_emitted() {
        let warning = crate::Warning::new(crate::WarningKind::Other, crate::Subject::Response, "w");
        let transcript = StreamTranscript {
            events: vec![StreamEvent::warning(warning)],
            terminal_error: None,
        };
        let back: StreamTranscript =
            serde_json::from_value(serde_json::to_value(&transcript).unwrap()).unwrap();
        let StreamEvent::Warning { warning } = &back.events[0] else {
            panic!("a warning event");
        };
        assert!(!crate::warning::was_emitted(warning));
        let replayed: Vec<_> = back.replay().collect().await;
        let Ok(StreamEvent::Warning { warning }) = &replayed[0] else {
            panic!("a warning event");
        };
        assert!(crate::warning::was_emitted(warning));
    }
}
