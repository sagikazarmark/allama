//! The chat stream driver over a response's SSE events: the SSE form of
//! [`FramedChatStream`](crate::framing::FramedChatStream).

use super::SseStream;
#[cfg(doc)]
use crate::framing::ChatFrameDecoder;
use crate::framing::FramedChatStream;

/// A chat stream decoded from a response's SSE events by an adapter's
/// [`ChatFrameDecoder`]`<`[`SseEvent`](super::SseEvent)`>`: the
/// [`FramedChatStream`] over an [`SseStream`], with the driver guarantees
/// the decoder trait lists.
///
/// [`SseChatStream::new`](FramedChatStream::new) classifies a failure with
/// [`SseError::into_error`](super::SseError::into_error), for a body whose
/// error converts into a [`TransportError`](crate::TransportError), as an
/// [`HttpResponse`](crate::HttpResponse) body's does;
/// [`with_error_map`](FramedChatStream::with_error_map) takes any other
/// body, with a map that keeps the rules of `into_error`: a timeout stays a
/// timeout, invalid UTF-8 is a stream decode error, an event past the size
/// cap a stream protocol error, and the messages are the adapter's own.
///
/// It is unboxed and `Send` (for a `Send` body); box it into a
/// [`ChatStream`](allama::ChatStream) with
/// [`Prepared::finish_stream`](allama::adapter::chat::Prepared::finish_stream):
///
/// ```
/// use allama::{Error, Result, StreamEvent, StreamErrorKind};
/// use allama_http::TransportError;
/// use allama_http::framing::ChatFrameDecoder;
/// use allama_http::sse::{SseChatStream, SseEvent, SseStream};
/// use futures_util::{FutureExt, StreamExt, stream};
///
/// /// `data:` text deltas, then `event: done`.
/// struct Lines {
///     started: bool,
///     done: bool,
/// }
///
/// impl ChatFrameDecoder<SseEvent> for Lines {
///     fn frame(&mut self, event: SseEvent) -> Vec<Result<StreamEvent>> {
///         let mut out = Vec::new();
///         if !self.started {
///             self.started = true;
///             out.push(Ok(StreamEvent::response_start(None, None)));
///             out.push(Ok(StreamEvent::block_start(0, allama::StreamBlockType::Text)));
///         }
///         if event.event == "done" {
///             self.done = true;
///             out.push(Ok(StreamEvent::block_stop(0)));
///             out.push(Ok(StreamEvent::ResponseStop));
///         } else {
///             out.push(Ok(StreamEvent::text_delta(0, event.data)));
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
///     &b"data: Hel\n\ndata: lo\n\nevent: done\ndata: {}\n\n"[..],
/// )]);
/// let stream = SseChatStream::new(SseStream::new(body), Lines { started: false, done: false });
/// let events: Vec<_> = stream.collect().now_or_never().unwrap();
/// assert_eq!(events.len(), 6);
/// assert!(matches!(events[5], Ok(StreamEvent::ResponseStop)));
/// ```
pub type SseChatStream<S, D> = FramedChatStream<SseStream<S>, D>;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::sync::{Arc, Mutex};

    use allama::{ChatStream, Error, StreamErrorKind};
    use futures_util::StreamExt;

    use super::*;
    use crate::framing::ChatFrameDecoder;
    use crate::sse::{SseError, SseEvent};
    use crate::{TransportError, TransportErrorKind};
    use allama::Result;
    use allama::StreamEvent;
    use futures_core::Stream;

    /// What the decoder was called with, shared with the test.
    #[derive(Debug, Default)]
    struct Calls {
        events: Vec<String>,
        ends: usize,
        fails: Vec<Error>,
    }

    /// A decoder that yields the scripted items for each event (by data),
    /// and records its calls.
    struct Scripted {
        calls: Arc<Mutex<Calls>>,
        script: fn(&str) -> Vec<Result<StreamEvent>>,
        end: Vec<Result<StreamEvent>>,
        done_after: Option<&'static str>,
        done: bool,
    }

    impl Scripted {
        fn new(script: fn(&str) -> Vec<Result<StreamEvent>>) -> (Self, Arc<Mutex<Calls>>) {
            let calls = Arc::default();
            let decoder = Self {
                calls: Arc::clone(&calls),
                script,
                end: Vec::new(),
                done_after: None,
                done: false,
            };
            (decoder, calls)
        }
    }

    impl ChatFrameDecoder<SseEvent> for Scripted {
        fn frame(&mut self, event: SseEvent) -> Vec<Result<StreamEvent>> {
            self.calls.lock().unwrap().events.push(event.data.clone());
            self.done |= self.done_after == Some(event.data.as_str());
            (self.script)(&event.data)
        }

        fn end(&mut self) -> Vec<Result<StreamEvent>> {
            self.calls.lock().unwrap().ends += 1;
            std::mem::take(&mut self.end)
        }

        fn fail(&mut self, error: Error) -> Vec<Result<StreamEvent>> {
            self.calls.lock().unwrap().fails.push(error.clone());
            vec![Err(error)]
        }

        fn is_done(&self) -> bool {
            self.done
        }
    }

    fn delta(data: &str) -> Vec<Result<StreamEvent>> {
        vec![Ok(StreamEvent::text_delta(0, data))]
    }

    /// A body of `chunks`, which counts its polls.
    fn body(
        chunks: Vec<std::result::Result<&'static str, TransportError>>,
        polls: &Arc<Mutex<usize>>,
    ) -> impl Stream<Item = std::result::Result<&'static str, TransportError>> + Send + Unpin + 'static
    {
        let polls = Arc::clone(polls);
        futures_util::stream::iter(chunks).inspect(move |_| *polls.lock().unwrap() += 1)
    }

    async fn run<D: ChatFrameDecoder<SseEvent>>(
        chunks: Vec<std::result::Result<&'static str, TransportError>>,
        decoder: D,
    ) -> (Vec<Result<StreamEvent>>, usize) {
        let polls = Arc::default();
        let stream = SseChatStream::new(SseStream::new(body(chunks, &polls)), decoder);
        let items: Vec<_> = stream.collect().await;
        let polls = *polls.lock().unwrap();
        (items, polls)
    }

    #[tokio::test]
    async fn end_runs_once_when_the_body_ends() {
        let (mut decoder, calls) = Scripted::new(delta);
        decoder.end = vec![Ok(StreamEvent::ResponseStop)];
        let (items, _) = run(vec![Ok("data: a\n\ndata: b\n\n")], decoder).await;
        assert_eq!(items.len(), 3);
        assert!(matches!(items[2], Ok(StreamEvent::ResponseStop)));
        let calls = calls.lock().unwrap();
        assert_eq!(calls.events, ["a", "b"]);
        assert_eq!(calls.ends, 1);
        assert!(calls.fails.is_empty());
    }

    #[tokio::test]
    async fn an_empty_end_terminates() {
        let (decoder, calls) = Scripted::new(delta);
        let (items, _) = run(vec![Ok("data: a\n\n")], decoder).await;
        assert_eq!(items.len(), 1);
        assert_eq!(calls.lock().unwrap().ends, 1);
    }

    #[tokio::test]
    async fn fail_runs_once_when_the_body_fails_and_nothing_follows() {
        let (decoder, calls) = Scripted::new(delta);
        let (items, polls) = run(
            vec![
                Ok("data: a\n\n"),
                Err(TransportError::new(TransportErrorKind::Body, "reset")),
                Ok("data: b\n\n"),
            ],
            decoder,
        )
        .await;
        assert_eq!(items.len(), 2);
        let error = items[1].as_ref().unwrap_err();
        assert!(matches!(
            error,
            Error::Stream {
                kind: StreamErrorKind::Transport,
                ..
            }
        ));
        assert_eq!(error.upstream_message(), Some("reset"));
        assert_eq!(polls, 2, "the body is not polled after it failed");
        let calls = calls.lock().unwrap();
        assert_eq!(calls.fails.len(), 1);
        assert_eq!(calls.ends, 0);
    }

    #[tokio::test]
    async fn a_timeout_stays_a_timeout() {
        let (decoder, _) = Scripted::new(delta);
        let (items, _) = run(
            vec![Err(TransportError::new(
                TransportErrorKind::Timeout,
                "idle",
            ))],
            decoder,
        )
        .await;
        assert!(matches!(items[..], [Err(Error::Timeout { .. })]));
    }

    #[tokio::test]
    async fn the_stream_ends_after_a_decoder_error() {
        let (decoder, calls) = Scripted::new(|data| match data {
            "bad" => vec![
                Ok(StreamEvent::text_delta(0, "before")),
                Err(Error::stream_decode("bad")),
                Ok(StreamEvent::text_delta(0, "dropped")),
            ],
            data => delta(data),
        });
        let (items, polls) = run(
            vec![
                Ok("data: a\n\n"),
                Ok("data: bad\n\n"),
                Ok("data: c\n\n"),
                Ok("data: d\n\n"),
            ],
            decoder,
        )
        .await;
        assert_eq!(items.len(), 3, "{items:?}");
        assert!(items[2].is_err());
        assert_eq!(polls, 2, "the body is not polled after the error");
        let calls = calls.lock().unwrap();
        assert_eq!(calls.events, ["a", "bad"]);
        assert_eq!(calls.ends, 0);
        assert!(calls.fails.is_empty());
    }

    #[tokio::test]
    async fn the_stream_ends_once_the_decoder_is_done() {
        let (mut decoder, calls) = Scripted::new(delta);
        decoder.done_after = Some("last");
        let (items, polls) = run(
            vec![
                Ok("data: a\n\ndata: last\n\n"),
                Ok("data: ignored\n\n"),
                Err(TransportError::new(TransportErrorKind::Body, "never read")),
            ],
            decoder,
        )
        .await;
        assert_eq!(items.len(), 2);
        assert_eq!(polls, 1);
        let calls = calls.lock().unwrap();
        assert_eq!(calls.events, ["a", "last"]);
        assert_eq!(calls.ends, 0);
        assert!(calls.fails.is_empty());
    }

    #[tokio::test]
    async fn sse_failures_are_classified() {
        let polls = Arc::default();
        let invalid = futures_util::stream::iter([Ok::<_, TransportError>(&b"data: \xFF\n\n"[..])]);
        let (decoder, _) = Scripted::new(delta);
        let items: Vec<_> = SseChatStream::new(SseStream::new(invalid), decoder)
            .collect()
            .await;
        let error = items[0].as_ref().unwrap_err();
        assert!(matches!(
            error,
            Error::Stream {
                kind: StreamErrorKind::Decode,
                ..
            }
        ));
        assert!(!error.is_retryable());
        assert_eq!(error.message_origin(), allama::MessageOrigin::Authored);

        let (decoder, _) = Scripted::new(delta);
        let events =
            SseStream::new(body(vec![Ok("data: 0123456789abcdef!\n\n")], &polls)).max_event_size(8);
        let items: Vec<_> = SseChatStream::new(events, decoder).collect().await;
        assert!(matches!(
            items[..],
            [Err(Error::Stream {
                kind: StreamErrorKind::Protocol,
                ..
            })]
        ));
    }

    #[tokio::test]
    async fn an_error_map_classifies_other_bodies() {
        let body = futures_util::stream::iter([
            Ok::<_, String>("data: a\n\n"),
            Err("platform failure".to_owned()),
        ]);
        let (decoder, _) = Scripted::new(delta);
        let stream =
            SseChatStream::with_error_map(SseStream::new(body), decoder, |error| match error {
                SseError::Transport(text) => Error::stream(StreamErrorKind::Transport, "failed")
                    .authored()
                    .with_upstream_message(text),
                other => Error::stream_decode(other.to_string()).authored(),
            });
        let items: Vec<_> = stream.collect().await;
        assert_eq!(
            items[1].as_ref().unwrap_err().upstream_message(),
            Some("platform failure")
        );
    }

    #[test]
    fn the_stream_is_send() {
        fn send<T: Send + 'static>() {}
        send::<SseChatStream<crate::Body, Scripted>>();
        let (decoder, _) = Scripted::new(delta);
        let events = SseStream::new(futures_util::stream::empty::<
            std::result::Result<bytes::Bytes, TransportError>,
        >());
        let _: ChatStream = Box::pin(SseChatStream::new(events, decoder));
    }
}
