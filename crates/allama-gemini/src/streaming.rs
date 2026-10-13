//! SSE transport for `streamGenerateContent?alt=sse`.
//!
//! This module only frames the byte stream into SSE events and parses each
//! event's JSON; all decoding happens in the shared [`Decoder`]. Gemini
//! sends unnamed events only; a named event (from a proxy or a future API
//! version) is never decoded as a chunk but surfaced verbatim.

use allama::{Error, Result, StreamEvent, Warning};
use allama_http::TransportError;
use allama_http::framing::ChatFrameDecoder;
use allama_http::sse::{SseChatStream, SseEvent, SseStream};

use crate::PROVIDER_NAME;
use crate::decode::{Decoder, End};
use crate::wire::GenerateContentResponse;

/// Decode an SSE stream with the shared [`Decoder`]. `warnings` from
/// request mapping are emitted right after `ResponseStart`.
pub(crate) fn sse_to_stream<S, B, E>(
    events: SseStream<S>,
    warnings: Vec<Warning>,
    request_id: Option<String>,
) -> SseChatStream<S, StreamDecoder>
where
    S: futures_core::Stream<Item = std::result::Result<B, E>> + Unpin,
    B: AsRef<[u8]>,
    E: Into<TransportError> + 'static,
{
    let decoder = StreamDecoder {
        decoder: Decoder::new(warnings).with_request_id(request_id),
    };
    SseChatStream::new(events, decoder)
}

/// The shared [`Decoder`] over SSE events.
pub(crate) struct StreamDecoder {
    decoder: Decoder,
}

impl ChatFrameDecoder<SseEvent> for StreamDecoder {
    fn frame(&mut self, event: SseEvent) -> Vec<Result<StreamEvent>> {
        if event.event != "message" {
            let events = self.decoder.unknown_event(&event.event, &event.data);
            return events.into_iter().map(Ok).collect();
        }
        if event.data.trim().is_empty() {
            return Vec::new();
        }
        match parse_chunk(&event.data).and_then(|chunk| self.decoder.push(chunk)) {
            Ok(events) => events.into_iter().map(Ok).collect(),
            Err(error) => vec![Err(error)],
        }
    }

    fn end(&mut self) -> Vec<Result<StreamEvent>> {
        match self.decoder.finish(End::Stream) {
            Ok(events) => events.into_iter().map(Ok).collect(),
            Err(error) => vec![Err(error)],
        }
    }

    fn fail(&mut self, error: Error) -> Vec<Result<StreamEvent>> {
        let request_id = self.decoder.request_id().map(str::to_owned);
        vec![Err(error
            .with_provider(PROVIDER_NAME)
            .with_request_id_opt(request_id.as_deref()))]
    }
}

fn parse_chunk(data: &str) -> Result<GenerateContentResponse> {
    serde_json::from_str(data).map_err(|err| {
        Error::stream_decode(format!("invalid Gemini stream chunk: {err}"))
            .with_provider(PROVIDER_NAME)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use allama::{ChatStream, ChatStreamExt, StreamErrorKind};
    use allama_http::TransportErrorKind;
    use futures_util::StreamExt;

    fn stream(text: &str) -> ChatStream {
        Box::pin(sse_to_stream(
            SseStream::new(futures_util::stream::iter([Ok::<_, TransportError>(
                text.as_bytes().to_vec(),
            )])),
            Vec::new(),
            None,
        ))
    }

    async fn items(text: &str) -> Vec<Result<StreamEvent>> {
        stream(text).collect().await
    }

    #[tokio::test]
    async fn malformed_json_is_a_decode_error() {
        let items = items("data: {not json}\n\n").await;
        assert_eq!(items.len(), 1);
        assert!(matches!(
            items[0],
            Err(Error::Stream {
                kind: StreamErrorKind::Decode,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn truncated_stream_ends_with_incomplete_error() {
        let items = items(
            "data: {\"responseId\":\"r\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hel\"}]}}]}\n\n",
        )
        .await;
        let last = items.last().unwrap();
        assert!(matches!(
            last,
            Err(Error::Stream {
                kind: StreamErrorKind::Incomplete,
                ..
            })
        ));
        let err = stream(
            "data: {\"responseId\":\"r\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hel\"}]}}]}\n\n",
        )
        .collect_response()
        .await
        .unwrap_err();
        assert!(err.is_retryable());
    }

    #[tokio::test]
    async fn mid_stream_error_event_is_typed() {
        let items = items(concat!(
            "data: {\"responseId\":\"r\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}]}}]}\n\n",
            "data: {\"error\":{\"code\":503,\"message\":\"The model is overloaded.\",\"status\":\"UNAVAILABLE\"}}\n\n",
        ))
        .await;
        assert!(matches!(
            items.last().unwrap(),
            Err(Error::Overloaded { .. })
        ));
    }

    #[tokio::test]
    async fn empty_data_lines_are_skipped() {
        let response = stream(concat!(
            "data: \n\n",
            "data: {\"responseId\":\"r\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}]},\"finishReason\":\"STOP\"}]}\n\n",
        ))
        .collect_response()
        .await
        .unwrap();
        assert_eq!(response.text().as_deref(), Some("Hi"));
    }

    #[tokio::test]
    async fn transport_error_is_retryable_stream_error() {
        let chunks = |error| -> Vec<std::result::Result<Vec<u8>, TransportError>> {
            vec![
                Ok(
                    b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}]}}]}\n\n"
                        .to_vec(),
                ),
                Err(error),
            ]
        };
        let items: Vec<_> = sse_to_stream(
            SseStream::new(futures_util::stream::iter(chunks(TransportError::new(
                TransportErrorKind::Body,
                "connection reset",
            )))),
            Vec::new(),
            Some("req_1".into()),
        )
        .collect()
        .await;
        let err = items.last().unwrap().as_ref().unwrap_err();
        assert!(matches!(
            err,
            Error::Stream {
                kind: StreamErrorKind::Transport,
                ..
            }
        ));
        assert!(err.is_retryable());
        assert_eq!(err.request_id(), Some("req_1"));
        assert_eq!(
            err.context().unwrap().provider.as_deref(),
            Some(PROVIDER_NAME)
        );

        // A stream timeout stays a timeout.
        let items: Vec<_> = sse_to_stream(
            SseStream::new(futures_util::stream::iter(chunks(TransportError::new(
                TransportErrorKind::Timeout,
                "idle",
            )))),
            Vec::new(),
            None,
        )
        .collect()
        .await;
        assert!(matches!(items.last(), Some(Err(Error::Timeout { .. }))));
    }
}
