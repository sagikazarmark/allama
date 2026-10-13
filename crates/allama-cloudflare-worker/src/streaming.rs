//! SSE parsing for Cloudflare Workers AI streaming responses.
//!
//! With `stream: true`, the binding returns a byte stream of SSE. Native
//! models stream their own chunks:
//!
//! ```text
//! data: {"response":"Hello","p":"abcd"}
//!
//! data: {"response":"","usage":{"prompt_tokens":12,"completion_tokens":2,"total_tokens":14},"finish_reason":"stop"}
//!
//! data: [DONE]
//! ```
//!
//! Newer models stream chat-completion chunks (`choices[0].delta`, tool
//! calls included), repeat their text and tool calls in the native fields,
//! carry per-chunk usage, and end with a native chunk holding the
//! cumulative usage (recorded in `fixtures/stream_chat_tools.sse`).
//!
//! Each chunk is normalized into a chat-completion chunk and decoded by
//! `allama-openai-compat`'s Chat Completions stream decoder, the decoder
//! of its `Cloudflare` preset, so both transports decode alike. This crate
//! keeps only what the binding does differently:
//!
//! - the native fields become a choice delta, and the ones that repeat a
//!   chat-completion delta are removed, as is the per-chunk usage of
//!   chat-completion chunks (the final native chunk carries the total);
//! - an in-band `error` is classified with the Workers AI codes;
//! - a stream without a finish reason (older runtimes) ends `Stop`, or
//!   `ToolCalls` after tool calls, with no raw reason;
//! - a stream whose body ends after its finish reason but before `[DONE]`
//!   is complete.
//!
//! The bytes are framed by `allama_http::sse::SseStream`, the framing every
//! HTTP adapter uses (so an event past its size cap fails the stream, and
//! an event without its blank line at the end of the body is discarded),
//! and the decoder runs under the shared `SseChatStream` driver.
//!
//! # Send safety
//!
//! `worker::ByteStream` is `!Send` because it wraps JS objects via
//! wasm-bindgen. Cloudflare Workers run on a single thread, so `Send` is
//! irrelevant at runtime; `send_wrapper::SendWrapper` satisfies the `Send`
//! bound of `ChatStream` and panics if the value is ever touched from another
//! thread.

use std::pin::Pin;
use std::task::{Context, Poll};

use allama::chat::stream::StreamBlockType;
use allama::{Error, FinishReason, Result, StreamEvent, Warning};
use allama_http::framing::ChatFrameDecoder;
use allama_http::sse::{SseChatStream, SseEvent, SseStream};
use allama_http::{TransportError, TransportErrorKind};
use allama_openai_compat::binding::ChunkDecoder;
use futures_util::{Stream, TryStreamExt};
use send_wrapper::SendWrapper;
use serde_json::{Map, Value};

use crate::error::classify;
use crate::{NAMESPACE, PROVIDER_NAME, wire};

/// Decode a `worker::ByteStream` of SSE into chat stream events.
///
/// `warnings` are the request-mapping warnings; they are emitted right after
/// `ResponseStart`. The runtime's read errors become [`TransportError`]s of
/// kind `Body`, so the shared driver classifies them as any adapter's body
/// failure.
pub(crate) fn byte_stream_to_chat_stream(
    byte_stream: worker::ByteStream,
    warnings: Vec<Warning>,
    max_event_size: usize,
) -> impl Stream<Item = Result<StreamEvent>> + Send + 'static {
    let body = SendByteStream(SendWrapper::new(byte_stream))
        .map_err(|error| TransportError::new(TransportErrorKind::Body, error.to_string()));
    SseChatStream::new(
        SseStream::new(body).max_event_size(max_event_size),
        SseState::new(warnings),
    )
}

/// A `worker::ByteStream` made `Send` for the single-threaded Workers runtime.
struct SendByteStream(SendWrapper<worker::ByteStream>);

impl Stream for SendByteStream {
    type Item = worker::Result<Vec<u8>>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut *self.get_mut().0).poll_next(cx)
    }
}

/// Normalizes Workers AI chunks into chat-completion chunks for the shared
/// decoder, and handles what the binding does differently.
struct SseState {
    decoder: ChunkDecoder,
    /// Request-mapping warnings, emitted after `ResponseStart`.
    warnings: Vec<Warning>,
    /// Whether the decoder reported a finish reason.
    finished: bool,
    /// Whether the decoder started a tool call.
    tool_calls: bool,
    /// The position of the next whole native tool call: past every
    /// position seen.
    native_calls: usize,
}

impl SseState {
    fn new(warnings: Vec<Warning>) -> Self {
        Self {
            decoder: ChunkDecoder::new(PROVIDER_NAME, NAMESPACE, true),
            warnings,
            finished: false,
            tool_calls: false,
            native_calls: 0,
        }
    }

    /// Decode one unnamed event's data: a chunk, or `[DONE]`.
    fn process_data(&mut self, data: &str) -> Vec<Result<StreamEvent>> {
        let data = data.trim();
        if self.decoder.is_done() || data.is_empty() {
            return Vec::new();
        }
        if data == "[DONE]" {
            return self.done();
        }
        let chunk = match serde_json::from_str::<Value>(data) {
            Ok(Value::Object(chunk)) => chunk,
            Ok(_) => {
                let events = self.decoder.chunk(data);
                return self.decoded(events);
            }
            Err(err) => {
                return self.decoder.fail(Error::stream_decode(format!(
                    "invalid Workers AI stream chunk: {err}"
                )));
            }
        };
        if let Some(error) = chunk.get("error").filter(|error| !error.is_null()) {
            return self.decoder.fail(in_band_error(error));
        }
        let chunk = self.normalize(chunk);
        let events = self.decoder.chunk(&Value::Object(chunk).to_string());
        self.decoded(events)
    }

    /// A Workers AI chunk as a chat-completion chunk (see the module docs).
    fn normalize(&mut self, mut chunk: Map<String, Value>) -> Map<String, Value> {
        let padded = chunk.remove("p").is_some();
        if let Some(choices) = chunk.get("choices").and_then(Value::as_array) {
            // The binding's own chat-completion chunks (padded, with the
            // native fields) repeat the delta in the native fields, and
            // their usage is the chunk's own share; the final native chunk
            // carries the total.
            let mixed = padded || chunk.contains_key("response");
            if mixed && !choices.is_empty() {
                chunk.remove("usage");
            }
            chunk.remove("response");
            chunk.remove("tool_calls");
        } else {
            let mut delta = Map::new();
            match chunk.remove("response") {
                Some(text @ Value::String(_)) => {
                    delta.insert("content".into(), text);
                }
                Some(Value::Null) | None => {}
                Some(json) => {
                    delta.insert("content".into(), Value::String(json.to_string()));
                }
            }
            if let Some(Value::Array(calls)) = chunk.remove("tool_calls")
                && !calls.is_empty()
            {
                let calls = calls
                    .into_iter()
                    .map(|call| self.native_fragment(call))
                    .collect();
                delta.insert("tool_calls".into(), Value::Array(calls));
            }
            let mut choice = Map::new();
            choice.insert("index".into(), 0.into());
            choice.insert("delta".into(), Value::Object(delta));
            if let Some(reason) = chunk.remove("finish_reason") {
                choice.insert("finish_reason".into(), reason);
            }
            chunk.insert("choices".into(), Value::Array(vec![Value::Object(choice)]));
        }
        wire::normalize_usage(&mut chunk);
        chunk
    }

    /// A native tool call (or fragment) as a chat-completion tool call
    /// fragment: one with an `index` continues that call, and one without
    /// is a whole call of its own, at the next position. Arguments become
    /// JSON text, and ids are left to the decoder, which scopes the ones it
    /// makes up by the response.
    fn native_fragment(&mut self, call: Value) -> Value {
        let Value::Object(mut call) = call else {
            return call;
        };
        let mut function = match call.remove("function") {
            Some(Value::Object(function)) => function,
            _ => Map::new(),
        };
        for key in ["name", "arguments"] {
            if let Some(value) = call.remove(key) {
                function.entry(key).or_insert(value);
            }
        }
        if let Some(arguments) = function.get_mut("arguments")
            && !arguments.is_string()
        {
            let text = match &*arguments {
                Value::Null => String::new(),
                other => other.to_string(),
            };
            *arguments = Value::String(text);
        }
        match call.get("index").and_then(Value::as_u64) {
            Some(index) => {
                let next = usize::try_from(index)
                    .unwrap_or(usize::MAX)
                    .saturating_add(1);
                self.native_calls = self.native_calls.max(next);
            }
            None => {
                call.insert("index".into(), self.native_calls.into());
                self.native_calls += 1;
            }
        }
        call.insert("function".into(), Value::Object(function));
        Value::Object(call)
    }

    /// Track what the decoder emitted, and put the request warnings right
    /// after `ResponseStart`.
    fn decoded(&mut self, events: Vec<Result<StreamEvent>>) -> Vec<Result<StreamEvent>> {
        let mut out = Vec::with_capacity(events.len() + self.warnings.len());
        for event in events {
            match &event {
                Ok(StreamEvent::ResponseStart { .. }) => {
                    out.push(event);
                    out.extend(
                        self.warnings
                            .drain(..)
                            .map(|warning| Ok(StreamEvent::warning(warning))),
                    );
                    continue;
                }
                Ok(StreamEvent::ResponseMetadata {
                    finish_reason: Some(_),
                    ..
                }) => self.finished = true,
                Ok(StreamEvent::BlockStart {
                    block_type: StreamBlockType::ToolCall,
                    ..
                }) => self.tool_calls = true,
                _ => {}
            }
            out.push(event);
        }
        out
    }

    /// `[DONE]`: the decoder finishes the response. A stream that reported
    /// no finish reason (older runtimes) ended its turn, so the reason is
    /// inferred, with no raw reason, before `ResponseStop`.
    fn done(&mut self) -> Vec<Result<StreamEvent>> {
        let events = self.decoder.chunk("[DONE]");
        let mut events = self.decoded(events);
        if !self.finished
            && let Some(stop) = events
                .iter()
                .position(|event| matches!(event, Ok(StreamEvent::ResponseStop)))
        {
            let mut metadata = StreamEvent::metadata();
            if let StreamEvent::ResponseMetadata { finish_reason, .. } = &mut metadata {
                *finish_reason = Some(if self.tool_calls {
                    FinishReason::ToolCalls
                } else {
                    FinishReason::Stop
                });
            }
            events.insert(stop, Ok(metadata));
        }
        events
    }
}

impl ChatFrameDecoder<SseEvent> for SseState {
    /// Workers AI sends unnamed events; a named one is the shared
    /// decoder's (`error`, or a vendor event surfaced as provider data).
    fn frame(&mut self, event: SseEvent) -> Vec<Result<StreamEvent>> {
        match event.event.as_str() {
            "" | "message" => self.process_data(&event.data),
            name => {
                let events = self.decoder.event(name, &event.data);
                self.decoded(events)
            }
        }
    }

    /// The body ended: complete when the final chunk (with the finish
    /// reason) arrived and only `[DONE]` is missing, `Incomplete`
    /// otherwise.
    fn end(&mut self) -> Vec<Result<StreamEvent>> {
        if self.decoder.is_done() {
            return Vec::new();
        }
        if self.finished {
            self.done()
        } else {
            self.decoder.end()
        }
    }

    fn fail(&mut self, error: Error) -> Vec<Result<StreamEvent>> {
        self.decoder.fail(error)
    }

    fn is_done(&self) -> bool {
        self.decoder.is_done()
    }
}

/// Map an in-band `error` payload with the Workers AI classifier. Its
/// message is upstream text; a payload without one is kept only as the
/// error's body.
fn in_band_error(error: &Value) -> Error {
    let (code, message) = match error {
        Value::String(message) => (None, Some(message.as_str())),
        Value::Object(object) => (
            object.get("code").map(|code| match code {
                Value::String(code) => code.clone(),
                other => other.to_string(),
            }),
            object.get("message").and_then(Value::as_str),
        ),
        _ => (None, None),
    };
    let mapped = classify(code.as_deref(), message.unwrap_or_default());
    match message {
        Some(_) => mapped,
        None => mapped.with_body(error.to_string()),
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn conformance_stream_from_sse_text(text: &str) -> allama::ChatStream {
    // With the hold `finish_stream` applies.
    Box::pin(allama::adapter::chat::start_first(stream_from_sse_text(
        text,
        Vec::new(),
    )))
}

/// `text` decoded as a Workers AI byte stream is.
#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn stream_from_sse_text(text: &str, warnings: Vec<Warning>) -> allama::ChatStream {
    let body = allama_conformance::byte_stream_from_sse_text(text)
        .map_err(|error| TransportError::new(TransportErrorKind::Body, error.to_string()));
    Box::pin(SseChatStream::new(
        SseStream::new(body),
        SseState::new(warnings),
    ))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use allama::{ChatStream, ChatStreamExt, RequestField, StreamErrorKind, StreamEvent};
    use futures_util::StreamExt;
    use serde_json::json;

    use super::*;

    /// Feed `text`'s SSE events to `state`.
    fn events(state: &mut SseState, text: &str) -> Vec<Result<StreamEvent>> {
        let mut decoder = allama_http::sse::SseDecoder::new();
        decoder.push(text.as_bytes());
        let mut out = Vec::new();
        while let Some(event) = decoder.next_event().unwrap() {
            out.extend(state.frame(event));
        }
        out
    }

    /// The stream of a body of `chunks`.
    fn stream_from_chunks(chunks: Vec<std::result::Result<Vec<u8>, TransportError>>) -> ChatStream {
        Box::pin(SseChatStream::new(
            SseStream::new(futures_util::stream::iter(chunks)),
            SseState::new(Vec::new()),
        ))
    }

    fn ok(events: &[Result<StreamEvent>]) -> Vec<&StreamEvent> {
        events.iter().filter_map(|e| e.as_ref().ok()).collect()
    }

    async fn collect_all(stream: ChatStream) -> Vec<Result<StreamEvent>> {
        stream.collect().await
    }

    #[test]
    fn done_sentinel_emits_terminal_events_once() {
        let mut state = SseState::new(Vec::new());
        let first = events(&mut state, "data: {\"response\":\"Hello\"}\n\n");
        assert!(matches!(ok(&first)[0], StreamEvent::ResponseStart { .. }));

        let done = events(&mut state, "data: [DONE]\n\n");
        let done = ok(&done);
        assert!(matches!(done[0], StreamEvent::BlockStop { index: 0, .. }));
        assert!(matches!(done[1], StreamEvent::ResponseMetadata { .. }));
        assert!(matches!(done[2], StreamEvent::ResponseStop));
        assert_eq!(done.len(), 3);

        assert!(events(&mut state, "data: [DONE]\n\n").is_empty());
        assert!(state.is_done());
        assert!(state.end().is_empty());
    }

    #[tokio::test]
    async fn final_usage_chunk_supplies_finish_reason_and_usage() {
        let response = stream_from_sse_text(
            "data: {\"response\":\"Hi\",\"p\":\"abc\"}\n\n\
             data: {\"response\":\"\",\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":2,\"total_tokens\":14},\"finish_reason\":\"length\"}\n\n\
             data: [DONE]\n\n",
            Vec::new(),
        )
        .collect_response()
        .await
        .unwrap();
        assert_eq!(response.finish_reason, Some(FinishReason::Length));
        assert_eq!(response.finish_reason_raw.as_deref(), Some("length"));
        let usage = response.usage.unwrap();
        assert_eq!(usage.input_tokens, Some(12));
        assert_eq!(usage.output_tokens, Some(2));
        assert_eq!(usage.total_tokens, Some(14));
        assert_eq!(response.provider.as_deref(), Some("cloudflare"));
        // The stream padding is not provider data.
        assert!(
            response.provider_data.is_empty(),
            "{:?}",
            response.provider_data
        );
    }

    #[tokio::test]
    async fn missing_finish_reason_and_usage_are_tolerated() {
        let response = stream_from_sse_text(
            "data: {\"response\":\"Hi\"}\n\ndata: [DONE]\n\n",
            Vec::new(),
        )
        .collect_response()
        .await
        .unwrap();
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(response.finish_reason_raw, None);
        assert_eq!(response.usage, None);
    }

    #[tokio::test]
    async fn eof_after_final_usage_chunk_without_done_is_complete() {
        let stream = stream_from_sse_text(
            "data: {\"response\":\"Hi\"}\n\n\
             data: {\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1},\"finish_reason\":\"stop\"}\n\n",
            Vec::new(),
        );
        let response = stream.collect_response().await.unwrap();
        assert_eq!(response.text().as_deref(), Some("Hi"));
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(response.usage.unwrap().total_tokens, Some(2));
    }

    #[tokio::test]
    async fn eof_without_done_is_incomplete() {
        let all = collect_all(stream_from_sse_text(
            "data: {\"response\":\"Hello\"}\n\n",
            Vec::new(),
        ))
        .await;
        let last = all.last().unwrap().as_ref().unwrap_err();
        assert!(matches!(
            last,
            Error::Stream {
                kind: StreamErrorKind::Incomplete,
                ..
            }
        ));
        assert!(last.is_retryable());
        // The open block is closed before the error.
        assert!(matches!(
            all[all.len() - 2].as_ref().unwrap(),
            StreamEvent::BlockStop { index: 0, .. }
        ));
    }

    #[tokio::test]
    async fn empty_stream_is_incomplete() {
        let all = collect_all(stream_from_chunks(Vec::new())).await;
        assert_eq!(all.len(), 1);
        assert!(matches!(
            all[0],
            Err(Error::Stream {
                kind: StreamErrorKind::Incomplete,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn transport_errors_are_retryable_stream_errors() {
        let all = collect_all(stream_from_chunks(vec![
            Ok(b"data: {\"response\":\"Hi\"}\n\n".to_vec()),
            Err(TransportError::new(
                TransportErrorKind::Body,
                "connection reset",
            )),
        ]))
        .await;
        let err = all.last().unwrap().as_ref().unwrap_err();
        assert!(matches!(
            err,
            Error::Stream {
                kind: StreamErrorKind::Transport,
                ..
            }
        ));
        assert!(err.is_retryable());
        assert_eq!(err.upstream_message(), Some("connection reset"));
        assert_eq!(
            err.context().unwrap().provider.as_deref(),
            Some(PROVIDER_NAME)
        );
        // The open block is closed before the error.
        assert!(matches!(
            all[all.len() - 2].as_ref().unwrap(),
            StreamEvent::BlockStop { index: 0, .. }
        ));
    }

    #[tokio::test]
    async fn utf8_split_across_chunks_is_buffered_until_complete() {
        let expected = String::from_utf8(vec![b'c', b'a', b'f', 0xC3, 0xA9]).unwrap();
        let first = [b"data: {\"response\":\"caf".as_slice(), &[0xC3]].concat();
        let all = collect_all(stream_from_chunks(vec![
            Ok(first),
            Ok(b"\xA9\"}\n\ndata: [DONE]\n\n".to_vec()),
        ]))
        .await;
        assert!(ok(&all).iter().any(
            |event| matches!(event, StreamEvent::TextDelta { text, .. } if text == &expected)
        ));

        // Invalid UTF-8 is a decode error, not a retryable transport one.
        let all = collect_all(stream_from_chunks(vec![Ok(b"data: \xFF\n\n".to_vec())])).await;
        let err = all.last().unwrap().as_ref().unwrap_err();
        assert!(matches!(
            err,
            Error::Stream {
                kind: StreamErrorKind::Decode,
                ..
            }
        ));
        assert!(!err.is_retryable());
    }

    #[test]
    fn invalid_json_is_a_decode_error() {
        let mut state = SseState::new(Vec::new());
        let all = events(&mut state, "data: {not json}\n\n");
        assert!(matches!(
            all.last().unwrap(),
            Err(Error::Stream {
                kind: StreamErrorKind::Decode,
                ..
            })
        ));
        assert!(state.is_done());
    }

    #[test]
    fn in_band_errors_use_the_shared_classifier() {
        let mut state = SseState::new(Vec::new());
        let all = events(
            &mut state,
            "data: {\"response\":\"Hi\"}\n\n\
             data: {\"error\":{\"code\":3040,\"message\":\"Capacity temporarily exceeded\"}}\n\n",
        );
        let err = all.last().unwrap().as_ref().unwrap_err();
        assert!(matches!(err, Error::Overloaded { .. }), "{err:?}");
        assert_eq!(err.status(), Some(429));
    }

    #[test]
    fn in_band_error_text_stays_out_of_the_log() {
        const SENTINEL: &str = "my SSN is 123-45-6789";
        for payload in [
            json!({"error": {"code": 5004, "message": format!("Invalid data: {SENTINEL}")}}),
            json!({"error": format!("Rate limit exceeded: {SENTINEL}")}),
            json!({"error": {"code": 3040, "detail": SENTINEL}}),
            json!({"error": [SENTINEL]}),
        ] {
            let mut state = SseState::new(Vec::new());
            let all = events(&mut state, &format!("data: {payload}\n\n"));
            let err = all.last().unwrap().as_ref().unwrap_err();
            let line = err.as_log().with_message(1000).to_string();
            assert!(
                !err.as_log().to_string().contains(SENTINEL),
                "{payload}: {line}"
            );
            // Only a message is upstream text; other payloads stay in the body.
            if line.contains(SENTINEL) {
                assert!(err.upstream_message().is_some_and(|m| m.contains(SENTINEL)));
            }
        }
    }

    #[test]
    fn request_warnings_follow_response_start() {
        let warning =
            Warning::unsupported(RequestField::Stop, "no stop").with_provider("cloudflare");
        let mut state = SseState::new(vec![warning.clone()]);
        let all = events(&mut state, "data: {\"response\":\"Hi\"}\n\n");
        let all = ok(&all);
        assert!(matches!(all[0], StreamEvent::ResponseStart { .. }));
        assert_eq!(all[1], &StreamEvent::warning(warning));
    }

    #[tokio::test]
    async fn chat_completion_shaped_stream_with_reasoning() {
        let stream = stream_from_sse_text(
            "data: {\"id\":\"c1\",\"model\":\"@cf/qwen/qwen3-30b-a3b-fp8\",\"choices\":[{\"delta\":{\"reasoning_content\":\"Think\"}}]}\n\n\
             data: {\"choices\":[{\"delta\":{\"content\":\"Answer\"}}]}\n\n\
             data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3,\"total_tokens\":8}}\n\n\
             data: [DONE]\n\n",
            Vec::new(),
        );
        let response = stream.collect_response().await.unwrap();
        assert_eq!(response.reasoning_text().as_deref(), Some("Think"));
        assert_eq!(response.text().as_deref(), Some("Answer"));
        assert_eq!(response.id.as_deref(), Some("c1"));
        assert_eq!(
            response.model.as_deref(),
            Some("@cf/qwen/qwen3-30b-a3b-fp8")
        );
        assert_eq!(response.finish_reason_raw.as_deref(), Some("stop"));
        assert_eq!(response.usage.unwrap().total_tokens, Some(8));
    }

    /// Streamed tool calls are decoded, in the native fields (as
    /// fragments with an `index`, or whole calls without) as in the
    /// chat-completion shape; a stream with calls and no finish reason
    /// ends `ToolCalls`.
    #[tokio::test]
    async fn streamed_tool_calls_are_decoded() {
        let response = stream_from_sse_text(
            "data: {\"tool_calls\":[{\"index\":0,\"id\":\"x\",\"function\":{\"name\":\"f\"}}]}\n\n\
             data: {\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{}\"}}]}\n\n\
             data: {\"tool_calls\":[{\"name\":\"g\",\"arguments\":{\"a\":1}}]}\n\n\
             data: [DONE]\n\n",
            Vec::new(),
        )
        .collect_response()
        .await
        .unwrap();
        let calls: Vec<_> = response
            .tool_calls()
            .map(|call| (call.name.as_str(), call.arguments.as_str()))
            .collect();
        assert_eq!(calls, [("f", "{}"), ("g", r#"{"a":1}"#)]);
        assert_eq!(response.first_tool_call().unwrap().id, "x");
        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
        assert!(response.warnings.is_empty(), "{:?}", response.warnings);
    }

    /// A chat-completion stream recorded from Workers AI
    /// (`@cf/qwen/qwen3-30b-a3b-fp8`, 2026-10-07) repeats its text and tool
    /// calls in the native fields and carries per-chunk usage, then the
    /// total in a final native chunk: the text is not doubled, the tool
    /// call is decoded, and the usage is the total.
    #[tokio::test]
    async fn recorded_chat_completion_streams_decode_once() {
        let response = stream_from_sse_text(
            include_str!("../fixtures/stream_chat_tools.sse"),
            Vec::new(),
        )
        .collect_response()
        .await
        .unwrap();
        assert!(response.reasoning_text().is_some());
        let call = response.first_tool_call().unwrap();
        assert_eq!(call.name, "get_weather");
        assert!(call.has_valid_arguments(), "{}", call.arguments);
        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
        let usage = response.usage.unwrap();
        assert_eq!(
            (usage.input_tokens, usage.output_tokens, usage.total_tokens),
            (Some(156), Some(188), Some(344))
        );

        // `@cf/meta/llama-3.3-70b-instruct-fp8-fast` repeats every delta in
        // `response`.
        let raw = include_str!("../fixtures/stream_chat_duplicated.sse");
        let response = stream_from_sse_text(raw, Vec::new())
            .collect_response()
            .await
            .unwrap();
        let expected: String = raw
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|data| serde_json::from_str::<serde_json::Value>(data).ok())
            .filter_map(|chunk| {
                chunk
                    .pointer("/choices/0/delta/content")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            })
            .collect();
        assert_eq!(response.text_or_empty(), expected);
        assert_eq!(response.usage.unwrap().total_tokens, Some(299));
    }

    /// Stream ends keep Workers AI's documented exceptions to the
    /// marker rule:
    ///
    /// - `[DONE]` without a finish reason is `Stop`, not a missing reason:
    ///   older runtimes report none (the crate README, "Finish reason and
    ///   usage"; captured in `fixtures/stream_legacy.sse`). The limit: on
    ///   such a runtime, a stream the runtime itself ended early (and closed
    ///   with `[DONE]`) reports `Stop` too, so `finished_normally()` is
    ///   `true` for it.
    /// - A body that ends without `[DONE]` after the finish-reason chunk is
    ///   complete: since the 2026-02-17 release the reason arrives with the
    ///   final usage chunk, which is all `[DONE]` would confirm.
    /// - Anything else that ends without `[DONE]` is `Incomplete`.
    #[tokio::test]
    async fn stream_ends_follow_the_documented_exceptions() {
        let legacy =
            stream_from_sse_text(include_str!("../fixtures/stream_legacy.sse"), Vec::new())
                .collect_response()
                .await
                .unwrap();
        assert_eq!(legacy.finish_reason, Some(FinishReason::Stop));
        assert_eq!(legacy.finish_reason_raw, None);
        assert!(legacy.finished_normally());
        assert!(legacy.warnings.is_empty());

        let without_done = stream_from_sse_text(
            "data: {\"response\":\"Hi\"}\n\n\
             data: {\"response\":\"\",\"finish_reason\":\"length\"}\n\n",
            Vec::new(),
        )
        .collect_response()
        .await
        .unwrap();
        assert_eq!(without_done.finish_reason, Some(FinishReason::Length));

        let cut_short = stream_from_sse_text("data: {\"response\":\"Hi\"}\n\n", Vec::new())
            .collect_response()
            .await
            .unwrap_err();
        assert!(matches!(
            cut_short,
            Error::Stream {
                kind: StreamErrorKind::Incomplete,
                ..
            }
        ));
    }

    /// `[DONE]` with no chunk before it is no response, on both Cloudflare
    /// transports.
    #[tokio::test]
    async fn done_without_content_is_incomplete() {
        let err = stream_from_sse_text("data: [DONE]\n\n", Vec::new())
            .collect_response()
            .await
            .unwrap_err();
        assert!(
            matches!(
                err,
                Error::Stream {
                    kind: StreamErrorKind::Incomplete,
                    ..
                }
            ),
            "{err:?}"
        );
    }
}
