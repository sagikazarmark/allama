//! Translates Anthropic SSE events into portable [`StreamEvent`]s.
//!
//! - Text, thinking, redacted thinking and tool use map to typed blocks;
//!   signatures and citations arrive as `BlockProviderData`.
//! - Server and MCP tool calls stream like tool calls (`ProviderToolCall`
//!   blocks with `ToolCallDelta` input); their results arrive whole as
//!   `ProviderToolResult` blocks.
//! - Other block types are buffered and emitted as complete `Other` blocks
//!   when they stop, so their streamed input is not lost.
//! - Usage is tracked cumulatively: `message_start` carries input and cache
//!   counts, `message_delta` cumulative output counts; every metadata event
//!   is a normalized snapshot.
//! - Delta types this adapter does not map surface verbatim as
//!   `ProviderDelta` events (origin `"anthropic"`).
//! - Blocks get dense local indices (0, 1, 2, … in start order): the wire
//!   index is only looked up, never trusted as a position, so a hostile
//!   index cannot make the collector allocate. A delta for a block that
//!   was never started is a protocol error.
//! - Event types this adapter does not know (Anthropic may add some, and
//!   proxies add their own) surface verbatim as `ResponseProviderData`
//!   under the `"anthropic"` namespace and the event type (an array of
//!   payloads, in arrival order), with one `Warning` per event type.

use std::collections::HashMap;

use allama::chat::stream::{MetadataBuilder, StreamBlockType, UsageMetadataMode};
use allama::{Error, ExtraMap, ProviderData, Result, StreamErrorKind, StreamEvent, Warning};
use allama_http::framing::ChatFrameDecoder;
use allama_http::sse::{SseEvent, VendorFrames};
use serde::Deserialize;
use serde_json::Value;

use crate::decode;
use crate::error::map_stream_api_error;
use crate::wire;
use crate::{NAMESPACE, PROVIDER_NAME};

/// Per-stream decoding state.
pub(crate) struct SseState {
    request_id: Option<String>,
    pending_warnings: Vec<Warning>,
    started: bool,
    finished: bool,
    usage: wire::Usage,
    /// Open blocks, by wire index.
    blocks: HashMap<usize, OpenBlock>,
    /// The local index of each open block, by wire index.
    local: HashMap<usize, usize>,
    /// The local index the next block gets.
    next_index: usize,
    /// Event types this adapter does not know.
    vendor_frames: VendorFrames,
}

/// The event types of the Messages streaming API.
const KNOWN_EVENTS: &[&str] = &[
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
    "ping",
    "error",
];

impl SseState {
    pub(crate) fn new(request_id: Option<String>, warnings: Vec<Warning>) -> Self {
        Self {
            request_id,
            pending_warnings: warnings,
            started: false,
            finished: false,
            usage: wire::Usage::default(),
            blocks: HashMap::new(),
            local: HashMap::new(),
            next_index: 0,
            vendor_frames: VendorFrames::new(PROVIDER_NAME, NAMESPACE),
        }
    }

    fn start(
        &mut self,
        id: Option<String>,
        model: Option<String>,
        out: &mut Vec<Result<StreamEvent>>,
    ) {
        if self.started {
            return;
        }
        self.started = true;
        out.push(Ok(StreamEvent::response_start(id, model)));
        for warning in self.pending_warnings.drain(..) {
            out.push(Ok(StreamEvent::warning(warning)));
        }
    }

    /// An event type this adapter does not know: surfaced verbatim, never
    /// the start of the response (`finish_stream` holds what arrives before
    /// it).
    fn unknown_event(&mut self, event_type: &str, data: &str, out: &mut Vec<Result<StreamEvent>>) {
        let events = self.vendor_frames.surface(event_type, data);
        out.extend(events.into_iter().map(Ok));
    }

    /// A metadata event with the usage so far, for the caller to complete.
    fn metadata(&self) -> MetadataBuilder {
        MetadataBuilder::new()
            .with_usage(decode::normalize_usage(&self.usage))
            .with_usage_mode(UsageMetadataMode::Snapshot)
            .with_provider(PROVIDER_NAME)
            .with_request_id_opt(self.request_id.as_deref())
    }

    /// Process one SSE event.
    pub(crate) fn process(&mut self, event_type: &str, data: &str) -> Vec<Result<StreamEvent>> {
        let mut out = Vec::new();
        if data.is_empty() || self.finished {
            return out;
        }
        if !KNOWN_EVENTS.contains(&event_type) {
            self.unknown_event(event_type, data, &mut out);
            return out;
        }
        let json: Value = match serde_json::from_str(data) {
            Ok(json) => json,
            Err(e) => {
                self.finished = true;
                out.push(Err(Error::stream_decode(format!(
                    "invalid JSON in `{event_type}` event: {e}"
                ))
                .with_provider(PROVIDER_NAME)));
                return out;
            }
        };

        let result = match event_type {
            "message_start" => self.message_start(json, &mut out),
            "content_block_start" => self.block_start(json, &mut out),
            "content_block_delta" => self.block_delta(json, &mut out),
            "content_block_stop" => self.block_stop(json, &mut out),
            "message_delta" => self.message_delta(json, &mut out),
            "message_stop" => {
                self.start(None, None, &mut out);
                self.finished = true;
                out.extend(self.vendor_frames.finish().map(Ok));
                out.push(Ok(StreamEvent::ResponseStop));
                Ok(())
            }
            "error" => {
                self.finished = true;
                let error = match serde_json::from_value::<wire::ErrorResponse>(json) {
                    Ok(error) => map_stream_api_error(&error, self.request_id.clone()),
                    Err(_) => Error::stream_protocol("malformed `error` event")
                        .authored()
                        .with_provider(PROVIDER_NAME)
                        .with_request_id_opt(self.request_id.as_deref()),
                };
                out.push(Err(error));
                Ok(())
            }
            // `ping`.
            _ => Ok(()),
        };
        if let Err(error) = result {
            self.finished = true;
            out.push(Err(error));
        }
        out
    }

    fn message_start(&mut self, json: Value, out: &mut Vec<Result<StreamEvent>>) -> Result<()> {
        let envelope: MessageStartEnvelope = parse(json, "message_start")?;
        let message = envelope.message;
        if let Some(usage) = message.usage {
            self.usage.merge(usage);
        }
        self.start(message.id.clone(), message.model.clone(), out);
        let mut extra = message.extra;
        for reported_later in [
            "content",
            "stop_reason",
            "stop_sequence",
            "stop_details",
            "container",
        ] {
            extra.remove(reported_later);
        }
        let mut provider_data = ProviderData::new();
        decode::unmodeled_fields(&mut provider_data, extra);
        out.push(Ok(self
            .metadata()
            .with_id_opt(message.id.as_deref())
            .with_model_opt(message.model.as_deref())
            .with_provider_data(provider_data)
            .build()));
        Ok(())
    }

    fn block_start(&mut self, json: Value, out: &mut Vec<Result<StreamEvent>>) -> Result<()> {
        self.start(None, None, out);
        let envelope: ContentBlockStartEnvelope = parse(json, "content_block_start")?;
        let wire_index = envelope.index;
        if self.local.contains_key(&wire_index) {
            return Err(protocol(&format!(
                "content_block_start for block {wire_index}, which is already open"
            )));
        }
        let index = self.next_index;
        self.next_index += 1;
        self.local.insert(wire_index, index);
        let Value::Object(mut block) = envelope.content_block else {
            return Err(protocol("content_block_start without an object block"));
        };
        let type_name = match block.remove("type") {
            Some(Value::String(t)) => t,
            _ => return Err(protocol("content_block_start without a block type")),
        };
        match type_name.as_str() {
            "text" => {
                out.push(Ok(StreamEvent::block_start(index, StreamBlockType::Text)));
                if let Some(text) = string(&block, "text").filter(|t| !t.is_empty()) {
                    out.push(Ok(StreamEvent::text_delta(index, text)));
                }
                let citations = match block.remove("citations") {
                    Some(Value::Array(items)) => items,
                    _ => Vec::new(),
                };
                out.extend(
                    citations
                        .iter()
                        .filter_map(decode::citation)
                        .map(|citation| Ok(StreamEvent::citation(index, citation))),
                );
                self.blocks
                    .insert(wire_index, OpenBlock::Text { citations });
            }
            "thinking" => {
                out.push(Ok(StreamEvent::block_start(
                    index,
                    StreamBlockType::Reasoning,
                )));
                if let Some(text) = string(&block, "thinking").filter(|t| !t.is_empty()) {
                    out.push(Ok(StreamEvent::reasoning_delta(index, text)));
                }
                let signature = decode::signature_data(string(&block, "signature"));
                if !signature.is_empty() {
                    out.push(Ok(StreamEvent::block_provider_data(index, signature)));
                }
                self.blocks.insert(wire_index, OpenBlock::Reasoning);
            }
            "redacted_thinking" => {
                let data = string(&block, "data")
                    .ok_or_else(|| protocol("redacted_thinking block without `data`"))?;
                let mut start = StreamEvent::block_start(index, StreamBlockType::Reasoning);
                if let StreamEvent::BlockStart { provider_data, .. } = &mut start {
                    *provider_data = decode::redacted_data(data);
                }
                out.push(Ok(start));
                self.blocks.insert(wire_index, OpenBlock::Reasoning);
            }
            "tool_use" => {
                let id = string(&block, "id").ok_or_else(|| protocol("tool_use without `id`"))?;
                let name =
                    string(&block, "name").ok_or_else(|| protocol("tool_use without `name`"))?;
                block.remove("id");
                block.remove("name");
                let input = block
                    .remove("input")
                    .filter(|i| i.as_object().is_some_and(|o| !o.is_empty()));
                let mut start = StreamEvent::tool_call_start(index, id, name);
                if let StreamEvent::BlockStart { provider_data, .. } = &mut start {
                    *provider_data = decode::tool_use_data(block);
                }
                out.push(Ok(start));
                if let Some(input) = input {
                    out.push(Ok(StreamEvent::tool_call_delta(index, input.to_string())));
                }
                self.blocks.insert(wire_index, OpenBlock::ToolCall);
            }
            kind if decode::is_provider_tool_call(kind) => {
                let input = block
                    .remove("input")
                    .filter(|i| i.as_object().is_some_and(|o| !o.is_empty()));
                let call =
                    decode::provider_tool_call(type_name, block, &Value::Object(ExtraMap::new()))
                        .map_err(|e| protocol(&e.message()))?;
                let mut start =
                    StreamEvent::provider_tool_call_start(index, call.id, call.name, call.origin);
                if let StreamEvent::BlockStart { provider_data, .. } = &mut start {
                    *provider_data = call.provider_data;
                }
                out.push(Ok(start));
                if let Some(input) = input {
                    out.push(Ok(StreamEvent::tool_call_delta(index, input.to_string())));
                }
                self.blocks.insert(wire_index, OpenBlock::ToolCall);
            }
            kind if decode::is_provider_tool_result(kind) => {
                let result = decode::provider_tool_result(type_name, block)
                    .map_err(|e| protocol(&e.message()))?;
                out.push(Ok(StreamEvent::provider_tool_result_start(index, result)));
                self.blocks.insert(wire_index, OpenBlock::Complete);
            }
            _ => {
                self.blocks.insert(
                    wire_index,
                    OpenBlock::Other {
                        type_name,
                        data: block,
                        partial_json: String::new(),
                    },
                );
            }
        }
        Ok(())
    }

    fn block_delta(&mut self, json: Value, out: &mut Vec<Result<StreamEvent>>) -> Result<()> {
        self.start(None, None, out);
        let envelope: ContentBlockDeltaEnvelope = parse(json, "content_block_delta")?;
        let wire_index = envelope.index;
        let index = *self.local.get(&wire_index).ok_or_else(|| {
            protocol(&format!(
                "content_block_delta for block {wire_index}, which was not started"
            ))
        })?;
        let mut delta = envelope.delta;
        let delta_type = string(&delta, "type")
            .ok_or_else(|| protocol("content_block_delta without a delta `type`"))?;
        delta.remove("type");

        if let Some(OpenBlock::Other {
            data, partial_json, ..
        }) = self.blocks.get_mut(&wire_index)
        {
            match delta_type.as_str() {
                "input_json_delta" => {
                    partial_json.push_str(&take_string(&mut delta, "partial_json"));
                }
                "text_delta" => {
                    let text = take_string(&mut delta, "text");
                    match data.get_mut("text") {
                        Some(Value::String(existing)) => existing.push_str(&text),
                        _ => {
                            data.insert("text".into(), Value::String(text));
                        }
                    }
                }
                _ => out.push(Ok(provider_delta(index, delta_type, delta))),
            }
            return Ok(());
        }

        match delta_type.as_str() {
            "text_delta" => out.push(Ok(StreamEvent::text_delta(
                index,
                take_string(&mut delta, "text"),
            ))),
            "input_json_delta" => out.push(Ok(StreamEvent::tool_call_delta(
                index,
                take_string(&mut delta, "partial_json"),
            ))),
            "thinking_delta" => out.push(Ok(StreamEvent::reasoning_delta(
                index,
                take_string(&mut delta, "thinking"),
            ))),
            "signature_delta" => {
                let signature =
                    Some(take_string(&mut delta, "signature")).filter(|s| !s.is_empty());
                let provider_data = decode::signature_data(signature);
                if !provider_data.is_empty() {
                    out.push(Ok(StreamEvent::block_provider_data(index, provider_data)));
                }
            }
            "citations_delta" => match (self.blocks.get_mut(&wire_index), delta.remove("citation"))
            {
                (Some(OpenBlock::Text { citations }), Some(raw)) => {
                    if let Some(citation) = decode::citation(&raw) {
                        out.push(Ok(StreamEvent::citation(index, citation)));
                    }
                    citations.push(raw);
                }
                (_, citation) => {
                    if let Some(citation) = citation {
                        delta.insert("citation".into(), citation);
                    }
                    out.push(Ok(provider_delta(index, delta_type, delta)));
                }
            },
            _ => out.push(Ok(provider_delta(index, delta_type, delta))),
        }
        Ok(())
    }

    fn block_stop(&mut self, json: Value, out: &mut Vec<Result<StreamEvent>>) -> Result<()> {
        self.start(None, None, out);
        let envelope: IndexEnvelope = parse(json, "content_block_stop")?;
        let Some(index) = self.local.remove(&envelope.index) else {
            return Ok(());
        };
        let Some(block) = self.blocks.remove(&envelope.index) else {
            return Ok(());
        };
        match block {
            OpenBlock::Text { citations } if !citations.is_empty() => {
                out.push(Ok(StreamEvent::block_provider_data(
                    index,
                    ProviderData::new().with_raw(NAMESPACE, "citations", Value::Array(citations)),
                )));
            }
            OpenBlock::Other {
                type_name,
                mut data,
                partial_json,
            } => {
                if !partial_json.trim().is_empty() {
                    let input =
                        serde_json::from_str(&partial_json).unwrap_or(Value::String(partial_json));
                    data.insert("input".into(), input);
                }
                let mut start = StreamEvent::block_start(index, StreamBlockType::Other);
                if let StreamEvent::BlockStart {
                    type_name: slot_type,
                    data: slot_data,
                    origin,
                    ..
                } = &mut start
                {
                    *slot_type = Some(type_name);
                    *slot_data = Some(data);
                    *origin = Some(NAMESPACE.into());
                }
                out.push(Ok(start));
            }
            _ => {}
        }
        out.push(Ok(StreamEvent::block_stop(index)));
        Ok(())
    }

    fn message_delta(&mut self, json: Value, out: &mut Vec<Result<StreamEvent>>) -> Result<()> {
        self.start(None, None, out);
        let envelope: MessageDeltaEnvelope = parse(json, "message_delta")?;
        if let Some(usage) = envelope.usage {
            // Anthropic reports cumulative counts here.
            self.usage.merge(usage);
        }
        let delta = envelope.delta;
        let provider_data = decode::response_provider_data(
            delta.stop_sequence,
            delta.stop_details,
            delta.container,
            &self.usage,
        );
        out.push(Ok(self
            .metadata()
            .with_finish_reason_opt(delta.stop_reason.as_deref().map(decode::finish_reason))
            .with_finish_reason_raw_opt(delta.stop_reason.as_deref())
            .with_provider_data(provider_data)
            .build()));
        Ok(())
    }

    /// The error for a body that ended before `message_stop`.
    fn incomplete(&mut self) -> Option<Error> {
        if self.finished {
            return None;
        }
        self.finished = true;
        Some(
            Error::stream(
                StreamErrorKind::Incomplete,
                "SSE stream ended before message_stop",
            )
            .authored()
            .with_provider(PROVIDER_NAME)
            .with_request_id_opt(self.request_id.as_deref()),
        )
    }
}

enum OpenBlock {
    Text {
        citations: Vec<Value>,
    },
    Reasoning,
    ToolCall,
    /// A block delivered whole by `content_block_start` (provider tool
    /// results).
    Complete,
    Other {
        type_name: String,
        data: ExtraMap,
        partial_json: String,
    },
}

#[derive(Deserialize)]
struct MessageStartEnvelope {
    message: MessageStartPayload,
}

#[derive(Deserialize)]
struct MessageStartPayload {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<wire::Usage>,
    /// What `message_start` carries beyond these: `type` and `role`, the
    /// empty `content`, and the stop fields `message_delta` reports; the
    /// rest (`diagnostics`, …) is kept as response provider data.
    #[serde(flatten)]
    extra: ExtraMap,
}

#[derive(Deserialize)]
struct ContentBlockStartEnvelope {
    index: usize,
    content_block: Value,
}

#[derive(Deserialize)]
struct ContentBlockDeltaEnvelope {
    index: usize,
    delta: ExtraMap,
}

#[derive(Deserialize)]
struct IndexEnvelope {
    index: usize,
}

#[derive(Deserialize)]
struct MessageDeltaEnvelope {
    #[serde(default)]
    delta: MessageDeltaPayload,
    #[serde(default)]
    usage: Option<wire::Usage>,
}

#[derive(Default, Deserialize)]
struct MessageDeltaPayload {
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    stop_sequence: Option<String>,
    #[serde(default)]
    stop_details: Option<Value>,
    #[serde(default)]
    container: Option<Value>,
}

fn parse<T: serde::de::DeserializeOwned>(json: Value, event: &str) -> Result<T> {
    serde_json::from_value(json).map_err(|e| protocol(&format!("malformed `{event}` event: {e}")))
}

fn protocol(message: &str) -> Error {
    Error::stream_protocol(message.to_owned()).with_provider(PROVIDER_NAME)
}

fn string(block: &ExtraMap, key: &str) -> Option<String> {
    block.get(key).and_then(Value::as_str).map(str::to_owned)
}

/// Remove a string field, defaulting to empty.
fn take_string(map: &mut ExtraMap, key: &str) -> String {
    match map.remove(key) {
        Some(Value::String(value)) => value,
        _ => String::new(),
    }
}

/// A delta the portable events do not represent, surfaced verbatim.
fn provider_delta(index: usize, delta_type: String, data: ExtraMap) -> StreamEvent {
    StreamEvent::provider_delta(index, delta_type, data, Some(NAMESPACE.to_owned()))
}

impl ChatFrameDecoder<SseEvent> for SseState {
    fn frame(&mut self, event: SseEvent) -> Vec<Result<StreamEvent>> {
        self.process(&event.event, &event.data)
    }

    fn end(&mut self) -> Vec<Result<StreamEvent>> {
        self.incomplete().map(Err).into_iter().collect()
    }

    fn fail(&mut self, error: Error) -> Vec<Result<StreamEvent>> {
        self.finished = true;
        vec![Err(error
            .with_provider(PROVIDER_NAME)
            .with_request_id_opt(self.request_id.as_deref()))]
    }

    fn is_done(&self) -> bool {
        self.finished
    }
}

#[cfg(test)]
mod tests;
