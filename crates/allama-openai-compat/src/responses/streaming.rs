//! Translates Responses API SSE events into portable [`StreamEvent`]s.
//!
//! Block indices are allocated in arrival order: one block per message
//! content part, function call, reasoning item, or other output item.
//!
//! - Text parts stream as text deltas, and refusal parts as refusal deltas
//!   of a `Refusal` block, never as text; annotations arrive with
//!   `response.content_part.done`, as `Citation` events (the portable view)
//!   and `BlockProviderData` (the raw annotations, for replay).
//! - Function call arguments stream as tool-call deltas.
//! - Reasoning streams as reasoning deltas: the summary (parts separated by
//!   a blank line), or raw reasoning text (`reasoning_text` content parts,
//!   `response.reasoning_text.delta`) when the endpoint returns no summary.
//!   Whichever starts first becomes the block text; `encrypted_content` and
//!   raw `content` arrive with `response.output_item.done` as
//!   `BlockProviderData`.
//! - Items are correlated by `item_id`, falling back to `output_index`, so
//!   endpoints that omit either (llama.cpp omits `output_index`) still work.
//!   A message content event for an unknown item that has an `item_id` opens
//!   the message implicitly (LiteLLM never announces message items).
//! - Hosted tool items start a `ProviderToolCall` block when added; their
//!   progress events become `ProviderDelta`s, and the arguments and the
//!   `ProviderToolResult` block follow when the item is done.
//! - Other items are emitted as complete `Other` blocks when they finish
//!   (their input deltas, such as `response.custom_tool_call_input.delta`,
//!   are not surfaced). A client-executed call among them that has no
//!   output makes the finish reason `ToolCalls`, as in a unary response.
//! - Provider data is written under the dialect namespace, exactly as the
//!   non-streaming decoder writes it.
//! - `response.completed` / `response.incomplete` carry usage and the
//!   finish reason; `response.failed` and `error` events end the stream with
//!   a typed error. A stream without a terminal event is incomplete.
//! - Events outside the `response.` namespace (other than `error`) are vendor
//!   additions: their data is surfaced as `ResponseProviderData` under the
//!   dialect namespace and the event type (an array of payloads, in arrival
//!   order), with one `Warning` per event type. Unknown `response.*` events
//!   (OpenAI's own, e.g. `*.done` events whose content arrives elsewhere) are
//!   not surfaced.

use std::borrow::Cow;
use std::collections::HashMap;

use allama::chat::stream::{StreamBlockType, UsageMetadataMode};
use allama::{ContentBlock, Error, ExtraMap, Result, StreamEvent, Subject, Warning, WarningKind};
use allama_http::framing::ChatFrameDecoder;
use allama_http::sse::{SseEvent, VendorFrames};
use serde::Deserialize;
use serde_json::Value;

use super::decode::{self, Origin, OutputFlags, ReasoningSource, SUMMARY_SEPARATOR};
use super::error::{incomplete, map_response_error, protocol};
use super::wire::{ResponseError, ResponseObject};
use crate::ResponseContext;

/// Per-stream decoding state.
pub(crate) struct SseState {
    provider: Cow<'static, str>,
    namespace: Cow<'static, str>,
    request_id: Option<String>,
    pending_warnings: Vec<Warning>,
    started: bool,
    finished: bool,
    next_index: usize,
    items: Items,
    flags: OutputFlags,
    /// Finished `Other` items, to tell whether a client-executed call waits
    /// for its output when the response ends.
    others: Vec<ContentBlock>,
    /// Response id, once known (for the resume point).
    response_id: Option<String>,
    /// `sequence_number` of the last event processed.
    last_sequence: Option<u64>,
    /// When resuming: events up to this sequence number were already
    /// delivered; they rebuild state but are not emitted again.
    skip_through: Option<u64>,
    /// Events outside the Responses API.
    vendor_frames: VendorFrames,
}

impl SseState {
    pub(crate) fn new(context: ResponseContext) -> Self {
        let vendor_frames = VendorFrames::new(context.provider.clone(), context.namespace.clone());
        Self {
            provider: context.provider,
            namespace: context.namespace,
            request_id: context.request_id,
            pending_warnings: context.warnings,
            started: false,
            finished: false,
            next_index: 0,
            items: Items::default(),
            flags: OutputFlags::default(),
            others: Vec::new(),
            response_id: None,
            last_sequence: None,
            skip_through: None,
            vendor_frames,
        }
    }

    /// Replay a stream whose events up to `sequence_number` were already
    /// delivered (see [`ResponsesProvider::resume_stream`](crate::ResponsesProvider::resume_stream)).
    pub(crate) fn skip_through(mut self, sequence_number: Option<u64>) -> Self {
        self.skip_through = sequence_number;
        self
    }

    /// Record where the stream stopped on an error that interrupted it, so
    /// the caller can resume it.
    fn with_resume_point(&self, mut error: Error) -> Error {
        if let (Some(id), Some(sequence), Some(context)) =
            (&self.response_id, self.last_sequence, error.context_mut())
        {
            context.metadata.insert(
                super::StreamPosition::RESPONSE_ID_KEY.into(),
                Value::String(id.clone()),
            );
            context.metadata.insert(
                super::StreamPosition::SEQUENCE_NUMBER_KEY.into(),
                sequence.into(),
            );
        }
        error
    }

    /// Process one SSE event.
    pub(crate) fn process(&mut self, event_name: &str, data: &str) -> Vec<Result<StreamEvent>> {
        let mut out = Vec::new();
        if self.finished || data.trim().is_empty() {
            return out;
        }
        if is_vendor_event(event_name) {
            // Not necessarily JSON: kept verbatim either way.
            let events = self.vendor_frames.surface(event_name, data);
            self.vendor_event(events, &mut out);
            return out;
        }
        let json: Value = match serde_json::from_str(data) {
            Ok(json) => json,
            Err(e) => {
                self.finished = true;
                out.push(Err(Error::stream_decode(format!(
                    "invalid JSON in `{event_name}` event: {e}"
                ))
                .with_provider(self.provider.as_ref())
                .with_request_id_opt(self.request_id.as_deref())));
                return out;
            }
        };
        let kind = json
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or(event_name)
            .to_owned();
        let sequence = json.get("sequence_number").and_then(Value::as_u64);
        if let Err(error) = self.dispatch(&kind, json, &mut out) {
            self.finished = true;
            out.push(Err(error
                .with_provider(self.provider.as_ref())
                .with_request_id_opt(self.request_id.as_deref())));
        }
        if let Some(sequence) = sequence {
            self.last_sequence = Some(sequence);
            if self.skip_through.is_some_and(|through| sequence <= through) {
                out.retain(Result::is_err);
            }
        }
        out
    }

    fn dispatch(
        &mut self,
        kind: &str,
        json: Value,
        out: &mut Vec<Result<StreamEvent>>,
    ) -> Result<()> {
        match kind {
            "response.created" | "response.in_progress" | "response.queued" => {
                let event: ResponseEnvelope = parse(json, kind)?;
                self.start(event.response.id, event.response.model, out);
            }
            "response.output_item.added" => {
                self.start(None, None, out);
                let event: ItemEnvelope = parse(json, kind)?;
                self.item_added(event.output_index, event.item, out)?;
            }
            "response.content_part.added" => {
                let event: PartEnvelope = parse(json, kind)?;
                let slot =
                    self.resolve_message(kind, event.item_id.as_deref(), event.output_index, out)?;
                self.part_added(slot, event.content_index, &event.part, out)?;
            }
            "response.output_text.delta" | "response.refusal.delta" => {
                let event: DeltaEnvelope = parse(json, kind)?;
                let slot =
                    self.resolve_message(kind, event.item_id.as_deref(), event.output_index, out)?;
                let refusal = kind == "response.refusal.delta";
                if let Some(part) = self.text_block(slot, event.content_index, refusal, out)? {
                    out.push(Ok(part.delta(event.delta)));
                }
            }
            "response.content_part.done" => {
                let event: PartEnvelope = parse(json, kind)?;
                let slot =
                    self.resolve_message(kind, event.item_id.as_deref(), event.output_index, out)?;
                self.part_done(slot, event.content_index, &event.part, out);
            }
            "response.function_call_arguments.delta" => {
                let event: DeltaEnvelope = parse(json, kind)?;
                let slot = self.resolve(kind, event.item_id.as_deref(), event.output_index)?;
                match self.items.get_mut(slot) {
                    Some(OpenItem::FunctionCall { index, streamed }) => {
                        *streamed |= !event.delta.is_empty();
                        out.push(Ok(StreamEvent::tool_call_delta(*index, event.delta)));
                    }
                    _ => return Err(mismatched(kind, "function_call")),
                }
            }
            "response.reasoning_summary_part.added" => {
                let event: SummaryPartEnvelope = parse(json, kind)?;
                let slot = self.resolve(kind, event.item_id.as_deref(), event.output_index)?;
                let reasoning = self.reasoning(slot, kind)?;
                if event.summary_index > 0 && reasoning.source == Some(ReasoningSource::Summary) {
                    out.push(Ok(StreamEvent::reasoning_delta(
                        reasoning.index,
                        SUMMARY_SEPARATOR,
                    )));
                }
            }
            "response.reasoning_summary_text.delta" => {
                let event: DeltaEnvelope = parse(json, kind)?;
                let slot = self.resolve(kind, event.item_id.as_deref(), event.output_index)?;
                let reasoning = self.reasoning(slot, kind)?;
                reasoning.push(ReasoningSource::Summary, 0, event.delta, out);
            }
            // OpenAI's name for raw reasoning deltas, and the Open Responses
            // spec's name for the same event.
            "response.reasoning_text.delta" | "response.reasoning.delta" => {
                let event: DeltaEnvelope = parse(json, kind)?;
                let slot = self.resolve(kind, event.item_id.as_deref(), event.output_index)?;
                let reasoning = self.reasoning(slot, kind)?;
                reasoning.push(
                    ReasoningSource::Content,
                    event.content_index,
                    event.delta,
                    out,
                );
            }
            "response.output_item.done" => {
                self.start(None, None, out);
                let event: ItemEnvelope = parse(json, kind)?;
                self.item_done(event.output_index, event.item, out)?;
            }
            "response.completed" | "response.incomplete" => {
                let event: ResponseEnvelope = parse(json, kind)?;
                self.terminal(event.response, out);
            }
            "response.failed" => {
                let event: ResponseEnvelope = parse(json, kind)?;
                self.finished = true;
                let error = event.response.error.unwrap_or_default();
                out.push(Err(map_response_error(
                    &error,
                    &self.provider,
                    self.request_id.clone(),
                )));
            }
            "error" => {
                self.finished = true;
                out.push(Err(map_response_error(
                    &error_event(json),
                    &self.provider,
                    self.request_id.clone(),
                )));
            }
            // Hosted-tool progress becomes `ProviderDelta`s. Ignored: `*.done`
            // events whose content arrives with the item, annotation events
            // (annotations arrive with the finished part), and events added
            // after this adapter was written.
            _ if is_vendor_event(kind) => {
                let events = self.vendor_frames.surface_payload(kind, json);
                self.vendor_event(events, out);
            }
            _ => self.progress(kind, json, out),
        }
        Ok(())
    }

    /// The surfaced events of an event outside the Responses API: never
    /// decoded, and never the start of the response (`finish_stream` holds
    /// what arrives before it).
    fn vendor_event(&mut self, events: Vec<StreamEvent>, out: &mut Vec<Result<StreamEvent>>) {
        out.extend(events.into_iter().map(Ok));
    }

    fn start(
        &mut self,
        id: Option<String>,
        model: Option<String>,
        out: &mut Vec<Result<StreamEvent>>,
    ) {
        if self.response_id.is_none() {
            self.response_id.clone_from(&id);
        }
        if self.started {
            return;
        }
        self.started = true;
        out.push(Ok(StreamEvent::response_start(id, model)));
        for warning in self.pending_warnings.drain(..) {
            out.push(Ok(StreamEvent::warning(warning)));
        }
    }

    /// Surface a progress event of an open hosted tool call
    /// (`response.web_search_call.searching`, …) as a `ProviderDelta`.
    fn progress(&mut self, kind: &str, json: Value, out: &mut Vec<Result<StreamEvent>>) {
        let Value::Object(mut data) = json else {
            return;
        };
        let item_id = string(&data, "item_id");
        let output_index = data.get("output_index").and_then(Value::as_u64);
        let Some(OpenItem::Hosted { index }) = self
            .items
            .resolve(item_id.as_deref(), output_index)
            .and_then(|slot| self.items.get(slot))
        else {
            return;
        };
        let index = *index;
        for key in ["type", "item_id", "output_index", "sequence_number"] {
            data.remove(key);
        }
        out.push(Ok(StreamEvent::provider_delta(
            index,
            kind.strip_prefix("response.").unwrap_or(kind),
            data,
            Some(self.namespace.clone().into_owned()),
        )));
    }

    fn allocate(&mut self) -> usize {
        let index = self.next_index;
        self.next_index += 1;
        index
    }

    /// The open item an event refers to.
    fn resolve(
        &self,
        event: &str,
        item_id: Option<&str>,
        output_index: Option<u64>,
    ) -> Result<usize> {
        self.items.resolve(item_id, output_index).ok_or_else(|| {
            protocol(format!(
                "`{event}` for unknown output item (item_id {item_id:?}, output_index {output_index:?})"
            ))
        })
    }

    /// The message item a content event refers to. LiteLLM never announces
    /// its message items with `response.output_item.added`, so a content event
    /// for an unknown item that has an id opens the message implicitly.
    fn resolve_message(
        &mut self,
        event: &str,
        item_id: Option<&str>,
        output_index: Option<u64>,
        out: &mut Vec<Result<StreamEvent>>,
    ) -> Result<usize> {
        if let Some(slot) = self.items.resolve(item_id, output_index) {
            return Ok(slot);
        }
        let Some(id) = item_id else {
            return self.resolve(event, item_id, output_index);
        };
        self.start(None, None, out);
        let mut item = ExtraMap::new();
        item.insert("type".into(), Value::String("message".into()));
        item.insert("id".into(), Value::String(id.to_owned()));
        self.item_added(output_index, item, out)
    }

    fn reasoning(&mut self, slot: usize, event: &str) -> Result<&mut ReasoningItem> {
        match self.items.get_mut(slot) {
            Some(OpenItem::Reasoning(reasoning)) => Ok(reasoning),
            _ => Err(mismatched(event, "reasoning")),
        }
    }

    fn item_added(
        &mut self,
        output_index: Option<u64>,
        item: ExtraMap,
        out: &mut Vec<Result<StreamEvent>>,
    ) -> Result<usize> {
        let open = match item.get("type").and_then(Value::as_str) {
            Some("message") => OpenItem::Message {
                phase: item.get("phase").filter(|p| !p.is_null()).cloned(),
                parts: HashMap::new(),
            },
            Some("function_call") => {
                let call_id = string(&item, "call_id")
                    .ok_or_else(|| protocol("function_call item without `call_id`"))?;
                let name = string(&item, "name")
                    .ok_or_else(|| protocol("function_call item without `name`"))?;
                let index = self.allocate();
                self.flags.tool_calls = true;
                let mut start = StreamEvent::tool_call_start(index, call_id, name);
                if let StreamEvent::BlockStart { provider_data, .. } = &mut start {
                    *provider_data = decode::item_id_data(string(&item, "id"), &self.namespace);
                }
                out.push(Ok(start));
                let arguments = string(&item, "arguments").unwrap_or_default();
                let streamed = !arguments.is_empty();
                if streamed {
                    out.push(Ok(StreamEvent::tool_call_delta(index, arguments)));
                }
                OpenItem::FunctionCall { index, streamed }
            }
            Some("reasoning") => {
                let index = self.allocate();
                let mut start = StreamEvent::block_start(index, StreamBlockType::Reasoning);
                if let StreamEvent::BlockStart { provider_data, .. } = &mut start {
                    *provider_data = decode::item_id_data(string(&item, "id"), &self.namespace);
                }
                out.push(Ok(start));
                OpenItem::Reasoning(ReasoningItem {
                    index,
                    source: None,
                    content_index: 0,
                })
            }
            Some(kind) if super::hosted::is_hosted(kind) => {
                let id = string(&item, "id")
                    .ok_or_else(|| protocol(format!("`{kind}` item without an `id`")))?;
                let name = super::hosted::tool_name(&item)
                    .ok_or_else(|| protocol(format!("`{kind}` item without a tool name")))?;
                let index = self.allocate();
                let mut start = StreamEvent::provider_tool_call_start(
                    index,
                    id,
                    name,
                    Some(self.namespace.clone().into_owned()),
                );
                if let StreamEvent::BlockStart { provider_data, .. } = &mut start {
                    *provider_data = super::hosted::item_type_data(kind, &self.namespace);
                }
                out.push(Ok(start));
                OpenItem::Hosted { index }
            }
            Some(_) => OpenItem::Other,
            None => return Err(protocol("output item without a `type`")),
        };
        Ok(self.items.insert(output_index, string(&item, "id"), open))
    }

    fn part_added(
        &mut self,
        slot: usize,
        content_index: u64,
        part: &Value,
        out: &mut Vec<Result<StreamEvent>>,
    ) -> Result<()> {
        let kind = part
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        if let Some(OpenItem::Reasoning(reasoning)) = self.items.get_mut(slot) {
            // Raw reasoning (`reasoning_text`); the text usually follows as
            // deltas, but may be sent whole with the part.
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                reasoning.push(
                    ReasoningSource::Content,
                    content_index,
                    text.to_owned(),
                    out,
                );
            }
            return Ok(());
        }
        let (refusal, text) = match kind {
            "output_text" => (false, part.get("text")),
            "refusal" => (true, part.get("refusal")),
            other => {
                out.push(Ok(dropped(
                    format!(
                        "the adapter cannot decode message content part `{other}`; it was dropped"
                    ),
                    self.provider.clone(),
                )));
                let Some(OpenItem::Message { parts, .. }) = self.items.get_mut(slot) else {
                    return Err(mismatched("response.content_part.added", "message"));
                };
                parts.insert(content_index, Part::Ignored);
                return Ok(());
            }
        };
        let part = self.open_text(slot, content_index, refusal, out)?;
        if let Some(text) = text.and_then(Value::as_str).filter(|t| !t.is_empty()) {
            self.mark_streamed(slot, content_index);
            out.push(Ok(part.delta(text.to_owned())));
        }
        Ok(())
    }

    /// Start a text (or refusal) block for a message content part.
    fn open_text(
        &mut self,
        slot: usize,
        content_index: u64,
        refusal: bool,
        out: &mut Vec<Result<StreamEvent>>,
    ) -> Result<TextPart> {
        let part = TextPart {
            index: self.next_index,
            refusal,
        };
        let Some(OpenItem::Message { phase, parts }) = self.items.get_mut(slot) else {
            return Err(mismatched("text content", "message"));
        };
        let block_type = if refusal {
            StreamBlockType::Refusal
        } else {
            StreamBlockType::Text
        };
        let mut start = StreamEvent::block_start(part.index, block_type);
        if let StreamEvent::BlockStart { provider_data, .. } = &mut start {
            *provider_data = decode::text_data(None, phase.as_ref(), &self.namespace);
        }
        parts.insert(
            content_index,
            Part::Text {
                part,
                streamed: false,
            },
        );
        self.next_index += 1;
        self.flags.refused |= refusal;
        out.push(Ok(start));
        Ok(part)
    }

    fn mark_streamed(&mut self, slot: usize, content_index: u64) {
        if let Some(OpenItem::Message { parts, .. }) = self.items.get_mut(slot)
            && let Some(Part::Text { streamed, .. }) = parts.get_mut(&content_index)
        {
            *streamed = true;
        }
    }

    /// The block for a text or refusal delta, opening one of that kind if
    /// the part was never announced. `None` for ignored parts.
    fn text_block(
        &mut self,
        slot: usize,
        content_index: u64,
        refusal: bool,
        out: &mut Vec<Result<StreamEvent>>,
    ) -> Result<Option<TextPart>> {
        let existing = match self.items.get(slot) {
            Some(OpenItem::Message { parts, .. }) => parts.get(&content_index).copied(),
            _ => return Err(mismatched("text delta", "message")),
        };
        let part = match existing {
            Some(Part::Ignored) => return Ok(None),
            Some(Part::Text { part, .. }) => part,
            Some(Part::Closed) => {
                return Err(protocol(format!(
                    "text delta for finished content part {content_index}"
                )));
            }
            None => self.open_text(slot, content_index, refusal, out)?,
        };
        self.mark_streamed(slot, content_index);
        Ok(Some(part))
    }

    /// Close a message content part. Raw reasoning parts need no work here:
    /// the item's `output_item.done` reconciles them.
    fn part_done(
        &mut self,
        slot: usize,
        content_index: u64,
        part: &Value,
        out: &mut Vec<Result<StreamEvent>>,
    ) {
        if let Some(OpenItem::Message { parts, .. }) = self.items.get_mut(slot)
            && let Some(part_slot) = parts.get_mut(&content_index)
        {
            finish_part(part_slot, part, &self.namespace, out);
        }
    }

    fn item_done(
        &mut self,
        output_index: Option<u64>,
        item: ExtraMap,
        out: &mut Vec<Result<StreamEvent>>,
    ) -> Result<()> {
        let id = string(&item, "id");
        let slot = match self.items.resolve(id.as_deref(), output_index) {
            Some(slot) => slot,
            None => self.item_added(output_index, item.clone(), out)?,
        };
        let Some(open) = self.items.remove(slot) else {
            return Ok(());
        };
        match open {
            OpenItem::Message { mut parts, phase } => {
                let content = item.get("content").and_then(Value::as_array);
                // The parts that streamed, and those that arrive only with
                // the finished item (servers that send no content-part
                // events), in content order.
                let mut indices: Vec<u64> = parts.keys().copied().collect();
                indices.extend((0..content.map_or(0, Vec::len)).map(|i| i as u64));
                indices.sort_unstable();
                indices.dedup();
                for content_index in indices {
                    let part = content
                        .and_then(|c| usize::try_from(content_index).ok().and_then(|i| c.get(i)))
                        .cloned()
                        .unwrap_or(Value::Null);
                    if let std::collections::hash_map::Entry::Vacant(entry) =
                        parts.entry(content_index)
                    {
                        let Some(slot) = self.unannounced_part(&part, phase.as_ref(), out) else {
                            continue;
                        };
                        entry.insert(slot);
                    }
                    if let Some(slot) = parts.get_mut(&content_index) {
                        finish_part(slot, &part, &self.namespace, out);
                    }
                }
            }
            OpenItem::FunctionCall { index, streamed } => {
                if !streamed
                    && let Some(arguments) = string(&item, "arguments").filter(|a| !a.is_empty())
                {
                    out.push(Ok(StreamEvent::tool_call_delta(index, arguments)));
                }
                out.push(Ok(StreamEvent::block_stop(index)));
            }
            OpenItem::Reasoning(reasoning) => {
                let index = reasoning.index;
                let source = match reasoning.source {
                    Some(source) => source,
                    // Nothing streamed: decode the finished item like a
                    // unary response.
                    None => {
                        let (text, source) = decode::reasoning_text(&item);
                        if !text.is_empty() {
                            out.push(Ok(StreamEvent::reasoning_delta(index, text)));
                        }
                        source
                    }
                };
                let provider_data = decode::reasoning_data(&item, source, &self.namespace);
                if !provider_data.is_empty() {
                    out.push(Ok(StreamEvent::block_provider_data(index, provider_data)));
                }
                out.push(Ok(StreamEvent::block_stop(index)));
            }
            OpenItem::Hosted { index } => {
                let (call, result) = super::hosted::decode(&item, &self.namespace)
                    .ok_or_else(|| protocol("hosted tool item could not be decoded when done"))?;
                out.push(Ok(StreamEvent::tool_call_delta(index, call.arguments)));
                // Fields known only when done (e.g. an MCP `server_label`) are
                // needed to replay the item.
                out.push(Ok(StreamEvent::block_provider_data(
                    index,
                    call.provider_data,
                )));
                out.push(Ok(StreamEvent::block_stop(index)));
                let index = self.allocate();
                out.push(Ok(StreamEvent::provider_tool_result_start(index, result)));
                out.push(Ok(StreamEvent::block_stop(index)));
            }
            OpenItem::Other => {
                let mut warnings = Vec::new();
                let origin = Origin {
                    provider: &self.provider,
                    namespace: &self.namespace,
                };
                let blocks = decode::decode_item(&Value::Object(item), origin, &mut warnings)
                    .map_err(|e| protocol(e.message().to_owned()))?;
                for block in blocks {
                    if let ContentBlock::Other(other) = block {
                        let index = self.allocate();
                        let mut start = StreamEvent::block_start(index, StreamBlockType::Other);
                        if let StreamEvent::BlockStart {
                            type_name,
                            data,
                            origin,
                            ..
                        } = &mut start
                        {
                            *type_name = Some(other.type_name.clone());
                            *data = Some(other.data.clone());
                            *origin = other.origin.clone();
                        }
                        self.others.push(ContentBlock::Other(other));
                        out.push(Ok(start));
                        out.push(Ok(StreamEvent::block_stop(index)));
                    }
                }
            }
        }
        Ok(())
    }

    /// The block of a content part that only the finished item holds: an
    /// `output_text` or `refusal` part opens one, which `finish_part` fills
    /// and closes; any other part is dropped with a warning, as when it is
    /// announced.
    fn unannounced_part(
        &mut self,
        part: &Value,
        phase: Option<&Value>,
        out: &mut Vec<Result<StreamEvent>>,
    ) -> Option<Part> {
        let refusal = match part.get("type").and_then(Value::as_str) {
            Some("output_text") => false,
            Some("refusal") => true,
            other => {
                out.push(Ok(dropped(
                    format!(
                        "the adapter cannot decode message content part `{}`; it was dropped",
                        other.unwrap_or("unknown")
                    ),
                    self.provider.clone(),
                )));
                return None;
            }
        };
        let text_part = TextPart {
            index: self.allocate(),
            refusal,
        };
        let block_type = if refusal {
            StreamBlockType::Refusal
        } else {
            StreamBlockType::Text
        };
        let mut start = StreamEvent::block_start(text_part.index, block_type);
        if let StreamEvent::BlockStart { provider_data, .. } = &mut start {
            *provider_data = decode::text_data(None, phase, &self.namespace);
        }
        self.flags.refused |= refusal;
        out.push(Ok(start));
        Some(Part::Text {
            part: text_part,
            streamed: false,
        })
    }

    fn terminal(&mut self, response: ResponseObject, out: &mut Vec<Result<StreamEvent>>) {
        self.start(response.id.clone(), response.model.clone(), out);
        self.finished = true;
        self.flags.tool_calls |= crate::client_tools::has_pending(&self.others);
        let (reason, raw) = decode::finish_reason(
            response.status.as_deref(),
            response
                .incomplete_details
                .as_ref()
                .and_then(|d| d.reason.as_deref()),
            self.flags,
        );
        let mut metadata = StreamEvent::metadata();
        if let StreamEvent::ResponseMetadata {
            finish_reason,
            finish_reason_raw,
            usage,
            usage_mode,
            id,
            model,
            request_id,
            provider,
            provider_data,
            ..
        } = &mut metadata
        {
            *finish_reason = reason;
            *finish_reason_raw = raw;
            *usage = decode::usage(response.usage.as_ref(), response.tool_usage.as_ref());
            *usage_mode = UsageMetadataMode::Snapshot;
            *id = response.id;
            *model = response.model;
            request_id.clone_from(&self.request_id);
            *provider = Some(self.provider.clone());
            *provider_data = decode::response_provider_data(
                response.service_tier,
                response.tool_usage,
                &self.namespace,
            );
        }
        out.push(Ok(metadata));
        out.extend(self.vendor_frames.finish().map(Ok));
        out.push(Ok(StreamEvent::ResponseStop));
    }

    /// The error for a body that ended before a terminal event.
    fn incomplete(&mut self) -> Option<Error> {
        if self.finished {
            return None;
        }
        self.finished = true;
        Some(self.with_resume_point(incomplete(&self.provider, self.request_id.clone())))
    }
}

/// Open output items, addressable by `item_id` and by `output_index`.
#[derive(Default)]
struct Items {
    next_slot: usize,
    open: HashMap<usize, OpenItem>,
    by_id: HashMap<String, usize>,
    by_index: HashMap<u64, usize>,
}

impl Items {
    fn insert(&mut self, output_index: Option<u64>, id: Option<String>, item: OpenItem) -> usize {
        let slot = self.next_slot;
        self.next_slot += 1;
        self.open.insert(slot, item);
        if let Some(id) = id {
            self.by_id.insert(id, slot);
        }
        if let Some(output_index) = output_index {
            self.by_index.insert(output_index, slot);
        }
        slot
    }

    fn resolve(&self, item_id: Option<&str>, output_index: Option<u64>) -> Option<usize> {
        item_id
            .and_then(|id| self.by_id.get(id))
            .or_else(|| output_index.and_then(|i| self.by_index.get(&i)))
            .copied()
            .filter(|slot| self.open.contains_key(slot))
    }

    fn get(&self, slot: usize) -> Option<&OpenItem> {
        self.open.get(&slot)
    }

    fn get_mut(&mut self, slot: usize) -> Option<&mut OpenItem> {
        self.open.get_mut(&slot)
    }

    fn remove(&mut self, slot: usize) -> Option<OpenItem> {
        self.open.remove(&slot)
    }
}

enum OpenItem {
    Message {
        phase: Option<Value>,
        parts: HashMap<u64, Part>,
    },
    FunctionCall {
        index: usize,
        streamed: bool,
    },
    Reasoning(ReasoningItem),
    /// A hosted tool call: started when added, progress events become
    /// `ProviderDelta`s, and the call and its result complete when done.
    Hosted {
        index: usize,
    },
    /// Buffered until `response.output_item.done`.
    Other,
}

struct ReasoningItem {
    index: usize,
    /// The source streamed as the block text, once one has started.
    source: Option<ReasoningSource>,
    /// The raw content part currently streaming.
    content_index: u64,
}

impl ReasoningItem {
    /// Stream reasoning text. The first source to arrive becomes the block
    /// text; the other stays in provider data (`summary` / `content`).
    fn push(
        &mut self,
        source: ReasoningSource,
        content_index: u64,
        text: String,
        out: &mut Vec<Result<StreamEvent>>,
    ) {
        if text.is_empty() {
            return;
        }
        match self.source {
            None => {
                self.source = Some(source);
                self.content_index = content_index;
            }
            Some(current) if current != source => return,
            Some(ReasoningSource::Content) if content_index != self.content_index => {
                self.content_index = content_index;
                out.push(Ok(StreamEvent::reasoning_delta(
                    self.index,
                    SUMMARY_SEPARATOR,
                )));
            }
            Some(_) => {}
        }
        out.push(Ok(StreamEvent::reasoning_delta(self.index, text)));
    }
}

#[derive(Debug, Clone, Copy)]
enum Part {
    Text { part: TextPart, streamed: bool },
    Ignored,
    Closed,
}

/// The block of a message content part that streams text: an
/// `output_text` part, or a `refusal` part, whose text is never a text
/// delta.
#[derive(Debug, Clone, Copy)]
struct TextPart {
    index: usize,
    refusal: bool,
}

impl TextPart {
    fn delta(self, text: String) -> StreamEvent {
        if self.refusal {
            StreamEvent::refusal_delta(self.index, text)
        } else {
            StreamEvent::text_delta(self.index, text)
        }
    }
}

/// Close a text or refusal part: reconcile text that never streamed,
/// attach annotations and logprobs, and stop the block.
fn finish_part(slot: &mut Part, part: &Value, namespace: &str, out: &mut Vec<Result<StreamEvent>>) {
    let Part::Text {
        part: text_part,
        streamed,
    } = *slot
    else {
        return;
    };
    let index = text_part.index;
    let text = part
        .get("text")
        .or_else(|| part.get("refusal"))
        .and_then(Value::as_str)
        .filter(|t| !t.is_empty());
    if let (false, Some(text)) = (streamed, text) {
        out.push(Ok(text_part.delta(text.to_owned())));
    }
    let full_text = part.get("text").and_then(Value::as_str).unwrap_or_default();
    out.extend(
        crate::citations::citations(part.get("annotations"), full_text, namespace)
            .into_iter()
            .map(|citation| Ok(StreamEvent::citation(index, citation))),
    );
    let data = decode::text_data(Some(part), None, namespace);
    if !data.is_empty() {
        out.push(Ok(StreamEvent::block_provider_data(index, data)));
    }
    out.push(Ok(StreamEvent::block_stop(index)));
    *slot = Part::Closed;
}

#[derive(Deserialize)]
struct ResponseEnvelope {
    #[serde(default)]
    response: ResponseObject,
}

// `output_index` is required by OpenAI and the Open Responses spec, but
// llama.cpp omits it and identifies items by `item_id` only.

#[derive(Deserialize)]
struct ItemEnvelope {
    #[serde(default)]
    output_index: Option<u64>,
    item: ExtraMap,
}

#[derive(Deserialize)]
struct PartEnvelope {
    #[serde(default)]
    output_index: Option<u64>,
    #[serde(default)]
    item_id: Option<String>,
    #[serde(default)]
    content_index: u64,
    #[serde(default)]
    part: Value,
}

#[derive(Deserialize)]
struct DeltaEnvelope {
    #[serde(default)]
    output_index: Option<u64>,
    #[serde(default)]
    item_id: Option<String>,
    #[serde(default)]
    content_index: u64,
    /// Some servers send `"delta": null`.
    #[serde(default, deserialize_with = "null_as_empty")]
    delta: String,
}

#[derive(Deserialize)]
struct SummaryPartEnvelope {
    #[serde(default)]
    output_index: Option<u64>,
    #[serde(default)]
    item_id: Option<String>,
    #[serde(default)]
    summary_index: u64,
}

fn null_as_empty<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<String, D::Error> {
    Ok(Option::<String>::deserialize(deserializer)?.unwrap_or_default())
}

/// `error` events carry the error inline (`{type: "error", code, message}`)
/// or nested under `error`.
fn error_event(json: Value) -> ResponseError {
    let nested = json.get("error").filter(|e| e.is_object()).cloned();
    let mut error: ResponseError =
        serde_json::from_value(nested.unwrap_or(json)).unwrap_or_default();
    if error.error_type.as_deref() == Some("error") {
        error.error_type = None;
    }
    if error.message.is_empty() && error.code.is_none() {
        error.message = "malformed `error` event".into();
    }
    error
}

fn parse<T: serde::de::DeserializeOwned>(json: Value, event: &str) -> Result<T> {
    serde_json::from_value(json).map_err(|e| protocol(format!("malformed `{event}` event: {e}")))
}

fn mismatched(event: &str, expected: &str) -> Error {
    protocol(format!(
        "`{event}` for an output item that is not a {expected}"
    ))
}

/// Whether an event (by SSE name or `type`) is outside the Responses API:
/// every Responses event is `error` or in the `response.` namespace, and
/// unnamed SSE events (`message`) carry their type in their data.
fn is_vendor_event(name: &str) -> bool {
    !matches!(name, "" | "message" | "error") && !name.starts_with("response.")
}

fn string(map: &ExtraMap, key: &str) -> Option<String> {
    map.get(key).and_then(Value::as_str).map(str::to_owned)
}

fn dropped(message: String, provider: Cow<'static, str>) -> StreamEvent {
    StreamEvent::warning(
        Warning::new(WarningKind::Dropped, Subject::Response, message).with_provider(provider),
    )
}

/// The decoder `SseChatStream` drives. A failed read ends the stream with
/// the failure, the provider, the request id and the resume point.
impl ChatFrameDecoder<SseEvent> for SseState {
    fn frame(&mut self, event: SseEvent) -> Vec<Result<StreamEvent>> {
        self.process(&event.event, &event.data)
    }

    fn end(&mut self) -> Vec<Result<StreamEvent>> {
        self.incomplete().map(Err).into_iter().collect()
    }

    fn fail(&mut self, error: Error) -> Vec<Result<StreamEvent>> {
        self.finished = true;
        let error = error
            .with_provider(self.provider.as_ref())
            .with_request_id_opt(self.request_id.as_deref());
        vec![Err(self.with_resume_point(error))]
    }

    fn is_done(&self) -> bool {
        self.finished
    }
}

#[cfg(test)]
mod tests;
