//! The Gemini response decoder: one pure, synchronous state machine shared by
//! the unary and the streaming paths.
//!
//! `streamGenerateContent` chunks are **deltas** (verified live on
//! 2026-10-02): each chunk carries only the new parts. The decoder turns
//! chunks into [`StreamEvent`]s; a unary `generateContent` response is decoded
//! as a single chunk and collected with [`StreamCollector`].
//!
//! Block rules:
//!
//! - Consecutive text parts of the same kind (answer text vs `thought: true`)
//!   append to the open block. A kind change, or any non-text part, closes it.
//! - A part carrying a `thoughtSignature` closes its block after it, so a
//!   signature is never merged with later parts. An empty text part that only
//!   carries a signature (Gemini 3 sends these at the end of a turn) attaches
//!   the signature to the open block of the same kind.
//! - `functionCall` parts are complete blocks (start, full arguments, stop).
//!   The call's `id` becomes [`ToolCall::id`](allama::ToolCall); Gemini 2.5
//!   omits it, so a deterministic synthetic id is used instead.
//! - `executableCode` and `codeExecutionResult` parts (the `code_execution`
//!   built-in tool) become [`ProviderToolCall`](allama::ProviderToolCall) and
//!   [`ProviderToolResult`](allama::ProviderToolResult) blocks with origin
//!   `"google-ai"`. Gemini sends no ids, so a synthetic
//!   `gemini_<responseId>_code_<n>` id links each result to the code before it.
//! - Image `inlineData`/`fileData` becomes an image block; other media
//!   (audio, …) and unknown part kinds become [`OtherContent`](allama::OtherContent)
//!   with origin `"google-ai"`, replayed verbatim.
//! - Answer text blocks stay open (their `BlockStop` is held back) until the
//!   candidate ends, because `groundingMetadata` arrives with the last chunk:
//!   its supports become [`Citation`](allama::Citation)s on the answer blocks
//!   their byte offsets fall in, and the raw metadata is kept in the response
//!   provider data under `"google-ai"` / `"grounding_metadata"`.
//! - `thoughtSignature` is stored in the block's provider data under
//!   `"google-ai"` / `"thought_signature"`, for `Other` content too. Thought (reasoning) blocks also
//!   carry `"google-ai"` / `"thought": true`, recording that the reasoning
//!   text is a Gemini thought summary that may be replayed to Gemini.
//!
//! A stream that ends without a `finishReason` fails with
//! [`StreamErrorKind::Incomplete`].

use allama::chat::stream::{StreamBlockType, StreamCollector};
use allama::{
    ChatResponse, Citation, CitationSource, Error, ExtraMap, FinishReason, Image, MediaSource,
    OtherContent, ProviderData, ProviderToolResult, Result, StreamErrorKind, StreamEvent, Subject,
    Usage, Warning, WarningKind,
};
use allama_http::sse::VendorFrames;
use serde_json::Value;

use crate::error::map_api_error;
use crate::wire::{GenerateContentResponse, PromptFeedback, ResponsePart, UsageMetadata};
use crate::{NAMESPACE, PROVIDER_NAME};

/// Provider-data key holding a part's `thoughtSignature`.
pub(crate) const THOUGHT_SIGNATURE_KEY: &str = "thought_signature";
/// Provider-data key marking a reasoning block as a Gemini thought part.
pub(crate) const THOUGHT_KEY: &str = "thought";
/// Part key of code Gemini runs itself (the `code_execution` tool).
pub(crate) const EXECUTABLE_CODE: &str = "executableCode";
/// Part key of a code-execution outcome.
pub(crate) const CODE_EXECUTION_RESULT: &str = "codeExecutionResult";
/// Provider tool name of Gemini code execution.
pub(crate) const CODE_EXECUTION_TOOL: &str = "code_execution";
/// Response provider-data key holding the raw `groundingMetadata`.
pub(crate) const GROUNDING_KEY: &str = "grounding_metadata";
/// Response provider-data key holding the raw `citationMetadata`.
pub(crate) const CITATION_METADATA_KEY: &str = "citation_metadata";

/// Decode a unary `generateContent` response through the shared decoder.
pub(crate) fn decode_response(
    response: GenerateContentResponse,
    warnings: Vec<Warning>,
    request_id: Option<String>,
) -> Result<ChatResponse> {
    let mut decoder = Decoder::new(warnings).with_request_id(request_id);
    let mut collector = StreamCollector::new();
    for event in decoder.push(response)? {
        collector.push(event)?;
    }
    for event in decoder.finish(End::Complete)? {
        collector.push(event)?;
    }
    collector.finish()
}

/// How the input ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum End {
    /// A complete unary body: a missing `finishReason` is not an error.
    Complete,
    /// End of an SSE stream: a missing `finishReason` means truncation.
    Stream,
}

/// Incremental decoder from Gemini chunks to [`StreamEvent`]s.
pub(crate) struct Decoder {
    pending_warnings: Vec<Warning>,
    started: bool,
    response_id: Option<String>,
    model: Option<String>,
    anonymous_scope: Option<String>,
    next_index: usize,
    open: Option<OpenBlock>,
    tool_calls: usize,
    /// Code-execution calls so far (for synthetic ids).
    code_calls: usize,
    /// Id of the latest `executableCode` part: a `codeExecutionResult` answers it.
    last_code_call: Option<String>,
    finish_reason: Option<String>,
    finish_message: Option<String>,
    usage: Option<UsageMetadata>,
    /// Answer text blocks of the candidate, for placing grounding citations.
    answer_blocks: Vec<AnswerBlock>,
    /// Length of the candidate's answer text so far (grounding offsets are
    /// bytes into it).
    answer_len: usize,
    /// Answer blocks closed before the candidate ended; their `BlockStop` waits
    /// for the grounding, which arrives with the last chunk.
    deferred_stops: Vec<usize>,
    /// Grounding not yet attached.
    grounding: Option<Value>,
    /// Grounding already attached, kept for the response provider data.
    attached_grounding: Option<Value>,
    /// `citationMetadata` not yet attached.
    recitation: Option<Value>,
    /// `citationMetadata` already attached, kept for the response provider
    /// data.
    attached_recitation: Option<Value>,
    /// Response and candidate fields this adapter does not model, by
    /// provider-data key; a later chunk's value replaces an earlier one.
    unmodeled: ExtraMap,
    /// Named stream events, which the API does not send.
    vendor_frames: VendorFrames,
    /// The request id from the response headers, for errors.
    request_id: Option<String>,
}

#[derive(Debug)]
struct AnswerBlock {
    index: usize,
    start: usize,
    text: String,
}

#[derive(Debug, Clone, Copy)]
struct OpenBlock {
    index: usize,
    kind: TextKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextKind {
    Text,
    Thought,
}

impl Decoder {
    /// Create a decoder; `warnings` (from request mapping) are emitted right
    /// after `ResponseStart`.
    pub(crate) fn new(warnings: Vec<Warning>) -> Self {
        Self {
            pending_warnings: warnings,
            started: false,
            response_id: None,
            model: None,
            anonymous_scope: None,
            next_index: 0,
            open: None,
            tool_calls: 0,
            code_calls: 0,
            last_code_call: None,
            finish_reason: None,
            finish_message: None,
            usage: None,
            answer_blocks: Vec::new(),
            answer_len: 0,
            deferred_stops: Vec::new(),
            grounding: None,
            attached_grounding: None,
            recitation: None,
            attached_recitation: None,
            unmodeled: ExtraMap::new(),
            vendor_frames: VendorFrames::new(PROVIDER_NAME, NAMESPACE),
            request_id: None,
        }
    }

    /// Set the request id errors carry (Gemini sends none; a proxy's is
    /// kept).
    pub(crate) fn with_request_id(mut self, request_id: Option<String>) -> Self {
        self.request_id = request_id;
        self
    }

    /// Feed a named stream event the API does not send (Gemini streams are
    /// unnamed `data:` frames): its data is surfaced verbatim as response
    /// provider data under the event name (an array of payloads, in arrival
    /// order), with one warning per event name. It never starts the
    /// response (`finish_stream` holds what arrives before it).
    pub(crate) fn unknown_event(&mut self, name: &str, data: &str) -> Vec<StreamEvent> {
        self.vendor_frames.surface(name, data)
    }

    /// The request id errors carry.
    pub(crate) fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    /// Feed one chunk (or the whole unary response).
    pub(crate) fn push(&mut self, chunk: GenerateContentResponse) -> Result<Vec<StreamEvent>> {
        if let Some(error) = &chunk.error {
            return Err(map_api_error(None, error, allama::ErrorContext::new())
                .with_request_id_opt(self.request_id.as_deref()));
        }
        if let Some(error) = chunk
            .prompt_feedback
            .as_ref()
            .and_then(prompt_feedback_error)
        {
            let error = error.with_request_id_opt(self.request_id.as_deref());
            return Err(match &chunk.usage_metadata {
                Some(usage) => with_usage(error, &usage_from_metadata(usage)),
                None => error,
            });
        }

        let mut out = Vec::new();
        self.keep_unmodeled(chunk.extra);
        if let Some(feedback) = chunk.prompt_feedback.filter(|f| !f.extra.is_empty()) {
            self.unmodeled
                .insert("prompt_feedback".into(), Value::Object(feedback.extra));
        }
        if chunk.response_id.is_some() {
            self.response_id = chunk.response_id;
        }
        if chunk.model_version.is_some() {
            self.model = chunk.model_version;
        }
        self.ensure_started(&mut out);
        if chunk.usage_metadata.is_some() {
            // Usage metadata is a cumulative snapshot; keep the latest.
            self.usage = chunk.usage_metadata;
        }

        if let Some(candidate) = chunk.candidates.into_iter().next() {
            if let Some(content) = candidate.content {
                for part in content.parts {
                    self.part(part, &mut out)?;
                }
            }
            if let Some(grounding) = candidate.grounding_metadata.filter(|g| !g.is_null()) {
                self.grounding = Some(grounding);
            }
            if let Some(recitation) = candidate.citation_metadata.filter(|c| !c.is_null()) {
                self.recitation = Some(recitation);
            }
            let mut extra = candidate.extra;
            // The candidate's position: always 0, as one candidate is asked for.
            extra.remove("index");
            self.keep_unmodeled(extra);
            if let Some(reason) = candidate.finish_reason {
                self.end_candidate(&mut out);
                self.finish_reason = Some(reason);
            }
            if candidate.finish_message.is_some() {
                self.finish_message = candidate.finish_message;
            }
        }
        Ok(out)
    }

    /// Signal the end of input and emit the terminal events.
    pub(crate) fn finish(&mut self, end: End) -> Result<Vec<StreamEvent>> {
        if end == End::Stream && self.finish_reason.is_none() {
            return Err(Error::stream(
                StreamErrorKind::Incomplete,
                "Gemini stream ended without a finishReason",
            )
            .authored()
            .with_provider(PROVIDER_NAME));
        }

        let mut out = Vec::new();
        self.ensure_started(&mut out);
        self.end_candidate(&mut out);

        let finish_reason = self.finish_reason.as_deref().map(|raw| {
            let (reason, warning) = map_finish_reason(raw, self.tool_calls > 0);
            if let Some((limitation, outcome)) = warning {
                let detail = self
                    .finish_message
                    .as_deref()
                    .map(|m| format!(" ({m})"))
                    .unwrap_or_default();
                out.push(StreamEvent::warning(
                    Warning::new(
                        WarningKind::Other,
                        Subject::Response,
                        format!("{limitation}{detail}; {outcome}"),
                    )
                    .with_provider(PROVIDER_NAME),
                ));
            }
            reason
        });

        let mut provider_data = ProviderData::new();
        if let Some(message) = self.finish_message.take() {
            // A safety finish's `finishMessage` explains the decline; its
            // text is the decline explanation every adapter stores under the
            // same key.
            if matches!(
                finish_reason,
                Some(FinishReason::ContentFilter | FinishReason::Refusal)
            ) {
                provider_data.insert_raw(
                    NAMESPACE,
                    allama::provider_data::DECLINE_EXPLANATION_KEY,
                    Value::String(message.clone()),
                );
            }
            provider_data.insert_raw(NAMESPACE, "finish_message", Value::String(message));
        }
        if let Some(grounding) = self.attached_grounding.take() {
            provider_data.insert_raw(NAMESPACE, GROUNDING_KEY, grounding);
        }
        if let Some(recitation) = self.attached_recitation.take() {
            provider_data.insert_raw(NAMESPACE, CITATION_METADATA_KEY, recitation);
        }
        if let Some(usage) = &self.usage {
            if let Some(tier) = &usage.service_tier {
                provider_data.insert_raw(NAMESPACE, "service_tier", Value::String(tier.clone()));
            }
            if !usage.extra.is_empty() {
                provider_data.insert_raw(NAMESPACE, "usage", Value::Object(usage.extra.clone()));
            }
        }
        for (key, value) in std::mem::take(&mut self.unmodeled) {
            provider_data.insert_raw(NAMESPACE, &key, value);
        }
        let mut metadata = StreamEvent::metadata();
        if let StreamEvent::ResponseMetadata {
            finish_reason: finish_reason_slot,
            finish_reason_raw,
            usage,
            id,
            model,
            provider,
            provider_data: provider_data_slot,
            ..
        } = &mut metadata
        {
            *finish_reason_slot = finish_reason;
            finish_reason_raw.clone_from(&self.finish_reason);
            *usage = self.usage.as_ref().map(usage_from_metadata);
            id.clone_from(&self.response_id);
            model.clone_from(&self.model);
            *provider = Some(PROVIDER_NAME.into());
            *provider_data_slot = provider_data;
        }
        out.push(metadata);
        out.extend(self.vendor_frames.finish());
        out.push(StreamEvent::ResponseStop);
        Ok(out)
    }

    fn ensure_started(&mut self, out: &mut Vec<StreamEvent>) {
        if self.started {
            return;
        }
        self.started = true;
        out.push(StreamEvent::response_start(
            self.response_id.clone(),
            self.model.clone(),
        ));
        out.extend(self.pending_warnings.drain(..).map(StreamEvent::warning));
    }

    fn part(&mut self, part: ResponsePart, out: &mut Vec<StreamEvent>) -> Result<()> {
        let ResponsePart {
            text,
            thought,
            thought_signature,
            function_call,
            inline_data,
            file_data,
            other,
        } = part;

        if let Some(call) = function_call {
            self.close_open(out);
            let index = self.alloc();
            let id = call
                .id
                .filter(|id| !id.is_empty())
                .unwrap_or_else(|| self.synthetic_tool_call_id());
            let arguments = match call.args {
                Some(args) => serde_json::to_string(&args)?,
                None => "{}".to_owned(),
            };
            let mut start = StreamEvent::tool_call_start(index, id, call.name);
            if let StreamEvent::BlockStart { provider_data, .. } = &mut start {
                *provider_data = signature_data(thought_signature);
            }
            out.push(start);
            out.push(StreamEvent::tool_call_delta(index, arguments));
            out.push(StreamEvent::block_stop(index));
            self.tool_calls += 1;
            return Ok(());
        }

        if let Some(blob) = inline_data {
            self.close_open(out);
            if blob.mime_type.starts_with("image/") {
                let image = Image::new(MediaSource::base64(blob.mime_type, blob.data));
                self.media_block(image, thought, thought_signature, out);
            } else {
                let mut data = ExtraMap::new();
                data.insert("inlineData".into(), serde_json::to_value(&blob)?);
                self.other_block("inline_data", data, thought_signature, out);
            }
            return Ok(());
        }

        if let Some(file) = file_data {
            self.close_open(out);
            let is_image = file
                .mime_type
                .as_deref()
                .is_some_and(|m| m.starts_with("image/"));
            if is_image {
                // The MIME type stays with the image: replaying a URL image
                // to Gemini needs it.
                let mut image = Image::new(MediaSource::url(file.file_uri));
                image.media_type = file.mime_type;
                self.media_block(image, thought, thought_signature, out);
            } else {
                let mut data = ExtraMap::new();
                data.insert("fileData".into(), serde_json::to_value(&file)?);
                self.other_block("file_data", data, thought_signature, out);
            }
            return Ok(());
        }

        if let Some(text) = text {
            let kind = if thought == Some(true) {
                TextKind::Thought
            } else {
                TextKind::Text
            };
            self.text(kind, text, thought_signature, out);
            return Ok(());
        }

        if let Some(code) = other.get(EXECUTABLE_CODE) {
            self.close_open(out);
            self.code_call(code, thought_signature, out)?;
            return Ok(());
        }
        if let Some(result) = other.get(CODE_EXECUTION_RESULT) {
            self.close_open(out);
            self.code_result(result, thought_signature, out);
            return Ok(());
        }

        if let Some(type_name) = other.keys().next().cloned() {
            self.close_open(out);
            let mut data = other;
            if let Some(thought) = thought {
                data.insert("thought".into(), Value::Bool(thought));
            }
            self.other_block(&type_name, data, thought_signature, out);
            return Ok(());
        }

        if thought_signature.is_some() {
            // A payload-less part carrying only a signature: attach it to the
            // open block (of either kind), or keep it as an empty text block.
            let kind = self.open.map_or(TextKind::Text, |open| open.kind);
            self.text(kind, String::new(), thought_signature, out);
        }
        Ok(())
    }

    fn text(
        &mut self,
        kind: TextKind,
        text: String,
        signature: Option<String>,
        out: &mut Vec<StreamEvent>,
    ) {
        let has_signature = signature.is_some();
        match self.open {
            Some(open) if open.kind == kind => {
                if !text.is_empty() {
                    self.record_answer(kind, &text);
                    out.push(delta(open.index, kind, text));
                }
                if has_signature {
                    out.push(StreamEvent::block_provider_data(
                        open.index,
                        signature_data(signature),
                    ));
                }
            }
            _ => {
                self.close_open(out);
                if text.is_empty() && !has_signature {
                    return;
                }
                let index = self.alloc();
                let mut provider_data = signature_data(signature);
                let block_type = match kind {
                    TextKind::Text => StreamBlockType::Text,
                    TextKind::Thought => {
                        provider_data.insert_raw(NAMESPACE, THOUGHT_KEY, Value::Bool(true));
                        StreamBlockType::Reasoning
                    }
                };
                let mut start = StreamEvent::block_start(index, block_type);
                if let StreamEvent::BlockStart {
                    provider_data: slot,
                    ..
                } = &mut start
                {
                    *slot = provider_data;
                }
                out.push(start);
                self.open = Some(OpenBlock { index, kind });
                if kind == TextKind::Text {
                    self.answer_blocks.push(AnswerBlock {
                        index,
                        start: self.answer_len,
                        text: String::new(),
                    });
                }
                if !text.is_empty() {
                    self.record_answer(kind, &text);
                    out.push(delta(index, kind, text));
                }
            }
        }
        if has_signature {
            // A signature terminates its part: never merge later parts into it.
            self.close_open(out);
        }
    }

    /// An image part. A draft image Gemini made while thinking
    /// (`thought: true`) keeps the flag under `"google-ai"` / `"thought"`, so
    /// it is replayed as one.
    fn media_block(
        &mut self,
        image: Image,
        thought: Option<bool>,
        signature: Option<String>,
        out: &mut Vec<StreamEvent>,
    ) {
        let index = self.alloc();
        let mut provider_data = signature_data(signature);
        if thought == Some(true) {
            provider_data.insert_raw(NAMESPACE, THOUGHT_KEY, Value::Bool(true));
        }
        let image = image.with_provider_data(provider_data);
        out.push(StreamEvent::image_start(index, image));
        out.push(StreamEvent::block_stop(index));
    }

    fn other_block(
        &mut self,
        type_name: &str,
        data: ExtraMap,
        signature: Option<String>,
        out: &mut Vec<StreamEvent>,
    ) {
        let index = self.alloc();
        let other = OtherContent::new(type_name, data)
            .with_origin(NAMESPACE)
            .with_provider_data(signature_data(signature));
        out.push(StreamEvent::other_start(index, other));
        out.push(StreamEvent::block_stop(index));
    }

    fn close_open(&mut self, out: &mut Vec<StreamEvent>) {
        match self.open.take() {
            Some(open) if open.kind == TextKind::Text => self.deferred_stops.push(open.index),
            Some(open) => out.push(StreamEvent::block_stop(open.index)),
            None => {}
        }
    }

    /// The candidate is complete: attach grounding citations to its answer
    /// blocks, then stop them.
    fn end_candidate(&mut self, out: &mut Vec<StreamEvent>) {
        self.close_open(out);
        if let Some(grounding) = self.grounding.take() {
            let unplaced = grounding_citations(&grounding, &self.answer_blocks, out);
            if unplaced > 0 {
                out.push(StreamEvent::warning(Warning::new(
                        WarningKind::Other,
                        Subject::Response,
                        format!(
                            "{unplaced} grounding support(s) did not match the answer text; they were kept only in the response provider data (`{GROUNDING_KEY}`)"
                        ),
                    )
                    .with_provider(PROVIDER_NAME)));
            }
            self.attached_grounding = Some(grounding);
        }
        if let Some(recitation) = self.recitation.take() {
            let unplaced = recitation_citations(&recitation, &self.answer_blocks, out);
            if unplaced > 0 {
                out.push(StreamEvent::warning(Warning::new(
                        WarningKind::Other,
                        Subject::Response,
                        format!(
                            "{unplaced} recitation source(s) had no URL or did not match the answer text; they were kept only in the response provider data (`{CITATION_METADATA_KEY}`)"
                        ),
                    )
                    .with_provider(PROVIDER_NAME)));
            }
            self.attached_recitation = Some(recitation);
        }
        out.extend(self.deferred_stops.drain(..).map(StreamEvent::block_stop));
    }

    /// Track answer text so grounding offsets can be mapped to blocks.
    fn record_answer(&mut self, kind: TextKind, text: &str) {
        if kind != TextKind::Text {
            return;
        }
        if let Some(block) = self.answer_blocks.last_mut() {
            block.text.push_str(text);
        }
        self.answer_len += text.len();
    }

    /// Keep fields this adapter does not model, under snake_case keys
    /// (`safetyRatings` → `safety_ratings`).
    fn keep_unmodeled(&mut self, fields: ExtraMap) {
        for (key, value) in fields.into_iter().filter(|(_, v)| !v.is_null()) {
            self.unmodeled.insert(snake_case(&key), value);
        }
    }

    fn alloc(&mut self) -> usize {
        let index = self.next_index;
        self.next_index += 1;
        index
    }

    /// An `executableCode` part: code Gemini runs itself, as a provider tool
    /// call. Gemini sends no id, so a synthetic one links it to the result.
    fn code_call(
        &mut self,
        code: &Value,
        signature: Option<String>,
        out: &mut Vec<StreamEvent>,
    ) -> Result<()> {
        let n = self.code_calls;
        self.code_calls += 1;
        let id = format!("gemini_{}_code_{n}", self.scope());
        self.last_code_call = Some(id.clone());
        let index = self.alloc();
        let mut start = StreamEvent::provider_tool_call_start(
            index,
            id,
            CODE_EXECUTION_TOOL,
            Some(NAMESPACE.to_owned()),
        );
        if let StreamEvent::BlockStart { provider_data, .. } = &mut start {
            *provider_data = signature_data(signature);
        }
        out.push(start);
        out.push(StreamEvent::tool_call_delta(
            index,
            serde_json::to_string(code)?,
        ));
        out.push(StreamEvent::block_stop(index));
        Ok(())
    }

    /// A `codeExecutionResult` part: the outcome of the latest code call.
    fn code_result(
        &mut self,
        result: &Value,
        signature: Option<String>,
        out: &mut Vec<StreamEvent>,
    ) {
        let call_id = self
            .last_code_call
            .clone()
            .unwrap_or_else(|| format!("gemini_{}_code_unmatched", self.scope()));
        let failed = result.get("outcome").and_then(Value::as_str) != Some("OUTCOME_OK");
        let index = self.alloc();
        let mut tool_result = ProviderToolResult::new(call_id, result.clone())
            .with_error(failed)
            .with_origin(NAMESPACE);
        tool_result.provider_data = signature_data(signature);
        out.push(StreamEvent::provider_tool_result_start(index, tool_result));
        out.push(StreamEvent::block_stop(index));
    }

    /// The scope of synthetic ids: the response id with characters outside
    /// `[A-Za-z0-9_-]` replaced by `_`, or `anonymous_<uuid>` when the
    /// response has none.
    fn scope(&mut self) -> String {
        match self.response_id.as_deref().filter(|id| !id.is_empty()) {
            Some(id) => id
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect(),
            None => self
                .anonymous_scope
                .get_or_insert_with(|| format!("anonymous_{}", uuid::Uuid::new_v4().simple()))
                .clone(),
        }
    }

    fn synthetic_tool_call_id(&mut self) -> String {
        let n = self.tool_calls;
        synthetic_tool_call_id(&self.scope(), n)
    }
}

/// Emit a citation per grounding chunk of every `groundingSupports` entry,
/// on the answer block its segment falls in. Returns how many supports could
/// not be placed.
///
/// Segment offsets are bytes into the candidate's answer text (verified on
/// recorded unary and streaming responses); the segment `text` is used to
/// check the span and to recover it when offsets disagree.
fn grounding_citations(
    grounding: &Value,
    blocks: &[AnswerBlock],
    out: &mut Vec<StreamEvent>,
) -> usize {
    let empty = Vec::new();
    let chunks = grounding
        .get("groundingChunks")
        .and_then(Value::as_array)
        .unwrap_or(&empty);
    let supports = grounding
        .get("groundingSupports")
        .and_then(Value::as_array)
        .unwrap_or(&empty);
    let index = |value: Option<&Value>| {
        value
            .and_then(Value::as_u64)
            .and_then(|i| usize::try_from(i).ok())
    };
    let mut unplaced = 0;
    for support in supports {
        let segment = support.get("segment");
        let start = index(segment.and_then(|s| s.get("startIndex"))).unwrap_or(0);
        let end = index(segment.and_then(|s| s.get("endIndex")));
        let segment_text = segment.and_then(|s| s.get("text")).and_then(Value::as_str);
        let Some(block) = blocks
            .iter()
            .find(|b| start >= b.start && start < b.start + b.text.len())
        else {
            unplaced += 1;
            continue;
        };
        let span = end.filter(|end| *end > start).and_then(|end| {
            let local = start - block.start..end.min(block.start + block.text.len()) - block.start;
            match (block.text.get(local.clone()), segment_text) {
                (Some(found), Some(expected)) if found == expected => Some(local),
                (Some(_), None) => Some(local),
                (_, Some(expected)) => block.text.find(expected).map(|at| at..at + expected.len()),
                (None, None) => None,
            }
        });
        let scores = support.get("confidenceScores").and_then(Value::as_array);
        let chunk_indices = support
            .get("groundingChunkIndices")
            .and_then(Value::as_array)
            .unwrap_or(&empty);
        for (position, chunk_index) in chunk_indices.iter().enumerate() {
            let Some(source) = index(Some(chunk_index))
                .and_then(|i| chunks.get(i))
                .and_then(grounding_source)
            else {
                continue;
            };
            let mut citation = Citation::new(source);
            if let Some(span) = &span {
                citation = citation.with_span(span.clone());
            }
            let mut provider_data = ProviderData::new().with_raw(
                NAMESPACE,
                "grounding_chunk_index",
                chunk_index.clone(),
            );
            if let Some(score) = scores.and_then(|s| s.get(position)) {
                provider_data.insert_raw(NAMESPACE, "confidence_score", score.clone());
            }
            out.push(StreamEvent::citation(
                block.index,
                citation.with_provider_data(provider_data),
            ));
        }
    }
    unplaced
}

/// Emit a citation per `citationMetadata.citationSources` entry (material
/// the answer recites), on the answer block its start offset falls in.
/// Returns how many sources could not be placed: no URL, or no matching
/// block. Offsets are bytes into the candidate's answer text, as for
/// grounding.
fn recitation_citations(
    recitation: &Value,
    blocks: &[AnswerBlock],
    out: &mut Vec<StreamEvent>,
) -> usize {
    let empty = Vec::new();
    let sources = recitation
        .get("citationSources")
        .and_then(Value::as_array)
        .unwrap_or(&empty);
    let offset = |source: &Value, key: &str| {
        source
            .get(key)
            .and_then(Value::as_u64)
            .and_then(|i| usize::try_from(i).ok())
    };
    let mut unplaced = 0;
    for source in sources {
        let start = offset(source, "startIndex").unwrap_or(0);
        let (Some(uri), Some(block)) = (
            source.get("uri").and_then(Value::as_str),
            blocks
                .iter()
                .find(|b| start >= b.start && start < b.start + b.text.len()),
        ) else {
            unplaced += 1;
            continue;
        };
        let title = source
            .get("title")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let mut citation = Citation::new(CitationSource::url(uri, title));
        if let Some(end) = offset(source, "endIndex").filter(|end| *end > start) {
            let end = end.min(block.start + block.text.len());
            citation = citation.with_span(start - block.start..end - block.start);
        }
        let mut raw = source.as_object().cloned().unwrap_or_default();
        for portable in ["uri", "title", "startIndex", "endIndex"] {
            raw.remove(portable);
        }
        let mut provider_data = ProviderData::new();
        for (key, value) in raw {
            // `license`, `publicationDate`.
            provider_data.insert_raw(NAMESPACE, snake_case(&key), value);
        }
        out.push(StreamEvent::citation(
            block.index,
            citation.with_provider_data(provider_data),
        ));
    }
    unplaced
}

/// `camelCase` to `snake_case`, for provider-data keys.
fn snake_case(key: &str) -> String {
    let mut out = String::with_capacity(key.len() + 4);
    for c in key.chars() {
        if c.is_ascii_uppercase() {
            out.push('_');
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// The source of one `groundingChunks` entry: a web page, a retrieved
/// document, or a place.
fn grounding_source(chunk: &Value) -> Option<CitationSource> {
    let (kind, body) = ["web", "retrievedContext", "maps"]
        .into_iter()
        .find_map(|kind| chunk.get(kind).map(|body| (kind, body)))?;
    let title = body.get("title").and_then(Value::as_str).map(str::to_owned);
    match body.get("uri").and_then(Value::as_str) {
        Some(uri) => Some(CitationSource::url(uri, title)),
        None if kind == "retrievedContext" => Some(CitationSource::document(None, title)),
        None => None,
    }
}

/// Synthetic id for a `functionCall` without an `id` (Gemini 2.5):
/// `gemini_<scope>_call_<n>`. Only `[A-Za-z0-9_-]`, so every provider takes
/// it after a failover (Anthropic refuses anything else).
pub(crate) fn synthetic_tool_call_id(scope: &str, index: usize) -> String {
    format!("gemini_{scope}_call_{index}")
}

/// Parse a synthetic id back into `(scope, index)`. The index is the digits
/// after the last `_call_`, so a scope containing `_call_` still parses.
pub(crate) fn parse_synthetic_tool_call_id(id: &str) -> Option<(&str, usize)> {
    let rest = id.strip_prefix("gemini_")?;
    let (scope, index) = rest.rsplit_once("_call_")?;
    if scope.is_empty() || index.is_empty() || !index.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((scope, index.parse().ok()?))
}

fn delta(index: usize, kind: TextKind, text: String) -> StreamEvent {
    match kind {
        TextKind::Text => StreamEvent::text_delta(index, text),
        TextKind::Thought => StreamEvent::reasoning_delta(index, text),
    }
}

fn signature_data(signature: Option<String>) -> ProviderData {
    let mut data = ProviderData::new();
    if let Some(signature) = signature {
        data.insert_raw(NAMESPACE, THOUGHT_SIGNATURE_KEY, Value::String(signature));
    }
    data
}

/// Map a Gemini `finishReason`, returning the limitation and the outcome of
/// a warning for endings that callers should notice (malformed or
/// unexpected tool calls).
pub(crate) fn map_finish_reason(
    raw: &str,
    has_tool_calls: bool,
) -> (FinishReason, Option<(&'static str, &'static str)>) {
    match raw {
        "STOP" if has_tool_calls => (FinishReason::ToolCalls, None),
        "STOP" => (FinishReason::Stop, None),
        "MAX_TOKENS" => (FinishReason::Length, None),
        "SAFETY"
        | "RECITATION"
        | "BLOCKLIST"
        | "PROHIBITED_CONTENT"
        | "SPII"
        | "IMAGE_SAFETY"
        | "IMAGE_PROHIBITED_CONTENT"
        | "IMAGE_RECITATION" => (FinishReason::ContentFilter, None),
        "MALFORMED_FUNCTION_CALL" => (
            FinishReason::Other("malformed_function_call".into()),
            Some((
                "Gemini generated a malformed function call",
                "no tool call was returned",
            )),
        ),
        "UNEXPECTED_TOOL_CALL" => (
            FinishReason::Other("unexpected_tool_call".into()),
            Some((
                "Gemini generated a tool call although no tools were enabled",
                "the response ends with finish reason `unexpected_tool_call`",
            )),
        ),
        other => (FinishReason::Other(other.to_ascii_lowercase()), None),
    }
}

/// Normalize Gemini usage to the portable [`Usage`] contract.
///
/// - input = `promptTokenCount` (includes cached content) + `toolUsePromptTokenCount`,
///   with the tool-use part also reported as the `tool_use_input_tokens`
///   breakdown
/// - cache reads = `cachedContentTokenCount`
/// - output = `candidatesTokenCount` + `thoughtsTokenCount` (Gemini excludes
///   thoughts from the candidate count)
/// - reasoning = `thoughtsTokenCount`
/// - total = `totalTokenCount`
/// - `promptTokensDetails` and `toolUsePromptTokensDetails` become the
///   `{modality}_input_tokens` breakdowns, `candidatesTokensDetails` the
///   `{modality}_output_tokens` ones; `cacheTokensDetails` and the
///   `serviceTier` stay in the response provider data
pub(crate) fn usage_from_metadata(meta: &UsageMetadata) -> Usage {
    let tool_use = meta.tool_use_prompt_token_count;
    let input = match (meta.prompt_token_count, tool_use) {
        (None, None) => None,
        (prompt, tool) => Some(prompt.unwrap_or(0).saturating_add(tool.unwrap_or(0))),
    };
    let output = match (meta.candidates_token_count, meta.thoughts_token_count) {
        (None, None) => None,
        (candidates, thoughts) => Some(
            candidates
                .unwrap_or(0)
                .saturating_add(thoughts.unwrap_or(0)),
        ),
    };
    let mut usage = Usage::new();
    usage.input_tokens = input;
    usage.output_tokens = output;
    usage.cache_read_input_tokens = meta.cached_content_token_count;
    usage.reasoning_tokens = meta.thoughts_token_count;
    usage.total_tokens = match (meta.total_token_count, input, output) {
        (Some(total), Some(i), Some(o)) => Some(total.max(i.saturating_add(o))),
        (total, _, _) => total,
    };
    if let Some(tool_use) = tool_use {
        usage
            .breakdowns
            .insert("tool_use_input_tokens".into(), tool_use);
    }
    // Modality breakdowns: the tool-use input counts in the input too.
    for (details, direction) in [
        (&meta.prompt_tokens_details, "input"),
        (&meta.tool_use_prompt_tokens_details, "input"),
        (&meta.candidates_tokens_details, "output"),
    ] {
        for detail in details {
            let (Some(modality), Some(count)) = (detail.modality.as_deref(), detail.token_count)
            else {
                continue;
            };
            let Some(modality) = modality_key(modality) else {
                continue;
            };
            let slot = usage
                .breakdowns
                .entry(format!("{modality}_{direction}_tokens"))
                .or_insert(0);
            *slot = slot.saturating_add(count);
        }
    }
    usage.with_derived_total()
}

/// The breakdown key prefix of a Gemini `Modality`; `None` for one the
/// usage contract has no key for (`MODALITY_UNSPECIFIED`).
fn modality_key(modality: &str) -> Option<&'static str> {
    Some(match modality {
        "TEXT" => "text",
        "IMAGE" => "image",
        "AUDIO" => "audio",
        "VIDEO" => "video",
        "DOCUMENT" => "document",
        _ => return None,
    })
}

/// The [`ErrorContext::metadata`](allama::ErrorContext::metadata) key of
/// usage a failed call reported.
const USAGE_KEY: &str = "usage";

/// The usage Gemini reported with a blocked prompt, when it reported any.
///
/// A prompt Gemini blocks arrives as a `200` with `promptFeedback` and no
/// candidate, which the adapter returns as [`Error::ContentFiltered`] (or
/// [`Error::InvalidRequest`] for a reason that is not a safety filter). The
/// response's `usageMetadata`, normalized as on a response, is kept in the
/// error's metadata under the `google-ai` namespace; read it here. Google
/// does not document whether blocked prompts are billed.
///
/// ```
/// use allama::Error;
///
/// let error = Error::content_filtered("prompt was blocked by Gemini: SAFETY");
/// assert!(allama_gemini::blocked_prompt_usage(&error).is_none());
/// ```
#[must_use]
pub fn blocked_prompt_usage(error: &Error) -> Option<Usage> {
    let usage = error
        .context()?
        .metadata
        .get(NAMESPACE)?
        .get(USAGE_KEY)?
        .clone();
    serde_json::from_value(usage).ok()
}

/// Attach `usage` to `error`, for [`blocked_prompt_usage`].
fn with_usage(mut error: Error, usage: &Usage) -> Error {
    if let (Some(context), Ok(usage)) = (error.context_mut(), serde_json::to_value(usage)) {
        let entry = context
            .metadata
            .entry(NAMESPACE.to_owned())
            .or_insert_with(|| Value::Object(serde_json::Map::new()));
        if let Value::Object(entry) = entry {
            entry.insert(USAGE_KEY.to_owned(), usage);
        }
    }
    error
}

fn prompt_feedback_error(feedback: &PromptFeedback) -> Option<Error> {
    let reason = feedback.block_reason.as_deref()?;
    let message = format!("prompt was blocked by Gemini: {reason}");
    let error = match reason {
        "SAFETY" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "IMAGE_SAFETY" => {
            Error::content_filtered(message)
        }
        _ => Error::invalid_request(message),
    }
    .authored()
    .with_upstream_message_opt(feedback.block_reason_message.as_deref());
    Some(
        error
            .with_provider_code(reason.to_owned())
            .with_provider(PROVIDER_NAME),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use allama::{ContentBlock, StreamEvent};
    use serde_json::json;

    fn chunk(value: Value) -> GenerateContentResponse {
        serde_json::from_value(value).unwrap()
    }

    fn decode_chunks(values: Vec<Value>, end: End) -> Result<Vec<StreamEvent>> {
        let mut decoder = Decoder::new(Vec::new());
        let mut out = Vec::new();
        for value in values {
            out.extend(decoder.push(chunk(value))?);
        }
        out.extend(decoder.finish(end)?);
        Ok(out)
    }

    fn collect(events: Vec<StreamEvent>) -> ChatResponse {
        let mut collector = StreamCollector::new();
        for event in events {
            collector.push(event).unwrap();
        }
        collector.finish().unwrap()
    }

    fn text_chunk(text: &str) -> Value {
        json!({"responseId": "r1", "modelVersion": "gemini-3.5-flash-lite",
               "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]})
    }

    #[test]
    fn deltas_append_to_one_text_block() {
        let events = decode_chunks(
            vec![
                text_chunk("one"),
                text_chunk(", two"),
                json!({"candidates": [{"content": {"parts": [{"text": ", three"}]}, "finishReason": "STOP"}]}),
            ],
            End::Stream,
        )
        .unwrap();
        let deltas: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::TextDelta { index, text, .. } => Some((*index, text.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(deltas, vec![(0, "one"), (0, ", two"), (0, ", three")]);
        let response = collect(events);
        assert_eq!(response.text().as_deref(), Some("one, two, three"));
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(response.finish_reason_raw.as_deref(), Some("STOP"));
        assert_eq!(response.provider.as_deref(), Some("gemini"));
        assert_eq!(response.id.as_deref(), Some("r1"));
    }

    #[test]
    fn kind_change_starts_new_block() {
        let response = collect(
            decode_chunks(
                vec![json!({"candidates": [{"content": {"parts": [
                    {"text": "thinking", "thought": true},
                    {"text": " more", "thought": true},
                    {"text": "answer"}
                ]}, "finishReason": "STOP"}]})],
                End::Complete,
            )
            .unwrap(),
        );
        assert_eq!(response.content.len(), 2);
        let reasoning = response.content[0].as_reasoning().unwrap();
        assert_eq!(reasoning.text, "thinking more");
        assert_eq!(
            reasoning.provider_data.get_raw(NAMESPACE, THOUGHT_KEY),
            Some(&json!(true))
        );
        assert_eq!(response.content[1].as_text(), Some("answer"));
    }

    #[test]
    fn empty_signature_part_attaches_to_open_text_block() {
        let response = collect(
            decode_chunks(
                vec![
                    text_chunk("3"),
                    text_chunk("91"),
                    json!({"candidates": [{"content": {"parts": [{"text": "", "thoughtSignature": "sig"}]}, "finishReason": "STOP"}]}),
                ],
                End::Stream,
            )
            .unwrap(),
        );
        assert_eq!(response.content.len(), 1);
        let ContentBlock::Text(text) = &response.content[0] else {
            panic!("expected text");
        };
        assert_eq!(text.text, "391");
        assert_eq!(
            text.provider_data.get_raw(NAMESPACE, THOUGHT_SIGNATURE_KEY),
            Some(&json!("sig"))
        );
    }

    #[test]
    fn empty_signature_part_without_open_block_is_kept() {
        let response = collect(
            decode_chunks(
                vec![json!({"candidates": [{"content": {"parts": [
                    {"functionCall": {"name": "f", "args": {}}},
                    {"text": "", "thoughtSignature": "sig"}
                ]}, "finishReason": "STOP"}]})],
                End::Complete,
            )
            .unwrap(),
        );
        assert_eq!(response.content.len(), 2);
        let ContentBlock::Text(text) = &response.content[1] else {
            panic!("expected text");
        };
        assert!(text.text.is_empty());
        assert_eq!(
            text.provider_data.get_raw(NAMESPACE, THOUGHT_SIGNATURE_KEY),
            Some(&json!("sig"))
        );
    }

    #[test]
    fn signature_terminates_its_part() {
        let response = collect(
            decode_chunks(
                vec![
                    json!({"candidates": [{"content": {"parts": [{"text": "a", "thoughtSignature": "s1"}]}}]}),
                    json!({"candidates": [{"content": {"parts": [{"text": "b"}]}, "finishReason": "STOP"}]}),
                ],
                End::Stream,
            )
            .unwrap(),
        );
        assert_eq!(response.content.len(), 2);
        assert_eq!(response.content[0].as_text(), Some("a"));
        assert_eq!(response.content[1].as_text(), Some("b"));
        assert!(response.content[1].provider_data().is_empty());
    }

    #[test]
    fn function_calls_are_complete_blocks_with_ids_and_signatures() {
        let response = collect(
            decode_chunks(
                vec![json!({"responseId": "r", "candidates": [{"content": {"parts": [
                    {"text": "Checking."},
                    {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}, "id": "call_1"}, "thoughtSignature": "sig"},
                    {"functionCall": {"name": "get_weather", "args": {"city": "Tokyo"}}}
                ]}, "finishReason": "STOP"}]})],
                End::Complete,
            )
            .unwrap(),
        );
        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(response.finish_reason_raw.as_deref(), Some("STOP"));
        let calls: Vec<_> = response.tool_calls().collect();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].arguments, r#"{"city":"Paris"}"#);
        assert_eq!(
            calls[0]
                .provider_data
                .get_raw(NAMESPACE, THOUGHT_SIGNATURE_KEY),
            Some(&json!("sig"))
        );
        assert_eq!(calls[1].id, "gemini_r_call_1");
        assert!(calls[1].provider_data.is_empty());
    }

    #[test]
    fn missing_args_become_empty_object() {
        let response = collect(
            decode_chunks(
                vec![json!({"candidates": [{"content": {"parts": [
                    {"functionCall": {"name": "ping"}}
                ]}, "finishReason": "STOP"}]})],
                End::Complete,
            )
            .unwrap(),
        );
        assert_eq!(response.first_tool_call().unwrap().arguments, "{}");
    }

    #[test]
    fn audio_inline_data_is_other_content() {
        let response = collect(
            decode_chunks(
                vec![json!({"candidates": [{"content": {"parts": [
                    {"inlineData": {"mimeType": "audio/L16;rate=24000", "data": "AAAA"}},
                    {"inlineData": {"mimeType": "image/png", "data": "iVBO"}}
                ]}, "finishReason": "STOP"}]})],
                End::Complete,
            )
            .unwrap(),
        );
        let ContentBlock::Other(other) = &response.content[0] else {
            panic!("expected other, got {:?}", response.content[0]);
        };
        assert_eq!(other.type_name, "inline_data");
        assert_eq!(other.origin.as_deref(), Some(NAMESPACE));
        assert_eq!(other.data["inlineData"]["mimeType"], "audio/L16;rate=24000");
        assert!(other.provider_data.is_empty());
        let image = response.content[1].as_image().unwrap();
        assert_eq!(image.source, MediaSource::base64("image/png", "iVBO"));
    }

    #[test]
    fn other_content_keeps_its_signature_in_provider_data() {
        let response = collect(
            decode_chunks(
                vec![json!({"candidates": [{"content": {"parts": [
                    {"inlineData": {"mimeType": "audio/wav", "data": "AAAA"}, "thoughtSignature": "sig"}
                ]}, "finishReason": "STOP"}]})],
                End::Complete,
            )
            .unwrap(),
        );
        let ContentBlock::Other(other) = &response.content[0] else {
            panic!("expected other, got {:?}", response.content[0]);
        };
        assert_eq!(
            Value::Object(other.data.clone()),
            json!({"inlineData": {"mimeType": "audio/wav", "data": "AAAA"}}),
            "the signature is not wire data of the part"
        );
        assert_eq!(
            other
                .provider_data
                .get_raw(NAMESPACE, THOUGHT_SIGNATURE_KEY),
            Some(&json!("sig"))
        );
    }

    #[test]
    fn code_execution_parts_become_provider_tool_blocks() {
        let response = collect(
            decode_chunks(
                vec![json!({"responseId": "r1", "candidates": [{"content": {"parts": [
                    {"executableCode": {"language": "PYTHON", "code": "print(1)"}, "thoughtSignature": "sig"},
                    {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "1"}},
                    {"executableCode": {"language": "PYTHON", "code": "1/0"}},
                    {"codeExecutionResult": {"outcome": "OUTCOME_FAILED", "output": "ZeroDivisionError"}},
                    {"futurePart": {"x": 1}}
                ]}, "finishReason": "STOP"}]})],
                End::Complete,
            )
            .unwrap(),
        );
        let ContentBlock::ProviderToolCall(call) = &response.content[0] else {
            panic!("{:?}", response.content)
        };
        assert_eq!(call.name, "code_execution");
        assert_eq!(call.id, "gemini_r1_code_0");
        assert_eq!(call.origin.as_deref(), Some(NAMESPACE));
        assert_eq!(
            serde_json::from_str::<Value>(&call.arguments).unwrap(),
            json!({"language": "PYTHON", "code": "print(1)"})
        );
        assert_eq!(
            call.provider_data.get_raw(NAMESPACE, THOUGHT_SIGNATURE_KEY),
            Some(&json!("sig"))
        );
        let ContentBlock::ProviderToolResult(ok) = &response.content[1] else {
            panic!("{:?}", response.content)
        };
        assert_eq!(
            (ok.call_id.as_str(), ok.is_error),
            ("gemini_r1_code_0", false)
        );
        assert_eq!(ok.output["output"], "1");
        let ContentBlock::ProviderToolResult(failed) = &response.content[3] else {
            panic!("{:?}", response.content)
        };
        assert_eq!(failed.call_id, "gemini_r1_code_1");
        assert!(failed.is_error);
        assert!(!response.has_tool_calls(), "nothing for the caller to run");
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        assert!(
            matches!(&response.content[4], ContentBlock::Other(o) if o.type_name == "futurePart"),
            "unknown parts stay verbatim"
        );
    }

    #[test]
    fn stream_without_finish_reason_is_incomplete() {
        let err = decode_chunks(vec![text_chunk("partial")], End::Stream).unwrap_err();
        assert!(matches!(
            err,
            Error::Stream {
                kind: StreamErrorKind::Incomplete,
                ..
            }
        ));
        assert!(err.is_retryable());
    }

    #[test]
    fn unary_without_candidates_is_complete() {
        let response = decode_response(
            chunk(json!({"responseId": "r", "candidates": [], "usageMetadata": {"promptTokenCount": 4, "totalTokenCount": 4}})),
            Vec::new(),
            None,
        )
        .unwrap();
        assert!(response.content.is_empty());
        assert_eq!(response.finish_reason, None);
        assert_eq!(response.usage.unwrap().input_tokens, Some(4));
    }

    #[test]
    fn empty_content_with_finish_reason_is_accepted() {
        let response = collect(
            decode_chunks(
                vec![
                    text_chunk("one"),
                    json!({"candidates": [{"content": {}, "finishReason": "RECITATION", "finishMessage": "filtered"}]}),
                ],
                End::Stream,
            )
            .unwrap(),
        );
        assert_eq!(response.finish_reason, Some(FinishReason::ContentFilter));
        assert_eq!(response.finish_reason_raw.as_deref(), Some("RECITATION"));
        assert_eq!(
            response.provider_data.get_raw(NAMESPACE, "finish_message"),
            Some(&json!("filtered"))
        );
    }

    #[test]
    fn malformed_function_call_is_other_with_warning() {
        let response = collect(
            decode_chunks(
                vec![json!({"candidates": [{"content": {}, "finishReason": "MALFORMED_FUNCTION_CALL", "finishMessage": "bad call"}]})],
                End::Complete,
            )
            .unwrap(),
        );
        assert_eq!(
            response.finish_reason,
            Some(FinishReason::Other("malformed_function_call".into()))
        );
        assert_eq!(
            response.finish_reason_raw.as_deref(),
            Some("MALFORMED_FUNCTION_CALL")
        );
        assert!(!response.has_tool_calls());
        assert_eq!(response.warnings.len(), 1);
        assert!(response.warnings[0].message.contains("bad call"));
    }

    #[test]
    fn finish_reasons_map() {
        for (raw, expected) in [
            ("STOP", FinishReason::Stop),
            ("MAX_TOKENS", FinishReason::Length),
            ("SAFETY", FinishReason::ContentFilter),
            ("RECITATION", FinishReason::ContentFilter),
            ("BLOCKLIST", FinishReason::ContentFilter),
            ("PROHIBITED_CONTENT", FinishReason::ContentFilter),
            ("SPII", FinishReason::ContentFilter),
            ("IMAGE_SAFETY", FinishReason::ContentFilter),
            ("LANGUAGE", FinishReason::Other("language".into())),
            ("OTHER", FinishReason::Other("other".into())),
        ] {
            assert_eq!(map_finish_reason(raw, false).0, expected, "{raw}");
        }
        assert_eq!(
            map_finish_reason("MAX_TOKENS", true).0,
            FinishReason::Length
        );
    }

    #[test]
    fn grounding_reaches_answer_blocks_closed_earlier() {
        let events = decode_chunks(
            vec![
                json!({"candidates": [{"content": {"parts": [{"text": "Rust 1.99 is out."}]}}]}),
                json!({"candidates": [{"content": {"parts": [{"functionCall": {"name": "lookup", "args": {}}}]}}]}),
                json!({"candidates": [{"content": {"parts": [{"text": " Upgrade now."}]},
                    "finishReason": "STOP",
                    "groundingMetadata": {
                        "groundingChunks": [
                            {"web": {"uri": "https://blog.rust-lang.org", "title": "Rust Blog"}},
                            {"retrievedContext": {"title": "Internal notes"}}
                        ],
                        "groundingSupports": [
                            {"segment": {"endIndex": 9, "text": "Rust 1.99"}, "groundingChunkIndices": [0, 1], "confidenceScores": [0.9, 0.4]},
                            {"segment": {"startIndex": 18, "endIndex": 30, "text": "Upgrade now."}, "groundingChunkIndices": [0]},
{"segment": {"startIndex": 5, "endIndex": 2, "text": "1.99"}, "groundingChunkIndices": [0]},
                            {"segment": {"startIndex": 500, "endIndex": 510, "text": "elsewhere"}, "groundingChunkIndices": [0]}
                        ]
                    }}]}),
            ],
            End::Stream,
        )
        .unwrap();
        // The first answer block stays open (no BlockStop) until the grounding
        // has been attached, even though the function call came after it.
        let first_stop = events
            .iter()
            .position(|e| matches!(e, StreamEvent::BlockStop { index: 0, .. }))
            .unwrap();
        let last_citation = events
            .iter()
            .rposition(|e| matches!(e, StreamEvent::Citation { index: 0, .. }))
            .unwrap();
        assert!(last_citation < first_stop);

        let response = collect(events);
        let ContentBlock::Text(first) = &response.content[0] else {
            panic!("{:?}", response.content)
        };
        assert_eq!(first.citations.len(), 3);
        assert_eq!(first.cited_span(&first.citations[0]), Some("Rust 1.99"));
        assert_eq!(
            first.citations[2].span, None,
            "a support whose end precedes its start keeps no span"
        );
        assert_eq!(
            first.citations[1].source,
            CitationSource::document(None, Some("Internal notes".into()))
        );
        assert_eq!(
            first.citations[0]
                .provider_data
                .get_raw(NAMESPACE, "confidence_score"),
            Some(&json!(0.9))
        );
        let ContentBlock::Text(second) = &response.content[2] else {
            panic!("{:?}", response.content)
        };
        assert_eq!(
            second.cited_span(&second.citations[0]),
            Some("Upgrade now.")
        );
        assert_eq!(response.warnings.len(), 1, "one support did not match");
        assert!(
            response
                .provider_data
                .get_raw(NAMESPACE, GROUNDING_KEY)
                .is_some()
        );
    }

    #[test]
    fn in_stream_error_is_typed() {
        let mut decoder = Decoder::new(Vec::new());
        let err = decoder
            .push(chunk(json!({"error": {"code": 429, "message": "slow down", "status": "RESOURCE_EXHAUSTED"}})))
            .unwrap_err();
        assert!(matches!(err, Error::RateLimited { .. }), "{err:?}");
    }

    #[test]
    fn blocked_prompt_is_content_filtered() {
        let err = decode_response(
            chunk(json!({"promptFeedback": {"blockReason": "SAFETY", "blockReasonMessage": "blocked"}})),
            Vec::new(),
            None,
        )
        .unwrap_err();
        assert!(matches!(err, Error::ContentFiltered { .. }));
        assert_eq!(err.message(), "prompt was blocked by Gemini: SAFETY");
        assert_eq!(err.upstream_message(), Some("blocked"));
        assert_eq!(
            err.as_log().to_string(),
            r#"kind=content_filtered provider=gemini provider_code=SAFETY message="prompt was blocked by Gemini: SAFETY""#
        );
        assert!(blocked_prompt_usage(&err).is_none());
    }

    /// A blocked prompt is a `200` with no candidate, so it stays an error;
    /// it carries the request id, and the usage Gemini reported.
    /// The shape is Gemini's documented `promptFeedback` with the
    /// `usageMetadata` every response carries
    /// (<https://ai.google.dev/api/generate-content#v1beta.GenerateContentResponse>).
    #[test]
    fn a_blocked_prompt_carries_the_request_id_and_usage() {
        let blocked = || {
            chunk(json!({
                "promptFeedback": {"blockReason": "PROHIBITED_CONTENT"},
                "usageMetadata": {"promptTokenCount": 7, "totalTokenCount": 7},
                "modelVersion": "gemini-2.5-flash"
            }))
        };
        let err = decode_response(blocked(), Vec::new(), Some("req_9".into())).unwrap_err();
        assert!(matches!(err, Error::ContentFiltered { .. }), "{err:?}");
        assert_eq!(err.request_id(), Some("req_9"));
        let usage = blocked_prompt_usage(&err).unwrap();
        assert_eq!(usage.input_tokens, Some(7));
        assert_eq!(usage.total_tokens, Some(7));

        // The streaming path decodes the same chunk.
        let mut decoder = Decoder::new(Vec::new()).with_request_id(Some("req_9".into()));
        let err = decoder.push(blocked()).unwrap_err();
        assert_eq!(err.request_id(), Some("req_9"));
        assert_eq!(blocked_prompt_usage(&err).unwrap().input_tokens, Some(7));
    }

    #[test]
    fn request_warnings_follow_response_start() {
        let warning = Warning::new(WarningKind::Unsupported, Subject::Response, "w");
        let mut decoder = Decoder::new(vec![warning.clone()]);
        let events = decoder.push(chunk(text_chunk("x"))).unwrap();
        assert!(matches!(events[0], StreamEvent::ResponseStart { .. }));
        assert_eq!(events[1], StreamEvent::warning(warning));
    }

    #[test]
    fn usage_is_normalized() {
        let usage = usage_from_metadata(&UsageMetadata {
            prompt_token_count: Some(56),
            candidates_token_count: Some(32),
            total_token_count: Some(172),
            cached_content_token_count: Some(16),
            thoughts_token_count: Some(74),
            tool_use_prompt_token_count: Some(10),
            ..UsageMetadata::default()
        });
        assert_eq!(usage.input_tokens, Some(66));
        assert_eq!(usage.output_tokens, Some(106));
        assert_eq!(usage.reasoning_tokens, Some(74));
        assert_eq!(usage.cache_read_input_tokens, Some(16));
        assert_eq!(usage.total_tokens, Some(172));
        assert_eq!(usage.breakdowns.get("tool_use_input_tokens"), Some(&10));
        assert_eq!(usage.contract_violation(), None);
    }

    /// Modality token details are input and output breakdowns.
    #[test]
    fn modality_details_are_usage_breakdowns() {
        let meta: UsageMetadata = serde_json::from_value(json!({
            "promptTokenCount": 261,
            "candidatesTokenCount": 4,
            "toolUsePromptTokenCount": 10,
            "promptTokensDetails": [
                {"modality": "TEXT", "tokenCount": 5},
                {"modality": "IMAGE", "tokenCount": 256}
            ],
            "toolUsePromptTokensDetails": [{"modality": "TEXT", "tokenCount": 10}],
            "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 4}],
            "cacheTokensDetails": [{"modality": "TEXT", "tokenCount": 0}],
            "serviceTier": "standard"
        }))
        .unwrap();
        let usage = usage_from_metadata(&meta);
        assert_eq!(usage.breakdowns.get("text_input_tokens"), Some(&15));
        assert_eq!(usage.breakdowns.get("image_input_tokens"), Some(&256));
        assert_eq!(usage.breakdowns.get("text_output_tokens"), Some(&4));
        assert_eq!(usage.contract_violation(), None);
    }

    /// Recitation sources become citations; fields the adapter does
    /// not model are kept, not dropped.
    #[test]
    fn recitation_citations_and_unmodeled_fields_are_kept() {
        let response = collect(
            decode_chunks(
                vec![json!({
                    "responseId": "r1",
                    "candidates": [{
                        "index": 0,
                        "content": {"role": "model", "parts": [{"text": "To be or not to be."}]},
                        "finishReason": "STOP",
                        "citationMetadata": {"citationSources": [
                            {"startIndex": 0, "endIndex": 18, "uri": "https://example.com/hamlet", "license": "pd"},
                            {"startIndex": 0, "endIndex": 5}
                        ]},
                        "safetyRatings": [{"category": "HARM_CATEGORY_HARASSMENT", "probability": "NEGLIGIBLE"}],
                        "avgLogprobs": -0.25
                    }],
                    "promptFeedback": {"safetyRatings": []},
                    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 6, "serviceTier": "flex", "trafficType": "ON_DEMAND"}
                })],
                End::Complete,
            )
            .unwrap(),
        );
        let ContentBlock::Text(text) = &response.content[0] else {
            panic!("{:?}", response.content);
        };
        assert_eq!(text.citations.len(), 1);
        assert_eq!(
            text.citations[0].source,
            CitationSource::url("https://example.com/hamlet", None)
        );
        assert_eq!(text.citations[0].span, Some(0..18));
        assert_eq!(
            text.citations[0]
                .provider_data
                .get_raw(NAMESPACE, "license"),
            Some(&json!("pd"))
        );
        let data = response.provider_data.namespace(NAMESPACE).unwrap();
        assert!(data.contains_key(CITATION_METADATA_KEY));
        assert_eq!(data["safety_ratings"][0]["probability"], "NEGLIGIBLE");
        assert_eq!(data["avg_logprobs"], json!(-0.25));
        assert_eq!(data["prompt_feedback"], json!({"safetyRatings": []}));
        assert_eq!(data["service_tier"], "flex");
        assert_eq!(data["usage"], json!({"trafficType": "ON_DEMAND"}));
        assert!(!data.contains_key("index"));
        // The source without a URL is reported.
        assert!(
            response
                .warnings
                .iter()
                .any(|w| w.message.contains("recitation source"))
        );
    }

    /// A `fileData` image keeps its MIME type, and a
    /// draft image made while thinking keeps its `thought` flag.
    #[test]
    fn images_keep_their_mime_type_and_thought_flag() {
        let response = collect(
            decode_chunks(
                vec![json!({
                    "candidates": [{
                        "content": {"role": "model", "parts": [
                            {"inlineData": {"mimeType": "image/png", "data": "AAAA"}, "thought": true},
                            {"fileData": {"mimeType": "image/jpeg", "fileUri": "https://example.com/a"}}
                        ]},
                        "finishReason": "STOP"
                    }]
                })],
                End::Complete,
            )
            .unwrap(),
        );
        let images: Vec<&Image> = response
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Image(image) => Some(image),
                _ => None,
            })
            .collect();
        assert_eq!(
            images[0].provider_data.get_raw(NAMESPACE, THOUGHT_KEY),
            Some(&json!(true))
        );
        assert_eq!(images[1].media_type.as_deref(), Some("image/jpeg"));
        assert!(
            images[1]
                .provider_data
                .get_raw(NAMESPACE, THOUGHT_KEY)
                .is_none()
        );
    }

    #[test]
    fn usage_without_counts_stays_empty() {
        let usage = usage_from_metadata(&UsageMetadata::default());
        assert!(usage.is_empty());
    }

    #[test]
    fn synthetic_ids_round_trip() {
        let id = synthetic_tool_call_id("resp", 3);
        assert_eq!(parse_synthetic_tool_call_id(&id), Some(("resp", 3)));
        assert_eq!(parse_synthetic_tool_call_id("call_1"), None);
        // The index is the digits after the last `_call_`.
        let id = synthetic_tool_call_id("a_call_b", 12);
        assert_eq!(parse_synthetic_tool_call_id(&id), Some(("a_call_b", 12)));
        assert_eq!(parse_synthetic_tool_call_id("gemini_r_call_x"), None);
        assert_eq!(parse_synthetic_tool_call_id("gemini_r_call_+1"), None);
        assert_eq!(parse_synthetic_tool_call_id("gemini__call_1"), None);
    }

    /// Synthetic ids use only `[A-Za-z0-9_-]`, which every provider takes
    /// (Anthropic refuses anything else), whatever the response id holds.
    #[test]
    fn synthetic_ids_are_url_safe() {
        let calls = |chunk: Value| {
            let response = collect(decode_chunks(vec![chunk], End::Complete).unwrap());
            response
                .tool_calls()
                .map(|call| call.id.clone())
                .collect::<Vec<_>>()
        };
        let part = json!({"functionCall": {"name": "f", "args": {}}});
        let ids = calls(
            json!({"responseId": "r:1/x", "candidates": [{"content": {"parts": [part.clone()]}, "finishReason": "STOP"}]}),
        );
        assert_eq!(ids, ["gemini_r_1_x_call_0"]);
        let ids = calls(
            json!({"candidates": [{"content": {"parts": [part.clone(), part]}, "finishReason": "STOP"}]}),
        );
        for id in &ids {
            assert!(id.starts_with("gemini_anonymous_"), "{id}");
            assert!(
                id.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
                "{id}"
            );
            assert!(id.len() <= 64, "{id}");
        }
        assert_eq!(
            parse_synthetic_tool_call_id(&ids[1]).map(|(_, n)| n),
            Some(1)
        );
        assert_ne!(ids[0], ids[1]);
    }
}
