//! Maps a portable [`ChatRequest`] to an Anthropic Messages API request.
//!
//! Portable fields Anthropic cannot honor are handled through
//! [`Degradations`]: hints (seed, penalties, image detail, cache key, …) are
//! dropped with a warning (or rejected under `UnsupportedPolicy::Error`),
//! semantic fields fail with `Error::Unsupported`, and replay state from
//! other providers is dropped with a `Dropped` warning.

use std::collections::{HashMap, HashSet};

use allama::adapter::chat::{EncodedTurn, ToolCallIdFormat, retain_turns};
use allama::adapter::schema::{Degradable, StrictOutcome, StrictSchemaError, apply_strict_policy};
use allama::adapter::{Degradations, describe_origins};
use allama::{
    CacheBreakpoint, CacheRetention, CapabilitySupport, ChatRequest, ContentBlock, ContentPart,
    Error, ExtraMap, File, Image, MediaSource, Message, OtherContent, PromptCache, ProviderData,
    ProviderToolCall, ProviderToolResult, Reasoning, ReasoningEffort, ReasoningOutput, ReplayState,
    RequestField, ResponseFormat, Result, Subject, ToolCall, ToolResultContent, UserContent,
    Warning,
};
use serde_json::Value;

use crate::ChatRequestOptions;
use crate::model::{self, DisableThinking, ModelProfile, ThinkingStyle};
use crate::schema::Complexity;
use crate::wire::{
    BlockParam, CacheControl, CreateMessageRequest, DocumentSource, KnownBlock, MediaSourceParam,
    MessageParam, OutputConfig, OutputFormat, RESERVED_FIELDS, RequestBlock, SystemBlock,
    ThinkingConfig, ToolChoice, ToolParam, ToolResultContentParam,
};
use crate::{NAMESPACE, PROVIDER_NAME};

/// Default `max_tokens` when the request leaves `max_output_tokens` unset.
///
/// Anthropic requires `max_tokens`. On adaptive-thinking models it caps
/// thinking and visible output together.
pub(crate) const DEFAULT_MAX_TOKENS: u32 = 4096;

/// The tool call ids Anthropic accepts: `tool_use.id` must match
/// `^[a-zA-Z0-9_-]+$` (a 400 `invalid_request_error` otherwise, verified
/// live on 2026-10-07; no length limit was hit at 300 characters). Ids from
/// other providers that do not (Gemini's made-up ids before 0.1, a custom
/// gateway's) are mapped, the same way for a call and its result.
const TOOL_CALL_IDS: ToolCallIdFormat = ToolCallIdFormat::UrlSafe;

/// Maximum number of `cache_control` breakpoints Anthropic accepts.
const MAX_CACHE_BREAKPOINTS: usize = 4;

/// Minimum Anthropic thinking budget.
///
/// Source: <https://platform.claude.com/docs/en/build-with-claude/extended-thinking#budget-rules-and-tuning>.
const MIN_THINKING_BUDGET: u32 = 1024;

/// The beta under which manual thinking interleaves with tool calls, and
/// `budget_tokens` may exceed `max_tokens`.
const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";

/// An encoded request plus what the adapter learned while encoding it.
#[derive(Debug)]
pub(crate) struct EncodedRequest {
    pub body: CreateMessageRequest,
    pub betas: Vec<String>,
    pub warnings: Vec<Warning>,
}

/// Encode `request` for the Messages API, with the built-in answers for its
/// model.
#[cfg(test)]
pub(crate) fn encode_request(request: &ChatRequest, stream: bool) -> Result<EncodedRequest> {
    encode_request_for(request, stream, ModelProfile::for_model(&request.model))
}

/// Encode `request` for the Messages API, for a model described by
/// `profile` (the built-in profile with the caller's capability overrides).
pub(crate) fn encode_request_for(
    request: &ChatRequest,
    stream: bool,
    profile: ModelProfile,
) -> Result<EncodedRequest> {
    let options = request.resolve_option::<ChatRequestOptions>()?;
    check_extra(&options.extra)?;
    let mut d = Degradations::new(
        PROVIDER_NAME,
        request.unsupported_policy.unwrap_or_default(),
    );

    hints(request, &mut d)?;

    let mut encoder = Encoder {
        request,
        profile,
        d,
        cache_slots: plan_cache_breakpoints(request),
        server_tools: &options.server_tools,
        output_complexity: Complexity::default(),
    };
    encoder.drop_excess_breakpoints();

    let output_format = encoder.output_format()?;
    let system = encoder.system()?;
    let messages = encoder.messages()?;
    let (tools, tool_choice) = encoder.tools()?;
    let has_tools = tools.as_ref().is_some_and(|tools| !tools.is_empty());
    let (thinking, effort, max_tokens) = encoder.thinking(&options, has_tools)?;

    // "Forced tool use [...] is incompatible with manual extended thinking
    // but works with adaptive thinking", for every model with manual
    // thinking (a protocol fact; the models that reject forced tool use
    // always are left to the API, whose refusal is mapped):
    // <https://platform.claude.com/docs/en/build-with-claude/thinking#limits-and-feature-compatibility>.
    if thinking.as_ref().is_some_and(ThinkingConfig::is_manual)
        && tool_choice
            .as_ref()
            .is_some_and(|c| matches!(c.kind, "any" | "tool"))
    {
        return Err(encoder.d.unsupported(
            RequestField::ToolChoice,
            "Anthropic does not allow forced tool choice with manual extended thinking (`budget_tokens`); use adaptive thinking (a reasoning effort) instead",
        ));
    }

    let thinking_on = thinking.as_ref().is_some_and(ThinkingConfig::is_on);
    let (temperature, top_p, top_k) = sampling(request, thinking_on, &mut encoder.d)?;
    let mut body = CreateMessageRequest {
        model: request.model.clone(),
        max_tokens,
        metadata: options.metadata.clone(),
        system: (!system.is_empty()).then_some(system),
        messages,
        tools,
        tool_choice,
        temperature,
        top_p,
        top_k,
        stop_sequences: (!request.stop.is_empty()).then(|| request.stop.clone()),
        thinking,
        output_config: (effort.is_some() || output_format.is_some()).then_some(OutputConfig {
            effort,
            format: output_format,
        }),
        cache_control: None,
        service_tier: options.service_tier.clone(),
        stream,
        extra: options.extra.clone(),
    };

    encoder.finish_cache(&mut body);

    Ok(EncodedRequest {
        body,
        betas: options.anthropic_beta.clone(),
        warnings: encoder.d.into_warnings(),
    })
}

/// The `tool` fields the adapter sets from the portable definition.
const TOOL_FIELDS: &[&str] = &[
    "name",
    "description",
    "input_schema",
    "strict",
    "cache_control",
];

/// The `document` block fields the adapter sets from the portable file.
const DOCUMENT_FIELDS: &[&str] = &["type", "source", "title", "cache_control"];

/// Refuse provider data that sets a field the adapter owns: flattened into
/// the block, it would send the key twice.
fn check_reserved(fields: &ExtraMap, reserved: &[&str], from: &str) -> Result<()> {
    match fields.keys().find(|key| reserved.contains(&key.as_str())) {
        Some(key) => Err(Error::invalid_request(format!(
            "`{key}` cannot be set through the `anthropic` {from}: the adapter sets it from the portable field"
        ))
        .authored()
        .with_provider(PROVIDER_NAME)),
        None => Ok(()),
    }
}

/// A media type without its parameters: `text/plain; charset=utf-8` is
/// `text/plain`.
fn essence(media_type: &str) -> &str {
    media_type.split(';').next().unwrap_or(media_type).trim()
}

/// Refuse an `extra` field the adapter owns: merged into the body, it would
/// send the key twice.
fn check_extra(extra: &ExtraMap) -> Result<()> {
    match extra.keys().find(|key| {
        RESERVED_FIELDS
            .iter()
            .any(|reserved| key.eq_ignore_ascii_case(reserved))
    }) {
        Some(key) => Err(Error::invalid_request(format!(
            "`{key}` cannot be set through ChatRequestOptions::extra: the adapter owns it; use the portable request field or the typed option"
        ))
        .authored()
        .with_provider(PROVIDER_NAME)),
        None => Ok(()),
    }
}

/// Portable hints Anthropic has no equivalent for.
fn hints(request: &ChatRequest, d: &mut Degradations) -> Result<()> {
    if request.seed.is_some() {
        d.unsupported_hint(
            RequestField::Seed,
            "Anthropic has no seed parameter; `seed` was not sent",
        )?;
    }
    if request.frequency_penalty.is_some() {
        d.unsupported_hint(
            RequestField::FrequencyPenalty,
            "Anthropic has no frequency penalty; `frequency_penalty` was not sent",
        )?;
    }
    if request.presence_penalty.is_some() {
        d.unsupported_hint(
            RequestField::PresencePenalty,
            "Anthropic has no presence penalty; `presence_penalty` was not sent",
        )?;
    }
    if request.cache.as_ref().is_some_and(|c| c.key.is_some()) {
        d.unsupported_hint(
            RequestField::Cache,
            "Anthropic has no prompt cache key, since it keys caches by prompt prefix; the key was not sent",
        )?;
    }
    if request
        .cache
        .as_ref()
        .is_some_and(|c| c.retention.is_some() && !c.automatic)
    {
        d.unsupported_hint(
            RequestField::Cache,
            "Anthropic applies a request-level retention only to automatic caching, which `PromptCache::automatic` turns on; the retention was not sent",
        )?;
    }
    Ok(())
}

/// The request-level settings when they ask for automatic caching, which
/// Anthropic bills as cache writes and so turns on only on explicit intent.
fn automatic_cache(request: &ChatRequest) -> Option<&PromptCache> {
    request.cache.as_ref().filter(|cache| cache.automatic)
}

/// The sampling parameters to send. Thinking (manual **or adaptive**)
/// restricts them, a rule the API documents for thinking as a whole:
/// "`temperature` and `top_k` are incompatible with thinking, and `top_p`
/// is allowed at values between 0.95 and 1", which applies "while thinking
/// is on"; adaptive thinking is thinking on
/// (<https://platform.claude.com/docs/en/build-with-claude/thinking#limits-and-feature-compatibility>).
/// Each value outside that range is dropped as an unsupported hint (an
/// error under the strict policy); a `temperature` of 1, the default, is
/// kept.
///
/// Rules of particular models (rejecting sampling outright, or
/// `temperature` and `top_p` together) are not applied here: the values
/// are sent, and the API's refusal is mapped (ADR 0015).
fn sampling(
    request: &ChatRequest,
    thinking_on: bool,
    d: &mut Degradations,
) -> Result<(Option<f64>, Option<f64>, Option<u32>)> {
    let (mut temperature, mut top_p, mut top_k) =
        (request.temperature, request.top_p, request.top_k);
    if thinking_on {
        if temperature.is_some_and(|t| t != 1.0) {
            d.unsupported_hint(
                RequestField::Temperature,
                "Anthropic accepts only `temperature` 1 while thinking is on; `temperature` was not sent",
            )?;
            temperature = None;
        }
        if top_p.is_some_and(|p| !(0.95..=1.0).contains(&p)) {
            d.unsupported_hint(
                RequestField::TopP,
                "Anthropic accepts only `top_p` between 0.95 and 1 while thinking is on; `top_p` was not sent",
            )?;
            top_p = None;
        }
        if top_k.is_some() {
            d.unsupported_hint(
                RequestField::TopK,
                "Anthropic does not accept `top_k` while thinking is on; `top_k` was not sent",
            )?;
            top_k = None;
        }
    }
    Ok((temperature, top_p, top_k))
}

/// Where a manual thinking budget came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Budget {
    /// The caller's `budget_tokens` option: a setting, never rewritten.
    Option,
    /// Derived from the reasoning effort: a hint, fitted to the output cap.
    Effort,
}

/// Where a cache breakpoint was requested, in render order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Slot {
    Tool(usize),
    System(usize),
    Message(usize),
}

impl Slot {
    fn subject(self) -> Subject {
        match self {
            Slot::Tool(_) => Subject::Field {
                field: RequestField::Tools,
            },
            Slot::System(index) => Subject::System { index },
            Slot::Message(message) => Subject::Content {
                message,
                block: None,
            },
        }
    }
}

fn plan_cache_breakpoints(request: &ChatRequest) -> Vec<Slot> {
    let mut slots = Vec::new();
    for (i, tool) in request.tools.iter().enumerate() {
        if tool.cache.is_some() {
            slots.push(Slot::Tool(i));
        }
    }
    for (i, prompt) in request.system.iter().enumerate() {
        if prompt.cache.is_some() && !prompt.content.is_empty() {
            slots.push(Slot::System(i));
        }
    }
    for (i, message) in request.messages.iter().enumerate() {
        if message.cache().is_some() {
            slots.push(Slot::Message(i));
        }
    }
    slots
}

fn cache_control(breakpoint: CacheBreakpoint) -> CacheControl {
    match breakpoint.retention {
        Some(CacheRetention::Long) => CacheControl::ephemeral(Some(CacheControl::LONG_TTL)),
        _ => CacheControl::ephemeral(None),
    }
}

struct Encoder<'a> {
    request: &'a ChatRequest,
    profile: ModelProfile,
    d: Degradations,
    /// Breakpoints that survive the per-request limit.
    cache_slots: Vec<Slot>,
    /// Server tool definitions from the options, sent verbatim.
    server_tools: &'a [ExtraMap],
    /// What the strict output format adds to the per-request strict-schema
    /// limits; the strict tools share what is left.
    output_complexity: Complexity,
}

impl Encoder<'_> {
    fn unsupported_content(&self, subject: Subject, message: impl Into<String>) -> Error {
        self.d.reject(subject, message)
    }

    /// Report opaque state from other providers, which Anthropic cannot
    /// use, as dropped.
    fn foreign_data(&mut self, data: &ProviderData, subject: Subject) {
        let foreign: Vec<&str> = data.foreign_namespaces(NAMESPACE).collect();
        if !foreign.is_empty() {
            self.d.dropped_replay_state(
                ReplayState::ForeignProviderData,
                subject,
                format!(
                    "provider data from {} can be replayed only to its origin; it was dropped",
                    describe_origins(&foreign)
                ),
            );
        }
    }

    /// Keep at most four breakpoints (three with automatic caching).
    ///
    /// Tool and system breakpoints (stable prefixes) win; among message
    /// breakpoints the most recent ones are kept.
    fn drop_excess_breakpoints(&mut self) {
        let automatic = automatic_cache(self.request).is_some();
        let limit = MAX_CACHE_BREAKPOINTS - usize::from(automatic);
        if self.cache_slots.len() <= limit {
            return;
        }
        let (prefix, messages): (Vec<Slot>, Vec<Slot>) = self
            .cache_slots
            .iter()
            .partition(|s| !matches!(s, Slot::Message(_)));
        let mut keep: Vec<Slot> = prefix.iter().copied().take(limit).collect();
        let room = limit - keep.len();
        keep.extend(messages.iter().rev().take(room));
        let keep_set: HashSet<Slot> = keep.iter().copied().collect();
        for slot in &self.cache_slots {
            if !keep_set.contains(slot) {
                self.d.dropped(
                    slot.subject(),
                    format!(
                        "Anthropic accepts at most {MAX_CACHE_BREAKPOINTS} cache breakpoints per request{}; it was dropped",
                        if automatic {
                            " (automatic caching uses one)"
                        } else {
                            ""
                        }
                    ),
                );
            }
        }
        self.cache_slots.retain(|s| keep_set.contains(s));
    }

    fn cache_for(&self, slot: Slot, breakpoint: Option<CacheBreakpoint>) -> Option<CacheControl> {
        let breakpoint = breakpoint?;
        self.cache_slots
            .contains(&slot)
            .then(|| cache_control(breakpoint))
    }

    fn output_format(&mut self) -> Result<Option<OutputFormat>> {
        match &self.request.response_format {
            None | Some(ResponseFormat::Text) => Ok(None),
            Some(ResponseFormat::Json) => Err(self.d.unsupported(
                RequestField::ResponseFormat,
                "Anthropic has no schemaless JSON mode; use `ResponseFormat::JsonSchema`",
            )),
            Some(ResponseFormat::JsonSchema { schema, .. }) => {
                if self.profile.structured_output() == allama::CapabilitySupport::Unsupported {
                    return Err(self.d.unsupported(
                        RequestField::ResponseFormat,
                        format!(
                            "`{}` does not support structured outputs",
                            self.request.model
                        ),
                    ));
                }
                // The per-request limits are counted before the policy runs.
                // `output_config.format` has no unconstrained mode, so a
                // schema strict mode cannot express, or one that alone
                // exceeds the limits, fails whatever the policy.
                let outcome = Rewritten::new(
                    schema,
                    RequestField::ResponseFormat,
                    self.request.unsupported_policy.unwrap_or_default(),
                )
                .and_then(|rewritten| match rewritten.complexity.overrun(0) {
                    Some(overrun) => Err(StrictSchemaError::not_representable(format!(
                        "it exceeds Anthropic's strict-schema limits per request: {overrun}"
                    ))),
                    None => Ok(rewritten),
                });
                let complexity = outcome
                    .as_ref()
                    .map_or_else(|_| Complexity::default(), |r| r.complexity);
                let StrictOutcome { schema, .. } = apply_strict_policy(
                    schema,
                    RequestField::ResponseFormat,
                    "the response schema",
                    &mut self.d,
                    Rewritten::replay(outcome),
                    Degradable::NoUnconstrainedForm,
                )?;
                self.output_complexity = complexity;
                Ok(Some(OutputFormat {
                    kind: "json_schema",
                    schema,
                }))
            }
            Some(_) => Err(self
                .d
                .unsupported(RequestField::ResponseFormat, "unknown response format")),
        }
    }

    fn system(&mut self) -> Result<Vec<SystemBlock>> {
        let mut blocks = Vec::new();
        for (index, prompt) in self.request.system.iter().enumerate() {
            if prompt.options.json(NAMESPACE).is_some() {
                self.d.dropped(
                    Subject::System { index },
                    "Anthropic reads no per-system-prompt options; the `anthropic` options were dropped",
                );
            }
            if prompt.content.is_empty() {
                continue;
            }
            blocks.push(SystemBlock {
                kind: "text",
                text: prompt.content.clone(),
                cache_control: self.cache_for(Slot::System(index), prompt.cache),
            });
        }
        Ok(blocks)
    }

    fn messages(&mut self) -> Result<Vec<MessageParam>> {
        let mut turns: Vec<(EncodedTurn, Turn)> = Vec::with_capacity(self.request.messages.len());
        let toolsets = toolset_names(&self.request.messages);
        let pairs = ProviderToolPairs::new(&self.request.messages);

        for (index, message) in self.request.messages.iter().enumerate() {
            self.foreign_data(
                message.provider_data(),
                Subject::Content {
                    message: index,
                    block: None,
                },
            );
            let cache = self.cache_for(Slot::Message(index), message.cache());
            let turn = match message {
                Message::User(user) => {
                    let content = match &user.content {
                        // Anthropic rejects empty text blocks; the turn
                        // they leave empty is skipped.
                        UserContent::Text(text) if text.is_empty() => Vec::new(),
                        UserContent::Text(text) => vec![text_block(text.clone(), None)],
                        UserContent::Parts(parts) => self.user_parts(index, parts)?,
                        other => self.user_parts(index, &other.to_parts())?,
                    };
                    self.message_turn(index, message, "user", content, cache)
                }
                Message::Assistant(assistant) => {
                    let mut content = Vec::with_capacity(assistant.content.len());
                    for (block_index, block) in assistant.content.iter().enumerate() {
                        if let Some(block) =
                            self.assistant_block(index, block_index, block, &pairs)?
                        {
                            content.push(block);
                        }
                    }
                    self.message_turn(index, message, "assistant", content, cache)
                }
                Message::Tool(tool) => {
                    let content = match &tool.content {
                        ToolResultContent::Text(text) => ToolResultContentParam::Text(text.clone()),
                        ToolResultContent::Object(object) => {
                            ToolResultContentParam::Text(Value::Object(object.clone()).to_string())
                        }
                        ToolResultContent::Parts(parts) => {
                            ToolResultContentParam::Blocks(self.user_parts(index, parts)?)
                        }
                        _ => {
                            return Err(self.unsupported_content(
                                Subject::Content {
                                    message: index,
                                    block: None,
                                },
                                "unknown tool result content is not supported by Anthropic",
                            ));
                        }
                    };
                    let mut block = RequestBlock::new(KnownBlock::ToolResult {
                        tool_use_id: TOOL_CALL_IDS.conform(&tool.tool_call_id).into_owned(),
                        toolset_name: toolsets.get(tool.tool_call_id.as_str()).copied().cloned(),
                        content,
                        is_error: tool.is_error,
                    });
                    block.cache_control = cache;
                    (EncodedTurn::tool(index), Turn::ToolResult(block))
                }
                _ => {
                    return Err(self.unsupported_content(
                        Subject::Content {
                            message: index,
                            block: None,
                        },
                        "unknown message variant is not supported by Anthropic",
                    ));
                }
            };
            turns.push(turn);
        }

        // Skip empty turns, and drop a reply the skips and drops left last.
        let turns = retain_turns(turns, &mut self.d, |(turn, _)| *turn)?;

        // A run of tool results is one user turn of `tool_result` blocks.
        let mut out: Vec<MessageParam> = Vec::with_capacity(turns.len());
        let mut pending_results: Vec<RequestBlock> = Vec::new();
        for (_, turn) in turns {
            match turn {
                Turn::Message(message) => {
                    flush(&mut out, &mut pending_results);
                    out.push(message);
                }
                Turn::ToolResult(block) => pending_results.push(block),
            }
        }
        flush(&mut out, &mut pending_results);

        // Anthropic requires alternating roles; merge adjacent same-role turns
        // (tool results followed by a user message, injected summaries, …).
        let mut merged = merge_consecutive_same_role(out);
        // "tool_result blocks must come first in the content array": a user
        // message merged after the results follows them.
        for message in merged.iter_mut().filter(|m| m.role == "user") {
            message.content.sort_by_key(|block| !block.is_tool_result());
        }
        Ok(merged)
    }

    /// A user or assistant turn, with its cache breakpoint attached unless
    /// it is empty and will be skipped.
    fn message_turn(
        &mut self,
        index: usize,
        message: &Message,
        role: &'static str,
        mut content: Vec<RequestBlock>,
        cache: Option<CacheControl>,
    ) -> (EncodedTurn, Turn) {
        let turn = EncodedTurn::new(index, message, content.is_empty());
        if turn.is_empty() {
            if cache.is_some() {
                self.d.dropped(
                    Subject::Content {
                        message: index,
                        block: None,
                    },
                    "an empty message is skipped and cannot carry a cache breakpoint; the breakpoint was dropped",
                );
            }
        } else {
            self.attach_cache(index, &mut content, cache);
        }
        (turn, Turn::Message(MessageParam { role, content }))
    }

    /// Put the breakpoint on the last block that can carry one.
    fn attach_cache(
        &mut self,
        message: usize,
        content: &mut [RequestBlock],
        cache: Option<CacheControl>,
    ) {
        let Some(cache) = cache else {
            return;
        };
        match content
            .iter_mut()
            .rev()
            .find(|b| b.accepts_cache_control())
        {
            Some(block) => block.cache_control = Some(cache),
            None => self.d.dropped(
                Subject::Content {
                    message,
                    block: None,
                },
                "Anthropic cannot cache thinking blocks, and the message has no other content; the cache breakpoint was dropped",
            ),
        }
    }

    fn user_parts(&mut self, message: usize, parts: &[ContentPart]) -> Result<Vec<RequestBlock>> {
        let mut blocks = Vec::with_capacity(parts.len());
        for (block, part) in parts.iter().enumerate() {
            let subject = Subject::Content {
                message,
                block: Some(block),
            };
            let encoded = match part {
                // Anthropic rejects empty text blocks; nothing is lost.
                ContentPart::Text(text) if text.text.is_empty() => continue,
                ContentPart::Text(text) => {
                    self.foreign_data(&text.provider_data, subject);
                    text_block(text.text.clone(), None)
                }
                ContentPart::Image(image) => {
                    self.foreign_data(&image.provider_data, subject.clone());
                    self.image_block(subject, image)?
                }
                ContentPart::File(file) => {
                    self.foreign_data(&file.provider_data, subject.clone());
                    RequestBlock::new(self.document_block(subject, file)?)
                }
                ContentPart::Other(other) => match raw_part(other) {
                    Some(raw) => raw,
                    None => {
                        self.d.dropped_replay_state(
                            ReplayState::ForeignContent,
                            subject,
                            format!(
                                "`{}` content from {} can be replayed only to its origin; it was dropped",
                                other.type_name,
                                describe_origins(other.origin.as_slice())
                            ),
                        );
                        continue;
                    }
                },
                _ => {
                    return Err(self.unsupported_content(
                        subject,
                        format!(
                            "content part `{}` is not supported by Anthropic",
                            part.type_name()
                        ),
                    ));
                }
            };
            blocks.push(encoded);
        }
        Ok(blocks)
    }

    fn image_block(&mut self, subject: Subject, image: &Image) -> Result<RequestBlock> {
        if let Some(detail) = image.detail {
            self.d.hint(
                subject.clone(),
                format!(
                    "Anthropic has no image detail control; `{}` was not sent",
                    detail.as_str()
                ),
            )?;
        }
        let source = match &image.source {
            MediaSource::Url { url } => MediaSourceParam::Url { url: url.clone() },
            MediaSource::Base64 {
                media_type, data, ..
            } => MediaSourceParam::Base64 {
                media_type: essence(media_type).to_owned(),
                data: data.to_string(),
            },
            MediaSource::ProviderFile { origin, id, .. } if origin == NAMESPACE => {
                MediaSourceParam::File {
                    file_id: id.clone(),
                }
            }
            _ => {
                return Err(self.unsupported_content(
                    subject,
                    "this image source is not supported by Anthropic",
                ));
            }
        };
        Ok(RequestBlock::new(KnownBlock::Image { source }))
    }

    fn document_block(&self, subject: Subject, file: &File) -> Result<KnownBlock> {
        // `text/plain; charset=utf-8` is a text document like `text/plain`.
        let media_type = essence(&file.media_type);
        let source = match (media_type, &file.source) {
            ("application/pdf" | "text/plain", MediaSource::ProviderFile { origin, id, .. })
                if origin == NAMESPACE =>
            {
                DocumentSource::File {
                    file_id: id.clone(),
                }
            }
            ("application/pdf", MediaSource::Base64 { data, .. }) => DocumentSource::Base64 {
                media_type: media_type.to_owned(),
                data: data.to_string(),
            },
            ("application/pdf", MediaSource::Url { url }) => {
                DocumentSource::Url { url: url.clone() }
            }
            ("text/plain", MediaSource::Base64 { data, .. }) => {
                let bytes = decode_base64(data).ok_or_else(|| {
                    Error::invalid_request("text/plain file data is not valid base64")
                        .authored()
                        .with_provider(PROVIDER_NAME)
                })?;
                let text = String::from_utf8(bytes).map_err(|_| {
                    Error::invalid_request("text/plain file data is not valid UTF-8")
                        .authored()
                        .with_provider(PROVIDER_NAME)
                })?;
                DocumentSource::Text {
                    media_type: media_type.to_owned(),
                    data: text,
                }
            }
            ("text/plain", MediaSource::Url { .. }) => {
                return Err(self.unsupported_content(
                    subject,
                    "Anthropic accepts text/plain documents inline or as a Files API file only, not by URL",
                ));
            }
            ("application/pdf" | "text/plain", _) => {
                return Err(self.unsupported_content(
                    subject,
                    "this file source is not supported by Anthropic",
                ));
            }
            _ => {
                return Err(self.unsupported_content(
                    subject,
                    format!(
                        "Anthropic does not accept `{media_type}` files (supported: application/pdf, text/plain)"
                    ),
                ));
            }
        };
        // Document fields without a portable form (`citations`, `context`)
        // come from the file's own provider data.
        let extra = file
            .provider_data
            .namespace(NAMESPACE)
            .cloned()
            .unwrap_or_default();
        check_reserved(&extra, DOCUMENT_FIELDS, "File::provider_data")?;
        Ok(KnownBlock::Document {
            source,
            title: file.filename.clone(),
            extra,
        })
    }

    fn assistant_block(
        &mut self,
        message: usize,
        index: usize,
        block: &ContentBlock,
        pairs: &ProviderToolPairs<'_>,
    ) -> Result<Option<RequestBlock>> {
        let subject = Subject::Content {
            message,
            block: Some(index),
        };
        Ok(match block {
            // Anthropic rejects empty text blocks (a unary response may hold
            // one); nothing is lost.
            ContentBlock::Text(text) if text.text.is_empty() => None,
            ContentBlock::Refusal(refusal) if refusal.text.is_empty() => None,
            ContentBlock::Text(text) => {
                self.foreign_data(&text.provider_data, subject);
                Some(text_block(
                    text.text.clone(),
                    text.provider_data.get_raw(NAMESPACE, "citations").cloned(),
                ))
            }
            // Anthropic has no refusal block: a model-written refusal from
            // another provider is replayed as the assistant text it was.
            ContentBlock::Refusal(refusal) => {
                self.foreign_data(&refusal.provider_data, subject);
                Some(text_block(refusal.text.clone(), None))
            }
            ContentBlock::ToolCall(call) => {
                self.foreign_data(&call.provider_data, subject.clone());
                Some(self.tool_use_block(subject, call)?)
            }
            ContentBlock::Reasoning(reasoning) => self.thinking_block(subject, reasoning),
            ContentBlock::Image(_) => {
                return Err(self.unsupported_content(
                    subject,
                    "Anthropic cannot replay images in assistant messages",
                ));
            }
            ContentBlock::Other(other) => match raw_block(other) {
                Some(raw) => Some(raw),
                None => {
                    self.d.dropped_replay_state(
                        ReplayState::ForeignContent,
                        subject,
                        format!(
                            "`{}` content from {} can be replayed only to its origin; it was dropped",
                            other.type_name,
                            describe_origins(other.origin.as_slice())
                        ),
                    );
                    None
                }
            },
            ContentBlock::ProviderToolCall(call) => match pairs.call(call) {
                Ok(()) => Some(provider_tool_call_block(call)),
                Err(unsent) => {
                    self.unsent_provider_tool_block(subject, block, unsent);
                    None
                }
            },
            ContentBlock::ProviderToolResult(result) => match pairs.result(result) {
                Ok(result_type) => Some(provider_tool_result_block(result, result_type)),
                Err(unsent) => {
                    self.unsent_provider_tool_block(subject, block, unsent);
                    None
                }
            },
            _ => {
                self.d.dropped(
                    subject,
                    format!(
                        "Anthropic has no `{}` block; it was dropped",
                        block.type_name()
                    ),
                );
                None
            }
        })
    }

    /// Report a server or MCP tool block that is not sent.
    fn unsent_provider_tool_block(
        &mut self,
        subject: Subject,
        block: &ContentBlock,
        unsent: Unsent,
    ) {
        match unsent {
            Unsent::Foreign => self.d.dropped_replay_state(
                ReplayState::ForeignContent,
                subject,
                format!(
                    "`{}` content from {} can be replayed only to its origin; it was dropped",
                    block.type_name(),
                    describe_origins(block.origin().as_slice())
                ),
            ),
            Unsent::Unpaired(reason) => {
                self.d
                    .dropped_replay_state(ReplayState::Content, subject, reason);
            }
        }
    }

    fn tool_use_block(&mut self, subject: Subject, call: &ToolCall) -> Result<RequestBlock> {
        let input = if call.arguments.trim().is_empty() {
            Value::Object(ExtraMap::new())
        } else {
            match serde_json::from_str::<Value>(&call.arguments) {
                Ok(value @ Value::Object(_)) => value,
                parsed => {
                    let reason = if parsed.is_ok() {
                        "are not a JSON object"
                    } else {
                        "are not valid JSON"
                    };
                    let message = format!(
                        "the arguments of tool call `{}` {reason}; they were sent as `{{}}`",
                        call.name
                    );
                    self.d.coded_lossy_hint(
                        subject,
                        message,
                        "approximated.tool_call_arguments",
                    )?;
                    Value::Object(ExtraMap::new())
                }
            }
        };
        // The fields the decoder kept (`caller`, `toolset_name`, …) go back
        // on the block; the ones the block already sets cannot be replaced.
        let mut extra = call
            .provider_data
            .namespace(NAMESPACE)
            .cloned()
            .unwrap_or_default();
        for key in ["type", "id", "name", "input"] {
            extra.remove(key);
        }
        Ok(RequestBlock::new(KnownBlock::ToolUse {
            id: TOOL_CALL_IDS.conform(&call.id).into_owned(),
            name: call.name.clone(),
            input,
            extra,
        }))
    }

    /// Replay only Anthropic-originated thinking.
    fn thinking_block(&mut self, subject: Subject, reasoning: &Reasoning) -> Option<RequestBlock> {
        let data = &reasoning.provider_data;
        if let Some(Value::String(redacted)) = data.get_raw(NAMESPACE, "redacted_data") {
            return Some(RequestBlock::new(KnownBlock::RedactedThinking {
                data: redacted.clone(),
            }));
        }
        if let Some(Value::String(signature)) = data.get_raw(NAMESPACE, "signature") {
            return Some(RequestBlock::new(KnownBlock::Thinking {
                thinking: reasoning.text.clone(),
                signature: signature.clone(),
            }));
        }
        let foreign: Vec<&str> = data.foreign_namespaces(NAMESPACE).collect();
        let (state, message) = if foreign.is_empty() {
            (
                ReplayState::Reasoning,
                "reasoning without an Anthropic signature cannot be replayed as thinking; it was dropped"
                    .to_owned(),
            )
        } else {
            (
                ReplayState::ForeignReasoning,
                format!(
                    "reasoning from {} can be replayed only to its origin; it was dropped",
                    describe_origins(&foreign)
                ),
            )
        };
        self.d.dropped_replay_state(state, subject, message);
        None
    }

    fn tools(&mut self) -> Result<(Option<Vec<ToolParam>>, Option<ToolChoice>)> {
        let tools = self.request.tools.as_slice();
        let mut definitions = Vec::with_capacity(tools.len());
        for (index, (tool, strict)) in tools.iter().zip(self.strict_tools(tools)).enumerate() {
            let (input_schema, strict_field) = match strict {
                StrictTool::NotRequested => (tool.parameters.clone(), None),
                // Strict is a hint on tools: when it cannot be honored, the
                // tool goes out without strict, as the caller wrote it.
                StrictTool::Unavailable(message) => {
                    self.d.unsupported_hint(RequestField::Tools, message)?;
                    (without_schema_uri(&tool.parameters), None)
                }
                StrictTool::Requested(outcome) => {
                    let StrictOutcome { schema, strict, .. } = apply_strict_policy(
                        &tool.parameters,
                        RequestField::Tools,
                        &format!("tool `{}`", tool.name),
                        &mut self.d,
                        Rewritten::replay(outcome),
                        Degradable::Yes,
                    )?;
                    (schema, strict.then_some(true))
                }
            };
            // Tool fields without a portable form (`defer_loading`,
            // `allowed_callers`, …) come from the tool's own provider data.
            let extra = tool
                .provider_data
                .namespace(NAMESPACE)
                .cloned()
                .unwrap_or_default();
            check_reserved(&extra, TOOL_FIELDS, "ToolDefinition::provider_data")?;
            definitions.push(ToolParam {
                name: Some(tool.name.clone()),
                description: tool.description.clone(),
                input_schema: Some(input_schema),
                strict: strict_field,
                cache_control: self.cache_for(Slot::Tool(index), tool.cache),
                extra,
            });
        }
        for raw in self.server_tools {
            let mut extra = raw.clone();
            let name = match extra.remove("name") {
                Some(Value::String(name)) => Some(name),
                // A toolset fixes its member tools' names and declares none.
                None if is_computer_toolset(&extra) => None,
                _ => {
                    return Err(self.d.unsupported(
                        RequestField::Tools,
                        "server tool definitions need a `name` (only the computer toolset has none)",
                    ));
                }
            };
            definitions.push(ToolParam {
                name,
                description: None,
                input_schema: None,
                strict: None,
                cache_control: None,
                extra,
            });
        }

        let has_tools = !definitions.is_empty();
        let mut choice = match &self.request.tool_choice {
            None => None,
            Some(allama::ToolChoice::Auto) => has_tools.then(|| ToolChoice::of("auto")),
            Some(allama::ToolChoice::Disabled) => has_tools.then(|| ToolChoice::of("none")),
            Some(allama::ToolChoice::Required | allama::ToolChoice::Specific { .. })
                if !has_tools =>
            {
                return Err(Error::invalid_request(
                    "a forced tool choice (`ToolChoice::Required` or `Specific`) needs at least one tool",
                )
                .authored()
                .with_provider(PROVIDER_NAME));
            }
            Some(allama::ToolChoice::Required) => Some(self.forced(ToolChoice::of("any"))?),
            Some(allama::ToolChoice::Specific { name }) => Some(self.forced(ToolChoice {
                name: Some(name.clone()),
                ..ToolChoice::of("tool")
            })?),
            Some(_) => {
                return Err(self
                    .d
                    .unsupported(RequestField::ToolChoice, "unknown tool choice"));
            }
        };

        if self.request.parallel_tool_calls == Some(false)
            && has_tools
            && choice.as_ref().is_none_or(|c| c.kind != "none")
        {
            choice
                .get_or_insert_with(|| ToolChoice::of("auto"))
                .disable_parallel_tool_use = Some(true);
        }

        Ok((has_tools.then_some(definitions), choice))
    }

    /// Decide, for each tool, what the strict policy is handed.
    ///
    /// Each strict tool is rewritten on its own first, its warnings held
    /// back. Then Anthropic's per-request limits (strict tools, optional and
    /// union-typed parameters, shared with the output format) are counted,
    /// before the policy runs: strict tools degrade from the last one until
    /// the request fits, each as not representable. The policy
    /// ([`apply_strict_policy`]) then sends each tool strict, with its
    /// warnings, or non-strict with an `Unsupported` warning, or fails under
    /// `UnsupportedPolicy::Error`.
    fn strict_tools(&self, tools: &[allama::ToolDefinition]) -> Vec<StrictTool> {
        let structured = self.profile.structured_output() != CapabilitySupport::Unsupported;
        let mut outcomes: Vec<StrictTool> = tools
            .iter()
            .map(|tool| {
                if tool.strict != Some(true) {
                    return StrictTool::NotRequested;
                }
                if !structured {
                    return StrictTool::Unavailable(format!(
                        "`{}` does not support strict tools; `{}` was sent without `strict`",
                        self.request.model, tool.name
                    ));
                }
                StrictTool::Requested(Rewritten::new(
                    &tool.parameters,
                    RequestField::Tools,
                    self.request.unsupported_policy.unwrap_or_default(),
                ))
            })
            .collect();

        let mut strict: Vec<(usize, Complexity)> = outcomes
            .iter()
            .enumerate()
            .filter_map(|(index, outcome)| match outcome {
                StrictTool::Requested(Ok(rewritten)) => Some((index, rewritten.complexity)),
                _ => None,
            })
            .collect();
        let mut total = strict
            .iter()
            .fold(self.output_complexity, |total, (_, c)| total + *c);
        while let Some(overrun) = total.overrun(strict.len()) {
            let Some((index, last)) = strict.pop() else {
                break;
            };
            outcomes[index] = StrictTool::Requested(Err(StrictSchemaError::not_representable(
                format!(
                    "the request exceeds Anthropic's strict-schema limits ({overrun}), so strict tools are relaxed from the last one"
                ),
            )));
            total = total - last;
        }
        outcomes
    }

    /// A forced tool choice, refused only where the caller's resolver
    /// answers `Unsupported`; the built-in answers never do (ADR 0015).
    fn forced(&self, choice: ToolChoice) -> Result<ToolChoice> {
        if self.profile.forced_tool_choice() == CapabilitySupport::Unsupported {
            return Err(self.d.unsupported(
                RequestField::ToolChoice,
                format!(
                    "`{}` rejects forced tool choice (`any`/`tool`); use `ToolChoice::Auto` with a prompt instruction, or `ResponseFormat::JsonSchema`",
                    self.request.model
                ),
            ));
        }
        Ok(choice)
    }

    /// Map reasoning settings. Returns the thinking config, effort, and the
    /// final `max_tokens`.
    fn thinking(
        &mut self,
        options: &ChatRequestOptions,
        has_tools: bool,
    ) -> Result<(Option<ThinkingConfig>, Option<&'static str>, u32)> {
        let mut max_tokens = self.request.max_output_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
        let interleaved = has_tools
            && self.profile.interleaves_manual_thinking()
            && options
                .anthropic_beta
                .iter()
                .any(|beta| beta == INTERLEAVED_THINKING_BETA);
        let reasoning = self.request.reasoning;
        let model = self.request.model.clone();
        let reasoning_subject = Subject::Field {
            field: RequestField::Reasoning,
        };

        let mut effort = None;
        let mut thinking = if let Some(budget) = options.budget_tokens {
            // A fixed budget is sent on every model; one that takes only
            // adaptive thinking refuses it, and the refusal is mapped.
            if budget < MIN_THINKING_BUDGET {
                return Err(Error::invalid_request(format!(
                    "Anthropic budget_tokens must be at least {MIN_THINKING_BUDGET}"
                ))
                .authored()
                .with_provider(PROVIDER_NAME));
            }
            if reasoning.and_then(|r| r.effort).is_some() {
                self.d.approximated(
                    reasoning_subject.clone(),
                    "the `anthropic` options set `budget_tokens`, which overrides the reasoning effort; `budget_tokens` was used instead",
                );
            }
            self.budget_thinking(budget, Budget::Option, &mut max_tokens, interleaved)?
        } else if let Some(config) = reasoning
            && self.profile.thinking() == CapabilitySupport::Unsupported
        {
            // Asking a model without extended thinking not to think is
            // already honored; anything else cannot be.
            if config.effort != Some(ReasoningEffort::None) {
                self.d.unsupported_hint(
                    RequestField::Reasoning,
                    format!(
                        "`{model}` does not support extended thinking; `reasoning` was not sent"
                    ),
                )?;
            }
            None
        } else if let Some(config) = reasoning {
            match (config.effort, self.profile.thinking_style()) {
                (Some(ReasoningEffort::None), _) => match self.profile.disable_thinking() {
                    DisableThinking::Omit => None,
                    DisableThinking::Disabled => Some(ThinkingConfig::of("disabled")),
                    DisableThinking::BetweenTools => Some(ThinkingConfig::of("between_tools")),
                },
                // No effort keeps the provider default: the output setting
                // alone never turns thinking on.
                (None, _) => self.default_thinking(config.output)?,
                (Some(requested), ThinkingStyle::Budget) => {
                    let budget = model::budget_for_effort(requested);
                    self.budget_thinking(budget, Budget::Effort, &mut max_tokens, interleaved)?
                }
                (Some(requested), ThinkingStyle::Adaptive) => {
                    if let Some(levels) = self.profile.effort_levels() {
                        let (level, lossy) = model::map_effort(requested, levels);
                        if lossy {
                            self.d.approximated(
                                reasoning_subject.clone(),
                                format!(
                                    "`{model}` has no reasoning effort `{}`; `{}` was used instead",
                                    requested.as_str(),
                                    level.as_str()
                                ),
                            );
                        }
                        effort = Some(level.as_str());
                    }
                    Some(ThinkingConfig::of("adaptive"))
                }
            }
        } else {
            None
        };

        if let (Some(config), Some(output)) = (
            thinking.as_mut().filter(|t| t.is_on()),
            reasoning.and_then(|r| r.output),
        ) {
            self.thinking_display(config, output);
        }
        // The caller's level replaces the mapped one, and nothing else.
        if let Some(level) = options.effort {
            effort = Some(level.as_str());
        }

        Ok((thinking, effort, max_tokens))
    }

    /// The thinking config for a reasoning config without an effort, which
    /// keeps the model's default. Where the model thinks by default, the
    /// output setting goes out as `display` on the adaptive form it already
    /// runs. Elsewhere sending it would turn thinking on, which only an
    /// effort asks for, so it has no effect: an unsupported hint.
    fn default_thinking(
        &mut self,
        output: Option<ReasoningOutput>,
    ) -> Result<Option<ThinkingConfig>> {
        let Some(output) = output else {
            return Ok(None);
        };
        if self.profile.thinking_style() == ThinkingStyle::Adaptive
            && self.profile.thinks_by_default()
        {
            return Ok(Some(ThinkingConfig::of("adaptive")));
        }
        self.d.unsupported_hint(
            RequestField::Reasoning,
            format!(
                "reasoning output `{output}` applies only while thinking is on, and `{}` does not think unless asked; it was not sent; set a reasoning effort to turn thinking on",
                self.request.model
            ),
        )?;
        Ok(None)
    }

    fn thinking_display(&mut self, config: &mut ThinkingConfig, output: ReasoningOutput) {
        let subject = Subject::Field {
            field: RequestField::Reasoning,
        };
        if !self.profile.supports_thinking_display() {
            if output != ReasoningOutput::Summary {
                self.d.approximated(
                    subject,
                    format!(
                        "`{}` always returns summarized thinking; reasoning output `summary` was used instead of `{output}`",
                        self.request.model
                    ),
                );
            }
            return;
        }
        config.display = Some(match output {
            ReasoningOutput::Omitted => "omitted",
            ReasoningOutput::Summary => "summarized",
            _ => {
                self.d.approximated(
                    subject,
                    "Anthropic never returns raw thinking; reasoning output `summary` was used instead",
                );
                "summarized"
            }
        });
    }

    /// `{type: "enabled", budget_tokens}`, fitted to `max_tokens`.
    ///
    /// Anthropic counts thinking against `max_tokens` and requires
    /// `budget_tokens < max_tokens`, except with interleaved thinking, where
    /// the budget spans every thinking block of the turn and may exceed it
    /// (<https://platform.claude.com/docs/en/build-with-claude/extended-thinking#budget-rules-and-tuning>).
    /// A caller-set `max_output_tokens` is never raised (ADR 0003,
    /// principle 4):
    ///
    /// - interleaved: the budget is sent as it is, and `max_tokens` too;
    /// - `max_output_tokens` unset: the default `max_tokens` is raised to fit;
    /// - an explicit `budget_tokens` option that does not fit an explicit
    ///   `max_output_tokens` contradicts it: `InvalidRequest`;
    /// - a budget derived from the effort is a hint. Under an explicit
    ///   `max_output_tokens` above the 1024-token minimum it is
    ///   `min(budget, max(cap / 2, 1024))`: half the cap is kept for the
    ///   answer where the minimum allows, and the budget never shrinks as the
    ///   cap grows. A lowered budget is an `Approximated` warning, never
    ///   refused. At or below the minimum, thinking is not enabled (an
    ///   unsupported hint).
    fn budget_thinking(
        &mut self,
        budget: u32,
        source: Budget,
        max_tokens: &mut u32,
        interleaved: bool,
    ) -> Result<Option<ThinkingConfig>> {
        let enabled = |budget| {
            Some(ThinkingConfig {
                budget_tokens: Some(budget),
                ..ThinkingConfig::of("enabled")
            })
        };
        if interleaved {
            return Ok(enabled(budget));
        }
        let Some(cap) = self.request.max_output_tokens else {
            if budget >= *max_tokens {
                *max_tokens = budget.checked_add(*max_tokens).ok_or_else(|| {
                    Error::invalid_request("Anthropic thinking budget is too large")
                        .authored()
                        .with_provider(PROVIDER_NAME)
                })?;
            }
            return Ok(enabled(budget));
        };
        match source {
            Budget::Option if budget < cap => Ok(enabled(budget)),
            Budget::Option => Err(Error::invalid_request(format!(
                "budget_tokens ({budget}) from the anthropic options must be less than max_output_tokens ({cap}): Anthropic counts thinking against max_tokens"
            ))
            .authored()
            .with_provider(PROVIDER_NAME)),
            Budget::Effort => {
                if cap <= MIN_THINKING_BUDGET {
                    self.d.unsupported_hint(
                        RequestField::Reasoning,
                        format!(
                            "thinking needs `max_output_tokens` above {MIN_THINKING_BUDGET}: Anthropic's minimum thinking budget is {MIN_THINKING_BUDGET} tokens, and part of `max_output_tokens` is kept for the answer; `reasoning` was not sent; raise `max_output_tokens` above {MIN_THINKING_BUDGET} to enable thinking"
                        ),
                    )?;
                    return Ok(None);
                }
                // Half the cap is kept for the answer, but never below the
                // minimum, so `Minimal` still thinks under caps 1025-2047.
                // Monotone in the cap: a larger cap never lowers the budget.
                let fitted = budget.min((cap / 2).max(MIN_THINKING_BUDGET));
                if fitted < budget {
                    self.d.approximated(
                        Subject::Field {
                            field: RequestField::Reasoning,
                        },
                        format!(
                            "`max_output_tokens` ({cap}) includes thinking, and half of it is kept for the answer; a thinking budget of {fitted} tokens was used instead of {budget}"
                        ),
                    );
                }
                Ok(enabled(fitted))
            }
        }
    }

    /// Enforce TTL ordering (1h entries must precede 5m entries) and add the
    /// request-level automatic cache marker.
    fn finish_cache(&mut self, body: &mut CreateMessageRequest) {
        let mut seen_short = false;
        let mut downgraded = false;
        let mut visit = |cc: &mut Option<CacheControl>| {
            if let Some(cc) = cc {
                if cc.is_long() && seen_short {
                    cc.ttl = None;
                    downgraded = true;
                } else if !cc.is_long() {
                    seen_short = true;
                }
            }
        };
        for tool in body.tools.iter_mut().flatten() {
            visit(&mut tool.cache_control);
        }
        for block in body.system.iter_mut().flatten() {
            visit(&mut block.cache_control);
        }
        for message in &mut body.messages {
            for block in &mut message.content {
                visit(&mut block.cache_control);
                if let BlockParam::Known(KnownBlock::ToolResult {
                    content: ToolResultContentParam::Blocks(inner),
                    ..
                }) = &mut block.block
                {
                    inner.iter_mut().for_each(|b| visit(&mut b.cache_control));
                }
            }
        }
        if downgraded {
            self.d.approximated(
                Subject::Field {
                    field: RequestField::Cache,
                },
                "Anthropic requires 1h cache entries before 5m entries; a long-retention breakpoint after a short one was sent as short",
            );
        }

        if let Some(cache) = automatic_cache(self.request) {
            let mut ttl = match cache.retention {
                Some(CacheRetention::Long) => Some(CacheControl::LONG_TTL),
                _ => None,
            };
            if ttl.is_some() && seen_short {
                ttl = None;
                self.d.approximated(
                    Subject::Field {
                        field: RequestField::Cache,
                    },
                    "Anthropic requires 1h cache entries before 5m entries, and an explicit 5m breakpoint precedes automatic caching; the short (5m) retention was used instead",
                );
            }
            body.cache_control = Some(CacheControl::ephemeral(ttl));
        }
    }
}

/// The conversation's server and MCP tool blocks, paired by call id across
/// blocks and messages: a call and its result can arrive in two responses
/// (a turn that waited on client tools, or on code calling them).
///
/// A call and its result are sent together or not at all. Anthropic
/// rejects a finished turn with a `server_tool_use` whose result is missing
/// ("`web_fetch` tool use with id `srvtoolu_…` was found without a
/// corresponding `web_fetch_tool_result` block":
/// <https://platform.claude.com/docs/en/agents-and-tools/tool-use/server-tools#mixing-server-tools-and-client-tools-in-one-turn>),
/// so a result that cannot be sent takes its call with it, and the reverse.
/// A call with no result anywhere in the history is sent alone, as
/// Anthropic returned it: a paused turn (`pause_turn`), or one waiting on
/// client tools, ends with one, and the API runs it on the next request.
struct ProviderToolPairs<'a> {
    /// The server and MCP tool calls, by id.
    calls: HashMap<&'a str, &'a ProviderToolCall>,
    /// The call ids whose call and result are not sent, with the reason.
    unpaired: HashMap<&'a str, String>,
}

impl<'a> ProviderToolPairs<'a> {
    fn new(messages: &'a [Message]) -> Self {
        let blocks = || {
            messages
                .iter()
                .filter_map(|message| match message {
                    Message::Assistant(assistant) => Some(&assistant.content),
                    _ => None,
                })
                .flatten()
        };
        let mut pairs = Self {
            calls: HashMap::new(),
            unpaired: HashMap::new(),
        };
        for block in blocks() {
            if let ContentBlock::ProviderToolCall(call) = block {
                pairs.calls.entry(call.id.as_str()).or_insert(call);
            }
        }
        for block in blocks() {
            let unpaired = match block {
                ContentBlock::ProviderToolCall(call) if !is_anthropic(&call.origin) => Some((
                    call.id.as_str(),
                    format!(
                        "server tool call `{}` is another provider's and was dropped, so its result was dropped with it",
                        call.id
                    ),
                )),
                ContentBlock::ProviderToolResult(result) => pairs.unpaired_result(result),
                _ => None,
            };
            if let Some((id, reason)) = unpaired {
                pairs.unpaired.entry(id).or_insert(reason);
            }
        }
        pairs
    }

    /// Why `result` cannot be sent, with its call id, if it cannot.
    fn unpaired_result(&self, result: &'a ProviderToolResult) -> Option<(&'a str, String)> {
        let id = result.call_id.as_str();
        if !is_anthropic(&result.origin) {
            return Some((
                id,
                format!(
                    "the result of server tool call `{id}` is another provider's and was dropped, so the call was dropped with it"
                ),
            ));
        }
        if self.result_type(result).is_some() {
            return None;
        }
        let reason = match self.anthropic_call(id) {
            Some(call) => format!(
                "the result of `{}` server tool call `{id}` has no recorded block type, and Anthropic documents none for that tool; the call and its result were dropped together, since Anthropic rejects a call without its result",
                call.name
            ),
            None => format!(
                "the result of server tool call `{id}` has no recorded block type and no call to infer it from; it was dropped"
            ),
        };
        Some((id, reason))
    }

    /// Whether `call` is sent.
    fn call(&self, call: &ProviderToolCall) -> std::result::Result<(), Unsent<'_>> {
        if !is_anthropic(&call.origin) {
            return Err(Unsent::Foreign);
        }
        match self.unpaired.get(call.id.as_str()) {
            Some(reason) => Err(Unsent::Unpaired(reason)),
            None => Ok(()),
        }
    }

    /// The block type `result` is sent as, or why it is not sent.
    fn result(&self, result: &'a ProviderToolResult) -> std::result::Result<&'a str, Unsent<'_>> {
        if !is_anthropic(&result.origin) {
            return Err(Unsent::Foreign);
        }
        if let Some(reason) = self.unpaired.get(result.call_id.as_str()) {
            return Err(Unsent::Unpaired(reason));
        }
        // Not unpaired, so the type was recorded or is inferred.
        self.result_type(result).ok_or(Unsent::Unpaired(
            "a server tool result without a block type was dropped",
        ))
    }

    /// The recorded block type of `result`, or the one its call fixes.
    fn result_type(&self, result: &'a ProviderToolResult) -> Option<&'a str> {
        recorded_type(&result.provider_data).or_else(|| {
            self.anthropic_call(&result.call_id)
                .and_then(documented_result_type)
        })
    }

    fn anthropic_call(&self, id: &str) -> Option<&'a ProviderToolCall> {
        self.calls
            .get(id)
            .copied()
            .filter(|call| is_anthropic(&call.origin))
    }
}

/// Why a server or MCP tool block is not sent.
enum Unsent<'a> {
    /// Another provider wrote it.
    Foreign,
    /// It is Anthropic's, but its call or result cannot be sent, so neither
    /// is; the reason says why.
    Unpaired(&'a str),
}

/// Whether a server tool definition is Anthropic's computer toolset
/// (`{"type": "computer_toolset_20260801"}`), which has no `name`: its
/// member tools (`screenshot`, `left_click`, …) are named by the toolset
/// version.
fn is_computer_toolset(definition: &ExtraMap) -> bool {
    definition
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| kind.starts_with("computer_toolset_"))
}

/// The `toolset_name` of every tool call that has one, by call id.
///
/// A computer toolset member's call (`computer_toolset_20260801`) carries
/// `toolset_name`, and its `tool_result` must repeat it. The tool message
/// answers by id only, so the encoder copies the name from the call.
fn toolset_names(messages: &[Message]) -> HashMap<&str, &Value> {
    messages
        .iter()
        .filter_map(|message| match message {
            Message::Assistant(assistant) => Some(&assistant.content),
            _ => None,
        })
        .flatten()
        .filter_map(|block| match block {
            ContentBlock::ToolCall(call) => call
                .provider_data
                .get_raw(NAMESPACE, "toolset_name")
                .map(|name| (call.id.as_str(), name)),
            _ => None,
        })
        .collect()
}

fn text_block(text: String, citations: Option<Value>) -> RequestBlock {
    RequestBlock::new(KnownBlock::Text { text, citations })
}

/// Replay an Anthropic-originated block verbatim.
fn raw_block(other: &OtherContent) -> Option<RequestBlock> {
    if other.origin.as_deref() != Some(NAMESPACE) {
        return None;
    }
    Some(verbatim(other))
}

/// Send an input part verbatim: an Anthropic-originated one, or the
/// caller's own raw part, which has no origin. A part from another provider
/// is never sent.
fn raw_part(other: &OtherContent) -> Option<RequestBlock> {
    if other
        .origin
        .as_deref()
        .is_some_and(|origin| origin != NAMESPACE)
    {
        return None;
    }
    Some(verbatim(other))
}

fn verbatim(other: &OtherContent) -> RequestBlock {
    let mut map = other.data.clone();
    map.insert("type".into(), Value::String(other.type_name.clone()));
    RequestBlock::new(BlockParam::Raw(map))
}

/// Rebuild a server or MCP tool call of Anthropic origin from its portable
/// fields and the wire fields kept in provider data.
fn provider_tool_call_block(call: &ProviderToolCall) -> RequestBlock {
    let mut map = wire_fields(&call.provider_data);
    map.entry("type")
        .or_insert_with(|| Value::String(SERVER_TOOL_USE.into()));
    map.insert("id".into(), Value::String(call.id.clone()));
    map.insert("name".into(), Value::String(call.name.clone()));
    let input = serde_json::from_str::<Value>(&call.arguments)
        .ok()
        .filter(Value::is_object)
        .unwrap_or_else(|| Value::Object(ExtraMap::new()));
    map.insert("input".into(), input);
    RequestBlock::new(BlockParam::Raw(map))
}

/// Rebuild a server or MCP tool result of Anthropic origin as a
/// `result_type` block (`web_search_tool_result`, …).
fn provider_tool_result_block(result: &ProviderToolResult, result_type: &str) -> RequestBlock {
    let mut map = wire_fields(&result.provider_data);
    map.insert("type".into(), Value::String(result_type.to_owned()));
    map.insert("tool_use_id".into(), Value::String(result.call_id.clone()));
    map.insert("content".into(), result.output.clone());
    RequestBlock::new(BlockParam::Raw(map))
}

/// The block type of a server tool call whose type was not recorded.
const SERVER_TOOL_USE: &str = "server_tool_use";

/// Whether a block's `origin` is Anthropic.
fn is_anthropic(origin: &Option<String>) -> bool {
    origin.as_deref() == Some(NAMESPACE)
}

/// The wire fields the decoder kept for a block, in the `anthropic`
/// namespace.
fn wire_fields(data: &ProviderData) -> ExtraMap {
    data.namespace(NAMESPACE).cloned().unwrap_or_default()
}

/// The block type of a server or MCP tool block, as the decoder recorded it.
fn recorded_type(data: &ProviderData) -> Option<&str> {
    data.get_raw(NAMESPACE, "type").and_then(Value::as_str)
}

/// The result block type Anthropic's documentation fixes for `call`, when
/// the result's own type was not recorded (a hand-built or older history).
///
/// Only documented pairings are inferred:
/// - `mcp_tool_use` → `mcp_tool_result`, whatever the tool's name:
///   <https://platform.claude.com/docs/en/agents-and-tools/mcp-connector>;
/// - `server_tool_use` by its `name`, which each server tool fixes:
///   - `web_search` → `web_search_tool_result`:
///     <https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool#response>;
///   - `web_fetch` → `web_fetch_tool_result`:
///     <https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool#response>;
///   - `code_execution` → `code_execution_tool_result` (Python, and
///     programmatic tool calling):
///     <https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling>;
///   - `bash_code_execution` → `bash_code_execution_tool_result` and
///     `text_editor_code_execution` →
///     `text_editor_code_execution_tool_result`:
///     <https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool#response-format>;
///   - `tool_search_tool_regex` and `tool_search_tool_bm25` →
///     `tool_search_tool_result`:
///     <https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool#response-format>;
///   - `advisor` → `advisor_tool_result`:
///     <https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool>.
///
/// Checked against those pages on 2026-10-12.
fn documented_result_type(call: &ProviderToolCall) -> Option<&'static str> {
    match recorded_type(&call.provider_data).unwrap_or(SERVER_TOOL_USE) {
        "mcp_tool_use" => Some("mcp_tool_result"),
        SERVER_TOOL_USE => match call.name.as_str() {
            "web_search" => Some("web_search_tool_result"),
            "web_fetch" => Some("web_fetch_tool_result"),
            "code_execution" => Some("code_execution_tool_result"),
            "bash_code_execution" => Some("bash_code_execution_tool_result"),
            "text_editor_code_execution" => Some("text_editor_code_execution_tool_result"),
            "tool_search_tool_regex" | "tool_search_tool_bm25" => Some("tool_search_tool_result"),
            "advisor" => Some("advisor_tool_result"),
            _ => None,
        },
        _ => None,
    }
}

/// How one tool's `strict` request is honoured.
enum StrictTool {
    /// The caller did not ask for strict.
    NotRequested,
    /// The model has no strict tools: sent without strict, as the caller
    /// wrote it; the message says why.
    Unavailable(String),
    /// What the strict policy is handed: the dialect's outcome, after the
    /// per-request limits.
    Requested(std::result::Result<Rewritten, StrictSchemaError>),
}

/// A schema rewritten by Anthropic's strict dialect, with its
/// `Approximated` warnings held back until the strict policy sends it
/// strict, and what it adds to the per-request limits.
struct Rewritten {
    schema: Value,
    warnings: Vec<Warning>,
    complexity: Complexity,
}

impl Rewritten {
    /// Run the dialect on `schema` for `field`, into a scratch helper, so
    /// the per-request limits can be counted before the policy decides.
    fn new(
        schema: &Value,
        field: RequestField,
        policy: allama::UnsupportedPolicy,
    ) -> std::result::Result<Self, StrictSchemaError> {
        let mut scratch = Degradations::new(PROVIDER_NAME, policy);
        let schema = crate::schema::strict_schema(schema, field, &mut scratch)?;
        Ok(Self {
            complexity: Complexity::of(&schema),
            schema,
            warnings: scratch.into_warnings(),
        })
    }

    /// The dialect to hand [`apply_strict_policy`]: `outcome`, with the
    /// held warnings recorded into the request's helper on success only.
    fn replay(
        outcome: std::result::Result<Self, StrictSchemaError>,
    ) -> impl FnOnce(&Value, &mut Degradations) -> std::result::Result<Value, StrictSchemaError>
    {
        move |_, degradations| {
            outcome.map(|rewritten| {
                degradations.extend(rewritten.warnings);
                rewritten.schema
            })
        }
    }
}

/// `schema` without its `$schema` annotation, for a tool sent without
/// strict.
fn without_schema_uri(schema: &Value) -> Value {
    let mut out = schema.clone();
    if let Value::Object(map) = &mut out {
        map.remove("$schema");
    }
    out
}

/// One encoded turn: a message, or a tool result that joins the user turn
/// of its run.
enum Turn {
    Message(MessageParam),
    ToolResult(RequestBlock),
}

fn flush(out: &mut Vec<MessageParam>, pending: &mut Vec<RequestBlock>) {
    if !pending.is_empty() {
        out.push(MessageParam {
            role: "user",
            content: std::mem::take(pending),
        });
    }
}

fn merge_consecutive_same_role(messages: Vec<MessageParam>) -> Vec<MessageParam> {
    let mut merged: Vec<MessageParam> = Vec::with_capacity(messages.len());
    for message in messages {
        if let Some(last) = merged.last_mut()
            && last.role == message.role
        {
            last.content.extend(message.content);
            continue;
        }
        merged.push(message);
    }
    merged
}

/// Standard (RFC 4648) base64 decoding, tolerant of missing padding and
/// whitespace.
fn decode_base64(input: &str) -> Option<Vec<u8>> {
    fn value(c: u8) -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => return None,
        } as u32)
    }
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    let mut buffer = 0u32;
    let mut bits = 0u32;
    for c in input.bytes() {
        if c == b'=' {
            break;
        }
        if c.is_ascii_whitespace() {
            continue;
        }
        buffer = (buffer << 6) | value(c)?;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buffer >> bits) as u8);
            buffer &= (1 << bits) - 1;
        }
    }
    Some(out)
}

#[cfg(test)]
pub(crate) fn conformance_to_api_request(request: &ChatRequest, stream: bool) -> Result<Value> {
    let encoded = encode_request(request, stream)?;
    serde_json::to_value(encoded.body).map_err(Error::from)
}

#[cfg(test)]
mod tests;
