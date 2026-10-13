//! Maps a portable [`ChatRequest`] to a stateless Responses API request.
//!
//! - System prompts become leading `system` / `developer` input messages
//!   (per [`SystemOptions`]), so several prompts and per-prompt roles map
//!   exactly; `instructions` is not used. Endpoints without the `developer`
//!   role get `system` with a warning.
//! - Assistant turns become `message` items (consecutive text blocks share
//!   one item), `function_call` items, and `reasoning` items replayed from
//!   the dialect namespace (`encrypted_content`, or raw `content` from
//!   endpoints that return it). Reasoning with neither, and state from
//!   other namespaces, is dropped with a warning.
//! - User and tool `Other` parts are sent verbatim when they come from the
//!   dialect namespace or have no origin (the caller's raw part), and are
//!   dropped with a warning when another provider produced them. Answers
//!   to client-executed tool calls (ADR 0014) are input items of their own;
//!   an answer whose call is in the request but not replayed is dropped
//!   with it, and one whose call is not in the request is sent, for
//!   server-side continuation.
//! - Hints the endpoint or model lacks (seed, stop sequences, sampling
//!   parameters on reasoning models, unsupported dialect features) warn or
//!   fail per [`allama::UnsupportedPolicy`]; unsupported semantic fields
//!   fail.

use std::borrow::Cow;

use allama::adapter::Degradations;
use allama::adapter::chat::{EncodedTurn, retain_turns};
use allama::adapter::schema::{Degradable, apply_strict_policy};
use allama::{
    CacheRetention, CapabilitySupport, ChatRequest, ContentBlock, ContentPart, Error, ExtraMap,
    File, Image, MediaSource, Message, OtherContent, ProviderData, Reasoning, ReasoningEffort,
    ReasoningOutput, ReplayState, RequestField, ResponseFormat, Result, Subject, ToolCall,
    ToolChoice, ToolResultContent, UserContent, Warning,
};
use serde_json::{Value, json};

use super::decode::TEXT_SOURCE_KEY;
use super::wire::{
    ContentParam, CreateResponse, FunctionOutput, InputItem, KnownContent, KnownItem,
    ReasoningParam, SummaryText, TextConfig,
};
use crate::client_tools::Pairing;
use crate::{
    ReasoningReplay, ResponsesModelRules, ResponsesQuirks, ResponsesWireOptions, SystemOptions,
    SystemRole,
};
use allama::adapter::describe_origins;

/// Body fields the adapter owns or exposes as typed options.
const RESERVED_EXTRA_FIELDS: &[&str] = &[
    "background",
    "conversation",
    "frequency_penalty",
    "input",
    "instructions",
    "max_output_tokens",
    "metadata",
    "model",
    "parallel_tool_calls",
    "presence_penalty",
    "previous_response_id",
    "prompt_cache_key",
    "prompt_cache_retention",
    "reasoning",
    "safety_identifier",
    "service_tier",
    "store",
    "stream",
    "temperature",
    "text",
    "tool_choice",
    "tools",
    "top_p",
    "truncation",
    "user",
];

/// The fields of a function tool definition the encoder writes: a tool's
/// provider data cannot replace them.
const FUNCTION_TOOL_FIELDS: &[&str] = &["description", "name", "parameters", "strict", "type"];

/// An encoded request plus the warnings produced while encoding it.
#[derive(Debug)]
pub(crate) struct EncodedRequest {
    pub body: CreateResponse,
    pub warnings: Vec<Warning>,
}

/// The endpoint and model a request is encoded for.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Target<'a> {
    /// Provider identity, for warnings and errors.
    pub provider: &'a Cow<'static, str>,
    pub quirks: &'a ResponsesQuirks,
    pub rules: &'a ResponsesModelRules,
}

/// Encode `request` for `POST /responses`.
pub(crate) fn encode_request(
    request: &ChatRequest,
    stream: bool,
    options: &ResponsesWireOptions,
    target: Target<'_>,
) -> Result<EncodedRequest> {
    let mut d = Degradations::new(
        target.provider.clone(),
        request.unsupported_policy.unwrap_or_default(),
    );
    for warning in &options.warnings {
        d.warn(warning.clone());
    }
    let ns = target.quirks.namespace.as_ref();
    let mut encoder = Encoder {
        request,
        provider: target.provider,
        quirks: target.quirks,
        ns,
        rules: target.rules,
        // Client tool calls are replayed only to their origin.
        pairing: Pairing::new(&request.messages, |call| call.origin.as_deref() == Some(ns)),
        d,
    };

    encoder.check_options(options, stream)?;
    encoder.hints()?;
    let reasoning = encoder.reasoning(options)?;
    let effort = reasoning.as_ref().and_then(|r| r.effort.as_deref());
    let sampling_ok = !encoder.rules.rejects_sampling(effort);
    encoder.sampling(sampling_ok)?;
    if encoder.request.top_k.is_some() {
        encoder.d.unsupported_hint(
            RequestField::TopK,
            "the Responses API has no `top_k`; `top_k` was not sent",
        )?;
    }

    let mut input = encoder.system()?;
    encoder.messages(&mut input)?;
    let (tools, strict_tool_sent) = encoder.tools(options)?;
    let tool_choice = encoder.tool_choice(options)?;
    let parallel_tool_calls = encoder.parallel_tool_calls()?;
    let parallel_tool_calls = crate::schema::serial_calls_for_strict(
        parallel_tool_calls,
        encoder.schema_dialect(),
        strict_tool_sent,
        encoder.quirks.supports_parallel_tool_calls,
        effort,
        &mut encoder.d,
    )?;
    let format = encoder.text_format()?;
    let (prompt_cache_key, prompt_cache_retention) = encoder.cache()?;

    let text = (format.is_some() || options.verbosity.is_some()).then(|| TextConfig {
        format,
        verbosity: options.verbosity.clone(),
    });
    let keep = |value: Option<f64>| value.filter(|_| sampling_ok);
    let mut extra = options.extra.clone();
    if !sampling_ok {
        crate::model_rules::drop_logprobs(
            &mut extra,
            &["top_logprobs"],
            &request.model,
            &mut encoder.d,
        )?;
    }
    let body = CreateResponse {
        model: request.model.clone(),
        input,
        store: options.store.unwrap_or(false),
        previous_response_id: options.previous_response_id.clone(),
        conversation: options.conversation.clone(),
        background: options.background,
        stream,
        tools,
        tool_choice,
        parallel_tool_calls,
        max_output_tokens: request.max_output_tokens,
        temperature: keep(request.temperature),
        top_p: keep(request.top_p),
        frequency_penalty: keep(request.frequency_penalty),
        presence_penalty: keep(request.presence_penalty),
        text,
        reasoning,
        prompt_cache_key,
        prompt_cache_retention,
        service_tier: options.service_tier.clone(),
        metadata: options.metadata.clone(),
        safety_identifier: options.safety_identifier.clone(),
        user: options.user.clone(),
        truncation: options.truncation.clone(),
        extra,
    };
    Ok(EncodedRequest {
        body,
        warnings: encoder.d.into_warnings(),
    })
}

struct Encoder<'a> {
    request: &'a ChatRequest,
    provider: &'a str,
    quirks: &'a ResponsesQuirks,
    /// Provider-data namespace: the only state replayed.
    ns: &'a str,
    rules: &'a ResponsesModelRules,
    /// Client tool calls that are not replayed, whose answers go with them.
    pairing: Pairing<'a>,
    d: Degradations,
}

impl Encoder<'_> {
    fn check_options(&mut self, options: &ResponsesWireOptions, stream: bool) -> Result<()> {
        crate::options::refuse_owned_fields(
            &options.extra,
            RESERVED_EXTRA_FIELDS,
            "ResponsesRequestOptions::extra",
            self.provider,
        )?;
        if options.background == Some(true) && options.store != Some(true) {
            return Err(Error::invalid_request(
                "background responses must be stored: set `store: true` alongside `background`",
            )
            .authored()
            .with_provider(self.provider));
        }
        if !self.quirks.streaming.allows(stream) {
            let mode = if stream { "streaming" } else { "non-streaming" };
            return Err(self.d.reject(
                None,
                format!("{mode} requests are not supported by this endpoint"),
            ));
        }
        Ok(())
    }

    fn hints(&mut self) -> Result<()> {
        if self.request.seed.is_some() {
            self.d.unsupported_hint(
                RequestField::Seed,
                "the Responses API has no seed parameter; `seed` was not sent",
            )?;
        }
        if !self.request.stop.is_empty() {
            self.d.unsupported_hint(
                RequestField::Stop,
                "the Responses API has no stop sequences; `stop` was not sent",
            )?;
        }
        Ok(())
    }

    /// Sampling parameters that the model would reject (HTTP 400).
    fn sampling(&mut self, accepted: bool) -> Result<()> {
        if accepted {
            return Ok(());
        }
        let model = &self.request.model;
        let fields = [
            (
                RequestField::Temperature,
                self.request.temperature,
                "temperature",
            ),
            (RequestField::TopP, self.request.top_p, "top_p"),
            (
                RequestField::FrequencyPenalty,
                self.request.frequency_penalty,
                "frequency_penalty",
            ),
            (
                RequestField::PresencePenalty,
                self.request.presence_penalty,
                "presence_penalty",
            ),
        ];
        for (field, value, name) in fields {
            if value.is_some() {
                self.d.unsupported_hint(
                    field,
                    format!("`{model}` does not accept `{name}` with reasoning enabled; `{name}` was not sent"),
                )?;
            }
        }
        Ok(())
    }

    fn reasoning(&mut self, options: &ResponsesWireOptions) -> Result<Option<ReasoningParam>> {
        let config = self.request.reasoning;
        let model = self.request.model.clone();
        let subject = Subject::Field {
            field: RequestField::Reasoning,
        };
        let requested = config.and_then(|c| c.effort);
        let reasons = self.rules.reasoning != CapabilitySupport::Unsupported;

        let mut effort = None;
        if let Some(raw) = &options.reasoning_effort {
            if requested.is_some() {
                self.d.approximated(
                    subject.clone(),
                    "the raw `reasoning_effort` option overrides the portable reasoning effort; the raw value was used instead",
                );
            }
            effort = Some(raw.clone());
        } else if let Some(requested) = requested {
            if !reasons {
                if requested != ReasoningEffort::None {
                    self.d.unsupported_hint(
                        RequestField::Reasoning,
                        format!("`{model}` does not reason; the reasoning effort was not sent"),
                    )?;
                }
            } else {
                let level = self
                    .rules
                    .effort_levels
                    .as_deref()
                    .and_then(|levels| requested.nearest(levels))
                    .unwrap_or(requested);
                if level != requested {
                    self.d.approximated(
                        subject.clone(),
                        format!(
                            "`{model}` has no reasoning effort `{}`; `{}` was used instead",
                            requested.as_str(),
                            level.as_str()
                        ),
                    );
                }
                effort = Some(level.as_str().to_owned());
            }
        }

        let summary = match config.and_then(|c| c.output) {
            None | Some(ReasoningOutput::Omitted) => None,
            Some(_) if !reasons => {
                self.d.unsupported_hint(
                    RequestField::Reasoning,
                    format!("`{model}` does not reason; the reasoning output setting was not sent"),
                )?;
                None
            }
            Some(ReasoningOutput::Summary) => Some("auto"),
            Some(_) => {
                self.d.approximated(
                    subject,
                    "the Responses API cannot request raw reasoning; a detailed reasoning summary was used instead",
                );
                Some("detailed")
            }
        };

        Ok((effort.is_some() || summary.is_some()).then_some(ReasoningParam { effort, summary }))
    }

    fn system(&mut self) -> Result<Vec<InputItem>> {
        let mut items = Vec::new();
        for (index, prompt) in self.request.system.iter().enumerate() {
            let role = match prompt.resolve_option::<SystemOptions>()?.role {
                Some(SystemRole::Developer) if self.quirks.supports_developer_role => "developer",
                Some(SystemRole::Developer) => {
                    if !prompt.content.is_empty() {
                        self.d.hint(
                            Subject::System { index },
                            "this endpoint has no `developer` role; `developer` was not sent, so the prompt went out as `system`",
                        )?;
                    }
                    "system"
                }
                _ => "system",
            };
            if prompt.content.is_empty() {
                continue;
            }
            items.push(message(
                role,
                vec![known(KnownContent::InputText {
                    text: prompt.content.clone(),
                })],
                None,
            ));
        }
        Ok(items)
    }

    fn messages(&mut self, input: &mut Vec<InputItem>) -> Result<()> {
        let mut turns = Vec::with_capacity(self.request.messages.len());
        for (index, msg) in self.request.messages.iter().enumerate() {
            let mut items = Vec::new();
            self.message(index, msg, &mut items)?;
            turns.push((EncodedTurn::new(index, msg, items.is_empty()), items));
        }
        // Skip empty turns, and drop a reply the skips and drops left last.
        for (_, items) in retain_turns(turns, &mut self.d, |(turn, _)| *turn)? {
            input.extend(items);
        }
        Ok(())
    }

    /// The input items of one message.
    fn message(&mut self, index: usize, msg: &Message, items: &mut Vec<InputItem>) -> Result<()> {
        let subject = Subject::Content {
            message: index,
            block: None,
        };
        match msg {
            Message::User(user) => {
                self.ignored_name(&user.provider_data, subject.clone())?;
                self.foreign_data(&user.provider_data, subject);
                match &user.content {
                    UserContent::Text(text) => items.push(message(
                        "user",
                        vec![known(KnownContent::InputText { text: text.clone() })],
                        None,
                    )),
                    UserContent::Parts(parts) => self.user_parts(index, parts, items)?,
                    other => self.user_parts(index, &other.to_parts(), items)?,
                }
            }
            Message::Assistant(assistant) => {
                self.ignored_name(&assistant.provider_data, subject.clone())?;
                self.foreign_data(&assistant.provider_data, subject);
                self.assistant(index, &assistant.content, items)?;
            }
            Message::Tool(tool) => {
                if tool.is_error {
                    // Coded: history repeats the result on every turn.
                    // Never refused: the model still reads the error text,
                    // only the flag is lost (ADR 0003).
                    self.d.coded_approximated(
                        subject.clone(),
                        "the Responses API has no tool error flag; the error text was sent as the tool result; the model still reads the error text; to send your own encoding to this endpoint, use `push_tool_result`",
                        crate::wire::TOOL_RESULT_ERROR,
                    );
                }
                self.foreign_data(&tool.provider_data, subject);
                let output = match &tool.content {
                    ToolResultContent::Text(text) => FunctionOutput::Text(text.clone()),
                    ToolResultContent::Object(object) => {
                        FunctionOutput::Text(Value::Object(object.clone()).to_string())
                    }
                    ToolResultContent::Parts(parts) => {
                        FunctionOutput::Parts(self.parts(index, parts)?)
                    }
                    _ => {
                        return Err(self.d.reject(
                            Subject::Content {
                                message: index,
                                block: None,
                            },
                            "unknown tool result content is not supported by the Responses API",
                        ));
                    }
                };
                items.push(InputItem::Known(KnownItem::FunctionCallOutput {
                    call_id: self
                        .quirks
                        .tool_call_id_format
                        .conform(&tool.tool_call_id)
                        .into_owned(),
                    output,
                }));
            }
            _ => {
                return Err(self.d.reject(
                    subject,
                    "unknown message variant is not supported by the Responses API",
                ));
            }
        }
        Ok(())
    }

    /// A Chat Completions participant name addressed to this namespace
    /// ([`PARTICIPANT_NAME_KEY`](crate::PARTICIPANT_NAME_KEY)), which the
    /// Responses API has no slot for.
    fn ignored_name(&mut self, data: &ProviderData, subject: Subject) -> Result<()> {
        if data.get_raw(self.ns, crate::PARTICIPANT_NAME_KEY).is_some() {
            self.d.hint(
                subject,
                "the Responses API has no participant names; the name was not sent",
            )?;
        }
        Ok(())
    }

    /// Report opaque state from other namespaces, which is never sent.
    fn foreign_data(&mut self, data: &ProviderData, subject: Subject) {
        let foreign = foreign_namespaces(data, self.ns);
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

    /// User message parts. Answers to client-executed tool calls (ADR 0014)
    /// are input items of their own, so they split the message around them.
    /// An answer whose call is in the request but not replayed is dropped
    /// with it; one whose call is not in the request is sent, for
    /// server-side continuation.
    fn user_parts(
        &mut self,
        index: usize,
        parts: &[ContentPart],
        items: &mut Vec<InputItem>,
    ) -> Result<()> {
        let mut content = Vec::new();
        for (block, part) in parts.iter().enumerate() {
            let subject = Subject::Content {
                message: index,
                block: Some(block),
            };
            if let ContentPart::Other(other) = part
                && let Some(item) = crate::client_tools::output_item(other, self.ns)
            {
                if self.pairing.is_unpaired(other) {
                    self.d.dropped(
                        subject,
                        format!(
                            "the call this `{}` answers was not replayed; it was dropped",
                            other.type_name
                        ),
                    );
                    continue;
                }
                if !content.is_empty() {
                    items.push(message("user", std::mem::take(&mut content), None));
                }
                self.foreign_data(&other.provider_data, subject);
                items.push(InputItem::Raw(item));
                continue;
            }
            content.extend(self.part(index, block, part)?);
        }
        if !content.is_empty() {
            items.push(message("user", content, None));
        }
        Ok(())
    }

    fn parts(&mut self, message: usize, parts: &[ContentPart]) -> Result<Vec<ContentParam>> {
        let mut params = Vec::with_capacity(parts.len());
        for (block, part) in parts.iter().enumerate() {
            params.extend(self.part(message, block, part)?);
        }
        Ok(params)
    }

    /// One input part, or `None` for a part from another provider, which is
    /// dropped with a warning.
    fn part(
        &mut self,
        message: usize,
        block: usize,
        part: &ContentPart,
    ) -> Result<Option<ContentParam>> {
        let subject = Subject::Content {
            message,
            block: Some(block),
        };
        Ok(Some(match part {
            ContentPart::Text(text) => {
                self.foreign_data(&text.provider_data, subject);
                known(KnownContent::InputText {
                    text: text.text.clone(),
                })
            }
            ContentPart::Image(image) => {
                self.foreign_data(&image.provider_data, subject.clone());
                known(self.image(subject, image)?)
            }
            ContentPart::File(file) => {
                self.foreign_data(&file.provider_data, subject.clone());
                known(self.file(subject, file)?)
            }
            ContentPart::Other(other) => match raw_part(other, self.ns) {
                Some(raw) => {
                    self.foreign_data(&other.provider_data, subject);
                    ContentParam::Raw(raw)
                }
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
                    return Ok(None);
                }
            },
            _ => {
                return Err(self.d.reject(
                    subject,
                    format!(
                        "content part `{}` is not supported by the Responses API",
                        part.type_name()
                    ),
                ));
            }
        }))
    }

    fn image(&self, subject: Subject, image: &Image) -> Result<KnownContent> {
        let (image_url, file_id) = match &image.source {
            MediaSource::Url { url } => (Some(url.clone()), None),
            MediaSource::Base64 {
                media_type, data, ..
            } => (Some(format!("data:{media_type};base64,{data}")), None),
            MediaSource::ProviderFile { .. } => {
                (None, Some(self.own_file(subject, &image.source)?))
            }
            _ => return Err(self.d.reject(subject, "unknown image source")),
        };
        Ok(KnownContent::InputImage {
            image_url,
            file_id,
            detail: image.detail.map(allama::ImageDetail::as_str),
        })
    }

    /// The annotations of a replayed text block, in the flat Responses
    /// shape (`{"type": "url_citation", "url": …, "start_index": …}`).
    /// Entries in another shape (Chat Completions nests the fields under the
    /// type, `{"type": "url_citation", "url_citation": {…}}`) are dropped
    /// with a warning.
    fn annotations(&mut self, raw: Option<&Value>, subject: Subject) -> Value {
        let Some(Value::Array(entries)) = raw else {
            return json!([]);
        };
        let (flat, nested): (Vec<&Value>, Vec<&Value>) = entries.iter().partition(|entry| {
            let kind = entry.get("type").and_then(Value::as_str);
            entry.is_object() && kind.is_none_or(|kind| entry.get(kind).is_none())
        });
        if !nested.is_empty() {
            self.d.dropped(
                subject,
                "text annotations in the Chat Completions shape cannot be replayed to the \
                 Responses API; they were dropped",
            );
        }
        Value::Array(flat.into_iter().cloned().collect())
    }

    /// The id of a file this endpoint stores. `prepare` refuses another
    /// provider's file before encoding; this refuses it again for a caller
    /// that encodes without it.
    fn own_file(&self, subject: Subject, source: &MediaSource) -> Result<String> {
        match source {
            MediaSource::ProviderFile { origin, id, .. } if origin == self.ns => Ok(id.clone()),
            MediaSource::ProviderFile { origin, .. } => Err(self.d.reject(
                subject,
                format!("a file stored by `{origin}` can be sent only to `{origin}`"),
            )),
            _ => Err(self.d.reject(subject, "not a stored file")),
        }
    }

    fn file(&self, subject: Subject, file: &File) -> Result<KnownContent> {
        let media_type = file.media_type.as_str();
        if media_type.starts_with("audio/") {
            return Err(self.d.reject(
                subject,
                format!(
                    "the Responses API does not accept audio input (`{media_type}`); use a Chat \
                     Completions provider"
                ),
            ));
        }
        match &file.source {
            MediaSource::ProviderFile { .. } => Ok(KnownContent::InputFile {
                file_id: Some(self.own_file(subject, &file.source)?),
                filename: None,
                file_data: None,
                file_url: None,
            }),
            MediaSource::Base64 { data, .. } => Ok(KnownContent::InputFile {
                file_id: None,
                filename: Some(
                    file.filename
                        .clone()
                        .unwrap_or_else(|| default_filename(media_type)),
                ),
                file_data: Some(format!("data:{media_type};base64,{data}")),
                file_url: None,
            }),
            MediaSource::Url { url } => Ok(KnownContent::InputFile {
                file_id: None,
                filename: file.filename.clone(),
                file_data: None,
                file_url: Some(url.clone()),
            }),
            _ => Err(self.d.reject(subject, "unknown file source")),
        }
    }

    fn assistant(
        &mut self,
        index: usize,
        content: &[ContentBlock],
        items: &mut Vec<InputItem>,
    ) -> Result<()> {
        let mut pending = PendingMessage::default();
        // Function calls held back to the end of the turn for endpoints that
        // reject them interleaved with other items.
        let mut deferred_calls = Vec::new();
        // Provider tool calls replayed so far; their results are part of the
        // same item.
        let mut replayed_calls: Vec<&str> = Vec::new();
        for (block, item) in content.iter().enumerate() {
            let subject = Subject::Content {
                message: index,
                block: Some(block),
            };
            match item {
                ContentBlock::Refusal(refusal) => {
                    self.foreign_data(&refusal.provider_data, subject);
                    let data = refusal.provider_data.namespace(self.ns);
                    pending.phase(data.and_then(|d| d.get("phase")).cloned(), items);
                    pending.content.push(known(KnownContent::Refusal {
                        refusal: refusal.text.clone(),
                    }));
                }
                ContentBlock::Text(text) => {
                    self.foreign_data(&text.provider_data, subject.clone());
                    let data = text.provider_data.namespace(self.ns);
                    pending.phase(data.and_then(|d| d.get("phase")).cloned(), items);
                    // A refusal an earlier version stored as flagged text.
                    let refusal = data
                        .and_then(|d| d.get("refusal"))
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    pending.content.push(known(if refusal {
                        KnownContent::Refusal {
                            refusal: text.text.clone(),
                        }
                    } else {
                        KnownContent::OutputText {
                            text: text.text.clone(),
                            annotations: self.annotations(
                                data.and_then(|d| d.get("annotations")),
                                subject,
                            ),
                        }
                    }));
                }
                ContentBlock::ToolCall(call) => {
                    self.foreign_data(&call.provider_data, subject);
                    let item = function_call(call, self.ns, self.quirks);
                    if self.quirks.interleaved_function_calls {
                        pending.flush(items);
                        items.push(item);
                    } else {
                        deferred_calls.push(item);
                    }
                }
                ContentBlock::Reasoning(reasoning) => {
                    pending.flush(items);
                    if let Some(item) = self.reasoning_item(subject, reasoning) {
                        items.push(item);
                    }
                }
                ContentBlock::Image(_) => {
                    return Err(self.d.reject(
                        subject,
                        "the Responses API cannot replay images in assistant messages",
                    ));
                }
                ContentBlock::ProviderToolCall(call) => {
                    // The call and its result are one output item.
                    let result = content.iter().find_map(|b| match b {
                        ContentBlock::ProviderToolResult(r) if r.call_id == call.id => Some(r),
                        _ => None,
                    });
                    match super::hosted::encode(call, result, self.ns) {
                        Some(raw) => {
                            pending.flush(items);
                            items.push(InputItem::Raw(raw));
                            replayed_calls.push(call.id.as_str());
                        }
                        None if call.origin.as_deref() != Some(self.ns) => {
                            self.d.dropped_replay_state(
                                ReplayState::ForeignContent,
                                subject,
                                format!(
                                    "provider tool call `{}` from {} can be replayed only to its origin; it was dropped",
                                    call.name,
                                    describe_origins(call.origin.as_slice())
                                ),
                            );
                        }
                        None => self.d.dropped_replay_state(
                            ReplayState::Content,
                            subject,
                            format!(
                                "the Responses API cannot replay provider tool call `{}`; it was dropped",
                                call.name
                            ),
                        ),
                    }
                }
                ContentBlock::ProviderToolResult(result)
                    if result.origin.as_deref() == Some(self.ns)
                        && replayed_calls.contains(&result.call_id.as_str()) => {}
                ContentBlock::ProviderToolResult(result)
                    if result.origin.as_deref() != Some(self.ns) =>
                {
                    self.d.dropped_replay_state(
                        ReplayState::ForeignContent,
                        subject,
                        format!(
                            "provider tool result from {} can be replayed only to its origin; it was dropped",
                            describe_origins(result.origin.as_slice())
                        ),
                    );
                }
                ContentBlock::ProviderToolResult(_) => self.d.dropped_replay_state(
                    ReplayState::Content,
                    subject,
                    "the call of this provider tool result was not replayed; it was dropped",
                ),
                ContentBlock::Other(other) => match raw(other, self.ns) {
                    Some(raw) => {
                        self.foreign_data(&other.provider_data, subject);
                        pending.flush(items);
                        items.push(InputItem::Raw(raw));
                    }
                    None => self.d.dropped_replay_state(
                        ReplayState::ForeignContent,
                        subject,
                        format!(
                            "`{}` content from {} can be replayed only to its origin; it was dropped",
                            other.type_name,
                            describe_origins(other.origin.as_slice())
                        ),
                    ),
                },
                _ => self.d.dropped(
                    subject,
                    format!(
                        "the Responses API has no `{}` content; it was dropped",
                        item.type_name()
                    ),
                ),
            }
        }
        pending.flush(items);
        items.append(&mut deferred_calls);
        Ok(())
    }

    /// Replay a reasoning item from this namespace's data: its encrypted
    /// content (OpenAI), its raw reasoning content (vLLM, llama.cpp), or
    /// both. Anything else cannot be replayed statelessly.
    fn reasoning_item(&mut self, subject: Subject, reasoning: &Reasoning) -> Option<InputItem> {
        if self.quirks.reasoning_replay == ReasoningReplay::Disabled {
            self.d.dropped_replay_state(
                ReplayState::Reasoning,
                subject,
                "this endpoint takes no reasoning back; it was dropped",
            );
            return None;
        }
        let own = reasoning.provider_data.namespace(self.ns);
        let field = |key: &str| own.and_then(|d| d.get(key));
        let encrypted = field("encrypted_content")
            .and_then(Value::as_str)
            .filter(|e| !e.is_empty());
        let content = field("content").filter(|c| c.as_array().is_some_and(|c| !c.is_empty()));
        if encrypted.is_none() && content.is_none() {
            let foreign = foreign_namespaces(&reasoning.provider_data, self.ns);
            let (state, message) = if foreign.is_empty() {
                (
                    ReplayState::Reasoning,
                    "reasoning without encrypted or raw content cannot be replayed statelessly; it was dropped".to_owned(),
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
            return None;
        }
        self.foreign_data(&reasoning.provider_data, subject);
        // Text decoded from raw content is replayed as `content`, not as a
        // summary.
        let from_content = field(TEXT_SOURCE_KEY).and_then(Value::as_str) == Some("content");
        let summary = if reasoning.text.is_empty() || from_content {
            Vec::new()
        } else {
            vec![SummaryText::new(reasoning.text.clone())]
        };
        Some(InputItem::Known(KnownItem::Reasoning {
            id: field("item_id").and_then(Value::as_str).map(str::to_owned),
            encrypted_content: encrypted.map(str::to_owned),
            summary,
            content: content.cloned(),
        }))
    }

    /// The tool definitions, and whether any went out strict.
    fn tools(&mut self, options: &ResponsesWireOptions) -> Result<(Vec<Value>, bool)> {
        let mut tools = Vec::new();
        let mut strict_tool_sent = false;
        for tool in &self.request.tools {
            self.foreign_data(
                &tool.provider_data,
                Subject::Field {
                    field: RequestField::Tools,
                },
            );
            let requested_strict = tool.strict == Some(true);
            if requested_strict && !self.quirks.supports_strict {
                self.d.unsupported_hint(
                    RequestField::Tools,
                    format!(
                        "strict tools are not supported by this endpoint; `{}` was sent without `strict`",
                        tool.name
                    ),
                )?;
            }
            let (parameters, strict) = if requested_strict && self.quirks.supports_strict {
                let subject = format!("tool `{}`", tool.name);
                let dialect = self.schema_dialect();
                let outcome = apply_strict_policy(
                    &tool.parameters,
                    RequestField::Tools,
                    &subject,
                    &mut self.d,
                    crate::schema::dialect_pass(
                        dialect,
                        self.quirks.dialect_verified,
                        RequestField::Tools,
                        &subject,
                    ),
                    Degradable::Yes,
                )?;
                (outcome.schema, outcome.strict)
            } else {
                (tool.parameters.clone(), false)
            };
            if self.quirks.object_schema_root {
                crate::schema::require_object_root(
                    &parameters,
                    RequestField::Tools,
                    &format!("tool `{}`", tool.name),
                    &self.d,
                )?;
            }
            strict_tool_sent |= strict;
            let mut definition = ExtraMap::new();
            definition.insert("type".into(), "function".into());
            definition.insert("name".into(), tool.name.clone().into());
            if let Some(description) = &tool.description {
                definition.insert("description".into(), description.clone().into());
            }
            definition.insert("parameters".into(), parameters);
            // Explicit: the Responses API strictifies tools that omit
            // `strict` (verified live on OpenAI), unlike other providers.
            if self.quirks.supports_strict {
                definition.insert("strict".into(), strict.into());
            }
            if let Some(extra) = tool.provider_data.namespace(self.ns) {
                // The definition's own fields (and the strict
                // policy's outcome) are the adapter's; provider data adds
                // fields, never replaces them.
                if let Some(key) = extra.keys().find(|key| {
                    FUNCTION_TOOL_FIELDS
                        .iter()
                        .any(|field| key.eq_ignore_ascii_case(field))
                }) {
                    return Err(Error::invalid_request(format!(
                        "the `{}` provider data of tool `{}` cannot set `{key}`: the adapter owns it",
                        self.ns, tool.name
                    ))
                    .authored()
                    .with_provider(self.provider));
                }
                definition.extend(extra.clone());
            }
            tools.push(Value::Object(definition));
        }
        tools.extend(options.hosted_tools.iter().cloned().map(Value::Object));
        tools.extend(options.client_tools.iter().cloned().map(Value::Object));
        Ok((tools, strict_tool_sent))
    }

    /// The strict-schema dialect for this request's model.
    fn schema_dialect(&self) -> Option<crate::schema::SchemaDialect> {
        self.quirks.schema_dialect_for(&self.request.model)
    }

    fn tool_choice(&mut self, options: &ResponsesWireOptions) -> Result<Option<Value>> {
        if let Some(raw) = &options.tool_choice {
            if self.request.tool_choice.is_some() {
                self.d.approximated(
                    Subject::Field {
                        field: RequestField::ToolChoice,
                    },
                    "the raw `tool_choice` option overrides the portable tool choice; the raw value was used instead",
                );
            }
            return Ok(Some(raw.clone()));
        }
        let choice = match &self.request.tool_choice {
            None => return Ok(None),
            Some(choice) => choice,
        };
        if !self.quirks.supports_tool_choice {
            // `auto` is the default behavior; omitting it changes nothing.
            if matches!(choice, ToolChoice::Auto) {
                return Ok(None);
            }
            return Err(self.d.unsupported(
                RequestField::ToolChoice,
                "this endpoint does not accept `tool_choice`",
            ));
        }
        Ok(Some(match choice {
            ToolChoice::Auto => "auto".into(),
            ToolChoice::Disabled => "none".into(),
            ToolChoice::Required => "required".into(),
            ToolChoice::Specific { name } => json!({"type": "function", "name": name}),
            _ => {
                return Err(self
                    .d
                    .unsupported(RequestField::ToolChoice, "unknown tool choice"));
            }
        }))
    }

    fn parallel_tool_calls(&mut self) -> Result<Option<bool>> {
        crate::schema::gate_parallel_tool_calls(
            self.request.parallel_tool_calls,
            self.quirks.supports_parallel_tool_calls,
            &mut self.d,
        )
    }

    fn text_format(&mut self) -> Result<Option<Value>> {
        Ok(match &self.request.response_format {
            None => None,
            Some(ResponseFormat::Text) => Some(json!({"type": "text"})),
            Some(ResponseFormat::Json) => Some(json!({"type": "json_object"})),
            Some(ResponseFormat::JsonSchema {
                name,
                schema,
                strict,
                ..
            }) => {
                if !self.quirks.supports_json_schema {
                    return Err(self.d.unsupported(
                        RequestField::ResponseFormat,
                        "JSON schema response formats are not supported by this endpoint",
                    ));
                }
                let requested_strict = *strict == Some(true);
                if requested_strict && !self.quirks.supports_strict {
                    self.d.unsupported_hint(
                        RequestField::ResponseFormat,
                        "strict JSON schemas are not supported by this endpoint; the schema was sent without `strict`",
                    )?;
                }
                let (schema, strict) = if requested_strict && self.quirks.supports_strict {
                    let subject = "the response schema";
                    let dialect = self.schema_dialect();
                    let outcome = apply_strict_policy(
                        schema,
                        RequestField::ResponseFormat,
                        subject,
                        &mut self.d,
                        crate::schema::dialect_pass(
                            dialect,
                            self.quirks.dialect_verified,
                            RequestField::ResponseFormat,
                            subject,
                        ),
                        Degradable::Yes,
                    )?;
                    (outcome.schema, outcome.strict)
                } else {
                    (schema.clone(), false)
                };
                if self.quirks.object_schema_root {
                    crate::schema::require_object_root(
                        &schema,
                        RequestField::ResponseFormat,
                        "the response schema",
                        &self.d,
                    )?;
                }
                let mut format = ExtraMap::new();
                format.insert("type".into(), "json_schema".into());
                format.insert(
                    "name".into(),
                    name.clone().unwrap_or_else(|| "response".into()).into(),
                );
                format.insert("schema".into(), schema);
                if self.quirks.supports_strict {
                    format.insert("strict".into(), strict.into());
                }
                Some(Value::Object(format))
            }
            Some(_) => {
                return Err(self
                    .d
                    .unsupported(RequestField::ResponseFormat, "unknown response format"));
            }
        })
    }

    /// Prompt-cache key and retention. Breakpoints need no mapping (OpenAI
    /// caches prompt prefixes automatically), but a breakpoint asking for
    /// long retention is honored like a request-level retention.
    fn cache(&mut self) -> Result<(Option<String>, Option<&'static str>)> {
        let key = self.request.cache.as_ref().and_then(|c| c.key.clone());
        let retention = self
            .request
            .cache
            .as_ref()
            .and_then(|c| c.retention)
            .or_else(|| {
                self.request
                    .cache_breakpoints()
                    .any(|b| b.retention == Some(CacheRetention::Long))
                    .then_some(CacheRetention::Long)
            })
            .map(|r| match r {
                CacheRetention::Long => "24h",
                _ => "in_memory",
            });
        if self.quirks.supports_prompt_cache_key {
            return Ok((key, retention));
        }
        if key.is_some() || retention.is_some() {
            self.d.unsupported_hint(
                RequestField::Cache,
                "this endpoint takes no prompt cache key or retention; they were not sent",
            )?;
        }
        Ok((None, None))
    }
}

/// Text blocks of one assistant turn, merged into one `message` item until
/// a non-text block (or a different `phase`) interrupts them.
#[derive(Default)]
struct PendingMessage {
    content: Vec<ContentParam>,
    phase: Option<Value>,
}

impl PendingMessage {
    /// Continue with content of `phase`: a change of phase starts a new
    /// message.
    fn phase(&mut self, phase: Option<Value>, items: &mut Vec<InputItem>) {
        if self.phase != phase {
            self.flush(items);
            self.phase = phase;
        }
    }

    fn flush(&mut self, items: &mut Vec<InputItem>) {
        if !self.content.is_empty() {
            items.push(message(
                "assistant",
                std::mem::take(&mut self.content),
                self.phase.clone(),
            ));
        }
    }
}

fn message(role: &'static str, content: Vec<ContentParam>, phase: Option<Value>) -> InputItem {
    InputItem::Known(KnownItem::Message {
        role,
        content,
        phase,
    })
}

fn known(content: KnownContent) -> ContentParam {
    ContentParam::Known(content)
}

fn function_call(call: &ToolCall, namespace: &str, quirks: &ResponsesQuirks) -> InputItem {
    InputItem::Known(KnownItem::FunctionCall {
        id: call
            .provider_data
            .get_raw(namespace, "item_id")
            .and_then(Value::as_str)
            .map(str::to_owned),
        call_id: quirks.tool_call_id_format.conform(&call.id).into_owned(),
        name: call.name.clone(),
        // Verbatim: the API accepts any string (verified live), so malformed
        // arguments are replayed as the model produced them.
        arguments: call.arguments.clone(),
        status: quirks.function_call_status_null.then_some(Value::Null),
    })
}

/// Replay an item that originated in `namespace` verbatim.
fn raw(other: &OtherContent, namespace: &str) -> Option<ExtraMap> {
    if other.origin.as_deref() != Some(namespace) {
        return None;
    }
    Some(verbatim(other))
}

/// Send an input part verbatim: one that originated in `namespace`, or the
/// caller's own raw part, which has no origin. A part from another provider
/// is never sent.
fn raw_part(other: &OtherContent, namespace: &str) -> Option<ExtraMap> {
    if other
        .origin
        .as_deref()
        .is_some_and(|origin| origin != namespace)
    {
        return None;
    }
    Some(verbatim(other))
}

fn verbatim(other: &OtherContent) -> ExtraMap {
    let mut map = other.data.clone();
    map.insert("type".into(), Value::String(other.type_name.clone()));
    map
}

fn foreign_namespaces<'a>(data: &'a ProviderData, namespace: &'a str) -> Vec<&'a str> {
    data.foreign_namespaces(namespace).collect()
}

/// `input_file` requires a filename with inline data (verified live).
fn default_filename(media_type: &str) -> String {
    let extension = match media_type {
        "application/pdf" => "pdf",
        "text/plain" => "txt",
        "text/markdown" => "md",
        "text/csv" => "csv",
        "text/html" => "html",
        "application/json" => "json",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => "docx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => "xlsx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation" => "pptx",
        _ => return "file".to_owned(),
    };
    format!("document.{extension}")
}

#[cfg(test)]
mod tests;
