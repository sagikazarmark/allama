//! Mapping of a portable [`ChatRequest`] to a Gemini `generateContent` body.
//!
//! Every portable feature is either mapped, approximated with a
//! [`Warning`], or rejected; nothing is dropped silently. Notable decisions:
//!
//! - **System prompts.** Gemini has one `systemInstruction`; each prompt is
//!   one of its parts, in order.
//! - **Image detail** is the part's `mediaResolution`: `Low` →
//!   `MEDIA_RESOLUTION_LOW`, `High` → `MEDIA_RESOLUTION_HIGH`, `Auto` leaves
//!   it to the model (verified live 2026-10-07).
//! - **Provider files.** A `google-ai` provider file (`files/<id>`, the name
//!   the Files API returns) is sent as `fileData` with its Files API URI.
//! - **Tools** send the caller's JSON Schema unchanged through
//!   `parametersJsonSchema` (`$defs`, `$ref`, `additionalProperties`, and
//!   recursion are accepted, verified live 2026-10-02).
//! - **Structured output** sends the schema through `responseJsonSchema`,
//!   after two rewrites that keep its meaning: a `$ref` with sibling keywords
//!   is merged with its target (the API reference forbids the siblings), and
//!   a string or number `const` becomes a one-value `enum`. Constructs
//!   outside the documented `responseJsonSchema` subset are still sent, and
//!   named in one `Approximated` warning on the response format, since
//!   Gemini has no strict mode that would reject them. Extraction then
//!   reports `SchemaEnforcement::Approximate`.
//! - **Thinking.** Gemini 3 uses `thinkingLevel`, Gemini 2.5 (and the
//!   older generations) `thinkingBudget`; see [`thinking_config`]. The two
//!   are never sent together.
//! - **Opaque state.** Only `"google-ai"` provider data is replayed.
//!   `thoughtSignature`s are re-attached to the part they arrived with. On
//!   Gemini 3, and on a model the adapter does not know (the newest
//!   behavior), the first `functionCall` of a model turn must carry a
//!   signature: when it has none (history from another provider or built by
//!   hand) the documented dummy `skip_thought_signature_validator` is sent
//!   with an `Approximated` warning.
//! - **Penalties** are sent as `presencePenalty` and `frequencyPenalty`. A
//!   model that does not enable them answers "Penalty is not enabled for
//!   this model", which maps to [`Error::Unsupported`] at the `Provider`
//!   stage.
//! - **`parallel_tool_calls`**: Gemini may always return parallel calls and
//!   has no switch, so `false` cannot be honored (hint warning); `true` is
//!   the native behavior.
//!
//! Nothing is refused or dropped because of the model id (ADR 0015): the
//! [`ModelProfile`] the provider passes in picks wire formats only.

use allama::adapter::chat::{EncodedTurn, retain_turns};
use allama::adapter::{Degradations, describe_origins};
use allama::{
    CacheRetention, ChatRequest, ContentBlock, ContentPart, Error, Image, ImageDetail, MediaSource,
    Message, OtherContent, ProviderData, ReasoningConfig, ReasoningEffort, ReasoningOutput,
    ReplayState, RequestField, ResponseFormat, Result, Subject, SystemPrompt, ToolChoice,
    ToolDefinition, ToolMessage, ToolResultContent, UserContent, Warning,
};
use serde_json::{Map, Value, json};

use crate::decode::{
    CODE_EXECUTION_RESULT, CODE_EXECUTION_TOOL, EXECUTABLE_CODE, THOUGHT_KEY,
    THOUGHT_SIGNATURE_KEY, parse_synthetic_tool_call_id,
};
use crate::model::{ModelFamily, ModelProfile};
use crate::options::refuse_owned_fields;
use crate::wire::{
    Blob, Content, FileData, FunctionCall, FunctionCallingConfig, FunctionDeclaration,
    FunctionResponse, FunctionResponsePart, GENERATION_CONFIG_FIELDS, GeminiTool,
    GenerateContentRequest, GenerationConfig, MediaResolution, Part, PartData, RESERVED_FIELDS,
    SystemInstruction, ThinkingConfig, ToolConfig,
};
use crate::{ChatRequestOptions, NAMESPACE, PROVIDER_NAME};

/// Google's documented placeholder signature for function calls that Gemini
/// did not generate (injected or foreign history).
pub(crate) const DUMMY_THOUGHT_SIGNATURE: &str = "skip_thought_signature_validator";

/// The wire field of a part's signature, which an `Other` part in wire form
/// keeps inside the verbatim part data.
const WIRE_THOUGHT_SIGNATURE: &str = "thoughtSignature";

/// A mapped request plus the warnings produced while mapping it.
pub(crate) struct EncodedRequest {
    pub body: GenerateContentRequest,
    pub warnings: Vec<Warning>,
}

/// Map a portable request to the Gemini wire format, with the wire formats
/// `profile` picks for its model.
pub(crate) fn encode_request(
    request: &ChatRequest,
    profile: ModelProfile,
) -> Result<EncodedRequest> {
    let options = request.resolve_option::<ChatRequestOptions>()?;
    refuse_owned_fields(&options.extra, RESERVED_FIELDS, "ChatRequestOptions::extra")?;
    refuse_owned_fields(
        &options.generation_config_extra,
        GENERATION_CONFIG_FIELDS,
        "ChatRequestOptions::generation_config_extra",
    )?;
    let family = profile.family;
    let mut d = Degradations::new(
        PROVIDER_NAME,
        request.unsupported_policy.unwrap_or_default(),
    );
    crate::options::report_options_under_name(&request.options, "ChatRequestOptions", &mut d);

    let system_instruction = encode_system(&request.system, &mut d)?;
    let contents = encode_messages(&request.messages, family, &mut d)?;
    let mut tools = encode_tools(&request.tools, &mut d)?;
    if !options.builtin_tools.is_empty() {
        tools.get_or_insert_with(Vec::new).extend(
            options
                .builtin_tools
                .iter()
                .cloned()
                .map(GeminiTool::Builtin),
        );
    }
    let tool_config = request
        .tool_choice
        .as_ref()
        .map(|choice| encode_tool_choice(choice, &d))
        .transpose()?;
    let generation_config = encode_generation_config(request, &options, family, &mut d)?;

    if request.parallel_tool_calls == Some(false) {
        d.unsupported_hint(
            RequestField::ParallelToolCalls,
            "Gemini has no `parallel_tool_calls` field; `false` was not sent, so the model may still call several tools at once",
        )?;
    }
    if let Some(cache) = &request.cache
        && (cache.key.is_some() || cache.retention.is_some())
    {
        d.unsupported_hint(
            RequestField::Cache,
            "Gemini implicit caching takes no cache key or retention; they were not sent; use \
             `ChatRequestOptions::cached_content` for explicit caches",
        )?;
    }
    // Plain breakpoints are satisfied by implicit prefix caching; a long
    // retention is not, just like a request-level one.
    if request
        .cache_breakpoints()
        .any(|breakpoint| breakpoint.retention == Some(CacheRetention::Long))
    {
        d.unsupported_hint(
            RequestField::Cache,
            "Gemini implicit caching has no retention control; the long retention of a \
             cache breakpoint was not sent; use `ChatRequestOptions::cached_content` for \
             explicit caches",
        )?;
    }

    Ok(EncodedRequest {
        body: GenerateContentRequest {
            contents,
            system_instruction,
            tools,
            tool_config,
            generation_config: (!generation_config.is_empty()).then_some(generation_config),
            safety_settings: options.safety_settings,
            cached_content: options.cached_content,
            extra: options.extra,
        },
        warnings: d.into_warnings(),
    })
}

fn encode_system(
    system: &[SystemPrompt],
    d: &mut Degradations,
) -> Result<Option<SystemInstruction>> {
    for (index, prompt) in system.iter().enumerate() {
        if prompt.options.json(NAMESPACE).is_some()
            || prompt.options.contains::<ChatRequestOptions>()
        {
            d.hint(
                Subject::System { index },
                "Gemini has no per-system-prompt options; they were not sent",
            )?;
        }
        // Cache breakpoints are satisfied by Gemini's implicit prefix caching
        // (long retention is reported in `encode_request`).
    }
    // Each prompt is one part of the system instruction, so their
    // boundaries survive.
    let parts: Vec<Part> = system
        .iter()
        .map(|prompt| prompt.content.as_str())
        .filter(|content| !content.is_empty())
        .map(Part::text)
        .collect();
    Ok((!parts.is_empty()).then_some(SystemInstruction { parts }))
}

fn encode_messages(
    messages: &[Message],
    family: ModelFamily,
    d: &mut Degradations,
) -> Result<Vec<Content>> {
    let mut turns: Vec<(EncodedTurn, &'static str, Vec<Part>)> = Vec::with_capacity(messages.len());
    let mut index = 0;
    while index < messages.len() {
        if matches!(messages[index], Message::Tool(_)) {
            let start = index;
            while index < messages.len() && matches!(messages[index], Message::Tool(_)) {
                index += 1;
            }
            for i in tool_result_order(messages, start, index) {
                let Message::Tool(message) = &messages[i] else {
                    unreachable!("run contains only tool messages");
                };
                let part = encode_tool_result(i, message, messages, d)?;
                turns.push((EncodedTurn::tool(i), "user", vec![part]));
            }
            continue;
        }

        let i = index;
        index += 1;
        match &messages[i] {
            Message::User(message) => {
                warn_foreign_data(&message.provider_data, content_subject(i, None), d);
                let parts = encode_user_content(i, &message.content, d)?;
                let turn = EncodedTurn::new(i, &messages[i], parts.is_empty());
                turns.push((turn, "user", parts));
            }
            Message::Assistant(message) => {
                warn_foreign_data(&message.provider_data, content_subject(i, None), d);
                let parts = encode_assistant_content(i, &message.content, d)?;
                let turn = EncodedTurn::new(i, &messages[i], parts.is_empty());
                turns.push((turn, "model", parts));
            }
            Message::Tool(_) => unreachable!("tool messages are handled as runs"),
            _ => {
                return Err(d.reject(
                    content_subject(i, None),
                    format!(
                        "message role `{}` is not supported by the Gemini adapter",
                        messages[i].role()
                    ),
                ));
            }
        }
    }

    // Skip empty turns, and drop a reply the skips and drops left last.
    let mut contents: Vec<Content> = Vec::new();
    for (_, role, parts) in retain_turns(turns, d, |(turn, _, _)| *turn)? {
        push_content(&mut contents, role, parts);
    }

    if family.validates_call_signatures() {
        add_dummy_signatures(&mut contents, d);
    }
    Ok(contents)
}

/// Gemini 3 validates the signature of the first `functionCall` in each model
/// turn, and so does an unknown id (the newest behavior). Calls without a
/// Gemini signature get the documented dummy.
fn add_dummy_signatures(contents: &mut [Content], d: &mut Degradations) {
    for content in contents.iter_mut().filter(|c| c.role == "model") {
        let Some(part) = content
            .parts
            .iter_mut()
            .find(|part| matches!(part.data, PartData::FunctionCall(_)))
        else {
            continue;
        };
        if part.thought_signature.is_some() {
            continue;
        }
        part.thought_signature = Some(DUMMY_THOUGHT_SIGNATURE.to_owned());
        let (message, block) = part.source.unwrap_or_default();
        // Coded and joined: history repeats the call on every turn, so one
        // warning lists every message (ADR 0003).
        d.coded_approximated(
            content_subject(message, Some(block)),
            "the function call has no Gemini thought signature; Google's documented \
             placeholder `skip_thought_signature_validator` was sent instead",
            "approximated.thought_signature",
        );
    }
}

/// Merge consecutive same-role contents: Gemini expects alternating turns.
fn push_content(contents: &mut Vec<Content>, role: &'static str, parts: Vec<Part>) {
    match contents.last_mut() {
        Some(last) if last.role == role => last.parts.extend(parts),
        _ => contents.push(Content { role, parts }),
    }
}

/// Order a run of tool results. Results answering synthetic ids from one
/// Gemini response are sorted by call index (Gemini 2.5 matches results to
/// id-less calls by position); otherwise the caller's order is kept.
fn tool_result_order(messages: &[Message], start: usize, end: usize) -> Vec<usize> {
    let mut keyed = Vec::with_capacity(end - start);
    for (i, message) in messages.iter().enumerate().take(end).skip(start) {
        let Some(key) = message
            .as_tool()
            .and_then(|m| parse_synthetic_tool_call_id(&m.tool_call_id))
        else {
            return (start..end).collect();
        };
        keyed.push((i, key));
    }
    let scope = keyed.first().map(|(_, (scope, _))| *scope);
    if keyed.iter().any(|(_, (s, _))| Some(*s) != scope) {
        return (start..end).collect();
    }
    keyed.sort_by_key(|(_, (_, n))| *n);
    keyed.into_iter().map(|(i, _)| i).collect()
}

fn encode_user_content(
    message: usize,
    content: &UserContent,
    d: &mut Degradations,
) -> Result<Vec<Part>> {
    let owned;
    let parts = match content {
        UserContent::Text(text) if text.is_empty() => return Ok(Vec::new()),
        UserContent::Text(text) => return Ok(vec![Part::text(text.clone())]),
        UserContent::Parts(parts) => parts,
        other => {
            owned = other.to_parts();
            &owned
        }
    };
    let mut out = Vec::with_capacity(parts.len());
    for (block, part) in parts.iter().enumerate() {
        let subject = content_subject(message, Some(block));
        let encoded = match part {
            ContentPart::Text(text) if text.text.is_empty() => continue,
            ContentPart::Text(text) => {
                warn_foreign_data(&text.provider_data, subject, d);
                Part::text(text.text.clone())
            }
            ContentPart::Image(image) => {
                warn_foreign_data(&image.provider_data, subject.clone(), d);
                let mut part = Part::new(image_part(image, subject.clone(), d)?)
                    .signature(own_signature(&image.provider_data));
                part.media_resolution = media_resolution(image, subject, d)?;
                part
            }
            ContentPart::File(file) => {
                if file.filename.is_some() {
                    d.dropped(
                        subject.clone(),
                        "Gemini file parts carry no filename; the filename was dropped",
                    );
                }
                warn_foreign_data(&file.provider_data, subject.clone(), d);
                Part::new(file_part(&file.media_type, &file.source, subject, d)?)
            }
            // Gemini's own part, or the caller's raw part (no origin).
            ContentPart::Other(other)
                if other
                    .origin
                    .as_deref()
                    .is_none_or(|origin| origin == NAMESPACE) =>
            {
                warn_foreign_data(&other.provider_data, subject, d);
                other_part(other)
            }
            ContentPart::Other(other) => {
                d.dropped_replay_state(
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
            other => {
                return Err(d.reject(
                    subject,
                    format!(
                        "content part `{}` is not supported by the Gemini adapter",
                        other.type_name()
                    ),
                ));
            }
        };
        out.push(encoded.source(message, block));
    }
    Ok(out)
}

fn encode_assistant_content(
    message: usize,
    content: &[ContentBlock],
    d: &mut Degradations,
) -> Result<Vec<Part>> {
    let mut out = Vec::with_capacity(content.len());
    for (block, item) in content.iter().enumerate() {
        let subject = content_subject(message, Some(block));
        let part = match item {
            ContentBlock::Text(text) => {
                warn_foreign_data(&text.provider_data, subject, d);
                let signature = own_signature(&text.provider_data);
                if text.text.is_empty() && signature.is_none() {
                    continue;
                }
                Part::text(text.text.clone()).signature(signature)
            }
            // Gemini has no refusal part: a model-written refusal from
            // another provider is replayed as the assistant text it was.
            ContentBlock::Refusal(refusal) => {
                warn_foreign_data(&refusal.provider_data, subject, d);
                if refusal.text.is_empty() {
                    continue;
                }
                Part::text(refusal.text.clone())
            }
            ContentBlock::Image(image) => {
                warn_foreign_data(&image.provider_data, subject.clone(), d);
                let mut part = Part::new(image_part(image, subject, d)?)
                    .signature(own_signature(&image.provider_data));
                // A draft image Gemini made while thinking goes back as one.
                if image.provider_data.get_raw(NAMESPACE, THOUGHT_KEY) == Some(&Value::Bool(true)) {
                    part.thought = Some(true);
                }
                part
            }
            ContentBlock::ToolCall(call) => {
                warn_foreign_data(&call.provider_data, subject.clone(), d);
                let args = replay_arguments(&call.name, &call.arguments, subject, d)?;
                Part::new(PartData::FunctionCall(FunctionCall {
                    id: (!call.id.is_empty()).then(|| call.id.clone()),
                    name: call.name.clone(),
                    args: Some(args),
                }))
                .signature(own_signature(&call.provider_data))
            }
            ContentBlock::Reasoning(reasoning) => {
                if reasoning.provider_data.namespace(NAMESPACE).is_none() {
                    let foreign: Vec<&str> = reasoning
                        .provider_data
                        .foreign_namespaces(NAMESPACE)
                        .collect();
                    if !foreign.is_empty() {
                        d.dropped_replay_state(
                            ReplayState::ForeignReasoning,
                            subject,
                            format!(
                                "reasoning from {} can be replayed only to its origin; it was dropped",
                                describe_origins(&foreign)
                            ),
                        );
                    } else if !reasoning.text.is_empty() || !reasoning.provider_data.is_empty() {
                        d.dropped_replay_state(
                            ReplayState::Reasoning,
                            subject,
                            "reasoning without Gemini thought data cannot be replayed; it was dropped",
                        );
                    }
                    continue;
                }
                warn_foreign_data(&reasoning.provider_data, subject, d);
                let mut part = Part::text(reasoning.text.clone())
                    .signature(own_signature(&reasoning.provider_data));
                part.thought = Some(true);
                part
            }
            ContentBlock::ProviderToolCall(call)
                if call.origin.as_deref() == Some(NAMESPACE)
                    && call.name == CODE_EXECUTION_TOOL =>
            {
                let code = serde_json::from_str::<Value>(&call.arguments)
                    .ok()
                    .filter(Value::is_object)
                    .ok_or_else(|| {
                        Error::invalid_request(
                            "a Gemini code-execution call needs its `executableCode` object as arguments",
                        ).authored()
                        .with_provider(PROVIDER_NAME)
                    })?;
                Part::new(PartData::Raw(Map::from_iter([(
                    EXECUTABLE_CODE.to_owned(),
                    code,
                )])))
                .signature(own_signature(&call.provider_data))
            }
            ContentBlock::ProviderToolResult(result)
                if result.origin.as_deref() == Some(NAMESPACE) =>
            {
                Part::new(PartData::Raw(Map::from_iter([(
                    CODE_EXECUTION_RESULT.to_owned(),
                    result.output.clone(),
                )])))
                .signature(own_signature(&result.provider_data))
            }
            ContentBlock::ProviderToolCall(_) | ContentBlock::ProviderToolResult(_) => {
                d.dropped_replay_state(
                    ReplayState::ForeignContent,
                    subject,
                    format!(
                        "`{}` content from {} can be replayed only to its origin; it was dropped",
                        item.type_name(),
                        describe_origins(item.origin().as_slice())
                    ),
                );
                continue;
            }
            ContentBlock::Other(other) if other.origin.as_deref() == Some(NAMESPACE) => {
                warn_foreign_data(&other.provider_data, subject, d);
                other_part(other)
            }
            ContentBlock::Other(other) => {
                d.dropped_replay_state(
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
            other => {
                return Err(d.reject(
                    subject,
                    format!(
                        "content block `{}` is not supported by the Gemini adapter",
                        other.type_name()
                    ),
                ));
            }
        };
        out.push(part.source(message, block));
    }
    Ok(out)
}

/// Gemini needs `functionCall.args` as an object. Invalid or non-object
/// arguments (kept verbatim on decode) are replayed as `{}` with a warning,
/// or rejected under the strict policy.
fn replay_arguments(
    name: &str,
    arguments: &str,
    subject: Subject,
    d: &mut Degradations,
) -> Result<Value> {
    if arguments.trim().is_empty() {
        return Ok(Value::Object(Map::new()));
    }
    match serde_json::from_str::<Value>(arguments) {
        Ok(value @ Value::Object(_)) => Ok(value),
        _ => {
            d.coded_lossy_hint(
                subject,
                format!(
                    "the arguments of tool call `{name}` are not a JSON object; they were sent as `{{}}`"
                ),
                "approximated.tool_call_arguments",
            )?;
            Ok(Value::Object(Map::new()))
        }
    }
}

fn encode_tool_result(
    index: usize,
    message: &ToolMessage,
    messages: &[Message],
    d: &mut Degradations,
) -> Result<Part> {
    let name = if message.name.is_empty() {
        tool_name_from_history(&messages[..index], &message.tool_call_id).ok_or_else(|| {
            Error::invalid_request(format!(
                "tool result `{}` has no tool name and no matching tool call",
                message.tool_call_id
            ))
            .authored()
            .with_provider(PROVIDER_NAME)
        })?
    } else {
        message.name.clone()
    };
    warn_foreign_data(&message.provider_data, content_subject(index, None), d);
    let key = if message.is_error { "error" } else { "result" };

    let (response, parts) = match &message.content {
        ToolResultContent::Text(text) => (json!({ key: text }), Vec::new()),
        ToolResultContent::Object(object) if message.is_error => {
            (json!({ "error": object }), Vec::new())
        }
        ToolResultContent::Object(object) => (Value::Object(object.clone()), Vec::new()),
        ToolResultContent::Parts(content) => {
            // Text parts are joined with blank lines, so their boundaries
            // survive; media goes in `parts`, after the text.
            let mut texts: Vec<&str> = Vec::new();
            let mut media = Vec::new();
            let mut reordered = false;
            for (block, part) in content.iter().enumerate() {
                let subject = content_subject(index, Some(block));
                match part {
                    ContentPart::Text(t) if t.text.is_empty() => {}
                    ContentPart::Text(t) => {
                        reordered |= !media.is_empty();
                        texts.push(&t.text);
                    }
                    ContentPart::Image(image) => {
                        if image.detail.is_some_and(|detail| detail != ImageDetail::Auto) {
                            d.hint(
                                subject,
                                "a Gemini function response part takes no media resolution; the image detail was not sent",
                            )?;
                        }
                        media.push(&image.source);
                    }
                    ContentPart::File(file) => {
                        if file.filename.is_some() {
                            d.dropped(subject, "Gemini file parts carry no filename; the filename was dropped");
                        }
                        media.push(&file.source);
                    }
                    other => d.dropped(
                        subject,
                        format!(
                            "a Gemini function response holds text and media only; `{}` content was dropped",
                            other.type_name()
                        ),
                    ),
                }
            }
            let text = texts.join("\n\n");
            if reordered {
                // Coded and joined: history repeats the result on every
                // turn, so one warning lists every message (ADR 0003).
                d.coded_approximated(
                    content_subject(index, None),
                    "a Gemini function response holds its text apart from its media; the media was sent after the text",
                    "approximated.tool_result_order",
                );
            }
            if media.is_empty() {
                (json!({ key: text }), Vec::new())
            } else {
                // Multimodal function responses, sent to every model: one
                // that lacks them refuses upstream (ADR 0015).
                let parts = media
                    .into_iter()
                    .map(|source| match source {
                        MediaSource::Base64 {
                            media_type, data, ..
                        } => Ok(FunctionResponsePart {
                            inline_data: Blob {
                                mime_type: media_type.clone(),
                                data: data.to_string(),
                            },
                        }),
                        _ => Err(d.reject(
                            content_subject(index, None),
                            "Gemini function responses accept inline (base64) media only",
                        )),
                    })
                    .collect::<Result<Vec<_>>>()?;
                let response = if text.is_empty() {
                    json!({})
                } else {
                    json!({ key: text })
                };
                (response, parts)
            }
        }
        _ => {
            return Err(d.reject(
                content_subject(index, None),
                "tool result content is not supported by the Gemini adapter",
            ));
        }
    };

    Ok(Part::new(PartData::FunctionResponse(FunctionResponse {
        id: (!message.tool_call_id.is_empty()).then(|| message.tool_call_id.clone()),
        name,
        response,
        parts,
    }))
    .source(index, 0))
}

fn tool_name_from_history(history: &[Message], id: &str) -> Option<String> {
    history.iter().rev().find_map(|message| {
        message
            .as_assistant()?
            .tool_calls()
            .find(|call| call.id == id)
            .map(|call| call.name.clone())
    })
}

fn encode_tools(tools: &[ToolDefinition], d: &mut Degradations) -> Result<Option<Vec<GeminiTool>>> {
    if tools.is_empty() {
        return Ok(None);
    }
    let mut declarations = Vec::with_capacity(tools.len());
    for tool in tools {
        if tool.strict == Some(true) {
            d.hint(
                Subject::Field {
                    field: RequestField::Tools,
                },
                format!(
                    "Gemini has no per-tool strict mode; `{}` was sent without `strict`",
                    tool.name
                ),
            )?;
        }
        // Tool cache breakpoints are satisfied by implicit prefix caching
        // (long retention is reported in `encode_request`).
        declarations.push(FunctionDeclaration {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters_json_schema: (!tool.parameters.is_null()).then(|| tool.parameters.clone()),
        });
    }
    Ok(Some(vec![GeminiTool::Functions {
        function_declarations: declarations,
    }]))
}

fn encode_tool_choice(choice: &ToolChoice, d: &Degradations) -> Result<ToolConfig> {
    let (mode, allowed) = match choice {
        ToolChoice::Auto => ("AUTO", None),
        ToolChoice::Disabled => ("NONE", None),
        ToolChoice::Required => ("ANY", None),
        ToolChoice::Specific { name } => ("ANY", Some(vec![name.clone()])),
        _ => {
            return Err(d.unsupported(
                RequestField::ToolChoice,
                "this tool choice is not supported by the Gemini adapter",
            ));
        }
    };
    Ok(ToolConfig {
        function_calling_config: FunctionCallingConfig {
            mode,
            allowed_function_names: allowed,
        },
    })
}

fn encode_generation_config(
    request: &ChatRequest,
    options: &ChatRequestOptions,
    family: ModelFamily,
    d: &mut Degradations,
) -> Result<GenerationConfig> {
    // A seed is a hint (ADR 0003): one Gemini cannot take is not sent.
    let seed = match request.seed.map(|seed| (seed, i32::try_from(seed))) {
        None => None,
        Some((_, Ok(seed))) => Some(seed),
        Some((seed, Err(_))) => {
            d.unsupported_hint(
                RequestField::Seed,
                format!(
                    "Gemini seeds are 32-bit signed integers; {seed} is out of range and was not sent"
                ),
            )?;
            None
        }
    };
    let (mut response_mime_type, response_json_schema) = match &request.response_format {
        None | Some(ResponseFormat::Text) => (None, None),
        Some(ResponseFormat::Json) => (Some("application/json".to_owned()), None),
        Some(ResponseFormat::JsonSchema { schema, .. }) => {
            let (schema, approximated) = crate::schema::response_schema(schema);
            if let Some(message) = approximated {
                d.approximated(
                    Subject::Field {
                        field: RequestField::ResponseFormat,
                    },
                    message,
                );
            }
            (Some("application/json".to_owned()), Some(schema))
        }
        Some(_) => {
            return Err(d.unsupported(
                RequestField::ResponseFormat,
                "this response format is not supported by the Gemini adapter",
            ));
        }
    };
    if options.response_mime_type.is_some() {
        response_mime_type = options.response_mime_type.clone();
    }

    Ok(GenerationConfig {
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        max_output_tokens: request.max_output_tokens,
        stop_sequences: (!request.stop.is_empty()).then(|| request.stop.clone()),
        seed,
        presence_penalty: request.presence_penalty,
        frequency_penalty: request.frequency_penalty,
        response_mime_type,
        response_json_schema,
        thinking_config: thinking_config(
            request.reasoning.as_ref(),
            options.thinking_budget,
            family,
            d,
        )?,
        extra: options.generation_config_extra.clone(),
    })
}

/// Map [`ReasoningConfig`] (and the `thinking_budget` override) to
/// `thinkingConfig`.
///
/// `includeThoughts`: `Summary`/`Full` → `true` (Gemini returns thought
/// summaries, so `Full` is approximated), `Omitted` → `false`; when reasoning
/// is requested without an output preference, thoughts are included. The
/// output setting only shows or hides thoughts: without an effort it keeps
/// the model's default thinking, and it never turns thinking on. Where the
/// thinking budget is 0 (thinking off), `Summary` or `Full` has no effect
/// and is an unsupported hint.
///
/// Effort, Gemini 3 (`thinkingLevel`): `Minimal` → `MINIMAL`, `Low` → `LOW`,
/// `Medium` → `MEDIUM`, `High` → `HIGH`, `XHigh` → `HIGH` (approximated).
/// `None` → `MINIMAL` (approximated: `thinkingLevel` cannot fully disable
/// thinking). Gemini 3 Pro rejects `MINIMAL`, so `None`/`Minimal` map to
/// `LOW` there.
///
/// Effort, Gemini 2.5 (`thinkingBudget` tokens): `None` → `0` (Pro cannot
/// disable thinking: its minimum, `128`, approximated), `Minimal` → `512`,
/// `Low` → `1024`, `Medium` → `8192`, `High` → `24576`, `XHigh` → `32768` on
/// Pro (its maximum) and `24576` on Flash (its maximum, approximated).
///
/// [`ChatRequestOptions::thinking_budget`] replaces the effort mapping and
/// is sent as-is (`-1` = dynamic). Gemini 1.x and 2.0 predate
/// `thinkingLevel`, so they get `thinkingBudget`, from the same ladder
/// without the 2.5 limits; a model without thinking refuses upstream (ADR
/// 0015: nothing is dropped because of the model). A model the adapter does
/// not know gets the current API's shape, Gemini 3's `thinkingLevel`, with
/// no Pro limits, and the API decides (ADR 0003, principle 2: `Unknown`
/// never drops).
///
/// Effort, Gemma 4 (`thinkingLevel`, on or off: "you can control it using
/// the API by setting the thinking level to "high" for enabled or "minimal"
/// for disabled", <https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api>):
/// `None` → `MINIMAL`, any other effort → `HIGH`, approximated unless it
/// is `High`.
pub(crate) fn thinking_config(
    reasoning: Option<&ReasoningConfig>,
    budget_override: Option<i32>,
    family: ModelFamily,
    d: &mut Degradations,
) -> Result<Option<ThinkingConfig>> {
    let field = Subject::Field {
        field: RequestField::Reasoning,
    };
    if reasoning.is_none() && budget_override.is_none() {
        return Ok(None);
    }

    let mut config = ThinkingConfig::default();
    let effort = reasoning.and_then(|r| r.effort);
    if let Some(budget) = budget_override {
        config.thinking_budget = Some(budget);
    } else if let Some(effort) = effort {
        match family {
            // A model the adapter does not know gets the current API's shape.
            ModelFamily::Gemini3 { .. } | ModelFamily::Other => {
                let pro = matches!(family, ModelFamily::Gemini3 { pro: true });
                let (level, note) = gemini3_level(effort, pro);
                config.thinking_level = Some(level);
                if let Some(note) = note {
                    d.approximated(field.clone(), note);
                }
            }
            ModelFamily::Gemini25 { .. } | ModelFamily::Legacy => {
                let gemini25_pro = match family {
                    ModelFamily::Gemini25 { pro } => Some(pro),
                    _ => None,
                };
                let (budget, note) = thinking_budget(effort, gemini25_pro);
                config.thinking_budget = Some(budget);
                if let Some(note) = note {
                    d.approximated(field.clone(), note);
                }
            }
            ModelFamily::Gemma4 => {
                let (level, note) = gemma4_level(effort);
                config.thinking_level = Some(level);
                if let Some(note) = note {
                    d.approximated(field.clone(), note);
                }
            }
        }
    }

    // A budget of 0 turns thinking off: there are no thoughts to show or
    // hide, so an output setting has no effect.
    let thinking_off = config.thinking_budget == Some(0);
    config.include_thoughts = match reasoning.and_then(|r| r.output) {
        Some(output @ (ReasoningOutput::Summary | ReasoningOutput::Full)) if thinking_off => {
            d.unsupported_hint(
                RequestField::Reasoning,
                format!(
                    "reasoning output `{output}` applies only while Gemini thinks, and the thinking budget is 0; `includeThoughts` was not sent"
                ),
            )?;
            None
        }
        Some(ReasoningOutput::Omitted) => Some(false),
        Some(ReasoningOutput::Summary) => Some(true),
        Some(ReasoningOutput::Full) => {
            d.approximated(
                field,
                "Gemini returns thought summaries, not the full reasoning; reasoning output `summary` was used instead",
            );
            Some(true)
        }
        None if reasoning.is_some() && effort != Some(ReasoningEffort::None) && !thinking_off => {
            Some(true)
        }
        _ => None,
    };

    Ok((config != ThinkingConfig::default()).then_some(config))
}

fn gemini3_level(effort: ReasoningEffort, pro: bool) -> (&'static str, Option<&'static str>) {
    match effort {
        ReasoningEffort::None if pro => (
            "LOW",
            Some("Gemini 3 Pro cannot disable thinking; `LOW`, its lowest level, was used instead"),
        ),
        ReasoningEffort::None => (
            "MINIMAL",
            Some("`thinkingLevel` cannot fully disable thinking; `MINIMAL` was used instead"),
        ),
        ReasoningEffort::Minimal if pro => (
            "LOW",
            Some("Gemini 3 Pro does not support `MINIMAL` thinking; `LOW` was used instead"),
        ),
        ReasoningEffort::Minimal => ("MINIMAL", None),
        ReasoningEffort::Low => ("LOW", None),
        ReasoningEffort::Medium => ("MEDIUM", None),
        ReasoningEffort::High => ("HIGH", None),
        ReasoningEffort::XHigh => (
            "HIGH",
            Some("Gemini has no thinking level above `HIGH`; `HIGH` was used instead"),
        ),
        _ => (
            "MEDIUM",
            Some("the reasoning effort is unknown to this adapter; `MEDIUM` was used instead"),
        ),
    }
}

fn gemma4_level(effort: ReasoningEffort) -> (&'static str, Option<&'static str>) {
    match effort {
        ReasoningEffort::None => ("MINIMAL", None),
        ReasoningEffort::High => ("HIGH", None),
        _ => (
            "HIGH",
            Some("Gemma 4 thinking is only on or off; `HIGH` was used instead"),
        ),
    }
}

/// The `thinkingBudget` for `effort`. `gemini25_pro` is the Gemini 2.5 tier
/// (Pro or Flash), whose limits the budget is fitted to; an older
/// generation (`None`) gets the ladder unchanged.
fn thinking_budget(
    effort: ReasoningEffort,
    gemini25_pro: Option<bool>,
) -> (i32, Option<&'static str>) {
    match (effort, gemini25_pro) {
        (ReasoningEffort::None, Some(true)) => (
            128,
            Some(
                "Gemini 2.5 Pro cannot disable thinking; its minimum budget of 128 tokens was used instead",
            ),
        ),
        (ReasoningEffort::None, _) => (0, None),
        (ReasoningEffort::Minimal, _) => (512, None),
        (ReasoningEffort::Low, _) => (1024, None),
        (ReasoningEffort::Medium, _) => (8192, None),
        (ReasoningEffort::High, _) => (24576, None),
        (ReasoningEffort::XHigh, Some(false)) => (
            24576,
            Some(
                "the maximum thinking budget of Gemini 2.5 Flash is 24576 tokens; it was used instead",
            ),
        ),
        (ReasoningEffort::XHigh, _) => (32768, None),
        _ => (
            8192,
            Some(
                "the reasoning effort is unknown to this adapter; a budget of 8192 tokens was used instead",
            ),
        ),
    }
}

/// An image part. Inline data carries its media type; a URL image uses
/// [`Image::media_type`](allama::Image::media_type), else its extension.
pub(crate) fn image_part(image: &Image, subject: Subject, d: &Degradations) -> Result<PartData> {
    let declared = image
        .resolved_media_type()
        .map_err(|err| err.with_provider(PROVIDER_NAME))?;
    match &image.source {
        MediaSource::Base64 {
            media_type, data, ..
        } => Ok(PartData::InlineData(Blob {
            mime_type: media_type.clone(),
            data: data.to_string(),
        })),
        MediaSource::Url { url } => {
            let mime_type = declared
                .or_else(|| infer_image_mime_type(url))
                .ok_or_else(|| {
                    d.reject(
                        subject.clone(),
                        "Gemini needs a MIME type for an image URL, and none can be inferred \
                         from its extension; set it with `Image::with_media_type`",
                    )
                })?;
            Ok(PartData::FileData(FileData {
                mime_type: Some(mime_type.to_owned()),
                file_uri: url.clone(),
            }))
        }
        // The Files API knows a stored file's MIME type.
        MediaSource::ProviderFile { origin, id, .. } if origin == NAMESPACE => {
            Ok(PartData::FileData(FileData {
                mime_type: declared.map(str::to_owned),
                file_uri: files_api_uri(id),
            }))
        }
        _ => Err(d.reject(subject, "unsupported media source for Gemini")),
    }
}

fn file_part(
    media_type: &str,
    source: &MediaSource,
    subject: Subject,
    d: &Degradations,
) -> Result<PartData> {
    match source {
        MediaSource::Url { url } => Ok(PartData::FileData(FileData {
            mime_type: Some(media_type.to_owned()),
            file_uri: url.clone(),
        })),
        MediaSource::Base64 { data, .. } => Ok(PartData::InlineData(Blob {
            mime_type: media_type.to_owned(),
            data: data.to_string(),
        })),
        MediaSource::ProviderFile { origin, id, .. } if origin == NAMESPACE => {
            Ok(PartData::FileData(FileData {
                mime_type: Some(media_type.to_owned()),
                file_uri: files_api_uri(id),
            }))
        }
        _ => Err(d.reject(subject, "unsupported media source for Gemini")),
    }
}

/// The Files API URI of a `google-ai` provider file: its `files/<id>` name
/// (or a bare id) under the Gemini API, or the id itself when it already is
/// one. The URI is Google's whatever the provider's base URL: the API
/// resolves it, not the transport.
fn files_api_uri(id: &str) -> String {
    if id.starts_with("https://") {
        return id.to_owned();
    }
    let name = id.strip_prefix("files/").unwrap_or(id);
    format!("{FILES_API_BASE}files/{name}")
}

/// Where Files API URIs live (`file.uri` in an upload response).
const FILES_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/";

/// The part `mediaResolution` for an image's detail: `Auto` (and no detail)
/// leaves it to the model.
fn media_resolution(
    image: &Image,
    subject: Subject,
    d: &mut Degradations,
) -> Result<Option<MediaResolution>> {
    let level = match image.detail {
        None | Some(ImageDetail::Auto) => return Ok(None),
        Some(ImageDetail::Low) => "MEDIA_RESOLUTION_LOW",
        Some(ImageDetail::High) => "MEDIA_RESOLUTION_HIGH",
        Some(detail) => {
            d.hint(
                subject,
                format!(
                    "Gemini has no media resolution for image detail `{}`; it was not sent",
                    detail.as_str()
                ),
            )?;
            return Ok(None);
        }
    };
    Ok(Some(MediaResolution { level }))
}

/// Best-effort image MIME type from a URL's file extension.
fn infer_image_mime_type(url: &str) -> Option<&'static str> {
    let path = url.split(['?', '#']).next().unwrap_or(url);
    let extension = path.rsplit('/').next()?.rsplit_once('.')?.1;
    Some(match extension.to_ascii_lowercase().as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "heic" => "image/heic",
        "heif" => "image/heif",
        _ => return None,
    })
}

/// Replay a part verbatim, with its signature.
///
/// This crate decodes the signature into provider data. A part in wire
/// form, a verbatim Gemini part such as one an application builds, keeps it
/// as `thoughtSignature` inside the part data; that is read too, and
/// provider data wins when both are present, so the key is sent once.
fn other_part(other: &OtherContent) -> Part {
    let mut data = other.data.clone();
    let own = own_signature(&other.provider_data);
    let wire = match data.remove(WIRE_THOUGHT_SIGNATURE) {
        Some(Value::String(signature)) => Some(signature),
        // Not a signature Gemini wrote: send it back untouched.
        Some(value) if own.is_none() => {
            data.insert(WIRE_THOUGHT_SIGNATURE.into(), value);
            None
        }
        _ => None,
    };
    Part::new(PartData::Raw(data)).signature(own.or(wire))
}

fn own_signature(data: &ProviderData) -> Option<String> {
    data.get_raw(NAMESPACE, THOUGHT_SIGNATURE_KEY)?
        .as_str()
        .map(str::to_owned)
}

/// Report opaque state from other providers that Gemini cannot use.
fn warn_foreign_data(data: &ProviderData, subject: Subject, d: &mut Degradations) {
    let foreign: Vec<&str> = data.foreign_namespaces(NAMESPACE).collect();
    if !foreign.is_empty() {
        d.dropped_replay_state(
            ReplayState::ForeignProviderData,
            subject,
            format!(
                "provider data from {} can be replayed only to its origin; it was dropped",
                describe_origins(&foreign)
            ),
        );
    }
}

fn content_subject(message: usize, block: Option<usize>) -> Subject {
    Subject::Content { message, block }
}

#[cfg(test)]
pub(crate) fn conformance_to_api_request(request: &ChatRequest) -> Result<Value> {
    let profile = ModelProfile::for_model(&request.model);
    Ok(serde_json::to_value(
        encode_request(request, profile)?.body,
    )?)
}

#[cfg(test)]
mod tests;
