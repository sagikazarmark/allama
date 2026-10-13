//! Wire types for the Cloudflare Workers AI binding and their conversions.
//!
//! Requests use the messages-based text-generation input
//! (`{ messages, max_tokens, ... }`), which this module encodes: the
//! binding's input differs from a Chat Completions request (no `model`,
//! native `top_k` and `repetition_penalty`, no `stop` or `tool_choice`), so
//! the encoder is this crate's own.
//!
//! Responses come in two shapes, depending on the model:
//!
//! - the native shape, `{ response, tool_calls, usage }`, used by most
//!   models (`tool_calls` items are flat `{ name, arguments }` or
//!   OpenAI-style `{ id, type, function: { name, arguments } }`);
//! - the OpenAI chat-completion shape, `{ id, model, choices: [{ message,
//!   finish_reason }], usage }`, used by newer models (for example
//!   `@cf/qwen/qwen3-30b-a3b-fp8`), often with the native fields repeated.
//!
//! Both are normalized into the chat-completion shape and decoded with
//! `allama-openai-compat`'s Chat Completions decoder
//! ([`decode_response`]), the decoder of the compat crate's `Cloudflare`
//! preset, so the two transports decode alike.

use allama::adapter::chat::{EncodedTurn, ToolCallIdFormat, retain_turns};
use allama::adapter::{Degradations, describe_origins};
use allama::{
    ChatRequest as AnyChatRequest, ChatResponse as AnyChatResponse, ContentBlock, ContentPart,
    Error, ExtraMap, FinishReason, Image, MediaSource, Message as AnyMessage, OtherContent,
    ProviderData, ReplayState, RequestField, ResponseFormat, Result, Subject, ToolChoice,
    ToolResultContent, UserContent, Warning, WarningKind,
};
use serde::Serialize;
use serde_json::{Map, Value};

use crate::PROVIDER_NAME;
use crate::options::refuse_owned_fields;
use crate::{ChatRequestOptions, NAMESPACE};

/// Request body for Workers AI text generation (messages-based).
#[derive(Debug, Serialize)]
pub(crate) struct ChatRequest {
    pub messages: Vec<Message>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<serde_json::Value>,

    /// The portable reasoning effort, by its name. Models that reason take
    /// it in their input schema (for example `@cf/moonshotai/kimi-k2.6`,
    /// `@cf/zai-org/glm-5.3`); which levels each takes is the model's.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<&'static str>,

    /// `ChatRequestOptions::extra`, checked against [`CHAT_INPUT_FIELDS`].
    #[serde(flatten)]
    pub extra: ExtraMap,
}

/// The input fields of [`ChatRequest`]: the adapter owns them, so
/// `ChatRequestOptions::extra` cannot set them.
pub(crate) const CHAT_INPUT_FIELDS: &[&str] = &[
    "frequency_penalty",
    "max_tokens",
    "messages",
    "presence_penalty",
    "reasoning_effort",
    "repetition_penalty",
    "response_format",
    "seed",
    "stream",
    "temperature",
    "tools",
    "top_k",
    "top_p",
];

/// A message in the Workers AI format.
#[derive(Debug, Serialize)]
pub(crate) struct Message {
    pub role: &'static str,

    /// Always present: the binding rejects `null` content on older runtimes.
    pub content: MessageContent,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<MessageToolCall>>,
}

/// Message content: a string, or (for user input with images) an array of
/// OpenAI-style parts, which vision models take in their input schema (for
/// example `@cf/meta/llama-4-scout-17b-16e-instruct`,
/// `@cf/mistralai/mistral-small-3.1-24b-instruct`).
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum MessageContent {
    Text(String),
    Parts(Vec<serde_json::Value>),
}

impl MessageContent {
    fn is_empty(&self) -> bool {
        match self {
            Self::Text(text) => text.is_empty(),
            Self::Parts(parts) => parts.is_empty(),
        }
    }
}

impl From<String> for MessageContent {
    fn from(text: String) -> Self {
        Self::Text(text)
    }
}

/// A tool call within an assistant message (OpenAI-style).
#[derive(Debug, Serialize)]
pub(crate) struct MessageToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: &'static str,
    pub function: MessageToolCallFunction,
}

/// The function payload within a replayed tool call.
#[derive(Debug, Serialize)]
pub(crate) struct MessageToolCallFunction {
    pub name: String,
    pub arguments: String,
}

/// Tool definition (OpenAI-style, accepted by Workers AI).
#[derive(Debug, Serialize)]
pub(crate) struct Tool {
    #[serde(rename = "type")]
    pub tool_type: &'static str,
    pub function: FunctionDef,
}

/// Function definition within a tool.
#[derive(Debug, Serialize)]
pub(crate) struct FunctionDef {
    pub name: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    pub parameters: serde_json::Value,
}

// --- Request encoding ---

/// Encode a portable request, applying the unsupported-feature policy.
///
/// Returns the wire body and the warnings to attach to the response.
pub(crate) fn encode_request(
    request: &AnyChatRequest,
    streaming: bool,
) -> Result<(ChatRequest, Vec<Warning>)> {
    let mut d = Degradations::new(
        PROVIDER_NAME,
        request.unsupported_policy.unwrap_or_default(),
    );
    crate::options::report_options_under_name(&request.options, "ChatRequestOptions", &mut d);
    let options = request
        .resolve_option::<ChatRequestOptions>()
        .map_err(|err| err.with_provider(PROVIDER_NAME))?;
    refuse_owned_fields(
        &options.extra,
        CHAT_INPUT_FIELDS,
        "ChatRequestOptions::extra",
    )?;

    let tools = encode_tools(request, &mut d)?;
    let response_format = encode_response_format(request, streaming, &mut d)?;

    if !request.stop.is_empty() {
        d.unsupported_hint(
            RequestField::Stop,
            "Workers AI text generation does not accept stop sequences; `stop` was not sent",
        )?;
    }
    // `true` is permission the model has anyway; only `false` asks for
    // something the binding cannot pass on.
    if request.parallel_tool_calls == Some(false) {
        d.unsupported_hint(
            RequestField::ParallelToolCalls,
            "Workers AI has no `parallel_tool_calls` field; `false` was not sent, so the model may still call several tools at once",
        )?;
    }
    let reasoning_effort = match request.reasoning {
        Some(config) => {
            if config.output.is_some() {
                d.unsupported_hint(
                    RequestField::Reasoning,
                    "Workers AI has no reasoning output setting; the reasoning output setting was not sent",
                )?;
            }
            config.effort.map(|effort| effort.as_str())
        }
        None => None,
    };
    if request.cache.is_some() {
        d.unsupported_hint(
            RequestField::Cache,
            "the Workers AI binding takes no prompt cache key or retention; they were not sent",
        )?;
    }

    let mut messages = Vec::with_capacity(request.system.len() + request.messages.len());
    for (index, prompt) in request.system.iter().enumerate() {
        check_breakpoint(prompt.cache.as_ref(), Subject::System { index }, &mut d)?;
        // Empty prompts carry no instruction; skip them like the HTTP adapters.
        if prompt.content.is_empty() {
            continue;
        }
        messages.push(Message {
            role: "system",
            content: prompt.content.clone().into(),
            tool_call_id: None,
            tool_calls: None,
        });
    }
    let mut turns = Vec::with_capacity(request.messages.len());
    for (index, message) in request.messages.iter().enumerate() {
        let encoded = encode_message(index, message, &mut d)?;
        let empty = encoded.content.is_empty() && encoded.tool_calls.is_none();
        turns.push((EncodedTurn::new(index, message, empty), encoded));
    }
    // Skip empty turns, and drop a reply the skips and drops left last.
    messages.extend(
        retain_turns(turns, &mut d, |(turn, _)| *turn)?
            .into_iter()
            .map(|(_, message)| message),
    );
    if mistral_model(&request.model) {
        conform_tool_call_ids(&mut messages, MISTRAL_TOOL_CALL_IDS);
    }

    let body = ChatRequest {
        messages,
        max_tokens: request.max_output_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        seed: request.seed,
        repetition_penalty: options.repetition_penalty,
        frequency_penalty: request.frequency_penalty,
        presence_penalty: request.presence_penalty,
        stream: streaming.then_some(true),
        tools,
        response_format,
        reasoning_effort,
        extra: options.extra,
    };
    Ok((body, d.into_warnings()))
}

fn encode_tools(request: &AnyChatRequest, d: &mut Degradations) -> Result<Option<Vec<Tool>>> {
    let tools = request.tools.as_slice();

    let send_tools = match &request.tool_choice {
        None | Some(ToolChoice::Auto) => !tools.is_empty(),
        // Not offering the tools is exactly "the model must not call tools".
        Some(ToolChoice::Disabled) => false,
        Some(ToolChoice::Required) => {
            return Err(d.unsupported(
                RequestField::ToolChoice,
                "Workers AI native chat cannot force a tool call",
            ));
        }
        Some(ToolChoice::Specific { name }) => {
            return Err(d.unsupported(
                RequestField::ToolChoice,
                format!("Workers AI native chat cannot force the `{name}` tool"),
            ));
        }
        Some(_) => {
            return Err(d.unsupported(
                RequestField::ToolChoice,
                "unknown tool choice is not supported by the Workers AI adapter",
            ));
        }
    };
    if !send_tools {
        return Ok(None);
    }
    // Streamed tool calls arrive in the chat-completion shape's
    // `delta.tool_calls` (recorded from `@cf/qwen/qwen3-30b-a3b-fp8`), which
    // the shared decoder reads, so tools stream too.

    let mut wire = Vec::with_capacity(tools.len());
    for tool in tools {
        if tool.strict == Some(true) {
            d.unsupported_hint(
                RequestField::Tools,
                format!(
                    "Workers AI does not support strict tool schemas; `{}` was sent without `strict`",
                    tool.name
                ),
            )?;
        }
        check_breakpoint(
            tool.cache.as_ref(),
            Subject::Field {
                field: RequestField::Tools,
            },
            d,
        )?;
        drop_foreign_provider_data(
            &tool.provider_data,
            Subject::Field {
                field: RequestField::Tools,
            },
            d,
        );
        wire.push(Tool {
            tool_type: "function",
            function: FunctionDef {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
            },
        });
    }
    Ok(Some(wire))
}

fn encode_response_format(
    request: &AnyChatRequest,
    streaming: bool,
    d: &mut Degradations,
) -> Result<Option<serde_json::Value>> {
    let format = match &request.response_format {
        // Plain text is the default; the binding only accepts JSON types here.
        None | Some(ResponseFormat::Text) => return Ok(None),
        Some(format) => format,
    };
    if streaming {
        return Err(d.unsupported(
            RequestField::ResponseFormat,
            "Workers AI JSON Mode does not support streaming",
        ));
    }
    match format {
        ResponseFormat::Json => Ok(Some(serde_json::json!({"type": "json_object"}))),
        ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
            ..
        } => {
            if name.is_some() {
                d.unsupported_hint(
                    RequestField::ResponseFormat,
                    "Workers AI JSON Mode takes a bare JSON Schema; the schema name was not sent",
                )?;
            }
            // Only a request for constrained decoding is lost: Workers AI
            // "can't guarantee that the model responds according to the
            // requested JSON Schema"
            // (https://developers.cloudflare.com/workers-ai/features/json-mode/).
            if *strict == Some(true) {
                d.unsupported_hint(
                    RequestField::ResponseFormat,
                    "Workers AI JSON Mode has no strict mode and does not guarantee the schema; it was sent without `strict`",
                )?;
            }
            Ok(Some(
                serde_json::json!({"type": "json_schema", "json_schema": schema}),
            ))
        }
        _ => Err(d.unsupported(
            RequestField::ResponseFormat,
            "unknown response format is not supported by the Workers AI adapter",
        )),
    }
}

fn encode_message(index: usize, message: &AnyMessage, d: &mut Degradations) -> Result<Message> {
    let subject = || Subject::Content {
        message: index,
        block: None,
    };
    let block_subject = |block: usize| Subject::Content {
        message: index,
        block: Some(block),
    };

    check_breakpoint(message.cache().as_ref(), subject(), d)?;
    drop_foreign_provider_data(message.provider_data(), subject(), d);

    match message {
        AnyMessage::User(user) => {
            let content = match &user.content {
                UserContent::Text(text) => MessageContent::Text(text.clone()),
                content => {
                    let owned;
                    let parts = match content {
                        UserContent::Parts(parts) => parts.as_slice(),
                        other => {
                            owned = other.to_parts();
                            owned.as_slice()
                        }
                    };
                    let mut texts = Vec::with_capacity(parts.len());
                    let mut wire_parts = Vec::with_capacity(parts.len());
                    for (block, part) in parts.iter().enumerate() {
                        match part {
                            ContentPart::Text(text) => {
                                drop_foreign_provider_data(
                                    &text.provider_data,
                                    block_subject(block),
                                    d,
                                );
                                texts.push(text.text.as_str());
                                wire_parts
                                    .push(serde_json::json!({"type": "text", "text": text.text}));
                            }
                            ContentPart::Image(image) => {
                                drop_foreign_provider_data(
                                    &image.provider_data,
                                    block_subject(block),
                                    d,
                                );
                                wire_parts.push(image_part(image, block_subject(block), d)?);
                            }
                            ContentPart::Other(other) if is_foreign(other) => {
                                drop_foreign_part(other, block_subject(block), d);
                            }
                            other => {
                                return Err(unsupported_part(other, block_subject(block), d));
                            }
                        }
                    }
                    // Text alone stays a string, which every model takes;
                    // an image needs the parts form.
                    if wire_parts.len() == texts.len() {
                        MessageContent::Text(texts.join("\n"))
                    } else {
                        MessageContent::Parts(wire_parts)
                    }
                }
            };
            Ok(Message {
                role: "user",
                content,
                tool_call_id: None,
                tool_calls: None,
            })
        }
        AnyMessage::Assistant(assistant) => {
            let mut text = String::new();
            let mut tool_calls = Vec::new();
            for (block, content) in assistant.content.iter().enumerate() {
                match content {
                    ContentBlock::Text(part) => {
                        drop_foreign_provider_data(&part.provider_data, block_subject(block), d);
                        text.push_str(&part.text);
                    }
                    // Workers AI has no refusal field: a model-written
                    // refusal is replayed as the assistant text it was.
                    ContentBlock::Refusal(refusal) => {
                        drop_foreign_provider_data(&refusal.provider_data, block_subject(block), d);
                        text.push_str(&refusal.text);
                    }
                    ContentBlock::ToolCall(call) => {
                        drop_foreign_provider_data(&call.provider_data, block_subject(block), d);
                        tool_calls.push(MessageToolCall {
                            id: call.id.clone(),
                            call_type: "function",
                            function: MessageToolCallFunction {
                                name: call.name.clone(),
                                arguments: call.arguments.clone(),
                            },
                        });
                    }
                    ContentBlock::Reasoning(reasoning) => {
                        let foreign = reasoning
                            .provider_data
                            .foreign_namespaces(NAMESPACE)
                            .next()
                            .is_some();
                        // Reasoning this binding returned: dropping it is
                        // expected, not a degradation.
                        if !foreign && reasoning.provider_data.namespace(NAMESPACE).is_some() {
                            continue;
                        }
                        let state = if foreign {
                            ReplayState::ForeignReasoning
                        } else {
                            ReplayState::Reasoning
                        };
                        d.dropped_replay_state(
                            state,
                            block_subject(block),
                            "Workers AI takes no reasoning back; it was dropped",
                        );
                    }
                    ContentBlock::Other(other) if is_foreign(other) => d.dropped_replay_state(
                        ReplayState::ForeignContent,
                        block_subject(block),
                        format!(
                            "`{}` content from `{}` can be replayed only to its origin; it was dropped",
                            other.type_name,
                            other.origin.as_deref().unwrap_or_default()
                        ),
                    ),
                    ContentBlock::Other(other) => {
                        let subject = block_subject(block);
                        let message = format!(
                            "Workers AI cannot replay `{}` content; it was dropped",
                            other.type_name
                        );
                        // Content this provider wrote recurs with the
                        // history; the caller's own raw content stays
                        // uncoded (ADR 0003).
                        match other.origin {
                            Some(_) => {
                                d.dropped_replay_state(ReplayState::Content, subject, message);
                            }
                            None => d.warn(Warning::new(WarningKind::Dropped, subject, message)),
                        }
                    }
                    // Workers AI runs no provider tools: the block is
                    // another provider's.
                    ContentBlock::ProviderToolCall(_) | ContentBlock::ProviderToolResult(_) => d
                        .dropped_replay_state(
                            ReplayState::ForeignContent,
                            block_subject(block),
                            format!(
                                "`{}` content from {} can be replayed only to its origin; it was dropped",
                                content.type_name(),
                                describe_origins(content.origin().as_slice())
                            ),
                        ),
                    ContentBlock::Image(_) => {
                        return Err(d.reject(
                            block_subject(block),
                            "Workers AI native chat cannot replay assistant images",
                        ));
                    }
                    _ => {
                        return Err(d.reject(
                            block_subject(block),
                            format!(
                                "`{}` assistant content is not supported by Workers AI",
                                content.type_name()
                            ),
                        ));
                    }
                }
            }
            Ok(Message {
                role: "assistant",
                content: text.into(),
                tool_call_id: None,
                tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            })
        }
        AnyMessage::Tool(tool) => {
            let not_text = |d: &Degradations, subject| {
                d.reject(subject, "Workers AI tool results must be text or JSON")
            };
            let content = match &tool.content {
                ToolResultContent::Parts(parts) => {
                    // Joined so the part boundaries survive, as the
                    // Chat Completions and Gemini encoders join them.
                    let mut texts = Vec::with_capacity(parts.len());
                    for (block, part) in parts.iter().enumerate() {
                        match part {
                            ContentPart::Text(part) => texts.push(part.text.as_str()),
                            ContentPart::Other(other) if is_foreign(other) => {
                                drop_foreign_part(other, block_subject(block), d);
                            }
                            _ => return Err(not_text(d, block_subject(block))),
                        }
                    }
                    texts.join("\n\n")
                }
                content => match content.to_text() {
                    Some(text) => text,
                    None => return Err(not_text(d, subject())),
                },
            };
            if tool.is_error {
                // Coded: history repeats the result on every turn. Never
                // refused: the model still reads the error text, only the
                // flag is lost (ADR 0003).
                d.coded_approximated(
                    subject(),
                    "Workers AI has no tool error flag; the error text was sent as the tool result; the model still reads the error text; to send your own encoding to this endpoint, use `push_tool_result`",
                    "approximated.tool_result_error",
                );
            }
            Ok(Message {
                role: "tool",
                content: content.into(),
                tool_call_id: Some(tool.tool_call_id.clone()),
                tool_calls: None,
            })
        }
        _ => Err(d.reject(subject(), "unknown message variant")),
    }
}

/// Cache breakpoints are satisfied by Workers AI's automatic prefix caching;
/// an explicit retention cannot be honored.
fn check_breakpoint(
    breakpoint: Option<&allama::CacheBreakpoint>,
    subject: Subject,
    d: &mut Degradations,
) -> Result<()> {
    if breakpoint.is_some_and(|b| b.retention.is_some()) {
        d.hint(
            subject,
            "Workers AI caches prompt prefixes automatically and takes no cache retention; the retention was not sent",
        )?;
    }
    Ok(())
}

/// Report (and skip) provider data that belongs to another provider.
fn drop_foreign_provider_data(data: &ProviderData, subject: Subject, d: &mut Degradations) {
    for namespace in data.foreign_namespaces(NAMESPACE) {
        d.dropped_replay_state(
            ReplayState::ForeignProviderData,
            subject.clone(),
            format!("provider data from `{namespace}` can be replayed only to its origin; it was dropped"),
        );
    }
}

/// Whether `other` came from another provider. Such content is never
/// sent: it is dropped with a warning, like any foreign state.
fn is_foreign(other: &OtherContent) -> bool {
    other
        .origin
        .as_deref()
        .is_some_and(|origin| origin != NAMESPACE)
}

fn drop_foreign_part(other: &OtherContent, subject: Subject, d: &mut Degradations) {
    d.dropped_replay_state(
        ReplayState::ForeignContent,
        subject,
        format!(
            "`{}` content from `{}` can be replayed only to its origin; it was dropped",
            other.type_name,
            other.origin.as_deref().unwrap_or_default()
        ),
    );
}

/// An image as an OpenAI-style `image_url` part: a URL as given, inline
/// data as a `data:` URI. Which forms a model takes (some take `data:` URIs
/// only) is the model's to say.
fn image_part(image: &Image, subject: Subject, d: &Degradations) -> Result<serde_json::Value> {
    let url = match &image.source {
        MediaSource::Url { url } => url.clone(),
        MediaSource::Base64 {
            media_type, data, ..
        } => format!("data:{media_type};base64,{data}"),
        _ => return Err(d.reject(subject, "unknown image source")),
    };
    let mut image_url = serde_json::Map::new();
    image_url.insert("url".into(), url.into());
    if let Some(detail) = image.detail {
        image_url.insert("detail".into(), detail.as_str().into());
    }
    Ok(serde_json::json!({"type": "image_url", "image_url": image_url}))
}

fn unsupported_part(part: &ContentPart, subject: Subject, d: &Degradations) -> Error {
    let kind = match part {
        ContentPart::Image(_) => "image".to_string(),
        ContentPart::File(_) => "file".to_string(),
        ContentPart::Other(other) => format!("`{}`", other.type_name),
        _ => format!("`{}`", part.type_name()),
    };
    d.reject(
        subject,
        format!("Workers AI native chat does not accept {kind} user content"),
    )
}

// --- Response decoding ---

/// Decode a non-streaming binding output.
///
/// The output is normalized into the chat-completion shape
/// ([`normalize_response`]) and decoded with `allama-openai-compat`'s Chat
/// Completions decoder, so both Cloudflare transports (this binding and the
/// compat crate's `Cloudflare` preset) map one shape one way: tool calls,
/// reasoning, finish reasons (`stop` with tool calls is `ToolCalls`; `eos`
/// and `model_length` are aliases) and vendor fields (`neurons`) alike.
///
/// `turn` is the index the assistant message will occupy in the
/// conversation (`request.messages.len()`). It scopes the ids made up for
/// native tool calls without one, so ids are deterministic and unique
/// within a conversation without global state.
pub(crate) fn decode_response(output: serde_json::Value, turn: usize) -> Result<AnyChatResponse> {
    let body = normalize_response(output, turn)?;
    let mut response =
        allama_openai_compat::binding::decode_response(body, PROVIDER_NAME, NAMESPACE)?;
    infer_finish_reason(&mut response);
    Ok(response)
}

/// Older models report no finish reason; a returned response means the turn
/// ended, so it is inferred (the raw reason stays unset).
fn infer_finish_reason(response: &mut AnyChatResponse) {
    if response.finish_reason.is_none() && response.finish_reason_raw.is_none() {
        response.finish_reason = Some(if response.has_tool_calls() {
            FinishReason::ToolCalls
        } else {
            FinishReason::Stop
        });
    }
}

/// The binding output in the chat-completion shape.
///
/// - The chat-completion shape (`choices`) is kept; the native fields that
///   repeat it (`response`, `tool_calls`) are removed.
/// - The native shape (`response`, `tool_calls`, `finish_reason`) becomes
///   one choice. A JSON-mode `response` object becomes its JSON text, and
///   native tool calls get the OpenAI shape, with string arguments and an
///   id made up from `turn` when they have none.
/// - Usage counts sent as integral floats (JS numbers) become integers, a
///   placeholder total below input plus output is dropped, and an empty
///   usage object is no usage.
fn normalize_response(output: Value, turn: usize) -> Result<Value> {
    let Value::Object(mut object) = output else {
        return Err(
            Error::unexpected_response("the Workers AI output is not an object")
                .authored()
                .with_provider(PROVIDER_NAME),
        );
    };
    let chat_shaped = object
        .get("choices")
        .and_then(Value::as_array)
        .is_some_and(|choices| !choices.is_empty());
    if chat_shaped {
        object.remove("response");
        object.remove("tool_calls");
        for choice in object
            .get_mut("choices")
            .and_then(Value::as_array_mut)
            .into_iter()
            .flatten()
        {
            if let Some(calls) = choice
                .pointer_mut("/message/tool_calls")
                .and_then(Value::as_array_mut)
            {
                calls.iter_mut().for_each(stringify_arguments);
            }
        }
    } else {
        let content = match object.remove("response") {
            None | Some(Value::Null) => Value::Null,
            Some(text @ Value::String(_)) => text,
            Some(json) => Value::String(json.to_string()),
        };
        let calls = native_tool_calls(object.remove("tool_calls"), |index| {
            Some(synthetic_tool_call_id(turn, index))
        })?;
        let mut message = Map::new();
        message.insert("role".into(), "assistant".into());
        message.insert("content".into(), content);
        if !calls.is_empty() {
            message.insert("tool_calls".into(), Value::Array(calls));
        }
        let mut choice = Map::new();
        choice.insert("index".into(), 0.into());
        choice.insert("message".into(), Value::Object(message));
        if let Some(reason) = object.remove("finish_reason") {
            choice.insert("finish_reason".into(), reason);
        }
        object.insert("choices".into(), Value::Array(vec![Value::Object(choice)]));
    }
    normalize_usage(&mut object);
    Ok(Value::Object(object))
}

/// Native tool calls (`{name, arguments}` or the OpenAI shape) in the
/// OpenAI shape, with string arguments, and ids from `made_up` where they
/// have none (no id where it gives none: the decoder then makes one up).
pub(crate) fn native_tool_calls(
    calls: Option<Value>,
    made_up: impl Fn(usize) -> Option<String>,
) -> Result<Vec<Value>> {
    let Some(Value::Array(calls)) = calls else {
        return Ok(Vec::new());
    };
    let mut normalized = Vec::with_capacity(calls.len());
    for (index, call) in calls.into_iter().enumerate() {
        let Value::Object(mut call) = call else {
            continue;
        };
        let function = match call.remove("function") {
            Some(Value::Object(function)) => function,
            _ => Map::new(),
        };
        let name = function
            .get("name")
            .or_else(|| call.get("name"))
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| {
                Error::unexpected_response(format!(
                    "Workers AI returned tool call {index} without a name"
                ))
                .authored()
                .with_provider(PROVIDER_NAME)
            })?
            .to_owned();
        let arguments = arguments_text(function.get("arguments").or_else(|| call.get("arguments")));
        let id = call
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .map(str::to_owned)
            .or_else(|| made_up(index));
        let mut normalized_call = serde_json::json!({
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        });
        if let Some(id) = id {
            normalized_call["id"] = id.into();
        }
        normalized.push(normalized_call);
    }
    Ok(normalized)
}

/// Tool call arguments as JSON text: a string as it is (even malformed),
/// nothing as `{}`, and an object as its JSON text.
fn arguments_text(arguments: Option<&Value>) -> String {
    match arguments {
        Some(Value::String(raw)) => raw.clone(),
        None | Some(Value::Null) => "{}".to_owned(),
        Some(value) => value.to_string(),
    }
}

/// Make an OpenAI-shaped tool call's arguments a string.
fn stringify_arguments(call: &mut Value) {
    if let Some(arguments) = call.pointer_mut("/function/arguments")
        && !arguments.is_string()
        && !arguments.is_null()
    {
        *arguments = Value::String(arguments.to_string());
    }
}

/// Normalize the `usage` of an output or chunk (see [`normalize_response`]).
pub(crate) fn normalize_usage(object: &mut Map<String, Value>) {
    let Some(Value::Object(usage)) = object.get_mut("usage") else {
        return;
    };
    if usage.is_empty() {
        object.remove("usage");
        return;
    }
    integral_counts(usage);
    let count = |usage: &Map<String, Value>, key: &str| usage.get(key).and_then(Value::as_u64);
    if let (Some(input), Some(output), Some(total)) = (
        count(usage, "prompt_tokens"),
        count(usage, "completion_tokens"),
        count(usage, "total_tokens"),
    ) && total < input.saturating_add(output)
    {
        // Workers AI documents a default of 0: a placeholder, not a total.
        usage.remove("total_tokens");
    }
}

/// Token counts sent as integral floats (JS numbers), as integers.
fn integral_counts(usage: &mut Map<String, Value>) {
    for (key, value) in usage.iter_mut() {
        match value {
            Value::Object(details) => integral_counts(details),
            Value::Number(number) if key.ends_with("_tokens") && !number.is_u64() => {
                if let Some(float) = number.as_f64().filter(|f| *f >= 0.0 && f.fract() == 0.0) {
                    *value = Value::from(float as u64);
                }
            }
            _ => {}
        }
    }
}

/// The tool call ids Mistral models take on Workers AI: any other id is
/// refused ("Tool call id was … but must be a-z, A-Z, 0-9, with a length of
/// 9", verified live on `@cf/mistralai/mistral-small-3.1-24b-instruct`).
/// Other models take any id.
const MISTRAL_TOOL_CALL_IDS: ToolCallIdFormat = ToolCallIdFormat::Alphanumeric(9);

/// A Mistral model id: `@cf/mistralai/…`, `@cf/mistral/…` or
/// `@hf/mistral/…`.
fn mistral_model(model: &str) -> bool {
    let mut segments = model.split('/');
    matches!(segments.next(), Some("@cf" | "@hf"))
        && segments
            .next()
            .is_some_and(|vendor| vendor.starts_with("mistral"))
}

/// Map the tool call ids of the encoded messages to ones `format` accepts,
/// the same way for a call and the `tool` message that answers it.
fn conform_tool_call_ids(messages: &mut [Message], format: ToolCallIdFormat) {
    let conform = |id: &mut String| {
        if let std::borrow::Cow::Owned(mapped) = format.conform(id) {
            *id = mapped;
        }
    };
    for message in messages {
        message
            .tool_calls
            .iter_mut()
            .flatten()
            .for_each(|call| conform(&mut call.id));
        if let Some(id) = &mut message.tool_call_id {
            conform(id);
        }
    }
}

/// Deterministic id for a tool call the upstream returned without one.
pub(crate) fn synthetic_tool_call_id(turn: usize, index: usize) -> String {
    format!("cf_call_{turn}_{index}")
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use allama::{
        CacheBreakpoint, CacheRetention, ChatRequest as AnyChatRequest, ContentPart, Image,
        ImageDetail, Message as AnyMessage, OtherContent, PromptCache, Reasoning, ReasoningConfig,
        ReasoningEffort, ReasoningOutput, ToolChoice, ToolDefinition, UnsupportedPolicy,
    };
    use serde_json::json;

    use super::*;

    const MODEL: &str = "@cf/meta/llama-3.1-8b-instruct";

    fn weather_tool() -> ToolDefinition {
        ToolDefinition::new(
            "lookup_weather",
            json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }),
        )
        .description("Look up the weather.")
    }

    fn encode(request: &AnyChatRequest) -> (serde_json::Value, Vec<Warning>) {
        let (body, warnings) = encode_request(request, false).unwrap();
        (serde_json::to_value(body).unwrap(), warnings)
    }

    fn decode(value: serde_json::Value) -> AnyChatResponse {
        decode_response(value, 3).unwrap()
    }

    /// JSON options under `cloudflare`, the
    /// provider name, are reported dropped; `cloudflare-workers-ai` is read.
    #[test]
    fn options_under_the_provider_name_are_reported_dropped() {
        let request = AnyChatRequest::new(MODEL)
            .user("q")
            .with_options_json(PROVIDER_NAME, json!({"extra": {"lora": "acme"}}));
        let (value, warnings) = encode(&request);
        assert!(value.get("lora").is_none(), "{value}");
        let [warning] = warnings.as_slice() else {
            panic!("one warning: {warnings:?}");
        };
        assert_eq!(warning.kind, WarningKind::Dropped);
        assert_eq!(
            warning.code.as_deref(),
            Some("dropped.provider_name_options")
        );
        assert_eq!(
            warning.message,
            "Workers AI reads options under `cloudflare-workers-ai`, not under its name `cloudflare`; they were dropped; pass them under `cloudflare-workers-ai`, or as `ChatRequestOptions`"
        );

        let request = AnyChatRequest::new(MODEL)
            .user("q")
            .with_options_json(crate::NAMESPACE, json!({"extra": {"lora": "acme"}}));
        let (value, warnings) = encode(&request);
        assert_eq!(value["lora"], "acme");
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    #[test]
    fn request_conversion_preserves_supported_controls() {
        let request = AnyChatRequest::new(MODEL)
            .system("You are concise.")
            .user("Say hello")
            .temperature(0.2)
            .top_p(0.9)
            .top_k(20)
            .max_output_tokens(64)
            .seed(42)
            .frequency_penalty(0.1)
            .presence_penalty(0.2)
            .tools(vec![weather_tool()])
            .response_format(ResponseFormat::json_schema(json!({"type": "object"})).non_strict());

        let (value, warnings) = encode(&request);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(value["max_tokens"], 64);
        assert_eq!(value["temperature"], 0.2);
        assert_eq!(value["top_p"], 0.9);
        assert_eq!(value["seed"], 42);
        assert_eq!(value["frequency_penalty"], 0.1);
        assert_eq!(value["presence_penalty"], 0.2);
        assert_eq!(value["tools"][0]["function"]["name"], "lookup_weather");
        assert_eq!(value["response_format"]["type"], "json_schema");
        assert_eq!(value["messages"][0]["role"], "system");
        assert!(value.get("stream").is_none());
        assert_eq!(value["top_k"], 20);
        assert!(value.get("repetition_penalty").is_none());
    }

    #[test]
    fn provider_options_resolve_from_typed_value() {
        let request = AnyChatRequest::new(MODEL).user("hi").with_option(
            ChatRequestOptions::new()
                .repetition_penalty(1.2)
                .extra("lora", "my-adapter"),
        );
        let (value, _) = encode(&request);
        assert_eq!(value["repetition_penalty"], 1.2);
        assert_eq!(value["lora"], "my-adapter");
    }

    #[test]
    fn extra_cannot_set_owned_fields() {
        for key in CHAT_INPUT_FIELDS.iter().copied().chain(["Messages"]) {
            let request = AnyChatRequest::new(MODEL)
                .user("hi")
                .with_option(ChatRequestOptions::new().extra(key, 1));
            let err = encode_request(&request, false).unwrap_err();
            assert!(
                matches!(err, Error::InvalidRequest { .. }),
                "{key}: {err:?}"
            );
        }
    }

    #[test]
    fn every_input_field_is_reserved() {
        // A field added to the input must be reserved too, or `extra` could
        // send it twice.
        let request = AnyChatRequest::new(MODEL)
            .user("hi")
            .temperature(0.2)
            .top_p(0.9)
            .top_k(1)
            .max_output_tokens(64)
            .seed(42)
            .frequency_penalty(0.1)
            .presence_penalty(0.2)
            .reasoning(allama::ReasoningEffort::Low)
            .tools(vec![weather_tool()])
            .response_format(ResponseFormat::json_schema(json!({"type": "object"})).non_strict())
            .with_option(ChatRequestOptions::new().repetition_penalty(1.0));
        for streaming in [false, true] {
            let request = if streaming {
                AnyChatRequest::new(MODEL).user("hi")
            } else {
                request.clone()
            };
            let (body, _) = encode_request(&request, streaming).unwrap();
            for key in serde_json::to_value(body)
                .unwrap()
                .as_object()
                .unwrap()
                .keys()
            {
                assert!(
                    CHAT_INPUT_FIELDS.contains(&key.as_str()),
                    "{key} is not reserved"
                );
            }
        }
    }

    #[test]
    fn provider_options_resolve_from_json() {
        let request = AnyChatRequest::new(MODEL)
            .user("hi")
            .with_options_json(NAMESPACE, json!({"repetition_penalty": 1.5}));
        let (value, _) = encode(&request);
        assert_eq!(value["repetition_penalty"].as_f64(), Some(1.5));
        assert!(value.get("top_k").is_none());
    }

    #[test]
    fn invalid_provider_options_fail() {
        let request = AnyChatRequest::new(MODEL)
            .user("hi")
            .with_options_json(NAMESPACE, json!({"repetition_penalty": "many"}));
        let err = encode_request(&request, false).unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }));
    }

    #[test]
    fn text_response_format_is_not_sent() {
        let request = AnyChatRequest::new(MODEL)
            .user("hi")
            .response_format(ResponseFormat::Text);
        let (value, warnings) = encode(&request);
        assert!(value.get("response_format").is_none());
        assert!(warnings.is_empty());
    }

    #[test]
    fn json_mode_is_sent_as_json_object() {
        let request = AnyChatRequest::new(MODEL)
            .user("hi")
            .response_format(ResponseFormat::Json);
        let (value, _) = encode(&request);
        assert_eq!(value["response_format"], json!({"type": "json_object"}));
    }

    /// `json_schema` asks for strict mode, which JSON Mode cannot promise:
    /// it is reported. `non_strict` asks for nothing more than JSON Mode
    /// gives. The wire is the same.
    #[test]
    fn strict_and_non_strict_schemas_encode_the_same() {
        let schema = json!({"type": "object", "properties": {"a": {"type": "string"}}});
        let strict = AnyChatRequest::new(MODEL)
            .user("Hello")
            .response_format(ResponseFormat::json_schema(schema.clone()));
        let (value, warnings) = encode(&strict);
        let expected = json!({"type": "json_schema", "json_schema": schema});
        assert_eq!(value["response_format"], expected);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert_eq!(warnings[0].kind, WarningKind::Unsupported);

        let relaxed = AnyChatRequest::new(MODEL)
            .user("Hello")
            .response_format(ResponseFormat::json_schema(schema).non_strict());
        let (value, warnings) = encode(&relaxed);
        assert_eq!(value["response_format"], expected);
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    #[test]
    fn hints_warn_under_default_policy() {
        let request = AnyChatRequest::new(MODEL)
            .user("Hello")
            .stop(["END"])
            .parallel_tool_calls(false)
            .reasoning(ReasoningConfig::new().output(ReasoningOutput::Summary))
            .cache(PromptCache::new().key("session"))
            .response_format(ResponseFormat::json_schema(json!({"type": "object"})).name("answer"));

        let (value, warnings) = encode(&request);
        let fields: Vec<_> = warnings
            .iter()
            .map(|w| {
                assert_eq!(w.kind, WarningKind::Unsupported);
                assert_eq!(w.provider.as_deref(), Some("cloudflare"));
                match &w.subject {
                    Subject::Field { field } => *field,
                    other => panic!("unexpected subject {other:?}"),
                }
            })
            .collect();
        assert_eq!(
            fields,
            vec![
                RequestField::ResponseFormat,
                RequestField::ResponseFormat,
                RequestField::Stop,
                RequestField::ParallelToolCalls,
                RequestField::Reasoning,
                RequestField::Cache,
            ]
        );
        assert!(value.get("stop").is_none());
        assert_eq!(
            value["response_format"],
            json!({"type": "json_schema", "json_schema": {"type": "object"}})
        );
    }

    /// `true` is permission the model has anyway: nothing is sent and
    /// nothing is warned, under any policy.
    #[test]
    fn parallel_tool_calls_true_is_never_a_degradation() {
        let request = AnyChatRequest::new(MODEL)
            .user("Hello")
            .parallel_tool_calls(true);
        let (value, warnings) = encode(&request);
        assert!(value.get("parallel_tool_calls").is_none());
        assert!(warnings.is_empty(), "{warnings:?}");
        let strict = request.unsupported_policy(UnsupportedPolicy::Error);
        assert!(encode_request(&strict, false).is_ok());

        let (_, warnings) = encode(
            &AnyChatRequest::new(MODEL)
                .user("Hello")
                .parallel_tool_calls(false),
        );
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert_eq!(
            warnings[0].message,
            "Workers AI has no `parallel_tool_calls` field; `false` was not sent, so the model may still call several tools at once"
        );
    }

    #[test]
    fn hints_fail_under_strict_policy() {
        let strict = |request: AnyChatRequest| {
            encode_request(&request.unsupported_policy(UnsupportedPolicy::Error), false)
                .unwrap_err()
        };
        for (request, needle) in [
            (AnyChatRequest::new(MODEL).user("x").stop(["END"]), "stop"),
            (
                AnyChatRequest::new(MODEL)
                    .user("x")
                    .parallel_tool_calls(false),
                "parallel",
            ),
            (
                AnyChatRequest::new(MODEL)
                    .user("x")
                    .reasoning(ReasoningConfig::new().output(ReasoningOutput::Omitted)),
                "reasoning output",
            ),
            (
                AnyChatRequest::new(MODEL)
                    .user("x")
                    .cache(PromptCache::new()),
                "cache",
            ),
            (
                AnyChatRequest::new(MODEL)
                    .system(
                        allama::SystemPrompt::new("s")
                            .with_cache(CacheBreakpoint::with_retention(CacheRetention::Long)),
                    )
                    .user("x"),
                "retention",
            ),
        ] {
            let err = strict(request);
            assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
            assert!(
                err.to_string().contains(needle),
                "{err} should mention {needle}"
            );
            assert_eq!(
                err.context().unwrap().provider.as_deref(),
                Some("cloudflare")
            );
        }
    }

    #[test]
    fn plain_cache_breakpoints_are_satisfied_without_warning() {
        let request = AnyChatRequest::new(MODEL)
            .system(allama::SystemPrompt::new("s").with_cache(CacheBreakpoint::new()))
            .message(AnyMessage::user("x").with_cache(CacheBreakpoint::new()))
            .unsupported_policy(UnsupportedPolicy::Error);
        let (_, warnings) = encode(&request);
        assert!(warnings.is_empty());
    }

    #[test]
    fn forced_tool_choice_is_rejected_even_when_lenient() {
        for choice in [
            ToolChoice::Required,
            ToolChoice::Specific {
                name: "lookup_weather".into(),
            },
        ] {
            let request = AnyChatRequest::new(MODEL)
                .user("Hello")
                .tools(vec![weather_tool()])
                .tool_choice(choice);
            let err = encode_request(&request, false).unwrap_err();
            assert!(matches!(err, Error::Unsupported { .. }));
            assert!(err.to_string().contains("tool_choice"));
        }
    }

    #[test]
    fn disabled_tool_choice_omits_tools() {
        let request = AnyChatRequest::new(MODEL)
            .user("Hello")
            .tools(vec![weather_tool()])
            .tool_choice(ToolChoice::Disabled);
        let (value, warnings) = encode(&request);
        assert!(value.get("tools").is_none());
        assert!(warnings.is_empty());

        // And streaming is allowed, since no tools are offered.
        assert!(encode_request(&request, true).is_ok());
    }

    #[test]
    fn auto_tool_choice_sends_tools() {
        let request = AnyChatRequest::new(MODEL)
            .user("Hello")
            .tools(vec![weather_tool()])
            .tool_choice(ToolChoice::Auto);
        let (value, _) = encode(&request);
        assert_eq!(value["tools"][0]["type"], "function");
    }

    #[test]
    fn strict_tools_warn() {
        let request = AnyChatRequest::new(MODEL)
            .user("Hello")
            .tools(vec![weather_tool().strict(true)]);
        let (_, warnings) = encode(&request);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].message.contains("strict"));
    }

    /// JSON mode does not stream; tools do (the shared decoder reads
    /// streamed tool calls).
    #[test]
    fn streaming_rejects_response_format_even_when_lenient() {
        let request = AnyChatRequest::new(MODEL)
            .user("Hello")
            .tools(vec![weather_tool()]);
        let (body, warnings) = encode_request(&request, true).unwrap();
        assert!(body.tools.is_some());
        assert!(warnings.is_empty(), "{warnings:?}");

        let request = AnyChatRequest::new(MODEL)
            .user("Hello")
            .response_format(ResponseFormat::Json);
        let err = encode_request(&request, true).unwrap_err();
        assert!(err.to_string().contains("response_format"));
    }

    #[test]
    fn streaming_sets_stream_flag() {
        let request = AnyChatRequest::new(MODEL).user("Hello");
        let (body, _) = encode_request(&request, true).unwrap();
        assert_eq!(body.stream, Some(true));
    }

    #[test]
    fn user_text_parts_join_with_newlines() {
        let request = AnyChatRequest::new(MODEL).message(AnyMessage::user_parts(vec![
            ContentPart::text("First line"),
            ContentPart::text("Second line"),
        ]));
        let (value, _) = encode(&request);
        assert_eq!(value["messages"][0]["content"], "First line\nSecond line");
    }

    /// The reasoning effort is sent by its name, for any model (ADR 0015):
    /// which levels a model takes is the model's to say.
    #[test]
    fn reasoning_effort_is_sent_whatever_the_model() {
        for model in [MODEL, "@cf/moonshotai/kimi-k2.6"] {
            let (value, warnings) = encode(
                &AnyChatRequest::new(model)
                    .user("x")
                    .reasoning(ReasoningEffort::Low),
            );
            assert_eq!(value["reasoning_effort"], "low", "{model}");
            assert!(warnings.is_empty(), "{model}: {warnings:?}");
        }
        let (value, _) = encode(&AnyChatRequest::new(MODEL).user("x"));
        assert!(value.get("reasoning_effort").is_none());
    }

    /// Images are sent as OpenAI-style parts, which vision models take, for
    /// any model (ADR 0015); text alone stays a string.
    #[test]
    fn user_images_are_sent_as_parts() {
        let request = AnyChatRequest::new(MODEL).message(AnyMessage::user_parts(vec![
            ContentPart::text("What is this?"),
            ContentPart::Image(
                Image::url("https://example.com/cat.png").with_detail(ImageDetail::Low),
            ),
            ContentPart::Image(Image::base64("image/png", "iVBORw0KGgo=")),
        ]));
        let (value, warnings) = encode(&request);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(
            value["messages"][0]["content"],
            json!([
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "low"}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            ])
        );
        let (value, _) = encode(
            &AnyChatRequest::new(MODEL).message(AnyMessage::user_parts(vec![
                ContentPart::text("a"),
                ContentPart::text("b"),
            ])),
        );
        assert_eq!(value["messages"][0]["content"], "a\nb");
    }

    #[test]
    fn user_file_and_other_parts_are_rejected() {
        for part in [
            ContentPart::file_base64("application/pdf", "AAAA"),
            ContentPart::Other(OtherContent::new("audio", serde_json::Map::new())),
        ] {
            let request = AnyChatRequest::new(MODEL)
                .message(AnyMessage::user_parts(vec![ContentPart::text("x"), part]));
            let err = encode_request(&request, false).unwrap_err();
            assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
        }
    }

    /// Another provider's part is dropped with a warning, in a user message
    /// and in a tool result, under every policy.
    #[test]
    fn foreign_parts_are_dropped() {
        let foreign = || {
            ContentPart::Other(
                OtherContent::new("custom_tool_call_output", serde_json::Map::new())
                    .with_origin("openai"),
            )
        };
        for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
            let mut request = AnyChatRequest::new(MODEL)
                .message(AnyMessage::user_parts(vec![
                    ContentPart::text("Weather?"),
                    foreign(),
                ]))
                .message(AnyMessage::assistant_blocks(vec![ContentBlock::tool_call(
                    "c1",
                    "get_weather",
                    "{}",
                )]))
                .message(AnyMessage::tool_result(
                    "c1",
                    "get_weather",
                    vec![ContentPart::text("Sunny"), foreign()],
                ));
            request.unsupported_policy = Some(policy);
            let (value, warnings) = encode(&request);
            assert_eq!(value["messages"][0]["content"], "Weather?");
            assert_eq!(value["messages"][2]["content"], "Sunny");
            let kinds: Vec<WarningKind> = warnings.iter().map(|w| w.kind).collect();
            // Both drops share a code: one warning per request.
            assert_eq!(kinds, [WarningKind::Dropped], "{policy:?}");
        }
    }

    /// A refusal another provider wrote (OpenAI's `refusal`) is replayed as
    /// the assistant text it was, without a warning.
    #[test]
    fn refusals_are_replayed_as_text() {
        let request = AnyChatRequest::new(MODEL)
            .user("Help me.")
            .message(AnyMessage::assistant_blocks(vec![ContentBlock::Refusal(
                allama::Refusal::new("I can't help with that."),
            )]))
            .user("Why not?");
        let (value, warnings) = encode(&request);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(value["messages"][1]["role"], "assistant");
        assert_eq!(value["messages"][1]["content"], "I can't help with that.");
    }

    #[test]
    fn assistant_image_replay_is_rejected() {
        let request = AnyChatRequest::new(MODEL).message(AnyMessage::assistant_blocks(vec![
            ContentBlock::Image(Image::url("https://example.com/cat.png")),
        ]));
        let err = encode_request(&request, false).unwrap_err();
        assert!(err.to_string().contains("assistant images"));
    }

    #[test]
    fn assistant_reasoning_and_other_blocks_are_dropped_with_warnings() {
        let mut signed = Reasoning::new("thinking");
        signed
            .provider_data
            .insert_raw("anthropic", "signature", json!("sig"));
        let request = AnyChatRequest::new(MODEL)
            .user("q")
            .message(AnyMessage::assistant_blocks(vec![
                ContentBlock::Reasoning(signed),
                ContentBlock::text("answer"),
                ContentBlock::Other(
                    OtherContent::new("server_tool_use", serde_json::Map::new())
                        .with_origin("anthropic"),
                ),
            ]));
        let (value, warnings) = encode(&request);
        assert_eq!(value["messages"][1]["content"], "answer");
        assert_eq!(warnings.len(), 2);
        assert!(warnings.iter().all(|w| w.kind == WarningKind::Dropped));
        assert_eq!(
            warnings[0].subject,
            Subject::Content {
                message: 1,
                block: Some(0)
            }
        );
        assert!(warnings[1].message.contains("anthropic"));

        let strict = request.unsupported_policy(UnsupportedPolicy::Error);
        // Dropping replay state is not a request hint; strict mode still succeeds.
        assert!(encode_request(&strict, false).is_ok());
    }

    /// `Other` content Workers AI wrote cannot be sent back: the drop is
    /// content replay state, which recurs with the history. The caller's
    /// own raw content (no origin) stays uncoded (ADR 0003).
    #[test]
    fn own_other_content_is_dropped_as_content_replay() {
        let request = AnyChatRequest::new(MODEL)
            .user("q")
            .message(AnyMessage::assistant_blocks(vec![
                ContentBlock::text("answer"),
                ContentBlock::Other(
                    OtherContent::new("citation", serde_json::Map::new()).with_origin(NAMESPACE),
                ),
                ContentBlock::Other(OtherContent::new("note", serde_json::Map::new())),
            ]));
        let (_, warnings) = encode(&request);
        assert_eq!(warnings.len(), 2, "{warnings:?}");
        let own = &warnings[0];
        assert_eq!(own.kind, WarningKind::Dropped);
        assert_eq!(
            own.subject,
            Subject::Content {
                message: 1,
                block: Some(1)
            }
        );
        assert_eq!(
            own.message,
            "Workers AI cannot replay `citation` content; it was dropped"
        );
        assert_eq!(own.code.as_deref(), Some("dropped.content_replay"));
        assert_eq!(own.replay_state(), Some(ReplayState::Content));
        assert_eq!(own.provider.as_deref(), Some(PROVIDER_NAME));
        let callers = &warnings[1];
        assert_eq!(
            callers.message,
            "Workers AI cannot replay `note` content; it was dropped"
        );
        assert_eq!(callers.replay_state(), None);
    }

    #[test]
    fn empty_turns_are_skipped_without_a_warning() {
        let baseline = AnyChatRequest::new(MODEL)
            .user("one")
            .assistant("answer")
            .user("two");
        let reasoning_only =
            AnyMessage::assistant_blocks(vec![ContentBlock::Reasoning(Reasoning::new("thinking"))]);
        let with_empty = AnyChatRequest::new(MODEL)
            .user("one")
            .message(AnyMessage::user(""))
            .message(AnyMessage::assistant_blocks(vec![]))
            .assistant("answer")
            .message(reasoning_only)
            .message(AnyMessage::user_parts(vec![]))
            .user("two");
        let (expected, _) = encode(&baseline);
        let (value, warnings) = encode(&with_empty);
        assert_eq!(value, expected);
        // Only the reasoning drop is reported, at the caller's index.
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert_eq!(
            warnings[0].subject,
            Subject::Content {
                message: 4,
                block: Some(0)
            }
        );
    }

    #[test]
    fn a_reply_left_last_by_skipped_turns_is_dropped_under_every_policy() {
        for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
            let request = AnyChatRequest::new(MODEL)
                .user("one")
                .assistant("answer")
                .message(AnyMessage::user(""))
                .unsupported_policy(policy);
            let (value, warnings) = encode(&request);
            assert_eq!(
                value["messages"],
                json!([{"role": "user", "content": "one"}])
            );
            assert_eq!(warnings.len(), 1, "{warnings:?}");
            assert_eq!(warnings[0].kind, WarningKind::Dropped);
        }
        let err = encode_request(
            &AnyChatRequest::new(MODEL).message(AnyMessage::user("")),
            false,
        )
        .unwrap_err();
        assert!(matches!(err, allama::Error::InvalidRequest { .. }), "{err}");
    }

    /// Mistral models refuse tool call ids other than nine letters and
    /// digits (verified live); ids from elsewhere are mapped for them, the
    /// same way for the call and its result. Other models get ids as they
    /// are.
    #[test]
    fn mistral_models_get_nine_character_tool_call_ids() {
        let request = |model: &str| {
            AnyChatRequest::new(model)
                .user("Weather?")
                .message(AnyMessage::assistant_blocks(vec![
                    ContentBlock::tool_call("gemini_r_call_0", "f", "{}"),
                    ContentBlock::tool_call("D681PevKs", "f", "{}"),
                ]))
                .message(AnyMessage::tool_result("D681PevKs", "f", "b"))
                .message(AnyMessage::tool_result("gemini_r_call_0", "f", "a"))
        };
        let (value, _) = encode(&request("@cf/mistralai/mistral-small-3.1-24b-instruct"));
        let calls = &value["messages"][1]["tool_calls"];
        let mapped = calls[0]["id"].as_str().unwrap();
        assert!(
            mapped.len() == 9 && mapped.chars().all(|c| c.is_ascii_alphanumeric()),
            "{mapped}"
        );
        assert_eq!(calls[1]["id"], "D681PevKs");
        assert_eq!(value["messages"][2]["tool_call_id"], "D681PevKs");
        assert_eq!(value["messages"][3]["tool_call_id"], mapped);

        let (value, _) = encode(&request(MODEL));
        assert_eq!(
            value["messages"][1]["tool_calls"][0]["id"],
            "gemini_r_call_0"
        );
        assert_eq!(value["messages"][3]["tool_call_id"], "gemini_r_call_0");
    }

    #[test]
    fn foreign_provider_data_on_tool_calls_is_dropped_with_warning() {
        let mut call = allama::ToolCall::new("call_1", "lookup_weather", r#"{"city":"Paris"}"#);
        call.provider_data
            .insert_raw("google-ai", "thoughtSignature", json!("abc"));
        let request = AnyChatRequest::new(MODEL)
            .user("q")
            .message(AnyMessage::assistant_blocks(vec![ContentBlock::ToolCall(
                call,
            )]))
            .message(AnyMessage::tool_result("call_1", "lookup_weather", "sunny"));
        let (value, warnings) = encode(&request);
        assert_eq!(value["messages"][1]["content"], "");
        assert_eq!(value["messages"][1]["tool_calls"][0]["id"], "call_1");
        assert_eq!(
            value["messages"][1]["tool_calls"][0]["function"]["arguments"],
            r#"{"city":"Paris"}"#
        );
        assert_eq!(value["messages"][2]["role"], "tool");
        assert_eq!(value["messages"][2]["tool_call_id"], "call_1");
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].message.contains("google-ai"));
    }

    #[test]
    fn tool_errors_and_objects_are_approximated_as_text() {
        let mut object = serde_json::Map::new();
        object.insert("temp".into(), json!(21));
        let request = AnyChatRequest::new(MODEL)
            .user("q")
            .message(AnyMessage::tool_result(
                "a",
                "lookup_weather",
                allama::ToolResultContent::Object(object),
            ))
            .message(AnyMessage::tool_error("b", "lookup_weather", "boom"));
        let (value, warnings) = encode(&request);
        assert_eq!(value["messages"][1]["content"], r#"{"temp":21}"#);
        assert_eq!(value["messages"][2]["content"], "boom");
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].kind, WarningKind::Approximated);
        assert_eq!(
            warnings[0].code.as_deref(),
            Some("approximated.tool_result_error")
        );

        assert_eq!(
            warnings[0].message,
            "Workers AI has no tool error flag; the error text was sent as the tool result; the model still reads the error text; to send your own encoding to this endpoint, use `push_tool_result`"
        );

        // Sent under a strict policy too: the model still reads the error
        // text.
        let (value, warnings) = encode(&request.unsupported_policy(UnsupportedPolicy::Error));
        assert_eq!(value["messages"][2]["content"], "boom");
        assert_eq!(warnings.len(), 1);
        assert_eq!(
            warnings[0].code.as_deref(),
            Some("approximated.tool_result_error")
        );
    }

    /// Regression: the text parts of one tool result were concatenated
    /// with no separator.
    #[test]
    fn multi_part_tool_results_keep_text_boundaries() {
        let request = AnyChatRequest::new(MODEL)
            .user("q")
            .message(AnyMessage::tool_result(
                "a",
                "lookup_weather",
                allama::ToolResultContent::Parts(vec![
                    ContentPart::text("before"),
                    ContentPart::text("after"),
                ]),
            ));
        let (value, warnings) = encode(&request);
        assert_eq!(value["messages"][1]["content"], "before\n\nafter");
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    #[test]
    fn tool_call_response_uses_deterministic_ids_scoped_to_the_turn() {
        let raw = json!({
            "response": null,
            "tool_calls": [
                {"name": "lookup_weather", "arguments": {"city": "London"}},
                {"name": "lookup_weather", "arguments": {"city": "Paris"}}
            ]
        });
        let first = decode(raw.clone());
        let second = decode(raw.clone());
        let ids: Vec<_> = first.tool_calls().map(|c| c.id.clone()).collect();
        assert_eq!(ids, vec!["cf_call_3_0", "cf_call_3_1"]);
        // Same input, same ids: no hidden counter.
        assert_eq!(
            ids,
            second
                .tool_calls()
                .map(|c| c.id.clone())
                .collect::<Vec<_>>()
        );
        // A later turn gets distinct ids.
        let later = decode_response(raw, 5).unwrap();
        assert_eq!(later.tool_calls().next().unwrap().id, "cf_call_5_0");
        assert_eq!(first.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(first.finish_reason_raw, None);
    }

    #[test]
    fn upstream_tool_call_ids_and_openai_shape_are_preserved() {
        let response = decode(json!({
            "response": "",
            "tool_calls": [{
                "id": "chatcmpl-tool-123",
                "type": "function",
                "function": {"name": "lookup_weather", "arguments": "{\"city\":\"Rome\"}"}
            }],
            "finish_reason": "stop"
        }));
        let call = response.tool_calls().next().unwrap();
        assert_eq!(call.id, "chatcmpl-tool-123");
        assert_eq!(call.arguments, r#"{"city":"Rome"}"#);
        // "stop" with tool calls is normalized, the raw value kept.
        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(response.finish_reason_raw.as_deref(), Some("stop"));
    }

    #[test]
    fn malformed_string_arguments_are_kept_verbatim() {
        let response = decode(json!({
            "tool_calls": [{"name": "f", "arguments": "{not json"}]
        }));
        let call = response.tool_calls().next().unwrap();
        assert_eq!(call.arguments, "{not json");
        assert!(!call.has_valid_arguments());
    }

    #[test]
    fn nameless_tool_call_is_an_unexpected_response() {
        let err = decode_response(json!({"tool_calls": [{"arguments": {}}]}), 0).unwrap_err();
        assert!(matches!(err, Error::UnexpectedResponse { .. }));
    }

    #[test]
    fn finish_reason_is_used_when_present() {
        let response = decode(json!({"response": "trunc", "finish_reason": "length"}));
        assert_eq!(response.finish_reason, Some(FinishReason::Length));
        assert_eq!(response.finish_reason_raw.as_deref(), Some("length"));

        let response = decode(json!({"response": "x", "finish_reason": "weird"}));
        assert_eq!(
            response.finish_reason,
            Some(FinishReason::Other("weird".into()))
        );

        let response = decode(json!({"response": "x"}));
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(response.finish_reason_raw, None);
        assert_eq!(response.provider.as_deref(), Some("cloudflare"));
    }

    #[test]
    fn chat_completion_shape_is_decoded() {
        let response = decode(json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "@cf/qwen/qwen3-30b-a3b-fp8",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "42",
                    "reasoning_content": "Let me think.",
                    "tool_calls": []
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }));
        assert_eq!(response.id.as_deref(), Some("chatcmpl-1"));
        assert_eq!(
            response.model.as_deref(),
            Some("@cf/qwen/qwen3-30b-a3b-fp8")
        );
        assert_eq!(response.reasoning_text().as_deref(), Some("Let me think."));
        assert_eq!(response.text().as_deref(), Some("42"));
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(response.usage.unwrap().total_tokens, Some(15));
    }

    /// The adapter never asks for several choices; the first is the
    /// response, as on the compat crate's preset.
    #[test]
    fn the_first_choice_is_the_response() {
        let response = decode(json!({
            "choices": [
                {"index": 0, "message": {"content": "a"}, "finish_reason": "stop"},
                {"index": 1, "message": {"content": "b"}, "finish_reason": "stop"}
            ]
        }));
        assert_eq!(response.text().as_deref(), Some("a"));
    }

    /// The binding's mixed output (recorded from
    /// `@cf/meta/llama-3.1-8b-instruct`, 2026-10-07) repeats the choice in
    /// `response` and `tool_calls`; the text is decoded once, and the
    /// `neurons` counter is provider data.
    #[test]
    fn mixed_output_is_decoded_once() {
        let response = decode(json!({
            "choices": [{"finish_reason": "length", "index": 0, "logprobs": null,
                "message": {"annotations": null, "audio": null, "content": "Hello, how can I",
                    "function_call": null, "reasoning": null, "refusal": null, "role": "assistant"},
                "routed_experts": null, "stop_reason": null, "token_ids": null}],
            "created": 1791353021, "id": "chatcmpl-2a52", "model": "@cf/meta/llama-3.1-8b-fast-v2",
            "object": "chat.completion", "response": "Hello, how can I", "tool_calls": [],
            "usage": {"prompt_tokens": 37, "completion_tokens": 5, "total_tokens": 42,
                "prompt_tokens_details": {"cached_tokens": 0}, "neurons": 0.326_725_929_975_509_64}
        }));
        assert_eq!(response.text().as_deref(), Some("Hello, how can I"));
        assert_eq!(response.content.len(), 1);
        assert_eq!(response.finish_reason, Some(FinishReason::Length));
        assert!(
            response
                .provider_data
                .get_raw(NAMESPACE, "usage")
                .and_then(|usage| usage.get("neurons"))
                .is_some()
        );
        assert_eq!(response.usage.unwrap().total_tokens, Some(42));
    }

    #[test]
    fn usage_follows_the_contract() {
        let usage = decode(json!({
            "response": "x",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20.0,
                "prompt_tokens_details": {"cached_tokens": 60},
                "completion_tokens_details": {"reasoning_tokens": 5}
            }
        }))
        .usage
        .unwrap();
        assert_eq!(usage.input_tokens, Some(100));
        assert_eq!(usage.output_tokens, Some(20));
        assert_eq!(usage.total_tokens, Some(120));
        assert_eq!(usage.cache_read_input_tokens, Some(60));
        assert_eq!(usage.reasoning_tokens, Some(5));
        assert!(usage.contract_violation().is_none());

        // A placeholder total of 0 is replaced by the derived total.
        let usage = decode(json!({
            "response": "x",
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 0}
        }))
        .usage
        .unwrap();
        assert_eq!(usage.total_tokens, Some(7));

        assert!(
            decode(json!({"response": "x", "usage": {}}))
                .usage
                .is_none()
        );
    }

    #[test]
    fn structured_json_response_is_serialized_into_text_content() {
        let response = decode(json!({
            "response": {"token": "cf-json-ok", "summary": "Cloudflare JSON mode works."}
        }));
        let parsed: serde_json::Value = serde_json::from_str(&response.text_or_empty()).unwrap();
        assert_eq!(
            parsed,
            json!({"token": "cf-json-ok", "summary": "Cloudflare JSON mode works."})
        );
    }

    /// The finish reasons and their aliases map as on the compat
    /// crate's preset, the decoder being the same.
    #[test]
    fn finish_reason_mapping() {
        for (raw, expected) in [
            ("stop", FinishReason::Stop),
            ("eos", FinishReason::Stop),
            ("end_turn", FinishReason::Stop),
            ("length", FinishReason::Length),
            ("model_length", FinishReason::Length),
            ("max_tokens", FinishReason::Length),
            ("tool_calls", FinishReason::ToolCalls),
            ("content_filter", FinishReason::ContentFilter),
            ("error", FinishReason::Other("error".into())),
        ] {
            let response = decode(json!({"response": "x", "finish_reason": raw}));
            assert_eq!(response.finish_reason, Some(expected), "{raw}");
        }
    }

    /// Replay state stays in a stored conversation, so a conversation that
    /// moved here from another provider drops it on every turn: each drop is
    /// coded, so its warning event goes out at `WARN` once per model, not once
    /// per turn, and points to `retain_provider_state`.
    #[test]
    fn a_long_foreign_reasoning_history_drops_with_codes() {
        let mut request = AnyChatRequest::new(MODEL);
        for turn in 0..10 {
            request =
                request
                    .user(format!("question {turn}"))
                    .message(AnyMessage::assistant_blocks(vec![
                        ContentBlock::Reasoning(Reasoning::new("thoughts").with_provider_data(
                            ProviderData::new().with_raw("openai", "signature", json!("opaque")),
                        )),
                        ContentBlock::text("answer"),
                    ]));
        }
        let (_, warnings) = encode(&request.user("next"));
        // One warning per code per request, listing the ten messages.
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(
            matches!(&warnings[0].subject, Subject::Messages { messages } if messages.len() == 10)
        );
        for warning in &warnings {
            assert_eq!(warning.code.as_deref(), Some("dropped.foreign_reasoning"));
            assert!(
                warning.message.contains("retain_provider_state"),
                "{}",
                warning.message
            );
        }
    }
}
