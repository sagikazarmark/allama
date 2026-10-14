//! Chat Completions wire types and conversions.

use std::borrow::Cow;

use allama::adapter::chat::{EncodedTurn, ToolCallIdFormat, retain_turns};
use allama::adapter::{Degradations, describe_origins};
use allama::{
    AssistantMessage, CacheRetention, ChatRequest, ChatResponse, ContentBlock, ContentPart, Error,
    ExtraMap, File, FinishReason, MediaSource, Message, OtherContent, Reasoning, ReasoningOutput,
    Refusal, ReplayState, RequestField, ResponseFormat, Subject, Text, ToolCall, ToolChoice,
    ToolMessage, ToolResultContent, Usage, UserContent, UserMessage, Warning,
};
use serde::{Deserialize, Serialize};

use crate::client_tools::{self, CustomToolCall, Pairing};
use crate::model_rules::ChatModelRules;
use crate::options::{SystemOptions, SystemRole, WireOptions, refuse_owned_fields};
use crate::quirks::{
    CompatQuirks, MaxTokensField, ReasoningReplayField, SchemaConstraint, SeedField,
};
use crate::schema::{EncodedSchema, encode_constrained};

const TOOL_TYPE_FUNCTION: &str = "function";
const TOOL_TYPE_CUSTOM: &str = "custom";

/// The code of the warning for a tool error sent to an endpoint with no
/// error flag, on Chat Completions and Responses alike.
pub(crate) const TOOL_RESULT_ERROR: &str = "approximated.tool_result_error";

/// The code of the warning for tool call arguments that are not a JSON
/// object, sent as `{}`: history repeats the call on every turn.
pub(crate) const TOOL_CALL_ARGUMENTS: &str = "approximated.tool_call_arguments";

/// The content part types of Chat Completions messages. An `Other` part of
/// the dialect's own origin is sent only for these: Responses shares the
/// `openai` namespace, and its input items are not Chat content.
const CONTENT_PART_TYPES: &[&str] = &["text", "image_url", "input_audio", "file"];

/// A Chat Completions request body.
#[derive(Debug, Serialize)]
#[non_exhaustive]
pub(crate) struct ChatCompletionRequest {
    /// Model id.
    pub model: String,
    /// Messages.
    pub messages: Vec<ApiMessage>,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Nucleus sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Top-k sampling (servers with
    /// [`supports_top_k`](CompatQuirks::supports_top_k)).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Legacy output cap.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Output cap including reasoning (OpenAI).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Frequency penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    /// Presence penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// Tool definitions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ApiTool>>,
    /// Tool choice.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    /// Response format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<serde_json::Value>,
    /// Seed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Reasoning effort.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// Parallel tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// End-user id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Stable end-user identifier for abuse detection (OpenAI's successor
    /// to `user`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    /// Service tier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    /// Store upstream.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ExtraMap>,
    /// Prompt cache key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    /// Prompt cache retention (`in_memory` or `24h`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<String>,
    /// Streaming.
    pub stream: bool,
    /// Stream options.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    /// Endpoint-specific fields, checked against
    /// [`CHAT_COMPLETION_FIELDS`].
    #[serde(flatten, skip_serializing_if = "ExtraMap::is_empty")]
    pub extra: ExtraMap,
}

/// The body fields of [`ChatCompletionRequest`] (and the seed's other
/// spelling): the adapter owns them, so `ChatRequestOptions::extra` cannot
/// set them (`top_k` only for endpoints with
/// [`supports_top_k`](CompatQuirks::supports_top_k)).
pub(crate) const CHAT_COMPLETION_FIELDS: &[&str] = &[
    "frequency_penalty",
    "max_completion_tokens",
    "max_tokens",
    "messages",
    "metadata",
    "model",
    "parallel_tool_calls",
    "presence_penalty",
    "prompt_cache_key",
    "prompt_cache_retention",
    "random_seed",
    "reasoning_effort",
    "response_format",
    "safety_identifier",
    "seed",
    "service_tier",
    "stop",
    "store",
    "stream",
    "stream_options",
    "temperature",
    "tool_choice",
    "tools",
    "top_k",
    "top_p",
    "user",
];

/// Stream options.
#[derive(Debug, Serialize)]
#[non_exhaustive]
pub(crate) struct StreamOptions {
    /// Emit a final usage chunk.
    pub include_usage: bool,
}

/// A wire message.
#[derive(Debug, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub(crate) struct ApiMessage {
    /// Role.
    pub role: String,
    /// Content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
    /// Participant name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Assistant tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ApiToolCall>>,
    /// Tool result correlation id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Assistant reasoning text (dialects that require it back in
    /// `reasoning_content`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Assistant reasoning text (dialects that require it back in
    /// `reasoning`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Assistant reasoning details, replayed verbatim (dialects with
    /// [`ReasoningReplayField::ReasoningDetails`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_details: Option<Vec<serde_json::Value>>,
    /// Assistant reasoning text (dialects with
    /// [`ReasoningReplayField::ReasoningText`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_text: Option<String>,
    /// The signature of the assistant reasoning, replayed verbatim
    /// (dialects with [`ReasoningReplayField::ReasoningText`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_opaque: Option<String>,
    /// Assistant refusal text (dialects with
    /// [`replay_refusal`](CompatQuirks::replay_refusal)).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

impl ApiMessage {
    /// Whether the message carries nothing: no content, tool calls, or
    /// reasoning.
    fn is_empty(&self) -> bool {
        let no_content = match &self.content {
            None => true,
            Some(serde_json::Value::String(text)) => text.is_empty(),
            Some(serde_json::Value::Array(parts)) => parts.is_empty(),
            Some(_) => false,
        };
        no_content
            && self.tool_calls.is_none()
            && self.reasoning_content.is_none()
            && self.reasoning.is_none()
            && self.reasoning_details.is_none()
            && self.reasoning_text.is_none()
            && self.reasoning_opaque.is_none()
            && self.refusal.is_none()
    }
}

/// A wire tool call: a function call, or a call to a custom tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub(crate) struct ApiToolCall {
    /// Call id. Some endpoints send none (or an empty one); the decoder
    /// makes one up.
    #[serde(default)]
    pub id: String,
    /// `function` or `custom`.
    #[serde(rename = "type", default = "function_type")]
    pub call_type: String,
    /// Function name and arguments (`function` calls).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<ApiToolCallFunction>,
    /// Custom tool name and freeform input (`custom` calls).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom: Option<ApiCustomToolCall>,
    /// Fields this crate does not model, such as Gemini's
    /// `extra_content.google.thought_signature`: kept in the call's
    /// provider data, and replayed to the endpoint they came from.
    #[serde(flatten, default, skip_serializing_if = "ExtraMap::is_empty")]
    pub extra: ExtraMap,
}

/// The tool call fields the Chat Completions wire models, and the keys the
/// Responses decoder writes to a tool call's provider data under the shared
/// namespace: neither is an unmodeled field to replay.
const TOOL_CALL_FIELDS: &[&str] = &[
    "custom", "function", "id", "index", "item_id", "status", "type",
];

fn function_type() -> String {
    TOOL_TYPE_FUNCTION.to_owned()
}

/// A wire function call.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub(crate) struct ApiToolCallFunction {
    /// Function name.
    pub name: String,
    /// Raw JSON arguments.
    #[serde(default)]
    pub arguments: String,
}

/// A wire custom tool call: the model's freeform text input.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub(crate) struct ApiCustomToolCall {
    /// Name of the custom tool.
    pub name: String,
    /// The model's freeform input.
    #[serde(default)]
    pub input: String,
}

/// A wire tool definition.
#[derive(Debug, Serialize)]
#[non_exhaustive]
pub(crate) struct ApiTool {
    /// `function`, or the type of a client tool definition (`custom`).
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function definition (`function` tools).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<ApiFunctionDef>,
    /// The other fields of a client tool definition (a custom tool's
    /// `custom` object), sent verbatim.
    #[serde(flatten, skip_serializing_if = "ExtraMap::is_empty")]
    pub extra: ExtraMap,
}

/// A wire function definition.
#[derive(Debug, Serialize)]
#[non_exhaustive]
pub(crate) struct ApiFunctionDef {
    /// Name.
    pub name: String,
    /// Description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Parameter schema.
    pub parameters: serde_json::Value,
    /// Strict mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// A Chat Completions response.
#[derive(Debug, Deserialize)]
#[non_exhaustive]
pub(crate) struct ChatCompletionResponse {
    /// Response id.
    #[serde(default)]
    pub id: Option<String>,
    /// Choices (only the first is used).
    pub choices: Vec<ChatCompletionChoice>,
    /// Model.
    #[serde(default)]
    pub model: Option<String>,
    /// Usage.
    #[serde(default)]
    pub usage: Option<ApiUsage>,
    /// System fingerprint.
    #[serde(default)]
    pub system_fingerprint: Option<String>,
    /// Service tier that served the request.
    #[serde(default)]
    pub service_tier: Option<String>,
    /// Top-level fields this crate does not model (vendor additions such as
    /// `citations`). [`from_api_response`] keeps them in the response's
    /// provider data.
    #[serde(flatten)]
    pub extra: ExtraMap,
}

/// A response choice.
#[derive(Debug, Deserialize)]
#[non_exhaustive]
pub(crate) struct ChatCompletionChoice {
    /// Choice index.
    #[serde(default)]
    pub index: u32,
    /// Message.
    #[serde(default, deserialize_with = "null_as_default")]
    pub message: ChatCompletionMessage,
    /// Finish reason.
    #[serde(default)]
    pub finish_reason: Option<String>,
    /// Choice fields this crate does not model (`logprobs`, OpenRouter's
    /// `native_finish_reason`, Azure's `content_filter_results`): kept in
    /// the response's provider data.
    #[serde(flatten)]
    pub extra: ExtraMap,
}

/// A response message.
#[derive(Debug, Default, Deserialize)]
#[non_exhaustive]
pub(crate) struct ChatCompletionMessage {
    /// Content: a string, an array of content parts (Mistral's `text` and
    /// `thinking` parts), or a JSON object some endpoints return for
    /// structured output.
    #[serde(default)]
    pub content: Option<serde_json::Value>,
    /// Tool calls.
    #[serde(default)]
    pub tool_calls: Option<Vec<ApiToolCall>>,
    /// Refusal text.
    #[serde(default)]
    pub refusal: Option<String>,
    /// Reasoning text (DeepSeek, vLLM, …).
    #[serde(default)]
    pub reasoning_content: Option<String>,
    /// Reasoning text (OpenRouter, Groq, …).
    #[serde(default)]
    pub reasoning: Option<String>,
    /// Structured reasoning (OpenRouter): text, summaries and encrypted
    /// reasoning, with their signatures.
    #[serde(default)]
    pub reasoning_details: Option<Vec<serde_json::Value>>,
    /// Reasoning text (GitHub Copilot).
    #[serde(default)]
    pub reasoning_text: Option<String>,
    /// The signature of the reasoning (GitHub Copilot).
    #[serde(default)]
    pub reasoning_opaque: Option<String>,
    /// Annotations (e.g. URL citations).
    #[serde(default)]
    pub annotations: Option<Vec<serde_json::Value>>,
    /// Message fields this crate does not model: kept in the response's
    /// provider data.
    #[serde(flatten)]
    pub extra: ExtraMap,
}

/// `null` read as the type's default.
fn null_as_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

/// Usage counters.
#[derive(Debug, Clone, Deserialize, Default)]
#[non_exhaustive]
pub(crate) struct ApiUsage {
    /// Prompt tokens (includes cached tokens).
    #[serde(default, deserialize_with = "null_as_default")]
    pub prompt_tokens: u64,
    /// Completion tokens (includes reasoning).
    #[serde(default, deserialize_with = "null_as_default")]
    pub completion_tokens: u64,
    /// Total.
    #[serde(default, deserialize_with = "null_as_default")]
    pub total_tokens: u64,
    /// Prompt token breakdown.
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// Completion token breakdown.
    #[serde(default)]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    /// DeepSeek's prompt cache hits: the tokens read from the cache.
    #[serde(default)]
    pub prompt_cache_hit_tokens: Option<u64>,
    /// Vendor counters this crate does not model (OpenRouter's `cost`,
    /// Workers AI's `neurons`): kept in the response's provider data under
    /// [`USAGE_KEY`].
    #[serde(flatten)]
    pub extra: ExtraMap,
}

/// The provider-data key of the vendor usage counters that are neither
/// token counts nor tool requests ([`ApiUsage::extra`]).
pub(crate) const USAGE_KEY: &str = "usage";

/// Prompt token breakdown.
#[derive(Debug, Clone, Deserialize, Default)]
#[non_exhaustive]
pub(crate) struct PromptTokensDetails {
    /// Tokens read from the prompt cache.
    #[serde(default)]
    pub cached_tokens: Option<u64>,
    /// Tokens written to the prompt cache.
    #[serde(default)]
    pub cache_write_tokens: Option<u64>,
    /// Audio input tokens.
    #[serde(default)]
    pub audio_tokens: Option<u64>,
}

/// Completion token breakdown.
#[derive(Debug, Clone, Deserialize, Default)]
#[non_exhaustive]
pub(crate) struct CompletionTokensDetails {
    /// Reasoning tokens.
    #[serde(default)]
    pub reasoning_tokens: Option<u64>,
    /// Audio output tokens.
    #[serde(default)]
    pub audio_tokens: Option<u64>,
}

/// A streaming chunk.
#[derive(Debug, Deserialize)]
#[non_exhaustive]
pub(crate) struct ChatCompletionChunk {
    /// Response id.
    #[serde(default)]
    pub id: Option<String>,
    /// Model.
    #[serde(default)]
    pub model: Option<String>,
    /// Choices.
    #[serde(default)]
    pub choices: Vec<ChunkChoice>,
    /// Final usage (when requested).
    #[serde(default)]
    pub usage: Option<ApiUsage>,
    /// System fingerprint.
    #[serde(default)]
    pub system_fingerprint: Option<String>,
    /// Top-level fields this crate does not model (vendor additions such as
    /// `citations`). The stream decoder surfaces them as
    /// `ResponseProviderData` events.
    #[serde(flatten)]
    pub extra: ExtraMap,
}

/// A streaming choice.
#[derive(Debug, Deserialize)]
#[non_exhaustive]
pub(crate) struct ChunkChoice {
    /// Choice index.
    #[serde(default)]
    pub index: u32,
    /// Delta.
    #[serde(default, deserialize_with = "null_as_default")]
    pub delta: ChunkDelta,
    /// Finish reason.
    #[serde(default)]
    pub finish_reason: Option<String>,
    /// Choice fields this crate does not model (`logprobs`, …).
    #[serde(flatten)]
    pub extra: ExtraMap,
}

/// A streaming delta.
#[derive(Debug, Deserialize, Default)]
#[non_exhaustive]
pub(crate) struct ChunkDelta {
    /// Content: a string, or (Mistral) an array of `text` and `thinking`
    /// parts.
    #[serde(default)]
    pub content: Option<serde_json::Value>,
    /// Refusal text.
    #[serde(default)]
    pub refusal: Option<String>,
    /// Reasoning text.
    #[serde(default)]
    pub reasoning_content: Option<String>,
    /// Reasoning text (alternative field name).
    #[serde(default)]
    pub reasoning: Option<String>,
    /// Structured reasoning fragments (OpenRouter).
    #[serde(default)]
    pub reasoning_details: Option<Vec<serde_json::Value>>,
    /// Reasoning text (GitHub Copilot).
    #[serde(default)]
    pub reasoning_text: Option<String>,
    /// The signature of the reasoning so far, whole (GitHub Copilot).
    #[serde(default)]
    pub reasoning_opaque: Option<String>,
    /// Tool call fragments.
    #[serde(default)]
    pub tool_calls: Option<Vec<ChunkToolCall>>,
    /// URL citation annotations (search models send them in a final delta,
    /// once the text is complete).
    #[serde(default)]
    pub annotations: Option<Vec<serde_json::Value>>,
    /// Delta fields this crate does not model.
    #[serde(flatten)]
    pub extra: ExtraMap,
}

/// A tool call fragment.
#[derive(Debug, Deserialize)]
#[non_exhaustive]
pub(crate) struct ChunkToolCall {
    /// Tool call index within the choice. Some endpoints omit it (and send
    /// each parallel call whole, with its own id).
    #[serde(default)]
    pub index: Option<u32>,
    /// Call id (first fragment).
    #[serde(default)]
    pub id: Option<String>,
    /// `function` or `custom` (first fragment).
    #[serde(rename = "type", default)]
    pub call_type: Option<String>,
    /// Function fragment.
    #[serde(default)]
    pub function: Option<ChunkToolCallFunction>,
    /// Custom tool call fragment.
    #[serde(default)]
    pub custom: Option<ChunkCustomToolCall>,
    /// Fields this crate does not model (Gemini's `extra_content`).
    #[serde(flatten)]
    pub extra: ExtraMap,
}

/// A custom tool call fragment.
#[derive(Debug, Deserialize)]
#[non_exhaustive]
pub(crate) struct ChunkCustomToolCall {
    /// Name (first fragment).
    #[serde(default)]
    pub name: Option<String>,
    /// Input fragment.
    #[serde(default)]
    pub input: Option<String>,
}

/// A function fragment.
#[derive(Debug, Deserialize)]
#[non_exhaustive]
pub(crate) struct ChunkToolCallFunction {
    /// Name (first fragment).
    #[serde(default)]
    pub name: Option<String>,
    /// Arguments fragment.
    #[serde(default)]
    pub arguments: Option<String>,
}

/// Error details.
#[derive(Debug, Deserialize, Clone)]
#[non_exhaustive]
pub(crate) struct ApiError {
    /// Message.
    #[serde(default)]
    pub message: String,
    /// Error type.
    #[serde(rename = "type", default)]
    pub error_type: Option<String>,
    /// Error code.
    #[serde(default, deserialize_with = "deserialize_code")]
    pub code: Option<String>,
    /// Offending parameter.
    #[serde(default)]
    pub param: Option<String>,
}

impl ApiError {
    /// An error with `message` and no code, type, or parameter.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            error_type: None,
            code: None,
            param: None,
        }
    }

    /// Set the error code.
    #[must_use]
    #[cfg(test)]
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// Set the error type.
    #[must_use]
    #[cfg(test)]
    pub fn with_type(mut self, error_type: impl Into<String>) -> Self {
        self.error_type = Some(error_type.into());
        self
    }

    /// Set the offending parameter.
    #[must_use]
    #[cfg(test)]
    pub fn with_param(mut self, param: impl Into<String>) -> Self {
        self.param = Some(param.into());
        self
    }
}

/// The fields of a recognised error envelope
/// ([`allama_http::error_envelope`]).
impl From<allama_http::ErrorEnvelope> for ApiError {
    fn from(envelope: allama_http::ErrorEnvelope) -> Self {
        Self {
            message: envelope.message.unwrap_or_default(),
            error_type: envelope.error_type,
            code: envelope.code,
            param: envelope.param,
        }
    }
}

pub(crate) fn deserialize_code<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<String>, D::Error> {
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(value.and_then(|v| match v {
        serde_json::Value::String(s) => Some(s),
        serde_json::Value::Null => None,
        other => Some(other.to_string()),
    }))
}

/// Who the response is attributed to, for stamping responses and events.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub(crate) struct ResponseContext {
    /// Provider identity (`provider_name()`).
    pub provider: Cow<'static, str>,
    /// Provider-data namespace.
    pub namespace: Cow<'static, str>,
    /// Request id from transport headers.
    pub request_id: Option<String>,
    /// Warnings produced while mapping the request.
    pub warnings: Vec<Warning>,
}

#[cfg(test)]
impl ResponseContext {
    /// A context for `provider`, whose state lives under `namespace`, with
    /// no request id and no warnings.
    #[must_use]
    pub fn new(
        provider: impl Into<Cow<'static, str>>,
        namespace: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self {
            provider: provider.into(),
            namespace: namespace.into(),
            request_id: None,
            warnings: Vec::new(),
        }
    }
}

/// Standard top-level response and chunk fields that carry nothing worth
/// keeping: object tags, creation timestamps, and OpenAI's stream padding.
/// Every other unmodeled top-level field is kept as provider data.
const IGNORED_TOP_LEVEL_FIELDS: &[&str] = &["object", "created", "obfuscation"];

/// Whether an unmodeled field is worth keeping as provider data: not null,
/// not an empty object or array, and not one of the
/// [ignored fields](IGNORED_TOP_LEVEL_FIELDS).
pub(crate) fn keep_extra_field(key: &str, value: &serde_json::Value) -> bool {
    let empty = match value {
        serde_json::Value::Null => true,
        serde_json::Value::Object(object) => object.is_empty(),
        serde_json::Value::Array(array) => array.is_empty(),
        _ => false,
    };
    !empty && !IGNORED_TOP_LEVEL_FIELDS.contains(&key)
}

/// How the endpoint constrains the arguments of the tool `name` under
/// `choice`: its [`tool_constraint`](CompatQuirks::tool_constraint), or
/// [`Always`](SchemaConstraint::Always) for a tool it does not otherwise
/// constrain when a forced choice that constrains the forced call
/// ([`forced_tool_choice_constrained`](CompatQuirks::forced_tool_choice_constrained))
/// forces it.
fn tool_constraint(
    quirks: &CompatQuirks,
    choice: Option<&ToolChoice>,
    name: &str,
) -> SchemaConstraint {
    let forced = match choice {
        Some(ToolChoice::Required) => true,
        Some(ToolChoice::Specific { name: forced }) => forced == name,
        _ => false,
    };
    if forced
        && quirks.forced_tool_choice_constrained
        && quirks.tool_constraint == SchemaConstraint::Never
    {
        SchemaConstraint::Always
    } else {
        quirks.tool_constraint
    }
}

/// Convert a portable request to a Chat Completions body, for a model
/// without rules of its own.
///
/// Returns the body and the warnings for degraded features.
///
/// Custom tool calls (`custom_tool_call` content of the dialect namespace,
/// ADR 0014) are replayed as `custom` tool calls, and their answers lifted
/// into `tool` messages. `Other` input parts of the dialect namespace are
/// sent only for Chat's own content types; other providers' parts, and the
/// Responses-only items that share the namespace, are dropped with
/// warnings.
///
/// # Errors
///
/// Fails for semantic features the dialect cannot express, and for hints
/// under `UnsupportedPolicy::Error`.
#[cfg(test)]
pub(crate) fn to_chat_completion_request(
    request: &ChatRequest,
    stream: bool,
    options: &WireOptions,
    quirks: &CompatQuirks,
    provider: impl Into<Cow<'static, str>>,
) -> allama::Result<(ChatCompletionRequest, Vec<Warning>)> {
    encode_chat_completion(
        request,
        stream,
        options,
        quirks,
        &ChatModelRules::new(),
        provider,
    )
}

/// [`to_chat_completion_request`] for a model described by `rules`.
pub(crate) fn encode_chat_completion(
    request: &ChatRequest,
    stream: bool,
    options: &WireOptions,
    quirks: &CompatQuirks,
    rules: &ChatModelRules,
    provider: impl Into<Cow<'static, str>>,
) -> allama::Result<(ChatCompletionRequest, Vec<Warning>)> {
    let provider = provider.into();
    // `top_k` is the adapter's only where the endpoint takes it; elsewhere
    // `extra` may still send it.
    let owned: Vec<&str> = CHAT_COMPLETION_FIELDS
        .iter()
        .copied()
        .filter(|field| quirks.supports_top_k || *field != "top_k")
        .collect();
    refuse_owned_fields(
        &options.extra,
        &owned,
        "ChatRequestOptions::extra",
        &provider,
    )?;
    let mut d = Degradations::new(provider, request.unsupported_policy.unwrap_or_default());
    let ns = quirks.namespace.as_ref();

    let mut messages = Vec::with_capacity(request.system.len() + request.messages.len());
    for (index, prompt) in request.system.iter().enumerate() {
        if prompt.content.is_empty() {
            continue;
        }
        let role = match prompt.resolve_option::<SystemOptions>()?.role {
            Some(SystemRole::Developer) if quirks.supports_developer_role => "developer",
            Some(SystemRole::Developer) => {
                d.hint(
                    Subject::System { index },
                    "this endpoint has no `developer` role; `developer` was not sent, so the prompt went out as `system`",
                )?;
                "system"
            }
            _ => "system",
        };
        messages.push(ApiMessage {
            role: role.into(),
            content: Some(prompt.content.clone().into()),
            ..ApiMessage::default()
        });
    }
    let turns = encode_messages(&request.messages, quirks, &mut d)?;
    // Skip empty turns, and drop a reply the skips and drops left last.
    messages.extend(
        retain_turns(turns, &mut d, |(turn, _)| *turn)?
            .into_iter()
            .map(|(_, api)| api),
    );
    conform_tool_call_ids(
        &mut messages,
        rules
            .tool_call_id_format
            .unwrap_or(quirks.tool_call_id_format),
    );

    let dialect = quirks.schema_dialect_for(&request.model);
    let mut strict_tool_sent = false;
    let mut api_tools = Vec::new();
    for tool in &request.tools {
        let subject = format!("tool `{}`", tool.name);
        let EncodedSchema {
            schema: parameters,
            strict,
            constrained,
        } = encode_constrained(
            &tool.parameters,
            tool.strict,
            tool_constraint(quirks, request.tool_choice.as_ref(), &tool.name),
            RequestField::Tools,
            &subject,
            || {
                format!(
                    "strict tools are not supported by this endpoint; `{}` was sent without `strict`",
                    tool.name
                )
            },
            (dialect, quirks.dialect_verified),
            &mut d,
        )?;
        if quirks.object_schema_root {
            crate::schema::require_object_root(&parameters, RequestField::Tools, &subject, &d)?;
        }
        strict_tool_sent |= constrained;
        if tool
            .provider_data
            .namespace(ns)
            .is_some_and(|data| !data.is_empty())
        {
            d.dropped(
                Subject::Field {
                    field: RequestField::Tools,
                },
                format!(
                    "Chat Completions function definitions take no provider data; the `{ns}` \
                     provider data of tool `{}` was not sent",
                    tool.name
                ),
            );
        }
        api_tools.push(ApiTool {
            tool_type: TOOL_TYPE_FUNCTION.into(),
            function: Some(ApiFunctionDef {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters,
                strict,
            }),
            extra: ExtraMap::new(),
        });
    }
    // Client tool definitions (custom tools), after the function tools.
    for definition in &options.client_tools {
        let mut extra = definition.clone();
        let Some(serde_json::Value::String(tool_type)) = extra.remove("type") else {
            return Err(Error::invalid_request(
                "a client tool definition needs a string `type`, such as `custom`",
            )
            .authored());
        };
        api_tools.push(ApiTool {
            tool_type,
            function: None,
            extra,
        });
    }
    let mut tools = (!api_tools.is_empty()).then_some(api_tools);

    let tool_choice = match &request.tool_choice {
        None => None,
        Some(choice) if !quirks.supports_tool_choice => match choice {
            // The default behavior: omitting it changes nothing.
            ToolChoice::Auto => None,
            // Not offering the tools is exactly "the model must not call
            // tools", so it records nothing, under every policy.
            ToolChoice::Disabled => {
                if tools.take().is_some() {
                    strict_tool_sent = false;
                }
                None
            }
            _ => {
                return Err(d.unsupported(
                    RequestField::ToolChoice,
                    "this endpoint ignores `tool_choice`, so it cannot force a tool call",
                ));
            }
        },
        Some(ToolChoice::Auto) => Some("auto".into()),
        Some(ToolChoice::Required) => Some("required".into()),
        Some(ToolChoice::Disabled) => Some("none".into()),
        Some(ToolChoice::Specific { name }) => {
            Some(serde_json::json!({"type": "function", "function": {"name": name}}))
        }
        Some(_) => {
            return Err(d.unsupported(RequestField::ToolChoice, "unknown tool choice"));
        }
    };

    let response_format = match &request.response_format {
        None => None,
        Some(ResponseFormat::Text) => Some(serde_json::json!({"type": "text"})),
        Some(ResponseFormat::Json) => Some(serde_json::json!({"type": "json_object"})),
        Some(ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
            ..
        }) => {
            if !quirks.supports_json_schema {
                return Err(d.unsupported(
                    RequestField::ResponseFormat,
                    "JSON schema response formats are not supported by this endpoint",
                ));
            }
            let EncodedSchema { schema, strict, .. } = encode_constrained(
                schema,
                *strict,
                rules
                    .response_format_constraint
                    .unwrap_or(quirks.response_format_constraint),
                RequestField::ResponseFormat,
                "the response schema",
                || {
                    match rules.response_format_constraint {
                    Some(_) => format!(
                        "`{}` does not take strict JSON schemas on this endpoint; the schema was sent without `strict`",
                        request.model
                    ),
                    None => "strict JSON schemas are not supported by this endpoint; the schema was sent without `strict`"
                        .to_owned(),
                }
                },
                (dialect, quirks.dialect_verified),
                &mut d,
            )?;
            if quirks.object_schema_root {
                crate::schema::require_object_root(
                    &schema,
                    RequestField::ResponseFormat,
                    "the response schema",
                    &d,
                )?;
            }
            let mut json_schema = serde_json::Map::new();
            json_schema.insert(
                "name".into(),
                name.clone().unwrap_or_else(|| "response".into()).into(),
            );
            json_schema.insert("schema".into(), schema);
            if let Some(strict) = strict {
                json_schema.insert("strict".into(), strict.into());
            }
            Some(serde_json::json!({"type": "json_schema", "json_schema": json_schema}))
        }
        Some(_) => {
            return Err(d.unsupported(RequestField::ResponseFormat, "unknown response format"));
        }
    };

    let mut reasoning_effort = options.reasoning_effort.clone();
    if let Some(config) = &request.reasoning {
        if reasoning_effort.is_none()
            && let Some(requested) = config.effort
        {
            if quirks.supports_reasoning_effort {
                let level = rules
                    .effort_levels
                    .as_deref()
                    .and_then(|levels| requested.nearest(levels))
                    .unwrap_or(requested);
                if level != requested {
                    d.approximated(
                        Subject::Field {
                            field: RequestField::Reasoning,
                        },
                        format!(
                            "`{}` has no reasoning effort `{}`; `{}` was used instead",
                            request.model,
                            requested.as_str(),
                            level.as_str()
                        ),
                    );
                }
                reasoning_effort = Some(level.as_str().into());
            } else {
                d.unsupported_hint(
                    RequestField::Reasoning,
                    "reasoning effort is not supported by this endpoint; the effort was not sent",
                )?;
            }
        }
        // Chat Completions has no output setting: whether reasoning comes
        // back is the endpoint's (OpenAI returns none, DeepSeek always
        // does). An output setting alone never turns reasoning on.
        if let Some(output) = config.output {
            let what = match output {
                ReasoningOutput::Omitted => "cannot be asked to leave reasoning out",
                _ => "does not return reasoning summaries",
            };
            d.unsupported_hint(
                RequestField::Reasoning,
                format!("Chat Completions {what}; the reasoning output setting was not sent"),
            )?;
        }
    }

    // Model rules: what the model behind the endpoint rejects while it
    // reasons, at the requested effort or its default one.
    let sent_effort = reasoning_effort.as_deref();
    if rules.rejects_tools(sent_effort) && !request.tools.is_empty() {
        return Err(d.unsupported(
            RequestField::Tools,
            format!(
                "`{}` takes tools on Chat Completions only with reasoning effort `none`",
                request.model
            ),
        ));
    }
    let sampling_ok = !rules.rejects_sampling(sent_effort);
    if !sampling_ok {
        let model = &request.model;
        for (field, set, name) in [
            (
                RequestField::Temperature,
                request.temperature.is_some(),
                "temperature",
            ),
            (RequestField::TopP, request.top_p.is_some(), "top_p"),
            (
                RequestField::TopK,
                request.top_k.is_some() && quirks.supports_top_k,
                "top_k",
            ),
            (
                RequestField::FrequencyPenalty,
                request.frequency_penalty.is_some(),
                "frequency_penalty",
            ),
            (
                RequestField::PresencePenalty,
                request.presence_penalty.is_some(),
                "presence_penalty",
            ),
        ] {
            if set {
                d.unsupported_hint(
                    field,
                    format!("`{model}` does not accept `{name}` with reasoning enabled; `{name}` was not sent"),
                )?;
            }
        }
    }
    let keep = |value: Option<f64>| value.filter(|_| sampling_ok);
    // OpenAI's Chat Completions has no `top_k`; servers that take it as an
    // extension (vLLM, Together, …) set `supports_top_k`.
    let top_k = if quirks.supports_top_k {
        request.top_k.filter(|_| sampling_ok)
    } else {
        if request.top_k.is_some() {
            d.unsupported_hint(
                RequestField::TopK,
                "this endpoint has no `top_k`; `top_k` was not sent",
            )?;
        }
        None
    };

    let mut extra = options.extra.clone();
    if !sampling_ok {
        crate::model_rules::drop_logprobs(
            &mut extra,
            &["logprobs", "top_logprobs"],
            &request.model,
            &mut d,
        )?;
    }
    let mut seed = None;
    if let Some(value) = request.seed {
        match quirks.seed_field {
            SeedField::Seed => seed = Some(value),
            SeedField::RandomSeed => {
                extra.insert("random_seed".into(), value.into());
            }
            _ => d.unsupported_hint(
                RequestField::Seed,
                "this endpoint has no seed; `seed` was not sent",
            )?,
        }
    }

    let parallel_tool_calls = crate::schema::gate_parallel_tool_calls(
        request.parallel_tool_calls,
        quirks.supports_parallel_tool_calls,
        &mut d,
    )?;
    let parallel_tool_calls = crate::schema::serial_calls_for_strict(
        parallel_tool_calls,
        dialect,
        strict_tool_sent,
        quirks.supports_parallel_tool_calls,
        reasoning_effort.as_deref(),
        &mut d,
    )?;

    // Breakpoints need no mapping (prefix caching is automatic), but a
    // breakpoint asking for long retention is honored like a request-level
    // retention.
    let key = request.cache.as_ref().and_then(|cache| cache.key.clone());
    let retention = request
        .cache
        .as_ref()
        .and_then(|cache| cache.retention)
        .or_else(|| {
            request
                .cache_breakpoints()
                .any(|b| b.retention == Some(CacheRetention::Long))
                .then_some(CacheRetention::Long)
        });
    // The key and the retention are separate fields: an endpoint may take
    // one without the other (Mistral takes a key only).
    let unsent = match (
        key.is_some() && !quirks.supports_prompt_cache_key,
        retention.is_some() && !quirks.supports_prompt_cache_retention,
    ) {
        (true, true) => {
            Some("this endpoint takes no prompt cache key or retention; they were not sent")
        }
        (true, false) => Some("this endpoint takes no prompt cache key; the key was not sent"),
        (false, true) => {
            Some("this endpoint takes no prompt cache retention; the retention was not sent")
        }
        (false, false) => None,
    };
    if let Some(message) = unsent {
        d.unsupported_hint(RequestField::Cache, message)?;
    }
    let prompt_cache_key = key.filter(|_| quirks.supports_prompt_cache_key);
    let prompt_cache_retention = retention
        .filter(|_| quirks.supports_prompt_cache_retention)
        .map(|r| match r {
            CacheRetention::Long => "24h".to_owned(),
            _ => "in_memory".to_owned(),
        });

    let (max_tokens, max_completion_tokens) = match quirks.max_tokens_field {
        MaxTokensField::MaxCompletionTokens => (None, request.max_output_tokens),
        _ => (request.max_output_tokens, None),
    };

    let body = ChatCompletionRequest {
        model: request.model.clone(),
        messages,
        temperature: keep(request.temperature),
        top_p: keep(request.top_p),
        top_k,
        max_tokens,
        max_completion_tokens,
        stop: (!request.stop.is_empty()).then(|| request.stop.clone()),
        frequency_penalty: keep(request.frequency_penalty),
        presence_penalty: keep(request.presence_penalty),
        tools,
        tool_choice,
        response_format,
        seed,
        reasoning_effort,
        parallel_tool_calls,
        user: options.user.clone(),
        safety_identifier: options.safety_identifier.clone(),
        service_tier: options.service_tier.clone(),
        store: options.store,
        metadata: options.metadata.clone(),
        prompt_cache_key,
        prompt_cache_retention,
        stream,
        stream_options: (stream && quirks.stream_usage).then_some(StreamOptions {
            include_usage: true,
        }),
        extra,
    };
    let _ = ns;
    Ok((body, d.into_warnings()))
}

/// Map the tool call ids of the encoded messages to ones the endpoint
/// accepts: the calls of assistant messages and the `tool` messages that
/// answer them alike, so they stay paired.
fn conform_tool_call_ids(messages: &mut [ApiMessage], format: ToolCallIdFormat) {
    let conform = |id: &mut String| {
        if let Cow::Owned(mapped) = format.conform(id) {
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

/// Encode the messages of a request, one turn each, plus a `tool` turn for
/// each custom tool answer lifted out of a user message.
fn encode_messages(
    messages: &[Message],
    quirks: &CompatQuirks,
    d: &mut Degradations,
) -> allama::Result<Vec<(EncodedTurn, ApiMessage)>> {
    let ns = quirks.namespace.as_ref();
    let pairing = Pairing::new(messages, |call| custom_call(call, ns).is_some());
    let mut turns = Vec::with_capacity(messages.len());
    // The custom tool calls of the last assistant turn, which the user
    // message right after it (and its tool messages) may answer.
    let mut answerable: Vec<String> = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        match message {
            Message::User(user) => {
                let answerable = std::mem::take(&mut answerable);
                let (lifted, api) = convert_user(user, index, &answerable, &pairing, ns, d)?;
                turns.extend(
                    lifted
                        .into_iter()
                        .map(|tool| (EncodedTurn::tool(index), tool)),
                );
                turns.push((EncodedTurn::new(index, message, api.is_empty()), api));
            }
            Message::Assistant(assistant) => {
                let api = convert_assistant(assistant, index, quirks, d)?;
                answerable = api
                    .tool_calls
                    .iter()
                    .flatten()
                    .filter(|call| call.custom.is_some())
                    .map(|call| call.id.clone())
                    .collect();
                turns.push((EncodedTurn::new(index, message, api.is_empty()), api));
            }
            Message::Tool(tool) => {
                let api = convert_tool(tool, index, ns, d)?;
                turns.push((EncodedTurn::tool(index), api));
            }
            _ => {
                let subject = Subject::Content {
                    message: index,
                    block: None,
                };
                return Err(d.reject(subject, "unknown message kind"));
            }
        }
    }
    Ok(turns)
}

/// A user message, and the `tool` messages lifted out of it.
///
/// A custom tool answer (`custom_tool_call_output` of this namespace) is
/// lifted into a `tool` message when it answers a call of `answerable` (the
/// assistant turn right before) and no content of the message comes before
/// it, so lifting never reorders the conversation. Every other answer is
/// dropped with a warning: Chat Completions has no server-side
/// continuation that could hold its call.
fn convert_user(
    user: &UserMessage,
    index: usize,
    answerable: &[String],
    pairing: &Pairing<'_>,
    ns: &str,
    d: &mut Degradations,
) -> allama::Result<(Vec<ApiMessage>, ApiMessage)> {
    let mut lifted = Vec::new();
    let content = match &user.content {
        UserContent::Text(text) => serde_json::Value::String(text.clone()),
        content => {
            let owned;
            let parts = match content {
                UserContent::Parts(parts) => parts.as_slice(),
                other => {
                    owned = other.to_parts();
                    owned.as_slice()
                }
            };
            let mut api_parts = Vec::with_capacity(parts.len());
            let mut answered: Vec<&str> = Vec::new();
            for (block, part) in parts.iter().enumerate() {
                let subject = || Subject::Content {
                    message: index,
                    block: Some(block),
                };
                let answer = match part {
                    ContentPart::Other(other)
                        if other.origin.as_deref() == Some(ns)
                            && client_tools::is_output(other) =>
                    {
                        other
                    }
                    _ => {
                        api_parts.extend(convert_part(part, index, block, ns, d)?);
                        continue;
                    }
                };
                if pairing.is_unpaired(answer) {
                    d.dropped(
                        subject(),
                        format!(
                            "the call this `{}` answers cannot be replayed to Chat Completions; it was dropped",
                            answer.type_name
                        ),
                    );
                    continue;
                }
                if answer.type_name != client_tools::CUSTOM_OUTPUT {
                    d.dropped(
                        subject(),
                        format!(
                            "Chat Completions has no `{}` content; it was dropped",
                            answer.type_name
                        ),
                    );
                    continue;
                }
                let call = client_tools::output_call_id(answer)
                    .filter(|id| answerable.iter().any(|call| call == id));
                match call {
                    Some(id) if api_parts.is_empty() && !answered.contains(&id) => {
                        answered.push(id);
                        lifted.push(lift_answer(answer, id, subject(), d));
                    }
                    Some(_) => d.dropped(
                        subject(),
                        "the custom tool answer follows other content of the user message \
                         (or answers its call twice), and lifting it into a tool message would \
                         reorder the conversation; it was dropped",
                    ),
                    None => d.dropped(
                        subject(),
                        "the custom tool answer answers no custom tool call of the assistant \
                         turn right before it, and Chat Completions has no server-side \
                         continuation to hold its call; it was dropped",
                    ),
                }
            }
            serde_json::Value::Array(api_parts)
        }
    };
    let api = ApiMessage {
        role: "user".into(),
        content: Some(content),
        name: participant_name(&user.provider_data, ns, index, d),
        ..ApiMessage::default()
    };
    Ok((lifted, api))
}

/// The Chat Completions participant `name` of a user or assistant message:
/// a string under [`PARTICIPANT_NAME_KEY`](crate::PARTICIPANT_NAME_KEY) in
/// this endpoint's own namespace of the message's provider data. A value
/// that is not a string is dropped with a warning.
fn participant_name(
    data: &allama::ProviderData,
    ns: &str,
    index: usize,
    d: &mut Degradations,
) -> Option<String> {
    let value = data.get_raw(ns, crate::PARTICIPANT_NAME_KEY)?;
    if let Some(name) = value.as_str() {
        return Some(name.to_owned());
    }
    d.dropped(
        Subject::Content {
            message: index,
            block: None,
        },
        format!(
            "the participant name (`{ns}.{}`) is not a string; it was not sent",
            crate::PARTICIPANT_NAME_KEY
        ),
    );
    None
}

/// The `tool` message for a lifted custom tool answer.
fn lift_answer(
    answer: &OtherContent,
    call_id: &str,
    subject: Subject,
    d: &mut Degradations,
) -> ApiMessage {
    let (text, skipped) = client_tools::output_text(answer);
    if !skipped.is_empty() {
        d.dropped(
            subject,
            format!(
                "a Chat Completions tool message holds text only; the non-text parts ({}) of \
                 the custom tool answer were dropped",
                skipped.join(", ")
            ),
        );
    }
    ApiMessage {
        role: "tool".into(),
        content: Some(text.into()),
        tool_call_id: Some(call_id.to_owned()),
        ..ApiMessage::default()
    }
}

fn convert_assistant(
    m: &AssistantMessage,
    index: usize,
    quirks: &CompatQuirks,
    d: &mut Degradations,
) -> allama::Result<ApiMessage> {
    let ns = quirks.namespace.as_ref();
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut details: Vec<serde_json::Value> = Vec::new();
    let mut signature: Option<String> = None;
    let mut refusal = String::new();
    let mut tool_calls = Vec::new();
    for (block, content) in m.content.iter().enumerate() {
        let subject = || Subject::Content {
            message: index,
            block: Some(block),
        };
        match content {
            // A refusal goes in its own field where the dialect has one,
            // and is assistant text otherwise, so it is never lost.
            ContentBlock::Refusal(r) if quirks.replay_refusal => refusal.push_str(&r.text),
            ContentBlock::Text(t) if quirks.replay_refusal && is_legacy_refusal(t, ns) => {
                refusal.push_str(&t.text);
            }
            ContentBlock::Refusal(r) => text.push_str(&r.text),
            ContentBlock::Text(t) => text.push_str(&t.text),
            ContentBlock::ToolCall(call) => {
                let arguments = if call.has_valid_arguments() {
                    if call.arguments.trim().is_empty() {
                        "{}".to_owned()
                    } else {
                        call.arguments.clone()
                    }
                } else {
                    d.coded_lossy_hint(
                        subject(),
                        format!(
                            "the arguments of tool call `{}` are not a JSON object; they were sent as `{{}}`",
                            call.name
                        ),
                        TOOL_CALL_ARGUMENTS,
                    )?;
                    "{}".to_owned()
                };
                tool_calls.push(ApiToolCall {
                    id: call.id.clone(),
                    call_type: TOOL_TYPE_FUNCTION.into(),
                    function: Some(ApiToolCallFunction {
                        name: call.name.clone(),
                        arguments,
                    }),
                    custom: None,
                    extra: replayed_call_fields(&call.provider_data, ns),
                });
            }
            ContentBlock::Reasoning(r) => {
                let foreign: Vec<&str> = r.provider_data.foreign_namespaces(ns).collect();
                // Reasoning this endpoint wrote: it is expected back where
                // the dialect takes it, and dropping it otherwise is no
                // degradation (ADR 0003).
                let own = r.provider_data.namespace(ns).is_some();
                let own_details = r
                    .provider_data
                    .get_raw(ns, REASONING_DETAILS_KEY)
                    .and_then(serde_json::Value::as_array);
                if !foreign.is_empty() {
                    d.dropped_replay_state(
                        ReplayState::ForeignReasoning,
                        subject(),
                        format!(
                            "reasoning from {} can be replayed only to its origin; it was dropped",
                            describe_origins(&foreign)
                        ),
                    );
                    continue;
                }
                match quirks.reasoning_replay {
                    ReasoningReplayField::ReasoningContent | ReasoningReplayField::Reasoning => {
                        reasoning.push_str(&r.text);
                    }
                    ReasoningReplayField::ReasoningDetails => match own_details {
                        Some(own_details) => details.extend(own_details.iter().cloned()),
                        None if own || r.text.is_empty() => {}
                        None => d.dropped_replay_state(
                            ReplayState::Reasoning,
                            subject(),
                            "this endpoint takes reasoning back only as the `reasoning_details` \
                             it sent, and this reasoning has none; it was dropped",
                        ),
                    },
                    ReasoningReplayField::ReasoningText if own => {
                        reasoning.push_str(&r.text);
                        if let Some(own_signature) = r
                            .provider_data
                            .get_raw(ns, REASONING_OPAQUE_KEY)
                            .and_then(serde_json::Value::as_str)
                        {
                            signature = Some(own_signature.to_owned());
                        }
                    }
                    ReasoningReplayField::ReasoningText if r.text.is_empty() => {}
                    ReasoningReplayField::ReasoningText => d.dropped_replay_state(
                        ReplayState::Reasoning,
                        subject(),
                        "this endpoint takes reasoning back only with the `reasoning_opaque` \
                         signature it sent, and this reasoning has none; it was dropped",
                    ),
                    _ if own || r.text.is_empty() => {}
                    _ => d.dropped_replay_state(
                        ReplayState::Reasoning,
                        subject(),
                        "this endpoint takes no reasoning back; it was dropped",
                    ),
                }
            }
            ContentBlock::Image(_) => {
                return Err(d.reject(
                    subject(),
                    "assistant image replay is not supported by Chat Completions",
                ));
            }
            ContentBlock::Other(other) => match custom_call(other, ns) {
                Some(call) => tool_calls.push(ApiToolCall {
                    id: call.call_id,
                    call_type: TOOL_TYPE_CUSTOM.into(),
                    function: None,
                    custom: Some(ApiCustomToolCall {
                        name: call.name,
                        input: call.input,
                    }),
                    extra: replayed_call_fields(&other.provider_data, ns),
                }),
                None if other.origin.as_deref() != Some(ns) => d.dropped_replay_state(
                    ReplayState::ForeignContent,
                    subject(),
                    format!(
                        "`{}` content from {} can be replayed only to its origin; it was dropped",
                        other.type_name,
                        describe_origins(other.origin.as_slice())
                    ),
                ),
                None => d.dropped_replay_state(
                    ReplayState::Content,
                    subject(),
                    format!(
                        "Chat Completions cannot replay `{}` content; it was dropped",
                        other.type_name
                    ),
                ),
            },
            ContentBlock::ProviderToolCall(_) | ContentBlock::ProviderToolResult(_)
                if content.origin() != Some(ns) =>
            {
                d.dropped_replay_state(
                    ReplayState::ForeignContent,
                    subject(),
                    format!(
                        "`{}` content from {} can be replayed only to its origin; it was dropped",
                        content.type_name(),
                        describe_origins(content.origin().as_slice())
                    ),
                );
            }
            _ => d.dropped(
                subject(),
                format!(
                    "Chat Completions cannot replay `{}` content; it was dropped",
                    content.type_name()
                ),
            ),
        }
    }
    Ok(ApiMessage {
        role: "assistant".into(),
        content: (!text.is_empty()).then(|| text.into()),
        name: participant_name(&m.provider_data, ns, index, d),
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
        refusal: (!refusal.is_empty()).then_some(refusal),
        ..reasoning_replay(quirks.reasoning_replay, reasoning, details, signature)
    })
}

/// The unmodeled fields of a tool call this endpoint returned (its own
/// namespace of the call's provider data), sent back with the call: Gemini
/// requires its `extra_content` thought signature back.
fn replayed_call_fields(data: &allama::ProviderData, ns: &str) -> ExtraMap {
    data.namespace(ns)
        .map(|own| tool_call_extra(own.clone()))
        .unwrap_or_default()
}

/// An assistant message carrying the replayed reasoning `text` or
/// `details` (if any) in the dialect's field. With
/// [`ReasoningReplayField::ReasoningText`], the text goes back only with
/// its `signature`, as the endpoint's own clients send it.
fn reasoning_replay(
    field: ReasoningReplayField,
    text: String,
    details: Vec<serde_json::Value>,
    signature: Option<String>,
) -> ApiMessage {
    let text = (!text.is_empty()).then_some(text);
    match field {
        ReasoningReplayField::ReasoningContent => ApiMessage {
            reasoning_content: text,
            ..ApiMessage::default()
        },
        ReasoningReplayField::Reasoning => ApiMessage {
            reasoning: text,
            ..ApiMessage::default()
        },
        ReasoningReplayField::ReasoningDetails => ApiMessage {
            reasoning_details: (!details.is_empty()).then_some(details),
            ..ApiMessage::default()
        },
        ReasoningReplayField::ReasoningText if signature.is_some() => ApiMessage {
            reasoning_text: text,
            reasoning_opaque: signature,
            ..ApiMessage::default()
        },
        _ => ApiMessage::default(),
    }
}

/// A refusal stored by an earlier version: a text block flagged
/// `refusal: true` in this namespace. It is replayed as a [`Refusal`](allama::Refusal).
pub(crate) fn is_legacy_refusal(text: &Text, ns: &str) -> bool {
    text.provider_data.get_raw(ns, "refusal") == Some(&serde_json::Value::Bool(true))
}

fn convert_tool(
    m: &ToolMessage,
    index: usize,
    ns: &str,
    d: &mut Degradations,
) -> allama::Result<ApiMessage> {
    let content = match &m.content {
        ToolResultContent::Text(text) => text.clone().into(),
        ToolResultContent::Object(object) => {
            serde_json::Value::Object(object.clone()).to_string().into()
        }
        ToolResultContent::Parts(parts) => {
            // Text, sent as a string unless an `Other` part is sent with it.
            // The string keeps the part boundaries, as Gemini does.
            let mut texts = Vec::with_capacity(parts.len());
            let mut api_parts = Vec::with_capacity(parts.len());
            let mut verbatim = false;
            for (block, part) in parts.iter().enumerate() {
                match part {
                    ContentPart::Text(t) => {
                        texts.push(t.text.as_str());
                        api_parts.push(serde_json::json!({"type": "text", "text": t.text}));
                    }
                    ContentPart::Other(other) => {
                        if let Some(value) = convert_other(other, index, block, ns, d) {
                            api_parts.push(value);
                            verbatim = true;
                        }
                    }
                    _ => {
                        let subject = Subject::Content {
                            message: index,
                            block: Some(block),
                        };
                        return Err(d.reject(
                            subject,
                            format!(
                                "`{}` tool result parts are not supported by Chat Completions",
                                part.type_name()
                            ),
                        ));
                    }
                }
            }
            if verbatim {
                serde_json::Value::Array(api_parts)
            } else {
                texts.join("\n\n").into()
            }
        }
        _ => {
            let subject = Subject::Content {
                message: index,
                block: None,
            };
            return Err(d.reject(subject, "unknown tool result content"));
        }
    };
    if m.is_error {
        // Coded: history repeats the result on every turn. Never refused:
        // the model still reads the error text, only the flag is lost
        // (ADR 0003).
        d.coded_approximated(
            Subject::Content {
                message: index,
                block: None,
            },
            "Chat Completions has no tool error flag; the error text was sent as the tool result; the model still reads the error text; to send your own encoding to this endpoint, use `push_tool_result`",
            TOOL_RESULT_ERROR,
        );
    }
    Ok(ApiMessage {
        role: "tool".into(),
        content: Some(content),
        tool_call_id: Some(m.tool_call_id.clone()),
        ..ApiMessage::default()
    })
}

/// The custom tool call an `Other` block of this namespace holds, which is
/// replayed as a `custom` tool call. Other client tool calls (shell,
/// computer use, `apply_patch`) are Responses-only.
fn custom_call(other: &OtherContent, ns: &str) -> Option<CustomToolCall> {
    if other.origin.as_deref() != Some(ns) {
        return None;
    }
    CustomToolCall::from_content(other)
}

fn convert_part(
    part: &ContentPart,
    message: usize,
    block: usize,
    ns: &str,
    d: &mut Degradations,
) -> allama::Result<Option<serde_json::Value>> {
    let subject = Subject::Content {
        message,
        block: Some(block),
    };
    Ok(Some(match part {
        ContentPart::Text(t) => serde_json::json!({"type": "text", "text": t.text}),
        ContentPart::Image(image) => {
            let url = match &image.source {
                MediaSource::Url { url } => url.clone(),
                MediaSource::Base64 {
                    media_type, data, ..
                } => {
                    format!("data:{media_type};base64,{data}")
                }
                MediaSource::ProviderFile { .. } => {
                    return Err(d.reject(
                        subject,
                        "Chat Completions takes images by URL or inline data, not by file id; \
                         the Responses API takes a stored image file",
                    ));
                }
                _ => return Err(d.reject(subject, "unknown image source")),
            };
            let mut image_url = serde_json::Map::new();
            image_url.insert("url".into(), url.into());
            if let Some(detail) = image.detail {
                image_url.insert("detail".into(), detail.as_str().into());
            }
            serde_json::json!({"type": "image_url", "image_url": image_url})
        }
        ContentPart::File(file) => convert_file(file, subject, ns, d)?,
        ContentPart::Other(other) => return Ok(convert_other(other, message, block, ns, d)),
        _ => {
            return Err(d.reject(
                subject,
                format!("`{}` parts are not supported", part.type_name()),
            ));
        }
    }))
}

/// An `Other` part, sent verbatim when it has no origin (the caller's own
/// raw part) or when it is of this namespace and a Chat Completions content
/// type. Parts from another provider, and the Responses-only items that
/// share the `openai` namespace, are dropped with a warning.
fn convert_other(
    other: &OtherContent,
    message: usize,
    block: usize,
    ns: &str,
    d: &mut Degradations,
) -> Option<serde_json::Value> {
    let subject = Subject::Content {
        message,
        block: Some(block),
    };
    match other.origin.as_deref() {
        Some(origin) if origin != ns => {
            d.dropped_replay_state(
                ReplayState::ForeignContent,
                subject,
                format!(
                    "`{}` content from `{origin}` can be replayed only to its origin; it was dropped",
                    other.type_name
                ),
            );
            return None;
        }
        Some(_) if !CONTENT_PART_TYPES.contains(&other.type_name.as_str()) => {
            d.dropped(
                subject,
                format!(
                    "Chat Completions has no `{}` content; it was dropped",
                    other.type_name
                ),
            );
            return None;
        }
        _ => {}
    }
    let mut object = other.data.clone();
    object.insert("type".into(), other.type_name.clone().into());
    Some(serde_json::Value::Object(object))
}

fn convert_file(
    file: &File,
    subject: Subject,
    ns: &str,
    d: &Degradations,
) -> allama::Result<serde_json::Value> {
    let data = match &file.source {
        MediaSource::Base64 { data, .. } => data,
        // A file this provider stores (`prepare` refuses another
        // provider's): Chat Completions references documents by id.
        MediaSource::ProviderFile { origin, id, .. } if origin == ns => {
            let mut part = serde_json::Map::new();
            part.insert("file_id".into(), id.clone().into());
            if let Some(filename) = &file.filename {
                part.insert("filename".into(), filename.clone().into());
            }
            return Ok(serde_json::json!({"type": "file", "file": part}));
        }
        MediaSource::ProviderFile { origin, .. } => {
            return Err(d.reject(
                subject,
                format!("a file stored by `{origin}` can be sent only to `{origin}`"),
            ));
        }
        _ => {
            return Err(d.reject(
                subject,
                "Chat Completions accepts files as inline base64 data or by file id only",
            ));
        }
    };
    let media_type = file.media_type.as_str();
    if media_type == "application/pdf" {
        let filename = file
            .filename
            .clone()
            .unwrap_or_else(|| "document.pdf".into());
        return Ok(serde_json::json!({
            "type": "file",
            "file": {"filename": filename, "file_data": format!("data:{media_type};base64,{data}")}
        }));
    }
    let format = match media_type {
        "audio/wav" | "audio/x-wav" | "audio/wave" => "wav",
        "audio/mpeg" | "audio/mp3" => "mp3",
        other => {
            return Err(d.reject(
                subject,
                format!(
                    "file media type `{other}` is not supported by Chat Completions (PDF, WAV, \
                     and MP3 are)"
                ),
            ));
        }
    };
    Ok(serde_json::json!({"type": "input_audio", "input_audio": {"data": data, "format": format}}))
}

/// Map a raw finish reason, with the aliases vendors use for the standard
/// ones: `eos`, `eos_token` and `end_turn` end the turn; `max_tokens` and
/// `model_length` (Mistral, Workers AI) hit a length limit.
///
/// `length` and its aliases map to [`FinishReason::Length`] even when the
/// context window, not the output cap, ran out: these servers do not tell
/// the two apart.
#[must_use]
pub(crate) fn parse_finish_reason(value: &str) -> FinishReason {
    match value {
        "stop" | "eos" | "eos_token" | "end_turn" => FinishReason::Stop,
        "length" | "max_tokens" | "model_length" => FinishReason::Length,
        "tool_calls" | "function_call" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.to_owned()),
    }
}

/// Map a raw finish reason for a turn whose `content` is known: a turn with
/// a function call or a custom tool call waits for the client, so it
/// stopped for tool calls even if the endpoint says `stop` (OpenAI does for
/// a forced tool choice, Gemini and vLLM for every tool call).
pub(crate) fn finish_reason(value: &str, content: &[ContentBlock]) -> FinishReason {
    let calls = content
        .iter()
        .any(|block| matches!(block, ContentBlock::ToolCall(_)))
        || client_tools::has_pending(content);
    promote_stop(parse_finish_reason(value), calls)
}

/// `Stop` becomes `ToolCalls` when the turn holds calls that wait for the
/// client.
pub(crate) fn promote_stop(reason: FinishReason, calls: bool) -> FinishReason {
    match reason {
        FinishReason::Stop if calls => FinishReason::ToolCalls,
        reason => reason,
    }
}

/// The provider-data key, on a decoded [`Reasoning`] block, of the wire
/// field the reasoning text arrived in (`reasoning_content`, `reasoning`,
/// Copilot's `reasoning_text`, or Mistral's `thinking` content parts). It
/// marks the reasoning as this endpoint's own, so an endpoint that takes no
/// reasoning back drops it without a warning.
pub(crate) const REASONING_FIELD_KEY: &str = "reasoning_field";

/// The provider-data key, on a decoded [`Reasoning`] block, of OpenRouter's
/// `reasoning_details`, replayed verbatim with
/// [`ReasoningReplayField::ReasoningDetails`].
pub(crate) const REASONING_DETAILS_KEY: &str = "reasoning_details";

/// The provider-data key, on a decoded [`Reasoning`] block, of GitHub
/// Copilot's `reasoning_opaque` signature, replayed verbatim with
/// [`ReasoningReplayField::ReasoningText`].
pub(crate) const REASONING_OPAQUE_KEY: &str = "reasoning_opaque";

/// The provider-data key, on a decoded [`Text`] block, of the raw Chat
/// Completions annotations (`{"type": "url_citation", "url_citation":
/// {…}}`). It differs from the Responses key (`annotations`, a flat shape)
/// that shares the `openai` namespace, so neither API is sent the other's.
pub(crate) const ANNOTATIONS_KEY: &str = "chat_annotations";

/// A decoded reasoning block: `text` from the wire `field`, with the
/// reasoning details and signature, when the endpoint sent them.
pub(crate) fn reasoning_block(
    text: String,
    field: &str,
    details: Option<Vec<serde_json::Value>>,
    signature: Option<String>,
    ns: &str,
) -> Reasoning {
    let mut data = allama::ProviderData::new().with_raw(ns, REASONING_FIELD_KEY, field.into());
    if let Some(details) = details.filter(|details| !details.is_empty()) {
        data.insert_raw(ns, REASONING_DETAILS_KEY, details.into());
    }
    if let Some(signature) = signature.filter(|signature| !signature.is_empty()) {
        data.insert_raw(ns, REASONING_OPAQUE_KEY, signature.into());
    }
    Reasoning::new(text).with_provider_data(data)
}

/// The readable text of OpenRouter reasoning details: the `text` of
/// `reasoning.text` entries and the `summary` of `reasoning.summary` ones.
pub(crate) fn reasoning_details_text(details: &[serde_json::Value]) -> String {
    details
        .iter()
        .filter_map(|detail| {
            detail
                .get("text")
                .or_else(|| detail.get("summary"))
                .and_then(serde_json::Value::as_str)
        })
        .collect()
}

/// One piece of message content.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ContentPiece {
    /// Answer text.
    Text(String),
    /// Reasoning text (Mistral's `thinking` parts).
    Thinking(String),
    /// A part of a type this crate does not decode: its type and the rest
    /// of its fields.
    Other(String, ExtraMap),
}

/// The pieces of a message's `content`: a string; an array of parts (text,
/// Mistral's `thinking` parts, and parts of other types); or, for a JSON
/// object some endpoints return for structured output, its JSON text.
pub(crate) fn content_pieces(content: Option<serde_json::Value>) -> Vec<ContentPiece> {
    match content {
        None | Some(serde_json::Value::Null) => Vec::new(),
        Some(serde_json::Value::String(text)) => vec![ContentPiece::Text(text)],
        Some(serde_json::Value::Array(parts)) => parts.into_iter().filter_map(part_piece).collect(),
        Some(other) => vec![ContentPiece::Text(other.to_string())],
    }
}

fn part_piece(part: serde_json::Value) -> Option<ContentPiece> {
    let serde_json::Value::Object(mut part) = part else {
        return match part {
            serde_json::Value::String(text) => Some(ContentPiece::Text(text)),
            _ => None,
        };
    };
    let kind = match part.remove("type") {
        Some(serde_json::Value::String(kind)) => kind,
        _ => "text".to_owned(),
    };
    match kind.as_str() {
        "text" => match part.get("text") {
            Some(serde_json::Value::String(text)) => Some(ContentPiece::Text(text.clone())),
            _ => Some(ContentPiece::Other(kind, part)),
        },
        "thinking" => {
            let text = match part.get("thinking") {
                Some(serde_json::Value::String(text)) => text.clone(),
                Some(serde_json::Value::Array(chunks)) => chunks
                    .iter()
                    .filter_map(|chunk| match chunk {
                        serde_json::Value::String(text) => Some(text.as_str()),
                        chunk => chunk.get("text").and_then(serde_json::Value::as_str),
                    })
                    .collect(),
                _ => return Some(ContentPiece::Other(kind, part)),
            };
            Some(ContentPiece::Thinking(text))
        }
        _ => Some(ContentPiece::Other(kind, part)),
    }
}

/// An id for the tool call at `position` that the endpoint sent without
/// one: `call_<scope>_<position>`, scoped by the response id, or by a random
/// id (made once per response, in `anonymous`) when the response has none,
/// so no two calls in a conversation share it (providers reject a reused
/// id). Characters outside `[A-Za-z0-9_-]` in the response id become `_`,
/// so the id stays valid where providers restrict tool call ids
/// (Anthropic).
pub(crate) fn made_up_call_id(
    response_id: Option<&str>,
    anonymous: &mut Option<String>,
    position: usize,
) -> String {
    let scope: String = match response_id.filter(|id| !id.is_empty()) {
        Some(id) => id
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect(),
        None => anonymous
            .get_or_insert_with(|| uuid::Uuid::new_v4().simple().to_string())
            .clone(),
    };
    format!("call_{scope}_{position}")
}

/// The unmodeled fields of a tool call worth keeping as its provider data.
pub(crate) fn tool_call_extra(extra: ExtraMap) -> ExtraMap {
    extra
        .into_iter()
        .filter(|(key, value)| !value.is_null() && !TOOL_CALL_FIELDS.contains(&key.as_str()))
        .collect()
}

/// A response tool call: a function call becomes a [`ToolCall`], and a
/// custom tool call an `Other` block of type `custom_tool_call` with the
/// dialect namespace as origin, the shape the Responses API decodes to.
/// Fields this crate does not model go in the call's provider data.
fn decode_tool_call(call: ApiToolCall, id: String, ns: &str) -> allama::Result<ContentBlock> {
    let extra = tool_call_extra(call.extra);
    if let Some(custom) = call.custom {
        let mut other = custom_call_content(id, custom.name, custom.input, ns);
        for (key, value) in extra {
            other.provider_data.insert_raw(ns, key, value);
        }
        return Ok(ContentBlock::Other(other));
    }
    match call.function {
        Some(function) => {
            let mut tool_call = ToolCall::new(id, function.name, function.arguments);
            for (key, value) in extra {
                tool_call.provider_data.insert_raw(ns, key, value);
            }
            Ok(ContentBlock::ToolCall(tool_call))
        }
        None => Err(Error::unexpected_response(format!(
            "Chat Completions tool call `{id}` of type `{}` has neither a function nor a custom \
             tool",
            call.call_type
        ))
        .authored()),
    }
}

/// The `Other` content of a custom tool call.
pub(crate) fn custom_call_content(
    call_id: String,
    name: String,
    input: String,
    ns: &str,
) -> OtherContent {
    let mut data = ExtraMap::new();
    data.insert("call_id".into(), call_id.into());
    data.insert("name".into(), name.into());
    data.insert("input".into(), input.into());
    OtherContent::new(client_tools::CUSTOM, data).with_origin(ns)
}

/// Map usage to the portable contract.
///
/// Cached prompt tokens come from `prompt_tokens_details.cached_tokens`, or
/// DeepSeek's `prompt_cache_hit_tokens`. Audio tokens are input and output
/// breakdowns. Other vendor counters stay in the response's provider data
/// ([`usage_extra`]).
#[must_use]
pub(crate) fn from_api_usage(usage: &ApiUsage) -> Usage {
    let prompt = usage.prompt_tokens_details.clone().unwrap_or_default();
    let completion = usage.completion_tokens_details.clone().unwrap_or_default();
    let mut converted = Usage::new()
        .input_tokens(usage.prompt_tokens)
        .output_tokens(usage.completion_tokens);
    converted.total_tokens = Some(if usage.total_tokens > 0 {
        usage.total_tokens
    } else {
        usage.prompt_tokens.saturating_add(usage.completion_tokens)
    });
    converted.cache_read_input_tokens = prompt.cached_tokens.or(usage.prompt_cache_hit_tokens);
    converted.cache_write_input_tokens = prompt.cache_write_tokens;
    converted.reasoning_tokens = completion.reasoning_tokens;
    if let Some(audio) = prompt.audio_tokens.filter(|v| *v > 0) {
        converted
            .breakdowns
            .insert("audio_input_tokens".into(), audio);
    }
    if let Some(audio) = completion.audio_tokens.filter(|v| *v > 0) {
        converted
            .breakdowns
            .insert("audio_output_tokens".into(), audio);
    }
    converted
}

/// The vendor usage counters [`from_api_usage`] does not map (OpenRouter's
/// `cost`, Workers AI's `neurons`, DeepSeek's cache misses), for the
/// response's provider data under [`USAGE_KEY`]. `None` when there are
/// none.
pub(crate) fn usage_extra(usage: &ApiUsage) -> Option<serde_json::Value> {
    let extra: ExtraMap = usage
        .extra
        .iter()
        .filter(|(_, value)| !value.is_null())
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    (!extra.is_empty()).then_some(serde_json::Value::Object(extra))
}

/// Convert a Chat Completions response.
///
/// Content comes in this order: reasoning (`reasoning_content`,
/// `reasoning`, `reasoning_text`, `reasoning_details`, with a
/// `reasoning_opaque` signature), the message content (with Mistral's
/// `thinking` parts as reasoning, in place), the refusal, then the tool
/// calls. Message and choice fields this crate does not model (`logprobs`,
/// …) are kept in the response's provider data.
///
/// # Errors
///
/// Fails when the response has no choices.
pub(crate) fn from_api_response(
    response: ChatCompletionResponse,
    context: ResponseContext,
) -> allama::Result<ChatResponse> {
    let ns = context.namespace.as_ref();
    let choice = response
        .choices
        .into_iter()
        .min_by_key(|c| c.index)
        .ok_or_else(|| {
            Error::unexpected_response("Chat Completions response had no choices")
                .authored()
                .with_provider(context.provider.as_ref())
        })?;
    let message = choice.message;

    let mut content = Vec::new();
    let (reasoning, field) = match (
        message.reasoning_content,
        message.reasoning,
        message.reasoning_text,
    ) {
        (Some(text), _, _) if !text.is_empty() => (text, "reasoning_content"),
        (_, Some(text), _) if !text.is_empty() => (text, "reasoning"),
        (_, _, Some(text)) if !text.is_empty() => (text, "reasoning_text"),
        _ => (String::new(), "reasoning"),
    };
    let details = message.reasoning_details.filter(|d| !d.is_empty());
    let signature = message.reasoning_opaque.filter(|s| !s.is_empty());
    if !reasoning.is_empty() || details.is_some() || signature.is_some() {
        let (text, field) = match (&details, reasoning.is_empty()) {
            (Some(details), true) => (reasoning_details_text(details), REASONING_DETAILS_KEY),
            (None, true) => (reasoning, "reasoning_text"),
            _ => (reasoning, field),
        };
        content.push(ContentBlock::Reasoning(reasoning_block(
            text, field, details, signature, ns,
        )));
    }
    let mut annotations = message.annotations.filter(|a| !a.is_empty());
    let mut text = String::new();
    let flush_text = |text: &mut String,
                      annotations: &mut Option<Vec<serde_json::Value>>,
                      content: &mut Vec<ContentBlock>| {
        if text.is_empty() {
            return;
        }
        let mut block = Text::new(std::mem::take(text));
        if let Some(annotations) = annotations.take() {
            let annotations = serde_json::Value::Array(annotations);
            block.citations = crate::citations::citations(Some(&annotations), &block.text, ns);
            block
                .provider_data
                .insert_raw(ns, ANNOTATIONS_KEY, annotations);
        }
        content.push(ContentBlock::Text(block));
    };
    for piece in content_pieces(message.content) {
        match piece {
            ContentPiece::Text(fragment) => text.push_str(&fragment),
            ContentPiece::Thinking(thinking) => {
                flush_text(&mut text, &mut annotations, &mut content);
                if !thinking.is_empty() {
                    content.push(ContentBlock::Reasoning(reasoning_block(
                        thinking, "thinking", None, None, ns,
                    )));
                }
            }
            ContentPiece::Other(kind, data) => {
                flush_text(&mut text, &mut annotations, &mut content);
                content.push(ContentBlock::Other(
                    OtherContent::new(kind, data).with_origin(ns),
                ));
            }
        }
    }
    flush_text(&mut text, &mut annotations, &mut content);
    let refused = message.refusal.as_ref().is_some_and(|r| !r.is_empty());
    if let Some(refusal) = message.refusal.filter(|r| !r.is_empty()) {
        content.push(ContentBlock::Refusal(Refusal::new(refusal)));
    }
    let mut anonymous = None;
    for (position, call) in message
        .tool_calls
        .unwrap_or_default()
        .into_iter()
        .enumerate()
    {
        let id = if call.id.is_empty() {
            made_up_call_id(response.id.as_deref(), &mut anonymous, position)
        } else {
            call.id.clone()
        };
        content.push(
            decode_tool_call(call, id, ns)
                .map_err(|err| err.with_provider(context.provider.as_ref()))?,
        );
    }

    let mut converted = ChatResponse::new(content);
    converted.finish_reason = if refused {
        Some(FinishReason::Refusal)
    } else {
        choice
            .finish_reason
            .as_deref()
            .map(|reason| finish_reason(reason, &converted.content))
    };
    converted.finish_reason_raw = choice.finish_reason;
    converted.usage = response.usage.as_ref().map(from_api_usage);
    if let Some(extra) = response.usage.as_ref().and_then(usage_extra) {
        converted.provider_data.insert_raw(ns, USAGE_KEY, extra);
    }
    converted.model = response.model;
    converted.id = response.id;
    converted.request_id = context.request_id;
    converted.provider = Some(context.provider.clone());
    converted.warnings = context.warnings;
    let message_extra = message.extra.into_iter().filter(|(key, _)| key != "role");
    for (key, value) in response
        .extra
        .into_iter()
        .chain(choice.extra)
        .chain(message_extra)
    {
        if keep_extra_field(&key, &value) && converted.provider_data.get_raw(ns, &key).is_none() {
            converted.provider_data.insert_raw(ns, key, value);
        }
    }
    if let Some(fingerprint) = response.system_fingerprint {
        converted
            .provider_data
            .insert_raw(ns, "system_fingerprint", fingerprint.into());
    }
    if let Some(tier) = response.service_tier {
        converted
            .provider_data
            .insert_raw(ns, "service_tier", tier.into());
    }
    Ok(converted)
}

#[cfg(test)]
mod client_tool_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use allama::{
        CacheBreakpoint, PromptCache, ProviderData, ReasoningConfig, ReasoningEffort, SystemPrompt,
        ToolDefinition, UnsupportedPolicy, WarningKind,
    };
    use serde_json::json;

    fn ctx() -> ResponseContext {
        ResponseContext {
            provider: "demo".into(),
            namespace: "openai-compat:demo".into(),
            request_id: Some("req_1".into()),
            warnings: Vec::new(),
        }
    }

    fn convert(request: &ChatRequest, quirks: &CompatQuirks) -> (serde_json::Value, Vec<Warning>) {
        let (body, warnings) =
            to_chat_completion_request(request, false, &WireOptions::default(), quirks, "demo")
                .unwrap();
        (serde_json::to_value(body).unwrap(), warnings)
    }

    #[test]
    fn reasoning_is_replayed_in_the_dialect_field() {
        let request = ChatRequest::new("m").extend_messages([
            Message::user("q"),
            Message::assistant_blocks(vec![
                ContentBlock::Reasoning(Reasoning::new("think")),
                ContentBlock::text("a"),
            ]),
            Message::user("again"),
        ]);
        let mut quirks = CompatQuirks::new("openai-compat");

        quirks.reasoning_replay = ReasoningReplayField::ReasoningContent;
        let (body, warnings) = convert(&request, &quirks);
        assert_eq!(body["messages"][1]["reasoning_content"], "think");
        assert!(body["messages"][1].get("reasoning").is_none());
        assert!(warnings.is_empty(), "{warnings:?}");

        quirks.reasoning_replay = ReasoningReplayField::Reasoning;
        let (body, warnings) = convert(&request, &quirks);
        assert_eq!(body["messages"][1]["reasoning"], "think");
        assert!(body["messages"][1].get("reasoning_content").is_none());
        assert!(warnings.is_empty(), "{warnings:?}");

        quirks.reasoning_replay = ReasoningReplayField::Disabled;
        let (body, warnings) = convert(&request, &quirks);
        assert!(body["messages"][1].get("reasoning").is_none());
        assert!(body["messages"][1].get("reasoning_content").is_none());
        assert_eq!(
            warnings.iter().map(|w| w.kind).collect::<Vec<_>>(),
            [WarningKind::Dropped]
        );
    }

    /// Ids the endpoint would refuse are mapped, the same way in
    /// `tool_calls` and in the `tool` messages; accepted ones are kept.
    #[test]
    fn foreign_tool_call_ids_follow_the_endpoint_format() {
        let long = format!("gemini_{}_call_0", "r".repeat(60));
        let request = ChatRequest::new("m").extend_messages([
            Message::user("q"),
            Message::assistant_blocks(vec![
                ContentBlock::tool_call("toolu_01ABC", "f", "{}"),
                ContentBlock::tool_call(long.clone(), "f", "{}"),
                ContentBlock::tool_call("D681PevKs", "f", "{}"),
            ]),
            Message::tool_result(long.clone(), "f", "a"),
            Message::tool_result("D681PevKs", "f", "b"),
            Message::tool_result("toolu_01ABC", "f", "c"),
        ]);
        let ids = |body: &serde_json::Value| {
            let calls: Vec<String> = body["messages"][1]["tool_calls"]
                .as_array()
                .unwrap()
                .iter()
                .map(|call| call["id"].as_str().unwrap().to_owned())
                .collect();
            let results: Vec<String> = (2..5)
                .map(|i| {
                    body["messages"][i]["tool_call_id"]
                        .as_str()
                        .unwrap()
                        .to_owned()
                })
                .collect();
            (calls, results)
        };

        // OpenAI's 64-character limit by default.
        let (body, warnings) = convert(&request, &CompatQuirks::default());
        assert!(warnings.is_empty(), "{warnings:?}");
        let (calls, results) = ids(&body);
        assert_eq!(calls[0], "toolu_01ABC");
        assert_eq!(calls[1].chars().count(), 64);
        assert_ne!(calls[1], long);
        assert_eq!(
            results,
            [calls[1].clone(), calls[2].clone(), calls[0].clone()]
        );

        // The Mistral preset: exactly nine letters and digits.
        let mistral = crate::providers::Mistral::builder("k")
            .unwrap()
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        let (body, warnings) = convert(&request, mistral.quirks());
        assert!(warnings.is_empty(), "{warnings:?}");
        let (calls, results) = ids(&body);
        for id in &calls {
            assert!(
                id.len() == 9 && id.chars().all(|c| c.is_ascii_alphanumeric()),
                "{id}"
            );
        }
        assert_eq!(calls[2], "D681PevKs");
        assert_ne!(calls[0], calls[1]);
        assert_eq!(
            results,
            [calls[1].clone(), calls[2].clone(), calls[0].clone()]
        );
    }

    #[test]
    fn extra_cannot_set_owned_fields() {
        let request = ChatRequest::new("m").user("q");
        for key in CHAT_COMPLETION_FIELDS
            .iter()
            .copied()
            .chain(["Stream", "MAX_TOKENS"])
        {
            let mut options = WireOptions::default();
            options.extra.insert(key.into(), json!(1));
            let quirks = CompatQuirks {
                supports_top_k: true,
                ..CompatQuirks::default()
            };
            let err =
                to_chat_completion_request(&request, false, &options, &quirks, "demo").unwrap_err();
            assert!(
                matches!(err, Error::InvalidRequest { .. }),
                "{key}: {err:?}"
            );
            assert!(
                err.to_string().contains("ChatRequestOptions::extra"),
                "{err}"
            );
        }
        // Fields the adapter does not own pass through, `top_k` included
        // where the endpoint has no `top_k` of its own.
        let mut options = WireOptions::default();
        options.extra.insert("safe_prompt".into(), json!(true));
        options.extra.insert("top_k".into(), json!(5));
        let (body, _) =
            to_chat_completion_request(&request, false, &options, &CompatQuirks::default(), "demo")
                .unwrap();
        assert_eq!(serde_json::to_value(body).unwrap()["safe_prompt"], true);
    }

    #[test]
    fn every_body_field_is_reserved() {
        // A field added to the body must be reserved too, or `extra` could
        // send it twice.
        let options = WireOptions {
            user: Some("u".into()),
            safety_identifier: Some("s".into()),
            service_tier: Some("flex".into()),
            store: Some(true),
            metadata: Some(ExtraMap::new()),
            reasoning_effort: Some("low".into()),
            ..WireOptions::default()
        };
        let request = ChatRequest::new("m")
            .system("s")
            .user("q")
            .temperature(0.5)
            .top_p(0.9)
            .top_k(40)
            .max_output_tokens(10)
            .stop(["x"])
            .frequency_penalty(0.1)
            .presence_penalty(0.1)
            .seed(1)
            .parallel_tool_calls(true)
            .response_format(ResponseFormat::Json)
            .tools([ToolDefinition::new("f", json!({"type": "object"}))])
            .tool_choice(ToolChoice::Auto)
            .cache(allama::PromptCache::new().key("k"));
        let mistral_like = CompatQuirks {
            seed_field: SeedField::RandomSeed,
            supports_top_k: true,
            ..CompatQuirks::default()
        };
        for quirks in [CompatQuirks::openai(), mistral_like] {
            let (body, _) =
                to_chat_completion_request(&request, true, &options, &quirks, "demo").unwrap();
            for key in serde_json::to_value(body)
                .unwrap()
                .as_object()
                .unwrap()
                .keys()
            {
                assert!(
                    CHAT_COMPLETION_FIELDS.contains(&key.as_str()),
                    "{key} is not reserved"
                );
            }
        }
    }

    #[test]
    fn empty_turns_are_skipped_without_a_warning() {
        let quirks = CompatQuirks::openai();
        let baseline = ChatRequest::new("m")
            .user("one")
            .assistant("answer")
            .user("two");
        let foreign_only = Message::user_parts(vec![ContentPart::Other(
            allama::OtherContent::new("search_result", Default::default()).with_origin("anthropic"),
        )]);
        let with_empty = ChatRequest::new("m")
            .user("one")
            .message(Message::user(""))
            .message(Message::assistant_blocks(vec![]))
            .assistant("answer")
            .message(foreign_only)
            .message(Message::user_parts(vec![]))
            .user("two");
        let (expected, _) = convert(&baseline, &quirks);
        let (body, warnings) = convert(&with_empty, &quirks);
        assert_eq!(body, expected);
        // Only the foreign part's drop is reported, at the caller's index.
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
        let quirks = CompatQuirks::openai();
        for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
            let mut request = ChatRequest::new("m")
                .user("one")
                .assistant("answer")
                .message(Message::user(""));
            request.unsupported_policy = Some(policy);
            let (body, warnings) = convert(&request, &quirks);
            assert_eq!(
                body["messages"],
                json!([{"role": "user", "content": "one"}])
            );
            assert_eq!(warnings.len(), 1, "{warnings:?}");
            assert_eq!(warnings[0].kind, WarningKind::Dropped);
        }
        // An empty tool result is an answer, and is sent.
        let request = ChatRequest::new("m")
            .user("Weather?")
            .message(Message::assistant_blocks(vec![ContentBlock::tool_call(
                "c1", "f", "{}",
            )]))
            .message(Message::tool_result("c1", "f", ""));
        let (body, _) = convert(&request, &quirks);
        assert_eq!(body["messages"][2]["role"], "tool");
        assert_eq!(body["messages"][2]["content"], "");
    }

    /// Participant names are provider data addressed to the endpoint:
    /// sent from its own namespace, never from another provider's, and a
    /// value that is not a string is reported.
    #[test]
    fn participant_names_come_from_own_namespace_provider_data() {
        let named = |namespace: &str, value: serde_json::Value| {
            let mut message = Message::user("hi");
            message
                .provider_data_mut()
                .insert_raw(namespace, crate::PARTICIPANT_NAME_KEY, value);
            message
        };
        let mut assistant = Message::assistant("hello");
        assistant.provider_data_mut().insert_raw(
            "openai",
            crate::PARTICIPANT_NAME_KEY,
            "bot".into(),
        );
        let request = ChatRequest::new("gpt-5-mini")
            .message(named("openai", "ada".into()))
            .message(assistant)
            .message(named("anthropic", "eve".into()));
        let (body, _) = convert(&request, &CompatQuirks::openai());
        assert_eq!(body["messages"][0]["name"], "ada");
        assert_eq!(body["messages"][1]["name"], "bot");
        assert!(body["messages"][2].get("name").is_none());

        let request = ChatRequest::new("gpt-5-mini").message(named("openai", json!(7)));
        let (body, warnings) = convert(&request, &CompatQuirks::openai());
        assert!(body["messages"][0].get("name").is_none());
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].kind, WarningKind::Dropped);
    }

    #[test]
    fn openai_dialect_uses_max_completion_tokens_and_developer_role() {
        let request = ChatRequest::new("gpt-5-mini")
            .system(SystemPrompt::new("rules").with_option(SystemOptions::developer()))
            .user("hi")
            .max_output_tokens(100)
            .reasoning(ReasoningEffort::Minimal)
            .cache(PromptCache::new().key("s").retention(CacheRetention::Long));
        let (body, warnings) = convert(&request, &CompatQuirks::openai());
        assert_eq!(body["max_completion_tokens"], 100);
        assert!(body.get("max_tokens").is_none());
        assert_eq!(body["messages"][0]["role"], "developer");
        assert_eq!(body["reasoning_effort"], "minimal");
        assert_eq!(body["prompt_cache_key"], "s");
        assert_eq!(body["prompt_cache_retention"], "24h");

        // A long-retention breakpoint alone requests extended retention.
        let request = ChatRequest::new("gpt-5-mini").system(
            allama::SystemPrompt::new("stable prefix").with_cache(
                allama::CacheBreakpoint::with_retention(CacheRetention::Long),
            ),
        );
        let body = serde_json::to_value(
            to_chat_completion_request(
                &request,
                false,
                &WireOptions::default(),
                &CompatQuirks::openai(),
                "openai",
            )
            .unwrap()
            .0,
        )
        .unwrap();
        assert_eq!(body["prompt_cache_retention"], "24h");
        assert!(warnings.is_empty());
    }

    #[test]
    fn generic_dialect_warns_for_unsupported_hints_and_errors_under_strict() {
        let quirks = CompatQuirks {
            seed_field: SeedField::Unsupported,
            supports_parallel_tool_calls: false,
            ..CompatQuirks::default()
        };
        let request = ChatRequest::new("m")
            .system(SystemPrompt::new("rules").with_option(SystemOptions::developer()))
            .user("hi")
            .seed(3)
            .parallel_tool_calls(false)
            .cache(PromptCache::new().key("k"));
        let (body, warnings) = convert(&request, &quirks);
        assert_eq!(body["messages"][0]["role"], "system");
        assert!(body.get("seed").is_none());
        assert_eq!(warnings.len(), 4);
        assert!(warnings.iter().all(|w| w.kind == WarningKind::Unsupported));

        let strict = request.unsupported_policy(UnsupportedPolicy::Error);
        assert!(
            to_chat_completion_request(&strict, false, &WireOptions::default(), &quirks, "demo")
                .is_err()
        );
    }

    /// `true` is permission: an endpoint without the field drops it
    /// silently. `false` cannot be honoured there: a hint.
    #[test]
    fn parallel_tool_calls_true_is_never_a_degradation() {
        let quirks = CompatQuirks {
            supports_parallel_tool_calls: false,
            ..CompatQuirks::default()
        };
        let request = ChatRequest::new("m").user("hi").parallel_tool_calls(true);
        let (body, warnings) = convert(&request, &quirks);
        assert!(body.get("parallel_tool_calls").is_none());
        assert!(warnings.is_empty(), "{warnings:?}");
        let strict = request.unsupported_policy(UnsupportedPolicy::Error);
        assert!(
            to_chat_completion_request(&strict, false, &WireOptions::default(), &quirks, "demo")
                .is_ok()
        );

        let request = ChatRequest::new("m").user("hi").parallel_tool_calls(false);
        let (body, warnings) = convert(&request, &quirks);
        assert!(body.get("parallel_tool_calls").is_none());
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert_eq!(warnings[0].kind, WarningKind::Unsupported);
        assert_eq!(
            warnings[0].message,
            "this endpoint has no `parallel_tool_calls` field; `false` was not sent, so the model may still call several tools at once"
        );
        let strict = request.unsupported_policy(UnsupportedPolicy::Error);
        assert!(
            to_chat_completion_request(&strict, false, &WireOptions::default(), &quirks, "demo")
                .is_err()
        );
    }

    #[test]
    fn mistral_seed_goes_to_random_seed() {
        let quirks = CompatQuirks {
            seed_field: SeedField::RandomSeed,
            ..CompatQuirks::default()
        };
        let (body, _) = convert(&ChatRequest::new("m").user("x").seed(7), &quirks);
        assert_eq!(body["random_seed"], 7);
        assert!(body.get("seed").is_none());
    }

    /// The endpoint's grammar engine has rules allama does not know: the
    /// schema goes out strict as written, and an `Approximated` warning on
    /// each field says it was not checked.
    #[test]
    fn strict_schemas_without_a_dialect_are_sent_as_written() {
        let schema = json!({"type": "object", "properties": {"a": {"type": "string"}}});
        let request = ChatRequest::new("m")
            .user("x")
            .tools([ToolDefinition::new("f", schema.clone()).strict(true)])
            .response_format(ResponseFormat::json_schema(schema.clone()));
        let quirks = CompatQuirks {
            schema_dialect: None,
            ..CompatQuirks::default()
        };
        let (body, warnings) = convert(&request, &quirks);
        let reported: Vec<_> = warnings.iter().map(|w| (w.kind, &w.subject)).collect();
        assert_eq!(
            reported,
            [
                (
                    WarningKind::Approximated,
                    &Subject::Field {
                        field: RequestField::Tools
                    }
                ),
                (
                    WarningKind::Approximated,
                    &Subject::Field {
                        field: RequestField::ResponseFormat
                    }
                ),
            ]
        );
        assert!(warnings[0].message.contains("unchecked"), "{warnings:?}");
        assert_eq!(body["tools"][0]["function"]["parameters"], schema);
        assert_eq!(body["tools"][0]["function"]["strict"], true);
        assert_eq!(body["response_format"]["json_schema"]["schema"], schema);
        assert_eq!(body["response_format"]["json_schema"]["strict"], true);
    }

    #[test]
    fn strict_is_never_dropped_silently() {
        let schema = json!({"type": "object", "properties": {"a": {"type": "string"}}});
        let request = ChatRequest::new("m")
            .user("x")
            .tools([ToolDefinition::new("f", schema.clone()).strict(true)])
            .response_format(ResponseFormat::json_schema(schema.clone()));
        let quirks = CompatQuirks {
            response_format_constraint: SchemaConstraint::Never,
            tool_constraint: SchemaConstraint::Never,
            ..CompatQuirks::default()
        };
        let (body, warnings) = convert(&request, &quirks);
        assert_eq!(body["tools"][0]["function"]["parameters"], schema);
        assert!(body["tools"][0]["function"].get("strict").is_none());
        assert_eq!(body["response_format"]["json_schema"]["schema"], schema);
        assert!(
            body["response_format"]["json_schema"]
                .get("strict")
                .is_none()
        );
        let fields: Vec<_> = warnings
            .iter()
            .map(|w| (w.kind, w.subject.clone()))
            .collect();
        assert_eq!(
            fields,
            vec![
                (
                    WarningKind::Unsupported,
                    Subject::Field {
                        field: RequestField::Tools
                    }
                ),
                (
                    WarningKind::Unsupported,
                    Subject::Field {
                        field: RequestField::ResponseFormat
                    }
                ),
            ]
        );

        let strict = request.unsupported_policy(UnsupportedPolicy::Error);
        assert!(matches!(
            to_chat_completion_request(&strict, false, &WireOptions::default(), &quirks, "demo"),
            Err(allama::Error::Unsupported { .. })
        ));
    }

    /// A forced choice that constrains the forced call takes the place of
    /// strict mode for the forced tool only: it goes out without `strict`,
    /// reported as the dialect pass reports it, while an unforced strict
    /// tool is still `Unsupported`.
    #[test]
    fn a_constraining_forced_choice_enforces_the_forced_tool() {
        let schema = json!({"type": "object", "properties": {"a": {"type": "string"}}});
        let quirks = CompatQuirks {
            tool_constraint: SchemaConstraint::Never,
            forced_tool_choice_constrained: true,
            schema_dialect: None,
            ..CompatQuirks::default()
        };
        let request = ChatRequest::new("m")
            .user("x")
            .tools([
                ToolDefinition::new("f", schema.clone()).strict(true),
                ToolDefinition::new("g", schema.clone()).strict(true),
            ])
            .tool_choice(ToolChoice::Specific { name: "f".into() });
        let (body, warnings) = convert(&request, &quirks);
        assert!(body["tools"][0]["function"].get("strict").is_none());
        assert!(body["tools"][1]["function"].get("strict").is_none());
        let reported: Vec<_> = warnings
            .iter()
            .map(|w| (w.kind, w.message.as_str()))
            .collect();
        assert_eq!(reported.len(), 2, "{reported:?}");
        assert_eq!(reported[0].0, WarningKind::Approximated);
        assert!(reported[0].1.contains("tool `f`"), "{reported:?}");
        assert_eq!(reported[1].0, WarningKind::Unsupported);
        assert!(reported[1].1.contains("`g`"), "{reported:?}");

        // `required` forces every tool.
        let required = request.clone().tool_choice(ToolChoice::Required);
        let (_, warnings) = convert(&required, &quirks);
        assert!(
            warnings.iter().all(|w| w.kind == WarningKind::Approximated),
            "{warnings:?}"
        );

        // Without the quirk, both are `Unsupported`.
        let plain = CompatQuirks {
            forced_tool_choice_constrained: false,
            ..quirks
        };
        let (_, warnings) = convert(&request, &plain);
        assert!(
            warnings.iter().all(|w| w.kind == WarningKind::Unsupported),
            "{warnings:?}"
        );
    }

    /// `json_schema` asks for strict mode: the schema goes out strict,
    /// rewritten into the dialect (every property required, optional ones
    /// nullable). `non_strict` sends it as written with `strict: false`.
    /// OpenAI rejects a non-object root in a response format or a
    /// function's parameters, strict or not (HTTP 400, verified live), so
    /// the OpenAI dialect refuses it before sending; an endpoint without
    /// the rule is sent it.
    #[test]
    fn a_non_object_root_is_refused_where_the_endpoint_requires_one() {
        let label = json!({"type": "string", "enum": ["billing", "bug"]});
        let format = |strict: bool| {
            let format = ResponseFormat::json_schema(label.clone());
            ChatRequest::new("m").user("x").response_format(if strict {
                format
            } else {
                format.non_strict()
            })
        };
        let tool = ChatRequest::new("m")
            .user("x")
            .tools([ToolDefinition::new("label", label.clone())]);
        let refused = |request: &ChatRequest, quirks: &CompatQuirks| {
            to_chat_completion_request(request, false, &WireOptions::default(), quirks, "demo")
                .unwrap_err()
        };
        let openai = CompatQuirks::openai();
        for request in [format(true), format(false)] {
            let err = refused(&request, &openai);
            assert_eq!(
                err.unsupported_field(),
                Some(RequestField::ResponseFormat),
                "{err:?}"
            );
        }
        let err = refused(&tool, &openai);
        assert_eq!(
            err.unsupported_field(),
            Some(RequestField::Tools),
            "{err:?}"
        );

        let generic = CompatQuirks::new("openai-compat:demo");
        let (body, _) = convert(&format(false), &generic);
        assert_eq!(body["response_format"]["json_schema"]["schema"], label);
        let (body, _) = convert(&tool, &generic);
        assert_eq!(body["tools"][0]["function"]["parameters"], label);
    }

    #[test]
    fn json_schema_is_strict_unless_relaxed() {
        let schema = json!({"type": "object", "properties": {"a": {"type": "string"}}});
        let quirks = CompatQuirks::openai();
        let strict = ChatRequest::new("m")
            .user("x")
            .response_format(ResponseFormat::json_schema(schema.clone()));
        let (body, warnings) = convert(&strict, &quirks);
        let sent = &body["response_format"]["json_schema"];
        assert_eq!(sent["strict"], true);
        assert_eq!(sent["schema"]["required"], json!(["a"]));
        assert_eq!(sent["schema"]["additionalProperties"], false);
        assert!(warnings.is_empty(), "{warnings:?}");

        let relaxed = ChatRequest::new("m")
            .user("x")
            .response_format(ResponseFormat::json_schema(schema.clone()).non_strict());
        let (body, warnings) = convert(&relaxed, &quirks);
        let sent = &body["response_format"]["json_schema"];
        assert_eq!(sent["strict"], false);
        assert_eq!(sent["schema"], schema);
        assert!(warnings.is_empty(), "{warnings:?}");

        // Where strict mode is unavailable only the strict request is
        // reported.
        let without = CompatQuirks {
            response_format_constraint: SchemaConstraint::Never,
            ..quirks
        };
        let (_, warnings) = convert(&strict, &without);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        let (_, warnings) = convert(&relaxed, &without);
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    /// A custom endpoint's strict mode is rewritten into OpenAI's dialect,
    /// but not claimed exact: an `Approximated` warning says its
    /// enforcement is unverified.
    #[test]
    fn an_unverified_dialect_is_reported_approximated() {
        let schema =
            json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]});
        let request = ChatRequest::new("m")
            .user("x")
            .response_format(ResponseFormat::json_schema(schema));
        let (body, warnings) = convert(&request, &CompatQuirks::default());
        assert_eq!(body["response_format"]["json_schema"]["strict"], true);
        assert_eq!(
            body["response_format"]["json_schema"]["schema"]["additionalProperties"],
            false
        );
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert_eq!(warnings[0].kind, WarningKind::Approximated);
        assert!(warnings[0].message.contains("not known to enforce"));

        let verified = CompatQuirks {
            dialect_verified: true,
            ..CompatQuirks::default()
        };
        let (_, warnings) = convert(&request, &verified);
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    #[test]
    fn strict_tools_and_schemas_are_rewritten_into_the_dialect() {
        let request = ChatRequest::new("m")
            .user("x")
            .tools([ToolDefinition::new(
                "f",
                json!({"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}, "required": ["a"]}),
            )
            .strict(true)])
            .response_format(ResponseFormat::json_schema(json!({"type": "object", "properties": {"x": {"type": "string"}}})));
        let (body, _) = convert(&request, &CompatQuirks::openai());
        let params = &body["tools"][0]["function"]["parameters"];
        assert_eq!(params["additionalProperties"], false);
        assert_eq!(params["required"], json!(["a", "b"]));
        assert_eq!(
            params["properties"]["b"]["type"],
            json!(["integer", "null"])
        );
        assert_eq!(body["response_format"]["json_schema"]["name"], "response");
        assert_eq!(body["response_format"]["json_schema"]["strict"], true);
    }

    #[test]
    fn files_map_to_file_and_audio_parts() {
        let request = ChatRequest::new("m").user(vec![
            ContentPart::file_base64("application/pdf", "JVBE"),
            ContentPart::file_base64("audio/wav", "UklG"),
        ]);
        let (body, _) = convert(&request, &CompatQuirks::openai());
        let parts = &body["messages"][0]["content"];
        assert_eq!(parts[0]["type"], "file");
        assert_eq!(
            parts[0]["file"]["file_data"],
            "data:application/pdf;base64,JVBE"
        );
        assert_eq!(parts[1]["input_audio"]["format"], "wav");

        let url = ChatRequest::new("m").user(vec![ContentPart::file_url(
            "application/pdf",
            "https://x/a.pdf",
        )]);
        assert!(
            to_chat_completion_request(
                &url,
                false,
                &WireOptions::default(),
                &CompatQuirks::openai(),
                "demo"
            )
            .is_err()
        );
    }

    #[test]
    fn replay_drops_foreign_reasoning_and_repairs_malformed_arguments() {
        let foreign = Reasoning::new("think").with_provider_data(ProviderData::new().with_raw(
            "anthropic",
            "signature",
            json!("s"),
        ));
        let request = ChatRequest::new("m").extend_messages([
            Message::user("q"),
            Message::assistant_blocks(vec![
                ContentBlock::Reasoning(foreign),
                ContentBlock::text("a"),
                ContentBlock::tool_call("c1", "f", "{bad"),
            ]),
            Message::tool_error("c1", "f", "boom"),
        ]);
        let (body, warnings) = convert(&request, &CompatQuirks::openai());
        assert_eq!(
            body["messages"][1]["tool_calls"][0]["function"]["arguments"],
            "{}"
        );
        assert!(body["messages"][1].get("reasoning_content").is_none());
        assert_eq!(body["messages"][2]["content"], "boom");
        let kinds: Vec<_> = warnings.iter().map(|w| w.kind).collect();
        assert!(kinds.contains(&WarningKind::Dropped));
        // Malformed arguments recur with the history, so they are coded.
        let arguments = warnings
            .iter()
            .find(|w| w.code.as_deref() == Some(TOOL_CALL_ARGUMENTS))
            .expect("the malformed arguments warning");
        assert_eq!(arguments.kind, WarningKind::Approximated);
        assert_eq!(
            arguments.message,
            "the arguments of tool call `f` are not a JSON object; they were sent as `{}`"
        );
        let flag = warnings
            .iter()
            .find(|w| w.code.as_deref() == Some(TOOL_RESULT_ERROR))
            .expect("the tool error flag warning");
        assert_eq!(flag.kind, WarningKind::Approximated);
        assert_eq!(
            flag.subject,
            Subject::Content {
                message: 2,
                block: None
            }
        );
    }

    /// A tool error from history written against a provider with an error
    /// flag (Anthropic's `is_error`, say) is sent as its error text with a
    /// coded warning, under a strict policy too: the model still reads the
    /// error, only the flag is lost, so there is nothing to refuse.
    #[test]
    fn tool_errors_are_sent_under_a_strict_policy_even_in_history() {
        let request = ChatRequest::new("m")
            .extend_messages([
                Message::user("weather?"),
                Message::assistant_blocks(vec![ContentBlock::tool_call("c1", "f", "{}")]),
                Message::tool_error("c1", "f", "the weather service is down"),
                Message::assistant("It is down."),
            ])
            .user("try again");
        for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
            let request = request.clone().unsupported_policy(policy);
            let (body, warnings) = convert(&request, &CompatQuirks::openai());
            assert_eq!(body["messages"][2]["role"], "tool");
            assert_eq!(
                body["messages"][2]["content"],
                "the weather service is down"
            );
            assert_eq!(warnings.len(), 1, "{policy:?}: {warnings:?}");
            assert_eq!(warnings[0].kind, WarningKind::Approximated);
            assert_eq!(warnings[0].code.as_deref(), Some(TOOL_RESULT_ERROR));
            assert_eq!(
                warnings[0].subject,
                Subject::Content {
                    message: 2,
                    block: None
                }
            );
            assert_eq!(
                warnings[0].message,
                "Chat Completions has no tool error flag; the error text was sent as the tool result; the model still reads the error text; to send your own encoding to this endpoint, use `push_tool_result`"
            );
        }
    }

    /// One warning per request however many tool errors the history holds,
    /// listing their messages.
    #[test]
    fn tool_errors_join_one_warning_per_request() {
        let call = |id| Message::assistant_blocks(vec![ContentBlock::tool_call(id, "f", "{}")]);
        let request = ChatRequest::new("m")
            .extend_messages([
                Message::user("weather?"),
                call("c1"),
                Message::tool_error("c1", "f", "down"),
                call("c2"),
                Message::tool_error("c2", "f", "still down"),
            ])
            .unsupported_policy(UnsupportedPolicy::Error);
        let (_, warnings) = convert(&request, &CompatQuirks::openai());
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert_eq!(
            warnings[0].subject,
            Subject::Messages {
                messages: vec![2, 4]
            }
        );
    }

    #[test]
    fn tool_object_results_are_serialized() {
        let request = ChatRequest::new("m").extend_messages([Message::tool_result(
            "c",
            "f",
            ToolResultContent::json(&json!({"ok": true})).unwrap(),
        )
        .with_cache(CacheBreakpoint::new())]);
        let (body, _) = convert(&request, &CompatQuirks::openai());
        assert_eq!(body["messages"][0]["content"], r#"{"ok":true}"#);
    }

    /// Regression: text parts of one tool result
    /// were concatenated with no separator, so "before" and "after" read as
    /// "beforeafter". They are joined as Gemini joins them.
    #[test]
    fn multi_part_tool_results_keep_text_boundaries() {
        let request = ChatRequest::new("m").extend_messages([Message::tool_result(
            "c",
            "f",
            ToolResultContent::Parts(vec![
                ContentPart::text("before"),
                ContentPart::text("after"),
            ]),
        )]);
        let (body, warnings) = convert(&request, &CompatQuirks::openai());
        assert_eq!(body["messages"][0]["content"], "before\n\nafter");
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    /// The finish-reason aliases vendors send map to
    /// the standard reasons, and anything else is kept as `Other`.
    #[test]
    fn finish_reason_aliases() {
        for (raw, expected) in [
            ("stop", FinishReason::Stop),
            ("eos", FinishReason::Stop),
            ("eos_token", FinishReason::Stop),
            ("end_turn", FinishReason::Stop),
            ("length", FinishReason::Length),
            ("max_tokens", FinishReason::Length),
            ("model_length", FinishReason::Length),
            ("tool_calls", FinishReason::ToolCalls),
            ("function_call", FinishReason::ToolCalls),
            ("content_filter", FinishReason::ContentFilter),
            ("error", FinishReason::Other("error".into())),
        ] {
            assert_eq!(parse_finish_reason(raw), expected, "{raw}");
        }
    }

    /// Unary path: a turn with tool calls
    /// that the endpoint finishes with `stop` (OpenAI for a forced tool
    /// choice, Gemini and vLLM always) stopped for the tool calls.
    #[test]
    fn unary_stop_with_tool_calls_is_tool_calls() {
        let response: ChatCompletionResponse = serde_json::from_value(json!({
            "id": "c1",
            "model": "m",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
                },
                "finish_reason": "stop"
            }]
        }))
        .unwrap();
        let converted = from_api_response(response, ctx()).unwrap();
        assert_eq!(converted.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(converted.finish_reason_raw.as_deref(), Some("stop"));
    }

    /// Regression: `prompt + completion` overflowed
    /// (a panic in debug builds, a wrapped total in release).
    #[test]
    fn absurd_usage_counters_saturate() {
        let usage: ApiUsage = serde_json::from_value(json!({
            "prompt_tokens": u64::MAX,
            "completion_tokens": 5,
            "total_tokens": 0
        }))
        .unwrap();
        assert_eq!(from_api_usage(&usage).total_tokens, Some(u64::MAX));
    }

    #[test]
    fn response_decodes_reasoning_refusal_annotations_and_usage() {
        let response: ChatCompletionResponse = serde_json::from_value(json!({
            "id": "chatcmpl-1",
            "model": "m",
            "system_fingerprint": "fp",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant",
                "reasoning_content": "thinking",
                "content": "hello",
                "annotations": [{"type": "url_citation"}],
                "refusal": null
            }}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                      "prompt_tokens_details": {"cached_tokens": 80},
                      "completion_tokens_details": {"reasoning_tokens": 5}}
        }))
        .unwrap();
        let converted = from_api_response(response, ctx()).unwrap();
        assert_eq!(converted.reasoning_text().as_deref(), Some("thinking"));
        assert_eq!(converted.text().as_deref(), Some("hello"));
        assert_eq!(converted.provider.as_deref(), Some("demo"));
        assert_eq!(converted.request_id.as_deref(), Some("req_1"));
        assert_eq!(converted.finish_reason_raw.as_deref(), Some("stop"));
        let usage = converted.usage.unwrap();
        assert_eq!(usage.cache_read_input_tokens, Some(80));
        assert_eq!(usage.reasoning_tokens, Some(5));
        assert!(usage.contract_violation().is_none());
        assert!(
            converted.content[1]
                .provider_data()
                .get_raw("openai-compat:demo", ANNOTATIONS_KEY)
                .is_some()
        );
        // Chat annotations live under their own key, which the
        // Responses encoder (sharing the namespace) never replays.
        assert!(
            converted.content[1]
                .provider_data()
                .get_raw("openai-compat:demo", "annotations")
                .is_none()
        );
        // The reasoning is marked as the endpoint's own.
        assert_eq!(
            converted.content[0]
                .provider_data()
                .get_raw("openai-compat:demo", REASONING_FIELD_KEY),
            Some(&json!("reasoning_content"))
        );
    }

    /// An endpoint that ignores `tool_choice` (Ollama): `Disabled` leaves the
    /// tools out without a warning, under every policy, since that is
    /// exactly "the model must not call tools"; a forced choice is refused
    /// before sending.
    #[test]
    fn disabled_tools_are_left_out_where_tool_choice_is_ignored() {
        let mut quirks = CompatQuirks::new("openai-compat:ollama");
        quirks.supports_tool_choice = false;
        let tool = ToolDefinition::new("lookup", json!({"type": "object"}));
        for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
            let request = ChatRequest::new("m")
                .user("hi")
                .tools([tool.clone()])
                .tool_choice(ToolChoice::Disabled)
                .unsupported_policy(policy);
            let (body, warnings) = convert(&request, &quirks);
            assert!(body.get("tools").is_none(), "{body}");
            assert!(body.get("tool_choice").is_none(), "{body}");
            assert!(warnings.is_empty(), "{warnings:?}");
        }

        let auto = ChatRequest::new("m")
            .user("hi")
            .tools([tool.clone()])
            .tool_choice(ToolChoice::Auto);
        let (body, warnings) = convert(&auto, &quirks);
        assert!(body.get("tool_choice").is_none(), "{body}");
        assert!(body.get("tools").is_some(), "{body}");
        assert!(warnings.is_empty(), "{warnings:?}");

        let forced = ChatRequest::new("m")
            .user("hi")
            .tools([tool])
            .tool_choice(ToolChoice::Required);
        let err =
            to_chat_completion_request(&forced, false, &WireOptions::default(), &quirks, "demo")
                .unwrap_err();
        assert_eq!(err.unsupported_field(), Some(RequestField::ToolChoice));
    }

    /// A structured-output refusal as OpenAI returns it
    /// (<https://platform.openai.com/docs/guides/structured-outputs#refusals>):
    /// `content` is null and the refusal is in `message.refusal`.
    #[test]
    fn a_refusal_decodes_as_a_refusal_block() {
        let refusal: ChatCompletionResponse = serde_json::from_value(json!({
            "id": "chatcmpl-9nYAG9LPNonX8DAyrkwYfemr3C8HC",
            "object": "chat.completion",
            "model": "gpt-4o-2024-08-06",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "refusal": "I'm sorry, I cannot assist with that request."
                },
                "logprobs": null,
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 81, "completion_tokens": 11, "total_tokens": 92}
        }))
        .unwrap();
        let converted = from_api_response(refusal, ctx()).unwrap();
        assert_eq!(
            converted.content,
            [ContentBlock::Refusal(Refusal::new(
                "I'm sorry, I cannot assist with that request."
            ))]
        );
        assert_eq!(
            converted.first_refusal(),
            Some("I'm sorry, I cannot assist with that request.")
        );
        assert_eq!(converted.text(), None);
        assert_eq!(converted.finish_reason, Some(FinishReason::Refusal));
        assert_eq!(converted.finish_reason_raw.as_deref(), Some("stop"));
    }

    /// OpenAI takes a refusal back in the assistant message's `refusal`
    /// field; a dialect without it gets the refusal as assistant text. A
    /// text block an earlier version flagged as a refusal replays the same
    /// way.
    #[test]
    fn refusals_replay_natively_or_as_text() {
        let legacy = |ns: &str| {
            ContentBlock::Text(
                Text::new("Not that.").with_provider_data(ProviderData::new().with_raw(
                    ns,
                    "refusal",
                    true.into(),
                )),
            )
        };
        let request = |ns: &str| {
            ChatRequest::new("m")
                .user("help")
                .message(Message::assistant_blocks(vec![
                    ContentBlock::text("Hmm."),
                    ContentBlock::Refusal(Refusal::new("I can't help with that.")),
                    legacy(ns),
                ]))
                .user("why?")
        };

        let (body, warnings) = convert(&request("openai"), &CompatQuirks::openai());
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(
            body["messages"][1],
            json!({
                "role": "assistant",
                "content": "Hmm.",
                "refusal": "I can't help with that.Not that."
            })
        );

        let quirks = CompatQuirks::new("openai-compat:demo");
        let (body, warnings) = convert(&request("openai-compat:demo"), &quirks);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(
            body["messages"][1],
            json!({"role": "assistant", "content": "Hmm.I can't help with that.Not that."})
        );
    }

    #[test]
    fn array_and_object_content_is_flattened() {
        let pieces = content_pieces(Some(
            json!([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]),
        ));
        assert_eq!(
            pieces,
            [
                ContentPiece::Text("a".into()),
                ContentPiece::Text("b".into())
            ]
        );
        let pieces = content_pieces(Some(json!({"x": 1})));
        assert_eq!(pieces, [ContentPiece::Text(r#"{"x":1}"#.into())]);
    }

    #[test]
    fn reasoning_summary_request_warns() {
        let request = ChatRequest::new("m")
            .user("x")
            .reasoning(ReasoningConfig::new().output(ReasoningOutput::Summary));
        let (_, warnings) = convert(&request, &CompatQuirks::openai());
        assert_eq!(warnings.len(), 1);
    }

    /// Model rules: with reasoning on (requested, or the model's default
    /// effort), sampling fields and log probabilities are dropped as hints,
    /// and tools are refused on a model that takes them only without
    /// reasoning.
    #[test]
    fn model_rules_follow_the_effective_effort() {
        use crate::{SamplingSupport, ToolCallSupport};
        let rules = ChatModelRules::new()
            .sampling(SamplingSupport::WithoutReasoning)
            .tool_calls(ToolCallSupport::WithoutReasoning)
            .default_effort(ReasoningEffort::Medium);
        let options = WireOptions {
            extra: serde_json::Map::from_iter([
                ("logprobs".into(), serde_json::json!(true)),
                ("top_logprobs".into(), serde_json::json!(2)),
            ]),
            ..WireOptions::default()
        };
        let request = ChatRequest::new("m").user("q").temperature(0.2).top_p(0.9);
        let encode = |request: &ChatRequest| {
            encode_chat_completion(
                request,
                false,
                &options,
                &CompatQuirks::openai(),
                &rules,
                "demo",
            )
        };

        let (body, warnings) = encode(&request).unwrap();
        assert!(body.temperature.is_none() && body.top_p.is_none());
        assert!(!body.extra.contains_key("logprobs") && !body.extra.contains_key("top_logprobs"));
        assert_eq!(warnings.len(), 4, "{warnings:?}");

        // Reasoning off: everything is sent.
        let off = request.clone().reasoning(ReasoningEffort::None);
        let (body, warnings) = encode(&off).unwrap();
        assert_eq!(body.temperature, Some(0.2));
        assert!(body.extra.contains_key("logprobs"));
        assert!(warnings.is_empty());

        let tools = request.clone().tools(vec![ToolDefinition::new(
            "t",
            serde_json::json!({"type": "object"}),
        )]);
        let err = encode(&tools).unwrap_err();
        assert_eq!(err.unsupported_field(), Some(RequestField::Tools));
        assert!(encode(&tools.reasoning(ReasoningEffort::None)).is_ok());

        // Without rules, nothing changes.
        let (body, warnings) =
            to_chat_completion_request(&request, false, &options, &CompatQuirks::openai(), "demo")
                .unwrap();
        assert_eq!(body.temperature, Some(0.2));
        assert!(warnings.is_empty());
    }

    /// Replay state stays in a stored conversation, so a conversation that
    /// moved here from another provider drops it on every turn: each drop is
    /// coded, so its warning event goes out at `WARN` once per model, not once
    /// per turn, and points to `retain_provider_state`.
    #[test]
    fn a_long_foreign_reasoning_history_drops_with_codes() {
        let mut request = ChatRequest::new("m");
        for turn in 0..10 {
            request = request
                .user(format!("question {turn}"))
                .message(Message::assistant_blocks(vec![
                    ContentBlock::Reasoning(Reasoning::new("thoughts").with_provider_data(
                        ProviderData::new().with_raw("anthropic", "signature", json!("opaque")),
                    )),
                    ContentBlock::text("answer"),
                ]));
        }
        let (_, warnings) = convert(&request.user("next"), &CompatQuirks::openai());
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
