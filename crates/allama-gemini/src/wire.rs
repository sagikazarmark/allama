//! Serde types for the Gemini `generateContent` wire format.
//!
//! Only the shapes `allama-gemini` reads or writes are modelled. Response
//! parts keep unknown keys (`executableCode`, `codeExecutionResult`, …) so
//! the decoder can preserve them verbatim as [`allama::OtherContent`].

use allama::ExtraMap;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GenerateContentRequest {
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<SystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<GeminiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<ToolConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub safety_settings: Vec<crate::SafetySetting>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_content: Option<String>,
    /// `ChatRequestOptions::extra`, checked against [`RESERVED_FIELDS`].
    #[serde(flatten)]
    pub extra: ExtraMap,
}

/// The body fields of [`GenerateContentRequest`]: the adapter owns them, so
/// `ChatRequestOptions::extra` cannot set them.
pub(crate) const RESERVED_FIELDS: &[&str] = &[
    "cachedContent",
    "contents",
    "generationConfig",
    "model",
    "safetySettings",
    "systemInstruction",
    "toolConfig",
    "tools",
];

#[derive(Debug, Serialize)]
pub(crate) struct Content {
    pub role: &'static str,
    pub parts: Vec<Part>,
}

#[derive(Debug, Serialize)]
pub(crate) struct SystemInstruction {
    pub parts: Vec<Part>,
}

/// One request part. `source` records where the part came from in the
/// portable request (message and block index) for warnings; it is never
/// serialized.
#[derive(Debug, Serialize)]
pub(crate) struct Part {
    #[serde(flatten)]
    pub data: PartData,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought: Option<bool>,
    #[serde(rename = "thoughtSignature", skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    /// Per-part media resolution (`{"level": "MEDIA_RESOLUTION_LOW"}`).
    #[serde(rename = "mediaResolution", skip_serializing_if = "Option::is_none")]
    pub media_resolution: Option<MediaResolution>,
    #[serde(skip)]
    pub source: Option<(usize, usize)>,
}

/// A part's `mediaResolution`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) struct MediaResolution {
    pub level: &'static str,
}

impl Part {
    pub(crate) fn new(data: PartData) -> Self {
        Self {
            data,
            thought: None,
            thought_signature: None,
            media_resolution: None,
            source: None,
        }
    }

    pub(crate) fn text(text: impl Into<String>) -> Self {
        Self::new(PartData::Text(text.into()))
    }

    pub(crate) fn signature(mut self, signature: Option<String>) -> Self {
        self.thought_signature = signature;
        self
    }

    pub(crate) fn source(mut self, message: usize, block: usize) -> Self {
        self.source = Some((message, block));
        self
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum PartData {
    Text(String),
    InlineData(Blob),
    FileData(FileData),
    FunctionCall(FunctionCall),
    FunctionResponse(FunctionResponse),
    /// A part replayed verbatim (an `OtherContent` this crate decoded).
    #[serde(untagged)]
    Raw(ExtraMap),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Blob {
    pub mime_type: String,
    pub data: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FileData {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    pub file_uri: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct FunctionCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub name: String,
    #[serde(default)]
    pub args: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub(crate) struct FunctionResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub name: String,
    pub response: serde_json::Value,
    /// Multimodal function response parts (Gemini 3+).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub parts: Vec<FunctionResponsePart>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FunctionResponsePart {
    pub inline_data: Blob,
}

/// One `tools` entry: function declarations, or a built-in tool definition
/// (`{"google_search": {}}`, …) sent verbatim.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum GeminiTool {
    Functions {
        #[serde(rename = "functionDeclarations")]
        function_declarations: Vec<FunctionDeclaration>,
    },
    Builtin(serde_json::Map<String, serde_json::Value>),
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FunctionDeclaration {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Full JSON Schema (including `$defs`/`$ref`), sent unmodified.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters_json_schema: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ToolConfig {
    pub function_calling_config: FunctionCallingConfig,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FunctionCallingConfig {
    pub mode: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>,
}

#[derive(Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
    /// JSON Schema for structured output (supersedes the deprecated
    /// OpenAPI-subset `responseSchema`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_json_schema: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<ThinkingConfig>,
    /// `ChatRequestOptions::generation_config_extra`, checked against
    /// [`GENERATION_CONFIG_FIELDS`].
    #[serde(flatten)]
    pub extra: ExtraMap,
}

/// The fields of [`GenerationConfig`]: the adapter owns them, so
/// `ChatRequestOptions::generation_config_extra` cannot set them. With
/// `candidateCount`, which the adapter never sends: a response holds one
/// candidate.
pub(crate) const GENERATION_CONFIG_FIELDS: &[&str] = &[
    "candidateCount",
    "frequencyPenalty",
    "maxOutputTokens",
    "presencePenalty",
    "responseJsonSchema",
    "responseMimeType",
    "seed",
    "stopSequences",
    "temperature",
    "thinkingConfig",
    "topK",
    "topP",
];

impl GenerationConfig {
    pub(crate) fn is_empty(&self) -> bool {
        self.temperature.is_none()
            && self.top_p.is_none()
            && self.top_k.is_none()
            && self.max_output_tokens.is_none()
            && self.stop_sequences.is_none()
            && self.seed.is_none()
            && self.presence_penalty.is_none()
            && self.frequency_penalty.is_none()
            && self.response_mime_type.is_none()
            && self.response_json_schema.is_none()
            && self.thinking_config.is_none()
            && self.extra.is_empty()
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ThinkingConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_thoughts: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<i32>,
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

/// A `generateContent` response, or one `streamGenerateContent` chunk.
///
/// Streams may carry an `error` object instead of a chunk.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GenerateContentResponse {
    #[serde(default)]
    pub response_id: Option<String>,
    #[serde(default)]
    pub model_version: Option<String>,
    #[serde(default)]
    pub prompt_feedback: Option<PromptFeedback>,
    #[serde(default)]
    pub candidates: Vec<Candidate>,
    #[serde(default)]
    pub usage_metadata: Option<UsageMetadata>,
    #[serde(default)]
    pub error: Option<GeminiError>,
    /// Response fields this adapter does not model, kept as response
    /// provider data.
    #[serde(flatten)]
    pub extra: ExtraMap,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PromptFeedback {
    #[serde(default)]
    pub block_reason: Option<String>,
    #[serde(default)]
    pub block_reason_message: Option<String>,
    /// `safetyRatings` and anything else, kept as response provider data.
    #[serde(flatten)]
    pub extra: ExtraMap,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Candidate {
    #[serde(default)]
    pub content: Option<ResponseContent>,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default)]
    pub finish_message: Option<String>,
    /// Search grounding (Google Search, retrieval), kept raw.
    #[serde(default)]
    pub grounding_metadata: Option<serde_json::Value>,
    /// Recitation sources, mapped to citations and kept raw.
    #[serde(default)]
    pub citation_metadata: Option<serde_json::Value>,
    /// Candidate fields this adapter does not model (`safetyRatings`,
    /// `urlContextMetadata`, `logprobsResult`, `avgLogprobs`, …), kept as
    /// response provider data.
    #[serde(flatten)]
    pub extra: ExtraMap,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct ResponseContent {
    #[serde(default)]
    pub parts: Vec<ResponsePart>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ResponsePart {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub thought: Option<bool>,
    #[serde(default)]
    pub thought_signature: Option<String>,
    #[serde(default)]
    pub function_call: Option<FunctionCall>,
    #[serde(default)]
    pub inline_data: Option<Blob>,
    #[serde(default)]
    pub file_data: Option<FileData>,
    /// Any other part payload (`executableCode`, …).
    #[serde(flatten)]
    pub other: ExtraMap,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UsageMetadata {
    #[serde(default)]
    pub prompt_token_count: Option<u64>,
    #[serde(default)]
    pub candidates_token_count: Option<u64>,
    #[serde(default)]
    pub total_token_count: Option<u64>,
    #[serde(default)]
    pub cached_content_token_count: Option<u64>,
    #[serde(default)]
    pub thoughts_token_count: Option<u64>,
    #[serde(default)]
    pub tool_use_prompt_token_count: Option<u64>,
    /// Input tokens by modality.
    #[serde(default)]
    pub prompt_tokens_details: Vec<ModalityTokenCount>,
    /// Input tokens from tool results, by modality.
    #[serde(default)]
    pub tool_use_prompt_tokens_details: Vec<ModalityTokenCount>,
    /// Output tokens by modality.
    #[serde(default)]
    pub candidates_tokens_details: Vec<ModalityTokenCount>,
    #[serde(default)]
    pub service_tier: Option<String>,
    /// Usage fields this adapter does not model (`cacheTokensDetails`,
    /// `trafficType`, …), kept as response provider data.
    #[serde(flatten)]
    pub extra: ExtraMap,
}

/// One `ModalityTokenCount`: the tokens of one modality.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ModalityTokenCount {
    #[serde(default)]
    pub modality: Option<String>,
    #[serde(default)]
    pub token_count: Option<u64>,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub(crate) struct GeminiErrorResponse {
    pub error: GeminiError,
}

/// A `google.rpc.Status` error object.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct GeminiError {
    #[serde(default)]
    pub code: Option<u16>,
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub details: Vec<serde_json::Value>,
}
