//! Wire types for `POST /responses`.
//!
//! Requests are typed (with raw escape hatches for items replayed verbatim);
//! responses are read leniently: output items are decoded by `type` from raw
//! JSON so unknown item kinds survive as `Other` content.

use allama::ExtraMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Request body.
#[derive(Debug, Serialize)]
pub(crate) struct CreateResponse {
    pub model: String,
    pub input: Vec<InputItem>,
    /// `false` unless the caller opts into server-side state.
    pub store: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ExtraMap>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,
    #[serde(flatten)]
    pub extra: ExtraMap,
}

/// `text` request parameter.
#[derive(Debug, Default, Serialize)]
pub(crate) struct TextConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
}

/// `reasoning` request parameter.
#[derive(Debug, Default, Serialize)]
pub(crate) struct ReasoningParam {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<&'static str>,
}

/// One `input` item: a typed item, or an item from this endpoint replayed
/// verbatim (hosted tool calls, …).
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum InputItem {
    Known(KnownItem),
    Raw(ExtraMap),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum KnownItem {
    Message {
        role: &'static str,
        content: Vec<ContentParam>,
        #[serde(skip_serializing_if = "Option::is_none")]
        phase: Option<Value>,
    },
    FunctionCall {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        call_id: String,
        name: String,
        arguments: String,
        /// `Some(Value::Null)` sends `"status": null`
        /// ([`ResponsesQuirks::function_call_status_null`](crate::ResponsesQuirks::function_call_status_null)).
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<Value>,
    },
    FunctionCallOutput {
        call_id: String,
        output: FunctionOutput,
    },
    Reasoning {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Absent for raw-content replay (vLLM rejects the field).
        #[serde(skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
        /// Required by the API, possibly empty.
        summary: Vec<SummaryText>,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<Value>,
    },
}

/// `function_call_output.output`: a string, or a list of input content.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum FunctionOutput {
    Text(String),
    Parts(Vec<ContentParam>),
}

/// A content part of an input message or tool output.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum ContentParam {
    Known(KnownContent),
    Raw(ExtraMap),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum KnownContent {
    InputText {
        text: String,
    },
    InputImage {
        #[serde(skip_serializing_if = "Option::is_none")]
        image_url: Option<String>,
        /// A Files API id, instead of `image_url`.
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<&'static str>,
    },
    InputFile {
        /// A Files API id, instead of the data or URL.
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_url: Option<String>,
    },
    OutputText {
        text: String,
        annotations: Value,
    },
    Refusal {
        refusal: String,
    },
}

#[derive(Debug, Serialize)]
pub(crate) struct SummaryText {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
}

impl SummaryText {
    pub(crate) fn new(text: String) -> Self {
        Self {
            kind: "summary_text",
            text,
        }
    }
}

/// Response object (`POST /responses` body, and the `response` field of
/// lifecycle stream events).
#[derive(Debug, Default, Deserialize)]
pub(crate) struct ResponseObject {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub output: Vec<Value>,
    #[serde(default)]
    pub incomplete_details: Option<IncompleteDetails>,
    #[serde(default)]
    pub error: Option<ResponseError>,
    #[serde(default)]
    pub usage: Option<ResponseUsage>,
    #[serde(default)]
    pub service_tier: Option<String>,
    /// Hosted tool usage (`{"web_search": {"num_requests": 1},
    /// "image_gen": {…tokens}}`).
    #[serde(default)]
    pub tool_usage: Option<Value>,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct IncompleteDetails {
    #[serde(default)]
    pub reason: Option<String>,
}

/// `response.error` and the payload of `error` stream events.
#[derive(Debug, Default, Clone, Deserialize)]
pub(crate) struct ResponseError {
    /// A string, or a number some endpoints send (an HTTP status).
    #[serde(default, deserialize_with = "crate::wire::deserialize_code")]
    pub code: Option<String>,
    #[serde(default, rename = "type")]
    pub error_type: Option<String>,
    #[serde(default)]
    pub message: String,
    /// The rejected parameter, on `error` stream events.
    #[serde(default)]
    pub param: Option<String>,
}

#[derive(Debug, Default, Clone, Deserialize)]
pub(crate) struct ResponseUsage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub input_tokens_details: Option<InputTokensDetails>,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub output_tokens_details: Option<OutputTokensDetails>,
    #[serde(default)]
    pub total_tokens: Option<u64>,
}

#[derive(Debug, Default, Clone, Deserialize)]
pub(crate) struct InputTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u64>,
    #[serde(default)]
    pub cache_write_tokens: Option<u64>,
}

#[derive(Debug, Default, Clone, Deserialize)]
pub(crate) struct OutputTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: Option<u64>,
}
