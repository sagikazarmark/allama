//! Anthropic Messages API wire types (crate-private).
//!
//! Request types serialize exactly what the adapter sends. Response content
//! blocks are decoded from raw JSON values in `decode.rs` so unknown block
//! types survive as [`allama::OtherContent`].

use allama::ExtraMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Serialize)]
pub(crate) struct CreateMessageRequest {
    pub model: String,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ExtraMap>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Vec<SystemBlock>>,
    pub messages: Vec<MessageParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolParam>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<OutputConfig>,
    /// Automatic (top-level) prompt caching.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    pub stream: bool,
    /// `ChatRequestOptions::extra`, checked against [`RESERVED_FIELDS`]
    /// before it gets here.
    #[serde(flatten)]
    pub extra: ExtraMap,
}

/// The body fields of [`CreateMessageRequest`]: the adapter owns them, so
/// `ChatRequestOptions::extra` cannot set them.
pub(crate) const RESERVED_FIELDS: &[&str] = &[
    "cache_control",
    "max_tokens",
    "messages",
    "metadata",
    "model",
    "output_config",
    "service_tier",
    "stop_sequences",
    "stream",
    "system",
    "temperature",
    "thinking",
    "tool_choice",
    "tools",
    "top_k",
    "top_p",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct CacheControl {
    #[serde(rename = "type")]
    pub kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<&'static str>,
}

impl CacheControl {
    pub(crate) const LONG_TTL: &'static str = "1h";

    pub(crate) fn ephemeral(ttl: Option<&'static str>) -> Self {
        Self {
            kind: "ephemeral",
            ttl,
        }
    }

    pub(crate) fn is_long(&self) -> bool {
        self.ttl == Some(Self::LONG_TTL)
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct SystemBlock {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

#[derive(Debug, Serialize)]
pub(crate) struct MessageParam {
    pub role: &'static str,
    pub content: Vec<RequestBlock>,
}

/// A request content block plus its optional cache marker.
#[derive(Debug, Serialize)]
pub(crate) struct RequestBlock {
    #[serde(flatten)]
    pub block: BlockParam,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl RequestBlock {
    pub(crate) fn new(block: impl Into<BlockParam>) -> Self {
        Self {
            block: block.into(),
            cache_control: None,
        }
    }

    /// A client tool's `tool_result`, which must lead its user message.
    pub(crate) fn is_tool_result(&self) -> bool {
        matches!(self.block, BlockParam::Known(KnownBlock::ToolResult { .. }))
    }

    /// Thinking blocks cannot carry `cache_control`.
    pub(crate) fn accepts_cache_control(&self) -> bool {
        !matches!(
            self.block,
            BlockParam::Known(KnownBlock::Thinking { .. } | KnownBlock::RedactedThinking { .. })
        )
    }
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum BlockParam {
    Known(KnownBlock),
    /// A block replayed verbatim (includes its `type`).
    Raw(ExtraMap),
}

impl From<KnownBlock> for BlockParam {
    fn from(value: KnownBlock) -> Self {
        Self::Known(value)
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum KnownBlock {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        citations: Option<Value>,
    },
    Image {
        source: MediaSourceParam,
    },
    Document {
        source: DocumentSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        /// Document fields from the file's `anthropic` provider data
        /// (`citations`, `context`).
        #[serde(flatten)]
        extra: ExtraMap,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
        /// Replayed `tool_use` fields from the call's provider data
        /// (`caller`, `toolset_name`, …).
        #[serde(flatten)]
        extra: ExtraMap,
    },
    ToolResult {
        tool_use_id: String,
        /// The matching call's `toolset_name`: a computer toolset member's
        /// result must repeat it.
        #[serde(skip_serializing_if = "Option::is_none")]
        toolset_name: Option<Value>,
        content: ToolResultContentParam,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum MediaSourceParam {
    Url {
        url: String,
    },
    Base64 {
        media_type: String,
        data: String,
    },
    /// A Files API file.
    File {
        file_id: String,
    },
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum DocumentSource {
    Url {
        url: String,
    },
    Base64 {
        media_type: String,
        data: String,
    },
    Text {
        media_type: String,
        data: String,
    },
    /// A Files API file.
    File {
        file_id: String,
    },
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum ToolResultContentParam {
    Text(String),
    Blocks(Vec<RequestBlock>),
}

#[derive(Debug, Serialize)]
pub(crate) struct ToolParam {
    /// `None` only for the computer toolset, which names its member tools
    /// itself.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// `None` for server tool definitions, which have no schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
    /// Extra tool fields from `ToolDefinition.provider_data["anthropic"]`.
    #[serde(flatten)]
    pub extra: ExtraMap,
}

#[derive(Debug, Serialize)]
pub(crate) struct ToolChoice {
    #[serde(rename = "type")]
    pub kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_parallel_tool_use: Option<bool>,
}

impl ToolChoice {
    pub(crate) fn of(kind: &'static str) -> Self {
        Self {
            kind,
            name: None,
            disable_parallel_tool_use: None,
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct ThinkingConfig {
    #[serde(rename = "type")]
    pub kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display: Option<&'static str>,
}

impl ThinkingConfig {
    pub(crate) fn of(kind: &'static str) -> Self {
        Self {
            kind,
            budget_tokens: None,
            display: None,
        }
    }

    pub(crate) fn is_on(&self) -> bool {
        matches!(self.kind, "enabled" | "adaptive")
    }

    /// Manual extended thinking (`{type: "enabled", budget_tokens}`).
    pub(crate) fn is_manual(&self) -> bool {
        self.kind == "enabled"
    }
}

#[derive(Debug, Default, Serialize)]
pub(crate) struct OutputConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<OutputFormat>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OutputFormat {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub schema: Value,
}

// --- Responses ---

#[derive(Debug, Deserialize)]
pub(crate) struct CreateMessageResponse {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub content: Vec<Value>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub stop_sequence: Option<String>,
    #[serde(default)]
    pub stop_details: Option<Value>,
    #[serde(default)]
    pub container: Option<Value>,
    #[serde(default)]
    pub usage: Usage,
    /// Fields this adapter does not model (`type`, `role`, `diagnostics`,
    /// …), kept as response provider data.
    #[serde(flatten)]
    pub extra: ExtraMap,
}

/// Raw Anthropic usage. `input_tokens` excludes cache reads and writes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub(crate) struct Usage {
    #[serde(default)]
    pub input_tokens: Option<u64>,
    #[serde(default)]
    pub output_tokens: Option<u64>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u64>,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(default)]
    pub cache_creation: Option<CacheCreation>,
    #[serde(default)]
    pub output_tokens_details: Option<OutputTokensDetails>,
    #[serde(default)]
    pub server_tool_use: Option<ExtraMap>,
    #[serde(default)]
    pub service_tier: Option<String>,
    /// Usage fields this adapter does not model (`inference_geo`, `speed`,
    /// …), kept as response provider data under `usage`.
    #[serde(flatten)]
    pub extra: ExtraMap,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub(crate) struct CacheCreation {
    #[serde(default)]
    pub ephemeral_5m_input_tokens: Option<u64>,
    #[serde(default)]
    pub ephemeral_1h_input_tokens: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub(crate) struct OutputTokensDetails {
    #[serde(default)]
    pub thinking_tokens: Option<u64>,
}

impl Usage {
    /// Overlay the fields present in a later (cumulative) usage report.
    pub(crate) fn merge(&mut self, later: Usage) {
        fn take<T>(slot: &mut Option<T>, value: Option<T>) {
            if value.is_some() {
                *slot = value;
            }
        }
        take(&mut self.input_tokens, later.input_tokens);
        take(&mut self.output_tokens, later.output_tokens);
        take(
            &mut self.cache_read_input_tokens,
            later.cache_read_input_tokens,
        );
        take(
            &mut self.cache_creation_input_tokens,
            later.cache_creation_input_tokens,
        );
        take(&mut self.cache_creation, later.cache_creation);
        take(&mut self.output_tokens_details, later.output_tokens_details);
        take(&mut self.server_tool_use, later.server_tool_use);
        take(&mut self.service_tier, later.service_tier);
        self.extra.extend(later.extra);
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct ErrorResponse {
    pub error: ApiError,
    #[serde(default)]
    pub request_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ApiError {
    #[serde(rename = "type")]
    pub error_type: String,
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub details: Option<ErrorDetails>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ErrorDetails {
    #[serde(default)]
    pub error_code: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn raw_blocks_serialize_verbatim_with_cache_control() {
        let mut map = ExtraMap::new();
        map.insert("type".into(), json!("server_tool_use"));
        map.insert("id".into(), json!("srvtoolu_1"));
        let block = RequestBlock {
            block: BlockParam::Raw(map),
            cache_control: Some(CacheControl::ephemeral(None)),
        };
        assert_eq!(
            serde_json::to_value(&block).unwrap(),
            json!({"type": "server_tool_use", "id": "srvtoolu_1", "cache_control": {"type": "ephemeral"}})
        );
    }

    #[test]
    fn known_blocks_serialize_with_type_tag() {
        let block = RequestBlock::new(KnownBlock::Thinking {
            thinking: "hmm".into(),
            signature: "sig".into(),
        });
        assert_eq!(
            serde_json::to_value(&block).unwrap(),
            json!({"type": "thinking", "thinking": "hmm", "signature": "sig"})
        );
        assert!(!block.accepts_cache_control());
    }

    #[test]
    fn usage_merge_overlays_present_fields() {
        let mut usage = Usage {
            input_tokens: Some(10),
            cache_read_input_tokens: Some(4),
            output_tokens: Some(1),
            ..Usage::default()
        };
        usage.merge(Usage {
            output_tokens: Some(30),
            ..Usage::default()
        });
        assert_eq!(usage.input_tokens, Some(10));
        assert_eq!(usage.cache_read_input_tokens, Some(4));
        assert_eq!(usage.output_tokens, Some(30));
    }
}
