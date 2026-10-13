use allama::{ExtraMap, ProviderOptions};
use serde::{Deserialize, Serialize};

/// Anthropic-specific request options.
///
/// Attach the typed value with [`ChatRequest::with_option`](allama::ChatRequest::with_option),
/// or send the same fields as JSON under the `"anthropic"` namespace with
/// [`ChatRequest::with_options_json`](allama::ChatRequest::with_options_json).
/// Unknown JSON fields are rejected.
///
/// The struct is `#[non_exhaustive]`: build it with [`new`](Self::new) and
/// the setters (fields stay public to read and to assign). Request fields
/// this crate does not model (`container`, `mcp_servers`,
/// `context_management`, …) go in [`extra`](Self::extra).
///
/// ```
/// use allama::ChatRequest;
/// use allama_anthropic::ChatRequestOptions;
///
/// let request = ChatRequest::new("claude-haiku-4-5")
///     .user("Think about it")
///     .top_k(20)
///     .with_option(ChatRequestOptions::new().budget_tokens(2048));
/// # let _ = request;
/// ```
///
/// ```compile_fail
/// # use allama_anthropic::ChatRequestOptions;
/// // Struct literals do not compile outside this crate: the struct grows
/// // with the API.
/// let options = ChatRequestOptions { budget_tokens: Some(2048), ..Default::default() };
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
#[non_exhaustive]
pub struct ChatRequestOptions {
    /// Request `metadata` object (for example `{"user_id": "..."}`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ExtraMap>,
    /// Beta feature names sent in the `anthropic-beta` header.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub anthropic_beta: Vec<String>,
    /// Fixed extended-thinking budget in tokens.
    ///
    /// Sends `thinking: {type: "enabled", budget_tokens}` and takes
    /// precedence over `ChatRequest::reasoning`'s effort. Only models that
    /// still accept fixed budgets (Claude Haiku 4.5, Sonnet/Opus 4.5 and
    /// older, and the deprecated path on the 4.6 generation) take it; it is
    /// sent to any model, and a model that takes only adaptive thinking
    /// refuses it with [`allama::Error::Unsupported`] (stage `Provider`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
    /// The `output_config.effort` level, sent as it is.
    ///
    /// It replaces the level the adapter maps `ChatRequest::reasoning`'s
    /// effort to, so it reaches a level the portable scale cannot name on
    /// that model (`max` where the model also has `xhigh`). It changes
    /// nothing else: thinking stays as the portable reasoning config sets
    /// it. A model without the level refuses it with
    /// [`allama::Error::Unsupported`] (stage `Provider`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<crate::EffortLevel>,
    /// Request `service_tier` (`"auto"` or `"standard_only"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    /// Server tool definitions (`{"type": "web_search_20250305", "name":
    /// "web_search"}`, code execution, web fetch, …), sent verbatim after the
    /// function tools. Anthropic runs them itself: their calls and results
    /// come back as [`allama::ProviderToolCall`] and
    /// [`allama::ProviderToolResult`] blocks. Anthropic may pause a long
    /// server tool turn (`finish_reason_raw` `pause_turn`,
    /// [`FinishReason::Other`](allama::FinishReason::Other)): append
    /// `response.to_assistant_message()` and call again unchanged to let
    /// it finish.
    ///
    /// Anthropic-defined tools that the client runs (`{"type":
    /// "bash_20250124", "name": "bash"}`, the text editor, the computer
    /// toolset) go here too. Their calls are ordinary `tool_use` blocks, so
    /// they come back as [`allama::ToolCall`]s and are answered with tool
    /// messages (ADR 0014).
    ///
    /// Every definition needs a `name` except the computer toolset
    /// (`{"type": "computer_toolset_20260801"}`), whose member tools
    /// (`screenshot`, `left_click`, …) are named by the toolset. A member
    /// call keeps its `toolset_name` in provider data, and the adapter
    /// repeats it on the `tool_result` that answers the call.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub server_tools: Vec<ExtraMap>,
    /// Keep the verbatim response body as a [`RawResponse`](crate::RawResponse)
    /// in the response's provider data. Unary calls only.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub raw_response: bool,
    /// Other Messages API body fields, merged into the top level of the
    /// request body (`{"container": "container_…"}`, `{"mcp_servers": […]}`,
    /// …). A field the adapter owns (`model`, `messages`, `tools`,
    /// `thinking`, `metadata`, …) is refused with
    /// [`Error::InvalidRequest`](allama::Error::InvalidRequest): set it
    /// through the portable request or a typed option. In the JSON form it is
    /// the nested `extra` object.
    #[serde(skip_serializing_if = "ExtraMap::is_empty")]
    pub extra: ExtraMap,
}

impl ProviderOptions for ChatRequestOptions {
    const NAMESPACE: &'static str = crate::NAMESPACE;
}

impl ChatRequestOptions {
    /// Empty options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the request `metadata` object.
    #[must_use]
    pub fn metadata(mut self, metadata: ExtraMap) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Add an `anthropic-beta` feature name.
    #[must_use]
    pub fn beta(mut self, name: impl Into<String>) -> Self {
        self.anthropic_beta.push(name.into());
        self
    }

    /// Set a fixed extended-thinking budget.
    #[must_use]
    pub fn budget_tokens(mut self, budget: u32) -> Self {
        self.budget_tokens = Some(budget);
        self
    }

    /// Set the `output_config.effort` level (see [`effort`](Self::effort)).
    #[must_use]
    pub fn effort(mut self, level: crate::EffortLevel) -> Self {
        self.effort = Some(level);
        self
    }

    /// Set `service_tier`.
    #[must_use]
    pub fn service_tier(mut self, tier: impl Into<String>) -> Self {
        self.service_tier = Some(tier.into());
        self
    }

    /// Add a server tool or Anthropic-defined client tool definition (see
    /// [`server_tools`](Self::server_tools)).
    #[must_use]
    pub fn server_tool(mut self, definition: ExtraMap) -> Self {
        self.server_tools.push(definition);
        self
    }

    /// Keep the verbatim response body (see [`RawResponse`](crate::RawResponse)).
    #[must_use]
    pub fn raw_response(mut self, enabled: bool) -> Self {
        self.raw_response = enabled;
        self
    }

    /// Set one body field this crate does not model (see
    /// [`extra`](Self::extra)).
    #[must_use]
    pub fn extra(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use allama::ChatRequest;
    use serde_json::json;

    #[test]
    fn typed_and_json_options_resolve_identically() {
        let typed = ChatRequest::new("m").with_option(
            ChatRequestOptions::new()
                .service_tier("auto")
                .beta("b1")
                .budget_tokens(1024),
        );
        let json = ChatRequest::new("m").with_options_json(
            "anthropic",
            json!({"service_tier": "auto", "anthropic_beta": ["b1"], "budget_tokens": 1024}),
        );
        assert_eq!(
            typed.resolve_option::<ChatRequestOptions>().unwrap(),
            json.resolve_option::<ChatRequestOptions>().unwrap()
        );
    }

    #[test]
    fn unknown_json_fields_are_rejected() {
        // `top_k` is portable (`ChatRequest::top_k`), not an option.
        for field in ["top_kk", "top_k"] {
            let request = ChatRequest::new("m").with_options_json("anthropic", json!({field: 5}));
            let err = request.resolve_option::<ChatRequestOptions>().unwrap_err();
            assert!(
                matches!(err, allama::Error::InvalidRequest { .. }),
                "{field}: {err:?}"
            );
        }
    }

    #[test]
    fn options_survive_export_and_import() {
        let request = ChatRequest::new("m").with_option(ChatRequestOptions::new().budget_tokens(3));
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["options"]["anthropic"], json!({"budget_tokens": 3}));
        let back: ChatRequest = serde_json::from_value(value).unwrap();
        assert_eq!(
            back.resolve_option::<ChatRequestOptions>()
                .unwrap()
                .budget_tokens,
            Some(3)
        );
    }
}
