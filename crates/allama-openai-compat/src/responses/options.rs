use allama::{ExtraMap, ProviderOptions, Warning};
use serde::{Deserialize, Serialize};

/// Responses request fields with no portable equivalent, as the encoder
/// sends them. Provider crates convert their own options type into this
/// (the Responses counterpart of [`WireOptions`](crate::WireOptions)).
#[doc(hidden)]
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct ResponsesWireOptions {
    /// Service tier.
    pub service_tier: Option<String>,
    /// Request metadata.
    pub metadata: Option<ExtraMap>,
    /// Stable, hashed end-user identifier for abuse detection.
    pub safety_identifier: Option<String>,
    /// Legacy end-user identifier.
    pub user: Option<String>,
    /// Context truncation strategy (`auto` or `disabled`).
    pub truncation: Option<String>,
    /// Output verbosity (`text.verbosity`).
    pub verbosity: Option<String>,
    /// Raw `reasoning.effort` value, sent instead of the portable effort.
    pub reasoning_effort: Option<String>,
    /// Raw `tool_choice` value, sent instead of the portable tool choice.
    pub tool_choice: Option<serde_json::Value>,
    /// Hosted tool definitions, sent after the function tools.
    pub hosted_tools: Vec<ExtraMap>,
    /// Definitions of client-executed tools that are not function tools,
    /// sent after the hosted tools.
    pub client_tools: Vec<ExtraMap>,
    /// Store the response server-side (`store`; default `false`).
    pub store: Option<bool>,
    /// Continue from a stored response (`previous_response_id`).
    pub previous_response_id: Option<String>,
    /// Continue a stored conversation (`conversation`).
    pub conversation: Option<String>,
    /// Run in the background (`background`; requires `store`).
    pub background: Option<bool>,
    /// Keep the verbatim response body in the response's provider data
    /// (unary calls only).
    pub raw_response: bool,
    /// Other body fields, merged into the request. Fields the adapter owns
    /// and stateful fields are rejected.
    pub extra: ExtraMap,
    /// Warnings the caller produced while resolving its own option types
    /// (for example options attached for another API that this request does
    /// not read). They are reported with the response, before the
    /// encoder's own warnings.
    pub warnings: Vec<Warning>,
}

/// Request options for the generic
/// [`ResponsesProvider`](crate::ResponsesProvider) (namespace
/// `"openai-compat-responses"`).
///
/// These are Responses request fields with no portable equivalent; whether
/// an endpoint honors each one is up to the endpoint. Reasoning effort,
/// tools, structured output, and prompt-cache keys are portable
/// [`ChatRequest`](allama::ChatRequest) fields.
///
/// The namespace differs from [`ChatRequestOptions`](crate::ChatRequestOptions)
/// (`"openai-compat"`) because the two APIs accept different fields, and
/// one request may carry options for both.
///
/// The struct is `#[non_exhaustive]`: build it with [`new`](Self::new) and
/// the setters, or give the JSON form.
///
/// ```
/// use allama::ChatRequest;
/// use allama_openai_compat::ResponsesRequestOptions;
/// use serde_json::json;
///
/// let typed = ChatRequest::new("m").with_option(ResponsesRequestOptions::new().truncation("auto"));
/// let json = ChatRequest::new("m")
///     .with_options_json("openai-compat-responses", json!({"truncation": "auto"}));
/// # let _ = (typed, json);
/// ```
///
/// ```compile_fail
/// # use allama_openai_compat::ResponsesRequestOptions;
/// // Struct literals do not compile outside this crate.
/// let options = ResponsesRequestOptions { user: None, ..Default::default() };
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct ResponsesRequestOptions {
    /// Service tier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    /// Request metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ExtraMap>,
    /// Stable, hashed end-user identifier for abuse detection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    /// Legacy end-user identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Context truncation strategy (`auto` or `disabled`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,
    /// Output verbosity (`text.verbosity`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
    /// Raw `reasoning.effort` value (e.g. a model-defined level), sent
    /// verbatim instead of the portable effort.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// Raw `tool_choice` value, sent verbatim instead of the portable
    /// [`ChatRequest::tool_choice`](allama::ChatRequest::tool_choice): forces
    /// a hosted or custom tool (`{"type": "web_search"}`, `{"type":
    /// "custom", "name": …}`), or restricts the choice
    /// (`{"type": "allowed_tools", …}`), which the portable choice, a
    /// function tool or a mode, cannot express. When both are set, this one
    /// is sent, with an `Approximated` warning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    /// Hosted tool definitions the endpoint offers, sent after the function
    /// tools. Their calls come back as
    /// [`ProviderToolCall`](allama::ProviderToolCall) and
    /// [`ProviderToolResult`](allama::ProviderToolResult) blocks with the
    /// endpoint's namespace as origin and are replayed as the original items.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub hosted_tools: Vec<ExtraMap>,
    /// Definitions of tools the client runs that are not function tools
    /// (`{"type": "custom", "name": …, "format": …}`, `{"type":
    /// "local_shell"}`, `{"type": "shell", "environment": {"type":
    /// "local"}}`, …), sent verbatim after the hosted tools. Their calls
    /// come back as [`Other`](allama::ContentBlock::Other) content; read and
    /// answer them with [`ClientToolCall`](crate::ClientToolCall).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub client_tools: Vec<ExtraMap>,
    /// Store the response on the endpoint (`store`). Defaults to `false`:
    /// the provider is stateless and every request carries the whole
    /// conversation. Needed by [`background`](Self::background), and on any
    /// response later continued with
    /// [`previous_response_id`](Self::previous_response_id).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Continue from a stored response (`previous_response_id`): the
    /// endpoint supplies that response's context, so the request's messages
    /// carry only the new turn. Use the earlier [`ChatResponse::id`](allama::ChatResponse::id).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    /// Continue a stored conversation (`conversation`, a Conversations API
    /// id). Mutually exclusive with `previous_response_id`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<String>,
    /// Run the response in the background (`background`; requires
    /// `store`). `chat` returns as soon as the response is queued, with no
    /// finish reason and the status in `finish_reason_raw`; poll it with
    /// [`ResponsesProvider::retrieve_response`](crate::ResponsesProvider::retrieve_response).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,
    /// Keep the verbatim response body (see
    /// [`ResponsesProvider::raw_response`](crate::ResponsesProvider::raw_response)).
    /// Unary calls only.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub raw_response: bool,
    /// Other Responses body fields, merged into the request body. Fields
    /// the adapter owns (`model`, `input`, `stream`, `tools`, …) or exposes
    /// as typed options (`store`, `previous_response_id`, `background`,
    /// `tool_choice`, …) are refused with
    /// [`Error::InvalidRequest`](allama::Error::InvalidRequest). In the JSON
    /// form it is the nested `extra` object.
    #[serde(skip_serializing_if = "ExtraMap::is_empty")]
    pub extra: ExtraMap,
}

impl ProviderOptions for ResponsesRequestOptions {
    const NAMESPACE: &'static str = "openai-compat-responses";
}

impl ResponsesRequestOptions {
    /// Empty options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the service tier (`service_tier`).
    #[must_use]
    pub fn service_tier(mut self, service_tier: impl Into<String>) -> Self {
        self.service_tier = Some(service_tier.into());
        self
    }

    /// Set the request `metadata`.
    #[must_use]
    pub fn metadata(mut self, metadata: ExtraMap) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Set the hashed end-user identifier (`safety_identifier`).
    #[must_use]
    pub fn safety_identifier(mut self, safety_identifier: impl Into<String>) -> Self {
        self.safety_identifier = Some(safety_identifier.into());
        self
    }

    /// Set the legacy end-user identifier (`user`).
    #[must_use]
    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Set the context truncation strategy (`truncation`).
    #[must_use]
    pub fn truncation(mut self, truncation: impl Into<String>) -> Self {
        self.truncation = Some(truncation.into());
        self
    }

    /// Set the output verbosity (`text.verbosity`).
    #[must_use]
    pub fn verbosity(mut self, verbosity: impl Into<String>) -> Self {
        self.verbosity = Some(verbosity.into());
        self
    }

    /// Send this raw `reasoning.effort` instead of the portable effort.
    #[must_use]
    pub fn reasoning_effort(mut self, reasoning_effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(reasoning_effort.into());
        self
    }

    /// Send this raw `tool_choice` instead of the portable tool choice (see [`tool_choice`](Self::tool_choice)).
    #[must_use]
    pub fn tool_choice(mut self, tool_choice: impl Into<serde_json::Value>) -> Self {
        self.tool_choice = Some(tool_choice.into());
        self
    }

    /// Add a hosted tool definition (see [`hosted_tools`](Self::hosted_tools)).
    #[must_use]
    pub fn hosted_tool(mut self, definition: ExtraMap) -> Self {
        self.hosted_tools.push(definition);
        self
    }

    /// Add a client-executed tool definition (see [`client_tools`](Self::client_tools)).
    #[must_use]
    pub fn client_tool(mut self, definition: ExtraMap) -> Self {
        self.client_tools.push(definition);
        self
    }

    /// Store the response on the endpoint (`store`).
    #[must_use]
    pub fn store(mut self, store: bool) -> Self {
        self.store = Some(store);
        self
    }

    /// Continue from a stored response (`previous_response_id`).
    #[must_use]
    pub fn previous_response_id(mut self, previous_response_id: impl Into<String>) -> Self {
        self.previous_response_id = Some(previous_response_id.into());
        self
    }

    /// Continue a stored conversation (`conversation`).
    #[must_use]
    pub fn conversation(mut self, conversation: impl Into<String>) -> Self {
        self.conversation = Some(conversation.into());
        self
    }

    /// Run the response in the background (`background`; requires `store`).
    #[must_use]
    pub fn background(mut self, background: bool) -> Self {
        self.background = Some(background);
        self
    }

    /// Keep the verbatim response body (see [`raw_response`](Self::raw_response)).
    #[must_use]
    pub fn raw_response(mut self, enabled: bool) -> Self {
        self.raw_response = enabled;
        self
    }

    /// Set one body field this crate does not model (see [`extra`](Self::extra)).
    #[must_use]
    pub fn extra(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }
}

impl From<ResponsesRequestOptions> for ResponsesWireOptions {
    fn from(options: ResponsesRequestOptions) -> Self {
        Self {
            service_tier: options.service_tier,
            metadata: options.metadata,
            safety_identifier: options.safety_identifier,
            user: options.user,
            truncation: options.truncation,
            verbosity: options.verbosity,
            reasoning_effort: options.reasoning_effort,
            tool_choice: options.tool_choice,
            hosted_tools: options.hosted_tools,
            client_tools: options.client_tools,
            store: options.store,
            previous_response_id: options.previous_response_id,
            conversation: options.conversation,
            background: options.background,
            raw_response: options.raw_response,
            extra: options.extra,
            warnings: Vec::new(),
        }
    }
}
