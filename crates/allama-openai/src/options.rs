use allama::{ExtraMap, ProviderOptions};
use allama_openai_compat::{EmbeddingWireOptions, WireOptions};
use serde::{Deserialize, Serialize};

/// OpenAI chat options (namespace `"openai"`).
///
/// Reasoning effort is portable (`ChatRequest::reasoning`); these are the
/// OpenAI-specific request fields. The struct is `#[non_exhaustive]`: build
/// it with [`new`](Self::new) and the setters, or give the JSON form.
///
/// ```
/// use allama::ChatRequest;
/// use allama_openai::ChatRequestOptions;
///
/// let request = ChatRequest::new("gpt-5-mini").user("Hello").with_option(
///     ChatRequestOptions::new()
///         .service_tier("flex")
///         .extra("verbosity", "low"),
/// );
/// # let _ = request;
/// ```
///
/// ```compile_fail
/// # use allama_openai::ChatRequestOptions;
/// // Struct literals do not compile outside this crate.
/// let options = ChatRequestOptions { user: None, ..Default::default() };
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct ChatRequestOptions {
    /// End-user identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// A stable identifier of the end user, hashed if it is personal, for
    /// OpenAI's abuse detection (`safety_identifier`, which replaces
    /// `user`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    /// Service tier (`auto`, `default`, `flex`, `priority`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    /// Store the completion for evals and distillation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Request metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ExtraMap>,
    /// Raw `reasoning_effort` override (e.g. a level newer than this crate).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// Other Chat Completions body fields (e.g. `logprobs`, `verbosity`,
    /// `prediction`), merged into the request body. A field the adapter
    /// owns (`model`, `messages`, `stream`, `tools`, …) is refused with
    /// [`Error::InvalidRequest`](allama::Error::InvalidRequest). In the JSON
    /// form it is the nested `extra` object.
    #[serde(skip_serializing_if = "ExtraMap::is_empty")]
    pub extra: ExtraMap,
    /// Definitions of custom tools (freeform text input), in the Chat
    /// Completions shape: `{"type": "custom", "custom": {"name": …,
    /// "description": …, "format": …}}`, with a Lark or regex grammar under
    /// `format.grammar`. They are sent verbatim after the function tools.
    /// Their calls come back as [`Other`](allama::ContentBlock::Other)
    /// content with origin `"openai"`, the shape the Responses API uses;
    /// read and answer them with [`ClientToolCall`](crate::ClientToolCall).
    ///
    /// A conversation with custom tool calls moves between Responses and
    /// Chat Completions (behind a fallback, for example), but each API
    /// reads its own tool definitions: give them in both
    /// [`ResponsesRequestOptions::client_tools`](crate::ResponsesRequestOptions::client_tools)
    /// and here.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub client_tools: Vec<ExtraMap>,
    /// Keep the verbatim response body as a [`RawResponse`](crate::RawResponse)
    /// in the response's provider data. Unary calls only.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub raw_response: bool,
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

    /// Set the end-user identifier (`user`).
    #[must_use]
    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Set the service tier (`service_tier`).
    #[must_use]
    pub fn service_tier(mut self, service_tier: impl Into<String>) -> Self {
        self.service_tier = Some(service_tier.into());
        self
    }

    /// Set the end-user identifier for abuse detection
    /// (`safety_identifier`).
    #[must_use]
    pub fn safety_identifier(mut self, safety_identifier: impl Into<String>) -> Self {
        self.safety_identifier = Some(safety_identifier.into());
        self
    }

    /// Store the completion for evals and distillation (`store`).
    #[must_use]
    pub fn store(mut self, store: bool) -> Self {
        self.store = Some(store);
        self
    }

    /// Set the request `metadata`.
    #[must_use]
    pub fn metadata(mut self, metadata: ExtraMap) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Send this raw `reasoning_effort` instead of the portable effort.
    #[must_use]
    pub fn reasoning_effort(mut self, reasoning_effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(reasoning_effort.into());
        self
    }

    /// Add a custom tool definition (see [`client_tools`](Self::client_tools)).
    #[must_use]
    pub fn client_tool(mut self, definition: ExtraMap) -> Self {
        self.client_tools.push(definition);
        self
    }

    /// Keep the verbatim response body (see [`RawResponse`](crate::RawResponse)).
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

impl From<ChatRequestOptions> for WireOptions {
    fn from(options: ChatRequestOptions) -> Self {
        let mut wire = WireOptions::default();
        wire.user = options.user;
        wire.safety_identifier = options.safety_identifier;
        wire.service_tier = options.service_tier;
        wire.store = options.store;
        wire.metadata = options.metadata;
        wire.reasoning_effort = options.reasoning_effort;
        wire.extra = options.extra;
        wire.client_tools = options.client_tools;
        wire.raw_response = options.raw_response;
        wire
    }
}

/// OpenAI embedding options (namespace `"openai"`).
///
/// ```
/// use allama::EmbeddingRequest;
/// use allama_openai::EmbeddingRequestOptions;
///
/// let request = EmbeddingRequest::new("text-embedding-3-small")
///     .input("hello")
///     .with_option(EmbeddingRequestOptions::new().user("user-123"));
/// # let _ = request;
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct EmbeddingRequestOptions {
    /// End-user identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Other embeddings body fields (e.g. `encoding_format`), merged into
    /// the request body. A field the adapter owns (`model`, `input`,
    /// `dimensions`, `user`) is refused with
    /// [`Error::InvalidRequest`](allama::Error::InvalidRequest).
    #[serde(skip_serializing_if = "ExtraMap::is_empty")]
    pub extra: ExtraMap,
}

impl ProviderOptions for EmbeddingRequestOptions {
    const NAMESPACE: &'static str = crate::NAMESPACE;
}

impl EmbeddingRequestOptions {
    /// Empty options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the end-user identifier (`user`).
    #[must_use]
    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Set one body field this crate does not model (see [`extra`](Self::extra)).
    #[must_use]
    pub fn extra(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }
}

impl From<EmbeddingRequestOptions> for EmbeddingWireOptions {
    fn from(options: EmbeddingRequestOptions) -> Self {
        let mut wire = EmbeddingWireOptions::default();
        wire.user = options.user;
        wire.extra = options.extra;
        wire
    }
}
