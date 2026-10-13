use allama::adapter::Degradations;
use allama::{
    Error, ExtraMap, ProviderOptions, RequestOptions, Result, Subject, Warning, WarningKind,
};
use serde::{Deserialize, Serialize};

/// Request fields shared by OpenAI-compatible dialects that have no portable
/// equivalent. Provider crates convert their own options type into this.
///
/// Start from [`WireOptions::default`] and set the fields you need:
///
/// ```
/// use allama_openai_compat::WireOptions;
///
/// let mut options = WireOptions::default();
/// options.service_tier = Some("flex".into());
/// options.extra.insert("safety_mode".into(), "strict".into());
/// # let _ = options;
/// ```
#[doc(hidden)]
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct WireOptions {
    /// End-user identifier.
    pub user: Option<String>,
    /// Stable end-user identifier for abuse detection (`safety_identifier`).
    pub safety_identifier: Option<String>,
    /// Service tier (e.g. `flex`, `priority`).
    pub service_tier: Option<String>,
    /// Store the completion upstream.
    pub store: Option<bool>,
    /// Request metadata.
    pub metadata: Option<ExtraMap>,
    /// Raw `reasoning_effort` value, overriding the portable effort.
    pub reasoning_effort: Option<String>,
    /// Endpoint-specific body fields, merged into the request. Typed by the
    /// owning provider crate; never derived from portable fields.
    pub extra: ExtraMap,
    /// Definitions of client-executed tools that are not function tools
    /// (`{"type": "custom", "custom": {…}}`), sent verbatim after the
    /// function tools.
    pub client_tools: Vec<ExtraMap>,
    /// Keep the verbatim response body in the response's provider data,
    /// under the provider namespace and
    /// [`RAW_RESPONSE_KEY`](allama::provider_data::RAW_RESPONSE_KEY). Unary
    /// calls only.
    pub raw_response: bool,
}

/// Options for the generic OpenAI-compatible [`Provider`](crate::Provider)
/// (namespace `"openai-compat"`).
///
/// The struct is `#[non_exhaustive]`: build it with [`new`](Self::new) and
/// the setters, or give the JSON form.
///
/// ```
/// use allama::ChatRequest;
/// use allama_openai_compat::ChatRequestOptions;
/// use serde_json::json;
///
/// // Typed …
/// let typed = ChatRequest::new("m").with_option(
///     ChatRequestOptions::new()
///         .service_tier("flex")
///         .extra("safe_prompt", true),
/// );
/// // … or as JSON.
/// let json = ChatRequest::new("m").with_options_json(
///     "openai-compat",
///     json!({"service_tier": "flex", "extra": {"safe_prompt": true}}),
/// );
/// assert_eq!(
///     typed.resolve_option::<ChatRequestOptions>().unwrap(),
///     json.resolve_option::<ChatRequestOptions>().unwrap(),
/// );
/// ```
///
/// ```compile_fail
/// # use allama_openai_compat::ChatRequestOptions;
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
    /// Stable identifier of the end user, hashed if it is personal, for
    /// abuse detection (`safety_identifier`; OpenAI's successor to `user`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    /// Service tier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    /// Store the completion upstream.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Request metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ExtraMap>,
    /// Raw `reasoning_effort` value, overriding the portable effort.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// Endpoint-specific body fields (for parameters this crate does not
    /// model, e.g. vendor sampling knobs), merged into the top level of the
    /// request body. A field the adapter owns (`model`, `messages`,
    /// `stream`, `tools`, `seed`, …) is refused with
    /// [`Error::InvalidRequest`]: set it through the portable request or a
    /// typed option. In the JSON form it is the nested `extra` object.
    #[serde(skip_serializing_if = "ExtraMap::is_empty")]
    pub extra: ExtraMap,
    /// Definitions of custom tools in the Chat Completions shape
    /// (`{"type": "custom", "custom": {"name": …, "description": …,
    /// "format": …}}`, with a grammar under `format.grammar`), sent verbatim
    /// after the function tools, for endpoints that offer them. Their calls
    /// come back as [`Other`](allama::ContentBlock::Other) content; read and
    /// answer them with [`ClientToolCall`](crate::ClientToolCall).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub client_tools: Vec<ExtraMap>,
    /// Keep the verbatim response body (see
    /// [`Provider::raw_response`](crate::Provider::raw_response)). Unary calls
    /// only.
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

    /// Set the end-user identifier for abuse detection
    /// (`safety_identifier`).
    #[must_use]
    pub fn safety_identifier(mut self, safety_identifier: impl Into<String>) -> Self {
        self.safety_identifier = Some(safety_identifier.into());
        self
    }

    /// Set the service tier (`service_tier`).
    #[must_use]
    pub fn service_tier(mut self, service_tier: impl Into<String>) -> Self {
        self.service_tier = Some(service_tier.into());
        self
    }

    /// Store the completion upstream (`store`).
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

impl From<ChatRequestOptions> for WireOptions {
    fn from(options: ChatRequestOptions) -> Self {
        Self {
            user: options.user,
            safety_identifier: options.safety_identifier,
            service_tier: options.service_tier,
            store: options.store,
            metadata: options.metadata,
            reasoning_effort: options.reasoning_effort,
            extra: options.extra,
            client_tools: options.client_tools,
            raw_response: options.raw_response,
        }
    }
}

/// Role used for a system prompt (OpenAI dialect).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SystemRole {
    /// `system` (default).
    #[default]
    System,
    /// `developer`: instructions from the application developer, which
    /// OpenAI reasoning models prioritize over `system`.
    Developer,
}

/// Per-system-prompt options (namespace `"openai"`).
///
/// Attach with `SystemPrompt::new("...").with_option(SystemOptions::developer())`.
/// Endpoints without a `developer` role fall back to `system` with a warning.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct SystemOptions {
    /// Role to send the prompt with.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<SystemRole>,
}

impl SystemOptions {
    /// Empty options: the prompt goes out with the `system` role.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Send the prompt with the `developer` role.
    #[must_use]
    pub fn developer() -> Self {
        Self::new().role(SystemRole::Developer)
    }

    /// Set the role to send the prompt with.
    #[must_use]
    pub fn role(mut self, role: SystemRole) -> Self {
        self.role = Some(role);
        self
    }
}

impl ProviderOptions for SystemOptions {
    const NAMESPACE: &'static str = "openai";
}

/// Embedding options for the generic provider (namespace `"openai-compat"`).
///
/// ```
/// use allama::EmbeddingRequest;
/// use allama_openai_compat::EmbeddingRequestOptions;
///
/// let request = EmbeddingRequest::new("m")
///     .input("hello")
///     .with_option(EmbeddingRequestOptions::new().extra("truncate", "END"));
/// # let _ = request;
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct EmbeddingRequestOptions {
    /// End-user identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Endpoint-specific body fields, merged into the request body. A field
    /// the adapter owns (`model`, `input`, `dimensions`, `user`) is refused
    /// with [`Error::InvalidRequest`]; the endpoint's input-type field
    /// ([`CompatQuirks::embedding_input_type`](crate::CompatQuirks::embedding_input_type))
    /// may be set here, overriding the portable input type with a warning.
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

/// Refuse an `extra` map that sets a field the adapter owns: merged into
/// the body, it would send the field twice. The comparison ignores ASCII
/// case, since some servers (Go's `encoding/json`) read keys that way.
/// `owner` names the map in the error.
pub(crate) fn refuse_owned_fields(
    extra: &ExtraMap,
    owned: &[&str],
    owner: &str,
    provider: &str,
) -> Result<()> {
    match extra
        .keys()
        .find(|key| owned.iter().any(|field| key.eq_ignore_ascii_case(field)))
    {
        Some(key) => Err(Error::invalid_request(format!(
            "`{key}` cannot be set through {owner}: the adapter owns it; use the portable request field or the typed option"
        ))
        .authored()
        .with_provider(provider)),
        None => Ok(()),
    }
}

/// The code of the warning [`report_options_under_name`] records.
pub(crate) const PROVIDER_NAME_OPTIONS: &str = "dropped.provider_name_options";

/// Report JSON options under `provider`, the provider's name, which it does
/// not read: a `Dropped` warning coded [`PROVIDER_NAME_OPTIONS`]. A
/// configuration file keys options by the name used everywhere else (the
/// model address, the provider name, the credential scope), and without
/// the warning they would be ignored silently. `namespace` is the options
/// namespace the provider reads, and `typed` the options type to name in
/// the remedy. Nothing is reported when the name is a namespace the provider
/// reads or already reports (`reported`).
pub(crate) fn report_options_under_name(
    options: &RequestOptions,
    provider: &str,
    namespace: &str,
    reported: &[&str],
    typed: &str,
    d: &mut Degradations,
) {
    if provider == namespace || reported.contains(&provider) || options.json(provider).is_none() {
        return;
    }
    d.warn(
        Warning::new(
            WarningKind::Dropped,
            Subject::Option {
                namespace: provider.to_owned(),
            },
            format!(
                "this provider reads options under `{namespace}`, not under its name `{provider}`; they were dropped; pass them under `{namespace}`, or as `{typed}`"
            ),
        )
        .with_code(PROVIDER_NAME_OPTIONS),
    );
}
