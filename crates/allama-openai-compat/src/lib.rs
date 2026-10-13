#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
//! OpenAI-compatible provider and adapter toolkit for `allama`.
//!
//! - [`Provider`]: a chat and embedding provider for any endpoint that speaks
//!   OpenAI Chat Completions, configured with a [`CompatQuirks`] dialect.
//! - [`ResponsesProvider`]: a stateless chat provider for any endpoint that
//!   speaks the OpenAI Responses API, configured with a [`ResponsesQuirks`]
//!   dialect and optional per-model [`ResponsesModelRules`].
//! - [`providers`]: presets for common endpoints (Groq, OpenRouter, Ollama,
//!   DeepSeek, Mistral, …).
//! - [`ClientToolCall`]: typed views of client-executed tool calls that are
//!   not function calls (custom tools on both APIs; shell, computer use and
//!   `apply_patch` on Responses), and the answers to them (ADR 0014).
//! - [`schema`]: the strict JSON Schema dialect of OpenAI-grammar endpoints
//!   ([`schema::strict_schema`]), reusable by adapters with their own
//!   envelope.
//!
//! The Chat Completions and Embeddings wire types are internal: they follow
//! the vendors and change with them. Build on [`Provider`] and
//! [`ResponsesProvider`], and send fields this crate does not model through
//! [`ChatRequestOptions::extra`] and [`ResponsesRequestOptions::extra`].
//!
//! ```no_run
//! use allama::prelude::*;
//! use allama_openai_compat::providers::Groq;
//!
//! # async fn example() -> allama::Result<()> {
//! let provider = Groq::from_env()?;
//! let text = provider.ask("llama-3.3-70b-versatile", "Say hello").await?;
//! # let _ = text; Ok(()) }
//! ```
//!
//! # Transport
//!
//! Requests go through `reqwest` (the default `reqwest` feature) unless the
//! builder is given another [`HttpTransport`]
//! with `http_transport`. With `default-features = false` and an injected
//! transport the crate builds without `reqwest`; the constructors that would
//! build a `reqwest` client (the presets' `new` and `from_env`) are then
//! not compiled, and `build()` without a transport is a configuration
//! error. TLS backends: `default-tls`, `native-tls`, `rustls`,
//! and `rustls-ring` (`rustls` with `ring` instead of `aws-lc-rs`, so musl
//! builds need no cmake).

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

pub use allama::adapter::chat::ToolCallIdFormat;
use allama::{
    CapabilitySupport, ChatCapability, ChatCapabilityResolver, EmbeddingCapability, Error, Result,
};
#[cfg(feature = "reqwest")]
use allama_http::ReqwestTransport;
pub use allama_http::StreamTimeouts;
use allama_http::config::redact_url;
use allama_http::sse::SseDecoder;
use allama_http::{
    CredentialPlacement, CredentialSlot, DEFAULT_REQUEST_TIMEOUT, Headers, HttpTransport,
};

#[doc(hidden)]
pub mod binding;
mod chat;
mod citations;
mod client_tools;
mod embedding;
mod error;
mod model_rules;
mod options;
pub mod providers;
mod quirks;
mod responses;
pub mod schema;
mod streaming;
mod wire;

/// The key, in an endpoint's own namespace of a user or assistant
/// message's provider data, of the Chat Completions per-message participant
/// `name`.
///
/// Participant names are not portable (only Chat Completions has them), so
/// they are provider data addressed to the endpoint, the way the developer
/// role is an option of the system prompt. Another provider
/// never reads them; the Responses API, which shares the `openai`
/// namespace, reports them as a hint it did not send.
///
/// ```
/// use allama::Message;
/// use allama_openai_compat::PARTICIPANT_NAME_KEY;
///
/// let mut message = Message::user("Hi, I'm Ada.");
/// message
///     .provider_data_mut()
///     .insert_raw("openai", PARTICIPANT_NAME_KEY, "ada".into());
/// ```
pub const PARTICIPANT_NAME_KEY: &str = "name";

pub use client_tools::{
    ClientToolCall, CustomToolCall, LocalShellCall, McpApprovalRequest, ShellCall,
    ShellCommandOutput, ShellOutcome,
};
pub use embedding::EmbeddingWireOptions;
pub(crate) use error::{map_api_error, map_http_error};
pub use model_rules::{ChatModelRules, ToolCallSupport};
pub use options::{
    ChatRequestOptions, EmbeddingRequestOptions, SystemOptions, SystemRole, WireOptions,
};
pub use quirks::{
    CompatQuirks, EmbeddingInputTypeField, MaxTokensField, ReasoningReplay, ReasoningReplayField,
    ResponsesQuirks, SchemaConstraint, SeedField, StreamingSupport,
};
pub use responses::{
    ResponsesModelRules, ResponsesProvider, ResponsesProviderBuilder, ResponsesRequestOptions,
    ResponsesWireOptions, SamplingSupport, StreamPosition,
};
pub(crate) use streaming::SseState;
pub(crate) use wire::{ApiError, ResponseContext, from_api_response};

/// Namespace of this crate's request options ([`ChatRequestOptions`],
/// [`EmbeddingRequestOptions`]) and the default provider-data namespace.
///
/// A provider with a custom [`provider_name`](ProviderBuilder::provider_name)
/// stores its state under `openai-compat:<provider name>` instead (see
/// [`CompatQuirks::namespace`]), so state from one endpoint is not replayed
/// to another; options stay under this namespace for every endpoint.
pub const NAMESPACE: &str = "openai-compat";

/// The default [`provider_name`](allama::ProviderIdentity::provider_name) of
/// [`Provider`] and [`ResponsesProvider`]. Presets in [`providers`] report
/// their own.
pub const PROVIDER_NAME: &str = "openai-compat";

/// The default credential scope of [`Provider`] and [`ResponsesProvider`]:
/// per-request credentials addressed to it
/// (`allama_http::HttpRequestOptions::credential`) are this provider's.
///
/// A provider with a custom
/// [`provider_name`](ProviderBuilder::provider_name) defaults to that name
/// instead, so each preset in [`providers`] has its own scope (its
/// `DEFAULT_CREDENTIAL_SCOPE`). The scope names an account, not a wire
/// format: it is independent of the provider name used for telemetry.
/// Change it with [`ProviderBuilder::credential_scope`].
pub const DEFAULT_CREDENTIAL_SCOPE: &str = PROVIDER_NAME;

/// An OpenAI-compatible chat and embedding provider.
#[derive(Clone)]
pub struct Provider {
    pub(crate) inner: Arc<Inner>,
}

#[derive(Clone)]
pub(crate) struct Inner {
    pub(crate) http: Arc<dyn HttpTransport>,
    pub(crate) transport: TransportConfig,
    pub(crate) quirks: CompatQuirks,
    pub(crate) chat_capabilities: HashMap<ChatCapability, CapabilitySupport>,
    pub(crate) chat_capability_resolver: Option<Arc<dyn ChatCapabilityResolver>>,
    pub(crate) model_rules: Option<ChatModelRulesFn>,
    pub(crate) embedding_capabilities: HashMap<EmbeddingCapability, CapabilitySupport>,
    pub(crate) max_embedding_inputs: Option<usize>,
    pub(crate) provider_name: Cow<'static, str>,
    pub(crate) credential_slot: CredentialSlot,
}

impl std::fmt::Debug for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Provider")
            .field("provider_name", &self.inner.provider_name)
            .field("base_url", &redact_url(&self.inner.transport.base_url))
            .finish_non_exhaustive()
    }
}

impl Provider {
    /// Start configuring a provider.
    #[must_use]
    pub fn builder() -> ProviderBuilder {
        ProviderBuilder::default()
    }

    /// Override capability answers with a resolver.
    #[must_use]
    pub fn with_chat_capabilities(self, resolver: impl ChatCapabilityResolver) -> Self {
        let mut inner = (*self.inner).clone();
        inner.chat_capability_resolver = Some(Arc::new(resolver));
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Replace the request rules per model id (see
    /// [`ProviderBuilder::model_rules`]). An adapter built on this provider
    /// uses it to make its capability overrides reach the rules.
    #[must_use]
    pub fn with_model_rules<F>(self, rules: F) -> Self
    where
        F: Fn(&str) -> ChatModelRules + Send + Sync + 'static,
    {
        let mut inner = (*self.inner).clone();
        inner.model_rules = Some(Arc::new(rules));
        Self {
            inner: Arc::new(inner),
        }
    }

    /// The request rules applied to `model`.
    #[must_use]
    pub fn model_rules(&self, model: &str) -> ChatModelRules {
        self.inner
            .model_rules
            .as_ref()
            .map_or_else(ChatModelRules::new, |rules| rules(model))
    }

    /// Use `scope` as the credential scope (see
    /// [`ProviderBuilder::credential_scope`]). A provider without configured
    /// auth then requires a credential for `scope` on every request.
    #[must_use]
    pub fn with_credential_scope(self, scope: impl Into<Cow<'static, str>>) -> Self {
        let mut inner = (*self.inner).clone();
        inner.credential_slot = rescope(
            &inner.credential_slot,
            scope.into(),
            inner.transport.auth.as_ref(),
        );
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Where this provider takes its per-request credential from, and where
    /// it puts it (`allama_http::CredentialSlot`).
    #[must_use]
    pub fn credential_slot(&self) -> &CredentialSlot {
        &self.inner.credential_slot
    }

    /// The dialect this provider speaks.
    #[must_use]
    pub fn quirks(&self) -> &CompatQuirks {
        &self.inner.quirks
    }

    /// The base URL requests go to, as configured (without a trailing
    /// slash). It may carry a user name and password: show it with
    /// [`allama_http::config::redact_url`].
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.inner.transport.base_url
    }

    /// The transport configuration.
    #[cfg(test)]
    pub(crate) fn transport(&self) -> &TransportConfig {
        &self.inner.transport
    }
}

/// Builder for [`Provider`].
///
/// `Debug` shows the settings without the auth value, header values, or
/// credentials in the base URL.
#[derive(Default)]
pub struct ProviderBuilder {
    base_url: Option<String>,
    chat_completions_path: Option<String>,
    embeddings_path: Option<String>,
    auth: Option<(String, String)>,
    organization_header: Option<(String, String)>,
    project_header: Option<(String, String)>,
    request_id_header_name: Option<String>,
    retry_after_header_name: Option<String>,
    chat_capabilities: HashMap<ChatCapability, CapabilitySupport>,
    embedding_capabilities: HashMap<EmbeddingCapability, CapabilitySupport>,
    max_embedding_inputs: Option<usize>,
    provider_name: Option<Cow<'static, str>>,
    credential_scope: Option<Cow<'static, str>>,
    credential_placement: Option<CredentialPlacement>,
    quirks: Option<CompatQuirks>,
    timeout: Option<Option<Duration>>,
    stream_timeouts: Option<StreamTimeouts>,
    max_stream_event_size: Option<usize>,
    max_response_size: Option<usize>,
    headers: Vec<(String, String)>,
    http: Option<Arc<dyn HttpTransport>>,
    model_rules: Option<ChatModelRulesFn>,
}

type ChatModelRulesFn = Arc<dyn Fn(&str) -> ChatModelRules + Send + Sync>;

impl std::fmt::Debug for ProviderBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderBuilder")
            .field("provider_name", &self.provider_name)
            .field("base_url", &self.base_url.as_deref().map(redact_url))
            .field(
                "auth",
                &self.auth.as_ref().map(|(name, _)| (name, Redacted)),
            )
            .field("credential_scope", &self.credential_scope)
            .field(
                "headers",
                &self
                    .organization_header
                    .iter()
                    .chain(&self.project_header)
                    .chain(&self.headers)
                    .map(|(name, _)| name)
                    .collect::<Vec<_>>(),
            )
            .field("http_transport", &self.http.is_some())
            .field("timeout", &self.timeout)
            .field("stream_timeouts", &self.stream_timeouts)
            .field("max_stream_event_size", &self.max_stream_event_size)
            .field("max_response_size", &self.max_response_size)
            .finish_non_exhaustive()
    }
}

impl ProviderBuilder {
    /// Request rules per model id ([`ChatModelRules`]). Without them every
    /// model gets [`ChatModelRules::new`], which sends sampling settings
    /// and tools as requested.
    #[must_use]
    pub fn model_rules<F>(mut self, rules: F) -> Self
    where
        F: Fn(&str) -> ChatModelRules + Send + Sync + 'static,
    {
        self.model_rules = Some(Arc::new(rules));
        self
    }

    /// Base URL, e.g. `https://api.example.com/v1`.
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Chat completions path (default `/chat/completions`).
    #[must_use]
    pub fn chat_completions_path(mut self, path: impl Into<String>) -> Self {
        self.chat_completions_path = Some(path.into());
        self
    }

    /// Embeddings path (default `/embeddings`).
    #[must_use]
    pub fn embeddings_path(mut self, path: impl Into<String>) -> Self {
        self.embeddings_path = Some(path.into());
        self
    }

    /// Authenticate with `Authorization: Bearer <token>`. Surrounding
    /// whitespace is removed from the token.
    #[must_use]
    pub fn bearer_token(self, token: impl Into<String>) -> Self {
        self.auth_header("authorization", format!("Bearer {}", token.into().trim()))
    }

    /// Authenticate with a custom header (e.g. `api-key`). Surrounding
    /// whitespace is removed from the value.
    ///
    /// A per-request `allama_http::Credential::api_key` then goes in this
    /// header too, unless [`credential_placement`](Self::credential_placement)
    /// says otherwise. An `authorization` header is taken as a bearer token.
    #[must_use]
    pub fn auth_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.auth = Some((name.into(), value.into()));
        self
    }

    /// Send the header `name: value` with every request, replacing a
    /// header of the same name the provider sets (later calls replace
    /// earlier ones). For headers that belong to this endpoint: a gateway's
    /// routing header, a tenant tag. `header("user-agent", ..)` replaces the
    /// default [`allama_http::DEFAULT_USER_AGENT`]. A request's own
    /// `allama_http::HttpRequestOptions` headers still replace these.
    ///
    /// Credential headers are refused at [`build`](Self::build): configure
    /// the key with [`bearer_token`](Self::bearer_token) or
    /// [`auth_header`](Self::auth_header), or send one per request as an
    /// `allama_http::Credential`.
    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// The credential scope (default: the
    /// [`provider_name`](Self::provider_name), else
    /// [`DEFAULT_CREDENTIAL_SCOPE`]). Per-request credentials addressed to
    /// it (`allama_http::HttpRequestOptions::credential`) are this
    /// provider's; credentials addressed to other scopes are never sent.
    ///
    /// Two accounts of one endpoint in a fallback chain need two scopes: by
    /// default they share one. Providers behind one gateway share one.
    ///
    /// Without configured auth, setting a scope means bring your own key:
    /// every request must carry a credential for the scope, or it fails
    /// with an authored [`Error::Configuration`] before anything is sent.
    /// With configured auth, a request without a credential for the scope
    /// is sent with the configured one, so a mistyped scope falls back to
    /// it.
    #[must_use]
    pub fn credential_scope(mut self, scope: impl Into<Cow<'static, str>>) -> Self {
        self.credential_scope = Some(scope.into());
        self
    }

    /// Where a per-request `allama_http::Credential::api_key` goes. By
    /// default, where the configured auth goes: a bearer token for
    /// [`bearer_token`](Self::bearer_token) (or no auth), the header for
    /// [`auth_header`](Self::auth_header). When it differs from the
    /// configured auth header, a per-request key replaces that header too.
    #[must_use]
    pub fn credential_placement(mut self, placement: CredentialPlacement) -> Self {
        self.credential_placement = Some(placement);
        self
    }

    /// Send an organization header.
    #[must_use]
    pub fn organization_header(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.organization_header = Some((name.into(), value.into()));
        self
    }

    /// Send a project header.
    #[must_use]
    pub fn project_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.project_header = Some((name.into(), value.into()));
        self
    }

    /// Header carrying the request id (default `x-request-id`).
    #[must_use]
    pub fn request_id_header_name(mut self, name: impl Into<String>) -> Self {
        self.request_id_header_name = Some(name.into());
        self
    }

    /// Header carrying the retry-after hint (default `retry-after`;
    /// `retry-after-ms` is always consulted too).
    #[must_use]
    pub fn retry_after_header_name(mut self, name: impl Into<String>) -> Self {
        self.retry_after_header_name = Some(name.into());
        self
    }

    /// Declare a chat capability.
    #[must_use]
    pub fn chat_capability(
        mut self,
        capability: ChatCapability,
        support: CapabilitySupport,
    ) -> Self {
        self.chat_capabilities.insert(capability, support);
        self
    }

    /// Declare an embedding capability.
    #[must_use]
    pub fn embedding_capability(
        mut self,
        capability: EmbeddingCapability,
        support: CapabilitySupport,
    ) -> Self {
        self.embedding_capabilities.insert(capability, support);
        self
    }

    /// Maximum embedding inputs per request.
    #[must_use]
    pub fn max_embedding_inputs(mut self, max: usize) -> Self {
        self.max_embedding_inputs = Some(max);
        self
    }

    /// Provider identity used for telemetry and `ChatResponse::provider`.
    #[must_use]
    pub fn provider_name(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        self.provider_name = Some(name.into());
        self
    }

    /// Dialect quirks. Defaults to [`CompatQuirks::new`] under namespace
    /// `openai-compat:<provider_name>`.
    #[must_use]
    pub fn quirks(mut self, quirks: CompatQuirks) -> Self {
        self.quirks = Some(quirks);
        self
    }

    /// The timeout for a whole non-streaming call; `None` sets none.
    ///
    /// Left unset, it is [`DEFAULT_REQUEST_TIMEOUT`] (120 s), whatever the
    /// transport: a client of your own keeps the bound unless you pass
    /// `None` here. Set, it applies in place of a client's own total
    /// timeout.
    ///
    /// Streams have no total timeout:
    /// [`stream_timeouts`](Self::stream_timeouts) bounds them.
    #[must_use]
    pub fn timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Bounds on waiting for a stream's headers and between its chunks
    /// (default [`StreamTimeouts::default`];
    /// [`StreamTimeouts::unbounded`] disables them). They apply whatever the
    /// transport.
    #[must_use]
    pub fn stream_timeouts(mut self, timeouts: StreamTimeouts) -> Self {
        self.stream_timeouts = Some(timeouts);
        self
    }

    /// The largest stream event, in bytes, the provider holds in memory
    /// (default [`SseDecoder::DEFAULT_MAX_EVENT_SIZE`], 64 MiB). A larger
    /// event fails the stream with a stream protocol error, which is not
    /// retryable. See [`SseDecoder::max_event_size`].
    ///
    /// [`SseDecoder::DEFAULT_MAX_EVENT_SIZE`]: allama_http::sse::SseDecoder::DEFAULT_MAX_EVENT_SIZE
    /// [`SseDecoder::max_event_size`]: allama_http::sse::SseDecoder::max_event_size
    #[must_use]
    pub fn max_stream_event_size(mut self, bytes: usize) -> Self {
        self.max_stream_event_size = Some(bytes);
        self
    }

    /// The largest non-streaming response body, in bytes, the provider
    /// reads (default [`allama_http::DEFAULT_MAX_RESPONSE_SIZE`], 256 MiB).
    /// A larger body fails the call with [`Error::UnexpectedResponse`],
    /// which is not retryable, without being read whole. See
    /// [`allama_http::HttpResponse::bytes_limited`].
    #[must_use]
    pub fn max_response_size(mut self, bytes: usize) -> Self {
        self.max_response_size = Some(bytes);
        self
    }

    /// Send requests through `transport` instead of the default `reqwest`
    /// client: an `allama_http::ReqwestTransport` over your own client (a
    /// proxy, other TLS settings), or an [`allama_http::ReplayTransport`]
    /// in tests. A client of your own keeps its redirect policy: build it
    /// with `.redirect(reqwest::redirect::Policy::none())` to keep the
    /// default transport's refusal of redirects.
    ///
    /// The [`timeout`](Self::timeout), 120 s unless set, goes to the
    /// transport with each non-streaming request, whatever the transport.
    #[must_use]
    pub fn http_transport(mut self, transport: impl HttpTransport) -> Self {
        self.http = Some(Arc::new(transport));
        self
    }

    /// Build the provider.
    ///
    /// # Errors
    ///
    /// An authored [`Error::Configuration`], naming the setting, when:
    ///
    /// - the base URL is missing, or is not an absolute `http` or `https`
    ///   URL without a query ([`allama_http::config::base_url`]);
    /// - the auth value holds a character no HTTP header can carry
    ///   ([`allama_http::config::secret`]);
    /// - an organization, project or [`header`](Self::header) setting is
    ///   malformed or carries a credential
    ///   ([`allama_http::config::static_headers`]);
    /// - no transport is set and the `reqwest` feature is off.
    pub fn build(self) -> Result<Provider> {
        let base_url = match self.base_url {
            Some(url) => allama_http::config::base_url("base_url", &url)?,
            None => return Err(Error::configuration("base_url is required").authored()),
        };
        let auth = checked_auth(self.auth)?;
        let provider_name = self.provider_name.unwrap_or(Cow::Borrowed(PROVIDER_NAME));
        let quirks = self
            .quirks
            .unwrap_or_else(|| CompatQuirks::new(default_namespace(&provider_name)));
        let http = match self.http {
            Some(http) => http,
            None => default_transport()?,
        };
        let credential_slot = credential_slot(
            self.credential_scope,
            provider_name.clone(),
            self.credential_placement,
            auth.as_ref(),
        );
        let headers = allama_http::config::static_headers(
            self.organization_header
                .into_iter()
                .chain(self.project_header)
                .chain(self.headers),
            &credential_slot,
        )?;
        Ok(Provider {
            inner: Arc::new(Inner {
                http,
                transport: TransportConfig {
                    base_url,
                    chat_completions_path: self
                        .chat_completions_path
                        .unwrap_or_else(|| "/chat/completions".into()),
                    embeddings_path: self.embeddings_path.unwrap_or_else(|| "/embeddings".into()),
                    auth,
                    headers,
                    request_id_header_name: self
                        .request_id_header_name
                        .unwrap_or_else(|| "x-request-id".into()),
                    retry_after_header_name: self
                        .retry_after_header_name
                        .unwrap_or_else(|| "retry-after".into()),
                    timeout: resolve_timeout(self.timeout),
                    stream_timeouts: self.stream_timeouts.unwrap_or_default(),
                    max_stream_event_size: self
                        .max_stream_event_size
                        .unwrap_or(SseDecoder::DEFAULT_MAX_EVENT_SIZE),
                    max_response_size: self
                        .max_response_size
                        .unwrap_or(allama_http::DEFAULT_MAX_RESPONSE_SIZE),
                },
                quirks,
                chat_capabilities: self.chat_capabilities,
                chat_capability_resolver: None,
                model_rules: self.model_rules,
                embedding_capabilities: self.embedding_capabilities,
                max_embedding_inputs: self.max_embedding_inputs,
                provider_name,
                credential_slot,
            }),
        })
    }
}

/// The credential slot of a provider with `auth` configured.
///
/// The scope defaults to the provider name, and an explicit scope without
/// configured auth requires a per-request credential. The placement
/// defaults to where `auth` goes, and replaces `auth` when it differs.
pub(crate) fn credential_slot(
    scope: Option<Cow<'static, str>>,
    provider_name: Cow<'static, str>,
    placement: Option<CredentialPlacement>,
    auth: Option<&(String, String)>,
) -> CredentialSlot {
    let explicit = scope.is_some();
    let placement = placement.unwrap_or_else(|| match auth {
        Some((name, _)) if !name.eq_ignore_ascii_case("authorization") => {
            CredentialPlacement::Header(Cow::Owned(name.clone()))
        }
        _ => CredentialPlacement::Bearer,
    });
    let mut slot = CredentialSlot::new(scope.unwrap_or(provider_name), placement)
        .required(explicit && auth.is_none());
    if let (Some((name, _)), Some(header)) = (auth, slot.header_name())
        && !header.eq_ignore_ascii_case(name)
    {
        slot = slot.replaces(name.clone());
    }
    slot
}

/// `slot` under the explicitly chosen `scope`.
pub(crate) fn rescope(
    slot: &CredentialSlot,
    scope: Cow<'static, str>,
    auth: Option<&(String, String)>,
) -> CredentialSlot {
    let mut slot = slot.clone();
    slot.scope = scope;
    slot.required = auth.is_none();
    slot
}

/// The default provider-data namespace for `provider_name`.
pub(crate) fn default_namespace(provider_name: &str) -> String {
    if provider_name == PROVIDER_NAME {
        NAMESPACE.to_owned()
    } else {
        format!("{NAMESPACE}:{provider_name}")
    }
}

/// The non-streaming timeout a builder resolves: the one set with
/// `timeout`, else [`DEFAULT_REQUEST_TIMEOUT`], whatever the transport.
pub(crate) fn resolve_timeout(timeout: Option<Option<Duration>>) -> Option<Duration> {
    timeout.unwrap_or(Some(DEFAULT_REQUEST_TIMEOUT))
}

/// The configured auth header with its value trimmed and checked
/// ([`allama_http::config::secret`]), named after the header.
pub(crate) fn checked_auth(auth: Option<(String, String)>) -> Result<Option<(String, String)>> {
    auth.map(|(name, value)| {
        let value = allama_http::config::secret(&format!("the `{name}` header"), &value)?;
        Ok((name, value))
    })
    .transpose()
}

/// The transport of a provider built without `http_transport`: `reqwest`
/// (feature `reqwest`), with tracing middleware under `http-tracing`.
pub(crate) fn default_transport() -> Result<Arc<dyn HttpTransport>> {
    #[cfg(feature = "reqwest")]
    return Ok(Arc::new(ReqwestTransport::new()));
    #[cfg(not(feature = "reqwest"))]
    return Err(Error::configuration(
        "no HTTP transport: set one with `http_transport` or enable the `reqwest` feature",
    )
    .authored());
}

/// Endpoint and header configuration.
///
/// `Debug` output redacts the `auth` value whatever its header name, the
/// other header values when their names look like credentials
/// ([`allama_http::is_sensitive_header`]), and the base URL's credentials.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct TransportConfig {
    /// Base URL without a trailing slash.
    pub(crate) base_url: String,
    /// Chat completions path.
    pub(crate) chat_completions_path: String,
    /// Embeddings path.
    pub(crate) embeddings_path: String,
    /// Auth header, when the endpoint requires one.
    pub(crate) auth: Option<(String, String)>,
    /// The headers sent with every request: the `User-Agent`, the
    /// organization and project headers, and the builder's own.
    pub(crate) headers: Headers,
    /// Request id header name.
    pub(crate) request_id_header_name: String,
    /// Retry-after header name.
    pub(crate) retry_after_header_name: String,
    /// Non-streaming request timeout.
    pub(crate) timeout: Option<Duration>,
    /// Streaming request timeouts.
    pub(crate) stream_timeouts: StreamTimeouts,
    /// The largest stream event, in bytes, the provider holds in memory.
    pub(crate) max_stream_event_size: usize,
    /// The largest non-streaming response body, in bytes.
    pub(crate) max_response_size: usize,
}

impl std::fmt::Debug for TransportConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransportConfig")
            .field("base_url", &redact_url(&self.base_url))
            .field("chat_completions_path", &self.chat_completions_path)
            .field("embeddings_path", &self.embeddings_path)
            .field(
                "auth",
                &self.auth.as_ref().map(|(name, _)| (name, Redacted)),
            )
            .field("headers", &self.headers)
            .field("request_id_header_name", &self.request_id_header_name)
            .field("retry_after_header_name", &self.retry_after_header_name)
            .field("timeout", &self.timeout)
            .field("stream_timeouts", &self.stream_timeouts)
            .field("max_stream_event_size", &self.max_stream_event_size)
            .field("max_response_size", &self.max_response_size)
            .finish()
    }
}

impl TransportConfig {
    /// Full chat completions URL.
    pub(crate) fn chat_completions_url(&self) -> String {
        format!("{}{}", self.base_url, self.chat_completions_path)
    }

    /// Full embeddings URL.
    pub(crate) fn embeddings_url(&self) -> String {
        format!("{}{}", self.base_url, self.embeddings_path)
    }
}

/// Read a header value as a string.
#[must_use]
pub(crate) fn extract_request_id(headers: &Headers, name: &str) -> Option<String> {
    headers
        .get(name)
        .filter(|value| !value.is_empty())
        .map(String::from)
}

/// Parse a retry-after hint from `name` and `retry-after-ms`.
#[must_use]
pub(crate) fn extract_retry_after(headers: &Headers, name: &str) -> Option<Duration> {
    headers.retry_after(name)
}

/// A `Debug` placeholder for a secret value.
struct Redacted;

impl std::fmt::Debug for Redacted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<redacted>")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One timeout model for every builder: unset, the default applies
    /// whatever the transport, a client of your own included; only an
    /// explicit `None` removes it.
    #[cfg(feature = "reqwest")]
    #[test]
    fn the_timeout_is_kept_whatever_the_transport() {
        let five = Some(Duration::from_secs(5));
        let client = || ReqwestTransport::from_client(reqwest::Client::new());
        let timeout = |builder| ProviderBuilder::build(builder).unwrap().transport().timeout;
        let builder = || Provider::builder().base_url("http://localhost:1/v1");
        assert_eq!(timeout(builder()), Some(DEFAULT_REQUEST_TIMEOUT));
        assert_eq!(timeout(builder().timeout(None)), None);
        assert_eq!(
            timeout(builder().http_transport(client())),
            Some(DEFAULT_REQUEST_TIMEOUT)
        );
        assert_eq!(
            timeout(builder().http_transport(client()).timeout(five)),
            five
        );
    }

    /// The resolved timeout is what the transport receives with a
    /// non-streaming request.
    #[tokio::test]
    async fn the_transport_receives_the_resolved_timeout() {
        let five = Some(Duration::from_secs(5));
        for (set, expected) in [
            (None, Some(DEFAULT_REQUEST_TIMEOUT)),
            (Some(five), five),
            (Some(None), None),
        ] {
            let transport =
                allama_http::ReplayTransport::new([allama_http::ReplayResponse::new(500, "")]);
            let mut builder = Provider::builder()
                .base_url("http://localhost:1/v1")
                .http_transport(transport.clone());
            if let Some(timeout) = set {
                builder = builder.timeout(timeout);
            }
            let request = allama::EmbeddingRequest::new("m").input("hi");
            let _ = allama::EmbeddingProvider::embed(&builder.build().unwrap(), &request).await;
            assert_eq!(transport.requests()[0].timeout, expected, "{set:?}");
        }
    }

    #[cfg(not(feature = "reqwest"))]
    #[test]
    fn building_without_a_transport_needs_reqwest() {
        let Err(err) = Provider::builder()
            .base_url("http://localhost:8000/v1")
            .build()
        else {
            panic!("built without a transport");
        };
        assert!(matches!(err, Error::Configuration { .. }), "{err}");
        assert!(err.to_string().contains("`http_transport`"), "{err}");
    }

    #[test]
    fn builder_defaults() {
        use allama::ProviderIdentity;

        let provider = Provider::builder()
            .base_url("https://example.com/v1///")
            .provider_name("demo")
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(
            provider.transport().chat_completions_url(),
            "https://example.com/v1/chat/completions"
        );
        assert_eq!(provider.quirks().namespace, "openai-compat:demo");
        assert!(provider.reads_namespace("openai-compat:demo"));
        assert!(!provider.reads_namespace("demo"));
        assert!(!provider.reads_namespace(NAMESPACE));
        assert_eq!(provider.transport().timeout, Some(DEFAULT_REQUEST_TIMEOUT));
        assert_eq!(
            provider.transport().stream_timeouts,
            StreamTimeouts::default()
        );
        assert!(provider.transport().auth.is_none());
        assert!(Provider::builder().build().is_err());
        assert!(Provider::builder().base_url("  ").build().is_err());
    }

    #[test]
    fn debug_output_never_shows_credentials() {
        let provider = Provider::builder()
            .base_url("https://example.com/v1")
            // Not a recognizable credential header name: redacted anyway.
            .auth_header("x-custom", "SECRET1")
            .organization_header("openai-organization", "org-visible")
            .project_header("x-project-token", "SECRET2")
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        let debug = format!("{:?} {:?}", provider, provider.transport());
        assert!(!debug.contains("SECRET"), "{debug}");
        assert!(debug.contains("x-custom"), "{debug}");
        assert!(debug.contains("org-visible"), "{debug}");

        let builder = Provider::builder()
            .base_url("https://user:PASSWORD@example.com/v1")
            .bearer_token("sk-SECRET4")
            .header("x-route", "ROUTE-VALUE")
            .http_transport(allama_http::ReplayTransport::default());
        let builder_debug = format!("{builder:?}");
        let provider = builder.build().unwrap();
        for debug in [
            builder_debug,
            format!("{provider:?}"),
            format!("{:?}", provider.transport()),
        ] {
            assert!(!debug.contains("SECRET"), "{debug}");
            assert!(!debug.contains("PASSWORD"), "{debug}");
            assert!(debug.contains("example.com/v1"), "{debug}");
        }

        let bearer = Provider::builder()
            .base_url("https://example.com/v1")
            .bearer_token("sk-live-SECRET3")
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        let debug = format!("{:?}", bearer.transport());
        assert!(!debug.contains("SECRET"), "{debug}");
    }

    /// Auth values are trimmed and checked, the
    /// builder's headers and the default `User-Agent` reach every request,
    /// and credential headers are refused.
    #[tokio::test]
    async fn builder_checks_and_sends_its_headers() {
        let transport =
            allama_http::ReplayTransport::new([allama_http::ReplayResponse::new(500, "")]);
        let provider = Provider::builder()
            .base_url("http://localhost:1/v1")
            .bearer_token(" sk-x\n")
            .organization_header("openai-organization", "org-1")
            .header("x-route", "eu")
            .http_transport(transport.clone())
            .build()
            .unwrap();
        let _ = allama::EmbeddingProvider::embed(
            &provider,
            &allama::EmbeddingRequest::new("m").input("hi"),
        )
        .await;
        let sent = &transport.requests()[0];
        assert_eq!(sent.headers.get("authorization"), Some("Bearer sk-x"));
        assert_eq!(sent.headers.get("openai-organization"), Some("org-1"));
        assert_eq!(sent.headers.get("x-route"), Some("eu"));
        assert_eq!(
            sent.headers.get("user-agent"),
            Some(allama_http::DEFAULT_USER_AGENT)
        );

        for builder in [
            Provider::builder().auth_header("api-key", "k\r\nx"),
            Provider::builder().header("authorization", "Bearer x"),
            Provider::builder()
                .auth_header("x-gw-key", "k")
                .header("x-gw-key", "other"),
            Provider::builder().header("x-ok", "a\nb"),
        ] {
            let Err(err) = builder
                .base_url("http://localhost:1/v1")
                .http_transport(allama_http::ReplayTransport::default())
                .build()
            else {
                panic!("built");
            };
            assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
        }
    }

    #[test]
    fn retry_after_prefers_milliseconds() {
        let headers: Headers = [("retry-after", "3"), ("Retry-After-Ms", "1500")]
            .into_iter()
            .collect();
        assert_eq!(
            extract_retry_after(&headers, "retry-after"),
            Some(Duration::from_millis(1500))
        );
    }
}
