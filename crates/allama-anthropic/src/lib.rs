// Tests set environment variables, which is `unsafe` in edition 2024.
#![cfg_attr(not(test), forbid(unsafe_code))]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
//! Anthropic provider adapter for `allama`.
//!
//! Implements [`allama::ChatProvider`] against the Anthropic Messages API:
//! text, vision, PDF and plain-text documents, tool use (including strict
//! tools and parallel-call control), native structured outputs, extended
//! thinking (adaptive effort or fixed budgets, with signatures replayed),
//! prompt caching (breakpoints and automatic caching), and streaming.
//!
//! Anthropic-specific state lives under the `"anthropic"` namespace:
//! request options in [`ChatRequestOptions`], replay state (thinking
//! signatures, redacted thinking, raw citations, server tool blocks) in
//! [`allama::ProviderData`] and [`allama::OtherContent`] with origin
//! `"anthropic"`. Citations are also decoded into portable
//! [`allama::Citation`]s on the text they support. Embeddings are out of
//! scope (Anthropic has no embeddings API).
//!
//! ```no_run
//! use allama::prelude::*;
//! use allama_anthropic::Provider;
//!
//! # #[cfg(feature = "reqwest")]
//! # async fn example() -> allama::Result<()> {
//! let provider = Provider::from_env()?;
//! let response = provider
//!     .chat(&ChatRequest::new("claude-haiku-4-5").user("Say hello."))
//!     .await?;
//! println!("{}", response.text_or_empty());
//! # Ok(()) }
//! ```
//!
//! # Transport
//!
//! Requests go through `reqwest` (the default `reqwest` feature) unless the
//! builder is given another [`HttpTransport`] with
//! [`http_transport`](ProviderBuilder::http_transport). With
//! `default-features = false` and an injected transport the crate builds
//! without `reqwest`; the constructors that would build a `reqwest` client
//! (`new`, `new_with_base_url`, `from_env`) are then not compiled,
//! and `build()` without a transport is a configuration error. TLS backends:
//! `default-tls`, `native-tls`, `rustls`, and `rustls-ring` (`rustls` with
//! `ring` instead of `aws-lc-rs`, so musl builds need no cmake).
//!
//! # Mapping notes
//!
//! - `max_output_tokens` maps to `max_tokens` and defaults to 4096 when unset
//!   (Anthropic requires it). It caps thinking and visible output together,
//!   and a caller-set value is never raised. A fixed thinking budget must be
//!   below it (except with the interleaved-thinking beta and tools):
//!   - when `max_output_tokens` is unset, the default grows to
//!     `budget + 4096`;
//!   - a budget derived from the effort is lowered to half of
//!     `max_output_tokens`, but not below the 1024-token minimum (an
//!     `Approximated` warning, never refused), so it never shrinks as the
//!     cap grows; at or below 1024, thinking is not enabled (an unsupported
//!     hint);
//!   - a `budget_tokens` option that does not fit is `Error::InvalidRequest`.
//! - `ReasoningConfig.effort` maps to `thinking: {type: "adaptive"}` plus
//!   `output_config.effort` on every model but the legacy generations
//!   (Claude 3, Claude 4.0 and 4.1, Opus 4.5, Sonnet and Haiku 4.5), which
//!   take a fixed `budget_tokens`. A later version of a family, and an id
//!   the adapter does not recognize, get the adaptive form. `ReasoningEffort::None` sends the
//!   model's off value; a model that cannot turn thinking off refuses it.
//!   A `ReasoningOutput` without an effort never turns thinking on: it is
//!   sent as `display` on models that think by default and is an
//!   unsupported hint elsewhere. [`ChatRequestOptions::effort`] replaces
//!   the mapped effort level (`max` on a model that also has `xhigh`).
//! - `PromptCache::automatic()` turns on automatic caching (top-level
//!   `cache_control`); a cache key, or a retention alone, does not.
//! - Nothing is refused, dropped or rewritten because of the model id
//!   (ADR 0015): a setting a model rejects is sent, and the API's refusal
//!   maps to `Error::Unsupported` with stage `Provider`. Answer a
//!   capability with [`Provider::with_chat_capabilities`] to refuse before
//!   sending, and replace a wire-format choice with
//!   [`Provider::with_model_rules`].
//! - A prefill (an assistant message sent last) is sent as written. Claude
//!   4.6 and later generations refuse it, and earlier ones refuse it with
//!   an output format; that refusal is `Error::Unsupported` with stage
//!   `Provider`, naming the message (`Subject::Content`, no block). End the
//!   conversation with a user message to avoid it.
//! - `ResponseFormat::JsonSchema` maps to `output_config.format`; schemaless
//!   `ResponseFormat::Json` is unsupported.
//! - `stop_reason: "pause_turn"` maps to `FinishReason::Other("pause_turn")`:
//!   Anthropic paused a long turn of [server
//!   tools](ChatRequestOptions::server_tools). To let it finish, append
//!   `response.to_assistant_message()` to the request and call again
//!   unchanged; the paused server tool call is sent back as it came.
//! - Seed and penalties are unsupported hints (warning, or error under
//!   `UnsupportedPolicy::Error`).
//!
//! # Per-request credentials
//!
//! A [`allama_http::Credential::api_key`] addressed to the provider's
//! credential scope ([`DEFAULT_CREDENTIAL_SCOPE`] unless changed with
//! [`ProviderBuilder::credential_scope`]) replaces the configured key in
//! `x-api-key` for that request; credentials addressed to other scopes are
//! never sent. See [`allama_http::HttpRequestOptions`].

use allama::{CapabilitySupport, ChatCapability, ChatCapabilityResolver, Error, Result};
#[cfg(feature = "reqwest")]
use allama_http::ReqwestTransport;
pub use allama_http::StreamTimeouts;
use allama_http::config::redact_url;
use allama_http::sse::SseDecoder;
use allama_http::{
    CredentialPlacement, CredentialSlot, DEFAULT_REQUEST_TIMEOUT, Headers, HttpTransport,
};
use std::borrow::Cow;
use std::{sync::Arc, time::Duration};

mod chat;
#[cfg(test)]
mod conformance_tests;
mod decode;
mod encode;
mod error;
mod model;
mod options;
pub mod schema;
mod streaming;
mod wire;

pub use model::{EffortLevel, ModelRules};
pub use options::ChatRequestOptions;

/// The verbatim Messages API response body, kept in
/// [`ChatResponse::provider_data`](allama::ChatResponse::provider_data) when
/// [`ChatRequestOptions::raw_response`] is set (unary calls only).
///
/// ```
/// # fn example(response: &allama::ChatResponse) -> allama::Result<()> {
/// use allama_anthropic::RawResponse;
///
/// if let Some(raw) = response.provider_data.get::<RawResponse>()? {
///     println!("{}", raw.0["stop_reason"]);
/// }
/// # Ok(()) }
/// ```
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct RawResponse(pub serde_json::Value);

impl allama::ProviderDataEntry for RawResponse {
    const NAMESPACE: &'static str = NAMESPACE;
    const KEY: &'static str = allama::provider_data::RAW_RESPONSE_KEY;
}

/// Namespace for Anthropic options and provider data, and the origin of
/// Anthropic-specific content.
pub const NAMESPACE: &str = "anthropic";

/// The [`provider_name`](allama::ProviderIdentity::provider_name) of
/// [`Provider`], stamped on responses, errors, and warnings.
pub const PROVIDER_NAME: &str = "anthropic";

/// The default credential scope of [`Provider`].
///
/// Per-request credentials addressed to it
/// ([`allama_http::HttpRequestOptions::credential`]) are this provider's:
/// a [`allama_http::Credential::api_key`] is sent in `x-api-key`, replacing
/// the configured key. The scope names an account, independent of the
/// provider name (telemetry) and the namespaces. Change it with
/// [`ProviderBuilder::credential_scope`].
pub const DEFAULT_CREDENTIAL_SCOPE: &str = "anthropic";

/// The header Anthropic reads its API key from.
const API_KEY_HEADER: &str = "x-api-key";

/// Anthropic API provider implementing `allama::ChatProvider`.
///
/// Clone is cheap: internals are wrapped in `Arc`. `Debug` shows the base
/// URL without credentials ([`allama_http::config::redact_url`]) and never
/// the API key.
#[derive(Clone)]
pub struct Provider {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Provider")
            .field("base_url", &redact_url(&self.inner.base_url))
            .field("credential_scope", &self.inner.credential_slot.scope)
            .field(
                "api_key",
                &self.inner.api_key.as_ref().map(|_| "<redacted>"),
            )
            .finish_non_exhaustive()
    }
}

pub(crate) struct Inner {
    pub(crate) http: Arc<dyn HttpTransport>,
    pub(crate) request_timeout: Option<Duration>,
    pub(crate) stream_timeouts: StreamTimeouts,
    pub(crate) max_stream_event_size: usize,
    pub(crate) max_response_size: usize,
    /// The headers sent with every request: the `User-Agent` and the
    /// builder's own.
    pub(crate) headers: Headers,
    /// The configured key; `None` when every request brings its own.
    pub(crate) api_key: Option<String>,
    pub(crate) credential_slot: CredentialSlot,
    pub(crate) base_url: String,
    pub(crate) chat_capability_resolver: Option<Arc<dyn ChatCapabilityResolver>>,
    pub(crate) model_rules: Option<Arc<ModelRulesFn>>,
}

/// A caller's model rules, as [`Provider::with_model_rules`] stores them.
type ModelRulesFn = dyn Fn(&str) -> Option<ModelRules> + Send + Sync;

/// The credential slot for `scope`: `x-api-key`, required when no key is
/// configured and the scope was chosen explicitly.
fn credential_slot(scope: Cow<'static, str>, has_key: bool, explicit: bool) -> CredentialSlot {
    CredentialSlot::new(scope, CredentialPlacement::Header(API_KEY_HEADER.into()))
        .required(!has_key && explicit)
}

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// The value of environment variable `name`, unless it is unset or blank.
#[cfg(feature = "reqwest")]
fn env_var(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

impl Provider {
    /// The transport of a provider built without
    /// [`http_transport`](ProviderBuilder::http_transport): `reqwest`, with
    /// the `reqwest` feature.
    fn default_transport() -> Result<Arc<dyn HttpTransport>> {
        #[cfg(feature = "reqwest")]
        return Ok(Arc::new(ReqwestTransport::new()));
        #[cfg(not(feature = "reqwest"))]
        return Err(Error::configuration(
            "no HTTP transport: set one with `http_transport` or enable the `reqwest` feature",
        )
        .authored());
    }

    /// Create with just an API key. Uses default base URL.
    ///
    /// # Errors
    ///
    /// See [`ProviderBuilder::build`].
    #[cfg(feature = "reqwest")]
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        Self::builder().api_key(api_key).build()
    }

    /// Create with an API key and explicit base URL.
    ///
    /// # Errors
    ///
    /// See [`ProviderBuilder::build`].
    #[cfg(feature = "reqwest")]
    pub fn new_with_base_url(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Result<Self> {
        Self::builder().api_key(api_key).base_url(base_url).build()
    }

    /// Install a resolver consulted before the provider's built-in capability logic.
    #[must_use]
    pub fn with_chat_capabilities(self, resolver: impl ChatCapabilityResolver) -> Self {
        let resolver: Arc<dyn ChatCapabilityResolver> = Arc::new(resolver);
        self.rebuild(|inner| inner.chat_capability_resolver = Some(resolver))
    }

    /// Install model rules: wire-format choices for a model, which replace
    /// what the adapter derives from the model id (see [`ModelRules`]).
    /// `rules` returns `None` for a model it has nothing to say about.
    ///
    /// Capability answers go through
    /// [`with_chat_capabilities`](Self::with_chat_capabilities); rules
    /// change only how a request is encoded.
    #[must_use]
    pub fn with_model_rules(
        self,
        rules: impl Fn(&str) -> Option<ModelRules> + Send + Sync + 'static,
    ) -> Self {
        let rules: Arc<ModelRulesFn> = Arc::new(rules);
        self.rebuild(|inner| inner.model_rules = Some(rules))
    }

    /// The same provider with its credential scope set to `scope` (see
    /// [`ProviderBuilder::credential_scope`]).
    ///
    /// A provider without a configured key then requires a credential for
    /// `scope` on every request.
    #[must_use]
    pub fn with_credential_scope(self, scope: impl Into<Cow<'static, str>>) -> Self {
        let scope = scope.into();
        self.rebuild(|inner| {
            inner.credential_slot = credential_slot(scope, inner.api_key.is_some(), true);
        })
    }

    /// The credential slot this provider passes to
    /// [`allama_http::HttpRequest::apply_options`]: its credential scope,
    /// and `x-api-key` as the placement.
    #[must_use]
    pub fn credential_slot(&self) -> &CredentialSlot {
        &self.inner.credential_slot
    }

    /// A copy of the provider with `change` applied to its configuration.
    fn rebuild(self, change: impl FnOnce(&mut Inner)) -> Self {
        let mut inner = Inner {
            http: self.inner.http.clone(),
            request_timeout: self.inner.request_timeout,
            stream_timeouts: self.inner.stream_timeouts,
            max_stream_event_size: self.inner.max_stream_event_size,
            max_response_size: self.inner.max_response_size,
            headers: self.inner.headers.clone(),
            api_key: self.inner.api_key.clone(),
            credential_slot: self.inner.credential_slot.clone(),
            base_url: self.inner.base_url.clone(),
            chat_capability_resolver: self.inner.chat_capability_resolver.clone(),
            model_rules: self.inner.model_rules.clone(),
        };
        change(&mut inner);
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Create from environment variables.
    ///
    /// Required: `ANTHROPIC_API_KEY`
    /// Optional: `ANTHROPIC_BASE_URL`
    ///
    /// # Errors
    ///
    /// An authored [`Error::Configuration`] naming the variable when
    /// `ANTHROPIC_API_KEY` is unset or blank, or when a variable holds a
    /// value [`ProviderBuilder::build`] refuses (a key no HTTP header can
    /// carry, a base URL that is not an absolute `http(s)` URL).
    #[cfg(feature = "reqwest")]
    pub fn from_env() -> Result<Self> {
        let api_key = env_var("ANTHROPIC_API_KEY")
            .ok_or_else(|| Error::configuration("ANTHROPIC_API_KEY not set").authored())?;
        let mut builder =
            Self::builder().api_key(allama_http::config::secret("ANTHROPIC_API_KEY", &api_key)?);
        if let Some(url) = env_var("ANTHROPIC_BASE_URL") {
            builder = builder.base_url(allama_http::config::base_url("ANTHROPIC_BASE_URL", &url)?);
        }
        builder.build()
    }

    /// Builder for full configuration.
    #[must_use]
    pub fn builder() -> ProviderBuilder {
        ProviderBuilder {
            api_key: None,
            credential_scope: None,
            base_url: None,
            http: None,
            timeout: None,
            stream_timeouts: StreamTimeouts::default(),
            max_stream_event_size: SseDecoder::DEFAULT_MAX_EVENT_SIZE,
            max_response_size: allama_http::DEFAULT_MAX_RESPONSE_SIZE,
            headers: Vec::new(),
        }
    }
}

/// Builder for configuring an Anthropic provider.
///
/// `Debug` shows the settings without the API key, header values, or
/// credentials in the base URL.
pub struct ProviderBuilder {
    api_key: Option<String>,
    credential_scope: Option<Cow<'static, str>>,
    base_url: Option<String>,
    http: Option<Arc<dyn HttpTransport>>,
    timeout: Option<Option<Duration>>,
    stream_timeouts: StreamTimeouts,
    max_stream_event_size: usize,
    max_response_size: usize,
    headers: Vec<(String, String)>,
}

impl std::fmt::Debug for ProviderBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderBuilder")
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("credential_scope", &self.credential_scope)
            .field("base_url", &self.base_url.as_deref().map(redact_url))
            .field("http_transport", &self.http.is_some())
            .field("timeout", &self.timeout)
            .field("stream_timeouts", &self.stream_timeouts)
            .field("max_stream_event_size", &self.max_stream_event_size)
            .field("max_response_size", &self.max_response_size)
            .field(
                "headers",
                &self
                    .headers
                    .iter()
                    .map(|(name, _)| name)
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl ProviderBuilder {
    /// The configured API key, sent in `x-api-key` unless a request carries
    /// a credential for the provider's credential scope. Surrounding
    /// whitespace is removed.
    #[must_use]
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// The credential scope (default [`DEFAULT_CREDENTIAL_SCOPE`]): a
    /// per-request [`allama_http::Credential`] addressed to it
    /// ([`allama_http::HttpRequestOptions::credential`]) is this provider's,
    /// and credentials addressed to other scopes are never sent.
    ///
    /// The scope names an account: two Anthropic accounts in one fallback
    /// chain need two scopes (the default makes them share one), while
    /// providers behind one gateway share one.
    ///
    /// Setting it lets [`build`](Self::build) go without an
    /// [`api_key`](Self::api_key): every request must then carry a
    /// credential for the scope (bring your own key), or it fails with an
    /// authored [`Error::Configuration`] before anything is sent. With a
    /// configured key, a request whose credential names a mistyped scope is
    /// sent with the configured key.
    #[must_use]
    pub fn credential_scope(mut self, scope: impl Into<Cow<'static, str>>) -> Self {
        self.credential_scope = Some(scope.into());
        self
    }

    /// Base URL (default `https://api.anthropic.com`), for a proxy or
    /// gateway that serves the same API. Surrounding whitespace and trailing
    /// slashes are removed; requests go to `{base_url}/v1/messages`. It
    /// must be an absolute `http` or `https` URL without a query
    /// ([`allama_http::config::base_url`]).
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Send requests through `transport` instead of the default `reqwest`
    /// client: an [`allama_http::ReqwestTransport`] over your own client
    /// (a proxy, other TLS settings), or an [`allama_http::ReplayTransport`]
    /// in tests.
    ///
    /// ```
    /// # #[cfg(feature = "reqwest")]
    /// # fn example() -> allama::Result<()> {
    /// use allama_http::ReqwestTransport;
    ///
    /// let client = reqwest::Client::builder()
    ///     // Keep the default transport's refusal of redirects.
    ///     .redirect(reqwest::redirect::Policy::none())
    ///     .build()
    ///     .map_err(|err| allama::Error::configuration(err.to_string()))?;
    /// let provider = allama_anthropic::Provider::builder()
    ///     .api_key("sk-ant-...")
    ///     .http_transport(ReqwestTransport::from_client(client))
    ///     .build()?;
    /// # let _ = provider; Ok(()) }
    /// ```
    ///
    /// The [`timeout`](Self::timeout), 120 s unless set, goes to the
    /// transport with each non-streaming request, whatever the transport.
    ///
    #[cfg_attr(
        not(feature = "reqwest"),
        doc = "[`allama_http::ReqwestTransport`]: crate#transport"
    )]
    #[must_use]
    pub fn http_transport(mut self, transport: impl HttpTransport) -> Self {
        self.http = Some(Arc::new(transport));
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
        self.stream_timeouts = timeouts;
        self
    }

    /// The largest stream event, in bytes, the provider holds in memory
    /// (default [`SseDecoder::DEFAULT_MAX_EVENT_SIZE`], 64 MiB). A larger
    /// event fails the stream with a stream protocol error, which is not
    /// retryable. See [`SseDecoder::max_event_size`].
    #[must_use]
    pub fn max_stream_event_size(mut self, bytes: usize) -> Self {
        self.max_stream_event_size = bytes;
        self
    }

    /// The largest non-streaming response body, in bytes, the provider
    /// reads (default [`allama_http::DEFAULT_MAX_RESPONSE_SIZE`], 256 MiB).
    /// A larger body fails the call with [`Error::UnexpectedResponse`],
    /// which is not retryable, without being read whole. See
    /// [`allama_http::HttpResponse::bytes_limited`].
    #[must_use]
    pub fn max_response_size(mut self, bytes: usize) -> Self {
        self.max_response_size = bytes;
        self
    }

    /// Send the header `name: value` with every request, replacing a
    /// header of the same name the provider sets (later calls replace
    /// earlier ones). For headers that belong to this provider: a gateway's
    /// routing header, a tenant tag, `anthropic-beta`. `header("user-agent",
    /// ..)` replaces the default [`allama_http::DEFAULT_USER_AGENT`]. A
    /// request's own [`allama_http::HttpRequestOptions`] headers still
    /// replace these.
    ///
    /// Credential headers are refused at [`build`](Self::build): set the
    /// key with [`api_key`](Self::api_key), or send one per request as an
    /// [`allama_http::Credential`].
    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Build the provider.
    ///
    /// # Errors
    ///
    /// An authored [`Error::Configuration`], naming the setting:
    ///
    /// - the API key is missing or empty and no
    ///   [`credential_scope`](Self::credential_scope) is set, or it holds a
    ///   character no HTTP header can carry
    ///   ([`allama_http::config::secret`]);
    /// - the base URL is not an absolute `http` or `https` URL without a
    ///   query ([`allama_http::config::base_url`]);
    /// - a [`header`](Self::header) is malformed or carries a credential
    ///   ([`allama_http::config::static_headers`]);
    /// - no transport is set and the `reqwest` feature is off.
    pub fn build(self) -> Result<Provider> {
        let explicit = self.credential_scope.is_some();
        let api_key = match self.api_key {
            None if explicit => None,
            None => return Err(Error::configuration("api_key is required").authored()),
            Some(key) => Some(allama_http::config::secret("api_key", &key)?),
        };
        let base_url = match self.base_url {
            Some(url) => allama_http::config::base_url("base_url", &url)?,
            None => DEFAULT_BASE_URL.to_owned(),
        };
        let scope = self
            .credential_scope
            .unwrap_or(Cow::Borrowed(DEFAULT_CREDENTIAL_SCOPE));
        let credential_slot = credential_slot(scope, api_key.is_some(), explicit);
        let headers = allama_http::config::static_headers(self.headers, &credential_slot)?;
        Ok(Provider {
            inner: Arc::new(Inner {
                http: match self.http {
                    Some(http) => http,
                    None => Provider::default_transport()?,
                },
                request_timeout: self.timeout.unwrap_or(Some(DEFAULT_REQUEST_TIMEOUT)),
                stream_timeouts: self.stream_timeouts,
                max_stream_event_size: self.max_stream_event_size,
                max_response_size: self.max_response_size,
                headers,
                credential_slot,
                api_key,
                base_url,
                chat_capability_resolver: None,
                model_rules: None,
            }),
        })
    }
}

/// Built-in capability answers; see [`Provider::with_chat_capabilities`] to
/// override them.
pub(crate) fn builtin_chat_capability(
    model: &str,
    capability: ChatCapability,
) -> CapabilitySupport {
    let profile = model::ModelProfile::for_model(model);
    match capability {
        ChatCapability::ToolCalls
        | ChatCapability::ParallelToolCalls
        | ChatCapability::Streaming
        | ChatCapability::NativeStreaming
        | ChatCapability::ImageInput
        | ChatCapability::FileInput
        | ChatCapability::PromptCache => CapabilitySupport::Supported,
        // `message_start` and `message_delta` carry usage on every model:
        // https://docs.anthropic.com/en/docs/build-with-claude/streaming
        // (captured in `fixtures/stream.sse`).
        ChatCapability::StreamUsage => CapabilitySupport::Supported,
        ChatCapability::ReasoningOutput
        | ChatCapability::ReasoningReplay
        | ChatCapability::ReasoningConfig => profile.thinking(),
        ChatCapability::StructuredOutput => profile.structured_output(),
        ChatCapability::ForcedToolChoice => profile.forced_tool_choice(),
        ChatCapability::ImageOutput | ChatCapability::ImageDetail | ChatCapability::ImageReplay => {
            CapabilitySupport::Unsupported
        }
        // Messages takes images and documents (PDF, plain text), no audio
        // or video: https://docs.anthropic.com/en/api/messages
        ChatCapability::AudioInput | ChatCapability::VideoInput => CapabilitySupport::Unsupported,
        _ => CapabilitySupport::Unknown,
    }
}

/// A chat stream decoded from recorded SSE `text`, as `chat_stream` drives
/// it (for the conformance fixtures).
#[cfg(test)]
pub(crate) fn conformance_stream_from_sse_text(text: &str) -> allama::ChatStream {
    let normalized = if text.ends_with("\n\n") {
        text.to_string()
    } else {
        format!("{text}\n\n")
    };

    // As `chat_stream` drives it, with the hold `finish_stream` applies.
    let events = allama_http::sse::SseStream::new(futures_util::stream::iter([Ok::<
        Vec<u8>,
        allama_http::TransportError,
    >(
        normalized.into_bytes(),
    )]));
    Box::pin(allama::adapter::chat::start_first(
        allama_http::sse::SseChatStream::new(events, streaming::SseState::new(None, Vec::new())),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use allama::{ChatCapability, ChatProvider};
    #[cfg(feature = "reqwest")]
    use std::sync::{LazyLock, Mutex};

    #[cfg(feature = "reqwest")]
    static ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    #[cfg(feature = "reqwest")]
    struct EnvVarGuard {
        name: &'static str,
        original: Option<String>,
    }

    #[cfg(feature = "reqwest")]
    impl EnvVarGuard {
        fn set(name: &'static str, value: Option<&str>) -> Self {
            let original = std::env::var(name).ok();

            unsafe {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }

            Self { name, original }
        }
    }

    #[cfg(feature = "reqwest")]
    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            unsafe {
                match &self.original {
                    Some(value) => std::env::set_var(self.name, value),
                    None => std::env::remove_var(self.name),
                }
            }
        }
    }

    /// A provider over a replay transport, so tests run without `reqwest`.
    fn test_provider() -> Provider {
        Provider::builder()
            .api_key("test-key")
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap()
    }

    #[test]
    fn reads_only_its_namespace() {
        use allama::ProviderIdentity;

        let provider = test_provider();
        assert!(provider.reads_namespace(NAMESPACE));
        assert!(!provider.reads_namespace("openai"));
        assert!(!provider.reads_namespace(allama::provider_data::ALLAMA_NAMESPACE));
    }

    #[test]
    fn builder_requires_api_key() {
        let result = Provider::builder().build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_rejects_empty_api_key() {
        let err = match Provider::builder().api_key("   ").build() {
            Ok(_) => panic!("expected empty API key to be rejected"),
            Err(err) => err,
        };
        assert!(
            matches!(&err, Error::Configuration { message, .. } if message == "`api_key` is empty")
        );
    }

    #[test]
    fn builder_rejects_empty_base_url() {
        let err = match Provider::builder()
            .api_key("test-key")
            .base_url("   ")
            .build()
        {
            Ok(_) => panic!("expected empty base URL to be rejected"),
            Err(err) => err,
        };
        assert!(
            matches!(&err, Error::Configuration { message, .. } if message == "`base_url` is empty")
        );
    }

    #[cfg(feature = "reqwest")]
    #[test]
    fn from_env_rejects_empty_api_key() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _api_key = EnvVarGuard::set("ANTHROPIC_API_KEY", Some("   "));
        let _base_url = EnvVarGuard::set("ANTHROPIC_BASE_URL", None);

        let err = match Provider::from_env() {
            Ok(_) => panic!("expected empty API key to be rejected"),
            Err(err) => err,
        };
        assert!(
            matches!(&err, Error::Configuration { message, .. } if message == "ANTHROPIC_API_KEY not set")
        );
    }

    #[test]
    fn chat_capability_resolver_overrides_builtin_answer() {
        let provider = test_provider().with_chat_capabilities(|model: &str, capability| {
            if model == "claude-sonnet-4-20250514" && capability == ChatCapability::ReasoningOutput
            {
                Some(CapabilitySupport::Unknown)
            } else {
                None
            }
        });

        assert_eq!(
            provider.chat_capability("claude-sonnet-4-20250514", ChatCapability::ReasoningOutput),
            CapabilitySupport::Unknown
        );
        assert_eq!(
            provider.chat_capability("claude-opus-4-1", ChatCapability::ReasoningOutput),
            CapabilitySupport::Supported
        );
    }

    #[test]
    fn builtin_anthropic_capabilities_report_native_streaming() {
        let provider = test_provider();
        assert_eq!(
            provider.chat_capability("claude-sonnet-4-20250514", ChatCapability::NativeStreaming),
            CapabilitySupport::Supported
        );
    }

    #[test]
    fn later_chat_capability_resolver_replaces_earlier_one() {
        let provider = test_provider()
            .with_chat_capabilities(|_: &str, capability| {
                if capability == ChatCapability::ToolCalls {
                    Some(CapabilitySupport::Unsupported)
                } else {
                    None
                }
            })
            .with_chat_capabilities(|_: &str, capability| {
                if capability == ChatCapability::ToolCalls {
                    Some(CapabilitySupport::Supported)
                } else {
                    None
                }
            });

        assert_eq!(
            provider.chat_capability("claude-sonnet-4-20250514", ChatCapability::ToolCalls),
            CapabilitySupport::Supported
        );
    }

    /// Built-in answers are endpoint facts or positive claims: a model the
    /// id does not settle is `Unknown`, never `Unsupported` (ADR 0015).
    #[test]
    fn builtin_capabilities_are_model_aware() {
        let provider = test_provider();
        let cap = |model: &str, c| provider.chat_capability(model, c);
        assert_eq!(
            cap("claude-haiku-4-5", ChatCapability::ImageOutput),
            CapabilitySupport::Unsupported
        );
        assert_eq!(
            cap("claude-haiku-4-5", ChatCapability::ParallelToolCalls),
            CapabilitySupport::Supported
        );
        assert_eq!(
            cap("claude-haiku-4-5", ChatCapability::FileInput),
            CapabilitySupport::Supported
        );
        assert_eq!(
            cap("claude-sonnet-5-5", ChatCapability::StructuredOutput),
            CapabilitySupport::Supported
        );
        assert_eq!(
            cap("claude-sonnet-4-20250514", ChatCapability::StructuredOutput),
            CapabilitySupport::Unknown
        );
        assert_eq!(
            cap("claude-3-5-haiku-latest", ChatCapability::ReasoningConfig),
            CapabilitySupport::Unknown
        );
        assert_eq!(
            cap("custom-proxy-model", ChatCapability::StructuredOutput),
            CapabilitySupport::Unknown
        );
        assert_eq!(
            cap("claude-haiku-4-5", ChatCapability::ForcedToolChoice),
            CapabilitySupport::Supported
        );
        assert_eq!(
            cap("claude-sonnet-5-5", ChatCapability::ForcedToolChoice),
            CapabilitySupport::Unknown
        );
        assert_eq!(
            cap("custom-proxy-model", ChatCapability::ForcedToolChoice),
            CapabilitySupport::Unknown
        );
    }

    #[cfg(feature = "reqwest")]
    #[test]
    fn base_url_is_normalized_when_constructed() {
        let provider = Provider::new_with_base_url("test-key", "https://example.com///").unwrap();
        assert_eq!(provider.inner.base_url, "https://example.com");
    }

    #[cfg(feature = "reqwest")]
    #[test]
    fn new_rejects_empty_api_key() {
        let err = match Provider::new("   ") {
            Ok(_) => panic!("expected empty API key to be rejected"),
            Err(err) => err,
        };
        assert!(
            matches!(&err, Error::Configuration { message, .. } if message == "`api_key` is empty")
        );
    }

    #[cfg(feature = "reqwest")]
    #[test]
    fn new_with_base_url_rejects_empty_base_url() {
        let err = match Provider::new_with_base_url("test-key", "   ") {
            Ok(_) => panic!("expected empty base URL to be rejected"),
            Err(err) => err,
        };
        assert!(
            matches!(&err, Error::Configuration { message, .. } if message == "`base_url` is empty")
        );
    }

    #[cfg(not(feature = "reqwest"))]
    #[test]
    fn building_without_a_transport_needs_reqwest() {
        let Err(err) = Provider::builder().api_key("test-key").build() else {
            panic!("built without a transport");
        };
        assert!(matches!(err, Error::Configuration { .. }), "{err}");
        assert!(err.to_string().contains("`http_transport`"), "{err}");
    }

    #[test]
    fn builder_normalizes_base_url() {
        let provider = Provider::builder()
            .api_key("test-key")
            .base_url("https://example.com///")
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();

        assert_eq!(provider.inner.base_url, "https://example.com");
    }

    /// One timeout model for every builder: unset, the default applies
    /// whatever the transport, a client of your own included; only an
    /// explicit `None` removes it.
    #[cfg(feature = "reqwest")]
    #[test]
    fn the_timeout_is_kept_whatever_the_transport() {
        let five = Some(Duration::from_secs(5));
        let client = || allama_http::ReqwestTransport::from_client(reqwest::Client::new());
        let timeout = |builder| {
            ProviderBuilder::build(builder)
                .unwrap()
                .inner
                .request_timeout
        };
        let builder = || Provider::builder().api_key("k");
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
        assert_eq!(
            timeout(builder().timeout(None).http_transport(client())),
            None
        );
    }

    /// Keys and URLs are checked at `build`, naming the setting,
    /// and a key read with a trailing newline is trimmed.
    #[test]
    fn the_builder_trims_keys_and_checks_urls() {
        let provider = Provider::builder()
            .api_key(" sk-ant-x\n")
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(provider.inner.api_key.as_deref(), Some("sk-ant-x"));
        for (key, url) in [
            ("sk-ant\nx", "https://api.anthropic.com"),
            ("k", "localhost:8080"),
            ("k", "https://gw.example/v1?x=1"),
        ] {
            let Err(err) = Provider::builder()
                .api_key(key)
                .base_url(url)
                .http_transport(allama_http::ReplayTransport::default())
                .build()
            else {
                panic!("built with {key:?} and {url:?}");
            };
            assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
        }
    }

    #[cfg(feature = "reqwest")]
    #[test]
    fn from_env_trims_the_key_and_names_a_bad_variable() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _api_key = EnvVarGuard::set("ANTHROPIC_API_KEY", Some("sk-ant-x\n"));
        let _base_url = EnvVarGuard::set("ANTHROPIC_BASE_URL", None);
        let provider = Provider::from_env().unwrap();
        assert_eq!(provider.inner.api_key.as_deref(), Some("sk-ant-x"));

        let _base_url = EnvVarGuard::set("ANTHROPIC_BASE_URL", Some("localhost:8080"));
        let err = Provider::from_env().unwrap_err();
        assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
        assert!(err.to_string().contains("ANTHROPIC_BASE_URL"), "{err}");
    }

    /// `Debug` never shows the key or the base URL's credentials.
    #[test]
    fn debug_output_is_redacted() {
        let builder = Provider::builder()
            .api_key("sk-ant-SECRET")
            .base_url("https://user:PASSWORD@gw.example/anthropic")
            .header("x-route", "ROUTE-VALUE")
            .http_transport(allama_http::ReplayTransport::default());
        let builder_debug = format!("{builder:?}");
        let provider = builder.build().unwrap();
        for debug in [builder_debug, format!("{provider:?}")] {
            assert!(!debug.contains("SECRET"), "{debug}");
            assert!(!debug.contains("PASSWORD"), "{debug}");
            assert!(!debug.contains("ROUTE-VALUE"), "{debug}");
            assert!(debug.contains("gw.example/anthropic"), "{debug}");
        }
    }

    /// The builder's headers and the default `User-Agent`
    /// reach every request, a request's own options still win, and
    /// credential headers are refused.
    #[tokio::test]
    async fn builder_headers_are_sent_with_every_request() {
        let transport =
            allama_http::ReplayTransport::new([allama_http::ReplayResponse::new(500, "")]);
        let provider = Provider::builder()
            .api_key("k")
            .header("x-route", "eu")
            .header("x-tenant", "acme")
            .http_transport(transport.clone())
            .build()
            .unwrap();
        let request = allama::ChatRequest::new("m")
            .user("hi")
            .with_option(allama_http::HttpRequestOptions::new().header("x-tenant", "other"));
        let _ = provider.chat(&request).await;
        let sent = &transport.requests()[0];
        assert_eq!(
            sent.headers.get("user-agent"),
            Some(allama_http::DEFAULT_USER_AGENT)
        );
        assert_eq!(sent.headers.get("x-route"), Some("eu"));
        assert_eq!(sent.headers.get("x-tenant"), Some("other"));
        assert_eq!(sent.headers.get("x-api-key"), Some("k"));

        for name in ["x-api-key", "Authorization"] {
            let Err(err) = Provider::builder()
                .api_key("k")
                .header(name, "v")
                .http_transport(allama_http::ReplayTransport::default())
                .build()
            else {
                panic!("built with a `{name}` header");
            };
            assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
        }
    }

    /// A success body past `max_response_size` fails the call as an
    /// unexpected response.
    #[tokio::test]
    async fn the_response_size_is_capped() {
        let body = r#"{"id":"m","type":"message","role":"assistant","model":"x","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}"#;
        let provider = |cap| {
            Provider::builder()
                .api_key("k")
                .max_response_size(cap)
                .http_transport(allama_http::ReplayTransport::new([
                    allama_http::ReplayResponse::json(200, body),
                ]))
                .build()
                .unwrap()
        };
        let request = allama::ChatRequest::new("m").user("hi");
        assert!(provider(body.len()).chat(&request).await.is_ok());
        let err = provider(body.len() - 1).chat(&request).await.unwrap_err();
        assert!(matches!(err, Error::UnexpectedResponse { .. }), "{err:?}");
        assert!(!err.is_retryable());
        assert_eq!(
            test_provider().inner.max_response_size,
            allama_http::DEFAULT_MAX_RESPONSE_SIZE
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
                .api_key("k")
                .http_transport(transport.clone());
            if let Some(timeout) = set {
                builder = builder.timeout(timeout);
            }
            let provider = builder.build().unwrap();
            let _ =
                allama::ChatProvider::chat(&provider, &allama::ChatRequest::new("m").user("hi"))
                    .await;
            assert_eq!(transport.requests()[0].timeout, expected, "{set:?}");
        }
    }

    #[test]
    fn timeouts_default_to_bounded_and_can_be_disabled() {
        let provider = test_provider();
        assert_eq!(
            provider.inner.request_timeout,
            Some(DEFAULT_REQUEST_TIMEOUT)
        );
        assert_eq!(provider.inner.stream_timeouts, StreamTimeouts::default());

        let provider = Provider::builder()
            .api_key("test-key")
            .stream_timeouts(StreamTimeouts::unbounded())
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(provider.inner.stream_timeouts, StreamTimeouts::unbounded());
    }

    #[test]
    fn the_stream_event_size_cap_defaults_to_the_decoders_and_can_be_set() {
        let provider = test_provider();
        assert_eq!(
            provider.inner.max_stream_event_size,
            SseDecoder::DEFAULT_MAX_EVENT_SIZE
        );
        let provider = Provider::builder()
            .api_key("test-key")
            .max_stream_event_size(1024)
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(provider.inner.max_stream_event_size, 1024);
    }
}
