// Tests set environment variables, which is `unsafe` in edition 2024.
#![cfg_attr(not(test), forbid(unsafe_code))]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
//! Google Gemini provider adapter for `allama`.
//!
//! Implements [`allama::ChatProvider`] and [`allama::EmbeddingProvider`]
//! against the Gemini API's `generateContent`, which Google now calls
//! legacy but fully supported; its Interactions API is not supported yet.
//! Supports tools (including parallel tool calls),
//! streaming, vision, structured output, and reasoning content as first-
//! class [`allama::ContentBlock::Reasoning`] blocks. See the Capabilities
//! section of the crate README for the current table.
//!
//! Built-in tools (Google Search, URL context, code execution) are enabled
//! through [`ChatRequestOptions::builtin_tools`]. Google Search grounding comes
//! back as [`allama::Citation`]s on the answer text, with the raw
//! `groundingMetadata` in the response provider data. Code execution comes
//! back as [`allama::ProviderToolCall`] and [`allama::ProviderToolResult`]
//! blocks, replayed with their thought signatures.
//!
//! ```no_run
//! use allama::prelude::*;
//! use allama_gemini::Provider;
//!
//! # #[cfg(feature = "reqwest")]
//! # async fn example() -> allama::Result<()> {
//! let provider = Provider::from_env()?;
//! let response = provider
//!     .chat(&ChatRequest::new("gemini-3.5-flash").user("Say hello."))
//!     .await?;
//! println!("{}", response.text_or_empty());
//! # Ok(()) }
//! ```
//!
//! # Transport
//!
//! Requests go through `reqwest` (the default `reqwest` feature) unless the
//! builder is given another [`HttpTransport`]
//! with `http_transport`. With `default-features = false` and an injected
//! transport the crate builds without `reqwest`; the constructors that would
//! build a `reqwest` client (`new`, `new_with_base_url`, `from_env`) are
//! then not compiled, and `build()` without a transport is a
//! configuration error. TLS backends: `default-tls`, `native-tls`, `rustls`,
//! and `rustls-ring` (`rustls` with `ring` instead of `aws-lc-rs`, so musl
//! builds need no cmake).
//!
//! # Per-request credentials
//!
//! A [`allama_http::Credential::api_key`] addressed to the provider's
//! credential scope ([`DEFAULT_CREDENTIAL_SCOPE`] unless changed with
//! [`ProviderBuilder::credential_scope`]) replaces the configured key in
//! `x-goog-api-key` for that request, on chat and embedding calls alike;
//! credentials addressed to other scopes are never sent. See
//! [`allama_http::HttpRequestOptions`].

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Duration;

use allama::{CapabilitySupport, ChatCapability, ChatCapabilityResolver, Error, Result};
#[cfg(feature = "reqwest")]
use allama_http::ReqwestTransport;
pub use allama_http::StreamTimeouts;
use allama_http::config::redact_url;
use allama_http::sse::SseDecoder;
use allama_http::{
    CredentialPlacement, CredentialSlot, DEFAULT_REQUEST_TIMEOUT, Headers, HttpTransport,
};
pub use decode::blocked_prompt_usage;

mod chat;
#[cfg(test)]
mod conformance_tests;
mod decode;
mod embedding;
mod error;
mod model;
mod options;
mod request;
mod schema;
mod streaming;
mod wire;

pub use options::{ChatRequestOptions, EmbeddingRequestOptions, EmbeddingTitle, SafetySetting};

/// The verbatim `generateContent` response body, kept in
/// [`ChatResponse::provider_data`](allama::ChatResponse::provider_data) when
/// [`ChatRequestOptions::raw_response`] is set (unary calls only).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct RawResponse(pub serde_json::Value);

impl allama::ProviderDataEntry for RawResponse {
    const NAMESPACE: &'static str = NAMESPACE;
    const KEY: &'static str = allama::provider_data::RAW_RESPONSE_KEY;
}

/// Provider-data namespace (and origin key) of the Gemini API: `"google-ai"`.
///
/// Thought signatures are stored on content blocks under this namespace as
/// `"thought_signature"`; Gemini thought (reasoning) blocks carry
/// `"thought": true`. Only data in this namespace is replayed to Gemini.
pub const NAMESPACE: &str = "google-ai";

/// Provider identity reported by [`allama::ProviderIdentity`] and stamped on
/// responses and errors.
pub const PROVIDER_NAME: &str = "gemini";

/// The default credential scope of [`Provider`]: its provider name, not
/// its [`NAMESPACE`].
///
/// Per-request credentials addressed to it
/// ([`allama_http::HttpRequestOptions::credential`]) are this provider's:
/// a [`allama_http::Credential::api_key`] is sent in `x-goog-api-key`,
/// replacing the configured key. The scope names an account, independent
/// of the provider name (telemetry) and the namespaces. Change it with
/// [`ProviderBuilder::credential_scope`].
pub const DEFAULT_CREDENTIAL_SCOPE: &str = "gemini";

/// The header the Gemini API reads its key from.
pub(crate) const API_KEY_HEADER: &str = "x-goog-api-key";

/// Google Gemini API provider implementing `allama::ChatProvider`.
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
}

/// The credential slot for `scope`: `x-goog-api-key`, required when no key
/// is configured and the scope was chosen explicitly.
fn credential_slot(scope: Cow<'static, str>, has_key: bool, explicit: bool) -> CredentialSlot {
    CredentialSlot::new(scope, CredentialPlacement::Header(API_KEY_HEADER.into()))
        .required(!has_key && explicit)
}

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

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
    /// and `x-goog-api-key` as the placement.
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
        };
        change(&mut inner);
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Create from environment variables.
    ///
    /// Required: `GEMINI_API_KEY`, or `GOOGLE_API_KEY` when it is unset
    /// (the variable Google's SDKs also read)
    /// Optional: `GEMINI_BASE_URL`
    ///
    /// # Errors
    ///
    /// An authored [`Error::Configuration`] naming both key variables when
    /// both are unset or blank, or naming the variable that holds a value
    /// [`ProviderBuilder::build`] refuses (a key no HTTP header can carry, a
    /// base URL that is not an absolute `http(s)` URL).
    #[cfg(feature = "reqwest")]
    pub fn from_env() -> Result<Self> {
        let (name, api_key) = ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
            .into_iter()
            .find_map(|name| env_var(name).map(|value| (name, value)))
            .ok_or_else(|| {
                Error::configuration("GEMINI_API_KEY or GOOGLE_API_KEY not set").authored()
            })?;
        let mut builder = Self::builder().api_key(allama_http::config::secret(name, &api_key)?);
        if let Some(url) = env_var("GEMINI_BASE_URL") {
            builder = builder.base_url(allama_http::config::base_url("GEMINI_BASE_URL", &url)?);
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

/// Builder for configuring a Gemini provider.
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
    /// The configured API key, sent in `x-goog-api-key` unless a request
    /// carries a credential for the provider's credential scope. Surrounding
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
    /// The scope names an account: two Gemini accounts in one fallback chain
    /// need two scopes (the default makes them share one), while providers
    /// behind one gateway share one.
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

    /// Base URL (default
    /// `https://generativelanguage.googleapis.com/v1beta`), for a proxy or
    /// gateway that serves the same API. Surrounding whitespace and trailing
    /// slashes are removed; requests go to
    /// `{base_url}/models/{model}:generateContent` (or
    /// `{base_url}/tunedModels/{id}:…` for a `tunedModels/` model) and the
    /// other model methods. It must be an absolute `http` or `https` URL
    /// without a query ([`allama_http::config::base_url`]). A model id that
    /// is not a model name (`<id>`, `models/<id>` or `tunedModels/<id>`, with
    /// `<id>` of letters, digits, `.`, `_` and `-`) is refused with
    /// [`Error::InvalidRequest`] before
    /// anything is sent, so an id from an end user cannot reach another API
    /// path.
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
    /// let provider = allama_gemini::Provider::builder()
    ///     .api_key("AIza...")
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
    /// routing header, a tenant tag, `x-goog-user-project`. `header("user-agent",
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
            }),
        })
    }
}

/// Built-in capability answers (before any installed resolver): what the
/// endpoint carries for every model, and the positive answers the model
/// pages document. Nothing is `Unsupported` because of the model (ADR 0015):
/// a model that lacks a feature gets it sent, and the API's refusal maps to
/// [`Error::Unsupported`] at the `Provider` stage where its wording is known.
///
/// - **Endpoint facts**, for every model: streaming is native and reports
///   usage. Image detail is a part's `mediaResolution`, which Gemini 3
///   honors (verified live 2026-10-07: the image token count follows the
///   level), so `ImageDetail` is `Supported` there and `Unknown` elsewhere,
///   where it is sent and the API decides.
/// - **Image variants** (classified before the generation) generate
///   images (`ImageOutput`), and think from Gemini 3 on
///   (<https://ai.google.dev/gemini-api/docs/models>); the rest is
///   `Unknown`. Specialty variants (TTS, Live, `computer-use`, `robotics`,
///   `embedding`, `imagen`) are `Unknown`: they do not inherit the chat
///   defaults of their generation.
/// - **Gemini generations** (1.x to 3.x): function calling (several calls
///   per turn, with no flag to disable them), structured output, image and
///   file input, image replay, caching and forced function calling;
///   reasoning from 2.5 on. Anything else is `Unknown`.
/// - **Gemma 4**: function calling, image input and thinking
///   (<https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api>); the
///   rest is `Unknown`.
/// - **Anything else** (`ModelFamily::Other`: tuned or experimental
///   models, earlier Gemma generations) is `Unknown`.
pub(crate) fn builtin_chat_capability(
    model: &str,
    capability: ChatCapability,
) -> CapabilitySupport {
    use CapabilitySupport::{Supported, Unknown};
    use model::{ModelFamily, Variant};
    let family = ModelFamily::of(model);
    match capability {
        ChatCapability::Streaming | ChatCapability::NativeStreaming => return Supported,
        // Streamed `GenerateContentResponse` chunks carry `usageMetadata`:
        // https://ai.google.dev/api/generate-content#v1beta.GenerateContentResponse
        // (captured in `fixtures/stream.sse`).
        ChatCapability::StreamUsage => return Supported,
        ChatCapability::ImageDetail => {
            return match (Variant::of(model), family) {
                (Variant::Text, ModelFamily::Gemini3 { .. }) => Supported,
                _ => Unknown,
            };
        }
        _ => {}
    }
    match Variant::of(model) {
        Variant::Text => {}
        Variant::Specialty => return Unknown,
        Variant::Image => {
            return match capability {
                ChatCapability::ImageOutput => Supported,
                ChatCapability::ReasoningOutput
                | ChatCapability::ReasoningReplay
                | ChatCapability::ReasoningConfig
                    if matches!(family, ModelFamily::Gemini3 { .. }) =>
                {
                    Supported
                }
                _ => Unknown,
            };
        }
    }
    match family {
        ModelFamily::Other => Unknown,
        ModelFamily::Gemma4 => match capability {
            ChatCapability::ToolCalls
            | ChatCapability::ImageInput
            | ChatCapability::ReasoningConfig => Supported,
            _ => Unknown,
        },
        ModelFamily::Legacy | ModelFamily::Gemini25 { .. } | ModelFamily::Gemini3 { .. } => {
            match capability {
                ChatCapability::ToolCalls
                | ChatCapability::ParallelToolCalls
                | ChatCapability::ImageInput
                | ChatCapability::ImageReplay
                | ChatCapability::FileInput
                | ChatCapability::AudioInput
                | ChatCapability::VideoInput
                | ChatCapability::StructuredOutput
                | ChatCapability::PromptCache
                | ChatCapability::ForcedToolChoice => Supported,
                ChatCapability::ReasoningOutput
                | ChatCapability::ReasoningReplay
                | ChatCapability::ReasoningConfig
                    if family.documents_thinking() =>
                {
                    Supported
                }
                _ => Unknown,
            }
        }
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
        streaming::sse_to_stream(events, Vec::new(), None),
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

    /// A provider over an empty replay transport, so tests run without
    /// `reqwest`.
    pub(crate) fn test_provider() -> Provider {
        Provider::builder()
            .api_key("test-key")
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap()
    }

    /// Gemini reports `gemini` but reads its state under `google-ai`.
    #[test]
    fn reads_its_namespace_not_its_name() {
        use allama::ProviderIdentity;

        let provider = test_provider();
        assert!(provider.reads_namespace(NAMESPACE));
        assert!(!provider.reads_namespace(PROVIDER_NAME));
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
            matches!(&err, Error::Configuration { .. } if err.message() == "`api_key` is empty")
        );
    }

    #[test]
    fn builder_rejects_empty_base_url() {
        let err = match Provider::builder()
            .api_key("test-key")
            .base_url("   ")
            .http_transport(allama_http::ReplayTransport::default())
            .build()
        {
            Ok(_) => panic!("expected empty base URL to be rejected"),
            Err(err) => err,
        };
        assert!(
            matches!(&err, Error::Configuration { .. } if err.message() == "`base_url` is empty")
        );
    }

    #[cfg(feature = "reqwest")]
    #[test]
    fn from_env_rejects_empty_api_key() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _api_key = EnvVarGuard::set("GEMINI_API_KEY", Some("   "));
        let _google_api_key = EnvVarGuard::set("GOOGLE_API_KEY", None);
        let _base_url = EnvVarGuard::set("GEMINI_BASE_URL", None);

        let err = match Provider::from_env() {
            Ok(_) => panic!("expected empty API key to be rejected"),
            Err(err) => err,
        };
        assert!(matches!(&err, Error::Configuration { .. }), "{err:?}");
        // Formatted like the other adapters' missing-key errors: no
        // provider prefix, and every variable that would do.
        assert_eq!(
            err.to_string(),
            "configuration error: GEMINI_API_KEY or GOOGLE_API_KEY not set"
        );
    }

    #[test]
    fn chat_capability_resolver_overrides_builtin_answer() {
        let provider = test_provider().with_chat_capabilities(|model: &str, capability| {
            if model == "gemini-2.5-pro" && capability == ChatCapability::ReasoningOutput {
                Some(CapabilitySupport::Unknown)
            } else {
                None
            }
        });

        assert_eq!(
            provider.chat_capability("gemini-2.5-pro", ChatCapability::ReasoningOutput),
            CapabilitySupport::Unknown
        );
        assert_eq!(
            provider.chat_capability("gemini-2.5-flash", ChatCapability::ReasoningOutput),
            CapabilitySupport::Supported
        );
    }

    #[test]
    fn builtin_gemini_capabilities_report_native_streaming() {
        let provider = test_provider();
        assert_eq!(
            provider.chat_capability("gemini-2.0-flash", ChatCapability::NativeStreaming),
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
            provider.chat_capability("gemini-2.5-pro", ChatCapability::ToolCalls),
            CapabilitySupport::Supported
        );
    }

    /// The positive answers, against the documentation:
    ///
    /// - model pages (<https://ai.google.dev/gemini-api/docs/models/gemini-3-pro-image>,
    ///   `gemini-3.1-flash-image`): image models generate images, and think
    ///   from Gemini 3 on;
    /// - Gemma 4 (<https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api>):
    ///   "You can pass a system instruction", "Define tools as function
    ///   declarations", "Gemma 4 models can process images", thinking via
    ///   the thinking level; no structured output is documented;
    /// - earlier Gemma generations are not listed on that page, so with no
    ///   fixture showing them served they fall to `Unknown`, like any
    ///   unclassified model.
    ///
    /// What a model page says a model lacks is not answered `Unsupported`
    /// (ADR 0015): it is model data, recorded with its evidence in
    /// `docs/catalog/gemini.md`, and the request is sent.
    #[test]
    fn capability_tables_follow_the_model_pages() {
        use CapabilitySupport::{Supported, Unknown};
        let answer = builtin_chat_capability;
        for model in [
            "gemini-2.5-flash-image",
            "gemini-3-pro-image",
            "gemini-2.5-flash-preview-tts",
            "gemini-2.5-pro-preview-tts",
            "gemini-2.5-flash-native-audio-preview-12-2025",
        ] {
            for capability in [
                ChatCapability::ToolCalls,
                ChatCapability::ParallelToolCalls,
                ChatCapability::ForcedToolChoice,
                ChatCapability::StructuredOutput,
                ChatCapability::PromptCache,
            ] {
                assert_eq!(answer(model, capability), Unknown, "{model} {capability:?}");
            }
        }
        for model in ["gemini-2.5-flash-image", "gemini-2.5-pro-preview-tts"] {
            assert_eq!(answer(model, ChatCapability::ReasoningConfig), Unknown);
        }
        assert_eq!(
            answer("gemini-3.1-flash-image", ChatCapability::ReasoningConfig),
            Supported
        );
        assert_eq!(
            answer("gemini-3-pro-image", ChatCapability::ImageOutput),
            Supported
        );

        for capability in [
            ChatCapability::ToolCalls,
            ChatCapability::ImageInput,
            ChatCapability::ReasoningConfig,
        ] {
            assert_eq!(answer("gemma-4-31b-it", capability), Supported);
        }
        assert_eq!(
            answer("gemma-4-31b-it", ChatCapability::StructuredOutput),
            Unknown
        );

        for model in [
            "gemma-3-27b-it",
            "tunedModels/my-model",
            "gemini-exp-1206",
            "gemini-2.5-computer-use-preview-10-2025",
        ] {
            for capability in [
                ChatCapability::ToolCalls,
                ChatCapability::StructuredOutput,
                ChatCapability::ImageInput,
                ChatCapability::ForcedToolChoice,
                ChatCapability::ReasoningConfig,
            ] {
                assert_eq!(answer(model, capability), Unknown, "{model} {capability:?}");
            }
            assert_eq!(answer(model, ChatCapability::Streaming), Supported);
        }
    }

    /// Only endpoint facts are `Unsupported`: no model id, known or not,
    /// gets a refusal or a drop from the built-in table (ADR 0015).
    #[test]
    fn only_endpoint_facts_are_unsupported() {
        for model in [
            "gemini-1.5-pro",
            "gemini-2.0-flash",
            "gemini-2.0-flash-thinking-exp",
            "gemini-2.5-flash",
            "gemini-2.5-flash-image",
            "gemini-2.5-flash-preview-tts",
            "gemini-3.1-flash-live-preview",
            "gemini-3.5-flash",
            "gemma-4-31b-it",
            "tunedModels/my-model",
        ] {
            for capability in ChatCapability::ALL {
                let answer = builtin_chat_capability(model, *capability);
                assert_ne!(
                    answer,
                    CapabilitySupport::Unsupported,
                    "{model} {capability:?}"
                );
            }
        }
        // Text models are not said to lack image output, and Gemini 1.x and
        // 2.0 are not said to lack thinking.
        assert_eq!(
            builtin_chat_capability("gemini-2.5-flash", ChatCapability::ImageOutput),
            CapabilitySupport::Unknown
        );
        assert_eq!(
            builtin_chat_capability("gemini-2.0-flash", ChatCapability::ReasoningConfig),
            CapabilitySupport::Unknown
        );
    }

    #[test]
    fn builtin_capabilities_report_parallel_tools_and_image_replay() {
        let provider = test_provider();

        assert_eq!(
            provider.chat_capability("gemini-2.5-pro", ChatCapability::ParallelToolCalls),
            CapabilitySupport::Supported
        );
        assert_eq!(
            provider.chat_capability("gemini-2.5-pro", ChatCapability::ImageReplay),
            CapabilitySupport::Supported
        );
        assert_eq!(
            provider.chat_capability("gemini-2.5-pro", ChatCapability::StructuredOutput),
            CapabilitySupport::Supported
        );
    }

    #[cfg(feature = "reqwest")]
    #[test]
    fn base_url_is_normalized_when_constructed() {
        let provider = Provider::new_with_base_url(
            "test-key",
            "https://generativelanguage.googleapis.com/v1beta///",
        )
        .unwrap();
        assert_eq!(
            provider.inner.base_url,
            "https://generativelanguage.googleapis.com/v1beta"
        );
    }

    #[cfg(feature = "reqwest")]
    #[test]
    fn new_rejects_empty_api_key() {
        let err = match Provider::new("   ") {
            Ok(_) => panic!("expected empty API key to be rejected"),
            Err(err) => err,
        };
        assert!(
            matches!(&err, Error::Configuration { .. } if err.message() == "`api_key` is empty")
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
            matches!(&err, Error::Configuration { .. } if err.message() == "`base_url` is empty")
        );
    }

    #[test]
    fn builder_normalizes_base_url() {
        let provider = Provider::builder()
            .api_key("test-key")
            .base_url("https://example.com/v1beta///")
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();

        assert_eq!(provider.inner.base_url, "https://example.com/v1beta");
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
    }

    /// `GOOGLE_API_KEY` is read when `GEMINI_API_KEY` is unset,
    /// and keys are trimmed.
    #[cfg(feature = "reqwest")]
    #[test]
    fn from_env_falls_back_to_google_api_key() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _gemini = EnvVarGuard::set("GEMINI_API_KEY", None);
        let _google = EnvVarGuard::set("GOOGLE_API_KEY", Some("AIza-google\n"));
        let _base_url = EnvVarGuard::set("GEMINI_BASE_URL", None);
        assert_eq!(
            Provider::from_env().unwrap().inner.api_key.as_deref(),
            Some("AIza-google")
        );
        let _gemini = EnvVarGuard::set("GEMINI_API_KEY", Some(" AIza-gemini "));
        assert_eq!(
            Provider::from_env().unwrap().inner.api_key.as_deref(),
            Some("AIza-gemini")
        );
    }

    /// `Debug` never shows the key or the base URL's
    /// credentials, and the builder's headers reach every request.
    #[tokio::test]
    async fn debug_is_redacted_and_headers_are_sent() {
        let transport =
            allama_http::ReplayTransport::new([allama_http::ReplayResponse::new(500, "")]);
        let builder = Provider::builder()
            .api_key("AIza-SECRET")
            .base_url("https://user:PASSWORD@gw.example/v1beta")
            .header("x-goog-user-project", "proj")
            .http_transport(transport.clone());
        let builder_debug = format!("{builder:?}");
        let provider = builder.build().unwrap();
        for debug in [builder_debug, format!("{provider:?}")] {
            assert!(
                !debug.contains("SECRET") && !debug.contains("PASSWORD"),
                "{debug}"
            );
        }
        let _ = provider
            .chat(&allama::ChatRequest::new("gemini-2.5-flash").user("hi"))
            .await;
        let sent = &transport.requests()[0];
        assert_eq!(sent.headers.get("x-goog-user-project"), Some("proj"));
        assert_eq!(
            sent.headers.get("user-agent"),
            Some(allama_http::DEFAULT_USER_AGENT)
        );
        let Err(err) = Provider::builder()
            .api_key("k")
            .header("x-goog-api-key", "other")
            .http_transport(allama_http::ReplayTransport::default())
            .build()
        else {
            panic!("built with a credential header");
        };
        assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    }

    /// A success body past `max_response_size` fails the call as an
    /// unexpected response, chat and embeddings alike.
    #[tokio::test]
    async fn the_response_size_is_capped() {
        use allama::EmbeddingProvider;

        let chat = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],"modelVersion":"gemini-2.5-flash"}"#;
        let embedding = r#"{"embeddings":[{"values":[0.1,0.2]}]}"#;
        let provider = |body: &'static str, cap: usize| {
            Provider::builder()
                .api_key("k")
                .max_response_size(cap)
                .http_transport(allama_http::ReplayTransport::new([
                    allama_http::ReplayResponse::json(200, body),
                ]))
                .build()
                .unwrap()
        };
        let request = allama::ChatRequest::new("gemini-2.5-flash").user("hi");
        assert!(provider(chat, chat.len()).chat(&request).await.is_ok());
        let err = provider(chat, chat.len() - 1)
            .chat(&request)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::UnexpectedResponse { .. }), "{err:?}");
        assert!(!err.is_retryable());

        let request = allama::EmbeddingRequest::new("gemini-embedding-001").input("hi");
        assert!(
            provider(embedding, embedding.len())
                .embed(&request)
                .await
                .is_ok()
        );
        let err = provider(embedding, embedding.len() - 1)
            .embed(&request)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::UnexpectedResponse { .. }), "{err:?}");
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
