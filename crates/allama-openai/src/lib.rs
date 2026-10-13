#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
//! OpenAI provider for `allama`: Chat Completions, Responses, and Embeddings.
//!
//! - [`Provider`] speaks Chat Completions (and Embeddings), built on the
//!   [`allama_openai_compat`] toolkit with the OpenAI dialect:
//!   `max_completion_tokens`, `developer` system prompts (see
//!   [`SystemOptions`]), strict tools and JSON schemas, `reasoning_effort`,
//!   prompt-cache keys, PDF and audio input. Chat Completions does not return
//!   reasoning content; reasoning blocks are never produced and are not
//!   replayed. Custom tools ([`ChatRequestOptions::client_tools`]) come
//!   back in the shape the Responses API uses; read and answer them with
//!   [`ClientToolCall`] on either API.
//! - [`ResponsesProvider`] speaks the Responses API statelessly
//!   (`store: false`): reasoning summaries are returned and encrypted
//!   reasoning is replayed across turns, which reasoning models need for
//!   multi-turn tool use. See the [`responses`] module for the mapping;
//!   import its types from the crate root, as the examples do.
//!
//! ```no_run
//! use allama::prelude::*;
//! use allama_openai::Provider;
//!
//! # async fn example() -> allama::Result<()> {
//! let provider = Provider::from_env()?;
//! let response = provider
//!     .chat(&ChatRequest::new("gpt-5-mini").user("Say hello.").reasoning(ReasoningEffort::Minimal))
//!     .await?;
//! println!("{}", response.text_or_empty());
//! # Ok(()) }
//! ```
//!
//! # Model rules
//!
//! Both providers send what the request asks for, whatever the model id:
//! nothing is refused, dropped or clamped because a table says a model
//! cannot take it (ADR 0015). OpenAI's rejection is the answer, as an
//! [`Error::Unsupported`](allama::Error::Unsupported) with stage `Provider` naming the refused field.
//! An application that knows a model's rules installs them with
//! [`Provider::with_model_rules`] ([`ChatModelRules`]) or
//! [`ResponsesProvider::with_model_rules`] ([`ResponsesModelRules`]), for
//! example to drop sampling while a model reasons:
//!
//! ```
//! use allama::ReasoningEffort;
//! use allama_openai::{ChatModelRules, Provider, SamplingSupport};
//!
//! # fn example(provider: Provider) {
//! let provider = provider.with_model_rules(|model: &str| {
//!     model.starts_with("gpt-5.1").then(|| {
//!         ChatModelRules::new()
//!             .sampling(SamplingSupport::WithoutReasoning)
//!             .default_effort(ReasoningEffort::None)
//!     })
//! });
//! # let _ = provider; }
//! ```
//!
//! # Per-request credentials
//!
//! A per-tenant key travels with the request as an `allama_http::Credential`
//! addressed to a credential scope ([`DEFAULT_CREDENTIAL_SCOPE`], `"openai"`,
//! unless the builder's `credential_scope` chose another). It replaces the
//! configured key for that request and reaches no other provider, even
//! through a fallback chain. A builder with a scope and no API key requires
//! one on every request.
//!
//! # Transport
//!
//! Requests go through `reqwest` (the default `reqwest` feature) unless the
//! builder is given another [`HttpTransport`](allama_http::HttpTransport)
//! with `http_transport`. With `default-features = false` and an injected
//! transport the crate builds without `reqwest`; the constructors that would
//! build a `reqwest` client (`new`, `new_with_base_url`, `from_env`) are
//! then not compiled, and `build()` without a transport is a
//! configuration error. TLS backends: `default-tls`, `native-tls`, `rustls`,
//! and `rustls-ring` (`rustls` with `ring` instead of `aws-lc-rs`, so musl
//! builds need no cmake).

use std::borrow::Cow;
use std::sync::Arc;

use allama::adapter::chat::Prepared;
use allama::{
    CapabilitySupport, ChatCapability, ChatCapabilityResolver, ChatProvider, ChatRequest,
    ChatResponse, ChatStream, ChatValidation, EmbeddingCapability, EmbeddingLimits,
    EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, EmbeddingValidation, ProviderIdentity,
    Result, ValidationCoverage,
};
use allama_http::config::redact_url;
use allama_http::{CredentialSlot, HttpRequest};
use allama_openai_compat::CompatQuirks;
use model::ModelProfile;

mod builder;
#[cfg(all(test, feature = "reqwest"))]
mod conformance_tests;
mod model;
mod options;
pub mod responses;

pub use allama_openai_compat::{
    ChatModelRules, ClientToolCall, CustomToolCall, LocalShellCall, McpApprovalRequest,
    ResponsesModelRules, SamplingSupport, ShellCall, ShellCommandOutput, ShellOutcome,
    StreamPosition, StreamTimeouts, SystemOptions, SystemRole, ToolCallSupport,
};
pub use options::{ChatRequestOptions, EmbeddingRequestOptions};
pub use responses::{ResponsesProvider, ResponsesProviderBuilder, ResponsesRequestOptions};

/// Namespace of OpenAI's request options and provider data.
pub const NAMESPACE: &str = "openai";

/// The [`provider_name`](allama::ProviderIdentity::provider_name) of
/// [`Provider`] and [`ResponsesProvider`].
pub const PROVIDER_NAME: &str = "openai";

/// The default credential scope of [`Provider`] and [`ResponsesProvider`]:
/// per-request credentials addressed to it
/// (`allama_http::HttpRequestOptions::credential`) are these providers'.
///
/// The scope names an account, not a wire format: it is independent of the
/// provider name used for telemetry. Change it with
/// [`ProviderBuilder::credential_scope`] or
/// [`ResponsesProviderBuilder::credential_scope`].
pub const DEFAULT_CREDENTIAL_SCOPE: &str = "openai";

/// The verbatim response body (Chat Completions or Responses), kept in
/// [`ChatResponse::provider_data`](allama::ChatResponse::provider_data) when
/// the request sets `raw_response` ([`ChatRequestOptions::raw_response`],
/// [`ResponsesRequestOptions::raw_response`]; unary calls only).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct RawResponse(pub serde_json::Value);

impl allama::ProviderDataEntry for RawResponse {
    const NAMESPACE: &'static str = NAMESPACE;
    const KEY: &'static str = allama::provider_data::RAW_RESPONSE_KEY;
}
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
/// OpenAI's documented limit on inputs per embeddings request.
const MAX_EMBEDDING_INPUTS: usize = 2048;

/// The OpenAI provider.
#[derive(Clone)]
pub struct Provider {
    compat: allama_openai_compat::Provider,
    resolver: Option<Arc<dyn ChatCapabilityResolver>>,
}

impl std::fmt::Debug for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Provider")
            .field("base_url", &redact_url(self.compat.base_url()))
            .finish_non_exhaustive()
    }
}

impl Provider {
    /// Create a provider with an API key.
    ///
    /// # Errors
    ///
    /// Fails when the key is empty.
    #[cfg(feature = "reqwest")]
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        Self::builder().api_key(api_key).build()
    }

    /// Create a provider for an OpenAI-compatible base URL with OpenAI
    /// semantics (e.g. a proxy).
    ///
    /// # Errors
    ///
    /// Fails when the key or URL is empty.
    #[cfg(feature = "reqwest")]
    pub fn new_with_base_url(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Result<Self> {
        Self::builder().api_key(api_key).base_url(base_url).build()
    }

    /// Create a provider from `OPENAI_API_KEY`, and optionally
    /// `OPENAI_BASE_URL`, `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`.
    ///
    /// # Errors
    ///
    /// Fails when `OPENAI_API_KEY` is not set.
    #[cfg(feature = "reqwest")]
    pub fn from_env() -> Result<Self> {
        let var = |name: &str| std::env::var(name).ok().filter(|v| !v.trim().is_empty());
        let secret = |name: &str, value: String| allama_http::config::secret(name, &value);
        let api_key = var("OPENAI_API_KEY")
            .ok_or_else(|| allama::Error::configuration("OPENAI_API_KEY not set").authored())?;
        let mut builder = Self::builder().api_key(secret("OPENAI_API_KEY", api_key)?);
        if let Some(url) = var("OPENAI_BASE_URL") {
            builder = builder.base_url(allama_http::config::base_url("OPENAI_BASE_URL", &url)?);
        }
        if let Some(org) = var("OPENAI_ORG_ID") {
            builder = builder.organization(org);
        }
        if let Some(project) = var("OPENAI_PROJECT_ID") {
            builder = builder.project(project);
        }
        builder.build()
    }

    /// Start configuring a provider.
    #[must_use]
    pub fn builder() -> ProviderBuilder {
        ProviderBuilder::default()
    }

    /// Override capability answers. The overrides reach this provider's
    /// own checks, so they can upgrade an answer as well as downgrade it.
    #[must_use]
    pub fn with_chat_capabilities(mut self, resolver: impl ChatCapabilityResolver) -> Self {
        self.resolver = Some(Arc::new(resolver));
        self
    }

    /// Set the request rules per model id: what a model rejects that the
    /// encoder should drop or refuse before sending, such as sampling while
    /// it reasons, or tools at some efforts. `None` keeps the built-in
    /// rules: the documented effort levels of the original GPT-5 models, the
    /// o-series and GPT-6 Astra, which a requested effort is mapped to, and
    /// nothing else (ADR 0015). Rules you return replace the built-in ones
    /// for that model. See [Model rules](crate#model-rules).
    #[must_use]
    pub fn with_model_rules<F>(mut self, rules: F) -> Self
    where
        F: Fn(&str) -> Option<ChatModelRules> + Send + Sync + 'static,
    {
        self.compat = self.compat.with_model_rules(move |model: &str| {
            rules(model).unwrap_or_else(|| ModelProfile::for_model(model).chat_rules())
        });
        self
    }

    /// The request rules applied to `model`.
    #[must_use]
    pub fn model_rules(&self, model: &str) -> ChatModelRules {
        self.compat.model_rules(model)
    }

    /// Use `scope` as the credential scope (see
    /// [`ProviderBuilder::credential_scope`]). A provider built without an
    /// API key then requires a credential for `scope` on every request.
    #[must_use]
    pub fn with_credential_scope(mut self, scope: impl Into<Cow<'static, str>>) -> Self {
        self.compat = self.compat.with_credential_scope(scope);
        self
    }

    /// Where this provider takes its per-request credential from, and where
    /// it puts it: `Authorization: Bearer`, under
    /// [`DEFAULT_CREDENTIAL_SCOPE`] unless the builder chose another scope.
    #[must_use]
    pub fn credential_slot(&self) -> &CredentialSlot {
        self.compat.credential_slot()
    }

    fn wire_options(request: &ChatRequest) -> Result<allama_openai_compat::WireOptions> {
        Ok(request.resolve_option::<ChatRequestOptions>()?.into())
    }

    /// Everything before sending: this crate's preflight already ran on
    /// `prepared`; the compat dialect's follows, then its encoding.
    fn encode(
        &self,
        prepared: &mut Prepared<'_>,
        options: &allama_openai_compat::WireOptions,
        stream: bool,
    ) -> Result<HttpRequest> {
        if stream {
            prepared.preflight_stream(self);
        }
        prepared.preflight_with(&self.compat)?;
        self.compat.encode_chat(prepared, options, stream)
    }

    /// A dry run of `chat` or `chat_stream`.
    fn validate(&self, request: &ChatRequest, stream: bool) -> Result<ChatValidation> {
        let options = Self::wire_options(request)?;
        let mut prepared = allama::adapter::chat::prepare_dry_run(self, request)?;
        self.encode(&mut prepared, &options, stream)?;
        Ok(prepared.into_validation(ValidationCoverage::Encoding))
    }

    fn embedding_wire_options(
        request: &EmbeddingRequest,
    ) -> Result<allama_openai_compat::EmbeddingWireOptions> {
        Ok(request.resolve_option::<EmbeddingRequestOptions>()?.into())
    }

    /// Everything before sending an embedding request: this crate's
    /// preflight already ran on `prepared`; the compat dialect's follows,
    /// then its encoding.
    fn encode_embedding(
        &self,
        prepared: &mut allama::adapter::embedding::Prepared<'_>,
        options: &allama_openai_compat::EmbeddingWireOptions,
    ) -> Result<HttpRequest> {
        prepared.preflight_with(&self.compat)?;
        self.compat.encode_embedding(prepared, options)
    }

    /// The (family, Chat Completions) answers of [`model::ModelProfile`].
    fn builtin_chat_capability(model: &str, capability: ChatCapability) -> CapabilitySupport {
        model::ModelProfile::for_model(model)
            .chat_capability(model::Endpoint::ChatCompletions, capability)
    }
}

impl ChatProvider for Provider {
    type Stream = ChatStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let options = Self::wire_options(request)?;
        let mut prepared = allama::adapter::chat::prepare(self, request)?;
        let http = self.encode(&mut prepared, &options, false)?;
        self.compat.send_chat(http, prepared, &options).await
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        let options = Self::wire_options(request)?;
        let mut prepared = allama::adapter::chat::prepare(self, request)?;
        let http = self.encode(&mut prepared, &options, true)?;
        self.compat.send_chat_stream(http, prepared).await
    }

    /// A dry run of `chat` ([`ValidationCoverage::Encoding`]).
    fn validate_chat_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        self.validate(request, false)
    }

    /// A dry run of `chat_stream` ([`ValidationCoverage::Encoding`]).
    fn validate_chat_stream_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        self.validate(request, true)
    }

    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        self.resolver
            .as_ref()
            .and_then(|r| r.chat_capability(model, capability))
            .unwrap_or_else(|| Self::builtin_chat_capability(model, capability))
    }
}

impl EmbeddingProvider for Provider {
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        let options = Self::embedding_wire_options(request)?;
        let request = honor_input_type(request);
        let mut prepared = allama::adapter::embedding::prepare(self, &request)?;
        let http = self.encode_embedding(&mut prepared, &options)?;
        self.compat.send_embedding(http, prepared).await
    }

    /// A dry run of `embed` ([`ValidationCoverage::Encoding`]).
    fn validate_embedding_request(
        &self,
        request: &EmbeddingRequest,
    ) -> Result<EmbeddingValidation> {
        let options = Self::embedding_wire_options(request)?;
        let request = honor_input_type(request);
        let mut prepared = allama::adapter::embedding::prepare_dry_run(self, &request)?;
        self.encode_embedding(&mut prepared, &options)?;
        Ok(prepared.into_validation(ValidationCoverage::Encoding))
    }

    /// Output dimensions are `Supported` for `text-embedding-3-*`; other
    /// models are `Unknown`, so `dimensions` is sent and the API decides
    /// (ADR 0015: no model is refused it). Image and
    /// composite inputs are `Unsupported`: the endpoint embeds one text per
    /// input. The input type is `Supported` for the symmetric models
    /// (`text-embedding-3-*`, `text-embedding-ada-002`), which embed queries
    /// and documents alike, so the hint is honored by sending nothing, under
    /// every policy. For other models it is `Unsupported`: the API has no
    /// query/document field, so it is dropped with a warning (an error under
    /// `UnsupportedPolicy::Error`).
    fn embedding_capability(
        &self,
        model: &str,
        capability: EmbeddingCapability,
    ) -> CapabilitySupport {
        match capability {
            EmbeddingCapability::BatchInput => CapabilitySupport::Supported,
            EmbeddingCapability::OutputDimensions if model.starts_with("text-embedding-3") => {
                CapabilitySupport::Supported
            }
            // The embeddings endpoint takes one text per input.
            EmbeddingCapability::ImageInput | EmbeddingCapability::CompositeInput => {
                CapabilitySupport::Unsupported
            }
            // No query/document field on the embeddings API: honored for
            // the symmetric models, which need none.
            EmbeddingCapability::InputType if is_symmetric_embedding_model(model) => {
                CapabilitySupport::Supported
            }
            EmbeddingCapability::InputType => CapabilitySupport::Unsupported,
            _ => CapabilitySupport::Unknown,
        }
    }

    fn embedding_limits(&self, _model: &str) -> EmbeddingLimits {
        EmbeddingLimits::new().max_inputs_per_call(MAX_EMBEDDING_INPUTS)
    }
}

impl ProviderIdentity for Provider {
    fn provider_name(&self) -> &str {
        PROVIDER_NAME
    }

    fn reads_namespace(&self, namespace: &str) -> bool {
        namespace == NAMESPACE
    }

    /// Exposes this provider only: the compat provider it is built on is an
    /// implementation detail, and calling it directly would skip OpenAI's
    /// request handling.
    fn downcast_any(&self, id: std::any::TypeId) -> Option<&(dyn std::any::Any + Send + Sync)> {
        allama::expose(self, id)
    }
}

/// Whether `model` is an OpenAI embedding model that embeds queries and
/// documents alike, so the query/document hint needs no field.
///
/// The embeddings API takes no role
/// (<https://platform.openai.com/docs/api-reference/embeddings/create>), and
/// its guide embeds search queries with the same call and model as the
/// documents they search
/// (<https://platform.openai.com/docs/guides/embeddings#use-cases>).
fn is_symmetric_embedding_model(model: &str) -> bool {
    model.starts_with("text-embedding-3-") || model == "text-embedding-ada-002"
}

/// `request` as sent: for a symmetric model the input type is honored by
/// sending nothing, so it is removed before the toolkit, which has no field
/// for it, would warn.
fn honor_input_type(request: &EmbeddingRequest) -> Cow<'_, EmbeddingRequest> {
    if request.input_type.is_some() && is_symmetric_embedding_model(&request.model) {
        let mut request = request.clone();
        request.input_type = None;
        Cow::Owned(request)
    } else {
        Cow::Borrowed(request)
    }
}

/// Builder for [`Provider`].
///
/// `Debug` shows the settings without the API key, header values, or
/// credentials in the base URL.
#[derive(Default)]
pub struct ProviderBuilder {
    settings: builder::Settings,
}

impl std::fmt::Debug for ProviderBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.settings.debug("ProviderBuilder", f)
    }
}

impl ProviderBuilder {
    builder::shared_setters!();

    /// The credential scope (default [`DEFAULT_CREDENTIAL_SCOPE`]).
    /// Per-request credentials addressed to it
    /// (`allama_http::HttpRequestOptions::credential`) are this provider's;
    /// credentials addressed to other scopes are never sent.
    ///
    /// Two OpenAI accounts in one fallback chain need two scopes: by default
    /// they share one. Providers behind one gateway share one.
    ///
    /// With a scope set, the API key is optional. Without one, bring your
    /// own key: every request must carry a credential for the scope, or it
    /// fails with an authored [`Error::Configuration`](allama::Error::Configuration) before anything is
    /// sent. With an API key, a request without a credential for the scope
    /// is sent with that key, so a mistyped scope falls back to it.
    #[must_use]
    pub fn credential_scope(mut self, scope: impl Into<Cow<'static, str>>) -> Self {
        self.settings.credential_scope = Some(scope.into());
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
        self.settings.max_stream_event_size = Some(bytes);
        self
    }

    /// Build the provider.
    ///
    /// # Errors
    ///
    /// An authored configuration error, naming the setting, when the API
    /// key is missing or empty and no credential scope is set, or holds a
    /// character no HTTP header can carry; when the base URL is not an
    /// absolute `http` or `https` URL without a query; when a
    /// [`header`](Self::header) is malformed or carries a credential; or
    /// when no transport is set and the `reqwest` feature is off.
    pub fn build(self) -> Result<Provider> {
        let builder = builder::configure!(allama_openai_compat::Provider::builder(), self.settings)
            .quirks(CompatQuirks::openai())
            .model_rules(|model: &str| ModelProfile::for_model(model).chat_rules());
        Ok(Provider {
            compat: builder.build()?,
            resolver: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use allama::Error;

    use super::*;

    /// The resolved timeout is what the transport receives with a
    /// non-streaming request.
    #[tokio::test]
    async fn the_transport_receives_the_resolved_timeout() {
        let five = Some(Duration::from_secs(5));
        for (set, expected) in [
            (None, Some(allama_http::DEFAULT_REQUEST_TIMEOUT)),
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
    fn builder_validates_and_configures() {
        assert!(Provider::builder().build().is_err());
        let provider = Provider::builder()
            .api_key("k")
            .base_url("https://example.com/v1///")
            .organization("org")
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(provider.compat.base_url(), "https://example.com/v1");
        assert_eq!(provider.compat.quirks().namespace, "openai");
        assert_eq!(provider.provider_name(), "openai");
        assert!(provider.reads_namespace(NAMESPACE));
        assert!(!provider.reads_namespace("openai-compat"));

        for builder in [
            Provider::builder().api_key("sk\nx"),
            Provider::builder()
                .api_key("k")
                .base_url("api.openai.com/v1"),
            Provider::builder()
                .api_key("k")
                .header("authorization", "Bearer x"),
        ] {
            let Err(err) = builder
                .http_transport(allama_http::ReplayTransport::default())
                .build()
            else {
                panic!("built");
            };
            assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
        }
    }

    /// `Debug` never shows the key, header values or the
    /// base URL's credentials; the key is trimmed, and the builder's headers
    /// and the default `User-Agent` reach every request.
    #[tokio::test]
    async fn debug_is_redacted_and_headers_are_sent() {
        let transport =
            allama_http::ReplayTransport::new([allama_http::ReplayResponse::new(500, "")]);
        let builder = Provider::builder()
            .api_key(" sk-SECRET\n")
            .base_url("https://user:PASSWORD@gw.example/v1")
            .header("x-route", "ROUTE-VALUE")
            .http_transport(transport.clone());
        let builder_debug = format!("{builder:?}");
        let provider = builder.build().unwrap();
        for debug in [builder_debug, format!("{provider:?}")] {
            assert!(
                !debug.contains("SECRET") && !debug.contains("PASSWORD"),
                "{debug}"
            );
            assert!(!debug.contains("ROUTE-VALUE"), "{debug}");
        }
        let _ =
            allama::ChatProvider::chat(&provider, &allama::ChatRequest::new("m").user("hi")).await;
        let sent = &transport.requests()[0];
        assert_eq!(sent.headers.get("authorization"), Some("Bearer sk-SECRET"));
        assert_eq!(sent.headers.get("x-route"), Some("ROUTE-VALUE"));
        assert_eq!(
            sent.headers.get("user-agent"),
            Some(allama_http::DEFAULT_USER_AGENT)
        );
    }

    #[test]
    fn capabilities_by_family() {
        let p = Provider::builder()
            .api_key("k")
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(
            p.chat_capability("gpt-5-mini", ChatCapability::ReasoningConfig),
            CapabilitySupport::Supported
        );
        // Whether gpt-4o reasons is not claimed: a reasoning effort is sent,
        // and OpenAI decides (ADR 0015).
        assert_eq!(
            p.chat_capability("gpt-4o", ChatCapability::ReasoningConfig),
            CapabilitySupport::Unknown
        );
        assert_eq!(p.model_rules("gpt-6-sol"), ChatModelRules::new());
        assert_eq!(
            p.chat_capability("gpt-4.1", ChatCapability::FileInput),
            CapabilitySupport::Supported
        );
        assert_eq!(
            p.chat_capability("gpt-5", ChatCapability::ReasoningOutput),
            CapabilitySupport::Unsupported
        );
        let overridden = p.with_chat_capabilities(|_: &str, c| {
            (c == ChatCapability::ReasoningOutput).then_some(CapabilitySupport::Supported)
        });
        assert_eq!(
            overridden.chat_capability("gpt-5", ChatCapability::ReasoningOutput),
            CapabilitySupport::Supported
        );
        assert_eq!(
            overridden
                .embedding_limits("text-embedding-3-small")
                .max_inputs_per_call,
            Some(2048)
        );
    }
}
