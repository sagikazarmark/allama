#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
//! A decision provider for the System One protocol (`POST /v1/systemone`).
//!
//! System One is the decision-model API defined by TypeSafe (model Jev) and
//! cloned by Ollama for local models (Nimble, Tev1). Gateways such as
//! OpenRouter, DigitalOcean, and LiteLLM speak it too, and Cloudflare
//! Workers AI serves its Clef models with it. [`Provider`] implements
//! [`allama::DecisionProvider`] for any of them; [`providers`] has presets.
//!
//! ```no_run
//! use allama::decision::Question;
//! use allama::{DecisionProvider, DecisionRequest};
//! use allama_systemone::providers::Ollama;
//!
//! # async fn example() -> allama::Result<()> {
//! let provider = Ollama::from_env()?;
//! let request = DecisionRequest::new("nimble", "Our checkout has returned 500 errors since 9am.")
//!     .question(
//!         "label",
//!         Question::choice("Which label fits this ticket?")
//!             .option("billing", "Payments and refunds")
//!             .option("bug", "Software errors")
//!             .option("account", "Login and account access"),
//!     );
//! let response = provider.decide(&request).await?;
//! println!("{}", response.choice("label").unwrap().chosen);
//! # Ok(()) }
//! ```
//!
//! # Mapping
//!
//! - Yes/no questions are System One `noul` questions; their probability is
//!   the `noul` field.
//! - Option descriptions become `criteria`; a score level becomes the
//!   string `label: description` (or its label alone), the convention of the
//!   protocol's own examples.
//! - OpenRouter's response `id` becomes [`DecisionResponse::id`]; its
//!   `provider` and `usage.cost` are kept in provider data as [`Gateway`].
//! - Each provider's `confidence` statistic is kept in provider data as
//!   [`Confidence`]: TypeSafe and Ollama compute it differently.
//! - [`DecisionRequest::images`](allama::DecisionRequest::images) go to the
//!   endpoints that take them, inline only: Ollama as bare base64, and
//!   Cloudflare as `{"content_type", "base64"}`. TypeSafe's protocol has no
//!   images, so its endpoints refuse them before sending.
//! - Ollama's `keep_alive` is a [`DecisionRequestOptions`] field.
//! - Cloudflare's own differences (a URL per model, a model selector, its
//!   envelopes) are the [`Cloudflare`](providers::Cloudflare) preset's.
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
//!
//! # Per-request credentials
//!
//! A [`allama_http::Credential::api_key`] addressed to the provider's
//! credential scope (its provider name unless changed with
//! [`ProviderBuilder::credential_scope`]) replaces the configured
//! credential for that request, placed where the configured one goes
//! (`Authorization: Bearer` by default, or the `auth_header` name);
//! credentials addressed to other scopes are never sent. See
//! [`allama_http::HttpRequestOptions`].

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use allama::{
    CapabilitySupport, DecisionCapability, DecisionLimits, DecisionProvider, DecisionRequest,
    DecisionResponse, DecisionValidation, Error, ProviderIdentity, Result, Subject,
    ValidationCoverage, Warning, WarningKind,
};
use allama_http::config::redact_url;
use allama_http::http::header::CONTENT_TYPE;
use allama_http::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use allama_http::{
    CredentialPlacement, CredentialSlot, DEFAULT_REQUEST_TIMEOUT, HttpRequest, HttpResponse,
    HttpTransport, request_uri,
};
use serde::{Deserialize, Serialize};

#[doc(hidden)]
pub mod binding;
mod error;
mod options;
pub mod providers;
mod wire;

pub use error::map_http_error;
pub use options::{DecisionRequestOptions, KeepAlive};

/// Namespace for System One options and provider data, shared by every
/// endpoint that speaks the System One protocol.
pub const NAMESPACE: &str = "systemone";

/// The default [`provider_name`](allama::ProviderIdentity::provider_name) of
/// [`Provider`]. Presets in [`providers`] report their own.
pub const PROVIDER_NAME: &str = "systemone";

/// The default credential scope of [`Provider`]: its default provider name.
/// A provider with another [`provider_name`](ProviderBuilder::provider_name),
/// such as a preset, defaults to that name instead.
///
/// Per-request credentials addressed to it
/// ([`allama_http::HttpRequestOptions::credential`]) are this provider's.
/// The scope names an account, independent of the provider name
/// (telemetry) and the namespaces. Change it with
/// [`ProviderBuilder::credential_scope`].
pub const DEFAULT_CREDENTIAL_SCOPE: &str = PROVIDER_NAME;

/// A System One decision provider.
#[derive(Clone)]
pub struct Provider {
    inner: Arc<Inner>,
}

#[derive(Clone)]
struct Inner {
    http: Arc<dyn HttpTransport>,
    transport: TransportConfig,
    credential_slot: CredentialSlot,
    capabilities: HashMap<DecisionCapability, CapabilitySupport>,
    limits: DecisionLimits,
    provider_name: Cow<'static, str>,
    missing_endpoint_hint: Option<&'static str>,
    dialect: wire::Dialect,
}

impl std::fmt::Debug for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Provider")
            .field("provider_name", &self.inner.provider_name)
            .field("url", &redact_url(&self.inner.transport.url()))
            .finish_non_exhaustive()
    }
}

impl Provider {
    /// Start configuring a provider.
    #[must_use]
    pub fn builder() -> ProviderBuilder {
        ProviderBuilder::default()
    }

    /// The endpoint URL decisions are posted to. It may carry a user name
    /// and password: show it with [`allama_http::config::redact_url`].
    ///
    /// The [`Cloudflare`](providers::Cloudflare) preset posts to each
    /// model's own URL under it: `{url}/{model}`.
    #[must_use]
    pub fn url(&self) -> String {
        self.inner.transport.url()
    }

    /// The transport configuration.
    #[cfg(test)]
    pub(crate) fn transport(&self) -> &TransportConfig {
        &self.inner.transport
    }

    /// The credential slot this provider passes to
    /// [`allama_http::HttpRequest::apply_options`]: its credential scope,
    /// where its credential goes, and the configured header it replaces.
    #[must_use]
    pub fn credential_slot(&self) -> &CredentialSlot {
        &self.inner.credential_slot
    }

    /// The same provider with its credential scope set to `scope` (see
    /// [`ProviderBuilder::credential_scope`]).
    ///
    /// A provider without a configured credential then requires one for
    /// `scope` on every request.
    #[must_use]
    pub fn with_credential_scope(self, scope: impl Into<Cow<'static, str>>) -> Self {
        let mut inner = (*self.inner).clone();
        inner.credential_slot.scope = scope.into();
        inner.credential_slot.required = inner.transport.auth.is_none();
        Self {
            inner: Arc::new(inner),
        }
    }

    /// POST a JSON body with the configured headers and the caller's
    /// transport `options`; map transport failures and error statuses.
    /// Everything before sending, shared by `decide` and its dry run: the
    /// options, the body, and the HTTP request with the caller's transport
    /// options applied.
    ///
    /// `keep_alive` is Ollama's: another endpoint is not sent it, and since
    /// it is a hint, it is dropped with a warning. Images go to the
    /// endpoints that take them (Ollama, Cloudflare), each in its own form;
    /// they change the answer, so another endpoint refuses them.
    fn encode(
        &self,
        prepared: &mut allama::adapter::decision::Prepared<'_>,
    ) -> Result<(HttpRequest, DecisionRequestOptions)> {
        let (request, d) = prepared.parts_mut();
        let provider = self.inner.provider_name.as_ref();
        // Options under the provider name (`typesafe`, `ollama`) are not
        // read: a configuration file that keys them so would otherwise be
        // ignored silently.
        if provider != NAMESPACE && request.options.json(provider).is_some() {
            d.warn(
                Warning::new(
                    WarningKind::Dropped,
                    Subject::Option {
                        namespace: provider.to_owned(),
                    },
                    format!(
                        "this provider reads options under `{NAMESPACE}`, not under its name `{provider}`; they were dropped; pass them under `{NAMESPACE}`, or as `DecisionRequestOptions`"
                    ),
                )
                .with_code("dropped.provider_name_options"),
            );
        }
        let mut options: DecisionRequestOptions = request.resolve_option()?;
        let dialect = self.inner.dialect;
        if dialect != wire::Dialect::Ollama && options.keep_alive.take().is_some() {
            d.hint(
                Subject::Other {
                    name: "`keep_alive`".into(),
                },
                "`keep_alive` is an Ollama option this endpoint does not take; it was not sent",
            )?;
        }
        let body = serde_json::to_string(&wire::encode(request, &options, dialect, d)?)?;
        let config = &self.inner.transport;
        let url = match dialect {
            wire::Dialect::WorkersAi => format!("{}/{}", config.url(), model_path(&request.model)?),
            wire::Dialect::SystemOne | wire::Dialect::Ollama => config.url(),
        };
        let uri = request_uri(url).map_err(|err| err.with_provider(self.provider_name()))?;
        let mut http = HttpRequest::post(uri, body)
            .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
            .timeout(config.timeout);
        if let Some((name, value)) = &config.auth {
            http = http.header(name.clone(), value.clone());
        }
        let http = http
            .with_headers(&config.headers)
            .apply_options(&request.options, &self.inner.credential_slot)
            .map_err(|err| err.with_provider(self.provider_name()))?;
        Ok((http, options))
    }

    async fn send(&self, request: HttpRequest) -> Result<HttpResponse> {
        let config = &self.inner.transport;
        let response = self
            .inner
            .http
            .send(request)
            .await
            .map_err(|err| err.into_error().with_provider(self.provider_name()))?;
        if response.is_success() {
            return Ok(response);
        }
        let error = response
            .map_error(|status, headers, body| {
                // The configured request id header takes precedence.
                error::map_http_error_with_hint(
                    status,
                    headers,
                    body,
                    self.inner.missing_endpoint_hint,
                )
                .with_request_id_opt(request_id(headers, &config.request_id_header_name).as_deref())
            })
            .await;
        Err(error.with_provider(self.provider_name()))
    }
}

impl ProviderIdentity for Provider {
    fn provider_name(&self) -> &str {
        &self.inner.provider_name
    }

    fn reads_namespace(&self, namespace: &str) -> bool {
        namespace == NAMESPACE
    }

    fn downcast_any(&self, id: std::any::TypeId) -> Option<&(dyn std::any::Any + Send + Sync)> {
        allama::expose(self, id)
    }
}

impl DecisionProvider for Provider {
    async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse> {
        let mut prepared = allama::adapter::decision::prepare(self, request)?;
        let (http, options) = self.encode(&mut prepared)?;
        prepared.mark_sent();
        let response = self.send(http).await?;
        let status = response.status;
        let headers = response.headers.clone();
        let request_id = request_id(
            &response.headers,
            &self.inner.transport.request_id_header_name,
        );
        let with_context = |err: Error| {
            err.with_provider(self.provider_name())
                .with_request_id_opt(request_id.as_deref())
        };
        let bytes = response
            .bytes_limited(self.inner.transport.max_response_size)
            .await
            .map_err(|err| with_context(err.into_error()))?;
        let body = match self.inner.dialect {
            // The Cloudflare API wraps the model's output.
            wire::Dialect::WorkersAi => serde_json::from_slice::<wire::CloudflareEnvelope>(&bytes)
                .and_then(|envelope| serde_json::from_str(envelope.result.get())),
            wire::Dialect::SystemOne | wire::Dialect::Ollama => {
                serde_json::from_slice::<wire::WireResponse>(&bytes)
            }
        };
        let body: wire::WireResponse = body.map_err(|err| {
            let text = String::from_utf8_lossy(&bytes);
            // Gateways sometimes report errors with a success status. A code
            // that is an HTTP status classifies the error; otherwise it is a
            // provider failure, which is worth retrying.
            let mapped = match error::error_in_body(&text) {
                Some(envelope) => match envelope
                    .code
                    .as_deref()
                    .and_then(|code| code.parse::<u16>().ok())
                    .filter(|code| (400..=599).contains(code))
                    .and_then(|code| StatusCode::from_u16(code).ok())
                {
                    // The response's headers carry its retry hint.
                    Some(code) => error::map_http_error_with_hint(
                        code,
                        &headers,
                        &text,
                        self.inner.missing_endpoint_hint,
                    ),
                    None => envelope
                        .apply(
                            Error::provider(format!(
                                "error envelope with HTTP {}",
                                status.as_u16()
                            ))
                            .authored(),
                        )
                        .with_body(text.into_owned()),
                },
                None => allama_http::malformed_response(status, &bytes, err),
            };
            with_context(mapped)
        })?;
        let wire::Decoded {
            mut response,
            confidence,
            gateway,
        } = wire::decode(body).map_err(with_context)?;
        response.validate(request).map_err(with_context)?;
        response.request_id.clone_from(&request_id);
        response.provider = Some(self.inner.provider_name.clone());
        let attach = || -> Result<()> {
            if !confidence.is_empty() {
                response.provider_data.insert(&Confidence(confidence))?;
            }
            if let Some(gateway) = &gateway {
                response.provider_data.insert(gateway)?;
            }
            if options.raw_response {
                let raw: serde_json::Value = serde_json::from_slice(&bytes)
                    .map_err(|err| allama_http::malformed_response(status, &bytes, err))?;
                response.provider_data.insert(&RawResponse(raw))?;
            }
            Ok(())
        };
        attach().map_err(with_context)?;
        Ok(prepared.finish(response))
    }

    /// A dry run of [`decide`](Self::decide): the same checks, options and
    /// encoding, without sending ([`ValidationCoverage::Encoding`]).
    fn validate_decision_request(&self, request: &DecisionRequest) -> Result<DecisionValidation> {
        let mut prepared = allama::adapter::decision::prepare_dry_run(self, request)?;
        self.encode(&mut prepared)?;
        Ok(prepared.into_validation(ValidationCoverage::Encoding))
    }

    fn decision_capability(
        &self,
        _model: &str,
        capability: DecisionCapability,
    ) -> CapabilitySupport {
        self.inner
            .capabilities
            .get(&capability)
            .copied()
            .unwrap_or(CapabilitySupport::Unknown)
    }

    fn decision_limits(&self, _model: &str) -> DecisionLimits {
        self.inner.limits
    }
}

/// Builder for [`Provider`].
///
/// `Debug` shows the settings without the auth value, header values, or
/// credentials in the base URL.
#[derive(Default)]
pub struct ProviderBuilder {
    base_url: Option<String>,
    path: Option<String>,
    auth: Option<(String, String)>,
    credential_scope: Option<Cow<'static, str>>,
    credential_placement: Option<CredentialPlacement>,
    headers: Vec<(String, String)>,
    request_id_header_name: Option<String>,
    capabilities: HashMap<DecisionCapability, CapabilitySupport>,
    limits: DecisionLimits,
    provider_name: Option<Cow<'static, str>>,
    timeout: Option<Option<Duration>>,
    max_response_size: Option<usize>,
    http: Option<Arc<dyn HttpTransport>>,
    missing_endpoint_hint: Option<&'static str>,
    dialect: Option<wire::Dialect>,
}

impl std::fmt::Debug for ProviderBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderBuilder")
            .field("provider_name", &self.provider_name)
            .field("base_url", &self.base_url.as_deref().map(redact_url))
            .field("path", &self.path)
            .field("auth", &self.auth.as_ref().map(|(name, _)| name))
            .field("credential_scope", &self.credential_scope)
            .field(
                "headers",
                &self
                    .headers
                    .iter()
                    .map(|(name, _)| name)
                    .collect::<Vec<_>>(),
            )
            .field("http_transport", &self.http.is_some())
            .field("timeout", &self.timeout)
            .field("max_response_size", &self.max_response_size)
            .finish_non_exhaustive()
    }
}

impl ProviderBuilder {
    /// Base URL, e.g. `https://api.typesafe.ai`.
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Endpoint path (default `/v1/systemone`).
    #[must_use]
    pub fn path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Authenticate with `Authorization: Bearer <token>`. Surrounding
    /// whitespace is removed from the token.
    #[must_use]
    pub fn bearer_token(self, token: impl Into<String>) -> Self {
        self.auth_header("authorization", format!("Bearer {}", token.into().trim()))
    }

    /// Authenticate with a custom header. Surrounding whitespace is removed
    /// from the value.
    ///
    /// A per-request [`allama_http::Credential::api_key`] then goes in this
    /// header, as its whole value, unless
    /// [`credential_placement`](Self::credential_placement) says otherwise.
    #[must_use]
    pub fn auth_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.auth = Some((name.into(), value.into()));
        self
    }

    /// The credential scope (default: the provider name): a per-request
    /// [`allama_http::Credential`] addressed to it
    /// ([`allama_http::HttpRequestOptions::credential`]) is this provider's,
    /// and credentials addressed to other scopes are never sent.
    ///
    /// The scope names an account: two accounts of one endpoint in a
    /// fallback chain need two scopes (the default makes them share one),
    /// while providers behind one gateway share one.
    ///
    /// Without a configured credential ([`bearer_token`](Self::bearer_token),
    /// [`auth_header`](Self::auth_header)), setting it makes every request
    /// carry a credential for the scope (bring your own key), or fail with
    /// an authored [`Error::Configuration`] before anything is sent. With a
    /// configured credential, a request whose credential names a mistyped
    /// scope is sent with the configured one.
    #[must_use]
    pub fn credential_scope(mut self, scope: impl Into<Cow<'static, str>>) -> Self {
        self.credential_scope = Some(scope.into());
        self
    }

    /// Where a per-request [`allama_http::Credential::api_key`] goes.
    ///
    /// The default follows the configured credential:
    /// `Authorization: Bearer` for [`bearer_token`](Self::bearer_token) (or
    /// no credential), the header named by
    /// [`auth_header`](Self::auth_header) otherwise. When the placement
    /// writes another header than the configured credential, the
    /// configured one is replaced too.
    #[must_use]
    pub fn credential_placement(mut self, placement: CredentialPlacement) -> Self {
        self.credential_placement = Some(placement);
        self
    }

    /// Send the header `name: value` with every request, replacing a
    /// header of the same name the provider sets (later calls replace
    /// earlier ones). `header("user-agent", ..)` replaces the default
    /// [`allama_http::DEFAULT_USER_AGENT`]. A request's own
    /// [`allama_http::HttpRequestOptions`] headers still replace these.
    ///
    /// Credential headers are refused at [`build`](Self::build): configure
    /// the key with [`bearer_token`](Self::bearer_token) or
    /// [`auth_header`](Self::auth_header), or send one per request as an
    /// [`allama_http::Credential`].
    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Header carrying the request id (default `x-request-id`).
    #[must_use]
    pub fn request_id_header_name(mut self, name: impl Into<String>) -> Self {
        self.request_id_header_name = Some(name.into());
        self
    }

    /// Declare support for a question kind. Unset kinds answer `Unknown`.
    #[must_use]
    pub fn decision_capability(
        mut self,
        capability: DecisionCapability,
        support: CapabilitySupport,
    ) -> Self {
        self.capabilities.insert(capability, support);
        self
    }

    /// Request limits enforced before sending (default: none).
    #[must_use]
    pub fn limits(mut self, limits: DecisionLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Provider identity used for telemetry and errors (default
    /// `"systemone"`).
    #[must_use]
    pub fn provider_name(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        self.provider_name = Some(name.into());
        self
    }

    /// The timeout for a whole call; `None` sets none.
    ///
    /// Left unset, it is [`DEFAULT_REQUEST_TIMEOUT`] (120 s), whatever the
    /// transport: a client of your own keeps the bound unless you pass
    /// `None` here. Set, it applies in place of a client's own total
    /// timeout.
    #[must_use]
    pub fn timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// The largest response body, in bytes, the provider reads (default
    /// [`allama_http::DEFAULT_MAX_RESPONSE_SIZE`], 256 MiB). A larger body
    /// fails the call with [`Error::UnexpectedResponse`], which is not
    /// retryable, without being read whole.
    #[must_use]
    pub fn max_response_size(mut self, bytes: usize) -> Self {
        self.max_response_size = Some(bytes);
        self
    }

    /// The dialect a preset's endpoint speaks.
    pub(crate) fn dialect(mut self, dialect: wire::Dialect) -> Self {
        self.dialect = Some(dialect);
        self
    }

    /// What a preset adds to the refusal for a missing System One endpoint:
    /// how to get a server that has one.
    pub(crate) fn missing_endpoint_hint(mut self, hint: &'static str) -> Self {
        self.missing_endpoint_hint = Some(hint);
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
    /// transport with each request, whatever the transport.
    #[must_use]
    pub fn http_transport(mut self, transport: impl HttpTransport) -> Self {
        self.http = Some(Arc::new(transport));
        self
    }

    /// Build the provider.
    ///
    /// # Errors
    ///
    /// An authored [`Error::Configuration`], naming the setting, when the
    /// base URL is missing or not an absolute `http` or `https` URL without
    /// a query ([`allama_http::config::base_url`]), the auth value holds a
    /// character no HTTP header can carry, a [`header`](Self::header) is
    /// malformed or carries a credential
    /// ([`allama_http::config::static_headers`]), or when no transport is
    /// set and the `reqwest` feature is off.
    pub fn build(self) -> Result<Provider> {
        let base_url = match self.base_url {
            Some(url) => allama_http::config::base_url("base_url", &url)?,
            None => return Err(Error::configuration("base_url is required").authored()),
        };
        let auth = self
            .auth
            .map(|(name, value)| {
                let setting = format!("the `{name}` header");
                let header = HeaderName::try_from(name.as_str()).map_err(|_| {
                    Error::configuration(format!("`{name}` is not a valid HTTP header name"))
                        .authored()
                })?;
                let value = allama_http::config::secret(&setting, &value)?;
                let value = allama_http::config::sensitive_value(&setting, &value)?;
                Ok::<_, Error>((header, value))
            })
            .transpose()?;
        let path = self.path.unwrap_or_else(|| "/v1/systemone".into());
        let path = if path.starts_with('/') {
            path
        } else {
            format!("/{path}")
        };
        let http = match self.http {
            Some(http) => http,
            None => default_transport()?,
        };
        let provider_name = self.provider_name.unwrap_or(Cow::Borrowed(PROVIDER_NAME));
        let credential_slot = credential_slot(
            self.credential_scope
                .clone()
                .unwrap_or_else(|| provider_name.clone()),
            self.credential_placement,
            auth.as_ref(),
            self.credential_scope.is_some(),
        );
        let headers = allama_http::config::static_headers(self.headers, &credential_slot)?;
        Ok(Provider {
            inner: Arc::new(Inner {
                http,
                credential_slot,
                transport: TransportConfig {
                    base_url,
                    path,
                    auth,
                    headers,
                    request_id_header_name: self
                        .request_id_header_name
                        .unwrap_or_else(|| "x-request-id".into()),
                    timeout: self.timeout.unwrap_or(Some(DEFAULT_REQUEST_TIMEOUT)),
                    max_response_size: self
                        .max_response_size
                        .unwrap_or(allama_http::DEFAULT_MAX_RESPONSE_SIZE),
                },
                capabilities: self.capabilities,
                limits: self.limits,
                provider_name,
                missing_endpoint_hint: self.missing_endpoint_hint,
                dialect: self.dialect.unwrap_or(wire::Dialect::SystemOne),
            }),
        })
    }
}

/// Endpoint and header configuration.
///
/// `Debug` output redacts the `auth` value whatever its header name, the
/// other header values when their names look like credentials
/// ([`allama_http::is_sensitive_header`]), and the base URL's credentials.
#[derive(Clone, PartialEq, Eq)]
struct TransportConfig {
    /// Base URL without a trailing slash.
    base_url: String,
    /// Endpoint path.
    path: String,
    /// Auth header, its value marked sensitive, when the endpoint requires
    /// one.
    auth: Option<(HeaderName, HeaderValue)>,
    /// The headers sent with every request: the `User-Agent` and the
    /// builder's own.
    headers: HeaderMap,
    /// Request id header name.
    request_id_header_name: String,
    /// Request timeout.
    timeout: Option<Duration>,
    /// The largest response body, in bytes.
    max_response_size: usize,
}

impl std::fmt::Debug for TransportConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransportConfig")
            .field("base_url", &redact_url(&self.base_url))
            .field("path", &self.path)
            .field(
                "auth",
                &self.auth.as_ref().map(|(name, _)| (name, Redacted)),
            )
            .field("headers", &allama_http::redacted_headers(&self.headers))
            .field("request_id_header_name", &self.request_id_header_name)
            .field("timeout", &self.timeout)
            .field("max_response_size", &self.max_response_size)
            .finish()
    }
}

impl TransportConfig {
    /// Full endpoint URL.
    fn url(&self) -> String {
        format!("{}{}", self.base_url, self.path)
    }
}

/// The provider's `confidence` statistic per question, in
/// [`DecisionResponse::provider_data`].
///
/// Confidence summarizes how concentrated a choice or score distribution is.
/// It is not calibrated correctness, and providers compute it differently
/// (Ollama: `1 − H(p)/ln N`; TypeSafe: its own formulas), so values are not
/// comparable across providers. Yes/no answers have none.
///
/// ```
/// # fn example(response: allama::DecisionResponse) -> allama::Result<()> {
/// use allama_systemone::Confidence;
///
/// if let Some(confidence) = response.provider_data.get::<Confidence>()? {
///     println!("label confidence: {:?}", confidence.0.get("label"));
/// }
/// # Ok(()) }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Confidence(pub BTreeMap<String, f64>);

impl allama::ProviderDataEntry for Confidence {
    const NAMESPACE: &'static str = NAMESPACE;
    const KEY: &'static str = "confidence";
}

/// What a gateway adds to a System One response, in
/// [`DecisionResponse::provider_data`]: OpenRouter reports the upstream provider
/// and the cost. Its response id goes to [`DecisionResponse::id`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Gateway {
    /// The provider the gateway routed to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// The cost the gateway charged, in its currency (credits or USD).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
}

impl allama::ProviderDataEntry for Gateway {
    const NAMESPACE: &'static str = NAMESPACE;
    const KEY: &'static str = "gateway";
}

/// The verbatim response body, kept when
/// [`DecisionRequestOptions::raw_response`] is set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RawResponse(pub serde_json::Value);

impl allama::ProviderDataEntry for RawResponse {
    const NAMESPACE: &'static str = NAMESPACE;
    const KEY: &'static str = allama::provider_data::RAW_RESPONSE_KEY;
}

/// The credential slot for `scope`: `placement`, or the placement the
/// configured `auth` header implies; replacing `auth` when the placement
/// writes another header; required when nothing is configured and the scope
/// was chosen explicitly.
fn credential_slot(
    scope: Cow<'static, str>,
    placement: Option<CredentialPlacement>,
    auth: Option<&(HeaderName, HeaderValue)>,
    explicit: bool,
) -> CredentialSlot {
    let placement = placement.unwrap_or_else(|| match auth {
        Some((name, value))
            if name != allama_http::http::header::AUTHORIZATION
                || !value.as_bytes().starts_with(b"Bearer ") =>
        {
            CredentialPlacement::Header(name.as_str().to_owned().into())
        }
        _ => CredentialPlacement::Bearer,
    });
    let mut slot = CredentialSlot::new(scope, placement).required(auth.is_none() && explicit);
    if let (Some((name, _)), Some(own)) = (auth, slot.header_name())
        && !own.eq_ignore_ascii_case(name.as_str())
    {
        slot = slot.replaces(name.as_str().to_owned());
    }
    slot
}

/// A Workers AI model name as a URL path (`@cf/cloudflare/clef`), refused
/// when it could leave the account's `run` path the token is sent to.
fn model_path(model: &str) -> Result<&str> {
    let safe = model
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'@' | b'/' | b'_' | b'.' | b'-'))
        && model
            .split('/')
            .all(|segment| !segment.is_empty() && segment != "." && segment != "..");
    if safe {
        Ok(model)
    } else {
        Err(Error::invalid_request(format!(
            "{model:?} is not a Workers AI model name such as `@cf/cloudflare/clef`"
        ))
        .authored())
    }
}

fn request_id(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .filter(|value| !value.is_empty())
        .map(|value| String::from_utf8_lossy(value.as_bytes()).into_owned())
}

/// A `Debug` placeholder for a secret value.
struct Redacted;

impl std::fmt::Debug for Redacted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<redacted>")
    }
}

/// The transport of a provider built without `http_transport`: `reqwest`
/// (feature `reqwest`).
fn default_transport() -> Result<Arc<dyn HttpTransport>> {
    #[cfg(feature = "reqwest")]
    return Ok(Arc::new(allama_http::ReqwestTransport::new()));
    #[cfg(not(feature = "reqwest"))]
    return Err(Error::configuration(
        "no HTTP transport: set one with `http_transport` or enable the `reqwest` feature",
    )
    .authored());
}

#[cfg(test)]
mod tests {
    use super::*;
    use allama_http::http::StatusCode;

    /// One timeout model for every builder: unset, the default applies
    /// whatever the transport, a client of your own included; only an
    /// explicit `None` removes it.
    #[cfg(feature = "reqwest")]
    #[test]
    fn the_timeout_is_kept_whatever_the_transport() {
        let five = Some(Duration::from_secs(5));
        let client = || allama_http::ReqwestTransport::from_client(reqwest::Client::new());
        let timeout = |builder| ProviderBuilder::build(builder).unwrap().transport().timeout;
        let builder = || Provider::builder().base_url("http://localhost:1");
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
            let transport = allama_http::ReplayTransport::new([allama_http::ReplayResponse::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "",
            )]);
            let mut builder = Provider::builder()
                .base_url("http://localhost:1")
                .http_transport(transport.clone());
            if let Some(timeout) = set {
                builder = builder.timeout(timeout);
            }
            let provider = builder.build().unwrap();
            let _ = allama::DecisionProvider::decide(
                &provider,
                &DecisionRequest::new("m", serde_json::json!({"text": "hi"}))
                    .question("q", allama::decision::Question::yes_no("Yes?")),
            )
            .await;
            assert_eq!(transport.requests()[0].timeout, expected, "{set:?}");
        }
    }

    #[cfg(not(feature = "reqwest"))]
    #[test]
    fn building_without_a_transport_needs_reqwest() {
        let Err(err) = Provider::builder()
            .base_url("http://localhost:11434")
            .build()
        else {
            panic!("built without a transport");
        };
        assert!(matches!(err, Error::Configuration { .. }), "{err}");
        assert!(err.to_string().contains("`http_transport`"), "{err}");
    }

    #[test]
    fn builder_defaults() {
        let provider = Provider::builder()
            .base_url("https://example.com///")
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(
            provider.transport().url(),
            "https://example.com/v1/systemone"
        );
        assert_eq!(provider.provider_name(), "systemone");
        assert!(provider.reads_namespace(NAMESPACE));
        assert!(!provider.reads_namespace("openai"));
        assert_eq!(provider.transport().timeout, Some(DEFAULT_REQUEST_TIMEOUT));
        assert_eq!(provider.decision_limits("m"), DecisionLimits::default());
        assert_eq!(
            provider.decision_capability("m", DecisionCapability::Score),
            CapabilitySupport::Unknown
        );
        assert!(Provider::builder().build().is_err());
        assert!(Provider::builder().base_url(" ").build().is_err());
    }

    #[test]
    fn debug_output_never_shows_credentials() {
        let provider = Provider::builder()
            .base_url("https://example.com")
            .auth_header("x-custom", "SECRET1")
            // Not on the credential list, so it is sent; redacted anyway.
            .header("x-gateway-token", "SECRET2")
            .header("x-tenant-id", "tenant-visible")
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        let debug = format!("{provider:?} {:?}", provider.transport());
        assert!(!debug.contains("SECRET"), "{debug}");
        assert!(debug.contains("tenant-visible"), "{debug}");

        let bearer = Provider::builder()
            .base_url("https://example.com")
            .bearer_token("sk-or-SECRET3")
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        let debug = format!("{:?}", bearer.transport());
        assert!(!debug.contains("SECRET"), "{debug}");

        let builder = Provider::builder()
            .base_url("https://user:PASSWORD@example.com")
            .bearer_token("SECRET4");
        let builder_debug = format!("{builder:?}");
        let debug = format!("{builder_debug} {:?}", builder_build(builder));
        assert!(
            !debug.contains("SECRET") && !debug.contains("PASSWORD"),
            "{debug}"
        );

        let Err(err) = Provider::builder()
            .base_url("https://example.com")
            .header("Authorization", "Bearer SECRET5")
            .http_transport(allama_http::ReplayTransport::default())
            .build()
        else {
            panic!("built with a credential header");
        };
        assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    }

    /// Options under the provider name are
    /// reported dropped; the `systemone` namespace is read.
    #[test]
    fn options_under_the_provider_name_are_reported_dropped() {
        let provider = Provider::builder()
            .base_url("https://example.com")
            .provider_name("typesafe")
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        let request = DecisionRequest::new("nimble", "Is this spam?")
            .question("spam", allama::decision::Question::yes_no("Is it spam?"))
            .with_options_json("typesafe", serde_json::json!({"keep_alive": "10m"}));
        let validation = provider.validate_decision_request(&request).unwrap();
        let [warning] = validation.warnings() else {
            panic!("one warning: {:?}", validation.warnings());
        };
        assert_eq!(warning.kind, WarningKind::Dropped);
        assert_eq!(
            warning.code.as_deref(),
            Some("dropped.provider_name_options")
        );
        assert_eq!(
            warning.subject,
            Subject::Option {
                namespace: "typesafe".into()
            }
        );
        let read = DecisionRequest::new("nimble", "Is this spam?")
            .question("spam", allama::decision::Question::yes_no("Is it spam?"))
            .with_options_json(NAMESPACE, serde_json::json!({"raw_response": true}));
        let validation = provider.validate_decision_request(&read).unwrap();
        assert!(
            validation.warnings().is_empty(),
            "{:?}",
            validation.warnings()
        );
    }

    fn builder_build(builder: ProviderBuilder) -> Provider {
        builder
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap()
    }

    #[test]
    fn the_credential_slot_follows_the_configured_credential() {
        let auth = |name: &str, value: &str| {
            Some((
                HeaderName::try_from(name).unwrap(),
                HeaderValue::try_from(value).unwrap(),
            ))
        };
        let slot = |placement, auth: Option<(HeaderName, HeaderValue)>, explicit| {
            credential_slot("s".into(), placement, auth.as_ref(), explicit)
        };
        assert_eq!(
            slot(None, auth("authorization", "Bearer k"), true),
            CredentialSlot::new("s", CredentialPlacement::Bearer)
        );
        assert_eq!(
            slot(None, auth("Authorization", "raw-token"), false),
            CredentialSlot::new("s", CredentialPlacement::Header("authorization".into()))
        );
        assert_eq!(
            slot(None, None, false),
            CredentialSlot::new("s", CredentialPlacement::Bearer)
        );
        assert_eq!(
            slot(None, None, true),
            CredentialSlot::new("s", CredentialPlacement::Bearer).required(true)
        );
        assert_eq!(
            slot(
                Some(CredentialPlacement::Header("api-key".into())),
                auth("authorization", "Bearer entra"),
                false
            ),
            CredentialSlot::new("s", CredentialPlacement::Header("api-key".into()))
                .replaces("authorization")
        );
        assert_eq!(
            slot(
                Some(CredentialPlacement::Custom),
                auth("x-sig", "configured"),
                false
            ),
            CredentialSlot::new("s", CredentialPlacement::Custom)
        );

        let named = Provider::builder()
            .base_url("https://example.com")
            .provider_name("acme")
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(named.credential_slot().scope, "acme");
        assert!(!named.credential_slot().required);
    }

    #[test]
    fn custom_path() {
        let provider = Provider::builder()
            .base_url("http://localhost:4000")
            .path("typesafe/v1/systemone")
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(
            provider.transport().url(),
            "http://localhost:4000/typesafe/v1/systemone"
        );
    }
}
