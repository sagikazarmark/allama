#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
//! Cloudflare Workers AI over its REST API: extraction models.
//!
//! [`Provider`] implements [`allama::ExtractionProvider`] for Clairic
//! (`@cf/cloudflare/clairic`), Cloudflare's extraction model, at
//! `https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}`.
//! Chat and embeddings on Workers AI are `allama_openai_compat`'s
//! `Cloudflare` preset, decisions `allama_systemone`'s; this crate hosts the
//! operation Workers AI serves in a shape of its own. Inside a Worker,
//! `allama-cloudflare-worker` runs the same models through the AI binding
//! with this crate's codec.
//!
//! ```no_run
//! use allama::extraction::{Field, ListItemKind};
//! use allama::{ExtractionProvider, ExtractionRequest};
//! use allama_cloudflare::Provider;
//!
//! # async fn example() -> allama::Result<()> {
//! let provider = Provider::from_env()?;
//! let request = ExtractionRequest::new(
//!     "@cf/cloudflare/clairic",
//!     "Order A-88412 arrived on Sept 28 with a cracked screen. I paid $649.00 plus $52.10 tax.",
//! )
//! .field("order_id", Field::text().instructions("Order number"))
//! .field("amount_paid", Field::number().decimals(2).unit("USD").instructions("Total paid including tax"))
//! .field("damage", Field::list(ListItemKind::Text).instructions("Each kind of damage described"));
//! let response = provider.extract_values(&request).await?;
//! println!("{:?}", response.number("amount_paid").unwrap().value);
//! # Ok(()) }
//! ```
//!
//! # Mapping
//!
//! - The body's `model` is the selector Clairic reads, the last segment of
//!   the model name (`clairic`); the model name is also the URL's.
//! - Every field parameter goes as given (ADR 0018): `instructions`,
//!   `integer`, `decimals`, `unit`, `items`, `min`, `max`, `step`, `scale`,
//!   `horizon`, and the forecast period as `interval`.
//! - Images go as `{"content_type", "base64"}`, inline only; an image
//!   detail is a hint Workers AI has no setting for, warned about.
//! - Text spans come back as the model's character offsets and are placed
//!   as byte spans into the request's input, by offset conversion or by
//!   finding the value; a span that cannot be placed is dropped with a
//!   warning coded `dropped.span`.
//! - Estimates keep `value`, `median` and the `p10`–`p90` interval as an
//!   80% [`PredictionInterval`](allama::extraction::PredictionInterval);
//!   forecasts are read in either of Clairic's shapes (parallel arrays, or
//!   `steps`).
//! - Clairic's statistics are provider data under
//!   [`NAMESPACE`]: [`Confidence`] and [`Guesses`] per field,
//!   [`Distributions`] (`mean`, `p_below`, `p_above`, `quantiles`) per
//!   estimate or forecast step, [`AnswerExtras`] for keys of an answer
//!   this adapter does not model, and the [`RunDetails`] (`passes`,
//!   `truncated`, `state_text`, the token counts as reported).
//!   `truncated` is also a warning coded `approximated.input_truncated`.
//! - Usage keeps Clairic's token breakdowns under `Usage`'s keys when they
//!   fit its contract (a subset of the input total): `text_input_tokens`,
//!   `image_input_tokens`, `schema_input_tokens`; the counts as reported
//!   are in [`RunDetails::tokens`] either way.
//! - Clairic's tuning is an [`ExtractionRequestOptions`]: per-field
//!   `null_threshold`, `include_quantiles`, `return_state_text`.
//!
//! Clairic's input schema is checked before sending, so a request the
//! model would refuse costs no call: field names of at most 64 of
//! `[A-Za-z0-9_.-]` starting with a letter or `_`, `decimals` up to 6,
//! images of 4 MiB each and 8 MiB together, inline only, each an
//! [`Error::InvalidRequest`] (or `Unsupported` for a URL image) that names
//! the field (ADR 0018, decision 10). Counts (the number of images, a
//! forecast's horizon) are model data and not claimed as limits: the model
//! refuses them with an `InvalidRequest` of its own (ADR 0015).
//!
//! # Transport
//!
//! Requests go through `reqwest` (the default `reqwest` feature) unless the
//! builder is given another [`HttpTransport`] with `http_transport`. With
//! `default-features = false` and an injected transport the crate builds
//! without `reqwest`; `new` and `from_env` are then not compiled.
//!
//! # Per-request credentials
//!
//! A [`allama_http::Credential::api_key`] addressed to the provider's
//! credential scope (`cloudflare` unless changed with
//! [`ProviderBuilder::credential_scope`]) replaces the configured token for
//! that request; credentials addressed to other scopes are never sent. See
//! [`allama_http::HttpRequestOptions`].

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use allama::{
    CapabilitySupport, Error, ExtractionCapability, ExtractionLimits, ExtractionProvider,
    ExtractionRequest, ExtractionResponse, ExtractionValidation, ProviderIdentity, Result,
    ValidationCoverage,
};
use allama_http::config::redact_url;
use allama_http::{
    CredentialPlacement, CredentialSlot, DEFAULT_REQUEST_TIMEOUT, Headers, HttpRequest,
    HttpResponse, HttpTransport,
};
use serde::{Deserialize, Serialize};

#[doc(hidden)]
pub mod binding;
mod error;
mod options;
mod wire;

pub use error::map_http_error;
pub use options::ExtractionRequestOptions;

/// Namespace for Workers AI options and provider data, shared with
/// `allama-cloudflare-worker` so both transports' data is interchangeable.
pub const NAMESPACE: &str = "cloudflare-workers-ai";

/// The [`provider_name`](allama::ProviderIdentity::provider_name) of
/// [`Provider`]: the one `allama_openai_compat`'s and `allama_systemone`'s
/// Cloudflare presets and `allama-cloudflare-worker` report, so every
/// operation on Workers AI shares one `gen_ai.provider.name`.
pub const PROVIDER_NAME: &str = "cloudflare";

/// The default credential scope of [`Provider`]: its provider name, the one
/// the other Cloudflare presets use, so one per-request credential reaches
/// the account for every operation.
pub const DEFAULT_CREDENTIAL_SCOPE: &str = PROVIDER_NAME;

/// A Workers AI extraction provider.
#[derive(Clone)]
pub struct Provider {
    inner: Arc<Inner>,
}

#[derive(Clone)]
struct Inner {
    http: Arc<dyn HttpTransport>,
    transport: TransportConfig,
    credential_slot: CredentialSlot,
    capabilities: HashMap<ExtractionCapability, CapabilitySupport>,
    limits: ExtractionLimits,
}

impl std::fmt::Debug for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Provider")
            .field("url", &redact_url(&self.inner.transport.base_url))
            .finish_non_exhaustive()
    }
}

impl Provider {
    /// Start configuring a provider.
    #[must_use]
    pub fn builder() -> ProviderBuilder {
        ProviderBuilder::default()
    }

    /// Create a provider for `account_id` with `api_token`.
    ///
    /// # Errors
    ///
    /// As [`ProviderBuilder::build`].
    #[cfg(feature = "reqwest")]
    pub fn new(account_id: &str, api_token: &str) -> Result<Self> {
        Self::builder()
            .account_id(account_id)
            .api_token(api_token)
            .build()
    }

    /// Create a provider from `CLOUDFLARE_ACCOUNT_ID` and
    /// `CLOUDFLARE_API_TOKEN`, the variables the other Cloudflare presets
    /// read.
    ///
    /// # Errors
    ///
    /// Fails when either variable is missing or malformed.
    #[cfg(feature = "reqwest")]
    pub fn from_env() -> Result<Self> {
        let account_id = optional_env_var("CLOUDFLARE_ACCOUNT_ID")
            .ok_or_else(|| Error::configuration("CLOUDFLARE_ACCOUNT_ID not set").authored())?;
        let token = optional_env_var("CLOUDFLARE_API_TOKEN")
            .ok_or_else(|| Error::configuration("CLOUDFLARE_API_TOKEN not set").authored())?;
        Self::new(&account_id, &token)
    }

    /// The URL requests for `model` are posted to. It may carry a user name
    /// and password: show it with [`allama_http::config::redact_url`].
    ///
    /// # Errors
    ///
    /// [`Error::InvalidRequest`] when `model` is not a Workers AI model name.
    pub fn url(&self, model: &str) -> Result<String> {
        Ok(format!(
            "{}/{}",
            self.inner.transport.base_url,
            model_path(model)?
        ))
    }

    /// The credential slot this provider passes to
    /// [`allama_http::HttpRequest::apply_options`].
    #[must_use]
    pub fn credential_slot(&self) -> &CredentialSlot {
        &self.inner.credential_slot
    }

    /// The same provider with its credential scope set to `scope` (see
    /// [`ProviderBuilder::credential_scope`]).
    #[must_use]
    pub fn with_credential_scope(self, scope: impl Into<Cow<'static, str>>) -> Self {
        let mut inner = (*self.inner).clone();
        inner.credential_slot.scope = scope.into();
        inner.credential_slot.required = inner.transport.auth.is_none();
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Everything before sending, shared by the call and its dry run: the
    /// options, the body, and the HTTP request with the caller's transport
    /// options applied.
    fn encode(
        &self,
        prepared: &mut allama::adapter::extraction::Prepared<'_>,
    ) -> Result<(HttpRequest, ExtractionRequestOptions)> {
        let (request, d) = prepared.parts_mut();
        let mut options: ExtractionRequestOptions = request.resolve_option()?;
        if !std::mem::take(&mut options.run_options).is_empty() {
            d.hint(
                allama::Subject::Other {
                    name: "`run_options`".into(),
                },
                "`run_options` belong to a Worker's binding call; the REST API has no such argument, so they were not sent",
            )?;
        }
        let body = serde_json::to_string(&wire::encode(request, &options, d)?)?;
        let config = &self.inner.transport;
        let url = format!("{}/{}", config.base_url, model_path(&request.model)?);
        let mut http = HttpRequest::post(url, body)
            .header("content-type", "application/json")
            .timeout(config.timeout);
        if let Some((name, value)) = &config.auth {
            http = http.header(name, value);
        }
        let http = http
            .with_headers(&config.headers)
            .apply_options(&request.options, &self.inner.credential_slot)
            .map_err(|err| err.with_provider(PROVIDER_NAME))?;
        Ok((http, options))
    }

    async fn send(&self, request: HttpRequest) -> Result<HttpResponse> {
        let response = self
            .inner
            .http
            .send(request)
            .await
            .map_err(|err| err.into_error().with_provider(PROVIDER_NAME))?;
        if response.is_success() {
            return Ok(response);
        }
        let error = response
            .map_error(|status, headers, body| {
                map_http_error(status, headers, body)
                    .with_request_id_opt(request_id(headers).as_deref())
            })
            .await;
        Err(error.with_provider(PROVIDER_NAME))
    }
}

impl ProviderIdentity for Provider {
    fn provider_name(&self) -> &str {
        PROVIDER_NAME
    }

    fn reads_namespace(&self, namespace: &str) -> bool {
        namespace == NAMESPACE
    }

    fn downcast_any(&self, id: std::any::TypeId) -> Option<&(dyn std::any::Any + Send + Sync)> {
        allama::expose(self, id)
    }
}

impl ExtractionProvider for Provider {
    async fn extract_values(&self, request: &ExtractionRequest) -> Result<ExtractionResponse> {
        let mut prepared = allama::adapter::extraction::prepare(self, request)?;
        let (http, options) = self.encode(&mut prepared)?;
        prepared.mark_sent();
        let response = self.send(http).await?;
        let status = response.status;
        let headers = response.headers.clone();
        let request_id = request_id(&response.headers);
        let with_context = |err: Error| {
            err.with_provider(PROVIDER_NAME)
                .with_request_id_opt(request_id.as_deref())
        };
        let bytes = response
            .bytes_limited(self.inner.transport.max_response_size)
            .await
            .map_err(|err| with_context(err.into_error()))?;
        // The Cloudflare API wraps the model's output.
        let body: wire::WireResponse = serde_json::from_slice::<wire::Envelope>(&bytes)
            .and_then(|envelope| serde_json::from_str(envelope.result.get()))
            .map_err(|err| {
                let text = String::from_utf8_lossy(&bytes);
                // An error with a success status: the envelope's code
                // classifies it when it is an HTTP status.
                let mapped = match error::error_in_body(&text) {
                    Some(envelope) => match envelope
                        .code
                        .as_deref()
                        .and_then(|code| code.parse::<u16>().ok())
                        .filter(|code| (400..=599).contains(code))
                    {
                        Some(code) => map_http_error(code, &headers, &text),
                        None => envelope
                            .apply(
                                Error::provider(format!("error envelope with HTTP {status}"))
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
            statistics,
        } = wire::decode(body, request).map_err(with_context)?;
        response.request_id.clone_from(&request_id);
        response.provider = Some(Cow::Borrowed(PROVIDER_NAME));
        attach_provider_data(&mut response, statistics).map_err(with_context)?;
        if options.raw_response {
            let raw: serde_json::Value = serde_json::from_slice(&bytes).map_err(|err| {
                with_context(allama_http::malformed_response(status, &bytes, err))
            })?;
            response
                .provider_data
                .insert(&RawResponse(raw))
                .map_err(with_context)?;
        }
        Ok(prepared.finish(response))
    }

    /// A dry run of [`extract_values`](Self::extract_values): the same
    /// checks, options and encoding, without sending
    /// ([`ValidationCoverage::Encoding`]).
    fn validate_extraction_request(
        &self,
        request: &ExtractionRequest,
    ) -> Result<ExtractionValidation> {
        let mut prepared = allama::adapter::extraction::prepare_dry_run(self, request)?;
        self.encode(&mut prepared)?;
        Ok(prepared.into_validation(ValidationCoverage::Encoding))
    }

    /// Every field kind is `Supported`: the extraction shape this endpoint
    /// serves is Clairic's, which declares them all. Whether a model reads
    /// images is model data, so `ImageInput` is `Unknown` (ADR 0015), as
    /// the decision presets answer it. The builder can say otherwise.
    fn extraction_capability(
        &self,
        _model: &str,
        capability: ExtractionCapability,
    ) -> CapabilitySupport {
        self.inner
            .capabilities
            .get(&capability)
            .copied()
            .unwrap_or_else(|| builtin_extraction_capability(capability))
    }

    fn extraction_limits(&self, _model: &str) -> ExtractionLimits {
        self.inner.limits
    }
}

/// Builder for [`Provider`].
///
/// `Debug` shows the settings without the token or header values.
#[derive(Default)]
pub struct ProviderBuilder {
    account_id: Option<String>,
    api_token: Option<String>,
    base_url: Option<String>,
    credential_scope: Option<Cow<'static, str>>,
    headers: Vec<(String, String)>,
    capabilities: HashMap<ExtractionCapability, CapabilitySupport>,
    limits: ExtractionLimits,
    timeout: Option<Option<Duration>>,
    max_response_size: Option<usize>,
    http: Option<Arc<dyn HttpTransport>>,
}

impl std::fmt::Debug for ProviderBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderBuilder")
            .field("account_id", &self.account_id)
            .field("api_token", &self.api_token.as_ref().map(|_| Redacted))
            .field("base_url", &self.base_url.as_deref().map(redact_url))
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
            .finish_non_exhaustive()
    }
}

impl ProviderBuilder {
    /// The Cloudflare account id: 32 hexadecimal digits.
    #[must_use]
    pub fn account_id(mut self, account_id: impl Into<String>) -> Self {
        self.account_id = Some(account_id.into());
        self
    }

    /// The API token, sent as `Authorization: Bearer`. Surrounding
    /// whitespace is removed.
    #[must_use]
    pub fn api_token(mut self, token: impl Into<String>) -> Self {
        self.api_token = Some(token.into());
        self
    }

    /// The base URL of the account's `run` endpoint, in place of
    /// `https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run`:
    /// for an AI Gateway or a test server. Models are posted to
    /// `{base_url}/{model}`.
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// The credential scope (default [`DEFAULT_CREDENTIAL_SCOPE`]): a
    /// per-request [`allama_http::Credential`] addressed to it is this
    /// provider's. Without a configured token, setting it makes every
    /// request carry a credential for the scope, or fail with an authored
    /// [`Error::Configuration`] before anything is sent.
    #[must_use]
    pub fn credential_scope(mut self, scope: impl Into<Cow<'static, str>>) -> Self {
        self.credential_scope = Some(scope.into());
        self
    }

    /// Send the header `name: value` with every request. Credential headers
    /// are refused at [`build`](Self::build).
    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Answer `support` for `capability` instead of `Supported`.
    #[must_use]
    pub fn extraction_capability(
        mut self,
        capability: ExtractionCapability,
        support: CapabilitySupport,
    ) -> Self {
        self.capabilities.insert(capability, support);
        self
    }

    /// Request limits enforced before sending (default: none).
    #[must_use]
    pub fn limits(mut self, limits: ExtractionLimits) -> Self {
        self.limits = limits;
        self
    }

    /// The timeout for a whole call; `None` sets none. Left unset, it is
    /// [`DEFAULT_REQUEST_TIMEOUT`] (120 s), whatever the transport.
    #[must_use]
    pub fn timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// The largest response body, in bytes, the provider reads (default
    /// [`allama_http::DEFAULT_MAX_RESPONSE_SIZE`]).
    #[must_use]
    pub fn max_response_size(mut self, bytes: usize) -> Self {
        self.max_response_size = Some(bytes);
        self
    }

    /// Send requests through `transport` instead of the default `reqwest`
    /// client.
    #[must_use]
    pub fn http_transport(mut self, transport: impl HttpTransport) -> Self {
        self.http = Some(Arc::new(transport));
        self
    }

    /// Build the provider.
    ///
    /// # Errors
    ///
    /// An authored [`Error::Configuration`] when neither an account id nor
    /// a base URL is set, the account id is not 32 hexadecimal digits, the
    /// base URL is not an absolute `http` or `https` URL without a query, the
    /// token is blank or malformed, a header is malformed or carries a
    /// credential, or no transport is set and the `reqwest` feature is off.
    pub fn build(self) -> Result<Provider> {
        let base_url = match (self.base_url, self.account_id) {
            (Some(url), _) => allama_http::config::base_url("base_url", &url)?,
            (None, Some(account_id)) => {
                let account_id = account_id.trim();
                if account_id.len() != 32 || !account_id.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return Err(Error::configuration(
                        "the Cloudflare account id must be 32 hexadecimal digits",
                    )
                    .authored());
                }
                format!("https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run")
            }
            (None, None) => {
                return Err(
                    Error::configuration("an account id or a base URL is required").authored(),
                );
            }
        };
        let auth = self
            .api_token
            .map(|token| {
                let token = allama_http::config::secret("api_token", token.trim())?;
                Ok::<_, Error>(("authorization".to_owned(), format!("Bearer {token}")))
            })
            .transpose()?;
        let http = match self.http {
            Some(http) => http,
            None => default_transport()?,
        };
        let scope = self
            .credential_scope
            .clone()
            .unwrap_or(Cow::Borrowed(DEFAULT_CREDENTIAL_SCOPE));
        let credential_slot = CredentialSlot::new(scope, CredentialPlacement::Bearer)
            .required(auth.is_none() && self.credential_scope.is_some());
        let headers = allama_http::config::static_headers(self.headers, &credential_slot)?;
        Ok(Provider {
            inner: Arc::new(Inner {
                http,
                credential_slot,
                transport: TransportConfig {
                    base_url,
                    auth,
                    headers,
                    timeout: self.timeout.unwrap_or(Some(DEFAULT_REQUEST_TIMEOUT)),
                    max_response_size: self
                        .max_response_size
                        .unwrap_or(allama_http::DEFAULT_MAX_RESPONSE_SIZE),
                },
                capabilities: self.capabilities,
                limits: self.limits,
            }),
        })
    }
}

/// Endpoint and header configuration.
#[derive(Clone)]
struct TransportConfig {
    /// The `run` URL without a trailing slash.
    base_url: String,
    /// The auth header, when a token is configured.
    auth: Option<(String, String)>,
    /// The headers sent with every request.
    headers: Headers,
    /// Request timeout.
    timeout: Option<Duration>,
    /// The largest response body, in bytes.
    max_response_size: usize,
}

/// Clairic's `confidence` per field, in
/// [`ExtractionResponse::provider_data`].
///
/// It summarises the model's certainty about the value it would give, not
/// about whether the value is null: a null answer carries a confidence
/// about its [guess](Guesses). It is provider-defined, so values are not
/// comparable with another provider's.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Confidence(pub BTreeMap<String, f64>);

impl allama::ProviderDataEntry for Confidence {
    const NAMESPACE: &'static str = NAMESPACE;
    const KEY: &'static str = "confidence";
}

/// Clairic's `guess` per scalar field, in
/// [`ExtractionResponse::provider_data`]: what the model would have
/// answered, as JSON, reported for null answers too.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Guesses(pub BTreeMap<String, serde_json::Value>);

impl allama::ProviderDataEntry for Guesses {
    const NAMESPACE: &'static str = NAMESPACE;
    const KEY: &'static str = "guesses";
}

/// Clairic's distribution details per estimate field, in
/// [`ExtractionResponse::provider_data`]: one [`Distribution`] for an
/// estimate, one per step for a forecast.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Distributions(pub BTreeMap<String, Vec<Distribution>>);

impl allama::ProviderDataEntry for Distributions {
    const NAMESPACE: &'static str = NAMESPACE;
    const KEY: &'static str = "distributions";
}

/// What Clairic reports about an estimate's distribution beyond the
/// portable value, median and interval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Distribution {
    /// The mean.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean: Option<f64>,
    /// `p_below`, undocumented by Cloudflare; read as the mass below `min`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p_below: Option<f64>,
    /// `p_above`, undocumented by Cloudflare; read as the mass above `max`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p_above: Option<f64>,
    /// The 99 percentiles, when
    /// [`include_quantiles`](ExtractionRequestOptions::include_quantiles)
    /// asked for them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantiles: Option<Vec<f64>>,
}

/// What Clairic reports about the run itself, in
/// [`ExtractionResponse::provider_data`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RunDetails {
    /// Whether the model truncated the input (also a warning coded
    /// `approximated.input_truncated`).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
    /// How many parallel passes answered the fields.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub passes: Option<u64>,
    /// The text the model's offsets index into, when
    /// [`return_state_text`](ExtractionRequestOptions::return_state_text)
    /// asked for it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_text: Option<String>,
    /// Clairic's token counts as reported. The ones that fit `Usage`'s
    /// breakdown contract (a subset of the input total) are also
    /// `Usage` breakdowns; these are the source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<TokenCounts>,
}

impl allama::ProviderDataEntry for RunDetails {
    const NAMESPACE: &'static str = NAMESPACE;
    const KEY: &'static str = "run";
}

/// Clairic's token counts, as reported.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TokenCounts {
    /// Tokens of the input text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_tokens: Option<u64>,
    /// Tokens of the images.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_tokens: Option<u64>,
    /// Tokens of the field definitions, summed over the passes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema_tokens: Option<u64>,
}

/// Keys of an answer this adapter does not model (`source`, …), per field,
/// in [`ExtractionResponse::provider_data`]: kept so nothing the model
/// reports is lost, with no promise about their meaning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AnswerExtras(pub BTreeMap<String, allama::ExtraMap>);

impl allama::ProviderDataEntry for AnswerExtras {
    const NAMESPACE: &'static str = NAMESPACE;
    const KEY: &'static str = "answer_extras";
}

/// The verbatim response body, kept when
/// [`ExtractionRequestOptions::raw_response`] is set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RawResponse(pub serde_json::Value);

impl allama::ProviderDataEntry for RawResponse {
    const NAMESPACE: &'static str = NAMESPACE;
    const KEY: &'static str = allama::provider_data::RAW_RESPONSE_KEY;
}

/// Put the decoded statistics into `response`'s provider data.
fn attach_provider_data(
    response: &mut ExtractionResponse,
    statistics: wire::Statistics,
) -> Result<()> {
    let wire::Statistics {
        confidence,
        guesses,
        distributions,
        extras,
        run,
    } = statistics;
    if !confidence.is_empty() {
        response.provider_data.insert(&Confidence(confidence))?;
    }
    if !guesses.is_empty() {
        response.provider_data.insert(&Guesses(guesses))?;
    }
    if !distributions.is_empty() {
        response
            .provider_data
            .insert(&Distributions(distributions))?;
    }
    if !extras.is_empty() {
        response.provider_data.insert(&AnswerExtras(extras))?;
    }
    if run != RunDetails::default() {
        response.provider_data.insert(&run)?;
    }
    Ok(())
}

/// Model-independent capability answers: every field kind, and `Unknown`
/// for images, which are model data (ADR 0015). `allama-cloudflare-worker`
/// answers the same.
#[doc(hidden)]
pub fn builtin_extraction_capability(capability: ExtractionCapability) -> CapabilitySupport {
    match capability {
        ExtractionCapability::ImageInput => CapabilitySupport::Unknown,
        _ => CapabilitySupport::Supported,
    }
}

/// A Workers AI model name as a URL path (`@cf/cloudflare/clairic`),
/// refused when it could leave the account's `run` path the token is sent
/// to.
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
            "{model:?} is not a Workers AI model name such as `@cf/cloudflare/clairic`"
        ))
        .authored())
    }
}

fn request_id(headers: &Headers) -> Option<String> {
    ["cf-ray", "x-request-id"]
        .into_iter()
        .find_map(|name| headers.get(name))
        .filter(|value| !value.is_empty())
        .map(String::from)
}

/// Environment variable `name`, trimmed; `None` when unset or blank.
#[cfg(feature = "reqwest")]
fn optional_env_var(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
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

    #[test]
    fn the_account_id_shapes_the_url() {
        let provider = Provider::builder()
            .account_id("0123456789abcdef0123456789ABCDEF")
            .api_token("t")
            .http_transport(allama_http::ReplayTransport::new(vec![]))
            .build()
            .unwrap();
        assert_eq!(
            provider.url("@cf/cloudflare/clairic").unwrap(),
            "https://api.cloudflare.com/client/v4/accounts/0123456789abcdef0123456789ABCDEF/ai/run/@cf/cloudflare/clairic"
        );
        assert!(provider.url("../other").is_err());
        assert_eq!(provider.provider_name(), "cloudflare");
        assert!(provider.reads_namespace(NAMESPACE));
        for account in ["", "short", "0123456789abcdef0123456789abcdeg"] {
            assert!(
                Provider::builder()
                    .account_id(account)
                    .api_token("t")
                    .http_transport(allama_http::ReplayTransport::new(vec![]))
                    .build()
                    .is_err(),
                "{account:?}"
            );
        }
        assert!(Provider::builder().api_token("t").build().is_err());
    }

    #[test]
    fn a_base_url_replaces_the_account() {
        let provider = Provider::builder()
            .base_url("http://localhost:8787/run/")
            .http_transport(allama_http::ReplayTransport::new(vec![]))
            .build()
            .unwrap();
        assert_eq!(
            provider.url("@cf/cloudflare/clairic").unwrap(),
            "http://localhost:8787/run/@cf/cloudflare/clairic"
        );
        assert!(!provider.credential_slot().required);
        let scoped = provider.with_credential_scope("tenant");
        assert_eq!(scoped.credential_slot().scope, "tenant");
        assert!(scoped.credential_slot().required);
    }
}
