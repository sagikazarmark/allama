//! Presets for System One endpoints.
//!
//! Each preset configures the base URL and path, authentication, provider
//! identity, request limits, and the question kinds the endpoint supports.
//! `builder(..)` returns a [`ProviderBuilder`] for further customization.
//!
//! | Preset | Endpoint | Credential | Limits |
//! |---|---|---|---|
//! | [`TypeSafe`] | `https://api.typesafe.ai/v1/systemone` (`TYPESAFE_BASE_URL`) | `TYPESAFE_API_KEY` | 255 options, 10 levels |
//! | [`Ollama`] | `http://127.0.0.1:11434/v1/systemone` (`OLLAMA_HOST`) | none | 64 questions, 26 options, 26 levels |
//! | [`OpenRouter`] | `https://openrouter.ai/api/v1/systemone` | `OPENROUTER_API_KEY` | TypeSafe's |
//! | [`DigitalOcean`] | `https://inference.do-ai.run/v1/systemone` | `MODEL_ACCESS_KEY` | TypeSafe's |
//! | [`LiteLlm`] | `{proxy}/typesafe/v1/systemone` (`LITELLM_PROXY_URL`) | `LITELLM_API_KEY` | TypeSafe's |
//! | [`Cloudflare`] | `https://api.cloudflare.com/client/v4/accounts/{id}/ai/run/{model}` | `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` | none: each model's own |
//!
//! Gateways forward to TypeSafe, so they take TypeSafe's limits; override
//! them with [`ProviderBuilder::limits`] when a gateway routes elsewhere.
//! Workers AI runs any model at the same endpoint, each with its own input
//! schema, so the Cloudflare preset claims no limit: a model refuses what
//! its schema does not allow (ADR 0015). Set
//! [`ProviderBuilder::limits`] to check your models' limits before sending.
//!
//! Images ([`DecisionRequest::images`](allama::DecisionRequest::images))
//! go to Ollama and Cloudflare, whose models read them: which models do is
//! model data, so [`DecisionCapability::ImageInput`] is `Unknown` there.
//! TypeSafe's API takes them as a preview, opted into for each request that
//! carries one; the gateways forward its protocol without them.

use allama::{CapabilitySupport, DecisionCapability, DecisionLimits, Error, Result};

use crate::wire::Dialect;
use crate::{Provider, ProviderBuilder};

/// The question kinds System One defines, supported. Listed by name, not
/// [`DecisionCapability::ALL`], so a kind added later is not claimed before
/// an endpoint is known to serve it.
fn all_kinds(builder: ProviderBuilder) -> ProviderBuilder {
    [
        DecisionCapability::YesNo,
        DecisionCapability::Choice,
        DecisionCapability::Score,
    ]
    .into_iter()
    .fold(builder, |builder, kind| {
        builder.decision_capability(kind, CapabilitySupport::Supported)
    })
}

/// [`all_kinds`] for a gateway to TypeSafe, which forwards its protocol
/// without images.
fn typesafe_protocol(builder: ProviderBuilder) -> ProviderBuilder {
    all_kinds(builder).decision_capability(
        DecisionCapability::ImageInput,
        CapabilitySupport::Unsupported,
    )
}

fn required(name: &str, value: &str) -> Result<String> {
    let value = value.trim();
    if value.is_empty() {
        return Err(Error::configuration(format!("{name} is required")).authored());
    }
    Ok(value.to_owned())
}

/// Environment variable `name`, trimmed; `None` when unset or blank.
#[cfg(feature = "reqwest")]
fn optional_env_var(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
}

/// A secret from environment variable `name`, checked as
/// [`allama_http::config::secret`] does: a configuration error naming the
/// variable when it is unset, blank or malformed.
#[cfg(feature = "reqwest")]
fn env_secret(name: &str) -> Result<String> {
    let value = optional_env_var(name)
        .ok_or_else(|| Error::configuration(format!("{name} not set")).authored())?;
    allama_http::config::secret(name, &value)
}

/// The base URL for an `OLLAMA_HOST` value, read the way Ollama reads it
/// (`envconfig.Host` followed by `ConnectableHost`):
///
/// - no scheme means `http` on port 11434, and `ollama.com` alone means
///   `https://ollama.com:443`;
/// - an explicit `http` or `https` scheme defaults to port 80 or 443;
/// - an invalid port falls back to the default;
/// - an unspecified address (`0.0.0.0`, `::`) is reached through loopback.
///
/// Unlike Ollama, an empty host (`:1234`) becomes `127.0.0.1`, since an HTTP
/// client needs a host to connect to. This function has a twin in the other
/// crate with an Ollama preset (`allama-openai-compat` and
/// `allama-systemone`); keep them identical.
fn ollama_base_url(raw: &str) -> String {
    let value = raw.trim().trim_matches(['"', '\'']).trim();
    let (scheme, rest, default_port) = match value.split_once("://") {
        None if value == "ollama.com" => ("https", "ollama.com:443", "443"),
        None => ("http", value, "11434"),
        Some(("http", rest)) => ("http", rest, "80"),
        Some(("https", rest)) => ("https", rest, "443"),
        Some((scheme, rest)) => (scheme, rest, "11434"),
    };
    let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
    // User info (`user:password@`) is kept as it is, ahead of the host.
    let (userinfo, host_port) = match authority.rsplit_once('@') {
        Some((userinfo, host_port)) => (Some(userinfo), host_port),
        None => (None, authority),
    };
    let (host, port) = split_host_port(host_port).unwrap_or_else(|| {
        let bare = host_port.trim_matches(['[', ']']);
        if bare.parse::<std::net::IpAddr>().is_ok() {
            (bare, default_port)
        } else {
            (host_port, default_port)
        }
    });
    let port = port
        .parse::<u16>()
        .map_or(default_port.to_owned(), |port| port.to_string());
    let host = match host.parse::<std::net::IpAddr>() {
        Ok(ip) if ip.is_unspecified() && ip.is_ipv4() => "127.0.0.1".to_owned(),
        Ok(ip) if ip.is_unspecified() => "[::1]".to_owned(),
        _ if host.is_empty() => "127.0.0.1".to_owned(),
        _ if host.contains(':') => format!("[{}]", host.replace('%', "%25")),
        _ => host.to_owned(),
    };
    let path = path.trim_end_matches('/');
    let userinfo = userinfo.map_or_else(String::new, |userinfo| format!("{userinfo}@"));
    if path.is_empty() {
        format!("{scheme}://{userinfo}{host}:{port}")
    } else {
        format!("{scheme}://{userinfo}{host}:{port}/{path}")
    }
}

/// Go's `net.SplitHostPort`: `host:port` or `[host]:port`, where an
/// unbracketed host has no colon.
fn split_host_port(value: &str) -> Option<(&str, &str)> {
    if let Some(rest) = value.strip_prefix('[') {
        let (host, after) = rest.split_once(']')?;
        let port = after.strip_prefix(':')?;
        (!port.contains([':', '[', ']'])).then_some((host, port))
    } else {
        let (host, port) = value.rsplit_once(':')?;
        (!host.contains([':', '[', ']'])).then_some((host, port))
    }
}

fn typesafe_limits() -> DecisionLimits {
    DecisionLimits::new()
        .max_choice_options(255)
        .max_score_levels(10)
}

/// TypeSafe's hosted System One API (models `jev-latest`, `jev-preview`).
///
/// Jev reads images, which TypeSafe calls a preview ("danger zone"): a
/// request with images is sent the `TypeSafe-Danger-Zone-Images: true`
/// header that opts into them. Each image goes in the body's `assets` under
/// a reference, `[image-0]` and so on, and the references are appended to
/// the state: after a blank line for text, as elements of a JSON array,
/// and as an `images` array field of a JSON object (an input object with
/// its own `images` field is refused). TypeSafe takes PNG and JPEG only;
/// other media types are refused before sending.
#[derive(Debug, Clone, Copy)]
pub struct TypeSafe;

impl TypeSafe {
    /// The [`provider_name`](allama::ProviderIdentity::provider_name) this
    /// preset reports. Options and provider data use the shared
    /// [`NAMESPACE`](crate::NAMESPACE).
    pub const PROVIDER_NAME: &'static str = "typesafe";

    /// The default credential scope of this preset: its provider name.
    /// Per-request credentials addressed to it
    /// (`allama_http::HttpRequestOptions::credential`) are this preset's.
    pub const DEFAULT_CREDENTIAL_SCOPE: &'static str = Self::PROVIDER_NAME;
    /// Default base URL.
    pub const BASE_URL: &'static str = "https://api.typesafe.ai";

    /// Create a provider with an API key.
    ///
    /// # Errors
    ///
    /// Fails when the key is empty.
    #[allow(clippy::new_ret_no_self)]
    #[cfg(feature = "reqwest")]
    pub fn new(api_key: &str) -> Result<Provider> {
        Self::builder(api_key)?.build()
    }

    /// A pre-configured builder.
    ///
    /// # Errors
    ///
    /// Fails when the key is empty.
    pub fn builder(api_key: &str) -> Result<ProviderBuilder> {
        Ok(all_kinds(
            Provider::builder()
                .base_url(Self::BASE_URL)
                .bearer_token(required("api_key", api_key)?)
                .provider_name(Self::PROVIDER_NAME)
                .request_id_header_name("x-typesafe-request-id")
                .limits(typesafe_limits())
                .dialect(Dialect::TypeSafe),
        )
        .decision_capability(DecisionCapability::ImageInput, CapabilitySupport::Supported))
    }

    /// Create a provider from `TYPESAFE_API_KEY` and, when set,
    /// `TYPESAFE_BASE_URL`.
    ///
    /// # Errors
    ///
    /// Fails when `TYPESAFE_API_KEY` is not set.
    #[cfg(feature = "reqwest")]
    pub fn from_env() -> Result<Provider> {
        let mut builder = Self::builder(&env_secret("TYPESAFE_API_KEY")?)?;
        if let Some(base_url) = optional_env_var("TYPESAFE_BASE_URL") {
            builder = builder.base_url(allama_http::config::base_url(
                "TYPESAFE_BASE_URL",
                &base_url,
            )?);
        }
        builder.build()
    }
}

/// Ollama's local System One endpoint (Ollama 0.35 or later, models such as
/// `nimble` and `tev1`).
#[derive(Debug, Clone, Copy)]
pub struct Ollama;

impl Ollama {
    /// The [`provider_name`](allama::ProviderIdentity::provider_name) this
    /// preset reports. Options and provider data use the shared
    /// [`NAMESPACE`](crate::NAMESPACE).
    pub const PROVIDER_NAME: &'static str = "ollama";

    /// The default credential scope of this preset: its provider name.
    /// Per-request credentials addressed to it
    /// (`allama_http::HttpRequestOptions::credential`) are this preset's.
    pub const DEFAULT_CREDENTIAL_SCOPE: &'static str = Self::PROVIDER_NAME;

    /// A provider for `http://127.0.0.1:11434`.
    ///
    /// # Errors
    ///
    /// Never fails in practice; kept fallible for consistency.
    #[allow(clippy::new_ret_no_self)]
    #[cfg(feature = "reqwest")]
    pub fn new() -> Result<Provider> {
        Self::builder("http://127.0.0.1:11434")?.build()
    }

    /// A builder for an Ollama host: a base URL such as
    /// `http://gpu-box:11434`, or a host without a scheme in any form
    /// `OLLAMA_HOST` takes (`gpu-box`, `127.0.0.1:11434`), read the way
    /// Ollama reads it: `http` on port 11434. A trailing `/v1` is ignored.
    ///
    /// Local models can take long to load, so there is no request timeout.
    /// The default transport's connect timeout
    /// ([`allama_http::DEFAULT_CONNECT_TIMEOUT`], 10 s) still applies, so an
    /// unreachable host fails fast; bound a slow one with the builder's
    /// `timeout`.
    ///
    /// # Errors
    ///
    /// Fails when the host is empty. `build()` fails when the result is not
    /// a valid base URL.
    ///
    #[cfg_attr(
        not(feature = "reqwest"),
        doc = "[`allama_http::DEFAULT_CONNECT_TIMEOUT`]: crate#transport"
    )]
    pub fn builder(host: &str) -> Result<ProviderBuilder> {
        let host = required("host", host)?;
        let host = if host.contains("://") {
            host
        } else {
            ollama_base_url(&host)
        };
        let host = host.trim_end_matches('/');
        let host = host.strip_suffix("/v1").unwrap_or(host);
        Ok(all_kinds(
            Provider::builder()
                .base_url(host)
                .provider_name(Self::PROVIDER_NAME)
                .timeout(None)
                .dialect(Dialect::Ollama)
                .missing_endpoint_hint(
                    "Ollama serves System One from 0.35: pull a decision model such as `nimble`",
                )
                .limits(
                    DecisionLimits::new()
                        .max_questions(64)
                        .max_choice_options(26)
                        .max_score_levels(26),
                ),
        ))
    }

    /// Create a provider from `OLLAMA_HOST`, read the way Ollama reads it
    /// (default `127.0.0.1:11434`; `0.0.0.0` is reached through loopback).
    ///
    /// # Errors
    ///
    /// An authored [`Error::Configuration`] naming `OLLAMA_HOST` when it
    /// does not make a valid base URL.
    #[cfg(feature = "reqwest")]
    pub fn from_env() -> Result<Provider> {
        let host = std::env::var("OLLAMA_HOST").unwrap_or_default();
        let base_url = allama_http::config::base_url("OLLAMA_HOST", &ollama_base_url(&host))?;
        Self::builder(&base_url)?.build()
    }
}

/// OpenRouter's TypeSafe-compatible endpoint (models such as
/// `~typesafe/jev-latest`).
#[derive(Debug, Clone, Copy)]
pub struct OpenRouter;

impl OpenRouter {
    /// The [`provider_name`](allama::ProviderIdentity::provider_name) this
    /// preset reports. Options and provider data use the shared
    /// [`NAMESPACE`](crate::NAMESPACE).
    pub const PROVIDER_NAME: &'static str = "openrouter";

    /// The default credential scope of this preset: its provider name.
    /// Per-request credentials addressed to it
    /// (`allama_http::HttpRequestOptions::credential`) are this preset's.
    pub const DEFAULT_CREDENTIAL_SCOPE: &'static str = Self::PROVIDER_NAME;

    /// Create a provider with an API key.
    ///
    /// # Errors
    ///
    /// Fails when the key is empty.
    #[allow(clippy::new_ret_no_self)]
    #[cfg(feature = "reqwest")]
    pub fn new(api_key: &str) -> Result<Provider> {
        Self::builder(api_key)?.build()
    }

    /// A pre-configured builder.
    ///
    /// # Errors
    ///
    /// Fails when the key is empty.
    pub fn builder(api_key: &str) -> Result<ProviderBuilder> {
        Ok(typesafe_protocol(
            Provider::builder()
                .base_url("https://openrouter.ai/api")
                .bearer_token(required("api_key", api_key)?)
                .provider_name(Self::PROVIDER_NAME)
                .limits(typesafe_limits()),
        ))
    }

    /// Create a provider from `OPENROUTER_API_KEY`.
    ///
    /// # Errors
    ///
    /// Fails when `OPENROUTER_API_KEY` is not set.
    #[cfg(feature = "reqwest")]
    pub fn from_env() -> Result<Provider> {
        Self::new(&env_secret("OPENROUTER_API_KEY")?)
    }
}

/// DigitalOcean serverless inference (models such as
/// `typesafe-jev-1.13.0`).
#[derive(Debug, Clone, Copy)]
pub struct DigitalOcean;

impl DigitalOcean {
    /// The [`provider_name`](allama::ProviderIdentity::provider_name) this
    /// preset reports. Options and provider data use the shared
    /// [`NAMESPACE`](crate::NAMESPACE).
    pub const PROVIDER_NAME: &'static str = "digitalocean";

    /// The default credential scope of this preset: its provider name.
    /// Per-request credentials addressed to it
    /// (`allama_http::HttpRequestOptions::credential`) are this preset's.
    pub const DEFAULT_CREDENTIAL_SCOPE: &'static str = Self::PROVIDER_NAME;

    /// Create a provider with a model access key.
    ///
    /// # Errors
    ///
    /// Fails when the key is empty.
    #[allow(clippy::new_ret_no_self)]
    #[cfg(feature = "reqwest")]
    pub fn new(access_key: &str) -> Result<Provider> {
        Self::builder(access_key)?.build()
    }

    /// A pre-configured builder.
    ///
    /// # Errors
    ///
    /// Fails when the key is empty.
    pub fn builder(access_key: &str) -> Result<ProviderBuilder> {
        Ok(typesafe_protocol(
            Provider::builder()
                .base_url("https://inference.do-ai.run")
                .bearer_token(required("access_key", access_key)?)
                .provider_name(Self::PROVIDER_NAME)
                .limits(typesafe_limits()),
        ))
    }

    /// Create a provider from `MODEL_ACCESS_KEY`, the variable DigitalOcean's
    /// inference docs use.
    ///
    /// # Errors
    ///
    /// Fails when `MODEL_ACCESS_KEY` is not set.
    #[cfg(feature = "reqwest")]
    pub fn from_env() -> Result<Provider> {
        Self::new(&env_secret("MODEL_ACCESS_KEY")?)
    }
}

/// A LiteLLM proxy's TypeSafe pass-through route
/// (`/typesafe/v1/systemone`).
///
/// The proxy forwards every path under `/typesafe/` to TypeSafe, adding
/// the `TYPESAFE_API_KEY` from its own environment, so the client sends
/// only a LiteLLM key. A proxy without the pass-through answers `404`,
/// which fails with `Error::Unsupported` and says so.
#[derive(Debug, Clone, Copy)]
pub struct LiteLlm;

impl LiteLlm {
    /// The [`provider_name`](allama::ProviderIdentity::provider_name) this
    /// preset reports. Options and provider data use the shared
    /// [`NAMESPACE`](crate::NAMESPACE).
    pub const PROVIDER_NAME: &'static str = "litellm";

    /// The default credential scope of this preset: its provider name.
    /// Per-request credentials addressed to it
    /// (`allama_http::HttpRequestOptions::credential`) are this preset's.
    pub const DEFAULT_CREDENTIAL_SCOPE: &'static str = Self::PROVIDER_NAME;

    /// Create a provider for the proxy at `base_url`.
    ///
    /// # Errors
    ///
    /// Fails when the base URL is empty.
    #[allow(clippy::new_ret_no_self)]
    #[cfg(feature = "reqwest")]
    pub fn new(base_url: &str, api_key: Option<&str>) -> Result<Provider> {
        Self::builder(base_url, api_key)?.build()
    }

    /// A pre-configured builder.
    ///
    /// # Errors
    ///
    /// Fails when the base URL is empty.
    pub fn builder(base_url: &str, api_key: Option<&str>) -> Result<ProviderBuilder> {
        let mut builder = Provider::builder()
            .base_url(required("base_url", base_url)?)
            .path("/typesafe/v1/systemone")
            .provider_name(Self::PROVIDER_NAME)
            .missing_endpoint_hint(
                "LiteLLM serves System One through its TypeSafe pass-through \
                 (`/typesafe/...`): run a proxy release that has it, with `TYPESAFE_API_KEY` \
                 set in the proxy's environment",
            )
            .limits(typesafe_limits());
        if let Some(key) = api_key.map(str::trim).filter(|key| !key.is_empty()) {
            builder = builder.bearer_token(key);
        }
        Ok(typesafe_protocol(builder))
    }

    /// Create a provider from `LITELLM_PROXY_URL` (default
    /// `http://127.0.0.1:4000` when unset or blank) and, when set,
    /// `LITELLM_API_KEY`.
    ///
    /// # Errors
    ///
    /// An authored [`Error::Configuration`] naming the variable when
    /// `LITELLM_PROXY_URL` is not a valid base URL or `LITELLM_API_KEY`
    /// holds a value no HTTP header can carry.
    #[cfg(feature = "reqwest")]
    pub fn from_env() -> Result<Provider> {
        let base_url = match optional_env_var("LITELLM_PROXY_URL") {
            Some(url) => allama_http::config::base_url("LITELLM_PROXY_URL", &url)?,
            None => "http://127.0.0.1:4000".to_owned(),
        };
        let api_key = optional_env_var("LITELLM_API_KEY")
            .map(|key| allama_http::config::secret("LITELLM_API_KEY", &key))
            .transpose()?;
        Self::new(&base_url, api_key.as_deref())
    }
}

/// Cloudflare Workers AI over HTTP: Clef decision models
/// (`@cf/cloudflare/clef`, `@cf/cloudflare/clef-flash`).
///
/// Clef speaks System One with Cloudflare's extensions, and the preset
/// handles the differences:
///
/// - each model has its own URL, `{url}/{model}`
///   (`.../ai/run/@cf/cloudflare/clef`), and the body's `model` is the
///   selector Clef reads, the last segment of the name (`clef`);
/// - images go as `{"content_type", "base64"}`, inline only (Clef takes no
///   URL);
/// - the output comes in Cloudflare's `{"result": ..}` envelope, and
///   errors in its `{"errors": [..]}` one, with Workers AI's error codes.
///
/// What a model's input schema allows (Clef: 64 questions, 255 options, 10
/// levels, 4 PNG, JPEG or WebP images, question names of at most 100 of
/// `[A-Za-z0-9_.-]`) is model data, so the preset does not check it: the
/// model refuses what it does not take with HTTP 422, an
/// [`Error::InvalidRequest`] that names the field (ADR 0015).
///
/// Its provider name, `cloudflare`, is the one
/// `allama_openai_compat::providers::Cloudflare` and
/// `allama-cloudflare-worker` report, so chat, embeddings and decisions on
/// Workers AI share one `gen_ai.provider.name`. Options use the shared
/// [`NAMESPACE`](crate::NAMESPACE), `systemone`.
///
/// Inside a Worker, `allama-cloudflare-worker` runs the same models through
/// the AI binding.
#[derive(Debug, Clone, Copy)]
pub struct Cloudflare;

impl Cloudflare {
    /// The [`provider_name`](allama::ProviderIdentity::provider_name) this
    /// preset reports. Options and provider data use the shared
    /// [`NAMESPACE`](crate::NAMESPACE).
    pub const PROVIDER_NAME: &'static str = "cloudflare";

    /// The default credential scope of this preset: its provider name, the
    /// one `allama-openai-compat`'s Cloudflare preset uses, so one
    /// per-request credential (`allama_http::HttpRequestOptions::credential`)
    /// reaches the account for chat and decisions alike.
    pub const DEFAULT_CREDENTIAL_SCOPE: &'static str = Self::PROVIDER_NAME;

    /// Create a provider.
    ///
    /// # Errors
    ///
    /// As [`builder`](Self::builder).
    #[allow(clippy::new_ret_no_self)]
    #[cfg(feature = "reqwest")]
    pub fn new(account_id: &str, api_token: &str) -> Result<Provider> {
        Self::builder(account_id, api_token)?.build()
    }

    /// A pre-configured builder.
    ///
    /// # Errors
    ///
    /// An authored [`Error::Configuration`] when the token is empty, or the
    /// account id is not 32 hexadecimal digits: anything else would change
    /// the path the token is sent to.
    pub fn builder(account_id: &str, api_token: &str) -> Result<ProviderBuilder> {
        let account_id = required("account_id", account_id)?;
        if account_id.len() != 32 || !account_id.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(Error::configuration(
                "the Cloudflare account id must be 32 hexadecimal digits",
            )
            .authored());
        }
        Ok(all_kinds(
            Provider::builder()
                .base_url(format!(
                    "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai"
                ))
                .path("/run")
                .bearer_token(required("api_token", api_token)?)
                .provider_name(Self::PROVIDER_NAME)
                .dialect(Dialect::WorkersAi),
        ))
    }

    /// Create a provider from `CLOUDFLARE_ACCOUNT_ID` and
    /// `CLOUDFLARE_API_TOKEN`, the variables
    /// `allama_openai_compat::providers::Cloudflare` reads.
    ///
    /// # Errors
    ///
    /// Fails when either variable is missing or malformed.
    #[cfg(feature = "reqwest")]
    pub fn from_env() -> Result<Provider> {
        let account_id = optional_env_var("CLOUDFLARE_ACCOUNT_ID")
            .ok_or_else(|| Error::configuration("CLOUDFLARE_ACCOUNT_ID not set").authored())?;
        Self::new(&account_id, &env_secret("CLOUDFLARE_API_TOKEN")?)
    }
}

#[cfg(test)]
mod tests {
    use allama::{DecisionProvider, ProviderIdentity};

    use super::*;

    /// Ollama's `TestHost` and `TestConnectableHost` tables
    /// (`envconfig/config_test.go`), as connectable URLs.
    #[test]
    fn ollama_host_is_read_like_ollama() {
        for (raw, expected) in [
            ("", "http://127.0.0.1:11434"),
            ("1.2.3.4", "http://1.2.3.4:11434"),
            // Ollama keeps the empty host; a client needs one.
            (":1234", "http://127.0.0.1:1234"),
            ("1.2.3.4:1234", "http://1.2.3.4:1234"),
            ("example.com", "http://example.com:11434"),
            ("example.com:1234", "http://example.com:1234"),
            (":0", "http://127.0.0.1:0"),
            (":66000", "http://127.0.0.1:11434"),
            (":-1", "http://127.0.0.1:11434"),
            ("[::1]", "http://[::1]:11434"),
            ("[::]", "http://[::1]:11434"),
            ("::1", "http://[::1]:11434"),
            ("[::1]:1337", "http://[::1]:1337"),
            (" 1.2.3.4 ", "http://1.2.3.4:11434"),
            ("\"1.2.3.4\"", "http://1.2.3.4:11434"),
            (" \" 1.2.3.4 \" ", "http://1.2.3.4:11434"),
            ("'1.2.3.4'", "http://1.2.3.4:11434"),
            ("http://1.2.3.4", "http://1.2.3.4:80"),
            ("http://1.2.3.4:4321", "http://1.2.3.4:4321"),
            ("https://1.2.3.4", "https://1.2.3.4:443"),
            ("https://1.2.3.4:4321", "https://1.2.3.4:4321"),
            (
                "https://example.com/ollama",
                "https://example.com:443/ollama",
            ),
            ("ollama.com", "https://ollama.com:443"),
            ("0.0.0.0", "http://127.0.0.1:11434"),
            ("0.0.0.0:1234", "http://127.0.0.1:1234"),
            ("[::]:1234", "http://[::1]:1234"),
            ("192.168.1.5:8080", "http://192.168.1.5:8080"),
            ("https://0.0.0.0:4321", "https://127.0.0.1:4321"),
            // Malformed ports fall back to the default, as in Ollama.
            ("http://host:99999", "http://host:80"),
            ("host:abc", "http://host:11434"),
            ("[::1]:", "http://[::1]:11434"),
            ("[fe80::1%en0]:1234", "http://[fe80::1%25en0]:1234"),
        ] {
            assert_eq!(ollama_base_url(raw), expected, "OLLAMA_HOST={raw:?}");
        }
    }

    #[test]
    fn presets() {
        let typesafe = TypeSafe::builder("key")
            .unwrap()
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(
            typesafe.transport().url(),
            "https://api.typesafe.ai/v1/systemone"
        );
        assert_eq!(
            typesafe.transport().auth,
            Some(("authorization".into(), "Bearer key".into()))
        );
        assert_eq!(
            typesafe.decision_limits("jev-latest").max_score_levels,
            Some(10)
        );
        assert_eq!(
            typesafe.decision_capability("jev-latest", DecisionCapability::YesNo),
            CapabilitySupport::Supported
        );
        assert!(TypeSafe::builder(" ").is_err());

        let ollama = Ollama::builder("http://gpu-box:11434/")
            .unwrap()
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(
            Ollama::builder("http://gpu-box:11434/v1")
                .unwrap()
                .http_transport(allama_http::ReplayTransport::default())
                .build()
                .unwrap()
                .transport()
                .url(),
            "http://gpu-box:11434/v1/systemone"
        );
        assert_eq!(
            ollama.transport().url(),
            "http://gpu-box:11434/v1/systemone"
        );
        assert_eq!(ollama.transport().auth, None);
        assert_eq!(ollama.transport().timeout, None);
        assert_eq!(ollama.decision_limits("nimble").max_questions, Some(64));
        assert_eq!(ollama.provider_name(), "ollama");

        let openrouter = OpenRouter::builder("key")
            .unwrap()
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(
            openrouter.transport().url(),
            "https://openrouter.ai/api/v1/systemone"
        );

        let digitalocean = DigitalOcean::builder("key")
            .unwrap()
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(
            digitalocean.transport().url(),
            "https://inference.do-ai.run/v1/systemone"
        );

        let litellm = LiteLlm::builder("http://127.0.0.1:4000", None)
            .unwrap()
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(
            litellm.transport().url(),
            "http://127.0.0.1:4000/typesafe/v1/systemone"
        );
        assert_eq!(litellm.transport().auth, None);
    }

    /// Regression (usability re-run): the builder refused the scheme-less
    /// host `OLLAMA_HOST` holds, which `from_env` accepts.
    #[test]
    fn the_builder_takes_an_ollama_host_value() {
        let url = |host: &str| {
            Ollama::builder(host)
                .unwrap()
                .http_transport(allama_http::ReplayTransport::default())
                .build()
                .unwrap()
                .transport()
                .url()
                .to_owned()
        };
        assert_eq!(url("gpu-box"), "http://gpu-box:11434/v1/systemone");
        assert_eq!(url("127.0.0.1:1234"), "http://127.0.0.1:1234/v1/systemone");
        assert_eq!(url("0.0.0.0"), "http://127.0.0.1:11434/v1/systemone");
        // A URL is taken as it is.
        assert_eq!(
            url("https://gpu-box/ollama"),
            "https://gpu-box/ollama/v1/systemone"
        );
    }

    /// Regression: only the local Ollama preset
    /// drops the request timeout; the hosted presets and the LiteLLM proxy
    /// keep the default, as the `allama-providers` latency table says.
    #[test]
    fn only_ollama_drops_the_request_timeout() {
        let timeout = |builder: Result<ProviderBuilder>| {
            builder
                .unwrap()
                .http_transport(allama_http::ReplayTransport::default())
                .build()
                .unwrap()
                .transport()
                .timeout
        };
        let default = Some(allama_http::DEFAULT_REQUEST_TIMEOUT);
        assert_eq!(timeout(Ollama::builder("http://127.0.0.1:11434")), None);
        assert_eq!(timeout(TypeSafe::builder("key")), default);
        assert_eq!(timeout(OpenRouter::builder("key")), default);
        assert_eq!(timeout(DigitalOcean::builder("key")), default);
        assert_eq!(
            timeout(LiteLlm::builder("http://127.0.0.1:4000", None)),
            default
        );
    }

    /// The account id goes in the path the token is sent to, so it is
    /// checked; limits and images are model data.
    #[test]
    fn cloudflare_preset() {
        for account in [
            "",
            "0123",
            "0123456789abcdef0123456789abcdeg",
            "../0123456789abcdef0123456789ab",
        ] {
            assert!(Cloudflare::builder(account, "t").is_err(), "{account:?}");
        }
        assert!(Cloudflare::builder("0123456789abcdef0123456789abcdef", " ").is_err());
        let provider = Cloudflare::builder("0123456789ABCDEF0123456789abcdef", "t")
            .unwrap()
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(
            provider.url(),
            "https://api.cloudflare.com/client/v4/accounts/0123456789ABCDEF0123456789abcdef/ai/run"
        );
        assert_eq!(provider.provider_name(), "cloudflare");
        let model = "@cf/cloudflare/clef";
        // Limits are each model's own (ADR 0015).
        assert_eq!(provider.decision_limits(model), DecisionLimits::new());
        for kind in [
            DecisionCapability::YesNo,
            DecisionCapability::Choice,
            DecisionCapability::Score,
        ] {
            assert_eq!(
                provider.decision_capability(model, kind),
                CapabilitySupport::Supported
            );
        }
        assert_eq!(
            provider.decision_capability(model, DecisionCapability::ImageInput),
            CapabilitySupport::Unknown
        );
        let typesafe = TypeSafe::builder("k")
            .unwrap()
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(
            typesafe.decision_capability("jev-latest", DecisionCapability::ImageInput),
            CapabilitySupport::Supported
        );
        let gateway = OpenRouter::builder("k")
            .unwrap()
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(
            gateway.decision_capability("typesafe/jev-1.13", DecisionCapability::ImageInput),
            CapabilitySupport::Unsupported
        );
    }
}
