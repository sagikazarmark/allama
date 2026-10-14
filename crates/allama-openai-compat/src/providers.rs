//! Presets for common OpenAI-compatible endpoints.
//!
//! Each preset configures the base URL, authentication, provider identity,
//! dialect [`CompatQuirks`], and the capabilities it is
//! known to support (everything else answers `Unknown`). Presets return a
//! [`ProviderBuilder`] from `builder(..)` for further
//! customization.
//!
//! Presets build Chat Completions providers. [`Azure`] also builds a
//! [`ResponsesProvider`] for the Responses API (`Azure::responses_builder`).
//!
//! Each preset's credential scope defaults to its provider name (its
//! `DEFAULT_CREDENTIAL_SCOPE`), so a per-request
//! `allama_http::Credential` addressed to `"groq"` reaches Groq only.
//!
//! | Preset | Base URL | Credential |
//! |---|---|---|
//! | [`Azure`] | `https://{resource}.openai.azure.com/openai/v1`, or `{endpoint}/openai/v1` (`AZURE_OPENAI_ENDPOINT`) | `AZURE_OPENAI_API_KEY` (`api-key` header) |
//! | [`Cloudflare`] | `https://api.cloudflare.com/client/v4/accounts/{id}/ai/v1` | `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` |
//! | [`DeepInfra`] | `https://api.deepinfra.com/v1/openai` | `DEEPINFRA_API_KEY` |
//! | [`DeepSeek`] | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` |
//! | [`Fireworks`] | `https://api.fireworks.ai/inference/v1` | `FIREWORKS_API_KEY` |
//! | [`GitHubCopilot`] | `https://api.githubcopilot.com` (`GITHUB_COPILOT_BASE_URL`) | `COPILOT_GITHUB_TOKEN` |
//! | [`Groq`] | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` |
//! | [`Mistral`] | `https://api.mistral.ai/v1` | `MISTRAL_API_KEY` |
//! | [`Ollama`] | `http://localhost:11434/v1` (`OLLAMA_HOST`) | none |
//! | [`OpenRouter`] | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
//! | [`Together`] | `https://api.together.xyz/v1` | `TOGETHER_API_KEY` |
//! | [`Vllm`] | `http://localhost:8000/v1` (`VLLM_BASE_URL`) | optional `VLLM_API_KEY` |
//! | [`XAi`] | `https://api.x.ai/v1` | `XAI_API_KEY` |
//!
//! # Options and provider data
//!
//! Every preset reads request options under the shared
//! [`NAMESPACE`](crate::NAMESPACE) (`"openai-compat"`), so one
//! [`ChatRequestOptions`](crate::ChatRequestOptions) value configures any of
//! them. Each preset stores the state it mints under its own namespace
//! (`"openai-compat:<name>"`, the preset's `NAMESPACE`), so state is never
//! replayed to a different endpoint.
//!
//! ```
//! use allama::prelude::*;
//! use allama_http::{ReplayResponse, ReplayTransport};
//! use allama_openai_compat::providers::Groq;
//! use serde_json::json;
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> allama::Result<()> {
//! let transport = ReplayTransport::new([ReplayResponse::json(
//!     200,
//!     r#"{"id":"c1","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}"#,
//! )]);
//! let groq = Groq::builder("key")?.http_transport(transport.clone()).build()?;
//!
//! let request = ChatRequest::new("m")
//!     .user("Hi")
//!     .with_options_json(allama_openai_compat::NAMESPACE, json!({"service_tier": "flex"}));
//! groq.chat(&request).await?;
//!
//! let body: serde_json::Value =
//!     serde_json::from_str(transport.requests()[0].body_text().unwrap()).unwrap();
//! assert_eq!(body["service_tier"], "flex");
//! assert_eq!(Groq::NAMESPACE, "openai-compat:groq");
//! # Ok(()) }
//! ```

use std::borrow::Cow;

use allama::{
    CapabilitySupport, ChatCapability, EmbeddingCapability, Error, ReasoningEffort, Result,
};
use allama_http::CredentialPlacement;

use crate::schema::SchemaDialect;
use crate::{
    ChatModelRules, CompatQuirks, EmbeddingInputTypeField, MaxTokensField, Provider,
    ProviderBuilder, ReasoningReplayField, ResponsesProvider, ResponsesProviderBuilder,
    ResponsesQuirks, SchemaConstraint, SeedField, StreamTimeouts, ToolCallIdFormat,
};

/// A Workers AI model id of a Mistral model: `@cf/mistralai/…`,
/// `@cf/mistral/…` or `@hf/mistral/…`.
fn workers_ai_mistral(model: &str) -> bool {
    let mut segments = model.split('/');
    matches!(segments.next(), Some("@cf" | "@hf"))
        && segments
            .next()
            .is_some_and(|vendor| vendor.starts_with("mistral"))
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

/// Environment variable `name`, trimmed: a configuration error naming it
/// when unset or blank.
#[cfg(feature = "reqwest")]
fn env_var(name: &str) -> Result<String> {
    optional_env_var(name).ok_or_else(|| Error::configuration(format!("{name} not set")).authored())
}

/// A secret from environment variable `name`, checked as
/// [`allama_http::config::secret`] does, naming the variable.
#[cfg(feature = "reqwest")]
fn env_secret(name: &str) -> Result<String> {
    allama_http::config::secret(name, &env_var(name)?)
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

/// Whether `name` is a DNS label: 2 to 63 ASCII letters, digits and
/// hyphens, starting and ending with a letter or digit.
fn is_dns_label(name: &str) -> bool {
    (2..=63).contains(&name.len())
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
        && !name.starts_with('-')
        && !name.ends_with('-')
}

/// Whether `base_url` is Ollama Cloud (`ollama.com`), whose models do not
/// support structured outputs.
fn is_ollama_cloud(base_url: &str) -> bool {
    let rest = base_url
        .split_once("://")
        .map_or(base_url, |(_, rest)| rest);
    let authority = rest.split('/').next().unwrap_or_default();
    let host = split_host_port(authority).map_or(authority, |(host, _)| host);
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    host == "ollama.com" || host.ends_with(".ollama.com")
}

/// Whether a checked base URL ([`allama_http::config::base_url`]) is
/// `https://`, or `http://` on a loopback host (`localhost`, `127.0.0.0/8`,
/// `::1`).
fn is_https_or_loopback(base_url: &str) -> bool {
    let Some((scheme, rest)) = base_url.split_once("://") else {
        return false;
    };
    if scheme.eq_ignore_ascii_case("https") {
        return true;
    }
    let authority = rest.split('/').next().unwrap_or_default();
    // User info (`user:password@`) is not the host.
    let host_port = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    let host = split_host_port(host_port).map_or(host_port, |(host, _)| host);
    let host = host.trim_matches(['[', ']']);
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
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

fn streaming() -> [(ChatCapability, CapabilitySupport); 2] {
    [
        (ChatCapability::Streaming, CapabilitySupport::Supported),
        (
            ChatCapability::NativeStreaming,
            CapabilitySupport::Supported,
        ),
    ]
}

/// Ollama: tool support depends on the model (its template), so it is not
/// claimed, and `tool_choice` is not among the request fields Ollama's
/// OpenAI compatibility reads
/// (<https://docs.ollama.com/api/openai-compatibility>), so a forced tool
/// choice would be silently ignored. Its message parser rejects a `file`
/// part for every model, before it looks the model up ("invalid message
/// format"), so file input is refused before sending; image and audio
/// parts pass the parser, and the model decides.
fn ollama_capabilities() -> [(ChatCapability, CapabilitySupport); 4] {
    let [streaming, native] = streaming();
    [
        streaming,
        native,
        (
            ChatCapability::ForcedToolChoice,
            CapabilitySupport::Unsupported,
        ),
        (ChatCapability::FileInput, CapabilitySupport::Unsupported),
    ]
}

impl ProviderBuilder {
    /// Declare each of a preset's chat capability answers, in order.
    fn declare_chat_capabilities(
        self,
        answers: impl IntoIterator<Item = (ChatCapability, CapabilitySupport)>,
    ) -> Self {
        answers
            .into_iter()
            .fold(self, |builder, (capability, support)| {
                builder.chat_capability(capability, support)
            })
    }
}

impl ResponsesProviderBuilder {
    /// Declare each of a preset's chat capability answers, in order.
    fn declare_chat_capabilities(
        self,
        answers: impl IntoIterator<Item = (ChatCapability, CapabilitySupport)>,
    ) -> Self {
        answers
            .into_iter()
            .fold(self, |builder, (capability, support)| {
                builder.chat_capability(capability, support)
            })
    }
}

/// A hosted preset: streaming is the endpoint's, and nothing else is
/// claimed. Tool calls and structured output depend on the model a hosted
/// endpoint routes to, so they answer `Unknown` (ADR 0015): the request is
/// sent and the endpoint decides, and extraction keeps its native →
/// forced-tool fallback.
fn hosted(
    base_url: &str,
    api_key: &str,
    name: &'static str,
    quirks: CompatQuirks,
) -> Result<ProviderBuilder> {
    // An endpoint whose streams are not asked for usage may send it anyway
    // (Mistral's last chunk carries it): `Unknown`, not `Unsupported`.
    let stream_usage =
        (!quirks.stream_usage).then_some((ChatCapability::StreamUsage, CapabilitySupport::Unknown));
    Ok(Provider::builder()
        .base_url(base_url)
        .bearer_token(required("api_key", api_key)?)
        .provider_name(name)
        .quirks(quirks)
        .declare_chat_capabilities(streaming())
        .declare_chat_capabilities(stream_usage))
}

macro_rules! bearer_preset {
    ($(#[$doc:meta])* $name:ident, $id:literal, $url:literal, $env:literal, $quirks:expr) => {
        bearer_preset!($(#[$doc])* $name, $id, $url, $env, $quirks, |builder| builder);
    };
    ($(#[$doc:meta])* $name:ident, $id:literal, $url:literal, $env:literal, $quirks:expr, $configure:expr) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy)]
        pub struct $name;

        impl $name {
            /// The [`provider_name`](allama::ProviderIdentity::provider_name)
            /// this preset reports.
            pub const PROVIDER_NAME: &'static str = $id;

            /// The default credential scope of this preset's providers, its
            /// provider name: per-request credentials addressed to it
            /// (`allama_http::HttpRequestOptions::credential`) are theirs.
            /// Change it with [`ProviderBuilder::credential_scope`].
            pub const DEFAULT_CREDENTIAL_SCOPE: &'static str = $id;

            /// The provider-data namespace of the state this preset mints
            /// (`openai-compat:<provider name>`). Request options use the
            /// shared [`NAMESPACE`](crate::NAMESPACE).
            pub const NAMESPACE: &'static str = concat!("openai-compat:", $id);

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
                let configure: fn(ProviderBuilder) -> ProviderBuilder = $configure;
                hosted($url, api_key, Self::PROVIDER_NAME, $quirks).map(configure)
            }

            /// Create a provider from the environment.
            ///
            /// # Errors
            ///
            #[doc = concat!("Fails when `", $env, "` is not set, or holds a value no HTTP header can carry; the error names the variable.")]
            #[cfg(feature = "reqwest")]
            pub fn from_env() -> Result<Provider> {
                Self::new(&env_secret($env)?)
            }
        }
    };
}

bearer_preset!(
    /// Groq.
    Groq, "groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY",
    {
        let mut q = CompatQuirks::new(Self::NAMESPACE);
        q.max_tokens_field = MaxTokensField::MaxCompletionTokens;
        // Groq's strict mode documents OpenAI's rules (closed objects, every
        // property required, `anyOf`, `$defs`), so the default OpenAI
        // dialect stays, and "with `strict: true`, the model uses
        // constrained decoding to guarantee that the output will always
        // match your schema exactly"
        // (https://console.groq.com/docs/structured-outputs). Which models
        // take strict mode is model data (ADR 0015): the endpoint's answer
        // holds for every model, and a model list belongs in the catalog.
        q.dialect_verified = true;
        q
    }
);

bearer_preset!(
    /// Together AI.
    Together, "together", "https://api.together.xyz/v1", "TOGETHER_API_KEY",
    {
        let mut q = CompatQuirks::new(Self::NAMESPACE);
        // https://docs.together.ai/reference/chat-completions-1 (`top_k`).
        q.supports_top_k = true;
        q.response_format_constraint = SchemaConstraint::Never;
        q.tool_constraint = SchemaConstraint::Never;
        q.supports_parallel_tool_calls = false;
        q
    }
);

bearer_preset!(
    /// Fireworks AI.
    Fireworks, "fireworks", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY",
    {
        let mut q = CompatQuirks::new(Self::NAMESPACE);
        // https://docs.fireworks.ai/api-reference/post-chatcompletions (`top_k`).
        q.supports_top_k = true;
        q.response_format_constraint = SchemaConstraint::Never;
        q.tool_constraint = SchemaConstraint::Never;
        q.supports_parallel_tool_calls = false;
        q
    }
);

bearer_preset!(
    /// DeepInfra. The embedding input type is sent as `input_type`
    /// (`query` or `document`), which asymmetric retrieval models use.
    DeepInfra, "deepinfra", "https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY",
    {
        let mut q = CompatQuirks::new(Self::NAMESPACE);
        // https://deepinfra.com/docs/openai_api (`top_k`).
        q.supports_top_k = true;
        q.response_format_constraint = SchemaConstraint::Never;
        q.tool_constraint = SchemaConstraint::Never;
        q.supports_parallel_tool_calls = false;
        q.supports_reasoning_effort = false;
        // "Role hint for asymmetric retrieval models: 'query' embeds a search
        // query, 'passage'/'document' embeds a document."
        q.embedding_input_type =
            Some(EmbeddingInputTypeField::new("input_type", "query", "document"));
        q
    }
);

bearer_preset!(
    /// OpenRouter. Reasoning text arrives in the `reasoning` field, and
    /// the structured `reasoning_details` (encrypted reasoning and
    /// signatures included) go back in later turns; the embedding input
    /// type is sent as `input_type` (`search_query` or `search_document`),
    /// for the routed model to use.
    OpenRouter, "openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
    {
        let mut q = CompatQuirks::new(Self::NAMESPACE);
        // "Pass `reasoning_details` back in subsequent requests to preserve
        // reasoning continuity" for tool use with reasoning models
        // (https://openrouter.ai/docs/use-cases/reasoning-tokens).
        q.reasoning_replay = ReasoningReplayField::ReasoningDetails;
        // https://openrouter.ai/docs/api-reference/parameters (`top_k`).
        q.supports_top_k = true;
        // Strict schemas stay rewritten for OpenAI's rules, but are not
        // verified: "Enforcement varies by provider: some guarantee
        // schema-conforming output, while others translate your schema into
        // their own structured-output format or treat it as a strong hint"
        // (https://openrouter.ai/docs/features/structured-outputs).
        // "The type of input (e.g. search_query, search_document)".
        q.embedding_input_type = Some(EmbeddingInputTypeField::new(
            "input_type",
            "search_query",
            "search_document",
        ));
        q
    }
);

bearer_preset!(
    /// DeepSeek. JSON mode only (no JSON Schema), so
    /// [`ChatCapability::StructuredOutput`] answers `Unsupported` and
    /// `ResponseFormat::Json` is sent; reasoning text arrives in
    /// `reasoning_content` and is replayed there, as thinking-mode tool
    /// calls need it.
    DeepSeek, "deepseek", "https://api.deepseek.com", "DEEPSEEK_API_KEY",
    {
        let mut q = CompatQuirks::new(Self::NAMESPACE);
        // Thinking-mode tool calls need the turn's `reasoning_content`
        // back, or the API answers 400
        // (https://api-docs.deepseek.com/guides/thinking_mode).
        q.reasoning_replay = ReasoningReplayField::ReasoningContent;
        q.supports_json_schema = false;
        q.response_format_constraint = SchemaConstraint::Never;
        q.tool_constraint = SchemaConstraint::Never;
        q.supports_reasoning_effort = false;
        q.supports_parallel_tool_calls = false;
        q
    }
);

bearer_preset!(
    /// xAI (Grok).
    ///
    /// Usage is decoded as for OpenAI. Whether xAI's `completion_tokens`
    /// includes reasoning tokens, as [`Usage::output_tokens`](allama::Usage::output_tokens)
    /// must, is not verified live: if it does not,
    /// `output_tokens` undercounts reasoning models.
    XAi, "xai", "https://api.x.ai/v1", "XAI_API_KEY",
    {
        let mut q = CompatQuirks::new(Self::NAMESPACE);
        // "When using supported schema features, the response is guaranteed
        // to match your schema" (https://docs.x.ai/docs/guides/structured-outputs).
        // OpenAI's strict rules are a subset of those features.
        q.dialect_verified = true;
        q
    }
);

bearer_preset!(
    /// Mistral. The seed is sent as `random_seed`; Magistral's `thinking`
    /// content parts arrive as reasoning.
    ///
    /// The reasoning effort, `top_k` and the prompt cache key are sent, as
    /// Mistral's chat API reference documents them
    /// (<https://docs.mistral.ai/api/endpoint/chat>); a prompt cache
    /// retention is not, since the API has none. Which models reason is
    /// model data, so [`ChatCapability::ReasoningConfig`] answers `Unknown`:
    /// the effort is sent, and the model decides. The [model
    /// rules](Self::model_rules) map the effort for the models that take
    /// fewer levels. These shapes are built from the documentation, not
    /// verified live.
    ///
    /// Reasoning is not replayed. Mistral asks for the assistant message
    /// back with its `thinking` chunks
    /// (<https://docs.mistral.ai/capabilities/reasoning>), but the preset
    /// keeps [`ReasoningReplayField::Disabled`]: no field sends them, so a
    /// later turn drops the reasoning Mistral returned, silently, as its
    /// own. Whether the model then reasons less well across turns is not
    /// verified live.
    Mistral, "mistral", "https://api.mistral.ai/v1", "MISTRAL_API_KEY",
    {
        let mut q = CompatQuirks::new(Self::NAMESPACE);
        q.seed_field = SeedField::RandomSeed;
        // `reasoning_effort` ("none" to "xhigh"), `top_k` and
        // `prompt_cache_key` are in the request schema; `prompt_cache_retention`
        // is not (https://docs.mistral.ai/api/endpoint/chat).
        q.supports_reasoning_effort = true;
        q.supports_top_k = true;
        q.supports_prompt_cache_key = true;
        q.supports_prompt_cache_retention = false;
        // Mistral's chat API takes no `stream_options` (its request schema
        // has none, and unknown fields are refused); the last chunk of a
        // stream carries the usage anyway.
        q.stream_usage = false;
        // Tool call ids are exactly nine letters and digits: Mistral's
        // `mistral_common` validator refuses others
        // (`^[a-zA-Z0-9]{9}$`, seen live through Workers AI: "Tool call id
        // was … but must be a-z, A-Z, 0-9, with a length of 9").
        q.tool_call_id_format = ToolCallIdFormat::Alphanumeric(9);
        q
    },
    |builder| builder.model_rules(Mistral::model_rules)
);

impl Mistral {
    /// The [model rules](ProviderBuilder::model_rules) this preset installs.
    ///
    /// `zai-glm-5-3` takes the efforts `low`, `high` and `max`, "not
    /// `none`" (<https://docs.mistral.ai/capabilities/reasoning>), so a
    /// portable effort is sent as the nearest of `low` and `high`, with an
    /// `Approximated` warning (`max` has no portable level; send it with
    /// [`ChatRequestOptions::reasoning_effort`](crate::ChatRequestOptions::reasoning_effort)).
    /// Other models take the effort as requested. Rules of your own replace
    /// these; start them from this function to keep the mapping.
    #[must_use]
    pub fn model_rules(model: &str) -> ChatModelRules {
        if model.starts_with("zai-glm-5-3") {
            ChatModelRules::new().effort_levels(&[ReasoningEffort::Low, ReasoningEffort::High])
        } else {
            ChatModelRules::new()
        }
    }
}

/// Ollama's OpenAI-compatible endpoint (no authentication).
///
/// File input ([`FileInput`](ChatCapability::FileInput)) answers
/// `Unsupported`: the endpoint rejects a `file` part for every model, so a
/// PDF fails before sending, where a fallback can route it, instead of as
/// an opaque `InvalidRequest`.
///
/// Requests have no total or stream timeouts, streaming or not: a local
/// server may load the model before it answers. The default transport's
/// connect timeout ([`allama_http::DEFAULT_CONNECT_TIMEOUT`], 10 s) still
/// applies, so an unreachable host fails fast; bound a slow one with the
/// builder's `timeout` and `stream_timeouts`, or a
/// `TimeoutProvider` around it.
///
#[cfg_attr(
    not(feature = "reqwest"),
    doc = "[`allama_http::DEFAULT_CONNECT_TIMEOUT`]: crate#transport"
)]
#[derive(Debug, Clone, Copy)]
pub struct Ollama;

impl Ollama {
    /// The [`provider_name`](allama::ProviderIdentity::provider_name) this
    /// preset reports.
    pub const PROVIDER_NAME: &'static str = "ollama";

    /// The default credential scope of this preset's providers, its
    /// provider name: per-request credentials addressed to it
    /// (`allama_http::HttpRequestOptions::credential`) are theirs. Change
    /// it with [`ProviderBuilder::credential_scope`].
    pub const DEFAULT_CREDENTIAL_SCOPE: &'static str = "ollama";

    /// The provider-data namespace of the state this preset mints. Request
    /// options use the shared [`NAMESPACE`](crate::NAMESPACE).
    pub const NAMESPACE: &'static str = "openai-compat:ollama";

    /// A provider for `http://localhost:11434/v1`.
    ///
    /// # Errors
    ///
    /// Never fails in practice; kept fallible for consistency.
    #[allow(clippy::new_ret_no_self)]
    #[cfg(feature = "reqwest")]
    pub fn new() -> Result<Provider> {
        Self::builder("http://localhost:11434")?.build()
    }

    /// A builder for an Ollama host: a base URL such as
    /// `http://gpu-box:11434`, or a host without a scheme in any form
    /// `OLLAMA_HOST` takes (`gpu-box`, `127.0.0.1:11434`, `ollama.com`),
    /// read the way Ollama reads it: `http` on port 11434, and `ollama.com`
    /// alone over `https`. A trailing `/v1` is optional.
    ///
    /// # Errors
    ///
    /// Fails when the host is empty. `build()` fails when the result is not
    /// a valid base URL.
    pub fn builder(host: &str) -> Result<ProviderBuilder> {
        let host = required("host", host)?;
        let host = if host.contains("://") {
            host
        } else {
            ollama_base_url(&host)
        };
        let host = host.trim_end_matches('/');
        let base_url = if host.ends_with("/v1") {
            host.to_owned()
        } else {
            format!("{host}/v1")
        };
        let structured_output = if is_ollama_cloud(&base_url) {
            CapabilitySupport::Unsupported
        } else {
            CapabilitySupport::Supported
        };
        Ok(Provider::builder()
            .base_url(base_url)
            .provider_name(Self::PROVIDER_NAME)
            .quirks(Self::quirks())
            .timeout(None)
            .stream_timeouts(StreamTimeouts::unbounded())
            .declare_chat_capabilities(ollama_capabilities())
            .chat_capability(ChatCapability::StructuredOutput, structured_output))
    }

    /// The dialect this preset configures, under
    /// [`NAMESPACE`](Self::NAMESPACE).
    ///
    /// - A `json_schema` response format is always constrained: it becomes
    ///   Ollama's `format`, enforced by llama.cpp's grammar, which takes no
    ///   `strict` flag and has rules of its own (no
    ///   [`schema_dialect`](CompatQuirks::schema_dialect)), so a strict
    ///   schema is reported `Approximated`
    ///   (<https://docs.ollama.com/capabilities/structured-outputs>).
    /// - Tool arguments are not constrained: a strict tool is reported
    ///   `Unsupported`.
    /// - `parallel_tool_calls` and `tool_choice` are not accepted.
    ///
    /// Ollama Cloud does not constrain the response format at all ("Ollama's
    /// Cloud currently does not support structured outputs"), so a provider
    /// for `ollama.com` answers
    /// [`StructuredOutput`](ChatCapability::StructuredOutput) `Unsupported`.
    /// That is a fact about the host. A cloud model a local server offloads
    /// there (`gemma4:cloud`) is a model claim, which this preset does not
    /// make (ADR 0015): install it with the builder's
    /// [model rules](ProviderBuilder::model_rules), for example
    /// `ChatModelRules::new().structured_output(CapabilitySupport::Unsupported)`.
    #[must_use]
    pub fn quirks() -> CompatQuirks {
        let mut q = CompatQuirks::new(Self::NAMESPACE);
        q.response_format_constraint = SchemaConstraint::Always;
        q.tool_constraint = SchemaConstraint::Never;
        q.schema_dialect = None;
        q.supports_parallel_tool_calls = false;
        // Ollama's OpenAI compatibility lists `tools` but not `tool_choice`
        // among the supported request fields, and ignores it
        // (https://docs.ollama.com/api/openai-compatibility).
        q.supports_tool_choice = false;
        q
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

/// A vLLM OpenAI-compatible server.
///
/// Requests have no total or stream timeouts, streaming or not: a
/// self-hosted server may queue a request behind others or load the model
/// before it answers. The default transport's connect timeout
/// ([`allama_http::DEFAULT_CONNECT_TIMEOUT`], 10 s) still applies, so an
/// unreachable host fails fast; bound a slow one with the builder's
/// `timeout` and `stream_timeouts`, or a `TimeoutProvider` around it.
///
#[cfg_attr(
    not(feature = "reqwest"),
    doc = "[`allama_http::DEFAULT_CONNECT_TIMEOUT`]: crate#transport"
)]
#[derive(Debug, Clone, Copy)]
pub struct Vllm;

impl Vllm {
    /// The [`provider_name`](allama::ProviderIdentity::provider_name) this
    /// preset reports.
    pub const PROVIDER_NAME: &'static str = "vllm";

    /// The default credential scope of this preset's providers, its
    /// provider name: per-request credentials addressed to it
    /// (`allama_http::HttpRequestOptions::credential`) are theirs. Change
    /// it with [`ProviderBuilder::credential_scope`].
    pub const DEFAULT_CREDENTIAL_SCOPE: &'static str = "vllm";

    /// The provider-data namespace of the state this preset mints. Request
    /// options use the shared [`NAMESPACE`](crate::NAMESPACE).
    pub const NAMESPACE: &'static str = "openai-compat:vllm";

    /// A provider for `base_url` (e.g. `http://localhost:8000/v1`).
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
        // Tool calls need server flags (`--enable-auto-tool-choice`,
        // `--tool-call-parser`), so they are not claimed:
        // <https://docs.vllm.ai/en/latest/features/tool_calling.html>.
        let mut builder = Provider::builder()
            .base_url(required("base_url", base_url)?)
            .provider_name(Self::PROVIDER_NAME)
            .quirks(Self::quirks())
            .timeout(None)
            .stream_timeouts(StreamTimeouts::unbounded())
            .declare_chat_capabilities(streaming())
            // "Structured outputs are supported by default in the
            // OpenAI-Compatible Server"
            // (https://docs.vllm.ai/en/latest/features/structured_outputs.html).
            .chat_capability(
                ChatCapability::StructuredOutput,
                CapabilitySupport::Supported,
            );
        if let Some(key) = api_key.filter(|k| !k.trim().is_empty()) {
            builder = builder.bearer_token(key);
        }
        Ok(builder)
    }

    /// The dialect this preset configures, under
    /// [`NAMESPACE`](Self::NAMESPACE). vLLM's structured-output backends
    /// (xgrammar, guidance, outlines) have rules of their own, so there is
    /// no [`schema_dialect`](CompatQuirks::schema_dialect) and a strict
    /// schema is reported `Approximated`; the constraint can also be off
    /// while a reasoning model reasons.
    ///
    /// - A `json_schema` response format is always constrained
    ///   (<https://docs.vllm.ai/en/latest/features/structured_outputs.html>).
    /// - A tool's arguments are constrained when the tool sets `strict: true`
    ///   ("The argument schema of an individual tool is pinned only when that
    ///   tool sets `strict: true`").
    /// - A named or `required` tool choice constrains the forced call through
    ///   structured outputs
    ///   (<https://docs.vllm.ai/en/latest/features/tool_calling.html>).
    #[must_use]
    pub fn quirks() -> CompatQuirks {
        let mut q = CompatQuirks::new(Self::NAMESPACE);
        q.response_format_constraint = SchemaConstraint::Always;
        q.tool_constraint = SchemaConstraint::WhenStrict;
        q.forced_tool_choice_constrained = true;
        q.schema_dialect = None;
        // A sampling extension of vLLM's OpenAI-compatible server
        // (https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).
        q.supports_top_k = true;
        q
    }

    /// Create a provider from `VLLM_BASE_URL` (default
    /// `http://localhost:8000/v1` when unset or blank) and optional
    /// `VLLM_API_KEY`.
    ///
    /// # Errors
    ///
    /// An authored [`Error::Configuration`] naming the variable when
    /// `VLLM_BASE_URL` is not a valid base URL or `VLLM_API_KEY` holds a
    /// value no HTTP header can carry.
    #[cfg(feature = "reqwest")]
    pub fn from_env() -> Result<Provider> {
        let base_url = match optional_env_var("VLLM_BASE_URL") {
            Some(url) => allama_http::config::base_url("VLLM_BASE_URL", &url)?,
            None => "http://localhost:8000/v1".to_owned(),
        };
        let api_key = optional_env_var("VLLM_API_KEY")
            .map(|key| allama_http::config::secret("VLLM_API_KEY", &key))
            .transpose()?;
        Self::new(&base_url, api_key.as_deref())
    }
}

/// GitHub Copilot's Chat Completions endpoint, billed to the Copilot
/// subscription of the token's owner.
///
/// This is not a published GitHub API: it is the endpoint Copilot's own
/// clients call, and GitHub may change it without notice. GitHub's
/// supported route for other applications is the Copilot SDK, which uses
/// the same tokens.
///
/// # Tokens
///
/// The token is sent as a bearer token, as it is; there is no token
/// exchange. GitHub's Copilot SDK documents these
/// (<https://docs.github.com/en/copilot/how-tos/copilot-sdk/auth/authenticate>):
///
/// - an OAuth user token (`gho_`, such as `gh auth token` prints) or a
///   GitHub App user token (`ghu_`);
/// - a fine-grained personal access token (`github_pat_`) with the
///   account permission "Copilot Requests";
/// - in GitHub Actions, the workflow's `GITHUB_TOKEN` with the
///   `copilot-requests: write` permission, in an organization-owned
///   repository whose organization bills Copilot to itself.
///
/// Classic personal access tokens (`ghp_`) are refused.
///
/// # Hosts
///
/// Copilot Business and Enterprise plans may be served from their own
/// host (`https://api.business.githubcopilot.com`,
/// `https://api.enterprise.githubcopilot.com`): address it with
/// [`builder_with_base_url`](Self::builder_with_base_url), or
/// `GITHUB_COPILOT_BASE_URL` for [`from_env`][Self::from_env].
///
/// # Integration id
///
/// Every request carries `Copilot-Integration-Id`
/// ([`INTEGRATION_ID`](Self::INTEGRATION_ID), the Copilot CLI's, which the
/// Copilot SDK uses). Send another with
/// [`ProviderBuilder::header`], which replaces it:
///
/// ```
/// use allama_openai_compat::providers::GitHubCopilot;
///
/// let builder = GitHubCopilot::builder("gho_token")?.header("copilot-integration-id", "my-app");
/// # let _ = builder;
/// # Ok::<(), allama::Error>(())
/// ```
///
/// # Models
///
/// Which model ids a token can use depends on its plan (`GET /models`
/// lists them). Ids are sent as written, dots included
/// (`claude-sonnet-4.5`). The host serves some models only on its
/// Responses or Messages API, which this preset does not speak; the
/// endpoint refuses those here (`unsupported_api_for_model`).
///
/// The output cap is sent as `max_completion_tokens`: the endpoint refuses
/// `max_tokens`.
///
/// The reasoning effort is sent as `reasoning_effort`, and which levels a
/// model takes is the model's to decide (ADR 0015): the endpoint refuses a
/// level the model does not offer (Claude ids have no `none`) and an effort
/// for a model without one (`claude-haiku-4.5`), which is
/// [`Error::Unsupported`] with the reasoning as its subject. Claude ids
/// reason adaptively, so the effort is a ceiling: an easy request gets no
/// reasoning at any effort. Reasoning text arrives in `reasoning_text`
/// with a `reasoning_opaque` signature, and both go back in later turns
/// ([`ReasoningReplayField::ReasoningText`]).
///
/// The endpoint serves no embeddings.
///
#[cfg_attr(not(feature = "reqwest"), doc = "[Self::from_env]: crate#transport")]
#[derive(Debug, Clone, Copy)]
pub struct GitHubCopilot;

impl GitHubCopilot {
    /// The [`provider_name`](allama::ProviderIdentity::provider_name) this
    /// preset reports.
    pub const PROVIDER_NAME: &'static str = "github-copilot";

    /// The default credential scope of this preset's providers, its
    /// provider name: per-request credentials addressed to it
    /// (`allama_http::HttpRequestOptions::credential`) are theirs. Change
    /// it with [`ProviderBuilder::credential_scope`].
    pub const DEFAULT_CREDENTIAL_SCOPE: &'static str = "github-copilot";

    /// The provider-data namespace of the state this preset mints. Request
    /// options use the shared [`NAMESPACE`](crate::NAMESPACE).
    pub const NAMESPACE: &'static str = "openai-compat:github-copilot";

    /// The base URL of [`builder`](Self::builder).
    pub const BASE_URL: &'static str = "https://api.githubcopilot.com";

    /// The `Copilot-Integration-Id` every request carries: the Copilot
    /// CLI's, which the Copilot SDK uses.
    pub const INTEGRATION_ID: &'static str = "copilot-developer-cli";

    /// A provider for `https://api.githubcopilot.com`.
    ///
    /// # Errors
    ///
    /// Fails when the token is empty.
    #[allow(clippy::new_ret_no_self)]
    #[cfg(feature = "reqwest")]
    pub fn new(token: &str) -> Result<Provider> {
        Self::builder(token)?.build()
    }

    /// A pre-configured builder for `https://api.githubcopilot.com`.
    ///
    /// # Errors
    ///
    /// Fails when the token is empty.
    pub fn builder(token: &str) -> Result<ProviderBuilder> {
        Self::builder_with_base_url(Self::BASE_URL, token)
    }

    /// A provider for another Copilot host. See
    /// [`builder_with_base_url`](Self::builder_with_base_url).
    ///
    /// # Errors
    ///
    /// See [`builder_with_base_url`](Self::builder_with_base_url).
    #[cfg(feature = "reqwest")]
    pub fn new_with_base_url(base_url: &str, token: &str) -> Result<Provider> {
        Self::builder_with_base_url(base_url, token)?.build()
    }

    /// A pre-configured builder for another Copilot host, such as
    /// `https://api.enterprise.githubcopilot.com`. Requests go to
    /// `{base_url}/chat/completions`.
    ///
    /// Plain `http://` is refused except on loopback (a local proxy): the
    /// token is a GitHub credential, and every Copilot host serves HTTPS.
    ///
    /// # Errors
    ///
    /// An authored [`Error::Configuration`] when the token is empty, or the
    /// base URL is not an absolute `http(s)` URL without a query
    /// ([`allama_http::config::base_url`]), or is `http://` on a host other
    /// than loopback.
    pub fn builder_with_base_url(base_url: &str, token: &str) -> Result<ProviderBuilder> {
        let base_url = allama_http::config::base_url("base_url", base_url)?;
        if !is_https_or_loopback(&base_url) {
            return Err(Error::configuration(
                "`base_url` must use `https://` (plain `http://` only on loopback): \
                 the GitHub Copilot token is a GitHub credential",
            )
            .authored());
        }
        Ok(
            hosted(&base_url, token, Self::PROVIDER_NAME, Self::quirks())?
                .header("copilot-integration-id", Self::INTEGRATION_ID),
        )
    }

    /// The dialect this preset configures, under
    /// [`NAMESPACE`](Self::NAMESPACE): reasoning goes back as
    /// `reasoning_text` with its `reasoning_opaque` signature. The endpoint
    /// routes to several vendors' models, so a strict schema is rewritten
    /// for OpenAI's rules but not [verified](CompatQuirks::dialect_verified).
    #[must_use]
    pub fn quirks() -> CompatQuirks {
        let mut q = CompatQuirks::new(Self::NAMESPACE);
        // `max_tokens` is a bare `400 Bad Request` (no JSON, nothing names
        // the field), and `max_completion_tokens` is accepted (recorded by
        // Pydantic AI, `test_github_copilot_rejects_max_tokens`).
        q.max_tokens_field = MaxTokensField::MaxCompletionTokens;
        // Copilot's editor clients send both back on the assistant message
        // (VS Code's `copilotChatEndpoint.ts`, Zed's `copilot_chat`).
        // Copilot also takes a turn without them (Pydantic AI, probed live
        // 2026-09-07), so dropping reasoning without a signature costs
        // nothing.
        q.reasoning_replay = ReasoningReplayField::ReasoningText;
        q
    }

    /// Create a provider from `COPILOT_GITHUB_TOKEN` (the variable the
    /// Copilot SDK reads first) and optional `GITHUB_COPILOT_BASE_URL`
    /// (default `https://api.githubcopilot.com`).
    ///
    /// `GITHUB_TOKEN` and `GH_TOKEN` are not read: they usually hold a
    /// token for GitHub's REST API, which is not a Copilot token.
    ///
    /// # Errors
    ///
    /// An authored [`Error::Configuration`] naming the variable when
    /// `COPILOT_GITHUB_TOKEN` is not set or holds a value no HTTP header can
    /// carry, or `GITHUB_COPILOT_BASE_URL` is not a valid base URL or is
    /// plain `http://` off loopback.
    #[cfg(feature = "reqwest")]
    pub fn from_env() -> Result<Provider> {
        let token = env_secret("COPILOT_GITHUB_TOKEN")?;
        match optional_env_var("GITHUB_COPILOT_BASE_URL") {
            Some(url) => Self::new_with_base_url(
                &allama_http::config::base_url("GITHUB_COPILOT_BASE_URL", &url)?,
                &token,
            ),
            None => Self::new(&token),
        }
    }
}

/// Azure OpenAI through its OpenAI-compatible v1 API.
///
/// Address a resource by name ([`builder`](Self::builder), for
/// `https://{resource}.openai.azure.com`) or by its endpoint URL
/// ([`builder_with_endpoint`](Self::builder_with_endpoint), for any host, Azure AI
/// Foundry's `*.services.ai.azure.com` included). [`from_env`][Self::from_env]
/// reads `AZURE_OPENAI_ENDPOINT`, as Azure's own SDKs do, or
/// `AZURE_OPENAI_RESOURCE`. The constructors pair up as the native
/// providers' do: [`new`][Self::new] and [`builder`](Self::builder) for the
/// usual address, `new_with_*` and `builder_with_*` for an explicit URL
/// (compare `allama_openai::Provider::new_with_base_url`).
///
/// The same resource serves two chat APIs. The constructors above build a
/// Chat Completions [`Provider`]; their `responses` twins
/// ([`responses`][Self::responses], [`responses_builder`](Self::responses_builder),
/// [`responses_from_env`][Self::responses_from_env], ...) build a
/// [`ResponsesProvider`] for the Responses API at the same address, with the
/// same key and the same [`NAMESPACE`](Self::NAMESPACE), so a conversation
/// can move between the two (each replays only the state its API knows).
///
/// Strict schemas use [`SchemaDialect::AzureStrict`] on every deployment,
/// and a request with a strict tool turns parallel tool calls off unless the
/// caller set them. A reasoning deployment that rejects
/// `parallel_tool_calls` (see Azure's reasoning-model feature matrix) needs
/// `supports_parallel_tool_calls = false`:
///
/// ```
/// use allama_openai_compat::providers::Azure;
///
/// let mut quirks = Azure::quirks();
/// quirks.supports_parallel_tool_calls = false;
/// let builder = Azure::builder("my-resource", "key")?.quirks(quirks);
/// # let _ = builder;
/// # Ok::<(), allama::Error>(())
/// ```
///
/// Model names are deployment names, which say nothing about the model
/// behind them, so the Responses provider assumes nothing per model
/// ([`ResponsesModelRules::new`](crate::ResponsesModelRules::new)).
/// Describe a reasoning deployment with
/// [`ResponsesProviderBuilder::model_rules`]:
///
/// ```
/// use allama::{CapabilitySupport, ReasoningEffort};
/// use allama_openai_compat::ResponsesModelRules;
/// use allama_openai_compat::providers::Azure;
///
/// let builder = Azure::responses_builder("my-resource", "key")?.model_rules(|deployment: &str| {
///     match deployment {
///         "my-reasoning-deployment" => ResponsesModelRules::new()
///             .reasoning(CapabilitySupport::Supported)
///             .effort_levels(&[ReasoningEffort::Low, ReasoningEffort::Medium, ReasoningEffort::High]),
///         _ => ResponsesModelRules::new(),
///     }
/// });
/// # let _ = builder;
/// # Ok::<(), allama::Error>(())
/// ```
#[cfg_attr(
    not(feature = "reqwest"),
    doc = "[Self::from_env]: crate#transport\n[Self::new]: crate#transport\n\
           [Self::responses]: crate#transport\n[Self::responses_from_env]: crate#transport"
)]
#[derive(Debug, Clone, Copy)]
pub struct Azure;

impl Azure {
    /// The [`provider_name`](allama::ProviderIdentity::provider_name) this
    /// preset reports.
    pub const PROVIDER_NAME: &'static str = "azure";

    /// The default credential scope of this preset's providers, its
    /// provider name: per-request credentials addressed to it
    /// (`allama_http::HttpRequestOptions::credential`) are theirs. Change
    /// it with [`ProviderBuilder::credential_scope`].
    pub const DEFAULT_CREDENTIAL_SCOPE: &'static str = "azure";

    /// The provider-data namespace of the state this preset mints. Request
    /// options use the shared [`NAMESPACE`](crate::NAMESPACE).
    pub const NAMESPACE: &'static str = "openai-compat:azure";

    /// A provider for an Azure OpenAI resource. Model names are deployment
    /// names.
    ///
    /// # Errors
    ///
    /// See [`builder`](Self::builder).
    #[allow(clippy::new_ret_no_self)]
    #[cfg(feature = "reqwest")]
    pub fn new(resource: &str, api_key: &str) -> Result<Provider> {
        Self::builder(resource, api_key)?.build()
    }

    /// A pre-configured builder for the resource named `resource`, at
    /// `https://{resource}.openai.azure.com/openai/v1`.
    ///
    /// # Errors
    ///
    /// An authored [`Error::Configuration`] when the key is empty, or when
    /// `resource` is not a resource name: 2 to 63 ASCII letters, digits
    /// and hyphens, starting and ending with a letter or digit. Anything
    /// else would change the host the key is sent to.
    pub fn builder(resource: &str, api_key: &str) -> Result<ProviderBuilder> {
        Self::builder_with_endpoint(&Self::resource_endpoint(resource)?, api_key)
    }

    /// A provider for an Azure OpenAI endpoint URL. See
    /// [`builder_with_endpoint`](Self::builder_with_endpoint).
    ///
    /// # Errors
    ///
    /// See [`builder_with_endpoint`](Self::builder_with_endpoint).
    #[cfg(feature = "reqwest")]
    pub fn new_with_endpoint(endpoint: &str, api_key: &str) -> Result<Provider> {
        Self::builder_with_endpoint(endpoint, api_key)?.build()
    }

    /// A pre-configured builder for the endpoint `endpoint`, such as
    /// `https://my-resource.openai.azure.com/` or
    /// `https://my-project.services.ai.azure.com/` (the value of
    /// `AZURE_OPENAI_ENDPOINT`). Requests go to `{endpoint}/openai/v1`.
    ///
    /// ```
    /// use allama_openai_compat::providers::Azure;
    ///
    /// let builder = Azure::builder_with_endpoint("https://my-project.services.ai.azure.com/", "key")?;
    /// # let _ = builder;
    /// # Ok::<(), allama::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// An authored [`Error::Configuration`] when the key is empty, or the
    /// endpoint is not an absolute `http(s)` URL without a query
    /// ([`allama_http::config::base_url`]).
    pub fn builder_with_endpoint(endpoint: &str, api_key: &str) -> Result<ProviderBuilder> {
        Ok(Provider::builder()
            .base_url(Self::v1_base_url(endpoint)?)
            .auth_header("api-key", required("api_key", api_key)?)
            // A per-request key always goes in `api-key`, replacing an
            // Entra ID token configured with `bearer_token`.
            .credential_placement(CredentialPlacement::Header(Cow::Borrowed("api-key")))
            .request_id_header_name("apim-request-id")
            .provider_name(Self::PROVIDER_NAME)
            .quirks(Self::quirks())
            // Tool calls depend on the deployed model: `Unknown`.
            .declare_chat_capabilities(streaming()))
    }

    /// The dialect this preset configures: OpenAI's, under
    /// [`NAMESPACE`](Self::NAMESPACE), with the
    /// [`AzureStrict`](SchemaDialect::AzureStrict) schema dialect for every
    /// deployment (deployment names are not model ids, so the `ft:` rule
    /// does not apply).
    #[must_use]
    pub fn quirks() -> CompatQuirks {
        CompatQuirks {
            namespace: Self::NAMESPACE.into(),
            schema_dialect: Some(SchemaDialect::AzureStrict),
            fine_tuned_dialect: false,
            ..CompatQuirks::openai()
        }
    }

    /// Create a provider from `AZURE_OPENAI_API_KEY` and either
    /// `AZURE_OPENAI_ENDPOINT` (an endpoint URL, which wins when both are
    /// set) or `AZURE_OPENAI_RESOURCE` (a resource name).
    ///
    /// # Errors
    ///
    /// An authored [`Error::Configuration`] naming the variable when the key
    /// or both address variables are missing, or a value is malformed.
    #[cfg(feature = "reqwest")]
    pub fn from_env() -> Result<Provider> {
        let (endpoint, api_key) = Self::env()?;
        Self::new_with_endpoint(&endpoint, &api_key)
    }

    /// A Responses API provider for an Azure OpenAI resource. Model names
    /// are deployment names.
    ///
    /// # Errors
    ///
    /// See [`responses_builder`](Self::responses_builder).
    #[cfg(feature = "reqwest")]
    pub fn responses(resource: &str, api_key: &str) -> Result<ResponsesProvider> {
        Self::responses_builder(resource, api_key)?.build()
    }

    /// A pre-configured Responses API builder for the resource named
    /// `resource`, sending to
    /// `https://{resource}.openai.azure.com/openai/v1/responses`.
    ///
    /// # Errors
    ///
    /// As for [`builder`](Self::builder).
    pub fn responses_builder(resource: &str, api_key: &str) -> Result<ResponsesProviderBuilder> {
        Self::responses_builder_with_endpoint(&Self::resource_endpoint(resource)?, api_key)
    }

    /// A Responses API provider for an Azure OpenAI endpoint URL. See
    /// [`responses_builder_with_endpoint`](Self::responses_builder_with_endpoint).
    ///
    /// # Errors
    ///
    /// See [`responses_builder_with_endpoint`](Self::responses_builder_with_endpoint).
    #[cfg(feature = "reqwest")]
    pub fn responses_with_endpoint(endpoint: &str, api_key: &str) -> Result<ResponsesProvider> {
        Self::responses_builder_with_endpoint(endpoint, api_key)?.build()
    }

    /// A pre-configured Responses API builder for the endpoint `endpoint`,
    /// as [`builder_with_endpoint`](Self::builder_with_endpoint) takes it.
    /// Requests go to `{endpoint}/openai/v1/responses`.
    ///
    /// # Errors
    ///
    /// As for [`builder_with_endpoint`](Self::builder_with_endpoint).
    pub fn responses_builder_with_endpoint(
        endpoint: &str,
        api_key: &str,
    ) -> Result<ResponsesProviderBuilder> {
        Ok(ResponsesProvider::builder()
            .base_url(Self::v1_base_url(endpoint)?)
            .auth_header("api-key", required("api_key", api_key)?)
            // A per-request key always goes in `api-key`, replacing an
            // Entra ID token configured with `bearer_token`.
            .credential_placement(CredentialPlacement::Header(Cow::Borrowed("api-key")))
            .request_id_header_name("apim-request-id")
            .provider_name(Self::PROVIDER_NAME)
            .quirks(Self::responses_quirks())
            .declare_chat_capabilities(streaming()))
    }

    /// The Responses dialect this preset configures: OpenAI's, under
    /// [`NAMESPACE`](Self::NAMESPACE), with the
    /// [`AzureStrict`](SchemaDialect::AzureStrict) schema dialect for every
    /// deployment, as [`quirks`](Self::quirks) sets for Chat Completions.
    #[must_use]
    pub fn responses_quirks() -> ResponsesQuirks {
        ResponsesQuirks {
            namespace: Self::NAMESPACE.into(),
            schema_dialect: Some(SchemaDialect::AzureStrict),
            fine_tuned_dialect: false,
            ..ResponsesQuirks::openai()
        }
    }

    /// Create a Responses API provider from the variables
    /// [`from_env`][Self::from_env] reads.
    ///
    /// # Errors
    ///
    /// As for [`from_env`][Self::from_env].
    #[cfg(feature = "reqwest")]
    pub fn responses_from_env() -> Result<ResponsesProvider> {
        let (endpoint, api_key) = Self::env()?;
        Self::responses_with_endpoint(&endpoint, &api_key)
    }

    /// The endpoint URL of the resource named `resource`, checked so that
    /// it cannot change the host the key is sent to.
    fn resource_endpoint(resource: &str) -> Result<String> {
        let resource = required("resource", resource)?;
        if !is_dns_label(&resource) {
            return Err(Error::configuration(
                "the Azure resource name must be 2 to 63 letters, digits and hyphens, \
                 starting and ending with a letter or digit",
            )
            .authored());
        }
        Ok(format!("https://{resource}.openai.azure.com"))
    }

    /// The v1 API base URL of an endpoint URL: `{endpoint}/openai/v1`.
    fn v1_base_url(endpoint: &str) -> Result<String> {
        let endpoint = allama_http::config::base_url("endpoint", endpoint)?;
        Ok(if endpoint.ends_with("/openai/v1") {
            endpoint
        } else {
            format!("{endpoint}/openai/v1")
        })
    }

    /// The endpoint URL and the key from the environment:
    /// `AZURE_OPENAI_ENDPOINT` (which wins) or `AZURE_OPENAI_RESOURCE`, and
    /// `AZURE_OPENAI_API_KEY`.
    #[cfg(feature = "reqwest")]
    fn env() -> Result<(String, String)> {
        let api_key = env_secret("AZURE_OPENAI_API_KEY")?;
        if let Some(endpoint) = optional_env_var("AZURE_OPENAI_ENDPOINT") {
            let endpoint = allama_http::config::base_url("AZURE_OPENAI_ENDPOINT", &endpoint)?;
            return Ok((endpoint, api_key));
        }
        let resource = optional_env_var("AZURE_OPENAI_RESOURCE").ok_or_else(|| {
            Error::configuration("AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_RESOURCE not set")
                .authored()
        })?;
        Ok((Self::resource_endpoint(&resource)?, api_key))
    }
}

/// Cloudflare Workers AI over HTTP.
#[derive(Debug, Clone, Copy)]
pub struct Cloudflare;

impl Cloudflare {
    /// The [`provider_name`](allama::ProviderIdentity::provider_name) this
    /// preset reports.
    pub const PROVIDER_NAME: &'static str = "cloudflare";

    /// The default credential scope of this preset's providers, its
    /// provider name: per-request credentials addressed to it
    /// (`allama_http::HttpRequestOptions::credential`) are theirs. Change
    /// it with [`ProviderBuilder::credential_scope`].
    pub const DEFAULT_CREDENTIAL_SCOPE: &'static str = "cloudflare";

    /// The provider-data namespace of the state this preset mints. Request
    /// options use the shared [`NAMESPACE`](crate::NAMESPACE).
    pub const NAMESPACE: &'static str = "openai-compat:cloudflare";

    /// Create a provider.
    ///
    /// # Errors
    ///
    /// Fails when the account id or token is empty.
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
        let mut q = CompatQuirks::new(Self::NAMESPACE);
        // Strict schemas stay rewritten for OpenAI's rules, but are not
        // verified: "Workers AI can't guarantee that the model responds
        // according to the requested JSON Schema"
        // (https://developers.cloudflare.com/workers-ai/features/json-mode/).
        // JSON mode and tool calls are served by some models only (the
        // json-mode and function-calling pages list them), so neither is
        // claimed: both answer `Unknown`, and extraction keeps its native →
        // forced-tool fallback.
        q.drop_zero_usage_chunks = true;
        // The reasoning effort is sent, as the binding adapter
        // (`allama-cloudflare-worker`) sends it: models that reason take it
        // in their input schema, and the others ignore it (verified on
        // `@cf/meta/llama-3.3-70b-instruct-fp8-fast`).
        q.supports_reasoning_effort = true;
        q.supports_parallel_tool_calls = false;
        Ok(Provider::builder()
            .base_url(format!(
                "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
            ))
            .bearer_token(required("api_token", api_token)?)
            .provider_name(Self::PROVIDER_NAME)
            .quirks(q)
            .model_rules(Self::model_rules)
            .declare_chat_capabilities(streaming())
            .embedding_capability(
                EmbeddingCapability::BatchInput,
                CapabilitySupport::Supported,
            ))
    }

    /// The [model rules](ProviderBuilder::model_rules) this preset installs.
    ///
    /// Mistral models (`@cf/mistralai/…`, `@cf/mistral/…`, `@hf/mistral/…`)
    /// refuse any tool call id but nine letters and digits ("Tool call id
    /// was … but must be a-z, A-Z, 0-9, with a length of 9", verified live
    /// on `@cf/mistralai/mistral-small-3.1-24b-instruct`), so ids from
    /// other providers are mapped for them. Rules of your own replace these;
    /// start them from this function to keep the mapping.
    #[must_use]
    pub fn model_rules(model: &str) -> ChatModelRules {
        if workers_ai_mistral(model) {
            ChatModelRules::new().tool_call_id_format(ToolCallIdFormat::Alphanumeric(9))
        } else {
            ChatModelRules::new()
        }
    }

    /// Create a provider from `CLOUDFLARE_ACCOUNT_ID` and
    /// `CLOUDFLARE_API_TOKEN`.
    ///
    /// # Errors
    ///
    /// Fails when either variable is missing or malformed.
    #[cfg(feature = "reqwest")]
    pub fn from_env() -> Result<Provider> {
        Self::new(
            &env_var("CLOUDFLARE_ACCOUNT_ID")?,
            &env_secret("CLOUDFLARE_API_TOKEN")?,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use allama::ChatProvider;

    /// Mistral models on Workers AI take only nine-character ids; the
    /// others take any, and keep the endpoint's rule.
    #[test]
    fn cloudflare_maps_tool_call_ids_for_mistral_models_only() {
        for model in [
            "@cf/mistralai/mistral-small-3.1-24b-instruct",
            "@cf/mistral/mistral-7b-instruct-v0.1",
            "@hf/mistral/mistral-7b-instruct-v0.2",
        ] {
            assert_eq!(
                Cloudflare::model_rules(model).tool_call_id_format,
                Some(ToolCallIdFormat::Alphanumeric(9)),
                "{model}"
            );
        }
        for model in [
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            "mistral",
            "@cf/x/mistral",
        ] {
            assert_eq!(
                Cloudflare::model_rules(model),
                ChatModelRules::new(),
                "{model}"
            );
        }
        let provider = Cloudflare::builder("0123456789abcdef0123456789abcdef", "k")
            .unwrap()
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(
            provider
                .model_rules("@cf/mistralai/mistral-small-3.1-24b-instruct")
                .tool_call_id_format,
            Some(ToolCallIdFormat::Alphanumeric(9))
        );
    }

    /// Ollama's OpenAI compatibility lists `tools` but not `tool_choice`
    /// among the request fields it reads
    /// (<https://docs.ollama.com/api/openai-compatibility>): a forced
    /// choice is refused rather than silently ignored, and tool support,
    /// which depends on the model's template, is not claimed. vLLM serves
    /// tool calls only with `--enable-auto-tool-choice` and
    /// `--tool-call-parser` set on the server
    /// (<https://docs.vllm.ai/en/latest/features/tool_calling/>).
    #[test]
    fn self_hosted_presets_do_not_claim_tools() {
        use allama::CapabilitySupport::{Supported, Unknown, Unsupported};
        let ollama = Ollama::builder("http://localhost:11434")
            .unwrap()
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        let vllm = Vllm::builder("http://localhost:8000/v1", None)
            .unwrap()
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        for (provider, forced) in [(&ollama, Unsupported), (&vllm, Unknown)] {
            let answer = |c| provider.chat_capability("any-model", c);
            assert_eq!(answer(ChatCapability::ToolCalls), Unknown);
            assert_eq!(answer(ChatCapability::ForcedToolChoice), forced);
            assert_eq!(answer(ChatCapability::Streaming), Supported);
        }
    }

    /// Ollama's Chat Completions endpoint rejects a `file` part for every
    /// model, before it even looks the model up: the fixture is its reply
    /// (Ollama 0.40.0, recorded 2026-10-13, the same with a model that does
    /// not exist). Sent, a PDF fails as an opaque `InvalidRequest` that a
    /// fallback does not route; the preset refuses it before sending
    /// instead. Image and audio parts pass Ollama's parser, so they stay
    /// the model's to answer.
    #[tokio::test]
    async fn ollama_refuses_file_input_before_sending() {
        use allama::CapabilitySupport::{Unknown, Unsupported};
        use allama::{ChatProvider, ChatRequest, ContentPart, Error, Subject, UnsupportedStage};
        use allama_http::{ReplayResponse, ReplayTransport};

        let reply = include_str!("../fixtures/presets/ollama_file_part_rejected.json");
        let request = ChatRequest::new("qwen3:0.6b").user(vec![
            ContentPart::text("Summarize this."),
            ContentPart::file_base64("application/pdf", "JVBERi0xLjQ="),
        ]);

        // What the endpoint answers when the part is sent.
        let transport = ReplayTransport::new([ReplayResponse::json(400, reply)]);
        let unclaimed = crate::Provider::builder()
            .base_url("http://localhost:11434/v1")
            .quirks(Ollama::quirks())
            .http_transport(transport.clone())
            .build()
            .unwrap();
        let error = unclaimed.chat(&request).await.unwrap_err();
        assert!(matches!(error, Error::InvalidRequest { .. }), "{error:?}");
        assert!(error.to_string().contains("invalid message format"));
        assert_eq!(error.unsupported_subject(), None);
        assert_eq!(transport.requests().len(), 1);

        // The preset refuses it first, naming the part.
        let transport = ReplayTransport::new([ReplayResponse::json(400, reply)]);
        let ollama = Ollama::builder("http://localhost:11434")
            .unwrap()
            .http_transport(transport.clone())
            .build()
            .unwrap();
        assert_eq!(
            ollama.chat_capability("qwen3:0.6b", ChatCapability::FileInput),
            Unsupported
        );
        let error = ollama.chat(&request).await.unwrap_err();
        assert!(matches!(error, Error::Unsupported { .. }), "{error:?}");
        assert!(matches!(
            error.unsupported_subject(),
            Some(Subject::Content {
                message: 0,
                block: Some(1),
                ..
            })
        ));
        assert_eq!(
            error.unsupported_stage(),
            Some(UnsupportedStage::BeforeSending)
        );
        assert!(transport.requests().is_empty());

        for capability in [ChatCapability::ImageInput, ChatCapability::AudioInput] {
            assert_eq!(
                ollama.chat_capability("qwen3:0.6b", capability),
                Unknown,
                "{capability:?}"
            );
        }
    }

    /// Which Groq models take strict mode, and which Ollama models run on
    /// Ollama Cloud, is model data (ADR 0015): neither preset refuses or
    /// relaxes anything because of the model id. A strict schema goes out
    /// strict to any Groq model, and a JSON response format to an Ollama
    /// cloud tag through a local server; the endpoint decides. The evidence
    /// for the removed rules is in `docs/catalog/`, and a caller's model
    /// rules still re-add them.
    #[test]
    fn presets_make_no_model_claims() {
        use allama::CapabilitySupport::{Supported, Unsupported};
        use allama::{ChatRequest, ResponseFormat};
        let groq = Groq::builder("k")
            .unwrap()
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": false
        });
        for model in ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"] {
            let request = ChatRequest::new(model)
                .user("Where?")
                .response_format(ResponseFormat::json_schema(schema.clone()));
            let validation = groq.validate_chat_request(&request).unwrap();
            assert!(
                validation.warnings().is_empty(),
                "{model}: {:?}",
                validation.warnings()
            );
            assert_eq!(groq.model_rules(model), ChatModelRules::new(), "{model}");
        }

        let ollama = Ollama::builder("http://localhost:11434")
            .unwrap()
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        for model in ["gemma4:cloud", "qwen3-coder:480b-cloud", "llama3.2"] {
            assert_eq!(
                ollama.chat_capability(model, ChatCapability::StructuredOutput),
                Supported,
                "{model}"
            );
            let request = ChatRequest::new(model)
                .user("Where?")
                .response_format(ResponseFormat::Json);
            assert!(ollama.validate_chat_request(&request).is_ok(), "{model}");
        }

        // The rule is the caller's to install.
        let rules = Ollama::builder("http://localhost:11434")
            .unwrap()
            .model_rules(|model: &str| {
                if model.ends_with(":cloud") {
                    ChatModelRules::new().structured_output(Unsupported)
                } else {
                    ChatModelRules::new()
                }
            })
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(
            rules.chat_capability("gemma4:cloud", ChatCapability::StructuredOutput),
            Unsupported
        );
    }

    /// GitHub Copilot: the token as a bearer token, the integration id on
    /// every request (replaceable), another host by base URL, and no rules
    /// per model id (ADR 0015).
    #[tokio::test]
    async fn github_copilot_sends_token_and_integration_id() {
        use allama::{ChatRequest, ProviderIdentity};
        use allama_http::{ReplayResponse, ReplayTransport};

        let reply = || {
            ReplayResponse::json(
                200,
                r#"{"id":"c1","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}"#,
            )
        };
        let transport = ReplayTransport::new([reply()]);
        let copilot = GitHubCopilot::builder(" gho_token ")
            .unwrap()
            .http_transport(transport.clone())
            .build()
            .unwrap();
        copilot
            .chat(
                &ChatRequest::new("claude-sonnet-4.5")
                    .user("Hi")
                    .max_output_tokens(100)
                    .reasoning(
                        allama::ReasoningConfig::new().effort(allama::ReasoningEffort::Medium),
                    ),
            )
            .await
            .unwrap();
        let sent = &transport.requests()[0];
        assert_eq!(sent.url, "https://api.githubcopilot.com/chat/completions");
        assert_eq!(sent.headers.get("authorization"), Some("Bearer gho_token"));
        assert_eq!(
            sent.headers.get("copilot-integration-id"),
            Some("copilot-developer-cli")
        );
        let body: serde_json::Value = serde_json::from_slice(sent.body.as_ref().unwrap()).unwrap();
        assert_eq!(body["model"], "claude-sonnet-4.5");
        assert_eq!(body["max_completion_tokens"], 100);
        assert_eq!(body["reasoning_effort"], "medium", "{body}");
        assert!(body.get("max_tokens").is_none(), "{body}");
        assert_eq!(copilot.provider_name(), "github-copilot");
        assert_eq!(copilot.quirks().namespace, GitHubCopilot::NAMESPACE);
        assert_eq!(
            copilot.quirks().reasoning_replay,
            ReasoningReplayField::ReasoningText
        );
        assert_eq!(
            copilot.model_rules("claude-haiku-4.5"),
            ChatModelRules::new()
        );

        let transport = ReplayTransport::new([reply()]);
        let enterprise = GitHubCopilot::builder_with_base_url(
            "https://api.enterprise.githubcopilot.com/",
            "gho_token",
        )
        .unwrap()
        .header("Copilot-Integration-Id", "my-app")
        .http_transport(transport.clone())
        .build()
        .unwrap();
        enterprise
            .chat(&ChatRequest::new("gpt-5.4").user("Hi"))
            .await
            .unwrap();
        let sent = &transport.requests()[0];
        assert_eq!(
            sent.url,
            "https://api.enterprise.githubcopilot.com/chat/completions"
        );
        assert_eq!(sent.headers.get("copilot-integration-id"), Some("my-app"));

        assert!(GitHubCopilot::builder(" ").is_err());
        assert!(GitHubCopilot::builder_with_base_url("api.githubcopilot.com", "t").is_err());

        // The token is a GitHub credential: never in the clear off loopback.
        for base_url in [
            "http://api.githubcopilot.com",
            "http://10.0.0.5:8080",
            "http://localhost.evil.example",
            "http://localhost@evil.example",
            "http://127.0.0.1.evil.example",
        ] {
            let err = GitHubCopilot::builder_with_base_url(base_url, "t").unwrap_err();
            assert!(
                matches!(err, Error::Configuration { .. }),
                "{base_url}: {err:?}"
            );
        }
        for base_url in [
            "http://localhost:8080",
            "HTTP://LOCALHOST",
            "http://127.0.0.1:4141/copilot",
            "http://[::1]:4141",
            "http://user:pw@127.0.0.1",
            "https://copilot-proxy.internal",
        ] {
            assert!(
                GitHubCopilot::builder_with_base_url(base_url, "t").is_ok(),
                "{base_url}"
            );
        }
    }

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
            // User info stays ahead of the host (and out of
            // `Debug`), instead of being read as part of it.
            ("http://user:pw@h:1", "http://user:pw@h:1"),
            ("user:pw@gpu-box", "http://user:pw@gpu-box:11434"),
        ] {
            assert_eq!(ollama_base_url(raw), expected, "OLLAMA_HOST={raw:?}");
        }
        let ollama = Ollama::builder(&ollama_base_url("http://user:SECRET@h:1"))
            .unwrap()
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(ollama.base_url(), "http://user:SECRET@h:1/v1");
        assert!(!format!("{ollama:?}").contains("SECRET"), "{ollama:?}");
    }

    /// The Azure resource and the Cloudflare account id are checked,
    /// so neither can move the key to another host or path.
    #[test]
    fn resource_names_cannot_change_where_the_key_goes() {
        let transport = allama_http::ReplayTransport::default;
        for resource in [
            "attacker.example/x?",
            "a.b",
            "-lead",
            "trail-",
            "x",
            "with space",
            "evil.com#",
        ] {
            let err = Azure::builder(resource, "OPERATOR-KEY").unwrap_err();
            assert!(
                matches!(err, Error::Configuration { .. }),
                "{resource}: {err:?}"
            );
            assert!(!err.to_string().contains(resource), "{err}");
        }
        let azure = Azure::builder("My-Resource01", "k")
            .unwrap()
            .http_transport(transport())
            .build()
            .unwrap();
        assert_eq!(
            azure.base_url(),
            "https://My-Resource01.openai.azure.com/openai/v1"
        );
        for (endpoint, expected) in [
            (
                "https://proj.services.ai.azure.com/",
                "https://proj.services.ai.azure.com/openai/v1",
            ),
            (
                "https://r.openai.azure.com/openai/v1",
                "https://r.openai.azure.com/openai/v1",
            ),
        ] {
            let azure = Azure::builder_with_endpoint(endpoint, "k")
                .unwrap()
                .http_transport(transport())
                .build()
                .unwrap();
            assert_eq!(azure.base_url(), expected);
        }
        assert!(Azure::builder_with_endpoint("https://x.example/?api-version=1", "k").is_err());

        for account in [
            "0123456789abcdef0123456789abcdeg",
            "0123456789abcdef",
            "../../../evil",
            "0123456789abcdef0123456789abcdef/x",
        ] {
            let err = Cloudflare::builder(account, "t").unwrap_err();
            assert!(
                matches!(err, Error::Configuration { .. }),
                "{account}: {err:?}"
            );
        }
        let cloudflare = Cloudflare::builder("0123456789ABCDEF0123456789abcdef", "t")
            .unwrap()
            .http_transport(transport())
            .build()
            .unwrap();
        assert!(
            cloudflare
                .base_url()
                .contains("/accounts/0123456789ABCDEF0123456789abcdef/")
        );
    }

    #[test]
    fn presets_configure_endpoints_and_dialects() {
        use allama::ProviderIdentity;

        let groq = Groq::builder("k")
            .unwrap()
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(
            groq.transport().chat_completions_url(),
            "https://api.groq.com/openai/v1/chat/completions"
        );
        assert_eq!(groq.quirks().namespace, "openai-compat:groq");
        assert!(groq.reads_namespace(Groq::NAMESPACE));
        assert!(!groq.reads_namespace(Groq::PROVIDER_NAME));
        assert_eq!(
            groq.quirks().max_tokens_field,
            MaxTokensField::MaxCompletionTokens
        );

        let mistral = Mistral::builder("k")
            .unwrap()
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(mistral.quirks().seed_field, SeedField::RandomSeed);

        let ollama = Ollama::builder("http://box:11434/")
            .unwrap()
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(ollama.transport().base_url, "http://box:11434/v1");
        // The scheme-less forms `OLLAMA_HOST` takes are read as Ollama
        // reads them (usability re-run: `build()` refused them).
        for (host, base_url) in [
            ("box", "http://box:11434/v1"),
            ("127.0.0.1:1234", "http://127.0.0.1:1234/v1"),
            ("0.0.0.0", "http://127.0.0.1:11434/v1"),
            ("ollama.com", "https://ollama.com:443/v1"),
        ] {
            let ollama = Ollama::builder(host)
                .unwrap()
                .http_transport(allama_http::ReplayTransport::default())
                .build()
                .unwrap();
            assert_eq!(ollama.transport().base_url, base_url, "{host}");
        }
        assert!(ollama.transport().auth.is_none());
        assert_eq!(ollama.transport().timeout, None);
        assert_eq!(
            ollama.transport().stream_timeouts,
            StreamTimeouts::unbounded()
        );
        let vllm = Vllm::builder("http://box:8000/v1", None)
            .unwrap()
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(
            (vllm.transport().timeout, vllm.transport().stream_timeouts),
            (None, StreamTimeouts::unbounded())
        );
        assert_eq!(
            Groq::builder("k")
                .unwrap()
                .http_transport(allama_http::ReplayTransport::default())
                .build()
                .unwrap()
                .transport()
                .stream_timeouts,
            StreamTimeouts::default()
        );

        let azure = Azure::builder("res", "k")
            .unwrap()
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(
            azure.transport().base_url,
            "https://res.openai.azure.com/openai/v1"
        );
        assert_eq!(azure.transport().auth.as_ref().unwrap().0, "api-key");

        let cf = Cloudflare::builder("0123456789abcdef0123456789abcdef", "tok")
            .unwrap()
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert!(cf.quirks().drop_zero_usage_chunks);

        assert!(Groq::builder(" ").is_err());
        assert!(Cloudflare::builder("", "t").is_err());
    }
}
