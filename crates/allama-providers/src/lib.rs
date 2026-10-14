#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
//! Build `allama` providers from a provider key or a model address.
//!
//! A **provider key** names a provider: `openai`, `anthropic`, `groq`. A
//! **model address** adds the model: `openai:gpt-5.5`. The address splits at
//! the first `:`, so model ids that contain colons (`ollama:llama3:8b`) pass
//! through unchanged. There are no aliases: the model id is sent to the
//! provider as written. [`ModelAddress`] parses one and pins the grammar:
//! keys are lowercase (`[a-z0-9][a-z0-9-]*`), and an address with
//! whitespace is refused rather than trimmed.
//!
//! The lookups return the provider and the model id as a pair, which
//! destructures directly into the request; parse a [`ModelAddress`] when
//! you need the key too.
//!
//! # Model addresses are trusted configuration
//!
//! A model address decides which server receives the request, and with it
//! the provider's API key. Read addresses from your own configuration,
//! never straight from an end user. The local presets (`ollama`, `vllm`,
//! `litellm`) need no configuration and default to services on
//! `localhost`, so a user-chosen `ollama:x` or `vllm:x` would make your
//! server call its own loopback ports. When users pick a model, map their
//! choice through an allowlist of addresses you configured:
//!
//! ```no_run
//! # fn example(user_choice: &str) -> allama::Result<()> {
//! const MODELS: [(&str, &str); 2] = [
//!     ("fast", "openai:gpt-5-mini"),
//!     ("smart", "anthropic:claude-opus-4-1"),
//! ];
//! let Some((_, address)) = MODELS.iter().find(|(name, _)| *name == user_choice) else {
//!     return Err(allama::Error::rejected(format!("unknown model {user_choice:?}")));
//! };
//! let (provider, model) = allama_providers::chat::from_model(address)?;
//! # let _ = (provider, model); Ok(()) }
//! ```
//!
//! Each model operation has its own module, mirroring `allama`:
//!
//! - [`chat`]: [`chat::from_provider`] and [`chat::from_model`].
//! - [`embedding`]: [`embedding::from_provider`] and
//!   [`embedding::from_model`].
//! - [`decision`]: [`decision::from_provider`] and [`decision::from_model`].
//!
//! Each has a `_with` twin taking [`ProviderSettings`], such as a credential
//! scope.
//!
//! Built-in providers are configured from the environment with their own
//! `from_env` constructors (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...).
//!
//! Every provider comes back **retrying transient errors** with
//! `allama`'s default [`RetryPolicy`](allama::wrappers::RetryPolicy), which
//! waits out a provider's `Retry-After` of up to 120 s. An interactive
//! application or a fallback pair sets a shorter ceiling, and nothing
//! should wrap the result in another retry (see
//! [The default stack](ProviderSettings#the-default-stack)):
//!
//! ```no_run
//! use std::time::Duration;
//!
//! use allama::wrappers::RetryPolicy;
//! use allama_providers::ProviderSettings;
//!
//! # fn example() -> allama::Result<()> {
//! let interactive = ProviderSettings::new()
//!     .retry(Some(RetryPolicy::new().max_retry_after(Duration::from_secs(5))));
//! let (provider, model) = allama_providers::chat::from_model_with("openai:gpt-5.5", &interactive)?;
//! # let _ = (provider, model); Ok(()) }
//! ```
//!
//! ```no_run
//! use allama::prelude::*;
//!
//! # async fn example() -> allama::Result<()> {
//! // The model chosen per request:
//! let provider = allama_providers::chat::from_provider("anthropic")?;
//! provider
//!     .chat(&ChatRequest::new("claude-haiku-4-5").user("Say hello."))
//!     .await?;
//!
//! // One string selects both, for example from configuration:
//! let (provider, model) = allama_providers::chat::from_model("openai:gpt-5.5")?;
//! provider.chat(&ChatRequest::new(model).user("Say hello.")).await?;
//! # Ok(()) }
//! ```
//!
//! # Built-in providers
//!
//! | Key | Chat | Embeddings | Decisions | Feature |
//! |---|---|---|---|---|
//! | `anthropic` | `allama_anthropic::Provider` | | | `anthropic` |
//! | `gemini` | `allama_gemini::Provider` | `allama_gemini::Provider` | | `gemini` |
//! | `openai` | `allama_openai::ResponsesProvider` | `allama_openai::Provider` | | `openai` |
//! | `openai-chat` | `allama_openai::Provider` (Chat Completions) | | | `openai` |
//! | `openai-responses` | `allama_openai::ResponsesProvider`, as `openai` | | | `openai` |
//! | `azure-responses` | `allama_openai_compat::ResponsesProvider` (`Azure::responses_from_env`) | | | `openai-compat` |
//! | `github-copilot` | `allama_openai_compat` preset (`GitHubCopilot`, Chat Completions) | | | `openai-compat` |
//! | `azure`, `deepinfra`, `deepseek`, `fireworks`, `groq`, `mistral`, `together`, `vllm`, `xai` | `allama_openai_compat` preset | same preset | | `openai-compat` |
//! | `cloudflare`, `ollama`, `openrouter` | `allama_openai_compat` preset | same preset | `allama_systemone` preset | `openai-compat`, `systemone` |
//! | `digitalocean`, `litellm`, `typesafe` | | | `allama_systemone` preset | `systemone` |
//!
//! A key names one provider configuration; its operations may come from
//! different crates. `ollama` reaches the same server (`OLLAMA_HOST`) for
//! chat, embeddings, and decisions, and `cloudflare` the same Workers AI
//! account (`CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`), with Clef
//! models (`cloudflare:@cf/cloudflare/clef`) for decisions.
//!
//! Azure, like OpenAI, has two chat APIs: `azure` is Chat Completions and
//! `azure-responses` the Responses API, at the same resource with the same
//! key. `openai-responses` is `openai` under the name Pydantic AI also
//! accepts, so its model strings resolve unchanged. The keys named after an
//! API (`openai-chat`, `openai-responses`, `azure-responses`) offer chat
//! only. `github-copilot`, Pydantic AI's name too, is chat only: Copilot
//! serves no embeddings.
//!
//! A built-in key reaches the same adapter and endpoint in every 0.x
//! release. `gemini` is Gemini's `generateContent` API, which Google now
//! calls legacy but fully supported; `gemini-interactions` is reserved for
//! its Interactions API, which is not supported yet.
//!
//! Embeddings are offered wherever the provider type implements
//! `EmbeddingProvider`; whether a given endpoint serves embeddings is up to
//! the endpoint.
//!
//! # Credential scopes
//!
//! A per-request credential (`allama_http::HttpRequestOptions::credential`)
//! is addressed to a credential scope, and only the provider with that scope
//! sends it. Each built-in provider's default scope is its provider name:
//!
//! | Key | Default credential scope |
//! |---|---|
//! | `anthropic` | `anthropic` |
//! | `gemini` | `gemini` |
//! | `openai`, `openai-chat`, `openai-responses` | `openai` |
//! | `azure`, `azure-responses` | `azure` |
//! | other `openai-compat` and `systemone` presets | the key (`groq`, `ollama`, `typesafe`, ...) |
//!
//! Each provider crate publishes its default as `DEFAULT_CREDENTIAL_SCOPE`
//! (on each preset type for presets). The `_with` functions build a provider
//! with another scope, for example one per tenant account:
//!
//! ```no_run
//! use allama_providers::ProviderSettings;
//!
//! # fn example() -> allama::Result<()> {
//! let settings = ProviderSettings::new().credential_scope("tenant-acme");
//! let (provider, model) = allama_providers::chat::from_model_with("openai:gpt-5.5", &settings)?;
//! # let _ = (provider, model); Ok(()) }
//! ```
//!
//! The provider still reads its key from the environment, and sends it when
//! a request carries no credential for the scope.
//!
//! # Custom providers
//!
//! A [`Registry`] maps keys to provider constructors. Start from
//! [`Registry::builtin`] to add or replace providers, or from
//! [`Registry::new`] to choose every one:
//!
//! ```no_run
//! # #[cfg(feature = "openai-compat")] {
//! use allama_openai_compat::providers::Vllm;
//! use allama_providers::Registry;
//!
//! # fn example() -> allama::Result<()> {
//! let mut registry = Registry::builtin();
//! registry.register_chat("local", || Vllm::new("http://gpu-box:8000/v1", None));
//!
//! let (provider, model) = registry.chat_from_model("local:qwen3-32b")?;
//! # Ok(()) }
//! # }
//! ```
//!
//! A registered provider gets the same retries and other wrappers as a
//! built-in one. A constructor registered with `register_chat` takes no
//! settings, so a `_with` lookup that asks for a credential scope or chat
//! capability overrides fails for its key; one registered with
//! [`Registry::register_chat_with`] takes them from [`AdapterSettings`] and
//! builds with them, and a lookup fails when it leaves one untaken.
//!
//! The free functions use the built-in registry, which cannot be changed:
//! a provider key of your own needs a `Registry` of your own, passed to
//! the code that looks addresses up. There is no process-wide registration
//! (see ADR 0016), so a library can never change what another library's
//! `chat::from_model` resolves to.
//!
//! A provider you build yourself, outside any registry, gets the same
//! wrappers from [`chat::wrap`], [`embedding::wrap`] or
//! [`decision::wrap`]. They refuse the settings only a constructor can
//! apply, as a lookup of a constructor registered without settings does:
//!
//! ```no_run
//! # #[cfg(feature = "openai-compat")] {
//! use allama_openai_compat::providers::Vllm;
//! use allama_providers::ProviderSettings;
//!
//! # fn example() -> allama::Result<()> {
//! let adapter = Vllm::new("http://gpu-box:8000/v1", None)?;
//! let provider = allama_providers::chat::wrap(adapter, &ProviderSettings::new())?;
//! # let _ = provider; Ok(()) }
//! # }
//! ```

mod address;
mod builtin;
mod registry;
mod settings;
mod stack;

pub mod chat;
pub mod decision;
pub mod embedding;

pub use address::ModelAddress;
pub use registry::Registry;
pub use settings::{AdapterSettings, ProviderSettings};
