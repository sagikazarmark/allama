#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
//! Cloudflare Workers AI provider for allama.
//!
//! This crate implements [`allama::ChatProvider`],
//! [`allama::EmbeddingProvider`] and [`allama::DecisionProvider`] on the
//! Worker's AI binding, calling Workers AI models directly from a
//! Cloudflare Worker without HTTP overhead.
//! Construction is runtime-bound: unlike the HTTP providers, this adapter is
//! not configured from process environment variables; it needs the live AI
//! binding of the Worker. The crate targets `wasm32-unknown-unknown`.
//!
//! The binding is reached through this crate's own `wasm_bindgen` import,
//! so no `workers-rs` (`worker`) type is part of its API: [`Provider::new`]
//! and [`Provider::from_env`] take the JS value, which `&worker::Env` and
//! `&worker::Ai` provide, and a `worker` upgrade on either side is not a
//! breaking change.
//!
//! # Usage
//!
//! ```rust,no_run
//! use allama::prelude::*;
//! use allama_cloudflare_worker::Provider;
//!
//! // `env` is the Worker's `&worker::Env` (anything that is a JS value).
//! async fn handler(env: &wasm_bindgen::JsValue) -> Result<String> {
//!     let provider = Provider::from_env(env, "AI")?;
//!
//!     let response = provider
//!         .chat(&ChatRequest::new("@cf/meta/llama-4-scout-17b-16e-instruct").user("What is Rust?"))
//!         .await?;
//!
//!     Ok(response.text_or_empty())
//! }
//! ```
//!
//! # Streaming
//!
//! Streaming runs the model with `stream: true` and parses the SSE byte
//! stream the binding returns, tool calls and the final usage chunk
//! (`usage` and `finish_reason`) included:
//!
//! ```rust,no_run
//! use allama::prelude::*;
//! use allama_cloudflare_worker::Provider;
//! use futures_util::StreamExt;
//!
//! async fn stream_handler(provider: &Provider) -> Result<()> {
//!     let mut stream = provider
//!         .chat_stream(&ChatRequest::new("@cf/meta/llama-4-scout-17b-16e-instruct").user("Tell me a story"))
//!         .await?;
//!
//!     while let Some(event) = stream.next().await {
//!         if let StreamEvent::TextDelta { text, .. } = event? {
//!             print!("{text}");
//!         }
//!     }
//!     Ok(())
//! }
//! ```
//!
//! # Decisions
//!
//! Clef decision models (`@cf/cloudflare/clef`, `@cf/cloudflare/clef-flash`)
//! answer typed questions about an input, and the images next to it, with
//! probabilities. Their input and output are encoded and decoded with
//! `allama-systemone`'s codec, the one its `Cloudflare` preset uses over
//! HTTP; each question's confidence is kept as [`Confidence`].
//!
//! ```rust,no_run
//! use allama::decision::Question;
//! use allama::{DecisionProvider, DecisionRequest};
//! use allama_cloudflare_worker::Provider;
//!
//! async fn is_urgent(provider: &Provider, ticket: &str) -> allama::Result<f64> {
//!     let request = DecisionRequest::new("@cf/cloudflare/clef", ticket)
//!         .question("urgent", Question::yes_no("Is this support request urgent?"));
//!     let response = provider.decide(&request).await?;
//!     Ok(response.yes_no("urgent").map_or(0.0, |answer| answer.probability))
//! }
//! ```
//!
//! # Provider options
//!
//! Workers AI sampling controls without a portable equivalent
//! (`repetition_penalty`) are set through [`ChatRequestOptions`], typed or as
//! JSON under the `"cloudflare-workers-ai"` namespace; `top_k` is the
//! portable `ChatRequest::top_k`. Model inputs this
//! crate does not model go in the options' `extra` map (embeddings too,
//! through [`EmbeddingRequestOptions`]), and the binding call's own options
//! (an AI Gateway, …) in `run_options`.
//!
//! The binding sends no HTTP request, so transport options for HTTP
//! adapters (`allama_http::HttpRequestOptions`, under the `allama-http`
//! options namespace) are ignored without a warning, like options for any
//! other namespace: a request that also goes to HTTP providers (through a
//! fallback, say) carries its headers to them and runs unchanged here.
//!
//! # Degradation
//!
//! Hints the binding cannot honor (stop sequences, `parallel_tool_calls:
//! false`, the reasoning output setting, prompt-cache key and retention,
//! strict tool schemas, JSON Schema name and strict flag) are dropped with
//! a [`allama::Warning`], or rejected under
//! [`allama::UnsupportedPolicy::Error`]. `parallel_tool_calls: true` is
//! dropped silently: it is permission, and models that call tools may
//! call several in one turn (`ChatCapability::ParallelToolCalls` is
//! `Unknown`). Forced tool choice, `response_format` on streaming
//! requests, and file input always fail with
//! [`allama::Error::Unsupported`].
//!
//! # Decoding
//!
//! Workers AI answers in its native shape (`{ response, tool_calls }`) or
//! in the chat-completion shape, often both at once. Responses and stream
//! chunks are normalized into the chat-completion shape and decoded with
//! `allama-openai-compat`'s Chat Completions decoder, the one its
//! `Cloudflare` preset uses, so the binding and the HTTP preset report the
//! same content, finish reasons and capabilities.
//!
//! Image input (OpenAI-style `image_url` parts) and the reasoning effort
//! (`reasoning_effort`) are sent for every model: each model's input schema
//! says whether it takes them, and nothing is refused on the model id
//! (ADR 0015).
//!
//! # Retries
//!
//! [`RetryingProvider`](allama::wrappers::RetryingProvider) backoff jitter
//! may be correlated across Workers isolates, which start fresh and see a
//! frozen clock. This is unverified; see
//! [`RetryPolicy::jitter`](allama::wrappers::RetryPolicy::jitter) if it
//! matters.

mod binding;
mod chat;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod conformance_tests;
mod decision;
mod embedding;
mod error;
mod extraction;
mod options;
mod streaming;
mod wire;

use std::sync::Arc;

use allama::ChatCapabilityResolver;
use wasm_bindgen::JsValue;

pub use allama_cloudflare::ExtractionRequestOptions;
pub use allama_systemone::Confidence;
pub use options::{ChatRequestOptions, DecisionRequestOptions, EmbeddingRequestOptions};

/// Clairic's statistics in an extraction response's provider data, from
/// `allama-cloudflare`, whose codec this crate runs the model with (the
/// two transports share the namespace `cloudflare-workers-ai`).
pub mod extraction_data {
    pub use allama_cloudflare::{
        AnswerExtras, Confidence, Distribution, Distributions, Guesses, RunDetails, TokenCounts,
    };
}

/// The Workers AI binding's output object, kept in
/// [`ChatResponse::provider_data`](allama::ChatResponse::provider_data) when
/// [`ChatRequestOptions::raw_response`] is set (unary calls only), and in
/// [`DecisionResponse::provider_data`](allama::DecisionResponse::provider_data)
/// when [`DecisionRequestOptions::raw_response`] is.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct RawResponse(pub serde_json::Value);

impl allama::ProviderDataEntry for RawResponse {
    const NAMESPACE: &'static str = NAMESPACE;
    const KEY: &'static str = allama::provider_data::RAW_RESPONSE_KEY;
}

/// Namespace of this provider's options and provider data.
pub const NAMESPACE: &str = "cloudflare-workers-ai";

/// The [`provider_name`](allama::ProviderIdentity::provider_name) of
/// [`Provider`], also stamped on
/// [`ChatResponse::provider`](allama::ChatResponse::provider) and on errors
/// and warnings.
///
/// Matches the identity reported by `allama-openai-compat`'s Cloudflare preset
/// so dashboards keyed on `gen_ai.provider.name` see a single value regardless
/// of which transport the user picked. It differs from [`NAMESPACE`]: the two
/// transports share a name but not options or provider data.
pub const PROVIDER_NAME: &str = "cloudflare";

/// Cloudflare Workers AI provider implementing [`allama::ChatProvider`],
/// [`allama::EmbeddingProvider`], [`allama::DecisionProvider`] and
/// [`allama::ExtractionProvider`] (unstable, ADR 0018).
///
/// Calls Workers AI models through the Worker's AI binding, the runtime's
/// internal binding rather than HTTP.
///
/// # Construction
///
/// Build it from the Worker's `env` and the binding name, with
/// [`Provider::from_env`], or wrap the binding object you already hold with
/// [`Provider::new`]. Both take the JS value: `&worker::Env` and
/// `&worker::Ai` from `workers-rs` work as they are, as does a
/// `wasm_bindgen::JsValue`, so no `worker` type is part of this crate's API
/// and any `workers-rs` version fits.
///
/// ```rust,no_run
/// # fn handler(env: &wasm_bindgen::JsValue) -> allama::Result<()> {
/// use allama_cloudflare_worker::Provider;
///
/// // `env` is the Worker's `worker::Env` (or its JS value).
/// let provider = Provider::from_env(env, "AI")?;
/// # let _ = provider; Ok(()) }
/// ```
///
/// The `"AI"` string must match the binding name in `wrangler.toml`:
///
/// ```toml
/// [ai]
/// binding = "AI"
/// ```
pub struct Provider {
    pub(crate) ai: binding::Binding,
    pub(crate) chat_capability_resolver: Option<Arc<dyn ChatCapabilityResolver>>,
    pub(crate) max_stream_event_size: usize,
}

impl std::fmt::Debug for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Provider")
            .field("max_stream_event_size", &self.max_stream_event_size)
            .field(
                "chat_capability_resolver",
                &self.chat_capability_resolver.is_some(),
            )
            .finish_non_exhaustive()
    }
}

impl Provider {
    /// Wrap the Workers AI binding object (`env.AI`): a `&worker::Ai`, or
    /// the binding's JS value.
    ///
    /// The value is not checked here; a value that is not the binding fails
    /// the first call. [`from_env`](Self::from_env) checks it.
    #[must_use]
    pub fn new(binding: impl AsRef<JsValue>) -> Self {
        Self {
            ai: binding::Binding::new(binding.as_ref()),
            chat_capability_resolver: None,
            max_stream_event_size: allama_http::sse::SseDecoder::DEFAULT_MAX_EVENT_SIZE,
        }
    }

    /// Create a provider from the Workers AI binding named `binding` in the
    /// Worker's `env`: a `&worker::Env`, or the `env` object's JS value.
    ///
    /// This reads a binding of the Worker, not process environment
    /// variables as the HTTP providers' `from_env` does.
    ///
    /// # Errors
    ///
    /// Fails with [`Error::Configuration`](allama::Error::Configuration) when
    /// `env` has no Workers AI binding of that name.
    pub fn from_env(env: impl AsRef<JsValue>, binding: &str) -> allama::Result<Self> {
        let value = js_sys::Reflect::get(env.as_ref(), &JsValue::from_str(binding))
            .unwrap_or(JsValue::UNDEFINED);
        if !binding::is_binding(&value) {
            return Err(error::missing_binding(
                binding,
                &format!("`env.{binding}` is not a Workers AI binding"),
            ));
        }
        Ok(Self::new(value))
    }

    /// Install a resolver consulted before the provider's built-in capability logic.
    #[must_use]
    pub fn with_chat_capabilities(mut self, resolver: impl ChatCapabilityResolver) -> Self {
        self.chat_capability_resolver = Some(Arc::new(resolver));
        self
    }

    /// The largest stream event, in bytes, the provider holds in memory
    /// (default [`SseDecoder::DEFAULT_MAX_EVENT_SIZE`], 64 MiB). A larger
    /// event fails the stream with a stream protocol error, which is not
    /// retryable. A Worker isolate has 128 MB of memory, so a Worker that
    /// runs several streams at once may want a lower cap.
    ///
    /// The HTTP adapters set the same limit with their builders'
    /// `max_stream_event_size`. This provider has no builder: it is built
    /// from the binding in one step, and, like
    /// [`with_chat_capabilities`](Self::with_chat_capabilities), its
    /// settings are `with_*` methods on the provider value, the convention
    /// for provider values across the workspace (`with_credential_scope`).
    ///
    /// [`SseDecoder::DEFAULT_MAX_EVENT_SIZE`]: allama_http::sse::SseDecoder::DEFAULT_MAX_EVENT_SIZE
    #[must_use]
    pub fn with_max_stream_event_size(mut self, bytes: usize) -> Self {
        self.max_stream_event_size = bytes;
        self
    }
}
