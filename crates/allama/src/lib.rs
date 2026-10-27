#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
//! Low-level, provider-agnostic LLM abstractions for Rust.
//!
//! `allama` is the low-level layer of an LLM stack: portable request and
//! response types, streaming, and provider traits, implemented by provider
//! crates (`allama-openai`, `allama-anthropic`, `allama-gemini`, …). It is a
//! building block, not an agent framework: there are no agent loops here.
//!
//! Status: pre-1.0, so a minor release may break the API; every `allama-*`
//! crate shares one version.
//!
//! # Quick start
//!
//! Pick the provider and model at runtime from one string, a model address
//! such as `openai:gpt-5.5`, with the `allama-providers` crate:
//!
//! ```toml
//! [dependencies]
//! allama = "0.1"
//! allama-providers = "0.1"
//! anyhow = "1"
//! tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
//! ```
//!
//! ```no_run
//! # // `allama-providers` depends on this crate, so the doc test stands in
//! # // for it here. The README runs the same program against the real
//! # // crate, and `allama-examples/tests/guide_quotes.rs` checks that the
//! # // two stay the same.
//! # mod allama_providers {
//! #     pub mod chat {
//! #         pub fn from_model(_: &str) -> allama::Result<(allama::DynChatProvider, String)> {
//! #             unimplemented!()
//! #         }
//! #     }
//! # }
//! use allama::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Reads OPENAI_API_KEY. Try "anthropic:claude-haiku-4-5",
//!     // "gemini:gemini-3.5-flash" or "ollama:qwen3:1.7b".
//!     let (provider, model) = allama_providers::chat::from_model("openai:gpt-5.5")?;
//!
//!     let request = ChatRequest::new(model).user("Explain Rust ownership in two sentences.");
//!     let response = provider.chat(&request).await?;
//!     println!("{}", response.text_or_empty());
//!     Ok(())
//! }
//! ```
//!
//! `main` returns `anyhow::Result` so that a failure prints the provider's
//! message, which [`Error`]'s `Debug` withholds.
//!
//! To print the answer as it is generated, stream it. The collector builds
//! the same [`ChatResponse`] that `chat` returns:
//!
//! ```
//! use allama::prelude::*;
//!
//! async fn stream(provider: &DynChatProvider, request: &ChatRequest) -> Result<ChatResponse> {
//!     let mut stream = provider.chat_stream(request).await?.with_collector();
//!     while let Some(item) = stream.next_event().await {
//!         if let StreamEvent::TextDelta { text, .. } = item?.event {
//!             print!("{text}");
//!         }
//!     }
//!     stream.finish()
//! }
//! ```
//!
//! Before production, read the README's
//! ["Before production"][before production]
//! checklist. Providers from `allama_providers` wait out a provider's
//! `Retry-After` of up to **120 s** by default, which suits batch work but
//! not someone waiting ([`RetryPolicy::max_retry_after`](wrappers::RetryPolicy::max_retry_after)).
//! The checklist also covers refusals and cut-off answers, reasoning under
//! `max_output_tokens`, settings a model cannot take, warnings, and
//! credentials.
//!
//! The terms these docs use, such as the
//! [commit point],
//! a [model address],
//! an [attempt timeout or a deadline]
//! and a [pending tool call],
//! are defined in the
//! [glossary].
//!
//! # Capabilities
//!
//! Each model operation is a sibling trait over [`ProviderIdentity`]:
//!
//! - [`ChatProvider`]: one-shot and streaming chat.
//! - [`EmbeddingProvider`]: batch text embedding.
//! - [`DecisionProvider`]: typed questions answered with probabilities
//!   (decision models; see [`decision`]).
//! - [`ExtractionProvider`]: typed fields answered with values (extraction
//!   models; see [`extraction`]). Unstable, see [Stability](#stability).
//!
//! # Portability model
//!
//! The portable core covers what genuinely aligns across providers. What
//! does not align stays provider-specific, in three deliberate places:
//!
//! - **Provider options** ([`RequestOptions`]): each provider declares an
//!   options type; callers attach the exact type or namespaced JSON.
//! - **Provider data** ([`ProviderData`]): namespaced, serializable state on
//!   content, messages, tools, and responses (signatures, item ids, …). The
//!   namespace is the data's origin; adapters only replay their own.
//! - **Other content** ([`OtherContent`]): blocks the portable model does not
//!   represent, preserved verbatim with their origin and provider data.
//!
//! Adapters never drop portable features silently: they fail with
//! [`Error::Unsupported`] or report a [`Warning`] (see [`UnsupportedPolicy`]).
//!
//! # Modules
//!
//! - [`wrappers`]: retry, fallback, timeout, model-mapping, request-transform,
//!   recording, and tracing wrappers.
//! - [`decision`]: decision-model questions and answers, and typed decisions
//!   (derives with feature `derive`).
//! - [`extraction`]: extraction-model fields and answers (unstable).
//! - [`chat`] and [`embedding`]: an overview of the chat and embedding
//!   types, which are exported at the crate root (see [Paths](#paths));
//!   [`chat::stream`] holds stream collection and commit-point helpers.
//! - [`adapter`]: the provider-author toolkit, such as the
//!   [`prepare`](adapter::chat::prepare) step providers run before
//!   sending, which makes validation ([`Validation`]) a dry run of the
//!   call, and the JSON Schema passes.
//! - [`warning`]: warnings, the unsupported-feature policy, and warning
//!   events.
//! - [`chat::extract`] (feature `chat-extract`): chat extraction, typed structured output from a chat model.
//! - [`provider_data`]: provider-scoped data and replay rules.
//! - [`history`]: versioned conversation history for persistence, and
//!   tool-call validation before sending.
//! - `mock` (feature `mock`): test doubles.
//!
//! # Paths
//!
//! The crate root is the path of every type it exports:
//! `allama::ChatRequest`, `allama::StreamEvent`, `allama::EmbeddingRequest`.
//! The [`prelude`] imports from it, and the examples and guides use it. The
//! modules hold only what the root does not export: the stream helpers in
//! [`chat::stream`] (`until_committed`, `StreamCollector`,
//! `StreamTranscript`, …) and the decision questions and answers in
//! [`decision`] (`Question`, `Answer`, `Decision`, …). Two kinds of root
//! type keep a second path: a few are defined and documented in a module
//! ([`StreamEvent`] in [`chat::stream`], [`Warning`] in [`warning`]; import
//! them from the root all the same), and [`decision`] re-exports the root
//! decision types (`DecisionRequest`, `DecisionProvider`, …) for code that
//! imports them alongside the questions and answers. Provider-author types
//! live only in [`adapter`].
//!
//! # Cargo features
//!
//! The core stays lean: every feature but `warning-events` is off by
//! default.
//!
//! | Feature | Enables | Pulls in |
//! | --- | --- | --- |
//! | `warning-events` (default) | Every warning a call reports is also a `tracing` event ([`warning::emit`]), deduplicated by its code, with a target per kind; [`RetryingProvider`](wrappers::RetryingProvider) retries are `allama::retry` events and [`FallbackProvider`](wrappers::FallbackProvider) failovers `allama::fallback` events | `tracing` |
//! | `schema` | `allama::schemars`, `ToolDefinition::from_schema`, `ResponseFormat::from_schema`, field paths in `ToolArgumentsError` | `schemars`, `serde_path_to_error` |
//! | `chat-extract` | `allama::chat::extract` (`ChatExtractExt::extract`, `ChatExtractor`), also in the prelude; implies `schema`; `extract` is an alias | `schemars`, `serde_path_to_error` |
//! | `derive` | `#[derive(Decision, ChoiceSet, Scale)]` in [`decision`] | `allama-derive` (`syn`, `quote`, `proc-macro2`, `heck`) |
//! | `mock` | `allama::mock`: `MockProvider`, `MockStreamingProvider`, `MockEmbeddingProvider`, `MockDecisionProvider`, `MockExtractionProvider`, and other test doubles | nothing |
//! | `tracing` | `wrappers::TracingProvider` (OpenTelemetry GenAI spans), with the resilience summary wrappers report through `wrappers::call_scope`, and retry and fallback attempt events | `tracing` |
//!
//! Deriving tool parameters from a Rust type (`ToolDefinition::from_schema`)
//! needs `schema`:
//!
//! ```toml
//! allama = { version = "0.1", features = ["schema"] }
//! ```
//!
//! # Common tasks
//!
//! Choosing a provider at runtime from a model address (`openai:gpt-5.5`)
//! is `allama_providers::chat::from_model`; the `web_backend` example in
//! `allama-examples` shows a whole chat turn (history, streaming, one tool
//! round trip, errors by kind, usage). The tasks below build on it. For a
//! web backend (storing conversations and the system prompt, an axum SSE
//! handler, failures after the first token, rendering history, testing
//! handlers with mocks), read the [applications guide].
//!
//! Map errors that occur before any output to HTTP status codes with
//! [`chat::stream::until_committed`]: it holds a stream until its first model
//! output, so a rate limit or an overload still becomes a `429` or `503`
//! with `Retry-After` instead of an error event after `200 OK`.
//!
//! ## A wrapper stack for providers built directly from adapter crates
//!
//! A provider from `allama_providers` (a model address, or a provider you
//! built and passed to `allama_providers::chat::wrap`) already retries,
//! with a per-attempt timeout from its `ProviderSettings`: add only
//! fallback, a deadline and tracing around it, as the
//! [applications guide's provider stack]
//! does. Adding a [`RetryingProvider`](wrappers::RetryingProvider) there
//! would nest retries.
//!
//! A provider built directly from an adapter crate (`allama_openai`,
//! `allama_anthropic`, …) retries nothing. Wrap it once at startup.
//! Outermost first: tracing, a deadline on the whole call, fallback, retry,
//! a timeout per attempt, and model mapping around the provider it targets
//! (see [`wrappers`] for why this order):
//!
//! ```
//! # #[cfg(feature = "tracing")] {
//! use std::time::Duration;
//!
//! use allama::wrappers::{
//!     FallbackProvider, MapModelProvider, RetryPolicy, RetryingProvider, TimeoutPolicy,
//!     TimeoutProvider, TracingProvider,
//! };
//! use allama::{ChatProvider, DynChatProvider};
//!
//! fn production(anthropic: DynChatProvider, openai: DynChatProvider) -> impl ChatProvider {
//!     // Bound each attempt; a timed-out attempt is retried. Three attempts
//!     // of 30 s and their backoff end inside the deadline below, with
//!     // about 30 s left for the fallback. A model whose calls take longer
//!     // needs both raised.
//!     let attempt = TimeoutPolicy::new()
//!         .request(Duration::from_secs(30))
//!         .stream_idle(Duration::from_secs(120));
//!     // Someone is waiting: a `Retry-After` longer than 5 s fails over at
//!     // once instead of holding the request.
//!     let retries = RetryPolicy::new().max_retry_after(Duration::from_secs(5));
//!     // A deadline across every attempt, backoff and failover, before any
//!     // output: the whole unary call, and a stream's wait for its first
//!     // output (never `stream_total`, which would cut off long answers).
//!     let deadline = TimeoutPolicy::new().request(Duration::from_secs(120));
//!
//!     TracingProvider::new(TimeoutProvider::new(
//!         FallbackProvider::new(
//!             RetryingProvider::new(TimeoutProvider::new(anthropic, attempt), retries.clone()),
//!             RetryingProvider::new(TimeoutProvider::new(
//!                 // Callers name Claude models; the OpenAI side gets its own.
//!                 MapModelProvider::new(openai, |model: &str| match model {
//!                     "claude-opus-4-1" => "gpt-5.5".into(),
//!                     _ => "gpt-5-mini".into(),
//!                 }),
//!                 attempt,
//!             ), retries),
//!         ),
//!         deadline,
//!     ))
//! }
//! # }
//! ```
//!
//! [`RetryingProvider`](wrappers::RetryingProvider) retries
//! [retryable](Error::is_retryable) errors before any output has streamed,
//! follows a server's [retry directive](Error::server_retry_directive)
//! (`x-should-retry`) when it sends one, honors a provider's `Retry-After`
//! up to its [`max_retry_after`](wrappers::RetryPolicy::max_retry_after)
//! (120 s by default, which suits batch work; the stack above sets 5 s for
//! a server with someone waiting), and gives up on longer hints so the
//! fallback takes over. [`Error::Configuration`] (a missing
//! key, an unknown model address) is never retried or failed over.
//!
//! ## Cross-provider fallback
//!
//! [`FallbackProvider`](wrappers::FallbackProvider) sends the same request
//! to its second provider when the first fails before producing output with
//! an error another provider may not have (an outage, a rate limit, a
//! request it cannot serve; not a bad request, a rejected key or a content
//! filter: see [`should_fallback`](wrappers::should_fallback)). Give
//! the second side its own model ids with
//! [`MapModelProvider`](wrappers::MapModelProvider), as above. Conversations
//! survive the switch: opaque state such as thinking signatures is
//! namespaced by its origin and only replayed to that provider. After a
//! failover, the response says who served it and what failed first:
//!
//! ```
//! use allama::wrappers::FallbackAttempts;
//! use allama::ChatResponse;
//!
//! fn report(response: &ChatResponse) -> allama::Result<()> {
//!     println!("served by {:?}", response.provider);
//!     if let Some(attempts) = response.provider_data.get::<FallbackAttempts>()? {
//!         for attempt in &attempts {
//!             println!("{} failed: {}", attempt.provider, attempt.error_kind);
//!         }
//!     }
//!     Ok(())
//! }
//! ```
//!
//! On failover, do nothing with the history: each adapter drops the state
//! it cannot use for that request only. When a conversation moves to
//! another provider for good, drop the old provider's state once with
//! [`History::retain_provider_state_for`](history::History::retain_provider_state_for),
//! passing the new provider.
//!
//! ## Extraction with cost accounting
//!
//! An extraction may take several provider calls (retries are opt-in), and a
//! failed one was still billed. Both outcomes carry the same
//! [`ChatExtractionMetadata`][chat::extract::ChatExtractionMetadata], with the usage of
//! every pass summed; a failure also keeps the last response:
//!
//! ```
//! # #[cfg(feature = "chat-extract")] {
//! use allama::chat::extract::ChatExtractor;
//! use allama::schemars::{self, JsonSchema};
//! use allama::{ChatProvider, ChatRequest, Error, Usage};
//!
//! #[derive(serde::Deserialize, JsonSchema)]
//! struct Invoice {
//!     total_cents: u64,
//! }
//!
//! async fn invoice<P: ChatProvider>(
//!     provider: &P,
//!     request: &ChatRequest,
//!     bill: impl Fn(&Usage),
//! ) -> allama::Result<Invoice> {
//!     let result = ChatExtractor::new()
//!         .retries(1)
//!         .extract::<_, Invoice>(provider, request)
//!         .await;
//!     match &result {
//!         Ok(extracted) => extracted.metadata.usage.iter().for_each(&bill),
//!         Err(Error::Extract { failure, .. }) => {
//!             failure.metadata().usage.iter().for_each(&bill);
//!             if let Some(response) = failure.response() {
//!                 eprintln!("unusable output: {}", response.text_or_empty());
//!             }
//!         }
//!         Err(_) => {}
//!     }
//!     result.map(|extracted| extracted.value)
//! }
//! # }
//! ```
//!
//! For one call without recovery, `provider.extract::<T>(&request)`
//! ([`ChatExtractExt`][chat::extract::ChatExtractExt]) is enough.
//!
//! ## Embeddings at scale
//!
//! Providers cap the inputs per request. [`embed_chunked`](EmbeddingProviderExt::embed_chunked)
//! splits a batch at the provider's
//! [`embedding_limits`](EmbeddingProvider::embedding_limits) (or the
//! size you pass when the provider reports none) and returns one response
//! with the vectors in input order and the usage summed:
//!
//! ```
//! use allama::{EmbeddingProvider, EmbeddingProviderExt, EmbeddingRequest};
//!
//! async fn index<P: EmbeddingProvider + Sync>(
//!     provider: &P,
//!     documents: Vec<String>,
//! ) -> allama::Result<Vec<Vec<f32>>> {
//!     let request = EmbeddingRequest::new("text-embedding-3-small").inputs(documents);
//!     Ok(provider.embed_chunked(&request, 100).await?.embeddings)
//! }
//! ```
//!
//! Chunks are sent one after another, and one failed chunk fails the call.
//! A provider from `allama_providers` already retries each chunk; wrap one
//! built directly from an adapter crate in a `RetryingProvider`, as
//! [above](#a-wrapper-stack-for-providers-built-directly-from-adapter-crates),
//! but never both, or retries nest. Split very large corpora yourself to
//! checkpoint progress.
//!
//! ## Persisting conversations
//!
//! Requests, responses, and messages serialize with serde.
//! [`History`](history::History) adds a format version, so stored
//! conversations load safely after the format evolves. A history stores
//! the messages only; set the system prompt on every request, so a new
//! prompt applies to old conversations:
//!
//! ```
//! use allama::history::History;
//! use allama::{ChatRequest, ChatResponse};
//!
//! const SYSTEM_PROMPT: &str = "You are a helpful assistant.";
//!
//! fn save(request: &ChatRequest, response: &ChatResponse) -> serde_json::Result<String> {
//!     let mut history = History::from_request(request);
//!     history.messages.push(response.to_assistant_message());
//!     serde_json::to_string(&history)
//! }
//!
//! fn resume(stored: &str, model: &str, message: &str) -> serde_json::Result<ChatRequest> {
//!     let history: History = serde_json::from_str(stored)?;
//!     Ok(history.into_request(model).system(SYSTEM_PROMPT).user(message))
//! }
//!
//! # let first = ChatRequest::new("demo-model").system(SYSTEM_PROMPT).user("Hi");
//! # let reply = ChatResponse::new(vec![allama::ContentBlock::text("Hello!")]);
//! let stored = save(&first, &reply)?;
//! let next = resume(&stored, "demo-model", "Thanks")?;
//! assert_eq!(next.system.len(), 1);
//! assert_eq!(next.messages.len(), 3);
//! # Ok::<(), serde_json::Error>(())
//! ```
//!
//! A conversation that keeps its own prompt, such as a persona the user
//! picked, stores what it needs to rebuild the prompt (the text, or a
//! persona id) next to the history, and builds the [`SystemPrompt`] (cache
//! breakpoint, options) on every request. Message builders append; to
//! replace the messages or the system prompts, assign the public field.
//!
//! The assistant message keeps only the response content. To keep the
//! response id, usage and the rest per turn, store
//! [`ChatResponse::envelope`] with the message (see
//! [The turn envelope](history#the-turn-envelope)).
//!
//! Provider state (signatures, encrypted reasoning) travels inside the
//! messages, so a reloaded conversation continues exactly where it stopped,
//! including on reasoning models. The application's own data about a
//! message (ids, timestamps, UI state) goes in its `app_data`, which is
//! persisted but never sent (see
//! [Application data](history#application-data)).
//!
//! # Stability
//!
//! A minor release (`0.x` to `0.(x+1)`) may break the API; a patch release
//! does not. Beyond that, one module is **unstable** (ADR 0018):
//! [`extraction`] and the [`ExtractionProvider`] trait, grounded in one
//! provider, may change in any minor release with a release note but no
//! migration path, and none of their names are persisted names. They
//! become stable when a second provider implements the trait.
//!
//! # Serialization and stability
//!
//! Most public types serialize with serde. How long their JSON stays
//! readable comes in three tiers:
//!
//! - **Versioned: [`History`](history::History)**, with the messages and
//!   content it holds. It records its format version, a build reads every
//!   version up to its own, and a version bump ships only in a minor
//!   release (see [Format evolution](history#format-evolution)). Store
//!   conversations this way.
//! - **Tolerant: [`TurnEnvelope`](history::TurnEnvelope), the persisted
//!   names, and warning codes.** The persisted names are the serialized
//!   names of capabilities, capability answers, usage counters, embedding
//!   limits, error kinds and extraction failures. Golden fixtures pin them
//!   and the [warning codes](Warning::code), so they change only on
//!   purpose, in a minor release. An envelope written by a newer
//!   version loads in an older one, with what it does not know read as a
//!   fallback (see [The turn envelope](history#the-turn-envelope)).
//! - **Same-version: everything else**, such as [`ChatRequest`],
//!   [`SystemPrompt`], [`ChatResponse`], [`StreamEvent`],
//!   [`ChatExchange`](wrappers::ChatExchange),
//!   [`StreamTranscript`](chat::stream::StreamTranscript) and [`Error`].
//!   Their JSON is for logs, recordings and fixtures that the same allama
//!   version reads back; a value from another version may not load, or
//!   may load with fields left out. For [`Error`] the names are pinned (its
//!   kind, and the persisted names above), while its shape is
//!   same-version.
//!
//! Content the portable model does not represent arrives as
//! [`Other`](ContentBlock::Other) content. Content allama later models
//! (audio output, for example) moves from `Other` to its own variant in a
//! minor release, so match `Other` only as a fallback, not as the contract.
//!
//! Compare stored JSON as values, not bytes. Key order in a JSON object,
//! such as [`ExtraMap`] (application data, provider data), follows
//! `serde_json`'s `preserve_order` feature, which any crate in the build
//! can turn on, so the bytes of the same [`History`](history::History)
//! differ between builds.
//!
#![doc = concat!(
    "[before production]: https://github.com/sagikazarmark/allama/blob/v",
    env!("CARGO_PKG_VERSION"),
    "/README.md#before-production",
)]
#![doc = concat!(
    "[commit point]: https://github.com/sagikazarmark/allama/blob/v",
    env!("CARGO_PKG_VERSION"),
    "/CONTEXT.md#commit-point",
)]
#![doc = concat!(
    "[model address]: https://github.com/sagikazarmark/allama/blob/v",
    env!("CARGO_PKG_VERSION"),
    "/CONTEXT.md#model-address",
)]
#![doc = concat!(
    "[attempt timeout or a deadline]: https://github.com/sagikazarmark/allama/blob/v",
    env!("CARGO_PKG_VERSION"),
    "/CONTEXT.md#attempt-timeout-vs-deadline",
)]
#![doc = concat!(
    "[pending tool call]: https://github.com/sagikazarmark/allama/blob/v",
    env!("CARGO_PKG_VERSION"),
    "/CONTEXT.md#pending-tool-call",
)]
#![doc = concat!(
    "[glossary]: https://github.com/sagikazarmark/allama/blob/v",
    env!("CARGO_PKG_VERSION"),
    "/CONTEXT.md",
)]
#![doc = concat!(
    "[applications guide's provider stack]: https://github.com/sagikazarmark/allama/blob/v",
    env!("CARGO_PKG_VERSION"),
    "/docs/guides/applications.md#the-provider-stack",
)]
#![doc = concat!(
    "[applications guide]: https://github.com/sagikazarmark/allama/blob/v",
    env!("CARGO_PKG_VERSION"),
    "/docs/guides/applications.md",
)]
#![cfg_attr(
    not(feature = "chat-extract"),
    doc = "[`chat::extract`]: crate#cargo-features\n\
           [chat::extract::ChatExtractionMetadata]: crate#cargo-features\n\
           [chat::extract::ChatExtractExt]: crate#cargo-features"
)]

mod capability;
mod error;
mod identity;
mod macros;
mod media;
mod options;
mod prepared;
mod usage;
mod utils;
mod validation;

pub mod adapter;
pub mod chat;
pub mod decision;
pub mod embedding;
pub mod extraction;
pub mod history;
#[cfg(any(test, feature = "mock"))]
pub mod mock;
pub mod provider_data;
pub(crate) mod schema;
pub mod warning;
pub mod wrappers;

pub use capability::{Capability, CapabilitySupport};
pub use chat::cache::{CacheBreakpoint, CacheRetention, PromptCache};
pub use chat::content::{
    Base64Data, Citation, CitationSource, ContentBlock, ContentPart, File, FileKind, Image,
    ImageDetail, MediaSource, OtherContent, ProviderToolCall, ProviderToolResult, Reasoning,
    Refusal, Text, ToolArgumentsError, ToolCall,
};
pub use chat::message::{
    AssistantMessage, Message, ToolMessage, ToolResultContent, UserContent, UserMessage,
};
pub use chat::provider::{
    ChatCapability, ChatCapabilityResolver, ChatProvider, ChatProviderExt, DynChatProvider,
};
pub use chat::request::{
    ChatRequest, ReasoningConfig, ReasoningEffort, ReasoningOutput, ResponseFormat,
};
pub use chat::response::{ChatResponse, FinishReason};
pub use chat::stream::{
    ChatStream, ChatStreamExt, CollectedEvent, SingleResponseStream, StreamBlockType, StreamEvent,
};
pub use chat::system::SystemPrompt;
pub use chat::tool::{ToolChoice, ToolDefinition};
#[doc(inline)]
pub use decision::{
    DecisionCapability, DecisionLimits, DecisionProvider, DecisionProviderExt, DecisionRequest,
    DecisionResponse, DynDecisionProvider,
};
pub use embedding::chunks::EmbeddingChunk;
pub use embedding::input::{EmbeddingInput, EmbeddingPart};
pub use embedding::progress::ChunkedEmbeddingProgress;
pub use embedding::provider::{
    DynEmbeddingProvider, EmbeddingCapability, EmbeddingProvider, EmbeddingProviderExt,
};
pub use embedding::request::{EmbeddingInputType, EmbeddingLimits, EmbeddingRequest};
pub use embedding::response::EmbeddingResponse;
pub use error::{
    Error, ErrorContext, ErrorLog, MessageOrigin, ParseEnumError, Result, SerializationError,
    StreamErrorKind, UnsupportedStage,
};
#[doc(inline)]
pub use extraction::{
    DynExtractionProvider, ExtractionCapability, ExtractionLimits, ExtractionProvider,
    ExtractionRequest, ExtractionResponse,
};
pub use identity::{ProviderIdentity, ProviderIdentityExt, expose};
pub use options::{ProviderOptions, RequestOptions};
pub use provider_data::{ProviderData, ProviderDataEntry};
pub use usage::Usage;
pub use validation::{
    ChatValidation, DecisionValidation, EmbeddingValidation, ExtractionValidation,
    FallbackValidation, Validation, ValidationCoverage,
};
pub(crate) use warning::Degradations;
pub use warning::{
    ReplayState, RequestField, SeenWarnings, Subject, UnsupportedPolicy, Warning, WarningCause,
    WarningKind,
};

/// Re-export of the [`schemars`](https://docs.rs/schemars) version `allama`
/// generates schemas with (feature `schema`).
///
/// Derive `JsonSchema` through it so your types implement the same trait
/// version that [`ToolDefinition::from_schema`] and extraction require,
/// without a separate, possibly mismatched `schemars` dependency: your
/// crate needs no `schemars` dependency of its own. Import `self` too: the
/// derive expands to `schemars::` paths, which resolve to this re-export
/// only when `schemars` is in scope.
///
/// ```
/// use allama::schemars::{self, JsonSchema};
///
/// #[derive(serde::Deserialize, JsonSchema)]
/// struct Weather {
///     city: String,
/// }
///
/// let tool = allama::ToolDefinition::from_schema::<Weather>("get_weather");
/// assert_eq!(tool.parameters["properties"]["city"]["type"], "string");
/// ```
///
/// Importing only the trait (`use allama::schemars::JsonSchema;`) fails
/// inside the derive's expansion with "cannot find module or crate
/// `schemars`", far from the cause. Either import `self` as above, or name
/// the path on the type:
///
/// ```
/// use allama::schemars::JsonSchema;
///
/// #[derive(serde::Deserialize, JsonSchema)]
/// #[schemars(crate = "allama::schemars")]
/// struct Weather {
///     city: String,
/// }
/// # let _ = allama::ToolDefinition::from_schema::<Weather>("get_weather");
/// ```
///
/// `schemars` 1.x is a public dependency on purpose: your types implement
/// its `JsonSchema`. Moving to a new major version of it is a breaking
/// allama release.
#[cfg(feature = "schema")]
pub use schemars;

/// A JSON object: the untyped key/value shape used for verbatim provider data.
///
/// Its key order follows `serde_json`'s `preserve_order` feature, so
/// compare stored JSON as values, not bytes (see
/// [Serialization and stability](crate#serialization-and-stability)).
pub type ExtraMap = serde_json::Map<String, serde_json::Value>;

/// Common imports for application code: `use allama::prelude::*;`.
///
/// The prelude names the request and response types of its traits, so code
/// that stores or forwards a stream or a response needs no second import,
/// and the types those requests and the errors are written with:
/// [`ReasoningConfig`] and [`ReasoningOutput`] beside [`ReasoningEffort`],
/// [`EmbeddingInputType`] beside [`EmbeddingRequest`], [`RequestField`]
/// for matching [`Error::unsupported_field`], and [`ChatCapability`] and
/// [`CapabilitySupport`] for asking a provider what it supports. It has
/// [`ProviderIdentity`], so every provider answers
/// [`provider_name`](ProviderIdentity::provider_name), and
/// [`StreamBlockType`], for matching the
/// [`StreamEvent::BlockStart`] of a tool call while streaming, and
/// [`CollectedEvent`], the item of
/// [`with_collector`](ChatStreamExt::with_collector), for matching it by
/// pattern. It has [`UnsupportedPolicy`], for
/// [`ChatRequest::unsupported_policy`].
///
/// It does not re-export `futures`' `StreamExt`: read a chat stream with
/// [`next_event`](ChatStreamExt::next_event), and import
/// `futures_util::StreamExt` yourself for other stream combinators.
///
/// ```
/// use allama::prelude::*;
///
/// struct Reply {
///     stream: ChatStream,
///     addressed: String,
/// }
///
/// async fn open(provider: &DynChatProvider) -> Result<Reply> {
///     let request = ChatRequest::new("demo-model").user("Hi");
///     let stream = provider.chat_stream(&request).await?;
///     let addressed = provider.provider_name().to_owned();
///     Ok(Reply { stream, addressed })
/// }
///
/// fn starts_tool_call(event: &StreamEvent) -> bool {
///     matches!(
///         event,
///         StreamEvent::BlockStart {
///             block_type: StreamBlockType::ToolCall,
///             ..
///         }
///     )
/// }
/// # assert!(starts_tool_call(&StreamEvent::block_start(0, StreamBlockType::ToolCall)));
///
/// fn text_of(item: &CollectedEvent) -> Option<&str> {
///     match item {
///         CollectedEvent {
///             event: StreamEvent::TextDelta { text, .. },
///             ..
///         } => Some(text),
///         _ => None,
///     }
/// }
/// # assert_eq!(text_of(&CollectedEvent::new(StreamEvent::text_delta(0, "Hi"), None)), Some("Hi"));
///
/// fn needs_reasoning_off(error: &Error) -> bool {
///     error.unsupported_field() == Some(RequestField::Reasoning)
/// }
///
/// let strict = ChatRequest::new("demo-model")
///     .user("Hi")
///     .unsupported_policy(UnsupportedPolicy::Error);
///
/// let query = EmbeddingRequest::new("demo-embed")
///     .input("Where is my order?")
///     .input_type(EmbeddingInputType::Query);
/// # let _ = (strict, query, needs_reasoning_off);
/// ```
pub mod prelude {
    pub use crate::{
        CapabilitySupport, ChatCapability, ChatProvider, ChatProviderExt, ChatRequest,
        ChatResponse, ChatStream, ChatStreamExt, CollectedEvent, ContentBlock, ContentPart,
        DecisionProvider, DecisionProviderExt, DecisionRequest, DecisionResponse, DynChatProvider,
        DynEmbeddingProvider, DynExtractionProvider, EmbeddingInputType, EmbeddingProvider,
        EmbeddingProviderExt, EmbeddingRequest, EmbeddingResponse, Error, ExtractionProvider,
        ExtractionRequest, ExtractionResponse, FinishReason, Message, ProviderIdentity,
        ProviderIdentityExt, ReasoningConfig, ReasoningEffort, ReasoningOutput, RequestField,
        ResponseFormat, Result, StreamBlockType, StreamEvent, SystemPrompt, ToolCall, ToolChoice,
        ToolDefinition, UnsupportedPolicy, Usage,
    };

    #[cfg(feature = "chat-extract")]
    pub use crate::chat::extract::ChatExtractExt;
}
