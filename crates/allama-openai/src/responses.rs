//! OpenAI Responses API provider (`POST /v1/responses`), stateless by default.
//!
//! [`ResponsesProvider`] sends every request with `store: false` and the
//! full conversation as `input`, unless the request opts into server-side
//! state (see [Server-side state](#server-side-state)). What
//! the Responses API adds over Chat Completions is reasoning state:
//! reasoning items come back with `encrypted_content` (by default when
//! `store` is false; no `include` is needed), which this adapter keeps on
//! the [`Reasoning`](allama::Reasoning) block and replays in later turns.
//! That makes multi-turn tool use with reasoning models work without
//! server-side storage, and conversations survive persistence with
//! [`allama::history::History`].
//!
//! It is the generic [`allama_openai_compat::ResponsesProvider`] with the
//! OpenAI dialect ([`ResponsesQuirks::openai`](allama_openai_compat::ResponsesQuirks::openai))
//! and the effort levels of the OpenAI models that document them, the way [`Provider`](crate::Provider)
//! builds on the compat Chat Completions provider. Other Responses-compatible
//! endpoints (vLLM, Ollama, OpenRouter, …) should use the compat provider
//! with their own identity and namespace instead of this one.
//!
//! ```no_run
//! use allama::prelude::*;
//! use allama::{ReasoningConfig, ReasoningOutput};
//! use allama_openai::ResponsesProvider;
//!
//! # async fn example() -> allama::Result<()> {
//! let provider = ResponsesProvider::from_env()?;
//! let request = ChatRequest::new("gpt-5-mini")
//!     .user("Why is the sky blue? One sentence.")
//!     .reasoning(
//!         ReasoningConfig::new()
//!             .effort(ReasoningEffort::Low)
//!             .output(ReasoningOutput::Summary),
//!     );
//! let response = provider.chat(&request).await?;
//! println!("{}", response.reasoning_text().unwrap_or_default());
//! println!("{}", response.text_or_empty());
//! # Ok(()) }
//! ```
//!
//! # Server-side state
//!
//! Stateless replay is the portable baseline. [`ResponsesRequestOptions`]
//! opts into OpenAI's stored state when that is worth it (long
//! conversations, background work):
//!
//! - `store: true` keeps the response with OpenAI;
//! - `previous_response_id` (an earlier stored [`ChatResponse::id`](allama::ChatResponse::id))
//!   or `conversation` continues from stored context, so the request's
//!   messages carry only the new turn;
//! - `background: true` (with `store`) returns as soon as the response is
//!   queued: `finish_reason` is `None` and `finish_reason_raw` holds the
//!   status until [`ResponsesProvider::retrieve_response`] reports it done;
//!   [`ResponsesProvider::cancel_response`] stops it.
//! - A background *stream* that drops can be resumed:
//!   [`StreamPosition::from_error`](crate::StreamPosition::from_error) reads
//!   where it stopped, and [`ResponsesProvider::resume_stream`] continues
//!   from there without repeating events.
//!
//! # Namespaces
//!
//! - Provider identity and provider-data namespace: `"openai"`, the same
//!   origin as the Chat Completions [`Provider`](crate::Provider). Sharing
//!   it is safe because Chat Completions never produces or replays reasoning
//!   items, function-call item ids, or OpenAI-origin `Other` content: the
//!   replay state this adapter reads is only ever written by this adapter,
//!   and conversations can move between the two OpenAI providers.
//! - Request options: [`ResponsesRequestOptions`], namespace
//!   `"openai-responses"` (the APIs accept different fields).
//! - System prompts: [`SystemOptions`](crate::SystemOptions) (namespace
//!   `"openai"`), shared with Chat Completions.
//!
//! # Mapping notes
//!
//! - System prompts become leading `system` input messages, or `developer`
//!   with [`SystemOptions::developer`](crate::SystemOptions::developer).
//! - `ToolDefinition.strict` is always sent explicitly: the Responses API makes
//!   tools that omit it strict (rewriting optional parameters as required),
//!   unlike other providers.
//! - `ReasoningConfig.effort` maps to `reasoning.effort`: the nearest level
//!   a model documents (the original GPT-5 models, the o-series and GPT-6
//!   Astra), with an `Approximated` warning, and unchanged for any other
//!   model. Replace the levels with
//!   [`ResponsesProvider::with_model_rules`];
//!   `ReasoningOutput::Summary` → `reasoning.summary: "auto"`, `Full` →
//!   `"detailed"` (OpenAI never returns raw reasoning; `Approximated`),
//!   `Omitted` → no summary.
//! - Temperature, top-p, penalties and reasoning settings are sent as asked,
//!   whatever the model (ADR 0015). A model that rejects them gets OpenAI's
//!   rejection, an [`Error::Unsupported`](allama::Error::Unsupported) with
//!   stage `Provider`; [`ResponsesProvider::with_model_rules`] drops them
//!   before sending instead. Seed and stop sequences do not exist in the
//!   Responses API: they are dropped with a warning, or fail under
//!   `UnsupportedPolicy::Error`.
//! - `PromptCache.key` → `prompt_cache_key`, retention →
//!   `prompt_cache_retention`; cache breakpoints are satisfied by automatic
//!   prefix caching.
//! - Reasoning and `Other` content from other providers are not replayed
//!   (`Dropped` warnings). Function calls are portable: another provider's
//!   tool calls and their results are replayed as `function_call` and
//!   `function_call_output` items.
//! - Hosted tools (web search, file search, …) are opt-in through
//!   [`ResponsesRequestOptions::hosted_tools`]; their calls and results come
//!   back as [`ProviderToolCall`](allama::ProviderToolCall) and
//!   [`ProviderToolResult`](allama::ProviderToolResult) blocks with origin
//!   `"openai"`.
//! - Client-executed tools that are not function tools (custom tools, the
//!   shell tools, computer use) come back as `Other` content with origin
//!   `"openai"`; see [Client-executed tools](#client-executed-tools).
//! - Non-streaming requests time out after 120 seconds by default; streams
//!   have no total timeout.
//!
//! # Client-executed tools
//!
//! Some tools are run by the application but are not function tools:
//! custom tools (`{"type": "custom"}`, freeform text input with an
//! optional grammar), the shell tool in local mode, the legacy local shell
//! tool, computer use and `apply_patch`. Their calls and answers are
//! specific to the Responses API, so they are not portable
//! [`ToolCall`](allama::ToolCall)s (ADR 0014):
//!
//! - declare them in [`ResponsesRequestOptions::client_tools`];
//! - their calls arrive as [`Other`](allama::ContentBlock::Other) content.
//!   [`ChatResponse::tool_calls`](allama::ChatResponse::tool_calls) does not
//!   return them, and the finish reason is
//!   [`ToolCalls`](allama::FinishReason::ToolCalls) while one waits for its
//!   answer;
//! - [`ClientToolCall::pending`](crate::ClientToolCall::pending) lists the
//!   calls of a turn that wait for an answer, as typed views
//!   ([`CustomToolCall`](crate::CustomToolCall),
//!   [`LocalShellCall`](crate::LocalShellCall),
//!   [`ShellCall`](crate::ShellCall)); each view builds its answer as an `Other`
//!   [`ContentPart`](allama::ContentPart) for a user message after the
//!   assistant turn;
//! - [`allama::history::validate`] checks only function tool calls: it does
//!   not report a client-executed call without an answer, and OpenAI
//!   rejects that request;
//! - custom tool calls have the same shape on Chat Completions
//!   ([`Provider`](crate::Provider)), so a conversation with them moves
//!   between the two APIs; declare the tools in both APIs' options
//!   ([`ChatRequestOptions::client_tools`](crate::ChatRequestOptions::client_tools)).
//!   An answer whose call is not in the request is sent, for
//!   [`previous_response_id`](ResponsesRequestOptions::previous_response_id)
//!   continuation; an answer whose call is in the request but cannot be
//!   replayed (another endpoint's) is dropped with it, with a warning.
//!
//! ```no_run
//! use allama::prelude::*;
//! use allama_openai::{ClientToolCall, ResponsesRequestOptions, ShellCommandOutput};
//! use serde_json::json;
//!
//! # async fn example(provider: allama_openai::ResponsesProvider) -> Result<()> {
//! let options = ResponsesRequestOptions::new()
//!     .client_tool(serde_json::from_value(json!({
//!         "type": "custom",
//!         "name": "sql",
//!         "description": "Run a read-only SQLite query.",
//!         "format": {"type": "grammar", "syntax": "regex", "definition": "SELECT .+"},
//!     }))?)
//!     .client_tool(serde_json::from_value(
//!         json!({"type": "shell", "environment": {"type": "local"}}),
//!     )?);
//! let mut request = ChatRequest::new("gpt-5.1")
//!     .user("How many users signed up today? Check the logs too.")
//!     .with_option(options);
//! loop {
//!     let response = provider.chat(&request).await?;
//!     let pending = ClientToolCall::pending(&response.content);
//!     if pending.is_empty() && response.tool_calls().next().is_none() {
//!         println!("{}", response.text_or_empty());
//!         return Ok(());
//!     }
//!     request.push_message(response.to_assistant_message());
//!     // Function tools are answered with tool messages, as usual.
//!     for call in response.tool_calls() {
//!         request.push_tool_error(call, "no function tools here");
//!     }
//!     let mut answers = Vec::new();
//!     for call in pending {
//!         answers.push(match call {
//!             ClientToolCall::Custom(call) => call.output(format!("ran `{}`: 42", call.input)),
//!             ClientToolCall::Shell(call) => call.output(
//!                 call.commands.iter().map(|_| ShellCommandOutput::exited("ok\n", "", 0)),
//!             ),
//!             other => {
//!                 return Err(Error::unsupported(format!("cannot run {:?}", other.call_id())));
//!             }
//!         });
//!     }
//!     if !answers.is_empty() {
//!         request.push_message(Message::user(answers));
//!     }
//! }
//! # }
//! ```

use std::borrow::Cow;
use std::sync::Arc;

use allama::adapter::chat::Prepared;
use allama::{
    CapabilitySupport, ChatCapability, ChatCapabilityResolver, ChatProvider, ChatRequest,
    ChatResponse, ChatStream, ChatValidation, ProviderIdentity, Result, Subject,
    ValidationCoverage, Warning, WarningKind,
};
use allama_http::config::redact_url;
use allama_http::{CredentialSlot, HttpRequest};
use allama_openai_compat::{ResponsesModelRules, ResponsesQuirks, ResponsesWireOptions};

#[cfg(all(test, feature = "reqwest"))]
mod conformance_tests;
mod options;

pub use options::ResponsesRequestOptions;

use crate::model::{Endpoint, ModelProfile};
use crate::{NAMESPACE, PROVIDER_NAME};

/// The OpenAI Responses API provider (stateless).
///
/// Clone is cheap: internals are wrapped in `Arc`.
#[derive(Clone)]
pub struct ResponsesProvider {
    compat: allama_openai_compat::ResponsesProvider,
    resolver: Option<Arc<dyn ChatCapabilityResolver>>,
    model_rules: Option<Arc<ModelRulesFn>>,
}

type ModelRulesFn = dyn Fn(&str) -> Option<ResponsesModelRules> + Send + Sync;

impl std::fmt::Debug for ResponsesProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResponsesProvider")
            .field("base_url", &redact_url(self.compat.base_url()))
            .finish_non_exhaustive()
    }
}

impl ResponsesProvider {
    /// Create a provider with an API key.
    ///
    /// # Errors
    ///
    /// Fails when the key is empty.
    #[cfg(feature = "reqwest")]
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        Self::builder().api_key(api_key).build()
    }

    /// Create a provider for a base URL that serves the Responses API with
    /// OpenAI semantics (e.g. a proxy).
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
    pub fn builder() -> ResponsesProviderBuilder {
        ResponsesProviderBuilder::default()
    }

    /// Override capability answers. The overrides reach this provider's
    /// own checks and the request rules (a reasoning override decides
    /// whether reasoning settings are sent), so they can upgrade an answer
    /// as well as downgrade it.
    #[must_use]
    pub fn with_chat_capabilities(mut self, resolver: impl ChatCapabilityResolver) -> Self {
        self.resolver = Some(Arc::new(resolver));
        self.install_model_rules()
    }

    /// Set the request rules per model id: the effort levels a model takes,
    /// whether it reasons, and when it rejects sampling. `None` keeps the
    /// built-in rules: the documented effort levels of the original GPT-5
    /// models, the o-series and GPT-6 Astra, and nothing else (ADR 0015).
    /// Rules you return replace the built-in ones for that model, capability
    /// overrides included. See [Model rules](crate#model-rules).
    ///
    /// ```
    /// use allama::ReasoningEffort;
    /// use allama_openai::{ResponsesModelRules, ResponsesProvider};
    ///
    /// # fn example(provider: ResponsesProvider) {
    /// // A newer o-series model takes `xhigh`: send it unchanged.
    /// let provider = provider.with_model_rules(|model: &str| {
    ///     (model == "o5").then(ResponsesModelRules::new)
    /// });
    /// # let _ = provider; }
    /// ```
    #[must_use]
    pub fn with_model_rules<F>(mut self, rules: F) -> Self
    where
        F: Fn(&str) -> Option<ResponsesModelRules> + Send + Sync + 'static,
    {
        self.model_rules = Some(Arc::new(rules));
        self.install_model_rules()
    }

    /// The request rules applied to `model`.
    #[must_use]
    pub fn model_rules(&self, model: &str) -> ResponsesModelRules {
        self.compat.model_rules(model)
    }

    /// Give the toolkit the rules per model: the caller's, or the built-in
    /// ones with the caller's reasoning answer.
    fn install_model_rules(mut self) -> Self {
        let resolver = self.resolver.clone();
        let custom = self.model_rules.clone();
        self.compat = self.compat.with_model_rules(move |model: &str| {
            custom
                .as_ref()
                .and_then(|rules| rules(model))
                .unwrap_or_else(|| {
                    ModelProfile::for_model(model)
                        .with_overrides(|capability| {
                            resolver
                                .as_ref()
                                .and_then(|r| r.chat_capability(model, capability))
                        })
                        .rules()
                })
        });
        self
    }

    /// Use `scope` as the credential scope (see
    /// [`ResponsesProviderBuilder::credential_scope`]). A provider built
    /// without an API key then requires a credential for `scope` on every
    /// request.
    #[must_use]
    pub fn with_credential_scope(mut self, scope: impl Into<Cow<'static, str>>) -> Self {
        self.compat = self.compat.with_credential_scope(scope);
        self
    }

    /// Where this provider takes its per-request credential from, and where
    /// it puts it: `Authorization: Bearer`, under
    /// [`DEFAULT_CREDENTIAL_SCOPE`](crate::DEFAULT_CREDENTIAL_SCOPE) unless the builder chose another scope.
    #[must_use]
    pub fn credential_slot(&self) -> &CredentialSlot {
        self.compat.credential_slot()
    }

    /// The base URL requests are sent to.
    #[must_use]
    pub fn base_url(&self) -> &str {
        self.compat.base_url()
    }

    /// Fetch a stored response by id, for example to poll a background
    /// response until it has a finish reason.
    ///
    /// # Errors
    ///
    /// Transport, HTTP (an unknown or unstored id is an HTTP error), and
    /// decoding errors.
    pub async fn retrieve_response(&self, id: &str) -> Result<ChatResponse> {
        self.compat.retrieve_response(id).await
    }

    /// Cancel a background response and return its state.
    ///
    /// # Errors
    ///
    /// Transport, HTTP, and decoding errors.
    pub async fn cancel_response(&self, id: &str) -> Result<ChatResponse> {
        self.compat.cancel_response(id).await
    }

    /// Stream a background response by id, skipping the events up to
    /// `after_sequence` that were already delivered (see
    /// [`StreamPosition`](allama_openai_compat::StreamPosition), recorded on the
    /// error that interrupted the original stream).
    ///
    /// # Errors
    ///
    /// Transport and HTTP errors.
    pub async fn resume_stream(
        &self,
        response_id: &str,
        after_sequence: Option<u64>,
    ) -> Result<ChatStream> {
        self.compat.resume_stream(response_id, after_sequence).await
    }

    /// The (family, Responses) answers of [`ModelProfile`].
    fn builtin_chat_capability(model: &str, capability: ChatCapability) -> CapabilitySupport {
        ModelProfile::for_model(model).chat_capability(Endpoint::Responses, capability)
    }

    /// Everything before sending: this crate's preflight already ran on
    /// `prepared`; the compat dialect's follows, then its encoding.
    fn encode(
        &self,
        prepared: &mut Prepared<'_>,
        options: &ResponsesWireOptions,
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

    /// Resolve [`ResponsesRequestOptions`], reporting Chat Completions
    /// options this API does not read.
    fn wire_options(request: &ChatRequest) -> Result<ResponsesWireOptions> {
        let options: ResponsesRequestOptions = request.resolve_option()?;
        let mut wire = ResponsesWireOptions::from(options);
        if request.options.contains::<crate::ChatRequestOptions>()
            || request.options.json(NAMESPACE).is_some()
        {
            wire.warnings.push(Warning::new(
                WarningKind::Dropped,
                Subject::Option {
                    namespace: NAMESPACE.into(),
                },
                "the Responses API does not read Chat Completions options (namespace `openai`); they were dropped; use `ResponsesRequestOptions` (namespace `openai-responses`)",
            )
            .with_code("dropped.chat_completions_options"));
        }
        Ok(wire)
    }
}

impl ChatProvider for ResponsesProvider {
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

impl ProviderIdentity for ResponsesProvider {
    fn provider_name(&self) -> &str {
        PROVIDER_NAME
    }

    fn reads_namespace(&self, namespace: &str) -> bool {
        namespace == crate::NAMESPACE
    }

    /// Exposes this provider only: the compat provider it is built on is an
    /// implementation detail, and calling it directly would skip OpenAI's
    /// request handling.
    fn downcast_any(&self, id: std::any::TypeId) -> Option<&(dyn std::any::Any + Send + Sync)> {
        allama::expose(self, id)
    }
}

/// Builder for [`ResponsesProvider`].
///
/// `Debug` shows the settings without the API key, header values, or
/// credentials in the base URL.
#[derive(Default)]
pub struct ResponsesProviderBuilder {
    settings: crate::builder::Settings,
}

impl std::fmt::Debug for ResponsesProviderBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.settings.debug("ResponsesProviderBuilder", f)
    }
}

impl ResponsesProviderBuilder {
    crate::builder::shared_setters!();

    /// The credential scope (default [`DEFAULT_CREDENTIAL_SCOPE`](crate::DEFAULT_CREDENTIAL_SCOPE)).
    /// Per-request credentials addressed to it
    /// (`allama_http::HttpRequestOptions::credential`) are this provider's;
    /// credentials addressed to other scopes are never sent.
    ///
    /// Two OpenAI accounts in one fallback chain need two scopes: by default
    /// they share one. Providers behind one gateway share one.
    ///
    /// With a scope set, the API key is optional. Without one, bring your
    /// own key: every request must carry a credential for the scope, or it
    /// fails with an authored [`allama::Error::Configuration`] before anything is
    /// sent; [`retrieve_response`](ResponsesProvider::retrieve_response),
    /// [`cancel_response`](ResponsesProvider::cancel_response) and
    /// [`resume_stream`](ResponsesProvider::resume_stream) take no request
    /// options, so they always fail that way. With an API key, a request
    /// without a credential for the scope is sent with that key, so a
    /// mistyped scope falls back to it.
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
    /// A `response.completed` event repeats every output of the response,
    /// generated images included: raise the cap for responses larger than
    /// the default.
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
    pub fn build(self) -> Result<ResponsesProvider> {
        let builder = crate::builder::configure!(
            allama_openai_compat::ResponsesProvider::builder(),
            self.settings
        )
        .quirks(ResponsesQuirks::openai())
        .model_rules(|model: &str| ModelProfile::for_model(model).rules());
        Ok(ResponsesProvider {
            compat: builder.build()?,
            resolver: None,
            model_rules: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use allama_http::http::StatusCode;
    use std::time::Duration;

    use serde_json::json;

    use super::*;
    use allama_http::DEFAULT_REQUEST_TIMEOUT;

    use crate::DEFAULT_BASE_URL;

    /// One timeout model for every builder: a
    /// transport over a client of your own keeps the default; only an
    /// explicit `None` removes it.
    #[cfg(feature = "reqwest")]
    #[tokio::test]
    async fn a_custom_client_keeps_the_default_timeout() {
        let transport = allama_http::ReplayTransport::new([allama_http::ReplayResponse::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "",
        )]);
        let provider = ResponsesProvider::builder()
            .api_key("k")
            .http_transport(transport.clone())
            .build()
            .unwrap();
        let _ =
            allama::ChatProvider::chat(&provider, &allama::ChatRequest::new("m").user("hi")).await;
        assert_eq!(
            transport.requests()[0].timeout,
            Some(DEFAULT_REQUEST_TIMEOUT)
        );
        let debug = format!(
            "{:?}",
            ResponsesProvider::builder()
                .api_key("sk-SECRET")
                .base_url("https://u:PASSWORD@gw.example/v1")
        );
        assert!(
            !debug.contains("SECRET") && !debug.contains("PASSWORD"),
            "{debug}"
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
            let mut builder = ResponsesProvider::builder()
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
    fn builder_validates_and_configures() {
        assert!(ResponsesProvider::builder().build().is_err());
        assert!(
            ResponsesProvider::builder()
                .api_key("k")
                .base_url(" ")
                .http_transport(allama_http::ReplayTransport::default())
                .build()
                .is_err()
        );
        let provider = ResponsesProvider::builder()
            .api_key("k")
            .base_url("https://example.com/v1///")
            .organization("org")
            .project(" ")
            .timeout(None)
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(provider.base_url(), "https://example.com/v1");
        assert_eq!(
            provider.compat.responses_url(),
            "https://example.com/v1/responses"
        );
        assert_eq!(provider.compat.quirks(), &ResponsesQuirks::openai());
        assert_eq!(provider.compat.provider_name(), "openai");
        assert_eq!(provider.provider_name(), "openai");
        // Responses state lives under `openai`, shared with Chat
        // Completions; `openai-responses` names only its options.
        assert!(provider.reads_namespace(crate::NAMESPACE));
        assert!(!provider.reads_namespace("openai-responses"));
        assert_eq!(
            provider.compat.model_rules("gpt-5-nano"),
            ModelProfile::for_model("gpt-5-nano").rules()
        );
        assert_eq!(
            ResponsesProvider::builder()
                .api_key("k")
                .http_transport(allama_http::ReplayTransport::default())
                .build()
                .unwrap()
                .base_url(),
            DEFAULT_BASE_URL
        );
    }

    #[test]
    fn capabilities_by_family() {
        let p = ResponsesProvider::builder()
            .api_key("k")
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        for capability in [
            ChatCapability::ReasoningConfig,
            ChatCapability::ReasoningOutput,
            ChatCapability::ReasoningReplay,
        ] {
            assert_eq!(
                p.chat_capability("gpt-5-nano", capability),
                CapabilitySupport::Supported
            );
            // Not claimed either way (ADR 0015): sent, and OpenAI decides.
            assert_eq!(
                p.chat_capability("gpt-4.1-nano", capability),
                CapabilitySupport::Unknown
            );
            assert_eq!(
                p.chat_capability("some-proxy-model", capability),
                CapabilitySupport::Unknown
            );
        }
        assert_eq!(
            p.chat_capability("gpt-4.1-nano", ChatCapability::FileInput),
            CapabilitySupport::Supported
        );
        assert_eq!(
            p.chat_capability("gpt-5", ChatCapability::ImageOutput),
            CapabilitySupport::Unsupported
        );
        let overridden = p.with_chat_capabilities(|_: &str, c| {
            (c == ChatCapability::ImageOutput).then_some(CapabilitySupport::Supported)
        });
        assert_eq!(
            overridden.chat_capability("gpt-5", ChatCapability::ImageOutput),
            CapabilitySupport::Supported
        );
    }

    #[test]
    fn raw_tool_choice_reaches_the_encoder() {
        let request = ChatRequest::new("m").user("hi").with_option(
            crate::ResponsesRequestOptions::new().tool_choice(json!({"type": "web_search"})),
        );
        let options = ResponsesProvider::wire_options(&request).unwrap();
        assert_eq!(options.tool_choice, Some(json!({"type": "web_search"})));
    }

    #[test]
    fn chat_completions_options_are_reported() {
        let typed = ChatRequest::new("m")
            .user("hi")
            .with_option(crate::ChatRequestOptions::new().service_tier("flex"));
        let json = ChatRequest::new("m")
            .user("hi")
            .with_options_json("openai", json!({"service_tier": "flex"}));
        for request in [typed, json] {
            let options = ResponsesProvider::wire_options(&request).unwrap();
            assert_eq!(options.service_tier, None);
            assert_eq!(options.warnings.len(), 1);
            assert_eq!(options.warnings[0].kind, WarningKind::Dropped);
        }
    }

    #[test]
    fn option_typos_fail() {
        let request = ChatRequest::new("m")
            .user("hi")
            .with_options_json("openai-responses", json!({"servce_tier": "flex"}));
        assert!(matches!(
            ResponsesProvider::wire_options(&request),
            Err(allama::Error::InvalidRequest { .. })
        ));
        let request = ChatRequest::new("m")
            .user("hi")
            .with_options_json("openai-responses", json!({"service_tier": "flex"}));
        let options = ResponsesProvider::wire_options(&request).unwrap();
        assert_eq!(options.service_tier.as_deref(), Some("flex"));
        assert!(options.warnings.is_empty());
    }
}
