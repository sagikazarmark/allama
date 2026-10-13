//! A provider for endpoints that speak the OpenAI Responses API.
//!
//! [`ResponsesProvider`] is stateless by default: it sends
//! `POST {base_url}/responses` with `store: false` and the full conversation
//! as `input`. Server-side state is opt-in through [`ResponsesRequestOptions`]:
//! `store`, `previous_response_id` or `conversation` to continue from stored
//! state (the messages then carry only the new turn), and `background` with
//! [`ResponsesProvider::retrieve_response`],
//! [`ResponsesProvider::cancel_response`] and [`ResponsesProvider::resume_stream`]
//! for background responses. How an endpoint deviates from OpenAI's wire
//! format is described by its [`ResponsesQuirks`]; what each model accepts
//! (reasoning effort levels, sampling parameters while reasoning) by
//! [`ResponsesModelRules`].
//!
//! # Namespaces
//!
//! Provider data (reasoning replay state, item ids, annotations, the origin
//! of `Other` content) is written and read under
//! [`ResponsesQuirks::namespace`]. Only state under that namespace is
//! replayed, so a conversation can move between endpoints without sending
//! one endpoint's opaque state to another: it is dropped with a warning.
//!
//! # Mapping notes
//!
//! - System prompts become leading `system` input messages, or `developer`
//!   with [`SystemOptions::developer`](crate::SystemOptions::developer)
//!   where the endpoint accepts it.
//! - `ToolDefinition.strict` is sent explicitly when the endpoint supports strict
//!   tools: OpenAI makes tools that omit it strict.
//! - `ReasoningConfig.effort` maps to `reasoning.effort` (the nearest level
//!   the model rules allow, with an `Approximated` warning);
//!   `ReasoningOutput::Summary` → `reasoning.summary: "auto"`, `Full` →
//!   `"detailed"` (`Approximated`), `Omitted` → no summary.
//! - Reasoning text comes from the summary, or from raw reasoning content
//!   when the endpoint returns no summary (vLLM, llama.cpp). Reasoning is
//!   replayed with the `encrypted_content` and/or raw `content` it was
//!   received with (see [`ReasoningReplay`](crate::ReasoningReplay)).
//! - Seed and stop sequences do not exist in the Responses API: they are
//!   dropped with a warning, or fail under `UnsupportedPolicy::Error`.
//! - Hosted tools are opt-in through
//!   [`ResponsesRequestOptions::hosted_tools`]; each hosted call item comes
//!   back as a [`ProviderToolCall`](allama::ProviderToolCall) and a
//!   [`ProviderToolResult`](allama::ProviderToolResult) with the namespace as
//!   origin, and is replayed as the original item. Other unknown items
//!   surface as `Other` content and are replayed verbatim.
//! - Non-streaming requests time out after [`DEFAULT_REQUEST_TIMEOUT`](allama_http::DEFAULT_REQUEST_TIMEOUT) by default;
//!   streams have no total timeout.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use allama::adapter::chat::Prepared;
use allama::{
    CapabilitySupport, ChatCapability, ChatCapabilityResolver, ChatProvider, ChatRequest,
    ChatResponse, ChatStream, ChatValidation, Error, ProviderIdentity, RequestOptions, Result,
    Subject, ValidationCoverage, Warning, WarningKind,
};

use allama_http::config::redact_url;
use allama_http::sse::{SseChatStream, SseDecoder, SseStream};
use allama_http::{
    Bytes, CredentialPlacement, CredentialSlot, Headers, HttpRequest, HttpResponse, HttpTransport,
    Method, StreamTimeouts,
};

use crate::error::{decode_body, read_body};
use crate::{
    ResponseContext, ResponsesQuirks, StreamingSupport, extract_request_id, extract_retry_after,
    map_http_error,
};

mod decode;
mod encode;
mod error;
mod hosted;
mod model;
mod options;
mod streaming;
mod wire;

pub use model::{ResponsesModelRules, SamplingSupport};
pub use options::{ResponsesRequestOptions, ResponsesWireOptions};

/// A stateless provider for an endpoint that speaks the OpenAI Responses
/// API, configured with a [`ResponsesQuirks`] dialect.
///
/// Clone is cheap: internals are wrapped in `Arc`.
///
/// ```no_run
/// use allama::prelude::*;
/// use allama_openai_compat::{ResponsesProvider, ResponsesQuirks};
///
/// # async fn example() -> allama::Result<()> {
/// let provider = ResponsesProvider::builder()
///     .base_url("http://localhost:8000/v1")
///     .provider_name("vllm")
///     .quirks(ResponsesQuirks::new("openai-compat:vllm"))
///     .build()?;
/// let text = provider.ask("openai/gpt-oss-20b", "Say hello").await?;
/// # let _ = text; Ok(()) }
/// ```
#[derive(Clone)]
pub struct ResponsesProvider {
    inner: Arc<Inner>,
}

#[derive(Clone)]
struct Inner {
    http: Arc<dyn HttpTransport>,
    base_url: String,
    responses_path: String,
    auth: Option<(String, String)>,
    /// The headers sent with every request: the `User-Agent`, the
    /// organization and project headers, and the builder's own.
    headers: Headers,
    request_id_header_name: String,
    retry_after_header_name: String,
    timeout: Option<Duration>,
    stream_timeouts: StreamTimeouts,
    max_stream_event_size: usize,
    max_response_size: usize,
    quirks: ResponsesQuirks,
    model_rules: Option<ModelRulesFn>,
    chat_capabilities: HashMap<ChatCapability, CapabilitySupport>,
    chat_capability_resolver: Option<Arc<dyn ChatCapabilityResolver>>,
    provider_name: Cow<'static, str>,
    credential_slot: CredentialSlot,
}

type ModelRulesFn = Arc<dyn Fn(&str) -> ResponsesModelRules + Send + Sync>;

impl std::fmt::Debug for ResponsesProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResponsesProvider")
            .field("provider_name", &self.inner.provider_name)
            .field("base_url", &redact_url(&self.inner.base_url))
            .finish_non_exhaustive()
    }
}

impl ResponsesProvider {
    /// Start configuring a provider.
    #[must_use]
    pub fn builder() -> ResponsesProviderBuilder {
        ResponsesProviderBuilder::default()
    }

    /// Override capability answers with a resolver.
    #[must_use]
    pub fn with_chat_capabilities(self, resolver: impl ChatCapabilityResolver) -> Self {
        let mut inner = (*self.inner).clone();
        inner.chat_capability_resolver = Some(Arc::new(resolver));
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Replace the request rules per model id (see
    /// [`ResponsesProviderBuilder::model_rules`]). An adapter built on this
    /// provider uses it to make its capability overrides reach the rules.
    #[must_use]
    pub fn with_model_rules<F>(self, rules: F) -> Self
    where
        F: Fn(&str) -> ResponsesModelRules + Send + Sync + 'static,
    {
        let mut inner = (*self.inner).clone();
        inner.model_rules = Some(Arc::new(rules));
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Use `scope` as the credential scope (see
    /// [`ResponsesProviderBuilder::credential_scope`]). A provider without
    /// configured auth then requires a credential for `scope` on every
    /// request.
    #[must_use]
    pub fn with_credential_scope(self, scope: impl Into<Cow<'static, str>>) -> Self {
        let mut inner = (*self.inner).clone();
        inner.credential_slot =
            crate::rescope(&inner.credential_slot, scope.into(), inner.auth.as_ref());
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Where this provider takes its per-request credential from, and where
    /// it puts it (`allama_http::CredentialSlot`).
    #[must_use]
    pub fn credential_slot(&self) -> &CredentialSlot {
        &self.inner.credential_slot
    }

    /// The dialect this provider speaks.
    #[must_use]
    pub fn quirks(&self) -> &ResponsesQuirks {
        &self.inner.quirks
    }

    /// The base URL requests are sent to, without a trailing slash. It may
    /// carry a user name and password: show it with
    /// [`allama_http::config::redact_url`].
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.inner.base_url
    }

    /// The full Responses endpoint URL.
    #[must_use]
    pub fn responses_url(&self) -> String {
        format!("{}{}", self.inner.base_url, self.inner.responses_path)
    }

    /// The request rules applied to `model`.
    #[must_use]
    pub fn model_rules(&self, model: &str) -> ResponsesModelRules {
        self.inner
            .model_rules
            .as_ref()
            .map_or_else(ResponsesModelRules::new, |rules| rules(model))
    }

    /// Encode a prepared chat request into the HTTP request this provider
    /// sends: everything before sending (the Responses wire format under
    /// the quirks and the model's rules, the wire option warnings, and the
    /// caller's transport options), so a dry run of it is a validation with
    /// [`ValidationCoverage::Encoding`]. It runs no preflight (see
    /// [`Provider::encode_chat`](crate::Provider::encode_chat)).
    ///
    /// # Errors
    ///
    /// Request mapping errors (including refusals under the request's
    /// policy) and transport option errors.
    #[doc(hidden)]
    pub fn encode_chat(
        &self,
        prepared: &mut Prepared<'_>,
        options: &ResponsesWireOptions,
        stream: bool,
    ) -> Result<HttpRequest> {
        let inner = &self.inner;
        let (request, degradations) = prepared.parts_mut();
        let rules = self.model_rules(&request.model);
        let encoded = encode::encode_request(
            request,
            stream,
            options,
            encode::Target {
                provider: &inner.provider_name,
                quirks: &inner.quirks,
                rules: &rules,
            },
        )?;
        // Into the same helper, so warnings name the caller's blocks.
        degradations.extend(encoded.warnings);
        let body = serde_json::to_vec(&encoded.body)?;
        let http = HttpRequest::post(self.responses_url(), body)
            .header("content-type", "application/json");
        self.authorize(http, &request.options)
    }

    /// Send a request [`encode_chat`](Self::encode_chat) built (not
    /// streaming), decode the response, and join `prepared`'s warnings to
    /// it.
    ///
    /// # Errors
    ///
    /// Transport, HTTP, and decoding errors.
    #[doc(hidden)]
    pub async fn send_chat(
        &self,
        request: HttpRequest,
        mut prepared: Prepared<'_>,
        options: &ResponsesWireOptions,
    ) -> Result<ChatResponse> {
        prepared.mark_sent();
        let (response, context) = self.send(request, false).await?;
        let response = self
            .read_response(response, context, options.raw_response)
            .await?;
        Ok(prepared.finish(response))
    }

    /// Send a streaming request [`encode_chat`](Self::encode_chat) built,
    /// and join `prepared`'s warnings to the stream.
    ///
    /// # Errors
    ///
    /// Transport and HTTP errors.
    #[doc(hidden)]
    pub async fn send_chat_stream(
        &self,
        request: HttpRequest,
        mut prepared: Prepared<'_>,
    ) -> Result<ChatStream> {
        prepared.mark_sent();
        let (response, context) = self.send(request, true).await?;
        let state = streaming::SseState::new(context);
        Ok(prepared.finish_stream(SseChatStream::new(self.sse(response), state)))
    }

    /// Fetch a stored response by id (`GET {responses_url}/{id}`), for
    /// example to poll a background response until it has a finish reason.
    ///
    /// # Errors
    ///
    /// Transport, HTTP (an unknown or unstored id is an HTTP error), and
    /// decoding errors.
    pub async fn retrieve_response(&self, id: &str) -> Result<ChatResponse> {
        self.response_by_id(Method::Get, id, "").await
    }

    /// Cancel a background response (`POST {responses_url}/{id}/cancel`)
    /// and return its state after cancellation.
    ///
    /// # Errors
    ///
    /// Transport, HTTP, and decoding errors.
    pub async fn cancel_response(&self, id: &str) -> Result<ChatResponse> {
        self.response_by_id(Method::Post, id, "/cancel").await
    }

    /// Stream a background response by id
    /// (`GET {responses_url}/{id}?stream=true`), for example to reconnect
    /// after a dropped connection.
    ///
    /// The endpoint replays the response's events from the start, so the
    /// stream decodes them all; events up to `after_sequence` (a
    /// [`StreamPosition`] from the interrupted stream's error) were already
    /// delivered and are not emitted again. With `None`, every event is
    /// emitted. The response must have been created with
    /// `background: true` and `stream: true`.
    ///
    /// ```no_run
    /// # use allama::prelude::*;
    /// # use allama_openai_compat::{ResponsesProvider, StreamPosition};
    /// # async fn example(provider: ResponsesProvider, error: allama::Error) -> allama::Result<()> {
    /// if let Some(position) = StreamPosition::from_error(&error) {
    ///     let rest = provider
    ///         .resume_stream(&position.response_id, Some(position.sequence_number))
    ///         .await?;
    ///     # let _ = rest;
    /// }
    /// # Ok(()) }
    /// ```
    ///
    /// # Errors
    ///
    /// Transport and HTTP errors (an unknown or non-background id is an
    /// HTTP error).
    pub async fn resume_stream(
        &self,
        response_id: &str,
        after_sequence: Option<u64>,
    ) -> Result<ChatStream> {
        let provider = self.inner.provider_name.as_ref();
        if response_id.is_empty() || response_id.contains(['/', '?', '#']) {
            return Err(
                Error::invalid_request(format!("invalid response id `{response_id}`"))
                    .authored()
                    .with_provider(provider),
            );
        }
        let url = format!("{}/{response_id}?stream=true", self.responses_url());
        let (response, context) = self
            .execute(HttpRequest::get(url), &RequestOptions::new(), true)
            .await?;
        let state = streaming::SseState::new(context).skip_through(after_sequence);
        // No `finish_stream` here: hold what arrives before the start as it
        // does.
        Ok(Box::pin(allama::adapter::chat::start_first(
            SseChatStream::new(self.sse(response), state),
        )))
    }

    async fn response_by_id(&self, method: Method, id: &str, suffix: &str) -> Result<ChatResponse> {
        let inner = &self.inner;
        let provider = inner.provider_name.as_ref();
        if id.is_empty() || id.contains(['/', '?', '#']) {
            return Err(
                Error::invalid_request(format!("invalid response id `{id}`"))
                    .authored()
                    .with_provider(provider),
            );
        }
        let url = format!("{}/{id}{suffix}", self.responses_url());
        let request = match method {
            Method::Get => HttpRequest::get(url),
            _ => HttpRequest::post(url, Bytes::new()),
        };
        let (response, context) = self.execute(request, &RequestOptions::new(), false).await?;
        self.read_response(response, context, false).await
    }

    /// Send `request` with the endpoint's auth headers, the caller's
    /// transport `options`, and timeouts (the stream bounds when `stream`,
    /// the whole-call timeout otherwise); map transport failures and
    /// non-success statuses to errors.
    async fn execute(
        &self,
        request: HttpRequest,
        options: &RequestOptions,
        stream: bool,
    ) -> Result<(HttpResponse, ResponseContext)> {
        let request = self.authorize(request, options)?;
        self.send(request, stream).await
    }

    /// Add the endpoint's auth headers and the caller's transport `options`.
    fn authorize(&self, mut request: HttpRequest, options: &RequestOptions) -> Result<HttpRequest> {
        let inner = &self.inner;
        if let Some((name, value)) = &inner.auth {
            request = request.header(name, value);
        }
        request
            .with_headers(&inner.headers)
            .apply_options(options, &inner.credential_slot)
            .map_err(|err| err.with_provider(inner.provider_name.as_ref()))
    }

    /// Send an authorized `request` with the timeouts (the stream bounds when
    /// `stream`, the whole-call timeout otherwise); map transport failures
    /// and non-success statuses to errors.
    async fn send(
        &self,
        request: HttpRequest,
        stream: bool,
    ) -> Result<(HttpResponse, ResponseContext)> {
        let inner = &self.inner;
        let provider = &inner.provider_name;
        let sent = if stream {
            inner.stream_timeouts.send(&*inner.http, request).await
        } else {
            inner.http.send(request.timeout(inner.timeout)).await
        };
        let response = sent.map_err(|e| e.into_error().with_provider(provider.as_ref()))?;
        let request_id = extract_request_id(&response.headers, &inner.request_id_header_name);
        if !response.is_success() {
            let error = response
                .map_error(|status, headers, body| {
                    // The configured headers take precedence over the common
                    // ones.
                    map_http_error(status, headers, body)
                        .with_request_id_opt(request_id.as_deref())
                        .with_retry_after_opt(extract_retry_after(
                            headers,
                            &inner.retry_after_header_name,
                        ))
                })
                .await;
            return Err(error.with_provider(provider.as_ref()));
        }
        let context = ResponseContext {
            provider: provider.clone(),
            namespace: inner.quirks.namespace.clone(),
            request_id,
            warnings: Vec::new(),
        };
        Ok((response, context))
    }

    /// Decode a successful non-streaming body.
    async fn read_response(
        &self,
        response: HttpResponse,
        context: ResponseContext,
        keep_raw: bool,
    ) -> Result<ChatResponse> {
        let provider = context.provider.clone();
        let namespace = context.namespace.clone();
        let request_id = context.request_id.clone();
        let (status, bytes) = read_body(
            response,
            self.inner.max_response_size,
            &provider,
            request_id.as_deref(),
        )
        .await?;
        let wire = decode_body(status, &bytes, &provider, request_id.as_deref())?;
        let mut decoded = decode::decode_response(wire, context)?;
        if keep_raw {
            let raw = decode_body(status, &bytes, &provider, request_id.as_deref())?;
            decoded.provider_data.insert_raw(
                namespace,
                allama::provider_data::RAW_RESPONSE_KEY,
                raw,
            );
        }
        Ok(decoded)
    }

    /// The verbatim response body kept when the request set `raw_response`
    /// (it lives under this provider's namespace).
    #[must_use]
    pub fn raw_response<'a>(&self, response: &'a ChatResponse) -> Option<&'a serde_json::Value> {
        response.provider_data.get_raw(
            &self.inner.quirks.namespace,
            allama::provider_data::RAW_RESPONSE_KEY,
        )
    }

    /// Frame a streamed response body, with this provider's size cap.
    fn sse(&self, response: HttpResponse) -> SseStream<allama_http::Body> {
        SseStream::new(response.into_body()).max_event_size(self.inner.max_stream_event_size)
    }

    /// Encode for this provider's own call or dry run: its stream
    /// capabilities are recorded for a stream.
    fn encode_own(
        &self,
        prepared: &mut Prepared<'_>,
        options: &ResponsesWireOptions,
        stream: bool,
    ) -> Result<HttpRequest> {
        if stream {
            prepared.preflight_stream(self);
        }
        let (request, degradations) = prepared.parts_mut();
        // Options under `openai-compat` are reported as Chat Completions
        // options already.
        crate::options::report_options_under_name(
            &request.options,
            self.provider_name(),
            <ResponsesRequestOptions as allama::ProviderOptions>::NAMESPACE,
            &[crate::NAMESPACE],
            "ResponsesRequestOptions",
            degradations,
        );
        self.encode_chat(prepared, options, stream)
    }

    /// A dry run of `chat` or `chat_stream`.
    fn validate(&self, request: &ChatRequest, stream: bool) -> Result<ChatValidation> {
        let options = Self::wire_options(request)?;
        let mut prepared = allama::adapter::chat::prepare_dry_run(self, request)?;
        self.encode_own(&mut prepared, &options, stream)?;
        Ok(prepared.into_validation(ValidationCoverage::Encoding))
    }

    /// Capability answers implied by the quirks and the model rules.
    fn implied_chat_capability(
        &self,
        model: &str,
        capability: ChatCapability,
    ) -> CapabilitySupport {
        let quirks = &self.inner.quirks;
        let reasoning = || self.model_rules(model).reasoning;
        match capability {
            ChatCapability::Streaming | ChatCapability::NativeStreaming
                if quirks.streaming == StreamingSupport::NonStreamingOnly =>
            {
                CapabilitySupport::Unsupported
            }
            ChatCapability::ForcedToolChoice if !quirks.supports_tool_choice => {
                CapabilitySupport::Unsupported
            }
            // The encoder refuses every audio part, and a `json_schema`
            // text format the endpoint does not take, whatever the model.
            ChatCapability::AudioInput => CapabilitySupport::Unsupported,
            ChatCapability::StructuredOutput if !quirks.supports_json_schema => {
                CapabilitySupport::Unsupported
            }
            ChatCapability::ReasoningConfig | ChatCapability::ReasoningOutput => reasoning(),
            ChatCapability::ReasoningReplay
                if quirks.reasoning_replay == crate::ReasoningReplay::Disabled
                    || reasoning() == CapabilitySupport::Unsupported =>
            {
                CapabilitySupport::Unsupported
            }
            _ => CapabilitySupport::Unknown,
        }
    }

    /// Resolve [`ResponsesRequestOptions`], reporting Chat Completions
    /// options this API does not read.
    fn wire_options(request: &ChatRequest) -> Result<ResponsesWireOptions> {
        let options: ResponsesRequestOptions = request.resolve_option()?;
        let mut wire = ResponsesWireOptions::from(options);
        let namespace = <crate::ChatRequestOptions as allama::ProviderOptions>::NAMESPACE;
        if request.options.contains::<crate::ChatRequestOptions>()
            || request.options.json(namespace).is_some()
        {
            wire.warnings.push(Warning::new(
                WarningKind::Dropped,
                Subject::Option {
                    namespace: namespace.into(),
                },
                "the Responses API does not read Chat Completions options (namespace `openai-compat`); they were dropped; use `ResponsesRequestOptions` (namespace `openai-compat-responses`)",
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
        let http = self.encode_own(&mut prepared, &options, false)?;
        self.send_chat(http, prepared, &options).await
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        let options = Self::wire_options(request)?;
        let mut prepared = allama::adapter::chat::prepare(self, request)?;
        let http = self.encode_own(&mut prepared, &options, true)?;
        self.send_chat_stream(http, prepared).await
    }

    /// A dry run of `chat` ([`ValidationCoverage::Encoding`]).
    fn validate_chat_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        self.validate(request, false)
    }

    /// A dry run of `chat_stream` ([`ValidationCoverage::Encoding`]).
    fn validate_chat_stream_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        self.validate(request, true)
    }

    /// Answers come from the resolver, then the declared table, then what
    /// the quirks and the model rules imply; anything else is `Unknown`.
    ///
    /// Implied answers:
    ///
    /// - streaming is `Unsupported` for non-streaming endpoints;
    /// - forced tool choice is `Unsupported` without `tool_choice` support;
    /// - reasoning configuration and output follow
    ///   [`ResponsesModelRules::reasoning`] when it is known;
    /// - reasoning replay is `Unsupported` when the dialect does not replay
    ///   reasoning or the model does not reason.
    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        let inner = &self.inner;
        inner
            .chat_capability_resolver
            .as_ref()
            .and_then(|resolver| resolver.chat_capability(model, capability))
            .or_else(|| inner.chat_capabilities.get(&capability).copied())
            .unwrap_or_else(|| self.implied_chat_capability(model, capability))
    }
}

impl ProviderIdentity for ResponsesProvider {
    fn provider_name(&self) -> &str {
        &self.inner.provider_name
    }

    fn reads_namespace(&self, namespace: &str) -> bool {
        namespace == self.inner.quirks.namespace
    }

    fn downcast_any(&self, id: std::any::TypeId) -> Option<&(dyn std::any::Any + Send + Sync)> {
        allama::expose(self, id)
    }
}

/// Where a Responses stream stopped: the response id and the
/// `sequence_number` of the last event received.
///
/// A stream that fails mid-way (transport error, truncation) records its
/// position in the error's [`ErrorContext::metadata`](allama::ErrorContext::metadata);
/// read it back with [`from_error`](Self::from_error) and pass it to
/// [`ResponsesProvider::resume_stream`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct StreamPosition {
    /// Id of the response being streamed.
    pub response_id: String,
    /// `sequence_number` of the last event received.
    pub sequence_number: u64,
}

impl StreamPosition {
    /// Error-metadata key holding the response id.
    pub const RESPONSE_ID_KEY: &'static str = "response_id";
    /// Error-metadata key holding the last sequence number.
    pub const SEQUENCE_NUMBER_KEY: &'static str = "sequence_number";

    /// A position.
    #[must_use]
    pub fn new(response_id: impl Into<String>, sequence_number: u64) -> Self {
        Self {
            response_id: response_id.into(),
            sequence_number,
        }
    }

    /// The position recorded on an error that interrupted a Responses
    /// stream, if any.
    #[must_use]
    pub fn from_error(error: &Error) -> Option<Self> {
        let metadata = &error.context()?.metadata;
        Some(Self::new(
            metadata.get(Self::RESPONSE_ID_KEY)?.as_str()?,
            metadata.get(Self::SEQUENCE_NUMBER_KEY)?.as_u64()?,
        ))
    }
}

/// Builder for [`ResponsesProvider`].
///
/// `Debug` shows the settings without the auth value, header values, or
/// credentials in the base URL.
#[derive(Default)]
pub struct ResponsesProviderBuilder {
    base_url: Option<String>,
    responses_path: Option<String>,
    auth: Option<(String, String)>,
    organization_header: Option<(String, String)>,
    project_header: Option<(String, String)>,
    request_id_header_name: Option<String>,
    retry_after_header_name: Option<String>,
    chat_capabilities: HashMap<ChatCapability, CapabilitySupport>,
    provider_name: Option<Cow<'static, str>>,
    credential_scope: Option<Cow<'static, str>>,
    credential_placement: Option<CredentialPlacement>,
    quirks: Option<ResponsesQuirks>,
    model_rules: Option<ModelRulesFn>,
    timeout: Option<Option<Duration>>,
    stream_timeouts: Option<StreamTimeouts>,
    max_stream_event_size: Option<usize>,
    max_response_size: Option<usize>,
    headers: Vec<(String, String)>,
    http: Option<Arc<dyn HttpTransport>>,
}

impl std::fmt::Debug for ResponsesProviderBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResponsesProviderBuilder")
            .field("provider_name", &self.provider_name)
            .field("base_url", &self.base_url.as_deref().map(redact_url))
            .field("auth", &self.auth.as_ref().map(|(name, _)| name))
            .field("credential_scope", &self.credential_scope)
            .field(
                "headers",
                &self
                    .organization_header
                    .iter()
                    .chain(&self.project_header)
                    .chain(&self.headers)
                    .map(|(name, _)| name)
                    .collect::<Vec<_>>(),
            )
            .field("http_transport", &self.http.is_some())
            .field("timeout", &self.timeout)
            .field("stream_timeouts", &self.stream_timeouts)
            .field("max_stream_event_size", &self.max_stream_event_size)
            .field("max_response_size", &self.max_response_size)
            .finish_non_exhaustive()
    }
}

impl ResponsesProviderBuilder {
    /// Base URL, e.g. `http://localhost:8000/v1`.
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Responses path (default `/responses`).
    #[must_use]
    pub fn responses_path(mut self, path: impl Into<String>) -> Self {
        self.responses_path = Some(path.into());
        self
    }

    /// Authenticate with `Authorization: Bearer <token>`. Surrounding
    /// whitespace is removed from the token.
    #[must_use]
    pub fn bearer_token(self, token: impl Into<String>) -> Self {
        self.auth_header("authorization", format!("Bearer {}", token.into().trim()))
    }

    /// Authenticate with a custom header (e.g. `api-key`). Surrounding
    /// whitespace is removed from the value.
    ///
    /// A per-request `allama_http::Credential::api_key` then goes in this
    /// header too, unless [`credential_placement`](Self::credential_placement)
    /// says otherwise. An `authorization` header is taken as a bearer token.
    #[must_use]
    pub fn auth_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.auth = Some((name.into(), value.into()));
        self
    }

    /// Send the header `name: value` with every request, replacing a
    /// header of the same name the provider sets (later calls replace
    /// earlier ones). `header("user-agent", ..)` replaces the default
    /// [`allama_http::DEFAULT_USER_AGENT`]. A request's own
    /// `allama_http::HttpRequestOptions` headers still replace these.
    ///
    /// Credential headers are refused at [`build`](Self::build): configure
    /// the key with [`bearer_token`](Self::bearer_token) or
    /// [`auth_header`](Self::auth_header), or send one per request as an
    /// `allama_http::Credential`.
    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// The credential scope (default: the
    /// [`provider_name`](Self::provider_name), else
    /// [`DEFAULT_CREDENTIAL_SCOPE`](crate::DEFAULT_CREDENTIAL_SCOPE)).
    /// Per-request credentials addressed to it
    /// (`allama_http::HttpRequestOptions::credential`) are this provider's;
    /// credentials addressed to other scopes are never sent.
    ///
    /// Two accounts of one endpoint in a fallback chain need two scopes: by
    /// default they share one. Providers behind one gateway share one.
    ///
    /// Without configured auth, setting a scope means bring your own key:
    /// every request must carry a credential for the scope, or it fails
    /// with an authored [`Error::Configuration`] before anything is sent.
    /// That includes [`retrieve_response`](ResponsesProvider::retrieve_response),
    /// [`cancel_response`](ResponsesProvider::cancel_response) and
    /// [`resume_stream`](ResponsesProvider::resume_stream), which take no
    /// request options. With configured auth, a request without a credential
    /// for the scope is sent with the configured one, so a mistyped scope
    /// falls back to it.
    #[must_use]
    pub fn credential_scope(mut self, scope: impl Into<Cow<'static, str>>) -> Self {
        self.credential_scope = Some(scope.into());
        self
    }

    /// Where a per-request `allama_http::Credential::api_key` goes. By
    /// default, where the configured auth goes: a bearer token for
    /// [`bearer_token`](Self::bearer_token) (or no auth), the header for
    /// [`auth_header`](Self::auth_header). When it differs from the
    /// configured auth header, a per-request key replaces that header too.
    #[must_use]
    pub fn credential_placement(mut self, placement: CredentialPlacement) -> Self {
        self.credential_placement = Some(placement);
        self
    }

    /// Send an organization header.
    #[must_use]
    pub fn organization_header(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.organization_header = Some((name.into(), value.into()));
        self
    }

    /// Send a project header.
    #[must_use]
    pub fn project_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.project_header = Some((name.into(), value.into()));
        self
    }

    /// Header carrying the request id (default `x-request-id`).
    #[must_use]
    pub fn request_id_header_name(mut self, name: impl Into<String>) -> Self {
        self.request_id_header_name = Some(name.into());
        self
    }

    /// Header carrying the retry-after hint (default `retry-after`;
    /// `retry-after-ms` is always consulted too).
    #[must_use]
    pub fn retry_after_header_name(mut self, name: impl Into<String>) -> Self {
        self.retry_after_header_name = Some(name.into());
        self
    }

    /// Declare a chat capability.
    #[must_use]
    pub fn chat_capability(
        mut self,
        capability: ChatCapability,
        support: CapabilitySupport,
    ) -> Self {
        self.chat_capabilities.insert(capability, support);
        self
    }

    /// Provider identity used for telemetry, errors, warnings, and
    /// `ChatResponse::provider` (default `openai-compat`).
    #[must_use]
    pub fn provider_name(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        self.provider_name = Some(name.into());
        self
    }

    /// Dialect quirks. Defaults to [`ResponsesQuirks::new`] under namespace
    /// `openai-compat:<provider_name>` (`openai-compat` without a provider
    /// name).
    #[must_use]
    pub fn quirks(mut self, quirks: ResponsesQuirks) -> Self {
        self.quirks = Some(quirks);
        self
    }

    /// Request rules per model id. Without them every model gets
    /// [`ResponsesModelRules::new`], which sends reasoning and sampling
    /// settings as requested.
    #[must_use]
    pub fn model_rules<F>(mut self, rules: F) -> Self
    where
        F: Fn(&str) -> ResponsesModelRules + Send + Sync + 'static,
    {
        self.model_rules = Some(Arc::new(rules));
        self
    }

    /// The timeout for a whole non-streaming call; `None` sets none.
    ///
    /// Left unset, it is [`DEFAULT_REQUEST_TIMEOUT`](allama_http::DEFAULT_REQUEST_TIMEOUT)
    /// (120 s), whatever the transport: a client of your own keeps the
    /// bound unless you pass `None` here. Set, it applies in place of a
    /// client's own total timeout.
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
        self.stream_timeouts = Some(timeouts);
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
    #[must_use]
    pub fn max_stream_event_size(mut self, bytes: usize) -> Self {
        self.max_stream_event_size = Some(bytes);
        self
    }

    /// The largest non-streaming response body, in bytes, the provider
    /// reads (default [`allama_http::DEFAULT_MAX_RESPONSE_SIZE`], 256 MiB).
    /// A larger body fails the call with [`Error::UnexpectedResponse`],
    /// which is not retryable, without being read whole.
    #[must_use]
    pub fn max_response_size(mut self, bytes: usize) -> Self {
        self.max_response_size = Some(bytes);
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
    /// transport with each non-streaming request, whatever the transport.
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
    /// base URL is missing or malformed, the auth value or a header is
    /// malformed, a header carries a credential (see
    /// [`ProviderBuilder::build`](crate::ProviderBuilder::build)), or when
    /// no transport is set and the `reqwest` feature is off.
    pub fn build(self) -> Result<ResponsesProvider> {
        let base_url = match self.base_url {
            Some(url) => allama_http::config::base_url("base_url", &url)?,
            None => return Err(Error::configuration("base_url is required").authored()),
        };
        let auth = crate::checked_auth(self.auth)?;
        let provider_name = self
            .provider_name
            .unwrap_or(Cow::Borrowed(crate::PROVIDER_NAME));
        let quirks = self
            .quirks
            .unwrap_or_else(|| ResponsesQuirks::new(crate::default_namespace(&provider_name)));
        let http = match self.http {
            Some(http) => http,
            None => crate::default_transport()?,
        };
        let credential_slot = crate::credential_slot(
            self.credential_scope,
            provider_name.clone(),
            self.credential_placement,
            auth.as_ref(),
        );
        let headers = allama_http::config::static_headers(
            self.organization_header
                .into_iter()
                .chain(self.project_header)
                .chain(self.headers),
            &credential_slot,
        )?;
        Ok(ResponsesProvider {
            inner: Arc::new(Inner {
                http,
                base_url,
                responses_path: self.responses_path.unwrap_or_else(|| "/responses".into()),
                auth,
                headers,
                request_id_header_name: self
                    .request_id_header_name
                    .unwrap_or_else(|| "x-request-id".into()),
                retry_after_header_name: self
                    .retry_after_header_name
                    .unwrap_or_else(|| "retry-after".into()),
                timeout: crate::resolve_timeout(self.timeout),
                stream_timeouts: self.stream_timeouts.unwrap_or_default(),
                max_stream_event_size: self
                    .max_stream_event_size
                    .unwrap_or(SseDecoder::DEFAULT_MAX_EVENT_SIZE),
                max_response_size: self
                    .max_response_size
                    .unwrap_or(allama_http::DEFAULT_MAX_RESPONSE_SIZE),
                quirks,
                model_rules: self.model_rules,
                chat_capabilities: self.chat_capabilities,
                chat_capability_resolver: None,
                provider_name,
                credential_slot,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use allama::ReasoningEffort;
    use serde_json::json;

    use super::*;
    use allama_http::DEFAULT_REQUEST_TIMEOUT;

    /// One timeout model for every builder: unset, the default applies
    /// whatever the transport, a client of your own included; only an
    /// explicit `None` removes it.
    #[cfg(feature = "reqwest")]
    #[test]
    fn the_timeout_is_kept_whatever_the_transport() {
        let five = Some(Duration::from_secs(5));
        let client = || allama_http::ReqwestTransport::from_client(reqwest::Client::new());
        let timeout = |builder| {
            ResponsesProviderBuilder::build(builder)
                .unwrap()
                .inner
                .timeout
        };
        let builder = || ResponsesProvider::builder().base_url("http://localhost:1/v1");
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

    /// `Debug` never shows the key or the base URL's
    /// credentials, and the builder's headers reach every request.
    #[tokio::test]
    async fn debug_is_redacted_and_headers_are_sent() {
        let transport =
            allama_http::ReplayTransport::new([allama_http::ReplayResponse::new(500, "")]);
        let builder = ResponsesProvider::builder()
            .base_url("https://user:PASSWORD@gw.example/v1")
            .bearer_token("sk-SECRET")
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
        let _ = provider
            .chat(&allama::ChatRequest::new("m").user("hi"))
            .await;
        let sent = &transport.requests()[0];
        assert_eq!(sent.headers.get("x-route"), Some("ROUTE-VALUE"));
        assert_eq!(
            sent.headers.get("user-agent"),
            Some(allama_http::DEFAULT_USER_AGENT)
        );
    }

    #[test]
    fn builder_defaults() {
        let provider = ResponsesProvider::builder()
            .base_url("https://example.com/v1///")
            .provider_name("demo")
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(provider.base_url(), "https://example.com/v1");
        assert_eq!(provider.responses_url(), "https://example.com/v1/responses");
        assert_eq!(provider.quirks().namespace, "openai-compat:demo");
        assert_eq!(provider.provider_name(), "demo");
        assert!(provider.reads_namespace("openai-compat:demo"));
        assert!(!provider.reads_namespace("demo"));
        assert_eq!(provider.inner.timeout, Some(DEFAULT_REQUEST_TIMEOUT));
        assert!(provider.inner.auth.is_none());
        assert_eq!(provider.model_rules("any"), ResponsesModelRules::new());

        let unnamed = ResponsesProvider::builder()
            .base_url("http://localhost:1")
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(unnamed.provider_name(), "openai-compat");
        assert_eq!(unnamed.quirks().namespace, "openai-compat");

        assert!(ResponsesProvider::builder().build().is_err());
        assert!(ResponsesProvider::builder().base_url("  ").build().is_err());
    }

    #[test]
    fn model_rules_are_resolved_per_model() {
        let provider = ResponsesProvider::builder()
            .base_url("http://localhost:1")
            .model_rules(|model: &str| {
                if model.starts_with("gpt-oss") {
                    ResponsesModelRules::new().effort_levels(&[ReasoningEffort::Low])
                } else {
                    ResponsesModelRules::new()
                }
            })
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert!(provider.model_rules("gpt-oss-20b").effort_levels.is_some());
        assert!(provider.model_rules("qwen3").effort_levels.is_none());
    }

    #[test]
    fn capabilities_come_from_resolver_table_then_quirks() {
        let mut quirks = ResponsesQuirks::new("openai-compat:demo");
        quirks.streaming = StreamingSupport::NonStreamingOnly;
        let provider = ResponsesProvider::builder()
            .base_url("http://localhost:1")
            .quirks(quirks)
            .chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported)
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(
            provider.chat_capability("m", ChatCapability::ToolCalls),
            CapabilitySupport::Supported
        );
        assert_eq!(
            provider.chat_capability("m", ChatCapability::Streaming),
            CapabilitySupport::Unsupported
        );
        assert_eq!(
            provider.chat_capability("m", ChatCapability::ReasoningReplay),
            CapabilitySupport::Unknown
        );
        let overridden = provider.with_chat_capabilities(|_: &str, c| {
            (c == ChatCapability::ToolCalls).then_some(CapabilitySupport::Unsupported)
        });
        assert_eq!(
            overridden.chat_capability("m", ChatCapability::ToolCalls),
            CapabilitySupport::Unsupported
        );
        assert_eq!(overridden.quirks().namespace, "openai-compat:demo");
    }

    #[test]
    fn capabilities_follow_model_rules_and_dialect() {
        use allama::CapabilitySupport::{Supported, Unknown, Unsupported};

        let mut quirks = ResponsesQuirks::new("openai-compat:demo");
        quirks.supports_tool_choice = false;
        quirks.supports_json_schema = false;
        let provider = ResponsesProvider::builder()
            .base_url("http://localhost:1")
            .quirks(quirks)
            .model_rules(|model: &str| match model {
                "thinker" => ResponsesModelRules::new().reasoning(Supported),
                "plain" => ResponsesModelRules::new().reasoning(Unsupported),
                _ => ResponsesModelRules::new(),
            })
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        let answer = |model, capability| provider.chat_capability(model, capability);
        assert_eq!(
            answer("thinker", ChatCapability::ReasoningConfig),
            Supported
        );
        assert_eq!(
            answer("thinker", ChatCapability::ReasoningOutput),
            Supported
        );
        assert_eq!(answer("thinker", ChatCapability::ReasoningReplay), Unknown);
        assert_eq!(
            answer("plain", ChatCapability::ReasoningConfig),
            Unsupported
        );
        assert_eq!(
            answer("plain", ChatCapability::ReasoningReplay),
            Unsupported
        );
        assert_eq!(answer("other", ChatCapability::ReasoningConfig), Unknown);
        assert_eq!(
            answer("other", ChatCapability::ForcedToolChoice),
            Unsupported
        );
        assert_eq!(answer("other", ChatCapability::ToolCalls), Unknown);
        // The encoder refuses every audio
        // part, and a JSON Schema the endpoint does not take, for every
        // model.
        assert_eq!(answer("other", ChatCapability::AudioInput), Unsupported);
        assert_eq!(
            answer("other", ChatCapability::StructuredOutput),
            Unsupported
        );

        let mut no_replay = ResponsesQuirks::new("openai-compat:demo");
        no_replay.reasoning_replay = crate::ReasoningReplay::Disabled;
        let provider = ResponsesProvider::builder()
            .base_url("http://localhost:1")
            .quirks(no_replay)
            .chat_capability(ChatCapability::ReasoningConfig, Unsupported)
            .model_rules(|_: &str| ResponsesModelRules::new().reasoning(Supported))
            .http_transport(allama_http::ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(
            provider.chat_capability("m", ChatCapability::ReasoningReplay),
            Unsupported
        );
        assert_eq!(
            provider.chat_capability("m", ChatCapability::ReasoningConfig),
            Unsupported,
            "declared answers win over the rules"
        );
    }

    #[test]
    fn chat_completions_options_are_reported() {
        let typed = ChatRequest::new("m")
            .user("hi")
            .with_option(crate::ChatRequestOptions {
                service_tier: Some("flex".into()),
                ..Default::default()
            });
        let json = ChatRequest::new("m")
            .user("hi")
            .with_options_json("openai-compat", json!({"service_tier": "flex"}));
        for request in [typed, json] {
            let options = ResponsesProvider::wire_options(&request).unwrap();
            assert_eq!(options.service_tier, None);
            assert_eq!(options.warnings.len(), 1);
            assert_eq!(options.warnings[0].kind, WarningKind::Dropped);
        }
    }

    #[test]
    fn responses_options_resolve_and_typos_fail() {
        let request = ChatRequest::new("m").with_options_json(
            "openai-compat-responses",
            json!({"truncation": "auto", "hosted_tools": [{"type": "web_search"}]}),
        );
        let options = ResponsesProvider::wire_options(&request).unwrap();
        assert_eq!(options.truncation.as_deref(), Some("auto"));
        assert_eq!(options.hosted_tools.len(), 1);
        assert!(options.warnings.is_empty());

        let typo = ChatRequest::new("m")
            .with_options_json("openai-compat-responses", json!({"truncaton": "auto"}));
        assert!(matches!(
            ResponsesProvider::wire_options(&typo),
            Err(Error::InvalidRequest { .. })
        ));
    }
}
