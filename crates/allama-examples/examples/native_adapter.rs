//! # A native HTTP adapter
//!
//! The adapter the [provider author guide] builds and quotes: a
//! `ChatProvider` for "Acme", a fictional chat API with its own wire format,
//! sending through `allama-http`. Every code block of the guide is a quote
//! of this file, and a test checks the quotes, so the guide's code compiles
//! and its contracts pass.
//!
//! - **Identity and capabilities**: `PROVIDER_NAME`, `NAMESPACE`,
//!   `downcast_any`, and honest capability answers.
//! - **Request mapping**: `allama::adapter::chat::prepare` first (the preflight),
//!   then the fields no capability covers through its `Degradations`; typed
//!   options; replay of the adapter's own reasoning state only; a reject arm
//!   for the `#[non_exhaustive]` enums. Everything before sending is one
//!   `encode` method, so validation is a dry run of it.
//! - **Errors**: a pure `map_http_error` built on `error_context` and
//!   `error_envelope`, with authored messages and upstream text in its own
//!   slot.
//! - **Streaming**: `SseStream` framing, `VendorFrames`, a synthesized
//!   `ResponseStart` (the Acme stream has no start frame), an in-band error
//!   that may arrive before it, and `MetadataBuilder`; the same decoder
//!   over NDJSON lines (`NdjsonStream`, `FramedChatStream`), for an API that
//!   streams in another framing.
//! - **Tests**: the probes and the conformance contracts that apply.
//!
//! Banners mark each section `// ── Required: … ──` (every native adapter
//! needs it) or `// ── Only if your API has …: … ──`, matching the guide's
//! "Minimal path" table. Read the required sections first.
//!
//! `main` replays canned wire data, so it needs no credentials:
//!
//! ```text
//! cargo run -p allama-examples --example native_adapter
//! cargo test -p allama-examples --example native_adapter
//! ```
//!
//! [provider author guide]: https://github.com/sagikazarmark/allama/blob/main/docs/guides/provider-authors.md

use std::sync::Arc;
use std::time::Duration;

use allama::adapter::chat::{EncodedTurn, Prepared, retain_turns};
use allama::adapter::{Degradations, describe_origins};
use allama::chat::stream::MetadataBuilder;
use allama::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, ChatStream,
    ChatValidation, ContentBlock, ContentPart, Error, ExtraMap, FinishReason, Message,
    OtherContent, ProviderData, ProviderDataEntry, ProviderIdentity, ProviderOptions, Reasoning,
    ReplayState, RequestField, ResponseFormat, Result, StreamBlockType, StreamErrorKind,
    StreamEvent, Subject, ToolChoice, ToolResultContent, Usage, UserContent, ValidationCoverage,
};
use allama_http::framing::{ChatFrameDecoder, FramedChatStream};
use allama_http::http::header::{AUTHORIZATION, CONTENT_TYPE};
use allama_http::http::{HeaderMap, HeaderValue, StatusCode, Uri};
use allama_http::ndjson::NdjsonStream;
use allama_http::sse::{SseChatStream, SseDecoder, SseEvent, SseStream, VendorFrames};
use allama_http::{
    Body, CredentialPlacement, CredentialSlot, DEFAULT_MAX_RESPONSE_SIZE, DEFAULT_REQUEST_TIMEOUT,
    HttpRequest, HttpResponse, HttpTransport, StreamTimeouts, TransportError, config,
    error_context, error_envelope, malformed_response, request_uri, status_error,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

// ── Required: identity and constants (guide §2) ──

/// Who serves calls: stamped on responses, errors, and warnings.
pub const PROVIDER_NAME: &str = "acme";

/// The key of Acme's provider options and provider data.
pub const NAMESPACE: &str = "acme";

/// The environment variable holding the API key.
pub const API_KEY_ENV: &str = "ACME_API_KEY";

/// The default credential scope: per-request keys addressed to it are
/// Acme's.
pub const DEFAULT_CREDENTIAL_SCOPE: &str = "acme";

const DEFAULT_BASE_URL: &str = "https://api.acme.example/v1";
const MAX_STOP_SEQUENCES: usize = 4;

/// Acme error codes that mean the account is out of credit.
const QUOTA_ERROR_CODES: &[&str] = &["credit_exhausted"];

/// The event types of the Acme stream.
const KNOWN_EVENTS: &[&str] = &["delta", "done", "error"];

// ── Required: the provider and its capability answers (guide §2, §3 "Preflight first") ──

/// A chat provider for the Acme API.
///
/// `Debug` never shows the API key.
#[derive(Clone)]
pub struct AcmeProvider {
    http: Arc<dyn HttpTransport>,
    /// The chat endpoint, built from the base URL once.
    chat_uri: Uri,
    /// `Bearer <key>`, marked sensitive so `Debug` output redacts it.
    authorization: HeaderValue,
    credential_slot: CredentialSlot,
    request_timeout: Option<Duration>,
    stream_timeouts: StreamTimeouts,
    max_stream_event_size: usize,
}

impl std::fmt::Debug for AcmeProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AcmeProvider")
            .field("chat_uri", &config::redact_url(&self.chat_uri.to_string()))
            .field("credential_scope", &self.credential_slot.scope)
            .field("api_key", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl AcmeProvider {
    /// A builder with the default base URL, timeouts and transport.
    #[must_use]
    pub fn builder() -> AcmeProviderBuilder {
        AcmeProviderBuilder::default()
    }

    /// Where a per-request key for this provider comes from and goes.
    #[must_use]
    pub fn credential_slot(&self) -> &CredentialSlot {
        &self.credential_slot
    }

    /// A provider configured from `ACME_API_KEY`.
    ///
    /// # Errors
    ///
    /// [`Error::Configuration`] when the variable is unset.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var(API_KEY_ENV).unwrap_or_default();
        Self::builder().api_key(api_key).build()
    }

    /// Everything before sending, shared by the calls and their dry runs
    /// (`validate_chat_request`): encode the prepared request, reporting
    /// through it, and build the HTTP request.
    fn encode(&self, prepared: &mut Prepared<'_>, stream: bool) -> Result<HttpRequest> {
        if stream {
            // What a stream relies on, so its validation reports `Unknown`
            // answers.
            prepared.preflight_stream(self);
        }
        let (request, degradations) = prepared.parts_mut();
        let body = encode_body(request, stream, degradations)?;
        HttpRequest::post(self.chat_uri.clone(), body)
            .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
            .header(AUTHORIZATION, self.authorization.clone())
            // Last: the caller's headers, and the credentials addressed to
            // your scope, which replace your configured key.
            .apply_options(&request.options, &self.credential_slot)
    }

    /// Send an encoded request; an error status is mapped to its error.
    async fn send(&self, request: HttpRequest, streaming: bool) -> Result<HttpResponse> {
        let response = if streaming {
            // No total deadline, but a setup and an idle bound.
            self.stream_timeouts.send(&*self.http, request).await
        } else {
            self.http.send(request.timeout(self.request_timeout)).await
        }
        .map_err(TransportError::into_error)?;
        if response.is_success() {
            return Ok(response);
        }
        // Reads at most 64 KiB of the body, records a failed read, and
        // maps a redirect (never followed) itself.
        Err(response.map_error(map_http_error).await)
    }

    /// A dry run of `chat` or `chat_stream`: the same preflight and
    /// encoding, without sending.
    fn validate(&self, request: &ChatRequest, stream: bool) -> Result<ChatValidation> {
        let mut prepared = allama::adapter::chat::prepare_dry_run(self, request)?;
        self.encode(&mut prepared, stream)?;
        Ok(prepared.into_validation(ValidationCoverage::Encoding))
    }
}

impl ProviderIdentity for AcmeProvider {
    fn provider_name(&self) -> &str {
        PROVIDER_NAME
    }

    /// Acme state lives under [`NAMESPACE`], and the adapter reads no other.
    /// `retain_provider_state_for` keeps the state this answers `true` for.
    fn reads_namespace(&self, namespace: &str) -> bool {
        namespace == NAMESPACE
    }

    fn downcast_any(&self, id: std::any::TypeId) -> Option<&(dyn std::any::Any + Send + Sync)> {
        allama::expose(self, id)
    }
}

impl ChatProvider for AcmeProvider {
    type Stream = ChatStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let mut prepared = allama::adapter::chat::prepare(self, request)?;
        let http = self.encode(&mut prepared, false)?;
        // Immediately before sending: from here on, an error (a 429) still
        // reports the warnings as warning events.
        prepared.mark_sent();
        let response = self.send(http, false).await?;
        let status = response.status;
        let request_id = error_context(status, &response.headers).request_id;
        // Bounded, like the builders' `max_response_size`: a body past it
        // fails as an unexpected response instead of filling memory.
        let body = response
            .text_limited(DEFAULT_MAX_RESPONSE_SIZE)
            .await
            .map_err(TransportError::into_error)?;
        let response = decode_response(status, &body, request_id)?;
        // The preflight and encoding warnings join the response.
        Ok(prepared.finish(response))
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
        let mut prepared = allama::adapter::chat::prepare(self, request)?;
        let http = self.encode(&mut prepared, true)?;
        prepared.mark_sent();
        let response = self.send(http, true).await?;
        let request_id = error_context(response.status, &response.headers).request_id;
        let events =
            SseStream::new(response.into_body()).max_event_size(self.max_stream_event_size);
        let stream = SseChatStream::new(events, StreamDecoder::new(request_id));
        // The warnings and vendor frames follow the stream's `ResponseStart`.
        Ok(prepared.finish_stream(stream))
    }

    fn chat_capability(&self, _model: &str, capability: ChatCapability) -> CapabilitySupport {
        match capability {
            // `Streaming` says `chat_stream` delivers incrementally; a
            // provider that replays a finished response as a stream answers
            // it too. `NativeStreaming` says the upstream API streams, as
            // Acme's does, so the first token arrives before the answer is
            // complete.
            ChatCapability::Streaming
            | ChatCapability::NativeStreaming
            | ChatCapability::StreamUsage
            | ChatCapability::ToolCalls
            | ChatCapability::ForcedToolChoice
            | ChatCapability::ReasoningOutput
            | ChatCapability::ReasoningReplay => CapabilitySupport::Supported,
            ChatCapability::ImageInput
            | ChatCapability::ImageDetail
            | ChatCapability::ImageOutput
            | ChatCapability::ImageReplay
            | ChatCapability::FileInput
            | ChatCapability::AudioInput
            | ChatCapability::VideoInput
            | ChatCapability::StructuredOutput
            | ChatCapability::ReasoningConfig
            | ChatCapability::PromptCache => CapabilitySupport::Unsupported,
            // Several tool calls in one turn are decoded, but whether the
            // model makes them is its own business: Acme has no control.
            ChatCapability::ParallelToolCalls => CapabilitySupport::Unknown,
            // A capability added after this adapter was written.
            _ => CapabilitySupport::Unknown,
        }
    }

    // The defaults check capability answers only; a dry run of the
    // encoding gives the answer the call would give.
    fn validate_chat_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        self.validate(request, false)
    }

    fn validate_chat_stream_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        self.validate(request, true)
    }
}

// ── Required: the builder, with the credential slot and timeouts (guide §5) ──

/// Builds an [`AcmeProvider`].
///
/// `Debug` never shows the API key.
pub struct AcmeProviderBuilder {
    api_key: Option<String>,
    credential_scope: String,
    base_url: String,
    http: Option<Arc<dyn HttpTransport>>,
    request_timeout: Option<Duration>,
    stream_timeouts: StreamTimeouts,
    max_stream_event_size: usize,
}

impl std::fmt::Debug for AcmeProviderBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AcmeProviderBuilder")
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("credential_scope", &self.credential_scope)
            .field("base_url", &self.base_url)
            .field("http_transport", &self.http.is_some())
            .field("request_timeout", &self.request_timeout)
            .field("stream_timeouts", &self.stream_timeouts)
            .field("max_stream_event_size", &self.max_stream_event_size)
            .finish()
    }
}

impl Default for AcmeProviderBuilder {
    fn default() -> Self {
        Self {
            api_key: None,
            credential_scope: DEFAULT_CREDENTIAL_SCOPE.to_owned(),
            base_url: DEFAULT_BASE_URL.to_owned(),
            http: None,
            request_timeout: Some(DEFAULT_REQUEST_TIMEOUT),
            stream_timeouts: StreamTimeouts::default(),
            max_stream_event_size: SseDecoder::DEFAULT_MAX_EVENT_SIZE,
        }
    }
}

impl AcmeProviderBuilder {
    /// Set the API key.
    #[must_use]
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the credential scope (default [`DEFAULT_CREDENTIAL_SCOPE`]): two
    /// Acme accounts in one fallback chain need two scopes.
    #[must_use]
    pub fn credential_scope(mut self, scope: impl Into<String>) -> Self {
        self.credential_scope = scope.into();
        self
    }

    /// Set the base URL.
    #[must_use]
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Send through `transport` instead of the default `reqwest` client.
    #[must_use]
    pub fn http_transport(mut self, transport: impl HttpTransport) -> Self {
        self.http = Some(Arc::new(transport));
        self
    }

    /// Bound a whole non-streaming call (`None` for no bound).
    #[must_use]
    pub fn timeout(mut self, timeout: Option<Duration>) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Bound the stream's headers and the gaps between its chunks.
    #[must_use]
    pub fn stream_timeouts(mut self, timeouts: StreamTimeouts) -> Self {
        self.stream_timeouts = timeouts;
        self
    }

    /// Cap the size of a stream event the provider holds in memory.
    #[must_use]
    pub fn max_stream_event_size(mut self, bytes: usize) -> Self {
        self.max_stream_event_size = bytes;
        self
    }

    /// Build the provider.
    ///
    /// # Errors
    ///
    /// [`Error::Configuration`] without an API key: nothing reached Acme,
    /// so this is no authentication failure.
    pub fn build(self) -> Result<AcmeProvider> {
        let api_key = self
            .api_key
            .filter(|key| !key.trim().is_empty())
            .ok_or_else(|| Error::configuration(format!("{API_KEY_ENV} is not set")).authored())?;
        // Checked once, here, so no request fails on the configuration: the
        // base URL and the key become the `http` types every request uses.
        let base_url = config::base_url("base_url", &self.base_url)?;
        let chat_uri = request_uri(format!("{base_url}/chat"))?;
        let api_key = config::secret("api_key", &api_key)?;
        let authorization = config::sensitive_value("api_key", &format!("Bearer {api_key}"))?;
        // A provider crate puts this default behind an optional, default
        // `reqwest` feature, and without it requires a transport here.
        let http = self
            .http
            .unwrap_or_else(|| Arc::new(allama_http::ReqwestTransport::new()));
        // Acme takes its key as a bearer token.
        let credential_slot =
            CredentialSlot::new(self.credential_scope, CredentialPlacement::Bearer);
        Ok(AcmeProvider {
            http,
            chat_uri,
            authorization,
            credential_slot,
            request_timeout: self.request_timeout,
            stream_timeouts: self.stream_timeouts,
            max_stream_event_size: self.max_stream_event_size,
        })
    }
}

// ── Only if your API has request fields of its own: typed options (guide §3) ──

/// Acme request fields with no portable meaning.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct ChatRequestOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_mode: Option<String>,
    /// Body fields this crate does not model yet, merged into the request.
    #[serde(skip_serializing_if = "ExtraMap::is_empty")]
    pub extra: ExtraMap,
}

impl ProviderOptions for ChatRequestOptions {
    const NAMESPACE: &'static str = NAMESPACE;
}

impl ChatRequestOptions {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn safety_mode(mut self, mode: impl Into<String>) -> Self {
        self.safety_mode = Some(mode.into());
        self
    }

    #[must_use]
    pub fn extra(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }
}

/// The body fields the adapter owns: `extra` cannot set them.
const OWNED_FIELDS: &[&str] = &[
    "max_tokens",
    "messages",
    "model",
    "safety_mode",
    "stop",
    "stream",
    "system",
    "temperature",
    "tool_choice",
    "tools",
    "top_p",
];

// ── Required: request encoding, every field sent or reported (guide §3) ──

/// The Acme request body.
#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    system: Vec<&'a str>,
    messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "<[String]>::is_empty")]
    stop: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    safety_mode: Option<String>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
    #[serde(flatten)]
    extra: ExtraMap,
}

#[derive(Serialize)]
struct WireMessage {
    role: &'static str,
    content: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    is_error: bool,
}

impl WireMessage {
    fn new(role: &'static str, content: Vec<Value>) -> Self {
        Self {
            role,
            content,
            tool_call_id: None,
            is_error: false,
        }
    }
}

/// Encode a request that passed preflight.
fn encode_body(
    request: &ChatRequest,
    stream: bool,
    degradations: &mut Degradations,
) -> Result<Vec<u8>> {
    if request.seed.is_some() {
        // A hint: dropped with a warning, or an error under UnsupportedPolicy::Error.
        degradations
            .unsupported_hint(RequestField::Seed, "Acme has no seed; `seed` was not sent")?;
    }
    if request.stop.len() > MAX_STOP_SEQUENCES {
        // Semantic: changing the meaning of the request is always an error.
        return Err(degradations.unsupported(RequestField::Stop, "at most 4 stop sequences"));
    }
    if request.top_k.is_some() {
        degradations.unsupported_hint(
            RequestField::TopK,
            "Acme has no top-k sampling; `top_k` was not sent",
        )?;
    }
    if request.frequency_penalty.is_some() {
        degradations.unsupported_hint(
            RequestField::FrequencyPenalty,
            "Acme has no frequency penalty; `frequency_penalty` was not sent",
        )?;
    }
    if request.presence_penalty.is_some() {
        degradations.unsupported_hint(
            RequestField::PresencePenalty,
            "Acme has no presence penalty; `presence_penalty` was not sent",
        )?;
    }
    // Every field is encoded or reported, never skipped: preflight refuses
    // or removes what `chat_capability` answers `Unsupported`, but an
    // `Unknown` answer (a capability resolver, a model this adapter does
    // not list) leaves it to the encoder.
    match &request.response_format {
        None | Some(ResponseFormat::Text) => {}
        Some(_) => {
            return Err(degradations.unsupported(
                RequestField::ResponseFormat,
                "Acme has no structured output",
            ));
        }
    }
    if request.reasoning.is_some() {
        degradations.unsupported_hint(
            RequestField::Reasoning,
            "Acme has no reasoning controls; `reasoning` was not sent",
        )?;
    }
    if request.cache.is_some() {
        degradations.unsupported_hint(
            RequestField::Cache,
            "Acme has no prompt cache; `cache` was not sent",
        )?;
    }
    // `true` is permission the model has anyway, so it is dropped silently;
    // only `false` asks for something Acme cannot be told.
    if request.parallel_tool_calls == Some(false) {
        degradations.unsupported_hint(
            RequestField::ParallelToolCalls,
            "Acme has no `parallel_tool_calls` field; `false` was not sent, so the model may still call several tools at once",
        )?;
    }

    // In `chat`: accepts the typed value or the same data as namespaced JSON
    // (`ChatRequest::with_options_json`), so options survive serialization.
    let options: ChatRequestOptions = request.resolve_option()?;
    // A field the adapter owns would go out twice: refuse it.
    if let Some(key) = options
        .extra
        .keys()
        .find(|k| OWNED_FIELDS.contains(&k.as_str()))
    {
        let message = format!("`{key}` cannot be set through ChatRequestOptions::extra");
        return Err(Error::invalid_request(message)
            .authored()
            .with_provider(PROVIDER_NAME));
    }

    let mut tools = Vec::new();
    for tool in &request.tools {
        if tool.strict == Some(true) {
            let message = format!(
                "Acme has no strict schemas; `{}` was sent without `strict`",
                tool.name
            );
            degradations.unsupported_hint(RequestField::Tools, message)?;
        }
        let mut wire = json!({"name": tool.name, "parameters": tool.parameters});
        if let Some(description) = &tool.description {
            wire["description"] = json!(description);
        }
        tools.push(wire);
    }
    let tool_choice = match &request.tool_choice {
        None => None,
        Some(ToolChoice::Auto) => Some(json!("auto")),
        Some(ToolChoice::Disabled) => Some(json!("none")),
        Some(ToolChoice::Required) => Some(json!("required")),
        Some(ToolChoice::Specific { name }) => Some(json!({"name": name})),
        // `ToolChoice` is `#[non_exhaustive]` too.
        Some(_) => {
            let subject = Subject::Field {
                field: RequestField::ToolChoice,
            };
            return Err(degradations.reject(subject, "unsupported tool choice"));
        }
    };

    let mut turns = Vec::with_capacity(request.messages.len());
    for (index, message) in request.messages.iter().enumerate() {
        let wire = encode_message(index, message, degradations)?;
        let turn = EncodedTurn::new(index, message, wire.content.is_empty());
        turns.push((turn, wire));
    }
    // Skip empty turns, and drop a reply that the skips and drops left last.
    let turns = retain_turns(turns, degradations, |(turn, _)| *turn)?;
    let messages = turns.into_iter().map(|(_, wire)| wire).collect();
    let wire = WireRequest {
        model: &request.model,
        system: request.system.iter().map(|s| s.content.as_str()).collect(),
        messages,
        tools,
        tool_choice,
        temperature: request.temperature,
        top_p: request.top_p,
        max_tokens: request.max_output_tokens,
        stop: &request.stop,
        safety_mode: options.safety_mode,
        stream,
        extra: options.extra,
    };
    serde_json::to_vec(&wire).map_err(Error::serialization)
}

fn encode_message(
    index: usize,
    message: &Message,
    degradations: &mut Degradations,
) -> Result<WireMessage> {
    let subject = Subject::Content {
        message: index,
        block: None,
    };
    let wire = match message {
        Message::User(user) => {
            let content = match &user.content {
                UserContent::Text(text) => vec![json!({"type": "text", "text": text})],
                UserContent::Parts(parts) => encode_parts(index, parts, degradations)?,
                // `UserContent` is `#[non_exhaustive]`: any other shape reads as parts.
                other => encode_parts(index, &other.to_parts(), degradations)?,
            };
            WireMessage::new("user", content)
        }
        Message::Assistant(assistant) => {
            let content = encode_assistant(index, &assistant.content, degradations)?;
            WireMessage::new("assistant", content)
        }
        Message::Tool(tool) => {
            let content = match &tool.content {
                ToolResultContent::Text(text) => vec![json!({"type": "text", "text": text})],
                ToolResultContent::Parts(parts) => encode_parts(index, parts, degradations)?,
                // Acme takes text results: an object goes as its JSON text.
                ToolResultContent::Object(object) => {
                    let text = serde_json::to_string(object).map_err(Error::serialization)?;
                    vec![json!({"type": "text", "text": text})]
                }
                _ => return Err(degradations.reject(subject, "unsupported tool result content")),
            };
            let mut wire = WireMessage::new("tool", content);
            wire.tool_call_id = Some(tool.tool_call_id.clone());
            wire.is_error = tool.is_error;
            return Ok(wire);
        }
        // `Message` is `#[non_exhaustive]`: refuse what this adapter does not know.
        _ => return Err(degradations.reject(subject, "unsupported message type")),
    };
    Ok(wire)
}

/// User or tool-result parts.
fn encode_parts(
    index: usize,
    parts: &[ContentPart],
    degradations: &mut Degradations,
) -> Result<Vec<Value>> {
    let mut wire = Vec::with_capacity(parts.len());
    for (block, part) in parts.iter().enumerate() {
        let subject = Subject::Content {
            message: index,
            block: Some(block),
        };
        match part {
            ContentPart::Text(text) => wire.push(json!({"type": "text", "text": text.text})),
            // Acme's own part, or the caller's raw part (no origin): verbatim.
            ContentPart::Other(other)
                if other
                    .origin
                    .as_deref()
                    .is_none_or(|origin| origin == NAMESPACE) =>
            {
                wire.push(other_part(other));
            }
            // Another provider's part is dropped, with a warning.
            ContentPart::Other(other) => drop_other(index, block, other, degradations),
            // Images and files: preflight refused them already.
            ContentPart::Image(_) | ContentPart::File(_) => {
                return Err(degradations.reject(subject, "only text input is supported"));
            }
            _ => return Err(degradations.reject(subject, "unsupported content part")),
        }
    }
    Ok(wire)
}

fn encode_assistant(
    index: usize,
    content: &[ContentBlock],
    degradations: &mut Degradations,
) -> Result<Vec<Value>> {
    let mut wire = Vec::with_capacity(content.len());
    for (block, content) in content.iter().enumerate() {
        let subject = Subject::Content {
            message: index,
            block: Some(block),
        };
        match content {
            ContentBlock::Text(text) => wire.push(json!({"type": "text", "text": text.text})),
            // Acme has no refusal block, so a refusal another provider
            // wrote goes back as the text it was: never dropped silently.
            ContentBlock::Refusal(refusal) => {
                wire.push(json!({"type": "text", "text": refusal.text}));
            }
            // Replay only the reasoning this adapter minted.
            ContentBlock::Reasoning(reasoning) => {
                match reasoning.provider_data.get_raw(NAMESPACE, "signature") {
                    Some(signature) => wire.push(json!({
                        "type": "reasoning",
                        "text": reasoning.text,
                        "signature": signature,
                    })),
                    None => {
                        let from: Vec<&str> = reasoning
                            .provider_data
                            .foreign_namespaces(NAMESPACE)
                            .collect();
                        // Replay state stays in the conversation: a coded
                        // drop logs one `WARN` per model, not one per turn.
                        let (state, message) = if from.is_empty() {
                            (
                                ReplayState::Reasoning,
                                "reasoning without an Acme signature cannot be replayed; it was dropped"
                                    .to_owned(),
                            )
                        } else {
                            (
                                ReplayState::ForeignReasoning,
                                format!(
                                    "reasoning from {} can be replayed only to its origin; it was dropped",
                                    describe_origins(&from)
                                ),
                            )
                        };
                        degradations.dropped_replay_state(state, subject, message);
                    }
                }
            }
            ContentBlock::Other(other) if other.origin.as_deref() == Some(NAMESPACE) => {
                wire.push(other_part(other));
            }
            ContentBlock::Other(other) => drop_other(index, block, other, degradations),
            ContentBlock::ProviderToolCall(_) | ContentBlock::ProviderToolResult(_) => {
                let origin = describe_origins(content.origin().as_slice());
                degradations.dropped_replay_state(
                    ReplayState::ForeignContent,
                    subject,
                    format!(
                        "`{}` content from {origin} can be replayed only to its origin; it was dropped",
                        content.type_name()
                    ),
                );
            }
            ContentBlock::ToolCall(call) => wire.push(json!({
                "type": "tool_call",
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            })),
            // Preflight refused assistant images already.
            ContentBlock::Image(_) => {
                return Err(degradations.reject(subject, "image replay is not supported"));
            }
            _ => return Err(degradations.reject(subject, "unsupported content block")),
        }
    }
    Ok(wire)
}

/// `Other` content Acme minted (or the caller wrote): its wire fields,
/// verbatim.
fn other_part(other: &OtherContent) -> Value {
    let mut part = other.data.clone();
    part.insert("type".to_owned(), json!(other.type_name));
    Value::Object(part.into_iter().collect())
}

fn drop_other(index: usize, block: usize, other: &OtherContent, degradations: &mut Degradations) {
    let subject = Subject::Content {
        message: index,
        block: Some(block),
    };
    let origin = describe_origins(other.origin.as_slice());
    degradations.dropped_replay_state(
        ReplayState::ForeignContent,
        subject,
        format!(
            "`{}` content from {origin} can be replayed only to its origin; it was dropped",
            other.type_name
        ),
    );
}

// ── Required: response decoding (guide §3, §6) ──

/// The Acme response body.
#[derive(Deserialize)]
struct WireResponse {
    id: Option<String>,
    model: Option<String>,
    #[serde(default)]
    output: Vec<WireOutput>,
    finish_reason: Option<String>,
    usage: Option<WireUsage>,
    region: Option<String>,
}

/// One output item: `{"type": "text" | "reasoning" | …, …}`.
#[derive(Deserialize)]
struct WireOutput {
    #[serde(rename = "type")]
    kind: String,
    #[serde(flatten)]
    fields: ExtraMap,
}

/// Acme usage: input includes cached tokens, output includes reasoning.
#[derive(Deserialize, Default, Clone, Copy)]
struct WireUsage {
    input_tokens: Option<u64>,
    cached_tokens: Option<u64>,
    output_tokens: Option<u64>,
    reasoning_tokens: Option<u64>,
}

fn decode_response(
    status: StatusCode,
    body: &str,
    request_id: Option<String>,
) -> Result<ChatResponse> {
    // A proxy's HTML page or a body of another shape: traceable to the
    // response, never retried.
    let wire: WireResponse = serde_json::from_str(body).map_err(|error| {
        malformed_response(status, body.as_bytes(), error)
            .with_provider(PROVIDER_NAME)
            .with_request_id_opt(request_id.as_deref())
    })?;
    let content = wire.output.into_iter().map(output_block).collect();
    let mut response = ChatResponse::new(content).provider(PROVIDER_NAME);
    if let Some(raw) = wire.finish_reason {
        response = response
            .finish_reason(finish_reason(&raw))
            .finish_reason_raw(raw);
    }
    if let Some(usage) = wire.usage {
        response = response.usage(normalize_usage(usage));
    }
    if let Some(id) = wire.id {
        response = response.id(id);
    }
    if let Some(model) = wire.model {
        response = response.model(model);
    }
    if let Some(request_id) = request_id {
        response = response.request_id(request_id);
    }
    if let Some(region) = &wire.region {
        response = response.provider_data(region_data(region)?);
    }
    Ok(response)
}

fn output_block(output: WireOutput) -> ContentBlock {
    let text = |key: &str| {
        output
            .fields
            .get(key)
            .and_then(Value::as_str)
            .map(str::to_owned)
    };
    match output.kind.as_str() {
        "text" => ContentBlock::text(text("text").unwrap_or_default()),
        "tool_call" => ContentBlock::tool_call(
            text("id").unwrap_or_default(),
            text("name").unwrap_or_default(),
            text("arguments").unwrap_or_default(),
        ),
        "reasoning" => {
            let summary = text("text").unwrap_or_default();
            let sig = text("signature").unwrap_or_default();
            let state = ProviderData::new().with_raw(NAMESPACE, "signature", json!(sig));
            ContentBlock::Reasoning(Reasoning::new(summary).with_provider_data(state))
        }
        // An item this adapter does not model: kept verbatim, replayed to Acme.
        _ => ContentBlock::Other(
            OtherContent::new(output.kind, output.fields).with_origin(NAMESPACE),
        ),
    }
}

fn finish_reason(raw: &str) -> FinishReason {
    match raw {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "tool_calls" => FinishReason::ToolCalls,
        "safety" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.to_owned()),
    }
}

fn normalize_usage(wire: WireUsage) -> Usage {
    let mut usage = Usage::new();
    usage.input_tokens = wire.input_tokens;
    usage.cache_read_input_tokens = wire.cached_tokens;
    usage.output_tokens = wire.output_tokens;
    usage.reasoning_tokens = wire.reasoning_tokens;
    usage.with_derived_total()
}

// ── Only if your API has response-level data: a typed provider data entry (guide §3) ──

/// The Acme region that served a call: response-level provider data, read
/// with `response.provider_data.get::<AcmeRegion>()`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcmeRegion {
    /// The region, such as `eu-west`.
    pub region: String,
}

impl ProviderDataEntry for AcmeRegion {
    const NAMESPACE: &'static str = NAMESPACE;
    const KEY: &'static str = "region";
}

/// The response-level provider data of a call Acme served in `region`,
/// on the unary and the streaming path alike.
fn region_data(region: &str) -> Result<ProviderData> {
    let mut provider_data = ProviderData::new();
    provider_data.insert(&AcmeRegion {
        region: region.to_owned(),
    })?;
    Ok(provider_data)
}

// ── Required: streaming through the shared driver (`SseChatStream`), in event order (guide §5) ──

/// Turns Acme SSE events into normalized stream events.
///
/// The Acme stream has no start frame: every `delta` carries the response
/// id (`response_id`) and model, so the decoder synthesizes `ResponseStart` from the first
/// one. An `error` event may arrive before any `delta`, and then the stream
/// fails before `ResponseStart`.
struct StreamDecoder {
    request_id: Option<String>,
    started: bool,
    finished: bool,
    /// The open block: its index and type.
    open: Option<(usize, StreamBlockType)>,
    next_index: usize,
    vendor_frames: VendorFrames,
}

impl StreamDecoder {
    fn new(request_id: Option<String>) -> Self {
        Self {
            request_id,
            started: false,
            finished: false,
            open: None,
            next_index: 0,
            vendor_frames: VendorFrames::new(PROVIDER_NAME, NAMESPACE),
        }
    }

    /// Process one SSE event.
    fn process(&mut self, event: &str, data: &str) -> Vec<Result<StreamEvent>> {
        let mut out = Vec::new();
        if self.finished {
            return out;
        }
        if !KNOWN_EVENTS.contains(&event) {
            // Before `ResponseStart` too: `finish_stream` holds them until it.
            let frames = self.vendor_frames.surface(event, data);
            out.extend(frames.into_iter().map(Ok));
            return out;
        }
        let json: Value = match serde_json::from_str(data) {
            Ok(json) => json,
            Err(error) => {
                self.finished = true;
                let error = Error::stream_decode(format!("invalid JSON in `{event}` event"))
                    .authored()
                    .with_upstream_message(error.to_string())
                    .with_provider(PROVIDER_NAME);
                out.push(Err(error));
                return out;
            }
        };
        match event {
            "delta" => self.delta(&json, &mut out),
            "done" => self.done(&json, &mut out),
            _ => {
                // `error`: the stream failed, possibly before it started.
                self.finished = true;
                out.push(Err(map_stream_api_error(&json, self.request_id.as_deref())));
            }
        }
        out
    }

    /// The end of the body, before `done`.
    fn incomplete(&mut self) -> Error {
        self.finished = true;
        Error::stream(
            StreamErrorKind::Incomplete,
            "the stream ended before `done`",
        )
        .authored()
        .with_provider(PROVIDER_NAME)
    }

    fn start(&mut self, json: &Value, out: &mut Vec<Result<StreamEvent>>) {
        if self.started {
            return;
        }
        self.started = true;
        let field = |key: &str| json.get(key).and_then(Value::as_str).map(str::to_owned);
        out.push(Ok(StreamEvent::response_start(
            field("response_id"),
            field("model"),
        )));
    }

    fn delta(&mut self, json: &Value, out: &mut Vec<Result<StreamEvent>>) {
        self.start(json, out);
        let kind = json.get("type").and_then(Value::as_str).unwrap_or_default();
        let block_type = match kind {
            "tool_call" => return self.tool_call_delta(json, out),
            "text" => StreamBlockType::Text,
            "reasoning" => StreamBlockType::Reasoning,
            _ => {
                // An item this adapter does not model, complete in one delta.
                self.close(out);
                let Value::Object(fields) = json.clone() else {
                    return;
                };
                let mut data: ExtraMap = fields.into_iter().collect();
                for key in ["type", "response_id", "model"] {
                    data.remove(key);
                }
                let index = self.open_index();
                let other = OtherContent::new(kind, data).with_origin(NAMESPACE);
                out.push(Ok(StreamEvent::other_start(index, other)));
                out.push(Ok(StreamEvent::block_stop(index)));
                return;
            }
        };
        let index = match self.open {
            Some((index, open)) if open == block_type => index,
            _ => {
                self.close(out);
                let index = self.open_index();
                self.open = Some((index, block_type));
                out.push(Ok(StreamEvent::block_start(index, block_type)));
                index
            }
        };
        if let Some(text) = json.get("text").and_then(Value::as_str) {
            out.push(Ok(match block_type {
                StreamBlockType::Reasoning => StreamEvent::reasoning_delta(index, text),
                _ => StreamEvent::text_delta(index, text),
            }));
        }
        if let Some(sig) = json.get("signature").and_then(Value::as_str) {
            let provider_data = ProviderData::new().with_raw(NAMESPACE, "signature", json!(sig));
            out.push(Ok(StreamEvent::block_provider_data(index, provider_data)));
        }
    }

    /// A tool call: the first delta carries its id and name, later ones
    /// more of its arguments.
    fn tool_call_delta(&mut self, json: &Value, out: &mut Vec<Result<StreamEvent>>) {
        let field = |key: &str| json.get(key).and_then(Value::as_str);
        let index = match (field("id"), self.open) {
            (Some(id), _) => {
                self.close(out);
                let index = self.open_index();
                self.open = Some((index, StreamBlockType::ToolCall));
                let name = field("name").unwrap_or_default();
                out.push(Ok(StreamEvent::tool_call_start(index, id, name)));
                index
            }
            (None, Some((index, StreamBlockType::ToolCall))) => index,
            (None, _) => {
                self.finished = true;
                let error = Error::stream_protocol("tool call arguments without a tool call")
                    .authored()
                    .with_provider(PROVIDER_NAME);
                out.push(Err(error));
                return;
            }
        };
        if let Some(arguments) = field("arguments") {
            out.push(Ok(StreamEvent::tool_call_delta(index, arguments)));
        }
    }

    fn done(&mut self, json: &Value, out: &mut Vec<Result<StreamEvent>>) {
        self.start(json, out);
        self.close(out);
        self.finished = true;
        let raw = json.get("finish_reason").and_then(Value::as_str);
        let usage = json
            .get("usage")
            .and_then(|usage| serde_json::from_value::<WireUsage>(usage.clone()).ok());
        let metadata = MetadataBuilder::new()
            .with_finish_reason_opt(raw.map(finish_reason))
            .with_finish_reason_raw_opt(raw)
            .with_usage_opt(usage.map(normalize_usage))
            .with_provider(PROVIDER_NAME)
            .with_request_id_opt(self.request_id.as_deref());
        // The collector merges it into the response's provider data.
        if let Some(region) = json.get("region").and_then(Value::as_str) {
            out.push(region_data(region).map(StreamEvent::response_provider_data));
        }
        out.push(Ok(metadata.build()));
        // A repeated vendor frame type's whole array, before the stop.
        out.extend(self.vendor_frames.finish().map(Ok));
        out.push(Ok(StreamEvent::ResponseStop));
    }

    fn open_index(&mut self) -> usize {
        let index = self.next_index;
        self.next_index += 1;
        index
    }

    fn close(&mut self, out: &mut Vec<Result<StreamEvent>>) {
        if let Some((index, _)) = self.open.take() {
            out.push(Ok(StreamEvent::block_stop(index)));
        }
    }
}

/// The driver (`SseChatStream`) calls the decoder for each event, and
/// once more when the body ends or fails before `done`.
impl ChatFrameDecoder<SseEvent> for StreamDecoder {
    fn frame(&mut self, event: SseEvent) -> Vec<Result<StreamEvent>> {
        self.process(&event.event, &event.data)
    }

    /// The body ended before `done`.
    fn end(&mut self) -> Vec<Result<StreamEvent>> {
        vec![Err(self.incomplete())]
    }

    /// Reading the body failed: `error` is already classified (a stream
    /// timeout stays a timeout); say who served and which request it was.
    fn fail(&mut self, error: Error) -> Vec<Result<StreamEvent>> {
        self.finished = true;
        let error = error
            .with_provider(PROVIDER_NAME)
            .with_request_id_opt(self.request_id.as_deref());
        vec![Err(error)]
    }

    /// After `done` or an error, the stream ends without waiting for the
    /// body to close.
    fn is_done(&self) -> bool {
        self.finished
    }
}

// ── Only if your API streams in another framing: NDJSON lines (guide §5) ──

/// Acme's NDJSON stream (`accept: application/x-ndjson`): one line per
/// event, `{"event":"delta","data":{..}}`. Only the frame differs from the
/// SSE stream, so the SSE decoder does the work; an API with lines of its
/// own decodes them here.
struct LineDecoder(StreamDecoder);

/// The driver (`FramedChatStream`) gives the decoder each line, and the
/// same `end`, `fail` and `is_done` calls as over SSE.
impl ChatFrameDecoder<String> for LineDecoder {
    fn frame(&mut self, line: String) -> Vec<Result<StreamEvent>> {
        let mut frame: Value = match serde_json::from_str(&line) {
            Ok(frame) => frame,
            Err(error) => {
                self.0.finished = true;
                let error = Error::stream_decode("invalid JSON in an Acme stream line")
                    .authored()
                    .with_upstream_message(error.to_string())
                    .with_provider(PROVIDER_NAME);
                return vec![Err(error)];
            }
        };
        let event = frame["event"].as_str().unwrap_or_default().to_owned();
        let data = frame["data"].take().to_string();
        self.0.process(&event, &data)
    }

    fn end(&mut self) -> Vec<Result<StreamEvent>> {
        vec![Err(self.0.incomplete())]
    }

    fn fail(&mut self, error: Error) -> Vec<Result<StreamEvent>> {
        ChatFrameDecoder::fail(&mut self.0, error)
    }

    fn is_done(&self) -> bool {
        self.0.finished
    }
}

/// The end of `chat_stream` for an NDJSON response.
fn ndjson_stream(
    response: HttpResponse,
    max_line_size: usize,
    request_id: Option<String>,
) -> FramedChatStream<NdjsonStream<Body>, LineDecoder> {
    let lines = NdjsonStream::new(response.into_body()).max_line_size(max_line_size);
    FramedChatStream::new(lines, LineDecoder(StreamDecoder::new(request_id)))
}

// ── Required: errors, with authored messages (guide §4) ──

/// Map an Acme HTTP error response: a pure function of the status, the
/// headers and the body, so the error contracts can call it.
pub fn map_http_error(status: StatusCode, headers: &HeaderMap, body: &str) -> Error {
    let envelope = error_envelope(body);
    let (kind, code, param, text) = match &envelope {
        Some(envelope) => (
            envelope.error_type.as_deref(),
            envelope.code.as_deref(),
            envelope.param.as_deref(),
            envelope.message.as_deref().unwrap_or_default(),
        ),
        // Classify from an unrecognised body, but never quote it.
        None => (None, None, None, body),
    };
    let error = classify(Some(status), kind, code, param, text)
        .with_context(error_context(status, headers))
        .authored();
    let error = match envelope {
        Some(envelope) => envelope.apply(error),
        None => error,
    };
    error
        .with_provider(PROVIDER_NAME)
        .with_body_opt((!body.is_empty()).then_some(body))
}

/// Map an in-band `error` event with the same classification as HTTP
/// errors, so retry and fallback behave the same whenever it arrives.
fn map_stream_api_error(json: &Value, request_id: Option<&str>) -> Error {
    let Some(envelope) = error_envelope(&json.to_string()) else {
        return Error::stream_protocol("malformed `error` event")
            .authored()
            .with_provider(PROVIDER_NAME)
            .with_request_id_opt(request_id);
    };
    let text = envelope.message.clone().unwrap_or_default();
    let error = classify(
        None,
        envelope.error_type.as_deref(),
        envelope.code.as_deref(),
        envelope.param.as_deref(),
        &text,
    );
    envelope
        .apply(error.authored())
        .with_provider(PROVIDER_NAME)
        .with_request_id_opt(request_id)
}

/// Pick the variant from the Acme error type and code, then the status.
/// `text` (the upstream message, or an unrecognised body) only informs the
/// choice; the message is always the adapter's own.
fn classify(
    status: Option<StatusCode>,
    kind: Option<&str>,
    code: Option<&str>,
    param: Option<&str>,
    text: &str,
) -> Error {
    let message = match status {
        // The code alone: a `StatusCode`'s `Display` adds the reason phrase.
        Some(status) => format!("HTTP {}", status.as_u16()),
        None => "the Acme stream reported an error".to_owned(),
    };
    // Exhausted credit is not retryable, even when Acme answers 429.
    if code.is_some_and(|code| QUOTA_ERROR_CODES.contains(&code)) {
        return Error::quota_exhausted(message);
    }
    match (kind, status) {
        (Some("authentication_error"), _) => Error::auth(message),
        (Some("rate_limit_error"), _) => Error::rate_limited(message),
        (Some("overloaded_error"), _) => Error::overloaded(message),
        (Some("context_length_exceeded"), _) => Error::context_length_exceeded(message, None),
        (Some("model_not_found"), _) => Error::model_not_found(message),
        // Acme's own refusal of a parameter the caller set: map it only
        // through this adapter's table of wire names, so a parameter the
        // adapter got wrong stays an invalid request.
        (Some("unsupported_parameter"), _) => match param.and_then(wire_field) {
            Some(field) => Error::unsupported_by_provider(
                Some(Subject::Field { field }),
                format!("{message}: the model does not support `{field}`"),
            ),
            None => Error::invalid_request(message),
        },
        (_, Some(StatusCode::NOT_FOUND)) if text.contains("model") => {
            Error::model_not_found(message)
        }
        (Some("invalid_request_error"), _) => Error::invalid_request(message),
        // What the status means on its own, the same on every adapter:
        // 408/504 time out, 529 is overload, a 499 is not retried.
        (_, Some(status)) => status_error(status, message),
        (_, None) => Error::provider(message),
    }
}

/// The portable field behind an Acme request parameter, for refusals.
fn wire_field(param: &str) -> Option<RequestField> {
    match param {
        "temperature" => Some(RequestField::Temperature),
        "top_p" => Some(RequestField::TopP),
        "stop" => Some(RequestField::Stop),
        _ => None,
    }
}

// ── Only if your API speaks Chat Completions: a `CompatQuirks` dialect instead (guide §1) ──

/// The first row of the guide's table: an OpenAI-compatible endpoint needs
/// only a dialect and a builder call.
fn internal_gateway(token: &str) -> Result<allama_openai_compat::Provider> {
    use allama_openai_compat::{CompatQuirks, Provider};

    let mut quirks = CompatQuirks::new("openai-compat:internal");
    quirks.supports_json_schema = false;
    quirks.quota_error_codes.push("quota_exceeded".into());

    let provider = Provider::builder()
        .base_url("https://llm.internal.example/v1")
        .bearer_token(token)
        .provider_name("internal")
        .quirks(quirks)
        .build()?;
    Ok(provider)
}

// ── Only if applications pick providers by model address: registration (guide §7) ──

/// Register the adapter, so applications reach it by model address.
fn registry() -> allama_providers::Registry {
    use allama_providers::Registry;

    let mut registry = Registry::builtin(); // or Registry::new() for none
    registry.register_chat("acme", AcmeProvider::from_env);
    registry
}

// ── Not part of an adapter: the demo's canned wire data and `main` ──

/// A canned non-streaming response.
const OK_BODY: &str = r#"{
  "id": "resp_1",
  "model": "acme-large",
  "output": [
    {"type": "reasoning", "text": "The user greets me.", "signature": "sig_1"},
    {"type": "text", "text": "Hello!"}
  ],
  "finish_reason": "stop",
  "usage": {"input_tokens": 12, "cached_tokens": 4, "output_tokens": 7, "reasoning_tokens": 3},
  "region": "eu-west"
}"#;

/// The same response, streamed. The stream has no start frame.
const OK_STREAM: &str = "event: delta\n\
data: {\"response_id\":\"resp_1\",\"model\":\"acme-large\",\"type\":\"reasoning\",\"text\":\"The user greets me.\"}\n\n\
event: delta\n\
data: {\"response_id\":\"resp_1\",\"model\":\"acme-large\",\"type\":\"reasoning\",\"signature\":\"sig_1\"}\n\n\
event: delta\n\
data: {\"response_id\":\"resp_1\",\"model\":\"acme-large\",\"type\":\"text\",\"text\":\"Hel\"}\n\n\
event: delta\n\
data: {\"response_id\":\"resp_1\",\"model\":\"acme-large\",\"type\":\"text\",\"text\":\"lo!\"}\n\n\
event: done\n\
data: {\"finish_reason\":\"stop\",\"usage\":{\"input_tokens\":12,\"cached_tokens\":4,\"output_tokens\":7,\"reasoning_tokens\":3},\"region\":\"eu-west\"}\n\n";

/// The same stream as NDJSON lines.
const OK_NDJSON: &str = "{\"event\":\"delta\",\"data\":{\"response_id\":\"resp_1\",\"model\":\"acme-large\",\"type\":\"reasoning\",\"text\":\"The user greets me.\"}}\n\
{\"event\":\"delta\",\"data\":{\"response_id\":\"resp_1\",\"model\":\"acme-large\",\"type\":\"reasoning\",\"signature\":\"sig_1\"}}\n\
{\"event\":\"delta\",\"data\":{\"response_id\":\"resp_1\",\"model\":\"acme-large\",\"type\":\"text\",\"text\":\"Hel\"}}\n\
{\"event\":\"delta\",\"data\":{\"response_id\":\"resp_1\",\"model\":\"acme-large\",\"type\":\"text\",\"text\":\"lo!\"}}\n\
{\"event\":\"done\",\"data\":{\"finish_reason\":\"stop\",\"usage\":{\"input_tokens\":12,\"cached_tokens\":4,\"output_tokens\":7,\"reasoning_tokens\":3},\"region\":\"eu-west\"}}\n";

#[tokio::main]
async fn main() -> Result<()> {
    use allama::ChatStreamExt;
    use allama_http::{ReplayResponse, ReplayTransport};

    let transport = ReplayTransport::new([
        ReplayResponse::json(StatusCode::OK, OK_BODY)
            .header("x-request-id", HeaderValue::from_static("req_1")),
        ReplayResponse::sse(OK_STREAM),
        ReplayResponse::json(
            StatusCode::TOO_MANY_REQUESTS,
            r#"{"error":{"type":"rate_limit_error","code":"credit_exhausted","message":"Out of credit for 'my prompt'"}}"#,
        ),
    ]);
    let provider = AcmeProvider::builder()
        .api_key("demo-key")
        .http_transport(transport.clone())
        .build()?;

    let request = ChatRequest::new("acme-large")
        .system("You are concise.")
        .user("Hi!")
        .seed(7)
        .with_options_json(NAMESPACE, json!({"safety_mode": "strict"}));
    let response = provider.chat(&request).await?;
    println!("chat: {}", response.text_or_empty());
    for warning in &response.warnings {
        println!("  warning: {warning}");
    }
    let sent = transport.requests()[0]
        .body_text()
        .unwrap_or_default()
        .to_owned();
    println!("  sent: {sent}");

    let streamed = provider
        .chat_stream(&request)
        .await?
        .collect_response()
        .await?;
    println!("chat_stream: {}", streamed.text_or_empty());
    if let Some(served) = streamed.provider_data.get::<AcmeRegion>()? {
        println!("  region: {}", served.region);
    }

    let lines = HttpResponse::from_bytes(StatusCode::OK, HeaderMap::new(), OK_NDJSON);
    let max_line_size = allama_http::ndjson::NdjsonDecoder::DEFAULT_MAX_LINE_SIZE;
    let from_lines = ndjson_stream(lines, max_line_size, None)
        .collect_response()
        .await?;
    println!("chat_stream over NDJSON: {}", from_lines.text_or_empty());

    let error = provider.chat(&request).await.unwrap_err();
    println!("error: {error}");
    println!("  log: {}", error.as_log());

    let gateway = internal_gateway("demo-token")?;
    println!("gateway: {}", gateway.provider_name());
    match registry().chat_from_model("acme:acme-large") {
        Ok((provider, model)) => println!("registry: {} {model}", provider.provider_name()),
        Err(error) => println!("registry: {error}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use allama::{ChatStreamExt, RequestOptions, ToolDefinition, WarningKind};
    use allama_conformance::contract::{
        assert_auth_error_contract, assert_context_length_exceeded_contract,
        assert_overloaded_error_contract, assert_quota_exhausted_contract,
        assert_rate_limited_error_contract, assert_strict_policy_rejects,
        assert_strict_policy_rejects_stream, assert_unsupported_hint_warns,
        assert_unsupported_hint_warns_stream, collect_chat_stream,
    };
    use allama_conformance::error_kinds::http_error_mapping;
    use allama_conformance::probe::ProbeOutcome;
    use allama_conformance::stream_end::StreamEnds;
    use allama_conformance::stream_errors::ErrorKindProbe;
    use allama_conformance::stream_frames::{StreamFraming, assert_unknown_stream_frames_surfaced};
    use allama_conformance::{
        AdapterUnderTest, Contract, ResponseCapture, TransportOutcome, assert_chat_adapter,
        assert_usage_contract,
    };
    use allama_http::{ReplayResponse, ReplayTransport};
    use futures_util::StreamExt;

    use super::*;

    // ── Required: the probes and the adapter aggregate (guide §6) ──

    const MODEL: &str = "acme-large";

    /// A response with a tool call, as `chat` gets it.
    const TOOL_BODY: &str = r#"{"id":"resp_2","output":[{"type":"tool_call","id":"call_1","name":"weather","arguments":"{\"city\":\"Lisbon\"}"}],"finish_reason":"tool_calls"}"#;

    /// The same response, streamed.
    const TOOL_STREAM: &str = "event: delta\n\
data: {\"response_id\":\"resp_2\",\"type\":\"tool_call\",\"id\":\"call_1\",\"name\":\"weather\",\"arguments\":\"{\\\"city\\\":\"}\n\n\
event: delta\n\
data: {\"response_id\":\"resp_2\",\"type\":\"tool_call\",\"arguments\":\"\\\"Lisbon\\\"}\"}\n\n\
event: done\n\
data: {\"finish_reason\":\"tool_calls\"}\n\n";

    /// A response cut off at `max_tokens`, as `chat` gets it.
    const LENGTH_BODY: &str = r#"{"id":"resp_3","model":"acme-large","output":[{"type":"text","text":"Hel"}],"finish_reason":"length","usage":{"input_tokens":12,"output_tokens":1}}"#;

    /// The same response, streamed.
    const LENGTH_STREAM: &str = "event: delta\n\
data: {\"response_id\":\"resp_3\",\"model\":\"acme-large\",\"type\":\"text\",\"text\":\"Hel\"}\n\n\
event: done\n\
data: {\"finish_reason\":\"length\",\"usage\":{\"input_tokens\":12,\"output_tokens\":1}}\n\n";

    fn provider(transport: &ReplayTransport) -> AcmeProvider {
        AcmeProvider::builder()
            .api_key("test-key")
            .http_transport(transport.clone())
            .build()
            .expect("the provider builds")
    }

    fn bodies(transport: &ReplayTransport) -> Vec<Value> {
        transport
            .requests()
            .iter()
            .map(|request| serde_json::from_str(request.body_text().unwrap()).unwrap())
            .collect()
    }

    async fn probe(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
        let transport = ReplayTransport::new([ReplayResponse::json(StatusCode::OK, OK_BODY)]);
        let result = provider(&transport).chat(&request).await;
        ProbeOutcome::new(result, bodies(&transport))
    }

    async fn stream_probe(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
        let transport = ReplayTransport::new([ReplayResponse::sse(OK_STREAM)]);
        // Collecting keeps every StreamEvent::Warning; an error from
        // chat_stream or any stream item becomes the outcome's error.
        let result = match provider(&transport).chat_stream(&request).await {
            Ok(stream) => stream.collect_response().await,
            Err(err) => Err(err),
        };
        ProbeOutcome::new(result, bodies(&transport))
    }

    /// Acme's HTTP error mapping, as the contracts call it.
    fn map(status: u16, headers: &[(&str, &str)], body: String) -> Error {
        http_error_mapping(map_http_error)(status, headers, body)
    }

    /// Acme's envelope for each kind, told apart by its type and code. The
    /// upstream text goes after the message, where a classifier that reads
    /// the wording would find it.
    fn error_envelope(kind: ErrorKindProbe, upstream: &str) -> Option<String> {
        let (error_type, code, message) = match kind {
            ErrorKindProbe::RateLimited => ("rate_limit_error", None, "Slow down."),
            ErrorKindProbe::Overloaded => ("overloaded_error", None, "Acme is busy."),
            ErrorKindProbe::InvalidRequest => ("invalid_request_error", None, "Bad request."),
            ErrorKindProbe::Auth => ("authentication_error", None, "Invalid API key."),
            ErrorKindProbe::QuotaExhausted => (
                "rate_limit_error",
                Some("credit_exhausted"),
                "Out of credit.",
            ),
            ErrorKindProbe::ContextLength => ("context_length_exceeded", None, "Too long."),
            ErrorKindProbe::ModelNotFound => ("model_not_found", None, "No such model."),
            ErrorKindProbe::Unsupported => ("unsupported_parameter", None, "Not supported."),
            // A kind Acme does not map, such as one added later.
            _ => return None,
        };
        let message = format!("{message} {upstream}");
        let mut error = json!({"type": error_type, "message": message.trim_end()});
        if error_type == "unsupported_parameter" {
            error["param"] = json!("temperature");
        }
        if let Some(code) = code {
            error["code"] = json!(code);
        }
        Some(json!({ "error": error }).to_string())
    }

    /// One call with transport options attached, on `chat` or
    /// `chat_stream`.
    async fn send(options: RequestOptions, streaming: bool) -> TransportOutcome {
        let transport = ReplayTransport::new([
            ReplayResponse::json(StatusCode::OK, OK_BODY),
            ReplayResponse::sse(OK_STREAM),
        ]);
        let mut request = ChatRequest::new(MODEL).user("Hi!");
        request.options = options;
        let provider = provider(&transport);
        let result = if streaming {
            provider.chat_stream(&request).await.map(drop)
        } else {
            provider.chat(&request).await.map(drop)
        };
        TransportOutcome::new(result, transport.requests())
    }

    #[tokio::test]
    async fn the_adapter_passes_every_contract() {
        // The capability answers shape the probe requests.
        let acme = provider(&ReplayTransport::new([]));
        // Validation is a dry run of the encoding, so it agrees both ways.
        let validation = acme
            .validate_chat_request(&ChatRequest::new(MODEL).user("Hi"))
            .unwrap();
        assert_eq!(validation.coverage(), ValidationCoverage::Encoding);

        // An in-band error arrives after the stream started.
        let first_delta = OK_STREAM.split_inclusive("\n\n").next().unwrap();
        // A stream cut short, and one that ends without a finish reason.
        let cut_short = OK_STREAM.split("event: done").next().unwrap();
        let without_finish = format!("{cut_short}event: done\ndata: {{}}\n\n");
        let slot = CredentialSlot::new(DEFAULT_CREDENTIAL_SCOPE, CredentialPlacement::Bearer);
        let under_test = AdapterUnderTest::new([MODEL], NAMESPACE, probe, stream_probe)
            // Acme has no strict mode.
            .strict_not_honoured([MODEL])
            .error_mapping(map)
            .error_envelope(error_envelope)
            .stream_body(move |envelope| format!("{first_delta}event: error\ndata: {envelope}\n\n"))
            .stream_replay(|sse| async move {
                let transport = ReplayTransport::new([ReplayResponse::sse(sse)]);
                provider(&transport)
                    .chat_stream(&ChatRequest::new(MODEL).user("Hi!"))
                    .await
            })
            .stream_ends(StreamEnds::new(cut_short).terminated_without_finish(without_finish))
            .sse_captures([OK_STREAM])
            .transport_options(
                slot,
                |options| send(options, false),
                |options| send(options, true),
            )
            .malformed_response(
                |body| async move {
                    let answer = ReplayResponse::new(StatusCode::OK, body)
                        .header("x-request-id", HeaderValue::from_static("req_1"));
                    provider(&ReplayTransport::new([answer]))
                        .chat(&ChatRequest::new(MODEL).user("Hi!"))
                        .await
                },
                Some("req_1"),
            )
            // The decoding contracts decode captured responses on both paths.
            .unary_replay(|body| async move {
                let transport = ReplayTransport::new([ReplayResponse::json(StatusCode::OK, body)]);
                provider(&transport)
                    .chat(&ChatRequest::new(MODEL).user("Hi!"))
                    .await
            })
            .response_captures([ResponseCapture::new(OK_BODY, OK_STREAM)])
            .tool_call_capture(
                ResponseCapture::new(TOOL_BODY, TOOL_STREAM).wire_tool_call_ids(["call_1"]),
            )
            .length_capture(ResponseCapture::new(LENGTH_BODY, LENGTH_STREAM))
            // Over the real HTTP client: a redirect is never followed.
            .redirect_not_followed(|base_url, streaming| async move {
                let acme = AcmeProvider::builder()
                    .api_key("test-key")
                    .base_url(base_url)
                    .build()?;
                let request = ChatRequest::new(MODEL).user("Hi!");
                match streaming {
                    true => acme.chat_stream(&request).await.map(drop),
                    false => acme.chat(&request).await.map(drop),
                }
            })
            // Acme's model replays reasoning, so no model drops it.
            .skip([Contract::UnreplayableReasoningDropped]);
        assert_chat_adapter(&acme, &under_test).await;
    }

    // ── Required: the targeted contracts and the error mapping (guide §4, §6) ──

    #[tokio::test]
    async fn unsupported_hints_warn_or_fail_on_both_paths() {
        let request = ChatRequest::new(MODEL).user("Hi!").seed(7);
        let ok = || {
            ReplayTransport::new([
                ReplayResponse::json(StatusCode::OK, OK_BODY),
                ReplayResponse::sse(OK_STREAM),
            ])
        };
        let transport = ok();
        assert_unsupported_hint_warns(&provider(&transport), &request).await;
        assert_unsupported_hint_warns_stream(&provider(&transport), &request).await;
        let transport = ok();
        assert_strict_policy_rejects(&provider(&transport), &request).await;
        assert_strict_policy_rejects_stream(&provider(&transport), &request).await;
        assert!(
            transport.requests().is_empty(),
            "a refused request was sent"
        );
    }

    #[tokio::test]
    async fn error_kinds_hold_on_both_paths() {
        let request = ChatRequest::new(MODEL).user("Hi!");
        // Each failure answers `chat`, then `chat_stream`.
        let failing = |status: StatusCode, body: &'static str| {
            let answer = || ReplayResponse::json(status, body);
            provider(&ReplayTransport::new([answer(), answer()]))
        };
        let auth = failing(
            StatusCode::UNAUTHORIZED,
            r#"{"error":{"type":"authentication_error","message":"bad key"}}"#,
        );
        assert_auth_error_contract(&auth, &request, |r| collect_chat_stream(&auth, r)).await;
        let rate = failing(
            StatusCode::TOO_MANY_REQUESTS,
            r#"{"error":{"type":"rate_limit_error","message":"slow down"}}"#,
        );
        assert_rate_limited_error_contract(&rate, &request, |r| collect_chat_stream(&rate, r))
            .await;
        let busy = failing(
            StatusCode::SERVICE_UNAVAILABLE,
            r#"{"error":{"type":"overloaded_error","message":"busy"}}"#,
        );
        assert_overloaded_error_contract(&busy, &request, |r| collect_chat_stream(&busy, r)).await;
        let long = failing(
            StatusCode::BAD_REQUEST,
            r#"{"error":{"type":"context_length_exceeded","message":"too long"}}"#,
        );
        assert_context_length_exceeded_contract(&long, &request, |r| collect_chat_stream(&long, r))
            .await;
        let credit = failing(
            StatusCode::TOO_MANY_REQUESTS,
            r#"{"error":{"type":"rate_limit_error","code":"credit_exhausted","message":"no credit"}}"#,
        );
        assert_quota_exhausted_contract(&credit, &request, |r| collect_chat_stream(&credit, r))
            .await;
    }

    #[test]
    fn error_mapping_classifies() {
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", HeaderValue::from_static("req_9"));
        headers.insert("retry-after", HeaderValue::from_static("3"));
        let quota = r#"{"error":{"type":"rate_limit_error","code":"credit_exhausted","message":"no credit"}}"#;
        let error = map_http_error(StatusCode::TOO_MANY_REQUESTS, &headers, quota);
        assert!(matches!(error, Error::QuotaExhausted { .. }), "{error:?}");
        assert!(!error.is_retryable());

        let error = map_http_error(
            StatusCode::TOO_MANY_REQUESTS,
            &headers,
            r#"{"error":{"message":"slow down"}}"#,
        );
        assert!(matches!(error, Error::RateLimited { .. }), "{error:?}");
        assert_eq!(error.retry_after(), Some(Duration::from_secs(3)));
        assert_eq!(error.request_id(), Some("req_9"));
        assert_eq!(error.upstream_message(), Some("slow down"));
        assert!(error.as_log().to_string().contains("provider=acme"));

        let error = map_http_error(
            StatusCode::BAD_GATEWAY,
            &HeaderMap::new(),
            "<html>bad gateway</html>",
        );
        assert_eq!(error.to_string(), "acme: provider error (502): HTTP 502");
    }

    /// The provenance example of the guide: what goes wrong when upstream
    /// text is marked authored.
    #[test]
    fn authored_messages_never_carry_upstream_text() {
        let (status, body) = (
            StatusCode::BAD_REQUEST,
            r#"{"error":{"message":"Invalid value"}}"#,
        );
        let headers = HeaderMap::new();
        let upstream = "Invalid value: 'my SSN is 123-45-6789'";

        // Bad: the upstream text reaches every log line and FallbackAttempt.
        let bad = Error::invalid_request(format!("rejected: {upstream}")).authored();

        // Good: authored message, upstream text in its own slot.
        let good = Error::invalid_request(format!("HTTP {}", status.as_u16()))
            .with_context(error_context(status, &headers))
            .authored()
            .with_upstream_message(upstream.to_owned())
            .with_body(body.to_owned());

        assert!(bad.as_log().to_string().contains("123-45-6789"));
        assert!(!good.as_log().to_string().contains("123-45-6789"));
        assert!(good.to_string().contains("123-45-6789"));
    }

    #[tokio::test]
    async fn stream_and_unary_agree() {
        let transport = ReplayTransport::new([
            ReplayResponse::json(StatusCode::OK, OK_BODY),
            ReplayResponse::sse(OK_STREAM),
        ]);
        let provider = provider(&transport);
        let request = ChatRequest::new(MODEL).user("Hi!");
        let unary = provider.chat(&request).await.unwrap();
        let streamed = provider
            .chat_stream(&request)
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        assert_eq!(streamed.content, unary.content);
        assert_eq!(streamed.usage, unary.usage);
        assert_eq!(streamed.finish_reason, unary.finish_reason);
        assert_eq!(streamed.id, unary.id);
        assert_eq!(streamed.model, unary.model);

        // A cache hit, normalized by hand from Acme's usage docs: its
        // `input_tokens` already counts the cached ones, and its
        // `output_tokens` the reasoning ones. The contract only checks that
        // the numbers are consistent.
        let usage = unary.usage.as_ref().unwrap();
        assert_eq!(usage.input_tokens, Some(12));
        assert_eq!(usage.cache_read_input_tokens, Some(4));
        assert_eq!(usage.output_tokens, Some(7));
        assert_eq!(usage.reasoning_tokens, Some(3));
        assert_eq!(usage.total_tokens, Some(19));
        assert_usage_contract(usage);

        // The typed response-level entry arrives on both paths.
        let region = unary.provider_data.get::<AcmeRegion>().unwrap();
        assert_eq!(region.unwrap().region, "eu-west");
        assert_eq!(streamed.provider_data, unary.provider_data);

        let bodies = bodies(&transport);
        assert_eq!(bodies[1]["stream"], json!(true));
        assert_eq!(bodies[0].get("stream"), None);
    }

    // ── Only if your API has reasoning, tool calls or typed options: round trips ──

    #[tokio::test]
    async fn reasoning_round_trips_and_foreign_reasoning_is_dropped() {
        let transport = ReplayTransport::new([
            ReplayResponse::json(StatusCode::OK, OK_BODY),
            ReplayResponse::json(StatusCode::OK, OK_BODY),
        ]);
        let provider = provider(&transport);
        let first = provider
            .chat(&ChatRequest::new(MODEL).user("Hi!"))
            .await
            .unwrap();
        let foreign = Message::assistant_blocks(vec![ContentBlock::Reasoning(
            Reasoning::new("theirs").with_provider_data(ProviderData::new().with_raw(
                "other",
                "signature",
                json!("x"),
            )),
        )]);
        let request = ChatRequest::new(MODEL)
            .user("Hi!")
            .message(first.to_assistant_message())
            .message(foreign)
            .user("Again");
        let second = provider.chat(&request).await.unwrap();

        let sent = &bodies(&transport)[1];
        assert_eq!(
            sent["messages"][1]["content"][0]["signature"],
            json!("sig_1")
        );
        // The foreign turn, left empty, is skipped.
        assert_eq!(sent["messages"].as_array().unwrap().len(), 3);
        assert_eq!(sent["messages"][2]["content"][0]["text"], json!("Again"));
        assert!(
            second
                .warnings
                .iter()
                .any(|w| w.kind == WarningKind::Dropped
                    && w.subject
                        == Subject::Content {
                            message: 2,
                            block: Some(0)
                        })
        );
    }

    #[tokio::test]
    async fn tool_calls_round_trip_on_both_paths() {
        let transport = ReplayTransport::new([
            ReplayResponse::json(StatusCode::OK, TOOL_BODY),
            ReplayResponse::sse(TOOL_STREAM),
            ReplayResponse::json(StatusCode::OK, OK_BODY),
        ]);
        let provider = provider(&transport);
        let weather = ToolDefinition::new("weather", json!({"type": "object"}));
        let request = ChatRequest::new(MODEL)
            .user("Weather in Lisbon?")
            .tools([weather])
            .tool_choice(ToolChoice::Required);
        let unary = provider.chat(&request).await.unwrap();
        let streamed = provider
            .chat_stream(&request)
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        assert_eq!(streamed.content, unary.content);
        assert_eq!(unary.finish_reason, Some(FinishReason::ToolCalls));
        let call = unary.first_tool_call().unwrap().clone();
        assert_eq!(call.arguments, r#"{"city":"Lisbon"}"#);

        let mut next = request.message(unary.to_assistant_message());
        next.push_tool_result(&call, "sunny");
        provider.chat(&next).await.unwrap();
        let sent = &bodies(&transport)[2];
        assert_eq!(sent["tool_choice"], json!("required"));
        assert_eq!(sent["tools"][0]["name"], json!("weather"));
        assert_eq!(sent["messages"][1]["content"][0]["id"], json!("call_1"));
        assert_eq!(sent["messages"][2]["role"], json!("tool"));
        assert_eq!(sent["messages"][2]["tool_call_id"], json!("call_1"));
    }

    #[tokio::test]
    async fn options_reach_the_wire() {
        let transport = ReplayTransport::new([ReplayResponse::json(StatusCode::OK, OK_BODY)]);
        let request = ChatRequest::new(MODEL).user("Hi!").with_option(
            ChatRequestOptions::new()
                .safety_mode("strict")
                .extra("seed_offset", 3),
        );
        provider(&transport).chat(&request).await.unwrap();
        assert_eq!(bodies(&transport)[0]["safety_mode"], json!("strict"));
        assert_eq!(bodies(&transport)[0]["seed_offset"], json!(3));

        // `extra` cannot send a field the adapter owns.
        let owned = ChatRequest::new(MODEL)
            .user("Hi!")
            .with_option(ChatRequestOptions::new().extra("stream", true));
        let err = provider(&transport).chat(&owned).await.unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");

        let typo = ChatRequest::new(MODEL)
            .user("Hi!")
            .with_options_json(NAMESPACE, json!({"safety": "strict"}));
        assert!(provider(&transport).chat(&typo).await.is_err());
    }

    // ── Required: refusals before sending, stream order and stream ends (guide §5) ──

    #[tokio::test]
    async fn too_many_stop_sequences_fail_under_every_policy() {
        let transport = ReplayTransport::new([]);
        let stop: Vec<String> = (0..5).map(|i| i.to_string()).collect();
        let mut request = ChatRequest::new(MODEL).user("Hi!");
        request.stop = stop;
        let error = provider(&transport).chat(&request).await.unwrap_err();
        assert!(matches!(error, Error::Unsupported { .. }), "{error:?}");
        assert!(transport.requests().is_empty());
    }

    /// Events of a stream, errors as their kind.
    async fn events(
        sse: &str,
        request: &ChatRequest,
    ) -> Vec<std::result::Result<StreamEvent, String>> {
        let transport = ReplayTransport::new([ReplayResponse::sse(sse)]);
        let stream = provider(&transport).chat_stream(request).await.unwrap();
        stream
            .map(|item| item.map_err(|error| error.kind_name().to_owned()))
            .collect()
            .await
    }

    #[tokio::test]
    async fn the_stream_starts_before_its_warnings_and_vendor_frames() {
        let sse = format!("event: x-progress\ndata: {{\"step\":1}}\n\n{OK_STREAM}");
        let request = ChatRequest::new(MODEL).user("Hi!").seed(7);
        let events = events(&sse, &request).await;
        assert!(
            matches!(&events[0], Ok(StreamEvent::ResponseStart { id: Some(id), .. }) if id == "resp_1"),
            "{events:?}"
        );
        // The request warning, then the held vendor frame (its warning and data).
        assert!(
            matches!(&events[1], Ok(StreamEvent::Warning { warning, .. })
            if warning.subject == Subject::Field { field: RequestField::Seed })
        );
        assert!(
            matches!(&events[2], Ok(StreamEvent::Warning { warning, .. })
            if warning.kind == WarningKind::Other && warning.subject == Subject::Response)
        );
        assert!(matches!(
            &events[3],
            Ok(StreamEvent::ResponseProviderData { .. })
        ));
        assert!(matches!(events.last(), Some(Ok(StreamEvent::ResponseStop))));
    }

    #[tokio::test]
    async fn an_in_band_error_may_precede_the_start() {
        let sse = "event: error\ndata: {\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}\n\n";
        let events = events(sse, &ChatRequest::new(MODEL).user("Hi!")).await;
        assert_eq!(events, vec![Err("overloaded".to_owned())]);
    }

    #[tokio::test]
    async fn a_truncated_stream_fails() {
        let sse = OK_STREAM.split("event: done").next().unwrap();
        let events = events(sse, &ChatRequest::new(MODEL).user("Hi!")).await;
        assert!(
            matches!(events.last(), Some(Err(kind)) if kind == "stream"),
            "{events:?}"
        );
    }

    /// The end of the body ends the stream even when the decoder's `end`
    /// yields nothing and it never saw a terminal event: a hand-written
    /// loop that recomputed `done` from the decoder alone spun on the ended
    /// body. The shared driver runs `end` once and ends.
    #[tokio::test]
    async fn a_body_that_ends_ends_the_stream_whatever_the_decoder_yields() {
        struct Silent {
            ended: bool,
        }
        impl ChatFrameDecoder<SseEvent> for Silent {
            fn frame(&mut self, event: SseEvent) -> Vec<Result<StreamEvent>> {
                vec![Ok(StreamEvent::text_delta(0, event.data))]
            }
            fn end(&mut self) -> Vec<Result<StreamEvent>> {
                assert!(!self.ended, "`end` ran twice: the stream did not end");
                self.ended = true;
                Vec::new()
            }
        }

        let body =
            HttpResponse::from_bytes(StatusCode::OK, HeaderMap::new(), "data: a\n\ndata: b\n\n");
        let events: Vec<_> =
            SseChatStream::new(SseStream::new(body.into_body()), Silent { ended: false })
                .collect()
                .await;
        assert_eq!(events.len(), 2, "{events:?}");
    }

    /// A failed read keeps its kind: a stream timeout is a timeout, with
    /// the provider and the request id, and nothing follows it.
    #[tokio::test]
    async fn a_stream_timeout_stays_a_timeout() {
        let first = OK_STREAM.split_inclusive("\n\n").next().unwrap().to_owned();
        let timeout = allama_http::TransportErrorKind::Timeout;
        let body: allama_http::Body = Box::pin(futures_util::stream::iter([
            Ok(first.clone().into()),
            Err(TransportError::new(timeout, "idle for 300s").authored()),
            Ok(first.into()),
        ]));
        let decoder = StreamDecoder::new(Some("req_1".into()));
        let items: Vec<_> = SseChatStream::new(SseStream::new(body), decoder)
            .collect()
            .await;
        let error = items.last().unwrap().as_ref().unwrap_err();
        assert!(matches!(error, Error::Timeout { .. }), "{error:?}");
        assert_eq!(error.request_id(), Some("req_1"));
        assert_eq!(error.message(), "idle for 300s");
        assert_eq!(items.iter().filter(|item| item.is_err()).count(), 1);
    }

    #[tokio::test]
    async fn invalid_utf8_is_a_decode_error_not_retried() {
        let first = OK_STREAM.split_inclusive("\n\n").next().unwrap();
        let mut body = first.as_bytes().to_vec();
        body.extend_from_slice(b"event: delta\ndata: {\"text\":\"caf\xC3\"}\n\n");
        let response = ReplayResponse::new(StatusCode::OK, body).header(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        );
        let transport = ReplayTransport::new([response]);
        let mut stream = provider(&transport)
            .chat_stream(&ChatRequest::new(MODEL).user("Hi!"))
            .await
            .unwrap();
        let error = loop {
            match stream.next().await.expect("the stream fails") {
                Ok(_) => {}
                Err(error) => break error,
            }
        };
        assert!(
            matches!(
                &error,
                Error::Stream {
                    kind: StreamErrorKind::Decode,
                    ..
                }
            ),
            "{error:?}"
        );
        assert!(!error.is_retryable(), "{error:?}");
        assert!(error.as_log().to_string().contains("UTF-8"), "{error:?}");
    }

    // ── Only if your API streams in another framing: NDJSON lines (guide §5) ──

    /// The NDJSON stream collects to the response the SSE stream does,
    /// however the network splits its lines.
    #[tokio::test]
    async fn the_ndjson_stream_collects_to_the_sse_response() {
        let sse = HttpResponse::from_bytes(StatusCode::OK, HeaderMap::new(), OK_STREAM);
        let expected =
            SseChatStream::new(SseStream::new(sse.into_body()), StreamDecoder::new(None))
                .collect_response()
                .await
                .unwrap();
        for size in [1, 7, OK_NDJSON.len()] {
            let transport = ReplayTransport::new([
                ReplayResponse::new(StatusCode::OK, OK_NDJSON).split_every(size)
            ]);
            let response = transport
                .send(HttpRequest::get(Uri::from_static("https://acme.test")))
                .await
                .unwrap();
            let collected = ndjson_stream(response, 1 << 20, None)
                .collect_response()
                .await
                .unwrap();
            assert_eq!(collected, expected, "chunks of {size}");
        }
    }

    /// Contract C1 in NDJSON: an unknown event, written as Acme writes a
    /// line, is surfaced, and `CRLF` line endings change nothing.
    #[tokio::test]
    async fn ndjson_vendor_frames_are_surfaced() {
        let ndjson =
            StreamFraming::ndjson(|event, data| json!({"event": event, "data": data}).to_string());
        assert_unknown_stream_frames_surfaced(OK_NDJSON, &ndjson, |body| async move {
            let response = HttpResponse::from_bytes(StatusCode::OK, HeaderMap::new(), body);
            // `Prepared::finish_stream` holds what arrives before the start.
            Ok(allama::adapter::chat::start_first(ndjson_stream(
                response,
                1 << 20,
                None,
            )))
        })
        .await;
    }

    // ── Required: a missing key is a configuration error (guide §2) ──

    #[test]
    fn a_missing_key_is_a_configuration_error() {
        let error = AcmeProvider::builder().build().err().unwrap();
        assert!(matches!(error, Error::Configuration { .. }), "{error:?}");
    }

    // ── Not part of an adapter: checks of the guide itself ──

    /// The provider guide's wrapper section: add context to an error you
    /// pass on without re-wrapping it.
    #[test]
    fn a_wrapper_prefixes_errors_it_passes_on() {
        use allama::Error;

        let err = Error::rate_limited("HTTP 429").authored().with_status(429);
        let err = err.with_message_prefix("summarize step");
        assert!(matches!(err, Error::RateLimited { .. }));
        assert_eq!(err.status(), Some(429));
        assert_eq!(err.message(), "summarize step: HTTP 429");
    }

    /// Every code block of the provider guide quotes this file, line for
    /// line (indentation aside).
    #[test]
    fn the_provider_guide_quotes_this_file() {
        let guide = include_str!("../../../docs/guides/provider-authors.md");
        let source: Vec<&str> = include_str!("native_adapter.rs")
            .lines()
            .map(str::trim)
            .collect();
        // Every block opened with ```rust (indented in a list or not), up to
        // its closing fence.
        let mut quotes = Vec::new();
        let mut lines = guide.lines().map(str::trim);
        while let Some(line) = lines.next() {
            if line == "```rust" {
                quotes.push(
                    lines
                        .by_ref()
                        .take_while(|line| *line != "```")
                        .collect::<Vec<_>>(),
                );
            }
        }
        let mut count = 0;
        for quote in quotes {
            count += 1;
            assert!(
                source.windows(quote.len()).any(|window| window == quote),
                "the provider guide quotes code that is not in native_adapter.rs:\n{}",
                quote.join("\n")
            );
        }
        assert!(
            count > 0,
            "the provider guide quotes no code from native_adapter.rs"
        );
    }
}
