//! Test doubles for chat providers.
//!
//! - [`MockProvider`]: scripted responses, recorded requests.
//! - [`MockStreamingProvider`]: scripted event streams.
//! - [`FunctionProvider`]: responses computed by a closure over the request.
//! - [`TestProvider`]: answers every request automatically, calling the first
//!   tool with schema-valid arguments when tools are present, and answering
//!   a JSON Schema response format with a schema-valid value.
//! - [`ChatResponseBuilder`]: concise response construction.
//!
//! Like a real adapter, [`MockProvider`], [`MockStreamingProvider`], and
//! [`FunctionProvider`] check requests with [`chat::prepare`](crate::adapter::chat::prepare)
//! against the capabilities they report: a request using a semantic feature
//! reported `Unsupported` fails before a response is consumed, and
//! unsupported hints come back as warnings on the response (or stream,
//! after its `ResponseStart`), emitted as warning events. They record or
//! receive the request as the caller sent it. They encode nothing, so their
//! validation keeps the default
//! [`Capabilities`](crate::ValidationCoverage::Capabilities) coverage.

use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, ChatStream,
    ChatStreamExt, ContentBlock, Error, ExtraMap, FinishReason, Message, OtherContent,
    ProviderData, ProviderIdentity, ResponseFormat, Result, SingleResponseStream, StreamEvent,
    ToolCall, ToolChoice, Usage, Warning,
};

type Capabilities = HashMap<ChatCapability, CapabilitySupport>;

fn capability(map: &Capabilities, capability: ChatCapability) -> CapabilitySupport {
    map.get(&capability).copied().unwrap_or_default()
}

/// A provider that returns scripted responses and records requests.
///
/// Each call to [`chat`](ChatProvider::chat) or
/// [`chat_stream`](ChatProvider::chat_stream) takes the next scripted
/// response. A scripted error fails the call itself, so `chat_stream` returns
/// `Err` before any stream exists. A call past the last scripted response
/// returns [`Error::Configuration`].
#[derive(Debug, Clone)]
pub struct MockProvider {
    state: Arc<Mutex<MockChatState>>,
    chat_capabilities: Capabilities,
    provider_name: Cow<'static, str>,
}

#[derive(Debug, Default)]
struct MockChatState {
    responses: VecDeque<MockResponse>,
    requests: Vec<ChatRequest>,
}

impl MockProvider {
    /// Build a provider with the builder.
    #[must_use]
    pub fn build(configure: impl FnOnce(MockProviderBuilder) -> MockProviderBuilder) -> Self {
        configure(MockProviderBuilder::new()).build()
    }

    /// A provider with no scripted responses.
    #[must_use]
    pub fn empty() -> Self {
        Self::new(std::iter::empty::<MockResponse>())
    }

    /// A provider returning `responses` in order.
    #[must_use]
    pub fn new<I, R>(responses: I) -> Self
    where
        I: IntoIterator<Item = R>,
        R: Into<MockResponse>,
    {
        Self {
            state: Arc::new(Mutex::new(MockChatState {
                responses: responses.into_iter().map(Into::into).collect(),
                requests: Vec::new(),
            })),
            chat_capabilities: Capabilities::new(),
            provider_name: Cow::Borrowed("mock"),
        }
    }

    /// A provider returning one response.
    #[must_use]
    pub fn with_response(response: ChatResponse) -> Self {
        Self::new([response])
    }

    /// A provider returning one error.
    ///
    /// The error fails the call itself: `chat` and `chat_stream` both return
    /// it, the way an HTTP provider reports a rejected request.
    #[must_use]
    pub fn with_error(error: Error) -> Self {
        Self::new([MockResponse::Error(error)])
    }

    /// A provider returning one text response.
    #[must_use]
    pub fn with_text(text: impl Into<String>) -> Self {
        Self::new([MockResponse::text(text)])
    }

    /// A provider returning one tool call.
    #[must_use]
    pub fn with_tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self::new([MockResponse::tool_call(id, name, arguments)])
    }

    /// A provider scripted for one tool round trip: a tool call, then text.
    #[must_use]
    pub fn tool_round_trip(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
        final_text: impl Into<String>,
    ) -> Self {
        Self::new([
            MockResponse::tool_call(id, name, arguments),
            MockResponse::text(final_text),
        ])
    }

    /// Override a capability answer.
    #[must_use]
    pub fn with_chat_capability(
        mut self,
        capability: ChatCapability,
        support: CapabilitySupport,
    ) -> Self {
        self.chat_capabilities.insert(capability, support);
        self
    }

    /// Mark capabilities as supported.
    #[must_use]
    pub fn with_supported_chat_capabilities<I>(mut self, capabilities: I) -> Self
    where
        I: IntoIterator<Item = ChatCapability>,
    {
        for capability in capabilities {
            self.chat_capabilities
                .insert(capability, CapabilitySupport::Supported);
        }
        self
    }

    /// Set the provider name.
    #[must_use]
    pub fn with_provider_name(mut self, provider_name: impl Into<Cow<'static, str>>) -> Self {
        self.provider_name = provider_name.into();
        self
    }

    /// Append a scripted response.
    pub fn push_response(&self, response: impl Into<MockResponse>) {
        self.state
            .lock()
            .unwrap()
            .responses
            .push_back(response.into());
    }

    /// All recorded requests.
    #[must_use]
    pub fn requests(&self) -> Vec<ChatRequest> {
        self.state.lock().unwrap().requests.clone()
    }

    /// The last recorded request.
    #[must_use]
    pub fn last_request(&self) -> Option<ChatRequest> {
        self.state.lock().unwrap().requests.last().cloned()
    }

    /// Number of calls made.
    #[must_use]
    pub fn call_count(&self) -> usize {
        self.state.lock().unwrap().requests.len()
    }

    /// Number of scripted responses left.
    #[cfg(test)]
    pub(crate) fn pending_responses(&self) -> usize {
        self.state.lock().unwrap().responses.len()
    }
}

impl ProviderIdentity for MockProvider {
    fn provider_name(&self) -> &str {
        &self.provider_name
    }

    /// Reads the namespace named like the mock, so a mock stands in for a
    /// provider whose namespace is its name.
    fn reads_namespace(&self, namespace: &str) -> bool {
        namespace == self.provider_name
    }

    fn downcast_any(&self, id: std::any::TypeId) -> Option<&(dyn std::any::Any + Send + Sync)> {
        crate::expose(self, id)
    }
}

impl ChatProvider for MockProvider {
    type Stream = SingleResponseStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let mut prepared = crate::adapter::chat::prepare(self, request)?;
        let response = {
            let mut state = self.state.lock().unwrap();
            state.requests.push(request.clone());
            state.responses.pop_front()
        };
        let Some(response) = response else {
            return Err(
                Error::configuration("MockProvider: no more responses configured").authored(),
            );
        };
        prepared.mark_sent();
        Ok(prepared.finish(response.into_result(request).await?))
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
        Ok(self.chat(request).await?.into())
    }

    fn chat_capability(&self, _model: &str, capability_: ChatCapability) -> CapabilitySupport {
        capability(&self.chat_capabilities, capability_)
    }
}

/// A scripted response.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
#[non_exhaustive]
pub enum MockResponse {
    /// A successful response.
    Success(ChatResponse),
    /// An error.
    Error(Error),
    /// A response delivered after a delay.
    Delayed(Duration, Box<MockResponse>),
    /// A structured value, in the form the request asks for it; see
    /// [`MockResponse::structured`].
    Structured(serde_json::Value),
}

impl MockResponse {
    /// A text response.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        ChatResponseBuilder::new().text(text).build().into()
    }

    /// A reasoning-then-text response.
    #[must_use]
    pub fn reasoning_text(reasoning: impl Into<String>, text: impl Into<String>) -> Self {
        ChatResponseBuilder::new()
            .reasoning(reasoning)
            .text(text)
            .build()
            .into()
    }

    /// A tool-call response.
    #[must_use]
    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        ChatResponseBuilder::new()
            .tool_call(id, name, arguments)
            .build()
            .into()
    }

    /// A response with several tool calls in one turn, in order (parallel
    /// tool calls).
    ///
    /// ```
    /// use allama::ToolCall;
    /// use allama::mock::MockResponse;
    ///
    /// let response = MockResponse::tool_calls([
    ///     ToolCall::new("call_1", "weather", r#"{"city":"Paris"}"#),
    ///     ToolCall::new("call_2", "weather", r#"{"city":"Rome"}"#),
    /// ]);
    /// # let _ = response;
    /// ```
    #[must_use]
    pub fn tool_calls(calls: impl IntoIterator<Item = ToolCall>) -> Self {
        ChatResponseBuilder::new().tool_calls(calls).build().into()
    }

    /// A structured value, in the form the request asks for it: "the model
    /// returns this value".
    ///
    /// When the request forces a call to one tool
    /// ([`ToolChoice::Specific`], or [`ToolChoice::Required`] with exactly
    /// one tool), the response calls that tool with `value` as its
    /// arguments. Otherwise the response is `value` as JSON text. So a test
    /// of an [extraction] scripts the value without knowing
    /// which mode the extraction picks: forced-tool mode gets a call to its
    /// synthetic tool, native mode gets JSON text.
    ///
    /// ```
    /// # #[cfg(feature = "chat-extract")]
    /// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
    /// use allama::chat::extract::{ChatExtractExt, ChatExtractionMode};
    /// use allama::mock::{MockProvider, MockResponse};
    /// use allama::schemars::{self, JsonSchema};
    /// use allama::ChatRequest;
    /// use serde_json::json;
    ///
    /// #[derive(serde::Deserialize, JsonSchema)]
    /// struct Person {
    ///     name: String,
    /// }
    ///
    /// for mode in [ChatExtractionMode::Native, ChatExtractionMode::ForcedTool] {
    ///     let provider = MockProvider::new([MockResponse::structured(json!({"name": "Ada"}))]);
    ///     let request = ChatRequest::new("m").user("Ada wrote it.").with_option(mode);
    ///     let extracted = provider.extract::<Person>(&request).await?;
    ///     assert_eq!(extracted.value.name, "Ada");
    /// }
    /// # Ok::<(), allama::Error>(()) }).unwrap();
    /// ```
    #[cfg_attr(feature = "chat-extract", doc = "[extraction]: crate::chat::extract")]
    #[cfg_attr(
        not(feature = "chat-extract"),
        doc = "[extraction]: crate#cargo-features"
    )]
    #[must_use]
    pub fn structured(value: serde_json::Value) -> Self {
        Self::Structured(value)
    }

    /// Deliver `response` after `duration`.
    #[must_use]
    pub fn delayed(duration: Duration, response: impl Into<MockResponse>) -> Self {
        Self::Delayed(duration, Box::new(response.into()))
    }

    async fn into_result(self, request: &ChatRequest) -> Result<ChatResponse> {
        let mut next = self;
        loop {
            match next {
                Self::Success(response) => return Ok(response),
                Self::Error(error) => return Err(error),
                Self::Delayed(duration, response) => {
                    futures_timer::Delay::new(duration).await;
                    next = *response;
                }
                Self::Structured(value) => {
                    let builder = ChatResponseBuilder::new();
                    let builder = match forced_tool(request) {
                        Some(name) => builder.tool_call("call_structured", name, value),
                        None => builder.text(value.to_string()),
                    };
                    return Ok(builder.build());
                }
            }
        }
    }
}

impl From<ChatResponse> for MockResponse {
    fn from(response: ChatResponse) -> Self {
        Self::Success(response)
    }
}

impl From<Error> for MockResponse {
    fn from(error: Error) -> Self {
        Self::Error(error)
    }
}

impl From<Result<ChatResponse>> for MockResponse {
    fn from(result: Result<ChatResponse>) -> Self {
        match result {
            Ok(response) => Self::Success(response),
            Err(error) => Self::Error(error),
        }
    }
}

/// Builder for [`MockProvider`].
#[derive(Debug)]
pub struct MockProviderBuilder {
    responses: Vec<MockResponse>,
    chat_capabilities: Capabilities,
    provider_name: Cow<'static, str>,
}

impl Default for MockProviderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MockProviderBuilder {
    /// An empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            responses: Vec::new(),
            chat_capabilities: Capabilities::new(),
            provider_name: Cow::Borrowed("mock"),
        }
    }

    /// Append a response.
    #[must_use]
    pub fn response(mut self, response: impl Into<MockResponse>) -> Self {
        self.responses.push(response.into());
        self
    }

    /// Append a text response.
    #[must_use]
    pub fn text(self, text: impl Into<String>) -> Self {
        self.response(MockResponse::text(text))
    }

    /// Append an error.
    #[must_use]
    pub fn error(self, error: Error) -> Self {
        self.response(error)
    }

    /// Append a tool-call response.
    #[must_use]
    pub fn tool_call(
        self,
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        self.response(MockResponse::tool_call(id, name, arguments))
    }

    /// Append a response with several tool calls in one turn; see
    /// [`MockResponse::tool_calls`].
    #[must_use]
    pub fn tool_calls(self, calls: impl IntoIterator<Item = ToolCall>) -> Self {
        self.response(MockResponse::tool_calls(calls))
    }

    /// Append a structured value, in the form the request asks for it; see
    /// [`MockResponse::structured`].
    #[must_use]
    pub fn structured(self, value: serde_json::Value) -> Self {
        self.response(MockResponse::structured(value))
    }

    /// Override a capability answer.
    #[must_use]
    pub fn chat_capability(
        mut self,
        capability: ChatCapability,
        support: CapabilitySupport,
    ) -> Self {
        self.chat_capabilities.insert(capability, support);
        self
    }

    /// Set the provider name.
    #[must_use]
    pub fn provider_name(mut self, provider_name: impl Into<Cow<'static, str>>) -> Self {
        self.provider_name = provider_name.into();
        self
    }

    /// Build the provider.
    #[must_use]
    pub fn build(self) -> MockProvider {
        let mut provider = MockProvider::new(self.responses).with_provider_name(self.provider_name);
        provider.chat_capabilities = self.chat_capabilities;
        provider
    }
}

/// A provider that returns scripted event streams.
///
/// Each call to [`chat_stream`](ChatProvider::chat_stream) takes the next
/// scripted entry, and errors surface in one of two places:
///
/// - An error scripted *inside* a stream ([`MockStreamEvent::Error`]) arrives
///   as a stream item, after `chat_stream` has returned `Ok`, like a
///   connection that drops mid-response.
/// - A [`setup_error`](MockStreamingProviderBuilder::setup_error) entry makes
///   `chat_stream` itself return `Err`, like a 429 that an HTTP provider
///   reports before any stream exists.
///
/// A call past the last scripted entry returns [`Error::Configuration`].
#[derive(Debug, Clone)]
pub struct MockStreamingProvider {
    state: Arc<Mutex<MockStreamingState>>,
    chat_capabilities: Capabilities,
    provider_name: Cow<'static, str>,
}

#[derive(Debug, Default)]
struct MockStreamingState {
    streams: VecDeque<ScriptedStream>,
    requests: Vec<ChatRequest>,
}

fn streaming_capabilities() -> Capabilities {
    HashMap::from([
        (ChatCapability::Streaming, CapabilitySupport::Supported),
        (
            ChatCapability::NativeStreaming,
            CapabilitySupport::Supported,
        ),
    ])
}

impl MockStreamingProvider {
    /// Build a provider with the builder.
    #[must_use]
    pub fn build(
        configure: impl FnOnce(MockStreamingProviderBuilder) -> MockStreamingProviderBuilder,
    ) -> Self {
        configure(MockStreamingProviderBuilder::new()).build()
    }

    /// A provider returning `streams` in order.
    #[must_use]
    pub fn new<I, S, E>(streams: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: IntoIterator<Item = E>,
        E: Into<MockStreamEvent>,
    {
        Self::scripted(streams.into_iter().map(ScriptedStream::events))
    }

    fn scripted(entries: impl IntoIterator<Item = ScriptedStream>) -> Self {
        Self {
            state: Arc::new(Mutex::new(MockStreamingState {
                streams: entries.into_iter().collect(),
                requests: Vec::new(),
            })),
            chat_capabilities: streaming_capabilities(),
            provider_name: Cow::Borrowed("mock_stream"),
        }
    }

    /// A provider returning one stream.
    #[must_use]
    pub fn with_stream<S, E>(stream: S) -> Self
    where
        S: IntoIterator<Item = E>,
        E: Into<MockStreamEvent>,
    {
        Self::new([stream])
    }

    /// A provider streaming one text response.
    #[must_use]
    pub fn with_text(text: impl Into<String>) -> Self {
        Self::with_stream(MockStreamEvent::text_response(text))
    }

    /// A provider streaming each response in order.
    #[must_use]
    pub fn from_responses<I: IntoIterator<Item = ChatResponse>>(responses: I) -> Self {
        Self::new(responses.into_iter().map(MockStreamEvent::from_response))
    }

    /// Override a capability answer.
    #[must_use]
    pub fn with_chat_capability(
        mut self,
        capability: ChatCapability,
        support: CapabilitySupport,
    ) -> Self {
        self.chat_capabilities.insert(capability, support);
        self
    }

    /// Set the provider name.
    #[must_use]
    pub fn with_provider_name(mut self, provider_name: impl Into<Cow<'static, str>>) -> Self {
        self.provider_name = provider_name.into();
        self
    }

    /// All recorded requests.
    #[must_use]
    pub fn requests(&self) -> Vec<ChatRequest> {
        self.state.lock().unwrap().requests.clone()
    }

    /// Number of calls made.
    #[must_use]
    pub fn call_count(&self) -> usize {
        self.state.lock().unwrap().requests.len()
    }
}

impl ProviderIdentity for MockStreamingProvider {
    fn provider_name(&self) -> &str {
        &self.provider_name
    }

    /// Reads the namespace named like the mock, so a mock stands in for a
    /// provider whose namespace is its name.
    fn reads_namespace(&self, namespace: &str) -> bool {
        namespace == self.provider_name
    }

    fn downcast_any(&self, id: std::any::TypeId) -> Option<&(dyn std::any::Any + Send + Sync)> {
        crate::expose(self, id)
    }
}

impl ChatProvider for MockStreamingProvider {
    type Stream = ChatStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        self.chat_stream(request).await?.collect_response().await
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
        let mut prepared = crate::adapter::chat::prepare(self, request)?;
        let entry = {
            let mut state = self.state.lock().unwrap();
            state.requests.push(request.clone());
            state.streams.pop_front()
        };
        let pending = match entry {
            Some(ScriptedStream::Events(events)) => events,
            Some(ScriptedStream::SetupError(error)) => return Err(error),
            None => {
                return Err(Error::configuration(
                    "MockStreamingProvider: no more streams configured",
                )
                .authored());
            }
        };
        prepared.mark_sent();
        let stream = futures_util::stream::unfold(pending, |mut pending| async move {
            loop {
                match pending.pop_front()? {
                    MockStreamEvent::Delay(duration) => futures_timer::Delay::new(duration).await,
                    MockStreamEvent::Event(event) => return Some((Ok(event), pending)),
                    MockStreamEvent::Error(error) => return Some((Err(error), pending)),
                }
            }
        });
        Ok(prepared.finish_stream(stream))
    }

    fn chat_capability(&self, _model: &str, capability_: ChatCapability) -> CapabilitySupport {
        capability(&self.chat_capabilities, capability_)
    }
}

/// One scripted `chat_stream` call: a stream to replay, or a setup failure.
#[derive(Debug)]
enum ScriptedStream {
    Events(VecDeque<MockStreamEvent>),
    SetupError(Error),
}

impl ScriptedStream {
    fn events<S, E>(stream: S) -> Self
    where
        S: IntoIterator<Item = E>,
        E: Into<MockStreamEvent>,
    {
        Self::Events(stream.into_iter().map(Into::into).collect())
    }
}

/// One scripted stream item.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
#[non_exhaustive]
pub enum MockStreamEvent {
    /// An event.
    Event(StreamEvent),
    /// An error.
    Error(Error),
    /// A pause before the next item.
    Delay(Duration),
}

impl MockStreamEvent {
    /// A pause.
    #[must_use]
    pub fn delayed(duration: Duration) -> Self {
        Self::Delay(duration)
    }

    /// The events of a complete response.
    #[must_use]
    pub fn from_response(response: ChatResponse) -> Vec<Self> {
        response.into_stream_events().map(Self::Event).collect()
    }

    /// The events of a text response.
    #[must_use]
    pub fn text_response(text: impl Into<String>) -> Vec<Self> {
        Self::from_response(ChatResponseBuilder::new().text(text).build())
    }

    /// The events of a tool-call response.
    #[must_use]
    pub fn tool_call_response(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Vec<Self> {
        Self::from_response(
            ChatResponseBuilder::new()
                .tool_call(id, name, arguments)
                .build(),
        )
    }
}

impl From<StreamEvent> for MockStreamEvent {
    fn from(event: StreamEvent) -> Self {
        Self::Event(event)
    }
}

impl From<Error> for MockStreamEvent {
    fn from(error: Error) -> Self {
        Self::Error(error)
    }
}

impl From<Result<StreamEvent>> for MockStreamEvent {
    fn from(result: Result<StreamEvent>) -> Self {
        result.map_or_else(Self::Error, Self::Event)
    }
}

/// Builder for [`MockStreamingProvider`].
#[derive(Debug)]
pub struct MockStreamingProviderBuilder {
    streams: Vec<ScriptedStream>,
    chat_capabilities: Capabilities,
    provider_name: Cow<'static, str>,
}

impl Default for MockStreamingProviderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MockStreamingProviderBuilder {
    /// An empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            streams: Vec::new(),
            chat_capabilities: streaming_capabilities(),
            provider_name: Cow::Borrowed("mock_stream"),
        }
    }

    /// Append a scripted stream.
    #[must_use]
    pub fn stream<S, E>(mut self, stream: S) -> Self
    where
        S: IntoIterator<Item = E>,
        E: Into<MockStreamEvent>,
    {
        self.streams.push(ScriptedStream::events(stream));
        self
    }

    /// Append a setup failure: that `chat_stream` call returns `error`
    /// itself, before any stream exists.
    ///
    /// This is how an HTTP provider reports a rejected request, such as a
    /// 429 or a 401. An error scripted inside a [`stream`](Self::stream)
    /// instead arrives as a stream item after `chat_stream` returns `Ok`.
    /// Setup failures and streams can be interleaved; the next entry streams
    /// normally.
    ///
    /// ```
    /// use allama::mock::MockStreamingProvider;
    /// use allama::prelude::*;
    ///
    /// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
    /// let provider = MockStreamingProvider::build(|b| {
    ///     b.setup_error(Error::rate_limited("slow down")).text("hello")
    /// });
    /// let request = ChatRequest::new("demo-model").user("Hi");
    ///
    /// assert!(provider.chat_stream(&request).await.is_err());
    /// let response = provider.chat_stream(&request).await?.collect_response().await?;
    /// assert_eq!(response.text().as_deref(), Some("hello"));
    /// # Ok::<(), allama::Error>(()) }).unwrap();
    /// ```
    #[must_use]
    pub fn setup_error(mut self, error: Error) -> Self {
        self.streams.push(ScriptedStream::SetupError(error));
        self
    }

    /// Append a stream replaying `response`.
    #[must_use]
    pub fn response(self, response: ChatResponse) -> Self {
        self.stream(MockStreamEvent::from_response(response))
    }

    /// Append a text stream.
    #[must_use]
    pub fn text(self, text: impl Into<String>) -> Self {
        self.stream(MockStreamEvent::text_response(text))
    }

    /// Append a tool-call stream.
    #[must_use]
    pub fn tool_call(
        self,
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        self.stream(MockStreamEvent::tool_call_response(id, name, arguments))
    }

    /// Append a stream with several tool calls in one turn; see
    /// [`MockResponse::tool_calls`].
    #[must_use]
    pub fn tool_calls(self, calls: impl IntoIterator<Item = ToolCall>) -> Self {
        self.response(ChatResponseBuilder::new().tool_calls(calls).build())
    }

    /// Override a capability answer.
    #[must_use]
    pub fn chat_capability(
        mut self,
        capability: ChatCapability,
        support: CapabilitySupport,
    ) -> Self {
        self.chat_capabilities.insert(capability, support);
        self
    }

    /// Set the provider name.
    #[must_use]
    pub fn provider_name(mut self, provider_name: impl Into<Cow<'static, str>>) -> Self {
        self.provider_name = provider_name.into();
        self
    }

    /// Build the provider.
    #[must_use]
    pub fn build(self) -> MockStreamingProvider {
        let mut provider =
            MockStreamingProvider::scripted(self.streams).with_provider_name(self.provider_name);
        provider.chat_capabilities = self.chat_capabilities;
        provider
    }
}

type ResponseFn = dyn Fn(&ChatRequest) -> Result<ChatResponse> + Send + Sync;

/// A provider whose responses are computed by a closure over the request.
///
/// ```
/// use allama::mock::{ChatResponseBuilder, FunctionProvider};
/// use allama::{ChatProvider, ChatRequest};
///
/// # async fn demo() -> allama::Result<()> {
/// let echo = FunctionProvider::new(|request: &ChatRequest| {
///     Ok(ChatResponseBuilder::new().text(format!("{} messages", request.messages.len())).build())
/// });
/// let response = echo.chat(&ChatRequest::new("m").user("hi")).await?;
/// assert_eq!(response.text().as_deref(), Some("1 messages"));
/// # Ok(()) }
/// ```
#[derive(Clone)]
pub struct FunctionProvider {
    respond: Arc<ResponseFn>,
    chat_capabilities: Capabilities,
    provider_name: Cow<'static, str>,
}

impl std::fmt::Debug for FunctionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionProvider")
            .field("provider_name", &self.provider_name)
            .finish_non_exhaustive()
    }
}

impl FunctionProvider {
    /// Create a provider from a closure.
    #[must_use]
    pub fn new<F>(respond: F) -> Self
    where
        F: Fn(&ChatRequest) -> Result<ChatResponse> + Send + Sync + 'static,
    {
        Self {
            respond: Arc::new(respond),
            chat_capabilities: Capabilities::new(),
            provider_name: Cow::Borrowed("function"),
        }
    }

    /// Override a capability answer.
    #[must_use]
    pub fn with_chat_capability(
        mut self,
        capability: ChatCapability,
        support: CapabilitySupport,
    ) -> Self {
        self.chat_capabilities.insert(capability, support);
        self
    }

    /// Set the provider name.
    #[must_use]
    pub fn with_provider_name(mut self, provider_name: impl Into<Cow<'static, str>>) -> Self {
        self.provider_name = provider_name.into();
        self
    }
}

impl ProviderIdentity for FunctionProvider {
    fn provider_name(&self) -> &str {
        &self.provider_name
    }

    /// Reads the namespace named like the mock, so a mock stands in for a
    /// provider whose namespace is its name.
    fn reads_namespace(&self, namespace: &str) -> bool {
        namespace == self.provider_name
    }

    fn downcast_any(&self, id: std::any::TypeId) -> Option<&(dyn std::any::Any + Send + Sync)> {
        crate::expose(self, id)
    }
}

impl ChatProvider for FunctionProvider {
    type Stream = SingleResponseStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let mut prepared = crate::adapter::chat::prepare(self, request)?;
        prepared.mark_sent();
        Ok(prepared.finish((self.respond)(request)?))
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
        Ok(self.chat(request).await?.into())
    }

    fn chat_capability(&self, _model: &str, capability_: ChatCapability) -> CapabilitySupport {
        capability(&self.chat_capabilities, capability_)
    }
}

/// A provider that answers automatically, for tests that do not care about
/// the exact output.
///
/// When the request has tools and the conversation does not end with a tool
/// result, it calls the first tool (or the one forced by `tool_choice`) with
/// arguments generated from the tool's JSON Schema. Otherwise, when the
/// request asks for a [`ResponseFormat::JsonSchema`], it answers with a
/// value generated from that schema, so `extract` (feature `extract`)
/// works against it in every `ChatExtractionMode`; it reports
/// [`StructuredOutput`](ChatCapability::StructuredOutput) as supported.
/// Otherwise it answers with text describing the conversation.
///
/// The generator covers the common subset of JSON Schema: `type`,
/// `properties` and `required`, `items`, `enum` and `const` (the first
/// value), `$ref`, the first branch of `anyOf`, `oneOf` and `allOf`, and a
/// numeric `minimum`. A string is always `"example"`, so a schema that
/// constrains strings (`pattern`, `format`, `minLength`) may reject the
/// value; script a [`MockProvider`] for those.
///
/// It never runs out of answers, so it is the right mock for tests that send
/// many requests, such as an HTTP handler test that calls the handler more
/// than once. The scripted mocks ([`MockProvider`],
/// [`MockStreamingProvider`]) return [`Error::Configuration`] once their
/// script is used up.
///
/// Every response carries a complete [envelope](ChatResponse::envelope), so
/// code that stores or bills turns can be tested with it:
///
/// - `model` echoes the request's model, and `provider` is `"test"`;
/// - `usage` is deterministic: one input token per whitespace-separated
///   word of the system prompts and message text
///   ([`Message::text_segments`]), and one output token per word of the
///   answer text or the tool call's arguments, each at least 1;
/// - `id` and the tool-call ids are derived from the conversation, not from
///   a per-instance counter, so they stay unique across the turns of a
///   conversation even when every turn builds a new `TestProvider` (as a
///   handler test whose registry builds a provider per request does):
///   `test_response_{n}` for the `n`th assistant turn, and `test_call_{n}`
///   numbered after the tool calls the conversation already holds,
///   skipping any id it already uses. A conversation never reuses a
///   tool-call id, which providers reject and
///   [`history::validate`](crate::history::validate) reports.
///
/// The request id stays unset: no transport is involved.
///
/// ```
/// # async fn example() -> allama::Result<()> {
/// use allama::mock::TestProvider;
/// use allama::{ChatProvider, ChatRequest};
///
/// let provider = TestProvider::new();
/// let response = provider.chat(&ChatRequest::new("demo").user("Say hello")).await?;
/// let envelope = response.envelope();
/// assert_eq!(envelope.model.as_deref(), Some("demo"));
/// assert_eq!(envelope.provider.as_deref(), Some("test"));
/// assert_eq!(envelope.usage.unwrap().input_tokens, Some(2));
/// # Ok(()) }
/// ```
#[derive(Debug, Clone, Default)]
pub struct TestProvider {
    calls: Arc<Mutex<usize>>,
}

impl TestProvider {
    /// Create a test provider.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of calls made.
    #[must_use]
    pub fn call_count(&self) -> usize {
        *self.calls.lock().unwrap()
    }

    fn respond(&self, request: &ChatRequest) -> ChatResponse {
        *self.calls.lock().unwrap() += 1;
        let mut response = self.answer(request);
        let input_tokens = word_count(
            request
                .system
                .iter()
                .map(|prompt| prompt.content.as_str())
                .chain(request.messages.iter().flat_map(Message::text_segments)),
        );
        let output_tokens = word_count(response.content.iter().filter_map(|block| match block {
            ContentBlock::Text(text) => Some(text.text.as_str()),
            ContentBlock::Refusal(refusal) => Some(refusal.text.as_str()),
            ContentBlock::ToolCall(call) => Some(call.arguments.as_str()),
            _ => None,
        }));
        let assistant_turns = request
            .messages
            .iter()
            .filter(|message| matches!(message, Message::Assistant(_)))
            .count();
        response.model = Some(request.model.clone());
        response.provider = Some(Cow::Borrowed(TEST_PROVIDER_NAME));
        response.id = Some(format!("test_response_{}", assistant_turns + 1));
        response.usage = Some(
            Usage::new()
                .input_tokens(input_tokens)
                .output_tokens(output_tokens),
        );
        response
    }

    fn answer(&self, request: &ChatRequest) -> ChatResponse {
        let answered_tool = matches!(request.messages.last(), Some(Message::Tool(_)));
        let forced = match &request.tool_choice {
            Some(ToolChoice::Specific { name }) => Some(name.as_str()),
            _ => None,
        };
        let disabled = matches!(request.tool_choice, Some(ToolChoice::Disabled));
        let tools = &request.tools;
        if !disabled
            && !answered_tool
            && let Some(first) = tools.first()
        {
            let tool = forced
                .and_then(|name| tools.iter().find(|t| t.name == name))
                .unwrap_or(first);
            let arguments = example_value(&tool.parameters, &tool.parameters, 0);
            return ChatResponseBuilder::new()
                .tool_call(next_tool_call_id(request), tool.name.clone(), arguments)
                .build();
        }
        match &request.response_format {
            Some(ResponseFormat::JsonSchema { schema, .. }) => ChatResponseBuilder::new()
                .text(example_value(schema, schema, 0).to_string())
                .build(),
            _ => text_answer(request),
        }
    }
}

const TEST_PROVIDER_NAME: &str = "test";

/// The one tool `request` forces a call to: the [`ToolChoice::Specific`]
/// tool, or the only tool under [`ToolChoice::Required`].
fn forced_tool(request: &ChatRequest) -> Option<&str> {
    match request.tool_choice.as_ref()? {
        ToolChoice::Specific { name } => Some(name),
        ToolChoice::Required => match request.tools.as_slice() {
            [tool] => Some(&tool.name),
            _ => None,
        },
        _ => None,
    }
}

fn text_answer(request: &ChatRequest) -> ChatResponse {
    ChatResponseBuilder::new()
        .text(format!(
            "test response ({} messages)",
            request.messages.len()
        ))
        .build()
}

/// The id of the next tool call in `request`'s conversation:
/// `test_call_{n}`, numbered after the tool calls it already holds and
/// skipping any id it already uses, so it is unique in the conversation.
fn next_tool_call_id(request: &ChatRequest) -> String {
    let used: std::collections::HashSet<&str> = request
        .messages
        .iter()
        .filter_map(Message::as_assistant)
        .flat_map(|message| &message.content)
        .filter_map(|block| match block {
            ContentBlock::ToolCall(call) => Some(call.id.as_str()),
            _ => None,
        })
        .collect();
    (used.len() + 1..)
        .map(|n| format!("test_call_{n}"))
        .find(|id| !used.contains(id.as_str()))
        .expect("an unused id exists")
}

/// One token per whitespace-separated word of `texts`, at least 1.
fn word_count<'a>(texts: impl Iterator<Item = &'a str>) -> u64 {
    let words: usize = texts.map(|text| text.split_whitespace().count()).sum();
    u64::try_from(words).unwrap_or(u64::MAX).max(1)
}

impl ProviderIdentity for TestProvider {
    fn provider_name(&self) -> &str {
        TEST_PROVIDER_NAME
    }

    /// Reads the namespace named like the mock, so a mock stands in for a
    /// provider whose namespace is its name.
    fn reads_namespace(&self, namespace: &str) -> bool {
        namespace == TEST_PROVIDER_NAME
    }

    fn downcast_any(&self, id: std::any::TypeId) -> Option<&(dyn std::any::Any + Send + Sync)> {
        crate::expose(self, id)
    }
}

impl ChatProvider for TestProvider {
    type Stream = SingleResponseStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        Ok(self.respond(request))
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
        Ok(self.respond(request).into())
    }

    fn chat_capability(&self, _model: &str, capability: ChatCapability) -> CapabilitySupport {
        match capability {
            ChatCapability::ToolCalls
            | ChatCapability::Streaming
            | ChatCapability::StructuredOutput => CapabilitySupport::Supported,
            _ => CapabilitySupport::Unknown,
        }
    }
}

/// Generate a value matching a (subset of) JSON Schema.
fn example_value(
    schema: &serde_json::Value,
    root: &serde_json::Value,
    depth: u8,
) -> serde_json::Value {
    use serde_json::{Value, json};
    if depth > 8 {
        return Value::Null;
    }
    let Some(object) = schema.as_object() else {
        return json!({});
    };
    if let Some(reference) = object.get("$ref").and_then(Value::as_str)
        && let Some(path) = reference.strip_prefix("#/")
    {
        let target = path
            .split('/')
            .try_fold(root, |node, key| node.get(key))
            .cloned()
            .unwrap_or(Value::Null);
        return example_value(&target, root, depth + 1);
    }
    if let Some(value) = object.get("const") {
        return value.clone();
    }
    if let Some(first) = object
        .get("enum")
        .and_then(Value::as_array)
        .and_then(|v| v.first())
    {
        return first.clone();
    }
    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(first) = object
            .get(key)
            .and_then(Value::as_array)
            .and_then(|v| v.first())
        {
            return example_value(first, root, depth + 1);
        }
    }
    let ty = match object.get("type") {
        Some(Value::String(t)) => t.as_str(),
        Some(Value::Array(types)) => types
            .iter()
            .filter_map(Value::as_str)
            .find(|t| *t != "null")
            .unwrap_or("null"),
        _ if object.contains_key("properties") => "object",
        _ => "object",
    };
    match ty {
        "string" => json!("example"),
        "integer" => json!(object.get("minimum").and_then(Value::as_i64).unwrap_or(1)),
        "number" => json!(object.get("minimum").and_then(Value::as_f64).unwrap_or(1.0)),
        "boolean" => json!(true),
        "null" => Value::Null,
        "array" => {
            let item = object
                .get("items")
                .map_or(Value::Null, |items| example_value(items, root, depth + 1));
            json!([item])
        }
        _ => {
            let mut out = serde_json::Map::new();
            if let Some(properties) = object.get("properties").and_then(Value::as_object) {
                let required: Vec<&str> = object
                    .get("required")
                    .and_then(Value::as_array)
                    .map(|r| r.iter().filter_map(Value::as_str).collect())
                    .unwrap_or_default();
                for (name, property) in properties {
                    if required.is_empty() || required.contains(&name.as_str()) {
                        out.insert(name.clone(), example_value(property, root, depth + 1));
                    }
                }
            }
            Value::Object(out)
        }
    }
}

/// Concise construction of [`ChatResponse`] values for tests.
#[derive(Debug, Clone)]
pub struct ChatResponseBuilder {
    response: ChatResponse,
    finish_reason_set: bool,
}

impl Default for ChatResponseBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ChatResponseBuilder {
    /// A builder whose finish reason defaults to `Stop` (or `ToolCalls` when
    /// tool calls are added).
    #[must_use]
    pub fn new() -> Self {
        Self {
            response: ChatResponse::default(),
            finish_reason_set: false,
        }
    }

    /// Append a text block.
    #[must_use]
    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.response.content.push(ContentBlock::text(text));
        self
    }

    /// Append a tool-call block.
    #[must_use]
    pub fn tool_call(
        mut self,
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        self.response
            .content
            .push(ContentBlock::ToolCall(ToolCall::new(
                id,
                name,
                arguments.to_string(),
            )));
        self
    }

    /// Append a tool-call block for each call, in order.
    #[must_use]
    pub fn tool_calls(mut self, calls: impl IntoIterator<Item = ToolCall>) -> Self {
        self.response
            .content
            .extend(calls.into_iter().map(ContentBlock::ToolCall));
        self
    }

    /// Append a reasoning block.
    #[must_use]
    pub fn reasoning(mut self, text: impl Into<String>) -> Self {
        self.response.content.push(ContentBlock::reasoning(text));
        self
    }

    /// Append a refusal block: text in which the model declined to answer.
    ///
    /// The finish reason is left as it is; a provider that refuses usually
    /// also reports [`FinishReason::Refusal`]:
    ///
    /// ```
    /// use allama::FinishReason;
    /// use allama::mock::ChatResponseBuilder;
    ///
    /// let response = ChatResponseBuilder::new()
    ///     .refusal("I can't help with that.")
    ///     .finish_reason(FinishReason::Refusal)
    ///     .build();
    /// assert!(response.is_refusal());
    /// assert_eq!(response.first_refusal(), Some("I can't help with that."));
    /// ```
    #[must_use]
    pub fn refusal(mut self, text: impl Into<String>) -> Self {
        self.response.content.push(ContentBlock::refusal(text));
        self
    }

    /// Append an `Other` block.
    #[must_use]
    pub fn other(mut self, type_name: impl Into<String>, data: ExtraMap) -> Self {
        self.response
            .content
            .push(ContentBlock::Other(OtherContent::new(type_name, data)));
        self
    }

    /// Append any block: a [`ContentBlock`], or a content struct that
    /// converts into one ([`Refusal`](crate::Refusal),
    /// [`ToolCall`](crate::ToolCall), …).
    #[must_use]
    pub fn block(mut self, block: impl Into<ContentBlock>) -> Self {
        self.response.content.push(block.into());
        self
    }

    /// Set the finish reason.
    #[must_use]
    pub fn finish_reason(mut self, reason: FinishReason) -> Self {
        self.response.finish_reason = Some(reason);
        self.finish_reason_set = true;
        self
    }

    /// Set input and output token usage.
    #[must_use]
    pub fn usage(mut self, input_tokens: u64, output_tokens: u64) -> Self {
        self.response.usage = Some(
            Usage::new()
                .input_tokens(input_tokens)
                .output_tokens(output_tokens),
        );
        self
    }

    /// Set usage.
    #[must_use]
    pub fn usage_value(mut self, usage: Usage) -> Self {
        self.response.usage = Some(usage);
        self
    }

    /// Set the model.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.response.model = Some(model.into());
        self
    }

    /// Set the response id.
    #[must_use]
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.response.id = Some(id.into());
        self
    }

    /// Append a warning.
    #[must_use]
    pub fn warning(mut self, warning: Warning) -> Self {
        self.response.warnings.push(warning);
        self
    }

    /// Set provider data.
    #[must_use]
    pub fn provider_data(mut self, data: ProviderData) -> Self {
        self.response.provider_data = data;
        self
    }

    /// Build the response.
    #[must_use]
    pub fn build(mut self) -> ChatResponse {
        if !self.finish_reason_set {
            self.response.finish_reason = Some(if self.response.has_tool_calls() {
                FinishReason::ToolCalls
            } else {
                FinishReason::Stop
            });
        }
        self.response
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToolDefinition;
    use serde_json::json;

    #[tokio::test]
    async fn tool_calls_script_one_turn_with_several_calls() {
        let calls = [
            ToolCall::new("call_1", "weather", r#"{"city":"Paris"}"#),
            ToolCall::new("call_2", "weather", r#"{"city":"Rome"}"#),
        ];
        let request = ChatRequest::new("m").user("weather?");
        let provider = MockProvider::build(|b| b.tool_calls(calls.clone()));
        let response = provider.chat(&request).await.unwrap();
        let ids: Vec<_> = response.tool_calls().map(|call| call.id.as_str()).collect();
        assert_eq!(ids, ["call_1", "call_2"]);
        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));

        let streaming = MockStreamingProvider::build(|b| b.tool_calls(calls));
        let stream = streaming.chat_stream(&request).await.unwrap();
        let collected = stream.collect_response().await.unwrap();
        assert_eq!(collected.tool_calls().count(), 2);
    }

    #[tokio::test]
    async fn structured_answers_in_the_form_the_request_asks_for() {
        let value = json!({"name": "Ada"});
        let tool = |name: &str| ToolDefinition::new(name, json!({"type": "object"}));
        let respond = |request: ChatRequest| {
            let value = value.clone();
            async move {
                MockProvider::new([MockResponse::structured(value)])
                    .chat(&request)
                    .await
                    .unwrap()
            }
        };
        let base = ChatRequest::new("m").user("q");

        // No forced tool: JSON text.
        let text = respond(base.clone()).await;
        assert_eq!(text.text().as_deref(), Some(r#"{"name":"Ada"}"#));
        let auto = respond(base.clone().tools([tool("a")])).await;
        assert!(auto.text().is_some());

        // A specific tool, or the only tool under `Required`: a call to it.
        for request in [
            base.clone()
                .tools([tool("a"), tool("b")])
                .tool_choice(ToolChoice::Specific { name: "b".into() }),
            base.clone()
                .tools([tool("b")])
                .tool_choice(ToolChoice::Required),
        ] {
            let response = respond(request).await;
            let call = response.first_tool_call().unwrap();
            assert_eq!(call.name, "b");
            assert_eq!(call.parse_arguments::<serde_json::Value>().unwrap(), value);
        }

        // `Required` over several tools names none: JSON text.
        let ambiguous = respond(
            base.tools([tool("a"), tool("b")])
                .tool_choice(ToolChoice::Required),
        )
        .await;
        assert!(ambiguous.text().is_some());
    }

    #[tokio::test]
    async fn mock_records_requests_in_order() {
        let provider = MockProvider::build(|b| b.text("first").text("second"));
        let request = ChatRequest::new("m").user("hi");
        assert_eq!(
            provider.chat(&request).await.unwrap().text().as_deref(),
            Some("first")
        );
        assert_eq!(
            provider.chat(&request).await.unwrap().text().as_deref(),
            Some("second")
        );
        assert_eq!(provider.call_count(), 2);
        assert_eq!(provider.last_request().unwrap().model, "m");
        assert_eq!(provider.pending_responses(), 0);
    }

    #[tokio::test]
    async fn mocks_preflight_like_an_adapter() {
        use crate::{ReasoningEffort, ResponseFormat, WarningKind};

        let hint = ChatRequest::new("m")
            .user("hi")
            .reasoning(ReasoningEffort::High);
        let semantic =
            ChatRequest::new("m")
                .user("hi")
                .response_format(ResponseFormat::json_schema(
                    serde_json::json!({"type": "object"}),
                ));
        let unsupported = [
            ChatCapability::ReasoningConfig,
            ChatCapability::StructuredOutput,
        ];

        let mock = unsupported
            .into_iter()
            .fold(MockProvider::build(|b| b.text("a").text("b")), |p, c| {
                p.with_chat_capability(c, CapabilitySupport::Unsupported)
            });
        let response = mock.chat(&hint).await.unwrap();
        assert_eq!(response.warnings[0].kind, WarningKind::Unsupported);
        assert_eq!(response.warnings[0].provider.as_deref(), Some("mock"));
        // The request is recorded as the caller sent it.
        assert!(mock.last_request().unwrap().reasoning.is_some());
        assert!(mock.chat(&semantic).await.is_err());
        assert_eq!((mock.call_count(), mock.pending_responses()), (1, 1));

        let streaming = unsupported.into_iter().fold(
            MockStreamingProvider::from_responses([ChatResponseBuilder::new().text("t").build()]),
            |p, c| p.with_chat_capability(c, CapabilitySupport::Unsupported),
        );
        assert!(streaming.chat_stream(&semantic).await.is_err());
        let response = streaming.chat(&hint).await.unwrap();
        assert_eq!(response.text().as_deref(), Some("t"));
        assert_eq!(response.warnings.len(), 1);
        assert_eq!(response.warnings[0].kind, WarningKind::Unsupported);

        let function = unsupported.into_iter().fold(
            FunctionProvider::new(|_| Ok(ChatResponseBuilder::new().text("f").build())),
            |p, c| p.with_chat_capability(c, CapabilitySupport::Unsupported),
        );
        let response = function.chat(&hint).await.unwrap();
        assert_eq!(response.warnings.len(), 1);
        assert!(function.chat(&semantic).await.is_err());
    }

    #[tokio::test]
    async fn delayed_responses_resolve() {
        let provider = MockProvider::new([MockResponse::delayed(
            Duration::from_millis(1),
            MockResponse::text("late"),
        )]);
        let response = provider.chat(&ChatRequest::new("m")).await.unwrap();
        assert_eq!(response.text().as_deref(), Some("late"));
    }

    #[tokio::test]
    async fn streaming_mock_streams_and_collects() {
        let provider = MockStreamingProvider::from_responses([ChatResponseBuilder::new()
            .reasoning("r")
            .text("t")
            .build()]);
        let response = provider.chat(&ChatRequest::new("m")).await.unwrap();
        assert_eq!(response.text().as_deref(), Some("t"));
        assert_eq!(response.reasoning_text().as_deref(), Some("r"));
    }

    #[tokio::test]
    async fn exhausted_mock_returns_configuration_error() {
        let provider = MockProvider::build(|b| b.text("only"));
        let request = ChatRequest::new("m");
        provider.chat(&request).await.unwrap();

        let error = provider.chat(&request).await.unwrap_err();
        assert!(matches!(error, Error::Configuration { .. }));
        assert_eq!(
            error.message(),
            "MockProvider: no more responses configured"
        );
        assert!(provider.chat_stream(&request).await.is_err());
        assert_eq!(provider.call_count(), 3);
    }

    #[tokio::test]
    async fn mock_errors_fail_the_stream_call() {
        let provider = MockProvider::with_error(Error::rate_limited("slow down"));
        let error = provider
            .chat_stream(&ChatRequest::new("m"))
            .await
            .err()
            .unwrap();
        assert!(matches!(error, Error::RateLimited { .. }));
    }

    #[tokio::test]
    async fn exhausted_streaming_mock_returns_configuration_error() {
        let provider = MockStreamingProvider::with_text("only");
        let request = ChatRequest::new("m");
        provider.chat(&request).await.unwrap();

        let error = provider.chat_stream(&request).await.err().unwrap();
        assert!(matches!(error, Error::Configuration { .. }));
        assert_eq!(
            error.message(),
            "MockStreamingProvider: no more streams configured"
        );
        assert!(provider.chat(&request).await.is_err());
        assert_eq!(provider.call_count(), 3);
    }

    #[tokio::test]
    async fn setup_errors_fail_the_call_and_the_next_entry_streams() {
        let provider = MockStreamingProvider::build(|b| {
            b.text("first")
                .setup_error(Error::rate_limited("slow down"))
                .text("second")
        });
        let request = ChatRequest::new("m").user("hi");

        let first = provider.chat(&request).await.unwrap();
        assert_eq!(first.text().as_deref(), Some("first"));

        let error = provider.chat_stream(&request).await.err().unwrap();
        assert!(matches!(error, Error::RateLimited { .. }));

        let second = provider
            .chat_stream(&request)
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        assert_eq!(second.text().as_deref(), Some("second"));
        assert_eq!(provider.call_count(), 3);
        assert_eq!(provider.requests()[1], request);
    }

    #[tokio::test]
    async fn stream_errors_arrive_after_the_call_succeeds() {
        use futures_util::StreamExt;

        let provider = MockStreamingProvider::with_stream([
            MockStreamEvent::from(StreamEvent::response_start(None, None)),
            MockStreamEvent::from(Error::stream(
                crate::StreamErrorKind::Transport,
                "connection dropped",
            )),
        ]);
        let mut stream = provider.chat_stream(&ChatRequest::new("m")).await.unwrap();
        assert!(stream.next().await.unwrap().is_ok());
        assert!(stream.next().await.unwrap().is_err());
    }

    #[test]
    fn builder_defaults_finish_reason() {
        assert_eq!(
            ChatResponseBuilder::new().text("x").build().finish_reason,
            Some(FinishReason::Stop)
        );
        assert_eq!(
            ChatResponseBuilder::new()
                .tool_call("c", "f", json!({}))
                .build()
                .finish_reason,
            Some(FinishReason::ToolCalls)
        );
    }

    #[tokio::test]
    async fn test_provider_calls_tools_with_schema_valid_arguments() {
        let schema = json!({
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer", "minimum": 2},
                "unit": {"enum": ["c", "f"]},
                "tags": {"type": "array", "items": {"$ref": "#/$defs/tag"}}
            },
            "required": ["city", "days", "unit", "tags"],
            "$defs": {"tag": {"type": "string"}}
        });
        let provider = TestProvider::new();
        let mut request = ChatRequest::new("m")
            .user("weather?")
            .tools([ToolDefinition::new("forecast", schema)]);
        let first = provider.chat(&request).await.unwrap();
        let call = first.first_tool_call().unwrap().clone();
        let args: serde_json::Value = call.parse_arguments().unwrap();
        assert_eq!(
            args,
            json!({"city": "example", "days": 2, "unit": "c", "tags": ["example"]})
        );

        request.push_message(first.to_assistant_message());
        request.push_tool_result(&call, "sunny");
        let second = provider.chat(&request).await.unwrap();
        assert!(second.text().is_some());
        assert_eq!(provider.call_count(), 2);
    }

    #[tokio::test]
    async fn test_provider_returns_a_complete_envelope() {
        let request = ChatRequest::new("demo-model")
            .system("Be brief.")
            .user("What is the  weather\ntoday?");
        for response in [
            TestProvider::new().chat(&request).await.unwrap(),
            TestProvider::new()
                .chat_stream(&request)
                .await
                .unwrap()
                .collect_response()
                .await
                .unwrap(),
        ] {
            let envelope = response.envelope();
            assert_eq!(envelope.model.as_deref(), Some("demo-model"));
            assert_eq!(envelope.provider.as_deref(), Some("test"));
            assert_eq!(envelope.id.as_deref(), Some("test_response_1"));
            assert_eq!(envelope.finish_reason, Some(FinishReason::Stop));
            assert_eq!(envelope.request_id, None);
            // "Be brief." and "What is the weather today?": 2 + 5 words in,
            // "test response (1 messages)": 4 words out.
            let usage = envelope.usage.unwrap();
            assert_eq!(usage.input_tokens, Some(7));
            assert_eq!(usage.output_tokens, Some(4));
        }
        // The same request gets the same usage.
        let again = TestProvider::new().chat(&request).await.unwrap();
        assert_eq!(again.usage.unwrap().input_tokens, Some(7));
    }

    #[tokio::test]
    async fn test_provider_answers_a_json_schema_format_with_a_schema_valid_value() {
        let schema = json!({
            "type": "object",
            "properties": {
                "category": {"enum": ["billing", "bug"]},
                "urgent": {"type": "boolean"},
                "score": {"type": "integer", "minimum": 3}
            },
            "required": ["category", "urgent", "score"]
        });
        let provider = TestProvider::new();
        assert_eq!(
            provider.chat_capability("m", ChatCapability::StructuredOutput),
            CapabilitySupport::Supported
        );
        let request = ChatRequest::new("m")
            .user("triage this")
            .response_format(ResponseFormat::json_schema(schema.clone()));
        let response = provider.chat(&request).await.unwrap();
        let value: serde_json::Value = serde_json::from_str(&response.text().unwrap()).unwrap();
        assert_eq!(
            value,
            json!({"category": "billing", "urgent": true, "score": 3})
        );

        // The tool branch runs first, unless tools are disabled.
        let with_tools = request
            .clone()
            .tools([ToolDefinition::new("lookup", json!({"type": "object"}))]);
        let response = provider.chat(&with_tools).await.unwrap();
        assert_eq!(response.first_tool_call().unwrap().name, "lookup");
        let disabled = with_tools.tool_choice(ToolChoice::Disabled);
        let response = provider.chat(&disabled).await.unwrap();
        assert!(response.first_tool_call().is_none());
        assert!(response.text().unwrap().starts_with('{'));

        // Other formats still get text describing the conversation.
        let text = ChatRequest::new("m")
            .user("hi")
            .response_format(ResponseFormat::Json);
        let response = provider.chat(&text).await.unwrap();
        assert_eq!(
            response.text().as_deref(),
            Some("test response (1 messages)")
        );
    }

    #[tokio::test]
    async fn test_provider_tool_call_usage_counts_the_arguments() {
        let request = ChatRequest::new("m").user("go").tools([ToolDefinition::new(
            "f",
            json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}),
        )]);
        let response = TestProvider::new().chat(&request).await.unwrap();
        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
        let usage = response.usage.unwrap();
        assert_eq!(usage.input_tokens, Some(1));
        // `{"q":"example"}` is one word; usage is never zero.
        assert_eq!(usage.output_tokens, Some(1));
        let empty = TestProvider::new()
            .chat(&ChatRequest::new("m"))
            .await
            .unwrap();
        assert_eq!(empty.usage.unwrap().input_tokens, Some(1));
    }

    #[tokio::test]
    async fn test_provider_ids_are_unique_across_a_conversation() {
        let tool = ToolDefinition::new("lookup", json!({"type": "object"}));
        let mut request = ChatRequest::new("m").user("look it up twice").tools([tool]);
        let mut ids = Vec::new();
        let mut response_ids = Vec::new();
        for _ in 0..3 {
            // A new provider per turn, as a registry that builds one per
            // request does: the ids come from the conversation.
            let response = TestProvider::new().chat(&request).await.unwrap();
            response_ids.push(response.id.clone().unwrap());
            let call = response.first_tool_call().unwrap().clone();
            ids.push(call.id.clone());
            request.push_message(response.to_assistant_message());
            request.push_tool_result(&call, "found");
            request.push_message(Message::user("again"));
        }
        assert_eq!(ids, ["test_call_1", "test_call_2", "test_call_3"]);
        assert_eq!(
            response_ids,
            ["test_response_1", "test_response_2", "test_response_3"]
        );
        crate::history::validate(&request.messages, crate::history::ValidationMode::Sendable)
            .unwrap();
    }

    #[tokio::test]
    async fn test_provider_skips_ids_the_conversation_already_uses() {
        let tool = ToolDefinition::new("lookup", json!({"type": "object"}));
        let earlier = ToolCall::new("test_call_2", "lookup", "{}");
        let mut request = ChatRequest::new("m").user("look").tools([tool]);
        request.push_message(Message::assistant_blocks(vec![ContentBlock::ToolCall(
            earlier.clone(),
        )]));
        request.push_tool_result(&earlier, "found");
        request.push_message(Message::user("again"));
        let response = TestProvider::new().chat(&request).await.unwrap();
        assert_eq!(response.first_tool_call().unwrap().id, "test_call_3");
    }
}
