use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures_core::Stream;
use serde::{Deserialize, Serialize};

use crate::macros::enum_names;
use crate::{
    CapabilitySupport, ChatRequest, ChatResponse, ChatStream, ChatValidation, Error,
    ProviderIdentity, Result, StreamEvent, ValidationCoverage,
};

/// Core trait for providers that execute chat requests, whether through a
/// hosted API, a gateway, or an in-process inference runtime.
///
/// Implement [`chat`](Self::chat), [`chat_stream`](Self::chat_stream), and
/// [`chat_capability`](Self::chat_capability).
/// Providers without native streaming can return
/// [`SingleResponseStream`](crate::SingleResponseStream) from `chat_stream`.
///
/// Methods return `impl Future + Send`; implementors should write `async fn`.
/// For dynamic dispatch use [`DynChatProvider`].
pub trait ChatProvider: ProviderIdentity {
    /// Concrete stream type returned by [`chat_stream`](Self::chat_stream).
    ///
    /// The stream is `'static`: it owns what it reads, and borrows neither
    /// the provider nor the request. Wrappers and [`DynChatProvider`] box it
    /// into a [`ChatStream`] with no bound of their own.
    type Stream: Stream<Item = Result<StreamEvent>> + Send + 'static;

    /// Send a request and return the complete response.
    fn chat(&self, request: &ChatRequest) -> impl Future<Output = Result<ChatResponse>> + Send;

    /// Send a request and return a stream of events.
    fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> impl Future<Output = Result<Self::Stream>> + Send;

    /// Whether the provider supports `capability` for `model`.
    ///
    /// Required: validation and extraction's `Auto` mode act on the answer.
    /// A provider that does not know answers
    /// [`CapabilitySupport::Unknown`]; a wrapper forwards the answer of the
    /// provider it wraps.
    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport;

    /// Validate a request without sending it: a dry run of
    /// [`chat`](Self::chat).
    ///
    /// `Ok` means `chat` would send the request, and holds what it would
    /// report about it ([`ChatValidation`]): the request-side warnings, and
    /// the capabilities the request relies on that the provider answered
    /// `Unknown`. `Err` is the error `chat` would return before sending.
    /// Validation never checks credentials or that the model exists: those
    /// need a real call.
    ///
    /// The default runs [`chat::prepare_dry_run`](crate::adapter::chat::prepare_dry_run),
    /// the capability check against [`chat_capability`](Self::chat_capability),
    /// under the request's [`UnsupportedPolicy`](crate::UnsupportedPolicy)
    /// exactly as `chat` does: a semantic feature the provider reports
    /// [`Unsupported`](CapabilitySupport::Unsupported) fails; an unsupported
    /// hint fails only under
    /// [`UnsupportedPolicy::Error`](crate::UnsupportedPolicy::Error), and is
    /// a warning otherwise; `Unknown` is allowed through. It reports
    /// [`ValidationCoverage::Capabilities`]. An adapter overrides it to run
    /// its encoding too and reports [`ValidationCoverage::Encoding`];
    /// wrappers forward it.
    ///
    /// # Errors
    ///
    /// The error `chat` would return before sending, such as
    /// [`Error::Unsupported`] for the first unsupported feature.
    fn validate_chat_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        let prepared = crate::adapter::chat::prepare_dry_run(self, request)?;
        Ok(prepared.into_validation(ValidationCoverage::Capabilities))
    }

    /// Validate a request for [`chat_stream`](Self::chat_stream) without
    /// sending it, as [`validate_chat_request`](Self::validate_chat_request)
    /// does for `chat`.
    ///
    /// Streaming can refuse what a unary call takes (and the reverse), so a
    /// streaming application validates with this one. The default also
    /// reports [`Streaming`](ChatCapability::Streaming) and
    /// [`NativeStreaming`](ChatCapability::NativeStreaming) when the
    /// provider answers them `Unknown`.
    ///
    /// # Errors
    ///
    /// The error `chat_stream` would return before sending.
    fn validate_chat_stream_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        let mut prepared = crate::adapter::chat::prepare_dry_run(self, request)?;
        prepared.preflight_stream(self);
        Ok(prepared.into_validation(ValidationCoverage::Capabilities))
    }
}

/// Portable chat features a provider or model may support.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatCapability {
    /// Tool definitions and tool-call output.
    ToolCalls,
    /// Several tool calls in one assistant turn.
    ParallelToolCalls,
    /// Incremental delivery through `chat_stream` (native or simulated).
    Streaming,
    /// Genuinely incremental upstream streaming (not a replayed response).
    NativeStreaming,
    /// Token usage on streamed responses: the response `chat_stream`
    /// collects into carries [`usage`](crate::ChatResponse::usage).
    ///
    /// It describes output; no request field asks for it. A budget guard
    /// that folds a stream's usage
    /// ([`StreamCollector::metadata_only`](crate::chat::stream::StreamCollector::metadata_only))
    /// checks it first: `Unsupported` means the stream reports no usage, so
    /// the guard needs another bound, such as `max_output_tokens` or the
    /// unary call, and `Unknown` means it may or may not.
    #[doc(alias = "include_usage")]
    #[doc(alias = "stream_options")]
    StreamUsage,
    /// Image parts in user input.
    ImageInput,
    /// Per-image detail hints.
    ImageDetail,
    /// Image blocks in assistant output.
    ImageOutput,
    /// Replaying assistant image blocks in later turns.
    ImageReplay,
    /// Document parts in user input and tool results: a
    /// [`File`](crate::File) whose media type is not audio or video (PDF,
    /// plain text, …; [`FileKind::Document`](crate::FileKind::Document)).
    ///
    /// Which document types and sources a provider takes still varies
    /// within a `Supported` answer: PDF almost everywhere, plain text on
    /// some, and a URL, inline data or a provider file depending on the API
    /// (OpenAI's Chat Completions takes a PDF as inline data or a file id,
    /// not a URL). An adapter refuses a type or source it cannot send with
    /// [`Error::Unsupported`](crate::Error::Unsupported).
    FileInput,
    /// Audio parts in user input and tool results: a
    /// [`File`](crate::File) with an `audio/*` media type
    /// ([`FileKind::Audio`](crate::FileKind::Audio)).
    AudioInput,
    /// Video parts in user input and tool results: a
    /// [`File`](crate::File) with a `video/*` media type
    /// ([`FileKind::Video`](crate::FileKind::Video)).
    VideoInput,
    /// JSON Schema response formats
    /// ([`ResponseFormat::JsonSchema`](crate::ResponseFormat::JsonSchema)).
    ///
    /// JSON mode ([`ResponseFormat::Json`](crate::ResponseFormat::Json)) is
    /// not covered: providers split on it apart from schemas (Anthropic
    /// takes a schema and has no JSON mode, DeepSeek has JSON mode and takes
    /// no schema), so no one answer would be true for both. Preflight never
    /// refuses JSON mode; an adapter whose API has none refuses it with
    /// [`Error::Unsupported`](crate::Error::Unsupported).
    StructuredOutput,
    /// Reasoning blocks in assistant output.
    ReasoningOutput,
    /// Replaying reasoning blocks (with their opaque state) in later turns.
    ReasoningReplay,
    /// Request-side reasoning configuration.
    ReasoningConfig,
    /// Prompt-cache hints (breakpoints, key, retention).
    PromptCache,
    /// Requiring a tool call: `ToolChoice::Required` or `ToolChoice::Specific`.
    ///
    /// Some models accept tools but reject forced tool use. Extraction's
    /// `Auto` mode only falls back to a forced tool call when this is not
    /// `Unsupported`.
    ForcedToolChoice,
}

enum_names!(ChatCapability, "chat capability", {
    ToolCalls => "tool_calls",
    ParallelToolCalls => "parallel_tool_calls",
    Streaming => "streaming",
    NativeStreaming => "native_streaming",
    StreamUsage => "stream_usage",
    ImageInput => "image_input",
    ImageDetail => "image_detail",
    ImageOutput => "image_output",
    ImageReplay => "image_replay",
    FileInput => "file_input",
    AudioInput => "audio_input",
    VideoInput => "video_input",
    StructuredOutput => "structured_output",
    ReasoningOutput => "reasoning_output",
    ReasoningReplay => "reasoning_replay",
    ReasoningConfig => "reasoning_config",
    PromptCache => "prompt_cache",
    ForcedToolChoice => "forced_tool_choice",
});

impl ChatCapability {
    /// Every capability, in declaration order (for audits and capability
    /// tables). Grows when capabilities are added.
    pub const ALL: &[Self] = &[
        Self::ToolCalls,
        Self::ParallelToolCalls,
        Self::Streaming,
        Self::NativeStreaming,
        Self::StreamUsage,
        Self::ImageInput,
        Self::ImageDetail,
        Self::ImageOutput,
        Self::ImageReplay,
        Self::FileInput,
        Self::AudioInput,
        Self::VideoInput,
        Self::StructuredOutput,
        Self::ReasoningOutput,
        Self::ReasoningReplay,
        Self::ReasoningConfig,
        Self::PromptCache,
        Self::ForcedToolChoice,
    ];
}

/// Optional override for a provider's capability answers.
///
/// Return `None` to defer to the provider, `Some(..)` to override.
pub trait ChatCapabilityResolver: Send + Sync + 'static {
    /// Override for `capability` on `model`, or `None` to defer.
    fn chat_capability(&self, model: &str, capability: ChatCapability)
    -> Option<CapabilitySupport>;
}

impl<F> ChatCapabilityResolver for F
where
    F: for<'a> Fn(&'a str, ChatCapability) -> Option<CapabilitySupport> + Send + Sync + 'static,
{
    fn chat_capability(
        &self,
        model: &str,
        capability: ChatCapability,
    ) -> Option<CapabilitySupport> {
        self(model, capability)
    }
}

/// A shared resolver, as providers store it, so one resolver can be passed
/// on to another provider (`with_*_capabilities` takes it as is).
impl ChatCapabilityResolver for Arc<dyn ChatCapabilityResolver> {
    fn chat_capability(
        &self,
        model: &str,
        capability: ChatCapability,
    ) -> Option<CapabilitySupport> {
        (**self).chat_capability(model, capability)
    }
}

macro_rules! forward_chat_provider {
    ($ty:ty, |$s:ident| $inner:expr) => {
        impl<T> ChatProvider for $ty
        where
            T: ChatProvider + ?Sized,
        {
            type Stream = T::Stream;

            async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
                let $s = self;
                let inner: &T = $inner;
                inner.chat(request).await
            }

            async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
                let $s = self;
                let inner: &T = $inner;
                inner.chat_stream(request).await
            }

            fn chat_capability(
                &self,
                model: &str,
                capability: ChatCapability,
            ) -> CapabilitySupport {
                let $s = self;
                let inner: &T = $inner;
                inner.chat_capability(model, capability)
            }

            fn validate_chat_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
                let $s = self;
                let inner: &T = $inner;
                inner.validate_chat_request(request)
            }

            fn validate_chat_stream_request(
                &self,
                request: &ChatRequest,
            ) -> Result<ChatValidation> {
                let $s = self;
                let inner: &T = $inner;
                inner.validate_chat_stream_request(request)
            }
        }
    };
}

forward_chat_provider!(&T, |s| *s);
forward_chat_provider!(Box<T>, |s| s.as_ref());
forward_chat_provider!(Arc<T>, |s| s.as_ref());

/// A type-erased chat provider for dynamic dispatch.
///
/// Wraps any `T: ChatProvider + 'static`, boxing futures only at the `dyn`
/// boundary.
#[derive(Clone)]
pub struct DynChatProvider(Arc<dyn ChatProviderErased>);

impl DynChatProvider {
    /// Erase a provider.
    #[must_use]
    pub fn new<T>(provider: T) -> Self
    where
        T: ChatProvider + 'static,
    {
        Self(Arc::new(provider))
    }
}

impl<T> From<Arc<T>> for DynChatProvider
where
    T: ChatProvider + 'static,
{
    fn from(provider: Arc<T>) -> Self {
        Self(provider)
    }
}

impl std::fmt::Debug for DynChatProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynChatProvider")
            .field("provider", &self.0.provider_name())
            .finish()
    }
}

impl ProviderIdentity for DynChatProvider {
    fn provider_name(&self) -> &str {
        self.0.provider_name()
    }

    fn reads_namespace(&self, namespace: &str) -> bool {
        self.0.reads_namespace(namespace)
    }

    fn downcast_any(&self, id: std::any::TypeId) -> Option<&(dyn std::any::Any + Send + Sync)> {
        self.0.downcast_any(id)
    }
}

impl ChatProvider for DynChatProvider {
    type Stream = ChatStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        self.0.chat_erased(request).await
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        self.0.chat_stream_erased(request).await
    }

    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        self.0.chat_capability_erased(model, capability)
    }

    fn validate_chat_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        self.0.validate_erased(request)
    }

    fn validate_chat_stream_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        self.0.validate_stream_erased(request)
    }
}

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

trait ChatProviderErased: ProviderIdentity {
    fn chat_erased<'a>(&'a self, request: &'a ChatRequest) -> BoxFuture<'a, Result<ChatResponse>>;
    fn chat_stream_erased<'a>(
        &'a self,
        request: &'a ChatRequest,
    ) -> BoxFuture<'a, Result<ChatStream>>;
    fn chat_capability_erased(&self, model: &str, capability: ChatCapability) -> CapabilitySupport;
    fn validate_erased(&self, request: &ChatRequest) -> Result<ChatValidation>;
    fn validate_stream_erased(&self, request: &ChatRequest) -> Result<ChatValidation>;
}

impl<T> ChatProviderErased for T
where
    T: ChatProvider,
{
    fn chat_erased<'a>(&'a self, request: &'a ChatRequest) -> BoxFuture<'a, Result<ChatResponse>> {
        Box::pin(ChatProvider::chat(self, request))
    }

    fn chat_stream_erased<'a>(
        &'a self,
        request: &'a ChatRequest,
    ) -> BoxFuture<'a, Result<ChatStream>> {
        Box::pin(async move {
            let stream = ChatProvider::chat_stream(self, request).await?;
            Ok(Box::pin(stream) as ChatStream)
        })
    }

    fn chat_capability_erased(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        ChatProvider::chat_capability(self, model, capability)
    }

    fn validate_erased(&self, request: &ChatRequest) -> Result<ChatValidation> {
        ChatProvider::validate_chat_request(self, request)
    }

    fn validate_stream_erased(&self, request: &ChatRequest) -> Result<ChatValidation> {
        ChatProvider::validate_chat_stream_request(self, request)
    }
}

/// Convenience methods for every [`ChatProvider`].
pub trait ChatProviderExt: ChatProvider {
    /// Send a single user message and return the response text.
    ///
    /// Returns only the text, even when the response did not finish
    /// normally; use [`chat`](ChatProvider::chat) for the finish reason,
    /// refusal, usage and warnings.
    ///
    /// # Errors
    ///
    /// The provider's error, or [`Error::UnexpectedResponse`] when the
    /// response has no text. When the model refused, the error's message
    /// says so and the refusal is its
    /// [upstream message](Error::upstream_message), kept out of
    /// [`as_log`](Error::as_log) like any text the model wrote.
    fn ask(
        &self,
        model: &str,
        message: impl Into<String>,
    ) -> impl Future<Output = Result<String>> + Send {
        let request = ChatRequest::new(model).user(message.into());
        async move { text_or_error(self, self.chat(&request).await?, "ask") }
    }

    /// Send a system prompt and a user message and return the response text.
    ///
    /// Returns only the text, even when the response did not finish
    /// normally; use [`chat`](ChatProvider::chat) for the finish reason,
    /// refusal, usage and warnings.
    ///
    /// # Errors
    ///
    /// As [`ask`](Self::ask).
    fn ask_with_system(
        &self,
        model: &str,
        system: impl Into<String>,
        message: impl Into<String>,
    ) -> impl Future<Output = Result<String>> + Send {
        let request = ChatRequest::new(model)
            .system(system.into())
            .user(message.into());
        async move { text_or_error(self, self.chat(&request).await?, "ask_with_system") }
    }
}

fn text_or_error<P: ChatProvider + ?Sized>(
    provider: &P,
    response: ChatResponse,
    method: &str,
) -> Result<String> {
    if let Some(text) = response.text() {
        return Ok(text);
    }
    // The refusal is the model's own text: it may quote the prompt, so it
    // goes where `as_log` leaves it out, never into the message.
    if let Some(refusal) = response.first_refusal() {
        return Err(Error::unexpected_response(format!(
            "the model refused and returned no text for {method}()"
        ))
        .authored()
        .with_provider(provider.provider_name())
        .with_upstream_message(refusal.to_owned()));
    }
    Err(
        Error::unexpected_response(format!("the response has no text content for {method}()"))
            .authored()
            .with_provider(provider.provider_name()),
    )
}

impl<T: ChatProvider> ChatProviderExt for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::{ChatResponseBuilder, MockProvider};
    use crate::{
        ChatStreamExt, ContentPart, ProviderIdentityExt, ResponseFormat, ToolChoice, ToolDefinition,
    };
    use serde_json::json;

    #[test]
    fn all_capabilities_listed() {
        // Exhaustive on purpose: a new variant fails to compile here, as a
        // reminder to add it to `ChatCapability::ALL` and bump the count.
        fn _exhaustive(capability: ChatCapability) {
            match capability {
                ChatCapability::ToolCalls
                | ChatCapability::ParallelToolCalls
                | ChatCapability::Streaming
                | ChatCapability::NativeStreaming
                | ChatCapability::StreamUsage
                | ChatCapability::ImageInput
                | ChatCapability::ImageDetail
                | ChatCapability::ImageOutput
                | ChatCapability::ImageReplay
                | ChatCapability::FileInput
                | ChatCapability::AudioInput
                | ChatCapability::VideoInput
                | ChatCapability::StructuredOutput
                | ChatCapability::ReasoningOutput
                | ChatCapability::ReasoningReplay
                | ChatCapability::ReasoningConfig
                | ChatCapability::PromptCache
                | ChatCapability::ForcedToolChoice => {}
            }
        }
        let unique: std::collections::HashSet<_> = ChatCapability::ALL.iter().collect();
        assert_eq!(unique.len(), ChatCapability::ALL.len());
        assert_eq!(ChatCapability::ALL.len(), 18);
    }

    #[tokio::test]
    async fn dyn_provider_forwards_everything() {
        let mock = MockProvider::with_response(ChatResponseBuilder::new().text("hi").build())
            .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Unsupported);
        let provider = DynChatProvider::new(mock);
        assert_eq!(provider.provider_name(), "mock");
        assert_eq!(
            provider.chat_capability("m", ChatCapability::ToolCalls),
            CapabilitySupport::Unsupported
        );
        let request = ChatRequest::new("m")
            .user("x")
            .tools([ToolDefinition::new("t", json!({"type": "object"}))]);
        let err = provider.validate_chat_request(&request).unwrap_err();
        // Preflight writes its own message: the log keeps it.
        assert_eq!(
            err.as_log().to_string(),
            r#"kind=unsupported provider=mock message="cannot send `tools`: tool calling is not supported by `mock`""#
        );
        assert!(format!("{provider:?}").contains("mock"));
        assert!(provider.downcast_ref::<MockProvider>().is_some());
    }

    #[tokio::test]
    async fn dyn_stream_erases_type() {
        let provider = DynChatProvider::new(MockProvider::with_response(
            ChatResponseBuilder::new().text("streamed").build(),
        ));
        let response = provider
            .chat_stream(&ChatRequest::new("m").user("x"))
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        assert_eq!(response.text().as_deref(), Some("streamed"));
    }

    #[tokio::test]
    async fn ask_helpers() {
        let provider = MockProvider::build(|b| b.text("one").text("two"));
        assert_eq!(provider.ask("m", "hi").await.unwrap(), "one");
        assert_eq!(
            provider.ask_with_system("m", "sys", "hi").await.unwrap(),
            "two"
        );
        let empty = MockProvider::with_response(ChatResponse::default());
        assert!(matches!(
            empty.ask("m", "hi").await.unwrap_err(),
            Error::UnexpectedResponse { .. }
        ));
    }

    #[tokio::test]
    async fn an_ask_refusal_keeps_the_model_text_out_of_the_log() {
        let refusal = ChatResponse::new(vec![crate::ContentBlock::Refusal(crate::Refusal::new(
            "I won't repeat your SSN 123-45-6789.",
        ))])
        .finish_reason(crate::FinishReason::Refusal);
        let provider = MockProvider::with_response(refusal);

        let err = provider.ask("m", "hi").await.unwrap_err();
        assert!(matches!(err, Error::UnexpectedResponse { .. }), "{err:?}");
        assert_eq!(
            err.message(),
            "the model refused and returned no text for ask()"
        );
        assert_eq!(err.message_origin(), crate::MessageOrigin::Authored);
        assert_eq!(
            err.upstream_message(),
            Some("I won't repeat your SSN 123-45-6789.")
        );
        let log = err.as_log().to_string();
        assert!(log.contains("refused"), "{log}");
        assert!(!log.contains("123-45-6789"), "{log}");
        // `Display` is not redacted.
        assert!(err.to_string().contains("123-45-6789"));
    }

    #[test]
    fn validation_is_conservative_on_unknown() {
        let provider = MockProvider::empty();
        let request = ChatRequest::new("m")
            .user(vec![ContentPart::file_base64(
                "application/pdf",
                "JVBERi0=",
            )])
            .response_format(ResponseFormat::Json);
        assert!(provider.validate_chat_request(&request).is_ok());

        let strict = MockProvider::empty()
            .with_chat_capability(ChatCapability::FileInput, CapabilitySupport::Unsupported);
        let err = strict.validate_chat_request(&request).unwrap_err();
        assert!(err.to_string().contains("file input"));

        let forced = ChatRequest::new("m")
            .user("hi")
            .tools([ToolDefinition::new("t", json!({"type": "object"}))])
            .tool_choice(ToolChoice::Required);
        assert!(provider.validate_chat_request(&forced).is_ok());
        let rejects = MockProvider::empty().with_chat_capability(
            ChatCapability::ForcedToolChoice,
            CapabilitySupport::Unsupported,
        );
        let err = rejects.validate_chat_request(&forced).unwrap_err();
        assert!(err.to_string().contains("forced tool choice"));
    }

    #[tokio::test]
    async fn validation_and_chat_agree_on_unsupported_hints() {
        use crate::{ReasoningEffort, UnsupportedPolicy, WarningKind};

        let provider = || {
            MockProvider::with_text("ok").with_chat_capability(
                ChatCapability::ReasoningConfig,
                CapabilitySupport::Unsupported,
            )
        };
        let request = ChatRequest::new("m")
            .user("hi")
            .reasoning(ReasoningEffort::High);

        // Warn (the default): both accept and report the hint.
        let lenient = provider();
        let validation = lenient.validate_chat_request(&request).unwrap();
        let response = lenient.chat(&request).await.unwrap();
        assert_eq!(response.warnings.len(), 1, "{:?}", response.warnings);
        assert_eq!(response.warnings[0].kind, WarningKind::Unsupported);
        assert_eq!(validation.warnings(), response.warnings);

        // Error: both refuse.
        let strict_request = request.unsupported_policy(UnsupportedPolicy::Error);
        let strict = provider();
        let err = strict.validate_chat_request(&strict_request).unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }), "{err}");
        let err = strict.chat(&strict_request).await.unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }), "{err}");
        assert_eq!(strict.call_count(), 0);
        assert_eq!(strict.pending_responses(), 1);
    }

    #[tokio::test]
    async fn validation_never_blocks_on_unknown() {
        use crate::{PromptCache, ReasoningEffort, UnsupportedPolicy};

        let request = ChatRequest::new("m")
            .user(vec![ContentPart::file_base64(
                "application/pdf",
                "JVBERi0=",
            )])
            .reasoning(ReasoningEffort::High)
            .parallel_tool_calls(true)
            .cache(PromptCache::new().key("k"))
            .unsupported_policy(UnsupportedPolicy::Error);
        let provider = MockProvider::with_text("ok");
        let validation = provider.validate_chat_request(&request).unwrap();
        assert!(validation.warnings().is_empty());
        // What the dry run could not know, with the first subject relying
        // on each.
        let unknown: Vec<_> = validation
            .relied_on_unknown()
            .iter()
            .map(|(capability, _)| *capability)
            .collect();
        assert_eq!(
            unknown,
            [
                ChatCapability::FileInput,
                ChatCapability::ReasoningConfig,
                ChatCapability::PromptCache,
            ]
        );
        assert_eq!(
            validation.relied_on_unknown()[0].1,
            crate::Subject::Content {
                message: 0,
                block: Some(0)
            }
        );
        let response = provider.chat(&request).await.unwrap();
        assert!(response.warnings.is_empty());
    }

    #[test]
    fn stream_validation_reports_what_streaming_relies_on() {
        let request = ChatRequest::new("m").user("hi");
        let unknown = |provider: &MockProvider| {
            provider
                .validate_chat_stream_request(&request)
                .unwrap()
                .relied_on_unknown()
                .iter()
                .map(|(capability, _)| *capability)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            unknown(&MockProvider::empty()),
            [ChatCapability::Streaming, ChatCapability::NativeStreaming]
        );
        let known = MockProvider::empty().with_supported_chat_capabilities([
            ChatCapability::Streaming,
            ChatCapability::NativeStreaming,
        ]);
        assert!(unknown(&known).is_empty());
        let unary = MockProvider::empty()
            .validate_chat_request(&request)
            .unwrap();
        assert!(unary.relied_on_unknown().is_empty());
        assert_eq!(unary.coverage(), ValidationCoverage::Capabilities);
    }

    #[test]
    fn closure_resolver() {
        let resolver = |model: &str, capability| {
            (model == "x" && capability == ChatCapability::ToolCalls)
                .then_some(CapabilitySupport::Supported)
        };
        assert_eq!(
            resolver.chat_capability("x", ChatCapability::ToolCalls),
            Some(CapabilitySupport::Supported)
        );
        assert_eq!(
            resolver.chat_capability("y", ChatCapability::ToolCalls),
            None
        );
    }
}
