use std::any::{Any, TypeId};
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::{
    CapabilitySupport, ChatCapability, ChatCapabilityResolver, ChatProvider, ChatRequest,
    ChatResponse, ChatStream, ChatValidation, Degradations, Error, ProviderIdentity, Result,
    Subject, Validation, Warning, WarningKind,
};

/// Overrides the chat capability answers of the provider it wraps, for
/// model ids the provider cannot classify: a fine-tuned model
/// (`ft:gpt-4.1:acme::1`), an Azure deployment name, a gateway alias.
///
/// It is the way to override the answers of a provider you hold only as a
/// [`DynChatProvider`](crate::DynChatProvider), such as one a registry
/// built, which no adapter-level override can reach. With the adapter at
/// hand, prefer its own `with_chat_capabilities`, which also feeds the
/// adapter's checks (see below).
///
/// ```
/// # #[cfg(feature = "mock")] {
/// use allama::mock::MockProvider;
/// use allama::wrappers::CapabilityOverrideProvider;
/// use allama::{CapabilitySupport, ChatCapability, ChatProvider};
///
/// // A gateway alias the provider answers `Unknown` for.
/// let provider = CapabilityOverrideProvider::new(MockProvider::empty(), |model: &str, capability| {
///     (model == "fast" && capability == ChatCapability::ToolCalls)
///         .then_some(CapabilitySupport::Supported)
/// });
/// assert_eq!(
///     provider.chat_capability("fast", ChatCapability::ToolCalls),
///     CapabilitySupport::Supported
/// );
/// # }
/// ```
///
/// A resolver returns `None` to keep the wrapped provider's answer. The
/// wrapper checks each request against the overridden answers with its own
/// [`chat::prepare`](crate::adapter::chat::prepare), forwards the request
/// it checked (less any hint a downgraded answer removed), and joins its
/// warnings to the response. Its
/// validation is the same check followed by the wrapped provider's own
/// validation of the forwarded request; the capabilities the resolver
/// settled are no longer listed as relied on `Unknown`.
///
/// It can **downgrade** an answer and **settle** an `Unknown` one, but it
/// cannot **upgrade** an answer the wrapped provider gives as
/// [`Unsupported`](CapabilitySupport::Unsupported): that provider checks
/// the request again before sending and would refuse it or drop the
/// feature anyway. Such an override is not applied: the wrapper answers
/// `Unsupported` too, and a request that relies on the capability gets a
/// warning (code `other.capability_override`) that says so. Override such
/// an answer on the adapter itself (`with_chat_capabilities` on every
/// first-party adapter), whose own checks then follow it. A resolver is a
/// function over arbitrary model ids, so this is only known per call.
///
/// Everything else is forwarded: the provider name, the namespaces it reads,
/// typed lookup.
///
/// It wraps chat providers only. Embedding and decision capability answers
/// have no override for now, on a wrapper, an adapter or
/// `allama_providers::ProviderSettings` (which carries chat overrides
/// only): an embedding model an adapter answers `Unknown` for keeps that
/// answer, so a request that relies on the capability is sent and
/// recorded as relying on an `Unknown` answer.
pub struct CapabilityOverrideProvider<P> {
    inner: P,
    resolver: Arc<dyn ChatCapabilityResolver>,
}

impl<P> CapabilityOverrideProvider<P> {
    /// Wrap `inner`, overriding its chat capability answers with `resolver`.
    #[must_use]
    pub fn new(inner: P, resolver: impl ChatCapabilityResolver) -> Self {
        Self {
            inner,
            resolver: Arc::new(resolver),
        }
    }

    /// Override the chat capability answers with `resolver`, replacing the
    /// current one.
    #[must_use]
    pub fn with_chat_capabilities(mut self, resolver: impl ChatCapabilityResolver) -> Self {
        self.resolver = Arc::new(resolver);
        self
    }

    /// Borrow the wrapped provider.
    #[must_use]
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Unwrap into the wrapped provider.
    #[must_use]
    pub fn into_inner(self) -> P {
        self.inner
    }
}

/// The answer a wrapper may give: the override, unless it upgrades an
/// `Unsupported` answer of the wrapped provider. The flag says the override
/// was refused.
fn effective(
    over: Option<CapabilitySupport>,
    inner: CapabilitySupport,
) -> (CapabilitySupport, bool) {
    match over {
        None => (inner, false),
        Some(over) if inner == CapabilitySupport::Unsupported && over != inner => {
            (CapabilitySupport::Unsupported, true)
        }
        Some(over) => (over, false),
    }
}

/// Whether the resolver settled `capability` (answered it either way).
fn settled(over: Option<CapabilitySupport>) -> bool {
    over.is_some_and(CapabilitySupport::is_known)
}

/// `inner`'s validation, less the `Unknown` answers the resolver settled.
fn without_settled<C: crate::Capability>(
    inner: Validation<C>,
    over: impl Fn(C) -> Option<CapabilitySupport>,
) -> Validation<C> {
    let settled: Vec<C> = inner
        .relied_on_unknown()
        .iter()
        .map(|(capability, _)| *capability)
        .filter(|&capability| settled(over(capability)))
        .collect();
    settled.into_iter().fold(inner, |validation, capability| {
        validation.without_unknown(capability)
    })
}

/// The validation `merged` from this wrapper's check and `inner`'s, less the
/// `Unknown` answers the check recorded only because the resolver deferred
/// to the wrapped provider: those are the wrapped provider's to report, and
/// its own validation lists the ones the request relies on. An answer the
/// resolver gives as `Unknown` stays.
fn without_deferred<C: crate::Capability>(
    merged: Validation<C>,
    inner: &[C],
    over: impl Fn(C) -> Option<CapabilitySupport>,
) -> Validation<C> {
    let deferred: Vec<C> = merged
        .relied_on_unknown()
        .iter()
        .map(|(capability, _)| *capability)
        .filter(|capability| over(*capability).is_none() && !inner.contains(capability))
        .collect();
    deferred.into_iter().fold(merged, |validation, capability| {
        validation.without_unknown(capability)
    })
}

/// The capabilities `validation` relies on as `Unknown`.
fn listed_unknown<C: crate::Capability>(validation: &Validation<C>) -> Vec<C> {
    validation
        .relied_on_unknown()
        .iter()
        .map(|(capability, _)| *capability)
        .collect()
}

/// Whether a failed call reached the transport, so the wrapper's warnings
/// describe a call that happened. A refusal says so with its stage; a
/// `Configuration` error and a local refusal (`Rejected`) come before
/// sending too.
fn reached_transport(error: &Error) -> bool {
    match error.unsupported_stage() {
        Some(stage) => stage != crate::UnsupportedStage::BeforeSending,
        None => !matches!(error, Error::Configuration { .. } | Error::Rejected { .. }),
    }
}

/// The wrapper's answers, recording the overrides it refused for the
/// capabilities a check asked about: those the request relies on.
struct Checked<'a, P> {
    wrapper: &'a CapabilityOverrideProvider<P>,
    refused: Mutex<Vec<String>>,
}

impl<'a, P: ProviderIdentity> Checked<'a, P> {
    fn new(wrapper: &'a CapabilityOverrideProvider<P>) -> Self {
        Self {
            wrapper,
            refused: Mutex::new(Vec::new()),
        }
    }

    fn answer<C: fmt::Debug>(
        &self,
        capability: C,
        over: Option<CapabilitySupport>,
        inner: CapabilitySupport,
    ) -> CapabilitySupport {
        let (answer, refused) = effective(over, inner);
        if refused {
            let name = format!("{capability:?}");
            let mut list = self
                .refused
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if !list.contains(&name) {
                list.push(name);
            }
        }
        answer
    }

    /// Warn about each refused override the check came across.
    fn report(self, degradations: &mut Degradations, model: &str) {
        let refused = self
            .refused
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for capability in refused {
            degradations.warn(
                Warning::new(
                    WarningKind::Other,
                    Subject::Other {
                        name: format!("capability {capability}"),
                    },
                    format!(
                        "the wrapped provider answers `{capability}` unsupported for `{model}`, which a wrapper cannot upgrade; the override was not applied; override it on the adapter (`with_chat_capabilities`) instead"
                    ),
                )
                .with_code("other.capability_override"),
            );
        }
    }
}

impl<P: ProviderIdentity> ProviderIdentity for Checked<'_, P> {
    fn provider_name(&self) -> &str {
        self.wrapper.inner.provider_name()
    }

    fn reads_namespace(&self, namespace: &str) -> bool {
        self.wrapper.inner.reads_namespace(namespace)
    }
}

impl<P: ChatProvider> ChatProvider for Checked<'_, P> {
    type Stream = ChatStream;

    async fn chat(&self, _request: &ChatRequest) -> Result<ChatResponse> {
        unreachable!("only asked for capability answers")
    }

    async fn chat_stream(&self, _request: &ChatRequest) -> Result<ChatStream> {
        unreachable!("only asked for capability answers")
    }

    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        let wrapper = self.wrapper;
        self.answer(
            capability,
            wrapper.chat_override(model, capability),
            wrapper.inner.chat_capability(model, capability),
        )
    }
}

impl<P> CapabilityOverrideProvider<P> {
    fn chat_override(&self, model: &str, capability: ChatCapability) -> Option<CapabilitySupport> {
        self.resolver.chat_capability(model, capability)
    }
}

impl<P: ChatProvider> CapabilityOverrideProvider<P> {
    /// The wrapper's own check of `request`, with a warning for each refused
    /// override it relies on.
    fn prepare_chat<'a>(
        &self,
        request: &'a ChatRequest,
        dry_run: bool,
        stream: bool,
    ) -> Result<crate::adapter::chat::Prepared<'a>> {
        let checked = Checked::new(self);
        let mut prepared = if dry_run {
            crate::adapter::chat::prepare_dry_run(&checked, request)?
        } else {
            crate::adapter::chat::prepare(&checked, request)?
        };
        if stream {
            prepared.preflight_stream(&checked);
        }
        checked.report(prepared.degradations_mut(), &request.model);
        Ok(prepared)
    }

    fn validate_chat(&self, request: &ChatRequest, stream: bool) -> Result<ChatValidation> {
        let prepared = self.prepare_chat(request, true, stream)?;
        let inner = if stream {
            self.inner
                .validate_chat_stream_request(prepared.request())?
        } else {
            self.inner.validate_chat_request(prepared.request())?
        };
        let inner = without_settled(inner, |c| self.chat_override(&request.model, c));
        let listed = listed_unknown(&inner);
        Ok(without_deferred(
            prepared.finish_validation(inner),
            &listed,
            |c| self.chat_override(&request.model, c),
        ))
    }
}

impl<P> ChatProvider for CapabilityOverrideProvider<P>
where
    P: ChatProvider,
{
    type Stream = ChatStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let mut prepared = self.prepare_chat(request, false, false)?;
        match self.inner.chat(prepared.request()).await {
            Ok(response) => Ok(prepared.finish(response)),
            Err(error) => {
                if reached_transport(&error) {
                    prepared.mark_sent();
                }
                Err(error)
            }
        }
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        let mut prepared = self.prepare_chat(request, false, true)?;
        match self.inner.chat_stream(prepared.request()).await {
            Ok(stream) => Ok(prepared.finish_stream(stream)),
            Err(error) => {
                if reached_transport(&error) {
                    prepared.mark_sent();
                }
                Err(error)
            }
        }
    }

    /// The override, or the wrapped provider's answer when the resolver
    /// defers. An override that would upgrade an `Unsupported` answer is not
    /// applied.
    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        effective(
            self.chat_override(model, capability),
            self.inner.chat_capability(model, capability),
        )
        .0
    }

    fn validate_chat_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        self.validate_chat(request, false)
    }

    fn validate_chat_stream_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        self.validate_chat(request, true)
    }
}

impl<P: ProviderIdentity> ProviderIdentity for CapabilityOverrideProvider<P> {
    fn provider_name(&self) -> &str {
        self.inner.provider_name()
    }

    fn reads_namespace(&self, namespace: &str) -> bool {
        self.inner.reads_namespace(namespace)
    }

    fn downcast_any(&self, id: TypeId) -> Option<&(dyn Any + Send + Sync)> {
        self.inner.downcast_any(id)
    }
}

impl<P: Clone> Clone for CapabilityOverrideProvider<P> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            resolver: Arc::clone(&self.resolver),
        }
    }
}

impl<P> fmt::Debug for CapabilityOverrideProvider<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CapabilityOverrideProvider")
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stage, not the kind, says whether a refusal reached the
    /// transport: a provider's own refusal did.
    #[test]
    fn a_refusals_stage_says_whether_it_was_sent() {
        let before = crate::Degradations::new("demo", UnsupportedPolicy::Warn)
            .unsupported(RequestField::Seed, "no seed");
        assert!(!reached_transport(&before));
        assert!(reached_transport(&Error::unsupported_by_provider(
            None, "HTTP 400"
        )));
        assert!(reached_transport(&Error::unsupported("unknown stage")));
        assert!(!reached_transport(&Error::configuration("no key")));
        assert!(reached_transport(&Error::rate_limited("HTTP 429")));
    }
    use crate::mock::MockProvider;
    use crate::{
        ReasoningEffort, RequestField, ToolDefinition, UnsupportedPolicy, ValidationCoverage,
    };

    fn tools_request() -> ChatRequest {
        ChatRequest::new("alias")
            .user("q")
            .tools(vec![ToolDefinition::new(
                "lookup",
                serde_json::json!({"type": "object"}),
            )])
    }

    fn reasoning_request() -> ChatRequest {
        ChatRequest::new("alias")
            .user("q")
            .reasoning(ReasoningEffort::High)
    }

    /// A provider that answers `Unknown` for everything, and checks
    /// requests with the default (capability) validation.
    fn unknown() -> MockProvider {
        MockProvider::build(|b| b.text("a").text("b"))
    }

    #[tokio::test]
    async fn a_downgrade_drops_the_hint_before_forwarding() {
        let inner = unknown().with_chat_capability(
            ChatCapability::ReasoningConfig,
            CapabilitySupport::Supported,
        );
        let provider = CapabilityOverrideProvider::new(inner, |_: &str, c| {
            (c == ChatCapability::ReasoningConfig).then_some(CapabilitySupport::Unsupported)
        });
        assert_eq!(
            provider.chat_capability("alias", ChatCapability::ReasoningConfig),
            CapabilitySupport::Unsupported
        );
        let response = provider.chat(&reasoning_request()).await.unwrap();
        assert_eq!(response.warnings.len(), 1);
        assert_eq!(
            response.warnings[0].subject,
            Subject::Field {
                field: RequestField::Reasoning
            }
        );
        assert!(provider.inner().requests()[0].reasoning.is_none());

        let validation = provider
            .validate_chat_request(&reasoning_request())
            .unwrap();
        assert_eq!(validation.warnings(), response.warnings);

        let strict = reasoning_request().unsupported_policy(UnsupportedPolicy::Error);
        assert!(matches!(
            provider.validate_chat_request(&strict),
            Err(Error::Unsupported { .. })
        ));
        assert!(matches!(
            provider.chat(&strict).await,
            Err(Error::Unsupported { .. })
        ));
    }

    #[test]
    fn a_settled_unknown_is_no_longer_relied_on() {
        let request = tools_request();
        let bare = unknown().validate_chat_request(&request).unwrap();
        assert_eq!(bare.relied_on_unknown().len(), 1);

        let provider = CapabilityOverrideProvider::new(unknown(), |_: &str, c| {
            (c == ChatCapability::ToolCalls).then_some(CapabilitySupport::Supported)
        });
        let validation = provider.validate_chat_request(&request).unwrap();
        assert!(validation.relied_on_unknown().is_empty(), "{validation:?}");
        // The wrapped provider's coverage.
        assert_eq!(validation.coverage(), ValidationCoverage::Capabilities);

        // Settled the other way: refused before forwarding.
        let provider = CapabilityOverrideProvider::new(unknown(), |_: &str, c| {
            (c == ChatCapability::ToolCalls).then_some(CapabilitySupport::Unsupported)
        });
        assert!(matches!(
            provider.validate_chat_request(&request),
            Err(Error::Unsupported { .. })
        ));
    }

    #[tokio::test]
    async fn an_upgrade_of_an_unsupported_answer_is_refused_and_reported() {
        let inner = unknown().with_chat_capability(
            ChatCapability::ReasoningConfig,
            CapabilitySupport::Unsupported,
        );
        let provider =
            CapabilityOverrideProvider::new(inner, |_: &str, _| Some(CapabilitySupport::Supported));
        assert_eq!(
            provider.chat_capability("alias", ChatCapability::ReasoningConfig),
            CapabilitySupport::Unsupported
        );
        let response = provider.chat(&reasoning_request()).await.unwrap();
        let kinds: Vec<_> = response.warnings.iter().map(|w| w.kind).collect();
        assert_eq!(kinds, [WarningKind::Unsupported, WarningKind::Other]);
        assert_eq!(
            response.warnings[1].code.as_deref(),
            Some("other.capability_override")
        );
        assert!(
            response.warnings[1]
                .message
                .contains("with_chat_capabilities")
        );
        let validation = provider
            .validate_chat_request(&reasoning_request())
            .unwrap();
        assert_eq!(validation.warnings(), response.warnings);

        // A request that does not rely on it gets no warning.
        let plain = ChatRequest::new("alias").user("q");
        assert!(provider.chat(&plain).await.unwrap().warnings.is_empty());
    }

    #[test]
    fn stream_validation_records_streaming_answers() {
        let request = ChatRequest::new("alias").user("q");
        let provider = CapabilityOverrideProvider::new(unknown(), |_: &str, c| {
            (c == ChatCapability::Streaming).then_some(CapabilitySupport::Supported)
        });
        let validation = provider.validate_chat_stream_request(&request).unwrap();
        let unknown: Vec<_> = validation
            .relied_on_unknown()
            .iter()
            .map(|(c, _)| *c)
            .collect();
        assert_eq!(unknown, [ChatCapability::NativeStreaming]);
    }

    /// The override wrapper's stream path: the adapter emits its warnings,
    /// the wrapper only its own, each once.
    #[cfg(feature = "warning-events")]
    #[test]
    fn layers_emit_each_warning_once() {
        use crate::mock::{ChatResponseBuilder, MockStreamingProvider};
        use crate::warning::capture::warning_events;

        let inner = MockStreamingProvider::from_responses([ChatResponseBuilder::new()
            .text("ok")
            .warning(Warning::unsupported(RequestField::Seed, "seed ignored"))
            .build()]);
        let provider = CapabilityOverrideProvider::new(inner, |_: &str, c| {
            (c == ChatCapability::ReasoningConfig).then_some(CapabilitySupport::Unsupported)
        });
        let request = reasoning_request().seed(1);
        let (collected, mut events) = warning_events(async {
            let stream = provider.chat_stream(&request).await.unwrap();
            crate::ChatStreamExt::collect_response(stream)
                .await
                .unwrap()
        });
        assert_eq!(collected.warnings.len(), 2);
        events.sort();
        assert_eq!(
            events,
            [
                "reasoning configuration is not supported by `mock_stream`; `reasoning` was not sent",
                "seed ignored"
            ]
        );
    }
}
