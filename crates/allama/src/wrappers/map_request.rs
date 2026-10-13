use std::fmt;

use crate::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, ChatValidation,
    DecisionProvider, DecisionRequest, DecisionResponse, EmbeddingProvider, EmbeddingRequest,
    EmbeddingResponse, Error, ProviderIdentity, Result,
};

/// Rewrites chat requests before they reach the wrapped provider.
///
/// The transform runs on an owned copy of each request, so the caller's
/// request is never changed. Use it for normalized policy that applies to
/// every call: defaults, sanitization, redaction, extra system prompts,
/// provider options.
///
/// Put a redaction or sanitization transform **outside**
/// `TracingProvider`, so captured span content shows its result: a span
/// records the request the tracing wrapper was given.
///
/// ```
/// # #[cfg(feature = "mock")] {
/// use allama::wrappers::MapRequestProvider;
/// use allama::{ChatRequest, mock::MockProvider};
///
/// let provider = MapRequestProvider::new(MockProvider::with_text("ok"), |request: &mut ChatRequest| {
///     request.temperature.get_or_insert(0.2);
///     request.system.insert(0, "Answer in English.".into());
/// });
/// # let _ = provider; }
/// ```
///
/// # Guarantees
///
/// - **Capability-preserving.** Capability answers come from the wrapped
///   provider for the caller's model, so a transform must not change what
///   they mean: it must not change the model (rejected with
///   [`Error::InvalidRequest`]) and should not add features the caller did
///   not ask for (tools, response formats, media) on providers that may not
///   support them. Change models with
///   [`MapModelProvider`](super::MapModelProvider) instead, which also asks
///   the capability questions about the model it sends.
/// - **Validated as sent.** [`validate_chat_request`](ChatProvider::validate_chat_request)
///   checks the transformed request.
/// - **Streams are untouched.** The transform applies to the request only;
///   stream events pass through unchanged.
/// - **Embeddings and decisions pass through** unchanged.
#[derive(Clone)]
pub struct MapRequestProvider<P, F> {
    inner: P,
    map: F,
}

impl<P, F> MapRequestProvider<P, F>
where
    F: Fn(&mut ChatRequest) + Send + Sync,
{
    /// Wrap `inner`, applying `map` to every chat request.
    #[must_use]
    pub fn new(inner: P, map: F) -> Self {
        Self { inner, map }
    }

    /// Borrow the wrapped provider.
    #[must_use]
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Unwrap the provider.
    #[must_use]
    pub fn into_inner(self) -> P {
        self.inner
    }

    /// The request the wrapped provider receives for `request`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`] when the transform changed the
    /// model; map models with [`MapModelProvider`](super::MapModelProvider).
    pub fn map_request(&self, request: &ChatRequest) -> Result<ChatRequest> {
        let mut mapped = request.clone();
        (self.map)(&mut mapped);
        if mapped.model != request.model {
            return Err(Error::invalid_request(format!(
                "request transforms must not change the model (`{}` -> `{}`); use MapModelProvider to map models",
                request.model, mapped.model
            ))
            .authored());
        }
        Ok(mapped)
    }
}

impl<P, F> ChatProvider for MapRequestProvider<P, F>
where
    P: ChatProvider,
    F: Fn(&mut ChatRequest) + Send + Sync,
{
    type Stream = P::Stream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let request = self.map_request(request)?;
        self.inner.chat(&request).await
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
        let request = self.map_request(request)?;
        self.inner.chat_stream(&request).await
    }

    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        self.inner.chat_capability(model, capability)
    }

    fn validate_chat_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        self.inner
            .validate_chat_request(&self.map_request(request)?)
    }

    fn validate_chat_stream_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        self.inner
            .validate_chat_stream_request(&self.map_request(request)?)
    }
}

impl<P, F> EmbeddingProvider for MapRequestProvider<P, F>
where
    P: EmbeddingProvider,
    F: Fn(&mut ChatRequest) + Send + Sync,
{
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        self.inner.embed(request).await
    }

    super::forward_embedding_questions!(inner);
}

impl<P, F> DecisionProvider for MapRequestProvider<P, F>
where
    P: DecisionProvider,
    F: Fn(&mut ChatRequest) + Send + Sync,
{
    async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse> {
        self.inner.decide(request).await
    }

    super::forward_decision_questions!(inner);
}

impl<P, F> ProviderIdentity for MapRequestProvider<P, F>
where
    P: ProviderIdentity,
    F: Send + Sync,
{
    super::forward_provider_identity!(inner);
}

impl<P, F> fmt::Debug for MapRequestProvider<P, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MapRequestProvider").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DecisionLimits;
    use crate::mock::{MockEmbeddingProvider, MockProvider};
    use crate::{ChatStreamExt, ProviderIdentityExt, ToolDefinition};
    use serde_json::json;

    fn defaults(request: &mut ChatRequest) {
        request.temperature.get_or_insert(0.2);
        request.system.insert(0, "policy".into());
    }

    #[tokio::test]
    async fn chat_and_stream_send_the_transformed_request() {
        let mock = MockProvider::build(|b| b.text("a").text("b"));
        let provider = MapRequestProvider::new(mock.clone(), defaults);
        let request = ChatRequest::new("m").user("hi").temperature(0.9);

        provider.chat(&request).await.unwrap();
        let sent = mock.last_request().unwrap();
        assert_eq!(sent.temperature, Some(0.9));
        assert_eq!(sent.system.len(), 1);
        assert!(request.system.is_empty(), "caller request is unchanged");

        let response = provider
            .chat_stream(&ChatRequest::new("m").user("hi"))
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        assert_eq!(response.text().as_deref(), Some("b"));
        assert_eq!(mock.last_request().unwrap().temperature, Some(0.2));
    }

    #[tokio::test]
    async fn changing_the_model_is_rejected() {
        let mock = MockProvider::with_text("x");
        let provider = MapRequestProvider::new(mock.clone(), |r: &mut ChatRequest| {
            r.model = "other".into();
        });
        let err = provider
            .chat(&ChatRequest::new("m").user("hi"))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err}");
        assert!(err.to_string().contains("MapModelProvider"), "{err}");
        assert_eq!(mock.call_count(), 0);
    }

    #[test]
    fn validation_sees_the_transformed_request() {
        let mock = MockProvider::empty()
            .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Unsupported);
        let provider = MapRequestProvider::new(mock, |r: &mut ChatRequest| {
            r.tools = vec![ToolDefinition::new("t", json!({"type": "object"}))];
        });
        assert!(
            provider
                .validate_chat_request(&ChatRequest::new("m").user("hi"))
                .is_err()
        );
        assert_eq!(
            provider.chat_capability("m", ChatCapability::ToolCalls),
            CapabilitySupport::Unsupported
        );
        assert!(provider.downcast_ref::<MockProvider>().is_some());
    }

    #[tokio::test]
    async fn decisions_pass_through() {
        use crate::decision::{Answer, Question, YesNoAnswer};
        use crate::mock::MockDecisionProvider;

        let mock = MockDecisionProvider::with_response(DecisionResponse::new([(
            "q",
            Answer::from(YesNoAnswer::new(0.4)),
        )]))
        .with_decision_limits(DecisionLimits::new().max_questions(1));
        let provider = MapRequestProvider::new(mock.clone(), defaults);
        let request = DecisionRequest::new("d", "x").question("q", Question::yes_no("?"));
        let response = provider.decide(&request).await.unwrap();
        assert!(!response.yes_no("q").unwrap().is_yes());
        assert_eq!(mock.requests(), vec![request.clone()]);
        assert_eq!(provider.decision_limits("d").max_questions, Some(1));
        let too_many = request.question("r", Question::yes_no("?"));
        assert!(provider.validate_decision_request(&too_many).is_err());
    }

    #[tokio::test]
    async fn embeddings_pass_through() {
        let provider = MapRequestProvider::new(
            MockEmbeddingProvider::with_vectors(vec![vec![1.0]]),
            defaults,
        );
        let response = provider
            .embed(&EmbeddingRequest::new("e").inputs(["a"]))
            .await
            .unwrap();
        assert_eq!(response.embeddings.len(), 1);
    }
}
