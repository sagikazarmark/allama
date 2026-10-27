use std::borrow::Cow;
use std::fmt;

use crate::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, ChatValidation,
    DecisionProvider, DecisionRequest, DecisionResponse, EmbeddingCapability, EmbeddingLimits,
    EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, EmbeddingValidation, ProviderIdentity,
    Result, UnsupportedPolicy,
};

/// Sets a provider-level default [`UnsupportedPolicy`]: a request that
/// leaves its [`unsupported_policy`](ChatRequest::unsupported_policy) unset
/// gets `policy`.
///
/// A request that sets its own policy keeps it. Use it to make every call to
/// a provider strict (or lenient) without touching each request, for example
/// when the application, not the caller, decides whether a dropped hint is
/// acceptable:
///
/// ```
/// # #[cfg(feature = "mock")] {
/// use allama::mock::MockProvider;
/// use allama::wrappers::UnsupportedPolicyProvider;
/// use allama::UnsupportedPolicy;
///
/// let strict = UnsupportedPolicyProvider::new(MockProvider::with_text("ok"), UnsupportedPolicy::Error);
/// # let _ = strict; }
/// ```
///
/// It applies to chat and embedding requests, in calls and in validation.
/// Decision requests have no hints and no policy, so they pass through.
/// Everything else is forwarded unchanged: it is a transparent wrapper.
pub struct UnsupportedPolicyProvider<P> {
    inner: P,
    policy: UnsupportedPolicy,
}

impl<P> UnsupportedPolicyProvider<P> {
    /// Wrap `inner`, giving requests without a policy `policy`.
    #[must_use]
    pub fn new(inner: P, policy: UnsupportedPolicy) -> Self {
        Self { inner, policy }
    }

    /// The default policy.
    #[must_use]
    pub fn policy(&self) -> UnsupportedPolicy {
        self.policy
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

    fn chat_request<'a>(&self, request: &'a ChatRequest) -> Cow<'a, ChatRequest> {
        if request.unsupported_policy.is_some() {
            return Cow::Borrowed(request);
        }
        let mut request = request.clone();
        request.unsupported_policy = Some(self.policy);
        Cow::Owned(request)
    }

    fn embedding_request<'a>(&self, request: &'a EmbeddingRequest) -> Cow<'a, EmbeddingRequest> {
        if request.unsupported_policy.is_some() {
            return Cow::Borrowed(request);
        }
        let mut request = request.clone();
        request.unsupported_policy = Some(self.policy);
        Cow::Owned(request)
    }
}

impl<P: ChatProvider> ChatProvider for UnsupportedPolicyProvider<P> {
    type Stream = P::Stream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        self.inner.chat(&self.chat_request(request)).await
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
        self.inner.chat_stream(&self.chat_request(request)).await
    }

    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        self.inner.chat_capability(model, capability)
    }

    fn validate_chat_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        self.inner
            .validate_chat_request(&self.chat_request(request))
    }

    fn validate_chat_stream_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        self.inner
            .validate_chat_stream_request(&self.chat_request(request))
    }
}

impl<P: EmbeddingProvider> EmbeddingProvider for UnsupportedPolicyProvider<P> {
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        self.inner.embed(&self.embedding_request(request)).await
    }

    fn embedding_capability(
        &self,
        model: &str,
        capability: EmbeddingCapability,
    ) -> CapabilitySupport {
        self.inner.embedding_capability(model, capability)
    }

    fn embedding_limits(&self, model: &str) -> EmbeddingLimits {
        self.inner.embedding_limits(model)
    }

    fn validate_embedding_request(
        &self,
        request: &EmbeddingRequest,
    ) -> Result<EmbeddingValidation> {
        self.inner
            .validate_embedding_request(&self.embedding_request(request))
    }
}

impl<P: DecisionProvider> DecisionProvider for UnsupportedPolicyProvider<P> {
    async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse> {
        self.inner.decide(request).await
    }

    super::forward_decision_questions!(inner);
}

impl<P: ProviderIdentity> ProviderIdentity for UnsupportedPolicyProvider<P> {
    super::forward_provider_identity!(inner);
}

impl<P: Clone> Clone for UnsupportedPolicyProvider<P> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            policy: self.policy,
        }
    }
}

impl<P> fmt::Debug for UnsupportedPolicyProvider<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UnsupportedPolicyProvider")
            .field("policy", &self.policy)
            .finish_non_exhaustive()
    }
}

impl<P: crate::ExtractionProvider> crate::ExtractionProvider for UnsupportedPolicyProvider<P> {
    async fn extract_values(
        &self,
        request: &crate::ExtractionRequest,
    ) -> Result<crate::ExtractionResponse> {
        self.inner.extract_values(request).await
    }

    super::forward_extraction_questions!(inner);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::{MockEmbeddingProvider, MockProvider};
    use crate::{Error, ReasoningEffort, WarningKind};

    fn no_reasoning() -> MockProvider {
        MockProvider::build(|b| b.text("a").text("b")).with_chat_capability(
            ChatCapability::ReasoningConfig,
            CapabilitySupport::Unsupported,
        )
    }

    #[tokio::test]
    async fn requests_without_a_policy_inherit_the_providers() {
        let request = ChatRequest::new("m")
            .user("q")
            .reasoning(ReasoningEffort::High);
        let strict = UnsupportedPolicyProvider::new(no_reasoning(), UnsupportedPolicy::Error);
        assert!(matches!(
            strict.validate_chat_request(&request),
            Err(Error::Unsupported { .. })
        ));
        assert!(matches!(
            strict.validate_chat_stream_request(&request),
            Err(Error::Unsupported { .. })
        ));
        assert!(matches!(
            strict.chat(&request).await,
            Err(Error::Unsupported { .. })
        ));

        // A request's own policy wins.
        let lenient = request.unsupported_policy(UnsupportedPolicy::Warn);
        let validation = strict.validate_chat_request(&lenient).unwrap();
        assert_eq!(validation.warnings().len(), 1);
        let response = strict.chat(&lenient).await.unwrap();
        assert_eq!(response.warnings[0].kind, WarningKind::Unsupported);
    }

    #[tokio::test]
    async fn without_any_policy_a_request_warns() {
        let request = ChatRequest::new("m")
            .user("q")
            .reasoning(ReasoningEffort::High);
        assert_eq!(request.unsupported_policy, None);
        let response = no_reasoning().chat(&request).await.unwrap();
        assert_eq!(response.warnings.len(), 1);
    }

    #[test]
    fn embeddings_inherit_too() {
        let request = EmbeddingRequest::new("e")
            .input("x")
            .input_type(crate::EmbeddingInputType::Query);
        let provider = MockEmbeddingProvider::empty().with_embedding_capability(
            EmbeddingCapability::InputType,
            CapabilitySupport::Unsupported,
        );
        assert!(provider.validate_embedding_request(&request).is_ok());
        let strict = UnsupportedPolicyProvider::new(provider, UnsupportedPolicy::Error);
        assert!(strict.validate_embedding_request(&request).is_err());
    }
}
