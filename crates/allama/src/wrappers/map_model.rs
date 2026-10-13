use std::any::{Any, TypeId};
use std::borrow::Cow;
use std::fmt;

use crate::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, ChatValidation,
    DecisionCapability, DecisionLimits, DecisionProvider, DecisionRequest, DecisionResponse,
    DecisionValidation, EmbeddingCapability, EmbeddingLimits, EmbeddingProvider, EmbeddingRequest,
    EmbeddingResponse, EmbeddingValidation, ProviderIdentity, Result,
};

/// Rewrites model ids before requests reach the wrapped provider.
///
/// Use it when the caller's model id is not the one this provider should
/// receive: to give each side of a
/// [`FallbackProvider`](super::FallbackProvider) a model from its own
/// catalog, or to resolve application aliases such as `"fast"`.
///
/// ```
/// # #[cfg(feature = "mock")] {
/// use allama::wrappers::{FallbackProvider, MapModelProvider};
/// use allama::mock::MockProvider;
/// # let (anthropic, openai) = (MockProvider::with_text("a"), MockProvider::with_text("b"));
///
/// // Callers ask for Claude models; the OpenAI fallback gets its own.
/// let provider = FallbackProvider::new(
///     anthropic,
///     MapModelProvider::new(openai, |model: &str| match model {
///         "claude-opus-4-1" => "gpt-5.5".into(),
///         _ => "gpt-5-mini".into(),
///     }),
/// );
/// # let _ = provider; }
/// ```
///
/// # Writing the mapping
///
/// The mapping returns a `Cow<'_, str>`, so it passes the caller's id on
/// without copying it (`Cow::Borrowed(model)`, or `model.into()`). Convert
/// what else it returns with `.into()` or `Cow::Owned`: a `String` such as
/// a model from configuration (`configured.clone().into()`) or a
/// `&'static str`. A `String` returned as it is fails to compile with
/// "expected `Cow<'_, str>`, found `String`". Annotate the argument
/// (`|model: &str|`), so the closure takes any borrowed id:
///
/// ```
/// # #[cfg(feature = "mock")] {
/// use std::borrow::Cow;
/// use std::collections::HashMap;
///
/// use allama::mock::MockProvider;
/// use allama::wrappers::MapModelProvider;
///
/// // Application aliases from configuration; other ids pass through.
/// let aliases = HashMap::from([("fast".to_owned(), "gpt-5-mini".to_owned())]);
/// let provider = MapModelProvider::new(MockProvider::with_text("hi"), move |model: &str| {
///     match aliases.get(model) {
///         Some(target) => Cow::Owned(target.clone()),
///         None => Cow::Borrowed(model),
///     }
/// });
/// assert_eq!(provider.map_model("fast"), "gpt-5-mini");
/// assert_eq!(provider.map_model("gpt-5.5"), "gpt-5.5");
/// # }
/// ```
///
/// # Guarantees
///
/// - **Consistent.** The mapping applies to every request the wrapper
///   forwards (`chat`, `chat_stream`, `embed`, `decide`) and to every question
///   asked by model id (capabilities, limits, validation), so the answers
///   describe the model that is actually sent. Otherwise it is a transparent
///   wrapper (see `allama_conformance::assert_transparent_model_mapping`).
/// - **Every model id is mapped.** Chat, embedding, and decision requests
///   all pass through the same mapping. A mapping that ignores its input
///   sends that one model for every operation; match on the ids you mean to
///   rewrite when the provider serves more than one.
/// - **Responses are untouched.** [`ChatResponse::model`] reports the model
///   the provider says served the call.
/// - **The caller's request is never changed.** Requests are copied only
///   when the mapping changes the model.
///
/// The mapping runs on every call and every capability question, so keep it
/// cheap and deterministic. Change other request fields with
/// [`MapRequestProvider`](super::MapRequestProvider), which keeps the model.
#[derive(Clone)]
pub struct MapModelProvider<P, F> {
    inner: P,
    map: F,
}

impl<P, F> MapModelProvider<P, F>
where
    F: Fn(&str) -> Cow<'_, str> + Send + Sync,
{
    /// Wrap `inner`, sending `map(model)` wherever the caller names `model`.
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

    /// The model id the wrapped provider receives for `model`.
    #[must_use]
    pub fn map_model<'a>(&self, model: &'a str) -> Cow<'a, str> {
        (self.map)(model)
    }

    /// `request` with its model mapped, copied only when the model changes.
    fn mapped<'r, R: Clone>(
        &self,
        request: &'r R,
        model: &str,
        set_model: fn(&mut R, String),
    ) -> Cow<'r, R> {
        let mapped = self.map_model(model);
        if mapped == model {
            return Cow::Borrowed(request);
        }
        let mut request = request.clone();
        set_model(&mut request, mapped.into_owned());
        Cow::Owned(request)
    }

    fn chat_request<'r>(&self, request: &'r ChatRequest) -> Cow<'r, ChatRequest> {
        self.mapped(request, &request.model, |r, model| r.model = model)
    }

    fn embedding_request<'r>(&self, request: &'r EmbeddingRequest) -> Cow<'r, EmbeddingRequest> {
        self.mapped(request, &request.model, |r, model| r.model = model)
    }

    fn decision_request<'r>(&self, request: &'r DecisionRequest) -> Cow<'r, DecisionRequest> {
        self.mapped(request, &request.model, |r, model| r.model = model)
    }
}

impl<P, F> ChatProvider for MapModelProvider<P, F>
where
    P: ChatProvider,
    F: Fn(&str) -> Cow<'_, str> + Send + Sync,
{
    type Stream = P::Stream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        self.inner.chat(&self.chat_request(request)).await
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
        self.inner.chat_stream(&self.chat_request(request)).await
    }

    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        self.inner
            .chat_capability(&self.map_model(model), capability)
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

impl<P, F> EmbeddingProvider for MapModelProvider<P, F>
where
    P: EmbeddingProvider,
    F: Fn(&str) -> Cow<'_, str> + Send + Sync,
{
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        self.inner.embed(&self.embedding_request(request)).await
    }

    fn embedding_capability(
        &self,
        model: &str,
        capability: EmbeddingCapability,
    ) -> CapabilitySupport {
        self.inner
            .embedding_capability(&self.map_model(model), capability)
    }

    fn embedding_limits(&self, model: &str) -> EmbeddingLimits {
        self.inner.embedding_limits(&self.map_model(model))
    }

    fn validate_embedding_request(
        &self,
        request: &EmbeddingRequest,
    ) -> Result<EmbeddingValidation> {
        self.inner
            .validate_embedding_request(&self.embedding_request(request))
    }
}

impl<P, F> DecisionProvider for MapModelProvider<P, F>
where
    P: DecisionProvider,
    F: Fn(&str) -> Cow<'_, str> + Send + Sync,
{
    async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse> {
        self.inner.decide(&self.decision_request(request)).await
    }

    fn decision_capability(
        &self,
        model: &str,
        capability: DecisionCapability,
    ) -> CapabilitySupport {
        self.inner
            .decision_capability(&self.map_model(model), capability)
    }

    fn decision_limits(&self, model: &str) -> DecisionLimits {
        self.inner.decision_limits(&self.map_model(model))
    }

    fn validate_decision_request(&self, request: &DecisionRequest) -> Result<DecisionValidation> {
        self.inner
            .validate_decision_request(&self.decision_request(request))
    }
}

impl<P, F> ProviderIdentity for MapModelProvider<P, F>
where
    P: ProviderIdentity,
    F: Send + Sync,
{
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

impl<P, F> fmt::Debug for MapModelProvider<P, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MapModelProvider").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::{Answer, Question, YesNoAnswer};
    use crate::mock::{MockDecisionProvider, MockEmbeddingProvider, MockProvider};
    use crate::{ChatStreamExt, Error, ProviderIdentityExt};

    fn aliases(model: &str) -> Cow<'_, str> {
        match model {
            "fast" => "provider-small".into(),
            other => other.into(),
        }
    }

    /// Answers capability and validation questions by model id.
    struct ByModel;

    impl ProviderIdentity for ByModel {
        fn provider_name(&self) -> &str {
            "by-model"
        }

        fn reads_namespace(&self, namespace: &str) -> bool {
            namespace == "by-model"
        }

        fn downcast_any(&self, id: TypeId) -> Option<&(dyn Any + Send + Sync)> {
            crate::expose(self, id)
        }
    }

    impl ChatProvider for ByModel {
        type Stream = crate::ChatStream;

        async fn chat(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            Err(Error::unsupported("questions only"))
        }

        async fn chat_stream(&self, _request: &ChatRequest) -> Result<Self::Stream> {
            Err(Error::unsupported("questions only"))
        }

        fn chat_capability(&self, model: &str, _capability: ChatCapability) -> CapabilitySupport {
            if model == "provider-small" {
                CapabilitySupport::Supported
            } else {
                CapabilitySupport::Unsupported
            }
        }

        fn validate_chat_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
            if request.model == "provider-small" {
                Ok(ChatValidation::new(crate::ValidationCoverage::Capabilities))
            } else {
                Err(Error::model_not_found(request.model.clone()))
            }
        }
    }

    #[tokio::test]
    async fn chat_and_stream_send_the_mapped_model() {
        let mock = MockProvider::build(|b| b.text("a").text("b"));
        let provider = MapModelProvider::new(mock.clone(), aliases);
        let request = ChatRequest::new("fast").user("hi").temperature(0.3);

        provider.chat(&request).await.unwrap();
        let sent = mock.last_request().unwrap();
        assert_eq!(sent.model, "provider-small");
        assert_eq!(sent.temperature, Some(0.3), "other fields are kept");
        assert_eq!(request.model, "fast", "caller request is unchanged");

        let response = provider
            .chat_stream(&request)
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        assert_eq!(response.text().as_deref(), Some("b"));
        assert_eq!(mock.last_request().unwrap().model, "provider-small");
    }

    #[tokio::test]
    async fn unmapped_models_pass_through() {
        let mock = MockProvider::with_text("a");
        let provider = MapModelProvider::new(mock.clone(), aliases);
        provider
            .chat(&ChatRequest::new("exact").user("hi"))
            .await
            .unwrap();
        assert_eq!(mock.last_request().unwrap().model, "exact");
        assert!(matches!(
            provider.map_model("exact"),
            Cow::Borrowed("exact")
        ));
    }

    #[test]
    fn questions_are_asked_about_the_mapped_model() {
        let provider = MapModelProvider::new(ByModel, aliases);
        assert_eq!(
            provider.chat_capability("fast", ChatCapability::ToolCalls),
            CapabilitySupport::Supported
        );
        assert_eq!(
            provider.chat_capability("slow", ChatCapability::ToolCalls),
            CapabilitySupport::Unsupported
        );
        assert!(
            provider
                .validate_chat_request(&ChatRequest::new("fast").user("hi"))
                .is_ok()
        );
        assert!(
            provider
                .validate_chat_request(&ChatRequest::new("slow").user("hi"))
                .is_err()
        );
        assert_eq!(provider.provider_name(), "by-model");
        assert!(provider.downcast_ref::<ByModel>().is_some());
    }

    #[tokio::test]
    async fn embeddings_use_the_mapped_model() {
        let mock = MockEmbeddingProvider::with_vectors(vec![vec![1.0]]);
        let provider = MapModelProvider::new(mock.clone(), aliases);
        provider
            .embed(&EmbeddingRequest::new("fast").inputs(["a"]))
            .await
            .unwrap();
        assert_eq!(mock.requests()[0].model, "provider-small");
    }

    #[tokio::test]
    async fn decisions_use_the_mapped_model() {
        let mock = MockDecisionProvider::with_response(DecisionResponse::new([(
            "q",
            Answer::from(YesNoAnswer::new(0.9)),
        )]));
        let provider = MapModelProvider::new(mock.clone(), aliases);
        let request = DecisionRequest::new("fast", "x").question("q", Question::yes_no("?"));
        provider.decide(&request).await.unwrap();
        assert_eq!(mock.requests()[0].model, "provider-small");
    }

    #[test]
    fn closures_can_pin_one_model() {
        let provider = MapModelProvider::new(MockProvider::empty(), |_: &str| "pinned".into());
        assert_eq!(provider.map_model("anything"), "pinned");

        let configured = String::from("from-config");
        let provider = MapModelProvider::new(MockProvider::empty(), move |_: &str| {
            configured.clone().into()
        });
        assert_eq!(provider.map_model("anything"), "from-config");
    }
}
