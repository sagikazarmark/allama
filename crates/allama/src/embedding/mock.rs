use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::{
    CapabilitySupport, EmbeddingCapability, EmbeddingLimits, EmbeddingProvider, EmbeddingRequest,
    EmbeddingResponse, Error, ProviderIdentity, Result,
};

/// Deterministic embedding provider for tests.
///
/// Returns canned responses (or errors) in sequence, records every request,
/// and exposes provider identity plus embedding capability overrides.
///
/// [`with_vectors`](Self::with_vectors) answers with one vector per input,
/// like a real provider. Responses queued with [`new`](Self::new) or
/// [`push_response`](Self::push_response) are returned exactly as given,
/// whatever the input count, which is how a test simulates a provider that
/// returns the wrong number of vectors.
#[derive(Debug, Clone)]
pub struct MockEmbeddingProvider {
    state: Arc<Mutex<MockEmbeddingState>>,
    embedding_capabilities: HashMap<EmbeddingCapability, CapabilitySupport>,
    limits: EmbeddingLimits,
    provider_name: Cow<'static, str>,
}

#[derive(Debug)]
struct MockEmbeddingState {
    responses: VecDeque<Scripted>,
    requests: Vec<EmbeddingRequest>,
}

/// A queued answer.
#[derive(Debug)]
enum Scripted {
    /// Returned as given (boxed: a response is far larger than a vector
    /// list).
    Exact(Box<Result<EmbeddingResponse>>),
    /// One vector per input, cycling through these.
    PerInput(Vec<Vec<f32>>),
}

impl MockEmbeddingProvider {
    /// Create a mock embedding provider with no queued responses.
    #[must_use]
    pub fn empty() -> Self {
        Self::new(std::iter::empty::<Result<EmbeddingResponse>>())
    }

    /// Create a mock embedding provider from an ordered queue of responses.
    #[must_use]
    pub fn new<I>(responses: I) -> Self
    where
        I: IntoIterator<Item = Result<EmbeddingResponse>>,
    {
        Self {
            state: Arc::new(Mutex::new(MockEmbeddingState {
                responses: responses
                    .into_iter()
                    .map(|response| Scripted::Exact(Box::new(response)))
                    .collect(),
                requests: Vec::new(),
            })),
            embedding_capabilities: HashMap::new(),
            limits: EmbeddingLimits::new(),
            provider_name: Cow::Borrowed("mock"),
        }
    }

    /// A single successful response with one vector per input of the
    /// request it answers, cycling through `vectors` (so `vectors` longer
    /// than the batch is truncated, and shorter is repeated). With no
    /// `vectors`, the response is empty.
    #[must_use]
    pub fn with_vectors(vectors: Vec<Vec<f32>>) -> Self {
        let provider = Self::empty();
        provider
            .state
            .lock()
            .unwrap()
            .responses
            .push_back(Scripted::PerInput(vectors));
        provider
    }

    /// Convenience for a single error response.
    #[must_use]
    pub fn with_error(error: Error) -> Self {
        Self::new([Err(error)])
    }

    /// Queue an additional response, returned as given, without mutating
    /// existing queue entries.
    pub fn push_response(&self, response: Result<EmbeddingResponse>) {
        self.state
            .lock()
            .unwrap()
            .responses
            .push_back(Scripted::Exact(Box::new(response)));
    }

    /// Override the provider identity name reported by this mock.
    #[must_use]
    pub fn with_provider_name(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        self.provider_name = name.into();
        self
    }

    /// Override support for an embedding capability.
    #[must_use]
    pub fn with_embedding_capability(
        mut self,
        capability: EmbeddingCapability,
        support: CapabilitySupport,
    ) -> Self {
        self.embedding_capabilities.insert(capability, support);
        self
    }

    /// Report a maximum number of inputs per request (default: no limit).
    /// The mock does not enforce it.
    #[must_use]
    pub fn with_max_inputs_per_call(mut self, max: usize) -> Self {
        self.limits.max_inputs_per_call = Some(max);
        self
    }

    /// Snapshot of every request the mock has received.
    #[must_use]
    pub fn requests(&self) -> Vec<EmbeddingRequest> {
        self.state.lock().unwrap().requests.clone()
    }
}

impl ProviderIdentity for MockEmbeddingProvider {
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

impl EmbeddingProvider for MockEmbeddingProvider {
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        let mut state = self.state.lock().unwrap();
        state.requests.push(request.clone());
        match state.responses.pop_front() {
            Some(Scripted::Exact(response)) => *response,
            Some(Scripted::PerInput(vectors)) => Ok(EmbeddingResponse::new(
                vectors
                    .iter()
                    .cycle()
                    .take(request.inputs.len())
                    .cloned()
                    .collect(),
            )),
            None => Err(Error::unexpected_response(format!(
                "mock embedding provider '{}' has no queued responses",
                self.provider_name
            ))
            .authored()),
        }
    }

    fn embedding_capability(
        &self,
        _model: &str,
        capability: EmbeddingCapability,
    ) -> CapabilitySupport {
        self.embedding_capabilities
            .get(&capability)
            .copied()
            .unwrap_or(CapabilitySupport::Unknown)
    }

    fn embedding_limits(&self, _model: &str) -> EmbeddingLimits {
        self.limits
    }
}

#[cfg(test)]
mod embedding_mock_tests {
    use super::*;
    use crate::{
        CapabilitySupport, EmbeddingCapability, EmbeddingProvider, EmbeddingRequest,
        EmbeddingResponse, Error, ProviderIdentity,
    };

    #[tokio::test]
    async fn mock_embedding_provider_returns_queued_response() {
        let provider = MockEmbeddingProvider::with_vectors(vec![vec![0.1, 0.2]]);
        let request = EmbeddingRequest::new("mock-embed").input("hello");
        let response = provider.embed(&request).await.unwrap();
        assert_eq!(response.embeddings, vec![vec![0.1, 0.2]]);

        let requests = provider.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0].inputs,
            vec![crate::EmbeddingInput::text("hello")]
        );
    }

    #[tokio::test]
    async fn with_vectors_answers_one_vector_per_input() {
        let request = EmbeddingRequest::new("m").inputs(["a", "b", "c"]);
        let cycled = MockEmbeddingProvider::with_vectors(vec![vec![1.0], vec![2.0]])
            .embed(&request)
            .await
            .unwrap();
        assert_eq!(cycled.embeddings, vec![vec![1.0], vec![2.0], vec![1.0]]);

        let truncated = MockEmbeddingProvider::with_vectors(vec![vec![1.0]; 5])
            .embed(&EmbeddingRequest::new("m").input("a"))
            .await
            .unwrap();
        assert_eq!(truncated.embeddings, vec![vec![1.0]]);
    }

    #[tokio::test]
    async fn queued_responses_can_mismatch_the_input_count() {
        let provider = MockEmbeddingProvider::new([Ok(EmbeddingResponse::new(vec![vec![1.0]]))]);
        let response = provider
            .embed(&EmbeddingRequest::new("m").inputs(["a", "b"]))
            .await
            .unwrap();
        assert_eq!(response.embeddings.len(), 1);
    }

    #[tokio::test]
    async fn mock_embedding_provider_returns_queued_error() {
        let provider = MockEmbeddingProvider::with_error(Error::auth("bad"));
        let err = provider
            .embed(&EmbeddingRequest::new("m").input("x"))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Auth { .. }));
    }

    #[tokio::test]
    async fn mock_embedding_provider_returns_responses_in_order() {
        let provider = MockEmbeddingProvider::new([
            Ok(EmbeddingResponse::new(vec![vec![1.0]])),
            Ok(EmbeddingResponse::new(vec![vec![2.0]])),
        ]);
        let first = provider
            .embed(&EmbeddingRequest::new("m").input("a"))
            .await
            .unwrap();
        let second = provider
            .embed(&EmbeddingRequest::new("m").input("b"))
            .await
            .unwrap();
        assert_eq!(first.embeddings, vec![vec![1.0]]);
        assert_eq!(second.embeddings, vec![vec![2.0]]);
    }

    #[tokio::test]
    async fn mock_embedding_provider_reports_exhaustion() {
        let provider = MockEmbeddingProvider::empty();
        let err = provider
            .embed(&EmbeddingRequest::new("m").input("x"))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::UnexpectedResponse { .. }));
    }

    #[test]
    fn mock_embedding_provider_exposes_provider_identity_and_capabilities() {
        let provider = MockEmbeddingProvider::empty()
            .with_provider_name("demo-embed")
            .with_embedding_capability(
                EmbeddingCapability::BatchInput,
                CapabilitySupport::Supported,
            );
        assert_eq!(provider.provider_name(), "demo-embed");
        assert_eq!(
            provider.embedding_capability("m", EmbeddingCapability::BatchInput),
            CapabilitySupport::Supported
        );
        assert_eq!(
            provider.embedding_capability("m", EmbeddingCapability::OutputDimensions),
            CapabilitySupport::Unknown
        );
    }
}
