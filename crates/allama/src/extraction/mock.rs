use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::{
    CapabilitySupport, Error, ExtractionCapability, ExtractionLimits, ExtractionProvider,
    ExtractionRequest, ExtractionResponse, ProviderIdentity, Result,
};

/// Deterministic extraction provider for tests.
///
/// Returns canned responses (or errors) in sequence, records every request,
/// and exposes provider identity plus capability and limit overrides.
/// Requests are checked with
/// [`validate_extraction_request`](ExtractionProvider::validate_extraction_request)
/// first, as a real provider would.
///
/// ```
/// use allama::extraction::{Answer, Field, NumberAnswer};
/// use allama::mock::MockExtractionProvider;
/// use allama::{ExtractionProvider, ExtractionRequest, ExtractionResponse};
///
/// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
/// let provider = MockExtractionProvider::with_response(ExtractionResponse::new([(
///     "total",
///     Answer::from(NumberAnswer::new(Some(1080.0))),
/// )]));
/// let request = ExtractionRequest::new("clairic", "Total due: $1,080").field("total", Field::number());
/// let response = provider.extract_values(&request).await?;
/// assert_eq!(response.number("total").unwrap().value, Some(1080.0));
/// # Ok::<_, allama::Error>(()) }).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MockExtractionProvider {
    state: Arc<Mutex<MockExtractionState>>,
    capabilities: HashMap<ExtractionCapability, CapabilitySupport>,
    limits: ExtractionLimits,
    provider_name: Cow<'static, str>,
}

#[derive(Debug)]
struct MockExtractionState {
    responses: VecDeque<Result<ExtractionResponse>>,
    requests: Vec<ExtractionRequest>,
}

impl MockExtractionProvider {
    /// Create a mock extraction provider with no queued responses.
    #[must_use]
    pub fn empty() -> Self {
        Self::new(std::iter::empty::<Result<ExtractionResponse>>())
    }

    /// Create a mock extraction provider from an ordered queue of responses.
    #[must_use]
    pub fn new<I>(responses: I) -> Self
    where
        I: IntoIterator<Item = Result<ExtractionResponse>>,
    {
        Self {
            state: Arc::new(Mutex::new(MockExtractionState {
                responses: responses.into_iter().collect(),
                requests: Vec::new(),
            })),
            capabilities: HashMap::new(),
            limits: ExtractionLimits::default(),
            provider_name: Cow::Borrowed("mock"),
        }
    }

    /// Convenience for a single successful response.
    #[must_use]
    pub fn with_response(response: ExtractionResponse) -> Self {
        Self::new([Ok(response)])
    }

    /// Convenience for a single error response.
    #[must_use]
    pub fn with_error(error: Error) -> Self {
        Self::new([Err(error)])
    }

    /// Queue an additional response.
    pub fn push_response(&self, response: Result<ExtractionResponse>) {
        self.state.lock().unwrap().responses.push_back(response);
    }

    /// Override the provider identity name reported by this mock.
    #[must_use]
    pub fn with_provider_name(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        self.provider_name = name.into();
        self
    }

    /// Override support for an extraction capability.
    #[must_use]
    pub fn with_extraction_capability(
        mut self,
        capability: ExtractionCapability,
        support: CapabilitySupport,
    ) -> Self {
        self.capabilities.insert(capability, support);
        self
    }

    /// Report `limits` for every model.
    #[must_use]
    pub fn with_extraction_limits(mut self, limits: ExtractionLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Snapshot of every request the mock has received.
    #[must_use]
    pub fn requests(&self) -> Vec<ExtractionRequest> {
        self.state.lock().unwrap().requests.clone()
    }

    /// Number of queued responses remaining.
    #[cfg(test)]
    pub(crate) fn pending_responses(&self) -> usize {
        self.state.lock().unwrap().responses.len()
    }
}

impl ProviderIdentity for MockExtractionProvider {
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

impl ExtractionProvider for MockExtractionProvider {
    async fn extract_values(&self, request: &ExtractionRequest) -> Result<ExtractionResponse> {
        let mut prepared = crate::adapter::extraction::prepare(self, request)?;
        let response = {
            let mut state = self.state.lock().unwrap();
            state.requests.push(request.clone());
            state.responses.pop_front()
        };
        prepared.mark_sent();
        match response {
            Some(response) => Ok(prepared.finish(response?)),
            None => Err(Error::unexpected_response(format!(
                "mock extraction provider '{}' has no queued responses",
                self.provider_name
            ))
            .authored()),
        }
    }

    fn extraction_capability(
        &self,
        _model: &str,
        capability: ExtractionCapability,
    ) -> CapabilitySupport {
        self.capabilities
            .get(&capability)
            .copied()
            .unwrap_or(CapabilitySupport::Unknown)
    }

    fn extraction_limits(&self, _model: &str) -> ExtractionLimits {
        self.limits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extraction::{Answer, Field, TextAnswer};

    fn request() -> ExtractionRequest {
        ExtractionRequest::new("m", "x").field("f", Field::text())
    }

    #[tokio::test]
    async fn returns_queued_responses_and_records_requests() {
        let provider = MockExtractionProvider::with_response(ExtractionResponse::new([(
            "f",
            Answer::from(TextAnswer::new(Some("x"))),
        )]));
        let response = provider.extract_values(&request()).await.unwrap();
        assert_eq!(response.text("f").unwrap().value.as_deref(), Some("x"));
        assert_eq!(provider.requests(), vec![request()]);
        assert_eq!(provider.pending_responses(), 0);

        let err = provider.extract_values(&request()).await.unwrap_err();
        assert!(matches!(err, Error::UnexpectedResponse { .. }));
    }

    #[tokio::test]
    async fn invalid_requests_fail_before_consuming_a_response() {
        let provider = MockExtractionProvider::with_error(Error::auth("bad"));
        let invalid = ExtractionRequest::new("m", "x");
        assert!(matches!(
            provider.extract_values(&invalid).await,
            Err(Error::InvalidRequest { .. })
        ));
        assert!(provider.requests().is_empty());
        assert!(matches!(
            provider.extract_values(&request()).await,
            Err(Error::Auth { .. })
        ));
    }
}
