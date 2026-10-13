use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::{
    CapabilitySupport, DecisionCapability, DecisionLimits, DecisionProvider, DecisionRequest,
    DecisionResponse, Error, ProviderIdentity, Result,
};

/// Deterministic decision provider for tests.
///
/// Returns canned responses (or errors) in sequence, records every request,
/// and exposes provider identity plus capability and limit overrides.
/// Requests are checked with
/// [`validate_decision_request`](DecisionProvider::validate_decision_request)
/// first, as a real provider would.
///
/// # Answers for a typed decision
///
/// Build a canned answer from the typed values a [`Decision`] decodes to:
/// [`Choice`], [`Score`] and [`YesNoAnswer`] convert into an [`Answer`], so
/// the option keys and level labels come from the derived types rather
/// than strings. Only the question names, the field names unless renamed,
/// are written out:
///
/// ```
/// # #[cfg(feature = "derive")] {
/// use allama::decision::{Answer, Choice, ChoiceSet, Decision, DecisionProviderExt, YesNoAnswer};
/// use allama::mock::MockDecisionProvider;
/// use allama::DecisionResponse;
///
/// #[derive(Debug, PartialEq, ChoiceSet)]
/// #[decision(rename_all = "lowercase")]
/// enum Topic {
///     Tech,
///     Food,
/// }
///
/// #[derive(Decision)]
/// struct Classified {
///     /// Which topic is this text about?
///     topic: Choice<Topic>,
///     /// Is the author's tone positive?
///     positive: YesNoAnswer,
/// }
///
/// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
/// let provider = MockDecisionProvider::with_response(DecisionResponse::new([
///     ("topic", Answer::from(Choice::new(Topic::Food, [(Topic::Tech, 0.1), (Topic::Food, 0.9)]))),
///     ("positive", YesNoAnswer::new(0.95).into()),
/// ]));
/// let decided = provider.decide_as::<Classified>("nimble", "Crusty sourdough!").await?;
/// assert_eq!(*decided.value.topic.value(), Topic::Food);
/// # Ok::<_, allama::Error>(()) }).unwrap();
/// # }
/// ```
///
/// [`Decision`]: crate::decision::Decision
/// [`Choice`]: crate::decision::Choice
/// [`Score`]: crate::decision::Score
/// [`YesNoAnswer`]: crate::decision::YesNoAnswer
/// [`Answer`]: crate::decision::Answer
#[derive(Debug, Clone)]
pub struct MockDecisionProvider {
    state: Arc<Mutex<MockDecisionState>>,
    decision_capabilities: HashMap<DecisionCapability, CapabilitySupport>,
    decision_limits: DecisionLimits,
    provider_name: Cow<'static, str>,
}

#[derive(Debug)]
struct MockDecisionState {
    responses: VecDeque<Result<DecisionResponse>>,
    requests: Vec<DecisionRequest>,
}

impl MockDecisionProvider {
    /// Create a mock decision provider with no queued responses.
    #[must_use]
    pub fn empty() -> Self {
        Self::new(std::iter::empty::<Result<DecisionResponse>>())
    }

    /// Create a mock decision provider from an ordered queue of responses.
    #[must_use]
    pub fn new<I>(responses: I) -> Self
    where
        I: IntoIterator<Item = Result<DecisionResponse>>,
    {
        Self {
            state: Arc::new(Mutex::new(MockDecisionState {
                responses: responses.into_iter().collect(),
                requests: Vec::new(),
            })),
            decision_capabilities: HashMap::new(),
            decision_limits: DecisionLimits::default(),
            provider_name: Cow::Borrowed("mock"),
        }
    }

    /// Convenience for a single successful response.
    #[must_use]
    pub fn with_response(response: DecisionResponse) -> Self {
        Self::new([Ok(response)])
    }

    /// Convenience for a single error response.
    #[must_use]
    pub fn with_error(error: Error) -> Self {
        Self::new([Err(error)])
    }

    /// Queue an additional response.
    pub fn push_response(&self, response: Result<DecisionResponse>) {
        self.state.lock().unwrap().responses.push_back(response);
    }

    /// Override the provider identity name reported by this mock.
    #[must_use]
    pub fn with_provider_name(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        self.provider_name = name.into();
        self
    }

    /// Override support for a decision capability.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn with_decision_capability(
        mut self,
        capability: DecisionCapability,
        support: CapabilitySupport,
    ) -> Self {
        self.decision_capabilities.insert(capability, support);
        self
    }

    /// Report `limits` for every model.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn with_decision_limits(mut self, limits: DecisionLimits) -> Self {
        self.decision_limits = limits;
        self
    }

    /// Snapshot of every request the mock has received.
    #[must_use]
    pub fn requests(&self) -> Vec<DecisionRequest> {
        self.state.lock().unwrap().requests.clone()
    }

    /// Number of queued responses remaining.
    #[cfg(test)]
    pub(crate) fn pending_responses(&self) -> usize {
        self.state.lock().unwrap().responses.len()
    }
}

impl ProviderIdentity for MockDecisionProvider {
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

impl DecisionProvider for MockDecisionProvider {
    async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse> {
        let mut prepared = crate::adapter::decision::prepare(self, request)?;
        let response = {
            let mut state = self.state.lock().unwrap();
            state.requests.push(request.clone());
            state.responses.pop_front()
        };
        prepared.mark_sent();
        match response {
            Some(response) => Ok(prepared.finish(response?)),
            None => Err(Error::unexpected_response(format!(
                "mock decision provider '{}' has no queued responses",
                self.provider_name
            ))
            .authored()),
        }
    }

    fn decision_capability(
        &self,
        _model: &str,
        capability: DecisionCapability,
    ) -> CapabilitySupport {
        self.decision_capabilities
            .get(&capability)
            .copied()
            .unwrap_or(CapabilitySupport::Unknown)
    }

    fn decision_limits(&self, _model: &str) -> DecisionLimits {
        self.decision_limits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::{Answer, Question, YesNoAnswer};

    fn request() -> DecisionRequest {
        DecisionRequest::new("m", "x").question("q", Question::yes_no("?"))
    }

    #[tokio::test]
    async fn returns_queued_responses_and_records_requests() {
        let provider = MockDecisionProvider::with_response(DecisionResponse::new([(
            "q",
            Answer::from(YesNoAnswer::new(0.7)),
        )]));
        let response = provider.decide(&request()).await.unwrap();
        assert_eq!(response.yes_no("q").unwrap().probability, 0.7);
        assert_eq!(provider.requests(), vec![request()]);
        assert_eq!(provider.pending_responses(), 0);

        let err = provider.decide(&request()).await.unwrap_err();
        assert!(matches!(err, Error::UnexpectedResponse { .. }));
    }

    #[tokio::test]
    async fn invalid_requests_fail_before_consuming_a_response() {
        let provider = MockDecisionProvider::with_error(Error::auth("bad"));
        let invalid = DecisionRequest::new("m", "x");
        assert!(matches!(
            provider.decide(&invalid).await,
            Err(Error::InvalidRequest { .. })
        ));
        assert!(provider.requests().is_empty());
        assert!(matches!(
            provider.decide(&request()).await,
            Err(Error::Auth { .. })
        ));
    }
}
