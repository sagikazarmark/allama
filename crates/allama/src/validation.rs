//! What a dry run of a request found: [`Validation`].

use crate::{
    Capability, ChatCapability, DecisionCapability, EmbeddingCapability, ExtractionCapability,
    Result, Subject, Warning,
};

/// What validating a request found, without sending it.
///
/// Returned by [`validate_chat_request`](crate::ChatProvider::validate_chat_request),
/// [`validate_chat_stream_request`](crate::ChatProvider::validate_chat_stream_request),
/// [`validate_embedding_request`](crate::EmbeddingProvider::validate_embedding_request)
/// and [`validate_decision_request`](crate::DecisionProvider::validate_decision_request)
/// when the provider would take the request; a request it would refuse is
/// an `Err` instead, the error the call would return.
///
/// - [`warnings`](Self::warnings) are exactly the request-side warnings the
///   call would report.
/// - [`relied_on_unknown`](Self::relied_on_unknown) lists what the dry run
///   could not know: the capabilities the request relies on that the
///   provider answered [`Unknown`](crate::CapabilitySupport::Unknown). The
///   provider decides those when the request is sent.
/// - [`coverage`](Self::coverage) says how far the dry run went.
/// - [`fallback`](Self::fallback) holds, for a pair of providers that fails
///   over, the other side's own dry run.
///
/// Validation never checks credentials or that the model exists: those need
/// a real call.
///
/// ```
/// # #[cfg(feature = "mock")] {
/// use allama::mock::MockProvider;
/// use allama::{ChatProvider, ChatRequest, ValidationCoverage};
///
/// let provider = MockProvider::empty();
/// let validation = provider.validate_chat_request(&ChatRequest::new("m").user("Hi"))?;
/// assert!(validation.warnings().is_empty());
/// // The mock checks capability answers only; first-party adapters encode.
/// assert_eq!(validation.coverage(), ValidationCoverage::Capabilities);
/// # }
/// # Ok::<(), allama::Error>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
#[non_exhaustive]
pub struct Validation<C: Capability> {
    warnings: Vec<Warning>,
    relied_on_unknown: Vec<(C, Subject)>,
    coverage: ValidationCoverage,
    fallback: Option<Box<FallbackValidation<C>>>,
}

/// What validating a chat request found.
pub type ChatValidation = Validation<ChatCapability>;

/// What validating an embedding request found.
pub type EmbeddingValidation = Validation<EmbeddingCapability>;

/// What validating a decision request found.
pub type DecisionValidation = Validation<DecisionCapability>;

/// What validating an extraction request found.
pub type ExtractionValidation = Validation<ExtractionCapability>;

/// How far a [`Validation`] went.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ValidationCoverage {
    /// The request was checked against the provider's capability answers
    /// ([`chat::preflight`](crate::adapter::chat::preflight)) only. A request that
    /// passes may still be refused while it is encoded, and the call may
    /// report warnings the validation did not.
    ///
    /// What the default `validate_*` methods report.
    Capabilities,
    /// Everything the call does before sending ran: capability checks,
    /// option resolution, encoding, and building the transport request. The
    /// validation gives the answer the call would give.
    ///
    /// What first-party adapters report.
    Encoding,
}

/// The other side's dry run in a pair of providers that fails over (see
/// [`Validation::fallback`]).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FallbackValidation<C: Capability> {
    /// The provider that would serve after a failover.
    pub provider: String,
    /// Its dry run: an `Err` when it would refuse the request.
    pub result: Result<Validation<C>>,
}

impl<C: Capability> Validation<C> {
    /// A validation with no warnings and nothing relied on, of `coverage`.
    pub fn new(coverage: ValidationCoverage) -> Self {
        Self {
            warnings: Vec::new(),
            relied_on_unknown: Vec::new(),
            coverage,
            fallback: None,
        }
    }

    /// The warnings the call would report about the request, in order.
    #[must_use]
    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }

    /// The capabilities the request relies on that the provider answered
    /// [`Unknown`](crate::CapabilitySupport::Unknown), each with the first
    /// subject that relies on it, in the order they were checked.
    ///
    /// Unknown never blocks a request: the provider decides when the request
    /// is sent. A request that must not depend on a guess fails on a
    /// non-empty list, or settles the answers with a capability resolver.
    #[must_use]
    pub fn relied_on_unknown(&self) -> &[(C, Subject)] {
        &self.relied_on_unknown
    }

    /// How far the dry run went.
    #[must_use]
    pub fn coverage(&self) -> ValidationCoverage {
        self.coverage
    }

    /// For a pair of providers that fails over, the dry run of the side that
    /// did not answer: whether it would take the request too, and what it
    /// would report.
    ///
    /// It is kept apart from [`warnings`](Self::warnings), which stay what
    /// the call would report.
    #[must_use]
    pub fn fallback(&self) -> Option<&FallbackValidation<C>> {
        self.fallback.as_deref()
    }

    /// Take the warnings.
    #[must_use]
    pub fn into_warnings(self) -> Vec<Warning> {
        self.warnings
    }

    /// Add a warning the call would report, after the others.
    pub fn with_warning(mut self, warning: Warning) -> Self {
        self.warnings.push(warning);
        self
    }

    /// Record that the request relies on `capability`, answered `Unknown`,
    /// first through `subject`. A capability already listed is left as it
    /// is.
    pub fn with_unknown(mut self, capability: C, subject: Subject) -> Self {
        if !self.relied_on_unknown.iter().any(|(c, _)| *c == capability) {
            self.relied_on_unknown.push((capability, subject));
        }
        self
    }

    /// Remove `capability` from [`relied_on_unknown`](Self::relied_on_unknown),
    /// for a wrapper that settled its answer.
    pub fn without_unknown(mut self, capability: C) -> Self {
        self.relied_on_unknown.retain(|(c, _)| *c != capability);
        self
    }

    /// Set the other side's dry run (see [`fallback`](Self::fallback)).
    pub fn with_fallback(mut self, fallback: FallbackValidation<C>) -> Self {
        self.fallback = Some(Box::new(fallback));
        self
    }

    /// Set the coverage.
    pub fn with_coverage(mut self, coverage: ValidationCoverage) -> Self {
        self.coverage = coverage;
        self
    }

    /// This dry run, followed by `inner`, the dry run of the provider a
    /// layer forwards to: `inner`'s warnings follow these, its `Unknown`
    /// capabilities follow the ones not listed yet, and its coverage and
    /// fallback replace these.
    pub(crate) fn followed_by(mut self, inner: Self) -> Self {
        self.warnings.extend(inner.warnings);
        for (capability, subject) in inner.relied_on_unknown {
            self = self.with_unknown(capability, subject);
        }
        self.coverage = inner.coverage;
        self.fallback = inner.fallback;
        self
    }

    /// The same validation with `f` applied to each warning.
    pub(crate) fn map_warnings(mut self, f: impl Fn(Warning) -> Warning) -> Self {
        self.warnings = self.warnings.into_iter().map(f).collect();
        self
    }

    /// Built by a `Prepared` from what its preflight and encoder recorded.
    pub(crate) fn from_parts(
        warnings: Vec<Warning>,
        relied_on_unknown: Vec<(C, Subject)>,
        coverage: ValidationCoverage,
    ) -> Self {
        Self {
            warnings,
            relied_on_unknown,
            coverage,
            fallback: None,
        }
    }
}

impl<C: Capability> FallbackValidation<C> {
    /// The dry run of `provider`.
    #[must_use]
    pub fn new(provider: impl Into<String>, result: Result<Validation<C>>) -> Self {
        Self {
            provider: provider.into(),
            result,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RequestField, WarningKind};

    #[test]
    fn builders_compose() {
        let tools = Subject::Field {
            field: RequestField::Tools,
        };
        let validation = ChatValidation::new(ValidationCoverage::Encoding)
            .with_warning(Warning::unsupported(RequestField::Seed, "no seed"))
            .with_unknown(ChatCapability::ToolCalls, tools.clone())
            .with_unknown(ChatCapability::ToolCalls, Subject::Response)
            .with_unknown(ChatCapability::StructuredOutput, Subject::Response)
            .without_unknown(ChatCapability::StructuredOutput)
            .with_fallback(FallbackValidation::new(
                "backup",
                Ok(ChatValidation::new(ValidationCoverage::Capabilities)),
            ));
        assert_eq!(validation.coverage(), ValidationCoverage::Encoding);
        assert_eq!(validation.warnings()[0].kind, WarningKind::Unsupported);
        assert_eq!(
            validation.relied_on_unknown(),
            [(ChatCapability::ToolCalls, tools)]
        );
        let fallback = validation.fallback().unwrap();
        assert_eq!(fallback.provider, "backup");
        assert_eq!(
            fallback.result.as_ref().unwrap().coverage(),
            ValidationCoverage::Capabilities
        );
    }
}
