use crate::prepared::PendingWarnings;
use crate::{
    DecisionCapability, DecisionProvider, DecisionRequest, DecisionResponse, DecisionValidation,
    Degradations, Result, Subject, UnsupportedPolicy, Validation, ValidationCoverage,
};

/// A decision request checked against a provider's limits and capability
/// answers, on its way to being encoded and sent: the decision counterpart
/// of [`chat::Prepared`](crate::adapter::chat::Prepared), with the same rules.
///
/// [`prepare`] builds it. An adapter encodes [`request`](Self::request),
/// reporting through [`degradations_mut`](Self::degradations_mut), calls
/// [`mark_sent`](Self::mark_sent) immediately before sending, and ends with
/// [`finish`](Self::finish). A dry run ([`prepare_dry_run`]) ends with
/// [`into_validation`](Self::into_validation) and never emits.
///
/// Every field of a decision request is semantic, so there is no hint to
/// degrade and no unsupported-feature policy: what an adapter cannot send,
/// it refuses.
#[must_use = "a prepared request reports its warnings through `finish` or `into_validation`"]
#[derive(Debug)]
pub struct Prepared<'a> {
    request: &'a DecisionRequest,
    pending: PendingWarnings,
}

/// Check `request` against `provider`'s limits for its model
/// ([`DecisionRequest::validate`]) and each question kind against
/// [`decision_capability`](DecisionProvider::decision_capability), and its
/// images, if any, against
/// [`ImageInput`](DecisionCapability::ImageInput), for a call that will
/// send it (see [`Prepared`]). A capability answered `Unknown` is allowed
/// through and recorded.
///
/// # Errors
///
/// [`Error::InvalidRequest`](crate::Error::InvalidRequest) for structural problems or exceeded limits,
/// and [`Error::Unsupported`](crate::Error::Unsupported) for a question kind
/// or images the model explicitly does not support.
pub fn prepare<'a, P: DecisionProvider + ?Sized>(
    provider: &P,
    request: &'a DecisionRequest,
) -> Result<Prepared<'a>> {
    Prepared::new(provider, request, false)
}

/// Check `request` as [`prepare`] does, for a dry run that never emits
/// warning events.
///
/// # Errors
///
/// As [`prepare`].
pub fn prepare_dry_run<'a, P: DecisionProvider + ?Sized>(
    provider: &P,
    request: &'a DecisionRequest,
) -> Result<Prepared<'a>> {
    Prepared::new(provider, request, true)
}

impl<'a> Prepared<'a> {
    fn new<P: DecisionProvider + ?Sized>(
        provider: &P,
        request: &'a DecisionRequest,
        dry_run: bool,
    ) -> Result<Self> {
        request.validate(&provider.decision_limits(&request.model))?;
        let mut degradations =
            Degradations::new(provider.provider_name().to_owned(), UnsupportedPolicy::Warn);
        for (name, question) in &request.questions {
            let capability = DecisionCapability::from(question.kind());
            let support = provider.decision_capability(&request.model, capability);
            let subject = || Subject::Other {
                name: format!("question {name:?}"),
            };
            if degradations.refuses(capability, support, subject) {
                return Err(degradations.reject(
                    subject(),
                    format!(
                        "`{}` does not support {} questions",
                        provider.provider_name(),
                        question.kind()
                    ),
                ));
            }
        }
        if !request.images.is_empty() {
            let capability = DecisionCapability::ImageInput;
            let support = provider.decision_capability(&request.model, capability);
            let subject = || Subject::Other {
                name: "images".into(),
            };
            if degradations.refuses(capability, support, subject) {
                return Err(degradations.reject(
                    subject(),
                    format!(
                        "`{}` does not take images in decision requests",
                        provider.provider_name()
                    ),
                ));
            }
        }
        Ok(Self {
            request,
            pending: PendingWarnings::new(request.model.clone(), degradations, dry_run),
        })
    }

    /// The request to encode.
    #[must_use]
    pub fn request(&self) -> &DecisionRequest {
        self.request
    }

    /// The helper to report through while encoding.
    pub fn degradations_mut(&mut self) -> &mut Degradations {
        self.pending.degradations_mut()
    }

    /// The request and the helper together.
    pub fn parts_mut(&mut self) -> (&DecisionRequest, &mut Degradations) {
        (self.request, self.pending.degradations_mut())
    }

    /// Whether this is a dry run ([`prepare_dry_run`]).
    #[must_use]
    pub fn is_dry_run(&self) -> bool {
        self.pending.is_dry_run()
    }

    /// Record that the request is about to reach the transport.
    pub fn mark_sent(&mut self) {
        self.pending.mark_sent();
    }

    /// Join the request-side warnings to `response`, ahead of its own, and
    /// emit them as warning events (not for a dry run).
    pub fn finish(self, mut response: DecisionResponse) -> DecisionResponse {
        let inner = std::mem::take(&mut response.warnings);
        response.warnings = self.pending.take().merge(inner);
        response
    }

    /// End a dry run of a layer that forwards the request to another
    /// provider: this layer's warnings, then `inner`'s (the forwarded
    /// request's own dry run, from its `validate_decision_request`), translated to the
    /// caller's indices, as [`finish`](Self::finish) joins them to a
    /// response. The capabilities either answered `Unknown` are listed, this
    /// layer's first; the coverage and fallback are `inner`'s. Nothing is
    /// emitted.
    pub fn finish_validation(self, inner: DecisionValidation) -> DecisionValidation {
        self.pending.take().merge_validation(inner)
    }

    /// End a dry run, emitting nothing.
    pub fn into_validation(self, coverage: ValidationCoverage) -> DecisionValidation {
        let (warnings, unknown) = self.pending.take().into_validation_parts();
        Validation::from_parts(warnings, unknown, coverage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::Question;
    use crate::decision::mock::MockDecisionProvider;
    use crate::{CapabilitySupport, Error, Image};

    fn request() -> DecisionRequest {
        DecisionRequest::new("m", "x")
            .image(Image::base64("image/png", "aGk="))
            .question("q", Question::yes_no("?"))
    }

    #[test]
    fn images_are_refused_where_unsupported() {
        let provider = MockDecisionProvider::empty().with_decision_capability(
            DecisionCapability::ImageInput,
            CapabilitySupport::Unsupported,
        );
        let err = prepare(&provider, &request()).unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
        assert!(err.message().contains("images"), "{err}");
        // A request without images does not rely on the capability.
        let text_only = DecisionRequest::new("m", "x").question("q", Question::yes_no("?"));
        let validation = prepare_dry_run(&provider, &text_only)
            .unwrap()
            .into_validation(ValidationCoverage::Capabilities);
        assert!(
            validation
                .relied_on_unknown()
                .iter()
                .all(|(capability, _)| *capability != DecisionCapability::ImageInput),
            "{validation:?}"
        );
    }

    #[test]
    fn images_rely_on_an_unknown_answer() {
        let provider = MockDecisionProvider::empty();
        let validation = prepare_dry_run(&provider, &request())
            .unwrap()
            .into_validation(ValidationCoverage::Capabilities);
        assert!(
            validation
                .relied_on_unknown()
                .iter()
                .any(|(capability, _)| *capability == DecisionCapability::ImageInput),
            "{validation:?}"
        );
    }
}
