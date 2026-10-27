use crate::prepared::PendingWarnings;
use crate::{
    Degradations, ExtractionCapability, ExtractionProvider, ExtractionRequest, ExtractionResponse,
    ExtractionValidation, Result, Subject, UnsupportedPolicy, Validation, ValidationCoverage,
};

/// An extraction request checked against a provider's limits and capability
/// answers, on its way to being encoded and sent: the extraction counterpart
/// of [`chat::Prepared`](crate::adapter::chat::Prepared), with the same rules.
///
/// [`prepare`] builds it. An adapter encodes [`request`](Self::request),
/// reporting through [`degradations_mut`](Self::degradations_mut), calls
/// [`mark_sent`](Self::mark_sent) immediately before sending, and ends with
/// [`finish`](Self::finish). A dry run ([`prepare_dry_run`]) ends with
/// [`into_validation`](Self::into_validation) and never emits.
///
/// Every parameter of an extraction field is semantic, so there is no hint
/// to degrade and no unsupported-feature policy: what an adapter cannot
/// send, it refuses.
#[must_use = "a prepared request reports its warnings through `finish` or `into_validation`"]
#[derive(Debug)]
pub struct Prepared<'a> {
    request: &'a ExtractionRequest,
    pending: PendingWarnings,
}

/// Check `request` against `provider`'s limits for its model
/// ([`ExtractionRequest::validate`]) and each field kind against
/// [`extraction_capability`](ExtractionProvider::extraction_capability), and
/// its images, if any, against
/// [`ImageInput`](ExtractionCapability::ImageInput), for a call that will
/// send it (see [`Prepared`]). A capability answered `Unknown` is allowed
/// through and recorded.
///
/// # Errors
///
/// [`Error::InvalidRequest`](crate::Error::InvalidRequest) for structural
/// problems or exceeded limits, and
/// [`Error::Unsupported`](crate::Error::Unsupported) for a field kind or
/// images the model explicitly does not support.
pub fn prepare<'a, P: ExtractionProvider + ?Sized>(
    provider: &P,
    request: &'a ExtractionRequest,
) -> Result<Prepared<'a>> {
    Prepared::new(provider, request, false)
}

/// Check `request` as [`prepare`] does, for a dry run that never emits
/// warning events.
///
/// # Errors
///
/// As [`prepare`].
pub fn prepare_dry_run<'a, P: ExtractionProvider + ?Sized>(
    provider: &P,
    request: &'a ExtractionRequest,
) -> Result<Prepared<'a>> {
    Prepared::new(provider, request, true)
}

impl<'a> Prepared<'a> {
    fn new<P: ExtractionProvider + ?Sized>(
        provider: &P,
        request: &'a ExtractionRequest,
        dry_run: bool,
    ) -> Result<Self> {
        request.validate(&provider.extraction_limits(&request.model))?;
        let mut degradations =
            Degradations::new(provider.provider_name().to_owned(), UnsupportedPolicy::Warn);
        for (name, field) in &request.fields {
            let capability = ExtractionCapability::from(field.kind());
            let support = provider.extraction_capability(&request.model, capability);
            let subject = || Subject::Other {
                name: format!("field {name:?}"),
            };
            if degradations.refuses(capability, support, subject) {
                return Err(degradations.reject(
                    subject(),
                    format!(
                        "`{}` does not support {} fields",
                        provider.provider_name(),
                        field.kind()
                    ),
                ));
            }
        }
        if !request.images.is_empty() {
            let capability = ExtractionCapability::ImageInput;
            let support = provider.extraction_capability(&request.model, capability);
            let subject = || Subject::Other {
                name: "images".into(),
            };
            if degradations.refuses(capability, support, subject) {
                return Err(degradations.reject(
                    subject(),
                    format!(
                        "`{}` does not take images in extraction requests",
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
    pub fn request(&self) -> &ExtractionRequest {
        self.request
    }

    /// The helper to report through while encoding.
    pub fn degradations_mut(&mut self) -> &mut Degradations {
        self.pending.degradations_mut()
    }

    /// The request and the helper together.
    pub fn parts_mut(&mut self) -> (&ExtractionRequest, &mut Degradations) {
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
    pub fn finish(self, mut response: ExtractionResponse) -> ExtractionResponse {
        let inner = std::mem::take(&mut response.warnings);
        response.warnings = self.pending.take().merge(inner);
        response
    }

    /// End a dry run of a layer that forwards the request to another
    /// provider: this layer's warnings, then `inner`'s, as
    /// [`finish`](Self::finish) joins them to a response. Nothing is
    /// emitted.
    pub fn finish_validation(self, inner: ExtractionValidation) -> ExtractionValidation {
        self.pending.take().merge_validation(inner)
    }

    /// End a dry run, emitting nothing.
    pub fn into_validation(self, coverage: ValidationCoverage) -> ExtractionValidation {
        let (warnings, unknown) = self.pending.take().into_validation_parts();
        Validation::from_parts(warnings, unknown, coverage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extraction::Field;
    use crate::extraction::mock::MockExtractionProvider;
    use crate::{CapabilitySupport, Error, Image};

    fn request() -> ExtractionRequest {
        ExtractionRequest::new("m", "x")
            .image(Image::base64("image/png", "aGk="))
            .field("f", Field::text())
    }

    #[test]
    fn images_are_refused_where_unsupported() {
        let provider = MockExtractionProvider::empty().with_extraction_capability(
            ExtractionCapability::ImageInput,
            CapabilitySupport::Unsupported,
        );
        let err = prepare(&provider, &request()).unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
        assert!(err.message().contains("images"), "{err}");
        let text_only = ExtractionRequest::new("m", "x").field("f", Field::text());
        let validation = prepare_dry_run(&provider, &text_only)
            .unwrap()
            .into_validation(ValidationCoverage::Capabilities);
        assert!(
            validation
                .relied_on_unknown()
                .iter()
                .all(|(capability, _)| *capability != ExtractionCapability::ImageInput),
            "{validation:?}"
        );
    }

    #[test]
    fn kinds_and_images_rely_on_unknown_answers() {
        let provider = MockExtractionProvider::empty();
        let validation = prepare_dry_run(&provider, &request())
            .unwrap()
            .into_validation(ValidationCoverage::Capabilities);
        let relied: Vec<_> = validation
            .relied_on_unknown()
            .iter()
            .map(|(capability, _)| *capability)
            .collect();
        assert_eq!(
            relied,
            [ExtractionCapability::Text, ExtractionCapability::ImageInput]
        );
    }
}
