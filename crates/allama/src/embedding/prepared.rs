use std::borrow::Cow;

use crate::prepared::PendingWarnings;
use crate::{
    Degradations, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, EmbeddingValidation,
    Result, Validation, ValidationCoverage,
};

/// An embedding request checked against a provider's capability answers,
/// on its way to being encoded and sent: the embedding counterpart of
/// [`chat::Prepared`](crate::adapter::chat::Prepared), with the same rules.
///
/// [`prepare`] builds it and runs [`preflight`](crate::adapter::embedding::preflight).
/// An adapter encodes [`request`](Self::request), reporting through
/// [`degradations_mut`](Self::degradations_mut), calls
/// [`mark_sent`](Self::mark_sent) immediately before sending, and ends with
/// [`finish`](Self::finish), which joins the request-side warnings to the
/// response and emits them as warning events. A dry run
/// ([`prepare_dry_run`]) ends with [`into_validation`](Self::into_validation)
/// and never emits. Dropped after `mark_sent`, it emits its warnings; dropped
/// before, it emits nothing.
#[must_use = "a prepared request reports its warnings through `finish` or `into_validation`"]
#[derive(Debug)]
pub struct Prepared<'a> {
    request: Cow<'a, EmbeddingRequest>,
    pending: PendingWarnings,
}

/// Check `request` against what `provider` reports for its model, for a call
/// that will send it (see [`Prepared`]). An unset
/// [`unsupported_policy`](EmbeddingRequest::unsupported_policy) is
/// [`UnsupportedPolicy::Warn`](crate::UnsupportedPolicy::Warn).
///
/// # Errors
///
/// What [`preflight`](crate::adapter::embedding::preflight) returns.
pub fn prepare<'a, P: EmbeddingProvider + ?Sized>(
    provider: &P,
    request: &'a EmbeddingRequest,
) -> Result<Prepared<'a>> {
    Prepared::new(provider, request, false)
}

/// Check `request` as [`prepare`] does, for a dry run that never emits
/// warning events.
///
/// # Errors
///
/// What [`preflight`](crate::adapter::embedding::preflight) returns.
pub fn prepare_dry_run<'a, P: EmbeddingProvider + ?Sized>(
    provider: &P,
    request: &'a EmbeddingRequest,
) -> Result<Prepared<'a>> {
    Prepared::new(provider, request, true)
}

impl<'a> Prepared<'a> {
    fn new<P: EmbeddingProvider + ?Sized>(
        provider: &P,
        request: &'a EmbeddingRequest,
        dry_run: bool,
    ) -> Result<Self> {
        let degradations = Degradations::new(
            provider.provider_name().to_owned(),
            request.unsupported_policy.unwrap_or_default(),
        );
        let mut prepared = Self {
            request: Cow::Borrowed(request),
            pending: PendingWarnings::new(request.model.clone(), degradations, dry_run),
        };
        crate::adapter::embedding::preflight(
            provider,
            &mut prepared.request,
            prepared.pending.degradations_mut(),
        )?;
        Ok(prepared)
    }

    /// The request to encode: the caller's, less what preflight removed.
    #[must_use]
    pub fn request(&self) -> &EmbeddingRequest {
        &self.request
    }

    /// The helper to report through while encoding.
    pub fn degradations_mut(&mut self) -> &mut Degradations {
        self.pending.degradations_mut()
    }

    /// The request and the helper together.
    pub fn parts_mut(&mut self) -> (&EmbeddingRequest, &mut Degradations) {
        (&self.request, self.pending.degradations_mut())
    }

    /// Whether this is a dry run ([`prepare_dry_run`]).
    #[must_use]
    pub fn is_dry_run(&self) -> bool {
        self.pending.is_dry_run()
    }

    /// Run a second preflight against `other`'s capability answers, for an
    /// adapter layered on another provider. `other`'s `Unknown` answers are
    /// not recorded.
    ///
    /// # Errors
    ///
    /// What [`preflight`](crate::adapter::embedding::preflight) returns for `other`.
    pub fn preflight_with<P: EmbeddingProvider + ?Sized>(&mut self, other: &P) -> Result<()> {
        let degradations = self.pending.degradations_mut();
        let recorded = degradations.unknown_len();
        let result = crate::adapter::embedding::preflight(other, &mut self.request, degradations);
        degradations.truncate_unknown(recorded);
        result
    }

    /// Record that the request is about to reach the transport.
    pub fn mark_sent(&mut self) {
        self.pending.mark_sent();
    }

    /// Join the request-side warnings to `response`, ahead of its own, and
    /// emit them as warning events (not for a dry run).
    ///
    /// A response without a [`provider`](EmbeddingResponse::provider) gets
    /// the name of the provider this request was prepared for. One that
    /// has a provider keeps it: a layer that forwards the request, such as a
    /// wrapper that checks requests with
    /// [`prepare`](crate::adapter::embedding::prepare), finishes the
    /// response of the provider that served it.
    pub fn finish(self, mut response: EmbeddingResponse) -> EmbeddingResponse {
        let inner = std::mem::take(&mut response.warnings);
        let taken = self.pending.take();
        if response.provider.is_none() {
            response.provider = Some(Cow::Owned(taken.degradations.provider().to_owned()));
        }
        response.warnings = taken.merge(inner);
        response
    }

    /// End a dry run of a layer that forwards the request to another
    /// provider: this layer's warnings, then `inner`'s (the forwarded
    /// request's own dry run, from its `validate_embedding_request`), translated to the
    /// caller's indices, as [`finish`](Self::finish) joins them to a
    /// response. The capabilities either answered `Unknown` are listed, this
    /// layer's first; the coverage and fallback are `inner`'s. Nothing is
    /// emitted.
    pub fn finish_validation(self, inner: EmbeddingValidation) -> EmbeddingValidation {
        self.pending.take().merge_validation(inner)
    }

    /// End a dry run, emitting nothing.
    pub fn into_validation(self, coverage: ValidationCoverage) -> EmbeddingValidation {
        let (warnings, unknown) = self.pending.take().into_validation_parts();
        Validation::from_parts(warnings, unknown, coverage)
    }
}
