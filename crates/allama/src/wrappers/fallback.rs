use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::Classifier;
use super::call_scope::{self, CallFact};
use crate::chat::stream::{MetadataBuilder, Prefixed, until_committed};
use crate::{
    Capability, CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse,
    ChatStream, ChatValidation, DecisionCapability, DecisionLimits, DecisionProvider,
    DecisionRequest, DecisionResponse, DecisionValidation, EmbeddingCapability, EmbeddingLimits,
    EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, EmbeddingValidation, Error,
    ExtractionCapability, ExtractionLimits, ExtractionProvider, ExtractionRequest,
    ExtractionResponse, ExtractionValidation, FallbackValidation, ProviderData, ProviderDataEntry,
    ProviderIdentity, Result, StreamEvent, Subject, Validation, Warning, WarningCause, WarningKind,
};

/// Fails over to a second provider when the primary fails before producing
/// output.
///
/// - Unary: a fallback-eligible error from the primary sends the same request
///   to the fallback.
/// - Streaming: an error before the primary's first model-output event (see
///   [`StreamEvent::is_output`]) fails over; errors after output has reached
///   the caller are passed through.
///
/// The request is forwarded unchanged. Opaque replay state in the history is
/// namespaced by origin, so the fallback adapter ignores state minted by the
/// primary (see [`crate::provider_data`]). Model ids are not translated: when
/// the providers do not share model ids, give a side its own with
/// [`MapModelProvider`](super::MapModelProvider), which also asks that side's
/// capability and validation questions about the model it would receive:
///
/// ```
/// # #[cfg(feature = "mock")] {
/// use allama::wrappers::{FallbackProvider, MapModelProvider};
/// # use allama::mock::MockProvider;
/// # let (anthropic, openai) = (MockProvider::with_text("a"), MockProvider::with_text("b"));
///
/// let provider = FallbackProvider::new(
///     anthropic,
///     MapModelProvider::new(openai, |_: &str| "gpt-5-mini".into()),
/// );
/// # let _ = provider; }
/// ```
///
/// Which errors fail over is decided by a classifier. The default,
/// [`should_fallback`], fails over on outages, overload, rate limits and
/// exhausted quotas, timeouts, requests the primary cannot serve and
/// prompts over its context window, and not on caller mistakes, client
/// misconfiguration, rejected credentials, a content filter, or a model the
/// primary does not know ([`Error::ModelNotFound`]): failing over on a
/// mistyped model id would hide the mistake behind another provider's
/// answer. [`should_fallback`] has recipes that opt in to content filters
/// and rejected credentials. Replace it with
/// [`fallback_if`](Self::fallback_if), for example to fail over when the
/// primary's model was retired or is not deployed in its region:
///
/// ```
/// # #[cfg(feature = "mock")] {
/// use allama::Error;
/// use allama::wrappers::{FallbackProvider, should_fallback};
/// # use allama::mock::MockProvider;
/// # let (primary, backup) = (MockProvider::with_text("a"), MockProvider::with_text("b"));
///
/// let provider = FallbackProvider::new(primary, backup).fallback_if(|error: &Error| {
///     matches!(error, Error::ModelNotFound { .. }) || should_fallback(error)
/// });
/// # let _ = provider; }
/// ```
///
/// **Retries on each side.** A side that retries (a
/// [`RetryingProvider`](super::RetryingProvider), or anything built with
/// `allama_providers::chat::from_model`, which retries by default) fails
/// over only once its retries give up. With the default
/// [`RetryPolicy`](super::RetryPolicy), a side waits out a `Retry-After` of
/// up to 120 s before failing over. Each side's worst case runs to
/// minutes, and the pair's is the sum of both; a
/// [`TimeoutProvider`](super::TimeoutProvider) deadline around the pair
/// bounds it (see
/// [worst-case latency][worst-case latency]
/// in `allama_providers::ProviderSettings`). An
/// interactive pair sets a hint ceiling of a few seconds on each side, so a
/// long rate limit fails over at once: `RetryPolicy::new().max_retry_after(Duration::from_secs(5))`,
/// through `allama_providers::ProviderSettings::retry` for sides built
/// there. Retry inside each side, not around the pair: a retry wrapper
/// outside would repeat the whole failover.
///
/// When failover happens, the served response carries a
/// [`WarningKind::Fallback`] warning attributed to the primary, which
/// displays as "\[anthropic\] failed with `auth`; served by `ollama`" (behind
/// a nested fallback, the serving provider is the innermost one). It is
/// emitted as a warning event whose `allama.warning.cause_kind` field is
/// the error kind, kept on the warning as its [`cause`](Warning::cause).
/// The response also records the
/// attempts in its provider data as [`FallbackAttempts`] and reports the
/// serving provider in [`ChatResponse::provider`] (or
/// [`DecisionResponse::provider`]). A chain
/// built by nesting on the fallback side
/// (`FallbackProvider::new(a, FallbackProvider::new(b, c))`) records every
/// failed attempt, oldest first. A stream reports them in a
/// `ResponseMetadata` event right after the fallback's `ResponseStart`, so
/// the stream keeps the usual event order.
///
/// Each side runs in a side of the call scope (see
/// [`call_scope`](super::call_scope)), and the wrapper reports each
/// failover and the serving provider to it. Inside a
/// `TracingProvider` (feature `tracing`), the span summarizes them once
/// (`allama.attempts`, `allama.fallback.*`, `allama.response.provider`) and
/// each failover is an `allama.fallback` span event.
///
/// For embeddings, both sides must serve the same model with equivalent
/// provider settings: vectors from different models live in different
/// spaces and cannot be compared, so failing over to another model (for
/// example through [`MapModelProvider`](super::MapModelProvider) on one
/// side) silently corrupts an index. Fall back between hosts of the same
/// model.
///
/// Either side may receive a request, so questions about the provider
/// combine both sides:
///
/// - [`provider_name`](ProviderIdentity::provider_name) comes from the
///   primary;
/// - request validation ([`validate_chat_request`](ChatProvider::validate_chat_request),
///   [`validate_chat_stream_request`](ChatProvider::validate_chat_stream_request),
///   [`validate_embedding_request`](EmbeddingProvider::validate_embedding_request),
///   [`validate_decision_request`](DecisionProvider::validate_decision_request))
///   follows the call through the classifier: when the primary accepts the
///   request, the pair gives the primary's validation, with the fallback's
///   own dry run in [`Validation::fallback`] (whether the backup could
///   serve, and what it would report); when the primary refuses with an
///   error the classifier fails over on, the pair gives the fallback's
///   validation plus the failover warning, or the fallback's error, with
///   the primary's [behind it](Error::fallback_from), when the fallback
///   refuses too; any other refusal is the pair's;
/// - capability answers agree or are unknown: the pair gives the answer
///   both sides give, and `Unknown` when they differ, since either side may
///   serve;
/// - [`embedding_limits`](EmbeddingProvider::embedding_limits) and each
///   [`decision_limits`](DecisionProvider::decision_limits) field are the
///   smaller known limit, so a request fits whichever side receives it;
/// - [`downcast_ref`](crate::ProviderIdentityExt::downcast_ref) looks in the
///   primary first, then the fallback. When both have the same type, it
///   always finds the primary; reach each one with [`primary`](Self::primary)
///   and [`fallback`](Self::fallback).
///
/// When both sides fail, the pair returns the fallback's error as itself
/// (same variant, status and kind), with the primary's error kept on it
/// ([`Error::fallback_from`]), so code that matches on errors handles a
/// failed failover like any other failure. A chain built by nesting returns
/// the last side's error, with every earlier one behind it, newest first.
/// When the last side fails with an error that has no context
/// ([`Error::Serialization`], or an extraction failure), the earlier errors
/// are dropped, and the `allama::fallback` event that each failover emits
/// is their only record.
///
/// Over two copies of the same provider it is a transparent wrapper (see
/// `allama_conformance::assert_wrapper`): when both sides fail, the pair
/// returns the provider's own error.
///
#[doc = concat!(
    "[worst-case latency]: https://docs.rs/allama-providers/",
    env!("CARGO_PKG_VERSION"),
    "/allama_providers/struct.ProviderSettings.html#worst-case-latency",
)]
#[derive(Clone)]
pub struct FallbackProvider<P, F> {
    primary: P,
    fallback: F,
    /// `None` is the default classifier, [`should_fallback`].
    fallback_if: Option<Classifier>,
}

/// One failed attempt recorded by [`FallbackProvider`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FallbackAttempt {
    /// Provider that was tried.
    pub provider: String,
    /// Error category (see [`Error::kind_name`]).
    pub error_kind: String,
    /// The error's redacted [`as_log`](Error::as_log) line, safe to persist:
    /// it carries no response body or upstream message, and the constructor
    /// message only when it is authored.
    pub error_message: String,
}

/// Attempts that failed before the serving provider succeeded, oldest first.
///
/// Read it from a response's provider data:
///
/// ```
/// # use allama::ChatResponse;
/// use allama::wrappers::FallbackAttempts;
///
/// # let response = ChatResponse::default();
/// if let Some(attempts) = response.provider_data.get::<FallbackAttempts>()? {
///     for attempt in &attempts {
///         eprintln!("{} failed: {}", attempt.provider, attempt.error_kind);
///     }
/// }
/// # Ok::<(), allama::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FallbackAttempts(Vec<FallbackAttempt>);

impl FallbackAttempts {
    /// Iterate over the failed attempts, oldest first.
    pub fn iter(&self) -> std::slice::Iter<'_, FallbackAttempt> {
        self.0.iter()
    }

    /// Number of failed attempts.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` when no attempt failed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Borrow the failed attempts as a slice, oldest first.
    #[must_use]
    pub fn as_slice(&self) -> &[FallbackAttempt] {
        &self.0
    }

    /// Unwrap into the failed attempts, oldest first.
    #[must_use]
    pub fn into_vec(self) -> Vec<FallbackAttempt> {
        self.0
    }
}

impl ProviderDataEntry for FallbackAttempts {
    const NAMESPACE: &'static str = crate::provider_data::ALLAMA_NAMESPACE;
    const KEY: &'static str = "fallback_attempts";
}

impl<'a> IntoIterator for &'a FallbackAttempts {
    type Item = &'a FallbackAttempt;
    type IntoIter = std::slice::Iter<'a, FallbackAttempt>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl IntoIterator for FallbackAttempts {
    type Item = FallbackAttempt;
    type IntoIter = std::vec::IntoIter<FallbackAttempt>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<P, F> FallbackProvider<P, F> {
    /// Wrap a primary provider with a fallback.
    #[must_use]
    pub fn new(primary: P, fallback: F) -> Self {
        Self {
            primary,
            fallback,
            fallback_if: None,
        }
    }

    /// Fail over only on the errors `classify` accepts, instead of the
    /// default [`should_fallback`]. Compose with it to widen or narrow the
    /// default rather than restating it.
    ///
    /// When the primary is itself a `FallbackProvider` and both of its sides
    /// failed, `classify` sees the error that ended that chain, which is
    /// the last side's error as itself (see [`Error::fallback_from`]), so a
    /// classifier needs no special case for nesting. A side that fails over
    /// on its own (on the fallback side of this one) uses its own
    /// classifier.
    #[must_use]
    pub fn fallback_if(
        mut self,
        classify: impl Fn(&Error) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.fallback_if = Some(Arc::new(classify));
        self
    }

    /// Whether the classifier accepts the primary's `error` for failover.
    fn should_fail_over(&self, error: &Error) -> bool {
        match &self.fallback_if {
            Some(classify) => classify(error),
            None => should_fallback(error),
        }
    }

    /// Borrow the primary provider.
    #[must_use]
    pub fn primary(&self) -> &P {
        &self.primary
    }

    /// Borrow the fallback provider.
    #[must_use]
    pub fn fallback(&self) -> &F {
        &self.fallback
    }

    /// Unwrap into `(primary, fallback)`.
    #[must_use]
    pub fn into_parts(self) -> (P, F) {
        (self.primary, self.fallback)
    }
}

impl<P, F> ChatProvider for FallbackProvider<P, F>
where
    P: ChatProvider,
    F: ChatProvider,
{
    type Stream = ChatStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let primary_error = match side(&self.primary, self.primary.chat(request)).await {
            Ok(response) => return Ok(response),
            Err(error) if self.should_fail_over(&error) => error,
            Err(error) => return Err(error),
        };
        self.fail_over(&primary_error);

        match side(&self.fallback, self.fallback.chat(request)).await {
            Ok(mut response) => {
                let served = response
                    .provider
                    .get_or_insert_with(|| self.fallback.provider_name().to_owned().into())
                    .clone();
                record_attempt(
                    &mut response.provider_data,
                    self.primary.provider_name(),
                    &primary_error,
                );
                response.warnings.push(self.failover_warning(
                    &request.model,
                    &primary_error,
                    &served,
                    true,
                ));
                Ok(response)
            }
            Err(fallback_error) => Err(fallback_error.with_fallback_from(primary_error)),
        }
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        let primary_error =
            match side(&self.primary, committed(self.primary.chat_stream(request))).await {
                Ok(prefixed) => return Ok(Box::pin(prefixed)),
                Err(error) => error,
            };
        if !self.should_fail_over(&primary_error) {
            return Err(primary_error);
        }
        self.fail_over(&primary_error);

        let fallback_stream = match side(
            &self.fallback,
            committed(self.fallback.chat_stream(request)),
        )
        .await
        {
            Ok(prefixed) => prefixed,
            Err(fallback_error) => {
                return Err(fallback_error.with_fallback_from(primary_error));
            }
        };

        let (mut buffered, rest) = fallback_stream.into_parts();
        let served = record_stream_attempt(
            &mut buffered,
            self.primary.provider_name(),
            self.fallback.provider_name(),
            &primary_error,
        );
        // After the fallback's own warnings, before its first block, as in
        // the unary response.
        let before_output = buffered
            .iter()
            .position(|event| event.is_output() || matches!(event, StreamEvent::BlockStart { .. }))
            .unwrap_or(buffered.len());
        buffered.insert(
            before_output,
            StreamEvent::warning(self.failover_warning(
                &request.model,
                &primary_error,
                &served,
                true,
            )),
        );
        Ok(Box::pin(Prefixed::new(buffered, rest)))
    }

    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        merge_capability_support(
            self.primary.chat_capability(model, capability),
            self.fallback.chat_capability(model, capability),
        )
    }

    fn validate_chat_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        self.pair_validation(
            &request.model,
            self.primary.validate_chat_request(request),
            || self.fallback.validate_chat_request(request),
        )
    }

    fn validate_chat_stream_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        self.pair_validation(
            &request.model,
            self.primary.validate_chat_stream_request(request),
            || self.fallback.validate_chat_stream_request(request),
        )
    }
}

impl<P, F> EmbeddingProvider for FallbackProvider<P, F>
where
    P: EmbeddingProvider,
    F: EmbeddingProvider,
{
    /// The response names the side that served it, unless a provider
    /// inside that side already did ([`EmbeddingResponse::provider`]).
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        let primary_error = match side(&self.primary, self.primary.embed(request)).await {
            Ok(response) => return Ok(served_by(response, self.primary.provider_name())),
            Err(error) if self.should_fail_over(&error) => error,
            Err(error) => return Err(error),
        };
        self.fail_over(&primary_error);

        match side(&self.fallback, self.fallback.embed(request)).await {
            Ok(response) => {
                let mut response = served_by(response, self.fallback.provider_name());
                record_attempt(
                    &mut response.provider_data,
                    self.primary.provider_name(),
                    &primary_error,
                );
                let served = response
                    .provider
                    .clone()
                    .unwrap_or(Cow::Borrowed(self.fallback.provider_name()));
                response.warnings.push(self.failover_warning(
                    &request.model,
                    &primary_error,
                    &served,
                    true,
                ));
                Ok(response)
            }
            Err(fallback_error) => Err(fallback_error.with_fallback_from(primary_error)),
        }
    }

    fn embedding_capability(
        &self,
        model: &str,
        capability: EmbeddingCapability,
    ) -> CapabilitySupport {
        merge_capability_support(
            self.primary.embedding_capability(model, capability),
            self.fallback.embedding_capability(model, capability),
        )
    }

    /// The smaller known limit of the two sides, field by field: a batch
    /// sized for the primary must also fit the fallback that may receive
    /// it.
    fn embedding_limits(&self, model: &str) -> EmbeddingLimits {
        let primary = self.primary.embedding_limits(model);
        let fallback = self.fallback.embedding_limits(model);
        let mut limits = EmbeddingLimits::new();
        limits.max_inputs_per_call =
            smaller_limit(primary.max_inputs_per_call, fallback.max_inputs_per_call);
        limits
    }

    fn validate_embedding_request(
        &self,
        request: &EmbeddingRequest,
    ) -> Result<EmbeddingValidation> {
        self.pair_validation(
            &request.model,
            self.primary.validate_embedding_request(request),
            || self.fallback.validate_embedding_request(request),
        )
    }
}

impl<P, F> DecisionProvider for FallbackProvider<P, F>
where
    P: DecisionProvider,
    F: DecisionProvider,
{
    async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse> {
        let primary_error = match side(&self.primary, self.primary.decide(request)).await {
            Ok(response) => return Ok(response),
            Err(error) if self.should_fail_over(&error) => error,
            Err(error) => return Err(error),
        };
        self.fail_over(&primary_error);

        match side(&self.fallback, self.fallback.decide(request)).await {
            Ok(mut response) => {
                let served = response
                    .provider
                    .get_or_insert_with(|| self.fallback.provider_name().to_owned().into())
                    .clone();
                record_attempt(
                    &mut response.provider_data,
                    self.primary.provider_name(),
                    &primary_error,
                );
                response.warnings.push(self.failover_warning(
                    &request.model,
                    &primary_error,
                    &served,
                    true,
                ));
                Ok(response)
            }
            Err(fallback_error) => Err(fallback_error.with_fallback_from(primary_error)),
        }
    }

    fn decision_capability(
        &self,
        model: &str,
        capability: DecisionCapability,
    ) -> CapabilitySupport {
        merge_capability_support(
            self.primary.decision_capability(model, capability),
            self.fallback.decision_capability(model, capability),
        )
    }

    /// The smaller known limit of the two sides, field by field: a request
    /// must fit whichever side receives it.
    fn decision_limits(&self, model: &str) -> DecisionLimits {
        let primary = self.primary.decision_limits(model);
        let fallback = self.fallback.decision_limits(model);
        DecisionLimits {
            max_questions: smaller_limit(primary.max_questions, fallback.max_questions),
            max_choice_options: smaller_limit(
                primary.max_choice_options,
                fallback.max_choice_options,
            ),
            max_score_levels: smaller_limit(primary.max_score_levels, fallback.max_score_levels),
            max_images: smaller_limit(primary.max_images, fallback.max_images),
        }
    }

    fn validate_decision_request(&self, request: &DecisionRequest) -> Result<DecisionValidation> {
        self.pair_validation(
            &request.model,
            self.primary.validate_decision_request(request),
            || self.fallback.validate_decision_request(request),
        )
    }
}

impl<P, F> ExtractionProvider for FallbackProvider<P, F>
where
    P: ExtractionProvider,
    F: ExtractionProvider,
{
    async fn extract_values(&self, request: &ExtractionRequest) -> Result<ExtractionResponse> {
        let primary_error = match side(&self.primary, self.primary.extract_values(request)).await {
            Ok(response) => return Ok(response),
            Err(error) if self.should_fail_over(&error) => error,
            Err(error) => return Err(error),
        };
        self.fail_over(&primary_error);

        match side(&self.fallback, self.fallback.extract_values(request)).await {
            Ok(mut response) => {
                let served = response
                    .provider
                    .get_or_insert_with(|| self.fallback.provider_name().to_owned().into())
                    .clone();
                record_attempt(
                    &mut response.provider_data,
                    self.primary.provider_name(),
                    &primary_error,
                );
                response.warnings.push(self.failover_warning(
                    &request.model,
                    &primary_error,
                    &served,
                    true,
                ));
                Ok(response)
            }
            Err(fallback_error) => Err(fallback_error.with_fallback_from(primary_error)),
        }
    }

    fn extraction_capability(
        &self,
        model: &str,
        capability: ExtractionCapability,
    ) -> CapabilitySupport {
        merge_capability_support(
            self.primary.extraction_capability(model, capability),
            self.fallback.extraction_capability(model, capability),
        )
    }

    /// The smaller known limit of the two sides, field by field.
    fn extraction_limits(&self, model: &str) -> ExtractionLimits {
        let primary = self.primary.extraction_limits(model);
        let fallback = self.fallback.extraction_limits(model);
        let mut limits = ExtractionLimits::new();
        limits.max_fields = smaller_limit(primary.max_fields, fallback.max_fields);
        limits.max_images = smaller_limit(primary.max_images, fallback.max_images);
        limits.max_forecast_horizon =
            smaller_limit(primary.max_forecast_horizon, fallback.max_forecast_horizon);
        limits
    }

    fn validate_extraction_request(
        &self,
        request: &ExtractionRequest,
    ) -> Result<ExtractionValidation> {
        self.pair_validation(
            &request.model,
            self.primary.validate_extraction_request(request),
            || self.fallback.validate_extraction_request(request),
        )
    }
}

impl<P: ProviderIdentity, F: ProviderIdentity> ProviderIdentity for FallbackProvider<P, F> {
    /// The wrapper reports the primary identity; the provider that actually
    /// served a request is in [`ChatResponse::provider`].
    fn provider_name(&self) -> &str {
        self.primary.provider_name()
    }

    /// Reads what either side reads: either may serve the next turn, so
    /// state for either stays.
    fn reads_namespace(&self, namespace: &str) -> bool {
        self.primary.reads_namespace(namespace) || self.fallback.reads_namespace(namespace)
    }

    /// Queries the primary, then the fallback.
    fn downcast_any(&self, id: std::any::TypeId) -> Option<&(dyn std::any::Any + Send + Sync)> {
        self.primary
            .downcast_any(id)
            .or_else(|| self.fallback.downcast_any(id))
    }
}

impl<P, F> fmt::Debug for FallbackProvider<P, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FallbackProvider").finish_non_exhaustive()
    }
}

impl<P: ProviderIdentity, F: ProviderIdentity> FallbackProvider<P, F> {
    /// Report that the primary failed with `error` and the request fails
    /// over: to the call scope, and as an `allama.fallback` event (target
    /// `allama::fallback`, `INFO`) in the current span.
    ///
    /// `INFO`, like a retry's `allama::retry` event: the `WARN` line of a
    /// failover is its [`WarningKind::Fallback`] warning event
    /// (`allama::warning::fallback`), which also names who served, so one
    /// failover logs one `WARN`.
    fn fail_over(&self, error: &Error) {
        call_scope::record(CallFact::failover(error));
        #[cfg(any(feature = "tracing", feature = "warning-events"))]
        tracing::event!(
            target: "allama::fallback",
            tracing::Level::INFO,
            "allama.attempt.provider" =
                super::semconv_provider_name(self.primary.provider_name()),
            "error.type" = error.kind_name(),
            "allama.fallback.provider" =
                super::semconv_provider_name(self.fallback.provider_name()),
            "allama.fallback"
        );
    }

    /// The warning a response served after a failover carries, attributed
    /// to the primary: how it failed, and which provider served
    /// ("\[anthropic\] failed with `auth`; served by `ollama`"). `served` is
    /// the provider the response reports, so behind a nested fallback it
    /// names the innermost side that served. `emit` emits it as a warning
    /// event for a call; a validation only reports it.
    fn failover_warning(&self, model: &str, error: &Error, served: &str, emit: bool) -> Warning {
        let primary = self.primary.provider_name();
        let mut warning = Warning::new(
            WarningKind::Fallback,
            Subject::Response,
            format!("failed with `{}`; served by `{served}`", error.kind_name()),
        )
        .with_provider(primary.to_owned())
        .with_cause(WarningCause::of(error));
        if emit {
            crate::warning::emit(primary, model, &mut warning);
        }
        warning
    }

    /// Validation through the classifier, as the call would go (see the
    /// type's documentation).
    fn pair_validation<C: Capability>(
        &self,
        model: &str,
        primary: Result<Validation<C>>,
        fallback: impl FnOnce() -> Result<Validation<C>>,
    ) -> Result<Validation<C>> {
        let fallback_name = self.fallback.provider_name().to_owned();
        match primary {
            Ok(validation) => {
                Ok(validation.with_fallback(FallbackValidation::new(fallback_name, fallback())))
            }
            Err(error) if self.should_fail_over(&error) => match fallback() {
                Ok(validation) => Ok(validation.with_warning(self.failover_warning(
                    model,
                    &error,
                    self.fallback.provider_name(),
                    false,
                ))),
                Err(fallback_error) => Err(fallback_error.with_fallback_from(error)),
            },
            Err(error) => Err(error),
        }
    }
}

/// Record `error` from `provider` as the oldest failed attempt in
/// `provider_data`.
///
/// Attempts already there come from a `FallbackProvider` nested on the
/// fallback side, which ran after this side's primary failed.
fn record_attempt(provider_data: &mut ProviderData, provider: &str, error: &Error) {
    let mut attempts = provider_data
        .get::<FallbackAttempts>()
        .ok()
        .flatten()
        .unwrap_or_default();
    attempts.0.insert(
        0,
        FallbackAttempt {
            provider: provider.to_owned(),
            error_kind: error.kind_name().to_owned(),
            error_message: error.as_log().to_string(),
        },
    );
    let _ = provider_data.insert(&attempts);
}

/// Record the primary's failed attempt in the committed prefix of the
/// fallback stream.
///
/// The prefix holds every event before the first output, so the stream
/// order is kept by placing the attempts there: a new `ResponseMetadata`
/// event, reporting the attempt and the fallback as the serving provider,
/// goes right after the fallback's `ResponseStart` (first when there is
/// none). A pre-output error never reaches here: it fails the fallback
/// before the stream is committed.
///
/// A `FallbackProvider` nested on the fallback side has already placed such
/// an event in the prefix; the attempt is added to it instead, keeping the
/// serving provider it reports.
///
/// Returns the serving provider: the one a nested fallback reported (the
/// innermost wins), or `fallback`.
fn record_stream_attempt(
    buffered: &mut Vec<StreamEvent>,
    primary: &str,
    fallback: &str,
    error: &Error,
) -> String {
    let nested = buffered.iter_mut().rev().find_map(|event| match event {
        StreamEvent::ResponseMetadata {
            provider,
            provider_data,
            ..
        } if matches!(provider_data.get::<FallbackAttempts>(), Ok(Some(_))) => {
            Some((provider.as_deref().map(str::to_owned), provider_data))
        }
        _ => None,
    });
    if let Some((served, provider_data)) = nested {
        record_attempt(provider_data, primary, error);
        return served.unwrap_or_else(|| fallback.to_owned());
    }

    let mut provider_data = ProviderData::new();
    record_attempt(&mut provider_data, primary, error);
    let attempts_event = MetadataBuilder::new()
        .with_provider(fallback.to_owned())
        .with_provider_data(provider_data)
        .build();
    let after_start = buffered
        .iter()
        .position(|event| matches!(event, StreamEvent::ResponseStart { .. }))
        .map_or(0, |start| start + 1);
    buffered.insert(after_start, attempts_event);
    fallback.to_owned()
}

/// Run one side's call in a [`call_scope::side`], reporting the side as
/// tried when it starts and as the serving provider when it succeeds.
fn side<'a, S, T>(
    provider: &'a S,
    call: impl Future<Output = Result<T>> + 'a,
) -> impl Future<Output = Result<T>> + 'a
where
    S: ProviderIdentity + ?Sized,
    T: 'a,
{
    // Built before the async block, so the block holds the boxed side rather
    // than another copy of the call.
    let side = call_scope::side(async move {
        call_scope::record(CallFact::tried(provider.provider_name().to_owned()));
        call.await
    });
    async move {
        let result = side.await;
        if result.is_ok() {
            call_scope::record(CallFact::served_by(provider.provider_name().to_owned()));
        }
        result
    }
}

/// A stream call up to its commit point: the stream with its committed
/// prefix, or the error that came before the first output.
async fn committed<S>(
    call: impl Future<Output = Result<S>>,
) -> Result<crate::chat::stream::Prefixed<std::pin::Pin<Box<S>>>>
where
    S: futures_core::Stream<Item = Result<StreamEvent>>,
{
    let stream = call.await?;
    until_committed(Box::pin(stream)).await
}

/// The default failover rule of [`FallbackProvider`]: whether `error` is
/// worth failing over.
///
/// Fails over when another provider may well serve the request: an outage
/// or overload, a rate limit or exhausted quota, a timeout, a provider
/// failure, a request the primary cannot serve ([`Error::Unsupported`],
/// before sending or from the provider), and a prompt over the primary's
/// context window ([`Error::ContextLengthExceeded`]: windows differ across
/// providers and are not exposed uniformly, so the backup is tried, at the
/// cost of one more call).
///
/// Does not fail over on:
///
/// - caller mistakes ([`Error::InvalidRequest`]) and client
///   misconfiguration ([`Error::Configuration`]), which the backup would
///   reject too;
/// - rejected credentials ([`Error::Auth`]): a key problem is the
///   operator's to fix, and with one key per vendor the backup's is a
///   different key (see the recipe below);
/// - a model the provider does not know ([`Error::ModelNotFound`]): failing
///   over on a mistyped model id would hide the mistake;
/// - a content filter ([`Error::ContentFiltered`]): sending a blocked
///   prompt to another vendor is a policy decision, not a recovery (see the
///   recipe below);
/// - a request the client closed (an [`Error::Provider`] with status
///   `499`): the cancellation came from the caller's side, so no provider
///   would serve anyone, and it is not retried either;
/// - responses or payloads that could not be understood
///   ([`Error::UnexpectedResponse`], [`Error::Serialization`]), and, with
///   the `extract` feature, extraction failures.
///
/// [`Error::Unsupported`] fails over at every
/// [stage](Error::unsupported_stage): an adapter's refusal before sending,
/// and a provider's own refusal that an adapter maps
/// ([`UnsupportedStage::Provider`](crate::UnsupportedStage::Provider)), since
/// another provider may support what this one refused.
///
/// An error that ended a nested failover is judged as itself, the last
/// side's error, whatever it [fell back from](Error::fallback_from). A
/// refusal or a filtered *response* is an `Ok` (see
/// [`ChatResponse::finished_normally`]), so no classifier ever sees one.
///
/// Compose with it in [`FallbackProvider::fallback_if`] to opt in to more:
///
/// ```
/// # #[cfg(feature = "mock")] {
/// use allama::Error;
/// use allama::wrappers::{FallbackProvider, should_fallback};
/// # use allama::mock::MockProvider;
/// # let (primary, backup) = (MockProvider::with_text("a"), MockProvider::with_text("b"));
///
/// // A blocked prompt may go to the backup vendor, whose policy differs.
/// let lenient = FallbackProvider::new(primary.clone(), backup.clone())
///     .fallback_if(|error: &Error| {
///         matches!(error, Error::ContentFiltered { .. }) || should_fallback(error)
///     });
///
/// // With a key per vendor, a rejected key on the primary does not affect
/// // the backup's; the failover warning keeps the problem visible.
/// let per_vendor_keys = FallbackProvider::new(primary, backup).fallback_if(|error: &Error| {
///     matches!(error, Error::Auth { .. }) || should_fallback(error)
/// });
/// # let _ = (lenient, per_vendor_keys); }
/// ```
///
/// A local policy refusal, such as a budget guard, returns
/// [`Error::Rejected`], which never fails over: a refused call is refused
/// on every side. Placing the guard *outside* the `FallbackProvider` also
/// keeps a refused request from reaching either side.
#[must_use]
pub fn should_fallback(error: &Error) -> bool {
    match error {
        Error::Auth { .. }
        | Error::Configuration { .. }
        | Error::Rejected { .. }
        | Error::ContentFiltered { .. }
        | Error::InvalidRequest { .. }
        | Error::ModelNotFound { .. }
        | Error::UnexpectedResponse { .. }
        | Error::Serialization { .. } => false,
        Error::Provider { .. } if error.status() == Some(CLIENT_CLOSED_REQUEST) => false,
        #[cfg(feature = "chat-extract")]
        Error::Extract { .. } => false,
        _ => true,
    }
}

/// The status of a request the client closed (nginx's `499`).
const CLIENT_CLOSED_REQUEST: u16 = 499;

/// Either side may serve a request, so the pair knows an answer only when
/// both sides give it; otherwise it is `Unknown` (a wrapper claims no more
/// than the side that would serve).
fn merge_capability_support(
    primary: CapabilitySupport,
    fallback: CapabilitySupport,
) -> CapabilitySupport {
    if primary == fallback {
        primary
    } else {
        CapabilitySupport::Unknown
    }
}

/// The smaller of two limits, where `None` means no known limit.
fn smaller_limit(primary: Option<usize>, fallback: Option<usize>) -> Option<usize> {
    match (primary, fallback) {
        (Some(primary), Some(fallback)) => Some(primary.min(fallback)),
        (primary, fallback) => primary.or(fallback),
    }
}

/// `response`, attributed to `provider` unless it names its provider.
fn served_by(mut response: EmbeddingResponse, provider: &str) -> EmbeddingResponse {
    if response.provider.is_none() {
        response.provider = Some(provider.to_owned().into());
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::stream::CollectedResponse;
    use crate::mock::{ChatResponseBuilder, MockProvider, MockStreamEvent, MockStreamingProvider};
    use crate::{ChatStreamExt, StreamErrorKind, WarningKind};

    /// Collect whatever `stream` yields, keeping its terminal error.
    async fn collect_partial(stream: impl ChatStreamExt + Unpin) -> CollectedResponse {
        let mut stream = stream.with_collector();
        while stream.next_event().await.is_some() {}
        stream.finish_partial().unwrap()
    }

    fn request() -> ChatRequest {
        ChatRequest::new("m").user("hi")
    }

    #[test]
    fn services_query_primary_then_fallback() {
        use crate::ProviderIdentityExt;
        use crate::mock::FunctionProvider;

        let wrapper = FallbackProvider::new(
            MockProvider::with_text("primary"),
            FunctionProvider::new(|_| Ok(ChatResponse::default())),
        );
        assert!(wrapper.downcast_ref::<MockProvider>().is_some());
        assert!(wrapper.downcast_ref::<FunctionProvider>().is_some());
        assert!(wrapper.downcast_ref::<MockStreamingProvider>().is_none());
    }

    #[tokio::test]
    async fn primary_success_skips_fallback() {
        let fallback = MockProvider::with_text("fallback");
        let wrapper = FallbackProvider::new(MockProvider::with_text("primary"), fallback.clone());
        let response = wrapper.chat(&request()).await.unwrap();
        assert_eq!(response.text().as_deref(), Some("primary"));
        assert_eq!(fallback.call_count(), 0);
        assert!(response.provider_data.is_empty());
    }

    #[tokio::test]
    async fn retryable_error_fails_over_and_records_attempt() {
        let wrapper = FallbackProvider::new(
            MockProvider::with_error(Error::overloaded("down")).with_provider_name("primary"),
            MockProvider::with_text("fallback").with_provider_name("backup"),
        );
        let response = wrapper.chat(&request()).await.unwrap();
        assert_eq!(response.text().as_deref(), Some("fallback"));
        assert_eq!(response.provider.as_deref(), Some("backup"));
        let attempts = response
            .provider_data
            .get::<FallbackAttempts>()
            .unwrap()
            .unwrap();
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts.as_slice()[0].provider, "primary");
        assert_eq!(attempts.as_slice()[0].error_kind, "overloaded");
    }

    #[tokio::test]
    async fn caller_errors_do_not_fail_over() {
        let fallback = MockProvider::with_text("fallback");
        let wrapper = FallbackProvider::new(
            MockProvider::with_error(Error::invalid_request("bad")),
            fallback.clone(),
        );
        assert!(matches!(
            wrapper.chat(&request()).await.unwrap_err(),
            Error::InvalidRequest { .. }
        ));
        assert_eq!(fallback.call_count(), 0);
    }

    #[tokio::test]
    async fn local_refusals_do_not_fail_over() {
        let fallback = MockProvider::with_text("fallback");
        let wrapper = FallbackProvider::new(
            MockProvider::with_error(Error::rejected("tenant budget exhausted")),
            fallback.clone(),
        );
        assert!(matches!(
            wrapper.chat(&request()).await.unwrap_err(),
            Error::Rejected { .. }
        ));
        assert_eq!(fallback.call_count(), 0);
    }

    #[tokio::test]
    async fn configuration_errors_do_not_fail_over() {
        let fallback = MockProvider::with_text("fallback");
        let wrapper = FallbackProvider::new(
            MockProvider::with_error(Error::configuration("API key not set")),
            fallback.clone(),
        );
        assert!(matches!(
            wrapper.chat(&request()).await.unwrap_err(),
            Error::Configuration { .. }
        ));
        assert_eq!(fallback.call_count(), 0);
    }

    /// An application's own mapping of errors to HTTP statuses, as the
    /// `Error` docs show it: a plain `match`, with no failover case.
    fn http_status(error: &Error) -> u16 {
        match error {
            Error::RateLimited { .. } => 429,
            Error::InvalidRequest { .. } => 400,
            _ if error.is_retryable() => 503,
            _ => 502,
        }
    }

    /// The kinds along an error's `fallback_from` chain, newest first.
    fn tried(error: &Error) -> Vec<&'static str> {
        std::iter::successors(error.fallback_from(), |e| e.fallback_from())
            .map(Error::kind_name)
            .collect()
    }

    /// Regression: a failed failover was an
    /// `Error::Fallback`, so a plain `match` answered a 429 with a 502, and
    /// `kind_name()` said `"fallback"`.
    #[tokio::test]
    async fn both_failing_returns_the_backups_error_as_itself() {
        let wrapper = FallbackProvider::new(
            MockProvider::with_error(Error::overloaded("HTTP 529").with_status(529))
                .with_provider_name("primary"),
            MockProvider::with_error(
                Error::rate_limited("HTTP 429")
                    .with_status(429)
                    .with_provider("backup"),
            )
            .with_provider_name("backup"),
        );
        let validation = wrapper
            .validate_chat_request(&request())
            .expect("both sides accept the request");
        assert!(validation.fallback().is_some());

        let err = wrapper.chat(&request()).await.unwrap_err();
        assert!(matches!(err, Error::RateLimited { .. }), "{err:?}");
        assert_eq!(http_status(&err), 429);
        assert_eq!(err.kind_name(), "rate_limited");
        assert_eq!(err.status(), Some(429));
        assert!(err.is_retryable());
        assert_eq!(err.to_string(), "backup: rate limited: HTTP 429");

        let primary = err.fallback_from().expect("the primary's error");
        assert!(matches!(primary, Error::Overloaded { .. }), "{primary:?}");
        assert_eq!(primary.status(), Some(529));
        assert!(primary.fallback_from().is_none());
        assert_eq!(
            err.as_log().to_string(),
            "kind=rate_limited provider=backup status=429 \
             fallback_from=[kind=overloaded status=529]"
        );

        // The stream path returns the same shape.
        let primary_stream = MockStreamingProvider::build(|b| {
            b.stream([MockStreamEvent::Error(Error::overloaded("HTTP 529"))])
        });
        let backup_stream = MockStreamingProvider::build(|b| {
            b.stream([MockStreamEvent::Error(
                Error::rate_limited("HTTP 429").with_status(429),
            )])
        });
        let wrapper = FallbackProvider::new(primary_stream, backup_stream);
        let err = match wrapper.chat_stream(&request()).await {
            Err(err) => err,
            Ok(_) => panic!("both sides fail before output"),
        };
        assert_eq!(http_status(&err), 429);
        assert_eq!(tried(&err), ["overloaded"]);
    }

    #[tokio::test]
    async fn nested_fallbacks_return_the_last_error_with_every_earlier_one_newest_first() {
        let side = |error: Error, name: &'static str| {
            MockProvider::with_error(error.with_provider(name)).with_provider_name(name)
        };
        let a = || side(Error::timeout("a"), "a");
        let b = || side(Error::overloaded("b"), "b");
        let c = || side(Error::rate_limited("c").with_status(429), "c");

        // Nested on the fallback side: the outer primary goes at the tail.
        let on_fallback = FallbackProvider::new(a(), FallbackProvider::new(b(), c()));
        // Nested on the primary side: the inner chain comes along.
        let on_primary = FallbackProvider::new(FallbackProvider::new(a(), b()), c());
        for err in [
            on_fallback.chat(&request()).await.unwrap_err(),
            on_primary.chat(&request()).await.unwrap_err(),
        ] {
            assert!(matches!(err, Error::RateLimited { .. }), "{err:?}");
            assert_eq!(http_status(&err), 429);
            assert_eq!(tried(&err), ["overloaded", "timeout"]);
            let providers: Vec<_> = std::iter::successors(Some(&err), |e| e.fallback_from())
                .map(|e| e.context().unwrap().provider.clone().unwrap())
                .collect();
            assert_eq!(providers, ["c", "b", "a"]);
            assert_eq!(
                err.as_log().to_string(),
                "kind=rate_limited provider=c status=429 fallback_from=[kind=overloaded \
                 provider=b fallback_from=[kind=timeout provider=a]]"
            );
        }
    }

    #[tokio::test]
    async fn a_last_error_without_a_context_drops_the_earlier_ones() {
        let wrapper = FallbackProvider::new(
            MockProvider::with_error(Error::overloaded("busy")),
            MockProvider::with_error(Error::serialization("bad option")),
        );
        let err = wrapper.chat(&request()).await.unwrap_err();
        assert!(matches!(err, Error::Serialization { .. }), "{err:?}");
        assert!(err.fallback_from().is_none());
    }

    #[tokio::test]
    async fn stream_fails_over_before_commit() {
        let primary = MockStreamingProvider::build(|b| {
            b.stream([
                MockStreamEvent::Event(StreamEvent::ResponseStart {
                    id: None,
                    model: None,
                }),
                MockStreamEvent::Error(Error::overloaded("down")),
            ])
        })
        .with_provider_name("primary");
        let fallback =
            MockProvider::with_response(ChatResponseBuilder::new().text("backup").build())
                .with_provider_name("backup");
        let wrapper = FallbackProvider::new(primary, fallback);
        let response = wrapper
            .chat_stream(&request())
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        assert_eq!(response.text().as_deref(), Some("backup"));
        assert_eq!(response.provider.as_deref(), Some("backup"));
        assert!(
            response
                .provider_data
                .get::<FallbackAttempts>()
                .unwrap()
                .is_some()
        );
    }

    #[tokio::test]
    async fn unknown_model_does_not_fail_over_by_default() {
        let fallback = MockProvider::with_text("fallback");
        let wrapper = FallbackProvider::new(
            MockProvider::with_error(Error::model_not_found("retired-model")),
            fallback.clone(),
        );
        assert!(matches!(
            wrapper.chat(&request()).await.unwrap_err(),
            Error::ModelNotFound { .. }
        ));
        assert_eq!(fallback.call_count(), 0);
    }

    fn or_unknown_model(error: &Error) -> bool {
        matches!(error, Error::ModelNotFound { .. }) || should_fallback(error)
    }

    #[tokio::test]
    async fn custom_classifier_fails_over_on_an_unknown_model() {
        let wrapper = FallbackProvider::new(
            MockProvider::with_error(Error::model_not_found("retired-model"))
                .with_provider_name("primary"),
            MockProvider::with_text("fallback").with_provider_name("backup"),
        )
        .fallback_if(or_unknown_model);
        let response = wrapper.chat(&request()).await.unwrap();
        assert_eq!(response.text().as_deref(), Some("fallback"));
        assert_eq!(response.provider.as_deref(), Some("backup"));
        let attempts = response
            .provider_data
            .get::<FallbackAttempts>()
            .unwrap()
            .unwrap();
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts.as_slice()[0].provider, "primary");
        assert_eq!(attempts.as_slice()[0].error_kind, "model_not_found");

        // The default rule still applies to everything else.
        let fallback = MockProvider::with_text("fallback");
        let wrapper = FallbackProvider::new(
            MockProvider::with_error(Error::invalid_request("bad")),
            fallback.clone(),
        )
        .fallback_if(or_unknown_model);
        assert!(wrapper.chat(&request()).await.is_err());
        assert_eq!(fallback.call_count(), 0);
    }

    #[tokio::test]
    async fn custom_classifier_applies_to_streams() {
        let primary = MockStreamingProvider::build(|b| {
            b.stream([MockStreamEvent::Error(Error::model_not_found("retired"))])
        })
        .with_provider_name("primary");
        let fallback =
            MockProvider::with_response(ChatResponseBuilder::new().text("backup").build())
                .with_provider_name("backup");
        let wrapper = FallbackProvider::new(primary, fallback).fallback_if(or_unknown_model);
        let response = wrapper
            .chat_stream(&request())
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        assert_eq!(response.text().as_deref(), Some("backup"));
        let attempts = response
            .provider_data
            .get::<FallbackAttempts>()
            .unwrap()
            .unwrap();
        assert_eq!(attempts.as_slice()[0].error_kind, "model_not_found");

        // A classifier that refuses everything keeps a stream on its primary.
        let primary = MockStreamingProvider::build(|b| {
            b.stream([MockStreamEvent::Error(Error::overloaded("busy"))])
        });
        let fallback = MockProvider::with_text("backup");
        let wrapper = FallbackProvider::new(primary, fallback.clone()).fallback_if(|_| false);
        assert!(wrapper.chat_stream(&request()).await.is_err());
        assert_eq!(fallback.call_count(), 0);
    }

    #[tokio::test]
    async fn an_embedding_names_the_side_that_served_it() {
        use crate::mock::MockEmbeddingProvider;

        let request = EmbeddingRequest::new("e").inputs(["a"]);
        let wrapper = FallbackProvider::new(
            MockEmbeddingProvider::with_error(Error::overloaded("down"))
                .with_provider_name("primary"),
            MockEmbeddingProvider::with_vectors(vec![vec![1.0]]).with_provider_name("backup"),
        );
        let response = wrapper.embed(&request).await.unwrap();
        assert_eq!(response.provider.as_deref(), Some("backup"));

        let wrapper = FallbackProvider::new(
            MockEmbeddingProvider::with_vectors(vec![vec![1.0]]).with_provider_name("primary"),
            MockEmbeddingProvider::with_vectors(vec![vec![2.0]]).with_provider_name("backup"),
        );
        let response = wrapper.embed(&request).await.unwrap();
        assert_eq!(response.provider.as_deref(), Some("primary"));

        // A side that names its provider keeps it: an inner fallback's
        // serving side, or a gateway's upstream.
        let wrapper = FallbackProvider::new(
            MockEmbeddingProvider::with_error(Error::overloaded("down"))
                .with_provider_name("primary"),
            MockEmbeddingProvider::new([Ok(
                EmbeddingResponse::new(vec![vec![1.0]]).provider("upstream")
            )])
            .with_provider_name("gateway"),
        );
        let response = wrapper.embed(&request).await.unwrap();
        assert_eq!(response.provider.as_deref(), Some("upstream"));
        // The failover warning names the provider that served, too.
        let failover = response
            .warnings
            .iter()
            .find(|w| w.kind == WarningKind::Fallback)
            .unwrap();
        assert_eq!(
            failover.to_string(),
            "[primary] failed with `overloaded`; served by `upstream`"
        );
    }

    /// A chunked call whose primary fails partway through mixes vectors from
    /// two providers; the combined response says so.
    #[tokio::test]
    async fn a_chunked_embedding_across_a_failover_warns() {
        use crate::EmbeddingProviderExt;
        use crate::mock::MockEmbeddingProvider;

        let wrapper = FallbackProvider::new(
            MockEmbeddingProvider::new([
                Ok(EmbeddingResponse::new(vec![vec![1.0]])),
                Err(Error::overloaded("down")),
            ])
            .with_provider_name("primary"),
            MockEmbeddingProvider::with_vectors(vec![vec![2.0]]).with_provider_name("backup"),
        );
        let request = EmbeddingRequest::new("e").inputs(["a", "b"]);
        let response = wrapper.embed_chunked(&request, 1).await.unwrap();

        assert_eq!(response.embeddings, vec![vec![1.0], vec![2.0]]);
        assert_eq!(response.provider.as_deref(), Some("primary"));
        let mixed: Vec<_> = response
            .warnings
            .iter()
            .filter(|warning| warning.message.contains("may not be comparable"))
            .collect();
        assert_eq!(mixed.len(), 1, "{:?}", response.warnings);
        assert_eq!(mixed[0].kind, WarningKind::Fallback);
        assert!(
            mixed[0].message.contains("`primary` and `backup`"),
            "{}",
            mixed[0].message
        );
    }

    #[tokio::test]
    async fn custom_classifier_applies_to_embeddings_and_decisions() {
        use crate::decision::{Answer, Question, YesNoAnswer};
        use crate::mock::{MockDecisionProvider, MockEmbeddingProvider};

        let wrapper = FallbackProvider::new(
            MockEmbeddingProvider::with_error(Error::model_not_found("retired")),
            MockEmbeddingProvider::with_vectors(vec![vec![1.0]]),
        )
        .fallback_if(or_unknown_model);
        let response = wrapper
            .embed(&EmbeddingRequest::new("e").inputs(["a"]))
            .await
            .unwrap();
        assert_eq!(response.embeddings.len(), 1);
        assert!(
            response
                .provider_data
                .get::<FallbackAttempts>()
                .unwrap()
                .is_some()
        );

        let wrapper = FallbackProvider::new(
            MockDecisionProvider::with_error(Error::model_not_found("retired")),
            MockDecisionProvider::with_response(DecisionResponse::new([(
                "q",
                Answer::from(YesNoAnswer::new(0.9)),
            )])),
        )
        .fallback_if(or_unknown_model);
        let request = DecisionRequest::new("d", "x").question("q", Question::yes_no("?"));
        let response = wrapper.decide(&request).await.unwrap();
        let attempts = response
            .provider_data
            .get::<FallbackAttempts>()
            .unwrap()
            .unwrap();
        assert_eq!(attempts.as_slice()[0].error_kind, "model_not_found");
    }

    /// A primary that is itself a fallback chain ending in an unknown model.
    fn chain_ending_in_unknown_model() -> FallbackProvider<MockProvider, MockProvider> {
        FallbackProvider::new(
            MockProvider::with_error(Error::timeout("slow")).with_provider_name("first"),
            MockProvider::with_error(Error::model_not_found("retired"))
                .with_provider_name("second"),
        )
    }

    #[tokio::test]
    async fn custom_classifier_sees_the_error_that_ended_a_nested_chain() {
        // The default judges the chain by its last error and stops.
        let third = MockProvider::with_text("third").with_provider_name("third");
        let wrapper = FallbackProvider::new(chain_ending_in_unknown_model(), third.clone());
        let err = wrapper.chat(&request()).await.unwrap_err();
        assert!(matches!(err, Error::ModelNotFound { .. }), "{err:?}");
        assert!(matches!(err.fallback_from(), Some(Error::Timeout { .. })));
        assert_eq!(third.call_count(), 0);

        // A classifier receives the unknown model itself.
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let record = Arc::clone(&seen);
        let wrapper = FallbackProvider::new(chain_ending_in_unknown_model(), third.clone())
            .fallback_if(move |error: &Error| {
                record.lock().unwrap().push(error.kind_name());
                or_unknown_model(error)
            });
        let response = wrapper.chat(&request()).await.unwrap();
        assert_eq!(response.text().as_deref(), Some("third"));
        assert_eq!(third.call_count(), 1);
        assert_eq!(*seen.lock().unwrap(), ["model_not_found"]);
        let attempts = response
            .provider_data
            .get::<FallbackAttempts>()
            .unwrap()
            .unwrap();
        assert_eq!(attempts.as_slice()[0].error_kind, "model_not_found");

        // A chain whose last error the classifier refuses does not fail over,
        // even though an earlier side failed with an eligible error.
        let third = MockProvider::with_text("third");
        let chain = FallbackProvider::new(
            MockProvider::with_error(Error::model_not_found("retired")),
            MockProvider::with_error(Error::invalid_request("bad")),
        )
        .fallback_if(or_unknown_model);
        let wrapper = FallbackProvider::new(chain, third.clone()).fallback_if(or_unknown_model);
        let err = wrapper.chat(&request()).await.unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
        assert!(matches!(
            err.fallback_from(),
            Some(Error::ModelNotFound { .. })
        ));
        assert_eq!(third.call_count(), 0);
    }

    /// The default classifier, kind by kind.
    #[test]
    fn the_default_rule_by_error_kind() {
        let client_closed = Error::provider("HTTP 499").with_status(499);
        let table = [
            // Another provider may well serve it.
            (Error::overloaded("HTTP 529"), true),
            (Error::rate_limited("HTTP 429"), true),
            (Error::quota_exhausted("spend limit"), true),
            (Error::timeout("HTTP 504"), true),
            (Error::transport("connection refused"), true),
            (Error::provider("HTTP 500").with_status(500), true),
            (Error::provider("in-band failure"), true),
            (Error::unsupported("no structured output"), true),
            (Error::context_length_exceeded("too long", Some(8192)), true),
            (
                Error::stream(StreamErrorKind::Incomplete, "cut short"),
                true,
            ),
            // The request, the setup, or a policy decision: not recovered by
            // another provider.
            (Error::invalid_request("bad"), false),
            (Error::configuration("no key"), false),
            (Error::auth("HTTP 401"), false),
            (Error::model_not_found("retired"), false),
            (Error::content_filtered("blocked"), false),
            (client_closed, false),
            (Error::unexpected_response("HTML"), false),
        ];
        for (error, expected) in table {
            assert_eq!(should_fallback(&error), expected, "{}", error.as_log());
        }
    }

    #[tokio::test]
    async fn a_content_filter_does_not_fail_over_unless_opted_in() {
        let backup = MockProvider::with_text("backup");
        let wrapper = FallbackProvider::new(
            MockProvider::with_error(Error::content_filtered("blocked")),
            backup.clone(),
        );
        assert!(matches!(
            wrapper.chat(&request()).await.unwrap_err(),
            Error::ContentFiltered { .. }
        ));
        assert_eq!(backup.call_count(), 0);

        let opted_in = FallbackProvider::new(
            MockProvider::with_error(Error::content_filtered("blocked")),
            backup.clone(),
        )
        .fallback_if(|error: &Error| {
            matches!(error, Error::ContentFiltered { .. }) || should_fallback(error)
        });
        let response = opted_in.chat(&request()).await.unwrap();
        assert_eq!(response.text().as_deref(), Some("backup"));
        let cause = response.warnings[0].cause.as_ref().unwrap();
        assert_eq!(cause.kind, "content_filtered");
    }

    #[tokio::test]
    async fn failover_warnings_carry_their_cause() {
        let wrapper = || {
            FallbackProvider::new(
                MockProvider::with_error(Error::rate_limited("HTTP 429")),
                MockProvider::with_text("backup"),
            )
        };
        let unary = wrapper().chat(&request()).await.unwrap();
        let streamed = wrapper()
            .chat_stream(&request())
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        for response in [unary, streamed] {
            let failover = &response.warnings[0];
            assert_eq!(failover.kind, WarningKind::Fallback);
            assert_eq!(failover.cause, Some(WarningCause::new("rate_limited")));
        }

        // A chain that failed on both sides is charted by what ended it.
        let chain = FallbackProvider::new(
            FallbackProvider::new(
                MockProvider::with_error(Error::overloaded("down")),
                MockProvider::with_error(Error::timeout("slow")),
            ),
            MockProvider::with_text("third"),
        );
        let response = chain.chat(&request()).await.unwrap();
        let last = response.warnings.last().unwrap();
        assert_eq!(last.cause, Some(WarningCause::new("timeout")));
    }

    #[cfg(feature = "warning-events")]
    #[test]
    fn failover_events_carry_the_cause_kind() {
        let (_, events) = crate::warning::capture::captured(async {
            let wrapper = FallbackProvider::new(
                MockProvider::with_error(Error::quota_exhausted("spend limit")),
                MockProvider::with_text("backup"),
            );
            wrapper
                .chat(&ChatRequest::new("failover-cause").user("hi"))
                .await
                .unwrap()
        });
        let failover: Vec<_> = events
            .iter()
            .filter(|event| event.target == "allama::warning::fallback")
            .collect();
        assert_eq!(failover.len(), 1, "{events:?}");
        assert_eq!(failover[0].cause_kind.as_deref(), Some("quota_exhausted"));
        assert_eq!(failover[0].cause_stage, None);
    }

    /// A failover on a provider's own refusal says where it was refused.
    #[cfg(feature = "warning-events")]
    #[test]
    fn failover_events_carry_the_cause_stage() {
        let (_, events) = crate::warning::capture::captured(async {
            let refused = Error::unsupported_by_provider(
                Some(crate::Subject::Field {
                    field: crate::RequestField::Temperature,
                }),
                "HTTP 400",
            );
            let wrapper = FallbackProvider::new(
                MockProvider::with_error(refused),
                MockProvider::with_text("backup"),
            );
            wrapper
                .chat(&ChatRequest::new("failover-stage").user("hi"))
                .await
                .unwrap()
        });
        let failover: Vec<_> = events
            .iter()
            .filter(|event| event.target == "allama::warning::fallback")
            .collect();
        assert_eq!(failover.len(), 1, "{events:?}");
        assert_eq!(failover[0].cause_kind.as_deref(), Some("unsupported"));
        assert_eq!(failover[0].cause_stage.as_deref(), Some("provider"));
    }

    #[test]
    fn default_rule_judges_a_failed_failover_by_its_last_side() {
        assert!(should_fallback(
            &Error::overloaded("busy").with_fallback_from(Error::auth("bad key"))
        ));
        assert!(!should_fallback(
            &Error::model_not_found("retired").with_fallback_from(Error::overloaded("busy"))
        ));
        assert!(!should_fallback(&Error::model_not_found("retired")));
    }

    /// The names of the events a stream yields (their serde tags), and the
    /// attempts reported in its `ResponseMetadata` events.
    async fn stream_shape(stream: ChatStream) -> (Vec<String>, Vec<String>) {
        use futures_util::StreamExt;

        let events: Vec<_> = stream.map(Result::unwrap).collect().await;
        let attempts = events
            .iter()
            .filter_map(|event| match event {
                StreamEvent::ResponseMetadata { provider_data, .. } => {
                    provider_data.get::<FallbackAttempts>().unwrap()
                }
                _ => None,
            })
            .flatten()
            .map(|attempt| attempt.provider)
            .collect();
        let names = events
            .iter()
            .map(|event| {
                serde_json::to_value(event).unwrap()["event"]
                    .as_str()
                    .unwrap()
                    .to_owned()
            })
            .collect();
        (names, attempts)
    }

    fn failing_primary() -> MockProvider {
        MockProvider::with_error(Error::overloaded("down")).with_provider_name("primary")
    }

    #[tokio::test]
    async fn failed_over_stream_reports_attempts_after_response_start() {
        let fallback = MockStreamingProvider::from_responses([ChatResponseBuilder::new()
            .text("backup")
            .usage(1, 1)
            .build()])
        .with_provider_name("backup");
        let wrapper = FallbackProvider::new(failing_primary(), fallback);
        let stream = wrapper.chat_stream(&request()).await.unwrap();
        let (names, attempts) = stream_shape(stream).await;
        assert_eq!(
            names,
            [
                "response_start",
                "response_metadata",
                "warning",
                "block_start",
                "text_delta",
                "block_stop",
                "response_metadata",
                "response_stop",
            ]
        );
        assert_eq!(attempts, ["primary"]);
    }

    #[tokio::test]
    async fn failed_over_stream_without_a_start_reports_attempts_first() {
        let fallback = MockStreamingProvider::with_stream([
            MockStreamEvent::Event(StreamEvent::block_start(
                0,
                crate::chat::stream::StreamBlockType::Text,
            )),
            MockStreamEvent::Event(StreamEvent::text_delta(0, "backup")),
            MockStreamEvent::Event(StreamEvent::block_stop(0)),
            MockStreamEvent::Event(StreamEvent::ResponseStop),
        ]);
        let wrapper = FallbackProvider::new(failing_primary(), fallback);
        let stream = wrapper.chat_stream(&request()).await.unwrap();
        let (names, attempts) = stream_shape(stream).await;
        assert_eq!(
            names,
            [
                "response_metadata",
                "warning",
                "block_start",
                "text_delta",
                "block_stop",
                // The stream reports no finish reason, and no provider.
                "warning",
                "response_metadata",
                "response_stop"
            ]
        );
        assert_eq!(attempts, ["primary"]);
    }

    /// A chain adds its attempts to the event the inner fallback placed
    /// after `ResponseStart`, rather than adding another.
    #[tokio::test]
    async fn failed_over_chain_reports_attempts_in_one_event() {
        let stream = chain().chat_stream(&request()).await.unwrap();
        let (names, attempts) = stream_shape(stream).await;
        // One failover warning per link of the chain.
        assert_eq!(
            names[..5],
            [
                "response_start",
                "response_metadata",
                "warning",
                "warning",
                "block_start"
            ]
        );
        assert_eq!(
            names
                .iter()
                .filter(|name| **name == "response_metadata")
                .count(),
            2,
            "{names:?}"
        );
        assert_eq!(attempts, ["first", "second"]);
    }

    #[tokio::test]
    async fn stream_errors_after_commit_pass_through() {
        let primary = MockStreamingProvider::build(|b| {
            b.stream([
                MockStreamEvent::Event(StreamEvent::ResponseStart {
                    id: None,
                    model: None,
                }),
                MockStreamEvent::Event(StreamEvent::TextDelta {
                    index: 0,
                    text: "partial".into(),
                }),
                MockStreamEvent::Error(Error::overloaded("down")),
            ])
        });
        let fallback = MockProvider::with_text("backup");
        let wrapper = FallbackProvider::new(primary, fallback.clone());
        let result = wrapper
            .chat_stream(&request())
            .await
            .unwrap()
            .collect_response()
            .await;
        assert!(result.is_err());
        assert_eq!(fallback.call_count(), 0);
    }

    /// A refusal-only stream that fails after its first delta has committed:
    /// the backup never answers in its place.
    #[tokio::test]
    async fn a_streamed_refusal_is_never_failed_over() {
        let primary = MockStreamingProvider::build(|b| {
            b.stream([
                MockStreamEvent::Event(StreamEvent::response_start(None, None)),
                MockStreamEvent::Event(StreamEvent::refusal_delta(0, "I can't")),
                MockStreamEvent::Error(Error::overloaded("down")),
            ])
        });
        let fallback = MockProvider::with_text("backup");
        let wrapper = FallbackProvider::new(primary, fallback.clone());
        let partial = collect_partial(wrapper.chat_stream(&request()).await.unwrap()).await;
        assert!(partial.terminal_error.is_some());
        assert_eq!(partial.response.first_refusal(), Some("I can't"));
        assert_eq!(fallback.call_count(), 0);
    }

    /// `Fallback(first, Fallback(second, third))`, where only `third` serves.
    fn chain() -> FallbackProvider<MockProvider, FallbackProvider<MockProvider, MockProvider>> {
        FallbackProvider::new(
            MockProvider::with_error(Error::overloaded("down")).with_provider_name("first"),
            FallbackProvider::new(
                MockProvider::with_error(Error::timeout("slow")).with_provider_name("second"),
                MockProvider::with_text("third").with_provider_name("third"),
            ),
        )
    }

    fn attempted(response: &ChatResponse) -> Vec<(String, String)> {
        response
            .provider_data
            .get::<FallbackAttempts>()
            .unwrap()
            .unwrap()
            .into_iter()
            .map(|attempt| (attempt.provider, attempt.error_kind))
            .collect()
    }

    #[tokio::test]
    async fn a_chain_records_every_attempt_oldest_first() {
        let expected = [
            ("first".to_owned(), "overloaded".to_owned()),
            ("second".to_owned(), "timeout".to_owned()),
        ];

        let response = chain().chat(&request()).await.unwrap();
        assert_eq!(attempted(&response), expected);
        assert_eq!(response.provider.as_deref(), Some("third"));

        let response = chain()
            .chat_stream(&request())
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        assert_eq!(attempted(&response), expected);
        assert_eq!(response.provider.as_deref(), Some("third"));
    }

    /// Each layer's warning is attributed to its own primary and names the
    /// provider that served, the innermost one, on both paths.
    #[tokio::test]
    async fn a_chain_names_the_innermost_server_in_every_failover_warning() {
        let unary = chain().chat(&request()).await.unwrap();
        let streamed = chain()
            .chat_stream(&request())
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        for response in [unary, streamed] {
            let failovers: Vec<_> = response
                .warnings
                .iter()
                .filter(|w| w.kind == WarningKind::Fallback)
                .map(ToString::to_string)
                .collect();
            assert_eq!(
                failovers,
                [
                    "[second] failed with `timeout`; served by `third`",
                    "[first] failed with `overloaded`; served by `third`",
                ]
            );
        }
    }

    #[test]
    fn batch_limit_is_the_smaller_known_limit() {
        use crate::mock::MockEmbeddingProvider;

        let limited = |max| MockEmbeddingProvider::empty().with_max_inputs_per_call(max);
        let unlimited = MockEmbeddingProvider::empty;
        let limit = |primary, fallback| {
            FallbackProvider::new(primary, fallback)
                .embedding_limits("m")
                .max_inputs_per_call
        };
        assert_eq!(limit(limited(100), limited(10)), Some(10));
        assert_eq!(limit(limited(10), limited(100)), Some(10));
        assert_eq!(limit(limited(10), unlimited()), Some(10));
        assert_eq!(limit(unlimited(), limited(10)), Some(10));
        assert_eq!(limit(unlimited(), unlimited()), None);
    }

    #[test]
    fn capabilities_are_an_agreement() {
        use CapabilitySupport::*;
        for answer in [Supported, Unsupported, Unknown] {
            assert_eq!(merge_capability_support(answer, answer), answer);
        }
        assert_eq!(merge_capability_support(Unsupported, Supported), Unknown);
        assert_eq!(merge_capability_support(Supported, Unsupported), Unknown);
        assert_eq!(merge_capability_support(Unknown, Unsupported), Unknown);
        assert_eq!(merge_capability_support(Supported, Unknown), Unknown);
        assert_eq!(merge_capability_support(Unknown, Supported), Unknown);
    }

    fn with_tools() -> ChatRequest {
        request().tools([crate::ToolDefinition::new(
            "t",
            serde_json::json!({"type": "object"}),
        )])
    }

    fn with_tool_support(name: &'static str) -> MockProvider {
        MockProvider::with_text(name)
            .with_provider_name(name)
            .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported)
    }

    fn without_tool_support(name: &'static str) -> MockProvider {
        MockProvider::with_text(name)
            .with_provider_name(name)
            .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Unsupported)
    }

    #[test]
    fn an_accepting_primary_gives_its_validation_and_the_backups_dry_run() {
        let wrapper =
            FallbackProvider::new(with_tool_support("tools"), without_tool_support("plain"));
        let validation = wrapper.validate_chat_request(&with_tools()).unwrap();
        // What the call would report: the primary serves.
        assert!(validation.warnings().is_empty());
        let fallback = validation.fallback().unwrap();
        assert_eq!(fallback.provider, "plain");
        let refusal = fallback.result.as_ref().unwrap_err();
        assert!(matches!(refusal, Error::Unsupported { .. }), "{refusal}");
        assert_eq!(
            refusal.unsupported_field(),
            Some(crate::RequestField::Tools)
        );

        let wrapper = FallbackProvider::new(with_tool_support("a"), with_tool_support("b"));
        let validation = wrapper.validate_chat_request(&with_tools()).unwrap();
        assert!(validation.fallback().unwrap().result.is_ok());
    }

    #[tokio::test]
    async fn a_refusing_primary_gives_the_backups_validation_and_the_failover() {
        let wrapper =
            FallbackProvider::new(without_tool_support("plain"), with_tool_support("tools"));
        let validation = wrapper.validate_chat_request(&with_tools()).unwrap();
        assert_eq!(validation.warnings().len(), 1);
        let failover = &validation.warnings()[0];
        assert_eq!(failover.kind, WarningKind::Fallback);
        assert_eq!(failover.subject, Subject::Response);
        assert_eq!(
            failover.message,
            "failed with `unsupported`; served by `tools`"
        );
        assert!(validation.fallback().is_none());

        // The call agrees, unary and streaming.
        let response = wrapper.chat(&with_tools()).await.unwrap();
        assert_eq!(response.text().as_deref(), Some("tools"));
        assert_eq!(response.warnings, validation.warnings());
        let wrapper =
            FallbackProvider::new(without_tool_support("plain"), with_tool_support("tools"));
        let stream = wrapper.chat_stream(&with_tools()).await.unwrap();
        let streamed = stream.collect_response().await.unwrap();
        assert_eq!(
            streamed.warnings,
            wrapper
                .validate_chat_stream_request(&with_tools())
                .unwrap()
                .warnings()
        );
    }

    #[test]
    fn refusals_on_both_sides_and_errors_that_do_not_fail_over() {
        let wrapper = FallbackProvider::new(without_tool_support("a"), without_tool_support("b"));
        let err = wrapper.validate_chat_request(&with_tools()).unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }), "{err}");
        assert!(
            matches!(err.fallback_from(), Some(Error::Unsupported { .. })),
            "{err:?}"
        );

        // An invalid request does not fail over, so the primary's error is
        // the pair's.
        let wrapper = FallbackProvider::new(
            crate::mock::MockDecisionProvider::empty()
                .with_decision_limits(DecisionLimits::new().max_questions(1)),
            crate::mock::MockDecisionProvider::empty(),
        );
        let two = DecisionRequest::new("d", "x")
            .question("a", crate::decision::Question::yes_no("?"))
            .question("b", crate::decision::Question::yes_no("?"));
        let err = wrapper.validate_decision_request(&two).unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err}");
    }

    /// The primary lacks structured output and the backup has it, and tool
    /// support disagrees too.
    #[tokio::test]
    async fn an_asymmetric_pair_reports_the_failover_it_would_make() {
        let primary = MockProvider::with_text("primary")
            .with_provider_name("primary")
            .with_chat_capability(
                ChatCapability::StructuredOutput,
                CapabilitySupport::Unsupported,
            )
            .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported);
        let backup = MockProvider::with_text("backup")
            .with_provider_name("backup")
            .with_chat_capability(
                ChatCapability::StructuredOutput,
                CapabilitySupport::Supported,
            )
            .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Unsupported);
        let wrapper = FallbackProvider::new(primary.clone(), backup.clone());
        for capability in [ChatCapability::StructuredOutput, ChatCapability::ToolCalls] {
            assert_eq!(
                wrapper.chat_capability("m", capability),
                CapabilitySupport::Unknown
            );
        }

        let structured = request().response_format(crate::ResponseFormat::json_schema(
            serde_json::json!({"type": "object"}),
        ));
        let validation = wrapper.validate_chat_request(&structured).unwrap();
        assert_eq!(validation.warnings().len(), 1);
        assert_eq!(validation.warnings()[0].kind, WarningKind::Fallback);
        assert_eq!(
            validation.warnings()[0].cause,
            Some(
                WarningCause::new("unsupported").with_stage(crate::UnsupportedStage::BeforeSending)
            )
        );
        let response = wrapper.chat(&structured).await.unwrap();
        assert_eq!(response.text().as_deref(), Some("backup"));
        assert_eq!(response.warnings, validation.warnings());
        assert_eq!(primary.call_count(), 0);

        // Tools go to the primary, and the backup could not take them.
        let validation = wrapper.validate_chat_request(&with_tools()).unwrap();
        assert!(validation.warnings().is_empty());
        assert!(validation.fallback().unwrap().result.is_err());
    }

    #[tokio::test]
    async fn a_side_that_would_serve_with_a_warning_passes_validation() {
        use crate::{ReasoningEffort, UnsupportedPolicy};

        let primary = MockProvider::with_error(Error::overloaded("down"));
        let backup = MockProvider::with_text("backup").with_chat_capability(
            ChatCapability::ReasoningConfig,
            CapabilitySupport::Unsupported,
        );
        let wrapper = FallbackProvider::new(primary, backup);
        let request = request().reasoning(ReasoningEffort::High);
        let validation = wrapper.validate_chat_request(&request).unwrap();
        let backup = validation.fallback().unwrap().result.as_ref().unwrap();
        assert_eq!(backup.warnings()[0].kind, WarningKind::Unsupported);
        let response = wrapper.chat(&request).await.unwrap();
        assert_eq!(response.text().as_deref(), Some("backup"));
        assert_eq!(response.warnings[0].kind, WarningKind::Unsupported);
        assert_eq!(response.warnings[1].kind, WarningKind::Fallback);

        // Under a strict policy the backup could not serve.
        let strict = request.unsupported_policy(UnsupportedPolicy::Error);
        let validation = wrapper.validate_chat_request(&strict).unwrap();
        assert!(validation.fallback().unwrap().result.is_err());
    }

    #[tokio::test]
    async fn attempts_record_the_redacted_log_line() {
        let wrapper = FallbackProvider::new(
            MockProvider::with_error(
                Error::overloaded("HTTP 503")
                    .authored()
                    .with_status(503)
                    .with_upstream_message("SENTINEL")
                    .with_body("SENTINEL"),
            )
            .with_provider_name("primary"),
            MockProvider::with_text("fallback").with_provider_name("backup"),
        );
        let response = wrapper.chat(&request()).await.unwrap();
        let attempts = response
            .provider_data
            .get::<FallbackAttempts>()
            .unwrap()
            .unwrap();
        assert_eq!(
            attempts.as_slice()[0].error_message,
            r#"kind=overloaded status=503 body_len=8 message="HTTP 503""#
        );
    }

    #[tokio::test]
    async fn each_side_receives_its_own_model() {
        use crate::wrappers::MapModelProvider;

        let primary = MockProvider::with_error(Error::overloaded("down"));
        let backup = MockProvider::with_text("backup");
        let wrapper = FallbackProvider::new(
            primary.clone(),
            MapModelProvider::new(backup.clone(), |model: &str| match model {
                "primary-model" => "backup-model".into(),
                other => other.into(),
            }),
        );
        let request = ChatRequest::new("primary-model").user("hi");
        assert!(wrapper.validate_chat_request(&request).is_ok());
        let response = wrapper.chat(&request).await.unwrap();
        assert_eq!(response.text().as_deref(), Some("backup"));
        assert_eq!(primary.last_request().unwrap().model, "primary-model");
        assert_eq!(backup.last_request().unwrap().model, "backup-model");
    }

    #[tokio::test]
    async fn a_request_the_fallback_cannot_take_is_reported_apart() {
        use crate::wrappers::MapModelProvider;

        let primary = MockProvider::with_error(Error::overloaded("down"))
            .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported);
        let backup = without_tool_support("backup");
        let wrapper = FallbackProvider::new(
            primary.clone(),
            MapModelProvider::new(backup.clone(), |_: &str| "backup-model".into()),
        );
        assert_eq!(
            wrapper.chat_capability("m", ChatCapability::ToolCalls),
            CapabilitySupport::Unknown
        );
        let validation = wrapper.validate_chat_request(&with_tools()).unwrap();
        let fallback = validation.fallback().unwrap();
        assert_eq!(fallback.provider, "backup");
        assert!(matches!(fallback.result, Err(Error::Unsupported { .. })));
        assert_eq!(primary.call_count() + backup.call_count(), 0);
    }

    #[test]
    fn decision_limits_and_validation_cover_both_sides() {
        use crate::decision::Question;
        use crate::mock::MockDecisionProvider;

        let limited = |limits| {
            MockDecisionProvider::empty()
                .with_provider_name("limited")
                .with_decision_limits(limits)
        };
        let wrapper = FallbackProvider::new(
            limited(DecisionLimits::new().max_questions(3).max_choice_options(4)),
            limited(DecisionLimits::new().max_questions(1).max_score_levels(5)),
        );
        assert_eq!(
            wrapper.decision_limits("d"),
            DecisionLimits::new()
                .max_questions(1)
                .max_choice_options(4)
                .max_score_levels(5)
        );

        let one = DecisionRequest::new("d", "x").question("a", Question::yes_no("?"));
        let validation = wrapper.validate_decision_request(&one).unwrap();
        assert!(validation.fallback().unwrap().result.is_ok());
        let two = one.question("b", Question::yes_no("?"));
        let validation = wrapper.validate_decision_request(&two).unwrap();
        assert!(matches!(
            validation.fallback().unwrap().result,
            Err(Error::InvalidRequest { .. })
        ));
    }

    #[tokio::test]
    async fn failovers_are_reported_as_warnings_on_every_operation() {
        let response = FallbackProvider::new(failing_primary(), MockProvider::with_text("b"))
            .chat(&request())
            .await
            .unwrap();
        assert_eq!(response.warnings.len(), 1);
        assert_eq!(response.warnings[0].kind, WarningKind::Fallback);

        let embedding = FallbackProvider::new(
            crate::mock::MockEmbeddingProvider::with_error(Error::overloaded("busy")),
            crate::mock::MockEmbeddingProvider::with_vectors(vec![vec![1.0]]),
        )
        .embed(&EmbeddingRequest::new("e").input("x"))
        .await
        .unwrap();
        assert_eq!(embedding.warnings[0].kind, WarningKind::Fallback);
    }
}
