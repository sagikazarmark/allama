use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use futures_core::Stream;
use futures_util::StreamExt;

use crate::adapter::embedding::check_response;
use crate::macros::enum_names;
use crate::{
    CapabilitySupport, ChunkedEmbeddingProgress, EmbeddingChunk, EmbeddingInput, EmbeddingLimits,
    EmbeddingRequest, EmbeddingResponse, EmbeddingValidation, Error, ProviderIdentity, Result,
    ValidationCoverage,
};

/// Portable embedding features a provider or model may support.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingCapability {
    /// Several inputs per request.
    BatchInput,
    /// Requesting a specific output dimensionality.
    OutputDimensions,
    /// Image parts ([`EmbeddingPart::Image`](crate::EmbeddingPart::Image)).
    ImageInput,
    /// Several parts in one input, combined into one vector in a
    /// model-defined way. Every model that supports it accepts at least one
    /// text part plus one image part, in either order; anything beyond that
    /// (more images, several texts) is model-specific and fails with
    /// [`Error::InvalidRequest`] when the model does not allow it.
    CompositeInput,
    /// The query/document [`input_type`](crate::EmbeddingRequest::input_type)
    /// hint, sent on the provider's native field (ADR 0003). A symmetric
    /// model, which embeds queries and documents alike, honors it by
    /// sending nothing: `Supported`, with no warning under any policy.
    InputType,
}

enum_names!(EmbeddingCapability, "embedding capability", {
    BatchInput => "batch_input",
    OutputDimensions => "output_dimensions",
    ImageInput => "image_input",
    CompositeInput => "composite_input",
    InputType => "input_type",
});

impl EmbeddingCapability {
    /// Every capability, in declaration order.
    pub const ALL: &[Self] = &[
        Self::BatchInput,
        Self::OutputDimensions,
        Self::ImageInput,
        Self::CompositeInput,
        Self::InputType,
    ];
}

/// Providers that turn inputs into vectors.
///
/// Every provider embeds text; other modalities depend on the model and are
/// reported through [`embedding_capability`](Self::embedding_capability).
/// Adapters reject content a model cannot embed before sending, under every
/// [`UnsupportedPolicy`](crate::UnsupportedPolicy): dropping or joining parts
/// would silently change the vectors.
pub trait EmbeddingProvider: ProviderIdentity {
    /// Embed every input of `request`.
    fn embed(
        &self,
        request: &EmbeddingRequest,
    ) -> impl Future<Output = Result<EmbeddingResponse>> + Send;

    /// Whether the provider supports `capability` for `model`.
    ///
    /// Required: a provider that does not know answers
    /// [`CapabilitySupport::Unknown`]; a wrapper forwards the answer of the
    /// provider it wraps.
    fn embedding_capability(
        &self,
        model: &str,
        capability: EmbeddingCapability,
    ) -> CapabilitySupport;

    /// The request limits of `model`, such as the most inputs one request
    /// takes. Callers can use [`EmbeddingProviderExt::embed_chunked`] to
    /// split larger batches.
    ///
    /// The default reports none. Wrappers forward it.
    fn embedding_limits(&self, _model: &str) -> EmbeddingLimits {
        EmbeddingLimits::default()
    }

    /// Validate a request without sending it: a dry run of
    /// [`embed`](Self::embed).
    ///
    /// `Ok` holds what `embed` would report about the request
    /// ([`EmbeddingValidation`]); `Err` is the error it would return before
    /// sending.
    ///
    /// The default runs [`embedding::prepare_dry_run`](crate::adapter::embedding::prepare_dry_run)
    /// ([`embedding::preflight`](crate::adapter::embedding::preflight)) and reports
    /// [`ValidationCoverage::Capabilities`]: an empty batch or one over
    /// [`max_inputs_per_call`](EmbeddingLimits::max_inputs_per_call) fails with
    /// [`Error::InvalidRequest`]; several inputs without
    /// [`BatchInput`](EmbeddingCapability::BatchInput), `dimensions` without
    /// [`OutputDimensions`](EmbeddingCapability::OutputDimensions), image parts
    /// without [`ImageInput`](EmbeddingCapability::ImageInput), and several
    /// parts in one input without
    /// [`CompositeInput`](EmbeddingCapability::CompositeInput) fail with
    /// [`Error::Unsupported`] when reported
    /// [`Unsupported`](CapabilitySupport::Unsupported). `Unknown` is allowed
    /// through and reported. Content is never dropped, so these fail under
    /// every [`UnsupportedPolicy`](crate::UnsupportedPolicy). An
    /// [`input_type`](EmbeddingRequest::input_type) without
    /// [`InputType`](EmbeddingCapability::InputType) is a hint: it fails only
    /// under [`UnsupportedPolicy::Error`](crate::UnsupportedPolicy::Error),
    /// and is a warning otherwise. Providers may override it with stricter
    /// checks; wrappers forward it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`] or [`Error::Unsupported`] describing
    /// the first problem found.
    fn validate_embedding_request(
        &self,
        request: &EmbeddingRequest,
    ) -> Result<EmbeddingValidation> {
        let prepared = crate::adapter::embedding::prepare_dry_run(self, request)?;
        Ok(prepared.into_validation(ValidationCoverage::Capabilities))
    }
}

macro_rules! forward_embedding_provider {
    ($ty:ty, |$s:ident| $inner:expr) => {
        impl<T> EmbeddingProvider for $ty
        where
            T: EmbeddingProvider + ?Sized,
        {
            async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
                let $s = self;
                let inner: &T = $inner;
                inner.embed(request).await
            }

            fn embedding_capability(
                &self,
                model: &str,
                capability: EmbeddingCapability,
            ) -> CapabilitySupport {
                let $s = self;
                let inner: &T = $inner;
                inner.embedding_capability(model, capability)
            }

            fn embedding_limits(&self, model: &str) -> EmbeddingLimits {
                let $s = self;
                let inner: &T = $inner;
                inner.embedding_limits(model)
            }

            fn validate_embedding_request(
                &self,
                request: &EmbeddingRequest,
            ) -> Result<EmbeddingValidation> {
                let $s = self;
                let inner: &T = $inner;
                inner.validate_embedding_request(request)
            }
        }
    };
}

forward_embedding_provider!(&T, |s| *s);
forward_embedding_provider!(Box<T>, |s| s.as_ref());
forward_embedding_provider!(Arc<T>, |s| s.as_ref());

/// A type-erased embedding provider.
#[derive(Clone)]
pub struct DynEmbeddingProvider(Arc<dyn EmbeddingProviderErased>);

impl DynEmbeddingProvider {
    /// Erase a provider.
    #[must_use]
    pub fn new<T: EmbeddingProvider + 'static>(provider: T) -> Self {
        Self(Arc::new(provider))
    }
}

impl<T: EmbeddingProvider + 'static> From<Arc<T>> for DynEmbeddingProvider {
    fn from(provider: Arc<T>) -> Self {
        Self(provider)
    }
}

impl std::fmt::Debug for DynEmbeddingProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynEmbeddingProvider")
            .field("provider", &self.0.provider_name())
            .finish()
    }
}

impl ProviderIdentity for DynEmbeddingProvider {
    fn provider_name(&self) -> &str {
        self.0.provider_name()
    }

    fn reads_namespace(&self, namespace: &str) -> bool {
        self.0.reads_namespace(namespace)
    }

    fn downcast_any(&self, id: std::any::TypeId) -> Option<&(dyn std::any::Any + Send + Sync)> {
        self.0.downcast_any(id)
    }
}

impl EmbeddingProvider for DynEmbeddingProvider {
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        self.0.embed_erased(request).await
    }

    fn embedding_capability(
        &self,
        model: &str,
        capability: EmbeddingCapability,
    ) -> CapabilitySupport {
        self.0.embedding_capability_erased(model, capability)
    }

    fn embedding_limits(&self, model: &str) -> EmbeddingLimits {
        self.0.embedding_limits_erased(model)
    }

    fn validate_embedding_request(
        &self,
        request: &EmbeddingRequest,
    ) -> Result<EmbeddingValidation> {
        self.0.validate_erased(request)
    }
}

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

trait EmbeddingProviderErased: ProviderIdentity {
    fn embed_erased<'a>(
        &'a self,
        request: &'a EmbeddingRequest,
    ) -> BoxFuture<'a, Result<EmbeddingResponse>>;
    fn embedding_capability_erased(
        &self,
        model: &str,
        capability: EmbeddingCapability,
    ) -> CapabilitySupport;
    fn embedding_limits_erased(&self, model: &str) -> EmbeddingLimits;
    fn validate_erased(&self, request: &EmbeddingRequest) -> Result<EmbeddingValidation>;
}

impl<T: EmbeddingProvider> EmbeddingProviderErased for T {
    fn embed_erased<'a>(
        &'a self,
        request: &'a EmbeddingRequest,
    ) -> BoxFuture<'a, Result<EmbeddingResponse>> {
        Box::pin(EmbeddingProvider::embed(self, request))
    }

    fn embedding_capability_erased(
        &self,
        model: &str,
        capability: EmbeddingCapability,
    ) -> CapabilitySupport {
        EmbeddingProvider::embedding_capability(self, model, capability)
    }

    fn embedding_limits_erased(&self, model: &str) -> EmbeddingLimits {
        EmbeddingProvider::embedding_limits(self, model)
    }

    fn validate_erased(&self, request: &EmbeddingRequest) -> Result<EmbeddingValidation> {
        EmbeddingProvider::validate_embedding_request(self, request)
    }
}

/// Convenience methods for every [`EmbeddingProvider`].
pub trait EmbeddingProviderExt: EmbeddingProvider {
    /// Embed a single input (a text, an image, or several parts combined
    /// into one vector) and return its vector.
    ///
    /// ```
    /// # #[cfg(feature = "mock")] {
    /// use allama::EmbeddingProviderExt;
    /// use allama::mock::MockEmbeddingProvider;
    ///
    /// # async fn example() -> allama::Result<()> {
    /// let provider = MockEmbeddingProvider::with_vectors(vec![vec![0.1, 0.2]]);
    /// let vector = provider.embed_input("m", "quick one-shot").await?;
    /// assert_eq!(vector, [0.1, 0.2]);
    /// # Ok(()) }
    /// # }
    /// ```
    fn embed_input(
        &self,
        model: &str,
        input: impl Into<EmbeddingInput>,
    ) -> impl Future<Output = Result<Vec<f32>>> + Send {
        let request = EmbeddingRequest::new(model).input(input.into());
        async move {
            let mut embeddings = self.embed(&request).await?.embeddings;
            if embeddings.len() != 1 {
                return Err(Error::unexpected_response(format!(
                    "the response holds {} embeddings for one input in embed_input()",
                    embeddings.len()
                ))
                .authored()
                .with_provider(self.provider_name()));
            }
            Ok(embeddings.remove(0))
        }
    }

    /// Embed a batch of any size in chunks, one after another, yielding each
    /// chunk as it completes: a pipeline can store each chunk's vectors as
    /// they arrive, and resume after a failure without paying for the
    /// chunks again.
    ///
    /// A chunk holds at most `max_chunk_inputs` inputs, and at most
    /// [`max_inputs_per_call`](EmbeddingLimits::max_inputs_per_call) when
    /// the provider reports a limit. Only the input count is split:
    /// per-request byte, token, or media limits can still be exceeded. Do
    /// not split requests whose inputs depend on each other (for example
    /// provider options that embed the batch as one document).
    ///
    /// Each chunk must come back with one vector per input and pass
    /// [`check_response`](crate::adapter::embedding::check_response) with vectors of
    /// the size of the first chunk's (a fallback that switched models
    /// mid-batch returns another size); a chunk that does not fails with
    /// [`Error::UnexpectedResponse`]. A warning about an input
    /// ([`Subject::EmbeddingInput`](crate::Subject::EmbeddingInput)) names its index in `request.inputs`.
    ///
    /// The stream ends after the first `Err`. The error carries a
    /// [`ChunkedEmbeddingProgress`](crate::ChunkedEmbeddingProgress) whose
    /// `completed_inputs` is the offset to resume from: embed
    /// `request.inputs[offset..]` in a new request. Its usage sums every
    /// chunk the provider answered, the chunks already yielded included.
    ///
    /// ```
    /// # #[cfg(feature = "mock")] {
    /// use allama::mock::MockEmbeddingProvider;
    /// use allama::{
    ///     ChunkedEmbeddingProgress, EmbeddingProviderExt, EmbeddingRequest, EmbeddingResponse, Error,
    /// };
    /// use futures_util::StreamExt;
    ///
    /// # async fn example() {
    /// let provider = MockEmbeddingProvider::new([
    ///     Ok(EmbeddingResponse::new(vec![vec![0.1], vec![0.2]])),
    ///     Err(Error::rate_limited("slow down")),
    ///     Ok(EmbeddingResponse::new(vec![vec![0.3]])),
    /// ]);
    /// let request = EmbeddingRequest::new("m").inputs(["a", "b", "c"]);
    /// let mut stored = Vec::new();
    /// let mut resume_at = None;
    /// let mut chunks = std::pin::pin!(provider.embed_chunks(&request, 2));
    /// while let Some(chunk) = chunks.next().await {
    ///     match chunk {
    ///         // Store each chunk's vectors as they arrive.
    ///         Ok(chunk) => stored.extend(chunk.response.embeddings),
    ///         Err(err) => {
    ///             let progress = ChunkedEmbeddingProgress::from_error(&err).unwrap();
    ///             resume_at = Some(progress.completed_inputs);
    ///         }
    ///     }
    /// }
    /// assert_eq!(resume_at, Some(2));
    ///
    /// // Later: embed what is left.
    /// let mut rest = request.clone();
    /// rest.inputs.drain(..2);
    /// let rest = provider.embed_chunked(&rest, 2).await.unwrap();
    /// stored.extend(rest.embeddings);
    /// assert_eq!(stored, vec![vec![0.1], vec![0.2], vec![0.3]]);
    /// # }
    /// # }
    /// ```
    fn embed_chunks<'a>(
        &'a self,
        request: &'a EmbeddingRequest,
        max_chunk_inputs: usize,
    ) -> impl Stream<Item = Result<EmbeddingChunk>> + Send + 'a
    where
        Self: Sync,
    {
        super::chunks::chunks(self, request, max_chunk_inputs)
    }

    /// Embed a batch of any size, splitting it into requests of at most
    /// [`max_inputs_per_call`](EmbeddingLimits::max_inputs_per_call)
    /// inputs (or `fallback_chunk` when the provider reports no limit), and
    /// return one response: [`embed_chunks`](Self::embed_chunks), merged.
    ///
    /// Every chunk is checked as `embed_chunks` checks it, and the merged
    /// response with [`check_response`](crate::adapter::embedding::check_response).
    /// The chunks' warnings are merged
    /// ([`EmbeddingResponse::extend`]): a warning about an input names its
    /// index in `request.inputs`, and a warning several chunks repeat (an
    /// unsupported input type, say) is reported once.
    ///
    /// The call is all-or-nothing: when a chunk fails, it returns that error
    /// and drops the vectors of the chunks before it. The error carries a
    /// [`ChunkedEmbeddingProgress`](crate::ChunkedEmbeddingProgress) with
    /// the number of inputs completed and the usage of every chunk the
    /// provider answered, so the tokens already billed are still accounted
    /// for. To keep the vectors of the chunks that succeeded, use
    /// `embed_chunks`.
    fn embed_chunked(
        &self,
        request: &EmbeddingRequest,
        fallback_chunk: usize,
    ) -> impl Future<Output = Result<EmbeddingResponse>> + Send
    where
        Self: Sync,
    {
        async move {
            let chunk = self
                .embedding_limits(&request.model)
                .max_inputs_per_call
                .unwrap_or(fallback_chunk);
            let mut chunks = std::pin::pin!(self.embed_chunks(request, chunk));
            let mut combined: Option<EmbeddingResponse> = None;
            while let Some(chunk) = chunks.next().await {
                let response = chunk?.response;
                match &mut combined {
                    Some(combined) => combined.extend(response),
                    None => combined = Some(response),
                }
            }
            let combined = combined.unwrap_or_default();
            check_response(request, &combined).map_err(|err| {
                let progress = ChunkedEmbeddingProgress {
                    completed_inputs: combined.embeddings.len(),
                    usage: combined.usage.clone(),
                };
                progress.attach_to(err.with_provider(self.provider_name().to_owned()))
            })?;
            Ok(combined)
        }
    }
}

impl<T: EmbeddingProvider + ?Sized> EmbeddingProviderExt for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockEmbeddingProvider;
    use crate::{EmbeddingPart, RequestField, Subject, Usage, Warning, WarningKind};

    #[test]
    fn all_capabilities_listed() {
        // Exhaustive on purpose: a new variant fails to compile here, as a
        // reminder to add it to `EmbeddingCapability::ALL` and bump the count.
        fn _exhaustive(capability: EmbeddingCapability) {
            match capability {
                EmbeddingCapability::BatchInput
                | EmbeddingCapability::OutputDimensions
                | EmbeddingCapability::ImageInput
                | EmbeddingCapability::CompositeInput
                | EmbeddingCapability::InputType => {}
            }
        }
        let unique: std::collections::HashSet<_> = EmbeddingCapability::ALL.iter().collect();
        assert_eq!(unique.len(), EmbeddingCapability::ALL.len());
        assert_eq!(EmbeddingCapability::ALL.len(), 5);
    }

    #[tokio::test]
    async fn dyn_provider_and_embed_input() {
        let provider =
            DynEmbeddingProvider::new(MockEmbeddingProvider::with_vectors(vec![vec![0.1, 0.2]]));
        assert_eq!(
            provider.embed_input("m", "x").await.unwrap(),
            vec![0.1, 0.2]
        );
        assert_eq!(provider.provider_name(), "mock");
    }

    #[tokio::test]
    async fn embed_input_requires_exactly_one_vector() {
        for vectors in [vec![], vec![vec![1.0], vec![2.0]]] {
            let provider =
                MockEmbeddingProvider::new([Ok(EmbeddingResponse::new(vectors.clone()))]);
            let err = provider.embed_input("m", "x").await.unwrap_err();
            assert!(
                matches!(err, Error::UnexpectedResponse { .. }),
                "{} vectors: {err}",
                vectors.len()
            );
        }
    }

    #[tokio::test]
    async fn embed_chunked_preserves_inputs_and_their_provider_data() {
        use crate::{EmbeddingInput, ProviderData};
        let titled = |text: &str| {
            EmbeddingInput::text(text).with_provider_data(ProviderData::new().with_raw(
                "demo",
                "title",
                serde_json::json!(text),
            ))
        };
        let provider = MockEmbeddingProvider::new([
            Ok(EmbeddingResponse::new(vec![vec![1.0], vec![2.0]])),
            Ok(EmbeddingResponse::new(vec![vec![3.0]])),
        ]);
        let request = EmbeddingRequest::new("m").inputs([titled("a"), titled("b"), titled("c")]);
        let response = provider.embed_chunked(&request, 2).await.unwrap();
        assert_eq!(response.embeddings, vec![vec![1.0], vec![2.0], vec![3.0]]);
        let sent = provider.requests();
        assert_eq!(sent[0].inputs, vec![titled("a"), titled("b")]);
        assert_eq!(sent[1].inputs, vec![titled("c")]);
    }

    fn answer(vectors: usize, input_tokens: u64) -> Result<EmbeddingResponse> {
        Ok(EmbeddingResponse::new(vec![vec![0.0]; vectors])
            .usage(Usage::new().input_tokens(input_tokens)))
    }

    #[tokio::test]
    async fn embed_chunked_rejects_a_chunk_with_the_wrong_vector_count() {
        let request = EmbeddingRequest::new("m").inputs(["a", "b", "c", "d", "e"]);
        for (answers, completed, tokens) in [
            // The second chunk is short, the first long.
            (vec![answer(2, 3), answer(1, 4)], 2, 7),
            (vec![answer(3, 3)], 0, 3),
        ] {
            let provider = MockEmbeddingProvider::new(answers).with_provider_name("demo");
            let err = provider.embed_chunked(&request, 2).await.unwrap_err();
            assert!(matches!(err, Error::UnexpectedResponse { .. }), "{err}");
            assert_eq!(err.context().unwrap().provider.as_deref(), Some("demo"));
            assert_eq!(err.message_origin(), crate::MessageOrigin::Authored);
            let progress = ChunkedEmbeddingProgress::from_error(&err).unwrap();
            assert_eq!(progress.completed_inputs, completed);
            assert_eq!(progress.usage.unwrap().input_tokens, Some(tokens));
        }

        // A batch within one chunk is checked too.
        let provider = MockEmbeddingProvider::new([answer(1, 2)]);
        let err = provider
            .embed_chunked(&EmbeddingRequest::new("m").inputs(["a", "b"]), 4)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::UnexpectedResponse { .. }), "{err}");
        assert!(
            err.message().contains("holds 1 embeddings for 2 inputs"),
            "{err}"
        );
    }

    /// A chunk whose vectors differ in size from the first chunk's (a
    /// fallback that switched models mid-batch) fails, as does one that
    /// ignores the requested dimensions.
    #[tokio::test]
    async fn embed_chunked_checks_each_chunk_against_the_first() {
        let request = EmbeddingRequest::new("m").inputs(["a", "b", "c", "d"]);
        let provider = MockEmbeddingProvider::new([
            Ok(EmbeddingResponse::new(vec![vec![0.1, 0.2]; 2])),
            Ok(EmbeddingResponse::new(vec![vec![0.1, 0.2, 0.3]; 2])),
        ])
        .with_provider_name("demo");
        let err = provider.embed_chunked(&request, 2).await.unwrap_err();
        assert!(matches!(err, Error::UnexpectedResponse { .. }), "{err}");
        assert_eq!(
            err.message(),
            "`m` returned vectors of different sizes: 2 dimensions before input 2, 3 for input 2"
        );
        assert_eq!(err.context().unwrap().provider.as_deref(), Some("demo"));
        let progress = ChunkedEmbeddingProgress::from_error(&err).unwrap();
        assert_eq!(progress.completed_inputs, 2);

        let provider =
            MockEmbeddingProvider::new([Ok(EmbeddingResponse::new(vec![vec![0.1, 0.2]; 2]))]);
        let err = provider
            .embed_chunked(&request.clone().dimensions(3), 2)
            .await
            .unwrap_err();
        assert_eq!(
            err.message(),
            "requested 3 dimensions, but `m` returned 2 (input 0)"
        );
        assert_eq!(
            ChunkedEmbeddingProgress::from_error(&err)
                .unwrap()
                .completed_inputs,
            0
        );
    }

    #[tokio::test]
    async fn a_failed_chunk_keeps_its_error_and_reports_completed_usage() {
        let request = EmbeddingRequest::new("m").inputs(["a", "b", "c", "d", "e"]);
        let provider = MockEmbeddingProvider::new([
            answer(2, 3),
            answer(2, 4),
            Err(Error::rate_limited("slow down")
                .with_retry_after(std::time::Duration::from_secs(2))),
        ]);
        let err = provider.embed_chunked(&request, 2).await.unwrap_err();
        assert!(matches!(err, Error::RateLimited { .. }), "{err}");
        assert_eq!(err.retry_after(), Some(std::time::Duration::from_secs(2)));
        let progress = ChunkedEmbeddingProgress::from_error(&err).unwrap();
        assert_eq!(progress.completed_inputs, 4);
        assert_eq!(progress.usage.unwrap().input_tokens, Some(7));

        // A first chunk that fails reports that nothing completed.
        let provider = MockEmbeddingProvider::with_error(Error::overloaded("busy"));
        let err = provider.embed_chunked(&request, 2).await.unwrap_err();
        assert_eq!(
            ChunkedEmbeddingProgress::from_error(&err),
            Some(ChunkedEmbeddingProgress::default())
        );
    }

    #[tokio::test]
    async fn embed_input_returns_the_single_vector() {
        let provider = MockEmbeddingProvider::with_vectors(vec![vec![0.5]]);
        let vector = provider
            .embed_input("m", EmbeddingInput::image_url("https://example.com/a.png"))
            .await
            .unwrap();
        assert_eq!(vector, vec![0.5]);
    }

    #[test]
    fn default_validation_rejects_only_unsupported_capabilities() {
        let image =
            EmbeddingRequest::new("m").input(EmbeddingInput::image_url("https://e.x/a.png"));
        let composite = EmbeddingRequest::new("m").input(vec![
            EmbeddingPart::text("a"),
            EmbeddingPart::image_url("https://e.x/a.png"),
        ]);
        let unknown = MockEmbeddingProvider::empty();
        let validation = unknown.validate_embedding_request(&image).unwrap();
        assert_eq!(
            validation.relied_on_unknown(),
            [(
                EmbeddingCapability::ImageInput,
                Subject::EmbeddingInput {
                    input: 0,
                    part: Some(0)
                }
            )]
        );
        let validation = unknown.validate_embedding_request(&composite).unwrap();
        let relied: Vec<_> = validation
            .relied_on_unknown()
            .iter()
            .map(|(capability, _)| *capability)
            .collect();
        assert_eq!(
            relied,
            [
                EmbeddingCapability::ImageInput,
                EmbeddingCapability::CompositeInput
            ]
        );

        let text_only = MockEmbeddingProvider::empty()
            .with_provider_name("text-only")
            .with_embedding_capability(
                EmbeddingCapability::ImageInput,
                CapabilitySupport::Unsupported,
            )
            .with_embedding_capability(
                EmbeddingCapability::CompositeInput,
                CapabilitySupport::Unsupported,
            );
        let err = text_only.validate_embedding_request(&image).unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }));
        assert_eq!(
            err.context().unwrap().provider.as_deref(),
            Some("text-only")
        );
        assert!(err.message().contains("image input"), "{err}");
        let err = DynEmbeddingProvider::new(text_only)
            .validate_embedding_request(
                &EmbeddingRequest::new("m")
                    .input(vec![EmbeddingPart::text("a"), EmbeddingPart::text("b")]),
            )
            .unwrap_err();
        assert!(err.message().contains("combining 2 parts"), "{err}");
    }

    #[tokio::test]
    async fn embed_chunked_splits_and_combines() {
        let provider = MockEmbeddingProvider::new([
            Ok(EmbeddingResponse::new(vec![vec![1.0], vec![2.0]])
                .usage(Usage::new().input_tokens(2))),
            Ok(EmbeddingResponse::new(vec![vec![3.0]]).usage(Usage::new().input_tokens(1))),
        ]);
        let request = EmbeddingRequest::new("m").inputs(["a", "b", "c"]);
        let response = provider.embed_chunked(&request, 2).await.unwrap();
        assert_eq!(response.embeddings, vec![vec![1.0], vec![2.0], vec![3.0]]);
        assert_eq!(response.usage.unwrap().input_tokens, Some(3));
        assert_eq!(provider.requests().len(), 2);
    }

    #[tokio::test]
    async fn embed_chunked_merges_chunk_warnings() {
        let hint =
            || Warning::unsupported(RequestField::InputType, "no input type").with_provider("mock");
        let image = |input| {
            Warning::new(
                WarningKind::Approximated,
                Subject::EmbeddingInput {
                    input,
                    part: Some(0),
                },
                "image resized",
            )
        };
        let provider = MockEmbeddingProvider::new([
            Ok(EmbeddingResponse::new(vec![vec![1.0], vec![2.0]])
                .warning(hint())
                .warning(image(1))),
            Ok(EmbeddingResponse::new(vec![vec![3.0], vec![4.0]])
                .warning(hint())
                .warning(image(0))
                .warning(image(1))),
            Ok(EmbeddingResponse::new(vec![vec![5.0]]).warning(hint())),
        ]);
        let request = EmbeddingRequest::new("m").inputs(["a", "b", "c", "d", "e"]);
        let response = provider.embed_chunked(&request, 2).await.unwrap();
        assert_eq!(response.embeddings.len(), 5);
        // The repeated hint is reported once; each input warning names the
        // input's index in the whole request, so none is merged away.
        assert_eq!(
            response.warnings,
            vec![hint(), image(1), image(2), image(3)]
        );
    }

    #[tokio::test]
    async fn embed_chunked_keeps_a_single_chunk_response_as_is() {
        let hint = Warning::unsupported(RequestField::InputType, "no input type");
        let provider =
            MockEmbeddingProvider::new([Ok(
                EmbeddingResponse::new(vec![vec![1.0], vec![2.0]]).warning(hint.clone())
            )]);
        let request = EmbeddingRequest::new("m").inputs(["a", "b"]);
        let response = provider.embed_chunked(&request, 2).await.unwrap();
        assert_eq!(response.warnings, vec![hint]);
    }
}
