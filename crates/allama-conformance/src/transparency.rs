//! The transparent wrapper contract (C4). Its description, and the table
//! of which check to call, are the crate docs' "Transparent wrappers"
//! section (`lib.rs`), since this module is private.

use std::any::{Any, TypeId};
use std::borrow::Cow;
use std::fmt::Debug;

use allama::{
    Capability, CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse,
    ChatStream, ChatValidation, DecisionCapability, DecisionLimits, DecisionProvider,
    DecisionRequest, DecisionResponse, DecisionValidation, EmbeddingCapability, EmbeddingLimits,
    EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, EmbeddingValidation, Error,
    ProviderIdentity, ProviderIdentityExt, Result, Subject, Validation, ValidationCoverage,
    Warning, WarningKind, decision::Question, expose,
};

use data_path::Scenario;

mod data_path;

/// Assert that a wrapper over every provider trait is transparent, data
/// path included: [`assert_chat_wrapper`], [`assert_embedding_wrapper`] and
/// [`assert_decision_wrapper`] on the one wrapper.
///
/// `wrap` builds the wrapper around a [`TransparencyProbe`].
///
/// ```
/// use allama::wrappers::TracingProvider;
/// use allama_conformance::assert_wrapper;
///
/// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
/// assert_wrapper(TracingProvider::new).await;
/// # });
/// ```
///
/// # Panics
///
/// Panics, naming the method, when the wrapper answers differently from the
/// probe, and listing every call whose data did not come through unchanged.
pub async fn assert_wrapper<W>(wrap: impl FnOnce(TransparencyProbe) -> W)
where
    W: ChatProvider + EmbeddingProvider + DecisionProvider,
{
    let wrapper = wrap(TransparencyProbe::new());
    check_identity(&wrapper);
    check_chat(&wrapper, &unmapped);
    check_embedding(&wrapper, &unmapped);
    check_decision(&wrapper, &unmapped);
    let mut problems = data_path::chat_problems(&wrapper).await;
    problems.extend(data_path::embedding_problems(&wrapper).await);
    problems.extend(data_path::decision_problems(&wrapper).await);
    data_path::report(std::any::type_name::<W>(), &problems);
}

/// Assert that a chat wrapper is transparent, data path included: it
/// forwards identity, capability answers and both validators
/// ([`assert_transparent_chat_wrapper`]), and passes the provider's
/// responses, streams and errors through unchanged
/// ([`assert_transparent_chat_data_path`]).
///
/// ```
/// use allama::DynChatProvider;
/// use allama_conformance::assert_chat_wrapper;
///
/// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
/// assert_chat_wrapper(DynChatProvider::new).await;
/// # });
/// ```
///
/// # Panics
///
/// Panics, naming the method, when the wrapper answers differently from the
/// probe, and listing every call whose data did not come through unchanged.
pub async fn assert_chat_wrapper<W: ChatProvider>(wrap: impl FnOnce(TransparencyProbe) -> W) {
    let wrapper = wrap(TransparencyProbe::new());
    check_identity(&wrapper);
    check_chat(&wrapper, &unmapped);
    let problems = data_path::chat_problems(&wrapper).await;
    data_path::report(std::any::type_name::<W>(), &problems);
}

/// [`assert_chat_wrapper`] for an embedding wrapper: identity, capability
/// answers, the batch limit and validation
/// ([`assert_transparent_embedding_wrapper`]), and the data path
/// ([`assert_transparent_embedding_data_path`]).
///
/// # Panics
///
/// As [`assert_chat_wrapper`].
pub async fn assert_embedding_wrapper<W: EmbeddingProvider>(
    wrap: impl FnOnce(TransparencyProbe) -> W,
) {
    let wrapper = wrap(TransparencyProbe::new());
    check_identity(&wrapper);
    check_embedding(&wrapper, &unmapped);
    let problems = data_path::embedding_problems(&wrapper).await;
    data_path::report(std::any::type_name::<W>(), &problems);
}

/// [`assert_chat_wrapper`] for a decision wrapper: identity, capability
/// answers, limits and validation ([`assert_transparent_decision_wrapper`]),
/// and the data path ([`assert_transparent_decision_data_path`]).
///
/// # Panics
///
/// As [`assert_chat_wrapper`].
pub async fn assert_decision_wrapper<W: DecisionProvider>(
    wrap: impl FnOnce(TransparencyProbe) -> W,
) {
    let wrapper = wrap(TransparencyProbe::new());
    check_identity(&wrapper);
    check_decision(&wrapper, &unmapped);
    let problems = data_path::decision_problems(&wrapper).await;
    data_path::report(std::any::type_name::<W>(), &problems);
}

/// Assert that a chat wrapper passes the provider's data through
/// unchanged.
///
/// `wrap` builds the wrapper around a [`TransparencyProbe`], whose calls
/// answer by the scenario the request names. Through the wrapper:
///
/// - **`chat`** returns the probe's response: content (a text block with
///   provider data, a refusal block), id, model, request id, provider,
///   finish reason, usage and provider data unchanged, and the probe's
///   warnings (one with a code, one with a cause, one naming content) in
///   order. The wrapper may add warnings of its own anywhere.
/// - **`chat_stream`** yields the probe's events in order: the start,
///   warnings, text and refusal deltas, block and response provider data,
///   and the metadata. The wrapper may add `Warning` events.
/// - **Errors**: for an error of every kind, each with a status, request
///   id, retry hint or directive, provider code and upstream message where
///   it has one, and the refused subject and stage of a refusal (one the
///   provider returned, one made before sending), `chat` fails with it,
///   `chat_stream` fails with it before its stream, and fails with it after
///   output started. Everything a caller decides on must be unchanged; the
///   message may carry a prefix ([`Error::with_message_prefix`]). A
///   wrapper that fails over returns its last side's error as itself, and
///   that error is what is compared; the errors it
///   [fell back from](Error::fallback_from) are not.
///
/// Validation is not a call: [`assert_transparent_chat_wrapper`] checks that
/// both validators are forwarded, coverage and `Unknown` answers included,
/// and [`assert_chat_wrapper`] runs both. A wrapper that maps model ids runs
/// this check with [`assert_transparent_chat_model_mapping`].
///
/// ```
/// use allama::wrappers::{FallbackProvider, RetryPolicy, RetryingProvider};
/// use allama_conformance::assert_transparent_chat_data_path;
///
/// # tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap().block_on(async {
/// assert_transparent_chat_data_path(|probe| FallbackProvider::new(probe, probe)).await;
/// // Without backoff, so the retries of retryable errors do not wait.
/// let policy = RetryPolicy::new().base_delay(std::time::Duration::ZERO).jitter(false);
/// assert_transparent_chat_data_path(|probe| RetryingProvider::new(probe, policy))
///     .await;
/// # });
/// ```
///
/// # Panics
///
/// Listing every call whose data did not come through unchanged.
pub async fn assert_transparent_chat_data_path<W: ChatProvider>(
    wrap: impl FnOnce(TransparencyProbe) -> W,
) {
    let wrapper = wrap(TransparencyProbe::new());
    let problems = data_path::chat_problems(&wrapper).await;
    data_path::report(std::any::type_name::<W>(), &problems);
}

/// [`assert_transparent_chat_data_path`] for an embedding wrapper: `embed`
/// returns the probe's vectors, model, usage, provider data and warnings,
/// and fails with each of its errors unchanged.
///
/// # Provisional
///
/// [`assert_embedding_wrapper`] runs this part with the rest of the
/// contract, and is the check to call. On its own, this function may
/// change or go in any minor release (ADR 0017).
///
/// # Panics
///
/// As [`assert_transparent_chat_data_path`].
pub async fn assert_transparent_embedding_data_path<W: EmbeddingProvider>(
    wrap: impl FnOnce(TransparencyProbe) -> W,
) {
    let wrapper = wrap(TransparencyProbe::new());
    let problems = data_path::embedding_problems(&wrapper).await;
    data_path::report(std::any::type_name::<W>(), &problems);
}

/// [`assert_transparent_chat_data_path`] for a decision wrapper: `decide`
/// returns the probe's answers, ids, usage, provider data and warnings, and
/// fails with each of its errors unchanged.
///
/// # Provisional
///
/// [`assert_decision_wrapper`] runs this part with the rest of the
/// contract, and is the check to call. On its own, this function may
/// change or go in any minor release (ADR 0017).
///
/// # Panics
///
/// As [`assert_transparent_chat_data_path`].
pub async fn assert_transparent_decision_data_path<W: DecisionProvider>(
    wrap: impl FnOnce(TransparencyProbe) -> W,
) {
    let wrapper = wrap(TransparencyProbe::new());
    let problems = data_path::decision_problems(&wrapper).await;
    data_path::report(std::any::type_name::<W>(), &problems);
}

/// Assert that a wrapper over every provider trait is transparent: it
/// forwards identity, capabilities, limits, and validation to the provider
/// it wraps.
///
/// `wrap` builds the wrapper around a [`TransparencyProbe`]. This runs
/// [`assert_transparent_chat_wrapper`],
/// [`assert_transparent_embedding_wrapper`], and
/// [`assert_transparent_decision_wrapper`] on the one wrapper.
///
/// ```
/// use allama::wrappers::{RetryPolicy, RetryingProvider};
/// use allama_conformance::assert_transparent_wrapper;
///
/// assert_transparent_wrapper(|probe| RetryingProvider::new(probe, RetryPolicy::default()));
/// ```
///
/// # Panics
///
/// Panics, naming the method, when the wrapper answers differently from the
/// probe.
pub fn assert_transparent_wrapper<W>(wrap: impl FnOnce(TransparencyProbe) -> W)
where
    W: ChatProvider + EmbeddingProvider + DecisionProvider,
{
    let wrapper = wrap(TransparencyProbe::new());
    check_identity(&wrapper);
    check_chat(&wrapper, &unmapped);
    check_embedding(&wrapper, &unmapped);
    check_decision(&wrapper, &unmapped);
}

/// Assert that a chat wrapper is transparent: it forwards
/// [`provider_name`](ProviderIdentity::provider_name),
/// [`reads_namespace`](ProviderIdentity::reads_namespace),
/// [`downcast_any`](ProviderIdentity::downcast_any),
/// [`chat_capability`](ChatProvider::chat_capability),
/// [`validate_chat_request`](ChatProvider::validate_chat_request), and
/// [`validate_chat_stream_request`](ChatProvider::validate_chat_stream_request).
///
/// ```
/// use allama::DynChatProvider;
/// use allama_conformance::assert_transparent_chat_wrapper;
///
/// assert_transparent_chat_wrapper(DynChatProvider::new);
/// ```
///
/// # Panics
///
/// Panics, naming the method, when the wrapper answers differently from the
/// probe.
pub fn assert_transparent_chat_wrapper<W: ChatProvider>(wrap: impl FnOnce(TransparencyProbe) -> W) {
    let wrapper = wrap(TransparencyProbe::new());
    check_identity(&wrapper);
    check_chat(&wrapper, &unmapped);
}

/// Assert that an embedding wrapper is transparent: it forwards
/// [`provider_name`](ProviderIdentity::provider_name),
/// [`reads_namespace`](ProviderIdentity::reads_namespace),
/// [`downcast_any`](ProviderIdentity::downcast_any),
/// [`embedding_capability`](EmbeddingProvider::embedding_capability),
/// [`embedding_limits`](EmbeddingProvider::embedding_limits), and
/// [`validate_embedding_request`](EmbeddingProvider::validate_embedding_request).
///
/// # Provisional
///
/// [`assert_embedding_wrapper`] runs this part with the rest of the
/// contract, and is the check to call. On its own, this function may
/// change or go in any minor release (ADR 0017).
///
/// # Panics
///
/// Panics, naming the method, when the wrapper answers differently from the
/// probe.
pub fn assert_transparent_embedding_wrapper<W: EmbeddingProvider>(
    wrap: impl FnOnce(TransparencyProbe) -> W,
) {
    let wrapper = wrap(TransparencyProbe::new());
    check_identity(&wrapper);
    check_embedding(&wrapper, &unmapped);
}

/// Assert that a decision wrapper is transparent: it forwards
/// [`provider_name`](ProviderIdentity::provider_name),
/// [`reads_namespace`](ProviderIdentity::reads_namespace),
/// [`downcast_any`](ProviderIdentity::downcast_any),
/// [`decision_capability`](DecisionProvider::decision_capability),
/// [`decision_limits`](DecisionProvider::decision_limits), and
/// [`validate_decision_request`](DecisionProvider::validate_decision_request).
///
/// # Provisional
///
/// [`assert_decision_wrapper`] runs this part with the rest of the
/// contract, and is the check to call. On its own, this function may
/// change or go in any minor release (ADR 0017).
///
/// # Panics
///
/// Panics, naming the method, when the wrapper answers differently from the
/// probe.
pub fn assert_transparent_decision_wrapper<W: DecisionProvider>(
    wrap: impl FnOnce(TransparencyProbe) -> W,
) {
    let wrapper = wrap(TransparencyProbe::new());
    check_identity(&wrapper);
    check_decision(&wrapper, &unmapped);
}

/// Assert that a wrapper forwards the [`ProviderIdentity`] surface:
/// [`provider_name`](ProviderIdentity::provider_name),
/// [`reads_namespace`](ProviderIdentity::reads_namespace) and
/// [`downcast_any`](ProviderIdentity::downcast_any).
///
/// The trait-specific checks include this one; use it directly for a type
/// that implements none of the provider traits.
///
/// # Panics
///
/// Panics when the wrapper reports another name, reads other namespaces,
/// or hides the probe.
pub fn assert_transparent_identity<W: ProviderIdentity>(wrap: impl FnOnce(TransparencyProbe) -> W) {
    check_identity(&wrap(TransparencyProbe::new()));
}

/// Assert that a wrapper over every provider trait that maps model ids is
/// transparent once the mapping is applied.
///
/// For each model the checks ask about ([`TransparencyProbe::MODELS`] and
/// [`TransparencyProbe::REJECTED_MODEL`]), the wrapper must give the probe's
/// answers for `map(model)`: capabilities, limits, and validation (it
/// rejects exactly the models that map to the rejected one). Name and
/// `downcast_any` must be forwarded as usual. Use a mapping that changes
/// models whose answers differ, such as one that swaps the probe's models
/// and maps one to [`TransparencyProbe::REJECTED_MODEL`]:
///
/// ```
/// use std::borrow::Cow;
///
/// use allama::wrappers::MapModelProvider;
/// use allama_conformance::{TransparencyProbe, assert_transparent_model_mapping};
///
/// fn map(model: &str) -> Cow<'_, str> {
///     let [a, b] = TransparencyProbe::MODELS;
///     match model {
///         m if m == a => b.into(),
///         m if m == b => TransparencyProbe::REJECTED_MODEL.into(),
///         _ => a.into(),
///     }
/// }
///
/// assert_transparent_model_mapping(|probe| MapModelProvider::new(probe, map), map);
/// ```
///
/// # Panics
///
/// Panics, naming the method, when the wrapper answers differently from the
/// probe for the mapped model, or when `map` leaves every model unchanged
/// (the check could not tell a mapping wrapper from a plain one).
pub fn assert_transparent_model_mapping<W>(
    wrap: impl FnOnce(TransparencyProbe) -> W,
    map: impl Fn(&str) -> Cow<'_, str>,
) where
    W: ChatProvider + EmbeddingProvider + DecisionProvider,
{
    check_mapping_changes_a_model(&map);
    let wrapper = wrap(TransparencyProbe::new());
    check_identity(&wrapper);
    check_chat(&wrapper, &map);
    check_embedding(&wrapper, &map);
    check_decision(&wrapper, &map);
}

/// Assert that a chat wrapper that maps model ids is transparent once the
/// mapping is applied.
///
/// The chat-only form of [`assert_transparent_model_mapping`]: it checks
/// identity, [`chat_capability`](ChatProvider::chat_capability), and
/// [`validate_chat_request`](ChatProvider::validate_chat_request) against
/// the probe's answers for `map(model)`.
///
/// # Panics
///
/// Panics, naming the method, when the wrapper answers differently from the
/// probe for the mapped model, or when `map` leaves every model unchanged.
pub fn assert_transparent_chat_model_mapping<W: ChatProvider>(
    wrap: impl FnOnce(TransparencyProbe) -> W,
    map: impl Fn(&str) -> Cow<'_, str>,
) {
    check_mapping_changes_a_model(&map);
    let wrapper = wrap(TransparencyProbe::new());
    check_identity(&wrapper);
    check_chat(&wrapper, &map);
}

/// The provider the transparency checks wrap.
///
/// It implements every provider trait so it fits any wrapper, and answers
/// every question distinctively:
///
/// - its name is [`PROVIDER_NAME`](Self::PROVIDER_NAME), and it reads
///   provider data under [`NAMESPACE`](Self::NAMESPACE) only, which differs
///   from the name, so a wrapper that answers by its name is caught;
/// - it exposes itself through
///   [`downcast_any`](ProviderIdentity::downcast_any);
/// - capability answers differ per capability and per model, mixing
///   `Supported`, `Unsupported`, and `Unknown`;
/// - limits depend on the model;
/// - validation rejects [`REJECTED_MODEL`](Self::REJECTED_MODEL) and
///   accepts every other model, with [`Encoding`](ValidationCoverage::Encoding)
///   coverage and a warning naming the validation method, so a wrapper that
///   answers with the default validation, or with another method's, is
///   caught.
///
/// Its calls (`chat`, `chat_stream`, `embed`, `decide`) answer by the
/// scenario their request names, for the data path checks
/// ([`assert_transparent_chat_data_path`]): a response full of data
/// (warnings, provider data, usage, ids), a stream of every kind of event,
/// or an error of each kind. Any other request gets the response.
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct TransparencyProbe;

impl TransparencyProbe {
    /// A probe.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// The probe's provider name.
    pub const PROVIDER_NAME: &str = "transparency-probe";

    /// The only namespace the probe reads.
    pub const NAMESPACE: &str = "transparency-probe-state";

    /// The model the probe's validation rejects.
    pub const REJECTED_MODEL: &str = "transparency-probe-rejected";

    /// The message of the probe's validation error.
    pub const REJECTION: &str = "the transparency probe rejects this model";

    /// The models the checks ask about. Their answers differ, so a wrapper
    /// that rewrites or drops the model is caught.
    pub const MODELS: [&str; 2] = ["probe-a", "probe-bcd"];

    fn support(model: &str, index: usize) -> CapabilitySupport {
        match (index + model.len()) % 3 {
            0 => CapabilitySupport::Supported,
            1 => CapabilitySupport::Unsupported,
            _ => CapabilitySupport::Unknown,
        }
    }

    fn validate<C: Capability>(model: &str, method: &str) -> Result<Validation<C>> {
        if model == Self::REJECTED_MODEL {
            Err(Error::invalid_request(Self::REJECTION))
        } else {
            Ok(
                Validation::new(ValidationCoverage::Encoding).with_warning(Warning::new(
                    WarningKind::Other,
                    Subject::Other {
                        name: method.to_owned(),
                    },
                    format!("the transparency probe validated `{model}` in `{method}`"),
                )),
            )
        }
    }
}

impl ProviderIdentity for TransparencyProbe {
    fn provider_name(&self) -> &str {
        Self::PROVIDER_NAME
    }

    fn reads_namespace(&self, namespace: &str) -> bool {
        namespace == Self::NAMESPACE
    }

    fn downcast_any(&self, id: TypeId) -> Option<&(dyn Any + Send + Sync)> {
        expose(self, id)
    }
}

impl ChatProvider for TransparencyProbe {
    type Stream = ChatStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        data_path::chat(Scenario::of_chat(request))
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
        data_path::chat_stream(Scenario::of_chat(request))
    }

    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        Self::support(model, index_of(ChatCapability::ALL, capability))
    }

    fn validate_chat_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        Self::validate(&request.model, "validate_chat_request")
    }

    fn validate_chat_stream_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        Self::validate(&request.model, "validate_chat_stream_request")
    }
}

impl EmbeddingProvider for TransparencyProbe {
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        data_path::embed(Scenario::of_embedding(request), request)
    }

    fn embedding_capability(
        &self,
        model: &str,
        capability: EmbeddingCapability,
    ) -> CapabilitySupport {
        Self::support(model, index_of(EmbeddingCapability::ALL, capability))
    }

    fn embedding_limits(&self, model: &str) -> EmbeddingLimits {
        EmbeddingLimits::new().max_inputs_per_call(model.len() * 7 + 3)
    }

    fn validate_embedding_request(
        &self,
        request: &EmbeddingRequest,
    ) -> Result<EmbeddingValidation> {
        Self::validate(&request.model, "validate_embedding_request")
    }
}

impl DecisionProvider for TransparencyProbe {
    async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse> {
        data_path::decide(Scenario::of_decision(request), request)
    }

    fn decision_capability(
        &self,
        model: &str,
        capability: DecisionCapability,
    ) -> CapabilitySupport {
        Self::support(model, index_of(DecisionCapability::ALL, capability))
    }

    fn decision_limits(&self, model: &str) -> DecisionLimits {
        DecisionLimits::new()
            .max_questions(model.len() + 2)
            .max_choice_options(model.len() + 5)
            .max_score_levels(model.len() + 11)
    }

    fn validate_decision_request(&self, request: &DecisionRequest) -> Result<DecisionValidation> {
        Self::validate(&request.model, "validate_decision_request")
    }
}

fn check_identity<W: ProviderIdentity + ?Sized>(wrapper: &W) {
    let wrapper_name = std::any::type_name::<W>();
    assert_eq!(
        wrapper.provider_name(),
        TransparencyProbe::PROVIDER_NAME,
        "`{wrapper_name}` does not forward `provider_name` to the provider it wraps",
    );
    let probe = TransparencyProbe::new();
    for namespace in [
        TransparencyProbe::NAMESPACE,
        TransparencyProbe::PROVIDER_NAME,
        allama::provider_data::ALLAMA_NAMESPACE,
        "",
    ] {
        assert_eq!(
            wrapper.reads_namespace(namespace),
            probe.reads_namespace(namespace),
            "`{wrapper_name}` does not forward `reads_namespace` to the provider it wraps \
             (asked about `{namespace}`)",
        );
    }
    assert!(
        wrapper.downcast_ref::<TransparencyProbe>().is_some(),
        "`{wrapper_name}` does not forward `downcast_any` to the provider it wraps",
    );
}

/// The model mapping a check applies: the wrapper's answer for a model must
/// equal the probe's answer for the mapped model.
type ModelMap<'m> = &'m dyn Fn(&str) -> Cow<'_, str>;

/// The mapping of wrappers that do not change models.
fn unmapped(model: &str) -> Cow<'_, str> {
    Cow::Borrowed(model)
}

/// The models every check asks about: the probe's models and the one it
/// rejects.
fn checked_models() -> impl Iterator<Item = &'static str> {
    TransparencyProbe::MODELS
        .into_iter()
        .chain([TransparencyProbe::REJECTED_MODEL])
}

fn check_mapping_changes_a_model(map: ModelMap<'_>) {
    assert!(
        checked_models().any(|model| map(model) != model),
        "the model mapping leaves every probe model unchanged ({:?}), so the check \
         cannot tell whether the wrapper applies it; map at least one of them",
        checked_models().collect::<Vec<_>>(),
    );
}

fn check_chat<W: ChatProvider + ?Sized>(wrapper: &W, map: ModelMap<'_>) {
    let wrapper_name = std::any::type_name::<W>();
    for model in checked_models() {
        let mapped = map(model);
        for &capability in ChatCapability::ALL {
            assert_same(
                wrapper_name,
                "chat_capability",
                (model, capability),
                wrapper.chat_capability(model, capability),
                TransparencyProbe.chat_capability(&mapped, capability),
            );
        }
    }
    check_chat_validation(wrapper, map);
}

fn check_chat_validation<W: ChatProvider + ?Sized>(wrapper: &W, map: ModelMap<'_>) {
    let wrapper_name = std::any::type_name::<W>();
    for model in checked_models() {
        let mapped = map(model);
        let request = ChatRequest::new(model).user("probe");
        check_validation(
            wrapper_name,
            "validate_chat_request",
            model,
            wrapper.validate_chat_request(&request),
            TransparencyProbe::validate(&mapped, "validate_chat_request"),
        );
        check_validation(
            wrapper_name,
            "validate_chat_stream_request",
            model,
            wrapper.validate_chat_stream_request(&request),
            TransparencyProbe::validate(&mapped, "validate_chat_stream_request"),
        );
    }
}

fn check_embedding<W: EmbeddingProvider + ?Sized>(wrapper: &W, map: ModelMap<'_>) {
    let wrapper_name = std::any::type_name::<W>();
    for model in checked_models() {
        let mapped = map(model);
        for &capability in EmbeddingCapability::ALL {
            assert_same(
                wrapper_name,
                "embedding_capability",
                (model, capability),
                wrapper.embedding_capability(model, capability),
                TransparencyProbe.embedding_capability(&mapped, capability),
            );
        }
        assert_same(
            wrapper_name,
            "embedding_limits",
            model,
            wrapper.embedding_limits(model),
            TransparencyProbe.embedding_limits(&mapped),
        );
    }
    check_embedding_validation(wrapper, map);
}

fn check_embedding_validation<W: EmbeddingProvider + ?Sized>(wrapper: &W, map: ModelMap<'_>) {
    let wrapper_name = std::any::type_name::<W>();
    for model in checked_models() {
        check_validation(
            wrapper_name,
            "validate_embedding_request",
            model,
            wrapper.validate_embedding_request(&EmbeddingRequest::new(model).input("probe")),
            TransparencyProbe::validate(&map(model), "validate_embedding_request"),
        );
    }
}

fn check_decision<W: DecisionProvider + ?Sized>(wrapper: &W, map: ModelMap<'_>) {
    let wrapper_name = std::any::type_name::<W>();
    for model in checked_models() {
        let mapped = map(model);
        for &capability in DecisionCapability::ALL {
            assert_same(
                wrapper_name,
                "decision_capability",
                (model, capability),
                wrapper.decision_capability(model, capability),
                TransparencyProbe.decision_capability(&mapped, capability),
            );
        }
        assert_same(
            wrapper_name,
            "decision_limits",
            model,
            wrapper.decision_limits(model),
            TransparencyProbe.decision_limits(&mapped),
        );
    }
    check_decision_validation(wrapper, map);
}

fn check_decision_validation<W: DecisionProvider + ?Sized>(wrapper: &W, map: ModelMap<'_>) {
    let wrapper_name = std::any::type_name::<W>();
    for model in checked_models() {
        check_validation(
            wrapper_name,
            "validate_decision_request",
            model,
            wrapper.validate_decision_request(&decision_request(model)),
            TransparencyProbe::validate(&map(model), "validate_decision_request"),
        );
    }
}

fn assert_same<A: Debug, T: Debug + PartialEq>(
    wrapper_name: &str,
    method: &str,
    args: A,
    actual: T,
    expected: T,
) {
    assert_eq!(
        actual, expected,
        "`{wrapper_name}` does not forward `{method}` to the provider it wraps (arguments: {args:?})",
    );
}

fn check_validation<C: Capability>(
    wrapper_name: &str,
    method: &str,
    model: &str,
    actual: Result<Validation<C>>,
    expected: Result<Validation<C>>,
) {
    match (actual, expected) {
        (Ok(actual), Ok(expected))
            if actual.warnings() == expected.warnings()
                && actual.coverage() == expected.coverage()
                && actual.relied_on_unknown() == expected.relied_on_unknown() => {}
        (Err(error), Err(_)) if error.to_string().contains(TransparencyProbe::REJECTION) => {}
        (actual, Ok(_)) => panic!(
            "`{wrapper_name}` does not forward `{method}` to the provider it wraps: \
             it rejects a request for `{model}` that the provider accepts, got {actual:?}"
        ),
        (actual, Err(_)) => panic!(
            "`{wrapper_name}` does not forward `{method}` to the provider it wraps: \
             expected the probe's rejection for `{model}`, got {actual:?}"
        ),
    }
}

/// A request with a question of a kind the probe does not answer
/// `Unsupported` for `model`, so a wrapper that checks requests itself
/// (with `allama::adapter::decision::prepare`) forwards it.
fn decision_request(model: &str) -> DecisionRequest {
    let kinds = [
        DecisionCapability::YesNo,
        DecisionCapability::Choice,
        DecisionCapability::Score,
    ];
    let question: Question = match kinds.iter().find(|&&kind| {
        TransparencyProbe.decision_capability(model, kind) != CapabilitySupport::Unsupported
    }) {
        Some(DecisionCapability::Choice) => Question::choice("probe?").options(["a", "b"]).into(),
        Some(DecisionCapability::Score) => Question::score("probe?").levels(["low", "high"]).into(),
        _ => Question::yes_no("probe?").into(),
    };
    DecisionRequest::new(model, "probe").question("q", question)
}

fn index_of<T: PartialEq>(all: &[T], item: T) -> usize {
    all.iter()
        .position(|candidate| *candidate == item)
        .unwrap_or(all.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A wrapper that forwards everything except one method.
    struct Leaky<P> {
        inner: P,
        leak: Leak,
    }

    #[derive(PartialEq)]
    enum Leak {
        Name,
        /// Answers `reads_namespace` by its name, the tempting default.
        NamespaceByName,
        Downcast,
        ChatCapability,
        ChatValidation,
        ChatStreamValidation,
        EmbeddingCapability,
        EmbeddingLimits,
        EmbeddingValidation,
        DecisionCapability,
        DecisionLimits,
        DecisionValidation,
        /// Drops the provider's warnings.
        SwallowWarnings,
        /// Adds a warning of its own: allowed.
        AddWarning,
        /// Drops stream warnings.
        SwallowStreamWarnings,
        /// Re-wraps errors in a provider error.
        RewrapErrors,
        /// Prefixes errors: allowed.
        PrefixErrors,
        /// Drops the usage of embedding responses.
        DropEmbeddingUsage,
        /// Re-wraps decision errors.
        RewrapDecisionErrors,
    }

    impl Leak {
        fn error(&self, error: Error) -> Error {
            match self {
                Leak::RewrapErrors | Leak::RewrapDecisionErrors => {
                    Error::provider(format!("leaky: {error}"))
                }
                Leak::PrefixErrors => error.with_message_prefix("leaky"),
                _ => error,
            }
        }
    }

    fn leaky(leak: Leak) -> impl FnOnce(TransparencyProbe) -> Leaky<TransparencyProbe> {
        |inner| Leaky { inner, leak }
    }

    impl<P: ProviderIdentity> ProviderIdentity for Leaky<P> {
        fn provider_name(&self) -> &str {
            if self.leak == Leak::Name {
                "leaky"
            } else {
                self.inner.provider_name()
            }
        }

        fn reads_namespace(&self, namespace: &str) -> bool {
            if self.leak == Leak::NamespaceByName {
                namespace == self.provider_name()
            } else {
                self.inner.reads_namespace(namespace)
            }
        }

        fn downcast_any(&self, id: TypeId) -> Option<&(dyn Any + Send + Sync)> {
            if self.leak == Leak::Downcast {
                None
            } else {
                self.inner.downcast_any(id)
            }
        }
    }

    impl<P: ChatProvider> ChatProvider for Leaky<P> {
        type Stream = ChatStream;

        async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
            let mut response = self
                .inner
                .chat(request)
                .await
                .map_err(|error| self.leak.error(error))?;
            match self.leak {
                Leak::SwallowWarnings => response.warnings.clear(),
                Leak::AddWarning => response.warnings.insert(
                    0,
                    Warning::new(WarningKind::Other, Subject::Response, "the wrapper's own"),
                ),
                _ => {}
            }
            Ok(response)
        }

        async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
            use futures_util::StreamExt;
            let stream = self
                .inner
                .chat_stream(request)
                .await
                .map_err(|error| self.leak.error(error))?;
            let swallow = self.leak == Leak::SwallowStreamWarnings;
            let rewrap = self.leak == Leak::RewrapErrors;
            Ok(Box::pin(stream.filter_map(move |item| {
                std::future::ready(match item {
                    Ok(allama::StreamEvent::Warning { .. }) if swallow => None,
                    Err(error) if rewrap => Some(Err(Error::provider(format!("leaky: {error}")))),
                    item => Some(item),
                })
            })))
        }

        fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
            if self.leak == Leak::ChatCapability {
                CapabilitySupport::Unknown
            } else {
                self.inner.chat_capability(model, capability)
            }
        }

        fn validate_chat_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
            if self.leak == Leak::ChatValidation {
                // What the default would answer.
                Ok(allama::adapter::chat::prepare_dry_run(self, request)?
                    .into_validation(ValidationCoverage::Capabilities))
            } else {
                self.inner.validate_chat_request(request)
            }
        }

        fn validate_chat_stream_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
            if self.leak == Leak::ChatStreamValidation {
                // A wrapper that forwards to the unary validator.
                self.inner.validate_chat_request(request)
            } else {
                self.inner.validate_chat_stream_request(request)
            }
        }
    }

    impl<P: EmbeddingProvider> EmbeddingProvider for Leaky<P> {
        async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
            let mut response = self.inner.embed(request).await?;
            if self.leak == Leak::DropEmbeddingUsage {
                response.usage = None;
            }
            Ok(response)
        }

        fn embedding_capability(
            &self,
            model: &str,
            capability: EmbeddingCapability,
        ) -> CapabilitySupport {
            if self.leak == Leak::EmbeddingCapability {
                CapabilitySupport::Unknown
            } else {
                self.inner.embedding_capability(model, capability)
            }
        }

        fn embedding_limits(&self, model: &str) -> EmbeddingLimits {
            if self.leak == Leak::EmbeddingLimits {
                EmbeddingLimits::default()
            } else {
                self.inner.embedding_limits(model)
            }
        }

        fn validate_embedding_request(
            &self,
            request: &EmbeddingRequest,
        ) -> Result<EmbeddingValidation> {
            if self.leak == Leak::EmbeddingValidation {
                Ok(allama::adapter::embedding::prepare_dry_run(self, request)?
                    .into_validation(ValidationCoverage::Capabilities))
            } else {
                self.inner.validate_embedding_request(request)
            }
        }
    }

    impl<P: DecisionProvider> DecisionProvider for Leaky<P> {
        async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse> {
            self.inner
                .decide(request)
                .await
                .map_err(|error| match self.leak {
                    Leak::RewrapDecisionErrors => self.leak.error(error),
                    _ => error,
                })
        }

        fn decision_capability(
            &self,
            model: &str,
            capability: DecisionCapability,
        ) -> CapabilitySupport {
            if self.leak == Leak::DecisionCapability {
                CapabilitySupport::Unknown
            } else {
                self.inner.decision_capability(model, capability)
            }
        }

        fn decision_limits(&self, model: &str) -> DecisionLimits {
            if self.leak == Leak::DecisionLimits {
                DecisionLimits::default()
            } else {
                self.inner.decision_limits(model)
            }
        }

        fn validate_decision_request(
            &self,
            request: &DecisionRequest,
        ) -> Result<DecisionValidation> {
            if self.leak == Leak::DecisionValidation {
                Ok(allama::adapter::decision::prepare_dry_run(self, request)?
                    .into_validation(ValidationCoverage::Capabilities))
            } else {
                self.inner.validate_decision_request(request)
            }
        }
    }

    #[test]
    fn the_probe_is_transparent_to_itself() {
        assert_transparent_wrapper(|probe| probe);
        assert_transparent_identity(|probe| probe);
    }

    #[tokio::test]
    async fn the_probe_passes_its_own_data_path() {
        assert_wrapper(|probe| probe).await;
    }

    #[tokio::test]
    async fn a_wrapper_may_add_warnings_and_prefix_errors() {
        assert_wrapper(leaky(Leak::AddWarning)).await;
        assert_wrapper(leaky(Leak::PrefixErrors)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "chat (Ok): the probe's warnings")]
    async fn a_wrapper_that_swallows_warnings_fails() {
        assert_transparent_chat_data_path(leaky(Leak::SwallowWarnings)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "chat_stream (Ok): the probe's event {\"event\":\"warning\"")]
    async fn a_wrapper_that_swallows_stream_warnings_fails() {
        assert_chat_wrapper(leaky(Leak::SwallowStreamWarnings)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "the rate_limited error changed")]
    async fn a_wrapper_that_rewraps_errors_fails() {
        assert_chat_wrapper(leaky(Leak::RewrapErrors)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "embed (Ok): the response changed")]
    async fn a_wrapper_that_drops_embedding_usage_fails() {
        assert_embedding_wrapper(leaky(Leak::DropEmbeddingUsage)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "decide (Error(0)): the rate_limited error changed")]
    async fn a_wrapper_that_rewraps_decision_errors_fails() {
        assert_decision_wrapper(leaky(Leak::RewrapDecisionErrors)).await;
    }

    #[test]
    fn the_probe_mixes_capability_answers() {
        for capabilities in [
            ChatCapability::ALL
                .iter()
                .map(|&c| TransparencyProbe.chat_capability(TransparencyProbe::MODELS[0], c))
                .collect::<Vec<_>>(),
            DecisionCapability::ALL
                .iter()
                .map(|&c| TransparencyProbe.decision_capability(TransparencyProbe::MODELS[0], c))
                .collect(),
        ] {
            assert!(capabilities.contains(&CapabilitySupport::Supported));
            assert!(capabilities.contains(&CapabilitySupport::Unsupported));
            assert!(capabilities.contains(&CapabilitySupport::Unknown));
        }
        let [a, b] = TransparencyProbe::MODELS;
        assert_ne!(
            TransparencyProbe.embedding_capability(a, EmbeddingCapability::BatchInput),
            TransparencyProbe.embedding_capability(b, EmbeddingCapability::BatchInput),
        );
    }

    /// Swaps the probe's models and maps the second to the rejected one.
    fn shuffle(model: &str) -> Cow<'_, str> {
        let [a, b] = TransparencyProbe::MODELS;
        match model {
            m if m == a => b.into(),
            m if m == b => TransparencyProbe::REJECTED_MODEL.into(),
            _ => a.into(),
        }
    }

    /// Asks the wrapped provider about `shuffle(model)`.
    struct Shuffled<P>(P);

    impl<P: ProviderIdentity> ProviderIdentity for Shuffled<P> {
        fn provider_name(&self) -> &str {
            self.0.provider_name()
        }

        fn reads_namespace(&self, namespace: &str) -> bool {
            self.0.reads_namespace(namespace)
        }

        fn downcast_any(&self, id: TypeId) -> Option<&(dyn Any + Send + Sync)> {
            self.0.downcast_any(id)
        }
    }

    impl<P: ChatProvider> ChatProvider for Shuffled<P> {
        type Stream = P::Stream;

        async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
            self.0.chat(request).await
        }

        async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
            self.0.chat_stream(request).await
        }

        fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
            self.0.chat_capability(&shuffle(model), capability)
        }

        fn validate_chat_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
            let mut request = request.clone();
            request.model = shuffle(&request.model).into_owned();
            self.0.validate_chat_request(&request)
        }

        fn validate_chat_stream_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
            let mut request = request.clone();
            request.model = shuffle(&request.model).into_owned();
            self.0.validate_chat_stream_request(&request)
        }
    }

    #[test]
    fn a_mapping_wrapper_passes_with_its_mapping() {
        assert_transparent_chat_model_mapping(Shuffled, shuffle);
    }

    #[test]
    #[should_panic(expected = "does not forward `chat_capability`")]
    fn a_mapping_wrapper_fails_the_plain_check() {
        assert_transparent_chat_wrapper(Shuffled);
    }

    #[test]
    #[should_panic(expected = "does not forward `chat_capability`")]
    fn a_wrapper_that_ignores_the_mapping_fails() {
        assert_transparent_chat_model_mapping(|probe| probe, shuffle);
    }

    #[test]
    #[should_panic(expected = "leaves every probe model unchanged")]
    fn an_identity_mapping_is_refused() {
        assert_transparent_chat_model_mapping(|probe| probe, unmapped);
    }

    #[test]
    #[should_panic(expected = "does not forward `provider_name`")]
    fn a_wrapper_that_renames_fails() {
        assert_transparent_identity(leaky(Leak::Name));
    }

    #[test]
    #[should_panic(expected = "does not forward `reads_namespace`")]
    fn a_wrapper_that_answers_namespaces_by_name_fails() {
        assert_transparent_identity(leaky(Leak::NamespaceByName));
    }

    #[test]
    #[should_panic(expected = "does not forward `downcast_any`")]
    fn a_wrapper_that_hides_the_provider_fails() {
        assert_transparent_wrapper(leaky(Leak::Downcast));
    }

    #[test]
    #[should_panic(expected = "does not forward `chat_capability`")]
    fn a_wrapper_that_drops_chat_capabilities_fails() {
        assert_transparent_chat_wrapper(leaky(Leak::ChatCapability));
    }

    #[test]
    #[should_panic(expected = "does not forward `validate_chat_request`")]
    fn a_wrapper_that_skips_chat_validation_fails() {
        assert_transparent_chat_wrapper(leaky(Leak::ChatValidation));
    }

    #[test]
    #[should_panic(expected = "does not forward `validate_chat_stream_request`")]
    fn a_wrapper_that_validates_streams_as_unary_calls_fails() {
        assert_transparent_chat_wrapper(leaky(Leak::ChatStreamValidation));
    }

    #[test]
    #[should_panic(expected = "does not forward `embedding_capability`")]
    fn a_wrapper_that_drops_embedding_capabilities_fails() {
        assert_transparent_embedding_wrapper(leaky(Leak::EmbeddingCapability));
    }

    #[test]
    #[should_panic(expected = "does not forward `embedding_limits`")]
    fn a_wrapper_that_drops_the_batch_limit_fails() {
        assert_transparent_embedding_wrapper(leaky(Leak::EmbeddingLimits));
    }

    #[test]
    #[should_panic(expected = "does not forward `validate_embedding_request`")]
    fn a_wrapper_that_skips_embedding_validation_fails() {
        assert_transparent_embedding_wrapper(leaky(Leak::EmbeddingValidation));
    }

    #[test]
    #[should_panic(expected = "does not forward `decision_capability`")]
    fn a_wrapper_that_drops_decision_capabilities_fails() {
        assert_transparent_decision_wrapper(leaky(Leak::DecisionCapability));
    }

    #[test]
    #[should_panic(expected = "does not forward `decision_limits`")]
    fn a_wrapper_that_drops_decision_limits_fails() {
        assert_transparent_decision_wrapper(leaky(Leak::DecisionLimits));
    }

    #[test]
    #[should_panic(expected = "does not forward `validate_decision_request`")]
    fn a_wrapper_that_skips_decision_validation_fails() {
        assert_transparent_decision_wrapper(leaky(Leak::DecisionValidation));
    }
}
