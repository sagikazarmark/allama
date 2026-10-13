//! The embedding aggregate.

use std::future::Future;
use std::time::Duration;

use allama::{
    CapabilitySupport, EmbeddingCapability, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse,
    Error, RequestOptions,
};
use allama_http::CredentialSlot;

use super::{BoxFuture, Contract, ErrorInputs, KnownGap, Runner, TransportSend, boxed_send};
use crate::TransportOutcome;
use crate::capabilities::{UNKNOWN_MODEL, assert_unsupported_embedding_capabilities_refused};
use crate::probe::{
    ProbeOutcome, VectorOrder, assert_batch_limit_enforced, assert_embedding_count,
    assert_embedding_response_checked, assert_input_provider_data_checked,
    assert_input_type_not_honoured_warns, assert_multi_part_input_refused,
};
use crate::stream_errors::ErrorKindProbe;

/// Assert that an embedding adapter passes every probe-based embedding
/// contract: [`assert_chat_adapter`](super::assert_chat_adapter) for
/// [`EmbeddingProvider`].
///
/// | Contract | Runs | Needs |
/// |---|---|---|
/// | [`ValidationAgrees`](Contract::ValidationAgrees) | [`embedding_drift_requests`](crate::embedding_drift_requests) of each model and [`UNKNOWN_MODEL`], and [`validation_requests`](EmbeddingAdapterUnderTest::validation_requests) | |
/// | [`UnsupportedCapabilitiesRefused`](Contract::UnsupportedCapabilitiesRefused) | each model and [`UNKNOWN_MODEL`] | |
/// | [`InputProviderDataChecked`](Contract::InputProviderDataChecked) | the first model, the namespace | |
/// | [`BatchLimitEnforced`](Contract::BatchLimitEnforced) | each model with a `max_inputs_per_call` | one such model |
/// | [`InputTypeNotHonouredWarns`](Contract::InputTypeNotHonouredWarns) | each model that answers `InputType` `Unsupported` | one such model |
/// | [`MultiPartInputRefused`](Contract::MultiPartInputRefused) | each model that answers `CompositeInput` `Unsupported` | one such model |
/// | [`EmbeddingResponseChecked`](Contract::EmbeddingResponseChecked) | each model | [`answering`](EmbeddingAdapterUnderTest::answering) |
/// | [`EmbeddingCount`](Contract::EmbeddingCount) | the first model | [`answering`](EmbeddingAdapterUnderTest::answering) |
/// | [`ErrorKindsMapped`](Contract::ErrorKindsMapped) | | [`error_mapping`](EmbeddingAdapterUnderTest::error_mapping), [`error_envelope`](EmbeddingAdapterUnderTest::error_envelope) |
/// | [`ErrorLogExcludesBody`](Contract::ErrorLogExcludesBody) | | [`error_mapping`](EmbeddingAdapterUnderTest::error_mapping), [`error_envelope`](EmbeddingAdapterUnderTest::error_envelope) |
/// | [`RetrySignalsSurfaced`](Contract::RetrySignalsSurfaced) | | [`error_mapping`](EmbeddingAdapterUnderTest::error_mapping) |
/// | [`TransportOptionsApplied`](Contract::TransportOptionsApplied) | `embed` | [`transport_options`](EmbeddingAdapterUnderTest::transport_options) |
///
/// An adapter that also serves chat passes the same error inputs to both
/// aggregates.
///
/// # Panics
///
/// As [`assert_chat_adapter`](super::assert_chat_adapter).
pub async fn assert_embedding_adapter<P, F, Fut>(
    provider: &P,
    under_test: &EmbeddingAdapterUnderTest<F>,
) where
    P: EmbeddingProvider,
    F: Fn(EmbeddingRequest) -> Fut,
    Fut: Future<Output = ProbeOutcome<EmbeddingResponse>>,
{
    let t = under_test;
    let name = provider.provider_name();
    let mut run = Runner::new("assert_embedding_adapter", name, &t.skip, &t.known_gaps);
    let first = t.models.first().map_or(UNKNOWN_MODEL, String::as_str);
    let probe = |request| (t.probe)(request);
    let with_unknown: Vec<&str> = t
        .models
        .iter()
        .map(String::as_str)
        .chain((!t.models.iter().any(|m| m == UNKNOWN_MODEL)).then_some(UNKNOWN_MODEL))
        .collect();
    let answering = |capability| {
        t.models
            .iter()
            .map(String::as_str)
            .filter(|model| {
                provider.embedding_capability(model, capability) == CapabilitySupport::Unsupported
            })
            .collect::<Vec<_>>()
    };
    let limited: Vec<&str> = t
        .models
        .iter()
        .map(String::as_str)
        .filter(|model| {
            provider
                .embedding_limits(model)
                .max_inputs_per_call
                .is_some()
        })
        .collect();
    let without_input_type = answering(EmbeddingCapability::InputType);
    let one_text_per_input = answering(EmbeddingCapability::CompositeInput);

    run.needs(
        Contract::BatchLimitEnforced,
        limited.is_empty(),
        "a listed model with a `max_inputs_per_call`",
    );
    run.needs(
        Contract::InputTypeNotHonouredWarns,
        without_input_type.is_empty(),
        "a listed model that answers InputType Unsupported",
    );
    run.needs(
        Contract::MultiPartInputRefused,
        one_text_per_input.is_empty(),
        "a listed model that answers CompositeInput Unsupported",
    );
    for contract in [Contract::EmbeddingResponseChecked, Contract::EmbeddingCount] {
        run.needs(
            contract,
            t.answering.is_none(),
            "`.answering(order, answer)`",
        );
    }
    t.errors.require(&mut run);
    run.needs(
        Contract::TransportOptionsApplied,
        t.transport.is_none(),
        "`.transport_options(slot, send)`",
    );
    run.check_inputs();

    run.contract(Contract::ValidationAgrees, async {
        let mut requests: Vec<EmbeddingRequest> = with_unknown
            .iter()
            .flat_map(|model| crate::embedding_drift_requests(model))
            .collect();
        requests.extend(t.validation_requests.iter().cloned());
        crate::assert_embedding_validation_agrees(provider, probe, &requests).await;
    })
    .await;
    let capability_gaps: Vec<EmbeddingCapability> = t
        .known_gaps
        .iter()
        .filter_map(|gap| match gap {
            KnownGap::EmbeddingCapability(capability) => Some(*capability),
            _ => None,
        })
        .collect();
    run.contract(Contract::UnsupportedCapabilitiesRefused, async {
        for model in &with_unknown {
            assert_unsupported_embedding_capabilities_refused(
                provider,
                model,
                &capability_gaps,
                probe,
            )
            .await;
        }
    })
    .await;
    run.contract(Contract::InputProviderDataChecked, async {
        assert_input_provider_data_checked(first, name, &t.namespace, probe).await;
    })
    .await;
    run.contract(Contract::BatchLimitEnforced, async {
        for model in &limited {
            assert_batch_limit_enforced(provider, model, probe).await;
        }
    })
    .await;
    run.contract(Contract::InputTypeNotHonouredWarns, async {
        for model in &without_input_type {
            assert_input_type_not_honoured_warns(model, probe).await;
        }
    })
    .await;
    run.contract(Contract::MultiPartInputRefused, async {
        for model in &one_text_per_input {
            assert_multi_part_input_refused(model, name, probe).await;
        }
    })
    .await;
    if let Some(answering) = &t.answering {
        let answer = &answering.answer;
        run.contract(Contract::EmbeddingResponseChecked, async {
            for model in &t.models {
                assert_embedding_response_checked(provider, model, |request, vectors| {
                    let vectors = vectors
                        .into_iter()
                        .enumerate()
                        .map(|(index, values)| AnsweredVector { index, values })
                        .collect();
                    answer(request, vectors)
                })
                .await;
            }
        })
        .await;
        run.contract(Contract::EmbeddingCount, async {
            assert_embedding_count(first, answering.order, |request, indices| {
                let vectors = indices
                    .into_iter()
                    .map(|index| AnsweredVector {
                        index,
                        values: vec![index as f64],
                    })
                    .collect();
                answer(request, vectors)
            })
            .await;
        })
        .await;
    }
    t.errors.run(&mut run).await;
    if let Some((slot, send)) = &t.transport {
        run.contract(Contract::TransportOptionsApplied, async {
            crate::assert_transport_options_applied(slot, send).await;
        })
        .await;
    }
    run.finish();
}

/// One vector an [`answering`](EmbeddingAdapterUnderTest::answering)
/// transport writes into its canned response.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct AnsweredVector {
    /// The input index to tag the vector with, for a wire format that has
    /// one ([`VectorOrder::Indexed`]); a format without one writes the
    /// vectors in the order given.
    pub index: usize,
    /// The values, written as JSON numbers.
    pub values: Vec<f64>,
}

/// What [`assert_embedding_adapter`] needs from an embedding adapter.
///
/// Built with [`new`](Self::new) and setters, like
/// [`AdapterUnderTest`](super::AdapterUnderTest).
#[non_exhaustive]
pub struct EmbeddingAdapterUnderTest<F> {
    models: Vec<String>,
    namespace: String,
    probe: F,
    known_gaps: Vec<KnownGap>,
    skip: Vec<Contract>,
    validation_requests: Vec<EmbeddingRequest>,
    answering: Option<Answering>,
    errors: ErrorInputs,
    transport: Option<(CredentialSlot, TransportSend)>,
}

/// An `embed` call over a transport that answers with given vectors.
type Answer = Box<
    dyn Fn(EmbeddingRequest, Vec<AnsweredVector>) -> BoxFuture<allama::Result<EmbeddingResponse>>,
>;

struct Answering {
    order: VectorOrder,
    answer: Answer,
}

impl<F> EmbeddingAdapterUnderTest<F> {
    /// The models to probe, the adapter's provider-data `namespace`, and
    /// the probe: it sends a request through `embed` over a transport that
    /// answers one vector per input (see [probes](crate::probe)).
    pub fn new<M: Into<String>>(
        models: impl IntoIterator<Item = M>,
        namespace: impl Into<String>,
        probe: F,
    ) -> Self {
        Self {
            models: models.into_iter().map(Into::into).collect(),
            namespace: namespace.into(),
            probe,
            known_gaps: Vec::new(),
            skip: Vec::new(),
            validation_requests: Vec::new(),
            answering: None,
            errors: ErrorInputs::default(),
            transport: None,
        }
    }

    /// Acknowledged defects (see
    /// [`AdapterUnderTest::known_gaps`](super::AdapterUnderTest::known_gaps)).
    #[must_use]
    pub fn known_gaps<K: Into<KnownGap>>(mut self, gaps: impl IntoIterator<Item = K>) -> Self {
        self.known_gaps.extend(gaps.into_iter().map(Into::into));
        self
    }

    /// Contracts that do not apply to the adapter.
    #[must_use]
    pub fn skip(mut self, contracts: impl IntoIterator<Item = Contract>) -> Self {
        self.skip.extend(contracts);
        self
    }

    /// Requests [`ValidationAgrees`](Contract::ValidationAgrees) checks
    /// beyond the drift requests.
    #[must_use]
    pub fn validation_requests(
        mut self,
        requests: impl IntoIterator<Item = EmbeddingRequest>,
    ) -> Self {
        self.validation_requests.extend(requests);
        self
    }

    /// Sends a request through `embed` over a transport that answers with
    /// the given vectors, for the response check and vector count
    /// contracts; `order` says whether the wire format tags vectors with
    /// their input index.
    #[must_use]
    pub fn answering<A, AFut>(mut self, order: VectorOrder, answer: A) -> Self
    where
        A: Fn(EmbeddingRequest, Vec<AnsweredVector>) -> AFut + 'static,
        AFut: Future<Output = allama::Result<EmbeddingResponse>> + 'static,
    {
        self.answering = Some(Answering {
            order,
            answer: Box::new(move |request, vectors| Box::pin(answer(request, vectors))),
        });
        self
    }

    /// The adapter's HTTP error mapping (see
    /// [`AdapterUnderTest::error_mapping`](super::AdapterUnderTest::error_mapping)).
    #[must_use]
    pub fn error_mapping(
        mut self,
        map: impl Fn(u16, &[(&str, &str)], String) -> Error + 'static,
    ) -> Self {
        self.errors.mapping = Some(Box::new(map));
        self
    }

    /// The adapter's error envelope for each [`ErrorKindProbe`] (see
    /// [`AdapterUnderTest::error_envelope`](super::AdapterUnderTest::error_envelope)).
    #[must_use]
    pub fn error_envelope(
        mut self,
        envelope: impl Fn(ErrorKindProbe, &str) -> Option<String> + 'static,
    ) -> Self {
        self.errors.envelope = Some(Box::new(envelope));
        self
    }

    /// Writes an error body that carries a retry hint (see
    /// [`AdapterUnderTest::retry_hint_body`](super::AdapterUnderTest::retry_hint_body)).
    #[must_use]
    pub fn retry_hint_body(mut self, encode: impl Fn(Duration) -> String + 'static) -> Self {
        self.errors.retry_hint_body = Some(Box::new(encode));
        self
    }

    /// The adapter's credential slot, and an `embed` call with transport
    /// options attached (see
    /// [`assert_transport_options_applied`](crate::assert_transport_options_applied)).
    #[must_use]
    pub fn transport_options<S, SFut>(mut self, slot: CredentialSlot, send: S) -> Self
    where
        S: Fn(RequestOptions) -> SFut + 'static,
        SFut: Future<Output = TransportOutcome> + 'static,
    {
        self.transport = Some((slot, boxed_send(send)));
        self
    }
}

impl<F> std::fmt::Debug for EmbeddingAdapterUnderTest<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingAdapterUnderTest")
            .field("models", &self.models)
            .field("namespace", &self.namespace)
            .field("known_gaps", &self.known_gaps)
            .field("skip", &self.skip)
            .field("validation_requests", &self.validation_requests.len())
            .finish_non_exhaustive()
    }
}
