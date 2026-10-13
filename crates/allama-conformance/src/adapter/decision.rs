//! The decision aggregate.

use std::future::Future;
use std::time::Duration;

use allama::{DecisionProvider, DecisionRequest, DecisionResponse, Error, RequestOptions};
use allama_http::CredentialSlot;

use super::{Contract, ErrorInputs, KnownGap, Runner, TransportSend, boxed_send};
use crate::TransportOutcome;
use crate::capabilities::UNKNOWN_MODEL;
use crate::probe::ProbeOutcome;
use crate::stream_errors::ErrorKindProbe;

/// Assert that a decision adapter passes every probe-based decision
/// contract: [`assert_chat_adapter`](super::assert_chat_adapter) for
/// [`DecisionProvider`].
///
/// | Contract | Runs | Needs |
/// |---|---|---|
/// | [`ValidationAgrees`](Contract::ValidationAgrees) | [`decision_drift_requests`](crate::decision_drift_requests) of each model and [`UNKNOWN_MODEL`], and [`validation_requests`](DecisionAdapterUnderTest::validation_requests) | |
/// | [`ErrorKindsMapped`](Contract::ErrorKindsMapped) | | [`error_mapping`](DecisionAdapterUnderTest::error_mapping), [`error_envelope`](DecisionAdapterUnderTest::error_envelope) |
/// | [`ErrorLogExcludesBody`](Contract::ErrorLogExcludesBody) | | [`error_mapping`](DecisionAdapterUnderTest::error_mapping), [`error_envelope`](DecisionAdapterUnderTest::error_envelope) |
/// | [`RetrySignalsSurfaced`](Contract::RetrySignalsSurfaced) | | [`error_mapping`](DecisionAdapterUnderTest::error_mapping) |
/// | [`TransportOptionsApplied`](Contract::TransportOptionsApplied) | `decide` | [`transport_options`](DecisionAdapterUnderTest::transport_options) |
///
/// # Panics
///
/// As [`assert_chat_adapter`](super::assert_chat_adapter).
pub async fn assert_decision_adapter<P, F, Fut>(
    provider: &P,
    under_test: &DecisionAdapterUnderTest<F>,
) where
    P: DecisionProvider,
    F: Fn(DecisionRequest) -> Fut,
    Fut: Future<Output = ProbeOutcome<DecisionResponse>>,
{
    let t = under_test;
    let mut run = Runner::new(
        "assert_decision_adapter",
        provider.provider_name(),
        &t.skip,
        &t.known_gaps,
    );
    let probe = |request| (t.probe)(request);
    t.errors.require(&mut run);
    run.needs(
        Contract::TransportOptionsApplied,
        t.transport.is_none(),
        "`.transport_options(slot, send)`",
    );
    run.check_inputs();

    run.contract(Contract::ValidationAgrees, async {
        let mut requests: Vec<DecisionRequest> = t
            .models
            .iter()
            .map(String::as_str)
            .chain((!t.models.iter().any(|m| m == UNKNOWN_MODEL)).then_some(UNKNOWN_MODEL))
            .flat_map(crate::decision_drift_requests)
            .collect();
        requests.extend(t.validation_requests.iter().cloned());
        crate::assert_decision_validation_agrees(provider, probe, &requests).await;
    })
    .await;
    t.errors.run(&mut run).await;
    if let Some((slot, send)) = &t.transport {
        run.contract(Contract::TransportOptionsApplied, async {
            crate::assert_transport_options_applied(slot, send).await;
        })
        .await;
    }
    run.finish();
}

/// What [`assert_decision_adapter`] needs from a decision adapter.
///
/// Built with [`new`](Self::new) and setters, like
/// [`AdapterUnderTest`](super::AdapterUnderTest).
#[non_exhaustive]
pub struct DecisionAdapterUnderTest<F> {
    models: Vec<String>,
    probe: F,
    known_gaps: Vec<KnownGap>,
    skip: Vec<Contract>,
    validation_requests: Vec<DecisionRequest>,
    errors: ErrorInputs,
    transport: Option<(CredentialSlot, TransportSend)>,
}

impl<F> DecisionAdapterUnderTest<F> {
    /// The models to probe, and the probe: it sends a request through
    /// `decide` over a transport that answers the request's questions (see
    /// [probes](crate::probe)).
    pub fn new<M: Into<String>>(models: impl IntoIterator<Item = M>, probe: F) -> Self {
        Self {
            models: models.into_iter().map(Into::into).collect(),
            probe,
            known_gaps: Vec::new(),
            skip: Vec::new(),
            validation_requests: Vec::new(),
            errors: ErrorInputs::default(),
            transport: None,
        }
    }

    /// Acknowledged defects (see
    /// [`AdapterUnderTest::known_gaps`](super::AdapterUnderTest::known_gaps)):
    /// for decisions, whole contracts.
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
        requests: impl IntoIterator<Item = DecisionRequest>,
    ) -> Self {
        self.validation_requests.extend(requests);
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

    /// The adapter's credential slot, and a `decide` call with transport
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

impl<F> std::fmt::Debug for DecisionAdapterUnderTest<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecisionAdapterUnderTest")
            .field("models", &self.models)
            .field("known_gaps", &self.known_gaps)
            .field("skip", &self.skip)
            .field("validation_requests", &self.validation_requests.len())
            .finish_non_exhaustive()
    }
}
