use allama::mock::{MockEmbeddingProvider, MockProvider};
use allama::{
    ChatResponse, DecisionRequest, DecisionResponse, EmbeddingRequest, EmbeddingResponse,
};

use super::*;

/// Every chat contract.
const CHAT: &[Contract] = &[
    Contract::CapabilitiesHonest,
    Contract::UnsupportedCapabilitiesRefused,
    Contract::StreamUsageMatchesCapability,
    Contract::ValidationAgrees,
    Contract::RequestFieldsAccounted,
    Contract::AppDataNeverSent,
    Contract::OtherProviderDataGated,
    Contract::EmptyTurnsSkipped,
    Contract::UnreplayableReasoningDropped,
    Contract::StrictNotHonouredWarns,
    Contract::ErrorKindsMapped,
    Contract::ErrorLogExcludesBody,
    Contract::RetrySignalsSurfaced,
    Contract::StreamErrorsClassified,
    Contract::StreamEndsClassified,
    Contract::UnknownStreamFramesSurfaced,
    Contract::TransportOptionsApplied,
    Contract::MalformedResponse,
    Contract::ProviderStamped,
    Contract::ValidationEncodingCoverage,
    Contract::UnaryStreamAgree,
    Contract::ToolCallIdsRoundTrip,
    Contract::LengthFinish,
    Contract::StreamEventOrder,
    Contract::RedirectNotFollowed,
];

/// A probe for runs whose contracts never send.
fn never(_: ChatRequest) -> std::future::Ready<ProbeOutcome<ChatResponse>> {
    panic!("a skipped contract sent a probe request")
}

/// `CHAT` less `run`.
fn all_but(run: &[Contract]) -> Vec<Contract> {
    CHAT.iter()
        .copied()
        .filter(|contract| !run.contains(contract))
        .collect()
}

/// An error mapping that follows the rules: the body only in its slot, the
/// retry signals from the headers.
fn honest_mapping(status: u16, headers: &[(&str, &str)], body: String) -> Error {
    let headers: allama_http::Headers = headers.iter().copied().collect();
    let error = match status {
        429 => Error::rate_limited(format!("HTTP {status}")),
        400..=499 => Error::invalid_request(format!("HTTP {status}")),
        _ => Error::provider(format!("HTTP {status}")),
    };
    error
        .authored()
        .with_context(allama_http::error_context(status, &headers))
        .with_body(body)
}

/// Quotes the body in its authored message, and drops the headers.
fn leaky_mapping(status: u16, _: &[(&str, &str)], body: String) -> Error {
    Error::invalid_request(format!("HTTP {status}: {body}"))
        .authored()
        .with_status(status)
}

#[tokio::test]
#[should_panic(expected = "StrictNotHonouredWarns needs `.strict_not_honoured(models)`")]
async fn missing_inputs_are_named_before_anything_runs() {
    assert_chat_adapter(
        &MockProvider::empty(),
        &AdapterUnderTest::new(["m"], "mock", never, never),
    )
    .await;
}

/// A model that answers tools and structured output `Unsupported` has no
/// strict mode to honour: the contract runs for it without being listed.
#[tokio::test]
async fn a_model_without_tools_or_structured_output_needs_no_strict_list() {
    let provider = || {
        MockProvider::with_text("ok")
            .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Unsupported)
            .with_chat_capability(
                ChatCapability::StructuredOutput,
                CapabilitySupport::Unsupported,
            )
    };
    let probe = move |request: ChatRequest| async move {
        ProbeOutcome::new(provider().chat(&request).await, Vec::new())
    };
    let ran = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counted = ran.clone();
    let counting = move |request: ChatRequest| {
        counted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        probe(request)
    };
    assert_chat_adapter(
        &provider(),
        &AdapterUnderTest::new(["m"], "mock", counting, probe)
            .skip(all_but(&[Contract::StrictNotHonouredWarns])),
    )
    .await;
    assert!(
        ran.load(std::sync::atomic::Ordering::Relaxed) > 0,
        "the contract ran"
    );
}

#[tokio::test]
#[should_panic(expected = "StreamErrorsClassified needs `.stream_replay(..)`")]
async fn every_missing_input_is_named() {
    assert_chat_adapter(
        &MockProvider::empty(),
        &AdapterUnderTest::new(["m"], "mock", never, never)
            .error_mapping(honest_mapping)
            .skip(all_but(&[Contract::StreamErrorsClassified])),
    )
    .await;
}

#[tokio::test]
#[should_panic(
    expected = "UnknownStreamFramesSurfaced needs `.sse_captures(..)` or `.stream_captures(..)`"
)]
async fn the_vendor_frame_contract_needs_captures_in_some_framing() {
    assert_chat_adapter(
        &MockProvider::empty(),
        &AdapterUnderTest::new(["m"], "mock", never, never)
            .stream_replay(|_| async { Ok(futures_util::stream::empty()) })
            .skip(all_but(&[Contract::UnknownStreamFramesSurfaced])),
    )
    .await;
}

#[tokio::test]
async fn skipped_contracts_need_nothing_and_do_not_run() {
    assert_chat_adapter(
        &MockProvider::empty(),
        &AdapterUnderTest::new(["m"], "mock", never, never).skip(CHAT.iter().copied()),
    )
    .await;
}

/// An error envelope for the kinds the contracts require, around upstream
/// text.
fn envelope(kind: ErrorKindProbe, upstream: &str) -> Option<String> {
    match kind {
        ErrorKindProbe::RateLimited | ErrorKindProbe::InvalidRequest => {
            Some(serde_json::json!({ "error": { "message": upstream } }).to_string())
        }
        _ => None,
    }
}

#[tokio::test]
#[should_panic(expected = "ErrorLogExcludesBody needs `.error_envelope(..)`")]
async fn the_redaction_contract_needs_the_envelope() {
    assert_chat_adapter(
        &MockProvider::empty(),
        &AdapterUnderTest::new(["m"], "mock", never, never)
            .error_mapping(honest_mapping)
            .skip(all_but(&[Contract::ErrorLogExcludesBody])),
    )
    .await;
}

#[tokio::test]
async fn the_error_mapping_contracts_run_on_their_inputs() {
    assert_chat_adapter(
        &MockProvider::empty(),
        &AdapterUnderTest::new(["m"], "mock", never, never)
            .error_mapping(honest_mapping)
            .error_envelope(envelope)
            .skip(all_but(&[
                Contract::ErrorKindsMapped,
                Contract::ErrorLogExcludesBody,
                Contract::RetrySignalsSurfaced,
            ])),
    )
    .await;
}

#[tokio::test]
#[should_panic(expected = "2 contract(s) failed")]
async fn every_failure_is_reported() {
    assert_chat_adapter(
        &MockProvider::empty(),
        &AdapterUnderTest::new(["m"], "mock", never, never)
            .error_mapping(leaky_mapping)
            .error_envelope(envelope)
            .skip(all_but(&[
                Contract::ErrorLogExcludesBody,
                Contract::RetrySignalsSurfaced,
            ])),
    )
    .await;
}

fn never_embed(_: EmbeddingRequest) -> std::future::Ready<ProbeOutcome<EmbeddingResponse>> {
    panic!("a skipped contract sent a probe request")
}

#[tokio::test]
#[should_panic(expected = "EmbeddingCount needs `.answering(order, answer)`")]
async fn the_embedding_aggregate_names_missing_inputs() {
    assert_embedding_adapter(
        &MockEmbeddingProvider::empty(),
        &EmbeddingAdapterUnderTest::new(["m"], "mock", never_embed),
    )
    .await;
}

#[tokio::test]
async fn the_embedding_aggregate_runs_the_error_contracts() {
    assert_embedding_adapter(
        &MockEmbeddingProvider::empty(),
        &EmbeddingAdapterUnderTest::new(["m"], "mock", never_embed)
            .error_mapping(honest_mapping)
            .error_envelope(envelope)
            .skip([
                Contract::ValidationAgrees,
                Contract::UnsupportedCapabilitiesRefused,
                Contract::InputProviderDataChecked,
                Contract::BatchLimitEnforced,
                Contract::InputTypeNotHonouredWarns,
                Contract::MultiPartInputRefused,
                Contract::EmbeddingResponseChecked,
                Contract::EmbeddingCount,
                Contract::ErrorKindsMapped,
                Contract::TransportOptionsApplied,
            ]),
    )
    .await;
}

fn never_decide(_: DecisionRequest) -> std::future::Ready<ProbeOutcome<DecisionResponse>> {
    panic!("a skipped contract sent a probe request")
}

#[tokio::test]
#[should_panic(expected = "TransportOptionsApplied needs `.transport_options(slot, send)`")]
async fn the_decision_aggregate_names_missing_inputs() {
    assert_decision_adapter(
        &allama::mock::MockDecisionProvider::empty(),
        &DecisionAdapterUnderTest::new(["m"], never_decide).error_mapping(honest_mapping),
    )
    .await;
}

#[tokio::test]
async fn a_known_gap_contract_may_fail() {
    assert_chat_adapter(
        &MockProvider::empty(),
        &AdapterUnderTest::new(["m"], "mock", never, never)
            .error_mapping(leaky_mapping)
            .error_envelope(envelope)
            .known_gaps([Contract::ErrorLogExcludesBody])
            .skip(all_but(&[Contract::ErrorLogExcludesBody])),
    )
    .await;
}

#[tokio::test]
#[should_panic(expected = "ErrorLogExcludesBody is listed as a known gap but passes; remove it")]
async fn a_known_gap_contract_that_passes_fails() {
    assert_chat_adapter(
        &MockProvider::empty(),
        &AdapterUnderTest::new(["m"], "mock", never, never)
            .error_mapping(honest_mapping)
            .error_envelope(envelope)
            .known_gaps([Contract::ErrorLogExcludesBody])
            .skip(all_but(&[Contract::ErrorLogExcludesBody])),
    )
    .await;
}

#[tokio::test]
#[should_panic(expected = "UnaryStreamAgree needs `.response_captures(..)`")]
async fn the_decoding_contracts_need_captures() {
    assert_chat_adapter(
        &MockProvider::empty(),
        &AdapterUnderTest::new(["m"], "mock", never, never)
            .stream_replay(|_| async { Ok(futures_util::stream::empty()) })
            .unary_replay(|_| async { Ok(ChatResponse::default()) })
            .skip(all_but(&[Contract::UnaryStreamAgree])),
    )
    .await;
}

/// A model that cannot call tools needs no tool-call capture.
#[tokio::test]
async fn an_adapter_without_tools_needs_no_tool_call_capture() {
    let provider = MockProvider::empty()
        .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Unsupported);
    assert_chat_adapter(
        &provider,
        &AdapterUnderTest::new(["m"], "mock", never, never)
            .skip(all_but(&[Contract::ToolCallIdsRoundTrip])),
    )
    .await;
}

#[tokio::test]
#[should_panic(expected = "ToolCallIdsRoundTrip needs `.tool_call_capture(..)`")]
async fn an_adapter_with_tools_needs_a_tool_call_capture() {
    assert_chat_adapter(
        &MockProvider::empty(),
        &AdapterUnderTest::new(["m"], "mock", never, never)
            .stream_replay(|_| async { Ok(futures_util::stream::empty()) })
            .unary_replay(|_| async { Ok(ChatResponse::default()) })
            .skip(all_but(&[Contract::ToolCallIdsRoundTrip])),
    )
    .await;
}

/// The default validation checks capabilities only.
#[tokio::test]
#[should_panic(expected = "validation must be a dry run of the encoding")]
async fn capabilities_only_validation_fails() {
    assert_chat_adapter(
        &MockProvider::empty(),
        &AdapterUnderTest::new(["m"], "mock", never, never)
            .skip(all_but(&[Contract::ValidationEncodingCoverage])),
    )
    .await;
}

/// A provisional contract whose input is missing is not run, and the
/// aggregate passes; a required one fails before anything runs.
#[tokio::test]
async fn a_provisional_contract_without_its_input_is_not_run() {
    let skip = [];
    let mut run = Runner::new("assert_chat_adapter", "mock", &skip, &[]);
    run.provisional = &[Contract::LengthFinish];
    run.needs(Contract::LengthFinish, true, "`.length_capture(..)`");
    run.check_inputs();
    run.contract(Contract::LengthFinish, async {
        panic!("a contract that is not run never runs")
    })
    .await;
    run.finish();
}

#[test]
#[should_panic(expected = "LengthFinish needs `.length_capture(..)`")]
fn a_required_contract_without_its_input_fails() {
    let skip = [];
    let mut run = Runner::new("assert_chat_adapter", "mock", &skip, &[]);
    run.needs(Contract::LengthFinish, true, "`.length_capture(..)`");
    run.check_inputs();
}

/// A provisional contract that runs and fails prints its failure, and
/// the aggregate passes.
#[tokio::test]
async fn a_failing_provisional_contract_does_not_fail_the_suite() {
    let skip = [];
    let mut run = Runner::new("assert_chat_adapter", "mock", &skip, &[]);
    run.provisional = &[Contract::LengthFinish];
    run.check_inputs();
    run.contract(Contract::LengthFinish, async {
        panic!("a provisional failure is printed")
    })
    .await;
    run.finish();
}

/// No contract is provisional in the first release;
/// `assert_vendor_fields_kept` stays outside the aggregate (ADR 0017).
#[test]
fn no_contract_is_provisional_in_the_first_release() {
    for contract in CHAT {
        assert!(!contract.is_provisional(), "{contract}");
    }
}
