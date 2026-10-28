//! Aggregate conformance: every contract an adapter must pass, from one call.
//!
//! [`assert_chat_adapter`] runs every probe-based chat contract of this
//! crate against an adapter: capability honesty and refusals, validation
//! agreement, request fields and what never reaches the wire, strict mode,
//! error kinds and redaction, retry signals, in-band stream errors and
//! stream ends, transport options, and a malformed response.
//! [`assert_embedding_adapter`], [`assert_decision_adapter`] and
//! [`assert_extraction_adapter`] are the same for embeddings, decisions and
//! extractions. The set grows with the crate: a contract
//! added later runs in every adapter's suite without a change to it.
//!
//! What a contract needs from the adapter is an input of the
//! `*UnderTest` value: the probes always, and the optional ones (error
//! mapping, envelope and stream-body builders, a credential slot and send
//! closures, ...) through setters. When a contract's input is missing, the
//! aggregate **fails** and names it, unless the contract is listed in
//! [`skip`](AdapterUnderTest::skip): a contract that does not apply (no wire
//! stream, no HTTP transport) is skipped on purpose, never silently.
//!
//! The per-contract functions stay public for targeted use: a test that
//! pins one behavior, or a contract an adapter runs for one model only.
//!
//! # How the set grows
//!
//! Contracts are added only in minor releases (`0.x` to `0.(x+1)`), never
//! in a patch release, so `cargo update` within a minor release never turns
//! an adapter's suite red (ADR 0017). A contract may join an aggregate as
//! *provisional* ([`Contract::is_provisional`]): it never fails a suite.
//! When its input is missing, the aggregate prints "not run: needs ...";
//! when it runs and fails, the aggregate prints the failure and passes. A
//! later minor release may make it required;
//! nothing promises which one. A contract that needs an input older suites
//! do not give is always provisional at first; one that needs no new input
//! may run as required at once.
//!
//! No contract is provisional in 0.1. A contract outside the aggregates,
//! such as
//! [`assert_vendor_fields_kept`](crate::decoding::assert_vendor_fields_kept),
//! may be provisional too: you run it yourself, and what it checks may
//! change in any minor release.
//!
//! ```no_run
//! use allama::{ChatProvider, ChatRequest, ChatResponse, ChatStreamExt};
//! use allama_conformance::adapter::{AdapterUnderTest, Contract, assert_chat_adapter};
//! use allama_conformance::probe::ProbeOutcome;
//! # struct Transport;
//! # impl Transport { fn bodies(&self) -> Vec<serde_json::Value> { Vec::new() } }
//! # fn provider_answering_ok() -> (allama::mock::MockStreamingProvider, Transport) {
//! #     (allama::mock::MockStreamingProvider::with_text("ok"), Transport)
//! # }
//! # fn map_http_error(_: u16, _: &[(&str, &str)], _: String) -> allama::Error {
//! #     allama::Error::provider("HTTP")
//! # }
//!
//! async fn probe(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
//!     let (provider, transport) = provider_answering_ok();
//!     let result = provider.chat(&request).await;
//!     ProbeOutcome::new(result, transport.bodies())
//! }
//!
//! async fn stream_probe(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
//!     let (provider, transport) = provider_answering_ok();
//!     let result = match provider.chat_stream(&request).await {
//!         Ok(stream) => stream.collect_response().await,
//!         Err(err) => Err(err),
//!     };
//!     ProbeOutcome::new(result, transport.bodies())
//! }
//!
//! # async fn example() {
//! let (provider, _) = provider_answering_ok();
//! assert_chat_adapter(
//!     &provider,
//!     &AdapterUnderTest::new(["my-model"], "my-namespace", probe, stream_probe)
//!         .error_mapping(map_http_error)
//!         // This adapter has no wire format and no HTTP transport.
//!         .skip([
//!             Contract::StreamErrorsClassified,
//!             Contract::StreamEndsClassified,
//!             Contract::UnknownStreamFramesSurfaced,
//!             Contract::TransportOptionsApplied,
//!             Contract::MalformedResponse,
//!             Contract::UnaryStreamAgree,
//!             Contract::ToolCallIdsRoundTrip,
//!             Contract::LengthFinish,
//!             Contract::StreamEventOrder,
//!             Contract::RedirectNotFollowed,
//!         ]),
//! )
//! .await;
//! # }
//! ```

use std::fmt;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::time::Duration;

use allama::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, ChatStream,
    EmbeddingCapability, Error, RequestOptions, StreamEvent, ValidationCoverage,
};
use allama_http::CredentialSlot;
use futures_core::Stream;
use futures_util::FutureExt;

pub use decision::{DecisionAdapterUnderTest, assert_decision_adapter};
pub use embedding::{AnsweredVector, EmbeddingAdapterUnderTest, assert_embedding_adapter};
pub use extraction::{ExtractionAdapterUnderTest, assert_extraction_adapter};

use crate::TransportOutcome;
use crate::capabilities::{
    HonestyCheck, HonestyUnderTest, UNKNOWN_MODEL, assert_capabilities_honest,
    assert_stream_usage_matches_capability, assert_unsupported_chat_capabilities_refused,
};
use crate::decoding::ResponseCapture;
use crate::diagnostics::elide_payloads;
use crate::probe::{
    Probe, ProbeOutcome, assert_app_data_never_sent, assert_empty_turns_skipped,
    assert_other_provider_data_gated, assert_request_fields_accounted,
    assert_strict_not_honoured_warns, assert_unreplayable_reasoning_dropped,
};
use crate::stream_end::StreamEnds;
use crate::stream_errors::ErrorKindProbe;
use crate::stream_frames::StreamFraming;

mod decision;
mod embedding;
mod extraction;

/// Assert that a chat adapter passes every probe-based chat contract.
///
/// `provider` answers capabilities and validation, configured as the
/// probes' providers are; `under_test` holds the probes and the other
/// inputs (see [`AdapterUnderTest`]). The contracts, each named by a
/// [`Contract`]:
///
/// | Contract | Runs | Needs |
/// |---|---|---|
/// | [`CapabilitiesHonest`](Contract::CapabilitiesHonest) | the models and [`UNKNOWN_MODEL`] | |
/// | [`UnsupportedCapabilitiesRefused`](Contract::UnsupportedCapabilitiesRefused) | each model and [`UNKNOWN_MODEL`] | |
/// | [`StreamUsageMatchesCapability`](Contract::StreamUsageMatchesCapability) | each model | |
/// | [`ValidationAgrees`](Contract::ValidationAgrees) | [`drift_requests`](crate::drift_requests) of each model and [`UNKNOWN_MODEL`], and [`validation_requests`](AdapterUnderTest::validation_requests) | |
/// | [`RequestFieldsAccounted`](Contract::RequestFieldsAccounted) | each model | |
/// | [`AppDataNeverSent`](Contract::AppDataNeverSent) | each model | |
/// | [`OtherProviderDataGated`](Contract::OtherProviderDataGated) | the first model, the namespace | |
/// | [`EmptyTurnsSkipped`](Contract::EmptyTurnsSkipped) | the first model | |
/// | [`UnreplayableReasoningDropped`](Contract::UnreplayableReasoningDropped) | each model that answers `ReasoningReplay` `Unsupported` | one such model |
/// | [`StrictNotHonouredWarns`](Contract::StrictNotHonouredWarns) | each model without strict mode, and each that answers `ToolCalls` and `StructuredOutput` `Unsupported` | [`strict_not_honoured`](AdapterUnderTest::strict_not_honoured), unless every model answers both `Unsupported` |
/// | [`ErrorKindsMapped`](Contract::ErrorKindsMapped) | | [`error_mapping`](AdapterUnderTest::error_mapping), [`error_envelope`](AdapterUnderTest::error_envelope) |
/// | [`ErrorLogExcludesBody`](Contract::ErrorLogExcludesBody) | | [`error_mapping`](AdapterUnderTest::error_mapping), [`error_envelope`](AdapterUnderTest::error_envelope) |
/// | [`RetrySignalsSurfaced`](Contract::RetrySignalsSurfaced) | | [`error_mapping`](AdapterUnderTest::error_mapping) |
/// | [`StreamErrorsClassified`](Contract::StreamErrorsClassified) | | [`error_mapping`](AdapterUnderTest::error_mapping), [`error_envelope`](AdapterUnderTest::error_envelope), [`stream_body`](AdapterUnderTest::stream_body), [`stream_replay`](AdapterUnderTest::stream_replay) |
/// | [`StreamEndsClassified`](Contract::StreamEndsClassified) | | [`stream_ends`](AdapterUnderTest::stream_ends), [`stream_replay`](AdapterUnderTest::stream_replay) |
/// | [`UnknownStreamFramesSurfaced`](Contract::UnknownStreamFramesSurfaced) | each stream capture | [`sse_captures`](AdapterUnderTest::sse_captures) or [`stream_captures`](AdapterUnderTest::stream_captures), [`stream_replay`](AdapterUnderTest::stream_replay) |
/// | [`TransportOptionsApplied`](Contract::TransportOptionsApplied) | `chat` and `chat_stream` | [`transport_options`](AdapterUnderTest::transport_options) |
/// | [`MalformedResponse`](Contract::MalformedResponse) | | [`malformed_response`](AdapterUnderTest::malformed_response) |
/// | [`ProviderStamped`](Contract::ProviderStamped) | the first model | |
/// | [`ValidationEncodingCoverage`](Contract::ValidationEncodingCoverage) | each model and [`UNKNOWN_MODEL`] | |
/// | [`UnaryStreamAgree`](Contract::UnaryStreamAgree) | each capture | [`unary_replay`](AdapterUnderTest::unary_replay), [`stream_replay`](AdapterUnderTest::stream_replay), [`response_captures`](AdapterUnderTest::response_captures) |
/// | [`ToolCallIdsRoundTrip`](Contract::ToolCallIdsRoundTrip) | the first model with tools | [`unary_replay`](AdapterUnderTest::unary_replay), [`stream_replay`](AdapterUnderTest::stream_replay), [`tool_call_capture`](AdapterUnderTest::tool_call_capture), unless every model answers `ToolCalls` `Unsupported` |
/// | [`LengthFinish`](Contract::LengthFinish) | | [`unary_replay`](AdapterUnderTest::unary_replay), [`stream_replay`](AdapterUnderTest::stream_replay), [`length_capture`](AdapterUnderTest::length_capture) |
/// | [`StreamEventOrder`](Contract::StreamEventOrder) | each stream capture and each capture's stream | [`stream_replay`](AdapterUnderTest::stream_replay), and a stream capture or a capture |
/// | [`RedirectNotFollowed`](Contract::RedirectNotFollowed) | `chat` and `chat_stream` | `redirect_not_followed` (feature `server`) |
///
/// [`known_gaps`](AdapterUnderTest::known_gaps) reach the contracts that
/// take them, and apply to every model; a [`KnownGap::Contract`] marks a
/// whole contract.
///
/// # Panics
///
/// Before running anything, when a contract that is not skipped lacks an
/// input, naming each (a [provisional](Contract::is_provisional) contract
/// that lacks one is reported as not run instead). Then, after running
/// every contract, when any failed, with each failure's message (a
/// provisional contract's failure is printed instead).
pub async fn assert_chat_adapter<P, F, Fut, G, GFut>(
    provider: &P,
    under_test: &AdapterUnderTest<F, G>,
) where
    P: ChatProvider,
    F: Fn(ChatRequest) -> Fut,
    Fut: Future<Output = ProbeOutcome<ChatResponse>>,
    G: Fn(ChatRequest) -> GFut,
    GFut: Future<Output = ProbeOutcome<ChatResponse>>,
{
    let t = under_test;
    let mut run = Runner::new(
        "assert_chat_adapter",
        provider.provider_name(),
        &t.skip,
        &t.known_gaps,
    );
    let first = t.models.first().map_or(UNKNOWN_MODEL, String::as_str);
    let probe = |request| (t.probe)(request);
    let stream_probe = |request| (t.stream_probe)(request);
    let with_unknown: Vec<&str> = t
        .models
        .iter()
        .map(String::as_str)
        .chain((!t.models.iter().any(|m| m == UNKNOWN_MODEL)).then_some(UNKNOWN_MODEL))
        .collect();
    let unreplayable: Vec<&str> = t
        .models
        .iter()
        .map(String::as_str)
        .filter(|model| {
            provider.chat_capability(model, ChatCapability::ReasoningReplay)
                == CapabilitySupport::Unsupported
        })
        .collect();

    run.needs(
        Contract::UnreplayableReasoningDropped,
        unreplayable.is_empty(),
        "a listed model that answers ReasoningReplay Unsupported",
    );
    // A model that answers both tools and structured output `Unsupported`
    // has no strict mode to honour: the contract runs for it without being
    // listed (its strict requests are refused before sending).
    let strict_free: Vec<&str> = t
        .models
        .iter()
        .map(String::as_str)
        .filter(|model| {
            [ChatCapability::ToolCalls, ChatCapability::StructuredOutput]
                .into_iter()
                .all(|c| provider.chat_capability(model, c) == CapabilitySupport::Unsupported)
        })
        .collect();
    run.needs(
        Contract::StrictNotHonouredWarns,
        t.strict_not_honoured.is_none() && strict_free.len() < t.models.len(),
        "`.strict_not_honoured(models)`",
    );
    t.errors.require(&mut run);
    let replay = t.stream_replay.is_none();
    run.needs(
        Contract::StreamErrorsClassified,
        t.errors.mapping.is_none(),
        "`.error_mapping(..)`",
    );
    run.needs(
        Contract::StreamErrorsClassified,
        t.errors.envelope.is_none(),
        "`.error_envelope(..)`",
    );
    run.needs(
        Contract::StreamErrorsClassified,
        t.stream_body.is_none(),
        "`.stream_body(..)`",
    );
    run.needs(
        Contract::StreamErrorsClassified,
        replay,
        "`.stream_replay(..)`",
    );
    run.needs(
        Contract::StreamEndsClassified,
        t.stream_ends.is_none(),
        "`.stream_ends(..)`",
    );
    run.needs(
        Contract::StreamEndsClassified,
        replay,
        "`.stream_replay(..)`",
    );
    run.needs(
        Contract::UnknownStreamFramesSurfaced,
        t.stream_captures.is_empty(),
        "`.sse_captures(..)` or `.stream_captures(..)`",
    );
    run.needs(
        Contract::UnknownStreamFramesSurfaced,
        replay,
        "`.stream_replay(..)`",
    );
    run.needs(
        Contract::TransportOptionsApplied,
        t.transport.is_none(),
        "`.transport_options(slot, send, stream_send)`",
    );
    run.needs(
        Contract::MalformedResponse,
        t.malformed.is_none(),
        "`.malformed_response(send, request_id)`",
    );
    // The first model that can call tools, for the tool-call round trip; an
    // adapter none of whose models can needs no tool-call capture.
    let with_tools = t.models.iter().map(String::as_str).find(|model| {
        provider.chat_capability(model, ChatCapability::ToolCalls) != CapabilitySupport::Unsupported
    });
    for contract in [
        Contract::UnaryStreamAgree,
        Contract::ToolCallIdsRoundTrip,
        Contract::LengthFinish,
    ] {
        if contract != Contract::ToolCallIdsRoundTrip || with_tools.is_some() {
            run.needs(contract, t.unary_replay.is_none(), "`.unary_replay(..)`");
            run.needs(contract, replay, "`.stream_replay(..)`");
        }
    }
    run.needs(
        Contract::UnaryStreamAgree,
        t.response_captures.is_empty(),
        "`.response_captures(..)`",
    );
    run.needs(
        Contract::ToolCallIdsRoundTrip,
        with_tools.is_some() && t.tool_call_capture.is_none(),
        "`.tool_call_capture(..)`",
    );
    run.needs(
        Contract::LengthFinish,
        t.length_capture.is_none(),
        "`.length_capture(..)`",
    );
    let order_bodies: Vec<(String, &str)> = t
        .stream_captures
        .iter()
        .flat_map(|(_, captures)| captures.iter())
        .enumerate()
        .map(|(n, body)| (format!("stream capture {n}"), body.as_str()))
        .chain(
            t.captures()
                .map(|(label, capture)| (label, capture.stream.as_str())),
        )
        .collect();
    run.needs(Contract::StreamEventOrder, replay, "`.stream_replay(..)`");
    run.needs(
        Contract::StreamEventOrder,
        order_bodies.is_empty(),
        "a stream capture (`.sse_captures(..)`, `.stream_captures(..)`) or a capture \
         (`.response_captures(..)`, ...)",
    );
    run.needs(
        Contract::RedirectNotFollowed,
        t.redirect.is_none(),
        "`.redirect_not_followed(send)` (feature `server`)",
    );
    run.check_inputs();

    let honesty_gaps = t.known_gaps.iter().filter_map(|gap| match gap {
        KnownGap::Honesty(check) => Some(*check),
        _ => None,
    });
    let honesty = HonestyUnderTest::new(t.models.iter().cloned(), probe, stream_probe)
        .known_gaps(honesty_gaps);
    run.contract(Contract::CapabilitiesHonest, async {
        assert_capabilities_honest(provider, &honesty).await;
    })
    .await;
    let capability_gaps: Vec<ChatCapability> = t
        .known_gaps
        .iter()
        .filter_map(|gap| match gap {
            KnownGap::Capability(capability) => Some(*capability),
            _ => None,
        })
        .collect();
    run.contract(Contract::UnsupportedCapabilitiesRefused, async {
        for model in &with_unknown {
            assert_unsupported_chat_capabilities_refused(
                provider,
                model,
                &capability_gaps,
                probe,
                stream_probe,
            )
            .await;
        }
    })
    .await;
    run.contract(Contract::StreamUsageMatchesCapability, async {
        for model in &t.models {
            assert_stream_usage_matches_capability(provider, model, stream_probe).await;
        }
    })
    .await;
    run.contract(Contract::ValidationAgrees, async {
        let mut requests: Vec<ChatRequest> = with_unknown
            .iter()
            .flat_map(|model| crate::drift_requests(model))
            .collect();
        requests.extend(t.validation_requests.iter().cloned());
        crate::assert_validation_agrees(provider, probe, stream_probe, &requests).await;
    })
    .await;
    let field_gaps: Vec<Probe> = t
        .known_gaps
        .iter()
        .filter_map(|gap| match gap {
            KnownGap::Field(probe) => Some(*probe),
            _ => None,
        })
        .collect();
    run.contract(Contract::RequestFieldsAccounted, async {
        for model in &t.models {
            assert_request_fields_accounted(provider, model, &field_gaps, probe, stream_probe)
                .await;
        }
    })
    .await;
    run.contract(Contract::AppDataNeverSent, async {
        for model in &t.models {
            assert_app_data_never_sent(provider, model, probe, stream_probe).await;
        }
    })
    .await;
    run.contract(Contract::OtherProviderDataGated, async {
        assert_other_provider_data_gated(first, &t.namespace, probe, stream_probe).await;
    })
    .await;
    run.contract(Contract::EmptyTurnsSkipped, async {
        assert_empty_turns_skipped(first, probe, stream_probe).await;
    })
    .await;
    run.contract(Contract::UnreplayableReasoningDropped, async {
        for model in &unreplayable {
            assert_unreplayable_reasoning_dropped(provider, model, probe, stream_probe).await;
        }
    })
    .await;
    if t.strict_not_honoured.is_some() || !strict_free.is_empty() {
        let mut models: Vec<&str> = t
            .strict_not_honoured
            .iter()
            .flatten()
            .map(String::as_str)
            .collect();
        for model in &strict_free {
            if !models.contains(model) {
                models.push(model);
            }
        }
        run.contract(Contract::StrictNotHonouredWarns, async {
            for model in models {
                assert_strict_not_honoured_warns(model, probe, stream_probe).await;
            }
        })
        .await;
    }
    t.errors.run(&mut run).await;
    if let (Some(envelope), Some(stream_body), Some(map), Some(replay)) = (
        &t.errors.envelope,
        &t.stream_body,
        &t.errors.mapping,
        &t.stream_replay,
    ) {
        run.contract(Contract::StreamErrorsClassified, async {
            crate::assert_stream_errors_classified(
                |kind| envelope(kind, ""),
                stream_body,
                |status, headers, body| map(status, headers, body),
                replay,
            )
            .await;
        })
        .await;
    }
    if let (Some(ends), Some(replay)) = (&t.stream_ends, &t.stream_replay) {
        run.contract(Contract::StreamEndsClassified, async {
            crate::assert_stream_ends_classified(ends.clone(), replay).await;
        })
        .await;
    }
    if let Some(replay) = &t.stream_replay {
        run.contract(Contract::UnknownStreamFramesSurfaced, async {
            for (framing, captures) in &t.stream_captures {
                for capture in captures {
                    crate::stream_frames::assert_unknown_stream_frames_surfaced(
                        capture, framing, replay,
                    )
                    .await;
                }
            }
        })
        .await;
    }
    if let Some(transport) = &t.transport {
        run.contract(Contract::TransportOptionsApplied, async {
            crate::assert_transport_options_applied(&transport.slot, |options| {
                (transport.send)(options)
            })
            .await;
            crate::assert_transport_options_applied(&transport.slot, |options| {
                (transport.stream_send)(options)
            })
            .await;
        })
        .await;
    }
    if let Some(malformed) = &t.malformed {
        run.contract(Contract::MalformedResponse, async {
            let result = (malformed.send)(crate::contract::MALFORMED_BODY.to_owned()).await;
            crate::contract::check_malformed_response(
                provider.provider_name(),
                result,
                malformed.request_id.as_deref(),
            );
        })
        .await;
    }
    run.contract(Contract::ProviderStamped, async {
        crate::decoding::assert_provider_stamped(
            provider.provider_name(),
            first,
            probe,
            stream_probe,
        )
        .await;
    })
    .await;
    run.contract(Contract::ValidationEncodingCoverage, async {
        for model in &with_unknown {
            let validation = provider
                .validate_chat_request(&ChatRequest::new(*model).user("Hi"))
                .unwrap_or_else(|err| {
                    panic!("validation_encoding_coverage: `{model}`: validation failed: {err:?}")
                });
            assert_eq!(
                validation.coverage(),
                ValidationCoverage::Encoding,
                "validation_encoding_coverage: `{model}`: validation must be a dry run of the \
                 encoding (`Prepared` in `validate_chat_request`), or the encoder refuses what \
                 validation never reports"
            );
        }
    })
    .await;
    if let (Some(unary), Some(stream)) = (&t.unary_replay, &t.stream_replay) {
        let unary = |body| unary(body);
        let stream = |body| stream(body);
        if !t.response_captures.is_empty() {
            run.contract(Contract::UnaryStreamAgree, async {
                for (_, capture) in t.captures() {
                    crate::decoding::assert_unary_stream_agree(capture, unary, stream).await;
                }
            })
            .await;
        }
        if let (Some(model), Some(capture)) = (with_tools, &t.tool_call_capture) {
            run.contract(Contract::ToolCallIdsRoundTrip, async {
                crate::decoding::assert_tool_call_ids_round_trip(
                    model, capture, unary, stream, probe,
                )
                .await;
            })
            .await;
        }
        if let Some(capture) = &t.length_capture {
            run.contract(Contract::LengthFinish, async {
                crate::decoding::assert_length_finish(capture, unary, stream).await;
            })
            .await;
        }
    }
    if let Some(replay) = &t.stream_replay {
        run.contract(Contract::StreamEventOrder, async {
            for (label, body) in &order_bodies {
                crate::stream_order::assert_stream_order_kept(label, body, replay).await;
            }
        })
        .await;
    }
    if let Some(redirect) = &t.redirect {
        run.contract(Contract::RedirectNotFollowed, async {
            redirect.run().await;
        })
        .await;
    }
    run.finish();
}

/// One contract of the aggregates, named after its function, for
/// [`skip`](AdapterUnderTest::skip) lists and failure reports.
///
/// A contract that does not apply to an adapter (no wire stream, no HTTP
/// transport) is skipped by listing it; the aggregate fails when an input
/// a contract needs is missing and the contract is not listed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Contract {
    /// [`assert_capabilities_honest`].
    CapabilitiesHonest,
    /// [`assert_unsupported_chat_capabilities_refused`], or its embedding
    /// variant.
    UnsupportedCapabilitiesRefused,
    /// [`assert_stream_usage_matches_capability`].
    StreamUsageMatchesCapability,
    /// [`assert_validation_agrees`](crate::assert_validation_agrees), or its
    /// embedding or decision variant.
    ValidationAgrees,
    /// [`assert_request_fields_accounted`].
    RequestFieldsAccounted,
    /// [`assert_app_data_never_sent`].
    AppDataNeverSent,
    /// [`assert_other_provider_data_gated`].
    OtherProviderDataGated,
    /// [`assert_empty_turns_skipped`].
    EmptyTurnsSkipped,
    /// [`assert_unreplayable_reasoning_dropped`].
    UnreplayableReasoningDropped,
    /// [`assert_strict_not_honoured_warns`].
    StrictNotHonouredWarns,
    /// [`assert_error_kinds_mapped`](crate::assert_error_kinds_mapped).
    ErrorKindsMapped,
    /// [`assert_error_log_excludes_body`](crate::assert_error_log_excludes_body).
    ErrorLogExcludesBody,
    /// [`assert_retry_signals_surfaced`](crate::assert_retry_signals_surfaced).
    RetrySignalsSurfaced,
    /// [`assert_stream_errors_classified`](crate::assert_stream_errors_classified).
    StreamErrorsClassified,
    /// [`assert_stream_ends_classified`](crate::assert_stream_ends_classified).
    StreamEndsClassified,
    /// [`assert_unknown_stream_frames_surfaced`](crate::stream_frames::assert_unknown_stream_frames_surfaced).
    UnknownStreamFramesSurfaced,
    /// [`assert_transport_options_applied`](crate::assert_transport_options_applied).
    TransportOptionsApplied,
    /// [`assert_malformed_response_contract`](crate::contract::assert_malformed_response_contract).
    MalformedResponse,
    /// [`assert_embedding_response_checked`](crate::probe::assert_embedding_response_checked).
    EmbeddingResponseChecked,
    /// [`assert_embedding_count`](crate::probe::assert_embedding_count).
    EmbeddingCount,
    /// [`assert_batch_limit_enforced`](crate::probe::assert_batch_limit_enforced).
    BatchLimitEnforced,
    /// [`assert_input_provider_data_checked`](crate::probe::assert_input_provider_data_checked).
    InputProviderDataChecked,
    /// [`assert_input_type_not_honoured_warns`](crate::probe::assert_input_type_not_honoured_warns).
    InputTypeNotHonouredWarns,
    /// [`assert_multi_part_input_refused`](crate::probe::assert_multi_part_input_refused).
    MultiPartInputRefused,
    /// [`assert_provider_stamped`](crate::decoding::assert_provider_stamped).
    ProviderStamped,
    /// Validation is a dry run of the encoding: `validate_chat_request`
    /// reports [`ValidationCoverage::Encoding`], for each model and
    /// [`UNKNOWN_MODEL`]. An adapter that can only check capabilities lists
    /// it in [`skip`](AdapterUnderTest::skip).
    ValidationEncodingCoverage,
    /// [`assert_unary_stream_agree`](crate::decoding::assert_unary_stream_agree).
    UnaryStreamAgree,
    /// [`assert_tool_call_ids_round_trip`](crate::decoding::assert_tool_call_ids_round_trip).
    ToolCallIdsRoundTrip,
    /// [`assert_length_finish`](crate::decoding::assert_length_finish).
    LengthFinish,
    /// [`assert_stream_order_kept`](crate::stream_order::assert_stream_order_kept).
    StreamEventOrder,
    /// `redirect::assert_redirect_not_followed` (feature `server`).
    RedirectNotFollowed,
}

/// Contracts provisional in this release: they never fail a suite. A
/// missing input reports "not run", and a failure is printed, until a
/// later minor release makes them required (ADR 0017).
const PROVISIONAL: &[Contract] = &[];

impl Contract {
    /// Whether the contract is provisional in this release: it never fails
    /// a suite. When its input is missing, the aggregate reports it as not
    /// run; when it fails, the aggregate prints the failure and passes (see
    /// [how the set grows](self#how-the-set-grows)). A later minor release
    /// may make it required.
    #[must_use]
    pub fn is_provisional(self) -> bool {
        PROVISIONAL.contains(&self)
    }
}

impl fmt::Display for Contract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// An acknowledged defect of the adapter, for
/// [`known_gaps`](AdapterUnderTest::known_gaps): the contract that finds it
/// skips it, and fails once it is fixed, so a fixed gap leaves the list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum KnownGap {
    /// A check of [`assert_capabilities_honest`].
    Honesty(HonestyCheck),
    /// A probe of [`assert_request_fields_accounted`].
    Field(Probe),
    /// A chat capability of [`assert_unsupported_chat_capabilities_refused`].
    Capability(ChatCapability),
    /// An embedding capability of
    /// [`assert_unsupported_embedding_capabilities_refused`](crate::capabilities::assert_unsupported_embedding_capabilities_refused).
    EmbeddingCapability(EmbeddingCapability),
    /// A whole contract, for a defect no finer gap names: it runs, and its
    /// failure is ignored until it passes.
    Contract(Contract),
}

impl From<Contract> for KnownGap {
    fn from(contract: Contract) -> Self {
        Self::Contract(contract)
    }
}

impl From<HonestyCheck> for KnownGap {
    fn from(check: HonestyCheck) -> Self {
        Self::Honesty(check)
    }
}

impl From<Probe> for KnownGap {
    fn from(probe: Probe) -> Self {
        Self::Field(probe)
    }
}

impl From<ChatCapability> for KnownGap {
    fn from(capability: ChatCapability) -> Self {
        Self::Capability(capability)
    }
}

impl From<EmbeddingCapability> for KnownGap {
    fn from(capability: EmbeddingCapability) -> Self {
        Self::EmbeddingCapability(capability)
    }
}

/// What [`assert_chat_adapter`] needs from a chat adapter.
///
/// Built with [`new`](Self::new) and setters, so new inputs and contracts
/// never break callers. The closures given to setters are kept for the
/// run, so they own what they use (`'static`).
#[non_exhaustive]
pub struct AdapterUnderTest<F, G> {
    models: Vec<String>,
    namespace: String,
    probe: F,
    stream_probe: G,
    known_gaps: Vec<KnownGap>,
    skip: Vec<Contract>,
    validation_requests: Vec<ChatRequest>,
    strict_not_honoured: Option<Vec<String>>,
    errors: ErrorInputs,
    stream_body: Option<Wrap>,
    stream_replay: Option<StreamReplay>,
    stream_ends: Option<StreamEnds>,
    stream_captures: Vec<(StreamFraming, Vec<String>)>,
    transport: Option<ChatTransport>,
    malformed: Option<Malformed>,
    unary_replay: Option<UnaryReplay>,
    response_captures: Vec<ResponseCapture>,
    tool_call_capture: Option<ResponseCapture>,
    length_capture: Option<ResponseCapture>,
    redirect: Option<Redirect>,
}

impl<F, G> AdapterUnderTest<F, G> {
    /// The models to probe (each listed model, and [`UNKNOWN_MODEL`] where a
    /// contract says so; list one of every kind the adapter tells apart:
    /// current, previous generation, reasoning, without tools), the
    /// adapter's provider-data `namespace`, and the chat and streaming
    /// probes (see [probes](crate::probe)): `probe` sends a request through
    /// `chat`, `stream_probe` through `chat_stream`, collecting the stream
    /// into a response.
    pub fn new<M: Into<String>>(
        models: impl IntoIterator<Item = M>,
        namespace: impl Into<String>,
        probe: F,
        stream_probe: G,
    ) -> Self {
        Self {
            models: models.into_iter().map(Into::into).collect(),
            namespace: namespace.into(),
            probe,
            stream_probe,
            known_gaps: Vec::new(),
            skip: Vec::new(),
            validation_requests: Vec::new(),
            strict_not_honoured: None,
            errors: ErrorInputs::default(),
            stream_body: None,
            stream_replay: None,
            stream_ends: None,
            stream_captures: Vec::new(),
            transport: None,
            malformed: None,
            unary_replay: None,
            response_captures: Vec::new(),
            tool_call_capture: None,
            length_capture: None,
            redirect: None,
        }
    }

    /// Every capture with a label, for the contracts that decode them all.
    fn captures(&self) -> impl Iterator<Item = (String, &ResponseCapture)> {
        self.response_captures
            .iter()
            .enumerate()
            .map(|(n, capture)| (format!("response capture {n}"), capture))
            .chain(
                self.tool_call_capture
                    .iter()
                    .map(|capture| ("tool-call capture".to_owned(), capture)),
            )
            .chain(
                self.length_capture
                    .iter()
                    .map(|capture| ("length capture".to_owned(), capture)),
            )
    }

    /// Acknowledged defects ([`KnownGap`], or anything that converts into
    /// one: a [`HonestyCheck`], a [`Probe`], a capability). Each is skipped
    /// by the contract that finds it, for every model, and fails once it
    /// passes. Call it once per kind of gap.
    #[must_use]
    pub fn known_gaps<K: Into<KnownGap>>(mut self, gaps: impl IntoIterator<Item = K>) -> Self {
        self.known_gaps.extend(gaps.into_iter().map(Into::into));
        self
    }

    /// Contracts that do not apply to the adapter, so their inputs are not
    /// needed and they do not run.
    #[must_use]
    pub fn skip(mut self, contracts: impl IntoIterator<Item = Contract>) -> Self {
        self.skip.extend(contracts);
        self
    }

    /// Requests [`ValidationAgrees`](Contract::ValidationAgrees) checks
    /// beyond the drift requests: provider options that change encoding,
    /// caller-set values that interact (a reasoning budget and an output
    /// cap).
    #[must_use]
    pub fn validation_requests(mut self, requests: impl IntoIterator<Item = ChatRequest>) -> Self {
        self.validation_requests.extend(requests);
        self
    }

    /// The models that cannot honour strict mode, for
    /// [`StrictNotHonouredWarns`](Contract::StrictNotHonouredWarns). An
    /// adapter whose every model honours it skips the contract.
    ///
    /// A model that answers both [`ToolCalls`](ChatCapability::ToolCalls)
    /// and [`StructuredOutput`](ChatCapability::StructuredOutput)
    /// `Unsupported` needs no listing: it has no strict mode to honour, so
    /// the contract runs for it anyway (its strict requests must be refused
    /// before sending), and an adapter whose every model answers so, such as
    /// a text-only one, leaves this input out.
    #[must_use]
    pub fn strict_not_honoured<M: Into<String>>(
        mut self,
        models: impl IntoIterator<Item = M>,
    ) -> Self {
        self.strict_not_honoured = Some(models.into_iter().map(Into::into).collect());
        self
    }

    /// The adapter's HTTP error mapping: an HTTP status, the response
    /// headers as `(name, value)` pairs, and the body, to the adapter's
    /// error. For the error kind, redaction, retry signal and in-band
    /// stream error contracts.
    #[must_use]
    pub fn error_mapping(
        mut self,
        map: impl Fn(u16, &[(&str, &str)], String) -> Error + 'static,
    ) -> Self {
        self.errors.mapping = Some(Box::new(map));
        self
    }

    /// The adapter's error envelope for each [`ErrorKindProbe`], as its API
    /// returns it, with its own vendor code and wording, or `None` for a
    /// kind it does not map (including kinds added later).
    ///
    /// The `&str` is upstream text: append it to the wording where your
    /// classification reads the message. The error kind and in-band stream
    /// error contracts pass `""`; the redaction contract
    /// ([`assert_error_log_excludes_body`](crate::assert_error_log_excludes_body))
    /// passes a sentinel through every kind, and fails if it changes the
    /// variant or reaches the redacted log.
    #[must_use]
    pub fn error_envelope(
        mut self,
        envelope: impl Fn(ErrorKindProbe, &str) -> Option<String> + 'static,
    ) -> Self {
        self.errors.envelope = Some(Box::new(envelope));
        self
    }

    /// Writes an error body that carries a retry hint, for an API that
    /// carries it in the body (see
    /// [`assert_retry_signals_surfaced`](crate::assert_retry_signals_surfaced)).
    /// Optional.
    #[must_use]
    pub fn retry_hint_body(mut self, encode: impl Fn(Duration) -> String + 'static) -> Self {
        self.errors.retry_hint_body = Some(Box::new(encode));
        self
    }

    /// Wraps an error envelope as the body of a streaming response that
    /// started (an SSE `error` event, an NDJSON line), for
    /// [`StreamErrorsClassified`](Contract::StreamErrorsClassified).
    #[must_use]
    pub fn stream_body(mut self, wrap: impl Fn(&str) -> String + 'static) -> Self {
        self.stream_body = Some(Box::new(wrap));
        self
    }

    /// Sends a request whose transport answers `200` with the given body,
    /// and returns `chat_stream`'s result: the stream the adapter decodes
    /// from the body. For the in-band stream error, stream end and vendor
    /// frame contracts.
    #[must_use]
    pub fn stream_replay<R, RFut, S>(mut self, replay: R) -> Self
    where
        R: Fn(String) -> RFut + 'static,
        RFut: Future<Output = allama::Result<S>> + 'static,
        S: Stream<Item = allama::Result<StreamEvent>> + Send + 'static,
    {
        self.stream_replay = Some(Box::new(move |body| {
            let replayed = replay(body);
            Box::pin(async move { replayed.await.map(|stream| Box::pin(stream) as ChatStream) })
        }));
        self
    }

    /// The stream bodies [`StreamEndsClassified`](Contract::StreamEndsClassified)
    /// decodes through [`stream_replay`](Self::stream_replay).
    #[must_use]
    pub fn stream_ends(mut self, ends: StreamEnds) -> Self {
        self.stream_ends = Some(ends);
        self
    }

    /// SSE captures of the adapter's streams (ideally recorded), which
    /// [`UnknownStreamFramesSurfaced`](Contract::UnknownStreamFramesSurfaced)
    /// injects vendor frames into and decodes through
    /// [`stream_replay`](Self::stream_replay):
    /// [`stream_captures`](Self::stream_captures) with
    /// [`StreamFraming::sse`].
    #[must_use]
    pub fn sse_captures<S: Into<String>>(self, captures: impl IntoIterator<Item = S>) -> Self {
        self.stream_captures(StreamFraming::sse(), captures)
    }

    /// Captures of the adapter's streams in another wire framing (NDJSON,
    /// ...), which
    /// [`UnknownStreamFramesSurfaced`](Contract::UnknownStreamFramesSurfaced)
    /// injects vendor frames into, as `framing` writes them, and decodes
    /// through [`stream_replay`](Self::stream_replay).
    ///
    /// ```
    /// # use allama_conformance::adapter::AdapterUnderTest;
    /// # use allama_conformance::probe::ProbeOutcome;
    /// use allama_conformance::stream_frames::StreamFraming;
    /// use serde_json::json;
    ///
    /// # async fn probe(_: allama::ChatRequest) -> ProbeOutcome<allama::ChatResponse> { unimplemented!() }
    /// const OK_STREAM: &str = "{\"type\":\"text\",\"delta\":\"Hi\"}\n{\"type\":\"done\"}\n";
    /// // An unknown line type, as Acme names a line's type.
    /// let ndjson = StreamFraming::ndjson(|frame_type, payload| {
    ///     let mut line = payload.clone();
    ///     line["type"] = json!(frame_type);
    ///     line.to_string()
    /// });
    /// let under_test = AdapterUnderTest::new(["acme-1"], "acme", probe, probe)
    ///     .stream_captures(ndjson, [OK_STREAM]);
    /// ```
    #[must_use]
    pub fn stream_captures<S: Into<String>>(
        mut self,
        framing: StreamFraming,
        captures: impl IntoIterator<Item = S>,
    ) -> Self {
        self.stream_captures
            .push((framing, captures.into_iter().map(Into::into).collect()));
        self
    }

    /// The adapter's credential slot, and the calls
    /// [`TransportOptionsApplied`](Contract::TransportOptionsApplied) makes
    /// with transport options attached: `send` through `chat`,
    /// `stream_send` through `chat_stream`, each returning the result and
    /// the HTTP requests sent (see
    /// [`assert_transport_options_applied`](crate::assert_transport_options_applied)).
    #[must_use]
    pub fn transport_options<S, SFut, T, TFut>(
        mut self,
        slot: CredentialSlot,
        send: S,
        stream_send: T,
    ) -> Self
    where
        S: Fn(RequestOptions) -> SFut + 'static,
        SFut: Future<Output = TransportOutcome> + 'static,
        T: Fn(RequestOptions) -> TFut + 'static,
        TFut: Future<Output = TransportOutcome> + 'static,
    {
        self.transport = Some(ChatTransport {
            slot,
            send: boxed_send(send),
            stream_send: boxed_send(stream_send),
        });
        self
    }

    /// Sends a request through `chat` over a transport that answers `200`
    /// with the given body and, when the adapter reads one, `request_id` in
    /// its request id header, for
    /// [`MalformedResponse`](Contract::MalformedResponse) (see
    /// [`assert_malformed_response_contract`](crate::contract::assert_malformed_response_contract)).
    #[must_use]
    pub fn malformed_response<S, SFut>(mut self, send: S, request_id: Option<&str>) -> Self
    where
        S: Fn(String) -> SFut + 'static,
        SFut: Future<Output = allama::Result<ChatResponse>> + 'static,
    {
        self.malformed = Some(Malformed {
            send: Box::new(move |body| Box::pin(send(body))),
            request_id: request_id.map(str::to_owned),
        });
        self
    }
}

impl<F, G> fmt::Debug for AdapterUnderTest<F, G> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AdapterUnderTest")
            .field("models", &self.models)
            .field("namespace", &self.namespace)
            .field("known_gaps", &self.known_gaps)
            .field("skip", &self.skip)
            .field("validation_requests", &self.validation_requests.len())
            .field("response_captures", &self.response_captures.len())
            .finish_non_exhaustive()
    }
}

impl<F, G> AdapterUnderTest<F, G> {
    /// Sends a request through `chat` over a transport that answers `200`
    /// with the given body, and returns the response the adapter decodes:
    /// the unary counterpart of [`stream_replay`](Self::stream_replay), for
    /// the decoding contracts ([`UnaryStreamAgree`](Contract::UnaryStreamAgree),
    /// [`ToolCallIdsRoundTrip`](Contract::ToolCallIdsRoundTrip),
    /// [`LengthFinish`](Contract::LengthFinish)).
    #[must_use]
    pub fn unary_replay<R, RFut>(mut self, replay: R) -> Self
    where
        R: Fn(String) -> RFut + 'static,
        RFut: Future<Output = allama::Result<ChatResponse>> + 'static,
    {
        self.unary_replay = Some(Box::new(move |body| Box::pin(replay(body))));
        self
    }

    /// Captured responses ([`ResponseCapture`]: the `chat` body and the
    /// `chat_stream` body of one response, ideally recorded) that
    /// [`UnaryStreamAgree`](Contract::UnaryStreamAgree) decodes through
    /// both paths and compares. Give at least a plain text response; the
    /// [tool-call](Self::tool_call_capture) and
    /// [length](Self::length_capture) captures are compared too, and every
    /// capture's stream is checked by
    /// [`StreamEventOrder`](Contract::StreamEventOrder).
    #[must_use]
    pub fn response_captures(
        mut self,
        captures: impl IntoIterator<Item = ResponseCapture>,
    ) -> Self {
        self.response_captures.extend(captures);
        self
    }

    /// A captured response with tool calls (ideally two parallel calls), for
    /// [`ToolCallIdsRoundTrip`](Contract::ToolCallIdsRoundTrip). Name the
    /// ids the wire carries with
    /// [`wire_tool_call_ids`](ResponseCapture::wire_tool_call_ids), unless
    /// the API sends none and the adapter synthesizes them.
    #[must_use]
    pub fn tool_call_capture(mut self, capture: ResponseCapture) -> Self {
        self.tool_call_capture = Some(capture);
        self
    }

    /// A captured response cut off at the output token cap, for
    /// [`LengthFinish`](Contract::LengthFinish).
    #[must_use]
    pub fn length_capture(mut self, capture: ResponseCapture) -> Self {
        self.length_capture = Some(capture);
        self
    }

    /// Builds the provider under test against a base URL, with its real
    /// HTTP client and a key, and makes one call (`chat`, or `chat_stream`
    /// when the flag is `true`), for
    /// [`RedirectNotFollowed`](Contract::RedirectNotFollowed) (see
    /// [`assert_redirect_not_followed`](crate::redirect::assert_redirect_not_followed)).
    #[cfg(feature = "server")]
    #[must_use]
    pub fn redirect_not_followed<S, SFut>(mut self, send: S) -> Self
    where
        S: Fn(String, bool) -> SFut + 'static,
        SFut: Future<Output = allama::Result<()>> + 'static,
    {
        self.redirect = Some(Redirect {
            send: Box::new(move |url, streaming| Box::pin(send(url, streaming))),
        });
        self
    }
}

/// `chat` over a transport that answers with a body.
type UnaryReplay = Box<dyn Fn(String) -> BoxFuture<allama::Result<ChatResponse>>>;

/// The redirect contract's call.
struct Redirect {
    #[cfg_attr(not(feature = "server"), allow(dead_code))]
    send: Box<dyn Fn(String, bool) -> BoxFuture<allama::Result<()>>>,
}

impl Redirect {
    async fn run(&self) {
        #[cfg(feature = "server")]
        crate::redirect::assert_redirect_not_followed(|url, streaming| (self.send)(url, streaming))
            .await;
    }
}

/// An adapter's HTTP error mapping.
type ErrorMapping = Box<dyn Fn(u16, &[(&str, &str)], String) -> Error>;

/// An adapter's error envelope for each kind, around upstream text.
type Envelope = Box<dyn Fn(ErrorKindProbe, &str) -> Option<String>>;

/// Wraps an error envelope in a stream body.
type Wrap = Box<dyn Fn(&str) -> String>;

/// A boxed future the aggregates hold.
type BoxFuture<T> = Pin<Box<dyn Future<Output = T>>>;

/// `chat_stream` over a transport that answers with a body.
type StreamReplay = Box<dyn Fn(String) -> BoxFuture<allama::Result<ChatStream>>>;

/// A call with transport options attached.
type TransportSend = Box<dyn Fn(RequestOptions) -> BoxFuture<TransportOutcome>>;

fn boxed_send<S, SFut>(send: S) -> TransportSend
where
    S: Fn(RequestOptions) -> SFut + 'static,
    SFut: Future<Output = TransportOutcome> + 'static,
{
    Box::new(move |options| Box::pin(send(options)))
}

struct ChatTransport {
    slot: CredentialSlot,
    send: TransportSend,
    stream_send: TransportSend,
}

struct Malformed {
    send: Box<dyn Fn(String) -> BoxFuture<allama::Result<ChatResponse>>>,
    request_id: Option<String>,
}

/// The adapter's error mapping inputs, which every aggregate shares.
#[derive(Default)]
struct ErrorInputs {
    mapping: Option<ErrorMapping>,
    envelope: Option<Envelope>,
    retry_hint_body: Option<Box<dyn Fn(Duration) -> String>>,
}

impl ErrorInputs {
    /// Record what the error mapping contracts lack.
    fn require(&self, run: &mut Runner<'_>) {
        for contract in [
            Contract::ErrorKindsMapped,
            Contract::ErrorLogExcludesBody,
            Contract::RetrySignalsSurfaced,
        ] {
            run.needs(contract, self.mapping.is_none(), "`.error_mapping(..)`");
        }
        for contract in [Contract::ErrorKindsMapped, Contract::ErrorLogExcludesBody] {
            run.needs(contract, self.envelope.is_none(), "`.error_envelope(..)`");
        }
    }

    /// Run the error mapping contracts that have their inputs.
    async fn run(&self, run: &mut Runner<'_>) {
        let Some(map) = &self.mapping else {
            return;
        };
        let map = |status: u16, headers: &[(&str, &str)], body: String| map(status, headers, body);
        if let Some(envelope) = &self.envelope {
            run.contract(Contract::ErrorKindsMapped, async {
                crate::assert_error_kinds_mapped(|kind| envelope(kind, ""), map);
            })
            .await;
            run.contract(Contract::ErrorLogExcludesBody, async {
                crate::assert_error_log_excludes_body(envelope, map);
            })
            .await;
        }
        run.contract(Contract::RetrySignalsSurfaced, async {
            crate::assert_retry_signals_surfaced(self.retry_hint_body.as_deref(), map);
        })
        .await;
    }
}

/// Runs an aggregate's contracts: checks their inputs first, then runs
/// each, catching its failure, and reports them all.
struct Runner<'a> {
    aggregate: &'static str,
    provider: String,
    skip: &'a [Contract],
    /// Contracts whose missing inputs report "not run", and whose failures
    /// are printed, instead of failing.
    provisional: &'a [Contract],
    /// Provisional contracts that lack an input, and the input.
    not_run: Vec<(Contract, &'static str)>,
    /// Contracts that fail with an acknowledged defect.
    gaps: Vec<Contract>,
    missing: Vec<(Contract, &'static str)>,
    failures: Vec<(Contract, String)>,
}

impl<'a> Runner<'a> {
    fn new(
        aggregate: &'static str,
        provider: &str,
        skip: &'a [Contract],
        known_gaps: &[KnownGap],
    ) -> Self {
        Self {
            aggregate,
            provider: provider.to_owned(),
            skip,
            provisional: PROVISIONAL,
            not_run: Vec::new(),
            gaps: known_gaps
                .iter()
                .filter_map(|gap| match gap {
                    KnownGap::Contract(contract) => Some(*contract),
                    _ => None,
                })
                .collect(),
            missing: Vec::new(),
            failures: Vec::new(),
        }
    }

    /// Record that `contract` lacks `input` when `missing` is true.
    fn needs(&mut self, contract: Contract, missing: bool, input: &'static str) {
        if !missing || self.skip.contains(&contract) {
            return;
        }
        if self.provisional.contains(&contract) {
            self.not_run.push((contract, input));
        } else {
            self.missing.push((contract, input));
        }
    }

    /// Fail, naming each missing input, before running anything, and
    /// report the provisional contracts that will not run.
    fn check_inputs(&self) {
        for (contract, input) in &self.not_run {
            eprintln!(
                "{} of `{}`: provisional {contract} not run: needs {input}",
                self.aggregate, self.provider
            );
        }
        assert!(
            self.missing.is_empty(),
            "{} of `{}`: contracts lack inputs; give them, or list the contract in \
             `.skip(..)` when it does not apply:\n  {}",
            self.aggregate,
            self.provider,
            self.missing
                .iter()
                .map(|(contract, input)| format!("{contract} needs {input}"))
                .collect::<Vec<_>>()
                .join("\n  ")
        );
    }

    /// Run `contract` unless it is skipped or not run, recording its
    /// failure.
    async fn contract(&mut self, contract: Contract, check: impl Future<Output = ()>) {
        if self.skip.contains(&contract)
            || self.not_run.iter().any(|(not_run, _)| *not_run == contract)
        {
            return;
        }
        let known_gap = self.gaps.contains(&contract);
        match AssertUnwindSafe(check).catch_unwind().await {
            Err(_) if known_gap => {}
            Err(panic) => {
                let message = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                    .unwrap_or_else(|| "the contract panicked".to_owned());
                if self.provisional.contains(&contract) {
                    eprintln!(
                        "{} of `{}`: provisional {contract} fails:\n{}",
                        self.aggregate,
                        self.provider,
                        elide_payloads(&message)
                    );
                } else {
                    self.failures.push((contract, message));
                }
            }
            Ok(()) if known_gap => self.failures.push((
                contract,
                format!("{contract} is listed as a known gap but passes; remove it"),
            )),
            Ok(()) => {}
        }
    }

    /// Fail with every contract that failed.
    fn finish(self) {
        assert!(
            self.failures.is_empty(),
            "{} of `{}`: {} contract(s) failed:\n\n{}",
            self.aggregate,
            self.provider,
            self.failures.len(),
            self.failures
                .iter()
                .map(|(contract, message)| format!("{contract}:\n{}", elide_payloads(message)))
                .collect::<Vec<_>>()
                .join("\n\n")
        );
    }
}

#[cfg(test)]
mod tests;
