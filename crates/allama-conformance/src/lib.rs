#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]
//! Conformance contracts and test support for `allama` providers and
//! wrappers.
//!
//! Add this crate as a dev-dependency of a provider or wrapper crate and
//! call its contracts from your tests. Every contract panics with a
//! description of the violation, so it fits any test harness.
//!
//! # Contracts
//!
//! - **Aggregates** ([`adapter`]): [`assert_chat_adapter`] runs every
//!   probe-based chat contract below against an adapter from one
//!   [`AdapterUnderTest`], and grows with the crate;
//!   [`assert_embedding_adapter`], [`assert_decision_adapter`] and
//!   [`assert_extraction_adapter`] do the same for embeddings, decisions
//!   and extractions. A contract whose input is missing
//!   fails the aggregate unless it is listed in `skip`, so nothing is skipped
//!   silently. For wrappers, [`assert_chat_wrapper`] (and
//!   [`assert_wrapper`], [`assert_embedding_wrapper`],
//!   [`assert_decision_wrapper`]) runs transparency and the data path. Every
//!   contract below also stays a function of its own, for targeted use.
//! - **Fixture assertions** ([`assert_json_fixture_eq`],
//!   [`assert_response_fixture_eq`], [`assert_stream_fixture_eq`],
//!   [`assert_error_fixture_eq`], ...) pin a provider's wire shapes and
//!   normalization against files in a [`FixtureDir`]. Errors are compared
//!   by their non-redacted [`ErrorFixtureView`], so fixtures pin message
//!   mapping too.
//! - **Behavioral contracts** ([`contract`]): finish reasons, tool calls,
//!   streaming, structured output, usage ([`assert_usage_contract`]), the
//!   unsupported-feature policy, and error classification, on `chat` and
//!   `chat_stream`, and a malformed success body
//!   ([`assert_malformed_response_contract`](contract::assert_malformed_response_contract)).
//! - **In-band stream errors** ([`stream_errors`]):
//!   [`assert_stream_errors_classified`] checks that an error reported
//!   inside a stream has the kind the adapter's HTTP mapping gives the same
//!   error, for each [`ErrorKindProbe`](stream_errors::ErrorKindProbe) the
//!   adapter maps.
//! - **Decoding** ([`decoding`]): captured responses decode the same
//!   through `chat` and `chat_stream`, both paths stamp the provider,
//!   tool-call ids survive a round trip, and a cut-off response finishes
//!   with `Length`.
//! - **Raw event order** ([`stream_order`]): the raw events of a stream
//!   keep the documented order, which the collector does not enforce.
//! - **Redirects** (`redirect`, feature `server`): an adapter's client
//!   never follows a redirect.
//! - **Stream ends** ([`stream_end`]): [`assert_stream_ends_classified`]
//!   checks that a stream cut short before its terminal marker fails with
//!   `Incomplete`, and that one ending with the marker but no finish reason
//!   collects, with a warning.
//! - **Vendor frames (C1)** ([`stream_frames`]):
//!   [`assert_unknown_stream_frames_surfaced`](stream_frames::assert_unknown_stream_frames_surfaced)
//!   checks that stream frames the decoder does not model are surfaced,
//!   never misdecoded or dropped, and that reframing a stream changes
//!   nothing, in any wire framing ([`StreamFraming`](stream_frames::StreamFraming)).
//! - **Request fields (C2)** ([`probe`]):
//!   [`assert_request_fields_accounted`](probe::assert_request_fields_accounted)
//!   checks that every field of `ChatRequest` reaches the wire, is warned
//!   about, or is rejected, and that state from another provider and
//!   application data never reach the wire, on `chat` and `chat_stream`
//!   (see [streaming probes](probe#streaming-probes)). [`ACCOUNTED_FIELDS`](probe::ACCOUNTED_FIELDS) lists
//!   every serialized request key with the [`Probe`](probe::Probe) that
//!   checks it or the reason it is exempt; it is derived from `ChatRequest`
//!   in `allama`, so a new field cannot go unprobed.
//! - **Capability audits (C3)** ([`capabilities`]):
//!   [`assert_capabilities_honest`](capabilities::assert_capabilities_honest)
//!   checks that no capability answer makes a request drop what it asks
//!   for, which is the evidence for a chat adapter's `Supported` claims;
//!   [`assert_embedding_capabilities_backed`](capabilities::assert_embedding_capabilities_backed)
//!   checks that every `Supported` embedding claim names its evidence;
//!   [`assert_unsupported_chat_capabilities_refused`](capabilities::assert_unsupported_chat_capabilities_refused)
//!   and its embedding variant check that capabilities reported
//!   `Unsupported` are refused (a warning naming the capability's field or
//!   content, or a refusal before sending), never sent silently; the chat
//!   audit runs on both paths.
//!   [`assert_stream_usage_matches_capability`](capabilities::assert_stream_usage_matches_capability)
//!   checks the `StreamUsage` answer against a collected stream.
//! - **Validation agreement**: [`assert_validation_agrees`]
//!   checks that `validate_chat_request` and `validate_chat_stream_request`
//!   give the answer `chat` and `chat_stream` give (a refusal, or the same
//!   warnings), for [`drift_requests`] and your own; two-way for an adapter
//!   whose validation reports `Encoding` coverage. The embedding and
//!   decision variants check `embed` and `decide`.
//! - **Transparent wrappers (C4)** ([below](#transparent-wrappers)):
//!   [`assert_transparent_wrapper`] and its per-trait variants check that a
//!   wrapper forwards identity, capabilities, limits, and validation to the
//!   provider it wraps;
//!   [`assert_transparent_chat_data_path`] and its embedding and decision
//!   variants check that the provider's responses, stream events and errors
//!   reach the caller unchanged.
//! - **Retry hints (C5)**: [`assert_retry_respects_hint`]
//!   checks that a retry wrapper never calls a provider again before its
//!   `Retry-After` hint.
//! - **Retry signals**: [`assert_retry_signals_surfaced`]
//!   checks that an HTTP error mapping surfaces the server's retry
//!   directive (`x-should-retry`) and retry hint on the error, from the
//!   headers and, for an API that has one, from the body.
//! - **Error kinds** ([`error_kinds`]): [`assert_error_kinds_mapped`]
//!   checks that an HTTP error mapping gives each error kind its variant,
//!   keeps the status, and marks a provider's refusal with its stage.
//! - **Transport options**:
//!   [`assert_transport_options_applied`] checks that an HTTP adapter sends
//!   the caller's per-request headers, replacing its own of the same name,
//!   and the credentials addressed to its credential scope only, refusing
//!   plain credential headers.
//! - **Strict mode** ([`probe`]):
//!   [`assert_strict_not_honoured_warns`](probe::assert_strict_not_honoured_warns)
//!   checks that a strict tool or response format a provider cannot honour
//!   is reported with a warning on `chat` and `chat_stream`, and fails under
//!   `UnsupportedPolicy::Error`.
//! - **`Other` provider data** ([`probe`]):
//!   [`assert_other_provider_data_gated`](probe::assert_other_provider_data_gated)
//!   checks that provider data in another namespace on an `Other` block or
//!   part never reaches the wire.
//! - **Error redaction (C7)** ([`redaction`]):
//!   [`assert_error_log_excludes_body`] checks that upstream text in error
//!   bodies and envelopes, including the adapter's own envelope for every
//!   error kind it maps, never reaches the redacted
//!   [`as_log`](allama::Error::as_log) view or changes the variant, and
//!   that prose in an envelope's error type, code and `param` reaches
//!   neither it nor `Debug`.
//! - **Application data** ([`probe`]):
//!   [`assert_app_data_never_sent`](probe::assert_app_data_never_sent)
//!   checks that application data on prompts and messages never reaches the
//!   wire. Its conversation follows the provider's capability answers, so an
//!   adapter without tool support can pass it.
//! - **Empty turns** ([`probe`]):
//!   [`assert_empty_turns_skipped`](probe::assert_empty_turns_skipped)
//!   checks that an empty user or assistant message changes nothing that is
//!   sent, and that a reply left last by empty turns is dropped with a
//!   warning.
//! - **Unreplayable reasoning** ([`probe`]):
//!   [`assert_unreplayable_reasoning_dropped`](probe::assert_unreplayable_reasoning_dropped)
//!   checks that reasoning never reaches the wire for a model that reports
//!   `ReasoningReplay` `Unsupported`, and that the warnings about the turn
//!   name the caller's blocks.
//! - **Batch limits (C6)** ([`probe`]):
//!   [`assert_batch_limit_enforced`](probe::assert_batch_limit_enforced)
//!   checks that an embedding batch over the provider's limit is never sent
//!   as one request.
//! - **Embedding vector count** ([`probe`]):
//!   [`assert_embedding_count`](probe::assert_embedding_count) checks that
//!   an embedding response that cannot be aligned with the inputs is
//!   rejected.
//! - **Live scenarios** ([`e2e`]): the scenarios every provider runs
//!   against its real API.
//!
//! # Transparent wrappers
//!
//! A wrapper (retry, tracing, timeout, a custom middleware) changes how calls
//! run, not who answers them. Everything a caller asks *about* the provider
//! must reach the provider it wraps:
//!
//! - [`provider_name`](allama::ProviderIdentity::provider_name), used in
//!   logs, spans, errors, and fixtures;
//! - [`reads_namespace`](allama::ProviderIdentity::reads_namespace), which
//!   decides what state `retain_provider_state_for` keeps;
//! - [`downcast_any`](allama::ProviderIdentity::downcast_any), so callers can
//!   reach the concrete provider behind the wrapper;
//! - the capability answers
//!   ([`chat_capability`](allama::ChatProvider::chat_capability),
//!   [`embedding_capability`](allama::EmbeddingProvider::embedding_capability),
//!   [`decision_capability`](allama::DecisionProvider::decision_capability)),
//!   which drive validation and extraction's `Auto` mode;
//! - the limits
//!   ([`embedding_limits`](allama::EmbeddingProvider::embedding_limits),
//!   [`decision_limits`](allama::DecisionProvider::decision_limits));
//! - request validation
//!   ([`validate_chat_request`](allama::ChatProvider::validate_chat_request),
//!   [`validate_chat_stream_request`](allama::ChatProvider::validate_chat_stream_request),
//!   [`validate_embedding_request`](allama::EmbeddingProvider::validate_embedding_request),
//!   [`validate_decision_request`](allama::DecisionProvider::validate_decision_request)),
//!   with the warnings, coverage and `Unknown` answers the provider reports.
//!
//! A wrapper that forgets one of these still compiles, then degrades
//! silently: its default answers hide the provider's. Each check builds the
//! wrapper around a [`TransparencyProbe`], a provider with distinctive
//! answers (a unique name, a different capability answer per capability and
//! model, odd limits, a model it rejects, a validation of its own for each
//! method), and asserts the wrapper gives the same answers.
//!
//! Its calls go through the wrapper too: the **data path**. Whatever the
//! wrapped provider answers must reach the caller unchanged: its warnings
//! (a wrapper may add its own), provider data, usage and ids, every stream
//! event (warnings, refusal deltas, metadata) in order, and every error with
//! the kind, status, retry hint, refused subject and stage a caller decides
//! on. A wrapper that adds context to an error prefixes its message with
//! [`Error::with_message_prefix`](allama::Error::with_message_prefix);
//! re-wrapping it in another error loses what the caller decides on.
//!
//! Pick the check for the traits the wrapper implements. The `assert_*_wrapper`
//! aggregates run everything (identity, answers and the data path), and are
//! the ones to call:
//!
//! | Wrapper implements | Check |
//! |---|---|
//! | chat, embeddings, and decisions | [`assert_wrapper`] |
//! | [`ChatProvider`](allama::ChatProvider) | [`assert_chat_wrapper`] |
//! | [`EmbeddingProvider`](allama::EmbeddingProvider) | [`assert_embedding_wrapper`] |
//! | [`DecisionProvider`](allama::DecisionProvider) | [`assert_decision_wrapper`] |
//! | [`ExtractionProvider`](allama::ExtractionProvider) (unstable, ADR 0018; not part of `assert_wrapper`) | [`assert_extraction_wrapper`] |
//! | [`ProviderIdentity`](allama::ProviderIdentity) only | [`assert_transparent_identity`] |
//! | a model mapping, every trait | [`assert_transparent_model_mapping`] |
//! | a model mapping, [`ChatProvider`](allama::ChatProvider) | [`assert_transparent_chat_model_mapping`] |
//!
//! Each part runs on its own too: [`assert_transparent_wrapper`] and its
//! per-trait variants check the answers, and
//! [`assert_transparent_chat_data_path`] and its embedding and decision
//! variants check the data path. The embedding and decision parts
//! ([`assert_transparent_embedding_wrapper`],
//! [`assert_transparent_decision_wrapper`],
//! [`assert_transparent_embedding_data_path`] and
//! [`assert_transparent_decision_data_path`]) are provisional: the
//! aggregates run them, and on their own they may change or go in any
//! minor release (ADR 0017). The extraction parts
//! ([`assert_transparent_extraction_wrapper`],
//! [`assert_transparent_extraction_data_path`]) follow the trait's
//! unstable status (ADR 0018).
//!
//! Wrappers that combine several providers or change requests on purpose
//! pass when every inner provider is the probe: `FallbackProvider` reports
//! the primary's name, reads the namespaces either side reads, gives the
//! primary's validation, keeps the answers both sides agree on, and takes
//! the smaller limits, so
//! `FallbackProvider::new(probe, probe)` is transparent; when both sides
//! fail, it returns the error the probe returned, with the primary's
//! [behind it](allama::Error::fallback_from), and the data path compares the
//! returned error itself.
//! `MapRequestProvider` validates the transformed request, so it is
//! transparent with a transform that keeps the request valid. A retry
//! wrapper calls the probe again for a retryable error and passes on the
//! same error when it gives up; give it a policy without backoff in the
//! test, so the data path does not wait.
//!
//! A wrapper that maps model ids (`MapModelProvider`, or a hand-written one
//! that pins a model) answers for the model it sends, not the one it was
//! asked about. The model mapping checks take the expected mapping and
//! compare the wrapper's answers for each model with the probe's answers for
//! the mapped model.
//!
//! # Feature flags
//!
//! - `server` (default): `TestHttpServer`, a local HTTP server that
//!   serves canned `MockHttpResponse`s and records what it receives, for
//!   transport tests. Pulls in `axum` and Tokio's networking. Turn it off
//!   (`default-features = false`) when you test a wrapper or a provider
//!   that does not speak HTTP. It brings no HTTP client: the provider under
//!   test reaches the server over `allama-http`'s `reqwest` transport, so a
//!   crate whose own `reqwest` feature is off in the test build adds
//!   `allama-http` with that feature as a dev-dependency.
//! - `extract` (default): the extraction scenarios in [`e2e`] (requires
//!   `allama/chat-extract`).
//!
//! The contracts that wait ([`assert_retry_respects_hint`]) need a Tokio
//! runtime with timers, such as `#[tokio::test]`.
//!
//! See the [provider author guide]
//! for how these fit into a provider's test suite.
//!
#![doc = concat!(
    "[provider author guide]: https://github.com/sagikazarmark/allama/blob/v",
    env!("CARGO_PKG_VERSION"),
    "/docs/guides/provider-authors.md",
)]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]

pub mod adapter;
pub mod capabilities;
pub mod contract;
pub mod decoding;
mod diagnostics;
pub mod e2e;
pub mod error_kinds;
#[cfg(feature = "server")]
mod http_test_server;
pub mod probe;
pub mod redaction;
#[cfg(feature = "server")]
pub mod redirect;
mod retry;
mod retry_signals;
pub mod stream_end;
pub mod stream_errors;
pub mod stream_frames;
pub mod stream_order;
mod transparency;
mod transport_options;
mod validation;

use std::path::{Path, PathBuf};

use allama::chat::stream::StreamCollector;
use allama::{ChatResponse, ChatStream, StreamEvent, Usage};
use futures_util::StreamExt;
use serde::Serialize;
use serde_json::{Value, json};

pub use adapter::{
    AdapterUnderTest, Contract, DecisionAdapterUnderTest, EmbeddingAdapterUnderTest,
    ExtractionAdapterUnderTest, KnownGap, assert_chat_adapter, assert_decision_adapter,
    assert_embedding_adapter, assert_extraction_adapter,
};
pub use decoding::ResponseCapture;
pub use error_kinds::assert_error_kinds_mapped;
#[cfg(feature = "server")]
pub use http_test_server::{
    MockHttpResponse, RecordedRequest, TestHttpServer, UNPLANNED_REQUEST_STATUS,
};
pub use redaction::assert_error_log_excludes_body;
pub use retry::{RetryHintProbe, assert_retry_respects_hint};
pub use retry_signals::assert_retry_signals_surfaced;
pub use stream_end::assert_stream_ends_classified;
pub use stream_errors::assert_stream_errors_classified;
pub use transparency::{
    TransparencyProbe, assert_chat_wrapper, assert_decision_wrapper, assert_embedding_wrapper,
    assert_extraction_wrapper, assert_transparent_chat_data_path,
    assert_transparent_chat_model_mapping, assert_transparent_chat_wrapper,
    assert_transparent_decision_data_path, assert_transparent_decision_wrapper,
    assert_transparent_embedding_data_path, assert_transparent_embedding_wrapper,
    assert_transparent_extraction_data_path, assert_transparent_extraction_wrapper,
    assert_transparent_identity, assert_transparent_model_mapping, assert_transparent_wrapper,
    assert_wrapper,
};
pub use transport_options::{TransportOutcome, assert_transport_options_applied};
pub use validation::{
    assert_decision_validation_agrees, assert_embedding_validation_agrees,
    assert_extraction_validation_agrees, assert_validation_agrees, decision_drift_requests,
    drift_requests, embedding_drift_requests, extraction_drift_requests,
};

/// A directory of fixture files, usually `<crate>/fixtures/`.
///
/// ```
/// use std::path::PathBuf;
///
/// use allama_conformance::FixtureDir;
///
/// let fixtures = FixtureDir::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"));
/// ```
#[derive(Clone, Debug)]
pub struct FixtureDir {
    root: PathBuf,
}

impl FixtureDir {
    /// A fixture directory rooted at `root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The directory's path.
    #[must_use]
    pub fn as_path(&self) -> &Path {
        &self.root
    }

    fn file(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }
}

/// Read and parse the JSON fixture `name`.
///
/// # Panics
///
/// When the file cannot be read or is not valid JSON.
#[must_use]
pub fn load_json_fixture(fixtures: &FixtureDir, name: &str) -> Value {
    let path = fixtures.file(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("failed to parse fixture {} as json: {e}", path.display()))
}

/// Read the text fixture `name` (an SSE capture, for example).
///
/// # Panics
///
/// When the file cannot be read.
#[must_use]
pub fn load_text_fixture(fixtures: &FixtureDir, name: &str) -> String {
    let path = fixtures.file(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()))
}

/// Returns `true` when `ALLAMA_UPDATE_FIXTURES` is `1` or `true` (any
/// case): every fixture assertion then (re)writes its expected file,
/// creating it if needed, instead of comparing. Any other value, `0`,
/// `false` and an empty one included, compares. Review the diff of every
/// regenerated fixture before keeping it.
pub fn update_fixtures() -> bool {
    std::env::var("ALLAMA_UPDATE_FIXTURES").is_ok_and(|value| update_requested(&value))
}

/// Whether an `ALLAMA_UPDATE_FIXTURES` value asks for an update.
fn update_requested(value: &str) -> bool {
    let value = value.trim();
    value == "1" || value.eq_ignore_ascii_case("true")
}

/// Assert that `actual`, serialized to JSON, equals the fixture `name`
/// (typically a provider's wire request). Under
/// [`update_fixtures`] the fixture is rewritten instead.
///
/// # Panics
///
/// On a mismatch, or when the fixture cannot be read.
pub fn assert_json_fixture_eq<T: Serialize>(actual: &T, fixtures: &FixtureDir, name: &str) {
    let actual = serde_json::to_value(actual).expect("serialize actual json fixture value");
    check_fixture(actual, fixtures, name, "json");
}

/// Assert that a normalized [`ChatResponse`] equals the fixture `name`.
/// Under [`update_fixtures`] the fixture is rewritten instead.
///
/// # Panics
///
/// On a mismatch, or when the fixture cannot be read.
pub fn assert_response_fixture_eq(actual: &ChatResponse, fixtures: &FixtureDir, name: &str) {
    let actual = serde_json::to_value(actual).expect("serialize response");
    check_fixture(actual, fixtures, name, "response");
}

/// Assert that an error's [`ErrorFixtureView`] equals the fixture `name`.
/// Under [`update_fixtures`] the fixture is rewritten with the same view
/// instead.
///
/// # Panics
///
/// On a mismatch, or when the fixture cannot be read.
pub fn assert_error_fixture_eq(actual: &allama::Error, fixtures: &FixtureDir, name: &str) {
    check_fixture(error_fixture_value(actual), fixtures, name, "error");
}

/// The non-redacted view of an error that error fixtures compare.
///
/// The fields of the redacted [`as_log`](allama::Error::as_log) view, plus
/// the constructor `message`, its `message_origin` when it is
/// [authored](allama::Error::authored), and the `upstream_message`, so a
/// fixture pins how a provider maps messages and which ones it authors.
/// An error that ended a failover nests the view of the error it
/// [fell back from](allama::Error::fallback_from) under `fallback_from`.
///
/// ```
/// use allama::Error;
/// use allama_conformance::ErrorFixtureView;
///
/// let err = Error::invalid_request("HTTP 400")
///     .authored()
///     .with_status(400)
///     .with_upstream_message("bad value");
/// assert_eq!(
///     serde_json::to_value(ErrorFixtureView::new(&err)).unwrap(),
///     serde_json::json!({
///         "type": "invalid_request",
///         "status": 400,
///         "message": "HTTP 400",
///         "message_origin": "authored",
///         "upstream_message": "bad value",
///     }),
/// );
///
/// let failover = Error::rate_limited("HTTP 429").with_fallback_from(err);
/// let view = serde_json::to_value(ErrorFixtureView::new(&failover)).unwrap();
/// assert_eq!(view["type"], "rate_limited");
/// assert_eq!(view["fallback_from"]["upstream_message"], "bad value");
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ErrorFixtureView<'a>(&'a allama::Error);

impl<'a> ErrorFixtureView<'a> {
    /// Wrap an error in its fixture view.
    #[must_use]
    pub const fn new(error: &'a allama::Error) -> Self {
        Self(error)
    }
}

impl Serialize for ErrorFixtureView<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let error = self.0;
        let mut view = match serde_json::to_value(error.as_log()) {
            Ok(Value::Object(map)) => map,
            _ => serde_json::Map::new(),
        };
        view.insert("message".into(), error.message().into());
        if error.message_origin() == allama::MessageOrigin::Authored {
            view.insert("message_origin".into(), "authored".into());
        }
        if let Some(upstream) = error.upstream_message() {
            view.insert("upstream_message".into(), upstream.into());
        }
        // In place of the redacted log of the earlier error.
        if let Some(earlier) = error.fallback_from() {
            let earlier = serde_json::to_value(ErrorFixtureView::new(earlier)).unwrap_or_default();
            view.insert("fallback_from".into(), earlier);
        }
        Value::Object(view).serialize(serializer)
    }
}

/// Assert that an [`EmbeddingRequest`](allama::EmbeddingRequest)'s model,
/// inputs, dimensions, and input type (when set) equal the fixture `name`.
/// Under [`update_fixtures`] the fixture is rewritten instead.
///
/// # Panics
///
/// On a mismatch, or when the fixture cannot be read.
pub fn assert_embedding_request_fixture_eq(
    actual: &allama::EmbeddingRequest,
    fixtures: &FixtureDir,
    name: &str,
) {
    let mut projected = serde_json::json!({
        "model": actual.model,
        "inputs": actual.inputs,
        "dimensions": actual.dimensions,
    });
    // Only when set, so fixtures written before the field keep matching.
    if let Some(input_type) = actual.input_type {
        projected["input_type"] = serde_json::json!(input_type);
    }
    check_fixture(projected, fixtures, name, "embedding request");
}

/// Assert that a normalized
/// [`EmbeddingResponse`](allama::EmbeddingResponse) equals the fixture
/// `name`. Under [`update_fixtures`] the fixture is rewritten instead.
///
/// # Panics
///
/// On a mismatch, or when the fixture cannot be read.
pub fn assert_embedding_response_fixture_eq(
    actual: &allama::EmbeddingResponse,
    fixtures: &FixtureDir,
    name: &str,
) {
    let actual = serde_json::to_value(actual).expect("serialize embedding response");
    check_fixture(actual, fixtures, name, "embedding response");
}

/// Assert that an embedding error's [`ErrorFixtureView`] equals the fixture
/// `name`. Under [`update_fixtures`] the fixture is rewritten with the same
/// view instead.
///
/// # Panics
///
/// On a mismatch, or when the fixture cannot be read.
pub fn assert_embedding_error_fixture_eq(
    actual: &allama::Error,
    fixtures: &FixtureDir,
    name: &str,
) {
    check_fixture(
        error_fixture_value(actual),
        fixtures,
        name,
        "embedding error",
    );
}

/// Assert that a sequence of stream items (events and errors, in order)
/// equals the fixture `name`. Errors are compared by their
/// [`ErrorFixtureView`]. Under [`update_fixtures`] the
/// fixture is rewritten instead.
///
/// # Panics
///
/// On a mismatch, or when the fixture cannot be read.
pub fn assert_event_results_fixture_eq(
    actual: &[allama::Result<StreamEvent>],
    fixtures: &FixtureDir,
    name: &str,
) {
    let actual: Vec<Value> = actual
        .iter()
        .map(|item| match item {
            Ok(event) => serde_json::to_value(event).expect("serialize stream event"),
            Err(err) => json!({"error": error_fixture_value(err)}),
        })
        .collect();
    check_fixture(Value::Array(actual), fixtures, name, "stream event");
}

/// Assert that `usage` is consistent with the [`Usage`] contract: cache
/// reads and writes, alone or together, do not exceed the input; reasoning
/// does not exceed the output; and the total is at least input plus output.
///
/// It checks consistency only. It cannot tell whether your adapter
/// normalized the counts: an adapter that forgot to add cached tokens to
/// an input that excludes them still passes, as long as the cached count
/// is the smaller one. Pin the normalization with a cache-hit fixture
/// (a response that reports cache reads, and reasoning when the API has
/// it) whose normalized numbers you assert by hand, with a comment citing
/// the vendor's usage documentation for whether its input includes cached
/// tokens and its output includes reasoning.
pub fn assert_usage_contract(usage: &Usage) {
    if let Some(violation) = usage.contract_violation() {
        panic!("usage contract violated: {violation}: {usage:?}");
    }
}

/// A byte stream that yields `text` as one chunk, terminated by a blank
/// line, for feeding an SSE fixture to a stream decoder.
pub fn byte_stream_from_sse_text(
    text: &str,
) -> impl futures_core::Stream<Item = Result<Vec<u8>, std::io::Error>> + Send + Unpin + 'static {
    let normalized = if text.ends_with("\n\n") {
        text.to_string()
    } else {
        format!("{text}\n\n")
    };
    futures_util::stream::iter(vec![Ok(normalized.into_bytes())])
}

/// The most items a fixture stream may yield: a backstop for a stream that
/// never ends, so a broken adapter fails the test instead of hanging it.
const MAX_STREAM_ITEMS: usize = 100_000;

/// Collect a stream's items up to and including its first error, where a
/// chat stream ends.
async fn collect_stream_events(
    mut stream: ChatStream,
) -> (StreamCollector, Vec<allama::Result<StreamEvent>>) {
    let mut collector = StreamCollector::new();
    let mut seen = Vec::new();

    while let Some(item) = stream.next().await {
        assert!(
            seen.len() < MAX_STREAM_ITEMS,
            "the stream yielded more than {MAX_STREAM_ITEMS} items without ending"
        );
        match item {
            Ok(event) => {
                collector
                    .push_ref(&event)
                    .expect("stream collector should accept fixture event");
                seen.push(Ok(event));
            }
            Err(err) => {
                seen.push(Err(err));
                break;
            }
        }
    }

    (collector, seen)
}

/// Assert a complete stream: its items equal the fixture `events_name`, and
/// strict collection ([`StreamCollector::finish`]) yields the response in
/// the fixture `response_name`.
///
/// # Panics
///
/// On a mismatch, when the stream does not collect, or when a fixture
/// cannot be read.
pub async fn assert_stream_fixture_eq(
    stream: ChatStream,
    fixtures: &FixtureDir,
    events_name: &str,
    response_name: &str,
) {
    let (collector, seen) = collect_stream_events(stream).await;

    assert_event_results_fixture_eq(&seen, fixtures, events_name);

    let response = collector
        .finish()
        .expect("stream fixture should collect response");
    assert_response_fixture_eq(&response, fixtures, response_name);
}

/// Assert an interrupted stream's recovery: its items equal the fixture
/// `events_name`, and partial collection
/// ([`StreamCollector::finish_partial`]) yields the response in
/// `response_name` and the [`StreamCompleteness`](allama::chat::stream::StreamCompleteness)
/// in `completeness_name`.
///
/// # Panics
///
/// On a mismatch, when the stream cannot be reconstructed, or when a
/// fixture cannot be read.
pub async fn assert_partial_stream_fixture_eq(
    stream: ChatStream,
    fixtures: &FixtureDir,
    events_name: &str,
    response_name: &str,
    completeness_name: &str,
) {
    let (collector, seen) = collect_stream_events(stream).await;

    assert_event_results_fixture_eq(&seen, fixtures, events_name);

    let collected = collector
        .finish_partial()
        .expect("partial stream fixture should reconstruct response");
    assert_response_fixture_eq(&collected.response, fixtures, response_name);

    let actual = serde_json::to_value(collected.completeness)
        .expect("serialize partial stream completeness");
    check_fixture(actual, fixtures, completeness_name, "stream completeness");
}

/// Assert an interrupted stream's strict failure: its items equal the
/// fixture `events_name`, and the first error in the stream (or, when there
/// is none, the error from [`StreamCollector::finish`]) equals the fixture
/// `error_name`.
///
/// # Panics
///
/// On a mismatch, when the stream collects successfully, or when a fixture
/// cannot be read.
pub async fn assert_stream_finish_error_fixture_eq(
    stream: ChatStream,
    fixtures: &FixtureDir,
    events_name: &str,
    error_name: &str,
) {
    let (collector, seen) = collect_stream_events(stream).await;

    assert_event_results_fixture_eq(&seen, fixtures, events_name);

    if let Some(err) = seen.iter().find_map(|item| item.as_ref().err()) {
        assert_error_fixture_eq(err, fixtures, error_name);
        return;
    }

    let err = collector
        .finish()
        .expect_err("stream fixture should fail strict collection");
    assert_error_fixture_eq(&err, fixtures, error_name);
}

/// The [`ErrorFixtureView`] of `error`, as JSON.
fn error_fixture_value(error: &allama::Error) -> Value {
    serde_json::to_value(ErrorFixtureView::new(error)).expect("serialize error fixture view")
}

/// Compare `actual` with the fixture `name`, or write it there under
/// [`update_fixtures`]. Every fixture assertion goes through here, so the
/// value written is the value compared.
fn check_fixture(actual: Value, fixtures: &FixtureDir, name: &str, what: &str) {
    check_fixture_with(update_fixtures(), actual, fixtures, name, what);
}

fn check_fixture_with(update: bool, actual: Value, fixtures: &FixtureDir, name: &str, what: &str) {
    if update {
        write_fixture(fixtures, name, &actual);
        return;
    }
    let path = fixtures.file(name);
    assert!(
        path.is_file(),
        "{what} fixture {} does not exist; run with ALLAMA_UPDATE_FIXTURES=1 to create it",
        path.display()
    );
    let expected = load_json_fixture(fixtures, name);
    assert_eq!(actual, expected, "{what} fixture mismatch: {name}");
}

fn write_fixture(fixtures: &FixtureDir, name: &str, value: &Value) {
    let path = fixtures.file(name);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .unwrap_or_else(|e| panic!("failed to create {}: {e}", dir.display()));
    }
    let mut text = serde_json::to_string_pretty(value).expect("serialize fixture");
    text.push('\n');
    std::fs::write(&path, text)
        .unwrap_or_else(|e| panic!("failed to write fixture {}: {e}", path.display()));
}

#[cfg(test)]
mod fixture_update_tests {
    use super::*;
    use tempfile::TempDir;

    fn rate_limited() -> allama::Error {
        allama::Error::rate_limited("HTTP 429")
            .authored()
            .with_status(429)
            .with_upstream_message("slow down")
    }

    #[test]
    fn update_mode_creates_a_missing_error_fixture_with_the_compared_view() {
        let dir = TempDir::new().unwrap();
        let fixtures = FixtureDir::new(dir.path().join("fixtures"));
        let err = rate_limited();
        check_fixture_with(
            true,
            error_fixture_value(&err),
            &fixtures,
            "error_rate_limited.json",
            "error",
        );
        let written = load_json_fixture(&fixtures, "error_rate_limited.json");
        assert_eq!(
            written,
            serde_json::to_value(ErrorFixtureView::new(&err)).unwrap()
        );
        assert_eq!(written["message"], "HTTP 429");
        assert_eq!(written["message_origin"], "authored");
        assert_eq!(written["upstream_message"], "slow down");
        // The comparison then passes against what was written.
        assert_error_fixture_eq(&err, &fixtures, "error_rate_limited.json");
    }

    #[test]
    #[should_panic(expected = "error fixture mismatch: error_rate_limited.json")]
    fn error_fixture_mismatch_panics() {
        let dir = TempDir::new().unwrap();
        let fixtures = FixtureDir::new(dir.path());
        write_fixture(
            &fixtures,
            "error_rate_limited.json",
            &json!({"type": "rate_limited", "message": "HTTP 429"}),
        );
        check_fixture_with(
            false,
            error_fixture_value(&rate_limited()),
            &fixtures,
            "error_rate_limited.json",
            "error",
        );
    }

    #[test]
    #[should_panic(expected = "run with ALLAMA_UPDATE_FIXTURES=1 to create it")]
    fn a_missing_fixture_names_the_update_flag() {
        let dir = TempDir::new().unwrap();
        check_fixture_with(
            false,
            error_fixture_value(&rate_limited()),
            &FixtureDir::new(dir.path()),
            "error_rate_limited.json",
            "error",
        );
    }
}

#[cfg(test)]
mod embedding_assertion_tests {
    use super::*;
    use allama::{EmbeddingResponse, Usage};
    use tempfile::TempDir;

    fn write_fixture(dir: &TempDir, name: &str, content: &str) -> FixtureDir {
        let path = dir.path().join(name);
        std::fs::write(&path, content).unwrap();
        FixtureDir::new(dir.path().to_path_buf())
    }

    #[test]
    fn response_fixture_eq_passes_for_matching_json() {
        let dir = TempDir::new().unwrap();
        let fixtures = write_fixture(
            &dir,
            "embed_response_expected.json",
            r#"{"embeddings":[[0.0,1.0]],"model":"m","usage":{"input_tokens":5}}"#,
        );
        let response = EmbeddingResponse::new(vec![vec![0.0, 1.0]])
            .model("m")
            .usage(Usage::new().input_tokens(5));
        assert_embedding_response_fixture_eq(&response, &fixtures, "embed_response_expected.json");
    }

    #[test]
    #[should_panic(expected = "embedding response fixture mismatch")]
    fn response_fixture_eq_panics_on_mismatch() {
        let dir = TempDir::new().unwrap();
        let fixtures = write_fixture(
            &dir,
            "embed_response_expected.json",
            r#"{"embeddings":[[0.0]],"model":"other"}"#,
        );
        let response = EmbeddingResponse::new(vec![vec![0.0, 1.0]]).model("m");
        // Compare mode explicitly, so the test holds under
        // `ALLAMA_UPDATE_FIXTURES` too.
        check_fixture_with(
            false,
            serde_json::to_value(&response).unwrap(),
            &fixtures,
            "embed_response_expected.json",
            "embedding response",
        );
    }
}

#[cfg(test)]
mod update_fixtures_tests {
    /// Only `1` or `true` rewrites fixtures.
    #[test]
    fn only_one_or_true_requests_an_update() {
        for value in ["1", "true", "TRUE", " True "] {
            assert!(super::update_requested(value), "{value:?}");
        }
        for value in ["", "0", "false", "no", "yes"] {
            assert!(!super::update_requested(value), "{value:?}");
        }
    }
}
