//! Request honesty contracts: what a portable request asks for either
//! reaches the wire, or the caller is told.
//!
//! These contracts send probe requests through a provider wired to a canned
//! transport and inspect both the outcome and the request bodies that reached
//! the transport. The provider crate supplies the wiring as a *probe*: an
//! async closure that takes the probe request, sends it through a fresh
//! provider whose transport answers every call with a success response, and
//! returns a [`ProbeOutcome`] with the result and the JSON bodies sent.
//!
//! - [`assert_request_fields_accounted`] (C2): every field of
//!   [`ChatRequest`] that [`ACCOUNTED_FIELDS`] marks as probed is on the
//!   wire, or a warning names it, or the call fails with
//!   [`Error::Unsupported`]; state from another provider and application
//!   data never reach the wire.
//! - [`assert_batch_limit_enforced`]: an embedding batch one input over
//!   [`embedding_limits`](EmbeddingProvider::embedding_limits) is never
//!   sent as one oversized request.
//! - [`assert_embedding_count`]: an embedding response with the wrong number
//!   of vectors, or indices that are not exactly one per input, is rejected
//!   rather than misaligned with the inputs.
//! - [`assert_embedding_response_checked`]: an embedding response with
//!   vectors of the wrong size, of different sizes, empty, or with a value
//!   that is not finite is rejected rather than returned.
//! - [`assert_multi_part_input_refused`]: a model that embeds one text per
//!   input refuses an input with several parts before sending; the parts are
//!   never joined.
//! - [`assert_input_provider_data_checked`]: an unknown key in the adapter's
//!   own namespace on an [`EmbeddingInput`] is refused before sending, while
//!   the same key on a part and keys in other namespaces are accepted.
//! - [`assert_other_provider_data_gated`]: provider data in another namespace
//!   on [`OtherContent`] (assistant blocks and user parts) never reaches the
//!   wire, like on every other block and part.
//! - [`assert_app_data_never_sent`]: application data on system prompts and
//!   messages of every role never reaches the wire and changes nothing about
//!   the request the adapter sends. Tool calls and results are part of the
//!   conversation only for a provider that serves them.
//! - [`assert_empty_turns_skipped`]: an empty user or assistant message
//!   changes nothing that is sent, and an assistant turn left last only by
//!   empty turns is dropped with a `Dropped` warning.
//! - [`assert_unreplayable_reasoning_dropped`]: reasoning never reaches the
//!   wire for a model that cannot replay it, and the `Dropped` warnings name
//!   the caller's blocks, including those after the removed reasoning.
//! - [`assert_strict_not_honoured_warns`]: a strict tool or response format
//!   a provider cannot honour is reported with a warning, and fails under
//!   [`UnsupportedPolicy::Error`].
//! - [`assert_input_type_not_honoured_warns`]: an embedding input type a
//!   provider cannot honour is dropped with a warning, and fails before
//!   sending under [`UnsupportedPolicy::Error`].
//!
//! The negative capability audits in [`capabilities`](crate::capabilities)
//! use the same probes.
//!
//! Probes report the bodies they sent as JSON ([`ProbeOutcome::new`]) or as
//! raw bytes ([`ProbeOutcome::raw`]) for adapters whose wire format is not
//! JSON (protobuf, NDJSON); every contract here runs on either, searching
//! the bytes when the bodies are not JSON.
//!
//! # Streaming probes
//!
//! An adapter serves a request on two paths, `chat` and `chat_stream`, and
//! each can drop a field or a warning on its own. Every chat contract here
//! therefore takes two probes of the same shape: `probe` calls `chat`, and
//! `stream_probe` calls `chat_stream` over a transport that answers with a
//! canned streamed success response, and collects the stream into a
//! response with [`collect_response`](allama::ChatStreamExt::collect_response),
//! which keeps every `StreamEvent::Warning`. An error from `chat_stream`
//! itself, or from any item of the stream, becomes the outcome's error.
//! Problems name the path they were found on (`chat: …`, `chat_stream: …`).
//!
//! ```no_run
//! use allama::{ChatProvider, ChatRequest, ChatResponse, ChatStreamExt};
//! use allama_conformance::probe::{ProbeOutcome, assert_request_fields_accounted};
//! # struct Transport;
//! # impl Transport { fn bodies(&self) -> Vec<serde_json::Value> { Vec::new() } }
//! # fn provider_answering_ok() -> (allama::mock::MockStreamingProvider, Transport) {
//! #     (allama::mock::MockStreamingProvider::with_text("ok"), Transport)
//! # }
//! # fn provider_streaming_ok() -> (allama::mock::MockStreamingProvider, Transport) {
//! #     (allama::mock::MockStreamingProvider::with_text("ok"), Transport)
//! # }
//!
//! async fn probe(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
//!     // A provider whose transport answers with a canned success response.
//!     let (provider, transport) = provider_answering_ok();
//!     let result = provider.chat(&request).await;
//!     ProbeOutcome::new(result, transport.bodies())
//! }
//!
//! async fn stream_probe(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
//!     // The same provider, answering with a canned streamed response.
//!     let (provider, transport) = provider_streaming_ok();
//!     let result = match provider.chat_stream(&request).await {
//!         Ok(stream) => stream.collect_response().await,
//!         Err(err) => Err(err),
//!     };
//!     ProbeOutcome::new(result, transport.bodies())
//! }
//!
//! # async fn example() {
//! // A provider whose capability answers match the probed one's.
//! let (provider, _) = provider_answering_ok();
//! assert_request_fields_accounted(&provider, "my-model", &[], probe, stream_probe).await;
//! # }
//! ```

use std::collections::BTreeSet;
use std::future::Future;

use allama::{
    CapabilitySupport, ChatProvider, ChatRequest, ChatResponse, ContentBlock, ContentPart,
    EmbeddingCapability, EmbeddingInput, EmbeddingInputType, EmbeddingPart, EmbeddingProvider,
    EmbeddingRequest, EmbeddingResponse, Error, ExtraMap, Message, OtherContent, ProviderData,
    RequestField, ResponseFormat, Subject, Text, ToolDefinition, UnsupportedPolicy, Warning,
    WarningKind,
};
use serde_json::Value;

use crate::diagnostics::elide_payloads;

pub(crate) use fields::WireValue;
pub use fields::{ACCOUNTED_FIELDS, Account, AccountedField, Probe, accounted_fields};
use fields::{CASES, Case, Evidence, Refusal, Rule, Supports};
pub use reasoning::{
    REASONING_PROBE_FOREIGN, REASONING_PROBE_TEXT, assert_unreplayable_reasoning_dropped,
};
pub use turns::assert_empty_turns_skipped;

mod fields;
mod reasoning;
mod turns;

/// What happened to a probe request.
#[derive(Debug)]
#[non_exhaustive]
pub struct ProbeOutcome<T> {
    /// The provider's answer.
    pub result: allama::Result<T>,
    /// The request bodies that reached the transport, parsed as JSON, in
    /// order. Empty when the provider failed before sending, or when the
    /// bodies are not JSON (see [`sent_raw`](Self::sent_raw)).
    pub sent: Vec<Value>,
    /// The request bodies that reached the transport, as bytes, in order.
    /// Empty when the provider failed before sending.
    pub sent_raw: Vec<Vec<u8>>,
}

impl<T> ProbeOutcome<T> {
    /// An outcome from the provider's answer and the JSON bodies it sent.
    #[must_use]
    pub fn new(result: allama::Result<T>, sent: Vec<Value>) -> Self {
        let sent_raw = sent
            .iter()
            .map(|body| serde_json::to_vec(body).expect("a JSON value serializes"))
            .collect();
        Self {
            result,
            sent,
            sent_raw,
        }
    }

    /// An outcome from the provider's answer and the bodies it sent, as
    /// bytes, for adapters whose wire format is not JSON.
    ///
    /// When every body is a JSON document, [`sent`](Self::sent) holds them
    /// parsed; otherwise it is empty and contracts search the bytes.
    #[must_use]
    pub fn raw<B: Into<Vec<u8>>>(
        result: allama::Result<T>,
        sent: impl IntoIterator<Item = B>,
    ) -> Self {
        let sent_raw: Vec<Vec<u8>> = sent.into_iter().map(Into::into).collect();
        let sent = sent_raw
            .iter()
            .map(|body| serde_json::from_slice(body).ok())
            .collect::<Option<Vec<Value>>>()
            .unwrap_or_default();
        Self {
            result,
            sent,
            sent_raw,
        }
    }

    /// Whether nothing reached the transport.
    pub(crate) fn nothing_sent(&self) -> bool {
        self.sent_raw.is_empty()
    }

    /// Whether the bodies are available as JSON.
    pub(crate) fn is_json(&self) -> bool {
        self.sent.len() == self.sent_raw.len()
    }

    /// Whether `value` appears in a body that was sent.
    pub(crate) fn contains(&self, value: &WireValue) -> bool {
        if self.is_json() {
            self.sent.iter().any(|body| json_contains(body, value))
        } else {
            self.sent_raw.iter().any(|body| raw_contains(body, value))
        }
    }

    /// Whether `text` appears in a body that was sent, searched in the bytes
    /// so a JSON key, a JSON value and a non-JSON body all count.
    fn contains_text(&self, text: &'static str) -> bool {
        self.sent_raw
            .iter()
            .any(|body| raw_contains(body, &WireValue::Text(text)))
    }

    /// The bodies sent, as text for problem reports.
    pub(crate) fn bodies(&self) -> Vec<String> {
        self.sent_raw
            .iter()
            .map(|body| String::from_utf8_lossy(body).into_owned())
            .collect()
    }

    /// Whether the bodies sent differ from those of `other`.
    pub(crate) fn differs<U>(&self, other: &ProbeOutcome<U>) -> bool {
        if self.is_json() && other.is_json() {
            self.sent != other.sent
        } else {
            self.sent_raw != other.sent_raw
        }
    }
}

/// Assert that every field of [`ChatRequest`] that [`ACCOUNTED_FIELDS`]
/// marks as probed is accounted for (contract C2).
///
/// Each [`Probe`] sends requests for `model` that set what it probes to a
/// distinctive value, and judges the outcome:
///
/// - **Hints** (sampling fields, cache breakpoints and settings, image
///   detail, strict tools, reasoning configuration, …) pass when the value
///   reaches the wire, or the call fails with [`Error::Unsupported`], or a
///   warning names it. An [`Approximated`](WarningKind::Approximated)
///   warning means it was honored approximately; a value dropped with any
///   other warning is probed again under [`UnsupportedPolicy::Error`], which
///   must fail with [`Error::Unsupported`] before anything is sent
///   (ADR 0003).
/// - **Metadata** without a wire slot everywhere (message names, file names,
///   tool-result errors) may also be dropped with a
///   [`Dropped`](WarningKind::Dropped) warning that names it.
/// - **State from another provider** (reasoning, `Other` content,
///   provider-executed tools, provider data) must never reach the wire;
///   content is dropped with a `Dropped` warning that names it (ADR 0002).
/// - **Application data** must change nothing: the request is sent and
///   answered exactly as without it (see [`assert_app_data_never_sent`]).
///
/// A refusal with stage
/// [`BeforeSending`](allama::UnsupportedStage::BeforeSending) means
/// "refused before sending", so on any probe it must come with nothing
/// sent; and a hint refused under [`UnsupportedPolicy::Error`] must not say
/// it was ignored or sent.
///
/// A value reaches the wire when it appears in a request body, or, for a
/// value with no distinctive form (a cache retention, an image detail
/// level), when the body differs from the one sent for a contrasting
/// request that sets another value; a warning about either value then
/// accounts for it, since an adapter may honor one value by default. A
/// plain cache breakpoint may be satisfied by automatic prefix caching, so
/// cache probes contrast long and short retention. Warnings name a probe
/// through their [`Subject`]: the request field, the
/// system prompt, or the message that carries it. [`Probe::describe`]
/// summarizes each rule.
///
/// Every case runs twice: through `probe`, which calls `chat`, and through
/// `stream_probe`, which collects `chat_stream` into a response (see
/// [streaming probes](self#streaming-probes)). Problems name the path that
/// failed.
///
/// `provider` answers capabilities for `model`. A check that does not accept
/// a refusal shapes its request to them: the application data conversation
/// holds a tool call and its result only when
/// [`ChatCapability::ToolCalls`](allama::ChatCapability::ToolCalls) is not
/// [`Unsupported`](allama::CapabilitySupport::Unsupported).
///
/// `known_gaps` lists probes with an acknowledged defect: they are skipped,
/// but listing a probe that passes on both paths fails, so fixed gaps leave
/// the list.
///
/// An `allama` newer than this crate may list fields this crate cannot
/// check: an entry of [`ACCOUNTED_FIELDS`] that names a probe it does not
/// know. The audit runs every probe it knows, then fails on those entries
/// too, asking you to upgrade `allama-conformance` to match `allama`. It
/// never passes a field it did not check.
///
/// # Panics
///
/// On any unaccounted field or stale known gap, and when
/// [`ACCOUNTED_FIELDS`] lists a field this crate cannot check.
pub async fn assert_request_fields_accounted<F, Fut, G, GFut>(
    provider: &impl ChatProvider,
    model: &str,
    known_gaps: &[Probe],
    probe: F,
    stream_probe: G,
) where
    F: FnMut(ChatRequest) -> Fut,
    Fut: Future<Output = ProbeOutcome<ChatResponse>>,
    G: FnMut(ChatRequest) -> GFut,
    GFut: Future<Output = ProbeOutcome<ChatResponse>>,
{
    request_field_audit(
        ACCOUNTED_FIELDS,
        provider,
        model,
        known_gaps,
        probe,
        stream_probe,
    )
    .await;
}

/// [`assert_request_fields_accounted`] against the accounted-fields `list`.
async fn request_field_audit<F, Fut, G, GFut>(
    list: &'static str,
    provider: &impl ChatProvider,
    model: &str,
    known_gaps: &[Probe],
    mut probe: F,
    mut stream_probe: G,
) where
    F: FnMut(ChatRequest) -> Fut,
    Fut: Future<Output = ProbeOutcome<ChatResponse>>,
    G: FnMut(ChatRequest) -> GFut,
    GFut: Future<Output = ProbeOutcome<ChatResponse>>,
{
    let (_, unread) = fields::parse_accounted_fields(list);
    let supports = Supports::of(provider, model);
    let mut report = Report::default();
    for &target in Probe::ALL {
        let mut problems = Vec::new();
        for case in CASES.iter().filter(|case| case.probe == target) {
            problems.extend(on_path(
                CHAT,
                run_case(case, model, supports, &mut probe).await,
            ));
            problems.extend(on_path(
                CHAT_STREAM,
                run_case(case, model, supports, &mut stream_probe).await,
            ));
        }
        let problem = (!problems.is_empty()).then(|| problems.join("\n  "));
        report.record(target, problem, known_gaps.contains(&target));
    }
    if !unread.is_empty() {
        report.fail(format!(
            "allama lists request fields this allama-conformance cannot check; upgrade \
             allama-conformance to the version of allama:\n    {}",
            unread.join("\n    ")
        ));
    }
    report.finish("request field audit");
}

/// Assert that an embedding batch one input over
/// [`embedding_limits`](EmbeddingProvider::embedding_limits) is never
/// sent as one oversized request: `embed` either fails with
/// [`Error::InvalidRequest`] before sending, or splits the batch and returns
/// one vector per input.
///
/// `provider` answers the limit; `probe` sends the request through the same
/// provider wired to a canned transport. The inputs are distinctive strings,
/// so the contract counts them in the bytes of each request body without
/// knowing the wire format: JSON ([`ProbeOutcome::new`]) and raw bodies
/// ([`ProbeOutcome::raw`]) are checked alike.
///
/// # Panics
///
/// When `provider` reports no limit for `model`, or the contract is
/// violated.
pub async fn assert_batch_limit_enforced<P, F, Fut>(provider: &P, model: &str, probe: F)
where
    P: EmbeddingProvider + ?Sized,
    F: FnOnce(EmbeddingRequest) -> Fut,
    Fut: Future<Output = ProbeOutcome<EmbeddingResponse>>,
{
    let max = provider
        .embedding_limits(model)
        .max_inputs_per_call
        .unwrap_or_else(|| {
            panic!("batch_limit: provider reports no max_inputs_per_call for `{model}`")
        });
    let inputs: Vec<String> = (0..=max)
        .map(|i| format!("{BATCH_PROBE_PREFIX}{i}"))
        .collect();
    let outcome = probe(EmbeddingRequest::new(model).inputs(inputs)).await;
    for (index, body) in outcome.sent_raw.iter().enumerate() {
        let found = batch_probe_inputs(body);
        assert!(
            found.len() <= max,
            "batch_limit: request {index} carried {} inputs, over the limit of {max}",
            found.len()
        );
    }
    match outcome.result {
        Ok(response) => assert_eq!(
            response.embeddings.len(),
            max + 1,
            "batch_limit: a split batch must return one vector per input"
        ),
        Err(Error::InvalidRequest { .. }) => {}
        // Split into conforming requests that the canned transport did not
        // all answer: the limit itself was honored.
        Err(_) if !outcome.nothing_sent() => {}
        Err(err) => panic!(
            "batch_limit: expected InvalidRequest before sending, got {}: {err}",
            err.kind_name()
        ),
    }
}

/// Prefix of the inputs [`assert_batch_limit_enforced`] sends.
pub const BATCH_PROBE_PREFIX: &str = "allama-batch-probe-";

/// How an adapter's embedding responses identify which input each vector
/// belongs to; see [`assert_embedding_count`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum VectorOrder {
    /// Each vector carries the index of its input (OpenAI's `data[].index`),
    /// and may arrive in any order.
    Indexed,
    /// Vectors arrive in input order with no index (Gemini's
    /// `batchEmbedContents`, Workers AI).
    InputOrder,
}

/// Contract: an adapter returns one vector per input, in input order, and
/// rejects a response that cannot be aligned with the inputs with
/// [`Error::UnexpectedResponse`].
///
/// `probe` sends the request (three inputs) through the adapter, wired to a
/// transport that answers with one vector for each entry of `answer`, in
/// that order: entry `i` is the vector `[i as f32]`, tagged with index `i`
/// when the wire format has one. The contract asks for:
///
/// - `[0, 1, 2]`: accepted, with the vectors in input order;
/// - too few (`[0, 1]`) and too many (`[0, 1, 2, 3]`) vectors: rejected;
/// - for [`VectorOrder::Indexed`], shuffled indices (`[2, 0, 1]`): accepted
///   and reordered; and a duplicate (`[0, 1, 1]`), a missing
///   (`[0, 2, 3]`), or an out-of-range index (`[0, 1, 3]`): rejected.
///
/// A RAG index built from misaligned vectors is silently wrong, so the
/// adapter must never guess.
///
/// # Panics
///
/// When the contract is violated.
pub async fn assert_embedding_count<F, Fut>(model: &str, order: VectorOrder, mut probe: F)
where
    F: FnMut(EmbeddingRequest, Vec<usize>) -> Fut,
    Fut: Future<Output = allama::Result<EmbeddingResponse>>,
{
    let request = || EmbeddingRequest::new(model).inputs(["first", "second", "third"]);
    let aligned = vec![vec![0.0], vec![1.0], vec![2.0]];
    let mut accepted = vec![vec![0, 1, 2]];
    let mut rejected = vec![vec![0, 1], vec![0, 1, 2, 3]];
    if order == VectorOrder::Indexed {
        accepted.push(vec![2, 0, 1]);
        rejected.extend([vec![0, 1, 1], vec![0, 2, 3], vec![0, 1, 3]]);
    }
    let mut problems = Vec::new();
    for answer in accepted {
        match probe(request(), answer.clone()).await {
            Ok(response) if response.embeddings == aligned => {}
            Ok(response) => problems.push(format!(
                "answer {answer:?}: expected the vectors in input order {aligned:?}, got {:?}",
                response.embeddings
            )),
            Err(err) => problems.push(format!(
                "answer {answer:?}: expected success, got {}: {err}",
                err.kind_name()
            )),
        }
    }
    for answer in rejected {
        match probe(request(), answer.clone()).await {
            Err(Error::UnexpectedResponse { .. }) => {}
            other => problems.push(format!(
                "answer {answer:?} for 3 inputs: expected UnexpectedResponse, got {}",
                describe(&other)
            )),
        }
    }
    assert!(
        problems.is_empty(),
        "embedding_count failed:\n  {}",
        problems.join("\n  ")
    );
}

/// A value [`assert_embedding_response_checked`] answers with that `f32`
/// cannot hold: a JSON number beyond its range, which an adapter decoding
/// `f32` reads as infinity.
pub const NON_FINITE_PROBE: f64 = 1e39;

/// Contract: an adapter rejects an embedding response whose vectors a
/// caller cannot use with [`Error::UnexpectedResponse`], attributed to the
/// provider, instead of returning it
/// ([`allama::adapter::embedding::check_response`]).
///
/// `probe` sends the request through the adapter, wired to a transport that
/// answers with the vectors in `answer`, in order (tagged with their index
/// when the wire format has one), written as JSON numbers. The contract asks
/// for:
///
/// - vectors of one size: accepted, as given;
/// - vectors of different sizes, for two inputs (unless `provider` answers
///   [`BatchInput`](EmbeddingCapability::BatchInput) `Unsupported` for
///   `model`): rejected;
/// - an empty vector: rejected;
/// - a value of [`NON_FINITE_PROBE`]: rejected;
/// - unless `provider` answers
///   [`OutputDimensions`](EmbeddingCapability::OutputDimensions)
///   `Unsupported`, a vector of 37 values for a request of 48 dimensions:
///   rejected with a message that names the model and both sizes; 48 values
///   are accepted.
///
/// # Panics
///
/// When the contract is violated.
pub async fn assert_embedding_response_checked<P, F, Fut>(provider: &P, model: &str, mut probe: F)
where
    P: EmbeddingProvider + ?Sized,
    F: FnMut(EmbeddingRequest, Vec<Vec<f64>>) -> Fut,
    Fut: Future<Output = allama::Result<EmbeddingResponse>>,
{
    let supports = |capability| {
        provider.embedding_capability(model, capability) != CapabilitySupport::Unsupported
    };
    let one = || EmbeddingRequest::new(model).input("first");
    let mut accepted = vec![(one(), vec![vec![0.25, 0.5]])];
    let mut rejected = vec![
        (one(), vec![vec![]], "an empty vector"),
        (
            one(),
            vec![vec![0.25, NON_FINITE_PROBE]],
            "a non-finite value",
        ),
    ];
    if supports(EmbeddingCapability::BatchInput) {
        rejected.push((
            EmbeddingRequest::new(model).inputs(["first", "second"]),
            vec![vec![0.25, 0.5], vec![0.25]],
            "vectors of different sizes",
        ));
    }
    let dimensions = supports(EmbeddingCapability::OutputDimensions);
    if dimensions {
        accepted.push((one().dimensions(48), vec![vec![0.25; 48]]));
        rejected.push((
            one().dimensions(48),
            vec![vec![0.25; 37]],
            "37 values for 48 dimensions",
        ));
    }
    let mut problems = Vec::new();
    for (request, answer) in accepted {
        let expected: Vec<Vec<f32>> = answer
            .iter()
            .map(|vector| vector.iter().map(|&value| value as f32).collect())
            .collect();
        match probe(request, answer).await {
            Ok(response) if response.embeddings == expected => {}
            Ok(response) => problems.push(format!(
                "expected the vectors {expected:?}, got {:?}",
                response.embeddings
            )),
            Err(err) => problems.push(format!(
                "vectors of {} values: expected success, got {}: {err}",
                expected[0].len(),
                err.kind_name()
            )),
        }
    }
    for (request, answer, what) in rejected {
        let result = probe(request, answer).await;
        match &result {
            Err(err @ Error::UnexpectedResponse { .. }) => {
                if err.context().and_then(|c| c.provider.as_deref())
                    != Some(provider.provider_name())
                {
                    problems.push(format!("{what}: the error does not name the provider"));
                }
                if what.contains("dimensions")
                    && !["48", "37", model]
                        .iter()
                        .all(|needle| err.message().contains(needle))
                {
                    problems.push(format!(
                        "{what}: the message must name the model and both sizes: {err}"
                    ));
                }
            }
            other => problems.push(format!(
                "{what}: expected UnexpectedResponse, got {}",
                describe(other)
            )),
        }
    }
    assert!(
        problems.is_empty(),
        "embedding_response_checked failed:\n  {}",
        problems.join("\n  ")
    );
}

/// Contract: a model that embeds one text per input refuses an input with
/// several parts before sending, under every
/// [`UnsupportedPolicy`], and attributes the error to `provider_name`. The
/// parts are never joined into one text.
///
/// # Panics
///
/// When the contract is violated.
pub async fn assert_multi_part_input_refused<F, Fut>(model: &str, provider_name: &str, probe: F)
where
    F: Fn(EmbeddingRequest) -> Fut,
    Fut: Future<Output = ProbeOutcome<EmbeddingResponse>>,
{
    for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
        let request = EmbeddingRequest::new(model)
            .input(vec![
                EmbeddingPart::text("first"),
                EmbeddingPart::text("second"),
            ])
            .unsupported_policy(policy);
        let outcome = probe(request).await;
        assert!(
            outcome.nothing_sent(),
            "multi_part_input: an input with several parts reached the wire: {:?}",
            outcome.sent
        );
        match &outcome.result {
            Err(err @ Error::Unsupported { .. }) => {
                if let Some(problem) = misshapen_refusal(&outcome) {
                    panic!("multi_part_input: {problem}");
                }
                assert_eq!(
                    err.context().and_then(|c| c.provider.as_deref()),
                    Some(provider_name),
                    "multi_part_input: the error must name the provider"
                );
            }
            other => panic!(
                "multi_part_input: expected Unsupported under {policy:?}, got {}",
                describe(other)
            ),
        }
    }
}

/// Key [`assert_input_provider_data_checked`] uses; no adapter reads it.
pub const PROVIDER_DATA_PROBE_KEY: &str = "allama-probe-unknown-key";

/// Contract: an adapter that reads `namespace` refuses an unknown key there on
/// an input's provider data with [`Error::InvalidRequest`] before sending, and
/// attributes the error to `provider_name`. The same key on a part (where chat
/// replay state legitimately lives) and keys in other namespaces are
/// accepted.
///
/// # Panics
///
/// When the contract is violated.
pub async fn assert_input_provider_data_checked<F, Fut>(
    model: &str,
    provider_name: &str,
    namespace: &str,
    probe: F,
) where
    F: Fn(EmbeddingRequest) -> Fut,
    Fut: Future<Output = ProbeOutcome<EmbeddingResponse>>,
{
    let data = |namespace: &str| {
        ProviderData::new().with_raw(namespace, PROVIDER_DATA_PROBE_KEY, Value::Bool(true))
    };

    let refused = probe(
        EmbeddingRequest::new(model)
            .input("first")
            .input(EmbeddingInput::text("second").with_provider_data(data(namespace))),
    )
    .await;
    assert!(
        refused.nothing_sent(),
        "input_provider_data: an input with an unknown `{namespace}` key reached the wire"
    );
    match &refused.result {
        Err(err @ Error::InvalidRequest { .. }) => {
            assert_eq!(
                err.context().and_then(|c| c.provider.as_deref()),
                Some(provider_name),
                "input_provider_data: the error must name the provider"
            );
            assert!(
                err.message().contains("input 1"),
                "input_provider_data: the error must name the input: {err}"
            );
        }
        other => panic!(
            "input_provider_data: expected InvalidRequest, got {}",
            describe(other)
        ),
    }

    let foreign = if namespace == "allama-probe-foreign" {
        "allama-probe-other"
    } else {
        "allama-probe-foreign"
    };
    let accepted = probe(
        EmbeddingRequest::new(model)
            .input(EmbeddingInput::from(
                Text::new("first").with_provider_data(data(namespace)),
            ))
            .input(EmbeddingInput::text("second").with_provider_data(data(foreign))),
    )
    .await;
    assert!(
        accepted.result.is_ok(),
        "input_provider_data: part-level and foreign keys must be accepted, got {}",
        describe(&accepted.result)
    );
}

/// Namespace [`assert_other_provider_data_gated`] puts foreign state under;
/// no adapter reads it.
pub const FOREIGN_PROBE_NAMESPACE: &str = "allama-probe-foreign";

/// Foreign state [`assert_other_provider_data_gated`] attaches; it must never
/// reach the wire.
pub const FOREIGN_PROBE_STATE: &str = "allama-probe-foreign-state";

/// Verbatim wire data of the probe content of
/// [`assert_other_provider_data_gated`].
pub const OTHER_PROBE_DATA: &str = "allama-probe-other-data";

/// Contract: provider data in a foreign namespace on [`OtherContent`] never
/// reaches the wire, on an assistant block or on a user part.
///
/// The probe content has origin `namespace` (the adapter's own), so the
/// adapter may replay its verbatim `data`; its provider data holds
/// [`FOREIGN_PROBE_STATE`] under [`FOREIGN_PROBE_NAMESPACE`]. The block and
/// the part are each sent in a request of their own. The adapter may replay
/// the content, drop it with a `Dropped` warning naming it, or refuse the
/// request; whatever it does, the foreign state stays off the wire. A
/// request that succeeds without the content on the wire must carry that
/// warning, so the content is never lost silently. Both requests go through
/// `probe` (`chat`) and `stream_probe` (`chat_stream`, see
/// [streaming probes](self#streaming-probes)). The bodies are searched as
/// bytes, so JSON and raw bodies ([`ProbeOutcome::raw`]) are checked alike.
///
/// # Panics
///
/// When the contract is violated.
pub async fn assert_other_provider_data_gated<F, Fut, G, GFut>(
    model: &str,
    namespace: &str,
    mut probe: F,
    mut stream_probe: G,
) where
    F: FnMut(ChatRequest) -> Fut,
    Fut: Future<Output = ProbeOutcome<ChatResponse>>,
    G: FnMut(ChatRequest) -> GFut,
    GFut: Future<Output = ProbeOutcome<ChatResponse>>,
{
    let other = || {
        let mut data = ExtraMap::new();
        data.insert("allama_probe".into(), Value::from(OTHER_PROBE_DATA));
        OtherContent::new("allama_probe", data)
            .with_origin(namespace)
            .with_provider_data(ProviderData::new().with_raw(
                FOREIGN_PROBE_NAMESPACE,
                "state",
                Value::from(FOREIGN_PROBE_STATE),
            ))
    };
    let cases = [
        (
            "part",
            ChatRequest::new(model).message(Message::user(vec![
                ContentPart::text("Read this."),
                ContentPart::Other(other()),
            ])),
            Subject::Content {
                message: 0,
                block: Some(1),
            },
        ),
        (
            "block",
            ChatRequest::new(model)
                .user("Hi.")
                .message(Message::assistant_blocks(vec![
                    ContentBlock::text("Hello."),
                    ContentBlock::Other(other()),
                ]))
                .user("Continue."),
            Subject::Content {
                message: 1,
                block: Some(1),
            },
        ),
    ];
    for (case, request, subject) in cases {
        for path in [CHAT, CHAT_STREAM] {
            let outcome = if path == CHAT {
                probe(request.clone()).await
            } else {
                stream_probe(request.clone()).await
            };
            assert!(
                !outcome.contains_text(FOREIGN_PROBE_STATE),
                "{path}: other_provider_data ({case}): foreign provider data reached the wire: \
                 {:?}",
                outcome.bodies()
            );
            if let Ok(response) = &outcome.result {
                let replayed = outcome.contains_text(OTHER_PROBE_DATA);
                let reported = response
                    .warnings
                    .iter()
                    .any(|w| w.kind == WarningKind::Dropped && w.subject == subject);
                assert!(
                    replayed || reported,
                    "{path}: other_provider_data ({case}): the content was neither sent nor \
                     reported as dropped: {:?}, warnings {:?}",
                    outcome.bodies(),
                    response.warnings
                );
            }
        }
    }
}

/// Key and value [`assert_app_data_never_sent`] stores as application data;
/// neither may reach the wire.
pub const APP_DATA_PROBE: &str = "allama-probe-app-data";

/// Contract: [application data](allama::history#application-data) never
/// reaches the wire.
///
/// A conversation with a system prompt and every message role (user text,
/// user parts, assistant text, and an assistant turn with a tool call and
/// its tool result) is sent twice: once with [`APP_DATA_PROBE`] stored as
/// application data on every prompt and message, and once without. The
/// adapter must send the same bodies and report the same outcome and
/// warnings both times, and [`APP_DATA_PROBE`] must appear in no body.
/// Adapters never read application data, so it can neither reach the wire
/// nor change what is sent. This is the [`Probe::AppData`] check of
/// [`assert_request_fields_accounted`], on its own, on `chat` (`probe`) and
/// `chat_stream` (`stream_probe`).
///
/// The conversation follows `provider`'s capability answers for `model`:
/// when it reports
/// [`ChatCapability::ToolCalls`](allama::ChatCapability::ToolCalls)
/// [`Unsupported`](allama::CapabilitySupport::Unsupported), the tool call
/// and its result are left out, since the adapter would refuse them, and
/// application data stays on the system prompt and the user and assistant
/// messages. A provider without tool support passes as long as it never
/// sends application data.
///
/// # Panics
///
/// When the request without application data fails, or the contract is
/// violated.
pub async fn assert_app_data_never_sent<F, Fut, G, GFut>(
    provider: &impl ChatProvider,
    model: &str,
    mut probe: F,
    mut stream_probe: G,
) where
    F: FnMut(ChatRequest) -> Fut,
    Fut: Future<Output = ProbeOutcome<ChatResponse>>,
    G: FnMut(ChatRequest) -> GFut,
    GFut: Future<Output = ProbeOutcome<ChatResponse>>,
{
    let supports = Supports::of(provider, model);
    let mut problems = Vec::new();
    for case in CASES.iter().filter(|case| case.probe == Probe::AppData) {
        problems.extend(on_path(
            CHAT,
            run_case(case, model, supports, &mut probe).await,
        ));
        problems.extend(on_path(
            CHAT_STREAM,
            run_case(case, model, supports, &mut stream_probe).await,
        ));
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// Assert that strict mode a provider cannot honour is never dropped
/// silently.
///
/// Run it against a provider configuration (endpoint, preset, or model)
/// that cannot honour strict mode. Two requests for `model` ask for it: one
/// with a strict tool and one with a strict JSON Schema response format,
/// both with a schema every strict dialect can express. Each is sent through
/// `probe`, which calls `chat`, and through `stream_probe`, which collects
/// `chat_stream` into a response (see [streaming probes](self#streaming-probes)).
/// On both paths, each request must either:
///
/// - fail with [`Error::Unsupported`] before anything is sent (the
///   configuration does not serve tools or JSON Schema output at all), or
/// - succeed with a warning about the field ([`RequestField::Tools`] or
///   [`RequestField::ResponseFormat`]). Strict mode on a tool or response
///   format is a hint (ADR 0003), so the same request is probed again
///   under [`UnsupportedPolicy::Error`], which must fail with
///   [`Error::Unsupported`] before anything is sent.
///
/// A refusal with stage
/// [`BeforeSending`](allama::UnsupportedStage::BeforeSending) claims
/// nothing was sent, which every refusal here must satisfy anyway;
/// extraction's `Auto` mode relies on it to retry a refused response format
/// in forced-tool mode.
///
/// Extraction depends on this report: its `SchemaEnforcement` is
/// `NotEnforced` exactly when an `Unsupported` warning about the field
/// arrives.
///
/// "Before anything is sent" is judged on the raw bodies, so a probe that
/// reports non-JSON bodies ([`ProbeOutcome::raw`]) is checked like a JSON
/// one.
///
/// ```no_run
/// use allama::{ChatProvider, ChatRequest, ChatResponse, ChatStreamExt};
/// use allama_conformance::probe::{ProbeOutcome, assert_strict_not_honoured_warns};
/// # struct Transport;
/// # impl Transport { fn bodies(&self) -> Vec<serde_json::Value> { Vec::new() } }
/// # fn provider_without_strict() -> (allama::mock::MockStreamingProvider, Transport) {
/// #     (allama::mock::MockStreamingProvider::with_text("ok"), Transport)
/// # }
///
/// async fn probe(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
///     let (provider, transport) = provider_without_strict();
///     let result = provider.chat(&request).await;
///     ProbeOutcome::new(result, transport.bodies())
/// }
///
/// async fn stream_probe(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
///     let (provider, transport) = provider_without_strict();
///     let result = match provider.chat_stream(&request).await {
///         Ok(stream) => stream.collect_response().await,
///         Err(err) => Err(err),
///     };
///     ProbeOutcome::new(result, transport.bodies())
/// }
///
/// # async fn example() {
/// assert_strict_not_honoured_warns("my-model", probe, stream_probe).await;
/// # }
/// ```
///
/// # Panics
///
/// When a strict request is served without a warning naming its field, or
/// is not refused before sending under [`UnsupportedPolicy::Error`].
pub async fn assert_strict_not_honoured_warns<F, Fut, G, GFut>(
    model: &str,
    mut probe: F,
    mut stream_probe: G,
) where
    F: FnMut(ChatRequest) -> Fut,
    Fut: Future<Output = ProbeOutcome<ChatResponse>>,
    G: FnMut(ChatRequest) -> GFut,
    GFut: Future<Output = ProbeOutcome<ChatResponse>>,
{
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": false
    });
    let base = || ChatRequest::new(model).user("What is the weather in Paris?");
    let cases = [
        (
            RequestField::Tools,
            base().tools([ToolDefinition::new("get_weather", schema.clone()).strict(true)]),
        ),
        (
            RequestField::ResponseFormat,
            base().response_format(ResponseFormat::json_schema(schema).name("weather")),
        ),
    ];
    let mut problems = Vec::new();
    for (field, request) in cases {
        for path in [CHAT, CHAT_STREAM] {
            let mut send = async |request: ChatRequest| {
                if path == CHAT {
                    probe(request).await
                } else {
                    stream_probe(request).await
                }
            };
            let outcome = send(request.clone()).await;
            let problem = match outcome.result {
                Err(Error::Unsupported { .. }) if outcome.nothing_sent() => {
                    misshapen_refusal(&outcome)
                }
                Err(Error::Unsupported { .. }) => {
                    Some("rejected with Unsupported only after sending".to_owned())
                }
                Err(err) => Some(format!("probe failed: {err}")),
                Ok(response) if names_field(&response.warnings, field) => {
                    let strict =
                        send(request.clone().unsupported_policy(UnsupportedPolicy::Error)).await;
                    match strict.result {
                        Err(Error::Unsupported { .. }) if strict.nothing_sent() => {
                            misshapen_refusal(&strict)
                        }
                        Err(Error::Unsupported { .. }) => Some(
                            "rejected under UnsupportedPolicy::Error only after sending".to_owned(),
                        ),
                        other => Some(format!(
                            "strict mode was degraded with a warning, but \
                             UnsupportedPolicy::Error did not fail with Unsupported: {}",
                            describe(&other)
                        )),
                    }
                }
                Ok(response) => Some(format!(
                    "strict mode was not honoured and no warning names the field (warnings: {:?})",
                    response.warnings
                )),
            };
            if let Some(problem) = problem {
                problems.push(format!("{field} ({path}): {problem}"));
            }
        }
    }
    assert!(
        problems.is_empty(),
        "strict_not_honoured failed:\n  {}",
        problems.join("\n  ")
    );
}

/// Contract: an adapter that cannot honour the query/document
/// [`input_type`](EmbeddingRequest::input_type) for `model` treats it as a
/// hint (ADR 0003).
///
/// For each [`EmbeddingInputType`]:
///
/// - under [`UnsupportedPolicy::Warn`] the request is served with a warning
///   on [`RequestField::InputType`], and what is sent is exactly what the
///   same request without the input type sends: the hint is dropped, not
///   sent in another form;
/// - under [`UnsupportedPolicy::Error`] it fails with [`Error::Unsupported`]
///   before anything is sent.
///
/// Run it for models whose
/// [`InputType`](allama::EmbeddingCapability::InputType) answer is
/// `Unsupported`, such as an OpenAI model not known to be symmetric. A
/// symmetric model (OpenAI `text-embedding-3-*`) answers `Supported` and
/// honors the hint by sending nothing.
///
/// # Panics
///
/// When the contract is violated.
pub async fn assert_input_type_not_honoured_warns<F, Fut>(model: &str, mut probe: F)
where
    F: FnMut(EmbeddingRequest) -> Fut,
    Fut: Future<Output = ProbeOutcome<EmbeddingResponse>>,
{
    let base = || EmbeddingRequest::new(model).input("allama-input-type-probe");
    let baseline = probe(base()).await;
    assert!(
        baseline.result.is_ok(),
        "input_type_not_honoured: the request without an input type failed: {}",
        describe(&baseline.result)
    );
    let mut problems = Vec::new();
    for input_type in [EmbeddingInputType::Query, EmbeddingInputType::Document] {
        let request = base().input_type(input_type);
        let lenient = probe(request.clone()).await;
        match &lenient.result {
            Ok(response) if !names_field(&response.warnings, RequestField::InputType) => {
                problems.push(format!(
                    "{input_type}: served without a warning on `input_type` (warnings: {:?})",
                    response.warnings
                ));
            }
            Ok(_) if lenient.differs(&baseline) => problems.push(format!(
                "{input_type}: the request sent differs from the one without an input type: \
                 {:?}",
                lenient.sent
            )),
            Ok(_) => {}
            Err(_) => problems.push(format!(
                "{input_type}: expected a warning under UnsupportedPolicy::Warn, got {}",
                describe(&lenient.result)
            )),
        }
        let strict = probe(request.unsupported_policy(UnsupportedPolicy::Error)).await;
        match &strict.result {
            Err(Error::Unsupported { .. }) if strict.nothing_sent() => {
                if let Some(problem) = misshapen_refusal(&strict) {
                    problems.push(format!("{input_type}: {problem}"));
                }
            }
            Err(Error::Unsupported { .. }) => problems.push(format!(
                "{input_type}: rejected under UnsupportedPolicy::Error only after sending"
            )),
            other => problems.push(format!(
                "{input_type}: expected Unsupported under UnsupportedPolicy::Error, got {}",
                describe(other)
            )),
        }
    }
    assert!(
        problems.is_empty(),
        "input_type_not_honoured failed:\n  {}",
        problems.join("\n  ")
    );
}

/// A refusal with stage
/// [`BeforeSending`](allama::UnsupportedStage::BeforeSending) (every
/// `Degradations` refusal) claims nothing was sent, so one that comes after
/// a request reached the transport is a problem.
pub(crate) fn refused_before_sending_after_sending<T>(outcome: &ProbeOutcome<T>) -> Option<String> {
    let err = outcome.result.as_ref().err()?;
    if err.unsupported_stage()? != allama::UnsupportedStage::BeforeSending || outcome.nothing_sent()
    {
        return None;
    }
    let subject = err
        .unsupported_subject()
        .map_or_else(|| "the request".to_owned(), ToString::to_string);
    Some(format!(
        "refused {subject} before sending (UnsupportedStage::BeforeSending) only after sending"
    ))
}

/// What is wrong with a refusal made in answer to a probe, or `None` when
/// the outcome is no [`Error::Unsupported`] or a well-formed one.
///
/// An adapter refuses what it cannot send before sending it, so a refusal
/// has stage [`BeforeSending`](allama::UnsupportedStage::BeforeSending)
/// (every refusal `Degradations` builds has it) and comes with nothing
/// sent. A refusal at an [unspecified](allama::UnsupportedStage::Unspecified)
/// stage ([`Error::unsupported`]) tells a caller nothing about what was
/// sent, so a router cannot fail over on it safely; one at the
/// [`Provider`](allama::UnsupportedStage::Provider) stage claims the
/// canned success response refused the request.
pub(crate) fn misshapen_refusal<T>(outcome: &ProbeOutcome<T>) -> Option<String> {
    let err = outcome.result.as_ref().err()?;
    let stage = err.unsupported_stage()?;
    if stage != allama::UnsupportedStage::BeforeSending {
        return Some(format!(
            "refused with stage {stage}, not before_sending: refuse what the adapter cannot \
             send through `Degradations`, which says nothing was sent ({err})"
        ));
    }
    refused_before_sending_after_sending(outcome)
}

/// What a refusal under [`UnsupportedPolicy::Error`] must not say: the hint
/// was refused, so nothing was ignored, dropped or sent without it.
pub(crate) fn claims_it_was_sent(err: &Error) -> Option<String> {
    const PHRASES: &[&str] = &[
        "was ignored",
        "were ignored",
        "was dropped",
        "were dropped",
        "was sent",
        "were sent",
        "sent as ",
        "replayed as",
    ];
    let text = err.to_string();
    let phrase = PHRASES.iter().find(|phrase| text.contains(*phrase))?;
    Some(format!(
        "the refusal says the hint {phrase:?}, but nothing was sent (\"{text}\")"
    ))
}

/// The label of problems found through a chat probe.
pub(crate) const CHAT: &str = "chat";

/// The label of problems found through a streaming probe.
pub(crate) const CHAT_STREAM: &str = "chat_stream";

/// `problem`, labelled with the path it was found on.
pub(crate) fn on_path(path: &str, problem: Option<String>) -> Option<String> {
    problem.map(|problem| format!("{path}: {problem}"))
}

/// Whether a warning is about `field`.
pub(crate) fn names_field(warnings: &[Warning], field: RequestField) -> bool {
    warnings
        .iter()
        .any(|w| matches!(w.subject, Subject::Field { field: f } if f == field))
}

pub(crate) fn describe<T>(result: &allama::Result<T>) -> String {
    match result {
        Ok(_) => "Ok".to_owned(),
        Err(err) => format!("{}: {err}", err.kind_name()),
    }
}

/// Problems collected by an audit, with known gaps tracked separately.
#[derive(Default)]
pub(crate) struct Report {
    problems: Vec<String>,
    stale: Vec<String>,
}

impl Report {
    pub(crate) fn record(
        &mut self,
        subject: impl std::fmt::Debug,
        problem: Option<String>,
        known_gap: bool,
    ) {
        match (problem, known_gap) {
            (Some(problem), false) => self.problems.push(problem),
            (None, true) => self.stale.push(format!(
                "{subject:?} is listed as a known gap but passes; remove it"
            )),
            (Some(_) | None, _) => {}
        }
    }

    /// Record a problem no single subject owns.
    pub(crate) fn fail(&mut self, problem: String) {
        self.problems.push(problem);
    }

    pub(crate) fn finish(mut self, audit: &str) {
        self.problems.append(&mut self.stale);
        assert!(
            self.problems.is_empty(),
            "{audit} failed:\n  {}",
            elide_payloads(&self.problems.join("\n  "))
        );
    }
}

async fn run_case<F, Fut>(
    case: &Case,
    model: &str,
    supports: Supports,
    probe: &mut F,
) -> Option<String>
where
    F: FnMut(ChatRequest) -> Fut,
    Fut: Future<Output = ProbeOutcome<ChatResponse>>,
{
    let name = case.name();
    let request = (case.request)(model, supports);
    let outcome = probe(request.clone()).await;
    if let Some(problem) = misshapen_refusal(&outcome) {
        return Some(format!("{name}: {problem}"));
    }

    if let (Rule::Withheld { .. } | Rule::Inert { .. }, Evidence::Value(value)) =
        (case.rule, case.evidence)
        && outcome.contains(&value)
    {
        return Some(format!("{name}: {value} reached the wire"));
    }
    if let Rule::Inert { without } = case.rule {
        let control = probe(without(model, supports)).await;
        return match (&control.result, &outcome.result) {
            (Err(err), _) => Some(format!("{name}: the request without it failed: {err}")),
            (Ok(_), Err(err)) => Some(format!("{name}: the request failed only with it: {err}")),
            (Ok(_), Ok(_)) if outcome.differs(&control) => {
                Some(format!("{name}: it changed the request bodies"))
            }
            (Ok(control), Ok(response)) if control.warnings != response.warnings => {
                Some(format!("{name}: it changed the warnings"))
            }
            (Ok(_), Ok(_)) => None,
        };
    }
    let response = match &outcome.result {
        Err(err @ Error::Unsupported { .. }) => {
            return match case.refusal() {
                Refusal::Allowed => None,
                Refusal::Unrelated(without) => match probe(without(model, supports)).await.result {
                    Err(Error::Unsupported { .. }) => None,
                    _ => Some(format!(
                        "{name}: refused, though the request without it is served: {err}"
                    )),
                },
                Refusal::Never => Some(format!("{name}: probe failed: {err}")),
            };
        }
        Err(err) => return Some(format!("{name}: probe failed: {err}")),
        Ok(response) => response,
    };
    if let Rule::Withheld { warn, .. } = case.rule {
        let dropped = response
            .warnings
            .iter()
            .any(|w| w.kind == WarningKind::Dropped && (case.names)(&w.subject));
        return (warn && !dropped).then(|| {
            format!(
                "{name}: dropped without a Dropped warning that names it (warnings: {:?})",
                response.warnings
            )
        });
    }

    let mut contrast = None;
    let (sent, what) = match case.evidence {
        Evidence::Value(value) => (
            outcome.contains(&value),
            format!("{value} is not on the wire"),
        ),
        Evidence::Change(build) => {
            let request = build(model);
            let contrasting = probe(request.clone()).await;
            let warnings = match &contrasting.result {
                Ok(response) => response.warnings.clone(),
                Err(err) => return Some(format!("{name}: the contrasting request failed: {err}")),
            };
            contrast = Some((request, warnings));
            (
                outcome.differs(&contrasting),
                "the request body does not change".to_owned(),
            )
        }
    };
    if sent {
        return None;
    }
    // A warning about either value accounts for it: an adapter may honor one
    // value by default and report only the other (contrast) value.
    let warned = [(request, response.warnings.clone())]
        .into_iter()
        .chain(contrast)
        .find(|(_, warnings)| warnings.iter().any(|w| (case.names)(&w.subject)));
    let Some((warned, warnings)) = warned else {
        return Some(format!(
            "{name}: {what} and no warning names it (warnings: {:?})",
            response.warnings
        ));
    };
    let named = |kind| {
        warnings
            .iter()
            .any(|w| (case.names)(&w.subject) && w.kind == kind)
    };
    // Honored approximately, or metadata dropped as state.
    if named(WarningKind::Approximated)
        || (matches!(case.rule, Rule::Droppable) && named(WarningKind::Dropped))
    {
        return None;
    }
    let strict = probe(warned.unsupported_policy(UnsupportedPolicy::Error)).await;
    match strict.result {
        Err(ref err @ Error::Unsupported { .. }) if strict.nothing_sent() => {
            misshapen_refusal(&strict)
                .or_else(|| claims_it_was_sent(err))
                .map(|problem| format!("{name}: {problem}"))
        }
        Err(Error::Unsupported { .. }) => Some(format!(
            "{name}: rejected under UnsupportedPolicy::Error only after sending"
        )),
        other => Some(format!(
            "{name}: dropped with a warning, but UnsupportedPolicy::Error \
             did not fail with Unsupported: {}",
            describe(&other)
        )),
    }
}

fn json_contains(body: &Value, value: &WireValue) -> bool {
    match (body, value) {
        // Tolerance absorbs f32 -> f64 widening (0.37f32 is 0.3700000047…):
        // allama's sampling fields are `f64`, but a third-party adapter's
        // wire struct may still hold `f32`.
        (Value::Number(n), WireValue::Number(expected)) => {
            n.as_f64().is_some_and(|n| (n - expected).abs() < 1e-6)
        }
        (Value::String(s), WireValue::Text(expected)) => s.contains(expected),
        (Value::Array(items), _) => items.iter().any(|item| json_contains(item, value)),
        (Value::Object(map), _) => map.values().any(|item| json_contains(item, value)),
        _ => false,
    }
}

/// Whether `value` appears in a body that is not JSON: as text, or as a
/// number in the common text and binary encodings (decimal, little-endian
/// `f32`/`f64`, and a protobuf varint for integers).
fn raw_contains(body: &[u8], value: &WireValue) -> bool {
    let has = |needle: &[u8]| body.windows(needle.len()).any(|window| window == needle);
    match *value {
        WireValue::Text(text) => has(text.as_bytes()),
        WireValue::Number(n) => {
            // Probe values are often `f32` fields, which a gateway may print
            // widened to `f64` (0.83f32 is 0.8299999833106995).
            let mut encodings = vec![
                n.to_string().into_bytes(),
                f64::from(n as f32).to_string().into_bytes(),
                (n as f32).to_le_bytes().to_vec(),
                n.to_le_bytes().to_vec(),
            ];
            if n.fract() == 0.0 && n >= 0.0 {
                encodings.push(varint(n as u64));
            }
            encodings.iter().any(|encoding| has(encoding))
        }
    }
}

fn varint(mut n: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let byte = (n & 0x7f) as u8;
        n >>= 7;
        if n == 0 {
            out.push(byte);
            return out;
        }
        out.push(byte | 0x80);
    }
}

/// The distinct [`BATCH_PROBE_PREFIX`] inputs in a body: each occurrence of
/// the prefix and the decimal index after it, in any wire format.
fn batch_probe_inputs(body: &[u8]) -> BTreeSet<Vec<u8>> {
    let prefix = BATCH_PROBE_PREFIX.as_bytes();
    let mut found = BTreeSet::new();
    let mut rest = body;
    while let Some(at) = rest
        .windows(prefix.len())
        .position(|window| window == prefix)
    {
        let after = &rest[at + prefix.len()..];
        let digits = after.iter().take_while(|b| b.is_ascii_digit()).count();
        if digits > 0 {
            found.insert(after[..digits].to_vec());
        }
        rest = &after[digits..];
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use allama::adapter::Degradations;
    use allama::mock::{MockEmbeddingProvider, MockProvider};
    use allama::{
        CapabilitySupport, ChatCapability, ContentBlock, ContentPart, EmbeddingProviderExt,
        Message, RequestField, Subject, ToolResultContent, UserContent, Warning,
    };

    /// An adapter over a mock that answers with `answer`, checking the
    /// response when `check` is set.
    async fn checked_embed(
        request: EmbeddingRequest,
        answer: Vec<Vec<f64>>,
        check: bool,
    ) -> allama::Result<EmbeddingResponse> {
        #[allow(clippy::cast_possible_truncation)]
        let vectors = answer
            .iter()
            .map(|vector| vector.iter().map(|&value| value as f32).collect())
            .collect();
        let response = MockEmbeddingProvider::new([Ok(EmbeddingResponse::new(vectors))])
            .embed(&request)
            .await?;
        if check {
            allama::adapter::embedding::check_response(&request, &response)
                .map_err(|err| err.with_provider("mock"))?;
        }
        Ok(response)
    }

    #[tokio::test]
    async fn an_adapter_that_checks_its_responses_passes() {
        assert_embedding_response_checked(
            &MockEmbeddingProvider::empty(),
            "m",
            async |request, answer| checked_embed(request, answer, true).await,
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "vectors of different sizes: expected UnexpectedResponse, got Ok")]
    async fn an_adapter_that_returns_ragged_vectors_fails() {
        assert_embedding_response_checked(
            &MockEmbeddingProvider::empty(),
            "m",
            async |request, answer| checked_embed(request, answer, false).await,
        )
        .await;
    }
    use serde_json::json;

    /// What the fake adapter gets wrong. `None` is an honest adapter.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Fault {
        None,
        /// Drops the seed as a hint, with a warning.
        SeedHint,
        SeedSilently,
        SeedWarningIgnoresPolicy,
        /// Drops file names with a `Dropped` warning.
        FileNameDropped,
        FileNameSilently,
        CacheSilently,
        ForeignReasoningSent,
        ForeignReasoningSilently,
        ForeignDataSent,
        /// Serves no tool calls: refuses tools, tool calls, and tool
        /// results with `Unsupported`, as preflight does. Honest.
        NoTools,
        /// Refuses a request holding another provider's `Other` part in a
        /// user or tool message, instead of dropping the part.
        ForeignPartRefused,
        /// Refuses another provider's `Other` part in a tool message only.
        ForeignToolPartRefused,
    }

    /// A fake adapter that sends the request as JSON, minus what it cannot
    /// honor. Its own namespace and origin is `fake`.
    fn adapter(
        fault: Fault,
    ) -> impl FnMut(ChatRequest) -> std::future::Ready<ProbeOutcome<ChatResponse>> {
        move |request: ChatRequest| {
            std::future::ready(match encode(request, fault) {
                Ok((body, warnings)) => {
                    let mut response = ChatResponse::new(vec![ContentBlock::text("ok")]);
                    response.warnings = warnings;
                    ProbeOutcome::new(Ok(response), vec![body])
                }
                Err(err) => ProbeOutcome::new(Err(err), Vec::new()),
            })
        }
    }

    /// [`adapter`], with a line-oriented text body (`path = value`), as a
    /// gateway with a non-JSON wire format would send.
    fn text_adapter(
        fault: Fault,
    ) -> impl FnMut(ChatRequest) -> std::future::Ready<ProbeOutcome<ChatResponse>> {
        let mut json = adapter(fault);
        move |request: ChatRequest| {
            let outcome = json(request).into_inner();
            let bodies: Vec<String> = outcome.sent.iter().map(to_lines).collect();
            std::future::ready(ProbeOutcome::raw(outcome.result, bodies))
        }
    }

    fn to_lines(body: &Value) -> String {
        fn walk(value: &Value, path: &str, out: &mut String) {
            match value {
                Value::Object(map) if map.is_empty() => out.push_str(&format!("{path} = {{}}\n")),
                Value::Object(map) => map
                    .iter()
                    .for_each(|(key, item)| walk(item, &format!("{path}/{key}"), out)),
                Value::Array(items) => items
                    .iter()
                    .enumerate()
                    .for_each(|(i, item)| walk(item, &format!("{path}/{i}"), out)),
                Value::String(s) => out.push_str(&format!("{path} = {s}\n")),
                other => out.push_str(&format!("{path} = {other}\n")),
            }
        }
        let mut out = String::new();
        walk(body, "", &mut out);
        out
    }

    fn encode(mut request: ChatRequest, fault: Fault) -> allama::Result<(Value, Vec<Warning>)> {
        if fault == Fault::NoTools && holds_tools(&request) {
            return Err(Degradations::new("fake", UnsupportedPolicy::Warn)
                .unsupported(RequestField::Tools, "fake: no tool calls"));
        }
        if let Some(message) = foreign_part(&request)
            && (fault == Fault::ForeignPartRefused
                || (fault == Fault::ForeignToolPartRefused && message.as_tool().is_some()))
        {
            return Err(Degradations::new("fake", UnsupportedPolicy::Warn).reject(
                Subject::Content {
                    message: 0,
                    block: None,
                },
                "fake: foreign content part",
            ));
        }
        let mut d = Degradations::new("fake", request.unsupported_policy.unwrap_or_default());
        let mut warnings = Vec::new();
        if request.seed.is_some() {
            match fault {
                Fault::SeedHint => d.unsupported_hint(RequestField::Seed, "no seed")?,
                Fault::SeedSilently => {}
                Fault::SeedWarningIgnoresPolicy => {
                    warnings.push(Warning::unsupported(RequestField::Seed, "no seed"));
                }
                _ => {}
            }
            if matches!(
                fault,
                Fault::SeedHint | Fault::SeedSilently | Fault::SeedWarningIgnoresPolicy
            ) {
                request.seed = None;
            }
        }
        for (index, message) in request.messages.iter_mut().enumerate() {
            let subject = |block| Subject::Content {
                message: index,
                block,
            };
            match message {
                Message::User(m) => {
                    if fault == Fault::CacheSilently {
                        m.cache = None;
                    }
                    if let UserContent::Parts(parts) = &mut m.content {
                        strip_file_names(parts, index, fault, &mut d);
                        withhold_parts(parts, index, &mut d);
                    }
                }
                Message::Tool(m) => {
                    if let ToolResultContent::Parts(parts) = &mut m.content {
                        strip_file_names(parts, index, fault, &mut d);
                        withhold_parts(parts, index, &mut d);
                    }
                }
                Message::Assistant(m) => {
                    let mut block = 0;
                    m.content.retain(|item| {
                        block += 1;
                        let foreign = match item {
                            ContentBlock::Reasoning(r) => {
                                if matches!(fault, Fault::ForeignReasoningSent) {
                                    return true;
                                }
                                r.provider_data.namespace("fake").is_none()
                            }
                            ContentBlock::Other(o) => o.origin.as_deref() != Some("fake"),
                            ContentBlock::ProviderToolCall(c) => {
                                c.origin.as_deref() != Some("fake")
                            }
                            ContentBlock::ProviderToolResult(r) => {
                                r.origin.as_deref() != Some("fake")
                            }
                            _ => false,
                        };
                        let silent = matches!(item, ContentBlock::Reasoning(_))
                            && fault == Fault::ForeignReasoningSilently;
                        if foreign && !silent {
                            d.dropped(subject(Some(block - 1)), "foreign state");
                        }
                        !foreign
                    });
                }
                _ => {}
            }
        }
        let mut body = serde_json::to_value(&request)?;
        if fault != Fault::ForeignDataSent {
            strip_foreign_data(&mut body);
        }
        warnings.splice(0..0, d.into_warnings());
        Ok((body, warnings))
    }

    /// Whether a request needs tool support: tools, tool calls, or tool
    /// results.
    fn holds_tools(request: &ChatRequest) -> bool {
        !request.tools.is_empty()
            || request.messages.iter().any(|message| match message {
                Message::Tool(_) => true,
                Message::Assistant(m) => m.tool_calls().next().is_some(),
                _ => false,
            })
    }

    /// The first user or tool message holding another provider's `Other`
    /// part.
    fn foreign_part(request: &ChatRequest) -> Option<&Message> {
        request.messages.iter().find(|message| {
            let parts = match message {
                Message::User(m) => match &m.content {
                    UserContent::Parts(parts) => parts.as_slice(),
                    _ => &[],
                },
                Message::Tool(m) => match &m.content {
                    ToolResultContent::Parts(parts) => parts.as_slice(),
                    _ => &[],
                },
                _ => &[],
            };
            parts.iter().any(
                |part| matches!(part, ContentPart::Other(o) if o.origin.as_deref() != Some("fake")),
            )
        })
    }

    /// A provider whose capability answers are all `Unknown`.
    fn provider() -> MockProvider {
        MockProvider::with_text("ok")
    }

    /// A provider that reports tool calls `Unsupported`.
    fn without_tools() -> MockProvider {
        provider().with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Unsupported)
    }

    fn strip_file_names(
        parts: &mut [ContentPart],
        message: usize,
        fault: Fault,
        d: &mut Degradations,
    ) {
        if !matches!(fault, Fault::FileNameDropped | Fault::FileNameSilently) {
            return;
        }
        for (block, part) in parts.iter_mut().enumerate() {
            if let ContentPart::File(file) = part
                && file.filename.take().is_some()
                && fault == Fault::FileNameDropped
            {
                d.dropped(
                    Subject::Content {
                        message,
                        block: Some(block),
                    },
                    "no file names",
                );
            }
        }
    }

    fn withhold_parts(parts: &mut Vec<ContentPart>, message: usize, d: &mut Degradations) {
        let mut part = 0;
        parts.retain(|item| {
            part += 1;
            let foreign =
                matches!(item, ContentPart::Other(o) if o.origin.as_deref() != Some("fake"));
            if foreign {
                d.dropped(
                    Subject::Content {
                        message,
                        block: Some(part - 1),
                    },
                    "foreign content",
                );
            }
            !foreign
        });
    }

    /// Removes provider data in other namespaces, and application data.
    fn strip_foreign_data(value: &mut Value) {
        match value {
            Value::Object(map) => {
                if let Some(Value::Object(data)) = map.get_mut("provider_data") {
                    data.retain(|namespace, _| namespace == "fake");
                }
                map.remove("app_data");
                map.values_mut().for_each(strip_foreign_data);
            }
            Value::Array(items) => items.iter_mut().for_each(strip_foreign_data),
            _ => {}
        }
    }

    #[tokio::test]
    async fn an_honest_adapter_passes() {
        assert_request_fields_accounted(
            &provider(),
            "m",
            &[],
            adapter(Fault::None),
            adapter(Fault::None),
        )
        .await;
        assert_request_fields_accounted(
            &provider(),
            "m",
            &[],
            adapter(Fault::SeedHint),
            adapter(Fault::SeedHint),
        )
        .await;
        assert_request_fields_accounted(
            &provider(),
            "m",
            &[],
            adapter(Fault::FileNameDropped),
            adapter(Fault::FileNameDropped),
        )
        .await;
    }

    /// A list from a newer `allama`, naming a probe this crate does not
    /// know beside the real list.
    fn newer_list() -> &'static str {
        let list = format!("{ACCOUNTED_FIELDS}\nfuture_field  probe: future_probe\n");
        Box::leak(list.into_boxed_str())
    }

    /// An unknown probe fails the audit with an upgrade report; it
    /// does not panic before the audit runs.
    #[tokio::test]
    #[should_panic(
        expected = "upgrade allama-conformance to the version of allama:\n    \
                               `future_field` names the probe(s) `future_probe`"
    )]
    async fn a_field_this_crate_cannot_check_fails_with_an_upgrade_report() {
        request_field_audit(
            newer_list(),
            &provider(),
            "m",
            &[],
            adapter(Fault::None),
            adapter(Fault::None),
        )
        .await;
    }

    /// The known probes still run beside an unknown one, so the
    /// report has their problems too.
    #[tokio::test]
    async fn the_known_probes_run_beside_an_unknown_one() {
        let provider = provider();
        let audit = request_field_audit(
            newer_list(),
            &provider,
            "m",
            &[],
            adapter(Fault::SeedSilently),
            adapter(Fault::SeedSilently),
        );
        let panic = futures_util::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(audit))
            .await
            .expect_err("the audit fails");
        let message = panic.downcast_ref::<String>().expect("a formatted message");
        assert!(
            message.contains("seed: 424242 is not on the wire and no warning names it"),
            "{message}"
        );
        assert!(message.contains("upgrade allama-conformance"), "{message}");
    }

    #[tokio::test]
    async fn an_adapter_without_tools_passes() {
        assert_request_fields_accounted(
            &without_tools(),
            "m",
            &[],
            adapter(Fault::NoTools),
            adapter(Fault::NoTools),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(
        expected = "chat_stream: seed: 424242 is not on the wire and no warning names it"
    )]
    async fn field_dropped_silently_only_when_streaming_fails() {
        assert_request_fields_accounted(
            &provider(),
            "m",
            &[],
            adapter(Fault::SeedHint),
            adapter(Fault::SeedSilently),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "chat: seed: 424242 is not on the wire and no warning names it")]
    async fn field_dropped_silently_only_by_chat_fails() {
        assert_request_fields_accounted(
            &provider(),
            "m",
            &[],
            adapter(Fault::SeedSilently),
            adapter(Fault::None),
        )
        .await;
    }

    #[tokio::test]
    async fn known_gap_on_one_path_is_skipped() {
        assert_request_fields_accounted(
            &provider(),
            "m",
            &[Probe::Seed],
            adapter(Fault::None),
            adapter(Fault::SeedSilently),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "seed: 424242 is not on the wire and no warning names it")]
    async fn silently_dropped_field_fails() {
        assert_request_fields_accounted(
            &provider(),
            "m",
            &[],
            adapter(Fault::SeedSilently),
            adapter(Fault::SeedSilently),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(
        expected = "seed: refused `seed` before sending (UnsupportedStage::BeforeSending) only \
                    after sending"
    )]
    async fn a_field_refusal_after_sending_fails() {
        // Refuses the seed through `Degradations`, but only after sending.
        let refuse_late = |request: ChatRequest| {
            let mut outcome = adapter(Fault::None)(request.clone()).into_inner();
            if request.seed.is_some() {
                outcome.result = Err(Degradations::new(
                    "fake",
                    request.unsupported_policy.unwrap_or_default(),
                )
                .unsupported(RequestField::Seed, "no seed"));
            }
            std::future::ready(outcome)
        };
        assert_request_fields_accounted(&provider(), "m", &[], refuse_late, refuse_late).await;
    }

    #[test]
    fn a_field_refusal_before_sending_and_an_unnamed_one_pass() {
        let refused = Degradations::new("fake", UnsupportedPolicy::Warn)
            .unsupported(RequestField::ResponseFormat, "no schema");
        let before = ProbeOutcome::<ChatResponse>::new(Err(refused), Vec::new());
        assert_eq!(refused_before_sending_after_sending(&before), None);
        let upstream = ProbeOutcome::<ChatResponse>::new(
            Err(Error::unsupported("HTTP 400")),
            vec![serde_json::json!({})],
        );
        assert_eq!(refused_before_sending_after_sending(&upstream), None);
    }

    #[tokio::test]
    #[should_panic(expected = "UnsupportedPolicy::Error did not fail")]
    async fn warned_field_must_fail_under_strict_policy() {
        assert_request_fields_accounted(
            &provider(),
            "m",
            &[],
            adapter(Fault::SeedWarningIgnoresPolicy),
            adapter(Fault::SeedWarningIgnoresPolicy),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(
        expected = "file_name (user message): \"allama-probe-file\" is not on the wire and no \
                    warning names it"
    )]
    async fn silently_dropped_file_name_fails() {
        assert_request_fields_accounted(
            &provider(),
            "m",
            &[],
            adapter(Fault::FileNameSilently),
            adapter(Fault::FileNameSilently),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(
        expected = "message_cache (user message): the request body does not change and no \
                    warning names it"
    )]
    async fn silently_dropped_cache_breakpoint_fails() {
        assert_request_fields_accounted(
            &provider(),
            "m",
            &[],
            adapter(Fault::CacheSilently),
            adapter(Fault::CacheSilently),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "foreign_reasoning: \"allama-probe-foreign\" reached the wire")]
    async fn sent_foreign_reasoning_fails() {
        assert_request_fields_accounted(
            &provider(),
            "m",
            &[],
            adapter(Fault::ForeignReasoningSent),
            adapter(Fault::ForeignReasoningSent),
        )
        .await;
    }

    /// Refusing a request over another provider's part would fail every
    /// failover to the adapter; only a refusal that has another cause (no
    /// tool calls on the path) is accepted.
    #[tokio::test]
    #[should_panic(expected = "foreign_other_content (user part): probe failed")]
    async fn a_refused_foreign_part_fails() {
        assert_request_fields_accounted(
            &provider(),
            "m",
            &[],
            adapter(Fault::ForeignPartRefused),
            adapter(Fault::ForeignPartRefused),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(
        expected = "foreign_other_content (tool result part): refused, though the request without it is served"
    )]
    async fn a_refused_foreign_tool_result_part_fails() {
        assert_request_fields_accounted(
            &provider(),
            "m",
            &[],
            adapter(Fault::ForeignToolPartRefused),
            adapter(Fault::ForeignToolPartRefused),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "foreign_reasoning: dropped without a Dropped warning")]
    async fn silently_dropped_foreign_reasoning_fails() {
        assert_request_fields_accounted(
            &provider(),
            "m",
            &[],
            adapter(Fault::ForeignReasoningSilently),
            adapter(Fault::ForeignReasoningSilently),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(
        expected = "foreign_provider_data (messages): \"allama-probe-foreign\" reached the wire"
    )]
    async fn sent_foreign_provider_data_fails() {
        assert_request_fields_accounted(
            &provider(),
            "m",
            &[],
            adapter(Fault::ForeignDataSent),
            adapter(Fault::ForeignDataSent),
        )
        .await;
    }

    #[tokio::test]
    async fn known_gap_is_skipped() {
        assert_request_fields_accounted(
            &provider(),
            "m",
            &[Probe::Seed],
            adapter(Fault::SeedSilently),
            adapter(Fault::SeedSilently),
        )
        .await;
        assert_request_fields_accounted(
            &provider(),
            "m",
            &[Probe::FileName],
            adapter(Fault::FileNameSilently),
            adapter(Fault::FileNameSilently),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "Seed is listed as a known gap but passes")]
    async fn stale_known_gap_fails() {
        assert_request_fields_accounted(
            &provider(),
            "m",
            &[Probe::Seed],
            adapter(Fault::SeedHint),
            adapter(Fault::SeedHint),
        )
        .await;
    }

    #[tokio::test]
    async fn non_json_bodies_run_the_audit() {
        assert_request_fields_accounted(
            &provider(),
            "m",
            &[],
            text_adapter(Fault::None),
            text_adapter(Fault::None),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "message_cache (user message): the request body does not change")]
    async fn non_json_bodies_catch_silent_drops() {
        assert_request_fields_accounted(
            &provider(),
            "m",
            &[],
            text_adapter(Fault::CacheSilently),
            text_adapter(Fault::CacheSilently),
        )
        .await;
    }

    #[test]
    fn raw_outcomes_parse_json_bodies() {
        let outcome = ProbeOutcome::<()>::raw(Ok(()), [br#"{"a": 1}"#.to_vec()]);
        assert_eq!(outcome.sent, [json!({"a": 1})]);
        let outcome = ProbeOutcome::<()>::raw(Ok(()), ["{\"a\": 1}\n{\"b\": 2}\n"]);
        assert!(outcome.sent.is_empty());
        assert_eq!(outcome.sent_raw.len(), 1);
    }

    #[test]
    fn raw_bodies_are_searched_in_text_and_binary_encodings() {
        let number = |n: f64| WireValue::Number(n);
        assert!(raw_contains(b"temperature: 0.37", &number(0.37)));
        assert!(raw_contains(&0.37f32.to_le_bytes(), &number(0.37)));
        assert!(raw_contains(&0.21f64.to_le_bytes(), &number(0.21)));
        // 1234 as a protobuf varint.
        assert!(raw_contains(&[0x08, 0xd2, 0x09], &number(1234.0)));
        assert!(raw_contains(
            b"\x0aallama-probe\x00",
            &WireValue::Text("allama-probe")
        ));
        assert!(!raw_contains(b"temperature: 0.5", &number(0.37)));
    }

    #[tokio::test]
    async fn other_content_replayed_without_foreign_data_passes() {
        assert_other_provider_data_gated("m", "fake", adapter(Fault::None), adapter(Fault::None))
            .await;
    }

    #[tokio::test]
    #[should_panic(expected = "foreign provider data reached the wire")]
    async fn foreign_data_on_other_content_fails() {
        assert_other_provider_data_gated(
            "m",
            "fake",
            adapter(Fault::ForeignDataSent),
            adapter(Fault::ForeignDataSent),
        )
        .await;
    }

    #[tokio::test]
    async fn raw_bodies_replaying_other_content_pass() {
        // Read as JSON only, the replayed content would be missing from the
        // wire and the contract would fail an honest adapter.
        assert_other_provider_data_gated(
            "m",
            "fake",
            text_adapter(Fault::None),
            text_adapter(Fault::None),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(
        expected = "chat: other_provider_data (part): foreign provider data reached \
                               the wire"
    )]
    async fn raw_bodies_with_foreign_data_on_other_content_fail() {
        assert_other_provider_data_gated(
            "m",
            "fake",
            text_adapter(Fault::ForeignDataSent),
            text_adapter(Fault::ForeignDataSent),
        )
        .await;
    }

    /// A fake adapter that sends the messages, serialized wholesale when
    /// `leak` is set or role and text only otherwise. Without `tools`, it
    /// refuses tool calls and tool results with `Unsupported`, as preflight
    /// does.
    fn message_adapter(
        leak: bool,
        tools: bool,
    ) -> impl FnMut(ChatRequest) -> std::future::Ready<ProbeOutcome<ChatResponse>> {
        move |request: ChatRequest| {
            if !tools && holds_tools(&request) {
                let refused = Err(Degradations::new("fake", UnsupportedPolicy::Warn)
                    .unsupported(RequestField::Tools, "fake: no tool calls"));
                return std::future::ready(ProbeOutcome::new(refused, Vec::new()));
            }
            let messages: Vec<Value> = request
                .messages
                .iter()
                .map(|message| {
                    if leak {
                        serde_json::to_value(message).unwrap()
                    } else {
                        json!({"role": message.role(), "text": message.to_string()})
                    }
                })
                .collect();
            let body = json!({"model": request.model, "messages": messages});
            let response = ChatResponse::new(vec![ContentBlock::text("ok")]);
            std::future::ready(ProbeOutcome::new(Ok(response), vec![body]))
        }
    }

    #[tokio::test]
    async fn app_data_off_the_wire_passes() {
        assert_app_data_never_sent(
            &provider(),
            "m",
            message_adapter(false, true),
            message_adapter(false, true),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "chat_stream: app_data: \"allama-probe-app-data\" reached the wire")]
    async fn app_data_on_the_wire_only_when_streaming_fails() {
        assert_app_data_never_sent(
            &provider(),
            "m",
            message_adapter(false, true),
            message_adapter(true, true),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "app_data: \"allama-probe-app-data\" reached the wire")]
    async fn app_data_on_the_wire_fails() {
        assert_app_data_never_sent(
            &provider(),
            "m",
            message_adapter(true, true),
            message_adapter(true, true),
        )
        .await;
    }

    #[tokio::test]
    async fn app_data_off_the_wire_without_tools_passes() {
        assert_app_data_never_sent(
            &without_tools(),
            "m",
            message_adapter(false, false),
            message_adapter(false, false),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "app_data: \"allama-probe-app-data\" reached the wire")]
    async fn app_data_on_the_wire_without_tools_fails() {
        assert_app_data_never_sent(
            &without_tools(),
            "m",
            message_adapter(true, false),
            message_adapter(true, false),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "chat: app_data: the request without it failed")]
    async fn tool_calls_stay_in_the_conversation_unless_reported_unsupported() {
        // Unknown support keeps the tool turn, which this adapter refuses.
        assert_app_data_never_sent(
            &provider(),
            "m",
            message_adapter(false, false),
            message_adapter(false, false),
        )
        .await;
    }

    #[test]
    fn the_conversation_follows_tool_support() {
        let with = fields::app_data_conversation("m", Supports { tool_calls: true }, true);
        let without = fields::app_data_conversation("m", Supports { tool_calls: false }, true);
        assert!(holds_tools(&with));
        assert!(!holds_tools(&without));
        assert!(without.system[0].app_data.contains_key(APP_DATA_PROBE));
        assert!(
            without
                .messages
                .iter()
                .all(|message| message.app_data().contains_key(APP_DATA_PROBE))
        );
        let roles: Vec<_> = without.messages.iter().map(Message::role).collect();
        assert_eq!(roles, ["user", "assistant", "user"]);
    }

    /// A fake indexed adapter: sorts by index and, unless `lenient`, checks
    /// that the indices are exactly `0..n`.
    fn indexed_adapter(
        lenient: bool,
    ) -> impl FnMut(EmbeddingRequest, Vec<usize>) -> std::future::Ready<allama::Result<EmbeddingResponse>>
    {
        move |request, answer| {
            let mut tagged: Vec<(usize, Vec<f32>)> =
                answer.into_iter().map(|i| (i, vec![i as f32])).collect();
            tagged.sort_by_key(|(index, _)| *index);
            let aligned = tagged.iter().enumerate().all(|(p, (i, _))| p == *i);
            let result = if !lenient && (!aligned || tagged.len() != request.inputs.len()) {
                Err(Error::unexpected_response("misaligned"))
            } else if tagged.len() != request.inputs.len() {
                Err(Error::unexpected_response("wrong count"))
            } else {
                Ok(EmbeddingResponse::new(
                    tagged.into_iter().map(|(_, v)| v).collect(),
                ))
            };
            std::future::ready(result)
        }
    }

    #[tokio::test]
    async fn aligned_indexed_adapter_passes() {
        assert_embedding_count("m", VectorOrder::Indexed, indexed_adapter(false)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "answer [0, 1, 1] for 3 inputs: expected UnexpectedResponse, got Ok")]
    async fn duplicate_indices_must_be_rejected() {
        assert_embedding_count("m", VectorOrder::Indexed, indexed_adapter(true)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "answer [0, 1] for 3 inputs: expected UnexpectedResponse, got Ok")]
    async fn a_short_response_must_be_rejected() {
        // Pads a short response, as a merge without a count check would.
        assert_embedding_count(
            "m",
            VectorOrder::InputOrder,
            |request: EmbeddingRequest, answer: Vec<usize>| {
                let mut vectors: Vec<Vec<f32>> = answer.iter().map(|&i| vec![i as f32]).collect();
                vectors.resize(request.inputs.len().max(vectors.len()), vec![2.0]);
                let result = if vectors.len() == request.inputs.len() {
                    Ok(EmbeddingResponse::new(vectors))
                } else {
                    Err(Error::unexpected_response("too many"))
                };
                std::future::ready(result)
            },
        )
        .await;
    }

    fn embedding_body(request: &EmbeddingRequest) -> Value {
        json!({"model": request.model, "input": request.inputs})
    }

    #[tokio::test]
    async fn preflight_rejection_passes() {
        let provider = MockEmbeddingProvider::with_vectors(vec![]).with_max_inputs_per_call(3);
        assert_batch_limit_enforced(&provider, "m", |request: EmbeddingRequest| async move {
            assert_eq!(request.inputs.len(), 4);
            ProbeOutcome::new(Err(Error::invalid_request("too many")), Vec::new())
        })
        .await;
    }

    #[tokio::test]
    async fn splitting_passes() {
        let provider = MockEmbeddingProvider::new([
            Ok(EmbeddingResponse::new(vec![vec![0.0]; 3])),
            Ok(EmbeddingResponse::new(vec![vec![0.0]; 1])),
        ])
        .with_max_inputs_per_call(3);
        assert_batch_limit_enforced(&provider, "m", |request: EmbeddingRequest| {
            let provider = provider.clone();
            async move {
                let result = provider.embed_chunked(&request, 3).await;
                let sent = provider.requests().iter().map(embedding_body).collect();
                ProbeOutcome::new(result, sent)
            }
        })
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "request 0 carried 4 inputs, over the limit of 3")]
    async fn oversized_request_fails() {
        let provider =
            MockEmbeddingProvider::with_vectors(vec![vec![0.0]; 4]).with_max_inputs_per_call(3);
        assert_batch_limit_enforced(&provider, "m", |request: EmbeddingRequest| {
            let provider = provider.clone();
            async move {
                let result = provider.embed(&request).await;
                let sent = provider.requests().iter().map(embedding_body).collect();
                ProbeOutcome::new(result, sent)
            }
        })
        .await;
    }

    /// [`embedding_body`] as lines of text, as a gateway with a non-JSON
    /// wire format would send.
    fn raw_embedding_body(request: &EmbeddingRequest) -> String {
        let inputs: Vec<&str> = request
            .inputs
            .iter()
            .filter_map(EmbeddingInput::as_text)
            .collect();
        format!("model {}\n{}\n", request.model, inputs.join("\n"))
    }

    #[tokio::test]
    async fn raw_splitting_passes() {
        let provider = MockEmbeddingProvider::new([
            Ok(EmbeddingResponse::new(vec![vec![0.0]; 3])),
            Ok(EmbeddingResponse::new(vec![vec![0.0]; 1])),
        ])
        .with_max_inputs_per_call(3);
        assert_batch_limit_enforced(&provider, "m", |request: EmbeddingRequest| {
            let provider = provider.clone();
            async move {
                let result = provider.embed_chunked(&request, 3).await;
                let sent: Vec<String> =
                    provider.requests().iter().map(raw_embedding_body).collect();
                ProbeOutcome::raw(result, sent)
            }
        })
        .await;
    }

    #[tokio::test]
    async fn raw_split_answered_with_an_error_passes() {
        // The second chunk fails: the limit was honoured even so.
        let provider = MockEmbeddingProvider::new([
            Ok(EmbeddingResponse::new(vec![vec![0.0]; 3])),
            Err(Error::provider("HTTP 500")),
        ])
        .with_max_inputs_per_call(3);
        assert_batch_limit_enforced(&provider, "m", |request: EmbeddingRequest| {
            let provider = provider.clone();
            async move {
                let result = provider.embed_chunked(&request, 3).await;
                let sent: Vec<String> =
                    provider.requests().iter().map(raw_embedding_body).collect();
                ProbeOutcome::raw(result, sent)
            }
        })
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "request 0 carried 4 inputs, over the limit of 3")]
    async fn raw_oversized_request_fails() {
        // Read as JSON only, the raw body would hold no inputs and pass.
        let provider =
            MockEmbeddingProvider::with_vectors(vec![vec![0.0]; 4]).with_max_inputs_per_call(3);
        assert_batch_limit_enforced(&provider, "m", |request: EmbeddingRequest| {
            let provider = provider.clone();
            async move {
                let result = provider.embed(&request).await;
                let sent: Vec<String> =
                    provider.requests().iter().map(raw_embedding_body).collect();
                ProbeOutcome::raw(result, sent)
            }
        })
        .await;
    }

    #[test]
    fn batch_inputs_are_counted_by_index() {
        let body = format!("{p}1 {p}10 {p}1 {p}2x {p}", p = BATCH_PROBE_PREFIX);
        let found: Vec<_> = batch_probe_inputs(body.as_bytes()).into_iter().collect();
        assert_eq!(found, [b"1".to_vec(), b"10".to_vec(), b"2".to_vec()]);
    }

    /// A fake adapter with no input-type field. `honest` runs preflight
    /// against an `Unsupported` answer; otherwise the hint is sent as
    /// `task` without a warning, or (`warn_and_send`) with one.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum InputTypeAdapter {
        Honest,
        Silent,
        WarnAndSend,
        WarnUnderError,
    }

    fn input_type_adapter(
        kind: InputTypeAdapter,
    ) -> impl FnMut(EmbeddingRequest) -> std::future::Ready<ProbeOutcome<EmbeddingResponse>> {
        move |request: EmbeddingRequest| {
            let provider = MockEmbeddingProvider::empty().with_embedding_capability(
                allama::EmbeddingCapability::InputType,
                allama::CapabilitySupport::Unsupported,
            );
            let mut d = Degradations::new("mock", request.unsupported_policy.unwrap_or_default());
            let mut sent = request.clone();
            let result = match kind {
                InputTypeAdapter::Honest => {
                    let mut checked = std::borrow::Cow::Borrowed(&request);
                    allama::adapter::embedding::preflight(&provider, &mut checked, &mut d)
                        .map(|()| sent = checked.into_owned())
                }
                InputTypeAdapter::Silent => Ok(()),
                InputTypeAdapter::WarnAndSend | InputTypeAdapter::WarnUnderError => {
                    if request.input_type.is_some() {
                        d.warn(Warning::unsupported(RequestField::InputType, "ignored"));
                    }
                    if kind == InputTypeAdapter::WarnUnderError {
                        sent.input_type = None;
                    }
                    Ok(())
                }
            };
            let outcome = match result {
                Ok(()) => {
                    let mut body = embedding_body(&sent);
                    if let Some(input_type) = sent.input_type {
                        body["task"] = json!(input_type.as_str());
                    }
                    let mut response = EmbeddingResponse::new(vec![vec![0.0]]);
                    response.warnings = d.into_warnings();
                    ProbeOutcome::new(Ok(response), vec![body])
                }
                Err(err) => ProbeOutcome::new(Err(err), Vec::new()),
            };
            std::future::ready(outcome)
        }
    }

    #[tokio::test]
    async fn input_type_dropped_with_a_warning_passes() {
        assert_input_type_not_honoured_warns("m", input_type_adapter(InputTypeAdapter::Honest))
            .await;
    }

    #[tokio::test]
    #[should_panic(expected = "query: served without a warning on `input_type`")]
    async fn input_type_ignored_silently_fails() {
        assert_input_type_not_honoured_warns("m", input_type_adapter(InputTypeAdapter::Silent))
            .await;
    }

    #[tokio::test]
    #[should_panic(expected = "query: the request sent differs from the one without an input type")]
    async fn input_type_warned_but_sent_fails() {
        assert_input_type_not_honoured_warns(
            "m",
            input_type_adapter(InputTypeAdapter::WarnAndSend),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "query: expected Unsupported under UnsupportedPolicy::Error, got Ok")]
    async fn input_type_warned_under_the_error_policy_fails() {
        assert_input_type_not_honoured_warns(
            "m",
            input_type_adapter(InputTypeAdapter::WarnUnderError),
        )
        .await;
    }
}
