//! The validation agreement contract: validation is a dry run of the call.
//!
//! [`assert_validation_agrees`] checks that
//! [`validate_chat_request`](ChatProvider::validate_chat_request) gives the
//! answer [`chat`](ChatProvider::chat) gives, and that
//! [`validate_chat_stream_request`](ChatProvider::validate_chat_stream_request)
//! gives the answer [`chat_stream`](ChatProvider::chat_stream) gives, for a
//! list of requests. [`drift_requests`] is the list of requests where the two
//! have drifted apart in practice; pass it, plus your own.
//! [`assert_embedding_validation_agrees`] and [`embedding_drift_requests`]
//! are the same contract for
//! [`validate_embedding_request`](EmbeddingProvider::validate_embedding_request)
//! and [`embed`](EmbeddingProvider::embed), and
//! [`assert_decision_validation_agrees`] and [`decision_drift_requests`]
//! for [`validate_decision_request`](DecisionProvider::validate_decision_request)
//! and [`decide`](DecisionProvider::decide), and
//! [`assert_extraction_validation_agrees`] and [`extraction_drift_requests`]
//! for [`validate_extraction_request`](ExtractionProvider::validate_extraction_request)
//! and [`extract_values`](ExtractionProvider::extract_values).

use std::future::Future;

use allama::decision::Question;
use allama::extraction::{DurationUnit, Field, ListItemKind};
use allama::{
    Capability, CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse,
    ChatValidation, DecisionCapability, DecisionProvider, DecisionRequest, DecisionResponse,
    DecisionValidation, EmbeddingCapability, EmbeddingInput, EmbeddingInputType, EmbeddingPart,
    EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, EmbeddingValidation,
    ExtractionCapability, ExtractionProvider, ExtractionRequest, ExtractionResponse,
    ExtractionValidation, Image, ImageDetail, ReasoningEffort, ResponseFormat, Result, ToolChoice,
    ToolDefinition, UnsupportedPolicy, Validation, ValidationCoverage, Warning,
};
use serde_json::json;

use crate::probe::ProbeOutcome;

/// Assert that `provider`'s validation agrees with its calls for every
/// request in `requests`, on both paths: `validate_chat_request` with
/// `probe` (which calls `chat`) and `validate_chat_stream_request` with
/// `stream_probe` (which collects `chat_stream`). The probes send through a
/// fresh provider configured as `provider` is, over a canned transport (see
/// [probes](crate::probe)).
///
/// What "agrees" means depends on the [`ValidationCoverage`] the validation
/// reports:
///
/// - **[`Encoding`](ValidationCoverage::Encoding)** (every first-party
///   adapter): both directions. Validation fails exactly when the call fails
///   before sending, with the same error kind and the same refused subject
///   ([`Error::unsupported_subject`](allama::Error::unsupported_subject));
///   when both succeed, the call's warnings
///   are the validation's warnings followed by those the canned response
///   itself causes (what a plain request's call reports beyond its
///   validation).
/// - **[`Capabilities`](ValidationCoverage::Capabilities)** (the default
///   validation): one direction. A validation error means the call fails
///   with the same kind and refused field, and the validation's warnings are
///   among the call's.
///
/// Both validations report the same coverage for a request both accept:
/// an adapter that overrides one and not the other fails.
///
/// On both, [`relied_on_unknown`](allama::Validation::relied_on_unknown)
/// lists only capabilities `provider` answers `Unknown` for the request's
/// model, and the same ones the core preflight records
/// ([`allama::adapter::chat::prepare_dry_run`]), plus `Streaming` and
/// `NativeStreaming` on the streaming path when they are `Unknown`.
///
/// Validation never checks credentials, so a provider that requires a
/// per-request credential is checked with a request that carries none: the
/// validation and the call both refuse it.
///
/// # Panics
///
/// On every disagreement, listed with the request and the path.
pub async fn assert_validation_agrees<P, F, Fut, G, GFut>(
    provider: &P,
    mut probe: F,
    mut stream_probe: G,
    requests: &[ChatRequest],
) where
    P: ChatProvider,
    F: FnMut(ChatRequest) -> Fut,
    Fut: Future<Output = ProbeOutcome<ChatResponse>>,
    G: FnMut(ChatRequest) -> GFut,
    GFut: Future<Output = ProbeOutcome<ChatResponse>>,
{
    let mut problems = Vec::new();
    for request in requests {
        let unary = provider.validate_chat_request(request);
        let base = baseline(provider, request, false, probe(plain(request)).await);
        let outcome = probe(request.clone()).await;
        problems.extend(
            agree(provider, request, false, &unary, &outcome, &base)
                .into_iter()
                .map(|problem| format!("validate_chat_request vs chat: {problem}")),
        );

        let stream = provider.validate_chat_stream_request(request);
        let base = baseline(provider, request, true, stream_probe(plain(request)).await);
        let outcome = stream_probe(request.clone()).await;
        problems.extend(
            agree(provider, request, true, &stream, &outcome, &base)
                .into_iter()
                .map(|problem| format!("validate_chat_stream_request vs chat_stream: {problem}")),
        );

        // An adapter that overrides one validation and not the other checks
        // one path less than it says.
        if let (Ok(unary), Ok(stream)) = (&unary, &stream)
            && unary.coverage() != stream.coverage()
        {
            problems.push(format!(
                "{}: validate_chat_request reports coverage {:?}, but \
                 validate_chat_stream_request reports {:?}; override both, or neither",
                summarize(request),
                unary.coverage(),
                stream.coverage()
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "validation disagrees with the call:\n  {}",
        problems.join("\n  ")
    );
}

/// Requests where validation and the call have drifted apart in practice,
/// for `model`. Pass them to [`assert_validation_agrees`] with the
/// requests specific to your adapter (provider options, such as manual
/// thinking, and a request for a model id you do not know).
///
/// - a hint under [`UnsupportedPolicy::Error`] (a seed) and under the
///   default policy (a seed, penalties, reasoning effort);
/// - more stop sequences than most APIs accept;
/// - a forced tool choice with reasoning;
/// - tools and a JSON Schema response format, which some endpoints refuse
///   only when streaming, and both strict, under each policy (an endpoint
///   without strict mode warns while encoding, or refuses under `Error`);
/// - a request relying on many capabilities at once (tools, parallel tool
///   calls, prompt caching, image input), whose
///   [`relied_on_unknown`](allama::Validation::relied_on_unknown) the
///   contract compares.
#[must_use]
pub fn drift_requests(model: &str) -> Vec<ChatRequest> {
    let base = || ChatRequest::new(model).user("Hi");
    let tool = || {
        ToolDefinition::new(
            "get_weather",
            json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        )
    };
    let schema = || {
        ResponseFormat::json_schema(json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": false
        }))
    };
    let strict = |policy| {
        base()
            .tools([tool().strict(true)])
            .response_format(schema())
            .unsupported_policy(policy)
    };
    vec![
        base(),
        base().seed(7).unsupported_policy(UnsupportedPolicy::Error),
        base()
            .seed(7)
            .frequency_penalty(0.5)
            .presence_penalty(0.5)
            .reasoning(ReasoningEffort::High),
        base().stop(["a", "b", "c", "d", "e", "f", "g", "h"]),
        base()
            .tools([tool()])
            .tool_choice(ToolChoice::Required)
            .reasoning(ReasoningEffort::High),
        base().tools([tool()]),
        base().response_format(schema()),
        strict(UnsupportedPolicy::Warn),
        strict(UnsupportedPolicy::Error),
        ChatRequest::new(model)
            .user(vec![
                allama::ContentPart::text("What is this?"),
                allama::ContentPart::image_url("https://example.com/cat.png"),
            ])
            .tools([tool()])
            .parallel_tool_calls(true)
            .cache(allama::PromptCache::new().key("drift")),
    ]
}

/// [`assert_validation_agrees`] for embeddings: `provider`'s
/// [`validate_embedding_request`](EmbeddingProvider::validate_embedding_request)
/// agrees with [`embed`](EmbeddingProvider::embed), which `probe` calls
/// through a fresh provider configured as `provider` is, over a canned
/// transport that answers with one vector per input.
///
/// Agreement means what it means for chat, by the
/// [`ValidationCoverage`] the validation reports; the plain request the
/// canned response is measured against embeds one text.
///
/// # Panics
///
/// On every disagreement, listed with the request.
pub async fn assert_embedding_validation_agrees<P, F, Fut>(
    provider: &P,
    mut probe: F,
    requests: &[EmbeddingRequest],
) where
    P: EmbeddingProvider,
    F: FnMut(EmbeddingRequest) -> Fut,
    Fut: Future<Output = ProbeOutcome<EmbeddingResponse>>,
{
    let mut problems = Vec::new();
    for request in requests {
        let plain = plain_embedding(request);
        let base = Baseline::of(
            provider.validate_embedding_request(&plain),
            &probe(plain).await.result,
            |response: &EmbeddingResponse| &response.warnings,
        );
        let validation = provider.validate_embedding_request(request);
        let outcome = probe(request.clone()).await;
        let summary = summarize_embedding(request);
        problems.extend(agree_on(
            &summary,
            &validation,
            &outcome.result,
            |response: &EmbeddingResponse| &response.warnings,
            outcome.sent_raw.is_empty(),
            &base,
        ));
        if let Ok(validation) = &validation {
            problems.extend(check_embedding_unknown(
                provider, request, validation, &summary,
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "embedding validation disagrees with the call:\n  {}",
        problems.join("\n  ")
    );
}

/// Embedding requests where validation and the call can drift apart, for
/// `model`. Pass them to [`assert_embedding_validation_agrees`] with the
/// requests specific to your adapter (provider options, a model id you do
/// not know).
///
/// - a batch, output dimensions, and both input types, under each policy;
/// - an image input, a text and an image combined into one input, and two
///   texts combined into one input;
/// - a request without inputs;
/// - a request relying on every capability at once, whose
///   [`relied_on_unknown`](allama::Validation::relied_on_unknown) the
///   contract compares.
#[must_use]
pub fn embedding_drift_requests(model: &str) -> Vec<EmbeddingRequest> {
    let base = || EmbeddingRequest::new(model).input("Hi");
    let image = || EmbeddingPart::image_url("https://example.com/cat.png");
    let mut requests = vec![
        base(),
        EmbeddingRequest::new(model).inputs(["first", "second"]),
        base().dimensions(64),
        EmbeddingRequest::new(model),
        EmbeddingRequest::new(model).input(EmbeddingInput::from(vec![image()])),
        EmbeddingRequest::new(model).input(vec![EmbeddingPart::text("A cat"), image()]),
        EmbeddingRequest::new(model)
            .input(vec![EmbeddingPart::text("a"), EmbeddingPart::text("b")]),
        EmbeddingRequest::new(model)
            .inputs(["first", "second"])
            .input(vec![EmbeddingPart::text("A cat"), image()])
            .dimensions(64)
            .input_type(EmbeddingInputType::Document),
    ];
    for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
        for input_type in [EmbeddingInputType::Query, EmbeddingInputType::Document] {
            requests.push(base().input_type(input_type).unsupported_policy(policy));
        }
    }
    requests
}

/// [`assert_validation_agrees`] for decisions: `provider`'s
/// [`validate_decision_request`](DecisionProvider::validate_decision_request)
/// agrees with [`decide`](DecisionProvider::decide), which `probe` calls
/// through a fresh provider configured as `provider` is, over a canned
/// transport that answers the request's questions.
///
/// Agreement means what it means for chat, by the [`ValidationCoverage`]
/// the validation reports; the plain request the canned response is
/// measured against asks one yes/no question.
///
/// # Panics
///
/// On every disagreement, listed with the request.
pub async fn assert_decision_validation_agrees<P, F, Fut>(
    provider: &P,
    mut probe: F,
    requests: &[DecisionRequest],
) where
    P: DecisionProvider,
    F: FnMut(DecisionRequest) -> Fut,
    Fut: Future<Output = ProbeOutcome<DecisionResponse>>,
{
    let mut problems = Vec::new();
    for request in requests {
        let plain = plain_decision(request);
        let base = Baseline::of(
            provider.validate_decision_request(&plain),
            &probe(plain).await.result,
            |response: &DecisionResponse| &response.warnings,
        );
        let validation = provider.validate_decision_request(request);
        let outcome = probe(request.clone()).await;
        let summary = summarize_decision(request);
        problems.extend(agree_on(
            &summary,
            &validation,
            &outcome.result,
            |response: &DecisionResponse| &response.warnings,
            outcome.sent_raw.is_empty(),
            &base,
        ));
        if let Ok(validation) = &validation {
            problems.extend(check_decision_unknown(
                provider, request, validation, &summary,
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "decision validation disagrees with the call:\n  {}",
        problems.join("\n  ")
    );
}

/// Decision requests where validation and the call can drift apart, for
/// `model`. Pass them to [`assert_decision_validation_agrees`] with the
/// requests specific to your adapter (provider options, a model id you do
/// not know).
///
/// - one question of each kind (yes/no, choice, score), and all three in
///   one request;
/// - a request without questions;
/// - more questions, choice options and score levels than most providers
///   accept, which [`decision_limits`](DecisionProvider::decision_limits)
///   must refuse before sending;
/// - images: one inline, one with a detail hint, one by URL, and more
///   than most providers accept.
#[must_use]
pub fn decision_drift_requests(model: &str) -> Vec<DecisionRequest> {
    let base = || DecisionRequest::new(model, "Our checkout has failed since 9am.");
    let yes_no = || Question::yes_no("Is this an outage?");
    let choice = || {
        Question::choice("Which team owns it?")
            .option("payments", "Payments and refunds")
            .option("platform", "Infrastructure")
            .option("other", "Anything else")
    };
    let score = || Question::score("How urgent is it?").levels(["low", "medium", "high"]);
    vec![
        base().question("outage", yes_no()),
        base().question("team", choice()),
        base().question("urgency", score()),
        base()
            .question("outage", yes_no())
            .question("team", choice())
            .question("urgency", score()),
        base(),
        base().questions((0..64).map(|i| (format!("q{i}"), yes_no()))),
        base().question(
            "team",
            Question::choice("Which team owns it?").options((0..64).map(|i| format!("team{i}"))),
        ),
        base().question(
            "urgency",
            Question::score("How urgent is it?").levels((0..32).map(|i| format!("level{i}"))),
        ),
        base().image(png()).question("outage", yes_no()),
        base()
            .image(png().with_detail(ImageDetail::High))
            .question("outage", yes_no()),
        base()
            .image(Image::url("https://example.com/dashboard.png"))
            .question("outage", yes_no()),
        (0..32)
            .fold(base(), |request, _| request.image(png()))
            .question("outage", yes_no()),
    ]
}

/// Assert that
/// [`validate_extraction_request`](ExtractionProvider::validate_extraction_request)
/// agrees with [`extract_values`](ExtractionProvider::extract_values) for
/// every request: [`assert_decision_validation_agrees`] for extraction
/// models. The plain request the canned response is measured against asks
/// one text field.
///
/// # Panics
///
/// On every disagreement, listed with the request.
pub async fn assert_extraction_validation_agrees<P, F, Fut>(
    provider: &P,
    mut probe: F,
    requests: &[ExtractionRequest],
) where
    P: ExtractionProvider,
    F: FnMut(ExtractionRequest) -> Fut,
    Fut: Future<Output = ProbeOutcome<ExtractionResponse>>,
{
    let mut problems = Vec::new();
    for request in requests {
        let plain = plain_extraction(request);
        let base = Baseline::of(
            provider.validate_extraction_request(&plain),
            &probe(plain).await.result,
            |response: &ExtractionResponse| &response.warnings,
        );
        let validation = provider.validate_extraction_request(request);
        let outcome = probe(request.clone()).await;
        let summary = summarize_extraction(request);
        problems.extend(agree_on(
            &summary,
            &validation,
            &outcome.result,
            |response: &ExtractionResponse| &response.warnings,
            outcome.sent_raw.is_empty(),
            &base,
        ));
        if let Ok(validation) = &validation {
            problems.extend(check_extraction_unknown(
                provider, request, validation, &summary,
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "extraction validation disagrees with the call:\n  {}",
        problems.join("\n  ")
    );
}

/// Extraction requests where validation and the call can drift apart, for
/// `model`. Pass them to [`assert_extraction_validation_agrees`] with the
/// requests specific to your adapter.
///
/// - one field of each kind, and every kind in one request;
/// - a request without fields, and one with an empty input;
/// - more fields and a longer forecast than most providers accept, which
///   [`extraction_limits`](ExtractionProvider::extraction_limits) must
///   refuse before sending;
/// - a reference date;
/// - images: one inline, one with a detail hint, one by URL, and more
///   than most providers accept.
#[must_use]
pub fn extraction_drift_requests(model: &str) -> Vec<ExtractionRequest> {
    let base = || {
        ExtractionRequest::new(
            model,
            "Order A-88412 arrived Sept 28. I paid $649.00 plus $52.10 tax.",
        )
    };
    let text = || Field::text().instructions("Order number");
    let all = || {
        base()
            .field("order_id", text())
            .field("total", Field::number().decimals(2).unit("USD"))
            .field("delivered", Field::date())
            .field("at", Field::time())
            .field("when", Field::datetime())
            .field("late_by", Field::duration(DurationUnit::Days).integer())
            .field("items", Field::list(ListItemKind::Text))
            .field(
                "repair",
                Field::estimate(10.0, 5000.0).log_scale().step(1.0),
            )
            .field("demand", Field::forecast(0.0, 1_000_000.0).horizon(3))
    };
    vec![
        base().field("order_id", text()),
        base().field("total", Field::number().decimals(2).unit("USD")),
        base().field("delivered", Field::date()),
        base().field("at", Field::time()),
        base().field("when", Field::datetime()),
        base().field("late_by", Field::duration(DurationUnit::Days).integer()),
        base().field("items", Field::list(ListItemKind::Text)),
        base().field(
            "repair",
            Field::estimate(10.0, 5000.0).log_scale().step(1.0),
        ),
        base().field("demand", Field::forecast(0.0, 1_000_000.0).horizon(3)),
        all(),
        base(),
        ExtractionRequest::new(model, "").field("order_id", text()),
        base().fields((0..256).map(|i| (format!("f{i}"), text()))),
        base().field("demand", Field::forecast(0.0, 1.0).horizon(256)),
        base()
            .field("delivered", Field::date())
            .reference_date("2026-10-09".parse().expect("a date")),
        base().image(png()).field("order_id", text()),
        base()
            .image(png().with_detail(ImageDetail::High))
            .field("order_id", text()),
        base()
            .image(Image::url("https://example.com/receipt.png"))
            .field("order_id", text()),
        (0..32)
            .fold(base(), |request, _| request.image(png()))
            .field("order_id", text()),
    ]
}

/// A one-pixel PNG.
fn png() -> Image {
    Image::base64(
        "image/png",
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    )
}

/// The plain request of the same model, whose call shows what the canned
/// response adds on its own.
fn plain(request: &ChatRequest) -> ChatRequest {
    let mut plain = ChatRequest::new(request.model.clone()).user("Hi");
    plain.options = request.options.clone();
    plain
}

/// The plain embedding request of the same model.
fn plain_embedding(request: &EmbeddingRequest) -> EmbeddingRequest {
    let mut plain = EmbeddingRequest::new(request.model.clone()).input("Hi");
    plain.options = request.options.clone();
    plain
}

/// The plain decision request of the same model.
fn plain_decision(request: &DecisionRequest) -> DecisionRequest {
    let mut plain = DecisionRequest::new(request.model.clone(), "Hi")
        .question("plain", Question::yes_no("Is this a greeting?"));
    plain.options = request.options.clone();
    plain
}

/// The plain extraction request of the same model.
fn plain_extraction(request: &ExtractionRequest) -> ExtractionRequest {
    let mut plain = ExtractionRequest::new(request.model.clone(), "Hi")
        .field("plain", Field::text().instructions("The greeting"));
    plain.options = request.options.clone();
    plain
}

/// What the call reports for a plain request beyond its validation: the
/// warnings the canned response causes, and the coverage the provider
/// reports.
struct Baseline {
    response_warnings: Vec<Warning>,
    coverage: Option<ValidationCoverage>,
}

impl Baseline {
    /// The baseline from the validation of the plain request and its call.
    fn of<C: Capability, T>(
        validation: Result<Validation<C>>,
        result: &Result<T>,
        warnings: impl Fn(&T) -> &Vec<Warning>,
    ) -> Self {
        let response_warnings = match (&validation, result) {
            (Ok(validation), Ok(response))
                if warnings(response).starts_with(validation.warnings()) =>
            {
                warnings(response)[validation.warnings().len()..].to_vec()
            }
            _ => Vec::new(),
        };
        Self {
            response_warnings,
            coverage: validation.ok().map(|validation| validation.coverage()),
        }
    }
}

fn baseline<P: ChatProvider>(
    provider: &P,
    request: &ChatRequest,
    stream: bool,
    outcome: ProbeOutcome<ChatResponse>,
) -> Baseline {
    let plain = plain(request);
    let validation = if stream {
        provider.validate_chat_stream_request(&plain)
    } else {
        provider.validate_chat_request(&plain)
    };
    Baseline::of(validation, &outcome.result, |response: &ChatResponse| {
        &response.warnings
    })
}

/// The disagreements between a validation and its call.
fn agree<P: ChatProvider>(
    provider: &P,
    request: &ChatRequest,
    stream: bool,
    validation: &Result<ChatValidation>,
    outcome: &ProbeOutcome<ChatResponse>,
    base: &Baseline,
) -> Vec<String> {
    let summary = summarize(request);
    let mut problems = agree_on(
        &summary,
        validation,
        &outcome.result,
        |response: &ChatResponse| &response.warnings,
        outcome.sent_raw.is_empty(),
        base,
    );
    if let Ok(validation) = validation {
        problems.extend(check_unknown(
            provider, request, stream, validation, &summary,
        ));
    }
    problems
}

/// The disagreements between a validation and the result of its call,
/// whatever the operation.
fn agree_on<C: Capability, T>(
    summary: &str,
    validation: &Result<Validation<C>>,
    result: &Result<T>,
    warnings: impl Fn(&T) -> &Vec<Warning>,
    nothing_sent: bool,
    base: &Baseline,
) -> Vec<String> {
    let mut problems = Vec::new();
    let coverage = match validation {
        Ok(validation) => validation.coverage(),
        Err(_) => base.coverage.unwrap_or(ValidationCoverage::Encoding),
    };
    match (validation, result) {
        (Err(refusal), Err(error)) => {
            if refusal.kind_name() != error.kind_name()
                || refusal.unsupported_subject() != error.unsupported_subject()
            {
                problems.push(format!(
                    "{summary}: validation refuses with {} ({:?}), the call with {} ({:?})",
                    refusal.kind_name(),
                    refusal.unsupported_subject(),
                    error.kind_name(),
                    error.unsupported_subject(),
                ));
            } else if coverage == ValidationCoverage::Encoding && !nothing_sent {
                problems.push(format!(
                    "{summary}: validation refuses ({refusal}), but the call sent the request \
                     before failing ({error})"
                ));
            }
        }
        (Err(refusal), Ok(_)) => problems.push(format!(
            "{summary}: validation refuses ({refusal}), but the call succeeds"
        )),
        (Ok(_), Err(error)) if coverage == ValidationCoverage::Encoding && nothing_sent => {
            problems.push(format!(
                "{summary}: validation accepts, but the call refuses before sending ({error})"
            ));
        }
        (Ok(_), Err(_)) => {}
        (Ok(validation), Ok(response)) => {
            let reported = warnings(response);
            match coverage {
                ValidationCoverage::Capabilities => {
                    if let Some(missing) = validation
                        .warnings()
                        .iter()
                        .find(|warning| !reported.contains(warning))
                    {
                        problems.push(format!(
                            "{summary}: validation reports {missing:?}, the call does not"
                        ));
                    }
                }
                _ => {
                    let mut expected = validation.warnings().to_vec();
                    expected.extend(base.response_warnings.iter().cloned());
                    if *reported != expected {
                        problems.push(format!(
                            "{summary}: the call reports {reported:?}, validation {:?}",
                            validation.warnings()
                        ));
                    }
                }
            }
        }
    }
    problems
}

/// The capabilities a validation relies on are the provider's `Unknown`
/// answers, as the core preflight records them.
fn check_unknown<P: ChatProvider>(
    provider: &P,
    request: &ChatRequest,
    stream: bool,
    validation: &ChatValidation,
    summary: &str,
) -> Vec<String> {
    let mut problems = Vec::new();
    for (capability, subject) in validation.relied_on_unknown() {
        let answer = provider.chat_capability(&request.model, *capability);
        if answer != CapabilitySupport::Unknown {
            problems.push(format!(
                "{summary}: relies on {capability:?} ({subject:?}) as unknown, but the provider \
                 answers {answer:?}"
            ));
        }
    }
    let reference = match allama::adapter::chat::prepare_dry_run(provider, request) {
        Ok(mut prepared) => {
            if stream {
                prepared.preflight_stream(provider);
            }
            prepared.into_validation(ValidationCoverage::Capabilities)
        }
        Err(_) => return problems,
    };
    let capabilities = |validation: &ChatValidation| -> Vec<ChatCapability> {
        validation
            .relied_on_unknown()
            .iter()
            .map(|(capability, _)| *capability)
            .collect()
    };
    if capabilities(validation) != capabilities(&reference) {
        problems.push(format!(
            "{summary}: relies on {:?} as unknown, the preflight records {:?}",
            validation.relied_on_unknown(),
            reference.relied_on_unknown()
        ));
    }
    problems
}

/// [`check_unknown`] for an embedding validation.
fn check_embedding_unknown<P: EmbeddingProvider>(
    provider: &P,
    request: &EmbeddingRequest,
    validation: &EmbeddingValidation,
    summary: &str,
) -> Vec<String> {
    let mut problems = Vec::new();
    for (capability, subject) in validation.relied_on_unknown() {
        let answer = provider.embedding_capability(&request.model, *capability);
        if answer != CapabilitySupport::Unknown {
            problems.push(format!(
                "{summary}: relies on {capability:?} ({subject:?}) as unknown, but the provider \
                 answers {answer:?}"
            ));
        }
    }
    let Ok(reference) = allama::adapter::embedding::prepare_dry_run(provider, request) else {
        return problems;
    };
    let reference = reference.into_validation(ValidationCoverage::Capabilities);
    let capabilities = |validation: &EmbeddingValidation| -> Vec<EmbeddingCapability> {
        validation
            .relied_on_unknown()
            .iter()
            .map(|(capability, _)| *capability)
            .collect()
    };
    if capabilities(validation) != capabilities(&reference) {
        problems.push(format!(
            "{summary}: relies on {:?} as unknown, the preflight records {:?}",
            validation.relied_on_unknown(),
            reference.relied_on_unknown()
        ));
    }
    problems
}

/// [`check_unknown`] for a decision validation.
fn check_decision_unknown<P: DecisionProvider>(
    provider: &P,
    request: &DecisionRequest,
    validation: &DecisionValidation,
    summary: &str,
) -> Vec<String> {
    let mut problems = Vec::new();
    for (capability, subject) in validation.relied_on_unknown() {
        let answer = provider.decision_capability(&request.model, *capability);
        if answer != CapabilitySupport::Unknown {
            problems.push(format!(
                "{summary}: relies on {capability:?} ({subject:?}) as unknown, but the provider \
                 answers {answer:?}"
            ));
        }
    }
    let Ok(reference) = allama::adapter::decision::prepare_dry_run(provider, request) else {
        return problems;
    };
    let reference = reference.into_validation(ValidationCoverage::Capabilities);
    let capabilities = |validation: &DecisionValidation| -> Vec<DecisionCapability> {
        validation
            .relied_on_unknown()
            .iter()
            .map(|(capability, _)| *capability)
            .collect()
    };
    if capabilities(validation) != capabilities(&reference) {
        problems.push(format!(
            "{summary}: relies on {:?} as unknown, the preflight records {:?}",
            validation.relied_on_unknown(),
            reference.relied_on_unknown()
        ));
    }
    problems
}

/// [`check_unknown`] for an extraction validation.
fn check_extraction_unknown<P: ExtractionProvider>(
    provider: &P,
    request: &ExtractionRequest,
    validation: &ExtractionValidation,
    summary: &str,
) -> Vec<String> {
    let mut problems = Vec::new();
    for (capability, subject) in validation.relied_on_unknown() {
        let answer = provider.extraction_capability(&request.model, *capability);
        if answer != CapabilitySupport::Unknown {
            problems.push(format!(
                "{summary}: relies on {capability:?} ({subject:?}) as unknown, but the provider \
                 answers {answer:?}"
            ));
        }
    }
    let Ok(reference) = allama::adapter::extraction::prepare_dry_run(provider, request) else {
        return problems;
    };
    let reference = reference.into_validation(ValidationCoverage::Capabilities);
    let capabilities = |validation: &ExtractionValidation| -> Vec<ExtractionCapability> {
        validation
            .relied_on_unknown()
            .iter()
            .map(|(capability, _)| *capability)
            .collect()
    };
    if capabilities(validation) != capabilities(&reference) {
        problems.push(format!(
            "{summary}: relies on {:?} as unknown, the preflight records {:?}",
            validation.relied_on_unknown(),
            reference.relied_on_unknown()
        ));
    }
    problems
}

/// A short description of an extraction request for problem reports.
fn summarize_extraction(request: &ExtractionRequest) -> String {
    let mut value = serde_json::to_value(request).unwrap_or_default();
    if let Some(object) = value.as_object_mut() {
        object.remove("input");
    }
    format!("request {value}")
}

/// A short description of a decision request for problem reports.
fn summarize_decision(request: &DecisionRequest) -> String {
    let value = serde_json::to_value(request).unwrap_or_default();
    format!("request {value}")
}

/// A short description of an embedding request for problem reports.
fn summarize_embedding(request: &EmbeddingRequest) -> String {
    let value = serde_json::to_value(request).unwrap_or_default();
    format!("request {value}")
}

/// A short description of a request for problem reports.
fn summarize(request: &ChatRequest) -> String {
    let mut value = serde_json::to_value(request).unwrap_or_default();
    if let Some(object) = value.as_object_mut() {
        object.remove("messages");
    }
    format!("request {value}")
}

#[cfg(test)]
mod tests {
    use allama::mock::MockProvider;
    use allama::{CapabilitySupport, ChatCapability};

    use super::*;

    /// A mock answers with "ok", like a probe's canned response.
    fn probe_of(provider: MockProvider) -> impl Fn(ChatRequest) -> ProbeFuture {
        move |request| {
            let provider = provider.clone();
            Box::pin(async move {
                provider.push_response(Ok(ChatResponse::new(vec![allama::ContentBlock::text(
                    "ok",
                )])));
                let calls = provider.call_count();
                let result = provider.chat(&request).await;
                let sent = if provider.call_count() > calls {
                    vec![serde_json::to_value(&request).unwrap()]
                } else {
                    Vec::new()
                };
                ProbeOutcome::new(result, sent)
            })
        }
    }

    type ProbeFuture = std::pin::Pin<Box<dyn Future<Output = ProbeOutcome<ChatResponse>> + Send>>;

    fn no_reasoning() -> MockProvider {
        MockProvider::empty().with_chat_capability(
            ChatCapability::ReasoningConfig,
            CapabilitySupport::Unsupported,
        )
    }

    #[tokio::test]
    async fn the_default_validation_agrees_with_a_mock() {
        let provider = no_reasoning();
        let requests = drift_requests("m");
        assert_validation_agrees(
            &provider,
            probe_of(provider.clone()),
            probe_of(provider.clone()),
            &requests,
        )
        .await;
    }

    #[test]
    #[should_panic(expected = "validation accepts, but the call refuses before sending")]
    fn a_validation_that_misses_a_refusal_fails() {
        // The call refuses reasoning; the validation knows nothing of it.
        let caller = no_reasoning();
        let validator = MockProvider::empty();
        let request = ChatRequest::new("m")
            .user("Hi")
            .reasoning(ReasoningEffort::High)
            .unsupported_policy(UnsupportedPolicy::Error);
        // Claims encoding coverage, so the check is two-way.
        struct Encoding(MockProvider);
        impl allama::ProviderIdentity for Encoding {
            fn provider_name(&self) -> &str {
                "mock"
            }

            fn reads_namespace(&self, namespace: &str) -> bool {
                namespace == "mock"
            }
        }
        impl ChatProvider for Encoding {
            type Stream = allama::SingleResponseStream;
            async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
                self.0.chat(request).await
            }
            async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
                self.0.chat_stream(request).await
            }
            fn chat_capability(
                &self,
                model: &str,
                capability: ChatCapability,
            ) -> CapabilitySupport {
                self.0.chat_capability(model, capability)
            }
            fn validate_chat_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
                Ok(self
                    .0
                    .validate_chat_request(request)?
                    .with_coverage(ValidationCoverage::Encoding))
            }
            fn validate_chat_stream_request(
                &self,
                request: &ChatRequest,
            ) -> Result<ChatValidation> {
                self.validate_chat_request(request)
            }
        }
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(assert_validation_agrees(
                &Encoding(validator),
                probe_of(caller.clone()),
                probe_of(caller),
                &[request],
            ));
    }

    /// An adapter that overrides `validate_chat_request` alone: the
    /// streaming validation is the default's, which checks capabilities
    /// only.
    struct UnaryOnly(MockProvider);

    impl allama::ProviderIdentity for UnaryOnly {
        fn provider_name(&self) -> &str {
            "mock"
        }

        fn reads_namespace(&self, namespace: &str) -> bool {
            namespace == "mock"
        }
    }

    impl ChatProvider for UnaryOnly {
        type Stream = allama::SingleResponseStream;
        async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
            self.0.chat(request).await
        }
        async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
            self.0.chat_stream(request).await
        }
        fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
            self.0.chat_capability(model, capability)
        }
        fn validate_chat_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
            Ok(self
                .0
                .validate_chat_request(request)?
                .with_coverage(ValidationCoverage::Encoding))
        }
    }

    #[tokio::test]
    #[should_panic(expected = "validate_chat_request reports coverage Encoding, but \
                               validate_chat_stream_request reports Capabilities")]
    async fn a_validation_overridden_on_one_path_only_fails() {
        let provider = MockProvider::empty();
        assert_validation_agrees(
            &UnaryOnly(provider.clone()),
            probe_of(provider.clone()),
            probe_of(provider),
            &[ChatRequest::new("m").user("Hi")],
        )
        .await;
    }
}
