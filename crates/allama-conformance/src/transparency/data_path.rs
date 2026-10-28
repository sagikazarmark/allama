//! The data path of a transparent wrapper: what the wrapped provider
//! answers reaches the caller unchanged.
//!
//! [`TransparencyProbe`] answers each call by the scenario its request
//! names (the last user message for chat, the first input for embeddings,
//! the first question for decisions): a response full of data, a stream of
//! every kind of event, or an error of each kind, before or in the middle of
//! a stream. The checks here send each scenario through the wrapper and
//! compare what comes out with what the probe sent.

use std::time::Duration;

use allama::adapter::Degradations;
use allama::chat::stream::{MetadataBuilder, StreamBlockType};
use allama::decision::{Answer, ChoiceAnswer, Question, ScoreAnswer, YesNoAnswer};
use allama::extraction::{self, Field};
use allama::{
    ChatProvider, ChatRequest, ChatResponse, ChatStream, ContentBlock, DecisionProvider,
    DecisionRequest, DecisionResponse, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse,
    Error, ExtractionProvider, ExtractionRequest, ExtractionResponse, FinishReason, Message,
    MessageOrigin, ProviderData, RequestField, StreamErrorKind, StreamEvent, Subject,
    UnsupportedPolicy, UnsupportedStage, Usage, Warning, WarningCause, WarningKind,
};
use futures_util::StreamExt;
use serde_json::{Value, json};

use super::TransparencyProbe;
use crate::diagnostics::elide_payloads;

/// The prefix of the text that names a scenario.
const SCENARIO: &str = "allama-data-path:";

/// What the probe answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Scenario {
    /// The canned response, or the canned stream.
    Ok,
    /// The error at this index of [`errors`], from the call itself (and from
    /// `chat_stream` before its stream).
    Error(usize),
    /// The canned stream's first events, then the error at this index of
    /// [`errors`] (`chat` fails with it as well).
    StreamError(usize),
}

impl Scenario {
    /// The text that names the scenario.
    fn text(self) -> String {
        match self {
            Self::Ok => format!("{SCENARIO}ok"),
            Self::Error(index) => format!("{SCENARIO}error:{index}"),
            Self::StreamError(index) => format!("{SCENARIO}stream-error:{index}"),
        }
    }

    /// The scenario `text` names, or [`Ok`](Self::Ok) for any other text.
    fn parse(text: Option<&str>) -> Self {
        let Some(rest) = text.and_then(|text| text.strip_prefix(SCENARIO)) else {
            return Self::Ok;
        };
        let index = |value: &str| value.parse().unwrap_or(0);
        if let Some(value) = rest.strip_prefix("stream-error:") {
            Self::StreamError(index(value))
        } else if let Some(value) = rest.strip_prefix("error:") {
            Self::Error(index(value))
        } else {
            Self::Ok
        }
    }

    /// The scenario a chat request names.
    pub(super) fn of_chat(request: &ChatRequest) -> Self {
        Self::parse(request.messages.last().and_then(Message::text).as_deref())
    }

    /// The scenario an embedding request names.
    pub(super) fn of_embedding(request: &EmbeddingRequest) -> Self {
        Self::parse(request.inputs.first().and_then(|input| input.as_text()))
    }

    /// The scenario a decision request names.
    pub(super) fn of_decision(request: &DecisionRequest) -> Self {
        Self::parse(request.questions.first().map(|(name, _)| name.as_str()))
    }

    /// The scenario an extraction request names.
    pub(super) fn of_extraction(request: &ExtractionRequest) -> Self {
        Self::parse(request.fields.first().map(|(name, _)| name.as_str()))
    }
}

/// The probe's provider data.
fn provider_data() -> ProviderData {
    ProviderData::new().with_raw(
        TransparencyProbe::PROVIDER_NAME,
        "trace",
        json!({"route": "probe", "attempt": 1}),
    )
}

/// The probe's usage.
fn usage() -> Usage {
    Usage::new()
        .input_tokens(21)
        .output_tokens(13)
        .cache_read_input_tokens(5)
        .total_tokens(34)
}

/// The warnings the probe reports, in order: one with a code, one with a
/// cause, one naming content of the caller's request.
fn warnings() -> Vec<Warning> {
    vec![
        Warning::new(
            WarningKind::Approximated,
            Subject::Field {
                field: RequestField::Temperature,
            },
            "the transparency probe rounded the temperature",
        )
        .with_code("approximated.temperature"),
        Warning::new(
            WarningKind::Other,
            Subject::Response,
            "the transparency probe reports a cause",
        )
        .with_cause(WarningCause::new("rate_limited")),
        Warning::new(
            WarningKind::Dropped,
            Subject::Content {
                message: 0,
                block: Some(0),
            },
            "the transparency probe dropped a part",
        ),
    ]
}

const RESPONSE_ID: &str = "probe-response-1";
const RESPONSE_MODEL: &str = "probe-served-model";
const REQUEST_ID: &str = "probe-request-1";
const TEXT: [&str; 2] = ["The data ", "path holds."];
const REFUSAL: &str = "The probe declines this part.";

/// The probe's response to a chat call.
pub(super) fn chat_response() -> ChatResponse {
    let mut response = ChatResponse::new(vec![
        ContentBlock::Text(allama::Text::new(TEXT.concat()).with_provider_data(provider_data())),
        ContentBlock::Refusal(allama::Refusal::new(REFUSAL)),
    ]);
    response.id = Some(RESPONSE_ID.into());
    response.model = Some(RESPONSE_MODEL.into());
    response.request_id = Some(REQUEST_ID.into());
    response.provider = Some(TransparencyProbe::PROVIDER_NAME.into());
    response.finish_reason = Some(FinishReason::Stop);
    response.finish_reason_raw = Some("probe_stop".into());
    response.usage = Some(usage());
    response.provider_data = provider_data();
    response.warnings = warnings();
    response
}

/// The probe's stream: every kind of event the data path carries.
fn chat_events() -> Vec<StreamEvent> {
    let [first, second, third] = <[Warning; 3]>::try_from(warnings()).expect("three warnings");
    vec![
        StreamEvent::response_start(Some(RESPONSE_ID.into()), Some(RESPONSE_MODEL.into())),
        StreamEvent::warning(first),
        StreamEvent::block_start(0, StreamBlockType::Text),
        StreamEvent::text_delta(0, TEXT[0]),
        StreamEvent::text_delta(0, TEXT[1]),
        StreamEvent::block_provider_data(0, provider_data()),
        StreamEvent::block_stop(0),
        StreamEvent::warning(second),
        StreamEvent::block_start(1, StreamBlockType::Refusal),
        StreamEvent::refusal_delta(1, REFUSAL),
        StreamEvent::block_stop(1),
        StreamEvent::response_provider_data(provider_data()),
        StreamEvent::warning(third),
        MetadataBuilder::new()
            .with_finish_reason(FinishReason::Stop)
            .with_finish_reason_raw("probe_stop")
            .with_usage(usage())
            .with_id(RESPONSE_ID)
            .with_model(RESPONSE_MODEL)
            .with_request_id(REQUEST_ID)
            .with_provider(TransparencyProbe::PROVIDER_NAME)
            .with_provider_data(provider_data())
            .build(),
        StreamEvent::ResponseStop,
    ]
}

/// How many of the canned events come before a mid-stream error: the start,
/// the first warning, and the first text, so output has started.
const EVENTS_BEFORE_ERROR: usize = 4;

/// The probe's stream for `scenario`, or the error `chat_stream` fails with.
pub(super) fn chat_stream(scenario: Scenario) -> allama::Result<ChatStream> {
    let items: Vec<allama::Result<StreamEvent>> = match scenario {
        Scenario::Ok => chat_events().into_iter().map(Ok).collect(),
        Scenario::Error(index) => return Err(error(index)),
        Scenario::StreamError(index) => chat_events()
            .into_iter()
            .take(EVENTS_BEFORE_ERROR)
            .map(Ok)
            .chain([Err(error(index))])
            .collect(),
    };
    Ok(Box::pin(futures_util::stream::iter(items)))
}

/// The probe's answer to a chat call for `scenario`.
pub(super) fn chat(scenario: Scenario) -> allama::Result<ChatResponse> {
    match scenario {
        Scenario::Ok => Ok(chat_response()),
        Scenario::Error(index) | Scenario::StreamError(index) => Err(error(index)),
    }
}

/// The probe's answer to an embedding call.
pub(super) fn embed(
    scenario: Scenario,
    request: &EmbeddingRequest,
) -> allama::Result<EmbeddingResponse> {
    match scenario {
        Scenario::Ok => Ok(embedding_response(request.inputs.len())),
        Scenario::Error(index) | Scenario::StreamError(index) => Err(error(index)),
    }
}

fn embedding_response(inputs: usize) -> EmbeddingResponse {
    let mut response = EmbeddingResponse::new(
        (0..inputs)
            .map(|i| vec![0.25, 0.5, 0.75 + i as f32])
            .collect(),
    )
    .model(RESPONSE_MODEL)
    .usage(usage())
    .provider(TransparencyProbe::PROVIDER_NAME)
    .provider_data(provider_data());
    response.warnings = warnings();
    response
}

/// The probe's answer to a decision call: a valid answer to every
/// question.
pub(super) fn decide(
    scenario: Scenario,
    request: &DecisionRequest,
) -> allama::Result<DecisionResponse> {
    match scenario {
        Scenario::Ok => Ok(decision_response(request)),
        Scenario::Error(index) | Scenario::StreamError(index) => Err(error(index)),
    }
}

fn decision_response(request: &DecisionRequest) -> DecisionResponse {
    let answers = request.questions.iter().map(|(name, question)| {
        let answer = match question {
            Question::Choice(choice) => {
                let share = 1.0 / choice.options.len().max(1) as f64;
                Answer::Choice(ChoiceAnswer::new(
                    choice
                        .options
                        .first()
                        .map(|option| option.key.clone())
                        .unwrap_or_default(),
                    choice
                        .options
                        .iter()
                        .map(|option| (option.key.clone(), share)),
                ))
            }
            Question::Score(score) => {
                let mut probabilities = vec![0.0; score.levels.len()];
                if let Some(first) = probabilities.first_mut() {
                    *first = 1.0;
                }
                Answer::Score(ScoreAnswer::new(0.0, probabilities))
            }
            _ => Answer::YesNo(YesNoAnswer::new(0.75)),
        };
        (name.clone(), answer)
    });
    let mut response = DecisionResponse::new(answers)
        .model(RESPONSE_MODEL)
        .id(RESPONSE_ID)
        .request_id(REQUEST_ID)
        .provider(TransparencyProbe::PROVIDER_NAME)
        .usage(usage())
        .provider_data(provider_data());
    response.warnings = warnings();
    response
}

/// The probe's answer to an extraction call: a valid answer to every
/// field.
pub(super) fn extract_values(
    scenario: Scenario,
    request: &ExtractionRequest,
) -> allama::Result<ExtractionResponse> {
    match scenario {
        Scenario::Ok => Ok(extraction_response(request)),
        Scenario::Error(index) | Scenario::StreamError(index) => Err(error(index)),
    }
}

fn extraction_response(request: &ExtractionRequest) -> ExtractionResponse {
    let answers = request.fields.iter().map(|(name, field)| {
        let answer: extraction::Answer = match field {
            Field::Text(_) => extraction::TextAnswer::new(Some("probe"))
                .span(extraction::TextSpan::new(0, 5))
                .null_probability(0.01)
                .into(),
            Field::Number(_) => extraction::NumberAnswer::new(Some(7.0))
                .null_probability(0.01)
                .into(),
            Field::Date(_) => extraction::DateAnswer::new(Some(
                "2026-01-02".parse::<extraction::Date>().expect("a date"),
            ))
            .null_probability(0.01)
            .into(),
            Field::Time(_) => extraction::TimeAnswer::new(Some(
                "13:34:00".parse::<extraction::Time>().expect("a time"),
            ))
            .into(),
            Field::DateTime(_) => extraction::DateTimeAnswer::new(Some(
                "2026-01-02T13:34:00"
                    .parse::<extraction::DateTime>()
                    .expect("a datetime"),
            ))
            .into(),
            Field::Duration(duration) => {
                extraction::DurationAnswer::new(Some(3.0), duration.unit).into()
            }
            Field::List(list) => extraction::ListAnswer::new([match list.items {
                extraction::ListItemKind::Number => extraction::ListItem::number(1.0),
                extraction::ListItemKind::Date => {
                    extraction::ListItem::date("2026-01-02".parse().expect("a date"))
                }
                _ => extraction::ListItem::text("probe").span(extraction::TextSpan::new(0, 5)),
            }])
            .into(),
            Field::Estimate(estimate) => {
                let mid = f64::midpoint(estimate.min, estimate.max);
                extraction::EstimateAnswer::new(mid)
                    .median(mid)
                    .interval(estimate.min, estimate.max, 0.8)
                    .into()
            }
            Field::Forecast(forecast) => {
                extraction::ForecastAnswer::new((0..forecast.horizon).map(|_| {
                    extraction::EstimateAnswer::new(f64::midpoint(forecast.min, forecast.max))
                }))
                .into()
            }
            _ => unreachable!("every field kind is answered"),
        };
        (name.clone(), answer)
    });
    let mut response = ExtractionResponse::new(answers)
        .model(RESPONSE_MODEL)
        .id(RESPONSE_ID)
        .request_id(REQUEST_ID)
        .provider(TransparencyProbe::PROVIDER_NAME)
        .usage(usage())
        .provider_data(provider_data());
    response.warnings = warnings();
    response
}

/// The errors the probe fails with: one of every kind, each with the
/// context a caller decides on (status, retry hint and directive, request
/// id, provider code, refused subject and stage).
fn errors() -> Vec<Error> {
    let hint = Some(Duration::from_millis(1));
    let http = |error: Error, status: u16, code: &str| {
        error
            .authored()
            .with_provider(TransparencyProbe::PROVIDER_NAME)
            .with_status(status)
            .with_request_id(format!("{REQUEST_ID}-{status}"))
            .with_provider_code(code.to_owned())
            .with_upstream_message(format!("upstream text for {code}"))
    };
    let before_sending =
        Degradations::new(TransparencyProbe::PROVIDER_NAME, UnsupportedPolicy::Warn).reject(
            Subject::Content {
                message: 0,
                block: Some(0),
            },
            "the transparency probe cannot send this part",
        );
    vec![
        http(Error::rate_limited("HTTP 429"), 429, "rate_limit")
            .with_retry_after_opt(hint)
            .with_server_retry_directive(true),
        http(Error::overloaded("HTTP 503"), 503, "overloaded").with_retry_after_opt(hint),
        http(Error::invalid_request("HTTP 400"), 400, "invalid")
            .with_param("messages.0")
            .with_server_retry_directive(false),
        http(Error::auth("HTTP 401"), 401, "bad_key"),
        http(Error::quota_exhausted("HTTP 429"), 429, "no_credit"),
        http(
            Error::context_length_exceeded("HTTP 400", Some(8192)),
            400,
            "too_long",
        ),
        http(
            Error::unsupported_by_provider(
                Some(Subject::Field {
                    field: RequestField::Temperature,
                }),
                "HTTP 400",
            ),
            400,
            "unsupported_value",
        ),
        before_sending,
        http(Error::model_not_found("HTTP 404"), 404, "no_model"),
        http(Error::timeout("HTTP 504"), 504, "gateway_timeout"),
        http(Error::provider("HTTP 500"), 500, "internal"),
        http(Error::unexpected_response("HTTP 200"), 200, "malformed"),
        http(Error::content_filtered("HTTP 400"), 400, "filtered"),
        Error::stream(StreamErrorKind::Incomplete, "the stream ended early")
            .with_provider(TransparencyProbe::PROVIDER_NAME),
        Error::configuration("no key for this scope")
            .with_provider(TransparencyProbe::PROVIDER_NAME),
    ]
}

/// The error at `index` of [`errors`].
fn error(index: usize) -> Error {
    errors().swap_remove(index.min(errors().len() - 1))
}

/// What a caller decides on: everything but the message, which a wrapper
/// may prefix ([`Error::with_message_prefix`]).
#[derive(Debug, PartialEq)]
struct ErrorView {
    kind: &'static str,
    stream_kind: Option<StreamErrorKind>,
    status: Option<u16>,
    request_id: Option<String>,
    retry_after: Option<Duration>,
    server_retry_directive: Option<bool>,
    upstream_message: Option<String>,
    provider_code: Option<String>,
    param: Option<String>,
    subject: Option<Subject>,
    stage: Option<UnsupportedStage>,
    origin: MessageOrigin,
    provider: Option<String>,
}

impl ErrorView {
    fn of(error: &Error) -> Self {
        Self {
            kind: error.kind_name(),
            stream_kind: match error {
                Error::Stream { kind, .. } => Some(*kind),
                _ => None,
            },
            status: error.status(),
            request_id: error.request_id().map(str::to_owned),
            retry_after: error.retry_after(),
            server_retry_directive: error.server_retry_directive(),
            upstream_message: error.upstream_message().map(str::to_owned),
            provider_code: error.provider_code().map(str::to_owned),
            param: error.param().map(str::to_owned),
            subject: error.unsupported_subject().cloned(),
            stage: error.unsupported_stage(),
            origin: error.message_origin(),
            provider: error.context().and_then(|context| context.provider.clone()),
        }
    }
}

/// What is wrong with `actual`, which a wrapper returned where the probe
/// failed with `expected`. A wrapper may fail over to a provider that fails
/// the same way, and then returns that error with the earlier ones
/// [behind it](Error::fallback_from), which are not compared; its message
/// may carry a prefix.
fn error_problem(actual: &Error, expected: &Error) -> Option<String> {
    if ErrorView::of(actual) != ErrorView::of(expected) {
        return Some(format!(
            "the {} error changed: expected {:?}, got {:?}",
            expected.kind_name(),
            ErrorView::of(expected),
            ErrorView::of(actual)
        ));
    }
    (!actual.message().ends_with(&expected.message())).then(|| {
        format!(
            "the {} error's message changed: expected it to end with {:?}, got {:?} (add \
             context with Error::with_message_prefix)",
            expected.kind_name(),
            expected.message(),
            actual.message()
        )
    })
}

/// What is wrong with `actual`, which a wrapper returned for the probe's
/// `expected` warnings: they must all be there, in order. The wrapper may
/// add its own.
fn warnings_problem(actual: &[Warning], expected: &[Warning]) -> Option<String> {
    let mut rest = expected.iter().peekable();
    for warning in actual {
        if rest.peek() == Some(&warning) {
            rest.next();
        }
    }
    let missing: Vec<&Warning> = rest.collect();
    (!missing.is_empty()).then(|| {
        format!("the probe's warnings {missing:?} are missing or out of order in {actual:?}")
    })
}

/// What is wrong with `actual`, a response a wrapper returned for the
/// probe's `expected`: the same, apart from warnings the wrapper added.
fn response_problem<T: serde::Serialize>(
    actual: &T,
    expected: &T,
    warnings: impl Fn(&T) -> &[Warning],
) -> Option<String> {
    let without_warnings = |response: &T| {
        let mut value = serde_json::to_value(response).unwrap_or_default();
        if let Some(object) = value.as_object_mut() {
            object.remove("warnings");
        }
        value
    };
    let (got, want): (Value, Value) = (without_warnings(actual), without_warnings(expected));
    if got != want {
        return Some(format!("the response changed: expected {want}, got {got}"));
    }
    warnings_problem(warnings(actual), warnings(expected))
}

/// What is wrong with the items a wrapper's stream yielded for the probe's
/// `events`, then `error`: the probe's events, in order, with only warnings
/// added, then the error.
fn stream_problem(
    actual: &[allama::Result<StreamEvent>],
    events: &[StreamEvent],
    error: Option<&Error>,
) -> Option<String> {
    let mut expected = events.iter().peekable();
    let mut failed = None;
    for item in actual {
        match item {
            Ok(event) if expected.peek() == Some(&event) => {
                expected.next();
            }
            // A wrapper may add warnings.
            Ok(StreamEvent::Warning { .. }) => {}
            Ok(event) => {
                return Some(match expected.peek() {
                    Some(next) => format!(
                        "the probe's event {} is missing or out of order: the stream yielded {} \
                         in its place",
                        show(next),
                        show(event)
                    ),
                    None => format!(
                        "the stream yielded {} after the probe's last event",
                        show(event)
                    ),
                });
            }
            Err(err) => {
                failed = Some(err);
                break;
            }
        }
    }
    let missing: Vec<String> = expected.map(show).collect();
    if !missing.is_empty() {
        return Some(format!(
            "the probe's events {} are missing",
            missing.join(", ")
        ));
    }
    match (failed, error) {
        (None, None) => None,
        (Some(actual), Some(expected)) => error_problem(actual, expected),
        (Some(actual), None) => Some(format!("the stream failed: {actual}")),
        (None, Some(expected)) => Some(format!(
            "the stream ended without the probe's {} error",
            expected.kind_name()
        )),
    }
}

/// An event, compactly, for problem reports: media payloads elided.
fn show(event: &StreamEvent) -> String {
    let shown = serde_json::to_string(event).unwrap_or_else(|_| format!("{event:?}"));
    elide_payloads(&shown)
}

/// Every item of the stream `chat_stream` gave, up to and including the
/// first error; the error alone when there is no stream.
async fn collect<S>(stream: allama::Result<S>) -> Vec<allama::Result<StreamEvent>>
where
    S: futures_core::Stream<Item = allama::Result<StreamEvent>>,
{
    let stream = match stream {
        Ok(stream) => stream,
        Err(err) => return vec![Err(err)],
    };
    let mut stream = std::pin::pin!(stream);
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        let failed = item.is_err();
        items.push(item);
        if failed {
            break;
        }
    }
    items
}

fn chat_request(scenario: Scenario) -> ChatRequest {
    ChatRequest::new(TransparencyProbe::MODELS[0]).user(scenario.text())
}

/// The data path checks of a chat wrapper, as problems.
pub(super) async fn chat_problems<W: ChatProvider + ?Sized>(wrapper: &W) -> Vec<String> {
    let mut problems = Vec::new();
    let mut check = |path: &str, scenario: Scenario, problem: Option<String>| {
        if let Some(problem) = problem {
            problems.push(format!("{path} ({scenario:?}): {problem}"));
        }
    };

    let expected = chat_response();
    let problem = match wrapper.chat(&chat_request(Scenario::Ok)).await {
        Ok(response) => response_problem(&response, &expected, |r| &r.warnings),
        Err(err) => Some(format!("the probe's response became an error: {err}")),
    };
    check("chat", Scenario::Ok, problem);
    let items = collect(wrapper.chat_stream(&chat_request(Scenario::Ok)).await).await;
    check(
        "chat_stream",
        Scenario::Ok,
        stream_problem(&items, &chat_events(), None),
    );

    for (index, expected) in errors().iter().enumerate() {
        let scenario = Scenario::Error(index);
        let problem = match wrapper.chat(&chat_request(scenario)).await {
            Ok(_) => Some(format!(
                "the probe's {} error became a response",
                expected.kind_name()
            )),
            Err(err) => error_problem(&err, expected),
        };
        check("chat", scenario, problem);
        let items = collect(wrapper.chat_stream(&chat_request(scenario)).await).await;
        check(
            "chat_stream",
            scenario,
            stream_problem(&items, &[], Some(expected)),
        );
        let scenario = Scenario::StreamError(index);
        let items = collect(wrapper.chat_stream(&chat_request(scenario)).await).await;
        check(
            "chat_stream",
            scenario,
            stream_problem(
                &items,
                &chat_events()[..EVENTS_BEFORE_ERROR],
                Some(expected),
            ),
        );
    }
    problems
}

fn embedding_request(scenario: Scenario) -> EmbeddingRequest {
    EmbeddingRequest::new(TransparencyProbe::MODELS[0]).input(scenario.text())
}

/// The data path checks of an embedding wrapper, as problems.
pub(super) async fn embedding_problems<W: EmbeddingProvider + ?Sized>(wrapper: &W) -> Vec<String> {
    let mut problems = Vec::new();
    let expected = embedding_response(1);
    let problem = match wrapper.embed(&embedding_request(Scenario::Ok)).await {
        Ok(response) => response_problem(&response, &expected, |r| &r.warnings),
        Err(err) => Some(format!("the probe's response became an error: {err}")),
    };
    problems.extend(problem.map(|problem| format!("embed (Ok): {problem}")));
    for (index, expected) in errors().iter().enumerate() {
        let scenario = Scenario::Error(index);
        let problem = match wrapper.embed(&embedding_request(scenario)).await {
            Ok(_) => Some(format!(
                "the probe's {} error became a response",
                expected.kind_name()
            )),
            Err(err) => error_problem(&err, expected),
        };
        problems.extend(problem.map(|problem| format!("embed ({scenario:?}): {problem}")));
    }
    problems
}

/// A question every probe model is asked about: the probe answers score
/// questions `Supported` for [`TransparencyProbe::MODELS`]`[0]`.
fn decision_request(scenario: Scenario) -> DecisionRequest {
    DecisionRequest::new(TransparencyProbe::MODELS[0], "probe").question(
        scenario.text(),
        Question::score("How well does the data path hold?").levels(["badly", "well"]),
    )
}

/// The data path checks of a decision wrapper, as problems.
pub(super) async fn decision_problems<W: DecisionProvider + ?Sized>(wrapper: &W) -> Vec<String> {
    let mut problems = Vec::new();
    let request = decision_request(Scenario::Ok);
    let expected = decision_response(&request);
    let problem = match wrapper.decide(&request).await {
        Ok(response) => response_problem(&response, &expected, |r| &r.warnings),
        Err(err) => Some(format!("the probe's response became an error: {err}")),
    };
    problems.extend(problem.map(|problem| format!("decide (Ok): {problem}")));
    for (index, expected) in errors().iter().enumerate() {
        let scenario = Scenario::Error(index);
        let problem = match wrapper.decide(&decision_request(scenario)).await {
            Ok(_) => Some(format!(
                "the probe's {} error became a response",
                expected.kind_name()
            )),
            Err(err) => error_problem(&err, expected),
        };
        problems.extend(problem.map(|problem| format!("decide ({scenario:?}): {problem}")));
    }
    problems
}

/// A field every probe model is asked about: the probe answers date
/// fields `Supported` for [`TransparencyProbe::MODELS`]`[0]`.
fn extraction_request(scenario: Scenario) -> ExtractionRequest {
    ExtractionRequest::new(TransparencyProbe::MODELS[0], "probe").field(
        scenario.text(),
        Field::date().instructions("When was the data path checked?"),
    )
}

/// The data path checks of an extraction wrapper, as problems.
pub(super) async fn extraction_problems<W: ExtractionProvider + ?Sized>(
    wrapper: &W,
) -> Vec<String> {
    let mut problems = Vec::new();
    let request = extraction_request(Scenario::Ok);
    let expected = extraction_response(&request);
    let problem = match wrapper.extract_values(&request).await {
        Ok(response) => response_problem(&response, &expected, |r| &r.warnings),
        Err(err) => Some(format!("the probe's response became an error: {err}")),
    };
    problems.extend(problem.map(|problem| format!("extract_values (Ok): {problem}")));
    for (index, expected) in errors().iter().enumerate() {
        let scenario = Scenario::Error(index);
        let problem = match wrapper.extract_values(&extraction_request(scenario)).await {
            Ok(_) => Some(format!(
                "the probe's {} error became a response",
                expected.kind_name()
            )),
            Err(err) => error_problem(&err, expected),
        };
        problems.extend(problem.map(|problem| format!("extract_values ({scenario:?}): {problem}")));
    }
    problems
}

/// Panic with `problems` of the wrapper `wrapper_name`, if there are any.
pub(super) fn report(wrapper_name: &str, problems: &[String]) {
    assert!(
        problems.is_empty(),
        "`{wrapper_name}` does not pass the provider's data through unchanged:\n  {}",
        problems.join("\n  ")
    );
}

#[cfg(test)]
mod tests {
    use allama::{CapabilitySupport, DecisionCapability};

    use super::*;

    #[test]
    fn scenarios_round_trip() {
        for scenario in [Scenario::Ok, Scenario::Error(3), Scenario::StreamError(11)] {
            assert_eq!(Scenario::parse(Some(&scenario.text())), scenario);
        }
        assert_eq!(Scenario::parse(Some("probe")), Scenario::Ok);
        assert_eq!(Scenario::parse(None), Scenario::Ok);
    }

    #[test]
    fn the_decision_question_is_supported() {
        assert_eq!(
            TransparencyProbe
                .decision_capability(TransparencyProbe::MODELS[0], DecisionCapability::Score),
            CapabilitySupport::Supported
        );
        let request = decision_request(Scenario::Ok);
        decision_response(&request).validate(&request).unwrap();
    }

    #[test]
    fn the_extraction_field_is_supported_and_every_kind_validates() {
        use allama::ExtractionCapability;

        assert_eq!(
            TransparencyProbe
                .extraction_capability(TransparencyProbe::MODELS[0], ExtractionCapability::Date),
            CapabilitySupport::Supported
        );
        let request = extraction_request(Scenario::Ok);
        extraction_response(&request).validate(&request).unwrap();
        // The probe answers every kind with a valid answer.
        let all = ExtractionRequest::new("m", "probe")
            .field("t", Field::text())
            .field("n", Field::number())
            .field("d", Field::date())
            .field("ti", Field::time())
            .field("dt", Field::datetime())
            .field("du", Field::duration(extraction::DurationUnit::Hours))
            .field("l", Field::list(extraction::ListItemKind::Date))
            .field("e", Field::estimate(1.0, 3.0))
            .field("f", Field::forecast(0.0, 10.0).horizon(4));
        extraction_response(&all).validate(&all).unwrap();
    }

    #[test]
    fn the_stream_collects_to_the_response() {
        let mut collector = allama::chat::stream::StreamCollector::new();
        for event in chat_events() {
            collector.push(event).unwrap();
        }
        let collected = collector.finish().unwrap();
        assert_eq!(
            serde_json::to_value(&collected).unwrap(),
            serde_json::to_value(chat_response()).unwrap()
        );
    }

    #[test]
    fn shown_events_elide_media_payloads() {
        let image = allama::Image::base64("image/png", crate::e2e::RED_PNG_BASE64);
        let shown = show(&StreamEvent::image_start(0, image));
        assert!(!shown.contains(crate::e2e::RED_PNG_BASE64), "{shown}");
        assert!(shown.contains("<base64: "), "{shown}");
    }

    #[test]
    fn a_prefixed_error_passes_and_a_rewrapped_one_fails() {
        for (index, expected) in errors().iter().enumerate() {
            let prefixed = error(index).with_message_prefix("wrapper");
            assert_eq!(error_problem(&prefixed, expected), None, "{expected:?}");
            let fallback = error(index).with_fallback_from(error(0));
            assert_eq!(error_problem(&fallback, expected), None, "{expected:?}");
        }
        let expected = error(0);
        let rewrapped = Error::provider(format!("wrapper: {expected}"));
        assert!(error_problem(&rewrapped, &expected).is_some());
    }
}
