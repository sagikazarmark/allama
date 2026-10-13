//! The strict-not-honoured contract tells an honest provider from one that
//! drops strict mode silently on either path.

use allama::adapter::Degradations;
use allama::{
    ChatRequest, ChatResponse, ContentBlock, Error, RequestField, ResponseFormat,
    UnsupportedPolicy, Warning,
};
use allama_conformance::probe::{ProbeOutcome, assert_strict_not_honoured_warns};
use serde_json::{Value, json};

/// A refusal of `field` before sending, as `Degradations` builds it.
fn refusal(field: RequestField, message: &str) -> Error {
    Degradations::new("fake", UnsupportedPolicy::Error).unsupported(field, message)
}

/// The field a probe request asks to be strict.
fn strict_field(request: &ChatRequest) -> RequestField {
    match &request.response_format {
        Some(ResponseFormat::JsonSchema {
            strict: Some(true), ..
        }) => RequestField::ResponseFormat,
        _ => RequestField::Tools,
    }
}

/// An endpoint without strict mode: it degrades with a warning, or fails
/// before sending under `UnsupportedPolicy::Error`. `warn: false` drops the
/// warning and `refuse: false` the refusal.
fn endpoint(request: &ChatRequest, warn: bool, refuse: bool) -> ProbeOutcome<ChatResponse> {
    if refuse && request.unsupported_policy == Some(UnsupportedPolicy::Error) {
        return ProbeOutcome::new(
            Err(refusal(strict_field(request), "strict is not supported")),
            Vec::new(),
        );
    }
    let mut response = ChatResponse::new(vec![ContentBlock::text("ok")]);
    if warn {
        response.warnings.push(Warning::unsupported(
            strict_field(request),
            "strict is not supported; sent without strict",
        ));
    }
    ProbeOutcome::new(Ok(response), vec![json!({"sent": true})])
}

async fn honest(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
    endpoint(&request, true, true)
}

async fn silent(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
    endpoint(&request, false, true)
}

async fn lenient(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
    endpoint(&request, true, false)
}

/// An endpoint that serves neither tools nor JSON Schema output.
async fn refuses_everything(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
    ProbeOutcome::new(
        Err(refusal(strict_field(&request), "not supported")),
        Vec::<Value>::new(),
    )
}

/// An endpoint that refuses with `Error::unsupported`, which says nothing
/// about whether the request was sent.
async fn refuses_at_an_unspecified_stage(_: ChatRequest) -> ProbeOutcome<ChatResponse> {
    ProbeOutcome::new(
        Err(Error::unsupported("tools are not supported")),
        Vec::<Value>::new(),
    )
}

#[tokio::test]
async fn an_honest_provider_passes() {
    assert_strict_not_honoured_warns("m", honest, honest).await;
    assert_strict_not_honoured_warns("m", refuses_everything, refuses_everything).await;
}

#[tokio::test]
#[should_panic(expected = "tools (chat): refused with stage unspecified, not before_sending")]
async fn a_refusal_at_an_unspecified_stage_fails() {
    assert_strict_not_honoured_warns(
        "m",
        refuses_at_an_unspecified_stage,
        refuses_at_an_unspecified_stage,
    )
    .await;
}

#[tokio::test]
#[should_panic(expected = "tools (chat_stream): strict mode was not honoured")]
async fn dropping_the_warning_on_the_stream_path_fails() {
    assert_strict_not_honoured_warns("m", honest, silent).await;
}

#[tokio::test]
#[should_panic(expected = "response_format (chat): strict mode was not honoured")]
async fn dropping_the_warning_on_the_chat_path_fails() {
    assert_strict_not_honoured_warns("m", silent, honest).await;
}

#[tokio::test]
#[should_panic(expected = "UnsupportedPolicy::Error did not fail")]
async fn ignoring_the_error_policy_fails() {
    assert_strict_not_honoured_warns("m", lenient, honest).await;
}

/// [`endpoint`] with a line-oriented text body, as an adapter with a
/// non-JSON wire format reports it. `late` sends the request before
/// refusing it under `UnsupportedPolicy::Error`.
fn raw_endpoint(request: &ChatRequest, late: bool) -> ProbeOutcome<ChatResponse> {
    let body = format!("model {}\nstrict true\n", request.model);
    if request.unsupported_policy == Some(UnsupportedPolicy::Error) {
        let sent = if late { vec![body] } else { Vec::new() };
        let refused = refusal(strict_field(request), "strict is not supported");
        return ProbeOutcome::raw(Err(refused), sent);
    }
    let mut response = ChatResponse::new(vec![ContentBlock::text("ok")]);
    response.warnings.push(Warning::unsupported(
        strict_field(request),
        "strict is not supported; sent without strict",
    ));
    ProbeOutcome::raw(Ok(response), [body])
}

async fn raw_honest(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
    raw_endpoint(&request, false)
}

async fn raw_refuses_late(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
    raw_endpoint(&request, true)
}

#[tokio::test]
async fn an_honest_raw_body_provider_passes() {
    assert_strict_not_honoured_warns("m", raw_honest, raw_honest).await;
}

#[tokio::test]
#[should_panic(
    expected = "tools (chat): rejected under UnsupportedPolicy::Error only after sending"
)]
async fn a_raw_body_provider_refusing_after_sending_fails() {
    // Read as JSON only, the raw body would look unsent and pass.
    assert_strict_not_honoured_warns("m", raw_refuses_late, raw_honest).await;
}
