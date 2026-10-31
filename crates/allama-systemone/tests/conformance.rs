//! Every decision contract, through the aggregate (`assert_decision_adapter`),
//! for the TypeSafe and Cloudflare presets.

use allama::decision::Question;
use allama::{
    DecisionProvider, DecisionRequest, DecisionResponse, RequestOptions, ValidationCoverage,
};
use allama_conformance::capabilities::UNKNOWN_MODEL;
use allama_conformance::probe::ProbeOutcome;
use allama_conformance::stream_errors::ErrorKindProbe;
use allama_conformance::{DecisionAdapterUnderTest, TransportOutcome, assert_decision_adapter};
use allama_http::http::StatusCode;
use allama_http::{CredentialPlacement, CredentialSlot, ReplayResponse, ReplayTransport};
use allama_systemone::providers::{Cloudflare, TypeSafe};
use allama_systemone::{Provider, map_http_error};
use serde_json::{Map, Value, json};

fn typesafe(transport: &ReplayTransport) -> Provider {
    TypeSafe::builder("ts-key")
        .unwrap()
        .http_transport(transport.clone())
        .build()
        .unwrap()
}

fn cloudflare(transport: &ReplayTransport) -> Provider {
    Cloudflare::builder("0123456789abcdef0123456789abcdef", "cf-token")
        .unwrap()
        .http_transport(transport.clone())
        .build()
        .unwrap()
}

fn bodies(transport: &ReplayTransport) -> Vec<Value> {
    transport
        .requests()
        .iter()
        .filter_map(|request| request.body_text())
        .map(|body| serde_json::from_str(body).expect("JSON request body"))
        .collect()
}

/// A System One answer to every question of `request`, as the wire
/// format writes it.
fn answers(request: &DecisionRequest) -> Value {
    let answers: Map<String, Value> = request
        .questions
        .iter()
        .map(|(name, question)| {
            let answer = match question {
                Question::Choice(choice) => {
                    let share = 1.0 / choice.options.len().max(1) as f64;
                    let probabilities: Map<String, Value> = choice
                        .options
                        .iter()
                        .map(|option| (option.key.clone(), json!(share)))
                        .collect();
                    json!({
                        "type": "choice",
                        "choice": choice.options.first().map(|option| option.key.clone()),
                        "probabilities": probabilities
                    })
                }
                Question::Score(score) => {
                    let probabilities: Map<String, Value> = (0..score.levels.len())
                        .map(|level| (level.to_string(), json!(if level == 0 { 1.0 } else { 0.0 })))
                        .collect();
                    json!({"type": "score", "score": 0.0, "probabilities": probabilities})
                }
                _ => json!({"type": "noul", "noul": 0.75}),
            };
            (name.clone(), answer)
        })
        .collect();
    json!({"model": request.model, "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 1}})
}

/// `decide` over a transport that answers the request's questions.
async fn probe(request: DecisionRequest) -> ProbeOutcome<DecisionResponse> {
    let transport = ReplayTransport::new([ReplayResponse::json(
        StatusCode::OK,
        answers(&request).to_string(),
    )]);
    let result = typesafe(&transport).decide(&request).await;
    ProbeOutcome::new(result, bodies(&transport))
}

/// The FastAPI `detail` envelope for each kind TypeSafe reports, with the
/// upstream text after its message. An invalid request is FastAPI's
/// validation envelope, a list of errors whose `msg` is upstream text too.
fn envelope(kind: ErrorKindProbe, upstream: &str) -> Option<String> {
    let (error_type, message) = match kind {
        ErrorKindProbe::RateLimited => ("rate_limit_error", "Too many requests"),
        ErrorKindProbe::Overloaded => ("overloaded_error", "Overloaded"),
        ErrorKindProbe::InvalidRequest => ("invalid_request_error", "questions: field required"),
        ErrorKindProbe::Auth => (
            "authentication_error",
            "Must supply an API key! Check your request and try again.",
        ),
        ErrorKindProbe::ContextLength => (
            "invalid_request_error",
            "The input exceeds the model's context window",
        ),
        ErrorKindProbe::ModelNotFound => ("not_found_error", "model 'jev-nope' not found"),
        ErrorKindProbe::Timeout => ("timeout_error", "Request timed out"),
        ErrorKindProbe::Unsupported => (
            "invalid_request_error",
            "model jev-tiny does not support decision questions of type score",
        ),
        _ => return None,
    };
    let message = format!("{message} {upstream}");
    let message = message.trim_end();
    if kind == ErrorKindProbe::InvalidRequest {
        return Some(
            json!({"detail": [{"loc": ["body", "questions"], "msg": message, "type": "missing"}]})
                .to_string(),
        );
    }
    Some(json!({"detail": {"error_type": error_type, "message": message}}).to_string())
}

#[tokio::test]
async fn typesafe_passes_every_contract() {
    let provider = typesafe(&ReplayTransport::default());
    for model in ["jev-latest", UNKNOWN_MODEL] {
        let request = DecisionRequest::new(model, "x").question("q", Question::yes_no("?"));
        let validation = provider.validate_decision_request(&request).unwrap();
        assert_eq!(validation.coverage(), ValidationCoverage::Encoding);
    }
    let slot = CredentialSlot::new(
        TypeSafe::DEFAULT_CREDENTIAL_SCOPE,
        CredentialPlacement::Bearer,
    );
    let under_test = DecisionAdapterUnderTest::new(["jev-latest"], probe)
        .error_mapping(allama_conformance::error_kinds::http_error_mapping(
            map_http_error,
        ))
        .error_envelope(envelope)
        .transport_options(slot, |options: RequestOptions| async move {
            let transport = ReplayTransport::default();
            let mut request = DecisionRequest::new("nimble", "Is this a refund request?")
                .question("refund", Question::yes_no("Is it?"));
            request.options = options;
            let result = typesafe(&transport).decide(&request).await;
            TransportOutcome::new(result, transport.requests())
        });
    assert_decision_adapter(&provider, &under_test).await;
}

/// `decide` over a transport that answers the request's questions inside
/// Cloudflare's `{"result": ..}` envelope.
async fn cloudflare_probe(request: DecisionRequest) -> ProbeOutcome<DecisionResponse> {
    let body = json!({"result": answers(&request), "success": true, "errors": [], "messages": []});
    let transport = ReplayTransport::new([ReplayResponse::json(StatusCode::OK, body.to_string())]);
    let result = cloudflare(&transport).decide(&request).await;
    ProbeOutcome::new(result, bodies(&transport))
}

/// Cloudflare's `errors` envelope for each kind Workers AI documents a code
/// for (<https://developers.cloudflare.com/workers-ai/platform/errors/>),
/// with the upstream text after its message. It documents no wording for
/// an unsupported feature or a context overflow.
fn cloudflare_envelope(kind: ErrorKindProbe, upstream: &str) -> Option<String> {
    let (code, message) = match kind {
        ErrorKindProbe::RateLimited => (3001, "Rate limited"),
        ErrorKindProbe::QuotaExhausted => (
            3036,
            "You have used up your daily free allocation of 10,000 neurons",
        ),
        ErrorKindProbe::Overloaded => (3040, "Capacity temporarily exceeded"),
        ErrorKindProbe::InvalidRequest => (5006, "Bad input"),
        ErrorKindProbe::Auth => (10000, "Authentication error"),
        ErrorKindProbe::ModelNotFound => (7000, "No route for that URI"),
        ErrorKindProbe::Timeout => (3007, "Request timeout"),
        _ => return None,
    };
    let message = format!("{message} {upstream}");
    Some(
        json!({
            "result": null,
            "success": false,
            "errors": [{"code": code, "message": message.trim_end()}],
            "messages": []
        })
        .to_string(),
    )
}

#[tokio::test]
async fn cloudflare_passes_every_contract() {
    let provider = cloudflare(&ReplayTransport::default());
    let slot = CredentialSlot::new(
        Cloudflare::DEFAULT_CREDENTIAL_SCOPE,
        CredentialPlacement::Bearer,
    );
    let under_test = DecisionAdapterUnderTest::new(
        ["@cf/cloudflare/clef", "@cf/cloudflare/clef-flash"],
        cloudflare_probe,
    )
    .error_mapping(allama_conformance::error_kinds::http_error_mapping(
        map_http_error,
    ))
    .error_envelope(cloudflare_envelope)
    .transport_options(slot, |options: RequestOptions| async move {
        let transport = ReplayTransport::default();
        let mut request = DecisionRequest::new("@cf/cloudflare/clef", "Is this a refund request?")
            .question("refund", Question::yes_no("Is it?"));
        request.options = options;
        let result = cloudflare(&transport).decide(&request).await;
        TransportOutcome::new(result, transport.requests())
    });
    assert_decision_adapter(&provider, &under_test).await;
}
