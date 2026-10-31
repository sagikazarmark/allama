//! Every extraction contract, through the aggregate
//! (`assert_extraction_adapter`), for the Workers AI provider.

use allama::extraction::{Field, ListItemKind};
use allama::{
    ExtractionProvider, ExtractionRequest, ExtractionResponse, RequestOptions, ValidationCoverage,
};
use allama_cloudflare::{DEFAULT_CREDENTIAL_SCOPE, Provider, map_http_error};
use allama_conformance::capabilities::UNKNOWN_MODEL;
use allama_conformance::probe::ProbeOutcome;
use allama_conformance::stream_errors::ErrorKindProbe;
use allama_conformance::{ExtractionAdapterUnderTest, TransportOutcome, assert_extraction_adapter};
use allama_http::http::StatusCode;
use allama_http::{CredentialPlacement, CredentialSlot, ReplayResponse, ReplayTransport};
use serde_json::{Map, Value, json};

fn provider(transport: &ReplayTransport) -> Provider {
    Provider::builder()
        .account_id("0123456789abcdef0123456789abcdef")
        .api_token("cf-token")
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

/// A Clairic answer to every field of `request`, as the wire format
/// writes it, inside Cloudflare's `{"result": ..}` envelope. Text answers
/// are the input's first word, with its span, so every span is placed.
fn answers(request: &ExtractionRequest) -> Value {
    let word = request.input.split_whitespace().next();
    let text = |value: Option<&str>| match value {
        Some(word) => {
            let start = request.input.find(word).unwrap_or(0);
            json!({"type": "text", "value": word, "span": [start, start + word.len()], "p_null": 0.01, "confidence": 0.9, "guess": word})
        }
        None => {
            json!({"type": "text", "value": null, "p_null": 1.0, "confidence": 0.5, "guess": "x"})
        }
    };
    let answers: Map<String, Value> = request
        .fields
        .iter()
        .map(|(name, field)| {
            let answer = match field {
                Field::Text(_) => text(word),
                Field::Number(_) => json!({"type": "number", "value": 7, "p_null": 0.01, "confidence": 0.9, "guess": 7}),
                Field::Date(_) => json!({"type": "date", "value": "2026-09-28", "p_null": 0.01, "confidence": 0.9}),
                Field::Time(_) => json!({"type": "time", "value": "13:34:00", "p_null": 0.01, "confidence": 0.9}),
                Field::DateTime(_) => json!({"type": "datetime", "value": "2026-09-28T13:34:00", "p_null": 0.01, "confidence": 0.9}),
                Field::Duration(duration) => json!({"type": "duration", "value": 3, "unit": duration.unit.as_str(), "p_null": 0.01, "confidence": 0.9}),
                Field::List(list) => match list.items {
                    ListItemKind::Number => json!({"type": "list", "value": [1, 2]}),
                    ListItemKind::Date => json!({"type": "list", "value": ["2026-09-28"]}),
                    _ => match word {
                        Some(word) => {
                            let start = request.input.find(word).unwrap_or(0);
                            json!({"type": "list", "value": [word], "spans": [[start, start + word.len()]]})
                        }
                        None => json!({"type": "list", "value": []}),
                    },
                },
                Field::Estimate(estimate) => {
                    let mid = f64::midpoint(estimate.min, estimate.max);
                    json!({"type": "estimate", "value": mid, "mean": mid, "median": mid, "p10": estimate.min, "p90": estimate.max, "p_below": 0.0, "p_above": 0.0})
                }
                Field::Forecast(forecast) => {
                    let mid = f64::midpoint(forecast.min, forecast.max);
                    let series: Vec<f64> = (0..forecast.horizon).map(|_| mid).collect();
                    json!({"type": "forecast", "value": series, "median": series, "p10": series, "p90": series})
                }
                _ => unreachable!("every field kind is answered"),
            };
            (name.clone(), answer)
        })
        .collect();
    json!({
        "result": {
            "model": "clairic",
            "answers": answers,
            "truncated": false,
            "usage": {"state_tokens": 20, "image_tokens": 0, "schema_tokens": 30, "input_tokens": 50, "output_tokens": 0, "passes": 1}
        },
        "success": true,
        "errors": [],
        "messages": []
    })
}

/// `extract_values` over a transport that answers the request's fields.
async fn probe(request: ExtractionRequest) -> ProbeOutcome<ExtractionResponse> {
    let transport = ReplayTransport::new([ReplayResponse::json(
        StatusCode::OK,
        answers(&request).to_string(),
    )]);
    let result = provider(&transport).extract_values(&request).await;
    ProbeOutcome::new(result, bodies(&transport))
}

/// Cloudflare's `errors` envelope for each kind Workers AI documents a code
/// for (<https://developers.cloudflare.com/workers-ai/platform/errors/>),
/// with the upstream text after its message.
fn envelope(kind: ErrorKindProbe, upstream: &str) -> Option<String> {
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
async fn passes_every_contract() {
    let under_test_provider = provider(&ReplayTransport::default());
    for model in ["@cf/cloudflare/clairic", UNKNOWN_MODEL] {
        let request = ExtractionRequest::new(model, "x").field("f", Field::text());
        let validation = under_test_provider
            .validate_extraction_request(&request)
            .unwrap();
        assert_eq!(validation.coverage(), ValidationCoverage::Encoding);
    }
    let slot = CredentialSlot::new(DEFAULT_CREDENTIAL_SCOPE, CredentialPlacement::Bearer);
    let under_test = ExtractionAdapterUnderTest::new(["@cf/cloudflare/clairic"], probe)
        .error_mapping(allama_conformance::error_kinds::http_error_mapping(
            map_http_error,
        ))
        .error_envelope(envelope)
        .transport_options(slot, |options: RequestOptions| async move {
            let transport = ReplayTransport::default();
            let mut request = ExtractionRequest::new("@cf/cloudflare/clairic", "Total: $1")
                .field("total", Field::number());
            request.options = options;
            let result = provider(&transport).extract_values(&request).await;
            TransportOutcome::new(result, transport.requests())
        });
    assert_extraction_adapter(&under_test_provider, &under_test).await;
}
