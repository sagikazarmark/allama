//! The whole adapter over wire examples, through a replay transport: request
//! encoding, headers, decoding, validation, and errors. See
//! `fixtures/README.md` for where each fixture comes from.

use std::time::Duration;

use allama::decision::{
    Choice, ChoiceSet, Decision, DecisionInput, Question, Scale, Score, YesNoAnswer,
};
use allama::{DecisionProvider, DecisionProviderExt, DecisionRequest, Error, Image};
use allama_http::http::{HeaderValue, StatusCode};
use allama_http::{ReplayResponse, ReplayTransport};
use allama_systemone::providers::{Cloudflare, LiteLlm, Ollama, OpenRouter, TypeSafe};
use allama_systemone::{Confidence, DecisionRequestOptions, Gateway, Provider, RawResponse};
use serde_json::{Value, json};

fn fixture(path: &str) -> String {
    std::fs::read_to_string(format!("{}/fixtures/{path}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

fn ollama(transport: &ReplayTransport) -> Provider {
    Ollama::builder("http://localhost:11434")
        .unwrap()
        .http_transport(transport.clone())
        .build()
        .unwrap()
}

fn typesafe(transport: &ReplayTransport) -> Provider {
    TypeSafe::builder("ts-key")
        .unwrap()
        .http_transport(transport.clone())
        .build()
        .unwrap()
}

const CLOUDFLARE_ACCOUNT: &str = "0123456789abcdef0123456789abcdef";

fn cloudflare(transport: &ReplayTransport) -> Provider {
    Cloudflare::builder(CLOUDFLARE_ACCOUNT, "cf-token")
        .unwrap()
        .http_transport(transport.clone())
        .build()
        .unwrap()
}

fn sent_body(transport: &ReplayTransport) -> Value {
    serde_json::from_str(transport.requests()[0].body_text().unwrap()).unwrap()
}

fn ticket() -> DecisionInput {
    json!({"ticket": "I was charged twice. Please refund the extra payment."}).into()
}

fn triage_request() -> DecisionRequest {
    DecisionRequest::new("nimble", ticket())
        .question(
            "label",
            Question::choice("Which label fits this ticket?")
                .option("billing", "Payments and refunds")
                .bare_option("bug")
                .option("account", "Login and account access"),
        )
        .question(
            "refund",
            Question::yes_no("Is the customer requesting a refund?")
                .no("No refund is requested")
                .yes("The customer requests a refund"),
        )
        .question(
            "urgency",
            Question::score("How urgently does this ticket need a response?")
                .level("Routine", "no time pressure")
                .level("Soon", "a customer is inconvenienced")
                .level("Immediate", "a critical service is unavailable"),
        )
}

#[tokio::test]
async fn ollama_triage_round_trip() {
    let transport = ReplayTransport::new([ReplayResponse::json(
        StatusCode::OK,
        fixture("ollama/triage.json"),
    )]);
    let response = ollama(&transport).decide(&triage_request()).await.unwrap();

    let request = &transport.requests()[0];
    assert_eq!(
        request.uri.to_string(),
        "http://localhost:11434/v1/systemone"
    );
    assert_eq!(request.headers.get("authorization"), None);
    assert_eq!(request.timeout, None);
    // The request of Ollama's decision guide, plus the label question.
    assert_eq!(
        sent_body(&transport),
        json!({
            "model": "nimble",
            "state": {"ticket": "I was charged twice. Please refund the extra payment."},
            "questions": {
                "label": {
                    "type": "choice",
                    "instructions": "Which label fits this ticket?",
                    "criteria": {"billing": "Payments and refunds", "bug": null, "account": "Login and account access"}
                },
                "refund": {
                    "type": "noul",
                    "instructions": "Is the customer requesting a refund?",
                    "criteria": {"false": "No refund is requested", "true": "The customer requests a refund"}
                },
                "urgency": {
                    "type": "score",
                    "instructions": "How urgently does this ticket need a response?",
                    "criteria": [
                        "Routine: no time pressure",
                        "Soon: a customer is inconvenienced",
                        "Immediate: a critical service is unavailable"
                    ]
                }
            }
        })
    );

    assert_eq!(response.model.as_deref(), Some("nimble"));
    assert_eq!(response.provider.as_deref(), Some("ollama"));
    let label = response.choice("label").unwrap();
    assert_eq!(label.chosen, "billing");
    assert_eq!(
        label.probabilities,
        [
            ("billing".to_owned(), 0.9781),
            ("bug".to_owned(), 0.0125),
            ("account".to_owned(), 0.0093)
        ]
    );
    assert_eq!(response.yes_no("refund").unwrap().probability, 0.9989);
    let urgency = response.score("urgency").unwrap();
    assert_eq!(urgency.score, 0.8308);
    assert_eq!(urgency.most_likely(), Some(1));
    let usage = response.usage.as_ref().unwrap();
    assert_eq!(
        (usage.input_tokens, usage.output_tokens),
        (Some(531), Some(4))
    );

    let confidence = response.provider_data.get::<Confidence>().unwrap().unwrap();
    assert_eq!(confidence.0["label"], 0.8906);
    assert_eq!(confidence.0["urgency"], 0.1371);
    assert!(
        response
            .provider_data
            .get::<RawResponse>()
            .unwrap()
            .is_none()
    );
    assert!(response.provider_data.get::<Gateway>().unwrap().is_none());
}

#[tokio::test]
async fn ollama_options_and_raw_response() {
    let transport = ReplayTransport::new([ReplayResponse::json(
        StatusCode::OK,
        fixture("ollama/label.json"),
    )]);
    let request = DecisionRequest::new("nimble", "Our checkout has returned 500 errors since 9am.")
        .question(
            "label",
            Question::choice("Which label fits this ticket?")
                .option("billing", "Payments and refunds")
                .option("bug", "Software errors")
                .option("account", "Login and account access"),
        )
        .image(Image::base64("image/png", "iVBORw0KGgo="))
        .with_option(
            DecisionRequestOptions::new()
                .keep_alive("10m")
                .raw_response(true),
        );
    let response = ollama(&transport).decide(&request).await.unwrap();

    let body = sent_body(&transport);
    assert_eq!(
        body["state"],
        json!("Our checkout has returned 500 errors since 9am.")
    );
    assert_eq!(body["keep_alive"], json!("10m"));
    assert_eq!(body["images"], json!(["iVBORw0KGgo="]));
    assert!(body.get("raw_response").is_none());

    assert_eq!(response.choice("label").unwrap().chosen, "bug");
    let raw = response
        .provider_data
        .get::<RawResponse>()
        .unwrap()
        .unwrap();
    assert_eq!(raw.0["usage"]["input_tokens"], json!(174));
}

#[tokio::test]
async fn typesafe_answer_types() {
    let transport = ReplayTransport::new([
        ReplayResponse::json(StatusCode::OK, fixture("typesafe/department.json"))
            .header("x-typesafe-request-id", HeaderValue::from_static("req_1")),
        ReplayResponse::json(StatusCode::OK, fixture("typesafe/frustration.json")),
        ReplayResponse::json(StatusCode::OK, fixture("typesafe/is_urgent.json")),
    ]);
    let provider = typesafe(&transport);
    let state = "Help! My payouts have been failing for 3 days.";

    let department = provider
        .decide(
            &DecisionRequest::new("jev-latest", state).question(
                "department",
                Question::choice("Which team should handle this?")
                    .option("billing", "Payments, invoicing, refunds")
                    .option("technical", "Bugs, outages, integrations")
                    .option("sales", "Pricing, upgrades, new accounts"),
            ),
        )
        .await
        .unwrap();
    assert_eq!(department.model.as_deref(), Some("jev-1.13.0"));
    assert_eq!(department.request_id.as_deref(), Some("req_1"));
    assert_eq!(department.provider.as_deref(), Some("typesafe"));
    assert_eq!(
        department
            .choice("department")
            .unwrap()
            .probability("sales"),
        Some(0.0)
    );

    let frustration = provider
        .decide(&DecisionRequest::new("jev-latest", state).question(
            "frustration",
            Question::score("How frustrated is the customer?").levels([
                "Calm",
                "Frustrated",
                "Very angry",
            ]),
        ))
        .await
        .unwrap();
    let score = frustration.score("frustration").unwrap();
    assert_eq!(score.score, 1.05);
    assert_eq!(score.probabilities, [0.0, 0.95, 0.05]);
    assert_eq!(frustration.request_id, None);

    let urgent = provider
        .decide(
            &DecisionRequest::new("jev-latest", state).question(
                "is_urgent",
                Question::yes_no("Does this convey urgency?")
                    .yes("Explicitly time-sensitive")
                    .no("No urgency expressed"),
            ),
        )
        .await
        .unwrap();
    assert!(urgent.yes_no("is_urgent").unwrap().is_yes());
    assert!(
        urgent.provider_data.get::<Confidence>().unwrap().is_none(),
        "yes/no answers carry no confidence"
    );

    let requests = transport.requests();
    let request = &requests[0];
    assert_eq!(
        request.uri.to_string(),
        "https://api.typesafe.ai/v1/systemone"
    );
    assert_eq!(
        request.headers.get("authorization"),
        Some(&HeaderValue::from_static("Bearer ts-key"))
    );
    assert_eq!(
        request.headers.get("content-type"),
        Some(&HeaderValue::from_static("application/json"))
    );
    // TypeSafe's own example request for a score question.
    assert_eq!(
        serde_json::from_str::<Value>(requests[1].body_text().unwrap()).unwrap(),
        json!({
            "state": "Help! My payouts have been failing for 3 days.",
            "model": "jev-latest",
            "questions": {
                "frustration": {
                    "type": "score",
                    "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]
                }
            }
        })
    );
}

#[tokio::test]
async fn typesafe_recorded_response() {
    // The request that produced `typesafe/recorded_triage.json`.
    let request = DecisionRequest::new(
        "jev-latest",
        // From text: `json!` would sort the keys before the input sees them.
        serde_json::from_str::<DecisionInput>(
            r#"{"ticket":"Checkout fails with 500 since 9am","plan":"pro"}"#,
        )
        .unwrap(),
    )
    .question(
        "label",
        Question::choice("Which label fits?")
            .option("billing", "Payments")
            .bare_option("bug")
            .option("account", "Login"),
    )
    .question("spam", Question::yes_no("Is this spam?"))
    .question(
        "urgency",
        Question::score("How urgent?")
            .level("Routine", "no time pressure")
            .bare_level("Soon")
            .bare_level("Immediate"),
    );
    let transport = ReplayTransport::new([ReplayResponse::json(
        StatusCode::OK,
        fixture("typesafe/recorded_triage.json"),
    )
    .header(
        "x-typesafe-request-id",
        HeaderValue::from_static("req_01a0fde849f77561881e25028b09ad5a"),
    )]);
    let response = typesafe(&transport).decide(&request).await.unwrap();

    assert_eq!(
        transport.requests()[0].body_text().unwrap(),
        concat!(
            r#"{"model":"jev-latest","state":{"ticket":"Checkout fails with 500 since 9am","plan":"pro"},"questions":{"#,
            r#""label":{"type":"choice","instructions":"Which label fits?","criteria":{"billing":"Payments","bug":null,"account":"Login"}},"#,
            r#""spam":{"type":"noul","instructions":"Is this spam?"},"#,
            r#""urgency":{"type":"score","instructions":"How urgent?","criteria":["Routine: no time pressure","Soon","Immediate"]}}}"#,
        )
    );
    assert_eq!(response.model.as_deref(), Some("jev-1.13.0"));
    assert_eq!(
        response.request_id.as_deref(),
        Some("req_01a0fde849f77561881e25028b09ad5a")
    );
    // TypeSafe sorts the keys; the answer keeps the reported order.
    let label = response.choice("label").unwrap();
    assert_eq!(label.chosen, "billing");
    assert_eq!(label.probabilities[0], ("account".to_owned(), 0.0));
    assert_eq!(response.yes_no("spam").unwrap().probability, 0.08);
    let urgency = response.score("urgency").unwrap();
    assert_eq!(urgency.most_likely(), Some(2));
    let confidence = response.provider_data.get::<Confidence>().unwrap().unwrap();
    assert_eq!(confidence.0["label"], 0.28);
}

#[tokio::test]
async fn gateway_fields_are_provider_data() {
    let transport = ReplayTransport::new([ReplayResponse::json(
        StatusCode::OK,
        r#"{"id":"gen-123","provider":"TypeSafe","model":"typesafe/jev-1.13","answers":{"spam":{"type":"noul","noul":0.02}},"usage":{"input_tokens":40,"output_tokens":1,"cost":0.0000017}}"#,
    )]);
    let provider = OpenRouter::builder("or-key")
        .unwrap()
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let response = provider
        .decide(
            &DecisionRequest::new("~typesafe/jev-latest", "Win a free cruise!")
                .question("spam", Question::yes_no("Is this spam?")),
        )
        .await
        .unwrap();
    assert_eq!(
        transport.requests()[0].uri.to_string(),
        "https://openrouter.ai/api/v1/systemone"
    );
    assert_eq!(response.id.as_deref(), Some("gen-123"));
    let gateway = response.provider_data.get::<Gateway>().unwrap().unwrap();
    assert_eq!(gateway.provider.as_deref(), Some("TypeSafe"));
    assert_eq!(gateway.cost, Some(0.0000017));
}

#[tokio::test]
async fn error_bodies_with_a_success_status_are_errors() {
    let transport = ReplayTransport::new([
        ReplayResponse::json(
            StatusCode::OK,
            r#"{"error":{"message":"Upstream provider unavailable","code":502}}"#,
        ),
        ReplayResponse::json(
            StatusCode::OK,
            r#"{"error":{"message":"Rate limit exceeded","code":429}}"#,
        ),
        ReplayResponse::json(
            StatusCode::OK,
            r#"{"error":{"message":"Something odd","code":"weird"}}"#,
        ),
    ]);
    let provider = typesafe(&transport);
    let request = DecisionRequest::new("m", "x").question("q", Question::yes_no("?"));

    let err = provider.decide(&request).await.unwrap_err();
    assert!(matches!(err, Error::Provider { .. }), "{err:?}");
    assert_eq!(err.message(), "HTTP 502");
    assert_eq!(
        err.upstream_message(),
        Some("Upstream provider unavailable")
    );
    assert!(
        err.is_retryable(),
        "a 502 reported in the body is transient"
    );
    assert_eq!(err.context().unwrap().provider.as_deref(), Some("typesafe"));

    let err = provider.decide(&request).await.unwrap_err();
    assert!(matches!(err, Error::RateLimited { .. }), "{err:?}");

    let err = provider.decide(&request).await.unwrap_err();
    assert!(matches!(err, Error::Provider { .. }), "{err:?}");
    assert!(
        err.is_retryable(),
        "an unclassified gateway failure is retried"
    );
}

#[tokio::test]
async fn malformed_success_bodies_are_unexpected_responses() {
    let transport = ReplayTransport::new([ReplayResponse::new(
        StatusCode::OK,
        "<html><body>502 Bad Gateway SENTINEL</body></html>",
    )
    .header("x-typesafe-request-id", HeaderValue::from_static("req_1"))]);
    let provider = typesafe(&transport);
    let request = DecisionRequest::new("m", "x").question("q", Question::yes_no("?"));

    let err = provider.decide(&request).await.unwrap_err();
    assert!(matches!(err, Error::UnexpectedResponse { .. }), "{err:?}");
    assert!(!err.is_retryable());
    assert_eq!(err.status(), Some(200));
    assert_eq!(err.request_id(), Some("req_1"));
    assert_eq!(err.context().unwrap().provider.as_deref(), Some("typesafe"));
    assert!(!err.as_log().to_string().contains("SENTINEL"));
}

#[tokio::test]
async fn invalid_requests_fail_before_sending() {
    let transport = ReplayTransport::new([]);
    let provider = typesafe(&transport);

    let levels: Vec<String> = (0..11).map(|level| format!("Level {level}")).collect();
    let too_many = DecisionRequest::new("jev-latest", "x")
        .question("rating", Question::score("Rate it").levels(levels));
    let err = provider.decide(&too_many).await.unwrap_err();
    assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
    assert!(err.message().contains("at most 10"), "{err}");

    let overriding = DecisionRequest::new("jev-latest", "x")
        .question("q", Question::yes_no("?"))
        .with_options_json("systemone", json!({"extra": {"model": "other"}}));
    let err = provider.decide(&overriding).await.unwrap_err();
    assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");

    // A `null` input is no input; the model never reads the word `null`.
    let null_input =
        DecisionRequest::new("jev-latest", json!(null)).question("q", Question::yes_no("?"));
    let err = provider.decide(&null_input).await.unwrap_err();
    assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
    assert!(err.message().contains("input must not be blank"), "{err}");

    assert!(transport.requests().is_empty());
}

#[tokio::test]
async fn answers_that_do_not_match_the_request_are_rejected() {
    let transport = ReplayTransport::new([ReplayResponse::json(
        StatusCode::OK,
        fixture("ollama/label.json"),
    )]);
    // The fixture answers "label" with keys billing/bug/account; ask for others.
    let request = DecisionRequest::new("nimble", "x").question(
        "label",
        Question::choice("?")
            .bare_option("billing")
            .bare_option("other"),
    );
    let err = ollama(&transport).decide(&request).await.unwrap_err();
    assert!(matches!(err, Error::UnexpectedResponse { .. }), "{err:?}");
    assert!(err.message().contains("unknown option"), "{err}");
}

#[tokio::test]
async fn http_errors_are_mapped() {
    let transport = ReplayTransport::new([
        ReplayResponse::json(
            StatusCode::FORBIDDEN,
            r#"{"detail":{"error_type":"authentication_error","message":"Must supply an API key! Check your request and try again."}}"#,
        )
        .header("x-typesafe-request-id", HeaderValue::from_static("req_403"))
        // The configured request id header takes precedence.
        .header("x-request-id", HeaderValue::from_static("req_common")),
        ReplayResponse::json(StatusCode::TOO_MANY_REQUESTS, r#"{"detail":"Too many requests"}"#)
            .header("retry-after", HeaderValue::from_static("2"))
            .header("x-should-retry", HeaderValue::from_static("false")),
        ReplayResponse::json(StatusCode::from_u16(529).unwrap(), r#"{"detail":"Overloaded"}"#)
            .header("retry-after", HeaderValue::from_static("2"))
            .header("retry-after-ms", HeaderValue::from_static("1500")),
    ]);
    let provider = typesafe(&transport);
    let request = DecisionRequest::new("jev-latest", "x").question("q", Question::yes_no("?"));

    let err = provider.decide(&request).await.unwrap_err();
    assert!(err.is_auth(), "{err:?}");
    assert_eq!(err.request_id(), Some("req_403"));
    assert_eq!(err.status(), Some(403));

    let err = provider.decide(&request).await.unwrap_err();
    assert!(matches!(err, Error::RateLimited { .. }), "{err:?}");
    assert_eq!(err.retry_after(), Some(Duration::from_secs(2)));
    assert!(err.is_retryable());
    assert_eq!(err.server_retry_directive(), Some(false));

    let err = provider.decide(&request).await.unwrap_err();
    assert!(matches!(err, Error::Overloaded { .. }), "{err:?}");
    assert_eq!(err.retry_after(), Some(Duration::from_millis(1500)));
}

#[tokio::test]
async fn ollama_chat_model_is_unsupported() {
    // What Ollama answers when the model lacks the decision capability.
    let transport = ReplayTransport::new([ReplayResponse::json(
        StatusCode::BAD_REQUEST,
        r#"{"error":"qwen3:1.7b does not support decision"}"#,
    )]);
    let request = DecisionRequest::new("qwen3:1.7b", "x").question("q", Question::yes_no("?"));
    let err = ollama(&transport).decide(&request).await.unwrap_err();
    assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
    assert_eq!(err.context().unwrap().provider.as_deref(), Some("ollama"));
}

#[derive(Debug, PartialEq, ChoiceSet)]
#[decision(rename_all = "lowercase")]
enum Label {
    #[decision(description = "Payments and refunds")]
    Billing,
    Bug,
    #[decision(description = "Login and account access")]
    Account,
}

#[derive(Debug, PartialEq, Scale)]
enum Urgency {
    #[decision(description = "no time pressure")]
    Routine,
    #[decision(description = "a customer is inconvenienced")]
    Soon,
    #[decision(description = "a critical service is unavailable")]
    Immediate,
}

#[derive(Decision)]
struct Triage {
    #[decision(instructions = "Which label fits this ticket?")]
    label: Choice<Label>,
    #[decision(
        instructions = "Is the customer requesting a refund?",
        yes = "The customer requests a refund",
        no = "No refund is requested"
    )]
    refund: YesNoAnswer,
    #[decision(instructions = "How urgently does this ticket need a response?")]
    urgency: Score<Urgency>,
}

#[tokio::test]
async fn typed_decision_round_trip() {
    let transport = ReplayTransport::new([
        ReplayResponse::json(StatusCode::OK, fixture("ollama/triage.json")),
        ReplayResponse::json(StatusCode::OK, fixture("ollama/triage.json")),
    ]);
    let provider = ollama(&transport);
    provider.decide(&triage_request()).await.unwrap();
    let decided = provider
        .decide_as::<Triage>("nimble", ticket())
        .await
        .unwrap();

    // The derived questions are the hand-written ones, byte for byte.
    let requests = transport.requests();
    assert_eq!(requests[1].body_text(), requests[0].body_text());
    let triage = decided.value;
    assert_eq!(*triage.label.value(), Label::Billing);
    assert!(triage.refund.is_yes());
    assert_eq!(triage.urgency.most_likely(), Some(Urgency::Soon));
    assert!(decided.response.usage.is_some());
}

/// Ollama before 0.35 answers the System One route with a plain-text 404:
/// the Ollama preset's refusal says which version serves it and what to
/// pull, and the generic provider's says only that the endpoint is missing.
#[tokio::test]
async fn a_missing_endpoint_names_the_ollama_version_on_the_ollama_preset() {
    let missing = || ReplayResponse::new(StatusCode::NOT_FOUND, "404 page not found");

    let transport = ReplayTransport::new([missing()]);
    let err = ollama(&transport)
        .decide(&triage_request())
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
    assert_eq!(
        err.message(),
        "the server has no System One endpoint here (HTTP 404); Ollama serves System One \
         from 0.35: pull a decision model such as `nimble`"
    );

    let transport = ReplayTransport::new([missing()]);
    let generic = Provider::builder()
        .base_url("http://localhost:8000")
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let err = generic.decide(&triage_request()).await.unwrap_err();
    assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
    assert_eq!(
        err.message(),
        "the server has no System One endpoint here (HTTP 404)"
    );
}

/// A LiteLLM proxy without the TypeSafe pass-through answers FastAPI's bare
/// `404`: the LiteLLM preset's refusal says what the proxy needs.
#[tokio::test]
async fn a_missing_endpoint_names_the_pass_through_on_the_litellm_preset() {
    let transport = ReplayTransport::new([ReplayResponse::json(
        StatusCode::NOT_FOUND,
        r#"{"detail": "Not Found"}"#,
    )]);
    let litellm = LiteLlm::builder("http://127.0.0.1:4000", Some("sk-litellm"))
        .unwrap()
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let err = litellm.decide(&triage_request()).await.unwrap_err();
    assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
    assert!(
        err.message().contains("TypeSafe pass-through"),
        "{}",
        err.message()
    );
    assert!(
        err.message().contains("TYPESAFE_API_KEY"),
        "{}",
        err.message()
    );
}

/// An error reported with a success status keeps the response's
/// headers, so its retry hint is read.
#[tokio::test]
async fn success_status_errors_keep_their_headers() {
    let transport = ReplayTransport::new([ReplayResponse::json(
        StatusCode::OK,
        r#"{"error":{"message":"Rate limit exceeded","code":429}}"#,
    )
    .header("retry-after", HeaderValue::from_static("7"))]);
    let err = typesafe(&transport)
        .decide(&DecisionRequest::new("m", "x").question("q", Question::yes_no("?")))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::RateLimited { .. }), "{err:?}");
    assert_eq!(err.retry_after(), Some(Duration::from_secs(7)));
}

/// `keep_alive` is Ollama's; another endpoint is not sent it, and
/// the hint is dropped with a warning. TypeSafe's protocol takes no images,
/// so they are refused before sending.
#[tokio::test]
async fn ollama_options_go_to_ollama_only() {
    let request = || DecisionRequest::new("nimble", "x").question("q", Question::yes_no("?"));
    let keep_alive = DecisionRequestOptions::new().keep_alive("10m");
    let with_image = || request().image(Image::base64("image/png", "aGVsbG8="));

    let answer = r#"{"model":"nimble","answers":{"q":{"type":"noul","noul":0.1}}}"#;
    let transport = ReplayTransport::new([
        ReplayResponse::json(StatusCode::OK, answer),
        ReplayResponse::json(StatusCode::OK, answer),
    ]);
    let provider = ollama(&transport);
    provider
        .decide(&request().with_option(keep_alive.clone()))
        .await
        .unwrap();
    provider.decide(&with_image()).await.unwrap();
    let bodies: Vec<Value> = transport
        .requests()
        .iter()
        .map(|r| serde_json::from_str(r.body_text().unwrap()).unwrap())
        .collect();
    assert_eq!(bodies[0]["keep_alive"], "10m");
    assert_eq!(bodies[1]["images"], json!(["aGVsbG8="]));

    let transport = ReplayTransport::new([ReplayResponse::json(StatusCode::OK, answer)]);
    let provider = typesafe(&transport);
    let response = provider
        .decide(&request().with_option(keep_alive))
        .await
        .unwrap();
    assert!(sent_body(&transport).get("keep_alive").is_none());
    assert_eq!(response.warnings.len(), 1, "{:?}", response.warnings);
    let err = provider
        .validate_decision_request(&with_image())
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
    assert!(err.message().contains("does not take images"), "{err}");
}

/// Builder headers go with every request
/// after the default User-Agent, a request's own headers replace them, and
/// a credential header is refused at `build()`.
#[tokio::test]
async fn builder_headers_are_sent_with_every_request() {
    let transport =
        ReplayTransport::new([ReplayResponse::new(StatusCode::INTERNAL_SERVER_ERROR, "")]);
    let provider = TypeSafe::builder("ts-key")
        .unwrap()
        .header("x-route", "eu")
        .header("x-tenant", "acme")
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let request = triage_request().with_option(
        allama_http::HttpRequestOptions::new()
            .header("x-tenant", HeaderValue::from_static("other")),
    );
    let _ = provider.decide(&request).await;
    let sent = &transport.requests()[0];
    assert_eq!(
        sent.headers.get("user-agent"),
        Some(&HeaderValue::try_from(allama_http::DEFAULT_USER_AGENT).unwrap())
    );
    assert_eq!(
        sent.headers.get("x-route"),
        Some(&HeaderValue::from_static("eu"))
    );
    assert_eq!(
        sent.headers.get("x-tenant"),
        Some(&HeaderValue::from_static("other"))
    );
    assert_eq!(
        sent.headers.get("authorization"),
        Some(&HeaderValue::from_static("Bearer ts-key"))
    );

    let user_agent =
        ReplayTransport::new([ReplayResponse::new(StatusCode::INTERNAL_SERVER_ERROR, "")]);
    let provider = ollama_builder()
        .header("User-Agent", "my-app/1.0")
        .http_transport(user_agent.clone())
        .build()
        .unwrap();
    let _ = provider.decide(&triage_request()).await;
    assert_eq!(
        user_agent.requests()[0].headers.get("user-agent"),
        Some(&HeaderValue::from_static("my-app/1.0"))
    );

    for name in ["Authorization", "x-api-key"] {
        let Err(err) = ollama_builder()
            .header(name, "v")
            .http_transport(ReplayTransport::default())
            .build()
        else {
            panic!("built with a `{name}` header");
        };
        assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    }
}

/// A success body past `max_response_size` fails the
/// call as an unexpected response, which is not retried.
#[tokio::test]
async fn the_response_size_is_capped() {
    let body = fixture("ollama/triage.json");
    let provider = |cap: usize| {
        ollama_builder()
            .max_response_size(cap)
            .http_transport(ReplayTransport::new([ReplayResponse::json(
                StatusCode::OK,
                body.clone(),
            )]))
            .build()
            .unwrap()
    };
    assert!(provider(body.len()).decide(&triage_request()).await.is_ok());
    let err = provider(body.len() - 1)
        .decide(&triage_request())
        .await
        .unwrap_err();
    assert!(matches!(err, Error::UnexpectedResponse { .. }), "{err:?}");
    assert!(!err.is_retryable());
}

fn ollama_builder() -> allama_systemone::ProviderBuilder {
    Ollama::builder("http://localhost:11434").unwrap()
}

/// The example request of Cloudflare's Clef page, and the response
/// recorded for it (`cloudflare/triage.json`).
fn clef_triage_request(model: &str) -> DecisionRequest {
    DecisionRequest::new(
        model,
        "Checkout has been failing for every customer for the last hour.",
    )
    .question(
        "urgent",
        Question::yes_no("Is this support request urgent?"),
    )
    .question(
        "team",
        Question::choice("Which team should handle this request?")
            .option("billing", "Payments, invoices, and refunds")
            .option("technical", "Outages, errors, and configuration")
            .option("sales", "Plans and upgrades"),
    )
    .question(
        "severity",
        Question::score("How severe is the customer impact?").levels([
            "No impact",
            "Minor",
            "Major",
            "Critical",
        ]),
    )
}

#[tokio::test]
async fn cloudflare_recorded_response() {
    let transport = ReplayTransport::new([ReplayResponse::json(
        StatusCode::OK,
        fixture("cloudflare/triage.json"),
    )]);
    let response = cloudflare(&transport)
        .decide(&clef_triage_request("@cf/cloudflare/clef"))
        .await
        .unwrap();

    let request = &transport.requests()[0];
    assert_eq!(
        request.uri.to_string(),
        format!(
            "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT}/ai/run/@cf/cloudflare/clef"
        )
    );
    assert_eq!(
        request.headers.get("authorization"),
        Some(&HeaderValue::from_static("Bearer cf-token"))
    );
    // The body of Cloudflare's own example, with its model selector.
    assert_eq!(
        sent_body(&transport),
        json!({
            "model": "clef",
            "state": "Checkout has been failing for every customer for the last hour.",
            "questions": {
                "urgent": {"type": "noul", "instructions": "Is this support request urgent?"},
                "team": {
                    "type": "choice",
                    "instructions": "Which team should handle this request?",
                    "criteria": {
                        "billing": "Payments, invoices, and refunds",
                        "technical": "Outages, errors, and configuration",
                        "sales": "Plans and upgrades"
                    }
                },
                "severity": {
                    "type": "score",
                    "instructions": "How severe is the customer impact?",
                    "criteria": ["No impact", "Minor", "Major", "Critical"]
                }
            }
        })
    );

    assert_eq!(response.provider.as_deref(), Some("cloudflare"));
    assert_eq!(response.model.as_deref(), Some("clef"));
    assert_eq!(response.yes_no("urgent").unwrap().probability, 0.9895);
    let team = response.choice("team").unwrap();
    assert_eq!(team.chosen, "technical");
    assert_eq!(team.probabilities[1], ("technical".to_owned(), 0.8274));
    let severity = response.score("severity").unwrap();
    assert_eq!(severity.score, 2.9484);
    assert_eq!(severity.probabilities, [0.0043, 0.0047, 0.0292, 0.9618]);
    let usage = response.usage.unwrap();
    assert_eq!(usage.input_tokens, Some(346));
    assert_eq!(usage.output_tokens, Some(0));
    let confidence = response.provider_data.get::<Confidence>().unwrap().unwrap();
    assert_eq!(confidence.0["team"], 0.5648);
    assert_eq!(confidence.0["severity"], 0.9012);
}

#[tokio::test]
async fn cloudflare_selects_clef_flash_by_name() {
    let transport = ReplayTransport::new([ReplayResponse::json(
        StatusCode::OK,
        fixture("cloudflare/flash.json"),
    )]);
    let request = DecisionRequest::new("@cf/cloudflare/clef-flash", "Checkout fails for everyone.")
        .question("q", Question::yes_no("Is this an outage?"));
    let response = cloudflare(&transport).decide(&request).await.unwrap();
    assert!(
        transport.requests()[0]
            .uri
            .to_string()
            .ends_with("/ai/run/@cf/cloudflare/clef-flash")
    );
    assert_eq!(sent_body(&transport)["model"], "clef-flash");
    assert_eq!(response.model.as_deref(), Some("clef-flash"));
    assert_eq!(response.yes_no("q").unwrap().probability, 0.9215);
}

#[tokio::test]
async fn cloudflare_sends_images_with_their_media_type() {
    let transport = ReplayTransport::new([ReplayResponse::json(
        StatusCode::OK,
        fixture("cloudflare/image.json"),
    )]);
    let request = DecisionRequest::new("@cf/cloudflare/clef", "A product photo.")
        .image(Image::base64("image/png", "iVBORw0KGgo="))
        .question(
            "color",
            Question::choice("What is the main color of the image?")
                .options(["red", "green", "blue"]),
        );
    let response = cloudflare(&transport).decide(&request).await.unwrap();
    assert_eq!(
        sent_body(&transport)["images"],
        json!([{"content_type": "image/png", "base64": "iVBORw0KGgo="}])
    );
    assert_eq!(response.choice("color").unwrap().chosen, "red");
}

/// Recorded error responses (2026-10-13) map to their kinds, with
/// Cloudflare's message and code kept.
#[tokio::test]
async fn cloudflare_errors() {
    for (status, file, kind) in [
        (400, "unknown_model", "model_not_found"),
        (401, "bad_auth", "auth"),
        (422, "bad_name", "invalid_request"),
        (422, "mismatch", "invalid_request"),
    ] {
        let transport = ReplayTransport::new([ReplayResponse::json(
            StatusCode::from_u16(status).unwrap(),
            fixture(&format!("cloudflare/{file}.json")),
        )]);
        let err = cloudflare(&transport)
            .decide(&clef_triage_request("@cf/cloudflare/clef"))
            .await
            .unwrap_err();
        assert_eq!(err.kind_name(), kind, "{file}: {err:?}");
        assert_eq!(
            err.context().unwrap().provider.as_deref(),
            Some("cloudflare"),
            "{file}"
        );
        assert!(err.upstream_message().is_some(), "{file}: {err:?}");
        assert!(
            err.context().unwrap().provider_code.is_some(),
            "{file}: {err:?}"
        );
    }
}

/// Only what no Workers AI decision model can take is refused before
/// sending: an image that is not inline data, and a model name that would
/// leave the account's `run` path.
#[tokio::test]
async fn cloudflare_refuses_before_sending() {
    let transport = ReplayTransport::default();
    let provider = cloudflare(&transport);
    let base = |model: &str| DecisionRequest::new(model, "x").question("q", Question::yes_no("?"));
    for request in [
        base("@cf/cloudflare/clef").image(Image::url("https://example.com/a.png")),
        base("@cf/cloudflare/clef").image(Image::provider_file("openai", "file-1")),
        base("@cf/cloudflare/../../tokens"),
        base("@cf/cloudflare/clef?x=1"),
    ] {
        let err = provider.decide(&request).await.unwrap_err();
        assert!(
            matches!(
                err,
                Error::InvalidRequest { .. } | Error::Unsupported { .. }
            ),
            "{request:?}: {err:?}"
        );
    }
    assert!(transport.requests().is_empty());
}

/// What a model's input schema allows is model data (ADR 0015): the
/// request is sent, and Clef's refusal (HTTP 422, recorded in
/// `cloudflare/bad_name.json`) is an invalid request that names the field.
#[tokio::test]
async fn cloudflare_leaves_the_model_schema_to_the_model() {
    let base =
        || DecisionRequest::new("@cf/cloudflare/clef", "x").question("q", Question::yes_no("?"));
    let png = || Image::base64("image/png", "aGk=");
    let requests = [
        DecisionRequest::new("@cf/cloudflare/clef", "x")
            .question("has space", Question::yes_no("?")),
        base().image(Image::base64("image/gif", "aGk=")),
        (0..5).fold(base(), |request, _| request.image(png())),
        base().question(
            "levels",
            Question::score("?").levels((0..11).map(|i| format!("level {i}"))),
        ),
    ];
    let transport = ReplayTransport::new(requests.iter().map(|_| {
        ReplayResponse::json(
            StatusCode::UNPROCESSABLE_ENTITY,
            fixture("cloudflare/bad_name.json"),
        )
    }));
    let provider = cloudflare(&transport);
    for request in &requests {
        let err = provider.decide(request).await.unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
        assert!(
            err.upstream_message()
                .unwrap()
                .contains("failed validation"),
            "{err:?}"
        );
    }
    assert_eq!(transport.requests().len(), requests.len());
    assert_eq!(
        serde_json::from_str::<Value>(transport.requests()[1].body_text().unwrap()).unwrap()["images"],
        json!([{"content_type": "image/gif", "base64": "aGk="}])
    );
}

/// A success status with Cloudflare's error envelope is an error, not a
/// malformed response.
#[tokio::test]
async fn cloudflare_error_envelope_with_a_success_status() {
    let transport = ReplayTransport::new([ReplayResponse::json(
        StatusCode::OK,
        r#"{"result":null,"success":false,"errors":[{"code":3040,"message":"Capacity temporarily exceeded"}],"messages":[]}"#,
    )]);
    let err = cloudflare(&transport)
        .decide(&clef_triage_request("@cf/cloudflare/clef"))
        .await
        .unwrap_err();
    assert_eq!(
        err.upstream_message(),
        Some("Capacity temporarily exceeded")
    );
    assert!(err.is_retryable(), "{err:?}");
}
