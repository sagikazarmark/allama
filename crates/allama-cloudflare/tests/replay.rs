//! Replays recorded and documented Workers AI bodies through the provider.

use allama::extraction::{Field, ListItemKind};
use allama::{
    Error, ExtractionProvider, ExtractionRequest, Image, ProviderIdentityExt, Subject, WarningKind,
};
use allama_cloudflare::{
    Confidence, Distributions, ExtractionRequestOptions, Guesses, Provider, RawResponse, RunDetails,
};
use allama_http::http::{HeaderValue, StatusCode};
use allama_http::{Credential, HttpRequestOptions, ReplayResponse, ReplayTransport};
use serde_json::Value;

fn fixture(path: &str) -> String {
    let path = format!("{}/fixtures/{path}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("{path}: {err}"))
}

fn provider(transport: &ReplayTransport) -> Provider {
    Provider::builder()
        .account_id("0123456789abcdef0123456789abcdef")
        .api_token("cf-token")
        .http_transport(transport.clone())
        .build()
        .unwrap()
}

fn sent_body(transport: &ReplayTransport) -> Value {
    let requests = transport.requests();
    serde_json::from_str(requests[0].body_text().expect("a JSON body")).unwrap()
}

/// The documentation example's input, laid out so that the page's spans
/// place: `INV-2291` at bytes 8..16, the contacts at 120..129 and 131..141.
const INVOICE: &str = "Invoice INV-2291 from Northwind Traders, issued 2026-09-02, is due 2026-10-02. Total: $1,080.00, net 30 days. Contacts: Dana Ruiz, Priya Shah.";

fn invoice_request() -> ExtractionRequest {
    ExtractionRequest::new("@cf/cloudflare/clairic", INVOICE)
        .field(
            "invoice_number",
            Field::text().instructions("Invoice number"),
        )
        .field("due_date", Field::date().instructions("Due date"))
        .field("total", Field::number().decimals(2).unit("USD"))
        .field(
            "contacts",
            Field::list(ListItemKind::Text).instructions("People named"),
        )
        .field(
            "po_number",
            Field::text().instructions("Purchase order number"),
        )
        .field(
            "days_to_pay",
            Field::estimate(0.0, 120.0)
                .unit("days")
                .instructions("Days until payment"),
        )
}

#[tokio::test]
async fn invoice_round_trip() {
    let transport = ReplayTransport::new([ReplayResponse::json(
        StatusCode::OK,
        fixture("clairic/invoice.json"),
    )]);
    let provider = provider(&transport);
    let request = invoice_request().with_option(ExtractionRequestOptions::new().raw_response());
    let response = provider.extract_values(&request).await.unwrap();

    let body = sent_body(&transport);
    assert_eq!(body["model"], "clairic");
    assert_eq!(body["state"], INVOICE);
    assert_eq!(body["fields"]["total"]["decimals"], 2);
    assert_eq!(body["fields"]["days_to_pay"]["max"], 120.0);
    assert!(body.get("images").is_none());
    let requests = transport.requests();
    assert!(
        requests[0]
            .uri
            .to_string()
            .ends_with("/ai/run/@cf/cloudflare/clairic"),
        "{}",
        requests[0].uri.to_string()
    );
    assert_eq!(
        requests[0].headers.get("authorization"),
        Some(&HeaderValue::from_static("Bearer cf-token"))
    );

    let invoice_number = response.text("invoice_number").unwrap();
    assert_eq!(invoice_number.value.as_deref(), Some("INV-2291"));
    assert_eq!(
        invoice_number.span.unwrap().text(&request.input),
        Some("INV-2291")
    );
    assert_eq!(
        response
            .date("due_date")
            .unwrap()
            .value
            .as_ref()
            .unwrap()
            .as_str(),
        "2026-10-02"
    );
    assert_eq!(response.number("total").unwrap().value, Some(1080.0));
    let contacts = response.list("contacts").unwrap();
    assert_eq!(
        contacts.texts().collect::<Vec<_>>(),
        ["Dana Ruiz", "Priya Shah"]
    );
    assert_eq!(
        contacts.items[1].span.unwrap().text(&request.input),
        Some("Priya Shah")
    );
    let po = response.text("po_number").unwrap();
    assert!(po.is_null());
    assert_eq!(po.null_probability, Some(1.0));
    assert_eq!(po.span, None);
    let days = response.estimate("days_to_pay").unwrap();
    assert_eq!(days.value, 30.0);
    assert_eq!(days.median, Some(30.0));
    assert_eq!(days.interval.unwrap().low, 21.0);
    assert_eq!(days.interval.unwrap().probability, 0.8);

    assert_eq!(response.model.as_deref(), Some("clairic"));
    assert_eq!(response.provider.as_deref(), Some("cloudflare"));
    let usage = response.usage.as_ref().unwrap();
    assert_eq!(usage.input_tokens, Some(655));
    assert_eq!(usage.breakdowns.get("schema_input_tokens"), Some(&472));
    assert!(usage.contract_violation().is_none());

    let confidence = response.provider_data.get::<Confidence>().unwrap().unwrap();
    assert_eq!(confidence.0["po_number"], 0.9995);
    let guesses = response.provider_data.get::<Guesses>().unwrap().unwrap();
    assert_eq!(guesses.0["po_number"], "INV-2291");
    let distributions = response
        .provider_data
        .get::<Distributions>()
        .unwrap()
        .unwrap();
    assert_eq!(distributions.0["days_to_pay"][0].mean, Some(29.535));
    let run = response.provider_data.get::<RunDetails>().unwrap().unwrap();
    assert_eq!(run.passes, Some(3));
    assert!(!run.truncated);
    let raw = response
        .provider_data
        .get::<RawResponse>()
        .unwrap()
        .unwrap();
    assert_eq!(raw.0["success"], true);
    assert!(response.warnings.is_empty(), "{:?}", response.warnings);
}

#[tokio::test]
async fn options_shape_the_body() {
    let transport = ReplayTransport::new([ReplayResponse::json(
        StatusCode::OK,
        fixture("clairic/invoice.json"),
    )]);
    let request = invoice_request()
        .image(Image::bytes("image/png", b"\x89PNG"))
        .reference_date("2026-10-09".parse().unwrap())
        .with_option(
            ExtractionRequestOptions::new()
                .null_threshold("po_number", 0.9)
                .include_quantiles()
                .return_state_text(),
        );
    provider(&transport).extract_values(&request).await.unwrap();
    let body = sent_body(&transport);
    assert_eq!(body["fields"]["po_number"]["null_threshold"], 0.9);
    assert_eq!(
        body["options"],
        serde_json::json!({"include_quantiles": true, "return_state_text": true})
    );
    assert_eq!(body["reference_date"], "2026-10-09");
    assert_eq!(body["images"][0]["content_type"], "image/png");
}

#[tokio::test]
async fn http_errors_are_mapped() {
    let cases = [
        (400, "clairic/unknown_model.json", "model_not_found"),
        (401, "clairic/bad_auth.json", "auth"),
        (400, "clairic/bad_field.json", "invalid_request"),
    ];
    for (status, path, kind) in cases {
        let transport = ReplayTransport::new([ReplayResponse::json(
            StatusCode::from_u16(status).unwrap(),
            fixture(path),
        )]);
        let err = provider(&transport)
            .extract_values(&invoice_request())
            .await
            .unwrap_err();
        assert_eq!(err.kind_name(), kind, "{path}: {err:?}");
        assert_eq!(err.status(), Some(status));
        assert!(err.upstream_message().is_some(), "{path}");
    }
}

#[tokio::test]
async fn an_error_envelope_with_a_success_status_is_an_error() {
    let transport = ReplayTransport::new([ReplayResponse::json(
        StatusCode::OK,
        fixture("clairic/unknown_model.json"),
    )]);
    let err = provider(&transport)
        .extract_values(&invoice_request())
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Provider { .. }), "{err:?}");
    assert_eq!(err.upstream_message(), Some("No route for that URI"));
}

/// JSON options under the provider's name are not the namespace this
/// provider reads; a dry run says so (ADR 0006). `run_options` belong to
/// the Worker binding and are dropped with a hint.
#[test]
fn options_the_provider_does_not_read_are_reported() {
    let provider = provider(&ReplayTransport::new([]));
    let request = invoice_request()
        .with_options_json("cloudflare", serde_json::json!({"include_quantiles": true}))
        .with_option(
            ExtractionRequestOptions::new().run_option("gateway", serde_json::json!({"id": "g"})),
        );
    let validation = provider.validate_extraction_request(&request).unwrap();
    let codes: Vec<_> = validation
        .warnings()
        .iter()
        .map(|warning| (warning.kind, warning.code.as_deref()))
        .collect();
    assert_eq!(
        codes,
        [
            (WarningKind::Dropped, Some("dropped.provider_name_options")),
            (WarningKind::Unsupported, None),
        ]
    );
    assert_eq!(
        validation.warnings()[0].subject,
        Subject::Option {
            namespace: "cloudflare".to_owned()
        }
    );
}

#[tokio::test]
async fn invalid_requests_fail_before_sending() {
    let transport = ReplayTransport::new([]);
    let provider = provider(&transport);
    let err = provider
        .extract_values(&ExtractionRequest::new("@cf/cloudflare/clairic", "x"))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
    let err = provider
        .extract_values(
            &ExtractionRequest::new("@cf/cloudflare/clairic", "x")
                .image(Image::url("https://example.com/a.png"))
                .field("f", Field::text()),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
    let err = provider
        .extract_values(&ExtractionRequest::new("../oops", "x").field("f", Field::text()))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
    // The schema's counts are the provider's limits.
    let limits = provider.extraction_limits("@cf/cloudflare/clairic");
    assert_eq!(
        (limits.max_images, limits.max_forecast_horizon),
        (Some(4), Some(64))
    );
    let mut request = ExtractionRequest::new("@cf/cloudflare/clairic", "x")
        .field("f", Field::forecast(0.0, 1.0).horizon(65));
    let err = provider.extract_values(&request).await.unwrap_err();
    assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
    request.fields.clear();
    for _ in 0..5 {
        request = request.image(Image::bytes("image/png", b"\x89PNG"));
    }
    let err = provider
        .extract_values(&request.field("f", Field::text()))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
    assert!(transport.requests().is_empty());
}

#[tokio::test]
async fn answers_that_do_not_match_the_request_are_rejected() {
    let transport = ReplayTransport::new([ReplayResponse::json(
        StatusCode::OK,
        fixture("clairic/invoice.json"),
    )]);
    let request = ExtractionRequest::new("@cf/cloudflare/clairic", INVOICE)
        .field("invoice_number", Field::number());
    let err = provider(&transport)
        .extract_values(&request)
        .await
        .unwrap_err();
    assert!(matches!(err, Error::UnexpectedResponse { .. }), "{err:?}");
}

#[tokio::test]
async fn a_per_request_credential_replaces_the_token() {
    let transport = ReplayTransport::new([ReplayResponse::json(
        StatusCode::OK,
        fixture("clairic/invoice.json"),
    )]);
    let request = invoice_request().with_option(
        HttpRequestOptions::new().credential("cloudflare", Credential::api_key("per-request")),
    );
    provider(&transport).extract_values(&request).await.unwrap();
    assert_eq!(
        transport.requests()[0].headers.get("authorization"),
        Some(&HeaderValue::from_static("Bearer per-request"))
    );
}

#[test]
fn the_provider_is_reachable_by_downcast() {
    let provider = provider(&ReplayTransport::default());
    assert!(provider.downcast_ref::<Provider>().is_some());
}

/// A non-ASCII input with a repeated value (ADR 0018, decision 7): the
/// model's character offsets become byte spans that equal the value, the
/// repeated value lands on the occurrence the model named, and nothing is
/// dropped.
#[tokio::test]
async fn spans_place_on_non_ascii_input() {
    let input =
        "Café Zoë — réf. A-1 (ancienne réf. A-1 annulée). Nouvelle réf. B-2, livrée le 3 mars.";
    let transport = ReplayTransport::new([ReplayResponse::json(
        StatusCode::OK,
        fixture("clairic/spans.json"),
    )]);
    let request = ExtractionRequest::new("@cf/cloudflare/clairic", input)
        .field("new_ref", Field::text().instructions("The new reference"))
        .field(
            "old_ref",
            Field::text().instructions("The cancelled reference"),
        )
        .field("refs", Field::list(ListItemKind::Text))
        .field("delivered_on", Field::date())
        .reference_date("2026-10-09".parse().unwrap());
    let response = provider(&transport).extract_values(&request).await.unwrap();
    assert!(response.warnings.is_empty(), "{:?}", response.warnings);

    let new_ref = response.text("new_ref").unwrap();
    assert_eq!(new_ref.span.unwrap().text(input), Some("B-2"));
    assert_eq!(new_ref.span.unwrap().start, input.find("B-2").unwrap());
    let old_ref = response.text("old_ref").unwrap();
    assert_eq!(old_ref.span.unwrap().start, input.rfind("A-1").unwrap());
    let refs = response.list("refs").unwrap();
    let starts: Vec<usize> = refs
        .items
        .iter()
        .map(|item| item.span.unwrap().start)
        .collect();
    assert_eq!(
        starts,
        [
            input.find("A-1").unwrap(),
            input.rfind("A-1").unwrap(),
            input.find("B-2").unwrap()
        ]
    );
    for item in &refs.items {
        let text = item.span.unwrap().text(input).unwrap();
        assert!(matches!(&item.value, allama::extraction::ListValue::Text(value) if value == text));
    }
    assert_eq!(
        response
            .date("delivered_on")
            .unwrap()
            .value
            .as_ref()
            .unwrap()
            .as_str(),
        "2026-03-03"
    );
}
