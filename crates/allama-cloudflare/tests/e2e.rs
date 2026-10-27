//! Live checks against Cloudflare Workers AI's Clairic. Ignored by default:
//!
//! ```text
//! CLOUDFLARE_ACCOUNT_ID=... CLOUDFLARE_API_TOKEN=... \
//!     cargo test -p allama-cloudflare --test e2e -- --ignored
//! ```
//!
//! `CLOUDFLARE_EXTRACTION_MODEL` (default `@cf/cloudflare/clairic`) selects
//! the model.

// Sends through the default `reqwest` transport.
#![cfg(feature = "reqwest")]

use allama::extraction::{DurationUnit, Field, ListItemKind};
use allama::{Error, ExtractionProvider, ExtractionRequest};
use allama_cloudflare::{Confidence, Provider, RunDetails};

fn provider() -> (Provider, String) {
    let model = std::env::var("CLOUDFLARE_EXTRACTION_MODEL")
        .unwrap_or_else(|_| "@cf/cloudflare/clairic".into());
    (Provider::from_env().unwrap(), model)
}

const EMAIL: &str = "Hi, order A-88412 arrived on Sept 28, 2026 with a cracked screen and a dented corner. I paid $649.00 plus $52.10 tax. It's now Oct 9, 2026 and I'd like this fixed or refunded. Thanks, Dana";

/// The blog post's support-email example: values, a computed total, a
/// list with spans, and a null.
#[tokio::test]
#[ignore]
async fn extracts_computes_and_reports_nulls() {
    let (provider, model) = provider();
    let request = ExtractionRequest::new(&model, EMAIL)
        .field("order_id", Field::text().instructions("Order number"))
        .field("delivered_on", Field::date().instructions("Delivery date"))
        .field(
            "amount_paid",
            Field::number()
                .decimals(2)
                .unit("USD")
                .instructions("Total paid including tax"),
        )
        .field(
            "days_since_delivery",
            Field::duration(DurationUnit::Days)
                .integer()
                .instructions("Days from delivery to the date the customer wrote"),
        )
        .field(
            "damage",
            Field::list(ListItemKind::Text).instructions("Each kind of damage described"),
        )
        .field("tracking_number", Field::text());
    let response = provider.extract_values(&request).await.unwrap();

    let order_id = response.text("order_id").unwrap();
    assert_eq!(order_id.value.as_deref(), Some("A-88412"));
    let span = order_id
        .span
        .expect("the span of a value in the input places");
    assert_eq!(span.text(&request.input), Some("A-88412"));
    assert_eq!(
        response
            .date("delivered_on")
            .unwrap()
            .value
            .as_ref()
            .unwrap()
            .as_str(),
        "2026-09-28"
    );
    let paid = response.number("amount_paid").unwrap().value.unwrap();
    assert!((paid - 701.1).abs() < 0.5, "{paid}");
    assert_eq!(
        response.duration("days_since_delivery").unwrap().value,
        Some(11.0)
    );
    let damage = response.list("damage").unwrap();
    assert_eq!(damage.items.len(), 2, "{damage:?}");
    assert!(
        damage.items.iter().all(|item| item.span.is_some()),
        "{damage:?}"
    );
    assert!(response.text("tracking_number").unwrap().is_null());

    let usage = response.usage.as_ref().unwrap();
    assert!(usage.input_tokens.unwrap() > 0);
    assert_eq!(usage.output_tokens, Some(0));
    assert!(usage.contract_violation().is_none(), "{usage:?}");
    let confidence = response.provider_data.get::<Confidence>().unwrap().unwrap();
    assert!(confidence.0.contains_key("order_id"));
    let run = response.provider_data.get::<RunDetails>().unwrap().unwrap();
    assert!(run.passes.unwrap_or(1) >= 1);
    assert!(response.warnings.is_empty(), "{:?}", response.warnings);
}

/// A non-ASCII input with a repeated value: spans place on the right
/// occurrence whatever offset unit the model reports (ADR 0018).
#[tokio::test]
#[ignore]
async fn spans_place_on_non_ascii_input() {
    let (provider, model) = provider();
    let input = "Café Zoë — réf. A-1 (ancienne réf. A-1 annulée). Nouvelle réf. B-2.";
    let request = ExtractionRequest::new(&model, input)
        .field("new_ref", Field::text().instructions("The new reference"))
        .field(
            "refs",
            Field::list(ListItemKind::Text).instructions("Every reference mentioned"),
        );
    let response = provider.extract_values(&request).await.unwrap();
    let new_ref = response.text("new_ref").unwrap();
    assert_eq!(new_ref.value.as_deref(), Some("B-2"));
    assert_eq!(new_ref.span.unwrap().text(input), Some("B-2"));
    for item in &response.list("refs").unwrap().items {
        let span = item.span.expect("every list item places");
        assert_eq!(Some(span.text(input).unwrap()), item.value_text());
    }
    assert!(response.warnings.is_empty(), "{:?}", response.warnings);
}

/// An estimate inside its bounds with an 80% interval, and a forecast
/// with one step per horizon.
#[tokio::test]
#[ignore]
async fn estimates_and_forecasts() {
    let (provider, model) = provider();
    let request = ExtractionRequest::new(
        &model,
        "Comparable sales:\nsqft,beds,year_built,neighborhood,sale_price\n1450,3,1998,Elm Park,412000\n2100,4,2006,Elm Park,585000\n980,2,1975,Riverside,265000\n1600,3,2003,Elm Park,449000\n1300,3,1990,Riverside,318000\n\nSubject property:\nsqft,beds,year_built,neighborhood\n1620,3,2001,Elm Park\n\nMonthly sales in Elm Park: 2026-05: 4, 2026-06: 6, 2026-07: 5, 2026-08: 7, 2026-09: 6",
    )
    .field(
        "sale_price",
        Field::estimate(50_000.0, 5_000_000.0)
            .log_scale()
            .step(1000.0)
            .unit("USD"),
    )
    .field(
        "monthly_sales",
        Field::forecast(0.0, 50.0)
            .horizon(2)
            .period(allama::extraction::ForecastPeriod::Month)
            .instructions("Sales in Elm Park in the next months"),
    );
    let response = provider.extract_values(&request).await.unwrap();
    let price = response.estimate("sale_price").unwrap();
    assert!((300_000.0..=600_000.0).contains(&price.value), "{price:?}");
    let interval = price.interval.expect("an 80% interval");
    assert!(
        interval.low <= price.value && price.value <= interval.high,
        "{price:?}"
    );
    assert_eq!(interval.probability, 0.8);
    let sales = response.forecast("monthly_sales").unwrap();
    assert_eq!(sales.steps.len(), 2);
    assert!(
        sales
            .steps
            .iter()
            .all(|step| (0.0..=50.0).contains(&step.value))
    );
}

#[tokio::test]
#[ignore]
async fn unknown_model_is_reported() {
    let (provider, _) = provider();
    let err = provider
        .extract_values(
            &ExtractionRequest::new("@cf/cloudflare/does-not-exist", "x").field("f", Field::text()),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, Error::ModelNotFound { .. }), "{err:?}");
}

trait ValueText {
    fn value_text(&self) -> Option<&str>;
}

impl ValueText for allama::extraction::ListItem {
    fn value_text(&self) -> Option<&str> {
        match &self.value {
            allama::extraction::ListValue::Text(text) => Some(text),
            _ => None,
        }
    }
}
