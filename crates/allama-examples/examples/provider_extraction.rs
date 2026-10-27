//! Turn a damaged-product claim into typed values with an extraction model:
//! the order id (a span of the email), the total paid (computed from two
//! amounts), how late the claim is (a duration), the kinds of damage (a
//! list), a repair cost estimate inside bounds, and a field the email does
//! not answer (null, with its probability).
//!
//! Runs on Cloudflare Workers AI's Clairic (`CLOUDFLARE_ACCOUNT_ID` and
//! `CLOUDFLARE_API_TOKEN`; model `CLOUDFLARE_EXTRACTION_MODEL`, default
//! `@cf/cloudflare/clairic`):
//!
//! ```text
//! cargo run -p allama-examples --example provider_extraction
//! cargo run -p allama-examples --example provider_extraction -- "Order B-2 came late and dented. Paid $20 + $2 tax."
//! ```
//!
//! Extraction models are unstable in allama (ADR 0018): the API may change
//! in a minor release until a second provider implements it.

use std::process::ExitCode;

use allama::extraction::{DurationUnit, Field, ListItemKind};
use allama::prelude::*;
use allama_cloudflare::{Confidence, Guesses, Provider, RunDetails};

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<()> {
    let email = std::env::args().nth(1).unwrap_or_else(|| {
        "Hi, order A-88412 arrived on Sept 28, 2026 with a cracked screen and a dented corner. \
         I paid $649.00 plus $52.10 tax. It's now Oct 9, 2026 and I'd like this fixed or \
         refunded. Thanks, Dana"
            .into()
    });
    let model = std::env::var("CLOUDFLARE_EXTRACTION_MODEL")
        .unwrap_or_else(|_| "@cf/cloudflare/clairic".into());
    let provider = Provider::from_env()?;

    let request = ExtractionRequest::new(&model, &email)
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
        .field(
            "repair_cost",
            Field::estimate(10.0, 5000.0)
                .log_scale()
                .step(1.0)
                .unit("USD")
                .instructions("Cost to repair the damage described"),
        )
        .field("tracking_number", Field::text());

    // Validation is a dry run of the call: it finds a malformed request
    // and a field kind the model does not support before anything is sent,
    // and reports what the call will warn about.
    let validation = provider.validate_extraction_request(&request)?;
    for warning in validation.warnings() {
        println!("validation: {warning}");
    }

    let response = provider.extract_values(&request).await?;
    let confidence = response.provider_data.get::<Confidence>()?;
    let guesses = response.provider_data.get::<Guesses>()?;
    let certainty = |name: &str| {
        confidence
            .as_ref()
            .and_then(|c| c.0.get(name))
            .map_or(String::new(), |c| format!(" (confidence {c:.3})"))
    };

    let order_id = response.text("order_id").expect("asked");
    match (&order_id.value, order_id.span) {
        (Some(value), Some(span)) => println!(
            "order_id: {value} at bytes {}..{} of the email{}",
            span.start,
            span.end,
            certainty("order_id")
        ),
        (Some(value), None) => println!("order_id: {value}{}", certainty("order_id")),
        (None, _) => println!("order_id: not in the email"),
    }
    if let Some(date) = &response.date("delivered_on").expect("asked").value {
        println!("delivered_on: {date}{}", certainty("delivered_on"));
    }
    if let Some(paid) = response.number("amount_paid").expect("asked").value {
        println!(
            "amount_paid: ${paid:.2} (computed from the price and the tax){}",
            certainty("amount_paid")
        );
    }
    if let Some(days) = response
        .duration("days_since_delivery")
        .expect("asked")
        .value
    {
        println!("days_since_delivery: {days:.0} days (computed from two dates)");
    }
    let damage: Vec<_> = response.list("damage").expect("asked").texts().collect();
    println!("damage: {damage:?}");
    let repair = response.estimate("repair_cost").expect("asked");
    match repair.interval {
        Some(interval) => println!(
            "repair_cost: ${:.0}, {:.0}% likely between ${:.0} and ${:.0}",
            repair.value,
            interval.probability * 100.0,
            interval.low,
            interval.high
        ),
        None => println!("repair_cost: ${:.0}", repair.value),
    }
    let tracking = response.text("tracking_number").expect("asked");
    if tracking.is_null() {
        let guess = guesses
            .as_ref()
            .and_then(|g| g.0.get("tracking_number"))
            .map_or(String::new(), |g| {
                format!("; the model would have guessed {g}")
            });
        println!(
            "tracking_number: null (probability {:.2}{guess})",
            tracking.null_probability.unwrap_or(f64::NAN)
        );
    }

    if let Some(usage) = &response.usage {
        println!(
            "\n{} input tokens, {} output tokens ({})",
            usage.input_tokens.unwrap_or(0),
            usage.output_tokens.unwrap_or(0),
            response.provider.as_deref().unwrap_or("?")
        );
    }
    if let Some(run) = response.provider_data.get::<RunDetails>()? {
        println!("{} parallel pass(es)", run.passes.unwrap_or(1));
    }
    for warning in &response.warnings {
        println!("warning: {warning}");
    }
    Ok(())
}
