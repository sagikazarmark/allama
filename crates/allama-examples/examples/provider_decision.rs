//! Triage a support ticket with a decision model, optionally with a
//! screenshot the customer attached.
//!
//! Uses TypeSafe when `TYPESAFE_API_KEY` is set (model `TYPESAFE_MODEL`,
//! default `jev-latest`), Cloudflare Workers AI when `CLOUDFLARE_API_TOKEN`
//! is (with `CLOUDFLARE_ACCOUNT_ID`; model `CLOUDFLARE_DECISION_MODEL`,
//! default `@cf/cloudflare/clef`), otherwise a local Ollama 0.35+
//! (`OLLAMA_HOST`, model `OLLAMA_DECISION_MODEL`, default `nimble`):
//!
//! ```text
//! cargo run -p allama-examples --example provider_decision -- "I was charged twice."
//! cargo run -p allama-examples --example provider_decision -- "Checkout fails." error.png
//! ```
//!
//! A screenshot (PNG, JPEG or WebP) needs a model that reads images, such
//! as Clef; TypeSafe refuses it before sending.

use std::process::ExitCode;

use allama::decision::{Choice, ChoiceSet, Decision, Scale, Score, YesNoAnswer};
use allama::prelude::*;
use allama::{Image, ProviderIdentity};
use allama_systemone::Confidence;
use allama_systemone::providers::{Cloudflare, Ollama, TypeSafe};

#[derive(Debug, ChoiceSet)]
#[decision(rename_all = "lowercase")]
enum Label {
    #[decision(description = "Payments and refunds")]
    Billing,
    #[decision(description = "Software errors")]
    Bug,
    #[decision(description = "Login and account access")]
    Account,
}

#[derive(Debug, Scale)]
enum Urgency {
    #[decision(description = "No time pressure")]
    Routine,
    #[decision(description = "A customer is inconvenienced")]
    Soon,
    #[decision(description = "A critical service is unavailable")]
    Immediate,
}

#[derive(Decision)]
struct Triage {
    #[decision(instructions = "Which label fits this ticket?")]
    label: Choice<Label>,
    #[decision(
        instructions = "Is the customer requesting a refund?",
        yes = "The customer requests a refund"
    )]
    refund: YesNoAnswer,
    #[decision(instructions = "How urgently does this ticket need a response?")]
    urgency: Score<Urgency>,
}

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
    let ticket = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "I was charged twice. Please refund the extra payment.".into());
    let screenshot = std::env::args().nth(2).map(screenshot).transpose()?;

    let (provider, model) = if std::env::var("TYPESAFE_API_KEY").is_ok() {
        let model = std::env::var("TYPESAFE_MODEL").unwrap_or_else(|_| "jev-latest".into());
        (TypeSafe::from_env()?, model)
    } else if std::env::var("CLOUDFLARE_API_TOKEN").is_ok() {
        let model = std::env::var("CLOUDFLARE_DECISION_MODEL")
            .unwrap_or_else(|_| "@cf/cloudflare/clef".into());
        (Cloudflare::from_env()?, model)
    } else {
        let model = std::env::var("OLLAMA_DECISION_MODEL").unwrap_or_else(|_| "nimble".into());
        (Ollama::from_env()?, model)
    };
    eprintln!("provider={} model={model}", provider.provider_name());

    // The struct is the questions and receives the answers; the response
    // keeps usage and the provider's own statistics. A screenshot goes on
    // the request the struct builds.
    let (triage, response) = match screenshot {
        None => {
            let decided = provider
                .decide_as::<Triage>(&model, ticket.as_str())
                .await?;
            (decided.value, decided.response)
        }
        Some(image) => {
            let request = Triage::request(&model, ticket.as_str()).image(image);
            let response = provider.decide(&request).await?;
            (Triage::from_response(&response)?, response)
        }
    };
    println!("label:   {:?}", triage.label.value());
    for (label, probability) in triage.label.probabilities() {
        println!("         {label:?}: {probability:.4}");
    }
    println!("refund:  p(yes) = {:.4}", triage.refund.probability);
    println!(
        "urgency: score {:.2}, most likely {:?}, nearest {:?}",
        triage.urgency.score(),
        triage.urgency.most_likely(),
        triage.urgency.nearest()
    );

    if let Some(confidence) = response.provider_data.get::<Confidence>()? {
        eprintln!(
            "provider confidence (not comparable across providers): {:?}",
            confidence.0
        );
    }
    if let Some(usage) = &response.usage {
        eprintln!("usage={usage:?}");
    }
    Ok(())
}

/// The image file at `path`, its media type from its extension.
fn screenshot(path: String) -> Result<Image> {
    let media_type = match path
        .rsplit('.')
        .next()
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        _ => {
            return Err(Error::invalid_request(format!(
                "{path}: expected a .png, .jpg or .webp screenshot"
            )));
        }
    };
    let bytes =
        std::fs::read(&path).map_err(|err| Error::invalid_request(format!("{path}: {err}")))?;
    Ok(Image::bytes(media_type, bytes))
}
