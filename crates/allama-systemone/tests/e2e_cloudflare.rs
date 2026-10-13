//! Live checks against Cloudflare Workers AI's Clef models. Ignored by
//! default:
//!
//! ```text
//! CLOUDFLARE_ACCOUNT_ID=... CLOUDFLARE_API_TOKEN=... \
//!     cargo test -p allama-systemone --test e2e_cloudflare -- --ignored
//! ```
//!
//! `CLOUDFLARE_DECISION_MODEL` (default `@cf/cloudflare/clef`) selects the
//! model.

// Sends through the default `reqwest` transport.
#![cfg(feature = "reqwest")]

mod support;

use allama::decision::Question;
use allama::{DecisionProvider, DecisionRequest, Image};
use allama_systemone::Provider;
use allama_systemone::providers::Cloudflare;

fn provider() -> (Provider, String) {
    let model =
        std::env::var("CLOUDFLARE_DECISION_MODEL").unwrap_or_else(|_| "@cf/cloudflare/clef".into());
    (Cloudflare::from_env().unwrap(), model)
}

#[tokio::test]
#[ignore]
async fn untyped() {
    let (provider, model) = provider();
    support::untyped(&provider, &model).await;
}

#[tokio::test]
#[ignore]
async fn typed() {
    let (provider, model) = provider();
    support::typed(&provider, &model).await;
}

#[tokio::test]
#[ignore]
async fn unknown_model() {
    let (provider, _) = provider();
    support::unknown_model(&provider).await;
}

/// A 64×64 red PNG, which Clef reads together with the input.
#[tokio::test]
#[ignore]
async fn image() {
    const RED: &str = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAb0lEQVR4nO3PAQkAAAyEwO9feoshgnABdLep8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3IPanc8OLDQitxAAAAAElFTkSuQmCC";
    let (provider, model) = provider();
    let request = DecisionRequest::new(&model, "A product photo.")
        .image(Image::base64("image/png", RED))
        .question(
            "color",
            Question::choice("What is the main color of the image?")
                .options(["red", "green", "blue"]),
        );
    let response = provider.decide(&request).await.unwrap();
    assert_eq!(response.choice("color").unwrap().chosen, "red");
}
