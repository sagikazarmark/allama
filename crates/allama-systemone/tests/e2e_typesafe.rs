//! Live checks against TypeSafe's hosted API. Ignored by default:
//!
//! ```text
//! TYPESAFE_API_KEY=... cargo test -p allama-systemone --test e2e_typesafe -- --ignored
//! ```
//!
//! `TYPESAFE_MODEL` (default `jev-latest`) selects the model.

// Sends through the default `reqwest` transport.
#![cfg(feature = "reqwest")]

mod support;

use allama::decision::Question;
use allama::{DecisionProvider, DecisionRequest, Image};
use allama_systemone::providers::TypeSafe;
use allama_systemone::{DecisionRequestOptions, Provider};

fn provider() -> (Provider, String) {
    let model = std::env::var("TYPESAFE_MODEL").unwrap_or_else(|_| "jev-latest".into());
    (TypeSafe::from_env().unwrap(), model)
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

/// A 280×280 red PNG, about TypeSafe's smallest accepted area, which Jev reads
/// together with the input behind TypeSafe's opt-in header.
#[tokio::test]
#[ignore]
async fn image() {
    const RED: &str = "iVBORw0KGgoAAAANSUhEUgAAARgAAAEYCAIAAAAI7H7bAAACPElEQVR42u3TMQ0AAAjAsPk3DSI4eJpUwZI1BRxJAEYCI4GRACOBkcBIYCTASGAkMBIYCTASGAmMBEYCjARGAiOBkQAjgZHASGAkwEhgJDASYCQwEhgJjAQYCYwERgIjAUYCI4GRwEiAkcBIYCQwEmAkMBIYCYwEGAmMBEYCIwFGAiOBkQAjgZHASGAkwEhgJDASGAkwEhgJjARGAowERgIjgZEAI4GRwEhgJMBIYCQwEhhJAjASGAmMBBgJjARGAiMBRgIjgZHASICRwEhgJDASYCQwEhgJjAQYCYwERgIjAUYCI4GRACOBkcBIYCTASGAkMBIYCTASGAmMBEYCjARGAiOBkQAjgZHASGAkwEhgJDASGAkwEhgJjAQYCYwERgIjAUYCI4GRwEiAkcBIYCQwEmAkMBIYCYwEGAmMBEYCIwFGAiOBkcBIgJHASGAkwEhgJDASGAkwEhgJjARGAowERgIjgZEAI4GRwEhgJMBIYCQwEhgJMBIYCYwESABGAiOBkQAjgZHASGAkwEhgJDASGAkwEhgJjARGAowERgIjgZEAI4GRwEhgJMBIYCQwEmAkMBIYCYwEGAmMBEYCIwFGAiOBkcBIgJHASGAkMBJgJDASGAmMBBgJjARGAiMBRgIjgZEAI4GRwEhgJMBIYCQwEhgJMBIYCYwERgKMBEYCI4GRACOBkcBIYCTASGAkMBIgARgJjARGAowERgIjgZEAI4GRwEhgJMBIYCQwEhgJMBIYCYwERgKMBC8WR5UfoHxpqCMAAAAASUVORK5CYII=";
    let (provider, model) = provider();
    for input in [
        DecisionRequest::new(&model, "A product photo."),
        DecisionRequest::new(&model, serde_json::json!({"caption": "A product photo."})),
    ] {
        let request = input.image(Image::base64("image/png", RED)).question(
            "color",
            Question::choice("What is the main color of the image?")
                .options(["red", "green", "blue"]),
        );
        let response = provider.decide(&request).await.unwrap();
        assert_eq!(response.choice("color").unwrap().chosen, "red");
    }
}

/// The long-context header is accepted.
#[tokio::test]
#[ignore]
async fn long_context() {
    let (provider, model) = provider();
    let request = DecisionRequest::new(&model, "The invoice was paid twice.")
        .question("refund", Question::yes_no("Is a refund due?"))
        .with_option(DecisionRequestOptions::new().long_context(true));
    let response = provider.decide(&request).await.unwrap();
    assert!(response.warnings.is_empty(), "{:?}", response.warnings);
    assert!(response.yes_no("refund").is_some());
}

/// A locate question on Jev 1.14 (`jev-preview`; Jev 1.13 has none): a
/// 400×400 white PNG with a red square spanning 0.5 to 0.9 across and 0.1 to
/// 0.5 down.
#[tokio::test]
#[ignore]
async fn locate() {
    const SQUARE: &str = "iVBORw0KGgoAAAANSUhEUgAAAZAAAAGQCAIAAAAP3aGbAAAFMElEQVR42u3UCQ0AMAwDsfAn3dLoYysIpvVSAEvEEwCCBSBYgGABCBaAYAGCBSBYAIIFCBaAYAEIFiBYAIIFIFiAYAEIFoBgAYIFIFgAggUIFoBgAQgWIFgAggUgWIBgAQgWIFgAggUgWIBgAQgWgGABggUgWACCBQgWgGDx4GfFLk+wECwTLMFCsEywQLAES7AQLBMswUKwTLBAsARLsBAsEyzBQrBMsECwBEuwECwTLMFCsEywwEkLlmAhWCZYIFgmWAiWCZZgIVgmWCBYJlgIlgmWYCFYJlggWCZYCJYJlmAhWCZYIFgmWAiWCZZgIVgmWCBYJlgIlgmWYCFYJlggWIIlWAiWCZZgIVgmWCBYgiVYCJYJlmAhWCZYIFiCJVgIlgmWYCFYJlggWIIlWAiWCZZgIVgmWAiWqxYswUKwTLBAsEywECwTLMFCsEywQLBMsBAsEyzBQrBMsECwTLAQLBMswUKwTLBAsEywECwTLMFCsEywQLBMsBAsEyzBQrBMsECwBEuwECwTLMFCsEywQLAES7AQLBMswUKwTLBAsARLsBAsEyzBQrBMsECwBEuwECwTLMFCsEywECwTLMFCsEywQLBMsBAsEyzBQrBMsECwTLAQLBMswUKwTLBAsEywECwTLMFCsEywQLBMsBAsEyzBQrBMsECwTLAQLBMswUKwTLBAsARLsBAsEyzBQrBMsECwBEuwECwTLMFCsEywQLAES7AQLBMswUKwTLBAsARLsBAsEyzBQrBMsBAsEyzBQrBMsECwTLAQLBMswUKwTLBAsEywECwTLMFCsEywQLBMsBAsEyzBQrBMsECwTLAQLBMswUKwTLBAsEywECwTLMFCsEywQLAES7AQLBMswUKwTLBAsARLsBAsEyzBQrBMsECwBEuwECwTLMFCsEywQLAES7AQLBMsl4VgmWAhWCZYgoVgmWCBYJlgIVgmWIKFYJlggWCZYCFYJliChWCZYIFgmWAhWCZYgoVgmWCBYJlgIVgmWIKFYJlggWCZYCFYJliChWCZYIFgCZZgIVgmWIKFYJlggWAJlmAhWCZYgoVgmWCBYAmWYCFYJliChWCZYIFgCZZgIVgmWCBYJlgIlgmWYCFYJlggWCZYAIIFCBaAYAEIFiBYAIIFIFiAYAEIFoBgAYIFIFgAggUIFoBgAQgWIFgAggUgWIBgAQgWgGABggUgWIBgAQgWgGABggUgWACCBQgWgGABCBYgWACCBSBYgGABCBaAYAGCBSBYAIIFCBaAYAEIFiBYAIIFIFiAYAEIFoBgAYIFIFiAYAEIFoBgAYIFIFgAggUIFoBgAQgWIFgAggUgWIBgAQgWgGABggUgWACCBQgWgGABCBYgWACCBSBYgGABCBaAYAGCBSBYgGABCBaAYAGCBSBYAIIFCBaAYAEIFiBYAIIFIFiAYAEIFoBgAYIFIFgAggUIFoBgAQgWIFgAggUgWIBgAQgWgGABggUgWIBgAQgWgGABggUgWACCBQgWgGABCBYgWACCBSBYgGABCBaAYAGCBSBYAIIFCBaAYAEIFiBYAIIFIFiAYAEIFoBgAYIFIFiAYAEIFoBgAYIFIFgAggUIFoBgAQgWIFgAggUgWIBgAQgWgGABggUgWACCBQgWgGABCBYgWACCBSBYgGABCBaAYAGCBSBYgGABCBaAYAGCBSBYAIIFCBaAYAEIFiBYAIIFIFiAYAEIFoBgAYIFIFgAggUIFoBgAQgWIFgAggUgWIBgAQgWgGABggUgWIBgAUzXQgbWvd2Cdh0AAAAASUVORK5CYII=";
    let (provider, _) = provider();
    let request = DecisionRequest::new("jev-preview", "A drawing.")
        .image(Image::base64("image/png", SQUARE))
        .question("square", Question::locate("the red square"));
    let response = provider.decide(&request).await.unwrap();
    let square = response.locate("square").unwrap();
    let expected = [0.5, 0.1, 0.9, 0.5];
    let edges = [square.left, square.top, square.right, square.bottom];
    assert!(
        edges
            .iter()
            .zip(expected)
            .all(|(edge, expected)| (edge - expected).abs() < 0.1),
        "{square:?}"
    );
}
