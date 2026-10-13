//! Vision: send an image with the question, and ask about it.
//!
//! Without an argument it sends a small PNG inline
//! (`ContentPart::image_base64`), which every provider takes; with one, it
//! sends that image URL (`ContentPart::image_url`). Ollama takes images
//! inline only, and Gemini needs a URL with a known image extension.
//!
//! It reads the model address from `ALLAMA_MODEL`, or uses
//! [`DEFAULT_MODEL`], a local Ollama with a model that takes images:
//!
//! ```text
//! cargo run -p allama-examples --example provider_vision
//! ALLAMA_MODEL=gemini:gemini-3.5-flash cargo run -p allama-examples --example provider_vision -- https://example.com/photo.png
//! ```
//!
//! It uses only public API and copies into an application with:
//!
//! ```toml
//! allama = "0.1"
//! allama-providers = "0.1"
//! tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
//! ```

use std::process::ExitCode;

use allama::prelude::*;

/// The model address used when `ALLAMA_MODEL` is unset: a local Ollama
/// model that takes images, which needs no credentials
/// (`ollama pull gemma3:4b`).
const DEFAULT_MODEL: &str = "ollama:gemma3:4b";

/// A 16x16 solid red PNG, so the example needs no file or URL.
const RED_SQUARE: &str = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAFklEQVR42mO4IyJCEmIY1TCqYfhqAAACcQQQFd0BdQAAAABJRU5ErkJggg==";

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
    let image = match std::env::args().nth(1) {
        Some(url) => ContentPart::image_url(url),
        None => ContentPart::image_base64("image/png", RED_SQUARE),
    };

    let address = std::env::var("ALLAMA_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
    let (provider, model) = allama_providers::chat::from_model(&address)?;
    eprintln!("using model address={address}");

    let request = ChatRequest::new(&model).user(vec![
        ContentPart::text("Describe what you see in this image in one sentence."),
        image,
    ]);

    let response = provider.chat(&request).await?;

    if let Some(finish_reason) = &response.finish_reason {
        eprintln!("finish_reason={finish_reason:?}");
    }
    if let Some(usage) = &response.usage {
        eprintln!("usage={usage:?}");
    }

    println!("{}", response.text_or_empty());
    Ok(())
}
