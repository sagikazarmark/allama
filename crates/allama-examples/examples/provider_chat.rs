//! One request, one response: the answer, its finish reason, usage and
//! warnings.
//!
//! It reads the model address from `ALLAMA_MODEL`, or uses
//! [`DEFAULT_MODEL`], a local Ollama:
//!
//! ```text
//! cargo run -p allama-examples --example provider_chat -- "Explain Rust ownership"
//! ALLAMA_MODEL=anthropic:claude-haiku-4-5 cargo run -p allama-examples --example provider_chat -- "Explain Rust ownership"
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

/// The model address used when `ALLAMA_MODEL` is unset: a local Ollama,
/// which needs no credentials (`ollama pull qwen3:1.7b`).
const DEFAULT_MODEL: &str = "ollama:qwen3:1.7b";

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
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Explain Rust ownership in one short paragraph.".into());

    let address = std::env::var("ALLAMA_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
    let (provider, model) = allama_providers::chat::from_model(&address)?;
    eprintln!("using model address={address}");

    let response = provider
        .chat(&ChatRequest::new(&model).user(prompt))
        .await?;

    if let Some(finish_reason) = &response.finish_reason {
        eprintln!("finish_reason={finish_reason:?}");
    }
    if let Some(usage) = &response.usage {
        eprintln!("usage={usage:?}");
    }
    // What the provider could not honor: a dropped hint, replay state it
    // cannot take back, a failover. The same warnings are `tracing` events
    // (targets `allama::warning::<kind>`) for an application that installs
    // a subscriber.
    for warning in &response.warnings {
        let code = warning.code.as_deref().unwrap_or("-");
        eprintln!("warning kind={} code={code}: {warning}", warning.kind);
    }

    println!("{}", response.text_or_empty());
    Ok(())
}
