//! Structured extraction: a typed Rust value from any chat provider, with
//! how strictly the provider held the output to the schema.
//!
//! It reads the model address from `ALLAMA_MODEL`, or uses
//! [`DEFAULT_MODEL`], a local Ollama:
//!
//! ```text
//! cargo run -p allama-examples --example provider_extract -- "Summarize why Rust is good for agents"
//! ALLAMA_MODEL=anthropic:claude-haiku-4-5 cargo run -p allama-examples --example provider_extract -- "Summarize why Rust is good for agents"
//! ```
//!
//! It uses only public API and copies into an application with:
//!
//! ```toml
//! allama = { version = "0.1", features = ["extract"] }
//! allama-providers = "0.1"
//! serde = { version = "1", features = ["derive"] }
//! serde_json = "1"
//! tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
//! ```

use std::process::ExitCode;

use allama::extract::Extracted;
use allama::prelude::*;
use allama::schemars::{self, JsonSchema};
use serde::{Deserialize, Serialize};

/// The model address used when `ALLAMA_MODEL` is unset: a local Ollama,
/// which needs no credentials (`ollama pull qwen3:1.7b`).
const DEFAULT_MODEL: &str = "ollama:qwen3:1.7b";

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct RequestSummary {
    summary: String,
    mentions_rust: bool,
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
    let prompt = std::env::args().nth(1).unwrap_or_else(|| {
        "Rust ownership and borrowing help keep agent state updates safe.".into()
    });

    let address = std::env::var("ALLAMA_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
    let (provider, model) = allama_providers::chat::from_model(&address)?;
    eprintln!("using model address={address}");

    let request = ChatRequest::new(&model)
        .system("Extract a concise summary and whether the request mentions Rust.")
        .user(prompt);

    let extracted: Extracted<RequestSummary> = provider.extract(&request).await?;

    // How strictly the provider held the output to the schema: `Exact`
    // means constrained decoding; anything less was checked after the fact.
    eprintln!(
        "mode={:?} passes={} enforcement={:?}",
        extracted.metadata.mode, extracted.metadata.passes, extracted.metadata.enforcement
    );
    // Every warning of the call; those that set the enforcement (a schema
    // keyword removed, strict mode not honored) are also in
    // `metadata.degradations`.
    for warning in &extracted.response.warnings {
        let code = warning.code.as_deref().unwrap_or("-");
        eprintln!("warning kind={} code={code}: {warning}", warning.kind);
    }
    eprintln!("raw_text={}", extracted.response.text_or_empty());
    println!(
        "{}",
        serde_json::to_string_pretty(&extracted.value).map_err(Error::from)?
    );
    Ok(())
}
