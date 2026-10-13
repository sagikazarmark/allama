//! Embeddings: one vector per input, with a preview of each.
//!
//! It reads the model address from `ALLAMA_EMBEDDING_MODEL`, or uses
//! [`DEFAULT_MODEL`], a local Ollama:
//!
//! ```text
//! cargo run -p allama-examples --example provider_embedding -- "Hello embeddings" "Second input"
//! ALLAMA_EMBEDDING_MODEL=openai:text-embedding-3-small cargo run -p allama-examples --example provider_embedding -- "Hello embeddings" "Second input"
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

/// The model address used when `ALLAMA_EMBEDDING_MODEL` is unset: a local
/// Ollama, which needs no credentials (`ollama pull nomic-embed-text`).
const DEFAULT_MODEL: &str = "ollama:nomic-embed-text";

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
    let args: Vec<String> = std::env::args().skip(1).collect();
    let inputs: Vec<String> = if args.is_empty() {
        vec![
            "Rust makes ownership explicit.".into(),
            "Embedding models turn text into vectors.".into(),
        ]
    } else {
        args
    };

    let address = std::env::var("ALLAMA_EMBEDDING_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
    let (provider, model) = allama_providers::embedding::from_model(&address)?;
    eprintln!("using model address={address}");

    let request = EmbeddingRequest::new(&model).inputs(inputs.iter().cloned());
    let response = provider.embed(&request).await?;

    for (index, (input, vector)) in inputs.iter().zip(response.embeddings.iter()).enumerate() {
        let preview: Vec<f32> = vector.iter().take(4).copied().collect();
        let suffix = if vector.len() > preview.len() {
            ", ..."
        } else {
            ""
        };
        println!(
            "[{index}] dims={:>4} input={:?} head={:.4?}{}",
            vector.len(),
            input,
            preview,
            suffix,
        );
    }

    if let Some(model) = &response.model {
        eprintln!("response model={model}");
    }
    if let Some(usage) = &response.usage {
        eprintln!("usage={usage:?}");
    }

    Ok(())
}
