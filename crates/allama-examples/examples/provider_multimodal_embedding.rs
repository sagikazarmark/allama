//! Cross-modal search: embed images and text queries with one model, then
//! rank the images for each query. It needs a model that embeds images.
//!
//! It reads the model address from `ALLAMA_EMBEDDING_MODEL`, or uses
//! [`DEFAULT_MODEL`], a Gemini model (`GEMINI_API_KEY`).
//!
//! ```text
//! cargo run -p allama-examples --example provider_multimodal_embedding -- "something red"
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
use allama::{CapabilitySupport, EmbeddingCapability, EmbeddingInput, EmbeddingPart};

/// The model address used when `ALLAMA_EMBEDDING_MODEL` is unset. Unlike
/// the other examples' defaults, it is not a local Ollama: Ollama embeds
/// text only.
const DEFAULT_MODEL: &str = "gemini:gemini-embedding-2";

/// 16x16 solid-color PNGs, so the example needs no files.
const RED_SQUARE: &str = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAFklEQVR42mO4IyJCEmIY1TCqYfhqAAACcQQQFd0BdQAAAABJRU5ErkJggg==";
const BLUE_SQUARE: &str = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAFklEQVR42mMQ0bhDEmIY1TCqYfhqAAAUJBgQCtUO5gAAAABJRU5ErkJggg==";

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
    let queries: Vec<String> = std::env::args().skip(1).collect();
    let queries = if queries.is_empty() {
        vec!["something red".to_owned(), "something blue".to_owned()]
    } else {
        queries
    };

    let address = std::env::var("ALLAMA_EMBEDDING_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
    let (provider, model) = allama_providers::embedding::from_model(&address)?;
    eprintln!("using model address={address}");
    let provider = &provider;
    let model = &model;

    if provider.embedding_capability(model, EmbeddingCapability::ImageInput)
        == CapabilitySupport::Unsupported
    {
        eprintln!("`{model}` does not embed images; try gemini:gemini-embedding-2");
        return Ok(());
    }

    // One input per image. The second document combines a caption and an
    // image into ONE vector, which needs `CompositeInput`.
    let mut documents = vec![
        (
            "red square",
            EmbeddingInput::image_base64("image/png", RED_SQUARE),
        ),
        (
            "blue square",
            EmbeddingInput::image_base64("image/png", BLUE_SQUARE),
        ),
    ];
    if provider.embedding_capability(model, EmbeddingCapability::CompositeInput)
        == CapabilitySupport::Supported
    {
        documents.push((
            "captioned blue square",
            EmbeddingInput::from(vec![
                EmbeddingPart::text("Product photo: a blue tile"),
                EmbeddingPart::image_base64("image/png", BLUE_SQUARE),
            ]),
        ));
    }

    let request = EmbeddingRequest::new(model.as_str())
        .inputs(documents.iter().map(|(_, input)| input.clone()));
    let response = provider.embed(&request).await?;

    let queries_response = provider
        .embed(&EmbeddingRequest::new(model.as_str()).inputs(&queries))
        .await?;

    for (query, query_vector) in queries.iter().zip(&queries_response.embeddings) {
        let mut ranked: Vec<(&str, f32)> = documents
            .iter()
            .zip(&response.embeddings)
            .map(|((name, _), vector)| (*name, cosine(query_vector, vector)))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!("{query:?}");
        for (name, score) in ranked {
            println!("  {score:.4}  {name}");
        }
    }

    if let Some(usage) = &response.usage {
        eprintln!("usage={usage:?}");
    }
    Ok(())
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}
