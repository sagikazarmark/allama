//! Extract a typed value with native structured output.

use allama::ChatCapability;
use allama::chat::extract::ChatExtracted;
use allama::mock::MockProvider;
use allama::prelude::*;
use allama::schemars::{self, JsonSchema};
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct Review {
    title: String,
    rating: u8,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = MockProvider::with_text(r#"{"title":"Dune","rating":9}"#)
        .with_supported_chat_capabilities([ChatCapability::StructuredOutput]);
    let request = ChatRequest::new("demo-model")
        .system("Extract a review as structured data.")
        .user("Dune is atmospheric and deserves a 9 out of 10.");

    let extracted: ChatExtracted<Review> = provider.extract(&request).await?;
    println!("title: {}", extracted.value.title);
    println!("rating: {}", extracted.value.rating);
    println!(
        "mode: {:?}, passes: {}, usage: {:?}",
        extracted.metadata.mode, extracted.metadata.passes, extracted.metadata.usage
    );
    // The adapter's report: `Exact`, `Approximate` (constrained over a
    // widened schema) or `NotEnforced` (strict mode was not honoured). Set
    // `UnsupportedPolicy::Error` on the request to fail instead of running
    // unconstrained.
    println!(
        "enforcement: {:?}, degradations: {:?}",
        extracted.metadata.enforcement, extracted.metadata.degradations
    );
    Ok(())
}
