//! The smallest chat round trip, against the built-in mock provider.

use allama::mock::MockProvider;
use allama::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = MockProvider::build(|b| b.text("Deterministic hello from allama."));
    let response = provider
        .chat(&ChatRequest::new("demo-model").user("Say hello"))
        .await?;

    println!("chat text: {}", response.text_or_empty());
    println!("chat calls recorded: {}", provider.call_count());
    Ok(())
}
