//! Consume a stream event by event and collect the final response.

use allama::mock::MockStreamingProvider;
use allama::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider =
        MockStreamingProvider::build(|b| b.text("Deterministic streamed hello from allama."));
    let mut stream = provider
        .chat_stream(&ChatRequest::new("demo-model").user("Say hello"))
        .await?
        .with_collector();

    while let Some(item) = stream.next_event().await {
        let item = item?;
        if let StreamEvent::TextDelta { text, .. } = &item.event {
            print!("{text}");
        }
    }
    println!();

    let response = stream.finish()?;
    println!("final text: {}", response.text_or_empty());
    Ok(())
}
