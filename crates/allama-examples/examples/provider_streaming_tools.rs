//! Streaming + tool calls example: iterate [`StreamEvent`]s, watch
//! tool-call blocks arrive mid-stream and act on each call as soon as it is
//! complete, then read the full [`ChatResponse`] the stream collected
//! ([`ChatStreamExt::with_collector`]).
//!
//! Works against any configured provider. Tool-call arguments arrive as
//! incremental `ToolCallDelta` events whose `arguments` field carries a
//! JSON string that accumulates across deltas.
//!
//! It reads the model address from `ALLAMA_MODEL`, or uses
//! [`DEFAULT_MODEL`], a local Ollama:
//!
//! ```text
//! cargo run -p allama-examples --example provider_streaming_tools -- Lisbon
//! ALLAMA_MODEL=openai:gpt-5-mini cargo run -p allama-examples --example provider_streaming_tools -- Lisbon
//! ```
//!
//! It uses only public API and copies into an application with:
//!
//! ```toml
//! allama = "0.1"
//! allama-providers = "0.1"
//! serde_json = "1"
//! tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
//! ```

use std::process::ExitCode;

use allama::prelude::*;
use serde_json::json;

/// The model address used when `ALLAMA_MODEL` is unset: a local Ollama,
/// which needs no credentials (`ollama pull qwen3:1.7b`).
const DEFAULT_MODEL: &str = "ollama:qwen3:1.7b";

fn weather_tool() -> ToolDefinition {
    ToolDefinition::new(
        "lookup_weather",
        json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
            "additionalProperties": false
        }),
    )
    .description("Look up the current weather for a city.")
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
    let city = std::env::args().nth(1).unwrap_or_else(|| "Berlin".into());

    let address = std::env::var("ALLAMA_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
    let (provider, model) = allama_providers::chat::from_model(&address)?;
    eprintln!("using model address={address}");

    let request = ChatRequest::new(&model)
        .system("Use the weather tool when asked about a city's weather.")
        .user(format!("What is the weather in {city}?"))
        .tools(vec![weather_tool()])
        .tool_choice(ToolChoice::Required);

    let mut stream = provider.chat_stream(&request).await?.with_collector();

    while let Some(item) = stream.next_event().await {
        let item = item?;
        // The `BlockStop` of a tool call comes with the parsed call.
        if let Some(ContentBlock::ToolCall(call)) = &item.completed {
            eprintln!("\ntool call complete: name={} id={}", call.name, call.id);
            continue;
        }
        match item.event {
            StreamEvent::BlockStart {
                block_type: StreamBlockType::ToolCall,
                id,
                name,
                ..
            } => {
                eprintln!(
                    "tool block start: name={} id={}",
                    name.as_deref().unwrap_or("?"),
                    id.as_deref().unwrap_or("?"),
                );
            }
            StreamEvent::ToolCallDelta { arguments, .. } => {
                eprint!("{arguments}");
            }
            StreamEvent::BlockStop { .. } => {
                eprintln!();
            }
            StreamEvent::TextDelta { text, .. } => {
                print!("{text}");
            }
            _ => {}
        }
    }
    println!();

    let response = stream.finish()?;
    eprintln!("finish_reason={:?}", response.finish_reason);

    for call in response.tool_calls() {
        eprintln!(
            "collected tool call: name={} id={} args={}",
            call.name, call.id, call.arguments
        );
    }

    Ok(())
}
