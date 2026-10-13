//! Streaming: print the normalized stream events as they arrive, then
//! collect the response.
//!
//! It reads the model address from `ALLAMA_MODEL`, or uses
//! [`DEFAULT_MODEL`], a local Ollama:
//!
//! ```text
//! cargo run -p allama-examples --example provider_stream -- "Explain Rust ownership"
//! ALLAMA_MODEL=gemini:gemini-3.5-flash cargo run -p allama-examples --example provider_stream -- "Explain Rust ownership"
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
        .unwrap_or_else(|| "Explain how streaming helps agent UIs.".into());

    let address = std::env::var("ALLAMA_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
    let (provider, model) = allama_providers::chat::from_model(&address)?;
    eprintln!("using model address={address}");

    let request = ChatRequest::new(&model).user(prompt);
    let mut stream = provider.chat_stream(&request).await?.with_collector();

    while let Some(item) = stream.next_event().await {
        let item = item?;

        match &item.event {
            StreamEvent::TextDelta { text, .. } => {
                print!("{text}");
            }
            StreamEvent::ReasoningDelta { text, .. } => {
                eprint!("[reasoning:{}]", text.replace('\n', "\\n"));
            }
            StreamEvent::BlockStart {
                block_type: StreamBlockType::ToolCall,
                index,
                name,
                ..
            } => {
                eprintln!(
                    "\n[tool_call_start index={index} name={}]",
                    name.as_deref().unwrap_or("?")
                );
            }
            StreamEvent::ResponseMetadata {
                finish_reason,
                usage,
                ..
            } => {
                eprintln!("\n[metadata finish_reason={finish_reason:?} usage={usage:?}]");
            }
            StreamEvent::Warning { warning, .. } => {
                eprintln!("\n[warning {warning}]");
            }
            StreamEvent::ResponseStop => {
                eprintln!("\n[response_stop]");
            }
            _ => {}
        }
    }

    let response = stream.finish()?;
    eprintln!("\n[final_text_len={}]", response.text_or_empty().len());
    Ok(())
}
