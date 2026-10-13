//! Reasoning example: enable thinking/reasoning on the target provider
//! and print the reasoning text alongside the final answer.
//!
//! Reasoning support is model-dependent:
//!
//! - Anthropic: Claude 4.5+ models return (summarized) thinking blocks.
//! - Gemini: 2.5 and 3.x models return thought summaries.
//! - OpenAI: reasoning models (gpt-5 family, o-series) accept the effort.
//!   The Responses API (`openai:<model>`) returns reasoning summaries; Chat
//!   Completions (`openai-chat:<model>`) never returns reasoning text, and
//!   only `usage.reasoning_tokens` shows the work.
//!
//! Portable reasoning is an effort level plus output visibility; token
//! budgets are provider options (see `allama_anthropic::ChatRequestOptions`
//! and `allama_gemini::ChatRequestOptions`).
//!
//! It reads the model address from `ALLAMA_MODEL`, or uses
//! [`DEFAULT_MODEL`], a local Ollama with a thinking model:
//!
//! ```text
//! cargo run -p allama-examples --example provider_reasoning
//! ALLAMA_MODEL=anthropic:claude-haiku-4-5 cargo run -p allama-examples --example provider_reasoning
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
use allama::{ReasoningConfig, ReasoningOutput};

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
    let prompt = std::env::args().nth(1).unwrap_or_else(|| {
        "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost? Explain your reasoning.".into()
    });

    let address = std::env::var("ALLAMA_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
    let (provider, model) = allama_providers::chat::from_model(&address)?;
    eprintln!("using model address={address}");

    let request = ChatRequest::new(&model)
        .reasoning(
            ReasoningConfig::new()
                .effort(ReasoningEffort::High)
                .output(ReasoningOutput::Summary),
        )
        .user(prompt);

    let response = provider.chat(&request).await?;

    if let Some(finish_reason) = &response.finish_reason {
        eprintln!("finish_reason={finish_reason:?}");
    }
    if let Some(usage) = &response.usage {
        eprintln!("usage={usage:?}");
    }

    if let Some(reasoning) = response.reasoning_text() {
        println!("--- reasoning ---");
        println!("{reasoning}");
        println!("--- answer ---");
    } else {
        eprintln!("(no reasoning blocks surfaced; provider may not expose reasoning content)");
    }
    println!("{}", response.text_or_empty());
    Ok(())
}
