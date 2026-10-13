//! Persist a conversation, reload it, and continue it.
//!
//! Requests and responses implement `Serialize`/`Deserialize`, and
//! [`History`] is a versioned envelope for a conversation's messages. The
//! system prompt is not stored: it is set on every request. Opaque provider
//! state (thinking signatures, thought signatures) is kept in each block's
//! provider data, so a reloaded conversation replays exactly what the
//! provider needs — including on reasoning models, which reject turns whose
//! signatures were lost.
//!
//! It reads the model address from `ALLAMA_MODEL`, or uses
//! [`DEFAULT_MODEL`], a local Ollama:
//!
//! ```text
//! cargo run -p allama-examples --example history_persistence -- "Summarize this failure"
//! ALLAMA_MODEL=gemini:gemini-3.5-flash cargo run -p allama-examples --example history_persistence -- "Summarize this failure"
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

use std::fs;
use std::process::ExitCode;

use allama::ProviderIdentity;
use allama::history::History;
use allama::prelude::*;

/// The model address used when `ALLAMA_MODEL` is unset: a local Ollama,
/// which needs no credentials (`ollama pull qwen3:1.7b`).
const DEFAULT_MODEL: &str = "ollama:qwen3:1.7b";

/// The system prompt, set on every request; the history holds messages only.
const SYSTEM_PROMPT: &str = "You are a concise assistant.";

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

async fn run() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Name one benefit of persisting conversations.".into());

    let address = std::env::var("ALLAMA_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
    let (provider, model) = allama_providers::chat::from_model(&address)?;
    eprintln!("using model address={address}");
    let path = std::env::temp_dir().join(format!(
        "allama-history-{}-{}.json",
        provider.provider_name(),
        std::process::id()
    ));

    // Turn 1.
    let request = ChatRequest::new(&model)
        .system(SYSTEM_PROMPT)
        .user(prompt)
        .max_output_tokens(400);
    let response = provider.chat(&request).await?;
    println!("turn 1: {}", response.text_or_empty());

    // Save the messages, including the assistant turn.
    let mut history = History::from_request(&request);
    history.messages.push(response.to_assistant_message());
    fs::write(&path, serde_json::to_string_pretty(&history)?)?;
    eprintln!("saved history to {}", path.display());

    // Later: reload and continue.
    let restored: History = serde_json::from_str(&fs::read_to_string(&path)?)?;
    assert_eq!(restored, history);
    let follow_up = restored
        .into_request(&model)
        .system(SYSTEM_PROMPT)
        .user("Now say it in five words.")
        .max_output_tokens(400);
    let response = provider.chat(&follow_up).await?;
    for warning in &response.warnings {
        eprintln!("warning: {warning}");
    }
    if let Some(usage) = &response.usage {
        eprintln!("usage={usage:?}");
    }
    println!("turn 2: {}", response.text_or_empty());

    // A full exchange can also be logged as JSON for debugging.
    eprintln!(
        "response json bytes: {}",
        serde_json::to_string(&response)?.len()
    );
    fs::remove_file(&path)?;
    Ok(())
}
