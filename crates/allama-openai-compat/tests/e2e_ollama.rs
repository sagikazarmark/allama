//! Live checks of the Chat Completions `Ollama` preset against a local Ollama.
//!
//! The provider comes from `Ollama::from_env()` (`OLLAMA_HOST`, default
//! `http://localhost:11434`). Run with a server and a small model available:
//!
//! ```text
//! ollama pull qwen3:1.7b
//! cargo test -p allama-openai-compat --test e2e_ollama -- --ignored
//! ```
//!
//! `OLLAMA_MODEL` (default `qwen3:1.7b`) overrides the model.

// Sends through the default `reqwest` transport.
#![cfg(feature = "reqwest")]

use allama::ReasoningEffort;
use allama_conformance::e2e;
use allama_openai_compat::Provider;
use allama_openai_compat::providers::Ollama;

fn provider() -> Provider {
    Ollama::from_env().expect("failed to build the Ollama provider")
}

fn model() -> String {
    std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3:1.7b".into())
}

#[tokio::test]
#[ignore]
async fn basic_chat() {
    e2e::basic_chat(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn streaming() {
    e2e::streaming(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn system_prompt() {
    e2e::system_prompt(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn multi_turn() {
    e2e::multi_turn(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn tool_round_trip_with_reasoning() {
    e2e::tool_round_trip(&provider(), &model(), Some(ReasoningEffort::Low.into())).await;
}

#[tokio::test]
#[ignore]
async fn streaming_tool_round_trip() {
    e2e::streaming_tool_round_trip(&provider(), &model(), None).await;
}

#[tokio::test]
#[ignore]
async fn structured_output() {
    e2e::structured_output(&provider(), &model()).await;
}
