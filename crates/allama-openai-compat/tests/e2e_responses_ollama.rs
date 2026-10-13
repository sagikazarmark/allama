//! Live checks of the generic Responses provider against a local Ollama.
//!
//! Run with a server and a small thinking model available:
//!
//! ```text
//! ollama pull qwen3:1.7b
//! cargo test -p allama-openai-compat --test e2e_responses_ollama -- --ignored
//! ```
//!
//! `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`) and
//! `OLLAMA_MODEL` (default `qwen3:1.7b`) override the target. Smaller models
//! (`qwen3:0.6b`) call tools only about half the time, so the tool scenarios
//! flake on them.

// Sends through the default `reqwest` transport.
#![cfg(feature = "reqwest")]

use allama::{
    ChatProvider, ChatRequest, ContentBlock, Error, ReasoningEffort, ToolChoice, ToolDefinition,
};
use allama_conformance::e2e;
use allama_openai_compat::{ResponsesProvider, ResponsesQuirks};
use serde_json::json;

const NAMESPACE: &str = "openai-compat:ollama";

fn provider() -> ResponsesProvider {
    let base_url =
        std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    ResponsesProvider::builder()
        .base_url(base_url)
        .provider_name("ollama")
        .quirks(quirks())
        .build()
        .expect("failed to build the Ollama Responses provider")
}

/// Ollama's dialect, as verified live: it echoes `tool_choice` as `"auto"`
/// and never forces a call.
fn quirks() -> ResponsesQuirks {
    let mut quirks = ResponsesQuirks::new(NAMESPACE);
    quirks.supports_tool_choice = false;
    quirks
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
async fn tool_round_trip_with_reasoning() {
    e2e::tool_round_trip(&provider(), &model(), Some(ReasoningEffort::Low.into())).await;
}

#[tokio::test]
#[ignore]
async fn streaming_tool_round_trip() {
    e2e::streaming_tool_round_trip(&provider(), &model(), Some(ReasoningEffort::Low.into())).await;
}

/// Ollama returns its thinking as summary text (and repeats it as
/// `encrypted_content`); both must be stored under the endpoint's own
/// namespace and replayed on the next turn without warnings.
#[tokio::test]
#[ignore]
async fn reasoning_is_namespaced_and_replayed() {
    let provider = provider();
    let weather = ToolDefinition::new(
        "get_weather",
        json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
    )
    .description("Current weather for a city.");
    let mut request = ChatRequest::new(model())
        .user("What is the weather in Paris? Use the tool.")
        .tools([weather])
        .reasoning(ReasoningEffort::Low);

    let first = provider.chat(&request).await.unwrap();
    assert_eq!(first.provider.as_deref(), Some("ollama"));
    let reasoning = first
        .content
        .iter()
        .find_map(|block| match block {
            ContentBlock::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .expect("expected a reasoning block");
    assert!(!reasoning.text.is_empty());
    assert!(reasoning.provider_data.namespace(NAMESPACE).is_some());
    assert!(reasoning.provider_data.namespace("openai").is_none());

    let call = first
        .tool_calls()
        .next()
        .expect("expected a tool call")
        .clone();
    request.push_message(first.to_assistant_message());
    request.push_tool_result(&call, "18°C, clear");
    let second = provider.chat(&request).await.unwrap();
    assert!(second.warnings.is_empty(), "{:?}", second.warnings);
    assert!(
        second.text().is_some_and(|t| t.contains("18")),
        "{:?}",
        second.text()
    );
}

/// Forced tool use is rejected before sending instead of silently becoming
/// `auto`.
#[tokio::test]
#[ignore]
async fn forced_tool_choice_is_rejected() {
    let request = ChatRequest::new(model())
        .user("Tell me a joke.")
        .tools([ToolDefinition::new(
            "get_weather",
            json!({"type": "object"}),
        )])
        .tool_choice(ToolChoice::Required);
    let err = provider().chat(&request).await.unwrap_err();
    assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
}
