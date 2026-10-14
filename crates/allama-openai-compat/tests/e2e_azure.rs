//! Live checks of the Azure preset, on both of its chat APIs.
//!
//! Model names are deployment names, so there is no default model: each
//! test runs when `AZURE_OPENAI_API_KEY` and `AZURE_E2E_MODEL` (a
//! deployment) are set, and skips otherwise. The resource comes from
//! `AZURE_OPENAI_ENDPOINT` or `AZURE_OPENAI_RESOURCE`, as `from_env` reads it.
//!
//! ```text
//! AZURE_E2E_MODEL=gpt-5.6-luna cargo test -p allama-openai-compat --test e2e_azure -- --ignored --nocapture
//! AZURE_E2E_MODEL=gpt-6-luna cargo test -p allama-openai-compat --test e2e_azure responses -- --ignored --nocapture
//! ```
//!
//! `gpt-6-luna` runs the Responses tests only: on Chat Completions it
//! refuses function tools at its default reasoning effort ("Function tools
//! with reasoning_effort are not supported for gpt-6-luna in
//! /v1/chat/completions. To use function tools, use /v1/responses or set
//! reasoning_effort to 'none'.", HTTP 400, checked 2026-10-14).

// Sends through the default `reqwest` transport.
#![cfg(feature = "reqwest")]

use allama::{ReasoningConfig, ReasoningEffort, ReasoningOutput};
use allama_conformance::e2e;
use allama_openai_compat::providers::Azure;

/// The deployment to test, or `None` (with a note) when the key or the
/// deployment is not set.
fn deployment() -> Option<String> {
    // A workflow passes a missing secret as an empty variable.
    for var in ["AZURE_OPENAI_API_KEY", "AZURE_E2E_MODEL"] {
        if std::env::var(var).map_or(true, |value| value.trim().is_empty()) {
            eprintln!("skipped: {var} is not set");
            return None;
        }
    }
    std::env::var("AZURE_E2E_MODEL").ok()
}

#[tokio::test]
#[ignore]
async fn chat_completions() {
    let Some(model) = deployment() else { return };
    let provider = Azure::from_env().unwrap_or_else(|err| panic!("{err}"));
    e2e::basic_chat(&provider, &model).await;
    e2e::streaming(&provider, &model).await;
    e2e::tool_calling(&provider, &model).await;
}

#[tokio::test]
#[ignore]
async fn responses() {
    let Some(model) = deployment() else { return };
    let provider = Azure::responses_from_env().unwrap_or_else(|err| panic!("{err}"));
    e2e::basic_chat(&provider, &model).await;
    e2e::streaming(&provider, &model).await;
    e2e::tool_calling(&provider, &model).await;
    e2e::tool_round_trip(&provider, &model, None).await;
    e2e::structured_output(&provider, &model).await;
}

/// Encrypted reasoning, returned and replayed statelessly across a tool
/// call: what the Responses API offers over Chat Completions. Needs a
/// reasoning deployment.
#[tokio::test]
#[ignore]
async fn responses_reasoning_replay() {
    let Some(model) = deployment() else { return };
    let provider = Azure::responses_from_env().unwrap_or_else(|err| panic!("{err}"));
    let reasoning = || {
        ReasoningConfig::new()
            .effort(ReasoningEffort::Low)
            .output(ReasoningOutput::Summary)
    };
    e2e::tool_round_trip(&provider, &model, Some(reasoning())).await;
    e2e::streaming_tool_round_trip(&provider, &model, Some(reasoning())).await;
}
