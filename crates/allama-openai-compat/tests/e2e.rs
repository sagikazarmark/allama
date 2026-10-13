// Sends through the default `reqwest` transport.
#![cfg(feature = "reqwest")]

use allama::{CapabilitySupport, ChatCapability};
use allama_conformance::e2e;
use allama_openai_compat::Provider;

fn make_provider() -> Provider {
    let base_url = std::env::var("OPENAI_COMPAT_BASE_URL")
        .expect("OPENAI_COMPAT_BASE_URL must be set to run e2e tests");
    let api_key = std::env::var("OPENAI_COMPAT_API_KEY")
        .expect("OPENAI_COMPAT_API_KEY must be set to run e2e tests");

    Provider::builder()
        .base_url(base_url)
        .bearer_token(api_key)
        .provider_name("openai-compat")
        // A generic endpoint answers `Unknown` for every capability. The CI
        // target (GitHub Models, gpt-4o-mini) supports these, so declare
        // them; extraction's `Auto` mode only uses capabilities known to be
        // supported.
        .chat_capability(
            ChatCapability::StructuredOutput,
            CapabilitySupport::Supported,
        )
        .chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported)
        .build()
        .expect("failed to create OpenAI-compatible provider")
}

fn model() -> String {
    std::env::var("OPENAI_COMPAT_MODEL").expect("OPENAI_COMPAT_MODEL must be set to run e2e tests")
}

fn embedding_model() -> String {
    std::env::var("OPENAI_COMPAT_EMBEDDING_MODEL")
        .expect("OPENAI_COMPAT_EMBEDDING_MODEL must be set to run e2e tests")
}

#[tokio::test]
#[ignore]
async fn basic_chat() {
    e2e::basic_chat(&make_provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn streaming() {
    e2e::streaming(&make_provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn system_prompt() {
    e2e::system_prompt(&make_provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn multi_turn() {
    e2e::multi_turn(&make_provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn tool_calling() {
    e2e::tool_calling(&make_provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn structured_output() {
    e2e::structured_output(&make_provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn extract() {
    e2e::extract(&make_provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn basic_embed() {
    e2e::basic_embed(&make_provider(), &embedding_model()).await;
}

#[tokio::test]
#[ignore]
async fn batch_embed() {
    e2e::batch_embed(&make_provider(), &embedding_model()).await;
}

#[tokio::test]
#[ignore]
async fn dimensions() {
    e2e::dimensions(&make_provider(), &embedding_model(), 32).await;
}

#[tokio::test]
#[ignore]
async fn streaming_multi_chunk() {
    e2e::streaming_multi_chunk(&make_provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn tool_round_trip() {
    e2e::tool_round_trip(&make_provider(), &model(), None).await;
}

#[tokio::test]
#[ignore]
async fn streaming_tool_round_trip() {
    e2e::streaming_tool_round_trip(&make_provider(), &model(), None).await;
}

#[tokio::test]
#[ignore]
async fn error_bad_key() {
    let base_url = std::env::var("OPENAI_COMPAT_BASE_URL").expect("OPENAI_COMPAT_BASE_URL");
    let provider = Provider::builder()
        .base_url(base_url)
        .bearer_token("invalid-key")
        .build()
        .unwrap();
    e2e::error_bad_key(&provider, &model()).await;
}
