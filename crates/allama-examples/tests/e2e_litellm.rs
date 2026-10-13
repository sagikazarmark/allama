//! Live checks of three wire protocols through a LiteLLM proxy: Chat
//! Completions, Responses, and Anthropic Messages.
//!
//! The proxy translates each protocol to whatever model it routes to, so one
//! local model exercises all three adapters without provider credentials.
//! The devenv setup runs LiteLLM in front of Ollama:
//!
//! ```text
//! ollama pull qwen3:1.7b
//! devenv up
//! cargo test -p allama-examples --test e2e_litellm -- --ignored
//! ```
//!
//! `LITELLM_PROXY_URL` (default `http://127.0.0.1:4000`), `LITELLM_MODEL`
//! (default `ollama/qwen3:1.7b`) and `LITELLM_API_KEY` (for a proxy that
//! checks keys) override the target. Model-specific features (Claude
//! thinking signatures, server tools) and embeddings are out of scope: they
//! depend on what the proxy routes to, not on the protocol.

use allama::{CapabilitySupport, ChatCapability};
use allama_conformance::e2e;

fn base_url() -> String {
    std::env::var("LITELLM_PROXY_URL").unwrap_or_else(|_| "http://127.0.0.1:4000".into())
}

fn api_key() -> String {
    std::env::var("LITELLM_API_KEY").unwrap_or_else(|_| "unused".into())
}

fn model() -> String {
    std::env::var("LITELLM_MODEL").unwrap_or_else(|_| "ollama/qwen3:1.7b".into())
}

mod chat_completions {
    use super::*;
    use allama_openai_compat::Provider;

    fn provider() -> Provider {
        Provider::builder()
            .base_url(format!("{}/v1", base_url()))
            .bearer_token(api_key())
            .provider_name("litellm")
            // The proxy answers `Unknown` for every capability; declare what
            // the default model supports so extraction's `Auto` mode uses it.
            .chat_capability(
                ChatCapability::StructuredOutput,
                CapabilitySupport::Supported,
            )
            .chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported)
            .build()
            .expect("failed to build the Chat Completions provider")
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
    async fn structured_output() {
        e2e::structured_output(&provider(), &model()).await;
    }

    #[tokio::test]
    #[ignore]
    async fn extract() {
        e2e::extract(&provider(), &model()).await;
    }

    #[tokio::test]
    #[ignore]
    async fn tool_round_trip() {
        e2e::tool_round_trip(&provider(), &model(), None).await;
    }

    #[tokio::test]
    #[ignore]
    async fn streaming_tool_round_trip() {
        e2e::streaming_tool_round_trip(&provider(), &model(), None).await;
    }
}

mod responses {
    use super::*;
    use allama_openai_compat::ResponsesProvider;

    fn provider() -> ResponsesProvider {
        ResponsesProvider::builder()
            .base_url(format!("{}/v1", base_url()))
            .bearer_token(api_key())
            .provider_name("litellm")
            // The proxy answers `Unknown` for every capability; declare what
            // the default model supports so extraction's `Auto` mode uses it.
            .chat_capability(
                ChatCapability::StructuredOutput,
                CapabilitySupport::Supported,
            )
            .chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported)
            .build()
            .expect("failed to build the Responses provider")
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
    async fn structured_output() {
        e2e::structured_output(&provider(), &model()).await;
    }

    #[tokio::test]
    #[ignore]
    async fn extract() {
        e2e::extract(&provider(), &model()).await;
    }

    #[tokio::test]
    #[ignore]
    async fn tool_round_trip() {
        e2e::tool_round_trip(&provider(), &model(), None).await;
    }

    #[tokio::test]
    #[ignore]
    async fn streaming_tool_round_trip() {
        e2e::streaming_tool_round_trip(&provider(), &model(), None).await;
    }
}

mod anthropic_messages {
    use super::*;
    use allama_anthropic::Provider;

    fn provider() -> Provider {
        Provider::builder()
            .api_key(api_key())
            .base_url(base_url())
            .build()
            .expect("failed to build the Anthropic provider")
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
    async fn structured_output() {
        e2e::structured_output(&provider(), &model()).await;
    }

    #[tokio::test]
    #[ignore]
    async fn extract() {
        e2e::extract(&provider(), &model()).await;
    }

    #[tokio::test]
    #[ignore]
    async fn tool_round_trip() {
        e2e::tool_round_trip(&provider(), &model(), None).await;
    }

    #[tokio::test]
    #[ignore]
    async fn streaming_tool_round_trip() {
        e2e::streaming_tool_round_trip(&provider(), &model(), None).await;
    }
}
