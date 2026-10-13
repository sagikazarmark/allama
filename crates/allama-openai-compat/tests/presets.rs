//! One fixture per preset: the same portable request goes
//! through each preset's dialect, and the wire body plus the reported
//! warnings are compared with `fixtures/presets/<name>.json`. Embedding
//! requests with each input type are compared with
//! `fixtures/presets/embeddings/<name>.json`.
//!
//! Regenerate with `ALLAMA_UPDATE_FIXTURES=1` and review the diff.

// Sends through the default `reqwest` transport (a local test server).
#![cfg(feature = "reqwest")]

use std::path::PathBuf;

use allama::{
    CapabilitySupport, ChatProvider, ChatRequest, EmbeddingCapability, EmbeddingInputType,
    EmbeddingProvider, EmbeddingRequest, PromptCache, ReasoningEffort, SystemPrompt,
    ToolDefinition,
};
use allama_conformance::{FixtureDir, MockHttpResponse, TestHttpServer, assert_json_fixture_eq};
use allama_openai_compat::providers::{
    Azure, Cloudflare, DeepInfra, DeepSeek, Fireworks, Groq, Mistral, Ollama, OpenRouter, Together,
    Vllm, XAi,
};
use allama_openai_compat::{ProviderBuilder, SystemOptions};
use serde_json::json;

fn fixtures() -> FixtureDir {
    FixtureDir::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/presets"))
}

fn request() -> ChatRequest {
    ChatRequest::new("model-x")
        .system(SystemPrompt::new("Be brief.").with_option(SystemOptions::developer()))
        .user("Weather in Paris?")
        .max_output_tokens(256)
        .seed(7)
        .parallel_tool_calls(false)
        .reasoning(ReasoningEffort::Low)
        .cache(PromptCache::new().key("session-1"))
        .tools([ToolDefinition::new(
            "get_weather",
            json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
        )
        .strict(true)])
}

fn presets() -> Vec<(&'static str, ProviderBuilder)> {
    vec![
        ("azure", Azure::builder("res", "k").unwrap()),
        (
            "cloudflare",
            Cloudflare::builder("0123456789abcdef0123456789abcdef", "k").unwrap(),
        ),
        ("deepinfra", DeepInfra::builder("k").unwrap()),
        ("deepseek", DeepSeek::builder("k").unwrap()),
        ("fireworks", Fireworks::builder("k").unwrap()),
        ("groq", Groq::builder("k").unwrap()),
        ("mistral", Mistral::builder("k").unwrap()),
        ("ollama", Ollama::builder("http://localhost:11434").unwrap()),
        ("openrouter", OpenRouter::builder("k").unwrap()),
        ("together", Together::builder("k").unwrap()),
        (
            "vllm",
            Vllm::builder("http://localhost:8000/v1", None).unwrap(),
        ),
        ("xai", XAi::builder("k").unwrap()),
    ]
}

#[tokio::test]
async fn each_preset_matches_its_fixture() {
    for (name, builder) in presets() {
        let server = TestHttpServer::spawn([MockHttpResponse::json_value(
            200,
            &json!({
                "id": "c1", "model": "model-x",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
            }),
        )])
        .await;
        let provider = builder.base_url(server.url()).build().unwrap();
        let response = provider.chat(&request()).await.unwrap();
        assert_eq!(
            response.provider.as_deref(),
            Some(
                provider
                    .quirks()
                    .namespace
                    .trim_start_matches("openai-compat:")
            ),
            "{name}: provider stamp"
        );
        let body = server.recorded_requests().await[0].body_json();
        let warnings: Vec<_> = response
            .warnings
            .iter()
            .map(|w| json!({"kind": w.kind, "subject": w.subject}))
            .collect();
        assert_json_fixture_eq(
            &json!({"body": body, "warnings": warnings}),
            &fixtures(),
            &format!("{name}.json"),
        );
    }
}

/// The query/document input type goes on the preset's field (DeepInfra and
/// OpenRouter: `input_type`), or is dropped with a warning where the
/// endpoint documents none.
#[tokio::test]
async fn each_preset_maps_the_embedding_input_type() {
    let embeddings_ok = || {
        MockHttpResponse::json_value(
            200,
            &json!({"data": [{"embedding": [0.1], "index": 0}], "model": "model-x"}),
        )
    };
    for (name, builder) in presets() {
        let server = TestHttpServer::spawn([embeddings_ok(), embeddings_ok()]).await;
        let provider = builder.base_url(server.url()).build().unwrap();
        let mut cases = serde_json::Map::new();
        for input_type in [EmbeddingInputType::Query, EmbeddingInputType::Document] {
            let request = EmbeddingRequest::new("model-x")
                .input("Weather in Paris?")
                .input_type(input_type);
            let response = provider.embed(&request).await.unwrap();
            let warnings: Vec<_> = response
                .warnings
                .iter()
                .map(|w| json!({"kind": w.kind, "subject": w.subject}))
                .collect();
            cases.insert(input_type.to_string(), json!({"warnings": warnings}));
        }
        for (index, recorded) in server.recorded_requests().await.iter().enumerate() {
            let key = if index == 0 { "query" } else { "document" };
            cases[key]["body"] = recorded.body_json();
        }
        let maps = provider.quirks().embedding_input_type.is_some();
        assert_eq!(
            provider.embedding_capability("model-x", EmbeddingCapability::InputType),
            if maps {
                CapabilitySupport::Unknown
            } else {
                CapabilitySupport::Unsupported
            },
            "{name}: an endpoint without a field reports the input type unsupported"
        );
        assert_json_fixture_eq(
            &serde_json::Value::Object(cases),
            &fixtures(),
            &format!("embeddings/{name}.json"),
        );
    }
}
