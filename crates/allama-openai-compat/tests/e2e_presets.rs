//! Live checks of the hosted presets.
//!
//! Each preset runs only when its key is set (`GROQ_API_KEY`, ...,
//! `COPILOT_GITHUB_TOKEN` for GitHub Copilot), and
//! skips otherwise, so the weekly workflow covers whichever presets have
//! secrets. `<PRESET>_E2E_MODEL` overrides the default model, a cheap one
//! that may be retired: override it rather than editing the test.
//!
//! ```text
//! GROQ_API_KEY=... cargo test -p allama-openai-compat --test e2e_presets -- --ignored --nocapture
//! ```

// Sends through the default `reqwest` transport.
#![cfg(feature = "reqwest")]

use allama::{
    ChatProvider, ChatRequest, ChatResponse, ChatStreamExt, ContentBlock, ReasoningConfig,
    ReasoningEffort, Result, ToolDefinition,
};
use allama_conformance::e2e;
use allama_openai_compat::Provider;
use allama_openai_compat::providers::{
    DeepInfra, DeepSeek, Fireworks, GitHubCopilot, Groq, Mistral, OpenRouter, Together, XAi,
};
use serde_json::json;

/// The preset and model to test, or `None` (with a note) when its key is
/// not set.
fn preset(
    key_var: &str,
    model_var: &str,
    default_model: &str,
    from_env: fn() -> Result<Provider>,
) -> Option<(Provider, String)> {
    // A workflow passes a missing secret as an empty variable.
    if std::env::var(key_var).is_ok_and(|key| key.trim().is_empty())
        || std::env::var_os(key_var).is_none()
    {
        eprintln!("skipped: {key_var} is not set");
        return None;
    }
    let provider = from_env().unwrap_or_else(|err| panic!("{key_var}: {err}"));
    let model = std::env::var(model_var).unwrap_or_else(|_| default_model.to_owned());
    Some((provider, model))
}

/// Chat, streaming and a tool call: the paths whose wire details differ
/// most between OpenAI-compatible servers (finish reasons, stream usage,
/// tool-call indices).
async fn run(provider: &Provider, model: &str) {
    e2e::basic_chat(provider, model).await;
    e2e::streaming(provider, model).await;
    e2e::tool_calling(provider, model).await;
}

macro_rules! preset_test {
    ($name:ident, $preset:ident, $key:literal, $model_var:literal, $model:literal) => {
        #[tokio::test]
        #[ignore]
        async fn $name() {
            if let Some((provider, model)) = preset($key, $model_var, $model, $preset::from_env) {
                run(&provider, &model).await;
            }
        }
    };
}

preset_test!(
    groq,
    Groq,
    "GROQ_API_KEY",
    "GROQ_E2E_MODEL",
    "llama-3.1-8b-instant"
);
preset_test!(
    openrouter,
    OpenRouter,
    "OPENROUTER_API_KEY",
    "OPENROUTER_E2E_MODEL",
    "openai/gpt-4o-mini"
);
preset_test!(
    deepseek,
    DeepSeek,
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_E2E_MODEL",
    "deepseek-chat"
);
preset_test!(
    mistral,
    Mistral,
    "MISTRAL_API_KEY",
    "MISTRAL_E2E_MODEL",
    "mistral-small-latest"
);
preset_test!(
    together,
    Together,
    "TOGETHER_API_KEY",
    "TOGETHER_E2E_MODEL",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo"
);
preset_test!(
    fireworks,
    Fireworks,
    "FIREWORKS_API_KEY",
    "FIREWORKS_E2E_MODEL",
    "accounts/fireworks/models/llama-v3p1-8b-instruct"
);
preset_test!(
    deepinfra,
    DeepInfra,
    "DEEPINFRA_API_KEY",
    "DEEPINFRA_E2E_MODEL",
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
);
preset_test!(xai, XAi, "XAI_API_KEY", "XAI_E2E_MODEL", "grok-3-mini");
preset_test!(
    github_copilot,
    GitHubCopilot,
    "COPILOT_GITHUB_TOKEN",
    "GITHUB_COPILOT_E2E_MODEL",
    "claude-haiku-4.5"
);

/// GitHub Copilot's reasoning state (`reasoning_text` and its
/// `reasoning_opaque` signature) goes back with a tool call, unary and
/// streamed. `GITHUB_COPILOT_E2E_REASONING_MODEL` overrides the model, one
/// the token's plan serves on Chat Completions that returns signed
/// reasoning there (Claude and Gemini ids do; GPT ids reason without
/// returning it, so they would pass without replaying anything, and fail
/// here instead).
#[tokio::test]
#[ignore]
async fn github_copilot_reasoning_round_trip() {
    let Some((provider, model)) = preset(
        "COPILOT_GITHUB_TOKEN",
        "GITHUB_COPILOT_E2E_REASONING_MODEL",
        "claude-sonnet-5.5",
        GitHubCopilot::from_env,
    ) else {
        return;
    };
    signed_reasoning_round_trip(&provider, &model, false).await;
    signed_reasoning_round_trip(&provider, &model, true).await;
}

/// A tool call whose turn carries signed reasoning, replayed with the tool
/// result: the endpoint must take the signature back and answer.
async fn signed_reasoning_round_trip(provider: &Provider, model: &str, stream: bool) {
    let mode = if stream { "streamed" } else { "unary" };
    let send = async |request: &ChatRequest| -> Result<ChatResponse> {
        if stream {
            provider
                .chat_stream(request)
                .await?
                .collect_response()
                .await
        } else {
            provider.chat(request).await
        }
    };
    let tool = ToolDefinition::new(
        "get_weather",
        json!({
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
            "additionalProperties": false
        }),
    )
    .description("Get the current weather for a city.");
    // Claude on Copilot reasons adaptively: the effort is a ceiling, and the
    // model decides per request. Live on 2026-10-14, with this prompt and
    // the tool, `claude-sonnet-5` and `claude-sonnet-5.5` returned signed
    // reasoning on 4 of 4 requests at `high` and 2 of 4 at `medium`, and an
    // easier prompt got none at `high`. Whether the model reasons is not
    // under test, so a first turn without signed reasoning and a tool call
    // is retried a few times.
    let mut request = ChatRequest::new(model)
        .tools([tool])
        .reasoning(ReasoningConfig::new().effort(ReasoningEffort::High))
        .user(
            "Factor 3599 into two primes. Then use the tool to get the weather in \
             Paris, and answer with both in one sentence.",
        );

    const ATTEMPTS: usize = 3;
    let mut first = None;
    for attempt in 1..=ATTEMPTS {
        let response = send(&request)
            .await
            .unwrap_or_else(|err| panic!("{mode}: first turn failed: {err}"));
        let signed = response.content.iter().any(|block| {
            matches!(block, ContentBlock::Reasoning(_))
                && block
                    .provider_data()
                    .get_raw(GitHubCopilot::NAMESPACE, "reasoning_opaque")
                    .is_some()
        });
        if signed && response.tool_calls().next().is_some() {
            first = Some(response);
            break;
        }
        // Reasoning tokens without a reasoning block would mean the
        // decoder missed it (unmodeled fields are in the provider data);
        // none mean the model did not reason.
        eprintln!(
            "{mode}: attempt {attempt}/{ATTEMPTS}: `{model}` returned no signed reasoning \
             with a tool call\n  content: {:?}\n  usage: {:?}\n  provider data: {:?}",
            response.content, response.usage, response.provider_data
        );
    }
    let first = first.unwrap_or_else(|| {
        panic!(
            "{mode}: `{model}` returned no signed reasoning with a tool call in \
             {ATTEMPTS} attempts, so nothing would be replayed"
        )
    });
    let calls: Vec<_> = first.tool_calls().cloned().collect();

    request.push_message(first.to_assistant_message());
    for call in &calls {
        request.push_tool_result(call, "Sunny, 18 degrees Celsius.");
    }
    let second = send(&request)
        .await
        .unwrap_or_else(|err| panic!("{mode}: replay turn failed: {err}"));
    assert!(
        second.text().is_some_and(|text| !text.is_empty()),
        "{mode}: expected a final text answer"
    );
    assert!(
        second.warnings.is_empty(),
        "{mode}: the replay dropped state: {:?}",
        second.warnings
    );
}
