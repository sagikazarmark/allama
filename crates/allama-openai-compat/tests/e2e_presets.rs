//! Live checks of the hosted presets.
//!
//! Each preset runs only when its key is set (`GROQ_API_KEY`, ...), and
//! skips otherwise, so the weekly workflow covers whichever presets have
//! secrets. `<PRESET>_E2E_MODEL` overrides the default model, a cheap one
//! that may be retired: override it rather than editing the test.
//!
//! ```text
//! GROQ_API_KEY=... cargo test -p allama-openai-compat --test e2e_presets -- --ignored --nocapture
//! ```

// Sends through the default `reqwest` transport.
#![cfg(feature = "reqwest")]

use allama::Result;
use allama_conformance::e2e;
use allama_openai_compat::Provider;
use allama_openai_compat::providers::{
    DeepInfra, DeepSeek, Fireworks, Groq, Mistral, OpenRouter, Together, XAi,
};

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
