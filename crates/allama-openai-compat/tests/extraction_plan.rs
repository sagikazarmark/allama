//! What an extraction plan reports through the OpenAI-compatible presets,
//! for the typical schemas in each mode, without sending anything.

use allama_http::ReplayTransport;
use allama_openai_compat::Provider;
use allama_openai_compat::providers::{DeepSeek, Groq, Ollama, Together, Vllm};

#[test]
fn plans_per_preset() {
    let transport = ReplayTransport::new([]);
    let build = |builder: allama_openai_compat::ProviderBuilder| {
        builder.http_transport(transport.clone()).build().unwrap()
    };
    let rows = |native: [&str; 3], forced: [&str; 3]| {
        let mut rows = Vec::new();
        for (mode, cells) in [("Native", native), ("ForcedTool", forced)] {
            for (name, cell) in ["Person", "Order", "Tagged"].into_iter().zip(cells) {
                rows.push(format!("{mode} {name}: {cell}"));
            }
        }
        rows
    };
    let (exact, approximate, not_enforced) = ("exact", "approximate", "not_enforced");
    for (name, provider, model, expected) in [
        (
            "generic",
            build(Provider::builder().base_url("http://upstream/v1")),
            "m",
            rows(
                [approximate, approximate, not_enforced],
                [approximate, approximate, not_enforced],
            ),
        ),
        // Groq documents strict mode for the endpoint
        // (https://console.groq.com/docs/structured-outputs); which models
        // take it is model data (ADR 0015), so the plan is the same for
        // every model.
        (
            "groq",
            build(Groq::builder("k").unwrap()),
            "openai/gpt-oss-120b",
            rows([exact, exact, not_enforced], [exact, exact, not_enforced]),
        ),
        (
            "groq, any model",
            build(Groq::builder("k").unwrap()),
            "llama-3.3-70b-versatile",
            rows([exact, exact, not_enforced], [exact, exact, not_enforced]),
        ),
        (
            "together",
            build(Together::builder("k").unwrap()),
            "m",
            rows([not_enforced; 3], [not_enforced; 3]),
        ),
        (
            "deepseek",
            build(DeepSeek::builder("k").unwrap()),
            "m",
            rows(["refused response_format"; 3], [not_enforced; 3]),
        ),
        (
            "ollama",
            build(Ollama::builder("http://localhost:11434").unwrap()),
            "m",
            rows([approximate; 3], ["refused tool_choice"; 3]),
        ),
        // A cloud tag through a local server is not refused: where the
        // model runs is model data (ADR 0015). Ollama Cloud's host is.
        (
            "ollama, a cloud model",
            build(Ollama::builder("http://localhost:11434").unwrap()),
            "gemma4:cloud",
            rows([approximate; 3], ["refused tool_choice"; 3]),
        ),
        (
            "ollama cloud",
            build(Ollama::builder("https://ollama.com").unwrap()),
            "gemma4:31b",
            rows(["refused response_format"; 3], ["refused tool_choice"; 3]),
        ),
        (
            "vllm",
            build(Vllm::builder("http://localhost:8000/v1", None).unwrap()),
            "m",
            rows([approximate; 3], [approximate; 3]),
        ),
    ] {
        assert_eq!(typical::plans(&provider, model), expected, "{name}");
    }
    assert!(transport.requests().is_empty());
}

// The fixture types are only planned for, never built.
#[allow(dead_code)]
mod typical {
    use std::collections::HashMap;

    use allama::chat::extract::{ChatExtractionMode, ChatExtractor};
    use allama::schemars::JsonSchema;
    use allama::{ChatProvider, ChatRequest, ValidationCoverage};
    use serde::Deserialize;

    /// Strings, one of them optional.
    #[derive(Deserialize, JsonSchema)]
    #[schemars(crate = "allama::schemars")]
    pub struct Person {
        name: String,
        nickname: Option<String>,
    }

    /// An unsigned integer (schemars writes `minimum: 0`).
    #[derive(Deserialize, JsonSchema)]
    #[schemars(crate = "allama::schemars")]
    pub struct Order {
        id: String,
        quantity: u32,
    }

    /// A map.
    #[derive(Deserialize, JsonSchema)]
    #[schemars(crate = "allama::schemars")]
    pub struct Tagged {
        title: String,
        labels: HashMap<String, String>,
    }

    /// The plan of each typical schema in each explicit mode, as
    /// `"<mode> <type>: <enforcement>"`, or `refused <field>` when the plan
    /// fails before sending. Every plan is a dry run (`Encoding`).
    pub fn plans<P: ChatProvider + ?Sized>(provider: &P, model: &str) -> Vec<String> {
        let mut rows = Vec::new();
        for mode in [ChatExtractionMode::Native, ChatExtractionMode::ForcedTool] {
            let request = ChatRequest::new(model).user("x").with_option(mode);
            let extractor = ChatExtractor::new();
            for (name, plan) in [
                ("Person", extractor.plan::<Person>(provider, &request)),
                ("Order", extractor.plan::<Order>(provider, &request)),
                ("Tagged", extractor.plan::<Tagged>(provider, &request)),
            ] {
                let cell = match plan {
                    Ok(plan) => {
                        assert_eq!(plan.coverage, ValidationCoverage::Encoding, "{name}");
                        plan.enforcement.to_string()
                    }
                    Err(err) => format!(
                        "refused {}",
                        err.unsupported_field()
                            .map_or("-".to_owned(), |f| f.to_string())
                    ),
                };
                rows.push(format!("{mode:?} {name}: {cell}"));
            }
        }
        rows
    }
}
