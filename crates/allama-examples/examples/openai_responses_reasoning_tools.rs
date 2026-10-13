//! OpenAI Responses API: a reasoning model calls a tool, and the next turn
//! replays its reasoning statelessly.
//!
//! `ResponsesProvider` always sends `store: false`. Reasoning comes back as
//! `Reasoning` blocks whose encrypted state lives in provider data under
//! `"openai"`; replaying the assistant turn sends it back, so the model keeps
//! its chain of thought across tool calls without server-side storage. The
//! conversation also survives a JSON round trip in between.
//!
//! Run:
//!     OPENAI_API_KEY=... cargo run -p allama-examples --example openai_responses_reasoning_tools

use std::process::ExitCode;

use allama::history::History;
use allama::prelude::*;
use allama::{ContentBlock, ReasoningConfig, ReasoningOutput};
use allama_openai::ResponsesProvider;
use serde_json::json;

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
    let provider = ResponsesProvider::from_env()?;
    let model = std::env::var("OPENAI_MODEL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "gpt-5-nano".into());

    let weather = ToolDefinition::new(
        "get_weather",
        json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": false}),
    )
    .description("Current weather for a city.")
    .strict(true);

    let mut request = ChatRequest::new(&model)
        .user("Should I bring an umbrella in Paris today? Use the tool.")
        .tools([weather])
        .reasoning(
            ReasoningConfig::new()
                .effort(ReasoningEffort::Low)
                .output(ReasoningOutput::Summary),
        );

    let first = provider.chat(&request).await?;
    for block in &first.content {
        if let ContentBlock::Reasoning(reasoning) = block {
            eprintln!(
                "reasoning: summary={:?} encrypted_state={}",
                reasoning.text,
                reasoning
                    .provider_data
                    .get_raw("openai", "encrypted_content")
                    .is_some()
            );
        }
    }
    request.push_message(first.to_assistant_message());
    for call in first.tool_calls() {
        eprintln!("tool call: {}({})", call.name, call.arguments);
        request.push_tool_result(call, "Light rain until 3pm, 14°C.");
    }

    // Persist and reload the conversation, then continue it.
    let saved = serde_json::to_string(&History::from_request(&request))?;
    let restored: History = serde_json::from_str(&saved)?;
    let mut follow_up = restored.into_request(&model);
    follow_up.tools = request.tools.clone();
    follow_up.reasoning = request.reasoning;

    let second = provider.chat(&follow_up).await?;
    for warning in &second.warnings {
        eprintln!("warning: {warning}");
    }
    if let Some(usage) = &second.usage {
        eprintln!(
            "usage: input={:?} output={:?} reasoning={:?}",
            usage.input_tokens, usage.output_tokens, usage.reasoning_tokens
        );
    }
    println!("{}", second.text_or_empty());
    Ok(())
}
