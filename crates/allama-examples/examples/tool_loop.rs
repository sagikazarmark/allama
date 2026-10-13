//! A tool loop: call the model, run the tools it asks for, send the results
//! back, and repeat until it answers.
//!
//! allama has no agent loop; this file is all of one. It shows:
//!
//! - a step cap, [`MAX_STEPS`]: a model that keeps calling tools fails the
//!   turn instead of running up a bill;
//! - tools dispatched with a `match` on the same name constants the
//!   definitions use, so a renamed tool cannot fall out of sync;
//! - the finish reason read before the tool calls: a reply cut off at the
//!   output token limit (`Length`) may hold half a tool call, and a
//!   `ToolCalls` finish without function calls waits on a tool this loop
//!   does not offer. Neither is acted on;
//! - the loop's own error type, [`TurnError`]: a reply the loop cannot act
//!   on fails the turn as the application's error, kept apart from the
//!   provider's [`Error`];
//! - the assistant message pushed before its tool results, and every call
//!   answered: [`ChatRequest::push_tool_result`] with the output, and
//!   [`ChatRequest::push_tool_error`] for an unknown tool or arguments that
//!   do not parse, so the model can correct itself;
//! - `parallel_tool_calls(true)` with `tool_choice` left unset (the model
//!   decides). A forced `ToolChoice::Required` would force a tool call on
//!   every follow-up too, and the loop would never end;
//! - each warning printed once, with [`SeenWarnings`];
//! - after the loop, the answer as data ([`Report`]), extracted from the
//!   conversation with one more call. The extraction request ends with a
//!   user turn that asks for the data ([`report_request`]): one that ends
//!   on the assistant's answer is a prefill, which Anthropic refuses with
//!   structured output.
//!
//! The provider comes from [`allama_providers::chat::from_model_with`], so
//! the file uses only `allama` and `allama-providers` and copies into an
//! application as is:
//!
//! ```toml
//! allama = { version = "0.1", features = ["extract"] }
//! allama-providers = "0.1"
//! serde = { version = "1", features = ["derive"] }
//! tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
//! ```
//!
//! It reads the model address from `ALLAMA_MODEL`, or uses
//! [`DEFAULT_MODEL`], a local Ollama:
//!
//! ```text
//! cargo run -p allama-examples --example tool_loop
//! ALLAMA_MODEL=openai:gpt-5-mini cargo run -p allama-examples --example tool_loop -- "Weather and local time in Lisbon and Oslo?"
//! ```

use std::fmt;
use std::process::ExitCode;
use std::time::Duration;

use allama::SeenWarnings;
use allama::prelude::*;
use allama::schemars::{self, JsonSchema};
use allama::wrappers::RetryPolicy;
use allama_providers::ProviderSettings;
use serde::Deserialize;

/// The model address used when `ALLAMA_MODEL` is unset: a local Ollama,
/// which needs no credentials (`ollama pull qwen3:1.7b`).
const DEFAULT_MODEL: &str = "ollama:qwen3:1.7b";

/// The most model calls one turn may make.
const MAX_STEPS: usize = 6;

const GET_WEATHER: &str = "get_weather";
const GET_LOCAL_TIME: &str = "get_local_time";

/// Arguments of both tools; the schema is derived from the type.
#[derive(Debug, Deserialize, JsonSchema)]
struct CityArgs {
    /// City name, such as "Lisbon".
    city: String,
}

/// The answer as data.
#[derive(Debug, Deserialize, JsonSchema)]
struct Report {
    /// One entry per city the question names.
    cities: Vec<CityReport>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct CityReport {
    city: String,
    /// The forecast, such as "sunny, 24 °C"; empty when it was not asked.
    weather: String,
    /// The local time, such as "14:30"; empty when it was not asked.
    local_time: String,
}

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

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let question = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "What is the weather and the local time in Lisbon?".into());
    let address = std::env::var("ALLAMA_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
    // A person is waiting: give up on a long `Retry-After` at once.
    let settings = ProviderSettings::new().retry(Some(
        RetryPolicy::new().max_retry_after(Duration::from_secs(5)),
    ));
    let (provider, model) = allama_providers::chat::from_model_with(&address, &settings)?;
    eprintln!("using model address={address}");

    let mut request = ChatRequest::new(model)
        .system("Use the tools for weather and time questions. Answer in one sentence.")
        .user(question)
        .tools(tools())
        .parallel_tool_calls(true);
    let answer = run_turn(&provider, &mut request, &mut SeenWarnings::new()).await?;

    if !answer.finished_normally() {
        eprintln!("[finish] {:?}", answer.finish_reason);
    }
    println!("{}", answer.text_or_empty());
    eprintln!("{} messages in the conversation", request.messages.len());

    // One more call, for code that acts on the answer.
    let report = provider
        .extract::<Report>(&report_request(&request))
        .await?;
    for city in report.value.cities {
        println!("{}: {}; {}", city.city, city.weather, city.local_time);
    }
    Ok(())
}

fn tools() -> [ToolDefinition; 2] {
    [
        ToolDefinition::from_schema::<CityArgs>(GET_WEATHER)
            .description("Look up the current weather for a city."),
        ToolDefinition::from_schema::<CityArgs>(GET_LOCAL_TIME)
            .description("Look up the current local time in a city."),
    ]
}

/// Run the turn in `request` to its answer. Every reply and tool result is
/// appended to `request.messages`, the answer included, so the caller can
/// store the conversation.
async fn run_turn<P: ChatProvider>(
    provider: &P,
    request: &mut ChatRequest,
    seen: &mut SeenWarnings,
) -> Result<ChatResponse, TurnError> {
    for step in 1..=MAX_STEPS {
        let response = provider.chat(request).await?;
        for warning in &response.warnings {
            if seen.insert(&request.model, warning) {
                eprintln!("[warning] {warning}");
            }
        }
        match response.finish_reason {
            // Cut off: a tool call's arguments may be incomplete.
            Some(FinishReason::Length) => return Err(TurnError::OutputLimit { step }),
            // The conversation no longer fits: shorten it, not the output.
            Some(FinishReason::ContextWindow) => return Err(TurnError::ContextWindow { step }),
            // The model waits on a call that is not a function call.
            Some(FinishReason::ToolCalls) if !response.has_tool_calls() => {
                return Err(TurnError::NotAFunctionCall { step });
            }
            _ => {}
        }

        request.push_message(response.to_assistant_message());
        if !response.has_tool_calls() {
            return Ok(response);
        }
        for call in response.tool_calls() {
            eprintln!("[tool] {}({})", call.name, call.arguments);
            match run_tool(call) {
                Ok(output) => request.push_tool_result(call, output),
                Err(error) => request.push_tool_error(call, error),
            }
        }
    }
    Err(TurnError::StepCap)
}

/// Why a turn failed: the provider's error, or a reply the loop cannot act
/// on. `step` counts the model calls of the turn, from 1.
#[derive(Debug)]
enum TurnError {
    /// A provider call failed.
    Provider(Error),
    /// The reply reached the output token limit, so a tool call's arguments
    /// may be incomplete.
    OutputLimit { step: usize },
    /// The conversation filled the model's context window: shorten the
    /// conversation, not the output.
    ContextWindow { step: usize },
    /// The model waits on a tool call that is not a function call, which
    /// this loop does not run.
    NotAFunctionCall { step: usize },
    /// No answer after [`MAX_STEPS`] model calls.
    StepCap,
}

impl fmt::Display for TurnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Provider(error) => error.fmt(f),
            Self::OutputLimit { step } => {
                write!(f, "step {step}: the reply reached the output token limit")
            }
            Self::ContextWindow { step } => {
                write!(
                    f,
                    "step {step}: the conversation filled the model's context window"
                )
            }
            Self::NotAFunctionCall { step } => write!(
                f,
                "step {step}: the model waits on a tool call that is not a function call"
            ),
            Self::StepCap => write!(f, "no answer after {MAX_STEPS} model calls"),
        }
    }
}

impl std::error::Error for TurnError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Provider(error) => Some(error),
            _ => None,
        }
    }
}

impl From<Error> for TurnError {
    fn from(error: Error) -> Self {
        Self::Provider(error)
    }
}

/// The extraction request for the answer as data: the conversation without
/// its tools, then a user turn that asks for the data.
fn report_request(turn: &ChatRequest) -> ChatRequest {
    ChatRequest::new(&turn.model)
        .extend_messages(turn.messages.iter().cloned())
        .user("Give the weather and the local time of each city in your answer as data.")
}

/// Run one tool call. An error goes back to the model as a tool error.
fn run_tool(call: &ToolCall) -> Result<String, String> {
    match call.name.as_str() {
        GET_WEATHER => {
            let args: CityArgs = call.parse_arguments().map_err(|error| error.to_string())?;
            Ok(format!(
                "{{\"city\":{:?},\"forecast\":\"sunny\",\"high_c\":24}}",
                args.city
            ))
        }
        GET_LOCAL_TIME => {
            let args: CityArgs = call.parse_arguments().map_err(|error| error.to_string())?;
            Ok(format!("{{\"city\":{:?},\"time\":\"14:30\"}}", args.city))
        }
        other => Err(format!(
            "unknown tool `{other}`; the tools are `{GET_WEATHER}` and `{GET_LOCAL_TIME}`"
        )),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use allama::mock::{ChatResponseBuilder, FunctionProvider};
    use serde_json::json;

    use super::*;

    fn request() -> ChatRequest {
        ChatRequest::new("demo-model")
            .user("Weather and time in Lisbon?")
            .tools(tools())
            .parallel_tool_calls(true)
    }

    /// A model that calls tools until the conversation ends with a tool
    /// result, then answers.
    fn model(calls: ChatResponseBuilder) -> FunctionProvider {
        FunctionProvider::new(move |request: &ChatRequest| {
            assert_eq!(request.tool_choice, None, "never forced");
            Ok(match request.messages.last() {
                Some(Message::Tool(_)) => ChatResponseBuilder::new().text("Sunny, 14:30.").build(),
                _ => calls.clone().build(),
            })
        })
    }

    fn tool_results(request: &ChatRequest) -> Vec<(String, bool)> {
        let results = request.messages.iter().filter_map(|message| match message {
            Message::Tool(tool) => Some((tool.name.clone(), tool.is_error)),
            _ => None,
        });
        results.collect()
    }

    #[tokio::test]
    async fn runs_every_call_then_returns_the_answer() {
        let provider = model(
            ChatResponseBuilder::new()
                .tool_call("c1", GET_WEATHER, json!({ "city": "Lisbon" }))
                .tool_call("c2", GET_LOCAL_TIME, json!({ "city": "Lisbon" })),
        );
        let mut request = request();

        let answer = run_turn(&provider, &mut request, &mut SeenWarnings::new())
            .await
            .unwrap();

        assert_eq!(answer.text_or_empty(), "Sunny, 14:30.");
        // user, assistant with two calls, two results, the answer
        assert_eq!(request.messages.len(), 5);
        let expected = [(GET_WEATHER.into(), false), (GET_LOCAL_TIME.into(), false)];
        assert_eq!(tool_results(&request), expected);
    }

    #[tokio::test]
    async fn unknown_tools_and_bad_arguments_go_back_as_tool_errors() {
        let provider = model(
            ChatResponseBuilder::new()
                .tool_call("c1", "get_stock_price", json!({ "symbol": "ACME" }))
                .tool_call("c2", GET_WEATHER, json!({ "town": "Lisbon" })),
        );
        let mut request = request();

        run_turn(&provider, &mut request, &mut SeenWarnings::new())
            .await
            .unwrap();

        let expected = [("get_stock_price".into(), true), (GET_WEATHER.into(), true)];
        assert_eq!(tool_results(&request), expected);
    }

    #[tokio::test]
    async fn a_reply_it_cannot_act_on_fails_the_turn() {
        let cut_off = ChatResponseBuilder::new()
            .tool_call("c1", GET_WEATHER, json!({ "city": "Lis" }))
            .finish_reason(FinishReason::Length);
        let no_calls = ChatResponseBuilder::new().finish_reason(FinishReason::ToolCalls);
        let error = fails_the_turn(cut_off).await;
        assert!(
            matches!(error, TurnError::OutputLimit { step: 1 }),
            "{error:?}"
        );
        let error = fails_the_turn(no_calls).await;
        assert!(
            matches!(error, TurnError::NotAFunctionCall { step: 1 }),
            "{error:?}"
        );
    }

    /// The error of a turn whose first reply is `reply`, which keeps
    /// nothing of the reply.
    async fn fails_the_turn(reply: ChatResponseBuilder) -> TurnError {
        let mut request = request();
        let error = run_turn(&model(reply), &mut request, &mut SeenWarnings::new())
            .await
            .unwrap_err();
        assert_eq!(request.messages.len(), 1, "nothing of the reply is kept");
        error
    }

    #[tokio::test]
    async fn the_report_is_asked_for_in_a_user_turn() {
        let provider = model(ChatResponseBuilder::new().tool_call(
            "c1",
            GET_WEATHER,
            json!({ "city": "Lisbon" }),
        ));
        let mut request = request();
        run_turn(&provider, &mut request, &mut SeenWarnings::new())
            .await
            .unwrap();

        let report = report_request(&request);
        assert!(matches!(report.messages.last(), Some(Message::User(_))));
        assert_eq!(report.messages.len(), request.messages.len() + 1);
        assert!(report.tools.is_empty());

        let data = r#"{"cities":[{"city":"Lisbon","weather":"sunny","local_time":"14:30"}]}"#;
        let extracted = FunctionProvider::new(move |_: &ChatRequest| {
            Ok(ChatResponseBuilder::new().text(data).build())
        })
        .extract::<Report>(&report)
        .await
        .unwrap();
        assert_eq!(extracted.value.cities[0].city, "Lisbon");
    }

    #[tokio::test]
    async fn stops_at_the_step_cap() {
        let steps = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&steps);
        let provider = FunctionProvider::new(move |_: &ChatRequest| {
            let id = counter.fetch_add(1, Ordering::Relaxed);
            let reply = ChatResponseBuilder::new();
            Ok(reply
                .tool_call(format!("c{id}"), GET_WEATHER, json!({ "city": "Lisbon" }))
                .build())
        });

        let error = run_turn(&provider, &mut request(), &mut SeenWarnings::new())
            .await
            .unwrap_err();

        assert!(matches!(error, TurnError::StepCap), "{error:?}");
        assert_eq!(steps.load(Ordering::Relaxed), MAX_STEPS);
    }
}
