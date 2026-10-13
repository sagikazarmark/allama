//! A chat in the terminal: read a line, stream the answer, repeat.
//!
//! The conversation lives in memory as a `Vec<Message>`, and each line
//! from stdin is one turn. It shows:
//!
//! - streaming with [`ChatStreamExt::with_collector`]: answer text to
//!   stdout; reasoning, refusals and tool progress to stderr;
//! - one local tool, `get_time`, whose schema is derived from a type
//!   ([`ToolDefinition::from_schema`]), and a streaming tool loop of at most
//!   [`MAX_STEPS`] model calls per turn;
//! - a turn stored whole or not at all ([`Chat::send`]): it goes into the
//!   history when it ends with an answer, and anything else rolls it back,
//!   the user's line included. [`storable`] decides which replies the
//!   history can take;
//! - token usage summed over every model call of a turn, and for the
//!   session, printed with `Usage`'s `Display`;
//! - optional settings from the environment assigned to the request's
//!   `Option` fields, with the reasoning effort parsed by `FromStr`
//!   (`ALLAMA_REASONING_EFFORT=low`);
//! - a provider that gives up at once on a long `Retry-After`
//!   (`max_retry_after`), because a person is waiting;
//! - a smoke call ([`smoke_call`]) at startup and on every switch: a
//!   rejected key, a model the provider does not have or a missing
//!   credential fails startup, or keeps the current model;
//! - warnings shown from the stream's warning events, each recurring one
//!   once, with [`SeenWarnings`]. No `tracing` subscriber is installed, so
//!   allama's warning events go nowhere and nothing is shown twice;
//! - `/model <provider:model>`, which switches the model, keeps the
//!   conversation, and clears the seen warnings.
//!
//! `/model` keeps the history as it is, with the previous provider's
//! replay state (signatures, encrypted reasoning): a switch in a REPL is
//! not a move for good, and switching back finds that state again. The new
//! provider leaves it out of its requests, with a `Dropped` warning shown
//! once. The applications guide's "Switching providers with stored
//! history" says when to remove it.
//!
//! `tool_loop` is the same loop without streaming, and extracts the answer
//! as data. This one builds its provider with
//! [`allama_providers::chat::from_model_with`] and uses only public API, so
//! it copies into an application as is:
//!
//! ```toml
//! allama = { version = "0.1", features = ["schema"] }
//! allama-providers = "0.1"
//! serde = { version = "1", features = ["derive"] }
//! tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
//! ```
//!
//! It reads the model address from `ALLAMA_MODEL`, or uses
//! [`DEFAULT_MODEL`], a local Ollama:
//!
//! ```text
//! cargo run -p allama-examples --example chat_repl
//! ALLAMA_MODEL=anthropic:claude-haiku-4-5 ALLAMA_REASONING_EFFORT=low cargo run -p allama-examples --example chat_repl
//! ```

use std::io::Write as _;
use std::process::ExitCode;
use std::str::FromStr;
use std::time::{Duration, SystemTime};

use allama::SeenWarnings;
use allama::prelude::*;
use allama::schemars::{self, JsonSchema};
use allama::wrappers::RetryPolicy;
use allama_providers::ProviderSettings;
use serde::Deserialize;

/// The model address used when `ALLAMA_MODEL` is unset: a local Ollama,
/// which needs no credentials (`ollama pull qwen3:1.7b`).
const DEFAULT_MODEL: &str = "ollama:qwen3:1.7b";

const SYSTEM_PROMPT: &str = "You are a concise assistant. Use get_time for the current time.";

/// The most model calls one turn may make.
const MAX_STEPS: usize = 4;

const GET_TIME: &str = "get_time";

/// One conversation with one model at a time.
struct Chat {
    provider: DynChatProvider,
    model: String,
    settings: Settings,
    history: Vec<Message>,
    seen: SeenWarnings,
    session: Usage,
}

#[tokio::main]
async fn main() -> ExitCode {
    match repl().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn repl() -> Result<(), Box<dyn std::error::Error>> {
    // `ReasoningEffort` parses the names it displays: `low`, `high`, ...
    let settings = Settings {
        effort: setting("ALLAMA_REASONING_EFFORT")?,
        max_output_tokens: setting("ALLAMA_MAX_OUTPUT_TOKENS")?,
    };
    let address = env("ALLAMA_MODEL").unwrap_or_else(|| DEFAULT_MODEL.into());
    let (provider, model) = connect(&address)?;
    smoke_call(&provider, &model).await?;
    let mut chat = Chat::new(provider, model, settings);
    eprintln!("chatting with {address}; `/model <provider:model>` switches, EOF ends");

    // A blocking read is fine here: nothing else runs while we wait.
    for line in std::io::stdin().lines() {
        let line = line?;
        let line = line.trim();
        match line.strip_prefix("/model") {
            Some(address) => {
                let switched = match connect(address.trim()) {
                    Ok((provider, model)) => chat.switch(provider, model).await,
                    Err(error) => Err(error),
                };
                if let Err(error) = switched {
                    eprintln!("[error] {error}; still talking to {}", chat.model);
                }
            }
            None if line.is_empty() => {}
            None => {
                if let Err(error) = chat.send(line).await {
                    eprintln!("[error] {error}; the turn was not stored");
                }
            }
        }
    }
    eprintln!("[session] {}", chat.session);
    Ok(())
}

impl Chat {
    fn new(provider: DynChatProvider, model: String, settings: Settings) -> Self {
        Self {
            provider,
            model,
            settings,
            history: Vec::new(),
            seen: SeenWarnings::new(),
            session: Usage::default(),
        }
    }

    /// Talk to another model from the next turn on, with the same history.
    /// A setup mistake the smoke call finds keeps the current model.
    async fn switch(&mut self, provider: DynChatProvider, model: String) -> Result<()> {
        smoke_call(&provider, &model).await?;
        self.provider = provider;
        self.model = model;
        self.seen.clear();
        eprintln!("[model] {}", self.model);
        Ok(())
    }

    /// Run one turn, and store it only when it ends with an answer.
    async fn send(&mut self, line: &str) -> Result<()> {
        let mut request = ChatRequest::new(&self.model)
            .system(SYSTEM_PROMPT)
            .extend_messages(self.history.iter().cloned())
            .user(line)
            .tools(tools());
        request.reasoning = self.settings.effort.map(Into::into);
        request.max_output_tokens = self.settings.max_output_tokens;

        let mut usage = Usage::default();
        let answered = self.answer(&mut request, &mut usage).await;
        // Every model call is billed, whether the turn is stored or not.
        eprintln!("[usage] {usage}");
        self.session += &usage;
        answered?;
        let turn = request.messages.split_off(self.history.len());
        self.history.extend(turn);
        Ok(())
    }

    /// The tool loop: stream a reply, run the tools it calls, and ask
    /// again, until the model answers. Each reply and tool result is
    /// appended to `request.messages`, and each reply's usage to `usage`.
    async fn answer(&mut self, request: &mut ChatRequest, usage: &mut Usage) -> Result<()> {
        for _ in 0..MAX_STEPS {
            let reply = self.stream(request).await?;
            if let Some(reply_usage) = &reply.usage {
                *usage += reply_usage;
            }
            storable(&reply)?;
            request.push_message(reply.to_assistant_message());
            if !reply.has_tool_calls() {
                return Ok(());
            }
            for call in reply.tool_calls() {
                eprintln!("[tool] {}({})", call.name, call.arguments);
                match run_tool(call) {
                    Ok(output) => request.push_tool_result(call, output),
                    Err(error) => request.push_tool_error(call, error),
                }
            }
        }
        // Stored, the turn would end on tool results with no answer.
        Err(
            Error::unexpected_response(format!("no answer after {MAX_STEPS} model calls"))
                .authored(),
        )
    }

    /// Stream one reply to the terminal and collect it.
    async fn stream(&mut self, request: &ChatRequest) -> Result<ChatResponse> {
        let mut stream = self.provider.chat_stream(request).await?.with_collector();
        let mut printed = false;
        while let Some(item) = stream.next_event().await {
            let event = match item {
                Ok(item) => item.event,
                Err(error) => {
                    // What arrived before the failure, for display only.
                    let partial = stream.finish_partial()?.response;
                    let shown = partial.text_or_empty().chars().count();
                    eprintln!("\n[interrupted] after {shown} characters");
                    return Err(error);
                }
            };
            match event {
                StreamEvent::TextDelta { text, .. } => {
                    printed = true;
                    print!("{text}");
                    let _ = std::io::stdout().flush();
                }
                StreamEvent::ReasoningDelta { text, .. }
                | StreamEvent::RefusalDelta { text, .. } => {
                    eprint!("{text}");
                }
                StreamEvent::BlockStart { block_type, .. } => match block_type {
                    StreamBlockType::Reasoning => eprint!("[reasoning] "),
                    StreamBlockType::Refusal => eprint!("[refusal] "),
                    _ => {}
                },
                StreamEvent::BlockStop { index, .. } => {
                    if let Some(StreamBlockType::Reasoning | StreamBlockType::Refusal) =
                        stream.collector().block_type(index)
                    {
                        eprintln!();
                    }
                }
                StreamEvent::Warning { warning, .. }
                    if self.seen.insert(&request.model, &warning) =>
                {
                    eprintln!("[warning] {warning}");
                }
                _ => {}
            }
        }
        if printed {
            println!();
        }
        stream.finish()
    }
}

/// Optional request settings; unset keeps the provider default.
#[derive(Debug, Default)]
struct Settings {
    effort: Option<ReasoningEffort>,
    max_output_tokens: Option<u32>,
}

/// Arguments of `get_time`; the schema is derived from the type.
#[derive(Debug, Deserialize, JsonSchema)]
struct TimeArgs {
    /// Offset from UTC in whole hours, such as 1 for Lisbon in summer.
    utc_offset_hours: i64,
}

/// Whether a reply can go into the history. An `Err` says why not, and
/// the whole turn is rolled back.
fn storable(reply: &ChatResponse) -> Result<()> {
    match reply.finish_reason {
        // Stored, it would trip the filter again on every turn.
        Some(FinishReason::ContentFilter) => {
            Err(Error::content_filtered("a content filter stopped the reply").authored())
        }
        // The arguments may be cut off too.
        Some(FinishReason::Length | FinishReason::ContextWindow) if reply.has_tool_calls() => {
            Err(Error::unexpected_response("a tool call was cut off").authored())
        }
        _ => Ok(()),
    }
}

fn tools() -> [ToolDefinition; 1] {
    [ToolDefinition::from_schema::<TimeArgs>(GET_TIME)
        .description("Get the current time at an offset from UTC.")]
}

/// Run one tool call. An error goes back to the model as a tool error.
fn run_tool(call: &ToolCall) -> Result<String, String> {
    match call.name.as_str() {
        GET_TIME => {
            let args: TimeArgs = call.parse_arguments().map_err(|error| error.to_string())?;
            if !(-12..=14).contains(&args.utc_offset_hours) {
                return Err("utc_offset_hours must be between -12 and 14".into());
            }
            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map_err(|error| error.to_string())?;
            let seconds = now.as_secs() as i64 + args.utc_offset_hours * 3600;
            let minutes = seconds.rem_euclid(86_400) / 60;
            Ok(format!("{:02}:{:02}", minutes / 60, minutes % 60))
        }
        other => Err(format!("unknown tool `{other}`; the tool is `{GET_TIME}`")),
    }
}

/// Prove the key and the model id with one small call.
async fn smoke_call(provider: &impl ChatProvider, model: &str) -> Result<()> {
    // The answer does not matter, and a cut-off one is expected. 16 is the
    // least OpenAI's Responses API accepts: below it, the call fails with
    // `InvalidRequest`, and the model id goes unchecked.
    let request = ChatRequest::new(model)
        .user("Reply with OK.")
        .max_output_tokens(16);
    match provider.chat(&request).await {
        Ok(_) => Ok(()),
        // A rejected key, a model the provider does not have, a missing
        // credential: every turn would fail the same way.
        Err(
            error
            @ (Error::Auth { .. } | Error::ModelNotFound { .. } | Error::Configuration { .. }),
        ) => Err(error),
        // An outage or a rate limit is not a setup mistake.
        Err(error) => {
            eprintln!("[warning] smoke call failed, going ahead: {error}");
            Ok(())
        }
    }
}

/// Build the provider for a model address.
fn connect(address: &str) -> Result<(DynChatProvider, String)> {
    let settings = ProviderSettings::new().retry(Some(
        RetryPolicy::new().max_retry_after(Duration::from_secs(5)),
    ));
    allama_providers::chat::from_model_with(address, &settings)
}

/// An optional setting from the environment; unset or empty is `None`.
fn setting<T: FromStr>(name: &str) -> Result<Option<T>, String>
where
    T::Err: std::fmt::Display,
{
    let parse = |value: String| value.parse().map_err(|error| format!("{name}: {error}"));
    env(name).map(parse).transpose()
}

fn env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use allama::mock::{ChatResponseBuilder, MockProvider, MockStreamEvent, MockStreamingProvider};
    use serde_json::json;

    use super::*;

    fn chat(provider: MockStreamingProvider) -> Chat {
        Chat::new(
            DynChatProvider::new(provider),
            "m".into(),
            Settings::default(),
        )
    }

    fn time_call(id: &str) -> ChatResponse {
        ChatResponseBuilder::new()
            .tool_call(id, GET_TIME, json!({ "utc_offset_hours": 1 }))
            .usage(10, 2)
            .build()
    }

    #[tokio::test]
    async fn a_reply_is_stored_with_its_question() {
        let reply = ChatResponseBuilder::new().text("Hi!").usage(3, 2).build();
        let mut chat = chat(MockStreamingProvider::from_responses([
            reply.clone(),
            reply,
        ]));

        chat.send("Hello").await.unwrap();
        chat.send("Again").await.unwrap();

        assert_eq!(chat.history.len(), 4);
        assert_eq!(chat.session.output_tokens, Some(4));
    }

    #[tokio::test]
    async fn a_tool_round_trip_is_stored_with_the_answer() {
        let answer = ChatResponseBuilder::new()
            .text("It is 14:30.")
            .usage(20, 5)
            .build();
        let mut chat = chat(MockStreamingProvider::from_responses([
            time_call("c1"),
            answer,
        ]));

        chat.send("What time is it in Lisbon?").await.unwrap();

        // The user's line, the call, its result, the answer.
        assert!(matches!(
            chat.history.as_slice(),
            [
                Message::User(_),
                Message::Assistant(_),
                Message::Tool(_),
                Message::Assistant(_)
            ]
        ));
        // Usage of both calls.
        assert_eq!(chat.session.input_tokens, Some(30));
        assert_eq!(chat.session.output_tokens, Some(7));
    }

    #[tokio::test]
    async fn the_step_cap_rolls_back_the_turn() {
        let calls = (0..MAX_STEPS).map(|step| time_call(&format!("c{step}")));
        let mut chat = chat(MockStreamingProvider::from_responses(calls));

        let error = chat.send("Time?").await.unwrap_err();

        assert_eq!(error.kind_name(), "unexpected_response");
        assert!(chat.history.is_empty());
        // Every call was billed.
        assert_eq!(chat.session.input_tokens, Some(10 * MAX_STEPS as u64));
    }

    #[tokio::test]
    async fn a_failure_after_a_tool_round_trip_rolls_back_the_turn() {
        let failed = vec![
            StreamEvent::response_start(None, None).into(),
            StreamEvent::text_delta(0, "It is").into(),
            MockStreamEvent::from(Error::overloaded("busy")),
        ];
        let mut chat = chat(MockStreamingProvider::new([
            MockStreamEvent::from_response(time_call("c1")),
            failed,
        ]));

        assert!(chat.send("Time?").await.is_err());
        assert!(chat.history.is_empty());
    }

    #[tokio::test]
    async fn a_turn_that_fails_partway_is_rolled_back() {
        let mut chat = chat(MockStreamingProvider::with_stream([
            StreamEvent::response_start(None, None).into(),
            StreamEvent::text_delta(0, "Light rain").into(),
            MockStreamEvent::from(Error::overloaded("busy")),
        ]));

        assert!(chat.send("Weather?").await.is_err());
        assert!(chat.history.is_empty());
    }

    #[tokio::test]
    async fn a_filtered_reply_is_dropped_and_a_cut_off_one_kept() {
        let filtered = ChatResponseBuilder::new()
            .text("Once upon")
            .finish_reason(FinishReason::ContentFilter)
            .build();
        let cut_off = ChatResponseBuilder::new()
            .text("Once upon")
            .finish_reason(FinishReason::Length)
            .build();
        let mut chat = chat(MockStreamingProvider::from_responses([filtered, cut_off]));

        let error = chat.send("A story?").await.unwrap_err();
        assert_eq!(error.kind_name(), "content_filtered");
        assert!(chat.history.is_empty());
        chat.send("A story?").await.unwrap();
        assert_eq!(chat.history.len(), 2);
    }

    #[test]
    fn a_cut_off_tool_call_is_not_stored() {
        let reply = ChatResponseBuilder::new()
            .tool_call("c1", GET_TIME, json!({ "utc_offset_hours": 1 }))
            .finish_reason(FinishReason::Length)
            .build();
        assert!(storable(&reply).is_err());
    }

    #[test]
    fn bad_arguments_go_back_as_tool_errors() {
        let call = ToolCall::new("c1", GET_TIME, r#"{"utc_offset_hours": 99}"#);
        assert!(run_tool(&call).is_err());
        let call = ToolCall::new("c2", "get_weather", "{}");
        assert!(run_tool(&call).is_err());
    }

    #[tokio::test]
    async fn a_setup_mistake_keeps_the_current_model() {
        let mut chat = chat(MockStreamingProvider::from_responses([]));
        let missing = MockProvider::with_error(Error::model_not_found("no such model"));

        let error = chat
            .switch(DynChatProvider::new(missing), "gpt-5-mini-typo".into())
            .await
            .unwrap_err();

        assert_eq!(error.kind_name(), "model_not_found");
        assert_eq!(chat.model, "m");
    }

    #[tokio::test]
    async fn an_outage_does_not_block_a_switch() {
        let mut chat = chat(MockStreamingProvider::from_responses([]));
        let busy = MockProvider::with_error(Error::overloaded("busy"));

        chat.switch(DynChatProvider::new(busy), "other".into())
            .await
            .unwrap();

        assert_eq!(chat.model, "other");
    }

    #[tokio::test]
    async fn the_smoke_call_asks_for_at_least_16_tokens() {
        let provider = MockProvider::with_text("OK");
        smoke_call(&provider, "m").await.unwrap();
        let sent = provider.requests();
        assert_eq!(sent[0].max_output_tokens, Some(16));
    }
}
