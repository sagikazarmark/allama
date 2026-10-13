//! A streaming chat endpoint with axum and server-sent events.
//!
//! This is the server the [applications guide] quotes: one `POST /chat`
//! request is one turn of a stored conversation, streamed back as SSE
//! events. Along the way the model may call one tool, which the server
//! runs itself.
//!
//! It shows:
//!
//! - storing a conversation as a [`History`], which holds messages only,
//!   and setting the system prompt on every request;
//! - the commit point: the handler waits for the first model output with
//!   [`until_committed`] before it answers, so an error before it (a rate
//!   limit, an overload) becomes an HTTP status code, while an error after
//!   `200 OK` becomes an SSE `error` event;
//! - the recommended provider stack for an interactive server,
//!   [`provider_stack`], defined once and used by `main` and the handler
//!   tests alike: tracing, a failover counter, a deadline on the first
//!   output, and an optional fallback (`ALLAMA_FALLBACK_MODEL`), around
//!   providers built with [`provider_settings`]: retries that give up at
//!   once on a long `Retry-After`, around a per-attempt timeout;
//! - warnings: the warning events the subscriber logs, and failovers counted
//!   for a metric with [`ObserveProvider`];
//! - the turn running in its own task from the start, reporting its commit
//!   point over a oneshot and its events over a channel, so it finishes
//!   and is stored even when the client goes away before the first output;
//! - each reply's envelope (usage, ids) stored in its message's application
//!   data under [`TURN_ENVELOPE_KEY`];
//! - tool-call progress as its own SSE events (`tool_call`, `tool_result`);
//! - what to store when the turn fails after the commit point, checked
//!   with [`history::validate`];
//! - logging errors at two levels: the redacted [`Error::as_log`] at info
//!   and the provider's own message at debug;
//! - handler tests with mocks: [`TestProvider`](allama::mock::TestProvider)
//!   behind a `Registry` key, and scripted failures with
//!   [`MockStreamingProvider`](allama::mock::MockStreamingProvider).
//!
//! The SSE events are `delta` (text), `refusal` (the model declining to
//! answer, which is never answer text), `tool_call` and `tool_result` (JSON
//! with the call id and tool name), `error` (a message for the user), and
//! `done` (JSON with the token total).
//!
//! To copy this file into an application, depend on:
//!
//! ```toml
//! allama = { version = "0.1", features = ["schema", "tracing"] }
//! allama-providers = "0.1"
//! axum = "0.8"
//! futures-util = "0.3"
//! serde = { version = "1", features = ["derive"] }
//! serde_json = "1"
//! tokio = { version = "1", features = ["macros", "net", "rt-multi-thread", "sync"] }
//! tracing = "0.1"
//! tracing-subscriber = "0.3"
//! ```
//!
//! Its tests also need `allama/mock`. Without `ALLAMA_MODEL` it uses
//! [`DEFAULT_MODEL`]; `ALLAMA_FALLBACK_MODEL`, when set, names the model
//! address a failed call fails over to. A local Ollama needs no credentials:
//!
//! ```text
//! ALLAMA_MODEL=ollama:qwen3:1.7b cargo run -p allama-examples --example axum_sse
//! curl -N 127.0.0.1:3000/chat -H 'content-type: application/json' \
//!     -d '{"conversation_id":"demo","message":"Do I need an umbrella in Lisbon?"}'
//! ```
//!
//! [applications guide]: https://github.com/sagikazarmark/allama/blob/main/docs/guides/applications.md

use std::borrow::Cow;
use std::collections::HashMap;
use std::convert::Infallible;
use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use allama::chat::stream::{Prefixed, until_committed};
use allama::history::{self, History, HistoryIssue, TURN_ENVELOPE_KEY, ValidationMode};
use allama::prelude::*;
use allama::schemars::{self, JsonSchema};
use allama::wrappers::{
    FallbackProvider, MapModelProvider, ObserveProvider, RetryPolicy, TimeoutPolicy,
    TimeoutProvider, TracingProvider,
};
use allama::{Subject, Warning, WarningKind};
use allama_providers::ProviderSettings;
use axum::extract::State;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::{mpsc, oneshot};
use tracing::Level;
use tracing_subscriber::filter::Targets;
use tracing_subscriber::prelude::*;

/// The model address used when `ALLAMA_MODEL` is unset. It reads
/// `OPENAI_API_KEY`.
const DEFAULT_MODEL: &str = "openai:gpt-5-mini";

/// Where the server listens.
const LISTEN: &str = "127.0.0.1:3000";

const SYSTEM_PROMPT: &str = "You are a concise travel assistant. \
     Use the get_weather tool for weather questions. Answer in two sentences at most.";

/// How much of a provider's error message the debug log keeps.
const DEBUG_MESSAGE_CHARS: usize = 200;

/// What the client sees when the provider is busy.
const BUSY: &str = "The assistant is busy. Try again shortly.";

/// How long a request may wait for the first model output, across every
/// attempt, backoff, and failover. Past it the client gets `503` before
/// anything was sent. Sized with the attempt bounds in
/// [`provider_settings`]: three waits of 15 s for a hung primary's headers,
/// and their backoff, take about 46 s, which leaves the fallback 14 s.
/// Reasoning models that do not stream their reasoning can take minutes to
/// their first output: raise it (and the per-attempt bounds) for them.
const FIRST_OUTPUT_DEADLINE: Duration = Duration::from_secs(60);

/// How long a unary call (`chat`) may take, across every attempt, backoff,
/// and failover. Sized with the attempt bounds in [`provider_settings`]:
/// three unary attempts of 30 s, and their backoff, take about 91 s, which
/// leaves the fallback about 30 s.
const UNARY_DEADLINE: Duration = Duration::from_secs(120);

/// The longest provider `Retry-After` a request waits out. A longer rate
/// limit fails over, or reaches the client as `429` with the hint, at once.
const MAX_RETRY_AFTER: Duration = Duration::from_secs(5);

/// State shared by every request.
struct App {
    provider: DynChatProvider,
    model: String,
    conversations: Store,
}

/// The body of `POST /chat`.
#[derive(Debug, Deserialize)]
struct ChatBody {
    conversation_id: String,
    message: String,
}

/// Arguments of the `get_weather` tool; the schema is derived from the type.
#[derive(Debug, Deserialize, JsonSchema)]
struct WeatherArgs {
    /// City name, such as "Lisbon".
    city: String,
}

/// Conversations by id, each stored as a serialized [`History`]. The
/// system prompt is not part of it: it comes from the code on
/// every request, so a deploy that changes it applies to old conversations
/// too. A real server keeps these in its database.
#[derive(Default)]
struct Store {
    conversations: Mutex<HashMap<String, String>>,
}

/// What one streamed provider call produced.
struct Reply {
    response: ChatResponse,
    /// Client tool calls the model made, for the server to run.
    tool_calls: Vec<ToolCall>,
}

#[tokio::main]
async fn main() -> ExitCode {
    // Debug logs from this server, info from everything else. That includes
    // allama's warning events (a target per kind, `allama::warning::dropped`
    // and so on): a warning about the request's shape, such as a hint the
    // model ignores, logs at WARN once and at DEBUG on later calls, while a
    // failover logs at WARN every time. Retries are `allama::retry` events
    // at INFO.
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(
            Targets::new()
                .with_target("axum_sse", Level::DEBUG)
                .with_default(Level::INFO),
        )
        .init();

    // Startup: build the provider stack once from configuration and share
    // it.
    let address = std::env::var("ALLAMA_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
    let (provider, model) =
        match allama_providers::chat::from_model_with(&address, &provider_settings()) {
            Ok(selected) => selected,
            Err(error) => {
                log_error(&error);
                return ExitCode::FAILURE;
            }
        };
    let fallback = match std::env::var("ALLAMA_FALLBACK_MODEL") {
        Ok(fallback) => match fallback_side(&fallback) {
            Ok(provider) => Some(provider),
            Err(error) => {
                log_error(&error);
                return ExitCode::FAILURE;
            }
        },
        Err(_) => None,
    };
    let app = App::new(provider_stack(provider, fallback), model);

    let listener = match tokio::net::TcpListener::bind(LISTEN).await {
        Ok(listener) => listener,
        Err(error) => {
            tracing::error!(%error, "cannot listen on {LISTEN}");
            return ExitCode::FAILURE;
        }
    };
    tracing::info!(%address, "listening on http://{LISTEN}/chat");
    match axum::serve(listener, router(app)).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!(%error, "server failed");
            ExitCode::FAILURE
        }
    }
}

/// How each provider is built: retries that give up at once on a long
/// `Retry-After`, around a bound on each attempt, so a stalled attempt is
/// retried. `allama_providers` wraps the provider in them, so the stack
/// adds no retries of its own.
fn provider_settings() -> ProviderSettings {
    ProviderSettings::new()
        .retry(Some(RetryPolicy::new().max_retry_after(MAX_RETRY_AFTER)))
        .attempt_timeout(Some(
            TimeoutPolicy::new()
                // A unary attempt: the whole generation, reasoning included.
                .request(Duration::from_secs(30))
                // A streamed attempt: until its headers, then each event.
                .stream_setup(Duration::from_secs(15))
                .stream_idle(Duration::from_secs(30)),
        ))
}

/// The provider stack, outermost first, around providers built with
/// [`provider_settings`]. `main` and the handler tests both build it here,
/// so the tests run on the real stack.
fn provider_stack(primary: DynChatProvider, fallback: Option<DynChatProvider>) -> DynChatProvider {
    let routed = match fallback {
        Some(fallback) => DynChatProvider::new(FallbackProvider::new(primary, fallback)),
        None => primary,
    };
    // A stream through retry and fallback opens at its first output, so
    // `stream_setup` here bounds the wait for it across every attempt, as
    // `request` bounds a whole unary call. Not `stream_total`, which would
    // cut off long answers. Inside tracing, so an expired deadline is
    // recorded as a timeout, not a cancelled call.
    let deadline = TimeoutProvider::new(
        routed,
        TimeoutPolicy::new()
            .request(UNARY_DEADLINE)
            .stream_setup(FIRST_OUTPUT_DEADLINE),
    );
    // Every warning of every call, for a metric: the failover warnings,
    // which the logs also show, counted where an alert can read them.
    let observed = ObserveProvider::new(deadline).on_warning(count_failovers);
    DynChatProvider::new(TracingProvider::new(observed))
}

/// Failovers since the server started. Export it as a metric: sustained
/// failover means the primary is down, even while every request succeeds.
static FAILOVERS: AtomicU64 = AtomicU64::new(0);

fn count_failovers(_model: &str, warning: &Warning) {
    if warning.kind == WarningKind::Fallback {
        FAILOVERS.fetch_add(1, Ordering::Relaxed);
    }
}

/// The fallback provider for a model address. Requests name the primary's
/// model, so the fallback sends its own.
fn fallback_side(address: &str) -> Result<DynChatProvider> {
    let (provider, model) = allama_providers::chat::from_model_with(address, &provider_settings())?;
    Ok(DynChatProvider::new(MapModelProvider::new(
        provider,
        move |_: &str| Cow::Owned(model.clone()),
    )))
}

fn router(app: Arc<App>) -> Router {
    Router::new().route("/chat", post(chat)).with_state(app)
}

async fn chat(State(app): State<Arc<App>>, Json(body): Json<ChatBody>) -> Response {
    // The system prompt comes from the code; the stored messages follow it.
    let request = ChatRequest::new(&app.model)
        .system(SYSTEM_PROMPT)
        .extend_messages(app.conversations.load(&body.conversation_id))
        .user(body.message)
        .tools([weather_tool()]);

    // The turn runs in its own task from the start, so it finishes and is
    // stored even if the client goes away before the first output. The
    // oneshot reports the commit point; the channel carries the SSE events.
    let (committed, commit) = oneshot::channel();
    let (events, mut receiver) = mpsc::channel(32);
    let turn = run_turn(app, body.conversation_id, request, committed, events);
    tokio::spawn(turn);

    // The commit point: nothing is sent until the first model output, so an
    // error before it still chooses the status code.
    match commit.await {
        Ok(Ok(())) => {}
        Ok(Err(error)) => return error_response(&error),
        // The task ended without an answer: it panicked.
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
    let events = futures_util::stream::poll_fn(move |cx| receiver.poll_recv(cx));
    Sse::new(events.map(Ok::<_, Infallible>))
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Start the provider call and wait for its first model output.
async fn open(provider: &DynChatProvider, request: &ChatRequest) -> Result<Prefixed<ChatStream>> {
    let stream = provider.chat_stream(request).await?;
    until_committed(stream).await
}

/// The whole turn: open the provider call and report the commit point to
/// the handler, then stream the reply, run one round of tool calls, and
/// store the conversation.
async fn run_turn(
    app: Arc<App>,
    conversation_id: String,
    mut request: ChatRequest,
    committed: oneshot::Sender<Result<()>>,
    events: mpsc::Sender<Event>,
) {
    let stream = match open(&app.provider, &request).await {
        Ok(stream) => stream,
        Err(error) => {
            // The handler answers with a status. If the client already
            // left, nobody does: log the error here.
            if let Err(Err(error)) = committed.send(Err(error)) {
                log_error(&error);
            }
            return;
        }
    };
    // Past the commit point: the handler sends `200 OK`, if it is still
    // there, and the turn goes on either way.
    let _ = committed.send(Ok(()));

    match finish_turn(&app.provider, &mut request, stream, &events).await {
        Ok(usage) => {
            app.conversations.save(&conversation_id, &request.messages);
            let total = usage.total().unwrap_or_default();
            let done = json!({ "total_tokens": total }).to_string();
            send(&events, Event::default().event("done").data(done)).await;
        }
        Err(error) => {
            // The status is already sent: report the failure as an event.
            log_error(&error);
            let (_, message) = public_error(&error);
            send(&events, Event::default().event("error").data(message)).await;
            // Keep the turn up to its last complete step.
            match checkpoint(request.messages) {
                Ok(messages) => app.conversations.save(&conversation_id, &messages),
                Err(issues) => tracing::warn!(?issues, "not storing a malformed conversation"),
            }
        }
    }
}

/// Stream the committed reply, then at most one round of tool calls and a
/// second reply. Returns the usage of both calls.
async fn finish_turn(
    provider: &DynChatProvider,
    request: &mut ChatRequest,
    stream: Prefixed<ChatStream>,
    events: &mpsc::Sender<Event>,
) -> Result<Usage> {
    let first = stream_reply(stream, events).await?;
    let mut usage: Vec<Usage> = first.response.usage.iter().cloned().collect();
    request.push_message(stored_reply(&first.response)?);
    if first.tool_calls.is_empty() {
        return Ok(usage.iter().sum());
    }

    // The tool round trip: one result per call, then ask again.
    for call in &first.tool_calls {
        let result = run_tool(call);
        let is_error = result.is_err();
        match result {
            Ok(output) => request.push_tool_result(call, output),
            Err(error) => request.push_tool_error(call, error),
        }
        let data = json!({ "id": call.id, "name": call.name, "is_error": is_error });
        send(
            events,
            Event::default().event("tool_result").data(data.to_string()),
        )
        .await;
    }
    // One round trip only: the follow-up must answer in text.
    request.tool_choice = Some(ToolChoice::Disabled);

    let second = stream_reply(provider.chat_stream(request).await?, events).await?;
    usage.extend(second.response.usage.iter().cloned());
    request.push_message(stored_reply(&second.response)?);
    Ok(usage.iter().sum())
}

/// Forward one provider stream as SSE events and collect its response.
async fn stream_reply(
    stream: impl ChatStreamExt + Unpin,
    events: &mpsc::Sender<Event>,
) -> Result<Reply> {
    let mut stream = stream.with_collector();
    let mut tool_calls = Vec::new();
    while let Some(item) = stream.next_event().await {
        let item = item?;
        match item.event {
            StreamEvent::TextDelta { text, .. } => {
                send(events, Event::default().event("delta").data(text)).await;
            }
            // A refusal is never answer text: the client shows it as such.
            StreamEvent::RefusalDelta { text, .. } => {
                send(events, Event::default().event("refusal").data(text)).await;
            }
            // Announce a tool call as soon as its name is known...
            StreamEvent::BlockStart {
                block_type: StreamBlockType::ToolCall,
                id,
                name,
                ..
            } => {
                let data = json!({ "id": id, "name": name });
                send(
                    events,
                    Event::default().event("tool_call").data(data.to_string()),
                )
                .await;
            }
            _ => {}
        }
        // ...and keep it once its arguments are complete.
        if let Some(ContentBlock::ToolCall(call)) = item.completed {
            tool_calls.push(call);
        }
    }
    Ok(Reply {
        response: stream.finish()?,
        tool_calls,
    })
}

/// The assistant message to store for a reply, with the reply's envelope
/// (id, request id, model, provider, usage) in its application data for
/// cost accounting and support. Providers never see it.
fn stored_reply(response: &ChatResponse) -> Result<Message> {
    let mut message = response.to_assistant_message();
    message.app_data_insert(TURN_ENVELOPE_KEY, &response.envelope())?;
    Ok(message)
}

/// What to store after a turn failed past the commit point: the turn up to
/// its last complete step.
///
/// The user message and every finished tool round trip stay. An assistant
/// message whose tool calls never got their results (the tool could not
/// run, or the task stopped) is the dangling tool call: providers reject it
/// with a 400, so it goes, with any partial results after it.
fn checkpoint(mut messages: Vec<Message>) -> Result<Vec<Message>, Vec<HistoryIssue>> {
    // Anything but calls still waiting at the end is a bug: store nothing.
    history::validate(&messages, ValidationMode::Resumable)?;
    // The calls still waiting, if any, belong to the last assistant message.
    if let Err(issues) = history::validate(&messages, ValidationMode::Sendable) {
        let dangling = issues.iter().map(HistoryIssue::message).min();
        messages.truncate(dangling.unwrap_or(messages.len()));
    }
    Ok(messages)
}

/// Send one SSE event. A client that went away is not an error: the turn
/// still finishes and is stored.
async fn send(events: &mpsc::Sender<Event>, event: Event) {
    let _ = events.send(event).await;
}

fn weather_tool() -> ToolDefinition {
    ToolDefinition::from_schema::<WeatherArgs>("get_weather")
        .description("Look up tomorrow's weather forecast for a city.")
}

/// The server-side tool. Bad arguments go back to the model as a tool
/// error so it can correct itself.
fn run_tool(call: &ToolCall) -> Result<String, String> {
    match call.name.as_str() {
        "get_weather" => {
            let args: WeatherArgs = call.parse_arguments().map_err(|error| error.to_string())?;
            Ok(
                json!({ "city": args.city, "forecast": "light rain until noon", "high_c": 19 })
                    .to_string(),
            )
        }
        other => Err(format!("unknown tool {other}")),
    }
}

/// The HTTP response for an error before the commit point.
fn error_response(error: &Error) -> Response {
    log_error(error);
    let (status, message) = public_error(error);
    let mut response = (status, Json(json!({ "error": message }))).into_response();
    if let Some(delay) = error.retry_after() {
        // Whole seconds, rounded up.
        let seconds = delay.as_secs() + u64::from(delay.subsec_nanos() > 0);
        response
            .headers_mut()
            .insert(header::RETRY_AFTER, HeaderValue::from(seconds));
    }
    response
}

/// The status code and the message a client sees, by error kind.
fn public_error(error: &Error) -> (StatusCode, &'static str) {
    // When every side of a fallback failed, the error is the last failure,
    // so it maps like any other.
    match error {
        // Content the user sent that the model cannot take, such as an
        // image for a text-only model.
        Error::Unsupported { .. }
            if matches!(error.unsupported_subject(), Some(Subject::Content { .. })) =>
        {
            (
                StatusCode::BAD_REQUEST,
                "The assistant cannot read this kind of content.",
            )
        }
        // The server is set up wrong: a missing API key, an unknown
        // provider, a model the provider does not have, or a feature the
        // server asks for (tools, a response format) that the model lacks.
        Error::Configuration { .. } | Error::ModelNotFound { .. } | Error::Unsupported { .. } => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "The assistant is not configured.",
        ),
        Error::ContextLengthExceeded { .. } => (
            StatusCode::PAYLOAD_TOO_LARGE,
            "This conversation is too long. Start a new one.",
        ),
        Error::ContentFiltered { .. } => (
            StatusCode::UNPROCESSABLE_ENTITY,
            "The request was blocked by the content filter.",
        ),
        Error::InvalidRequest { .. } => (
            StatusCode::BAD_REQUEST,
            "The request could not be processed.",
        ),
        Error::RateLimited { .. } => (StatusCode::TOO_MANY_REQUESTS, BUSY),
        _ if error.is_retryable() => (StatusCode::SERVICE_UNAVAILABLE, BUSY),
        // Bad credentials, exhausted quota, and other provider failures.
        _ => (StatusCode::BAD_GATEWAY, "The assistant is unavailable."),
    }
}

/// Log an error at two levels: the redacted view at info, and the
/// provider's own message, which may echo the prompt, at debug.
fn log_error(error: &Error) {
    // One metric label per kind; a failed failover counts as its last
    // failure, and its redacted log names the earlier ones.
    let kind = error.kind_name();
    tracing::info!(kind, error = %error.as_log(), "chat turn failed");
    tracing::debug!(
        error = %error.as_log().with_message(DEBUG_MESSAGE_CHARS),
        "chat turn failed",
    );
}

impl App {
    fn new(provider: DynChatProvider, model: impl Into<String>) -> Arc<Self> {
        Arc::new(Self {
            provider,
            model: model.into(),
            conversations: Store::default(),
        })
    }
}

impl Store {
    /// The stored messages of a conversation; none for a new one.
    fn load(&self, conversation_id: &str) -> Vec<Message> {
        let conversations = self.conversations.lock().unwrap();
        conversations
            .get(conversation_id)
            .map_or_else(Vec::new, |stored| {
                let history: History = serde_json::from_str(stored).expect("stored histories load");
                history.messages
            })
    }

    fn save(&self, conversation_id: &str, messages: &[Message]) {
        let history = History::new(messages.to_vec());
        let stored = serde_json::to_string(&history).expect("histories serialize");
        let mut conversations = self.conversations.lock().unwrap();
        conversations.insert(conversation_id.to_owned(), stored);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use allama::adapter::Degradations;
    use allama::history::TurnEnvelope;
    use allama::mock::{MockStreamEvent, MockStreamingProvider, TestProvider};
    use allama::{RequestField, UnsupportedPolicy};
    use allama_providers::Registry;

    use super::*;

    /// A refusal arrives as its own SSE event, never as a `delta`.
    #[tokio::test]
    async fn a_refusal_is_its_own_event() {
        let refused = MockStreamingProvider::with_stream([
            StreamEvent::response_start(None, None).into(),
            StreamEvent::block_start(0, StreamBlockType::Refusal).into(),
            StreamEvent::refusal_delta(0, "I can't help with that.").into(),
            StreamEvent::block_stop(0).into(),
            MockStreamEvent::from(
                allama::chat::stream::MetadataBuilder::new()
                    .with_finish_reason(allama::FinishReason::Refusal)
                    .build(),
            ),
            StreamEvent::ResponseStop.into(),
        ]);
        let app = App::new(stack(refused), "mock-model");
        let (status, _, body) = post(&app, "Help me with something bad.").await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            body.contains("event: refusal\ndata: I can't help with that."),
            "{body}"
        );
        assert!(!body.contains("event: delta"), "{body}");
    }

    /// A rate limit hint longer than the stack waits out.
    const LONG_HINT: Duration = Duration::from_secs(30);

    /// A mock built the way `main` builds a provider: through a registry,
    /// with [`provider_settings`].
    fn side<P>(provider: P) -> DynChatProvider
    where
        P: ChatProvider + Clone + Send + Sync + 'static,
    {
        let mut registry = Registry::new();
        registry.register_chat("mock", move || Ok(provider.clone()));
        let (provider, _) = registry
            .chat_from_model_with("mock:mock-model", &provider_settings())
            .unwrap();
        provider
    }

    /// The real provider stack around a mock, without a fallback.
    fn stack<P>(provider: P) -> DynChatProvider
    where
        P: ChatProvider + Clone + Send + Sync + 'static,
    {
        provider_stack(side(provider), None)
    }

    /// Call the handler the way axum does and read the whole response.
    async fn post(app: &Arc<App>, message: &str) -> (StatusCode, Option<String>, String) {
        let body = ChatBody {
            conversation_id: "c1".into(),
            message: message.into(),
        };
        let response = chat(State(app.clone()), Json(body)).await;
        let status = response.status();
        let retry_after = response
            .headers()
            .get(header::RETRY_AFTER)
            .map(|value| value.to_str().unwrap().to_owned());
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        (
            status,
            retry_after,
            String::from_utf8(body.to_vec()).unwrap(),
        )
    }

    #[tokio::test]
    async fn a_conversation_continues_across_requests() {
        // The application builds its provider from a model address; the
        // test registers a mock under a key of its own.
        let mut registry = Registry::new();
        registry.register_chat("mock", || Ok(TestProvider::new()));
        let (provider, model) = registry
            .chat_from_model_with("mock:any-model", &provider_settings())
            .unwrap();
        let app = App::new(provider_stack(provider, None), model);

        // `TestProvider` answers every request, so one test can send many.
        for message in ["Rain in Lisbon?", "And in Porto?"] {
            let (status, _, body) = post(&app, message).await;
            assert_eq!(status, StatusCode::OK);
            assert!(body.contains("event: tool_call"));
            assert!(body.contains("event: done"));
        }

        // Per turn: user, assistant tool call, tool result, assistant answer.
        assert_eq!(app.conversations.load("c1").len(), 8);
    }

    #[tokio::test]
    async fn every_provider_call_stores_its_envelope() {
        let app = App::new(stack(TestProvider::new()), "mock-model");
        post(&app, "Rain in Lisbon?").await;

        // The tool call and the answer are two provider calls: two
        // assistant messages, each with the envelope of its own call.
        let stored = app.conversations.load("c1");
        let envelopes: Vec<TurnEnvelope> = stored
            .iter()
            .filter(|message| message.role() == "assistant")
            .map(|message| {
                message
                    .app_data_get(TURN_ENVELOPE_KEY)
                    .unwrap()
                    .expect("stored with the reply")
            })
            .collect();
        assert_eq!(envelopes.len(), 2);
        assert!(envelopes.iter().all(|turn| turn.usage.is_some()));
        assert_ne!(envelopes[0].id, envelopes[1].id);
    }

    #[tokio::test]
    async fn a_resumed_conversation_sends_one_system_prompt() {
        let provider = MockStreamingProvider::build(|b| b.text("Hello!").text("Still here."));
        let app = App::new(stack(provider.clone()), "mock-model");

        post(&app, "Hi").await;
        post(&app, "Are you there?").await;

        let resumed = &provider.requests()[1];
        assert_eq!(resumed.system.len(), 1);
        assert_eq!(resumed.messages.len(), 3);
    }

    /// A client that leaves before the first output, while a reasoning
    /// model thinks, loses nothing: the turn runs in its own task from the
    /// start, so it finishes and is stored.
    #[tokio::test]
    async fn a_client_that_leaves_before_the_first_output_keeps_the_turn() {
        let thinking = Duration::from_millis(200);
        let mut events = vec![MockStreamEvent::delayed(thinking)];
        events.extend(MockStreamEvent::text_response("Take an umbrella."));
        let app = App::new(
            stack(MockStreamingProvider::with_stream(events)),
            "mock-model",
        );

        // The client gives up first: axum drops the handler.
        let body = ChatBody {
            conversation_id: "c1".into(),
            message: "Rain in Lisbon?".into(),
        };
        let handler = chat(State(app.clone()), Json(body));
        let left = tokio::time::timeout(thinking / 4, handler).await;
        assert!(
            left.is_err(),
            "the handler answered before the commit point"
        );

        // The turn still finishes and is stored: the user message and the
        // answer.
        let started = std::time::Instant::now();
        while app.conversations.load("c1").is_empty() && started.elapsed() < 10 * thinking {
            tokio::time::sleep(thinking / 10).await;
        }
        assert_eq!(app.conversations.load("c1").len(), 2);
    }

    #[tokio::test]
    async fn errors_before_the_first_output_choose_the_status() {
        let retry_after = Duration::from_secs(7);
        // The provider rejects the request: `chat_stream` itself fails.
        let rejected = MockStreamingProvider::build(|b| {
            b.setup_error(Error::rate_limited("slow down").with_retry_after(retry_after))
        });
        // The stream opens, then fails before any model output.
        let overloaded = MockStreamingProvider::with_stream([
            MockStreamEvent::from(StreamEvent::response_start(None, None)),
            MockStreamEvent::from(Error::overloaded("busy").with_retry_after(retry_after)),
        ]);

        let app = App::new(stack(rejected), "mock-model");
        let (status, retry, _) = post(&app, "Hi").await;
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(retry.as_deref(), Some("7"));

        let app = App::new(stack(overloaded), "mock-model");
        let (status, retry, _) = post(&app, "Hi").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(retry.as_deref(), Some("7"));
        assert!(app.conversations.load("c1").is_empty());
    }

    /// A rate limit longer than the stack waits out reaches the client at
    /// once, as `429` with the provider's hint.
    #[tokio::test]
    async fn a_long_rate_limit_is_handed_to_the_client_at_once() {
        let provider = MockStreamingProvider::build(|b| {
            b.setup_error(Error::rate_limited("slow down").with_retry_after(LONG_HINT))
                .text("too early")
        });
        let app = App::new(stack(provider.clone()), "mock-model");

        let started = std::time::Instant::now();
        let (status, retry, _) = post(&app, "Hi").await;

        assert!(started.elapsed() < Duration::from_secs(1));
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(retry.as_deref(), Some("30"));
        assert_eq!(provider.call_count(), 1, "not retried before the hint");
    }

    /// With a fallback configured, the same rate limit fails over at once.
    #[tokio::test]
    async fn a_long_rate_limit_fails_over_at_once() {
        let primary = MockStreamingProvider::build(|b| {
            b.setup_error(Error::rate_limited("slow down").with_retry_after(LONG_HINT))
        });
        let fallback = MockStreamingProvider::build(|b| b.text("From the fallback."));
        let app = App::new(
            provider_stack(side(primary.clone()), Some(side(fallback.clone()))),
            "mock-model",
        );

        let failovers = FAILOVERS.load(Ordering::Relaxed);
        let started = std::time::Instant::now();
        let (status, _, body) = post(&app, "Hi").await;

        assert!(started.elapsed() < Duration::from_secs(1));
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("From the fallback."));
        assert_eq!(primary.call_count(), 1);
        assert_eq!(fallback.call_count(), 1);
        // The failover was counted for the metric.
        assert!(FAILOVERS.load(Ordering::Relaxed) > failovers);
    }

    #[tokio::test]
    async fn a_failure_after_the_tool_round_trip_keeps_the_round_trip() {
        // The first reply calls the tool; the follow-up call is rejected,
        // with a hint too long to retry.
        let provider = MockStreamingProvider::build(|b| {
            b.tool_call("call_1", "get_weather", json!({ "city": "Lisbon" }))
                .setup_error(Error::overloaded("busy").with_retry_after(LONG_HINT))
        });
        let app = App::new(stack(provider), "mock-model");

        let (status, _, body) = post(&app, "Rain in Lisbon?").await;

        // The first reply committed the response, so the error is an event.
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("event: tool_result"));
        assert!(body.contains(&format!("event: error\ndata: {BUSY}")));
        // Stored: the user message and the complete round trip.
        let stored = app.conversations.load("c1");
        assert_eq!(stored.len(), 3);
        assert_eq!(history::validate(&stored, ValidationMode::Sendable), Ok(()));
    }

    #[test]
    fn checkpoint_drops_the_dangling_tool_call() {
        let messages = vec![
            Message::user("Weather in Paris and Tokyo?"),
            Message::assistant_blocks(vec![
                ContentBlock::tool_call("call_1", "get_weather", r#"{"city":"Paris"}"#),
                ContentBlock::tool_call("call_2", "get_weather", r#"{"city":"Tokyo"}"#),
            ]),
            Message::tool_result("call_1", "get_weather", "Sunny"),
        ];

        let stored = checkpoint(messages.clone()).unwrap();
        assert_eq!(stored, messages[..1]);

        // A complete round trip is kept as it is.
        let mut complete = messages.clone();
        complete.push(Message::tool_result("call_2", "get_weather", "Rain"));
        assert_eq!(checkpoint(complete.clone()).unwrap(), complete);

        // A result that answers nothing is not a checkpoint.
        let mut orphan = messages[..1].to_vec();
        orphan.push(Message::tool_result("call_9", "get_weather", "Sunny"));
        assert!(checkpoint(orphan).is_err());
    }

    #[test]
    fn errors_map_to_statuses_by_kind() {
        let status = |error: Error| public_error(&error).0;
        assert_eq!(
            status(Error::configuration("OPENAI_API_KEY not set")),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(
            status(Error::model_not_found("no such model")),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(
            status(Error::context_length_exceeded("too long", Some(128_000))),
            StatusCode::PAYLOAD_TOO_LARGE
        );
        assert_eq!(
            status(Error::timeout("slow")),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(status(Error::auth("bad key")), StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn unsupported_maps_by_what_was_refused() {
        let status = |error: Error| public_error(&error).0;
        let degradations = Degradations::new("demo", UnsupportedPolicy::Warn);
        // A feature the server asks for: a setup mistake.
        assert_eq!(
            status(degradations.unsupported(RequestField::Tools, "no tool calls")),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        // An image the user sent to a text-only model: their request.
        let image = Subject::Content {
            message: 0,
            block: Some(1),
        };
        assert_eq!(
            status(degradations.reject(image, "no image input")),
            StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn a_failed_failover_maps_like_its_last_failure() {
        let status = |error: Error| public_error(&error).0;
        let rate_limited = Error::rate_limited("slow down").with_status(429);
        // What a failed failover returns: the last error, with the
        // earlier ones behind it.
        assert_eq!(
            status(
                rate_limited
                    .clone()
                    .with_fallback_from(Error::overloaded("busy"))
            ),
            StatusCode::TOO_MANY_REQUESTS
        );
        // Nested chains too.
        let chain = rate_limited
            .with_fallback_from(Error::overloaded("busy"))
            .with_fallback_from(Error::timeout("slow"));
        assert_eq!(status(chain.clone()), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(chain.kind_name(), "rate_limited");
        assert_eq!(
            status(Error::auth("bad key").with_fallback_from(Error::rate_limited("slow down"))),
            StatusCode::BAD_GATEWAY
        );
    }
}
