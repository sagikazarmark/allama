//! A chat endpoint the way a web backend writes one.
//!
//! One HTTP request is one turn: the client sends a new message, the
//! server loads the conversation so far from its database, streams the
//! answer back as it is generated, and stores the updated conversation.
//! The client never sends the history: messages from an end user are
//! untrusted (see the trust boundary in `allama::history`). Along the way the
//! model may call one tool, which the server runs itself. This example
//! plays both sides on the command line: text deltas go to stdout (SSE
//! `delta` events in a real handler), everything else to stderr.
//!
//! It shows:
//!
//! - choosing provider and model from configuration with
//!   [`allama_providers::chat::from_model`] (a model address such as
//!   `openai:gpt-5-mini`, read from `ALLAMA_MODEL`), once at startup;
//! - a system prompt plus the stored history on every request: the server
//!   sets the system prompt on every request and stores only the messages,
//!   so it appends them to a new request with
//!   [`ChatRequest::extend_messages`];
//! - streaming text deltas, and acting on a tool call as soon as its block
//!   completes ([`ChatStreamExt::with_collector`]);
//! - one tool round trip by hand: append the assistant turn, then
//!   [`ChatRequest::push_tool_result`], then ask again. There is no agent
//!   loop in allama; this is all of it;
//! - holding the response until the first model output with
//!   [`stream::until_committed`](allama::chat::stream::until_committed), so an
//!   error before it (a rate limit, an overload) still gets an HTTP status
//!   code, while an error after `200 OK` ends the stream with an SSE
//!   `error` event;
//! - logging errors at two levels: the redacted [`Error::as_log`] at info,
//!   and the provider's own message, opted in with
//!   [`ErrorLog::with_message`](allama::ErrorLog::with_message), at debug;
//! - turning errors into HTTP responses by kind: rate limits and overload
//!   with their `Retry-After` hint, configuration mistakes, and
//!   conversations that outgrew the context window; a failover that failed
//!   on every side returns its last failure, which maps like any other,
//!   with the earlier ones in [`fallback_from`](Error::fallback_from);
//! - the recommended provider stack for an interactive server
//!   ([`provider_stack`]): tracing, a deadline on the first output, and an
//!   optional fallback (`ALLAMA_FALLBACK_MODEL`), around providers built
//!   with [`provider_settings`]: retries that give up at once on a long
//!   `Retry-After`, around a per-attempt timeout;
//! - each reply's envelope (id, model, serving provider, usage, finish
//!   reason) stored on its assistant message under [`TURN_ENVELOPE_KEY`]:
//!   one per provider call, so a tool round trip stores two;
//! - token usage summed over both calls;
//! - what to store when a turn fails: nothing. The server rolls back to the
//!   conversation as it was before the turn, and the client re-sends its
//!   message on retry. (`axum_sse` makes the other choice and stores the
//!   turn up to its last complete step.)
//!
//! The `axum_sse` example is the same turn as a real axum endpoint, and the
//! applications guide (`docs/guides/applications.md`) explains the
//! patterns both use.
//!
//! Requires `allama/schema`, because the tool's parameter schema is derived
//! from a Rust type with `allama::schemars` and
//! [`ToolDefinition::from_schema`], and `allama/tracing` for
//! [`TracingProvider`]. To copy this file into an application, depend on:
//!
//! ```toml
//! allama = { version = "0.1", features = ["schema", "tracing"] }
//! allama-providers = "0.1"
//! serde = { version = "1", features = ["derive"] }
//! tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
//! ```
//!
//! It uses only `allama` and `allama-providers`, so it runs unchanged in an
//! application; its tests also need `allama/mock` as a dev-dependency.
//! Without `ALLAMA_MODEL` it uses [`DEFAULT_MODEL`]. `ALLAMA_FALLBACK_MODEL`,
//! when set, names the model address a failed call fails over to.
//!
//! ```text
//! ALLAMA_MODEL=openai:gpt-5-mini cargo run -p allama-examples --example web_backend -- "Do I need an umbrella in Lisbon?"
//! ALLAMA_MODEL=anthropic:claude-haiku-4-5 ALLAMA_FALLBACK_MODEL=openai:gpt-5-mini cargo run -p allama-examples --example web_backend
//! ```

use std::borrow::Cow;
use std::io::Write as _;
use std::process::ExitCode;
use std::time::Duration;

use allama::Subject;
use allama::chat::stream::until_committed;
use allama::history::TURN_ENVELOPE_KEY;
use allama::prelude::*;
use allama::schemars::{self, JsonSchema};
use allama::wrappers::{
    FallbackProvider, MapModelProvider, RetryPolicy, TimeoutPolicy, TimeoutProvider,
    TracingProvider,
};
use allama_providers::ProviderSettings;
use serde::Deserialize;

/// The model address used when `ALLAMA_MODEL` is unset. It reads
/// `OPENAI_API_KEY`.
const DEFAULT_MODEL: &str = "openai:gpt-5-mini";

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

const SYSTEM_PROMPT: &str = "You are a concise travel assistant. \
     Use the get_weather tool for weather questions. Answer in two sentences at most.";

/// Arguments of the `get_weather` tool; the schema is derived from the type.
#[derive(Debug, Deserialize, JsonSchema)]
struct WeatherArgs {
    /// City name, such as "Lisbon".
    city: String,
}

/// The result of a turn that the server stores and reports.
struct Turn {
    /// The whole conversation after this turn, to persist for the next one.
    /// Each assistant message carries the envelope of the provider call
    /// that produced it.
    messages: Vec<Message>,
    /// Tokens used by every provider call of the turn.
    usage: Usage,
}

/// What one streamed provider call produced.
struct StreamedReply {
    response: ChatResponse,
    /// Tool calls the model made, each with the tool's output (or error).
    tool_results: Vec<(ToolCall, Result<String, String>)>,
}

/// The HTTP response of one request, written the way a handler writes it:
/// the status line and headers once, then SSE events.
///
/// Until [`HttpResponse::start`] nothing has been sent, so a failure still
/// chooses the status code. After `200 OK` the status is fixed, and a
/// failure can only end the stream with an SSE `error` event. This example
/// prints the response and records it for the tests; a real handler writes
/// it to the connection.
#[derive(Debug, Default)]
struct HttpResponse {
    /// The status code, once sent.
    status: Option<u16>,
    /// The `Retry-After` header sent with an error status, in seconds.
    retry_after: Option<u64>,
    /// SSE events sent after `200 OK`, as `(event, data)`.
    events: Vec<(&'static str, String)>,
}

/// An HTTP error response for a failed turn.
#[derive(Debug)]
struct ErrorReply {
    status: u16,
    retry_after: Option<Duration>,
    message: String,
}

#[tokio::main]
async fn main() -> ExitCode {
    let message = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Do I need an umbrella in Lisbon tomorrow?".into());

    // Startup: build the provider stack once from configuration and share
    // it between requests (see `provider_stack`).
    let address = std::env::var("ALLAMA_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
    let (provider, model) =
        match allama_providers::chat::from_model_with(&address, &provider_settings()) {
            Ok(selected) => selected,
            Err(error) => {
                HttpResponse::default().fail(&error);
                return ExitCode::FAILURE;
            }
        };
    let fallback = match std::env::var("ALLAMA_FALLBACK_MODEL") {
        Ok(fallback) => match fallback_side(&fallback) {
            Ok(provider) => Some(provider),
            Err(error) => {
                HttpResponse::default().fail(&error);
                return ExitCode::FAILURE;
            }
        },
        Err(_) => None,
    };
    eprintln!("using model address={address}");
    let provider = provider_stack(provider, fallback);

    // Per request: the conversation so far, as your database holds it.
    // `allama::history::History` is a versioned envelope for it.
    let history = vec![
        Message::user("Hi! I'm travelling next week."),
        Message::assistant("Nice! Where are you heading?"),
    ];

    let (_, turn) = handle_request(&provider, &model, history, message).await;
    match turn {
        Some(turn) => {
            eprintln!(
                "{} messages to store; usage: input={:?} output={:?} total={:?}",
                turn.messages.len(),
                turn.usage.input_tokens,
                turn.usage.output_tokens,
                turn.usage.total(),
            );
            ExitCode::SUCCESS
        }
        None => {
            eprintln!("rolled back: the stored conversation stays as it was");
            ExitCode::FAILURE
        }
    }
}

/// How each provider is built: `allama_providers` wraps it in a
/// `RetryingProvider` that waits out a `Retry-After` of at most
/// [`MAX_RETRY_AFTER`], around a `TimeoutProvider` that bounds each
/// attempt, so a stalled attempt is retried. The stack below adds no
/// retries of its own.
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

/// The provider stack for an interactive server, outermost first, around
/// providers built with [`provider_settings`]:
///
/// - `TracingProvider`, so one span covers the whole call;
/// - a `TimeoutProvider` with a `stream_setup` deadline: a retrying or
///   failing-over stream only opens at its first model output, so this
///   bounds the wait for it across every attempt and backoff. Its
///   `request` bound does the same for a whole unary call. It sits inside
///   tracing, so an expired deadline is recorded as a timeout, not as a
///   cancelled call. Never `stream_total` here: it would cut off a long
///   answer after it started;
/// - a `FallbackProvider` when a fallback is configured.
fn provider_stack(primary: DynChatProvider, fallback: Option<DynChatProvider>) -> DynChatProvider {
    let routed = match fallback {
        Some(fallback) => DynChatProvider::new(FallbackProvider::new(primary, fallback)),
        None => primary,
    };
    DynChatProvider::new(TracingProvider::new(TimeoutProvider::new(
        routed,
        TimeoutPolicy::new()
            .request(UNARY_DEADLINE)
            .stream_setup(FIRST_OUTPUT_DEADLINE),
    )))
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

/// Handle one request: run the turn and write its outcome to the response.
/// Returns the conversation to store when the turn succeeded.
///
/// A failed turn returns `None`, and the server stores nothing: it **rolls
/// back** to the conversation as it was before the turn, and the client
/// re-sends the user's message on retry. That holds after the commit point
/// too, once a tool round trip has run: the round trip is lost with the
/// rest of the turn, the `error` event tells the client to try again, and
/// the tool runs again then. The other choice is to store the turn up to
/// its last complete step, as `axum_sse` does with `history::validate`.
/// Either is valid; the client must agree with the one the server makes.
async fn handle_request<P: ChatProvider>(
    provider: &P,
    model: &str,
    history: Vec<Message>,
    message: String,
) -> (HttpResponse, Option<Turn>) {
    let mut response = HttpResponse::default();
    match handle_turn(provider, model, history, message, &mut response).await {
        Ok(turn) => {
            let total = turn.usage.total().unwrap_or_default();
            response.event("done", format!("{{\"total_tokens\":{total}}}"));
            (response, Some(turn))
        }
        // Roll back: nothing of the failed turn is stored.
        Err(error) => {
            response.fail(&error);
            (response, None)
        }
    }
}

/// Run one turn: stream the answer, run at most one round of tool calls,
/// and return the conversation to store.
async fn handle_turn<P: ChatProvider>(
    provider: &P,
    model: &str,
    history: Vec<Message>,
    message: String,
    response: &mut HttpResponse,
) -> Result<Turn> {
    // The system prompt lives in the code, not in storage, so a deploy that
    // changes it applies to stored conversations too. Message builders
    // append: the stored messages, then the new one.
    let mut request = ChatRequest::new(model)
        .system(SYSTEM_PROMPT)
        .extend_messages(history)
        .user(message)
        .tools([ToolDefinition::from_schema::<WeatherArgs>("get_weather")
            .description("Look up tomorrow's weather forecast for a city.")])
        .max_output_tokens(1_000);

    let first = stream_reply(provider, &request, response).await?;
    let mut usages: Vec<Usage> = first.response.usage.iter().cloned().collect();
    request.push_message(stored_reply(&first.response)?);

    if !first.tool_results.is_empty() {
        // The tool round trip: the assistant turn with its calls is already
        // in the request; add one result per call and ask again.
        for (call, result) in &first.tool_results {
            match result {
                Ok(output) => request.push_tool_result(call, output.as_str()),
                Err(error) => request.push_tool_error(call, error.as_str()),
            }
        }
        // One round trip only: the follow-up must answer in text.
        request.tool_choice = Some(ToolChoice::Disabled);

        let second = stream_reply(provider, &request, response).await?;
        usages.extend(second.response.usage.iter().cloned());
        request.push_message(stored_reply(&second.response)?);
    }

    Ok(Turn {
        messages: request.messages,
        usage: usages.iter().sum(),
    })
}

/// Stream one provider call to the client and run tool calls as soon as
/// they are complete.
async fn stream_reply<P: ChatProvider>(
    provider: &P,
    request: &ChatRequest,
    response: &mut HttpResponse,
) -> Result<StreamedReply> {
    let stream = provider.chat_stream(request).await?;
    // The commit point: wait for the first model output before sending the
    // headers. An error before it (a rate limit or an overload reported
    // mid-stream) returns here, while the status code can still say so; the
    // buffered events are replayed, so nothing is lost.
    let stream = until_committed(Box::pin(stream)).await?;
    response.start();

    let mut stream = stream.with_collector();
    let mut tool_results = Vec::new();
    while let Some(item) = stream.next_event().await {
        // An error from here on comes after `200 OK`: the handler ends the
        // stream with an SSE `error` event (see `HttpResponse::fail`).
        let item = item?;
        match item.event {
            StreamEvent::TextDelta { text, .. } => response.event("delta", text),
            // A refusal is never answer text: the client shows it as such.
            StreamEvent::RefusalDelta { text, .. } => response.event("refusal", text),
            _ => {}
        }
        // The `BlockStop` of a tool call comes with the finished call.
        if let Some(ContentBlock::ToolCall(call)) = item.completed {
            eprintln!("\n[tool] {}({})", call.name, call.arguments);
            let result = run_tool(&call);
            tool_results.push((call, result));
        }
    }
    println!();

    let reply = stream.finish()?;
    for warning in &reply.warnings {
        eprintln!("[warning] {warning}");
    }
    Ok(StreamedReply {
        response: reply,
        tool_results,
    })
}

/// The assistant message to store for a reply: its content, which the next
/// request replays, plus the reply's envelope (id, model, serving provider,
/// usage, finish reason) in its application data for cost accounting and
/// support. Providers never see it. Every provider call goes through here,
/// so each call's usage is stored with the message it produced.
fn stored_reply(response: &ChatResponse) -> Result<Message> {
    let mut message = response.to_assistant_message();
    message.app_data_insert(TURN_ENVELOPE_KEY, &response.envelope())?;
    Ok(message)
}

/// The server-side tool. Bad arguments go back to the model as a tool
/// error so it can correct itself.
fn run_tool(call: &ToolCall) -> Result<String, String> {
    match call.name.as_str() {
        "get_weather" => {
            let args: WeatherArgs = call.parse_arguments().map_err(|error| error.to_string())?;
            Ok(format!(
                "{{\"city\":{:?},\"forecast\":\"light rain until noon, then sunny\",\"high_c\":19}}",
                args.city
            ))
        }
        other => Err(format!("unknown tool {other}")),
    }
}

impl HttpResponse {
    /// Send `200 OK` and the SSE headers, once. The status is fixed from
    /// here on.
    fn start(&mut self) {
        if self.status.is_none() {
            eprintln!("200 OK (Content-Type: text/event-stream)");
            self.status = Some(200);
        }
    }

    /// Send one SSE event.
    fn event(&mut self, event: &'static str, data: impl Into<String>) {
        let data = data.into();
        if event == "delta" {
            print!("{data}");
            let _ = std::io::stdout().flush();
        } else {
            eprintln!("event: {event}\ndata: {data}");
        }
        self.events.push((event, data));
    }

    /// Fail the request: with an error status while nothing has been sent,
    /// with an SSE `error` event after `200 OK`.
    fn fail(&mut self, error: &Error) {
        // Two levels. At info, the redacted view: kind, provider, status,
        // request id, the rejected field, and messages allama or the adapter
        // wrote, never the response body or the provider's own text, which
        // may echo the prompt. At debug, opt in to the provider's message,
        // truncated.
        // One metric label per kind; a failed failover counts as its last
        // failure, and its redacted view names the earlier ones.
        let kind = error.kind_name();
        eprintln!("[metric] turn_failures kind={kind}");
        eprintln!("[info] turn failed: {}", error.as_log());
        eprintln!(
            "[debug] turn failed: {}",
            error.as_log().with_message(DEBUG_MESSAGE_CHARS)
        );

        let reply = error_reply(error);
        if self.status.is_some() {
            self.event("error", reply.message);
            return;
        }
        self.status = Some(reply.status);
        self.retry_after = reply.retry_after.map(retry_after_seconds);
        match self.retry_after {
            Some(seconds) => eprintln!(
                "{} (Retry-After: {seconds}): {}",
                reply.status, reply.message
            ),
            None => eprintln!("{}: {}", reply.status, reply.message),
        }
    }
}

/// How much of a provider's error message the debug log keeps.
const DEBUG_MESSAGE_CHARS: usize = 200;

/// Map an error to the response a client should see, by kind.
fn error_reply(error: &Error) -> ErrorReply {
    // When every side of a fallback failed, the error is the last failure,
    // so it maps like any other.
    let reply = |status, message: &str| ErrorReply {
        status,
        retry_after: None,
        message: message.to_string(),
    };
    let retry_later = |status| ErrorReply {
        status,
        retry_after: error.retry_after(),
        message: "The assistant is busy. Try again shortly.".into(),
    };

    match error {
        // Content the user sent that the model cannot take, such as an
        // image for a text-only model: the client's request.
        Error::Unsupported { .. }
            if matches!(error.unsupported_subject(), Some(Subject::Content { .. })) =>
        {
            reply(400, "The assistant cannot read this kind of content.")
        }
        // The server is set up wrong (missing API key, unknown provider key,
        // malformed model address, a model the provider does not have, a
        // feature the server asks for that the model lacks). Not the
        // client's fault and not retryable: page someone.
        Error::Configuration { .. } | Error::ModelNotFound { .. } | Error::Unsupported { .. } => {
            reply(500, "The assistant is not configured.")
        }
        // The stored conversation outgrew the model's context window.
        Error::ContextLengthExceeded { context_window, .. } => ErrorReply {
            status: 413,
            retry_after: None,
            message: match context_window {
                Some(limit) => {
                    format!("This conversation is too long ({limit} tokens max). Start a new one.")
                }
                None => "This conversation is too long. Start a new one.".into(),
            },
        },
        Error::ContentFiltered { .. } => {
            reply(422, "The request was blocked by the content filter.")
        }
        Error::InvalidRequest { .. } => reply(400, "The request could not be processed."),
        // The client may try again, after the provider's hint when there is
        // one: 429 for a rate limit, 503 for overload, timeouts, and
        // dropped connections.
        Error::RateLimited { .. } => retry_later(429),
        _ if error.is_retryable() => retry_later(503),
        // Bad credentials, exhausted quota, and other provider failures.
        _ => reply(502, "The assistant is unavailable."),
    }
}

/// The `Retry-After` header value: whole seconds, rounded up.
fn retry_after_seconds(delay: Duration) -> u64 {
    delay.as_secs() + u64::from(delay.subsec_nanos() > 0)
}

#[cfg(test)]
mod tests {
    use allama::history::TurnEnvelope;
    use allama::mock::{MockStreamEvent, MockStreamingProvider, TestProvider};

    use super::*;

    const RETRY_AFTER: Duration = Duration::from_secs(7);

    /// Run one request through the real provider stack around `provider`,
    /// built the way `main` builds a provider: through a registry, with
    /// [`provider_settings`].
    async fn request<P>(provider: &P) -> (HttpResponse, Option<Turn>)
    where
        P: ChatProvider + Clone + Send + Sync + 'static,
    {
        let mut registry = allama_providers::Registry::new();
        let mock = provider.clone();
        registry.register_chat("mock", move || Ok(mock.clone()));
        let (provider, _) = registry
            .chat_from_model_with("mock:mock-model", &provider_settings())
            .unwrap();
        let stack = provider_stack(provider, None);
        handle_request(&stack, "mock-model", Vec::new(), "Rain in Lisbon?".into()).await
    }

    /// A refusal goes out as its own SSE event, never as a `delta`, so the
    /// client never shows it as the answer.
    #[tokio::test]
    async fn a_refusal_is_never_a_delta() {
        let provider = MockStreamingProvider::with_stream([
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

        let (response, turn) = request(&provider).await;

        assert_eq!(response.status, Some(200));
        assert!(
            response
                .events
                .contains(&("refusal", "I can't help with that.".to_owned())),
            "{:?}",
            response.events
        );
        assert!(!response.events.iter().any(|(event, _)| *event == "delta"));
        let turn = turn.expect("a refusal is a complete turn");
        assert!(turn.messages.iter().any(|message| matches!(
            message,
            Message::Assistant(reply)
                if reply.content.iter().any(|block| block.as_refusal().is_some())
        )));
    }

    #[tokio::test]
    async fn setup_errors_get_an_http_status() {
        // `chat_stream` itself fails: the provider rejected the request.
        let provider = MockStreamingProvider::build(|b| {
            b.setup_error(Error::rate_limited("slow down").with_retry_after(RETRY_AFTER))
        });

        let (response, turn) = request(&provider).await;

        assert!(turn.is_none());
        assert_eq!(response.status, Some(429));
        assert_eq!(response.retry_after, Some(7));
        assert!(response.events.is_empty());
    }

    #[tokio::test]
    async fn a_refused_credential_gets_an_http_status() {
        use allama::wrappers::MapRequestProvider;
        use allama_http::{HttpRequestOptions, ReplayTransport};

        // A per-tenant key sent as a plain header, as a tenant wrapper might
        // attach it: every provider in a fallback chain would send it, so
        // the adapter refuses it before sending, and so before the commit
        // point. It is the server's bug: 500, not retried.
        let transport = ReplayTransport::default();
        let openai = allama_openai::Provider::builder()
            .api_key("server-key")
            .http_transport(transport.clone())
            .build()
            .unwrap();
        let provider = DynChatProvider::new(MapRequestProvider::new(
            openai,
            |request: &mut ChatRequest| {
                request.options.insert(
                    HttpRequestOptions::new().header("authorization", "Bearer TENANT-SECRET"),
                );
            },
        ));

        let (response, turn) = request(&provider).await;

        assert!(turn.is_none());
        assert_eq!(response.status, Some(500));
        assert!(response.events.is_empty());
        assert!(
            transport.requests().is_empty(),
            "nothing reached the provider"
        );
    }

    /// A rate limit longer than the stack waits out reaches the client at
    /// once, as `429` with the provider's hint.
    #[tokio::test]
    async fn a_long_rate_limit_is_handed_to_the_client_at_once() {
        let hint = Duration::from_secs(30);
        let provider = MockStreamingProvider::build(|b| {
            b.setup_error(Error::rate_limited("slow down").with_retry_after(hint))
                .text("too early")
        });

        let started = std::time::Instant::now();
        let (response, turn) = request(&provider).await;

        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(turn.is_none());
        assert_eq!(response.status, Some(429));
        assert_eq!(response.retry_after, Some(30));
        assert_eq!(provider.call_count(), 1, "not retried before the hint");
    }

    #[tokio::test]
    async fn errors_before_the_first_output_get_an_http_status() {
        // The stream opens, then fails before any model output.
        let provider = MockStreamingProvider::with_stream([
            MockStreamEvent::from(StreamEvent::response_start(Some("r1".into()), None)),
            MockStreamEvent::from(Error::overloaded("busy").with_retry_after(RETRY_AFTER)),
        ]);

        let (response, turn) = request(&provider).await;

        assert!(turn.is_none());
        assert_eq!(response.status, Some(503));
        assert_eq!(response.retry_after, Some(7));
        assert!(response.events.is_empty());
    }

    #[tokio::test]
    async fn errors_after_the_first_output_end_the_stream() {
        let provider = MockStreamingProvider::with_stream([
            MockStreamEvent::from(StreamEvent::response_start(Some("r1".into()), None)),
            MockStreamEvent::from(StreamEvent::text_delta(0, "Light rain")),
            MockStreamEvent::from(Error::overloaded("busy").with_retry_after(RETRY_AFTER)),
        ]);

        let (response, turn) = request(&provider).await;

        assert!(turn.is_none());
        assert_eq!(response.status, Some(200));
        assert_eq!(response.retry_after, None);
        assert_eq!(
            response.events,
            [
                ("delta", "Light rain".to_string()),
                (
                    "error",
                    "The assistant is busy. Try again shortly.".to_string()
                ),
            ]
        );
    }

    #[tokio::test]
    async fn a_tool_round_trip_streams_one_response() {
        let provider = MockStreamingProvider::new([
            MockStreamEvent::tool_call_response(
                "c1",
                "get_weather",
                serde_json::json!({"city": "Lisbon"}),
            ),
            MockStreamEvent::text_response("Bring an umbrella until noon."),
        ]);

        let (response, turn) = request(&provider).await;

        let turn = turn.expect("the turn succeeds");
        // user, assistant tool call, tool result, assistant answer
        assert_eq!(turn.messages.len(), 4);
        assert_eq!(response.status, Some(200));
        assert!(
            response
                .events
                .contains(&("delta", "Bring an umbrella until noon.".to_string()))
        );
        assert_eq!(
            response.events.last().map(|(event, _)| *event),
            Some("done")
        );
    }

    #[tokio::test]
    async fn every_provider_call_stores_its_envelope() {
        // `TestProvider` calls the tool, then answers: two provider calls.
        let (_, turn) = request(&TestProvider::new()).await;

        let turn = turn.expect("the turn succeeds");
        let envelopes: Vec<TurnEnvelope> = turn
            .messages
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
        assert_eq!(envelopes[0].provider.as_deref(), Some("test"));
        assert_ne!(envelopes[0].id, envelopes[1].id);
        // The stored envelopes add up to the usage the turn reports.
        let stored: Usage = envelopes.iter().filter_map(|turn| turn.usage.clone()).sum();
        assert_eq!(stored.total(), turn.usage.total());
        assert!(stored.total().is_some_and(|total| total > 0));
    }

    #[tokio::test]
    async fn a_failure_after_the_tool_round_trip_rolls_back() {
        // The first reply calls the tool and commits the response; the
        // follow-up call is rejected with a hint too long to retry.
        let provider = MockStreamingProvider::build(|b| {
            b.tool_call(
                "call_1",
                "get_weather",
                serde_json::json!({ "city": "Lisbon" }),
            )
            .setup_error(Error::overloaded("busy").with_retry_after(Duration::from_secs(30)))
        });

        let (response, turn) = request(&provider).await;

        // The round trip ran, but nothing of the turn is stored: the client
        // re-sends its message and the conversation continues from before.
        assert_eq!(provider.call_count(), 2);
        assert!(turn.is_none());
        assert_eq!(response.status, Some(200));
        assert_eq!(
            response.events.last(),
            Some(&(
                "error",
                "The assistant is busy. Try again shortly.".to_string()
            ))
        );
    }

    #[test]
    fn errors_map_to_http_responses_by_kind() {
        assert_eq!(
            error_reply(&Error::configuration("OPENAI_API_KEY not set")).status,
            500
        );
        assert_eq!(
            error_reply(&Error::model_not_found("no such model")).status,
            500
        );

        let too_long = error_reply(&Error::context_length_exceeded("too long", Some(128_000)));
        assert_eq!(too_long.status, 413);
        assert!(too_long.message.contains("128000"));

        let limited = error_reply(
            &Error::rate_limited("slow down").with_retry_after(Duration::from_secs(30)),
        );
        assert_eq!(limited.status, 429);
        assert_eq!(limited.retry_after, Some(Duration::from_secs(30)));

        let overloaded = error_reply(&Error::overloaded("busy"));
        assert_eq!(overloaded.status, 503);
        assert_eq!(overloaded.retry_after, None);

        assert_eq!(error_reply(&Error::timeout("slow")).status, 503);
        assert_eq!(error_reply(&Error::auth("bad key")).status, 502);

        // `Unsupported` by what was refused: a field the server chose is a
        // setup mistake, content the user sent is their request.
        let degradations =
            allama::adapter::Degradations::new("demo", allama::UnsupportedPolicy::Warn);
        let tools = degradations.unsupported(allama::RequestField::Tools, "no tool calls");
        assert_eq!(error_reply(&tools).status, 500);
        let image = Subject::Content {
            message: 0,
            block: Some(1),
        };
        assert_eq!(
            error_reply(&degradations.reject(image, "no image input")).status,
            400
        );
    }

    #[test]
    fn a_failed_failover_maps_like_its_last_failure() {
        let rate_limited =
            Error::rate_limited("slow down").with_retry_after(Duration::from_secs(30));
        let failover = error_reply(&rate_limited.with_fallback_from(Error::overloaded("busy")));
        assert_eq!(failover.status, 429);
        assert_eq!(failover.retry_after, Some(Duration::from_secs(30)));

        let too_long = Error::context_length_exceeded("too long", Some(128_000));
        let chain = too_long
            .with_fallback_from(Error::rate_limited("slow down"))
            .with_fallback_from(Error::timeout("slow"));
        assert_eq!(error_reply(&chain).status, 413);
    }

    #[test]
    fn retry_after_rounds_up_to_whole_seconds() {
        assert_eq!(retry_after_seconds(Duration::from_secs(7)), 7);
        assert_eq!(retry_after_seconds(Duration::from_millis(6_200)), 7);
        assert_eq!(retry_after_seconds(Duration::ZERO), 0);
    }
}
