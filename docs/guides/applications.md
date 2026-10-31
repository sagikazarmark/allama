# Building servers and terminal apps with allama

This guide is for application code: a backend that stores conversations,
streams answers to a browser, runs tools, and has to say something useful
when a provider fails, or a terminal chat that does the same for one
person. It collects the patterns and the traps in one place.

Four runnable examples back it:

- [`axum_sse`](../../crates/allama-examples/examples/axum_sse.rs): a
  streaming chat endpoint with axum and server-sent events (SSE). The
  handler, error mapping and test excerpts below are quoted from it. The
  excerpts share the file's `use` block, quoted with the handler
  ([A streaming endpoint with axum](#a-streaming-endpoint-with-axum)).
- [`web_backend`](../../crates/allama-examples/examples/web_backend.rs):
  the same turn without a web framework, which prints what a handler would
  send. It makes the other choice of what to store when a turn fails
  ([When a turn fails after the commit point](#when-a-turn-fails-after-the-commit-point)).
- [`chat_repl`](../../crates/allama-examples/examples/chat_repl.rs): a
  chat in the terminal, one line per turn
  ([Chat in a terminal](#chat-in-a-terminal)). The terminal excerpts and
  the smoke call are quoted from it.
- [`tower_transport`](../../crates/allama-examples/examples/tower_transport.rs):
  a `tower` stack between the providers and the HTTP client
  ([A tower stack as the transport](#a-tower-stack-as-the-transport)),
  against a stand-in server it starts itself.

Run the axum one against a local Ollama, which needs no credentials:

```text
ALLAMA_MODEL=ollama:qwen3:1.7b cargo run -p allama-examples --example axum_sse
curl -N 127.0.0.1:3000/chat -H 'content-type: application/json' \
    -d '{"conversation_id":"demo","message":"Do I need an umbrella in Lisbon?"}'
```

[`guide_quotes.rs`](../../crates/allama-examples/tests/guide_quotes.rs)
checks that every quoted excerpt still matches its example. The other code
blocks are self-contained and run as doc tests of `allama-examples`. The
API reference is on
[docs.rs](https://docs.rs/allama); the
[crate docs' "Common tasks"](https://docs.rs/allama/latest/allama/#common-tasks)
cover a wrapper stack for providers built directly from adapter crates,
fallback, extraction costs and embeddings at scale.

## Contents

Sections marked *(server)* are about serving HTTP: status codes, SSE
events, tenants. A terminal app can skip them; the rest applies to both.
A backend that returns whole results, such as a unary API or a job
queue, starts at [Unary and job-queue backends](#unary-and-job-queue-backends).

1. [Starting a turn](#starting-a-turn)
2. [Application data and the turn envelope](#application-data-and-the-turn-envelope)
3. [Unary and job-queue backends](#unary-and-job-queue-backends)
4. [A streaming endpoint with axum](#a-streaming-endpoint-with-axum) *(server)*
5. [The provider stack](#the-provider-stack) *(server)*
6. [Reading the traces](#reading-the-traces)
7. [Chat in a terminal](#chat-in-a-terminal)
8. [The commit point](#the-commit-point) *(server)*
9. [Tool-call progress](#tool-call-progress) *(server)*
10. [Tools that are not function calls](#tools-that-are-not-function-calls)
11. [When a turn fails after the commit point](#when-a-turn-fails-after-the-commit-point) *(server)*
    - [Stopping when the client leaves](#stopping-when-the-client-leaves) *(server)*
12. [Switching providers with stored history](#switching-providers-with-stored-history)
13. [Rendering history](#rendering-history)
14. [Per-tenant API keys](#per-tenant-api-keys) *(server)*
15. [Building providers from your own configuration](#building-providers-from-your-own-configuration)
    - [A tower stack as the transport](#a-tower-stack-as-the-transport)
16. [Handling errors](#handling-errors)
17. [Capabilities, warnings and policy](#capabilities-warnings-and-policy)
    - [Checking requests at startup](#checking-requests-at-startup)
    - [A smoke call for credentials and the model id](#a-smoke-call-for-credentials-and-the-model-id)
    - [Models the provider cannot classify](#models-the-provider-cannot-classify)
    - [Strict or lenient: the unsupported-feature policy](#strict-or-lenient-the-unsupported-feature-policy)
    - [Sampling with reasoning](#sampling-with-reasoning)
    - [Warnings](#warnings)
    - [Requests the model cannot take](#requests-the-model-cannot-take)
    - [When a turn does not finish normally](#when-a-turn-does-not-finish-normally)
18. [Testing](#testing)

## Starting a turn

The system prompt is set on every request: a `History` stores the
messages only. A conversation that keeps its own prompt, such as a persona
the user picked, stores what it needs to rebuild the prompt (the text, or a
persona id) next to the history, and builds the `SystemPrompt` (cache
breakpoint, options) on every request, as `resume` does below. A
`SystemPrompt` is not versioned like a `History`, so don't store it.

```rust
use allama::history::History;
use allama::{ChatRequest, ChatResponse, ContentBlock};

const SYSTEM_PROMPT: &str = "You are a concise travel assistant.";

/// Store the request's messages and the reply.
fn save(request: &ChatRequest, response: &ChatResponse) -> serde_json::Result<String> {
    let mut history = History::from_request(request);
    history.messages.push(response.to_assistant_message());
    serde_json::to_string(&history)
}

/// Start the next turn: the stored messages, today's system prompt.
fn resume(stored: &str, model: &str, message: &str) -> serde_json::Result<ChatRequest> {
    let history: History = serde_json::from_str(stored)?;
    Ok(history.into_request(model).system(SYSTEM_PROMPT).user(message))
}

fn main() -> serde_json::Result<()> {
    let first = ChatRequest::new("demo-model").system(SYSTEM_PROMPT).user("Hi");
    let reply = ChatResponse::new(vec![ContentBlock::text("Hello! Where to?")]);
    let stored = save(&first, &reply)?;
    let request = resume(&stored, "demo-model", "Lisbon, next week")?;
    assert_eq!(request.system.len(), 1);
    assert_eq!(request.messages.len(), 3);
    Ok(())
}
```

`history.into_request(model)` is the same as
`ChatRequest::new(model).extend_messages(history.messages)`. Then append
the new user message with `.user(..)`, and set tools and sampling for this
turn. Which builders append and which replace:

- **Append:** `system`, `message`, `user`, `assistant`, `extend_messages`,
  and the in-place `push_message`, `push_tool_result`, `push_tool_error`.
- **Replace:** the collection setters `tools` and `stop`, and every scalar
  setter (`temperature`, `max_output_tokens`, `tool_choice`, `reasoning`,
  …).
- To replace the messages or the system prompts, assign the public
  `messages` or `system` field.

The messages keep provider state, such as thinking signatures and
encrypted reasoning, so a reloaded conversation continues where
it stopped, including on reasoning models.

**Load the history from your database, not from the client.** Messages,
histories and requests from an end user are untrusted. Provider data,
`Other` content and provider files reach the provider as they are, with
your key: a client that sends the whole conversation back can inject raw
provider items (an `item_reference`, a document by `file_id`) and write
the assistant's turns. Take only the new message from the client. Where
the client must hold the conversation, pass it through
`history::retain_portable_content` first, which keeps text, tool calls and
results, reasoning text and inline or URL media, and removes every escape
hatch; it cannot make the turns authentic. Never deserialize a whole
`ChatRequest` from a client: its `options` set provider headers and body
fields.

## Application data and the turn envelope

Your own data about a message, its
[application data](../../CONTEXT.md#application-data) (its id in your database, a
timestamp, UI state, per-turn usage), goes in the message's `app_data`. System prompts
have one too. Adapters never read or send it, `retain_provider_state`
never touches it, and `History` persists it.

`ChatResponse::to_assistant_message` keeps only what the next request
replays: the content. The rest of the response (id, request id, model,
[serving provider](../../CONTEXT.md#serving-provider), usage, finish reason and the provider's raw finish
reason, warnings) is what cost accounting
and support tickets need. `ChatResponse::envelope()` returns it as a
`TurnEnvelope`, the [turn envelope](../../CONTEXT.md#turn-envelope); store it under `TURN_ENVELOPE_KEY` so other tools can read
it from your histories:

```rust
use allama::history::{History, TURN_ENVELOPE_KEY, TurnEnvelope};
use allama::{ChatResponse, ContentBlock, Message, Usage};
use serde_json::json;

/// Store a finished turn, with the application's data on its messages.
fn record(history: &mut History, text: &str, response: &ChatResponse) -> allama::Result<()> {
    let user = Message::user(text).with_app_data("myapp.message_id", json!("msg_0041"));
    let mut reply = response.to_assistant_message();
    reply.app_data_insert("myapp.message_id", &"msg_0042")?;
    reply.app_data_insert(TURN_ENVELOPE_KEY, &response.envelope())?;
    history.messages.extend([user, reply]);
    Ok(())
}

/// Output tokens spent on a stored conversation.
fn output_tokens(history: &History) -> allama::Result<u64> {
    let mut total = 0;
    for message in &history.messages {
        if let Some(turn) = message.app_data_get::<TurnEnvelope>(TURN_ENVELOPE_KEY)? {
            total += turn.usage.and_then(|usage| usage.output_tokens).unwrap_or(0);
        }
    }
    Ok(total)
}

fn main() -> anyhow::Result<()> {
    let response = ChatResponse::new(vec![ContentBlock::text("Hello!")])
        .usage(Usage::new().input_tokens(12).output_tokens(3));
    let mut history = History::new(Vec::new());
    record(&mut history, "Hi", &response)?;

    // A serde_json error converts into `allama::Error`, so `?` is enough.
    let stored = serde_json::to_string(&history)?;
    let loaded: History = serde_json::from_str(&stored)?;
    assert_eq!(output_tokens(&loaded)?, 3);
    Ok(())
}
```

Prefix your keys with your application's name, as `myapp.` above; keys
starting with `allama.` are reserved for conventions allama publishes,
such as `TURN_ENVELOPE_KEY`.

**One envelope per provider call.** A turn can take more than one provider
call: a tool round trip takes two, the call that asks for the tool and the
call that answers. Each call has its own id, usage and finish reason, and
each produces its own assistant message. Store each call's envelope on its
message, and add up the messages for the turn's cost, as `output_tokens`
does above. The example passes every reply through one function:

```rust,ignore
/// The assistant message to store for a reply, with the reply's envelope
/// (id, request id, model, provider, usage) in its application data for
/// cost accounting and support. Providers never see it.
fn stored_reply(response: &ChatResponse) -> Result<Message> {
    let mut message = response.to_assistant_message();
    message.app_data_insert(TURN_ENVELOPE_KEY, &response.envelope())?;
    Ok(message)
}
```

Its `finish_turn` appends `stored_reply(&first.response)?` after the first
call and `stored_reply(&second.response)?` after the follow-up, so a turn
with a tool round trip stores two envelopes; `web_backend` does the
same. Storing one envelope for the
whole turn would lose the first call's usage, which was billed too.

The envelope holds what every response says about itself. What a wrapper
adds goes in provider data instead: to keep a `FallbackProvider`'s failed
attempts with the turn, store them beside the envelope, under a key you
own:

```rust
use allama::history::TURN_ENVELOPE_KEY;
use allama::wrappers::FallbackAttempts;
use allama::{ChatResponse, Message};

/// The model's turn, with its envelope and any failed fallback attempts.
fn reply(response: &ChatResponse) -> allama::Result<Message> {
    let mut reply = response.to_assistant_message();
    reply.app_data_insert(TURN_ENVELOPE_KEY, &response.envelope())?;
    if let Some(attempts) = response.provider_data.get::<FallbackAttempts>()? {
        reply.app_data_insert("myapp.fallback_attempts", &attempts)?;
    }
    Ok(reply)
}
```

An envelope written by a newer allama still loads in an older build: a
warning kind, request field or subject the older build does not know reads
as `WarningKind::Other`, `RequestField::Unknown` or `Subject::Unknown`.
That is read-only: if the older build stores the envelope again, the
unknown values are written as those fallbacks, so let the newest version
rewrite stored envelopes.

Do not keep application state in content. A `ContentPart::Other` or
`ContentBlock::Other` you add for your own use carries an origin no
provider shares: adapters refuse to send it (they fail the request with
`Error::Unsupported` or drop the block with a warning, depending on the
request's `UnsupportedPolicy`), and `retain_provider_state` removes it
when the conversation moves to another provider. See
[Application data](https://docs.rs/allama/latest/allama/history/index.html#application-data)
and
[The turn envelope](https://docs.rs/allama/latest/allama/history/index.html#the-turn-envelope)
in `allama::history`.

## Unary and job-queue backends

A backend that answers a request with one complete result, or a worker
that takes jobs off a queue, needs none of the streaming machinery below:
no commit point, no SSE, nothing to store after a partial answer. It calls
`chat`, or `extract` for typed output, and decides what to do with a
failure: store the result, retry the job later, set the job aside for a
person, or stop the worker. This section covers that path; the sections
marked *(server)* after it are about streaming to a browser.

**The stack.** Providers from `allama_providers` retry by default, and
the default policy waits out a `Retry-After` of up to 120 s, which suits
a job that would rather wait than fail. Bound each call with a deadline,
`TimeoutPolicy::request`, around the retries and their backoff, inside
tracing. A job that hits it fails with a retryable `Error::Timeout`, and
the queue can try it again later:

```rust
use std::time::Duration;

use allama::DynChatProvider;
use allama::wrappers::{TimeoutPolicy, TimeoutProvider, TracingProvider};

/// A provider for the worker: the lookup's retries, a deadline on each
/// call across every attempt and backoff, and tracing around both.
fn worker_provider(address: &str) -> allama::Result<(DynChatProvider, String)> {
    let (provider, model) = allama_providers::chat::from_model(address)?;
    let deadline = TimeoutPolicy::new().request(Duration::from_secs(300));
    let provider =
        DynChatProvider::new(TracingProvider::new(TimeoutProvider::new(provider, deadline)));
    Ok((provider, model))
}

fn main() {
    // Without the key the lookup fails, as a missing key should.
    let _ = worker_provider("openai:gpt-5-mini");
}
```

To requeue a long rate limit instead of holding the worker, lower the
retry layer's ceiling (`RetryPolicy::max_retry_after` in
`ProviderSettings`): a longer `Retry-After` then comes back at once, and
`retry_after()` says when to try again.

**Classifying a failure.** Each error kind says whose problem it is, and
so what the queue does with the job. A failover that failed on every
side returns its last failure as itself, so it classifies like any other
error; read an extraction failure's reason from `Error::Extract { failure }`:

- **Retry later**: a transient failure the retries gave up on, or the
  deadline (`is_retryable()`), after `retry_after()` when the provider
  sent one.
- **Set the job aside**: the input is the problem, so it fails the same
  way every time: too long for the model, blocked by a content filter,
  content the model cannot read, declined by the model, or output that
  does not fit the type. Keep what a person needs to look at it: the
  model's raw output (`ChatExtractError::raw_output()`) and the response
  (`ChatExtractFailure::response()`).
- **Stop the worker**: a setup mistake fails every job the same way, so
  stop taking jobs and alert: a rejected key, a missing credential, a
  model the provider does not have, a feature the model lacks, an
  exhausted quota, or an output cap too small for the answer (reasoning
  counts against it; see [Sampling with reasoning](#sampling-with-reasoning)).

```rust
use std::time::Duration;

use allama::chat::extract::{ChatExtractError, ChatExtractExt};
use allama::mock::MockProvider;
use allama::schemars::{self, JsonSchema};
use allama::{ChatProvider, ChatRequest, Error, Subject};
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct Invoice {
    vendor: String,
    total_cents: u64,
}

/// What the queue does with a job that failed.
#[derive(Debug)]
enum JobFailure {
    /// Put the job back, after `after` when the provider said when.
    Retry { after: Option<Duration> },
    /// The input is the problem: set the job aside for a person, with what
    /// they need to look at it.
    SetAside { reason: String, raw_output: Option<String> },
    /// A setup mistake: every job would fail the same way. Stop and alert.
    Stop { reason: String },
}

async fn run_job(
    provider: &impl ChatProvider,
    model: &str,
    document: &str,
) -> Result<Invoice, JobFailure> {
    let request = ChatRequest::new(model)
        .system("Extract the vendor and the total, in cents, from the invoice.")
        .user(document);
    match provider.extract::<Invoice>(&request).await {
        Ok(extracted) => Ok(extracted.value),
        Err(error) => Err(classify(&error)),
    }
}

fn classify(error: &Error) -> JobFailure {
    let set_aside = |reason: &str| JobFailure::SetAside {
        reason: reason.to_owned(),
        raw_output: None,
    };
    let stop = |reason: &str| JobFailure::Stop {
        reason: reason.to_owned(),
    };
    match error {
        Error::Extract { failure, .. } => match failure.error() {
            // Asking again pays for the same answer.
            ChatExtractError::Refused { .. } => set_aside("the model declined"),
            // Every long enough invoice would be cut off.
            ChatExtractError::Truncated { .. } => stop("max_output_tokens is too small"),
            // Output that does not fit `Invoice`: keep it for review.
            other => JobFailure::SetAside {
                reason: other.to_string(),
                raw_output: other.raw_output().map(str::to_owned),
            },
        },
        Error::Auth { .. }
        | Error::Configuration { .. }
        | Error::ModelNotFound { .. }
        | Error::QuotaExhausted { .. } => stop(error.kind_name()),
        Error::ContextLengthExceeded { .. } | Error::ContentFiltered { .. } => {
            set_aside(error.kind_name())
        }
        // Content the model cannot read is the input's; any other refused
        // feature is the worker's.
        Error::Unsupported { .. } => match error.unsupported_subject() {
            Some(Subject::Content { .. }) => set_aside("content the model cannot read"),
            _ => stop("a feature the model lacks"),
        },
        _ if error.is_retryable() => JobFailure::Retry {
            after: error.retry_after(),
        },
        // An invalid request or an unexpected response: look before
        // trying again.
        _ => set_aside(error.kind_name()),
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let invoice = "ACME Ltd. Total due: $12.50";
    let provider = MockProvider::build(|b| {
        b.text(r#"{"vendor": "ACME Ltd.", "total_cents": 1250}"#)
            .error(
                Error::rate_limited("HTTP 429").with_retry_after(Duration::from_secs(30)),
            )
            .text("I could not find a total.")
            .error(Error::auth("HTTP 401"))
    });

    let done = run_job(&provider, "m", invoice).await.unwrap();
    assert_eq!(done.total_cents, 1250);
    assert!(matches!(
        run_job(&provider, "m", invoice).await,
        Err(JobFailure::Retry { after: Some(_) })
    ));
    assert!(matches!(
        run_job(&provider, "m", invoice).await,
        Err(JobFailure::SetAside { raw_output: Some(_), .. })
    ));
    assert!(matches!(
        run_job(&provider, "m", invoice).await,
        Err(JobFailure::Stop { .. })
    ));
}
```

`extract` makes one provider call; an `ChatExtractor` sends a parse failure
back to the model and asks again, which can save a job from being set
aside (see
[`allama::chat::extract`, "Recovery is opt-in"](https://docs.rs/allama/latest/allama/chat-extract/index.html#recovery-is-opt-in)).
Run the [smoke call](#a-smoke-call-for-credentials-and-the-model-id) when
the worker starts, so a setup mistake stops it before it takes a job, and
bound how many jobs call one provider at once, since every job past its
rate limit waits in the retries. The
[`provider_extract`](../../crates/allama-examples/examples/provider_extract.rs)
example runs one extraction against a real provider.

## A streaming endpoint with axum

The excerpts in this guide use the example's imports. Most names come from
`allama::prelude::*`, which includes `StreamBlockType`:

```rust,ignore
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
```

One `POST /chat` request is one turn. The handler builds the request with
the system prompt from the code and the stored messages, starts the turn in
its own task, waits for the turn to reach the
[commit point](../../CONTEXT.md#commit-point) ([The commit point](#the-commit-point)),
then answers with an SSE stream:

```rust,ignore
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
```

**Spawn first, then wait for the commit point.** The task makes the
provider call, and a oneshot hands the handler what happened before the
commit point: the first output, or the error that still chooses the status
code. From then on a channel carries the SSE events:

```rust,ignore
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
```

The turn is independent of the connection from its start: if the user
closes the tab, even before the first output, axum drops the handler, the
sends fail quietly, the turn still finishes, and the conversation is
stored. That window is long: the wait for the commit point includes the
model's whole reasoning phase, since a model that does not stream its
reasoning sends no output until it has finished thinking. On a short
question it took 8.3 s on `gpt-5-mini`, 16.9 s on `gemini-3.5-flash` and
0.6 s on `claude-haiku-4-5`. A turn that ran in the handler until the
commit point would be lost, with the user's message, by a client that
gave up in it. The task also leaves room for a tool round trip, which needs
a second provider call in the middle of the response. To stop the turn when
the client leaves instead, see
[Stopping when the client leaves](#stopping-when-the-client-leaves).

The alternative is to map the provider stream straight into SSE events
(`stream.map(|event| ...)`) and hand that to `Sse::new`. It is shorter and
needs no task, but axum drops the stream when the client disconnects,
which cancels the provider call and anything you meant to do after it.
It fits a single call whose result you do not store.

The provider is built once at startup and shared: `DynChatProvider` is
cheap to clone and `Send + Sync`. Build it from a
[model address](../../CONTEXT.md#model-address) (`allama_providers::chat::from_model_with`, with the settings of the next
section) and wrap it in the stack of the next section.

## The provider stack

An interactive server wraps its provider in tracing, a deadline, fallback,
retries and per-attempt timeouts (see
[attempt timeout vs deadline](../../CONTEXT.md#attempt-timeout-vs-deadline)). Providers
from `allama_providers` retry by default, so the retries and per-attempt
timeouts come from the [provider settings](../../CONTEXT.md#provider-settings)
(`ProviderSettings`) each side is built with, and the stack adds none of its
own: a `RetryingProvider` around them would nest retries. The settings:

```rust,ignore
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
```

The example builds the rest of the stack in one function, which `main` and
the handler tests both call, so the tests run on the real stack:

```rust,ignore
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
```

Outermost first, `Tracing(Observe(Timeout(Fallback(Retry(Timeout(provider)), ..))))`:

- **Tracing** (`TracingProvider`, feature `tracing`): one span covers the
  call, with every retry and failover in its summary.
- **A failover counter** (`ObserveProvider`): every warning of every call
  reaches `count_failovers`, which counts the failover warnings for a
  metric (see [Warnings](#warnings)).
- **A deadline on the first output.** `RetryingProvider` and
  `FallbackProvider` return a stream only once it has its first model
  output, so `stream_setup` on the outer `TimeoutProvider` bounds the wait
  for that output across every attempt, backoff and failover. That is the
  deadline a handler needs before it sends a status: when it expires,
  `chat_stream` fails with `Error::Timeout` and the handler answers `503`.
  Its `request` bound does the same for a unary call (`chat`), as a whole.
  Never set `stream_total` here: it bounds the whole stream, so it would
  cut off a long answer after it started.
- **The deadline sits inside tracing.** Outside it, an expiry drops the
  traced call, so the span records a cancellation instead of
  `error.type = timeout`, and the timeout is missing from error rates.
- **Fallback**, when `ALLAMA_FALLBACK_MODEL` names a second model address.
  `should_fallback` decides which errors fail over: outages, overload,
  rate limits and exhausted quotas, timeouts, requests the primary cannot
  serve, and prompts over its context window (windows are not exposed
  uniformly, so the backup is tried at the cost of one more call). It does
  not fail over on caller mistakes, setup mistakes, an unknown model, a
  request the client closed (`499`), rejected credentials, or a content
  filter: sending a blocked prompt to another vendor is a policy decision,
  and a bad key is the operator's to fix. Opt in with `fallback_if`; its
  rustdoc has the recipes for content filters, and for credentials when
  each vendor has its own key. A [refusal](../../CONTEXT.md#refusal) or a filtered
  *response* is an
  `Ok`, so fallback never acts on it (see
  [When a turn does not finish normally](#when-a-turn-does-not-finish-normally)).
  A local policy guard, such as a per-tenant budget check, refuses with
  `Error::rejected(..)` (kind `rejected`): not retryable, never failed
  over, and blaming no provider, so it stays out of upstream error and
  quota dashboards. Put the guard outside the `FallbackProvider`, so a
  refused request reaches neither side, and attach a typed cause with
  `with_source` if callers need one.
- **Per provider, retries around a per-attempt timeout**, from
  `ProviderSettings::retry` and `ProviderSettings::attempt_timeout`. The
  inner timeout bounds each attempt, unary or streamed, so a stalled
  attempt is retried.
  `max_retry_after` is how long a provider's `Retry-After` may hold the
  request. Its default, 120 s, suits batch work that would rather wait
  than fail: with it, a side waits out a `Retry-After` of up to 120 s
  before failing over. An interactive server sets a few seconds, so a
  longer rate limit fails over, or reaches the client as `429` with
  `Retry-After`, at once. Without the deadline above, a call can take
  minutes; "Worst-case latency" in the `ProviderSettings` rustdoc has
  every timeout layer and its default.
- **Size the attempt bounds and the deadline together.** A hung primary
  uses up every attempt, plus the backoff, before the fallback runs. If
  that does not end well inside the deadline, the deadline expires first
  and the call never fails over. Here three unary attempts of 30 s take
  about 91 s of the 120 s deadline, and three 15 s waits for a stream's
  headers about 46 s of 60 s. Without a `request` bound on the attempts, a
  unary attempt to a hung server runs until the transport's 120 s timeout
  (forever on `ollama` and `vllm`), so the deadline expires during the
  first attempt.
- **The unary attempt bound is a tradeoff for reasoning models.** It
  bounds the whole generation, reasoning included, so it cuts off a slow
  answer as well as a hung server: a call that takes longer times out on
  every attempt, each of which may be billed, then fails over. Set it
  above the longest unary call you expect, and raise the deadline with it
  to attempts × bound + backoff + room for the fallback, or lower
  `max_attempts`.
- **`409 Conflict` is not retried**, unlike in the official OpenAI and
  Anthropic SDKs, unless the provider sends `x-should-retry: true`. To
  retry it anyway, add it to the classifier:
  `RetryPolicy::new().retry_if(|e: &Error| e.server_retry_directive().unwrap_or(e.is_retryable() || e.status() == Some(409)))`.
- **`stream_total` applies per attempt.** The stack sets none. Set on the
  inner timeout, it bounds one attempt's stream from that attempt's own
  `chat_stream` call, and every retry or failover starts a new clock, so
  it caps a runaway generation but never the request as a whole. The
  request's bound is the outer timeout's setup bound.

The values:

```rust,ignore
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
```

## Reading the traces

`TracingProvider` records one span per call, with a summary of every
retry and failover inside it. The
[observability guide](observability/README.md) covers wiring it to
OpenTelemetry, and how to read the spans: which provider served a call
after a failover (`allama.response.provider`, not
`gen_ai.provider.name`), and how often a rate limit reached the client.

## Chat in a terminal

A terminal chat runs the same turn for one person: read a line, stream
the answer, run the tools the model calls, and keep the conversation in
memory. The
[`chat_repl`](../../crates/allama-examples/examples/chat_repl.rs) example
is a whole one, with one local tool (`get_time`), and a local Ollama runs
it without credentials:

```text
ALLAMA_MODEL=ollama:qwen3:1.7b cargo run -p allama-examples --example chat_repl
```

The excerpts below are quoted from it, with `allama::prelude::*` in
scope. There is no status code to choose, so a terminal app needs no
[commit point](#the-commit-point): print the deltas as they arrive.
Answer text goes to stdout and everything else (reasoning, refusals, tool
progress, warnings, usage) to stderr, so the answer can be piped.

**Which turns to keep.** A turn adds everything it produced to the
history (the user's line, each tool round trip, the answer), or nothing:

- a turn that ends with an answer is kept: a reply that stopped or
  refused, or one cut off at the token limit (`Length`) without tool
  calls;
- a turn that fails partway through a stream prints `[interrupted]` and
  is rolled back whole, the user's line included, so the user can send it
  again. `finish_partial()` gives what arrived, for display only (a server
  can [keep it](#when-a-turn-fails-after-the-commit-point) instead);
- a reply a content filter stopped (`ContentFilter`) rolls the turn back
  with the line that caused it, which would trip the filter again on
  every turn;
- a reply cut off at the token limit with tool calls rolls it back too,
  since the calls' arguments may be cut off;
- so does a turn that reaches the step cap: stored, it would end on tool
  results with no answer.

The turn runs on a copy of the history, inside the request, and the
history gains the new messages only when the loop returns `Ok`:

```rust,ignore
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
```

`storable` turns the replies the history cannot take into errors, so
every way out of the loop but an answer rolls the turn back:

```rust,ignore
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
```

A tool that fails, or arguments that do not parse, go back to the model
with `push_tool_error`, so it can correct itself; the turn goes on. The
[`tool_loop`](../../crates/allama-examples/examples/tool_loop.rs) example
is the same loop without streaming, and extracts the answer as data after
it.

**Warnings.** Show the stream's `StreamEvent::Warning` events, each
recurring one once with `SeenWarnings` (see [Warnings](#warnings)), and
install no `tracing` subscriber that prints allama's warning events to
the same terminal, or filter them out with `allama::warning=off`:

```rust,ignore
                StreamEvent::Warning { warning, .. }
                    if self.seen.insert(&request.model, &warning) =>
                {
                    eprintln!("[warning] {warning}");
                }
```

The model is part of the key, so a switch to another model shows its
warnings anyway; `chat_repl` also `clear()`s the seen warnings on a
switch, so going back to an earlier model shows its warnings again too.

**Switching models.** `/model <provider:model>` runs the
[smoke call](#a-smoke-call-for-credentials-and-the-model-id) first, and
a rejected key, an unknown model or a missing credential keeps the
current model, so a typo does not break the next turn. The history stays
as it is, with the previous provider's replay state: a switch in a REPL
is not a move for good
([Switching providers with stored history](#switching-providers-with-stored-history)).

**Settings from configuration.** A setting your configuration may leave
unset goes in the request's public `Option` field, so unset keeps the
provider default, as `send` does above. The builders take a value, not an
`Option`. Parse the reasoning effort with `str::parse`, which reads the
names `Display` writes (`none`, `minimal`, `low`, `medium`, `high`,
`xhigh`):

```rust,ignore
/// An optional setting from the environment; unset or empty is `None`.
fn setting<T: FromStr>(name: &str) -> Result<Option<T>, String>
where
    T::Err: std::fmt::Display,
{
    let parse = |value: String| value.parse().map_err(|error| format!("{name}: {error}"));
    env(name).map(parse).transpose()
}
```

`Usage` prints as one line for people, such as `input 34, output 87,
total 121`, and adds up with `+=` for a turn and a session total.

## The commit point

Until the first byte of the response goes out, the handler can still
choose its status code. After `200 OK`, it can only end the stream. A
provider can fail at either point: it can reject the request (`chat_stream`
returns `Err`), or open the stream and then report a rate limit or an
overload as the first event.

`allama::chat::stream::until_committed` reads the stream until its first model
output and returns a stream that replays the buffered events, so nothing
is lost and time to first token is unchanged. An error before that point
comes back as the stream's `allama::Error`, so `?` passes it on:

```rust,ignore
/// Start the provider call and wait for its first model output.
async fn open(provider: &DynChatProvider, request: &ChatRequest) -> Result<Prefixed<ChatStream>> {
    let stream = provider.chat_stream(request).await?;
    until_committed(stream).await
}
```

`until_committed` takes the stream by value and hands it back inside
`Prefixed`, to be polled again, so the stream must be `Unpin`.
`DynChatProvider::chat_stream` returns a `ChatStream`, which is already a
pinned box, so `open` passes it as it is. A generic provider's `P::Stream`
need not be `Unpin`: pin it first with `until_committed(Box::pin(stream))`,
as `web_backend` does.

Errors before the commit point become an HTTP response with a status, and
`Retry-After` when the provider sent a hint:

```rust,ignore
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
```

Errors after it become an SSE `error` event carrying the same user-facing
message; [When a turn fails after the commit point](#when-a-turn-fails-after-the-commit-point)
shows that branch.

## Tool-call progress

A tool round trip takes seconds: the model streams the call's arguments,
the server runs the tool, and a second call writes the answer. Tell the
client what is happening with events of their own, so it can show
"Checking the weather…" instead of a stalled cursor.

`with_collector` pairs every stream event with the content block it
completed, if any (a `CollectedEvent`: `item.event` and `item.completed`). The `BlockStart` of a tool call carries its id and
name before the arguments arrive; the completed `ToolCall` block carries
the parsed call:

```rust,ignore
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
```

After running each tool, the example sends a `tool_result` event with the
call id, the tool name and whether it failed. Do not send tool arguments
or results to the browser unless the user may see them; they can carry
data the model fetched for the server.

The round trip itself is plain request building, with no loop in allama:
append the assistant turn, then `push_tool_result` (or `push_tool_error`)
for each call, then call the provider again. `finish_turn` in the example
does it once and sets `ToolChoice::Disabled` on the follow-up so the
model answers in text.

## Tools that are not function calls

Some provider tools are run by your code but are not function calls. The
OpenAI Responses API has custom tools, which take freeform text (a SQL
query, a patch) constrained by a grammar, plus a shell tool and computer
use. Their calls and answers have shapes no other provider shares, so
allama keeps them provider-specific
([ADR 0014](../decisions/0014-client-executed-non-function-tools.md)):

- they are declared with `ResponsesRequestOptions::client_tools`;
- their calls arrive as `ContentBlock::Other`. `tool_calls()` does not
  return them, but the finish reason is `ToolCalls` while one waits;
- `allama_openai::ClientToolCall::pending` lists the calls of a turn that
  wait for an answer, and each typed view builds its answer;
- the answers go in one user message after the tool messages that answer
  the function calls;
- `history::validate` checks only function calls, so it does not report an
  unanswered custom or shell call. OpenAI rejects the request instead.

Chat Completions has custom tools too, declared in its own shape with
`ChatRequestOptions::client_tools` (`{"type": "custom", "custom": {…}}`).
Their calls and answers have the same shape on both OpenAI APIs, so the
same loop serves both, and a conversation moves between them: Chat
Completions sends the answer as a tool message. Each API reads only its
own options, so give the definitions in both when one fails over to the
other.

On failover to any other provider, do nothing here either. The other
provider drops the custom or shell call and its answer with `Dropped`
warnings, and a reply the drops leave last goes with them, so the request
succeeds under every policy. The same happens on Chat Completions to the
calls it cannot express (shell, computer use). An answer whose call is not
in the request is sent only by the Responses API, where server-side
continuation (`previous_response_id`) holds the call; Chat Completions
drops it with a warning.

A loop that does not offer these tools needs nothing. A loop that offers
them answers both kinds after each turn and refuses kinds it cannot run,
so an unanswered call never goes back to the provider:

```rust
use allama::{ChatRequest, ChatResponse, ContentBlock, Error, Message, OtherContent, Result};
use allama_openai::{ClientToolCall, ShellCommandOutput};
use serde_json::json;

/// Append the model's turn and an answer to every call in it. Returns
/// `false` when the model asked for nothing and the turn is the answer.
fn answer_calls(request: &mut ChatRequest, response: &ChatResponse) -> Result<bool> {
    let pending = ClientToolCall::pending(&response.content);
    if pending.is_empty() && response.tool_calls().next().is_none() {
        return Ok(false);
    }
    request.push_message(response.to_assistant_message());
    // Function calls: tool messages right after the turn, as usual.
    for call in response.tool_calls() {
        request.push_tool_error(call, format!("unknown tool `{}`", call.name));
    }
    // Client-executed calls: typed answers in one user message.
    let mut answers = Vec::new();
    for call in pending {
        answers.push(match call {
            ClientToolCall::Custom(call) if call.name == "sql" => call.output(run_sql(&call.input)),
            ClientToolCall::Shell(call) => {
                call.output(call.commands.iter().map(|command| run_sandboxed(command)))
            }
            other => {
                return Err(Error::unsupported(format!(
                    "no handler for client tool call {:?}",
                    other.call_id()
                )));
            }
        });
    }
    if !answers.is_empty() {
        request.push_message(Message::user(answers));
    }
    Ok(true)
}

fn run_sql(query: &str) -> String {
    format!("ran `{query}`: 42 rows")
}

fn run_sandboxed(_command: &str) -> ShellCommandOutput {
    ShellCommandOutput::exited("ok\n", "", 0)
}

fn main() -> anyhow::Result<()> {
    // A turn as the Responses provider decodes it: one custom tool call.
    let call = OtherContent::new(
        "custom_tool_call",
        serde_json::from_value(json!({
            "id": "ctc_1", "call_id": "call_1", "name": "sql",
            "input": "SELECT count(*) FROM users", "status": "completed",
        }))?,
    )
    .with_origin("openai");
    let response = ChatResponse::new(vec![ContentBlock::Other(call)]);

    let mut request = ChatRequest::new("gpt-5.1").user("How many users?");
    assert!(answer_calls(&mut request, &response)?);
    // The user message, the model's turn, and the answer.
    assert_eq!(request.messages.len(), 3);
    Ok(())
}
```

Where a provider expresses its own tools as function calls (Anthropic's
bash and computer-use tools, Gemini's computer use), they are ordinary
`ToolCall`s: run them and answer with tool messages.

## When a turn fails after the commit point

The status is already `200 OK`, so the client learns about the failure from
an `error` event. What remains is what to store. Take the turn from the
example: the first call asks for `get_weather`, the server runs it and
appends the result, and the second call fails with an overload.

Store the turn up to its last complete step:

- keep the user message, so the conversation shows what was asked and a
  retry continues from it;
- keep every finished tool round trip (the assistant message with its
  calls and one result per call), so the tool does not run again;
- drop an assistant message whose tool calls have no results, the
  *dangling tool call* (a [pending tool call](../../CONTEXT.md#pending-tool-call) that
  will never get its results), with any partial results after it. Providers
  reject a history like that with a 400 on the next turn. It happens when
  a tool cannot run at all, or the task stops between the model's turn and
  the results.

`allama::history::validate` checks the shape. `ValidationMode::Resumable`
accepts calls still waiting for results at the end of the history and
rejects anything else; `ValidationMode::Sendable` also rejects the waiting
calls, and reports which message they belong to:

```rust,ignore
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
```

The spawned task sends the event, then stores the checkpoint:

```rust,ignore
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
```

`validate` checks only function tool calls (`ToolCall`s): it reads no
provider-specific content. An application that offers client-executed
provider tools (OpenAI's custom tools, or a local shell) also checks the
last assistant message with the provider crate: when
`allama_openai::ClientToolCall::pending` returns calls for it and the next
user message holds no answer for them, that message is dangling too, and
the checkpoint truncates it the same way.

An application that resumes interrupted turns by running the pending tool
calls (a job queue, say) stores the `Resumable` history as it is instead,
and runs the calls before the next send. Either way, validate before
sending with `Sendable`. The
[tool calls and results](https://docs.rs/allama/latest/allama/history/index.html#tool-calls-and-results)
section of `allama::history` shows both: validate before `chat`, and
`Resumable` after a failed turn.

Rolling back to the conversation as it was before the turn is also valid,
when the client re-sends the user's message on retry. `web_backend` makes
that choice: on any failure it stores nothing.

**Keeping what the user saw** is a third choice. The checkpoint drops the
reply in progress, though the user has read part of it. To keep that
part, store the text that arrived as an assistant message after the
checkpoint, flagged in its
[application data](#application-data-and-the-turn-envelope) so a
transcript can mark it as cut short. Take the text from
`finish_partial()` where the stream fails, as `chat_repl` does to show how
much arrived, and nothing else from it:

- incomplete tool calls are already gone (`finish_partial()` drops them),
  and a complete one would have no result;
- partial reasoning may lack the signature its provider needs to replay
  it.

```rust
use allama::history::{self, HistoryIssue, ValidationMode};
use allama::{ChatResponse, ContentBlock, Message};

/// The application data key that marks a reply the stream did not finish.
const PARTIAL_KEY: &str = "myapp.partial";

/// What to store when the stream failed partway: the checkpoint, then the
/// text the user saw, flagged as partial.
fn keep_what_was_seen(
    mut messages: Vec<Message>,
    partial: &ChatResponse,
) -> Result<Vec<Message>, Vec<HistoryIssue>> {
    // The text only: no tool calls, no reasoning.
    let text = partial.text_or_empty();
    if !text.is_empty() {
        messages.push(Message::assistant(text).with_app_data(PARTIAL_KEY, true));
    }
    // The next turn sends it as it is.
    history::validate(&messages, ValidationMode::Sendable)?;
    Ok(messages)
}

fn main() {
    let checkpoint = vec![Message::user("Do I need an umbrella in Lisbon?")];
    // What `finish_partial()` collected before the provider failed.
    let partial = ChatResponse::new(vec![
        ContentBlock::reasoning("The user asks about rain"),
        ContentBlock::text("Light rain is likely, so"),
    ]);

    let stored = keep_what_was_seen(checkpoint, &partial).unwrap();
    assert_eq!(stored[1].text().as_deref(), Some("Light rain is likely, so"));
    assert_eq!(stored[1].app_data()[PARTIAL_KEY], true);
}
```

Replayed, the message tells the model that it said that text, which suits
a "continue" prompt. Its usage is undercounted: a stream that failed has
no envelope (below). This is a policy for the application to choose, not
a rule of the library.

Choose one, and make the client agree: a client that re-sends its message
to a server that kept it asks the question twice.

**Stored envelopes undercount a failed turn.** Billing from stored
envelopes counts only the calls whose messages are stored. Truncating a
dangling assistant message drops its envelope, the usage of a call that
was billed. A call that failed mid-stream never produced a response, so it
has no envelope at all, though the provider may bill the tokens it
streamed. Rolling back drops the envelopes of every call in the turn. When
accounting must be complete, record each envelope in a usage ledger as its
call finishes, as well as in the history.

### Stopping when the client leaves

The example lets a turn finish after the client leaves and stores it, so
a client that reconnects finds the answer. To stop paying for an answer
nobody will read, stop the turn when the client disconnects instead. The
handler, and after it the SSE response, owns the channel's receiver, so a
disconnect closes the channel at any point of the turn, and the task can
wait for that alongside it:

```rust,no_run
use std::future::Future;

use tokio::sync::mpsc;

/// Run a step of the turn until it ends or the client disconnects.
/// `None` when the client left: `turn` was dropped, which cancels the
/// provider call in flight.
async fn unless_closed<T, E>(turn: impl Future<Output = T>, events: &mpsc::Sender<E>) -> Option<T> {
    tokio::select! {
        output = turn => Some(output),
        () = events.closed() => None,
    }
}
```

In `run_turn`, wrap the `open` call and the `finish_turn` call in it, and
handle `None` as a third outcome:

- **Before the commit point, store nothing**, as for an error there.
- **After it, store the checkpoint**, as for a failure: the turn up to
  its last complete step. The reply in progress is dropped with the turn; to
  [keep what the user saw](#when-a-turn-fails-after-the-commit-point),
  record the text as it is sent.
- **Send no event.** Nobody receives a `done` or an `error` event.
- **Expect a cancelled span, and a bill.** `TracingProvider` records the
  call as cancelled (`allama.response.cancelled`), not as an error. The
  provider may bill the tokens it streamed before the call was dropped,
  and no response, so no envelope, records them.

## Switching providers with stored history

Opaque provider state, such as Anthropic thinking signatures or OpenAI
encrypted reasoning, is stored in the messages under the namespace of the
provider that produced it, and adapters only replay their own. A
conversation can therefore move between providers.

On failover, do nothing. When a `FallbackProvider` serves a turn from its
second provider, that provider's adapter drops the foreign state for that
request only, with a `Dropped` warning, and never touches the history.
Store the history as it is: the next turn may go back to the first
provider, which still finds its own state.

When a conversation moves to another provider for good, remove the old
provider's state once with `retain_provider_state_for`, passing the new
provider, and store the result. Otherwise the new provider's adapter drops
the same foreign state, with a `Dropped` warning, on every request. Those
warnings are coded (`dropped.foreign_reasoning`,
`dropped.foreign_provider_data`, `dropped.foreign_content`), so their
events go out at `WARN` once per model rather than once per turn, and
their messages name `retain_provider_state_for`. A tool result marked as
an error is not replay state, but it recurs the same way on an endpoint
with no tool error flag (OpenAI-compatible Chat Completions and Responses,
Cloudflare Workers): the error text is sent as the result, with an
`Approximated` warning coded `approximated.tool_result_error`, one per
request listing every such message. No policy refuses it, so a stored
history with a tool failure replays under `UnsupportedPolicy::Error` too:
the model reads the error text either way. Keep `push_tool_error` in
portable code, since Anthropic and Gemini carry the failure natively (see
its documentation). To send your own encoding to such an endpoint (as
`{"error": "…"}`, say), send it with `push_tool_result`; the warning's
message names that remedy. Other warnings a stored history repeats are coded too, one
per request: tool call arguments that are not a JSON object, sent as `{}`
(`approximated.tool_call_arguments`), content an endpoint cannot take
back from its own namespace (`dropped.content_replay`), and, on Gemini, a
function call sent with Google's placeholder thought signature
(`approximated.thought_signature`) and a tool result whose media was sent
after its text (`approximated.tool_result_order`).

The provider answers which namespaces it reads
(`ProviderIdentity::reads_namespace`), so pass the provider the
application built from configuration, wrappers and all. Its namespace need
not be its name: Gemini reports `gemini` and reads `google-ai`, an
OpenAI-compatible preset such as `ollama` reads `openai-compat:ollama`, and
both OpenAI endpoints read `openai`. A `FallbackProvider` keeps the state
of either side. The free function takes the messages, for an application
that stores a `Vec<Message>` rather than a `History` (whose `messages`
field is one):

```rust
use allama::history::{self, History};
use allama::{DynChatProvider, Message, ProviderIdentity, Warning};

/// Move a stored conversation to the provider the application now uses.
fn move_history(history: &mut History, provider: &DynChatProvider) -> Vec<Warning> {
    // Keep only state the provider reads; portable content is never removed.
    history.retain_provider_state_for(provider)
}

/// The same for a conversation stored as plain messages.
fn move_messages(messages: &mut [Message], provider: &DynChatProvider) -> Vec<Warning> {
    history::retain_provider_state_for(messages, provider)
}

fn main() -> anyhow::Result<()> {
    let (provider, _model) = allama_providers::chat::from_model("ollama:qwen3:1.7b")?;
    assert_eq!(provider.provider_name(), "ollama");
    assert!(provider.reads_namespace("openai-compat:ollama"));

    let mut history = History::new(vec![Message::user("Hi")]);
    for warning in move_history(&mut history, &provider) {
        eprintln!("removed: {warning}");
    }

    let mut messages = vec![Message::user("Hi")];
    assert!(move_messages(&mut messages, &provider).is_empty());
    Ok(())
}
```

It removes provider data in other namespaces, reasoning that only other
providers could replay, and `Other` content and provider tool blocks from
other origins. Reasoning with no provider data stays on purpose: it is
portable text, which endpoints that take reasoning back as text replay. An
adapter that cannot take it drops it for that request only, with a coded
`dropped.reasoning_replay` warning. It never removes a message, so application data, such as a
stored turn envelope, stays with its message, and warning indices point at
the messages you passed. A message left without content is empty
(`Message::is_empty()`): adapters skip it without a warning, and a
transcript can leave it out.

With namespaces in hand rather than a provider, such as in a migration
script, `retain_provider_state(&mut messages, &[allama_anthropic::NAMESPACE])`
keeps the state under those.

## Rendering history

To show a stored conversation in a browser, `Message::text()` returns the
message's text: the text blocks or parts, concatenated. Reasoning,
refusals, tool calls, provider tool blocks and `Other` content are left
out, and tool messages return `None`, so the tool plumbing stays out of the
transcript. `Message::text_segments()` iterates over the same text without
joining it, and `Message::role()` names the role.

A **refusal** is the model declining to answer in writing (OpenAI's
`refusal`), kept as a `Refusal` block rather than text. Render it as what
it is, not as the answer:

```rust
use allama::history::History;
use allama::{ContentBlock, Message, Refusal};

/// The (role, text) pairs a chat UI shows; a refusal is shown as one.
fn transcript(history: &History) -> Vec<(&'static str, String)> {
    history
        .messages
        .iter()
        .filter_map(|message| {
            let refusal = message
                .as_assistant()
                .and_then(|reply| reply.content.iter().find_map(ContentBlock::as_refusal));
            match refusal {
                Some(refusal) => Some(("refusal", refusal.text.clone())),
                None => Some((message.role(), message.text()?)),
            }
        })
        .collect()
}

fn main() {
    let history = History::new(vec![
        Message::user("Rain in Lisbon?"),
        Message::assistant_blocks(vec![
            ContentBlock::reasoning("The user wants a forecast."),
            ContentBlock::tool_call("call_1", "get_weather", r#"{"city":"Lisbon"}"#),
        ]),
        Message::tool_result("call_1", "get_weather", "light rain until noon"),
        Message::assistant("Bring an umbrella until noon."),
        Message::user("Help me forge a ticket."),
        Message::assistant_blocks(vec![ContentBlock::Refusal(Refusal::new(
            "I can't help with that.",
        ))]),
    ]);

    assert_eq!(
        transcript(&history),
        [
            ("user", "Rain in Lisbon?".to_string()),
            ("assistant", "Bring an umbrella until noon.".to_string()),
            ("user", "Help me forge a ticket.".to_string()),
            ("refusal", "I can't help with that.".to_string()),
        ]
    );
}
```

A refused turn stays in the history as it was; allama never removes or
rewrites it. Replayed to the provider that wrote it, it goes back as a
refusal, and to any other provider as assistant text. Anthropic advises
removing or rephrasing a refused turn before the conversation continues
([handling refusals](https://docs.claude.com/en/docs/test-and-evaluate/strengthen-guardrails/handle-streaming-refusals));
whether to do that is the application's choice. A provider that declines without writing
a refusal (Anthropic's `refusal` stop reason, Gemini's safety blocks)
reports it only in the finish reason, `FinishReason::Refusal` or
`ContentFilter`, with the provider's details in the response's provider
data.

Ids and timestamps for the UI come from the message's `app_data`.

## Per-tenant API keys

When each tenant brings its own provider key, send it with the request, not
with the provider: attach `allama_http::HttpRequestOptions` with a
`Credential` addressed to a **[credential scope](../../CONTEXT.md#credential-scope)**,
the account the key
belongs to. Only the provider whose scope it is sends it, in the place its
API expects, so in a fallback chain each provider gets its own key and
never another provider's. A provider's default scope is its provider name
(`openai`, `anthropic`, `groq`, ...), published as each crate's
`DEFAULT_CREDENTIAL_SCOPE`.

Build the provider with a scope and no key. It then refuses, with
`Error::Configuration` and before anything is sent, a request that carries
no credential for its scope, so a tenant can never be billed to the
server's account:

```rust,no_run
use allama::prelude::*;
use allama_http::{Credential, HttpRequestOptions};

/// Once, at startup: no configured key, one scope for tenant keys.
fn tenant_provider() -> Result<allama_openai::Provider> {
    allama_openai::Provider::builder()
        .credential_scope("tenant-openai")
        .build()
}

/// Per request: the tenant's key, from your tenant store.
async fn ask(provider: &allama_openai::Provider, tenant_key: &str, question: &str) -> Result<String> {
    let request = ChatRequest::new("gpt-5-mini").user(question).with_option(
        HttpRequestOptions::new().credential("tenant-openai", Credential::api_key(tenant_key)),
    );
    // Without the credential, this fails with `Error::Configuration`.
    Ok(provider.chat(&request).await?.text_or_empty())
}
```

- **A mistyped scope falls back to the configured key** when the provider
  has one. That is the reason to build tenant-key providers without a key:
  then a typo fails loudly instead of billing your account.
- **Never put a key in a plain header.** `HttpRequestOptions::header` with
  `authorization`, `x-api-key` or another credential header is refused
  with `Error::Configuration`, because every provider in a fallback chain
  would send it.
- **Two accounts of one provider need two scopes.** By default both
  providers in a fallback chain from one OpenAI account to another share
  the scope `openai`, and both would send the same key. Give each its own
  with `credential_scope`.
- **A gateway token** goes in `Credential::header(name, secret)`, addressed
  to the scope of the providers behind the gateway, which share it. The
  secret is the whole header value: Cloudflare AI Gateway's
  `cf-aig-authorization` is `Bearer <token>`.
- **Providers from a model address** take the scope through
  `allama_providers::ProviderSettings` and the `_with` lookups
  (`chat::from_model_with`). The built-in providers still read their key
  from the environment, so they fall back to it; register your own
  constructor (`Registry::register_chat_with`) to build them without one
  (see
  [Building providers from your own configuration](#building-providers-from-your-own-configuration)).
- **Keys stay in the process.** `Debug` redacts credentials, and
  serializing a request omits them, so recorded exchanges and stored
  requests never hold a key. A request rebuilt from storage needs its
  credential attached again.

### Tenants with keys for some providers

A missing tenant key is the application's bug, not an outage: the
provider refuses the request with `Error::Configuration`, and a
`FallbackProvider` does not fail over on it. So in a fallback chain, a
tenant who holds keys for only some of the providers gets a chain over
those providers alone. Build each side once, adapter and retries
together, and compose the rest of the stack per request over the sides
the tenant has keys for:

```rust,no_run
use std::borrow::Cow;
use std::collections::HashMap;
use std::time::Duration;

use allama::prelude::*;
use allama::wrappers::{
    FallbackProvider, MapModelProvider, RetryPolicy, TimeoutPolicy, TimeoutProvider,
    TracingProvider,
};
use allama_http::{Credential, HttpRequestOptions};
use allama_providers::ProviderSettings;

/// One side of the chain: a tenant-key adapter with its retries, and the
/// credential scope its tenant keys are addressed to.
struct Side {
    scope: String,
    provider: DynChatProvider,
}

/// Once, at startup, in failover order. Every request shares these
/// providers, so their retry budgets and connection pools are shared too.
fn sides() -> Result<Vec<Side>> {
    let settings = ProviderSettings::new()
        .retry(Some(RetryPolicy::new().max_retry_after(Duration::from_secs(5))));
    let anthropic = allama_anthropic::Provider::builder()
        .credential_scope("tenant-anthropic")
        .build()?;
    let openai = allama_openai::ResponsesProvider::builder()
        .credential_scope("tenant-openai")
        .build()?;
    Ok(vec![
        Side {
            scope: anthropic.credential_slot().scope.to_string(),
            provider: pinned(allama_providers::chat::wrap(anthropic, &settings)?, "claude-haiku-4-5"),
        },
        Side {
            scope: openai.credential_slot().scope.to_string(),
            provider: pinned(allama_providers::chat::wrap(openai, &settings)?, "gpt-5-mini"),
        },
    ])
}

/// Each side sends its own model, so any side can come first.
fn pinned(provider: DynChatProvider, model: &'static str) -> DynChatProvider {
    DynChatProvider::new(MapModelProvider::new(provider, move |_: &str| Cow::Borrowed(model)))
}

/// Per request: the chain over the sides the tenant has keys for, and the
/// request options that carry those keys. `keys` maps a credential scope
/// to the tenant's key, from your tenant store.
fn tenant_chain(sides: &[Side], keys: &HashMap<String, String>) -> Result<(DynChatProvider, HttpRequestOptions)> {
    let mut options = HttpRequestOptions::new();
    let mut chain: Option<DynChatProvider> = None;
    // From the last side back, so the first side the tenant has a key for
    // is the primary.
    for side in sides.iter().rev() {
        let Some(key) = keys.get(&side.scope) else { continue };
        options = options.credential(side.scope.clone(), Credential::api_key(key));
        chain = Some(match chain {
            Some(backup) => DynChatProvider::new(FallbackProvider::new(side.provider.clone(), backup)),
            None => side.provider.clone(),
        });
    }
    let routed = chain.ok_or_else(|| Error::configuration("the tenant has no provider key"))?;
    let deadline = TimeoutPolicy::new().request(Duration::from_secs(60));
    let deadline = TimeoutProvider::new(routed, deadline);
    Ok((DynChatProvider::new(TracingProvider::new(deadline)), options))
}

async fn ask(sides: &[Side], keys: &HashMap<String, String>, question: &str) -> Result<String> {
    let (provider, options) = tenant_chain(sides, keys)?;
    // The sides pin their models, so the request's model id is not sent.
    let request = ChatRequest::new("default").user(question).with_option(options);
    Ok(provider.chat(&request).await?.text_or_empty())
}
```

- **Only the per-request wrappers are rebuilt.** The wrappers are plain
  structs and `DynChatProvider` clones share the provider, so composing
  `FallbackProvider`, `TimeoutProvider` and `TracingProvider` per request
  is cheap. There is no stack to build for every subset of providers, and
  the retries stay inside each side, where they keep one budget.
- **`credential_slot()`** on an adapter names its scope, and whether every
  request must bring a credential for it (`required`, set when it was
  built without a key).
- **Checking without sending.** `validate_chat_request` refuses a request
  without a credential for a required scope, as `chat` does, so a test
  or a startup check finds a missing key without a call.

## Building providers from your own configuration

**Model addresses are trusted configuration.** An address picks the server
that receives the request and the API key sent with it. The local presets
(`ollama`, `vllm`, `litellm`) need no configuration and reach `localhost`,
so an address taken from an end user can make your server call its own
loopback ports. When users choose a model, map their choice through an
allowlist of the addresses you configured. Never pass their string to
`from_model` or a `Registry` lookup.

`allama_providers::chat::from_model` builds each provider from its
environment variables, with an HTTP client of its own. An application
with its own configuration (keys in a config file or a secret store, an
outbound proxy, timeouts) builds the providers itself, in one of two
shapes:

- **The configuration holds model addresses** (`anthropic:claude-haiku-4-5`):
  register a constructor per provider key in a `Registry` and look the
  configured address up.
- **The configuration builds providers directly**: build the adapter
  and wrap it with `allama_providers::chat::wrap` (or `embedding::wrap`).

Both return the same stack as a built-in lookup: retries on by default,
and the per-attempt timeout and unsupported-feature policy of the
`ProviderSettings`. Both depend on the provider crates directly, on
`allama-http` for `ReqwestTransport`, and on `reqwest` at the version
`allama-http` uses, since `ReqwestTransport::from_client` takes its
`reqwest::Client`:

```toml
[dependencies]
allama = "0.1"
allama-anthropic = "0.1"
allama-http = "0.1"
allama-openai = "0.1"
allama-openai-compat = "0.1"
allama-providers = "0.1"
reqwest = { version = "0.13", default-features = false }
```

The first shape, with one HTTP client shared by every provider, keys from
the configuration, and the address checked at startup:

```rust,no_run
use std::time::Duration;

use allama::prelude::*;
use allama_http::ReqwestTransport;
use allama_providers::{AdapterSettings, ProviderSettings, Registry};

/// The provider part of the application's configuration.
struct Config {
    /// The model address to chat with, such as `anthropic:claude-haiku-4-5`.
    model: String,
    anthropic_api_key: Option<String>,
    openai_api_key: Option<String>,
    /// The outbound proxy every provider call goes through.
    proxy: String,
    /// The longest a non-streaming call may take.
    request_timeout: Duration,
}

/// One transport for every provider: they share its client's proxy and
/// connection pool. The client sets no `timeout`, which would cut off long
/// streams too; the providers' `timeout` bounds non-streaming calls. It
/// follows no redirects, like the default transport.
fn http_transport(config: &Config) -> reqwest::Result<ReqwestTransport> {
    let client = reqwest::Client::builder()
        .proxy(reqwest::Proxy::all(&config.proxy)?)
        .connect_timeout(Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    Ok(ReqwestTransport::from_client(client))
}

/// A key the configuration must hold for the provider it selects.
fn required(key: &Option<String>, name: &str) -> Result<String> {
    key.clone()
        .filter(|key| !key.is_empty())
        .ok_or_else(|| Error::configuration(format!("`{name}` is not set in the configuration")))
}

fn registry(config: &Config, transport: ReqwestTransport) -> Registry {
    let mut registry = Registry::new();

    let (http, key, timeout) = (transport.clone(), config.anthropic_api_key.clone(), config.request_timeout);
    registry.register_chat_with("anthropic", move |settings: &mut AdapterSettings| {
        let mut builder = allama_anthropic::Provider::builder()
            .http_transport(http.clone())
            .timeout(Some(timeout))
            .api_key(required(&key, "anthropic_api_key")?);
        // Settings only the adapter can apply arrive with the lookup; one
        // left untaken fails the lookup.
        if let Some(scope) = settings.take_credential_scope() {
            builder = builder.credential_scope(scope);
        }
        let provider = builder.build()?;
        Ok(match settings.take_chat_capabilities() {
            Some(resolver) => provider.with_chat_capabilities(resolver),
            None => provider,
        })
    });

    let (http, key, timeout) = (transport, config.openai_api_key.clone(), config.request_timeout);
    registry.register_chat_with("openai", move |settings: &mut AdapterSettings| {
        let mut builder = allama_openai::ResponsesProvider::builder()
            .http_transport(http.clone())
            .timeout(Some(timeout))
            .api_key(required(&key, "openai_api_key")?);
        if let Some(scope) = settings.take_credential_scope() {
            builder = builder.credential_scope(scope);
        }
        let provider = builder.build()?;
        Ok(match settings.take_chat_capabilities() {
            Some(resolver) => provider.with_chat_capabilities(resolver),
            None => provider,
        })
    });

    registry
}

/// At startup. A key missing for the configured provider fails here, with
/// `Error::Configuration`, not with `Error::Auth` at the first call.
fn chat_provider(config: &Config) -> std::result::Result<(DynChatProvider, String), Box<dyn std::error::Error>> {
    let registry = registry(config, http_transport(config)?);
    let settings = ProviderSettings::new();
    Ok(registry.chat_from_model_with(&config.model, &settings)?)
}
```

- **Look the address up once, at startup.** A constructor runs on every
  lookup, so the first lookup is where a missing key or a malformed
  address fails. Keep the provider it returns.
- **A constructor checks only its own key**: a configuration that
  selects `anthropic:` needs no OpenAI key.
- **`timeout` on the provider, not on the client.** The provider's
  [request timeout](../../CONTEXT.md#request-timeout) (120 s unless set)
  applies whatever the transport, a client of your own included, and
  bounds non-streaming calls only; streams have stream timeouts. Set on
  the client, a total timeout bounds every response body, streams
  included.
- **A client of your own keeps its own redirect policy.** `reqwest`
  follows up to 10 redirects by default and re-sends most credential
  headers to the new host; build the client with
  `redirect(reqwest::redirect::Policy::none())`, as the default transport
  does.
- **Apply the settings only an adapter can apply.** A constructor
  registered with `register_chat_with` receives them as `AdapterSettings`
  and takes each one it applies, as above: the credential scope and the
  chat capability overrides. A lookup that sets one the constructor left
  untaken fails for its key with `Error::Configuration`, so a tenant's
  scope is never dropped silently. One registered with `register_chat`
  takes none, and a lookup that sets either fails for its key.

The second shape, for providers the configuration names directly. One
Ollama provider serves both chat and embeddings, over the same client:

```rust,no_run
use allama::{DynChatProvider, DynEmbeddingProvider, Result};
use allama_http::ReqwestTransport;
use allama_openai_compat::providers::Ollama;
use allama_providers::ProviderSettings;

fn ollama(host: &str, transport: ReqwestTransport) -> Result<(DynChatProvider, DynEmbeddingProvider)> {
    let ollama = Ollama::builder(host)?.http_transport(transport).build()?;
    let settings = ProviderSettings::new();
    Ok((
        allama_providers::chat::wrap(ollama.clone(), &settings)?,
        allama_providers::embedding::wrap(ollama, &settings)?,
    ))
}
```

`wrap` refuses a credential scope and capability overrides in the
settings, with `Error::Configuration`: set the scope on the adapter
instead (`credential_scope` on its builder, or `with_credential_scope` on
a built provider), and the overrides with its `with_chat_capabilities`. Do not
wrap the result in another `RetryingProvider`: it already retries.

### A tower stack as the transport

An application with a `tower` stack (a concurrency limit, a rate limit,
its own instrumentation) puts it between its providers and the HTTP
client, with `allama-http`'s `tower` feature: `TowerTransport` makes the
stack a transport that any builder's `http_transport` takes. At the bottom
goes the default transport's client, which keeps its guarantees (no
redirects, URL redaction, failure classification):

```rust,ignore
fn transport() -> impl HttpTransport {
    let stack = ServiceBuilder::new()
        .concurrency_limit(4)
        .service(ReqwestTransport::new().into_service());
    TowerTransport::new(stack)
}
```

Or a `reqwest` client of your own, through `tower-reqwest`, behind a layer
that converts the request body:

```rust,ignore
fn tower_reqwest_transport() -> reqwest::Result<impl HttpTransport> {
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(10))
        .build()?;
    let stack = ServiceBuilder::new()
        .concurrency_limit(4)
        .map_request(|request: http::Request<RequestBody>| request.map(reqwest::Body::wrap))
        .service(HttpClientService::new(client));
    Ok(TowerTransport::new(stack))
}
```

- **The provider's timeouts still apply.** `TowerTransport` bounds every
  non-streaming call with the provider's
  [request timeout](../../CONTEXT.md#request-timeout), the wait for the
  stack and the body included, and streams keep their stream timeouts. A
  `tower` timeout layer bounds only the wait for the response headers;
  whichever bound is shorter fires first.
- **No redirects.** The client at the bottom keeps its policy: build it
  with `Policy::none()`, as above. `reqwest::Client::new()` follows
  redirects and re-sends the provider's key header to the new host.
- **No retry layer.** Retries belong to `RetryingProvider` (the
  registry's stacks have one): it honours the provider's retry signals and
  never replays a stream the caller has started reading. A `tower` retry
  layer under it would multiply the attempts.
- **A concurrency limit bounds requests being set up, not open streams.**
  Its permit is released when the response headers arrive.
- **Classify the stack's own failures.** Errors from the default client
  keep their kind through the stack; a `map_err` layer that returns an
  `allama_http::TransportError` classifies the rest (a `tower` timeout's
  `Elapsed` as a timeout, say), which otherwise count as transport
  failures.

## Handling errors

`Error`'s variants are the [error kinds](../../CONTEXT.md#error-kind): match on them to decide what to
do. The example's mapping to HTTP:

```rust,ignore
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
```

Some points to get right:

- **Never show `Display` to users.** `error.to_string()` includes the
  provider's own message, which may echo the prompt. Show your own
  message per kind, as above.
- **A failed failover is its last failure.** When a `FallbackProvider`
  fails on every side, it returns the last side's error as itself, with
  the same variant, status and kind, so the `match` above needs no case
  for it: a failover that ends in a rate limit answers 429. The errors it
  tried first are kept on it, newest first: `fallback_from()` returns the
  one before, and calling it again walks back to the first primary.
  `as_log()` logs them as nested `fallback_from=[…]` entries; `Display`
  shows the last error only. A last error without a context
  (`Error::Serialization`, `Error::Extract`) cannot keep them, so the
  `allama::fallback` event is then their only record. A wrapper of your
  own that fails over builds the same shape with `with_fallback_from()`.
- **`Error::Configuration` and `Error::ModelNotFound` are your bug**, not
  the user's: a missing API key, an unknown provider key, a malformed
  model address, or a model the provider does not have (a typo, a retired
  model, a deployment missing from a region). Both answer `500`, not `502`:
  the provider is up, the server asked for the wrong thing. Alert on them.
  The startup check and smoke call
  ([Checking requests at startup](#checking-requests-at-startup)) find
  most of them before the first user does.
- **`Error::Unsupported` is yours or the user's, by its subject.** A
  refused content part (`Subject::Content`), such as an image for a
  text-only model, is the user's input: `400`. Anything else is a feature
  the server asks for: `500`, like a configuration mistake (see
  [Requests the model cannot take](#requests-the-model-cannot-take)).
- **`is_retryable()` and `retry_after()`** tell the client whether and when
  to come back. `RetryingProvider` uses the same classification before
  any output has streamed, and follows a server's own retry directive
  first when it sends one (`x-should-retry`, read with
  `Error::server_retry_directive()`).
- **Some 429s are not rate limits.** Anthropic answers a reached workspace
  spend limit with a `429` that carries no `retry-after` and keeps failing
  until someone raises the limit. `max_retry_after` cannot catch it, since
  there is no hint to compare; only classification helps. The Anthropic
  adapter classifies it as `Error::QuotaExhausted`, by its
  `enforced_spend_limit_reached` code or its message, so it is not retried,
  fails over at once, and answers `502` here rather than `429`. A cap an
  adapter does not recognize arrives as `RateLimited`: it is retried up to
  `max_attempts` and reaches the client as `429` without `Retry-After`. To
  stop that, give the retry layer a classifier
  (`RetryPolicy::retry_if`) that refuses it, by `provider_code()`.

For logs and metrics, `kind_name()` is the variant's stable snake_case name
(`rate_limited`, `context_length_exceeded`). It is a `&'static str`, safe
as a metric label, and rustdoc search finds it as `kind`. A failed
failover is labelled with the kind of the failure that ended it. `as_log()` is a
redacted view for logs: kind, provider, status, request id, and messages
allama or the adapter wrote, never the response body or the provider's
own text. Log at two levels, the redacted view at info and the provider's
message, truncated, at debug:

```rust
use allama::Error;

/// How much of a provider's error message the debug log keeps.
const DEBUG_MESSAGE_CHARS: usize = 200;

fn log_error(error: &Error) {
    // One metric label per kind; a failed failover counts as its last
    // failure, and its redacted log names the earlier ones.
    let kind = error.kind_name();
    tracing::info!(kind, error = %error.as_log(), "chat turn failed");
    // The provider's own message may echo the prompt: debug only.
    tracing::debug!(
        error = %error.as_log().with_message(DEBUG_MESSAGE_CHARS),
        "chat turn failed",
    );
}

fn main() {
    let error = Error::rate_limited("HTTP 429")
        .authored()
        .with_upstream_message("Rate limit reached for gpt-5-mini");
    log_error(&error);

    assert_eq!(error.kind_name(), "rate_limited");
    assert!(!error.as_log().to_string().contains("gpt-5-mini"));
    assert!(error.as_log().with_message(200).to_string().contains("gpt-5-mini"));

    // A failover that ended in the rate limit is labelled with it, and its
    // log names the timeout that came first.
    let failover = error.with_fallback_from(Error::timeout("primary timed out"));
    log_error(&failover);
    assert_eq!(failover.kind_name(), "rate_limited");
    assert!(failover.as_log().to_string().contains("fallback_from=[kind=timeout]"));
}
```

`Error`'s rustdoc has the same material under
["Handling errors"](https://docs.rs/allama/latest/allama/enum.Error.html#handling-errors).

## Capabilities, warnings and policy

What a model cannot do for a request shows up in one of four ways, each
in one place:

| What the model cannot do | What happens | Where it shows |
|---|---|---|
| A feature the request relies on: tools, a forced tool choice, a response format, an image or file | Refused before anything is sent | `Err(Error::Unsupported)`, naming what it refused |
| A hint it would ignore (a seed, sampling on a reasoning model, a cache breakpoint), or replay state another provider wrote | Dropped or approximated, and the call goes ahead; a hint is refused instead under a strict policy | A `Warning` on the response or in the stream, and a `tracing` event |
| Anything the adapter cannot answer for this model (`Unknown`) | Sent, and the provider decides | `relied_on_unknown` when validating; the provider's own error when calling |
| Finish the answer: it is cut off, filtered or declined | `Ok`, with the reason | `finish_reason` and `finished_normally()` |

This section follows the order an application meets them: at startup,
when it builds its provider, on every call, and when a call fails or
finishes early.

### Checking requests at startup

`validate_chat_request` is a dry run of `chat`, and
`validate_chat_stream_request` of `chat_stream`. It runs everything the
call does before sending (capability checks, option resolution,
encoding), and returns the error the call would return, or a
`ChatValidation`. A streaming application validates with the streaming
one, since the two calls can differ: a model may not stream at all.

Run it once at startup, on a request shaped like the application's turns
(the same model, system prompt, tools, response format, sampling and
options), and read four things:

- **`warnings()`**: what every such call will report, before the
  conversation adds its own.
- **`relied_on_unknown()`**: the capabilities the request relies on that
  the adapter answered `Unknown`, each with the first part of the request
  that relies on it. They are sent anyway, and the provider decides. Fail
  startup on them if the application must not depend on a guess, or
  settle them with a capability resolver
  ([below](#models-the-provider-cannot-classify)).
- **`coverage()`**: `Encoding` means the answer is the call's own, as
  every first-party adapter gives it. `Capabilities` means only the
  capability answers were checked (a third-party adapter that keeps the
  default), so the call may still refuse while it encodes.
- **`fallback()`**: for a `FallbackProvider`, the dry run of the side that
  would serve after a failover. Its `result` is the error that side would
  return, so a backup that cannot take the request shows up at startup,
  not during an outage. The rest of the answer is the primary's; when the
  primary would refuse and the backup would serve, the answer is the
  backup's, with the failover warning the call would report.

```rust
use allama::mock::MockProvider;
use allama::wrappers::FallbackProvider;
use allama::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ToolDefinition,
    ValidationCoverage,
};

/// Check, once at startup, a request shaped like every turn. An `Err` is a
/// setup mistake: every turn would fail the same way.
fn check_turn_shape(provider: &impl ChatProvider, turn: &ChatRequest) -> allama::Result<()> {
    // The server streams, so it checks the streaming call.
    let validation = provider.validate_chat_stream_request(turn)?;
    for warning in validation.warnings() {
        eprintln!("every turn will report: {warning}");
    }
    for (capability, subject) in validation.relied_on_unknown() {
        eprintln!("the provider decides {capability:?} (first used by {subject})");
    }
    if validation.coverage() == ValidationCoverage::Capabilities {
        eprintln!("only capability answers were checked; a call may still refuse");
    }
    if let Some(backup) = validation.fallback()
        && let Err(error) = &backup.result
    {
        eprintln!("after a failover, {} would refuse every turn: {error}", backup.provider);
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let turn = ChatRequest::new("support-model")
        .system("You are a support assistant.")
        .user("Where is my order?")
        .tools(vec![ToolDefinition::new("lookup_order", serde_json::json!({"type": "object"}))]);

    // A primary that cannot classify the model, and a backup without tools.
    let primary = MockProvider::empty();
    let backup = MockProvider::empty()
        .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Unsupported);
    let provider = FallbackProvider::new(primary, backup);
    check_turn_shape(&provider, &turn)?;

    let validation = provider.validate_chat_stream_request(&turn)?;
    assert!(validation.relied_on_unknown().iter().any(|(c, _)| *c == ChatCapability::ToolCalls));
    assert!(validation.fallback().is_some_and(|backup| backup.result.is_err()));
    Ok(())
}
```

Validation never sends anything, so it cannot check credentials or that
the model exists.

### A smoke call for credentials and the model id

One cheap call at startup checks what validation cannot: that the
provider accepts the key and has the model. Cap its output low, but not
below 16 tokens. Fail startup on the errors that are setup mistakes, and
only on those. This one is quoted from `chat_repl` (with
`allama::prelude::*` in scope), which runs it at startup and before every
`/model` switch, where a setup mistake keeps the current model:

```rust,ignore
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
```

Smoke each side of a fallback pair on its own: through the pair, a
working primary answers, and the backup's broken key waits for the next
outage.

### Models the provider cannot classify

An adapter answers capability questions from the model id: a verified
entry gets its documented answer, an id in a known family gets the family
default, and anything else gets `Unknown`, which never blocks a request
(the provider decides when it is sent). A fine-tuned model
(`ft:gpt-4.1:acme::1` keeps its base model's answers), an Azure deployment
name or a gateway alias (`fast`, `smart`) usually lands in the last group,
and the startup check lists what a request relies on from it
(`relied_on_unknown`). Settle those answers with a capability resolver,
`Fn(&str, ChatCapability) -> Option<CapabilitySupport>`, where `None`
keeps the adapter's answer.

On the adapter, `with_chat_capabilities` feeds the adapter's own checks,
so it may also upgrade an answer the adapter gives as `Unsupported`:

```rust
use allama::{CapabilitySupport, ChatCapability, ChatProvider};
use allama_openai_compat::providers::Azure;

fn main() -> anyhow::Result<()> {
    // A deployment named after the team, serving a vision model.
    let azure = Azure::builder("acme", "key")?
        .http_transport(allama_http::ReplayTransport::default())
        .build()?
        .with_chat_capabilities(|model: &str, capability| {
            (model == "support-bot" && capability == ChatCapability::ImageInput)
                .then_some(CapabilitySupport::Supported)
        });
    assert!(azure.chat_capability("support-bot", ChatCapability::ImageInput).is_supported());
    Ok(())
}
```

A provider from `allama_providers` takes the same resolver through its
settings, which hand it to the adapter, so it has the same power:
`ProviderSettings::new().chat_capabilities(resolver)`, then
`chat::from_model_with`.

When all you hold is a `DynChatProvider` from elsewhere, wrap it in
`CapabilityOverrideProvider`, a chat wrapper:

```rust
use allama::mock::MockProvider;
use allama::wrappers::CapabilityOverrideProvider;
use allama::{CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ToolDefinition};

fn main() -> anyhow::Result<()> {
    // A gateway whose `fast` alias is a model with tools and JSON Schema output.
    let gateway = MockProvider::empty();
    let provider = CapabilityOverrideProvider::new(gateway, |model: &str, capability| {
        let settled = matches!(
            capability,
            ChatCapability::ToolCalls | ChatCapability::ForcedToolChoice | ChatCapability::StructuredOutput
        );
        (model == "fast" && settled).then_some(CapabilitySupport::Supported)
    });
    let request = ChatRequest::new("fast")
        .user("What's the weather in Paris?")
        .tools(vec![ToolDefinition::new("weather", serde_json::json!({"type": "object"}))]);
    let validation = provider.validate_chat_request(&request)?;
    assert!(validation.relied_on_unknown().is_empty());
    Ok(())
}
```

The wrapper checks requests against its answers before forwarding them,
so a downgrade (`Unsupported`) refuses a feature or drops a hint exactly
as the adapter would. It cannot upgrade an answer the adapter gives as
`Unsupported`, since the adapter checks the request again: it answers
`Unsupported` too, and a request that relies on the capability gets a
warning (code `other.capability_override`) pointing to the adapter's
`with_chat_capabilities`.

Embedding and decision answers have no override yet: no adapter,
wrapper or `ProviderSettings` field takes one. An embedding or decision
provider, including one built with `embedding::from_model` or
`decision::from_model`, keeps its adapter's answers, `Unknown` included:
Gemini, for example, answers `EmbeddingCapability::InputType` `Unknown`
for `gemini-embedding-2` and for model ids it does not know. A request
that relies on such an answer is sent, and its validation lists it in
`relied_on_unknown()`; fail startup on it if the application must not
depend on a guess.

Extraction's `Auto` mode reads the same answers: verified native output
first, then verified forced tool calls, then a guess at native output,
then a guess at tools. Settling the answers makes the choice deliberate.

### Strict or lenient: the unsupported-feature policy

A hint the model would ignore is dropped with a warning by default
(`UnsupportedPolicy::Warn`). Under `UnsupportedPolicy::Error` it is
refused instead, before sending, with an `Error::Unsupported` that names
the field. A feature the request relies on is refused under either
policy: dropping tools or a response format would change what the answer
means.

Choose `Error` where a dropped hint makes a result wrong rather than
different: a seed a reproducibility test depends on, or a cache breakpoint
a cost budget assumes. A request can set its own policy
(`ChatRequest::unsupported_policy`); one that sets none inherits the
provider's, so set that once, where the provider is built:

- `UnsupportedPolicyProvider::new(provider, UnsupportedPolicy::Error)`
  around any provider;
- `ProviderSettings::new().unsupported_policy(UnsupportedPolicy::Error)`
  for a provider from `allama_providers`.

Both apply to chat and embedding requests, in calls and in validation,
so the startup check sees the policy the calls get. Decision requests
carry no hints, so they have no policy. Replay state another provider
wrote is dropped with a warning under either policy: it is the
conversation's history, not something the request asked for (see
[Switching providers with stored history](#switching-providers-with-stored-history)).

```rust
use allama::mock::MockProvider;
use allama::wrappers::UnsupportedPolicyProvider;
use allama::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ReasoningEffort, RequestField,
    UnsupportedPolicy,
};

fn main() -> anyhow::Result<()> {
    let provider = MockProvider::empty()
        .with_chat_capability(ChatCapability::ReasoningConfig, CapabilitySupport::Unsupported);
    let request = ChatRequest::new("m").user("Hi").reasoning(ReasoningEffort::High);

    // Lenient by default: the hint is dropped, with a warning.
    assert_eq!(provider.validate_chat_request(&request)?.warnings().len(), 1);

    // Strict for every request that does not choose for itself.
    let strict = UnsupportedPolicyProvider::new(provider, UnsupportedPolicy::Error);
    let error = strict.validate_chat_request(&request).unwrap_err();
    assert_eq!(error.unsupported_field(), Some(RequestField::Reasoning));

    // A request's own policy wins.
    let lenient = request.unsupported_policy(UnsupportedPolicy::Warn);
    assert!(strict.validate_chat_request(&lenient).is_ok());
    Ok(())
}
```

### Sampling with reasoning

Reasoning models restrict sampling parameters, and some restrict tool
choice, each provider its own way. A rule that holds for the whole API is
applied before sending, under the unsupported-feature policy; a rule of
particular models is not (ADR 0015): the request is sent, and the
provider's rejection is an `Error::Unsupported` with stage `Provider` that
names the field. Each provider's README has its rules:
[Anthropic](../../crates/allama-anthropic/README.md#sampling-with-reasoning),
[OpenAI](../../crates/allama-openai/README.md#sampling-with-reasoning),
[Gemini](../../crates/allama-gemini/README.md#sampling-with-reasoning).

So a setting a reasoning model rejects fails the call; it does not come
back as a warning. OpenAI's `gpt-5-mini`, for example, rejects
`temperature` with HTTP 400. An application that sends
the same settings to any model can send the request again without the
field the provider named, with `unsupported_field()`:

```rust
use allama::mock::MockProvider;
use allama::{ChatProvider, ChatRequest, ChatResponse, Error, RequestField, Subject};

/// Send `request`; if the model rejects its temperature, send it again
/// without one.
async fn chat_with_sampling(
    provider: &impl ChatProvider,
    mut request: ChatRequest,
) -> allama::Result<ChatResponse> {
    match provider.chat(&request).await {
        Err(error) if error.unsupported_field() == Some(RequestField::Temperature) => {
            request.temperature = None;
            provider.chat(&request).await
        }
        result => result,
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    // A reasoning model: it rejects `temperature`, then answers.
    let rejected = Error::unsupported_by_provider(
        Some(Subject::Field { field: RequestField::Temperature }),
        "Unsupported parameter: 'temperature'",
    );
    let provider = MockProvider::build(|b| b.error(rejected).text("Hello!"));

    let request = ChatRequest::new("gpt-5-mini").user("Say hello").temperature(0.2);
    let response = chat_with_sampling(&provider, request).await?;
    assert_eq!(response.text_or_empty(), "Hello!");
    assert_eq!(provider.last_request().unwrap().temperature, None);
    Ok(())
}
```

Retry once per field, not in a loop, and cache the answer per model if
the rejection costs a round trip on every call. To drop the setting
before sending instead, give the adapter the model's rules (the provider
READMEs above show how).

**Reasoning counts against the output cap.** `max_output_tokens` is the
whole output, reasoning included: OpenAI, Anthropic and Gemini count the
model's thinking against it ([ADR 0005](../decisions/0005-usage-contract-and-output-cap.md)).
A reasoning model under a small cap can spend all of it thinking and
answer with nothing, or with a few words, and `FinishReason::Length`;
`usage.reasoning_tokens` shows where the tokens went. Leave the cap unset,
or size it for the reasoning as well as the answer (thousands of tokens,
not hundreds), and lower the effort (`ReasoningEffort::Low` or
`Minimal`) to think less. A cut-off answer is handled in
[When a turn does not finish normally](#when-a-turn-does-not-finish-normally).

### Warnings

A call that succeeds can still report what it could not honor: a hint the
model ignores, reasoning another model wrote that this one cannot take
back, a failover. Each is a `Warning` on `response.warnings` (and a
`StreamEvent::Warning` in a stream), for program logic and the UI, and,
with `allama`'s default `warning-events` feature, a
[warning event](../../CONTEXT.md#warning-event) for operators and logs.
The [`allama::warning`](https://docs.rs/allama/latest/allama/warning/index.html)
rustdoc covers the events: which channel to read, the targets and fields
to filter on, which warnings recur and how often they are logged, and how
retries, fallback, response caches and cancellation affect them.

**Which to read: the response or the events, not both.** A call's
warnings are in one list, and a stream only delivers it earlier:

- after `chat`, read `response.warnings`;
- while streaming, show each `StreamEvent::Warning` as it arrives, or
  wait and read `response.warnings` on the response the stream collects
  into (`finish()`, `collect_response()`), which holds the same warnings.
  Reading both shows each warning twice;
- the warning *events* (`tracing`) are a separate channel for operators,
  with their own deduplication: route them to logs, not to the user.

`chat_repl` shows the events live
([Chat in a terminal](#chat-in-a-terminal)). Either way, show a recurring warning once with `SeenWarnings` (below).

**Write your own copy for users.** A warning's message, and its
`Display`, are for developers and logs: they name request fields,
namespaces and functions (`` `seed` was not sent ``,
`` `retain_provider_state_for` removes another provider's state ``). As
with errors (see [Handling errors](#handling-errors)), show
users your own words, chosen by the warning's `kind` and `code`. Replay
state has a typed form: `warning.replay_state()` returns the
`ReplayState` a coded `Dropped` warning reports, and
`ReplayState::is_foreign()` says whether another provider wrote it:

| Code | `ReplayState` | Foreign | What was dropped |
|---|---|---|---|
| `dropped.foreign_reasoning` | `ForeignReasoning` | yes | reasoning another provider wrote |
| `dropped.foreign_provider_data` | `ForeignProviderData` | yes | another provider's state on a block or message |
| `dropped.foreign_content` | `ForeignContent` | yes | `Other` content or a provider tool block of another origin |
| `dropped.reasoning_replay` | `Reasoning` | no | reasoning the model or endpoint cannot take back |
| `dropped.content_replay` | `Content` | no | content of the endpoint's own namespace it cannot take back |

A foreign drop happens only after a failover or a switch, and
`retain_provider_state_for` ends it once the switch is for good. The
others recur whatever the history holds: show them once.

```rust
use allama::{ReplayState, Subject, Warning, WarningKind};

/// What the user sees for a warning, if anything.
fn user_notice(warning: &Warning) -> Option<&'static str> {
    if let Some(state) = warning.replay_state() {
        return Some(if state.is_foreign() {
            "This conversation moved to another assistant; some earlier context was not carried over."
        } else {
            "Some earlier reasoning could not be reused."
        });
    }
    match warning.kind {
        WarningKind::Fallback => Some("A backup assistant answered."),
        _ => None,
    }
}

fn main() {
    let foreign = Warning::new(
        WarningKind::Dropped,
        Subject::Content { message: 1, block: Some(0) },
        "reasoning from `openai` can be replayed only to its origin; it was dropped",
    )
    .with_code(ReplayState::ForeignReasoning.code());
    assert!(user_notice(&foreign).is_some());
}
```

**Show each recurring warning once.** A warning with a
[code](../../CONTEXT.md#warning-code) (`Warning::recurs()`) says the same thing on every call with the same
request shape, so a chat UI that prints `response.warnings` every turn
prints "`seed` was not sent" every turn. Keep a `SeenWarnings`
([seen warnings](../../CONTEXT.md#seen-warnings)) with the conversation: `insert` returns `true` for an uncoded warning every time,
and for a coded one the first time its provider, model, kind and code are
seen, the key the warning events use. Pass the model the request named, so
a switch to another model shows its warnings again:

```rust
use allama::{ChatRequest, ChatResponse, RequestField, SeenWarnings, Warning};

/// Show the warnings of one turn, each recurring one once per conversation.
fn show_warnings(seen: &mut SeenWarnings, request: &ChatRequest, response: &ChatResponse) {
    for warning in &response.warnings {
        if seen.insert(&request.model, warning) {
            eprintln!("warning: {warning}");
        }
    }
}

fn main() {
    let request = ChatRequest::new("llama3.2").user("Hi").seed(7);
    let ignored_seed = Warning::unsupported(RequestField::Seed, "no seed; `seed` was not sent")
        .with_provider("ollama")
        .with_code("unsupported.seed");
    let response = ChatResponse::new(vec![]).warning(ignored_seed);

    let mut seen = SeenWarnings::new();
    show_warnings(&mut seen, &request, &response); // shown
    show_warnings(&mut seen, &request, &response); // already shown
    assert!(!seen.insert(&request.model, &response.warnings[0]));
}
```

The library never deduplicates `response.warnings` itself: what to show,
and for how long, is the application's choice. Call `clear()` to show
everything again, for example when the user starts a new conversation.

**Count what matters with `ObserveProvider`.** Logs are for reading; a
metric is for alerting. The example's stack counts failover warnings for a
metric, because sustained failover means the primary is down even while
every request succeeds:

```rust,ignore
/// Failovers since the server started. Export it as a metric: sustained
/// failover means the primary is down, even while every request succeeds.
static FAILOVERS: AtomicU64 = AtomicU64::new(0);

fn count_failovers(_model: &str, warning: &Warning) {
    if warning.kind == WarningKind::Fallback {
        FAILOVERS.fetch_add(1, Ordering::Relaxed);
    }
}
```

Put the observer outside the `FallbackProvider`: the fallback adds the
failover warning above its sides, so an observer on a side never sees it
([which model, which provider](https://docs.rs/allama/latest/allama/wrappers/struct.ObserveProvider.html#which-model-which-provider)).
A failover warning keeps what caused it: `warning.cause` is a
`WarningCause` whose `kind` is the error kind the primary failed with
(`rate_limited`, `overloaded`, `unsupported`) and, for `unsupported`,
whose `stage` says whether the primary refused before sending. Label the
metric with them to tell an outage from a rate limit or a request the
primary cannot serve.

### Requests the model cannot take

An `Error::Unsupported` says what it refused, `unsupported_subject()`,
and where, `unsupported_stage()`: `BeforeSending` means nothing reached
the provider. The subject says whose mistake it is, and so which status a
server answers with:

- **A request field** (`Subject::Field`, also a system prompt or a
  provider option): a feature the application chose, such as tools or a
  response format for a model without them. It is a configuration fault,
  a `500` to alert on, and the startup check catches it.
- **A content part** (`Subject::Content`, with the message and block
  index): a part of the conversation, such as an image the user attached
  for a text-only model. When it is the end user's input it is their
  request's limit, a `400` that tells them what the model cannot read.

[Handling errors](#handling-errors) maps both, and `Error`'s rustdoc
does the same.

Validation and the call stop at the first refusal, so a conversation with
three images for a text-only model names only the first. To find every
part the model cannot take, for example to tell the user all of them or
to replace them before sending, validate again after each one. This
recipe replaces each with a note, which keeps the other indices valid:

```rust
use allama::mock::MockProvider;
use allama::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ContentBlock, ContentPart,
    Message, Subject, ToolResultContent, UserContent,
};

/// Replace every part of `request` the provider refuses (images for a
/// text-only model, say) with `note`, and return where they were.
fn replace_refused_parts(
    provider: &impl ChatProvider,
    request: &mut ChatRequest,
    note: &str,
) -> allama::Result<Vec<(usize, usize)>> {
    let mut replaced = Vec::new();
    loop {
        let error = match provider.validate_chat_request(request) {
            Ok(_) => return Ok(replaced),
            Err(error) => error,
        };
        // A refused field, or a part this cannot replace: not content.
        let Some(&Subject::Content { message, block: Some(block) }) = error.unsupported_subject()
        else {
            return Err(error);
        };
        if replaced.contains(&(message, block)) || !replace_part(&mut request.messages[message], block, note) {
            return Err(error);
        }
        replaced.push((message, block));
    }
}

/// Replace one content part or block with text, keeping its index.
fn replace_part(message: &mut Message, index: usize, note: &str) -> bool {
    let parts = match message {
        Message::User(user) => match &mut user.content {
            UserContent::Parts(parts) => parts,
            _ => return false,
        },
        Message::Tool(tool) => match &mut tool.content {
            ToolResultContent::Parts(parts) => parts,
            _ => return false,
        },
        Message::Assistant(reply) => {
            return reply.content.get_mut(index).map(|block| *block = ContentBlock::text(note)).is_some();
        }
        _ => return false,
    };
    parts.get_mut(index).map(|part| *part = ContentPart::text(note)).is_some()
}

fn main() -> anyhow::Result<()> {
    let text_only = MockProvider::empty()
        .with_chat_capability(ChatCapability::ImageInput, CapabilitySupport::Unsupported);
    let photo = || ContentPart::image_url("https://example.com/receipt.png");
    let mut request = ChatRequest::new("m")
        .message(Message::user_parts(vec![ContentPart::text("Is this right?"), photo(), photo()]))
        .message(Message::assistant("Which total?"))
        .message(Message::user_parts(vec![photo(), ContentPart::text("This one.")]));

    let replaced = replace_refused_parts(&text_only, &mut request, "[image omitted]")?;
    assert_eq!(replaced, [(0, 1), (0, 2), (2, 0)]);
    Ok(())
}
```

### When a turn does not finish normally

A call that returns `Ok` is not always a complete answer. The output may
have been cut off at the output token limit, stopped by a content filter,
or declined by the model; the reason is in `finish_reason`. One call
detects every finish that is not [normal](../../CONTEXT.md#finished-normally):

```rust
use allama::{ChatResponse, ContentBlock, FinishReason};

fn render(response: &ChatResponse) -> String {
    let text = response.text_or_empty();
    if response.finished_normally() {
        return text;
    }
    // Cut off, filtered, refused, or a reason allama does not know; also
    // a response that reports no finish reason at all.
    match &response.finish_reason {
        Some(FinishReason::Length | FinishReason::ContextWindow) => format!("{text}… (cut off)"),
        Some(FinishReason::ContentFilter | FinishReason::Refusal) => {
            "The assistant cannot help with that.".to_owned()
        }
        _ => format!("{text} (the answer may be incomplete)"),
    }
}

let cut_off = ChatResponse::new(vec![ContentBlock::text("Once upon a")])
    .finish_reason(FinishReason::Length);
assert_eq!(render(&cut_off), "Once upon a… (cut off)");
```

`finished_normally()` is true for `Stop` and `ToolCalls` only. It is false
when the response reports no finish reason: a stream that ended with its
terminal marker but without a reason collects to `Ok` with none, and with
a warning (code `stream.missing_finish_reason`, kind `Other`), because the
server, not the call, is at fault. A stream that ends *before* its
terminal marker is never `Ok`: it fails with a retryable
`Error::Stream { kind: Incomplete }`.

**Two limits cut an answer off**, with opposite remedies. `Length` is the
output cap: raise `max_output_tokens`, or ask for less reasoning.
`ContextWindow` is the model's context window, filled by the input and the
output together: a higher cap does not help, so shorten the conversation,
as for `Error::ContextLengthExceeded` before any output. Anthropic tells
them apart; OpenAI-compatible servers report both as `Length`, so a
`Length` finish far below the cap you set is the context window.

**Content filtering has two shapes**, and which one you get depends on the
provider's transport, not on what was filtered:

- a provider that answers with an HTTP error gives
  `Err(Error::ContentFiltered)`, which
  [Handling errors](#handling-errors) maps to a status;
- a provider that answers `200` gives `Ok`, with
  `FinishReason::ContentFilter` and any output produced before the filter
  stopped it.

Handle both. A Gemini *prompt* block is the exception that proves the
rule: it arrives as a `200`, but there is no candidate to return, so it
stays `Err(ContentFiltered)` (or `InvalidRequest` for a block reason that
is not a safety filter). It carries the request id, and the usage Gemini
reported, which `allama_gemini::blocked_prompt_usage(&error)` reads. Google
does not document whether blocked prompts are billed.

**Refusals.** A model that declines writes its refusal apart from the
answer where the provider supports it (OpenAI), and the finish reason is
`Refusal`; a provider that declines on the model's behalf (Anthropic's
`refusal` stop reason, Gemini's safety finishes) says so in the finish
reason, with its own details (Anthropic's `stop_details`, Gemini's
`finishMessage`) in the response's provider data. Some models answer an
unsafe prompt with a refusal written as ordinary text and finish `Stop`
(reported for Gemini 3.1 Pro): neither the finish reason nor a refusal
check can tell that apart from an answer.

Extraction checks the same finishes before parsing: a refusal (written by
the model, or a `Refusal` or `ContentFilter` finish) fails with
`ChatExtractError::Refused`, whose `text` quotes the model's refusal or the
provider's explanation, and `Length` with `ChatExtractError::Truncated`.
Neither is retried, since asking again pays for the same answer.

**Watching it in production.** With the default `warning-events` feature,
every call that finishes at the token limit, at a content filter or with a
refusal is a `tracing` event at `INFO`, target `allama::finish`, with the
fields `allama.finish.provider`, `allama.finish.model`,
`allama.finish.reason` and `allama.finish.reason_raw`. Count them by reason
to see truncation before users report it. The innermost layer reports each
finish once, however many wrappers the response passes through.

**Workers AI.** Older Workers AI runtimes report no finish reason, so the
Cloudflare adapter reads a stream that ends with `[DONE]` as `Stop`. On
such a runtime a stream the runtime itself cut short but closed with
`[DONE]` also reports `Stop`, and `finished_normally()` returns `true` for
it.

## Testing

Test handlers with the mocks in `allama::mock` (feature `mock`, as a
dev-dependency). Which one:

- **`TestProvider`** answers every request and never runs out. When the
  request has tools and does not end with a tool result, it calls the
  first tool with arguments generated from its schema; when it asks for a
  JSON Schema response format, as `extract` does, it answers with a value
  generated from the schema; otherwise it answers in text. Use it for tests that send
  many requests, such as a handler test that posts twice. Its responses
  carry a complete envelope (the request's model, `provider` `"test"`, a
  deterministic usage, and ids derived from the conversation, so tool-call
  ids stay unique across turns even with a new provider per request), so
  the code that stores and bills turns runs too.
- **`MockStreamingProvider`** replays one scripted stream per call and
  records every request. Use it for exact output and exact failures. A
  call past the end of the script returns `Error::Configuration`.
- **Failures surface where you script them.** `setup_error` makes the
  `chat_stream` call itself fail, the way a provider rejects a request with
  a 429. An `Error` inside a scripted stream arrives as a stream item after
  `chat_stream` returns `Ok`, the way an overload reported mid-stream does.
  `MockProvider::with_error` fails the call too.
- **Extraction and parallel tool calls** script with
  `MockResponse::structured(json)` and `.tool_calls([..])`. A
  structured response answers in whichever form the `ChatExtractor` asked for
  (the forced tool call, or JSON text), so a test does not depend on the
  extraction strategy. `.tool_calls([..])` puts several calls in one
  reply, the way a model calls tools in parallel. Both are on
  `MockProviderBuilder` too.

When the application builds its provider from a model address, give tests
a `Registry` with a mock under a key of its own, looked up with the
application's settings: a registered mock gets the same retries and
per-attempt timeouts as a built-in provider. The handler is a plain
async function, so a test calls it with the extractors axum would pass and
reads the whole SSE body:

```rust,ignore
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
```

The commit point needs both kinds of scripted failure: one before the
stream exists and one inside it, before any output. Both must keep their
status code:

```rust,ignore
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
```

And the failure after the commit point, scripted as a tool call followed by
a rejected follow-up call:

```rust,ignore
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
```

`MockStreamingProvider` is `Clone` and shares its script and records, so
keep a clone to inspect `requests()` after the handler ran.

### Testing tenant keys

The mocks have no credential scope, so they never refuse a request that
lacks a tenant's key (see
[Per-tenant API keys](#per-tenant-api-keys)). To test that path, build the
real adapter the application uses, with its scope, no key, and a
`ReplayTransport` (`allama-http`, as a dev-dependency) in place of the
network. A request without the tenant's credential is refused, and
nothing reaches the transport:

```rust
use allama::prelude::*;
use allama_http::{Credential, HttpRequestOptions, ReplayTransport};

async fn a_request_without_the_tenant_key_is_not_sent() {
    let transport = ReplayTransport::default();
    let provider = allama_openai::ResponsesProvider::builder()
        .credential_scope("tenant-openai")
        .http_transport(transport.clone())
        .build()
        .unwrap();
    // A key for another scope is not this provider's.
    let request = ChatRequest::new("gpt-5-mini").user("Hi").with_option(
        HttpRequestOptions::new().credential("tenant-anthropic", Credential::api_key("sk-other")),
    );

    let err = provider.chat(&request).await.unwrap_err();
    assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    assert!(provider.validate_chat_request(&request).is_err());
    assert!(transport.requests().is_empty(), "nothing was sent");
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    a_request_without_the_tenant_key_is_not_sent().await;
}
```
