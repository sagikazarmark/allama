//! `TracingProvider` spans, asserted on what `tracing-opentelemetry` exports.
//!
//! [`otel`] is the exporter harness; [`names`] pins the attribute names
//! against the checked-in [`semconv`] tables; [`summary`] covers the
//! call-scope summary and the serving provider; [`cancellation`] covers
//! `allama.response.cancelled` and when the span starts; [`stack`] covers
//! the recommended interactive stack.

#![cfg(all(feature = "tracing", feature = "mock"))]

mod cancellation;
mod names;
mod otel;
mod semconv;
mod stack;
mod summary;

use std::time::Duration;

use allama::chat::stream::StreamBlockType;
use allama::decision::{Answer, ChoiceAnswer, Question, YesNoAnswer};
use allama::mock::{
    ChatResponseBuilder, MockDecisionProvider, MockEmbeddingProvider, MockProvider, MockResponse,
    MockStreamEvent, MockStreamingProvider,
};
use allama::wrappers::{
    FallbackProvider, RetryBudget, RetryPolicy, RetryingProvider, TimeoutPolicy, TimeoutProvider,
    TracingContentConfig, TracingProvider,
};
use allama::{
    ChatProvider, ChatRequest, ChatStreamExt, ContentBlock, DecisionProvider, DecisionRequest,
    DecisionResponse, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, Error, Message,
    RequestField, StreamEvent, SystemPrompt, Usage, Warning,
};
use opentelemetry::Value;
use opentelemetry::trace::{SpanKind, Status};

use futures_util::StreamExt;
use otel::{ExportedSpan, event_value, export};

fn request() -> ChatRequest {
    ChatRequest::new("m")
        .system("secret instructions")
        .user("hi")
        .max_output_tokens(32)
}

#[tokio::test]
async fn chat_span_has_genai_attributes() {
    let provider = TracingProvider::new(
        MockProvider::with_response(
            ChatResponseBuilder::new()
                .text("hello")
                .usage_value(
                    Usage::new()
                        .input_tokens(10)
                        .output_tokens(5)
                        .cache_read_input_tokens(4),
                )
                .model("m-1")
                .id("resp-1")
                .warning(Warning::unsupported(RequestField::Seed, "seed ignored"))
                .build(),
        )
        .with_provider_name("gemini"),
    );
    let (result, exported) = export(async { provider.chat(&request()).await }).await;
    result.unwrap();
    let chat = exported.span("chat");
    assert_eq!(chat.name(), "chat m");
    assert_eq!(chat.status(), &Status::Unset);
    assert_eq!(
        chat.value("gen_ai.provider.name"),
        &Value::from("gcp.gemini")
    );
    assert_eq!(chat.value("gen_ai.request.model"), &Value::from("m"));
    assert_eq!(chat.value("gen_ai.request.max_tokens"), &Value::I64(32));
    assert_eq!(chat.value("gen_ai.usage.input_tokens"), &Value::I64(10));
    assert_eq!(chat.value("gen_ai.usage.output_tokens"), &Value::I64(5));
    assert_eq!(
        chat.value("gen_ai.usage.cache_read.input_tokens"),
        &Value::I64(4)
    );
    assert_eq!(chat.value("gen_ai.response.id"), &Value::from("resp-1"));
    assert_eq!(chat.value("gen_ai.response.model"), &Value::from("m-1"));
    assert_eq!(
        chat.value("gen_ai.response.finish_reasons"),
        &Value::from(r#"["stop"]"#)
    );
    assert_eq!(chat.value("allama.response.warnings"), &Value::I64(1));
    assert!(!chat.has("gen_ai.input.messages"), "content is opt-in");
    assert!(!chat.has("error.type"));
    assert!(
        !chat.has("gen_ai.request.stream"),
        "a unary call is not streamed"
    );

    let [warning] = chat.events() else {
        panic!("expected one warning event, got {:?}", chat.events());
    };
    assert_eq!(warning.name, "seed ignored");
    assert_eq!(
        event_value(warning, "allama.warning.kind"),
        Some(&Value::from("unsupported"))
    );
}

/// Sampling parameters export as the decimals they were written as
/// (`0.2`, not `0.20000000298023224`).
#[tokio::test]
async fn sampling_parameters_export_as_written() {
    let provider = TracingProvider::new(MockProvider::with_text("hi"));
    let request = ChatRequest::new("m")
        .user("x")
        .temperature(0.2)
        .top_p(0.9)
        .frequency_penalty(0.1)
        .presence_penalty(-0.3);
    let (_, exported) = export(async { provider.chat(&request).await }).await;
    let chat = exported.span("chat");
    assert_eq!(chat.value("gen_ai.request.temperature"), &Value::F64(0.2));
    assert_eq!(chat.value("gen_ai.request.top_p"), &Value::F64(0.9));
    assert_eq!(
        chat.value("gen_ai.request.frequency_penalty"),
        &Value::F64(0.1)
    );
    assert_eq!(
        chat.value("gen_ai.request.presence_penalty"),
        &Value::F64(-0.3)
    );
}

#[tokio::test]
async fn integer_values_beyond_i64_export_as_strings() {
    let provider = TracingProvider::new(MockProvider::with_text("hi"));
    let (_, exported) = export(async {
        provider
            .chat(&ChatRequest::new("m").user("x").seed(u64::MAX))
            .await
    })
    .await;
    assert_eq!(
        exported.span("chat").value("gen_ai.request.seed"),
        &Value::from(u64::MAX.to_string())
    );
}

#[tokio::test]
async fn content_capture_is_opt_in_and_redacted() {
    let provider = TracingProvider::new(MockProvider::with_text("out"))
        .with_content_capture(TracingContentConfig::all());
    let (_, exported) = export(async { provider.chat(&request()).await }).await;
    let chat = exported.span("chat");
    let text = |key| chat.value(key).as_str().into_owned();
    assert!(text("gen_ai.system_instructions").contains("secret instructions"));
    assert!(text("gen_ai.input.messages").contains("\"hi\""));
    assert!(text("gen_ai.output.messages").contains("out"));
}

#[tokio::test]
async fn content_capture_never_records_app_data() {
    const SENTINEL: &str = "app-data-sentinel";
    let request = ChatRequest::new("m")
        .system(SystemPrompt::new("instructions").with_app_data(SENTINEL, SENTINEL))
        .message(Message::user("hi").with_app_data(SENTINEL, SENTINEL))
        .message(
            Message::assistant_blocks(vec![ContentBlock::tool_call("c1", "f", "{}")])
                .with_app_data(SENTINEL, SENTINEL),
        )
        .message(Message::tool_result("c1", "f", "ok").with_app_data(SENTINEL, SENTINEL));
    let provider = TracingProvider::new(MockProvider::with_text("out"))
        .with_content_capture(TracingContentConfig::all());
    let (result, exported) = export(async { provider.chat(&request).await }).await;
    result.unwrap();
    let chat = exported.span("chat");
    let text = |key| chat.value(key).as_str().into_owned();
    assert!(text("gen_ai.system_instructions").contains("instructions"));
    assert!(text("gen_ai.input.messages").contains("\"ok\""));
    for attribute in chat.attributes() {
        assert!(
            !attribute.value.as_str().contains(SENTINEL),
            "{} recorded application data",
            attribute.key
        );
    }
}

#[tokio::test]
async fn stream_span_records_usage_and_time_to_first() {
    let provider = TracingProvider::new(MockStreamingProvider::from_responses([
        ChatResponseBuilder::new()
            .text("streamed")
            .usage(3, 2)
            .build(),
    ]));
    let (result, exported) = export(async {
        provider
            .chat_stream(&request())
            .await
            .unwrap()
            .collect_response()
            .await
    })
    .await;
    result.unwrap();
    let chat = exported.span("chat");
    assert_eq!(chat.value("gen_ai.usage.output_tokens"), &Value::I64(2));
    assert_eq!(chat.value("gen_ai.request.stream"), &Value::Bool(true));
    let chunk = seconds(chat, "gen_ai.response.time_to_first_chunk");
    let output = seconds(chat, "allama.response.time_to_first_output");
    assert!(
        chunk.is_sign_positive() && chunk <= output,
        "{chunk} > {output}"
    );
    assert!(
        !chat.has("allama.response.cancelled"),
        "a stream read to the end is not cancelled"
    );
}

/// A stream records each warning as one event, when it arrives, and the
/// warning count once.
#[tokio::test]
async fn stream_span_records_each_warning_once() {
    let provider = TracingProvider::new(MockStreamingProvider::from_responses([
        ChatResponseBuilder::new()
            .text("streamed")
            .warning(Warning::unsupported(RequestField::Seed, "seed ignored"))
            .warning(Warning::unsupported(RequestField::TopP, "top_p ignored"))
            .build(),
    ]));
    let (result, exported) = export(async {
        provider
            .chat_stream(&request())
            .await
            .unwrap()
            .collect_response()
            .await
    })
    .await;
    assert_eq!(result.unwrap().warnings.len(), 2);
    let chat = exported.span("chat");
    let names: Vec<_> = chat
        .events()
        .iter()
        .map(|event| event.name.as_ref())
        .collect();
    assert_eq!(names, ["seed ignored", "top_p ignored"]);
    let counts: Vec<_> = chat
        .attributes()
        .iter()
        .filter(|kv| kv.key.as_str() == "allama.response.warnings")
        .map(|kv| &kv.value)
        .collect();
    assert_eq!(counts, [&Value::I64(2)], "recorded once, with the total");
}

/// A delay the time-to-first measures must include.
const DELAY: Duration = Duration::from_millis(50);

/// A double attribute, in seconds.
fn seconds(span: ExportedSpan<'_>, key: &str) -> f64 {
    match span.value(key) {
        Value::F64(seconds) => *seconds,
        other => panic!("`{key}` exported as {other:?}, expected a double"),
    }
}

/// The events of a text response, paused for [`DELAY`] before the first
/// output event.
fn delayed_first_output() -> Vec<MockStreamEvent> {
    let mut events = Vec::new();
    let mut delayed = false;
    for event in ChatResponseBuilder::new()
        .text("late")
        .build()
        .into_stream_events()
    {
        if !delayed && event.is_output() {
            events.push(MockStreamEvent::delayed(DELAY));
            delayed = true;
        }
        events.push(MockStreamEvent::Event(event));
    }
    events
}

/// Under retry and fallback the stream is handed over only once its first
/// output exists, so a clock started then would read close to zero.
#[tokio::test]
async fn time_to_first_includes_retry_and_fallback() {
    let primary = RetryingProvider::new(
        TimeoutProvider::new(
            MockStreamingProvider::with_stream(delayed_first_output()),
            TimeoutPolicy::new()
                .stream_setup(Duration::from_secs(5))
                .stream_idle(Duration::from_secs(5)),
        ),
        RetryPolicy::new().base_delay(Duration::ZERO),
    );
    let provider = TracingProvider::new(FallbackProvider::new(
        primary,
        MockProvider::with_text("backup").with_provider_name("backup"),
    ));
    let (result, exported) = export(async {
        provider
            .chat_stream(&request())
            .await
            .unwrap()
            .collect_response()
            .await
    })
    .await;
    assert_eq!(result.unwrap().text().as_deref(), Some("late"));
    let chat = exported.span("chat");
    assert!(!chat.has("allama.fallback.used"));
    for key in [
        "gen_ai.response.time_to_first_chunk",
        "allama.response.time_to_first_output",
    ] {
        let elapsed = seconds(chat, key);
        assert!(elapsed >= DELAY.as_secs_f64(), "`{key}` = {elapsed}s");
    }
}

/// The clock starts when the call is issued, before the response headers.
#[tokio::test]
async fn time_to_first_includes_time_to_headers() {
    let provider = TracingProvider::new(MockProvider::new([MockResponse::delayed(
        DELAY,
        MockResponse::text("hi"),
    )]));
    let (result, exported) = export(async {
        provider
            .chat_stream(&request())
            .await
            .unwrap()
            .collect_response()
            .await
    })
    .await;
    result.unwrap();
    let chat = exported.span("chat");
    for key in [
        "gen_ai.response.time_to_first_chunk",
        "allama.response.time_to_first_output",
    ] {
        let elapsed = seconds(chat, key);
        assert!(elapsed >= DELAY.as_secs_f64(), "`{key}` = {elapsed}s");
    }
}

/// The first chunk is any event (here `ResponseStart`); the first output
/// is the first model-output event.
#[tokio::test]
async fn time_to_first_chunk_precedes_first_output() {
    let provider = TracingProvider::new(MockStreamingProvider::with_stream(delayed_first_output()));
    let (result, exported) = export(async {
        provider
            .chat_stream(&request())
            .await
            .unwrap()
            .collect_response()
            .await
    })
    .await;
    result.unwrap();
    let chat = exported.span("chat");
    let chunk = seconds(chat, "gen_ai.response.time_to_first_chunk");
    let output = seconds(chat, "allama.response.time_to_first_output");
    assert!(
        output - chunk >= DELAY.as_secs_f64(),
        "first chunk {chunk}s, first output {output}s"
    );
}

#[tokio::test]
async fn stream_dropped_before_its_end_is_cancelled_not_failed() {
    let provider = TracingProvider::new(MockStreamingProvider::with_stream([
        MockStreamEvent::Event(StreamEvent::response_start(Some("resp-1".into()), None)),
        MockStreamEvent::Event(StreamEvent::block_start(0, StreamBlockType::Text)),
        MockStreamEvent::Event(StreamEvent::text_delta(0, "first")),
        MockStreamEvent::Event(StreamEvent::text_delta(0, " never read")),
    ]));
    let ((), exported) = export(async {
        let mut stream = provider.chat_stream(&request()).await.unwrap();
        while let Some(event) = stream.next().await {
            if matches!(event.unwrap(), StreamEvent::TextDelta { .. }) {
                break;
            }
        }
        drop(stream);
    })
    .await;
    let chat = exported.span("chat");
    assert_eq!(chat.value("allama.response.cancelled"), &Value::Bool(true));
    assert_eq!(chat.status(), &Status::Unset);
    assert!(!chat.has("error.type"));
    // What arrived before the drop is still recorded.
    assert_eq!(chat.value("gen_ai.response.id"), &Value::from("resp-1"));
}

#[tokio::test]
async fn stream_span_records_retries() {
    let provider = TracingProvider::new(RetryingProvider::new(
        MockStreamingProvider::build(|b| {
            b.stream([MockStreamEvent::Error(Error::overloaded("busy"))])
                .response(ChatResponseBuilder::new().text("second").build())
        }),
        RetryPolicy::new()
            .max_attempts(4)
            .base_delay(Duration::ZERO),
    ));
    let (result, exported) = export(async {
        provider
            .chat_stream(&request())
            .await
            .unwrap()
            .collect_response()
            .await
    })
    .await;
    result.unwrap();
    let chat = exported.span("chat");
    assert_eq!(chat.value("allama.attempts"), &Value::I64(2));
    assert_eq!(chat.value("allama.retry.total_delay_ms"), &Value::I64(0));
    assert_eq!(chat.value("allama.retry.last_delay_ms"), &Value::I64(0));
    assert!(!chat.has("allama.retry.give_up_reason"));
    // Configuration is not on the span; the retry event carries it.
    assert!(!chat.has("allama.retry.max_attempts"));
    let [retry] = chat.events() else {
        panic!("expected one retry event, got {:?}", chat.events());
    };
    assert_eq!(
        event_value(retry, "allama.retry.max_attempts"),
        Some(&Value::I64(4))
    );
}

/// The server retry directive is recorded when it decided, in both
/// directions, and not when a custom classifier did.
#[tokio::test]
async fn span_records_the_server_retry_directive() {
    let conflict = |directive| {
        Error::invalid_request("HTTP 409")
            .with_status(409)
            .with_server_retry_directive(directive)
    };
    let policy = || RetryPolicy::new().base_delay(Duration::ZERO);

    let provider = TracingProvider::new(RetryingProvider::new(
        MockProvider::build(|b| b.error(conflict(true)).text("ok")),
        policy(),
    ));
    let (result, exported) = export(async { provider.chat(&request()).await }).await;
    result.unwrap();
    let chat = exported.span("chat");
    assert_eq!(
        chat.value("allama.retry.server_directive"),
        &Value::Bool(true)
    );
    assert_eq!(chat.value("allama.attempts"), &Value::I64(2));

    let provider = TracingProvider::new(RetryingProvider::new(
        MockProvider::with_error(
            Error::rate_limited("HTTP 429").with_server_retry_directive(false),
        ),
        policy(),
    ));
    let (result, exported) = export(async { provider.chat(&request()).await }).await;
    assert!(result.is_err());
    let chat = exported.span("chat");
    assert_eq!(
        chat.value("allama.retry.server_directive"),
        &Value::Bool(false)
    );
    assert!(!chat.has("allama.retry.total_delay_ms"));
    assert_eq!(
        chat.value("allama.retry.give_up_reason"),
        &Value::from("not_retryable")
    );

    let provider = TracingProvider::new(RetryingProvider::new(
        MockProvider::build(|b| b.error(conflict(true)).text("ok")),
        policy().retry_if(|_| true),
    ));
    let (result, exported) = export(async { provider.chat(&request()).await }).await;
    result.unwrap();
    assert!(!exported.span("chat").has("allama.retry.server_directive"));
}

/// A stream attempt that fails before output, after reporting input usage
/// the way Anthropic's `message_start` does.
fn uncommitted_attempt_with_usage() -> Vec<MockStreamEvent> {
    let mut metadata = StreamEvent::metadata();
    if let StreamEvent::ResponseMetadata { usage, .. } = &mut metadata {
        *usage = Some(Usage::new().input_tokens(10_000));
    }
    vec![
        MockStreamEvent::Event(StreamEvent::response_start(Some("resp-x".into()), None)),
        MockStreamEvent::Event(metadata),
        MockStreamEvent::Error(Error::overloaded("busy")),
    ]
}

/// `allama.attempts` counts every attempt, counting the first: a
/// setup failure and each stream discarded before output count as one,
/// whether the call is served or the attempts run out. The span's usage is
/// the served attempt's; what discarded attempts reported is not added.
#[tokio::test]
async fn retry_attempts_count_every_attempt_including_uncommitted_streams() {
    let policy = |max| {
        RetryPolicy::new()
            .max_attempts(max)
            .base_delay(Duration::ZERO)
            .max_delay(Duration::ZERO)
    };

    // Served on the fourth attempt.
    let served = MockStreamingProvider::build(|b| {
        b.setup_error(Error::overloaded("busy"))
            .stream(uncommitted_attempt_with_usage())
            .stream(uncommitted_attempt_with_usage())
            .response(ChatResponseBuilder::new().text("ok").usage(3, 1).build())
    });
    let provider = TracingProvider::new(RetryingProvider::new(served.clone(), policy(4)));
    let (result, exported) = export(async {
        provider
            .chat_stream(&request())
            .await?
            .collect_response()
            .await
    })
    .await;
    assert_eq!(result.unwrap().text().as_deref(), Some("ok"));
    assert_eq!(served.call_count(), 4);
    let chat = exported.span("chat");
    assert_eq!(chat.value("allama.attempts"), &Value::I64(4));
    assert_eq!(chat.events().len(), 3, "one retry event per failed attempt");
    assert_eq!(chat.value("gen_ai.usage.input_tokens"), &Value::I64(3));

    // Every attempt fails before output.
    let exhausted = MockStreamingProvider::build(|b| {
        b.stream(uncommitted_attempt_with_usage())
            .setup_error(Error::overloaded("busy"))
            .stream(uncommitted_attempt_with_usage())
    });
    let provider = TracingProvider::new(RetryingProvider::new(exhausted.clone(), policy(3)));
    let (result, exported) =
        export(async { provider.chat_stream(&request()).await.map(drop) }).await;
    assert!(result.is_err());
    assert_eq!(exhausted.call_count(), 3);
    let chat = exported.span("chat");
    assert_eq!(chat.value("allama.attempts"), &Value::I64(3));
    assert_eq!(
        chat.value("allama.retry.give_up_reason"),
        &Value::from("max_attempts")
    );
    assert_eq!(chat.events().len(), 2);
    assert!(!chat.has("gen_ai.usage.input_tokens"));
}

/// A budget denial is a span field and an attempt event, on chat (unary and
/// streaming), decision, and embedding spans; a call the budget did not limit records
/// neither.
#[tokio::test]
async fn span_records_a_retry_budget_denial() {
    let policy = |budget| RetryPolicy::new().base_delay(Duration::ZERO).budget(budget);
    let denied_event = |event: &opentelemetry::trace::Event| {
        (event.name == "allama.retry.budget_denied").then(|| event_attributes(event))
    };
    let expected = vec![
        (
            "allama.attempt.provider".to_owned(),
            Value::from("anthropic"),
        ),
        ("error.type".to_owned(), Value::from("overloaded")),
        ("allama.retry.attempt".to_owned(), Value::I64(1)),
    ];

    for call in Call::ALL {
        // A budget that cannot afford one transient retry.
        let mock = MockProvider::build(|b| b.error(Error::overloaded("busy")).text("ok"))
            .with_provider_name("anthropic");
        let provider = TracingProvider::new(RetryingProvider::new(
            mock.clone(),
            policy(RetryBudget::new().capacity(13)),
        ));
        let (result, exported) = export(call.run(&provider)).await;
        assert_eq!(result.unwrap_err().kind_name(), "overloaded", "{call:?}");
        assert_eq!(mock.call_count(), 1, "{call:?}");
        let chat = exported.span("chat");
        assert_eq!(
            chat.value("allama.retry.budget_denied"),
            &Value::Bool(true),
            "{call:?}"
        );
        assert_eq!(
            chat.value("allama.retry.give_up_reason"),
            &Value::from("budget"),
            "{call:?}"
        );
        assert_eq!(chat.value("allama.attempts"), &Value::I64(1), "{call:?}");
        assert!(!chat.has("allama.retry.total_delay_ms"), "{call:?}");
        let events: Vec<_> = chat.events().iter().filter_map(denied_event).collect();
        assert_eq!(events, std::slice::from_ref(&expected), "{call:?}");

        // With tokens to spare the retry runs and nothing is recorded.
        let provider = TracingProvider::new(RetryingProvider::new(
            MockProvider::build(|b| b.error(Error::overloaded("busy")).text("ok")),
            policy(RetryBudget::new()),
        ));
        let (result, exported) = export(call.run(&provider)).await;
        result.unwrap();
        let chat = exported.span("chat");
        assert!(!chat.has("allama.retry.budget_denied"), "{call:?}");
        assert!(!chat.has("allama.retry.give_up_reason"), "{call:?}");
        assert!(chat.events().iter().all(|e| denied_event(e).is_none()));
    }

    let decider =
        MockDecisionProvider::new([Err(Error::overloaded("busy"))]).with_provider_name("anthropic");
    let provider = TracingProvider::new(RetryingProvider::new(
        decider,
        policy(RetryBudget::new().capacity(0)),
    ));
    let request = DecisionRequest::new("d", "x").question("q", Question::yes_no("?"));
    let (result, exported) = export(async { provider.decide(&request).await }).await;
    assert!(result.is_err());
    let decision = exported.span("decision");
    assert_eq!(
        decision.value("allama.retry.budget_denied"),
        &Value::Bool(true)
    );
    let events: Vec<_> = decision.events().iter().filter_map(denied_event).collect();
    assert_eq!(events, std::slice::from_ref(&expected));

    let embedder = MockEmbeddingProvider::new([Err(Error::overloaded("busy"))])
        .with_provider_name("anthropic");
    let provider = TracingProvider::new(RetryingProvider::new(
        embedder,
        policy(RetryBudget::new().capacity(0)),
    ));
    let request = EmbeddingRequest::new("e").inputs(["a"]);
    let (result, exported) = export(async { provider.embed(&request).await }).await;
    assert!(result.is_err());
    let embeddings = exported.span("embeddings");
    assert_eq!(
        embeddings.value("allama.retry.budget_denied"),
        &Value::Bool(true)
    );
    let events: Vec<_> = embeddings
        .events()
        .iter()
        .filter_map(denied_event)
        .collect();
    assert_eq!(events, [expected]);
}

/// Upstream text that must never reach a span status description.
const SENTINEL: &str = "SENTINEL-1f9c my SSN is 123-45-6789";

/// An upstream failure whose response body carries [`SENTINEL`].
fn upstream_error() -> Error {
    Error::overloaded("upstream unavailable")
        .with_provider("mock")
        .with_status(503)
        .with_request_id("req-503")
        .with_body(format!("<html><body>{SENTINEL}</body></html>"))
}

/// A failed call's span: kind `CLIENT`, `error.type`, and status `Error`
/// described by the redacted log view of `error` without its `kind=`.
fn assert_failed_span(span: ExportedSpan<'_>, error: &Error) {
    assert_eq!(span.kind(), &SpanKind::Client);
    assert_eq!(span.value("error.type"), &Value::from(error.kind_name()));
    let log = error.as_log().to_string();
    let kind = format!("kind={} ", error.kind_name());
    let expected = log
        .strip_prefix(kind.as_str())
        .expect("log view starts with the kind");
    assert_eq!(span.status(), &Status::error(expected.to_owned()));
    let Status::Error { description } = span.status() else {
        unreachable!()
    };
    assert!(
        !description.contains(SENTINEL),
        "response body leaked into the status description: {description}"
    );
    assert!(
        !description.contains(&kind),
        "status description repeats `error.type`: {description}"
    );
}

#[tokio::test]
async fn failed_chat_exports_error_status() {
    let provider = TracingProvider::new(MockProvider::with_error(upstream_error()));
    let (result, exported) = export(async { provider.chat(&request()).await }).await;
    assert!(result.is_err());
    assert_failed_span(exported.span("chat"), &upstream_error());
}

#[tokio::test]
async fn stream_setup_failure_exports_error_status() {
    let provider = TracingProvider::new(MockProvider::with_error(upstream_error()));
    let (result, exported) = export(async { provider.chat_stream(&request()).await.err() }).await;
    assert!(result.is_some());
    assert_failed_span(exported.span("chat"), &upstream_error());
}

#[tokio::test]
async fn mid_stream_error_exports_error_status() {
    let provider = TracingProvider::new(MockStreamingProvider::with_stream([
        MockStreamEvent::Event(StreamEvent::response_start(None, None)),
        MockStreamEvent::Event(StreamEvent::block_start(0, StreamBlockType::Text)),
        MockStreamEvent::Event(StreamEvent::text_delta(0, "partial")),
        MockStreamEvent::Error(upstream_error()),
    ]));
    let (result, exported) = export(async {
        provider
            .chat_stream(&request())
            .await
            .unwrap()
            .collect_response()
            .await
    })
    .await;
    assert!(result.is_err());
    let chat = exported.span("chat");
    assert_failed_span(chat, &upstream_error());
    assert!(
        !chat.has("allama.response.cancelled"),
        "a stream that failed is not cancelled"
    );
}

#[tokio::test]
async fn failed_embedding_exports_error_status() {
    let provider = TracingProvider::new(MockEmbeddingProvider::new([Err(upstream_error())]));
    let (result, exported) =
        export(async { provider.embed(&EmbeddingRequest::new("e").input("x")).await }).await;
    assert!(result.is_err());
    assert_failed_span(exported.span("embeddings"), &upstream_error());
}

#[tokio::test]
async fn failed_decision_exports_error_status() {
    let provider = TracingProvider::new(MockDecisionProvider::new([Err(upstream_error())]));
    let request = DecisionRequest::new("d", "Win a cruise!")
        .question("spam", Question::yes_no("Is this spam?"));
    let (result, exported) = export(async { provider.decide(&request).await }).await;
    assert!(result.is_err());
    assert_failed_span(exported.span("decision"), &upstream_error());
}

#[tokio::test]
async fn recovered_failures_leave_status_unset() {
    let provider = TracingProvider::new(RetryingProvider::new(
        MockProvider::build(|b| b.error(upstream_error()).text("second")),
        RetryPolicy::new().base_delay(Duration::ZERO),
    ));
    let (result, exported) = export(async { provider.chat(&request()).await }).await;
    result.unwrap();
    let chat = exported.span("chat");
    assert_eq!(chat.value("allama.attempts"), &Value::I64(2));
    assert_eq!(chat.status(), &Status::Unset);
    assert!(!chat.has("error.type"));
}

#[tokio::test]
async fn retry_and_fallback_annotate_outer_span() {
    let retrying = RetryingProvider::new(
        MockProvider::build(|b| {
            b.error(Error::timeout("slow"))
                .error(Error::timeout("slow"))
        }),
        RetryPolicy::new()
            .max_attempts(2)
            .base_delay(Duration::ZERO)
            .max_delay(Duration::ZERO),
    );
    let provider = TracingProvider::new(FallbackProvider::new(
        retrying,
        MockProvider::with_text("backup").with_provider_name("backup"),
    ));
    let (result, exported) = export(async { provider.chat(&request()).await }).await;
    assert_eq!(result.unwrap().provider.as_deref(), Some("backup"));
    let chat = exported.span("chat");
    assert_eq!(chat.value("allama.attempts"), &Value::I64(3));
    assert_eq!(
        chat.value("allama.retry.give_up_reason"),
        &Value::from("max_attempts")
    );
    assert_eq!(chat.value("allama.fallback.used"), &Value::Bool(true));
    assert_eq!(chat.value("allama.fallback.attempts"), &Value::I64(1));
    assert_eq!(
        chat.value("allama.fallback.error_type"),
        &Value::from("timeout")
    );
    assert_eq!(
        chat.value("allama.response.provider"),
        &Value::from("backup")
    );
    // The addressed provider, set when the span starts and never again.
    assert_eq!(chat.value("gen_ai.provider.name"), &Value::from("mock"));
    assert_eq!(chat.value("allama.request.provider"), &Value::from("mock"));
}

/// How a test calls the provider: unary `chat` or a collected stream.
#[derive(Debug, Clone, Copy)]
enum Call {
    Unary,
    Stream,
}

impl Call {
    const ALL: [Call; 2] = [Call::Unary, Call::Stream];

    async fn run<P: ChatProvider>(self, provider: &P) -> allama::Result<allama::ChatResponse> {
        match self {
            Call::Unary => provider.chat(&request()).await,
            Call::Stream => {
                provider
                    .chat_stream(&request())
                    .await?
                    .collect_response()
                    .await
            }
        }
    }
}

/// The attributes of an event allama recorded, by name, without the ones
/// the exporting layer adds.
/// The resilience events of a span: everything but warning events, whose
/// failover warnings the tests check apart.
fn resilience_events(events: &[opentelemetry::trace::Event]) -> Vec<&opentelemetry::trace::Event> {
    events
        .iter()
        .filter(|event| !is_warning_event(event))
        .collect()
}

/// Whether `event` is a warning event (`allama::warning::emit`).
fn is_warning_event(event: &opentelemetry::trace::Event) -> bool {
    event
        .attributes
        .iter()
        .any(|kv| kv.key.as_str() == "target" && kv.value.as_str().starts_with("allama::warning::"))
}

/// The kinds of the warning events of a span.
fn warning_kinds(events: &[opentelemetry::trace::Event]) -> Vec<String> {
    events
        .iter()
        .filter(|event| is_warning_event(event))
        .filter_map(|event| {
            event
                .attributes
                .iter()
                .find(|kv| kv.key.as_str() == "allama.warning.kind")
                .map(|kv| kv.value.as_str().into_owned())
        })
        .collect()
}

fn event_attributes(event: &opentelemetry::trace::Event) -> Vec<(String, Value)> {
    event
        .attributes
        .iter()
        .filter(|kv| !otel::LAYER_EVENT_ATTRIBUTES.contains(&kv.key.as_str()))
        .map(|kv| (kv.key.to_string(), kv.value.clone()))
        .collect()
}

/// `gen_ai.provider.name` keeps the addressed provider; the serving one is
/// `allama.response.provider`, in the same (semconv) value space.
#[tokio::test]
async fn fallback_served_call_records_the_serving_provider() {
    for call in Call::ALL {
        let provider = TracingProvider::new(FallbackProvider::new(
            MockProvider::with_error(Error::overloaded("down")).with_provider_name("xai"),
            MockProvider::with_text("backup").with_provider_name("mistral"),
        ));
        let (result, exported) = export(call.run(&provider)).await;
        assert_eq!(
            result.unwrap().text().as_deref(),
            Some("backup"),
            "{call:?}"
        );
        let chat = exported.span("chat");
        assert_eq!(
            chat.value("gen_ai.provider.name"),
            &Value::from("x_ai"),
            "{call:?}"
        );
        assert_eq!(
            chat.value("allama.request.provider"),
            &Value::from("x_ai"),
            "{call:?}"
        );
        assert_eq!(
            chat.value("allama.response.provider"),
            &Value::from("mistral_ai"),
            "{call:?}"
        );
        assert_eq!(
            chat.value("allama.fallback.attempts"),
            &Value::I64(1),
            "{call:?}"
        );
        assert_eq!(chat.value("allama.attempts"), &Value::I64(2), "{call:?}");
        // Warnings and resilience events are `tracing` events (`warning-events`).
        if cfg!(feature = "warning-events") {
            assert_eq!(warning_kinds(chat.events()), ["fallback"], "{call:?}");
            let [fallback] = resilience_events(chat.events())[..] else {
                panic!("{call:?}: expected one event, got {:?}", chat.events());
            };
            assert_eq!(fallback.name, "allama.fallback", "{call:?}");
            assert_eq!(
                event_attributes(fallback),
                [
                    ("allama.attempt.provider".into(), Value::from("x_ai")),
                    ("error.type".into(), Value::from("overloaded")),
                    ("allama.fallback.provider".into(), Value::from("mistral_ai")),
                ],
                "{call:?}"
            );
        }
    }
}

#[tokio::test]
async fn a_call_the_addressed_provider_serves_keeps_its_name() {
    for call in Call::ALL {
        let provider = TracingProvider::new(
            MockProvider::with_response(ChatResponseBuilder::new().text("hi").build())
                .with_provider_name("gemini"),
        );
        let (result, exported) = export(call.run(&provider)).await;
        result.unwrap();
        let chat = exported.span("chat");
        assert_eq!(
            chat.value("gen_ai.provider.name"),
            &Value::from("gcp.gemini")
        );
        assert_eq!(
            chat.value("allama.request.provider"),
            &Value::from("gcp.gemini")
        );
        assert_eq!(
            chat.value("allama.response.provider"),
            &Value::from("gcp.gemini")
        );
        assert_eq!(chat.value("allama.attempts"), &Value::I64(1), "{call:?}");
        assert!(!chat.has("allama.fallback.used"), "{call:?}");
        assert!(!chat.has("allama.fallback.attempts"), "{call:?}");
    }
}

/// With a retry layer on each side of a fallback, each failed attempt is
/// its own event, in order, with its provider, error type, and delay.
#[tokio::test]
async fn every_retry_and_fallback_attempt_is_an_event() {
    // No backoff or jitter, so each delay is exactly its hint.
    let policy = RetryPolicy::new()
        .max_attempts(2)
        .base_delay(Duration::ZERO)
        .max_delay(Duration::from_secs(1))
        .jitter(false);
    let hinted = |error: Error, millis| error.with_retry_after(Duration::from_millis(millis));
    for call in Call::ALL {
        let primary = RetryingProvider::new(
            MockProvider::new([
                hinted(Error::overloaded("busy"), 7),
                hinted(Error::timeout("slow"), 7),
            ])
            .with_provider_name("anthropic"),
            policy.clone(),
        );
        let backup = RetryingProvider::new(
            MockProvider::build(|b| {
                b.error(hinted(Error::rate_limited("slow down"), 3))
                    .text("ok")
            })
            .with_provider_name("xai"),
            policy.clone(),
        );
        let provider = TracingProvider::new(FallbackProvider::new(primary, backup));
        let (result, exported) = export(call.run(&provider)).await;
        assert_eq!(result.unwrap().text().as_deref(), Some("ok"), "{call:?}");
        let chat = exported.span("chat");

        // Warnings and resilience events are `tracing` events (`warning-events`).
        if cfg!(feature = "warning-events") {
            assert_eq!(warning_kinds(chat.events()), ["fallback"], "{call:?}");
            let events: Vec<_> = resilience_events(chat.events())
                .into_iter()
                .map(|event| (event.name.to_string(), event_attributes(event)))
                .collect();
            let retry = |provider: &str, error: &str, delay_ms| {
                (
                    "allama.retry".to_owned(),
                    vec![
                        (
                            "allama.attempt.provider".to_owned(),
                            Value::from(provider.to_owned()),
                        ),
                        ("error.type".to_owned(), Value::from(error.to_owned())),
                        ("allama.retry.attempt".to_owned(), Value::I64(1)),
                        ("allama.retry.max_attempts".to_owned(), Value::I64(2)),
                        ("allama.retry.delay_ms".to_owned(), Value::I64(delay_ms)),
                    ],
                )
            };
            assert_eq!(
                events,
                [
                    retry("anthropic", "overloaded", 7),
                    (
                        "allama.fallback".to_owned(),
                        vec![
                            (
                                "allama.attempt.provider".to_owned(),
                                Value::from("anthropic")
                            ),
                            ("error.type".to_owned(), Value::from("timeout")),
                            ("allama.fallback.provider".to_owned(), Value::from("x_ai")),
                        ],
                    ),
                    retry("x_ai", "rate_limited", 3),
                ],
                "{call:?}"
            );
        }
        assert_eq!(
            chat.value("gen_ai.provider.name"),
            &Value::from("anthropic")
        );
        assert_eq!(
            chat.value("allama.request.provider"),
            &Value::from("anthropic")
        );
        assert_eq!(chat.value("allama.response.provider"), &Value::from("x_ai"));
        assert_eq!(chat.value("allama.fallback.attempts"), &Value::I64(1));
    }
}

/// A chain of fallbacks counts every provider that failed before the one
/// that served, and attributes the call to that one.
#[tokio::test]
async fn fallback_chain_counts_every_failed_provider() {
    for call in Call::ALL {
        let provider = TracingProvider::new(FallbackProvider::new(
            MockProvider::with_error(Error::overloaded("down")).with_provider_name("first"),
            FallbackProvider::new(
                MockProvider::with_error(Error::timeout("slow")).with_provider_name("second"),
                MockProvider::with_text("third").with_provider_name("third"),
            ),
        ));
        let (result, exported) = export(call.run(&provider)).await;
        let response = result.unwrap();
        let attempts = response
            .provider_data
            .get::<allama::wrappers::FallbackAttempts>()
            .unwrap()
            .unwrap();
        let providers: Vec<_> = attempts.iter().map(|a| a.provider.as_str()).collect();
        assert_eq!(providers, ["first", "second"], "{call:?}");
        let chat = exported.span("chat");
        assert_eq!(
            chat.value("allama.fallback.attempts"),
            &Value::I64(2),
            "{call:?}"
        );
        assert_eq!(chat.value("allama.attempts"), &Value::I64(3), "{call:?}");
        assert_eq!(
            chat.value("allama.response.provider"),
            &Value::from("third"),
            "{call:?}"
        );
        assert_eq!(
            chat.value("gen_ai.provider.name"),
            &Value::from("first"),
            "{call:?}"
        );
        assert_eq!(chat.value("allama.request.provider"), &Value::from("first"));
    }
}

/// When every provider fails, all of them count. Nothing served, and these
/// errors name no provider (as a wrapper's own errors do not), so the
/// response provider is the addressed one.
#[tokio::test]
async fn failed_fallback_counts_both_providers() {
    for call in Call::ALL {
        let provider = TracingProvider::new(FallbackProvider::new(
            MockProvider::with_error(Error::overloaded("down")).with_provider_name("primary"),
            MockProvider::with_error(Error::timeout("slow")).with_provider_name("backup"),
        ));
        let (result, exported) = export(call.run(&provider)).await;
        assert!(result.is_err(), "{call:?}");
        let chat = exported.span("chat");
        assert_eq!(
            chat.value("allama.fallback.attempts"),
            &Value::I64(2),
            "{call:?}"
        );
        assert_eq!(chat.value("gen_ai.provider.name"), &Value::from("primary"));
        assert_eq!(
            chat.value("allama.response.provider"),
            &Value::from("primary"),
            "{call:?}"
        );
        assert_eq!(chat.value("allama.attempts"), &Value::I64(2), "{call:?}");
        // The error the backup ended the call with, not `fallback`.
        assert_eq!(chat.value("error.type"), &Value::from("timeout"));
    }
}

/// When every side fails, the response provider is the
/// one whose error ended the call, as `error.type` is, not the addressed
/// primary.
#[tokio::test]
async fn failed_fallback_names_the_provider_whose_error_ended_it() {
    for call in Call::ALL {
        let provider = TracingProvider::new(FallbackProvider::new(
            MockProvider::with_error(Error::transport("refused").with_provider("ollama"))
                .with_provider_name("ollama"),
            MockProvider::with_error(Error::auth("bad key").with_provider("anthropic"))
                .with_provider_name("anthropic"),
        ));
        let (result, exported) = export(call.run(&provider)).await;
        assert_eq!(
            result.unwrap_err().context().unwrap().provider.as_deref(),
            Some("anthropic")
        );
        let chat = exported.span("chat");
        assert_eq!(chat.value("gen_ai.provider.name"), &Value::from("ollama"));
        assert_eq!(
            chat.value("allama.response.provider"),
            &Value::from("anthropic"),
            "{call:?}"
        );
        assert_eq!(chat.value("error.type"), &Value::from("auth"), "{call:?}");
        assert_eq!(
            chat.value("allama.fallback.error_type"),
            &Value::from("transport"),
            "{call:?}"
        );
    }
}

#[tokio::test]
async fn embedding_span() {
    let provider = TracingProvider::new(MockEmbeddingProvider::new([Ok(EmbeddingResponse::new(
        vec![vec![0.1]],
    )
    .usage(Usage::new().input_tokens(2)))]));
    let (result, exported) = export(async {
        provider
            .embed(&EmbeddingRequest::new("e").input("x").dimensions(8))
            .await
    })
    .await;
    result.unwrap();
    let embeddings = exported.span("embeddings");
    assert_eq!(embeddings.name(), "embeddings e");
    assert_eq!(
        embeddings.value("gen_ai.embeddings.dimension.count"),
        &Value::I64(8)
    );
    assert_eq!(
        embeddings.value("allama.embeddings.input_count"),
        &Value::I64(1)
    );
    assert_eq!(
        embeddings.value("gen_ai.usage.input_tokens"),
        &Value::I64(2)
    );
}

/// An embedding call that retries once and then fails over records the
/// same resilience fields and attempt events as chat.
#[tokio::test]
async fn embedding_span_with_retry_and_fallback() {
    let hinted = |error: Error| error.with_retry_after(Duration::from_millis(5));
    let retrying = RetryingProvider::new(
        MockEmbeddingProvider::new([
            Err(hinted(Error::overloaded("busy"))),
            Err(hinted(Error::timeout("slow"))),
        ])
        .with_provider_name("xai"),
        // No backoff or jitter, so the delay is exactly the hint.
        RetryPolicy::new()
            .max_attempts(2)
            .base_delay(Duration::ZERO)
            .max_delay(Duration::from_secs(1))
            .jitter(false),
    );
    let provider = TracingProvider::new(FallbackProvider::new(
        retrying,
        MockEmbeddingProvider::new([Ok(EmbeddingResponse::new(vec![vec![0.1]])
            .model("served-model")
            .usage(Usage::new().input_tokens(3)))])
        .with_provider_name("mistral"),
    ));
    let (result, exported) =
        export(async { provider.embed(&EmbeddingRequest::new("e").input("x")).await }).await;
    let response = result.unwrap();
    let attempts = response
        .provider_data
        .get::<allama::wrappers::FallbackAttempts>()
        .unwrap()
        .unwrap();
    assert_eq!(attempts.len(), 1);

    let embeddings = exported.span("embeddings");
    assert_eq!(embeddings.status(), &Status::Unset);
    assert_eq!(
        embeddings.value("allama.request.provider"),
        &Value::from("x_ai")
    );
    assert_eq!(embeddings.value("allama.attempts"), &Value::I64(3));
    assert_eq!(
        embeddings.value("allama.retry.total_delay_ms"),
        &Value::I64(5)
    );
    assert_eq!(
        embeddings.value("allama.retry.last_delay_ms"),
        &Value::I64(5)
    );
    assert_eq!(
        embeddings.value("allama.retry.give_up_reason"),
        &Value::from("max_attempts")
    );
    assert_eq!(embeddings.value("allama.fallback.used"), &Value::Bool(true));
    assert_eq!(
        embeddings.value("allama.response.provider"),
        &Value::from("mistral_ai")
    );
    assert_eq!(
        embeddings.value("gen_ai.provider.name"),
        &Value::from("x_ai")
    );
    assert_eq!(
        embeddings.value("allama.fallback.error_type"),
        &Value::from("timeout")
    );
    assert_eq!(embeddings.value("allama.fallback.attempts"), &Value::I64(1));
    assert_eq!(
        embeddings.value("gen_ai.response.model"),
        &Value::from("served-model")
    );

    // Warnings and resilience events are `tracing` events (`warning-events`).
    if cfg!(feature = "warning-events") {
        assert_eq!(warning_kinds(embeddings.events()), ["fallback"]);
        let events: Vec<_> = resilience_events(embeddings.events())
            .into_iter()
            .map(|event| (event.name.to_string(), event_attributes(event)))
            .collect();
        assert_eq!(
            events,
            [
                (
                    "allama.retry".to_owned(),
                    vec![
                        ("allama.attempt.provider".to_owned(), Value::from("x_ai")),
                        ("error.type".to_owned(), Value::from("overloaded")),
                        ("allama.retry.attempt".to_owned(), Value::I64(1)),
                        ("allama.retry.max_attempts".to_owned(), Value::I64(2)),
                        ("allama.retry.delay_ms".to_owned(), Value::I64(5)),
                    ],
                ),
                (
                    "allama.fallback".to_owned(),
                    vec![
                        ("allama.attempt.provider".to_owned(), Value::from("x_ai")),
                        ("error.type".to_owned(), Value::from("timeout")),
                        (
                            "allama.fallback.provider".to_owned(),
                            Value::from("mistral_ai")
                        ),
                    ],
                ),
            ]
        );
    }
}

/// When both sides of an embedding fallback fail, both count.
#[tokio::test]
async fn failed_embedding_fallback_counts_both_providers() {
    let provider = TracingProvider::new(FallbackProvider::new(
        MockEmbeddingProvider::with_error(Error::overloaded("down")),
        MockEmbeddingProvider::with_error(Error::timeout("slow")),
    ));
    let (result, exported) =
        export(async { provider.embed(&EmbeddingRequest::new("e").input("x")).await }).await;
    assert!(result.is_err());
    let embeddings = exported.span("embeddings");
    assert_eq!(embeddings.value("allama.fallback.attempts"), &Value::I64(2));
    assert_eq!(embeddings.value("allama.attempts"), &Value::I64(2));
    assert_eq!(
        embeddings.value("allama.response.provider"),
        &Value::from("mock")
    );
    assert_eq!(embeddings.value("error.type"), &Value::from("timeout"));
}

#[tokio::test]
async fn decision_span_with_retry_and_fallback() {
    let answered = || {
        DecisionResponse::new([
            ("spam", Answer::from(YesNoAnswer::new(0.1))),
            (
                "lang",
                ChoiceAnswer::new("en", [("en", 0.9), ("de", 0.1)]).into(),
            ),
        ])
        .model("served-model")
        .id("resp-1")
        .request_id("req-1")
        .provider("backup")
        .usage(Usage::new().input_tokens(12).output_tokens(1))
        .warning(Warning::unsupported(RequestField::Seed, "demo"))
    };
    let retrying = RetryingProvider::new(
        MockDecisionProvider::new([Err(Error::timeout("slow")), Err(Error::timeout("slow"))]),
        RetryPolicy::new()
            .max_attempts(2)
            .base_delay(Duration::ZERO)
            .max_delay(Duration::ZERO),
    );
    let provider = TracingProvider::new(FallbackProvider::new(
        retrying,
        MockDecisionProvider::new([Ok(answered())]).with_provider_name("backup"),
    ));
    let request = DecisionRequest::new("d", "Win a cruise!")
        .question("spam", Question::yes_no("Is this spam?"))
        .question(
            "lang",
            Question::choice("Language?")
                .bare_option("en")
                .bare_option("de"),
        );
    let (result, exported) = export(async { provider.decide(&request).await }).await;
    result.unwrap();
    let decision = exported.span("decision");
    assert_eq!(decision.name(), "decision d");
    assert_eq!(decision.value("gen_ai.request.model"), &Value::from("d"));
    assert_eq!(
        decision.value("allama.decision.question_count"),
        &Value::I64(2)
    );
    assert_eq!(decision.value("gen_ai.usage.input_tokens"), &Value::I64(12));
    assert_eq!(decision.value("gen_ai.usage.output_tokens"), &Value::I64(1));
    assert_eq!(
        decision.value("gen_ai.response.model"),
        &Value::from("served-model")
    );
    assert_eq!(
        decision.value("allama.response.provider"),
        &Value::from("backup")
    );
    assert_eq!(
        decision.value("allama.response.request_id"),
        &Value::from("req-1")
    );
    assert_eq!(decision.value("gen_ai.response.id"), &Value::from("resp-1"));
    // The backup's warning, and the failover.
    assert_eq!(decision.value("allama.response.warnings"), &Value::I64(2));
    assert_eq!(decision.value("allama.attempts"), &Value::I64(3));
    assert_eq!(
        decision.value("allama.retry.give_up_reason"),
        &Value::from("max_attempts")
    );
    assert_eq!(decision.value("allama.fallback.used"), &Value::Bool(true));
    assert_eq!(
        decision.value("allama.fallback.error_type"),
        &Value::from("timeout")
    );
    assert_eq!(decision.value("allama.fallback.attempts"), &Value::I64(1));
    assert_eq!(decision.value("gen_ai.provider.name"), &Value::from("mock"));
    assert_eq!(
        decision.value("allama.request.provider"),
        &Value::from("mock")
    );
    let names: Vec<_> = decision
        .events()
        .iter()
        .map(|event| event.name.as_ref())
        .collect();
    assert_eq!(
        names,
        [
            "allama.retry",
            "allama.fallback",
            "demo",
            "failed with `timeout`; served by `backup`"
        ]
    );
}

/// A decision response without a provider is attributed to the fallback
/// that served it.
#[tokio::test]
async fn fallback_decision_without_a_provider_reports_the_fallback() {
    let provider = TracingProvider::new(FallbackProvider::new(
        MockDecisionProvider::new([Err(Error::overloaded("down"))]).with_provider_name("primary"),
        MockDecisionProvider::new([Ok(DecisionResponse::new([(
            "spam",
            Answer::from(YesNoAnswer::new(0.1)),
        )]))])
        .with_provider_name("backup"),
    ));
    let request = DecisionRequest::new("d", "Win a cruise!")
        .question("spam", Question::yes_no("Is this spam?"));
    let (result, exported) = export(async { provider.decide(&request).await }).await;
    assert_eq!(result.unwrap().provider.as_deref(), Some("backup"));
    let decision = exported.span("decision");
    assert_eq!(
        decision.value("allama.response.provider"),
        &Value::from("backup")
    );
    assert_eq!(
        decision.value("gen_ai.provider.name"),
        &Value::from("primary")
    );
}

/// A provider whose transport reports the server it sends to, as
/// allama-http's reqwest transport does.
struct Hosted<P> {
    inner: P,
    host: &'static str,
}

impl<P: ChatProvider> ChatProvider for Hosted<P> {
    type Stream = P::Stream;

    async fn chat(&self, request: &ChatRequest) -> allama::Result<allama::ChatResponse> {
        allama::wrappers::call_scope::record(allama::wrappers::call_scope::CallFact::server(
            self.host,
            Some(8443),
        ));
        self.inner.chat(request).await
    }

    async fn chat_stream(&self, request: &ChatRequest) -> allama::Result<Self::Stream> {
        self.inner.chat_stream(request).await
    }

    allama::wrappers::forward_chat_questions!(inner);
}

impl<P: allama::ProviderIdentity> allama::ProviderIdentity for Hosted<P> {
    allama::wrappers::forward_provider_identity!(inner);
}

/// The span names the server that served, also
/// after a failover between two hosts of the same provider.
#[tokio::test]
async fn spans_record_the_server_that_served() {
    let provider = TracingProvider::new(FallbackProvider::new(
        Hosted {
            inner: MockProvider::with_error(Error::overloaded("busy")),
            host: "gpu1.internal",
        },
        Hosted {
            inner: MockProvider::with_text("ok"),
            host: "gpu2.internal",
        },
    ));
    let (result, exported) = export(async { provider.chat(&request()).await }).await;
    result.unwrap();
    let chat = exported.span("chat");
    assert_eq!(chat.value("server.address"), &Value::from("gpu2.internal"));
    assert_eq!(chat.value("server.port"), &Value::I64(8443));
}

/// The span hook adds the application's attributes
/// to the GenAI span itself.
#[tokio::test]
async fn the_span_hook_sets_the_conversation_id() {
    #[derive(Clone)]
    struct Conversation(&'static str);

    let provider = TracingProvider::new(MockProvider::with_text("ok")).on_span(|span, request| {
        if let allama::wrappers::TracedRequest::Chat(request) = request
            && let Some(Conversation(id)) = request.option::<Conversation>()
        {
            span.record("gen_ai.conversation.id", *id);
        }
    });
    let request = request().with_opaque_option(Conversation("conv_42"));
    let (result, exported) = export(async { provider.chat(&request).await }).await;
    result.unwrap();
    assert_eq!(
        exported.span("chat").value("gen_ai.conversation.id"),
        &Value::from("conv_42")
    );
}
