//! Warning events (`warning-events`): each warning is emitted once per
//! occurrence on every path, codes decide whether it recurs at `WARN`, and
//! the events of retried attempts are held back. The attempt scopes that
//! hold them are unit-tested in `warning/attempt.rs`.
//!
//! The deduplication state is process-wide: every test addresses a model
//! of its own and reads only that model's events.

#![cfg(all(feature = "mock", feature = "warning-events"))]

#[path = "support/warning_capture.rs"]
mod support;

use std::time::Duration;

use allama::adapter::chat::{self, Prepared};
use allama::mock::{
    ChatResponseBuilder, FunctionProvider, MockProvider, MockStreamEvent, MockStreamingProvider,
};
use allama::warning;
use allama::wrappers::{FallbackProvider, RetryPolicy, RetryingProvider};
use allama::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, ChatStream,
    ChatStreamExt, ContentBlock, Error, FinishReason, Message, ProviderIdentity, ReasoningEffort,
    Result, StreamEvent, Subject, Warning, WarningKind,
};
use support::{Event, capture, levels};

/// A request with a reasoning hint, which the providers below cannot take.
fn hinted(model: &str) -> ChatRequest {
    ChatRequest::new(model)
        .user("q")
        .reasoning(ReasoningEffort::High)
}

fn fast() -> RetryPolicy {
    RetryPolicy::new()
        .base_delay(Duration::ZERO)
        .max_delay(Duration::ZERO)
        .no_budget()
}

const NO_REASONING: (ChatCapability, CapabilitySupport) = (
    ChatCapability::ReasoningConfig,
    CapabilitySupport::Unsupported,
);

fn messages(events: &[Event]) -> Vec<&str> {
    events.iter().map(|event| event.message.as_str()).collect()
}

/// A wrapper built on `prepare`, like a capability override: it reports a
/// hint of its own and passes the inner response through its `finish`.
struct Override<P>(P);

impl<P: ProviderIdentity> ProviderIdentity for Override<P> {
    fn provider_name(&self) -> &str {
        self.0.provider_name()
    }

    fn reads_namespace(&self, namespace: &str) -> bool {
        self.0.reads_namespace(namespace)
    }
}

impl<P: ChatProvider> ChatProvider for Override<P> {
    type Stream = ChatStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let mut prepared: Prepared<'_> = chat::prepare(self, request)?;
        prepared.mark_sent();
        let response = self.0.chat(prepared.request()).await?;
        Ok(prepared.finish(response))
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        let mut prepared = chat::prepare(self, request)?;
        prepared.mark_sent();
        let stream = self.0.chat_stream(prepared.request()).await?;
        Ok(prepared.finish_stream(stream))
    }

    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        if capability == ChatCapability::ReasoningConfig {
            return CapabilitySupport::Unsupported;
        }
        self.0.chat_capability(model, capability)
    }
}

/// A content warning, uncoded: about this call alone.
fn content_warning(text: &str) -> Warning {
    Warning::new(
        WarningKind::Dropped,
        Subject::Content {
            message: 0,
            block: Some(0),
        },
        text,
    )
}

#[test]
fn unary_and_stream_calls_emit_each_warning_once() {
    let ((), events) = capture(async {
        let unary =
            MockProvider::with_text("ok").with_chat_capability(NO_REASONING.0, NO_REASONING.1);
        let response = unary.chat(&hinted("once-unary")).await.unwrap();
        assert_eq!(response.warnings.len(), 1);

        let stream = MockStreamingProvider::with_text("ok")
            .with_chat_capability(NO_REASONING.0, NO_REASONING.1);
        let response = stream
            .chat_stream(&hinted("once-stream"))
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        assert_eq!(response.warnings.len(), 1);
    });
    for model in ["once-unary", "once-stream"] {
        let model_events = events.for_model(model);
        assert_eq!(model_events.len(), 1, "{model}: {model_events:?}");
        assert_eq!(model_events[0].target, "allama::warning::unsupported");
        assert_eq!(
            model_events[0].code.as_deref(),
            Some("unsupported.reasoning")
        );
    }
}

/// A retried attempt's warnings are dropped: the call emits each warning
/// once, also when the failed attempt reached the transport (and so emits
/// on drop) or yielded the warning before failing.
#[test]
fn retried_calls_emit_each_warning_once() {
    let ((), events) = capture(async {
        let unary = MockProvider::build(|b| b.error(Error::timeout("slow")).text("ok"))
            .with_chat_capability(NO_REASONING.0, NO_REASONING.1);
        let retrying = RetryingProvider::new(unary.clone(), fast());
        let response = retrying.chat(&hinted("retry-unary")).await.unwrap();
        assert_eq!(unary.call_count(), 2);
        assert_eq!(response.warnings.len(), 1);

        // The first stream yields its warning, then fails before output.
        let stream = MockStreamingProvider::build(|b| {
            b.stream([
                MockStreamEvent::Event(StreamEvent::response_start(None, None)),
                MockStreamEvent::Error(Error::overloaded("busy")),
            ])
            .response(ChatResponseBuilder::new().text("second").build())
            .chat_capability(NO_REASONING.0, NO_REASONING.1)
        });
        let retrying = RetryingProvider::new(stream.clone(), fast());
        let response = retrying
            .chat_stream(&hinted("retry-stream"))
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        assert_eq!(stream.call_count(), 2);
        assert_eq!(response.text().as_deref(), Some("second"));

        // The last attempt's warnings go out with its error.
        let failing = MockProvider::new([Error::timeout("one"), Error::timeout("two")])
            .with_chat_capability(NO_REASONING.0, NO_REASONING.1);
        let retrying = RetryingProvider::new(failing, fast().max_attempts(2));
        assert!(retrying.chat(&hinted("retry-exhausted")).await.is_err());
    });
    for model in ["retry-unary", "retry-stream", "retry-exhausted"] {
        let model_events = events.for_model(model);
        assert_eq!(model_events.len(), 1, "{model}: {model_events:?}");
        assert_eq!(levels(&model_events), (1, 0), "{model}");
    }
    let retries = events.under("allama::retry");
    assert_eq!(retries.len(), 3, "{retries:?}");
    assert!(retries.iter().all(|event| event.level == "INFO"));
}

/// The backup's warning and the failover warning, once each.
#[test]
fn fallback_calls_emit_each_warning_once() {
    let ((), events) = capture(async {
        let backup = || {
            MockProvider::with_text("backup")
                .with_provider_name("backup")
                .with_chat_capability(NO_REASONING.0, NO_REASONING.1)
        };
        let pair = FallbackProvider::new(
            MockProvider::with_error(Error::overloaded("busy")),
            backup(),
        );
        let response = pair.chat(&hinted("fallback-unary")).await.unwrap();
        assert_eq!(response.warnings.len(), 2);

        let pair = FallbackProvider::new(
            MockProvider::with_error(Error::overloaded("busy")),
            backup(),
        );
        let response = pair
            .chat_stream(&hinted("fallback-stream"))
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        assert_eq!(response.warnings.len(), 2);
    });
    for model in ["fallback-unary", "fallback-stream"] {
        let model_events = events.for_model(model);
        let targets: Vec<_> = model_events.iter().map(|e| e.target.as_str()).collect();
        assert_eq!(
            targets,
            ["allama::warning::unsupported", "allama::warning::fallback"],
            "{model}: {model_events:?}"
        );
    }
}

/// A wrapper built on `prepare` emits its own warnings; the inner
/// provider's pass through its `finish` without a second event.
#[test]
fn wrappers_built_on_prepare_emit_only_their_own_warnings() {
    let ((), events) = capture(async {
        let inner = || {
            FunctionProvider::new(|_| {
                Ok(ChatResponse::new(vec![ContentBlock::text("ok")])
                    .finish_reason(FinishReason::Stop)
                    .warning(content_warning("inner content dropped")))
            })
        };
        let wrapper = Override(inner());
        let response = wrapper.chat(&hinted("override-unary")).await.unwrap();
        assert_eq!(response.warnings.len(), 2);

        let wrapper = Override(inner());
        let response = wrapper
            .chat_stream(&hinted("override-stream"))
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        assert_eq!(response.warnings.len(), 2);

        // Retry around the wrapper, with a failed first attempt.
        let flaky = MockProvider::build(|b| {
            b.error(Error::timeout("slow")).response(
                ChatResponse::new(vec![ContentBlock::text("ok")])
                    .warning(content_warning("inner content dropped")),
            )
        });
        let retrying = RetryingProvider::new(Override(flaky), fast());
        let response = retrying.chat(&hinted("override-retry")).await.unwrap();
        assert_eq!(response.warnings.len(), 2);
    });
    for model in ["override-unary", "override-stream", "override-retry"] {
        let model_events = events.for_model(model);
        let mut seen = messages(&model_events);
        seen.sort_unstable();
        assert_eq!(seen.len(), 2, "{model}: {model_events:?}");
        assert_eq!(seen[0], "inner content dropped", "{model}");
        assert!(
            seen[1].starts_with("reasoning configuration is not supported"),
            "{model}: {model_events:?}"
        );
    }
}

/// Batch jobs: the same hint on 1,000 calls is one `WARN`, then `DEBUG`.
#[test]
fn a_thousand_identical_calls_warn_once() {
    let ((), events) = capture(async {
        let provider =
            FunctionProvider::new(|_| Ok(ChatResponse::new(vec![ContentBlock::text("ok")])))
                .with_chat_capability(NO_REASONING.0, NO_REASONING.1);
        for _ in 0..1_000 {
            provider.chat(&hinted("thousand")).await.unwrap();
        }
    });
    let model_events = events.for_model("thousand");
    assert_eq!(levels(&model_events), (1, 999));
    assert!(
        model_events[0].repeats,
        "the first WARN says repeats go to DEBUG"
    );
    assert!(model_events[1..].iter().all(|event| !event.repeats));
}

/// Content and failover warnings have no code: `WARN` every time.
#[test]
fn uncoded_warnings_warn_every_time() {
    let ((), events) = capture(async {
        let provider = FunctionProvider::new(|_| {
            Ok(ChatResponse::new(vec![ContentBlock::text("ok")])
                .warning(content_warning("attachment dropped")))
        });
        let pair = FallbackProvider::new(
            FunctionProvider::new(|_| Err(Error::overloaded("busy"))),
            FunctionProvider::new(|_| Ok(ChatResponse::new(vec![ContentBlock::text("ok")]))),
        );
        for _ in 0..3 {
            provider
                .chat(&ChatRequest::new("uncoded-content").user("q"))
                .await
                .unwrap();
            pair.chat(&ChatRequest::new("uncoded-fallback").user("q"))
                .await
                .unwrap();
        }
    });
    let content = events.for_model("uncoded-content");
    assert_eq!(levels(&content), (3, 0));
    assert!(content.iter().all(|e| e.code.is_none() && !e.repeats));
    let fallback = events.for_model("uncoded-fallback");
    assert_eq!(levels(&fallback), (3, 0));
    assert!(
        fallback
            .iter()
            .all(|e| e.target == "allama::warning::fallback")
    );
}

/// Replay state recurs on every turn: a 10-turn history whose reasoning
/// the model cannot take back is one warning per call (listing the ten
/// messages), logged at `WARN` once, on the first call.
#[test]
fn replay_drops_warn_once_across_turns() {
    let mut request = ChatRequest::new("ten-turns").user("q0");
    for turn in 1..=10 {
        request = request
            .message(Message::assistant_blocks(vec![
                ContentBlock::reasoning(format!("thinking {turn}")),
                ContentBlock::text(format!("answer {turn}")),
            ]))
            .user(format!("q{turn}"));
    }
    let ((), events) = capture(async {
        let provider =
            FunctionProvider::new(|_| Ok(ChatResponse::new(vec![ContentBlock::text("ok")])))
                .with_chat_capability(
                    ChatCapability::ReasoningReplay,
                    CapabilitySupport::Unsupported,
                );
        let first = provider.chat(&request).await.unwrap();
        assert_eq!(first.warnings.len(), 1);
        assert_eq!(
            first.warnings[0].subject,
            Subject::Messages {
                messages: (1..=10).map(|turn| turn * 2 - 1).collect()
            }
        );
    });
    let first = events.for_model("ten-turns");
    assert_eq!(levels(&first), (1, 0));
    assert!(first.iter().all(|e| e.target == "allama::warning::dropped"
        && e.code.as_deref() == Some("dropped.reasoning_replay")));

    let ((), events) = capture(async {
        let provider =
            FunctionProvider::new(|_| Ok(ChatResponse::new(vec![ContentBlock::text("ok")])))
                .with_chat_capability(
                    ChatCapability::ReasoningReplay,
                    CapabilitySupport::Unsupported,
                );
        provider.chat(&request).await.unwrap();
    });
    assert_eq!(levels(&events.for_model("ten-turns")), (0, 1));
}

#[test]
fn each_kind_has_its_own_target() {
    let kinds = [
        (WarningKind::Unsupported, "allama::warning::unsupported"),
        (WarningKind::Approximated, "allama::warning::approximated"),
        (WarningKind::Dropped, "allama::warning::dropped"),
        (WarningKind::Deprecated, "allama::warning::deprecated"),
        (WarningKind::Fallback, "allama::warning::fallback"),
        (WarningKind::Other, "allama::warning::other"),
    ];
    let ((), events) = capture(async {
        for (kind, _) in kinds {
            let mut warning = Warning::new(kind, Subject::Response, kind.as_str());
            warning::emit("p", "targets", &mut warning);
        }
    });
    let targets: Vec<_> = events
        .for_model("targets")
        .into_iter()
        .map(|e| e.target)
        .collect();
    assert_eq!(targets, kinds.map(|(_, target)| target.to_owned()));
}

/// `Retry(Fallback(Retry(a), Retry(b)))`: each warning once.
#[test]
fn nested_retries_emit_once() {
    let ((), events) = capture(async {
        let a = MockProvider::new([Error::timeout("a1"), Error::timeout("a2")])
            .with_provider_name("a")
            .with_chat_capability(NO_REASONING.0, NO_REASONING.1);
        let b = MockProvider::build(|b| b.error(Error::timeout("b1")).text("ok"))
            .with_provider_name("b")
            .with_chat_capability(NO_REASONING.0, NO_REASONING.1);
        let stack = RetryingProvider::new(
            FallbackProvider::new(
                RetryingProvider::new(a, fast().max_attempts(2)),
                RetryingProvider::new(b, fast()),
            ),
            fast(),
        );
        let response = stack.chat(&hinted("nested-retry")).await.unwrap();
        assert_eq!(response.text().as_deref(), Some("ok"));
    });
    let nested = events.for_model("nested-retry");
    let mut providers: Vec<_> = nested
        .iter()
        .map(|e| (e.provider.clone().unwrap_or_default(), e.target.clone()))
        .collect();
    providers.sort();
    assert_eq!(
        providers,
        [
            ("a".to_owned(), "allama::warning::fallback".to_owned()),
            ("a".to_owned(), "allama::warning::unsupported".to_owned()),
            ("b".to_owned(), "allama::warning::unsupported".to_owned()),
        ],
        "{nested:?}"
    );
}

/// A failover logs one `WARN` (its fallback warning)
/// and an `INFO` event under the stable target `allama::fallback`, with or
/// without the `tracing` feature.
#[test]
fn a_failover_logs_one_warn_and_a_stable_info_event() {
    let ((), events) = capture(async {
        let pair = FallbackProvider::new(
            MockProvider::with_error(Error::overloaded("busy")).with_provider_name("primary"),
            MockProvider::with_text("ok").with_provider_name("backup"),
        );
        pair.chat(&ChatRequest::new("one-failover").user("q"))
            .await
            .unwrap();
    });
    let failover = events.under("allama::fallback");
    assert_eq!(failover.len(), 1, "{failover:?}");
    assert_eq!(failover[0].level, "INFO");
    let warns: Vec<_> = events
        .under("allama::")
        .into_iter()
        .filter(|event| event.level == "WARN")
        .collect();
    assert_eq!(warns.len(), 1, "{warns:?}");
    assert_eq!(warns[0].target, "allama::warning::fallback");
}
