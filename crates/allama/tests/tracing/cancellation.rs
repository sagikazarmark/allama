//! `allama.response.cancelled`: marked exactly when the caller abandoned a
//! call it had issued, and the span starting when the call is issued.

use std::panic::AssertUnwindSafe;
use std::time::Duration;

use allama::decision::Question;
use allama::extraction::Field;
use allama::mock::{
    ChatResponseBuilder, FunctionProvider, MockDecisionProvider, MockEmbeddingProvider,
    MockProvider, MockResponse, MockStreamEvent, MockStreamingProvider,
};
use allama::wrappers::{RetryingProvider, TracingProvider};
use allama::{
    CapabilitySupport, ChatProvider, ChatStreamExt, DecisionCapability, DecisionProvider,
    DecisionRequest, DecisionResponse, EmbeddingProvider, EmbeddingRequest, Error,
    ExtractionCapability, ExtractionProvider, ExtractionRequest, ExtractionResponse,
    ProviderIdentity, StreamEvent,
};
use futures_util::{FutureExt, StreamExt};
use opentelemetry::Value;
use opentelemetry::trace::Status;

use super::otel::{ExportedSpan, export};
use super::request;
use super::stack::recommended_stack;

/// Longer than any test runs: the call is still waiting when it is dropped.
const STALL: Duration = Duration::from_secs(30);

/// A cancelled span: marked, and not failed.
fn assert_cancelled(span: ExportedSpan<'_>) {
    assert_eq!(
        span.value("allama.response.cancelled"),
        &Value::Bool(true),
        "{}",
        span.name()
    );
    assert_eq!(span.status(), &Status::Unset, "{}", span.name());
    assert!(!span.has("error.type"), "{}", span.name());
}

/// Dropped while the stack waits for the primary's first token: before the
/// commit point, so no stream was handed over yet.
#[tokio::test]
async fn a_stream_dropped_before_its_first_token_under_the_recommended_stack_is_cancelled() {
    let provider = recommended_stack(
        MockStreamingProvider::with_stream([
            MockStreamEvent::Event(StreamEvent::response_start(None, None)),
            MockStreamEvent::delayed(STALL),
            MockStreamEvent::Event(StreamEvent::text_delta(0, "too late")),
        ]),
        MockProvider::with_text("backup"),
        Duration::from_secs(60),
    );
    let ((), exported) = export(async {
        assert!(provider.chat_stream(&request()).now_or_never().is_none());
    })
    .await;
    let chat = exported.span("chat");
    assert_cancelled(chat);
    assert_eq!(chat.value("gen_ai.request.stream"), &Value::Bool(true));
    // Nothing served yet: the addressed provider.
    assert_eq!(
        chat.value("allama.response.provider"),
        &Value::from("mock_stream")
    );
}

/// A unary call dropped before it completed, chat, embedding, decision and
/// extraction alike; the embedding call is dropped while its retry backs
/// off.
#[tokio::test]
async fn a_unary_call_dropped_before_it_completed_is_cancelled() {
    let chat = TracingProvider::new(MockProvider::new([MockResponse::delayed(
        STALL,
        MockResponse::text("too late"),
    )]));
    // The default policy waits out a hint up to 120 s.
    let embedding = TracingProvider::new(RetryingProvider::new(
        MockEmbeddingProvider::new([Err(Error::overloaded("busy").with_retry_after(STALL))]),
        allama::wrappers::RetryPolicy::default(),
    ));
    let decision = TracingProvider::new(StalledDecider);
    let extraction = TracingProvider::new(StalledDecider);
    let ((), exported) = export(async {
        assert!(chat.chat(&request()).now_or_never().is_none());
        assert!(
            embedding
                .embed(&EmbeddingRequest::new("e").input("x"))
                .now_or_never()
                .is_none()
        );
        assert!(
            decision
                .decide(&decision_request())
                .now_or_never()
                .is_none()
        );
        assert!(
            extraction
                .extract_values(&ExtractionRequest::new("x", "text").field("f", Field::text()))
                .now_or_never()
                .is_none()
        );
    })
    .await;
    let chat = exported.span("chat");
    assert_cancelled(chat);
    assert!(!chat.has("gen_ai.request.stream"));
    let embeddings = exported.span("embeddings");
    assert_cancelled(embeddings);
    assert_eq!(embeddings.value("allama.attempts"), &Value::I64(1));
    assert_cancelled(exported.span("decision"));
    assert_cancelled(exported.span("extraction"));
}

/// A future that is never polled issues no call, so it creates no span.
#[tokio::test]
async fn a_future_never_polled_exports_no_span() {
    let chat = TracingProvider::new(MockProvider::with_text("unused"));
    let embedding = TracingProvider::new(MockEmbeddingProvider::with_vectors(vec![vec![0.1]]));
    let decision = TracingProvider::new(MockDecisionProvider::empty());
    let ((), exported) = export(async {
        drop(chat.chat(&request()));
        drop(chat.chat_stream(&request()));
        drop(embedding.embed(&EmbeddingRequest::new("e").input("x")));
        drop(decision.decide(&decision_request()));
    })
    .await;
    assert_eq!(exported.spans().count(), 0);
}

/// The response ends with `ResponseStop`: a consumer that stops reading
/// there has the whole response and did not cancel.
#[tokio::test]
async fn a_stream_left_at_its_response_stop_is_not_cancelled() {
    let provider = TracingProvider::new(MockStreamingProvider::from_responses([
        ChatResponseBuilder::new()
            .text("complete")
            .usage(3, 2)
            .build(),
    ]));
    let ((), exported) = export(async {
        let mut stream = provider.chat_stream(&request()).await.unwrap();
        while let Some(event) = stream.next().await {
            if matches!(event.unwrap(), StreamEvent::ResponseStop) {
                break;
            }
        }
        // The end of the stream, after `ResponseStop`, is never read.
        drop(stream);
    })
    .await;
    let chat = exported.span("chat");
    assert!(!chat.has("allama.response.cancelled"));
    assert_eq!(chat.status(), &Status::Unset);
    assert_eq!(chat.value("gen_ai.usage.output_tokens"), &Value::I64(2));
}

/// A future or stream dropped while the thread unwinds from a panic was not
/// abandoned by choice: a provider that panics, and an application that
/// panics while it reads a stream.
#[tokio::test]
async fn a_call_dropped_by_a_panic_is_not_cancelled() {
    let panicking = TracingProvider::new(FunctionProvider::new(|_| panic!("provider bug")));
    let streaming = TracingProvider::new(MockStreamingProvider::from_responses([
        ChatResponseBuilder::new().text("partial").build(),
    ]));
    let ((), exported) = export(async {
        let unary = AssertUnwindSafe(panicking.chat(&request()))
            .catch_unwind()
            .await;
        assert!(unary.is_err());
        let stream = AssertUnwindSafe(async {
            let mut stream = streaming.chat_stream(&request()).await.unwrap();
            stream.next().await;
            panic!("application bug");
        })
        .catch_unwind()
        .await;
        assert!(stream.is_err());
    })
    .await;
    let spans: Vec<_> = exported.spans_for("chat").collect();
    assert_eq!(spans.len(), 2);
    for chat in spans {
        assert!(!chat.has("allama.response.cancelled"), "{:?}", chat);
    }
}

/// The span starts when the call is issued (first poll), not when its future
/// is created: time spent before the first poll is not on the span.
#[tokio::test]
async fn the_span_starts_at_the_first_poll() {
    const WAIT: Duration = Duration::from_millis(100);
    let provider = TracingProvider::new(MockStreamingProvider::from_responses([
        ChatResponseBuilder::new().text("hi").build(),
    ]));
    let ((), exported) = export(async {
        let request = request();
        let call = provider.chat_stream(&request);
        tokio::time::sleep(WAIT).await;
        call.await.unwrap().collect_response().await.unwrap();
    })
    .await;
    let chat = exported.span("chat");
    assert!(chat.duration() < WAIT, "span lasted {:?}", chat.duration());
}

fn decision_request() -> DecisionRequest {
    DecisionRequest::new("d", "text").question("q", Question::yes_no("Is it?"))
}

/// A decision and extraction provider that never answers.
struct StalledDecider;

impl ProviderIdentity for StalledDecider {
    fn provider_name(&self) -> &str {
        "stalled"
    }

    fn reads_namespace(&self, namespace: &str) -> bool {
        namespace == "stalled"
    }
}

impl DecisionProvider for StalledDecider {
    async fn decide(&self, _request: &DecisionRequest) -> allama::Result<DecisionResponse> {
        std::future::pending().await
    }

    fn decision_capability(
        &self,
        _model: &str,
        _capability: DecisionCapability,
    ) -> CapabilitySupport {
        CapabilitySupport::Unknown
    }
}

impl ExtractionProvider for StalledDecider {
    async fn extract_values(
        &self,
        _request: &ExtractionRequest,
    ) -> allama::Result<ExtractionResponse> {
        std::future::pending().await
    }

    fn extraction_capability(
        &self,
        _model: &str,
        _capability: ExtractionCapability,
    ) -> CapabilitySupport {
        CapabilitySupport::Unknown
    }
}
