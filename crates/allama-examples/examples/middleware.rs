//! # Writing a middleware wrapper
//!
//! Middleware in `allama` is a wrapper: a type that implements the provider
//! traits over an inner provider. [`MapRequestProvider`] covers a
//! synchronous request rewrite; anything more (an async lookup before the
//! call, watching what comes back) is a hand-written wrapper like the one
//! below. It shows:
//!
//! 1. **Async request mutation.** `before` awaits a lookup (memory, RAG, a
//!    policy service) and adds what it found to an owned copy of the
//!    request. The caller's request is never changed, and the model is
//!    never changed: capability answers come from the inner provider for
//!    the caller's model.
//! 2. **Response observation.** `chat` sees the result, success or error,
//!    before handing it back unchanged.
//! 3. **Stream observation.** `chat_stream` wraps the inner stream and
//!    returns it as a boxed [`ChatStream`]. Boxing needs `P::Stream:
//!    'static`; put that bound on the `impl` block, not on the method (a
//!    method cannot add bounds the trait does not have: E0276).
//! 4. **Transparency.** Everything a caller asks *about* the provider is
//!    forwarded: [`provider_name`](ProviderIdentity::provider_name),
//!    [`downcast_any`](ProviderIdentity::downcast_any), the capability
//!    answers, the limits (`embedding_limits`, `decision_limits`), and
//!    request validation. The `allama::wrappers::forward_*` macros write
//!    all of it, so a question added to a provider trait later is forwarded
//!    by recompiling; the defaults of the methods are right for providers
//!    and wrong for wrappers. What the inner provider answers
//!    reaches the caller unchanged: its warnings, provider data, usage,
//!    stream events and errors (a wrapper may add warnings, and adds context
//!    to an error with `Error::with_message_prefix`). The test at the
//!    bottom runs `allama_conformance::assert_wrapper`, which checks all of
//!    it: run it on your own wrappers too.
//!
//! This wrapper changes nothing, so it reports nothing. One that does
//! reports each change as a coded warning of the matching kind (a
//! redaction wrapper: `Approximated`, code `approximated.redacted_content`)
//! and sends pure observations (what `before` recalled, timings) to
//! `tracing` or the application's own channel. It never writes provider
//! data: not under `allama` (reserved; `FallbackAttempts` is the one
//! exception), not under a provider's namespace, and not on blocks. An
//! annotation that is not a degradation has no portable place on the
//! response yet. A wrapper that rewrites or annotates responses beyond
//! adding warnings is not transparent: it runs
//! `assert_transparent_chat_wrapper` instead of `assert_wrapper`'s data
//! path.
//!
//! Embeddings and decisions pass through unchanged here. Implement the
//! traits your callers use; a wrapper that only serves chat can skip the
//! other two.
//!
//! A wrapper never builds HTTP requests, so it injects headers and
//! credentials through the transport: per-call headers (a request id, a
//! tenant tag) and per-tenant keys go on the request as
//! `allama_http::HttpRequestOptions`, which every HTTP adapter applies when
//! it sends. Add to the options the request already carries, so the
//! caller's headers and credentials survive (the `HttpRequestOptions`
//! rustdoc shows a request id header). For latency and attempts, wrap in
//! `TracingProvider` rather than timing calls here.
//!
//! It needs no credentials. Run it with `cargo run -p allama-examples
//! --example middleware`, and its tests with `cargo test -p allama-examples
//! --example middleware`.

use std::sync::{Arc, Mutex};

use allama::mock::{MockProvider, MockStreamingProvider};
use allama::prelude::*;
use allama::wrappers::MapRequestProvider;
use allama::{DecisionResponse, EmbeddingResponse, ProviderIdentity};
use futures_util::StreamExt;

/// Recalls notes for the latest user message before each chat call, and
/// logs every chat outcome.
struct RecallProvider<P> {
    inner: P,
    notes: NoteStore,
    log: Arc<CallLog>,
}

impl<P> RecallProvider<P> {
    fn new(inner: P, notes: NoteStore) -> Self {
        Self {
            inner,
            notes,
            log: Arc::default(),
        }
    }

    /// The request the inner provider receives: an owned copy with the
    /// recalled notes as an extra system prompt.
    async fn before(&self, request: &ChatRequest) -> ChatRequest {
        let query = request
            .messages
            .iter()
            .rev()
            .find_map(|message| message.as_user().and(message.text()))
            .unwrap_or_default();
        let notes = self.notes.recall(&query).await;

        let mut request = request.clone();
        if !notes.is_empty() {
            request
                .system
                .push(format!("Known about the user: {}", notes.join("; ")).into());
        }
        request
    }
}

// Every provider's stream is `'static`, so boxing it into a `ChatStream`
// needs no bound beyond `P: ChatProvider`.
impl<P> ChatProvider for RecallProvider<P>
where
    P: ChatProvider,
{
    type Stream = ChatStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let request = self.before(request).await;
        let result = self.inner.chat(&request).await;
        match &result {
            Ok(response) => self.log.record(format!(
                "chat: finish reason {:?}, output tokens {:?}",
                response.finish_reason,
                response
                    .usage
                    .as_ref()
                    .and_then(|usage| usage.output_tokens),
            )),
            Err(error) => self.log.record(format!("chat failed: {}", error.as_log())),
        }
        result
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        let request = self.before(request).await;
        let stream = match self.inner.chat_stream(&request).await {
            Ok(stream) => stream,
            Err(error) => {
                self.log
                    .record(format!("stream failed to open: {}", error.as_log()));
                return Err(error);
            }
        };

        // The stream outlives this call, so it owns its handle on the log.
        let log = Arc::clone(&self.log);
        let mut streamed = 0;
        Ok(Box::pin(stream.inspect(move |event| match event {
            Ok(StreamEvent::TextDelta { text, .. }) => streamed += text.len(),
            Ok(StreamEvent::ResponseStop) => {
                log.record(format!("stream: {streamed} bytes of text"));
            }
            Ok(_) => {}
            Err(error) => log.record(format!("stream failed: {}", error.as_log())),
        })))
    }

    allama::wrappers::forward_chat_questions!(inner);

    // The lookup is async and validation is not, so this validates the
    // caller's request. That is sound because `before` only adds a system
    // prompt, which needs no capability. A middleware whose changes do need
    // one should validate what it sends, like `MapRequestProvider`.

    // Streaming can refuse what a unary call takes, so the stream validator
    // is forwarded too.
}

impl<P: EmbeddingProvider> EmbeddingProvider for RecallProvider<P> {
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        self.inner.embed(request).await
    }

    allama::wrappers::forward_embedding_questions!(inner);
}

impl<P: DecisionProvider> DecisionProvider for RecallProvider<P> {
    async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse> {
        self.inner.decide(request).await
    }

    allama::wrappers::forward_decision_questions!(inner);
}

impl<P: ProviderIdentity> ProviderIdentity for RecallProvider<P> {
    allama::wrappers::forward_provider_identity!(inner);
}

/// A stand-in for an async store: a vector database, a memory service.
#[derive(Default)]
struct NoteStore {
    notes: Vec<(&'static str, &'static str)>,
}

impl NoteStore {
    fn new(notes: impl IntoIterator<Item = (&'static str, &'static str)>) -> Self {
        Self {
            notes: notes.into_iter().collect(),
        }
    }

    /// The notes whose keyword appears in `query`.
    async fn recall(&self, query: &str) -> Vec<String> {
        // A real store does I/O here.
        tokio::task::yield_now().await;
        let query = query.to_lowercase();
        self.notes
            .iter()
            .filter(|(keyword, _)| query.contains(keyword))
            .map(|(_, note)| (*note).to_owned())
            .collect()
    }
}

/// What the wrapper observed, for the demo and the tests.
#[derive(Default)]
struct CallLog(Mutex<Vec<String>>);

impl CallLog {
    fn record(&self, line: String) {
        eprintln!("[recall] {line}");
        self.0.lock().expect("log lock").push(line);
    }

    fn lines(&self) -> Vec<String> {
        self.0.lock().expect("log lock").clone()
    }
}

fn notes() -> NoteStore {
    NoteStore::new([
        ("coffee", "prefers oat milk"),
        ("weather", "lives in Lisbon"),
    ])
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // A single call: the wrapper adds a recalled note and logs the outcome.
    let provider = RecallProvider::new(MockProvider::with_text("Lisbon: sunny, 24°C."), notes());
    let request = ChatRequest::new("demo-model").user("What's the weather like?");
    let response = provider.chat(&request).await?;
    println!("chat: {}", response.text_or_empty());

    let sent = provider.inner.last_request().expect("one call");
    println!("system prompts sent: {:?}", sent.system);
    assert!(
        request.system.is_empty(),
        "the caller's request is unchanged"
    );
    assert_eq!(provider.log.lines().len(), 1);

    // A stream: events pass through untouched while the wrapper watches.
    let provider = RecallProvider::new(
        MockStreamingProvider::with_text("Oat milk latte, coming up."),
        notes(),
    );
    let mut stream = provider
        .chat_stream(&ChatRequest::new("demo-model").user("Order me a coffee"))
        .await?;
    print!("stream: ");
    while let Some(event) = stream.next_event().await {
        if let StreamEvent::TextDelta { text, .. } = event? {
            print!("{text}");
        }
    }
    println!();

    // Wrappers compose and erase like any provider, and the concrete
    // provider stays reachable behind them.
    let provider = DynChatProvider::new(MapRequestProvider::new(
        RecallProvider::new(MockProvider::with_text("ok"), notes()),
        |request: &mut ChatRequest| {
            request.temperature.get_or_insert(0.2);
        },
    ));
    println!("served by: {}", provider.provider_name());
    assert!(provider.downcast_ref::<MockProvider>().is_some());
    provider.ask("demo-model", "Hello").await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use allama::mock::MockStreamEvent;
    use allama::{CapabilitySupport, ChatCapability, Error, UnsupportedPolicy};
    use allama_conformance::assert_wrapper;

    use super::*;

    /// The wrapper forwards identity, capabilities, limits, and validation,
    /// and passes the inner provider's responses, streams and errors
    /// through unchanged.
    #[tokio::test]
    async fn recall_provider_is_transparent() {
        assert_wrapper(|probe| RecallProvider::new(probe, notes())).await;
    }

    #[tokio::test]
    async fn adds_recalled_notes_to_an_owned_copy() {
        let provider = RecallProvider::new(MockProvider::with_text("ok"), notes());
        let request = ChatRequest::new("m")
            .system("Be brief.")
            .user("Weather tomorrow?");
        provider.chat(&request).await.unwrap();

        let sent = provider.inner.last_request().unwrap();
        assert_eq!(sent.model, "m");
        assert_eq!(sent.system.len(), 2);
        assert_eq!(
            sent.system[1].content,
            "Known about the user: lives in Lisbon"
        );
        assert_eq!(request.system.len(), 1, "caller's request untouched");
    }

    #[tokio::test]
    async fn sends_unchanged_when_nothing_is_recalled() {
        let provider = RecallProvider::new(MockProvider::with_text("ok"), notes());
        let request = ChatRequest::new("m").user("Hello");
        provider.chat(&request).await.unwrap();
        assert_eq!(provider.inner.last_request().unwrap(), request);
    }

    #[tokio::test]
    async fn observes_responses_and_errors() {
        let provider = RecallProvider::new(
            MockProvider::build(|b| b.text("ok").error(Error::overloaded("busy"))),
            NoteStore::default(),
        );
        let request = ChatRequest::new("m").user("hi");
        provider.chat(&request).await.unwrap();
        let err = provider.chat(&request).await.unwrap_err();
        assert!(
            matches!(err, Error::Overloaded { .. }),
            "returned unchanged"
        );

        let lines = provider.log.lines();
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert!(lines[0].starts_with("chat: "), "{lines:?}");
        assert!(
            lines[1].starts_with("chat failed: kind=overloaded"),
            "{lines:?}"
        );
    }

    #[tokio::test]
    async fn observes_streams_without_changing_them() {
        let provider =
            RecallProvider::new(MockStreamingProvider::with_text("Hello there"), notes());
        let response = provider
            .chat_stream(&ChatRequest::new("m").user("coffee?"))
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        assert_eq!(response.text().as_deref(), Some("Hello there"));
        assert_eq!(provider.log.lines(), ["stream: 11 bytes of text"]);

        let sent = &provider.inner.requests()[0];
        assert_eq!(
            sent.system[0].content,
            "Known about the user: prefers oat milk"
        );
    }

    #[tokio::test]
    async fn observes_stream_errors() {
        let mut events = MockStreamEvent::text_response("partial");
        events.insert(events.len() - 1, Error::overloaded("busy").into());
        let provider = RecallProvider::new(
            MockStreamingProvider::with_stream(events),
            NoteStore::default(),
        );
        let err = provider
            .chat_stream(&ChatRequest::new("m").user("hi"))
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Overloaded { .. }));
        let lines = provider.log.lines();
        assert!(
            lines[0].starts_with("stream failed: kind=overloaded"),
            "{lines:?}"
        );
    }

    #[test]
    fn validation_reaches_the_inner_provider() {
        let provider = RecallProvider::new(
            MockProvider::empty().with_chat_capability(
                ChatCapability::ReasoningConfig,
                CapabilitySupport::Unsupported,
            ),
            notes(),
        );
        let request = ChatRequest::new("m")
            .user("hi")
            .reasoning(ReasoningEffort::High)
            .unsupported_policy(UnsupportedPolicy::Error);
        let err = provider.validate_chat_request(&request).unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }), "{err}");
    }
}
