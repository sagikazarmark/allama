//! Test doubles for chat, embedding, and decision providers (feature `mock`).
//!
//! # Choosing a chat mock
//!
//! - [`MockProvider`] and [`MockStreamingProvider`] replay a script, one
//!   entry per call, and record every request. Use them when the test cares
//!   about exact output or exact failures. A call past the end of the script
//!   returns [`Error::Configuration`](crate::Error::Configuration).
//! - [`TestProvider`] answers every request and never runs out. Use it for
//!   tests that send many requests, such as HTTP handler tests, or that do
//!   not care about the exact output. Its responses carry a complete
//!   envelope (model, provider, usage, ids).
//! - [`FunctionProvider`] computes each response from the request.
//!
//! # When scripted errors surface
//!
//! A real streaming call can fail in two places: before the stream exists
//! (the provider rejects the request, say with a 429) or partway through it
//! (the connection drops). The mocks script both:
//!
//! | Scripted with | Surfaces |
//! | --- | --- |
//! | [`MockProvider::with_error`], [`MockProviderBuilder::error`] | from the call: `chat` and `chat_stream` return `Err` |
//! | [`MockStreamingProviderBuilder::setup_error`] | from the call: `chat_stream` returns `Err` |
//! | [`MockStreamEvent::Error`] inside a stream | as a stream item, after `chat_stream` returns `Ok` |
//!
//! # Transport options
//!
//! The mocks send no HTTP request, so transport options such as
//! `allama_http::HttpRequestOptions` (headers, under the `allama-http`
//! options namespace) have no effect, like options for any other
//! namespace. They stay on the recorded request, so a test can assert that
//! the application attached them.
//!
//! The mocks have no credential scope either, by design: `allama` does not
//! depend on `allama-http`, where credentials live, so a mock never refuses
//! a request that lacks a tenant's key. To test that path, use a real
//! adapter built with a scope and no key over an
//! `allama_http::ReplayTransport`: a request without the tenant's
//! credential fails with [`Error::Configuration`](crate::Error::Configuration),
//! and the transport's `requests()` stays empty, since nothing was sent.
//! The applications guide's "Testing" section has the recipe.
//!
//! # Timers in tests
//!
//! [`RetryingProvider`](crate::wrappers::RetryingProvider) backoff,
//! `Retry-After` waits and [`TimeoutProvider`](crate::wrappers::TimeoutProvider)
//! deadlines use `futures-timer`, so they run on any runtime and real
//! time passes: `tokio::time::pause()` does not speed them up. Keep the
//! waits short in tests instead: a [`RetryPolicy`](crate::wrappers::RetryPolicy)
//! with a zero `base_delay` and no jitter, a small `max_retry_after`, and
//! scripted `Retry-After` hints of a few milliseconds
//! ([`MockResponse::delayed`] scripts a slow answer the same way).
//!
//! # Replaying a recorded call
//!
//! A [`ChatExchange`](crate::wrappers::ChatExchange) that a
//! [`RecordingProvider`](crate::wrappers::RecordingProvider) recorded holds
//! everything a mock needs to answer the same way: the response or the
//! error of a unary call, and the [`StreamTranscript`](crate::chat::stream::StreamTranscript)
//! of a stream, with the error that ended it. One [`MockStreamingProvider`]
//! replays a whole recorded turn, unary calls and streams alike, since it
//! answers `chat` by collecting its next scripted stream. Script one entry
//! per exchange, in order, then send the recorded requests through the code
//! under test:
//!
//! ```
//! use std::sync::{Arc, Mutex};
//!
//! use allama::mock::{
//!     MockProvider, MockStreamEvent, MockStreamingProvider, MockStreamingProviderBuilder,
//! };
//! use allama::prelude::*;
//! use allama::wrappers::{ChatExchange, RecordingProvider};
//!
//! /// A mock that answers each call like the recorded exchange in its place.
//! fn replay(exchanges: Vec<ChatExchange>) -> MockStreamingProvider {
//!     MockStreamingProvider::build(|b| exchanges.into_iter().fold(b, replay_one))
//! }
//!
//! fn replay_one(
//!     b: MockStreamingProviderBuilder,
//!     exchange: ChatExchange,
//! ) -> MockStreamingProviderBuilder {
//!     match (exchange.stream, exchange.response, exchange.error) {
//!         // A stream: every event that arrived, then the error that ended it.
//!         (Some(transcript), _, _) => b.stream(
//!             transcript
//!                 .events
//!                 .into_iter()
//!                 .map(MockStreamEvent::from)
//!                 .chain(transcript.terminal_error.map(MockStreamEvent::from)),
//!         ),
//!         // A unary call that succeeded.
//!         (None, Some(response), _) => b.response(response),
//!         // A unary call that failed, or a stream that never opened.
//!         (None, None, error) => b.setup_error(error.expect("a call that failed")),
//!     }
//! }
//!
//! # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
//! // Record a turn that makes a unary call, then streams, and keep it as a
//! // fixture.
//! let log: Arc<Mutex<Vec<ChatExchange>>> = Arc::default();
//! let sink = log.clone();
//! let live = MockProvider::build(|b| b.text("billing").text("hello"));
//! let live = RecordingProvider::new(live, move |exchange| sink.lock().unwrap().push(exchange));
//! let triage = ChatRequest::new("demo-model").user("Which team?");
//! let reply = ChatRequest::new("demo-model").user("Hi");
//! live.chat(&triage).await?;
//! live.chat_stream(&reply).await?.collect_response().await?;
//! let fixture = serde_json::to_string(&*log.lock().unwrap())?;
//!
//! // In the test: load it and replay it through one mock.
//! let exchanges: Vec<ChatExchange> = serde_json::from_str(&fixture)?;
//! let recorded: Vec<ChatRequest> = exchanges.iter().map(|e| e.request.clone()).collect();
//! let mock = replay(exchanges);
//! let team = mock.chat(&recorded[0]).await?;
//! assert_eq!(team.text().as_deref(), Some("billing"));
//! let response = mock.chat_stream(&recorded[1]).await?.collect_response().await?;
//! assert_eq!(response.text().as_deref(), Some("hello"));
//! assert_eq!(mock.requests(), recorded);
//!
//! // A failed call replays as the same error.
//! let failed = ChatRequest::new("demo-model").user("Again");
//! let recorder = RecordingProvider::new(MockProvider::with_error(Error::rate_limited("HTTP 429")), {
//!     let log = log.clone();
//!     move |exchange| log.lock().unwrap().push(exchange)
//! });
//! assert!(recorder.chat(&failed).await.is_err());
//! let mock = replay(vec![log.lock().unwrap().pop().unwrap()]);
//! assert!(matches!(mock.chat(&failed).await, Err(Error::RateLimited { .. })));
//! # Ok::<(), allama::Error>(()) }).unwrap();
//! ```
//!
//! An exchange is a fixture for the allama version that recorded it, not a
//! storage format: re-record fixtures after upgrading allama (see
//! [`ChatExchange`](crate::wrappers::ChatExchange#stability)).

pub use crate::chat::mock::{
    ChatResponseBuilder, FunctionProvider, MockProvider, MockProviderBuilder, MockResponse,
    MockStreamEvent, MockStreamingProvider, MockStreamingProviderBuilder, TestProvider,
};
pub use crate::decision::mock::MockDecisionProvider;
pub use crate::embedding::mock::MockEmbeddingProvider;
