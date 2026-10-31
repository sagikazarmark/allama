//! # Sending trace context to a gateway
//!
//! A gateway or server that joins distributed traces (LiteLLM proxy, vLLM,
//! an instrumented internal gateway) continues the application's trace
//! when each request carries a W3C `traceparent` header. allama sends none
//! of its own and depends on no OpenTelemetry crate: the application adds
//! the header, with its own OpenTelemetry version, in one of two places.
//!
//! 1. **A provider you build** with its builder: wrap the transport.
//!    [`PropagatingTransport`] writes the current span's context into the
//!    request's headers just before it sends. Adapters apply the caller's
//!    `HttpRequestOptions` before they send, so it sees the final headers.
//!    It covers every call the provider makes (chat, embedding, decision)
//!    and every retry attempt.
//! 2. **A provider from the registry** (`allama_providers::chat::from_model`),
//!    which takes no transport: wrap the chat provider.
//!    [`TraceContextProvider`] adds the header to the request's
//!    `HttpRequestOptions`, which every HTTP adapter sends. Its limits:
//!    - it clones the request on each call;
//!    - it must sit inside `TracingProvider`, so that the current span is
//!      the GenAI span;
//!    - it adds the header once per call, not once per retry attempt, so
//!      every attempt carries the same one;
//!    - it covers chat only: embedding and decision providers need the
//!      same wrapper over their own traits.
//!
//! Neither overwrites a `traceparent` the caller set. Both read the span
//! context through `tracing-opentelemetry`, so the subscriber needs its
//! layer, and write it through the global propagator, which sends nothing
//! until the application installs one (the W3C `TraceContextPropagator`
//! sends `traceparent` and `tracestate`). This file is written against
//! `opentelemetry` and `opentelemetry_sdk` 0.33 and `tracing-opentelemetry`
//! 0.34; with other releases, use the ones your application already has.
//!
//! The observability guide quotes it
//! (`docs/guides/observability/README.md`, "Trace context propagation").
//! It needs no credentials: the provider sends to a `ReplayTransport`. Run
//! it with `cargo run -p allama-examples --example trace_context`, and its
//! tests with `cargo test -p allama-examples --example trace_context`.

use allama::prelude::*;
use allama::wrappers::TracingProvider;
use allama::{ProviderIdentity, Result};
use allama_http::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use allama_http::{
    HttpRequest, HttpRequestOptions, HttpTransport, ReplayResponse, ReplayTransport, SendFuture,
};
use allama_providers::Registry;
use opentelemetry::propagation::Injector;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::prelude::*;

/// An [`HttpTransport`] that sends the current span's trace context with
/// every request.
struct PropagatingTransport<T> {
    inner: T,
}

impl<T: HttpTransport> HttpTransport for PropagatingTransport<T> {
    fn send(&self, mut request: HttpRequest) -> SendFuture<'_> {
        inject_trace_context(&mut request.headers);
        self.inner.send(request)
    }
}

/// A chat wrapper that sends the current span's trace context with every
/// request, through the request's `HttpRequestOptions`.
struct TraceContextProvider<P> {
    inner: P,
}

impl<P: ChatProvider> ChatProvider for TraceContextProvider<P> {
    type Stream = P::Stream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        self.inner.chat(&with_trace_context(request)?).await
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<P::Stream> {
        self.inner.chat_stream(&with_trace_context(request)?).await
    }

    allama::wrappers::forward_chat_questions!(inner);
}

impl<P: ProviderIdentity> ProviderIdentity for TraceContextProvider<P> {
    allama::wrappers::forward_provider_identity!(inner);
}

/// An owned copy of `request` whose `HttpRequestOptions` carry the current
/// span's trace context, next to the caller's headers and credentials.
fn with_trace_context(request: &ChatRequest) -> Result<ChatRequest> {
    let mut request = request.clone();
    let options = request.resolve_option_mut::<HttpRequestOptions>()?;
    inject_trace_context(&mut options.headers);
    Ok(request)
}

/// Write the current span's trace context into `headers` through the
/// global propagator, unless they already carry a `traceparent`.
fn inject_trace_context(headers: &mut HeaderMap) {
    if headers.contains_key("traceparent") {
        return;
    }
    let context = tracing::Span::current().context();
    opentelemetry::global::get_text_map_propagator(|propagator| {
        propagator.inject_context(&context, &mut HeaderInjector(headers));
    });
}

/// Writes the propagator's fields as headers, one value per name. A field
/// no header can carry is left out.
struct HeaderInjector<'a>(&'a mut HeaderMap);

impl Injector for HeaderInjector<'_> {
    fn set(&mut self, name: &str, value: String) {
        if let (Ok(name), Ok(value)) = (HeaderName::try_from(name), HeaderValue::try_from(value)) {
            self.0.insert(name, value);
        }
    }
}

const MODEL: &str = "gpt-5-mini";

/// The provider's network: one canned Chat Completions answer per call.
fn wire(calls: usize) -> ReplayTransport {
    const ANSWER: &str = r#"{"id":"c1","model":"gpt-5-mini","choices":[{"index":0,"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}]}"#;
    ReplayTransport::new((0..calls).map(|_| ReplayResponse::json(StatusCode::OK, ANSWER)))
}

/// The OpenAI provider, sending through `transport`.
fn openai(transport: impl HttpTransport) -> Result<allama_openai::Provider> {
    allama_openai::Provider::builder()
        .api_key("demo-key")
        .http_transport(transport)
        .build()
}

/// The `traceparent` values of the last request sent through `wire`.
fn sent_traceparent(wire: &ReplayTransport) -> Vec<String> {
    let requests = wire.requests();
    let request = requests.last().expect("a request was sent");
    request
        .headers
        .get_all("traceparent")
        .iter()
        .map(|value| String::from_utf8_lossy(value.as_bytes()).into_owned())
        .collect()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // The application's OpenTelemetry setup: a tracer (exporting to memory
    // here, OTLP in production), its `tracing` layer, and the W3C
    // propagator, without which nothing is sent.
    let exporter = InMemorySpanExporter::default();
    let tracer_provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    tracing_subscriber::registry()
        .with(tracing_opentelemetry::layer().with_tracer(tracer_provider.tracer("my-app")))
        .init();
    opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());

    // 1. A provider you build: wrap its transport (in production,
    // `ReqwestTransport::new()?` in place of the replayed network).
    let network = wire(2);
    let provider = TracingProvider::new(
        allama_openai::Provider::builder()
            .api_key("demo-key")
            .http_transport(PropagatingTransport {
                inner: network.clone(),
            })
            .build()?,
    );
    provider
        .chat(&ChatRequest::new(MODEL).user("Say hello"))
        .await?;
    println!("builder:  traceparent {:?}", sent_traceparent(&network));

    // A `traceparent` the caller sets is sent as it is.
    let caller_set = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
    let request = ChatRequest::new(MODEL).user("Say hello").with_option(
        HttpRequestOptions::new().header("traceparent", HeaderValue::from_static(caller_set)),
    );
    provider.chat(&request).await?;
    assert_eq!(sent_traceparent(&network), [caller_set]);
    println!("caller:   traceparent {:?}", sent_traceparent(&network));

    // 2. A provider from the registry: wrap the chat provider, inside
    // `TracingProvider`. `allama_providers::chat::from_model("openai:..")`
    // returns the same kind of provider; this registry stands in for it so
    // that the example needs no key and no network.
    let network = wire(1);
    let mut registry = Registry::new();
    let transport = network.clone();
    registry.register_chat("openai", move || openai(transport.clone()));
    let (provider, model) = registry.chat_from_model("openai:gpt-5-mini")?;
    let provider = TracingProvider::new(TraceContextProvider { inner: provider });
    provider
        .chat(&ChatRequest::new(model).user("Say hello"))
        .await?;
    println!("registry: traceparent {:?}", sent_traceparent(&network));

    // Each injected `traceparent` names its GenAI span: its trace id and,
    // as the parent, its span id.
    tracer_provider.force_flush()?;
    for span in exporter.get_finished_spans()? {
        println!(
            "span {:?}: trace {} span {}",
            span.name,
            span.span_context.trace_id(),
            span.span_context.span_id()
        );
    }
    tracer_provider.shutdown()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use allama::DynChatProvider;
    use allama_conformance::assert_chat_wrapper;
    use tracing::subscriber::DefaultGuard;

    use super::*;

    const CALLER_SET: &str = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";

    /// A tracer exporting to memory, its layer as this thread's
    /// subscriber, and the W3C propagator.
    struct Telemetry {
        exporter: InMemorySpanExporter,
        _subscriber: DefaultGuard,
    }

    impl Telemetry {
        fn install() -> Self {
            let exporter = InMemorySpanExporter::default();
            let tracer_provider = SdkTracerProvider::builder()
                .with_simple_exporter(exporter.clone())
                .build();
            let subscriber = tracing_subscriber::registry()
                .with(tracing_opentelemetry::layer().with_tracer(tracer_provider.tracer("test")));
            opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());
            Self {
                exporter,
                _subscriber: tracing::subscriber::set_default(subscriber),
            }
        }

        /// The `traceparent` of the one finished GenAI chat span.
        fn genai_traceparent(&self) -> String {
            let spans = self.exporter.get_finished_spans().unwrap();
            let chat: Vec<_> = spans
                .iter()
                .filter(|span| span.name == format!("chat {MODEL}"))
                .collect();
            assert_eq!(chat.len(), 1, "{spans:?}");
            let context = &chat[0].span_context;
            format!("00-{}-{}-01", context.trace_id(), context.span_id())
        }
    }

    fn built(network: &ReplayTransport) -> TracingProvider<allama_openai::Provider> {
        TracingProvider::new(
            openai(PropagatingTransport {
                inner: network.clone(),
            })
            .unwrap(),
        )
    }

    fn from_registry(
        network: &ReplayTransport,
    ) -> TracingProvider<TraceContextProvider<DynChatProvider>> {
        let mut registry = Registry::new();
        let transport = network.clone();
        registry.register_chat("openai", move || openai(transport.clone()));
        let (provider, _) = registry.chat_from_model("openai:gpt-5-mini").unwrap();
        TracingProvider::new(TraceContextProvider { inner: provider })
    }

    fn request() -> ChatRequest {
        ChatRequest::new(MODEL).user("Say hello")
    }

    fn caller_set_request() -> ChatRequest {
        request().with_option(
            HttpRequestOptions::new().header("traceparent", HeaderValue::from_static(CALLER_SET)),
        )
    }

    #[tokio::test]
    async fn the_transport_sends_the_genai_spans_context() {
        let telemetry = Telemetry::install();
        let network = wire(1);
        built(&network).chat(&request()).await.unwrap();
        assert_eq!(sent_traceparent(&network), [telemetry.genai_traceparent()]);
    }

    #[tokio::test]
    async fn the_transport_keeps_a_traceparent_the_caller_set() {
        let _telemetry = Telemetry::install();
        let network = wire(1);
        built(&network).chat(&caller_set_request()).await.unwrap();
        assert_eq!(sent_traceparent(&network), [CALLER_SET]);
    }

    #[tokio::test]
    async fn the_transport_sends_nothing_outside_a_span() {
        let _telemetry = Telemetry::install();
        let network = wire(1);
        let provider = openai(PropagatingTransport {
            inner: network.clone(),
        })
        .unwrap();
        provider.chat(&request()).await.unwrap();
        assert!(sent_traceparent(&network).is_empty());
    }

    #[tokio::test]
    async fn the_chat_wrapper_sends_the_genai_spans_context() {
        let telemetry = Telemetry::install();
        let network = wire(1);
        from_registry(&network).chat(&request()).await.unwrap();
        assert_eq!(sent_traceparent(&network), [telemetry.genai_traceparent()]);
    }

    #[tokio::test]
    async fn the_chat_wrapper_keeps_a_traceparent_the_caller_set() {
        let _telemetry = Telemetry::install();
        let network = wire(1);
        from_registry(&network)
            .chat(&caller_set_request())
            .await
            .unwrap();
        assert_eq!(sent_traceparent(&network), [CALLER_SET]);
    }

    #[tokio::test]
    async fn the_chat_wrapper_keeps_the_callers_other_headers() {
        let _telemetry = Telemetry::install();
        let network = wire(1);
        let request = request().with_option(
            HttpRequestOptions::new().header("x-tenant-id", HeaderValue::from_static("acme")),
        );
        from_registry(&network).chat(&request).await.unwrap();
        let sent = &network.requests()[0];
        assert_eq!(sent.headers["x-tenant-id"], "acme");
        assert_eq!(sent_traceparent(&network).len(), 1);
    }

    /// The chat wrapper forwards identity, capability answers and
    /// validation, and passes responses, streams and errors through.
    #[tokio::test]
    async fn the_chat_wrapper_is_transparent() {
        assert_chat_wrapper(|inner| TraceContextProvider { inner }).await;
    }
}
