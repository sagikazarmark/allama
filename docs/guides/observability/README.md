# Observability with allama

This guide covers what allama records about model calls and how to get it
into an OpenTelemetry backend: the versions that work together, wiring
the subscriber, capturing message content safely, propagating trace
context to gateways, deriving metrics in the Collector, and attributing
calls to the provider that served them.

allama records two kinds of telemetry, both through
[`tracing`](https://docs.rs/tracing):

- **Spans**, from `TracingProvider` (feature `tracing`): one span per
  chat, embedding or decision call, with the attributes of the
  OpenTelemetry GenAI semantic conventions and a summary of every retry
  and failover inside it, collected through its
  [call scope](../../../CONTEXT.md#call-scope). The
  [`TracingProvider` docs](https://docs.rs/allama/latest/allama/wrappers/struct.TracingProvider.html)
  list every attribute.
- **Events**, with the default `warning-events` feature:
  [warning events](../../../CONTEXT.md#warning-event) under
  the `allama::warning` targets, retries under `allama::retry`, failovers
  under `allama::fallback` (both `INFO`; a failover's `WARN` line is its
  `allama::warning::fallback` warning), and abnormal finishes under
  `allama::finish`. These targets are stable: filter on them. Any `tracing` subscriber shows
  them, with or without `TracingProvider`; the
  [applications guide](../applications.md#warnings) explains who each
  warning channel is for.

allama emits no OpenTelemetry metrics. The spans carry what the GenAI
client metrics need, and the Collector derives them
([Metrics](#metrics)).

## Contents

1. [Versions](#versions)
2. [Wiring the subscriber](#wiring-the-subscriber)
3. [Content capture and redaction](#content-capture-and-redaction)
4. [Trace context propagation](#trace-context-propagation)
5. [Metrics](#metrics)
6. [Which provider served the call](#which-provider-served-the-call)
7. [Reading the traces](#reading-the-traces)

## Versions

| Piece | Version | Why |
| --- | --- | --- |
| GenAI semantic conventions | v1.41.0 or later | Every `gen_ai.*` name allama emits is defined there; everything else is `allama.*`. |
| `tracing-opentelemetry`, for spans | 0.32 or later | allama sets the `otel.*` span fields (name, kind, status) as 0.32 reads them. With 0.31 or earlier, the status description exports as a plain attribute. |
| `opentelemetry`, `opentelemetry_sdk` | the ones your `tracing-opentelemetry` uses | `allama` itself does not depend on them. |
| Trace context propagation | the ones your application uses | allama sends no `traceparent` itself: your code injects it ([recipe](#trace-context-propagation)). The recipe is written against `opentelemetry` 0.33 and `tracing-opentelemetry` 0.34. |

No allama crate pins an OpenTelemetry release: spans work with any
`tracing-opentelemetry` from 0.32, and propagation runs in your code.

## Wiring the subscriber

Install a `tracing-opentelemetry` layer over an OpenTelemetry tracer, and
wrap the provider in `TracingProvider`. Here the spans go to an
in-memory exporter (`opentelemetry_sdk` with its `testing` feature), so
the program runs without a collector:

```rust,no_run
use allama::mock::MockProvider;
use allama::prelude::*;
use allama::wrappers::TracingProvider;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};
use tracing_subscriber::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // The exporter decides where spans go: memory here, OTLP in production.
    let exporter = InMemorySpanExporter::default();
    let tracer_provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    tracing_subscriber::registry()
        .with(tracing_opentelemetry::layer().with_tracer(tracer_provider.tracer("my-app")))
        .init();

    let provider = TracingProvider::new(MockProvider::with_text("Hello!"));
    provider
        .chat(&ChatRequest::new("demo-model").user("Say hello"))
        .await?;

    // Export what is buffered before the process exits.
    tracer_provider.force_flush()?;
    for span in exporter.get_finished_spans()? {
        println!("{}", span.name); // chat demo-model
    }
    Ok(())
}
```

The [`otel`](../../../crates/allama/examples/otel.rs) example is the same
wiring as a runnable program, and checks that the `chat demo-model` span
arrives:

```text
cargo run -p allama --example otel --features mock,tracing
```

To send the spans to a collector instead, swap the exporter for the
OTLP one from `opentelemetry-otlp` (the release that matches your
`opentelemetry_sdk`, with a transport feature such as `grpc-tonic`), and
batch it:

```rust,ignore
let exporter = opentelemetry_otlp::SpanExporter::builder().with_tonic().build()?;
let tracer_provider = SdkTracerProvider::builder()
    .with_batch_exporter(exporter)
    .build();
```

**In the console, during development.** `TracingProvider` records spans,
not log events, so `tracing_subscriber`'s `fmt` subscriber with its
defaults prints nothing for a call that succeeds. Tell it to print each
span when it closes, which is when the call has recorded every
attribute:

```rust,no_run
use tracing_subscriber::fmt::format::FmtSpan;

tracing_subscriber::fmt()
    .with_span_events(FmtSpan::CLOSE)
    .init();
```

Each call is then one line, `chat{otel.name=chat demo-model …}: close`,
with its attributes. The
[`tracing`](../../../crates/allama/examples/tracing.rs) example prints
them, content capture included:

```text
cargo run -p allama --example tracing --features mock,tracing
```

- **Put `TracingProvider` outermost**, around retry, fallback and any
  deadline, so one span covers the whole call and its resilience summary
  counts every attempt. A deadline outside it would record its expiry as
  a cancelled call, not as `error.type = timeout`; the applications
  guide's [provider stack](../applications.md#the-provider-stack) shows
  the order.
- **Spans start on the first poll** of the call's future, as children of
  the span current then. Instrument the request handler, and the model
  calls nest under it.
- **A dropped call is not an error.** A call whose future or stream the
  caller dropped records `allama.response.cancelled = true`, with the
  span status unset.

## Content capture and redaction

Spans carry no message content unless you ask for it:

```rust,no_run
use allama::mock::MockProvider;
use allama::wrappers::{TracingContentConfig, TracingProvider};

let provider = TracingProvider::new(MockProvider::with_text("Hello!"))
    .with_content_capture(TracingContentConfig::all());
```

Captured content goes to `gen_ai.system_instructions`,
`gen_ai.input.messages` and `gen_ai.output.messages`, as JSON in the
conventions' shape.

A span records the request `TracingProvider` was given. To keep text out
of captured content with a redaction or sanitization transform
(`MapRequestProvider`), put the transform **outside** `TracingProvider`:
`MapRequestProvider::new(TracingProvider::new(p), redact)`. Inside it, the
span records the caller's request before the transform runs.

- **Keys are redacted, text is not.** In every JSON object of a payload
  (tool call arguments, object tool results, other content), a key whose
  name, lowercased with `-` and `_` removed, is `apikey`,
  `authorization`, `password`, `secret`, `token`, `accesstoken`,
  `refreshtoken` or `clientsecret`, or contains `apikey`, `secret` or
  `token`, has its value replaced by `"[REDACTED]"`. So `api_key`,
  `X-Api-Key`, `client_secret` and `refreshToken` all match.
- **Text is recorded verbatim**: user, assistant and system messages, text
  tool results, reasoning and refusals. A secret a user pastes into a
  message is captured as written. Capture content only where the trace
  backend may hold what your users send.
- **Inline media data is left out** unless you set `capture_media_data`.
  An image or file sent as base64 keeps its part, with its media type and
  its decoded size in `allama.omitted_content_bytes` in place of the
  data, so one image does not use up the payload limit and cut off the
  text after it. Media by URL is recorded as its URI without the query
  and fragment, where presigned URLs (`X-Amz-Signature`, `sig=`) carry
  their credentials, unless you set `capture_media_urls`.
- **Payloads are truncated** after `max_payload_chars` characters (4096 by
  default), with `...[TRUNCATED]` appended, so a truncated payload is no
  longer valid JSON. Redaction runs before the cut.

The
[`TracingContentConfig` docs](https://docs.rs/allama/latest/allama/wrappers/struct.TracingContentConfig.html)
state the same rule.

## Trace context propagation

`TracingProvider` traces the call in your process. To continue the trace
in a gateway or server that joins distributed traces (LiteLLM proxy,
vLLM, an instrumented internal gateway), send the current span's W3C
`traceparent` with each request. allama sends none and depends on no
OpenTelemetry crate, so your application adds it, with its own
OpenTelemetry version and no pin to keep in step. The
[`trace_context`](../../../crates/allama-examples/examples/trace_context.rs)
example is this recipe as a program, with tests, written against
`opentelemetry` and `opentelemetry_sdk` 0.33 and `tracing-opentelemetry`
0.34:

```text
cargo run -p allama-examples --example trace_context
```

The span context comes from `tracing-opentelemetry`, so the subscriber
needs its layer ([Wiring the subscriber](#wiring-the-subscriber)). The
context goes out through the global propagator, which sends nothing until
the application installs one; the W3C `TraceContextPropagator` sends
`traceparent` and `tracestate`:

```rust,ignore
opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());
```

One function writes the context into a request's headers. It never
overwrites a `traceparent` the caller set:

```rust,ignore
/// Write the current span's trace context into `headers` through the
/// global propagator, unless they already carry a `traceparent`.
fn inject_trace_context(headers: &mut Headers) {
    if headers.get("traceparent").is_some() {
        return;
    }
    let context = tracing::Span::current().context();
    opentelemetry::global::get_text_map_propagator(|propagator| {
        propagator.inject_context(&context, &mut HeaderInjector(headers));
    });
}

/// Writes the propagator's fields as headers, one value per name.
struct HeaderInjector<'a>(&'a mut Headers);

impl Injector for HeaderInjector<'_> {
    fn set(&mut self, name: &str, value: String) {
        self.0.remove(name);
        self.0.insert(name, value);
    }
}
```

Where it runs depends on how you build the provider.

**A provider you build: wrap the transport.** An `HttpTransport` wrapper
injects into `HttpRequest.headers` just before it sends. Adapters apply
the caller's `HttpRequestOptions` before they send, so it sees the final
headers, and a `traceparent` the caller set there wins. It covers every
call the provider makes (chat, embedding and decision) and every retry
attempt:

```rust,ignore
struct PropagatingTransport<T> {
    inner: T,
}

impl<T: HttpTransport> HttpTransport for PropagatingTransport<T> {
    fn send(&self, mut request: HttpRequest) -> SendFuture<'_> {
        inject_trace_context(&mut request.headers);
        self.inner.send(request)
    }
}
```

Hand it to the builder, around any transport. The example wraps a
`ReplayTransport` (`network`), so that it runs without a key; in
production, wrap `ReqwestTransport::new()?`:

```rust,ignore
let provider = TracingProvider::new(
    allama_openai::Provider::builder()
        .api_key("demo-key")
        .http_transport(PropagatingTransport {
            inner: network.clone(),
        })
        .build()?,
);
```

**A provider from the registry: wrap the chat provider.**
`allama_providers::chat::from_model` builds the provider and its
transport itself, and takes no transport. A chat wrapper adds the header
to the request's `HttpRequestOptions` instead, which every HTTP adapter
sends:

```rust,ignore
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
```

Put it inside `TracingProvider`. The example looks the address up in a
`Registry` of its own, over a replayed network; the provider that
`allama_providers::chat::from_model("openai:gpt-5-mini")?` returns is
wrapped the same way:

```rust,ignore
let (provider, model) = registry.chat_from_model("openai:gpt-5-mini")?;
let provider = TracingProvider::new(TraceContextProvider { inner: provider });
```

The chat wrapper has limits the transport wrapper does not:

- **It clones the request** on each call, to add the header without
  changing the caller's request.
- **It must sit inside `TracingProvider`**, so that the current span is
  the GenAI span. Outside it, the header names the caller's span, or no
  span at all.
- **It adds the header once per call**, not once per retry attempt, so
  every attempt of a call carries the same `traceparent`.
- **It covers chat only.** Embedding and decision providers from the
  registry need the same wrapper over `EmbeddingProvider` and
  `DecisionProvider`.

## Metrics

Derive `gen_ai.client.operation.duration` and `gen_ai.client.token.usage`
from the spans in the OpenTelemetry Collector.
[`otelcol-genai-metrics.yaml`](otelcol-genai-metrics.yaml) does so with
the `signal_to_metrics` connector (OpenTelemetry Collector Contrib), with
the conventions' names, units and buckets, and `allama.response.cancelled`
as an optional dimension. It runs the `transform/genai_provider`
processor first, which attributes each span to the provider that served
it ([Which provider served the call](#which-provider-served-the-call)).
Run it with:

```text
otelcol-contrib --config docs/guides/observability/otelcol-genai-metrics.yaml
```

and export the spans to it over OTLP. The file's comments cover the
pipeline order, cumulative backends such as Prometheus, and forwarding
the spans to a tracing backend.

To break the metrics down by something of your own (a tenant, a
conversation), put it on the GenAI span itself: a parent span's
attributes do not reach metrics derived from the GenAI span.
`TracingProvider::on_span` hands you each span before the call runs;
record `gen_ai.conversation.id`, which chat and decision spans declare, or
set any attribute with `tracing-opentelemetry`'s `OpenTelemetrySpanExt`.
Carry the per-call value on the request with `with_opaque_option`.

Without a Collector (a Prometheus pull endpoint in the service),
`ObserveProvider` hands each call's outcome to callbacks: its usage, its
error kind and its latency.

## Which provider served the call

Four identifiers name a provider, and they answer different questions:

| Identifier | Example | What it names | Where you see it |
| --- | --- | --- | --- |
| [Provider key](../../../CONTEXT.md#provider-key) | `openai-chat` | A configuration in `allama_providers`: which API, which environment variables | Model addresses (`openai-chat:gpt-5-mini`), `Registry` |
| [Provider name](../../../CONTEXT.md#provider-name) | `openai` | Who serves the call, as the provider reports it | `ProviderIdentity::provider_name`, `ChatResponse::provider` (the serving provider), errors, the `[openai]` prefix of warnings |
| `gen_ai.provider.name` | `openai`, `azure.ai.openai` | The provider the request was addressed to, as the conventions' well-known value | Spans, set when the span starts |
| `allama.response.provider` | `anthropic` | The provider that served the call, in the same value space; when nothing served, the one that ended it | Spans, recorded when the call ends |

Two hosts of one provider (two Ollama boxes) share a provider name. The
spans tell them apart by `server.address` and `server.port`, the endpoint
the request that ended the call was sent to, so the serving host after a
failover.

`otel_genai_provider_name` maps a provider name to its conventions value
(`azure` to `azure.ai.openai`, `gemini` to `gcp.gemini`, `xai` to `x_ai`);
other names pass through.

The last two differ after a failover, when the
[serving provider](../../../CONTEXT.md#serving-provider) is not the addressed one. `gen_ai.provider.name` is the
primary, the provider the request was addressed to, set when the span
starts so samplers see it. After a failover, the response, its tokens
and its cost belong to the fallback side, which `allama.response.provider`
names. When every side failed, it names the side whose error ended the
call, as `error.type` and `server.address` do, so an error rate grouped by
it blames the provider that failed. It is on every finished span, so query
and group by it directly, with no fallback to the other attribute:

```text
{ span.allama.response.provider = "anthropic" }
{ span.allama.fallback.used = true } | by(span.allama.response.provider)
```

Backends that read only `gen_ai.provider.name` (Datadog LLM
Observability, Langfuse, OpenInference mappings) need it rewritten in the
Collector. The `TracingProvider` docs give the `transform` processor, and
[`otelcol-genai-metrics.yaml`](otelcol-genai-metrics.yaml) runs it in
front of the metrics it derives.

## Reading the traces

Besides the provider, the resilience summary on each span answers the
question an interactive server gets asked most: **how often a rate limit
reached the client.**

When a provider's `Retry-After` is longer than `max_retry_after`, the
retry layer gives up at once and records
`allama.retry.give_up_reason = "hint_over_ceiling"`. On a failed call
(`error.type = "rate_limited"`), that is the `429` the client received,
with the provider's hint. With a fallback, the same reason on a
successful call with `allama.fallback.used = true` means the rate limit
failed over instead:

```text
{ span.allama.retry.give_up_reason = "hint_over_ceiling" && span.error.type = "rate_limited" }
```

If that count matters to you, configure a fallback, or raise
`max_retry_after` as far as your users will wait. The other reasons are
`max_attempts`, `budget` and `not_retryable`.

The rest of the summary (`allama.attempts`, the retry delays, budget
denials, `allama.fallback.attempts`) and the `allama.retry` and
`allama.fallback` attempt events are listed in the `TracingProvider`
docs.
