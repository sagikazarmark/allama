use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures_core::Stream;
use serde::Serialize;
use tracing::{Instrument, Span};
// `std::time::Instant::now` panics on `wasm32-unknown-unknown`.
use web_time::Instant;

use super::call_scope::{self, Scope, Summary};
use crate::chat::stream::StreamCollector;
use crate::{
    ChatProvider, ChatRequest, ChatResponse, ChatStream, ContentBlock, ContentPart,
    DecisionProvider, DecisionRequest, DecisionResponse, EmbeddingProvider, EmbeddingRequest,
    EmbeddingResponse, Error, ExtractionProvider, ExtractionRequest, ExtractionResponse,
    FinishReason, MediaSource, Message, ProviderIdentity, ResponseFormat, Result, StreamEvent,
    SystemPrompt, ToolResultContent, Usage, UserContent, Warning,
};

/// Emits OpenTelemetry GenAI spans (via `tracing`) around chat, embedding, and
/// decision calls.
///
/// The repository's
/// [observability guide]
/// wires it to OpenTelemetry: versions, the subscriber, a recipe that
/// propagates trace context to gateways, metrics, and reading the spans.
///
/// Message content is captured only when enabled with
/// [`TracingContentConfig`]; secret-looking keys are
/// [redacted](TracingContentConfig#redaction), message text is recorded
/// verbatim, and payloads are truncated. Warnings are recorded as span
/// events.
///
/// # Conventions and versions
///
/// Spans follow the OpenTelemetry [GenAI semantic conventions] for client
/// spans, **v1.41.0 or later** (the GenAI conventions now continue in
/// [`open-telemetry/semantic-conventions-genai`]). Every `gen_ai.*` name
/// allama emits is defined as a span attribute in that version; anything
/// else allama records is named `allama.*`. Attribute names, types, and
/// units are public API: allama's tests export spans through
/// `tracing-opentelemetry` and check them against the pinned span tables.
///
/// Export the spans with [`tracing-opentelemetry`] **0.32 or later**: the
/// `otel.*` fields allama sets (`otel.name`, `otel.kind`,
/// `otel.status_code`, and `otel.status_description`) follow that version.
/// 0.31 and earlier read the span status description from a differently
/// named field, so it would export as a plain attribute instead.
///
/// Where `tracing` cannot express a convention type:
///
/// - `string[]` attributes (`gen_ai.request.stop_sequences`,
///   `gen_ai.response.finish_reasons`) are JSON arrays encoded as strings,
///   since `tracing` fields cannot carry arrays;
/// - structured (`any`) attributes (`gen_ai.system_instructions`,
///   `gen_ai.input.messages`, `gen_ai.output.messages`) are JSON strings, as
///   the conventions allow;
/// - integers are 64-bit signed; a value beyond `i64::MAX` (a large
///   `gen_ai.request.seed`) is recorded as its decimal string.
///
/// # Metrics
///
/// allama emits spans, not OpenTelemetry metrics. The spans carry what the
/// GenAI client metrics need, so derive `gen_ai.client.operation.duration`
/// and `gen_ai.client.token.usage` from them in the OpenTelemetry Collector:
/// `docs/guides/observability/otelcol-genai-metrics.yaml` in the repository
/// does so with the `signal_to_metrics` connector, after attributing each
/// span to its serving provider (see
/// [Provider attribution](#provider-attribution)), and adds
/// `allama.response.cancelled` as an optional dimension.
///
/// # Spans
///
/// Each call gets one span named `{operation} {model}`, of kind `CLIENT`.
/// The span starts when the call is issued, on the first poll of the call's
/// future, as a child of the span current then; a future that is never
/// polled sends nothing and creates no span. Attributes are recorded when
/// the value is known.
///
/// **Chat** (`gen_ai.operation.name = "chat"`, unary and streaming):
///
/// - Request: `gen_ai.operation.name`, `gen_ai.provider.name` (the
///   addressed provider; see [Provider attribution](#provider-attribution)),
///   `gen_ai.request.model`,
///   `gen_ai.request.max_tokens`, `gen_ai.request.temperature`,
///   `gen_ai.request.top_p`, `gen_ai.request.top_k`, `gen_ai.request.frequency_penalty`,
///   `gen_ai.request.presence_penalty`, `gen_ai.request.stop_sequences`,
///   `gen_ai.request.seed`, `gen_ai.output.type`; `allama.request.provider`
///   (string).
/// - Response: `gen_ai.response.id`, `gen_ai.response.model`,
///   `gen_ai.response.finish_reasons`, `gen_ai.usage.input_tokens`,
///   `gen_ai.usage.output_tokens`, `gen_ai.usage.cache_read.input_tokens`,
///   `gen_ai.usage.cache_creation.input_tokens`,
///   `gen_ai.usage.reasoning.output_tokens`; `allama.response.provider` and
///   `allama.response.request_id` (strings), `allama.response.warnings`
///   (int, the warning count).
/// - Streaming: `gen_ai.request.stream` (`true`);
///   `gen_ai.response.time_to_first_chunk` (double, seconds until the first
///   stream event of any kind, as the conventions define it);
///   `allama.response.time_to_first_output` (double, seconds until the first
///   model-output event, see [`StreamEvent::is_output`]).
///
///   Both time-to-first measures start when the call is issued, with the
///   span (the first poll of the `chat_stream` future), so they include the
///   time to the response headers. They differ when a provider sends
///   metadata such as `ResponseStart` before the model produces output: the
///   first output is the latency a user perceives. When this wrapper wraps
///   [`RetryingProvider`](super::RetryingProvider) or
///   [`FallbackProvider`](super::FallbackProvider), both include failed
///   attempts, retry backoff, and fallback time; those wrappers hold events
///   back until the first output, so the two measures are then close.
///
///   On a cancelled stream (see [Cancellation](#cancellation)) the response
///   attributes describe what arrived before the drop.
/// - Content, opt-in: `gen_ai.system_instructions`, `gen_ai.input.messages`,
///   `gen_ai.output.messages`.
///
/// **Embeddings** (`gen_ai.operation.name = "embeddings"`):
/// `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`,
/// `gen_ai.embeddings.dimension.count`, `gen_ai.response.model`,
/// `gen_ai.usage.input_tokens`; `allama.embeddings.input_count` (int),
/// `allama.request.provider`, `allama.response.provider`,
/// `allama.response.cancelled`.
///
/// **Decision** (`gen_ai.operation.name = "decision"`, which is not a
/// well-known value in the conventions): `gen_ai.operation.name`,
/// `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.id`,
/// `gen_ai.response.model`, `gen_ai.usage.input_tokens`,
/// `gen_ai.usage.output_tokens`; `allama.decision.question_count` (int),
/// `allama.request.provider`, `allama.response.provider`,
/// `allama.response.request_id`, `allama.response.warnings`,
/// `allama.response.cancelled`.
///
/// **Extraction** (`gen_ai.operation.name = "extraction"`, not a well-known
/// value either; the operation is unstable, ADR 0018): the decision
/// attributes, with `allama.extraction.field_count` (int) in place of the
/// question count.
///
/// **Server**, on every span: `server.address` and `server.port`, the
/// endpoint the last request of the call was sent to, so the serving host
/// after a failover between two hosts of one provider. The HTTP transport
/// reports it ([`CallFact::server`](super::call_scope::CallFact::server));
/// a provider whose transport does not leaves them unset.
///
/// **Application attributes**: chat and decision spans declare
/// `gen_ai.conversation.id`, which allama never sets itself;
/// a callback registered with [`on_span`](Self::on_span) sets it, and any
/// attribute of the application's own (a tenant), on the span itself.
///
/// **Resilience summary**, recorded on chat, embedding, and decision spans,
/// once per call, when it completes or its future or stream is dropped.
/// Wrappers inside this one ([`RetryingProvider`](super::RetryingProvider),
/// [`FallbackProvider`](super::FallbackProvider), and third-party wrappers)
/// report what they did to the call scope this wrapper opens (see
/// [`call_scope`](super::call_scope)), and the span records the sum:
///
/// | Field | Meaning |
/// |---|---|
/// | `allama.attempts` (int) | physical attempts across all providers, counting the first; a failed stream setup and a stream discarded before its first output each count as one, and so does a fallback side without a retry wrapper (`Fallback(a, b)` with `a` failing counts 2). Always recorded. |
/// | `allama.retry.total_delay_ms` (int) | total backoff waited, in milliseconds |
/// | `allama.retry.last_delay_ms` (int) | the last backoff waited |
/// | `allama.retry.server_directive` (bool) | the last server retry directive that decided whether to retry (see [`RetryPolicy`](super::RetryPolicy)); absent if none did |
/// | `allama.retry.budget_denied` (bool) | `true` when a [`RetryBudget`](super::RetryBudget) refused a retry the policy allowed |
/// | `allama.retry.give_up_reason` (string) | why the last retry layer to give up returned the error: `max_attempts`, `hint_over_ceiling`, `budget`, or `not_retryable` |
/// | `allama.fallback.used` (bool) | `true` when the call failed over |
/// | `allama.fallback.attempts` (int) | providers that failed, each counted once, also in nested chains: those tried before the one that served, or all of them when the call failed |
/// | `allama.fallback.error_type` (string) | the error type of the last failure that triggered a failover |
///
/// Fields without a value are not recorded. With retry layers on both sides
/// of a fallback, the delays add up and the last values come from the side
/// that ran last. A `TracingProvider` nested inside another keeps what is
/// reported inside it on its own span; the outer span counts the inner call
/// as one attempt. Work spawned onto another task is outside the call scope,
/// and what it reports is not seen. Token usage on the span is the serving
/// attempt's: usage a discarded attempt reported before it failed is not
/// added.
///
/// **Attempt events**: one event per failed attempt that was recovered
/// from, or whose retry a budget denied, with what the summary leaves out:
///
/// - `allama.retry`, emitted by [`RetryingProvider`](super::RetryingProvider)
///   before it waits: `allama.attempt.provider` (the provider that failed),
///   `error.type`, `allama.retry.attempt` (int, the number of the failed
///   attempt, from 1), `allama.retry.max_attempts` (int, the policy's
///   limit), and `allama.retry.delay_ms` (int, the wait before the next
///   attempt);
/// - `allama.retry.budget_denied`, emitted by
///   [`RetryingProvider`](super::RetryingProvider) when its
///   [`RetryBudget`](super::RetryBudget) denies a retry and it returns the
///   error: `allama.attempt.provider`, `error.type`, and
///   `allama.retry.attempt` (int, the number of the failed attempt);
/// - `allama.fallback`, emitted by
///   [`FallbackProvider`](super::FallbackProvider) when it fails over:
///   `allama.attempt.provider` (the primary that failed), `error.type`, and
///   `allama.fallback.provider` (the provider it fails over to).
///
/// Provider names on attempt events and in `allama.request.provider` and
/// `allama.response.provider` are [`otel_genai_provider_name`] values, like
/// `gen_ai.provider.name`. These events are emitted in the current span, so
/// they also appear without this wrapper wherever the application's
/// subscriber records events: `allama.retry` at `INFO` and
/// `allama.retry.budget_denied` at `WARN`, under the target
/// `allama::retry` (also with the `warning-events` feature alone), and
/// `allama.fallback` at `INFO` under the target `allama::fallback` (likewise).
///
/// **Errors**: `error.type`, the [`Error`] kind (for example `rate_limited`),
/// and span status `Error`. The status description is the redacted
/// [`Error::as_log`] view without its leading `kind=`, which `error.type`
/// already carries; it never includes the response body. A successful call
/// leaves the status unset. A stream records a failure to start and an
/// error the stream yields the same way.
///
/// **Warning events**: one event per warning, named by the warning message,
/// with `allama.warning.kind` (its snake_case name) and
/// `allama.warning.provider`. With the `warning-events` feature, the
/// warning event [`warning::emit`](crate::warning::emit) sent while the call
/// ran is that event, with more fields, and this wrapper adds none for it:
/// the span records the warning instead of emitting it again. A coded
/// warning seen before in the process is that event at `DEBUG`, so a
/// subscriber that filters span events by level drops it (see
/// [warning events](crate::warning#warning-events)).
///
/// # Cancellation
///
/// `allama.response.cancelled` (bool) is `true`, on every kind of span,
/// exactly when the caller abandoned the call:
///
/// - it dropped the call's future after polling it and before the call
///   completed, unary or streaming; under
///   [`RetryingProvider`](super::RetryingProvider) and
///   [`FallbackProvider`](super::FallbackProvider) that includes a drop
///   while they wait for a stream's first output;
/// - it dropped the stream before the response ended. The response ends
///   with its `ResponseStop` event, so a consumer that stops reading there
///   did not cancel; nor did one whose stream ended or yielded an error.
///
/// Otherwise the attribute is absent. A future never polled has no span,
/// and a future or stream dropped while the thread panics is not marked.
/// Cancellation is not an error: the status stays unset and `error.type` is
/// not recorded.
///
/// A deadline outside this wrapper, as in `Timeout(Tracing(p))`, expires by
/// dropping the call, so the span records the expiry as cancelled and it is
/// missing from error rates. Put deadlines inside this wrapper
/// (`Tracing(Timeout(..))`): their expiry is then a failed call with
/// `error.type = timeout`.
///
/// # Provider attribution
///
/// `gen_ai.provider.name` is the provider the request was addressed to, the
/// [`otel_genai_provider_name`] value of the wrapped provider's name. It is
/// set when the span starts, so samplers see it, as the conventions ask,
/// and it is never recorded again. `allama.request.provider` repeats it.
///
/// `allama.response.provider` names the provider that served the call, in
/// the same value space. It is recorded once on every span, when the call
/// completes or is dropped:
///
/// - whenever a provider served, that provider, whatever the outcome: a
///   stream that reached its first output and then failed or was dropped
///   still happened on that provider, which also incurred its tokens. It
///   comes from the response ([`ChatResponse::provider`], a stream's
///   `ResponseMetadata` provider, [`EmbeddingResponse::provider`],
///   [`DecisionResponse::provider`]), or, when the response names none,
///   from the call scope, where [`FallbackProvider`](super::FallbackProvider)
///   reports the side that served, innermost first;
/// - when nothing served, the provider that ended the call: for a failure,
///   the provider the error names (the last side's, when every side of a
///   fallback chain failed); for a drop, the last side a
///   [`FallbackProvider`](super::FallbackProvider) tried;
/// - otherwise (an error a wrapper raised, which names no provider, or a
///   drop with no fallback side started), the addressed provider.
///
/// The two differ after a failover; then the response model, id, and usage
/// on the span belong to the serving provider. When every side failed, the
/// span's `error.type` and `server.address` are those of the side that
/// ended the call, and so is `allama.response.provider`, so error metrics
/// grouped by it blame the provider that failed. Query and group spans by
/// `allama.response.provider` directly (TraceQL, APM queries); it is always
/// present on a finished span.
///
/// Backends that read `gen_ai.provider.name` itself (Datadog LLM
/// Observability, Langfuse, OpenInference mappings, Tempo and Jaeger
/// queries) see the addressed provider after a failover. To attribute calls
/// to the serving provider there, rewrite the attribute in the
/// OpenTelemetry Collector with the `transform` processor:
///
/// ```yaml
/// processors:
///   transform/genai_provider:
///     error_mode: ignore
///     trace_statements:
///       - set(span.attributes["gen_ai.provider.name"], span.attributes["allama.response.provider"]) where span.attributes["allama.response.provider"] != nil
/// ```
///
/// Put it in the traces pipeline in front of the exporters that feed those
/// backends. A `tail_sampling` processor placed before it still sees the
/// addressed provider, which is often what sampling wants.
/// `docs/guides/observability/otelcol-genai-metrics.yaml` in the repository
/// runs it in front of the metrics it derives.
///
/// [GenAI semantic conventions]: https://github.com/open-telemetry/semantic-conventions/blob/v1.41.0/docs/gen-ai/gen-ai-spans.md
/// [`open-telemetry/semantic-conventions-genai`]: https://github.com/open-telemetry/semantic-conventions-genai
/// [`tracing-opentelemetry`]: https://docs.rs/tracing-opentelemetry
#[doc = concat!(
    "[observability guide]: https://github.com/sagikazarmark/allama/blob/v",
    env!("CARGO_PKG_VERSION"),
    "/docs/guides/observability/README.md",
)]
#[derive(Clone)]
pub struct TracingProvider<T> {
    inner: T,
    content_config: TracingContentConfig,
    span_hook: Option<SpanHook>,
}

/// What [`TracingProvider::on_span`] calls with each span.
type SpanHook = Arc<dyn Fn(&Span, TracedRequest<'_>) + Send + Sync>;

/// The request a [`TracingProvider`] span is about, handed to its
/// [span hook](TracingProvider::on_span).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum TracedRequest<'a> {
    /// A `chat` or `chat_stream` call.
    Chat(&'a ChatRequest),
    /// An `embed` call.
    Embedding(&'a EmbeddingRequest),
    /// A `decide` call.
    Decision(&'a DecisionRequest),
    /// An `extract_values` call.
    Extraction(&'a ExtractionRequest),
}

impl<T> fmt::Debug for TracingProvider<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TracingProvider")
            .field("content_config", &self.content_config)
            .finish_non_exhaustive()
    }
}

impl<T> TracingProvider<T> {
    /// Wrap `inner` without content capture.
    #[must_use]
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            content_config: TracingContentConfig::default(),
            span_hook: None,
        }
    }

    /// Call `hook` with each span right after it is created, before the
    /// call runs, to add attributes of the application's own: a tenant, an
    /// endpoint label, the conversation.
    ///
    /// Chat and decision spans declare the semantic conventions'
    /// `gen_ai.conversation.id`, so `span.record("gen_ai.conversation.id",
    /// id)` sets it. Any other
    /// attribute goes through the OpenTelemetry layer, with
    /// `tracing_opentelemetry::OpenTelemetrySpanExt::set_attribute`: a
    /// `tracing` span records only the fields it declared. Attributes on
    /// the GenAI span itself, unlike on a parent span, reach the
    /// Collector's metrics derived from it (per-tenant token counts).
    ///
    /// The request carries per-call values as
    /// [opaque options](crate::ChatRequest::with_opaque_option):
    ///
    /// ```
    /// # #[cfg(feature = "mock")] {
    /// use allama::mock::MockProvider;
    /// use allama::wrappers::{TracedRequest, TracingProvider};
    /// use allama::{ChatProvider, ChatRequest};
    ///
    /// #[derive(Clone)]
    /// struct Conversation(String);
    ///
    /// let provider = TracingProvider::new(MockProvider::with_text("ok")).on_span(
    ///     |span, request| {
    ///         if let TracedRequest::Chat(request) = request
    ///             && let Some(Conversation(id)) = request.option::<Conversation>()
    ///         {
    ///             span.record("gen_ai.conversation.id", id.as_str());
    ///         }
    ///         // With tracing-opentelemetry's `OpenTelemetrySpanExt`:
    ///         // span.set_attribute("app.tenant", tenant);
    ///     },
    /// );
    /// let request = ChatRequest::new("m")
    ///     .user("Hi")
    ///     .with_opaque_option(Conversation("conv_42".into()));
    /// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
    /// provider.chat(&request).await.unwrap();
    /// # });
    /// # }
    /// ```
    #[must_use]
    pub fn on_span(
        mut self,
        hook: impl Fn(&Span, TracedRequest<'_>) + Send + Sync + 'static,
    ) -> Self {
        self.span_hook = Some(Arc::new(hook));
        self
    }

    fn run_span_hook(&self, span: &Span, request: TracedRequest<'_>) {
        if let Some(hook) = &self.span_hook {
            hook(span, request);
        }
    }

    /// Enable content capture.
    #[must_use]
    pub fn with_content_capture(mut self, config: TracingContentConfig) -> Self {
        self.content_config = config;
        self
    }

    /// Borrow the wrapped provider.
    #[must_use]
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Unwrap the provider.
    #[must_use]
    pub fn into_inner(self) -> T {
        self.inner
    }
}

/// Map an allama provider name to the GenAI `gen_ai.provider.name` value.
///
/// The conventions list well-known values that MUST be used when one
/// applies. Provider names with a well-known value (as of semantic
/// conventions v1.41.0) are mapped to it; every other name passes through
/// unchanged.
///
/// | Provider name | `gen_ai.provider.name` |
/// |---|---|
/// | `azure` | `azure.ai.openai` |
/// | `gemini`, `google-ai` | `gcp.gemini` |
/// | `google-vertex` | `gcp.vertex_ai` |
/// | `mistral` | `mistral_ai` |
/// | `xai` | `x_ai` |
///
/// `anthropic`, `deepseek`, `groq`, and `openai` are already well-known
/// values.
///
/// ```
/// use allama::wrappers::otel_genai_provider_name;
///
/// assert_eq!(otel_genai_provider_name("xai"), "x_ai");
/// assert_eq!(otel_genai_provider_name("openai"), "openai");
/// assert_eq!(otel_genai_provider_name("ollama"), "ollama");
/// ```
#[must_use]
pub fn otel_genai_provider_name(provider_name: &str) -> &str {
    super::semconv_provider_name(provider_name)
}

/// What message content to capture on spans.
///
/// Off by default. When on, the content is recorded as JSON in
/// `gen_ai.system_instructions` and `gen_ai.input.messages`
/// ([`capture_input_messages`](Self::capture_input_messages)) and
/// `gen_ai.output.messages`
/// ([`capture_output_messages`](Self::capture_output_messages)), in the
/// shape the GenAI conventions define.
///
/// # Redaction
///
/// Redaction matches **keys**, never text. In every JSON object of a
/// captured payload (tool call arguments, object tool results, other
/// content), a key is redacted, its value replaced by `"[REDACTED]"`, when,
/// lowercased and with `-` and `_` removed, it:
///
/// - is `apikey`, `authorization`, `password`, `secret`, `token`,
///   `accesstoken`, `refreshtoken` or `clientsecret`; or
/// - contains `apikey`, `secret` or `token`.
///
/// So `api_key`, `apiKey`, `X-Api-Key`, `client_secret`, `refreshToken`
/// and `bearer_token` all match. The rule errs towards redaction, like
/// `allama_http::is_sensitive_header`, so a field such as `max_tokens` in
/// tool arguments is redacted too. It leaves out that rule's `key` and
/// `auth` fragments, which would redact ordinary argument names such as
/// `keywords` and `author`.
///
/// **Text is recorded verbatim**: user, assistant and system message text,
/// text tool results, reasoning and refusals. A secret a user pastes into a
/// message, or a tool returns as text, is captured as written. Capture
/// content only where the trace backend may hold what your users send.
///
/// # Media
///
/// Inline media (an image or file sent as base64) is left out by default,
/// so that one image does not use up the payload limit and cut off the
/// text after it. Its part keeps its `type` (`blob`), `modality` and
/// `mime_type`, and records, in place of the conventions' `content`, the
/// decoded size in bytes as `allama.omitted_content_bytes`:
///
/// ```json
/// {"type":"blob","modality":"image","mime_type":"image/png","allama.omitted_content_bytes":12345}
/// ```
///
/// Set [`capture_media_data`](Self::capture_media_data) to record the
/// base64 data as `content`, and raise
/// [`max_payload_chars`](Self::max_payload_chars) to fit it.
///
/// Media by URL is recorded as its `uri` without the query and the
/// fragment, where presigned URLs carry their credentials
/// (`X-Amz-Signature`, `sig=`): `https://bucket.example/cat.png?X-Amz-Signature=...`
/// is recorded as `https://bucket.example/cat.png`. Set
/// [`capture_media_urls`](Self::capture_media_urls) to record URLs as they
/// are.
///
/// ```
/// use allama::wrappers::TracingContentConfig;
///
/// let mut config = TracingContentConfig::all();
/// config.capture_media_data = true;
/// config.max_payload_chars = 1 << 20;
/// ```
///
/// # Truncation
///
/// Each attribute's JSON is cut after
/// [`max_payload_chars`](Self::max_payload_chars) characters, and
/// `...[TRUNCATED]` is appended, so a truncated payload is no longer valid
/// JSON. Redaction runs first, so a cut never exposes a redacted value.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct TracingContentConfig {
    /// Capture system instructions and input messages.
    pub capture_input_messages: bool,
    /// Capture output messages.
    pub capture_output_messages: bool,
    /// Truncate captured payloads to this many characters.
    pub max_payload_chars: usize,
    /// Capture the data of inline media as well; off by default (see
    /// [Media](Self#media)).
    pub capture_media_data: bool,
    /// Capture media URLs whole, with their query and fragment; off by
    /// default, since presigned URLs carry credentials there (see
    /// [Media](Self#media)).
    pub capture_media_urls: bool,
}

impl Default for TracingContentConfig {
    fn default() -> Self {
        Self {
            capture_input_messages: false,
            capture_output_messages: false,
            max_payload_chars: 4096,
            capture_media_data: false,
            capture_media_urls: false,
        }
    }
}

impl TracingContentConfig {
    /// Capture input and output with the default payload limit, leaving out
    /// inline media data.
    #[must_use]
    pub fn all() -> Self {
        Self {
            capture_input_messages: true,
            capture_output_messages: true,
            ..Self::default()
        }
    }
}

impl<T> ChatProvider for TracingProvider<T>
where
    T: ChatProvider,
{
    type Stream = ChatStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        // The span starts when the call is issued, on first poll: a
        // future that is never polled sends nothing and creates no span.
        let addressed = self.inner.provider_name();
        let span = chat_span(addressed, request, &self.content_config);
        self.run_span_hook(&span, TracedRequest::Chat(request));
        let scope = Scope::new();
        let call = CallRecord::new(span.clone(), Arc::clone(&scope), addressed);
        let result = call_scope::within(&scope, self.inner.chat(request))
            .instrument(span.clone())
            .await;
        match &result {
            Ok(response) => {
                record_response(&span, response, &self.content_config);
                call.complete(response.provider.as_deref());
            }
            Err(err) => call.fail(err),
        }
        result
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
        // The call is issued on first poll, not when the future is
        // created: the span starts then, and so does the clock, which
        // covers stream setup (and any retry or fallback inside) as well
        // as the first chunk's arrival.
        let addressed = self.inner.provider_name();
        let span = chat_span(addressed, request, &self.content_config);
        span.record("gen_ai.request.stream", true);
        self.run_span_hook(&span, TracedRequest::Chat(request));
        let issued_at = Instant::now();
        let scope = Scope::new();
        let call = CallRecord::new(span.clone(), Arc::clone(&scope), addressed);
        let inner = match call_scope::within(&scope, self.inner.chat_stream(request))
            .instrument(span.clone())
            .await
        {
            Ok(stream) => stream,
            Err(err) => {
                call.fail(&err);
                return Err(err);
            }
        };
        let stream = TracingStream::new(
            inner,
            span,
            self.content_config,
            issued_at,
            scope,
            addressed.to_owned(),
        );
        // The stream records the summary when it ends or is dropped.
        call.hand_over();
        Ok(Box::pin(stream) as ChatStream)
    }

    super::forward_chat_questions!(inner);
}

impl<T> EmbeddingProvider for TracingProvider<T>
where
    T: EmbeddingProvider,
{
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        let addressed = self.inner.provider_name();
        let span = tracing::info_span!(
            "embeddings",
            "otel.name" = %format_args!("embeddings {}", request.model),
            "otel.kind" = "client",
            "otel.status_code" = tracing::field::Empty,
            "otel.status_description" = tracing::field::Empty,
            "gen_ai.operation.name" = "embeddings",
            "gen_ai.provider.name" = otel_genai_provider_name(addressed),
            "gen_ai.request.model" = %request.model,
            "allama.request.provider" = otel_genai_provider_name(addressed),
            "server.address" = tracing::field::Empty,
            "server.port" = tracing::field::Empty,
            "gen_ai.embeddings.dimension.count" = tracing::field::Empty,
            "allama.embeddings.input_count" = otel_count(request.inputs.len()),
            "gen_ai.usage.input_tokens" = tracing::field::Empty,
            "gen_ai.response.model" = tracing::field::Empty,
            "allama.response.provider" = tracing::field::Empty,
            "allama.attempts" = tracing::field::Empty,
            "allama.retry.total_delay_ms" = tracing::field::Empty,
            "allama.retry.last_delay_ms" = tracing::field::Empty,
            "allama.retry.server_directive" = tracing::field::Empty,
            "allama.retry.budget_denied" = tracing::field::Empty,
            "allama.retry.give_up_reason" = tracing::field::Empty,
            "allama.fallback.used" = tracing::field::Empty,
            "allama.fallback.attempts" = tracing::field::Empty,
            "allama.fallback.error_type" = tracing::field::Empty,
            "allama.response.cancelled" = tracing::field::Empty,
            "error.type" = tracing::field::Empty,
        );
        if let Some(dimensions) = request.dimensions {
            span.record("gen_ai.embeddings.dimension.count", i64::from(dimensions));
        }
        self.run_span_hook(&span, TracedRequest::Embedding(request));
        let scope = Scope::new();
        let call = CallRecord::new(span.clone(), Arc::clone(&scope), addressed);
        let result = call_scope::within(&scope, self.inner.embed(request))
            .instrument(span.clone())
            .await;
        match &result {
            Ok(response) => {
                if let Some(input) = response.usage.as_ref().and_then(|u| u.input_tokens) {
                    record_int(&span, "gen_ai.usage.input_tokens", input);
                }
                if let Some(model) = &response.model {
                    span.record("gen_ai.response.model", model.as_str());
                }
                record_warnings(&span, &response.warnings);
                call.complete(response.provider.as_deref());
            }
            Err(err) => call.fail(err),
        }
        result
    }

    super::forward_embedding_questions!(inner);
}

impl<T> DecisionProvider for TracingProvider<T>
where
    T: DecisionProvider,
{
    async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse> {
        let addressed = self.inner.provider_name();
        let span = tracing::info_span!(
            "decision",
            "otel.name" = %format_args!("decision {}", request.model),
            "otel.kind" = "client",
            "otel.status_code" = tracing::field::Empty,
            "otel.status_description" = tracing::field::Empty,
            "gen_ai.operation.name" = "decision",
            "gen_ai.provider.name" = otel_genai_provider_name(addressed),
            "gen_ai.request.model" = %request.model,
            "allama.request.provider" = otel_genai_provider_name(addressed),
            "server.address" = tracing::field::Empty,
            "server.port" = tracing::field::Empty,
            "gen_ai.conversation.id" = tracing::field::Empty,
            "allama.decision.question_count" = otel_count(request.questions.len()),
            "allama.response.provider" = tracing::field::Empty,
            "allama.response.request_id" = tracing::field::Empty,
            "gen_ai.response.id" = tracing::field::Empty,
            "gen_ai.usage.input_tokens" = tracing::field::Empty,
            "gen_ai.usage.output_tokens" = tracing::field::Empty,
            "gen_ai.response.model" = tracing::field::Empty,
            "allama.response.warnings" = tracing::field::Empty,
            "allama.attempts" = tracing::field::Empty,
            "allama.retry.total_delay_ms" = tracing::field::Empty,
            "allama.retry.last_delay_ms" = tracing::field::Empty,
            "allama.retry.server_directive" = tracing::field::Empty,
            "allama.retry.budget_denied" = tracing::field::Empty,
            "allama.retry.give_up_reason" = tracing::field::Empty,
            "allama.fallback.used" = tracing::field::Empty,
            "allama.fallback.attempts" = tracing::field::Empty,
            "allama.fallback.error_type" = tracing::field::Empty,
            "allama.response.cancelled" = tracing::field::Empty,
            "error.type" = tracing::field::Empty,
        );
        self.run_span_hook(&span, TracedRequest::Decision(request));
        let scope = Scope::new();
        let call = CallRecord::new(span.clone(), Arc::clone(&scope), addressed);
        let result = call_scope::within(&scope, self.inner.decide(request))
            .instrument(span.clone())
            .await;
        match &result {
            Ok(response) => {
                if let Some(usage) = &response.usage {
                    if let Some(input) = usage.input_tokens {
                        record_int(&span, "gen_ai.usage.input_tokens", input);
                    }
                    if let Some(output) = usage.output_tokens {
                        record_int(&span, "gen_ai.usage.output_tokens", output);
                    }
                }
                if let Some(request_id) = &response.request_id {
                    span.record("allama.response.request_id", request_id.as_str());
                }
                if let Some(id) = &response.id {
                    span.record("gen_ai.response.id", id.as_str());
                }
                if let Some(model) = &response.model {
                    span.record("gen_ai.response.model", model.as_str());
                }
                record_warnings(&span, &response.warnings);
                call.complete(response.provider.as_deref());
            }
            Err(err) => call.fail(err),
        }
        result
    }

    super::forward_decision_questions!(inner);
}

impl<T> ExtractionProvider for TracingProvider<T>
where
    T: ExtractionProvider,
{
    async fn extract_values(&self, request: &ExtractionRequest) -> Result<ExtractionResponse> {
        let addressed = self.inner.provider_name();
        let span = tracing::info_span!(
            "extraction",
            "otel.name" = %format_args!("extraction {}", request.model),
            "otel.kind" = "client",
            "otel.status_code" = tracing::field::Empty,
            "otel.status_description" = tracing::field::Empty,
            "gen_ai.operation.name" = "extraction",
            "gen_ai.provider.name" = otel_genai_provider_name(addressed),
            "gen_ai.request.model" = %request.model,
            "allama.request.provider" = otel_genai_provider_name(addressed),
            "server.address" = tracing::field::Empty,
            "server.port" = tracing::field::Empty,
            "gen_ai.conversation.id" = tracing::field::Empty,
            "allama.extraction.field_count" = otel_count(request.fields.len()),
            "allama.response.provider" = tracing::field::Empty,
            "allama.response.request_id" = tracing::field::Empty,
            "gen_ai.response.id" = tracing::field::Empty,
            "gen_ai.usage.input_tokens" = tracing::field::Empty,
            "gen_ai.usage.output_tokens" = tracing::field::Empty,
            "gen_ai.response.model" = tracing::field::Empty,
            "allama.response.warnings" = tracing::field::Empty,
            "allama.attempts" = tracing::field::Empty,
            "allama.retry.total_delay_ms" = tracing::field::Empty,
            "allama.retry.last_delay_ms" = tracing::field::Empty,
            "allama.retry.server_directive" = tracing::field::Empty,
            "allama.retry.budget_denied" = tracing::field::Empty,
            "allama.retry.give_up_reason" = tracing::field::Empty,
            "allama.fallback.used" = tracing::field::Empty,
            "allama.fallback.attempts" = tracing::field::Empty,
            "allama.fallback.error_type" = tracing::field::Empty,
            "allama.response.cancelled" = tracing::field::Empty,
            "error.type" = tracing::field::Empty,
        );
        self.run_span_hook(&span, TracedRequest::Extraction(request));
        let scope = Scope::new();
        let call = CallRecord::new(span.clone(), Arc::clone(&scope), addressed);
        let result = call_scope::within(&scope, self.inner.extract_values(request))
            .instrument(span.clone())
            .await;
        match &result {
            Ok(response) => {
                if let Some(usage) = &response.usage {
                    if let Some(input) = usage.input_tokens {
                        record_int(&span, "gen_ai.usage.input_tokens", input);
                    }
                    if let Some(output) = usage.output_tokens {
                        record_int(&span, "gen_ai.usage.output_tokens", output);
                    }
                }
                if let Some(request_id) = &response.request_id {
                    span.record("allama.response.request_id", request_id.as_str());
                }
                if let Some(id) = &response.id {
                    span.record("gen_ai.response.id", id.as_str());
                }
                if let Some(model) = &response.model {
                    span.record("gen_ai.response.model", model.as_str());
                }
                record_warnings(&span, &response.warnings);
                call.complete(response.provider.as_deref());
            }
            Err(err) => call.fail(err),
        }
        result
    }

    super::forward_extraction_questions!(inner);
}

impl<T: ProviderIdentity> ProviderIdentity for TracingProvider<T> {
    super::forward_provider_identity!(inner);
}

/// Records the call's outcome (see [`record_outcome`]) on its span exactly
/// once: when the call completes, or when its future is dropped before
/// that, which also marks the call cancelled. A stream call hands the job to
/// its [`TracingStream`].
struct CallRecord<'a> {
    span: Span,
    scope: Arc<Scope>,
    addressed: &'a str,
    armed: bool,
}

impl<'a> CallRecord<'a> {
    fn new(span: Span, scope: Arc<Scope>, addressed: &'a str) -> Self {
        Self {
            span,
            scope,
            addressed,
            armed: true,
        }
    }

    /// The call completed; `reported` is the provider its response names.
    fn complete(mut self, reported: Option<&str>) {
        self.armed = false;
        record_outcome(
            &self.span,
            &self.scope.summary(),
            reported,
            Ending::Completed,
            self.addressed,
        );
    }

    /// The call failed with `err`.
    fn fail(mut self, err: &Error) {
        self.armed = false;
        record_error(&self.span, err);
        let failed_on = err
            .context()
            .and_then(|context| context.provider.as_deref());
        record_outcome(
            &self.span,
            &self.scope.summary(),
            None,
            Ending::Failed(failed_on),
            self.addressed,
        );
    }

    /// The call's stream records the outcome instead.
    fn hand_over(mut self) {
        self.armed = false;
    }
}

impl Drop for CallRecord<'_> {
    fn drop(&mut self) {
        if self.armed {
            // The caller dropped the polled future before the call completed
            // (a client disconnect, a deadline outside this wrapper), also
            // before a stream's first output under retry or fallback.
            record_cancelled(&self.span);
            record_outcome(
                &self.span,
                &self.scope.summary(),
                None,
                Ending::Dropped,
                self.addressed,
            );
        }
    }
}

struct TracingStream<S> {
    inner: Pin<Box<S>>,
    span: Span,
    content_config: TracingContentConfig,
    /// When the call was issued, the start of the time-to-first measures.
    issued_at: Instant,
    /// Collects the response the span records: metadata only, unless output
    /// messages are captured.
    collector: StreamCollector,
    first_chunk_recorded: bool,
    first_output_recorded: bool,
    /// The consumer read to the end of the response: its `ResponseStop`,
    /// the end of the stream, or an error. Dropped before that, the stream
    /// was cancelled.
    ended: bool,
    /// An error was recorded; later ones are not.
    failed: bool,
    /// The provider the recorded error names.
    failed_on: Option<String>,
    finalized: bool,
    /// The call's scope, installed while the stream is polled.
    scope: Arc<Scope>,
    /// The provider the request was addressed to.
    addressed: String,
}

impl<S> TracingStream<S> {
    fn new(
        inner: S,
        span: Span,
        content_config: TracingContentConfig,
        issued_at: Instant,
        scope: Arc<Scope>,
        addressed: String,
    ) -> Self {
        // Without output capture only the response metadata is recorded, so
        // the stream's content is not buffered a second time.
        let collector = if content_config.capture_output_messages {
            StreamCollector::new()
        } else {
            StreamCollector::metadata_only()
        };
        Self {
            inner: Box::pin(inner),
            span,
            content_config,
            issued_at,
            collector,
            first_chunk_recorded: false,
            first_output_recorded: false,
            ended: false,
            failed: false,
            failed_on: None,
            finalized: false,
            scope,
            addressed,
        }
    }

    fn observe(&mut self, event: &StreamEvent) {
        if !self.first_chunk_recorded {
            self.first_chunk_recorded = true;
            self.span.record(
                "gen_ai.response.time_to_first_chunk",
                self.issued_at.elapsed().as_secs_f64(),
            );
        }
        if !self.first_output_recorded && event.is_output() {
            self.first_output_recorded = true;
            self.span.record(
                "allama.response.time_to_first_output",
                self.issued_at.elapsed().as_secs_f64(),
            );
        }
        match event {
            // Each warning is an event when it arrives; the count is
            // recorded once, at the end.
            StreamEvent::Warning { warning } => record_warning_event(&self.span, warning),
            // The response is complete: a consumer that stops reading here
            // has everything, so it did not cancel.
            StreamEvent::ResponseStop => self.ended = true,
            _ => {}
        }
        // Protocol errors only affect what is recorded, never the stream.
        let _ = self.collector.push_ref(event);
    }

    fn finalize_once(&mut self) {
        if self.finalized {
            return;
        }
        self.finalized = true;
        let collector = std::mem::take(&mut self.collector);
        let mut reported = None;
        if let Ok(collected) = collector.finish_partial() {
            let config = TracingContentConfig {
                capture_input_messages: false,
                ..self.content_config
            };
            record_response_fields(&self.span, &collected.response);
            record_warning_count(&self.span, collected.response.warnings.len());
            record_output_messages(&self.span, &collected.response, &config);
            reported = collected.response.provider;
        }
        let ending = if self.failed {
            Ending::Failed(self.failed_on.as_deref())
        } else if self.ended {
            Ending::Completed
        } else {
            Ending::Dropped
        };
        record_outcome(
            &self.span,
            &self.scope.summary(),
            reported.as_deref(),
            ending,
            &self.addressed,
        );
    }
}

impl<S> Drop for TracingStream<S> {
    fn drop(&mut self) {
        // The consumer stopped reading before the response ended (a client
        // disconnect, an application timeout). That is the consumer's
        // choice, not a failure, so it is not recorded as an error.
        if !self.ended {
            record_cancelled(&self.span);
        }
        self.finalize_once();
    }
}

impl<S> Stream for TracingStream<S>
where
    S: Stream<Item = Result<StreamEvent>> + Send,
{
    type Item = Result<StreamEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let span = this.span.clone();
        let _guard = span.enter();
        let poll = {
            let _scope = call_scope::enter(&this.scope);
            this.inner.as_mut().poll_next(cx)
        };
        match &poll {
            Poll::Ready(Some(Ok(event))) => this.observe(event),
            Poll::Ready(Some(Err(err))) => {
                // The span records the first error. A chat stream ends
                // there, but one built by hand may not: polled past it, it
                // records no other.
                if !this.failed {
                    record_error(&this.span, err);
                    this.failed_on = err.context().and_then(|context| context.provider.clone());
                }
                this.failed = true;
                this.ended = true;
            }
            Poll::Ready(None) => {
                this.ended = true;
                this.finalize_once();
            }
            Poll::Pending => {}
        }
        poll
    }
}

fn chat_span(provider_name: &str, request: &ChatRequest, cfg: &TracingContentConfig) -> Span {
    let span = tracing::info_span!(
        "chat",
        "otel.name" = %format_args!("chat {}", request.model),
        "otel.kind" = "client",
        "otel.status_code" = tracing::field::Empty,
        "otel.status_description" = tracing::field::Empty,
        "gen_ai.operation.name" = "chat",
        "gen_ai.provider.name" = otel_genai_provider_name(provider_name),
        "gen_ai.request.model" = %request.model,
        "allama.request.provider" = otel_genai_provider_name(provider_name),
        "server.address" = tracing::field::Empty,
        "server.port" = tracing::field::Empty,
        "gen_ai.conversation.id" = tracing::field::Empty,
        "gen_ai.request.max_tokens" = tracing::field::Empty,
        "gen_ai.request.temperature" = tracing::field::Empty,
        "gen_ai.request.top_p" = tracing::field::Empty,
        "gen_ai.request.top_k" = tracing::field::Empty,
        "gen_ai.request.frequency_penalty" = tracing::field::Empty,
        "gen_ai.request.presence_penalty" = tracing::field::Empty,
        "gen_ai.request.stop_sequences" = tracing::field::Empty,
        "gen_ai.request.seed" = tracing::field::Empty,
        "gen_ai.request.stream" = tracing::field::Empty,
        "gen_ai.output.type" = tracing::field::Empty,
        "gen_ai.usage.input_tokens" = tracing::field::Empty,
        "gen_ai.usage.output_tokens" = tracing::field::Empty,
        "gen_ai.usage.cache_read.input_tokens" = tracing::field::Empty,
        "gen_ai.usage.cache_creation.input_tokens" = tracing::field::Empty,
        "gen_ai.usage.reasoning.output_tokens" = tracing::field::Empty,
        "gen_ai.response.finish_reasons" = tracing::field::Empty,
        "gen_ai.response.id" = tracing::field::Empty,
        "gen_ai.response.model" = tracing::field::Empty,
        "allama.response.provider" = tracing::field::Empty,
        "allama.response.request_id" = tracing::field::Empty,
        "allama.response.warnings" = tracing::field::Empty,
        "allama.attempts" = tracing::field::Empty,
        "allama.retry.total_delay_ms" = tracing::field::Empty,
        "allama.retry.last_delay_ms" = tracing::field::Empty,
        "allama.retry.server_directive" = tracing::field::Empty,
        "allama.retry.budget_denied" = tracing::field::Empty,
        "allama.retry.give_up_reason" = tracing::field::Empty,
        "allama.fallback.used" = tracing::field::Empty,
        "allama.fallback.attempts" = tracing::field::Empty,
        "allama.fallback.error_type" = tracing::field::Empty,
        "gen_ai.response.time_to_first_chunk" = tracing::field::Empty,
        "allama.response.time_to_first_output" = tracing::field::Empty,
        "allama.response.cancelled" = tracing::field::Empty,
        "gen_ai.system_instructions" = tracing::field::Empty,
        "gen_ai.input.messages" = tracing::field::Empty,
        "gen_ai.output.messages" = tracing::field::Empty,
        "error.type" = tracing::field::Empty,
    );

    if let Some(v) = request.max_output_tokens {
        span.record("gen_ai.request.max_tokens", i64::from(v));
    }
    if let Some(v) = request.temperature {
        span.record("gen_ai.request.temperature", v);
    }
    if let Some(v) = request.top_p {
        span.record("gen_ai.request.top_p", v);
    }
    if let Some(v) = request.top_k {
        span.record("gen_ai.request.top_k", f64::from(v));
    }
    if let Some(v) = request.frequency_penalty {
        span.record("gen_ai.request.frequency_penalty", v);
    }
    if let Some(v) = request.presence_penalty {
        span.record("gen_ai.request.presence_penalty", v);
    }
    if !request.stop.is_empty() {
        let stop = &request.stop;
        span.record(
            "gen_ai.request.stop_sequences",
            serde_json::to_string(stop).unwrap_or_default().as_str(),
        );
    }
    if let Some(seed) = request.seed {
        record_int(&span, "gen_ai.request.seed", seed);
    }
    if let Some(format) = &request.response_format {
        span.record(
            "gen_ai.output.type",
            match format {
                ResponseFormat::Json | ResponseFormat::JsonSchema { .. } => "json",
                _ => "text",
            },
        );
    }
    if cfg.capture_input_messages {
        if !request.system.is_empty() {
            let instructions = otel_system_instructions(&request.system);
            if let Some(payload) = serialize_redacted(&instructions, cfg.max_payload_chars) {
                span.record("gen_ai.system_instructions", payload.as_str());
            }
        }
        let messages: Vec<_> = request
            .messages
            .iter()
            .map(|message| otel_message(message, cfg))
            .collect();
        if let Some(payload) = serialize_redacted(&messages, cfg.max_payload_chars) {
            span.record("gen_ai.input.messages", payload.as_str());
        }
    }
    span
}

fn record_response(span: &Span, response: &ChatResponse, cfg: &TracingContentConfig) {
    record_response_fields(span, response);
    record_warnings(span, &response.warnings);
    record_output_messages(span, response, cfg);
}

fn record_response_fields(span: &Span, response: &ChatResponse) {
    if let Some(usage) = &response.usage {
        record_usage(span, usage);
    }
    if let Some(reason) = &response.finish_reason {
        record_finish_reason(span, reason);
    }
    if let Some(id) = &response.id {
        span.record("gen_ai.response.id", id.as_str());
    }
    if let Some(model) = &response.model {
        span.record("gen_ai.response.model", model.as_str());
    }
    if let Some(request_id) = &response.request_id {
        span.record("allama.response.request_id", request_id.as_str());
    }
}

/// Record the outcome of a call on its span, once, when it completes, fails
/// or is dropped: `allama.response.provider` and the call scope's summary.
///
/// The serving provider is the one the response reports (`reported`), else
/// the one the call scope recorded. When nothing served, it is the provider
/// that ended the call: the one the error names, for a failure, or the last
/// fallback side tried, for a drop. Failing those (an error raised by a
/// wrapper, a call dropped outside a fallback), it is the provider the
/// request was addressed to.
fn record_outcome(
    span: &Span,
    summary: &Summary,
    reported: Option<&str>,
    ending: Ending<'_>,
    addressed: &str,
) {
    let ended_by = match ending {
        Ending::Completed => None,
        Ending::Failed(provider) => provider,
        Ending::Dropped => summary.tried.as_deref(),
    };
    let provider = reported
        .or(summary.served_by.as_deref())
        .or(ended_by)
        .unwrap_or(addressed);
    span.record(
        "allama.response.provider",
        otel_genai_provider_name(provider),
    );
    record_summary(span, summary);
}

/// How a call ended, for [`record_outcome`].
#[derive(Clone, Copy)]
enum Ending<'a> {
    /// With a response, or a stream read to its end.
    Completed,
    /// With an error, naming the provider that raised it, if any.
    Failed(Option<&'a str>),
    /// The caller dropped it first.
    Dropped,
}

/// Mark a call the caller abandoned: `allama.response.cancelled = true`.
///
/// Not while the thread panics: a future or stream dropped while unwinding
/// was not given up by choice.
fn record_cancelled(span: &Span) {
    if !std::thread::panicking() {
        span.record("allama.response.cancelled", true);
    }
}

/// Record the resilience summary of a call (see [`TracingProvider`]).
fn record_summary(span: &Span, summary: &Summary) {
    span.record("allama.attempts", otel_count(summary.call_attempts()));
    if let Some((address, port)) = &summary.server {
        span.record("server.address", address.as_ref());
        if let Some(port) = port {
            span.record("server.port", i64::from(*port));
        }
    }
    if let Some(last_delay) = summary.last_delay {
        span.record(
            "allama.retry.total_delay_ms",
            otel_millis(summary.total_delay),
        );
        span.record("allama.retry.last_delay_ms", otel_millis(last_delay));
    }
    if let Some(directive) = summary.server_directive {
        span.record("allama.retry.server_directive", directive);
    }
    if summary.budget_denied {
        span.record("allama.retry.budget_denied", true);
    }
    if let Some(reason) = summary.give_up {
        span.record("allama.retry.give_up_reason", reason.as_str());
    }
    if summary.failed_over {
        span.record("allama.fallback.used", true);
        span.record(
            "allama.fallback.attempts",
            otel_count(summary.failed_providers),
        );
        if let Some(error_type) = summary.failover_error {
            span.record("allama.fallback.error_type", error_type);
        }
    }
}

/// Record a response's warnings: the count and one event each.
fn record_warnings(span: &Span, warnings: &[Warning]) {
    record_warning_count(span, warnings.len());
    for warning in warnings {
        record_warning_event(span, warning);
    }
}

fn record_warning_count(span: &Span, count: usize) {
    if count > 0 {
        span.record("allama.response.warnings", otel_count(count));
    }
}

/// One event for a warning, unless it was emitted as a warning event
/// already (feature `warning-events`): that event is the span's.
fn record_warning_event(span: &Span, warning: &Warning) {
    if crate::warning::was_emitted(warning) {
        return;
    }
    tracing::event!(
        parent: span,
        tracing::Level::WARN,
        allama.warning.kind = warning.kind.as_str(),
        allama.warning.provider = warning.provider.as_deref().unwrap_or(""),
        "{}",
        warning.message
    );
}

fn record_output_messages(span: &Span, response: &ChatResponse, cfg: &TracingContentConfig) {
    if !cfg.capture_output_messages {
        return;
    }
    let mut message = serde_json::Map::new();
    message.insert("role".into(), "assistant".into());
    message.insert(
        "parts".into(),
        response
            .content
            .iter()
            .map(|block| otel_block(block, cfg))
            .collect(),
    );
    if let Some(reason) = &response.finish_reason {
        message.insert("finish_reason".into(), reason.as_str().into());
    }
    let output = serde_json::Value::Array(vec![serde_json::Value::Object(message)]);
    if let Some(payload) = serialize_redacted(&output, cfg.max_payload_chars) {
        span.record("gen_ai.output.messages", payload.as_str());
    }
}

fn otel_system_instructions(system: &[SystemPrompt]) -> serde_json::Value {
    system.iter().map(|p| text_part(&p.content)).collect()
}

fn otel_message(message: &Message, cfg: &TracingContentConfig) -> serde_json::Value {
    let (role, parts) = match message {
        Message::User(m) => (
            "user",
            match &m.content {
                UserContent::Text(text) => vec![text_part(text)],
                UserContent::Parts(parts) => {
                    parts.iter().map(|part| otel_part(part, cfg)).collect()
                }
            },
        ),
        Message::Assistant(m) => (
            "assistant",
            m.content
                .iter()
                .map(|block| otel_block(block, cfg))
                .collect(),
        ),
        Message::Tool(m) => {
            let mut part = serde_json::Map::new();
            part.insert("type".into(), "tool_call_response".into());
            part.insert("id".into(), m.tool_call_id.clone().into());
            part.insert("allama.tool_name".into(), m.name.clone().into());
            part.insert(
                "response".into(),
                match &m.content {
                    ToolResultContent::Text(text) => text.clone().into(),
                    ToolResultContent::Object(object) => serde_json::Value::Object(object.clone()),
                    ToolResultContent::Parts(parts) => {
                        parts.iter().map(|part| otel_part(part, cfg)).collect()
                    }
                },
            );
            if m.is_error {
                part.insert("is_error".into(), true.into());
            }
            ("tool", vec![serde_json::Value::Object(part)])
        }
    };
    let mut map = serde_json::Map::new();
    map.insert("role".into(), role.into());
    map.insert("parts".into(), serde_json::Value::Array(parts));
    serde_json::Value::Object(map)
}

fn otel_part(part: &ContentPart, cfg: &TracingContentConfig) -> serde_json::Value {
    match part {
        ContentPart::Text(text) => text_part(&text.text),
        ContentPart::Image(image) => {
            let mut value = media_part("image", &image.source, cfg);
            if let (Some(detail), serde_json::Value::Object(map)) = (image.detail, &mut value) {
                map.insert("detail".into(), detail.as_str().into());
            }
            value
        }
        ContentPart::File(file) => media_part("file", &file.source, cfg),
        ContentPart::Other(other) => generic_part(&other.type_name, &other.data),
    }
}

fn otel_block(block: &ContentBlock, cfg: &TracingContentConfig) -> serde_json::Value {
    match block {
        ContentBlock::Text(text) => text_part(&text.text),
        ContentBlock::Image(image) => media_part("image", &image.source, cfg),
        ContentBlock::ToolCall(call) => serde_json::json!({
            "type": "tool_call",
            "id": call.id,
            "name": call.name,
            "arguments": serde_json::from_str::<serde_json::Value>(&call.arguments)
                .unwrap_or_else(|_| call.arguments.clone().into()),
        }),
        ContentBlock::Reasoning(reasoning) => {
            serde_json::json!({"type": "reasoning", "content": reasoning.text})
        }
        ContentBlock::Refusal(refusal) => {
            serde_json::json!({"type": "refusal", "content": refusal.text})
        }
        ContentBlock::ProviderToolCall(call) => serde_json::json!({
            "type": "server_tool_call",
            "id": call.id,
            "name": call.name,
            "arguments": serde_json::from_str::<serde_json::Value>(&call.arguments)
                .unwrap_or_else(|_| call.arguments.clone().into()),
        }),
        ContentBlock::ProviderToolResult(result) => serde_json::json!({
            "type": "server_tool_call_response",
            "id": result.call_id,
            "response": result.output,
        }),
        ContentBlock::Other(other) => generic_part(&other.type_name, &other.data),
    }
}

fn text_part(text: &str) -> serde_json::Value {
    serde_json::json!({"type": "text", "content": text})
}

/// A media part: a `uri` part for a URL, a `blob` part for inline data.
/// Without [`capture_media_data`](TracingContentConfig::capture_media_data),
/// a blob part leaves out its `content` and records the decoded size in
/// `allama.omitted_content_bytes` instead.
fn media_part(
    modality: &str,
    source: &MediaSource,
    cfg: &TracingContentConfig,
) -> serde_json::Value {
    match source {
        MediaSource::Url { url } if cfg.capture_media_urls => {
            serde_json::json!({"type": "uri", "modality": modality, "uri": url})
        }
        MediaSource::Url { url } => {
            serde_json::json!({"type": "uri", "modality": modality, "uri": without_query(url)})
        }
        MediaSource::Base64 {
            media_type, data, ..
        } if cfg.capture_media_data => serde_json::json!({
            "type": "blob", "modality": modality, "mime_type": media_type, "content": data,
        }),
        MediaSource::Base64 {
            media_type, data, ..
        } => serde_json::json!({
            "type": "blob",
            "modality": modality,
            "mime_type": media_type,
            "allama.omitted_content_bytes": base64_decoded_len(data),
        }),
        MediaSource::ProviderFile { origin, id, .. } => serde_json::json!({
            "type": "file", "modality": modality, "file_id": id, "allama.origin": origin,
        }),
    }
}

/// `url` without its query and fragment, where presigned URLs carry their
/// credentials.
fn without_query(url: &str) -> &str {
    url.find(['?', '#']).map_or(url, |end| &url[..end])
}

/// The number of bytes `data`, base64, decodes to: three for every four
/// digits, padding and whitespace aside. `data` is not checked.
fn base64_decoded_len(data: &str) -> usize {
    let digits = data
        .bytes()
        .filter(|byte| *byte != b'=' && !byte.is_ascii_whitespace())
        .count();
    digits / 4 * 3 + (digits % 4 * 3) / 4
}

fn generic_part(type_name: &str, data: &crate::ExtraMap) -> serde_json::Value {
    let mut map = data.clone();
    map.insert("type".into(), type_name.into());
    serde_json::Value::Object(map)
}

fn serialize_redacted<T: Serialize>(value: &T, max_chars: usize) -> Option<String> {
    let mut json = serde_json::to_value(value).ok()?;
    redact(&mut json);
    let text = serde_json::to_string(&json).ok()?;
    if text.chars().count() > max_chars {
        let truncated: String = text.chars().take(max_chars).collect();
        return Some(format!("{truncated}...[TRUNCATED]"));
    }
    Some(text)
}

fn redact(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, val) in map.iter_mut() {
                if is_sensitive_key(key) {
                    *val = "[REDACTED]".into();
                } else {
                    redact(val);
                }
            }
        }
        serde_json::Value::Array(items) => items.iter_mut().for_each(redact),
        _ => {}
    }
}

/// Whether a key in a captured payload names a secret (see
/// [Redaction](TracingContentConfig#redaction)): lowercased, with `-` and
/// `_` removed, it is one of the exact names or contains a fragment.
fn is_sensitive_key(key: &str) -> bool {
    const EXACT: [&str; 8] = [
        "accesstoken",
        "apikey",
        "authorization",
        "clientsecret",
        "password",
        "refreshtoken",
        "secret",
        "token",
    ];
    const FRAGMENTS: [&str; 3] = ["apikey", "secret", "token"];
    let key: String = key
        .chars()
        .filter(|c| !matches!(c, '-' | '_'))
        .map(|c| c.to_ascii_lowercase())
        .collect();
    EXACT.contains(&key.as_str()) || FRAGMENTS.iter().any(|fragment| key.contains(fragment))
}

/// Record a failed call: `error.type` and span status `Error`.
fn record_error(span: &Span, err: &Error) {
    span.record("error.type", err.kind_name());
    // The exporting layer sets the status from either field, and the code
    // alone sets an empty description, so the description comes last.
    span.record("otel.status_code", "ERROR");
    span.record("otel.status_description", status_description(err).as_str());
}

/// The span status description of a failed call: the redacted
/// [`Error::as_log`] view without its leading `kind=…`, which `error.type`
/// already records.
fn status_description(err: &Error) -> String {
    let log = err.as_log().to_string();
    let kind = format!("kind={}", err.kind_name());
    match log.strip_prefix(kind.as_str()) {
        Some(rest) => rest.trim_start().to_owned(),
        None => log,
    }
}

fn record_usage(span: &Span, usage: &Usage) {
    if let Some(v) = usage.input_tokens {
        record_int(span, "gen_ai.usage.input_tokens", v);
    }
    if let Some(v) = usage.output_tokens {
        record_int(span, "gen_ai.usage.output_tokens", v);
    }
    if let Some(v) = usage.cache_read_input_tokens {
        record_int(span, "gen_ai.usage.cache_read.input_tokens", v);
    }
    if let Some(v) = usage.cache_write_input_tokens {
        record_int(span, "gen_ai.usage.cache_creation.input_tokens", v);
    }
    if let Some(v) = usage.reasoning_tokens {
        record_int(span, "gen_ai.usage.reasoning.output_tokens", v);
    }
}

/// Record an integer attribute. OpenTelemetry integers are `i64`, and
/// `tracing-opentelemetry` exports `u64` field values as strings, so the
/// value is recorded as `i64`; one that does not fit (a large seed) is
/// recorded as its decimal string rather than altered.
fn record_int(span: &Span, field: &str, value: u64) {
    match i64::try_from(value) {
        Ok(value) => span.record(field, value),
        Err(_) => span.record(field, value.to_string().as_str()),
    };
}

/// A count as an OpenTelemetry integer (see [`record_int`]).
fn otel_count(count: usize) -> i64 {
    i64::try_from(count).unwrap_or(i64::MAX)
}

/// A duration in whole milliseconds, as an OpenTelemetry integer.
fn otel_millis(duration: std::time::Duration) -> i64 {
    i64::try_from(duration.as_millis()).unwrap_or(i64::MAX)
}

fn record_finish_reason(span: &Span, reason: &FinishReason) {
    let reasons = serde_json::to_string(&[reason.as_str()]).unwrap_or_default();
    span.record("gen_ai.response.finish_reasons", reasons.as_str());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChatStreamExt;
    use crate::mock::{ChatResponseBuilder, MockProvider};

    #[tokio::test]
    async fn passes_responses_through() {
        let provider = TracingProvider::new(MockProvider::with_text("hi"))
            .with_content_capture(TracingContentConfig::all());
        let response = provider
            .chat(&ChatRequest::new("m").user("x"))
            .await
            .unwrap();
        assert_eq!(response.text().as_deref(), Some("hi"));
    }

    #[tokio::test]
    async fn streams_pass_through() {
        let provider = TracingProvider::new(MockProvider::with_response(
            ChatResponseBuilder::new().text("streamed").build(),
        ));
        let response = provider
            .chat_stream(&ChatRequest::new("m").user("x"))
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        assert_eq!(response.text().as_deref(), Some("streamed"));
    }

    /// The content a [`TracingStream`] has buffered after passing on every
    /// event of a text response (before the end, which hands the collector
    /// to the span).
    async fn buffered_content(config: TracingContentConfig) -> Vec<ContentBlock> {
        let response = ChatResponseBuilder::new()
            .text("streamed text")
            .usage(3, 2)
            .model("m-1")
            .build();
        let event_count = response.clone().into_stream_events().count();
        let mut stream = TracingStream::new(
            response.into_stream(),
            Span::none(),
            config,
            Instant::now(),
            Scope::new(),
            "mock".to_owned(),
        );
        for _ in 0..event_count {
            let event = futures_util::StreamExt::next(&mut stream).await;
            event.unwrap().unwrap();
        }
        let collected = std::mem::take(&mut stream.collector)
            .finish_partial()
            .unwrap();
        assert_eq!(
            collected.response.usage,
            Some(Usage::new().input_tokens(3).output_tokens(2))
        );
        assert_eq!(collected.response.model.as_deref(), Some("m-1"));
        collected.response.content
    }

    #[tokio::test]
    async fn streams_buffer_content_only_when_output_is_captured() {
        assert!(
            buffered_content(TracingContentConfig::default())
                .await
                .is_empty()
        );
        let captured = buffered_content(TracingContentConfig::all()).await;
        assert_eq!(captured.len(), 1);
    }

    #[test]
    fn redaction_and_truncation() {
        let value = serde_json::json!({"api_key": "secret", "nested": [{"token": "t"}], "text": "x".repeat(20)});
        let out = serialize_redacted(&value, 30).unwrap();
        assert!(out.contains("[REDACTED]"));
        assert!(out.ends_with("[TRUNCATED]"));
    }

    #[test]
    fn sensitive_keys_match_in_any_spelling() {
        for key in [
            "api_key",
            "apiKey",
            "API-Key",
            "x-api-key",
            "Authorization",
            "password",
            "Password",
            "secret",
            "client_secret",
            "clientSecret",
            "token",
            "access_token",
            "refreshToken",
            "bearer_token",
            "x-gateway-token",
            "aws_secret_access_key",
        ] {
            assert!(is_sensitive_key(key), "{key}");
        }
        for key in ["author", "keywords", "city", "content", "passcode", "auth"] {
            assert!(!is_sensitive_key(key), "{key}");
        }
    }

    #[test]
    fn redaction_matches_nested_keys_in_any_spelling() {
        let value = serde_json::json!({
            "apiKey": "k1",
            "headers": {"X-Api-Key": "k2", "Authorization": "Bearer k3"},
            "accounts": [{"clientSecret": "k4", "refresh-token": "k5"}],
            "city": "Lisbon",
        });
        let out = serialize_redacted(&value, 4096).unwrap();
        for secret in ["k1", "k2", "k3", "k4", "k5"] {
            assert!(!out.contains(secret), "{secret} leaked: {out}");
        }
        assert!(out.contains("Lisbon"), "{out}");
    }

    #[test]
    fn text_is_recorded_verbatim() {
        // Redaction matches keys, not text: a secret pasted into a message
        // is captured as written.
        let request = ChatRequest::new("m")
            .system("token: s1")
            .user("my api_key is s2");
        let config = TracingContentConfig::all();
        let messages: Vec<_> = request
            .messages
            .iter()
            .map(|message| otel_message(message, &config))
            .collect();
        let out = serialize_redacted(&messages, 4096).unwrap();
        assert!(out.contains("my api_key is s2"), "{out}");
        let instructions = otel_system_instructions(&request.system);
        let out = serialize_redacted(&instructions, 4096).unwrap();
        assert!(out.contains("token: s1"), "{out}");
    }

    /// A request with an inline PNG of 3,000 bytes between two texts.
    fn request_with_inline_image() -> ChatRequest {
        let data = "A".repeat(4_000);
        ChatRequest::new("m").message(Message::user_parts(vec![
            ContentPart::text("What is in this picture?"),
            ContentPart::image_base64("image/png", data),
            ContentPart::text("Answer in one word."),
        ]))
    }

    fn input_messages(request: &ChatRequest, config: &TracingContentConfig) -> String {
        let messages: Vec<_> = request
            .messages
            .iter()
            .map(|message| otel_message(message, config))
            .collect();
        serialize_redacted(&messages, config.max_payload_chars).unwrap()
    }

    #[test]
    fn inline_media_data_is_left_out_by_default() {
        let out = input_messages(&request_with_inline_image(), &TracingContentConfig::all());
        assert!(!out.contains("AAAA"), "{out}");
        assert!(
            out.contains(
                r#"{"allama.omitted_content_bytes":3000,"mime_type":"image/png","modality":"image","type":"blob"}"#
            ),
            "{out}"
        );
        // The text after the image is not cut off.
        assert!(out.contains("Answer in one word."), "{out}");
        assert!(!out.ends_with("[TRUNCATED]"), "{out}");
    }

    #[test]
    fn inline_media_data_is_captured_when_asked() {
        let mut config = TracingContentConfig::all();
        config.capture_media_data = true;
        config.max_payload_chars = 8_192;
        let out = input_messages(&request_with_inline_image(), &config);
        assert!(out.contains(&"A".repeat(4_000)), "{out}");
        assert!(!out.contains("allama.omitted_content_bytes"), "{out}");
    }

    #[test]
    fn media_by_url_keeps_its_uri() {
        let request =
            ChatRequest::new("m").message(Message::user_parts(vec![ContentPart::image_url(
                "https://example.com/cat.png",
            )]));
        let out = input_messages(&request, &TracingContentConfig::all());
        assert!(out.contains("https://example.com/cat.png"), "{out}");
    }

    /// A presigned URL's credentials stay out of
    /// the trace unless asked for.
    #[test]
    fn media_urls_lose_their_query_unless_asked() {
        let presigned =
            "https://bucket.example/cat.png?X-Amz-Signature=SECRET&X-Amz-Expires=60#frag";
        let request = ChatRequest::new("m")
            .message(Message::user_parts(vec![ContentPart::image_url(presigned)]));
        let out = input_messages(&request, &TracingContentConfig::all());
        assert!(
            out.contains(r#""uri":"https://bucket.example/cat.png""#),
            "{out}"
        );
        assert!(!out.contains("SECRET") && !out.contains("frag"), "{out}");

        let mut config = TracingContentConfig::all();
        config.capture_media_urls = true;
        let out = input_messages(&request, &config);
        assert!(out.contains("X-Amz-Signature=SECRET"), "{out}");
    }

    #[test]
    fn output_media_data_is_left_out_by_default() {
        let block = ContentBlock::Image(crate::Image::base64("image/png", "aGVsbG8="));
        let part = otel_block(&block, &TracingContentConfig::all());
        assert_eq!(part["allama.omitted_content_bytes"], 5);
        assert!(part.get("content").is_none(), "{part}");
    }

    #[test]
    fn base64_decoded_len_counts_bytes() {
        assert_eq!(base64_decoded_len(""), 0);
        assert_eq!(base64_decoded_len("aGk="), 2);
        assert_eq!(base64_decoded_len("aGVsbG8="), 5);
        assert_eq!(base64_decoded_len("aGVs\nbG8h"), 6);
    }

    #[test]
    fn status_description_is_the_redacted_log_without_the_kind() {
        let err = Error::rate_limited("HTTP 429")
            .authored()
            .with_provider("openai")
            .with_status(429)
            .with_upstream_message("SENTINEL-UPSTREAM")
            .with_body("SENTINEL-BODY");
        assert_eq!(
            status_description(&err),
            r#"provider=openai status=429 body_len=13 message="HTTP 429""#
        );

        // A failed failover is described as its last error, with the
        // earlier ones nested.
        let fallback = Error::overloaded("busy").with_fallback_from(Error::timeout("slow"));
        assert_eq!(
            status_description(&fallback),
            "fallback_from=[kind=timeout]"
        );
    }

    #[test]
    fn provider_name_mapping() {
        for (name, expected) in [
            ("azure", "azure.ai.openai"),
            ("gemini", "gcp.gemini"),
            ("google-ai", "gcp.gemini"),
            ("google-vertex", "gcp.vertex_ai"),
            ("mistral", "mistral_ai"),
            ("xai", "x_ai"),
            ("openai", "openai"),
            ("custom-gateway", "custom-gateway"),
        ] {
            assert_eq!(otel_genai_provider_name(name), expected, "{name}");
        }
    }
}
