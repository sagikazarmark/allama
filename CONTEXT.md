# allama

Provider-agnostic language for low-level LLM operations and their normalized behavior.

## Language

### Chat, messages, streaming and tools

**Content block**:
One piece of a model's output in an assistant message or response (`ContentBlock`): text, an image, a tool call, reasoning, a refusal, a provider tool call or result, or other content. Blocks keep the order the model produced them in, and a stream builds each one between its `BlockStart` and `BlockStop`.
_Avoid_: Content part (for output), message part, chunk

**Content part**:
One piece of input in a user message or a tool result (`ContentPart`): text, an image, a file, or other content. Parts are what the caller sends; blocks are what the model sends back, and an assistant message replayed as input keeps its blocks.
_Avoid_: Content block (for input), attachment (for a file part)

**Tool call**:
A model's request that the caller run one of the tools it declared (`ContentBlock::ToolCall`, `ToolCall { id, name, arguments }`): the caller runs it and answers with a tool message that names the call's id. The id is the one the wire carried, or one the adapter made up, unique in the conversation, when the wire carries none. `ChatResponse::tool_calls()` returns only these, so a loop that runs them never sees a provider tool call. Arguments the model wrote as invalid JSON stay a tool call with the raw text, on `chat` and `chat_stream` alike.
_Avoid_: Function call (the OpenAI wire name), tool use (Anthropic's), client tool (for the call)

**Provider tool call** and **provider tool result**:
A tool the provider ran itself during the call, such as a web search, code execution or a remote MCP call (`ContentBlock::ProviderToolCall` and `ProviderToolResult`, linked by the call's id). What is portable: that it ran, its name and input, which result belongs to which call, and whether it failed; the result's output keeps the provider's shape. Both carry their origin and replay only to it (ADR 0002). The tools are enabled through provider options, never tool definitions, and their billable requests are counted in `Usage::tool_requests`. See ADR 0008.
_Avoid_: Server tool (Anthropic's name), hosted tool (OpenAI's), built-in tool (Gemini's), grounding (for a search result)

<a id="reasoning"></a>**Reasoning**:
What a model writes while working out its answer, before or between the parts of it (`ContentBlock::Reasoning`). Its text may be the full reasoning, a summary, or empty; the provider's state for replaying it (a signature, encrypted content) is opaque reasoning kept in provider data under the provider's namespace, and a block with no text, only that state, is redacted reasoning. It is never answer text: `text()` leaves it out, and `reasoning_text()` reads it. Reasoning tokens count as output tokens and against the output cap. Reasoning replays only to the provider that wrote it.
_Avoid_: Thinking (Anthropic's and Gemini's name), thoughts, chain of thought

**Reasoning effort**:
How hard the model should reason (`ReasoningEffort`, ordered `None < Minimal < Low < Medium < High < XHigh`). An adapter sends the nearest level the provider and model take (`ReasoningEffort::nearest`), with an `Approximated` warning when it differs. Anthropic refuses `None` on models that always think; see ADR 0004. Token budgets are not portable: they are provider options (Anthropic `budget_tokens`, Gemini `thinking_budget`). See ADR 0004.
_Avoid_: Thinking budget (a provider option), thinking level

**Reasoning output**:
How much of the reasoning the response should show (`ReasoningOutput`: `Omitted`, `Summary` or `Full`). It sets visibility only: an output setting without an effort never turns reasoning on, and where the provider has no such setting it is warned about.
_Avoid_: Include thoughts, reasoning summary (one of its values)

<a id="citation"></a>**Citation**:
A link from a stretch of generated text to the source that supports it (`Citation { source, span, cited_text }` on `Text::citations`, streamed as `StreamEvent::Citation`): a web page, a document from the request, or a file the provider stores. The span is a byte range into the block's text, or none for the whole block. Citations are output only: an adapter replays its own raw annotations, never the portable list. See ADR 0007.
_Avoid_: Source block, grounding (Gemini's name for the metadata), annotation (OpenAI's wire name)

<a id="finish-reason"></a>**Finish reason**:
Why the model stopped writing (`FinishReason`): it ended its turn (`Stop`), waits for tool results (`ToolCalls`), reached the output cap (`Length`) or the context window (`ContextWindow`), was blocked by a filter (`ContentFilter`), declined (`Refusal`), or stopped for a reason allama does not know (`Other`). The provider's own value is kept in `finish_reason_raw`. Only `Stop` and `ToolCalls` mean finished normally. One wire value means the same finish on every adapter.
_Avoid_: Stop reason (Anthropic's field), finish status

<a id="prompt-cache"></a>**Prompt cache**:
A provider's store of a processed prompt prefix, which later requests that start with the same prefix read instead of processing again, for less latency and cost. A *cache breakpoint* (`CacheBreakpoint`, on a system prompt, message or tool definition) marks the end of a prefix to cache, for providers with explicit markers (Anthropic); providers that cache prefixes automatically (OpenAI, Gemini) treat breakpoints as already met. The request-level `PromptCache` carries a *cache key*, which groups requests that share a prefix (a session id) so the provider routes them together, and a *retention*, the preferred lifetime of an entry (`CacheRetention::Short` or `Long`). A key only groups requests: automatic caching that a provider makes opt-in and bills (Anthropic's request-level `cache_control`) needs explicit intent, never a key alone. Cache reads and writes count in `input_tokens`.
_Avoid_: Context caching (Gemini's name for its explicit cache resource), cache control (Anthropic's wire name)

**Tool definition**:
What a chat request tells the model about a tool it may call: a name, a description, and a JSON Schema for the arguments. It declares the tool and does not run it; the caller runs tool calls. Tools the provider runs itself are configured through provider options, not tool definitions.
_Avoid_: Tool (for the declaration alone), function, tool spec

**Strict schema**:
A tool or response-format JSON Schema the caller marks `strict`, asking the provider to constrain output to it with a grammar. A response format built with `ResponseFormat::json_schema` is strict unless relaxed with `non_strict`, which means constraining is not required (the provider may still constrain). Each grammar engine accepts its own subset of JSON Schema, its schema dialect (for example OpenAI strict mode), and the adapter rewrites the schema into that dialect just before sending. A schema the dialect cannot express (a map, for example) is not strict-representable: a strict tool then goes out non-strict with an `Unsupported` warning, or fails under `UnsupportedPolicy::Error`; a field with no unconstrained form (Anthropic's `output_config.format`) fails under every policy, and a malformed schema always fails. That decision is the same for every grammar and lives in core (`allama::adapter::schema::apply_strict_policy`, with `StrictSchemaError`); the adapter passes in its dialect, which reports its approximations only when it succeeds. The same holds when the endpoint or model has no strict mode at all; strict is never dropped silently.
_Avoid_: Strictified schema, strict mode schema (for the caller's schema)

**Schema constraint**:
Whether an endpoint constrains decoding to a JSON Schema in one position, the response format or a tool's parameters: only when the schema is sent strict (OpenAI's strict mode), always (Ollama's `format`, vLLM's structured outputs, which take no `strict` flag), or never (`allama_openai_compat::SchemaConstraint`). It is separate from whether a forced tool choice constrains the forced call's arguments (vLLM's named and `required` choice do; OpenAI forces only the call), and from whether the endpoint is known to enforce its schema dialect as written: a custom OpenAI-compatible endpoint's strict schemas are rewritten into OpenAI's dialect but reported approximated, so extraction never claims exact enforcement for an endpoint nobody verified. A preset sets each from its provider's documentation.
_Avoid_: Supports strict (for the three facts as one), JSON mode

**Strict null**:
A `null` the model sends for an optional property because a strict dialect that requires every property made it nullable. It stands for "left out", not for a null value. Null normalization (`allama::adapter::schema::normalize_strict_nulls`) removes a `null` only where the caller's original schema makes it a stand-in: the key is in `properties`, in no `required`, and its schema admits no `null`, judged across the schemas that apply (`$ref`, `allOf`, the matching `anyOf`/`oneOf` variant). serde then applies the field's default. A required property's `null` and a map entry's `null` are values, kept for serde to accept or reject. Extraction and `ToolCall::parse_arguments_with` apply it.
_Avoid_: Null stripping (it keeps the nulls the schema admits)

<a id="commit-point"></a>**Commit point**:
The first event of a stream that carries model output (`StreamEvent::is_output`: text, reasoning, a refusal, a tool call, an image), after which the caller may have seen the answer. Before it nothing is exposed, so an error can still be retried, failed over, or answered with an HTTP status; after it an error can only end the stream, and retry and fallback pass it through. `RetryingProvider` and `FallbackProvider` hold a stream back until its commit point, and `allama::chat::stream::until_committed` gives an application the same wait, replaying the held events so nothing is lost. A stream that ends without output commits at its end.
_Avoid_: First token (metadata events come before it), committed stream

<a id="timeouts"></a>**Timeouts**:
The bounds on a call, by phase: the request bound (a whole unary call: `chat`, `embed`, `decide`), the setup bound (until a stream exists), the idle bound (each wait for the next stream event) and the total bound (a whole stream, which also cuts off a long answer). Two layers apply them. The provider builders' transport timeouts catch dead connections: `timeout` (the request timeout) and `stream_timeouts` (setup and idle). `TimeoutProvider` applies workload-specific ones portably, with the `TimeoutPolicy` methods `request`, `stream_setup`, `stream_idle` and `stream_total`; an unset setup bound takes the request bound, and where it sits decides what they bound (attempt timeout vs deadline). An expiry is `Error::Timeout`, retryable and eligible for fallback. Each layer's default and the worst case they leave: "Worst-case latency" in the `allama_providers::ProviderSettings` rustdoc.
_Avoid_: Read timeout (a client-wide socket setting)

<a id="stream-timeouts"></a>**Stream timeouts**:
The bounds on a streaming call: the setup bound (until the response starts: headers for an HTTP transport), the idle bound (the longest wait for the next chunk or event, the first one included; only time the consumer spends waiting counts, not time before its first poll or spent handling the previous item), and the total bound (the whole stream, measured from the call, which also cuts off a stream that keeps producing events). HTTP provider crates apply generous setup and idle defaults to catch dead connections (`allama_http::StreamTimeouts`) and no total bound; `TimeoutProvider` applies workload-specific ones to decoded events, the total bound included (`TimeoutPolicy::stream_total`, unset by default).
_Avoid_: Read timeout (a client-wide socket setting), stream deadline

<a id="request-timeout"></a>**Request timeout**:
The bound on a whole non-streaming call, set with a provider builder's `timeout(Option<Duration>)`. One rule on every HTTP provider builder: left unset, it is `allama_http::DEFAULT_REQUEST_TIMEOUT` (120 s) whatever the transport, a `ReqwestTransport` over a client of your own included; only `timeout(None)` removes it, and once set it replaces a client's own total timeout. Streams have stream timeouts instead.
_Avoid_: Read timeout, client timeout (the reqwest client's own setting)

<a id="attempt-timeout-vs-deadline"></a>**Attempt timeout vs deadline**:
Where a `TimeoutProvider` sits decides what its policy bounds. An attempt timeout sits inside the retries (`ProviderSettings::attempt_timeout`, or `RetryingProvider::new(TimeoutProvider::new(p, policy), retry_policy)`) and bounds one attempt: a stalled attempt fails with `Timeout` and is retried, and a `stream_total` there restarts with every attempt. A deadline sits outside retries and fallback and bounds the whole call, across every attempt, backoff and failover. Since retry and fallback hand a stream over only at its commit point, a deadline's setup bound is the wait for the first output: an interactive server sets `request`, which bounds unary calls and that wait alike, never `stream_total`, which would cut off a long answer after it started. Keep a deadline inside `TracingProvider`, so its expiry is recorded as a timeout, not a cancelled call.
_Avoid_: Global timeout, overall timeout

<a id="finished-normally"></a>**Finished normally**:
A response whose model ended its turn (`FinishReason::Stop`) or waits for tool results (`ToolCalls`), as `ChatResponse::finished_normally` reports. Every other finish is abnormal: cut off at the output token limit (`Length`) or at the model's context window (`ContextWindow`, where the remedy is a shorter input, as for `Error::ContextLengthExceeded`; OpenAI-compatible servers report both as `Length`), stopped by a content filter (`ContentFilter`), declined (`Refusal`), or ended for a reason allama does not know (`Other`); so is a response that reports no finish reason. An abnormal finish is still `Ok`, so one call tells a complete answer from the rest. A stream that ends before its terminal marker is not a finish but an `Incomplete` stream error; one that ends with its marker but no finish reason is `Ok`, with none and a warning (`adapter::chat::MISSING_FINISH_REASON`). With `warning-events`, `Length`, `ContextWindow`, `ContentFilter` and `Refusal` finishes are also `INFO` events under `allama::finish`, reported once per call by the innermost layer built on `prepare`.
_Avoid_: Successful response, complete response (for the check), stop reason (Anthropic's field name)

**Provider file**:
A file a provider stores, referenced in a request by the id it issued (`MediaSource::ProviderFile { origin, id }`): an OpenAI or Anthropic Files API `file_id`, a Mistral file id. The id means something only to the provider that issued it, so the source names that provider's namespace as its origin, and a provider that does not read the origin refuses the request with `Unsupported` before sending (ADR 0002). A file URI that is a URL (a Gemini Files API URI) is a `Url` source instead. Not to be confused with `CitationSource::File`, which is a file the model cited.
_Avoid_: file reference, uploaded file

**File kind**:
What a `File` part holds, from its media type (`FileKind`): a document (PDF, plain text, any type that is not audio or video), audio (`audio/*`) or video (`video/*`). Each kind has its own capability: `FileInput` means documents, `AudioInput` audio and `VideoInput` video, so a provider that takes PDFs but no audio says so.

**Other content**:
Content the portable model does not represent (`OtherContent`), kept so a conversation loses nothing: a wire type name, the wire fields verbatim in `data` (any key is allowed there), the origin namespace of the provider that produced it, and namespaced provider data for state an adapter keeps about it, such as a signature. Adapters replay it only to its origin and replay its provider data only from their own namespace; another provider's other content is dropped with a `Dropped` warning under every policy, in input parts as in assistant blocks. A part with no origin is the caller's own raw part and is sent verbatim. Endpoints that share a namespace (OpenAI Chat Completions and Responses) replay it only for the types each one knows. It serializes under its own reserved kind, as `{"type": "other", "type_name", "origin", "data", "provider_data"}`, so a wire type name never collides with one of allama's kinds, today's or a later one; only that form is read, and a `type` this version does not know fails to load.
_Avoid_: Unknown content, raw block

**Vendor frame**:
Stream data in a provider's wire format that the adapter does not model: a frame of a type the API does not define (a named SSE event, `event: search_results`; an NDJSON line of an unknown `type`), or an unmodeled top-level field of a chunk (`citations`). A *frame* is the unit of a stream's wire framing: an SSE event, an NDJSON line. An adapter never decodes a vendor frame as its own and never drops its data: it surfaces it verbatim as response-level provider data under its namespace (`StreamEvent::ResponseProviderData`), and warns once per unknown frame type, as it arrives: `Prepared::finish_stream` holds the ones read before `ResponseStart` until it (`allama::adapter::chat::start_first`). SSE comments and `retry:` lines, and blank NDJSON lines, are framing, not vendor frames. `allama_http::framing::VendorFrames` surfaces vendor frames this way. Checked by `allama_conformance::stream_frames::assert_unknown_stream_frames_surfaced` (over SSE with `StreamFraming::sse()`), which injects a vendor frame in the stream's own framing and also checks that reframing a stream (`CRLF` line endings, multi-line `data:`) changes nothing.
_Avoid_: Unknown chunk, extra event

<a id="refusal"></a>**Refusal**:
Text in which the model declines to answer, written in place of the answer (`ContentBlock::Refusal`, streamed as `StreamEvent::RefusalDelta`). It is never answer text: `text()` leaves it out, `ChatResponse::refusal_text()` reads it (and `is_refusal()` is true for it or a `Refusal` finish), and a streamed refusal is a commit point, so it is never retried or failed over. Only providers that mark refusal text produce one (OpenAI's `refusal` fields). A *provider decline* is different: the provider stops without writing a refusal (Anthropic's `refusal` stop reason, Gemini's safety blocks) and reports it only in the finish reason (`Refusal`, `ContentFilter`), with its details in response-level provider data. A refusal is replayed in its own form to OpenAI and as assistant text to every other provider. An adapter refusing a request it cannot send is a different thing: an `Unsupported` error.
_Avoid_: Refusal text (for a text block), declined answer

**Tool result run**:
The tool messages immediately after an assistant message, answering its client tool calls in any order. OpenAI Chat Completions, Anthropic and Gemini expect each call's result there and reject a result anywhere else, so a user message between a call and its result breaks the conversation. A tool call id names one call in the whole conversation: Anthropic rejects an id an earlier turn already used, so an adapter that makes up an id for a call sent without one scopes it by the response, and uses only `[A-Za-z0-9_-]` so any provider accepts it. Checked by `history::validate`. The id is only a correlation key: an encoder whose API refuses an id from another provider maps it to one it accepts (`ToolCallIdFormat`), the same way for the call and its result.
_Avoid_: Tool batch, tool turn

**Client-executed provider tool**:
A provider-defined tool the caller runs whose call is not a function call, such as an OpenAI Responses custom tool (freeform text input, optionally constrained by a grammar), shell call or computer-use call. Its call arrives as other content, never as a `ToolCall`, so `tool_calls()` and `history::validate` do not see it; the provider crate reads it and builds its answer (`allama_openai::ClientToolCall`), and the finish reason is `ToolCalls` while it waits. OpenAI's custom tool calls have one shape in Responses and Chat Completions, so they move between the two. Its answer is paired with it: an answer whose call is in the request but not replayed is dropped with the call. A provider tool expressed as an ordinary function call (Anthropic's bash and computer-use tools, Gemini's computer use) is a plain tool call. See ADR 0014.
_Avoid_: Custom tool (OpenAI's name for one kind), non-function tool call, provider tool call (which the provider runs itself)

**Empty message**:
A user or assistant message with nothing to send (`Message::is_empty`): an assistant message with no blocks, or a user message with no parts or only empty text. A message an adapter's own drops emptied, such as one holding only another provider's state, counts as empty for that adapter. A tool message is never empty: an empty result is a valid answer. Adapters skip empty messages without a warning, since nothing is lost, and drop an assistant message left last only because the messages after it were skipped or dropped (`allama::adapter::chat::retain_turns`). `retain_provider_state` leaves emptied messages in place, so their application data stays and warning indices remain the caller's.
_Avoid_: Blank message, removed message

**Prefill**:
An assistant message the caller sends last on purpose, so the model continues from it. It is the conversation's shape, not a request field: an adapter sends it as written or refuses it with `Error::Unsupported` naming that message (`Subject::Content` with no block), and never drops it. An assistant message left last only because the messages after it are empty is not a prefill; it is dropped (see Empty message). Whether a model takes a prefill is the provider's answer (ADR 0015): Anthropic refuses one on Claude 4.6 and later generations, and on earlier ones with an output format, and that refusal is `Unsupported` at stage `Provider`. An extraction request should end on a user message for that reason.
_Avoid_: Assistant prefill as a request option, partial response, continuation (a server-side continuation is something else)

<a id="pending-tool-call"></a>**Pending tool call**:
A client tool call whose result is not yet in its tool result run, as in an agent loop between the model's turn and the tool results, or after a turn that failed there. A history that is sent has none; a resumable history has them only in its last assistant message.
_Avoid_: Dangling tool call (for one that can still be answered), open tool call

**Output cap**:
The most tokens one response may produce (`ChatRequest::max_output_tokens`), reasoning included: the provider counts hidden reasoning against it. A cap the caller set is a bound an adapter never loosens: a reasoning hint that does not fit inside it (an Anthropic thinking budget derived from the effort) is lowered or dropped as a hint, and a larger cap never gives it less room, and a provider option that contradicts it (an Anthropic `budget_tokens` at or above it) is an invalid request. Only a cap the adapter chose itself, because the provider requires one, may grow to fit.
_Avoid_: Max tokens (Anthropic's wire name), answer length (it covers reasoning too)

**Sampling parameters**:
The request fields that shape how the model picks each token: `temperature`, `top_p`, `top_k`, `frequency_penalty` and `presence_penalty`. The fractional ones are `f64`, so a value is sent as written (`0.2`, not the `0.20000000298023224` an `f32` widens to); `top_k` is a count. An adapter that cannot send one drops it with an `Unsupported` warning, or refuses it under `UnsupportedPolicy::Error`.
_Avoid_: Generation config (Gemini's wire name), inference parameters

**Automatic caching**:
Caching the longest reusable prompt prefix without breakpoints, asked for with `PromptCache::automatic()`. Some providers always do it (OpenAI, Gemini implicit caching), so the intent is already satisfied there; others do it only on request and bill the cache writes (Anthropic), so only this explicit intent turns it on. A cache key or a retention alone is a hint for providers that take one and never turns caching on.
_Avoid_: Prompt cache key as an on switch, implicit caching (Gemini's name for its always-on form)

### Providers and models

**Provider**:
Anything that executes model operations for a requested model: a hosted API, a gateway, or an in-process inference runtime. It is not tied to a vendor or transport, and one provider may serve many models.
_Avoid_: Vendor, API client

<a id="provider-key"></a>**Provider key**:
A short name that selects a provider configuration, such as `openai`, `anthropic`, or `groq`. It names how to reach a provider, not a vendor: `openai` and `openai-chat` reach the same vendor over different APIs.

<a id="provider-name"></a>**Provider name**:
Who served a call, as the provider reports it (`ProviderIdentity::provider_name`), such as `anthropic`, `gemini`, or `groq`. Used in logs, spans, errors, and the provider stamped on responses. Wrappers report the name of the provider they wrap.
_Avoid_: Provider key (which selects a configuration), namespace

<a id="model-address"></a>**Model address**:
A provider key and a model id joined by the first colon, such as `openai:gpt-5.5` or `ollama:llama3:8b`. Keys are lowercase ASCII letters, digits and `-`. The model id passes to the provider unchanged, and an address never resolves aliases. An address with whitespace is refused, never trimmed (`allama_providers::ModelAddress`, ADR 0016). Addresses are trusted configuration: they choose the server and the key.
_Avoid_: Model string, model spec, model URI

<a id="provider-settings"></a>**Provider settings**:
How a lookup by provider key or model address builds a provider (`allama_providers::ProviderSettings`): what the constructor configures on the adapter (a credential scope, a capability resolver), and the default stack the lookup wraps around it, outermost first: retries (on by default, with `RetryPolicy::default()`), a per-attempt timeout, and a default unsupported-feature policy, each built once and erased into one `Dyn*Provider`. Built-in and registered constructors get the same stack, and `chat::wrap` (with `embedding::wrap` and `decision::wrap`) puts it around a provider built directly, refusing the settings only a constructor applies. Core `allama` stays bare: a provider built directly retries nothing until it is wrapped, so retries are configured in one place and never nest.
_Avoid_: Client options, provider config (the adapter's own builder)

**Model mapping**:
Rewriting the model id a caller names into the one a particular provider receives (`MapModelProvider`), such as giving a fallback side a model from its own catalog. The mapping applies to requests and to every question asked by model id (capabilities, limits, validation), so the answers describe the model actually sent. It never chooses between providers.
_Avoid_: Model routing (which picks a provider), model translation

**Transparent wrapper**:
A provider that wraps another to change how calls run (retries, timeouts, tracing, recording) without changing who answers questions about the provider: it forwards the provider name, the namespaces it reads, typed lookup, capability answers, limits, and validation, and passes the provider's answers through unchanged (the data path). Checked by `allama_conformance::assert_chat_wrapper` (and `assert_wrapper`).
_Avoid_: Middleware (for the forwarding guarantee), decorator

**Data path**:
What a wrapped provider's calls answer, on its way to the caller through a wrapper: the response's warnings (in order; a wrapper may add its own), provider data, usage and ids, every stream event in order (a chat stream ends after its first error, which is its last item), and every error with what a caller decides on (kind, status, retry hint and directive, refused subject and stage). A wrapper adds context to an error with a message prefix, never by re-wrapping it. Checked by `allama_conformance::assert_transparent_chat_data_path` and its variants.
_Avoid_: Passthrough (for the whole guarantee)

**Middleware**:
A transparent wrapper that adds behaviour around calls: an async request hook (a memory or retrieval lookup), or observing responses and streams. It is not a separate concept from wrappers: it forwards everything a transparent wrapper forwards. It reports what it changed as coded warnings and observations through `tracing`, never as provider data (`FallbackAttempts` is the reserved exception). The `middleware` example in `allama-examples` is the recipe; a `Layered<P, M>` wrapper type that owns the forwarding is designed in ADR 0013 and comes after 0.1.
_Avoid_: Interceptor, advisor, hook (for the whole wrapper)

<a id="serving-provider"></a>**Serving provider**:
The provider that answered a call, as opposed to the addressed provider, the one the request was sent to. They differ after a fallback served the call. Chat, embedding and decision responses name it in `provider`. Spans keep the addressed provider in `gen_ai.provider.name` and the serving one in `allama.response.provider`; when nothing served, `allama.response.provider` names the provider that ended the call (the one the error names, or the last fallback side tried when the call was dropped), else the addressed one.
_Avoid_: Actual provider, backend

<a id="call-scope"></a>**Call scope**:
What one call's resilience wrappers report into (`allama::wrappers::call_scope`): attempts, backoff, retry directives, budget denials, give-up reasons, failovers, the serving provider, and the server each request went to, each a `CallFact`. `TracingProvider` opens one per call and records its summary on the span once. A fallback runs each provider it tries in a side, which counts as one attempt when nothing inside reported any. Facts reach the innermost scope; outside any scope, reporting does nothing. Only `RetryingProvider` and `FallbackProvider` report the resilience facts; outside allama, a transport reports the server (`CallFact::server`).
_Avoid_: Call context, span fields (the scope does not depend on tracing)

**Capability answer**:
What a provider reports about a portable capability for a model: `Supported`, `Unsupported`, or `Unknown`. Answers follow three tiers: a verified entry gets its documented answer; a parseable model id in a known family, with no specialty marker (`-tts`, `-image`, `realtime`), gets the family default; anything else gets `Unknown`, or `Unsupported` where the marker settles it. Answers must be true in both directions: a claim needs evidence (a captured fixture or a cited provider doc), and a request that uses an `Unsupported` capability ends in a warning or `Error::Unsupported`, never a silent send. `Unknown` means "send it and let the provider decide": it never blocks a request and never drops anything, so the encoder sends what the request asks for or reports it. A field the wire format cannot carry is `Unsupported` for every model, as is one the endpoint documents it ignores, and one the encoder always refuses (audio on the Responses API). A capability covers only what one answer can be true for: `StructuredOutput` is JSON Schema response formats, since providers split on JSON mode apart from schemas (Anthropic takes schemas and has no JSON mode, DeepSeek the reverse), so JSON mode has no capability and an adapter without it refuses it. Checked by `allama_conformance::capabilities` (`assert_capabilities_honest` probes every answer, a model the provider cannot know included).
_Avoid_: Feature flag, support level

<a id="usage"></a>**Usage**:
The tokens a call consumed, as the provider reported them (`Usage`), on one contract for every adapter: `input_tokens` is the whole prompt, cache reads and writes included; `output_tokens` is the whole output, reasoning included; `total_tokens` is at least their sum; `None` means not reported. Breakdowns of either total and counts of provider tool requests sit beside them, never added to them. Every adapter normalizes to this contract (`Usage::contract_violation` checks it), and all usage arithmetic saturates. See ADR 0005.
_Avoid_: Token count (for the set), billing (usage is what was reported, not what was charged)

**Usage breakdown**:
A counter in `Usage::breakdowns`: tokens already counted in `input_tokens` or `output_tokens`, such as `audio_input_tokens` or `cache_write_1h_input_tokens`. Its key ends in `_input_tokens` or `_output_tokens` to say which total it is part of, so adding it to that total double counts.
_Avoid_: Additional usage, extra tokens

**Tool requests** (usage):
The billable requests provider-executed tools made during a call, counted by tool (`Usage::tool_requests`, keyed `web_search`, `web_fetch`, …). Counts, not tokens, and part of no token counter.
_Avoid_: Server tool use (Anthropic's wire name), tool usage

**Stream usage**:
Whether the response a stream collects into carries token usage (`ChatCapability::StreamUsage`). A capability that describes output: nothing in a request asks for it, and its check looks at a collected stream, which has usage when the answer is `Supported` and none when it is `Unsupported`. A budget guard folds a stream's usage with `StreamCollector::metadata_only()`, which applies each `ResponseMetadata` event by its usage mode (a snapshot replaces the usage so far, a delta adds to it), and relies on the result only under `Supported`.
_Avoid_: Usage streaming, stream options (the OpenAI request field that asks for it)

**Capability resolver**:
A caller's override of capability answers for model ids a provider cannot classify, such as a fine-tuned model, a deployment name or a gateway alias: a function of the model id and the capability that returns an answer, or `None` to keep the provider's (`ChatCapabilityResolver`). Given to an adapter (`with_chat_capabilities`, or `ProviderSettings::chat_capabilities` for a provider the registry builds), it feeds the adapter's own checks, so it may upgrade an answer as well as downgrade it; given to `CapabilityOverrideProvider`, which wraps any chat provider, a `DynChatProvider` included, it may downgrade an answer or settle an `Unknown` one, but not upgrade an `Unsupported` answer, which the wrapped provider would still refuse. Chat only: embedding and decision answers have no override yet, so an `Unknown` one stays.
_Avoid_: Capability override (for the function itself), capability map

**Model rules** (OpenAI-compatible):
Facts about the model behind an OpenAI-compatible endpoint, as opposed to the endpoint's dialect (`CompatQuirks`): what it rejects while it reasons, whether it takes tools, how it constrains a response schema, and whether it takes a JSON Schema response format at all (`ChatModelRules`, per model id through `ProviderBuilder::model_rules`). A preset sets them where its provider's docs differ by model, such as Groq's strict mode, which covers only the models Groq lists, or an Ollama cloud model reached through a local server. A model's `StructuredOutput` rule outranks the provider's declared capabilities, and a capability resolver outranks the rule.
_Avoid_: Model profile (the OpenAI adapter's table), quirks (for model facts)

**Transport options**:
Per-request settings for how a request travels rather than what it asks the model: HTTP headers and credentials to send with one call (`allama_http::HttpRequestOptions`, under the `allama-http` options namespace), such as gateway tags or a per-tenant key. Never a field of the portable request. Every HTTP adapter applies them with `HttpRequest::apply_options` and its credential slot once its own headers are set: a plain header replaces the adapter's header of the same name, and a credential reaches only the adapter whose credential scope it is addressed to. A plain header that would carry a credential is refused with `Error::Configuration`. Credentials never leave the process through them: `Debug` redacts them and serialization omits them. Providers that send no HTTP request (the Workers AI binding, mocks) ignore them. Checked by `allama_conformance::assert_transport_options_applied`.
_Avoid_: Extra headers (for the option as a whole), interceptor, request metadata

<a id="credential-scope"></a>**Credential scope**:
The account a per-request credential belongs to, and so its addressee: `HttpRequestOptions::credential(scope, credential)` addresses an `allama_http::Credential` (an API key, or another auth header such as a gateway token) to it, and only the adapter whose credential slot has that scope sends it. A second addressee beside the namespace: the namespace names a wire format, the scope an account. It is independent of the provider name and the namespaces, so changing it never changes telemetry identity or where state lives. Bundled providers default it to their provider name and publish it as `DEFAULT_CREDENTIAL_SCOPE`; builders change it with `credential_scope`. Two accounts of one provider need two scopes; providers behind one gateway share one. A provider built with a scope and no configured key requires a credential on every request (bring your own key).
_Avoid_: Tenant, account id, auth namespace

**Credential slot**:
An HTTP adapter's side of a per-request credential (`allama_http::CredentialSlot`): its credential scope, its placement (`Bearer`, `Header(name)`, or `Custom` for an adapter that signs or places the credential itself), the configured auth headers a credential replaces, and whether a credential is required. The adapter passes it to `HttpRequest::apply_options`, which places the scope's API key, adds its header credentials, and refuses plain headers the slot owns. There is no query-string placement, because spans record the URL.
_Avoid_: Auth config, credential placement (for the whole slot)

### History and namespaces

**History**:
A persistable conversation (`allama::history::History`): its messages, in order, in an envelope that records the format version (`HISTORY_FORMAT_VERSION`). It holds no system prompt: the system prompt is set on every request, and a conversation that keeps its own prompt stores what it needs to rebuild the prompt (the text, or a persona id) next to the history and builds the `SystemPrompt` on every request: a `SystemPrompt` is not versioned. Saving writes the current format version; loading reads every version up to it and fails on a newer one.
_Avoid_: Transcript (a recorded stream), session

**Namespace**:
The key under which a provider's provider data lives, such as `anthropic`, `google-ai`, or `openai-compat:groq`. It names the only provider that may read an entry: the origin of state a provider minted (a provider replays only state under its own namespace), and the addressee of settings a caller attaches to a request item, such as an embedding input. An empty namespace is the same as an absent one. It may differ from the provider name: Gemini reports `gemini` and stores state under `google-ai`. Every provider crate exports both as `NAMESPACE` and `PROVIDER_NAME`. A provider answers which namespaces it reads (`ProviderIdentity::reads_namespace`), a question rather than an identifier, so a wrapper can answer for every provider that may serve it: a fallback reads what either side reads. Per-request credentials are not addressed by namespace but by credential scope, since they belong to an account rather than a wire format.
_Avoid_: Origin (as a separate concept), provider id

**Options namespace**:
The key a request options type is read under (`ProviderOptions::NAMESPACE`), and the key of its JSON form in `with_options_json`. It usually equals the provider's namespace, but need not: every OpenAI-compatible preset reads options under `openai-compat` and stores state under `openai-compat:groq`, `openai-compat:ollama`, and so on. Some options namespaces belong to no provider: `allama.extraction` is read by extraction, and `allama-http` holds transport options, which the transport code of every HTTP adapter reads. No provider mints state under an options-only namespace. One options bag holds one options type per namespace: `RequestOptions::resolve` fails with `Error::InvalidRequest` when a typed value of another type was attached under the namespace it reads, instead of ignoring it.
_Avoid_: Provider namespace (for options), option key

**Extra fields**:
The `extra` map of a provider options type: request body fields the crate does not model, merged into the body as given. They never override a field the adapter owns (one its encoder can send, or a typed option): such a key is refused as `Error::InvalidRequest` before anything is sent (ADR 0006, amended 2026-10-07). In the options' JSON form they are the nested `extra` object, so a misspelled option is still an error rather than a body field.
_Avoid_: Passthrough, raw body, unknown fields (those are refused)

**Reserved key**:
A key allama itself uses in a map it shares with providers or applications, in one of three styles. `allama` is the provider-data namespace allama writes under (`ALLAMA_NAMESPACE`, holding `FallbackAttempts`), spelled like a provider's namespace because it is one. `allama.<area>` names a core record by the area that owns it: application data (`allama.turn`), error metadata (`allama.extraction`, `allama.embedding`), the extraction options namespace (`allama.extraction`), and span attributes (`allama.retry.*`); every key starting with `allama.` is reserved. `allama-http` is the options namespace of the transport options the `allama-http` crate defines, named after that crate as a provider's options namespace is named after its provider. A new core record takes `allama.<area>`; options defined by another allama crate take the crate's name.
_Avoid_: allama namespace (for the dotted keys), internal key

**Decline explanation**:
The text a provider gives for declining on the model's behalf, such as Anthropic's `stop_details.explanation` or Gemini's `finishMessage` on a safety finish. The provider's decline data stays its own, in response-level provider data under its namespace; by convention the adapter also stores the explanation's text there under `DECLINE_EXPLANATION_KEY` (`"decline_explanation"`), so code that reports a decline (extraction) can quote it without knowing the provider. A refusal the model writes is content, not a decline explanation.
_Avoid_: Refusal message, stop details (Anthropic's field)

**Trust boundary**:
Messages, histories and requests from an end user are untrusted. Provider data, `Other` content and provider files are escape hatches adapters send as they are, with the operator's key, and a `ChatRequest`'s options set provider headers and body fields; namespace gating keeps one provider's state from another, not the application's state from a forgery. Keep conversations server-side, never deserialize a whole request from a client, and pass a client-held history through `history::retain_portable_content`, which keeps only portable content.

**Replay state**:
What a provider wrote into a conversation for its own later use: reasoning with its signature or encrypted content, provider data under its namespace, and `Other` content or provider tool calls and results of its origin. It stays in the stored conversation, so an adapter that cannot send it (another provider's state, or reasoning the model cannot take back) drops it on every turn; the drop is a coded `Dropped` warning (`Degradations::dropped_replay_state`, with a `ReplayState`), one per code per request whatever the number of messages (`Subject::Messages` lists them), logged at `WARN` once per provider and model; reasoning the receiving provider wrote itself drops without one, and for another provider's state its message points to `retain_provider_state_for`, which removes it once when the conversation moves for good, keeping what the new provider reads. `Warning::replay_state` reads the `ReplayState` back, and `ReplayState::is_foreign` tells another provider's state from state the receiving endpoint cannot take back (`dropped.reasoning_replay`, `dropped.content_replay`). Content the application sent is not replay state, and its drops stay uncoded.
_Avoid_: Opaque state (for the whole set), foreign data (for the reasoning and content)

<a id="application-data"></a>**Application data**:
The application's own data about a message or system prompt (`app_data`): message ids, timestamps, UI state, per-turn usage. The application owns it: adapters never read or send it, `retain_provider_state` never touches it (it never removes a message), and tracing never records it, while `History` and `RecordingProvider` keep it. Keys starting with `allama.` are reserved for conventions allama publishes.
_Avoid_: Metadata (taken by response, error and extraction metadata), extensions, app namespace (provider data belongs to providers)

<a id="turn-envelope"></a>**Turn envelope**:
What a chat response says about itself, apart from its content: the response id, request id, model, serving provider, usage, finish reason (normalized and raw) and warnings (`TurnEnvelope`, from `ChatResponse::envelope()`). Replay never needs it; cost accounting, support and audit do. Applications that keep it store it as application data under `allama::history::TURN_ENVELOPE_KEY` (`"allama.turn"`) on the assistant message each provider call produced: one envelope per call, so a turn with a tool round trip stores two. Nothing stores it automatically. What a wrapper adds in provider data, such as `FallbackAttempts`, is not part of it; an application stores that beside it under a key of its own. An envelope written by a newer version loads in an older one, with unknown warning values read as `WarningKind::Other`, `RequestField::Unknown` and `Subject::Unknown`; that tolerance is read-only and lossy.
_Avoid_: Response metadata, turn metadata

### Errors and warnings

**Configuration error**:
A failure raised before any request reaches a provider because the client is set up wrong: a missing credential, an unknown provider key, a known provider key without the operation (a registry lookup of chat for a decision-only key), a malformed model address, or a missing runtime binding. Retrying or failing over does not fix it. A provider rejecting the credentials it was given is an authentication error instead.
_Avoid_: Auth error (for a missing key), invalid request (for a bad provider key)

**Local refusal**:
A call a wrapper or the application refused before any provider saw it: a tenant budget, a policy or moderation pre-check (`Error::Rejected`, kind `rejected`). It blames no provider, is not retryable and never fails over, since every side would be refused the same way. A typed cause goes in `Error::with_source`.
_Avoid_: Quota exhausted (a provider's account), invalid request (a malformed request), configuration error

**Failed failover**:
A call that failed on every provider a failover tried. It returns the last provider's error as itself, with that error's kind, status and context, so it is handled like any other failure; the errors tried before it are kept on it, newest first (`Error::fallback_from`, set with `Error::with_fallback_from`), and logged as nested redacted logs. A last error without a context (`Serialization`, `Extract`) keeps none, and the `allama::fallback` event is then their only record.
_Avoid_: Fallback error (there is no such kind), final error

<a id="error-kind"></a>**Error kind**:
What a failure tells the caller to do, as the `Error` variant (`Error::kind_name`): retry the same provider (rate limited, overloaded, timeout, a provider failure with no status or a 5xx status except `501` and `505`), fail over, fix the request, or fix the setup. One upstream signal has one kind on every adapter: an HTTP status means what `allama_http::status_error` says (`408`, `504` and `524` time out, `503` and `529` are overload, `499`, a request the client closed, is a provider failure that is never retried, and so are `501` and `505`, which a server answers the same way every time), which an adapter refines from what the body says (a quota code, a named model, a prompt over the context window). Two adapters also refine a bare `404` (an empty or non-JSON body) from the status: Gemini reads it as an unknown model, System One as a server with no System One endpoint. A success response whose body cannot be decoded is an *unexpected response*, carrying the provider, status and request id, or a response without what the call needs (no text for `ask()`; a refusal there is the error's upstream message); `Serialization` is only for the caller's own data. A timeout may still have been billed. Checked in part: `allama_conformance::assert_stream_errors_classified` checks that each error envelope the adapter builds has its kind over HTTP and inside a stream, and `assert_malformed_response_contract` the undecodable success body. No contract yet checks how an adapter maps a bare status.
_Avoid_: Error code (the vendor's `provider_code`), error type (the vendor's `provider_type`)

**Preflight**:
Checking a request against what the serving provider reports for its model before anything is sent (`allama::adapter::chat::preflight`, `allama::adapter::embedding::preflight`). Under the request's unsupported-feature policy (unset inherits a provider-level default, `UnsupportedPolicyProvider`, and is `Warn` otherwise), a semantic feature the provider reports unsupported fails, an unsupported hint is removed with a warning (or fails under the strict policy), and an unknown answer never blocks but is recorded. Adapters run it through `chat::prepare` first in every call, and validation runs it on a dry run.
_Avoid_: Validation (for the check a provider runs inside `chat`), capability gate

**Prepared request**:
A request on its way to being sent (`allama::adapter::chat::Prepared`, and its embedding and decision counterparts): preflighted by `prepare`, with the `Degradations` its encoding reports through. An adapter splits its call at the transport around it: everything before sending (option resolution, encoding, building the transport request) is one `encode` step, then the call sends and ends with `finish` or `finish_stream`, which join the warnings to the response and emit them as warning events. A dry run (`prepare_dry_run`) runs the same `encode` and ends with `into_validation`. Dropped on an error path, it emits its warnings only if the request reached the transport (`mark_sent`), since a request refused before sending describes a call that never happened.
_Avoid_: Encoded request (the wire form), request context

**Wire support**:
What an adapter's wire format can carry of a chat request (`allama::adapter::chat::WireSupport`): the portable request fields it has a slot for, and the kinds of content it takes (images, files, image detail, replayed images and reasoning, provider content). Declared once, it drives the standard degradation pass, `Prepared::degrade_to`, which refuses what the wire format cannot carry and must not lose, removes hints with a coded warning (or refuses them under the strict policy), and drops replay state with its warnings, so the encoder only encodes what it can send; and it answers the capabilities it settles (`WireSupport::chat_capability`: `Unsupported` for what it cannot carry). It describes the endpoint, not a model: a capability resolver may still answer for the model, and the pass still refuses what the wire format cannot carry.
_Avoid_: Supported fields (for the declaration), feature set, provider profile

**Validation**:
A dry run of a call (`validate_chat_request`, `validate_chat_stream_request`, `validate_embedding_request`, `validate_decision_request`): the error the call would return before sending, or a `Validation` with exactly the request-side warnings it would report, the capabilities the request relies on that the provider answered unknown (what the dry run could not know), and its coverage, `Capabilities` (only the capability answers were checked, the default) or `Encoding` (the adapter's own encoding ran, so the answer is the call's; every first-party adapter). It never checks credentials or that the model exists. A pair that fails over gives the validation of the side that would serve, and the other side's dry run apart (`Validation::fallback`). Checked by `allama_conformance::assert_validation_agrees`.
_Avoid_: Preflight (the capability check inside it), lint, check

**Accounted field**:
A key of a serialized `ChatRequest` that the request field audit (C2) either probes or exempts for a stated reason. The list (`allama_conformance::probe::ACCOUNTED_FIELDS`) is derived from `ChatRequest` itself, so a new portable field arrives with a probe or an exemption. An adapter accounts for a probed field by sending it, refusing the request with `Unsupported`, or naming it in a warning; state from another provider and application data are accounted for by never being sent.
_Avoid_: Field probe list, supported fields

**Probe** (conformance):
An async closure a provider crate hands to an `allama-conformance` honesty contract: it sends a request through a fresh provider over a canned transport and returns a `ProbeOutcome` with the result and the request bodies that reached the transport. A chat contract takes two: a chat probe, which calls `chat`, and a streaming probe, which collects `chat_stream` into the same outcome, so each check runs on both paths.
_Avoid_: Harness, fake provider

**Adapter under test** (conformance):
What an adapter hands the conformance aggregates (`AdapterUnderTest` for `assert_chat_adapter`, and its embedding and decision counterparts): the models to probe, its namespace, its probes, and the inputs some contracts need (its error mapping and its error envelope for each error kind, a stream replay, transport options, ...). A contract whose input is missing fails the aggregate, unless it is skipped as not applying to the adapter; an acknowledged defect is a known gap, which fails once it is fixed.
_Avoid_: Test config, harness settings

**Refusal shape** (conformance):
What makes a refusal of a capability or field acceptable to the conformance contracts: a warning whose subject names the capability's field or content, or an `Error::Unsupported` with stage `BeforeSending` and nothing sent. A warning about something else, or a refusal at another stage (an `Error::unsupported` built in an encoder), does not count.
_Avoid_: Any warning, any `Unsupported`

<a id="warning-event"></a>**Warning event**:
A warning a call reported, emitted as a `tracing` event by `allama::warning::emit`, under a target per kind (`allama::warning::unsupported`, `::dropped`, ...), with the `warning-events` feature: on by default in `allama`, off in every library that depends on it, so the application decides. A warning remembers that it was emitted, so the innermost layer that sees it emits it once, however many wrappers pass it on; a dry run, a request refused before sending, a replayed transcript and a recorded exchange never emit. A coded warning goes out at `WARN` once per provider, model, kind and code, then at `DEBUG`; an uncoded one at `WARN` every time. A warning a failure caused, a failover, keeps the failure's error kind as its cause (`Warning::cause`), which its event carries as `allama.warning.cause_kind`.
_Avoid_: Warning log, diagnostic

<a id="warning-code"></a>**Warning code**:
A stable name for the condition a warning describes (`Warning::code`, such as `unsupported.seed` or `dropped.reasoning_replay`). A code says the warning recurs with the request's shape or the configuration, so its warning event is deduplicated; a warning about this call alone (a failover, content the application sent) has none, unless the conversation's shape repeats it on every turn (`approximated.tool_result_error`, a tool error sent to an endpoint with no error flag; `approximated.tool_call_arguments`, tool call arguments that are not a JSON object, sent as `{}`). `Degradations` gives a warning about a request field or a system prompt the code `<kind>.<field>` when its producer set none.
_Avoid_: Warning id, error code

<a id="warning-message"></a>**Warning message**:
The text of a warning, a hint or a refusal before sending (`Warning::message`), written for people in one shape: `<limitation>; <outcome>[; <remedy>]`, with `"; "` only between the parts. The limitation says what the provider lacks and reads true on its own; the outcome's verb follows the warning kind ("`x` was not sent", "sent as X", "it was dropped", "served by `X`"); the remedy says what the caller can do. Identifiers are in backticks. A warning displays as `[provider]`, then the location for content, then the message, so the message names neither. A refusal under the strict policy is the same message without its outcome. Codes, kinds and subjects, not the text, are for programs.
_Avoid_: Warning description, reason (for the whole message)

<a id="seen-warnings"></a>**Seen warnings**:
The recurring warnings an application already showed (`allama::SeenWarnings`), so it shows each one once: `insert(model, &warning)` is `true` for a warning without a code every time, and for a coded one the first time its provider, model, kind and code are seen, the key the warning events deduplicate on. The application owns it and decides its scope (a conversation, a user, the process); the library never deduplicates `response.warnings` itself.
_Avoid_: Warning cache, deduplicated warnings

**Attempt scope**:
One attempt of a call that `RetryingProvider` may throw away (internal to `allama`). The warning events emitted while it runs wait in it: a committed attempt's go out (into the enclosing attempt scope, when there is one), a discarded attempt's are dropped, so a retried call logs each warning once. Separate from the call scope, which collects resilience facts for tracing.
_Avoid_: Retry scope, emission guard

**Retry hint**:
How long a provider asks the caller to wait before trying again (`Retry-After`, `retry-after-ms`), exposed as `Error::retry_after`. A retry policy treats a positive hint as the shortest wait: it waits at least the hint (the longer of the hint and its own backoff, plus bounded jitter) or, when the hint exceeds its hint ceiling (`RetryPolicy::max_retry_after`, separate from the cap on its own backoff), gives up and returns the error with the hint intact; it never retries sooner. A zero hint, including an HTTP date already past, counts as no hint.
_Avoid_: Backoff (the client's own schedule), cooldown

**Server retry directive**:
Whether the server asks the caller to retry a failed request (`x-should-retry: true|false`), exposed as `Error::server_retry_directive`. The default retry policy follows it ahead of the error kind and status, in both directions, but it never lifts the attempt limit, the retry hint give-up rule, or the rule against retrying after output, and it does not affect fallback. A custom classifier decides for itself.
_Avoid_: should-retry header (the transport detail)

**Retry budget**:
A shared allowance of retries across calls (`RetryBudget`): every retry draws tokens from it and successes refill it, so during an outage callers stop retrying and fail fast while first attempts still go through. It follows the AWS SDK retry quota and applies after every other retry rule, so neither the classifier nor the server retry directive can override a denial. Each `RetryingProvider` owns one by default, shared by its clones; sharing one across wrappers is explicit (`RetryPolicy::budget`). A budget is in memory, so it bounds one process.
_Avoid_: Retry quota (the AWS name), retry limit (that is `max_attempts`), rate limit

**Message provenance**:
Where an error's text came from. The constructor message is *authored* when library or adapter code wrote it with no upstream text in it (`Error::authored`), and of *unknown* origin otherwise, including every message an application builds. The provider's own text, such as an error envelope's `message`, is the *upstream message* (`Error::with_upstream_message`). The text of a foreign error (an HTTP client's, a JSON decoder's) counts as upstream: it can contain the request URL or quote the body. Only `Error::authored` sets the origin: `Error::with_context` merges a context into the error and never changes it.
_Avoid_: Trusted message, safe message

**Message prefix**:
Context a wrapper or application puts in front of an error's constructor message to say what it was doing (`Error::with_message_prefix`), as in `summarize step: HTTP 429`. It keeps the error kind, the context and the message provenance, so retries, fallback and the redacted error view treat the error as before; on a failed failover it applies to the returned error, not the ones it fell back from. The prefix is shown wherever the message is, so it is text the code wrote. Re-wrapping the error in a new one instead loses its kind and context.
_Avoid_: Error wrapping, error chaining (the error stays the same error)

**Redacted error view**:
`Error::as_log()`, the form of an error that is safe to log, persist (`FallbackAttempt`), and export: the kind, provider, status, provider code and type and the rejected field (`param`) when each is a bounded token (otherwise `<redacted>`; the filter keeps out prose, not values shaped like codes), the request id, the retry hint, the server retry directive, the body's length, the constructor message only when it is authored, an extraction's pass record (`allama.extraction`), a chunked embedding's progress (`allama.embedding`), both usage rather than input, and the errors a failed failover tried first, each as a nested view (`fallback_from`). It never includes the response body, other metadata, or the upstream message, except that `as_log().with_message(n)` opts in to the upstream message, truncated. `Display` is not redacted; `Debug` is compact (the variant, its message, and the context fields that are set) and shows the upstream message's length, not its text (`<withheld N bytes; print with {err}>`), the body's length, and neither a serialization failure's text nor an extraction's model output; it filters code, type and `param` the same way.
_Avoid_: Sanitized error, safe display

**Unsupported stage**:
Where a refusal (`Error::Unsupported`) happened (`UnsupportedStage`, read with `Error::unsupported_stage`): `BeforeSending` (the adapter refused through `Degradations`, so nothing reached the provider and nothing was billed), `Provider` (the provider refused the request it received, and the adapter mapped the reply with `Error::unsupported_by_provider`, only through its own table of wire paths, for replies a fixture pins), or `Unspecified` (`Error::unsupported`, which promises neither: the request may have been sent). The refusal also names what it refused (`Error::unsupported_subject`: a request field, or a content block or part with the caller's indices). Extraction's `Auto` mode falls back on a `BeforeSending` refusal, and on a `Provider` one only when native output was a guess; `FallbackProvider` fails over at every stage.
_Avoid_: Refused field (for the stage), local refusal, upstream refusal (as a type)

### Extraction

**Extraction request**:
The chat request an extraction derives its passes from (`Extractor::extract(&provider, &request)`, `ExtractExt::extract`). Every field is kept except the extraction mode option, which extraction consumes, and the fields the selected mode replaces: the response format in native mode, the tools and tool choice in forced-tool mode. A request with tools of its own fails with `ToolConflict` in every mode, before sending, so the same request behaves the same on every provider. To extract from a conversation alone, the caller starts a fresh chat request from its messages, or clears the tools on a copy; there is no separate extraction-only request type.
_Avoid_: Direct extraction, dedicated extraction, extraction orchestration

**Extraction pass**:
One provider chat dispatch made by an extraction operation. It counts dispatches owned by extraction, not retry or fallback attempts inside provider adapters. An extraction reports its passes and their summed usage whether it succeeds or fails.
_Avoid_: LLM call

**Extraction failure**:
An extraction operation that produced no value: why (no usable mode, a conflicting request, truncated or unusable output), the last provider response, and the record of its passes. Provider errors during an extraction are not extraction failures: they are returned unchanged, so retry and fallback policies keep their classification, and carry the extraction's pass record (mode, passes made before the error, summed usage) in their error context under `allama.extraction` (`ExtractionMetadata::from_error`). A stream that fails mid-way adds the usage and degradations it reported before failing to that record, without counting as a pass.

**Schema enforcement**:
How far the provider reported constraining an extraction's output to the target schema (`ExtractionMetadata::enforcement`, a `SchemaEnforcement`). Extraction always requests strict mode, and reads the warnings the adapter reported on the field its mode relies on: **exact** (no warning; constrained decoding ran over the schema with only rewrites that change nothing the target type can deserialize), **approximate** (an `Approximated` warning; constrained decoding ran over a widened schema, or one sent unchecked to an engine that does not report what it enforces), or **not enforced** (an `Unsupported` warning; strict mode was not honoured, or no call was made). It is the adapter's own report, not a verified grammar guarantee, and even exact does not guarantee every value (refusals, token-limit cut-offs, the capitalisation of string `enum` values on some engines). Under the strict unsupported-feature policy a strict drop fails the call instead, so the policy means "constrained, possibly approximately, or fail"; a pipeline that needs exact sets an enforcement floor. Below exact, the enforcement is reported in terms of the caller's type and is a warning event once per provider, model and type.
_Avoid_: Validated, guaranteed schema, schema enforced (as a yes/no)

**Extraction refusal**:
An extraction failure because the model or the provider declined to produce the output (`ExtractError::Refused`): a refusal the model wrote in place of the output, or a `Refusal` or `ContentFilter` finish. It is checked before parsing, next to truncation, and never retried, since asking again pays for the same refusal. Its text is the model's refusal or the provider's decline explanation, and `RefusalSource` (`Model`, `Provider`, `None`) says which, or that nothing explains it. Other abnormal finishes are parsed as usual; a parse failure then names the finish reason. Empty or whitespace-only native output is missing structured text, not a parse failure.
_Avoid_: Parse failure (for a refusal), filtered extraction

**Undeclared key**:
An object key in extraction output that no schema of the target type that may apply there declares, such as a misnamed `e_mail` for `email`. serde would ignore it and leave the intended optional field empty, so extraction rejects it on every pass that is not exactly enforced, as a retryable parse failure naming the key and its path, and counts the rejections. An object without `additionalProperties` counts as closed (stricter than JSON Schema, Pydantic, Zod and serde); maps, flattened maps and open schemas take any key; in a union, a key is undeclared only when no branch that may apply declares it. The extractor can allow them.
_Avoid_: Extra field, unknown field (serde's term for the same thing under `deny_unknown_fields`)

**Enforcement floor**:
The least schema enforcement an extraction accepts (`Extractor::min_enforcement`), off by default. Below it, extraction fails with `EnforcementBelow`: before sending when the provider's validation is a dry run of the call, after the pass (a paid call) otherwise. Approximate is the practical portable floor, since some providers never enforce common types exactly. An **extraction plan** (`Extractor::plan`) is what the floor is checked against: the mode, enforcement, degradations and validation coverage of an extraction, found by validating a schema-only copy of the request.
_Avoid_: Minimum strictness, guarantee level

**Automatic fallback (extraction)**:
An `Auto` extraction that picked native output and switched to forced-tool mode because the adapter refused the response format before sending (`Error::unsupported_field` is the response format and `Error::unsupported_stage` is `BeforeSending`), such as a schema with a map on a provider whose native output has no unconstrained form. When native output was only a guess from an `Unknown` answer, the provider's own refusal of the response format (stage `Provider`, a 4xx with no output to bill) falls back too. The refused attempt produced nothing and is not a pass; the metadata reports forced-tool mode and the refusal with its stage (`ExtractionMetadata::auto_fallback`). An explicitly chosen mode never falls back.
_Avoid_: Retry (nothing was sent), mode downgrade

**Degradation (extraction)**:
A warning an adapter reported on an extraction pass about the field the extraction mode relies on: the response format in native mode, the tools in forced-tool mode (`ExtractionMetadata::degradations`). The degradations decide the schema enforcement; warnings about other fields are not degradations.
_Avoid_: Extraction warning (the response keeps every warning)

### Decisions

**Decision model**:
A model that answers typed questions about an input with probabilities over a fixed set of candidate answers instead of generating text. A chat model prompted to estimate probabilities is not a decision model.
_Avoid_: System One model, judge, classifier, evaluation model

**Decision input**:
What a decision model reads before answering: text, or a JSON object or array whose key order is preserved. Images the model reads with it are a separate request field (`DecisionRequest::images`), not a form of the input, so the serialized input stays unambiguous; every question is about the input and the images alike.
_Avoid_: State

**Question**:
A named request for one answer about the decision input: a yes/no question, a choice among options, a score on an ordered scale of levels, or a locate question. Questions are answered independently; answers are not passed to later questions.

**Locate question**:
A question that finds one target in the request's images and answers with its bounding box (`Question::locate`, `LocateAnswer`): the box's left, top, right and bottom edges as fractions of the image's width and height, measured from its top-left corner. It is localization (one box for one target), not detection (any number of boxes). The answer carries no probability and has no "not found" form, so a yes/no question in the same request tells whether the target is there. A locate question needs an image in the request. TypeSafe calls it a Box question. See ADR 0018.
_Avoid_: Box question (TypeSafe's wire name), detection
_Avoid_: Primitive, field

**Option**:
One candidate answer of a choice question: a key the model can pick, with an optional description. The model reads both, so a key is prompt text.
_Avoid_: Choice (the question kind and its answer), criterion

**Choice set**:
The options of a choice question taken together, as in decision theory: the alternatives available, not the ones picked. In typed decisions, an enum whose variants are the options.
_Avoid_: Choices (candidate completions in chat APIs)

**Yes/no question**:
A question answered with the probability that the answer is yes.
_Avoid_: Noul, predicate, boolean question

**Threshold**:
The probability one side of a yes/no answer must reach before an application acts on it (`YesNoAnswer::decide`). It is above 0.5, so at most one side reaches it; when neither does, the answer **abstains**, for example to send the input to a person. It is the application's choice, not part of the answer.
_Avoid_: Confidence (provider data), cut-off

**Confidence**:
A provider-specific summary of how concentrated an answer's distribution is. It is not calibrated correctness, and providers compute it differently, so it is provider data rather than part of an answer.

**Scale**:
The ordered levels of a score question taken together, lowest first. A score is a position on a scale. In typed decisions, an enum whose variants are the levels.
_Avoid_: Rubric, criteria

**Level**:
One step of a score question's ordered scale, with a short label and an optional description. Levels are numbered from 0, lowest first, and a score answer is the probability-weighted average of their numbers, so the level nearest a score can be one the model gave probability 0 (a split between the ends of the scale averages to the middle); the most likely level is the one the model favoured.
_Avoid_: Criterion, rubric entry

### Embeddings

**Embedding input**:
One item to embed: an ordered list of embedding parts that the model combines into one vector. A request holds one embedding input per vector it returns. Most inputs are a single text; how a model combines several parts, and which combinations it accepts, is model-defined, and a model that embeds one part per input refuses several parts instead of joining them.
_Avoid_: Document, chunk, text (for the general case)

**Embedding part**:
One piece of an embedding input: a text or an image. Parts reuse chat's content types; which kinds a model embeds, and whether it combines several parts, are capability answers (`ImageInput`, `CompositeInput`).
_Avoid_: Segment, item

**Embedding response check**:
What every adapter checks on the vectors it decodes before returning them (`allama::adapter::embedding::check_response`): one vector per input, none empty, all of one size and of the requested dimensions when the request set them, every value finite. A response that fails is an unexpected response naming the model and both sizes, never repaired, since an index built from such vectors is silently wrong. A chunked call also checks every chunk against the size of the first.
_Avoid_: Validation (that is the dry run of a request), vector sanity check

**Input type**:
What the inputs of an embedding request are for in asymmetric retrieval: a search query or a document it searches (`EmbeddingInputType`). A hint: providers with a native field send it (Gemini `taskType`, `input_type` on some endpoints), a symmetric model honors it by sending nothing, and the others drop it with a warning or fail under the strict policy. Only this distinction is portable; other embedding tasks (classification, clustering, similarity) are provider options.
_Avoid_: Task type (for the portable field), purpose, mode

**Symmetric embedding model**:
A model that embeds a query and a document the same way, so the input type is honored without sending anything (OpenAI `text-embedding-3-*`, `text-embedding-ada-002`). Only a model the adapter positively identifies counts: it answers `InputType` supported, and the hint is removed without a warning under every policy. An unknown model is never assumed symmetric.
_Avoid_: Role-free model, untyped model

**Role prefix**:
Text a model expects at the start of an input to tell a query from a document, instead of a request field: nomic's `search_query: `, E5's `query: ` / `passage: `, BGE v1.5's instruction on queries only, Google's `task: search result | query: `. A dropped input type's warning names the prefix the model's card documents, or says the model may expect one; the library never adds it, since a prefix silently changes what is embedded.
_Avoid_: Instruction (for the general case), prompt

**Embedding chunk**:
One request of a batch split by input count (`EmbeddingChunk`, yielded by `embed_chunks`): its offset in the whole batch and its response, whose input warnings name indices in the whole batch. Chunks run one after another; after a failure, the error's chunked embedding progress names the offset to resume from, and `embed_chunked` is the all-or-nothing merge of them.
_Avoid_: Page, batch (for one chunk)
