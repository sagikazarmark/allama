# allama-examples

Runnable end-to-end examples for `allama` against real providers.

## Quick start

Every example takes its provider and model from one **model address**
(`provider:model`), read from `ALLAMA_MODEL` and built with
`allama_providers::chat::from_model`, the way an application reads its model
from configuration. Unset, it is a local Ollama model, which needs no
credentials (`ollama pull qwen3:1.7b`):

```bash
cargo run -p allama-examples --example provider_chat -- "Explain Rust ownership"
ALLAMA_MODEL=openai:gpt-5-mini cargo run -p allama-examples --example provider_chat -- "Explain Rust ownership"
```

Any [`allama-providers`](../allama-providers) key works: `openai` (Responses
API), `openai-chat` (Chat Completions), `anthropic`, `gemini`, `groq`,
`ollama`, and the rest. Each provider reads its own environment
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OLLAMA_HOST`, ...).

Embedding examples read `ALLAMA_EMBEDDING_MODEL` the same way, with
`ollama:nomic-embed-text` unset (`openai:text-embedding-3-small`,
`gemini:gemini-embedding-001`). Two examples default to another model:
`provider_vision` to `ollama:gemma3:4b`, which takes images, and
`provider_multimodal_embedding` to `gemini:gemini-embedding-2`, since
Ollama embeds text only.

Each example builds its provider in two lines, with no helpers from this
crate, and its module docs list the dependencies it needs, so it copies
into an application as is.

The default model of each hosted provider is in `KNOWN_PROVIDERS` in
`tests/support.rs`, the one maintained list, which the live tests run:
`openai:gpt-5-mini`, `openai-chat:gpt-5-mini`,
`anthropic:claude-haiku-4-5`, `gemini:gemini-3.5-flash`; for embeddings
`openai:text-embedding-3-small` and `gemini:gemini-embedding-001`.

## Examples

Start here:

- `web_backend`: one chat turn the way a web backend handles it: system
  prompt and stored history, streamed text deltas, one tool round trip by
  hand (`with_collector`, `push_tool_result`), errors before the
  first output kept as HTTP status codes (`until_committed`), errors
  mapped to HTTP responses by kind (`is_retryable`, `retry_after`,
  `Configuration` and `ModelNotFound` as `500`, `ContextLengthExceeded`),
  the recommended provider stack (tracing, a deadline on the first output,
  an optional fallback from `ALLAMA_FALLBACK_MODEL`, retries with a 5 s
  `max_retry_after`), each reply's turn envelope stored on its message
  (one per provider call), usage summed over the turn, and a rollback when
  a turn fails (nothing is stored; `axum_sse` stores a checkpoint
  instead). It uses only public API (no helpers from this crate), so it
  copies into an application as is with `allama` (features `schema` and
  `tracing`) and `allama-providers`; without `ALLAMA_MODEL` it uses
  `openai:gpt-5-mini`, not a local Ollama
- `axum_sse`: the same turn as a real axum endpoint streaming server-sent
  events, the server the [applications guide](../../docs/guides/applications.md)
  quotes: the same provider stack, defined once for `main` and the handler
  tests, conversations stored as a `History` of messages, the commit
  point, tool-call progress events, what to store when a turn fails after
  the first token (`history::validate`), two-level error logging, and
  handler tests with `TestProvider` behind a `Registry` key and scripted
  failures. It serves `127.0.0.1:3000` until stopped

Chat examples (any chat provider):

- `chat_repl`: a chat in the terminal, one stdin line per turn: streamed
  text, reasoning, refusals and tool progress on stderr, one local tool
  (`get_time`) in a streaming tool loop with a step cap, a turn stored
  whole or rolled back whole, warnings shown once with `SeenWarnings`, a
  smoke call at startup and on `/model <provider:model>`, which switches
  models mid conversation and keeps the current one on a setup mistake,
  usage summed per turn and per session, and optional settings from the
  environment (`ALLAMA_REASONING_EFFORT`, `ALLAMA_MAX_OUTPUT_TOKENS`).
  Like `tool_loop` it uses only public API and a local Ollama without
  `ALLAMA_MODEL`
- `provider_chat`: one-shot request/response
- `provider_stream`: normalized streaming text and metadata events
- `tool_loop`: a tool loop with a step cap: tools dispatched by name,
  `push_tool_result` and `push_tool_error`, `parallel_tool_calls(true)`, the
  finish reasons it refuses to act on, and warnings shown once with
  `SeenWarnings`. It uses only public API and builds its provider with
  `allama_providers::chat::from_model_with`; without `ALLAMA_MODEL` it uses
  a local Ollama (`ollama:qwen3:1.7b`). After the loop it extracts the
  answer as data, with a request that ends with a user turn. Its tests run
  against a mock (`cargo test -p allama-examples --example tool_loop`)
- `provider_streaming_tools`: raw `StreamEvent` iteration showing tool-call blocks mid-stream
- `provider_vision`: an image sent with the question, inline (`ContentPart::image_base64`) or as a URL
- `provider_reasoning`: `ReasoningConfig` + `reasoning_text()` inspection (see the example for per-provider notes)
- `provider_extract`: env-driven structured extraction across providers
- `history_persistence`: persist a conversation with `History`, reload it, and continue it (signatures included)

Embedding examples (`ALLAMA_EMBEDDING_MODEL`; Anthropic has no embedding API):

- `provider_embedding`: batch text embedding with vector preview
- `provider_multimodal_embedding`: cross-modal search, ranking images (and a
  caption combined with an image) for text queries; needs a model that embeds
  images, such as `gemini:gemini-embedding-2`

Decision example (decision models over System One; it does not use `ALLAMA_MODEL`):

- `provider_decision`: triage a ticket with `#[derive(Decision)]` and
  `decide_as`. Uses TypeSafe when `TYPESAFE_API_KEY` is set (`TYPESAFE_MODEL`,
  default `jev-latest`), otherwise a local Ollama 0.35+ (`OLLAMA_HOST`,
  `OLLAMA_DECISION_MODEL`, default `nimble`).

Extraction example (an extraction model, unstable in allama; it does not use
`ALLAMA_MODEL`):

- `provider_extraction`: turn a damaged-product claim into typed values
  (a span, a computed total, a duration, a list, an estimate with its
  interval, a null with its probability) with Cloudflare Workers AI's
  Clairic (`CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`;
  `CLOUDFLARE_EXTRACTION_MODEL`, default `@cf/cloudflare/clairic`).

Multi-provider and provider-specific:

- `cross_provider_fallback`: `FallbackProvider` failing over from an unreachable OpenAI endpoint to Anthropic; shows the served provider, recorded attempts, and non-retryable errors
- `openai_responses_reasoning_tools`: OpenAI Responses API with a reasoning model; encrypted reasoning replayed statelessly across a tool call and a JSON round trip
- `anthropic_prompt_caching`: portable `CacheBreakpoint` on a system prompt; second call reads the cache

Observability:

- `trace_context`: the observability guide's trace context recipe: the
  current span's W3C `traceparent` sent with every request, with the
  application's own OpenTelemetry version, through a transport wrapper
  (providers you build) or a chat wrapper (providers from the registry),
  and a `traceparent` the caller set kept. Its tests check both (`cargo
  test -p allama-examples --example trace_context`); `main` replays canned
  wire data, so it needs no credentials

Writing a provider:

- `text_gateway`: the shortest complete native adapter, for a text-only
  gateway with JSON answers and NDJSON streaming. It declares what its wire
  format carries (`WireSupport`) and lets `Prepared::degrade_to` refuse or
  drop the rest, and its test runs the whole conformance aggregate (`cargo
  test -p allama-examples --example text_gateway`); `main` replays canned
  wire data, so it needs no credentials
- `native_adapter`: the native HTTP adapter the
  [provider author guide](../../docs/guides/provider-authors.md) builds and
  quotes, for a fictional "Acme" API: preflight and degradations, typed
  options, provider data, tool calls, a pure error mapping, and SSE
  decoding. Its tests run the conformance contracts (`cargo test -p
  allama-examples --example native_adapter`); it needs no credentials, since
  `main` replays canned wire data

Example commands:

```bash
ALLAMA_MODEL=openai:gpt-5-mini cargo run -p allama-examples --example provider_chat -- "Explain Rust ownership"
ALLAMA_MODEL=openai-chat:gpt-5-mini cargo run -p allama-examples --example provider_chat -- "Explain Rust ownership"
ALLAMA_MODEL=anthropic:claude-haiku-4-5 cargo run -p allama-examples --example provider_chat -- "Explain Rust ownership"
ALLAMA_MODEL=gemini:gemini-3.5-flash cargo run -p allama-examples --example provider_stream -- "Explain Rust ownership"
ALLAMA_MODEL=ollama:qwen3:1.7b cargo run -p allama-examples --example chat_repl
ALLAMA_MODEL=openai:gpt-5-mini cargo run -p allama-examples --example tool_loop -- "Weather and local time in Lisbon and Oslo?"
ALLAMA_MODEL=openai:gpt-5-mini cargo run -p allama-examples --example web_backend -- "Do I need an umbrella in Lisbon?"
ALLAMA_MODEL=anthropic:claude-haiku-4-5 cargo run -p allama-examples --example provider_extract -- "Summarize why Rust is good for agents"
ALLAMA_MODEL=gemini:gemini-3.5-flash cargo run -p allama-examples --example history_persistence -- "Summarize this failure"
ALLAMA_EMBEDDING_MODEL=openai:text-embedding-3-small cargo run -p allama-examples --example provider_embedding -- "Hello embeddings" "Second input"
ALLAMA_EMBEDDING_MODEL=gemini:gemini-embedding-001 cargo run -p allama-examples --example provider_embedding -- "Hello embeddings" "Second input"
ALLAMA_EMBEDDING_MODEL=gemini:gemini-embedding-2 cargo run -p allama-examples --example provider_multimodal_embedding -- "something red"
```

## Live tests

The crate also includes an opt-in live integration suite that calls the real
provider APIs.

It runs each selected provider key at its default model (see
`KNOWN_PROVIDERS` in `tests/support.rs`): `openai` (Responses API), `openai-chat`
(Chat Completions), `anthropic`, and `gemini`.

Run one provider:

```bash
ALLAMA_LIVE_PROVIDER=openai cargo test -p allama-examples --test providers -- --nocapture --test-threads=1
```

Run all configured providers:

```bash
ALLAMA_LIVE_PROVIDER=all cargo test -p allama-examples --test providers -- --nocapture --test-threads=1
```

`ALLAMA_LIVE_PROVIDER=configured` is accepted as an alias for `all`.

When using `all` or `configured`, the HTTP suite skips providers whose primary
credential env var is unset, but the overall live gate now fails if the
selection resolves to no runnable targets at all.

When selecting a provider explicitly, missing required env vars are treated as a
hard failure rather than a skip.

The HTTP live test runs basic chat, streaming, a validated tool round-trip, and
structured extraction sequentially for each selected provider.


## Live evidence checks

`tests/live_evidence` records how the real provider APIs behave, as
evidence for later design decisions: Gemini, OpenAI, Azure and
Anthropic schema keywords, Azure's `parallel_tool_calls` on a
reasoning deployment, the Chat Completions custom tool stream, and real
429/529 rate-limit headers. The checks are `#[ignore]`d and run in no CI
job; they assert nothing about the answers, and the rate-limit checks send
bursts until the API throttles them. Each prints what it observed and, with
`ALLAMA_EVIDENCE_DIR` set, writes it there as JSON:

```bash
ALLAMA_EVIDENCE_DIR=target/evidence cargo test -p allama-examples --test live_evidence -- --ignored --nocapture gemini_
```

The filters (`gemini_`, `openai_`, `azure_`, `anthropic_`, `custom_tool_`,
`rate_limit_`) and the environment each needs are listed in the module docs
of `tests/live_evidence/main.rs`. Review the files before committing any.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
