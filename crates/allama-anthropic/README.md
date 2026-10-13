# allama-anthropic

[![crates.io](https://img.shields.io/crates/v/allama-anthropic?style=flat-square)](https://crates.io/crates/allama-anthropic)
[![docs.rs](https://img.shields.io/docsrs/allama-anthropic?style=flat-square)](https://docs.rs/allama-anthropic)

Anthropic (Claude Messages API) provider adapter for `allama`.

## Use it when

- you want Claude models behind the shared `allama::ChatProvider` trait
- you need thinking, tool use, documents, structured outputs, and prompt
  caching with the portable request/response model
- you want Anthropic-specific state (thinking signatures, server tool blocks,
  citations) preserved for replay without leaking it into the portable API

## Quick start

```toml
[dependencies]
allama = "0.1.0"
allama-anthropic = "0.1.0"
anyhow = "1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust,no_run
use allama::prelude::*;
use allama_anthropic::Provider;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = Provider::from_env()?;

    let response = provider
        .chat(&ChatRequest::new("claude-haiku-4-5")
            .user("Summarize why Rust is useful for agents."))
        .await?;

    println!("{}", response.text_or_empty());
    Ok(())
}
```

Environment variables:

- required: `ANTHROPIC_API_KEY`
- optional: `ANTHROPIC_BASE_URL`

## Capabilities

| Capability | Support |
|---|---|
| Tools, parallel tool calls (`parallel_tool_calls(false)` → `disable_parallel_tool_use`), strict tools | yes; a strict tool whose schema strict mode cannot express (maps, recursion) is sent without strict with an `Unsupported` warning (an error under `UnsupportedPolicy::Error`) |
| Streaming (native SSE) | yes |
| Image input (URL, base64) | yes; `ImageDetail` is an unsupported hint |
| File input | `application/pdf` (base64, URL or Files API id) and `text/plain` (base64 or Files API id) as `document` blocks; media type parameters (`; charset=utf-8`) are ignored |
| Structured output (`ResponseFormat::JsonSchema` → `output_config.format`) | `Supported` for Claude Haiku 4.5, Sonnet 4.5+, Opus 4.1+, Fable, Mythos; `Unknown` (sent, the API decides) for other models |
| Reasoning (thinking) output, replay, and configuration | `Supported` for Claude 3.7 Sonnet and Claude 4+; `Unknown` for other models |
| Forced tool choice | `Supported` for the generations before Claude Opus/Sonnet 5.5 and Fable/Mythos 5.1; `Unknown` for the rest |
| Prompt caching | breakpoints on system prompts, messages and tools; request-level automatic caching |
| Image output | no |
| Embeddings | no (Anthropic has no embeddings API; pair with `allama-openai` or `allama-gemini`) |

`chat_capability` answers for the Messages API, and claims support per model
only where it is documented. It never answers `Unsupported`, refuses, drops or
rewrites a request because of the model id (ADR 0015): a setting a model
rejects (sampling on Claude Opus 4.7 and 4.8 and every Claude 5 model, a
forced tool choice on Claude Sonnet 5.5, a fixed thinking budget on an
adaptive-only model, turning thinking off on a model that always thinks) is
sent, and the API's refusal maps to `Error::Unsupported` with stage
`Provider`. To refuse such requests before sending, answer the capability
with `Provider::with_chat_capabilities`; an `Unsupported` answer reaches the
encoder as well as preflight.

Wire formats (budget or adaptive thinking, how thinking is turned off, the
effort levels) are picked from the model id, with the newest format for an id
the adapter does not recognize and for a version of a family it has not
seen. `Provider::with_model_rules` replaces them for a model:
`ModelRules::effort_levels` sets the `output_config.effort` levels.

## Mapping notes

- **Reasoning.** `ReasoningConfig.effort` maps to
  `thinking: {type: "adaptive"}` plus `output_config.effort`, except on the
  legacy generations (Claude 3 including Sonnet 3.7, Claude 4.0 and 4.1,
  Opus 4.5, Sonnet and Haiku 4.5), where it maps to a fixed
  `budget_tokens`. A later version of a family, and an id the adapter does
  not recognize, get the adaptive form with the effort unchanged.
  `ReasoningEffort::None` sends `{type: "disabled"}` on adaptive models
  (`between_tools` on Claude Sonnet 5.5; a model that cannot turn thinking
  off refuses it, as `Error::Unsupported` with stage `Provider`) and omits
  `thinking` on older ones. The Claude 4.6 generation has no `xhigh`: `XHigh` is sent as `max`
  with an `Approximated` warning. `ReasoningOutput` maps to
  `thinking.display` while thinking is on. Without an effort, a
  `ReasoningOutput` never turns thinking on: models that think by default
  (Claude Opus and Sonnet 5 and newer, Haiku 5.5 and newer, Fable, Mythos)
  get `{type: "adaptive", display}`, and on the others it has no effect and is
  an unsupported hint. `ChatRequestOptions::effort` replaces the mapped
  `output_config.effort` level, so `max` is reachable on models that also
  have `xhigh`. A fixed budget can be set with
  `ChatRequestOptions::budget_tokens`; it is sent to any model.
- **Output cap.** `max_output_tokens` maps to `max_tokens`, defaulting to
  4096. It covers thinking and the answer together, and a value you set is
  never raised. A fixed thinking budget must be below it, except with the
  `interleaved-thinking-2025-05-14` beta and tools on a model that
  interleaves manual thinking (Claude 4 through Opus 4.5 and Sonnet 4.6):
  - with `max_output_tokens` unset, the default grows to `budget + 4096`;
  - a budget derived from the effort is lowered to half of
    `max_output_tokens`, but not below the 1024-token minimum, with an
    `Approximated` warning that is never refused. A larger cap never gives
    a smaller budget, and `Minimal` (1024) still thinks under caps
    1025-2047. With `max_output_tokens` at or below 1024 there is no room,
    and thinking is not enabled, as an unsupported hint;
  - a `budget_tokens` option that does not fit fails with
    `Error::InvalidRequest`, since the two settings contradict each other.
- **Tool choice.** `ToolChoice::Disabled` sends `{type: "none"}` and keeps the
  tool definitions. A forced choice (`Required`/`Specific`) with manual
  thinking (`budget_tokens`) fails with `Error::Unsupported` before sending,
  as the API documents for manual thinking. Claude Sonnet 5.5, Opus 5.5,
  Fable 5.1 and Mythos 5.1 reject forced tool choice on every request; the
  request is sent, and the refusal is `Error::Unsupported` with stage
  `Provider`. `extract()` picks native structured output first on those
  models.
- **Tool call ids from other providers.** Anthropic takes only ids matching
  `^[a-zA-Z0-9_-]+$`. An id from another provider that does not match is
  replaced by a stable one derived from it
  (`ToolCallIdFormat::UrlSafe`), the same for the `tool_use` block and its
  `tool_result`, so a conversation continues here after a failover. Valid
  ids are sent unchanged; nothing is lost, so there is no warning.
- **Structured output.** Schemas are rewritten for Anthropic: every object is
  closed (`additionalProperties: false`), unsupported constraints (`minimum`,
  `maxLength`, …) and string formats Anthropic does not list (`regex`, …)
  are removed with an `Approximated` warning, numeric formats (`uint32`,
  `double`) are dropped silently, `$ref` is kept, recursive schemas and
  map-typed objects fail (`output_config.format` has no unconstrained mode).
  Strict tools use the same rewrite, but strict is a hint there: a recursive
  or map-typed tool schema is sent as written (minus `$schema`) without
  strict, with an `Unsupported` warning. A malformed schema (an unresolved
  `$ref`) fails either way. The core `allama::adapter::schema::apply_strict_policy`
  makes these decisions; the pass is public as `schema::strict_schema`, a
  dialect to pass it. Schemaless `ResponseFormat::Json` is unsupported.
- **Strict-schema limits.** Anthropic allows at most 20 strict tools, 24
  optional parameters and 16 union-typed parameters (`Option<T>` fields
  count as both) per request, across the strict tools and the output
  format. Over a limit, strict tools are sent without strict from the last
  one, each with an `Unsupported` warning, until the request fits (an error
  under `UnsupportedPolicy::Error`); an output format that alone exceeds a
  limit fails.
- **Caching.** `CacheBreakpoint` on a system prompt, message (last cacheable
  block) or tool becomes `cache_control`; `CacheRetention::Long` uses the 1h
  TTL. Anthropic allows four breakpoints (automatic caching uses one); extras
  are dropped with a warning. `PromptCache::automatic()` turns on automatic
  caching (top-level `cache_control`, with its retention); nothing else
  does, since Anthropic bills cache writes. A `key`, or a `retention`
  without `automatic`, is an unsupported hint.
- **Replay.** Thinking signatures, redacted thinking, text citations, and
  unknown blocks (server tool use and results, container uploads, …) are kept
  under the `"anthropic"` namespace/origin and replayed verbatim to Anthropic.
  Reasoning or blocks from other providers are dropped with a `Dropped`
  warning. A server tool result stored without its block type takes the
  type Anthropic documents for its call; otherwise the call and the result
  are dropped together with a `dropped.content_replay` warning. Invalid tool-call arguments replay as `{}` with a warning (an error
  under `UnsupportedPolicy::Error`). Empty text is never sent, since the API
  refuses empty text blocks (a unary response drops it on decode, as the
  stream collector does).
- **Turns.** Adjacent same-role messages merge into one turn, as Anthropic
  requires alternating roles. In a user turn the `tool_result` blocks come
  first, as the API requires, so a user message between tool results
  follows them.
- **Hints.** Seed and frequency/presence penalties are dropped with an
  `Unsupported` warning, or rejected under `UnsupportedPolicy::Error`.
  Sampling while thinking is on follows the API's rule for thinking (see
  [Sampling with reasoning](#sampling-with-reasoning)).
- **Responses.** `provider` is `"anthropic"`, `request_id` comes from the
  `request-id` header, `finish_reason_raw` is the raw `stop_reason`
  (`refusal` → `FinishReason::Refusal`, `pause_turn` → `Other`). On
  `pause_turn`, Anthropic paused a long turn of server tools: append
  `response.to_assistant_message()` to the request and call again unchanged
  to let it finish. The matched
  stop sequence, refusal `stop_details`, container and service tier are in
  `provider_data["anthropic"]`, with a refusal's explanation also under
  `decline_explanation` (`allama::provider_data::DECLINE_EXPLANATION_KEY`). Usage follows the `allama` contract
  (`input_tokens` includes cache reads and writes).
- **Errors.** Spend-limit and billing exhaustion map to
  `Error::QuotaExhausted`; mid-stream `error` events map to the same typed
  errors as HTTP failures. Anthropic's documented refusals of a request field
  (a sampling parameter the model no longer takes, sampling under thinking,
  `temperature` with `top_p`, a forced tool choice, a thinking configuration
  the model does not take, a schema too complex to compile) map to
  `Error::Unsupported` with stage `Provider` and the field as its subject.

## Sampling with reasoning

While thinking is on, manual or adaptive, a `temperature` other than 1,
`top_k`, and a `top_p` outside 0.95–1 are dropped with an `Unsupported`
warning (an error under `UnsupportedPolicy::Error`), as the API documents
for thinking on every model. A forced tool choice with manual thinking
(`budget_tokens`) fails with `Error::Unsupported` before sending.

Rules of particular models are not applied (ADR 0015): Claude Opus 4.7 and
4.8 and every Claude 5 model reject `temperature`, `top_p` and `top_k`
outright, and Claude Opus 4.1 and Haiku 4.5 take `temperature` or `top_p`,
not both. Such a request is sent, and the refusal is `Error::Unsupported`
with stage `Provider`, naming the field, so a `FallbackProvider` fails over
and a caller can resend without it. Leave sampling unset for those models.
A model that rejects a forced tool choice can be refused before sending:
answer `ChatCapability::ForcedToolChoice` with
`Provider::with_chat_capabilities`.

## Features

Structured-output extraction (`allama::ExtractExt`) is `allama`'s
`extract` feature: enable it on your `allama` dependency.

- `reqwest` (default): the default transport. With default features off,
  build the provider with `http_transport(...)` (any `allama_http::HttpTransport`);
  the crate then builds without `reqwest`, and the constructors that would
  build a `reqwest` client (`new`, `from_env`) are not compiled.
- `default-tls` (default), `rustls`, `native-tls`, `rustls-ring`: the TLS
  backend of the default `reqwest` transport (no effect on `wasm32`, where
  the host's `fetch` does TLS); each implies `reqwest`. `rustls-ring` uses
  `ring` instead of `aws-lc-rs`, so static musl builds need no cmake. With
  `reqwest` on and default features off, pick one or HTTPS endpoints cannot
  be reached.
- `http-tracing`: implies `reqwest` and turns on `allama-http`'s
  `http-tracing`, which wraps every client a `ReqwestTransport` builds (or
  gets through `ReqwestTransport::from_client(..)`) with `reqwest-tracing`,
  so HTTP spans fire on outgoing requests. Every adapter shares one
  `allama-http`, so enabling it on any provider crate traces the HTTP
  requests of every provider in the build, not only this one's.

To send through your own `reqwest-middleware` stack, enable `allama-http`'s
`reqwest-middleware` feature and pass
`http_transport(ReqwestTransport::from_middleware(..))`.

## Provider options

```rust
use allama::ChatRequest;
use allama_anthropic::ChatRequestOptions;

let request = ChatRequest::new("claude-haiku-4-5")
    .user("Hello")
    .top_k(20)
    .with_option(
        ChatRequestOptions::new()
            .budget_tokens(2048)
            .beta("some-beta-2026-01-01"),
    );
```

Sampling controls are portable request fields (`temperature`, `top_p`,
`top_k`), not options. The same options can be given as JSON under the
`"anthropic"` namespace
(`request.with_options_json("anthropic", json!({"budget_tokens": 2048}))`).
`ChatRequestOptions` is `#[non_exhaustive]`: build it with `new()` and the
setters. Messages API fields this crate does not model (`container`,
`mcp_servers`, `context_management`, …) go in `extra`
(`ChatRequestOptions::new().extra("container", id)`, or `{"extra": {…}}` in
the JSON form), merged into the request body; a field the adapter owns
(`model`, `messages`, `tools`, `thinking`, …) is refused with
`Error::InvalidRequest`. Extra tool fields (for example
`eager_input_streaming`) go in `ToolDefinition.provider_data["anthropic"]`,
and extra document fields (`citations`, `context`) in
`File.provider_data["anthropic"]`; a field the adapter sets from the
portable value (`name`, `input_schema`, `source`, `title`, …) is refused
there too.

Files uploaded with the Files API are referenced by id, on images and
documents alike: `Image::provider_file("anthropic", id)` and
`File::provider_file("application/pdf", "anthropic", id)`. An id another
provider issued is refused before sending.

Document citations are turned on per document:

```rust
use allama::{ContentPart, File, ProviderData};
use serde_json::json;

let report = File::base64("application/pdf", "JVBERi0x…")
    .with_filename("q3.pdf")
    .with_provider_data(ProviderData::new().with_raw(
        "anthropic",
        "citations",
        json!({"enabled": true}),
    ));
let parts = vec![ContentPart::File(report), ContentPart::text("Summarize, citing pages.")];
```

The answer's text blocks then carry `Citation`s with a
`CitationSource::Document` source.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
