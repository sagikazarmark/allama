# allama-openai

[![crates.io](https://img.shields.io/crates/v/allama-openai?style=flat-square)](https://crates.io/crates/allama-openai)
[![docs.rs](https://img.shields.io/docsrs/allama-openai?style=flat-square)](https://docs.rs/allama-openai)

OpenAI provider adapter for `allama`.

## Use it when

- you want to call OpenAI models through the shared `allama::ChatProvider` API
- you want tools, streaming, and model-dependent OpenAI features like vision or native structured outputs
- you may later swap providers without rewriting application logic

## Capabilities

`allama-openai` is built on the `allama-openai-compat` toolkit with the
OpenAI dialect (`CompatQuirks::openai()`). It reports:

- tools, including parallel tool calls and strict schemas
- streaming, with usage on the final chunk
- image and file (PDF) input with per-image detail, and native structured
  output, for the gpt-4o, gpt-4.1, gpt-5 and o-series families
- reasoning effort for the gpt-5, GPT-6 and o-series reasoning models
- prompt caching via `PromptCache::key` (`prompt_cache_key`)
- embeddings via `/v1/embeddings`, with output `dimensions` for
  `text-embedding-3-*` (sent for any other model, for OpenAI to decide) and
  at most 2048 inputs per call: a larger batch fails before it is sent, so
  split it with `EmbeddingProviderExt::embed_chunked`

`max_output_tokens` is sent as `max_completion_tokens`, which reasoning models
require.

OpenAI's Chat Completions API does not return reasoning content; only
`reasoning_tokens` appears in usage. Reasoning output and replay are reported
as `Unsupported`.

Answers that depend on the model are positive or `Unknown`: nothing is
refused, dropped or clamped because of the model id (ADR 0015). A model that
rejects what the request asks for (sampling while it reasons, tools on Chat
Completions, an endpoint that does not serve it) gets OpenAI's rejection, an
`Error::Unsupported` with stage `Provider` that names the field. A content
part the model does not take (audio sent to a Chat Completions model
without audio input) is the same refusal, with no subject: OpenAI's
`messages[N]` counts the system prompt, so it does not name the caller's
message. To drop such fields before sending instead, install the model's rules with
`Provider::with_model_rules` (`ChatModelRules`) or
`ResponsesProvider::with_model_rules` (`ResponsesModelRules`).

## Sampling with reasoning

Nothing is dropped because a model reasons (ADR 0015). Sampling
(`temperature`, `top_p`, penalties, `logprobs`) and tools are sent as asked,
and a model that rejects them while it reasons (GPT-5, the o-series, GPT-5.1
and newer at an effort other than `none`, GPT-6 tools on Chat Completions)
answers with an `Error::Unsupported` whose stage is `Provider` and whose
subject names the field. GPT-6's refusal of function tools with an effort
on Chat Completions names `RequestField::Reasoning`, since its remedy there
is the effort `none`. To drop them before sending instead, give the
provider the model's rules with `with_model_rules`:
`SamplingSupport::WithoutReasoning` drops sampling while the model reasons,
`ToolCallSupport::WithoutReasoning` (Chat Completions) refuses tools while
it reasons, and `default_effort` says whether a request that sets no effort
reasons.

## Quick start

```toml
[dependencies]
allama = "0.1.0"
allama-openai = "0.1.0"
anyhow = "1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust,no_run
use allama::prelude::*;
use allama_openai::Provider;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = Provider::from_env()?;

    let response = provider
        .chat(
            &ChatRequest::new("gpt-5-mini")
                .user("Say hello in one sentence.")
                .reasoning(ReasoningEffort::Minimal),
        )
        .await?;

    println!("{}", response.text_or_empty());
    Ok(())
}
```

Environment variables:

- required: `OPENAI_API_KEY`
- optional: `OPENAI_BASE_URL`, `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`

## Provider options

OpenAI-only request fields go in `ChatRequestOptions` (namespace `"openai"`),
given as the typed struct or as JSON:

```rust
use allama::ChatRequest;
use allama_openai::{ChatRequestOptions, SystemOptions};
use serde_json::json;

let typed = ChatRequest::new("gpt-5-mini")
    .system(allama::SystemPrompt::new("Follow the style guide.").with_option(SystemOptions::developer()))
    .user("hi")
    .with_option(ChatRequestOptions::new().service_tier("flex"));

let from_config = ChatRequest::new("gpt-5-mini")
    .user("hi")
    .with_options_json("openai", json!({"service_tier": "flex", "store": false}));
```

Unknown option fields fail with `Error::InvalidRequest`. Portable controls
(`seed`, `parallel_tool_calls`, reasoning effort) belong on `ChatRequest`.
The option types are `#[non_exhaustive]`: build them with `new()` and the
setters. Body fields this crate does not model go in `extra`
(`ChatRequestOptions::new().extra("verbosity", "low")`, or `{"extra": {…}}`
in the JSON form); a field the adapter owns (`model`, `messages`, `stream`,
`tools`, …) is refused with `Error::InvalidRequest`. `EmbeddingRequestOptions`
has the same `extra`.

`client_tools` declares custom tools (freeform text input with an optional
grammar) in Chat's shape, `{"type": "custom", "custom": {"name": …,
"format": …}}`. Their calls come back as `OtherContent` in the same shape
the Responses API uses, so `ClientToolCall::pending` reads them on both APIs
and a conversation with them moves between `Provider` and
`ResponsesProvider` (ADR 0014). The answer, built with
`CustomToolCall::output`, goes in a user message after the turn; Chat
Completions sends it as a `tool` message. Responses-only client tool items
(shell, computer use) are dropped with warnings, with their answers.

## Response details

- `ChatResponse::provider` is `"openai"`; `request_id` comes from
  `x-request-id`.
- `service_tier` and `system_fingerprint` are in `provider_data` under
  `"openai"`.
- `insufficient_quota` maps to `Error::QuotaExhausted` (not retried);
  `retry-after-ms` / `retry-after` are honored on rate limits.

## Responses API

`ResponsesProvider` speaks OpenAI's Responses API (`POST /v1/responses`),
**statelessly by default**: a request sends `store: false` and the whole
conversation, and the reasoning state travels in the conversation. Use it
for reasoning models when you want reasoning summaries, or multi-turn tool
use where the model keeps its reasoning between calls.

Server-side state is opt-in, per request, through `ResponsesRequestOptions`:
`store(true)` keeps the response on OpenAI's side, `previous_response_id`
continues a stored response (the request's messages then carry only the new
turn), `conversation` continues a stored conversation, and `background(true)`
(which needs `store`) queues the response and returns at once; poll it with
`ResponsesProvider::retrieve_response`. Continuation stays a provider
option because no other API family allama implements keeps server-side
state.

It is built on `allama_openai_compat::ResponsesProvider` with the OpenAI
dialect and OpenAI's per-model rules. For other Responses-compatible
endpoints (Ollama, vLLM, OpenRouter, …) use the compat provider with the
endpoint's own identity and provider-data namespace, so their reasoning state
is never mixed with OpenAI's.

```rust,no_run
use allama::prelude::*;
use allama::{ReasoningConfig, ReasoningOutput};
use allama_openai::ResponsesProvider;
use serde_json::json;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = ResponsesProvider::from_env()?; // same variables as `Provider`
    let weather = ToolDefinition::new(
        "get_weather",
        json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
    );
    let mut request = ChatRequest::new("gpt-5-mini")
        .tools([weather])
        .reasoning(
            ReasoningConfig::new()
                .effort(ReasoningEffort::Low)
                .output(ReasoningOutput::Summary),
        )
        .user("What is the weather in Paris?");

    let first = provider.chat(&request).await?;
    println!("reasoning: {}", first.reasoning_text().unwrap_or_default());
    // The assistant turn keeps the encrypted reasoning item.
    request.push_message(first.to_assistant_message());
    for call in first.tool_calls() {
        request.push_tool_result(call, "Sunny, 18 degrees Celsius.");
    }
    // The reasoning item is replayed statelessly.
    let second = provider.chat(&request).await?;
    println!("{}", second.text_or_empty());
    Ok(())
}
```

What maps where:

- Reasoning items become `Reasoning` blocks with the summary as text;
  `encrypted_content` and the item id live in `provider_data["openai"]` and
  are replayed in later turns (they survive `allama::history::History`).
  With `store: false` (the default) OpenAI returns encrypted reasoning.
- `ReasoningOutput::Summary` → `reasoning.summary: "auto"`; `Full` →
  `"detailed"` with an `Approximated` warning (OpenAI never returns raw
  reasoning); `Omitted` → no summary. Effort maps to the nearest level a
  model documents (`gpt-5-nano` has no `none`/`xhigh`), with a warning, and
  is sent unchanged for any other model; `with_model_rules` replaces the
  levels.
- Function calls become `ToolCall`s whose id is the `call_id`; the item id is
  kept for replay. Hosted-tool items (web search, file search, …) become
  `ProviderToolCall` and `ProviderToolResult` blocks with origin `"openai"`
  and are replayed as the original items; enable hosted tools with
  `ResponsesRequestOptions::hosted_tools`.
- Client-executed tools that are not function tools (custom tools with
  freeform input, the shell tools, computer use) are declared with
  `ResponsesRequestOptions::client_tools`. Their calls stay `OtherContent`
  with origin `"openai"`, so `tool_calls()` never returns them, and the
  finish reason is `ToolCalls` while one waits.
  `ClientToolCall::pending` reads them as typed views, and each view builds
  its answer for a user message (ADR 0014). Custom tool calls share their
  shape with Chat Completions. An answer whose call is not in the request
  is sent, for `previous_response_id` continuation; one whose call is in
  the request but cannot be replayed is dropped with it.
- System prompts become `system` input messages, or `developer` with
  `SystemOptions::developer()`.
- `ToolDefinition.strict` is always sent: the Responses API would otherwise make every
  tool strict and rewrite optional parameters as required.
- `temperature`, `top_p`, and penalties are sent as asked, whatever the
  model. `seed` and `stop` do not exist in the Responses API: they are
  dropped with a warning, or fail under `UnsupportedPolicy::Error`. Audio
  input is not accepted: `ChatCapability::AudioInput` answers
  `Unsupported`, and an audio part fails with `Error::Unsupported` before
  sending; send audio through Chat Completions (`Provider`).
- Reasoning and opaque state from other providers is dropped with `Dropped`
  warnings. Conversations can move between `Provider` and
  `ResponsesProvider`: both use the `"openai"` provider-data namespace, and
  Chat Completions never produces or replays reasoning items.
- `finish_reason_raw` is the response `status`, or the
  `incomplete_details.reason` (`max_output_tokens` → `Length`,
  `content_filter` → `ContentFilter`); refusals → `Refusal`.

Responses-only request fields go in `ResponsesRequestOptions` (namespace
`"openai-responses"`, distinct from Chat Completions' `"openai"` so one
request can carry both): `service_tier`, `metadata`, `safety_identifier`,
`user`, `truncation`, `verbosity`, a raw `reasoning_effort`, a raw
`tool_choice` (to force a hosted or custom tool, such as `{"type":
"web_search"}`, which the portable `ToolChoice` cannot name), `hosted_tools`,
`client_tools`, `store`, `previous_response_id`, `conversation`,
`background`, `raw_response`, and `extra` (body fields such as
`top_logprobs`; fields the adapter owns or types, like `input` or
`previous_response_id`, are refused). Token log-probabilities requested
with `include: ["message.output_text.logprobs"]` come back as provider data
on each text block, under the `logprobs` key of the `"openai"`
provider-data namespace.

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

## Testing

```sh
cargo test -p allama-openai                                   # fixtures and local HTTP server
cargo test -p allama-openai --test e2e -- --ignored            # live Chat Completions; needs OPENAI_API_KEY
cargo test -p allama-openai --test e2e_responses -- --ignored  # live Responses API
```

Recorded API responses live in `fixtures/recorded/` (Chat Completions) and
`fixtures/responses/` (Responses; record more with
`scripts/capture-fixture.sh openai-responses`); regenerate expected fixtures
with `ALLAMA_UPDATE_FIXTURES=1` and review the diff.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
