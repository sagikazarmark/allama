# allama

[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/sagikazarmark/allama/ci.yaml?style=flat-square)](https://github.com/sagikazarmark/allama/actions/workflows/ci.yaml)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/sagikazarmark/allama/badge?style=flat-square)](https://securityscorecards.dev/viewer/?uri=github.com/sagikazarmark/allama)
[![crates.io](https://img.shields.io/crates/v/allama?style=flat-square)](https://crates.io/crates/allama)
[![docs.rs](https://img.shields.io/docsrs/allama?style=flat-square)](https://docs.rs/allama)

**Low-level, provider-agnostic LLM abstractions for Rust.**

## Features

- 🔌 **One API, many providers.** OpenAI (Responses and Chat Completions),
  Anthropic, Gemini, a dozen OpenAI-compatible endpoints (Groq, OpenRouter,
  Ollama, vLLM, Azure, …), Cloudflare Workers AI, and System One decision
  models.
- 🧭 **Portable where it is real.** Messages, tools, streaming, structured
  output, reasoning effort, prompt caching, files, usage and errors share one
  model. Everything else is a typed provider option or namespaced provider
  data, never a field only one provider understands.
- 🔍 **Nothing dropped silently.** A setting a provider cannot carry fails,
  or comes back as a coded `Warning` on the response;
  `UnsupportedPolicy::Error` turns those warnings into errors.
- 🔁 **Conversations move between providers.** Opaque state such as thinking
  signatures and encrypted reasoning records which provider produced it, so
  a conversation continues on another provider, for example after a failover.
- 🧱 **Composable wrappers.** Retries with a budget, fallback, timeouts,
  model mapping and OpenTelemetry tracing are ordinary providers you stack.
- 🧾 **Typed extraction.** `extract::<T>()` picks native structured output or
  a forced tool call, and reports how strictly the provider enforced the
  schema.
- 🧮 **Beyond chat.** Embeddings (text and multimodal) and decision models
  (typed answers with probabilities) are sibling traits with the same design.
- 🗺️ **Model addresses.** `openai:gpt-5.5` picks the provider and the model
  from one configuration string, with a grammar that stays put.
- 🌐 **Runtime-agnostic.** The core depends on no async runtime, builds for
  wasm32, and runs inside a Cloudflare Worker.
- 🧩 **A building block.** No agent loop: tools are declared and parsed, and
  your code runs them.
- ✅ **Conformance-tested.** Every adapter, third-party ones included, runs
  the same contracts from `allama-conformance`.
- 🧪 **Testable.** Mock providers, and a replay transport that runs real
  adapters over recorded wire data.

| Provider | Crate | Model address key | Chat | Embeddings | Decisions |
| --- | --- | --- | :---: | :---: | :---: |
| OpenAI Responses API | [`allama-openai`](crates/allama-openai) | `openai`, `openai-responses` | ✓ | | |
| OpenAI Chat Completions | [`allama-openai`](crates/allama-openai) | `openai-chat` (chat), `openai` (embeddings) | ✓ | ✓ | |
| Anthropic | [`allama-anthropic`](crates/allama-anthropic) | `anthropic` | ✓ | | |
| Gemini | [`allama-gemini`](crates/allama-gemini) | `gemini` | ✓ | ✓ | |
| OpenAI-compatible endpoints | [`allama-openai-compat`](crates/allama-openai-compat) | `azure`, `groq`, `ollama`, `openrouter`, … | ✓ | ✓ | |
| Azure OpenAI Responses API | [`allama-openai-compat`](crates/allama-openai-compat) | `azure-responses` | ✓ | | |
| Cloudflare Workers AI (inside a Worker) | [`allama-cloudflare-worker`](crates/allama-cloudflare-worker) | | ✓ | ✓ | |
| System One (TypeSafe, Ollama, gateways) | [`allama-systemone`](crates/allama-systemone) | `typesafe`, `ollama`, … | | | ✓ |

`openai:` is the Responses API, the default for chat and the right choice for
reasoning models; `openai-chat:` is Chat Completions, for when you need that
wire format. `openai-responses:` is the same as `openai:`, so Pydantic AI
model strings work unchanged. On Azure, `azure:` is Chat Completions and
`azure-responses:` the Responses API.

## Quick start

Pick the provider and model at runtime from one string, a **model address**
such as `openai:gpt-5.5`, with [`allama-providers`](crates/allama-providers):

```toml
[dependencies]
allama = "0.1"
allama-providers = "0.1"
anyhow = "1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust,no_run
use allama::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Reads OPENAI_API_KEY. Try "anthropic:claude-haiku-4-5",
    // "gemini:gemini-3.5-flash" or "ollama:qwen3:1.7b".
    let (provider, model) = allama_providers::chat::from_model("openai:gpt-5.5")?;

    let request = ChatRequest::new(model).user("Explain Rust ownership in two sentences.");
    let response = provider.chat(&request).await?;
    println!("{}", response.text_or_empty());
    Ok(())
}
```

Each provider reads its own key (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`; a
local Ollama needs none). The examples' README keeps a
[default model for each provider](crates/allama-examples/README.md#quick-start).

To print the answer as it is generated, stream it. The collector builds the
same `ChatResponse` that `chat` returns:

```rust
use allama::prelude::*;

async fn stream(provider: &DynChatProvider, request: &ChatRequest) -> Result<ChatResponse> {
    let mut stream = provider.chat_stream(request).await?.with_collector();
    while let Some(item) = stream.next_event().await {
        if let StreamEvent::TextDelta { text, .. } = item?.event {
            print!("{text}");
        }
    }
    stream.finish()
}
```

### Tools from Rust types

With the `schema` feature, a tool's parameter schema comes from a Rust type:

```toml
[dependencies]
allama = { version = "0.1", features = ["schema"] }
allama-providers = "0.1"
anyhow = "1"
serde = { version = "1", features = ["derive"] }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust,no_run
use allama::prelude::*;
use allama::schemars::{self, JsonSchema};
use serde::Deserialize;

/// Arguments of the `get_weather` tool; the schema is derived from the type.
#[derive(Deserialize, JsonSchema)]
struct WeatherArgs {
    /// City name, such as "Lisbon".
    city: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let (provider, model) = allama_providers::chat::from_model("openai:gpt-5.5")?;

    let request = ChatRequest::new(model)
        .user("Do I need an umbrella in Lisbon tomorrow?")
        .tools([ToolDefinition::from_schema::<WeatherArgs>("get_weather")
            .description("Look up tomorrow's weather forecast for a city.")]);

    let response = provider.chat(&request).await?;
    for call in response.tool_calls() {
        let args: WeatherArgs = call.parse_arguments()?;
        println!("the model asks for the weather in {}", args.city);
    }
    Ok(())
}
```

Your code runs the tool and sends the result back with
`ChatRequest::push_tool_result`.
[`tool_loop`](crates/allama-examples/examples/tool_loop.rs) keeps calling
tools until the model answers, and
[`web_backend`](crates/allama-examples/examples/web_backend.rs) handles a
whole turn the way a server does.

### Before production

- **Retries wait up to 120 s.** Providers from `allama_providers` retry
  transient errors up to 3 attempts and wait out a `Retry-After` of up to
  120 s. When someone is waiting, lower it with
  `RetryPolicy::new().max_retry_after(..)` in `ProviderSettings::retry`.
- **Check how the answer ended.** `is_refusal()` and `finished_normally()`
  tell a complete answer from a refused, filtered or cut-off one; see
  [when a turn does not finish normally](docs/guides/applications.md#when-a-turn-does-not-finish-normally).
- **Reasoning counts against `max_output_tokens`.** A reasoning model under a
  small cap can spend all of it reasoning and return an empty answer.
- **Read warnings in one place**: `response.warnings`, or the
  `StreamEvent::Warning` events while streaming, not both; see
  [warnings](docs/guides/applications.md#warnings).
- **Settings only some models reject are sent**, and the provider decides;
  see [sampling with reasoning](docs/guides/applications.md#sampling-with-reasoning).
- **Check credentials at startup** with a
  [smoke call](docs/guides/applications.md#a-smoke-call-for-credentials-and-the-model-id).
- **Bound the wait**: a server sets a deadline on the whole call, and
  usually a fallback; see
  [the provider stack](docs/guides/applications.md#the-provider-stack).

### Without credentials

The `mock` feature has providers that run anywhere, for examples and tests:

```rust
use allama::mock::MockProvider;
use allama::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = MockProvider::build(|b| b.text("Deterministic hello from allama."));
    let response = provider
        .chat(&ChatRequest::new("demo-model").user("Say hello"))
        .await?;

    println!("chat text: {}", response.text_or_empty());
    Ok(())
}
```

### Examples

Each runs against a local [Ollama](https://ollama.com) with no key
(`ollama pull qwen3:1.7b`); any other model address works too:

```sh
ALLAMA_MODEL=ollama:qwen3:1.7b cargo run -p allama-examples --example chat_repl         # a terminal chat with tools
ALLAMA_MODEL=ollama:qwen3:1.7b cargo run -p allama-examples --example axum_sse          # an SSE endpoint
ALLAMA_MODEL=ollama:qwen3:1.7b cargo run -p allama-examples --example provider_extract  # typed extraction
ALLAMA_EMBEDDING_MODEL=ollama:nomic-embed-text cargo run -p allama-examples --example provider_embedding
cargo run -p allama-examples --example provider_decision  # Ollama 0.35+ after `ollama pull nimble`
cargo run -p allama-examples --example middleware         # a custom wrapper, no provider needed
cargo run -p allama --example otel --features mock,tracing
```

The [`allama-examples` README](crates/allama-examples/README.md) lists every
example.

## Cargo features

Every feature but `warning-events` is off by default.

| Feature | Enables |
| --- | --- |
| `warning-events` (default) | Every warning is also a `tracing` event under `allama::warning`, deduplicated by code |
| `schema` | `ToolDefinition::from_schema`, `ResponseFormat::from_schema`, `allama::schemars` |
| `extract` | `allama::extract` (`ExtractExt::extract`); implies `schema` |
| `derive` | `#[derive(Decision, ChoiceSet, Scale)]` for typed decisions |
| `mock` | `allama::mock`: mock providers and other test doubles |
| `tracing` | `wrappers::TracingProvider`: OpenTelemetry GenAI spans |

Provider crates add transport features of their own (TLS backends,
`http-tracing`); see their READMEs.

## Documentation

- [Applications guide](docs/guides/applications.md): servers and terminal
  apps, storing conversations, streaming over SSE, the provider stack,
  failures after the first token, testing with mocks.
- [Observability guide](docs/guides/observability/README.md): OpenTelemetry
  spans, content capture, trace context propagation, GenAI metrics.
- [Provider author guide](docs/guides/provider-authors.md): compat presets,
  quirks or a native adapter, and the conformance contracts.
- [Design decisions](docs/README.md#decisions), the
  [glossary](CONTEXT.md) and the [API reference](https://docs.rs/allama).

## Project status

- **MSRV: Rust 1.92**, checked in CI. Raising it is a minor release.
- **One version.** Every crate shares one version and is released together.
- **Public dependencies.** `allama` exposes `serde`, `serde_json` and
  `futures-core`, plus `schemars` 1.x with `schema` and `tracing` 0.1 with
  `tracing`. A new major version of any of them is a breaking release.
- **Maintained by one person** ([@sagikazarmark](https://github.com/sagikazarmark)).
  Open an issue before large contributions.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
