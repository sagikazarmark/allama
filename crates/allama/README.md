# allama

[![crates.io](https://img.shields.io/crates/v/allama?style=flat-square)](https://crates.io/crates/allama)
[![docs.rs](https://img.shields.io/docsrs/allama?style=flat-square)](https://docs.rs/allama)

**Low-level, provider-agnostic LLM abstractions for Rust.**

`allama` is a low-level crate for code that wants one interface for LLM APIs
(chat, embeddings, and decision models) while leaving HTTP transport, request translation,
and response parsing to provider implementations.

Use it when you want to:

- write application code against one provider-neutral contract
- implement a provider adapter on top of shared request and response types
- normalize streaming, tool calls, usage, and errors across providers
- move conversations between providers without leaking opaque state
- produce embedding vectors from multiple providers with a single trait
- ask decision models typed questions (yes/no, choice, score) and read
  probabilities, untyped or through `#[derive(Decision)]`

`allama` is intentionally not a provider client and not an agent framework.

## You May Be Looking For

- API docs: <https://docs.rs/allama>
- `ChatProvider`: <https://docs.rs/allama/latest/allama/trait.ChatProvider.html>
- `EmbeddingProvider`: <https://docs.rs/allama/latest/allama/trait.EmbeddingProvider.html>
- `DecisionProvider`: <https://docs.rs/allama/latest/allama/trait.DecisionProvider.html>
- chat example: [`chat.rs`](examples/chat.rs)
- embedding example: [`embedding.rs`](examples/embedding.rs)
- decision example: [`decision.rs`](examples/decision.rs)
- streaming example: [`stream.rs`](examples/stream.rs)
- provider implementation example: [`provider_impl.rs`](examples/provider_impl.rs)

## Quick start

Pick the provider and model at runtime from one string, a model address
such as `openai:gpt-5.5`, with
[`allama-providers`](https://crates.io/crates/allama-providers):

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

Streaming, conversations, and what to settle before production (the
120 s `Retry-After` ceiling of the default retries among them) are in the
[workspace README](../../README.md#quick-start).
The terms the docs use are defined in the
[glossary](../../CONTEXT.md).

## Without credentials

This example uses the built-in mock provider, so it runs without credentials.

```toml
[dependencies]
allama = { version = "0.1", features = ["mock"] }
anyhow = "1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

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

From the workspace root:

```bash
cargo run -p allama --example chat --features mock
```

`MockProvider` replays a script and returns an error once the script runs
out. For tests that send many requests, such as HTTP handler tests, use
`TestProvider`, which answers every request. The
[`allama::mock`](https://docs.rs/allama/latest/allama/mock/) docs explain
which mock to pick and when scripted errors surface.

## Core concepts

| Concept | Types |
| --- | --- |
| Requests | `ChatRequest`, `Message`, `ContentPart` (`Text`, `Image`, `File`), `MediaSource` (`Url`, `Base64`, `ProviderFile`), `SystemPrompt`, `ToolDefinition`, `ToolChoice`, `ResponseFormat`, `ReasoningConfig`, `PromptCache` |
| Responses | `ChatResponse`, `ContentBlock` (`Text`, `Image`, `ToolCall`, `Reasoning`, `Refusal`, `ProviderToolCall`, `ProviderToolResult`, `Other`), `FinishReason`, `Usage`, `Warning` |
| Decisions | `DecisionRequest`, `DecisionResponse`, `DecisionLimits`; in `decision`: `DecisionInput`, `Question` (`yes_no`, `choice`, `score`), `Answer`; typed: `Decision`, `ChoiceSet`, `Scale`, `Choice`, `Score`, `DecisionProviderExt::decide_as` |
| Streaming | `StreamEvent`, `ChatStreamExt::{next_event, collect_response, with_collector}`, `CollectedEvent`; in `chat::stream`: `StreamCollector`, `StreamTranscript`, `until_committed` |
| Provider escape hatches | `ProviderOptions` / `RequestOptions` (typed or JSON request options), `ProviderData` (namespaced provider state on blocks, messages, responses) |
| Policy | `UnsupportedPolicy` (warn or fail on unsupported hints) |
| Provider authoring | `adapter::{chat, embedding, decision, schema}` (`prepare`, `preflight`, `WireSupport`, the JSON Schema passes), `adapter::Degradations` |
| Errors | `Error` (`RateLimited`, `QuotaExhausted`, `ContextLengthExceeded`, `Stream { kind }`, …) with `retry_after`, `request_id`, `is_retryable` |
| Composition | `wrappers::{RetryingProvider, FallbackProvider, TimeoutProvider, MapModelProvider, MapRequestProvider, RecordingProvider, TracingProvider}`, `extract::ExtractExt`, typed lookup through wrappers (`ProviderIdentityExt::downcast_ref`) |
| Persistence | `history::History` (versioned envelope), `history::retain_portable_content` for client-held history; requests and responses are `Serialize`/`Deserialize` |

## Features

The only default feature is `warning-events`. A library that depends on
`allama` turns default features off, so the application alone decides.

| Feature | What it enables |
| --- | --- |
| `warning-events` (default) | every warning a call reports is also a `tracing` event (targets `allama::warning::<kind>`, deduplicated by code), and retries and failovers are `allama::retry` and `allama::fallback` events |
| `schema` | `ToolDefinition::from_schema::<T>()`, `ResponseFormat::from_schema::<T>()` and `allama::adapter::schema::schema_for` via `schemars`, re-exported as `allama::schemars`; field paths in `ToolArgumentsError` via `serde_path_to_error` |
| `derive` | `#[derive(Decision, ChoiceSet, Scale)]` for typed decisions (`allama-derive`) |
| `extract` | structured extraction (`ExtractExt`, `Extractor`); implies `schema` |
| `mock` | test doubles: `MockProvider`, `MockStreamingProvider`, `MockEmbeddingProvider`, `MockDecisionProvider`, `FunctionProvider`, `TestProvider` |
| `tracing` | OpenTelemetry GenAI spans via `wrappers::TracingProvider` |

## Implementing A Provider

Implement `ProviderIdentity`, then any of `ChatProvider`,
`EmbeddingProvider`, and `DecisionProvider`. A provider:

- reads its options with `request.resolve_option::<YourOptions>()`,
- reports unsupported fields through `Degradations` (semantic fields fail,
  hints warn or fail under `UnsupportedPolicy::Error`),
- stores replay state in `ProviderData` under its own namespace and replays
  only that namespace, which `ProviderIdentity::reads_namespace` answers
  `true` for,
- names itself with `ProviderIdentity::provider_name` and answers
  `chat_capability` / `embedding_capability` / `decision_capability`
  (`CapabilitySupport::Unknown` when it does not know); both are required,
- exposes itself through `ProviderIdentity::downcast_any` (`expose`), so
  callers can reach it behind wrappers with `downcast_ref`,
- normalizes usage to the contract in `Usage` (input includes cache, output
  includes reasoning) and stamps `ChatResponse::provider`.

A wrapper is transparent: it forwards the name, the namespaces it reads,
`downcast_any`, capability answers, limits, and validation to the provider
it wraps, and passes its responses, stream events and errors through
unchanged. The `allama::wrappers::forward_*` macros forward every
question, so a wrapper writes only its operations, and
`allama_conformance::assert_wrapper` checks it. Middleware (an
async request hook, response and stream observation) is a wrapper; the
[`middleware` example](../allama-examples/examples/middleware.rs)
is the recipe.

Providers without native streaming can return `SingleResponseStream`.
`allama-conformance` provides fixture helpers and shared behavioral
contracts. The annotated guide is
[`provider_impl.rs`](examples/provider_impl.rs).

## Status

- Pre-1.0: a minor release (0.x to 0.x+1) may break the API; a patch release does not.
- Every `allama-*` crate shares one version and is released with the others.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
