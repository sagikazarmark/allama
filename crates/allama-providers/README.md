# allama-providers

[![crates.io](https://img.shields.io/crates/v/allama-providers?style=flat-square)](https://crates.io/crates/allama-providers)
[![docs.rs](https://img.shields.io/docsrs/allama-providers?style=flat-square)](https://docs.rs/allama-providers)

Build `allama` providers from a provider key (`openai`) or a model address
(`openai:gpt-5.5`).

allama is not affiliated with [Ollama](https://ollama.com); Ollama is one of
the backends it supports.

## Use it when

- the provider is chosen at runtime, from configuration or a command-line flag
- one string should select both the provider and the model
- you want built-in providers behind feature flags, configured from their
  usual environment variables

Depend on a provider crate directly when the provider is fixed at compile
time: you keep its concrete type and its provider-specific methods.

## Quick start

```toml
[dependencies]
allama = "0.1.0"
allama-providers = "0.1.0"
anyhow = "1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust,no_run
use allama::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let (provider, model) = allama_providers::chat::from_model("anthropic:claude-haiku-4-5")?;

    let response = provider
        .chat(&ChatRequest::new(model).user("Say hello."))
        .await?;

    println!("{}", response.text_or_empty());
    Ok(())
}
```

`chat::from_provider("anthropic")` returns the provider alone, for choosing
the model per request. The `embedding` and `decision` modules have the same
pair for embeddings and decision models.

## Retries by default

Every provider comes back wrapped in `allama`'s `RetryingProvider` with
the default `RetryPolicy`: up to 3 attempts, backoff from 250 ms to 5 s
with jitter, the server's `x-should-retry` followed, a retry budget per
provider, and a provider's `Retry-After` waited out **up to 120 s** (a
longer one gives up at once). That suits batch work: a unary call can take
about 10 minutes before its error reaches you, and nothing bounds the wait
for a stream's first output. The `ProviderSettings` rustdoc's "Worst-case
latency" section has every timeout layer and how to bound a call.

An interactive application, or a side of a `FallbackProvider`, sets a
shorter hint ceiling, so a long rate limit fails over or reaches the user
at once:

```rust,ignore
use std::time::Duration;

use allama::wrappers::{RetryPolicy, TimeoutPolicy};
use allama_providers::ProviderSettings;

let settings = ProviderSettings::new()
    .retry(Some(RetryPolicy::new().max_retry_after(Duration::from_secs(5))))
    // Optional: bound each attempt, so a stalled one is retried.
    .attempt_timeout(Some(TimeoutPolicy::new().stream_idle(Duration::from_secs(30))));
let (provider, model) = allama_providers::chat::from_model_with("openai:gpt-5.5", &settings)?;
```

- Do not wrap the result in another `RetryingProvider`: retries would
  nest. Configure them here, or turn them off with `.retry(None)`.
- `409 Conflict` is not retried, unlike in the official OpenAI and
  Anthropic SDKs (which treat it as a lock timeout), unless the server
  sends `x-should-retry: true`. To retry it anyway:
  `RetryPolicy::new().retry_if(|e: &Error| e.server_retry_directive().unwrap_or(e.is_retryable() || e.status() == Some(409)))`.
- `ProviderSettings` also sets a default `unsupported_policy` for chat and
  embedding requests, and `chat_capabilities` overrides for model ids the
  adapter cannot classify (fine-tuned models, deployment names, gateway
  aliases).

## Model addresses

A model address is a provider key and a model id joined by the first `:`.
The model id passes through unchanged, so `ollama:llama3:8b` selects Ollama
with model `llama3:8b`. There are no aliases.

The grammar is fixed, so a configured address resolves the same way in
every release
([ADR 0016](../../docs/decisions/0016-model-addresses-and-the-builtin-registry.md)):

- Keys are lowercase ASCII letters, digits and `-` (`openai-chat`), and a
  `Registry` registers no other key.
- Whitespace is refused, never trimmed: `openai: gpt-5` and a trailing
  newline from a config file fail with `Error::Configuration`. So does
  `OpenAI:gpt-5`. The error suggests `openai:gpt-5` when that key is
  registered, as it does for `openai/gpt-5`.
- The model id is not empty and does not start with `:` (`ollama::` is
  refused).

`ModelAddress` parses an address with the same rules, to check
configuration at startup or to keep the key:

```rust,ignore
use allama_providers::ModelAddress;

let address: ModelAddress = "ollama:qwen3:1.7b".parse()?;
assert_eq!((address.key(), address.model()), ("ollama", "qwen3:1.7b"));
```

The lookups return the provider and the model id as a pair, which
destructures straight into a request. Parse a `ModelAddress` when you need
more, such as the key for logs.

### Addresses are trusted configuration

An address chooses the server that receives the request, and the API key
sent with it. The local presets (`ollama`, `vllm`, `litellm`) work without
configuration and reach `localhost`, so an address taken from an end user
can make your server call its own loopback ports. Read addresses from
your configuration. When users choose a model, map their choice through
an allowlist of addresses you configured, and never pass their string to
`from_model`.

## Built-in providers

| Key | Chat | Embeddings | Decisions | Feature |
|---|---|---|---|---|
| `anthropic` | Messages API | no | no | `anthropic` |
| `gemini` | Gemini API | yes | no | `gemini` |
| `openai` | Responses API | yes | no | `openai` |
| `openai-chat` | Chat Completions | no | no | `openai` |
| `openai-responses` | Responses API, as `openai` | no | no | `openai` |
| `azure-responses` | Azure OpenAI Responses API | no | no | `openai-compat` |
| `github-copilot` | GitHub Copilot, Chat Completions | no | no | `openai-compat` |
| `azure`, `cloudflare`, `deepinfra`, `deepseek`, `fireworks`, `groq`, `mistral`, `together`, `vllm`, `xai` | `allama-openai-compat` presets | yes | no | `openai-compat` |
| `ollama`, `openrouter` | `allama-openai-compat` presets | yes | `allama-systemone` presets | `openai-compat`, `systemone` |
| `digitalocean`, `litellm`, `typesafe` | no | no | `allama-systemone` presets | `systemone` |

A key names one provider configuration, and its operations may come from
different crates: `ollama` reaches the same server (`OLLAMA_HOST`) for chat,
embeddings, and decisions.

**`openai` is the Responses API.** OpenAI has two chat APIs: `openai`
builds `allama_openai::ResponsesProvider` (Responses) and `openai-chat`
builds `allama_openai::Provider` (Chat Completions). Embeddings under
`openai` come from `allama_openai::Provider`, the only OpenAI embedding API.
Pick `openai` by default, and always for reasoning models, whose reasoning
it carries across tool calls; pick `openai-chat` when you need the Chat
Completions wire format. `openai-responses` is `openai` under the name
Pydantic AI also accepts, so its model strings (`openai-responses:gpt-5.5`)
resolve unchanged.

**Azure has both APIs too.** `azure` is Azure OpenAI's Chat Completions
API, and `azure-responses` its Responses API, at the same resource with
the same environment (`AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_ENDPOINT`
or `AZURE_OPENAI_RESOURCE`). The model is the deployment name:
`azure-responses:my-deployment`. Embeddings stay under `azure`; the keys
named after an API (`openai-chat`, `openai-responses`, `azure-responses`)
offer chat only.

**A key keeps its API.** A built-in key reaches the same adapter and
endpoint in every 0.x release, so a configured address never changes
behaviour under you (ADR 0016). `gemini` is Gemini's `generateContent`
API, which Google now calls legacy but fully supported; its newer
Interactions API is not supported yet, and `gemini-interactions` is the
key reserved for it.

All features are on by default; disable default features to pick some.
The provider crates come without their own default features: `extract` is
not forced on (enable `allama/extract` yourself), and the TLS backend is
one of this crate's `default-tls` (default), `rustls`, `native-tls` or
`rustls-ring` (`ring` instead of `aws-lc-rs`, for static musl builds
without cmake) features, forwarded to every enabled provider. Every
enabled provider comes with its `reqwest` transport, since this crate
builds providers from configuration:

```toml
allama-providers = { version = "0.1", default-features = false, features = ["anthropic", "openai", "rustls"] }
```

No feature sends trace context. To continue your trace in a gateway (W3C
`traceparent`), wrap the providers this crate builds in a chat wrapper
that adds the header to each request, with your own OpenTelemetry
version: the
[observability guide](../../docs/guides/observability/README.md#trace-context-propagation)
has the recipe.

Each provider reads its environment with its own `from_env` (for example
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `OLLAMA_HOST`,
`TYPESAFE_API_KEY`).

## Credential scopes

A per-request credential (`allama_http::HttpRequestOptions::credential`) is
addressed to a credential scope, and only the provider with that scope
sends it, so in a fallback chain each provider gets its own key. Each
built-in provider's default scope is its provider name:

| Key | Default credential scope |
|---|---|
| `anthropic` | `anthropic` |
| `gemini` | `gemini` |
| `openai`, `openai-chat`, `openai-responses` | `openai` |
| `azure`, `azure-responses` | `azure` |
| other `openai-compat` and `systemone` presets | the key (`groq`, `ollama`, `typesafe`, ...) |

Each provider crate publishes its default as `DEFAULT_CREDENTIAL_SCOPE` (on
each preset type for presets). The `_with` functions build a provider with
another scope, for example one per tenant account:

```rust,ignore
use allama_providers::ProviderSettings;

let settings = ProviderSettings::new().credential_scope("tenant-acme");
let (provider, model) = allama_providers::chat::from_model_with("openai:gpt-5.5", &settings)?;
```

The provider still reads its key from the environment, and sends it when a
request carries no credential for the scope.

## Errors

Problems with the configuration itself fail with `Error::Configuration`,
before any request is sent: an unknown key, a provider without the
requested operation (`anthropic` for embeddings), a key whose feature is
off, a malformed model address (no `:`, whitespace, an uppercase key),
and a provider whose environment is incomplete (for example
`OPENAI_API_KEY` is not set). For an unknown
key or a missing operation, the message lists the keys that offer the
operation. Configuration errors are not retryable and are never failed
over, so report them instead of retrying.

```rust,ignore
match allama_providers::chat::from_model(&address) {
    Ok((provider, model)) => { /* ... */ }
    Err(error @ Error::Configuration { .. }) => panic!("check the model setting: {error}"),
    Err(error) => return Err(error),
}
```

## Custom providers

```rust,ignore
use allama_providers::Registry;

let mut registry = Registry::builtin();
registry.register_chat("myco", MyCoProvider::from_env);

let (provider, model) = registry.chat_from_model("myco:large")?;
```

`Registry::new()` starts empty when you want to choose every provider.
The `chat::from_model` family uses the built-in registry, which cannot be
changed: a provider key of your own, or one from a third-party provider
crate, needs a `Registry` of your own, passed to the code that resolves
addresses. There is no process-wide registration, so no dependency can
change what `chat::from_model` resolves to (ADR 0016).
A registered provider gets the same retries and other wrappers as a
built-in one. A constructor registered with `register_chat` takes no
settings, so a `_with` lookup that asks for a credential scope or chat
capability overrides fails for its key with `Error::Configuration`;
register one with `register_chat_with` to receive them as
`AdapterSettings` and apply them to the adapter. It takes each one
(`take_credential_scope`, `take_chat_capabilities`), and a lookup fails
when the constructor leaves one untaken, so a tenant's scope is never
dropped silently.

A provider you build yourself gets the same stack from `chat::wrap`,
`embedding::wrap` or `decision::wrap`. Do not wrap the result in another
`RetryingProvider`: it already retries.

```rust,ignore
use allama_openai_compat::providers::Vllm;
use allama_providers::ProviderSettings;

let adapter = Vllm::new("http://gpu-box:8000/v1", None)?;
let provider = allama_providers::chat::wrap(adapter, &ProviderSettings::new())?;
```

`wrap` refuses a credential scope (and, for chat, capability overrides)
with `Error::Configuration`, as a lookup does for a constructor registered
without settings: build the adapter with `with_credential_scope` or
`with_chat_capabilities` instead.

The applications guide's recipe,
[Building providers from your own configuration](../../docs/guides/applications.md#building-providers-from-your-own-configuration),
shows both shapes with keys from a configuration file, one shared
`ReqwestTransport` over a `reqwest::Client` with a proxy, and the request
timeout.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
