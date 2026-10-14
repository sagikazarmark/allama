# allama-openai-compat

[![crates.io](https://img.shields.io/crates/v/allama-openai-compat?style=flat-square)](https://crates.io/crates/allama-openai-compat)
[![docs.rs](https://img.shields.io/docsrs/allama-openai-compat?style=flat-square)](https://docs.rs/allama-openai-compat)

OpenAI-compatible endpoints for `allama`: presets for applications, and a
toolkit for provider authors.

allama is not affiliated with [Ollama](https://ollama.com); Ollama is one of
the backends it supports.

Applications use its presets for OpenAI-compatible endpoints (Groq,
Ollama, vLLM, …). Provider authors build on it when an endpoint speaks
OpenAI Chat Completions or the OpenAI Responses API with its own URL,
auth, headers or request fields.

Build on `Provider` and `ResponsesProvider` with a dialect (`CompatQuirks`,
`ResponsesQuirks`). The Chat Completions and Embeddings wire types are
internal: they follow the vendors and change with them. Fields this crate
does not model go in the options' `extra` map, which refuses the fields the
adapter owns.

## Presets

```rust
use allama::prelude::*;
use allama_openai_compat::providers::Groq;

async fn hello() -> allama::Result<String> {
    let provider = Groq::from_env()?; // GROQ_API_KEY
    let response = provider
        .chat(&ChatRequest::new("llama-3.3-70b-versatile").user("Hello"))
        .await?;
    Ok(response.text_or_empty())
}
```

| Preset | Credential |
| --- | --- |
| `Azure` | `AZURE_OPENAI_RESOURCE`, `AZURE_OPENAI_API_KEY` |
| `Cloudflare` | `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` |
| `DeepInfra`, `DeepSeek`, `Fireworks`, `Groq`, `Mistral`, `OpenRouter`, `Together`, `XAi` | `<NAME>_API_KEY` |
| `GitHubCopilot` | `COPILOT_GITHUB_TOKEN`, optional `GITHUB_COPILOT_BASE_URL` |
| `Ollama` | none (`OLLAMA_HOST`) |
| `Vllm` | `VLLM_BASE_URL`, optional `VLLM_API_KEY` |

Each preset sets the provider identity (`ChatResponse::provider`, the
provider-data namespace `openai-compat:<name>`) and a `CompatQuirks` dialect,
and answers `Unknown` for capabilities it cannot vouch for. Use
`<Preset>::builder(..)` to adjust anything. Every preset reads its request
options under the shared namespace `openai-compat` (`ChatRequestOptions`),
not under its name: JSON options under `groq` are not read, and come back
as a `Dropped` warning coded `dropped.provider_name_options`.

Capabilities declared `Unsupported` (builder or resolver) are enforced
before a request is sent, by `allama::adapter::chat::preflight` and
`allama::adapter::embedding::preflight` (provider crates built on this toolkit call
them with their own answers too): semantic features (tools, a forced tool
choice, a JSON Schema response format, image or file input, embedding
`dimensions`) fail
with `Error::Unsupported`; hints (image detail, reasoning configuration,
prompt caching) are removed with a warning, or fail under
`UnsupportedPolicy::Error`; reasoning that cannot be replayed is
dropped with a warning. An embedding batch over `max_embedding_inputs` fails with
`Error::InvalidRequest` (split it with `EmbeddingProviderExt::embed_chunked`).

### How the presets are verified

Known gaps for 0.1, recorded so you can weigh them:

- `fixtures/presets/*.json` are snapshots of the requests each preset
  builds: they pin the dialect against changes in this crate, not against
  the vendor. The weekly key-gated job (`tests/e2e_presets.rs`) calls
  DeepInfra, DeepSeek, Fireworks, GitHub Copilot, Groq, Mistral,
  OpenRouter, Together and xAI when their keys are set, and Azure, on both of its APIs
  (`tests/e2e_azure.rs`), when its key and resource are. vLLM has no live
  job.
- The Mistral (`thinking` parts) and OpenRouter (`reasoning_details`)
  response fixtures are built from the vendors' documentation, not
  recorded. So is what the `Mistral` preset sends: `reasoning_effort`,
  `top_k` and `prompt_cache_key`, and no `prompt_cache_retention`, as
  Mistral's API reference documents; `zai-glm-5-3` is sent `low` or
  `high` only (`Mistral::model_rules`). No live Mistral call has checked
  it yet.
- GitHub Copilot's endpoint is not a published GitHub API. The
  `GitHubCopilot` preset's reasoning shapes (`reasoning_text` with a
  `reasoning_opaque` signature) are built from what Copilot's own editor
  clients (VS Code, Zed) read and send back, and the refusals it maps
  (`invalid_reasoning_effort`, `unsupported_api_for_model`) from Pydantic
  AI's recordings; a model the token's plan does not serve
  (`model_not_supported`, `ModelNotFound`) was seen live. Models the host
  serves only on its Responses API are out of reach. The weekly job calls
  it when `COPILOT_GITHUB_TOKEN` is set.
- Mistral reasoning is not replayed. Mistral asks for the assistant
  message back with its `thinking` chunks, but no `ReasoningReplayField`
  sends them, so the `Mistral` preset drops the reasoning it returned on
  later turns, silently, as its own reasoning.
- xAI: `completion_tokens` becomes `Usage::output_tokens` as sent. Whether
  xAI counts reasoning tokens in it, as ADR 0005 requires of
  `output_tokens`, is not verified (no key was available); if it does not,
  `output_tokens` undercounts xAI reasoning models.

## Dialects

`CompatQuirks` describes how an endpoint deviates from OpenAI: which field
carries the output cap (`max_tokens` or `max_completion_tokens`), whether
`seed`, `developer` roles, strict schemas, JSON Schema output, reasoning
effort, `top_k`, prompt-cache keys and retention (separately), stream usage, or parallel tool calls are
accepted, and which assistant-message field replays reasoning
(`ReasoningReplayField`: `reasoning_content`, `reasoning`, OpenRouter's
`reasoning_details`, or none). Reasoning an endpoint returned carries its
namespace, so with no replay field the endpoint's own reasoning is dropped
without a warning (Mistral's included, which Mistral would take back in the
assistant content).

Vendor response shapes decode as faithfully as the portable model allows:
parallel tool calls without `index`, Mistral's `thinking` content parts and
OpenRouter's `reasoning_details` (as reasoning), fields the crate does not
model on a tool call (Gemini's thought signature, replayed with the call) or
on a choice or message (`logprobs`, `native_finish_reason`, kept in provider
data), vendor usage counters (`cost`, kept in provider data under `usage`),
`stop` with tool calls (`ToolCalls`), and in-band errors in every shape
(numeric codes, bare strings, vLLM's envelope, an error with HTTP 200).
Raw Chat Completions annotations are kept under `chat_annotations`, apart
from the Responses `annotations` that share the `openai` namespace.
Unsupported fields are dropped with a `Warning` (or rejected under
`UnsupportedPolicy::Error`), never silently. The one exception is
`parallel_tool_calls: true` on an endpoint without the field: it is
permission the model has anyway, so it is dropped without a warning
(`false` is still a hint).

Tool call ids from another provider are mapped to ones the endpoint
accepts (`tool_call_id_format`, a `ToolCallIdFormat`): at most 64
characters by default and on Responses (OpenAI's limit), exactly nine
letters and digits on `Mistral` and for Mistral models on the `Cloudflare`
preset (`Cloudflare::model_rules`). An id that already fits is sent as it
is; one that does not is replaced by a stable id derived from it, the same
in `tool_calls` and in the `tool` message that answers it. Set
`ToolCallIdFormat::Alphanumeric(9)` yourself for a vLLM server running
Mistral models with the Mistral tokenizer.

Strict schemas are rewritten into the endpoint's `schema_dialect`
(`SchemaDialect::OpenAiStrict` by default; `None` sends them as written,
with an `Approximated` warning that the endpoint's grammar engine received
them unchecked). The OpenAI dialects (`fine_tuned_dialect`) switch to
`OpenAiStrictFineTuned` for `ft:` models; the `Azure` preset uses
`SchemaDialect::AzureStrict` on every deployment. On both, strict tools do
not hold under parallel tool calls, so a request with a strict tool sends
`parallel_tool_calls: false` unless the caller set it, and an explicit
`true` is reported as an `Unsupported` warning.
OpenAI strict mode cannot express maps (`HashMap<String, T>`), unconstrained
values (`serde_json::Value`), a root that is not an object, or the
composition keywords OpenAI documents as unsupported
(`SchemaDialect::rejected_keywords()`: `allOf`, `not`, `dependentRequired`,
`dependentSchemas`, `if`/`then`/`else`; a single-element `allOf` is
flattened instead); such a tool or
response format goes out without strict, as written, with an `Unsupported`
warning, or fails under `UnsupportedPolicy::Error` (the core
`allama::adapter::schema::apply_strict_policy` decides). The dialect pass is public
as `schema::strict_schema` for adapters that speak OpenAI's grammar with
their own envelope: pass it to `apply_strict_policy`.

OpenAI strict mode requires every property, so an optional property goes out
nullable, unless it has a `default` (a `#[serde(default)]` field whose type
is `Serialize`), which keeps it non-nullable. Parse strict tool arguments with
`ToolCall::parse_arguments_with`, which removes the `null`s the model sends
for optional properties; extraction does the same.

For a custom endpoint:

```rust
use allama_openai_compat::{CompatQuirks, Provider};

fn internal() -> allama::Result<Provider> {
    let mut quirks = CompatQuirks::new("openai-compat:internal");
    // The endpoint reports billing exhaustion as `quota_exceeded`: map it to
    // `Error::QuotaExhausted` (not retryable) instead of a rate limit.
    quirks.quota_error_codes.push("quota_exceeded".into());

    Provider::builder()
        .base_url("https://llm.internal.example/v1")
        .bearer_token("token")
        .provider_name("internal")
        .quirks(quirks)
        .build()
}
```

## Vendor additions

Endpoints add data the Chat Completions format does not define. Nothing is
dropped and nothing is misread:

- Unmodeled top-level response and chunk fields (`citations`,
  `search_results`, …) are kept in `ChatResponse::provider_data` under the
  dialect namespace; streams surface them as `ResponseProviderData` events.
- Named SSE events (`event: search_results`) are never decoded as chunks.
  Their data is kept under the dialect namespace and the event name, as an
  array of payloads in arrival order, with one `Warning` per event name.
  `event: error` frames are typed errors.

```rust
use allama::{ChatProvider, ChatRequest, ChatStreamExt};
use allama_openai_compat::Provider;

async fn citations(provider: &Provider, request: &ChatRequest) -> allama::Result<()> {
    let response = provider.chat_stream(request).await?.collect_response().await?;
    let citations = response.provider_data.get_raw("openai-compat:internal", "citations");
    if let Some(citations) = citations {
        println!("citations: {citations}");
    }
    Ok(())
}
```

## Responses API

`ResponsesProvider` speaks the OpenAI Responses API (`POST {base_url}/responses`)
for any endpoint that serves it, **statelessly by default**: a request sends
`store: false` and the whole conversation. Server-side state is opt-in through
`ResponsesRequestOptions`: `store`, `previous_response_id` (the request's
messages then carry only the new turn), `conversation`, and `background`
(which needs `store`; poll it with `ResponsesProvider::retrieve_response`).
It is the codec behind `allama_openai::ResponsesProvider`, which only adds
the OpenAI dialect and the effort levels OpenAI documents per model.

- `ResponsesQuirks` describes the endpoint: its provider-data `namespace`,
  whether reasoning is replayed (`ReasoningReplay::AsReceived` replays the
  `encrypted_content` and/or raw reasoning `content` it returned;
  `Disabled` drops it with a warning), which request modes it serves
  (`StreamingSupport`), and whether it accepts strict tools, `"status": null`
  on replayed function calls, function calls interleaved with other items,
  JSON Schema output, the `developer` role, `parallel_tool_calls`,
  prompt-cache keys, and `tool_choice`. `ResponsesQuirks::new` is a
  conservative OpenAI-like dialect; `ResponsesQuirks::openai()` is OpenAI's.
- `ResponsesModelRules` (via `.model_rules(|model: &str| ...)`) describes what
  a model accepts: reasoning effort levels (others map to the nearest, with an
  `Approximated` warning), whether it reasons, and whether it rejects sampling
  parameters while reasoning. The default assumes nothing and sends settings
  as requested.
- Only state under the endpoint's namespace is replayed, so reasoning and item
  ids from one endpoint are never sent to another (they are dropped with a
  warning). Give each endpoint its own namespace.
- Request fields with no portable equivalent go in `ResponsesRequestOptions`
  (namespace `"openai-compat-responses"`).

```rust
use allama::prelude::*;
use allama_openai_compat::{ResponsesProvider, ResponsesQuirks};

async fn hello() -> allama::Result<String> {
    // Ollama ignores `tool_choice`, so forced tool use is rejected up front
    // instead of silently becoming `auto`.
    let mut quirks = ResponsesQuirks::new("openai-compat:ollama");
    quirks.supports_tool_choice = false;
    let provider = ResponsesProvider::builder()
        .base_url("http://localhost:11434/v1")
        .provider_name("ollama")
        .quirks(quirks)
        .build()?;
    provider.ask("qwen3:1.7b", "Say hello").await
}
```

Endpoint dialects become presets only after live verification through this
provider. Azure is the one preset with a Responses constructor:
`Azure::responses_builder(resource, key)` (and `responses`,
`responses_with_endpoint`, `responses_from_env`) builds a
`ResponsesProvider` for the same resource, key and namespace as the Chat
Completions preset, so a conversation can move between the two APIs.

| Endpoint | Base URL | Quirks | Status |
| --- | --- | --- | --- |
| Azure OpenAI | `https://{resource}.openai.azure.com/openai/v1`, or `{endpoint}/openai/v1` | `Azure::responses_quirks()`: OpenAI's dialect under `openai-compat:azure`, with the `AzureStrict` schema dialect on every deployment. Deployment names are not model ids, so no model rules are assumed; set them per deployment with `model_rules`. | Verified live (2026-10-14, deployments `gpt-5.6-luna` and `gpt-6-luna`, whose function tools Chat Completions refuses at its default reasoning effort; chat, streaming, tools, strict structured output, and encrypted reasoning replayed across a tool call): `tests/e2e_azure.rs` |
| Ollama | `http://localhost:11434/v1` | `ResponsesQuirks::new("openai-compat:ollama")` with `supports_tool_choice = false` (Ollama echoes `"auto"` and never forces a call). Reasoning comes back as summary text plus a plaintext `encrypted_content`, replayed as received; `reasoning_tokens` is always reported as 0. | Verified live (Ollama 0.34, `qwen3:1.7b`): `tests/e2e_responses_ollama.rs` |
| LiteLLM proxy | `http://127.0.0.1:4000/v1` | none needed: message items streamed without `response.output_item.added` and raw reasoning as `output_text` parts are accepted by the decoder. Requests with `reasoning` fail inside LiteLLM 1.98's Ollama bridge (a LiteLLM bug). | Verified live (LiteLLM 1.98 in front of Ollama `qwen3:1.7b`): `allama-examples/tests/e2e_litellm.rs`, recorded `fixtures/responses/litellm_*` |
| vLLM | `http://localhost:8000/v1` | `ResponsesQuirks::new("openai-compat:vllm")`; set `function_call_status_null` for releases that require `"status": null` on replayed function calls | Not yet verified |
| Cloudflare Workers AI | account-specific | `streaming: StreamingSupport::NonStreamingOnly` | Not yet verified |

## Building a provider on it

Start from the first row that fits, as the
[provider authors' guide](../../docs/guides/provider-authors.md#which-starting-point)
does:

| The endpoint… | Start from |
| --- | --- |
| is already a preset | the preset; adjust it with `<Preset>::builder(..)` |
| speaks Chat Completions, with its own URL, auth and accepted fields | `Provider::builder()` with a `CompatQuirks` dialect ([Dialects](#dialects)) |
| speaks the Responses API | `ResponsesProvider::builder()` with `ResponsesQuirks`, and `ResponsesModelRules` for per-model facts ([Responses API](#responses-api)) |
| has request fields of its own | `Provider`, with the fields in `ChatRequestOptions::extra` |
| has its own wire format | a native adapter, not this crate |

Vendor body fields go in `ChatRequestOptions::extra`, which refuses the
fields the adapter owns. Capability answers go on the builder
(`chat_capability`, `embedding_capability`), or in a resolver for per-model
facts (`Provider::with_chat_capabilities`). The builder also sets the
paths and header names an endpoint uses (`chat_completions_path`,
`embeddings_path`, `request_id_header_name`, `retry_after_header_name`).

```rust
use allama::{CapabilitySupport, ChatCapability, ChatRequest};
use allama_openai_compat::{ChatRequestOptions, CompatQuirks, Provider};

fn internal() -> allama::Result<Provider> {
    Provider::builder()
        .base_url("https://llm.internal.example/v1")
        .bearer_token("token")
        .provider_name("internal")
        .quirks(CompatQuirks::new("openai-compat:internal"))
        .chat_capability(ChatCapability::ImageInput, CapabilitySupport::Supported)
        .build()
}

// A body field the endpoint adds, which this crate does not model.
fn request() -> ChatRequest {
    ChatRequest::new("internal-large")
        .user("Hello")
        .with_option(ChatRequestOptions::new().extra("safe_mode", true))
}
```

A crate that wraps `Provider` and has an options type of its own maps it
onto `ChatRequestOptions` on the request it forwards. The `WireOptions`
types and the hooks around them are `#[doc(hidden)]`: they exist for
`allama-openai` and `allama-cloudflare-worker`, which release in lockstep
with this crate, and carry no semver promise. Do not build on them.

## Observability

Without `.provider_name(..)`, the provider identifies as `"openai-compat"`
(namespace `"openai-compat"`). Presets set a specific identity; set one for
custom upstreams so spans, warnings, and replayed state are attributed
correctly.

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

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
