# allama-cloudflare-worker

[![crates.io](https://img.shields.io/crates/v/allama-cloudflare-worker?style=flat-square)](https://crates.io/crates/allama-cloudflare-worker)
[![docs.rs](https://img.shields.io/docsrs/allama-cloudflare-worker?style=flat-square)](https://docs.rs/allama-cloudflare-worker)

Cloudflare Workers AI provider adapter for `allama`, built on the Worker's
native AI binding. It works with `workers-rs` (`worker`) without naming any
of its types: pass `&worker::Env` or `&worker::Ai` (or a
`wasm_bindgen::JsValue`), so it fits any `worker` version.

## Use it when

- you are already running inside a Cloudflare Worker (`wasm32-unknown-unknown`)
- you want to call Workers AI through the shared `allama::ChatProvider`,
  `allama::EmbeddingProvider`, `allama::DecisionProvider` and
  `allama::ExtractionProvider` (unstable, ADR 0018) traits
- you want native binding-based execution instead of outbound HTTP calls

Outside a Worker, use `allama-openai-compat`'s Cloudflare preset (chat and
embeddings), `allama-systemone`'s (decisions) and `allama-cloudflare`
(extraction) instead.

## Identity

`Provider::provider_name()` returns `"cloudflare"`, and responses, stream
metadata, errors and warnings are stamped with the same value, so OTEL
`gen_ai.provider.name` matches `allama-openai-compat`'s Cloudflare preset
regardless of transport. Provider options use the `"cloudflare-workers-ai"`
namespace (`allama_cloudflare_worker::NAMESPACE`).

## Capabilities

- Chat, non-streaming and streaming (`stream: true`, decoded from the SSE
  byte stream the binding returns).
- Tool calls, streaming or not. Upstream tool call ids are kept; when a
  model returns none, ids are synthesized deterministically as
  `cf_call_{turn}_{index}` (non-streaming), where `turn` is the index the
  assistant message takes in the conversation (`request.messages.len()`).
  Mistral models (`@cf/mistralai/…`, `@cf/mistral/…`, `@hf/mistral/…`)
  take only ids of nine letters and digits, so for them any other id (one
  from another provider, or a synthesized one) is replaced by a stable
  nine-character id derived from it, the same for the call and its result.
- Structured output through `response_format` (Workers AI JSON Mode,
  non-streaming). `allama`'s `ChatExtractExt` works through its blanket impl.
- Both response shapes Workers AI returns: the native
  `{ response, tool_calls, usage }` shape (flat or OpenAI-style tool calls)
  and the chat-completion `{ choices, usage }` shape newer models use, often
  with the native fields repeated. Both are normalized into the
  chat-completion shape and decoded with `allama-openai-compat`'s decoder,
  the one its Cloudflare preset uses, so the two transports decode alike
  (tool calls, reasoning, finish reasons, vendor fields such as the
  `neurons` usage counter, kept in provider data).
  `reasoning_content` becomes a `Reasoning` block.
- Image input: a user message with images is sent as OpenAI-style parts
  (`{"type": "image_url", "image_url": {"url", "detail"}}`, inline data as a
  `data:` URI), which vision models such as
  `@cf/meta/llama-4-scout-17b-16e-instruct` take. Text alone stays a string.
- Reasoning effort: `ReasoningConfig.effort` is sent as `reasoning_effort`
  by its name (`low`, `high`, …), which reasoning models such as
  `@cf/moonshotai/kimi-k2.6` take.
- Embeddings (`{ text: [...] }` models such as `@cf/baai/bge-base-en-v1.5`).
  `embedding_limits` claims no limit: a model's own limit (100 texts on
  the BGE English v1.5 models) is the model's to enforce.
- Decisions with Clef models (`@cf/cloudflare/clef`,
  `@cf/cloudflare/clef-flash`; see [Decisions](#decisions)).
- Extraction with Clairic (`@cf/cloudflare/clairic`; see
  [Extraction](#extraction)).

Tool calls, structured output, image input, image detail and reasoning
configuration are `Unknown`, as on the compat crate's Cloudflare preset:
which models take them is model data (ADR 0015), so they are sent for every
model, and a model that does not take them answers with an error. Nothing
is refused on the model id; claim them for your models with
`Provider::with_chat_capabilities`.

The request encoder stays this crate's own: the binding's input is not a
Chat Completions request (no `model`, native `top_k` and
`repetition_penalty`, no `stop`, no `tool_choice`).

### Finish reason and usage

Since the 2026-02-17 Workers AI release, streams report `finish_reason` only
on the final usage chunk. The adapter reads `usage` and `finish_reason` from
that chunk and emits them in the final `ResponseMetadata` event. When a model
or older runtime reports no finish reason, the adapter infers `Stop` (or
`ToolCalls` when tool calls are present) and leaves `finish_reason_raw` empty;
an upstream `"stop"` alongside tool calls is normalized to `ToolCalls`, with
the raw value kept. The aliases `eos`, `end_turn`, `max_tokens` and
`model_length` map to `Stop` and `Length`, as on the compat preset.

Usage follows the `allama` contract: `prompt_tokens` (which includes
`prompt_tokens_details.cached_tokens`) maps to `input_tokens`, cached tokens
to `cache_read_input_tokens`, and `total_tokens` is derived when missing.

A stream that ends without `[DONE]` or a final usage chunk yields a retryable
`Error::Stream` of kind `Incomplete`. A stream that ends without `[DONE]`
after the final usage chunk is complete: the chunk carries the finish reason,
which is all `[DONE]` would confirm.

The limit of inferring `Stop`: on a runtime that reports no finish reason, a
stream the runtime itself cut short but closed with `[DONE]` also reports
`Stop`, so `ChatResponse::finished_normally()` returns `true` for it. Other
adapters report no finish reason there, with a warning.

## Decisions

Clef is Cloudflare's multimodal decision model: it answers typed questions
about an input (text, JSON, images) with probabilities instead of writing
text. Its input and output are the System One protocol with Cloudflare's
extensions, encoded and decoded with `allama-systemone`'s codec, the one
its `Cloudflare` preset uses over HTTP, so both transports send and read
alike:

- the model selector Clef reads (`clef`, `clef-flash`) is the last segment
  of the model name;
- `DecisionRequest::images` go as `{"content_type", "base64"}`, inline
  only; a URL or a provider file is refused before the model runs;
- every question kind is `Supported`; `DecisionCapability::ImageInput` is
  `Unknown`, and `decision_limits` claims no limit, since both are model
  data (ADR 0015): a model refuses what its input schema does not allow
  (Clef: 64 questions, 255 options, 10 levels, 4 PNG, JPEG or WebP images,
  question names of at most 100 of `[A-Za-z0-9_.-]`) with an error that
  names the field;
- each question's `confidence` is kept as `allama_cloudflare_worker::Confidence`
  (re-exported from `allama-systemone`), and answers keep the order the
  model wrote them in.

```rust,no_run
use allama::decision::{Choice, ChoiceSet, Decision, YesNoAnswer};
use allama::DecisionProviderExt;
use allama_cloudflare_worker::Provider;

#[derive(Debug, ChoiceSet)]
#[decision(rename_all = "lowercase")]
enum Team {
    #[decision(description = "Payments, invoices, and refunds")]
    Billing,
    #[decision(description = "Outages, errors, and configuration")]
    Technical,
}

#[derive(Decision)]
struct Triage {
    #[decision(instructions = "Is this support request urgent?")]
    urgent: YesNoAnswer,
    #[decision(instructions = "Which team should handle this request?")]
    team: Choice<Team>,
}

async fn triage(env: &worker::Env, ticket: &str) -> allama::Result<Triage> {
    let provider = Provider::from_env(env, "AI")?;
    let decided = provider
        .decide_as::<Triage>("@cf/cloudflare/clef", ticket)
        .await?;
    Ok(decided.value)
}
```

The typed example uses `allama`'s `derive` feature. Binding call options
and model inputs this crate does not model go in `DecisionRequestOptions`
(`run_options`, `extra`), under the `cloudflare-workers-ai` namespace.

## Extraction

Clairic is Cloudflare's extraction model: it answers named, typed fields
about an input (text, images) with values instead of writing text: a span
of the input, a number read or computed, a date, a duration, a list, an
estimate inside bounds you declare, or a forecast of a series, each scalar
one a value or null with the probability that the input does not determine
it. In `allama` it is an `ExtractionProvider`, **unstable** (ADR 0018).
Its input and output are `allama-cloudflare`'s wire format without the
REST envelope, encoded and decoded with that crate's codec, so both
transports send and read alike and share the options type
(`ExtractionRequestOptions`, re-exported) and provider data
(`extraction_data::{Confidence, Guesses, Distributions, RunDetails}`, namespace
`cloudflare-workers-ai`):

- the model selector Clairic reads (`clairic`) is the last segment of the
  model name;
- `ExtractionRequest::images` go as `{"content_type", "base64"}`, inline
  only;
- every field kind is `Supported`, `ExtractionCapability::ImageInput` is
  `Unknown` (model data, ADR 0015), and `extraction_limits` claims what
  Clairic's input schema fixes (4 images, a forecast horizon of 64), as
  over REST;
- text spans are placed as byte spans into the request's input (a number
  or date list item's, with no text to compare, as character offsets that
  land in it); one the codec cannot place is dropped with a `dropped.span`
  warning;
- the binding call's options (an AI Gateway) go in
  `ExtractionRequestOptions::run_options`.

```rust,no_run
use allama::extraction::{Field, ListItemKind};
use allama::{ExtractionProvider, ExtractionRequest};
use allama_cloudflare_worker::Provider;

async fn claim(env: &worker::Env, email: &str) -> allama::Result<Option<f64>> {
    let provider = Provider::from_env(env, "AI")?;
    let request = ExtractionRequest::new("@cf/cloudflare/clairic", email)
        .field("order_id", Field::text().instructions("Order number"))
        .field("amount_paid", Field::number().decimals(2).unit("USD"))
        .field("damage", Field::list(ListItemKind::Text).instructions("Each kind of damage"));
    let response = provider.extract_values(&request).await?;
    Ok(response.number("amount_paid").unwrap().value)
}
```

## Provider options

```rust,no_run
use allama::{ChatRequest, EmbeddingRequest};
use allama_cloudflare_worker::{ChatRequestOptions, EmbeddingRequestOptions};
use serde_json::json;

let request = ChatRequest::new("@cf/meta/llama-4-scout-17b-16e-instruct")
    .user("Hello")
    // Portable sampling controls, `top_k` included, are request fields.
    .top_k(40)
    .with_option(
        ChatRequestOptions::new()
            .repetition_penalty(1.1)
            // A model input this crate does not model.
            .extra("lora", "my-adapter")
            // The binding call's own options: route through an AI Gateway.
            .run_option("gateway", json!({"id": "my-gateway"})),
    );

// Or as JSON (unknown fields are rejected). The namespace is
// `cloudflare-workers-ai`: options under the provider name, `cloudflare`,
// are not read, and come back as a `dropped.provider_name_options` warning.
let request = request.with_options_json(
    "cloudflare-workers-ai",
    json!({ "repetition_penalty": 1.1, "extra": {"lora": "my-adapter"} }),
);

// Embedding and decision models take inputs of their own too
// (`EmbeddingRequestOptions`, `DecisionRequestOptions`).
let embedding = EmbeddingRequest::new("@cf/baai/bge-base-en-v1.5")
    .input("Hello")
    .with_option(EmbeddingRequestOptions::new().extra("pooling", "cls"));
```

The option types are `#[non_exhaustive]`: build them with `new()` and the
setters. `extra` is merged into the input object the binding receives; a
field the adapter owns (`messages`, `max_tokens`, `stream`, `tools`, `text`,
…) is refused with `Error::InvalidRequest`. `run_options` is the third
argument of `env.AI.run(model, input, options)`, sent as given.

## Unsupported features

The adapter never drops a request feature silently.

Always rejected with `Error::Unsupported`:

- `tool_choice` `Required` / `Specific` (the binding cannot force a tool);
  `Disabled` is honored by not sending the tools
- `response_format` on `chat_stream` (JSON Mode is non-streaming)
- file user input, and assistant image replay
- embedding `dimensions`

Dropped with a `Warning` (or rejected under `UnsupportedPolicy::Error`):

- `stop`, `parallel_tool_calls: false`, the reasoning output setting
  (`ReasoningConfig.output`)
- request-level prompt cache key and retention, and explicit breakpoint
  retention (plain cache breakpoints need no action: Workers AI prefix
  caching is automatic for supported models)
- strict tool schemas, and the JSON Schema `name` / `strict` flags (JSON Mode
  takes a bare schema)

`parallel_tool_calls: true` is dropped without a warning: it is permission,
and a model that calls tools may call several in one turn anyway.

Reported as `Dropped` or `Approximated` warnings during replay: reasoning
blocks and foreign `Other` blocks in assistant history, other providers'
provider data, and tool-result `is_error`. The error text is sent as the
tool result, with an `Approximated` warning coded
`approximated.tool_result_error`, under `UnsupportedPolicy::Error` too:
the model still reads the error text. Keep `push_tool_error` in portable
code; to send your own encoding to this endpoint, use `push_tool_result`.

## Errors

The binding exposes no HTTP status. Errors are classified from the structured
`worker::Error` variants, then from the documented Workers AI error codes
(e.g. `3040` → `Overloaded`, `3036` → `QuotaExhausted`, `5007` →
`ModelNotFound`, read from the JS error `code` or the `"<code>: "` message
prefix, with the documented status attached), then from message substrings.

## Retries

When you wrap the provider in `allama::wrappers::RetryingProvider`, its
backoff jitter may be correlated across isolates: it hashes `RandomState`,
the clock, and a call counter, and in a fresh isolate with a frozen clock all
three may repeat. This is unverified. If isolates that fail together must
not retry together, see `RetryPolicy::jitter` for supplying your own
randomness.

## Quick start

```toml
[dependencies]
allama = "0.1.0"
allama-cloudflare-worker = "0.1.0"
futures-util = "0.3"
worker = "0.8"
```

```rust,no_run
use allama::prelude::*;
use allama_cloudflare_worker::Provider;

async fn handler(env: &worker::Env) -> Result<String> {
    // Reads the `AI` binding of the Worker (not process environment
    // variables). A missing binding fails with `Error::Configuration`.
    let provider = Provider::from_env(env, "AI")?;

    let response = provider
        .chat(&ChatRequest::new("@cf/meta/llama-4-scout-17b-16e-instruct").user("What is Rust?"))
        .await?;

    Ok(response.text_or_empty())
}
```

## Notes

- `Provider::from_env(env, "AI")` reads the binding from the Worker's `env`
  (a `&worker::Env`); unlike the HTTP providers' `from_env()`, it reads no
  process environment variables. `Provider::new(&ai)` wraps a binding you
  already hold (a `&worker::Ai`).
- The binding name must match your `wrangler.toml` configuration:

  ```toml
  [ai]
  binding = "AI"
  ```

- Inputs are converted with `serde_wasm_bindgen`'s JSON-compatible serializer
  so JSON objects (tool schemas, `response_format`) reach the binding as plain
  objects rather than ES `Map`s.
- Tests are fixture-based (`fixtures/`); the binding cannot run outside the
  Workers runtime. Regenerate expected fixtures with
  `CF_REGENERATE_FIXTURES=1 cargo test -p allama-cloudflare-worker` and review
  the diff. The binding calls themselves run under `wasm-bindgen-test`
  against a stand-in `AI` object
  (`cargo test --target wasm32-unknown-unknown -p allama-cloudflare-worker --lib`);
  no job calls Workers AI through a real binding yet.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
