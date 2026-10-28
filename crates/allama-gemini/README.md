# allama-gemini

[![crates.io](https://img.shields.io/crates/v/allama-gemini?style=flat-square)](https://crates.io/crates/allama-gemini)
[![docs.rs](https://img.shields.io/docsrs/allama-gemini?style=flat-square)](https://docs.rs/allama-gemini)

Google Gemini API (`generativelanguage.googleapis.com`) provider adapter for
`allama`.

It speaks the `generateContent` API. Google now calls it legacy, "fully
supported", and recommends its Interactions API for new projects
(<https://ai.google.dev/gemini-api/docs/interactions>). The Interactions
API is not supported yet; it still lacks safety settings and explicit
caching, which this adapter's options cover. The `gemini` key of
`allama-providers` stays on `generateContent`.

## Use it when

- you want Gemini models behind the shared `allama::ChatProvider` trait
- you need Gemini 3 multi-turn tool use, which requires replaying thought
  signatures exactly
- you want thinking, structured output, files, and embeddings through the
  portable request/response model

## Quick start

```toml
[dependencies]
allama = "0.1.0"
allama-gemini = "0.1.0"
anyhow = "1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust,no_run
use allama::prelude::*;
use allama_gemini::Provider;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = Provider::from_env()?;

    let response = provider
        .chat(
            &ChatRequest::new("gemini-3.5-flash")
                .user("Explain tool calling in two sentences.")
                .reasoning(ReasoningEffort::Low),
        )
        .await?;

    println!("{}", response.text_or_empty());
    Ok(())
}
```

Environment variables:

- required: `GEMINI_API_KEY`
- optional: `GEMINI_BASE_URL`

## Capabilities

- tools, including parallel calls (Gemini has no switch to disable them;
  `parallel_tool_calls(false)` produces a warning)
- native streaming (SSE chunks are deltas)
- image and file input (`inlineData` / `fileData`), image output on image
  models, and assistant image replay (a draft image made while thinking
  keeps `"google-ai"` / `"thought": true` and is replayed as one)
- audio output (a TTS model such as `gemini-3.8-flash-tts`, with
  `responseModalities: ["AUDIO"]` in
  `ChatRequestOptions::generation_config_extra`) arrives as `Other` content
  of type `inline_data`, holding the `inlineData` part as Gemini sent it.
  allama may give audio output its own variant in a minor release, so
  match this `Other` content only as a fallback
- image detail as each image's `mediaResolution` (`Low` →
  `MEDIA_RESOLUTION_LOW`, `High` → `MEDIA_RESOLUTION_HIGH`, `Auto` unset)
- Files API files: `File::provider_file(media_type, "google-ai", "files/abc")`
  (the `name` an upload returns) is sent as `fileData` with its Files API
  URI; the upload's `uri` also works as a URL source
- response data without a portable form stays in the response provider
  data under `"google-ai"`: `service_tier`, `safety_ratings`,
  `prompt_feedback`, `url_context_metadata`, `logprobs_result`,
  `avg_logprobs`, `citation_metadata` (its sources also become `Citation`s
  on the answer text), and unmodeled usage fields under `usage`; the
  modality token counts are `Usage::breakdowns`
  (`text_input_tokens`, `image_input_tokens`, …)
- synthetic tool-call ids: Gemini 2.5 sends calls without ids, so the
  adapter makes them up as `gemini_<responseId>_call_<n>` (or
  `gemini_anonymous_<uuid>_call_<n>` without a response id), and code
  execution as `gemini_<responseId>_code_<n>`. They use only
  `[A-Za-z0-9_-]` (other characters of the response id become `_`), so a
  conversation continues on any provider. The format is stable: the
  adapter parses it back to order tool results by call. Gemini takes any
  `functionCall.id`, so ids from other providers are replayed unchanged.
- structured output (`responseJsonSchema`) and tool schemas
  (`parametersJsonSchema`), sent unchanged: `$ref`, `$defs`, type arrays, and
  recursion are accepted
- reasoning: thought summaries as `Reasoning` blocks, effort mapping, replay
- prompt caching: implicit caching satisfies cache breakpoints; explicit
  cached content via `ChatRequestOptions::cached_content`
- embeddings via `batchEmbedContents` (at most 100 inputs per call;
  `embed_chunked` splits larger batches); `gemini-embedding-2` also embeds
  images and composite inputs (a caption and an image as one vector, at most
  six images per input)
- `EmbeddingRequest::dimensions` maps to `outputDimensionality`. Google
  normalizes only the full-size vectors: a truncated one is not unit
  length, so normalize it yourself before comparing by dot product
  (<https://ai.google.dev/gemini-api/docs/embeddings#quality-for-smaller-dimensions>)

Capability answers describe the API, not the model (ADR 0015). The adapter
answers `Supported` where Google documents a feature for a model family
(and image detail on Gemini 3, verified live), and `Unknown` otherwise. It never refuses, drops or clamps a
feature because a model page says the model lacks it: TTS, image and Live
models get tools and structured output sent, Gemini 2.0 gets its reasoning
config, every model gets penalties and multimodal tool results. The model
refuses upstream, and a refusal Google words in a known way is
`Error::Unsupported` at the `Provider` stage (for example "Penalty is not
enabled for this model", or "Function calling is not enabled for
models/…"). For refusals before sending, install a capability resolver
(`with_chat_capabilities`).

## Reasoning

| Model family | `ReasoningEffort` maps to |
| --- | --- |
| Gemini 3.x | `thinkingLevel`: `None` → MINIMAL (LOW on Gemini 3 Pro), `XHigh` → HIGH (both reported as approximations) |
| Gemini 2.5 | `thinkingBudget`: None 0 (Pro: 128), Minimal 512, Low 1024, Medium 8192, High 24576, XHigh 32768 (Flash: 24576) |
| Gemini 1.x, 2.0 | `thinkingBudget`, the same ladder without the 2.5 limits; a model without thinking refuses upstream |
| Gemma 4 | `thinkingLevel`, on or off: `None` → MINIMAL, any other effort → HIGH |
| Any other model id | `thinkingLevel`, as on Gemini 3, without the Pro limits |

`ReasoningOutput::Omitted` turns off `includeThoughts`; `Full` is
approximated as summaries, which is all Gemini returns. The output setting
only shows or hides thoughts: without an effort the model keeps its default
thinking, and a `Summary` or `Full` output where the thinking budget is 0
has no effect and is an unsupported hint. An explicit
`ChatRequestOptions::thinking_budget` overrides the mapping. A thinking
level a model refuses ("Thinking level MINIMAL is not supported for this
model", on Gemini 3 Pro models whose id the adapter does not recognize as
Pro) is `Error::Unsupported` (stage `Provider`) on the reasoning field.

Replaying thought summaries costs prompt tokens. Use
`ReasoningOutput::Omitted` when you do not need them.

## Sampling with reasoning

Nothing is dropped or refused because a model thinks: sampling is sent as
asked. Google "strongly recommend[s] keeping the temperature parameter at
its default value of `1.0`" for every Gemini 3 model
(<https://ai.google.dev/gemini-api/docs/gemini-3>).

## Thought signatures

Signatures are stored on the block they arrived with, in
`provider_data["google-ai"]["thought_signature"]`, and replayed on the same
part. They survive serialization (`allama::history::History`).

If a Gemini 3 conversation, or one with a model id the adapter does not
know, contains a function call without a Gemini signature (for example,
history from another provider), the adapter sends Google's documented
placeholder signature and reports an `Approximated` warning. State from other providers is never replayed.

## Features

Structured-output extraction (`allama::ChatExtractExt`) is `allama`'s
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

`ChatRequestOptions` and `EmbeddingRequestOptions` (namespace `"google-ai"`)
hold Gemini-only fields: `response_mime_type`, `cached_content`,
`thinking_budget`, `builtin_tools` and `safety_settings` (`SafetySetting`:
a harm category and its blocking threshold); `task_type` and `title` for
embeddings.
For retrieval, prefer the portable `EmbeddingRequest::input_type`
(`Query`, `Document`), which maps to `taskType` `RETRIEVAL_QUERY` and
`RETRIEVAL_DOCUMENT`, sent to every model. `gemini-embedding-2` takes no
task type ("include the task as an instruction in your prompt"): leave the
input type unset there and put the task in the text.

Both types are `#[non_exhaustive]`: build them with `new()` and the setters,
or pass the JSON form. Fields this crate does not model go in
`generation_config_extra` (merged into `generationConfig`:
`responseModalities`, `mediaResolution`, `responseLogprobs`, …) and `extra`
(merged into the request body, or into each embedding request). A field
the adapter owns, in either spelling (`topK` or `top_k`), is refused with
`Error::InvalidRequest`.

```rust
use allama::ChatRequest;
use allama_gemini::{ChatRequestOptions, SafetySetting};
use serde_json::json;

// `top_k` is a portable request field, not an option.
let request = ChatRequest::new("gemini-3.5-flash").user("hi").top_k(20).with_option(
    ChatRequestOptions::new()
        .safety_setting(SafetySetting::new("HARM_CATEGORY_HARASSMENT", "BLOCK_ONLY_HIGH"))
        .generation_config_extra("responseModalities", json!(["TEXT"])),
);

// The same as JSON:
let request = ChatRequest::new("gemini-3.5-flash").user("hi").top_k(20).with_options_json(
    "google-ai",
    json!({
        "safety_settings": [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"}],
        "generation_config_extra": {"responseModalities": ["TEXT"]}
    }),
);
```

The options namespace is `google-ai`, not the provider name `gemini`: JSON
options under `gemini` are not read, and come back as a `Dropped` warning
coded `dropped.provider_name_options`.

## Degradations

- Frequency and presence penalties are sent as `frequencyPenalty` and
  `presencePenalty`; a model that does not enable them refuses with
  `Error::Unsupported` (stage `Provider`) naming the penalty.
- Several system prompts are sent as the parts of one `systemInstruction`,
  in order.
- A `seed` outside Gemini's 32-bit range is not sent, with an `Unsupported`
  warning (an error under `UnsupportedPolicy::Error`).
- A tool result's text parts are joined with blank lines; its media go
  after the text, and when that moves media ahead of text the adapter
  reports an `Approximated` warning.
- Image URLs need a known image extension; otherwise use
  `File::url(media_type, url)` or base64.
- Image and file tool results are sent as multimodal function responses to
  every model; Google documents them for Gemini 3, and an earlier model
  refuses upstream.

## Errors

- `API_KEY_INVALID` (HTTP 400) → `Error::Auth`
- 402, or 429 with spend-cap wording → `Error::QuotaExhausted`; other 429 →
  `RateLimited`, with `RetryInfo.retryDelay` taking priority over headers
- 503 → `Overloaded`, 504 → `Timeout`, 404 → `ModelNotFound`
- a missing or expired explicit cache (`cached_content`) or Files API file
  → `InvalidRequest` naming the resource, although Gemini answers 403
  `PERMISSION_DENIED` (or 404)
- a model id that is not a model name (`<id>`, `models/<id>`,
  `tunedModels/<id>`) → `InvalidRequest` before anything is sent
- errors inside an SSE stream are mapped the same way

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
