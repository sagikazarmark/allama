# Workers AI model claims

Model-specific rules that the Cloudflare Workers AI binding adapter
(`allama-cloudflare-worker`) does not apply (ADR 0015), with the evidence a
model catalog needs to re-add them as data. The OpenAI-compatible
Cloudflare `/ai/v1` endpoint is covered in [openai.md](openai.md) (C5, C6).

Without a catalog, every row's request is sent as the caller asked. The
binding exposes no HTTP status, so a model's refusal is classified from the
`worker::Error` text and code (`allama-cloudflare-worker/src/error.rs`).

## Removed rules

| ID | Trigger | Old behaviour | Evidence | Upstream mapping |
|---|---|---|---|---|
| W5 | `@cf/baai/bge-small-en-v1.5`, `@cf/baai/bge-base-en-v1.5`, `@cf/baai/bge-large-en-v1.5` (exact) | `max_inputs_per_call` returned 100, so preflight refused a larger batch (`InvalidRequest`) and `embed_chunked` split at 100; the encoder re-checked it (`` `{model}` embeds at most 100 inputs per call, got {n}; split the batch with EmbeddingProviderExt::embed_chunked ``). Now no limit is claimed, the encoder's check is gone, and the batch is sent. | "`maxItems: 100` in their published input schema" (no URL in code). The schemas are published at <https://developers.cloudflare.com/workers-ai/models/bge-base-en-v1.5/> (and the `-small`/`-large` pages), source `src/content/workers-ai-models/<model>.json` in <https://github.com/cloudflare/cloudflare-docs>. | Pending: no documented error text for an oversized batch. |
| W3, image part | Every model | `ImageInput` and `ImageDetail` answered `Unsupported`; a user image part refused before sending (`Workers AI native chat does not accept image user content`). Now `Unknown`: a user message with images is sent as OpenAI-style parts (`{"type": "image_url", "image_url": {"url", "detail"}}`, inline data as a `data:` URI). | The binding passes each model's own input schema. Vision models take these parts: `@cf/meta/llama-4-scout-17b-16e-instruct` ("image uri with data (e.g. data:image/jpeg;base64,/9j/...). HTTP URL will not be accepted"), `@cf/mistralai/mistral-small-3.1-24b-instruct`, `@cf/moonshotai/kimi-k2.6` and `@cf/zai-org/glm-5.3` (with `detail`), per their schemas in <https://github.com/cloudflare/cloudflare-docs/tree/production/src/content/workers-ai-models>, checked 2026-10-04. Text models (for example `@cf/qwen/qwen3-30b-a3b-fp8`) take string content only. Catalog data: which models take images, and whether they take `https` URLs. | Pending: no documented error text for an image sent to a text model. |
| W3, reasoning | Every model | `ReasoningConfig` answered `Unsupported`; a reasoning config dropped as an unsupported hint (`the Workers AI binding adapter does not map reasoning configuration`), an error under `UnsupportedPolicy::Error`. Now `Unknown`: the effort is sent as `reasoning_effort` by its portable name (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`); only the reasoning output setting, which no schema carries, is still an unsupported hint. | `@cf/moonshotai/kimi-k2.6` takes `reasoning_effort` `high`/`none` ("Compatibility aliases: low maps to high; medium maps to high; max maps to high"), `@cf/zai-org/glm-5.3` takes `max`/`high`/`low` ("Reasoning cannot be disabled"), per their schemas (same source, checked 2026-10-04). `@cf/openai/gpt-oss-*` take `reasoning: {effort}` only on their Responses-shaped input, which this adapter does not send. Catalog data: each model's level list. | Pending: no documented error text. |

## Kept

| ID | Rule | Kind | Why |
|---|---|---|---|
| W1, W2 | `ToolCalls` / `StructuredOutput` `Supported` for the listed models, `Unknown` otherwise | Positive claim | Safe when stale (the model decides). |
| W3, rest | `ParallelToolCalls`, `ForcedToolChoice`, `FileInput`, `ImageOutput`, `ImageReplay`, `ReasoningReplay` `Unsupported` for every model | Adapter fact | The adapter encodes none of them: no parallel-call control, no `tool_choice`, no file parts, no assistant images, no replayed reasoning. Newer model schemas (`kimi-k2.6`, `glm-5.3`) do list `parallel_tool_calls`, `tool_choice` and file parts, so encoding them would make these per-model too; that is adapter work still to do. |
| W4 | `StreamUsage` `Unknown` | Unknown claim | Unchanged. |
| W6 | Embeddings: `dimensions`, image, composite input, `input_type` `Unsupported` | Protocol fact | The embedding schema is `{text: [...]}` for every model. |
