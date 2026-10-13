# Vendor response shapes

Each file is one response body of an OpenAI-compatible endpoint whose shape
differs from OpenAI's. The `.events.json` and `.response.json` files next to
a stream, and the `.expected.json` file next to a unary body, are what the
decoder makes of it; regenerate them with `ALLAMA_UPDATE_FIXTURES=1` and
review the diff (`tests/vendor_shapes.rs`).

| File | Source | Shape |
| --- | --- | --- |
| `gemini_parallel_no_index.sse` | Recorded 2026-10-07, Gemini's OpenAI-compatible endpoint, `gemini-2.5-flash` | Two parallel tool calls in one delta, without `index` |
| `gemini3_thought_signature.sse` | Recorded 2026-10-07, same endpoint, `gemini-3-flash-preview` | Parallel calls in separate chunks without `index`, the first with `extra_content.google.thought_signature`; `finish_reason: "stop"` |
| `mistral_thinking.sse`, `mistral_thinking.json` | Hand-built from Mistral's documented Magistral shape | `content` as an array of `thinking` and `text` parts |
| `openrouter_reasoning_details.sse`, `openrouter_reasoning_details.json` | Hand-built from OpenRouter's reasoning-tokens guide | `reasoning` with `reasoning_details` (text with a signature, and encrypted reasoning) |
| `reasoning_after_text.sse` | Hand-built | Reasoning that resumes after text |
| `openrouter_usage_cost.json` | Hand-built from OpenRouter's usage accounting | Vendor usage counters (`cost`), `native_finish_reason`, `logprobs` |
| `tool_call_without_id.json` | Hand-built | A unary tool call without an `id`, and one with an empty `id` |
