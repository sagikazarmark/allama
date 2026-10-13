# OpenAI model claims

Model-specific rules that `allama-openai` and `allama-openai-compat` do not
apply (ADR 0015). Each row is a negative model claim, or a per-model answer
that belongs with them; a model catalog can re-add them as data, with the
evidence below.

- **Trigger:** the model ids, families or markers the rule matched, as the
  code parsed them before the rule was removed. Families: `gpt-4o*` and
  `gpt-4.1*` (GPT-4); `gpt-5[.n][-size][-date]` (GPT-5, minor `n`);
  `gpt-5*-chat*`; `gpt-6[.n]-{astra,sol,luna}`; `o<digits>[-…]` (o-series),
  with `o1-mini*` apart. A fine-tuned id `ft:<base>:…` was its base model.
- **Evidence:** copied from the removed code, with the date it was checked.
  Model pages are `https://developers.openai.com/api/docs/models/<id>`.
- **Upstream mapping:** whether OpenAI's (or the endpoint's) rejection maps
  to `Error::Unsupported { stage: Provider, subject }`. *Pending* means no
  documented or captured rejection text exists, so no pattern was added; the
  rejection stays `InvalidRequest`.

## Models an endpoint does not serve

| ID | Trigger | Old behavior | Evidence | Upstream mapping |
|---|---|---|---|---|
| O1 | An id with a `pro` token (`o1-pro`, `o3-pro`, `gpt-5-pro`), or containing `codex` (`gpt-5-codex`, `codex-mini-latest`) or `deep-research` (`o3-deep-research`, `o4-mini-deep-research`) | On Chat Completions, every capability `Unsupported`: tools, schema, images and files refused; reasoning, cache and parallel-call hints dropped; replayed reasoning dropped | Model pages, checked 2026-10-03: `o1-pro`, `o3-pro`, `gpt-5-pro` "available in the Responses API only"; `gpt-5-codex` "It's available in the Responses API only"; `codex-mini-latest`, `o3-deep-research`, `o4-mini-deep-research` list no Chat Completions endpoint. The `pro`/`codex`/`deep-research` *generalisation* had no evidence. | Pending. OpenAI's reply for a Responses-only model on Chat Completions is not documented, and no reply is captured. A catalog entry should list the exact ids, not the marker. |
| O2 | `o1-mini*` | On Responses, every capability `Unsupported` | Model page `o1-mini` (endpoints: Chat Completions and Assistants), checked 2026-10-03 | Pending (no documented or captured reply). |

## Withdrawn preset-wide answers

These were positive claims, not negative ones, but the catalog is where
per-model answers for these endpoints belong:

| ID | Trigger | Old behavior | Evidence | Upstream mapping |
|---|---|---|---|---|
| C5 | Every model on Groq, Together, Fireworks, DeepInfra, OpenRouter, DeepSeek, xAI, Mistral, Azure, Cloudflare `/ai/v1` | `ToolCalls` `Supported` | None cited | Ollama's "does not support tools" is mapped; OpenAI-shaped `unsupported_parameter` with `param: tools` maps generically. Other vendors' texts pending. |
| C6 | Every model on Cloudflare `/ai/v1` | `StructuredOutput` `Supported`, so extraction picked Native as verified and never fell back | The json-mode page lists the models that take JSON mode, and says "Workers AI can't guarantee that the model responds according to the requested JSON Schema" (<https://developers.cloudflare.com/workers-ai/features/json-mode/>); `allama-cloudflare-worker` lists them in `JSON_MODE_MODELS` | OpenAI-shaped `unsupported_parameter` with `param: response_format` maps generically; Cloudflare's own text pending. |

## `allama-openai`: removed rules

| ID | Trigger | Old behavior | Evidence | Upstream mapping |
|---|---|---|---|---|
| O3 | `-search-preview` (`gpt-4o-search-preview`, `gpt-4o-mini-search-preview`) | `ToolCalls`, `ParallelToolCalls`, `ForcedToolChoice` `Unsupported`: tools refused | Model page `gpt-4o-search-preview`: "Function calling" not supported (quote, no URL in code; checked 2026-10-03) | Generic only: `unsupported_parameter`/`unsupported_value` with `param: tools` or `tool_choice` maps; this model's reply is not captured. |
| O4 | `-audio-preview` | `StructuredOutput` `Unsupported`: JSON schema refused | Model page `gpt-4o-audio-preview`: "Structured outputs" not supported (quote, no URL in code) | Generic only (`param: response_format`); reply not captured. |
| O6 | gpt-4o, gpt-4.1, `gpt-5*-chat*`, `o1-mini*` (and their `ft:` fine-tunes) | `ReasoningConfig`, `ReasoningOutput`, `ReasoningReplay` `Unsupported`: effort dropped with a warning (an error under `Error`); replayed reasoning dropped; the Responses encoder dropped `reasoning.effort` and `summary` | None for gpt-4o, gpt-4.1, `gpt-5-chat` (a "no reasoning" comment only). o1-mini: "`o1-mini` doesn't support `reasoning_effort`" (<https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reasoning>) | Mapped: `reasoning_effort` (Chat Completions) and `reasoning`, `reasoning.effort`, `reasoning.summary` (Responses) are in the `unsupported_parameter`/`unsupported_value` table; the Chat Completions spelling is pinned too, in `allama-openai-compat` `error.rs` (`parameters_a_reasoning_model_refuses_name_their_subject`). The exact reply for these models is not captured. |
| O8 | `o1-mini*` | Every non-reasoning capability `Unsupported`: tools, schema, images and files refused; parallel-call hint dropped | Model page `o1-mini`: "Function calling" and "Structured outputs" not supported; Azure reasoning guide (above) | Generic (`param: tools`, `response_format`); image and file refusals name a message path and stay `InvalidRequest`. Pending a captured reply. |
| O9 | `gpt-6[.n]-astra`, `gpt-6.<n≥1>-sol` | `ToolCalls`, `ParallelToolCalls`, `ForcedToolChoice` `Unsupported` on Chat Completions | "GPT-6 Astra and GPT-6.1 Sol support Chat Completions, but tool calling requires Responses" (<https://developers.openai.com/api/docs/guides/latest-model>, checked 2026-10-03). Tool answers on Chat Completions are now `Unknown` for every GPT-6 tier. | Generic (`param: tools`); reply not captured. |
| O10 | `gpt-6-sol`, `gpt-6-luna` (minor 0) | On Chat Completions, a request with tools refused (`Error::Unsupported`, every policy) while the effort, or the default effort, was not `none` (`ToolCallSupport::WithoutReasoning`) | "GPT-6 Sol and GPT-6 Luna support function calling in Chat Completions only with reasoning_effort: "none"" (latest-model guide, above) | Generic (`param: tools`); reply not captured. |
| O13 | `gpt-5` and its sizes at minor 0 (not `-chat`), the o-series, `o1-mini` | `SamplingSupport::Rejected` on both APIs: `temperature`, `top_p`, `frequency_penalty`, `presence_penalty` dropped with an `Unsupported` warning (an error under `Error`); `logprobs` and `top_logprobs` removed from the options' `extra` the same way | "HTTP 400" (comment only, no URL or fixture) | Added: `temperature`, `top_p` and the penalties are mapped (fixture "Unsupported parameter: 'temperature' is not supported with this model."); `logprobs` and `top_logprobs` map too, to a `Subject::Other` that names the parameter (the subject of the old drop warning). |
| O14 | `gpt-5.<n≥1>` (any size, not `-chat`), every GPT-6 tier | `SamplingSupport::WithoutReasoning`: the O13 drops while the request's effort, or the default effort (O16), was not `none` | Verified live on `gpt-5.1` (no fixture); "When reasoning effort is not none, remove temperature, top_p, and top_logprobs. For Chat Completions, also remove logprobs" (latest-model guide, above) | Added, as O13. |
| O16 | gpt-5 (minor 0) and the o-series: `medium`; gpt-5.1: `none`; `gpt-6-sol`, `gpt-6.1-sol`, `gpt-6-luna`: `medium` | `default_effort` in both rule sets: the effort a request without one was taken to reason at, the input to O10, O13 and O14 | gpt-5.1: "none (default)" (model page); "All models before `gpt-5.1` default to `medium`" (openai-python `types/shared/reasoning.py` v2.14.0); GPT-6 Sol, 6.1 Sol and Luna "also default to medium reasoning effort" (<https://developers.openai.com/api/docs/guides/reasoning>) | n/a: an input to the rules above, not a refusal. |
| OE2 | `text-embedding-ada*` | `EmbeddingCapability::OutputDimensions` `Unsupported`: `dimensions` refused before sending | None in code | Pending. No documented reply; one reported in the community is "This model does not support specifying dimensions." with no `param`, unverified, so no pattern was added. |

Kept in the adapter:

- **O15, effort levels, wire format.** The Responses encoder maps a
  requested level to the nearest one a model documents: `gpt-5`, `gpt-5-mini`,
  `gpt-5-nano` at minor 0 (`minimal`…`high`; `gpt-5-nano` rejects `none` and
  `xhigh`, verified live), the o-series (`low`…`high`, no evidence), and
  `gpt-6-astra` at minor 0 (`low`…`xhigh`, reasoning-guide URL). An Astra
  point release (`gpt-6.1-astra`) does not inherit the list, and any other
  id gets its effort unchanged. `ResponsesProvider::with_model_rules`
  replaces the levels for a model. The catalog may carry level lists too.
- **O18, the `ft:` dialect.** It is a schema dialect chosen by OpenAI's
  fine-tuned id convention (`ft:gpt-4.1-nano-2025-04-14:openai::BTz2REMH`,
  <https://developers.openai.com/api/docs/guides/supervised-fine-tuning>).
- **O21, O22, O23, O24, OE3, OE5, OE6:** protocol facts.
- **O5, O7, O11, O12, O17, O19, OE1, OE4:** positive or `Unknown` answers.

## `allama-openai-compat`: removed rules

| ID | Trigger | Old behavior | Evidence | Upstream mapping |
|---|---|---|---|---|
| C1 | Groq preset, any model not in {`openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`} | `ChatModelRules::response_format_constraint(SchemaConstraint::Never)`: a strict `json_schema` response format went out without `strict`, with an `Unsupported` warning (an error under `Error`); extraction reported `NotEnforced` | Strict mode is "currently only available on select models": GPT-OSS 20B and 120B and Qwen 3.8 27B; every other model gets best-effort JSON (<https://console.groq.com/docs/structured-outputs>, "Supported Models") | Pending. Groq does not document its reply to `strict: true` on another model. Until the catalog lists the models, every Groq model reports the endpoint's enforcement (`Exact` for a strict-expressible schema). |
| C2 | Ollama preset, a model whose tag is `cloud` or ends in `-cloud` (`gemma4:cloud`, `qwen3-coder:480b-cloud`) | `ChatModelRules::structured_output(Unsupported)`: a JSON or JSON Schema response format refused before sending, ahead of the declared answer | Ollama Cloud "currently does not support structured outputs" (<https://docs.ollama.com/capabilities/structured-outputs>); "Through a signed-in local server, select a cloud model such as `gemma4:cloud`" (<https://docs.ollama.com/api/openai-compatibility>); `qwen3-coder:480b-cloud` as a model id for a local base URL (<https://docs.ollama.com/integrations/roo-code>) | Pending. The docs do not say whether Ollama Cloud rejects `format` or ignores it. |

Kept in the presets: C3 (the Ollama Cloud *host*), C4 (Ollama ignores
`tool_choice`), C7 (vLLM's server-side structured outputs), C8, C9.

The Workers AI binding's per-model rules are in [workers-ai.md](workers-ai.md).
