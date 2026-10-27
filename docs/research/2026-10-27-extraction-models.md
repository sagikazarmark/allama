# Extraction models (Cloudflare Clairic) and allama

- Date: 2026-10-27
- Status: research note. The decision is
  [ADR 0018](../decisions/0018-extraction-models.md), which departs from
  this note's recommendation: it introduces the trait now, under an
  unstable status, together with the provider crate, rather than a
  provider crate alone first.

## Question

Cloudflare announced [Clairic](https://blog.cloudflare.com/clairic-extraction-model/)
on 2026-10-19, "a new System One class of extraction models". How is the
ecosystem (Ollama, the Vercel AI SDK, other AI libraries and gateways)
integrating this class, and if and how should allama?

## Summary

- **allama already covers the first System One class.** Decision models
  (TypeSafe Jev, Cloudflare Clef, Ollama's local models) are
  `DecisionProvider` in core, served by `allama-systemone` over HTTP (with
  a `Cloudflare` Workers AI preset) and by `allama-cloudflare-worker` through
  the AI binding. The open question is only the second class: extraction.
- **Extraction is a different operation, not a decision and not chat
  structured output.** A decision answers with probabilities over a finite
  option set; an extraction answers with a typed value per named field
  (a text span, number, date, duration, list, bounded distribution, or
  forecast), a null probability, and a confidence. The shared
  `{model, answers, usage}` envelope is transport, not domain.
- **Nobody has a portable extraction abstraction, and there is exactly one
  provider.** Clairic is hosted-only, in preview, "an early checkpoint";
  open weights are promised but not released. Ollama, llama.cpp, vLLM, the
  Vercel AI SDK, Cloudflare's own AI SDK path, LangChain, BAML, DSPy and
  Instructor have nothing. LiteLLM exposes a Cloudflare-only pass-through
  route; Pydantic AI has an open, explicitly Clairic-specific PR; Rust has
  three single-vendor crates.
- **ADR 0001's gate for a core trait is not met** (a ready provider crate
  *and* two providers whose semantics it covers). Designing
  `ExtractionProvider` now would rename Clairic's wire format, which ADR
  0001 names as the failure mode.
- **Recommendation: a provider crate first, no core trait yet.** Ship a
  typed, Clairic-shaped adapter over `allama-http` that reuses the Workers
  AI transport already in `allama-systemone`, with the same escape hatches
  (`extra`, raw response in provider data), fixtures, conformance-style
  tests and a runnable example. Promote to a core sibling trait with an
  ADR when a second implementation exists (open weights served by
  llama.cpp/Ollama/vLLM, or another vendor). This is the path rig,
  rust-genai, LangChain, and Pydantic AI took for the *decision* class
  before it converged, and the one the ecosystem is taking for extraction.
- **One naming conflict to settle before any code:** allama's glossary
  already defines "extraction" as the chat-based structured-output
  operation (`allama::extract`, `ExtractExt::extract`, extraction pass,
  extraction failure). The new class needs a name that does not collide.

## 1. What the class is

Sources: [Clairic blog](https://blog.cloudflare.com/clairic-extraction-model/),
[model page](https://developers.cloudflare.com/workers-ai/models/clairic/),
[input schema](https://developers.cloudflare.com/workers-ai/models/clairic/schema-input.json),
[output schema](https://developers.cloudflare.com/workers-ai/models/clairic/schema-output.json),
[concept page](https://developers.cloudflare.com/workers-ai/concepts/extraction-models/).

Cloudflare's own taxonomy (blog table):

| | System One: Extract | System One: Decide | System Two |
| --- | --- | --- | --- |
| Models | Clairic | Jev, Clef | LLMs |
| ML framing | Regression and extraction | Classification | Generation |
| Output | Typed values: spans, numbers, dates, lists, distributions | Options with probabilities | Free-form text |
| Output space | Continuous, typed, bounded by your schema | A finite set you list | Anything |

### Request

`POST /accounts/{id}/ai/run/@cf/cloudflare/clairic` (or `env.AI.run` in a
Worker). Required: `model`, `fields`.

- `state`: string. "Must be non-empty unless images are sent. Text answers
  are spans of this string." Unlike Clef/Jev, **not** a JSON object/array.
- `fields`: map of name → `{type, instructions?, ...}`; at least one, no
  upper limit (fields that do not fit "next to the state in one
  8,192-token pass are split across parallel passes"). Names match
  `^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$`.
- Field `type` ∈ `text | number | date | time | datetime | duration | list |
  estimate | forecast`, with per-type parameters:
  `integer`, `decimals` (0–6), `unit` (duration unit enum), `items`
  (list: `text|number|date`), `min`/`max` (estimate/forecast, number or ISO
  date), `step` (number, or `"month"`/`"year"`), `scale` (`linear|log`),
  `null_threshold` (default 0.5; schema only, not in prose),
  `horizon` (1–64), `interval` (`step|minute|…|year`).
- `images`: ≤ 4, PNG/JPEG/WebP, inline only (`data:` URL or
  `{content_type, base64}`), 4 MiB each, 8 MiB total; body ≤ 16 MiB.
- `reference_date`: ISO date anchoring dates written without a year.
- `options.include_quantiles` (adds 99 percentiles to estimates),
  `options.return_state_text` (returns the text spans index into).

### Response

`{model, answers, truncated, usage[, state_text]}`. Each answer has `type`
and `value` (null when the input does not determine it) plus, by type as
seen in examples (the schema is one loose object):

| Type | Keys |
| --- | --- |
| `text` | `p_null`, `span: [start, end]`, `guess`, `confidence` |
| `number`, `date`, `time`, `datetime`, `duration` | `p_null`, `guess`, `confidence` (`unit` echoed on duration) |
| `list` | `value: [...]`, `spans: [[..], ..]`, `confidence` (present in docs, absent in blog) |
| `estimate` | `value`, `mean`, `median`, `p10`, `p90`, `p_below`, `p_above`, `quantiles?` |
| `forecast` | arrays per step in the blog; a `steps` array in the schema; neither matches the other |

`usage`: `input_tokens`, `output_tokens` (always 0), plus `state_tokens`,
`image_tokens`, `schema_tokens`, `passes`. Pricing is input-only
($0.10/M), listed under "Other model pricing", not LLMs. Context window
65,536 tokens; its relation to the 8,192-token pass is not explained.

### What is unsettled in the API itself

The first subagent's comparison of blog, model page and schemas found
these gaps; they matter for how much of the shape to type:

1. `p_below`/`p_above` are undefined in every source (inferred: mass
   clipped below `min` / above `max`).
2. `truncated` is required but its meaning is undocumented; `state_text`
   implies spans index into a server-normalised copy of `state`, not
   necessarily the caller's bytes. Span offset units (bytes, UTF-8 chars,
   UTF-16 units) are unstated.
3. `confidence` on a null answer describes the `guess`, not the null
   decision (docs: `p_null: 1, confidence: 0.9995`).
4. Forecast shape differs between blog and schema; estimate `mean` and
   list `confidence` appear in one source but not the other.
5. The page is filed under "Text Generation" and has no error section;
   malformed field specs have no documented failure.
6. Clairic's `state` is string-only and `instructions` string-only, while
   Clef/Jev accept JSON for both. The sibling classes diverge on input.

### Relation to decision models

Clef declares itself "fully Jev-API compatible" and labels `images` a
"Clef extension to the System One API". Clairic makes no such claim: it
shares the `model`/`answers`/`usage` envelope and the name-keyed typed map
pattern, and diverges on everything else (`fields` vs `questions`,
value vocabulary, string-only state, `reference_date`, `options`).
TypeSafe has no extraction model; its nearest move is Jev 1.14's `box`
type (image localisation, 2026-10-14). TypeSafe's cookbooks do extraction
*via decisions* (regex candidates → choice question), which is what
`DecisionProvider` already supports.

### Second implementations

None that matter for a domain model, as of today:

- Open weights: not released. Cloudflare's HF org has `clef`,
  `clef-flash`, `clef-omni` only.
- [`tasksource/UltraBERT-150M-clairic-extraction`](https://huggingface.co/tasksource/UltraBERT-150M-clairic-extraction):
  community token classifier that borrows the request shape; text and list
  spans only, "can't compute values". Not affiliated with Cloudflare.
- [`LocalLLaMA/typed-extraction`](https://huggingface.co/datasets/LocalLLaMA/typed-extraction)
  benchmark calls `state`/`fields`/`reference_date` "the System One
  extraction request shape" and covers `text, number, date, time, datetime,
  duration, list` (no estimate/forecast). The leader (ScaleDown Extract on
  OpenRouter) is an ordinary chat-completions model the benchmark adapts.

## 2. How the ecosystem integrated System One models

### Decision class (shipped, converged)

This is the precedent for how a new model class gets abstracted, and it
validates allama's existing design.

| Project | Shape | Notes |
| --- | --- | --- |
| Ollama | `POST /v1/systemone` (v0.35.0, 2026-09-28), "based on TypeSafe's Jev API"; `decision` capability; `ollama run --ask`; `systemone()` in Python/JS SDKs | Separate class: decision models do not serve `/api/chat`. Subset of TypeSafe's schema ([#18718](https://github.com/ollama/ollama/issues/18718) open). Scoring now delegated to llama.cpp ([#18917](https://github.com/ollama/ollama/pull/18917)). |
| llama.cpp | `/v1/systemone` ([#29818](https://github.com/ggml-org/llama.cpp/pull/29818), 2026-10-02) keyed by GGUF `decision.type` | "Decision models are just fancy wrappers around traditional embedding models." |
| vLLM | `/v1/systemone` + OpenAI `/v1/decisions` ([#59299](https://github.com/vllm-project/vllm/pull/59299), [#60465](https://github.com/vllm-project/vllm/pull/60465)) | Both pooling-model and generative-scoring paths. |
| Vercel AI SDK | `experimental_decide` + `DecisionModelV4.doDecide` ([#20846](https://github.com/vercel/ai/issues/20846)); `noul`→`boolean`; confidence in `providerMetadata.<provider>`; "fail the entire call when any question type is unsupported" | Shipped 2026-09-16 with a native TypeSafe provider; OpenAI, Perplexity, Gateway native; Anthropic/Google as prompted adapters. |
| Pydantic AI | `DecisionModel.decide(DecisionRequest) → DecisionResponse`; TypeSafe, generic System One, OpenAI backends; `output_type` → questions | Most complete shipped core abstraction. |
| LiteLLM | `/v1/decisions` and `/v1/systemone`, translating between them; 13 providers incl. Clef | Separate route/type. |
| LangChain (py) | Provider-specific `TypeSafeClassifier`, `OpenAIDecisions`; core `BaseDecisionModel` deferred: "feels early to call what will be the industry standard" ([#41193](https://github.com/langchain-ai/langchain/issues/41193)) | |
| BAML, DSPy, Instructor, Mastra | Shipped; BAML rejects new vendors "unless multiple users request this" | Instructor is the only one framing it as a structured-output mode. |
| rig (Rust) | Separate `rig-typesafeai` crate: "not entirely sure where this part of the industry will go" ([#2550](https://github.com/0xPlaygrounds/rig/pull/2550)) | |
| rust-genai | Out of scope "until it gets stabilized by more providers"; separate `sysone` crate ([#338](https://github.com/jeremychone/rust-genai/issues/338)) | |

Patterns: every shipped abstraction treats decisions as a **distinct model
type**, not a chat mode; `state` + name-keyed questions → name-keyed
answers is the portable core; images, `box`, limits, and `confidence`
semantics stay provider-specific. The vocabulary split (`noul` /
`predicate` / `boolean`) is still open. allama's `DecisionProvider`,
`DecisionInput`, "yes/no question", confidence as provider data, and
limits via `decision_limits` line up with where the ecosystem landed.

### Extraction class (nothing portable anywhere)

| Project | Status | Source |
| --- | --- | --- |
| Ollama, llama.cpp, vLLM | No issues, PRs, or library entries for Clairic or "extraction model" | Searches recorded in the subagent report; nothing to serve without weights |
| Vercel AI SDK | No `extraction-model` spec, no `extract` function, zero issues; Cloudflare's `createAI` path has not even added decisions yet ([cloudflare/agents#2630](https://github.com/cloudflare/agents/issues/2630)) | `packages/provider/src`, `packages/ai/src` trees |
| LiteLLM | **Shipped** `POST /v1/extraction` (v1.108.0+), `litellm.extraction(model="cloudflare/@cf/cloudflare/clairic", state=…, fields={…})`; passes `fields`/`options` through unchanged; Cloudflare is the only provider | [docs](https://docs.litellm.ai/docs/extraction) |
| Pydantic AI | **Open PR** [#10635](https://github.com/pydantic/pydantic-ai/pull/10635) `CloudflareClairicModel`, `ClairicField`, `ClairicNumberEstimate`, `ClairicDateEstimate`, `ClairicForecast`: "These helpers remain specific to Clairic; this does not introduce a shared extraction abstraction." | |
| Rust crates | `clairic-client` 0.1.0 (standalone), `jsonllm` (+ derive, "generate typed Rust values through Clairic"), `llm-wires` (`Extract` trait "over Cloudflare Workers AI's Clairic") | crates.io |
| LangChain, LlamaIndex, BAML, DSPy, Instructor, Mastra, OpenRouter, Portkey, Helicone, cloudflare-go | Nothing | |

Two shipped shapes, both single-vendor: a **pass-through route** (LiteLLM)
and a **provider-specific model class** (Pydantic AI PR). Neither claims
portability. Vercel's bar for a new core category, stated when reranking
was proposed, was "very hesitant to add any abstraction that is based on
a single provider implementation"; reranking merged only after Cohere,
Together and Bedrock existed.

## 3. Fit with allama

### Precedent: how decisions landed

- ADR 0001: "Adjacent operations get their own traits beside
  `ChatProvider`"; a new trait "is designed only when a provider crate is
  ready to implement it, and is grounded in at least two providers whose
  semantics it covers … One provider's wire format renamed is not a domain
  model." Rejected: "Designing an operation ahead of a provider crate, from
  documentation alone."
- `DecisionProvider` (`crates/allama/src/decision/provider.rs:57`) ships
  with `decision_capability`, `decision_limits`, a dry-run
  `validate_decision_request`, `DynDecisionProvider`, forwarding impls,
  wrapper support, registry support, a typed layer (`Decision`,
  `#[derive(Decision)]`), and two providers (`allama-systemone`,
  `allama-cloudflare-worker`). ADR 0001 notes "every sibling trait costs its
  plumbing", which is why it waits for two providers.
- `allama-systemone` is named after the protocol, not a vendor, and its
  `Cloudflare` preset already handles Workers AI REST: per-model URL
  (`…/ai/run/{model}`), the `{"result": …}` envelope, Cloudflare's error
  envelope, `{content_type, base64}` images, `CLOUDFLARE_ACCOUNT_ID` /
  `CLOUDFLARE_API_TOKEN` (`crates/allama-systemone/src/lib.rs:238`,
  `src/providers.rs`). `allama-cloudflare-worker` reuses its codec through
  `allama_systemone::binding` so both transports encode alike.

### Why extraction is neither a decision nor chat extraction

- **Not `DecisionProvider`.** A decision's answer space is the finite set
  the caller lists; validation checks "a choice must be one of the
  question's options and carry exactly one probability per option"
  (`decision/response.rs:150`). An extraction's answer space is continuous
  and typed, with `null`, a null probability, a span into the input, and
  for estimates a distribution. Widening `Question`/`Answer` to carry them
  would break every invariant the decision module enforces and every
  provider that implements it. Cloudflare's own table separates the two.
- **Not `allama::extract`.** That operation drives a *chat model* through
  `ResponseFormat::JsonSchema` or a forced tool, with passes, repair,
  retries, and schema enforcement reporting. AGENTS.md: "Do not force
  adjacent capabilities into chat-shaped APIs." The glossary makes the
  same distinction for decisions: "A chat model prompted to estimate
  probabilities is not a decision model." A Pydantic/Instructor-style
  "struct → fields" mapping is a *typed layer* on top (like `Decision`),
  not the core.

### Naming collision

allama's glossary (CONTEXT.md "Extraction") owns "extraction" for the
chat structured-output operation: extraction request, extraction pass,
extraction failure, extraction refusal, `ExtractExt::extract`,
`ExtractionMode`, `ExtractionMetadata`, namespace `allama.extraction`.
Calling the new class "extraction models" and its trait
`ExtractionProvider` would put two different things under one word in the
same crate. Options, none settled here:

- Rename the new class. Cloudflare's own framing offers "regression",
  "predict values", "typed values". Candidates: *prediction model*
  (`PredictionProvider`, `PredictionRequest`, "field"), *value model*,
  *reading model*. "Prediction" covers extract/compute/estimate/forecast
  and is the blog's verb ("Purpose: Predict values"); it is also a loaded
  ML word.
- Keep "extraction model" for the class and qualify the chat operation
  ("structured output" already names its mechanism). This touches a lot
  of stable public API and documentation.
- Decide later, in the ADR, once a second provider shows what the class
  is actually called in the wild. The provider crate can use Clairic's
  own vocabulary (`fields`, `answers`) meanwhile, since it is
  provider-specific by construction.

### Options

**A. Do nothing; watch.** Zero cost, no learning. Loses the chance to
have an adapter and fixtures ready when weights ship, which for Clef took
Ollama two weeks and llama.cpp four days.

**B. Provider crate, no core trait (recommended).** A crate implementing
Clairic over `allama-http`, shaped like `allama-systemone` without the
core trait: a `Provider` with the Workers AI transport (shared with, or
lifted from, `allama-systemone`'s `Cloudflare` preset), typed request and
answer types mirroring the published schema, `extra: ExtraMap` on the
field spec and `RawResponse` in provider data for everything unsettled,
`ProviderIdentity`, per-request credentials, fixtures captured from the
live endpoint, an e2e test, and a runnable example. Add the same model to
`allama-cloudflare-worker` through the binding, sharing the codec, as
decisions do. Open questions for this option:

- Crate name. Provider crates are named by vendor (`allama-anthropic`),
  protocol (`allama-systemone`), or runtime (`allama-cloudflare-worker`).
  There is no REST `allama-cloudflare`/`allama-workers-ai` crate today
  (chat goes through the `allama-openai-compat` preset, decisions through
  `allama-systemone`). A model-named crate (`allama-clairic`) is the
  honest name until the shape has a protocol name; a `allama-workers-ai`
  crate would also be the natural home for other Workers-AI-native shapes.
- Whether a `#[derive]` typed layer (struct → fields, like `jsonllm` and
  `Decision`) is worth doing before the trait exists. It can be
  crate-local.
- Whether the types are designed with promotion in mind (neutral field
  kinds, `Option<T>` values, `allama::Image` for images, `Usage` for
  tokens) while still exposing Clairic's own `p_null`/`guess`/`passes`
  as typed fields rather than hiding them.

This matches ADR 0001's "a provider crate is ready to implement it", is
what rig and rust-genai did for the decision class while it was
single-vendor, and is the only shape (Pydantic AI, LiteLLM) anyone has
shipped for extraction.

**C. Core sibling trait now.** `ExtractionProvider`-style trait, request,
answer enum, capability enum, limits, dyn form, forwarding, wrapper and
registry support. Fails ADR 0001's two-provider gate and the AGENTS.md
rule "When promoting a provider feature into the core API, design the
neutral domain model first. Do not wrap one provider and rename it." With
one preview provider whose own docs disagree on the forecast shape, the
neutral model cannot be checked against anything.

**D. Extend `allama::extract` or `DecisionProvider`.** Rejected above.

### What a future core model would need to settle

Recorded so the provider crate can avoid foreclosing it:

- Field kinds: text span, number, date, time, datetime, duration, list of
  (text | number | date), estimate (bounded distribution with a grid and
  scale), forecast (a series of estimates with horizon and interval).
  Whether estimate/forecast belong in the same trait as extraction, or
  are a capability (`EstimateValues`, `ForecastSeries`) a model may not
  support (the UltraBERT model supports neither).
- Answer: `value: Option<T>`; null probability (portable in meaning: the
  probability the input does not determine the value); `guess` (what the
  model would have answered); spans (offset units must be pinned; a
  server-normalised `state_text` suggests spans may not index the caller's
  string); `confidence` (provider data, as for decisions, unless a second
  provider agrees on its meaning).
- Input: Clairic is string-only with images beside it, which matches
  `DecisionRequest`'s `input` + `images` split more than it matches a
  JSON `DecisionInput`. `reference_date` is a portable concept.
- Usage: input tokens only; `passes`, `schema_tokens`, `state_tokens` as
  provider data until a second provider reports them.
- Capabilities and limits per ADR 0015: image input, per-type support,
  field-name grammar, image count/size, as provider-level answers.
- Degradation policy: Clairic's request has no hints to drop
  (every field is semantic), so, like decisions, probably no
  `UnsupportedPolicy`.

### Triggers to revisit

- Clairic open weights published and served by llama.cpp, Ollama or vLLM
  (expect a cloned endpoint shape, as happened with `/v1/systemone`).
- A second vendor or gateway (OpenRouter, LiteLLM providers beyond
  Cloudflare) exposing an extraction-shaped API.
- TypeSafe adding value types beyond `box`.
- Cloudflare fixing the documented inconsistencies (forecast shape,
  `p_below`/`p_above`, `truncated`, span units) or publishing a versioned
  spec.

## Sources

Primary sources consulted by the research subagents, with the claims they
support, are listed inline above. Key entry points:

- Cloudflare: [Clairic blog](https://blog.cloudflare.com/clairic-extraction-model/),
  [model page](https://developers.cloudflare.com/workers-ai/models/clairic/),
  [input](https://developers.cloudflare.com/workers-ai/models/clairic/schema-input.json) /
  [output](https://developers.cloudflare.com/workers-ai/models/clairic/schema-output.json) schemas,
  [extraction models concept](https://developers.cloudflare.com/workers-ai/concepts/extraction-models/),
  [Clef blog](https://blog.cloudflare.com/clef-decision-models/),
  [Clef schema](https://developers.cloudflare.com/workers-ai/models/clef/schema-input.json),
  [pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/).
- TypeSafe: [Jev launch](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
  [Jev 1.14](https://typesafe.ai/blog/jev-1-14),
  [OpenAPI](https://api.typesafe.ai/openapi.json),
  [API reference](https://docs.typesafe.ai/api/system-one.md).
- Ollama: [v0.35.0](https://github.com/ollama/ollama/releases/tag/v0.35.0),
  [#18606](https://github.com/ollama/ollama/pull/18606),
  [#18741](https://github.com/ollama/ollama/pull/18741),
  [#18917](https://github.com/ollama/ollama/pull/18917),
  [`/v1/systemone` reference](https://docs.ollama.com/api/systemone),
  [`decision` package](https://github.com/ollama/ollama/tree/main/decision).
- llama.cpp [#29818](https://github.com/ggml-org/llama.cpp/pull/29818);
  vLLM [#59299](https://github.com/vllm-project/vllm/pull/59299),
  [#60465](https://github.com/vllm-project/vllm/pull/60465).
- Vercel AI SDK: [#20846](https://github.com/vercel/ai/issues/20846),
  [#21935](https://github.com/vercel/ai/pull/21935),
  [decisions guide](https://ai-sdk.dev/docs/ai-sdk-core/decisions),
  [`decision-model/v4`](https://github.com/vercel/ai/tree/main/packages/provider/src/decision-model/v4),
  [reranking #3764](https://github.com/vercel/ai/pull/3764);
  [cloudflare/ai#670](https://github.com/cloudflare/ai/pull/670),
  [cloudflare/agents#2630](https://github.com/cloudflare/agents/issues/2630).
- Others: [LiteLLM extraction](https://docs.litellm.ai/docs/extraction),
  [LiteLLM decisions](https://docs.litellm.ai/docs/decisions),
  [Pydantic AI #10635](https://github.com/pydantic/pydantic-ai/pull/10635),
  [Pydantic AI decision models](https://pydantic.dev/docs/ai/models/decision/index.md),
  [LangChain #41193](https://github.com/langchain-ai/langchain/issues/41193),
  [rig #2550](https://github.com/0xPlaygrounds/rig/pull/2550),
  [rust-genai #338](https://github.com/jeremychone/rust-genai/issues/338),
  [BAML #5416](https://github.com/BoundaryML/baml/pull/5416),
  [OpenRouter decisions](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request),
  [HN thread](https://news.ycombinator.com/item?id=50155907).
- Hugging Face: [Cloudflare org](https://huggingface.co/Cloudflare),
  [UltraBERT clairic-extraction](https://huggingface.co/tasksource/UltraBERT-150M-clairic-extraction),
  [typed-extraction benchmark](https://huggingface.co/datasets/LocalLLaMA/typed-extraction).
