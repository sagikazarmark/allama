# ADR 0004: Portable reasoning is effort plus output visibility

- Status: accepted

## Context

Providers control reasoning in different units. OpenAI and Anthropic's
adaptive thinking take an effort level; older Anthropic models and Gemini
take a token budget; some models cannot turn reasoning off. Reasoning
models also reject sampling parameters under rules that differ per
provider and per model.

## Decision

- `ReasoningConfig` carries two portable settings: an `effort`
  (`ReasoningEffort`, ordered `None < Minimal < Low < Medium < High <
  XHigh`) and an `output` visibility (`ReasoningOutput`: `Omitted`,
  `Summary`, `Full`). Leaving the effort unset keeps the provider default;
  `ReasoningEffort::None` asks for no reasoning.
- **Token budgets are not portable.** An exact budget is a provider option
  (Anthropic `budget_tokens`, Gemini `thinking_budget`). An adapter whose
  model takes only budgets translates the effort into one of its own
  choosing; the portable API never exposes a budget, and no adapter infers
  an effort from a budget.
- **Effort levels are a wire format.** An adapter sends the nearest level
  the model documents (`ReasoningEffort::nearest`) and reports the
  substitution as a warning. For a model id it does not recognize, it
  sends the effort unchanged and lets the provider decide (ADR 0015).
- **Reasoning fits inside the output cap.** `max_output_tokens` bounds
  reasoning too (ADR 0005), so an effort-derived budget is fitted into an
  explicit cap, never the cap raised to fit the budget (ADR 0003).
- **Sampling with reasoning follows protocol facts only.** Adapters drop
  sampling parameters as unsupported hints only where the API documents
  the rule for reasoning as a whole:
  - Anthropic, while thinking is on (manual or adaptive): `temperature`
    other than 1, `top_k`, and `top_p` outside [0.95, 1]. A forced tool
    choice with manual thinking is refused before sending.
  - OpenAI and Gemini: nothing is dropped; parameters are sent as asked.
- **Per-model rules are catalog data, not adapter defaults** (ADR 0015).
  Sampling rejected outright, `top_p` beside `temperature` on particular
  Anthropic models, forced tool choice on models that reject it, and
  OpenAI's sampling and tool limits while reasoning are sent as asked;
  the provider's refusal maps to `Error::Unsupported { stage: Provider }`.
  The OpenAI adapters keep the mechanism (`ChatModelRules` and
  `ResponsesModelRules`, with `SamplingSupport`, `ToolCallSupport` and
  `default_effort`), applied when a caller or the catalog installs rules
  through `with_model_rules`. The evidence for these rules is in
  [`docs/catalog/anthropic.md`](../catalog/anthropic.md) and
  [`docs/catalog/openai.md`](../catalog/openai.md).

## Consequences

- One effort setting works across providers; callers who need an exact
  budget or a provider-only level use typed provider options (ADR 0006).
- On Anthropic, `ReasoningEffort::None` sends the model format's off value
  even to a model that cannot turn thinking off, rather than the lowest
  effort, and an explicit `budget_tokens` is sent to every model: the API
  decides.
- Without the catalog, a model-specific sampling conflict fails at the
  provider rather than before sending.

## Rejected alternatives

- **A portable budget field.** Only some providers use budgets; deriving
  budgets from effort, or effort from budgets, in the shared API would be
  fake portability. AI SDK, Microsoft.Extensions.AI and pydantic-ai
  converged on the same effort-plus-visibility shape.
- **Dropping sampling parameters by model in the adapter.** Such negative
  model claims go stale as models change; they belong in the catalog.
