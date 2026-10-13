# Anthropic model claims

Model-specific rules that `allama-anthropic` does not apply (ADR 0015), each
with the trigger, the behaviour the adapter used to have, and the evidence,
so a model catalog can re-add the rule as data.

Without a catalog, every row's request is sent as the caller asked. The
"Upstream mapping" column says whether the provider's refusal maps to
`Error::Unsupported { stage: Provider }` (`allama-anthropic/src/error.rs`,
`upstream_refusal`, pinned in `refused_fields_are_unsupported_by_the_provider`).

## Removed rules

| ID | Trigger | Old behaviour | Evidence | Upstream mapping |
|---|---|---|---|---|
| A1 | Parsed family below Claude Haiku 4.5, Sonnet 4.5 or Opus 4.1: every `claude-3-*` id, `claude-sonnet-4-0`, `claude-opus-4-0` (e.g. `claude-sonnet-4-20250514`, `claude-3-5-haiku-latest`) | `StructuredOutput` answered `Unsupported`: `ResponseFormat::JsonSchema` refused before sending (`` `{model}` does not support structured outputs ``); strict tools sent without `strict`, with an `Unsupported` warning (`` `{model}` does not support strict tools; `{tool}` was sent without strict ``), an error under `UnsupportedPolicy::Error`; extraction Native skipped. Now `Unknown`. | None in code; the thresholds were pinned only by the test `model::tests::structured_output_support`. The structured-outputs guide lists the supported models: <https://platform.claude.com/docs/en/build-with-claude/structured-outputs>. | Pending: no documented reply text for structured output on an older model. |
| A3 | Parsed major version below 4, except Claude Sonnet 3.7 (`claude-3-5-*`, `claude-3-opus`, `claude-3-haiku`, …) | `ReasoningConfig`, `ReasoningOutput`, `ReasoningReplay` answered `Unsupported`: a reasoning effort dropped as an unsupported hint (`` `{model}` does not support extended thinking ``), an error under `UnsupportedPolicy::Error`; replayed reasoning blocks dropped by preflight. Now `Unknown`, and the effort is sent in the budget format. | None in code (the "no thinking before 4" comment only). | Pending: no documented reply text. |
| A5 | Claude Opus ≥ 5.5, Sonnet ≥ 5.5, Fable ≥ 5.1, Mythos ≥ 5.1 | `ForcedToolChoice` answered `Unsupported`: `ToolChoice::Required`/`Specific` refused before sending (`` `{model}` rejects forced tool choice (`any`/`tool`); use ToolChoice::Auto with a prompt instruction, or ResponseFormat::JsonSchema ``); extraction ForcedTool skipped. Now `Unknown` (the positive `Supported` answer is kept for the earlier generations). | "Adaptive thinking … supports forced tool use, except on Claude Opus 5.5, Claude Sonnet 5.5, Claude Fable 5.1, and Claude Mythos 5.1", <https://platform.claude.com/docs/en/build-with-claude/thinking#limits-and-feature-compatibility>; "Claude Sonnet 5.5 rejects both with a 400 error", <https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide#forced-tool-use>. | Added: "tool_choice: type \"tool\" and \"any\" are not supported for this model." → subject `ToolChoice` (the migration guide quotes the text). |
| A9 | A parsed adaptive-style id other than the 4.6 generation and Mythos Preview: Claude Opus/Sonnet ≥ 4.7, every 5.x, Fable, Mythos | An explicit `ChatRequestOptions::budget_tokens` refused before sending under every policy (`` `{model}` does not accept a fixed thinking budget (`budget_tokens`); use ChatRequest::reasoning effort instead ``), subject `Option { namespace: "anthropic" }`. Now sent as `{type: "enabled", budget_tokens}`. | "Claude 4.7 and later models do not support it and reject requests that use it, returning a 400 error … Claude Mythos Preview supports both modes", <https://platform.claude.com/docs/en/build-with-claude/extended-thinking>; the 4.6 generation is "deprecated (requests using it still succeed)" (same page). | Added: "\"thinking.type.enabled\" is not supported for this model. …" → subject `Reasoning`, <https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting>. |
| A10b | Claude Opus ≥ 5.5, Fable, Mythos, Mythos Preview | `ReasoningEffort::None` approximated: thinking left on at the lowest effort (`output_config.effort: "low"`), with an `Approximated` warning (`` thinking cannot be disabled on `{model}`; the lowest effort (`low`) was used ``). Now the adaptive format's off value, `{type: "disabled"}`, is sent. | The per-model table of <https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting> (cited in code for Mythos Preview only): Fable 5.1, Mythos 5.1, Fable 5, Mythos 5, Opus 5.5 and Mythos Preview reject `"disabled"`, "Thinking can't be turned off on these models" (<https://platform.claude.com/docs/en/build-with-claude/thinking>). The docs advise omitting `thinking` on these models; the catalog can restore that wire choice. | Added: "\"thinking.type.disabled\" is not supported for this model. …" (both documented wordings) → subject `Reasoning`. |
| A14 | Claude Opus ≥ 4.7, Sonnet ≥ 5.0, Fable, Mythos, Mythos Preview | `temperature` and `top_p` dropped as unsupported hints (`` `{model}` does not accept temperature `` / `top_p`), `top_k` dropped as a hint (`` `{model}` does not accept top_k ``), errors under `UnsupportedPolicy::Error`; the thinking-on limits skipped. Now sent; the thinking-on limits (A15) still apply. | Cited in code for Mythos Preview only (<https://platform.claude.com/docs/en/build-with-claude/thinking#limits-and-feature-compatibility>). The page now reads: "On Claude Fable 5.1, Claude Mythos 5.1, Claude Fable 5, Claude Mythos 5, Claude Mythos Preview, Claude Opus 5.5, Claude Opus 5, Claude Opus 4.8, Claude Opus 4.7, Claude Sonnet 5.5, and Claude Sonnet 5, non-default `temperature`, `top_p`, or `top_k` values return a 400 error on every request". Also <https://platform.claude.com/docs/en/models/opus-5-5/migration-guide> and <https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide>. | Pending: the docs say "a 400 error" without the reply text. |
| A16 | Claude Opus 4.1 and Haiku 4.5 (exact) | With both `temperature` and `top_p` set, `top_p` dropped as an unsupported hint (`` `{model}` does not accept temperature and top_p together; top_p was dropped ``). Now both are sent. | "Opus 4.1 does not allow both `temperature` and `top_p` parameters to be specified", <https://platform.claude.com/docs/en/release-notes/overview>; "Setting both returns a 400 error on Claude Haiku 4.5", <https://platform.claude.com/docs/en/models/haiku-4-5/migration-guide>. | Existing: "`temperature` and `top_p` cannot both be specified for this model" → subject `TopP`. |

## Kept, and overridable for the catalog

These stay in the adapter (wire formats or protocol facts), listed here so
the catalog knows which hook reaches them.

| ID | Rule | Kind | Hook |
|---|---|---|---|
| A7 | Forced tool choice refused with manual thinking (`type: "enabled"`) | Protocol fact: documented for every manual-thinking model | none needed |
| A15 | While thinking is on: `temperature` ≠ 1, `top_p` ∉ [0.95, 1] and `top_k` dropped | Protocol fact: documented for thinking as a whole. A model that differs gets a catalog *exception* | none yet |
| A8, A10a, A13, A21 | Budget vs adaptive style; `between_tools` on Sonnet ≥ 5.5; interleaving; newest format for unknown ids | Wire format | none yet |
| A11 | Effort levels: the 4.6 generation has no `xhigh` (sent as `max`); unknown ids get every level, so an effort is sent unchanged | Wire format; the list is negative data | `Provider::with_model_rules` → `ModelRules::effort_levels` |
| A17 | `thinking.display` only on adaptive models | Wire format | follows A8 |

The answers for the A5 ids are now `Unknown`, not `Supported`: the positive
claim covers only the generations documented to take forced tool choice.
Structured output stays `Unknown` for Mythos Preview, as before.
