# Gemini model claims

Model-specific rules that `allama-gemini` (chat and embeddings) does not
apply (ADR 0015). A model catalog can re-add them as data, through the chat
capability resolver, the embedding resolver and a Gemini model-rules hook.
Each row keeps the evidence the removed code cited, word for word where it
quoted a source.

Without the catalog the adapter sends each of these requests as the caller
asked, and the model's refusal comes back from the API.

## Removed rules

| ID | Trigger (ids, families, markers) | Old behavior | Evidence (moved from the code) | Upstream mapping |
|---|---|---|---|---|
| G2 | Image variants (`-image`, `nano-banana`: `gemini-2.5-flash-image`, `gemini-3-pro-image`, `gemini-3.1-flash-image`, `nano-banana-pro-preview`), TTS variants (`-tts`: `gemini-2.5-flash-preview-tts`, `gemini-2.5-pro-preview-tts`), Live variants (`-live`, `gemini-live*`, `native-audio`: `gemini-2.5-flash-native-audio-preview-12-2025`, `gemini-3.1-flash-live-preview`, `gemini-live-2.5-flash-preview`) | `StructuredOutput` and `PromptCache` `Unsupported`: a JSON Schema response format refused before sending (and extraction's Native mode skipped), cache hints dropped with a warning | Model pages, <https://ai.google.dev/gemini-api/docs/models> (`/models/gemini-2.5-flash-image`, `gemini-3-pro-image`, `gemini-3.1-flash-image`, `gemini-2.5-flash-preview-tts`, `gemini-2.5-pro-preview-tts`, the Live models): "Structured outputs: Not supported", "Caching: Not supported". | Pending. The existing `JSON mode is not enabled for models/<model>` pattern (seen for Gemma 3) maps to `Unsupported` on `ResponseFormat` if Google words it the same for these models; not captured for them. |
| G3 | Image and TTS variants (markers as in G2; not Live) | `ToolCalls`, `ParallelToolCalls`, `ForcedToolChoice` `Unsupported`: tools refused before sending, extraction's ForcedTool mode skipped | Same model pages: "Function calling: Not supported". Live models serve function calling on the Live API only, so they stayed `Unknown`. | Pending. The existing `Function calling is not enabled for models/<model>` pattern (seen for Gemma 3) maps to `Unsupported` on `Tools` if Google words it the same; not captured for these models. |
| G5a | TTS variants (any generation); Gemini 2.5 image variants (`gemini-2.5-flash-image`) | `ReasoningConfig`, `ReasoningOutput`, `ReasoningReplay` `Unsupported`: a reasoning config dropped with a warning (refused under `UnsupportedPolicy::Error`), reasoning blocks dropped from replay | Same model pages: thinking "Not supported" on 2.5 image and TTS models ("Supported" on Gemini 3 image models, which stays `Supported`, G5b). | Pending: no captured rejection text. |
| G7 | Text models of every Gemini generation (`gemini-<n>…` with no variant marker) | `ImageOutput` `Unsupported` (answer only; preflight never acts on it) | None cited in code. | Not applicable: answer-only, nothing is sent differently. |
| G8 | `Legacy` family: `gemini-1.*`, `gemini-2.0*` (e.g. `gemini-2.0-flash`, `gemini-1.5-pro`, `gemini-2.0-flash-thinking-exp`) | `ReasoningConfig`, `ReasoningOutput`, `ReasoningReplay` `Unsupported` in the answers, **and** the encoder dropped a reasoning config with an `Unsupported` hint warning ("this model has no Gemini thinking controls; the reasoning config was ignored") even when a resolver answered `Supported`. Only `ChatRequestOptions::thinking_budget` was still sent. | None cited in code (the family doc: "Gemini 1.x and 2.0: no thinking controls"). | Pending: no captured rejection text for `thinkingConfig` on a model without thinking. The adapter now sends the config as `thinkingBudget` (the format that predates `thinkingLevel`, a wire-format choice), from the 2.5 ladder without the 2.5 limits. |
| G15 | Every family but Gemini 3: `Gemini25` (`gemini-2.5-*`), `Gemma4` (`gemma-4-*`), `Legacy`; formerly also `Other` (unknown ids: `tunedModels/…`, `gemini-exp-1206`) | An image or file in a tool result refused before sending: "multimodal tool results require a Gemini 3 model (multimodal function responses)" (`d.reject`, every policy) | None cited in code; the README stated "Gemini 2.5 rejects them". To verify for the catalog: the multimodal function responses section of <https://ai.google.dev/gemini-api/docs/function-calling> (not cited by the removed code). The refusal was removed for `Other` first, before the rest. | Pending: no captured rejection text. |
| G16 | Every model | `frequency_penalty` and `presence_penalty` dropped with an `Unsupported` hint warning ("no current Gemini model enables penalties"), refused under `UnsupportedPolicy::Error`; never sent | The rejection text, "Penalty is not enabled for this model", verified live (no fixture); some earlier Gemini models accepted penalties. | **Added.** A `400` containing "Penalty is not enabled for this model" maps to `Error::Unsupported { stage: Provider }` on the penalty field the request set (`error.rs`: `PENALTY_NOT_ENABLED`, `name_refused_penalty`; tests `penalties_not_enabled_are_unsupported_by_the_provider`, `a_penalty_refusal_is_unsupported_by_the_provider`). A live fixture is still to be captured. |
| GE3 | Text embedding models: `gemini-embedding-001*`, `text-embedding-*`, `embedding-*` (after an optional `models/` prefix) | `EmbeddingCapability::ImageInput` and `CompositeInput` `Unsupported`: an image input, or an input of several parts, refused before sending (no override reached it) | None cited in code (the module doc: "`gemini-embedding-001` embeds one text per input"). | Pending: no captured rejection text. |
| GE4 | `gemini-embedding-2*` (e.g. `gemini-embedding-2`, `gemini-embedding-2-preview`); the original `embedding-001` (`embedding-*`) | `EmbeddingCapability::InputType` `Unsupported`: the portable input type dropped with a warning (core's role-prefix message: "`gemini-embedding-2` takes the role as a text prefix, not a field: start each query with `task: search result \| query: `"), refused under `UnsupportedPolicy::Error`; `taskType` not sent | Code comment quoting the guide: "Not supported on earlier models (`models/embedding-001`)"; the task goes in the text for `gemini-embedding-2`. Embeddings guide, <https://ai.google.dev/gemini-api/docs/embeddings>: "The `task_type` parameter does not apply to gemini-embedding-2", "include the task as an instruction in your prompt". | Pending: no captured rejection text (the guide's "does not apply" may mean the field is ignored, not refused). |

Totals: 9 rules removed; 1 mapped upstream (G16), 1 answer-only (G7),
7 pending.

## Kept in the adapter (not catalog data)

For reference, the model-dependent rules that stay:

- **Wire formats:** G11 thinking format (`thinkingLevel` for Gemini 3,
  Gemma 4 and unknown ids; `thinkingBudget` for 2.5 and, now, 1.x and 2.0);
  G12 the Pro and Flash clamps (Gemini 3 Pro `None`/`Minimal` → `LOW`,
  2.5 Pro `None` → 128, 2.5 Flash `XHigh` → 24576), applied only to ids
  that parse as those families, never to unknown ids; G13 Gemma 4 on or off;
  G14 the placeholder signature `skip_thought_signature_validator`, applied
  to unknown ids too, the newest behavior. The catalog's model-rules hook
  should be able to override these per id.
- **Protocol facts:** G17 (`ImageDetail` `Unsupported`,
  `parallel_tool_calls: false` cannot be honoured, cache key and retention
  are hints); GE7 (100 inputs per call, 6 images per input).
- **Positive or `Unknown` answers:** G1, G4, G5b, G6, G9, G10, G18; GE1,
  GE2, GE5, GE6.

## Plumbing for the model-rules hook

The encoder does not parse the model id: `encode_request` takes a
`ModelProfile` that `Provider::profile(model)` builds (`chat.rs`). Today it
holds only the family, which picks wire formats; the encoder has no
capability gate left for a resolver answer to lift, since every capability
answer acts through preflight. A model-rules hook (`with_model_rules`, as
the other adapters have) would plug in at `Provider::profile`.
