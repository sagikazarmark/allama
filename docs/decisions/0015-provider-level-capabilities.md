# ADR 0015: Capabilities belong to the provider; model knowledge is an opt-in catalog

- Status: accepted (decisions 1, 2 and 4 are implemented; the catalog,
  decision 3, and provenance, decision 5, are pending)
- Evidence: [`docs/catalog/`](../catalog/)

## Context

Knowledge about which model supports what is a snapshot of provider docs,
and only an allama release updates it. A stale entry fails in one of two
ways, depending on its direction:

- A stale **positive** or `Unknown` answer lets the request through, and
  the provider rejects it. That maps to `Error::Unsupported { stage:
  Provider }`, the same match arm as a refusal before sending: late, but
  typed, and the user is not stuck.
- A stale **negative** answer ("this model has no tools", "drop
  temperature while reasoning") refuses, drops, strips or clamps something
  the model can now do. The user cannot use it until they upgrade allama
  or install an override, and a model change on the provider's side can
  turn a correct rule harmful with no change in the user's code.

So negative knowledge about models is what goes stale in a way that
breaks users. Hardcoded knowledge falls into four kinds:

| Kind | Example | Changes with | If stale |
|---|---|---|---|
| Protocol fact | Anthropic has no `seed`; Ollama `/v1` ignores `tool_choice` | the provider's API (the adapter changes anyway) | Fixed in the adapter. |
| Wire format | budget vs adaptive thinking; Gemini `thinkingBudget` vs `thinkingLevel` | the model | Late provider 400, typed. |
| Negative model claim | "no image input", "refuse forced tool choice" | the model | Feature blocked until an upgrade. |
| Positive or `Unknown` claim | "this family supports tools" | the model | Late provider 400, typed. |

## Decision

1. **Capability answers describe the provider's endpoint, not the model.**
   - An adapter answers `Unsupported` only for what its API cannot carry
     for *any* model, including what its encoder always refuses: the
     Responses API refuses every audio part, so `AudioInput` there is
     `Unsupported` for every model.
   - A model-dependent capability answers `Supported` where the model is
     documented to support it, and otherwise `Unknown`, unless a
     capability resolver answers it. `Unknown` means send it and let the
     provider decide (ADR 0003).
   - Messages built from an answer name the provider, not the model ("image
     input is not supported by `ollama`"), even when a resolver settled
     the answer. Messages from a model-rules hook are model claims and
     name the model.
   - A capability covers only what one answer can be true for.
     `StructuredOutput` covers JSON Schema response formats, not JSON mode:
     Anthropic takes a schema and has no JSON mode, DeepSeek the reverse.
     JSON mode has no capability; preflight never refuses it, and an
     adapter whose API lacks it refuses with `Error::Unsupported`. An
     OpenAI-compatible dialect without `json_schema`
     (`supports_json_schema: false`) answers `StructuredOutput`
     `Unsupported` for every model. Within a `Supported` answer, accepted
     media types and sources may still vary, and the adapter refuses the
     rest with `Error::Unsupported`.
2. **By default, nothing is refused, dropped, stripped, clamped or warned
   away because of the model id.** Only protocol facts and the caller's
   `UnsupportedPolicy` refuse or drop. The exceptions:
   - **Wire-format choices.** An adapter picks the encoding a model takes
     (ADR 0004). An id it does not recognize gets the newest documented
     format, and so does a known family at a version the adapter was not
     written for: older formats belong to an enumerated list of legacy
     models (on Anthropic, the budget form is for Claude 3, Claude 4.0 and
     4.1, Opus 4.5, and Sonnet and Haiku 4.5), because a newer model of a
     known family may refuse the older format (Claude Haiku 5.5 does). A
     caller overrides the choice with the adapter's model rules
     (`with_model_rules`) or a typed option. This is encoding, not a
     capability claim.
   - **Caller-set bounds** are never loosened (ADR 0003). Where fitting a
     reasoning budget under the output cap depends on the wire format, it
     follows the format chosen above.
   - **Negative answers from a resolver** are honored: a caller or
     catalog that says a model cannot do something gets refusals and drops
     before sending.
3. **Model knowledge lives in an opt-in catalog** (pending).
   - A separate crate holds negative model claims and model-specific rules
     as **data**, each with evidence: a provider doc URL or a captured
     fixture. The rows in [`docs/catalog/`](../catalog/) are its input.
   - It plugs into existing seams: capability resolvers
     (`ProviderSettings::chat_capabilities`, `with_chat_capabilities`,
     `CapabilityOverrideProvider`) and the model-rules hooks. These seams
     are chat only; an embedding or decision catalog adds its own seam.
   - It is versioned and released apart from the adapters, so a data
     update is a `cargo update`. It can also be loaded from JSON at
     runtime, or refreshed from model APIs that publish capabilities
     (Anthropic `GET /v1/models`, Gemini `models.list`, Ollama
     `/api/show`).
   - An app that installs it opts into the staleness risk knowingly; an
     app that does not is never blocked by stale model data.
4. **Upstream refusal mapping is the primary path without a catalog.**
   Provider replies that reject a parameter or model feature map to
   `Error::Unsupported { stage: Provider, subject }`, through patterns
   pinned by fixtures and kept current by a scheduled live check. A missed
   pattern degrades to `InvalidRequest`, still an error.
5. **Validation says where an answer came from** (pending).
   `Validation::relied_on_unknown` lists model-dependent capabilities
   nothing settled. A catalog answer will be reported as coming from the
   catalog, so a startup check can tell "the provider can't" (a protocol
   fact) from "the catalog says this model can't" (data that may be stale).

## Consequences

- Model changes on the provider's side never break user code through
  allama, and new model capabilities work the day they ship.
- Without a catalog, model-specific gaps fail late, with a typed,
  usually unbilled `stage: Provider` refusal. The applications guide
  recommends the catalog to apps that want early refusals.
- Extraction's auto mode chooses from endpoint-level answers: with
  `StructuredOutput` `Unknown` it tries native structured output and falls
  back to a forced tool call on a `Provider`-stage response-format refusal.
- The conformance honesty contract checks that no refusal is based on the
  model; with a catalog it will also check that every catalog claim is
  honored on the wire.
- The catalog needs upkeep: a scheduled drift check against provider model
  APIs and community datasets (models.dev, LiteLLM) that reports, with a
  person reviewing each change. Nothing is generated automatically.

## Rejected alternatives

- **Model tiers in the adapters** (verified entries, family defaults
  parsed from the id, `Unknown` for the rest), with negative rules such as
  dropping sampling while reasoning. Accurate on release day, but every
  negative entry blocks users once the model changes.
- **A `JsonMode` capability**, or wire-support facts telling JSON mode
  from a schema: new surface for a legacy feature that Workers AI and
  DeepSeek still need only as `ResponseFormat::Json`.
