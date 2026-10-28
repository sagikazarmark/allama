# ADR 0018: Extraction models are a sibling trait, unstable until a second provider; chat extraction takes its name

- Status: accepted

## Context

Cloudflare's Clairic (2026-10-19) is the first *extraction model*: a
single-pass model that reads an input and answers named, typed fields with
values instead of generating text. Each field asks for a text span, a
number, a date, time or datetime, a duration, a list, an estimate (a
distribution inside declared bounds) or a forecast (one estimate per step
ahead); each scalar answer is a value or null, with the probability that
the input does not determine it. It is the second class of non-generative
model after decision models, which `DecisionProvider` already covers. The
research behind this ADR is `docs/research/2026-10-27-extraction-models.md`.

Three facts shape the decision:

- **There is one backend, in preview.** Clairic is hosted-only on Workers
  AI, "an early checkpoint"; open weights are promised, not released. No
  inference server, SDK or gateway offers a second backend (LiteLLM's
  `/v1/extraction` proxies Cloudflare alone), and Clairic's documentation
  is inconsistent (the forecast answer is parallel arrays in the blog and
  a `steps` array in the schema) and incomplete (`p_below`/`p_above` and
  `truncated` are undefined; spans are "character offsets" with the
  encoding unstated). ADR 0001 requires a ready provider crate *and* two
  providers before a sibling trait, so that the core does not rename one
  provider's wire format.
- **Nothing in allama can hold it.** A decision's answer space is the
  finite option set the caller lists, checked by `DecisionResponse::validate`;
  an extraction's is continuous and typed, with null, spans and
  distributions. Chat extraction (`allama::extract`) drives a *chat model*
  through a response format or a forced tool; the new class generates no
  tokens. AGENTS.md: adjacent capabilities are not forced into chat-shaped
  APIs.
- **The name is taken.** allama's glossary owns "extraction" for the
  chat-based operation: extraction request, pass, failure, refusal,
  `ExtractExt::extract`, `ExtractionMode`, `ExtractionMetadata`, feature
  `extract`, namespace `allama.extraction`. The ecosystem already uses
  "extraction model" for the new class (Cloudflare, LiteLLM's
  `/v1/extraction`, the `typed-extraction` benchmark), as it settled on
  "decision model", which allama adopted over "System One".

A fourth, mechanical fact: `ExtractExt` is blanket-implemented for every
`ChatProvider` (`extract.rs:554`), so a trait method named `extract` on
any concrete type that implements both traits (the Worker provider, and
every wrapper around it) is `E0034: multiple applicable items in scope`
whenever both traits are imported, which the prelude does. Rust resolves
methods by receiver and name, not by argument types; the erased `Dyn*`
handles do not collide, but they are not the common path.

## Decision

### Unstable status for a sibling trait (amends ADR 0001)

1. **A sibling trait may ship on one provider when it is unstable.** ADR
   0001's two-provider rule is kept for stable traits. An unstable item
   differs from a stable one in four ways: it may ship with one provider;
   its serialized forms and persisted names are same-version (nothing is
   added to the pinned names); a change or removal in a minor release
   (`0.x` to `0.(x+1)`) needs a release note, not a migration path; and
   `allama-providers` lookups for it are unstable too, an exception to ADR
   0016's promise that a built-in key keeps its adapter. A minor release
   may already break stable items before 1.0; what the status removes is
   ADR 0001's gate and the pinning. It is recorded in a "Stability" note
   next to "Serialization and stability" in `allama`'s crate docs, in the
   module and trait docs, and in release notes; no Cargo feature or `cfg`
   marks it (see Rejected alternatives). The trait becomes stable, and ADR
   0001's rule applies to it from then on, when a second provider
   implements it and a release note says so.
2. **An unstable trait still pays its plumbing in full.** Dynamic form,
   forwarding impls, every wrapper (nine types and the forwarding macro),
   the mock, the adapter toolkit, the conformance aggregate and the
   registry are delivered with the trait, not after it: a wrapper without
   the impl stops being a provider of the operation, which breaks the
   common path. `assert_extraction_adapter` ships with required contracts
   (no third-party suite exists to turn red, ADR 0017) while its signature
   follows the trait's status, and the provider-authors guide gains an
   extraction section in the same release.

### Extraction models are `ExtractionProvider`

3. **The trait.** `ExtractionProvider: ProviderIdentity` with
   `extract_values(&ExtractionRequest) -> ExtractionResponse`,
   `extraction_capability(model, ExtractionCapability) -> CapabilitySupport`
   (required), `extraction_limits(model) -> ExtractionLimits` (default:
   none), and `validate_extraction_request` (default: a dry run through
   `adapter::extraction::prepare_dry_run`), the shape of `DecisionProvider`.
   `DynExtractionProvider`, `MockExtractionProvider`,
   `ExtractionValidation` and `adapter::extraction` follow the decision
   module one for one. The verb is `extract_values`, not `extract`, because
   of the method collision above, and because it names what comes back:
   chat extraction yields a `T`, an extraction model yields a value per
   field. The chat verb keeps `extract::<T>()`, the common path.
4. **The request.** `ExtractionRequest { model, input: String, images:
   Vec<Image>, fields: Vec<(String, Field)>, reference_date: Option<Date>,
   options: RequestOptions }`. The input is text, because text answers are
   spans of it; a caller with structured data serializes it, and spans
   index the serialization. Images are read beside the input, as in
   `DecisionRequest`. `reference_date` anchors dates written without a
   year. Every field parameter is semantic (decision 5), so, as for
   decisions, there is no `UnsupportedPolicy`; the one hint on the
   request, `Image::detail`, is warned about under every policy, as
   decisions do.
5. **Fields are a closed set of kinds with answer constraints.** `Field`
   is a `#[non_exhaustive]` enum over `TextField`, `NumberField`,
   `DateField`, `TimeField`, `DateTimeField`, `DurationField`,
   `ListField`, `EstimateField`, `ForecastField`, each with optional
   `instructions` and the parameters that define its answer: `integer`,
   `decimals` and a `unit` label on numbers; `integer`, `decimals` and a
   `DurationUnit` (seconds to years) on durations; the item kind
   (`ListItemKind::{Text, Number, Date}`) and `decimals` for number items
   on lists; required `min` and `max`, optional `step`, an
   `EstimateScale::{Linear, Log}` and a `unit` label on estimates; a
   `horizon` of steps, a `ForecastPeriod` (`Step`, `Minute`, `Hour`,
   `Day`, `Week`, `Month`, `Quarter`, `Year`), `min`, `max` and a scale on
   forecasts. Constructors take what is required and builders append the
   rest: `Field::number().decimals(2).unit("USD").instructions(..)`,
   `Field::estimate(10.0, 5000.0).log_scale()`. Each parameter changes
   what the model computes (the unit it answers in, the precision it
   rounds to, the grid and scale it estimates on), so none is a hint: a
   provider that cannot send a parameter as given refuses the field
   before sending, with `Unsupported`; nothing is dropped or folded into
   the instructions. `FieldKind` names the kinds, and
   `ExtractionCapability` is one variant per kind plus `ImageInput`, so a
   span-only endpoint (the community UltraBERT extractor) answers
   `Unsupported` for every kind it cannot produce.
6. **Answers are typed values with a null probability.** `Answer` is a
   `#[non_exhaustive]` enum over the same kinds. Scalar answers
   (`TextAnswer`, `NumberAnswer`, `DateAnswer`, `TimeAnswer`,
   `DateTimeAnswer`, `DurationAnswer`) hold `value: Option<T>` and
   `null_probability: Option<f64>`: the probability the input does not
   determine the value, as the model reports it, `None` when the provider
   reports no such statistic; `value` is the provider's determination
   either way, and the relation between the two is the provider's
   (Clairic's per-field threshold is a provider option). `TextAnswer` also
   holds `span: Option<TextSpan>`; `DurationAnswer` holds the request's
   `DurationUnit`. `ListAnswer` holds `items: Vec<ListItem>`, each a text,
   number or date value with an optional span; an empty list is an answer,
   not null. `EstimateAnswer` holds `value: f64` (the point estimate the
   provider reports), `median: Option<f64>` and `interval:
   Option<PredictionInterval { low, high, probability }>`, so an 80%
   interval is reported as one rather than assumed. `ForecastAnswer`
   holds `steps: Vec<EstimateAnswer>`, one per step ahead. Dates, times
   and datetimes are RFC 3339 text in validated newtypes (`Date`, `Time`,
   `DateTime`) with no date-time dependency in core and no place in the
   prelude; conversions to date crates can come as optional features.
   `ExtractionResponse` mirrors `DecisionResponse`: `answers: Vec<(String,
   Answer)>`, `id`, `model`, `request_id`, `provider`, `usage`, `warnings`,
   `provider_data`, and accessors by field name (`response.number("total")`).
7. **Spans are byte offsets into the request's input, verified.**
   `TextSpan { start, end }` indexes `ExtractionRequest::input` so that
   `span.text(&input)` is the extracted text, the convention ADR 0007 set
   for citation ranges. An adapter converts the provider's offsets (its
   unit, and its copy of the input when it reports one), and keeps the
   span when it lands on char boundaries and, for text values, equals the
   value. When it does not, the adapter searches the input for the value
   and keeps a unique match, or the match nearest the provider's offset;
   otherwise the span is `None` and the response carries a warning. A
   wrong span is worse than no span, and no span at all is no feature:
   the Clairic fixtures include a non-ASCII input with a repeated value.
8. **Provider statistics are provider data.** As for decisions,
   `confidence` is provider-defined (on Clairic it describes the guess,
   not the null decision) and lives in response provider data keyed by
   field name, typed by the provider crate; so do Clairic's per-answer
   `guess`, `mean`, `p_below`, `p_above`, `quantiles`, and any per-answer
   key the schema does not name. Its response-level `truncated`,
   `state_text` and `passes` are provider data too, and `truncated: true`
   is also a warning. Token counts that fit `Usage`'s breakdown contract
   (`text_input_tokens`, `image_input_tokens`, and `schema_input_tokens`
   as a new prefix, each a subset of `input_tokens`, checked against a
   multi-pass fixture) are breakdowns; a count that does not fit stays in
   provider data. Request tuning is a provider option: the per-field
   `null_threshold`, keyed by field name in the Cloudflare options type,
   and the request-level `include_quantiles` and `return_state_text`.
   Each is promoted to the portable model when a second provider reports
   it with the same meaning.
9. **Responses are checked, never repaired.** `ExtractionResponse::validate`
   requires exactly one answer per field, of the field's kind, and nothing
   else; list items of the field's item kind; no span on a null value;
   `null_probability` and `PredictionInterval::probability` in `[0, 1]`
   and `low <= high`; spans inside the input on char boundaries; a
   forecast with `horizon` steps; and an estimate's or forecast step's
   value, median and interval inside the field's bounds within the
   tolerance decisions use, since bounds are a constraint of the declared
   type, as options are for a choice; values are never clamped. Anything
   else is `Error::UnexpectedResponse`, which retries and fallback treat
   as they do today.
10. **Limits.** `ExtractionLimits { max_fields, max_images,
    max_forecast_horizon }`, queried per model, defaulting to none, as
    ADR 0010 shapes limits; adapters claim only what the endpoint fixes
    (ADR 0015). `ExtractionRequest::validate` checks structure and limits
    before any I/O: a blank model, a blank input with no images, no
    fields, blank or duplicate names, `min >= max`, a non-positive `step`,
    a zero `horizon`. Rules of one endpoint (Clairic's field-name grammar,
    inline-only images, image sizes, `decimals` up to 6) are the adapter's
    `InvalidRequest` or `Unsupported` before sending, not portable limits.
11. **A typed layer waits, and is chained, not fused.** An `Extraction`
    trait (`fields()`, `from_response()`) and `#[derive(Extraction)]`, the
    counterparts of `Decision`, are designed after the first adapter runs,
    unstable like the trait. The struct is the single source of the
    fields: `T::request(model, input)` builds the request and
    `response.decode::<T>()` (borrowing; `T::from_response` underneath)
    validates the answers against `T::fields()` and reads them, so a
    field name cannot drift between the two and leave an `Option` silently
    `None`. There is no fused `extract_values_as`: the chained form keeps
    the response in hand (spans, null probabilities, usage) without a
    wrapper, and a second way to do the same thing is not worth one saved
    repetition of the type. Giving `DecisionResponse` the same `decode`
    for symmetry with `decide_as` is a separate change.

### Chat extraction takes its name

12. **The chat-based operation is "chat extraction".** It is literally an
    extension of `ChatProvider` (the glossary: "the chat request an
    extraction derives its passes from"), and the qualifier follows the
    trait it extends. Renamed in the release that introduces
    `ExtractionProvider`: module `allama::extract` to
    `allama::chat::extract`; `ExtractExt` to `ChatExtractExt` (the one
    compiler-level collision, through the prelude); `Extractor`,
    `Extracted<T>`, `ExtractionMode`, `ExtractionMetadata`,
    `ExtractionPlan`, `ExtractionStream`, `ExtractError`, `ExtractFailure`
    to their `Chat`-prefixed forms, a vocabulary change rather than a
    collision, since they are module-scoped: a bare `ExtractionMode` on a
    request would read as the model class's; feature `extract` to
    `chat-extract`, with `extract` kept as an alias so no manifest breaks;
    the options namespace, error-metadata key and log fields
    `allama.extraction` to `allama.chat.extraction`, and the enforcement
    warning code `extraction.<enforcement>.<Type>` to
    `chat.extraction.<enforcement>.<Type>`, because extraction-model
    attributes and codes would otherwise land beside them. Kept:
    `SchemaEnforcement`, `AutoFallback`, `RefusalSource` (no collision),
    `Error::Extract` with its constructor and persisted kind `extract`
    (nothing collides with it, and ADR 0006 counts a persisted rename as
    a cost to stored errors and dashboards), and the methods
    `extract::<T>()` and `extract_stream::<T>()`. The glossary section
    becomes "Chat extraction", its "no separate extraction-only request
    type" sentence is revised, and entries are added for "Chat" (one
    assistant turn over messages, conversational or not; avoid
    "completion", "generation"), "Extraction model", and "Field
    (extraction)", reconciling the word with request fields and with the
    decision glossary's "avoid: field" for questions.
13. **`ChatProvider` keeps its name.** "Completion" names the legacy
    prompt-continuation API and an OpenAI endpoint; allama's traits are
    named by operation (chat, embed, decide, extract values).

### Where the first provider lives

14. **A Workers AI REST crate, `allama-cloudflare`.** Workers AI over HTTP
    has had no crate of its own: chat and embeddings are a preset of
    `allama-openai-compat`, decisions a preset of `allama-systemone`.
    Clairic speaks neither protocol, so the new crate hosts it: provider
    name `cloudflare` and namespace `cloudflare-workers-ai`, shared with
    `allama-cloudflare-worker` so provider data and options from the two
    transports are interchangeable; the `…/accounts/{account}/ai/run/{model}`
    URL, `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` (allama's
    existing names for Workers AI), the `{"result"}` and error envelopes,
    `allama-http` transport, per-request credentials, fixtures and e2e
    tests. Its options type and provider-data entries are defined once,
    and its codec is exposed `#[doc(hidden)]` under an exact-version
    requirement, as `allama_systemone::binding` is, for
    `allama-cloudflare-worker` to implement `ExtractionProvider` through
    the AI binding. The `cloudflare` registry key gains an extraction
    constructor beside its chat, embedding and decision ones. The
    `allama-systemone` preset keeps its own copy of the envelope code;
    consolidating the two is out of scope.

## Consequences

- allama gains a fourth operation, usable from any Rust application over
  REST and from Workers through the binding, with one provider; the
  ecosystem has no portable alternative to compare against yet.
- The unstable status is a new promise, recorded where readers look for
  stability. The chat-side rename is a breaking change in the same
  release, so there is one migration; it touches about seventy files
  across the workspace, most of them tests, manifests and docs.
- The plumbing is the size of the decision module's: around thirty files
  in core, the conformance aggregate, registry lookups, the mock, and an
  example.
- Clairic's documented inconsistencies are absorbed by the adapter and
  pinned by fixtures (forecast arrays and `steps`, list `confidence`,
  estimate `mean`, span offsets), not by the portable types.
- `Field`, `Answer`, `FieldKind` and `ExtractionCapability` are
  `#[non_exhaustive]`, so a date-valued estimate (Clairic accepts ISO
  dates as bounds) is a later, additive kind, not a change to `min`/`max`.
- Anchoring `Time` and `DateTime` answers (a reference time, a time zone)
  is a known gap, not a choice; `reference_date` covers what Clairic
  takes.
- Field parameters as constraints, `null_probability` as an option, and
  the response-level home of per-answer statistics are the choices most
  likely to move when a second provider arrives; the status exists for
  that.
- Structured input, streaming, batching unrelated inputs, partial
  answers, and per-field errors are out of scope until a provider needs
  them.

## Rejected alternatives

- **A provider-only crate first, no core trait** (the research note's
  recommendation). It defers the plumbing but also the design: writing
  the neutral model down now is what lets the adapter be checked against
  it rather than the reverse, and the name collision would be deferred,
  not solved.
- **Shipping the trait without amending ADR 0001.** The status records
  the exception as a rule.
- **A Cargo feature or a `--cfg` flag as the unstable marker.** Removing a
  feature later is itself a breaking change, and allama's features gate
  optional dependencies or opt-in surface, not stability. A `cfg` cannot
  be enabled by a dependency and would put `allama-cloudflare`, the
  binding, the registry and the aggregate behind a `RUSTFLAGS` setting
  for something meant to be tried.
- **Widening `DecisionProvider`** with value-typed questions. It breaks
  the invariant every decision provider and `DecisionResponse::validate`
  rely on (answers are probabilities over listed options), for a class
  Cloudflare itself separates from decisions.
- **Folding extraction models into chat extraction** as another
  `ExtractionMode`. They generate no tokens, take no chat request, and
  return spans, null probabilities and distributions that chat extraction
  has no place for.
- **Renaming the new class** ("prediction model", "value model") to dodge
  the collision. It would fight the term the ecosystem uses, as "System
  One model" would have for decisions.
- **`ExtractionProvider::extract` as the verb**, or renaming the chat
  method to `chat_extract::<T>()`. The first is `E0034` on every
  multi-operation type under the prelude; the second changes the common
  one-liner for the sake of a rarer call. `extract_fields` names the
  request's slots rather than the result, and `extract_data` describes
  chat extraction just as well.
- **Renaming only what the compiler rejects** (`ExtractExt` and the
  namespace). Cheaper, but it leaves `ExtractionMode` and
  `ExtractionMetadata` meaning chat extraction next to
  `ExtractionRequest` meaning the model class; the vocabulary rename is
  taken knowingly, with its cost.
- **A required `null_probability: f64`.** A span-tagging provider would
  have to invent one; and `f64` to `Option<f64>` breaks call sites later,
  while the reverse does not.
- **Per-answer provider data** for `confidence` and `guess`. Answers stay
  plain values; the decision module keeps provider statistics at response
  level keyed by name, and one shape for both is easier to learn.
- **Spans in the provider's units, unverified, or verified without a
  search fallback.** The first two panic or lie on non-ASCII input; the
  third drops every span on a provider that normalises its input.
- **Clairic in `allama-systemone`.** It is not the System One decision
  protocol; the crate's dialects, limits and provider data are decision
  ones.
- **The Worker binding as the only implementation.** No live tests, no
  fixture capture, and unusable outside Workers: the trait would ship with
  no implementation most applications can call.
- **`CompletionProvider`.** See decision 13.
