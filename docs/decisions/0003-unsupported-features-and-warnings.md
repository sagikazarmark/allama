# ADR 0003: Unsupported features fail or warn; nothing is dropped silently

- Status: accepted

## Context

Providers accept different subsets of a portable request. An adapter that
quietly leaves out what its provider cannot take makes the caller unable
to tell what was honoured, and adapters that each decide on their own (one
refusing `seed`, another dropping it) make the same request mean different
things on different providers. allama needs one policy for what fails, what
warns, how a refusal and a warning are typed and worded, and how often a
recurring warning is reported, so callers can rely on it and third-party
adapters can follow it.

## Decision

### Principles

1. **Nothing is dropped silently.** A semantic field the provider cannot
   honour fails; a hint it cannot honour is left out with a warning, or
   fails under the strict policy.
2. **`Unknown` never drops.** `Unknown` means "send it and let the provider
   decide".
3. **An exactly equivalent encoding is not a degradation.** It is silent
   under every policy.
4. **A caller-set bound is never loosened.**
5. **A warning history replay can produce is coded and joined by code,**
   so a stored conversation does not repeat it on every turn.

### Semantic fields and hints

- **Semantic fields** change the output when lost: `response_format`,
  tools, `tool_choice`, embedding `dimensions`, content parts (images,
  files, audio, video) and embedding input parts a model cannot embed
  (ADR 0012). They fail with `Error::Unsupported` before the request is
  sent. Content is never dropped or joined.
- **Hints** do not change what was asked for: `seed`, penalties,
  `parallel_tool_calls: false`, image detail, prompt-cache settings and
  breakpoints, reasoning configuration, `strict` on a tool, and the
  embedding `input_type` (query or document). An unsupported hint is left
  out with an `Unsupported` warning, or refused under the strict policy.
- **`parallel_tool_calls: true` is permission, not a hint.** It is never
  warned about or refused, and an endpoint without the field drops it
  silently: the model may call several tools at once anyway.
- **Principle 3 in practice.** `ReasoningEffort::None` on a model without
  reasoning controls, `ToolChoice::Disabled` encoded by leaving the tools
  out on an endpoint that ignores `tool_choice`, and strict mode's
  nullable spelling of an optional property are all silent.

### The policy

- `UnsupportedPolicy` is `Warn` (the default) or `Error`. Chat and
  embedding requests carry an `Option<UnsupportedPolicy>`: `None` inherits
  from `UnsupportedPolicyProvider` (or `ProviderSettings::unsupported_policy`),
  and a policy still unset is `Warn`. Decision requests carry none: every
  field is semantic, and a policy field that governs nothing would fake a
  capability.
- The policy governs hints. It never refuses a replay-state drop (history
  the request did not ask to change) or a coded approximation (below).
- Every adapter applies it through `allama::adapter::Degradations`,
  starting with `chat::preflight` (or `embedding::preflight`), and reports
  encoding degradations into the same instance, which translates content
  indices back to the caller's. Refusals before sending go through it
  too, never a bare `Error::unsupported`, so they carry subject and stage.
- **Lossy hint or approximation.** A limitation that changes what the
  model is asked to act on (tool call arguments that are not a JSON object,
  sent as `{}`) is a `coded_lossy_hint`: an `Approximated` warning, refused
  under `Error`. One that changes only how the model is told something it
  still sees is a `coded_approximated`: recorded under every policy. A tool
  result marked as an error, sent to an endpoint with no error flag, is
  the second kind (`approximated.tool_result_error`): the model reads the
  same error text either way, and refusing it would make any stored
  conversation that ever held a tool failure unsendable under `Error`.
  `ChatRequest::push_tool_error` stays the portable call; the warning's
  remedy names `push_tool_result` for an application that wants its own
  encoding on that endpoint.

### `Unknown` never drops

- Preflight leaves an `Unknown` feature in the request, so the encoder
  sends every request field and content part or reports it through
  `Degradations`. A wildcard arm that skips one is a silent drop. A field
  the wire format cannot carry is `Unsupported` for every model.
- Some OpenAI-compatible servers ignore fields they do not know. A preset
  answers `Unsupported` for a field its server documents as ignored, or
  encodes an equivalent the server honours.
- Preflight records each capability the request relies on that answered
  `Unknown`; validation reports them (`Validation::relied_on_unknown`).

### A caller-set bound is never loosened

- `max_output_tokens` is the provider's total output cap, reasoning
  included (ADR 0005). An adapter fits reasoning into it, never the other
  way round, and never raises a cap the caller set. Two caller-set values
  that contradict each other are an error.
- Where the provider needs a reasoning budget below the cap (Anthropic's
  manual thinking: at least 1024 and below `max_tokens`), a budget derived
  from the reasoning effort is a hint: under an explicit cap it becomes
  `min(budget, max(cap / 2, 1024))`, an `Approximated` warning never
  refused (it loosens nothing), and at or below the minimum thinking is
  not enabled, as an unsupported hint. An explicit `budget_tokens` option
  is a setting, so one that does not fit an explicit cap is
  `Error::InvalidRequest`. The cap is raised to fit a budget only when the
  caller left it unset.

### Strict schemas

- **One policy, in core.** `allama::adapter::schema::apply_strict_policy`
  applies the rules below with any adapter's dialect (its grammar's
  rewrite). A dialect fails with `StrictSchemaError::NotRepresentable`
  (the grammar cannot express it) or `Invalid` (malformed, such as an
  unresolved `$ref`), which is always `Error::InvalidRequest`.
- **Strict degrades where an unconstrained form exists.** A tool or
  response format whose schema is not representable, or sent where strict
  is unavailable, goes out non-strict with the caller's original schema
  (minus `$schema`, never a partly rewritten one) and an `Unsupported`
  warning; under `Error` it is refused. A field with no unconstrained form
  (`Degradable::NoUnconstrainedForm`, such as Anthropic's
  `output_config.format`) fails under every policy. An endpoint that does
  not accept `strict` reports it as an unsupported hint.
- **Dialects warn only on success**, so the policy never keeps a warning
  about a rewrite it then sent non-strict. A rewrite that narrows what the
  model may produce (`oneOf` with overlapping branches as `anyOf`, a
  removed string `format`) is `Approximated`; one that changes nothing it
  may produce (annotations, a provably disjoint `oneOf` as `anyOf`, a
  `$ref` merged with its siblings) is silent. Neutral passes
  (`merge_sibling_refs`, `branches_disjoint`) are shared in core.
- **Per-request limits** that depend on the whole request (Anthropic's
  counts of strict tools and optional and union parameters) are checked
  before the policy runs: strict tools degrade from the last until the
  request fits, any overrun is refused under `Error`, and an output format
  that alone exceeds a limit fails, naming `RequestField::ResponseFormat`.
- **`ResponseFormat::json_schema` is strict; tools are not.** A response
  format is one per request and states the shape the caller wants, so it
  asks for constrained decoding. `non_strict()` means "constraining is not
  required", not "must not constrain". Tools are many per request, many
  endpoints have no strict tools, and OpenAI's strict rewrite changes what
  a raw caller parses, so `strict` on a tool stays opt-in.
- **Stand-in nulls.** Strict mode spells an optional property as
  nullable, so the model sends `null` where it would have left the key
  out. `normalize_strict_nulls` removes only those, judged from the
  caller's original schema: never a `null` the original admits, one on a
  required property, or a map entry. Extraction and
  `ToolCall::parse_arguments_with` apply it, with nothing from the adapter.
- **Unverified engines are approximate.** OpenAI-compatible endpoints
  report separately whether the response format and a tool's `strict`
  constrain decoding (`SchemaConstraint`). A custom endpoint's schemas
  get an `Approximated` warning that it is not known to enforce them; a
  preset is exact only where a provider doc promises enforcement.

### Extraction and enforcement

- `ExtractionMetadata::enforcement` is a `SchemaEnforcement`: `Exact`
  (constrained decoding over the caller's schema, with only rewrites that
  change nothing the target type can deserialize), `Approximate` (over a
  widened schema, or one sent unchecked to an engine that does not report
  what it enforces), or `NotEnforced`. It counts only the warnings on the
  mode's own field: `ResponseFormat` in native mode, `Tools` in forced-tool
  mode. `Exact` is the adapter's report, not a verified guarantee.
- Under `UnsupportedPolicy::Error`, extraction means "constrained decoding
  is in force, possibly over an approximated schema, or fail". A pipeline
  that needs more sets the opt-in floor (`Extractor::min_enforcement`),
  checked before sending where validation has `Encoding` coverage.
  `Approximate` is the practical portable floor.
- Without `Exact` enforcement, extraction rejects keys the schema of `T`
  does not declare, as a retryable parse error, and treats an object
  without `additionalProperties` as closed. This is deliberately stricter
  than JSON Schema and serde: a misnamed key otherwise becomes a silent
  default. `allow_undeclared_keys(true)` opts out.
- Enforcement is a warning event per provider, model and type (code
  `extraction.<enforcement>.<Type>`; `chat.extraction.<enforcement>.<Type>` since ADR 0018), not an entry in `response.warnings`.
- `Auto` mode prefers a verified mode to a guess: native when structured
  output is `Supported`, forced tool when both tool answers are, native
  when structured output is `Unknown`, forced tool when no tool answer is
  `Unsupported`, and fails only when the answers rule out both. It falls
  back on a `BeforeSending` refusal, and on a `Provider` refusal only when
  native output was a guess from `Unknown`.

### Refusals

- `Error::Unsupported` carries a `subject` (a request field, or content
  with the caller's indices) and a `stage` (`UnsupportedStage`):
  - `BeforeSending`: every refusal `Degradations` builds. It means nothing
    was sent; the C2 audit fails one that came after sending;
  - `Provider`: an upstream reply the adapter mapped with
    `Error::unsupported_by_provider`;
  - `Unspecified`: `Error::unsupported(message)`, which promises nothing,
    so a caller must assume the request may have been sent. An unknown
    stage reads as `Unspecified`.
- An upstream 4xx becomes `Unsupported { stage: Provider }` only through
  the adapter's own table from wire paths to request fields, for reply
  patterns a fixture pins. Anything else stays `InvalidRequest`. A nested
  wire path (`messages.[1]…`) is not mapped: its index is the wire's, not
  the caller's, since system prompts and lifted tool results shift it.
- `FallbackProvider` fails over at every stage; the failover warning's
  cause carries the stage.
- A refusal before sending reads `cannot send <subject>: <reason>`,
  semantic or hint alike.

### Model refusals and abnormal finishes

- **Refusal text is never answer text.** A refusal the model wrote is a
  `ContentBlock::Refusal`, left out of `ChatResponse::text()` and streamed
  as `StreamEvent::RefusalDelta`. It is output, so an error after it is
  neither retried nor failed over. An endpoint without native refusals
  gets it back as assistant text, without a warning: nothing is lost.
- A provider that declines without the model writing a refusal reports
  `FinishReason::Refusal` or `ContentFilter`, keeps its details as
  provider data, and stores any explanation under
  `provider_data::DECLINE_EXPLANATION_KEY` so core can quote it.
- Extraction fails a refusal with `ExtractError::Refused` before parsing
  and never retries it: asking again pays for the same refusal.
- **Abnormal finishes stay `Ok`**: the output before the cut-off is the
  caller's. `ChatResponse::finished_normally()` is the one check, false
  also when no finish reason was reported. A stream that closes without
  its terminal marker is `Stream { Incomplete }`; one that ends with its
  marker but no finish reason is `Ok` with a `stream.missing_finish_reason`
  warning, added by `Prepared::finish_stream` so no adapter decides it.
  Content filtering follows the transport: an HTTP error is
  `Err(ContentFiltered)`, a `200` is `Ok` with `ContentFilter` and any
  partial output.

### Capability answers

- **Three tiers.** A verified entry gets the table's answer. A parseable
  id in a known family gets the family default (a fine-tuned id is its
  base model). Anything else is `Unknown`, or `Unsupported` where a marker
  settles it. A family is never matched by a bare prefix.
- **Claims need evidence.** Adding or removing a `Supported` claim, or a
  refusal or drop rule, needs a captured fixture or a cited provider doc
  in its test. A rule not yet confirmed stays out, and the model falls to
  the third tier.
- **Overrides.** A resolver on the adapter (`with_chat_capabilities`,
  `ProviderSettings::chat_capabilities`) feeds its preflight and encoder,
  so it can upgrade as well as downgrade. `CapabilityOverrideProvider` can
  only downgrade or settle: an `Unsupported` answer it cannot lift gets
  `other.capability_override`. Embedding and decision answers have no
  override path.

### Validation is a dry run

- `validate_chat_request` and `validate_chat_stream_request` (and their
  embedding counterpart) give the call's answer: an `Err` is the error the
  call returns before sending, and `Ok(Validation)` holds exactly the
  request-side warnings the call reports. A request the call serves with a
  warning is not rejected by validation.
- First-party adapters run their whole pre-send path in it (preflight,
  option resolution, encoding, building the transport request) and report
  `ValidationCoverage::Encoding`; the default methods check capability
  answers only (`Capabilities`). Validation never checks credentials or
  that the model exists, and never emits warning events.
- The call enforces refusals itself; it never relies on the caller having
  validated.

### Warnings

- **Shape.** A `Warning` has a `kind` (`WarningKind`), a `subject`, the
  reporting `provider`, an optional `code` and a `message`. Warnings are
  on `ChatResponse::warnings` and `EmbeddingResponse::warnings`, and as
  `StreamEvent::Warning` at any point of a stream.
- **Codes** are stable names of a condition that recurs with the
  request's shape, listed in the `warning` module docs. `Degradations`
  gives a warning about a request field or system prompt `<kind>.<field>`
  (`unsupported.seed`) when its producer set none. A warning about this
  call alone (a failover, content the application sent: a participant
  name, a cache breakpoint, a file name) stays uncoded.
- **History warnings join by code** (principle 5). What a model or provider
  wrote into the conversation, and a limitation the conversation's shape
  repeats, is recorded through `dropped_replay_state`, `coded_lossy_hint`
  or `coded_approximated`, never `warn`. A request then carries one
  warning per code, its subject a `Subject::Messages` listing every
  message affected.
- **Replay state.** Another provider's reasoning, signatures, provider
  data and `Other` content, and reasoning a model cannot take back, are
  `Dropped` warnings coded by `ReplayState` (`dropped.foreign_reasoning`,
  `dropped.foreign_provider_data`, `dropped.foreign_content`,
  `dropped.content_replay`, `dropped.reasoning_replay`). For another
  provider's state the message points to `retain_provider_state_for`.
  A provider's own reasoning, on an endpoint that takes no reasoning
  back, is dropped silently, recognised by its namespace
  (`ProviderIdentity::reads_namespace`): that is what the endpoint does,
  not a degradation. A block with no provider data has no known origin and
  is still reported.

### How a warning reads

1. `Display` prefixes `[provider] ` (the reporting provider) and, for
   content, its location. The message repeats neither; it names a
   protocol, endpoint or provider only where that is the reason.
2. The message is `<limitation>; <outcome>[; <remedy>]`. The limitation
   reads true on its own, since a refusal gives it as its reason.
   Identifiers go in backticks; protocol and product names do not.
3. The outcome's verb follows the kind: `Unsupported` "`x` was not sent"
   (or "sent without `x`"), `Approximated` "sent as X" or "X was used
   instead", `Dropped` "it was dropped", `Fallback` "served by `X`".
4. A refusal leaves the outcome out, since nothing happened to the hint:
   `Degradations` cuts the part between the first and second `"; "` and
   keeps the remedy. The C2 audit (`claims_it_was_sent`) fails a strict
   refusal that says the hint was sent, ignored or dropped.

Code reads a message only to cut the outcome from a refusal; codes, kinds
and subjects are the machine-readable part.

### Recurrence

- With the default `warning-events` feature, every warning is also a
  `tracing` event, emitted once by construction: a private mark makes the
  innermost layer emit it and outer layers pass it on. Libraries depending
  on `allama` turn default features off, so the application decides.
- A coded warning's event goes out at `WARN` once per process and
  `(provider, model, kind, code)`, then at `DEBUG`; an uncoded one at
  `WARN` every time.
- `response.warnings` and `StreamEvent::Warning` still carry every
  occurrence. The application chooses what to show, with
  `Warning::recurs()` and an application-owned `SeenWarnings`.

### Conformance

- A capability refusal is accepted in two shapes only: a warning whose
  subject names the field or content, or `Error::Unsupported` at
  `BeforeSending` with nothing sent.
- `assert_capabilities_honest` probes every capability, whatever its
  answer, for each listed model and an unknown one, from exhaustive lists
  of capabilities, fields, content types and tool choices, so a new
  variant is probed or fails. Content must reach the body, be warned
  about, or be refused; a setting must change the body, be warned about,
  or be refused. The C2 request-field audit and the C3 negative audit
  apply the same rule.
- `assert_validation_agrees` checks validation against the call. Probe
  contracts run on `chat` and `chat_stream`, since each can lose a warning
  on its own.

## Consequences

Callers can tell what each provider honoured from the result alone.
Adapters agree because the policy, strict-schema rules and message form
live in core and the conformance contracts hold every adapter to them.
The cost is ceremony in each encoder: every field and content part needs
an explicit arm, and every new warning a code decision.

## Rejected alternatives

- **Deduplicating `response.warnings` in the library**, or warning only
  "since the last call". A response would stop saying what happened to
  its own request, observers and stored turns would miss warnings, and
  the library has no per-conversation state; a model switch would hide
  warnings true of the new model.
- **Warning about a tool error only on its first delivery** (the trailing
  tool-result batch). The trailing batch is not the first delivery, a
  conversation moved from a provider with an error flag would never report
  the lost flag, and validation of a stored history would report no loss.
- **Sending `{"error": text}` in band on endpoints with no error flag.** It
  is a convention, not a documented channel, so allama would rewrite the
  application's content, and it cannot wrap multi-part results.
- **A silent carve-out for all unsigned reasoning.** Reasoning of unknown
  or foreign origin dropped is still a drop; only a provider's own
  reasoning is silent.
- **Merging every block warning into one aggregate without subjects.**
  Joined warnings keep their subjects at message granularity.
- **Encoding maps as key/value arrays to fit strict mode.** It changes
  the wire shape of tool arguments, which fakes portability.
- **Stripping nulls at parse time without the schema.** Parsing sees only
  `T: DeserializeOwned`, and tool arguments would stay broken.
- **A yes/no "schema enforced" flag.** It would call every tagged enum
  (schemars writes `oneOf`) "not enforced" and an unchecked engine
  "enforced".
