# allama documentation index

Where to find the reasoning behind allama, and how current each kind of
document is. API documentation lives on [docs.rs](https://docs.rs/allama)
and in the crate READMEs; the domain glossary is
[`CONTEXT.md`](../CONTEXT.md).

| Kind | Where | Status |
| --- | --- | --- |
| Guides | [`guides/`](guides) | Current. How-to material for library users and provider authors. |
| Decisions (ADRs) | [`decisions/`](decisions) | Current unless marked superseded. Each records one design choice and why. |
| Model catalog evidence | [`catalog/`](catalog) | Current. Model-specific claims the adapters do not apply (ADR 0015), with their evidence. |
| Research notes | [`research/`](research) | Dated. What was found about a question before a decision; a decision, if one follows, is an ADR. |

## Guides

- [Building servers and terminal apps with allama](guides/applications.md):
  storing conversations and starting a turn with `History::into_request`,
  application data and the turn envelope, an
  axum SSE handler, a chat in the terminal, the commit point, tool-call
  progress, failures after the first token, stopping when the client
  leaves, switching providers, rendering history, per-tenant keys,
  building providers from your own configuration, errors and logging, and
  testing handlers with mocks.
- [Observability with allama](guides/observability/README.md): the
  versions that work together, wiring `TracingProvider` to OpenTelemetry,
  content capture and redaction, a trace context propagation recipe,
  GenAI metrics in the Collector, and which provider served a call.
- [Writing an allama provider](guides/provider-authors.md): compat preset vs
  `CompatQuirks` vs a native adapter, options, provider data, warnings,
  streaming over `allama-http`, the conformance contracts, and registering
  with `allama-providers`.

More task-oriented material for application code (a wrapper stack for
providers built directly from adapter crates, fallback with model mapping,
extraction costs, embeddings at scale, persistence) is in the `allama`
crate documentation under "Common tasks".
Runnable examples are in [`crates/allama/examples`](../crates/allama/examples)
(no credentials needed) and
[`crates/allama-examples`](../crates/allama-examples) (real providers).

## Decisions

| ADR | Decision |
| --- | --- |
| [0001](decisions/0001-single-candidate-chat.md) | Chat is single-candidate; other operations are sibling traits |
| [0002](decisions/0002-opaque-provider-state.md) | Opaque provider state is namespaced and origin-gated |
| [0003](decisions/0003-unsupported-features-and-warnings.md) | Unsupported features fail or warn; nothing is dropped silently |
| [0004](decisions/0004-portable-reasoning-configuration.md) | Portable reasoning is effort plus output visibility |
| [0005](decisions/0005-usage-contract-and-output-cap.md) | Usage accounting contract and output cap |
| [0006](decisions/0006-provider-options.md) | Provider options are provider-declared types, given exactly or as JSON |
| [0007](decisions/0007-citations-on-text.md) | Citations are annotations on text |
| [0008](decisions/0008-provider-executed-tool-blocks.md) | Provider-executed tool blocks |
| [0009](decisions/0009-http-transport-seam.md) | An HTTP transport seam owned by the provider crates |
| [0010](decisions/0010-transparent-provider-identity.md) | Provider identity, transparent wrappers, and call attribution |
| [0011](decisions/0011-retry-after-give-up.md) | Retries wait out a provider's hint or give up, within a budget |
| [0012](decisions/0012-embedding-inputs.md) | Embedding inputs: one trait, ordered parts, modality per model |
| [0013](decisions/0013-middleware-as-wrappers.md) | Middleware is a wrapper; failover is visible but not a new error |
| [0014](decisions/0014-client-executed-non-function-tools.md) | Client-executed tools that are not function calls |
| [0015](decisions/0015-provider-level-capabilities.md) | Capabilities belong to the provider; model knowledge is an opt-in catalog |
| [0016](decisions/0016-model-addresses-and-the-builtin-registry.md) | Model addresses have a fixed grammar; the built-in registry is closed |
| [0017](decisions/0017-conformance-contract-lifecycle.md) | Conformance contracts are added in minor releases, and required only when adapters pass them |
| [0018](decisions/0018-extraction-models.md) | Extraction models are a sibling trait, unstable until a second provider; chat extraction takes its name |
| [0019](decisions/0019-http-vocabulary.md) | The transport seam speaks the `http` crate's vocabulary |
| [0020](decisions/0020-tower-services-as-transports.md) | A `tower` service is a transport through an adapter that keeps the seam's guarantees |

## Research notes

| Date | Question |
| --- | --- |
| [2026-10-27](research/2026-10-27-extraction-models.md) | Extraction models (Cloudflare Clairic): how the ecosystem integrates the second System One class, and whether allama should |

