# ADR 0010: Provider identity, transparent wrappers, and call attribution

- Status: accepted

## Context

Some questions about a provider can only be answered by the leaf provider:
its name, which provider data it reads, and what each model supports. A
default answer (`"unknown"`, `CapabilitySupport::Unknown`) is merely vague
for a leaf provider, but wrong for a wrapper: a retry, tracing or custom
wrapper that forgot to forward it still compiled, then degraded silently.
Spans and errors named the wrong provider, validation let everything
through, and limits such as the embedding batch size were lost behind a
wrapper.

Failover adds a second problem. A wrapper reports the name of the provider
it wraps, so a `FallbackProvider` names its primary even when its fallback
served or failed, and telemetry must still say who did the work.

## Decision

### Name and namespace

1. **`provider_name` is a borrowed, open string** (`&str`), so names from
   configuration (a registry entry, an OpenAI-compatible endpoint named at
   runtime) need no leaked `String`. Builders take
   `impl Into<Cow<'static, str>>`, so static names stay free.
2. **The provider name and the namespace are different identifiers and may
   differ.** The name says who served a call, for people and telemetry. The
   namespace keys typed options and the provider data a provider mints and
   reads (ADR 0002, ADR 0006). Gemini reports `gemini` but stores state
   under `google-ai`; Cloudflare Workers AI reports `cloudflare` under
   `cloudflare-workers-ai`; OpenAI-compatible presets share the
   `openai-compat` options namespace and store state under
   `openai-compat:<name>`. Every provider crate exports both `NAMESPACE`
   and `PROVIDER_NAME`, even when they are equal.
3. **A provider answers `reads_namespace(namespace) -> bool`, not an
   origin.** It is a predicate because a wrapper answers for every provider
   that may serve its calls: a `FallbackProvider` reads what either side
   reads, which no single origin can name. Code that holds only a provider
   uses it to keep the state that provider needs
   (`history::retain_provider_state_for`).

### Required answers, transparent wrappers

4. **Identity and capability answers are required.** `provider_name`,
   `reads_namespace`, `chat_capability`, `embedding_capability` and
   `decision_capability` have no default. A leaf provider that does not
   know writes `CapabilitySupport::Unknown` explicitly; a wrapper that
   forgets to forward does not compile.
5. **Defaults stay only where they are right for a provider**:
   `downcast_any` (expose nothing), `embedding_limits` and
   `decision_limits` (no known limits), and the `validate_*_request`
   methods, which run the same preflight checks the call runs, so
   validation follows the request's `UnsupportedPolicy` exactly as the call
   would. Limits are `#[non_exhaustive]` structs (`EmbeddingLimits`,
   `DecisionLimits`), so a new limit is a field that a forwarding wrapper
   passes on, not one more defaulted method it can forget.
6. **Wrappers are transparent.** A wrapper forwards every question about
   the provider (name, namespaces, `downcast_any`, capabilities, limits,
   validation) and answers nothing for itself. Wrappers forward with the
   `allama::wrappers::forward_*` macros (ADR 0013), and a question added to
   a provider trait is added to its macro in the same release.
7. **A conformance contract checks transparency.**
   `allama_conformance::assert_transparent_wrapper` (and its per-trait
   variants) wraps a `TransparencyProbe` (a provider with a distinctive
   name, mixed per-model answers, model-dependent limits and a model it
   rejects) and asserts the wrapper gives the probe's answers. Every
   built-in wrapper, the `Dyn*` providers and `Box`/`Arc` run it; ADR 0013
   adds the data-path checks. Wrappers that legitimately change answers
   still pass:
   - wrappers over two providers pass over `(probe, probe)` and document
     how they combine: `FallbackProvider` takes the primary's name, gives a
     capability answer when both sides agree and `Unknown` otherwise, takes
     the smaller known limit, and validates the way the call would fail
     over;
   - `MapRequestProvider` validates the transformed request;
   - `MapModelProvider` answers for the model it sends, checked by
     `assert_transparent_model_mapping`.

### Attribution in telemetry

8. **Each span attribute is recorded once.** OpenTelemetry appends a field
   recorded twice as a second attribute, and lookups return the first, so
   a corrected value is silently ignored. The exporter tests fail on a
   duplicate key.
9. **`gen_ai.provider.name` is the addressed provider**, set when the span
   starts so samplers see it, as the semantic conventions ask.
   `allama.request.provider` repeats it.
10. **`allama.response.provider` names who the call happened on**, in the
    same value space, recorded once on every finished span:
    - when a provider served, that provider, whatever the outcome: a stream
      that reached its first output and then failed or was dropped
      happened there, and that provider incurred the tokens;
    - when nothing served, the provider that ended the call: for a failure,
      the provider its error names (`ErrorContext::provider`, the last
      side's when a whole fallback chain failed); for a drop, the last
      fallback side started. This keeps the attribute consistent with the
      span's `error.type` and `server.address`, so error metrics blame the
      provider that failed;
    - otherwise (an error a wrapper raised itself, such as a deadline, or a
      drop before any fallback side started), the addressed provider.

    The serving provider comes from the response when it names one, and
    from the call scope otherwise. Backends that read `gen_ai.provider.name`
    directly get the serving provider through a Collector `transform`
    processor that copies `allama.response.provider` over it; the shipped
    `docs/guides/observability/otelcol-genai-metrics.yaml` runs it in front
    of the metrics it derives.
11. **Resilience wrappers report facts to a call scope**
    (`allama::wrappers::call_scope`), not to span fields. `TracingProvider`
    installs one scope per call (a thread-local swapped in for each poll,
    so it works on `wasm32` without `unsafe`) and records one summary at
    completion or drop: `allama.attempts`, `allama.retry.*` and
    `allama.fallback.*`. Facts reach the innermost scope only; the first
    serving provider reported, the innermost, wins, so an outer fallback
    cannot overwrite the side that actually served. Work spawned onto
    another task is not seen.
12. **The call scope is internal except for transports.** Only
    `call_scope::record` and `CallFact::server` are public, so a transport
    reports the server each request went to. The other facts are reported
    by the built-in retry and fallback wrappers only; a third-party
    resilience wrapper appears as the attempts made under it. A public fact
    set would make every summary field a semver commitment; the rest can be
    opened in a minor release when a wrapper needs it.

## Consequences

- Every provider and wrapper writes out its identity and capability
  answers; the contract covers the methods that keep defaults.
- After a failover, the response model, id and usage on a span belong to
  `allama.response.provider`, not `gen_ai.provider.name`. Queries group by
  the former; the attribute name is slightly loose for a call with no
  response, accepted over adding an attribute the metrics would not use.
- The call summary never goes into provider data or `ErrorContext`: it
  describes the call, not the response or the error.

## Rejected alternatives

- **Keep the defaults and rely on the contract.** The contract only helps
  authors who run it; required methods catch the commonest mistake at
  compile time.
- **An `origin()` method** instead of `reads_namespace`: a wrapper over two
  providers has no single origin.
- **Recording `gen_ai.provider.name` at completion**, or rewriting it after
  a failover: samplers would not see it, and the second value would be lost.
