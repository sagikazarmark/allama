# ADR 0006: Provider options are provider-declared types, given exactly or as JSON

- Status: accepted

## Context

Every provider has request settings the portable request does not model,
and vendors add new ones almost monthly. Applications set them in two
ways: in code, where a typed value catches mistakes at compile time, and
in configuration files, where only JSON is available. One request may
also reach several providers (fallback), so it has to carry options for
all of them without becoming generic over any one.

## Decision

1. **Each provider declares its option types**, implementing
   `ProviderOptions` with a `const NAMESPACE`. Callers attach the exact
   type (`with_option`) or namespaced JSON (`with_options_json`).
2. **Adapters read both through `RequestOptions::resolve`.** A typed value
   wins; otherwise the JSON is deserialized into the type, and unknown
   fields are `Error::InvalidRequest`. JSON is never forwarded to the wire
   as-is, so a typo is an error rather than a body field sent to the vendor.
3. **One options type per namespace.** `resolve` and `resolve_mut` fail
   with `InvalidRequest`, naming the namespace and both types, when a typed
   value of another type sits under the same namespace: an embedding
   options value on a chat request, or `allama-http` from two crate
   versions, whose transport options would otherwise drop a tenant's
   credential without a word.
4. **Options export losslessly, but never secrets.** Typed options must be
   `Serialize`, so a request, a recording or a fixture keeps its options
   (`insert_opaque` holds non-data values, skipped on export).
   `allama_http::HttpRequestOptions` omits credentials when serialized and
   its JSON form rejects them, so a serialized request never holds a key
   and a request rebuilt from one carries none (ADR 0009).
5. **The options namespace is not the provider-data namespace.** The
   provider-data namespace names the only provider that may read an entry
   (ADR 0002); the options namespace is the key an options type is read
   under. They usually match but need not: every OpenAI-compatible preset
   reads options under `openai-compat` and keeps state under
   `openai-compat:<name>`, so one options value configures every preset
   while state stays with the endpoint that minted it. Per-request
   credentials are addressed to a third key, the credential scope.
6. **Options under the provider's name are reported.** The model address,
   provider name and credential scope all use the provider's name, so a
   configuration file naturally writes `with_options_json("gemini", ..)`.
   Each first-party adapter whose name differs from its options namespace
   (`gemini` reads `google-ai`, `cloudflare` reads `cloudflare-workers-ai`,
   presets read `openai-compat`, `openai-compat-responses` or `systemone`)
   reports JSON under its own name as `dropped.provider_name_options`,
   naming the namespace to use. The Responses adapters report Chat
   Completions options the same way (`dropped.chat_completions_options`).
   A core-level "unread namespace" warning is impossible: a request behind
   a `FallbackProvider` carries other providers' options.
7. **allama's own keys follow three styles**, each meaning a kind of key:
   - `allama` is a provider-data namespace (`ALLAMA_NAMESPACE`) for what
     wrappers record, such as `FallbackAttempts`. It is spelled like a
     provider's namespace because it is one: no adapter sends it, and none
     treats it as foreign state.
   - `allama.<area>` names a core record by the area that owns it:
     application data (`allama.turn`), error metadata, the extraction
     options namespace (`allama.extraction`), span attributes
     (`allama.retry.*`, following OpenTelemetry's dotted names). Every key
     starting with `allama.` is reserved.
   - `allama-http` is the options namespace of the `allama-http` crate's
     transport options, named after its crate as a provider crate's is
     named after its provider. It is the one options namespace unrelated
     providers share; only transport code reads it, never request
     semantics, and nothing mints state under it.

   A new core record takes `allama.<area>`; options defined by another
   allama crate take the crate's name.
8. **Option types grow without breaking.** Every provider options type is
   `#[non_exhaustive]` with `new()` and a chained setter per field; fields
   stay public to read and assign, and only struct literals are ruled out.
9. **`extra` carries body fields the crate does not model**, merged into
   the request body (Gemini also has `generation_config_extra`). It never
   overrides a field the adapter owns: every field the encoder can send
   and every typed option is reserved, and a reserved key is refused with
   `InvalidRequest` before sending. The comparison ignores ASCII case,
   and on Gemini also underscores, since its JSON reads both `topK` and
   `top_k`. Each encoder has a test that every key of its body is reserved.
10. **One way to set a field.** A portable field is not duplicated in
    options (`top_k` is `ChatRequest::top_k` only). Typed options that
    refine a portable field (`reasoning_effort`, Responses `tool_choice`,
    Gemini `thinking_budget`) win, with an `Approximated` warning when the
    portable field is set too.

## Consequences

- One request reaches any number of providers, each reading only its own
  options; misspelled and misplaced options fail or warn instead of
  vanishing.
- Vendor fields are reachable on the day they ship, through `extra`, and
  become typed setters in any release without breaking callers.
- Options serialized by a newer crate with a new field fail to load in an
  older one, the same rule as any unknown field.

## Rejected alternatives

- **A generic request type (`ChatRequest<P::Options>`).** It breaks `dyn`
  dispatch, keeps one request from reaching several providers, and makes
  extraction, recordings and history generic.
- **Flattening `extra` into the options.** Serde cannot combine it with
  `deny_unknown_fields`, and the two want opposite things: a flattened
  map would turn every misspelled option into a body field and leak
  non-body options (`raw_response`, `budget_tokens`) under their own
  names. In JSON, `extra` is a nested object:
  `{"service_tier": "auto", "extra": {"container": "…"}}`.
- **Exhaustive option structs that grow only in minor releases.** Every
  vendor field would still break struct literals and need a release.
- **Renaming namespaces to match provider names** (`google-ai` to
  `gemini`), or allama's keys to one style: Gemini would have two
  namespaces, the styles would still mean different things, and each
  rename breaks stored histories, recorded errors or dashboards.
