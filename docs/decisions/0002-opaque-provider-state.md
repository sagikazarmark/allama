# ADR 0002: Opaque provider state is namespaced and origin-gated

- Status: accepted

## Context

Providers require some state to be replayed verbatim in later turns:
Anthropic thinking signatures and redacted thinking, Gemini thought
signatures on function calls (Gemini 3 rejects a replay without them with
HTTP 400), OpenAI Responses encrypted reasoning and item ids. That state
means nothing to any other provider, and sending it elsewhere fails or
misleads. Libraries that keep such state without recording who produced it
break when history moves between providers (LiteLLM #32140, #37849).
Conversations also outlive the build that wrote them, so the stored form
of this state needs rules of its own.

## Decision

1. **Every block, part, message, tool and response carries
   `ProviderData`**: a serializable map of namespace to JSON object, with
   typed access through `ProviderDataEntry`. Content kinds are
   `#[non_exhaustive]` structs, so a new field is additive.
2. **The namespace names the only provider that may read an entry.** For
   state a provider minted, that is its origin; for a setting a caller
   attaches to a request item (a tool definition, an embedding input, the
   OpenAI participant name under `PARTICIPANT_NAME_KEY`), it is the
   addressee. Adapters write and read only their own namespace, so gating
   is structural: foreign state is never sent. An empty namespace is the
   same as an absent one.
3. **Dropping foreign state that matters is reported** with a `Dropped`
   warning (ADR 0003), under every `UnsupportedPolicy`, so failover works
   even under `Error`. Gemini substitutes its documented dummy signature
   for a function call that has no Gemini signature.
4. **`Other` content keeps provider shapes allama does not model.**
   - It serializes as `{"type": "other", "type_name": …, "origin": …,
     "data": …}`. `"other"` is a reserved kind and the wire fields nest
     under `data`, so a wire type named like a native kind (a Gemini raw
     `text` part) cannot be misread as one, and a kind added later cannot
     collide with stored wire data. Wire fields beside `type` fail to
     load rather than being guessed at.
   - It also carries `ProviderData`, for state the adapter keeps about it
     (a Gemini thought signature), apart from the wire fields.
   - A foreign-origin `Other` part is dropped with a `Dropped` warning,
     like foreign reasoning. One with **no origin** is the caller's own
     raw part and is sent verbatim wherever the wire can hold it.
     Own-origin content an adapter does not model is replayed verbatim.
   - Endpoints that share a namespace replay own-origin `Other` content
     only for types each one knows. OpenAI Chat Completions and Responses
     both write `openai`, so Chat Completions sends only its own part types
     (`text`, `image_url`, `input_audio`, `file`), lifts custom tool
     answers into tool messages (ADR 0014), and drops Responses-only items
     with a warning.
5. **A client tool answer travels with its call.** When a provider-specific
   client tool call (ADR 0014) in the request is not replayed, its answer is
   dropped with a `Dropped` warning, so a provider never receives an output
   for a call it was not sent. An answer whose call is not in the request is
   sent only where the provider holds earlier turns server-side (Responses
   `previous_response_id`); elsewhere it is dropped with a warning. Portable
   `ToolCall`s and tool messages are never dropped by pairing; a misplaced
   one is a `history::validate` issue to fix.
6. **Providers answer a read predicate, not an origin.**
   `ProviderIdentity::reads_namespace` says whether a provider reads a
   namespace, and `history::retain_provider_state_for` keeps exactly that
   state when a conversation moves for good. The serving provider is
   stamped in `ChatResponse::provider`. The predicate is required, with no
   default: `false` would make a wrapper that forgot it strip every
   provider's state, and comparing with the provider name is wrong for
   Gemini (`gemini` reads `google-ai`), Workers AI and every
   OpenAI-compatible preset (`openai-compat:<name>`). `FallbackProvider`
   answers primary *or* fallback, since either may serve the next turn.
7. **Provider files are content, not replay state.**
   `MediaSource::ProviderFile { origin, id }` is gated like other content,
   but a provider that does not read its origin **refuses** the request
   with `Error::Unsupported` before sending, under every policy: the id
   means nothing elsewhere, and dropping it would answer without the
   document. `chat::prepare` applies this for every adapter, and
   `retain_provider_state` never removes provider files.
8. **`app_data` belongs to the application.** Messages and system prompts
   carry a JSON map the application owns (message ids, timestamps,
   per-turn usage). Adapters never read, send or warn about it, and
   `retain_provider_state` never touches it. It is not a namespace inside
   provider data, because adapters treat every namespace but their own as
   foreign state to drop or report. Keys starting with `allama.` are
   reserved (ADR 0006 lists allama's key styles).
9. **A history stores messages only.** The system prompt is set on every
   request; a conversation that keeps its own prompt stores what rebuilds
   it (the text, a persona id) beside the history. Storing it inside
   would make a resumed conversation that sets its prompt again send it
   twice, and would put a `SystemPrompt`'s unversioned option fields into
   stored data. A stored `system` key fails to load with an error that
   names it.
10. **The history format evolves by version** (`HISTORY_FORMAT_VERSION`):
    - Saving always writes the current version, whatever was loaded;
      `History::loaded_format_version` reports what was read. Writing the
      loaded version would label newer content as older.
    - Every change to the persisted shape bumps the version, additive ones
      included, because an older build ignores an unknown field on load and
      loses it on save. A build reads every version up to its own; newer
      versions, and a content `type` it does not know ("written by a newer
      allama?"), fail to load. A rollback across a bump needs a migration.
    - A bump ships only in a minor release: since saving writes the current
      version, a bump in a patch would let a `cargo update` leave older
      pods of a rolling deploy unable to read what newer ones save.
    - Provider data keys an adapter writes are part of the stored format,
      but the core version does not cover them: an adapter keeps reading
      every shape it ever wrote.
    - A test compares the key paths of a fully populated history with a
      pinned list, and the golden fixture's kinds are matched
      exhaustively, so a new field or kind fails until the version is
      bumped.

## Consequences

- Conversations are safe to move between providers (fallback, routing)
  without the caller knowing which state belongs to whom.
- Signatures and response metadata live in provider data, which survives
  persistence, rather than in typed in-memory fields.
- Namespace gating separates providers; it does not tell an application's
  state from a forgery. Histories from end users need
  `history::retain_portable_content`.

## Rejected alternatives

- **A trait-level `origin()` or `namespace()`.** A wrapper over two
  providers has no single origin (ADR 0010). A set-valued `namespaces()`
  allocates and makes every wrapper merge sets, where a predicate composes
  with `||`.
- **An application namespace inside provider data** instead of `app_data`
  (point 8).
