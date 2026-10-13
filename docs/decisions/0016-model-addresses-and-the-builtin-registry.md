# ADR 0016: Model addresses have a fixed grammar; the built-in registry is closed

- Status: accepted

## Context

`allama-providers` resolves a model address such as `openai:gpt-5.5` to a
provider and a model id. Addresses live in configuration files and
environment variables, so an address must keep resolving to the same
provider, model and API across releases. Any leniency is a trap: trimming
or case-folding added later silently changes what an existing
configuration resolves to, and tightening later breaks configurations that
work today. So the grammar is pinned, and the registry's openness, the
lookup's return shape and the meaning of each key are decided with it.

## Decision

1. **The grammar is fixed.** A model address is `key:model`, split at the
   first `:`.
   - A key matches `[a-z0-9][a-z0-9-]*`. `Registry` panics when asked to
     register any other key: registration is code, so a bad key is a
     programming error.
   - The model id is everything after the first `:`, sent unchanged, so
     it may hold further colons (`ollama:qwen3:1.7b`). It is not empty
     and does not start with `:`.
   - Whitespace and control characters are refused anywhere, with
     `Error::Configuration`. Nothing is trimmed or case-folded. A lookup's
     error suggests the address it most likely meant when that names a
     registered key (`OpenAI:gpt-5` and `openai/gpt-5` suggest
     `openai:gpt-5`), so a typo fails at startup with a hint.
   - `ModelAddress` is the public parser, and the lookups use it, so an
     application validates a configured address the same way before it
     builds anything.
2. **Lookups return a `(provider, model)` pair.** It destructures straight
   into a request, the common case. Anything richer (the key, the original
   address for logs) comes from `ModelAddress`, which an application
   parses itself when it needs it.
3. **The built-in registry is closed.** The free functions
   (`chat::from_model`) resolve only built-in keys. An application that
   uses a third-party provider builds a `Registry` (`Registry::builtin()`
   plus `register_chat_with`) and passes it to the code that resolves
   addresses. Global registration would let any dependency change what
   another library's `chat::from_model` resolves to, and would need an
   ordering rule for conflicting registrations. A process-wide hook can be
   added later without breaking anything, if a concrete use case appears.
4. **Model addresses are trusted configuration.** An address picks the
   server that receives the request and the API key sent with it. The
   local presets (`ollama`, `vllm`, `litellm`) need no configuration and
   reach `localhost`, so an address an end user chooses can make a server
   call its own loopback ports. The presets keep their zero-configuration
   defaults; the docs tell applications to map a user's choice through an
   allowlist of configured addresses.
5. **A built-in key keeps its adapter and endpoint** in every 0.x release.
   An address in configuration means the API it reached when it was
   written, so a vendor's new API gets a new key and never takes over an
   existing one.
   - `gemini` stays on `generateContent`, which Google calls legacy but
     fully supported. `gemini-interactions` is reserved for an adapter for
     its Interactions API, which does not exist yet; nothing else will be
     registered under it.
   - This is asymmetric with OpenAI on purpose: `openai` is the newer
     Responses API and `openai-chat` the older Chat Completions, because
     both existed when the keys got their meaning and the recommended one
     took the bare key. Gemini's newer API arrived after `gemini` was
     settled.

## Consequences

- An address resolves to the same provider, model and API in every 0.x
  release. Loosening the grammar (a new key character) is additive;
  tightening it is a breaking change.
- Configurations with stray whitespace or capitalized keys fail at startup
  instead of reaching the provider.
- Libraries that resolve addresses should take a `&Registry` rather than
  call the free functions, so applications can add third-party providers.

## Rejected alternatives

- **A `#[non_exhaustive] struct Resolved<P> { provider, model }`** as the
  lookup result: it costs every caller a field access to serve the rare
  one that needs more.
- **Registering `gemini-generate-content` as an alias now**, so `gemini`
  could move to Interactions later: it adds a key with no user and still
  leaves `gemini` to change meaning.
