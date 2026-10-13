# ADR 0001: Chat is single-candidate; other operations are sibling traits

- Status: accepted

## Context

Some chat APIs can return several alternatives for one request (OpenAI's
request-side `n` and response `choices[]`, Gemini's candidate count). The
capability is real but not universal, and supporting it in the portable
core would reshape request modeling, response envelopes, stream event
identity, collectors, and replay formats at once.

`allama` also grows beyond chat: embeddings and decisions exist, and
transcription, speech, or reranking may follow. Each could be bolted onto
`ChatProvider`, or given its own trait.

## Decision

### Chat models one assistant turn

- One `ChatRequest` produces one `ChatResponse`; one chat stream is the
  lifecycle of one assistant turn. The portable core has no request-side
  candidate count and no response-side candidate list.
- When a provider returns several alternatives, the adapter selects the
  primary one (the bundled adapters take the first). Keeping the others,
  if ever needed, belongs in provider data or a provider crate.

### Other operations are sibling traits

- Adjacent operations get their own traits beside `ChatProvider`
  (`EmbeddingProvider`, `DecisionProvider`, and any future transcription,
  speech, or rerank trait). `ChatProvider` is never widened to carry a
  non-chat operation, and a non-chat operation is never modeled as a chat
  request.
- A new operation trait, or a concept promoted into the core, is designed
  only when a provider crate is ready to implement it, and is grounded in
  at least two providers whose semantics it covers. A third, differently
  shaped provider is used to check the model where one exists. One
  provider's wire format renamed is not a domain model.

## Consequences

- Streaming stays single-lane and replayable without candidate grouping,
  and tool calls and continuation stay centered on one turn.
- Adapters may discard secondary candidates; applications that need
  several alternatives make several requests or use a provider crate.
- Each operation states its own capabilities, limits, and validation
  without inheriting chat's. Every sibling trait costs its plumbing
  (a dynamic form, forwarding impls, support in every wrapper and the
  registry), which is why a trait waits for a ready provider crate and a
  second provider.

## Rejected alternatives

- **First-class multi-candidate chat.** It would touch the whole chat
  abstraction for a capability no concrete use case needs. Reopen it if
  an adapter or a library built on `allama` cannot work without portable
  candidates, and a cross-provider shape exists that keeps the common
  path as simple.
- **Keeping extra candidates on the response only.** It leaves request
  semantics, streaming, and replay underspecified.
- **One provider trait with optional methods for every operation.** It
  makes every provider claim (or stub) operations it does not serve, and
  forces non-chat operations into chat-shaped types.
- **Designing an operation ahead of a provider crate,** from documentation
  alone. Without a real adapter and a second provider, the model tends to
  copy the first provider's wire.
