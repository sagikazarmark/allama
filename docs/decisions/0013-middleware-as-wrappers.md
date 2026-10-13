# ADR 0013: Middleware is a wrapper; failover is visible but not a new error

- Status: accepted

## Context

Wrappers are the extension seam (ADR 0010): a type that implements the
provider traits over an inner provider and passes the transparent wrapper
contract. `MapRequestProvider` covers a synchronous request rewrite, but
real middleware needs an async request hook (a memory or retrieval lookup)
and a way to observe responses and streams. The questions: does
middleware need a concept of its own, how does a hand-written wrapper stay
transparent as the provider traits grow, and how do wrappers that call
more than one provider (retry, fallback) report what happened without
changing what callers match on. Prior art is the Vercel AI SDK's
`wrapLanguageModel`, where one wrapper owns all forwarding and the
middleware supplies optional `transformParams`, `wrapGenerate` and
`wrapStream`.

## Decision

### Middleware is a wrapper, written from a recipe

There is no `Interceptor` or middleware concept next to wrappers. The
reference is `crates/allama-examples/examples/middleware.rs`, which runs
`assert_wrapper`. Rules a hand-written middleware follows:

- **An async hook edits an owned copy of the request and never changes the
  model**, so capability answers from the inner provider stay right for
  the caller's model. Models are mapped with `MapModelProvider`.
- **Validation is synchronous and the hook is not**, so the wrapper
  validates the caller's request. That is sound only while the hook is
  capability-preserving (it adds no tools, response formats or media the
  caller did not ask for); a hook that cannot promise this validates what
  it sends, as `MapRequestProvider` does.
- **What a wrapper changed is a coded warning of the matching kind** (a
  redaction wrapper: `Approximated`, code `approximated.redacted_content`).
  Pure observations go to `tracing` or the application's own channel. A
  wrapper never writes provider data: not under `allama`, which is
  reserved for conventions allama publishes (ADR 0002; `FallbackAttempts`
  is the one exception), not under a provider's namespace, and not on
  blocks. An annotation that is not a degradation has no portable channel
  yet; that is not a reason to borrow a namespace.
- **Headers and credentials go through the transport seam**: a wrapper
  adds `allama_http::HttpRequestOptions` to the request, and each HTTP
  adapter applies them with its own `CredentialSlot`, so in a fallback
  chain each provider sends only the keys addressed to it (ADR 0009).

### Forwarding is done by macros

A method added to a provider trait must have a default, and a default is
right for a provider and wrong for a wrapper. A wrapper that copied a
forwarding snippet before the method existed compiles warning-free and
silently stops being transparent. So `allama::wrappers` publishes
`forward_provider_identity!`, `forward_chat_questions!`,
`forward_embedding_questions!` and `forward_decision_questions!`, which
expand inside the wrapper's `impl` to every question about the provider,
through a field. A wrapper writes only its operations and the macros.

**A question added to a provider trait is added to its macro in the same
release**, next to the `TransparencyProbe` check ADR 0010 requires.
Built-in transparent wrappers and the example use the macros. A wrapper
that answers a trait's questions differently (`MapModelProvider`,
`FallbackProvider`, `CapabilityOverrideProvider`) writes that trait's
questions by hand.

### The data path is part of transparency

A transparent wrapper also passes through what the provider answers:
warnings in order (it may add its own), provider data, usage, ids, every
stream event in order, and every error with what a caller decides on
(kind, status, retry hint and directive, request id, provider code,
refused subject and stage). It adds context to an error with
`Error::with_message_prefix`; re-wrapping loses the kind. `assert_wrapper`
(and its per-trait variants) checks the questions and the data path in one
call. A wrapper that rewrites or annotates responses beyond adding
warnings is not transparent and runs only `assert_transparent_wrapper`.

A wrapper that adds warnings builds on `allama::adapter::chat::prepare`
(or its embedding and decision counterparts), so the inner provider's
warnings are marked emitted and only its own are emitted again. Response
caches belong outside such wrappers: a cached response's warnings lose
their mark and would be logged again on every hit.
`CapabilityOverrideProvider` is built this way against its own answers;
like a fallback pair it claims no more than the side that serves, so it
never upgrades the wrapped provider's `Unsupported`. It has only a chat
form, because an opaque `DynChatProvider` is the one place no
builder-level override reaches.

### `Layered<P, M>` is the planned end state, not built yet

One wrapper type will own all forwarding; the middleware supplies only
behaviour through `ChatMiddleware`, `EmbeddingMiddleware` and
`DecisionMiddleware`, whose methods all have defaults so later ones are
not breaking. Its shape is settled so the first release ships it whole:

- The next layer is an erased `ChatNext` that can only run the call and
  read the inner provider's name and capability answers, so a middleware
  is written once for every provider and cannot bypass layers.
- `transform_request` receives a `ChatCallKind` (unary or streaming), the
  one signal a middleware needs to treat the two calls differently. It is
  not a request field, which would leak a call detail into the portable
  request and recordings.
- `Layered` validates the caller's request (the async hook rule above) and
  rejects a changed model with `Error::InvalidRequest`.
- Erased `Dyn*Middleware` forms ship with it, built like
  `DynChatProvider`, so a stack chosen at run time folds into nested
  layers.

It lands additively once the trait shape has met a few real middlewares
written with the recipe.

### Failover is a warning, and a failed failover is its last error

- **What fails over.** `should_fallback` accepts rate limits, quotas,
  timeouts, outages, `Unsupported`, and `ContextLengthExceeded` (context
  windows are not exposed uniformly, so the backup is worth one call). It
  refuses `Auth`; `ContentFiltered`, since sending a blocked prompt to
  another vendor is a policy decision, not a recovery; `Rejected`, since a
  local policy refusal applies on every side; a client-closed request
  (`499`); and errors in the request or response themselves. `fallback_if`
  recipes opt in to `ContentFiltered`, and to `Auth` with a key per
  vendor. The server retry directive never affects failover (ADR 0011).
  Retry inside each side, not around the pair, which would repeat the
  whole failover.
- **A pair validates the way it would call**, through its classifier: the
  primary's validation, with the backup's own dry run in
  `Validation::fallback()`; or, when the primary refuses with an error the
  classifier fails over on, the backup's validation plus the failover
  warning. Capability answers are the agreement of both sides, else
  `Unknown` (ADR 0010).
- **Failovers are warnings.** A response served after a failover carries a
  `WarningKind::Fallback` warning attributed to the primary ("[anthropic]
  failed with `auth`; served by `ollama`"), naming the provider the
  response reports, so in a nested chain every warning names the innermost
  server. Its `cause` (`WarningCause`) holds the primary's error kind, on
  the warning and not only the event, because metrics read warnings
  (`ObserveProvider::on_warning`), not log events. In a stream it comes
  after the backup's own warnings, before its first block, so a collected
  stream reports what the unary call does.
- **A failed failover returns its last error as itself**: same variant,
  status and kind, so code that matches on errors needs no failover case.
  Earlier errors are kept behind it, newest first: `Error::fallback_from`
  walks back, and `Error::with_fallback_from` appends, which is how a
  third-party failover or hedging wrapper builds the same chain. The chain
  compares equal to any other (so `ErrorContext` keeps `Eq`), and
  `Display` shows only the returned error. A last error without a context
  (`Serialization`, extraction) drops the chain; the `allama::fallback`
  event is then its only record.

### Retry attempts log their warnings once

`RetryingProvider` runs each attempt in a crate-private attempt scope,
installed per poll so it follows the call across threads, and discards the
warning events of attempts it retries; a retried call logs each warning
once. `FallbackProvider` enters no scope: a primary that warned and then
failed made a real call, so its warnings are logged, then the failover
warning.

## Consequences

- Middleware today is a hand-written wrapper with the forwarding macros
  and `assert_wrapper`; the example is the template.
- Failover is observable (warning, cause, `FallbackAttempts`, the error
  chain) without changing the error a caller matches on.

## Rejected alternatives

- **A wrapper error for failed failovers** (`Fallback { primary, fallback
  }`). It changes the shape every caller matches on: a plain `match` that
  answers `RateLimited` with 429 would hit its catch-all, and the kind
  would be one no metric or retry rule meant.
- **The recipe only, forever.** The macros remove the forwarding trap, but
  every middleware would still write the stream boxing and both chat calls
  itself; `Layered` lets it supply only behaviour.
- **A blanket `impl<W: Wrapper> ChatProvider for W`.** It overlaps the
  impls for `&T`, `Box<T>` and `Arc<T>`, and a wrapper trait with
  defaulted forwarding has the same silent-default problem one level up.
- **A generic `wrap_chat<P: ChatProvider>(&self, inner: &P, ..)`.** The
  middleware could call anything on the provider, and every method would
  be generic.
- **Object-safe middleware traits.** Every middleware would box every
  call; the provider traits set the pattern of static traits plus an
  erased wrapper.
- **tower `Layer`/`Service`.** Its readiness and ownership model does not
  fit capability questions and validation, and it would tie the core to
  tower.
