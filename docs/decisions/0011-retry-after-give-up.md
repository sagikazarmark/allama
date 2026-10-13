# ADR 0011: Retries wait out a provider's hint or give up, within a budget

- Status: accepted

## Context

`RetryingProvider` decides, after each failed attempt, whether to retry and
how long to wait. Providers often say how long: a retry hint (`Retry-After`,
`retry-after-ms`, surfaced as `Error::retry_after`), and sometimes whether
to retry at all (`x-should-retry`).

The tempting design clamps a long hint to the policy's own backoff cap and
retries anyway. That calls the provider before it said it would accept the
request: the retry is almost certain to be rejected, spends an attempt, can
extend the rate limit, and delays failover, because `FallbackProvider` sees
the error only after the retries are used up. Retrying without a shared
limit has a second cost: in a full outage every caller retries every error,
and the default policy triples the load on the failing provider.

## Decision

### Retry hints are a floor, and a long hint gives up

- **A positive hint up to `max_retry_after` is waited out, never less.**
  The wait is `max(hint, backoff)`, where `backoff` is the policy's
  exponential backoff without jitter, capped at `max_delay`. With jitter on,
  a uniform extra of up to `base_delay` is added and clamped at
  `max_retry_after`; the clamp trims only the extra, never the floor. With
  jitter off the wait is exactly `max(hint, backoff)`. The extra exists
  because servers send the same hint to every client at a rate-limit
  reset, and without it they all retry at the same instant. (The AWS SDKs
  add no jitter to a server-directed delay, expecting servers to jitter
  their hints; the servers allama talks to do not.)
- **A hint longer than `max_retry_after` gives up at once.** The error is
  returned unchanged, `retry_after` included. Rate limits are
  fallback-eligible, so a `FallbackProvider` around the wrapper takes over;
  without one, the caller sees the hint and decides. Tracing records
  `allama.retry.give_up_reason = hint_over_ceiling`.
- **A zero hint is no hint.** `Retry-After: 0`, or an HTTP date already
  past (clock skew produces these), gets the normal jittered backoff, so it
  never makes the wrapper retry at once; it says nothing about how long to
  wait, so it does not floor at `base_delay` either. `parse_retry_after`
  still returns `Duration::ZERO`, so callers see what the server sent.
- `respect_retry_after(false)` ignores hints entirely.

### Two bounds: `max_delay` and `max_retry_after`

`max_delay` (5 s by default) caps only the policy's own backoff.
`max_retry_after` (120 s by default, openai-python's
`MAX_RETRY_AFTER_DELAY`) is the hint ceiling: the longest wait a provider
can ask of the wrapper. One value cannot do both jobs. A cap of a few
seconds is right for unhinted backoff but, as a ceiling, gives up on every
rate limit longer than that; a ceiling of minutes, as a backoff cap, lets
unhinted retries grow to minutes.

The default ceiling suits batch callers, which have no one waiting and
often no fallback. Interactive servers set `max_retry_after` to a few
seconds on each side, so a long rate limit fails over, or reaches the
client as `429` with `Retry-After`, at once. The recommended interactive
stack, `Tracing(Timeout{request}(Fallback(Retry(Timeout(p)), ..)))` with
`max_retry_after(5 s)`, bounds time to first output across every retry with
the outer timeout, never with `stream_total`, which would cut off a long
answer after it started.

### Backoff jitter

Unhinted backoff uses equal jitter: a wait drawn uniformly between half
the backoff and the full backoff. Full jitter decorrelates slightly better
but can retry almost immediately, which wastes the few attempts a default
policy has. Randomness comes from `RandomState`, the wall clock
(`web-time`) and a counter, so it needs no `getrandom` and works on
`wasm32-unknown-unknown`. Unlike the AWS SDKs, throttling uses the same
base backoff as other errors, because most rate limits carry a hint, which
takes precedence.

### The server retry directive

`x-should-retry: true|false` is honoured by the OpenAI and Anthropic SDKs
ahead of their status rules; allama reads it from any provider and
surfaces it as `Error::server_retry_directive()`.

- **Only the default classifier follows it**, in both directions: `true`
  retries an error the kind and status would not (a `409`), `false` stops
  one they would (a `429`, even with a hint). `Error::is_retryable` never
  reads it.
- **It replaces the classification, nothing else.** The attempt limit, the
  hint ceiling, the budget and the rule against retrying a stream after its
  first output still hold. A server cannot lift the client's limits.
- **It stops retries, not failover.** Fallback eligibility does not read
  it.
- **A `retry_if` classifier owns the decision**, directive included; the
  policy never overrides it. A classifier that wants the directive reads
  the accessor.

`409 Conflict` is not retryable by default. The official SDKs retry it as
a lock timeout; allama classifies it as an invalid request, retries it
only on `x-should-retry: true`, and documents the one-line `retry_if` that
adds it.

### Retry budgets

`RetryBudget` copies the AWS SDK retry quota, with the costs of the
standard retry strategy as the AWS SDKs and Tools Reference Guide
documents it in 2026
(<https://docs.aws.amazon.com/sdkref/latest/guide/feature-retry-behavior.html>):
500 tokens, starting full; a retry after `Error::RateLimited` costs 5 and
any other retry 14; a retry the bucket cannot afford is denied; a call that
succeeds on a retry refunds that retry's cost, a first-try success adds 1;
the first attempt of a call never needs a token. Only the capacity is
configurable (`RetryBudget::capacity`); the costs and refund are fixed.

- **`Error::Overloaded` costs a transient retry**, not a throttling one:
  an overloaded provider is the outage the budget exists for. A failed
  failover returns its last error as itself (ADR 0013), so it is costed by
  that error's kind with no special case.
- **It applies last.** The classifier, the attempt limit and the hint rule
  decide first; the budget can only deny a retry they allowed. Not even
  `x-should-retry: true` overrides a denial. A denied retry returns the
  error unchanged.
- **Each wrapper owns a budget by default**, as each AWS client owns its
  quota: an opt-in budget protects only the applications that know to ask.
  Every `RetryingProvider::new` creates one; clones of the wrapper share
  it. Clones of a *policy* do not share a budget implicitly, since two
  wrappers built from one policy value would drain one bucket without
  saying so. Sharing is explicit with `RetryPolicy::budget(shared)`, e.g.
  across both sides of a fallback; `no_budget()` opts out.
- **Every operation draws on it**: chat (a stream that reaches its first
  output counts as a success), embeddings and decisions.
- **`retry_delay` stays pure**: it applies every rule except the budget. A
  custom retry wrapper takes its budget from `make_budget()` once, when it
  is built, calls `try_acquire` before each retry, and returns the opaque
  `RetryPermit` to `record_success`. `RetryPermit` is exported so a
  streaming wrapper can hold it in a named field until the first output.
- Tracing records `allama.retry.server_directive` when the directive
  decided and `allama.retry.budget_denied` when the budget did.

Known limits, documented: a wrapper built per request gets a fresh, full
budget and so no protection; a drained default budget regains one transient
retry per 14 first-try successes; a budget bounds one process, not a fleet.

### Configured providers retry by default

The core stays bare: a provider built directly retries nothing until it is
wrapped. `allama_providers` lookups (chat, embedding, decision, the
`Registry` and registered constructors) build
`Retry(Timeout?(UnsupportedPolicy?(provider)))` from `ProviderSettings`
(`retry`, `attempt_timeout`, `unsupported_policy`), with
`RetryPolicy::default()`, so a transient `503` is retried as the official
SDKs would. The per-attempt timeout sits inside the retries, so a stalled
attempt is retried. The rustdocs warn against wrapping a lookup in another
`RetryingProvider` and state the default worst case (about 10 minutes).

### Conformance

`allama_conformance::assert_retry_respects_hint` checks, on `chat` and
`chat_stream`, that a wrapper never retries before a 30 s hint and that an
error it returns still carries the hint.

## Consequences

- With the defaults a call can wait up to 120 s per retry on a rate
  limit; interactive callers must lower `max_retry_after`.
- Tests that need exact delays turn jitter off.

## Rejected alternatives

- **A policy enum** (`Clamp`, `GiveUp`, `WaitUpTo(d)`). Only one answer is
  honest: retrying before the provider's hint is never what it asked for.
  Callers who want longer waits raise `max_retry_after`; callers who want
  to ignore hints turn off `respect_retry_after`.
- **Always waiting the full hint, or an unbounded default ceiling.** It
  hides a long outage behind a silent wait and keeps a fallback from
  taking over.
- **gRPC retry throttling** (`maxTokens` / `tokenRatio`). The AWS quota is
  documented with concrete defaults, distinguishes throttling from other
  failures, and never blocks a first attempt.
- **A process-wide default budget.** Hidden global state would couple
  unrelated providers and tests; AWS scopes its quota to the client.
