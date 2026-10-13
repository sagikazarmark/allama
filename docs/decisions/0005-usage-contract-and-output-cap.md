# ADR 0005: Usage accounting contract and output cap

- Status: accepted

## Context

Providers count tokens differently. Anthropic reports cache reads and
writes beside the uncached input; Gemini reports thoughts beside the
candidate tokens; OpenAI counts both inside its totals. Providers also
report counters that are not tokens (server-tool requests) or that are
subsets of the totals (per-modality counts). A cost dashboard or budget
guard summing usage across providers double counts or under counts
unless every counter has one meaning.

## Decision

### What `Usage` counts

- `input_tokens` is the total prompt size, **including** cache reads and
  writes; `cache_read_input_tokens` and `cache_write_input_tokens` are
  subsets of it. `output_tokens` **includes** reasoning;
  `reasoning_tokens` is a subset of it. `total_tokens` is at least input
  plus output (a provider billing extra categories may report more).
  `None` means not reported; `Some(0)` is a reported zero.
- `breakdowns` holds only tokens already counted in `input_tokens` or
  `output_tokens`, under keys ending in `_input_tokens` or
  `_output_tokens`, so a consumer knows which total each is part of. The
  name says the one rule that matters: these are subsets, and adding them
  double counts.
- `tool_requests` holds counts that are not tokens: billable requests
  made by provider-executed tools (ADR 0008), keyed by tool (`web_search`,
  `web_fetch`).
- A provider counter that is neither stays in the response's provider
  data.
- Every adapter normalizes to this contract. `Usage::contract_violation`
  checks it (including unknown breakdown keys and a breakdown larger than
  its total), and the conformance helper `assert_usage_contract` holds
  adapters to it.
- Summing keeps `total >= input + output`: when either side reported a
  total, the sum adds each side's total, derived from its input and output
  where it reported none. All usage arithmetic saturates.

### Failed attempts carry no usage

A failed attempt before any output (a retried attempt, a fallback attempt
that failed over) reports no usage, because providers document no billing
for them. Reopen this if a provider starts billing rejected attempts.

### Usage on streams is a capability

`ChatCapability::StreamUsage` says whether a collected `chat_stream`
response carries usage, so a budget guard knows up front whether it can
count a stream. It describes output, so no request field asks for it and
preflight never checks it. Like every capability answer it needs evidence
(ADR 0015); `assert_stream_usage_matches_capability` checks it against a
collected stream. `StreamCollector::metadata_only` folds a stream's usage
and metadata (by each event's `usage_mode`) without buffering content,
for wrappers that pass the stream on.

### `max_output_tokens` is the total output cap

`ChatRequest::max_output_tokens` is the provider's total output cap,
including reasoning where the provider counts it so (OpenAI
`max_completion_tokens`, Gemini `maxOutputTokens`, Anthropic
`max_tokens`). It is not a cap on visible output, because no provider can
split it from reasoning. An adapter fits reasoning into an explicit cap
and never raises it (ADR 0003, ADR 0004).

## Consequences

- Costs can be summed across providers without knowing their
  conventions. The adapters do the arithmetic: Anthropic's adds cache
  tokens to input, Gemini's adds thoughts to output.
- `Usage` is persisted with each turn in a stored history, so its field
  names, including `breakdowns`, are a serialized format.

## Rejected alternatives

- **A free-form "additional" map.** Without a rule, adapters put request
  counts, breakdowns, and duplicate counts side by side, and consumers
  could not tell an extra billable dimension from a subset.
- **A visible-output cap.** It cannot be enforced where reasoning and
  answer share one limit.
