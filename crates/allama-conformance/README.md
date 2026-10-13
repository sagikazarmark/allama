# allama-conformance

[![crates.io](https://img.shields.io/crates/v/allama-conformance?style=flat-square)](https://crates.io/crates/allama-conformance)
[![docs.rs](https://img.shields.io/docsrs/allama-conformance?style=flat-square)](https://docs.rs/allama-conformance)

Conformance contracts and test support for `allama` provider adapters and
wrappers.

This crate exists for provider authors. It gives new adapters a shared bar for
request shaping, response normalization, streaming event ordering, error
mapping, and honest capability claims, so the provider ecosystem can grow
without each crate inventing its own test contract. Start with the
[provider author guide](../../docs/guides/provider-authors.md).

Add it as a dev-dependency:

```toml
[dev-dependencies]
allama-conformance = "0.1"
tokio = { version = "1", features = ["macros", "rt"] }
```

From the workspace root, its own tests run with:

```bash
cargo test -p allama-conformance
```

## Feature flags

| Feature | Default | Enables |
| --- | --- | --- |
| `server` | yes | `TestHttpServer`, `MockHttpResponse`, and `RecordedRequest`: a local HTTP server for transport tests (pulls in `axum` and Tokio's networking) |
| `extract` | yes | the extraction scenarios in `e2e` (`e2e::extract`, `e2e::extract_enum_root`) |

`server` is on by default because most provider crates speak HTTP and the
recommended transport tests need it; it is a dev-dependency, so its cost
never reaches the provider's users. A wrapper crate, or a provider that does
not speak HTTP (an in-process binding, a wasm-only target), turns it off:

```toml
[dev-dependencies]
allama-conformance = { version = "0.1", default-features = false }
```

`server` brings no HTTP client. The provider under test reaches
`TestHttpServer` over `allama-http`'s `reqwest` transport, so a crate whose
own `reqwest` feature is off (or absent) in the test build adds it as a
dev-dependency:

```toml
[dev-dependencies]
allama-http = { version = "0.1", features = ["reqwest"] }
```

Everything else (fixture assertions, the behavioral contracts, every
contract below) works without it. The contracts that wait
(`assert_retry_respects_hint`) need a Tokio runtime with timers, such as
`#[tokio::test]`.

## Versioning

This crate is a dev-dependency, but its public API names third-party
crates, so a new major (pre-1.0: minor) version of any of them is a
breaking release of `allama-conformance`:

- `serde` and `serde_json` 1: fixtures and probes carry
  `serde_json::Value`, and `assert_json_fixture_eq` and
  `MockHttpResponse::json_value` take any `Serialize`;
- `bytes` 1, as `allama_http::Bytes`: `MockHttpResponse::json` bodies;
- `futures-core` 0.3: the `Stream` trait, which the stream replays and
  stream fixture assertions take and `byte_stream_from_sse_text` returns.

It also names `allama` and `allama-http` types, which are released at the
same version as this crate.

## Contracts

Run them through the aggregates: `assert_chat_adapter` runs every
probe-based chat contract below against an adapter, from one
`AdapterUnderTest` (its models, namespace and probes, and setters for the
inputs some contracts need: error mapping and envelope, a unary and a
stream replay, captured responses (text, tool calls, a length cut-off),
stream ends, stream captures in SSE or another framing, transport options,
a malformed response, a redirect call).
`assert_embedding_adapter` and `assert_decision_adapter` are the same for
embeddings and decisions. A contract whose input is missing fails the
aggregate unless it is listed in `skip`, so nothing is skipped silently,
and `known_gaps` lists acknowledged defects until they are fixed.
Every contract stays a function of its own, for targeted tests.

The set grows only in minor releases (`0.x` to `0.(x+1)`), never in a
patch release, so `cargo update` never turns your suite red. A contract
may join an aggregate as *provisional* (`Contract::is_provisional`): it
never fails a suite. Without its input the aggregate prints "not run:
needs ..."; when it runs and fails, the aggregate prints the failure. A
later minor release may make it required. A
contract that needs an input older suites lack is always provisional at
first. No contract is provisional in 0.1.

A contract outside the aggregates may be provisional too, with no promise
that it joins one: you call it yourself, and what it checks may change in
any minor release. In 0.1 that is `decoding::assert_vendor_fields_kept`,
which adds an unknown field to every object of a captured `chat` body and
expects each one in the decoded response; no bundled adapter keeps them
all yet. The
[conformance lifecycle ADR](../../docs/decisions/0017-conformance-contract-lifecycle.md)
records these rules.

Your adapter brings its namespace, two probes (encode a request and
capture what the adapter would send, unary and streamed), and its error
mapping; the aggregate runs every contract from them:

```rust,no_run
use allama::{ChatProvider, ChatRequest, ChatResponse, Error};
use allama_conformance::probe::ProbeOutcome;
use allama_conformance::stream_errors::ErrorKindProbe;
use allama_conformance::{AdapterUnderTest, Contract, assert_chat_adapter};

const NAMESPACE: &str = "my-namespace";

async fn probe(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
    todo!("encode `request` and capture it")
}

async fn stream_probe(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
    todo!("encode `request` for streaming and capture it")
}

fn map_http_error(status: u16, headers: &[(&str, &str)], body: String) -> Error {
    todo!("your adapter's HTTP error mapping")
}

fn error_envelope(kind: ErrorKindProbe, message: &str) -> Option<String> {
    todo!("your vendor's error body for `kind`, if it has one")
}

async fn conformance(provider: &impl ChatProvider) {
    let under_test = AdapterUnderTest::new(["my-model"], NAMESPACE, probe, stream_probe)
        .error_mapping(map_http_error)
        .error_envelope(error_envelope)
        // ...
        .skip([Contract::StrictNotHonouredWarns]);
    assert_chat_adapter(provider, &under_test).await;
}
```

| Contract | Entry point | Checks |
| --- | --- | --- |
| Fixtures | `assert_json_fixture_eq`, `assert_response_fixture_eq`, `assert_stream_fixture_eq`, `assert_error_fixture_eq`, ... | wire requests, normalized responses, stream events, and error mapping match recorded fixtures |
| Behavior | `contract::*`, `assert_usage_contract` | finish reasons, tool-call JSON, streaming, structured output, usage accounting, the unsupported-feature policy, error classification; structured output, the policy, and the error kinds on `chat` and `chat_stream` (the error-kind contracts take a stream probe, usually `contract::collect_chat_stream`) |
| In-band stream errors | `assert_stream_errors_classified` | an error reported inside a started stream has the kind the adapter's HTTP mapping gives the same envelope at the kind's canonical status, for each `stream_errors::ErrorKindProbe` the adapter maps (at least `RateLimited` and `InvalidRequest`) |
| Vendor frames (C1) | `stream_frames::assert_unknown_stream_frames_surfaced` | frames the decoder does not model, injected in the stream's own framing (`StreamFraming::sse`, `StreamFraming::ndjson` with your vendor frame, or `StreamFraming::new`), are surfaced as provider data with a warning, never misdecoded or dropped, and reframing the stream (`CRLF` endings, multi-line SSE data) changes no event |
| Request fields (C2) | `probe::assert_request_fields_accounted` | every `ChatRequest` field listed in `probe::ACCOUNTED_FIELDS` as probed (sampling fields, system prompts, message names, cache breakpoints, image detail, file names, strict tools, tool-result errors, reasoning configuration, …) reaches the wire, is warned about, or is rejected; state from another provider and application data never reach the wire. Takes the provider, whose capability answers shape the application data conversation. Runs on JSON or raw (`ProbeOutcome::raw`) request bodies, through a chat probe and a streaming probe. A field listed with a probe this crate does not know (`allama` is newer) fails it with a request to upgrade `allama-conformance` |
| Decoding | `decoding::assert_unary_stream_agree`, `decoding::assert_provider_stamped`, `decoding::assert_tool_call_ids_round_trip`, `decoding::assert_length_finish` | a captured response (`ResponseCapture`: its `chat` body and its `chat_stream` body) decodes to the same content, finish reason, usage, ids, provider and provider data on both paths; both paths stamp `provider`; tool-call ids are the wire's (or synthesized uniquely), finish with `ToolCalls`, and reach the next request; a response cut off at the cap finishes with `Length` |
| Vendor fields (provisional, not in the aggregate) | `decoding::assert_vendor_fields_kept` | a field the adapter does not model, added to every object of a captured `chat` body, reaches the decoded response at any level: its provider data, a warning, or the content; the failure names the JSON pointer of each object whose field was dropped |
| Raw event order | `stream_order::assert_stream_event_order`, `stream_order::assert_stream_order_kept` | the raw events keep the documented order, which the lenient collector does not enforce: `ResponseStart` first, a block's events between its `BlockStart` and `BlockStop` and of its type, every block stopped before the finish reason, `ResponseStop` last |
| Validation coverage | `Contract::ValidationEncodingCoverage` (aggregate only) | `validate_chat_request` is a dry run of the encoding (`ValidationCoverage::Encoding`); skip it only when the adapter cannot encode without sending |
| Redirects | `redirect::assert_redirect_not_followed` (feature `server`) | the adapter's client never follows a redirect: the other host sees nothing, and the call fails with a non-retryable `UnexpectedResponse` carrying the `3xx`, on `chat` and `chat_stream` |
| Capability honesty | `capabilities::assert_capabilities_honest` | whatever a provider answers (`Supported`, `Unsupported`, `Unknown`), a request that uses a capability sends it, warns about it, or is refused before sending, for each listed model and one the provider cannot know: the evidence for `Supported` chat claims |
| Negative capability audit (C3) | `capabilities::assert_unsupported_chat_capabilities_refused`, `capabilities::assert_unsupported_embedding_capabilities_refused` | a request that uses a capability reported `Unsupported` is refused: a warning whose subject names the capability's field or content, or `Error::Unsupported` with stage `BeforeSending` and nothing sent; never a silent send; the chat audit runs on `chat` and `chat_stream` |
| Stream usage | `capabilities::assert_stream_usage_matches_capability` | the response collected from `chat_stream` has usage when `StreamUsage` is `Supported` and none when it is `Unsupported` |
| Embedding capability evidence | `capabilities::assert_embedding_capabilities_backed` | every `Supported` embedding claim is backed by a fixture or test |
| Validation agreement | `assert_validation_agrees`, `assert_embedding_validation_agrees`, `assert_decision_validation_agrees` | validation gives the answer the call gives, for the drift requests and your own |
| Transparent wrappers (C4) | `assert_chat_wrapper` (and `assert_wrapper`, `assert_embedding_wrapper`, `assert_decision_wrapper`); the parts: `assert_transparent_wrapper` and its variants, `assert_transparent_chat_data_path` and its variants | a wrapper forwards identity, capabilities, limits, and validation, and passes the provider's warnings (it may add its own), provider data, usage, ids, stream events and errors (kind, status, retry hint, refused subject and stage; the message may carry a prefix) through unchanged and in order |
| Retry hints (C5) | `assert_retry_respects_hint` | a retry wrapper never calls again before the `Retry-After` hint, and gives up with the hint intact |
| Retry signals | `assert_retry_signals_surfaced` | an HTTP error mapping surfaces the server's retry directive (`x-should-retry`) and `Retry-After` hint on the error, and a retry hint carried in the error body when the API has one |
| Error kinds | `assert_error_kinds_mapped` | an HTTP error mapping gives each `ErrorKindProbe` its variant at the kind's statuses, keeps the status, and gives a provider's refusal stage `Provider` |
| Transport options | `assert_transport_options_applied` | an HTTP adapter sends the caller's per-request headers (`allama-http` transport options), replacing its own of the same name; given the adapter's `CredentialSlot`, its own credential is sent once in its slot, replacing the configured one, a credential addressed to another scope is never sent, a plain credential header is refused with `Error::Configuration` before sending, and no secret appears in `Debug`, serialized options, sent requests or error text |
| Strict mode | `probe::assert_strict_not_honoured_warns` | a strict tool or response format the configuration cannot honour is reported with a `Tools` or `ResponseFormat` warning on `chat` and `chat_stream`, and fails before sending under `UnsupportedPolicy::Error`. Runs on JSON or raw request bodies |
| `Other` provider data | `probe::assert_other_provider_data_gated` | foreign provider data on an `Other` block or part never reaches the wire, on `chat` and `chat_stream`. Runs on JSON or raw request bodies |
| Application data | `probe::assert_app_data_never_sent` | `app_data` on system prompts and messages of every role never reaches the wire and changes nothing that is sent, on `chat` and `chat_stream`. The conversation holds a tool call and its result unless the provider reports tool calls `Unsupported` |
| Empty turns | `probe::assert_empty_turns_skipped` | an empty user or assistant message in the middle of a conversation sends exactly the bodies of the conversation without it, with the same warnings; an assistant turn left last only by empty turns is dropped with a `Dropped` warning under every policy, a prefill is kept, and a conversation of empty turns fails with `InvalidRequest` before sending, on `chat` and `chat_stream` |
| Unreplayable reasoning | `probe::assert_unreplayable_reasoning_dropped` | for a model that reports `ReasoningReplay` `Unsupported`, a replayed reasoning block and the foreign `Other` block after it never reach the wire, the request is served under every policy, and `Dropped` warnings name both at the caller's indices (blocks 0 and 1), on `chat` and `chat_stream` |
| Batch limits (C6) | `probe::assert_batch_limit_enforced` | an embedding batch over `EmbeddingLimits::max_inputs_per_call` is never sent as one request. Runs on JSON or raw request bodies |
| Embedding vector count | `probe::assert_embedding_count` | a response with the wrong number of vectors, or indices that are not exactly one per input, fails with `UnexpectedResponse`; vectors come back in input order |
| Embedding input type | `probe::assert_input_type_not_honoured_warns` | an embedding `input_type` the provider cannot honour is dropped with a warning (the request otherwise unchanged) and fails before sending under `UnsupportedPolicy::Error` |
| Error redaction (C7) | `assert_error_log_excludes_body` | a sentinel in unrecognised bodies, the common envelopes' messages, messages that steer classification, and the adapter's own envelope for every `ErrorKindProbe` it maps never reaches the redacted `as_log()` view, and never changes the variant the adapter's envelope maps to; one with spaces and punctuation in the envelopes' error type, code and `param` never reaches `as_log()` or `Debug` |
| Live scenarios | `e2e::*` | tool round trips, streaming tool calls, parallel tools, vision, file input, extraction, embeddings against the real API |

Every probe-driven chat contract takes two probes of the same shape: one
that calls `chat`, and one that collects `chat_stream` into a
`ProbeOutcome` (see the `probe` module docs). Each check runs on both paths
and names the path that failed.

## What it checks

The harness supports four main fixture categories:

- request fixtures: provider wire request output
- response fixtures: normalized `ChatResponse` output
- stream fixtures: ordered `StreamEvent` output plus collected final response
- error fixtures: normalized `allama::Error` mapping

## Writing a provider

The [provider author guide](../../docs/guides/provider-authors.md#6-contracts-and-tests)
owns the walkthrough: what a provider crate proves (fixture tests,
transport tests, the behavioral contracts on `chat` and `chat_stream`, the
adapter aggregate, live tests), with code quoted from a runnable adapter.
This README is the reference: the contracts and their entry points (above),
and the fixture files (below).

## Transparent wrappers

A wrapper (retry, tracing, timeout, your own middleware) changes how calls
run, not who answers them. The
[transparent wrapper contracts](https://docs.rs/allama-conformance/latest/allama_conformance/#transparent-wrappers)
check that a wrapper forwards everything a caller asks about the provider
(`provider_name`, `downcast_any`, the capability answers, the limits, and
request validation), and that what the provider answers reaches the caller
unchanged: its warnings, provider data, usage, ids, stream events and errors.
It wraps a `TransparencyProbe`, a provider with a unique name, mixed
per-capability answers, odd limits, a model it rejects, and calls that answer
with a response full of data, a stream of every kind of event, or an error of
each kind:

```rust
use allama::wrappers::TracingProvider;
use allama_conformance::{assert_chat_wrapper, assert_wrapper};

#[tokio::test]
async fn my_wrappers_are_transparent() {
    // Chat, embeddings, and decisions.
    assert_wrapper(TracingProvider::new).await;
    // Chat only.
    assert_chat_wrapper(|probe| MyChatMiddleware::new(probe)).await;
}
```

Use `assert_embedding_wrapper` and `assert_decision_wrapper` for wrappers
over a single other trait, and `assert_transparent_identity` for
`ProviderIdentity` alone. A wrapper may add warnings, and adds context to an
error with `Error::with_message_prefix`, never by re-wrapping it. A wrapper
that combines providers passes when every inner provider is the probe (for
example `FallbackProvider::new(probe, probe)`, whose error after both sides
fail ends in the probe's). A retry wrapper gets a policy without backoff in
the test.

## Complementary transport tests

Conformance fixtures prove request shaping, normalization, and streaming logic in
isolation. For first-release provider crates, add a second layer of transport
integration tests that exercise the real provider implementation against a local
mock HTTP server.

These tests should verify at least:

1. URL and endpoint selection
2. required headers and auth wiring
3. one-shot response decoding through `provider.chat(...)`
4. streaming response decoding through `provider.chat_stream(...)`
5. non-2xx HTTP error mapping with real response headers

`allama-conformance` exposes a small `TestHttpServer` helper (feature
`server`, on by default) for this purpose so provider crates can keep the
tests focused on provider behavior rather than on server scaffolding. Each
`RecordedRequest` it captures exposes the method, path, query, headers
in the order received (a header sent twice is listed twice:
`header_values(name)`), and body (`body_json()`, `body_text()`).

A request that arrives when no response is queued gets status `599`
(`UNPLANNED_REQUEST_STATUS`), and the server panics when it is dropped, so
an unintended extra request (a retry the test did not plan) fails the test
even when the client absorbed the error. `assert_exhausted()` also checks
that every queued response was taken; `remaining()` counts the untaken
ones.

Typical transport suite commands:

- `cargo test -p allama-openai --test http_integration`
- `cargo test -p allama-anthropic --test http_integration`
- `cargo test -p allama-gemini --test http_integration`

## Typical test pattern

Provider crates usually add a `conformance_tests.rs` module that:

```rust
use std::path::PathBuf;

use allama_conformance::{
    assert_error_fixture_eq,
    assert_json_fixture_eq,
    assert_response_fixture_eq,
    assert_stream_fixture_eq,
    FixtureDir,
    load_json_fixture,
    load_text_fixture,
};

fn fixtures() -> FixtureDir {
    FixtureDir::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"))
}

#[test]
fn request_fixture_matches() {
    let fixtures = fixtures();
    let actual = provider_specific_request_value();
    assert_json_fixture_eq(&actual, &fixtures, "request.json");
}

#[test]
fn response_fixture_matches() {
    let fixtures = fixtures();
    let raw = load_json_fixture(&fixtures, "response_raw.json");
    let response = provider_specific_response_conversion(raw).unwrap();
    assert_response_fixture_eq(&response, &fixtures, "response_expected.json");
}

#[tokio::test]
async fn stream_fixture_matches() {
    let fixtures = fixtures();
    let sse = load_text_fixture(&fixtures, "stream.sse");
    let stream = provider_specific_stream_conversion(&sse);
    assert_stream_fixture_eq(
        stream,
        &fixtures,
        "stream_events.json",
        "stream_response_expected.json",
    )
    .await;
}

#[test]
fn error_fixtures_match() {
    let fixtures = fixtures();
    let error = provider_specific_error_mapping();
    assert_error_fixture_eq(&error, &fixtures, "error_auth.json");
}
```

## Fixture layout

Each provider crate owns its own fixtures under `<provider-crate>/fixtures/`.

Examples:

- `crates/allama-openai/fixtures/`
- `crates/allama-anthropic/fixtures/`
- `crates/allama-gemini/fixtures/`

Common files:

- `request.json`
- `response_raw.json`
- `response_expected.json`
- `stream.sse`
- `stream_events.json`
- `stream_response_expected.json`
- `error_auth.json`
- `error_rate_limited.json`
- `error_context_length.json`
- `error_model_not_found.json` when the provider maps it distinctly
- `error_timeout.json`

You can add more provider-specific fixtures if a provider has meaningful edge
cases, but the common set should stay recognizable across crates.

When a provider has important interrupted-stream behavior to prove, add the
matching extra fixtures explicitly instead of overloading the happy-path files:

- `stream_truncated.sse`
- `stream_truncated_events.json`
- `stream_truncated_response_expected.json`
- `stream_truncated_completeness.json`
- `stream_truncated_error.json`

That lets a provider prove both sides of the contract:

- strict collection via `StreamCollector::finish()` rejects incomplete streams
- recovery via `StreamCollector::finish_partial()` preserves salvageable
  non-tool content, drops incomplete tool calls, and reports
  `StreamCompleteness`

## Important contract details

- Stream conformance now assumes strict `StreamCollector::finish()` semantics.
  A successful stream fixture must include a complete event sequence ending in a
  normal `ResponseStop`.
- Truncated stream fixtures should normally assert both the strict failure and
  the recovery path. In practice that means checking the same event sequence
  with `assert_stream_finish_error_fixture_eq(...)` and
  `assert_partial_stream_fixture_eq(...)`.
- `StreamCompleteness` is part of the public recovery contract. Providers should
  prove whether a truncated stream missed `ResponseStop`, left blocks open, or
  dropped incomplete tool calls.
- Response fixture comparisons serialize `ChatResponse` directly (it is
  `Serialize`), so provider data, warnings, and usage are all compared.
- Set `ALLAMA_UPDATE_FIXTURES=1` to rewrite every fixture an assertion
  compares (requests, responses, stream events, stream completeness,
  errors, embeddings) from the current output instead of comparing,
  creating missing files; review every regenerated file. An error fixture
  is written as the same `ErrorFixtureView` the comparison uses.
- Prefer recorded real responses over hand-written ones for response and
  stream fixtures. The allama repository's `scripts/capture-fixture.sh raw`
  records one from any endpoint.
- Error fixture comparisons serialize `ErrorFixtureView`: the redacted
  `Error::as_log()` fields plus the constructor message, its origin when it is
  authored, and the upstream message, so fixtures pin message mapping without
  depending on the full serde form of `allama::Error`.
- `assert_error_log_excludes_body` feeds a sentinel through your error
  mapping (unrecognised bodies, envelope messages, messages that steer
  classification) and fails if it reaches the redacted `as_log()` view.
  It takes the envelope builder the error kind contracts take
  (`Fn(ErrorKindProbe, &str) -> Option<String>`, the `&str` being upstream
  text) and sends the sentinel through every kind you map, so it reaches
  every branch of your classification.

## Why this matters

This crate is one of the main reasons the current provider ecosystem is more
than a loose abstraction demo. It gives adapter authors a concrete way to prove
that a new provider behaves like the existing ones at the `allama` boundary.

Used together, fixture conformance tests, mock transport integration tests, and
opt-in live provider tests provide a much better release bar than any one of
those layers on its own.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
