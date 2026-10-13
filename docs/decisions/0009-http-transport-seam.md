# ADR 0009: An HTTP transport seam owned by the provider crates

- Status: accepted

## Context

HTTP provider adapters all build requests, classify transport failures,
read error responses, place credentials, bound calls in time and decode
streamed bodies. Done separately in each crate, these rules drift, a
third-party adapter copies whichever crate it read, and tests need a live
server. The core crate is a portable domain model and must not depend on
an HTTP client.

## Decision

### The seam

- `allama-http` holds the seam and the HTTP rules every adapter shares,
  first-party or not. The core `allama` crate stays free of HTTP; transport
  middleware is not a core concern.
- `HttpTransport::send` takes an `HttpRequest` (plain data: method, URL,
  headers, body, optional timeout) and returns an `HttpResponse` (status,
  headers, streamed body). An error status is a response, not an error.
  Only a missing response is a `TransportError`, classified once by kind
  and mapped to a portable error by `TransportError::into_error`.
- Provider builders take a transport through `http_transport(..)` and
  nothing `reqwest`-shaped. A client of your own goes in as
  `ReqwestTransport::from_client` or `from_middleware`, so `reqwest`
  appears in `allama-http`'s public API only. A `reqwest` or
  `reqwest-middleware` major bump is a breaking release of `allama-http`.
- `ReplayTransport` serves canned responses and records requests, so a
  real adapter runs over recorded wire data with no server, on any target.
- The default `ReqwestTransport` never follows a redirect, and adapters
  map a `3xx` to `Error::UnexpectedResponse` naming the target host
  (`redirect_error`). `reqwest` strips only `Authorization` on a
  cross-host redirect, so `x-api-key` and similar headers would follow a
  `307`/`308` along with the prompt. A client handed in keeps its policy.
- Bodies are read under caps: success bodies through
  `HttpResponse::bytes_limited` with the builder's `max_response_size`
  (default 256 MiB, because a full batch of large embeddings in JSON can
  pass 128 MiB), error bodies through `HttpResponse::map_error` (64 KiB,
  within 30 s). A failed error-body read is added to the error's upstream
  message rather than becoming an empty body.
- Builders validate configuration at `build()` (`allama_http::config`):
  keys must fit a header, base URLs must be absolute `http(s)` without a
  query. Failures are authored `Error::Configuration` errors naming the
  setting. Builders send `DEFAULT_USER_AGENT` unless a static
  `header("user-agent", ..)` replaces it; static headers refuse
  credential header names.

### Error helpers

- `error_context` reads an error response's status, request id, retry hint
  and server retry directive (ADR 0011); `error_envelope` reads the common
  JSON error envelopes. Every bundled adapter starts its errors from them,
  so a new header or redaction rule reaches third-party adapters through
  the same public function. An adapter whose API names a header
  differently overrides that field afterwards.
- Text that came from the wire is attached as the error's upstream
  message, never its authored message, so it stays out of the redacted
  log form. `TransportError::into_error` authors its message from the
  failure kind and keeps the HTTP client's text as upstream: that text can
  contain the request URL, and some APIs put the key in the query string.
  A transport marks text it wrote itself with `TransportError::authored`.
- Classification is shared only where it depends on the failure alone: a
  failed body read (`TransportError::into_stream_error`: a timeout is
  `Error::Timeout`, anything else a retryable stream transport error) and
  the framing failures below. What a provider's own error codes and
  in-band errors mean stays with the provider.

### Credentials are typed, scoped and placed by the adapter

A per-request key is not a header. Every adapter in a fallback chain
receives the same request options, so a key sent as a plain header
reaches every provider tried: a leaked key and a 401 where the fallback
should have served.

- The caller addresses a typed, redacted `Credential` to a credential
  scope, the account it belongs to, through
  `HttpRequestOptions::credential(scope, ..)`. `Credential::api_key` is
  the provider's own key; `Credential::header` is any other auth header
  (a gateway token, a third-party adapter's header), with the secret as
  the whole value. Other per-request headers ride in `HttpRequestOptions`
  too, a typed option (ADR 0006), because headers are transport, not
  request semantics.
- Each builder has `credential_scope(..)`, defaulting to the crate's
  `DEFAULT_CREDENTIAL_SCOPE`. The scope is independent of the provider
  name and the options namespaces. Two accounts of one provider in a
  fallback need two scopes; providers behind one gateway share one.
- The adapter declares a `CredentialSlot` (scope, placement, the
  configured headers it replaces, whether a credential is required) and
  passes it to `HttpRequest::apply_options`, after setting its own
  headers. Credentials addressed to other scopes are never sent.
  `CredentialPlacement::Custom` lets an adapter place the credential
  itself, for example to sign the request after the plain headers are in.
- There is no query-string placement: `http-tracing` records the full URL
  on spans. An adapter whose API takes its key in the query uses `Custom`
  and owns the redaction.
- `apply_options` refuses, at send time and before anything is sent, with
  an authored `Error::Configuration` (the application's bug: not retried,
  not failed over): a plain header on the exact `is_credential_header`
  list; a plain header the adapter's slot writes or replaces; a
  credential kind the adapter cannot place (`Credential::unsupported`);
  and a missing credential for a required slot. An adapter never falls
  back to its configured key for a kind it cannot place, because that
  bills the wrong account.
- Refusal is per adapter, so a plain third-party header can reach the
  providers tried before the one that owns it. Only the global exact list
  protects every hop; third parties put their keys in `Credential::header`.
- The broader substring rule (`key`, `token`, `secret`, `auth`) is for
  redaction only, so `Idempotency-Key` is still sent.
- Secrets never leave the process through options: `Debug`, error text,
  spans, recorded exchanges and the serialized form omit them.
  `Credential` deserializes with its secret (for configuration files) but
  serializes without it. The JSON form of the options takes no
  credentials, because JSON options are exported and printed as given.
- A builder may omit the configured key when a scope is set (bring your
  own key); the slot is then required. With a configured key, a mistyped
  scope silently falls back to it, which the builder docs warn about.

### Timeouts

- `allama_http::DEFAULT_REQUEST_TIMEOUT` (120 s) bounds every unary HTTP
  call, whatever the transport. Only an explicit `timeout(None)` removes
  it; handing in a client, for a proxy say, must not quietly remove the
  bound. A set timeout replaces a client's own total timeout.
- The rule is for every HTTP adapter, not just the bundled builders: an
  adapter that sends without a timeout waits forever on a server that
  accepts the connection and never answers.
- Streams carry no total timeout, which would cut long generations; they
  are bounded by `StreamTimeouts` through `stream_timeouts`.

### Stream framing

- `SseStream` and `SseDecoder` follow the WHATWG event stream format (all
  line endings, including a `CRLF` split across chunks; multi-line data;
  an unterminated last event discarded) with one deliberate exception:
  **invalid UTF-8 fails the stream** (`SseError::InvalidUtf8`). A browser
  shows event text to a person, but an adapter decodes JSON from it, and a
  replacement character silently changes generated text, tool arguments
  or an opaque signature replayed on the next turn (ADR 0003). A chunk
  boundary inside a UTF-8 sequence is not an error.
- **The decoder is linear and bounded.** It examines each byte once, and
  caps a line and an event's data at `max_stream_event_size` (default
  `SseDecoder::DEFAULT_MAX_EVENT_SIZE`, 64 MiB, since a Responses
  `response.completed` event repeats the whole output and streamed base64
  images pass 16 MiB). The cap is checked before a chunk is buffered and
  the buffer never grows past it, which matters in a wasm isolate.
  Exceeding it fails with `SseError::EventTooLarge`; events completed
  before the oversized line are returned first.
- Both deliberate failures are not retryable, because asking again
  streams the same bytes: `InvalidUtf8` maps to `Error::stream_decode`,
  `EventTooLarge` to `Error::stream_protocol` (`SseError::into_error`). A
  failed read stays the body's own error (`SseError::Transport`).
- `NdjsonStream` applies the same rules where they fit. Unlike SSE, a
  last line without its newline is still a line; a body cut mid-line
  yields a partial line, which the adapter's JSON decoding rejects.

### Chat stream driver

- **A chat stream ends after its first error.** An `Err` item is the last
  item of a `ChatStream`; events after it would belong to no response. A
  decoder that reports an error is finished and adds no second error when
  the body ends. Hand-built streams (wrappers, test doubles) must end
  after yielding an error and never poll what they wrap again.
- `FramedChatStream` runs an adapter's `ChatFrameDecoder` over any framer
  and enforces this: `end` or `fail` runs at most once and only while the
  decoder is not done, the stream ends once the decoder is done without
  waiting for the body to close, and the body is never polled after the
  stream ends. `SseChatStream` is a type alias for it over `SseStream`,
  not a second concept.
- `Prepared::finish_stream`, `with_collector()` and the extraction stream
  stop at the first error too, so a decoder that breaks the rule cannot
  reach callers through them. Conformance collects up to the first error,
  with an item cap as backstop rather than a watchdog.
- Core holds pre-start events: `finish_stream` holds the `Warning` and
  `ResponseProviderData` events a stream yields before its first other
  event, and puts them after `ResponseStart` (or right before an early
  error). Decoders yield vendor frames as they arrive and keep no buffer.
  `allama::adapter::chat::start_first` applies the same rule to a stream
  that bypasses `finish_stream`.
- `VendorFrames` emits each unmodelled frame's payload alone, and
  `VendorFrames::finish`, yielded before `ResponseStop`, carries the full
  array of each repeated event type. Re-sending the array so far with
  every frame would make a gateway that sends a side event per chunk
  quadratic in time and in recording size.

### Trace context belongs to the application

allama does not inject OpenTelemetry trace context and no published crate
depends on OpenTelemetry. Reading the current span's context pins an
`opentelemetry`/`tracing-opentelemetry` release as a behaviour, not a
type: an application on another release gets a different global
propagator, and nothing is sent and nothing fails. The application
injects with its own version, through an `HttpTransport` wrapper (covers
every adapter and every retry attempt) or, for a registry provider that
takes no transport, a chat wrapper that adds the header to
`HttpRequestOptions`. A `traceparent` the caller set always wins. The
`trace_context` example in `allama-examples` shows both.

### Cargo features

- Every provider crate has a default `reqwest` feature. Without it the
  crate builds without `reqwest` and works over `http_transport(..)`;
  `build()` without a transport is a configuration error. An application
  with its own transport, or a library reusing a crate's encoding, should
  not pay for `reqwest`. CI builds and tests the crates that way.
- `rustls-ring` selects `rustls` with `ring`, avoiding `aws-lc-sys`
  (cmake and a C toolchain on musl). `ReqwestTransport::new` installs
  `ring` as the process default unless the application installed one.
- Provider crates forward `http-tracing` but not `reqwest-middleware`.
  Its only use is `ReqwestTransport::from_middleware`, an `allama-http`
  type, so its caller already depends on `allama-http`. `http-tracing`
  is the only way to get HTTP spans for a registry provider, and a version
  split there is a compile error rather than a silent failure.

## Consequences

- Third-party adapters get the same error, credential, timeout and
  framing rules by calling the public helpers, and
  `allama-conformance` checks them (`assert_transport_options_applied`,
  `assert_retry_signals_surfaced`, `assert_unknown_stream_frames_surfaced`
  with a `StreamFraming`).
- Adapters are tested over `ReplayTransport` without a server, and a
  `fetch`-based transport needs no adapter change.
- A binary framing (AWS event stream) needs byte bodies in the
  conformance contract first.

## Rejected alternatives

- **One decoder for unary and streaming responses.** Only Gemini shares a
  shape between the two, and it keeps one decoder. Elsewhere it would
  mean synthesizing stream events from unary bodies; separate decoders
  are kept honest by parity tests over recorded fixtures.
- **An async auth interceptor.** Per-request credentials in
  `HttpRequestOptions` cover per-call keys without a second extension
  point.
- **Refusing credential headers when options are built.** The adapter's
  slot is not known there.
- **A substring rule for refusals.** It would refuse `Idempotency-Key`.
- **Lossy UTF-8 decoding, as browsers do.** It hides corruption.
- **An OpenTelemetry propagation feature in `allama-http`**, or
  `reqwest-tracing`'s OpenTelemetry features. Both tie allama's releases
  to OpenTelemetry's and fail silently on a version split. A feature
  with the version in its name, or a separately versioned crate, could be
  added later without breaking anything.
