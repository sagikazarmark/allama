# allama-http

[![crates.io](https://img.shields.io/crates/v/allama-http?style=flat-square)](https://crates.io/crates/allama-http)
[![docs.rs](https://img.shields.io/docsrs/allama-http?style=flat-square)](https://docs.rs/allama-http)

The HTTP transport seam of `allama`'s HTTP provider crates
(`allama-anthropic`, `allama-gemini`, `allama-openai-compat`,
`allama-openai`, `allama-systemone`).

Providers build an `HttpRequest`, send it through an `HttpTransport`, and
decode the `HttpResponse`; request mapping, error classification and stream
decoding never touch an HTTP client directly. That makes the transport
replaceable:

| Transport | Use |
| --- | --- |
| `ReqwestTransport` (feature `reqwest`, default) | What providers use unless told otherwise, with a 10 s connect timeout (`DEFAULT_CONNECT_TIMEOUT`) unless you pass your own client. `reqwest-middleware` sends through your `reqwest-middleware` client; `http-tracing` adds `reqwest-tracing` spans (which do not propagate trace context; see below). TLS: `default-tls` (default), `rustls`, `native-tls`, or `rustls-ring` (`ring` instead of `aws-lc-rs`, so static musl builds need no cmake). On `wasm32-unknown-unknown` it sends through the host's `fetch` |
| `ReplayTransport` | Serves canned responses in memory and records requests: run a real adapter over recorded wire data with no server, on any target |
| your own | Another HTTP stack, a proxy, recording |

```rust
use allama::prelude::*;
use allama_http::{ReplayResponse, ReplayTransport};

#[tokio::main]
async fn main() -> allama::Result<()> {
    let transport = ReplayTransport::new([ReplayResponse::json(
        200,
        r#"{"id":"msg_1","model":"claude-haiku-4-5","stop_reason":"end_turn",
            "content":[{"type":"text","text":"Hello"}],"usage":{"input_tokens":3,"output_tokens":1}}"#,
    )]);
    let provider = allama_anthropic::Provider::builder()
        .api_key("test")
        .http_transport(transport.clone())
        .build()?;
    let response = provider.chat(&ChatRequest::new("claude-haiku-4-5").user("Hi")).await?;
    assert_eq!(response.text().as_deref(), Some("Hello"));
    assert_eq!(transport.requests()[0].url, "https://api.anthropic.com/v1/messages");
    Ok(())
}
```

## Timeouts

| Bound | Default | Change it |
| --- | --- | --- |
| Connect (TCP + TLS) | 10 s in `ReqwestTransport::new` (not on `wasm32`) | the provider builder's `http_transport(ReqwestTransport::from_client(...))` |
| Whole non-streaming call | 120 s (`DEFAULT_REQUEST_TIMEOUT`), whatever the transport, set per request by the provider crates | the provider builder's `timeout(...)`; `timeout(None)` removes it |
| Stream headers (`setup`), gap between stream chunks (`idle`) | 120 s and 300 s (`StreamTimeouts::default`); none for the Ollama and vLLM presets | the provider builder's `stream_timeouts(...)` |

The stream bounds are enforced above the transport, so they apply to any
`HttpTransport`. The idle default is long on purpose: reasoning models can
think for minutes before their first token, and the bound exists to catch
dead connections, not slow models. For tighter, workload-specific limits
wrap the provider in `allama::wrappers::TimeoutProvider`; whichever bound is
shorter fires first, and both fail with a retryable `Error::Timeout`.

## Network defaults

What a broken or hostile endpoint can make a caller do is bounded:

- **No redirects.** `ReqwestTransport::new` never follows one: `reqwest`
  would re-send `x-api-key`, `x-goog-api-key`, `api-key` and gateway
  credential headers (and, on a `307`/`308`, the prompt) to the
  `Location` host. The bundled providers report a `3xx` as
  `Error::UnexpectedResponse` naming that host (`redirect_error`). A
  client passed to `ReqwestTransport::from_client` keeps its own policy:
  build it with `.redirect(reqwest::redirect::Policy::none())`. On
  `wasm32`, the host `fetch`'s policy applies.
- **Bounded bodies.** `HttpResponse::bytes_limited` reads a success body
  under the builder's `max_response_size` (`DEFAULT_MAX_RESPONSE_SIZE`,
  256 MiB); `HttpResponse::map_error` reads at most 64 KiB of an error
  body within 30 s and records a failed read on the error. A stream is
  bounded per event (`max_stream_event_size`) and in time (the stream
  timeouts), not in total bytes: its length is the answer's, which
  `max_output_tokens` bounds, and a byte cap would cut off a legitimate
  long answer mid-stream.
- **Checked configuration.** `config::base_url`, `config::secret` and
  `config::static_headers` refuse malformed base URLs, keys no header can
  carry, and credential headers among a builder's `header(..)` settings,
  at `build()`, with a `Configuration` error naming the setting.
- **A `User-Agent`.** The bundled providers send `allama/<version>`
  (`DEFAULT_USER_AGENT`) unless `header("user-agent", ..)` replaces it.

## Versioning

This crate names these third-party crates in its public API, so a new
major (pre-1.0: minor) version of any of them is a breaking release of
`allama-http`:

- `bytes` 1: `Bytes`, re-exported, carries request and response bodies;
- `futures-core` 0.3: the `Body` stream and the framers' stream bounds;
- `serde` and `serde_json` 1: the JSON form of `HttpRequestOptions`,
  `ReplayResponse::json_value`, and the JSON the error and framing helpers
  read;
- `reqwest` 0.13 (feature `reqwest`): `ReqwestTransport::from_client`;
- `reqwest-middleware` 0.5 (feature `reqwest-middleware`):
  `ReqwestTransport::from_middleware`.

Provider crates take a client only as
`http_transport(ReqwestTransport::from_client(..))`, so `reqwest` appears
in no other public signature.

The allama crates are released together, at one version, for now, so a
breaking release of `allama-http` is today one of every allama crate.
That is how 0.1 is released, not a promise about every later release. The
reason is `HttpRequestOptions`: options are found by their type, so with
`allama-http` at two versions over one `allama`, an adapter built on one
version reads a different type than the one attached with the other. The
call fails with `Error::InvalidRequest`, naming the `allama-http`
namespace and both types, instead of sending the configured key without
the tenant's credential and headers; with one version for every crate,
the same mix is a compile error.

## Per-request headers and credentials

`HttpRequestOptions` sends extra HTTP headers and credentials with one
request: gateway tags (tenant, route, trace) and per-tenant keys. Attach it
to a chat, embedding or decision request like any options; every bundled
HTTP adapter applies it with `HttpRequest::apply_options`.

```rust
use allama::ChatRequest;
use allama_http::{Credential, HttpRequestOptions};

let request = ChatRequest::new("gpt-4o-mini").user("Hi").with_option(
    HttpRequestOptions::new()
        .header("x-tenant-id", "acme")
        .credential("openai", Credential::api_key("sk-tenant-key")),
);
```

A plain header replaces the adapter's header of the same name, and others
are added. A credential is addressed to a **credential scope**, the account
it belongs to, and only the adapter with that scope sends it: in a fallback
chain from OpenAI to Anthropic, an OpenAI-scoped key never reaches
Anthropic. `Credential::api_key` replaces the adapter's configured key in
the place its API expects; `Credential::header` is any other auth header,
such as a gateway token (`cf-aig-authorization`). Bundled adapters default
their scope to their provider name, publish it as `DEFAULT_CREDENTIAL_SCOPE`,
and change it with their builder's `credential_scope`; two accounts of one
provider need two scopes. A builder given a scope and no key requires a
credential on every request (bring your own key).

A plain header that would carry a credential (`authorization`, `x-api-key`,
`cf-aig-authorization`, ... see `is_credential_header`, or the adapter's own
key header) is refused with `Error::Configuration` before anything is sent.
Credentials never leave the process through the option: `Debug` redacts
them, serialization omits them (so recorded exchanges never hold them), and
the JSON form (`with_options_json("allama-http", ..)`) rejects them.

Third-party HTTP adapters declare a `CredentialSlot` (scope, placement:
`Bearer`, `Header(name)` or `Custom`, and the configured headers it
replaces), call `apply_options(&request.options, &slot)` after setting their
own headers, and run `allama_conformance::assert_transport_options_applied`.

## Trace context propagation

`allama-http` sends no trace context of its own and depends on no
OpenTelemetry crate. To continue the application's trace in a gateway
(LiteLLM proxy, vLLM, an instrumented internal gateway), wrap the
transport in an `HttpTransport` that injects the current span's W3C
`traceparent` into `HttpRequest::headers` before it sends, with your own
OpenTelemetry version. Adapters call `apply_options` before `send`, so the
wrapper sees the final headers and can keep a `traceparent` the caller
set. The
[observability guide](../../docs/guides/observability/README.md#trace-context-propagation)
has the recipe, for provider builders and for the provider registry, and
the `trace_context` example in `allama-examples` runs it.

## Secrets in logs

`Debug` output of `Headers`, `HttpRequest` and `HttpResponse` redacts
credential headers (`authorization`, `x-api-key`, `cookie`, … and any name
containing `key`, `token`, `secret` or `auth`) and every credential
`apply_options` placed; `is_sensitive_header` is the rule, for provider
crates that print their own configuration. It only redacts: a plain
`Idempotency-Key` header is still sent.

## Error helpers

Every HTTP provider, bundled or not, shares these:

- `error_context(status, &headers)` reads an error response's status,
  request id (`x-request-id` or `request-id`), retry hint (`Retry-After`,
  `retry-after-ms`) and server retry directive (`x-should-retry`) into an
  `ErrorContext`. Start the error from it, then mark the message authored.
- `error_envelope(body)` reads the common JSON error envelopes (OpenAI,
  which current vLLM, Ollama `/v1` and Anthropic also send; legacy vLLM;
  native Ollama). `ErrorEnvelope::apply` attaches the provider's message as
  the error's upstream message, so it stays out of the redacted
  `Error::as_log()`; a body that is no envelope stays only in the capped
  body. `error_envelope_value(&value)` reads a body already parsed as JSON
  (an NDJSON line, an SSE event's data).
- `TransportError::into_error` maps a request that got no response. Its
  message names the failure; the transport's own text, which can contain
  the request URL, is the upstream message. `ReqwestTransport` also drops
  the query string (some APIs put the key there) from its error text.
  `TransportError::into_stream_error` maps a streamed body that could not
  be read to the end the same way: a timeout stays `Error::Timeout`,
  anything else is a retryable transport stream error.

## Streaming helpers

A streamed response arrives in frames of its wire framing. A framer turns
the body into frames, and one driver runs the adapter's decoder over them,
whatever the framing:

- `SseStream::new(response.into_body())` (module `sse`) yields the body's
  SSE events (`event`, `data`, `id`). It handles `LF`, `CR` and `CRLF` line
  endings (also split across chunks), multi-line `data:` fields, a byte
  order mark, comments and `retry:` lines, scanning each byte once. A
  failed read stays the body's own error (`SseError::Transport`), so a
  stream timeout stays a timeout. A line or an event's data past the size
  cap (64 MiB by default, `max_event_size`, which provider builders expose
  as `max_stream_event_size`) fails with `SseError::EventTooLarge`, a
  stream protocol error, not a retryable one. A line that is not UTF-8
  fails with `SseError::InvalidUtf8`, where the WHATWG standard would
  replace the bytes with U+FFFD: a replaced byte would silently change the
  JSON an adapter decodes. It is a stream decode error, not a retryable one
  either. `SseError::into_error` maps all three. `SseDecoder` is the same
  decoder without the stream.
- `NdjsonStream::new(response.into_body())` (module `ndjson`) yields the
  body's newline-delimited JSON lines, with the same rules where they
  apply: `CRLF` endings (also split across chunks), blank lines skipped, a
  last line without its line ending kept, a size cap (`max_line_size`,
  64 MiB by default, `NdjsonError::LineTooLarge`), and invalid UTF-8
  failing (`NdjsonError::InvalidUtf8`). `NdjsonDecoder` is the same framer
  without the stream.
- `FramedChatStream::new(frames, decoder)` (module `framing`) drives an
  adapter's `ChatFrameDecoder` (`frame`, `end`, `fail`, `is_done`) over any
  stream of frames: the stream ends after its first error, the end-of-body
  or failure hook runs at most once, a failure is classified with the
  framer's error conversion (`SseError::into_error`,
  `NdjsonError::into_error`), and the body is never polled after the
  stream ended. `with_error_map` takes a body whose error is not a
  `TransportError`. `SseChatStream` is its SSE form, a type alias for
  `FramedChatStream<SseStream<_>, _>` driving a `ChatFrameDecoder<SseEvent>`.
- `VendorFrames::surface(event, data)` (module `framing`, also in `sse`)
  turns a frame of a type the adapter does not model into response-level
  provider data under its namespace (that frame's payload), with one
  warning per frame type. Yield it as it comes: `Prepared::finish_stream`
  holds what arrives before `ResponseStart` until it. `finish()`, yielded
  just before `ResponseStop`, carries the whole array of each type that
  repeated, so each frame costs constant work.

This is a seam for provider crates, not middleware for applications: it
carries bytes, the shared error and streaming helpers, and transport options, and
the portable `allama` types stay above it. See ADR 0009.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
