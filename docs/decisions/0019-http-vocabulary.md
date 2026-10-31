# ADR 0019: The transport seam speaks the `http` crate's vocabulary

- Status: accepted
- Amends: ADR 0009 ("The seam": `HttpRequest` is plain data)

## Context

ADR 0009 made `HttpTransport` the seam between provider crates and HTTP
clients, with `HttpRequest` and `HttpResponse` as plain data of our own:
a `Method { Get, Post }`, a `url: String`, a `Headers` list of string
pairs, and a `status: u16`. Every transport we want next speaks `http`
1.x: a `tower::Service` adapter, a Workers `fetch` transport, `hyper`,
and the OpenTelemetry injectors an application wraps around a transport.
Each would re-implement the mapping between our types and `http`'s, and
`reqwest` itself already converts our strings back into `http` types on
every request.

The plain strings also deferred every check to the transport: a builder
setting no header could carry, or a model id no URI can hold, failed
inside `ReqwestTransport::build` as an `InvalidRequest`, on every request,
in a different place for every transport.

We considered making a `tower::Service` the seam instead. It is not
object-safe without boxing, takes `&mut self` and `poll_ready`, has no
error taxonomy of its own, and would make every provider crate's public
API name `tower`. Smithy, `azure_core` and `opentelemetry-http` keep a
trait of their own and make its vocabulary `http`-shaped; we do the same.

## Decision

- `HttpTransport` stays the seam: object-safe, `&self`, a streamed body,
  and `TransportError` as its error taxonomy.
- `HttpRequest` and `HttpResponse` stay our own `#[non_exhaustive]`
  structs, so `apply_options`, `sensitive_header`, `bytes_limited` and
  `map_error` keep a home, but they are made of `http` 1 types:
  `method: http::Method`, `uri: http::Uri`, `headers: http::HeaderMap`,
  `status: http::StatusCode`, and a `body: Bytes` that is empty when the
  request has none. `allama-http` re-exports the crate as
  `allama_http::http`, so an adapter or transport names the same version;
  `http` and `bytes` are its only HTTP-related public dependencies.
- They convert, losslessly for what they model, to and from
  `http::Request<Bytes>` and `http::Response<Body>`. The request's
  timeout, which `http` has no field for, travels as a `RequestTimeout`
  extension.
- The shared helpers take the same types: `error_context`,
  `status_error`, `redirect_error` and `malformed_response` take a
  `StatusCode`, and `map_error` passes `(StatusCode, &HeaderMap, &str)`.
  The portable `allama::Error::status()` stays a `u16`: the core crate
  does not depend on `http`.
- **Validation moves to build time.** Builders turn their configuration
  into the types requests are made of once: `config::static_headers`
  returns a `HeaderMap`, `config::sensitive_value` a `HeaderValue` marked
  sensitive for a key, and `config::base_url` checks that the base URL is
  a `Uri`. `request_uri` checks the part a request adds (a model or
  response id) as an authored `InvalidRequest` that never quotes the URL.
  A transport can no longer fail on a bad header.
- `HttpRequestOptions.headers` is a `HeaderMap` too. Its JSON form is
  untrusted input: a name or value no header can carry is rejected with
  `InvalidRequest`. Header names are exported lowercase.
- **Redaction is unchanged.** `Debug` of requests, responses and options
  redacts values marked `HeaderValue::set_sensitive` (every credential
  `apply_options` places) and names `is_sensitive_header` matches;
  `redacted_headers` prints any `HeaderMap` the same way.
- The conformance error contracts keep calling a mapping with a plain
  status and header pairs, since a mapping need not be HTTP's (the
  Workers AI binding's is not); `http_error_mapping` wraps an HTTP
  mapping for them.

## Consequences

- A `tower` service, a Workers `fetch` or `hyper` plugs in as a
  transport by converting through `http::Request<Bytes>` and
  `http::Response<Body>`, with no mapping of its own. ADR 0020 refines
  this for `tower`: an adapter still picks a request body every client
  takes and keeps the seam's timeout, failure classes and redaction.
- A malformed builder setting fails at `build()`, and a request fails
  before it is sent only on what it adds itself, with the same error
  whatever the transport.
- `http` 1 is in `allama-http`'s public API: its next major version is a
  breaking release of `allama-http` (and, while the crates release
  together, of every allama crate).
- `HeaderName` lowercases names, so `Debug` output and the exported JSON
  of `HttpRequestOptions` show lowercase names; HTTP names are
  case-insensitive, so nothing on the wire changes.
- A header value that is not visible ASCII (obs-text) is kept as bytes;
  the helpers read it lossily rather than dropping it.
- A URI holds fewer characters than `reqwest::Url` silently encoded: an id
  with a space now fails as an `InvalidRequest` instead of being encoded.

## Rejected alternatives

- **A `tower::Service` as the seam.** See Context.
- **A `Headers` newtype over `HeaderMap`.** It would keep a friendlier
  `get(&str) -> Option<&str>`, but every adapter would still convert, and
  the crate would carry two header vocabularies.
- **Generic `TryInto` builders that store the first error** (as
  `http::request::Builder` does). Call sites would barely change, but the
  request would stop being plain, inspectable data, and every transport
  would have to check for the stored error.
- **Fallible builders** (`try_header`). Every provider's request chain
  grows `?`s for values it validated at build time.
