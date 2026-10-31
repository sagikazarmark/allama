# ADR 0020: A tower service is a transport through an adapter that keeps the seam's guarantees

- Status: accepted
- Amends: ADR 0009 (Timeouts; Cargo features), ADR 0019 (Consequences)

## Context

Applications that already run a `tower` stack (concurrency and rate
limits, `tower-http` instrumentation, `tower-reqwest`, a Workers `fetch`
service) could not put it under a provider without writing an
`HttpTransport` of their own. ADR 0019 made the seam's vocabulary `http`'s
so that such a service would plug in "with no mapping of its own". It
does not quite: no client service accepts `http::Request<Bytes>` (`Bytes`
is not an `http_body::Body`; `tower-reqwest` takes `reqwest::Body`), and
nothing in the `tower` ecosystem reads allama's `RequestTimeout`
extension (`reqwest` reads a private extension of its own,
`tower-worker-fetch` its own `Timeout`). A service that is simply called
would drop the seam's guarantees: the request timeout, which bounds every
unary call whatever the transport (ADR 0009); the failure classes
providers and retries decide on; URL redaction in error text; and no
redirects.

ADR 0013 rejected `tower` for provider middleware, and ADR 0019 rejected a
`tower::Service` as the seam. Neither applies here: this is about what
sits under the seam, carrying bytes.

## Decision

- **`TowerTransport<S>`** (feature `tower` of `allama-http`, off by
  default) is an `HttpTransport` over any `tower` service that takes an
  `http::Request<RequestBody>` and returns an `http::Response` with any
  `http_body::Body`. `RequestBody` is `http_body_util::Full<Bytes>`: a
  body every client service accepts or converts in one `map_request`
  (`reqwest::Body::wrap`). One fixed body type keeps the transport's type
  inferable and nameable; a generic body type made inference fail for
  services generic over their body.
- **The service is `Clone + Send + Sync`**, cloned for each request and
  driven once (`poll_ready`, then `call`), as `tower` expects. A shared
  instance would lose wakers between concurrent requests; a stack that is
  not `Sync` uses `BoxCloneSyncService`.
- **The adapter enforces the request timeout** over the whole exchange
  (readiness, headers, body), checked before every body poll, as
  `reqwest`'s total timeout is. It still sets the `RequestTimeout`
  extension for a service that reads it. A timeout in the stack still
  applies, and the shorter fires first.
- **Failures are classified**: a `TransportError` the service returns is
  passed through, and one found inside a layer's own error (`Buffer`'s)
  keeps its kind and message; a `reqwest` error is classified as
  `ReqwestTransport` does and kept as the source, its URL redacted; an
  I/O timeout is a timeout; anything else is `Other` (`Body` while reading
  the body), its text as the upstream message with URLs and the request's
  query and user info removed, the error itself not kept. A stack
  classifies its own failures by returning a `TransportError` from a
  `map_err` layer; there is no classifier hook.
- **Parity with `ReqwestTransport`**: credential headers are marked
  sensitive when an `HttpRequest` converts to an `http::Request`, so a
  layer that logs them redacts them; the server is recorded for the call
  scope; a relative URI is refused before the service is called.
- **`ReqwestTransport::into_service()`** returns `ReqwestService`, the
  default transport's client as a `tower` service over the same request
  body, whose errors are `TransportError`s. It is the recommended bottom
  of a stack: no redirects, URL redaction and classification survive the
  layers, and it runs on `wasm32`.
- **Redirects stay the client's.** The adapter cannot stop a service from
  following one; the docs say to build the client with
  `Policy::none()` and keep `tower-http`'s `FollowRedirect` out.
- **Dependencies**: `tower-service` 0.3, `http-body` 1 and
  `http-body-util` 0.1, optional, under `tower`; not `tower` itself. They
  are public API under the feature, so a new major (pre-1.0: minor)
  version of any of them is a breaking release of `allama-http`.
  `tower-service` has stayed at 0.3 across `tower` 0.4 and 0.5.
- **Provider crates do not forward `tower`.** Its types are
  `allama-http`'s, so whoever builds a `TowerTransport` depends on
  `allama-http` already (ADR 0009's reasoning for `reqwest-middleware`).

## Consequences

- A `tower` stack goes under any provider built with its builder, with
  the same timeouts, failure classes and redaction as the default
  transport. Registry providers take no transport, so it is for providers
  the application builds.
- `tower`'s own timeout and load-shedding errors are transport failures
  until a `map_err` layer classifies them.
- A concurrency limit in the stack bounds requests being set up, not open
  streams, since `tower` releases its permit when the response headers
  arrive.
- Retry stays above the seam, in `RetryingProvider`; the docs say not to
  add a `tower` retry layer.
- The Workers `fetch` transport (`tower-worker-fetch`) needs no adapter of
  its own beyond its redirect and timeout extensions.

## Rejected alternatives

- **A generic request body (`TowerTransport<S, B: From<Bytes>>`).** The
  acceptance one-liner over `tower-reqwest` compiles without a
  `map_request`, but inference fails for services generic over their body
  (`FetchService`, `hyper-util`), and the body type leaks into every
  stored transport's type.
- **Holding the service in a `Mutex`** to accept stacks that are not
  `Sync`. It rules out `get_ref`, and `BoxCloneSyncService` covers the
  case.
- **Leaving the timeout to the service.** Nothing reads allama's
  extension, so a unary call over `tower-reqwest` would have no bound.
- **`impl Service for Arc<dyn HttpTransport>`**, or a generic adapter from
  any `HttpTransport` to a `tower` service. The first is an orphan
  implementation; the second is deferred until it is needed, since
  `ReqwestService` covers the stack over the default client.
- **A classifier hook** (`with_classifier(fn(&BoxError) -> ...)`). A
  `map_err` layer returning a `TransportError` is the `tower` way, and the
  adapter already recognizes it anywhere in the error's chain.
- **A `tower` retry layer** in the recommended stack. Retries need the
  provider's retry signals (ADR 0011) and must stop at a stream's commit
  point, which only `RetryingProvider` sees.
