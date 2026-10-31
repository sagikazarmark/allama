use std::error::Error as StdError;
use std::fmt;
use std::io::ErrorKind;
use std::time::Duration;

use futures_util::StreamExt;

use crate::{
    HttpRequest, HttpResponse, HttpTransport, SendFuture, TransportError, TransportErrorKind,
    is_sensitive_header,
};

#[cfg(feature = "reqwest-middleware")]
type Client = reqwest_middleware::ClientWithMiddleware;
#[cfg(not(feature = "reqwest-middleware"))]
type Client = reqwest::Client;

/// The connect timeout (TCP and TLS handshake) of [`ReqwestTransport::new`]:
/// 10 s.
///
/// Not applied on `wasm32`, where `reqwest` sends through the host's `fetch`
/// and has no connect timeout, and never applied to a client you pass to
/// [`ReqwestTransport::from_client`].
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// An [`HttpTransport`] backed by `reqwest`.
///
/// # Redirects
///
/// The client [`new`](Self::new) builds never follows a redirect: the
/// `3xx` response comes back as it is, and the bundled providers report it
/// as an [`allama::Error::UnexpectedResponse`] naming the target host
/// ([`redirect_error`](crate::redirect_error)). An LLM API never answers a
/// request with a redirect, and following one is unsafe: `reqwest` strips
/// only `Authorization` (and cookies) on a redirect to another host, so
/// `x-api-key`, `x-goog-api-key`, `api-key` and gateway credential headers
/// would reach the new host, a `307` or `308` would replay the prompt to
/// it, and an `https` to `http` downgrade would be followed. A redirect
/// would also bypass any allowlist the application checks a base URL
/// against.
///
/// A client you pass to [`from_client`](Self::from_client) or
/// `from_middleware` (feature `reqwest-middleware`) keeps its own redirect
/// policy, which is `reqwest`'s default (up to 10 redirects) unless you set
/// one: build it with `.redirect(reqwest::redirect::Policy::none())` to keep
/// this guarantee. On `wasm32`, `reqwest` sends through the host's `fetch`,
/// whose redirect handling applies and cannot be changed from here.
///
/// With the `reqwest-middleware` feature, requests go through a
/// `reqwest-middleware` client (`from_middleware`
/// takes your own); the `http-tracing` feature also adds
/// `reqwest_tracing::TracingMiddleware` to the clients
/// [`new`](Self::new) and [`from_client`](Self::from_client) build.
///
/// Clone is cheap: `reqwest` clients are reference counted.
#[derive(Clone)]
pub struct ReqwestTransport {
    client: Client,
}

impl fmt::Debug for ReqwestTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReqwestTransport").finish_non_exhaustive()
    }
}

impl Default for ReqwestTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl ReqwestTransport {
    /// A transport with a default `reqwest` client that never follows
    /// redirects (see [Redirects](Self#redirects)), with a
    /// [`DEFAULT_CONNECT_TIMEOUT`] (both except on `wasm32`).
    ///
    /// With the `rustls-ring` feature, it first installs `rustls`' `ring`
    /// crypto provider as the process default, unless the application
    /// installed one. A client you build yourself for
    /// [`from_client`](Self::from_client) needs a provider installed the
    /// same way (`rustls::crypto::ring::default_provider().install_default()`)
    /// before it is built.
    ///
    /// # Panics
    ///
    /// When the TLS backend cannot be initialized, like
    /// `reqwest::Client::new`.
    #[must_use]
    pub fn new() -> Self {
        // `Err` means a provider is already installed, which is then used.
        #[cfg(all(feature = "rustls-ring", not(target_arch = "wasm32")))]
        let _ = rustls::crypto::ring::default_provider().install_default();
        let builder = reqwest::Client::builder();
        #[cfg(not(target_arch = "wasm32"))]
        let builder = builder
            .connect_timeout(DEFAULT_CONNECT_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none());
        Self::from_client(builder.build().expect("failed to build the reqwest client"))
    }

    /// A transport over `client`, whose timeouts, redirect policy and
    /// other settings are used as they are (wrapped with tracing middleware
    /// under `http-tracing`). Unlike [`new`](Self::new)'s, a client from
    /// `reqwest::Client::new()` follows redirects: see
    /// [Redirects](Self#redirects).
    ///
    /// ```
    /// use std::time::Duration;
    /// use allama_http::ReqwestTransport;
    ///
    /// let client = reqwest::Client::builder()
    ///     .redirect(reqwest::redirect::Policy::none())
    ///     .connect_timeout(Duration::from_secs(5))
    ///     .build()?;
    /// let transport = ReqwestTransport::from_client(client);
    /// # let _ = transport;
    /// # Ok::<(), reqwest::Error>(())
    /// ```
    #[must_use]
    pub fn from_client(client: reqwest::Client) -> Self {
        #[cfg(feature = "reqwest-middleware")]
        let client = {
            let builder = reqwest_middleware::ClientBuilder::new(client);
            #[cfg(feature = "http-tracing")]
            let builder = builder.with(reqwest_tracing::TracingMiddleware::<
                reqwest_tracing::SpanBackendWithUrl,
            >::new());
            builder.build()
        };
        Self { client }
    }

    /// A transport over a middleware client, used unchanged (no tracing
    /// middleware is added, even under `http-tracing`), its redirect policy
    /// included: see [Redirects](Self#redirects).
    #[cfg(feature = "reqwest-middleware")]
    #[must_use]
    pub fn from_middleware(client: reqwest_middleware::ClientWithMiddleware) -> Self {
        Self { client }
    }
}

impl HttpTransport for ReqwestTransport {
    fn send(&self, request: HttpRequest) -> SendFuture<'_> {
        Box::pin(single_threaded(async move {
            let built = self.build(request)?;
            record_server(built.url());
            let mut response = self.client.execute(built).await.map_err(send_error)?;
            let status = response.status();
            let headers = std::mem::take(response.headers_mut());
            let body = response
                .bytes_stream()
                .map(|chunk| chunk.map_err(|e| with_cause(e, TransportErrorKind::Body)));
            Ok(HttpResponse::new(
                status,
                headers,
                Box::pin(single_threaded(body)),
            ))
        }))
    }
}

impl ReqwestTransport {
    /// The `reqwest` request for `request`, the values of credential
    /// headers marked sensitive.
    ///
    /// `reqwest` speaks the same `http` types, so only the URI is converted:
    /// to the `reqwest::Url` it sends to, which fails (an
    /// [`InvalidRequest`](TransportErrorKind::InvalidRequest)) for a URI
    /// that is not absolute.
    fn build(&self, request: HttpRequest) -> Result<reqwest::Request, TransportError> {
        let HttpRequest {
            method,
            uri,
            mut headers,
            body,
            timeout,
        } = request;
        let url = reqwest::Url::parse(&uri.to_string()).map_err(|_| {
            TransportError::new(
                TransportErrorKind::InvalidRequest,
                "the request URI is not an absolute URL",
            )
            .authored()
        })?;
        // Marked sensitive, so `reqwest` and the `http` crate keep them out
        // of their own `Debug` output.
        for (name, value) in &mut headers {
            if is_sensitive_header(name.as_str()) {
                value.set_sensitive(true);
            }
        }
        let mut built = reqwest::Request::new(method, url);
        *built.headers_mut() = headers;
        if !body.is_empty() {
            *built.body_mut() = Some(body.into());
        }
        *built.timeout_mut() = timeout;
        Ok(built)
    }
}

/// On `wasm32`, `reqwest` futures and streams wrap JS objects and are
/// `!Send`. The JS runtimes `reqwest` targets there run them on one thread,
/// so `SendWrapper` satisfies the `Send` bounds of [`SendFuture`] and
/// [`Body`](crate::Body); it panics if the value is ever touched from
/// another thread.
#[cfg(target_arch = "wasm32")]
fn single_threaded<T>(value: T) -> send_wrapper::SendWrapper<T> {
    send_wrapper::SendWrapper::new(value)
}

#[cfg(not(target_arch = "wasm32"))]
fn single_threaded<T>(value: T) -> T {
    value
}

#[cfg(feature = "reqwest-middleware")]
fn send_error(err: reqwest_middleware::Error) -> TransportError {
    match err {
        // `Reqwest` is transparent: its `source` skips the reqwest error.
        reqwest_middleware::Error::Reqwest(inner) => with_cause(inner, TransportErrorKind::Connect),
        // A middleware's message is its own text, which may quote the URL:
        // redacted the same way. Its error is not attached: its chain is
        // the middleware's own and may quote the URL too.
        err @ reqwest_middleware::Error::Middleware(_) => {
            let message = crate::config::redact_urls_in(&err.to_string()).into_owned();
            classify(&err, TransportErrorKind::Connect, message)
        }
    }
}

/// Tell the call scope which server the request goes to, for the span's
/// `server.address` and `server.port`.
fn record_server(url: &reqwest::Url) {
    use allama::wrappers::call_scope::{self, CallFact};

    if let Some(host) = url.host_str() {
        call_scope::record(CallFact::server(
            host.to_owned(),
            url.port_or_known_default(),
        ));
    }
}

#[cfg(not(feature = "reqwest-middleware"))]
fn send_error(err: reqwest::Error) -> TransportError {
    with_cause(err, TransportErrorKind::Connect)
}

/// The transport error for a `reqwest` failure, with its URL redacted and
/// the redacted error attached as the source, so a caller can tell a TLS
/// failure from a DNS one or a reset by downcasting it.
fn with_cause(err: reqwest::Error, fallback: TransportErrorKind) -> TransportError {
    let err = redact_url(err);
    classify(&err, fallback, err.to_string()).with_source(err)
}

/// Drop the query string, fragment, user name and password from the URL a
/// `reqwest` error shows: some APIs put the key in the query string
/// (`?key=`), and a base URL can carry a password.
fn redact_url(mut err: reqwest::Error) -> reqwest::Error {
    if let Some(url) = err.url_mut() {
        url.set_query(None);
        url.set_fragment(None);
        let _ = url.set_password(None);
        let _ = url.set_username("");
    }
    err
}

/// Classify `err` (walking its sources) as a timeout, an unbuildable
/// request, or `fallback`.
fn classify(
    err: &(dyn StdError + 'static),
    fallback: TransportErrorKind,
    message: String,
) -> TransportError {
    let mut kind = fallback;
    let mut current = Some(err);
    while let Some(err) = current {
        if let Some(reqwest_err) = err.downcast_ref::<reqwest::Error>() {
            if reqwest_err.is_timeout() {
                kind = TransportErrorKind::Timeout;
                break;
            }
            if reqwest_err.is_builder() {
                kind = TransportErrorKind::InvalidRequest;
                break;
            }
        }
        if let Some(io_err) = err.downcast_ref::<std::io::Error>()
            && io_err.kind() == ErrorKind::TimedOut
        {
            kind = TransportErrorKind::Timeout;
            break;
        }
        current = err.source();
    }
    TransportError::new(kind, message)
}

// These tests need `allama-conformance`, a path-only dev-dependency that
// publishing strips (it depends on this crate), so a packaged `allama-http`
// builds its tests only without `reqwest`.
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::time::Duration;

    use allama_conformance::{MockHttpResponse, TestHttpServer};
    use http::{HeaderValue, StatusCode, Uri};

    use super::*;

    fn uri(url: &str) -> Uri {
        Uri::try_from(url).unwrap()
    }

    #[tokio::test]
    async fn sends_requests_and_streams_bodies() {
        let server = TestHttpServer::spawn([MockHttpResponse::text(StatusCode::CREATED, "hello")
            .header("x-request-id", HeaderValue::from_static("req_1"))])
        .await;
        let transport = ReqwestTransport::new();
        let response = transport
            .send(
                HttpRequest::post(uri(&format!("{}/v1/items", server.url())), "{\"a\":1}")
                    .header("authorization", HeaderValue::from_static("Bearer k"))
                    .timeout(Some(Duration::from_secs(5))),
            )
            .await
            .unwrap();
        assert_eq!(response.status, StatusCode::CREATED);
        assert_eq!(response.headers["X-Request-Id"], "req_1");
        assert_eq!(response.text().await.unwrap(), "hello");

        let recorded = &server.recorded_requests().await[0];
        assert_eq!(
            (recorded.method.as_str(), recorded.path.as_str()),
            ("POST", "/v1/items")
        );
        assert_eq!(recorded.header("authorization"), Some("Bearer k"));
        assert_eq!(recorded.body_json()["a"], 1);
    }

    /// An empty body is sent as no body, on a `GET` and on a `POST` (the
    /// Responses API's cancel) alike.
    #[tokio::test]
    async fn empty_bodies_are_sent_as_no_body() {
        let server = TestHttpServer::spawn([
            MockHttpResponse::text(StatusCode::OK, "a"),
            MockHttpResponse::text(StatusCode::OK, "b"),
        ])
        .await;
        let transport = ReqwestTransport::new();
        for request in [
            HttpRequest::get(uri(&format!("{}/v1/models", server.url()))),
            HttpRequest::post(uri(&format!("{}/v1/cancel", server.url())), ""),
        ] {
            transport.send(request).await.unwrap();
        }
        let recorded = server.recorded_requests().await;
        assert_eq!(recorded.len(), 2);
        assert_eq!(recorded[0].method, "GET");
        assert_eq!(recorded[1].method, "POST");
        for request in &recorded {
            assert!(request.body.is_empty(), "{request:?}");
            assert_eq!(request.header("transfer-encoding"), None);
        }
    }

    /// With `rustls-ring` alone, `reqwest` ships no crypto provider and
    /// panics when it builds a client unless one is installed.
    #[cfg(feature = "rustls-ring")]
    #[test]
    fn rustls_ring_installs_a_crypto_provider() {
        let _transport = ReqwestTransport::new();
        assert!(rustls::crypto::CryptoProvider::get_default().is_some());
        // A second transport finds the provider installed.
        let _transport = ReqwestTransport::new();
    }

    #[tokio::test]
    async fn failures_are_classified() {
        let refused = ReqwestTransport::new()
            .send(HttpRequest::get(uri("http://127.0.0.1:1/")))
            .await
            .unwrap_err();
        assert_eq!(refused.kind(), TransportErrorKind::Connect);

        // A URI `http` accepts but no client can send to: a path alone.
        let unbuildable = ReqwestTransport::new()
            .send(HttpRequest::get(uri("/v1/models")))
            .await
            .unwrap_err();
        assert_eq!(unbuildable.kind(), TransportErrorKind::InvalidRequest);
    }

    /// Credential headers are marked sensitive on the `reqwest` request,
    /// so `reqwest` and `http` redact them in their own `Debug` output.
    #[test]
    fn sensitive_headers_are_marked_on_the_request() {
        let built = ReqwestTransport::new()
            .build(
                HttpRequest::get(uri("http://127.0.0.1:1/v1/models"))
                    .header("authorization", HeaderValue::from_static("Bearer SECRET"))
                    .header("x-api-key", HeaderValue::from_static("SECRET"))
                    .sensitive_header("x-route", HeaderValue::from_static("SECRET"))
                    .header("x-request-id", HeaderValue::from_static("req_1")),
            )
            .unwrap();
        let headers = built.headers();
        for name in ["authorization", "x-api-key", "x-route"] {
            assert!(headers[name].is_sensitive(), "{name}");
        }
        assert!(!headers["x-request-id"].is_sensitive());
        assert!(!format!("{built:?}").contains("SECRET"));
    }

    /// A response header value that is not visible ASCII (obs-text) is
    /// kept as sent, not dropped.
    #[tokio::test]
    async fn obs_text_response_headers_are_kept() {
        let server = TestHttpServer::spawn([MockHttpResponse::text(StatusCode::OK, "ok").header(
            "x-region",
            HeaderValue::from_bytes("zürich".as_bytes()).unwrap(),
        )])
        .await;
        let response = ReqwestTransport::new()
            .send(HttpRequest::get(uri(&format!(
                "{}/v1/models",
                server.url()
            ))))
            .await
            .unwrap();
        assert_eq!(response.headers["x-region"].as_bytes(), "zürich".as_bytes());
    }

    #[tokio::test]
    async fn error_text_drops_the_query_string() {
        let refused = ReqwestTransport::new()
            .send(HttpRequest::get(uri(
                "http://user:pw@127.0.0.1:1/v1/models?key=SECRET",
            )))
            .await
            .unwrap_err();
        let text = refused.to_string();
        assert!(!text.contains("SECRET"), "{text}");
        assert!(!text.contains("pw"), "{text}");
        assert!(!text.contains("user"), "{text}");
        assert!(text.contains("127.0.0.1:1/v1/models"), "{text}");

        // The redacted `reqwest` error is the source, and nothing in the
        // chain shows the secrets.
        let err = refused.into_error();
        let source = std::error::Error::source(&err).expect("the cause is attached");
        let cause = source
            .downcast_ref::<reqwest::Error>()
            .expect("the source is the reqwest error");
        assert!(cause.is_connect() || cause.is_request(), "{cause:?}");
        let mut current: Option<&(dyn std::error::Error + 'static)> = Some(source);
        while let Some(link) = current {
            for text in [link.to_string(), format!("{link:?}")] {
                assert!(!text.contains("SECRET"), "{text}");
                assert!(!text.contains("pw"), "{text}");
                assert!(!text.contains("user"), "{text}");
            }
            current = link.source();
        }
    }

    /// A key in the query string (Gemini's `?key=`) reaches neither the
    /// redacted log, with or without the upstream message, nor `Display`.
    #[tokio::test]
    async fn query_string_key_stays_out_of_the_error() {
        let err = ReqwestTransport::new()
            .send(HttpRequest::get(uri(
                "http://127.0.0.1:1/v1beta/models/m:generateContent?key=secret",
            )))
            .await
            .unwrap_err()
            .into_error();
        assert!(matches!(err, allama::Error::Transport { .. }), "{err:?}");
        for text in [
            err.as_log().to_string(),
            err.as_log().with_message(10_000).to_string(),
            serde_json::to_string(&err.as_log().with_message(10_000)).unwrap(),
            err.to_string(),
            format!("{err:?}"),
        ] {
            assert!(!text.contains("secret"), "{text}");
        }
        assert_eq!(
            err.as_log().to_string(),
            r#"kind=transport message="the request got no response""#
        );
    }
}

/// Run on `wasm32` with `wasm-bindgen-test`
/// (`cargo test --target wasm32-unknown-unknown -p allama-http`): `reqwest`
/// sends through the `fetch` of Node, against a Node HTTP server.
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests {
    use std::time::Duration;

    use http::{HeaderValue, StatusCode, Uri};
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    use super::*;

    fn uri(url: &str) -> Uri {
        Uri::try_from(url).unwrap()
    }

    #[wasm_bindgen(inline_js = r#"
        export async function startServer() {
            const http = await import("node:http");
            const server = http.createServer((req, res) => {
                if (req.url === "/slow") {
                    setTimeout(() => res.end("late"), 1000);
                    return;
                }
                let body = "";
                req.on("data", (chunk) => (body += chunk));
                req.on("end", () => {
                    res.writeHead(201, { "x-request-id": "req_1" });
                    res.write(`${req.method} ${req.url} ${req.headers.authorization} `);
                    setTimeout(() => res.end(body), 10);
                });
            });
            server.unref();
            await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
            return `http://127.0.0.1:${server.address().port}`;
        }
    "#)]
    extern "C" {
        #[wasm_bindgen(js_name = startServer)]
        async fn start_server() -> JsValue;
    }

    async fn server_url() -> String {
        start_server().await.as_string().unwrap()
    }

    #[wasm_bindgen_test]
    async fn sends_requests_and_streams_bodies() {
        let url = server_url().await;
        let response = ReqwestTransport::new()
            .send(
                HttpRequest::post(uri(&format!("{url}/v1/items")), "{\"a\":1}")
                    .header("authorization", HeaderValue::from_static("Bearer k"))
                    .timeout(Some(Duration::from_secs(5))),
            )
            .await
            .unwrap();
        assert_eq!(response.status, StatusCode::CREATED);
        assert_eq!(response.headers["X-Request-Id"], "req_1");
        assert_eq!(
            response.text().await.unwrap(),
            "POST /v1/items Bearer k {\"a\":1}"
        );
    }

    #[wasm_bindgen_test]
    async fn failures_are_classified() {
        let url = server_url().await;
        let slow = ReqwestTransport::new()
            .send(
                HttpRequest::get(uri(&format!("{url}/slow")))
                    .timeout(Some(Duration::from_millis(50))),
            )
            .await
            .unwrap_err();
        assert_eq!(slow.kind(), TransportErrorKind::Timeout);

        let unbuildable = ReqwestTransport::new()
            .send(HttpRequest::get(uri("/v1/models")))
            .await
            .unwrap_err();
        assert_eq!(unbuildable.kind(), TransportErrorKind::InvalidRequest);
    }
}
