use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_util::TryStreamExt;
use http_body::Frame;
use http_body_util::combinators::UnsyncBoxBody;
use http_body_util::{BodyExt, StreamBody};
use tower_service::Service;

use super::ReqwestTransport;
use crate::{HttpRequest, HttpTransport, RequestBody, TransportError};

/// [`ReqwestTransport`] as a `tower` service, for the bottom of a stack that
/// a [`TowerTransport`](crate::TowerTransport) makes a transport again:
/// the default transport's client, with your layers on top.
///
/// It keeps what the transport guarantees: a client from
/// [`ReqwestTransport::new`] follows no redirect, errors are
/// [`TransportError`]s, classified and with URLs redacted, which
/// `TowerTransport` passes through unchanged, and the request's
/// [`RequestTimeout`](crate::RequestTimeout) is honoured. It runs on
/// `wasm32` too, where `reqwest` sends through the host's `fetch`.
///
/// ```
/// use allama_http::{ReqwestTransport, TowerTransport};
/// use tower::ServiceBuilder;
///
/// let stack = ServiceBuilder::new()
///     .concurrency_limit(8)
///     .service(ReqwestTransport::new().into_service());
/// let transport = TowerTransport::new(stack);
/// # let _ = transport;
/// ```
#[derive(Clone)]
pub struct ReqwestService {
    transport: ReqwestTransport,
}

impl fmt::Debug for ReqwestService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReqwestService").finish_non_exhaustive()
    }
}

impl ReqwestTransport {
    /// This transport as a `tower` service ([`ReqwestService`]), for the
    /// bottom of a stack of `tower` layers that a
    /// [`TowerTransport`](crate::TowerTransport) wraps (feature `tower`).
    #[must_use]
    pub fn into_service(self) -> ReqwestService {
        ReqwestService { transport: self }
    }
}

impl Service<http::Request<RequestBody>> for ReqwestService {
    type Response = http::Response<ReqwestBody>;
    type Error = TransportError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    /// Always ready: the client's connection pool queues requests itself.
    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: http::Request<RequestBody>) -> Self::Future {
        let transport = self.transport.clone();
        Box::pin(async move {
            let (parts, body) = request.into_parts();
            let Ok(body) = body.collect().await;
            let request = HttpRequest::from(http::Request::from_parts(parts, body.to_bytes()));
            let response = transport.send(request).await?;
            let response = http::Response::<crate::Body>::from(response);
            Ok(response.map(|body| {
                StreamBody::new(body.map_ok(Frame::data as fn(Bytes) -> Frame<Bytes>))
                    .boxed_unsync()
            }))
        })
    }
}

/// The response body of [`ReqwestService`]: the transport's streamed body
/// as an [`http_body::Body`], its errors [`TransportError`]s.
pub type ReqwestBody = UnsyncBoxBody<Bytes, TransportError>;

// These tests need `allama-conformance`, a path-only dev-dependency that
// publishing strips, as the `reqwest` module's own tests do.
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::time::Duration;

    use allama_conformance::{MockHttpResponse, TestHttpServer};
    use http::{HeaderValue, StatusCode, Uri};

    use super::*;
    use crate::{TowerTransport, TransportErrorKind};

    fn uri(url: &str) -> Uri {
        Uri::try_from(url).unwrap()
    }

    fn transport() -> TowerTransport<ReqwestService> {
        TowerTransport::new(ReqwestTransport::new().into_service())
    }

    #[tokio::test]
    async fn sends_requests_and_streams_bodies_through_a_stack() {
        let server = TestHttpServer::spawn([MockHttpResponse::text(StatusCode::CREATED, "hello")
            .header("x-request-id", HeaderValue::from_static("req_1"))])
        .await;
        let stack = ::tower::ServiceBuilder::new()
            .concurrency_limit(2)
            .service(ReqwestTransport::new().into_service());
        let response = TowerTransport::new(stack)
            .send(
                HttpRequest::post(uri(&format!("{}/v1/items", server.url())), "{\"a\":1}")
                    .header("authorization", HeaderValue::from_static("Bearer k"))
                    .timeout(Some(Duration::from_secs(5))),
            )
            .await
            .unwrap();
        assert_eq!(response.status, StatusCode::CREATED);
        assert_eq!(response.headers["x-request-id"], "req_1");
        assert_eq!(response.text().await.unwrap(), "hello");

        let recorded = &server.recorded_requests().await[0];
        assert_eq!(
            (recorded.method.as_str(), recorded.path.as_str()),
            ("POST", "/v1/items")
        );
        assert_eq!(recorded.header("authorization"), Some("Bearer k"));
        assert_eq!(recorded.body_json()["a"], 1);
    }

    /// An empty body is still sent as no body.
    #[tokio::test]
    async fn empty_bodies_are_sent_as_no_body() {
        let server = TestHttpServer::spawn([MockHttpResponse::text(StatusCode::OK, "a")]).await;
        transport()
            .send(HttpRequest::get(uri(&format!(
                "{}/v1/models",
                server.url()
            ))))
            .await
            .unwrap();
        let recorded = &server.recorded_requests().await[0];
        assert!(recorded.body.is_empty(), "{recorded:?}");
        assert_eq!(recorded.header("transfer-encoding"), None);
    }

    /// A connection failure keeps the transport's classification and its
    /// redacted cause through the stack.
    #[tokio::test]
    async fn failures_keep_their_classification() {
        let err = transport()
            .send(HttpRequest::get(uri(
                "http://127.0.0.1:1/v1/models?key=SECRET",
            )))
            .await
            .unwrap_err();
        assert_eq!(err.kind(), TransportErrorKind::Connect);
        assert!(!err.to_string().contains("SECRET"), "{err}");
        let err = err.into_error();
        let source = std::error::Error::source(&err).expect("the cause is attached");
        assert!(
            source.downcast_ref::<reqwest::Error>().is_some(),
            "{source:?}"
        );
    }

    /// The request's timeout reaches the client, which bounds the exchange
    /// itself.
    #[tokio::test]
    async fn the_request_timeout_is_honoured() {
        let silent = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/v1/models", silent.local_addr().unwrap());
        std::thread::spawn(move || {
            let _held: Vec<_> = silent.incoming().collect();
        });
        let service = ReqwestTransport::new().into_service();
        let request = http::Request::<Bytes>::from(
            HttpRequest::get(uri(&url)).timeout(Some(Duration::from_millis(100))),
        )
        .map(RequestBody::new);
        let err = tower_service::Service::call(&mut service.clone(), request)
            .await
            .unwrap_err();
        assert_eq!(err.kind(), TransportErrorKind::Timeout);
    }
}

/// Run on `wasm32` with `wasm-bindgen-test`: `reqwest` sends through the
/// `fetch` of Node, against a Node HTTP server, under a `TowerTransport`.
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests {
    use http::{HeaderValue, StatusCode, Uri};
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    use super::*;
    use crate::TowerTransport;

    #[wasm_bindgen(inline_js = r#"
        export async function startEchoServer() {
            const http = await import("node:http");
            const server = http.createServer((req, res) => {
                let body = "";
                req.on("data", (chunk) => (body += chunk));
                req.on("end", () => {
                    res.writeHead(201, { "x-request-id": "req_1" });
                    res.end(`${req.method} ${req.url} ${req.headers.authorization} ${body}`);
                });
            });
            server.unref();
            await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
            return `http://127.0.0.1:${server.address().port}`;
        }
    "#)]
    extern "C" {
        #[wasm_bindgen(js_name = startEchoServer)]
        async fn start_echo_server() -> JsValue;
    }

    #[wasm_bindgen_test]
    async fn sends_through_a_tower_transport() {
        let url = start_echo_server().await.as_string().unwrap();
        let response = TowerTransport::new(ReqwestTransport::new().into_service())
            .send(
                HttpRequest::post(Uri::try_from(format!("{url}/v1/items")).unwrap(), "{}")
                    .header("authorization", HeaderValue::from_static("Bearer k")),
            )
            .await
            .unwrap();
        assert_eq!(response.status, StatusCode::CREATED);
        assert_eq!(response.headers["x-request-id"], "req_1");
        assert_eq!(response.text().await.unwrap(), "POST /v1/items Bearer k {}");
    }
}
