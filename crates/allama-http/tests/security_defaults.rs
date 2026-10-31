//! The security defaults of the default transport, against local servers:
//! a redirect is never followed, and bodies are read under a size cap.
//!
//! [`assert_redirect_not_followed`] is the transport-level form of the
//! cross-adapter check: an origin answers `307` with a `Location` on
//! another host, and the other host must never see the request.

#![cfg(all(not(target_arch = "wasm32"), feature = "reqwest"))]

use std::future::Future;

use allama::Error;
use allama_conformance::{MockHttpResponse, TestHttpServer};
use allama_http::{
    HttpRequest, HttpResponse, HttpTransport, ReqwestTransport, TransportError, TransportErrorKind,
    status_error,
};
use http::{HeaderMap, HeaderValue, StatusCode, Uri};

/// Spawn a target server and an origin that redirects every request to it
/// with `status`, run `send` against the origin's URL, and assert that the
/// target never received a request. Returns what `send` returned.
async fn assert_redirect_not_followed<F, Fut, T>(status: u16, send: F) -> T
where
    F: FnOnce(String) -> Fut,
    Fut: Future<Output = T>,
{
    let target =
        TestHttpServer::spawn([MockHttpResponse::json(StatusCode::OK, r#"{"stolen":true}"#)]).await;
    // `localhost` rather than `127.0.0.1`: another host as far as the
    // client can tell, so a followed redirect would be cross-host.
    let location = format!(
        "{}/v1/messages",
        target.url().replace("127.0.0.1", "localhost")
    );
    let origin =
        TestHttpServer::spawn([
            MockHttpResponse::text(StatusCode::from_u16(status).unwrap(), "")
                .header("location", HeaderValue::try_from(location).unwrap()),
        ])
        .await;
    let result = send(origin.url().to_owned()).await;
    assert_eq!(origin.recorded_requests().await.len(), 1);
    let leaked = target.recorded_requests().await;
    assert!(
        leaked.is_empty(),
        "the redirect was followed: the target got {:?}",
        leaked
            .iter()
            .map(|request| (request.header("x-api-key"), request.body_text().len()))
            .collect::<Vec<_>>()
    );
    result
}

fn map_http(status: StatusCode, _: &HeaderMap, _: &str) -> Error {
    status_error(status, format!("HTTP {}", status.as_u16())).authored()
}

/// The default transport returns the redirect instead of sending the
/// key and the prompt to the `Location` host, for every redirect status.
#[tokio::test]
async fn the_default_transport_never_follows_a_redirect() {
    for status in [301, 302, 303, 307, 308] {
        let response = assert_redirect_not_followed(status, |url| async move {
            ReqwestTransport::new()
                .send(
                    HttpRequest::post(
                        Uri::try_from(format!("{url}/v1/messages")).unwrap(),
                        r#"{"prompt":"secret"}"#,
                    )
                    .header("x-api-key", HeaderValue::from_static("SECRET-KEY-123")),
                )
                .await
                .unwrap()
        })
        .await;
        assert_eq!(response.status, status);
        let err = response.map_error(map_http).await;
        assert!(
            matches!(err, Error::UnexpectedResponse { .. }),
            "{status}: {err:?}"
        );
        assert_eq!(err.status(), Some(status));
        assert!(!err.is_retryable());
        assert!(
            err.as_log().to_string().contains("`localhost`"),
            "{}",
            err.as_log()
        );
    }
}

/// A client built by the caller keeps its own redirect policy, which is
/// theirs to choose (documented on `ReqwestTransport`).
#[tokio::test]
async fn a_client_without_redirects_keeps_the_guarantee() {
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    let response = assert_redirect_not_followed(307, |url| async move {
        ReqwestTransport::from_client(client)
            .send(HttpRequest::post(Uri::try_from(url).unwrap(), "{}"))
            .await
            .unwrap()
    })
    .await;
    assert_eq!(response.status, 307);
}

/// A success body past the limit fails without being buffered whole.
#[tokio::test]
async fn success_bodies_are_read_under_a_limit() {
    let big = "x".repeat(1024 * 1024);
    let server = TestHttpServer::spawn([
        MockHttpResponse::text(StatusCode::OK, big.clone()),
        MockHttpResponse::text(StatusCode::OK, big),
    ])
    .await;
    let transport = ReqwestTransport::new();
    let send = || transport.send(HttpRequest::get(Uri::try_from(server.url()).unwrap()));
    let err = send()
        .await
        .unwrap()
        .bytes_limited(64 * 1024)
        .await
        .unwrap_err();
    assert_eq!(err.kind(), TransportErrorKind::TooLarge);
    assert!(matches!(err.into_error(), Error::UnexpectedResponse { .. }));
    let whole = send()
        .await
        .unwrap()
        .bytes_limited(allama_http::DEFAULT_MAX_RESPONSE_SIZE)
        .await
        .unwrap();
    assert_eq!(whole.len(), 1024 * 1024);
}

/// An error body is read up to its cap, whatever its size.
#[tokio::test]
async fn error_bodies_are_capped() {
    let server = TestHttpServer::spawn([MockHttpResponse::text(
        StatusCode::INTERNAL_SERVER_ERROR,
        "e".repeat(1024 * 1024),
    )])
    .await;
    let response = ReqwestTransport::new()
        .send(HttpRequest::get(Uri::try_from(server.url()).unwrap()))
        .await
        .unwrap();
    let mut seen = 0;
    let err = response
        .map_error(|status, headers, body| {
            seen = body.len();
            map_http(status, headers, body)
        })
        .await;
    assert_eq!(seen, allama_http::MAX_ERROR_BODY_SIZE);
    assert!(matches!(err, Error::Provider { .. }), "{err:?}");
}

/// The cap applies to any body, whatever the transport: an endless error
/// body from a custom transport still ends the call.
#[tokio::test]
async fn an_endless_error_body_ends_the_call() {
    let endless = HttpResponse::new(
        StatusCode::SERVICE_UNAVAILABLE,
        HeaderMap::new(),
        Box::pin(futures_util::stream::repeat_with(|| {
            Ok::<_, TransportError>(allama_http::Bytes::from(vec![b'x'; 64 * 1024]))
        })),
    );
    let err = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        endless.map_error(map_http),
    )
    .await
    .expect("the error body read did not end");
    assert!(matches!(err, Error::Overloaded { .. }), "{err:?}");
}
