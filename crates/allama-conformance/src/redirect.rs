//! Redirects are never followed.
//!
//! A provider's HTTP client must not follow a redirect: `reqwest` strips
//! the `Authorization` header on a cross-host redirect, but not the
//! vendor key headers (`x-api-key`, `x-goog-api-key`, ...), and never the
//! prompt. The default transport refuses redirects, and an adapter must
//! surface the `3xx` as an error. [`assert_redirect_not_followed`] checks it
//! end to end, through the adapter's own client.

use std::future::Future;

use allama::Error;

use crate::{MockHttpResponse, TestHttpServer};

/// The redirect statuses the contract sends.
pub const REDIRECT_STATUSES: [u16; 5] = [301, 302, 303, 307, 308];

/// Assert that an adapter never follows a redirect, on `chat` and on
/// `chat_stream`.
///
/// For each redirect status, an origin server answers every request with
/// that status and a `Location` on another host (a second server). `send`
/// builds the provider under test with the given base URL (its real HTTP
/// client, with a key configured) and makes one call: `chat` when its
/// second argument is `false`, `chat_stream` when `true` (returning once
/// the stream starts). The contract checks that:
///
/// - the other host never receives a request (the key and the prompt stay
///   put);
/// - the call fails with [`Error::UnexpectedResponse`] carrying the
///   redirect status, and is not retryable.
///
/// Requires the `server` feature and a Tokio runtime.
///
/// # Panics
///
/// When a redirect is followed or the call does not fail as above.
pub async fn assert_redirect_not_followed<S, SFut>(send: S)
where
    S: Fn(String, bool) -> SFut,
    SFut: Future<Output = allama::Result<()>>,
{
    for status in REDIRECT_STATUSES {
        for streaming in [false, true] {
            let path = if streaming { "chat_stream" } else { "chat" };
            let target =
                TestHttpServer::spawn([MockHttpResponse::json(200, r#"{"stolen":true}"#)]).await;
            // `localhost` rather than `127.0.0.1`: another host as far as the
            // client can tell, so a followed redirect would be cross-host.
            let location = format!(
                "{}/v1/stolen",
                target.url().replace("127.0.0.1", "localhost")
            );
            // A retrying adapter may ask again; the origin keeps answering.
            let origin = TestHttpServer::spawn(
                (0..8).map(|_| MockHttpResponse::text(status, "").header("location", &location)),
            )
            .await;
            let result = send(origin.url().to_owned(), streaming).await;
            let leaked = target.recorded_requests().await;
            assert!(
                leaked.is_empty(),
                "redirect_not_followed: {path} followed a {status} redirect to another host, \
                 sending it {} request(s)",
                leaked.len()
            );
            assert!(
                !origin.recorded_requests().await.is_empty(),
                "redirect_not_followed: {path} sent nothing to the base URL it was given"
            );
            match result {
                Err(err @ Error::UnexpectedResponse { .. }) => {
                    assert_eq!(
                        err.status(),
                        Some(status),
                        "redirect_not_followed: {path}: the error keeps the redirect status"
                    );
                    assert!(
                        !err.is_retryable(),
                        "redirect_not_followed: {path}: a redirect is not retryable"
                    );
                }
                other => panic!(
                    "redirect_not_followed: {path}: a {status} redirect must fail with \
                     UnexpectedResponse, got {other:?}"
                ),
            }
            // The target must stay untaken: the response there is a trap.
            assert_eq!(target.remaining(), 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use allama_http::{HttpRequest, HttpTransport, ReqwestTransport, status_error};

    use super::*;

    /// An adapter's call over `transport`: the error mapping every adapter
    /// uses (`HttpResponse::map_error`).
    async fn call(transport: ReqwestTransport, url: String) -> allama::Result<()> {
        let response = transport
            .send(HttpRequest::post(format!("{url}/v1/chat"), "{}").header("x-api-key", "k"))
            .await
            .expect("the local server answers");
        if response.status == 200 {
            return Ok(());
        }
        Err(response
            .map_error(|status, _, _| status_error(status, format!("HTTP {status}")))
            .await)
    }

    #[tokio::test]
    async fn the_default_transport_passes() {
        assert_redirect_not_followed(|url, _| call(ReqwestTransport::new(), url)).await;
    }

    #[tokio::test]
    #[should_panic(expected = "followed a 301 redirect to another host")]
    async fn a_client_that_follows_redirects_fails() {
        assert_redirect_not_followed(|url, _| {
            call(ReqwestTransport::from_client(reqwest::Client::new()), url)
        })
        .await;
    }
}
