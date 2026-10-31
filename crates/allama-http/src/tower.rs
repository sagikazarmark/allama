//! [`TowerTransport`]: a `tower` service as an [`HttpTransport`] (feature
//! `tower`).

use std::error::Error as StdError;
use std::fmt;
use std::future::{Future, poll_fn};
use std::pin::{Pin, pin};
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::{Buf, Bytes};
use futures_core::Stream;
use futures_timer::Delay;
use futures_util::future::{Either, select};
use http::Uri;
use http_body_util::Full;
use tower_service::Service;

use crate::timeout::expired;
use crate::{
    HttpRequest, HttpResponse, HttpTransport, SendFuture, TransportError, TransportErrorKind,
};

/// A `tower` [`Service`] as an [`HttpTransport`]: put a stack of `tower`
/// layers (a concurrency limit, instrumentation) between a provider and the
/// HTTP client, or bring an HTTP client that speaks `tower`.
///
/// The service takes an `http::Request<`[`RequestBody`]`>` and returns an
/// `http::Response` with any [`http_body::Body`]; its error is anything
/// boxable. At the bottom of the stack:
///
/// - [`ReqwestTransport::into_service`] (with the `reqwest` feature): the
///   default transport's client, with its guarantees (no redirects, URL
///   redaction, failure classification), on `wasm32` too.
/// - `tower-reqwest`'s `HttpClientService`, behind a `map_request` layer
///   that converts the body (`|request| request.map(reqwest::Body::wrap)`).
///   Native only: `reqwest`'s futures are not `Send` on `wasm32`.
/// - A `fetch` service on Cloudflare Workers (`tower-worker-fetch`), which
///   takes any body.
///
/// ```
/// use allama_http::http::{Request, Response, StatusCode, Uri};
/// use allama_http::{Bytes, HttpRequest, HttpTransport, RequestBody, TowerTransport};
/// use tower::ServiceBuilder;
///
/// # async fn example() -> Result<(), allama_http::TransportError> {
/// let client = tower::service_fn(|_request: Request<RequestBody>| async {
///     Ok::<_, std::convert::Infallible>(Response::new(RequestBody::new(Bytes::from("ok"))))
/// });
/// let transport = TowerTransport::new(ServiceBuilder::new().concurrency_limit(8).service(client));
/// let response = transport.send(HttpRequest::get(Uri::from_static("https://api.example.com/v1/models"))).await?;
/// assert_eq!(response.status, StatusCode::OK);
/// assert_eq!(response.text().await?, "ok");
/// # Ok(()) }
/// ```
///
/// A provider builder takes it like any transport:
/// `.http_transport(TowerTransport::new(stack))`.
///
/// # What the adapter guarantees
///
/// - **The request timeout.** [`HttpRequest::timeout`], which the bundled
///   providers set on every non-streaming request
///   ([`DEFAULT_REQUEST_TIMEOUT`](crate::DEFAULT_REQUEST_TIMEOUT)), bounds
///   the whole exchange: waiting for the service to be ready, the response
///   headers and reading the body, as `reqwest`'s own request timeout does.
///   Expiry is a [`TransportErrorKind::Timeout`] error, and drops the
///   service's future or the body, cancelling the request. The timeout also
///   travels as a [`RequestTimeout`](crate::RequestTimeout) extension, for a
///   service that reads it. A timeout layer of your own still applies: the
///   shorter bound fires first. A `tower` timeout layer bounds only the wait
///   for the response headers, not the body that follows.
/// - **Credential headers are marked
///   [sensitive](http::HeaderValue::set_sensitive)**, so a layer that logs
///   the request's headers (`tower-http`'s tracing, at a verbose level)
///   prints them redacted. Headers a layer adds itself are not marked.
/// - **Failures are classified.** An error the service returns, from
///   `poll_ready`, `call` or the body, becomes a [`TransportError`]:
///   - a [`TransportError`] is passed through, so a stack built on
///     [`ReqwestTransport::into_service`] reports failures exactly as the
///     transport alone does. Found inside an error a layer wrapped it in
///     (`tower`'s `Buffer`), its kind and message are kept;
///   - a `reqwest` error (with the `reqwest` feature, from
///     `tower-reqwest`) is classified as [`ReqwestTransport`] does, its
///     URL redacted, and kept as the source;
///   - a timeout I/O error is a [`Timeout`](TransportErrorKind::Timeout);
///   - anything else is [`Other`](TransportErrorKind::Other) (while reading
///     the body, [`Body`](TransportErrorKind::Body)). Its text, the error
///     and its sources, becomes the error's upstream message with URLs and
///     the request's query and user info removed; the error itself is not
///     kept, because its chain can quote the URL.
///
///   A stack classifies its own failures by returning a [`TransportError`],
///   with a `map_err` layer. For `tower`'s timeout layer:
///
///   ```
///   # use allama_http::{TransportError, TransportErrorKind};
///   # type BoxError = Box<dyn std::error::Error + Send + Sync>;
///   let classify = |error: BoxError| -> BoxError {
///       if error.is::<tower::timeout::error::Elapsed>() {
///           Box::new(TransportError::new(TransportErrorKind::Timeout, "the stack's timeout elapsed").authored())
///       } else {
///           error
///       }
///   };
///   # let _ = classify;
///   ```
/// - **The server is recorded** for the call scope, so a `TracingProvider`
///   span records `server.address` and `server.port`.
///
/// # Redirects
///
/// The adapter does not follow redirects, but it cannot stop the service
/// from following them: a `3xx` that reaches it comes back as it is, and the
/// bundled providers report it ([`redirect_error`](crate::redirect_error)).
/// Configure the service not to follow them, or a key in `x-api-key` (and
/// the prompt, on a `307` or `308`) reaches the redirect's host:
/// [`ReqwestTransport::into_service`] never follows one; a `reqwest` client
/// you build for `tower-reqwest` needs
/// `.redirect(reqwest::redirect::Policy::none())` (`reqwest::Client::new()`
/// follows up to 10); `tower-worker-fetch` needs its `Redirect::Manual`
/// extension; and do not add `tower-http`'s `FollowRedirect`.
///
/// # Layers
///
/// The service is cloned for each request, then driven once (`poll_ready`,
/// then `call`), as `tower` services expect, so every layer must be `Clone`:
/// `tower`'s `RateLimit` is not, and sits behind a `Buffer`. A concurrency
/// limit releases its permit when the response headers arrive, so it bounds
/// requests being set up, not open streams. Do not add a retry layer:
/// `allama::wrappers::RetryingProvider` retries whole calls, with the
/// provider's retry signals and without replaying a stream the caller has
/// started to read.
///
/// The service, its future and the response body must be `Send`, and the
/// service `Sync`; for a type-erased stack, `tower`'s `BoxCloneSyncService`
/// is.
///
#[cfg_attr(
    feature = "reqwest",
    doc = "[`ReqwestTransport::into_service`]: crate::ReqwestTransport::into_service\n[`ReqwestTransport`]: crate::ReqwestTransport"
)]
#[cfg_attr(
    not(feature = "reqwest"),
    doc = "[`ReqwestTransport::into_service`]: crate\n[`ReqwestTransport`]: crate"
)]
#[derive(Clone)]
pub struct TowerTransport<S> {
    service: S,
}

impl<S> fmt::Debug for TowerTransport<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TowerTransport").finish_non_exhaustive()
    }
}

impl<S> TowerTransport<S> {
    /// A transport over `service`.
    #[must_use]
    pub fn new(service: S) -> Self {
        Self { service }
    }

    /// The service.
    #[must_use]
    pub fn get_ref(&self) -> &S {
        &self.service
    }

    /// The service, unwrapped.
    #[must_use]
    pub fn into_inner(self) -> S {
        self.service
    }
}

impl<S, R> HttpTransport for TowerTransport<S>
where
    S: Service<http::Request<RequestBody>, Response = http::Response<R>>
        + Clone
        + Send
        + Sync
        + 'static,
    S::Future: Send,
    S::Error: Into<BoxError>,
    R: http_body::Body + Send + 'static,
    R::Data: Buf,
    R::Error: Into<BoxError>,
{
    fn send(&self, request: HttpRequest) -> SendFuture<'_> {
        Box::pin(send(self.service.clone(), request))
    }
}

/// The request body a service behind [`TowerTransport`] receives: the
/// whole body, as an [`http_body::Body`] (`Bytes` is not one), cheap to
/// clone, so a layer can replay it.
pub type RequestBody = Full<Bytes>;

/// A boxed error, as `tower` services return them.
type BoxError = Box<dyn StdError + Send + Sync>;

/// Drive `service` once for `request`, under the request's timeout.
async fn send<S, R>(mut service: S, request: HttpRequest) -> Result<HttpResponse, TransportError>
where
    S: Service<http::Request<RequestBody>, Response = http::Response<R>>,
    S::Error: Into<BoxError>,
    R: http_body::Body + Send + 'static,
    R::Data: Buf,
    R::Error: Into<BoxError>,
{
    if request.uri.scheme().is_none() || request.uri.authority().is_none() {
        return Err(TransportError::new(
            TransportErrorKind::InvalidRequest,
            "the request URI is not an absolute URL",
        )
        .authored());
    }
    record_server(&request.uri);
    let secrets = UriSecrets::of(&request.uri);
    let mut deadline = request.timeout.map(Deadline::new);
    let response = {
        let exchange = pin!(async {
            poll_fn(|cx| service.poll_ready(cx))
                .await
                .map_err(|e| from_service(e.into(), Stage::Ready, &secrets))?;
            // Built after the wait, so the future holds no request body
            // across it: the body need not be `Send`.
            let request = http::Request::<Bytes>::from(request).map(Full::new);
            service
                .call(request)
                .await
                .map_err(|e| from_service(e.into(), Stage::Call, &secrets))
        });
        match &mut deadline {
            Some(deadline) => match select(exchange, &mut deadline.timer).await {
                Either::Left((response, _)) => response?,
                Either::Right(((), _)) => return Err(deadline.expired()),
            },
            None => exchange.await?,
        }
    };
    let (parts, body) = response.into_parts();
    let body = BodyStream {
        body: Some(Box::pin(body)),
        deadline,
        secrets,
    };
    Ok(HttpResponse::new(
        parts.status,
        parts.headers,
        Box::pin(body),
    ))
}

/// The bound on a whole exchange: one timer, from sending to the end of the
/// body.
struct Deadline {
    timer: Delay,
    limit: Duration,
}

impl Deadline {
    fn new(limit: Duration) -> Self {
        Self {
            timer: Delay::new(limit),
            limit,
        }
    }

    fn expired(&self) -> TransportError {
        expired("the exchange", self.limit)
    }
}

/// A response body as a [`Body`](crate::Body): its data frames, under what
/// is left of the request's timeout.
///
/// Trailers and empty data frames are skipped. After an error or the end,
/// the inner body is dropped and never polled again.
struct BodyStream<R> {
    body: Option<Pin<Box<R>>>,
    deadline: Option<Deadline>,
    secrets: UriSecrets,
}

/// How many frames without data [`BodyStream`] skips in one poll before
/// yielding, so a body that sends nothing else cannot hold the thread.
const MAX_SKIPPED_FRAMES: usize = 32;

impl<R> Stream for BodyStream<R>
where
    R: http_body::Body,
    R::Data: Buf,
    R::Error: Into<BoxError>,
{
    type Item = Result<Bytes, TransportError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let Some(body) = this.body.as_mut() else {
            return Poll::Ready(None);
        };
        // The deadline first, as `reqwest` does: a body that always has data
        // ready must not outlive it.
        if let Some(deadline) = &mut this.deadline
            && Pin::new(&mut deadline.timer).poll(cx).is_ready()
        {
            let error = deadline.expired();
            this.body = None;
            return Poll::Ready(Some(Err(error)));
        }
        for _ in 0..MAX_SKIPPED_FRAMES {
            match body.as_mut().poll_frame(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    this.body = None;
                    return Poll::Ready(None);
                }
                Poll::Ready(Some(Err(error))) => {
                    this.body = None;
                    let error = from_service(error.into(), Stage::Body, &this.secrets);
                    return Poll::Ready(Some(Err(error)));
                }
                Poll::Ready(Some(Ok(frame))) => {
                    // Trailers carry nothing a provider reads.
                    let Ok(mut data) = frame.into_data() else {
                        continue;
                    };
                    let len = data.remaining();
                    if len > 0 {
                        return Poll::Ready(Some(Ok(data.copy_to_bytes(len))));
                    }
                }
            }
        }
        cx.waker().wake_by_ref();
        Poll::Pending
    }

    /// Known to have ended only once the inner body says so: `http_body`'s
    /// own size hint counts bytes, not chunks.
    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.body {
            Some(body) if !body.is_end_stream() => (0, None),
            _ => (0, Some(0)),
        }
    }
}

/// Where a service's error came from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// `poll_ready`.
    Ready,
    /// `call` or its future.
    Call,
    /// Reading the response body.
    Body,
}

/// The [`TransportError`] for an error from the service or its body.
fn from_service(error: BoxError, stage: Stage, secrets: &UriSecrets) -> TransportError {
    let fallback = match stage {
        Stage::Ready | Stage::Call => TransportErrorKind::Other,
        Stage::Body => TransportErrorKind::Body,
    };
    let error = match error.downcast::<TransportError>() {
        Ok(error) => return *error,
        Err(error) => error,
    };
    #[cfg(feature = "reqwest")]
    let error = match error.downcast::<::reqwest::Error>() {
        Ok(error) => {
            let fallback = match stage {
                Stage::Ready | Stage::Call => TransportErrorKind::Connect,
                Stage::Body => TransportErrorKind::Body,
            };
            return crate::reqwest::with_cause(*error, fallback);
        }
        Err(error) => error,
    };
    if let Some(inner) = sources(&*error).find_map(|e| e.downcast_ref::<TransportError>()) {
        let message = if inner.authored {
            inner.message.clone()
        } else {
            secrets.redact(&inner.message)
        };
        return TransportError {
            kind: inner.kind,
            message,
            authored: inner.authored,
            source: None,
        };
    }
    let kind = crate::error_kind(&*error, fallback);
    let mut message = secrets.redact(&chain_text(&*error));
    if stage == Stage::Ready {
        message.insert_str(0, "the service is not ready: ");
    }
    TransportError::new(kind, message)
}

/// `error` and its sources, outermost first.
fn sources<'a>(
    error: &'a (dyn StdError + 'static),
) -> impl Iterator<Item = &'a (dyn StdError + 'static)> {
    std::iter::successors(Some(error), |&error| error.source())
}

/// The text of `error` and its sources, joined with `": "`, skipping a
/// source whose text the outer errors already show.
fn chain_text(error: &(dyn StdError + 'static)) -> String {
    let mut text = String::new();
    for error in sources(error) {
        let part = error.to_string();
        if part.is_empty() || text.contains(&part) {
            continue;
        }
        if !text.is_empty() {
            text.push_str(": ");
        }
        text.push_str(&part);
    }
    text
}

/// The parts of a request URI an error's text must not show: the query
/// (some APIs take the key there) and the user info.
struct UriSecrets {
    query: Option<String>,
    userinfo: Option<String>,
}

impl UriSecrets {
    fn of(uri: &Uri) -> Self {
        let query = uri.query().filter(|q| !q.is_empty()).map(str::to_owned);
        let userinfo = uri
            .authority()
            .and_then(|authority| authority.as_str().rsplit_once('@'))
            .map(|(userinfo, _)| userinfo.to_owned())
            .filter(|userinfo| !userinfo.is_empty());
        Self { query, userinfo }
    }

    /// `text` without this request's query and user info, wherever they
    /// appear, and without the query, fragment and user info of every URL
    /// in it.
    fn redact(&self, text: &str) -> String {
        let mut text = crate::config::redact_urls_in(text).into_owned();
        for secret in [&self.query, &self.userinfo].into_iter().flatten() {
            text = text.replace(secret.as_str(), "<redacted>");
        }
        text
    }
}

/// Tell the call scope which server the request goes to, as
/// `ReqwestTransport` does: the lowercase host
/// and the port, the scheme's default when the URI has none.
fn record_server(uri: &Uri) {
    use allama::wrappers::call_scope::{self, CallFact};

    let Some(host) = uri.host() else {
        return;
    };
    let port = uri.port_u16().or(match uri.scheme_str() {
        Some("http" | "ws") => Some(80),
        Some("https" | "wss") => Some(443),
        _ => None,
    });
    call_scope::record(CallFact::server(host.to_ascii_lowercase(), port));
}

#[cfg(test)]
mod tests;
