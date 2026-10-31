use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use futures_core::Stream;
use futures_timer::Delay;
use futures_util::future::{Either, select};

use crate::{Body, HttpRequest, HttpResponse, HttpTransport, TransportError, TransportErrorKind};

/// Time bounds for a streaming request, enforced above the transport.
///
/// A streaming response has no total deadline (a long answer is not a
/// failure), but a dead endpoint or a stalled connection must not hang a
/// caller forever. Two bounds catch that:
///
/// - `setup`: from sending the request until the response status and
///   headers arrive.
/// - `idle`: the longest wait for a body chunk, the first one included.
///   Only waiting on the transport counts: for each chunk, the clock starts
///   when the reader polls and the chunk is not ready yet, so the time
///   before the reader's first poll or between its polls does not count.
///
/// [`send`](Self::send) enforces both on any [`HttpTransport`]; provider
/// crates call it for their streaming requests (non-streaming requests carry
/// a whole-exchange [`HttpRequest::timeout`] instead). Expiry fails with a
/// [`TransportErrorKind::Timeout`] error, which providers surface as
/// [`allama::Error::Timeout`], retryable and fallback-eligible.
///
/// The defaults ([`DEFAULT_SETUP`](Self::DEFAULT_SETUP),
/// [`DEFAULT_IDLE`](Self::DEFAULT_IDLE)) are deliberately generous: they
/// detect dead connections, not slow models. Reasoning models can stay
/// silent for minutes before their first token, and an idle bound that cuts
/// such a stream off bills the reasoning and returns nothing. Tighter,
/// workload-specific bounds belong in `allama::wrappers::TimeoutProvider`,
/// which wraps the decoded stream and fires first when its limits are
/// shorter. It can also bound a stream's total duration
/// (`TimeoutPolicy::stream_total`), which this transport layer leaves to
/// the application.
///
/// Timers come from `futures-timer` (the same timers `TimeoutProvider`
/// uses), so the bounds work on any async runtime and, through the browser
/// or worker `setTimeout`, on `wasm32-unknown-unknown`.
///
/// ```
/// use std::time::Duration;
/// use allama_http::StreamTimeouts;
///
/// let tight = StreamTimeouts::default().idle(Some(Duration::from_secs(30)));
/// assert_eq!(tight.setup, Some(StreamTimeouts::DEFAULT_SETUP));
///
/// // A local server that may load a model first: no bounds at all.
/// let local = StreamTimeouts::unbounded();
/// assert_eq!((local.setup, local.idle), (None, None));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct StreamTimeouts {
    /// Bound on the wait for the response status and headers.
    pub setup: Option<Duration>,
    /// Bound on the wait for each body chunk, the first one included,
    /// counted from the reader's poll that finds the chunk not ready.
    pub idle: Option<Duration>,
}

impl Default for StreamTimeouts {
    /// [`DEFAULT_SETUP`](Self::DEFAULT_SETUP) and
    /// [`DEFAULT_IDLE`](Self::DEFAULT_IDLE).
    fn default() -> Self {
        Self {
            setup: Some(Self::DEFAULT_SETUP),
            idle: Some(Self::DEFAULT_IDLE),
        }
    }
}

impl StreamTimeouts {
    /// Default [`setup`](Self::setup) bound: 120 s, the same budget the
    /// provider crates give a whole non-streaming call. A streaming endpoint
    /// normally answers with headers right away and keeps the model's
    /// latency for the body.
    pub const DEFAULT_SETUP: Duration = Duration::from_secs(120);

    /// Default [`idle`](Self::idle) bound: 300 s. Longer than a non-streaming
    /// call's whole default budget on purpose: streaming is how callers reach
    /// reasoning models that think for minutes before their first token, and
    /// not every endpoint sends keep-alive events meanwhile.
    pub const DEFAULT_IDLE: Duration = Duration::from_secs(300);

    /// No bounds: the stream may wait forever (for example for a local
    /// server that loads the model before answering).
    #[must_use]
    pub const fn unbounded() -> Self {
        Self {
            setup: None,
            idle: None,
        }
    }

    /// Set (or clear) the setup bound.
    #[must_use]
    pub const fn setup(mut self, timeout: Option<Duration>) -> Self {
        self.setup = timeout;
        self
    }

    /// Set (or clear) the idle bound.
    #[must_use]
    pub const fn idle(mut self, timeout: Option<Duration>) -> Self {
        self.idle = timeout;
        self
    }

    /// Send `request` through `transport` with these bounds: the response
    /// headers must arrive within `setup`, and its body then fails when no
    /// chunk arrives within `idle`.
    ///
    /// Expiry drops the transport's future or body, cancelling the request.
    ///
    /// # Errors
    ///
    /// The transport's errors, or a [`TransportErrorKind::Timeout`] error
    /// when the setup bound expires. An idle expiry is the body's last item.
    pub async fn send<T>(
        self,
        transport: &T,
        request: HttpRequest,
    ) -> Result<HttpResponse, TransportError>
    where
        T: HttpTransport + ?Sized,
    {
        let sent = transport.send(request);
        let response = match self.setup {
            Some(limit) => match select(sent, Delay::new(limit)).await {
                Either::Left((result, _)) => result?,
                Either::Right(((), _)) => return Err(expired("stream setup", limit)),
            },
            None => sent.await?,
        };
        let Some(idle) = self.idle else {
            return Ok(response);
        };
        let status = response.status;
        let headers = response.headers.clone();
        let body = IdleTimeoutBody::new(response.into_body(), idle);
        Ok(HttpResponse::new(status, headers, Box::pin(body)))
    }
}

/// A body that fails when no chunk arrives within the idle bound, then ends.
///
/// For each chunk, the first one included, the clock starts when the reader
/// polls and the chunk is not ready yet: time the reader spends before its
/// first poll or between polls does not count. Bodies such as hyper's
/// HTTP/1 bodies read from the socket only while polled, so a timer left
/// running across a slow reader would expire on the first poll after it.
struct IdleTimeoutBody {
    inner: Option<Body>,
    idle: Duration,
    timer: Delay,
    /// The timer is running: set on the first `Pending` of each wait (the
    /// first one included), cleared when a chunk is handed out.
    armed: bool,
}

impl IdleTimeoutBody {
    fn new(inner: Body, idle: Duration) -> Self {
        Self {
            inner: Some(inner),
            idle,
            timer: Delay::new(idle),
            // Disarmed until the reader first waits: an adapter may do
            // other work between the headers and its first read.
            armed: false,
        }
    }
}

impl Stream for IdleTimeoutBody {
    type Item = Result<Bytes, TransportError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        let Some(inner) = this.inner.as_mut() else {
            return Poll::Ready(None);
        };
        match inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(item)) => {
                this.armed = false;
                Poll::Ready(Some(item))
            }
            Poll::Ready(None) => {
                this.inner = None;
                Poll::Ready(None)
            }
            Poll::Pending => {
                if !this.armed {
                    this.timer.reset(this.idle);
                    this.armed = true;
                }
                if Pin::new(&mut this.timer).poll(cx).is_pending() {
                    return Poll::Pending;
                }
                this.inner = None;
                Poll::Ready(Some(Err(expired(
                    "wait for the next body chunk",
                    this.idle,
                ))))
            }
        }
    }
}

/// The error for an expired bound, phrased like `TimeoutProvider`'s:
/// `"<what> exceeded <limit>"`.
pub(crate) fn expired(what: &str, limit: Duration) -> TransportError {
    TransportError::new(
        TransportErrorKind::Timeout,
        format!("{what} exceeded {limit:?}"),
    )
    .authored()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use futures_util::StreamExt;

    use super::*;
    use http::{HeaderMap, StatusCode, Uri};

    use crate::{ReplayResponse, ReplayTransport, SendFuture};

    const SHORT: Duration = Duration::from_millis(20);

    /// Headers after `headers_after`; then each chunk after `gap`, or a body
    /// that never produces anything when `gap` is `None`.
    struct Slow {
        headers_after: Duration,
        gap: Option<Duration>,
    }

    impl HttpTransport for Slow {
        fn send(&self, _request: HttpRequest) -> SendFuture<'_> {
            let gap = self.gap;
            let headers_after = self.headers_after;
            Box::pin(async move {
                Delay::new(headers_after).await;
                let body: Body = match gap {
                    Some(gap) => Box::pin(futures_util::stream::iter(["a", "b", "c"]).then(
                        move |chunk| async move {
                            Delay::new(gap).await;
                            Ok(Bytes::from_static(chunk.as_bytes()))
                        },
                    )),
                    None => Box::pin(futures_util::stream::pending()),
                };
                Ok(HttpResponse::new(StatusCode::OK, HeaderMap::new(), body))
            })
        }
    }

    fn request() -> HttpRequest {
        HttpRequest::post(Uri::from_static("https://x"), "{}")
    }

    #[tokio::test]
    async fn slow_headers_fail_the_setup_bound() {
        let transport = Slow {
            headers_after: Duration::from_secs(5),
            gap: Some(Duration::ZERO),
        };
        let err = StreamTimeouts::unbounded()
            .setup(Some(SHORT))
            .send(&transport, request())
            .await
            .unwrap_err();
        assert_eq!(err.kind(), TransportErrorKind::Timeout);
        let err = err.into_error();
        assert!(matches!(err, allama::Error::Timeout { .. }));
        assert_eq!(err.message(), "stream setup exceeded 20ms");
        // Written by the bound itself: the redacted log keeps it.
        assert_eq!(
            err.as_log().to_string(),
            r#"kind=timeout message="stream setup exceeded 20ms""#
        );
    }

    #[tokio::test]
    async fn a_stalled_body_fails_the_idle_bound_then_ends() {
        let transport = Slow {
            headers_after: Duration::ZERO,
            gap: None,
        };
        let response = StreamTimeouts::default()
            .idle(Some(SHORT))
            .send(&transport, request())
            .await
            .unwrap();
        assert_eq!(response.status, StatusCode::OK);
        let mut body = response.into_body();
        let err = body.next().await.unwrap().unwrap_err();
        assert_eq!(err.kind(), TransportErrorKind::Timeout);
        assert_eq!(
            err.to_string(),
            "request timed out: wait for the next body chunk exceeded 20ms"
        );
        assert!(body.next().await.is_none());
    }

    #[tokio::test]
    async fn steady_chunks_reset_the_idle_bound() {
        let transport = Slow {
            headers_after: Duration::ZERO,
            gap: Some(Duration::from_millis(30)),
        };
        // Three 30 ms gaps: 90 ms in total, but never 60 ms of silence.
        let response = StreamTimeouts::unbounded()
            .idle(Some(Duration::from_millis(60)))
            .send(&transport, request())
            .await
            .unwrap();
        assert_eq!(response.text().await.unwrap(), "abc");
    }

    /// The idle bound measures the transport, not the consumer: time spent
    /// between polls does not count, even though the body is read only
    /// while polled.
    #[tokio::test]
    async fn slow_consumers_do_not_trip_the_idle_bound() {
        let transport = Slow {
            headers_after: Duration::ZERO,
            gap: Some(SHORT),
        };
        let response = StreamTimeouts::unbounded()
            .idle(Some(Duration::from_millis(100)))
            .send(&transport, request())
            .await
            .unwrap();
        let mut body = response.into_body();
        let mut chunks = Vec::new();
        while let Some(chunk) = body.next().await {
            chunks.push(chunk.unwrap());
            Delay::new(Duration::from_millis(150)).await;
        }
        assert_eq!(chunks.concat(), b"abc");
    }

    /// The idle bound starts at the first wait, not at the headers: an
    /// adapter that reads the body late is not timed out for it.
    #[tokio::test]
    async fn a_late_first_poll_does_not_trip_the_idle_bound() {
        let transport = Slow {
            headers_after: Duration::ZERO,
            gap: Some(Duration::from_millis(10)),
        };
        let response = StreamTimeouts::unbounded()
            .idle(Some(Duration::from_millis(50)))
            .send(&transport, request())
            .await
            .unwrap();
        // The first chunk is not ready when first polled: a timer started at
        // the headers would have expired by then.
        Delay::new(Duration::from_millis(100)).await;
        assert_eq!(response.text().await.unwrap(), "abc");
    }

    /// A first chunk that never comes still trips the bound, measured from
    /// the first poll.
    #[tokio::test]
    async fn a_stalled_first_chunk_fails_the_idle_bound() {
        let body: Body = Box::pin(futures_util::stream::pending());
        let mut body = IdleTimeoutBody::new(body, Duration::from_millis(50));
        Delay::new(Duration::from_millis(100)).await;
        let started = std::time::Instant::now();
        let err = body.next().await.unwrap().unwrap_err();
        assert_eq!(err.kind(), TransportErrorKind::Timeout);
        assert!(
            started.elapsed() >= Duration::from_millis(50),
            "{:?}",
            started.elapsed()
        );
        assert!(body.next().await.is_none());
    }

    /// A body that stalls after a chunk still trips the bound, measured from
    /// the poll that found nothing ready.
    #[tokio::test]
    async fn a_body_stalled_after_a_chunk_fails_the_idle_bound() {
        let body: Body = Box::pin(
            futures_util::stream::iter([Ok(Bytes::from_static(b"a"))])
                .chain(futures_util::stream::pending()),
        );
        let mut body = IdleTimeoutBody::new(body, SHORT);
        assert_eq!(body.next().await.unwrap().unwrap(), "a");
        Delay::new(SHORT * 3).await;
        let started = std::time::Instant::now();
        let err = body.next().await.unwrap().unwrap_err();
        assert_eq!(err.kind(), TransportErrorKind::Timeout);
        assert!(started.elapsed() >= SHORT, "{:?}", started.elapsed());
        assert!(body.next().await.is_none());
    }

    #[tokio::test]
    async fn unbounded_passes_responses_and_errors_through() {
        let transport = ReplayTransport::new([ReplayResponse::new(StatusCode::CREATED, "ok")]);
        let response = StreamTimeouts::unbounded()
            .send(&transport, request())
            .await
            .unwrap();
        assert_eq!(response.status, StatusCode::CREATED);
        assert_eq!(response.text().await.unwrap(), "ok");

        // The queue is empty now: the transport's own error comes back.
        let err = StreamTimeouts::default()
            .send(&transport, request())
            .await
            .unwrap_err();
        assert_eq!(err.kind(), TransportErrorKind::Connect);
    }
}

/// Run on `wasm32` with `wasm-bindgen-test`: the timers go through the
/// host's `setTimeout`.
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests {
    use futures_util::StreamExt;
    use wasm_bindgen_test::wasm_bindgen_test;

    use super::*;
    use http::{HeaderMap, StatusCode, Uri};

    use crate::SendFuture;

    struct Stalled;

    impl HttpTransport for Stalled {
        fn send(&self, _request: HttpRequest) -> SendFuture<'_> {
            Box::pin(async {
                Ok(HttpResponse::new(
                    StatusCode::OK,
                    HeaderMap::new(),
                    Box::pin(futures_util::stream::pending()),
                ))
            })
        }
    }

    #[wasm_bindgen_test]
    async fn a_stalled_body_fails_the_idle_bound() {
        let response = StreamTimeouts::default()
            .idle(Some(Duration::from_millis(20)))
            .send(&Stalled, HttpRequest::get(Uri::from_static("https://x")))
            .await
            .unwrap();
        let err = response.into_body().next().await.unwrap().unwrap_err();
        assert_eq!(err.kind(), TransportErrorKind::Timeout);
    }
}
