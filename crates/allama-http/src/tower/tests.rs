//! The tests of `TowerTransport`, over hand-written services and bodies.

// The `wasm32` tests use part of the scaffolding.
#![cfg_attr(target_arch = "wasm32", allow(dead_code))]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use http::HeaderMap;
use http_body::Frame;

use super::*;

type Reply = Result<http::Response<TestBody>, BoxError>;

/// One piece of a [`TestBody`].
enum Item {
    Data(&'static str),
    Trailers,
    Error(BoxError),
    /// Never produce anything again.
    Stall,
}

/// A response body that hands out its items in order, and records when it
/// is dropped.
struct TestBody {
    items: VecDeque<Item>,
    /// Report the end as soon as no item is left (`is_end_stream`).
    reports_end: bool,
    dropped: Arc<AtomicBool>,
}

impl TestBody {
    fn new(items: impl IntoIterator<Item = Item>) -> Self {
        Self {
            items: items.into_iter().collect(),
            reports_end: true,
            dropped: Arc::default(),
        }
    }

    fn data(text: &'static str) -> Self {
        Self::new([Item::Data(text)])
    }
}

impl Drop for TestBody {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

impl http_body::Body for TestBody {
    type Data = Bytes;
    type Error = BoxError;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, BoxError>>> {
        match self.items.pop_front() {
            None => Poll::Ready(None),
            Some(Item::Data(text)) => Poll::Ready(Some(Ok(Frame::data(Bytes::from(text))))),
            Some(Item::Trailers) => Poll::Ready(Some(Ok(Frame::trailers(HeaderMap::new())))),
            Some(Item::Error(error)) => Poll::Ready(Some(Err(error))),
            Some(Item::Stall) => {
                self.items.push_front(Item::Stall);
                Poll::Pending
            }
        }
    }

    fn is_end_stream(&self) -> bool {
        self.reports_end && self.items.is_empty()
    }
}

/// A service that answers each request with the next reply from `replies`
/// and records the requests it was called with.
#[derive(Clone)]
struct Scripted {
    replies: Arc<Mutex<VecDeque<Reply>>>,
    requests: Arc<Mutex<Vec<http::Request<RequestBody>>>>,
}

impl Scripted {
    fn new(replies: impl IntoIterator<Item = Reply>) -> Self {
        Self {
            replies: Arc::new(Mutex::new(replies.into_iter().collect())),
            requests: Arc::default(),
        }
    }

    fn ok(body: TestBody) -> Self {
        Self::new([Ok(http::Response::new(body))])
    }

    fn failing(error: impl Into<BoxError>) -> Self {
        Self::new([Err(error.into())])
    }

    fn requests(&self) -> std::sync::MutexGuard<'_, Vec<http::Request<RequestBody>>> {
        self.requests.lock().unwrap()
    }
}

impl Service<http::Request<RequestBody>> for Scripted {
    type Response = http::Response<TestBody>;
    type Error = BoxError;
    type Future = std::future::Ready<Reply>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), BoxError>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: http::Request<RequestBody>) -> Self::Future {
        self.requests.lock().unwrap().push(request);
        let reply = self.replies.lock().unwrap().pop_front();
        std::future::ready(reply.expect("a reply for every request"))
    }
}

/// A service whose `call` never completes, and records when its future is
/// dropped.
#[derive(Clone, Default)]
struct Hangs {
    dropped: Arc<AtomicBool>,
}

/// Sets its flag when dropped.
struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

impl Service<http::Request<RequestBody>> for Hangs {
    type Response = http::Response<TestBody>;
    type Error = BoxError;
    type Future = Pin<Box<dyn Future<Output = Reply> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), BoxError>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _request: http::Request<RequestBody>) -> Self::Future {
        let flag = DropFlag(self.dropped.clone());
        Box::pin(async move {
            let _flag = flag;
            std::future::pending().await
        })
    }
}

fn get() -> HttpRequest {
    HttpRequest::get(Uri::from_static("https://api.example.com/v1/models"))
}

/// What a service failed with, sent through a `TowerTransport`.
async fn failure(service: Scripted) -> TransportError {
    TowerTransport::new(service).send(get()).await.unwrap_err()
}

/// Every chunk of a body, or the first error.
async fn chunks(response: HttpResponse) -> Result<Vec<Bytes>, TransportError> {
    use futures_util::TryStreamExt;
    response.into_body().try_collect().await
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::sync::atomic::AtomicUsize;
    use std::time::{Duration, Instant};

    use futures_util::{FutureExt, StreamExt};
    use http::{HeaderValue, StatusCode};
    use http_body_util::BodyExt;

    use super::*;
    use crate::{RequestTimeout, StreamTimeouts};

    const SHORT: Duration = Duration::from_millis(50);

    #[tokio::test]
    async fn the_request_reaches_the_service_with_its_timeout_and_credentials_marked() {
        let service = Scripted::ok(TestBody::data("ok"));
        let request = HttpRequest::post(Uri::from_static("https://api.example.com/v1/chat"), "{}")
            .header("authorization", HeaderValue::from_static("Bearer SECRET1"))
            .header("x-gateway-token", HeaderValue::from_static("SECRET2"))
            .header("x-request-id", HeaderValue::from_static("req_1"))
            .timeout(Some(Duration::from_secs(5)));
        TowerTransport::new(service.clone())
            .send(request)
            .await
            .unwrap();

        let sent = service.requests().pop().unwrap();
        assert_eq!(sent.method(), http::Method::POST);
        assert_eq!(sent.uri(), "https://api.example.com/v1/chat");
        assert_eq!(
            sent.extensions().get(),
            Some(&RequestTimeout(Duration::from_secs(5)))
        );
        let headers = sent.headers();
        assert!(headers["authorization"].is_sensitive());
        assert!(headers["x-gateway-token"].is_sensitive());
        assert!(!headers["x-request-id"].is_sensitive());
        // What a layer that logs the request would print.
        let debug = format!("{sent:?}");
        assert!(!debug.contains("SECRET"), "{debug}");
        assert!(debug.contains("req_1"), "{debug}");
        let body = sent.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(body, "{}");
    }

    #[tokio::test]
    async fn the_response_streams_its_data_without_trailers_or_empty_frames() {
        let mut response = http::Response::new(TestBody::new([
            Item::Data("Hel"),
            Item::Data(""),
            Item::Trailers,
            Item::Data("lo"),
            Item::Trailers,
        ]));
        *response.status_mut() = StatusCode::CREATED;
        response
            .headers_mut()
            .insert("x-request-id", HeaderValue::from_static("req_1"));
        let response = TowerTransport::new(Scripted::new([Ok(response)]))
            .send(get())
            .await
            .unwrap();
        assert_eq!(response.status, StatusCode::CREATED);
        assert_eq!(response.headers["x-request-id"], "req_1");
        assert_eq!(chunks(response).await.unwrap(), ["Hel", "lo"]);
    }

    /// A body of trailers alone is an empty body.
    #[tokio::test]
    async fn a_body_of_trailers_alone_is_empty() {
        let service = Scripted::ok(TestBody::new([Item::Trailers]));
        let response = TowerTransport::new(service).send(get()).await.unwrap();
        assert!(chunks(response).await.unwrap().is_empty());
    }

    /// A body that sends only empty frames does not hold the thread, and
    /// its data still arrives.
    #[tokio::test]
    async fn many_empty_frames_still_end() {
        let items = std::iter::repeat_with(|| Item::Data(""))
            .take(10 * MAX_SKIPPED_FRAMES)
            .chain([Item::Data("x")]);
        let service = Scripted::ok(TestBody::new(items));
        let response = TowerTransport::new(service).send(get()).await.unwrap();
        assert_eq!(chunks(response).await.unwrap(), ["x"]);
    }

    /// One poll of a body that sends only empty frames reads at most
    /// `MAX_SKIPPED_FRAMES` of them, then yields, asking to be polled again.
    #[test]
    fn a_poll_skips_a_bounded_number_of_empty_frames() {
        struct EmptyForever(Arc<AtomicUsize>);

        impl http_body::Body for EmptyForever {
            type Data = Bytes;
            type Error = BoxError;

            fn poll_frame(
                self: Pin<&mut Self>,
                _cx: &mut Context<'_>,
            ) -> Poll<Option<Result<Frame<Bytes>, BoxError>>> {
                let polls = self.0.fetch_add(1, Ordering::SeqCst) + 1;
                assert!(
                    polls <= MAX_SKIPPED_FRAMES,
                    "polled {polls} frames in one poll"
                );
                Poll::Ready(Some(Ok(Frame::data(Bytes::new()))))
            }
        }

        struct CountWakes(AtomicUsize);

        impl std::task::Wake for CountWakes {
            fn wake(self: Arc<Self>) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let frames = Arc::new(AtomicUsize::new(0));
        let mut stream = BodyStream {
            body: Some(Box::pin(EmptyForever(frames.clone()))),
            deadline: None,
            secrets: UriSecrets::of(&Uri::from_static("https://x")),
        };
        let wakes = Arc::new(CountWakes(AtomicUsize::new(0)));
        let waker = std::task::Waker::from(wakes.clone());
        let mut cx = Context::from_waker(&waker);
        assert!(Pin::new(&mut stream).poll_next(&mut cx).is_pending());
        assert_eq!(frames.load(Ordering::SeqCst), MAX_SKIPPED_FRAMES);
        assert_eq!(wakes.0.load(Ordering::SeqCst), 1);
    }

    /// Data in a `Buf` of several pieces arrives whole.
    #[tokio::test]
    async fn non_contiguous_data_is_copied_whole() {
        struct Chained(Option<bytes::buf::Chain<Bytes, Bytes>>);

        impl http_body::Body for Chained {
            type Data = bytes::buf::Chain<Bytes, Bytes>;
            type Error = BoxError;

            fn poll_frame(
                mut self: Pin<&mut Self>,
                _cx: &mut Context<'_>,
            ) -> Poll<Option<Result<Frame<Self::Data>, BoxError>>> {
                Poll::Ready(self.0.take().map(|data| Ok(Frame::data(data))))
            }
        }

        let service = ::tower::service_fn(|_request: http::Request<RequestBody>| async {
            let data = Bytes::from("Hel").chain(Bytes::from("lo"));
            Ok::<_, BoxError>(http::Response::new(Chained(Some(data))))
        });
        let response = TowerTransport::new(service).send(get()).await.unwrap();
        assert_eq!(chunks(response).await.unwrap(), ["Hello"]);
    }

    /// The end of the body is known once the inner body reports it, so
    /// `bytes_limited` returns a single chunk without a copy.
    #[tokio::test]
    async fn the_end_of_the_body_is_reported() {
        let service = Scripted::ok(TestBody::data("whole"));
        let response = TowerTransport::new(service).send(get()).await.unwrap();
        let mut body = response.into_body();
        assert_eq!(body.size_hint(), (0, None));
        assert_eq!(body.next().await.unwrap().unwrap(), "whole");
        assert_eq!(body.size_hint(), (0, Some(0)));

        let mut silent = TestBody::data("whole");
        silent.reports_end = false;
        let response = TowerTransport::new(Scripted::ok(silent))
            .send(get())
            .await
            .unwrap();
        let mut body = response.into_body();
        body.next().await.unwrap().unwrap();
        assert_eq!(body.size_hint(), (0, None));
        assert!(body.next().await.is_none());
        assert_eq!(body.size_hint(), (0, Some(0)));
    }

    #[tokio::test]
    async fn a_uri_that_is_not_absolute_is_refused_before_the_service() {
        let service = Scripted::new([]);
        let err = TowerTransport::new(service.clone())
            .send(HttpRequest::get(Uri::from_static("/v1/models")))
            .await
            .unwrap_err();
        assert_eq!(err.kind(), TransportErrorKind::InvalidRequest);
        assert!(service.requests().is_empty());
    }

    /// The service is asked to be ready, and waited for, before it is
    /// called.
    #[tokio::test]
    async fn the_service_is_called_once_ready() {
        #[derive(Clone)]
        struct Slow {
            polls: Arc<AtomicUsize>,
            inner: Scripted,
        }

        impl Service<http::Request<RequestBody>> for Slow {
            type Response = http::Response<TestBody>;
            type Error = BoxError;
            type Future = std::future::Ready<Reply>;

            fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), BoxError>> {
                if self.polls.fetch_add(1, Ordering::SeqCst) < 2 {
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
                Poll::Ready(Ok(()))
            }

            fn call(&mut self, request: http::Request<RequestBody>) -> Self::Future {
                assert!(
                    self.polls.load(Ordering::SeqCst) >= 3,
                    "called before ready"
                );
                self.inner.call(request)
            }
        }

        let slow = Slow {
            polls: Arc::default(),
            inner: Scripted::ok(TestBody::data("ok")),
        };
        let response = TowerTransport::new(slow.clone()).send(get()).await.unwrap();
        assert_eq!(response.text().await.unwrap(), "ok");
        assert_eq!(slow.polls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn a_readiness_failure_says_so() {
        #[derive(Clone)]
        struct Broken;

        impl Service<http::Request<RequestBody>> for Broken {
            type Response = http::Response<TestBody>;
            type Error = BoxError;
            type Future = std::future::Ready<Reply>;

            fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), BoxError>> {
                Poll::Ready(Err("the pool is closed".into()))
            }

            fn call(&mut self, _request: http::Request<RequestBody>) -> Self::Future {
                unreachable!("never ready")
            }
        }

        let err = TowerTransport::new(Broken).send(get()).await.unwrap_err();
        assert_eq!(err.kind(), TransportErrorKind::Other);
        assert_eq!(
            err.to_string(),
            "the service is not ready: the pool is closed"
        );
    }

    /// A `TransportError` from the service is the error itself: its kind,
    /// its message, and whether the message is authored.
    #[tokio::test]
    async fn a_transport_error_is_passed_through() {
        let err = failure(Scripted::failing(
            TransportError::new(TransportErrorKind::Timeout, "the stack gave up").authored(),
        ))
        .await;
        assert_eq!(err.kind(), TransportErrorKind::Timeout);
        assert_eq!(
            err.into_error().as_log().to_string(),
            r#"kind=timeout message="the stack gave up""#
        );
    }

    /// `tower`'s `Buffer` wraps a failed `poll_ready` in an error of its
    /// own: the `TransportError` inside keeps its kind and message.
    #[tokio::test]
    async fn a_transport_error_wrapped_by_a_layer_keeps_its_kind() {
        #[derive(Clone)]
        struct NotReady;

        impl Service<http::Request<RequestBody>> for NotReady {
            type Response = http::Response<TestBody>;
            type Error = TransportError;
            type Future = std::future::Ready<Result<http::Response<TestBody>, TransportError>>;

            fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), TransportError>> {
                Poll::Ready(Err(TransportError::new(
                    TransportErrorKind::Connect,
                    "refused https://api.example.com/v1/models?key=SECRET",
                )))
            }

            fn call(&mut self, _request: http::Request<RequestBody>) -> Self::Future {
                unreachable!("never ready")
            }
        }

        let stack = ::tower::ServiceBuilder::new().buffer(1).service(NotReady);
        let err = TowerTransport::new(stack).send(get()).await.unwrap_err();
        assert_eq!(err.kind(), TransportErrorKind::Connect);
        assert_eq!(err.to_string(), "refused https://api.example.com/v1/models");

        // And from `call`, which `Buffer` passes on as it is.
        let stack = ::tower::ServiceBuilder::new()
            .buffer(1)
            .service(Scripted::failing(TransportError::new(
                TransportErrorKind::Body,
                "reset",
            )));
        let err = TowerTransport::new(stack).send(get()).await.unwrap_err();
        assert_eq!(err.kind(), TransportErrorKind::Body);
    }

    #[tokio::test]
    async fn a_timed_out_io_error_is_a_timeout() {
        let io = std::io::Error::new(std::io::ErrorKind::TimedOut, "read timed out");
        let err = failure(Scripted::failing(io)).await;
        assert_eq!(err.kind(), TransportErrorKind::Timeout);
    }

    /// Anything else is `Other`, with the text of the error and its sources
    /// as the upstream message, and no secret from the request's URI.
    #[tokio::test]
    async fn other_errors_keep_their_text_without_the_uri_secrets() {
        #[derive(Debug)]
        struct Outer(std::io::Error);

        impl fmt::Display for Outer {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(
                    "error sending request for url (https://user:pw@api.example.com/v1/models?key=SECRET1) \
                     with query `key=SECRET1`",
                )
            }
        }

        impl StdError for Outer {
            fn source(&self) -> Option<&(dyn StdError + 'static)> {
                Some(&self.0)
            }
        }

        let service = Scripted::failing(Outer(std::io::Error::other("connection reset")));
        let request = HttpRequest::get(Uri::from_static(
            "https://user:pw@api.example.com/v1/models?key=SECRET1",
        ));
        let err = TowerTransport::new(service)
            .send(request)
            .await
            .unwrap_err();
        assert_eq!(err.kind(), TransportErrorKind::Other);
        let text = err.to_string();
        assert!(!text.contains("SECRET"), "{text}");
        assert!(!text.contains("pw"), "{text}");
        assert!(text.contains("api.example.com/v1/models"), "{text}");
        assert!(text.ends_with(": connection reset"), "{text}");

        let err = err.into_error();
        assert!(matches!(err, allama::Error::Transport { .. }), "{err:?}");
        // The redacted log shows only the authored message.
        assert_eq!(
            err.as_log().to_string(),
            r#"kind=transport message="the transport failed""#
        );
        assert!(std::error::Error::source(&err).is_none());
    }

    /// A `tower` timeout layer's error is not recognized without a
    /// `map_err` layer that says what it is.
    #[tokio::test]
    async fn a_tower_timeout_is_classified_through_map_err() {
        let stack = || {
            ::tower::ServiceBuilder::new()
                .timeout(SHORT)
                .service(Hangs::default())
        };
        let err = TowerTransport::new(stack()).send(get()).await.unwrap_err();
        assert_eq!(err.kind(), TransportErrorKind::Other);
        assert_eq!(err.to_string(), "request timed out");

        let classified = ::tower::ServiceBuilder::new()
            .map_err(|error: BoxError| -> BoxError {
                if error.is::<::tower::timeout::error::Elapsed>() {
                    Box::new(
                        TransportError::new(TransportErrorKind::Timeout, "the stack timed out")
                            .authored(),
                    )
                } else {
                    error
                }
            })
            .service(stack());
        let err = TowerTransport::new(classified)
            .send(get())
            .await
            .unwrap_err();
        assert_eq!(err.kind(), TransportErrorKind::Timeout);
    }

    /// A `reqwest` error, as `tower-reqwest` returns it, is classified as
    /// `ReqwestTransport` does and kept as the source, its URL redacted.
    #[cfg(feature = "reqwest")]
    #[tokio::test]
    async fn a_reqwest_error_is_classified_and_kept() {
        let request = || {
            HttpRequest::get(Uri::from_static(
                "http://user:pw@127.0.0.1:1/v1beta/models?key=SECRET",
            ))
        };
        let err = TowerTransport::new(reqwest_service())
            .send(request())
            .await
            .unwrap_err();
        assert_eq!(err.kind(), TransportErrorKind::Connect);
        let err = err.into_error();
        for text in [err.to_string(), format!("{err:?}")] {
            assert!(!text.contains("SECRET"), "{text}");
            assert!(!text.contains("pw"), "{text}");
        }
        let source = std::error::Error::source(&err).expect("the cause is attached");
        assert!(source.downcast_ref::<::reqwest::Error>().is_some());

        // Wrapped in a layer's own error, it is still a request that got no
        // response, and its URL is still redacted.
        #[derive(Debug)]
        struct Wrapped(::reqwest::Error);

        impl fmt::Display for Wrapped {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("the layer failed")
            }
        }

        impl StdError for Wrapped {
            fn source(&self) -> Option<&(dyn StdError + 'static)> {
                Some(&self.0)
            }
        }

        let wrapped = ::tower::ServiceBuilder::new()
            .map_err(Wrapped)
            .service(reqwest_service());
        let err = TowerTransport::new(wrapped)
            .send(request())
            .await
            .unwrap_err();
        assert_eq!(err.kind(), TransportErrorKind::Connect);
        let text = err.to_string();
        assert!(text.starts_with("the layer failed: "), "{text}");
        assert!(!text.contains("SECRET"), "{text}");
        assert!(!text.contains("pw"), "{text}");
    }

    /// A `reqwest` client as a `tower` service, as `tower-reqwest` makes
    /// one.
    #[cfg(feature = "reqwest")]
    fn reqwest_service() -> impl Service<
        http::Request<RequestBody>,
        Response = http::Response<TestBody>,
        Error = ::reqwest::Error,
        Future = impl Send,
    > + Clone
    + Send
    + Sync
    + 'static {
        let client = ::reqwest::Client::new();
        ::tower::service_fn(move |request: http::Request<RequestBody>| {
            let request =
                ::reqwest::Request::try_from(request.map(|_| ::reqwest::Body::from(""))).unwrap();
            let response = client.execute(request);
            async move {
                response
                    .await
                    .map(|_| http::Response::new(TestBody::new([])))
            }
        })
    }

    /// A failed body read is a `Body` error (or the `TransportError` the
    /// body failed with), the last item of the body: the inner body is
    /// dropped and never polled again.
    #[tokio::test]
    async fn a_body_error_ends_the_body() {
        let body = TestBody::new([
            Item::Data("a"),
            Item::Error("reset by peer".into()),
            Item::Data("b"),
        ]);
        let dropped = body.dropped.clone();
        let response = TowerTransport::new(Scripted::ok(body))
            .send(get())
            .await
            .unwrap();
        let mut body = response.into_body();
        assert_eq!(body.next().await.unwrap().unwrap(), "a");
        let err = body.next().await.unwrap().unwrap_err();
        assert_eq!(err.kind(), TransportErrorKind::Body);
        assert_eq!(err.to_string(), "reset by peer");
        assert!(dropped.load(Ordering::SeqCst));
        assert!(body.next().await.is_none());

        let failed = TestBody::new([Item::Error(Box::new(TransportError::new(
            TransportErrorKind::Timeout,
            "idle",
        )))]);
        let response = TowerTransport::new(Scripted::ok(failed))
            .send(get())
            .await
            .unwrap();
        let err = chunks(response).await.unwrap_err();
        assert_eq!(err.kind(), TransportErrorKind::Timeout);
    }

    /// The request timeout bounds the wait for the response, and cancels
    /// the service's future.
    #[tokio::test]
    async fn the_timeout_bounds_the_response() {
        let hangs = Hangs::default();
        let err = TowerTransport::new(hangs.clone())
            .send(get().timeout(Some(SHORT)))
            .await
            .unwrap_err();
        assert_eq!(err.kind(), TransportErrorKind::Timeout);
        assert_eq!(
            err.into_error().as_log().to_string(),
            r#"kind=timeout message="the exchange exceeded 50ms""#
        );
        assert!(hangs.dropped.load(Ordering::SeqCst));
    }

    /// The wait for the service to be ready counts too.
    #[tokio::test]
    async fn the_timeout_bounds_the_wait_for_readiness() {
        #[derive(Clone)]
        struct NeverReady;

        impl Service<http::Request<RequestBody>> for NeverReady {
            type Response = http::Response<TestBody>;
            type Error = BoxError;
            type Future = std::future::Ready<Reply>;

            fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), BoxError>> {
                Poll::Pending
            }

            fn call(&mut self, _request: http::Request<RequestBody>) -> Self::Future {
                unreachable!("never ready")
            }
        }

        let err = TowerTransport::new(NeverReady)
            .send(get().timeout(Some(SHORT)))
            .await
            .unwrap_err();
        assert_eq!(err.kind(), TransportErrorKind::Timeout);
    }

    /// The same timer bounds the body: a stalled body fails with a timeout
    /// and is dropped.
    #[tokio::test]
    async fn the_timeout_bounds_the_body() {
        let body = TestBody::new([Item::Data("a"), Item::Stall]);
        let dropped = body.dropped.clone();
        let response = TowerTransport::new(Scripted::ok(body))
            .send(get().timeout(Some(SHORT)))
            .await
            .unwrap();
        let started = Instant::now();
        let mut body = response.into_body();
        assert_eq!(body.next().await.unwrap().unwrap(), "a");
        let err = body.next().await.unwrap().unwrap_err();
        assert_eq!(err.kind(), TransportErrorKind::Timeout);
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(dropped.load(Ordering::SeqCst));
        assert!(body.next().await.is_none());
    }

    /// A body that always has data ready cannot outlive the timeout.
    #[tokio::test]
    async fn the_timeout_bounds_a_body_that_never_waits() {
        struct Endless;

        impl http_body::Body for Endless {
            type Data = Bytes;
            type Error = BoxError;

            fn poll_frame(
                self: Pin<&mut Self>,
                _cx: &mut Context<'_>,
            ) -> Poll<Option<Result<Frame<Bytes>, BoxError>>> {
                Poll::Ready(Some(Ok(Frame::data(Bytes::from_static(b"x")))))
            }
        }

        let service = ::tower::service_fn(|_request: http::Request<RequestBody>| async {
            Ok::<_, BoxError>(http::Response::new(Endless))
        });
        let response = TowerTransport::new(service)
            .send(get().timeout(Some(SHORT)))
            .await
            .unwrap();
        let mut body = response.into_body();
        let err = loop {
            match body.next().await.unwrap() {
                Ok(_) => tokio::task::yield_now().await,
                Err(err) => break err,
            }
        };
        assert_eq!(err.kind(), TransportErrorKind::Timeout);
    }

    /// Without a timeout, nothing bounds the exchange: a stalled body is
    /// still pending later.
    #[tokio::test]
    async fn no_timeout_no_bound() {
        let body = TestBody::new([Item::Stall]);
        let response = TowerTransport::new(Scripted::ok(body))
            .send(get())
            .await
            .unwrap();
        let mut body = response.into_body();
        let next =
            futures_util::future::select(body.next(), futures_timer::Delay::new(SHORT * 2)).await;
        assert!(matches!(next, Either::Right(_)));
    }

    /// The stream timeouts providers apply to streaming requests work over
    /// the adapter.
    #[tokio::test]
    async fn the_stream_idle_bound_fires_over_the_adapter() {
        let body = TestBody::new([Item::Data("data: a\n\n"), Item::Stall]);
        let transport = TowerTransport::new(Scripted::ok(body));
        let response = StreamTimeouts::unbounded()
            .idle(Some(SHORT))
            .send(&transport, get())
            .await
            .unwrap();
        let mut body = response.into_body();
        assert_eq!(body.next().await.unwrap().unwrap(), "data: a\n\n");
        let err = body.next().await.unwrap().unwrap_err();
        assert_eq!(err.kind(), TransportErrorKind::Timeout);
    }

    /// Dropping the call drops the service's future, and dropping the body
    /// drops the inner body.
    #[tokio::test]
    async fn dropping_cancels() {
        let hangs = Hangs::default();
        let transport = TowerTransport::new(hangs.clone());
        let mut send = transport.send(get());
        assert!((&mut send).now_or_never().is_none());
        assert!(!hangs.dropped.load(Ordering::SeqCst));
        drop(send);
        assert!(hangs.dropped.load(Ordering::SeqCst));

        let body = TestBody::new([Item::Data("a"), Item::Stall]);
        let dropped = body.dropped.clone();
        let response = TowerTransport::new(Scripted::ok(body))
            .send(get())
            .await
            .unwrap();
        let mut body = response.into_body();
        body.next().await.unwrap().unwrap();
        assert!(!dropped.load(Ordering::SeqCst));
        drop(body);
        assert!(dropped.load(Ordering::SeqCst));
    }

    /// The service is cloned for each request, so a concurrency limit
    /// holds across them, and a request waiting for a permit gives it back
    /// when dropped.
    #[tokio::test]
    async fn a_concurrency_limit_holds_across_requests() {
        let gate = Arc::new(tokio::sync::Semaphore::new(0));
        let calls = Arc::new(AtomicUsize::new(0));
        let service = ::tower::service_fn({
            let (gate, calls) = (gate.clone(), calls.clone());
            move |_request: http::Request<RequestBody>| {
                calls.fetch_add(1, Ordering::SeqCst);
                let gate = gate.clone();
                async move {
                    gate.acquire().await.unwrap().forget();
                    Ok::<_, BoxError>(http::Response::new(TestBody::data("ok")))
                }
            }
        });
        let transport = TowerTransport::new(
            ::tower::ServiceBuilder::new()
                .concurrency_limit(1)
                .service(service),
        );

        let mut first = transport.send(get());
        assert!((&mut first).now_or_never().is_none());
        let mut second = transport.send(get());
        assert!((&mut second).now_or_never().is_none());
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "the limit let both through"
        );

        gate.add_permits(2);
        first.await.unwrap();
        second.await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);

        // A request waiting for the permit, dropped, does not keep it.
        let mut holder = transport.send(get());
        assert!((&mut holder).now_or_never().is_none());
        let mut waiting = transport.send(get());
        assert!((&mut waiting).now_or_never().is_none());
        drop(waiting);
        gate.add_permits(2);
        holder.await.unwrap();
        transport.send(get()).await.unwrap();
    }

    #[test]
    fn the_transport_is_send_sync_clone_and_debug() {
        fn assert<T: HttpTransport + Clone + fmt::Debug>() {}
        assert::<TowerTransport<Scripted>>();
        let transport = TowerTransport::new(Scripted::new([]));
        assert_eq!(format!("{transport:?}"), "TowerTransport { .. }");
        let _service: &Scripted = transport.get_ref();
        let _service: Scripted = transport.into_inner();
    }
}

/// Run on `wasm32` with `wasm-bindgen-test`: the timeout's timer goes
/// through the host's `setTimeout`.
#[cfg(target_arch = "wasm32")]
mod wasm {
    use std::time::Duration;

    use futures_util::StreamExt;
    use wasm_bindgen_test::wasm_bindgen_test;

    use super::*;

    #[wasm_bindgen_test]
    async fn the_timeout_bounds_the_response_and_the_body() {
        let err = TowerTransport::new(Hangs::default())
            .send(get().timeout(Some(Duration::from_millis(20))))
            .await
            .unwrap_err();
        assert_eq!(err.kind(), TransportErrorKind::Timeout);

        let body = TestBody::new([Item::Data("a"), Item::Stall]);
        let response = TowerTransport::new(Scripted::ok(body))
            .send(get().timeout(Some(Duration::from_millis(20))))
            .await
            .unwrap();
        let mut body = response.into_body();
        assert_eq!(body.next().await.unwrap().unwrap(), "a");
        let err = body.next().await.unwrap().unwrap_err();
        assert_eq!(err.kind(), TransportErrorKind::Timeout);
    }

    #[wasm_bindgen_test]
    async fn a_transport_error_is_passed_through() {
        let err = failure(Scripted::failing(TransportError::new(
            TransportErrorKind::Connect,
            "refused",
        )))
        .await;
        assert_eq!(err.kind(), TransportErrorKind::Connect);
    }
}
