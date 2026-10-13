use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use futures_util::StreamExt;

use crate::{
    Body, Headers, HttpRequest, HttpResponse, HttpTransport, SendFuture, TransportError,
    TransportErrorKind,
};

/// An [`HttpTransport`] that serves canned responses in order and records
/// every request.
///
/// Use it to run a real provider adapter (request encoding, error
/// classification, SSE decoding) against recorded wire data without a
/// server: in unit tests, on `wasm32`, or to replay a captured exchange.
/// When the queue runs out, requests fail with a
/// [`Connect`](TransportErrorKind::Connect) error.
///
/// Clone is cheap and clones share the queue and the recorded requests.
#[derive(Clone, Default)]
pub struct ReplayTransport {
    state: Arc<Mutex<State>>,
}

#[derive(Default)]
struct State {
    responses: VecDeque<ReplayResponse>,
    requests: Vec<HttpRequest>,
}

impl fmt::Debug for ReplayTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.lock();
        f.debug_struct("ReplayTransport")
            .field("remaining", &state.responses.len())
            .field("requests", &state.requests.len())
            .finish()
    }
}

impl ReplayTransport {
    /// A transport serving `responses` in order.
    #[must_use]
    pub fn new(responses: impl IntoIterator<Item = ReplayResponse>) -> Self {
        Self {
            state: Arc::new(Mutex::new(State {
                responses: responses.into_iter().collect(),
                requests: Vec::new(),
            })),
        }
    }

    /// Queue another response.
    pub fn push(&self, response: ReplayResponse) {
        self.lock().responses.push_back(response);
    }

    /// The requests sent so far, in order.
    #[must_use]
    pub fn requests(&self) -> Vec<HttpRequest> {
        self.lock().requests.clone()
    }

    /// How many queued responses are left.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.lock().responses.len()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl HttpTransport for ReplayTransport {
    fn send(&self, request: HttpRequest) -> SendFuture<'_> {
        let next = {
            let mut state = self.lock();
            state.requests.push(request);
            state.responses.pop_front()
        };
        Box::pin(async move {
            let response = next.ok_or_else(|| {
                TransportError::new(TransportErrorKind::Connect, "no replayed response left")
            })?;
            if let Some((kind, message)) = response.failure {
                return Err(TransportError::new(kind, message));
            }
            let chunks = futures_util::stream::iter(response.chunks.into_iter().map(Ok));
            let body: Body = if response.stalls {
                Box::pin(chunks.chain(futures_util::stream::pending()))
            } else {
                Box::pin(chunks)
            };
            Ok(HttpResponse::new(response.status, response.headers, body))
        })
    }
}

/// A canned response served by [`ReplayTransport`].
///
/// The body arrives in the chunks it was built with: one for
/// [`new`](Self::new) and [`json`](Self::json), one per event for
/// [`sse`](Self::sse), the given ones for [`chunks`](Self::chunks).
/// [`split_every`](Self::split_every) re-chunks any body, down to a byte at
/// a time, to test that a decoder does not depend on where the network
/// splits it.
#[derive(Debug, Clone)]
pub struct ReplayResponse {
    status: u16,
    headers: Headers,
    chunks: Vec<Bytes>,
    failure: Option<(TransportErrorKind, String)>,
    stalls: bool,
}

impl ReplayResponse {
    /// A response with `body` in one chunk.
    #[must_use]
    pub fn new(status: u16, body: impl Into<Bytes>) -> Self {
        Self::chunks(status, [body])
    }

    /// A response whose body arrives in `chunks`, in order: a stream body
    /// in another framing than SSE (NDJSON lines), or chunk boundaries a
    /// test places on purpose.
    ///
    /// ```
    /// use allama_http::{HttpRequest, HttpTransport, ReplayResponse, ReplayTransport};
    /// use futures_util::StreamExt;
    ///
    /// # async fn example() {
    /// let ndjson = ReplayResponse::chunks(200, ["{\"text\":\"Hel\"}\n{\"te", "xt\":\"lo\"}\n"])
    ///     .header("content-type", "application/x-ndjson");
    /// let transport = ReplayTransport::new([ndjson]);
    /// let response = transport.send(HttpRequest::get("https://x")).await.unwrap();
    /// assert_eq!(response.into_body().count().await, 2);
    /// # }
    /// ```
    #[must_use]
    pub fn chunks(status: u16, chunks: impl IntoIterator<Item = impl Into<Bytes>>) -> Self {
        Self {
            status,
            headers: Headers::new(),
            chunks: chunks.into_iter().map(Into::into).collect(),
            failure: None,
            stalls: false,
        }
    }

    /// A JSON response with `body` as written: a captured body, or one
    /// that is not valid JSON on purpose.
    #[must_use]
    pub fn json(status: u16, body: impl Into<Bytes>) -> Self {
        Self::new(status, body).header("content-type", "application/json")
    }

    /// A JSON response with `value` serialized as its body.
    ///
    /// # Panics
    ///
    /// When `value` does not serialize.
    #[must_use]
    pub fn json_value(status: u16, value: &impl serde::Serialize) -> Self {
        let body = serde_json::to_vec(value).expect("serialize replayed JSON response");
        Self::json(status, body)
    }

    /// A `200` server-sent-events response, delivered one event per chunk
    /// the way a network would split it.
    #[must_use]
    pub fn sse(body: impl AsRef<str>) -> Self {
        let chunks = body
            .as_ref()
            .split_inclusive("\n\n")
            .map(|event| Bytes::from(event.to_owned()));
        Self::chunks(200, chunks).header("content-type", "text/event-stream")
    }

    /// A `200` newline-delimited JSON response, delivered one line per
    /// chunk the way a network would split it: the NDJSON counterpart of
    /// [`sse`](Self::sse). Use [`chunks`](Self::chunks) to place chunk
    /// boundaries yourself, or another status.
    ///
    /// ```
    /// use allama_http::{HttpRequest, HttpTransport, ReplayResponse, ReplayTransport};
    /// use futures_util::StreamExt;
    ///
    /// # async fn example() {
    /// let transport = ReplayTransport::new([ReplayResponse::ndjson(
    ///     "{\"delta\":\"Hel\"}\n{\"delta\":\"lo\"}\n{\"done\":true}\n",
    /// )]);
    /// let response = transport.send(HttpRequest::get("https://x")).await.unwrap();
    /// assert_eq!(response.headers.get("content-type"), Some("application/x-ndjson"));
    /// assert_eq!(response.into_body().count().await, 3);
    /// # }
    /// ```
    #[must_use]
    pub fn ndjson(body: impl AsRef<str>) -> Self {
        let chunks = body
            .as_ref()
            .split_inclusive('\n')
            .map(|line| Bytes::from(line.to_owned()));
        Self::chunks(200, chunks).header("content-type", "application/x-ndjson")
    }

    /// A request that gets no response, failing with `kind`.
    #[must_use]
    pub fn failure(kind: TransportErrorKind, message: impl Into<String>) -> Self {
        Self {
            status: 0,
            headers: Headers::new(),
            chunks: Vec::new(),
            failure: Some((kind, message.into())),
            stalls: false,
        }
    }

    /// Add a response header.
    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name, value);
        self
    }

    /// Deliver the body in chunks of `size` bytes (the last one may be
    /// shorter), whatever chunks it was built with: `split_every(1)` sends
    /// one byte at a time, splitting every line ending and UTF-8 sequence.
    ///
    /// ```
    /// use allama_http::{HttpRequest, HttpTransport, ReplayResponse, ReplayTransport};
    /// use futures_util::StreamExt;
    ///
    /// # async fn example() {
    /// let transport = ReplayTransport::new([ReplayResponse::sse("data: hi\n\n").split_every(4)]);
    /// let response = transport.send(HttpRequest::get("https://x")).await.unwrap();
    /// assert_eq!(response.into_body().count().await, 3);
    /// # }
    /// ```
    ///
    /// # Panics
    ///
    /// When `size` is 0.
    #[must_use]
    pub fn split_every(mut self, size: usize) -> Self {
        assert!(
            size > 0,
            "a replayed body cannot be split into empty chunks"
        );
        let body: Vec<u8> = self.chunks.concat();
        self.chunks = body.chunks(size).map(Bytes::copy_from_slice).collect();
        self
    }

    /// Never end the body after its chunks, like a connection that stalls
    /// mid-stream: for testing idle timeouts such as [`StreamTimeouts`].
    ///
    /// [`StreamTimeouts`]: crate::StreamTimeouts
    #[must_use]
    pub fn then_stall(mut self) -> Self {
        self.stalls = true;
        self
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use futures_util::StreamExt;

    use super::*;

    #[tokio::test]
    async fn responses_are_served_in_order_and_requests_recorded() {
        let transport = ReplayTransport::new([
            ReplayResponse::json(200, r#"{"n":1}"#).header("request-id", "req_1"),
            ReplayResponse::failure(TransportErrorKind::Timeout, "slow"),
        ]);
        let first = transport
            .send(HttpRequest::post("https://x/a", "1"))
            .await
            .unwrap();
        assert_eq!(first.headers.get("Request-Id"), Some("req_1"));
        assert_eq!(first.text().await.unwrap(), r#"{"n":1}"#);
        let second = transport
            .send(HttpRequest::get("https://x/b"))
            .await
            .unwrap_err();
        assert_eq!(second.kind(), TransportErrorKind::Timeout);
        let exhausted = transport
            .send(HttpRequest::get("https://x/c"))
            .await
            .unwrap_err();
        assert_eq!(exhausted.kind(), TransportErrorKind::Connect);

        let urls: Vec<_> = transport.requests().into_iter().map(|r| r.url).collect();
        assert_eq!(urls, ["https://x/a", "https://x/b", "https://x/c"]);
        assert_eq!(transport.remaining(), 0);
    }

    #[tokio::test]
    async fn sse_is_chunked_per_event() {
        let transport = ReplayTransport::default();
        transport.push(ReplayResponse::sse("data: a\n\ndata: b\n\n"));
        let response = transport.send(HttpRequest::get("https://x")).await.unwrap();
        assert_eq!(
            response.headers.get("content-type"),
            Some("text/event-stream")
        );
        let chunks: Vec<_> = response.into_body().collect().await;
        assert_eq!(chunks.len(), 2);
    }

    #[tokio::test]
    async fn bodies_arrive_in_the_chunks_they_were_built_with_or_split() {
        let body = |response: ReplayResponse| async {
            let transport = ReplayTransport::new([response]);
            let response = transport.send(HttpRequest::get("https://x")).await.unwrap();
            let chunks: Vec<Bytes> = response.into_body().map(Result::unwrap).collect().await;
            chunks
        };
        assert_eq!(
            body(ReplayResponse::chunks(200, ["ab", "", "cde"])).await,
            ["ab", "", "cde"]
        );
        assert_eq!(
            body(ReplayResponse::chunks(200, ["ab", "", "cde"]).split_every(2)).await,
            ["ab", "cd", "e"]
        );
        assert_eq!(
            body(ReplayResponse::new(200, "héllo").split_every(1))
                .await
                .len(),
            "héllo".len()
        );
        assert!(
            body(ReplayResponse::new(200, "").split_every(3))
                .await
                .is_empty()
        );
    }

    #[test]
    fn json_takes_the_body_as_written_and_json_value_serializes() {
        let written = ReplayResponse::json(400, "{not json");
        assert_eq!(written.chunks, ["{not json"]);
        let value = ReplayResponse::json_value(200, &serde_json::json!({"ok": true}));
        assert_eq!(value.chunks, [r#"{"ok":true}"#]);
        assert_eq!(value.headers.get("content-type"), Some("application/json"));
    }

    #[tokio::test]
    async fn stalled_bodies_never_end() {
        let transport = ReplayTransport::new([ReplayResponse::sse("data: a\n\n").then_stall()]);
        let mut body = transport
            .send(HttpRequest::get("https://x"))
            .await
            .unwrap()
            .into_body();
        assert!(body.next().await.is_some());
        let next = futures_util::future::select(
            body.next(),
            futures_timer::Delay::new(std::time::Duration::from_millis(20)),
        )
        .await;
        assert!(matches!(next, futures_util::future::Either::Right(_)));
    }
}
