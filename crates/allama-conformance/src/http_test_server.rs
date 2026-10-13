//! A local HTTP server for provider transport tests.

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use allama_http::{Bytes, Headers};
use axum::Router;
use axum::body::{Body, to_bytes};
use axum::extract::{Request, State};
use axum::response::Response;
use axum::routing::any;
use serde::Serialize;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

/// The status [`TestHttpServer`] answers a request with when no response
/// is queued for it: not a status any real API sends, so a failure caused
/// by an unplanned request is recognisable in the error.
pub const UNPLANNED_REQUEST_STATUS: u16 = 599;

/// A canned response that [`TestHttpServer`] answers with.
///
/// Its constructors match [`ReplayResponse`](allama_http::ReplayResponse)'s,
/// so a test moves between the server and the replay transport unchanged:
/// [`json`](Self::json) takes the body as written, [`json_value`](Self::json_value)
/// serializes a value, and [`header`](Self::header) adds a header.
#[derive(Debug, Clone)]
pub struct MockHttpResponse {
    status: u16,
    headers: Vec<(String, String)>,
    body: Bytes,
}

impl MockHttpResponse {
    /// A response with `status` and a plain `body` (no `content-type`).
    #[must_use]
    pub fn text(status: u16, body: impl Into<String>) -> Self {
        Self::bytes(status, body.into())
    }

    /// A response with `status` and the JSON `body` as written, as
    /// `application/json`: a captured body, or one that is not valid JSON
    /// on purpose.
    #[must_use]
    pub fn json(status: u16, body: impl Into<Bytes>) -> Self {
        Self::bytes(status, body).header("content-type", "application/json")
    }

    /// A response with `status` and `value` serialized as an
    /// `application/json` body.
    ///
    /// # Panics
    ///
    /// When `value` does not serialize.
    #[must_use]
    pub fn json_value(status: u16, value: &impl Serialize) -> Self {
        Self::json(
            status,
            serde_json::to_vec(value).expect("serialize mock JSON response"),
        )
    }

    /// A `200` `text/event-stream` response with `body` as the SSE stream,
    /// terminated by a blank line if it is not already.
    #[must_use]
    pub fn sse(body: impl Into<String>) -> Self {
        let body = body.into();
        let normalized = if body.ends_with("\n\n") {
            body
        } else {
            format!("{body}\n\n")
        };

        Self::text(200, normalized).header("content-type", "text/event-stream")
    }

    /// A `200` `application/x-ndjson` response with `body` as the
    /// newline-delimited JSON stream, as written.
    #[must_use]
    pub fn ndjson(body: impl Into<String>) -> Self {
        Self::text(200, body).header("content-type", "application/x-ndjson")
    }

    /// Add a response header (`retry-after` on a `429`, for example).
    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    fn bytes(status: u16, body: impl Into<Bytes>) -> Self {
        Self {
            status,
            headers: Vec::new(),
            body: body.into(),
        }
    }
}

/// A request [`TestHttpServer`] received.
///
/// Its `Debug` output prints the body as text when it is UTF-8 (a JSON
/// body reads as JSON), and as bytes otherwise.
#[derive(Clone)]
#[non_exhaustive]
pub struct RecordedRequest {
    /// The HTTP method (`POST`, for example).
    pub method: String,
    /// The URI path, without the query.
    pub path: String,
    /// The raw query string, without the leading `?`.
    pub query: Option<String>,
    /// The headers in the order received, names in lowercase, a header
    /// sent twice listed twice (see [`header_values`](Self::header_values)).
    /// Headers whose value is not visible ASCII are left out. `Debug`
    /// redacts credential values.
    pub headers: Headers,
    /// The raw request body.
    pub body: Vec<u8>,
}

impl RecordedRequest {
    /// The first value of header `name` (matched case-insensitively).
    #[must_use]
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name)
    }

    /// Every value of header `name` (matched case-insensitively), in the
    /// order received: a credential sent once has one.
    #[must_use]
    pub fn header_values(&self, name: &str) -> Vec<&str> {
        self.headers
            .iter()
            .filter(|(header, _)| header.eq_ignore_ascii_case(name))
            .map(|(_, value)| value)
            .collect()
    }

    /// The body as UTF-8 text.
    ///
    /// # Panics
    ///
    /// When the body is not valid UTF-8.
    #[must_use]
    pub fn body_text(&self) -> &str {
        std::str::from_utf8(&self.body).expect("recorded request body should be UTF-8")
    }

    /// The body parsed as JSON.
    ///
    /// # Panics
    ///
    /// When the body is not valid JSON.
    #[must_use]
    pub fn body_json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).expect("recorded request body should be valid JSON")
    }
}

impl fmt::Debug for RecordedRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("RecordedRequest");
        debug
            .field("method", &self.method)
            .field("path", &self.path)
            .field("query", &self.query)
            .field("headers", &self.headers);
        match std::str::from_utf8(&self.body) {
            Ok(text) => debug.field("body", &text),
            Err(_) => debug.field("body", &self.body),
        };
        debug.finish()
    }
}

/// What the server and its handle share.
#[derive(Default)]
struct Shared {
    requests: Vec<RecordedRequest>,
    responses: VecDeque<MockHttpResponse>,
    /// `METHOD path` of each request that arrived with no response queued.
    unplanned: Vec<String>,
}

type AppState = Arc<Mutex<Shared>>;

/// Lock the shared state, also after a panic elsewhere poisoned it.
fn lock(state: &Mutex<Shared>) -> MutexGuard<'_, Shared> {
    state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn recorded_headers(headers: &axum::http::HeaderMap) -> Headers {
    let mut recorded = Headers::new();
    for (name, value) in headers {
        if let Ok(value) = value.to_str() {
            recorded.insert(name.as_str().to_ascii_lowercase(), value);
        }
    }
    recorded
}

async fn handle_request(State(state): State<AppState>, request: Request) -> Response {
    let (parts, body) = request.into_parts();
    let body = to_bytes(body, usize::MAX)
        .await
        .expect("read mock server request body");

    let response = {
        let mut shared = lock(&state);
        shared.requests.push(RecordedRequest {
            method: parts.method.to_string(),
            path: parts.uri.path().to_string(),
            query: parts.uri.query().map(str::to_string),
            headers: recorded_headers(&parts.headers),
            body: body.to_vec(),
        });
        let next = shared.responses.pop_front();
        if next.is_none() {
            let request = format!("{} {}", parts.method, parts.uri.path());
            shared.unplanned.push(request);
        }
        next
    };
    let response = response.unwrap_or_else(|| {
        MockHttpResponse::text(
            UNPLANNED_REQUEST_STATUS,
            "TestHttpServer: no response is queued for this request; queue one for each \
             request the test expects",
        )
        .header("x-test-http-server", "unplanned-request")
    });

    let mut builder = Response::builder().status(response.status);
    for (name, value) in response.headers {
        builder = builder.header(name, value);
    }

    builder
        .body(Body::from(response.body))
        .expect("build mock HTTP response")
}

/// A local HTTP server that answers each request with the next queued
/// [`MockHttpResponse`] and records every request it receives.
///
/// Point the provider under test at [`url`](Self::url) to exercise its real
/// transport: URL and endpoint selection, auth headers, unary and streaming
/// decoding, and HTTP error mapping. The server stops when dropped.
///
/// A request that arrives when no response is queued is answered with
/// [`UNPLANNED_REQUEST_STATUS`], and the test fails: the server panics when
/// it is dropped, so an extra request (an unintended retry, say) never
/// passes unnoticed, whatever the client made of the answer.
/// [`assert_exhausted`](Self::assert_exhausted) also fails on a queued
/// response no request took, for a test that expects every one taken; a
/// test may queue a response it expects to stay untaken (a redirect target
/// that must not be followed), and check [`remaining`](Self::remaining).
///
/// ```no_run
/// use allama_conformance::{MockHttpResponse, TestHttpServer};
///
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// let server = TestHttpServer::spawn([
///     MockHttpResponse::json(429, r#"{"error":"slow down"}"#).header("retry-after", "2"),
/// ])
/// .await;
/// // Build the provider with `server.url()` as its base URL and call it.
/// let requests = server.recorded_requests().await;
/// # let _ = requests;
/// # }
/// ```
///
/// Requires the `server` feature (on by default) and a Tokio runtime.
pub struct TestHttpServer {
    base_url: String,
    state: AppState,
    task: JoinHandle<()>,
}

impl TestHttpServer {
    /// Start a server on a free local port that answers with `responses`,
    /// in order.
    ///
    /// Must be called within a Tokio runtime. A request that arrives after
    /// the queue is empty is answered with [`UNPLANNED_REQUEST_STATUS`] and
    /// fails the test when the server is dropped.
    ///
    /// # Panics
    ///
    /// When no local port can be bound.
    pub async fn spawn(responses: impl IntoIterator<Item = MockHttpResponse>) -> Self {
        let state: AppState = Arc::new(Mutex::new(Shared {
            responses: VecDeque::from_iter(responses),
            ..Shared::default()
        }));

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock HTTP server listener");
        let base_url = format!(
            "http://{}",
            listener.local_addr().expect("local listener addr")
        );
        let app = Router::new()
            .fallback(any(handle_request))
            .with_state(Arc::clone(&state));

        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        Self {
            base_url,
            state,
            task,
        }
    }

    /// The server's base URL (`http://127.0.0.1:<port>`, no trailing slash).
    #[must_use]
    pub fn url(&self) -> &str {
        &self.base_url
    }

    /// The requests received so far, in order, unplanned ones included.
    pub async fn recorded_requests(&self) -> Vec<RecordedRequest> {
        lock(&self.state).requests.clone()
    }

    /// How many queued responses no request has taken yet.
    #[must_use]
    pub fn remaining(&self) -> usize {
        lock(&self.state).responses.len()
    }

    /// Assert that every queued response was taken and no request arrived
    /// without one. Dropping the server checks the second.
    ///
    /// # Panics
    ///
    /// When a request arrived with no response queued, or a queued
    /// response is left.
    pub fn assert_exhausted(&self) {
        if let Some(problem) = self.problem() {
            panic!("{problem}");
        }
    }

    /// What [`assert_exhausted`](Self::assert_exhausted) fails with.
    fn problem(&self) -> Option<String> {
        if let Some(unplanned) = self.unplanned_problem() {
            return Some(unplanned);
        }
        let shared = lock(&self.state);
        if !shared.responses.is_empty() {
            return Some(format!(
                "TestHttpServer still holds {} queued response(s) after {} request(s); the \
                 adapter sent fewer requests than the test planned",
                shared.responses.len(),
                shared.requests.len()
            ));
        }
        None
    }

    /// The unplanned requests dropping the server fails on.
    fn unplanned_problem(&self) -> Option<String> {
        let shared = lock(&self.state);
        if !shared.unplanned.is_empty() {
            return Some(format!(
                "TestHttpServer received {} request(s) with no response queued (answered \
                 {UNPLANNED_REQUEST_STATUS}): {}; the adapter sent more requests than the test \
                 planned",
                shared.unplanned.len(),
                shared.unplanned.join(", ")
            ));
        }
        None
    }
}

impl fmt::Debug for TestHttpServer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shared = lock(&self.state);
        f.debug_struct("TestHttpServer")
            .field("url", &self.base_url)
            .field("requests", &shared.requests.len())
            .field("remaining", &shared.responses.len())
            .field("unplanned", &shared.unplanned)
            .finish()
    }
}

impl Drop for TestHttpServer {
    fn drop(&mut self) {
        self.task.abort();
        // A test that already fails keeps its own message.
        if !std::thread::panicking()
            && let Some(problem) = self.unplanned_problem()
        {
            panic!("{problem}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recorded(body: &[u8]) -> RecordedRequest {
        RecordedRequest {
            method: "POST".into(),
            path: "/v1/chat".into(),
            query: None,
            headers: Headers::new(),
            body: body.to_vec(),
        }
    }

    async fn post(server: &TestHttpServer, headers: &[(&str, &str)]) -> u16 {
        use allama_http::{HttpRequest, HttpTransport, ReqwestTransport};
        let mut request = HttpRequest::post(format!("{}/v1/chat", server.url()), "{}");
        for (name, value) in headers {
            request = request.header(*name, *value);
        }
        ReqwestTransport::new().send(request).await.unwrap().status
    }

    fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
        panic
            .downcast_ref::<String>()
            .cloned()
            .unwrap_or_else(|| panic.downcast_ref::<&str>().unwrap().to_string())
    }

    #[tokio::test]
    async fn an_unplanned_request_gets_a_distinctive_status_and_fails_the_test() {
        let server = TestHttpServer::spawn([MockHttpResponse::json(200, "{}")]).await;
        assert_eq!(post(&server, &[]).await, 200);
        assert_eq!(post(&server, &[]).await, UNPLANNED_REQUEST_STATUS);
        assert_eq!(server.remaining(), 0);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(server)))
            .expect_err("dropping a server that answered an unplanned request panics");
        let message = panic_message(panic);
        assert!(
            message.contains("1 request(s) with no response queued"),
            "{message}"
        );
        assert!(message.contains("POST /v1/chat"), "{message}");
    }

    #[tokio::test]
    async fn a_leftover_response_fails_the_test() {
        let server = TestHttpServer::spawn([
            MockHttpResponse::json(200, "{}"),
            MockHttpResponse::json(200, "{}"),
        ])
        .await;
        assert_eq!(post(&server, &[]).await, 200);
        assert_eq!(server.remaining(), 1);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            server.assert_exhausted();
        }))
        .expect_err("a queued response is left");
        assert!(panic_message(panic).contains("still holds 1 queued response(s)"));
        // Taking it leaves the server exhausted.
        assert_eq!(post(&server, &[]).await, 200);
        server.assert_exhausted();
    }

    #[tokio::test]
    async fn an_untaken_response_alone_does_not_fail_the_drop() {
        // A redirect target that must not be followed.
        let server = TestHttpServer::spawn([MockHttpResponse::json(200, "{}")]).await;
        assert_eq!(server.remaining(), 1);
        drop(server);
    }

    #[tokio::test]
    async fn a_header_sent_twice_is_recorded_twice() {
        let server = TestHttpServer::spawn([MockHttpResponse::json(200, "{}")]).await;
        post(
            &server,
            &[
                ("Authorization", "Bearer one"),
                ("authorization", "Bearer two"),
            ],
        )
        .await;
        let [request] = &server.recorded_requests().await[..] else {
            panic!("one request")
        };
        assert_eq!(
            request.header_values("authorization"),
            ["Bearer one", "Bearer two"]
        );
        assert_eq!(request.header("AUTHORIZATION"), Some("Bearer one"));
        // Credential values stay out of `Debug`.
        assert!(!format!("{request:?}").contains("Bearer"), "{request:?}");
    }

    #[test]
    fn debug_prints_a_utf8_body_as_text() {
        let debug = format!("{:?}", recorded(br#"{"model":"m"}"#));
        assert!(debug.contains(r#"body: "{\"model\":\"m\"}""#), "{debug}");
    }

    #[test]
    fn debug_prints_other_bodies_as_bytes() {
        let debug = format!("{:?}", recorded(&[0xff, 0x00]));
        assert!(debug.contains("body: [255, 0]"), "{debug}");
    }

    #[test]
    fn an_ndjson_response_keeps_its_body_as_written() {
        let response = MockHttpResponse::ndjson("{\"a\":1}\n{\"b\":2}\n");
        assert_eq!(response.status, 200);
        assert_eq!(
            response.headers,
            [("content-type".to_owned(), "application/x-ndjson".to_owned())]
        );
        assert_eq!(&response.body[..], b"{\"a\":1}\n{\"b\":2}\n");
    }
}
