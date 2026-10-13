//! `scripts/capture-fixture.sh raw` records an exchange with any endpoint.
//!
//! These tests run the script against a local `TestHttpServer`: it must
//! write exactly the response body, unary or streaming, send the request as
//! given with headers expanded from the environment, and refuse to write a
//! capture that contains one of those secrets. They live in this unpublished
//! crate because the script is outside every published package, and need
//! `bash` and `curl`.
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use allama_conformance::{MockHttpResponse, TestHttpServer};
use serde_json::json;

const SECRET: &str = "sk-capture-test-not-a-real-key";
const REQUEST: &str = r#"{"model":"demo","stream":true,"messages":[]}"#;

#[tokio::test(flavor = "multi_thread")]
async fn raw_mode_captures_a_unary_response() {
    let body = json!({"id": "resp_1", "output": [{"type": "text", "text": "Hello"}]});
    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(200, &body).header("x-request-id", "req_1")
    ])
    .await;
    let dir = Workdir::new("unary");
    let output = dir.path("response_raw.json");

    let result = capture(
        &format!("{}/v1/chat", server.url()),
        &dir,
        &output,
        &["--header", "Authorization: Bearer $CAPTURE_TEST_KEY"],
    )
    .await;

    assert!(result.status.success(), "{}", stderr(&result));
    assert_eq!(
        std::fs::read_to_string(&output).unwrap(),
        serde_json::to_string(&body).unwrap(),
        "only the response body is written"
    );
    assert!(stdout(&result).contains("HTTP 200"), "{}", stdout(&result));

    let requests = server.recorded_requests().await;
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v1/chat");
    assert_eq!(request.body_text(), REQUEST, "the body is sent as-is");
    assert_eq!(
        request.header("authorization"),
        Some(format!("Bearer {SECRET}").as_str()),
        "`$NAME` in a header is read from the environment"
    );
    assert_eq!(request.header("content-type"), Some("application/json"));
}

#[tokio::test(flavor = "multi_thread")]
async fn raw_mode_captures_a_streaming_response() {
    let stream = "event: delta\ndata: {\"text\":\"Hel\"}\n\n\
                  event: delta\ndata: {\"text\":\"lo\"}\n\n\
                  data: [DONE]\n\n";
    let server = TestHttpServer::spawn([MockHttpResponse::sse(stream)]).await;
    let dir = Workdir::new("stream");
    let output = dir.path("stream.sse");

    let result = capture(
        &format!("{}/v1/chat?alt=sse", server.url()),
        &dir,
        &output,
        &[
            "--header=x-api-key: ${CAPTURE_TEST_KEY}",
            "-H",
            "x-api-version: 2026-01-01",
        ],
    )
    .await;

    assert!(result.status.success(), "{}", stderr(&result));
    assert_eq!(std::fs::read_to_string(&output).unwrap(), stream);

    let requests = server.recorded_requests().await;
    assert_eq!(requests[0].query.as_deref(), Some("alt=sse"));
    assert_eq!(requests[0].header("x-api-key"), Some(SECRET));
    assert_eq!(requests[0].header("x-api-version"), Some("2026-01-01"));
}

#[tokio::test(flavor = "multi_thread")]
async fn raw_mode_keeps_a_content_type_header_given() {
    let server = TestHttpServer::spawn([MockHttpResponse::text(200, "ok")]).await;
    let dir = Workdir::new("content-type");
    let output = dir.path("response_raw.txt");

    let result = capture(
        server.url(),
        &dir,
        &output,
        &["--header", "Content-Type: application/vnd.example+json"],
    )
    .await;

    assert!(result.status.success(), "{}", stderr(&result));
    let requests = server.recorded_requests().await;
    assert_eq!(
        requests[0].header("content-type"),
        Some("application/vnd.example+json")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn raw_mode_writes_error_responses_with_their_status() {
    let body = json!({"error": {"type": "rate_limit_error", "message": "slow down"}});
    let server =
        TestHttpServer::spawn(
            [MockHttpResponse::json_value(429, &body).header("retry-after", "3")],
        )
        .await;
    let dir = Workdir::new("error");
    let output = dir.path("error_rate_limited.json");

    let result = capture(server.url(), &dir, &output, &[]).await;

    assert!(result.status.success(), "{}", stderr(&result));
    assert_eq!(
        std::fs::read_to_string(&output).unwrap(),
        serde_json::to_string(&body).unwrap()
    );
    assert!(stdout(&result).contains("HTTP 429"), "{}", stdout(&result));
}

#[tokio::test(flavor = "multi_thread")]
async fn raw_mode_refuses_a_capture_that_contains_a_secret() {
    // An endpoint that echoes the credential back, as some error pages do.
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(
        401,
        &json!({"error": format!("invalid key {SECRET}")}),
    )])
    .await;
    let dir = Workdir::new("secret");
    let output = dir.path("error_auth.json");

    let result = capture(
        server.url(),
        &dir,
        &output,
        &["--header", "Authorization: Bearer $CAPTURE_TEST_KEY"],
    )
    .await;

    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("refusing to write"),
        "{}",
        stderr(&result)
    );
    assert!(
        !stderr(&result).contains(SECRET),
        "errors never echo secrets"
    );
    assert!(!output.exists(), "nothing is written");
    assert!(
        !output.with_extension("json.tmp").exists(),
        "the partial capture is removed"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn raw_mode_rejects_a_header_naming_an_unset_variable() {
    let server = TestHttpServer::spawn([]).await;
    let dir = Workdir::new("unset");
    let output = dir.path("response_raw.json");

    let result = capture(
        server.url(),
        &dir,
        &output,
        &["--header", "Authorization: Bearer $CAPTURE_TEST_UNSET_KEY"],
    )
    .await;

    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("CAPTURE_TEST_UNSET_KEY is not set"),
        "{}",
        stderr(&result)
    );
    assert!(
        server.recorded_requests().await.is_empty(),
        "nothing is sent"
    );
    assert!(!output.exists());
}

/// Run `capture-fixture.sh raw <url> <request> <output> <args>` with
/// `CAPTURE_TEST_KEY` set to [`SECRET`].
async fn capture(url: &str, dir: &Workdir, output: &Path, args: &[&str]) -> Output {
    let request = dir.path("request.json");
    std::fs::write(&request, REQUEST).unwrap();
    let mut command = Command::new("bash");
    command
        .arg(script())
        .arg("raw")
        .arg(url)
        .arg(&request)
        .arg(output)
        .args(args)
        .env("CAPTURE_TEST_KEY", SECRET)
        .env_remove("CAPTURE_TEST_UNSET_KEY");
    tokio::task::spawn_blocking(move || command.output().expect("bash runs"))
        .await
        .unwrap()
}

fn script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts/capture-fixture.sh")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// A fresh directory under Cargo's integration-test scratch space.
struct Workdir(PathBuf);

impl Workdir {
    fn new(name: &str) -> Self {
        let dir =
            PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("capture-fixture-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn path(&self, file: &str) -> PathBuf {
        self.0.join(file)
    }
}
