//! Fixture-based conformance tests.
//!
//! The binding cannot be exercised outside the Workers runtime, so these
//! tests drive the request encoder, response decoder, stream parser and
//! error mapper with recorded payloads.
//!
//! Set `CF_REGENERATE_FIXTURES=1` to rewrite the expected fixtures from the
//! current output (review the diff before committing).

use std::path::PathBuf;

use allama::chat::stream::StreamCollector;
use allama::{
    ChatRequest, ChatStream, ContentBlock, Message, ResponseFormat, StreamEvent, ToolDefinition,
    UnsupportedPolicy,
};
use allama_conformance::stream_errors::ErrorKindProbe;
use allama_conformance::{
    FixtureDir, assert_error_fixture_eq, assert_json_fixture_eq, assert_partial_stream_fixture_eq,
    assert_response_fixture_eq, assert_stream_finish_error_fixture_eq, assert_stream_fixture_eq,
    assert_usage_contract, load_json_fixture, load_text_fixture,
};
use futures_util::StreamExt;
use serde::Serialize;
use serde_json::json;

use crate::error::map_worker_error;
use crate::streaming::conformance_stream_from_sse_text;
use crate::wire;

fn fixtures() -> FixtureDir {
    FixtureDir::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"))
}

fn regenerate() -> bool {
    std::env::var_os("CF_REGENERATE_FIXTURES").is_some()
}

fn write_fixture(name: &str, value: &impl Serialize) {
    let path = fixtures().as_path().join(name);
    // Through `Value`, so the fixture holds what the assertions compare.
    let value = serde_json::to_value(value).unwrap();
    let mut text = serde_json::to_string_pretty(&value).unwrap();
    text.push('\n');
    std::fs::write(path, text).unwrap();
}

async fn regenerate_stream(sse: &str, events_name: &str, response_name: Option<&str>) {
    let mut stream: ChatStream = conformance_stream_from_sse_text(sse);
    let mut collector = StreamCollector::new();
    let mut seen = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(event) => {
                collector.push_ref(&event).unwrap();
                seen.push(serde_json::to_value(&event).unwrap());
            }
            Err(err) => seen.push(json!({"error": serde_json::to_value(err.as_log()).unwrap()})),
        }
    }
    write_fixture(events_name, &seen);
    if let Some(response_name) = response_name {
        let collected = collector.finish_partial().unwrap();
        write_fixture(response_name, &collected.response);
    }
}

fn decode_fixture(raw: &str, turn: usize) -> allama::ChatResponse {
    let raw = load_json_fixture(&fixtures(), raw);
    wire::decode_response(raw, turn).unwrap()
}

fn check_response(raw: &str, expected: &str, turn: usize) {
    let response = decode_fixture(raw, turn);
    if regenerate() {
        write_fixture(expected, &response);
    }
    assert_response_fixture_eq(&response, &fixtures(), expected);
    if let Some(usage) = &response.usage {
        assert_usage_contract(usage);
    }
}

#[test]
fn request_fixture_matches() {
    let request = ChatRequest::new("@cf/meta/llama-3.1-8b-instruct")
        .system("You are concise.")
        .message(Message::user("Find rust docs"))
        .temperature(0.2)
        .top_p(0.9)
        .max_output_tokens(64)
        .seed(42)
        .frequency_penalty(0.1)
        .presence_penalty(0.2)
        .tools(vec![
            ToolDefinition::new(
                "search",
                json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}),
            )
            .description("Search docs"),
        ])
        .response_format(
            ResponseFormat::json_schema(
                json!({"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}),
            )
            .non_strict(),
        )
        .top_k(40)
        .with_option(crate::ChatRequestOptions::new().repetition_penalty(1.1))
        .unsupported_policy(UnsupportedPolicy::Error);

    let (actual, warnings) = wire::encode_request(&request, false).unwrap();
    assert!(warnings.is_empty());
    if regenerate() {
        write_fixture("request.json", &actual);
    }
    assert_json_fixture_eq(&actual, &fixtures(), "request.json");
}

#[test]
fn tool_round_trip_request_fixture_matches() {
    let request = ChatRequest::new("@cf/meta/llama-3.3-70b-instruct-fp8-fast")
        .user("Weather in London?")
        .message(Message::assistant_blocks(vec![ContentBlock::tool_call(
            "cf_call_1_0",
            "lookup_weather",
            r#"{"city":"London"}"#,
        )]))
        .message(Message::tool_result("cf_call_1_0", "lookup_weather", "12C, cloudy"))
        .tools(vec![ToolDefinition::new(
            "lookup_weather",
            json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
        )]);

    let (actual, warnings) = wire::encode_request(&request, false).unwrap();
    assert!(warnings.is_empty());
    if regenerate() {
        write_fixture("request_tool_round_trip.json", &actual);
    }
    assert_json_fixture_eq(&actual, &fixtures(), "request_tool_round_trip.json");
}

#[test]
fn response_fixture_matches() {
    check_response("response_raw.json", "response_expected.json", 1);
}

#[test]
fn final_usage_response_fixture_matches() {
    check_response(
        "response_finish_reason_raw.json",
        "response_finish_reason_expected.json",
        1,
    );
}

#[test]
fn tool_response_fixture_matches() {
    // The request had one message, so the assistant turn is index 1.
    check_response("response_tools_raw.json", "response_tools_expected.json", 1);
}

#[test]
fn openai_style_tool_response_fixture_matches() {
    check_response(
        "response_tools_openai_raw.json",
        "response_tools_openai_expected.json",
        1,
    );
}

#[test]
fn chat_completion_response_fixture_matches() {
    check_response(
        "response_chat_completion_raw.json",
        "response_chat_completion_expected.json",
        1,
    );
}

#[test]
fn structured_response_fixture_matches() {
    check_response("response_json_raw.json", "response_json_expected.json", 1);
}

#[tokio::test]
async fn stream_fixture_matches() {
    let sse = load_text_fixture(&fixtures(), "stream.sse");
    if regenerate() {
        regenerate_stream(
            &sse,
            "stream_events.json",
            Some("stream_response_expected.json"),
        )
        .await;
    }
    assert_stream_fixture_eq(
        conformance_stream_from_sse_text(&sse),
        &fixtures(),
        "stream_events.json",
        "stream_response_expected.json",
    )
    .await;

    let response = conformance_stream_from_sse_text(&sse)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .filter_map(Result::ok)
        .find_map(|event| match event {
            StreamEvent::ResponseMetadata { usage, .. } => usage,
            _ => None,
        })
        .expect("final usage chunk is reported");
    assert_usage_contract(&response);
}

#[tokio::test]
async fn legacy_stream_without_usage_chunk_fixture_matches() {
    let sse = load_text_fixture(&fixtures(), "stream_legacy.sse");
    if regenerate() {
        regenerate_stream(
            &sse,
            "stream_legacy_events.json",
            Some("stream_legacy_response_expected.json"),
        )
        .await;
    }
    assert_stream_fixture_eq(
        conformance_stream_from_sse_text(&sse),
        &fixtures(),
        "stream_legacy_events.json",
        "stream_legacy_response_expected.json",
    )
    .await;
}

#[tokio::test]
async fn truncated_stream_reports_incomplete_response() {
    let sse = load_text_fixture(&fixtures(), "stream_truncated.sse");
    if regenerate() {
        regenerate_stream(
            &sse,
            "stream_truncated_events.json",
            Some("stream_truncated_response_expected.json"),
        )
        .await;
        let err = conformance_stream_from_sse_text(&sse)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .find_map(Result::err)
            .unwrap();
        write_fixture("stream_truncated_error.json", &err.as_log());
    }

    assert_stream_finish_error_fixture_eq(
        conformance_stream_from_sse_text(&sse),
        &fixtures(),
        "stream_truncated_events.json",
        "stream_truncated_error.json",
    )
    .await;

    assert_partial_stream_fixture_eq(
        conformance_stream_from_sse_text(&sse),
        &fixtures(),
        "stream_truncated_events.json",
        "stream_truncated_response_expected.json",
        "stream_truncated_completeness.json",
    )
    .await;
}

#[test]
fn error_fixtures_match() {
    let cases = [
        ("Unauthorized request", "error_auth.json"),
        ("Rate limit exceeded", "error_rate_limited.json"),
        ("Worker request timed out", "error_timeout.json"),
        (
            "This model's maximum context length is 32768 tokens",
            "error_context_length.json",
        ),
        (
            "Model not found: @cf/meta/missing",
            "error_model_not_found.json",
        ),
        (
            "Safety system blocked this response",
            "error_content_filtered.json",
        ),
        (
            "InferenceUpstreamError: 3040: Capacity temporarily exceeded, please try again.",
            "error_overloaded.json",
        ),
        (
            "3036: You have used up your daily free allocation of 10,000 neurons",
            "error_quota_exhausted.json",
        ),
        (
            "AiError: 5007: No such model @cf/meta/missing or task",
            "error_no_such_model.json",
        ),
    ];
    for (message, name) in cases {
        let err = map_worker_error(worker::Error::RustError(message.to_string()));
        if regenerate() {
            write_fixture(name, &err.as_log());
        }
        assert_error_fixture_eq(&err, &fixtures(), name);
    }
}

/// The Workers AI error payload for each probed kind: a documented internal
/// error code where Workers AI has one, the message otherwise.
fn error_payload(kind: ErrorKindProbe, upstream: &str) -> Option<String> {
    let (code, message) = match kind {
        ErrorKindProbe::RateLimited => (None, "Rate limit exceeded"),
        ErrorKindProbe::Overloaded => (
            Some(3040),
            "Capacity temporarily exceeded, please try again.",
        ),
        ErrorKindProbe::InvalidRequest => (Some(5004), "Invalid data type for base64 input"),
        ErrorKindProbe::Auth => (
            Some(5018),
            "The account is not allowed to access this model",
        ),
        ErrorKindProbe::QuotaExhausted => (
            Some(3036),
            "You have used up your daily free allocation of 10,000 neurons",
        ),
        ErrorKindProbe::ContextLength => {
            (None, "This model's maximum context length is 32768 tokens")
        }
        ErrorKindProbe::ModelNotFound => (Some(5007), "No such model @cf/meta/missing or task"),
        ErrorKindProbe::Timeout => (Some(3007), "Request timeout"),
        _ => return None,
    };
    let message = format!("{message} {upstream}");
    let mut error = json!({"message": message.trim_end()});
    if let Some(code) = code {
        error["code"] = json!(code);
    }
    Some(json!({"error": error}).to_string())
}

/// The binding reports no HTTP status: a failed call is a JS error whose
/// message carries the internal code (`AiError: 3040: Capacity ...`). This
/// rebuilds that error from the payload and maps it as the binding's
/// failures are mapped, so the status is unused.
fn map_binding_error(_status: u16, _headers: &[(&str, &str)], payload: String) -> allama::Error {
    let payload: serde_json::Value = serde_json::from_str(&payload).unwrap();
    let message = payload["error"]["message"].as_str().unwrap();
    let text = match &payload["error"]["code"] {
        serde_json::Value::Null => message.to_owned(),
        code => format!("AiError: {code}: {message}"),
    };
    map_worker_error(worker::Error::RustError(text))
}

/// Every chat contract, through the binding stand-in, for a listed model,
/// one with tools only, one without either, and one Workers AI does not
/// list. The binding has no HTTP response, so the contracts on an HTTP
/// error mapping, transport options and a malformed body do not apply; its
/// in-band errors are classified like the errors the binding throws.
#[tokio::test]
async fn the_adapter_passes_every_contract() {
    use allama::{ChatProvider, ValidationCoverage};
    use allama_conformance::capabilities::UNKNOWN_MODEL;
    use allama_conformance::stream_end::StreamEnds;
    use allama_conformance::{AdapterUnderTest, Contract, ResponseCapture, assert_chat_adapter};

    use crate::chat::tests::{Binding, probe, stream_probe};

    /// A unary output and a stream of the same answer, from
    /// `fixtures/<name>.json` and `fixtures/<name>.sse`.
    fn capture(name: &str) -> ResponseCapture {
        ResponseCapture::new(
            load_text_fixture(&fixtures(), &format!("{name}.json")),
            load_text_fixture(&fixtures(), &format!("{name}.sse")),
        )
    }

    const MODELS: &[&str] = &[
        "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "@cf/meta/llama-4-scout-17b-16e-instruct",
        "@cf/meta/llama-3.1-8b-instruct",
    ];
    for model in MODELS.iter().chain(&[UNKNOWN_MODEL]) {
        let validation = Binding::default()
            .validate_chat_request(&ChatRequest::new(*model).user("Hi"))
            .unwrap();
        assert_eq!(validation.coverage(), ValidationCoverage::Encoding);
    }
    let started = "data: {\"response\":\"Hel\"}\n\n";
    let under_test = AdapterUnderTest::new(
        MODELS.iter().copied(),
        crate::NAMESPACE,
        probe,
        stream_probe,
    )
    // JSON mode and function calling have no strict mode.
    .strict_not_honoured(MODELS.iter().copied())
    .error_mapping(map_binding_error)
    .error_envelope(error_payload)
    .stream_body(move |payload| format!("{started}data: {payload}\n\n"))
    .stream_replay(|sse| async move {
        Binding::streaming(sse)
            .chat_stream(&ChatRequest::new(MODELS[0]).user("hi"))
            .await
    })
    // A legacy runtime ends with `[DONE]` alone for a normal stop, so a
    // marker without a finish reason is no error case here.
    .stream_ends(StreamEnds::new(started))
    .sse_captures([
        load_text_fixture(&fixtures(), "stream.sse"),
        load_text_fixture(&fixtures(), "stream_legacy.sse"),
    ])
    // `Provider::chat`'s decoding, with the binding's output given as JSON:
    // the binding stand-in answers "ok" without decoding anything.
    .unary_replay(|output| async move {
        let binding = Binding::default();
        let request = ChatRequest::new(MODELS[0]).user("hi");
        let mut prepared = allama::adapter::chat::prepare(&binding, &request)?;
        let messages = prepared.request().messages.len();
        prepared.mark_sent();
        let output: serde_json::Value = serde_json::from_str(&output).unwrap();
        Ok(prepared.finish(wire::decode_response(output, messages)?))
    })
    .response_captures([capture("agree_text")])
    // The binding's mixed shape (recorded from `@cf/qwen/qwen3-30b-a3b-fp8`
    // in `stream_chat_tools.sse`), with two parallel calls.
    .tool_call_capture(
        capture("agree_tool_calls").wire_tool_call_ids(["chatcmpl-tool-a", "chatcmpl-tool-b"]),
    )
    .length_capture(capture("agree_length"))
    .skip([
        // The binding is no HTTP client: there is no redirect to follow.
        Contract::RedirectNotFollowed,
        // The binding throws errors without an HTTP status or headers.
        Contract::ErrorKindsMapped,
        Contract::ErrorLogExcludesBody,
        Contract::RetrySignalsSurfaced,
        // It takes no HTTP headers, and returns decoded values.
        Contract::TransportOptionsApplied,
        Contract::MalformedResponse,
    ]);
    assert_chat_adapter(&Binding::default(), &under_test).await;
}

/// Extraction refusals: a `content_filter` finish fails extraction
/// before parsing, unary and streamed. The binding cannot run here, so the
/// decoded responses go through `ChatExtractor::parse`, the step a call ends
/// with.
#[tokio::test]
async fn content_filter_finishes_fail_extraction() {
    use allama::chat::extract::{
        ChatExtractError, ChatExtractionMode, ChatExtractor, RefusalSource,
    };
    use allama::{ChatStreamExt, FinishReason};

    #[derive(Debug, serde::Deserialize, allama::schemars::JsonSchema)]
    #[schemars(crate = "allama::schemars")]
    #[allow(dead_code)]
    struct Invoice {
        total: f64,
    }

    fn assert_refused(response: allama::ChatResponse) {
        let error = ChatExtractor::new()
            .parse::<Invoice>(response, ChatExtractionMode::Native)
            .unwrap_err();
        let allama::Error::Extract { failure, .. } = error else {
            panic!("an extraction failure expected: {error:?}");
        };
        assert!(
            matches!(
                failure.error(),
                ChatExtractError::Refused {
                    finish_reason: Some(FinishReason::ContentFilter),
                    source: RefusalSource::None,
                    ..
                }
            ),
            "{failure:?}"
        );
    }

    let unary = json!({
        "response": "{\"total\":",
        "finish_reason": "content_filter"
    });
    assert_refused(wire::decode_response(unary, 1).unwrap());

    let streamed = conformance_stream_from_sse_text(
        "data: {\"response\":\"{\\\"total\\\":\"}\n\n\
         data: {\"response\":\"\",\"finish_reason\":\"content_filter\"}\n\n\
         data: [DONE]\n\n",
    )
    .collect_response()
    .await
    .unwrap();
    assert_refused(streamed);
}
