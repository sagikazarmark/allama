// Sends through the default `reqwest` transport.
#![cfg(feature = "reqwest")]

use std::path::PathBuf;

use allama::{ChatProvider, ChatRequest, ChatStreamExt, EmbeddingProvider, EmbeddingRequest};
use allama_conformance::contract::{
    assert_auth_error_contract, assert_chat_success_contract,
    assert_context_length_exceeded_contract, assert_overloaded_error_contract,
    assert_quota_exhausted_contract, assert_rate_limited_error_contract,
    assert_streaming_chat_contract, assert_streaming_tool_call_contract,
    assert_strict_policy_rejects, assert_strict_policy_rejects_stream,
    assert_structured_output_contract, assert_structured_output_contract_stream,
    assert_tool_call_contract, assert_unsupported_hint_warns, assert_unsupported_hint_warns_stream,
    collect_chat_stream,
};
use allama_conformance::{
    FixtureDir, MockHttpResponse, TestHttpServer, assert_embedding_response_fixture_eq,
    assert_response_fixture_eq, assert_stream_fixture_eq, load_json_fixture, load_text_fixture,
};
use allama_gemini::{ChatRequestOptions, Provider, SafetySetting};
use serde_json::{Value, json};

fn fixtures() -> FixtureDir {
    FixtureDir::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"))
}

fn provider(server: &TestHttpServer) -> Provider {
    Provider::builder()
        .api_key("gemini-test-key")
        .base_url(format!("{}/v1beta/", server.url()))
        .build()
        .unwrap()
}

fn text_response(text: &str) -> Value {
    json!({
        "responseId": "resp-1",
        "modelVersion": "gemini-2.5-flash",
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "thoughtsTokenCount": 3, "totalTokenCount": 10}
    })
}

fn error_body(code: u16, message: &str, status: &str) -> Value {
    json!({"error": {"code": code, "message": message, "status": status}})
}

#[tokio::test]
async fn chat_posts_expected_request_and_parses_response() {
    let fixtures = fixtures();
    let raw = load_json_fixture(&fixtures, "response_raw.json");
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(200, &raw)]).await;

    let response = provider(&server)
        .chat(
            &ChatRequest::new("gemini-2.5-pro")
                .user("Hello from the transport integration test.")
                .with_option(
                    ChatRequestOptions::new()
                        .cached_content("cachedContents/integration-test")
                        .safety_setting(SafetySetting::new(
                            "HARM_CATEGORY_HARASSMENT",
                            "BLOCK_ONLY_HIGH",
                        ))
                        .generation_config_extra("responseLogprobs", true)
                        .extra("labels", json!({"suite": "integration"})),
                ),
        )
        .await
        .unwrap();

    assert_response_fixture_eq(&response, &fixtures, "response_expected.json");

    let requests = server.recorded_requests().await;
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    let body = request.body_json();
    assert_eq!(request.method, "POST");
    assert_eq!(
        request.path,
        "/v1beta/models/gemini-2.5-pro:generateContent"
    );
    assert_eq!(request.header("x-goog-api-key"), Some("gemini-test-key"));
    assert_eq!(request.header("content-type"), Some("application/json"));
    assert_eq!(body["cachedContent"], "cachedContents/integration-test");
    assert!(body["generationConfig"].get("candidateCount").is_none());
    assert_eq!(
        body["safetySettings"],
        json!([{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"}])
    );
    assert_eq!(body["generationConfig"]["responseLogprobs"], true);
    assert_eq!(body["labels"], json!({"suite": "integration"}));
}

#[tokio::test]
async fn models_prefix_is_not_duplicated_in_the_path() {
    let server =
        TestHttpServer::spawn([MockHttpResponse::json_value(200, &text_response("hi"))]).await;
    provider(&server)
        .chat(&ChatRequest::new("models/gemini-2.5-flash").user("hi"))
        .await
        .unwrap();
    let requests = server.recorded_requests().await;
    assert_eq!(
        requests[0].path,
        "/v1beta/models/gemini-2.5-flash:generateContent"
    );
}

/// The PoC model id that reached `tunedModels/x:delete` with
/// the operator's key is refused before anything is sent, on every path;
/// a tuned model keeps its own collection.
#[tokio::test]
async fn hostile_model_ids_send_nothing() {
    let server =
        TestHttpServer::spawn([MockHttpResponse::json_value(200, &text_response("hi"))]).await;
    let provider = provider(&server);
    let hostile = "../../v1/tunedModels/x:delete?ignored=";
    let err = provider
        .chat(&ChatRequest::new(hostile).user("hi"))
        .await
        .unwrap_err();
    assert!(
        matches!(err, allama::Error::InvalidRequest { .. }),
        "{err:?}"
    );
    let err = provider
        .chat_stream(&ChatRequest::new(hostile).user("hi"))
        .await
        .err()
        .expect("refused");
    assert!(
        matches!(err, allama::Error::InvalidRequest { .. }),
        "{err:?}"
    );
    let err = provider
        .embed(&EmbeddingRequest::new(hostile).inputs(["hi"]))
        .await
        .unwrap_err();
    assert!(
        matches!(err, allama::Error::InvalidRequest { .. }),
        "{err:?}"
    );
    assert!(server.recorded_requests().await.is_empty());

    provider
        .chat(&ChatRequest::new("tunedModels/my-model").user("hi"))
        .await
        .unwrap();
    assert_eq!(
        server.recorded_requests().await[0].path,
        "/v1beta/tunedModels/my-model:generateContent"
    );
}

#[tokio::test]
async fn chat_stream_uses_stream_endpoint_and_collects_recorded_fixture() {
    let fixtures = fixtures();
    let server = TestHttpServer::spawn([MockHttpResponse::sse(load_text_fixture(
        &fixtures,
        "stream.sse",
    ))])
    .await;

    let stream = provider(&server)
        .chat_stream(&ChatRequest::new("gemini-3.5-flash-lite").user("Count."))
        .await
        .unwrap();
    assert_stream_fixture_eq(
        stream,
        &fixtures,
        "stream_events.json",
        "stream_response_expected.json",
    )
    .await;

    let requests = server.recorded_requests().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].path,
        "/v1beta/models/gemini-3.5-flash-lite:streamGenerateContent"
    );
    assert_eq!(requests[0].query.as_deref(), Some("alt=sse"));
}

#[tokio::test]
async fn request_warnings_reach_unary_and_streaming_responses() {
    let sse = format!("data: {}\n\n", text_response("hi"));
    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(200, &text_response("hi")),
        MockHttpResponse::sse(sse),
    ])
    .await;
    let provider = provider(&server);
    // Gemini has no switch to disable parallel function calls.
    let request = ChatRequest::new("gemini-2.5-flash")
        .user("hi")
        .parallel_tool_calls(false);

    let response = provider.chat(&request).await.unwrap();
    assert_eq!(response.warnings.len(), 1);
    let streamed = provider
        .chat_stream(&request)
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();
    assert_eq!(streamed.warnings, response.warnings);

    let body = server.recorded_requests().await[0].body_json();
    assert!(body.get("generationConfig").is_none());
}

#[tokio::test]
async fn success_contracts_hold() {
    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(200, &text_response("Hello!")),
        MockHttpResponse::sse(format!("data: {}\n\n", text_response("Hello!"))),
        MockHttpResponse::json_value(200, &text_response("{\"greeting\":\"hi\"}")),
        MockHttpResponse::sse(format!(
            "data: {}\n\n",
            text_response("{\"greeting\":\"hi\"}")
        )),
    ])
    .await;
    let provider = provider(&server);
    let request = ChatRequest::new("gemini-2.5-flash").user("hi");
    assert_chat_success_contract(&provider, &request).await;
    assert_streaming_chat_contract(&provider, &request).await;
    let structured = request
        .clone()
        .response_format(allama::ResponseFormat::Json);
    assert_structured_output_contract(&provider, &structured).await;
    assert_structured_output_contract_stream(&provider, &structured).await;
}

#[tokio::test]
async fn tool_call_contracts_hold_for_recorded_response() {
    let fixtures = fixtures();
    let raw = load_json_fixture(&fixtures, "gemini3_tool_response_raw.json");
    let chunk = serde_json::to_string(&raw).unwrap();
    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(200, &raw),
        MockHttpResponse::sse(format!("data: {chunk}\n\n")),
    ])
    .await;
    let provider = provider(&server);
    let request = ChatRequest::new("gemini-3.5-flash")
        .user("Weather in Paris and Tokyo?")
        .tools([allama::ToolDefinition::new(
            "get_weather",
            json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        )]);
    assert_tool_call_contract(&provider, &request).await;
    assert_streaming_tool_call_contract(&provider, &request).await;
}

#[tokio::test]
async fn error_contracts_hold() {
    let failures = [
        MockHttpResponse::json_value(
            400,
            &json!({"error": {
                "code": 400,
                "message": "API key not valid. Please pass a valid API key.",
                "status": "INVALID_ARGUMENT",
                "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}]
            }}),
        ),
        MockHttpResponse::json_value(
            429,
            &error_body(429, "Resource exhausted", "RESOURCE_EXHAUSTED"),
        ),
        MockHttpResponse::json_value(
            503,
            &error_body(503, "The model is overloaded.", "UNAVAILABLE"),
        ),
        MockHttpResponse::json_value(
            400,
            &error_body(
                400,
                "The input token count (2000000) exceeds the maximum number of tokens allowed (1048576).",
                "INVALID_ARGUMENT",
            ),
        ),
        MockHttpResponse::json_value(
            402,
            &error_body(402, "Prepayment credits depleted.", "RESOURCE_EXHAUSTED"),
        ),
    ];
    // Each failure answers `chat`, then `chat_stream`.
    let server = TestHttpServer::spawn(
        failures
            .into_iter()
            .flat_map(|failure| [failure.clone(), failure]),
    )
    .await;
    let provider = provider(&server);
    let request = ChatRequest::new("gemini-2.5-flash").user("hi");
    let stream = |request| collect_chat_stream(&provider, request);
    assert_auth_error_contract(&provider, &request, stream).await;
    assert_rate_limited_error_contract(&provider, &request, stream).await;
    assert_overloaded_error_contract(&provider, &request, stream).await;
    assert_context_length_exceeded_contract(&provider, &request, stream).await;
    assert_quota_exhausted_contract(&provider, &request, stream).await;
}

#[tokio::test]
async fn unsupported_hints_warn_and_strict_policy_rejects_before_sending() {
    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(200, &text_response("hi")),
        MockHttpResponse::sse(format!("data: {}\n\n", text_response("hi"))),
    ])
    .await;
    let provider = provider(&server);
    let request = ChatRequest::new("gemini-2.5-flash")
        .user("hi")
        .parallel_tool_calls(false);
    assert_unsupported_hint_warns(&provider, &request).await;
    assert_strict_policy_rejects(&provider, &request).await;
    assert_unsupported_hint_warns_stream(&provider, &request).await;
    assert_strict_policy_rejects_stream(&provider, &request).await;
    assert_eq!(server.recorded_requests().await.len(), 2);
}

/// Penalties are sent to every model (ADR 0015). A model that does not
/// enable them refuses with "Penalty is not enabled for this model"
/// (verified live): the provider's refusal of the penalty the request set,
/// on a unary call and when a stream starts.
#[tokio::test]
async fn a_penalty_refusal_is_unsupported_by_the_provider() {
    let refusal = || {
        MockHttpResponse::json_value(
            400,
            &error_body(
                400,
                "Penalty is not enabled for this model",
                "INVALID_ARGUMENT",
            ),
        )
    };
    let server = TestHttpServer::spawn([refusal(), refusal()]).await;
    let provider = provider(&server);
    let request = ChatRequest::new("gemini-2.5-flash")
        .user("hi")
        .presence_penalty(0.2);

    let unary = provider.chat(&request).await.unwrap_err();
    let Err(stream) = provider.chat_stream(&request).await else {
        panic!("the stream started");
    };
    for err in [unary, stream] {
        assert!(matches!(err, allama::Error::Unsupported { .. }), "{err:?}");
        assert_eq!(
            err.unsupported_stage(),
            Some(allama::UnsupportedStage::Provider)
        );
        assert_eq!(
            err.unsupported_field(),
            Some(allama::RequestField::PresencePenalty)
        );
    }
    let body = server.recorded_requests().await[0].body_json();
    assert_eq!(body["generationConfig"]["presencePenalty"], 0.2);
}

#[tokio::test]
async fn rate_limit_prefers_retry_info_then_header() {
    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(
            429,
            &json!({"error": {
                "code": 429,
                "message": "Quota exceeded",
                "status": "RESOURCE_EXHAUSTED",
                "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"}]
            }}),
        )
        .header("retry-after", "1.5"),
        MockHttpResponse::json_value(429, &error_body(429, "Quota exceeded", "RESOURCE_EXHAUSTED"))
            .header("retry-after", "1.5"),
        MockHttpResponse::json_value(429, &error_body(429, "Quota exceeded", "RESOURCE_EXHAUSTED"))
            .header("retry-after", "-1"),
    ])
    .await;
    let provider = provider(&server);
    let request = ChatRequest::new("gemini-2.5-flash").user("hi");

    let err = provider.chat(&request).await.unwrap_err();
    assert!(matches!(err, allama::Error::RateLimited { .. }));
    assert_eq!(err.retry_after(), Some(std::time::Duration::from_secs(7)));
    assert_eq!(err.request_id(), None);

    let err = provider.chat(&request).await.unwrap_err();
    assert_eq!(
        err.retry_after(),
        Some(std::time::Duration::from_secs_f64(1.5))
    );

    let err = provider.chat(&request).await.unwrap_err();
    assert_eq!(err.retry_after(), None);
}

#[tokio::test]
async fn http_errors_carry_the_server_retry_directive() {
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(
        503,
        &error_body(503, "The model is overloaded.", "UNAVAILABLE"),
    )
    .header("x-should-retry", "false")])
    .await;
    let err = provider(&server)
        .chat(&ChatRequest::new("gemini-2.5-flash").user("hi"))
        .await
        .unwrap_err();
    assert!(matches!(err, allama::Error::Overloaded { .. }), "{err:?}");
    assert_eq!(err.server_retry_directive(), Some(false));
    assert!(!allama::wrappers::RetryPolicy::new().should_retry(&err));
}

#[tokio::test]
async fn mid_stream_error_is_typed() {
    let sse = concat!(
        "data: {\"responseId\":\"r\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}]}}]}\n\n",
        "data: {\"error\":{\"code\":429,\"message\":\"Resource exhausted\",\"status\":\"RESOURCE_EXHAUSTED\"}}\n\n",
    );
    let server = TestHttpServer::spawn([MockHttpResponse::sse(sse)]).await;
    let err = provider(&server)
        .chat_stream(&ChatRequest::new("gemini-2.5-flash").user("hi"))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap_err();
    assert!(matches!(err, allama::Error::RateLimited { .. }), "{err:?}");
}

#[tokio::test]
async fn chat_accepts_success_response_without_candidates() {
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(
        200,
        &json!({
            "responseId": "resp-empty",
            "modelVersion": "gemini-2.5-pro",
            "candidates": [],
            "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 0, "totalTokenCount": 4}
        }),
    )])
    .await;

    let response = provider(&server)
        .chat(&ChatRequest::new("gemini-2.5-pro").user("Return nothing."))
        .await
        .unwrap();

    assert!(response.content.is_empty());
    assert_eq!(response.id.as_deref(), Some("resp-empty"));
    assert_eq!(response.model.as_deref(), Some("gemini-2.5-pro"));
    assert_eq!(response.provider.as_deref(), Some("gemini"));
    let usage = response.usage.expect("expected usage metadata");
    assert_eq!(usage.input_tokens, Some(4));
    assert_eq!(usage.output_tokens, Some(0));
    assert_eq!(usage.total_tokens, Some(4));
}

#[tokio::test]
async fn chat_rejects_blocked_prompt_feedback() {
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(
        200,
        &json!({
            "responseId": "resp-blocked",
            "promptFeedback": {"blockReason": "SAFETY", "blockReasonMessage": "Prompt blocked by safety filters"},
            "candidates": []
        }),
    )])
    .await;

    let err = provider(&server)
        .chat(&ChatRequest::new("gemini-2.5-pro").user("Return nothing."))
        .await
        .unwrap_err();
    assert!(matches!(err, allama::Error::ContentFiltered { .. }));
    assert_eq!(err.message(), "prompt was blocked by Gemini: SAFETY");
    assert_eq!(
        err.upstream_message(),
        Some("Prompt blocked by safety filters")
    );
}

#[tokio::test]
async fn embed_posts_expected_request_and_parses_response() {
    let fixtures = fixtures();
    let raw = load_json_fixture(&fixtures, "embed_response_raw.json");
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(200, &raw)]).await;

    let provider = Provider::builder()
        .api_key("test-key")
        .base_url(format!("{}/v1beta", server.url()))
        .build()
        .unwrap();
    assert_eq!(
        provider
            .embedding_limits("gemini-embedding-001")
            .max_inputs_per_call,
        Some(100)
    );

    let request = EmbeddingRequest::new("gemini-embedding-001")
        .inputs(["hello world", "gemini embeddings"])
        // The fixture's vectors have 3 values; a response of another size
        // is rejected.
        .dimensions(3)
        .with_option(allama_gemini::EmbeddingRequestOptions::new().task_type("RETRIEVAL_QUERY"));
    let response = provider.embed(&request).await.unwrap();
    assert_embedding_response_fixture_eq(&response, &fixtures, "embed_response_expected.json");

    let recorded = server.recorded_requests().await;
    assert_eq!(recorded.len(), 1);
    let recorded = &recorded[0];
    assert_eq!(recorded.method, "POST");
    assert_eq!(
        recorded.path,
        "/v1beta/models/gemini-embedding-001:batchEmbedContents"
    );
    assert_eq!(recorded.header("x-goog-api-key"), Some("test-key"));
    let body = recorded.body_json();
    let requests = body["requests"].as_array().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0]["model"], "models/gemini-embedding-001");
    assert_eq!(requests[0]["outputDimensionality"], 3);
    assert_eq!(requests[0]["taskType"], "RETRIEVAL_QUERY");
    assert_eq!(requests[0]["content"]["parts"][0]["text"], "hello world");
}

/// The portable input type reaches the wire as `taskType` on every entry.
#[tokio::test]
async fn embed_sends_the_input_type_as_task_type() {
    let fixtures = fixtures();
    let raw = load_json_fixture(&fixtures, "embed_response_raw.json");
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(200, &raw)]).await;
    let request = EmbeddingRequest::new("gemini-embedding-001")
        .inputs(["hello world", "gemini embeddings"])
        .input_type(allama::EmbeddingInputType::Document);
    let response = provider(&server).embed(&request).await.unwrap();
    assert!(response.warnings.is_empty(), "{:?}", response.warnings);

    let body = server.recorded_requests().await[0].body_json();
    for entry in body["requests"].as_array().unwrap() {
        assert_eq!(entry["taskType"], "RETRIEVAL_DOCUMENT");
    }
}

#[tokio::test]
async fn embed_maps_errors() {
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(
        404,
        &error_body(404, "models/nope is not found", "NOT_FOUND"),
    )])
    .await;
    let err = provider(&server)
        .embed(&EmbeddingRequest::new("nope").input("x"))
        .await
        .unwrap_err();
    assert!(matches!(err, allama::Error::ModelNotFound { .. }));
}

#[tokio::test]
async fn raw_response_is_kept_only_when_requested() {
    let raw = load_json_fixture(&fixtures(), "response_raw.json");
    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(200, &raw),
        MockHttpResponse::json_value(200, &raw),
    ])
    .await;
    let provider = provider(&server);
    let request = ChatRequest::new("gemini-2.5-flash").user("hi");
    let plain = provider.chat(&request).await.unwrap();
    assert!(
        plain
            .provider_data
            .get::<allama_gemini::RawResponse>()
            .unwrap()
            .is_none()
    );
    let with_raw = provider
        .chat(&request.with_options_json("google-ai", json!({"raw_response": true})))
        .await
        .unwrap();
    assert_eq!(
        with_raw
            .provider_data
            .get::<allama_gemini::RawResponse>()
            .unwrap()
            .unwrap()
            .0,
        raw
    );
}

/// The adapter runs end to end over an in-memory transport.
#[tokio::test]
async fn replay_transport_runs_the_adapter_without_a_server() {
    use allama_http::{ReplayResponse, ReplayTransport};

    let transport = ReplayTransport::new([ReplayResponse::sse(load_text_fixture(
        &fixtures(),
        "grounding_stream.sse",
    ))]);
    let provider = Provider::builder()
        .api_key("replay-key")
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let response = provider
        .chat_stream(&ChatRequest::new("gemini-2.5-flash").user("Latest Rust release?"))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();
    assert!(response.citations().count() > 0);
    let sent = &transport.requests()[0];
    assert!(
        sent.url
            .ends_with("/models/gemini-2.5-flash:streamGenerateContent?alt=sse"),
        "{}",
        sent.url
    );
    assert_eq!(sent.headers.get("x-goog-api-key"), Some("replay-key"));
}

/// A stream whose connection stalls after its first event fails with a
/// retryable timeout instead of hanging.
#[tokio::test]
async fn stalled_stream_times_out() {
    use allama_gemini::StreamTimeouts;
    use allama_http::{ReplayResponse, ReplayTransport};

    let sse = load_text_fixture(&fixtures(), "grounding_stream.sse");
    let first_event = sse.split_inclusive("\n\n").next().unwrap();
    let transport = ReplayTransport::new([ReplayResponse::sse(first_event).then_stall()]);
    let provider = Provider::builder()
        .api_key("replay-key")
        .http_transport(transport)
        .stream_timeouts(StreamTimeouts::default().idle(Some(std::time::Duration::from_millis(30))))
        .build()
        .unwrap();
    let err = provider
        .chat_stream(&ChatRequest::new("gemini-2.5-flash").user("hi"))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap_err();
    assert!(matches!(err, allama::Error::Timeout { .. }), "{err:?}");
    assert!(err.is_retryable());
    assert!(
        err.to_string()
            .contains("wait for the next body chunk exceeded 30ms"),
        "{err}"
    );
}

/// An event past the configured size cap fails the stream with a stream
/// protocol error, which retrying would only repeat.
#[tokio::test]
async fn stream_event_past_the_size_cap_fails() {
    use allama::{Error, StreamErrorKind};
    use allama_http::{ReplayResponse, ReplayTransport};

    let sse = load_text_fixture(&fixtures(), "grounding_stream.sse");
    let transport = ReplayTransport::new([ReplayResponse::sse(sse)]);
    let provider = Provider::builder()
        .api_key("replay-key")
        .http_transport(transport)
        .max_stream_event_size(64)
        .build()
        .unwrap();
    let err = provider
        .chat_stream(&ChatRequest::new("gemini-2.5-flash").user("hi"))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap_err();
    assert!(
        matches!(
            err,
            Error::Stream {
                kind: StreamErrorKind::Protocol,
                ..
            }
        ),
        "{err:?}"
    );
    assert!(!err.is_retryable());
    assert!(
        err.to_string()
            .contains("an SSE event passed the size cap of 64 bytes"),
        "{err}"
    );
    assert_eq!(
        err.context().and_then(|c| c.provider.as_deref()),
        Some("gemini")
    );
}

/// A stream line that is not UTF-8 fails the stream with a stream decode
/// error, not a retryable transport failure: nothing replaces the bytes,
/// and asking again streams the same ones.
#[tokio::test]
async fn stream_invalid_utf8_fails_without_a_retry() {
    use std::time::Duration;

    use allama::wrappers::{RetryPolicy, RetryingProvider};
    use allama::{Error, StreamErrorKind, StreamEvent};
    use allama_http::{ReplayResponse, ReplayTransport};
    use futures_util::StreamExt;

    let chunk = |text: &[u8]| {
        let mut chunk =
            b"data: {\"candidates\": [{\"content\": {\"parts\": [{\"text\": \"".to_vec();
        chunk.extend_from_slice(text);
        chunk.extend_from_slice(b"\"}],\"role\": \"model\"},\"index\": 0}]}\n\n");
        chunk
    };
    let sse =
        |body: Vec<u8>| ReplayResponse::new(200, body).header("content-type", "text/event-stream");
    let provider = |transport: &ReplayTransport| {
        Provider::builder()
            .api_key("replay-key")
            .http_transport(transport.clone())
            .build()
            .unwrap()
    };
    let assert_decode_error = |err: &Error| {
        assert!(
            matches!(
                err,
                Error::Stream {
                    kind: StreamErrorKind::Decode,
                    ..
                }
            ),
            "{err:?}"
        );
        assert!(!err.is_retryable());
        assert!(
            err.to_string().contains("invalid UTF-8 in the SSE stream"),
            "{err}"
        );
        assert!(err.as_log().to_string().contains("invalid UTF-8"));
        assert_eq!(
            err.context().and_then(|c| c.provider.as_deref()),
            Some("gemini")
        );
    };

    // Mid-stream: the text before the bad line arrives, then the error.
    let mut body = chunk(b"Hel");
    body.extend(chunk(b"caf\xC3"));
    let transport = ReplayTransport::new([sse(body.clone())]);
    let mut stream = provider(&transport)
        .chat_stream(&ChatRequest::new("gemini-2.5-flash").user("hi"))
        .await
        .unwrap();
    let mut text = String::new();
    let err = loop {
        match stream.next().await.expect("the stream ends with an error") {
            Ok(StreamEvent::TextDelta { text: delta, .. }) => text.push_str(&delta),
            Ok(_) => {}
            Err(err) => break err,
        }
    };
    assert_eq!(text, "Hel");
    assert_decode_error(&err);

    // Before any output: the retry wrapper does not ask again.
    let transport = ReplayTransport::new([sse(chunk(b"\xFF")), sse(body)]);
    let retrying = RetryingProvider::new(
        provider(&transport),
        RetryPolicy::new().base_delay(Duration::ZERO).jitter(false),
    );
    let result = match retrying
        .chat_stream(&ChatRequest::new("gemini-2.5-flash").user("hi"))
        .await
    {
        Ok(stream) => stream.collect_response().await,
        Err(err) => Err(err),
    };
    assert_decode_error(&result.unwrap_err());
    assert_eq!(transport.requests().len(), 1);
}
