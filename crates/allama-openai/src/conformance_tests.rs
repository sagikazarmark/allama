//! Fixture-based conformance: requests are captured by a local HTTP server,
//! responses come from recordings of the real API (`fixtures/recorded/`).
//! `fixtures/hand_written/` holds custom tool responses written by hand:
//! the unary one from the SDK types (`ChatCompletionMessageCustomToolCall`),
//! the streamed one in the shape the gated `chat_custom_tool_round_trip`
//! live test checks, since the stream shape is not documented.

use std::path::PathBuf;

use allama::{
    ChatProvider, ChatRequest, ChatStreamExt, EmbeddingProvider, EmbeddingRequest, Error,
    FinishReason, PromptCache, ReasoningEffort, ResponseFormat, SystemPrompt, ToolChoice,
    ToolDefinition, UnsupportedPolicy,
};
use allama_conformance::{
    FixtureDir, MockHttpResponse, TestHttpServer, assert_json_fixture_eq, assert_usage_contract,
    contract, load_text_fixture,
};
use serde_json::json;

use crate::{ChatRequestOptions, Provider, SystemOptions};
use allama_http::http::{HeaderValue, StatusCode};

fn fixtures() -> FixtureDir {
    FixtureDir::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"))
}

fn provider(server: &TestHttpServer) -> Provider {
    Provider::new_with_base_url("test-key", server.url()).unwrap()
}

fn ok_response() -> serde_json::Value {
    json!({
        "id": "chatcmpl-1", "model": "gpt-4o",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    })
}

#[tokio::test]
async fn request_shape_matches_fixture() {
    let server =
        TestHttpServer::spawn([MockHttpResponse::json_value(StatusCode::OK, &ok_response())]).await;
    // A model the profile does not know, so every field is sent as asked:
    // a known reasoning model would drop the sampling fields.
    let request = ChatRequest::new("acme-reasoner")
        .system(SystemPrompt::new("You are a precise assistant.").with_option(SystemOptions::developer()))
        .user("Find rust docs")
        .temperature(0.7)
        .top_p(0.9)
        .max_output_tokens(256)
        .stop(["END"])
        .frequency_penalty(0.1)
        .presence_penalty(0.2)
        .seed(99)
        .parallel_tool_calls(true)
        .tools([ToolDefinition::new(
            "search",
            json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}),
        )
        .description("Search docs")
        .strict(true)])
        .tool_choice(ToolChoice::Specific { name: "search".into() })
        .response_format(ResponseFormat::json_schema(json!({"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]})).name("answer"))
        .reasoning(ReasoningEffort::High)
        .cache(PromptCache::new().key("session-1"))
        .with_option(
            ChatRequestOptions::new()
                .user("user_123")
                .service_tier("flex")
                .store(true)
                .metadata(serde_json::Map::from_iter([(
                    "trace_id".into(),
                    json!("trace-123"),
                )])),
        );
    let response = provider(&server).chat(&request).await.unwrap();
    assert!(response.warnings.is_empty(), "{:?}", response.warnings);

    let sent = &server.recorded_requests().await[0];
    assert_eq!(sent.path, "/chat/completions");
    assert_eq!(sent.header("authorization"), Some("Bearer test-key"));
    assert_json_fixture_eq(&sent.body_json(), &fixtures(), "request.json");
}

#[tokio::test]
async fn recorded_text_stream_collects_with_identity_and_usage() {
    let server = TestHttpServer::spawn([MockHttpResponse::sse(load_text_fixture(
        &fixtures(),
        "recorded/stream_text.sse",
    ))
    .header("x-request-id", HeaderValue::from_static("req_stream"))])
    .await;
    let response = provider(&server)
        .chat_stream(&ChatRequest::new("gpt-4.1-nano").user("Count from 1 to 12 in words."))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();
    let text = response.text().unwrap().to_lowercase();
    assert!(text.contains("one") && text.contains("twelve"), "{text}");
    assert_eq!(response.finish_reason, Some(FinishReason::Stop));
    assert_eq!(response.finish_reason_raw.as_deref(), Some("stop"));
    assert_eq!(response.provider.as_deref(), Some("openai"));
    assert_eq!(response.request_id.as_deref(), Some("req_stream"));
    assert_eq!(response.model.as_deref(), Some("gpt-4.1-nano-2025-04-14"));
    let usage = response.usage.unwrap();
    assert_eq!(usage.input_tokens, Some(17));
    assert_usage_contract(&usage);
}

#[tokio::test]
async fn recorded_parallel_tool_stream() {
    let server = TestHttpServer::spawn([MockHttpResponse::sse(load_text_fixture(
        &fixtures(),
        "recorded/stream_tools.sse",
    ))])
    .await;
    let response = provider(&server)
        .chat_stream(&ChatRequest::new("gpt-4.1-nano").user("Weather in Paris and Tokyo?"))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();
    let calls: Vec<_> = response.tool_calls().collect();
    assert_eq!(calls.len(), 2);
    assert!(
        calls
            .iter()
            .all(|c| c.name == "get_weather" && c.has_valid_arguments())
    );
    let cities: Vec<String> = calls
        .iter()
        .map(|c| {
            c.parse_arguments::<serde_json::Value>().unwrap()["city"]
                .as_str()
                .unwrap()
                .to_owned()
        })
        .collect();
    assert_eq!(cities, ["Paris", "Tokyo"]);
    assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
}

#[tokio::test]
async fn recorded_reasoning_model_response() {
    let raw = load_text_fixture(&fixtures(), "recorded/response_reasoning_model.json");
    let server = TestHttpServer::spawn([MockHttpResponse::text(StatusCode::OK, raw)
        .header("content-type", HeaderValue::from_static("application/json"))])
    .await;
    let response = provider(&server)
        .chat(&ChatRequest::new("gpt-5-nano").user("Say hello in three words."))
        .await
        .unwrap();
    assert_eq!(response.text().as_deref(), Some("Hello there friend."));
    assert_eq!(
        response.provider_data.get_raw("openai", "service_tier"),
        Some(&json!("default"))
    );
    assert_usage_contract(&response.usage.unwrap());
}

#[tokio::test]
async fn recorded_search_model_citations() {
    let raw = load_text_fixture(&fixtures(), "recorded/response_search_citations.json");
    let server = TestHttpServer::spawn([MockHttpResponse::text(StatusCode::OK, raw)
        .header("content-type", HeaderValue::from_static("application/json"))])
    .await;
    let response = provider(&server)
        .chat(&ChatRequest::new("gpt-5-search-api").user("Latest Rust release?"))
        .await
        .unwrap();
    let allama::ContentBlock::Text(text) = &response.content[0] else {
        panic!("expected text, got {:?}", response.content)
    };
    assert_eq!(text.citations.len(), 1);
    let citation = &text.citations[0];
    assert_eq!(
        citation.source,
        allama::CitationSource::url(
            "https://rust.dev/updates?utm_source=openai",
            Some("Rust Updates — rust.dev".into())
        )
    );
    assert_eq!(
        text.cited_span(citation),
        Some("([rust.dev](https://rust.dev/updates?utm_source=openai))")
    );
    assert!(
        text.provider_data
            .get_raw("openai", "chat_annotations")
            .is_some(),
        "raw annotations are kept for replay"
    );

    // The Chat annotations never reach a Responses request, which
    // shares the `openai` namespace but takes another shape.
    let follow_up = TestHttpServer::spawn([MockHttpResponse::text(
        StatusCode::OK,
        load_text_fixture(&fixtures(), "responses/text.json"),
    )
    .header("content-type", HeaderValue::from_static("application/json"))])
    .await;
    let responses =
        crate::ResponsesProvider::new_with_base_url("test-key", follow_up.url()).unwrap();
    let mut next = ChatRequest::new("gpt-5").user("Latest Rust release?");
    next.push_message(response.to_assistant_message());
    next.push_message(allama::Message::user("Thanks"));
    responses.chat(&next).await.unwrap();
    let body = follow_up.recorded_requests().await[0].body_json();
    let content = &body["input"][1]["content"][0];
    assert_eq!(content["type"], "output_text");
    assert_eq!(content["annotations"], json!([]), "{content}");
}

#[tokio::test]
async fn error_mapping() {
    let cases = [
        (
            401,
            json!({"error": {"message": "Incorrect API key provided", "type": "invalid_request_error", "code": "invalid_api_key"}}),
        ),
        (
            429,
            json!({"error": {"message": "Rate limit reached", "type": "requests", "code": "rate_limit_exceeded"}}),
        ),
        (
            429,
            json!({"error": {"message": "You exceeded your current quota", "type": "insufficient_quota", "code": "insufficient_quota"}}),
        ),
        (
            404,
            json!({"error": {"message": "The model `x` does not exist", "type": "invalid_request_error", "code": "model_not_found"}}),
        ),
        (
            400,
            json!({"error": {"message": "This model's maximum context length is 128000 tokens", "type": "invalid_request_error", "code": "context_length_exceeded"}}),
        ),
    ];
    // The contracts call `chat` and then `chat_stream`, so their failures
    // (all but the rate limit and the missing model) answer twice.
    let responses = cases.iter().enumerate().flat_map(|(i, (status, body))| {
        let response = MockHttpResponse::json_value(StatusCode::from_u16(*status).unwrap(), body)
            .header("x-request-id", HeaderValue::from_static("req_err"))
            .header("retry-after-ms", HeaderValue::from_static("1500"));
        let times = if matches!(i, 1 | 3) { 1 } else { 2 };
        std::iter::repeat_n(response, times)
    });
    let server = TestHttpServer::spawn(responses).await;
    let provider = provider(&server);
    let request = ChatRequest::new("gpt-4o").user("hi");
    let stream = |request| contract::collect_chat_stream(&provider, request);

    contract::assert_auth_error_contract(&provider, &request, stream).await;
    let rate = provider.chat(&request).await.unwrap_err();
    assert!(matches!(rate, Error::RateLimited { .. }));
    assert_eq!(
        rate.retry_after(),
        Some(std::time::Duration::from_millis(1500))
    );
    assert_eq!(rate.request_id(), Some("req_err"));
    contract::assert_quota_exhausted_contract(&provider, &request, stream).await;
    assert!(matches!(
        provider.chat(&request).await.unwrap_err(),
        Error::ModelNotFound { .. }
    ));
    contract::assert_context_length_exceeded_contract(&provider, &request, stream).await;
}

#[tokio::test]
async fn strict_policy_and_option_typos_fail_before_sending() {
    let server = TestHttpServer::spawn([]).await;
    let provider = provider(&server);
    let typo = ChatRequest::new("gpt-4o")
        .user("hi")
        .with_options_json("openai", json!({"servce_tier": "flex"}));
    assert!(matches!(
        provider.chat(&typo).await.unwrap_err(),
        Error::InvalidRequest { .. }
    ));
    let strict = ChatRequest::new("gpt-4o")
        .user("hi")
        .reasoning(allama::ReasoningConfig::new().output(allama::ReasoningOutput::Summary))
        .unsupported_policy(UnsupportedPolicy::Error);
    assert!(matches!(
        provider.chat(&strict).await.unwrap_err(),
        Error::Unsupported { .. }
    ));
    assert!(server.recorded_requests().await.is_empty());
}

#[tokio::test]
async fn embeddings_request_and_response() {
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(
        StatusCode::OK,
        &json!({"data": [{"embedding": [0.1, 0.2], "index": 0}], "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 2, "total_tokens": 2}}),
    )])
    .await;
    let response = provider(&server)
        .embed(
            &EmbeddingRequest::new("text-embedding-3-small")
                .input("hello")
                .dimensions(2)
                .with_option(
                    crate::EmbeddingRequestOptions::new()
                        .user("u")
                        .extra("encoding_format", "float"),
                ),
        )
        .await
        .unwrap();
    assert_eq!(response.embeddings, vec![vec![0.1, 0.2]]);
    let body = server.recorded_requests().await[0].body_json();
    assert_eq!(
        body,
        json!({
            "model": "text-embedding-3-small",
            "input": ["hello"],
            "dimensions": 2,
            "user": "u",
            "encoding_format": "float"
        })
    );
}

/// OpenAI annotation offsets count Unicode code points: with emoji (outside
/// the Basic Multilingual Plane) and accented letters before the citation,
/// the span still lands exactly on the cited link. Recorded with
/// gpt-5-search-api.
#[tokio::test]
async fn recorded_citation_offsets_are_code_points() {
    let raw = load_text_fixture(&fixtures(), "recorded/response_search_citations_emoji.json");
    let server = TestHttpServer::spawn([MockHttpResponse::text(StatusCode::OK, raw)
        .header("content-type", HeaderValue::from_static("application/json"))])
    .await;
    let response = provider(&server)
        .chat(&ChatRequest::new("gpt-5-search-api").user("Latest Rust release?"))
        .await
        .unwrap();
    let allama::ContentBlock::Text(text) = &response.content[0] else {
        panic!("expected text, got {:?}", response.content)
    };
    assert!(text.text.starts_with('🦀'));
    assert_eq!(
        text.cited_span(&text.citations[0]),
        Some("([blog.rust-lang.org](https://blog.rust-lang.org/?utm_source=openai))")
    );
}

/// Search models stream the annotations in a final delta, after the text;
/// they become citations on the streamed text block, as in unary responses.
/// Recorded with gpt-5-search-api.
#[tokio::test]
async fn recorded_search_model_stream_citations() {
    let server = TestHttpServer::spawn([MockHttpResponse::sse(load_text_fixture(
        &fixtures(),
        "recorded/stream_search_citations.sse",
    ))])
    .await;
    let response = provider(&server)
        .chat_stream(&ChatRequest::new("gpt-5-search-api").user("Latest Rust release?"))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();
    let allama::ContentBlock::Text(text) = &response.content[0] else {
        panic!("expected text, got {:?}", response.content)
    };
    assert_eq!(text.citations.len(), 1);
    assert_eq!(
        text.cited_span(&text.citations[0]),
        Some("([rust.dev](https://rust.dev/updates?utm_source=openai))")
    );
    assert_eq!(
        text.provider_data
            .get_raw("openai", "chat_annotations")
            .unwrap()[0]["type"],
        "url_citation"
    );
    assert_eq!(response.finish_reason, Some(FinishReason::Stop));
}

fn sql_tool() -> serde_json::Map<String, serde_json::Value> {
    json!({"type": "custom", "custom": {
        "name": "sql",
        "description": "Run a read-only SQLite query.",
        "format": {"type": "grammar", "grammar": {"syntax": "regex", "definition": "SELECT .+"}}
    }})
    .as_object()
    .cloned()
    .unwrap()
}

/// A custom tool call decodes to the shape the Responses API uses, waits
/// for the client, and its typed answer continues the conversation as a
/// `custom` tool call and a `tool` message (ADR 0014).
#[tokio::test]
async fn custom_tool_calls_round_trip() {
    use crate::ClientToolCall;

    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(
            StatusCode::OK,
            &allama_conformance::load_json_fixture(&fixtures(), "hand_written/custom_tool.json"),
        ),
        MockHttpResponse::json_value(StatusCode::OK, &ok_response()),
    ])
    .await;
    let provider = provider(&server);
    let mut request = ChatRequest::new("gpt-5.1")
        .user("How many users signed up today?")
        .with_option(ChatRequestOptions::new().client_tool(sql_tool()));
    let response = provider.chat(&request).await.unwrap();
    assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
    assert!(
        response.tool_calls().next().is_none(),
        "not a function call"
    );
    let sent = server.recorded_requests().await[0].body_json();
    assert_eq!(sent["tools"], json!([sql_tool()]));

    let pending = ClientToolCall::pending(&response.content);
    let [ClientToolCall::Custom(call)] = pending.as_slice() else {
        panic!("expected one custom call: {:?}", response.content)
    };
    assert_eq!(call.name, "sql");
    assert!(call.input.starts_with("SELECT count(*)"));
    let answer = call.output("42");
    request.push_message(response.to_assistant_message());
    request.push_message(allama::Message::user(vec![answer]));

    let next = provider.chat(&request).await.unwrap();
    assert!(next.warnings.is_empty(), "{:?}", next.warnings);
    let body = server.recorded_requests().await[1].body_json();
    assert_eq!(
        body["messages"],
        json!([
            {"role": "user", "content": "How many users signed up today?"},
            {"role": "assistant", "tool_calls": [{
                "id": "call_Rk3XbQ1mZc7Vn2Pq8Ld5Ws0Y", "type": "custom",
                "custom": {"name": "sql", "input": "SELECT count(*) FROM users WHERE created_at >= date('now')"}
            }]},
            {"role": "tool", "content": "42", "tool_call_id": "call_Rk3XbQ1mZc7Vn2Pq8Ld5Ws0Y"}
        ])
    );
}

/// A streamed custom tool call is never decoded as a function call.
#[tokio::test]
async fn streamed_custom_tool_calls_wait_for_the_client() {
    use crate::ClientToolCall;

    let server = TestHttpServer::spawn([MockHttpResponse::sse(load_text_fixture(
        &fixtures(),
        "hand_written/custom_tool_stream.sse",
    ))])
    .await;
    let response = provider(&server)
        .chat_stream(&ChatRequest::new("gpt-5.1").user("How many users?"))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();
    assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
    assert!(
        response.tool_calls().next().is_none(),
        "not a function call"
    );
    let pending = ClientToolCall::pending(&response.content);
    let [ClientToolCall::Custom(call)] = pending.as_slice() else {
        panic!("expected one custom call: {:?}", response.content)
    };
    assert_eq!(call.call_id, "call_Ht6YmP2qWx9Lc4Rd8Kb1Vn5S");
    assert_eq!(call.input, "SELECT count(*) FROM users");
    assert_eq!(call.origin.as_deref(), Some("openai"));
}

/// Extraction refusals: OpenAI's refusal field
/// (<https://platform.openai.com/docs/guides/structured-outputs#refusals>)
/// finishes with `Refusal`, which fails extraction before parsing on both
/// paths and is never retried.
#[tokio::test]
async fn refusals_fail_extraction_once() {
    use allama::chat::extract::{ChatExtractError, ChatExtractExt, ChatExtractor};
    use allama::schemars::{self, JsonSchema};
    use allama_http::{ReplayResponse, ReplayTransport};

    #[derive(Debug, serde::Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct Invoice {
        total: f64,
    }

    let refusal = "I'm sorry, I can't help with that.";
    // The model wrote the refusal: its text comes from the `Refusal` block.
    let assert_refused = |error: Error| {
        let Error::Extract { failure, .. } = error else {
            panic!("an extraction failure expected: {error:?}");
        };
        assert!(
            matches!(
                failure.error(),
                ChatExtractError::Refused {
                    finish_reason: Some(FinishReason::Refusal),
                    source: allama::chat::extract::RefusalSource::Model,
                    text: Some(text),
                    ..
                } if text == refusal
            ),
            "{failure:?}"
        );
    };
    let transport = ReplayTransport::new([
        ReplayResponse::json(
            StatusCode::OK,
            json!({
                "id": "chatcmpl-1", "model": "gpt-5-mini",
                "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": null, "refusal": refusal}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14}
            })
            .to_string(),
        ),
        ReplayResponse::sse(format!(
            "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
            json!({"id": "c1", "model": "gpt-5-mini", "choices": [{"index": 0, "delta": {"refusal": refusal}}]}),
            json!({"id": "c1", "model": "gpt-5-mini", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        )),
    ]);
    let provider = Provider::builder()
        .api_key("k")
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let request = ChatRequest::new("gpt-5-mini").user("total?");
    let error = ChatExtractor::new()
        .retries(2)
        .extract::<_, Invoice>(&provider, &request)
        .await
        .unwrap_err();
    assert_eq!(transport.requests().len(), 1, "a refusal is never retried");
    assert_refused(error);

    let mut stream = provider.extract_stream::<Invoice>(&request).await.unwrap();
    while futures_util::StreamExt::next(&mut stream).await.is_some() {}
    assert_refused(stream.finish().unwrap_err());
}
