//! Fixture-based conformance tests.
//!
//! `stream.sse`, `stream_signature.sse`, `stream_thoughts.sse`,
//! `gemini3_tool_response_raw.json`, the Google Search grounding recordings
//! `grounding_response_raw.json` and `grounding_stream.sse` (gemini-2.5-flash),
//! and the code-execution recordings `code_execution_gemini3_raw.json`
//! (gemini-3.5-flash) and `code_execution_stream.sse` (gemini-2.5-flash) are recorded from the live API (2026-10-02);
//! `stream_function_call.sse` (gemini-3.5-flash-lite, two parallel calls
//! with ids, a request with a two-part system instruction) on 2026-10-07.
//! The other raw fixtures are hand-written. Set `ALLAMA_UPDATE_FIXTURES=1` to rewrite
//! the expected files, then review every value.

use std::path::PathBuf;

use allama::{
    ChatProvider, ChatRequest, ChatResponse, ChatStreamExt, EmbeddingProvider, EmbeddingRequest,
    EmbeddingResponse, Message, ReasoningConfig, ReasoningEffort, ResponseFormat, ToolChoice,
    ToolDefinition, ValidationCoverage,
};
use allama_conformance::capabilities::UNKNOWN_MODEL;
use allama_conformance::probe::{ProbeOutcome, VectorOrder};
use allama_conformance::stream_end::StreamEnds;
use allama_conformance::stream_errors::ErrorKindProbe;
use allama_conformance::{
    AdapterUnderTest, Contract, EmbeddingAdapterUnderTest, FixtureDir, TransportOutcome,
    assert_chat_adapter, assert_embedding_adapter, assert_error_fixture_eq, assert_json_fixture_eq,
    assert_partial_stream_fixture_eq, assert_response_fixture_eq,
    assert_stream_finish_error_fixture_eq, assert_stream_fixture_eq, assert_usage_contract,
    load_json_fixture, load_text_fixture,
};
use allama_http::{CredentialPlacement, CredentialSlot, ReplayResponse, ReplayTransport};
use serde_json::{Value, json};

use crate::ChatRequestOptions;

fn fixtures() -> FixtureDir {
    FixtureDir::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"))
}

fn decode_fixture(name: &str) -> allama::ChatResponse {
    let raw = load_json_fixture(&fixtures(), name);
    crate::decode::decode_response(serde_json::from_value(raw).unwrap(), Vec::new(), None).unwrap()
}

#[test]
fn request_fixture_matches() {
    let request = ChatRequest::new("gemini-2.5-pro")
        .system("You are a precise assistant.")
        .message(Message::user("Find rust docs"))
        .tools(vec![
            ToolDefinition::new(
                "search",
                json!({
                    "type": "object",
                    "properties": {"q": {"type": "string"}, "filter": {"$ref": "#/$defs/Filter"}},
                    "required": ["q"],
                    "additionalProperties": false,
                    "$defs": {"Filter": {"type": "object", "properties": {"site": {"type": "string"}}}}
                }),
            )
            .description("Search docs"),
        ])
        .tool_choice(ToolChoice::Specific {
            name: "search".into(),
        })
        .temperature(0.4)
        .top_p(0.8)
        .top_k(7)
        .max_output_tokens(256)
        .stop(["END"])
        .seed(7)
        .response_format(ResponseFormat::json_schema(json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": false
        })))
        .reasoning(ReasoningConfig::new().effort(ReasoningEffort::Low))
        .with_option(ChatRequestOptions {
            cached_content: Some("cachedContents/abc".into()),
            ..Default::default()
        });

    let actual = crate::request::conformance_to_api_request(&request).unwrap();
    assert_json_fixture_eq(&actual, &fixtures(), "request.json");
}

#[test]
fn response_fixture_matches() {
    let response = decode_fixture("response_raw.json");
    assert_usage_contract(response.usage.as_ref().unwrap());
    assert_response_fixture_eq(&response, &fixtures(), "response_expected.json");
}

#[test]
fn recorded_gemini3_parallel_tool_response_matches() {
    let response = decode_fixture("gemini3_tool_response_raw.json");
    assert_usage_contract(response.usage.as_ref().unwrap());
    assert_response_fixture_eq(
        &response,
        &fixtures(),
        "gemini3_tool_response_expected.json",
    );
}

#[test]
fn recorded_grounding_becomes_citations() {
    let response = decode_fixture("grounding_response_raw.json");
    assert_response_fixture_eq(&response, &fixtures(), "grounding_response_expected.json");
    assert_grounded(&response);
}

#[tokio::test]
async fn recorded_grounding_stream_matches() {
    let sse = load_text_fixture(&fixtures(), "grounding_stream.sse");
    assert_stream_fixture_eq(
        crate::conformance_stream_from_sse_text(&sse),
        &fixtures(),
        "grounding_stream_events.json",
        "grounding_stream_response_expected.json",
    )
    .await;
    let response: allama::ChatResponse = serde_json::from_value(load_json_fixture(
        &fixtures(),
        "grounding_stream_response_expected.json",
    ))
    .unwrap();
    assert_grounded(&response);
}

/// Both recordings (gemini-2.5-flash with Google Search) ground the second
/// half of a one-sentence answer in web sources.
fn assert_grounded(response: &allama::ChatResponse) {
    let allama::ContentBlock::Text(text) = &response.content[0] else {
        panic!("expected answer text, got {:?}", response.content)
    };
    assert!(text.citations.len() >= 2, "{:?}", text.citations);
    for citation in &text.citations {
        let allama::CitationSource::Url { url, title, .. } = &citation.source else {
            panic!("expected web sources, got {:?}", citation.source)
        };
        assert!(url.starts_with("https://"));
        assert!(title.is_some());
        assert_eq!(
            text.cited_span(citation),
            Some("0, which was announced on October 1, 2026.")
        );
    }
    assert!(
        response
            .provider_data
            .get_raw("google-ai", "grounding_metadata")
            .is_some(),
        "raw grounding is kept"
    );
    assert!(response.warnings.is_empty(), "{:?}", response.warnings);
}

#[test]
fn builtin_tools_are_sent_after_function_declarations() {
    let request = ChatRequest::new("gemini-2.5-flash")
        .user("Latest Rust release?")
        .tools([ToolDefinition::new("lookup", json!({"type": "object"}))])
        .with_option(ChatRequestOptions {
            builtin_tools: vec![serde_json::from_value(json!({"google_search": {}})).unwrap()],
            ..Default::default()
        });
    let body = crate::request::conformance_to_api_request(&request).unwrap();
    assert_eq!(
        body["tools"][0]["functionDeclarations"][0]["name"],
        "lookup"
    );
    assert_eq!(body["tools"][1], json!({"google_search": {}}));

    let builtin_only = ChatRequest::new("gemini-2.5-flash")
        .user("Latest Rust release?")
        .with_options_json(
            "google-ai",
            json!({"builtin_tools": [{"google_search": {}}]}),
        );
    let body = crate::request::conformance_to_api_request(&builtin_only).unwrap();
    assert_eq!(body["tools"], json!([{"google_search": {}}]));
}

#[tokio::test]
async fn recorded_multi_chunk_stream_matches() {
    let sse = load_text_fixture(&fixtures(), "stream.sse");
    assert_stream_fixture_eq(
        crate::conformance_stream_from_sse_text(&sse),
        &fixtures(),
        "stream_events.json",
        "stream_response_expected.json",
    )
    .await;
}

#[tokio::test]
async fn recorded_signature_stream_matches() {
    let sse = load_text_fixture(&fixtures(), "stream_signature.sse");
    assert_stream_fixture_eq(
        crate::conformance_stream_from_sse_text(&sse),
        &fixtures(),
        "stream_signature_events.json",
        "stream_signature_response_expected.json",
    )
    .await;
}

/// A recorded stream of parallel `functionCall` parts, the first
/// signed, ending with an empty text part.
#[tokio::test]
async fn recorded_function_call_stream_matches() {
    let sse = load_text_fixture(&fixtures(), "stream_function_call.sse");
    assert_stream_fixture_eq(
        crate::conformance_stream_from_sse_text(&sse),
        &fixtures(),
        "stream_function_call_events.json",
        "stream_function_call_response_expected.json",
    )
    .await;
    let response = crate::conformance_stream_from_sse_text(&sse)
        .collect_response()
        .await
        .unwrap();
    let calls: Vec<_> = response.tool_calls().collect();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].id, "call_112681");
    assert_eq!(calls[1].arguments, r#"{"city":"Rome"}"#);
    assert_eq!(
        response.finish_reason,
        Some(allama::FinishReason::ToolCalls)
    );
}

#[tokio::test]
async fn recorded_thought_stream_matches() {
    let sse = load_text_fixture(&fixtures(), "stream_thoughts.sse");
    assert_stream_fixture_eq(
        crate::conformance_stream_from_sse_text(&sse),
        &fixtures(),
        "stream_thoughts_events.json",
        "stream_thoughts_response_expected.json",
    )
    .await;
}

#[tokio::test]
async fn truncated_stream_fails_incomplete() {
    let sse = load_text_fixture(&fixtures(), "stream_truncated.sse");
    assert_stream_finish_error_fixture_eq(
        crate::conformance_stream_from_sse_text(&sse),
        &fixtures(),
        "stream_truncated_events.json",
        "stream_truncated_error.json",
    )
    .await;
    assert_partial_stream_fixture_eq(
        crate::conformance_stream_from_sse_text(&sse),
        &fixtures(),
        "stream_truncated_events.json",
        "stream_truncated_partial_response.json",
        "stream_truncated_completeness.json",
    )
    .await;
}

fn error_body(code: u16, message: &str, status: &str) -> String {
    json!({"error": {"code": code, "message": message, "status": status}}).to_string()
}

#[test]
fn error_fixtures_match() {
    use crate::error::conformance_map_http_error as map;
    let fixtures = fixtures();
    let auth = map(
        400,
        &json!({"error": {
            "code": 400,
            "message": "API key not valid. Please pass a valid API key.",
            "status": "INVALID_ARGUMENT",
            "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID", "domain": "googleapis.com"}]
        }})
        .to_string(),
        None,
    );
    assert_error_fixture_eq(&auth, &fixtures, "error_auth.json");

    let rate_limited = map(
        429,
        &json!({"error": {
            "code": 429,
            "message": "Resource has been exhausted (e.g. check quota).",
            "status": "RESOURCE_EXHAUSTED",
            "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "30s"}]
        }})
        .to_string(),
        None,
    );
    assert_error_fixture_eq(&rate_limited, &fixtures, "error_rate_limited.json");

    let quota = map(
        402,
        &error_body(
            402,
            "Your prepayment credits are depleted.",
            "RESOURCE_EXHAUSTED",
        ),
        None,
    );
    assert_error_fixture_eq(&quota, &fixtures, "error_quota_exhausted.json");

    let context = map(
        400,
        &error_body(
            400,
            "The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).",
            "INVALID_ARGUMENT",
        ),
        None,
    );
    assert_error_fixture_eq(&context, &fixtures, "error_context_length.json");

    let model_not_found = map(
        404,
        &error_body(
            404,
            "models/gemini-9-nope is not found for API version v1beta, or is not supported for generateContent.",
            "NOT_FOUND",
        ),
        None,
    );
    assert_error_fixture_eq(&model_not_found, &fixtures, "error_model_not_found.json");

    let timeout = crate::error::conformance_map_timeout_transport_error();
    assert_error_fixture_eq(&timeout, &fixtures, "error_timeout.json");

    // Bodies recorded live on 2026-10-07. An expired
    // cache or file is a request problem, not a credential one.
    let cached_content = map(
        403,
        &error_body(
            403,
            "CachedContent not found (or permission denied)",
            "PERMISSION_DENIED",
        ),
        None,
    );
    assert_error_fixture_eq(&cached_content, &fixtures, "error_cached_content.json");

    let file = map(
        403,
        &error_body(
            403,
            "You do not have permission to access the File doesnotexist123 or it may not exist.",
            "PERMISSION_DENIED",
        ),
        None,
    );
    assert_error_fixture_eq(&file, &fixtures, "error_expired_file.json");

    // Gemini 3.1 Pro refusing `thinkingLevel: MINIMAL`, recorded live.
    let thinking_level = map(
        400,
        &error_body(
            400,
            "Thinking level MINIMAL is not supported for this model. Please retry with other thinking level.",
            "INVALID_ARGUMENT",
        ),
        None,
    );
    assert_error_fixture_eq(&thinking_level, &fixtures, "error_thinking_level.json");
}

#[test]
fn embed_request_fixture_matches() {
    use allama::EmbeddingRequest;
    use allama_conformance::assert_embedding_request_fixture_eq;

    let request = EmbeddingRequest::new("gemini-embedding-001")
        .inputs(["hello world", "gemini embeddings"])
        .dimensions(256);
    assert_embedding_request_fixture_eq(&request, &fixtures(), "embed_request.json");
    let wire =
        crate::embedding::to_api_request(&request, &crate::EmbeddingRequestOptions::default())
            .unwrap()
            .0;
    assert_json_fixture_eq(&wire, &fixtures(), "embed_request_wire.json");
}

/// The portable input type maps to `taskType`: `RETRIEVAL_QUERY` for
/// queries, `RETRIEVAL_DOCUMENT` for documents.
#[test]
fn embed_input_type_fixtures_match() {
    use allama::{EmbeddingInputType, EmbeddingRequest};
    use allama_conformance::assert_embedding_request_fixture_eq;

    for (input_type, name) in [
        (EmbeddingInputType::Query, "query"),
        (EmbeddingInputType::Document, "document"),
    ] {
        let request = EmbeddingRequest::new("gemini-embedding-001")
            .inputs(["hello world", "gemini embeddings"])
            .input_type(input_type);
        assert_embedding_request_fixture_eq(
            &request,
            &fixtures(),
            &format!("embed_input_type_{name}_request.json"),
        );
        let (wire, warnings) =
            crate::embedding::to_api_request(&request, &crate::EmbeddingRequestOptions::default())
                .unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_json_fixture_eq(
            &wire,
            &fixtures(),
            &format!("embed_input_type_{name}_wire.json"),
        );
    }
}

#[test]
fn embed_response_fixture_matches() {
    use allama_conformance::assert_embedding_response_fixture_eq;

    let raw = load_json_fixture(&fixtures(), "embed_response_raw.json");
    let wire: crate::embedding::BatchEmbedContentsResponse = serde_json::from_value(raw).unwrap();
    let converted = crate::embedding::from_api_response(wire, "gemini-embedding-001", 2).unwrap();
    assert_embedding_response_fixture_eq(&converted, &fixtures(), "embed_response_expected.json");
}

#[test]
fn embed_error_fixtures_match() {
    use crate::error::conformance_map_http_error as map;
    use allama_conformance::assert_embedding_error_fixture_eq;

    let fixtures = fixtures();
    let auth = map(
        403,
        &error_body(403, "Permission denied", "PERMISSION_DENIED"),
        None,
    );
    assert_embedding_error_fixture_eq(&auth, &fixtures, "embed_error_auth.json");

    let rate_limited = map(
        429,
        &error_body(429, "Rate limit exceeded", "RESOURCE_EXHAUSTED"),
        Some(std::time::Duration::from_secs(30)),
    );
    assert_embedding_error_fixture_eq(&rate_limited, &fixtures, "embed_error_rate_limited.json");

    let context = map(
        400,
        &error_body(
            400,
            "The input token count (3000) exceeds the maximum number of tokens allowed (2048).",
            "INVALID_ARGUMENT",
        ),
        None,
    );
    assert_embedding_error_fixture_eq(&context, &fixtures, "embed_error_context_length.json");

    let model_not_found = map(
        404,
        &error_body(404, "models/text-embedding-9999 is not found", "NOT_FOUND"),
        None,
    );
    assert_embedding_error_fixture_eq(
        &model_not_found,
        &fixtures,
        "embed_error_model_not_found.json",
    );

    let timeout = crate::error::conformance_map_timeout_transport_error();
    assert_embedding_error_fixture_eq(&timeout, &fixtures, "embed_error_timeout.json");
}

/// Code execution (Gemini 3, recorded): the code and its outcome become
/// provider tool blocks, and the turn replays as the original parts, thought
/// signatures included.
#[test]
fn recorded_code_execution_round_trips() {
    let raw = load_json_fixture(&fixtures(), "code_execution_gemini3_raw.json");
    let response = decode_fixture("code_execution_gemini3_raw.json");
    assert_response_fixture_eq(
        &response,
        &fixtures(),
        "code_execution_gemini3_expected.json",
    );
    let call = response.content[0].as_provider_tool_call().unwrap();
    assert_eq!(call.name, "code_execution");
    let result = response.content[1].as_provider_tool_result().unwrap();
    assert_eq!(result.call_id, call.id);
    assert!(!result.is_error);
    assert!(!response.has_tool_calls());

    let request = ChatRequest::new("gemini-3.5-flash")
        .user("What is the sum of the first 20 prime numbers? Compute it with code.")
        .message(response.to_assistant_message())
        .user("Thanks!");
    let body = crate::request::conformance_to_api_request(&request).unwrap();
    assert_eq!(
        body["contents"][1]["parts"], raw["candidates"][0]["content"]["parts"],
        "the model turn replays verbatim"
    );
}

#[tokio::test]
async fn recorded_code_execution_stream_matches() {
    let sse = load_text_fixture(&fixtures(), "code_execution_stream.sse");
    assert_stream_fixture_eq(
        crate::conformance_stream_from_sse_text(&sse),
        &fixtures(),
        "code_execution_stream_events.json",
        "code_execution_stream_response_expected.json",
    )
    .await;
    let response: allama::ChatResponse = serde_json::from_value(load_json_fixture(
        &fixtures(),
        "code_execution_stream_response_expected.json",
    ))
    .unwrap();
    let call = response
        .content
        .iter()
        .find_map(allama::ContentBlock::as_provider_tool_call)
        .expect("a code-execution call");
    let result = response
        .content
        .iter()
        .find_map(allama::ContentBlock::as_provider_tool_result)
        .expect("its result");
    assert_eq!(result.call_id, call.id);
}

/// Gemini's `google.rpc.Status` for each probed kind, told apart by RPC
/// status, `ErrorInfo` reason, and message, with the upstream text after
/// the message. An invalid argument also names the field it violates; its
/// description is upstream text too.
fn error_envelope(kind: ErrorKindProbe, upstream: &str) -> Option<String> {
    let (code, status, message, reason) = match kind {
        ErrorKindProbe::RateLimited => (
            429,
            "RESOURCE_EXHAUSTED",
            "Resource has been exhausted (e.g. check quota).",
            None,
        ),
        ErrorKindProbe::Overloaded => (503, "UNAVAILABLE", "The model is overloaded.", None),
        ErrorKindProbe::InvalidRequest => (
            400,
            "INVALID_ARGUMENT",
            "Invalid value at 'generation_config.temperature'",
            None,
        ),
        ErrorKindProbe::Auth => (
            400,
            "INVALID_ARGUMENT",
            "API key not valid. Please pass a valid API key.",
            Some("API_KEY_INVALID"),
        ),
        ErrorKindProbe::QuotaExhausted => (
            429,
            "RESOURCE_EXHAUSTED",
            "Your project has exceeded its monthly spending cap.",
            None,
        ),
        ErrorKindProbe::ContextLength => (
            400,
            "INVALID_ARGUMENT",
            "The input token count (2000000) exceeds the maximum number of tokens allowed (1048576).",
            None,
        ),
        // https://ai.google.dev/gemini-api/docs/troubleshooting: `NOT_FOUND`
        // (404) and `DEADLINE_EXCEEDED` (504).
        ErrorKindProbe::ModelNotFound => (
            404,
            "NOT_FOUND",
            "models/gemini-9-nope is not found for API version v1beta",
            None,
        ),
        ErrorKindProbe::Timeout => (
            504,
            "DEADLINE_EXCEEDED",
            "Deadline expired before operation could complete.",
            None,
        ),
        // A feature the model lacks (see `error::not_enabled`).
        ErrorKindProbe::Unsupported => (
            400,
            "INVALID_ARGUMENT",
            "Function calling is not enabled for models/gemma-3-27b-it",
            None,
        ),
        _ => return None,
    };
    let message = format!("{message} {upstream}");
    let message = message.trim_end();
    let mut details: Vec<_> = reason
        .map(
            |reason| json!({"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": reason}),
        )
        .into_iter()
        .collect();
    if kind == ErrorKindProbe::InvalidRequest {
        details.push(json!({
            "@type": "type.googleapis.com/google.rpc.BadRequest",
            "fieldViolations": [{"field": "generation_config.temperature", "description": message}]
        }));
    }
    Some(
        json!({"error": {"code": code, "message": message, "status": status, "details": details}})
            .to_string(),
    )
}

/// An `error` object after the stream started.
fn error_stream(envelope: &str) -> String {
    let chunk = json!({
        "candidates": [{"content": {"role": "model", "parts": [{"text": "Hel"}]}, "index": 0}],
        "modelVersion": "gemini-2.5-flash"
    });
    format!("data: {chunk}\n\ndata: {envelope}\n\n")
}

/// The embedding path maps a malformed body like `chat` (which the
/// aggregate checks).
#[tokio::test]
async fn malformed_embedding_bodies_are_unexpected_responses() {
    use allama::{EmbeddingProvider, EmbeddingRequest, Error};
    use allama_conformance::contract::MALFORMED_BODY;
    // Gemini sends no request id; a proxy's is kept.
    let malformed = ReplayTransport::new([
        ReplayResponse::new(200, MALFORMED_BODY).header("x-request-id", "req_1")
    ]);
    let err = provider(&malformed)
        .embed(&EmbeddingRequest::new("gemini-embedding-001").input("hi"))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::UnexpectedResponse { .. }), "{err:?}");
    assert_eq!(err.status(), Some(200));
    assert_eq!(err.request_id(), Some("req_1"));
    assert_eq!(err.context().unwrap().provider.as_deref(), Some("gemini"));
}

/// Current, thinking, and older model ids.
const MODELS: &[&str] = &[
    "gemini-2.5-flash",
    "gemini-3-pro-preview",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
];

/// The text and multimodal embedding models, and the legacy one.
const EMBEDDING_MODELS: &[&str] = &[
    "gemini-embedding-001",
    "gemini-embedding-2",
    "embedding-001",
];

fn provider(transport: &ReplayTransport) -> crate::Provider {
    crate::Provider::builder()
        .api_key("test-key")
        .http_transport(transport.clone())
        .build()
        .unwrap()
}

fn bodies(transport: &ReplayTransport) -> Vec<Value> {
    transport
        .requests()
        .iter()
        .filter_map(|r| r.body.as_ref())
        .map(|body| serde_json::from_slice(body).expect("JSON request body"))
        .collect()
}

async fn chat_probe(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
    let ok = ReplayResponse::json(
        200,
        json!({
            "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]},
                            "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
            "modelVersion": "gemini-2.5-flash"
        })
        .to_string(),
    );
    let transport = ReplayTransport::new([ok]);
    let result = provider(&transport).chat(&request).await;
    ProbeOutcome::new(result, bodies(&transport))
}

/// Answers with a captured stream (`fixtures/stream.sse`), whose chunks
/// carry `usageMetadata`.
async fn chat_stream_probe(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
    let transport =
        ReplayTransport::new([ReplayResponse::sse(include_str!("../fixtures/stream.sse"))]);
    let result = match provider(&transport).chat_stream(&request).await {
        Ok(stream) => stream.collect_response().await,
        Err(err) => Err(err),
    };
    ProbeOutcome::new(result, bodies(&transport))
}

async fn embed_probe(request: EmbeddingRequest) -> ProbeOutcome<EmbeddingResponse> {
    let embeddings: Vec<Value> = (0..request.inputs.len())
        .map(|_| json!({"values": [0.1, 0.2]}))
        .collect();
    let ok = ReplayResponse::json(200, json!({"embeddings": embeddings}).to_string());
    let transport = ReplayTransport::new([ok]);
    let result = provider(&transport).embed(&request).await;
    ProbeOutcome::new(result, bodies(&transport))
}

fn map_http_error(status: u16, headers: &[(&str, &str)], body: String) -> allama::Error {
    crate::error::map_http_error(status, &headers.iter().copied().collect(), &body)
}

/// The slot every Gemini provider declares.
fn slot() -> CredentialSlot {
    CredentialSlot::new(
        crate::DEFAULT_CREDENTIAL_SCOPE,
        CredentialPlacement::Header("x-goog-api-key".into()),
    )
}

/// One call with transport options attached: `chat`, `chat_stream`, or
/// `embed`.
async fn send(options: allama::RequestOptions, call: &str) -> TransportOutcome {
    let transport = ReplayTransport::default();
    let provider = provider(&transport);
    let mut chat = ChatRequest::new("gemini-2.5-flash").user("Hi");
    chat.options = options.clone();
    let mut embed = EmbeddingRequest::new("gemini-embedding-001").inputs(["hello"]);
    embed.options = options;
    let result = match call {
        "chat" => provider.chat(&chat).await.map(drop),
        "chat_stream" => provider.chat_stream(&chat).await.map(drop),
        _ => provider.embed(&embed).await.map(drop),
    };
    TransportOutcome::new(result, transport.requests())
}

/// Every chat contract, for every model kind and for one Gemini does not
/// know. Validation is a dry run of the encoding (`Encoding` coverage).
#[tokio::test]
async fn the_chat_adapter_passes_every_contract() {
    let gemini = provider(&ReplayTransport::new([]));
    for model in MODELS.iter().chain(&[UNKNOWN_MODEL]) {
        let validation = gemini
            .validate_chat_request(&ChatRequest::new(*model).user("Hi"))
            .unwrap();
        assert_eq!(validation.coverage(), ValidationCoverage::Encoding);
    }
    // Gemini has no terminal marker apart from the `finishReason` on the
    // last chunk, so a stream whose last chunk has none is `Incomplete`, and
    // there is no "marker without a reason" case.
    let chunk = json!({
        "candidates": [{"content": {"role": "model", "parts": [{"text": "Hel"}]}, "index": 0}],
        "modelVersion": "gemini-2.5-flash"
    });
    let under_test = AdapterUnderTest::new(
        MODELS.iter().copied(),
        crate::NAMESPACE,
        chat_probe,
        chat_stream_probe,
    )
    .error_mapping(map_http_error)
    .error_envelope(error_envelope)
    .stream_body(error_stream)
    .stream_replay(|body| async move {
        provider(&ReplayTransport::new([ReplayResponse::sse(body)]))
            .chat_stream(&ChatRequest::new("gemini-2.5-flash").user("hi"))
            .await
    })
    .stream_ends(StreamEnds::new(format!("data: {chunk}\n\n")))
    .sse_captures(
        [
            "stream.sse",
            "stream_signature.sse",
            "stream_thoughts.sse",
            "grounding_stream.sse",
            "code_execution_stream.sse",
        ]
        .map(|name| load_text_fixture(&fixtures(), name)),
    )
    .transport_options(
        slot(),
        |options| send(options, "chat"),
        |options| send(options, "chat_stream"),
    )
    // Gemini sends no request id; a proxy's is kept.
    .malformed_response(
        |body| async move {
            let answer = ReplayResponse::new(200, body).header("x-request-id", "req_1");
            provider(&ReplayTransport::new([answer]))
                .chat(&ChatRequest::new("gemini-2.5-flash").user("hi"))
                .await
        },
        Some("req_1"),
    )
    // Every listed model constrains a JSON Schema response format, so no
    // configuration lacks strict mode (strict tools are reported as
    // approximated, which the request field audit checks). No model answers
    // `ReasoningReplay` `Unsupported`: what a model lacks is model data,
    // which belongs to the catalog (ADR 0015).
    .skip([
        Contract::StrictNotHonouredWarns,
        Contract::UnreplayableReasoningDropped,
    ])
    // The decoding contracts: each pair is one response as
    // `generateContent` and `streamGenerateContent` send it. The tool calls
    // carry wire ids (`functionCall.id`), which the adapter keeps.
    .unary_replay(|body| async move {
        provider(&ReplayTransport::new([ReplayResponse::json(200, body)]))
            .chat(&ChatRequest::new("gemini-3.5-flash").user("hi"))
            .await
    })
    .response_captures([capture("agree_text")])
    .tool_call_capture(
        capture("agree_tool_calls").wire_tool_call_ids(["call_734101", "call_734102"]),
    )
    .length_capture(capture("agree_length"));
    #[cfg(feature = "reqwest")]
    let under_test = under_test.redirect_not_followed(|base_url, streaming| async move {
        let provider = crate::Provider::builder()
            .api_key("test-key")
            .base_url(base_url)
            .build()?;
        let request = ChatRequest::new("gemini-2.5-flash").user("hi");
        if streaming {
            provider.chat_stream(&request).await.map(drop)
        } else {
            provider.chat(&request).await.map(drop)
        }
    });
    // Without the default client there is no redirect policy to check.
    #[cfg(not(feature = "reqwest"))]
    let under_test = under_test.skip([Contract::RedirectNotFollowed]);
    assert_chat_adapter(&gemini, &under_test).await;
}

/// `fixtures/<name>.json` and `fixtures/<name>.sse`: one response as the
/// Gemini API sends it unary and streamed.
fn capture(name: &str) -> allama_conformance::ResponseCapture {
    allama_conformance::ResponseCapture::new(
        load_text_fixture(&fixtures(), &format!("{name}.json")),
        load_text_fixture(&fixtures(), &format!("{name}.sse")),
    )
}

/// Every embedding contract, for the text and multimodal models, the
/// legacy one, and a model id the adapter does not know.
#[tokio::test]
async fn the_embedding_adapter_passes_every_contract() {
    let gemini = provider(&ReplayTransport::new([]));
    for model in EMBEDDING_MODELS.iter().chain(&[UNKNOWN_MODEL]) {
        let validation = gemini
            .validate_embedding_request(&EmbeddingRequest::new(*model).input("Hi"))
            .unwrap();
        assert_eq!(validation.coverage(), ValidationCoverage::Encoding);
    }
    // Gemini checks the image count and the task type option while
    // encoding.
    let validation_requests = EMBEDDING_MODELS
        .iter()
        .chain(&[UNKNOWN_MODEL])
        .flat_map(|model| {
            [
                EmbeddingRequest::new(*model).input(allama::EmbeddingInput::from(
                    vec![allama::EmbeddingPart::image_url("https://example.com/a.png"); 7],
                )),
                EmbeddingRequest::new(*model)
                    .input("Hi")
                    .input_type(allama::EmbeddingInputType::Query)
                    .with_option(crate::EmbeddingRequestOptions {
                        task_type: Some("CLASSIFICATION".into()),
                        ..Default::default()
                    }),
            ]
        });
    let under_test = EmbeddingAdapterUnderTest::new(
        EMBEDDING_MODELS.iter().copied(),
        crate::NAMESPACE,
        embed_probe,
    )
    .validation_requests(validation_requests)
    .answering(VectorOrder::InputOrder, |request, vectors| async move {
        let embeddings: Vec<Value> = vectors
            .iter()
            .map(|vector| json!({"values": vector.values}))
            .collect();
        let body = json!({"embeddings": embeddings}).to_string();
        provider(&ReplayTransport::new([ReplayResponse::json(200, body)]))
            .embed(&request)
            .await
    })
    .error_mapping(map_http_error)
    .error_envelope(error_envelope)
    .transport_options(slot(), |options| send(options, "embed"))
    // The wire carries a task type and several parts for every model, and
    // what a model lacks is model data, which belongs to the catalog (ADR
    // 0015): no model answers `InputType` or `CompositeInput` `Unsupported`.
    .skip([
        Contract::InputTypeNotHonouredWarns,
        Contract::MultiPartInputRefused,
    ]);
    assert_embedding_adapter(&gemini, &under_test).await;
}
