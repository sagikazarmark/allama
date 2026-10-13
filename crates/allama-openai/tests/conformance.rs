//! Every probe-based contract, through the aggregates: Chat Completions and
//! Responses (`assert_chat_adapter`) and embeddings
//! (`assert_embedding_adapter`), for every model kind and for one OpenAI
//! does not know.

use allama::prelude::*;
use allama::{EmbeddingResponse, RequestOptions, ValidationCoverage};
use allama_conformance::capabilities::UNKNOWN_MODEL;
use allama_conformance::probe::{ProbeOutcome, VectorOrder};
use allama_conformance::stream_end::StreamEnds;
use allama_conformance::stream_errors::ErrorKindProbe;
use allama_conformance::{
    AdapterUnderTest, Contract, EmbeddingAdapterUnderTest, ResponseCapture, TransportOutcome,
    assert_chat_adapter, assert_embedding_adapter,
};
use allama_http::{CredentialPlacement, CredentialSlot, Headers, ReplayResponse, ReplayTransport};
use allama_openai::{Provider, ResponsesProvider};
use allama_openai_compat::CompatQuirks;
use serde_json::{Value, json};

/// Current, reasoning, and legacy chat models.
const MODELS: &[&str] = &["gpt-4.1", "gpt-5-mini", "o4-mini", "o3", "gpt-3.5-turbo"];

/// Embedding models: the symmetric ones this crate knows, and one it does
/// not (whose input type is a hint).
const EMBEDDING_MODELS: &[&str] = &[
    "text-embedding-3-small",
    "text-embedding-ada-002",
    "text-embedding-4-large",
];

fn chat_ok() -> ReplayResponse {
    ReplayResponse::json(
        200,
        json!({
            "id": "chatcmpl-1", "object": "chat.completion", "model": "gpt-4.1",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })
        .to_string(),
    )
}

fn responses_ok() -> ReplayResponse {
    ReplayResponse::json(
        200,
        json!({
            "id": "resp_1", "object": "response", "status": "completed", "model": "gpt-4.1",
            "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "ok", "annotations": []}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        })
        .to_string(),
    )
}

fn bodies(transport: &ReplayTransport) -> Vec<Value> {
    transport
        .requests()
        .iter()
        .filter_map(|r| r.body.as_ref())
        .map(|body| serde_json::from_slice(body).expect("JSON request body"))
        .collect()
}

fn provider(transport: &ReplayTransport) -> Provider {
    Provider::builder()
        .api_key("k")
        .http_transport(transport.clone())
        .build()
        .unwrap()
}

fn responses_provider(transport: &ReplayTransport) -> ResponsesProvider {
    ResponsesProvider::builder()
        .api_key("k")
        .http_transport(transport.clone())
        .build()
        .unwrap()
}

async fn chat_probe(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
    let transport = ReplayTransport::new([chat_ok()]);
    let result = provider(&transport).chat(&request).await;
    ProbeOutcome::new(result, bodies(&transport))
}

/// Answers with a captured stream whose final chunk carries usage.
async fn chat_stream_probe(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
    let transport = ReplayTransport::new([ReplayResponse::sse(CHAT_STREAM)]);
    let result = match provider(&transport).chat_stream(&request).await {
        Ok(stream) => stream.collect_response().await,
        Err(err) => Err(err),
    };
    ProbeOutcome::new(result, bodies(&transport))
}

async fn responses_probe(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
    let transport = ReplayTransport::new([responses_ok()]);
    let result = responses_provider(&transport).chat(&request).await;
    ProbeOutcome::new(result, bodies(&transport))
}

/// Answers with a captured stream whose `response.completed` carries usage.
async fn responses_stream_probe(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
    let transport = ReplayTransport::new([ReplayResponse::sse(RESPONSES_STREAM)]);
    let result = match responses_provider(&transport).chat_stream(&request).await {
        Ok(stream) => stream.collect_response().await,
        Err(err) => Err(err),
    };
    ProbeOutcome::new(result, bodies(&transport))
}

async fn embed_probe(request: EmbeddingRequest) -> ProbeOutcome<EmbeddingResponse> {
    let data: Vec<Value> = (0..request.inputs.len())
        .map(|i| json!({"object": "embedding", "index": i, "embedding": [0.1, 0.2]}))
        .collect();
    let body = json!({"object": "list", "model": "text-embedding-3-small", "data": data,
                      "usage": {"prompt_tokens": 1, "total_tokens": 1}});
    let transport = ReplayTransport::new([ReplayResponse::json(200, body.to_string())]);
    let result = provider(&transport).embed(&request).await;
    ProbeOutcome::new(result, bodies(&transport))
}

const CHAT_STREAM: &str = include_str!("../fixtures/recorded/stream_text.sse");
const RESPONSES_STREAM: &str = include_str!("../fixtures/responses/text_stream.sse");

/// OpenAI's error object for each probed kind, told apart by `code`: a rate
/// limit and exhausted quota are both `429`s.
fn error_object(kind: ErrorKindProbe) -> Option<Value> {
    let (message, error_type, code) = match kind {
        ErrorKindProbe::RateLimited => (
            "Rate limit reached for gpt-4o",
            "requests",
            "rate_limit_exceeded",
        ),
        ErrorKindProbe::Overloaded => (
            "The server is overloaded",
            "server_error",
            "server_is_overloaded",
        ),
        ErrorKindProbe::InvalidRequest => (
            "Invalid value for 'temperature'",
            "invalid_request_error",
            "invalid_value",
        ),
        ErrorKindProbe::Auth => (
            "Incorrect API key provided",
            "invalid_request_error",
            "invalid_api_key",
        ),
        ErrorKindProbe::QuotaExhausted => (
            "You exceeded your current quota",
            "insufficient_quota",
            "insufficient_quota",
        ),
        ErrorKindProbe::ContextLength => (
            "This model's maximum context length is 128000 tokens",
            "invalid_request_error",
            "context_length_exceeded",
        ),
        ErrorKindProbe::ModelNotFound => (
            "The model `gpt-nope` does not exist or you do not have access to it.",
            "invalid_request_error",
            "model_not_found",
        ),
        // A Responses error code
        // (https://platform.openai.com/docs/api-reference/responses/object).
        ErrorKindProbe::Timeout => (
            "The vector store search timed out",
            "server_error",
            "vector_store_timeout",
        ),
        // A reasoning model refusing a sampling parameter: the provider's
        // own refusal (`UnsupportedStage::Provider`).
        ErrorKindProbe::Unsupported => {
            return Some(json!({
                "message": "Unsupported parameter: 'temperature' is not supported with this model.",
                "type": "invalid_request_error",
                "code": "unsupported_parameter",
                "param": "temperature"
            }));
        }
        _ => return None,
    };
    Some(json!({"message": message, "type": error_type, "code": code, "param": null}))
}

/// [`error_object`] with `upstream` text after the message, where the
/// classification reads it.
fn error_with_upstream(kind: ErrorKindProbe, upstream: &str) -> Option<Value> {
    let mut error = error_object(kind)?;
    let message = format!(
        "{} {upstream}",
        error["message"].as_str().unwrap_or_default()
    );
    error["message"] = json!(message.trim_end());
    Some(error)
}

fn envelope(kind: ErrorKindProbe, upstream: &str) -> Option<String> {
    error_with_upstream(kind, upstream).map(|error| json!({"error": error}).to_string())
}

/// The Responses API's error object: as Chat Completions', except that it
/// refuses a sampling parameter with `code: null` (the live reply for
/// `temperature` on a reasoning model), which is classified from the
/// message's "Unsupported parameter:" prefix, so the upstream text follows
/// it.
fn responses_envelope(kind: ErrorKindProbe, upstream: &str) -> Option<String> {
    let mut error = error_with_upstream(kind, upstream)?;
    if kind == ErrorKindProbe::Unsupported {
        error["code"] = Value::Null;
    }
    Some(json!({"error": error}).to_string())
}

/// Chat Completions' error mapping: the compat toolkit's, with OpenAI's
/// quirks.
fn chat_error(status: u16, headers: &[(&str, &str)], body: String) -> allama::Error {
    let headers: Headers = headers.iter().copied().collect();
    CompatQuirks::openai().map_http_error(status, &headers, &body)
}

/// Responses' error mapping: the compat toolkit's built-in classification
/// (no extra quota codes).
fn responses_error(status: u16, headers: &[(&str, &str)], body: String) -> allama::Error {
    let headers: Headers = headers.iter().copied().collect();
    CompatQuirks::default().map_http_error(status, &headers, &body)
}

fn slot() -> CredentialSlot {
    CredentialSlot::new(
        allama_openai::DEFAULT_CREDENTIAL_SCOPE,
        CredentialPlacement::Bearer,
    )
}

/// One call with transport options attached: `chat`, `chat_stream` or
/// `embed` of Chat Completions, or `responses` / `responses_stream`.
async fn send(options: RequestOptions, call: &'static str) -> TransportOutcome {
    let transport = ReplayTransport::default();
    let mut chat = ChatRequest::new("gpt-4o-mini").user("Hi");
    chat.options = options.clone();
    let mut embed = EmbeddingRequest::new("text-embedding-3-small").inputs(["hello"]);
    embed.options = options;
    let result = match call {
        "chat" => provider(&transport).chat(&chat).await.map(drop),
        "chat_stream" => provider(&transport).chat_stream(&chat).await.map(drop),
        "responses" => responses_provider(&transport).chat(&chat).await.map(drop),
        "responses_stream" => responses_provider(&transport)
            .chat_stream(&chat)
            .await
            .map(drop),
        _ => provider(&transport).embed(&embed).await.map(drop),
    };
    TransportOutcome::new(result, transport.requests())
}

/// A unary body and a stream body of the same response, from
/// `fixtures/<name>.json` and `fixtures/<name>.sse`.
fn capture(name: &str) -> ResponseCapture {
    let read = |extension: &str| {
        let path = format!("{}/fixtures/{name}.{extension}", env!("CARGO_MANIFEST_DIR"));
        std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {path}: {err}"))
    };
    ResponseCapture::new(read("json"), read("sse"))
}

/// The parallel tool-call ids in `agree_tool_calls` (Chat Completions).
const CHAT_TOOL_CALL_IDS: [&str; 2] = [
    "call_Sc2XmUdB2yP3h3MwRNQMcNBT",
    "call_cNeHvSaJ4H4XiLBXG5axpNqZ",
];

/// The parallel tool-call ids in `agree_tool_calls` (Responses `call_id`s).
const RESPONSES_TOOL_CALL_IDS: [&str; 2] = ["call_agree_paris", "call_agree_tokyo"];

/// `chat` (Responses when `responses`) or `chat_stream` of a provider built
/// with its default `reqwest` client against `base_url`, for the redirect
/// contract.
#[cfg(feature = "reqwest")]
async fn redirect_call(base_url: String, responses: bool, streaming: bool) -> allama::Result<()> {
    let request = ChatRequest::new("gpt-4.1").user("hi");
    if responses {
        let provider = ResponsesProvider::builder()
            .api_key("k")
            .base_url(base_url)
            .build()?;
        if streaming {
            provider.chat_stream(&request).await.map(drop)
        } else {
            provider.chat(&request).await.map(drop)
        }
    } else {
        let provider = Provider::builder()
            .api_key("k")
            .base_url(base_url)
            .build()?;
        if streaming {
            provider.chat_stream(&request).await.map(drop)
        } else {
            provider.chat(&request).await.map(drop)
        }
    }
}

/// A `200` with `body` and a request id.
fn malformed(body: String) -> ReplayTransport {
    ReplayTransport::new([ReplayResponse::new(200, body).header("x-request-id", "req_1")])
}

/// Chat Completions: `[DONE]` is the stream's terminal marker.
#[tokio::test]
async fn chat_completions_pass_every_contract() {
    let openai = provider(&ReplayTransport::new([]));
    for model in MODELS.iter().chain(&[UNKNOWN_MODEL]) {
        let validation = openai
            .validate_chat_request(&ChatRequest::new(*model).user("Hi"))
            .unwrap();
        assert_eq!(validation.coverage(), ValidationCoverage::Encoding);
    }
    let chunk = json!({
        "id": "c1", "object": "chat.completion.chunk", "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}]
    });
    let started = format!("data: {chunk}\n\n");
    let under_test = AdapterUnderTest::new(
        MODELS.iter().copied(),
        allama_openai::NAMESPACE,
        chat_probe,
        chat_stream_probe,
    )
    .error_mapping(chat_error)
    .error_envelope(envelope)
    .stream_body({
        let started = started.clone();
        move |envelope| format!("{started}data: {envelope}\n\n")
    })
    .stream_replay(|body| async move {
        provider(&ReplayTransport::new([ReplayResponse::sse(body)]))
            .chat_stream(&ChatRequest::new("gpt-4o").user("hi"))
            .await
    })
    .stream_ends(
        StreamEnds::new(started.clone())
            .terminated_without_finish(format!("{started}data: [DONE]\n\n")),
    )
    .sse_captures([CHAT_STREAM])
    .transport_options(
        slot(),
        |options| send(options, "chat"),
        |options| send(options, "chat_stream"),
    )
    .malformed_response(
        |body| async move {
            provider(&malformed(body))
                .chat(&ChatRequest::new("gpt-4o").user("hi"))
                .await
        },
        Some("req_1"),
    )
    .unary_replay(|body| async move {
        provider(&ReplayTransport::new([ReplayResponse::json(200, body)]))
            .chat(&ChatRequest::new("gpt-4.1").user("hi"))
            .await
    })
    .response_captures([capture("agree/agree_text")])
    .tool_call_capture(capture("agree/agree_tool_calls").wire_tool_call_ids(CHAT_TOOL_CALL_IDS))
    .length_capture(capture("agree/agree_length"))
    // Every OpenAI model honours strict mode.
    .skip([Contract::StrictNotHonouredWarns]);
    #[cfg(feature = "reqwest")]
    let under_test = under_test
        .redirect_not_followed(|base_url, streaming| redirect_call(base_url, false, streaming));
    // Without `reqwest` there is no HTTP client of the adapter's own.
    #[cfg(not(feature = "reqwest"))]
    let under_test = under_test.skip([Contract::RedirectNotFollowed]);
    assert_chat_adapter(&openai, &under_test).await;
}

/// Responses: `response.completed` is the stream's terminal marker, and an
/// `error` event carries its error inline.
#[tokio::test]
async fn responses_pass_every_contract() {
    let openai = responses_provider(&ReplayTransport::new([]));
    for model in MODELS.iter().chain(&[UNKNOWN_MODEL]) {
        let validation = openai
            .validate_chat_request(&ChatRequest::new(*model).user("Hi"))
            .unwrap();
        assert_eq!(validation.coverage(), ValidationCoverage::Encoding);
    }
    let created = json!({
        "type": "response.created",
        "response": {"id": "resp_1", "object": "response", "status": "in_progress", "model": "gpt-4.1", "output": []},
        "sequence_number": 0
    });
    let started = format!("event: response.created\ndata: {created}\n\n");
    let completed = json!({"type": "response.completed", "sequence_number": 1,
        "response": {"id": "resp_1", "model": "gpt-4.1", "output": []}});
    let under_test = AdapterUnderTest::new(
        MODELS.iter().copied(),
        allama_openai::NAMESPACE,
        responses_probe,
        responses_stream_probe,
    )
    .error_mapping(responses_error)
    .error_envelope(responses_envelope)
    .stream_body({
        let started = started.clone();
        move |envelope| {
            let mut error = serde_json::from_str::<Value>(envelope).unwrap()["error"].clone();
            error["type"] = json!("error");
            error["sequence_number"] = json!(1);
            format!("{started}event: error\ndata: {error}\n\n")
        }
    })
    .stream_replay(|body| async move {
        responses_provider(&ReplayTransport::new([ReplayResponse::sse(body)]))
            .chat_stream(&ChatRequest::new("gpt-4o").user("hi"))
            .await
    })
    .stream_ends(
        StreamEnds::new(started.clone()).terminated_without_finish(format!(
            "{started}event: response.completed\ndata: {completed}\n\n"
        )),
    )
    .sse_captures([RESPONSES_STREAM])
    .transport_options(
        slot(),
        |options| send(options, "responses"),
        |options| send(options, "responses_stream"),
    )
    .malformed_response(
        |body| async move {
            responses_provider(&malformed(body))
                .chat(&ChatRequest::new("gpt-4o").user("hi"))
                .await
        },
        Some("req_1"),
    )
    .unary_replay(|body| async move {
        responses_provider(&ReplayTransport::new([ReplayResponse::json(200, body)]))
            .chat(&ChatRequest::new("gpt-4.1").user("hi"))
            .await
    })
    .response_captures([capture("responses/agree_text")])
    .tool_call_capture(
        capture("responses/agree_tool_calls").wire_tool_call_ids(RESPONSES_TOOL_CALL_IDS),
    )
    .length_capture(capture("responses/agree_length"))
    // Every OpenAI model honours strict mode. No model answers
    // `ReasoningReplay` `Unsupported` on Responses: which models reason is
    // model data, which this adapter no longer ships (ADR 0015).
    .skip([
        Contract::StrictNotHonouredWarns,
        Contract::UnreplayableReasoningDropped,
    ]);
    #[cfg(feature = "reqwest")]
    let under_test = under_test
        .redirect_not_followed(|base_url, streaming| redirect_call(base_url, true, streaming));
    // Without `reqwest` there is no HTTP client of the adapter's own.
    #[cfg(not(feature = "reqwest"))]
    let under_test = under_test.skip([Contract::RedirectNotFollowed]);
    assert_chat_adapter(&openai, &under_test).await;
}

/// Embeddings, including a provider that needs a per-request credential
/// (checked on its own in `request_honesty.rs`).
#[tokio::test]
async fn embeddings_pass_every_contract() {
    let openai = provider(&ReplayTransport::new([]));
    for model in EMBEDDING_MODELS.iter().chain(&[UNKNOWN_MODEL]) {
        let validation = openai
            .validate_embedding_request(&EmbeddingRequest::new(*model).input("Hi"))
            .unwrap();
        assert_eq!(validation.coverage(), ValidationCoverage::Encoding);
    }
    let under_test = EmbeddingAdapterUnderTest::new(
        EMBEDDING_MODELS.iter().copied(),
        allama_openai::NAMESPACE,
        embed_probe,
    )
    .answering(VectorOrder::Indexed, |request, vectors| async move {
        let data: Vec<Value> = vectors
            .iter()
            .map(|vector| json!({"object": "embedding", "index": vector.index, "embedding": vector.values}))
            .collect();
        let body = json!({"object": "list", "model": request.model, "data": data});
        provider(&ReplayTransport::new([ReplayResponse::json(200, body.to_string())]))
            .embed(&request)
            .await
    })
    .error_mapping(chat_error)
    .error_envelope(envelope)
    .transport_options(slot(), |options| send(options, "embed"));
    assert_embedding_adapter(&openai, &under_test).await;
}
