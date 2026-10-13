//! Classification on both paths, for Chat Completions and Responses: an HTTP
//! failure surfaces the same error kind from `chat` and `chat_stream`, an
//! error reported inside a started stream is classified like the same error
//! over HTTP, and structured output is JSON on both paths.

use allama::{ChatProvider, ChatRequest, ChatStream, ResponseFormat};
use allama_conformance::contract::{
    MALFORMED_BODY, assert_auth_error_contract, assert_context_length_exceeded_contract,
    assert_overloaded_error_contract, assert_quota_exhausted_contract,
    assert_rate_limited_error_contract, assert_structured_output_contract,
    assert_structured_output_contract_stream, collect_chat_stream,
};
use allama_conformance::stream_errors::{ErrorKindProbe, assert_stream_errors_classified};
use allama_http::{Headers, ReplayResponse, ReplayTransport};
use allama_openai_compat::{CompatQuirks, Provider, ResponsesProvider};
use serde_json::{Value, json};
use support::envelope;

mod support;

/// The vendor's own quota code, beside OpenAI's `insufficient_quota`.
const VENDOR_QUOTA_CODE: &str = "quota_exceeded";

fn headers(pairs: &[(&str, &str)]) -> Headers {
    pairs.iter().copied().collect()
}

/// The failures the error-kind contracts expect, in contract order, each
/// answering `chat` and then `chat_stream`.
fn failures() -> ReplayTransport {
    let kinds = [
        (401, ErrorKindProbe::Auth),
        (429, ErrorKindProbe::RateLimited),
        (503, ErrorKindProbe::Overloaded),
        (400, ErrorKindProbe::ContextLength),
        (429, ErrorKindProbe::QuotaExhausted),
    ];
    ReplayTransport::new(kinds.into_iter().flat_map(|(status, kind)| {
        let body = envelope(kind, "insufficient_quota").unwrap();
        [
            ReplayResponse::json(status, body.clone()),
            ReplayResponse::json(status, body),
        ]
    }))
}

fn chat_provider(quirks: CompatQuirks, transport: ReplayTransport) -> Provider {
    Provider::builder()
        .base_url("http://upstream/v1")
        .provider_name("vendor")
        .quirks(quirks)
        .http_transport(transport)
        .build()
        .unwrap()
}

fn responses_provider(transport: ReplayTransport) -> ResponsesProvider {
    ResponsesProvider::builder()
        .base_url("http://upstream/v1")
        .provider_name("vendor")
        .http_transport(transport)
        .build()
        .unwrap()
}

async fn assert_error_kinds_on_both_paths(provider: &impl ChatProvider) {
    let request = ChatRequest::new("m").user("hi");
    let stream = |request| collect_chat_stream(provider, request);
    assert_auth_error_contract(provider, &request, stream).await;
    assert_rate_limited_error_contract(provider, &request, stream).await;
    assert_overloaded_error_contract(provider, &request, stream).await;
    assert_context_length_exceeded_contract(provider, &request, stream).await;
    assert_quota_exhausted_contract(provider, &request, stream).await;
}

#[tokio::test]
async fn chat_error_kinds_hold_on_both_paths() {
    let provider = chat_provider(CompatQuirks::new("openai-compat:vendor"), failures());
    assert_error_kinds_on_both_paths(&provider).await;
}

#[tokio::test]
async fn responses_error_kinds_hold_on_both_paths() {
    assert_error_kinds_on_both_paths(&responses_provider(failures())).await;
}

/// A `200` whose body is no response: what a proxy answers in place of
/// the endpoint's JSON.
fn malformed() -> ReplayTransport {
    ReplayTransport::new([ReplayResponse::new(200, MALFORMED_BODY).header("x-request-id", "req_1")])
}

#[tokio::test]
async fn malformed_embedding_bodies_are_unexpected_responses() {
    use allama::{EmbeddingProvider, EmbeddingRequest, Error};
    let provider = chat_provider(CompatQuirks::new("openai-compat:vendor"), malformed());
    let err = provider
        .embed(&EmbeddingRequest::new("e").input("hi"))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::UnexpectedResponse { .. }), "{err:?}");
    assert_eq!(err.status(), Some(200));
    assert_eq!(err.request_id(), Some("req_1"));
    assert_eq!(err.context().unwrap().provider.as_deref(), Some("vendor"));
}

/// A chunk of text before the error, so the stream has started.
fn chat_chunk() -> Value {
    json!({
        "id": "c1", "object": "chat.completion.chunk", "model": "m",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}]
    })
}

async fn chat_stream_with(quirks: CompatQuirks, body: String) -> allama::Result<ChatStream> {
    let transport = ReplayTransport::new([ReplayResponse::sse(body)]);
    chat_provider(quirks, transport)
        .chat_stream(&ChatRequest::new("m").user("hi"))
        .await
}

#[tokio::test]
async fn chat_stream_error_events_are_classified_like_http_errors() {
    // An `event: error` frame; the aggregates check an error chunk.
    assert_stream_errors_classified(
        |kind| envelope(kind, "insufficient_quota"),
        |envelope| {
            format!(
                "data: {}\n\nevent: error\ndata: {envelope}\n\n",
                chat_chunk()
            )
        },
        |status, pairs, body| {
            CompatQuirks::default().map_http_error(status, &headers(pairs), &body)
        },
        |body| chat_stream_with(CompatQuirks::new("openai-compat:vendor"), body),
    )
    .await;
}

#[tokio::test]
async fn vendor_quota_codes_are_classified_on_both_paths() {
    let mut quirks = CompatQuirks::new("openai-compat:vendor");
    quirks.quota_error_codes.push(VENDOR_QUOTA_CODE.into());
    assert_stream_errors_classified(
        |kind| envelope(kind, VENDOR_QUOTA_CODE),
        |envelope| format!("data: {}\n\ndata: {envelope}\n\n", chat_chunk()),
        |status, pairs, body| quirks.map_http_error(status, &headers(pairs), &body),
        |body| chat_stream_with(quirks.clone(), body),
    )
    .await;
}

/// `response.created`, so the Responses stream has started.
fn response_created() -> String {
    let event = json!({
        "type": "response.created",
        "response": {"id": "resp_1", "object": "response", "status": "in_progress", "model": "m", "output": []},
        "sequence_number": 0
    });
    format!("event: response.created\ndata: {event}\n\n")
}

async fn responses_stream(body: String) -> allama::Result<ChatStream> {
    let transport = ReplayTransport::new([ReplayResponse::sse(body)]);
    responses_provider(transport)
        .chat_stream(&ChatRequest::new("m").user("hi"))
        .await
}

#[tokio::test]
async fn responses_stream_errors_are_classified_like_http_errors() {
    // `response.failed` carries the error on the response, as a code and a
    // message only, so it cannot name a refused parameter. (The aggregate
    // checks an `error` event, which carries it inline.)
    let failed = |envelope: &str| {
        let error = &serde_json::from_str::<Value>(envelope).unwrap()["error"];
        let failed = json!({
            "type": "response.failed",
            "response": {
                "id": "resp_1", "object": "response", "status": "failed", "model": "m",
                "output": [], "error": {"code": error["code"], "message": error["message"]}
            },
            "sequence_number": 1
        });
        format!(
            "{}event: response.failed\ndata: {failed}\n\n",
            response_created()
        )
    };
    assert_stream_errors_classified(
        |kind| {
            (kind != ErrorKindProbe::Unsupported)
                .then(|| envelope(kind, "insufficient_quota"))
                .flatten()
        },
        failed,
        |status, pairs, body| {
            CompatQuirks::default().map_http_error(status, &headers(pairs), &body)
        },
        responses_stream,
    )
    .await;
}

fn structured_request() -> ChatRequest {
    ChatRequest::new("m").user("Which city?").response_format(
        ResponseFormat::json_schema(json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": false
        }))
        .name("city"),
    )
}

const ANSWER: &str = r#"{"city":"Paris"}"#;

#[tokio::test]
async fn chat_structured_output_is_json_on_both_paths() {
    let response = json!({
        "id": "c1", "object": "chat.completion", "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": ANSWER}}]
    });
    let chunk = json!({
        "id": "c1", "object": "chat.completion.chunk", "model": "m",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ANSWER}, "finish_reason": "stop"}]
    });
    let transport = ReplayTransport::new([
        ReplayResponse::json(200, response.to_string()),
        ReplayResponse::sse(format!("data: {chunk}\n\ndata: [DONE]\n\n")),
    ]);
    let provider = chat_provider(CompatQuirks::new("openai-compat:vendor"), transport);
    assert_structured_output_contract(&provider, &structured_request()).await;
    assert_structured_output_contract_stream(&provider, &structured_request()).await;
}

#[tokio::test]
async fn responses_structured_output_is_json_on_both_paths() {
    let message = |status: &str| {
        json!({
            "type": "message", "id": "msg_1", "role": "assistant", "status": status,
            "content": [{"type": "output_text", "text": ANSWER, "annotations": []}]
        })
    };
    let response = json!({
        "id": "resp_1", "object": "response", "status": "completed", "model": "m",
        "output": [message("completed")],
        "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10}
    });
    let events = [
        json!({"type": "response.output_item.added", "output_index": 0,
               "item": {"type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": []}}),
        json!({"type": "response.content_part.added", "item_id": "msg_1", "output_index": 0, "content_index": 0,
               "part": {"type": "output_text", "text": "", "annotations": []}}),
        json!({"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "content_index": 0,
               "delta": ANSWER}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": message("completed")}),
        json!({"type": "response.completed", "response": response}),
    ];
    let mut sse = response_created();
    for (sequence, mut event) in events.into_iter().enumerate() {
        event["sequence_number"] = json!(sequence + 1);
        sse.push_str(&format!(
            "event: {}\ndata: {event}\n\n",
            event["type"].as_str().unwrap()
        ));
    }
    let transport = ReplayTransport::new([
        ReplayResponse::json(200, response.to_string()),
        ReplayResponse::sse(sse),
    ]);
    let provider = responses_provider(transport);
    assert_structured_output_contract(&provider, &structured_request()).await;
    assert_structured_output_contract_stream(&provider, &structured_request()).await;
}

/// Ollama passes llama.cpp's `501` through for an embeddings request to a
/// chat-only model: the embedding call fails once with the provider's
/// refusal, which keeps the status, request id and upstream text.
#[tokio::test]
async fn ollama_refuses_embeddings_for_a_chat_only_model() {
    use allama::{EmbeddingProvider, EmbeddingRequest, Error, UnsupportedStage};
    use allama_openai_compat::providers::Ollama;

    let message = "This server does not support embeddings. Start it with `--embeddings`";
    let transport = ReplayTransport::new([ReplayResponse::json(
        501,
        json!({"error": {"message": message, "type": "api_error", "param": null, "code": null}})
            .to_string(),
    )
    .header("x-request-id", "req_501")]);
    let provider = Ollama::builder("http://localhost:11434")
        .unwrap()
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let err = provider
        .embed(&EmbeddingRequest::new("qwen3:0.6b").input("hello"))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
    assert_eq!(err.unsupported_stage(), Some(UnsupportedStage::Provider));
    assert_eq!(err.status(), Some(501));
    assert_eq!(err.request_id(), Some("req_501"));
    assert_eq!(err.upstream_message(), Some(message));
    assert!(!err.is_retryable());
    assert_eq!(transport.requests().len(), 1);
}
