//! Stream ends beyond the contract the aggregates run
//! (`request_honesty.rs`): Responses' other terminal events are finishes.

use allama::{ChatProvider, ChatRequest, ChatStream};
use allama_http::{ReplayResponse, ReplayTransport};
use allama_openai_compat::ResponsesProvider;
use serde_json::json;

async fn responses(body: String) -> allama::Result<ChatStream> {
    ResponsesProvider::builder()
        .base_url("http://upstream/v1")
        .provider_name("vendor")
        .http_transport(ReplayTransport::new([ReplayResponse::sse(body)]))
        .build()
        .unwrap()
        .chat_stream(&ChatRequest::new("m").user("q"))
        .await
}

/// `response.incomplete` is terminal: it is a finish (`Length` here), never
/// a stream error.
#[tokio::test]
async fn an_incomplete_response_is_a_finish() {
    use allama::{ChatStreamExt, FinishReason};
    let incomplete = json!({"type": "response.incomplete", "sequence_number": 0,
        "response": {"id": "resp_1", "model": "m", "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"}, "output": []}});
    let response = responses(format!(
        "event: response.incomplete\ndata: {incomplete}\n\n"
    ))
    .await
    .unwrap()
    .collect_response()
    .await
    .unwrap();
    assert_eq!(response.finish_reason, Some(FinishReason::Length));
    assert!(!response.finished_normally());
}
