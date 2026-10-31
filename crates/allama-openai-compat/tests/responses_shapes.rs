//! Responses API shapes the generic Responses provider decodes and encodes:
//! hosted MCP approval, message content only in finished items, interleaved
//! parallel function calls, hosted tool usage, stored files, tool provider
//! data, replayed annotations and numeric error codes.

use std::path::PathBuf;

use allama::{
    ChatProvider, ChatRequest, ChatResponse, ChatStreamExt, ContentBlock, ContentPart, Error, File,
    FinishReason, Image, MediaSource, Message, ProviderData, Text, ToolDefinition, WarningKind,
};
use allama_conformance::{FixtureDir, load_text_fixture};
use allama_http::http::StatusCode;
use allama_http::{ReplayResponse, ReplayTransport};
use allama_openai_compat::{ClientToolCall, ResponsesProvider, ResponsesQuirks};
use serde_json::{Value, json};

const NAMESPACE: &str = "openai-compat:local";

fn fixtures() -> FixtureDir {
    FixtureDir::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/responses"))
}

fn provider(responses: Vec<ReplayResponse>) -> (ResponsesProvider, ReplayTransport) {
    let transport = ReplayTransport::new(responses);
    let provider = ResponsesProvider::builder()
        .base_url("http://upstream/v1")
        .provider_name("local")
        .quirks(ResponsesQuirks::new(NAMESPACE))
        .http_transport(transport.clone())
        .build()
        .unwrap();
    (provider, transport)
}

fn text_body() -> ReplayResponse {
    ReplayResponse::json(
        StatusCode::OK,
        json!({
            "id": "resp_t", "object": "response", "status": "completed", "model": "m",
            "output": [{"type": "message", "id": "msg_t", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "ok", "annotations": []}]}]
        })
        .to_string(),
    )
}

fn sent_body(transport: &ReplayTransport) -> Value {
    serde_json::from_str(transport.requests()[0].body_text().unwrap()).unwrap()
}

async fn streamed(name: &str) -> ChatResponse {
    let (provider, _) = provider(vec![ReplayResponse::sse(load_text_fixture(
        &fixtures(),
        name,
    ))]);
    provider
        .chat_stream(&ChatRequest::new("m").user("q"))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap()
}

/// A response that only asks to approve a hosted MCP call waits
/// for the client (`ToolCalls`), and the approval goes back as an input item
/// of its own, not inside a message.
#[tokio::test]
async fn mcp_approval_waits_for_the_client_and_is_answered_as_an_item() {
    let (provider, transport) = provider(vec![
        ReplayResponse::json(
            StatusCode::OK,
            load_text_fixture(&fixtures(), "mcp_approval_request.json"),
        ),
        text_body(),
    ]);
    let response = provider
        .chat(&ChatRequest::new("m").user("MSRV policy?"))
        .await
        .unwrap();
    assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
    let pending = ClientToolCall::pending(&response.content);
    let [ClientToolCall::McpApproval(request)] = pending.as_slice() else {
        panic!("expected one approval request: {pending:?}")
    };
    assert_eq!(
        (
            request.id.as_str(),
            request.server_label.as_str(),
            request.name.as_str()
        ),
        ("mcpr_1", "deepwiki", "ask_question")
    );

    let mut next = ChatRequest::new("m").user("MSRV policy?");
    next.push_message(response.to_assistant_message());
    next.push_message(Message::user(vec![request.approve()]));
    provider.chat(&next).await.unwrap();
    let body: Value = serde_json::from_str(transport.requests()[1].body_text().unwrap()).unwrap();
    let items = body["input"].as_array().unwrap();
    assert!(
        items.iter().any(|item| *item
            == json!({"type": "mcp_approval_response", "approval_request_id": "mcpr_1", "approve": true})),
        "{items:#?}"
    );
    assert!(
        items
            .iter()
            .any(|item| item["type"] == "mcp_approval_request" && item["id"] == "mcpr_1"),
        "{items:#?}"
    );
}

/// A server that sends message content only with the finished
/// item (no content-part or delta events) still yields the text and the
/// refusal, as the unary path does.
#[tokio::test]
async fn content_only_in_finished_items_is_decoded() {
    let response = streamed("finished_items_only.sse").await;
    assert_eq!(response.text().as_deref(), Some("Paris is sunny."));
    assert_eq!(response.first_refusal(), Some("I won't say more."));
    assert_eq!(response.finish_reason, Some(FinishReason::Refusal));
    assert!(response.warnings.is_empty(), "{:?}", response.warnings);
}

/// Parallel function calls whose argument deltas interleave, and
/// which finish out of order, keep their own arguments; hosted tool
/// requests are tool requests in the usage.
#[tokio::test]
async fn interleaved_parallel_calls_and_tool_usage() {
    let response = streamed("parallel_calls_interleaved.sse").await;
    let calls: Vec<_> = response
        .tool_calls()
        .map(|call| (call.id.as_str(), call.arguments.as_str()))
        .collect();
    assert_eq!(
        calls,
        [
            ("call_a", r#"{"city":"Paris"}"#),
            ("call_b", r#"{"city":"Tokyo"}"#)
        ]
    );
    assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
    let usage = response.usage.unwrap();
    assert_eq!(usage.tool_requests.get("web_search"), Some(&2));
    assert_eq!(usage.input_tokens, Some(50));
    assert_eq!(
        response
            .provider_data
            .get_raw(NAMESPACE, "tool_usage")
            .and_then(|usage| usage.pointer("/web_search/num_requests")),
        Some(&json!(2))
    );
}

/// A file the endpoint stores goes by its id, for images and files.
#[tokio::test]
async fn stored_files_are_sent_by_id() {
    let (provider, transport) = provider(vec![text_body()]);
    let image = Image::new(MediaSource::provider_file(NAMESPACE, "file-img"));
    let document = File::new(
        "application/pdf",
        MediaSource::provider_file(NAMESPACE, "file-doc"),
    );
    let request = ChatRequest::new("m").message(Message::user(vec![
        ContentPart::Image(image),
        ContentPart::File(document),
        ContentPart::text("Summarize both."),
    ]));
    provider.chat(&request).await.unwrap();
    let content = sent_body(&transport)["input"][0]["content"].clone();
    assert_eq!(
        content[0],
        json!({"type": "input_image", "file_id": "file-img"})
    );
    assert_eq!(
        content[1],
        json!({"type": "input_file", "file_id": "file-doc"})
    );

    // Another provider's file is refused before anything is sent.
    let foreign = ChatRequest::new("m").message(Message::user(vec![ContentPart::Image(
        Image::new(MediaSource::provider_file("anthropic", "file-x")),
    )]));
    assert!(provider.validate_chat_request(&foreign).is_err());
}

/// A tool's provider data adds definition fields, but cannot
/// replace those the adapter writes (or the strict policy's outcome).
#[tokio::test]
async fn tool_provider_data_cannot_replace_owned_fields() {
    let (provider, transport) = provider(vec![text_body()]);
    let tool = |key: &str| {
        let mut tool = ToolDefinition::new("f", json!({"type": "object"}));
        tool.provider_data = ProviderData::new().with_raw(NAMESPACE, key, json!(true));
        tool
    };
    for key in ["strict", "parameters", "Name", "type"] {
        let request = ChatRequest::new("m").user("q").tools([tool(key)]);
        let err = provider.validate_chat_request(&request).unwrap_err();
        assert!(
            matches!(err, Error::InvalidRequest { .. }),
            "{key}: {err:?}"
        );
    }
    let request = ChatRequest::new("m")
        .user("q")
        .tools([tool("defer_loading")]);
    provider.chat(&request).await.unwrap();
    assert_eq!(sent_body(&transport)["tools"][0]["defer_loading"], true);
}

/// Annotations in the Chat Completions shape stored under the
/// Responses key are not replayed as if they were Responses annotations.
#[tokio::test]
async fn chat_shaped_annotations_are_not_replayed() {
    let (provider, transport) = provider(vec![text_body()]);
    let annotations = json!([
        {"type": "url_citation", "url_citation": {"url": "https://a", "start_index": 0, "end_index": 1}},
        {"type": "url_citation", "url": "https://b", "start_index": 0, "end_index": 1, "title": "B"}
    ]);
    let text = Text::new("a").with_provider_data(ProviderData::new().with_raw(
        NAMESPACE,
        "annotations",
        annotations,
    ));
    let mut request = ChatRequest::new("m").user("q");
    request.push_message(Message::assistant_blocks(vec![ContentBlock::Text(text)]));
    request.push_message(Message::user("more"));
    let response = provider.chat(&request).await.unwrap();
    let replayed = &sent_body(&transport)["input"][1]["content"][0]["annotations"];
    assert_eq!(
        *replayed,
        json!([{"type": "url_citation", "url": "https://b", "start_index": 0, "end_index": 1, "title": "B"}])
    );
    assert!(
        response
            .warnings
            .iter()
            .any(|w| w.kind == WarningKind::Dropped),
        "{:?}",
        response.warnings
    );
}

/// A numeric error code is read, and classified as the status it
/// is; an image the endpoint cannot use is an invalid request, a policy
/// violation a content filter.
#[tokio::test]
async fn in_band_error_codes_are_classified() {
    let failed = |code: Value| {
        ReplayResponse::json(
            StatusCode::OK,
            json!({"id": "r", "status": "failed", "output": [],
                   "error": {"code": code, "message": "nope"}})
            .to_string(),
        )
    };
    for (code, check) in [
        (
            json!(429),
            (|e: &Error| matches!(e, Error::RateLimited { .. })) as fn(&Error) -> bool,
        ),
        (json!("image_too_large"), |e| {
            matches!(e, Error::InvalidRequest { .. })
        }),
        (json!("failed_to_download_image"), |e| {
            matches!(e, Error::InvalidRequest { .. })
        }),
        (json!("image_content_policy_violation"), |e| {
            matches!(e, Error::ContentFiltered { .. })
        }),
    ] {
        let (provider, _) = provider(vec![failed(code.clone())]);
        let err = provider
            .chat(&ChatRequest::new("m").user("q"))
            .await
            .unwrap_err();
        assert!(check(&err), "{code}: {err:?}");
        assert!(!err.is_retryable() || code == json!(429), "{code}: {err:?}");
    }
}
