use allama::chat::stream::StreamCollector;
use allama::{ChatStream, FinishReason, StreamErrorKind};
use allama_http::TransportError;
use allama_http::sse::{SseChatStream, SseStream};
use futures_util::StreamExt;
use serde_json::json;

use super::*;
use crate::responses::decode::{TEST_NAMESPACE, test_context};

/// Render events as SSE (`event:` names the type, like the live API).
fn sse(events: &[Value]) -> String {
    events
        .iter()
        .map(|e| format!("event: {}\ndata: {e}\n\n", e["type"].as_str().unwrap()))
        .collect()
}

async fn run(events: &[Value]) -> Vec<Result<StreamEvent>> {
    stream_from_sse_text(&sse(events)).collect().await
}

fn stream_from_sse_text(text: &str) -> ChatStream {
    stream_with(text, SseState::new(test_context(None)))
}

/// `text` decoded as `chat_stream` decodes a body, with the hold
/// `finish_stream` applies.
fn stream_with(text: &str, state: SseState) -> ChatStream {
    let body = allama_conformance::byte_stream_from_sse_text(text).map(|chunk| {
        chunk.map_err(|error| {
            TransportError::new(allama_http::TransportErrorKind::Body, error.to_string())
        })
    });
    Box::pin(allama::adapter::chat::start_first(SseChatStream::new(
        SseStream::new(body),
        state,
    )))
}

fn collect(events: &[Result<StreamEvent>]) -> Result<allama::ChatResponse> {
    let mut collector = StreamCollector::new();
    for event in events {
        collector.push(event.clone()?)?;
    }
    collector.finish()
}

fn created() -> Value {
    json!({"type": "response.created", "response": {"id": "resp_1", "model": "gpt-5-mini", "status": "in_progress", "output": []}})
}

fn completed(status: &str, extra: Value) -> Value {
    let mut response = json!({
        "id": "resp_1", "model": "gpt-5-mini", "status": status, "output": [],
        "usage": {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 2}, "output_tokens": 20,
                  "output_tokens_details": {"reasoning_tokens": 8}, "total_tokens": 30}
    });
    response
        .as_object_mut()
        .unwrap()
        .extend(extra.as_object().cloned().unwrap_or_default());
    json!({"type": format!("response.{status}"), "response": response})
}

/// Regression: the finished part's `logprobs`
/// reach the text block as provider data, as on the unary path.
#[tokio::test]
async fn output_text_logprobs_are_provider_data() {
    let logprobs =
        json!([{"token": "Hi", "logprob": -0.01, "bytes": [72, 105], "top_logprobs": []}]);
    let part =
        json!({"type": "output_text", "text": "Hi", "annotations": [], "logprobs": logprobs});
    let events = run(&[
        created(),
        json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "id": "msg_1", "content": []}}),
        json!({"type": "response.content_part.added", "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []}}),
        json!({"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "Hi", "logprobs": logprobs}),
        json!({"type": "response.content_part.done", "output_index": 0, "content_index": 0, "part": part}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": {"type": "message", "id": "msg_1", "content": [part]}}),
        completed("completed", json!({})),
    ])
    .await;
    let response = collect(&events).unwrap();
    let ContentBlock::Text(text) = &response.content[0] else {
        panic!("expected text: {:?}", response.content)
    };
    assert_eq!(
        text.provider_data.get_raw(TEST_NAMESPACE, "logprobs"),
        Some(&logprobs)
    );
    let decoded = decode::decode_item(
        &json!({"type": "message", "id": "msg_1", "content": [part]}),
        crate::responses::decode::TEST_ORIGIN,
        &mut Vec::new(),
    )
    .unwrap();
    assert_eq!(response.content, decoded);
}

#[tokio::test]
async fn refusal_and_annotations_stream_as_their_own_blocks() {
    let events = run(&[
        created(),
        json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "id": "msg_1", "phase": "final_answer", "content": []}}),
        json!({"type": "response.content_part.added", "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}}),
        json!({"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "See "}),
        json!({"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "docs."}),
        json!({"type": "response.content_part.done", "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": "See docs.", "annotations": [{"type": "url_citation", "url": "https://example.com"}]}}),
        json!({"type": "response.content_part.added", "output_index": 0, "content_index": 1, "part": {"type": "refusal", "refusal": ""}}),
        json!({"type": "response.refusal.delta", "output_index": 0, "content_index": 1, "delta": "I can't."}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": {"type": "message", "id": "msg_1", "content": [
            {"type": "output_text", "text": "See docs.", "annotations": [{"type": "url_citation", "url": "https://example.com"}]},
            {"type": "refusal", "refusal": "I can't."}
        ]}}),
        completed("completed", json!({})),
    ])
    .await;
    // The refusal is never a text delta.
    assert!(
        !events.iter().any(|event| matches!(
            event,
            Ok(StreamEvent::TextDelta { text, .. }) if text.contains("can't")
        )),
        "{events:?}"
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, Ok(e) if *e == StreamEvent::refusal_delta(1, "I can't.")))
    );
    let response = collect(&events).unwrap();
    assert_eq!(response.finish_reason, Some(FinishReason::Refusal));
    assert_eq!(response.content.len(), 2);
    let ContentBlock::Text(text) = &response.content[0] else {
        panic!()
    };
    assert_eq!(text.text, "See docs.");
    assert_eq!(
        text.provider_data.get_raw(TEST_NAMESPACE, "annotations"),
        Some(&json!([{"type": "url_citation", "url": "https://example.com"}]))
    );
    assert_eq!(
        text.provider_data.get_raw(TEST_NAMESPACE, "phase"),
        Some(&json!("final_answer"))
    );
    let ContentBlock::Refusal(refusal) = &response.content[1] else {
        panic!()
    };
    assert_eq!(refusal.text, "I can't.");
    assert_eq!(response.first_refusal(), Some("I can't."));
    assert_eq!(response.text().as_deref(), Some("See docs."));
    let usage = response.usage.unwrap();
    assert_eq!(usage.cache_read_input_tokens, Some(2));
    assert!(usage.contract_violation().is_none());
    assert_eq!(response.provider.as_deref(), Some("test"));
}

/// A refusal delta for a part that was never announced opens a refusal
/// block, and a refusal part that never streams is reconciled when its
/// item is done, still as refusal deltas.
#[tokio::test]
async fn unannounced_and_unstreamed_refusals_stay_refusals() {
    let events = run(&[
        created(),
        json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "id": "msg_1", "content": []}}),
        json!({"type": "response.refusal.delta", "item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": "No."}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": {"type": "message", "id": "msg_1", "content": [
            {"type": "refusal", "refusal": "No."}
        ]}}),
        json!({"type": "response.output_item.added", "output_index": 1, "item": {"type": "message", "id": "msg_2", "content": []}}),
        json!({"type": "response.content_part.added", "output_index": 1, "content_index": 0, "part": {"type": "refusal", "refusal": ""}}),
        json!({"type": "response.output_item.done", "output_index": 1, "item": {"type": "message", "id": "msg_2", "content": [
            {"type": "refusal", "refusal": "Still no."}
        ]}}),
        completed("completed", json!({})),
    ])
    .await;
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, Ok(StreamEvent::TextDelta { .. }))),
        "{events:?}"
    );
    let response = collect(&events).unwrap();
    assert_eq!(response.finish_reason, Some(FinishReason::Refusal));
    let refusals: Vec<_> = response
        .content
        .iter()
        .filter_map(ContentBlock::as_refusal)
        .map(|refusal| refusal.text.as_str())
        .collect();
    assert_eq!(refusals, ["No.", "Still no."]);
}

#[tokio::test]
async fn reasoning_summary_parts_and_encrypted_content() {
    let events = run(&[
        created(),
        json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning", "id": "rs_1", "summary": []}}),
        json!({"type": "response.reasoning_summary_part.added", "output_index": 0, "summary_index": 0, "part": {"type": "summary_text", "text": ""}}),
        json!({"type": "response.reasoning_summary_text.delta", "output_index": 0, "summary_index": 0, "delta": "First."}),
        json!({"type": "response.reasoning_summary_part.added", "output_index": 0, "summary_index": 1, "part": {"type": "summary_text", "text": ""}}),
        json!({"type": "response.reasoning_summary_text.delta", "output_index": 0, "summary_index": 1, "delta": "Second."}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": {"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAA",
            "summary": [{"type": "summary_text", "text": "First."}, {"type": "summary_text", "text": "Second."}]}}),
        completed("completed", json!({})),
    ])
    .await;
    let response = collect(&events).unwrap();
    let ContentBlock::Reasoning(reasoning) = &response.content[0] else {
        panic!()
    };
    assert_eq!(reasoning.text, "First.\n\nSecond.");
    assert_eq!(
        reasoning
            .provider_data
            .get_raw(TEST_NAMESPACE, "encrypted_content"),
        Some(&json!("gAAA"))
    );
    assert_eq!(
        reasoning.provider_data.get_raw(TEST_NAMESPACE, "item_id"),
        Some(&json!("rs_1"))
    );
    // Same content as the non-streaming decoder produces.
    let decoded = decode::decode_item(
        &json!({"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAA",
                "summary": [{"type": "summary_text", "text": "First."}, {"type": "summary_text", "text": "Second."}]}),
        crate::responses::decode::TEST_ORIGIN,
        &mut Vec::new(),
    )
    .unwrap();
    assert_eq!(response.content, decoded);
}

#[tokio::test]
async fn items_without_deltas_are_reconciled_on_done() {
    let events = run(&[
        created(),
        json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "f", "arguments": ""}}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "f", "arguments": "{\"a\":1}"}}),
        json!({"type": "response.output_item.added", "output_index": 1, "item": {"type": "message", "content": []}}),
        json!({"type": "response.content_part.added", "output_index": 1, "content_index": 0, "part": {"type": "output_text", "text": ""}}),
        json!({"type": "response.output_item.done", "output_index": 1, "item": {"type": "message", "content": [{"type": "output_text", "text": "Hi", "annotations": []}]}}),
        completed("completed", json!({})),
    ])
    .await;
    let response = collect(&events).unwrap();
    let call = response.first_tool_call().unwrap();
    assert_eq!(call.arguments, r#"{"a":1}"#);
    assert_eq!(
        call.provider_data.get_raw(TEST_NAMESPACE, "item_id"),
        Some(&json!("fc_1"))
    );
    assert_eq!(response.text().as_deref(), Some("Hi"));
    assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
}

#[tokio::test]
async fn hosted_tool_items_become_provider_tool_blocks() {
    let item = json!({"type": "web_search_call", "id": "ws_1", "status": "completed", "action": {"type": "search", "query": "rust"}});
    let events = run(&[
        created(),
        json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "web_search_call", "id": "ws_1", "status": "in_progress"}}),
        json!({"type": "response.web_search_call.searching", "output_index": 0, "item_id": "ws_1", "sequence_number": 3}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": item}),
        completed("completed", json!({})),
    ])
    .await;
    let progress: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Ok(StreamEvent::ProviderDelta {
                index, type_name, ..
            }) => Some((*index, type_name.as_str())),
            _ => None,
        })
        .collect();
    assert_eq!(progress, [(0, "web_search_call.searching")]);

    let response = collect(&events).unwrap();
    let ContentBlock::ProviderToolCall(call) = &response.content[0] else {
        panic!("{:?}", response.content)
    };
    assert_eq!(
        (call.id.as_str(), call.name.as_str()),
        ("ws_1", "web_search")
    );
    assert_eq!(call.origin.as_deref(), Some(TEST_NAMESPACE));
    let ContentBlock::ProviderToolResult(result) = &response.content[1] else {
        panic!("{:?}", response.content)
    };
    assert_eq!(result.call_id, "ws_1");
    assert_eq!(result.output, json!({"status": "completed"}));
    assert_eq!(response.finish_reason, Some(FinishReason::Stop));
}

#[tokio::test]
async fn incomplete_response_maps_to_length() {
    let events = run(&[
        created(),
        completed(
            "incomplete",
            json!({"incomplete_details": {"reason": "max_output_tokens"}}),
        ),
    ])
    .await;
    let response = collect(&events).unwrap();
    assert_eq!(response.finish_reason, Some(FinishReason::Length));
    assert_eq!(
        response.finish_reason_raw.as_deref(),
        Some("max_output_tokens")
    );
}

#[tokio::test]
async fn error_events_map_to_typed_errors() {
    let inline = run(&[
        created(),
        json!({"type": "error", "code": "rate_limit_exceeded", "message": "slow down", "param": null}),
    ])
    .await;
    let err = inline.last().unwrap().as_ref().unwrap_err();
    assert!(matches!(err, Error::RateLimited { .. }), "{err:?}");
    assert_eq!(err.context().unwrap().provider_type, None);

    let nested = run(&[json!({"type": "error", "error": {"type": "insufficient_quota", "code": "insufficient_quota", "message": "quota"}})]).await;
    assert!(matches!(
        nested.last().unwrap().as_ref().unwrap_err(),
        Error::QuotaExhausted { .. }
    ));

    let failed = run(&[
        created(),
        json!({"type": "response.failed", "response": {"id": "resp_1", "status": "failed", "output": [],
            "error": {"code": "server_error", "message": "The server had an error"}}}),
    ])
    .await;
    let err = failed.last().unwrap().as_ref().unwrap_err();
    assert!(matches!(err, Error::Provider { .. }));
    assert!(err.is_retryable());
}

#[tokio::test]
async fn stream_without_terminal_event_is_incomplete() {
    let events = run(&[
        created(),
        json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "content": []}}),
        json!({"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "partial"}),
    ])
    .await;
    let err = events.last().unwrap().as_ref().unwrap_err();
    assert!(matches!(
        err,
        Error::Stream {
            kind: StreamErrorKind::Incomplete,
            ..
        }
    ));
    assert!(err.is_retryable());
    // The lazily opened text block carried the delta.
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Ok(StreamEvent::TextDelta { text, .. }) if text == "partial"))
    );
}

#[tokio::test]
async fn protocol_violations_fail() {
    let unknown = run(&[
        created(),
        json!({"type": "response.function_call_arguments.delta", "output_index": 3, "delta": "{}"}),
    ])
    .await;
    assert!(matches!(
        unknown.last().unwrap().as_ref().unwrap_err(),
        Error::Stream {
            kind: StreamErrorKind::Protocol,
            ..
        }
    ));
    let invalid = stream_from_sse_text("data: {not json\n\n")
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(
        invalid[0].as_ref().unwrap_err(),
        Error::Stream {
            kind: StreamErrorKind::Decode,
            ..
        }
    ));
}

#[tokio::test]
async fn request_warnings_follow_response_start() {
    let warning = Warning::new(WarningKind::Unsupported, Subject::Response, "seed ignored");
    let mut context = test_context(Some("req_1"));
    context.warnings = vec![warning.clone()];
    let state = SseState::new(context);
    let events: Vec<_> = stream_with(&sse(&[created(), completed("completed", json!({}))]), state)
        .collect()
        .await;
    assert!(matches!(events[0], Ok(StreamEvent::ResponseStart { .. })));
    assert_eq!(events[1].as_ref().unwrap(), &StreamEvent::warning(warning));
    let response = collect(&events).unwrap();
    assert_eq!(response.request_id.as_deref(), Some("req_1"));
    assert_eq!(response.id.as_deref(), Some("resp_1"));
}

/// vLLM / OpenRouter (gpt-oss): raw reasoning as `reasoning_text` content
/// parts and `response.reasoning_text.delta`, no summary.
#[tokio::test]
async fn raw_reasoning_content_streams_as_reasoning_text() {
    let reasoning_done = json!({"type": "reasoning", "id": "rs_1", "summary": [],
        "content": [{"type": "reasoning_text", "text": "User wants a greeting."}]});
    let message_done = json!({"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": "Hello!", "annotations": []}]});
    let events = run(&[
        created(),
        json!({"type": "response.output_item.added", "output_index": 0,
               "item": {"type": "reasoning", "id": "rs_1", "summary": [], "content": []}}),
        json!({"type": "response.content_part.added", "output_index": 0, "item_id": "rs_1", "content_index": 0,
               "part": {"type": "reasoning_text", "text": ""}}),
        json!({"type": "response.reasoning_text.delta", "output_index": 0, "item_id": "rs_1", "content_index": 0, "delta": "User wants "}),
        json!({"type": "response.reasoning_text.delta", "output_index": 0, "item_id": "rs_1", "content_index": 0, "delta": "a greeting."}),
        json!({"type": "response.reasoning_text.done", "output_index": 0, "item_id": "rs_1", "content_index": 0, "text": "User wants a greeting."}),
        json!({"type": "response.content_part.done", "output_index": 0, "item_id": "rs_1", "content_index": 0,
               "part": {"type": "reasoning_text", "text": "User wants a greeting."}}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": reasoning_done.clone()}),
        json!({"type": "response.output_item.added", "output_index": 1,
               "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": []}}),
        json!({"type": "response.output_text.delta", "output_index": 1, "item_id": "msg_1", "content_index": 0, "delta": "Hello!"}),
        json!({"type": "response.output_item.done", "output_index": 1, "item": message_done.clone()}),
        completed("completed", json!({"output": [reasoning_done, message_done]})),
    ])
    .await;
    let response = collect(&events).unwrap();
    assert_eq!(
        response.reasoning_text().as_deref(),
        Some("User wants a greeting.")
    );
    assert_eq!(response.text().as_deref(), Some("Hello!"));
    let ContentBlock::Reasoning(reasoning) = &response.content[0] else {
        panic!("expected reasoning");
    };
    assert_eq!(
        reasoning
            .provider_data
            .get_raw(TEST_NAMESPACE, "text_source"),
        Some(&json!("content"))
    );
    assert!(
        reasoning
            .provider_data
            .get_raw(TEST_NAMESPACE, "content")
            .is_some()
    );
}

/// llama.cpp: no `output_index` or `sequence_number`; items are identified
/// by `item_id` only, and carry `encrypted_content: ""`.
#[tokio::test]
async fn items_without_output_index_are_correlated_by_item_id() {
    let events = run(&[
        created(),
        json!({"type": "response.output_item.added",
               "item": {"type": "reasoning", "id": "rs_a", "summary": [], "encrypted_content": ""}}),
        json!({"type": "response.reasoning_text.delta", "item_id": "rs_a", "delta": "thinking"}),
        json!({"type": "response.output_item.done",
               "item": {"type": "reasoning", "id": "rs_a", "summary": [], "encrypted_content": "",
                        "content": [{"type": "reasoning_text", "text": "thinking"}]}}),
        json!({"type": "response.output_item.added",
               "item": {"type": "function_call", "id": "fc_a", "call_id": "call_a", "name": "get_weather", "arguments": ""}}),
        json!({"type": "response.function_call_arguments.delta", "item_id": "fc_a", "delta": "{\"city\":"}),
        json!({"type": "response.function_call_arguments.delta", "item_id": "fc_a", "delta": "\"Paris\"}"}),
        json!({"type": "response.output_item.done",
               "item": {"type": "function_call", "id": "fc_a", "call_id": "call_a", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}),
        completed("completed", json!({})),
    ])
    .await;
    let response = collect(&events).unwrap();
    assert_eq!(response.reasoning_text().as_deref(), Some("thinking"));
    let call = response.tool_calls().next().unwrap();
    assert_eq!(call.arguments, "{\"city\":\"Paris\"}");
    let ContentBlock::Reasoning(reasoning) = &response.content[0] else {
        panic!("expected reasoning");
    };
    assert!(
        reasoning
            .provider_data
            .get_raw(TEST_NAMESPACE, "encrypted_content")
            .is_none(),
        "empty encrypted content is not stored"
    );
}

#[tokio::test]
async fn null_deltas_are_empty() {
    let events = run(&[
        created(),
        json!({"type": "response.output_item.added", "output_index": 0,
               "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": []}}),
        json!({"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": null}),
        json!({"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "ok"}),
        json!({"type": "response.output_item.done", "output_index": 0,
               "item": {"type": "message", "id": "msg_1", "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok", "annotations": []}]}}),
        completed("completed", json!({})),
    ])
    .await;
    assert_eq!(collect(&events).unwrap().text().as_deref(), Some("ok"));
}

#[tokio::test]
async fn streamed_mcp_calls_keep_what_replay_needs() {
    let item = json!({"type": "mcp_call", "id": "mcp_1", "name": "roll", "arguments": "{\"sides\":6}",
                      "server_label": "dice", "output": "4", "error": null});
    let events = run(&[
        created(),
        json!({"type": "response.output_item.added", "output_index": 0,
               "item": {"type": "mcp_call", "id": "mcp_1", "name": "roll", "server_label": "dice"}}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": item}),
        completed("completed", json!({})),
    ])
    .await;
    let response = collect(&events).unwrap();
    let ContentBlock::ProviderToolCall(call) = &response.content[0] else {
        panic!("{:?}", response.content)
    };
    let ContentBlock::ProviderToolResult(result) = &response.content[1] else {
        panic!("{:?}", response.content)
    };
    let rebuilt = super::super::hosted::encode(call, Some(result), TEST_NAMESPACE).unwrap();
    assert_eq!(serde_json::Value::Object(rebuilt), item);
}

#[tokio::test]
async fn unannounced_messages_open_only_with_an_item_id() {
    let events = run(&[
        created(),
        json!({"type": "response.output_text.delta", "output_index": 0, "item_id": "msg_1", "content_index": 0, "delta": "Hi"}),
        json!({"type": "response.output_item.done", "output_index": 0,
               "item": {"type": "message", "id": "msg_1", "content": [{"type": "output_text", "text": "Hi", "annotations": []}]}}),
        completed("completed", json!({})),
    ])
    .await;
    assert_eq!(collect(&events).unwrap().text().as_deref(), Some("Hi"));

    let anonymous = run(&[
        created(),
        json!({"type": "response.output_text.delta", "output_index": 3, "content_index": 0, "delta": "Hi"}),
    ])
    .await;
    assert!(
        anonymous.iter().any(|e| matches!(
            e,
            Err(Error::Stream {
                kind: allama::StreamErrorKind::Protocol,
                ..
            })
        )),
        "{anonymous:?}"
    );
}

#[tokio::test]
async fn client_executed_calls_stream_as_other_and_wait_for_the_caller() {
    let call = json!({"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1", "name": "sql", "input": "SELECT 1", "status": "completed"});
    let events = run(&[
        created(),
        json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1", "name": "sql", "input": "", "status": "in_progress"}}),
        json!({"type": "response.custom_tool_call_input.delta", "output_index": 0, "item_id": "ctc_1", "delta": "SELECT 1"}),
        json!({"type": "response.custom_tool_call_input.done", "output_index": 0, "item_id": "ctc_1", "input": "SELECT 1"}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": call}),
        completed("completed", json!({})),
    ])
    .await;
    let response = collect(&events).unwrap();
    assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
    assert!(response.tool_calls().next().is_none());
    let pending = crate::ClientToolCall::pending(&response.content);
    assert!(
        matches!(&pending[..], [crate::ClientToolCall::Custom(c)] if c.input == "SELECT 1" && c.origin.as_deref() == Some(TEST_NAMESPACE)),
        "{pending:?}"
    );

    // A hosted shell call answered in the same response waits for nothing.
    let events = run(&[
        created(),
        json!({"type": "response.output_item.done", "output_index": 0, "item": {"type": "shell_call", "id": "sh_1", "call_id": "call_2", "status": "completed", "action": {"commands": ["ls"]}}}),
        json!({"type": "response.output_item.done", "output_index": 1, "item": {"type": "shell_call_output", "id": "sho_1", "call_id": "call_2", "output": []}}),
        completed("completed", json!({})),
    ])
    .await;
    assert_eq!(
        collect(&events).unwrap().finish_reason,
        Some(FinishReason::Stop)
    );
}
