use super::*;
use allama::chat::stream::StreamCollector;
use allama::{ChatResponse, ContentBlock, FinishReason, Subject, WarningKind};
use futures_util::StreamExt;
use serde_json::json;

/// Feed `(event, data)` pairs through a fresh state.
fn run(events: &[(&str, Value)]) -> Vec<Result<StreamEvent>> {
    let mut state = SseState::new(Some("req_1".into()), Vec::new());
    let mut out = Vec::new();
    for (event, data) in events {
        out.extend(state.process(event, &data.to_string()));
    }
    out
}

fn collect(events: &[Result<StreamEvent>]) -> ChatResponse {
    let mut collector = StreamCollector::new();
    for event in events {
        collector
            .push(event.clone().expect("no stream errors"))
            .unwrap();
    }
    collector.finish().unwrap()
}

fn message_start(usage: Value) -> (&'static str, Value) {
    (
        "message_start",
        json!({"type": "message_start", "message": {"id": "msg_1", "model": "claude-sonnet-4-5", "usage": usage}}),
    )
}

fn block_start(index: usize, block: Value) -> (&'static str, Value) {
    (
        "content_block_start",
        json!({"type": "content_block_start", "index": index, "content_block": block}),
    )
}

fn delta(index: usize, delta: Value) -> (&'static str, Value) {
    (
        "content_block_delta",
        json!({"type": "content_block_delta", "index": index, "delta": delta}),
    )
}

fn block_stop(index: usize) -> (&'static str, Value) {
    (
        "content_block_stop",
        json!({"type": "content_block_stop", "index": index}),
    )
}

fn message_delta(stop_reason: &str, usage: Value) -> (&'static str, Value) {
    (
        "message_delta",
        json!({"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": null}, "usage": usage}),
    )
}

fn message_stop() -> (&'static str, Value) {
    ("message_stop", json!({"type": "message_stop"}))
}

#[test]
fn text_stream_collects_with_metadata() {
    let events = run(&[
        message_start(json!({"input_tokens": 100, "output_tokens": 1})),
        block_start(0, json!({"type": "text", "text": ""})),
        delta(0, json!({"type": "text_delta", "text": "Hello"})),
        delta(0, json!({"type": "text_delta", "text": " world"})),
        block_stop(0),
        message_delta("end_turn", json!({"output_tokens": 42})),
        message_stop(),
    ]);
    assert!(matches!(
        &events[0],
        Ok(StreamEvent::ResponseStart { id, model, .. })
            if id.as_deref() == Some("msg_1") && model.as_deref() == Some("claude-sonnet-4-5")
    ));
    let response = collect(&events);
    assert_eq!(response.text().as_deref(), Some("Hello world"));
    assert_eq!(response.finish_reason, Some(FinishReason::Stop));
    assert_eq!(response.finish_reason_raw.as_deref(), Some("end_turn"));
    assert_eq!(response.provider.as_deref(), Some("anthropic"));
    assert_eq!(response.request_id.as_deref(), Some("req_1"));
    let usage = response.usage.unwrap();
    assert_eq!(usage.input_tokens, Some(100));
    assert_eq!(usage.output_tokens, Some(42));
    assert_eq!(usage.total_tokens, Some(142));
}

#[test]
fn streaming_usage_keeps_input_and_cache_counts() {
    let events = run(&[
        message_start(json!({
            "input_tokens": 12,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 50,
            "output_tokens": 1
        })),
        block_start(0, json!({"type": "text"})),
        delta(0, json!({"type": "text_delta", "text": "x"})),
        block_stop(0),
        message_delta(
            "end_turn",
            json!({"output_tokens": 30, "output_tokens_details": {"thinking_tokens": 10}}),
        ),
        message_stop(),
    ]);
    let usage = collect(&events).usage.unwrap();
    assert_eq!(usage.input_tokens, Some(162));
    assert_eq!(usage.cache_read_input_tokens, Some(100));
    assert_eq!(usage.cache_write_input_tokens, Some(50));
    assert_eq!(usage.output_tokens, Some(30));
    assert_eq!(usage.reasoning_tokens, Some(10));
    assert_eq!(usage.total_tokens, Some(192));
    allama_conformance::assert_usage_contract(&usage);
    // Every metadata event is a snapshot.
    assert!(events.iter().all(|e| !matches!(
        e,
        Ok(StreamEvent::ResponseMetadata {
            usage_mode: UsageMetadataMode::Delta,
            ..
        })
    )));
}

#[test]
fn tool_use_stream() {
    let events = run(&[
        message_start(json!({"input_tokens": 10})),
        block_start(
            0,
            json!({"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {}, "caller": {"type": "direct"}}),
        ),
        delta(0, json!({"type": "input_json_delta", "partial_json": ""})),
        delta(
            0,
            json!({"type": "input_json_delta", "partial_json": "{\"path\": \"foo"}),
        ),
        delta(
            0,
            json!({"type": "input_json_delta", "partial_json": ".txt\"}"}),
        ),
        block_stop(0),
        message_delta("tool_use", json!({"output_tokens": 30})),
        message_stop(),
    ]);
    let response = collect(&events);
    let call = response.first_tool_call().unwrap();
    assert_eq!(call.id, "toolu_1");
    assert_eq!(call.name, "read_file");
    assert_eq!(call.arguments, r#"{"path": "foo.txt"}"#);
    assert!(call.provider_data.is_empty());
    assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
}

#[test]
fn computer_toolset_call_streams_its_toolset_name() {
    // `computer_toolset_20260801` members carry `toolset_name` beside the
    // member name; it must survive for the `tool_result` to repeat it.
    let events = run(&[
        message_start(json!({"input_tokens": 10})),
        block_start(
            0,
            json!({"type": "tool_use", "id": "toolu_1", "name": "left_click", "toolset_name": "computer", "input": {}}),
        ),
        delta(
            0,
            json!({"type": "input_json_delta", "partial_json": "{\"coordinate\": [512, 742]}"}),
        ),
        block_stop(0),
        message_delta("tool_use", json!({"output_tokens": 30})),
        message_stop(),
    ]);
    let response = collect(&events);
    let call = response.first_tool_call().unwrap();
    assert_eq!(
        (call.id.as_str(), call.name.as_str()),
        ("toolu_1", "left_click")
    );
    assert_eq!(call.arguments, r#"{"coordinate": [512, 742]}"#);
    assert_eq!(
        call.provider_data.namespace(NAMESPACE),
        Some(
            &json!({"toolset_name": "computer"})
                .as_object()
                .unwrap()
                .clone()
        )
    );
}

#[test]
fn thinking_stream_carries_signature_as_provider_data() {
    let events = run(&[
        message_start(json!({"input_tokens": 10})),
        block_start(
            0,
            json!({"type": "thinking", "thinking": "", "signature": ""}),
        ),
        delta(
            0,
            json!({"type": "thinking_delta", "thinking": "Let me think"}),
        ),
        delta(
            0,
            json!({"type": "signature_delta", "signature": "sig_abc"}),
        ),
        block_stop(0),
        block_start(1, json!({"type": "redacted_thinking", "data": "opaque"})),
        block_stop(1),
        block_start(2, json!({"type": "text", "text": ""})),
        delta(2, json!({"type": "text_delta", "text": "Answer"})),
        block_stop(2),
        message_delta("end_turn", json!({"output_tokens": 5})),
        message_stop(),
    ]);
    assert!(events.iter().any(|e| matches!(
        e,
        Ok(StreamEvent::BlockProviderData { index: 0, provider_data, .. })
            if provider_data.get_raw("anthropic", "signature") == Some(&json!("sig_abc"))
    )));
    let response = collect(&events);
    let thinking = response.content[0].as_reasoning().unwrap();
    assert_eq!(thinking.text, "Let me think");
    assert_eq!(
        thinking.provider_data.get_raw("anthropic", "signature"),
        Some(&json!("sig_abc"))
    );
    let redacted = response.content[1].as_reasoning().unwrap();
    assert!(redacted.is_opaque());
    assert_eq!(
        redacted.provider_data.get_raw("anthropic", "redacted_data"),
        Some(&json!("opaque"))
    );
}

#[test]
fn server_tool_blocks_stream_as_provider_tool_blocks() {
    let events = run(&[
        message_start(json!({"input_tokens": 10})),
        block_start(
            0,
            json!({"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}}),
        ),
        delta(
            0,
            json!({"type": "input_json_delta", "partial_json": "{\"query\":"}),
        ),
        delta(
            0,
            json!({"type": "input_json_delta", "partial_json": "\"rust\"}"}),
        ),
        block_stop(0),
        block_start(
            1,
            json!({"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [{"type": "web_search_result", "url": "https://rust-lang.org"}]}),
        ),
        block_stop(1),
        block_start(2, json!({"type": "text", "text": ""})),
        delta(2, json!({"type": "text_delta", "text": "Rust"})),
        delta(
            2,
            json!({"type": "citations_delta", "citation": {"type": "web_search_result_location", "cited_text": "Rust"}}),
        ),
        block_stop(2),
        message_delta(
            "end_turn",
            json!({"output_tokens": 5, "server_tool_use": {"web_search_requests": 1}}),
        ),
        message_stop(),
    ]);
    let response = collect(&events);
    let ContentBlock::ProviderToolCall(call) = &response.content[0] else {
        panic!(
            "expected a provider tool call, got {:?}",
            response.content[0]
        );
    };
    assert_eq!(call.name, "web_search");
    assert_eq!(call.origin.as_deref(), Some("anthropic"));
    assert_eq!(call.arguments, r#"{"query":"rust"}"#);
    let ContentBlock::ProviderToolResult(result) = &response.content[1] else {
        panic!("expected a provider tool result");
    };
    assert_eq!(result.call_id, "srvtoolu_1");
    assert_eq!(
        result.provider_data.get_raw("anthropic", "type"),
        Some(&json!("web_search_tool_result"))
    );
    let ContentBlock::Text(text) = &response.content[2] else {
        panic!("expected text");
    };
    assert_eq!(
        text.provider_data.get_raw("anthropic", "citations"),
        Some(&json!([{"type": "web_search_result_location", "cited_text": "Rust"}]))
    );
    assert_eq!(
        response.usage.unwrap().tool_requests.get("web_search"),
        Some(&1)
    );
}

#[test]
fn unknown_delta_types_surface_as_provider_deltas() {
    let events = run(&[
        message_start(json!({"input_tokens": 1})),
        block_start(0, json!({"type": "text", "text": ""})),
        delta(0, json!({"type": "brand_new_delta", "x": 1})),
        block_stop(0),
        message_delta("end_turn", json!({"output_tokens": 1})),
        message_stop(),
    ]);
    let deltas: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Ok(StreamEvent::ProviderDelta {
                index,
                type_name,
                data,
                origin,
                ..
            }) => Some((*index, type_name.clone(), data.clone(), origin.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(deltas.len(), 1);
    let (index, type_name, data, origin) = &deltas[0];
    assert_eq!((*index, type_name.as_str()), (0, "brand_new_delta"));
    assert_eq!(data.get("x"), Some(&json!(1)));
    assert_eq!(origin.as_deref(), Some("anthropic"));
    // Observation only: the collected response is unaffected.
    let response = collect(&events);
    assert!(response.warnings.is_empty());
}

#[test]
fn stop_sequence_refusal_and_pause_turn() {
    let events = run(&[
        message_start(json!({"input_tokens": 1})),
        (
            "message_delta",
            json!({"type": "message_delta", "delta": {"stop_reason": "stop_sequence", "stop_sequence": "END"}, "usage": {"output_tokens": 2}}),
        ),
        message_stop(),
    ]);
    let response = collect(&events);
    assert_eq!(response.finish_reason, Some(FinishReason::Stop));
    assert_eq!(response.finish_reason_raw.as_deref(), Some("stop_sequence"));
    assert_eq!(
        response.provider_data.get_raw("anthropic", "stop_sequence"),
        Some(&json!("END"))
    );

    let events = run(&[
        message_start(json!({"input_tokens": 1})),
        message_delta("refusal", json!({"output_tokens": 0})),
        message_stop(),
    ]);
    assert_eq!(collect(&events).finish_reason, Some(FinishReason::Refusal));

    let events = run(&[
        message_start(json!({"input_tokens": 1})),
        message_delta("pause_turn", json!({"output_tokens": 0})),
        message_stop(),
    ]);
    assert_eq!(
        collect(&events).finish_reason,
        Some(FinishReason::Other("pause_turn".into()))
    );
}

#[test]
fn mid_stream_errors_are_typed() {
    for (error_type, check) in [
        (
            "overloaded_error",
            (|e: &Error| matches!(e, Error::Overloaded { .. })) as fn(&Error) -> bool,
        ),
        ("rate_limit_error", |e| {
            matches!(e, Error::RateLimited { .. })
        }),
        ("authentication_error", |e| matches!(e, Error::Auth { .. })),
        ("api_error", |e| matches!(e, Error::Provider { .. })),
    ] {
        let events = run(&[
            message_start(json!({"input_tokens": 1})),
            (
                "error",
                json!({"type": "error", "error": {"type": error_type, "message": "boom"}}),
            ),
        ]);
        let err = events.last().unwrap().as_ref().unwrap_err();
        assert!(check(err), "{error_type}: {err:?}");
        assert_eq!(err.request_id(), Some("req_1"));
    }

    let events = run(&[("error", json!({"type": "error", "unexpected": true}))]);
    assert!(matches!(
        events.last().unwrap(),
        Err(Error::Stream {
            kind: StreamErrorKind::Protocol,
            ..
        })
    ));
}

#[test]
fn invalid_json_is_a_decode_error() {
    let mut state = SseState::new(None, Vec::new());
    let events = state.process("message_start", "{not json");
    assert!(matches!(
        events.last().unwrap(),
        Err(Error::Stream {
            kind: StreamErrorKind::Decode,
            ..
        })
    ));
}

#[test]
fn request_warnings_are_emitted_after_response_start() {
    let warning = Warning::new(
        WarningKind::Unsupported,
        Subject::Field {
            field: allama::RequestField::Seed,
        },
        "no seed",
    );
    let mut state = SseState::new(None, vec![warning.clone()]);
    let events = state.process(
        "message_start",
        &json!({"type": "message_start", "message": {"id": "m", "model": "x", "usage": {"input_tokens": 1}}})
            .to_string(),
    );
    assert!(matches!(events[0], Ok(StreamEvent::ResponseStart { .. })));
    assert!(matches!(&events[1], Ok(StreamEvent::Warning { warning: w, .. }) if *w == warning));
}

#[tokio::test]
async fn truncated_stream_reports_incomplete_without_fake_block_stops() {
    let sse = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"x\",\"usage\":{\"input_tokens\":1}}}\n\n\
event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n";
    let mut stream = crate::conformance_stream_from_sse_text(sse);
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, Ok(StreamEvent::BlockStop { .. })))
    );
    assert!(matches!(
        events.last().unwrap(),
        Err(Error::Stream {
            kind: StreamErrorKind::Incomplete,
            ..
        })
    ));
    assert!(events.last().unwrap().as_ref().unwrap_err().is_retryable());
}

/// A failed body read is classified by its kind: a stream timeout stays a
/// timeout, with the provider and the request id, and nothing follows it.
#[tokio::test]
async fn a_stream_timeout_stays_a_timeout() {
    use allama_http::sse::{SseChatStream, SseStream};
    use allama_http::{TransportError, TransportErrorKind};

    let start = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"x\",\"usage\":{\"input_tokens\":1}}}\n\n";
    for (kind, timeout) in [
        (TransportErrorKind::Timeout, true),
        (TransportErrorKind::Body, false),
    ] {
        let body = futures_util::stream::iter([
            Ok(start.as_bytes().to_vec()),
            Err(TransportError::new(kind, "the wait passed 30s")),
            Ok(start.as_bytes().to_vec()),
        ]);
        let state = SseState::new(Some("req_1".into()), Vec::new());
        let items: Vec<_> = SseChatStream::new(SseStream::new(body), state)
            .collect()
            .await;
        let error = items.last().unwrap().as_ref().unwrap_err();
        assert_eq!(matches!(error, Error::Timeout { .. }), timeout, "{error:?}");
        assert!(error.is_retryable());
        assert_eq!(error.request_id(), Some("req_1"));
        assert_eq!(
            error.context().unwrap().provider.as_deref(),
            Some(PROVIDER_NAME)
        );
        assert_eq!(error.upstream_message(), Some("the wait passed 30s"));
        assert_eq!(
            items
                .iter()
                .filter(|item| matches!(item, Ok(StreamEvent::ResponseStart { .. })))
                .count(),
            1
        );
    }
}

/// A hostile wire index never reaches the collector.
/// Blocks get dense local indices in start order, so a stream with index
/// `u64::MAX` collects like any other.
#[test]
fn wire_block_indices_are_remapped_to_dense_ones() {
    let huge = usize::MAX;
    let events = run(&[
        message_start(json!({"input_tokens": 1, "output_tokens": 0})),
        block_start(7, json!({"type": "text", "text": ""})),
        delta(7, json!({"type": "text_delta", "text": "Hi"})),
        block_stop(7),
        block_start(
            huge,
            json!({"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}}),
        ),
        delta(
            huge,
            json!({"type": "input_json_delta", "partial_json": "{}"}),
        ),
        block_stop(huge),
        message_delta("tool_use", json!({"output_tokens": 3})),
        ("message_stop", json!({"type": "message_stop"})),
    ]);
    let indices: Vec<usize> = events
        .iter()
        .filter_map(|event| match event {
            Ok(StreamEvent::BlockStart { index, .. }) => Some(*index),
            _ => None,
        })
        .collect();
    assert_eq!(indices, vec![0, 1]);
    let response = collect(&events);
    assert_eq!(response.text().as_deref(), Some("Hi"));
    assert_eq!(response.tool_calls().count(), 1);
}

#[test]
fn a_delta_for_a_block_that_never_started_is_a_protocol_error() {
    let events = run(&[
        message_start(json!({"input_tokens": 1, "output_tokens": 0})),
        delta(3, json!({"type": "text_delta", "text": "Hi"})),
    ]);
    let error = events
        .iter()
        .find_map(|event| event.as_ref().err())
        .expect("a protocol error");
    assert_eq!(error.kind_name(), "stream");
    assert!(error.message().contains("not started"), "{error}");
}

#[test]
fn a_block_started_twice_is_a_protocol_error() {
    let events = run(&[
        message_start(json!({"input_tokens": 1, "output_tokens": 0})),
        block_start(0, json!({"type": "text", "text": ""})),
        block_start(0, json!({"type": "text", "text": ""})),
    ]);
    assert!(events.iter().any(|event| {
        event
            .as_ref()
            .is_err_and(|e| e.message().contains("already open"))
    }));
}
