//! Structured output is JSON on both paths, for Chat Completions and
//! Responses, and OpenAI's refusals of what a model does not take are
//! `Error::Unsupported`. (In-band stream errors are classified by the
//! aggregates in `conformance.rs`.)

use allama::{
    ChatProvider, ChatRequest, ContentPart, Error, ReasoningEffort, RequestField, ResponseFormat,
    ToolDefinition, UnsupportedStage,
};
use allama_conformance::contract::{
    assert_structured_output_contract, assert_structured_output_contract_stream,
};
use allama_http::{ReplayResponse, ReplayTransport};
use allama_openai::{Provider, ResponsesProvider};
use serde_json::json;

fn chat_provider(transport: ReplayTransport) -> Provider {
    Provider::builder()
        .api_key("k")
        .http_transport(transport)
        .build()
        .unwrap()
}

fn responses_provider(transport: ReplayTransport) -> ResponsesProvider {
    ResponsesProvider::builder()
        .api_key("k")
        .http_transport(transport)
        .build()
        .unwrap()
}

/// `response.created`, so the Responses stream has started.
fn response_created() -> String {
    let event = json!({
        "type": "response.created",
        "response": {"id": "resp_1", "object": "response", "status": "in_progress", "model": "gpt-4.1", "output": []},
        "sequence_number": 0
    });
    format!("event: response.created\ndata: {event}\n\n")
}

fn structured_request(model: &str) -> ChatRequest {
    ChatRequest::new(model).user("Which city?").response_format(
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
        "id": "c1", "object": "chat.completion", "model": "gpt-4o",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": ANSWER}}]
    });
    let chunk = json!({
        "id": "c1", "object": "chat.completion.chunk", "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ANSWER}, "finish_reason": "stop"}]
    });
    let provider = chat_provider(ReplayTransport::new([
        ReplayResponse::json(200, response.to_string()),
        ReplayResponse::sse(format!("data: {chunk}\n\ndata: [DONE]\n\n")),
    ]));
    assert_structured_output_contract(&provider, &structured_request("gpt-4o")).await;
    assert_structured_output_contract_stream(&provider, &structured_request("gpt-4o")).await;
}

#[tokio::test]
async fn responses_structured_output_is_json_on_both_paths() {
    let message = json!({
        "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": ANSWER, "annotations": []}]
    });
    let response = json!({
        "id": "resp_1", "object": "response", "status": "completed", "model": "gpt-4.1",
        "output": [message],
        "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10}
    });
    let events = [
        json!({"type": "response.output_item.added", "output_index": 0,
               "item": {"type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": []}}),
        json!({"type": "response.content_part.added", "item_id": "msg_1", "output_index": 0, "content_index": 0,
               "part": {"type": "output_text", "text": "", "annotations": []}}),
        json!({"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "content_index": 0,
               "delta": ANSWER}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": message}),
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
    let provider = responses_provider(ReplayTransport::new([
        ReplayResponse::json(200, response.to_string()),
        ReplayResponse::sse(sse),
    ]));
    assert_structured_output_contract(&provider, &structured_request("gpt-4.1")).await;
    assert_structured_output_contract_stream(&provider, &structured_request("gpt-4.1")).await;
}

/// Chat Completions refuses an audio part for a model without
/// audio input with a structured `invalid_value` on `messages[N]`,
/// recorded live from `gpt-5-mini`. It is `Error::Unsupported` with stage
/// `Provider`, so an application can fall back (transcribe, switch
/// model). The refusal names no message: `N` counts the system prompt.
#[tokio::test]
async fn chat_completions_content_part_refusal_is_unsupported() {
    let refusal = |index: usize| {
        ReplayResponse::json(
            400,
            json!({"error": {
                "message": format!("Invalid 'messages[{index}]'. Content blocks are expected to be either text or image_url type."),
                "type": "invalid_request_error",
                "param": format!("messages[{index}]"),
                "code": "invalid_value"
            }})
            .to_string(),
        )
    };
    let audio = ChatRequest::new("gpt-5-mini").user(vec![
        ContentPart::text("Transcribe this."),
        ContentPart::file_base64("audio/wav", "UklGRiQAAABXQVZF"),
    ]);
    let provider = chat_provider(ReplayTransport::new([refusal(0), refusal(1)]));
    for request in [audio.clone(), audio.system("You are concise.")] {
        let err = provider.chat(&request).await.unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
        assert_eq!(err.unsupported_stage(), Some(UnsupportedStage::Provider));
        assert_eq!(err.unsupported_subject(), None);
    }
}

/// Chat Completions refuses function tools with a
/// reasoning effort on `gpt-6-astra` (recorded live, `code: null`). The
/// reply's remedy on this endpoint is the effort `none`, so the refusal is
/// `Error::Unsupported` with stage `Provider` naming the reasoning field,
/// and a fallback fails over instead of stopping at an invalid request.
#[tokio::test]
async fn chat_completions_tools_with_reasoning_refusal_names_the_reasoning() {
    let refusal = ReplayResponse::json(
        400,
        json!({"error": {
            "message": "Function tools with reasoning_effort are not supported for gpt-6-astra in /v1/chat/completions. To use function tools, use /v1/responses or set reasoning_effort to 'none'.",
            "type": "invalid_request_error",
            "param": "reasoning_effort",
            "code": null
        }})
        .to_string(),
    );
    let tool = ToolDefinition::new(
        "get_weather",
        json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
    );
    let request = ChatRequest::new("gpt-6-astra")
        .user("Weather in Paris?")
        .tools([tool])
        .reasoning(ReasoningEffort::Low);
    let transport = ReplayTransport::new([refusal]);
    let err = chat_provider(transport.clone())
        .chat(&request)
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
    assert_eq!(err.unsupported_stage(), Some(UnsupportedStage::Provider));
    assert_eq!(err.unsupported_field(), Some(RequestField::Reasoning));
    assert!(allama::wrappers::should_fallback(&err));
    let sent = transport.requests();
    let body: serde_json::Value =
        serde_json::from_slice(sent[0].body.as_ref().expect("a request body")).unwrap();
    assert_eq!(body["reasoning_effort"], "low");
    assert_eq!(body["tools"][0]["function"]["name"], "get_weather");
}
