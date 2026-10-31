//! The generic Responses provider against a local HTTP server.

// Sends through the default `reqwest` transport (a local test server).
#![cfg(feature = "reqwest")]

use allama::prelude::*;
use allama::{
    ChatStreamExt, ContentBlock, FinishReason, ProviderIdentity, SystemPrompt, WarningKind,
};
use allama_conformance::{MockHttpResponse, TestHttpServer};
use allama_http::http::{HeaderValue, StatusCode};
use allama_openai_compat::{
    ResponsesProvider, ResponsesQuirks, ResponsesRequestOptions, StreamingSupport, SystemOptions,
};
use serde_json::{Value, json};

const NAMESPACE: &str = "openai-compat:local";

/// A reasoning turn as Ollama returns it: the thinking in both the summary
/// and `encrypted_content`, then a function call.
fn reasoning_tool_body() -> Value {
    json!({
        "id": "resp_1", "object": "response", "status": "completed", "model": "qwen3:0.6b",
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "Need the weather."}],
             "encrypted_content": "Need the weather."},
            {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_weather",
             "arguments": "{\"city\":\"Paris\"}", "status": "completed"}
        ],
        "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30,
                  "output_tokens_details": {"reasoning_tokens": 0}}
    })
}

fn text_body() -> Value {
    json!({
        "id": "resp_2", "object": "response", "status": "completed", "model": "qwen3:0.6b",
        "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "Sunny.", "annotations": []}]}],
        "usage": {"input_tokens": 30, "output_tokens": 2, "total_tokens": 32}
    })
}

fn provider(server: &TestHttpServer, quirks: ResponsesQuirks) -> ResponsesProvider {
    ResponsesProvider::builder()
        .base_url(server.url())
        .bearer_token("local-key")
        .provider_name("local")
        .quirks(quirks)
        .build()
        .unwrap()
}

fn weather_request() -> ChatRequest {
    ChatRequest::new("qwen3:0.6b")
        .tools([ToolDefinition::new(
            "get_weather",
            json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
        )])
        .user("Weather in Paris?")
}

#[tokio::test]
async fn identity_namespace_and_stateless_replay() {
    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(StatusCode::OK, &reasoning_tool_body())
            .header("x-request-id", HeaderValue::from_static("req_1")),
        MockHttpResponse::json_value(StatusCode::OK, &text_body()),
    ])
    .await;
    let provider = provider(&server, ResponsesQuirks::new(NAMESPACE));
    assert_eq!(provider.provider_name(), "local");

    let mut request = weather_request();
    let response = provider.chat(&request).await.unwrap();
    assert_eq!(response.provider.as_deref(), Some("local"));
    assert_eq!(response.request_id.as_deref(), Some("req_1"));
    assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
    let ContentBlock::Reasoning(reasoning) = &response.content[0] else {
        panic!("expected reasoning, got {:?}", response.content)
    };
    assert_eq!(reasoning.text, "Need the weather.");
    assert_eq!(
        reasoning
            .provider_data
            .get_raw(NAMESPACE, "encrypted_content"),
        Some(&json!("Need the weather."))
    );
    assert!(reasoning.provider_data.namespace("openai").is_none());

    request.push_message(response.to_assistant_message());
    request.push_tool_result(response.first_tool_call().unwrap(), "Sunny");
    let next = provider.chat(&request).await.unwrap();
    assert!(next.warnings.is_empty(), "{:?}", next.warnings);
    assert_eq!(next.text().as_deref(), Some("Sunny."));

    let sent = server.recorded_requests().await;
    assert_eq!(sent[0].path, "/responses");
    assert_eq!(sent[0].header("authorization"), Some("Bearer local-key"));
    let first = sent[0].body_json();
    assert_eq!(first["store"], false);
    assert_eq!(first["tools"][0]["strict"], false);
    let replay = sent[1].body_json();
    assert!(replay.get("previous_response_id").is_none());
    assert_eq!(
        replay["input"][1],
        json!({"type": "reasoning", "id": "rs_1", "encrypted_content": "Need the weather.",
               "summary": [{"type": "summary_text", "text": "Need the weather."}]})
    );
    assert_eq!(
        replay["input"][2],
        json!({"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_weather",
               "arguments": "{\"city\":\"Paris\"}"})
    );
    assert_eq!(replay["input"][3]["type"], "function_call_output");
}

/// State one endpoint minted is never sent to another, even though both
/// speak the same wire format.
#[tokio::test]
async fn state_from_another_namespace_is_not_replayed() {
    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(StatusCode::OK, &reasoning_tool_body()),
        MockHttpResponse::json_value(StatusCode::OK, &text_body()),
    ])
    .await;
    let local = provider(&server, ResponsesQuirks::new(NAMESPACE));
    let other = provider(&server, ResponsesQuirks::new("openai-compat:other"));

    let mut request = weather_request();
    let response = local.chat(&request).await.unwrap();
    request.push_message(response.to_assistant_message());
    request.push_tool_result(response.first_tool_call().unwrap(), "Sunny");
    let next = other.chat(&request).await.unwrap();
    let kinds: Vec<_> = next.warnings.iter().map(|w| w.kind).collect();
    assert_eq!(kinds, [WarningKind::Dropped, WarningKind::Dropped]);
    assert!(
        next.warnings
            .iter()
            .all(|w| w.provider.as_deref() == Some("local"))
    );

    let replay = server.recorded_requests().await[1].body_json();
    let input = replay["input"].as_array().unwrap();
    assert_eq!(
        input
            .iter()
            .map(|i| i["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["message", "function_call", "function_call_output"]
    );
    assert!(input[1].get("id").is_none());
}

#[tokio::test]
async fn streams_are_stamped_with_identity_and_namespace() {
    let events = [
        json!({"type": "response.created", "response": {"id": "resp_1", "model": "qwen3:0.6b", "status": "in_progress", "output": []}}),
        json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning", "id": "rs_1", "summary": []}}),
        json!({"type": "response.reasoning_summary_text.delta", "output_index": 0, "item_id": "rs_1", "summary_index": 0, "delta": "Hmm."}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": {"type": "reasoning", "id": "rs_1",
               "summary": [{"type": "summary_text", "text": "Hmm."}], "encrypted_content": "Hmm."}}),
        json!({"type": "response.output_item.added", "output_index": 1, "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": []}}),
        json!({"type": "response.content_part.added", "output_index": 1, "item_id": "msg_1", "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}}),
        json!({"type": "response.output_text.delta", "output_index": 1, "item_id": "msg_1", "content_index": 0, "delta": "Hi"}),
        json!({"type": "response.output_item.done", "output_index": 1, "item": {"type": "message", "id": "msg_1", "role": "assistant",
               "content": [{"type": "output_text", "text": "Hi", "annotations": []}]}}),
        json!({"type": "response.completed", "response": {"id": "resp_1", "model": "qwen3:0.6b", "status": "completed", "output": [],
               "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}}}),
    ];
    let sse: String = events
        .iter()
        .map(|e| format!("event: {}\ndata: {e}\n\n", e["type"].as_str().unwrap()))
        .collect();
    let server = TestHttpServer::spawn([MockHttpResponse::sse(sse)]).await;
    let provider = provider(&server, ResponsesQuirks::new(NAMESPACE));
    let response = provider
        .chat_stream(&ChatRequest::new("qwen3:0.6b").user("hi"))
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();
    assert_eq!(response.provider.as_deref(), Some("local"));
    assert_eq!(response.text().as_deref(), Some("Hi"));
    let ContentBlock::Reasoning(reasoning) = &response.content[0] else {
        panic!("expected reasoning, got {:?}", response.content)
    };
    assert_eq!(reasoning.text, "Hmm.");
    assert_eq!(
        reasoning.provider_data.get_raw(NAMESPACE, "item_id"),
        Some(&json!("rs_1"))
    );
    let body = server.recorded_requests().await[0].body_json();
    assert_eq!(body["stream"], true);
    assert_eq!(body["store"], false);
}

#[tokio::test]
async fn quirks_shape_the_request() {
    let server =
        TestHttpServer::spawn([MockHttpResponse::json_value(StatusCode::OK, &text_body())]).await;
    let mut quirks = ResponsesQuirks::new(NAMESPACE);
    quirks.supports_strict = false;
    quirks.function_call_status_null = true;
    let provider = provider(&server, quirks);
    let request = ChatRequest::new("m")
        .system(SystemPrompt::new("Be brief.").with_option(SystemOptions::developer()))
        .tools([ToolDefinition::new("f", json!({"type": "object"})).strict(true)])
        .user("hi")
        .message(Message::assistant_blocks(vec![ContentBlock::tool_call(
            "c", "f", "{}",
        )]))
        .message(Message::tool_result("c", "f", "ok"))
        .with_option(ResponsesRequestOptions::new().truncation("auto"));
    let response = provider.chat(&request).await.unwrap();
    let kinds: Vec<_> = response.warnings.iter().map(|w| w.kind).collect();
    // Developer role and strict tool, both unsupported hints.
    assert_eq!(kinds, [WarningKind::Unsupported, WarningKind::Unsupported]);

    let body = server.recorded_requests().await[0].body_json();
    assert_eq!(body["input"][0]["role"], "system");
    assert!(body["tools"][0].get("strict").is_none());
    assert_eq!(body["input"][2]["status"], Value::Null);
    assert!(body["input"][2].as_object().unwrap().contains_key("status"));
    assert_eq!(body["truncation"], "auto");
}

#[tokio::test]
async fn unsupported_mode_and_errors_never_send_or_are_stamped() {
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(
        StatusCode::UNAUTHORIZED,
        &json!({"error": {"message": "bad key", "type": "invalid_request_error", "code": "invalid_api_key"}}),
    )
    .header("x-request-id", HeaderValue::from_static("req_err"))])
    .await;
    let mut quirks = ResponsesQuirks::new(NAMESPACE);
    quirks.streaming = StreamingSupport::NonStreamingOnly;
    let provider = provider(&server, quirks);
    let request = ChatRequest::new("m").user("hi");

    let Err(err) = provider.chat_stream(&request).await else {
        panic!("streaming must be rejected")
    };
    assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
    assert!(server.recorded_requests().await.is_empty());

    let err = provider.chat(&request).await.unwrap_err();
    assert!(matches!(err, Error::Auth { .. }), "{err:?}");
    assert_eq!(err.request_id(), Some("req_err"));
    assert_eq!(err.context().unwrap().provider.as_deref(), Some("local"));
}

#[tokio::test]
async fn background_responses_are_polled_and_cancelled_by_id() {
    let queued = json!({"id": "resp_bg", "object": "response", "status": "queued", "model": "m", "output": []});
    let mut done = text_body();
    done["id"] = json!("resp_bg");
    let mut cancelled = queued.clone();
    cancelled["status"] = json!("cancelled");
    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(StatusCode::OK, &queued),
        MockHttpResponse::json_value(StatusCode::OK, &done),
        MockHttpResponse::json_value(StatusCode::OK, &cancelled),
    ])
    .await;
    let provider = provider(&server, ResponsesQuirks::new(NAMESPACE));

    let request = ChatRequest::new("m")
        .user("Write a long essay.")
        .with_option(ResponsesRequestOptions::new().store(true).background(true));
    let started = provider.chat(&request).await.unwrap();
    assert_eq!(started.id.as_deref(), Some("resp_bg"));
    assert_eq!(started.finish_reason, None, "still running");
    assert_eq!(started.finish_reason_raw.as_deref(), Some("queued"));

    let finished = provider.retrieve_response("resp_bg").await.unwrap();
    assert_eq!(finished.finish_reason, Some(FinishReason::Stop));
    assert_eq!(finished.text().as_deref(), Some("Sunny."));

    let stopped = provider.cancel_response("resp_bg").await.unwrap();
    assert_eq!(
        stopped.finish_reason,
        Some(FinishReason::Other("cancelled".into()))
    );

    let requests = server.recorded_requests().await;
    let body = requests[0].body_json();
    assert_eq!(
        (body["store"].clone(), body["background"].clone()),
        (json!(true), json!(true))
    );
    assert_eq!(
        (requests[1].method.as_str(), requests[1].path.as_str()),
        ("GET", "/responses/resp_bg")
    );
    assert_eq!(
        (requests[2].method.as_str(), requests[2].path.as_str()),
        ("POST", "/responses/resp_bg/cancel")
    );
    assert_eq!(
        requests[1].header("authorization"),
        Some("Bearer local-key")
    );

    assert!(matches!(
        provider.retrieve_response("../models").await,
        Err(allama::Error::InvalidRequest { .. })
    ));
}

#[tokio::test]
async fn continuation_sends_only_the_new_turn() {
    let server =
        TestHttpServer::spawn([MockHttpResponse::json_value(StatusCode::OK, &text_body())]).await;
    let provider = provider(&server, ResponsesQuirks::new(NAMESPACE));
    let request = ChatRequest::new("m")
        .user("And tomorrow?")
        .with_options_json(
            "openai-compat-responses",
            json!({"store": true, "previous_response_id": "resp_1"}),
        );
    provider.chat(&request).await.unwrap();
    let body = server.recorded_requests().await[0].body_json();
    assert_eq!(body["previous_response_id"], "resp_1");
    assert_eq!(body["store"], true);
    assert_eq!(body["input"].as_array().unwrap().len(), 1);
    assert!(body.get("background").is_none());

    let unstored = ChatRequest::new("m")
        .user("hi")
        .with_options_json("openai-compat-responses", json!({"background": true}));
    assert!(matches!(
        provider.chat(&unstored).await,
        Err(allama::Error::InvalidRequest { .. })
    ));
    let smuggled = ChatRequest::new("m")
        .user("hi")
        .with_options_json("openai-compat-responses", json!({"extra": {"store": true}}));
    assert!(matches!(
        provider.chat(&smuggled).await,
        Err(allama::Error::InvalidRequest { .. })
    ));
}

#[tokio::test]
async fn raw_response_is_read_through_the_provider_namespace() {
    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(StatusCode::OK, &text_body()),
        MockHttpResponse::json_value(StatusCode::OK, &text_body()),
    ])
    .await;
    let provider = provider(&server, ResponsesQuirks::new(NAMESPACE));
    let plain = provider
        .chat(&ChatRequest::new("m").user("hi"))
        .await
        .unwrap();
    assert!(provider.raw_response(&plain).is_none());
    let kept = provider
        .chat(
            &ChatRequest::new("m")
                .user("hi")
                .with_options_json("openai-compat-responses", json!({"raw_response": true})),
        )
        .await
        .unwrap();
    assert_eq!(provider.raw_response(&kept), Some(&text_body()));
    assert!(kept.provider_data.namespace(NAMESPACE).is_some());
}

/// LiteLLM's Responses API (1.98, in front of Ollama `qwen3:1.7b`), from
/// recordings: raw reasoning arrives as `output_text` parts, and message
/// items are streamed without `response.output_item.added` (also after a
/// tool call), on an `output_index` already used by the reasoning item.
mod litellm {
    use super::*;
    use allama_http::{ReplayResponse, ReplayTransport};

    fn fixture(name: &str) -> String {
        std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/responses")
                .join(name),
        )
        .unwrap()
    }

    fn provider(response: ReplayResponse) -> ResponsesProvider {
        ResponsesProvider::builder()
            .base_url("http://127.0.0.1:4000/v1")
            .provider_name("litellm")
            .http_transport(ReplayTransport::new([response]))
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn unary_reasoning_comes_from_output_text_parts() {
        let response = provider(ReplayResponse::json(
            StatusCode::OK,
            fixture("litellm_text.json"),
        ))
        .chat(&ChatRequest::new("ollama/qwen3:1.7b").user("Say hi in one word."))
        .await
        .unwrap();
        assert!(response.reasoning_text().is_some_and(|r| !r.is_empty()));
        assert!(response.text().is_some_and(|t| !t.is_empty()));
    }

    #[tokio::test]
    async fn unannounced_messages_stream() {
        let response = provider(ReplayResponse::sse(fixture("litellm_text_stream.sse")))
            .chat_stream(&ChatRequest::new("ollama/qwen3:1.7b").user("Say hi in one word."))
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        assert!(response.reasoning_text().is_some_and(|r| !r.is_empty()));
        assert!(response.text().is_some_and(|t| !t.is_empty()));
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
    }

    #[tokio::test]
    async fn tool_calls_stream_with_a_trailing_unannounced_message() {
        let response = provider(ReplayResponse::sse(fixture("litellm_tool_stream.sse")))
            .chat_stream(&weather_request())
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        let call = response.first_tool_call().expect("a tool call");
        assert_eq!(call.name, "get_weather");
        assert_eq!(
            serde_json::from_str::<Value>(&call.arguments).unwrap()["city"],
            "Paris"
        );
        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
    }

    #[tokio::test]
    async fn unknown_stream_frames_are_surfaced() {
        for name in ["litellm_text_stream.sse", "litellm_tool_stream.sse"] {
            allama_conformance::stream_frames::assert_unknown_stream_frames_surfaced(
                &fixture(name),
                &allama_conformance::stream_frames::StreamFraming::sse(),
                |sse| async move {
                    provider(ReplayResponse::sse(sse))
                        .chat_stream(&weather_request())
                        .await
                },
            )
            .await;
        }
    }
}
