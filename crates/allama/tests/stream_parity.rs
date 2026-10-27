//! A streamed response collects to what the unary call returns, including
//! when a tool call's arguments are not valid JSON.

#![cfg(all(feature = "mock", feature = "chat-extract"))]

use allama::chat::extract::{ChatExtractError, ChatExtractExt, ChatExtractionMode};
use allama::mock::{ChatResponseBuilder, MockProvider, MockStreamingProvider};
use allama::prelude::*;
use allama::schemars::{self, JsonSchema};

#[allow(dead_code)]
#[derive(Debug, serde::Deserialize, JsonSchema)]
struct Person {
    name: String,
}

/// A forced-tool extraction cut off at the output cap.
fn truncated() -> ChatResponse {
    ChatResponseBuilder::new()
        .tool_calls([ToolCall::new(
            "c1",
            "submit_structured_output",
            r#"{"name": "Ad"#,
        )])
        .finish_reason(FinishReason::Length)
        .usage(10, 4096)
        .build()
}

#[tokio::test]
async fn malformed_tool_arguments_collect_like_unary() {
    let response = || {
        ChatResponseBuilder::new()
            .tool_calls([ToolCall::new("c1", "lookup", r#"{"city": "Par"#)])
            .finish_reason(FinishReason::Length)
            .build()
    };
    let request = ChatRequest::new("m").user("hi");
    let unary = MockProvider::with_response(response())
        .chat(&request)
        .await
        .unwrap();
    let streamed = MockProvider::with_response(response())
        .chat_stream(&request)
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();
    assert_eq!(
        streamed.tool_calls().collect::<Vec<_>>(),
        unary.tool_calls().collect::<Vec<_>>()
    );
    assert_eq!(streamed.finish_reason, Some(FinishReason::Length));
    assert!(!streamed.first_tool_call().unwrap().has_valid_arguments());
}

#[tokio::test]
async fn truncation_is_reported_the_same_unary_and_streamed() {
    let request = ChatRequest::new("m")
        .user("Ada")
        .with_option(ChatExtractionMode::ForcedTool);
    let unary = MockProvider::with_response(truncated())
        .extract::<Person>(&request)
        .await
        .unwrap_err();
    let provider = MockStreamingProvider::from_responses([truncated()]);
    let mut stream = provider.extract_stream::<Person>(&request).await.unwrap();
    while stream.next_event().await.is_some() {}
    let streamed = stream.finish().unwrap_err();
    assert_eq!(unary.kind_name(), streamed.kind_name());
    let Error::Extract { failure, .. } = &streamed else {
        panic!("expected an extraction failure, got {streamed:?}");
    };
    assert!(
        matches!(failure.error(), ChatExtractError::Truncated { .. }),
        "{streamed}"
    );
}
