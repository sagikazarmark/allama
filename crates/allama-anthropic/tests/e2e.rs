//! Live end-to-end tests against the Anthropic API.
//!
//! Ignored by default. Run with:
//!
//! ```sh
//! ANTHROPIC_API_KEY=... cargo test -p allama-anthropic --test e2e -- --ignored
//! ```
//!
//! `ANTHROPIC_MODEL` overrides the default (cheap) model;
//! `ANTHROPIC_LATEST_MODEL` overrides the newest-model tests
//! (`claude-sonnet-5-5`).

// Sends through the default `reqwest` transport.
#![cfg(feature = "reqwest")]

use allama::{ReasoningConfig, ReasoningEffort};
use allama_anthropic::Provider;
use allama_conformance::e2e;

fn make_provider() -> Provider {
    let api_key =
        std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set to run e2e tests");
    Provider::new(api_key).expect("failed to create Anthropic provider")
}

fn model() -> String {
    std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-haiku-4-5-20251001".into())
}

fn latest_model() -> String {
    std::env::var("ANTHROPIC_LATEST_MODEL").unwrap_or_else(|_| "claude-sonnet-5-5".into())
}

#[tokio::test]
#[ignore]
async fn basic_chat() {
    e2e::basic_chat(&make_provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn streaming() {
    e2e::streaming(&make_provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn streaming_multi_chunk() {
    e2e::streaming_multi_chunk(&make_provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn system_prompt() {
    e2e::system_prompt(&make_provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn multi_turn() {
    e2e::multi_turn(&make_provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn tool_calling() {
    e2e::tool_calling(&make_provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn tool_round_trip() {
    e2e::tool_round_trip(&make_provider(), &model(), None).await;
}

/// Thinking signatures must survive the tool round trip.
#[tokio::test]
#[ignore]
async fn anthropic_thinking_tool_roundtrip() {
    e2e::tool_round_trip(
        &make_provider(),
        &model(),
        Some(ReasoningConfig::new().effort(ReasoningEffort::Low)),
    )
    .await;
}

/// Adaptive thinking on the newest model family.
#[tokio::test]
#[ignore]
async fn anthropic_latest_thinking_tool_roundtrip() {
    e2e::tool_round_trip(
        &make_provider(),
        &latest_model(),
        Some(ReasoningConfig::new().effort(ReasoningEffort::Low)),
    )
    .await;
}

#[tokio::test]
#[ignore]
async fn streaming_tool_round_trip() {
    e2e::streaming_tool_round_trip(
        &make_provider(),
        &model(),
        Some(ReasoningConfig::new().effort(ReasoningEffort::Low)),
    )
    .await;
}

#[tokio::test]
#[ignore]
async fn structured_output() {
    e2e::structured_output(&make_provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn extract() {
    e2e::extract(&make_provider(), &model()).await;
}

/// The newest models reject forced tool choice; extraction must pick
/// native structured outputs.
#[tokio::test]
#[ignore]
async fn anthropic_latest_extract() {
    e2e::extract(&make_provider(), &latest_model()).await;
}

/// Numeric fields (`u32`, `f64`) carry `format`s Anthropic does not list;
/// the strict pass drops them, so native structured output and the strict
/// forced tool are both accepted.
#[tokio::test]
#[ignore]
async fn anthropic_strict_numeric_fields() {
    use allama::chat::extract::{ChatExtractExt, ChatExtractionMode, SchemaEnforcement};
    use allama::prelude::*;

    #[derive(Debug, serde::Deserialize, allama::schemars::JsonSchema)]
    #[schemars(crate = "allama::schemars")]
    struct Order {
        quantity: u32,
        unit_price: f64,
        ordered_at: String,
    }

    let provider = make_provider();
    for mode in [ChatExtractionMode::Native, ChatExtractionMode::ForcedTool] {
        let request = ChatRequest::new(model())
            .user("Three widgets at 2.5 each, ordered 2026-01-02T10:00:00Z.")
            .with_option(mode);
        let extracted = provider
            .extract::<Order>(&request)
            .await
            .unwrap_or_else(|e| panic!("{mode:?}: {e}"));
        // What the strict pass reported.
        println!(
            "{mode:?}: enforcement {:?}, degradations {:?}",
            extracted.metadata.enforcement,
            extracted
                .metadata
                .degradations
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        );
        assert_eq!(extracted.value.quantity, 3, "{mode:?}");
        assert!((extracted.value.unit_price - 2.5).abs() < 1e-9, "{mode:?}");
        assert!(!extracted.value.ordered_at.is_empty(), "{mode:?}");
        // `u32`'s `minimum: 0` is removed, so the schema is approximated;
        // the dropped `format`s are not what makes it so.
        assert_ne!(
            extracted.metadata.enforcement,
            SchemaEnforcement::NotEnforced,
            "{mode:?}"
        );
        for warning in &extracted.metadata.degradations {
            assert!(!warning.message.contains("format"), "{mode:?}: {warning}");
        }
    }
}

#[tokio::test]
#[ignore]
async fn error_bad_key() {
    let provider = Provider::new("sk-ant-invalid").expect("provider");
    e2e::error_bad_key(&provider, &model()).await;
}

#[tokio::test]
#[ignore]
async fn vision() {
    e2e::vision(&make_provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn file_input() {
    e2e::file_input(&make_provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn parallel_tools() {
    e2e::parallel_tools(&make_provider(), &model()).await;
}

/// Thinking blocks carry signatures and replay without degradation
/// warnings; `ToolChoice::Disabled` and `parallel_tool_calls(false)` are
/// accepted by the API.
#[tokio::test]
#[ignore]
async fn anthropic_signature_replay_and_tool_controls() {
    use allama::{ChatProvider, ChatRequest, ReasoningOutput, ToolChoice, ToolDefinition};

    let provider = make_provider();
    let tool = ToolDefinition::new(
        "get_weather",
        serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": false
        }),
    )
    .description("Get the current weather for a city.")
    .strict(true);
    let mut request = ChatRequest::new(latest_model())
        .tools([tool])
        .parallel_tool_calls(false)
        .reasoning(
            ReasoningConfig::new()
                .effort(ReasoningEffort::Low)
                .output(ReasoningOutput::Summary),
        )
        .user("What is the weather in Paris? Use the tool.");
    let first = provider.chat(&request).await.expect("first turn");
    assert!(first.warnings.is_empty(), "{:?}", first.warnings);
    assert_eq!(first.tool_calls().count(), 1, "{:?}", first.content);
    for reasoning in first.content.iter().filter_map(|b| b.as_reasoning()) {
        assert!(
            reasoning
                .provider_data
                .get_raw("anthropic", "signature")
                .is_some()
                || reasoning
                    .provider_data
                    .get_raw("anthropic", "redacted_data")
                    .is_some(),
            "{reasoning:?}"
        );
    }
    request.push_message(first.to_assistant_message());
    let call = first.first_tool_call().unwrap().clone();
    request.push_tool_result(&call, "Sunny, 18 degrees Celsius.");
    request.tool_choice = Some(ToolChoice::Disabled);
    let second = provider.chat(&request).await.expect("replay turn");
    assert!(second.warnings.is_empty(), "{:?}", second.warnings);
    assert!(!second.has_tool_calls());
    assert!(second.text().is_some_and(|t| !t.is_empty()));
    assert!(
        second
            .request_id
            .as_deref()
            .is_some_and(|id| id.starts_with("req_"))
    );
}

/// C2 + C5: web search runs as a server tool; its call and result come back
/// as provider tool blocks, the answer cites the results, and the whole turn
/// replays on the next request.
#[tokio::test]
#[ignore]
async fn anthropic_web_search_server_tool() {
    use allama::{ChatProvider, ChatRequest, ChatStreamExt, ContentBlock};

    let provider = make_provider();
    let request = ChatRequest::new(model())
        .user("In one sentence, what is the latest stable Rust release? Search the web.")
        .with_options_json(
            allama_anthropic::NAMESPACE,
            serde_json::json!({"server_tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}]}),
        );
    let unary = provider.chat(&request).await.expect("web search chat");
    let streamed = provider
        .chat_stream(&request)
        .await
        .expect("web search stream")
        .collect_response()
        .await
        .expect("web search stream collects");
    for response in [&unary, &streamed] {
        let call = response
            .content
            .iter()
            .find_map(ContentBlock::as_provider_tool_call)
            .expect("a server tool call");
        assert_eq!(call.name, "web_search");
        let result = response
            .content
            .iter()
            .find_map(ContentBlock::as_provider_tool_result)
            .expect("a server tool result");
        assert_eq!(result.call_id, call.id);
        assert!(response.tool_calls().next().is_none());
        assert!(response.citations().count() > 0, "{response:?}");
    }

    let follow_up = request
        .message(unary.to_assistant_message())
        .user("Answer again in three words.");
    let next = provider.chat(&follow_up).await.expect("replayed turn");
    assert!(next.warnings.is_empty(), "{:?}", next.warnings);
    assert!(next.text().is_some());
}

/// `computer_toolset_20260801` is declared without a name, its member
/// calls come back with `toolset_name` (unary and streamed), and the turn
/// replays with each `tool_result` repeating it.
#[tokio::test]
#[ignore]
async fn anthropic_computer_toolset_round_trip() {
    use allama::{ChatProvider, ChatRequest, ChatStreamExt, ContentPart, Message, ToolChoice};

    // A 1×1 PNG stands in for the screenshot.
    const SCREENSHOT: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

    let provider = make_provider();
    let mut request = ChatRequest::new(latest_model())
        .user("Take a screenshot of the screen. Use the computer tool.")
        .with_options_json(
            allama_anthropic::NAMESPACE,
            serde_json::json!({"server_tools": [{"type": "computer_toolset_20260801"}]}),
        );
    let unary = provider.chat(&request).await.expect("computer use chat");
    let streamed = provider
        .chat_stream(&request)
        .await
        .expect("computer use stream")
        .collect_response()
        .await
        .expect("computer use stream collects");
    for response in [&unary, &streamed] {
        assert!(response.warnings.is_empty(), "{:?}", response.warnings);
        let call = response.first_tool_call().expect("a toolset member call");
        assert_eq!(
            call.provider_data.get_raw("anthropic", "toolset_name"),
            Some(&serde_json::json!("computer")),
            "{call:?}"
        );
    }

    request.push_message(unary.to_assistant_message());
    for call in unary.tool_calls() {
        request.push_message(Message::tool_result(
            call.id.clone(),
            call.name.clone(),
            vec![ContentPart::image_base64("image/png", SCREENSHOT)],
        ));
    }
    request.tool_choice = Some(ToolChoice::Disabled);
    let next = provider.chat(&request).await.expect("replayed turn");
    assert!(next.warnings.is_empty(), "{:?}", next.warnings);
    assert!(next.text().is_some());
}
