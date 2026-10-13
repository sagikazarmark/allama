//! Live end-to-end tests against the Gemini API.
//!
//! Ignored by default. Run with `GEMINI_API_KEY` set:
//!
//! ```sh
//! cargo test -p allama-gemini --test e2e -- --ignored
//! ```
//!
//! `GEMINI_MODEL` (default `gemini-3.5-flash-lite`), `GEMINI_25_MODEL`
//! (default `gemini-2.5-flash`), `GEMINI_3_MODEL` (default
//! `gemini-3.5-flash`), `GEMINI_EMBEDDING_MODEL` (default
//! `gemini-embedding-001`), and `GEMINI_MULTIMODAL_EMBEDDING_MODEL` (default
//! `gemini-embedding-2`) select the models.

// Sends through the default `reqwest` transport.
#![cfg(feature = "reqwest")]

use allama::{ChatProvider, ChatRequest, ReasoningConfig, ReasoningEffort};
use allama_conformance::e2e;
use allama_gemini::Provider;

fn make_provider() -> Provider {
    let api_key =
        std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set to run e2e tests");
    Provider::new(api_key).expect("failed to create Gemini provider")
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.into())
}

fn model() -> String {
    env_or("GEMINI_MODEL", "gemini-3.5-flash-lite")
}

fn gemini25_model() -> String {
    env_or("GEMINI_25_MODEL", "gemini-2.5-flash")
}

fn gemini3_model() -> String {
    env_or("GEMINI_3_MODEL", "gemini-3.5-flash")
}

fn embedding_model() -> String {
    env_or("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
}

fn multimodal_embedding_model() -> String {
    env_or("GEMINI_MULTIMODAL_EMBEDDING_MODEL", "gemini-embedding-2")
}

#[tokio::test]
#[ignore]
async fn image_embed() {
    e2e::image_embed(&make_provider(), &multimodal_embedding_model()).await;
}

#[tokio::test]
#[ignore]
async fn composite_embed() {
    e2e::composite_embed(&make_provider(), &multimodal_embedding_model()).await;
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

/// Real replies arrive as several delta chunks.
///
/// Runs on Gemini 2.5: Gemini 3 models stop the shared "count to sixty"
/// prompt with `finishReason: RECITATION` after about twenty numbers
/// (observed 2026-10-02), which is a content filter, not a decoding issue.
#[tokio::test]
#[ignore]
async fn gemini_stream_multi_chunk() {
    e2e::streaming_multi_chunk(&make_provider(), &gemini25_model()).await;
}

/// Multi-chunk streaming on Gemini 3, with a prose prompt that does not
/// trip the recitation filter.
#[tokio::test]
#[ignore]
async fn gemini3_stream_multi_chunk() {
    use allama::chat::stream::StreamCollector;
    use allama::{FinishReason, StreamEvent};
    use futures_util::StreamExt;

    let request = ChatRequest::new(model())
        .user("Write about 150 words on why rivers meander. Plain prose, no lists.");
    let mut stream = make_provider()
        .chat_stream(&request)
        .await
        .expect("chat_stream failed");
    let mut collector = StreamCollector::new();
    let mut deltas = 0;
    while let Some(event) = stream.next().await {
        let event = event.expect("stream error");
        deltas += usize::from(matches!(event, StreamEvent::TextDelta { .. }));
        collector.push(event).expect("protocol error");
    }
    let response = collector.finish().expect("incomplete stream");
    assert!(deltas > 1, "expected several text deltas, got {deltas}");
    assert_eq!(response.finish_reason, Some(FinishReason::Stop));
    assert!(
        response
            .text()
            .is_some_and(|t| t.split_whitespace().count() > 50)
    );
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

/// A Gemini 3 function call is replayed with its thought signature.
#[tokio::test]
#[ignore]
async fn gemini3_tool_roundtrip() {
    e2e::tool_round_trip(
        &make_provider(),
        &gemini3_model(),
        Some(ReasoningConfig::new().effort(ReasoningEffort::Low)),
    )
    .await;
}

#[tokio::test]
#[ignore]
async fn gemini25_tool_roundtrip_with_thoughts() {
    e2e::tool_round_trip(
        &make_provider(),
        &gemini25_model(),
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

/// History with a function call Gemini did not sign (for example from
/// another provider) is accepted on Gemini 3 via the dummy signature.
#[tokio::test]
#[ignore]
async fn gemini3_foreign_tool_history() {
    use allama::{ContentBlock, Message, ToolDefinition, WarningKind};
    let request = ChatRequest::new(gemini3_model())
        .tools([ToolDefinition::new(
            "get_weather",
            serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
        )])
        .user("What is the weather in Paris? Answer in one sentence.")
        .message(Message::assistant_blocks(vec![ContentBlock::tool_call(
            "toolu_01",
            "get_weather",
            r#"{"city":"Paris"}"#,
        )]))
        .message(Message::tool_result("toolu_01", "get_weather", "Sunny, 18C"));
    let response = make_provider().chat(&request).await.expect("replay failed");
    assert!(response.text().is_some_and(|t| !t.is_empty()));
    assert!(
        response
            .warnings
            .iter()
            .any(|w| w.kind == WarningKind::Approximated)
    );
}

#[tokio::test]
#[ignore]
async fn reasoning_output_includes_thoughts() {
    let response = make_provider()
        .chat(
            &ChatRequest::new(gemini25_model())
                .user("What is 17 * 23? Answer with the number.")
                .reasoning(ReasoningConfig::new().effort(ReasoningEffort::Low)),
        )
        .await
        .expect("chat failed");
    assert!(response.reasoning_text().is_some_and(|t| !t.is_empty()));
    let usage = response.usage.expect("usage");
    assert!(usage.reasoning_tokens.is_some_and(|t| t > 0));
    assert_eq!(usage.contract_violation(), None);
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

#[tokio::test]
#[ignore]
async fn error_bad_key() {
    let provider = Provider::new("AIzaSy-invalid-key-for-e2e").unwrap();
    e2e::error_bad_key(&provider, &model()).await;
}

#[tokio::test]
#[ignore]
async fn basic_embed() {
    e2e::basic_embed(&make_provider(), &embedding_model()).await;
}

#[tokio::test]
#[ignore]
async fn batch_embed() {
    e2e::batch_embed(&make_provider(), &embedding_model()).await;
}

#[tokio::test]
#[ignore]
async fn dimensions() {
    e2e::dimensions(&make_provider(), &embedding_model(), 32).await;
}

#[tokio::test]
#[ignore]
async fn input_type_embed() {
    e2e::input_type_embed(&make_provider(), &embedding_model()).await;
}

/// C2: Google Search grounding comes back as citations on the answer text,
/// unary and streamed.
#[tokio::test]
#[ignore]
async fn gemini_grounding_citations() {
    use allama::ChatStreamExt;

    let provider = make_provider();
    let request = ChatRequest::new(gemini25_model())
        .user("In one sentence, what is the latest stable Rust release? Use Google Search.")
        .with_options_json(
            allama_gemini::NAMESPACE,
            serde_json::json!({"builtin_tools": [{"google_search": {}}]}),
        );
    let unary = provider.chat(&request).await.expect("grounded chat");
    let streamed = provider
        .chat_stream(&request)
        .await
        .expect("grounded stream")
        .collect_response()
        .await
        .expect("grounded stream collects");
    for response in [unary, streamed] {
        assert!(
            response.citations().count() > 0,
            "expected citations: {response:?}"
        );
        for block in &response.content {
            if let allama::ContentBlock::Text(text) = block {
                for citation in &text.citations {
                    assert!(text.cited_span(citation).is_some(), "{citation:?}");
                }
            }
        }
    }
}

/// C5: code execution comes back as provider tool blocks, unary and
/// streamed, and the turn (thought signatures included) replays on Gemini 3.
#[tokio::test]
#[ignore]
async fn gemini_code_execution_provider_tools() {
    use allama::{ChatStreamExt, ContentBlock};

    let provider = make_provider();
    let request = ChatRequest::new(gemini3_model())
        .user("What is the sum of the first 20 prime numbers? Compute it with code.")
        .with_options_json(
            allama_gemini::NAMESPACE,
            serde_json::json!({"builtin_tools": [{"code_execution": {}}]}),
        );
    let unary = provider.chat(&request).await.expect("code execution chat");
    let streamed = provider
        .chat_stream(&request)
        .await
        .expect("code execution stream")
        .collect_response()
        .await
        .expect("code execution stream collects");
    for response in [&unary, &streamed] {
        let call = response
            .content
            .iter()
            .find_map(ContentBlock::as_provider_tool_call)
            .expect("a code-execution call");
        assert_eq!(call.name, "code_execution");
        let result = response
            .content
            .iter()
            .find_map(ContentBlock::as_provider_tool_result)
            .expect("its result");
        assert_eq!(result.call_id, call.id);
        assert!(!response.has_tool_calls());
    }

    let follow_up = request
        .message(unary.to_assistant_message())
        .user("Now in one word: is that number prime?");
    let next = provider.chat(&follow_up).await.expect("replayed turn");
    assert!(next.warnings.is_empty(), "{:?}", next.warnings);
    assert!(next.text().is_some());
}
