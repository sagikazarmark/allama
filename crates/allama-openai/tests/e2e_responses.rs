//! Live tests against the OpenAI Responses API.
//!
//! Ignored by default. Run with:
//!
//! ```sh
//! OPENAI_API_KEY=... cargo test -p allama-openai --test e2e_responses -- --ignored
//! ```
//!
//! `OPENAI_MODEL` overrides the default non-reasoning model (`gpt-4.1-nano`),
//! `OPENAI_REASONING_MODEL` the reasoning model (`gpt-5-nano`).

// Sends through the default `reqwest` transport.
#![cfg(feature = "reqwest")]

use allama::history::History;
use allama::{
    ChatProvider, ChatRequest, ChatStreamExt, ContentBlock, ReasoningConfig, ReasoningEffort,
    ReasoningOutput, ToolDefinition,
};
use allama_conformance::e2e;
use allama_openai::ResponsesProvider;
use serde_json::json;

fn provider() -> ResponsesProvider {
    ResponsesProvider::from_env().expect("OPENAI_API_KEY must be set to run e2e tests")
}

fn model() -> String {
    std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4.1-nano".into())
}

fn reasoning_model() -> String {
    std::env::var("OPENAI_REASONING_MODEL").unwrap_or_else(|_| "gpt-5-nano".into())
}

fn low_reasoning_with_summary() -> ReasoningConfig {
    ReasoningConfig::new()
        .effort(ReasoningEffort::Low)
        .output(ReasoningOutput::Summary)
}

#[tokio::test]
#[ignore]
async fn responses_basic_chat() {
    e2e::basic_chat(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn responses_streaming() {
    e2e::streaming(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn responses_system_prompt() {
    e2e::system_prompt(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn responses_multi_turn() {
    e2e::multi_turn(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn responses_tool_round_trip() {
    e2e::tool_round_trip(&provider(), &model(), None).await;
}

#[tokio::test]
#[ignore]
async fn responses_reasoning_tool_round_trip() {
    e2e::tool_round_trip(
        &provider(),
        &reasoning_model(),
        Some(low_reasoning_with_summary()),
    )
    .await;
}

#[tokio::test]
#[ignore]
async fn responses_streaming_tool_round_trip() {
    e2e::streaming_tool_round_trip(&provider(), &model(), None).await;
}

#[tokio::test]
#[ignore]
async fn responses_streaming_reasoning_tool_round_trip() {
    e2e::streaming_tool_round_trip(
        &provider(),
        &reasoning_model(),
        Some(low_reasoning_with_summary()),
    )
    .await;
}

#[tokio::test]
#[ignore]
async fn responses_parallel_tools() {
    e2e::parallel_tools(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn responses_structured_output() {
    e2e::structured_output(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn responses_extract() {
    e2e::extract(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn responses_extract_enum_root() {
    e2e::extract_enum_root(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn responses_sampling_on_reasoning_model() {
    e2e::sampling_on_reasoning_model(&provider(), &reasoning_model()).await;
}

#[tokio::test]
#[ignore]
async fn responses_vision() {
    e2e::vision(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn responses_file_input() {
    e2e::file_input(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn responses_error_bad_key() {
    e2e::error_bad_key(&ResponsesProvider::new("sk-invalid").unwrap(), &model()).await;
}

/// A multi-turn tool conversation on a
/// reasoning model where the second request replays the encrypted reasoning
/// item statelessly (`store: false`, no `previous_response_id`), with the
/// history persisted to JSON and restored between turns.
#[tokio::test]
#[ignore]
async fn responses_encrypted_reasoning_round_trip() {
    let provider = provider();
    let model = reasoning_model();
    let tool = ToolDefinition::new(
        "get_weather",
        json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
    )
    .description("Get the current weather for a city")
    .strict(true);
    let reasoning = low_reasoning_with_summary();

    let mut request = ChatRequest::new(&model)
        .tools([tool.clone()])
        .reasoning(reasoning)
        .user("What is the weather in Paris? Use the tool, then answer in one sentence.");
    let first = provider.chat(&request).await.expect("first turn failed");
    assert!(first.warnings.is_empty(), "{:?}", first.warnings);
    let encrypted = first
        .content
        .iter()
        .filter_map(ContentBlock::as_reasoning)
        .find_map(|r| r.provider_data.get_raw("openai", "encrypted_content"))
        .expect("first turn returned no encrypted reasoning")
        .clone();
    let call = first
        .first_tool_call()
        .expect("expected a tool call")
        .clone();
    request.push_message(first.to_assistant_message());
    request.push_tool_result(&call, "Sunny, 18 degrees Celsius.");

    // Persist the conversation and continue from the stored copy.
    let saved = serde_json::to_string(&History::from_request(&request)).unwrap();
    let restored: History = serde_json::from_str(&saved).unwrap();
    let restored_encrypted = restored
        .messages
        .iter()
        .filter_map(|m| m.as_assistant())
        .flat_map(|m| m.content.iter())
        .filter_map(ContentBlock::as_reasoning)
        .find_map(|r| r.provider_data.get_raw("openai", "encrypted_content"))
        .cloned();
    assert_eq!(restored_encrypted.as_ref(), Some(&encrypted));
    let next = restored
        .into_request(&model)
        .tools([tool])
        .reasoning(reasoning);

    let second = provider.chat(&next).await.expect("replay turn failed");
    // No `Dropped` warning: the reasoning item was replayed, not discarded.
    assert!(second.warnings.is_empty(), "{:?}", second.warnings);
    let text = second.text().expect("expected a final answer");
    assert!(
        text.to_lowercase().contains("sunny") || text.contains("18"),
        "unexpected answer: {text}"
    );

    // The same history also replays through streaming.
    let streamed = provider
        .chat_stream(&next)
        .await
        .expect("streaming replay failed")
        .collect_response()
        .await
        .expect("streaming replay collection failed");
    assert!(streamed.text().is_some_and(|t| !t.is_empty()));
}

/// C2 + C5: hosted web search comes back as provider tool blocks and its
/// annotations as citations, unary and streamed, and the turn replays.
#[tokio::test]
#[ignore]
async fn responses_web_search_citations() {
    let provider = provider();
    let request = ChatRequest::new("gpt-4.1-mini")
        .user("In one sentence, what is the latest stable Rust release? Cite the official announcement.")
        .with_options_json(
            "openai-responses",
            json!({"hosted_tools": [{"type": "web_search"}]}),
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
        let citations: Vec<_> = response.citations().collect();
        assert!(!citations.is_empty(), "expected citations: {response:?}");
        assert!(
            citations
                .iter()
                .all(|c| matches!(c.source, allama::CitationSource::Url { .. })),
            "{citations:?}"
        );
        let call = response
            .content
            .iter()
            .find_map(ContentBlock::as_provider_tool_call)
            .expect("a hosted web search call");
        assert_eq!(call.name, "web_search");
        assert!(
            response
                .content
                .iter()
                .filter_map(ContentBlock::as_provider_tool_result)
                .any(|r| r.call_id == call.id)
        );
    }

    // C5: the hosted call replays statelessly as its original item.
    let follow_up = request
        .message(unary.to_assistant_message())
        .user("Answer again in three words.");
    let next = provider.chat(&follow_up).await.expect("replayed turn");
    assert!(next.warnings.is_empty(), "{:?}", next.warnings);
    assert!(next.text().is_some());
}

/// Issue 22: continue from stored state with `previous_response_id`, so the
/// second request carries only the new turn.
#[tokio::test]
#[ignore]
async fn responses_previous_response_id_continuation() {
    let provider = provider();
    let first = provider
        .chat(
            &ChatRequest::new(model())
                .user("My favorite color is teal. Reply with OK.")
                .with_options_json("openai-responses", json!({"store": true})),
        )
        .await
        .expect("stored first turn");
    let id = first.id.clone().expect("response id");
    let second = provider
        .chat(
            &ChatRequest::new(model())
                .user("What is my favorite color? One word.")
                .with_options_json(
                    "openai-responses",
                    json!({"store": true, "previous_response_id": id}),
                ),
        )
        .await
        .expect("continued turn");
    assert!(
        second.text_or_empty().to_lowercase().contains("teal"),
        "{second:?}"
    );
}

/// Issue 22: a background response is queued, then polled to completion.
#[tokio::test]
#[ignore]
async fn responses_background_polling() {
    let provider = provider();
    let started = provider
        .chat(
            &ChatRequest::new(model())
                .user("Say hello in one word.")
                .with_options_json(
                    "openai-responses",
                    json!({"store": true, "background": true}),
                ),
        )
        .await
        .expect("background request");
    let id = started.id.clone().expect("response id");
    let mut response = started;
    for _ in 0..60 {
        if response.finish_reason.is_some() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        response = provider.retrieve_response(&id).await.expect("poll");
    }
    assert_eq!(
        response.finish_reason,
        Some(allama::FinishReason::Stop),
        "{response:?}"
    );
    assert!(response.text().is_some());
}

/// Issue 22: a background stream can be dropped and picked up again by id.
#[tokio::test]
#[ignore]
async fn responses_background_stream_resumes_by_id() {
    use futures_util::StreamExt;

    let provider = provider();
    let request = ChatRequest::new(model())
        .user("Count from 1 to 30 in words, comma separated.")
        .with_options_json(
            "openai-responses",
            json!({"store": true, "background": true}),
        );
    let mut stream = provider
        .chat_stream(&request)
        .await
        .expect("background stream");
    let mut id = None;
    while let Some(event) = stream.next().await {
        if let allama::StreamEvent::ResponseStart { id: started, .. } = event.expect("event") {
            id = started;
            break;
        }
    }
    drop(stream);
    let id = id.expect("response id");
    let resumed = provider
        .resume_stream(&id, None)
        .await
        .expect("resumed stream")
        .collect_response()
        .await
        .expect("resumed stream collects");
    assert_eq!(resumed.id.as_deref(), Some(id.as_str()));
    assert!(
        resumed.text_or_empty().to_lowercase().contains("thirty"),
        "{resumed:?}"
    );
}

/// ADR 0014: a freeform custom tool call (with a grammar) and a local shell
/// call come back as `Other` content, wait for the caller, and their typed
/// answers continue the conversation, unary and streamed.
#[tokio::test]
#[ignore]
async fn responses_client_tools_round_trip() {
    use allama_openai::{ClientToolCall, ShellCommandOutput};

    let provider = provider();
    let request = ChatRequest::new("gpt-5.1")
        .user(
            "Use the sql tool to count the users table, and the shell tool to run `date`. \
             Then answer in one sentence.",
        )
        .reasoning(ReasoningConfig::new().effort(ReasoningEffort::Low))
        .with_options_json(
            "openai-responses",
            json!({"client_tools": [
                {"type": "custom", "name": "sql", "description": "Run a read-only SQLite query.",
                 "format": {"type": "grammar", "syntax": "regex", "definition": "SELECT .+"}},
                {"type": "shell", "environment": {"type": "local"}}
            ]}),
        );
    let unary = provider.chat(&request).await.expect("client tools chat");
    let streamed = provider
        .chat_stream(&request)
        .await
        .expect("client tools stream")
        .collect_response()
        .await
        .expect("client tools stream collects");
    for response in [&unary, &streamed] {
        assert_eq!(
            response.finish_reason,
            Some(allama::FinishReason::ToolCalls),
            "{response:?}"
        );
        assert!(!ClientToolCall::pending(&response.content).is_empty());
    }

    let mut follow_up = request.clone();
    follow_up.push_message(unary.to_assistant_message());
    let mut answers = Vec::new();
    for call in ClientToolCall::pending(&unary.content) {
        answers.push(match call {
            ClientToolCall::Custom(call) => {
                assert!(call.input.starts_with("SELECT"), "{call:?}");
                call.output("42")
            }
            ClientToolCall::Shell(call) => call.output(
                call.commands
                    .iter()
                    .map(|_| ShellCommandOutput::exited("Sat Oct  3 12:00:00 UTC 2026\n", "", 0)),
            ),
            other => panic!("unexpected client tool call {other:?}"),
        });
    }
    follow_up.push_message(allama::Message::user(answers));
    let next = provider.chat(&follow_up).await.expect("answered turn");
    assert!(next.warnings.is_empty(), "{:?}", next.warnings);
    assert!(next.text().is_some() || !ClientToolCall::pending(&next.content).is_empty());
}
