//! Live tests against the OpenAI API (`cargo test -p allama-openai --test e2e -- --ignored`).

// Sends through the default `reqwest` transport.
#![cfg(feature = "reqwest")]

use allama::{ChatProvider, ChatRequest, Error, ReasoningEffort, ResponseFormat};
use allama_conformance::e2e;
use allama_openai::Provider;
use allama_openai_compat::CompatQuirks;
use serde_json::json;

fn provider() -> Provider {
    Provider::from_env().expect("OPENAI_API_KEY must be set to run e2e tests")
}

fn model() -> String {
    std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4.1-nano".into())
}

fn reasoning_model() -> String {
    std::env::var("OPENAI_REASONING_MODEL").unwrap_or_else(|_| "gpt-5-nano".into())
}

fn embedding_model() -> String {
    std::env::var("OPENAI_EMBEDDING_MODEL").unwrap_or_else(|_| "text-embedding-3-small".into())
}

#[tokio::test]
#[ignore]
async fn basic_chat() {
    e2e::basic_chat(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn streaming() {
    e2e::streaming(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn streaming_multi_chunk() {
    e2e::streaming_multi_chunk(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn system_prompt() {
    e2e::system_prompt(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn multi_turn() {
    e2e::multi_turn(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn tool_calling() {
    e2e::tool_calling(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn tool_round_trip() {
    e2e::tool_round_trip(&provider(), &model(), None).await;
}

#[tokio::test]
#[ignore]
async fn streaming_tool_round_trip() {
    e2e::streaming_tool_round_trip(&provider(), &model(), None).await;
}

#[tokio::test]
#[ignore]
async fn parallel_tools() {
    e2e::parallel_tools(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn structured_output() {
    e2e::structured_output(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn extract() {
    e2e::extract(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn extract_enum_root() {
    e2e::extract_enum_root(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn sampling_on_reasoning_model() {
    e2e::sampling_on_reasoning_model(&provider(), &reasoning_model()).await;
}

#[tokio::test]
#[ignore]
async fn vision() {
    e2e::vision(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn file_input() {
    e2e::file_input(&provider(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn error_bad_key() {
    e2e::error_bad_key(&Provider::new("sk-invalid").unwrap(), &model()).await;
}

#[tokio::test]
#[ignore]
async fn basic_embed() {
    e2e::basic_embed(&provider(), &embedding_model()).await;
}

#[tokio::test]
#[ignore]
async fn batch_embed() {
    e2e::batch_embed(&provider(), &embedding_model()).await;
}

#[tokio::test]
#[ignore]
async fn dimensions() {
    e2e::dimensions(&provider(), &embedding_model(), 256).await;
}

/// Reasoning models reject `max_tokens`; the adapter must send
/// `max_completion_tokens`.
#[tokio::test]
#[ignore]
async fn openai_reasoning_model_max_tokens() {
    let response = provider()
        .chat(
            &ChatRequest::new(reasoning_model())
                .user("Say hello in three words.")
                .max_output_tokens(400)
                .reasoning(ReasoningEffort::Minimal),
        )
        .await
        .expect("reasoning model request failed");
    assert!(response.text().is_some_and(|t| !t.is_empty()));
}

#[tokio::test]
#[ignore]
async fn reasoning_tool_round_trip() {
    e2e::tool_round_trip(
        &provider(),
        &reasoning_model(),
        Some(ReasoningEffort::Low.into()),
    )
    .await;
}

/// Pins the OpenAI strict-mode rule the `OpenAiStrict` dialect relies on:
/// sent as written (no dialect), a `$ref` with sibling keywords is
/// rejected, which is why the dialect merges it with its target. The
/// schema is otherwise strict (closed, every property required), so the
/// `$ref` is the reason for the rejection.
///
/// A non-null `default` was rejected too until OpenAI began accepting it
/// (observed 2026-10-13). The dialect still removes every `default` after
/// using a non-null one as the signal to keep an optional property
/// non-nullable: strict mode requires every property, so a default never
/// applies, and the Structured Outputs guide does not list the keyword.
#[tokio::test]
#[ignore]
async fn openai_strict_rejects_ref_siblings() {
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let mut quirks = CompatQuirks::openai();
    quirks.schema_dialect = None;
    let provider = allama_openai_compat::Provider::builder()
        .base_url("https://api.openai.com/v1")
        .bearer_token(api_key)
        .provider_name("openai")
        .quirks(quirks)
        .build()
        .unwrap();

    let ref_with_sibling = json!({
        "type": "object",
        "properties": {"owner": {"$ref": "#/$defs/Person", "description": "Who owns it"}},
        "required": ["owner"],
        "additionalProperties": false,
        "$defs": {"Person": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": false
        }}
    });
    let request = ChatRequest::new(model())
        .user("Reply with JSON.")
        .max_output_tokens(50)
        .response_format(ResponseFormat::json_schema(ref_with_sibling).name("probe"));
    let result = provider.chat(&request).await;
    assert!(
        matches!(result, Err(Error::InvalidRequest { .. })),
        "expected OpenAI to reject a `$ref` with siblings, got {result:?}"
    );
}

fn sql_tool() -> serde_json::Value {
    json!({"type": "custom", "custom": {
        "name": "sql",
        "description": "Run a read-only SQLite query.",
        "format": {"type": "grammar", "grammar": {"syntax": "regex", "definition": "SELECT .+"}}
    }})
}

/// ADR 0014: a custom tool call (freeform input with a grammar) on Chat
/// Completions, unary and streamed. The streamed shape is not documented;
/// this checks the decoder's reading of it (`fixtures/hand_written/`) and
/// that the typed answer continues the conversation.
#[tokio::test]
#[ignore]
async fn chat_custom_tool_round_trip() {
    use allama::ChatStreamExt;
    use allama_openai::ClientToolCall;

    let provider = provider();
    let request = ChatRequest::new(reasoning_model())
        .user("Use the sql tool to count the rows of the users table.")
        .with_options_json("openai", json!({"client_tools": [sql_tool()]}));
    let unary = provider.chat(&request).await.expect("custom tool chat");
    let streamed = provider
        .chat_stream(&request)
        .await
        .expect("custom tool stream")
        .collect_response()
        .await
        .expect("custom tool stream collects");
    for response in [&unary, &streamed] {
        assert_eq!(
            response.finish_reason,
            Some(allama::FinishReason::ToolCalls),
            "{response:?}"
        );
        assert!(response.tool_calls().next().is_none(), "{response:?}");
        let pending = ClientToolCall::pending(&response.content);
        let [ClientToolCall::Custom(call)] = pending.as_slice() else {
            panic!("expected one custom call: {response:?}")
        };
        assert!(call.input.starts_with("SELECT"), "{call:?}");
    }

    let mut follow_up = request.clone();
    follow_up.push_message(unary.to_assistant_message());
    let answers: Vec<_> = ClientToolCall::pending(&unary.content)
        .into_iter()
        .map(|call| match call {
            ClientToolCall::Custom(call) => call.output("42"),
            other => panic!("unexpected client tool call {other:?}"),
        })
        .collect();
    follow_up.push_message(allama::Message::user(answers));
    let next = provider.chat(&follow_up).await.expect("answered turn");
    assert!(next.warnings.is_empty(), "{:?}", next.warnings);
    assert!(next.text_or_empty().contains("42"), "{next:?}");
}

/// Evidence for the failover guidance: whether Chat
/// Completions accepts a replayed custom tool call when the request does
/// not declare the tool. The outcome is printed, not asserted.
#[tokio::test]
#[ignore]
async fn chat_custom_call_replay_without_its_declaration() {
    use allama_openai::ClientToolCall;

    let provider = provider();
    let request = ChatRequest::new(reasoning_model())
        .user("Use the sql tool to count the rows of the users table.")
        .with_options_json("openai", json!({"client_tools": [sql_tool()]}));
    let first = provider.chat(&request).await.expect("custom tool chat");
    let mut follow_up = ChatRequest::new(reasoning_model())
        .user("Use the sql tool to count the rows of the users table.");
    follow_up.push_message(first.to_assistant_message());
    let answers: Vec<_> = ClientToolCall::pending(&first.content)
        .into_iter()
        .filter_map(|call| match call {
            ClientToolCall::Custom(call) => Some(call.output("42")),
            _ => None,
        })
        .collect();
    follow_up.push_message(allama::Message::user(answers));
    match provider.chat(&follow_up).await {
        Ok(response) => println!("accepted: {}", response.text_or_empty()),
        Err(err) => println!("rejected: {err}"),
    }
}
