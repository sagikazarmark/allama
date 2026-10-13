//! Replaying the golden history (`allama/tests/fixtures/history_v1.json`) to
//! the Responses API.
//!
//! The fixture holds a Responses turn with the state the adapter writes
//! back: a reasoning item with encrypted content, a failed web search call,
//! a message whose citations are kept as wire annotations, and function
//! calls with their item ids; a later turn holds a refusal with its message
//! phase. All of it must reach the wire as OpenAI wrote
//! it, so a change to how that state is stored cannot strand histories
//! written earlier; other providers' state never does.

use std::path::PathBuf;

use allama::history::History;
use allama::{
    ChatProvider, ChatRequest, ContentBlock, ContentPart, Message, Subject, UnsupportedPolicy,
    UserContent, Warning, WarningKind,
};
use allama_http::{ReplayResponse, ReplayTransport};
use allama_openai::{NAMESPACE, ResponsesProvider};
use serde_json::{Value, json};

/// Index of the Responses assistant turn in the golden history.
const RESPONSES_TURN: usize = 5;

/// Index of the failed tool result that answers it.
const FAILED_RESULT: usize = 6;

/// Index of the Responses turn that holds a refusal.
const REFUSAL_TURN: usize = 12;

fn golden_history() -> History {
    // A copy of allama's golden history, so the test runs from the
    // published package; a workspace test keeps the copy identical.
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/history_v1.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    let mut history: History = serde_json::from_str(&text).expect("the golden history loads");
    // Refused before sending, by design: a file another provider stores
    // (ADR 0002), and an image in an assistant turn, which Responses takes
    // no input for. Leave them out to replay the rest.
    for message in &mut history.messages {
        match message {
            Message::User(user) => {
                if let UserContent::Parts(parts) = &mut user.content {
                    parts.retain(|part| {
                        !matches!(part, ContentPart::File(file) if file.source.provider_file_id().is_some())
                    });
                }
            }
            Message::Assistant(assistant) => assistant
                .content
                .retain(|block| !matches!(block, ContentBlock::Image(_))),
            _ => {}
        }
    }
    history
}

/// Send `request` through the Responses adapter and return the request
/// body and the response warnings.
async fn replay(request: &ChatRequest) -> (Value, Vec<Warning>) {
    let transport = ReplayTransport::new([ReplayResponse::json(
        200,
        json!({
            "id": "resp_replay", "object": "response", "status": "completed", "model": "gpt-5.5",
            "output": [{"type": "message", "id": "msg_replay", "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "ok", "annotations": []}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        })
        .to_string(),
    )]);
    let provider = ResponsesProvider::builder()
        .api_key("replay-key")
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let response = provider.chat(request).await.unwrap();
    let sent = &transport.requests()[0];
    let body = serde_json::from_slice(sent.body.as_deref().expect("a request body")).unwrap();
    (body, response.warnings)
}

/// The input items from the golden Responses turn's reasoning item up to,
/// and including, the results that answer it.
fn responses_turn(body: &Value) -> Vec<Value> {
    let input = body["input"].as_array().unwrap();
    let start = input
        .iter()
        .position(|item| item["id"] == "rs_01")
        .expect("the Responses turn is replayed");
    let end = input
        .iter()
        .position(|item| item["type"] == "function_call_output" && item["call_id"] == "call_02")
        .expect("its results are replayed");
    input[start..=end].to_vec()
}

#[tokio::test]
async fn golden_history_replays_the_responses_turn_as_written() {
    let expected = json!([
        {"type": "reasoning", "id": "rs_01", "encrypted_content": "gAAAAABo1xyz",
         "summary": [{"type": "summary_text",
                      "text": "Check the issue tracker, then fetch the benchmark."}]},
        {"type": "web_search_call", "id": "ws_01", "status": "failed",
         "action": {"type": "search", "query": "rust 1.99 regressions"}},
        {"type": "message", "role": "assistant",
         "content": [{"type": "output_text",
                      "text": "The search failed; the notes file covers regressions.",
                      "annotations": [{"type": "file_citation", "file_id": "file-abc123",
                                       "filename": "notes.md", "index": 32}]}]},
        {"type": "function_call", "id": "fc_01", "call_id": "call_01",
         "name": "fetch_benchmark", "arguments": "{\"suite\":\"compile\"}"},
        // Verbatim: the API accepts any string, so malformed arguments are
        // replayed as the model produced them.
        {"type": "function_call", "call_id": "call_02", "name": "render_chart",
         "arguments": "{\"suite\":\"compile\""},
        // No error flag on the wire: the error text is the output.
        {"type": "function_call_output", "call_id": "call_01",
         "output": "benchmark service unavailable"},
        {"type": "function_call_output", "call_id": "call_02",
         "output": [{"type": "input_text", "text": "Rendered chart:"},
                    {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="}]},
    ]);

    // As stored, and after preparing the conversation for OpenAI.
    let mut prepared = golden_history();
    prepared.retain_provider_state(&[NAMESPACE]);
    for history in [golden_history(), prepared] {
        assert!(
            history.messages[RESPONSES_TURN]
                .as_assistant()
                .is_some_and(|turn| turn.tool_calls().any(|call| call.id == "call_01")),
            "RESPONSES_TURN points at the Responses turn"
        );
        let request = history.into_request("gpt-5.5");
        let (body, warnings) = replay(&request).await;
        assert_eq!(Value::Array(responses_turn(&body)), expected);
        // Nothing OpenAI produced is dropped.
        let losses: Vec<_> = warnings
            .iter()
            .filter(|w| w.kind == WarningKind::Dropped)
            .filter(|w| match &w.subject {
                Subject::Content { message, .. } => *message == RESPONSES_TURN,
                Subject::Messages { messages } => messages.contains(&RESPONSES_TURN),
                _ => false,
            })
            .map(|w| w.message.as_str())
            .collect();
        assert!(
            losses.is_empty(),
            "the Responses turn replays without loss: {losses:?}"
        );
    }
}

/// A stored refusal goes back as a `refusal` part of an assistant message,
/// in the phase it was written in.
#[tokio::test]
async fn golden_history_replays_the_refusal_as_written() {
    let mut prepared = golden_history();
    prepared.retain_provider_state(&[NAMESPACE]);
    for history in [golden_history(), prepared] {
        assert!(
            history.messages[REFUSAL_TURN]
                .as_assistant()
                .is_some_and(|turn| matches!(turn.content[..], [ContentBlock::Refusal(_)])),
            "REFUSAL_TURN points at the refusal"
        );
        let request = history.into_request("gpt-5.5");
        let (body, warnings) = replay(&request).await;
        let refusal = body["input"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["content"][0]["type"] == "refusal")
            .expect("the refusal is replayed");
        assert_eq!(
            refusal,
            &json!({"type": "message", "role": "assistant", "phase": "final_answer",
                    "content": [{"type": "refusal",
                                 "refusal": "I can't help with finding someone's home address."}]})
        );
        assert!(
            !warnings.iter().any(|w| match &w.subject {
                Subject::Content { message, .. } => *message == REFUSAL_TURN,
                Subject::Messages { messages } => messages.contains(&REFUSAL_TURN),
                _ => false,
            }),
            "{warnings:?}"
        );
    }
}

#[tokio::test]
async fn golden_history_reports_what_responses_cannot_replay() {
    let request = golden_history().into_request("gpt-5.5");
    let (body, warnings) = replay(&request).await;

    // Other providers' opaque state is dropped with warnings, never sent.
    let sent = body.to_string();
    for foreign in [
        "EqQBCkYIBxgCKkA",
        "EmwKAhgBEgy3va",
        "srvtoolu_01",
        "Eo8BCioIBxgC",
        "container_upload",
        "search_result",
        "compaction",
        "att_01",
        "CiQB0e2Kb3",
        "UklGRiQAAABXQVZF",
    ] {
        assert!(!sent.contains(foreign), "{foreign} was sent to Responses");
    }
    let codes: Vec<_> = warnings
        .iter()
        .filter(|w| w.kind == WarningKind::Dropped)
        .filter_map(|w| w.code.as_deref())
        .collect();
    for code in [
        "dropped.foreign_content",
        "dropped.foreign_provider_data",
        "dropped.foreign_reasoning",
    ] {
        assert!(codes.contains(&code), "{code} missing from {warnings:?}");
    }
    // The failed tool result is approximated: Responses has no error flag.
    let flag = warnings
        .iter()
        .find(|w| w.code.as_deref() == Some("approximated.tool_result_error"))
        .expect("the lost error flag is reported");
    assert_eq!(
        flag.subject,
        Subject::Content {
            message: FAILED_RESULT,
            block: None
        }
    );
}

/// A stored history that holds a failed tool
/// result replays under the strict policy too. The model reads the error
/// text either way; only the flag is lost, so nothing is refused.
#[tokio::test]
async fn golden_history_replays_under_the_strict_policy() {
    let mut history = golden_history();
    history.retain_provider_state(&[NAMESPACE]);
    let request = history
        .into_request("gpt-5.5")
        .unsupported_policy(UnsupportedPolicy::Error);
    let (body, warnings) = replay(&request).await;
    assert!(
        body["input"].as_array().unwrap().iter().any(|item| {
            item["call_id"] == "call_01" && item["output"] == "benchmark service unavailable"
        }),
        "{body}"
    );
    assert!(
        warnings
            .iter()
            .any(|w| w.code.as_deref() == Some("approximated.tool_result_error")),
        "{warnings:?}"
    );
}
