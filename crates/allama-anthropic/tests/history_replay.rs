//! Replaying the golden history (`allama/tests/fixtures/history_v1.json`) to
//! Anthropic.
//!
//! The fixture holds an Anthropic turn with every kind of state the adapter
//! writes back: signed and redacted thinking, a server tool call and its
//! result (stored without its wire type, as a hand-built history may be),
//! a text block whose citations carry an encrypted index, and verbatim `Other`
//! blocks (`container_upload`, `compaction`), next to a user turn with an
//! Anthropic `search_result` part and a file Anthropic stores. All of it
//! must reach the wire as Anthropic wrote it, so a change to how that state
//! is stored cannot strand histories written earlier; other providers'
//! state never does.

use std::path::PathBuf;

use allama::history::History;
use allama::{
    ChatProvider, ChatRequest, ContentBlock, ContentPart, MediaSource, Message, Subject,
    UserContent, Warning, WarningKind,
};
use allama_anthropic::{NAMESPACE, Provider};
use allama_http::{ReplayResponse, ReplayTransport};
use serde_json::{Value, json};

/// Index of the Anthropic assistant turn in the golden history.
const ANTHROPIC_TURN: usize = 2;

/// Index of the OpenAI turn that holds a refusal.
const REFUSAL_TURN: usize = 12;

fn golden_history() -> History {
    // A copy of allama's golden history, so the test runs from the
    // published package; a workspace test keeps the copy identical.
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/history_v1.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    let mut history: History = serde_json::from_str(&text).expect("the golden history loads");
    // Refused before sending, by design: Anthropic takes text/plain
    // documents inline or as a stored file only, and no image in an
    // assistant turn. Leave them out to replay the rest.
    for message in &mut history.messages {
        match message {
            Message::User(user) => {
                if let UserContent::Parts(parts) = &mut user.content {
                    parts.retain(|part| {
                        !matches!(part, ContentPart::File(file)
                            if file.media_type == "text/plain"
                                && matches!(file.source, MediaSource::Url { .. }))
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

/// Send `request` through the Anthropic adapter and return the request body
/// and the response warnings.
async fn replay(request: &ChatRequest) -> (Value, Vec<Warning>) {
    let transport = ReplayTransport::new([ReplayResponse::json(
        200,
        json!({
            "id": "msg_replay", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })
        .to_string(),
    )]);
    let provider = Provider::builder()
        .api_key("replay-key")
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let response = provider.chat(request).await.unwrap();
    let sent = &transport.requests()[0];
    let body = serde_json::from_slice(sent.body.as_deref().expect("a request body")).unwrap();
    (body, response.warnings)
}

/// The assistant content holding the golden Anthropic turn.
fn anthropic_turn(body: &Value) -> &Vec<Value> {
    body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|message| message["role"] == "assistant")
        .map(|message| message["content"].as_array().unwrap())
        .find(|content| content.iter().any(|block| block["id"] == "toolu_01"))
        .expect("the Anthropic turn is replayed")
}

#[tokio::test]
async fn golden_history_replays_the_anthropic_turn_as_written() {
    let expected = json!([
        {"type": "thinking", "thinking": "The user wants the highlights; search first.",
         "signature": "EqQBCkYIBxgCKkA"},
        {"type": "redacted_thinking", "data": "EmwKAhgBEgy3va"},
        {"type": "server_tool_use", "id": "srvtoolu_01", "name": "web_search",
         "input": {"query": "Rust 1.99 release"}},
        // The stored result has no wire `type`; Anthropic's docs fix it for
        // a `web_search` call, so it is sent with its call.
        {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_01",
         "content": [{"type": "web_search_result",
                      "url": "https://blog.rust-lang.org/2026/09/18/Rust-1.99.0/",
                      "title": "Announcing Rust 1.99.0",
                      "encrypted_content": "Eo8BCioIBxgC"}]},
        // The wire citations, encrypted index included, from provider data.
        {"type": "text", "text": "Rust 1.99 stabilizes async closures in traits.",
         "citations": [{"type": "web_search_result_location",
                        "url": "https://blog.rust-lang.org/2026/09/18/Rust-1.99.0/",
                        "cited_text": "Announcing Rust 1.99.0",
                        "encrypted_index": "Eo8BCioIBxgC"}]},
        {"type": "container_upload", "file_id": "file_011CNha8iCJcU1wXNR6q4V8w"},
        {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "Paris"}},
        {"type": "tool_use", "id": "toolu_02", "name": "lookup_docs",
         "input": {"topic": "async closures"}},
        // The message's cache breakpoint goes on its last block.
        {"type": "compaction", "content": "The user asked what changed in Rust 1.99.",
         "cache_control": {"type": "ephemeral"}},
    ]);

    // As stored, and after preparing the conversation for Anthropic.
    let mut prepared = golden_history();
    prepared.retain_provider_state(&[NAMESPACE]);
    for history in [golden_history(), prepared] {
        assert!(
            history.messages[ANTHROPIC_TURN]
                .as_assistant()
                .is_some_and(|turn| turn.tool_calls().any(|call| call.id == "toolu_01")),
            "ANTHROPIC_TURN points at the Anthropic turn"
        );
        let request = history.into_request("claude-opus-5-5");
        let (body, warnings) = replay(&request).await;
        assert_eq!(&Value::Array(anthropic_turn(&body).clone()), &expected);

        // The user turn's Anthropic parts are sent as written too.
        let user = body["messages"][0]["content"].as_array().unwrap();
        assert!(user.contains(&json!({
            "type": "search_result",
            "source": "https://example.com/blog/rust-1-99",
            "title": "Rust 1.99 in review",
            "content": [{"type": "text", "text": "Async closures in traits are stable."}],
        })));
        assert!(user.iter().any(|part| part["type"] == "document"
            && part["source"]
                == json!({"type": "file", "file_id": "file_011CNha8iCJcU1wXNR6q4V8w"})));

        // Nothing of the Anthropic turn is dropped.
        let about_turn = |subject: &Subject| match subject {
            Subject::Content { message, .. } => *message == ANTHROPIC_TURN,
            Subject::Messages { messages } => messages.contains(&ANTHROPIC_TURN),
            _ => false,
        };
        assert!(
            !warnings
                .iter()
                .any(|w| w.kind == WarningKind::Dropped && about_turn(&w.subject)),
            "{warnings:?}"
        );
    }
}

#[tokio::test]
async fn golden_history_reports_what_anthropic_cannot_replay() {
    let request = golden_history().into_request("claude-opus-5-5");
    let (body, warnings) = replay(&request).await;

    // Other providers' opaque state is dropped with warnings, never sent.
    let sent = body.to_string();
    for foreign in [
        "gAAAAABo1xyz",
        "rs_01",
        "ws_01",
        "fc_01",
        "CiQB0e2Kb3",
        "att_01",
        "UklGRiQAAABXQVZF",
        "OUTCOME_OK",
    ] {
        assert!(!sent.contains(foreign), "{foreign} was sent to Anthropic");
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
    // Arguments that are not JSON are sent as `{}`, with a coded warning.
    let render = body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|message| message["content"].as_array().unwrap())
        .find(|block| block["id"] == "call_02")
        .expect("the malformed call is replayed");
    assert_eq!(render["input"], json!({}));
    assert!(
        warnings
            .iter()
            .any(|w| w.code.as_deref() == Some("approximated.tool_call_arguments"))
    );
    // Anthropic has a tool error flag: the error is sent with it, no warning.
    let failed = body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|message| message["content"].as_array().unwrap())
        .find(|block| block["tool_use_id"] == "call_01")
        .expect("the failed result is replayed");
    assert_eq!(failed["is_error"], true);
    assert!(
        !warnings
            .iter()
            .any(|w| w.code.as_deref() == Some("approximated.tool_result_error"))
    );
}

/// A refusal another provider wrote is replayed as assistant text; only its
/// provider data is dropped.
#[tokio::test]
async fn golden_history_replays_a_foreign_refusal_as_text() {
    let history = golden_history();
    assert!(
        history.messages[REFUSAL_TURN]
            .as_assistant()
            .is_some_and(|turn| matches!(turn.content[..], [ContentBlock::Refusal(_)])),
        "REFUSAL_TURN points at the refusal"
    );
    let (body, warnings) = replay(&history.into_request("claude-opus-5-5")).await;
    let text = json!({"type": "text", "text": "I can't help with finding someone's home address."});
    assert!(
        body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .any(|message| message["role"] == "assistant" && message["content"] == json!([text])),
        "{body}"
    );
    let about_refusal: Vec<_> = warnings
        .iter()
        .filter(|w| match &w.subject {
            Subject::Content { message, .. } => *message == REFUSAL_TURN,
            Subject::Messages { messages } => messages.contains(&REFUSAL_TURN),
            _ => false,
        })
        .filter_map(|w| w.code.as_deref())
        .collect();
    assert_eq!(about_refusal, ["dropped.foreign_provider_data"]);
}
