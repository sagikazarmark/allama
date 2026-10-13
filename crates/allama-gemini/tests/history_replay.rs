//! Replaying the golden history (`allama/tests/fixtures/history_v1.json`) to
//! Gemini.
//!
//! The fixture holds a Gemini turn with thought signatures in both stored
//! forms: in block provider data (`"google-ai"` / `"thought_signature"`, for
//! `Other` content too) and, for `Other` content, the wire form that keeps
//! `thoughtSignature` inside the verbatim part data. Both must reach the wire
//! on the part they arrived with, so a change to how signatures are stored
//! cannot strand histories written earlier.

use std::path::PathBuf;

use allama::history::History;
use allama::{
    ChatProvider, ChatRequest, ContentBlock, ContentPart, Message, Subject, UserContent,
    WarningKind,
};
use allama_gemini::{NAMESPACE, Provider};
use allama_http::{ReplayResponse, ReplayTransport};
use serde_json::{Value, json};

/// Index of the Gemini assistant turn in the golden history.
const GEMINI_TURN: usize = 8;

/// Index of the OpenAI turn that holds a refusal.
const REFUSAL_TURN: usize = 12;

fn golden_history() -> History {
    // A copy of allama's golden history, so the test runs from the
    // published package; a workspace test keeps the copy identical.
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/history_v1.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    let mut history: History = serde_json::from_str(&text).expect("the golden history loads");
    // A file another provider stores is refused before sending, by design
    // (ADR 0002): leave it out to replay the rest.
    for message in &mut history.messages {
        if let Message::User(user) = message
            && let UserContent::Parts(parts) = &mut user.content
        {
            parts.retain(|part| {
                !matches!(part, ContentPart::File(file) if file.source.provider_file_id().is_some())
            });
        }
    }
    history
}

/// Send `request` through the Gemini adapter and return the request body
/// and the response warnings.
async fn replay(request: &ChatRequest) -> (Value, Vec<allama::Warning>) {
    let transport = ReplayTransport::new([ReplayResponse::json(
        200,
        json!({
            "responseId": "resp-replay",
            "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
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

/// The model content holding the golden Gemini turn.
fn gemini_turn(body: &Value) -> &Vec<Value> {
    body["contents"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|content| content["role"] == "model")
        .map(|content| content["parts"].as_array().unwrap())
        .find(|parts| {
            parts
                .iter()
                .any(|part| part["functionCall"]["name"] == "publish_summary")
        })
        .expect("the Gemini turn is replayed")
}

#[tokio::test]
async fn golden_history_replays_both_thought_signature_forms() {
    let expected = json!([
        {"text": "Summarize, then compute the speedup.", "thought": true,
         "thoughtSignature": "CiQB0e2Kb3-thought"},
        {"text": "Compile times improved.", "thoughtSignature": "CiQB0e2Kb3-text"},
        {"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="},
         "thoughtSignature": "CiQB0e2Kb3-image"},
        {"executableCode": {"code": "print(round(41.2 / 37.9, 2))", "language": "PYTHON"},
         "thoughtSignature": "CiQB0e2Kb3-code"},
        {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "1.09\n"}},
        // Wire form: the signature stored inside `Other` data.
        {"inlineData": {"mimeType": "audio/wav", "data": "UklGRiQAAABXQVZF"},
         "thoughtSignature": "CiQB0e2Kb3-wire"},
        // Decoded form: the signature in the `Other` block's provider data.
        {"inlineData": {"mimeType": "audio/wav", "data": "UklGRigAAABXQVZF"},
         "thoughtSignature": "CiQB0e2Kb3-other"},
        {"functionCall": {"id": "fc_g1", "name": "publish_summary", "args": {"speedup": 1.09}},
         "thoughtSignature": "CiQB0e2Kb3-call"},
    ]);

    // As stored, and after preparing the conversation for Gemini.
    let mut prepared = golden_history();
    prepared.retain_provider_state(&[NAMESPACE]);
    for history in [golden_history(), prepared] {
        assert!(
            history.messages[GEMINI_TURN]
                .as_assistant()
                .is_some_and(|turn| turn.tool_calls().any(|call| call.id == "fc_g1")),
            "GEMINI_TURN points at the Gemini turn"
        );
        let request = history.into_request("gemini-3-flash");
        let (body, warnings) = replay(&request).await;
        assert_eq!(&Value::Array(gemini_turn(&body).clone()), &expected);
        // Nothing Gemini produced is dropped, and the call's `"allama"`
        // provider data is core-owned, so it is not reported either.
        let losses: Vec<_> = warnings
            .iter()
            .filter(|w| {
                matches!(
                    w.subject,
                    Subject::Content {
                        message: GEMINI_TURN,
                        ..
                    }
                )
            })
            .map(|w| w.message.as_str())
            .collect();
        assert!(
            losses.is_empty(),
            "the Gemini turn replays without loss: {losses:?}"
        );
    }
}

#[tokio::test]
async fn golden_history_reports_what_gemini_cannot_replay() {
    let request = golden_history().into_request("gemini-3-flash");
    let (body, warnings) = replay(&request).await;

    // Foreign opaque state is dropped with warnings, never sent.
    let sent = body.to_string();
    for foreign in [
        "EqQBCkYIBxgCKkA",
        "EmwKAhgBEgy3va",
        "srvtoolu_01",
        "gAAAAABo1xyz",
        "ws_01",
        "container_upload",
        "search_result",
    ] {
        assert!(!sent.contains(foreign), "{foreign} was sent to Gemini");
    }
    assert!(
        warnings.iter().any(|w| w.kind == WarningKind::Dropped),
        "{warnings:?}"
    );
    // Calls without a Gemini signature get Google's documented placeholder.
    assert!(sent.contains("skip_thought_signature_validator"));
}

/// A refusal another provider wrote is replayed as model text; only its
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
    let (body, warnings) = replay(&history.into_request("gemini-3-flash")).await;
    let text = json!({"text": "I can't help with finding someone's home address."});
    assert!(
        body["contents"]
            .as_array()
            .unwrap()
            .iter()
            .any(|content| content["role"] == "model" && content["parts"] == json!([text])),
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

/// The wire form on its own: an `Other` part whose signature sits inside
/// its nested wire data, with no provider data, still reaches the wire.
#[tokio::test]
async fn wire_signature_inside_other_data_replays() {
    let history: History = serde_json::from_value(json!({
        "format_version": 1,
        "messages": [
            {"role": "user", "content": "Read it aloud."},
            {"role": "assistant", "content": [{
                "type": "other",
                "type_name": "inline_data",
                "origin": "google-ai",
                "data": {
                    "inlineData": {"mimeType": "audio/wav", "data": "UklGRiQAAABXQVZF"},
                    "thoughtSignature": "CiQB0e2Kb3-wire"
                }
            }]},
            {"role": "user", "content": "Again."}
        ]
    }))
    .expect("the nested wire form loads");
    let (body, warnings) = replay(&history.into_request("gemini-2.5-flash")).await;
    assert_eq!(
        body["contents"][1]["parts"],
        json!([{"inlineData": {"mimeType": "audio/wav", "data": "UklGRiQAAABXQVZF"},
                "thoughtSignature": "CiQB0e2Kb3-wire"}])
    );
    assert!(warnings.is_empty(), "{warnings:?}");
}

/// A file Anthropic stores cannot be sent to Gemini: the request is
/// refused before sending.
#[tokio::test]
async fn foreign_provider_file_is_refused() {
    let request = ChatRequest::new("gemini-3-flash").message(Message::user_parts(vec![
        ContentPart::text("Summarize it."),
        ContentPart::File(allama::File::provider_file(
            "application/pdf",
            "anthropic",
            "file_011CNha8iCJcU1wXNR6q4V8w",
        )),
    ]));
    let provider = Provider::builder()
        .api_key("replay-key")
        .http_transport(ReplayTransport::default())
        .build()
        .unwrap();
    let err = provider.chat(&request).await.unwrap_err();
    assert!(matches!(err, allama::Error::Unsupported { .. }), "{err:?}");
    assert!(err.to_string().contains("anthropic"), "{err}");
}
