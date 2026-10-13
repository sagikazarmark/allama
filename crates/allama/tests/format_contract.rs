//! Serialization format contract (history format version 1).
//!
//! These shapes are what persisted conversations and recorded exchanges
//! contain.
//!
//! # The golden history
//!
//! [`GOLDEN_HISTORY`] (`tests/fixtures/history_v1.json`) is one complete
//! [`History`]: every message role, with every content
//! block, content part, user content, tool result content, media source and
//! citation source kind that exists, plus the opaque provider state real
//! adapters store (reasoning signatures, citations' raw annotations, Gemini
//! thought signatures in both the provider-data form and the wire form, a
//! verbatim Gemini part with its `thoughtSignature` inside `Other` data).
//! `Other` content appears with
//! provider data and with wire data whose keys are named like the fields
//! allama owns (`type`, `type_name`, `origin`, `provider_data`). Application data appears
//! on a message of every role; the messages without it pin that an empty
//! map is not written.
//!
//! Three tests guard the format, each against a different change:
//!
//! - `golden_history_round_trips` loads the fixture and writes it back, so a
//!   change to a shape the fixture holds fails there. A change the fixture
//!   does not exercise, such as a new optional field, round-trips unnoticed.
//! - `history_format_paths_are_pinned` (in `src/chat/accounted.rs`)
//!   serializes a history with every field set and compares its key paths
//!   with `src/chat/history_v1_paths.txt`, so a new, renamed or removed
//!   field fails there.
//! - `golden_history_covers_every_kind` (in `src/history/golden.rs`) lists
//!   what the fixture covers. Its matches are exhaustive, so a new block,
//!   part, source or enum value fails to compile there until it has a label,
//!   and the fixture must then hold it.
//!
//! # Changing the format deliberately
//!
//! The fixture changes **only** in the change that alters the format, never
//! to make an unrelated change pass:
//!
//! 1. Edit `history_v1.json` by hand: add an entry for each new block, part,
//!    message or field, or rewrite the entries whose shape changed.
//! 2. Extend `golden_history_covers_every_kind` with the new kinds,
//!    `history_v1_paths.txt` with the new key paths, and
//!    `golden_history_decodes_to_typed_values` with a typed check of the new
//!    shape.
//!
//! Before 0.1 the format stays at version 1 and the fixture is edited in
//! place. After 0.1, any change also bumps
//! `allama::history::HISTORY_FORMAT_VERSION`, additive ones included, and
//! ships only in a minor release (the rules are in the `allama::history`
//! docs, "Format evolution"): keep `history_v1.json` as the proof that
//! older histories still load, and add a fixture for the new version.
//!
//! Provider crates replay the same file through a symlink in their own
//! package (`allama-anthropic`, `allama-gemini` and `allama-openai`, in
//! `tests/history_replay.rs`), so a change here reaches them, and may need
//! their expectations updated too.

use allama::history::{HISTORY_FORMAT_VERSION, History, TURN_ENVELOPE_KEY, TurnEnvelope};
use allama::{
    CacheBreakpoint, CacheRetention, ChatRequest, ChatResponse, Citation, CitationSource,
    ContentBlock, ContentPart, File, FinishReason, Image, ImageDetail, Message, OtherContent,
    PromptCache, ProviderData, ProviderToolCall, ProviderToolResult, Reasoning, ReasoningEffort,
    Refusal, RequestField, StreamEvent, Subject, Text, ToolCall, ToolResultContent, Usage,
    UserContent, Warning, WarningKind,
};
use serde_json::json;

/// The golden history: every persisted shape in one conversation.
const GOLDEN_HISTORY: &str = include_str!("fixtures/history_v1.json");

fn round_trip<T>(value: &T, expected: serde_json::Value)
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let json = serde_json::to_value(value).unwrap();
    assert_eq!(json, expected);
    let back: T = serde_json::from_value(json).unwrap();
    assert_eq!(&back, value);
}

fn golden_history() -> History {
    serde_json::from_str(GOLDEN_HISTORY).expect("the golden history loads")
}

/// Normalized JSON of a history: compares equal when the bytes would be
/// equal up to whitespace and key order.
fn golden_value() -> serde_json::Value {
    serde_json::from_str(GOLDEN_HISTORY).unwrap()
}

#[test]
fn format_version_is_one() {
    assert_eq!(HISTORY_FORMAT_VERSION, 1);
}

#[test]
fn golden_history_round_trips() {
    let history = golden_history();
    assert_eq!(
        history.loaded_format_version(),
        Some(HISTORY_FORMAT_VERSION)
    );

    let written = serde_json::to_value(&history).unwrap();
    assert_eq!(
        written,
        golden_value(),
        "the History format changed; see the header of format_contract.rs"
    );
    let reloaded: History = serde_json::from_value(written).unwrap();
    assert_eq!(reloaded, history);
}

#[test]
fn golden_history_decodes_to_typed_values() {
    let history = golden_history();
    let messages = &history.messages;
    // User input.
    let user = messages[0].as_user().unwrap();
    assert_eq!(
        user.content,
        UserContent::Text("What changed in Rust 1.99?".into())
    );
    assert_eq!(
        messages[0]
            .app_data_get::<String>("myapp.created_at")
            .unwrap()
            .as_deref(),
        Some("2026-10-03T09:00:00Z")
    );
    assert!(
        messages[0].app_data_get::<u64>("myapp.message_id").is_err(),
        "a type mismatch is an error"
    );
    assert!(messages[1].app_data().is_empty());
    let user = messages[1].as_user().unwrap();
    let UserContent::Parts(parts) = &user.content else {
        panic!("expected parts");
    };
    assert_eq!(
        parts[1],
        ContentPart::Image(
            Image::url("https://example.com/charts/compile-times")
                .with_media_type("image/png")
                .with_detail(ImageDetail::Low)
        )
    );
    assert_eq!(
        parts[3],
        ContentPart::File(
            File::base64("application/pdf", "JVBERi0xLjcK").with_filename("release-notes.pdf")
        )
    );
    assert_eq!(
        parts[4],
        ContentPart::File(
            File::url("text/plain", "https://example.com/CHANGELOG.txt")
                .with_filename("CHANGELOG.txt")
        )
    );
    assert_eq!(
        parts.last(),
        Some(&ContentPart::File(File::provider_file(
            "application/pdf",
            "anthropic",
            "file_011CNha8iCJcU1wXNR6q4V8w"
        )))
    );
    let ContentPart::Other(search_result) = &parts[5] else {
        panic!("expected other content");
    };
    assert_eq!(search_result.type_name, "search_result");
    assert_eq!(search_result.origin.as_deref(), Some("anthropic"));
    assert_eq!(search_result.data["title"], "Rust 1.99 in review");
    assert_eq!(
        parts[6],
        ContentPart::Other(
            OtherContent::new(
                "transcript",
                serde_json::from_value(json!({
                    "type": "meeting",
                    "type_name": "minutes",
                    "origin": "zoom",
                    "provider_data": {"speaker": "mark"},
                    "text": "Ship the release notes on Friday.",
                }))
                .unwrap(),
            )
            .with_origin("openai-compat:acme")
            .with_provider_data(ProviderData::new().with_raw(
                "openai-compat:acme",
                "attachment_id",
                json!("att_01")
            ))
        ),
        "wire data keeps keys named like the fields allama owns"
    );
    assert_eq!(
        user.cache,
        Some(CacheBreakpoint::with_retention(CacheRetention::Short))
    );

    // An Anthropic turn: reasoning, provider tools, citations, `Other`.
    let turn = messages[2].as_assistant().unwrap();
    #[derive(Debug, PartialEq, serde::Deserialize)]
    struct TurnUsage {
        input_tokens: u64,
        output_tokens: u64,
    }
    assert_eq!(
        messages[2]
            .app_data_get::<TurnUsage>("myapp.usage")
            .unwrap(),
        Some(TurnUsage {
            input_tokens: 1200,
            output_tokens: 310
        })
    );
    assert!(
        turn.provider_data.is_empty(),
        "application data is not provider data"
    );
    assert_eq!(turn.cache, Some(CacheBreakpoint::new()));
    assert_eq!(
        turn.content[0],
        ContentBlock::Reasoning(
            Reasoning::new("The user wants the highlights; search first.").with_provider_data(
                ProviderData::new().with_raw("anthropic", "signature", json!("EqQBCkYIBxgCKkA"))
            )
        )
    );
    assert!(turn.content[1].as_reasoning().unwrap().is_opaque());
    assert_eq!(
        turn.content[2],
        ContentBlock::ProviderToolCall(
            ProviderToolCall::new(
                "srvtoolu_01",
                "web_search",
                r#"{"query":"Rust 1.99 release"}"#
            )
            .with_origin("anthropic")
        )
    );
    let result = turn.content[3].as_provider_tool_result().unwrap();
    assert_eq!(result.call_id, "srvtoolu_01");
    assert_eq!(result.origin.as_deref(), Some("anthropic"));
    assert_eq!(result.output[0]["type"], "web_search_result");
    let ContentBlock::Text(text) = &turn.content[4] else {
        panic!("expected text");
    };
    assert_eq!(
        text.citations[0],
        Citation::new(CitationSource::url(
            "https://blog.rust-lang.org/2026/09/18/Rust-1.99.0/",
            Some("Announcing Rust 1.99.0".into()),
        ))
        .with_span(0..9)
        .with_cited_text("Announcing Rust 1.99.0")
    );
    assert_eq!(text.cited_span(&text.citations[0]), Some("Rust 1.99"));
    assert_eq!(
        text.citations[1].source,
        CitationSource::document(Some(0), Some("release-notes.pdf".into()))
    );
    assert!(
        text.provider_data
            .get_raw("anthropic", "citations")
            .is_some()
    );
    assert_eq!(turn.content[5].type_name(), "container_upload");
    assert_eq!(turn.content[5].origin(), Some("anthropic"));
    assert_eq!(turn.tool_calls().count(), 2);
    assert_eq!(
        turn.content[8],
        ContentBlock::Other(
            OtherContent::new(
                "compaction",
                serde_json::from_value(
                    json!({"content": "The user asked what changed in Rust 1.99."})
                )
                .unwrap(),
            )
            .with_origin("anthropic")
            .with_provider_data(ProviderData::new().with_raw(
                "anthropic",
                "signature",
                json!("EpIBCkgIBxgC-compaction")
            ))
        )
    );

    // Tool results: text, object.
    assert_eq!(
        messages[3],
        Message::tool_result("toolu_01", "get_weather", "Sunny, 18 C")
            .with_app_data("myapp.duration_ms", 412)
    );
    assert_eq!(
        messages[4].as_tool().unwrap().content.as_object().unwrap()["stable_since"],
        "1.99"
    );

    // An OpenAI turn: encrypted reasoning, a failed provider tool, a file
    // citation, malformed arguments, message-level provider data.
    let turn = messages[5].as_assistant().unwrap();
    assert_eq!(
        turn.provider_data.get_raw("openai", "response_id"),
        Some(&json!("resp_01"))
    );
    assert_eq!(
        turn.content[2],
        ContentBlock::ProviderToolResult(
            ProviderToolResult::new("ws_01", json!({"status": "failed"}))
                .with_error(true)
                .with_origin("openai")
                .with_provider_data(ProviderData::new().with_raw(
                    "openai",
                    "type",
                    json!("web_search_call")
                ))
        )
    );
    let ContentBlock::Text(text) = &turn.content[3] else {
        panic!("expected text");
    };
    assert_eq!(text.cited_span(&text.citations[0]), Some("notes file"));
    assert_eq!(
        turn.content[5],
        ContentBlock::ToolCall(ToolCall::new(
            "call_02",
            "render_chart",
            r#"{"suite":"compile""#
        )),
        "malformed arguments are kept verbatim"
    );

    // Tool results: error, multimodal parts.
    assert_eq!(
        messages[6],
        Message::tool_error(
            "call_01",
            "fetch_benchmark",
            "benchmark service unavailable"
        )
    );
    assert_eq!(
        messages[7].as_tool().unwrap().content,
        ToolResultContent::Parts(vec![
            ContentPart::text("Rendered chart:"),
            ContentPart::image_base64("image/png", "iVBORw0KGgo="),
        ])
    );

    // A Gemini turn: thought signatures in the provider-data form on every
    // block that has provider data, and in the wire form inside `Other`
    // data.
    let turn = messages[8].as_assistant().unwrap();
    let signature = |block: &ContentBlock| {
        block
            .provider_data()
            .get_raw("google-ai", "thought_signature")
            .cloned()
    };
    assert_eq!(
        turn.content[0],
        ContentBlock::Reasoning(
            Reasoning::new("Summarize, then compute the speedup.").with_provider_data(
                ProviderData::new()
                    .with_raw("google-ai", "thought", json!(true))
                    .with_raw(
                        "google-ai",
                        "thought_signature",
                        json!("CiQB0e2Kb3-thought")
                    )
            )
        )
    );
    assert_eq!(
        turn.content[1],
        ContentBlock::Text(Text::new("Compile times improved.").with_provider_data(
            ProviderData::new().with_raw(
                "google-ai",
                "thought_signature",
                json!("CiQB0e2Kb3-text")
            )
        ))
    );
    assert_eq!(signature(&turn.content[2]), Some(json!("CiQB0e2Kb3-image")));
    assert_eq!(signature(&turn.content[3]), Some(json!("CiQB0e2Kb3-code")));
    assert_eq!(
        turn.content[5],
        ContentBlock::Other(
            OtherContent::new(
                "inline_data",
                serde_json::from_value(json!({
                    "inlineData": {"mimeType": "audio/wav", "data": "UklGRiQAAABXQVZF"},
                    "thoughtSignature": "CiQB0e2Kb3-wire",
                }))
                .unwrap(),
            )
            .with_origin("google-ai")
        ),
        "the wire form keeps the signature inside the verbatim part"
    );
    assert_eq!(turn.content[6].type_name(), "inline_data");
    assert_eq!(
        signature(&turn.content[6]),
        Some(json!("CiQB0e2Kb3-other")),
        "the current form keeps the signature in provider data"
    );
    assert_eq!(signature(&turn.content[7]), Some(json!("CiQB0e2Kb3-call")));
    assert!(
        turn.content[7]
            .provider_data()
            .namespace("allama")
            .is_some()
    );

    // A request the model refused: an image the provider sizes itself, a
    // long-lived cache breakpoint, and a refusal with OpenAI's phase.
    let user = messages[11].as_user().unwrap();
    let UserContent::Parts(parts) = &user.content else {
        panic!("expected parts");
    };
    assert_eq!(
        parts[1],
        ContentPart::Image(
            Image::url("https://example.com/charts/regressions")
                .with_media_type("image/png")
                .with_detail(ImageDetail::Auto)
        )
    );
    assert_eq!(
        user.cache,
        Some(CacheBreakpoint::with_retention(CacheRetention::Long))
    );
    assert_eq!(
        messages[12].as_assistant().unwrap().content,
        [ContentBlock::Refusal(
            Refusal::new("I can't help with finding someone's home address.").with_provider_data(
                ProviderData::new().with_raw("openai", "phase", json!("final_answer"))
            )
        )]
    );

    // Application data holds any JSON the application stores.
    let last = messages.last().unwrap();
    assert_eq!(
        last.app_data()["myapp.ui"],
        json!({"collapsed": false, "reactions": ["thumbs_up"]})
    );
}

/// Switching providers on a mixed history: what remains still round-trips,
/// carries no foreign state, and a second pass finds nothing to remove.
#[test]
fn golden_history_retains_provider_state() {
    let app_data = |history: &History| -> Vec<serde_json::Value> {
        history
            .messages
            .iter()
            .map(Message::app_data)
            .filter(|data| !data.is_empty())
            .map(|data| serde_json::Value::Object(data.clone()))
            .collect()
    };
    let original_app_data = app_data(&golden_history());
    for keep in [&["anthropic"][..], &["openai"], &["google-ai"], &[]] {
        let mut history = golden_history();
        let warnings = history.retain_provider_state(keep);
        assert_eq!(
            history.messages.len(),
            golden_history().messages.len(),
            "{keep:?}: no message is removed"
        );
        assert_eq!(
            app_data(&history),
            original_app_data,
            "{keep:?}: application data is untouched"
        );
        assert!(!warnings.is_empty(), "{keep:?}");
        assert!(history.retain_provider_state(keep).is_empty(), "{keep:?}");

        let kept = |namespace: &str| namespace == "allama" || keep.contains(&namespace);
        for message in &history.messages {
            assert!(message.provider_data().namespaces().all(|(ns, _)| kept(ns)));
            match message {
                Message::Assistant(turn) => {
                    for block in &turn.content {
                        assert!(block.origin().is_none_or(kept), "{keep:?}: {block}");
                        assert!(
                            block.provider_data().namespaces().all(|(ns, _)| kept(ns)),
                            "{keep:?}: {block}"
                        );
                    }
                }
                Message::User(user) => {
                    if let UserContent::Parts(parts) = &user.content {
                        for part in parts {
                            if let ContentPart::Other(other) = part {
                                assert!(other.origin.as_deref().is_none_or(kept), "{keep:?}");
                                assert!(
                                    other.provider_data.namespaces().all(|(ns, _)| kept(ns)),
                                    "{keep:?}"
                                );
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        let written = serde_json::to_value(&history).unwrap();
        assert_eq!(serde_json::from_value::<History>(written).unwrap(), history);
    }
}

#[test]
fn content_blocks() {
    let mut call = ToolCall::new("call_1", "lookup", r#"{"q":"x"}"#);
    call.provider_data
        .insert_raw("google-ai", "thought_signature", json!("sig"));
    round_trip(
        &vec![
            ContentBlock::text("hello"),
            ContentBlock::ToolCall(call),
            ContentBlock::Reasoning(Reasoning::new("think").with_provider_data(
                ProviderData::new().with_raw("anthropic", "signature", json!("s")),
            )),
            ContentBlock::Image(Image::base64("image/png", "AAAA")),
            ContentBlock::Other(
                OtherContent::new(
                    "server_tool_use",
                    serde_json::from_value(json!({"id": "x"})).unwrap(),
                )
                .with_origin("anthropic"),
            ),
        ],
        json!([
            {"type": "text", "text": "hello"},
            {"type": "tool_call", "id": "call_1", "name": "lookup", "arguments": "{\"q\":\"x\"}",
             "provider_data": {"google-ai": {"thought_signature": "sig"}}},
            {"type": "reasoning", "text": "think", "provider_data": {"anthropic": {"signature": "s"}}},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
            {"type": "other", "type_name": "server_tool_use", "origin": "anthropic", "data": {"id": "x"}},
        ]),
    );
}

#[test]
fn messages_and_parts() {
    round_trip(
        &vec![
            Message::user(vec![
                ContentPart::text("look"),
                ContentPart::Image(Image::url("https://x/y.png").with_detail(ImageDetail::Low)),
                ContentPart::File(
                    File::base64("application/pdf", "JVBERi0=").with_filename("doc.pdf"),
                ),
            ])
            .with_cache(CacheBreakpoint::with_retention(CacheRetention::Short)),
            Message::assistant("ok"),
            Message::tool_result(
                "call_1",
                "lookup",
                ToolResultContent::json(&json!({"temp": 18})).unwrap(),
            ),
            Message::tool_error("call_2", "lookup", "boom"),
        ],
        json!([
            {"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}, "detail": "low"},
                {"type": "file", "media_type": "application/pdf",
                 "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="},
                 "filename": "doc.pdf"},
             ], "cache": {"retention": "short"}},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "tool", "tool_call_id": "call_1", "name": "lookup", "content": {"temp": 18}},
            {"role": "tool", "tool_call_id": "call_2", "name": "lookup", "content": "boom", "is_error": true},
        ]),
    );
}

#[test]
fn request_shape() {
    let request = ChatRequest::new("m")
        .system("Be brief")
        .user("hi")
        .max_output_tokens(64)
        .reasoning(ReasoningEffort::Low)
        .cache(PromptCache::new().key("s1"))
        .with_options_json("openai", json!({"service_tier": "flex"}));
    round_trip(
        &request,
        json!({
            "model": "m",
            "system": [{"content": "Be brief"}],
            "messages": [{"role": "user", "content": "hi"}],
            "max_output_tokens": 64,
            "reasoning": {"effort": "low"},
            "cache": {"key": "s1"},
            "options": {"openai": {"service_tier": "flex"}},
        }),
    );
}

#[test]
fn response_shape() {
    let response = ChatResponse::new(vec![ContentBlock::text("hi")])
        .finish_reason(FinishReason::Refusal)
        .finish_reason_raw("refusal")
        .usage(
            Usage::new()
                .input_tokens(5)
                .output_tokens(1)
                .cache_read_input_tokens(3),
        )
        .provider("anthropic")
        .warning(Warning::unsupported(RequestField::Seed, "ignored").with_provider("anthropic"));
    round_trip(
        &response,
        json!({
            "content": [{"type": "text", "text": "hi"}],
            "finish_reason": "refusal",
            "finish_reason_raw": "refusal",
            "usage": {"input_tokens": 5, "output_tokens": 1, "cache_read_input_tokens": 3},
            "provider": "anthropic",
            "warnings": [{"kind": "unsupported", "subject": {"type": "field", "field": "seed"},
                          "provider": "anthropic", "message": "ignored"}],
        }),
    );
}

#[test]
fn stream_event_shapes() {
    round_trip(
        &vec![
            StreamEvent::response_start(Some("r".into()), None),
            StreamEvent::tool_call_start(0, "c", "f"),
            StreamEvent::tool_call_delta(0, "{}"),
            StreamEvent::block_provider_data(
                0,
                ProviderData::new().with_raw("google-ai", "thought_signature", json!("s")),
            ),
            StreamEvent::block_stop(0),
            StreamEvent::ResponseStop,
        ],
        json!([
            {"event": "response_start", "id": "r"},
            {"event": "block_start", "index": 0, "block_type": "tool_call", "id": "c", "name": "f"},
            {"event": "tool_call_delta", "index": 0, "arguments": "{}"},
            {"event": "block_provider_data", "index": 0, "provider_data": {"google-ai": {"thought_signature": "s"}}},
            {"event": "block_stop", "index": 0},
            {"event": "response_stop"},
        ]),
    );
}

/// An `Other` block starts a stream with the `other` block type and its
/// wire type name in `type_name`, the same key the content form uses.
#[test]
fn other_block_start_shape() {
    let other = OtherContent::new(
        "compaction",
        serde_json::from_value(json!({"content": "Summary."})).unwrap(),
    )
    .with_origin("anthropic")
    .with_provider_data(ProviderData::new().with_raw("anthropic", "signature", json!("s")));
    round_trip(
        &StreamEvent::other_start(1, other),
        json!({
            "event": "block_start",
            "index": 1,
            "block_type": "other",
            "type_name": "compaction",
            "data": {"content": "Summary."},
            "origin": "anthropic",
            "provider_data": {"anthropic": {"signature": "s"}},
        }),
    );
}

/// A turn envelope stored under the published key survives a `History`
/// round trip and switching providers, with the shape tools read.
#[test]
fn turn_envelope_in_application_data() {
    assert_eq!(TURN_ENVELOPE_KEY, "allama.turn");
    let response = ChatResponse::new(vec![ContentBlock::text("Hello!")])
        .finish_reason(FinishReason::Stop)
        .finish_reason_raw("end_turn")
        .usage(Usage::new().input_tokens(12).output_tokens(3))
        .model("claude-sonnet-4-5")
        .id("msg_01")
        .request_id("req_01")
        .provider("anthropic")
        .warning(Warning::unsupported(RequestField::Seed, "ignored").with_provider("anthropic"));
    let envelope = response.envelope();

    let mut history = golden_history();
    let mut reply = response.to_assistant_message();
    reply.app_data_insert(TURN_ENVELOPE_KEY, &envelope).unwrap();
    history.messages.push(reply);

    assert_eq!(
        history.messages.last().unwrap().app_data()[TURN_ENVELOPE_KEY],
        json!({
            "id": "msg_01",
            "request_id": "req_01",
            "model": "claude-sonnet-4-5",
            "provider": "anthropic",
            "usage": {"input_tokens": 12, "output_tokens": 3},
            "finish_reason": "stop",
            "finish_reason_raw": "end_turn",
            "warnings": [{"kind": "unsupported", "subject": {"type": "field", "field": "seed"},
                          "provider": "anthropic", "message": "ignored"}],
        }),
    );

    let stored = serde_json::to_string(&history).unwrap();
    let loaded: History = serde_json::from_str(&stored).unwrap();
    assert_eq!(loaded, history);
    // `retain_provider_state` never removes a message, so the reply stays
    // last with its envelope.
    let read = |history: &History| {
        history
            .messages
            .last()
            .unwrap()
            .app_data_get::<TurnEnvelope>(TURN_ENVELOPE_KEY)
            .unwrap()
    };
    assert_eq!(read(&loaded), Some(envelope.clone()));

    for keep in [&["anthropic"][..], &["openai"], &[]] {
        let mut history = loaded.clone();
        history.retain_provider_state(keep);
        assert_eq!(read(&history), Some(envelope.clone()), "{keep:?}");
    }
}

/// An envelope written by a newer allama, with a warning kind, request
/// field and subject this version does not know (and a field it does not
/// have), still loads: the unknown values read as `WarningKind::Other`,
/// `RequestField::Unknown` and `Subject::Unknown`.
#[test]
fn turn_envelope_from_a_newer_version() {
    let stored = json!({
        "id": "msg_02",
        "finish_reason": "tool_calls",
        "finish_reason_raw": "tool_use",
        "served_region": "eu",
        "warnings": [
            {"kind": "rerouted", "subject": {"type": "response"}, "message": "a"},
            {"kind": "unsupported", "subject": {"type": "field", "field": "logit_bias"},
             "message": "b"},
            {"kind": "dropped", "subject": {"type": "audio_track", "track": 1}, "message": "c"},
        ],
    });
    let mut reply = Message::assistant("Calling a tool.");
    reply
        .app_data_mut()
        .insert(TURN_ENVELOPE_KEY.into(), stored);
    let loaded: History = serde_json::from_str(
        &serde_json::to_string(&History::new(vec![Message::user("hi"), reply])).unwrap(),
    )
    .unwrap();

    let turn: TurnEnvelope = loaded.messages[1]
        .app_data_get(TURN_ENVELOPE_KEY)
        .unwrap()
        .unwrap();
    assert_eq!(turn.id.as_deref(), Some("msg_02"));
    assert_eq!(turn.finish_reason, Some(FinishReason::ToolCalls));
    assert_eq!(turn.finish_reason_raw.as_deref(), Some("tool_use"));
    let read: Vec<_> = turn
        .warnings
        .iter()
        .map(|warning| (warning.kind, warning.subject.clone()))
        .collect();
    assert_eq!(
        read,
        [
            (WarningKind::Other, Subject::Response),
            (
                WarningKind::Unsupported,
                Subject::Field {
                    field: RequestField::Unknown
                }
            ),
            (WarningKind::Dropped, Subject::Unknown),
        ]
    );
}

#[test]
fn history_envelope() {
    let history = History::new(vec![Message::user("hi")]);
    round_trip(
        &history,
        json!({"format_version": 1, "messages": [{"role": "user", "content": "hi"}]}),
    );
}

/// `Other` content named like each of allama's own kinds keeps its own
/// tag, so it loads as `Other` and never as the native kind.
#[test]
fn other_content_named_like_a_native_kind() {
    for name in [
        "text",
        "image",
        "file",
        "tool_call",
        "reasoning",
        "refusal",
        "provider_tool_call",
        "provider_tool_result",
    ] {
        let other = OtherContent::new(
            name,
            serde_json::from_value(json!({"text": "verbatim"})).unwrap(),
        )
        .with_origin("google-ai");
        let expected = json!({"type": "other", "type_name": name, "origin": "google-ai", "data": {"text": "verbatim"}});
        round_trip(&ContentBlock::Other(other.clone()), expected.clone());
        round_trip(&ContentPart::Other(other), expected);
    }
}

/// `Other` content is read only in the form it is written in. A history
/// holding a form written before (wire fields beside `type`, or the wire
/// type name in `type` itself) fails to load instead of being reinterpreted.
#[test]
fn earlier_other_content_forms_do_not_load() {
    let load = |block: serde_json::Value| {
        serde_json::from_value::<History>(json!({
            "format_version": 1,
            "messages": [
                {"role": "user", "content": "Read it aloud."},
                {"role": "assistant", "content": [block]}
            ]
        }))
        .unwrap_err()
        .to_string()
    };
    let flat = load(json!({
        "type": "other",
        "type_name": "inline_data",
        "inlineData": {"mimeType": "audio/wav", "data": "UklGRiQAAABXQVZF"},
        "thoughtSignature": "CiQB0e2Kb3-flat",
        "origin": "google-ai"
    }));
    assert!(
        flat.contains("`Other` content `inline_data` must name its wire type in `type_name` and nest its wire fields under `data`"),
        "{flat}"
    );
    let untagged = load(json!({
        "type": "inline_data",
        "origin": "google-ai",
        "data": {"inlineData": {"mimeType": "audio/wav", "data": "UklGRiQAAABXQVZF"}}
    }));
    assert!(
        untagged.contains("unknown content block type `inline_data` (written by a newer allama?)"),
        "{untagged}"
    );
}
