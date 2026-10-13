//! `history::validate` on a realistic mixed history: the golden fixture,
//! with client tool calls, provider-executed calls from three providers,
//! `Other` content and application data.

use allama::history::{self, History, HistoryIssue, ValidationMode};
use allama::{ChatRequest, Message};

const GOLDEN_HISTORY: &str = include_str!("fixtures/history_v1.json");

fn golden() -> History {
    serde_json::from_str(GOLDEN_HISTORY).expect("the golden history loads")
}

#[test]
fn the_golden_history_is_sendable_and_resumable() {
    let history = golden();
    for mode in [ValidationMode::Sendable, ValidationMode::Resumable] {
        assert_eq!(history.validate(mode), Ok(()), "{mode:?}");
        assert_eq!(history::validate(&history.messages, mode), Ok(()));
    }
}

#[test]
fn a_request_built_from_the_golden_history_validates() {
    let request = ChatRequest::new("model")
        .extend_messages(golden().messages)
        .user("One more question.");
    assert_eq!(
        history::validate(&request.messages, ValidationMode::Sendable),
        Ok(())
    );
}

#[test]
fn every_prefix_of_the_golden_history_is_resumable() {
    // Stopping anywhere, mid tool round trip included, leaves a history a
    // loop can resume; only some prefixes can be sent as they are.
    let messages = golden().messages;
    let mut sendable = Vec::new();
    for end in 0..=messages.len() {
        let prefix = &messages[..end];
        assert_eq!(
            history::validate(prefix, ValidationMode::Resumable),
            Ok(()),
            "prefix of {end} messages"
        );
        if history::validate(prefix, ValidationMode::Sendable).is_ok() {
            sendable.push(end);
        }
    }
    // Prefixes ending on an assistant message with tool calls (3, 6, 9) or
    // inside its run of results (4, 7) have pending calls.
    assert_eq!(sendable, [0, 1, 2, 5, 8, 10, 11, 12, 13, 14]);
}

/// The issues, as `(kind, message, call id)`: the variants are
/// non-exhaustive, so they are matched, not built.
fn summary(issues: &[HistoryIssue]) -> Vec<(&'static str, usize, &str)> {
    issues
        .iter()
        .map(|issue| {
            let kind = match issue {
                HistoryIssue::UnansweredToolCall { .. } => "unanswered",
                HistoryIssue::OrphanToolResult { .. } => "orphan",
                HistoryIssue::DuplicateToolCallId { .. } => "duplicate call id",
                HistoryIssue::DuplicateToolResult { .. } => "duplicate result",
                _ => "other",
            };
            (kind, issue.message(), issue.call_id().unwrap_or_default())
        })
        .collect()
}

#[test]
fn a_removed_result_is_reported_against_its_call() {
    let mut messages = golden().messages;
    let removed = messages.remove(4);
    let call_id = removed.as_tool().unwrap().tool_call_id.as_str();

    let expected = [("unanswered", 2, call_id)];
    let sendable = history::validate(&messages, ValidationMode::Sendable).unwrap_err();
    assert_eq!(summary(&sendable), expected);
    let resumable = history::validate(&messages, ValidationMode::Resumable).unwrap_err();
    assert_eq!(
        summary(&resumable),
        expected,
        "the call is not in the last assistant message"
    );
}

#[test]
fn a_user_message_inside_a_run_splits_it() {
    let mut messages = golden().messages;
    messages.insert(4, Message::user("Are you done?"));

    let issues = history::validate(&messages, ValidationMode::Resumable).unwrap_err();
    assert_eq!(
        summary(&issues),
        [("unanswered", 2, "toolu_02"), ("orphan", 5, "toolu_02")]
    );
}
