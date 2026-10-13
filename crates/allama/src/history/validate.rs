use std::collections::HashSet;
use std::fmt;

use crate::Message;

/// Check that tool calls and tool results in `messages` are well formed.
///
/// Providers reject a history whose tool results are out of place with an
/// HTTP 400 that rarely says which message is wrong. This check applies
/// the rule they share (see
/// [Tool calls and results](crate::history#tool-calls-and-results)) and
/// names every message that breaks it:
///
/// - the results for an assistant message's tool calls form one
///   contiguous run of [`Message::Tool`] messages immediately after it, in
///   any order ([`HistoryIssue::UnansweredToolCall`],
///   [`HistoryIssue::OrphanToolResult`]);
/// - tool call ids are unique across the whole history, and each call has
///   one result ([`HistoryIssue::DuplicateToolCallId`],
///   [`HistoryIssue::DuplicateToolResult`]). Anthropic rejects an id used
///   twice anywhere in the conversation, not only within one turn.
///
/// Only client tool calls ([`ContentBlock::ToolCall`](crate::ContentBlock::ToolCall))
/// take part. Calls the provider ran itself
/// ([`ContentBlock::ProviderToolCall`](crate::ContentBlock::ProviderToolCall))
/// carry their results in the same assistant message and never produce an
/// issue. Nothing else is checked: content, roles around the tool calls,
/// provider data and application data are left to the adapters.
///
/// `mode` says whether the history is about to be sent
/// ([`ValidationMode::Sendable`]) or may stop in the middle of a tool round
/// trip ([`ValidationMode::Resumable`]).
///
/// Returns every issue found, in message order.
///
/// # Errors
///
/// Returns the issues when there is at least one.
pub fn validate(messages: &[Message], mode: ValidationMode) -> Result<(), Vec<HistoryIssue>> {
    let mut issues = Vec::new();
    // Client tool call ids of the history so far, this turn included.
    let mut seen: HashSet<&str> = HashSet::new();
    let mut index = 0;
    while index < messages.len() {
        let turn = match &messages[index] {
            Message::Assistant(turn) => turn,
            Message::Tool(result) => {
                issues.push(HistoryIssue::OrphanToolResult {
                    message: index,
                    call_id: result.tool_call_id.clone(),
                });
                index += 1;
                continue;
            }
            Message::User(_) => {
                index += 1;
                continue;
            }
        };

        // Client tool calls of this turn, in order, without duplicates. A
        // call reusing an earlier turn's id is reported, and still belongs
        // to this turn, so the results right after it answer it.
        let mut calls: Vec<&str> = Vec::new();
        for call in turn.tool_calls() {
            let id = call.id.as_str();
            if !seen.insert(id) {
                issues.push(HistoryIssue::DuplicateToolCallId {
                    message: index,
                    call_id: call.id.clone(),
                });
            }
            if !calls.contains(&id) {
                calls.push(id);
            }
        }

        // The run of tool messages right after the turn holds its results.
        let mut answered: HashSet<&str> = HashSet::new();
        let mut run_issues = Vec::new();
        let mut next = index + 1;
        while let Some(Message::Tool(result)) = messages.get(next) {
            let id = result.tool_call_id.as_str();
            if !calls.contains(&id) {
                run_issues.push(HistoryIssue::OrphanToolResult {
                    message: next,
                    call_id: result.tool_call_id.clone(),
                });
            } else if !answered.insert(id) {
                run_issues.push(HistoryIssue::DuplicateToolResult {
                    message: next,
                    call_id: result.tool_call_id.clone(),
                });
            }
            next += 1;
        }

        // Calls still waiting at the end of the history are pending, which
        // only a resumable history allows.
        let pending_allowed = mode == ValidationMode::Resumable && next == messages.len();
        if !pending_allowed {
            issues.extend(calls.iter().filter(|id| !answered.contains(*id)).map(|id| {
                HistoryIssue::UnansweredToolCall {
                    message: index,
                    call_id: (*id).to_owned(),
                }
            }));
        }
        issues.append(&mut run_issues);
        index = next;
    }

    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

/// What [`validate`] checks a history for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ValidationMode {
    /// The history is about to be sent: every client tool call has its
    /// result. Use it before
    /// [`ChatProvider::chat`](crate::ChatProvider::chat).
    Sendable,
    /// The history may stop in the middle of a tool round trip, as an agent
    /// loop's does between the model's turn and the tool results: the last
    /// assistant message may have calls without results, or only some of
    /// them, as long as nothing but tool results follows it. Use it to
    /// check what a failed or interrupted turn left behind before storing
    /// it; run the pending calls and append their results before sending.
    Resumable,
}

/// A tool call or tool result that providers would reject; see [`validate`].
///
/// `message` is the index of the message the issue is about, in the slice
/// passed to [`validate`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HistoryIssue {
    /// A client tool call in the assistant message at `message` has no
    /// result in the run of tool messages right after it.
    ///
    /// Either its result is missing, or it comes later, after another
    /// message (then it is also an [`OrphanToolResult`](Self::OrphanToolResult)).
    /// In [`ValidationMode::Resumable`], calls of the last assistant message
    /// may still be waiting for results.
    #[non_exhaustive]
    UnansweredToolCall {
        /// Index of the assistant message.
        message: usize,
        /// Id of the tool call.
        call_id: String,
    },
    /// The tool message at `message` answers no client tool call of the
    /// assistant message before its run of tool messages.
    ///
    /// The call may not exist, belong to an earlier assistant message, or
    /// be a provider-executed call (which has no tool message), or the tool
    /// message may follow a user message.
    #[non_exhaustive]
    OrphanToolResult {
        /// Index of the tool message.
        message: usize,
        /// The tool call id the tool message names.
        call_id: String,
    },
    /// The assistant message at `message` has a client tool call whose id
    /// an earlier client tool call already uses, in the same message or an
    /// earlier one; reported once for each repeat, at the message of the
    /// repeat.
    ///
    /// Ids must be unique across the whole history: Anthropic rejects a
    /// conversation that reuses one, even in another turn. Test scripts
    /// that answer every turn with the same id (`"c1"`) hit this too.
    #[non_exhaustive]
    DuplicateToolCallId {
        /// Index of the assistant message.
        message: usize,
        /// The repeated tool call id.
        call_id: String,
    },
    /// The tool message at `message` answers a tool call that an earlier
    /// tool message in the same run already answered.
    #[non_exhaustive]
    DuplicateToolResult {
        /// Index of the repeated tool message.
        message: usize,
        /// Id of the tool call.
        call_id: String,
    },
}

impl HistoryIssue {
    /// Index of the message the issue is about.
    #[must_use]
    pub fn message(&self) -> usize {
        match self {
            Self::UnansweredToolCall { message, .. }
            | Self::OrphanToolResult { message, .. }
            | Self::DuplicateToolCallId { message, .. }
            | Self::DuplicateToolResult { message, .. } => *message,
        }
    }

    /// Id of the tool call the issue is about, if it concerns one.
    #[must_use]
    pub fn call_id(&self) -> Option<&str> {
        match self {
            Self::UnansweredToolCall { call_id, .. }
            | Self::OrphanToolResult { call_id, .. }
            | Self::DuplicateToolCallId { call_id, .. }
            | Self::DuplicateToolResult { call_id, .. } => Some(call_id),
        }
    }
}

impl fmt::Display for HistoryIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnansweredToolCall { message, call_id } => write!(
                f,
                "message {message}: tool call `{call_id}` has no result in the tool messages right after it"
            ),
            Self::OrphanToolResult { message, call_id } => write!(
                f,
                "message {message}: tool result for `{call_id}` answers no tool call of the assistant message before it"
            ),
            Self::DuplicateToolCallId { message, call_id } => write!(
                f,
                "message {message}: tool call id `{call_id}` is used more than once"
            ),
            Self::DuplicateToolResult { message, call_id } => write!(
                f,
                "message {message}: tool call `{call_id}` already has a result"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContentBlock, ProviderToolCall, ProviderToolResult};
    use serde_json::json;

    use HistoryIssue::{
        DuplicateToolCallId, DuplicateToolResult, OrphanToolResult, UnansweredToolCall,
    };
    use ValidationMode::{Resumable, Sendable};

    const BOTH: [ValidationMode; 2] = [Sendable, Resumable];

    fn calls(ids: &[&str]) -> Message {
        Message::assistant_blocks(
            ids.iter()
                .map(|id| ContentBlock::tool_call(*id, "lookup", "{}"))
                .collect(),
        )
    }

    fn result(id: &str) -> Message {
        Message::tool_result(id, "lookup", "ok")
    }

    fn unanswered(message: usize, id: &str) -> HistoryIssue {
        UnansweredToolCall {
            message,
            call_id: id.into(),
        }
    }

    fn orphan(message: usize, id: &str) -> HistoryIssue {
        OrphanToolResult {
            message,
            call_id: id.into(),
        }
    }

    fn issues(messages: &[Message], mode: ValidationMode) -> Vec<HistoryIssue> {
        validate(messages, mode).err().unwrap_or_default()
    }

    #[test]
    fn complete_tool_round_trips_pass_in_both_modes() {
        let messages = [
            Message::user("Weather in Paris and Tokyo?"),
            calls(&["a", "b"]),
            result("b"),
            result("a"),
            Message::assistant("Sunny in both."),
            Message::user("And Rome?"),
            calls(&["c"]),
            result("c"),
        ];
        for mode in BOTH {
            assert_eq!(validate(&messages, mode), Ok(()), "{mode:?}");
        }
        assert_eq!(validate(&[], Sendable), Ok(()));
    }

    #[test]
    fn unanswered_call_followed_by_another_message_fails_in_both_modes() {
        let messages = [
            Message::user("go"),
            calls(&["a", "b"]),
            result("a"),
            Message::user("never mind"),
        ];
        for mode in BOTH {
            assert_eq!(issues(&messages, mode), [unanswered(1, "b")], "{mode:?}");
        }

        let messages = [Message::user("go"), calls(&["a"]), Message::assistant("…")];
        for mode in BOTH {
            assert_eq!(issues(&messages, mode), [unanswered(1, "a")], "{mode:?}");
        }
    }

    #[test]
    fn pending_calls_at_the_end_are_only_resumable() {
        // A trailing assistant message whose calls have no results yet.
        let trailing = [Message::user("go"), calls(&["a", "b"])];
        assert_eq!(
            issues(&trailing, Sendable),
            [unanswered(1, "a"), unanswered(1, "b")]
        );
        assert_eq!(validate(&trailing, Resumable), Ok(()));

        // A partial run of results at the end.
        let partial = [Message::user("go"), calls(&["a", "b"]), result("b")];
        assert_eq!(issues(&partial, Sendable), [unanswered(1, "a")]);
        assert_eq!(validate(&partial, Resumable), Ok(()));
    }

    #[test]
    fn resumable_allows_pending_calls_only_in_the_last_turn() {
        let messages = [
            Message::user("go"),
            calls(&["a"]),
            Message::user("still there?"),
            calls(&["b"]),
        ];
        assert_eq!(issues(&messages, Resumable), [unanswered(1, "a")]);
        assert_eq!(
            issues(&messages, Sendable),
            [unanswered(1, "a"), unanswered(3, "b")]
        );
    }

    #[test]
    fn orphan_results_fail_in_both_modes() {
        let cases: [(&[Message], HistoryIssue); 4] = [
            // No assistant message before it.
            (&[result("a")], orphan(0, "a")),
            // After a user message.
            (&[Message::user("hi"), result("a")], orphan(1, "a")),
            // After an assistant message without tool calls.
            (
                &[
                    Message::user("hi"),
                    Message::assistant("hello"),
                    result("a"),
                ],
                orphan(2, "a"),
            ),
            // Answering a call the assistant message did not make.
            (
                &[Message::user("go"), calls(&["a"]), result("a"), result("z")],
                orphan(3, "z"),
            ),
        ];
        for (messages, expected) in cases {
            for mode in BOTH {
                assert_eq!(
                    issues(messages, mode),
                    std::slice::from_ref(&expected),
                    "{mode:?}"
                );
            }
        }
    }

    #[test]
    fn a_result_after_another_message_is_out_of_place() {
        // The result exists, but a user message separates it from its call:
        // OpenAI and Anthropic both reject this.
        let messages = [
            Message::user("go"),
            calls(&["a"]),
            Message::user("hurry up"),
            result("a"),
        ];
        for mode in BOTH {
            assert_eq!(
                issues(&messages, mode),
                [unanswered(1, "a"), orphan(3, "a")],
                "{mode:?}"
            );
        }

        // A result answering a call of an earlier assistant message.
        let messages = [
            Message::user("go"),
            calls(&["a"]),
            result("a"),
            Message::assistant("done"),
            Message::user("again"),
            calls(&["b"]),
            result("a"),
            result("b"),
        ];
        for mode in BOTH {
            assert_eq!(issues(&messages, mode), [orphan(6, "a")], "{mode:?}");
        }
    }

    #[test]
    fn duplicate_call_ids_fail_in_both_modes() {
        let messages = [
            Message::user("go"),
            calls(&["a", "a", "b"]),
            result("a"),
            result("b"),
        ];
        for mode in BOTH {
            assert_eq!(
                issues(&messages, mode),
                [DuplicateToolCallId {
                    message: 1,
                    call_id: "a".into()
                }],
                "{mode:?}"
            );
        }
    }

    /// Anthropic rejects an id reused in a later turn: the repeat is
    /// reported where it occurs, and its results still answer it.
    #[test]
    fn call_ids_reused_across_turns_fail_in_both_modes() {
        let messages = [
            Message::user("Weather in Paris?"),
            calls(&["call_0"]),
            result("call_0"),
            Message::assistant("Sunny."),
            Message::user("And Tokyo?"),
            calls(&["call_0"]),
            result("call_0"),
        ];
        for mode in BOTH {
            assert_eq!(
                issues(&messages, mode),
                [DuplicateToolCallId {
                    message: 5,
                    call_id: "call_0".into()
                }],
                "{mode:?}"
            );
        }

        // A pending repeat in the last turn is still a repeat, and each
        // repeat is reported, whichever turn the first use is in.
        let messages = [
            Message::user("go"),
            calls(&["a"]),
            result("a"),
            Message::user("again"),
            calls(&["a", "a"]),
        ];
        let duplicate = DuplicateToolCallId {
            message: 4,
            call_id: "a".into(),
        };
        assert_eq!(
            issues(&messages, Resumable),
            [duplicate.clone(), duplicate.clone()]
        );
        assert_eq!(
            issues(&messages, Sendable),
            [duplicate.clone(), duplicate, unanswered(4, "a")]
        );
    }

    #[test]
    fn duplicate_results_fail_in_both_modes() {
        let messages = [
            Message::user("go"),
            calls(&["a", "b"]),
            result("a"),
            result("a"),
            result("b"),
        ];
        for mode in BOTH {
            assert_eq!(
                issues(&messages, mode),
                [DuplicateToolResult {
                    message: 3,
                    call_id: "a".into()
                }],
                "{mode:?}"
            );
        }

        // A duplicate in a trailing partial run is still a duplicate.
        let messages = [
            Message::user("go"),
            calls(&["a", "b"]),
            result("a"),
            result("a"),
        ];
        assert_eq!(
            issues(&messages, Resumable),
            [DuplicateToolResult {
                message: 3,
                call_id: "a".into()
            }]
        );
    }

    #[test]
    fn issues_are_reported_in_message_order() {
        let messages = [
            result("x"),
            Message::user("go"),
            calls(&["a", "a"]),
            result("z"),
            Message::user("again"),
        ];
        let found = issues(&messages, Sendable);
        assert_eq!(
            found,
            [
                orphan(0, "x"),
                DuplicateToolCallId {
                    message: 2,
                    call_id: "a".into()
                },
                unanswered(2, "a"),
                orphan(3, "z"),
            ]
        );
        let order: Vec<usize> = found.iter().map(HistoryIssue::message).collect();
        assert!(order.is_sorted(), "{order:?}");
    }

    #[test]
    fn provider_executed_calls_never_produce_an_issue() {
        let provider_call = |id: &str| {
            ContentBlock::from(
                ProviderToolCall::new(id, "web_search", "{}").with_origin("anthropic"),
            )
        };
        let provider_result = |id: &str| {
            ContentBlock::from(
                ProviderToolResult::new(id, json!({"status": "completed"}))
                    .with_origin("anthropic"),
            )
        };
        let histories: [Vec<Message>; 4] = [
            // A provider call with its result, followed by a user message.
            vec![
                Message::user("Search"),
                Message::assistant_blocks(vec![
                    provider_call("s1"),
                    provider_result("s1"),
                    ContentBlock::text("Found it."),
                ]),
                Message::user("Thanks"),
            ],
            // A provider call whose result never came, mid-history.
            vec![
                Message::user("Search"),
                Message::assistant_blocks(vec![provider_call("s1")]),
                Message::user("Hello?"),
            ],
            // Mixed with a client call that is answered; the provider call
            // and a client call share an id without clashing.
            vec![
                Message::user("Search and look up"),
                Message::assistant_blocks(vec![
                    provider_call("s1"),
                    provider_result("s1"),
                    ContentBlock::tool_call("c1", "lookup", "{}"),
                    ContentBlock::tool_call("s1", "lookup", "{}"),
                ]),
                result("c1"),
                result("s1"),
            ],
            // Repeated provider call ids.
            vec![
                Message::user("Search twice"),
                Message::assistant_blocks(vec![provider_call("s1"), provider_call("s1")]),
            ],
        ];
        for messages in &histories {
            for mode in BOTH {
                assert_eq!(validate(messages, mode), Ok(()), "{mode:?}: {messages:?}");
            }
        }

        // A tool message cannot answer a provider-executed call.
        let messages = [
            Message::user("Search"),
            Message::assistant_blocks(vec![provider_call("s1")]),
            result("s1"),
        ];
        assert_eq!(issues(&messages, Sendable), [orphan(2, "s1")]);
    }

    #[test]
    fn accessors_and_display() {
        let issue = unanswered(1, "call_1");
        assert_eq!(issue.message(), 1);
        assert_eq!(issue.call_id(), Some("call_1"));
        assert_eq!(
            issue.to_string(),
            "message 1: tool call `call_1` has no result in the tool messages right after it"
        );
        assert_eq!(
            orphan(3, "call_9").to_string(),
            "message 3: tool result for `call_9` answers no tool call of the assistant message before it"
        );
    }
}
