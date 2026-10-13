//! The empty-turn contract: an empty user or assistant message changes
//! nothing that is sent, and a reply left last by skipped turns is dropped.

use std::future::Future;

use allama::{
    ChatRequest, ChatResponse, ContentPart, Error, Message, Subject, UnsupportedPolicy, Warning,
    WarningKind,
};

use super::{CHAT, CHAT_STREAM, ProbeOutcome, describe, on_path};

/// Assert that empty user and assistant turns are skipped, and that an
/// assistant turn left last only by skipped turns is dropped.
///
/// A user message with no parts or only empty text, and an assistant
/// message with no blocks, is [empty](Message::is_empty): it holds nothing
/// to send. The contract sends conversations for `model` through `probe`,
/// which calls `chat`, and `stream_probe`, which collects `chat_stream` into
/// a response (see [streaming probes](super#streaming-probes)), and checks
/// on both paths that:
///
/// - **an empty turn in the middle of the conversation changes nothing**:
///   the request sends exactly the bodies of the same conversation without
///   it, with the same warnings (skipping an empty turn loses nothing, so it
///   is not reported);
/// - **an assistant turn that ends the conversation only because the turns
///   after it are empty is dropped**: the request sends the bodies of the
///   conversation without it and without the empty turns, and reports a
///   [`Dropped`](WarningKind::Dropped) warning naming the assistant message
///   (`Subject::Content` with no block). Like every `Dropped` warning, it
///   does not depend on the [`UnsupportedPolicy`], so the request succeeds
///   under [`UnsupportedPolicy::Error`] too;
/// - **an assistant turn the caller sent last**, as a prefill, is sent (or
///   refused with [`Error::Unsupported`]), never dropped;
/// - **a conversation of empty turns only** fails with
///   [`Error::InvalidRequest`] before anything is sent.
///
/// Adapters get this behaviour from
/// [`allama::adapter::chat::retain_turns`]. Bodies are compared as JSON, or as bytes
/// for probes that report raw bodies ([`ProbeOutcome::raw`]).
///
/// # Panics
///
/// When a conversation without empty turns fails, or the contract is
/// violated.
pub async fn assert_empty_turns_skipped<F, Fut, G, GFut>(
    model: &str,
    mut probe: F,
    mut stream_probe: G,
) where
    F: FnMut(ChatRequest) -> Fut,
    Fut: Future<Output = ProbeOutcome<ChatResponse>>,
    G: FnMut(ChatRequest) -> GFut,
    GFut: Future<Output = ProbeOutcome<ChatResponse>>,
{
    let mut problems = Vec::new();
    for case in CASES {
        for policy in case.policies() {
            problems.extend(on_path(CHAT, check(case, model, *policy, &mut probe).await));
            problems.extend(on_path(
                CHAT_STREAM,
                check(case, model, *policy, &mut stream_probe).await,
            ));
        }
    }
    for path in [CHAT, CHAT_STREAM] {
        let request = ChatRequest::new(model).message(Message::user(""));
        let outcome = if path == CHAT {
            probe(request).await
        } else {
            stream_probe(request).await
        };
        let problem = match &outcome.result {
            Err(Error::InvalidRequest { .. }) if outcome.nothing_sent() => None,
            Err(Error::InvalidRequest { .. }) => {
                Some("nothing to send: refused only after sending".to_owned())
            }
            other => Some(format!(
                "nothing to send: expected InvalidRequest before sending, got {}",
                describe(other)
            )),
        };
        problems.extend(on_path(path, problem));
    }
    assert!(
        problems.is_empty(),
        "empty turn contract failed:\n  {}",
        problems.join("\n  ")
    );
}

/// A conversation with empty turns, and what it must send.
struct Case {
    label: &'static str,
    /// The conversation, as messages.
    messages: fn() -> Vec<Message>,
    /// What must be sent instead.
    expect: Expect,
}

#[derive(Clone, Copy)]
enum Expect {
    /// The bodies of the conversation without the empty turns, with the
    /// same warnings.
    Skipped(fn() -> Vec<Message>),
    /// The bodies of the conversation without the empty turns and the
    /// assistant message at this index, which is reported as dropped.
    ReplyDropped(fn() -> Vec<Message>, usize),
    /// The conversation as written, with its last assistant turn.
    PrefillSent,
}

impl Case {
    /// Policies to probe under: a dropped reply must not depend on it.
    fn policies(&self) -> &'static [UnsupportedPolicy] {
        match self.expect {
            Expect::ReplyDropped(..) => &[UnsupportedPolicy::Warn, UnsupportedPolicy::Error],
            Expect::Skipped(_) | Expect::PrefillSent => &[UnsupportedPolicy::Warn],
        }
    }
}

const QUESTION: &str = "What is 2 + 2?";
const ANSWER: &str = "4";
const FOLLOW_UP: &str = "And 3 + 3?";

fn conversation() -> Vec<Message> {
    vec![
        Message::user(QUESTION),
        Message::assistant(ANSWER),
        Message::user(FOLLOW_UP),
    ]
}

fn question() -> Vec<Message> {
    vec![Message::user(QUESTION)]
}

fn no_parts() -> Message {
    Message::user_parts(Vec::new())
}

fn no_blocks() -> Message {
    Message::assistant_blocks(Vec::new())
}

const CASES: &[Case] = &[
    Case {
        label: "empty user text before the reply",
        messages: || {
            vec![
                Message::user(QUESTION),
                Message::user(""),
                Message::assistant(ANSWER),
                Message::user(FOLLOW_UP),
            ]
        },
        expect: Expect::Skipped(conversation),
    },
    Case {
        label: "user message without parts after the reply",
        messages: || {
            vec![
                Message::user(QUESTION),
                Message::assistant(ANSWER),
                no_parts(),
                Message::user(FOLLOW_UP),
            ]
        },
        expect: Expect::Skipped(conversation),
    },
    Case {
        label: "user message with only an empty text part",
        messages: || {
            vec![
                Message::user(QUESTION),
                Message::assistant(ANSWER),
                Message::user_parts(vec![ContentPart::text("")]),
                Message::user(FOLLOW_UP),
            ]
        },
        expect: Expect::Skipped(conversation),
    },
    Case {
        label: "assistant message without blocks",
        messages: || {
            vec![
                Message::user(QUESTION),
                no_blocks(),
                Message::assistant(ANSWER),
                no_blocks(),
                Message::user(FOLLOW_UP),
            ]
        },
        expect: Expect::Skipped(conversation),
    },
    Case {
        label: "reply followed only by an empty user message",
        messages: || {
            vec![
                Message::user(QUESTION),
                Message::assistant(ANSWER),
                Message::user(""),
            ]
        },
        expect: Expect::ReplyDropped(question, 1),
    },
    Case {
        label: "reply followed only by empty turns",
        messages: || {
            vec![
                Message::user(QUESTION),
                Message::assistant(ANSWER),
                no_blocks(),
                no_parts(),
            ]
        },
        expect: Expect::ReplyDropped(question, 1),
    },
    Case {
        label: "prefill",
        messages: || vec![Message::user(QUESTION), Message::assistant("{")],
        expect: Expect::PrefillSent,
    },
];

async fn check<F, Fut>(
    case: &Case,
    model: &str,
    policy: UnsupportedPolicy,
    probe: &mut F,
) -> Option<String>
where
    F: FnMut(ChatRequest) -> Fut,
    Fut: Future<Output = ProbeOutcome<ChatResponse>>,
{
    let request = |messages: Vec<Message>| {
        ChatRequest::new(model)
            .extend_messages(messages)
            .unsupported_policy(policy)
    };
    let label = match policy {
        UnsupportedPolicy::Error => format!("{} (UnsupportedPolicy::Error)", case.label),
        _ => case.label.to_owned(),
    };
    let outcome = probe(request((case.messages)())).await;
    let (expected, dropped) = match case.expect {
        Expect::Skipped(expected) => (expected, None),
        Expect::ReplyDropped(expected, message) => (expected, Some(message)),
        Expect::PrefillSent => {
            return check_prefill(&label, &outcome, probe(request(question())).await);
        }
    };
    let control = probe(request(expected())).await;
    let control_warnings = match &control.result {
        Ok(response) => &response.warnings,
        Err(err) => {
            return Some(format!(
                "{label}: the conversation without empty turns failed: {err}"
            ));
        }
    };
    let warnings = match &outcome.result {
        Ok(response) => &response.warnings,
        Err(err) => return Some(format!("{label}: failed: {err}")),
    };
    if outcome.differs(&control) {
        return Some(format!(
            "{label}: sent other bodies than the conversation without empty turns"
        ));
    }
    let (reported, rest): (Vec<&Warning>, Vec<&Warning>) = warnings
        .iter()
        .partition(|warning| dropped.is_some_and(|message| drops_reply(warning, message)));
    if let Some(message) = dropped
        && reported.is_empty()
    {
        return Some(format!(
            "{label}: no Dropped warning names the reply (message {message}) it dropped \
             (warnings: {warnings:?})"
        ));
    }
    let control_warnings: Vec<&Warning> = control_warnings.iter().collect();
    (rest != control_warnings).then(|| {
        format!(
            "{label}: the warnings differ from the conversation without empty turns \
             ({rest:?} instead of {control_warnings:?})"
        )
    })
}

/// A prefill is sent: the body differs from the question alone, and no
/// warning reports the reply dropped.
fn check_prefill(
    label: &str,
    outcome: &ProbeOutcome<ChatResponse>,
    question: ProbeOutcome<ChatResponse>,
) -> Option<String> {
    match &outcome.result {
        Err(Error::Unsupported { .. }) => None,
        Err(err) => Some(format!("{label}: failed: {err}")),
        Ok(response) if response.warnings.iter().any(|w| drops_reply(w, 1)) => Some(format!(
            "{label}: the assistant turn the caller sent last was dropped"
        )),
        Ok(_) if question.result.is_ok() && !outcome.differs(&question) => Some(format!(
            "{label}: the assistant turn the caller sent last is not on the wire"
        )),
        Ok(_) => None,
    }
}

fn drops_reply(warning: &Warning, message: usize) -> bool {
    warning.kind == WarningKind::Dropped
        && warning.subject
            == Subject::Content {
                message,
                block: None,
            }
}

#[cfg(test)]
mod tests {
    use allama::adapter::Degradations;
    use allama::adapter::chat::{EncodedTurn, retain_turns};
    use allama::{ChatResponse, ContentBlock, Result};
    use serde_json::{Value, json};

    use super::*;

    /// How the fake adapter treats empty turns.
    #[derive(Clone, Copy)]
    enum Adapter {
        /// Correct: uses `retain_turns`.
        Correct,
        /// Sends every message, empty or not.
        SendsEmpty,
        /// Skips empty turns but keeps a reply they leave last.
        KeepsTrailingReply,
        /// Skips empty turns with a warning.
        WarnsOnSkip,
        /// Sends an empty conversation.
        SendsNothingToSend,
    }

    /// Encode `request` the way `adapter` does, as one JSON body.
    fn send(adapter: Adapter, request: &ChatRequest) -> Result<(Value, Vec<Warning>)> {
        let mut degradations =
            Degradations::new("fake", request.unsupported_policy.unwrap_or_default());
        let turns: Vec<(EncodedTurn, Value)> = request
            .messages
            .iter()
            .enumerate()
            .map(|(index, message)| {
                let text = message.text().unwrap_or_default();
                let wire = json!({"role": message.role(), "text": text});
                (EncodedTurn::new(index, message, text.is_empty()), wire)
            })
            .collect();
        let kept: Vec<Value> = match adapter {
            Adapter::Correct => retain_turns(turns, &mut degradations, |(turn, _)| *turn)?
                .into_iter()
                .map(|(_, wire)| wire)
                .collect(),
            Adapter::SendsEmpty => turns.into_iter().map(|(_, wire)| wire).collect(),
            Adapter::KeepsTrailingReply | Adapter::SendsNothingToSend => turns
                .into_iter()
                .filter(|(turn, _)| !turn.is_empty())
                .map(|(_, wire)| wire)
                .collect(),
            Adapter::WarnsOnSkip => turns
                .into_iter()
                .filter(|(turn, _)| {
                    if turn.is_empty() {
                        degradations.dropped(
                            Subject::Content {
                                message: turn.message(),
                                block: None,
                            },
                            "skipped",
                        );
                    }
                    !turn.is_empty()
                })
                .map(|(_, wire)| wire)
                .collect(),
        };
        if kept.is_empty() && !matches!(adapter, Adapter::SendsNothingToSend) {
            return Err(Error::invalid_request("nothing to send"));
        }
        Ok((json!({"messages": kept}), degradations.into_warnings()))
    }

    async fn run(adapter: Adapter, request: ChatRequest) -> ProbeOutcome<ChatResponse> {
        match send(adapter, &request) {
            Ok((body, warnings)) => {
                let response = warnings.into_iter().fold(
                    ChatResponse::new(vec![ContentBlock::text("ok")]),
                    ChatResponse::warning,
                );
                ProbeOutcome::new(Ok(response), vec![body])
            }
            Err(err) => ProbeOutcome::new(Err(err), Vec::new()),
        }
    }

    async fn run_raw(adapter: Adapter, request: ChatRequest) -> ProbeOutcome<ChatResponse> {
        let outcome = run(adapter, request).await;
        let raw: Vec<Vec<u8>> = outcome
            .sent
            .iter()
            .map(|body| format!("raw:{body}").into_bytes())
            .collect();
        ProbeOutcome::raw(outcome.result, raw)
    }

    async fn contract(adapter: Adapter) {
        assert_empty_turns_skipped(
            "m",
            move |request| run(adapter, request),
            move |request| run_raw(adapter, request),
        )
        .await;
    }

    #[tokio::test]
    async fn a_correct_adapter_passes_on_json_and_raw_bodies() {
        contract(Adapter::Correct).await;
    }

    #[tokio::test]
    #[should_panic(expected = "chat: empty user text before the reply: sent other bodies")]
    async fn sending_empty_turns_fails() {
        contract(Adapter::SendsEmpty).await;
    }

    #[tokio::test]
    #[should_panic(
        expected = "chat_stream: reply followed only by empty turns (UnsupportedPolicy::Error): \
                    sent other bodies"
    )]
    async fn keeping_a_reply_left_last_fails_on_raw_bodies_too() {
        contract(Adapter::KeepsTrailingReply).await;
    }

    #[tokio::test]
    #[should_panic(expected = "assistant message without blocks: the warnings differ")]
    async fn warning_about_a_skip_fails() {
        contract(Adapter::WarnsOnSkip).await;
    }

    #[tokio::test]
    #[should_panic(expected = "chat: nothing to send: expected InvalidRequest before sending")]
    async fn sending_an_empty_conversation_fails() {
        contract(Adapter::SendsNothingToSend).await;
    }
}
