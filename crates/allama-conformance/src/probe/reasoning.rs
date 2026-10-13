//! The unreplayable-reasoning contract: reasoning a model cannot take back
//! never reaches the wire, and the warnings name the caller's blocks.

use std::future::Future;

use allama::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, ContentBlock,
    Message, OtherContent, Subject, UnsupportedPolicy, Warning, WarningKind,
};
use serde_json::Value;

use super::{CHAT, CHAT_STREAM, ProbeOutcome, describe, on_path};

/// Text of the reasoning block [`assert_unreplayable_reasoning_dropped`]
/// replays; it must never reach the wire.
pub const REASONING_PROBE_TEXT: &str = "allama-probe-unreplayable-reasoning";

/// Origin and verbatim data of the foreign `Other` block
/// [`assert_unreplayable_reasoning_dropped`] puts after the reasoning; it
/// must never reach the wire.
pub const REASONING_PROBE_FOREIGN: &str = "allama-probe-foreign-after-reasoning";

/// Assert that assistant reasoning never reaches the wire for a model whose
/// [`ReasoningReplay`](ChatCapability::ReasoningReplay) is
/// [`Unsupported`](CapabilitySupport::Unsupported), and that the warnings
/// about the turn name the caller's blocks.
///
/// The conversation replays an assistant turn of three blocks (message 1):
/// a reasoning block ([`REASONING_PROBE_TEXT`]), an `Other` block from
/// another provider ([`REASONING_PROBE_FOREIGN`]), and the answer text. It
/// is sent through `probe`, which calls `chat`, and `stream_probe`, which
/// collects `chat_stream` into a response (see
/// [streaming probes](super#streaming-probes)), under both
/// [`UnsupportedPolicy`] values. On every path and policy:
///
/// - the request is sent and succeeds: dropping replay state is a
///   [`Dropped`](WarningKind::Dropped) warning under every policy, never a
///   refusal;
/// - neither probe value appears in any body, searched as bytes so JSON and
///   raw bodies ([`ProbeOutcome::raw`]) are checked alike;
/// - a `Dropped` warning names the reasoning, `Subject::Content` with
///   message 1 and block 0, and another names the foreign block, message 1
///   and block 1. [`allama::adapter::chat::preflight`] removes the reasoning before
///   the encoder runs, so the encoder sees the foreign block at index 0; an
///   adapter that records the encoder's warnings into the same
///   [`Degradations`](allama::adapter::Degradations) reports it at the caller's
///   index 1, and one that collects them apart reports index 0.
///
/// `provider` answers capabilities for `model`.
///
/// # Panics
///
/// When `provider` does not report `ReasoningReplay` `Unsupported` for
/// `model`, or the contract is violated.
pub async fn assert_unreplayable_reasoning_dropped<F, Fut, G, GFut>(
    provider: &impl ChatProvider,
    model: &str,
    mut probe: F,
    mut stream_probe: G,
) where
    F: FnMut(ChatRequest) -> Fut,
    Fut: Future<Output = ProbeOutcome<ChatResponse>>,
    G: FnMut(ChatRequest) -> GFut,
    GFut: Future<Output = ProbeOutcome<ChatResponse>>,
{
    assert_eq!(
        provider.chat_capability(model, ChatCapability::ReasoningReplay),
        CapabilitySupport::Unsupported,
        "unreplayable reasoning contract: `{model}` must report ReasoningReplay Unsupported"
    );
    let mut problems = Vec::new();
    for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
        let outcome = probe(request(model, policy)).await;
        problems.extend(on_path(CHAT, check(policy, &outcome)));
        let outcome = stream_probe(request(model, policy)).await;
        problems.extend(on_path(CHAT_STREAM, check(policy, &outcome)));
    }
    assert!(
        problems.is_empty(),
        "unreplayable reasoning contract failed:\n  {}",
        problems.join("\n  ")
    );
}

fn request(model: &str, policy: UnsupportedPolicy) -> ChatRequest {
    let mut data = serde_json::Map::new();
    data.insert(
        "allama_probe".into(),
        Value::String(REASONING_PROBE_FOREIGN.into()),
    );
    let foreign =
        OtherContent::new("allama_probe_content", data).with_origin(REASONING_PROBE_FOREIGN);
    ChatRequest::new(model)
        .user("What is 2 + 2?")
        .message(Message::assistant_blocks(vec![
            ContentBlock::reasoning(REASONING_PROBE_TEXT),
            ContentBlock::Other(foreign),
            ContentBlock::text("4"),
        ]))
        .user("And 3 + 3?")
        .unsupported_policy(policy)
}

fn check(policy: UnsupportedPolicy, outcome: &ProbeOutcome<ChatResponse>) -> Option<String> {
    let label = match policy {
        UnsupportedPolicy::Error => "under UnsupportedPolicy::Error: ",
        _ => "",
    };
    let warnings = match &outcome.result {
        Ok(response) => &response.warnings,
        Err(_) => {
            return Some(format!(
                "{label}expected the request to be served with the reasoning dropped, got {}",
                describe(&outcome.result)
            ));
        }
    };
    for (what, value) in [
        ("reasoning", REASONING_PROBE_TEXT),
        ("foreign content", REASONING_PROBE_FOREIGN),
    ] {
        if outcome.contains_text(value) {
            return Some(format!(
                "{label}the {what} reached the wire: {:?}",
                outcome.bodies()
            ));
        }
    }
    for (what, block) in [("reasoning", 0), ("foreign content", 1)] {
        if !warnings.iter().any(|w| drops_block(w, block)) {
            return Some(format!(
                "{label}no Dropped warning names the {what} (message 1, block {block}); \
                 warnings: {warnings:?}"
            ));
        }
    }
    None
}

fn drops_block(warning: &Warning, block: usize) -> bool {
    warning.kind == WarningKind::Dropped
        && warning.subject
            == Subject::Content {
                message: 1,
                block: Some(block),
            }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use allama::Result;
    use allama::adapter::Degradations;
    use allama::mock::MockProvider;
    use serde_json::json;

    use super::*;

    /// How the fake adapter treats the replayed turn.
    #[derive(Clone, Copy)]
    enum Adapter {
        /// Correct: preflight, then encoder warnings into the same helper.
        Correct,
        /// Skips preflight and sends reasoning as text.
        SendsReasoning,
        /// Preflights, but collects the encoder's warnings apart.
        CollectsApart,
        /// Drops the reasoning without a warning.
        DropsSilently,
        /// Refuses under a strict policy.
        RefusesWhenStrict,
    }

    fn provider() -> MockProvider {
        MockProvider::empty().with_chat_capability(
            ChatCapability::ReasoningReplay,
            CapabilitySupport::Unsupported,
        )
    }

    /// Encode the assistant turns of `request`: text is sent, reasoning is
    /// sent as text, and foreign `Other` blocks are dropped.
    fn encode(request: &ChatRequest, d: &mut Degradations) -> Value {
        let mut sent = Vec::new();
        for (message, m) in request.messages.iter().enumerate() {
            let Message::Assistant(m) = m else {
                sent.push(json!(message.to_string()));
                continue;
            };
            for (block, content) in m.content.iter().enumerate() {
                match content {
                    ContentBlock::Text(t) => sent.push(json!(t.text)),
                    ContentBlock::Reasoning(r) => sent.push(json!(r.text)),
                    _ => d.dropped(
                        Subject::Content {
                            message,
                            block: Some(block),
                        },
                        "foreign content",
                    ),
                }
            }
        }
        json!({ "messages": sent })
    }

    fn send(adapter: Adapter, request: &ChatRequest) -> Result<(Value, Vec<Warning>)> {
        let policy = request.unsupported_policy.unwrap_or_default();
        if matches!(adapter, Adapter::RefusesWhenStrict) && policy == UnsupportedPolicy::Error {
            return Err(Degradations::new("fake", policy).reject(
                Subject::Content {
                    message: 1,
                    block: Some(0),
                },
                "no reasoning",
            ));
        }
        let mut d = Degradations::new("fake", policy);
        if matches!(adapter, Adapter::SendsReasoning) {
            let body = encode(request, &mut d);
            return Ok((body, d.into_warnings()));
        }
        let mut checked = Cow::Borrowed(request);
        if matches!(adapter, Adapter::DropsSilently) {
            for message in &mut checked.to_mut().messages {
                if let Message::Assistant(m) = message {
                    m.content
                        .retain(|b| !matches!(b, ContentBlock::Reasoning(_)));
                }
            }
        } else {
            allama::adapter::chat::preflight(&provider(), &mut checked, &mut d)?;
        }
        if matches!(adapter, Adapter::CollectsApart) {
            let mut apart = Degradations::new("fake", policy);
            let body = encode(&checked, &mut apart);
            let mut warnings = d.into_warnings();
            warnings.extend(apart.into_warnings());
            return Ok((body, warnings));
        }
        let body = encode(&checked, &mut d);
        Ok((body, d.into_warnings()))
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
        assert_unreplayable_reasoning_dropped(
            &provider(),
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
    #[should_panic(expected = "chat: the reasoning reached the wire")]
    async fn sending_reasoning_fails() {
        contract(Adapter::SendsReasoning).await;
    }

    #[tokio::test]
    #[should_panic(
        expected = "chat: no Dropped warning names the foreign content (message 1, block 1)"
    )]
    async fn collecting_encoder_warnings_apart_fails() {
        contract(Adapter::CollectsApart).await;
    }

    /// Without preflight's record, the foreign block's warning lands on
    /// block 0, so block 1 goes unnamed.
    #[tokio::test]
    #[should_panic(
        expected = "chat: no Dropped warning names the foreign content (message 1, block 1)"
    )]
    async fn dropping_silently_fails() {
        contract(Adapter::DropsSilently).await;
    }

    #[tokio::test]
    #[should_panic(
        expected = "chat_stream: under UnsupportedPolicy::Error: expected the request to be served"
    )]
    async fn refusing_under_a_strict_policy_fails() {
        contract(Adapter::RefusesWhenStrict).await;
    }

    #[tokio::test]
    #[should_panic(expected = "must report ReasoningReplay Unsupported")]
    async fn a_model_that_replays_reasoning_is_not_probed() {
        assert_unreplayable_reasoning_dropped(
            &MockProvider::empty(),
            "m",
            |request| run(Adapter::Correct, request),
            |request| run(Adapter::Correct, request),
        )
        .await;
    }
}
