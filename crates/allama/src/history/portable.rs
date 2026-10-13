//! Keeping only portable content: the sanitizer for messages that come
//! from outside the application.

use crate::{
    ContentBlock, ContentPart, Message, ProviderData, Subject, ToolResultContent, UserContent,
    Warning, WarningKind,
};

/// Keep only portable content in `messages`: the sanitizer for a
/// conversation an end user sent (see [The trust boundary](super#the-trust-boundary)).
///
/// Provider data, provider files and content the portable model does not
/// represent are escape hatches into a provider's own wire format and
/// account. From a trusted store they carry the state a conversation needs
/// (signatures, item ids); from an end user they are injection: a raw
/// `Other` part that an adapter sends verbatim (an OpenAI `item_reference`,
/// an Anthropic `document` by `file_id`), a file id that reaches the
/// operator's provider account, or replay state. This removes, from every
/// message:
///
/// - all provider data, on messages, blocks, parts and citations, under
///   every namespace (`allama`'s too);
/// - [`Other`](crate::OtherContent) parts and blocks, whatever their origin;
/// - provider-executed tool calls and results;
/// - images and files whose source is a
///   [provider file](crate::MediaSource::ProviderFile).
///
/// What stays is portable: text, refusals, reasoning text (without its
/// state, so no provider replays it as its own), tool calls and results,
/// and inline or URL media. Messages are never removed, so warning indices
/// stay the caller's, and [application data](super#application-data) is
/// never touched: adapters never send it, and the application decides what
/// to trust in it.
///
/// This removes what an end user could inject into the provider's wire
/// format; it does not make the conversation authentic. A client that holds
/// the history can still write the assistant's turns and the tool results:
/// store conversations server-side when that matters.
///
/// Returns one `Dropped` warning per removed block or stripped state.
///
/// ```
/// use allama::history::retain_portable_content;
/// use allama::{ContentPart, File, Message, OtherContent, ProviderData};
/// use serde_json::json;
///
/// // A conversation as a browser sent it.
/// let mut messages = vec![Message::user_parts(vec![
///     ContentPart::text("Summarize these."),
///     ContentPart::Other(OtherContent::new("item_reference", json!({"id": "msg_secret"}).as_object().unwrap().clone())),
///     ContentPart::File(File::provider_file("application/pdf", "anthropic", "file_other_tenant")),
/// ])];
/// messages[0]
///     .provider_data_mut()
///     .insert_raw("openai", "item_id", json!("msg_forged"));
///
/// let warnings = retain_portable_content(&mut messages);
/// assert_eq!(warnings.len(), 3);
/// assert_eq!(messages[0], Message::user_parts(vec![ContentPart::text("Summarize these.")]));
/// ```
pub fn retain_portable_content(messages: &mut [Message]) -> Vec<Warning> {
    let mut warnings = Vec::new();
    for (index, message) in messages.iter_mut().enumerate() {
        let mut warn = |block: Option<usize>, text: String| {
            warnings.push(Warning::new(
                WarningKind::Dropped,
                Subject::Content {
                    message: index,
                    block,
                },
                text,
            ));
        };
        if clear(message.provider_data_mut()) {
            warn(
                None,
                "the message's provider data is not portable; it was dropped".into(),
            );
        }
        match message {
            Message::User(m) => {
                if let UserContent::Parts(parts) = &mut m.content {
                    retain_parts(parts, &mut warn);
                }
            }
            Message::Tool(m) => {
                if let ToolResultContent::Parts(parts) = &mut m.content {
                    retain_parts(parts, &mut warn);
                }
            }
            Message::Assistant(m) => {
                let mut position = 0;
                m.content.retain_mut(|block| {
                    let at = Some(position);
                    position += 1;
                    let state = match block {
                        ContentBlock::Other(_)
                        | ContentBlock::ProviderToolCall(_)
                        | ContentBlock::ProviderToolResult(_) => {
                            warn(at, not_portable(block.type_name()));
                            return false;
                        }
                        ContentBlock::Image(image) if image.source.provider_file_id().is_some() => {
                            warn(at, provider_file());
                            return false;
                        }
                        ContentBlock::Text(text) => {
                            let citations = text
                                .citations
                                .iter_mut()
                                .fold(false, |any, c| clear(&mut c.provider_data) | any);
                            clear(&mut text.provider_data) | citations
                        }
                        ContentBlock::Image(image) => clear(&mut image.provider_data),
                        ContentBlock::Reasoning(reasoning) => clear(&mut reasoning.provider_data),
                        ContentBlock::Refusal(refusal) => clear(&mut refusal.provider_data),
                        ContentBlock::ToolCall(call) => clear(&mut call.provider_data),
                    };
                    if state {
                        warn(at, state_dropped());
                    }
                    true
                });
            }
        }
    }
    warnings
}

fn retain_parts(parts: &mut Vec<ContentPart>, warn: &mut impl FnMut(Option<usize>, String)) {
    let mut position = 0;
    parts.retain_mut(|part| {
        let at = Some(position);
        position += 1;
        let state = match part {
            ContentPart::Other(other) => {
                warn(at, not_portable(&other.type_name));
                return false;
            }
            ContentPart::Image(image) if image.source.provider_file_id().is_some() => {
                warn(at, provider_file());
                return false;
            }
            ContentPart::File(file) if file.source.provider_file_id().is_some() => {
                warn(at, provider_file());
                return false;
            }
            ContentPart::Text(text) => {
                let citations = text
                    .citations
                    .iter_mut()
                    .fold(false, |any, c| clear(&mut c.provider_data) | any);
                clear(&mut text.provider_data) | citations
            }
            ContentPart::Image(image) => clear(&mut image.provider_data),
            ContentPart::File(file) => clear(&mut file.provider_data),
        };
        if state {
            warn(at, state_dropped());
        }
        true
    });
}

/// Empty `data`, returning whether it held anything.
fn clear(data: &mut ProviderData) -> bool {
    let had = !data.is_empty();
    *data = ProviderData::new();
    had
}

fn not_portable(type_name: &str) -> String {
    format!("`{type_name}` content is not portable; it was dropped")
}

fn provider_file() -> String {
    "a file a provider stores is not portable; it was dropped".to_owned()
}

fn state_dropped() -> String {
    "provider data is not portable; it was dropped".to_owned()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{Image, OtherContent, ProviderToolCall, Reasoning, ToolCall};

    /// Every escape hatch goes, portable content stays.
    #[test]
    fn assistant_turns_keep_only_portable_blocks() {
        let state = || ProviderData::new().with_raw("anthropic", "signature", json!("sig"));
        let mut messages = vec![
            Message::user("q"),
            Message::assistant_blocks(vec![
                ContentBlock::Reasoning(Reasoning::new("think").with_provider_data(state())),
                ContentBlock::Other(OtherContent::new("compaction", Default::default())),
                ContentBlock::ProviderToolCall(
                    ProviderToolCall::new("srv_1", "web_search", "{}").with_origin("anthropic"),
                ),
                ContentBlock::Image(Image::provider_file("openai", "file-1")),
                ContentBlock::ToolCall(ToolCall::new("c1", "f", "{}").with_provider_data(state())),
                ContentBlock::text("answer"),
            ]),
            Message::tool_result("c1", "f", "ok"),
        ];
        let warnings = retain_portable_content(&mut messages);
        assert_eq!(
            messages[1],
            Message::assistant_blocks(vec![
                ContentBlock::reasoning("think"),
                ContentBlock::ToolCall(ToolCall::new("c1", "f", "{}")),
                ContentBlock::text("answer"),
            ])
        );
        // Reasoning state, Other, provider tool, provider file, call state.
        assert_eq!(warnings.len(), 5);
        assert!(warnings.iter().all(|w| w.kind == WarningKind::Dropped));
    }
}
