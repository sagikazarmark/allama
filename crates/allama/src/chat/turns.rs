use crate::{Degradations, Error, Message, Result, Subject};

/// Skip the empty turns of an encoded conversation, and drop an assistant
/// turn that the skips and drops left last.
///
/// Adapters call this after encoding the messages of a request, before
/// assembling the wire body, with one [`EncodedTurn`] per turn they encoded
/// (`turn` reads it from each item of `turns`). It returns the turns to
/// send, in order:
///
/// - **Empty user and assistant turns are skipped without a warning**:
///   nothing of them was lost, and whatever the adapter dropped from them
///   was already reported. A tool turn is never empty, since an empty
///   result is a valid answer.
/// - **An assistant turn left last only because the turns after it were
///   skipped or dropped is dropped** with a
///   [`Dropped`](crate::WarningKind::Dropped) warning, repeatedly, so a
///   replayed reply is never sent as a prefill the caller did not ask for.
///   Like every `Dropped` warning, this does not depend on the request's
///   [`UnsupportedPolicy`](crate::UnsupportedPolicy). An assistant turn the
///   caller sent last on purpose, as a prefill, is kept.
///
/// Warnings name the caller's message indices
/// ([`EncodedTurn::message`]), recorded in `degradations` with every other
/// warning of the request.
///
/// ```
/// use allama::adapter::chat::{EncodedTurn, retain_turns};
/// use allama::adapter::Degradations;
/// use allama::{ChatRequest, ContentPart, Message, UnsupportedPolicy, WarningKind};
///
/// let request = ChatRequest::new("demo-model")
///     .user("What is 2 + 2?")
///     .assistant("4")
///     .message(Message::user_parts(Vec::<ContentPart>::new()));
/// let mut degradations = Degradations::new("demo", UnsupportedPolicy::Error);
///
/// // Encode each message; here, the wire form is the role and the text.
/// let encoded: Vec<(EncodedTurn, (&str, String))> = request
///     .messages
///     .iter()
///     .enumerate()
///     .map(|(index, message)| {
///         let text = message.text().unwrap_or_default();
///         let turn = EncodedTurn::new(index, message, text.is_empty());
///         (turn, (message.role(), text))
///     })
///     .collect();
///
/// let kept = retain_turns(encoded, &mut degradations, |(turn, _)| *turn)?;
///
/// // The empty user turn is skipped, and the reply it left last is dropped.
/// let wire: Vec<_> = kept.into_iter().map(|(_, wire)| wire).collect();
/// assert_eq!(wire, [("user", "What is 2 + 2?".to_owned())]);
/// assert_eq!(degradations.warnings()[0].kind, WarningKind::Dropped);
/// # Ok::<(), allama::Error>(())
/// ```
///
/// # Errors
///
/// Returns an [authored](Error::authored)
/// [`Error::InvalidRequest`](crate::Error::InvalidRequest) when `turns` is
/// not empty but no turn is left to send. An empty `turns` is returned as
/// is.
pub fn retain_turns<T>(
    turns: Vec<T>,
    degradations: &mut Degradations,
    turn: impl Fn(&T) -> EncodedTurn,
) -> Result<Vec<T>> {
    if turns.is_empty() {
        return Ok(turns);
    }
    let total = turns.len();
    let mut kept: Vec<(usize, T)> = turns
        .into_iter()
        .enumerate()
        .filter(|(_, item)| !turn(item).is_skipped())
        .collect();

    // Drop assistant turns that only the removal of later turns left last.
    while let Some((position, item)) = kept.last() {
        let encoded = turn(item);
        if encoded.role != Role::Assistant || *position + 1 == total {
            break;
        }
        degradations.dropped(
            Subject::Content {
                message: encoded.message,
                block: None,
            },
            "the turns after the assistant turn were empty or dropped, and sending it last \
             would ask the model to continue it; it was dropped",
        );
        kept.pop();
    }

    if kept.is_empty() {
        return Err(Error::invalid_request(
            "nothing to send: every message of the conversation is empty or was dropped",
        )
        .authored()
        .with_provider(degradations.provider().to_owned()));
    }
    Ok(kept.into_iter().map(|(_, item)| item).collect())
}

/// One turn of a conversation as an adapter encoded it, for
/// [`retain_turns`].
///
/// A turn is usually one message of the request. An adapter that splits a
/// message into several turns, or makes a turn from part of one (a tool
/// answer lifted out of a user message, for example), describes each turn
/// with the index of the message it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodedTurn {
    message: usize,
    role: Role,
    empty: bool,
}

impl EncodedTurn {
    /// The turn encoded from `message`, at index `index` of
    /// `ChatRequest::messages`.
    ///
    /// `encoded_empty` tells whether the encoding holds nothing, for
    /// example because the adapter dropped every block of the message. The
    /// turn is empty when either the encoding or the message
    /// ([`Message::is_empty`]) is; a tool turn is never empty.
    #[must_use]
    pub fn new(index: usize, message: &Message, encoded_empty: bool) -> Self {
        let empty = encoded_empty || message.is_empty();
        match message {
            Message::User(_) => Self::user(index, empty),
            Message::Assistant(_) => Self::assistant(index, empty),
            Message::Tool(_) => Self::tool(index),
        }
    }

    /// A user turn from message `message`, empty or not.
    #[must_use]
    pub fn user(message: usize, empty: bool) -> Self {
        Self {
            message,
            role: Role::User,
            empty,
        }
    }

    /// An assistant turn from message `message`, empty or not.
    #[must_use]
    pub fn assistant(message: usize, empty: bool) -> Self {
        Self {
            message,
            role: Role::Assistant,
            empty,
        }
    }

    /// A tool turn from message `message`. A tool turn is never empty: an
    /// empty result is a valid answer.
    #[must_use]
    pub fn tool(message: usize) -> Self {
        Self {
            message,
            role: Role::Tool,
            empty: false,
        }
    }

    /// The index in `ChatRequest::messages` of the message the turn was
    /// encoded from.
    #[must_use]
    pub fn message(&self) -> usize {
        self.message
    }

    /// Whether the turn holds nothing to send.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.empty
    }

    fn is_skipped(self) -> bool {
        self.empty && self.role != Role::Tool
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    User,
    Assistant,
    Tool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContentPart, UnsupportedPolicy, WarningKind};

    fn run(
        turns: &[EncodedTurn],
        policy: UnsupportedPolicy,
    ) -> (Result<Vec<usize>>, Vec<crate::Warning>) {
        let mut degradations = Degradations::new("demo", policy);
        let kept = retain_turns(turns.to_vec(), &mut degradations, |turn| *turn)
            .map(|kept| kept.iter().map(EncodedTurn::message).collect());
        (kept, degradations.into_warnings())
    }

    fn dropped(message: usize) -> Subject {
        Subject::Content {
            message,
            block: None,
        }
    }

    #[test]
    fn empty_user_and_assistant_turns_are_skipped_without_a_warning() {
        let turns = [
            EncodedTurn::user(0, false),
            EncodedTurn::assistant(1, true),
            EncodedTurn::assistant(2, false),
            EncodedTurn::user(3, true),
            EncodedTurn::user(4, false),
        ];
        let (kept, warnings) = run(&turns, UnsupportedPolicy::Warn);
        assert_eq!(kept.unwrap(), [0, 2, 4]);
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    #[test]
    fn a_tool_turn_is_never_skipped() {
        let turns = [
            EncodedTurn::user(0, false),
            EncodedTurn::assistant(1, false),
            EncodedTurn::tool(2),
        ];
        let (kept, warnings) = run(&turns, UnsupportedPolicy::Warn);
        assert_eq!(kept.unwrap(), [0, 1, 2]);
        assert!(warnings.is_empty());
        assert!(!EncodedTurn::tool(2).is_empty());
    }

    #[test]
    fn an_assistant_turn_left_last_is_dropped_under_every_policy() {
        for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
            let turns = [
                EncodedTurn::user(0, false),
                EncodedTurn::assistant(1, false),
                EncodedTurn::user(2, true),
            ];
            let (kept, warnings) = run(&turns, policy);
            assert_eq!(kept.unwrap(), [0], "{policy:?}");
            assert_eq!(warnings.len(), 1, "{warnings:?}");
            assert_eq!(warnings[0].kind, WarningKind::Dropped);
            assert_eq!(warnings[0].subject, dropped(1));
            assert_eq!(warnings[0].provider.as_deref(), Some("demo"));
        }
    }

    #[test]
    fn trailing_drops_repeat_until_a_turn_that_may_end_the_conversation() {
        let turns = [
            EncodedTurn::user(0, false),
            EncodedTurn::assistant(1, false),
            EncodedTurn::tool(2),
            EncodedTurn::assistant(3, false),
            EncodedTurn::assistant(4, false),
            EncodedTurn::assistant(5, true),
            EncodedTurn::user(6, true),
        ];
        let (kept, warnings) = run(&turns, UnsupportedPolicy::Warn);
        assert_eq!(kept.unwrap(), [0, 1, 2]);
        let subjects: Vec<_> = warnings.into_iter().map(|w| w.subject).collect();
        assert_eq!(subjects, [dropped(4), dropped(3)]);
    }

    #[test]
    fn a_prefill_the_caller_sent_last_is_kept() {
        let turns = [
            EncodedTurn::user(0, false),
            EncodedTurn::assistant(1, false),
        ];
        let (kept, warnings) = run(&turns, UnsupportedPolicy::Error);
        assert_eq!(kept.unwrap(), [0, 1]);
        assert!(warnings.is_empty());
    }

    #[test]
    fn several_turns_may_come_from_one_message() {
        let turns = [
            EncodedTurn::user(0, false),
            EncodedTurn::assistant(1, false),
            EncodedTurn::tool(2),
            EncodedTurn::user(2, true),
        ];
        let (kept, warnings) = run(&turns, UnsupportedPolicy::Warn);
        assert_eq!(kept.unwrap(), [0, 1, 2]);
        assert!(warnings.is_empty());
    }

    #[test]
    fn nothing_left_to_send_is_an_authored_invalid_request() {
        let turns = [EncodedTurn::user(0, true), EncodedTurn::assistant(1, true)];
        let (kept, warnings) = run(&turns, UnsupportedPolicy::Warn);
        let err = kept.unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err}");
        assert_eq!(err.message_origin(), crate::MessageOrigin::Authored);
        assert_eq!(
            err.context().and_then(|c| c.provider.as_deref()),
            Some("demo")
        );
        assert!(warnings.is_empty());

        // The reply dropped on the way out is reported too.
        let turns = [EncodedTurn::assistant(0, false), EncodedTurn::user(1, true)];
        let (kept, warnings) = run(&turns, UnsupportedPolicy::Warn);
        assert!(matches!(kept, Err(Error::InvalidRequest { .. })));
        assert_eq!(warnings.len(), 1);
    }

    #[test]
    fn a_request_without_turns_is_left_to_the_provider() {
        let (kept, warnings) = run(&[], UnsupportedPolicy::Warn);
        assert!(kept.unwrap().is_empty());
        assert!(warnings.is_empty());
    }

    #[test]
    fn turns_from_messages_combine_the_encoding_and_the_message() {
        let empty_user = Message::user("");
        let empty_parts = Message::user_parts(vec![]);
        let reply = Message::assistant("4");
        let empty_reply = Message::assistant_blocks(vec![]);
        let empty_result = Message::tool_result("c1", "f", "");
        assert!(EncodedTurn::new(0, &empty_user, false).is_empty());
        assert!(EncodedTurn::new(0, &empty_parts, false).is_empty());
        assert!(EncodedTurn::new(0, &empty_reply, false).is_empty());
        assert!(!EncodedTurn::new(0, &reply, false).is_empty());
        assert!(EncodedTurn::new(0, &reply, true).is_empty());
        assert!(!EncodedTurn::new(0, &empty_result, true).is_empty());
        let foreign = Message::user_parts(vec![ContentPart::Other(
            crate::OtherContent::new("x", Default::default()).with_origin("other"),
        )]);
        assert!(!EncodedTurn::new(0, &foreign, false).is_empty());
        assert!(EncodedTurn::new(0, &foreign, true).is_empty());
    }
}
