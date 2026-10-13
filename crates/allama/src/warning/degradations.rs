//! [`Degradations`], the helper adapters apply the unsupported-feature
//! policy with. It is public at one path,
//! [`adapter::Degradations`](crate::adapter::Degradations).

use std::borrow::Cow;
use std::collections::BTreeMap;

use super::{ReplayState, RequestField, Subject, UnsupportedPolicy, Warning, WarningKind};
use crate::capability::sealed::AnyCapability;
use crate::{Capability, CapabilitySupport, Error, Result};

/// Shared helper adapters use to apply [`UnsupportedPolicy`] and collect
/// warnings while mapping a request.
///
/// The errors it returns are [authored](crate::Error::authored), so their
/// message appears in the redacted [`ErrorLog`](crate::ErrorLog). Pass only
/// text your adapter wrote, never a provider's error text.
///
/// Every refusal it returns has the stage
/// [`BeforeSending`](crate::UnsupportedStage::BeforeSending) and names its
/// subject ([`Error::unsupported_subject`](crate::Error::unsupported_subject),
/// translated to the caller's indices like a warning). Apply it while
/// mapping the request, before sending: callers read the stage as "nothing
/// was sent", and extraction's `Auto` mode retries a refused response format
/// in forced-tool mode. Its message reads `cannot send <subject>: <reason>`
/// for every kind of refusal.
///
/// # Methods
///
/// | Method | [`Warn`](UnsupportedPolicy::Warn) (default) | [`Error`](UnsupportedPolicy::Error) |
/// |---|---|---|
/// | [`dropped`](Self::dropped), [`dropped_replay_state`](Self::dropped_replay_state), [`warn`](Self::warn), [`approximated`](Self::approximated), [`coded_approximated`](Self::coded_approximated) | records the warning | records the warning: it only records, under every policy |
/// | [`unsupported_hint`](Self::unsupported_hint), [`hint`](Self::hint) | records an `Unsupported` warning | returns the refusal |
/// | [`coded_lossy_hint`](Self::coded_lossy_hint) | records an `Approximated` warning | returns the refusal |
/// | [`unsupported`](Self::unsupported), [`reject`](Self::reject) | returns the refusal as an [`Error`] for you to return; records nothing | the same: it always refuses |
///
/// # Messages
///
/// A message reads `<limitation>; <outcome>[; <remedy>]`, with `"; "`
/// between the parts and nowhere else: what the provider lacks, what
/// happened to the request, and optionally what the caller can do
/// (``"Gemini has no seed; `seed` was not sent"``). The outcome's verb
/// follows the [kind](WarningKind): "`x` was not sent" for `Unsupported`,
/// "sent as X" or "X was used instead" for `Approximated`, "it was
/// dropped" for `Dropped`. Identifiers go in backticks. A [`Warning`]'s
/// `Display` already names the provider and, for content, the location,
/// so the message names neither; it names a protocol, endpoint or
/// provider only where that is the reason (ADR 0003, "how a warning
/// reads").
///
/// A hint's message is its warning text. Under [`UnsupportedPolicy::Error`]
/// nothing happened to the hint, so the refusal leaves the outcome out and
/// keeps the limitation and the remedy: ``cannot send `seed`: Gemini has
/// no seed``. A message that is only ever a refusal
/// ([`unsupported`](Self::unsupported), [`reject`](Self::reject)) has no
/// outcome: `<limitation>[; <remedy>]`.
///
/// # Codes
///
/// Every warning recorded here that is about a request field
/// ([`Subject::Field`]) or a system prompt ([`Subject::System`]) and has no
/// [code](Warning::code) gets `<kind>.<field>`: `unsupported.seed`,
/// `approximated.reasoning`, `dropped.system`. Its warning event then goes
/// out at `WARN` once per provider, model and code, not on every call. Give
/// a warning about content a code yourself ([`Warning::with_code`], recorded
/// with [`warn`](Self::warn)) when it recurs with the request's shape, such
/// as replay state another provider wrote, or a limitation the
/// conversation's shape repeats on every turn
/// ([`coded_lossy_hint`](Self::coded_lossy_hint), such as tool call
/// arguments sent as `{}`, or [`coded_approximated`](Self::coded_approximated),
/// such as a tool error sent to an endpoint with no error flag); leave other
/// content the application sent uncoded.
///
/// # Indices
///
/// [`chat::preflight`](crate::adapter::chat::preflight) may remove assistant
/// reasoning blocks, which shifts the block indices of the messages that
/// held them. It records which blocks it kept here, and every
/// [`Subject::Content`] warning recorded afterwards has its block index
/// translated back to the caller's request. Adapters therefore report
/// indices relative to the request they received, the preflighted one, and
/// never translate them themselves. Warnings recorded before preflight, and
/// preflight's own, are left as they are.
///
/// Record an encoder's warnings into the same helper ([`warn`](Self::warn)
/// or [`Extend`]), not next to it: warnings collected apart from it keep
/// the preflighted indices.
///
/// ```
/// use allama::adapter::Degradations;
/// use allama::{RequestField, UnsupportedPolicy};
///
/// let mut degradations = Degradations::new("demo", UnsupportedPolicy::Warn);
/// let seed = "demo has no seed; `seed` was not sent";
/// degradations.unsupported_hint(RequestField::Seed, seed).unwrap();
/// assert_eq!(degradations.warnings().len(), 1);
///
/// let mut strict = Degradations::new("demo", UnsupportedPolicy::Error);
/// assert!(strict.unsupported_hint(RequestField::Seed, seed).is_err());
/// ```
#[derive(Debug, Clone)]
pub struct Degradations {
    provider: Cow<'static, str>,
    policy: UnsupportedPolicy,
    warnings: Vec<Warning>,
    /// The blocks preflight kept, for translating later warnings back to
    /// the caller's indices.
    blocks: BlockMap,
    /// The capabilities the request relies on that the provider answered
    /// `Unknown`, each with the first subject that relies on it.
    unknown: Vec<(AnyCapability, Subject)>,
}

impl Degradations {
    /// Create a helper for `provider` with the request's policy.
    #[must_use]
    pub fn new(provider: impl Into<Cow<'static, str>>, policy: UnsupportedPolicy) -> Self {
        Self {
            provider: provider.into(),
            policy,
            warnings: Vec::new(),
            blocks: BlockMap::default(),
            unknown: Vec::new(),
        }
    }

    /// The policy in effect.
    #[must_use]
    pub fn policy(&self) -> UnsupportedPolicy {
        self.policy
    }

    /// The provider warnings and errors are attributed to.
    pub(crate) fn provider(&self) -> &str {
        &self.provider
    }

    /// An unsupported *hint*: warn (and drop) or fail, per policy.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] under [`UnsupportedPolicy::Error`].
    pub fn unsupported_hint(
        &mut self,
        field: RequestField,
        message: impl Into<String>,
    ) -> Result<()> {
        self.hint(Subject::Field { field }, message)
    }

    /// An unsupported hint about an arbitrary subject.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] under [`UnsupportedPolicy::Error`].
    pub fn hint(&mut self, subject: Subject, message: impl Into<String>) -> Result<()> {
        let message = message.into();
        match self.policy {
            UnsupportedPolicy::Warn => {
                self.push(WarningKind::Unsupported, subject, message);
                Ok(())
            }
            UnsupportedPolicy::Error => Err(self.error(&without_outcome(&message), Some(subject))),
        }
    }

    /// [`hint`](Self::hint), with `code` for the warning: for a hint about
    /// content that recurs with the request's shape.
    pub(crate) fn coded_hint(
        &mut self,
        subject: Subject,
        message: impl Into<String>,
        code: &'static str,
    ) -> Result<()> {
        let message = message.into();
        match self.policy {
            UnsupportedPolicy::Warn => {
                self.warn(Warning::new(WarningKind::Unsupported, subject, message).with_code(code));
                Ok(())
            }
            UnsupportedPolicy::Error => Err(self.error(&without_outcome(&message), Some(subject))),
        }
    }

    /// An unsupported *semantic* field: always an error, which names
    /// `field` ([`Error::unsupported_field`]).
    #[must_use]
    pub fn unsupported(&self, field: RequestField, message: impl Into<String>) -> Error {
        self.error(&message.into(), Some(Subject::Field { field }))
    }

    /// An unsupported semantic feature that is not a request field (a
    /// content block or part, a tool result shape, an option): always an
    /// error, which names `subject` when there is one
    /// ([`Error::unsupported_subject`]). A [`Subject::Content`] is
    /// translated to the caller's indices.
    #[must_use]
    pub fn reject(&self, subject: impl Into<Option<Subject>>, message: impl Into<String>) -> Error {
        self.error(&message.into(), subject.into())
    }

    /// A lossy approximation that changes what the model sees (for example
    /// replaying malformed tool arguments as `{}`), with a stable `code`: an
    /// `Approximated` warning coded `code` by default, an error under
    /// [`UnsupportedPolicy::Error`].
    ///
    /// Such an approximation recurs with the request's shape (history
    /// replays the same arguments, and the same system prompt, on every
    /// turn), so its warning always has a code. For a request field or a
    /// system prompt, pass the code [Codes](Self#codes) would give it
    /// (`approximated.system`); for content, name one
    /// (`approximated.tool_call_arguments`).
    ///
    /// The message reads `<limitation>; <outcome>[; <remedy>]`, as for
    /// any hint (see [Messages](Self#messages)). Under
    /// [`UnsupportedPolicy::Error`] nothing was sent, so the refusal leaves
    /// the outcome out and keeps the limitation and the remedy. Like every
    /// warning and refusal here, a [`Subject::Content`] is translated to
    /// the caller's indices.
    ///
    /// Use it for a limitation that history repeats on every turn and that
    /// changes what the model is asked to act on, such as tool call
    /// arguments that are not a JSON object, replayed as `{}`: the code
    /// keeps the warning event to one `WARN` per provider and model, and a
    /// strict policy still refuses it in any message. For one that changes
    /// only how the model is told something it still sees, such as a tool
    /// error sent as a plain result, use
    /// [`coded_approximated`](Self::coded_approximated), which never refuses.
    ///
    /// One warning per code per request, as for
    /// [`dropped_replay_state`](Self::dropped_replay_state): a later hint
    /// with the same code joins the first warning, whose subject becomes
    /// [`Subject::Messages`] when it affects other messages, listing every
    /// one, and whose message stays the first hint's. So the warnings of a
    /// request do not grow with the conversation.
    ///
    /// ```
    /// use allama::adapter::Degradations;
    /// use allama::{Subject, UnsupportedPolicy, WarningKind};
    ///
    /// let message = "the arguments of tool call `f` are not a JSON object; they were sent as `{}`";
    /// let call = Subject::Content { message: 1, block: Some(0) };
    ///
    /// let mut d = Degradations::new("demo", UnsupportedPolicy::Warn);
    /// d.coded_lossy_hint(call.clone(), message, "approximated.tool_call_arguments")
    ///     .unwrap();
    /// assert_eq!(d.warnings()[0].kind, WarningKind::Approximated);
    /// assert_eq!(d.warnings()[0].code.as_deref(), Some("approximated.tool_call_arguments"));
    ///
    /// let mut strict = Degradations::new("demo", UnsupportedPolicy::Error);
    /// let error = strict
    ///     .coded_lossy_hint(call, message, "approximated.tool_call_arguments")
    ///     .unwrap_err();
    /// assert!(!error.to_string().contains("were sent"));
    /// assert!(error.to_string().contains("not a JSON object"));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] under [`UnsupportedPolicy::Error`].
    pub fn coded_lossy_hint(
        &mut self,
        subject: impl Into<Subject>,
        message: impl Into<String>,
        code: &'static str,
    ) -> Result<()> {
        let subject = subject.into();
        let message = message.into();
        match self.policy {
            UnsupportedPolicy::Warn => {
                if !self.join_coded(code, &subject) {
                    self.warn(
                        Warning::new(WarningKind::Approximated, subject, message).with_code(code),
                    );
                }
                Ok(())
            }
            UnsupportedPolicy::Error => Err(self.error(&without_outcome(&message), Some(subject))),
        }
    }

    /// Join `subject` to the warning coded `code` when there is one and
    /// both name messages: its subject becomes [`Subject::Messages`],
    /// listing every message. Returns whether it joined.
    fn join_coded(&mut self, code: &str, subject: &Subject) -> bool {
        // Message indices are the caller's already: preflight removes
        // blocks, never messages.
        let Some(existing) = self
            .warnings
            .iter_mut()
            .find(|warning| warning.code.as_deref() == Some(code))
        else {
            return false;
        };
        let (Some(mut messages), Some(more)) = (
            subject_messages(&existing.subject),
            subject_messages(subject),
        ) else {
            return false;
        };
        for index in more {
            if let Err(at) = messages.binary_search(&index) {
                messages.insert(at, index);
            }
        }
        if messages.len() > 1 {
            existing.subject = Subject::Messages { messages };
        }
        true
    }

    /// Record that a feature was mapped to its closest equivalent.
    pub fn approximated(&mut self, subject: Subject, message: impl Into<String>) {
        self.push(WarningKind::Approximated, subject, message.into());
    }

    /// [`approximated`](Self::approximated), with a stable `code`, for an
    /// approximation that history repeats on every turn: an `Approximated`
    /// warning coded `code`, recorded under every policy.
    ///
    /// Use it where the model still sees what the caller sent, in the form
    /// closest to it, so a strict policy has nothing to refuse: a tool
    /// result marked as an error, sent to an endpoint with no error flag,
    /// reaches the model as its error text, and only the flag is lost. Where
    /// the approximation changes what the model is asked to act on, use
    /// [`coded_lossy_hint`](Self::coded_lossy_hint), which a strict policy
    /// refuses.
    ///
    /// The code keeps the warning event to one `WARN` per provider and
    /// model, and, as for [`coded_lossy_hint`](Self::coded_lossy_hint), one
    /// warning per code per request: a later approximation with the same
    /// code joins the first warning, whose subject becomes
    /// [`Subject::Messages`] when it affects other messages, listing every
    /// one, and whose message stays the first one's. So the warnings of a
    /// request do not grow with the conversation. The message reads
    /// `<limitation>; <outcome>[; <remedy>]` (see [Messages](Self#messages)),
    /// and a [`Subject::Content`] is translated to the caller's indices.
    ///
    /// ```
    /// use allama::adapter::Degradations;
    /// use allama::{Subject, UnsupportedPolicy, WarningKind};
    ///
    /// let message = "demo has no tool error flag; the error text was sent as the tool result; \
    ///                the model still reads the error text; \
    ///                to send your own encoding to this endpoint, use `push_tool_result`";
    /// let code = "approximated.tool_result_error";
    ///
    /// // Never refused: the strict policy records it too.
    /// let mut d = Degradations::new("demo", UnsupportedPolicy::Error);
    /// d.coded_approximated(Subject::Content { message: 2, block: None }, message, code);
    /// d.coded_approximated(Subject::Content { message: 6, block: None }, message, code);
    ///
    /// let [warning] = d.warnings() else { panic!("one warning per code") };
    /// assert_eq!(warning.kind, WarningKind::Approximated);
    /// assert_eq!(warning.code.as_deref(), Some(code));
    /// assert_eq!(warning.subject, Subject::Messages { messages: vec![2, 6] });
    /// ```
    pub fn coded_approximated(
        &mut self,
        subject: impl Into<Subject>,
        message: impl Into<String>,
        code: &'static str,
    ) {
        let subject = subject.into();
        if !self.join_coded(code, &subject) {
            self.warn(Warning::new(WarningKind::Approximated, subject, message).with_code(code));
        }
    }

    /// Record that content or state was dropped.
    pub fn dropped(&mut self, subject: Subject, message: impl Into<String>) {
        self.push(WarningKind::Dropped, subject, message.into());
    }

    /// Record that replay state was dropped: a [`Dropped`](WarningKind::Dropped)
    /// warning coded [`state.code()`](ReplayState::code), so it is logged at
    /// `WARN` once per provider and model however many turns hold the state.
    ///
    /// One warning per code per request: a later drop with the same code
    /// joins the first warning, whose subject becomes
    /// [`Subject::Messages`], listing every message affected, and whose
    /// message stays the first drop's. So the warnings of a request do not
    /// grow with the conversation.
    ///
    /// For state another provider wrote, the message also points to
    /// [`retain_provider_state_for`](crate::history::retain_provider_state_for),
    /// which removes it once from a conversation that moved for good. Use
    /// [`dropped`](Self::dropped) for content the application sent, which
    /// stays uncoded.
    ///
    /// ```
    /// use allama::adapter::Degradations;
    /// use allama::{ReplayState, Subject, UnsupportedPolicy};
    ///
    /// let mut d = Degradations::new("demo", UnsupportedPolicy::Warn);
    /// let block = Subject::Content { message: 1, block: Some(0) };
    /// d.dropped_replay_state(
    ///     ReplayState::ForeignReasoning,
    ///     block,
    ///     "reasoning from `openai` can be replayed only to its origin; it was dropped",
    /// );
    /// let warning = &d.warnings()[0];
    /// assert_eq!(warning.code.as_deref(), Some("dropped.foreign_reasoning"));
    /// assert!(warning.message.contains("retain_provider_state_for"));
    /// assert_eq!(warning.replay_state(), Some(ReplayState::ForeignReasoning));
    /// ```
    pub fn dropped_replay_state(
        &mut self,
        state: ReplayState,
        subject: Subject,
        message: impl Into<String>,
    ) {
        if self.join_coded(state.code(), &subject) {
            return;
        }
        let mut message = message.into();
        if state.is_foreign() {
            message.push_str(
                "; once a conversation moves to this provider for good, \
                 `retain_provider_state_for` removes another provider's state",
            );
        }
        self.warn(Warning::new(WarningKind::Dropped, subject, message).with_code(state.code()));
    }

    /// Record an arbitrary warning (the provider is filled in when missing).
    ///
    /// Like every warning recorded here, a [`Subject::Content`] warning is
    /// translated to the caller's indices (see [Indices](Self#indices)).
    pub fn warn(&mut self, mut warning: Warning) {
        if warning.provider.is_none() {
            warning.provider = Some(self.provider.clone());
        }
        let warning = self.translate(warning).with_default_code();
        self.warnings.push(warning);
    }

    /// Translate a warning about the preflighted request to the caller's
    /// indices, without recording it: for the warnings of a provider the
    /// preflighted request was delegated to, which
    /// [`Prepared::finish`](crate::adapter::chat::Prepared::finish) passes on.
    /// A warning about anything other than a block of a message preflight
    /// changed is returned as it is.
    #[must_use]
    pub(crate) fn translate(&self, mut warning: Warning) -> Warning {
        warning.subject = self.translate_subject(warning.subject);
        warning
    }

    fn translate_subject(&self, mut subject: Subject) -> Subject {
        if let Subject::Content {
            message,
            block: Some(block),
        } = &mut subject
        {
            *block = self.blocks.original(*message, *block);
        }
        subject
    }

    /// Record that the request relies on `capability`, which the provider
    /// answered [`Unknown`](CapabilitySupport::Unknown), first through
    /// `subject`.
    ///
    /// A [`Validation`](crate::Validation) reports these as
    /// [`relied_on_unknown`](crate::Validation::relied_on_unknown): what a
    /// dry run could not know. Each capability is recorded once, with the
    /// first subject; later calls for it change nothing. Like a warning, a
    /// [`Subject::Content`] is translated to the caller's indices.
    /// [`chat::preflight`](crate::adapter::chat::preflight) and
    /// [`embedding::preflight`](crate::adapter::embedding::preflight) record every
    /// capability they query.
    pub(crate) fn record_unknown<C: Capability>(&mut self, capability: C, subject: Subject) {
        let capability = capability.into_any();
        if self.unknown.iter().any(|(known, _)| *known == capability) {
            return;
        }
        let subject = self.translate_subject(subject);
        self.unknown.push((capability, subject));
    }

    /// Whether `support` is `Unsupported`, recording `capability` when it
    /// is `Unknown` (see [`record_unknown`](Self::record_unknown)).
    pub(crate) fn refuses<C: Capability>(
        &mut self,
        capability: C,
        support: CapabilitySupport,
        subject: impl FnOnce() -> Subject,
    ) -> bool {
        if support == CapabilitySupport::Unknown {
            self.record_unknown(capability, subject());
        }
        support == CapabilitySupport::Unsupported
    }

    /// The recorded capabilities of one operation.
    pub(crate) fn unknown<C: Capability>(&self) -> Vec<(C, Subject)> {
        self.unknown
            .iter()
            .filter_map(|(capability, subject)| Some((C::from_any(*capability)?, subject.clone())))
            .collect()
    }

    /// How many capabilities are recorded, to roll back to with
    /// [`truncate_unknown`](Self::truncate_unknown).
    pub(crate) fn unknown_len(&self) -> usize {
        self.unknown.len()
    }

    /// Forget the capabilities recorded after the first `len`.
    pub(crate) fn truncate_unknown(&mut self, len: usize) {
        self.unknown.truncate(len);
    }

    /// Record that preflight kept, of the blocks of `message`, those at
    /// `kept` (indices into the message as preflight received it, of
    /// `len` blocks). Later warnings about the message are translated back.
    pub(crate) fn record_kept_blocks(&mut self, message: usize, len: usize, kept: Vec<usize>) {
        self.blocks.record(message, len, kept);
    }

    /// Borrow the collected warnings.
    #[must_use]
    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }

    /// Take the collected warnings.
    #[must_use]
    pub fn into_warnings(self) -> Vec<Warning> {
        self.warnings
    }

    fn push(&mut self, kind: WarningKind, subject: Subject, message: String) {
        let warning = Warning::new(kind, subject, message).with_provider(self.provider.clone());
        self.warn(warning);
    }

    /// A refusal before sending: `cannot send <subject>: <reason>`.
    fn error(&self, reason: &str, subject: Option<Subject>) -> Error {
        let subject = subject.map(|subject| self.translate_subject(subject));
        let message = match &subject {
            Some(subject) => format!("cannot send {subject}: {reason}"),
            None => format!("cannot send the request: {reason}"),
        };
        Error::unsupported_before_sending(message, subject)
            .authored()
            .with_provider(self.provider.to_string())
    }
}

/// A hint's message without what happened to the hint, for its refusal:
/// the limitation, and the remedy when one follows the outcome
/// (`<limitation>; <outcome>; <remedy>`, see
/// [Messages](Degradations#messages)).
/// The messages a replay-drop subject refers to.
fn subject_messages(subject: &Subject) -> Option<Vec<usize>> {
    match subject {
        Subject::Content { message, .. } => Some(vec![*message]),
        Subject::Messages { messages } => Some(messages.clone()),
        _ => None,
    }
}

fn without_outcome(message: &str) -> Cow<'_, str> {
    let mut parts = message.splitn(3, "; ");
    let limitation = parts.next().unwrap_or(message);
    match (parts.next(), parts.next()) {
        (Some(_), Some(remedy)) => Cow::Owned(format!("{limitation}; {remedy}")),
        _ => Cow::Borrowed(limitation),
    }
}

impl Extend<Warning> for Degradations {
    /// Record warnings collected elsewhere, such as an encoder's, as
    /// [`warn`](Self::warn) does.
    fn extend<I: IntoIterator<Item = Warning>>(&mut self, warnings: I) {
        for warning in warnings {
            self.warn(warning);
        }
    }
}

/// Which blocks of each message preflight kept, by their index in the
/// caller's request.
#[derive(Debug, Clone, Default)]
struct BlockMap {
    /// Message index to the blocks kept in it.
    messages: BTreeMap<usize, KeptBlocks>,
}

#[derive(Debug, Clone)]
struct KeptBlocks {
    /// How many blocks the caller's message held.
    original_len: usize,
    /// The caller's index of each block that is left, in order.
    kept: Vec<usize>,
}

impl BlockMap {
    /// The caller's index of block `block` of message `message`.
    fn original(&self, message: usize, block: usize) -> usize {
        match self.messages.get(&message) {
            Some(blocks) => blocks.original(block),
            None => block,
        }
    }

    /// Record a removal from `message`, of `len` blocks, that kept `kept`.
    /// The indices are relative to the message as it was before the
    /// removal, so a second removal composes with the first.
    fn record(&mut self, message: usize, len: usize, kept: Vec<usize>) {
        let blocks = match self.messages.get(&message) {
            Some(earlier) => KeptBlocks {
                original_len: earlier.original_len,
                kept: kept.into_iter().map(|i| earlier.original(i)).collect(),
            },
            None => KeptBlocks {
                original_len: len,
                kept,
            },
        };
        self.messages.insert(message, blocks);
    }
}

impl KeptBlocks {
    fn original(&self, block: usize) -> usize {
        // An index past the kept blocks (an adapter naming a position after
        // the last block) keeps its distance from the end.
        self.kept
            .get(block)
            .copied()
            .unwrap_or_else(|| self.original_len + (block - self.kept.len()))
    }
}
