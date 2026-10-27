//! Non-fatal warnings and the unsupported-feature policy.
//!
//! Adapters never drop a portable request feature silently. When a provider
//! cannot honor something, the adapter either fails with
//! [`Error::Unsupported`] or proceeds and reports a
//! [`Warning`]:
//!
//! - **Semantic fields** whose loss changes the output (`response_format`,
//!   tools, `tool_choice`, embedding `dimensions`, embedding input parts)
//!   always fail. Content is never dropped or joined.
//! - **Hints** (seed, penalties, `parallel_tool_calls`, image detail, cache
//!   hints, …) are dropped with a [`WarningKind::Unsupported`] warning by
//!   default. Set [`UnsupportedPolicy::Error`] on the request to make them
//!   fail instead.
//!
//! Adapters apply the policy while mapping the request, before anything is
//! sent, through the shared [`Degradations`] helper. They start with
//! [`chat::preflight`](crate::adapter::chat::preflight), which applies the policy to
//! what the provider reports `Unsupported`, and keep reporting through the
//! same helper while encoding.
//!
//! # Which channel
//!
//! A warning reaches the application on two channels, for two readers:
//!
//! - **Warning events** are for operators and logs. They are `tracing`
//!   events (below), deduplicated per process: a coded warning goes out at
//!   `WARN` once per provider, model, kind and code.
//! - **`response.warnings`** and **[`StreamEvent::Warning`](crate::StreamEvent::Warning)**
//!   are for program logic and the UI. They carry every occurrence, on
//!   every call; show each [recurring](Warning::recurs) warning once with
//!   [`SeenWarnings`].
//!
//! From a stream, show the warning events the stream yields or the
//! collected response's `warnings`, never both: the collected response
//! holds the warnings the stream yielded. An application that shows
//! warnings in its UI *and* logs to the same place (a terminal app with a
//! `tracing` subscriber on stderr) sees each one twice; it silences the
//! events with `allama::warning=off` in its filter, or builds `allama`
//! with `default-features = false` to drop the `warning-events` feature.
//!
//! A warning is not a reason to change what the application stores. A
//! `Dropped` warning about reasoning (`dropped.reasoning_replay`) says this
//! request went without it, not that the history should: store reasoning
//! as the response returned it, since the next provider or model may take
//! it back. Show the warning once, with [`SeenWarnings`], instead of
//! stripping the reasoning to silence it.
//!
//! # Warning events
//!
//! With the default `warning-events` feature, every warning a call reports
//! is also emitted as a [`tracing`](https://docs.rs/tracing) event by
//! [`emit`]. Without a subscriber the events cost nothing. A library that
//! depends on `allama` turns default features off, so the application
//! decides.
//!
//! - **Targets.** One per [`WarningKind`]: `allama::warning::unsupported`,
//!   `allama::warning::approximated`, `allama::warning::dropped`,
//!   `allama::warning::deprecated`, `allama::warning::fallback` and
//!   `allama::warning::other`. Filtering by target is the way to tune them,
//!   for example `RUST_LOG=allama::warning=warn,allama::warning::dropped=error`.
//!   A response cut off, filtered or refused is not a warning but a finish:
//!   its event is under `allama::finish`, at `INFO` (see
//!   [finish events](crate::chat#finish-events)).
//! - **Fields.** `allama.warning.provider`, `allama.warning.model`,
//!   `allama.warning.kind`, `allama.warning.subject` and
//!   `allama.warning.code`, and for a warning a failure caused (a
//!   failover), `allama.warning.cause_kind` and, for an `Unsupported`
//!   cause, `allama.warning.cause_stage` ([`Warning::cause`]); the event's
//!   message is the warning's message.
//! - **Recurrence.** A warning's [`code`](Warning::code) decides it. A
//!   warning without a code is about this call alone (a failover, content
//!   the application sent) and goes out at `WARN` every time. A coded
//!   warning describes the shape of the request or the configuration, so
//!   it recurs on every call with that shape: it goes out at `WARN` once per
//!   process and `(provider, model, kind, code)`, with
//!   `allama.warning.repeats` saying so, and at `DEBUG` after that: a batch
//!   job that sends the same ignored hint a thousand times logs one `WARN`.
//!   The keys are kept for the 4,096 most recently seen; when the set first
//!   fills, one event at `WARN` (target `allama::warning`) says so.
//!   [`Degradations`] gives a warning about a request field (or a system
//!   prompt) the code `<kind>.<field>`, such as `unsupported.seed`, when its
//!   producer set none.
//! - **Exactly once.** A warning remembers that it was emitted, so the
//!   innermost layer that sees it emits it, and wrappers that pass it on
//!   (through [`Prepared::finish`](crate::adapter::chat::Prepared::finish) or their
//!   own [`emit`]) do not emit it again. A clone keeps the mark: a warning
//!   copied into another response is not emitted again either. A
//!   deserialized warning is unmarked, so a response cache or replay layer
//!   *inside* a wrapper built on [`chat::prepare`](crate::adapter::chat::prepare)
//!   has its cached warnings emitted again on a hit; put response caches
//!   outside such wrappers. [`StreamTranscript::replay`](crate::chat::stream::StreamTranscript::replay)
//!   and [`RecordingProvider`](crate::wrappers::RecordingProvider) mark the
//!   warnings they replay or record.
//! - **Retries.** A retried attempt's warnings are not emitted:
//!   [`RetryingProvider`](crate::wrappers::RetryingProvider) runs each
//!   attempt in an attempt scope, which holds the events back until the
//!   attempt commits (or is the last), and drops them when it is retried.
//!   The scope is internal: a third-party retry or hedging wrapper's
//!   attempts each emit their warnings. The retries themselves are
//!   `allama::retry` events at `INFO`.
//! - **Fallback.** [`FallbackProvider`](crate::wrappers::FallbackProvider)
//!   holds nothing back: a primary stream that yields warnings and then
//!   fails before its first output has logged them, and the failover
//!   warning follows. That is two calls' worth of warnings, not a double
//!   report.
//! - **Refusals before sending.** A request refused before it was sent
//!   ([`Error::Unsupported`], or any error while encoding) emits nothing:
//!   the error is the only signal.
//! - **Cancellation.** A call dropped between polls drops its prepared
//!   request outside any attempt scope, so the warnings of a request that
//!   was sent go straight to `tracing`.
//!
//! [`ObserveProvider`](crate::wrappers::ObserveProvider) and
//! `response.warnings` see every occurrence, whatever the events do.
//!
//! # Warning codes
//!
//! A warning's [code](Warning::code) names the condition, so it is stable:
//! filter and alert on it, and match on it in program logic. A code is
//! never reused for another condition. A warning about this call alone (a failover, content
//! the application sent) has none.
//!
//! | Code | Kind | Written by | Condition |
//! | --- | --- | --- | --- |
//! | `<kind>.<field>`, such as `unsupported.seed` or `approximated.reasoning` | any | every adapter, through [`Degradations`] | a warning about a request field, or `<kind>.system` about a system prompt, whose producer set no code. `<kind>` is a [`WarningKind`] name, `<field>` a [`RequestField`] name |
//! | `unsupported.image_detail` | `Unsupported` | `allama` (preflight) | an image's detail was not sent: the provider has no detail setting |
//! | `dropped.reasoning_replay` | `Dropped` | `allama` and adapters ([`ReplayState::Reasoning`]) | reasoning the model or endpoint cannot take back (it takes none, or the block lacks a signature or encrypted content) was not sent |
//! | `dropped.foreign_reasoning` | `Dropped` | `allama` and adapters ([`ReplayState::ForeignReasoning`]) | reasoning another provider wrote was not sent |
//! | `dropped.foreign_provider_data` | `Dropped` | `allama` and adapters ([`ReplayState::ForeignProviderData`]) | provider data another provider wrote was not sent |
//! | `dropped.foreign_content` | `Dropped` | `allama` and adapters ([`ReplayState::ForeignContent`]) | `Other` content or a provider tool block from another provider was not sent |
//! | `dropped.content_replay` | `Dropped` | `allama` and adapters ([`ReplayState::Content`]) | content of the endpoint's own namespace that it cannot take back (a Responses item sent to Chat Completions) was not sent |
//! | `stream.missing_finish_reason` | `Other` | `allama` ([`MISSING_FINISH_REASON`](crate::adapter::chat::MISSING_FINISH_REASON)) | a stream ended with its terminal marker but no finish reason |
//! | `other.capability_override` | `Other` | `allama` (`CapabilityOverrideProvider`) | a capability override the wrapper cannot apply, since the wrapped provider answers it unsupported |
//! | `approximated.tool_call_arguments` | `Approximated` | `allama-anthropic`, `allama-gemini`, `allama-openai-compat` | tool call arguments that are not a JSON object were sent as `{}` |
//! | `approximated.tool_result_error` | `Approximated` | `allama-openai-compat`, `allama-cloudflare-worker` | a tool result marked as an error was sent as a plain result: the endpoint has no error flag |
//! | `approximated.thought_signature` | `Approximated` | `allama-gemini` | a function call without a Gemini thought signature was sent with Google's documented placeholder |
//! | `approximated.tool_result_order` | `Approximated` | `allama-gemini` | a tool result's media was sent after its text |
//! | `dropped.chat_completions_options` | `Dropped` | `allama-openai`, `allama-openai-compat` | Chat Completions options given to a Responses endpoint were not sent |
//! | `dropped.provider_name_options` | `Dropped` | `allama-gemini`, `allama-openai-compat`, `allama-cloudflare-worker`, `allama-systemone` | JSON options under the provider's name, which is not the options namespace it reads (`gemini`, not `google-ai`), were not read |
//! | `unsupported.logprobs`, `unsupported.top_logprobs` | `Unsupported` | `allama-openai-compat` | the model does not take `logprobs` or `top_logprobs` with reasoning enabled, so they were not sent |
//! | `approximated.input_truncated` | `Approximated` | `allama-cloudflare` | an extraction model reports that it truncated the input to fit its context |
//! | `dropped.span` | `Dropped` | `allama-cloudflare` | the span an extraction model reported for a text answer could not be placed in the request's input, so the answer has none |
//!
//! The codes are pinned by golden tests: `allama`'s `warning_codes` test
//! for the kind and field names and the derived codes, and a workspace
//! test for the codes each crate writes.

use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::Error;
use crate::macros::enum_names;

mod attempt;
pub(crate) mod degradations;
#[cfg(feature = "warning-events")]
mod events;

pub(crate) use attempt::AttemptScope;
pub(crate) use degradations::Degradations;

/// Emit `warning` as a warning event, unless it was emitted already, and
/// mark it emitted.
///
/// `provider` and `model` name the call the warning belongs to. With the
/// `warning-events` feature (on by default) the event is a
/// [`tracing`](https://docs.rs/tracing) event under the target of the
/// warning's kind (`allama::warning::unsupported`, ...), whose message is
/// the warning's message, with the fields `allama.warning.provider`,
/// `allama.warning.model`, `allama.warning.kind`, `allama.warning.subject`
/// and `allama.warning.code`. A warning without a code goes out at `WARN`; a
/// coded one at `WARN` the first time for its provider, model, kind and
/// code, and at `DEBUG` after that (see the [module docs](self#warning-events)).
/// Inside a `TracingProvider` span (feature `tracing`) it is a span event of
/// the call. Inside an attempt of a
/// [`RetryingProvider`](crate::wrappers::RetryingProvider) it waits for the
/// attempt to commit.
/// Without the feature, `emit` does nothing.
///
/// [`Prepared::finish`](crate::adapter::chat::Prepared::finish) and
/// [`finish_stream`](crate::adapter::chat::Prepared::finish_stream) call it for every
/// warning of a call, so an adapter built on
/// [`chat::prepare`](crate::adapter::chat::prepare) needs nothing else. Call it from a
/// wrapper for a warning the wrapper adds itself, and give the warning a
/// [code](Warning::with_code) when it recurs with the request's shape.
/// Because the mark travels with the warning, a warning passed through
/// several layers is emitted once, by the innermost one.
///
/// ```
/// use allama::{Subject, Warning, WarningKind};
///
/// let message = "the answer is cached; the provider was not called";
/// let mut warning = Warning::new(WarningKind::Other, Subject::Response, message);
/// allama::warning::emit("my-wrapper", "demo-model", &mut warning);
/// // A second call emits nothing.
/// allama::warning::emit("my-wrapper", "demo-model", &mut warning);
/// ```
pub fn emit(provider: &str, model: &str, warning: &mut Warning) {
    if warning.emitted {
        return;
    }
    #[cfg(feature = "warning-events")]
    {
        warning.emitted = true;
        events::emit(provider, model, warning);
    }
    #[cfg(not(feature = "warning-events"))]
    let _ = (provider, model);
}

/// Whether `warning` was emitted as a warning event already.
#[cfg(any(feature = "tracing", test))]
pub(crate) fn was_emitted(warning: &Warning) -> bool {
    warning.emitted
}

/// A non-fatal issue reported alongside a successful response.
///
/// # Emission
///
/// A warning remembers whether it was already emitted as a warning event
/// ([`emit`]), so a wrapper that passes it on emits it no second time. The
/// mark is not serialized and is ignored by `==`. A clone keeps it: a
/// warning copied into another response and passed through
/// [`Prepared::finish`](crate::adapter::chat::Prepared::finish) again is not logged
/// again. A deserialized warning is unmarked, so it is emitted again by the
/// next layer that finishes a call with it (see the
/// [module docs](self#warning-events) on response caches).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Warning {
    /// What kind of degradation happened.
    pub kind: WarningKind,
    /// What the warning is about.
    pub subject: Subject,
    /// Provider that produced the warning, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<Cow<'static, str>>,
    /// A stable code for the condition (`unsupported.seed`), when the
    /// producer gives one.
    ///
    /// A code says the warning describes the shape of a request or of the
    /// configuration, so it recurs on every call that has that shape: its
    /// warning event goes out at `WARN` once per provider, model, kind and
    /// code, then at `DEBUG` (see the [module docs](self#warning-events)).
    /// A warning without a code is about this call alone (a failover,
    /// content the application sent) and its event is `WARN` every time.
    /// A limitation about content that history repeats on every turn is
    /// coded all the same (`approximated.tool_result_error`, see
    /// [`Degradations::coded_approximated`]).
    /// [`Degradations`] gives a warning about a request field or a system
    /// prompt the code `<kind>.<field>` (`unsupported.seed`,
    /// `dropped.system`) when it has none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<Cow<'static, str>>,
    /// Human-readable explanation.
    pub message: String,
    /// The failure behind the warning, for a warning a failure caused: the
    /// error a [`FallbackProvider`](crate::wrappers::FallbackProvider)
    /// failed over on. Structured, so failovers can be counted by cause.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cause: Option<WarningCause>,
    /// Whether the warning was emitted as a warning event already.
    #[serde(skip)]
    emitted: bool,
}

impl PartialEq for Warning {
    /// Compares everything but whether the warning was emitted.
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.subject == other.subject
            && self.provider == other.provider
            && self.code == other.code
            && self.message == other.message
            && self.cause == other.cause
    }
}

/// The failure behind a [`Warning`] ([`Warning::cause`]), such as the error
/// a fallback failed over on.
///
/// Its warning event carries it as `allama.warning.cause_kind` (and
/// `allama.warning.cause_stage` for an `Unsupported` cause), so failovers
/// can be charted by cause; an
/// [`ObserveProvider`](crate::wrappers::ObserveProvider) reads it to count
/// them.
///
/// ```
/// use allama::{Error, WarningCause};
///
/// let cause = WarningCause::of(&Error::rate_limited("HTTP 429"));
/// assert_eq!(cause.kind, "rate_limited");
/// assert_eq!(cause.stage, None);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct WarningCause {
    /// The error's kind, as [`Error::kind_name`] names it
    /// (`rate_limited`, `unsupported`). A failed fallback chain returns the
    /// error that ended it as itself (see [`Error::fallback_from`]), so
    /// this is that error's kind.
    pub kind: Cow<'static, str>,
    /// Where an `Unsupported` cause was refused
    /// ([`Error::unsupported_stage`]): before sending, or by the provider.
    /// `None` for other kinds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<crate::UnsupportedStage>,
}

impl WarningCause {
    /// A cause of the given kind.
    #[must_use]
    pub fn new(kind: impl Into<Cow<'static, str>>) -> Self {
        Self {
            kind: kind.into(),
            stage: None,
        }
    }

    /// Set where an `Unsupported` cause was refused.
    #[must_use]
    pub fn with_stage(mut self, stage: crate::UnsupportedStage) -> Self {
        self.stage = Some(stage);
        self
    }

    /// The cause `error` is.
    #[must_use]
    pub fn of(error: &Error) -> Self {
        Self {
            kind: error.kind_name().into(),
            stage: error.unsupported_stage(),
        }
    }
}

impl Eq for Warning {}

/// Category of a [`Warning`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum WarningKind {
    /// A requested feature is not supported and was ignored.
    Unsupported,
    /// A requested feature was mapped to the closest supported equivalent.
    Approximated,
    /// Content or state was removed, e.g. another provider's reasoning
    /// signature during replay.
    Dropped,
    /// A requested feature is deprecated upstream.
    Deprecated,
    /// The call was served by a fallback provider after the primary
    /// failed (subject [`Subject::Response`]).
    Fallback,
    /// Any other warning.
    ///
    /// Also what a kind added by a newer allama reads as, so a stored
    /// warning (a [`TurnEnvelope`](crate::history::TurnEnvelope)) written
    /// by a newer version still loads. The tolerance is read-only and
    /// lossy: writing the value back stores `other`.
    #[serde(other)]
    Other,
}

/// What a [`Warning`] refers to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Subject {
    /// A portable request field.
    Field {
        /// The field.
        field: RequestField,
    },
    /// A content block or part in the conversation.
    Content {
        /// Index into `ChatRequest::messages`, from zero. System prompts
        /// are not messages ([`Subject::System`]), so they do not count.
        message: usize,
        /// Index of the block or part within the message, when applicable.
        ///
        /// Both indices are those of the caller's request, also when
        /// [`chat::preflight`](crate::adapter::chat::preflight) removed blocks before
        /// encoding (see [`Degradations`](Degradations#indices)).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        block: Option<usize>,
    },
    /// An embedding input, or one part of it.
    EmbeddingInput {
        /// Index into `EmbeddingRequest::inputs`, from zero.
        input: usize,
        /// Index of the part within the input, when applicable.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        part: Option<usize>,
    },
    /// Several messages of the conversation, by index into
    /// `ChatRequest::messages` (the caller's, as for [`Subject::Content`]),
    /// ascending: what one warning about a drop that recurs across
    /// messages refers to, such as the reasoning an endpoint takes no
    /// reasoning back for (see [`Degradations::dropped_replay_state`]).
    Messages {
        /// The messages, ascending and without repeats.
        messages: Vec<usize>,
    },
    /// A system prompt, by index into `ChatRequest::system`.
    System {
        /// Index into `ChatRequest::system`.
        index: usize,
    },
    /// A provider option, by namespace.
    Option {
        /// Namespace of the option.
        namespace: String,
    },
    /// The response itself (for example a lossy decode).
    Response,
    /// Anything else.
    Other {
        /// Free-form description of the subject.
        name: String,
    },
    /// A subject added by a newer allama, read by this version.
    ///
    /// Never emitted by allama, and adapters must not emit it: it exists
    /// so a stored warning (a
    /// [`TurnEnvelope`](crate::history::TurnEnvelope)) written by a newer
    /// version still loads. The tolerance is read-only and lossy: the
    /// subject's fields are discarded, and writing the value back stores
    /// `{"type": "unknown"}`.
    #[serde(other)]
    Unknown,
}

/// Portable request fields that warnings and policy decisions refer to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RequestField {
    /// `ChatRequest::temperature`.
    Temperature,
    /// `ChatRequest::max_output_tokens`.
    MaxOutputTokens,
    /// `ChatRequest::top_p`.
    TopP,
    /// `ChatRequest::top_k`.
    TopK,
    /// `ChatRequest::stop`.
    Stop,
    /// `ChatRequest::frequency_penalty`.
    FrequencyPenalty,
    /// `ChatRequest::presence_penalty`.
    PresencePenalty,
    /// `ChatRequest::seed`.
    Seed,
    /// `ChatRequest::tools`.
    Tools,
    /// `ChatRequest::tool_choice`.
    ToolChoice,
    /// `ChatRequest::parallel_tool_calls`.
    ParallelToolCalls,
    /// `ChatRequest::response_format`.
    ResponseFormat,
    /// `ChatRequest::reasoning`.
    Reasoning,
    /// `ChatRequest::system`.
    System,
    /// Prompt-cache hints.
    Cache,
    /// `EmbeddingRequest::dimensions`.
    Dimensions,
    /// `EmbeddingRequest::input_type`.
    InputType,
    /// A field added by a newer allama, read by this version.
    ///
    /// Never emitted by allama, and adapters must not emit it: it exists
    /// so a stored warning (a
    /// [`TurnEnvelope`](crate::history::TurnEnvelope)) written by a newer
    /// version still loads. The tolerance is read-only and lossy: writing
    /// the value back stores `unknown`.
    #[serde(other)]
    Unknown,
}

impl WarningKind {
    /// Stable snake_case name, as serialized.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unsupported => "unsupported",
            Self::Approximated => "approximated",
            Self::Dropped => "dropped",
            Self::Deprecated => "deprecated",
            Self::Fallback => "fallback",
            Self::Other => "other",
        }
    }
}

impl fmt::Display for WarningKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A phrase naming the subject, for messages: `` `seed` ``, `message 1,
/// block 0`, `system prompt 0`.
impl fmt::Display for Subject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Subject::Field { field } => write!(f, "`{field}`"),
            Subject::Content {
                message,
                block: Some(block),
            } => write!(f, "message {message}, block {block}"),
            Subject::Content {
                message,
                block: None,
            } => write!(f, "message {message}"),
            Subject::EmbeddingInput {
                input,
                part: Some(part),
            } => write!(f, "embedding input {input}, part {part}"),
            Subject::EmbeddingInput { input, part: None } => write!(f, "embedding input {input}"),
            Subject::Messages { messages } => {
                f.write_str("messages ")?;
                write_indices(f, messages, ", ")
            }
            Subject::System { index } => write!(f, "system prompt {index}"),
            Subject::Option { namespace } => write!(f, "the `{namespace}` options"),
            Subject::Response => f.write_str("the response"),
            Subject::Other { name } => f.write_str(name),
            Subject::Unknown => f.write_str("an unknown subject"),
        }
    }
}

/// `indices` joined by `separator`.
pub(crate) fn write_indices(
    f: &mut fmt::Formatter<'_>,
    indices: &[usize],
    separator: &str,
) -> fmt::Result {
    for (i, index) in indices.iter().enumerate() {
        if i > 0 {
            f.write_str(separator)?;
        }
        write!(f, "{index}")?;
    }
    Ok(())
}

impl RequestField {
    /// Every field allama emits, in declaration order (for audits such as
    /// the conformance honesty contract). [`Unknown`](Self::Unknown) is
    /// left out: it is only read. Grows when fields are added.
    pub const ALL: &'static [Self] = &[
        Self::Temperature,
        Self::MaxOutputTokens,
        Self::TopP,
        Self::TopK,
        Self::Stop,
        Self::FrequencyPenalty,
        Self::PresencePenalty,
        Self::Seed,
        Self::Tools,
        Self::ToolChoice,
        Self::ParallelToolCalls,
        Self::ResponseFormat,
        Self::Reasoning,
        Self::System,
        Self::Cache,
        Self::Dimensions,
        Self::InputType,
    ];

    /// Stable snake_case name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Temperature => "temperature",
            Self::MaxOutputTokens => "max_output_tokens",
            Self::TopP => "top_p",
            Self::TopK => "top_k",
            Self::Stop => "stop",
            Self::FrequencyPenalty => "frequency_penalty",
            Self::PresencePenalty => "presence_penalty",
            Self::Seed => "seed",
            Self::Tools => "tools",
            Self::ToolChoice => "tool_choice",
            Self::ParallelToolCalls => "parallel_tool_calls",
            Self::ResponseFormat => "response_format",
            Self::Reasoning => "reasoning",
            Self::System => "system",
            Self::Cache => "cache",
            Self::Dimensions => "dimensions",
            Self::InputType => "input_type",
            Self::Unknown => "unknown",
        }
    }
}

impl fmt::Display for RequestField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Warning {
    /// Create a warning.
    #[must_use]
    pub fn new(kind: WarningKind, subject: Subject, message: impl Into<String>) -> Self {
        Self {
            kind,
            subject,
            provider: None,
            code: None,
            message: message.into(),
            cause: None,
            emitted: false,
        }
    }

    /// An unsupported-field warning.
    #[must_use]
    pub fn unsupported(field: RequestField, message: impl Into<String>) -> Self {
        Self::new(WarningKind::Unsupported, Subject::Field { field }, message)
    }

    /// Set the provider identity.
    #[must_use]
    pub fn with_provider(mut self, provider: impl Into<Cow<'static, str>>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Set a stable code for the condition (see [`code`](Self::code)).
    ///
    /// Give one to a warning that recurs on every call with the same
    /// request shape or configuration, such as a hint the model never
    /// takes; leave it out for a warning about this call alone.
    #[must_use]
    pub fn with_code(mut self, code: &'static str) -> Self {
        self.code = Some(Cow::Borrowed(code));
        self
    }

    /// Whether the warning recurs on every call with the same request shape
    /// or configuration: whether it has a [code](Self::code).
    ///
    /// A recurring warning says the same thing on the next call, so an
    /// application shows it once, with [`SeenWarnings`]. A warning that does
    /// not recur is about this call alone (a failover, content the
    /// application sent) and is worth showing every time.
    ///
    /// ```
    /// use allama::{RequestField, Subject, Warning, WarningKind};
    ///
    /// let seed = Warning::unsupported(RequestField::Seed, "demo has no seed; `seed` was not sent")
    ///     .with_code("unsupported.seed");
    /// assert!(seed.recurs());
    ///
    /// let failover = Warning::new(
    ///     WarningKind::Fallback,
    ///     Subject::Response,
    ///     "failed with `overloaded`; served by `b`",
    /// );
    /// assert!(!failover.recurs());
    /// ```
    #[must_use]
    pub fn recurs(&self) -> bool {
        self.code.is_some()
    }

    /// The replay state the warning reports as dropped: a
    /// [`Dropped`](WarningKind::Dropped) warning coded with a
    /// [`ReplayState::code`], as [`Degradations::dropped_replay_state`]
    /// records it.
    ///
    /// Match on it, rather than on the message, to tell the user about
    /// replay state in your own words:
    ///
    /// ```
    /// use allama::adapter::Degradations;
    /// use allama::{ReplayState, Subject, UnsupportedPolicy};
    ///
    /// let mut d = Degradations::new("demo", UnsupportedPolicy::Warn);
    /// d.dropped_replay_state(
    ///     ReplayState::ForeignReasoning,
    ///     Subject::Content { message: 1, block: Some(0) },
    ///     "reasoning from `openai` can be replayed only to its origin; it was dropped",
    /// );
    /// let state = d.warnings()[0].replay_state();
    /// assert_eq!(state, Some(ReplayState::ForeignReasoning));
    /// assert!(state.is_some_and(ReplayState::is_foreign));
    /// ```
    #[must_use]
    pub fn replay_state(&self) -> Option<ReplayState> {
        if self.kind != WarningKind::Dropped {
            return None;
        }
        let code = self.code.as_deref()?;
        ReplayState::ALL
            .into_iter()
            .find(|state| state.code() == code)
    }

    /// Set the failure behind the warning (see [`cause`](Self::cause)).
    #[must_use]
    pub fn with_cause(mut self, cause: WarningCause) -> Self {
        self.cause = Some(cause);
        self
    }

    /// Mark the warning as emitted without emitting it: for warnings a
    /// replay or record hands back, which describe a call that was logged
    /// when it happened.
    pub(crate) fn mark_emitted(&mut self) {
        self.emitted = true;
    }

    /// Give the warning the code `<kind>.<field>` when it has none and is
    /// about a request field or a system prompt.
    fn with_default_code(mut self) -> Self {
        if self.code.is_none() {
            let field = match &self.subject {
                Subject::Field { field } => Some(field.as_str()),
                Subject::System { .. } => Some(RequestField::System.as_str()),
                _ => None,
            };
            if let Some(field) = field {
                self.code = Some(Cow::Owned(format!("{}.{field}", self.kind.as_str())));
            }
        }
        self
    }
}

/// The provider in brackets, then where the warning applies for a warning
/// about content, then the message: `[ollama] message 3, block 0: …`.
///
/// The location is shown for [`Subject::Content`] (`message 3, block 0`)
/// and [`Subject::EmbeddingInput`] (`embedding input 2, part 1`), whose
/// message cannot say which block it is about. Indices are zero-based and
/// those of the caller's request: `message` indexes
/// `ChatRequest::messages`, so system prompts, which are
/// `ChatRequest::system`, do not count. Other subjects are named by the
/// message itself (`` `seed` was not sent``), so nothing is added.
///
/// ```
/// use allama::{Subject, Warning, WarningKind};
///
/// let content = Subject::Content { message: 3, block: Some(0) };
/// let warning = Warning::new(WarningKind::Dropped, content, "`ollama` takes no reasoning back; it was dropped")
///     .with_provider("ollama");
/// assert_eq!(
///     warning.to_string(),
///     "[ollama] message 3, block 0: `ollama` takes no reasoning back; it was dropped"
/// );
/// ```
impl fmt::Display for Warning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(provider) = &self.provider {
            write!(f, "[{provider}] ")?;
        }
        if let Subject::Content { .. } | Subject::EmbeddingInput { .. } = self.subject {
            write!(f, "{}: ", self.subject)?;
        }
        f.write_str(&self.message)
    }
}

/// How adapters treat portable *hints* they cannot honor.
///
/// Semantic fields (`response_format`, tools, `tool_choice`, embedding
/// `dimensions`, embedding input parts a model cannot embed) always fail
/// regardless of this policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum UnsupportedPolicy {
    /// Drop the hint and report a [`Warning`] (default).
    #[default]
    Warn,
    /// Fail with [`Error::Unsupported`].
    Error,
}

enum_names!(UnsupportedPolicy, "unsupported policy", {
    Warn => "warn",
    Error => "error",
});

/// Replay state an adapter cannot send: what a provider wrote into a
/// conversation for its own later use, such as reasoning with its
/// signature or encrypted content, provider data, or provider-specific
/// content.
///
/// Replay state stays in the conversation, so it would be dropped again on
/// every later turn. [`Degradations::dropped_replay_state`] therefore
/// reports a drop with a stable [code](Self::code), and its warning event
/// goes out at `WARN` once per provider and model, not on every turn.
/// [`Warning::replay_state`] reads it back from a warning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ReplayState {
    /// Reasoning the model or endpoint cannot take back, whoever wrote it:
    /// it takes no reasoning back, or the block lacks the state replay needs
    /// (a signature, encrypted content). Code `dropped.reasoning_replay`.
    ///
    /// Compare [`ForeignReasoning`](Self::ForeignReasoning): this drop is
    /// about what the receiving model or endpoint can take back, so keeping
    /// the conversation on one provider does not avoid it, and
    /// `retain_provider_state_for` does not remove the block.
    Reasoning,
    /// Reasoning another provider wrote, under another namespace. Code
    /// `dropped.foreign_reasoning`.
    ///
    /// Compare [`Reasoning`](Self::Reasoning): this drop is about who wrote
    /// the block, and happens only after a provider switch or failover.
    /// [`retain_provider_state_for`](crate::history::retain_provider_state_for)
    /// removes such blocks for good.
    ForeignReasoning,
    /// Provider data under another provider's namespace. Code
    /// `dropped.foreign_provider_data`.
    ForeignProviderData,
    /// `Other` content, or a provider tool call or result, from another
    /// origin. Code `dropped.foreign_content`.
    ForeignContent,
    /// Content of the endpoint's own namespace that it cannot take back,
    /// such as a Responses item replayed to Chat Completions, which share
    /// the `openai` namespace. Code `dropped.content_replay`.
    ///
    /// Like [`Reasoning`](Self::Reasoning), this drop is about what the
    /// receiving endpoint can take back, so `retain_provider_state_for`
    /// does not remove the block.
    Content,
}

impl ReplayState {
    const ALL: [Self; 5] = [
        Self::Reasoning,
        Self::ForeignReasoning,
        Self::ForeignProviderData,
        Self::ForeignContent,
        Self::Content,
    ];

    /// The warning code a drop of this state carries.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::Reasoning => "dropped.reasoning_replay",
            Self::ForeignReasoning => "dropped.foreign_reasoning",
            Self::ForeignProviderData => "dropped.foreign_provider_data",
            Self::ForeignContent => "dropped.foreign_content",
            Self::Content => "dropped.content_replay",
        }
    }

    /// Whether another provider wrote the state.
    ///
    /// Foreign state is dropped only after a provider switch or a
    /// failover: on a failover, leave the history alone; once the
    /// conversation moves for good,
    /// [`retain_provider_state_for`](crate::history::retain_provider_state_for)
    /// removes it. Other replay state is dropped because the receiving
    /// model or endpoint cannot take it back, which no history change
    /// avoids.
    #[must_use]
    pub fn is_foreign(self) -> bool {
        matches!(
            self,
            Self::ForeignReasoning | Self::ForeignProviderData | Self::ForeignContent
        )
    }
}

/// Remembers which recurring warnings an application already showed, so it
/// shows each one once.
///
/// A [recurring](Warning::recurs) warning (one with a
/// [code](Warning::code)) says the same thing on every call with the same
/// request shape: a chat UI that prints `response.warnings` on every turn
/// would print "`seed` was not sent" on every turn. Keep a `SeenWarnings`
/// next to the conversation and print a warning only when
/// [`insert`](Self::insert) returns `true`.
///
/// The key is the one the [warning events](self#warning-events) deduplicate
/// on: the provider (the warning's own [`provider`](Warning::provider), the
/// name its event goes out under), the model the caller passes, the kind
/// and the code. The model is part of the key, so a warning shown for one
/// model is shown again after a switch to another. A warning without a
/// code is about this call alone and is always shown. Unlike the warning
/// events, which deduplicate once per process, a `SeenWarnings` is the
/// application's own: keep one per conversation, per user, or per process,
/// and [`clear`](Self::clear) it when the warnings should be shown again.
/// It grows by one entry per distinct key, and codes are few.
///
/// The library never deduplicates `response.warnings` itself: every
/// occurrence is reported, and what to show is the application's choice.
///
/// ```
/// use allama::{RequestField, SeenWarnings, Subject, Warning, WarningKind};
///
/// let seed = Warning::unsupported(RequestField::Seed, "demo has no seed; `seed` was not sent")
///     .with_provider("ollama")
///     .with_code("unsupported.seed");
/// let failover = Warning::new(
///     WarningKind::Fallback,
///     Subject::Response,
///     "failed with `overloaded`; served by `b`",
/// )
/// .with_provider("a");
///
/// let mut seen = SeenWarnings::new();
/// assert!(seen.insert("llama3.2", &seed));
/// assert!(!seen.insert("llama3.2", &seed), "shown once per model");
/// assert!(seen.insert("qwen3", &seed), "another model shows it again");
/// assert!(seen.insert("llama3.2", &failover));
/// assert!(seen.insert("llama3.2", &failover), "uncoded: always shown");
///
/// seen.clear();
/// assert!(seen.insert("llama3.2", &seed));
/// ```
#[derive(Debug, Clone, Default)]
pub struct SeenWarnings {
    seen: HashSet<SeenKey>,
}

impl SeenWarnings {
    /// An empty set: every warning is shown.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record `warning`, reported by a call to `model`, and return whether
    /// to show it.
    ///
    /// Always `true` for a warning without a code. For a coded one, `true`
    /// the first time its provider, `model`, kind and code are seen, and
    /// `false` after that. Pass the model the request named
    /// ([`ChatRequest::model`](crate::ChatRequest::model)), as the warning
    /// events do.
    pub fn insert(&mut self, model: &str, warning: &Warning) -> bool {
        let Some(code) = &warning.code else {
            return true;
        };
        self.seen.insert(SeenKey {
            provider: warning.provider.clone(),
            model: model.to_owned(),
            kind: warning.kind,
            code: code.clone(),
        })
    }

    /// Forget every warning seen, so each is shown again.
    pub fn clear(&mut self) {
        self.seen.clear();
    }
}

/// What [`SeenWarnings`] remembers a coded warning by: the warning events'
/// dedupe key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SeenKey {
    provider: Option<Cow<'static, str>>,
    model: String,
    kind: WarningKind,
    code: Cow<'static, str>,
}

/// Captures the warning events emitted while a test runs.
///
/// One global subscriber, installed once for the test binary, hands each
/// warning event to the capture of the thread that emitted it. A scoped
/// subscriber per test (`with_default`) would race: `tracing` caches each
/// callsite's interest process-wide, and a callsite first hit on another
/// thread while a test's subscriber is being registered can be cached as
/// uninteresting, dropping that test's events. A global subscriber that is
/// always interested never leaves a callsite cached that way.
///
/// Installing it has the same race in a smaller window: registering the
/// subscriber raises the global level filter before the subscriber becomes
/// the global default, and a callsite another test hits in between caches
/// "never" against no subscriber, possibly after the install's own rebuild
/// of the cache. So the subscriber is installed disarmed, with a level hint
/// of `OFF` that keeps every callsite unregistered, and armed only once it
/// is the global default (`install`).
#[cfg(all(test, feature = "warning-events"))]
pub(crate) mod capture {
    use std::cell::RefCell;
    use std::future::Future;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex, Once};

    use tracing::field::{Field, Visit};
    use tracing::level_filters::LevelFilter;
    use tracing::span::{Attributes, Id, Record};
    use tracing::subscriber::Interest;
    use tracing::{Event, Metadata, Subscriber};

    /// One warning event, as the subscriber saw it.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct Captured {
        pub(crate) target: String,
        pub(crate) level: tracing::Level,
        pub(crate) message: String,
        /// A warning event's `allama.warning.model`.
        pub(crate) model: Option<String>,
        pub(crate) code: Option<String>,
        pub(crate) repeats: bool,
        /// A finish event's `allama.finish.reason`.
        pub(crate) finish_reason: Option<String>,
        /// A failover warning's `allama.warning.cause_kind`.
        pub(crate) cause_kind: Option<String>,
        /// A failover warning's `allama.warning.cause_stage`.
        pub(crate) cause_stage: Option<String>,
    }

    type Sink = Arc<Mutex<Vec<Captured>>>;

    thread_local! {
        static SINK: RefCell<Option<Sink>> = const { RefCell::new(None) };
    }

    /// The captures that take the events of every thread (see
    /// [`captured_on_threads`]).
    static EVERY_THREAD: Mutex<Vec<Sink>> = Mutex::new(Vec::new());

    /// Run `future` on a current-thread runtime and return its output with
    /// the messages of the warning events it emitted, in order.
    pub(crate) fn warning_events<T>(future: impl Future<Output = T>) -> (T, Vec<String>) {
        let (output, events) = captured(future);
        let messages = events
            .into_iter()
            .filter(|e| e.target.starts_with("allama::warning"))
            .map(|e| e.message)
            .collect();
        (output, messages)
    }

    /// Run `future` on a current-thread runtime and return its output with
    /// the warning events (targets under `allama::warning`) and finish
    /// events (`allama::finish`) it emitted.
    pub(crate) fn captured<T>(future: impl Future<Output = T>) -> (T, Vec<Captured>) {
        install();
        let sink = Sink::default();
        SINK.with(|current| *current.borrow_mut() = Some(sink.clone()));
        let output = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .expect("a runtime")
            .block_on(future);
        SINK.with(|current| current.borrow_mut().take());
        let events = sink.lock().unwrap().clone();
        (output, events)
    }

    /// Run `future` on a multi-thread runtime of four workers and return its
    /// output with the warning events of `model_prefix`'s models, emitted on
    /// any thread.
    ///
    /// Events of every thread reach the capture while it runs, other tests'
    /// too, so the caller's models share a prefix no other test uses.
    pub(crate) fn captured_on_threads<T>(
        model_prefix: &str,
        future: impl Future<Output = T>,
    ) -> (T, Vec<Captured>) {
        install();
        let sink = Sink::default();
        EVERY_THREAD.lock().unwrap().push(sink.clone());
        let output = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_time()
            .build()
            .expect("a runtime")
            .block_on(future);
        EVERY_THREAD
            .lock()
            .unwrap()
            .retain(|every| !Arc::ptr_eq(every, &sink));
        let events = sink
            .lock()
            .unwrap()
            .iter()
            .filter(|e| {
                e.model
                    .as_deref()
                    .is_some_and(|m| m.starts_with(model_prefix))
            })
            .cloned()
            .collect();
        (output, events)
    }

    /// Make [`Global`] the global subscriber, once for the test binary.
    fn install() {
        static INSTALL: Once = Once::new();
        INSTALL.call_once(|| {
            // Disarmed, the level hint is `OFF`: registering the subscriber
            // keeps the global level filter at `OFF`, so no callsite caches
            // an interest before the subscriber is the global default.
            tracing::subscriber::set_global_default(Global)
                .expect("no other global subscriber in the unit tests");
            ARMED.store(true, Ordering::SeqCst);
            // Raise the level filter, and compute every callsite's interest
            // against the global subscriber.
            tracing::callsite::rebuild_interest_cache();
        });
    }

    /// Whether [`Global`] is the global default yet.
    static ARMED: AtomicBool = AtomicBool::new(false);

    /// Interested in every callsite; takes the warning events of threads
    /// with a capture, and nothing else (no spans).
    struct Global;

    impl Subscriber for Global {
        fn register_callsite(&self, _: &'static Metadata<'static>) -> Interest {
            Interest::sometimes()
        }

        fn max_level_hint(&self) -> Option<LevelFilter> {
            if ARMED.load(Ordering::SeqCst) {
                None
            } else {
                Some(LevelFilter::OFF)
            }
        }

        fn enabled(&self, metadata: &Metadata<'_>) -> bool {
            !metadata.is_span()
                && (metadata.target().starts_with("allama::warning")
                    || metadata.target() == "allama::finish")
                && (SINK.with(|sink| sink.borrow().is_some())
                    || !EVERY_THREAD.lock().unwrap().is_empty())
        }

        fn new_span(&self, _: &Attributes<'_>) -> Id {
            Id::from_u64(1)
        }

        fn record(&self, _: &Id, _: &Record<'_>) {}

        fn record_follows_from(&self, _: &Id, _: &Id) {}

        fn event(&self, event: &Event<'_>) {
            let metadata = event.metadata();
            let mut fields = Fields::default();
            event.record(&mut fields);
            let captured = Captured {
                target: metadata.target().to_owned(),
                level: *metadata.level(),
                message: fields.message,
                model: fields.model,
                code: fields.code,
                repeats: fields.repeats,
                finish_reason: fields.finish_reason,
                cause_kind: fields.cause_kind,
                cause_stage: fields.cause_stage,
            };
            for every in EVERY_THREAD.lock().unwrap().iter() {
                every.lock().unwrap().push(captured.clone());
            }
            SINK.with(|sink| {
                if let Some(sink) = sink.borrow().as_ref() {
                    sink.lock().unwrap().push(captured);
                }
            });
        }

        fn enter(&self, _: &Id) {}

        fn exit(&self, _: &Id) {}
    }

    #[derive(Default)]
    struct Fields {
        message: String,
        model: Option<String>,
        code: Option<String>,
        repeats: bool,
        finish_reason: Option<String>,
        cause_kind: Option<String>,
        cause_stage: Option<String>,
    }

    impl Visit for Fields {
        fn record_str(&mut self, field: &Field, value: &str) {
            match field.name() {
                "allama.warning.model" => self.model = Some(value.to_owned()),
                "allama.warning.code" => self.code = Some(value.to_owned()),
                "allama.warning.repeats" => self.repeats = true,
                "allama.finish.reason" => self.finish_reason = Some(value.to_owned()),
                "allama.warning.cause_kind" => self.cause_kind = Some(value.to_owned()),
                "allama.warning.cause_stage" => self.cause_stage = Some(value.to_owned()),
                _ => {}
            }
        }

        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            if field.name() == "message" {
                self.message = format!("{value:?}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A lossy hint that history repeats is one
    /// warning per code per request, listing the messages, as replay drops
    /// are.
    #[test]
    fn coded_lossy_hints_merge_per_code() {
        let mut d = Degradations::new("demo", UnsupportedPolicy::Warn);
        let code = "approximated.tool_result_error";
        for message in [5, 2, 5] {
            d.coded_lossy_hint(
                Subject::Content {
                    message,
                    block: None,
                },
                format!("demo has no tool error flag (message {message}); it was sent as text"),
                code,
            )
            .unwrap();
        }
        d.coded_lossy_hint(
            Subject::Content {
                message: 1,
                block: None,
            },
            "other; it was approximated",
            "approximated.tool_call_arguments",
        )
        .unwrap();
        d.coded_lossy_hint(
            Subject::Content {
                message: 1,
                block: Some(3),
            },
            "other; it was approximated",
            "approximated.tool_call_arguments",
        )
        .unwrap();
        let warnings = d.warnings();
        assert_eq!(warnings.len(), 2, "{warnings:?}");
        assert_eq!(warnings[0].code.as_deref(), Some(code));
        assert_eq!(
            warnings[0].subject,
            Subject::Messages {
                messages: vec![2, 5]
            }
        );
        assert!(warnings[0].message.contains("(message 5)"));
        assert_eq!(
            warnings[1].subject,
            Subject::Content {
                message: 1,
                block: None
            }
        );
    }

    /// A coded approximation is recorded, never
    /// refused, under every policy; it joins by code as a coded lossy hint
    /// does, and its subject is translated to the caller's indices.
    #[test]
    fn coded_approximations_are_recorded_under_every_policy() {
        let code = "approximated.tool_result_error";
        for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
            let mut d = Degradations::new("demo", policy);
            // Preflight removed block 0 of message 1.
            d.record_kept_blocks(1, 2, vec![1]);
            d.coded_approximated(
                Subject::Content {
                    message: 1,
                    block: Some(0),
                },
                "demo has no flag; it was sent as text",
                "approximated.demo",
            );
            for message in [6, 3, 6] {
                d.coded_approximated(
                    Subject::Content {
                        message,
                        block: None,
                    },
                    format!("demo has no tool error flag (message {message}); it was sent as text"),
                    code,
                );
            }
            let warnings = d.warnings();
            assert_eq!(warnings.len(), 2, "{policy:?}: {warnings:?}");
            assert_eq!(
                warnings[0].subject,
                Subject::Content {
                    message: 1,
                    block: Some(1)
                }
            );
            let joined = &warnings[1];
            assert_eq!(joined.kind, WarningKind::Approximated);
            assert_eq!(joined.code.as_deref(), Some(code));
            assert_eq!(joined.provider.as_deref(), Some("demo"));
            assert_eq!(
                joined.subject,
                Subject::Messages {
                    messages: vec![3, 6]
                }
            );
            assert!(joined.message.contains("(message 6)"), "{joined:?}");
        }
    }

    #[test]
    fn replay_state_reads_every_state_back() {
        // Exhaustive on purpose: a new variant fails to compile here, as a
        // reminder to add it to `ReplayState::ALL`.
        fn _exhaustive(state: ReplayState) {
            match state {
                ReplayState::Reasoning
                | ReplayState::ForeignReasoning
                | ReplayState::ForeignProviderData
                | ReplayState::ForeignContent
                | ReplayState::Content => {}
            }
        }
        let subject = Subject::Content {
            message: 0,
            block: None,
        };
        for state in ReplayState::ALL {
            let mut d = Degradations::new("demo", UnsupportedPolicy::Warn);
            d.dropped_replay_state(state, subject.clone(), "x; it was dropped");
            let warning = &d.warnings()[0];
            assert_eq!(warning.replay_state(), Some(state));
            assert_eq!(
                warning.message.contains("retain_provider_state_for"),
                state.is_foreign(),
                "{state:?}"
            );
        }

        // An adapter that codes the drop itself is read the same way.
        let coded = Warning::new(WarningKind::Dropped, subject.clone(), "x; it was dropped")
            .with_code("dropped.content_replay");
        assert_eq!(coded.replay_state(), Some(ReplayState::Content));
        assert!(!ReplayState::Content.is_foreign());

        // Another kind, another code, or no code is not replay state.
        let approximated = Warning::new(WarningKind::Approximated, subject.clone(), "x")
            .with_code("dropped.foreign_content");
        assert_eq!(approximated.replay_state(), None);
        let system =
            Warning::new(WarningKind::Dropped, subject.clone(), "x").with_code("dropped.system");
        assert_eq!(system.replay_state(), None);
        assert_eq!(
            Warning::new(WarningKind::Dropped, subject, "x").replay_state(),
            None
        );
    }

    /// The warning-codes golden (`tests/warning_codes.rs`) names every
    /// kind, so the codes derived from a new one are pinned too.
    #[test]
    fn the_warning_codes_golden_names_every_kind() {
        // Exhaustive on purpose: a new variant fails to compile here, as a
        // reminder to list it and regenerate the golden.
        fn index(kind: WarningKind) -> usize {
            match kind {
                WarningKind::Unsupported => 0,
                WarningKind::Approximated => 1,
                WarningKind::Dropped => 2,
                WarningKind::Deprecated => 3,
                WarningKind::Fallback => 4,
                WarningKind::Other => 5,
            }
        }
        let all = [
            WarningKind::Unsupported,
            WarningKind::Approximated,
            WarningKind::Dropped,
            WarningKind::Deprecated,
            WarningKind::Fallback,
            WarningKind::Other,
        ];
        for (i, kind) in all.into_iter().enumerate() {
            assert_eq!(index(kind), i);
        }
        let golden: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/warning_codes.json")).unwrap();
        let names: Vec<&str> = all.iter().map(|kind| kind.as_str()).collect();
        assert_eq!(golden["kinds"], serde_json::json!(names));
    }

    #[test]
    fn all_request_fields_listed() {
        // Exhaustive on purpose: a new variant fails to compile here, as a
        // reminder to add it to `RequestField::ALL` and bump the count.
        fn _exhaustive(field: RequestField) {
            match field {
                RequestField::Temperature
                | RequestField::MaxOutputTokens
                | RequestField::TopP
                | RequestField::TopK
                | RequestField::Stop
                | RequestField::FrequencyPenalty
                | RequestField::PresencePenalty
                | RequestField::Seed
                | RequestField::Tools
                | RequestField::ToolChoice
                | RequestField::ParallelToolCalls
                | RequestField::ResponseFormat
                | RequestField::Reasoning
                | RequestField::System
                | RequestField::Cache
                | RequestField::Dimensions
                | RequestField::InputType
                | RequestField::Unknown => {}
            }
        }
        let unique: std::collections::HashSet<_> = RequestField::ALL.iter().collect();
        assert_eq!(unique.len(), RequestField::ALL.len());
        assert_eq!(RequestField::ALL.len(), 17);
        assert!(!RequestField::ALL.contains(&RequestField::Unknown));
    }

    #[test]
    fn errors_are_authored_and_logged() {
        let mut strict = Degradations::new("demo", UnsupportedPolicy::Error);
        let errors = [
            strict.unsupported(RequestField::Seed, "no seed"),
            strict.reject(None, "no audio parts"),
            strict
                .hint(
                    Subject::Field {
                        field: RequestField::Seed,
                    },
                    "no seed; it was ignored",
                )
                .unwrap_err(),
            strict
                .coded_lossy_hint(
                    Subject::Field {
                        field: RequestField::Seed,
                    },
                    "args replayed as {}",
                    "approximated.seed",
                )
                .unwrap_err(),
        ];
        let lines: Vec<String> = errors.iter().map(|e| e.as_log().to_string()).collect();
        assert_eq!(
            lines,
            [
                r#"kind=unsupported provider=demo message="cannot send `seed`: no seed""#,
                r#"kind=unsupported provider=demo message="cannot send the request: no audio parts""#,
                r#"kind=unsupported provider=demo message="cannot send `seed`: no seed""#,
                r#"kind=unsupported provider=demo message="cannot send `seed`: args replayed as {}""#,
            ]
        );
    }

    #[test]
    fn warn_policy_collects() {
        let mut d = Degradations::new("demo", UnsupportedPolicy::Warn);
        d.unsupported_hint(RequestField::Seed, "no seed").unwrap();
        d.dropped(
            Subject::Content {
                message: 1,
                block: Some(0),
            },
            "foreign signature",
        );
        let warnings = d.into_warnings();
        assert_eq!(warnings.len(), 2);
        assert_eq!(warnings[0].kind, WarningKind::Unsupported);
        assert_eq!(warnings[0].provider.as_deref(), Some("demo"));
        assert_eq!(warnings[1].kind, WarningKind::Dropped);
    }

    #[test]
    fn error_policy_fails_hints_and_semantic_fields_always_fail() {
        let mut strict = Degradations::new("demo", UnsupportedPolicy::Error);
        let err = strict
            .unsupported_hint(RequestField::Seed, "no seed")
            .unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }));
        assert_eq!(err.context().unwrap().provider.as_deref(), Some("demo"));

        let lenient = Degradations::new("demo", UnsupportedPolicy::Warn);
        let err = lenient.unsupported(RequestField::ResponseFormat, "no json schema");
        assert!(err.to_string().contains("response_format"));
    }

    #[test]
    fn lossy_hints_warn_or_fail_and_rejections_are_attributed() {
        let subject = Subject::Content {
            message: 1,
            block: Some(0),
        };
        let mut lenient = Degradations::new("demo", UnsupportedPolicy::Warn);
        lenient
            .coded_lossy_hint(
                subject.clone(),
                "arguments replayed as {}",
                "approximated.tool_call_arguments",
            )
            .unwrap();
        assert_eq!(lenient.warnings()[0].kind, WarningKind::Approximated);
        assert_eq!(lenient.warnings()[0].subject, subject);

        let mut strict = Degradations::new("demo", UnsupportedPolicy::Error);
        assert!(
            strict
                .coded_lossy_hint(subject, "x", "approximated.tool_call_arguments")
                .is_err()
        );
        let err = strict.reject(None, "audio input is not supported");
        assert!(matches!(err, Error::Unsupported { .. }));
        assert_eq!(err.context().unwrap().provider.as_deref(), Some("demo"));
    }

    #[test]
    fn coded_lossy_hints_strip_the_outcome_and_translate_subjects() {
        let message = "demo has no tool error flag; the error text was sent as the tool result; \
                       to send it under a strict policy, push it with `push_tool_result`";
        let code = "approximated.tool_result_error";
        // Preflight removed block 0 of message 1, so block 0 is the
        // caller's block 1.
        let removed = |policy| {
            let mut d = Degradations::new("demo", policy);
            d.record_kept_blocks(1, 2, vec![1]);
            d
        };
        let subject = Subject::Content {
            message: 1,
            block: Some(0),
        };
        let caller = Subject::Content {
            message: 1,
            block: Some(1),
        };

        let mut lenient = removed(UnsupportedPolicy::Warn);
        lenient
            .coded_lossy_hint(subject.clone(), message, code)
            .unwrap();
        let warning = &lenient.warnings()[0];
        assert_eq!(warning.kind, WarningKind::Approximated);
        assert_eq!(warning.code.as_deref(), Some(code));
        assert_eq!(warning.subject, caller);
        assert_eq!(warning.message, message);
        assert_eq!(warning.provider.as_deref(), Some("demo"));

        let mut strict = removed(UnsupportedPolicy::Error);
        let err = strict.coded_lossy_hint(subject, message, code).unwrap_err();
        assert_eq!(err.unsupported_subject(), Some(&caller));
        assert_eq!(
            err.message(),
            "cannot send message 1, block 1: demo has no tool error flag; \
             to send it under a strict policy, push it with `push_tool_result`"
        );
        assert!(strict.warnings().is_empty());

        // Without a remedy, the refusal is the limitation alone.
        let err = strict
            .coded_lossy_hint(
                Subject::Content {
                    message: 0,
                    block: None,
                },
                "limited; it was approximated",
                code,
            )
            .unwrap_err();
        assert_eq!(err.message(), "cannot send message 0: limited");
    }

    /// Every policy-aware method refuses with the limitation and the
    /// remedy, never the outcome (ADR 0003, "how a warning reads").
    #[test]
    fn every_hint_refusal_drops_the_outcome_and_keeps_the_remedy() {
        let message = "demo has no seed; `seed` was not sent; remove it to send the request";
        let expected =
            "cannot send `seed`: demo has no seed; remove it to send the request".to_owned();
        let field = || Subject::Field {
            field: RequestField::Seed,
        };
        let mut strict = Degradations::new("demo", UnsupportedPolicy::Error);
        let refusals = [
            strict.unsupported_hint(RequestField::Seed, message),
            strict.hint(field(), message),
            strict.coded_hint(field(), message, "unsupported.seed"),
            strict.coded_lossy_hint(field(), message, "unsupported.seed"),
        ];
        for refusal in refusals {
            assert_eq!(refusal.unwrap_err().message(), expected);
        }
        let err = strict
            .unsupported_hint(RequestField::Seed, "demo has no seed; `seed` was not sent")
            .unwrap_err();
        assert_eq!(err.message(), "cannot send `seed`: demo has no seed");
    }

    /// A semantic refusal and a hint refused under `Error` read the same
    /// way, and the refused hint never claims it was sent or ignored.
    #[test]
    fn refusal_messages_share_one_form() {
        let mut strict = Degradations::new("demo", UnsupportedPolicy::Error);
        let semantic = strict.unsupported(RequestField::ResponseFormat, "no JSON Schema output");
        let hint = strict
            .unsupported_hint(RequestField::Seed, "Demo has no seed; it was ignored")
            .unwrap_err();
        let content = strict.reject(
            Subject::Content {
                message: 1,
                block: Some(0),
            },
            "audio input is not supported",
        );
        assert_eq!(
            semantic.to_string(),
            "demo: unsupported: cannot send `response_format`: no JSON Schema output"
        );
        assert_eq!(
            hint.to_string(),
            "demo: unsupported: cannot send `seed`: Demo has no seed"
        );
        assert_eq!(
            content.to_string(),
            "demo: unsupported: cannot send message 1, block 0: audio input is not supported"
        );
    }

    #[test]
    fn every_refusal_is_before_sending() {
        let mut strict = Degradations::new("demo", UnsupportedPolicy::Error);
        let refusals = [
            strict.unsupported(RequestField::Seed, "x"),
            strict.reject(None, "x"),
            strict
                .unsupported_hint(RequestField::Seed, "x")
                .unwrap_err(),
            strict
                .coded_lossy_hint(Subject::Response, "x", "approximated.response")
                .unwrap_err(),
        ];
        for err in refusals {
            assert_eq!(
                err.unsupported_stage(),
                Some(crate::UnsupportedStage::BeforeSending),
                "{err}"
            );
        }
    }

    #[test]
    fn refused_content_is_translated_to_the_callers_indices() {
        let mut d = Degradations::new("demo", UnsupportedPolicy::Error);
        // Preflight removed block 0 of message 1.
        d.record_kept_blocks(1, 2, vec![1]);
        let err = d.reject(
            Subject::Content {
                message: 1,
                block: Some(0),
            },
            "no images",
        );
        assert_eq!(
            err.unsupported_subject(),
            Some(&Subject::Content {
                message: 1,
                block: Some(1)
            })
        );
    }

    #[test]
    fn refusals_name_the_refused_field() {
        let mut strict = Degradations::new("demo", UnsupportedPolicy::Error);
        let field = |field| Subject::Field { field };
        let content = Subject::Content {
            message: 0,
            block: None,
        };
        let refusals = [
            (
                strict.unsupported(RequestField::ResponseFormat, "no schema"),
                Some(RequestField::ResponseFormat),
            ),
            (
                strict
                    .unsupported_hint(RequestField::Seed, "no seed")
                    .unwrap_err(),
                Some(RequestField::Seed),
            ),
            (
                strict
                    .hint(field(RequestField::Tools), "strict dropped")
                    .unwrap_err(),
                Some(RequestField::Tools),
            ),
            (
                strict
                    .coded_lossy_hint(
                        field(RequestField::Reasoning),
                        "approximated",
                        "approximated.reasoning",
                    )
                    .unwrap_err(),
                Some(RequestField::Reasoning),
            ),
            (strict.hint(content.clone(), "detail").unwrap_err(), None),
            (
                strict
                    .coded_lossy_hint(content, "args", "approximated.tool_call_arguments")
                    .unwrap_err(),
                None,
            ),
            (strict.reject(None, "no audio parts"), None),
        ];
        for (err, expected) in refusals {
            assert_eq!(err.unsupported_field(), expected, "{err}");
        }
        assert_eq!(Error::unsupported("HTTP 400").unsupported_field(), None);
    }

    #[test]
    fn the_refused_field_survives_serde_and_fallback() {
        let refused = Degradations::new("demo", UnsupportedPolicy::Warn)
            .unsupported(RequestField::ResponseFormat, "no schema");
        let json = serde_json::to_value(&refused).unwrap();
        assert_eq!(
            json["data"]["subject"],
            serde_json::json!({"type": "field", "field": "response_format"})
        );
        assert_eq!(json["data"]["stage"], "before_sending");
        let back: Error = serde_json::from_value(json).unwrap();
        assert_eq!(back.unsupported_field(), Some(RequestField::ResponseFormat));
        assert_eq!(
            back.unsupported_stage(),
            Some(crate::UnsupportedStage::BeforeSending)
        );

        let unnamed = serde_json::to_value(Error::unsupported("x")).unwrap();
        assert!(unnamed["data"].get("subject").is_none(), "{unnamed}");
        assert!(unnamed["data"].get("stage").is_none(), "{unnamed}");

        // A malformed subject fails with an error that says what is wrong.
        let err = serde_json::from_value::<Error>(serde_json::json!({
            "type": "unsupported",
            "data": {"message": "x", "subject": {"type": "content", "message": "three"}}
        }))
        .unwrap_err()
        .to_string();
        assert!(err.contains("invalid type"), "{err}");
        assert!(!err.contains("did not match any variant"), "{err}");
        // A stage this version does not know reads as unspecified.
        let newer: Error = serde_json::from_value(serde_json::json!({
            "type": "unsupported",
            "data": {"message": "x", "stage": "after_dinner",
                     "subject": {"type": "content", "message": 2, "block": 1}}
        }))
        .unwrap();
        assert_eq!(
            newer.unsupported_stage(),
            Some(crate::UnsupportedStage::Unspecified)
        );
        assert_eq!(
            newer.unsupported_subject(),
            Some(&Subject::Content {
                message: 2,
                block: Some(1)
            })
        );

        let chained = refused
            .clone()
            .with_fallback_from(Error::rate_limited("slow down"));
        assert_eq!(
            chained.unsupported_field(),
            Some(RequestField::ResponseFormat)
        );
        let reversed = Error::rate_limited("slow down").with_fallback_from(refused);
        assert_eq!(reversed.unsupported_field(), None);
        assert_eq!(
            reversed.fallback_from().unwrap().unsupported_field(),
            Some(RequestField::ResponseFormat)
        );
    }

    #[test]
    fn a_second_removal_composes_with_the_first() {
        let mut d = Degradations::new("demo", UnsupportedPolicy::Warn);
        // [r, a, r, b, c] -> [a, b, c] -> [a, c].
        d.record_kept_blocks(0, 5, vec![1, 3, 4]);
        d.record_kept_blocks(0, 3, vec![0, 2]);
        let content = |block| Subject::Content {
            message: 0,
            block: Some(block),
        };
        let translated: Vec<_> = (0..3)
            .map(|block| {
                d.translate(Warning::new(WarningKind::Other, content(block), "x"))
                    .subject
            })
            .collect();
        assert_eq!(translated, [content(1), content(4), content(5)]);
    }

    #[test]
    fn translation_leaves_the_provider_and_message_alone() {
        let mut d = Degradations::new("demo", UnsupportedPolicy::Warn);
        d.record_kept_blocks(1, 2, vec![1]);
        let warning = Warning::new(
            WarningKind::Dropped,
            Subject::Content {
                message: 1,
                block: Some(0),
            },
            "inner",
        )
        .with_provider("inner");
        let translated = d.translate(warning.clone());
        assert_eq!(translated.provider.as_deref(), Some("inner"));
        assert_eq!(translated.message, "inner");
        assert_eq!(
            translated.subject,
            Subject::Content {
                message: 1,
                block: Some(1)
            }
        );
        assert!(d.warnings().is_empty());
    }

    #[test]
    fn codes_round_trip_and_equality_ignores_emission() {
        let mut warning = Warning::unsupported(RequestField::Seed, "no seed").with_code("x.seed");
        let unmarked = warning.clone();
        emit("demo", "m", &mut warning);
        assert_eq!(warning, unmarked);
        let json = serde_json::to_value(&warning).unwrap();
        assert_eq!(json["code"], "x.seed");
        assert!(json.get("emitted").is_none(), "{json}");
        let back: Warning = serde_json::from_value(json).unwrap();
        assert_eq!(back, warning);
        assert!(!back.emitted);
        // Older warnings have no code.
        let old: Warning = serde_json::from_value(serde_json::json!({
            "kind": "unsupported", "subject": {"type": "response"}, "message": "m"
        }))
        .unwrap();
        assert_eq!(old.code, None);
    }

    #[test]
    fn degradations_give_field_and_system_warnings_a_default_code() {
        let mut d = Degradations::new("demo", UnsupportedPolicy::Warn);
        d.unsupported_hint(RequestField::Seed, "no seed").unwrap();
        d.approximated(
            Subject::Field {
                field: RequestField::Reasoning,
            },
            "budget rounded",
        );
        d.dropped(Subject::System { index: 1 }, "options ignored");
        d.dropped(
            Subject::Content {
                message: 0,
                block: None,
            },
            "attachment dropped",
        );
        d.warn(Warning::unsupported(RequestField::TopP, "x").with_code("custom.top_p"));
        d.warn(Warning::new(WarningKind::Other, Subject::Response, "x"));
        d.coded_hint(
            Subject::Content {
                message: 0,
                block: Some(0),
            },
            "detail",
            "unsupported.image_detail",
        )
        .unwrap();
        let codes: Vec<_> = d.warnings().iter().map(|w| w.code.as_deref()).collect();
        assert_eq!(
            codes,
            [
                Some("unsupported.seed"),
                Some("approximated.reasoning"),
                Some("dropped.system"),
                None,
                Some("custom.top_p"),
                None,
                Some("unsupported.image_detail"),
            ]
        );
        let mut strict = Degradations::new("demo", UnsupportedPolicy::Error);
        let content = Subject::Content {
            message: 0,
            block: Some(0),
        };
        assert!(strict.coded_hint(content, "detail", "x.y").is_err());
    }

    #[cfg(feature = "warning-events")]
    #[test]
    fn emit_sends_a_warning_once() {
        let mut warning = Warning::new(WarningKind::Other, Subject::Response, "once");
        let ((), events) = capture::warning_events(async {
            emit("demo", "m", &mut warning);
            let mut copy = warning.clone();
            emit("wrapper", "m", &mut copy);
        });
        assert_eq!(events, ["once"]);
    }

    /// `SeenWarnings` keys on provider, model, kind and code, like the
    /// warning events: message text and subject are not in the key, and an
    /// uncoded warning is always shown.
    #[test]
    fn seen_warnings_key_on_provider_model_kind_and_code() {
        let coded = |provider: &'static str, kind: WarningKind, message: &str| {
            Warning::new(
                kind,
                Subject::Content {
                    message: 0,
                    block: Some(1),
                },
                message,
            )
            .with_provider(provider)
            .with_code("dropped.reasoning_replay")
        };
        let mut seen = SeenWarnings::new();
        assert!(seen.insert("m", &coded("a", WarningKind::Dropped, "first")));
        assert!(
            !seen.insert("m", &coded("a", WarningKind::Dropped, "other text")),
            "message text is not in the key"
        );
        let mut moved = coded("a", WarningKind::Dropped, "first");
        moved.subject = Subject::Content {
            message: 5,
            block: None,
        };
        assert!(!seen.insert("m", &moved), "the subject is not in the key");
        assert!(seen.insert("m", &coded("b", WarningKind::Dropped, "first")));
        assert!(seen.insert("m", &coded("a", WarningKind::Other, "first")));
        assert!(seen.insert("n", &coded("a", WarningKind::Dropped, "first")));

        let unnamed = Warning::new(WarningKind::Dropped, Subject::Response, "x")
            .with_code("dropped.reasoning_replay");
        assert!(seen.insert("m", &unnamed), "no provider is its own key");
        assert!(!seen.insert("m", &unnamed));

        let uncoded = Warning::new(WarningKind::Dropped, Subject::Response, "x");
        assert!(!uncoded.recurs());
        assert!(seen.insert("m", &uncoded));
        assert!(seen.insert("m", &uncoded));

        seen.clear();
        assert!(seen.insert("m", &coded("a", WarningKind::Dropped, "first")));
    }

    /// `Display` shows where a content or embedding-input warning applies,
    /// after the provider, and nothing extra for other subjects.
    #[test]
    fn display_shows_the_location_of_content_and_embedding_inputs() {
        let shown = |subject: Subject, provider: Option<&'static str>| {
            let warning = Warning::new(WarningKind::Dropped, subject, "it was dropped");
            match provider {
                Some(provider) => warning.with_provider(provider),
                None => warning,
            }
            .to_string()
        };
        let content = |message, block| Subject::Content { message, block };
        assert_eq!(
            shown(content(3, Some(0)), Some("ollama")),
            "[ollama] message 3, block 0: it was dropped"
        );
        assert_eq!(
            shown(content(1, None), Some("ollama")),
            "[ollama] message 1: it was dropped"
        );
        assert_eq!(
            shown(content(0, Some(2)), None),
            "message 0, block 2: it was dropped"
        );
        assert_eq!(
            shown(
                Subject::EmbeddingInput {
                    input: 2,
                    part: Some(1)
                },
                Some("gemini")
            ),
            "[gemini] embedding input 2, part 1: it was dropped"
        );
        assert_eq!(
            shown(
                Subject::EmbeddingInput {
                    input: 4,
                    part: None
                },
                None
            ),
            "embedding input 4: it was dropped"
        );
        for subject in [
            Subject::Field {
                field: RequestField::Seed,
            },
            Subject::System { index: 0 },
            Subject::Response,
            Subject::Option {
                namespace: "x".into(),
            },
            Subject::Other { name: "y".into() },
            Subject::Unknown,
        ] {
            assert_eq!(shown(subject, Some("p")), "[p] it was dropped");
        }
    }

    #[test]
    fn warning_serde_round_trip() {
        let w = Warning::unsupported(RequestField::TopP, "ignored").with_provider("gemini");
        let json = serde_json::to_value(&w).unwrap();
        assert_eq!(json["subject"]["field"], "top_p");
        let back: Warning = serde_json::from_value(json).unwrap();
        assert_eq!(back, w);
    }

    /// A warning written by a newer allama, with a kind, field or subject
    /// this version does not know, still loads; writing it back stores the
    /// fallback value.
    #[test]
    fn newer_kinds_fields_and_subjects_read_as_their_fallbacks() {
        use serde_json::json;

        let read = |value: serde_json::Value| serde_json::from_value::<Warning>(value).unwrap();
        let kind = read(json!({"kind": "throttled", "subject": {"type": "response"},
                               "message": "m"}));
        assert_eq!(kind.kind, WarningKind::Other);
        assert_eq!(kind.subject, Subject::Response);

        let field = read(json!({"kind": "unsupported",
                                "subject": {"type": "field", "field": "logit_bias"},
                                "message": "m"}));
        assert_eq!(
            field.subject,
            Subject::Field {
                field: RequestField::Unknown
            }
        );
        assert_eq!(RequestField::Unknown.as_str(), "unknown");

        let subject = read(json!({"kind": "dropped",
                                  "subject": {"type": "audio_track", "track": 2, "label": "x"},
                                  "provider": "p", "message": "m"}));
        assert_eq!(subject.kind, WarningKind::Dropped);
        assert_eq!(subject.subject, Subject::Unknown);
        assert_eq!(subject.provider.as_deref(), Some("p"));

        // Read-only and lossy: what this version did not know is not kept.
        assert_eq!(serde_json::to_value(&kind).unwrap()["kind"], json!("other"));
        assert_eq!(
            serde_json::to_value(&field).unwrap()["subject"],
            json!({"type": "field", "field": "unknown"})
        );
        assert_eq!(
            serde_json::to_value(&subject).unwrap()["subject"],
            json!({"type": "unknown"})
        );
    }
}
