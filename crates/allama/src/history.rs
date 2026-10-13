//! Versioned conversation history for persistence.
//!
//! [`History`] wraps a conversation's messages in an envelope that records
//! the serialization format version, so stored conversations can be loaded
//! safely after the format evolves. Loading a history written by a
//! newer, unknown format version fails instead of silently misreading it
//! (see [Format evolution](#format-evolution)).
//!
//! [`retain_provider_state_for`] prepares a conversation for a permanent
//! move to another provider: it removes opaque state that the provider
//! cannot use, once, so the next requests do not report the same loss on
//! every turn. On failover, do nothing: adapters drop foreign state for each
//! request without touching the history.
//!
//! [`validate`] checks that tool calls and tool results are where every
//! provider expects them, before a request is sent or a history is stored.
//!
//! [`retain_portable_content`] sanitizes a conversation an end user sent
//! (see [The trust boundary](#the-trust-boundary)).
//!
//! # Tool calls and results
//!
//! Each client tool call in an assistant message needs its result in the
//! run of [`Message::Tool`] messages immediately after that assistant
//! message. Within the run the results may come in any order. Providers
//! reject anything else with an HTTP 400, each in its own words: OpenAI
//! Chat Completions wants an assistant message with tool calls followed by
//! a tool message for each call, Anthropic wants every `tool_result` in the
//! next user turn before any text (its adapter merges the run, and a user
//! message right after it, into that turn), and Gemini wants as many
//! function responses as calls in the next turn. The OpenAI Responses API
//! is looser; the shared rule is safe there too. Tool call ids are unique
//! across the whole conversation: Anthropic rejects an id that an earlier
//! turn already used. Adapters that make up an id for a call the provider
//! sent without one make it unique too.
//!
//! Some provider tools are run by the client but are not function calls:
//! a freeform custom tool (OpenAI Responses and Chat Completions), or a
//! shell command or a computer-use action in the OpenAI Responses API, for
//! example. Their call and answer shapes do not map across providers, so
//! they are not [`ToolCall`](crate::ToolCall)s (ADR 0014): the call arrives
//! as [`Other`](crate::ContentBlock::Other) content, which
//! [`ChatResponse::tool_calls`](crate::ChatResponse::tool_calls) does not
//! return, and the provider crate reads it and builds its answer (for
//! OpenAI, `allama_openai::ClientToolCall`). On failover, another provider
//! drops such a call and its answer with warnings. An agent loop
//! that offers such tools asks the provider crate for them after every
//! turn. Adapters report the finish reason
//! [`ToolCalls`](crate::FinishReason::ToolCalls) while one waits, even when
//! `tool_calls` is empty, so a loop that only
//! knows function tools sees that the model is waiting rather than
//! finished. [`validate`] checks only function
//! tool calls: it reads no provider-specific content, so it neither
//! reports such a call without an answer nor checks where the answer is.
//! Where a provider expresses its tools as ordinary function calls
//! (Anthropic's bash and computer-use tools, Gemini's computer use), they
//! are `ToolCall`s, and everything here applies to them.
//!
//! A history that an agent loop holds between the model's turn and the
//! tool results does not meet the rule yet: the last assistant message
//! has *pending* calls. [`ValidationMode::Sendable`] rejects them, and
//! [`ValidationMode::Resumable`] accepts them at the end of the history,
//! so a loop can check what a failed or interrupted turn left behind
//! before storing it. Since [`validate`] does not read provider-specific
//! calls, a loop that offers client-executed provider tools also asks the
//! provider crate what is pending (for OpenAI,
//! `allama_openai::ClientToolCall::pending`) before it treats a stored
//! history as complete.
//!
//! ```
//! use allama::history::{self, HistoryIssue, ValidationMode};
//! use allama::{ChatProvider, ChatRequest, ChatResponse, ContentBlock, Error, Message, Result};
//!
//! /// Send the next turn, refusing a history providers would reject.
//! async fn next_turn(provider: &impl ChatProvider, request: &ChatRequest) -> Result<ChatResponse> {
//!     history::validate(&request.messages, ValidationMode::Sendable).map_err(invalid)?;
//!     provider.chat(request).await
//! }
//!
//! /// After a failed turn, check that the messages can be stored and resumed:
//! /// the last assistant message may still wait for tool results.
//! fn checkpoint(messages: &[Message]) -> Result<String> {
//!     history::validate(messages, ValidationMode::Resumable).map_err(invalid)?;
//!     serde_json::to_string(&history::History::new(messages.to_vec())).map_err(Error::serialization)
//! }
//!
//! fn invalid(issues: Vec<HistoryIssue>) -> Error {
//!     let issues: Vec<String> = issues.iter().map(ToString::to_string).collect();
//!     Error::invalid_request(issues.join("; ")).authored()
//! }
//!
//! // The model asked for two tools; the process stopped after the first result.
//! let mut messages = vec![
//!     Message::user("Weather in Paris and Tokyo?"),
//!     Message::assistant_blocks(vec![
//!         ContentBlock::tool_call("call_1", "get_weather", r#"{"city":"Paris"}"#),
//!         ContentBlock::tool_call("call_2", "get_weather", r#"{"city":"Tokyo"}"#),
//!     ]),
//!     Message::tool_result("call_1", "get_weather", "Sunny"),
//! ];
//! assert!(checkpoint(&messages).is_ok());
//! let issues = history::validate(&messages, ValidationMode::Sendable).unwrap_err();
//! assert!(matches!(
//!     &issues[..],
//!     [HistoryIssue::UnansweredToolCall { message: 1, call_id, .. }] if call_id == "call_2"
//! ));
//!
//! // On resume, run the pending call and append its result before sending.
//! messages.push(Message::tool_result("call_2", "get_weather", "Rain"));
//! assert!(history::validate(&messages, ValidationMode::Sendable).is_ok());
//! ```
//!
//! # Application data
//!
//! Every message and system prompt has an `app_data` map for the
//! application's own data about it: message ids, timestamps, UI state,
//! per-turn usage. Unlike [`ProviderData`], which belongs to providers,
//! application data belongs to the application:
//!
//! - adapters never read it and never send it;
//! - [`retain_provider_state_for`] and [`retain_provider_state`] never
//!   touch it;
//! - [`History`] persists it (an empty map is not serialized);
//! - [`ChatResponse::to_assistant_message`](crate::ChatResponse::to_assistant_message)
//!   leaves it empty;
//! - `TracingProvider` (feature `tracing`) never records it, since it may
//!   hold personal data, while
//!   [`RecordingProvider`](crate::wrappers::RecordingProvider) keeps it
//!   unless told to strip it, so recorded exchanges replay the request
//!   exactly.
//!
//! [`Message::app_data_get`] and [`Message::app_data_insert`] encode and
//! decode typed values; [`Message::app_data`] and
//! [`Message::app_data_mut`] work on any role.
//!
//! ```
//! use allama::Message;
//! use allama::history::History;
//! use serde_json::json;
//!
//! let mut reply = Message::assistant("Hello!");
//! reply.app_data_insert("myapp.message_id", &"msg_02")?;
//! let history = History::new(vec![
//!     Message::user("Hi").with_app_data("myapp.message_id", json!("msg_01")),
//!     reply,
//! ]);
//!
//! let stored = serde_json::to_string(&history)?;
//! let loaded: History = serde_json::from_str(&stored)?;
//! let id: Option<String> = loaded.messages[1].app_data_get("myapp.message_id")?;
//! assert_eq!(id.as_deref(), Some("msg_02"));
//! # Ok::<(), allama::Error>(())
//! ```
//!
//! Keys starting with `allama.` are reserved for conventions allama
//! publishes; prefix your own keys with something else, such as the name
//! of your application.
//!
//! ## The turn envelope
//!
//! [`ChatResponse::to_assistant_message`](crate::ChatResponse::to_assistant_message)
//! keeps only what the next request replays, the content. The rest of the
//! response (id, request id, model, serving provider, usage, finish
//! reason, raw finish reason, warnings) is what cost accounting, support
//! tickets and audit logs need per turn. [`ChatResponse::envelope`](crate::ChatResponse::envelope)
//! returns it as a [`TurnEnvelope`], and [`TURN_ENVELOPE_KEY`] is the key
//! allama publishes for it, so tools can read each other's stored
//! histories. Nothing stores it automatically; storing it is the
//! application's choice. Each provider call has its own envelope: a turn
//! that takes several calls, such as a tool round trip, stores one on each
//! assistant message a call produced.
//!
//! ```
//! use allama::history::{History, TURN_ENVELOPE_KEY, TurnEnvelope};
//! use allama::{ChatResponse, ContentBlock, Message, Usage};
//!
//! # let response = ChatResponse::new(vec![ContentBlock::text("Hello!")])
//! #     .id("resp_01")
//! #     .usage(Usage::new().input_tokens(12).output_tokens(3));
//! let mut message = response.to_assistant_message();
//! message.app_data_insert(TURN_ENVELOPE_KEY, &response.envelope())?;
//! let history = History::new(vec![Message::user("Hi"), message]);
//!
//! let stored = serde_json::to_string(&history)?;
//! let loaded: History = serde_json::from_str(&stored)?;
//! let turn: TurnEnvelope = loaded.messages[1]
//!     .app_data_get(TURN_ENVELOPE_KEY)?
//!     .expect("stored with the reply");
//! assert_eq!(turn.id.as_deref(), Some("resp_01"));
//! assert_eq!(turn.usage.and_then(|usage| usage.output_tokens), Some(3));
//! # Ok::<(), allama::Error>(())
//! ```
//!
//! An envelope written by a newer allama still loads: a warning kind,
//! request field or subject this version does not know reads as
//! [`WarningKind::Other`], [`RequestField::Unknown`](crate::RequestField::Unknown)
//! or [`Subject::Unknown`], and fields this version does not have are
//! ignored. The tolerance is read-only and lossy: storing such an envelope
//! again writes the fallback values, so rewrite stored envelopes only with
//! the newest version that reads them.
//!
//! The envelope holds what every response says about itself, not what a
//! wrapper adds in provider data. To keep a
//! [`FallbackProvider`](crate::wrappers::FallbackProvider)'s failed
//! attempts with the turn, store
//! [`FallbackAttempts`](crate::wrappers::FallbackAttempts) beside the
//! envelope, under a key your application owns:
//!
//! ```
//! use allama::history::TURN_ENVELOPE_KEY;
//! use allama::wrappers::FallbackAttempts;
//! # use allama::ChatResponse;
//!
//! # let response = ChatResponse::default();
//! let mut message = response.to_assistant_message();
//! message.app_data_insert(TURN_ENVELOPE_KEY, &response.envelope())?;
//! if let Some(attempts) = response.provider_data.get::<FallbackAttempts>()? {
//!     message.app_data_insert("myapp.fallback_attempts", &attempts)?;
//! }
//! # Ok::<(), allama::Error>(())
//! ```
//!
//! # Format evolution
//!
//! A stored history outlives the build that wrote it: it is read after an
//! upgrade, and during a rolling deploy or after a rollback an older build
//! may read what a newer one wrote. The format follows these rules.
//!
//! - **Saving writes the current version.** A [`History`] always
//!   serializes [`HISTORY_FORMAT_VERSION`], whatever version it was loaded
//!   from; [`History::loaded_format_version`] reports the version it was
//!   read from.
//! - **Every change bumps the version.** After 0.1, any change to the
//!   persisted shape bumps [`HISTORY_FORMAT_VERSION`], additive ones
//!   included: a new field, content kind or role. A build reads every
//!   version up to its own, so histories written earlier keep loading.
//! - **A bump ships only in a minor release.** Never in a patch release,
//!   so a `cargo update` within a minor line cannot leave an older build
//!   (the other pods of a rolling deploy) unable to read what a newer one
//!   saves.
//! - **Newer data fails to load.** A history at a newer version fails with
//!   an error that names both versions, and a content `type` this build
//!   does not know fails with "unknown content block type `audio` (written
//!   by a newer allama?)". A field this build does not know is ignored on
//!   load and lost on the next save; the version bump is what keeps an
//!   older build from reading data that has one. Rolling back across a
//!   bump therefore needs a migration of the histories written since.
//! - **Provider state is part of the format.** The provider data keys an
//!   adapter writes into a history, or reads from one (signatures, item
//!   ids, encrypted reasoning), are persisted with it, but the core version
//!   does not cover them. An
//!   adapter keeps reading every shape it ever wrote: a provider crate that
//!   renames or restructures a key keeps reading the old one, so stored
//!   conversations keep their signatures.
//!
//! Application data is the application's own and follows its own rules.
//! [Turn envelopes](#the-turn-envelope) are read tolerantly, as described
//! there.
//!
//! # The trust boundary
//!
//! Messages, histories and requests that come from an end user are
//! untrusted input. A history is more than text: provider data, `Other`
//! content and [provider files](crate::MediaSource::ProviderFile) are
//! escape hatches that adapters send to the provider as they are, with the
//! operator's key. Namespace gating (ADR 0002) keeps one provider's state
//! from another provider; it does not tell the application's state from a
//! forgery. A chat UI that sends the whole conversation back on every turn
//! therefore lets the user:
//!
//! - write the assistant's turns and the tool results;
//! - inject raw provider items through `Other` parts without an origin
//!   (an OpenAI `item_reference` or `input_file`, an Anthropic `document`
//!   by `file_id`), which reach resources of the operator's account;
//! - replay provider state, and reference files by id.
//!
//! Keep conversations server-side and send only the new message from the
//! client, as the `web_backend` example does with its database. Where the
//! client must hold the history, pass it through
//! [`retain_portable_content`] (or [`History::retain_portable_content`])
//! before use: it removes every escape hatch and keeps the portable
//! content. Never deserialize a whole [`ChatRequest`] from an end user:
//! its [`options`](ChatRequest::options) set HTTP headers and body fields
//! for the provider.

use std::borrow::Cow;

use serde::{Deserialize, Deserializer, Serialize};

#[cfg(test)]
mod golden;
mod portable;
mod validate;

pub use portable::retain_portable_content;
pub use validate::{HistoryIssue, ValidationMode, validate};

use crate::adapter::describe_origins;
use crate::provider_data::ALLAMA_NAMESPACE;
use crate::{
    ChatRequest, ContentBlock, ContentPart, FinishReason, Message, ProviderData, ProviderIdentity,
    Subject, ToolResultContent, Usage, UserContent, Warning, WarningKind,
};

/// Current serialization format version of [`History`], content blocks, and
/// messages.
///
/// The golden fixture `crates/allama/tests/fixtures/history_v1.json` and the
/// key paths in `crates/allama/src/chat/history_v1_paths.txt` pin the
/// format; any change to either is a format change (see
/// `crates/allama/tests/format_contract.rs`).
pub const HISTORY_FORMAT_VERSION: u32 = 1;

/// The [application data](self#application-data) key for a
/// [`TurnEnvelope`] stored on the assistant message of its turn (see
/// [The turn envelope](self#the-turn-envelope)).
pub const TURN_ENVELOPE_KEY: &str = "allama.turn";

/// A persistable conversation: its messages, in order.
///
/// A history holds no system prompt. The system prompt is set on every
/// request; an application that keeps a prompt per conversation stores
/// what it needs to rebuild the prompt (the text, or a persona id) next to
/// the history, and builds the [`SystemPrompt`](crate::SystemPrompt) (cache
/// breakpoint, options) on every request. A `SystemPrompt` is not versioned
/// (see [Serialization and stability](crate#serialization-and-stability)).
///
/// Serializing always writes the current [`HISTORY_FORMAT_VERSION`], and
/// loading reads every version up to it (see
/// [Format evolution](self#format-evolution)).
/// [`loaded_format_version`](Self::loaded_format_version) reports the
/// version a loaded history was written with; equality ignores it.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct History {
    /// Messages, in order.
    pub messages: Vec<Message>,
    /// The format version this history was read from; `None` when it was
    /// built in memory.
    loaded_format_version: Option<u32>,
}

impl History {
    /// A history at the current format version.
    #[must_use]
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            loaded_format_version: None,
        }
    }

    /// The format version this history was written with, when it was
    /// loaded; `None` for a history built in memory.
    ///
    /// Saving it again always writes the current [`HISTORY_FORMAT_VERSION`],
    /// so a history loaded from an older version is upgraded on its next
    /// save. Applications that keep older binaries running (a rolling
    /// deploy, a rollback) can compare it to decide when to rewrite stored
    /// conversations; the version changes only in a minor release (see
    /// [Format evolution](self#format-evolution)).
    #[must_use]
    pub fn loaded_format_version(&self) -> Option<u32> {
        self.loaded_format_version
    }

    /// Capture the messages of a request.
    ///
    /// The request's system prompts are not captured: set them on every
    /// request (see [`History`]).
    #[must_use]
    pub fn from_request(request: &ChatRequest) -> Self {
        Self::new(request.messages.clone())
    }

    /// Remove opaque provider state `provider` cannot use; see
    /// [`retain_provider_state_for`].
    pub fn retain_provider_state_for<P>(&mut self, provider: &P) -> Vec<Warning>
    where
        P: ProviderIdentity + ?Sized,
    {
        retain_provider_state_for(&mut self.messages, provider)
    }

    /// Keep only portable content, for a history an end user sent; see
    /// [`retain_portable_content`].
    pub fn retain_portable_content(&mut self) -> Vec<Warning> {
        retain_portable_content(&mut self.messages)
    }

    /// Remove opaque provider state outside `keep`; see
    /// [`retain_provider_state`].
    pub fn retain_provider_state(&mut self, keep: &[&str]) -> Vec<Warning> {
        retain_provider_state(&mut self.messages, keep)
    }

    /// Check that tool calls and tool results are well formed; see
    /// [`validate`].
    ///
    /// # Errors
    ///
    /// Returns every issue found, in message order.
    pub fn validate(&self, mode: ValidationMode) -> Result<(), Vec<HistoryIssue>> {
        validate(&self.messages, mode)
    }

    /// Start a request for `model` with this history's messages.
    ///
    /// The request has no system prompt; add it with
    /// [`ChatRequest::system`], as on every request. The same as
    /// `ChatRequest::new(model).extend_messages(history.messages)`.
    #[must_use]
    pub fn into_request(self, model: impl Into<String>) -> ChatRequest {
        ChatRequest::new(model).extend_messages(self.messages)
    }
}

/// Compares the conversation, not where it was loaded from.
impl PartialEq for History {
    fn eq(&self, other: &Self) -> bool {
        self.messages == other.messages
    }
}

impl Serialize for History {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Raw<'a> {
            format_version: u32,
            messages: &'a [Message],
        }
        Raw {
            format_version: HISTORY_FORMAT_VERSION,
            messages: &self.messages,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for History {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            format_version: u32,
            // Not a field: named only to refuse it (see below).
            #[serde(default)]
            system: Option<serde::de::IgnoredAny>,
            messages: Vec<Message>,
        }
        let raw = Raw::deserialize(deserializer)?;
        if raw.format_version == 0 || raw.format_version > HISTORY_FORMAT_VERSION {
            return Err(serde::de::Error::custom(format!(
                "unsupported history format version {} (this build supports up to {HISTORY_FORMAT_VERSION})",
                raw.format_version
            )));
        }
        // An unknown field is otherwise ignored on load; a system prompt
        // dropped that way would change what the next request sends.
        if raw.system.is_some() {
            return Err(serde::de::Error::custom(
                "a history holds no `system` field: store what rebuilds the system prompt next to the history and set it on every request",
            ));
        }
        Ok(Self {
            messages: raw.messages,
            loaded_format_version: Some(raw.format_version),
        })
    }
}

/// The envelope of a [`ChatResponse`](crate::ChatResponse): everything but
/// its content and provider data, from
/// [`ChatResponse::envelope`](crate::ChatResponse::envelope).
///
/// Looking for turn metadata (usage, finish reason, model, request id) to
/// store beside each message? This is it.
///
/// Adapters never need it to replay a conversation; applications keep it
/// per turn for cost accounting, support, and audit, stored under
/// [`TURN_ENVELOPE_KEY`] (see [The turn envelope](self#the-turn-envelope)).
/// Absent fields are not serialized. An envelope written by a newer
/// version loads, with unknown warning values read as their fallbacks and
/// unknown fields ignored; that tolerance is read-only and lossy.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TurnEnvelope {
    /// Provider-assigned response identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Provider request identifier (from transport headers), for support.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// Model that produced the response, as reported by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Identity of the provider that served the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<Cow<'static, str>>,
    /// Token usage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Normalized reason the model stopped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    /// The provider's raw finish reason, before normalization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason_raw: Option<String>,
    /// Non-fatal warnings the request produced.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<Warning>,
}

/// Remove opaque provider state that `provider` cannot use: the state under
/// every namespace it does not [read](ProviderIdentity::reads_namespace).
///
/// Pass the provider the conversation continues on, as the application holds
/// it: one built from a model address
/// ([`DynChatProvider`](crate::DynChatProvider)), behind wrappers, or a
/// [`FallbackProvider`](crate::wrappers::FallbackProvider), which keeps
/// the state of either side. Its namespaces need not match its
/// [name](ProviderIdentity::provider_name): Gemini reads `google-ai`, an
/// OpenAI-compatible preset `openai-compat:<name>`.
///
/// ```
/// # #[cfg(feature = "mock")] {
/// use allama::history::retain_provider_state_for;
/// use allama::mock::MockProvider;
/// use allama::{ContentBlock, DynChatProvider, Message, ProviderData, Reasoning};
/// use serde_json::json;
///
/// // A provider the application built at runtime; the mock reads `mock`.
/// let provider = DynChatProvider::new(MockProvider::empty());
/// let signed = |namespace: &str| {
///     ContentBlock::from(Reasoning::new("…").with_provider_data(
///         ProviderData::new().with_raw(namespace, "signature", json!("sig")),
///     ))
/// };
/// let mut messages = vec![
///     Message::user("Hi"),
///     Message::assistant_blocks(vec![signed("mock"), signed("anthropic"), ContentBlock::text("Hello")]),
/// ];
///
/// let warnings = retain_provider_state_for(&mut messages, &provider);
/// assert_eq!(warnings.len(), 1); // the `anthropic` reasoning
/// assert_eq!(messages[1].as_assistant().unwrap().content.len(), 2);
/// # }
/// ```
///
/// Otherwise the same as [`retain_provider_state`], which takes the
/// namespaces to keep instead of a provider.
pub fn retain_provider_state_for<P>(messages: &mut [Message], provider: &P) -> Vec<Warning>
where
    P: ProviderIdentity + ?Sized,
{
    retain_where(messages, &|namespace| provider.reads_namespace(namespace))
}

/// Remove opaque provider state that only providers outside `keep` can use.
///
/// `keep` lists the provider-data namespaces of the provider(s) the
/// conversation continues on, e.g. `&[allama_anthropic::NAMESPACE]`. With
/// the provider at hand, [`retain_provider_state_for`] asks it instead,
/// which also works for a provider whose namespaces are known only at
/// runtime. The `"allama"` namespace is always kept. Portable content is
/// never removed:
///
/// - provider data in other namespaces is removed from every message, block,
///   and part;
/// - reasoning blocks whose provider data was entirely foreign are removed
///   (without their opaque state no provider can replay them); reasoning
///   with no provider data is kept on purpose: it is portable text, which
///   endpoints that take reasoning back as text replay, and an adapter
///   that cannot take it drops it for that request only, with a coded
///   `dropped.reasoning_replay` warning (show it once with
///   [`SeenWarnings`](crate::SeenWarnings) rather than strip the
///   reasoning);
/// - `Other` content and provider tool calls and results whose origin is
///   outside `keep` are removed.
///
/// Messages are never removed, so warning indices stay the caller's, and a
/// message's [application data](self#application-data), which is never
/// touched, stays with it. A message left without content is
/// [empty](Message::is_empty): adapters skip it without a warning, and an
/// application can leave it out of a transcript.
///
/// Returns one `Dropped` warning per removed block or stripped state. Call
/// it once when a conversation moves to another provider for good; adapters
/// would otherwise drop the same state, with a warning, on every request.
/// On failover, do nothing: each adapter drops the state it cannot use for
/// that request only, and the history stays ready for the provider that
/// produced it.
pub fn retain_provider_state(messages: &mut [Message], keep: &[&str]) -> Vec<Warning> {
    retain_where(messages, &|namespace| keep.contains(&namespace))
}

/// Remove the state under every namespace but `allama` and those `reads`
/// accepts.
fn retain_where(messages: &mut [Message], reads: &dyn Fn(&str) -> bool) -> Vec<Warning> {
    let kept = |namespace: &str| namespace == ALLAMA_NAMESPACE || reads(namespace);
    let mut warnings = Vec::new();
    let mut warn = |message: usize, block: Option<usize>, text: String| {
        warnings.push(Warning::new(
            WarningKind::Dropped,
            Subject::Content { message, block },
            text,
        ));
    };
    // Strip foreign namespaces; report which ones were removed.
    let strip = |data: &mut ProviderData| -> Vec<String> {
        let foreign: Vec<String> = data
            .namespaces()
            .map(|(namespace, _)| namespace)
            .filter(|namespace| !kept(namespace))
            .map(str::to_owned)
            .collect();
        data.retain_namespaces(kept);
        foreign
    };
    let foreign_origin = |origin: &Option<String>| origin.as_deref().is_some_and(|o| !kept(o));

    for (index, message) in messages.iter_mut().enumerate() {
        let message_data = match message {
            Message::User(m) => &mut m.provider_data,
            Message::Assistant(m) => &mut m.provider_data,
            Message::Tool(m) => &mut m.provider_data,
        };
        let removed = strip(message_data);
        if !removed.is_empty() {
            warn(
                index,
                None,
                format!(
                    "message state from {} belongs to a provider that is not kept; it was dropped",
                    describe_origins(&removed)
                ),
            );
        }

        match message {
            Message::User(m) => {
                if let UserContent::Parts(parts) = &mut m.content {
                    retain_parts(parts, index, &strip, &foreign_origin, &mut warn);
                }
            }
            Message::Tool(m) => {
                if let ToolResultContent::Parts(parts) = &mut m.content {
                    retain_parts(parts, index, &strip, &foreign_origin, &mut warn);
                }
            }
            Message::Assistant(m) => {
                let mut position = 0;
                m.content.retain_mut(|block| {
                    let block_index = position;
                    position += 1;
                    match block {
                        ContentBlock::Reasoning(reasoning) => {
                            let had_state = !reasoning.provider_data.is_empty();
                            let removed = strip(&mut reasoning.provider_data);
                            if had_state && reasoning.provider_data.is_empty() {
                                warn(
                                    index,
                                    Some(block_index),
                                    format!("reasoning from {} belongs to a provider that is not kept; it was dropped", describe_origins(&removed)),
                                );
                                return false;
                            }
                            true
                        }
                        ContentBlock::ProviderToolCall(_) | ContentBlock::ProviderToolResult(_)
                            if block.origin().is_some_and(|origin| !kept(origin)) =>
                        {
                            warn(
                                index,
                                Some(block_index),
                                format!(
                                    "`{}` content from `{}` belongs to a provider that is not kept; it was dropped",
                                    block.type_name(),
                                    block.origin().unwrap_or_default()
                                ),
                            );
                            false
                        }
                        ContentBlock::ProviderToolCall(call) => {
                            report(
                                index,
                                block_index,
                                strip(&mut call.provider_data),
                                &mut warn,
                            );
                            true
                        }
                        ContentBlock::ProviderToolResult(result) => {
                            report(
                                index,
                                block_index,
                                strip(&mut result.provider_data),
                                &mut warn,
                            );
                            true
                        }
                        ContentBlock::Other(other) if foreign_origin(&other.origin) => {
                            warn(
                                index,
                                Some(block_index),
                                format!(
                                    "`{}` content from `{}` belongs to a provider that is not kept; it was dropped",
                                    other.type_name,
                                    other.origin.as_deref().unwrap_or_default()
                                ),
                            );
                            false
                        }
                        ContentBlock::Text(text) => {
                            report(
                                index,
                                block_index,
                                strip(&mut text.provider_data),
                                &mut warn,
                            );
                            true
                        }
                        ContentBlock::Refusal(refusal) => {
                            report(
                                index,
                                block_index,
                                strip(&mut refusal.provider_data),
                                &mut warn,
                            );
                            true
                        }
                        ContentBlock::Image(image) => {
                            report(
                                index,
                                block_index,
                                strip(&mut image.provider_data),
                                &mut warn,
                            );
                            true
                        }
                        ContentBlock::ToolCall(call) => {
                            report(
                                index,
                                block_index,
                                strip(&mut call.provider_data),
                                &mut warn,
                            );
                            true
                        }
                        ContentBlock::Other(other) => {
                            report(
                                index,
                                block_index,
                                strip(&mut other.provider_data),
                                &mut warn,
                            );
                            true
                        }
                    }
                });
            }
        }
    }

    warnings
}

fn retain_parts(
    parts: &mut Vec<ContentPart>,
    message: usize,
    strip: &impl Fn(&mut ProviderData) -> Vec<String>,
    foreign_origin: &impl Fn(&Option<String>) -> bool,
    warn: &mut impl FnMut(usize, Option<usize>, String),
) {
    let mut position = 0;
    parts.retain_mut(|part| {
        let block = position;
        position += 1;
        match part {
            ContentPart::Other(other) if foreign_origin(&other.origin) => {
                warn(
                    message,
                    Some(block),
                    format!(
                        "`{}` content from `{}` belongs to a provider that is not kept; it was dropped",
                        other.type_name,
                        other.origin.as_deref().unwrap_or_default()
                    ),
                );
                false
            }
            ContentPart::Text(text) => {
                report(message, block, strip(&mut text.provider_data), warn);
                true
            }
            ContentPart::Image(image) => {
                report(message, block, strip(&mut image.provider_data), warn);
                true
            }
            ContentPart::File(file) => {
                report(message, block, strip(&mut file.provider_data), warn);
                true
            }
            ContentPart::Other(other) => {
                report(message, block, strip(&mut other.provider_data), warn);
                true
            }
        }
    });
}

fn report(
    message: usize,
    block: usize,
    removed: Vec<String>,
    warn: &mut impl FnMut(usize, Option<usize>, String),
) {
    if !removed.is_empty() {
        warn(
            message,
            Some(block),
            format!(
                "state from {} belongs to a provider that is not kept; it was dropped",
                describe_origins(&removed)
            ),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OtherContent, Reasoning, Text, ToolCall};
    use serde_json::json;

    fn data(namespace: &str) -> ProviderData {
        ProviderData::new().with_raw(namespace, "signature", json!("sig"))
    }

    /// Reads `google-ai`, under a name that differs, like Gemini.
    struct Gemini;

    impl ProviderIdentity for Gemini {
        fn provider_name(&self) -> &str {
            "gemini"
        }

        fn reads_namespace(&self, namespace: &str) -> bool {
            namespace == "google-ai"
        }
    }

    /// Reads `anthropic`.
    struct Anthropic;

    impl ProviderIdentity for Anthropic {
        fn provider_name(&self) -> &str {
            "anthropic"
        }

        fn reads_namespace(&self, namespace: &str) -> bool {
            namespace == "anthropic"
        }
    }

    fn mixed_conversation() -> Vec<Message> {
        vec![
            Message::user("Hi"),
            Message::Assistant(crate::AssistantMessage::new(vec![
                Reasoning::new("a")
                    .with_provider_data(data("google-ai"))
                    .into(),
                Reasoning::new("b")
                    .with_provider_data(data("anthropic"))
                    .into(),
                Text::new("Hello")
                    .with_provider_data(data("gemini").with_raw(ALLAMA_NAMESPACE, "note", json!(1)))
                    .into(),
            ])),
        ]
    }

    #[test]
    fn retain_for_keeps_the_namespaces_the_provider_reads() {
        let mut messages = mixed_conversation();
        let warnings = retain_provider_state_for(&mut messages, &Gemini);

        // The `anthropic` reasoning and the `gemini` state go: Gemini reads
        // `google-ai`, not its own name.
        assert_eq!(warnings.len(), 2, "{warnings:?}");
        let blocks = &messages[1].as_assistant().unwrap().content;
        assert_eq!(blocks.len(), 2);
        let ContentBlock::Text(text) = &blocks[1] else {
            panic!("{blocks:?}")
        };
        let namespaces: Vec<_> = text.provider_data.namespaces().map(|(n, _)| n).collect();
        assert_eq!(namespaces, [ALLAMA_NAMESPACE]);

        let mut by_namespace = mixed_conversation();
        retain_provider_state(&mut by_namespace, &["google-ai"]);
        assert_eq!(messages, by_namespace);
    }

    #[test]
    fn retain_for_a_fallback_keeps_what_either_side_reads() {
        let provider = crate::wrappers::FallbackProvider::new(Anthropic, Gemini);
        let mut history = History::new(mixed_conversation());
        let warnings = history.retain_provider_state_for(&provider);

        assert_eq!(
            warnings.len(),
            1,
            "only the `gemini` state goes: {warnings:?}"
        );
        assert_eq!(history.messages[1].as_assistant().unwrap().content.len(), 3);
    }

    #[test]
    fn retain_for_works_through_type_erasure() {
        let boxed: Box<dyn ProviderIdentity> = Box::new(Gemini);
        let mut messages = mixed_conversation();
        let warnings = retain_provider_state_for(&mut messages, boxed.as_ref());
        assert_eq!(warnings.len(), 2);
    }

    #[test]
    fn provider_tool_blocks_follow_their_origin() {
        let call = |origin: &str| {
            ContentBlock::from(
                crate::ProviderToolCall::new("c", "web_search", "{}").with_origin(origin),
            )
        };
        let result = |origin: &str| {
            ContentBlock::from(
                crate::ProviderToolResult::new("c", json!({"status": "completed"}))
                    .with_origin(origin),
            )
        };
        let mut messages = vec![
            Message::user("Search"),
            Message::Assistant(crate::AssistantMessage::new(vec![
                call("openai"),
                result("openai"),
                call("anthropic"),
                result("anthropic"),
                ContentBlock::text("Done."),
            ])),
        ];
        let warnings = retain_provider_state(&mut messages, &["anthropic"]);
        assert_eq!(warnings.len(), 2);
        let kept = &messages[1].as_assistant().unwrap().content;
        assert_eq!(kept.len(), 3);
        assert!(kept.iter().take(2).all(|b| b.origin() == Some("anthropic")));
    }

    #[test]
    fn retain_provider_state_removes_only_foreign_opaque_state() {
        let mut history = History::new(vec![
            Message::user("Weather in Paris?"),
            Message::assistant_blocks(vec![
                ContentBlock::Reasoning(
                    Reasoning::new("thinking").with_provider_data(data("anthropic")),
                ),
                ContentBlock::Reasoning(Reasoning::new("portable reasoning")),
                ContentBlock::Other(
                    OtherContent::new("server_tool_use", Default::default())
                        .with_origin("anthropic"),
                ),
                ContentBlock::Text(Text::new("Checking.").with_provider_data(data("anthropic"))),
                ContentBlock::ToolCall(
                    ToolCall::new("call_1", "get_weather", r#"{"city":"Paris"}"#)
                        .with_provider_data(data("google-ai").with_raw("allama", "note", json!(1))),
                ),
            ]),
            Message::tool_result("call_1", "get_weather", "Sunny"),
            Message::assistant_blocks(vec![ContentBlock::Reasoning(
                Reasoning::new("").with_provider_data(data("anthropic")),
            )]),
        ]);

        let warnings = history.retain_provider_state(&["google-ai"]);

        assert_eq!(history.messages.len(), 4, "no message is removed");
        assert!(
            history.messages[3].is_empty(),
            "the opaque-only assistant turn is left empty"
        );
        let Message::Assistant(turn) = &history.messages[1] else {
            panic!("expected assistant turn");
        };
        assert_eq!(turn.content.len(), 3);
        assert!(
            matches!(&turn.content[0], ContentBlock::Reasoning(r) if r.text == "portable reasoning")
        );
        assert!(matches!(&turn.content[1], ContentBlock::Text(t) if t.provider_data.is_empty()));
        let ContentBlock::ToolCall(call) = &turn.content[2] else {
            panic!("expected tool call");
        };
        assert!(
            call.provider_data.namespace("google-ai").is_some(),
            "kept namespace survives"
        );
        assert!(
            call.provider_data.namespace("allama").is_some(),
            "allama namespace survives"
        );

        assert!(warnings.iter().all(|w| w.kind == WarningKind::Dropped));
        assert_eq!(warnings.len(), 4, "{warnings:?}");
        assert_eq!(
            warnings[0].subject,
            Subject::Content {
                message: 1,
                block: Some(0)
            }
        );

        // Idempotent: a second pass finds nothing to remove.
        assert!(history.retain_provider_state(&["google-ai"]).is_empty());
    }

    #[test]
    fn kept_other_content_loses_only_foreign_provider_data() {
        let other = || {
            OtherContent::new("search_result", Default::default())
                .with_origin("anthropic")
                .with_provider_data(
                    data("anthropic")
                        .with_raw("google-ai", "thought_signature", json!("g"))
                        .with_raw("allama", "note", json!(1)),
                )
        };
        let mut messages = vec![
            Message::user(vec![ContentPart::Other(other())]),
            Message::assistant_blocks(vec![ContentBlock::Other(other())]),
        ];

        let warnings = retain_provider_state(&mut messages, &["anthropic"]);

        let expected = data("anthropic").with_raw("allama", "note", json!(1));
        let Message::User(user) = &messages[0] else {
            panic!("expected user message");
        };
        let UserContent::Parts(parts) = &user.content else {
            panic!("expected parts");
        };
        let ContentPart::Other(part) = &parts[0] else {
            panic!("expected other part");
        };
        assert_eq!(part.provider_data, expected);
        assert_eq!(
            messages[1].as_assistant().unwrap().content[0].provider_data(),
            &expected
        );
        let subjects: Vec<_> = warnings.iter().map(|w| w.subject.clone()).collect();
        assert_eq!(
            subjects,
            [
                Subject::Content {
                    message: 0,
                    block: Some(0)
                },
                Subject::Content {
                    message: 1,
                    block: Some(0)
                },
            ]
        );
        assert!(warnings.iter().all(|w| w.message.contains("google-ai")));
        assert!(retain_provider_state(&mut messages, &["anthropic"]).is_empty());
    }

    #[test]
    fn retain_provider_state_leaves_app_data_untouched() {
        let app_data = |id: &str| {
            let mut map = crate::ExtraMap::new();
            map.insert("myapp.id".into(), json!(id));
            map.insert("anthropic".into(), json!({"looks": "like a namespace"}));
            map
        };
        let mut user = crate::UserMessage::new(vec![ContentPart::text("Hi")]);
        user.provider_data = data("anthropic");
        user.app_data = app_data("u1");
        let mut assistant = crate::AssistantMessage::new(vec![ContentBlock::Text(
            Text::new("Hello").with_provider_data(data("anthropic")),
        )]);
        assistant.provider_data = data("anthropic");
        assistant.app_data = app_data("a1");
        let mut tool = crate::ToolMessage::new("c1", "f", "ok");
        tool.provider_data = data("anthropic");
        tool.app_data = app_data("t1");
        let mut messages = vec![
            Message::User(user),
            Message::Assistant(assistant),
            Message::Tool(tool),
        ];

        let warnings = retain_provider_state(&mut messages, &["openai"]);

        assert_eq!(warnings.len(), 4, "{warnings:?}");
        for (message, id) in messages.iter().zip(["u1", "a1", "t1"]) {
            assert!(message.provider_data().is_empty());
            assert_eq!(message.app_data(), &app_data(id));
        }
    }

    #[test]
    fn an_emptied_message_keeps_its_app_data_and_turn_envelope() {
        // The reply held only another provider's reasoning, and the
        // application stored its turn envelope on it.
        let mut reply = Message::assistant_blocks(vec![ContentBlock::Reasoning(
            Reasoning::new("").with_provider_data(data("anthropic")),
        )]);
        let envelope = TurnEnvelope {
            id: Some("msg_01".into()),
            usage: Some(Usage::new().output_tokens(42)),
            ..TurnEnvelope::default()
        };
        reply.app_data_insert(TURN_ENVELOPE_KEY, &envelope).unwrap();
        reply.app_data_insert("myapp.id", &"a1").unwrap();
        let mut history = History::new(vec![
            Message::user("Hi"),
            reply,
            Message::user(vec![ContentPart::Other(
                OtherContent::new("custom_tool_call_output", Default::default())
                    .with_origin("openai"),
            )]),
            Message::user("Still there?"),
        ]);
        let before = history.messages.clone();

        let warnings = history.retain_provider_state(&["google-ai"]);

        assert_eq!(
            history.messages.len(),
            before.len(),
            "no message is removed"
        );
        for (after, before) in history.messages.iter().zip(&before) {
            assert_eq!(after.app_data(), before.app_data());
        }
        assert!(history.messages[1].is_empty());
        assert!(history.messages[2].is_empty());
        let stored: Option<TurnEnvelope> =
            history.messages[1].app_data_get(TURN_ENVELOPE_KEY).unwrap();
        assert_eq!(stored, Some(envelope));
        // Warnings point at the caller's messages.
        let subjects: Vec<_> = warnings.into_iter().map(|w| w.subject).collect();
        assert_eq!(
            subjects,
            [
                Subject::Content {
                    message: 1,
                    block: Some(0)
                },
                Subject::Content {
                    message: 2,
                    block: Some(0)
                },
            ]
        );
    }

    #[test]
    fn round_trip_and_request_conversion() {
        let request = ChatRequest::new("m")
            .system("sys")
            .user("hi")
            .assistant("hello");
        let history = History::from_request(&request);
        assert_eq!(history.loaded_format_version(), None);
        let value = serde_json::to_value(&history).unwrap();
        assert_eq!(value["format_version"], HISTORY_FORMAT_VERSION);
        assert!(value.get("system").is_none(), "{value}");
        let back: History = serde_json::from_value(value).unwrap();
        assert_eq!(back, history);
        assert_eq!(back.loaded_format_version(), Some(HISTORY_FORMAT_VERSION));
        let rebuilt = back.into_request("m2");
        assert_eq!(rebuilt.messages, request.messages);
        assert!(rebuilt.system.is_empty());
    }

    #[test]
    fn a_stored_system_key_fails_to_load_and_names_it() {
        let err = serde_json::from_value::<History>(json!({
            "format_version": 1,
            "system": [{"content": "Be brief."}],
            "messages": []
        }))
        .unwrap_err();
        assert!(err.to_string().contains("no `system` field"), "{err}");
    }

    #[test]
    fn newer_versions_are_rejected() {
        let err = serde_json::from_value::<History>(json!({"format_version": 99, "messages": []}))
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("unsupported history format version 99")
        );
    }

    /// A history loaded from an older version is written back at the
    /// current one, never at the version it was read from.
    #[test]
    fn saving_writes_the_current_version() {
        let mut history: History =
            serde_json::from_value(json!({"format_version": 1, "messages": []})).unwrap();
        history.loaded_format_version = Some(0);
        history.messages.push(Message::user("hi"));
        let value = serde_json::to_value(&history).unwrap();
        assert_eq!(value["format_version"], HISTORY_FORMAT_VERSION);
        assert_eq!(
            history,
            History::new(vec![Message::user("hi")]),
            "equality ignores the loaded version"
        );
    }

    /// The system prompts a provider received.
    fn sent_prompts(provider: &crate::mock::MockProvider) -> Vec<String> {
        let request = provider.last_request().expect("a request was sent");
        request
            .system
            .into_iter()
            .map(|prompt| prompt.content)
            .collect()
    }

    /// Resuming a stored conversation and setting the prompt, as every
    /// request does, sends the prompt once, however often it is resumed.
    #[tokio::test]
    async fn every_resume_sends_the_system_prompt_once() {
        use crate::ChatProvider;
        use crate::mock::MockProvider;

        const PROMPT: &str = "You are concise.";
        let provider = MockProvider::build(|b| b.text("1").text("2").text("3").text("4"));
        let save = |request: &ChatRequest, reply: &crate::ChatResponse| {
            let mut history = History::from_request(request);
            history.messages.push(reply.to_assistant_message());
            let stored = serde_json::to_string(&history).unwrap();
            assert!(!stored.contains(PROMPT), "{stored}");
            stored
        };

        let first = ChatRequest::new("m").system(PROMPT).user("Hi");
        let mut stored = save(&first, &provider.chat(&first).await.unwrap());
        for turn in ["Go on", "And?", "Thanks"] {
            let loaded: History = serde_json::from_str(&stored).unwrap();
            let request = loaded.into_request("m").system(PROMPT).user(turn);
            let reply = provider.chat(&request).await.unwrap();
            assert_eq!(
                sent_prompts(&provider),
                [PROMPT],
                "resuming before {turn:?}"
            );
            stored = save(&request, &reply);
        }
        assert_eq!(provider.last_request().unwrap().messages.len(), 7);
    }

    #[tokio::test]
    async fn a_reloaded_conversation_with_a_new_prompt_sends_only_the_new_prompt() {
        use crate::ChatProvider;
        use crate::mock::MockProvider;

        let provider = MockProvider::build(|b| b.text("Hello!").text("Sure."));
        let first = ChatRequest::new("m").system("v1 prompt").user("Hi");
        let reply = provider.chat(&first).await.unwrap();

        let mut history = History::from_request(&first);
        history.messages.push(reply.to_assistant_message());
        let stored = serde_json::to_string(&history).unwrap();

        let loaded: History = serde_json::from_str(&stored).unwrap();
        let next = loaded.into_request("m").system("v2 prompt").user("Thanks");
        provider.chat(&next).await.unwrap();

        assert_eq!(sent_prompts(&provider), ["v2 prompt"]);
        assert_eq!(provider.last_request().unwrap().messages.len(), 3);
    }
}
