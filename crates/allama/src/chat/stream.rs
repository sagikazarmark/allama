//! Chat streams: events, collection, transcripts, and commit-point helpers.
//!
//! The types every stream consumer names ([`StreamEvent`],
//! [`ChatStream`], [`ChatStreamExt`], [`CollectedEvent`]) are also
//! exported at the crate root. Other operations that stream (transcription,
//! speech, …) get their own event types beside their own modules; these
//! are the chat ones.
//!
//! # The commit point
//!
//! A stream's commit point is its first event that carries model output
//! ([`StreamEvent::is_output`](crate::StreamEvent::is_output): text,
//! reasoning, a refusal, a tool call, an image), after which the caller may
//! have seen the answer. A stream that ends without output commits at its
//! end.
//!
//! - **Before it**, nothing is exposed, so an error can still be retried,
//!   sent to another provider, or answered with an HTTP status code.
//!   [`RetryingProvider`](crate::wrappers::RetryingProvider) and
//!   [`FallbackProvider`](crate::wrappers::FallbackProvider) hold a stream
//!   back until its commit point, and [`until_committed`] gives an
//!   application the same wait, replaying the held events so nothing is
//!   lost and the time to the first token is unchanged.
//! - **After it**, an error can only end the stream: retry and fallback
//!   pass it through, and a server that already sent `200 OK` reports it as
//!   an event.
//!
//! The commit point also decides what a timeout outside retry and fallback
//! bounds: its `stream_setup` is the wait for the first output (see
//! [attempt timeout vs deadline](crate::wrappers::TimeoutPolicy#attempt-timeout-vs-deadline)).

use std::borrow::Cow;
use std::collections::VecDeque;
use std::fmt;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures_core::Stream;
use serde::{Deserialize, Serialize};

use crate::chat::content::is_valid_json;
pub use crate::chat::transcript::StreamTranscript;
use crate::{
    ChatResponse, Citation, ContentBlock, Error, ExtraMap, FinishReason, Image, MediaSource,
    OtherContent, ProviderData, ProviderToolCall, ProviderToolResult, Reasoning, Refusal, Result,
    StreamErrorKind, Text, ToolCall, Usage, Warning,
};

/// A boxed, type-erased chat stream.
///
/// # A chat stream ends after its first error
///
/// An `Err` item is the last item of a chat stream: nothing follows it, and
/// polling again yields `None`. Consumers rely on it: collecting a stream
/// ([`collect_response`](ChatStreamExt::collect_response)) and
/// [`until_committed`] stop at the first
/// error and treat it as the stream's terminal error.
///
/// The streams allama builds keep the rule however the stream under them
/// behaves: [`Prepared::finish_stream`](crate::adapter::chat::Prepared::finish_stream),
/// [`with_collector`](ChatStreamExt::with_collector) and the extraction
/// stream stop polling what they wrap after its first error. A stream built
/// by hand (an adapter without `finish_stream`, a wrapper, a test double)
/// must honour it too: end after yielding an error, and do not poll what
/// it wraps again.
pub type ChatStream = Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send + 'static>>;

/// How to interpret the `usage` of a [`StreamEvent::ResponseMetadata`] event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum UsageMetadataMode {
    /// Incremental usage, added to prior usage.
    Delta,
    /// Cumulative usage, replacing prior usage.
    #[default]
    Snapshot,
}

fn is_snapshot(mode: &UsageMetadataMode) -> bool {
    *mode == UsageMetadataMode::Snapshot
}

/// Portable type of a streamed content block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum StreamBlockType {
    /// A text block.
    Text,
    /// An image block (complete at start).
    Image,
    /// A reasoning block.
    Reasoning,
    /// A refusal block; its text streams as `RefusalDelta`.
    Refusal,
    /// A client tool call; its arguments stream as `ToolCallDelta`. Start
    /// it with [`StreamEvent::tool_call_start`], which carries the call's
    /// id and name: a tool call without them is dropped as incomplete.
    ToolCall,
    /// A provider-executed tool call; its arguments stream as
    /// `ToolCallDelta`. Start it with
    /// [`StreamEvent::provider_tool_call_start`], which carries its id and
    /// name.
    ProviderToolCall,
    /// A provider tool result (complete at start: `id` is the call id,
    /// `data` holds `output` and `is_error`).
    ProviderToolResult,
    /// A block the portable model does not represent (complete at start).
    Other,
}

impl StreamBlockType {
    /// Every block type, in declaration order (for audits such as the
    /// conformance honesty contract). Grows when block types are added.
    pub const ALL: &'static [Self] = &[
        Self::Text,
        Self::Image,
        Self::Reasoning,
        Self::Refusal,
        Self::ToolCall,
        Self::ProviderToolCall,
        Self::ProviderToolResult,
        Self::Other,
    ];

    /// The name used in protocol error messages (the serialized form).
    fn label(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Reasoning => "reasoning",
            Self::Refusal => "refusal",
            Self::ToolCall => "tool_call",
            Self::ProviderToolCall => "provider_tool_call",
            Self::ProviderToolResult => "provider_tool_result",
            Self::Other => "other",
        }
    }
}

/// One event of a normalized chat stream.
///
/// A stream is `ResponseStart`, then for each block `BlockStart`, deltas and
/// `BlockStop`, then `ResponseMetadata` (possibly several), then
/// `ResponseStop`. `Warning` and `ResponseProviderData` events may appear at
/// any point. Blocks may overlap: a block can stay open while later blocks
/// start and stream (Gemini keeps a text block open until its candidate
/// ends, to attach grounding citations), so route deltas by their `index`,
/// not by the last `BlockStart`. [`StreamCollector`] is lenient about the
/// order (a delta before its `BlockStart` opens the block); adapters keep
/// it anyway, for consumers of the raw events. Wrappers keep this order when they add events: for example
/// [`FallbackProvider`](crate::wrappers::FallbackProvider) reports its
/// failed attempts in a `ResponseMetadata` event right after the fallback's
/// `ResponseStart`.
///
/// The enum and its struct variants are `#[non_exhaustive]`, so new events and
/// new fields are not breaking changes. Build events with the constructors
/// ([`StreamEvent::text_delta`], [`StreamEvent::metadata`], ...) or, for
/// `ResponseMetadata`, a [`MetadataBuilder`];
/// set further fields through a mutable match, and match variants with `..`:
///
/// ```
/// use allama::chat::stream::MetadataBuilder;
/// use allama::{FinishReason, StreamBlockType, StreamEvent};
///
/// let metadata = MetadataBuilder::new()
///     .with_finish_reason(FinishReason::Stop)
///     .build();
/// assert!(matches!(metadata, StreamEvent::ResponseMetadata { .. }));
///
/// let mut start = StreamEvent::block_start(0, StreamBlockType::Other);
/// if let StreamEvent::BlockStart { type_name, .. } = &mut start {
///     *type_name = Some("vendor_block".into());
/// }
///
/// match StreamEvent::text_delta(0, "Hello") {
///     StreamEvent::TextDelta { text, .. } => assert_eq!(text, "Hello"),
///     _ => unreachable!(),
/// }
/// ```
///
/// A tool call starts as a `BlockStart` with the `ToolCall` block type,
/// which names the tool before its arguments stream in:
///
/// ```
/// use allama::{StreamBlockType, StreamEvent};
///
/// fn announce(event: &StreamEvent) {
///     if let StreamEvent::BlockStart {
///         block_type: StreamBlockType::ToolCall,
///         name: Some(name),
///         ..
///     } = event
///     {
///         println!("calling {name}…");
///     }
/// }
/// # announce(&StreamEvent::block_start(0, StreamBlockType::ToolCall));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
#[non_exhaustive]
pub enum StreamEvent {
    /// The response started.
    #[non_exhaustive]
    ResponseStart {
        /// Response identifier, when known.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Model identifier, when known.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
    },
    /// A content block started. A tool call's start names the tool (see
    /// the [enum docs](Self)).
    #[doc(alias = "ToolCallStart")]
    #[non_exhaustive]
    BlockStart {
        /// Block index, stable across the block's events.
        index: usize,
        /// Portable block type.
        block_type: StreamBlockType,
        /// Tool call id (tool-call blocks).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Tool name (tool-call blocks).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// Wire type name (`Other` blocks).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        type_name: Option<String>,
        /// Block payload for blocks that are complete at start (image, other).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<ExtraMap>,
        /// Origin namespace (`Other` blocks).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        origin: Option<String>,
        /// Provider data known at block start.
        #[serde(default, skip_serializing_if = "ProviderData::is_empty")]
        provider_data: ProviderData,
    },
    /// Incremental text.
    #[non_exhaustive]
    TextDelta {
        /// Block index.
        index: usize,
        /// Text chunk.
        text: String,
    },
    /// Incremental reasoning text.
    #[non_exhaustive]
    ReasoningDelta {
        /// Block index.
        index: usize,
        /// Reasoning chunk.
        text: String,
    },
    /// Incremental refusal text: the model declining to answer (see
    /// [`Refusal`]).
    ///
    /// It is never a `TextDelta`, so a loop that prints only text deltas
    /// never shows a refusal as the answer. It is model output, so it is a
    /// commit point ([`is_output`](Self::is_output)).
    #[non_exhaustive]
    RefusalDelta {
        /// Block index.
        index: usize,
        /// Refusal chunk.
        text: String,
    },
    /// Incremental tool-call arguments (a JSON fragment).
    #[non_exhaustive]
    ToolCallDelta {
        /// Block index.
        index: usize,
        /// Arguments chunk.
        arguments: String,
    },
    /// A citation for a text block, appended to the block's
    /// [`citations`](crate::Text::citations).
    #[non_exhaustive]
    Citation {
        /// Block index.
        index: usize,
        /// The citation.
        citation: Citation,
    },
    /// Provider data for a block (e.g. a signature), merged into the block.
    #[non_exhaustive]
    BlockProviderData {
        /// Block index.
        index: usize,
        /// Data to merge (per key, later values win).
        provider_data: ProviderData,
    },
    /// Provider data for the response as a whole, merged into
    /// [`ChatResponse::provider_data`](crate::ChatResponse::provider_data)
    /// (per key, later values win).
    ///
    /// The response-level counterpart of `BlockProviderData`, for data that
    /// arrives on its own rather than with `ResponseMetadata`: for example
    /// vendor stream frames the adapter does not understand, surfaced
    /// verbatim under the provider's namespace.
    #[non_exhaustive]
    ResponseProviderData {
        /// Data to merge (per key, later values win).
        provider_data: ProviderData,
    },
    /// A provider-specific incremental update that the portable events do
    /// not represent: hosted-tool progress, delta types an adapter does not
    /// map, and so on.
    ///
    /// Observation only: collectors do not fold it into the response, so it
    /// never changes what [`collect_response`](ChatStreamExt::collect_response)
    /// returns. Adapters emit it only when no portable event fits; it is not
    /// raw wire passthrough.
    #[non_exhaustive]
    ProviderDelta {
        /// Index of the block the update belongs to.
        index: usize,
        /// Provider event or delta type, e.g. `"web_search_call.searching"`.
        type_name: String,
        /// Payload, verbatim.
        #[serde(default, skip_serializing_if = "ExtraMap::is_empty")]
        data: ExtraMap,
        /// Namespace of the provider that produced the update.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        origin: Option<String>,
    },
    /// A content block finished.
    #[non_exhaustive]
    BlockStop {
        /// Block index.
        index: usize,
    },
    /// A non-fatal warning.
    #[non_exhaustive]
    Warning {
        /// The warning.
        warning: Warning,
    },
    /// Response-level metadata. May arrive several times; later values win.
    #[non_exhaustive]
    ResponseMetadata {
        /// Normalized finish reason.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        finish_reason: Option<FinishReason>,
        /// Raw provider finish reason.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        finish_reason_raw: Option<String>,
        /// Usage, interpreted per `usage_mode`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
        /// How to interpret `usage`.
        #[serde(default, skip_serializing_if = "is_snapshot")]
        usage_mode: UsageMetadataMode,
        /// Response identifier.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Model identifier.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        /// Provider request identifier.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        request_id: Option<String>,
        /// Provider that served the request.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider: Option<Cow<'static, str>>,
        /// Response-level provider data (merged).
        #[serde(default, skip_serializing_if = "ProviderData::is_empty")]
        provider_data: ProviderData,
    },
    /// The response ended.
    ///
    /// A unit variant, so it stays constructible and matchable as
    /// `StreamEvent::ResponseStop`. Response-level data belongs in
    /// `ResponseMetadata`.
    ResponseStop,
}

impl StreamEvent {
    /// A `ResponseStart` event.
    #[must_use]
    pub fn response_start(id: Option<String>, model: Option<String>) -> Self {
        Self::ResponseStart { id, model }
    }

    /// A `TextDelta` event.
    #[must_use]
    pub fn text_delta(index: usize, text: impl Into<String>) -> Self {
        Self::TextDelta {
            index,
            text: text.into(),
        }
    }

    /// A `ReasoningDelta` event.
    #[must_use]
    pub fn reasoning_delta(index: usize, text: impl Into<String>) -> Self {
        Self::ReasoningDelta {
            index,
            text: text.into(),
        }
    }

    /// A `RefusalDelta` event.
    #[must_use]
    pub fn refusal_delta(index: usize, text: impl Into<String>) -> Self {
        Self::RefusalDelta {
            index,
            text: text.into(),
        }
    }

    /// A `ToolCallDelta` event carrying a JSON fragment of the arguments.
    #[must_use]
    pub fn tool_call_delta(index: usize, arguments: impl Into<String>) -> Self {
        Self::ToolCallDelta {
            index,
            arguments: arguments.into(),
        }
    }

    /// A `Citation` event for the text block at `index`.
    #[must_use]
    pub fn citation(index: usize, citation: Citation) -> Self {
        Self::Citation { index, citation }
    }

    /// A `BlockProviderData` event, merged into the block at `index`.
    #[must_use]
    pub fn block_provider_data(index: usize, provider_data: ProviderData) -> Self {
        Self::BlockProviderData {
            index,
            provider_data,
        }
    }

    /// A `ResponseProviderData` event, merged into the response's provider
    /// data.
    #[must_use]
    pub fn response_provider_data(provider_data: ProviderData) -> Self {
        Self::ResponseProviderData { provider_data }
    }

    /// A `BlockStop` event.
    #[must_use]
    pub fn block_stop(index: usize) -> Self {
        Self::BlockStop { index }
    }

    /// A `Warning` event.
    #[must_use]
    pub fn warning(warning: Warning) -> Self {
        Self::Warning { warning }
    }

    /// A `ResponseMetadata` event with every field empty.
    #[must_use]
    pub fn metadata() -> Self {
        Self::ResponseMetadata {
            finish_reason: None,
            finish_reason_raw: None,
            usage: None,
            usage_mode: UsageMetadataMode::Snapshot,
            id: None,
            model: None,
            request_id: None,
            provider: None,
            provider_data: ProviderData::new(),
        }
    }

    /// A `BlockStart` event for a block of `block_type` with no payload:
    /// text, reasoning, or a refusal.
    ///
    /// A tool call needs its id and name, so start one with
    /// [`tool_call_start`](Self::tool_call_start) or
    /// [`provider_tool_call_start`](Self::provider_tool_call_start); a
    /// `ToolCall` started here has neither and is dropped as incomplete. A
    /// block that is complete at start needs its payload:
    /// [`image_start`](Self::image_start),
    /// [`provider_tool_result_start`](Self::provider_tool_result_start),
    /// [`other_start`](Self::other_start).
    #[must_use]
    pub fn block_start(index: usize, block_type: StreamBlockType) -> Self {
        Self::BlockStart {
            index,
            block_type,
            id: None,
            name: None,
            type_name: None,
            data: None,
            origin: None,
            provider_data: ProviderData::new(),
        }
    }

    /// A `BlockStart` event for a provider-executed tool call; its arguments
    /// follow as `ToolCallDelta` events.
    #[must_use]
    pub fn provider_tool_call_start(
        index: usize,
        id: impl Into<String>,
        name: impl Into<String>,
        origin: Option<String>,
    ) -> Self {
        Self::BlockStart {
            index,
            block_type: StreamBlockType::ProviderToolCall,
            id: Some(id.into()),
            name: Some(name.into()),
            type_name: None,
            data: None,
            origin,
            provider_data: ProviderData::new(),
        }
    }

    /// The `BlockStart` event of a provider tool result, which is complete at
    /// start.
    #[must_use]
    pub fn provider_tool_result_start(index: usize, result: ProviderToolResult) -> Self {
        let mut data = ExtraMap::new();
        data.insert("output".into(), result.output);
        if result.is_error {
            data.insert("is_error".into(), true.into());
        }
        Self::BlockStart {
            index,
            block_type: StreamBlockType::ProviderToolResult,
            id: Some(result.call_id),
            name: None,
            type_name: None,
            data: Some(data),
            origin: result.origin,
            provider_data: result.provider_data,
        }
    }

    /// The `BlockStart` event of an image block, which is complete at start.
    #[must_use]
    pub fn image_start(index: usize, image: Image) -> Self {
        Self::BlockStart {
            index,
            block_type: StreamBlockType::Image,
            id: None,
            name: None,
            type_name: None,
            data: Some(image_data(&image.source, image.media_type.as_deref())),
            origin: None,
            provider_data: image.provider_data,
        }
    }

    /// The `BlockStart` event of a block the portable model does not
    /// represent, which is complete at start.
    #[must_use]
    pub fn other_start(index: usize, other: OtherContent) -> Self {
        Self::BlockStart {
            index,
            block_type: StreamBlockType::Other,
            id: None,
            name: None,
            type_name: Some(other.type_name),
            data: Some(other.data),
            origin: other.origin,
            provider_data: other.provider_data,
        }
    }

    /// A `ProviderDelta` event.
    #[must_use]
    pub fn provider_delta(
        index: usize,
        type_name: impl Into<String>,
        data: ExtraMap,
        origin: Option<String>,
    ) -> Self {
        Self::ProviderDelta {
            index,
            type_name: type_name.into(),
            data,
            origin,
        }
    }

    /// A `BlockStart` event for a tool call.
    #[must_use]
    pub fn tool_call_start(index: usize, id: impl Into<String>, name: impl Into<String>) -> Self {
        Self::BlockStart {
            index,
            block_type: StreamBlockType::ToolCall,
            id: Some(id.into()),
            name: Some(name.into()),
            type_name: None,
            data: None,
            origin: None,
            provider_data: ProviderData::new(),
        }
    }

    /// Returns `true` for events that carry model output. Fallback and retry
    /// treat the first such event as the commit point.
    ///
    /// A refusal is output: once the caller may have seen the model decline,
    /// a later error must not retry the call or send it to another provider.
    #[must_use]
    pub fn is_output(&self) -> bool {
        match self {
            Self::TextDelta { .. }
            | Self::ReasoningDelta { .. }
            | Self::RefusalDelta { .. }
            | Self::ToolCallDelta { .. } => true,
            Self::BlockStart { block_type, .. } => {
                matches!(
                    block_type,
                    StreamBlockType::Image
                        | StreamBlockType::Refusal
                        | StreamBlockType::ProviderToolResult
                        | StreamBlockType::Other
                )
            }
            _ => false,
        }
    }

    /// Returns `true` for events that address a content block by index.
    fn is_block_event(&self) -> bool {
        matches!(
            self,
            Self::BlockStart { .. }
                | Self::TextDelta { .. }
                | Self::ReasoningDelta { .. }
                | Self::RefusalDelta { .. }
                | Self::ToolCallDelta { .. }
                | Self::Citation { .. }
                | Self::BlockProviderData { .. }
                | Self::ProviderDelta { .. }
                | Self::BlockStop { .. }
        )
    }

    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::ResponseStart { .. } => "response_start",
            Self::BlockStart { .. } => "block_start",
            Self::TextDelta { .. } => "text_delta",
            Self::ReasoningDelta { .. } => "reasoning_delta",
            Self::RefusalDelta { .. } => "refusal_delta",
            Self::ToolCallDelta { .. } => "tool_call_delta",
            Self::Citation { .. } => "citation",
            Self::BlockProviderData { .. } => "block_provider_data",
            Self::ResponseProviderData { .. } => "response_provider_data",
            Self::ProviderDelta { .. } => "provider_delta",
            Self::BlockStop { .. } => "block_stop",
            Self::Warning { .. } => "warning",
            Self::ResponseMetadata { .. } => "response_metadata",
            Self::ResponseStop => "response_stop",
        }
    }
}

/// Builds a [`StreamEvent::ResponseMetadata`] event field by field.
///
/// An alternative to setting the fields of [`StreamEvent::metadata`]
/// through a mutable match. Unset fields stay empty, and `usage_mode`
/// defaults to [`UsageMetadataMode::Snapshot`].
///
/// The setters follow the rule of the [`Error`] builders:
/// **`with_*` takes the value**, and each field a wire response may lack
/// also has a **`with_*_opt`** twin that takes an `Option` and keeps the
/// current value on `None`. Text twins take an `Option<&str>`, so a
/// decoder passes an owned field with `.as_deref()` and a bare `None`
/// compiles.
///
/// ```
/// use allama::chat::stream::{MetadataBuilder, UsageMetadataMode};
/// use allama::{FinishReason, StreamEvent, Usage};
///
/// // Whatever the wire had.
/// let request_id: Option<String> = Some("req_1".into());
/// let model: Option<String> = None;
/// let event = MetadataBuilder::new()
///     .with_finish_reason(FinishReason::Stop)
///     .with_finish_reason_raw("end_turn")
///     .with_usage(Usage::new().input_tokens(12).output_tokens(3))
///     .with_request_id_opt(request_id.as_deref())
///     .with_model_opt(model.as_deref())
///     .with_provider("anthropic")
///     .build();
///
/// let StreamEvent::ResponseMetadata {
///     finish_reason,
///     usage_mode,
///     request_id,
///     model,
///     ..
/// } = &event
/// else {
///     unreachable!()
/// };
/// assert_eq!(*finish_reason, Some(FinishReason::Stop));
/// assert_eq!(*usage_mode, UsageMetadataMode::Snapshot);
/// assert_eq!(request_id.as_deref(), Some("req_1"));
/// assert_eq!(*model, None);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub struct MetadataBuilder {
    /// Always a `ResponseMetadata` event.
    event: StreamEvent,
}

impl Default for MetadataBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MetadataBuilder {
    /// A builder with every field empty (as [`StreamEvent::metadata`]).
    pub fn new() -> Self {
        Self {
            event: StreamEvent::metadata(),
        }
    }

    /// Set the normalized finish reason.
    pub fn with_finish_reason(self, value: FinishReason) -> Self {
        self.with_finish_reason_opt(Some(value))
    }

    /// Set the normalized finish reason when there is one; `None` keeps
    /// the current value. The `_opt` form of
    /// [`with_finish_reason`](Self::with_finish_reason).
    pub fn with_finish_reason_opt(mut self, value: Option<FinishReason>) -> Self {
        if let StreamEvent::ResponseMetadata { finish_reason, .. } = &mut self.event {
            merge(finish_reason, value);
        }
        self
    }

    /// Set the raw provider finish reason.
    pub fn with_finish_reason_raw(mut self, value: impl Into<String>) -> Self {
        if let StreamEvent::ResponseMetadata {
            finish_reason_raw, ..
        } = &mut self.event
        {
            *finish_reason_raw = Some(value.into());
        }
        self
    }

    /// Set the raw provider finish reason when there is one; `None` keeps
    /// the current value. The `_opt` form of
    /// [`with_finish_reason_raw`](Self::with_finish_reason_raw).
    pub fn with_finish_reason_raw_opt(self, value: Option<&str>) -> Self {
        match value {
            Some(value) => self.with_finish_reason_raw(value),
            None => self,
        }
    }

    /// Set the usage, interpreted per [`with_usage_mode`](Self::with_usage_mode).
    pub fn with_usage(self, value: Usage) -> Self {
        self.with_usage_opt(Some(value))
    }

    /// Set the usage when the wire reported one; `None` keeps the current
    /// value. The `_opt` form of [`with_usage`](Self::with_usage).
    pub fn with_usage_opt(mut self, value: Option<Usage>) -> Self {
        if let StreamEvent::ResponseMetadata { usage, .. } = &mut self.event {
            merge(usage, value);
        }
        self
    }

    /// Set how to interpret the usage (default: a snapshot).
    pub fn with_usage_mode(mut self, value: UsageMetadataMode) -> Self {
        if let StreamEvent::ResponseMetadata { usage_mode, .. } = &mut self.event {
            *usage_mode = value;
        }
        self
    }

    /// Set the response identifier.
    pub fn with_id(mut self, value: impl Into<String>) -> Self {
        if let StreamEvent::ResponseMetadata { id, .. } = &mut self.event {
            *id = Some(value.into());
        }
        self
    }

    /// Set the response identifier when there is one; `None` keeps the
    /// current value. The `_opt` form of [`with_id`](Self::with_id).
    pub fn with_id_opt(self, value: Option<&str>) -> Self {
        match value {
            Some(value) => self.with_id(value),
            None => self,
        }
    }

    /// Set the model identifier.
    pub fn with_model(mut self, value: impl Into<String>) -> Self {
        if let StreamEvent::ResponseMetadata { model, .. } = &mut self.event {
            *model = Some(value.into());
        }
        self
    }

    /// Set the model identifier when there is one; `None` keeps the
    /// current value. The `_opt` form of [`with_model`](Self::with_model).
    pub fn with_model_opt(self, value: Option<&str>) -> Self {
        match value {
            Some(value) => self.with_model(value),
            None => self,
        }
    }

    /// Set the provider request identifier.
    pub fn with_request_id(mut self, value: impl Into<String>) -> Self {
        if let StreamEvent::ResponseMetadata { request_id, .. } = &mut self.event {
            *request_id = Some(value.into());
        }
        self
    }

    /// Set the provider request identifier when there is one; `None` keeps
    /// the current value. The `_opt` form of
    /// [`with_request_id`](Self::with_request_id).
    pub fn with_request_id_opt(self, value: Option<&str>) -> Self {
        match value {
            Some(value) => self.with_request_id(value),
            None => self,
        }
    }

    /// Set the provider that served the request, such as an adapter's
    /// `PROVIDER_NAME` constant.
    ///
    /// An adapter built on [`prepare`](crate::adapter::chat::prepare) need
    /// not set it: [`finish_stream`](crate::adapter::chat::Prepared::finish_stream)
    /// fills it in when the stream leaves it empty.
    pub fn with_provider(self, value: impl Into<Cow<'static, str>>) -> Self {
        self.with_provider_opt(Some(value.into()))
    }

    /// Set the provider that served the request when there is one; `None`
    /// keeps the current value. The `_opt` form of
    /// [`with_provider`](Self::with_provider); it takes a `Cow` so a
    /// `&'static str` name is not copied.
    pub fn with_provider_opt(mut self, value: Option<Cow<'static, str>>) -> Self {
        if let StreamEvent::ResponseMetadata { provider, .. } = &mut self.event {
            merge(provider, value);
        }
        self
    }

    /// Set the response-level provider data.
    pub fn with_provider_data(mut self, value: ProviderData) -> Self {
        if let StreamEvent::ResponseMetadata { provider_data, .. } = &mut self.event {
            *provider_data = value;
        }
        self
    }

    /// The `ResponseMetadata` event.
    #[must_use]
    pub fn build(self) -> StreamEvent {
        self.event
    }
}

impl From<MetadataBuilder> for StreamEvent {
    fn from(builder: MetadataBuilder) -> Self {
        builder.build()
    }
}

/// Completeness flags of a collected stream.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StreamCompleteness {
    /// A `ResponseStart` was seen.
    pub saw_response_start: bool,
    /// A `ResponseStop` was seen.
    pub saw_response_stop: bool,
    /// Some blocks never received `BlockStop`.
    pub has_open_blocks: bool,
    /// Incomplete tool calls were dropped from the partial response.
    pub dropped_incomplete_tool_calls: bool,
}

impl StreamCompleteness {
    /// Returns `true` when the stream was fully delivered.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.saw_response_start
            && self.saw_response_stop
            && !self.has_open_blocks
            && !self.dropped_incomplete_tool_calls
    }
}

/// A best-effort collected response plus completeness diagnostics.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CollectedResponse {
    /// The response assembled so far.
    pub response: ChatResponse,
    /// Completeness flags.
    pub completeness: StreamCompleteness,
    /// The error that ended the stream, if any.
    pub terminal_error: Option<Error>,
}

/// Accumulates [`StreamEvent`]s into a [`ChatResponse`].
///
/// A block is final once its `BlockStop` arrives: [`push`](Self::push)
/// returns it then, so callers can act on complete blocks (a tool call, a
/// paragraph) while the stream continues. Events for a stopped block are
/// protocol errors.
///
/// Block indices should be dense (`0, 1, 2, ...`). A decoder may skip the
/// indices of blocks it does not model, but a stream that skips more than
/// 1024 indices in total is a protocol error: the collector keeps a slot per
/// index, so its memory stays linear in the events pushed.
///
/// Application code that reads a stream usually wants
/// [`ChatStreamExt::with_collector`], which drives a collector for it. Push
/// events into a collector directly when you write a wrapper or a stream
/// adapter that sees the events one at a time.
#[derive(Default)]
pub struct StreamCollector {
    blocks: Vec<Option<Slot>>,
    /// Indices left empty by gaps so far (see [`reserve`](Self::reserve)).
    skipped_indices: usize,
    response: ChatResponse,
    saw_response_start: bool,
    saw_response_stop: bool,
    terminal_error: Option<Error>,
    /// Skip block events; see [`metadata_only`](Self::metadata_only).
    metadata_only: bool,
}

enum Slot {
    Open(BlockAccumulator),
    Closed {
        block_type: StreamBlockType,
        block: Finished,
    },
}

/// What a block turned into.
enum Finished {
    Block(ContentBlock),
    /// Text or reasoning with nothing in it.
    Empty,
    /// A closed tool call whose arguments are not valid JSON.
    InvalidToolCall(ToolCall),
    /// A tool call without an id or a name, or one that never closed.
    IncompleteToolCall,
}

struct BlockAccumulator {
    kind: BlockKind,
    provider_data: ProviderData,
    started: bool,
}

impl BlockAccumulator {
    fn finish(self, closed: bool) -> Finished {
        let provider_data = self.provider_data;
        match self.kind {
            BlockKind::Text { text, citations }
                if text.is_empty() && citations.is_empty() && provider_data.is_empty() =>
            {
                Finished::Empty
            }
            BlockKind::Text { text, citations } => Finished::Block(ContentBlock::Text(
                Text::new(text)
                    .with_citations(citations)
                    .with_provider_data(provider_data),
            )),
            BlockKind::Image { source, media_type } => {
                let mut image = Image::new(source);
                image.media_type = media_type;
                image.provider_data = provider_data;
                Finished::Block(ContentBlock::Image(image))
            }
            BlockKind::Reasoning(text) if text.is_empty() && provider_data.is_empty() => {
                Finished::Empty
            }
            BlockKind::Reasoning(text) => Finished::Block(ContentBlock::Reasoning(
                Reasoning::new(text).with_provider_data(provider_data),
            )),
            BlockKind::Refusal(text) if text.is_empty() && provider_data.is_empty() => {
                Finished::Empty
            }
            BlockKind::Refusal(text) => Finished::Block(ContentBlock::Refusal(
                Refusal::new(text).with_provider_data(provider_data),
            )),
            BlockKind::ToolCall {
                id: Some(id),
                name: Some(name),
                arguments,
            } if closed => {
                let arguments = if arguments.trim().is_empty() {
                    "{}".to_owned()
                } else {
                    arguments
                };
                let valid = is_valid_json(&arguments);
                let call = ToolCall::new(id, name, arguments).with_provider_data(provider_data);
                if valid {
                    Finished::Block(ContentBlock::ToolCall(call))
                } else {
                    Finished::InvalidToolCall(call)
                }
            }
            BlockKind::ToolCall { .. } => Finished::IncompleteToolCall,
            BlockKind::ProviderToolCall {
                id: Some(id),
                name: Some(name),
                arguments,
                origin,
            } if closed => {
                let arguments = if arguments.trim().is_empty() {
                    "{}".to_owned()
                } else {
                    arguments
                };
                let mut call =
                    ProviderToolCall::new(id, name, arguments).with_provider_data(provider_data);
                call.origin = origin;
                Finished::Block(ContentBlock::ProviderToolCall(call))
            }
            BlockKind::ProviderToolCall { .. } => Finished::IncompleteToolCall,
            BlockKind::ProviderToolResult {
                call_id,
                output,
                is_error,
                origin,
            } => {
                let mut result = ProviderToolResult::new(call_id, output)
                    .with_error(is_error)
                    .with_provider_data(provider_data);
                result.origin = origin;
                Finished::Block(ContentBlock::ProviderToolResult(result))
            }
            BlockKind::Other {
                type_name,
                data,
                origin,
            } => {
                let mut other =
                    OtherContent::new(type_name, data).with_provider_data(provider_data);
                other.origin = origin;
                Finished::Block(ContentBlock::Other(other))
            }
        }
    }
}

enum BlockKind {
    Text {
        text: String,
        citations: Vec<Citation>,
    },
    Image {
        source: MediaSource,
        /// The declared media type of a URL image.
        media_type: Option<String>,
    },
    Reasoning(String),
    Refusal(String),
    ToolCall {
        id: Option<String>,
        name: Option<String>,
        arguments: String,
    },
    ProviderToolCall {
        id: Option<String>,
        name: Option<String>,
        arguments: String,
        origin: Option<String>,
    },
    ProviderToolResult {
        call_id: String,
        output: serde_json::Value,
        is_error: bool,
        origin: Option<String>,
    },
    Other {
        type_name: String,
        data: ExtraMap,
        origin: Option<String>,
    },
}

impl BlockKind {
    fn block_type(&self) -> StreamBlockType {
        match self {
            Self::Text { .. } => StreamBlockType::Text,
            Self::Image { .. } => StreamBlockType::Image,
            Self::Reasoning(_) => StreamBlockType::Reasoning,
            Self::Refusal(_) => StreamBlockType::Refusal,
            Self::ToolCall { .. } => StreamBlockType::ToolCall,
            Self::ProviderToolCall { .. } => StreamBlockType::ProviderToolCall,
            Self::ProviderToolResult { .. } => StreamBlockType::ProviderToolResult,
            Self::Other { .. } => StreamBlockType::Other,
        }
    }

    fn label(&self) -> &'static str {
        self.block_type().label()
    }

    fn text() -> Self {
        Self::Text {
            text: String::new(),
            citations: Vec::new(),
        }
    }
}

impl StreamCollector {
    /// Create an empty collector.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a collector that merges response metadata and ignores
    /// content blocks.
    ///
    /// Response-level events (`ResponseStart`, `ResponseMetadata`,
    /// `ResponseProviderData`, `Warning`, `ResponseStop`) are merged exactly
    /// as [`new`](Self::new) merges them. Usage is folded by each event's
    /// [`UsageMetadataMode`]: a `Snapshot` replaces the usage so far, a
    /// `Delta` adds to it. Block events are dropped without buffering or
    /// protocol checks, so the collected response has no content and never
    /// reports open blocks or dropped tool calls.
    ///
    /// For observers that need usage, finish reason, id and model but not
    /// the output itself, such as a budget guard that passes a stream on
    /// and counts its tokens. Whether a stream reports usage at all is the
    /// provider's [`ChatCapability::StreamUsage`](crate::ChatCapability::StreamUsage)
    /// answer for the model.
    ///
    /// ```
    /// use allama::chat::stream::{MetadataBuilder, StreamCollector, UsageMetadataMode};
    /// use allama::{StreamEvent, Usage};
    ///
    /// let delta = |usage: Usage| {
    ///     MetadataBuilder::new()
    ///         .with_usage(usage)
    ///         .with_usage_mode(UsageMetadataMode::Delta)
    ///         .build()
    /// };
    /// let mut budget = StreamCollector::metadata_only();
    /// for event in [
    ///     StreamEvent::response_start(Some("resp_1".into()), None),
    ///     delta(Usage::new().input_tokens(12).output_tokens(1)),
    ///     StreamEvent::text_delta(0, "Hello"),
    ///     delta(Usage::new().output_tokens(4)),
    ///     StreamEvent::ResponseStop,
    /// ] {
    ///     budget.push_ref(&event)?; // and pass `event` on
    /// }
    ///
    /// let collected = budget.finish_partial()?;
    /// let usage = collected.response.usage.expect("the stream reported usage");
    /// assert_eq!((usage.input_tokens, usage.output_tokens), (Some(12), Some(5)));
    /// assert!(collected.response.content.is_empty());
    /// # Ok::<(), allama::Error>(())
    /// ```
    #[must_use]
    pub fn metadata_only() -> Self {
        Self {
            metadata_only: true,
            ..Self::default()
        }
    }

    /// Feed one event.
    ///
    /// # Errors
    ///
    /// Returns a protocol [`Error::Stream`] when events violate the stream
    /// protocol (events after `ResponseStop`, conflicting blocks, …).
    pub fn push(&mut self, event: StreamEvent) -> Result<Option<&ContentBlock>> {
        if self.saw_response_stop {
            return Err(Error::stream_protocol(match event {
                StreamEvent::ResponseStop => "duplicate response_stop".to_owned(),
                other => format!("{} received after response_stop", other.name()),
            }));
        }

        if self.metadata_only && event.is_block_event() {
            return Ok(None);
        }

        match event {
            StreamEvent::ResponseStart { id, model } => {
                self.saw_response_start = true;
                merge(&mut self.response.id, id);
                merge(&mut self.response.model, model);
            }
            StreamEvent::BlockStart {
                index,
                block_type,
                id,
                name,
                type_name,
                data,
                origin,
                provider_data,
            } => self.start_block(
                index,
                block_type,
                id,
                name,
                type_name,
                data,
                origin,
                provider_data,
            )?,
            StreamEvent::TextDelta { index, text } => {
                match &mut self.block(index, BlockKind::text)?.kind {
                    BlockKind::Text { text: buf, .. } => buf.push_str(&text),
                    other => return Err(mismatch("text_delta", other, index)),
                }
            }
            StreamEvent::ReasoningDelta { index, text } => {
                match &mut self
                    .block(index, || BlockKind::Reasoning(String::new()))?
                    .kind
                {
                    BlockKind::Reasoning(buf) => buf.push_str(&text),
                    other => return Err(mismatch("reasoning_delta", other, index)),
                }
            }
            StreamEvent::RefusalDelta { index, text } => {
                match &mut self
                    .block(index, || BlockKind::Refusal(String::new()))?
                    .kind
                {
                    BlockKind::Refusal(buf) => buf.push_str(&text),
                    other => return Err(mismatch("refusal_delta", other, index)),
                }
            }
            StreamEvent::ToolCallDelta { index, arguments } => {
                let block = self.block(index, || BlockKind::ToolCall {
                    id: None,
                    name: None,
                    arguments: String::new(),
                })?;
                match &mut block.kind {
                    BlockKind::ToolCall { arguments: buf, .. }
                    | BlockKind::ProviderToolCall { arguments: buf, .. } => {
                        buf.push_str(&arguments)
                    }
                    other => return Err(mismatch("tool_call_delta", other, index)),
                }
            }
            StreamEvent::Citation { index, citation } => {
                match &mut self.block(index, BlockKind::text)?.kind {
                    BlockKind::Text { citations, .. } => citations.push(citation),
                    other => return Err(mismatch("citation", other, index)),
                }
            }
            StreamEvent::BlockProviderData {
                index,
                provider_data,
            } => {
                let block = self.open_block(index, "block_provider_data")?;
                block.provider_data.merge(provider_data);
            }
            StreamEvent::ResponseProviderData { provider_data } => {
                self.response.provider_data.merge(provider_data);
            }
            StreamEvent::ProviderDelta { .. } => {}
            StreamEvent::BlockStop { index } => return self.stop_block(index),
            StreamEvent::Warning { warning } => self.response.warnings.push(warning),
            StreamEvent::ResponseMetadata {
                finish_reason,
                finish_reason_raw,
                usage,
                usage_mode,
                id,
                model,
                request_id,
                provider,
                provider_data,
            } => {
                merge(&mut self.response.finish_reason, finish_reason);
                merge(&mut self.response.finish_reason_raw, finish_reason_raw);
                merge(&mut self.response.id, id);
                merge(&mut self.response.model, model);
                merge(&mut self.response.request_id, request_id);
                merge(&mut self.response.provider, provider);
                self.response.provider_data.merge(provider_data);
                if let Some(usage) = usage {
                    match (usage_mode, &mut self.response.usage) {
                        (UsageMetadataMode::Delta, Some(existing)) => *existing += usage,
                        _ => self.response.usage = Some(usage),
                    }
                }
            }
            StreamEvent::ResponseStop => self.saw_response_stop = true,
        }
        Ok(None)
    }

    /// Feed one event by reference.
    ///
    /// # Errors
    ///
    /// See [`push`](Self::push).
    pub fn push_ref(&mut self, event: &StreamEvent) -> Result<Option<&ContentBlock>> {
        // A skipped block event needs no copy.
        if self.metadata_only && event.is_block_event() && !self.saw_response_stop {
            return Ok(None);
        }
        self.push(event.clone())
    }

    /// Record the error that ended the stream.
    pub fn set_terminal_error(&mut self, error: Error) {
        self.terminal_error = Some(error);
    }

    /// The response-level metadata merged so far (usage, warnings, id,
    /// model, finish reason), without content: blocks are assembled when
    /// the collector finishes.
    #[cfg_attr(not(feature = "chat-extract"), allow(dead_code))]
    pub(crate) fn response_metadata(&self) -> &ChatResponse {
        &self.response
    }

    /// The type of the block at `index`, open or closed, or `None` when no
    /// block has that index.
    ///
    /// Lets a UI route a delta without tracking `BlockStart`s itself: a
    /// `ToolCallDelta` belongs to a client call
    /// ([`StreamBlockType::ToolCall`]) or a provider-executed one
    /// ([`StreamBlockType::ProviderToolCall`]). A block that began with a
    /// delta has the type that delta implies.
    #[must_use]
    pub fn block_type(&self, index: usize) -> Option<StreamBlockType> {
        match self.blocks.get(index)?.as_ref()? {
            Slot::Open(block) => Some(block.kind.block_type()),
            Slot::Closed { block_type, .. } => Some(*block_type),
        }
    }

    /// Finish collection, requiring a complete stream.
    ///
    /// A complete stream collects to `Ok` whatever its finish reason, as a
    /// unary response does: a tool call whose arguments are not valid JSON
    /// (cut off at the output cap, or malformed) is kept with the arguments
    /// as they arrived ([`ToolCall::has_valid_arguments`] is `false`).
    /// Errors are reserved for streams that ended early or broke the
    /// protocol.
    ///
    /// # Errors
    ///
    /// Returns an [`Error::Stream`] of kind
    /// [`Incomplete`](StreamErrorKind::Incomplete) (retryable) when the
    /// stream ended early, or the terminal error when one was recorded.
    pub fn finish(self) -> Result<ChatResponse> {
        let unidentified = self.has_unidentified_tool_call();
        let collected = self.build()?;
        if let Some(error) = collected.terminal_error {
            return Err(error);
        }
        let c = &collected.completeness;
        if !c.is_complete() {
            // An adapter that opened a tool call with `block_start` gave it
            // no id or name: say which constructor carries them.
            let hint = if unidentified {
                "; a tool call has no id or name: start it with \
                 `StreamEvent::tool_call_start` (`provider_tool_call_start` for a \
                 provider-executed one), not `block_start`"
            } else {
                ""
            };
            return Err(Error::stream(
                StreamErrorKind::Incomplete,
                format!(
                    "stream incomplete: start={}, stop={}, open_blocks={}, dropped_incomplete_tool_calls={}{hint}",
                    c.saw_response_start,
                    c.saw_response_stop,
                    c.has_open_blocks,
                    c.dropped_incomplete_tool_calls
                ),
            ));
        }
        Ok(collected.response)
    }

    /// Finish collection, returning whatever was assembled.
    ///
    /// # Errors
    ///
    /// Never fails for incomplete streams; reserved for future validation.
    pub fn finish_partial(self) -> Result<CollectedResponse> {
        self.build()
    }

    /// The open block at `index`, created from `default` when the stream
    /// sends deltas without a `BlockStart`.
    fn block(
        &mut self,
        index: usize,
        default: impl FnOnce() -> BlockKind,
    ) -> Result<&mut BlockAccumulator> {
        self.reserve(index)?;
        let slot = self.blocks[index].get_or_insert_with(|| {
            Slot::Open(BlockAccumulator {
                kind: default(),
                provider_data: ProviderData::new(),
                started: false,
            })
        });
        match slot {
            Slot::Open(block) => Ok(block),
            Slot::Closed { block_type, .. } => Err(Error::stream_protocol(format!(
                "delta received for closed {} block at index {index}",
                block_type.label()
            ))),
        }
    }

    /// Make room for a block at `index`, refusing to skip more than
    /// [`MAX_BLOCK_INDEX_GAP`] indices over the whole stream: indices come
    /// from the wire, and neither one absurd index nor many small gaps may
    /// allocate memory for blocks that never arrive (or overflow). Memory
    /// stays linear in the events pushed.
    fn reserve(&mut self, index: usize) -> Result<()> {
        if index >= self.blocks.len() {
            let gap = index - self.blocks.len();
            let skipped = self.skipped_indices.saturating_add(gap);
            if skipped > MAX_BLOCK_INDEX_GAP {
                return Err(Error::stream_protocol(format!(
                    "block index {index} leaves {skipped} block indices unused in this \
                     stream (at most {MAX_BLOCK_INDEX_GAP}); after {} blocks",
                    self.blocks.len()
                )));
            }
            self.skipped_indices = skipped;
            self.blocks.resize_with(index + 1, || None);
        }
        Ok(())
    }

    /// The open block at `index`, which must exist.
    fn open_block(&mut self, index: usize, event: &str) -> Result<&mut BlockAccumulator> {
        match self.blocks.get_mut(index).and_then(Option::as_mut) {
            Some(Slot::Open(block)) => Ok(block),
            Some(Slot::Closed { block_type, .. }) => Err(Error::stream_protocol(format!(
                "{event} for closed {} block at index {index}",
                block_type.label()
            ))),
            None => Err(Error::stream_protocol(format!(
                "{event} for unknown index {index}"
            ))),
        }
    }

    fn stop_block(&mut self, index: usize) -> Result<Option<&ContentBlock>> {
        let slot = self
            .blocks
            .get_mut(index)
            .and_then(Option::take)
            .ok_or_else(|| {
                Error::stream_protocol(format!("block_stop for unknown index {index}"))
            })?;
        let closed = match slot {
            Slot::Open(block) => Slot::Closed {
                block_type: block.kind.block_type(),
                block: block.finish(true),
            },
            Slot::Closed { block_type, block } => {
                self.blocks[index] = Some(Slot::Closed { block_type, block });
                return Err(Error::stream_protocol(format!(
                    "duplicate block_stop for {} block at index {index}",
                    block_type.label()
                )));
            }
        };
        let slot = self.blocks[index].insert(closed);
        Ok(match slot {
            Slot::Closed {
                block: Finished::Block(block),
                ..
            } => Some(block),
            _ => None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn start_block(
        &mut self,
        index: usize,
        block_type: StreamBlockType,
        id: Option<String>,
        name: Option<String>,
        type_name: Option<String>,
        data: Option<ExtraMap>,
        origin: Option<String>,
        provider_data: ProviderData,
    ) -> Result<()> {
        let fresh = match block_type {
            StreamBlockType::Text => BlockKind::text(),
            StreamBlockType::Reasoning => BlockKind::Reasoning(String::new()),
            StreamBlockType::Refusal => BlockKind::Refusal(String::new()),
            StreamBlockType::ToolCall => BlockKind::ToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments: String::new(),
            },
            StreamBlockType::Image => {
                let (source, media_type) = image_from_data(data.clone())?;
                BlockKind::Image { source, media_type }
            }
            StreamBlockType::ProviderToolCall => BlockKind::ProviderToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments: String::new(),
                origin,
            },
            StreamBlockType::ProviderToolResult => {
                let call_id = id.clone().ok_or_else(|| {
                    Error::stream_protocol(format!(
                        "provider_tool_result block at index {index} has no call id"
                    ))
                })?;
                let mut data = data.clone().unwrap_or_default();
                BlockKind::ProviderToolResult {
                    call_id,
                    output: data.remove("output").unwrap_or_default(),
                    is_error: data
                        .remove("is_error")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                    origin,
                }
            }
            StreamBlockType::Other => BlockKind::Other {
                type_name: type_name.unwrap_or_else(|| "other".into()),
                data: data.unwrap_or_default(),
                origin,
            },
        };

        self.reserve(index)?;
        let slot = &mut self.blocks[index];
        match slot {
            None => {
                *slot = Some(Slot::Open(BlockAccumulator {
                    kind: fresh,
                    provider_data,
                    started: true,
                }));
                Ok(())
            }
            Some(Slot::Closed { block_type, .. }) => Err(Error::stream_protocol(format!(
                "block_start for closed {} block at index {index}",
                block_type.label()
            ))),
            Some(Slot::Open(block)) => {
                let label = block.kind.label();
                if block.started {
                    return Err(Error::stream_protocol(format!(
                        "duplicate block_start for {label} block at index {index}"
                    )));
                }
                if label != fresh.label() {
                    return Err(Error::stream_protocol(format!(
                        "block_start for {} conflicted with existing {label} block at index {index}",
                        fresh.label()
                    )));
                }
                block.started = true;
                block.provider_data.merge(provider_data);
                if let BlockKind::ToolCall {
                    id: existing_id,
                    name: existing_name,
                    ..
                } = &mut block.kind
                {
                    merge_strict(existing_id, id, index, "id")?;
                    merge_strict(existing_name, name, index, "name")?;
                }
                Ok(())
            }
        }
    }

    /// Whether a tool call ended up without its id or name.
    fn has_unidentified_tool_call(&self) -> bool {
        self.blocks.iter().flatten().any(|slot| match slot {
            Slot::Open(block) => matches!(
                &block.kind,
                BlockKind::ToolCall { id, name, .. }
                    | BlockKind::ProviderToolCall { id, name, .. }
                    if id.is_none() || name.is_none()
            ),
            // A closed call is incomplete only when it lacks either.
            Slot::Closed { block, .. } => matches!(block, Finished::IncompleteToolCall),
        })
    }

    fn build(self) -> Result<CollectedResponse> {
        let mut content = Vec::with_capacity(self.blocks.len());
        let mut has_open_blocks = false;
        let mut dropped_incomplete_tool_calls = false;

        for slot in self.blocks.into_iter().flatten() {
            let finished = match slot {
                Slot::Open(block) => {
                    has_open_blocks = true;
                    block.finish(false)
                }
                Slot::Closed { block, .. } => block,
            };
            match finished {
                Finished::Block(block) => content.push(block),
                Finished::Empty => {}
                // A closed call keeps its arguments as they arrived, as a
                // unary response does: `ToolCall::parse_arguments` reports
                // them, and the finish reason says why (an output cap cut
                // them off, for example).
                Finished::InvalidToolCall(call) => content.push(ContentBlock::ToolCall(call)),
                Finished::IncompleteToolCall => {
                    dropped_incomplete_tool_calls = true;
                }
            }
        }

        let mut response = self.response;
        response.content = content;
        Ok(CollectedResponse {
            response,
            completeness: StreamCompleteness {
                saw_response_start: self.saw_response_start,
                saw_response_stop: self.saw_response_stop,
                has_open_blocks,
                dropped_incomplete_tool_calls,
            },
            terminal_error: self.terminal_error,
        })
    }
}

/// Counts and state, never content.
impl fmt::Debug for StreamCollector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamCollector")
            .field("blocks", &self.blocks.len())
            .field("saw_response_start", &self.saw_response_start)
            .field("saw_response_stop", &self.saw_response_stop)
            .field(
                "terminal_error",
                &self.terminal_error.as_ref().map(Error::kind_name),
            )
            .field("metadata_only", &self.metadata_only)
            .finish_non_exhaustive()
    }
}

/// How many block indices a stream may skip in total. Decoders number
/// blocks densely, so more gaps are a protocol violation, not a stream to
/// allocate for.
const MAX_BLOCK_INDEX_GAP: usize = 1024;

fn merge<T>(slot: &mut Option<T>, value: Option<T>) {
    if value.is_some() {
        *slot = value;
    }
}

fn merge_strict(
    existing: &mut Option<String>,
    incoming: Option<String>,
    index: usize,
    field: &str,
) -> Result<()> {
    match (existing.as_ref(), incoming) {
        (Some(current), Some(new)) if current != &new => Err(Error::stream_protocol(format!(
            "conflicting {field} for tool call block at index {index}"
        ))),
        (None, Some(new)) => {
            *existing = Some(new);
            Ok(())
        }
        _ => Ok(()),
    }
}

fn mismatch(event: &str, kind: &BlockKind, index: usize) -> Error {
    Error::stream_protocol(format!(
        "{event} received for {} block at index {index}",
        kind.label()
    ))
}

/// An image block's source and declared media type, from its `BlockStart`
/// data.
fn image_from_data(data: Option<ExtraMap>) -> Result<(MediaSource, Option<String>)> {
    let mut data = data.ok_or_else(|| Error::stream_protocol("image block missing source"))?;
    let source = data
        .remove("source")
        .ok_or_else(|| Error::stream_protocol("image block missing source"))?;
    let source = serde_json::from_value(source)
        .map_err(|e| Error::stream_decode(format!("image block source was invalid: {e}")))?;
    let media_type = match data.remove("media_type") {
        Some(serde_json::Value::String(media_type)) => Some(media_type),
        _ => None,
    };
    Ok((source, media_type))
}

fn image_data(source: &MediaSource, media_type: Option<&str>) -> ExtraMap {
    let mut data = ExtraMap::new();
    data.insert(
        "source".into(),
        serde_json::to_value(source).expect("media source serializes"),
    );
    if let Some(media_type) = media_type {
        data.insert(
            "media_type".into(),
            serde_json::Value::String(media_type.to_owned()),
        );
    }
    data
}

/// A stream that replays a complete [`ChatResponse`] as events.
///
/// Providers without native streaming return this from `chat_stream`.
pub struct SingleResponseStream {
    events: std::vec::IntoIter<StreamEvent>,
}

impl SingleResponseStream {
    /// Create a stream from a response.
    #[must_use]
    pub fn new(response: ChatResponse) -> Self {
        Self {
            events: response
                .into_stream_events()
                .collect::<Vec<_>>()
                .into_iter(),
        }
    }
}

impl fmt::Debug for SingleResponseStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SingleResponseStream")
            .field("remaining_events", &self.events.len())
            .finish()
    }
}

impl From<ChatResponse> for SingleResponseStream {
    fn from(response: ChatResponse) -> Self {
        Self::new(response)
    }
}

impl Stream for SingleResponseStream {
    type Item = Result<StreamEvent>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(self.events.next().map(Ok))
    }
}

impl ChatResponse {
    /// Replay this response as a stream.
    #[must_use]
    pub fn into_stream(self) -> SingleResponseStream {
        SingleResponseStream::new(self)
    }

    /// Replay this response as stream events.
    pub fn into_stream_events(self) -> impl Iterator<Item = StreamEvent> {
        let ChatResponse {
            content,
            finish_reason,
            finish_reason_raw,
            usage,
            model,
            id,
            request_id,
            provider,
            warnings,
            provider_data,
            finish_reported: _,
        } = self;

        let mut events = Vec::with_capacity(content.len() * 3 + warnings.len() + 3);
        events.push(StreamEvent::ResponseStart {
            id: id.clone(),
            model: model.clone(),
        });
        events.extend(
            warnings
                .into_iter()
                .map(|warning| StreamEvent::Warning { warning }),
        );
        for (index, block) in content.into_iter().enumerate() {
            block_events(index, block, &mut events);
        }
        events.push(StreamEvent::ResponseMetadata {
            finish_reason,
            finish_reason_raw,
            usage,
            usage_mode: UsageMetadataMode::Snapshot,
            id,
            model,
            request_id,
            provider,
            provider_data,
        });
        events.push(StreamEvent::ResponseStop);
        events.into_iter()
    }
}

fn block_events(index: usize, block: ContentBlock, events: &mut Vec<StreamEvent>) {
    let mut start = |block_type, id, name, type_name, data, origin, provider_data| {
        events.push(StreamEvent::BlockStart {
            index,
            block_type,
            id,
            name,
            type_name,
            data,
            origin,
            provider_data,
        });
    };
    match block {
        ContentBlock::Text(text) => {
            start(
                StreamBlockType::Text,
                None,
                None,
                None,
                None,
                None,
                text.provider_data,
            );
            if !text.text.is_empty() {
                events.push(StreamEvent::TextDelta {
                    index,
                    text: text.text,
                });
            }
            events.extend(
                text.citations
                    .into_iter()
                    .map(|citation| StreamEvent::Citation { index, citation }),
            );
        }
        ContentBlock::Image(image) => events.push(StreamEvent::image_start(index, image)),
        ContentBlock::Reasoning(reasoning) => {
            start(
                StreamBlockType::Reasoning,
                None,
                None,
                None,
                None,
                None,
                reasoning.provider_data,
            );
            if !reasoning.text.is_empty() {
                events.push(StreamEvent::ReasoningDelta {
                    index,
                    text: reasoning.text,
                });
            }
        }
        ContentBlock::Refusal(refusal) => {
            start(
                StreamBlockType::Refusal,
                None,
                None,
                None,
                None,
                None,
                refusal.provider_data,
            );
            if !refusal.text.is_empty() {
                events.push(StreamEvent::RefusalDelta {
                    index,
                    text: refusal.text,
                });
            }
        }
        ContentBlock::ToolCall(call) => {
            start(
                StreamBlockType::ToolCall,
                Some(call.id),
                Some(call.name),
                None,
                None,
                None,
                call.provider_data,
            );
            events.push(StreamEvent::ToolCallDelta {
                index,
                arguments: call.arguments,
            });
        }
        ContentBlock::ProviderToolCall(call) => {
            start(
                StreamBlockType::ProviderToolCall,
                Some(call.id),
                Some(call.name),
                None,
                None,
                call.origin,
                call.provider_data,
            );
            events.push(StreamEvent::ToolCallDelta {
                index,
                arguments: call.arguments,
            });
        }
        ContentBlock::ProviderToolResult(result) => {
            events.push(StreamEvent::provider_tool_result_start(index, result));
        }
        ContentBlock::Other(other) => events.push(StreamEvent::other_start(index, other)),
    }
    events.push(StreamEvent::BlockStop { index });
}

/// Extension methods for chat streams.
pub trait ChatStreamExt: Stream<Item = Result<StreamEvent>> + Send {
    /// Collect a complete response.
    ///
    /// # Errors
    ///
    /// Fails on the first stream error, on protocol violations, and with a
    /// retryable [`StreamErrorKind::Incomplete`] error when the stream ends
    /// early.
    fn collect_response(self) -> impl std::future::Future<Output = Result<ChatResponse>> + Send;

    /// The next event, or `None` at the end of the stream.
    ///
    /// The same as `StreamExt::next` from `futures-util`, without importing
    /// a stream extension trait: a loop over a chat stream needs nothing
    /// beyond [`allama::prelude`](crate::prelude), and a `tokio_stream` or
    /// `futures` import of your own never makes the call ambiguous.
    ///
    /// ```
    /// # #[cfg(feature = "mock")]
    /// # async fn example() -> allama::Result<()> {
    /// use allama::prelude::*;
    /// use allama::mock::MockProvider;
    ///
    /// let provider = MockProvider::with_text("Hello");
    /// let mut stream = provider.chat_stream(&ChatRequest::new("m").user("hi")).await?;
    /// while let Some(event) = stream.next_event().await {
    ///     if let StreamEvent::TextDelta { text, .. } = event? {
    ///         print!("{text}");
    ///     }
    /// }
    /// # Ok(()) }
    /// ```
    fn next_event(
        &mut self,
    ) -> impl std::future::Future<Output = Option<Result<StreamEvent>>> + Send
    where
        Self: Unpin,
    {
        std::future::poll_fn(move |cx| Pin::new(&mut *self).poll_next(cx))
    }

    /// Collect the response while streaming; each event comes with the block
    /// it completes.
    ///
    /// Use it to show events as they arrive and still get the
    /// [`ChatResponse`] at the end; use [`collect_response`](Self::collect_response)
    /// when only the response matters.
    ///
    /// Each item is a [`CollectedEvent`]: the [`event`](CollectedEvent::event)
    /// and, for a `BlockStop`, the block it [`completed`](CollectedEvent::completed),
    /// so a caller can act on a complete tool call or text block while the
    /// rest of the response is still streaming. Read the response with
    /// [`WithCollector::finish`]. [`WithCollector::next_event`] reads the
    /// next item without a stream extension trait in scope:
    ///
    /// ```
    /// # #[cfg(feature = "mock")]
    /// # async fn example() -> allama::Result<()> {
    /// use allama::prelude::*;
    /// use allama::mock::MockProvider;
    ///
    /// let provider = MockProvider::with_text("Hello");
    /// let mut stream = provider
    ///     .chat_stream(&ChatRequest::new("m").user("hi"))
    ///     .await?
    ///     .with_collector();
    /// while let Some(item) = stream.next_event().await {
    ///     let item = item?;
    ///     if let StreamEvent::TextDelta { text, .. } = &item.event {
    ///         print!("{text}");
    ///     }
    ///     if let Some(ContentBlock::ToolCall(call)) = &item.completed {
    ///         println!("call {}", call.name);
    ///     }
    /// }
    /// let response = stream.finish()?;
    /// # let _ = response; Ok(()) }
    /// ```
    fn with_collector(self) -> WithCollector<Self>
    where
        Self: Sized,
    {
        WithCollector {
            inner: Box::pin(self),
            collector: StreamCollector::new(),
            done: false,
        }
    }
}

/// One item of a [`WithCollector`] stream: an event, and the block it
/// completed.
///
/// The struct is `#[non_exhaustive]`, so it can grow (a block index, a
/// usage snapshot, …) without breaking the loops that read it; read its
/// fields by name, or take it apart with [`into_parts`](Self::into_parts).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct CollectedEvent {
    /// The event, as the stream yielded it.
    pub event: StreamEvent,
    /// The block this event completed: `Some` for the `BlockStop` that
    /// closed a block the collector assembled, `None` for every other
    /// event.
    ///
    /// A tool call's `BlockStop` has `None` for one of two reasons:
    ///
    /// - its arguments are not valid JSON. The call is still in the
    ///   collected response, with the arguments as they arrived
    ///   ([`ToolCall::has_valid_arguments`] is `false`); answer it with
    ///   [`ChatRequest::push_tool_error`](crate::ChatRequest::push_tool_error).
    /// - it is incomplete: it never got an id or a name. The call is dropped,
    ///   and the stream does not collect as complete
    ///   ([`StreamCompleteness::dropped_incomplete_tool_calls`]).
    pub completed: Option<ContentBlock>,
}

impl CollectedEvent {
    /// An event and the block it completed (for test doubles and
    /// adapters of `WithCollector`-shaped streams).
    #[must_use]
    pub fn new(event: StreamEvent, completed: Option<ContentBlock>) -> Self {
        Self { event, completed }
    }

    /// The event and the block it completed.
    #[must_use]
    pub fn into_parts(self) -> (StreamEvent, Option<ContentBlock>) {
        (self.event, self.completed)
    }
}

impl From<CollectedEvent> for (StreamEvent, Option<ContentBlock>) {
    fn from(item: CollectedEvent) -> Self {
        item.into_parts()
    }
}

/// A stream that collects the response while yielding each event with the
/// block it completes ([`CollectedEvent`]). Created by
/// [`ChatStreamExt::with_collector`].
///
/// A stream error is passed through, recorded as the terminal error, and
/// ends the stream, as does a protocol violation (events that cannot be
/// assembled), which is yielded as an error. The wrapped stream is not
/// polled after either.
pub struct WithCollector<S> {
    inner: Pin<Box<S>>,
    collector: StreamCollector,
    done: bool,
}

impl<S> WithCollector<S>
where
    S: Stream<Item = Result<StreamEvent>>,
{
    /// The next item, or `None` at the end of the stream.
    ///
    /// The same as `StreamExt::next`, without importing a stream extension
    /// trait (see [`ChatStreamExt::next_event`]).
    pub async fn next_event(&mut self) -> Option<Result<CollectedEvent>> {
        std::future::poll_fn(|cx| Pin::new(&mut *self).poll_next(cx)).await
    }
}

impl<S> WithCollector<S> {
    /// Borrow the collector that has seen every event so far.
    #[must_use]
    pub fn collector(&self) -> &StreamCollector {
        &self.collector
    }

    /// The collected response; see [`StreamCollector::finish`].
    ///
    /// # Errors
    ///
    /// Fails when the stream was incomplete or ended with an error.
    pub fn finish(self) -> Result<ChatResponse> {
        self.collector.finish()
    }

    /// Whatever was collected; see [`StreamCollector::finish_partial`].
    ///
    /// # Errors
    ///
    /// Reserved for future validation.
    pub fn finish_partial(self) -> Result<CollectedResponse> {
        self.collector.finish_partial()
    }
}

impl<S> fmt::Debug for WithCollector<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WithCollector")
            .field("collector", &self.collector)
            .field("done", &self.done)
            .finish_non_exhaustive()
    }
}

impl<S> Stream for WithCollector<S>
where
    S: Stream<Item = Result<StreamEvent>>,
{
    type Item = Result<CollectedEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        if this.done {
            return Poll::Ready(None);
        }
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(event))) => match this.collector.push_ref(&event) {
                Ok(completed) => {
                    let completed = completed.cloned();
                    Poll::Ready(Some(Ok(CollectedEvent { event, completed })))
                }
                Err(error) => {
                    this.done = true;
                    this.collector.set_terminal_error(error.clone());
                    Poll::Ready(Some(Err(error)))
                }
            },
            Poll::Ready(Some(Err(error))) => {
                this.done = true;
                this.collector.set_terminal_error(error.clone());
                Poll::Ready(Some(Err(error)))
            }
            Poll::Ready(None) => {
                this.done = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<S: Stream<Item = Result<StreamEvent>> + Send> ChatStreamExt for S {
    async fn collect_response(self) -> Result<ChatResponse> {
        use futures_util::StreamExt;
        let mut collector = StreamCollector::new();
        let mut stream = std::pin::pin!(self);
        while let Some(event) = stream.next().await {
            collector.push(event?)?;
        }
        collector.finish()
    }
}

/// A stream that first yields buffered events, then the rest of `inner`.
///
/// It is the committed stream [`until_committed`] returns: an ordinary
/// `Stream` of the same items, to use as you would the original.
///
/// [`until_committed`] returns one holding the events up to the commit
/// point. A wrapper that adds to that prefix before handing the stream on,
/// as [`FallbackProvider`](crate::wrappers::FallbackProvider) does with its
/// failed attempts, takes it apart with [`into_parts`](Self::into_parts)
/// and rebuilds it with [`new`](Self::new).
#[doc(alias = "CommittedStream")]
pub struct Prefixed<S> {
    buffer: VecDeque<StreamEvent>,
    inner: Option<S>,
}

impl<S> Prefixed<S> {
    /// Create a prefixed stream.
    #[must_use]
    pub fn new(buffer: impl IntoIterator<Item = StreamEvent>, inner: Option<S>) -> Self {
        Self {
            buffer: buffer.into_iter().collect(),
            inner,
        }
    }

    /// The buffered events not yet yielded, and the rest of the stream
    /// (`None` when it already ended).
    ///
    /// Pass both back to [`new`](Self::new) to resume the stream where it
    /// was, with any events added to the prefix:
    ///
    /// ```
    /// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
    /// use allama::StreamEvent;
    /// use allama::chat::stream::{Prefixed, until_committed};
    /// use futures_util::{StreamExt, stream};
    ///
    /// let events = stream::iter([
    ///     Ok(StreamEvent::response_start(None, None)),
    ///     Ok(StreamEvent::text_delta(0, "Hi")),
    /// ]);
    /// let committed = until_committed(events).await.unwrap();
    ///
    /// let (mut buffered, rest) = committed.into_parts();
    /// buffered.insert(1, StreamEvent::metadata());
    /// let resumed: Vec<_> = Prefixed::new(buffered, rest).collect().await;
    /// assert_eq!(resumed.len(), 3);
    /// # });
    /// ```
    #[must_use]
    pub fn into_parts(self) -> (Vec<StreamEvent>, Option<S>) {
        (self.buffer.into(), self.inner)
    }
}

impl<S> fmt::Debug for Prefixed<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Prefixed")
            .field("buffered_events", &self.buffer.len())
            .field("ended", &self.inner.is_none())
            .finish()
    }
}

impl<S: Stream<Item = Result<StreamEvent>> + Unpin> Stream for Prefixed<S> {
    type Item = Result<StreamEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(event) = self.buffer.pop_front() {
            return Poll::Ready(Some(Ok(event)));
        }
        match self.inner.as_mut() {
            Some(inner) => Pin::new(inner).poll_next(cx),
            None => Poll::Ready(None),
        }
    }
}

/// Read `stream` until its [commit point](crate::chat::stream#the-commit-point):
/// the first event carrying model output (see [`StreamEvent::is_output`]),
/// or the end of the stream. Before that point an error can still be
/// retried, failed over, or answered with an HTTP status; after it, an
/// error can only end the stream.
///
/// Returns a [`Prefixed`] stream (an ordinary `Stream` of the same items)
/// that replays the buffered events followed by the rest, so the caller
/// observes every event and time-to-first-token is unchanged.
/// When an error arrives before the commit point, nothing has been exposed
/// yet, so the caller may safely retry or fail over.
///
/// The events received before such an error carry no model output, so they
/// are dropped with it.
///
/// ```
/// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
/// use allama::StreamEvent;
/// use allama::chat::stream::until_committed;
/// use futures_util::stream;
///
/// async fn respond() -> allama::Result<usize> {
///     let events = stream::iter([
///         Ok(StreamEvent::response_start(None, None)),
///         Ok(StreamEvent::text_delta(0, "Hi")),
///     ]);
///     let committed = until_committed(events).await?;
///     Ok(committed.into_parts().0.len())
/// }
///
/// async fn handler() -> anyhow::Result<usize> {
///     let failing = stream::iter([Err(allama::Error::overloaded("busy"))]);
///     let committed = until_committed(failing).await?;
///     Ok(committed.into_parts().0.len())
/// }
///
/// assert_eq!(respond().await.unwrap(), 2);
/// assert!(handler().await.is_err());
/// # });
/// ```
///
/// # Errors
///
/// Returns the stream's error when it fails before committing.
pub async fn until_committed<S>(mut stream: S) -> Result<Prefixed<S>>
where
    S: Stream<Item = Result<StreamEvent>> + Unpin,
{
    use futures_util::StreamExt;
    let mut buffered = Vec::new();
    loop {
        match stream.next().await {
            Some(Ok(event)) => {
                let committed = event.is_output();
                buffered.push(event);
                if committed {
                    return Ok(Prefixed::new(buffered, Some(stream)));
                }
            }
            Some(Err(error)) => return Err(error),
            None => return Ok(Prefixed::new(buffered, None)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CitationSource, RequestField, WarningKind};
    use futures_util::StreamExt;
    use serde_json::json;

    /// Application state holding a collecting stream can derive `Debug`.
    /// The output shows counts and state, never the streamed text.
    #[tokio::test]
    async fn stream_types_debug_without_content() {
        let stream: ChatStream =
            Box::pin(ChatResponse::new(vec![ContentBlock::text("secret answer")]).into_stream());
        let single = SingleResponseStream::from(ChatResponse::new(vec![ContentBlock::text(
            "secret answer",
        )]));
        assert!(format!("{single:?}").starts_with("SingleResponseStream { remaining_events: "));

        let mut collecting = stream.with_collector();
        while collecting.next_event().await.is_some() {}
        let debug = format!("{collecting:?}");
        assert!(
            debug.starts_with("WithCollector { collector: StreamCollector { blocks: 1"),
            "{debug}"
        );
        assert!(!debug.contains("secret"), "{debug}");

        let prefixed = Prefixed::<ChatStream>::new([StreamEvent::text_delta(0, "secret")], None);
        assert_eq!(
            format!("{prefixed:?}"),
            "Prefixed { buffered_events: 1, ended: true }"
        );
    }

    #[test]
    fn fieldless_enums_are_hashable() {
        use std::collections::HashSet;

        use crate::{CapabilitySupport, ChatCapability, StreamErrorKind};
        let capabilities: HashSet<(ChatCapability, CapabilitySupport)> =
            [(ChatCapability::ToolCalls, CapabilitySupport::Supported)].into();
        assert!(capabilities.contains(&(ChatCapability::ToolCalls, CapabilitySupport::Supported)));
        let _: HashSet<StreamErrorKind> = HashSet::new();
        let _: HashSet<UsageMetadataMode> = HashSet::new();
        #[cfg(feature = "chat-extract")]
        {
            let _: HashSet<crate::chat::extract::RefusalSource> = HashSet::new();
            let _: HashSet<crate::chat::extract::ChatExtractionMode> = HashSet::new();
        }
    }

    fn sample_response() -> ChatResponse {
        let mut call = ToolCall::new("c1", "lookup", r#"{"q":1}"#);
        call.provider_data
            .insert_raw("google-ai", "thought_signature", json!("sig"));
        ChatResponse::new(vec![
            ContentBlock::Reasoning(Reasoning::new("think").with_provider_data(
                ProviderData::new().with_raw("anthropic", "signature", json!("s")),
            )),
            ContentBlock::Text(Text::new("hello").with_citations(vec![
                Citation::new(CitationSource::url("https://x", Some("X".into()))).with_span(0..5),
            ])),
            ContentBlock::Refusal(Refusal::new("I can't help with that.")),
            ContentBlock::ToolCall(call),
            ContentBlock::Image(Image::url("https://x/y.png")),
            ContentBlock::Other(
                OtherContent::new("citation", ExtraMap::new())
                    .with_origin("anthropic")
                    .with_provider_data(ProviderData::new().with_raw(
                        "anthropic",
                        "signature",
                        json!("other-sig"),
                    )),
            ),
        ])
        .finish_reason(FinishReason::ToolCalls)
        .finish_reason_raw("tool_use")
        .usage(Usage::new().input_tokens(1).output_tokens(2))
        .id("r1")
        .model("m")
        .request_id("req")
        .provider("demo")
        .warning(Warning::unsupported(RequestField::Seed, "ignored"))
    }

    #[tokio::test]
    async fn response_round_trips_through_stream() {
        let response = sample_response();
        let collected = response
            .clone()
            .into_stream()
            .collect_response()
            .await
            .unwrap();
        assert_eq!(collected, response);
    }

    /// A refusal streams as its own block and delta, never as text, and
    /// collects back into a `Refusal` block that `text()` skips.
    #[tokio::test]
    async fn refusals_stream_apart_from_text() {
        let response = ChatResponse::new(vec![
            ContentBlock::text("Sure."),
            ContentBlock::Refusal(Refusal::new("I can't help with that.")),
        ]);
        let events: Vec<StreamEvent> = response.clone().into_stream_events().collect();
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::BlockStart {
                index: 1,
                block_type: StreamBlockType::Refusal,
                ..
            }
        )));
        assert!(events.contains(&StreamEvent::refusal_delta(1, "I can't help with that.")));
        assert!(!events.iter().any(|event| matches!(
            event,
            StreamEvent::TextDelta { text, .. } if text.contains("can't")
        )));

        let collected = response
            .clone()
            .into_stream()
            .collect_response()
            .await
            .unwrap();
        assert_eq!(collected, response);
        assert_eq!(collected.first_refusal(), Some("I can't help with that."));
        assert_eq!(collected.text().as_deref(), Some("Sure."));

        // A delta without a `BlockStart` opens a refusal block.
        let mut collector = StreamCollector::new();
        collector
            .push(StreamEvent::refusal_delta(0, "No."))
            .unwrap();
        assert_eq!(collector.block_type(0), Some(StreamBlockType::Refusal));
        let error = collector.push(StreamEvent::text_delta(0, "x")).unwrap_err();
        assert!(error.to_string().contains("refusal block"), "{error}");
    }

    #[test]
    fn all_block_types_listed() {
        // Exhaustive on purpose: a new variant fails to compile here, as a
        // reminder to add it to `StreamBlockType::ALL` and bump the count.
        fn _exhaustive(block_type: StreamBlockType) {
            match block_type {
                StreamBlockType::Text
                | StreamBlockType::Image
                | StreamBlockType::Reasoning
                | StreamBlockType::Refusal
                | StreamBlockType::ToolCall
                | StreamBlockType::ProviderToolCall
                | StreamBlockType::ProviderToolResult
                | StreamBlockType::Other => {}
            }
        }
        let unique: std::collections::HashSet<_> = StreamBlockType::ALL.iter().collect();
        assert_eq!(unique.len(), StreamBlockType::ALL.len());
        assert_eq!(StreamBlockType::ALL.len(), 8);
    }

    #[test]
    fn refusals_are_output() {
        assert!(StreamEvent::refusal_delta(0, "No.").is_output());
        assert!(StreamEvent::block_start(0, StreamBlockType::Refusal).is_output());
        assert!(StreamEvent::refusal_delta(0, "No.").is_block_event());
    }

    #[test]
    fn events_serialize_round_trip() {
        let events: Vec<StreamEvent> = sample_response().into_stream_events().collect();
        let value = serde_json::to_value(&events).unwrap();
        let back: Vec<StreamEvent> = serde_json::from_value(value).unwrap();
        assert_eq!(back, events);
    }

    #[test]
    fn block_provider_data_merges() {
        let mut c = StreamCollector::new();
        c.push(StreamEvent::ResponseStart {
            id: None,
            model: None,
        })
        .unwrap();
        c.push(StreamEvent::block_start(0, StreamBlockType::Reasoning))
            .unwrap();
        c.push(StreamEvent::ReasoningDelta {
            index: 0,
            text: "a".into(),
        })
        .unwrap();
        c.push(StreamEvent::BlockProviderData {
            index: 0,
            provider_data: ProviderData::new().with_raw("anthropic", "signature", json!("s")),
        })
        .unwrap();
        c.push(StreamEvent::BlockStop { index: 0 }).unwrap();
        c.push(StreamEvent::Warning {
            warning: Warning::new(WarningKind::Dropped, crate::Subject::Response, "x"),
        })
        .unwrap();
        c.push(StreamEvent::ResponseStop).unwrap();
        let response = c.finish().unwrap();
        let reasoning = response.content[0].as_reasoning().unwrap();
        assert_eq!(
            reasoning.provider_data.get_raw("anthropic", "signature"),
            Some(&json!("s"))
        );
        assert_eq!(response.warnings.len(), 1);
    }

    /// Regression: provider data aimed at an `Other` block was accepted and
    /// then dropped when the block finished.
    #[tokio::test]
    async fn other_block_keeps_its_provider_data() {
        let other = OtherContent::new("compaction", ExtraMap::new())
            .with_origin("anthropic")
            .with_provider_data(ProviderData::new().with_raw("anthropic", "start", json!(1)));
        let stream = futures_util::stream::iter(vec![
            Ok(StreamEvent::response_start(None, None)),
            Ok(StreamEvent::other_start(0, other)),
            Ok(StreamEvent::block_provider_data(
                0,
                ProviderData::new().with_raw("anthropic", "signature", json!("other-sig")),
            )),
            Ok(StreamEvent::block_stop(0)),
            Ok(StreamEvent::ResponseStop),
        ]);
        let response = stream.collect_response().await.unwrap();
        assert_eq!(
            response.content[0].provider_data(),
            &ProviderData::new()
                .with_raw("anthropic", "start", json!(1))
                .with_raw("anthropic", "signature", json!("other-sig"))
        );
    }

    #[test]
    fn response_provider_data_merges_into_the_response() {
        let event = StreamEvent::response_provider_data(ProviderData::new().with_raw(
            "openai-compat:lumen",
            "search_results",
            json!([{"url": "https://a"}]),
        ));
        let value = serde_json::to_value(&event).unwrap();
        assert_eq!(
            value,
            json!({"event": "response_provider_data", "provider_data": {"openai-compat:lumen": {"search_results": [{"url": "https://a"}]}}})
        );
        assert_eq!(serde_json::from_value::<StreamEvent>(value).unwrap(), event);
        assert!(!event.is_output());

        let mut c = StreamCollector::new();
        for item in sample_response().into_stream_events() {
            if matches!(item, StreamEvent::BlockStart { index: 0, .. }) {
                c.push(event.clone()).unwrap();
            }
            c.push(item).unwrap();
        }
        let response = c.finish().unwrap();
        assert_eq!(
            response
                .provider_data
                .get_raw("openai-compat:lumen", "search_results"),
            Some(&json!([{"url": "https://a"}]))
        );
        let mut expected = sample_response();
        expected.provider_data.insert_raw(
            "openai-compat:lumen",
            "search_results",
            json!([{"url": "https://a"}]),
        );
        assert_eq!(response, expected);

        // After `ResponseStop` it is a protocol error, like every event.
        let mut c = StreamCollector::new();
        c.push(StreamEvent::ResponseStop).unwrap();
        let err = c
            .push(StreamEvent::response_provider_data(ProviderData::new()))
            .unwrap_err();
        assert!(err.to_string().contains("response_provider_data"), "{err}");
    }

    #[test]
    fn provider_deltas_are_observed_not_collected() {
        let mut data = ExtraMap::new();
        data.insert("query".into(), json!("rust"));
        let delta = StreamEvent::provider_delta(
            3,
            "web_search_call.searching",
            data,
            Some("openai".into()),
        );
        let value = serde_json::to_value(&delta).unwrap();
        assert_eq!(
            value,
            json!({"event": "provider_delta", "index": 3, "type_name": "web_search_call.searching", "data": {"query": "rust"}, "origin": "openai"})
        );
        assert_eq!(serde_json::from_value::<StreamEvent>(value).unwrap(), delta);
        assert!(!delta.is_output());

        let mut c = StreamCollector::new();
        for event in sample_response().into_stream_events() {
            if matches!(event, StreamEvent::ResponseStop) {
                c.push(delta.clone()).unwrap();
            }
            c.push(event).unwrap();
        }
        assert_eq!(c.finish().unwrap(), sample_response());
    }

    #[test]
    fn push_returns_blocks_as_they_complete() {
        let mut c = StreamCollector::new();
        let mut completed = Vec::new();
        for event in sample_response().into_stream_events() {
            if let Some(block) = c.push(event).unwrap() {
                completed.push(block.clone());
            }
        }
        assert_eq!(completed, sample_response().content);

        // Empty text blocks and incomplete tool calls complete as nothing.
        let mut c = StreamCollector::new();
        c.push(StreamEvent::block_start(0, StreamBlockType::Text))
            .unwrap();
        assert!(
            c.push(StreamEvent::BlockStop { index: 0 })
                .unwrap()
                .is_none()
        );
        c.push(StreamEvent::block_start(1, StreamBlockType::ToolCall))
            .unwrap();
        assert!(
            c.push(StreamEvent::BlockStop { index: 1 })
                .unwrap()
                .is_none()
        );
        c.push(StreamEvent::tool_call_start(2, "c", "f")).unwrap();
        c.push(StreamEvent::ToolCallDelta {
            index: 2,
            arguments: "{bad".into(),
        })
        .unwrap();
        assert!(
            c.push(StreamEvent::BlockStop { index: 2 })
                .unwrap()
                .is_none()
        );
        // The call with invalid arguments is kept; the one with no id or
        // name is dropped.
        let partial = c.finish_partial().unwrap();
        let calls: Vec<_> = partial.response.tool_calls().collect();
        assert_eq!(calls, [&ToolCall::new("c", "f", "{bad")]);
        assert!(!calls[0].has_valid_arguments());
        assert!(partial.completeness.dropped_incomplete_tool_calls);
    }

    #[test]
    fn a_tool_call_opened_without_an_id_names_its_constructor() {
        let mut c = StreamCollector::new();
        c.push(StreamEvent::response_start(Some("r".into()), None))
            .unwrap();
        c.push(StreamEvent::block_start(0, StreamBlockType::ToolCall))
            .unwrap();
        c.push(StreamEvent::BlockStop { index: 0 }).unwrap();
        c.push(StreamEvent::ResponseStop).unwrap();
        let err = c.finish().unwrap_err();
        assert!(
            err.message().contains("`StreamEvent::tool_call_start`"),
            "{err}"
        );

        // A call that has its id and name but never closed gets no hint.
        let mut c = StreamCollector::new();
        c.push(StreamEvent::response_start(Some("r".into()), None))
            .unwrap();
        c.push(StreamEvent::tool_call_start(0, "c", "f")).unwrap();
        let err = c.finish().unwrap_err();
        assert!(!err.message().contains("tool_call_start"), "{err}");
    }

    #[test]
    fn unfinished_provider_tool_calls_are_dropped_from_partial_responses() {
        let mut c = StreamCollector::new();
        c.push(StreamEvent::provider_tool_call_start(
            0,
            "srv_1",
            "web_search",
            Some("anthropic".into()),
        ))
        .unwrap();
        c.push(StreamEvent::ToolCallDelta {
            index: 0,
            arguments: r#"{"query":"ru"#.into(),
        })
        .unwrap();
        let partial = c.finish_partial().unwrap();
        assert!(partial.response.content.is_empty());
        assert!(partial.completeness.dropped_incomplete_tool_calls);
    }

    #[test]
    fn open_tool_calls_are_dropped_from_partial_responses() {
        let mut c = StreamCollector::new();
        c.push(StreamEvent::tool_call_start(0, "c1", "get_weather"))
            .unwrap();
        c.push(StreamEvent::tool_call_delta(0, r#"{"city":"Pa"#))
            .unwrap();
        let partial = c.finish_partial().unwrap();
        assert_eq!(partial.response.tool_calls().count(), 0);
        assert!(partial.completeness.dropped_incomplete_tool_calls);
    }

    #[test]
    fn open_tool_call_without_id_or_name_has_a_block_type() {
        let mut c = StreamCollector::new();
        // A delta with no `BlockStart` opens a call with no id or name.
        c.push(StreamEvent::tool_call_delta(0, "{")).unwrap();
        assert_eq!(c.block_type(0), Some(StreamBlockType::ToolCall));
    }

    #[test]
    fn closed_tool_calls_complete_and_are_collected() {
        let mut c = StreamCollector::new();
        c.push(StreamEvent::tool_call_start(0, "c1", "get_weather"))
            .unwrap();
        c.push(StreamEvent::tool_call_delta(0, r#"{"city":"Paris"}"#))
            .unwrap();
        let first = c.push(StreamEvent::block_stop(0)).unwrap().cloned();
        c.push(StreamEvent::tool_call_start(1, "c2", "now"))
            .unwrap();
        let second = c.push(StreamEvent::block_stop(1)).unwrap().cloned();
        let expected = [
            ToolCall::new("c1", "get_weather", r#"{"city":"Paris"}"#),
            ToolCall::new("c2", "now", "{}"),
        ];
        assert_eq!(
            [first, second],
            expected
                .clone()
                .map(|call| Some(ContentBlock::ToolCall(call)))
        );
        let partial = c.finish_partial().unwrap();
        assert!(partial.response.tool_calls().eq(&expected));
    }

    #[test]
    fn invalid_arguments_complete_as_nothing_but_are_collected() {
        let mut c = StreamCollector::new();
        c.push(StreamEvent::tool_call_start(0, "c1", "get_weather"))
            .unwrap();
        c.push(StreamEvent::tool_call_delta(0, r#"{"city":"#))
            .unwrap();
        assert!(c.push(StreamEvent::block_stop(0)).unwrap().is_none());
        let partial = c.finish_partial().unwrap();
        let call = partial.response.first_tool_call().unwrap();
        assert_eq!(
            (call.id.as_str(), call.arguments.as_str()),
            ("c1", r#"{"city":"#)
        );
        assert!(!call.has_valid_arguments());
    }

    #[test]
    fn provider_tool_calls_never_appear_as_tool_calls() {
        let mut c = StreamCollector::new();
        c.push(StreamEvent::provider_tool_call_start(
            0,
            "srv_1",
            "web_search",
            Some("anthropic".into()),
        ))
        .unwrap();
        c.push(StreamEvent::tool_call_delta(0, r#"{"query":"rust"}"#))
            .unwrap();
        c.push(StreamEvent::block_stop(0)).unwrap();

        c.push(StreamEvent::tool_call_start(1, "c1", "get_weather"))
            .unwrap();
        c.push(StreamEvent::block_stop(1)).unwrap();
        let partial = c.finish_partial().unwrap();
        let ids: Vec<_> = partial
            .response
            .tool_calls()
            .map(|call| call.id.as_str())
            .collect();
        assert_eq!(ids, ["c1"]);
        assert_eq!(partial.response.content.len(), 2);
    }

    #[test]
    fn block_type_reports_open_and_closed_blocks() {
        let mut c = StreamCollector::new();
        assert_eq!(c.block_type(0), None);
        c.push(StreamEvent::text_delta(0, "Hi")).unwrap();
        c.push(StreamEvent::provider_tool_call_start(
            1,
            "srv_1",
            "web_search",
            None,
        ))
        .unwrap();
        c.push(StreamEvent::tool_call_start(3, "c1", "f")).unwrap();
        assert_eq!(c.block_type(0), Some(StreamBlockType::Text));
        assert_eq!(c.block_type(1), Some(StreamBlockType::ProviderToolCall));
        assert_eq!(c.block_type(2), None, "gap in the indexes");
        assert_eq!(c.block_type(3), Some(StreamBlockType::ToolCall));
        assert_eq!(c.block_type(4), None);

        for index in [0, 1, 3] {
            c.push(StreamEvent::block_stop(index)).unwrap();
        }
        assert_eq!(c.block_type(0), Some(StreamBlockType::Text));
        assert_eq!(c.block_type(1), Some(StreamBlockType::ProviderToolCall));
        assert_eq!(c.block_type(2), None);
        assert_eq!(c.block_type(3), Some(StreamBlockType::ToolCall));

        // An empty text block closes as nothing but keeps its type.
        c.push(StreamEvent::block_start(5, StreamBlockType::Text))
            .unwrap();
        c.push(StreamEvent::block_stop(5)).unwrap();
        assert_eq!(c.block_type(5), Some(StreamBlockType::Text));
    }

    #[test]
    fn stopped_blocks_reject_further_events() {
        let mut c = StreamCollector::new();
        c.push(StreamEvent::block_start(0, StreamBlockType::Text))
            .unwrap();
        c.push(StreamEvent::BlockStop { index: 0 }).unwrap();
        for event in [
            StreamEvent::TextDelta {
                index: 0,
                text: "late".into(),
            },
            StreamEvent::BlockProviderData {
                index: 0,
                provider_data: ProviderData::new().with_raw("x", "k", json!(1)),
            },
            StreamEvent::block_start(0, StreamBlockType::Text),
            StreamEvent::BlockStop { index: 0 },
        ] {
            let err = c.push(event).unwrap_err();
            assert!(err.to_string().contains("text block"), "{err}");
        }
    }

    #[tokio::test]
    async fn with_collector_pairs_stops_with_blocks() {
        let mut stream = sample_response().into_stream().with_collector();
        let mut completed = Vec::new();
        while let Some(item) = stream.next_event().await {
            let (event, block) = item.unwrap().into_parts();
            assert_eq!(
                block.is_some(),
                matches!(event, StreamEvent::BlockStop { .. })
            );
            completed.extend(block);
        }
        assert_eq!(completed, sample_response().content);
        assert_eq!(stream.finish().unwrap(), sample_response());
    }

    #[tokio::test]
    async fn with_collector_records_stream_errors() {
        let events = futures_util::stream::iter(vec![
            Ok(StreamEvent::ResponseStart {
                id: None,
                model: None,
            }),
            Ok(StreamEvent::TextDelta {
                index: 0,
                text: "partial".into(),
            }),
            Err(Error::overloaded("busy")),
        ]);
        let after = futures_util::stream::iter([Ok(StreamEvent::text_delta(0, "after"))]);
        let mut stream = events.chain(after).with_collector();
        let items: Vec<_> = stream.by_ref().collect().await;
        // The stream ends after its first error.
        assert_eq!(items.len(), 3);
        assert!(items[2].is_err());
        let partial = stream.finish_partial().unwrap();
        assert_eq!(partial.response.text().as_deref(), Some("partial"));
        assert!(matches!(
            partial.terminal_error,
            Some(Error::Overloaded { .. })
        ));
    }

    #[test]
    fn incomplete_stream_is_retryable() {
        let mut c = StreamCollector::new();
        c.push(StreamEvent::ResponseStart {
            id: None,
            model: None,
        })
        .unwrap();
        c.push(StreamEvent::TextDelta {
            index: 0,
            text: "partial".into(),
        })
        .unwrap();
        let err = c.finish().unwrap_err();
        assert!(matches!(
            err,
            Error::Stream {
                kind: StreamErrorKind::Incomplete,
                ..
            }
        ));
        assert!(err.is_retryable());
    }

    #[test]
    fn protocol_violations_are_rejected() {
        let mut c = StreamCollector::new();
        c.push(StreamEvent::ResponseStop).unwrap();
        assert!(c.push(StreamEvent::ResponseStop).is_err());

        let mut c = StreamCollector::new();
        c.push(StreamEvent::block_start(0, StreamBlockType::Text))
            .unwrap();
        assert!(
            c.push(StreamEvent::ToolCallDelta {
                index: 0,
                arguments: "{}".into()
            })
            .is_err()
        );

        let mut c = StreamCollector::new();
        c.push(StreamEvent::tool_call_start(0, "a", "f")).unwrap();
        assert!(c.push(StreamEvent::tool_call_start(0, "b", "f")).is_err());
    }

    #[test]
    fn invalid_tool_arguments_are_kept_like_unary() {
        let events = || {
            vec![
                StreamEvent::ResponseStart {
                    id: None,
                    model: None,
                },
                StreamEvent::tool_call_start(0, "c", "f"),
                StreamEvent::ToolCallDelta {
                    index: 0,
                    arguments: "{bad".into(),
                },
                StreamEvent::BlockStop { index: 0 },
                StreamEvent::ResponseStop,
            ]
        };
        // The same response is `Ok` unary and
        // streamed, with the call's raw arguments and the finish reason.
        let mut strict = StreamCollector::new();
        for e in events() {
            strict.push(e).unwrap();
        }
        let response = strict.finish().unwrap();
        let call = response.first_tool_call().unwrap();
        assert_eq!(call.arguments, "{bad");
        assert!(!call.has_valid_arguments());

        let mut partial = StreamCollector::new();
        for e in events() {
            partial.push(e).unwrap();
        }
        let collected = partial.finish_partial().unwrap();
        assert!(!collected.completeness.dropped_incomplete_tool_calls);
        assert_eq!(collected.response.content.len(), 1);
    }

    /// One absurd block index from the wire is a protocol error, not a
    /// panic or an allocation for every block before it.
    #[test]
    fn absurd_block_indices_are_protocol_errors() {
        for index in [usize::MAX, usize::MAX - 1, 10_000_000_000, 50_000_000] {
            let mut c = StreamCollector::new();
            let err = c.push(StreamEvent::text_delta(index, "x")).unwrap_err();
            assert!(
                matches!(
                    err,
                    Error::Stream {
                        kind: StreamErrorKind::Protocol,
                        ..
                    }
                ),
                "{err:?}"
            );
            let mut c = StreamCollector::new();
            assert!(
                c.push(StreamEvent::block_start(index, StreamBlockType::Text))
                    .is_err()
            );
        }
        // A modest gap is fine: decoders may skip blocks they do not model.
        let mut c = StreamCollector::new();
        c.push(StreamEvent::text_delta(MAX_BLOCK_INDEX_GAP, "x"))
            .unwrap();
        assert_eq!(
            c.block_type(MAX_BLOCK_INDEX_GAP),
            Some(StreamBlockType::Text)
        );
    }

    /// Gaps are bounded over the whole stream, not per event: an index
    /// that climbs by the per-event limit on every small event must not
    /// grow memory a thousand times faster than the input.
    #[test]
    fn stepped_block_indices_are_bounded_in_total() {
        let mut c = StreamCollector::new();
        let mut failed = None;
        for step in 1..=4 {
            let index = step * (MAX_BLOCK_INDEX_GAP / 2 + 1);
            if let Err(err) = c.push(StreamEvent::text_delta(index, "x")) {
                failed = Some((step, err));
                break;
            }
        }
        let (step, err) = failed.expect("stepped indices are refused");
        assert_eq!(step, 2, "the second gap passes the stream's total");
        assert!(
            matches!(
                err,
                Error::Stream {
                    kind: StreamErrorKind::Protocol,
                    ..
                }
            ),
            "{err:?}"
        );
        assert!(
            c.blocks.len() <= MAX_BLOCK_INDEX_GAP + 2,
            "{}",
            c.blocks.len()
        );

        // Dense blocks are never limited, however many there are.
        let mut c = StreamCollector::new();
        for index in 0..4 * MAX_BLOCK_INDEX_GAP {
            c.push(StreamEvent::text_delta(index, "x")).unwrap();
        }
    }

    #[test]
    fn delta_usage_accumulates() {
        let mut c = StreamCollector::new();
        for usage in [Usage::new().output_tokens(2), Usage::new().output_tokens(3)] {
            c.push(StreamEvent::ResponseMetadata {
                usage: Some(usage),
                usage_mode: UsageMetadataMode::Delta,
                finish_reason: None,
                finish_reason_raw: None,
                id: None,
                model: None,
                request_id: None,
                provider: None,
                provider_data: ProviderData::new(),
            })
            .unwrap();
        }
        let collected = c.finish_partial().unwrap();
        assert_eq!(collected.response.usage.unwrap().output_tokens, Some(5));
    }

    /// Every recorded `*_events.json` stream fixture in the workspace, or
    /// `None` outside the workspace (a packaged `allama` has no sibling
    /// crates). In the workspace there are at least ten.
    fn event_fixtures() -> Option<Vec<(String, Vec<StreamEvent>)>> {
        fn visit(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for path in entries.map(|entry| entry.unwrap().path()) {
                if path.is_dir() {
                    visit(&path, out);
                } else if path.to_string_lossy().ends_with("_events.json") {
                    out.push(path);
                }
            }
        }
        let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let workspace = std::fs::read_to_string(crates.join("../Cargo.toml"))
            .is_ok_and(|manifest| manifest.contains("[workspace]"));
        if !workspace {
            eprintln!("skipped: the workspace stream fixtures are not available");
            return None;
        }
        let mut paths = Vec::new();
        for entry in std::fs::read_dir(&crates).unwrap() {
            visit(&entry.unwrap().path().join("fixtures"), &mut paths);
        }
        paths.sort();
        assert!(paths.len() >= 10, "found {} fixtures", paths.len());
        let fixtures = paths
            .into_iter()
            .map(|path| {
                let text = std::fs::read_to_string(&path).unwrap();
                let items: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap();
                // A recorded stream error ends the events.
                let events = items
                    .into_iter()
                    .take_while(|item| item.get("error").is_none())
                    .map(|item| {
                        serde_json::from_value(item)
                            .unwrap_or_else(|err| panic!("{}: {err}", path.display()))
                    })
                    .collect();
                (path.display().to_string(), events)
            })
            .collect();
        Some(fixtures)
    }

    /// The response `events` collect to, in full and metadata-only mode.
    fn collect_both(events: &[StreamEvent]) -> (CollectedResponse, CollectedResponse) {
        let mut full = StreamCollector::new();
        let mut metadata = StreamCollector::metadata_only();
        for event in events {
            let _ = full.push_ref(event);
            metadata.push_ref(event).unwrap();
        }
        (
            full.finish_partial().unwrap(),
            metadata.finish_partial().unwrap(),
        )
    }

    /// Metadata-only collection matches full collection on everything but
    /// content.
    fn assert_same_metadata(label: &str, events: &[StreamEvent]) {
        let (full, metadata) = collect_both(events);
        assert!(metadata.response.content.is_empty(), "{label}");
        assert_eq!(
            metadata.response.usage, full.response.usage,
            "{label}: usage"
        );
        assert_eq!(
            metadata.response.finish_reason, full.response.finish_reason,
            "{label}: finish reason"
        );
        assert_eq!(metadata.response.id, full.response.id, "{label}: id");
        assert_eq!(
            metadata.response.model, full.response.model,
            "{label}: model"
        );
        let mut expected = full.response;
        expected.content.clear();
        assert_eq!(metadata.response, expected, "{label}");
    }

    #[test]
    fn metadata_only_collection_matches_full_collection_on_fixtures() {
        let Some(fixtures) = event_fixtures() else {
            return;
        };
        let mut with_usage = 0;
        for (label, events) in &fixtures {
            assert_same_metadata(label, events);
            if collect_both(events).1.response.usage.is_some() {
                with_usage += 1;
            }
        }
        assert!(with_usage > 0);
    }

    #[test]
    fn metadata_only_collection_matches_full_collection_in_delta_usage_mode() {
        // The recorded fixtures report cumulative usage; replay each with
        // its usage as deltas, so later reports add to earlier ones.
        for (label, mut events) in event_fixtures().unwrap_or_default() {
            for event in &mut events {
                if let StreamEvent::ResponseMetadata { usage_mode, .. } = event {
                    *usage_mode = UsageMetadataMode::Delta;
                }
            }
            assert_same_metadata(&format!("{label} (delta)"), &events);
        }
    }

    #[test]
    fn metadata_only_collection_merges_metadata_and_skips_blocks() {
        let delta = |usage: Usage| {
            let mut event = StreamEvent::metadata();
            if let StreamEvent::ResponseMetadata {
                usage: slot,
                usage_mode,
                ..
            } = &mut event
            {
                *slot = Some(usage);
                *usage_mode = UsageMetadataMode::Delta;
            }
            event
        };
        let mut finish = StreamEvent::metadata();
        if let StreamEvent::ResponseMetadata {
            finish_reason,
            model,
            provider,
            ..
        } = &mut finish
        {
            *finish_reason = Some(FinishReason::Length);
            *model = Some("m-2".into());
            *provider = Some("demo".into());
        }
        let mut events = vec![
            StreamEvent::response_start(Some("r1".into()), Some("m".into())),
            delta(Usage::new().input_tokens(4).output_tokens(1)),
        ];
        events.extend(sample_response().into_stream_events().filter(|event| {
            !matches!(
                event,
                StreamEvent::ResponseStart { .. }
                    | StreamEvent::ResponseMetadata { .. }
                    | StreamEvent::ResponseStop
            )
        }));
        // An unclosed block and an out-of-protocol delta are not checked.
        events.push(StreamEvent::text_delta(9, "dangling"));
        events.push(StreamEvent::block_stop(42));
        events.push(delta(Usage::new().output_tokens(2)));
        events.push(finish);
        events.push(StreamEvent::ResponseStop);

        let mut collector = StreamCollector::metadata_only();
        for event in &events {
            assert!(collector.push_ref(event).unwrap().is_none());
        }
        assert!(
            collector
                .push_ref(&StreamEvent::text_delta(0, "late"))
                .is_err()
        );
        let collected = collector.finish_partial().unwrap();
        let response = &collected.response;
        assert!(response.content.is_empty());
        assert_eq!(
            response.usage,
            Some(Usage::new().input_tokens(4).output_tokens(3))
        );
        assert_eq!(response.finish_reason, Some(FinishReason::Length));
        assert_eq!(response.id.as_deref(), Some("r1"));
        assert_eq!(response.model.as_deref(), Some("m-2"));
        assert_eq!(response.provider.as_deref(), Some("demo"));
        assert_eq!(response.warnings, sample_response().warnings);
        assert!(collected.completeness.is_complete());
    }

    #[tokio::test]
    async fn until_committed_buffers_and_replays() {
        let events = sample_response().into_stream();
        let prefixed = until_committed(events).await.unwrap();
        let all: Vec<_> = prefixed.map(|e| e.unwrap()).collect().await;
        assert_eq!(
            all,
            sample_response().into_stream_events().collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn until_committed_reports_pre_output_errors() {
        let stream = futures_util::stream::iter(vec![
            Ok(StreamEvent::ResponseStart {
                id: None,
                model: None,
            }),
            Err(Error::overloaded("busy")),
        ]);
        let err = until_committed(stream).await.err().unwrap();
        assert!(matches!(err, Error::Overloaded { .. }));
    }

    #[tokio::test]
    async fn until_committed_passes_post_output_errors_through() {
        let stream = futures_util::stream::iter(vec![
            Ok(StreamEvent::ResponseStart {
                id: None,
                model: None,
            }),
            Ok(StreamEvent::TextDelta {
                index: 0,
                text: "x".into(),
            }),
            Err(Error::overloaded("busy")),
        ]);
        let prefixed = until_committed(stream).await.unwrap();
        let items: Vec<_> = prefixed.collect().await;
        assert_eq!(items.len(), 3);
        assert!(items[2].is_err());
    }

    #[tokio::test]
    async fn into_parts_returns_the_unyielded_prefix_and_the_rest() {
        let start = StreamEvent::response_start(None, None);
        let first = StreamEvent::text_delta(0, "a");
        let second = StreamEvent::text_delta(0, "b");
        let rest = futures_util::stream::iter(vec![Ok(second.clone())]);
        let mut prefixed = Prefixed::new([start.clone(), first.clone()], Some(rest));

        // One buffered event is already yielded; it is not handed back.
        assert_eq!(prefixed.next().await.unwrap().unwrap(), start);
        let (buffered, rest) = prefixed.into_parts();
        assert_eq!(buffered, std::slice::from_ref(&first));

        let resumed: Vec<_> = Prefixed::new(buffered, rest)
            .map(Result::unwrap)
            .collect()
            .await;
        assert_eq!(resumed, [first, second]);
    }

    #[tokio::test]
    async fn into_parts_of_an_ended_stream_has_no_rest() {
        let events = futures_util::stream::iter(vec![Ok(StreamEvent::response_start(None, None))]);
        let (buffered, rest) = until_committed(events).await.unwrap().into_parts();
        assert_eq!(buffered.len(), 1);
        assert!(rest.is_none());
    }

    #[test]
    fn metadata_builder_sets_every_field_of_the_event() {
        let provider_data = ProviderData::new().with_raw("demo", "k", serde_json::json!(1));
        let usage = Usage::new().input_tokens(4).output_tokens(2);
        let built: StreamEvent = MetadataBuilder::new()
            .with_finish_reason(FinishReason::Length)
            .with_finish_reason_raw("max_tokens")
            .with_usage(usage.clone())
            .with_usage_mode(UsageMetadataMode::Delta)
            .with_id("r1".to_owned())
            .with_model("m")
            .with_request_id_opt(None)
            .with_provider("demo")
            .with_provider_data(provider_data.clone())
            .into();

        let mut expected = StreamEvent::metadata();
        if let StreamEvent::ResponseMetadata {
            finish_reason,
            finish_reason_raw,
            usage: slot_usage,
            usage_mode,
            id,
            model,
            provider,
            provider_data: slot_provider_data,
            ..
        } = &mut expected
        {
            *finish_reason = Some(FinishReason::Length);
            *finish_reason_raw = Some("max_tokens".into());
            *slot_usage = Some(usage);
            *usage_mode = UsageMetadataMode::Delta;
            *id = Some("r1".into());
            *model = Some("m".into());
            *provider = Some("demo".into());
            *slot_provider_data = provider_data;
        }
        assert_eq!(built, expected);
        assert_eq!(MetadataBuilder::default().build(), StreamEvent::metadata());
    }

    /// Values, and `_opt` twins that keep the current
    /// value on `None` (the `Error` builder rule).
    #[test]
    fn metadata_builder_opt_twins_keep_the_current_value_on_none() {
        let set = MetadataBuilder::new()
            .with_finish_reason(FinishReason::Stop)
            .with_finish_reason_raw("end_turn")
            .with_usage(Usage::new().output_tokens(1))
            .with_id("r1")
            .with_model("m")
            .with_request_id("req_1")
            .with_provider("demo");
        let kept = set
            .clone()
            .with_finish_reason_opt(None)
            .with_finish_reason_raw_opt(None)
            .with_usage_opt(None)
            .with_id_opt(None)
            .with_model_opt(None)
            .with_request_id_opt(None)
            .with_provider_opt(None);
        assert_eq!(kept, set);

        let wire_id: Option<String> = Some("r2".to_owned());
        let StreamEvent::ResponseMetadata { id, .. } = set.with_id_opt(wire_id.as_deref()).build()
        else {
            unreachable!()
        };
        assert_eq!(id.as_deref(), Some("r2"));

        let served_by = Some(Cow::Owned("compat:groq".to_owned()));
        let StreamEvent::ResponseMetadata { provider, .. } =
            MetadataBuilder::new().with_provider_opt(served_by).build()
        else {
            unreachable!()
        };
        assert_eq!(provider.as_deref(), Some("compat:groq"));
    }

    /// A URL image keeps its declared media type through the stream events,
    /// as on a unary response.
    #[test]
    fn image_media_type_survives_the_stream_events() {
        let image =
            Image::new(MediaSource::url("https://example.com/a")).with_media_type("image/jpeg");
        let response = ChatResponse::new(vec![ContentBlock::Image(image.clone())]);
        let mut collector = StreamCollector::new();
        for event in response.into_stream_events() {
            collector.push(event).unwrap();
        }
        let collected = collector.finish().unwrap();
        assert_eq!(collected.content, vec![ContentBlock::Image(image)]);
    }
}
