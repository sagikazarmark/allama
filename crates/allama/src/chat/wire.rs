use std::borrow::Cow;

use crate::chat::preflight::{strip_cache, strip_image_detail};
use crate::{
    CapabilitySupport, ChatCapability, ChatRequest, ContentBlock, ContentPart, Degradations,
    FileKind, Message, OtherContent, ReasoningEffort, ReplayState, RequestField, ResponseFormat,
    Result, Subject, ToolChoice, ToolResultContent, UserContent,
};

/// What an adapter's wire format can carry of a chat request: the portable
/// request fields it has a slot for, and the kinds of content it takes.
///
/// An adapter whose API takes less than the full portable request declares
/// it once, usually as a `const`, and calls
/// [`Prepared::degrade_to`](crate::adapter::chat::Prepared::degrade_to) while
/// encoding. That one pass applies ADR 0003 to everything the wire format
/// cannot carry, so the encoder only encodes what it can send and nothing
/// is dropped silently:
///
/// - **Semantic fields and content are refused** before sending, under
///   every policy: tools and tool turns, a forced tool choice, a JSON
///   response format, an output cap, system prompts, images, documents,
///   audio and video, and
///   content the caller wrote for a provider that this wire format cannot
///   carry.
/// - **Hints are removed with an `Unsupported` warning** coded
///   `unsupported.<field>`, or refused under
///   [`UnsupportedPolicy::Error`](crate::UnsupportedPolicy::Error):
///   `temperature`, `top_p`, `top_k`, `stop`, the penalties, `seed`, reasoning
///   configuration, prompt caching, image detail, and
///   `parallel_tool_calls: false`. What is honoured anyway is removed
///   without a warning: `parallel_tool_calls: true`, an automatic tool
///   choice, a text response format, `ReasoningEffort::None`, and a
///   disabled tool choice, which leaves the tools out.
/// - **Replay state is dropped with a coded `Dropped` warning** under every
///   policy ([`Degradations::dropped_replay_state`]): reasoning the wire
///   format cannot take back, and `Other` content or provider tool blocks
///   another provider wrote (always, whatever the wire format carries) or
///   this provider wrote but cannot take back.
///
/// The same declaration answers capabilities: [`chat_capability`](Self::chat_capability)
/// is `Unsupported` for every capability the wire format cannot carry, so
/// the capability answers and the encoder agree by construction. The pass
/// runs after [`preflight`](crate::adapter::chat::preflight) because a capability
/// resolver may answer `Unknown` or `Supported` for something the wire
/// format still cannot carry.
///
/// [`text`](Self::text) carries the text of user and assistant turns and
/// nothing else; add what the wire format has a slot for.
///
/// # What the encoder still handles
///
/// The pass decides per field and per kind of content. Once a field or a
/// kind is declared, its details are the encoder's, under the same rules
/// (refuse what is semantic, warn about a hint it drops or changes):
///
/// - **Limits within a field:** how many `stop` sequences the API takes,
///   the range of `temperature`, an output cap above the model's maximum.
/// - **Tool details**, once tools are declared:
///   [`ToolDefinition::strict`](crate::ToolDefinition::strict) on an API
///   without a strict mode (warn that it is not honoured), and
///   [`ToolMessage::is_error`](crate::ToolMessage::is_error) on an API
///   whose tool results carry no error flag (say so in the result text,
///   with an `Approximated` warning).
/// - **Media details**, once a kind is declared: the media types, sizes and
///   sources (URL, inline data, a provider file) the API takes.
/// - **An assistant [refusal](crate::ContentBlock::Refusal)**, which a wire
///   format without a refusal form replays as text.
/// - **The adapter's own options** and their `extra` fields.
/// - **The wildcard arms** of the `#[non_exhaustive]` enums: content, tool
///   choices and fields added in later releases.
///
/// ```
/// use allama::adapter::chat::WireSupport;
/// use allama::{CapabilitySupport, ChatCapability, RequestField};
///
/// /// A gateway that takes role-tagged text, a system prompt and a token cap.
/// const WIRE: WireSupport =
///     WireSupport::text().fields([RequestField::System, RequestField::MaxOutputTokens]);
///
/// assert!(WIRE.carries(RequestField::System));
/// assert!(!WIRE.carries(RequestField::Temperature));
/// assert_eq!(
///     WIRE.chat_capability(ChatCapability::ToolCalls),
///     Some(CapabilitySupport::Unsupported)
/// );
/// // Streaming is not about the request: the adapter answers it.
/// assert_eq!(WIRE.chat_capability(ChatCapability::Streaming), None);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WireSupport {
    /// One bit per carried request field (see `field_bit`).
    fields: u32,
    /// One bit per carried kind of content (the `CONTENT_*` constants).
    content: u8,
}

impl WireSupport {
    /// A wire format that carries the text of user and assistant turns and
    /// nothing else: no request field, no media, no tools, no replay state.
    #[must_use]
    pub const fn text() -> Self {
        Self {
            fields: 0,
            content: 0,
        }
    }

    /// Also carry `field`. The embedding fields
    /// ([`Dimensions`](RequestField::Dimensions),
    /// [`InputType`](RequestField::InputType)) are not chat fields and are
    /// ignored.
    #[must_use]
    pub const fn field(mut self, field: RequestField) -> Self {
        self.fields |= field_bit(field);
        self
    }

    /// Also carry each of `fields`.
    #[must_use]
    pub const fn fields<const N: usize>(mut self, fields: [RequestField; N]) -> Self {
        let mut i = 0;
        while i < N {
            self = self.field(fields[i]);
            i += 1;
        }
        self
    }

    /// Also carry images in user input and tool results.
    #[must_use]
    pub const fn image_input(mut self) -> Self {
        self.content |= CONTENT_IMAGE_INPUT;
        self
    }

    /// Also carry per-image detail hints (with [`image_input`](Self::image_input)).
    #[must_use]
    pub const fn image_detail(mut self) -> Self {
        self.content |= CONTENT_IMAGE_DETAIL;
        self
    }

    /// Also carry documents in user input and tool results: files whose
    /// media type is not audio or video
    /// ([`FileKind::Document`](crate::FileKind::Document)).
    #[must_use]
    pub const fn file_input(mut self) -> Self {
        self.content |= CONTENT_FILE_INPUT;
        self
    }

    /// Also carry audio files in user input and tool results.
    #[must_use]
    pub const fn audio_input(mut self) -> Self {
        self.content |= CONTENT_AUDIO_INPUT;
        self
    }

    /// Also carry video files in user input and tool results.
    #[must_use]
    pub const fn video_input(mut self) -> Self {
        self.content |= CONTENT_VIDEO_INPUT;
        self
    }

    /// Also carry assistant images replayed in later turns.
    #[must_use]
    pub const fn image_replay(mut self) -> Self {
        self.content |= CONTENT_IMAGE_REPLAY;
        self
    }

    /// Also carry assistant reasoning replayed in later turns. The encoder
    /// still replays only reasoning it can take back (its own signature or
    /// state) and drops the rest with a warning.
    #[must_use]
    pub const fn reasoning_replay(mut self) -> Self {
        self.content |= CONTENT_REASONING_REPLAY;
        self
    }

    /// Also carry provider content: [`Other`](crate::OtherContent) content
    /// and provider tool calls and results of this provider's namespace, and
    /// `Other` input parts without an origin (the caller's raw parts). The
    /// encoder sends them; another provider's are dropped either way.
    #[must_use]
    pub const fn provider_content(mut self) -> Self {
        self.content |= CONTENT_PROVIDER;
        self
    }

    /// Whether the wire format has a slot for `field`.
    #[must_use]
    pub const fn carries(&self, field: RequestField) -> bool {
        let bit = field_bit(field);
        bit != 0 && self.fields & bit == bit
    }

    const fn carries_content(&self, content: u8) -> bool {
        self.content & content == content
    }

    /// `Some(Unsupported)` for a capability the wire format cannot carry,
    /// and `None` for one it can or that is not about the request
    /// (streaming, stream usage, image and reasoning output): the model
    /// decides those, so the adapter answers them.
    ///
    /// Use it first in [`chat_capability`](crate::ChatProvider::chat_capability),
    /// so the answers match what [`Prepared::degrade_to`](crate::adapter::chat::Prepared::degrade_to)
    /// refuses and drops:
    ///
    /// ```
    /// # use allama::adapter::chat::WireSupport;
    /// # use allama::{CapabilitySupport, ChatCapability};
    /// # const WIRE: WireSupport = WireSupport::text();
    /// fn chat_capability(_model: &str, capability: ChatCapability) -> CapabilitySupport {
    ///     WIRE.chat_capability(capability).unwrap_or(match capability {
    ///         ChatCapability::Streaming | ChatCapability::NativeStreaming => {
    ///             CapabilitySupport::Supported
    ///         }
    ///         _ => CapabilitySupport::Unknown,
    ///     })
    /// }
    /// # assert_eq!(chat_capability("m", ChatCapability::ToolCalls), CapabilitySupport::Unsupported);
    /// ```
    #[must_use]
    pub const fn chat_capability(&self, capability: ChatCapability) -> Option<CapabilitySupport> {
        let tools = self.carries(RequestField::Tools);
        let carried = match capability {
            ChatCapability::ToolCalls | ChatCapability::ParallelToolCalls => tools,
            ChatCapability::ForcedToolChoice => tools && self.carries(RequestField::ToolChoice),
            ChatCapability::StructuredOutput => self.carries(RequestField::ResponseFormat),
            ChatCapability::ReasoningConfig => self.carries(RequestField::Reasoning),
            ChatCapability::PromptCache => self.carries(RequestField::Cache),
            ChatCapability::ImageInput => self.carries_content(CONTENT_IMAGE_INPUT),
            ChatCapability::ImageDetail => {
                self.carries_content(CONTENT_IMAGE_INPUT | CONTENT_IMAGE_DETAIL)
            }
            ChatCapability::FileInput => self.carries_content(CONTENT_FILE_INPUT),
            ChatCapability::AudioInput => self.carries_content(CONTENT_AUDIO_INPUT),
            ChatCapability::VideoInput => self.carries_content(CONTENT_VIDEO_INPUT),
            ChatCapability::ImageReplay => self.carries_content(CONTENT_IMAGE_REPLAY),
            ChatCapability::ReasoningReplay => self.carries_content(CONTENT_REASONING_REPLAY),
            ChatCapability::Streaming
            | ChatCapability::NativeStreaming
            | ChatCapability::StreamUsage
            | ChatCapability::ImageOutput
            | ChatCapability::ReasoningOutput => true,
        };
        if carried {
            None
        } else {
            Some(CapabilitySupport::Unsupported)
        }
    }
}

const CONTENT_IMAGE_INPUT: u8 = 1;
const CONTENT_IMAGE_DETAIL: u8 = 1 << 1;
const CONTENT_FILE_INPUT: u8 = 1 << 2;
const CONTENT_IMAGE_REPLAY: u8 = 1 << 3;
const CONTENT_REASONING_REPLAY: u8 = 1 << 4;
const CONTENT_PROVIDER: u8 = 1 << 5;
const CONTENT_AUDIO_INPUT: u8 = 1 << 6;
const CONTENT_VIDEO_INPUT: u8 = 1 << 7;

/// The bit of a chat request field; `0` for a field that is not one.
const fn field_bit(field: RequestField) -> u32 {
    let index = match field {
        RequestField::Temperature => 0,
        RequestField::MaxOutputTokens => 1,
        RequestField::TopP => 2,
        RequestField::Stop => 3,
        RequestField::FrequencyPenalty => 4,
        RequestField::PresencePenalty => 5,
        RequestField::Seed => 6,
        RequestField::Tools => 7,
        RequestField::ToolChoice => 8,
        RequestField::ParallelToolCalls => 9,
        RequestField::ResponseFormat => 10,
        RequestField::Reasoning => 11,
        RequestField::System => 12,
        RequestField::Cache => 13,
        RequestField::TopK => 14,
        RequestField::Dimensions | RequestField::InputType | RequestField::Unknown => return 0,
    };
    1 << index
}

/// Apply `wire` to `request` (see [`WireSupport`]): refuse what the wire
/// format cannot carry and must not lose, and remove the rest with the
/// warnings it calls for. `reads_namespace` tells this provider's content
/// from another's.
pub(crate) fn degrade(
    wire: &WireSupport,
    reads_namespace: &dyn Fn(&str) -> bool,
    request: &mut Cow<'_, ChatRequest>,
    d: &mut Degradations,
) -> Result<()> {
    let name = d.provider().to_owned();
    let not_supported = |what: &str| format!("{what} is not supported by `{name}`");
    let carries = |field| wire.carries(field);

    // A disabled tool choice the wire format cannot pass on is honoured by
    // leaving the tools out: exactly what the request asks for.
    if request.tool_choice == Some(ToolChoice::Disabled)
        && !(carries(RequestField::Tools) && carries(RequestField::ToolChoice))
    {
        let request = request.to_mut();
        request.tools = Vec::new();
        request.tool_choice = None;
    }

    // Semantic fields first: they fail under every policy, so a strict
    // policy never reports a hint in their place.
    if !carries(RequestField::Tools) && !request.tools.is_empty() {
        return Err(d.unsupported(RequestField::Tools, not_supported("tool calling")));
    }
    match request.tool_choice {
        Some(ToolChoice::Auto) if !carries(RequestField::ToolChoice) => {
            request.to_mut().tool_choice = None;
        }
        Some(ToolChoice::Auto | ToolChoice::Disabled) | None => {}
        Some(_) if carries(RequestField::Tools) && carries(RequestField::ToolChoice) => {}
        Some(_) => {
            return Err(d.unsupported(
                RequestField::ToolChoice,
                not_supported("forced tool choice"),
            ));
        }
    }
    if !carries(RequestField::ResponseFormat) {
        match request.response_format {
            None => {}
            Some(ResponseFormat::Text) => request.to_mut().response_format = None,
            Some(_) => {
                return Err(d.unsupported(
                    RequestField::ResponseFormat,
                    not_supported("structured output"),
                ));
            }
        }
    }
    if request.max_output_tokens.is_some() && !carries(RequestField::MaxOutputTokens) {
        return Err(d.unsupported(
            RequestField::MaxOutputTokens,
            not_supported("an output cap"),
        ));
    }
    if !request.system.is_empty() && !carries(RequestField::System) {
        return Err(d.unsupported(RequestField::System, not_supported("a system prompt")));
    }
    refuse_content(wire, reads_namespace, request, d, &not_supported)?;

    // Hints: removed with a warning, or an error under a strict policy.
    let hint = |d: &mut Degradations, field: RequestField| {
        d.unsupported_hint(
            field,
            format!(
                "{}; `{field}` was not sent",
                not_supported(&format!("`{field}`"))
            ),
        )
    };
    let fields = [
        (RequestField::Temperature, request.temperature.is_some()),
        (RequestField::TopP, request.top_p.is_some()),
        (RequestField::TopK, request.top_k.is_some()),
        (RequestField::Stop, !request.stop.is_empty()),
        (
            RequestField::FrequencyPenalty,
            request.frequency_penalty.is_some(),
        ),
        (
            RequestField::PresencePenalty,
            request.presence_penalty.is_some(),
        ),
        (RequestField::Seed, request.seed.is_some()),
    ];
    for (field, set) in fields {
        if set && !carries(field) {
            hint(d, field)?;
            let request = request.to_mut();
            match field {
                RequestField::Temperature => request.temperature = None,
                RequestField::TopP => request.top_p = None,
                RequestField::TopK => request.top_k = None,
                RequestField::Stop => request.stop = Vec::new(),
                RequestField::FrequencyPenalty => request.frequency_penalty = None,
                RequestField::PresencePenalty => request.presence_penalty = None,
                _ => request.seed = None,
            }
        }
    }
    if let Some(parallel) = request.parallel_tool_calls
        && !carries(RequestField::ParallelToolCalls)
    {
        // `true` is permission the model has anyway; only `false` asks for
        // something the wire format cannot pass on.
        if !parallel {
            d.unsupported_hint(
                RequestField::ParallelToolCalls,
                format!(
                    "{}; `false` was not sent, and the model may call several tools at once",
                    not_supported("`parallel_tool_calls`")
                ),
            )?;
        }
        request.to_mut().parallel_tool_calls = None;
    }
    if let Some(reasoning) = request.reasoning
        && !carries(RequestField::Reasoning)
    {
        // Asking not to reason is honoured by a wire format without
        // reasoning controls.
        if reasoning.effort != Some(ReasoningEffort::None) {
            d.unsupported_hint(
                RequestField::Reasoning,
                format!(
                    "{}; `reasoning` was not sent",
                    not_supported("reasoning configuration")
                ),
            )?;
        }
        request.to_mut().reasoning = None;
    }
    if (request.cache.is_some() || request.cache_breakpoints().next().is_some())
        && !carries(RequestField::Cache)
    {
        d.unsupported_hint(
            RequestField::Cache,
            format!("{}; `cache` was not sent", not_supported("prompt caching")),
        )?;
        strip_cache(request.to_mut());
    }
    if wire.carries_content(CONTENT_IMAGE_INPUT) && !wire.carries_content(CONTENT_IMAGE_DETAIL) {
        let mut stripped = false;
        for (message, block, part) in input_parts(request) {
            if let ContentPart::Image(image) = part
                && image.detail.is_some()
            {
                d.coded_hint(
                    Subject::Content {
                        message,
                        block: Some(block),
                    },
                    format!("{}; the detail was not sent", not_supported("image detail")),
                    "unsupported.image_detail",
                )?;
                stripped = true;
            }
        }
        if stripped {
            strip_image_detail(request.to_mut());
        }
    }

    drop_content(wire, reads_namespace, request, d);
    Ok(())
}

/// Refuse the content the wire format cannot carry and the request must
/// not lose: tool turns, media, and the caller's raw parts.
fn refuse_content(
    wire: &WireSupport,
    reads_namespace: &dyn Fn(&str) -> bool,
    request: &ChatRequest,
    d: &Degradations,
    not_supported: &dyn Fn(&str) -> String,
) -> Result<()> {
    let content = |message, block| Subject::Content { message, block };
    let tools = wire.carries(RequestField::Tools);
    for (index, message) in request.messages.iter().enumerate() {
        match message {
            Message::Tool(_) if !tools => {
                return Err(d.reject(content(index, None), not_supported("tool results")));
            }
            Message::Assistant(m) => {
                for (block, b) in m.content.iter().enumerate() {
                    let what = match b {
                        ContentBlock::ToolCall(_) if !tools => "tool calls",
                        ContentBlock::Image(_) if !wire.carries_content(CONTENT_IMAGE_REPLAY) => {
                            "replaying assistant images"
                        }
                        _ => continue,
                    };
                    return Err(d.reject(content(index, Some(block)), not_supported(what)));
                }
            }
            _ => {}
        }
    }
    for (index, block, part) in input_parts(request) {
        let what = match part {
            ContentPart::Image(_) if !wire.carries_content(CONTENT_IMAGE_INPUT) => {
                Cow::Borrowed("image input")
            }
            ContentPart::File(file) => match file.kind() {
                FileKind::Audio if !wire.carries_content(CONTENT_AUDIO_INPUT) => {
                    Cow::Borrowed("audio input")
                }
                FileKind::Video if !wire.carries_content(CONTENT_VIDEO_INPUT) => {
                    Cow::Borrowed("video input")
                }
                FileKind::Document if !wire.carries_content(CONTENT_FILE_INPUT) => {
                    Cow::Borrowed("file input")
                }
                _ => continue,
            },
            // The caller's raw part: it must be sent verbatim or refused.
            ContentPart::Other(other)
                if !wire.carries_content(CONTENT_PROVIDER) && !foreign(other, reads_namespace) =>
            {
                Cow::Owned(format!("`{}` content", other.type_name))
            }
            _ => continue,
        };
        return Err(d.reject(content(index, Some(block)), not_supported(&what)));
    }
    Ok(())
}

/// Drop the replay state the wire format cannot carry, with
/// their warnings, recording which blocks are left so later warnings name
/// the caller's.
fn drop_content(
    wire: &WireSupport,
    reads_namespace: &dyn Fn(&str) -> bool,
    request: &mut Cow<'_, ChatRequest>,
    d: &mut Degradations,
) {
    let name = d.provider().to_owned();
    let reasoning = wire.carries_content(CONTENT_REASONING_REPLAY);
    let provider = wire.carries_content(CONTENT_PROVIDER);
    for index in 0..request.messages.len() {
        let subject = |block| Subject::Content {
            message: index,
            block,
        };
        // Which blocks go, each with its warning, in order.
        let mut dropped = Vec::new();
        let message = &request.messages[index];
        match message {
            Message::Assistant(m) => {
                for (block, b) in m.content.iter().enumerate() {
                    let drop = match b {
                        ContentBlock::Reasoning(_) if !reasoning => Some((
                            ReplayState::Reasoning,
                            format!("`{name}` takes no reasoning back; it was dropped"),
                        )),
                        ContentBlock::Other(_)
                        | ContentBlock::ProviderToolCall(_)
                        | ContentBlock::ProviderToolResult(_) => provider_drop(
                            b.type_name(),
                            b.origin(),
                            provider,
                            reads_namespace,
                            &name,
                        ),
                        _ => None,
                    };
                    if let Some(drop) = drop {
                        dropped.push((block, drop));
                    }
                }
            }
            Message::User(_) | Message::Tool(_) => {
                for (block, part) in message_parts(message).iter().enumerate() {
                    if let ContentPart::Other(other) = part
                        && foreign(other, reads_namespace)
                    {
                        let drop = provider_drop(
                            &other.type_name,
                            other.origin.as_deref(),
                            provider,
                            reads_namespace,
                            &name,
                        );
                        dropped.extend(drop.map(|drop| (block, drop)));
                    }
                }
            }
        }
        if dropped.is_empty() {
            continue;
        }
        // The warnings first: they name the blocks as this pass received
        // them.
        for (block, (state, message)) in &dropped {
            d.dropped_replay_state(*state, subject(Some(*block)), message.clone());
        }
        let message = &mut request.to_mut().messages[index];
        let len = message_len(message);
        let kept: Vec<usize> = (0..len)
            .filter(|block| !dropped.iter().any(|(gone, _)| gone == block))
            .collect();
        d.record_kept_blocks(index, len, kept.clone());
        retain_blocks(message, &kept);
    }
}

/// The drop of an `Other` or provider tool block, or `None` when it stays
/// for the encoder: another provider's always goes, and this provider's
/// goes when the wire format carries no provider content. A block without
/// an origin in an assistant turn is replay state of unknown origin.
fn provider_drop(
    type_name: &str,
    origin: Option<&str>,
    carried: bool,
    reads_namespace: &dyn Fn(&str) -> bool,
    name: &str,
) -> Option<(ReplayState, String)> {
    match origin {
        Some(origin) if !reads_namespace(origin) => Some((
            ReplayState::ForeignContent,
            format!(
                "`{type_name}` content from `{origin}` can be replayed only to its origin; it was dropped"
            ),
        )),
        _ if carried => None,
        _ => Some((
            ReplayState::Content,
            format!("`{type_name}` content cannot be replayed to `{name}`; it was dropped"),
        )),
    }
}

/// Whether `other` is another provider's.
fn foreign(other: &OtherContent, reads_namespace: &dyn Fn(&str) -> bool) -> bool {
    other
        .origin
        .as_deref()
        .is_some_and(|origin| !reads_namespace(origin))
}

/// `(message, part, content)` of every part of user input and tool results.
fn input_parts(request: &ChatRequest) -> impl Iterator<Item = (usize, usize, &ContentPart)> {
    request
        .messages
        .iter()
        .enumerate()
        .flat_map(|(index, message)| {
            message_parts(message)
                .iter()
                .enumerate()
                .map(move |(part, content)| (index, part, content))
        })
}

/// The parts of a user message or a tool result; none for text.
pub(crate) fn message_parts(message: &Message) -> &[ContentPart] {
    match message {
        Message::User(m) => match &m.content {
            UserContent::Parts(parts) => parts,
            _ => &[],
        },
        Message::Tool(m) => match &m.content {
            ToolResultContent::Parts(parts) => parts,
            _ => &[],
        },
        Message::Assistant(_) => &[],
    }
}

/// How many blocks or parts `message` holds.
fn message_len(message: &Message) -> usize {
    match message {
        Message::Assistant(m) => m.content.len(),
        other => message_parts(other).len(),
    }
}

/// Keep the blocks or parts of `message` at `kept`.
fn retain_blocks(message: &mut Message, kept: &[usize]) {
    fn retain<T>(items: &mut Vec<T>, kept: &[usize]) {
        let mut index = 0;
        items.retain(|_| {
            let keep = kept.contains(&index);
            index += 1;
            keep
        });
    }
    match message {
        Message::Assistant(m) => retain(&mut m.content, kept),
        Message::User(m) => {
            if let UserContent::Parts(parts) = &mut m.content {
                retain(parts, kept);
            }
        }
        Message::Tool(m) => {
            if let ToolResultContent::Parts(parts) = &mut m.content {
                retain(parts, kept);
            }
        }
    }
}

#[cfg(test)]
mod tests;
