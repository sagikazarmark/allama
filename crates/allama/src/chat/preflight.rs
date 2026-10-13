use std::borrow::Cow;

use crate::{
    ChatCapability, ChatProvider, ChatRequest, ContentBlock, ContentPart, Degradations, FileKind,
    Message, ReasoningEffort, ReplayState, RequestField, ResponseFormat, Result, Subject,
    ToolChoice, ToolResultContent, UserContent,
};

/// Check `request` against the capabilities `provider` reports for the
/// request's model, before it is encoded and sent.
///
/// Every capability reported [`Unsupported`](crate::CapabilitySupport::Unsupported)
/// is applied as ADR 0003 prescribes:
///
/// - **Semantic features** fail with [`Error::Unsupported`](crate::Error::Unsupported)
///   under every policy: tools ([`ToolCalls`](ChatCapability::ToolCalls)), a
///   forced tool choice ([`ForcedToolChoice`](ChatCapability::ForcedToolChoice)),
///   a JSON Schema response format
///   ([`StructuredOutput`](ChatCapability::StructuredOutput); JSON mode is
///   left to the adapter),
///   image, document, audio or video input
///   ([`ImageInput`](ChatCapability::ImageInput),
///   [`FileInput`](ChatCapability::FileInput),
///   [`AudioInput`](ChatCapability::AudioInput),
///   [`VideoInput`](ChatCapability::VideoInput)), and assistant images
///   ([`ImageReplay`](ChatCapability::ImageReplay)).
/// - **Hints** are removed from `request` with an
///   [`Unsupported`](crate::WarningKind::Unsupported) warning, or fail under
///   [`UnsupportedPolicy::Error`](crate::UnsupportedPolicy::Error): reasoning
///   configuration ([`ReasoningConfig`](ChatCapability::ReasoningConfig)),
///   prompt-cache settings and breakpoints
///   ([`PromptCache`](ChatCapability::PromptCache)), and image detail
///   ([`ImageDetail`](ChatCapability::ImageDetail)). Asking a model without
///   reasoning controls not to reason (`ReasoningEffort::None`) is already
///   honored, so that configuration is removed without a warning.
/// - **`parallel_tool_calls`** is left as it is, whatever
///   [`ParallelToolCalls`](ChatCapability::ParallelToolCalls) answers:
///   `true` is permission, never a degradation, and `false` is honored by a
///   model that makes one call per turn. An adapter whose wire format has
///   no such field drops `true` silently and reports `false` as a hint.
/// - **Assistant reasoning blocks** are removed with a
///   [`Dropped`](crate::WarningKind::Dropped) warning when
///   [`ReasoningReplay`](ChatCapability::ReasoningReplay) is unsupported:
///   they are replay state, not a request feature. The warning has the
///   code `dropped.reasoning_replay` (and an image detail hint
///   `unsupported.image_detail`), so its event recurs at `DEBUG` on later
///   turns; hint warnings about request fields get `unsupported.<field>`.
///
/// [`Unknown`](crate::CapabilitySupport::Unknown) never blocks a request and never
/// removes anything. Preflight records each capability the request relies
/// on that the provider answered `Unknown`, with the first subject that
/// relies on it in its `Degradations`, so a [`Validation`](crate::Validation)
/// can report what it could not know
/// ([`relied_on_unknown`](crate::Validation::relied_on_unknown)).
///
/// **`Unknown` means "send it and let the provider decide", never "drop
/// it"** (ADR 0003, principle 2). Preflight leaves an `Unknown` feature in
/// the request, so the encoder sends every request field and content part,
/// or reports it through `degradations` (a warning, or a refusal before
/// sending); skipping one with a wildcard arm drops it silently. A field
/// the wire format cannot carry is answered `Unsupported` for every model,
/// and an OpenAI-compatible preset answers `Unsupported` for a field its
/// server documents it ignores (or encodes an equivalent the server
/// honors). `allama_conformance::capabilities::assert_capabilities_honest`
/// checks this for every answer, a model the provider does not know
/// included.
///
/// `degradations` carries the policy and collects the warnings; build it
/// from `request.unsupported_policy` (unset means
/// [`Warn`](crate::UnsupportedPolicy::Warn)) and keep using it while encoding, so
/// preflight and encoding warnings are reported together. `request` stays
/// borrowed unless something is removed. Messages are never removed, but
/// removing reasoning blocks shifts the block indices of the assistant
/// messages that held them. Preflight records the shift in `degradations`,
/// which translates the block index of every
/// [`Subject::Content`] warning recorded after it back to the caller's
/// request, so report indices relative to the preflighted `request` and do
/// not translate them yourself (see [`Degradations`](crate::adapter::Degradations#indices)).
///
/// Adapters reach it through [`chat::prepare`](crate::adapter::chat::prepare), which
/// runs it first in `chat` and `chat_stream` and keeps its warnings with the
/// request; the default
/// [`validate_chat_request`](ChatProvider::validate_chat_request) runs it on
/// a dry run. Call it directly only to check a request against a second
/// provider's answers outside a [`Prepared`](crate::adapter::chat::Prepared) request:
///
/// ```
/// use std::borrow::Cow;
///
/// use allama::adapter::Degradations;
/// use allama::{ChatProvider, ChatRequest, Result, Warning};
///
/// fn check<'a, P: ChatProvider>(
///     provider: &P,
///     request: &'a ChatRequest,
/// ) -> Result<(Cow<'a, ChatRequest>, Vec<Warning>)> {
///     let mut degradations = Degradations::new(
///         provider.provider_name().to_owned(),
///         request.unsupported_policy.unwrap_or_default(),
///     );
///     let mut request = Cow::Borrowed(request);
///     allama::adapter::chat::preflight(provider, &mut request, &mut degradations)?;
///     Ok((request, degradations.into_warnings()))
/// }
/// ```
///
/// # Errors
///
/// Returns [`Error::Unsupported`](crate::Error::Unsupported) for the first
/// unsupported semantic feature, or, under
/// [`UnsupportedPolicy::Error`](crate::UnsupportedPolicy::Error), the first
/// unsupported hint.
pub fn preflight<P: ChatProvider + ?Sized>(
    provider: &P,
    request: &mut Cow<'_, ChatRequest>,
    degradations: &mut Degradations,
) -> Result<()> {
    let model = request.model.clone();
    let answer = |capability| provider.chat_capability(&model, capability);
    // Capability answers describe the endpoint, not the model (ADR 0015), so
    // the messages name the provider.
    let name = provider.provider_name();
    let not_supported = |what: &str| format!("{what} is not supported by `{name}`");
    // A hint's message goes on to say what happened to it; a refusal under
    // `UnsupportedPolicy::Error` keeps only the part before `"; "`.
    let not_sent = |what: &str, field: RequestField| {
        format!("{}; `{field}` was not sent", not_supported(what))
    };
    let field = |field| move || Subject::Field { field };
    let content = |(message, block): (usize, usize)| {
        move || Subject::Content {
            message,
            block: Some(block),
        }
    };
    let d = degradations;

    // Semantic features first: they fail under every policy, so a strict
    // policy never reports a hint in their place.
    if !request.tools.is_empty()
        && d.refuses(
            ChatCapability::ToolCalls,
            answer(ChatCapability::ToolCalls),
            field(RequestField::Tools),
        )
    {
        return Err(d.unsupported(RequestField::Tools, not_supported("tool calling")));
    }
    if matches!(
        request.tool_choice,
        Some(ToolChoice::Required | ToolChoice::Specific { .. })
    ) && d.refuses(
        ChatCapability::ForcedToolChoice,
        answer(ChatCapability::ForcedToolChoice),
        field(RequestField::ToolChoice),
    ) {
        return Err(d.unsupported(
            RequestField::ToolChoice,
            not_supported("forced tool choice"),
        ));
    }
    if matches!(
        request.response_format,
        Some(ResponseFormat::JsonSchema { .. })
    ) && d.refuses(
        ChatCapability::StructuredOutput,
        answer(ChatCapability::StructuredOutput),
        field(RequestField::ResponseFormat),
    ) {
        return Err(d.unsupported(
            RequestField::ResponseFormat,
            not_supported("structured output"),
        ));
    }
    let uses = ContentUse::of(request);
    if let Some(at) = uses.image_input
        && d.refuses(
            ChatCapability::ImageInput,
            answer(ChatCapability::ImageInput),
            content(at),
        )
    {
        return Err(d.reject(content(at)(), not_supported("image input")));
    }
    for (at, capability, what) in [
        (uses.file_input, ChatCapability::FileInput, "file input"),
        (uses.audio_input, ChatCapability::AudioInput, "audio input"),
        (uses.video_input, ChatCapability::VideoInput, "video input"),
    ] {
        if let Some(at) = at
            && d.refuses(capability, answer(capability), content(at))
        {
            return Err(d.reject(content(at)(), not_supported(what)));
        }
    }
    if let Some(at) = uses.image_replay
        && d.refuses(
            ChatCapability::ImageReplay,
            answer(ChatCapability::ImageReplay),
            content(at),
        )
    {
        return Err(d.reject(content(at)(), not_supported("replaying assistant images")));
    }

    // Hints: removed with a warning, or an error under a strict policy.
    // `parallel_tool_calls` is not one: `true` is permission and `false` is
    // honored by a model that makes one call per turn.
    if let Some(reasoning) = request.reasoning
        && d.refuses(
            ChatCapability::ReasoningConfig,
            answer(ChatCapability::ReasoningConfig),
            field(RequestField::Reasoning),
        )
    {
        if reasoning.effort != Some(ReasoningEffort::None) {
            d.unsupported_hint(
                RequestField::Reasoning,
                not_sent("reasoning configuration", RequestField::Reasoning),
            )?;
        }
        request.to_mut().reasoning = None;
    }
    if (request.cache.is_some() || request.cache_breakpoints().next().is_some())
        && d.refuses(
            ChatCapability::PromptCache,
            answer(ChatCapability::PromptCache),
            field(RequestField::Cache),
        )
    {
        d.unsupported_hint(
            RequestField::Cache,
            not_sent("prompt caching", RequestField::Cache),
        )?;
        strip_cache(request.to_mut());
    }
    if let Some(&first) = uses.image_detail.first()
        && d.refuses(
            ChatCapability::ImageDetail,
            answer(ChatCapability::ImageDetail),
            content(first),
        )
    {
        for &(message, block) in &uses.image_detail {
            d.coded_hint(
                Subject::Content {
                    message,
                    block: Some(block),
                },
                format!("{}; the detail was not sent", not_supported("image detail")),
                "unsupported.image_detail",
            )?;
        }
        strip_image_detail(request.to_mut());
    }

    // Replay state the model cannot take back.
    if let Some(&first) = uses.reasoning.first()
        && d.refuses(
            ChatCapability::ReasoningReplay,
            answer(ChatCapability::ReasoningReplay),
            content(first),
        )
    {
        // Replay state: it recurs on every turn of the conversation, so the
        // warning is coded, logged at `WARN` once per model, and one per
        // request however many messages hold reasoning. Reasoning this
        // provider wrote is not reported: dropping its own reasoning is
        // what the endpoint does, not a degradation (ADR 0003, amended).
        for &(message, block) in &uses.reasoning {
            if written_by(provider, &request.messages[message], block) {
                continue;
            }
            d.dropped_replay_state(
                ReplayState::Reasoning,
                Subject::Content {
                    message,
                    block: Some(block),
                },
                format!("`{name}` takes no reasoning back; it was dropped"),
            );
        }
        for (index, message) in request.to_mut().messages.iter_mut().enumerate() {
            let Message::Assistant(m) = message else {
                continue;
            };
            if !m
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::Reasoning(_)))
            {
                continue;
            }
            let kept = m
                .content
                .iter()
                .enumerate()
                .filter(|(_, block)| !matches!(block, ContentBlock::Reasoning(_)))
                .map(|(block, _)| block)
                .collect();
            // After the warnings above, which use the caller's indices.
            d.record_kept_blocks(index, m.content.len(), kept);
            m.content
                .retain(|block| !matches!(block, ContentBlock::Reasoning(_)));
        }
    }
    Ok(())
}

/// Whether reasoning block `block` of `message` carries state only under
/// namespaces `provider` reads: it came from this provider. A block with no
/// provider data has no known origin.
fn written_by<P: ChatProvider + ?Sized>(provider: &P, message: &Message, block: usize) -> bool {
    let Message::Assistant(m) = message else {
        return false;
    };
    let Some(ContentBlock::Reasoning(reasoning)) = m.content.get(block) else {
        return false;
    };
    let mut namespaces = reasoning.provider_data.namespaces().peekable();
    namespaces.peek().is_some()
        && namespaces.all(|(namespace, _)| provider.reads_namespace(namespace))
}

/// Capability-gated content found in a request's messages.
#[derive(Default)]
struct ContentUse {
    /// `(message, part)` of the first input image.
    image_input: Option<(usize, usize)>,
    /// `(message, part)` of the first input document.
    file_input: Option<(usize, usize)>,
    /// `(message, part)` of the first input audio file.
    audio_input: Option<(usize, usize)>,
    /// `(message, part)` of the first input video file.
    video_input: Option<(usize, usize)>,
    /// `(message, block)` of the first assistant image.
    image_replay: Option<(usize, usize)>,
    /// `(message, part)` of every input image with a detail hint.
    image_detail: Vec<(usize, usize)>,
    /// `(message, block)` of every assistant reasoning block.
    reasoning: Vec<(usize, usize)>,
}

impl ContentUse {
    fn of(request: &ChatRequest) -> Self {
        let mut uses = Self::default();
        for (index, message) in request.messages.iter().enumerate() {
            if let Message::Assistant(m) = message {
                for (block, content) in m.content.iter().enumerate() {
                    match content {
                        ContentBlock::Image(_) => {
                            uses.image_replay.get_or_insert((index, block));
                        }
                        ContentBlock::Reasoning(_) => uses.reasoning.push((index, block)),
                        _ => {}
                    }
                }
                continue;
            }
            for (part, content) in input_parts(message).iter().enumerate() {
                match content {
                    ContentPart::Image(image) => {
                        uses.image_input.get_or_insert((index, part));
                        if image.detail.is_some() {
                            uses.image_detail.push((index, part));
                        }
                    }
                    ContentPart::File(file) => {
                        let first = match file.kind() {
                            FileKind::Audio => &mut uses.audio_input,
                            FileKind::Video => &mut uses.video_input,
                            _ => &mut uses.file_input,
                        };
                        first.get_or_insert((index, part));
                    }
                    _ => {}
                }
            }
        }
        uses
    }
}

/// The content parts of a user message or tool result.
fn input_parts(message: &Message) -> &[ContentPart] {
    match message {
        Message::User(m) => match &m.content {
            UserContent::Parts(parts) => parts,
            _ => &[],
        },
        Message::Tool(m) => match &m.content {
            ToolResultContent::Parts(parts) => parts,
            _ => &[],
        },
        _ => &[],
    }
}

pub(super) fn strip_image_detail(request: &mut ChatRequest) {
    for message in &mut request.messages {
        let parts = match message {
            Message::User(m) => match &mut m.content {
                UserContent::Parts(parts) => parts,
                _ => continue,
            },
            Message::Tool(m) => match &mut m.content {
                ToolResultContent::Parts(parts) => parts,
                _ => continue,
            },
            _ => continue,
        };
        for part in parts {
            if let ContentPart::Image(image) = part {
                image.detail = None;
            }
        }
    }
}

pub(super) fn strip_cache(request: &mut ChatRequest) {
    request.cache = None;
    for prompt in &mut request.system {
        prompt.cache = None;
    }
    for tool in &mut request.tools {
        tool.cache = None;
    }
    for message in &mut request.messages {
        match message {
            Message::User(m) => m.cache = None,
            Message::Assistant(m) => m.cache = None,
            Message::Tool(m) => m.cache = None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::CapabilitySupport;
    use serde_json::json;

    use super::*;
    use crate::mock::MockProvider;
    use crate::{
        CacheBreakpoint, Error, Image, ImageDetail, OtherContent, PromptCache, SystemPrompt,
        ToolDefinition, UnsupportedPolicy, Warning, WarningKind,
    };

    fn without(capability: ChatCapability) -> MockProvider {
        MockProvider::empty().with_chat_capability(capability, CapabilitySupport::Unsupported)
    }

    fn run(provider: &MockProvider, request: &ChatRequest) -> Result<(ChatRequest, Degradations)> {
        let mut d = Degradations::new("mock", request.unsupported_policy.unwrap_or_default());
        let mut checked = Cow::Borrowed(request);
        preflight(provider, &mut checked, &mut d)?;
        Ok((checked.into_owned(), d))
    }

    fn image(detail: Option<ImageDetail>) -> ContentPart {
        let image = Image::url("https://example.com/a.png");
        ContentPart::Image(match detail {
            Some(detail) => image.with_detail(detail),
            None => image,
        })
    }

    fn tool() -> ToolDefinition {
        ToolDefinition::new("t", json!({"type": "object"}))
    }

    /// One request that uses every capability preflight checks.
    fn everything() -> ChatRequest {
        ChatRequest::new("m")
            .system(SystemPrompt::new("s").with_cache(CacheBreakpoint::new()))
            .user(vec![
                ContentPart::text("q"),
                image(Some(ImageDetail::Low)),
                ContentPart::file_base64("application/pdf", "JVBERi0="),
            ])
            .message(Message::assistant_blocks(vec![
                ContentBlock::reasoning("think"),
                ContentBlock::text("a"),
                ContentBlock::Image(Image::url("https://example.com/b.png")),
            ]))
            .user("q2")
            .tools([tool()])
            .tool_choice(ToolChoice::Required)
            .parallel_tool_calls(true)
            .response_format(ResponseFormat::json_schema(json!({"type": "object"})))
            .reasoning(ReasoningEffort::High)
            .cache(PromptCache::new().key("k"))
    }

    #[test]
    fn unknown_capabilities_never_block_or_remove() {
        for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
            let request = everything().unsupported_policy(policy);
            let mut d = Degradations::new("mock", policy);
            let mut checked = Cow::Borrowed(&request);
            preflight(&MockProvider::empty(), &mut checked, &mut d).unwrap();
            assert!(matches!(checked, Cow::Borrowed(_)));
            assert!(d.warnings().is_empty());
        }
    }

    /// `FileInput` means documents; audio and video have their own
    /// capabilities.
    #[test]
    fn files_are_checked_by_their_kind() {
        let request = |media_type: &str| {
            ChatRequest::new("m").message(Message::user_parts(vec![
                ContentPart::text("What is this?"),
                ContentPart::file_base64(media_type, "AA"),
            ]))
        };
        for (capability, media_type, what) in [
            (ChatCapability::FileInput, "application/pdf", "file input"),
            (ChatCapability::AudioInput, "audio/wav", "audio input"),
            (ChatCapability::VideoInput, "video/mp4", "video input"),
        ] {
            let err = run(&without(capability), &request(media_type)).unwrap_err();
            assert!(err.message().contains(what), "{capability:?}: {err}");
            // The other kinds' capabilities do not decide this file.
            for other in [
                ChatCapability::FileInput,
                ChatCapability::AudioInput,
                ChatCapability::VideoInput,
            ] {
                if other != capability {
                    assert!(run(&without(other), &request(media_type)).is_ok());
                }
            }
        }
    }

    #[test]
    fn supported_capabilities_pass_unchanged() {
        let provider =
            MockProvider::empty().with_supported_chat_capabilities(ChatCapability::ALL.to_vec());
        let request = everything().unsupported_policy(UnsupportedPolicy::Error);
        let (checked, d) = run(&provider, &request).unwrap();
        assert_eq!(checked, request);
        assert!(d.warnings().is_empty());
    }

    #[test]
    fn semantic_features_fail_under_every_policy() {
        let cases = [
            (ChatCapability::ToolCalls, "tool calling"),
            (ChatCapability::ForcedToolChoice, "forced tool choice"),
            (ChatCapability::StructuredOutput, "structured output"),
            (ChatCapability::ImageInput, "image input"),
            (ChatCapability::FileInput, "file input"),
            (ChatCapability::ImageReplay, "replaying assistant images"),
        ];
        for (capability, what) in cases {
            for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
                let request = everything().unsupported_policy(policy);
                let err = run(&without(capability), &request).unwrap_err();
                assert!(matches!(err, Error::Unsupported { .. }), "{capability:?}");
                assert!(err.message().contains(what), "{capability:?}: {err}");
                assert_eq!(err.context().unwrap().provider.as_deref(), Some("mock"));
            }
        }
    }

    /// `StructuredOutput` covers JSON Schema
    /// formats only. JSON mode is left to the adapter, so no answer for it
    /// refuses it, and it relies on none.
    #[test]
    fn json_mode_is_not_structured_output() {
        let request = ChatRequest::new("m")
            .user("Reply in JSON.")
            .response_format(ResponseFormat::Json)
            .unsupported_policy(UnsupportedPolicy::Error);
        for support in [CapabilitySupport::Unsupported, CapabilitySupport::Unknown] {
            let provider = MockProvider::empty()
                .with_chat_capability(ChatCapability::StructuredOutput, support);
            let (checked, d) = run(&provider, &request).unwrap();
            assert_eq!(checked, request);
            assert!(d.warnings().is_empty());
            assert!(d.unknown::<ChatCapability>().is_empty(), "{support:?}");
        }
    }

    #[test]
    fn empty_tools_and_automatic_choices_are_not_semantic_uses() {
        let request = ChatRequest::new("m")
            .user("q")
            .tools(Vec::<ToolDefinition>::new())
            .tool_choice(ToolChoice::Auto)
            .response_format(ResponseFormat::Text);
        let provider = without(ChatCapability::ToolCalls).with_chat_capability(
            ChatCapability::StructuredOutput,
            CapabilitySupport::Unsupported,
        );
        let provider = provider.with_chat_capability(
            ChatCapability::ForcedToolChoice,
            CapabilitySupport::Unsupported,
        );
        run(&provider, &request).unwrap();
    }

    #[test]
    fn hints_are_removed_with_a_warning() {
        let request = ChatRequest::new("m")
            .system(SystemPrompt::new("s").with_cache(CacheBreakpoint::new()))
            .user(vec![ContentPart::text("q"), image(Some(ImageDetail::High))])
            .tools([tool().with_cache(CacheBreakpoint::new())])
            .reasoning(ReasoningEffort::High)
            .cache(PromptCache::new().key("k"));
        let provider = [
            ChatCapability::ReasoningConfig,
            ChatCapability::PromptCache,
            ChatCapability::ImageDetail,
        ]
        .into_iter()
        .fold(MockProvider::empty(), |p, c| {
            p.with_chat_capability(c, CapabilitySupport::Unsupported)
        });
        let (checked, d) = run(&provider, &request).unwrap();
        assert_eq!(checked.reasoning, None);
        assert_eq!(checked.cache, None);
        assert_eq!(checked.cache_breakpoints().count(), 0);
        assert_eq!(ContentUse::of(&checked).image_detail, vec![]);

        let warnings = d.warnings();
        assert!(
            warnings.iter().all(|w| w.kind == WarningKind::Unsupported),
            "{warnings:?}"
        );
        let subjects: Vec<_> = warnings.iter().map(|w| w.subject.clone()).collect();
        assert_eq!(
            subjects,
            vec![
                Subject::Field {
                    field: RequestField::Reasoning
                },
                Subject::Field {
                    field: RequestField::Cache
                },
                Subject::Content {
                    message: 0,
                    block: Some(1)
                },
            ]
        );
    }

    #[test]
    fn hints_fail_under_a_strict_policy() {
        let cases: [(ChatCapability, ChatRequest); 3] = [
            (
                ChatCapability::ReasoningConfig,
                ChatRequest::new("m")
                    .user("q")
                    .reasoning(ReasoningEffort::Low),
            ),
            (
                ChatCapability::PromptCache,
                ChatRequest::new("m")
                    .message(Message::user("q").with_cache(CacheBreakpoint::new())),
            ),
            (
                ChatCapability::ImageDetail,
                ChatRequest::new("m").user(vec![image(Some(ImageDetail::Low))]),
            ),
        ];
        for (capability, request) in cases {
            run(&without(capability), &request).unwrap();
            let strict = request.unsupported_policy(UnsupportedPolicy::Error);
            let err = run(&without(capability), &strict).unwrap_err();
            assert!(matches!(err, Error::Unsupported { .. }), "{capability:?}");
        }
    }

    /// Capability answers describe the endpoint (ADR 0015): the messages
    /// name the provider, never the model, and a hint says what happened
    /// to it while its refusal does not.
    #[test]
    fn messages_name_the_provider_not_the_model() {
        let provider = [
            ChatCapability::ReasoningConfig,
            ChatCapability::ImageDetail,
            ChatCapability::ReasoningReplay,
        ]
        .into_iter()
        .fold(
            MockProvider::empty().with_provider_name("acme-gateway"),
            |p, c| p.with_chat_capability(c, CapabilitySupport::Unsupported),
        );
        let request = ChatRequest::new("acme/model-7")
            .user(vec![image(Some(ImageDetail::High))])
            .message(Message::assistant_blocks(vec![
                ContentBlock::reasoning("think"),
                ContentBlock::text("a"),
            ]))
            .user("q")
            .reasoning(ReasoningEffort::High);
        let (_, d) = run(&provider, &request).unwrap();
        let messages: Vec<_> = d.warnings().iter().map(|w| w.message.as_str()).collect();
        assert_eq!(
            messages,
            [
                "reasoning configuration is not supported by `acme-gateway`; `reasoning` was not sent",
                "image detail is not supported by `acme-gateway`; the detail was not sent",
                "`acme-gateway` takes no reasoning back; it was dropped",
            ]
        );
        assert!(messages.iter().all(|m| !m.contains("model-7")));

        let strict = request.unsupported_policy(UnsupportedPolicy::Error);
        let err = run(&provider, &strict).unwrap_err();
        assert_eq!(
            err.message(),
            "cannot send `reasoning`: reasoning configuration is not supported by `acme-gateway`"
        );

        let tools = ChatRequest::new("acme/model-7").user("q").tools([tool()]);
        let err = run(
            &without(ChatCapability::ToolCalls).with_provider_name("acme-gateway"),
            &tools,
        )
        .unwrap_err();
        assert_eq!(
            err.message(),
            "cannot send `tools`: tool calling is not supported by `acme-gateway`"
        );
    }

    #[test]
    fn honored_hints_are_kept_or_removed_silently() {
        // Disabling parallel calls on a model without them is honored, and
        // allowing them is permission, never a degradation, even under a
        // strict policy.
        for setting in [false, true] {
            let request = ChatRequest::new("m")
                .user("q")
                .parallel_tool_calls(setting)
                .unsupported_policy(UnsupportedPolicy::Error);
            let (checked, d) = run(&without(ChatCapability::ParallelToolCalls), &request).unwrap();
            assert_eq!(checked.parallel_tool_calls, Some(setting));
            assert!(d.warnings().is_empty());
            assert!(d.unknown::<ChatCapability>().is_empty());
        }

        // So is asking a model without reasoning controls not to reason,
        // even under a strict policy.
        let request = ChatRequest::new("m")
            .user("q")
            .reasoning(ReasoningEffort::None)
            .unsupported_policy(UnsupportedPolicy::Error);
        let (checked, d) = run(&without(ChatCapability::ReasoningConfig), &request).unwrap();
        assert_eq!(checked.reasoning, None);
        assert!(d.warnings().is_empty());

        // An image without a detail hint uses no detail.
        let request = ChatRequest::new("m").user(vec![image(None)]);
        let (_, d) = run(&without(ChatCapability::ImageDetail), &request).unwrap();
        assert!(d.warnings().is_empty());
    }

    #[test]
    fn unreplayable_reasoning_is_dropped_with_a_warning_under_every_policy() {
        for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
            let request = ChatRequest::new("m")
                .user("q")
                .message(Message::assistant_blocks(vec![
                    ContentBlock::reasoning("think"),
                    ContentBlock::text("a"),
                ]))
                .user("q2")
                .unsupported_policy(policy);
            let (checked, d) = run(&without(ChatCapability::ReasoningReplay), &request).unwrap();
            let Message::Assistant(m) = &checked.messages[1] else {
                panic!("assistant message expected");
            };
            assert_eq!(m.content, vec![ContentBlock::text("a")]);
            assert_eq!(d.warnings().len(), 1);
            assert_eq!(d.warnings()[0].kind, WarningKind::Dropped);
            assert_eq!(
                d.warnings()[0].subject,
                Subject::Content {
                    message: 1,
                    block: Some(0)
                }
            );
        }
    }

    /// Reasoning the provider wrote itself is
    /// dropped without a warning; the rest of the drops are one warning
    /// per request, listing the messages.
    #[test]
    fn own_reasoning_drops_silently_and_the_rest_warn_once_per_request() {
        use crate::ProviderData;

        let own = |text: &str| {
            ContentBlock::Reasoning(crate::Reasoning::new(text).with_provider_data(
                ProviderData::new().with_raw("mock", "signature", serde_json::json!("s")),
            ))
        };
        let request = ChatRequest::new("m")
            .user("q")
            .message(Message::assistant_blocks(vec![
                own("mine"),
                ContentBlock::text("a"),
            ]))
            .user("q2")
            .message(Message::assistant_blocks(vec![
                ContentBlock::reasoning("no origin"),
                ContentBlock::text("b"),
            ]))
            .user("q3")
            .message(Message::assistant_blocks(vec![
                ContentBlock::reasoning("no origin either"),
                own("mine again"),
                ContentBlock::text("c"),
            ]))
            .user("q4");
        let (checked, d) = run(&without(ChatCapability::ReasoningReplay), &request).unwrap();
        // Every reasoning block is dropped.
        assert!(checked.messages.iter().all(|m| {
            match m {
                Message::Assistant(m) => !m
                    .content
                    .iter()
                    .any(|b| matches!(b, ContentBlock::Reasoning(_))),
                _ => true,
            }
        }));
        // One warning, naming the messages with reasoning of no known origin.
        assert_eq!(d.warnings().len(), 1);
        assert_eq!(
            d.warnings()[0].subject,
            Subject::Messages {
                messages: vec![3, 5]
            }
        );
        assert_eq!(
            d.warnings()[0].code.as_deref(),
            Some("dropped.reasoning_replay")
        );

        // Only its own reasoning: nothing to report.
        let request = ChatRequest::new("m")
            .user("q")
            .message(Message::assistant_blocks(vec![
                own("mine"),
                ContentBlock::text("a"),
            ]))
            .user("q2");
        let (_, d) = run(&without(ChatCapability::ReasoningReplay), &request).unwrap();
        assert!(d.warnings().is_empty());
    }

    /// Tomás's repro: an assistant turn of foreign reasoning and foreign
    /// `Other` content, to a model that cannot replay reasoning.
    #[test]
    fn later_warnings_name_the_callers_blocks() {
        let foreign =
            OtherContent::new("server_tool_use", serde_json::Map::new()).with_origin("elsewhere");
        let request = ChatRequest::new("m")
            .user("q")
            .message(Message::assistant_blocks(vec![
                ContentBlock::reasoning("think"),
                ContentBlock::Other(foreign),
            ]))
            .user("q2");
        let (checked, mut d) = run(&without(ChatCapability::ReasoningReplay), &request).unwrap();
        let Message::Assistant(m) = &checked.messages[1] else {
            panic!("assistant message expected");
        };
        assert!(matches!(m.content[..], [ContentBlock::Other(_)]));

        // The encoder drops the `Other` block, block 0 of what it received.
        d.dropped(
            Subject::Content {
                message: 1,
                block: Some(0),
            },
            "foreign content",
        );
        let blocks: Vec<_> = d.warnings().iter().map(|w| w.subject.clone()).collect();
        assert_eq!(
            blocks,
            [
                Subject::Content {
                    message: 1,
                    block: Some(0)
                },
                Subject::Content {
                    message: 1,
                    block: Some(1)
                },
            ]
        );
    }

    #[test]
    fn every_way_of_recording_translates_content_after_a_removed_block() {
        let request = ChatRequest::new("m")
            .message(Message::assistant_blocks(vec![
                ContentBlock::text("a"),
                ContentBlock::reasoning("first"),
                ContentBlock::text("b"),
                ContentBlock::reasoning("second"),
                ContentBlock::text("c"),
            ]))
            .user("q")
            .message(Message::assistant_blocks(vec![ContentBlock::text("d")]));
        let mut d = Degradations::new("mock", UnsupportedPolicy::Warn);
        let before = Subject::Content {
            message: 0,
            block: Some(2),
        };
        // Recorded before preflight: already the caller's.
        d.approximated(before.clone(), "before");
        let mut checked = Cow::Borrowed(&request);
        preflight(
            &without(ChatCapability::ReasoningReplay),
            &mut checked,
            &mut d,
        )
        .unwrap();
        let content = |message, block| Subject::Content { message, block };
        // Preflighted blocks of message 0: a, b, c.
        d.hint(content(0, Some(1)), "b").unwrap();
        d.coded_lossy_hint(content(0, Some(2)), "c", "approximated.c")
            .unwrap();
        d.warn(Warning::new(WarningKind::Other, content(0, Some(0)), "a"));
        d.extend([Warning::new(WarningKind::Other, content(0, Some(2)), "c")]);
        // A block past the end keeps its distance from it.
        d.dropped(content(0, Some(3)), "past the end");
        // Untouched: the whole message, other messages, other subjects.
        d.dropped(content(0, None), "message");
        d.dropped(content(2, Some(0)), "other message");
        d.unsupported_hint(RequestField::Seed, "seed").unwrap();

        let subjects: Vec<_> = d.warnings().iter().map(|w| w.subject.clone()).collect();
        assert_eq!(
            subjects,
            vec![
                before,
                // Preflight's own warning, not translated twice: one for
                // both reasoning blocks of the message.
                content(0, Some(1)),
                content(0, Some(2)),
                content(0, Some(4)),
                content(0, Some(0)),
                content(0, Some(4)),
                content(0, Some(5)),
                content(0, None),
                content(2, Some(0)),
                Subject::Field {
                    field: RequestField::Seed
                },
            ]
        );
    }

    #[test]
    fn a_second_preflight_composes_with_the_first() {
        let request = ChatRequest::new("m").message(Message::assistant_blocks(vec![
            ContentBlock::reasoning("think"),
            ContentBlock::text("a"),
            ContentBlock::text("b"),
        ]));
        let mut d = Degradations::new("mock", UnsupportedPolicy::Warn);
        let mut checked = Cow::Borrowed(&request);
        preflight(
            &without(ChatCapability::ReasoningReplay),
            &mut checked,
            &mut d,
        )
        .unwrap();
        // A wrapper's preflight leaves nothing for an inner one to remove,
        // so the inner one records no further shift.
        preflight(
            &without(ChatCapability::ReasoningReplay),
            &mut checked,
            &mut d,
        )
        .unwrap();
        d.dropped(
            Subject::Content {
                message: 0,
                block: Some(1),
            },
            "b",
        );
        assert_eq!(
            d.warnings().last().unwrap().subject,
            Subject::Content {
                message: 0,
                block: Some(2)
            }
        );
    }

    #[test]
    fn warnings_join_those_already_collected() {
        let mut d = Degradations::new("mock", UnsupportedPolicy::Warn);
        d.unsupported_hint(RequestField::Seed, "no seed").unwrap();
        let request = ChatRequest::new("m")
            .user("q")
            .reasoning(ReasoningEffort::High);
        let mut checked = Cow::Borrowed(&request);
        preflight(
            &without(ChatCapability::ReasoningConfig),
            &mut checked,
            &mut d,
        )
        .unwrap();
        let fields: Vec<_> = d.warnings().iter().map(|w| w.subject.clone()).collect();
        assert_eq!(
            fields,
            vec![
                Subject::Field {
                    field: RequestField::Seed
                },
                Subject::Field {
                    field: RequestField::Reasoning
                }
            ]
        );
    }

    #[test]
    fn tool_result_parts_are_checked_too() {
        let request = ChatRequest::new("m")
            .user("q")
            .message(Message::tool_result(
                "call_1",
                "t",
                vec![ContentPart::file_base64("application/pdf", "JVBERi0=")],
            ));
        let err = run(&without(ChatCapability::FileInput), &request).unwrap_err();
        assert!(err.message().contains("file input"), "{err}");
    }
}
