use std::borrow::Cow;

use serde_json::json;

use super::*;
use crate::mock::MockProvider;
use crate::{
    CacheBreakpoint, ChatProvider, Error, File, Image, ImageDetail, PromptCache, ProviderToolCall,
    SystemPrompt, ToolDefinition, UnsupportedPolicy, UnsupportedStage, Warning, WarningKind,
};

const NAMESPACE: &str = "gateway";

fn run(wire: &WireSupport, request: &ChatRequest) -> Result<(ChatRequest, Vec<Warning>)> {
    let mut d = Degradations::new("gateway", request.unsupported_policy.unwrap_or_default());
    let mut checked = Cow::Borrowed(request);
    degrade(wire, &|ns| ns == NAMESPACE, &mut checked, &mut d)?;
    Ok((checked.into_owned(), d.into_warnings()))
}

fn refused(wire: &WireSupport, request: &ChatRequest) -> Error {
    let err = run(wire, request).unwrap_err();
    assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
    assert_eq!(
        err.unsupported_stage(),
        Some(UnsupportedStage::BeforeSending)
    );
    err
}

fn tool() -> ToolDefinition {
    ToolDefinition::new("t", json!({"type": "object"}))
}

fn content(message: usize, block: Option<usize>) -> Subject {
    Subject::Content { message, block }
}

fn other(origin: Option<&str>) -> OtherContent {
    let other = OtherContent::new("vendor_block", serde_json::Map::new());
    match origin {
        Some(origin) => other.with_origin(origin),
        None => other,
    }
}

#[test]
fn a_text_request_passes_unchanged() {
    let request = ChatRequest::new("m")
        .user("q")
        .assistant("a")
        .user("q2")
        .unsupported_policy(UnsupportedPolicy::Error);
    let mut d = Degradations::new("gateway", UnsupportedPolicy::Error);
    let mut checked = Cow::Borrowed(&request);
    degrade(&WireSupport::text(), &|_| false, &mut checked, &mut d).unwrap();
    assert!(matches!(checked, Cow::Borrowed(_)));
    assert!(d.warnings().is_empty());
}

#[test]
fn semantic_fields_are_refused_under_every_policy() {
    let cases: [(RequestField, ChatRequest); 6] = [
        (
            RequestField::Tools,
            ChatRequest::new("m").user("q").tools([tool()]),
        ),
        (
            RequestField::ToolChoice,
            ChatRequest::new("m")
                .user("q")
                .tools([tool()])
                .tool_choice(ToolChoice::Required),
        ),
        (
            RequestField::ResponseFormat,
            ChatRequest::new("m")
                .user("q")
                .response_format(ResponseFormat::Json),
        ),
        (
            RequestField::MaxOutputTokens,
            ChatRequest::new("m").user("q").max_output_tokens(10),
        ),
        (
            RequestField::System,
            ChatRequest::new("m").system("s").user("q"),
        ),
        (
            RequestField::ToolChoice,
            ChatRequest::new("m")
                .user("q")
                .tool_choice(ToolChoice::Specific { name: "t".into() }),
        ),
    ];
    // Tools carried but no tool choice: a forced choice is still refused.
    let tools_only = WireSupport::text().field(RequestField::Tools);
    for (wire, (field, request)) in [WireSupport::text(), tools_only]
        .into_iter()
        .flat_map(|wire| cases.iter().map(move |case| (wire, case)))
    {
        // Tools are refused before the choice that forces them.
        let field = if !request.tools.is_empty() && !wire.carries(RequestField::Tools) {
            &RequestField::Tools
        } else {
            field
        };
        if wire == tools_only && *field == RequestField::Tools {
            continue;
        }
        for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
            let request = request.clone().unsupported_policy(policy);
            let err = refused(&wire, &request);
            assert_eq!(err.unsupported_field(), Some(*field), "{err}");
            assert!(err.message().contains("`gateway`"), "{err}");
        }
    }
}

#[test]
fn carried_fields_are_left_to_the_encoder() {
    let wire = WireSupport::text().fields([
        RequestField::Tools,
        RequestField::ToolChoice,
        RequestField::ResponseFormat,
        RequestField::MaxOutputTokens,
        RequestField::System,
        RequestField::Temperature,
        RequestField::TopP,
        RequestField::Stop,
        RequestField::FrequencyPenalty,
        RequestField::PresencePenalty,
        RequestField::Seed,
        RequestField::ParallelToolCalls,
        RequestField::Reasoning,
        RequestField::Cache,
    ]);
    let request = ChatRequest::new("m")
        .system("s")
        .user("q")
        .tools([tool()])
        .tool_choice(ToolChoice::Required)
        .response_format(ResponseFormat::Json)
        .max_output_tokens(10)
        .temperature(0.1)
        .top_p(0.9)
        .stop(["x"])
        .frequency_penalty(0.1)
        .presence_penalty(0.1)
        .seed(1)
        .parallel_tool_calls(false)
        .reasoning(ReasoningEffort::High)
        .cache(PromptCache::new().key("k"))
        .unsupported_policy(UnsupportedPolicy::Error);
    let (checked, warnings) = run(&wire, &request).unwrap();
    assert_eq!(checked, request);
    assert!(warnings.is_empty());
}

#[test]
fn hints_are_removed_with_a_coded_warning_or_refused_when_strict() {
    let request = ChatRequest::new("m")
        .message(Message::user("q").with_cache(CacheBreakpoint::new()))
        .temperature(0.1)
        .top_p(0.9)
        .stop(["x"])
        .frequency_penalty(0.1)
        .presence_penalty(0.1)
        .seed(1)
        .parallel_tool_calls(false)
        .reasoning(ReasoningEffort::High)
        .cache(PromptCache::new().key("k"));
    let (checked, warnings) = run(&WireSupport::text(), &request).unwrap();
    assert_eq!(checked, ChatRequest::new("m").user("q"));
    let fields: Vec<_> = warnings
        .iter()
        .map(|w| {
            assert_eq!(w.kind, WarningKind::Unsupported);
            let Subject::Field { field } = w.subject else {
                panic!("{w:?}");
            };
            assert_eq!(w.code.as_deref(), Some(&*format!("unsupported.{field}")));
            field
        })
        .collect();
    assert_eq!(
        fields,
        [
            RequestField::Temperature,
            RequestField::TopP,
            RequestField::Stop,
            RequestField::FrequencyPenalty,
            RequestField::PresencePenalty,
            RequestField::Seed,
            RequestField::ParallelToolCalls,
            RequestField::Reasoning,
            RequestField::Cache,
        ]
    );
    assert_eq!(
        warnings[0].message,
        "`temperature` is not supported by `gateway`; `temperature` was not sent"
    );

    let strict = request.unsupported_policy(UnsupportedPolicy::Error);
    let err = refused(&WireSupport::text(), &strict);
    assert_eq!(
        err.message(),
        "cannot send `temperature`: `temperature` is not supported by `gateway`"
    );
}

#[test]
fn what_is_honoured_anyway_is_removed_without_a_warning() {
    let request = ChatRequest::new("m")
        .user("q")
        .tools(Vec::<ToolDefinition>::new())
        .tool_choice(ToolChoice::Auto)
        .response_format(ResponseFormat::Text)
        .stop(Vec::<String>::new())
        .parallel_tool_calls(true)
        .reasoning(ReasoningEffort::None)
        .unsupported_policy(UnsupportedPolicy::Error);
    let (checked, warnings) = run(&WireSupport::text(), &request).unwrap();
    assert_eq!(
        checked,
        ChatRequest::new("m")
            .user("q")
            .unsupported_policy(UnsupportedPolicy::Error)
    );
    assert!(warnings.is_empty(), "{warnings:?}");
}

#[test]
fn a_disabled_tool_choice_leaves_the_tools_out() {
    for wire in [
        WireSupport::text(),
        WireSupport::text().field(RequestField::Tools),
    ] {
        let request = ChatRequest::new("m")
            .user("q")
            .tools([tool()])
            .tool_choice(ToolChoice::Disabled)
            .unsupported_policy(UnsupportedPolicy::Error);
        let (checked, warnings) = run(&wire, &request).unwrap();
        assert_eq!((checked.tools, checked.tool_choice), (Vec::new(), None));
        assert!(warnings.is_empty());
    }
}

#[test]
fn content_the_wire_format_cannot_carry_is_refused() {
    let image = || ContentPart::Image(Image::url("https://example.com/a.png"));
    let cases = [
        (
            ChatRequest::new("m").user(vec![ContentPart::text("q"), image()]),
            content(0, Some(1)),
            "image input",
        ),
        (
            ChatRequest::new("m").user(vec![ContentPart::File(File::url(
                "application/pdf",
                "https://example.com/a.pdf",
            ))]),
            content(0, Some(0)),
            "file input",
        ),
        (
            ChatRequest::new("m")
                .user("q")
                .message(Message::assistant_blocks(vec![
                    ContentBlock::text("a"),
                    ContentBlock::tool_call("c1", "t", "{}"),
                ]))
                .message(Message::tool_result("c1", "t", "ok")),
            content(1, Some(1)),
            "tool calls",
        ),
        (
            ChatRequest::new("m")
                .user("q")
                .message(Message::tool_result("c1", "t", "ok")),
            content(1, None),
            "tool results",
        ),
        (
            ChatRequest::new("m")
                .user("q")
                .message(Message::assistant_blocks(vec![ContentBlock::Image(
                    Image::url("https://example.com/b.png"),
                )]))
                .user("q2"),
            content(1, Some(0)),
            "replaying assistant images",
        ),
        (
            // The caller's raw part, or this provider's: never dropped.
            ChatRequest::new("m").user(vec![ContentPart::Other(other(None))]),
            content(0, Some(0)),
            "`vendor_block` content",
        ),
        (
            ChatRequest::new("m").user(vec![ContentPart::Other(other(Some(NAMESPACE)))]),
            content(0, Some(0)),
            "`vendor_block` content",
        ),
    ];
    for (request, subject, what) in cases {
        let err = refused(&WireSupport::text(), &request);
        assert_eq!(err.unsupported_subject(), Some(&subject), "{err}");
        assert!(err.message().contains(what), "{err}");
    }

    let wire = WireSupport::text()
        .field(RequestField::Tools)
        .image_input()
        .file_input()
        .image_replay()
        .provider_content();
    let request = ChatRequest::new("m")
        .user(vec![
            ContentPart::text("q"),
            image(),
            ContentPart::File(File::url("application/pdf", "https://example.com/a.pdf")),
            ContentPart::Other(other(None)),
        ])
        .message(Message::assistant_blocks(vec![
            ContentBlock::Image(Image::url("https://example.com/b.png")),
            ContentBlock::tool_call("c1", "t", "{}"),
        ]))
        .message(Message::tool_result("c1", "t", "ok"));
    let (checked, warnings) = run(&wire, &request).unwrap();
    assert_eq!(checked, request);
    assert!(warnings.is_empty());
}

#[test]
fn image_detail_is_a_hint() {
    let request = ChatRequest::new("m").user(vec![
        ContentPart::text("q"),
        ContentPart::Image(Image::url("https://example.com/a.png").with_detail(ImageDetail::Low)),
    ]);
    let (checked, warnings) = run(&WireSupport::text().image_input(), &request).unwrap();
    assert_eq!(
        checked,
        ChatRequest::new("m").user(vec![
            ContentPart::text("q"),
            ContentPart::Image(Image::url("https://example.com/a.png")),
        ])
    );
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].subject, content(0, Some(1)));
    assert_eq!(
        warnings[0].code.as_deref(),
        Some("unsupported.image_detail")
    );

    let wire = WireSupport::text().image_input().image_detail();
    assert_eq!(run(&wire, &request).unwrap().0, request);
    let strict = request.unsupported_policy(UnsupportedPolicy::Error);
    refused(&WireSupport::text().image_input(), &strict);
}

#[test]
fn replay_state_is_dropped_with_a_coded_warning_under_every_policy() {
    let foreign_call = ProviderToolCall::new("s1", "web_search", "{}").with_origin("other");
    for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
        let request = ChatRequest::new("m")
            .user(vec![
                ContentPart::text("q"),
                ContentPart::Other(other(Some("other"))),
            ])
            .message(Message::assistant_blocks(vec![
                ContentBlock::reasoning("think"),
                ContentBlock::Other(other(Some("other"))),
                ContentBlock::text("a"),
                ContentBlock::Other(other(Some(NAMESPACE))),
                ContentBlock::ProviderToolCall(foreign_call.clone()),
            ]))
            .user("q2")
            .unsupported_policy(policy);
        let (checked, warnings) = run(&WireSupport::text(), &request).unwrap();
        assert_eq!(
            checked,
            ChatRequest::new("m")
                .user(vec![ContentPart::text("q")])
                .message(Message::assistant_blocks(vec![ContentBlock::text("a")]))
                .user("q2")
                .unsupported_policy(policy)
        );
        let seen: Vec<_> = warnings
            .iter()
            .map(|w| (w.subject.clone(), w.replay_state()))
            .collect();
        assert_eq!(
            seen,
            [
                // One warning per code: the foreign content of both
                // messages is one warning listing them.
                (
                    Subject::Messages {
                        messages: vec![0, 1]
                    },
                    Some(ReplayState::ForeignContent)
                ),
                (content(1, Some(0)), Some(ReplayState::Reasoning)),
                (content(1, Some(3)), Some(ReplayState::Content)),
            ]
        );
    }
}

#[test]
fn carried_replay_state_is_left_but_another_providers_still_goes() {
    let wire = WireSupport::text().reasoning_replay().provider_content();
    let request = ChatRequest::new("m")
        .user("q")
        .message(Message::assistant_blocks(vec![
            ContentBlock::reasoning("think"),
            ContentBlock::Other(other(Some("other"))),
            ContentBlock::Other(other(Some(NAMESPACE))),
        ]))
        .user("q2");
    let (checked, warnings) = run(&wire, &request).unwrap();
    let Message::Assistant(m) = &checked.messages[1] else {
        panic!("an assistant message");
    };
    assert_eq!(
        m.content,
        [
            ContentBlock::reasoning("think"),
            ContentBlock::Other(other(Some(NAMESPACE)))
        ]
    );
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].subject, content(1, Some(1)));
}

/// Blocks preflight removed and blocks this pass removes compose: every
/// warning names the caller's block.
#[test]
fn warnings_after_preflight_and_the_pass_name_the_callers_blocks() {
    let provider = MockProvider::empty().with_chat_capability(
        ChatCapability::ReasoningReplay,
        CapabilitySupport::Unsupported,
    );
    let request = ChatRequest::new("m")
        .user("q")
        .message(Message::assistant_blocks(vec![
            ContentBlock::reasoning("think"),
            ContentBlock::Other(other(Some("other"))),
            ContentBlock::text("a"),
            ContentBlock::text("b"),
        ]))
        .user("q2");
    let mut prepared = crate::adapter::chat::prepare(&provider, &request).unwrap();
    prepared
        .degrade_to(&provider, &WireSupport::text())
        .unwrap();
    // The encoder reports block 1 of what it received: the caller's 3.
    prepared
        .degradations_mut()
        .dropped(content(1, Some(1)), "b");
    let validation = prepared.into_validation(crate::ValidationCoverage::Encoding);
    let subjects: Vec<_> = validation
        .warnings()
        .iter()
        .map(|w| w.subject.clone())
        .collect();
    assert_eq!(
        subjects,
        [
            content(1, Some(0)),
            content(1, Some(1)),
            content(1, Some(3))
        ]
    );
}

#[test]
fn capability_answers_follow_the_declaration() {
    let unsupported = Some(CapabilitySupport::Unsupported);
    let text = WireSupport::text();
    for capability in ChatCapability::ALL {
        let answer = text.chat_capability(*capability);
        let request_side = !matches!(
            capability,
            ChatCapability::Streaming
                | ChatCapability::NativeStreaming
                | ChatCapability::StreamUsage
                | ChatCapability::ImageOutput
                | ChatCapability::ReasoningOutput
        );
        assert_eq!(answer.is_some(), request_side, "{capability:?}");
    }
    let tools = text.field(RequestField::Tools);
    assert_eq!(tools.chat_capability(ChatCapability::ToolCalls), None);
    assert_eq!(
        tools.chat_capability(ChatCapability::ForcedToolChoice),
        unsupported
    );
    let images = text.image_detail();
    assert_eq!(
        images.chat_capability(ChatCapability::ImageDetail),
        unsupported
    );
    assert_eq!(
        images
            .image_input()
            .chat_capability(ChatCapability::ImageDetail),
        None
    );
    assert!(
        !text
            .field(RequestField::Dimensions)
            .carries(RequestField::Dimensions)
    );
}

/// An adapter that answers from the declaration passes the core's own
/// capability check and the pass the same way: preflight refuses first.
#[test]
fn preflight_and_the_pass_agree_on_a_declared_wire() {
    struct Gateway;
    impl crate::ProviderIdentity for Gateway {
        fn provider_name(&self) -> &str {
            "gateway"
        }
        fn reads_namespace(&self, namespace: &str) -> bool {
            namespace == NAMESPACE
        }
    }
    impl ChatProvider for Gateway {
        type Stream = crate::ChatStream;
        async fn chat(&self, _: &ChatRequest) -> Result<crate::ChatResponse> {
            unreachable!()
        }
        async fn chat_stream(&self, _: &ChatRequest) -> Result<Self::Stream> {
            unreachable!()
        }
        fn chat_capability(&self, _: &str, capability: ChatCapability) -> CapabilitySupport {
            WireSupport::text()
                .chat_capability(capability)
                .unwrap_or(CapabilitySupport::Unknown)
        }
    }
    let request = ChatRequest::new("m")
        .system(SystemPrompt::new("s"))
        .user("q")
        .tools([tool()]);
    let err = crate::adapter::chat::prepare(&Gateway, &request).unwrap_err();
    assert_eq!(err.unsupported_field(), Some(RequestField::Tools));
}
