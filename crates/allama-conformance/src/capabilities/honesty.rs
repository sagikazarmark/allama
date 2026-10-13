//! Capability honesty: whatever a provider answers about a capability, a
//! request that uses it never loses what it asks for silently.
//!
//! [`assert_capabilities_honest`] probes every capability, request field,
//! content part, content block and tool choice, for each listed model and
//! for a model the provider cannot know ([`UNKNOWN_MODEL`]), whose answers
//! are typically `Unknown`. `Unknown` means "send it and let the provider
//! decide" (ADR 0003, principle 2), so an encoder sends what the request
//! asks for, or reports it through `Degradations`. A `Supported` answer
//! promises more: the request is sent with what it asks for, neither
//! refused nor warned about.

use std::fmt;
use std::future::Future;

use allama::chat::stream::StreamBlockType;
use allama::{
    CacheBreakpoint, CacheRetention, CapabilitySupport, ChatCapability, ChatProvider, ChatRequest,
    ChatResponse, ContentBlock, ContentPart, Error, ExtraMap, File, FileKind, Image, ImageDetail,
    Message, OtherContent, PromptCache, ProviderToolCall, ProviderToolResult, Reasoning,
    ReasoningEffort, Refusal, ReplayState, RequestField, ResponseFormat, Subject, ToolCall,
    ToolChoice, ToolDefinition, WarningKind,
};
use serde_json::json;

use crate::diagnostics::elide_payloads;
use crate::e2e::{PINEAPPLE_PDF_BASE64, RED_PNG_BASE64};
use crate::probe::{CHAT, CHAT_STREAM, ProbeOutcome, WireValue, describe, misshapen_refusal};

/// The model id the honesty contract adds to the listed models: no
/// provider knows it, so its capability answers are what an unlisted model
/// gets (usually `Unknown`).
pub const UNKNOWN_MODEL: &str = "allama-probe-unknown-model";

/// The namespace of the foreign state the replay probes carry.
const FOREIGN: &str = "allama-probe-foreign";

/// What [`assert_capabilities_honest`] needs from the adapter under test.
///
/// Built with [`new`](Self::new) and setters, so new inputs never break
/// callers.
#[non_exhaustive]
pub struct HonestyUnderTest<F, G> {
    models: Vec<String>,
    probe: F,
    stream_probe: G,
    known_gaps: Vec<HonestyCheck>,
}

impl<F, G> HonestyUnderTest<F, G> {
    /// The models to probe (each is probed, then [`UNKNOWN_MODEL`]), and
    /// the chat and streaming probes (see [probes](crate::probe)): `probe`
    /// sends a request through `chat`, `stream_probe` through
    /// `chat_stream`, collecting the stream into a response.
    pub fn new<M: Into<String>>(
        models: impl IntoIterator<Item = M>,
        probe: F,
        stream_probe: G,
    ) -> Self {
        Self {
            models: models.into_iter().map(Into::into).collect(),
            probe,
            stream_probe,
            known_gaps: Vec::new(),
        }
    }

    /// Checks with an acknowledged defect: skipped, but listing one that
    /// passes for every model on both paths fails, so a fixed gap leaves
    /// the list.
    #[must_use]
    pub fn known_gaps(mut self, gaps: impl IntoIterator<Item = HonestyCheck>) -> Self {
        self.known_gaps.extend(gaps);
        self
    }
}

impl<F, G> fmt::Debug for HonestyUnderTest<F, G> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HonestyUnderTest")
            .field("models", &self.models)
            .field("known_gaps", &self.known_gaps)
            .finish_non_exhaustive()
    }
}

/// One check of [`assert_capabilities_honest`], for its problem reports and
/// [`HonestyUnderTest::known_gaps`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HonestyCheck {
    /// A portable request field.
    Field(RequestField),
    /// A user content part, by its [type name](ContentPart::type_name).
    /// A `file` part holds a document.
    Part(&'static str),
    /// A user file part of an audio or video media type, by its kind.
    Media(FileKind),
    /// A replayed assistant block, by its stream block type.
    Block(StreamBlockType),
    /// Image detail on an input image.
    ImageDetail,
    /// A tool choice, by its [kind name](ToolChoice::kind_name).
    ToolChoice(&'static str),
}

impl fmt::Display for HonestyCheck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Field(field) => write!(f, "field `{field}`"),
            Self::Part(name) => write!(f, "`{name}` part"),
            Self::Media(kind) => write!(f, "{kind:?} file part"),
            Self::Block(block_type) => write!(f, "{block_type:?} block"),
            Self::ImageDetail => f.write_str("image detail"),
            Self::ToolChoice(kind) => write!(f, "tool choice `{kind}`"),
        }
    }
}

/// Assert that no capability answer makes a request drop what it asks for
/// silently.
///
/// For each listed model and for [`UNKNOWN_MODEL`], on `chat` and on
/// `chat_stream`, it sends one request per check and judges it:
///
/// - **Content** (a content part, a replayed block, a tool definition, a
///   response schema, a system prompt): its payload must be in a request
///   body, or the request must be warned about (a warning whose subject
///   names it) or refused with `Unsupported` before anything was sent
///   (stage [`BeforeSending`](allama::UnsupportedStage::BeforeSending)).
/// - **Configuration** (sampling fields, seed, stop sequences, reasoning,
///   prompt cache, parallel tool calls, image detail): the request body
///   must differ from the one sent for a contrasting value (or none), or
///   the setting must be warned about or refused before sending. Encoding
///   a setting into another field passes, as in the request field audit;
///   so does a value the provider takes by default when the contrasting
///   value is the one it warns about.
/// - **Tool choice**: each kind is sent with a tool; the kinds that are
///   neither warned about nor refused must send pairwise different bodies,
///   so a choice the endpoint ignores cannot pass as `auto`.
///
/// A capability `provider` answers `Supported` for the model takes neither
/// way out: its check must be sent, with no refusal and no warning of kind
/// [`Unsupported`](WarningKind::Unsupported) or
/// [`Dropped`](WarningKind::Dropped) that names it, and then pass as above.
/// An [`Approximated`](WarningKind::Approximated) warning keeps the claim:
/// the capability was used, in the closest form the provider takes. So
/// this contract is the evidence for every `Supported` chat claim.
/// `provider` must answer as the probes' providers do.
///
/// A claim is judged on what every provider with the capability takes:
/// `StructuredOutput` on a JSON Schema without a name or strict mode, and
/// `PromptCache` on a plain cache breakpoint (their checks also send a
/// name, strict mode, a cache key and a retention, which not every
/// provider has). `ReasoningReplay` keeps through a drop coded
/// `dropped.reasoning_replay`, since the probe's reasoning carries no
/// provider's replay state.
///
/// On `chat_stream`, a refusal before sending keeps a `Supported` claim.
/// This is deliberate: a capability answer has no path, and streaming can
/// refuse what `chat` takes (Workers AI refuses JSON mode and tools on
/// `chat_stream` only), so the claim is judged on `chat`, where it must be
/// sent. The streaming refusal must still come before sending, which
/// [`validate_chat_stream_request`](ChatProvider::validate_chat_stream_request)
/// reports. A warning that drops it on `chat_stream` fails, as on `chat`.
///
/// The checks come from exhaustive lists: [`ChatCapability::ALL`] (each
/// request capability maps to a check below; output capabilities have
/// none), [`RequestField::ALL`], [`ContentPart::TYPE_NAMES`],
/// [`StreamBlockType::ALL`] (every kind of content block, see
/// [`ContentBlock::block_type`]) and [`ToolChoice::KIND_NAMES`]. A value this
/// crate has no check for fails, so an `allama` newer than
/// `allama-conformance` is noticed.
///
/// # Panics
///
/// On any silent drop, a `Supported` claim refused or warned about, a probe
/// that fails another way, a missing check, or a stale known gap.
pub async fn assert_capabilities_honest<P, F, Fut, G, GFut>(
    provider: &P,
    under_test: &HonestyUnderTest<F, G>,
) where
    P: ChatProvider,
    F: Fn(ChatRequest) -> Fut,
    Fut: Future<Output = ProbeOutcome<ChatResponse>>,
    G: Fn(ChatRequest) -> GFut,
    GFut: Future<Output = ProbeOutcome<ChatResponse>>,
{
    let mut problems = Problems::default();
    let mut unlisted = Vec::new();
    let mut models = under_test.models.clone();
    if !models.iter().any(|model| model == UNKNOWN_MODEL) {
        models.push(UNKNOWN_MODEL.to_owned());
    }

    for model in &models {
        let (checks, missing) = checks(model);
        if unlisted.is_empty() {
            unlisted = missing;
        }
        // Every request capability must map to a check that runs.
        let mut claims = Vec::new();
        for &capability in ChatCapability::ALL {
            match capability_check(capability) {
                Covered::By(check) if !checks.iter().any(|c| c.id == check) => {
                    unlisted.push(format!(
                        "{capability:?} maps to {check}, which has no probe"
                    ));
                }
                Covered::By(check)
                    if provider.chat_capability(model, capability)
                        == CapabilitySupport::Supported =>
                {
                    claims.push((check, capability));
                }
                Covered::Unlisted => unlisted.push(format!(
                    "{capability:?} has no honesty check; upgrade allama-conformance"
                )),
                Covered::By(_) | Covered::Output => {}
            }
        }
        for (path, stream) in [(CHAT, false), (CHAT_STREAM, true)] {
            let send = |request: ChatRequest| async move {
                if stream {
                    (under_test.stream_probe)(request).await
                } else {
                    (under_test.probe)(request).await
                }
            };
            let mut tool_choices = Vec::new();
            for check in &checks {
                let outcome = send(check.request.clone()).await;
                let claim = claims.iter().find(|(id, _)| *id == check.id);
                let contradicted = match claim.map(|&(_, capability)| capability) {
                    Some(capability) => match claim_probe(model, capability) {
                        Some((request, names)) => {
                            contradicted_claim(capability, &send(request).await, names, stream)
                        }
                        None => {
                            contradicted_claim(capability, &outcome, check.judge.names(), stream)
                        }
                    },
                    None => None,
                };
                // A refusal or warning the claim rules out settles the
                // check: the judges below would take it.
                let problem = match &check.judge {
                    _ if contradicted.is_some() => contradicted,
                    Judge::Content { payload, names } => judge_content(&outcome, payload, *names),
                    Judge::Config { contrast, names } => {
                        let contrast = send(ChatRequest::clone(contrast)).await;
                        judge_config(&outcome, &contrast, *names)
                    }
                    Judge::ToolChoice => match settled(&outcome, Names::ToolChoice) {
                        Ok(true) => None,
                        Ok(false) => {
                            tool_choices.push((check.id, outcome));
                            None
                        }
                        Err(problem) => Some(problem),
                    },
                };
                if let Some(problem) = problem {
                    problems.push(check.id, format!("{path} `{model}`: {problem}"));
                }
            }
            for (i, (a, first)) in tool_choices.iter().enumerate() {
                for (b, second) in &tool_choices[i + 1..] {
                    if !first.differs(second) {
                        problems.push(
                            *a,
                            format!(
                                "{path} `{model}`: sends the same body as {b} without a \
                                 warning, so the choice is dropped silently"
                            ),
                        );
                    }
                }
            }
        }
    }

    let mut report = Vec::new();
    unlisted.sort();
    unlisted.dedup();
    report.extend(unlisted);
    for (check, found) in &problems.0 {
        if !under_test.known_gaps.contains(check) {
            report.push(format!("{check}: {}", found.join("; ")));
        }
    }
    for gap in &under_test.known_gaps {
        if !problems.0.iter().any(|(check, _)| check == gap) {
            report.push(format!(
                "{gap} is listed as a known gap but passes; remove it"
            ));
        }
    }
    assert!(
        report.is_empty(),
        "capability honesty of `{}` failed:\n  {}",
        provider.provider_name(),
        elide_payloads(&report.join("\n  "))
    );
}

/// Problems by check, in the order they were found.
#[derive(Default)]
struct Problems(Vec<(HonestyCheck, Vec<String>)>);

impl Problems {
    fn push(&mut self, check: HonestyCheck, problem: String) {
        match self.0.iter_mut().find(|(found, _)| *found == check) {
            Some((_, problems)) => problems.push(problem),
            None => self.0.push((check, vec![problem])),
        }
    }
}

/// How a probe request is judged.
enum Judge {
    /// The payload reaches a body, or the request is warned about or
    /// refused before sending.
    Content { payload: &'static str, names: Names },
    /// The body differs from the contrast's, or the setting is warned
    /// about or refused before sending.
    Config {
        contrast: Box<ChatRequest>,
        names: Names,
    },
    /// One tool choice kind; the kinds are compared with each other.
    ToolChoice,
}

impl Judge {
    /// Which warnings name what the check probes.
    fn names(&self) -> Names {
        match self {
            Self::Content { names, .. } | Self::Config { names, .. } => *names,
            Self::ToolChoice => Names::ToolChoice,
        }
    }
}

struct Check {
    id: HonestyCheck,
    request: ChatRequest,
    judge: Judge,
}

/// How a capability is checked.
enum Covered {
    /// By this check.
    By(HonestyCheck),
    /// It describes output: no request uses it.
    Output,
    /// This crate does not know it.
    Unlisted,
}

fn capability_check(capability: ChatCapability) -> Covered {
    use HonestyCheck::{Block, Field, Part, ToolChoice};
    Covered::By(match capability {
        ChatCapability::ToolCalls => Field(RequestField::Tools),
        ChatCapability::ParallelToolCalls => Field(RequestField::ParallelToolCalls),
        ChatCapability::ImageInput => Part("image"),
        ChatCapability::ImageDetail => HonestyCheck::ImageDetail,
        ChatCapability::ImageReplay => Block(StreamBlockType::Image),
        ChatCapability::FileInput => Part("file"),
        ChatCapability::AudioInput => HonestyCheck::Media(FileKind::Audio),
        ChatCapability::VideoInput => HonestyCheck::Media(FileKind::Video),
        ChatCapability::StructuredOutput => Field(RequestField::ResponseFormat),
        ChatCapability::ReasoningReplay => Block(StreamBlockType::Reasoning),
        ChatCapability::ReasoningConfig => Field(RequestField::Reasoning),
        ChatCapability::PromptCache => Field(RequestField::Cache),
        ChatCapability::ForcedToolChoice => ToolChoice("required"),
        ChatCapability::Streaming
        | ChatCapability::NativeStreaming
        | ChatCapability::ImageOutput
        | ChatCapability::ReasoningOutput
        // Checked on the output by `assert_stream_usage_matches_capability`.
        | ChatCapability::StreamUsage => return Covered::Output,
        _ => return Covered::Unlisted,
    })
}

/// Every check for `model`, and the list values this crate has no check
/// for.
fn checks(model: &str) -> (Vec<Check>, Vec<String>) {
    let mut checks = Vec::new();
    let mut missing = Vec::new();

    for &field in RequestField::ALL {
        match field_check(model, field) {
            FieldCheck::Run(check) => checks.push(*check),
            FieldCheck::Elsewhere => {}
            FieldCheck::Unlisted => missing.push(format!(
                "request field `{field}` has no honesty check; upgrade allama-conformance"
            )),
        }
    }
    for &name in ContentPart::TYPE_NAMES {
        match part_check(model, name) {
            Some(check) => checks.push(check),
            None => missing.push(format!(
                "`{name}` parts have no honesty check; upgrade allama-conformance"
            )),
        }
    }
    for (kind, media_type, payload) in [
        (FileKind::Audio, "audio/wav", AUDIO_PAYLOAD),
        (FileKind::Video, "video/mp4", VIDEO_PAYLOAD),
    ] {
        checks.push(Check {
            id: HonestyCheck::Media(kind),
            request: ChatRequest::new(model).message(Message::user_parts(vec![
                ContentPart::text("Describe this."),
                ContentPart::File(File::base64(media_type, payload)),
            ])),
            judge: Judge::Content {
                payload,
                names: Names::Message(0),
            },
        });
    }
    for &block_type in StreamBlockType::ALL {
        match block_check(model, block_type) {
            Some(check) => checks.push(check),
            None => missing.push(format!(
                "{block_type:?} blocks have no honesty check; upgrade allama-conformance"
            )),
        }
    }
    checks.push(Check {
        id: HonestyCheck::ImageDetail,
        request: image_request(model, Some(ImageDetail::High)),
        judge: Judge::Config {
            contrast: Box::new(image_request(model, Some(ImageDetail::Low))),
            names: Names::Message(0),
        },
    });
    for &kind in ToolChoice::KIND_NAMES {
        let choice = match kind {
            "auto" => ToolChoice::Auto,
            "disabled" => ToolChoice::Disabled,
            "required" => ToolChoice::Required,
            "specific" => ToolChoice::Specific {
                name: TOOL_NAME.into(),
            },
            _ => {
                missing.push(format!(
                    "tool choice `{kind}` has no honesty check; upgrade allama-conformance"
                ));
                continue;
            }
        };
        checks.push(Check {
            id: HonestyCheck::ToolChoice(kind),
            request: tool_request(model).tool_choice(choice),
            judge: Judge::ToolChoice,
        });
    }
    (checks, missing)
}

enum FieldCheck {
    Run(Box<Check>),
    /// Checked by another list (tool choice), or not a chat field.
    Elsewhere,
    Unlisted,
}

const TOOL_NAME: &str = "allama_honesty_tool";
/// Base64 of an audio file's first bytes (a WAV header), for the audio
/// check: the contract looks for it on the wire, it is never played.
const AUDIO_PAYLOAD: &str = "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=";
/// Base64 of a video file's first bytes (an MP4 `ftyp` box), for the video
/// check.
const VIDEO_PAYLOAD: &str = "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDE=";
const SCHEMA_PROPERTY: &str = "allama_honesty_property";
const SYSTEM_TEXT: &str = "allama-honesty-system";

fn field_check(model: &str, field: RequestField) -> FieldCheck {
    let base = || ChatRequest::new(model).user("Say hello.");
    let names = Names::Field(field);
    let config = |contrast: ChatRequest| Judge::Config {
        contrast: Box::new(contrast),
        names,
    };
    let (request, judge) = match field {
        RequestField::Temperature => {
            let request = base().temperature(0.37);
            (request, config(base().temperature(0.73)))
        }
        RequestField::MaxOutputTokens => {
            let request = base().max_output_tokens(1234);
            (request, config(base().max_output_tokens(4321)))
        }
        RequestField::TopP => {
            let request = base().top_p(0.37);
            (request, config(base().top_p(0.73)))
        }
        RequestField::TopK => {
            let request = base().top_k(37);
            (request, config(base().top_k(73)))
        }
        RequestField::Stop => {
            let request = base().stop(["allama-stop-a"]);
            (request, config(base().stop(["allama-stop-b"])))
        }
        RequestField::FrequencyPenalty => {
            let request = base().frequency_penalty(0.37);
            (request, config(base().frequency_penalty(0.73)))
        }
        RequestField::PresencePenalty => {
            let request = base().presence_penalty(0.37);
            (request, config(base().presence_penalty(0.73)))
        }
        RequestField::Seed => {
            let request = base().seed(37);
            (request, config(base().seed(73)))
        }
        RequestField::Tools => (
            tool_request(model),
            Judge::Content {
                payload: TOOL_NAME,
                names,
            },
        ),
        RequestField::ParallelToolCalls => {
            let request = tool_request(model).parallel_tool_calls(true);
            let contrast = tool_request(model).parallel_tool_calls(false);
            (request, config(contrast))
        }
        RequestField::ResponseFormat => (
            base().response_format(
                ResponseFormat::json_schema(honesty_schema()).name("allama_honesty"),
            ),
            Judge::Content {
                payload: SCHEMA_PROPERTY,
                names,
            },
        ),
        RequestField::Reasoning => {
            let request = base().reasoning(ReasoningEffort::High);
            (request, config(base().reasoning(ReasoningEffort::Low)))
        }
        RequestField::System => (
            base().system(SYSTEM_TEXT),
            Judge::Content {
                payload: SYSTEM_TEXT,
                names: Names::System,
            },
        ),
        RequestField::Cache => {
            let request = base().cache(
                PromptCache::new()
                    .key("allama-honesty-cache")
                    .retention(CacheRetention::Long),
            );
            (request, config(base()))
        }
        // Checked by the tool choice list; embedding fields are not chat's.
        RequestField::ToolChoice | RequestField::Dimensions | RequestField::InputType => {
            return FieldCheck::Elsewhere;
        }
        _ => return FieldCheck::Unlisted,
    };
    FieldCheck::Run(Box::new(Check {
        id: HonestyCheck::Field(field),
        request,
        judge,
    }))
}

fn part_check(model: &str, name: &str) -> Option<Check> {
    let (id, part, payload): (_, ContentPart, &'static str) = match name {
        "text" => (
            "text",
            ContentPart::text("allama-honesty-text"),
            "allama-honesty-text",
        ),
        "image" => (
            "image",
            ContentPart::Image(Image::base64("image/png", RED_PNG_BASE64)),
            RED_PNG_BASE64,
        ),
        "file" => (
            "file",
            ContentPart::File(File::base64("application/pdf", PINEAPPLE_PDF_BASE64)),
            PINEAPPLE_PDF_BASE64,
        ),
        _ => return None,
    };
    Some(Check {
        id: HonestyCheck::Part(id),
        request: ChatRequest::new(model).message(Message::user_parts(vec![
            ContentPart::text("Describe this."),
            part,
        ])),
        judge: Judge::Content {
            payload,
            names: Names::Message(0),
        },
    })
}

fn block_check(model: &str, block_type: StreamBlockType) -> Option<Check> {
    let replay = |blocks: Vec<ContentBlock>| {
        ChatRequest::new(model)
            .user("Help me.")
            .message(Message::assistant_blocks(blocks))
            .user("Go on.")
    };
    let (request, payload) = match block_type {
        StreamBlockType::Text => (
            replay(vec![ContentBlock::text("allama-honesty-reply")]),
            "allama-honesty-reply",
        ),
        StreamBlockType::Image => (
            replay(vec![ContentBlock::Image(Image::base64(
                "image/png",
                RED_PNG_BASE64,
            ))]),
            RED_PNG_BASE64,
        ),
        StreamBlockType::Reasoning => (
            replay(vec![
                ContentBlock::Reasoning(Reasoning::new("allama-honesty-reasoning")),
                ContentBlock::text("Sure."),
            ]),
            "allama-honesty-reasoning",
        ),
        StreamBlockType::Refusal => (
            replay(vec![ContentBlock::Refusal(Refusal::new(
                "allama-honesty-refusal",
            ))]),
            "allama-honesty-refusal",
        ),
        // Judged on the arguments: the id is a correlation key an adapter
        // may map to one its wire accepts (`ToolCallIdFormat`).
        StreamBlockType::ToolCall => {
            let call = ToolCall::new("call_honesty", TOOL_NAME, r#"{"q":"allama-honesty-call"}"#);
            let request = tool_request(model)
                .message(Message::assistant_blocks(vec![ContentBlock::ToolCall(
                    call,
                )]))
                .message(Message::tool_result("call_honesty", TOOL_NAME, "Rust."))
                .user("Thanks.");
            (request, "allama-honesty-call")
        }
        StreamBlockType::ProviderToolCall => (
            replay(vec![
                ContentBlock::ProviderToolCall(
                    ProviderToolCall::new(
                        "srvtoolu_allama_honesty",
                        "web_search",
                        r#"{"query":"allama-honesty-search"}"#,
                    )
                    .with_origin(FOREIGN),
                ),
                ContentBlock::text("Found it."),
            ]),
            "allama-honesty-search",
        ),
        StreamBlockType::ProviderToolResult => (
            replay(vec![
                ContentBlock::ProviderToolResult(
                    ProviderToolResult::new(
                        "srvtoolu_allama_honesty",
                        json!({"results": ["allama-honesty-result"]}),
                    )
                    .with_origin(FOREIGN),
                ),
                ContentBlock::text("Found it."),
            ]),
            "allama-honesty-result",
        ),
        StreamBlockType::Other => {
            let mut data = ExtraMap::new();
            data.insert("text".into(), json!("allama-honesty-other"));
            (
                replay(vec![
                    ContentBlock::Other(
                        OtherContent::new("vendor_block", data).with_origin(FOREIGN),
                    ),
                    ContentBlock::text("Done."),
                ]),
                "allama-honesty-other",
            )
        }
        _ => return None,
    };
    Some(Check {
        id: HonestyCheck::Block(block_type),
        request,
        judge: Judge::Content {
            payload,
            names: Names::Message(1),
        },
    })
}

/// The request a `Supported` claim of `capability` is judged on, when its
/// check asks for more than every provider with the capability takes.
///
/// - Structured output is a JSON Schema response format; its name is an
///   optional label and strict mode is a capability of its own
///   ([`assert_strict_not_honoured_warns`](crate::probe::assert_strict_not_honoured_warns)),
///   so the claim is judged on a schema without either.
/// - Prompt caching is breakpoints, a key and a retention, and providers
///   take different parts of it: Anthropic has no key, Gemini's implicit
///   caching takes neither key nor retention. A plain breakpoint is what
///   each takes (automatic prefix caching satisfies it), so the claim is
///   judged on one.
///
/// The checks themselves still probe the name, strict mode, key and
/// retention.
fn claim_probe(model: &str, capability: ChatCapability) -> Option<(ChatRequest, Names)> {
    match capability {
        ChatCapability::StructuredOutput => Some((
            ChatRequest::new(model)
                .user("Say hello.")
                .response_format(ResponseFormat::json_schema(honesty_schema()).non_strict()),
            Names::Field(RequestField::ResponseFormat),
        )),
        ChatCapability::PromptCache => Some((
            ChatRequest::new(model)
                .message(Message::user("Say hello.").with_cache(CacheBreakpoint::new())),
            Names::Cache,
        )),
        _ => None,
    }
}

fn honesty_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {SCHEMA_PROPERTY: {"type": "string"}},
        "required": [SCHEMA_PROPERTY],
        "additionalProperties": false
    })
}

fn tool_request(model: &str) -> ChatRequest {
    ChatRequest::new(model)
        .user("Look up rust.")
        .tools([ToolDefinition::new(
            TOOL_NAME,
            json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}),
        )
        .description("Look something up.")])
}

fn image_request(model: &str, detail: Option<ImageDetail>) -> ChatRequest {
    let mut image = Image::base64("image/png", RED_PNG_BASE64);
    if let Some(detail) = detail {
        image = image.with_detail(detail);
    }
    ChatRequest::new(model).message(Message::user_parts(vec![
        ContentPart::text("Describe the image."),
        ContentPart::Image(image),
    ]))
}

/// Which warnings name what a check probes: a refusal by warning must name
/// the capability's field or content.
#[derive(Clone, Copy, Debug)]
pub(super) enum Names {
    /// The request field.
    Field(RequestField),
    /// A system prompt.
    System,
    /// Message `index`, or a part or block of it.
    Message(usize),
    /// The tool choice (or the tools it applies to).
    ToolChoice,
    /// The prompt-cache settings, or a breakpoint on message 0.
    Cache,
    /// An embedding input, or a part of it.
    EmbeddingInput,
}

impl fmt::Display for Names {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Field(field) => write!(f, "field `{field}`"),
            Self::System => f.write_str("the system prompt"),
            Self::Message(index) => write!(f, "message {index}"),
            Self::ToolChoice => f.write_str("field `tool_choice` or `tools`"),
            Self::Cache => f.write_str("field `cache` or message 0"),
            Self::EmbeddingInput => f.write_str("an input"),
        }
    }
}

impl Names {
    pub(super) fn matches(self, subject: &Subject) -> bool {
        match (self, subject) {
            (Self::Field(field), Subject::Field { field: named }) => field == *named,
            (Self::System, Subject::System { .. }) => true,
            (
                Self::System,
                Subject::Field {
                    field: RequestField::System,
                },
            ) => true,
            (Self::Message(index), Subject::Content { message, .. }) => index == *message,
            (
                Self::ToolChoice,
                Subject::Field {
                    field: RequestField::ToolChoice | RequestField::Tools,
                },
            ) => true,
            (
                Self::Cache,
                Subject::Field {
                    field: RequestField::Cache,
                },
            ) => true,
            (Self::Cache, Subject::Content { message: 0, .. }) => true,
            (Self::EmbeddingInput, Subject::EmbeddingInput { .. }) => true,
            _ => false,
        }
    }
}

/// Whether the outcome settles the check on its own: `Ok(true)` when the
/// request was refused before sending or warned about, `Ok(false)` when it
/// was sent without a warning about it.
fn settled(outcome: &ProbeOutcome<ChatResponse>, names: Names) -> Result<bool, String> {
    match &outcome.result {
        Err(err @ Error::Unsupported { .. }) if !outcome.nothing_sent() => {
            Err(format!("refused only after the request was sent: {err}"))
        }
        Err(Error::Unsupported { .. }) => misshapen_refusal(outcome).map_or(Ok(true), Err),
        Err(_) => Err(format!("the probe failed: {}", describe(&outcome.result))),
        Ok(response) => Ok(response.warnings.iter().any(|w| names.matches(&w.subject))),
    }
}

/// What contradicts a `Supported` claim: a refusal before sending, or a
/// warning of kind `Unsupported` or `Dropped` that names what the request
/// uses. `Approximated` keeps the claim; a refusal after sending and other
/// failures are the judges' to report.
///
/// The replayed reasoning of the probe carries no provider's state (a
/// signature, encrypted content), which replay may need, so a drop coded
/// [`ReplayState::Reasoning`] keeps a `ReasoningReplay` claim. A capability
/// answer has no path, and streaming can refuse what `chat` takes, so on
/// `chat_stream` (`stream`) a refusal before sending keeps it too.
fn contradicted_claim(
    capability: ChatCapability,
    outcome: &ProbeOutcome<ChatResponse>,
    names: Names,
    stream: bool,
) -> Option<String> {
    let fix = "answer `Unknown` (or `Unsupported` if the endpoint cannot carry it), or \
               send it";
    match &outcome.result {
        Err(err @ Error::Unsupported { .. }) if outcome.nothing_sent() && !stream => Some(format!(
            "{capability:?} is claimed Supported, but the request was refused before sending \
             ({err}); {fix}"
        )),
        Ok(response) => response
            .warnings
            .iter()
            .find(|warning| {
                let stateless_reasoning = capability == ChatCapability::ReasoningReplay
                    && warning.code.as_deref() == Some(ReplayState::Reasoning.code());
                matches!(
                    warning.kind,
                    WarningKind::Unsupported | WarningKind::Dropped
                ) && names.matches(&warning.subject)
                    && !stateless_reasoning
            })
            .map(|warning| {
                format!(
                    "{capability:?} is claimed Supported, but the request was sent with a \
                     warning of kind `{}` naming {names} ({warning}); {fix}",
                    warning.kind
                )
            }),
        Err(_) => None,
    }
}

fn judge_content(
    outcome: &ProbeOutcome<ChatResponse>,
    payload: &'static str,
    names: Names,
) -> Option<String> {
    match settled(outcome, names) {
        Ok(true) => None,
        Ok(false) if outcome.contains(&WireValue::Text(payload)) => None,
        Ok(false) => Some(
            "the content never reached the wire, and no warning names it (dropped silently)"
                .to_owned(),
        ),
        Err(problem) => Some(problem),
    }
}

fn judge_config(
    outcome: &ProbeOutcome<ChatResponse>,
    contrast: &ProbeOutcome<ChatResponse>,
    names: Names,
) -> Option<String> {
    match settled(outcome, names) {
        Ok(true) => None,
        // The contrasting value is the one the provider cannot honor, so
        // this one is its default behavior (`parallel_tool_calls: true`
        // where only `false` is refused).
        Ok(false) if settled(contrast, names) == Ok(true) => None,
        Ok(false) => match &contrast.result {
            Ok(_) if outcome.differs(contrast) => None,
            Ok(_) => Some(
                "the request body is the same as without the setting, and no warning names it \
                 (dropped silently)"
                    .to_owned(),
            ),
            Err(_) => Some(format!(
                "the contrasting request failed: {}",
                describe(&contrast.result)
            )),
        },
        Err(problem) => Some(problem),
    }
}

#[cfg(test)]
mod tests {
    use allama::adapter::Degradations;
    use allama::mock::MockProvider;
    use allama::{ContentBlock, UnsupportedPolicy, UserContent, Warning, WarningKind};

    use super::*;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Fault {
        None,
        /// Leaves input images out without a word.
        DropImages,
        /// Never sends `tool_choice`.
        IgnoreToolChoice,
        /// Refuses images, but only after sending.
        RefuseImagesLate,
        /// Refuses images before sending, with `Error::unsupported`, whose
        /// stage says nothing about sending.
        RefuseImagesUnspecified,
        /// Refuses response schemas before sending.
        RefuseSchemas,
        /// Leaves response schemas out, with a warning of `kind`.
        WarnSchemas(WarningKind),
    }

    /// An adapter whose wire format is the serialized request.
    fn adapter(
        fault: Fault,
    ) -> impl Fn(ChatRequest) -> std::future::Ready<ProbeOutcome<ChatResponse>> {
        move |mut request: ChatRequest| {
            let has_image = request.messages.iter().any(|message| {
                matches!(message, Message::User(user)
                    if matches!(&user.content, UserContent::Parts(parts)
                        if parts.iter().any(|part| part.as_image().is_some())))
            });
            if fault == Fault::DropImages {
                for message in &mut request.messages {
                    if let Message::User(user) = message
                        && let UserContent::Parts(parts) = &mut user.content
                    {
                        parts.retain(|part| part.as_image().is_none());
                    }
                }
            }
            if fault == Fault::IgnoreToolChoice {
                request.tool_choice = None;
            }
            if fault == Fault::RefuseImagesUnspecified && has_image {
                let refused = Err(Error::unsupported("no images"));
                return std::future::ready(ProbeOutcome::new(refused, Vec::new()));
            }
            let has_schema = request.response_format.is_some();
            if fault == Fault::RefuseSchemas && has_schema {
                let refused = Degradations::new("fake", UnsupportedPolicy::Warn)
                    .unsupported(RequestField::ResponseFormat, "no schemas");
                return std::future::ready(ProbeOutcome::new(Err(refused), Vec::new()));
            }
            let mut warnings = Vec::new();
            if let Fault::WarnSchemas(kind) = fault
                && has_schema
            {
                if kind != WarningKind::Approximated {
                    request.response_format = None;
                }
                let subject = Subject::Field {
                    field: RequestField::ResponseFormat,
                };
                warnings.push(Warning::new(kind, subject, "schemas are not sent"));
            }
            let body = serde_json::to_value(&request).unwrap();
            let result = if fault == Fault::RefuseImagesLate && has_image {
                Err(Error::unsupported_by_provider(None, "HTTP 400"))
            } else {
                let mut response = ChatResponse::new(vec![ContentBlock::text("ok")]);
                response.warnings = warnings;
                Ok(response)
            };
            std::future::ready(ProbeOutcome::new(result, vec![body]))
        }
    }

    async fn run(fault: Fault, gaps: &[HonestyCheck]) {
        run_claiming(fault, &[], gaps).await;
    }

    /// [`run`] with `supported` claimed `Supported` for every model.
    async fn run_claiming(fault: Fault, supported: &[ChatCapability], gaps: &[HonestyCheck]) {
        let provider = MockProvider::with_text("ok")
            .with_supported_chat_capabilities(supported.iter().copied());
        assert_capabilities_honest(
            &provider,
            &HonestyUnderTest::new(["m"], adapter(fault), adapter(fault))
                .known_gaps(gaps.iter().copied()),
        )
        .await;
    }

    #[tokio::test]
    async fn an_adapter_that_sends_everything_passes() {
        run(Fault::None, &[]).await;
    }

    #[tokio::test]
    #[should_panic(expected = "`image` part: chat `m`: the content never reached the wire")]
    async fn a_silently_dropped_image_fails() {
        run(Fault::DropImages, &[]).await;
    }

    #[tokio::test]
    #[should_panic(expected = "sends the same body as tool choice")]
    async fn an_ignored_tool_choice_fails() {
        run(Fault::IgnoreToolChoice, &[]).await;
    }

    #[tokio::test]
    #[should_panic(expected = "refused only after the request was sent")]
    async fn a_refusal_after_sending_fails() {
        run(Fault::RefuseImagesLate, &[]).await;
    }

    #[tokio::test]
    #[should_panic(expected = "refused with stage unspecified, not before_sending")]
    async fn a_refusal_at_an_unspecified_stage_fails() {
        run(Fault::RefuseImagesUnspecified, &[]).await;
    }

    #[tokio::test]
    async fn a_known_gap_is_skipped() {
        // Dropping images drops their detail with them.
        run(
            Fault::DropImages,
            &[HonestyCheck::Part("image"), HonestyCheck::ImageDetail],
        )
        .await;
    }

    #[tokio::test]
    async fn a_refusal_of_an_unclaimed_capability_passes() {
        run(Fault::RefuseSchemas, &[]).await;
    }

    #[tokio::test]
    #[should_panic(
        expected = "field `response_format`: chat `m`: StructuredOutput is claimed Supported, \
                    but the request was refused before sending"
    )]
    async fn a_refused_supported_claim_fails() {
        run_claiming(
            Fault::RefuseSchemas,
            &[ChatCapability::StructuredOutput],
            &[],
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(
        expected = "StructuredOutput is claimed Supported, but the request was sent with a \
                    warning of kind `unsupported` naming field `response_format` (schemas are \
                    not sent)"
    )]
    async fn an_unsupported_warning_on_a_supported_claim_fails() {
        let fault = Fault::WarnSchemas(WarningKind::Unsupported);
        run_claiming(fault, &[ChatCapability::StructuredOutput], &[]).await;
    }

    #[tokio::test]
    #[should_panic(expected = "warning of kind `dropped` naming field `response_format`")]
    async fn a_dropped_warning_on_a_supported_claim_fails() {
        let fault = Fault::WarnSchemas(WarningKind::Dropped);
        run_claiming(fault, &[ChatCapability::StructuredOutput], &[]).await;
    }

    #[tokio::test]
    async fn an_approximated_supported_claim_passes() {
        let fault = Fault::WarnSchemas(WarningKind::Approximated);
        run_claiming(fault, &[ChatCapability::StructuredOutput], &[]).await;
    }

    /// Streaming can refuse what `chat` takes; the claim holds on `chat`.
    #[tokio::test]
    async fn a_supported_claim_refused_only_when_streaming_passes() {
        let provider = MockProvider::with_text("ok")
            .with_supported_chat_capabilities([ChatCapability::StructuredOutput]);
        assert_capabilities_honest(
            &provider,
            &HonestyUnderTest::new(["m"], adapter(Fault::None), adapter(Fault::RefuseSchemas)),
        )
        .await;
    }

    #[tokio::test]
    async fn a_refused_supported_claim_can_be_a_known_gap() {
        run_claiming(
            Fault::RefuseSchemas,
            &[ChatCapability::StructuredOutput],
            &[HonestyCheck::Field(RequestField::ResponseFormat)],
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "`image` part is listed as a known gap but passes; remove it")]
    async fn a_stale_known_gap_fails() {
        run(Fault::None, &[HonestyCheck::Part("image")]).await;
    }

    #[test]
    fn every_capability_has_a_check() {
        for &capability in ChatCapability::ALL {
            assert!(
                !matches!(capability_check(capability), Covered::Unlisted),
                "{capability:?}"
            );
        }
        let (_, missing) = checks("m");
        assert!(missing.is_empty(), "{missing:?}");
    }
}
