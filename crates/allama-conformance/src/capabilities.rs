//! Capability claim audits.
//!
//! Capability answers are only useful when they are true in both directions:
//!
//! - **Honest whatever the answer** ([`assert_capabilities_honest`]): a
//!   request that uses a capability sends it, or warns, or is refused
//!   before sending, whether the answer is `Supported`, `Unsupported` or
//!   `Unknown` (for listed models, and for a model the provider cannot
//!   know). `Unknown` never means "drop it". It is the evidence for a chat
//!   adapter's `Supported` claims: a claimed capability that is refused,
//!   warned about as not sent, or never reaches the wire fails it.
//! - **Negative** ([`assert_unsupported_chat_capabilities_refused`],
//!   [`assert_unsupported_embedding_capabilities_refused`]): a request that
//!   uses a capability reported `Unsupported` must be refused: a warning
//!   whose subject names the capability's field or content, or an
//!   [`Error::Unsupported`] with stage
//!   [`BeforeSending`](allama::UnsupportedStage::BeforeSending) and nothing
//!   sent. Never a silent send, and never a refusal that leaves the caller
//!   guessing whether the request went out. These send the request through
//!   a [probe](crate::probe); the chat audit runs it on `chat` and on
//!   `chat_stream`.
//! - **Output** ([`assert_stream_usage_matches_capability`]): a capability
//!   that describes output is checked on the output. A stream reports usage
//!   when the provider answers `StreamUsage` `Supported`, and none when it
//!   answers `Unsupported`.
//! - **Embedding evidence** ([`assert_embedding_capabilities_backed`]):
//!   every embedding capability a provider reports as `Supported` names a
//!   fixture or test that exercises it.

use std::fmt::Debug;
use std::future::Future;
use std::path::Path;

use allama::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, ContentBlock,
    ContentPart, EmbeddingCapability, EmbeddingInput, EmbeddingInputType, EmbeddingPart,
    EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, Error, Image, ImageDetail, Message,
    PromptCache, ReasoningEffort, RequestField, ResponseFormat, ToolChoice, ToolDefinition,
    Warning,
};
use serde_json::json;

pub use honesty::{HonestyCheck, HonestyUnderTest, UNKNOWN_MODEL, assert_capabilities_honest};

use crate::FixtureDir;

use honesty::Names;

mod honesty;
use crate::probe::{CHAT, CHAT_STREAM, ProbeOutcome, Report, describe, misshapen_refusal, on_path};

/// Where a capability claim is proven.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Evidence {
    /// A fixture file (relative to the provider's fixture directory) that a
    /// test exercises; it must exist.
    Fixture(&'static str),
    /// The name of an offline test (for example an HTTP integration test
    /// against the local server).
    Test(&'static str),
    /// The name of a live e2e test.
    E2e(&'static str),
}

/// Assert that every `Supported` embedding capability of `model` is listed
/// in `evidence`, that every listed fixture exists, and that every named
/// test is defined in the crate (`src/`, `tests/` or `examples/` next to the
/// fixture directory).
///
/// Listing evidence for a capability that is not `Supported` also fails,
/// so the table stays in sync with the claims. Chat capabilities have no
/// such table: [`assert_capabilities_honest`] probes every one of them.
///
/// # Panics
///
/// On any unbacked claim, stale entry, or missing fixture.
pub fn assert_embedding_capabilities_backed(
    provider: &impl EmbeddingProvider,
    model: &str,
    fixtures: &FixtureDir,
    evidence: &[(EmbeddingCapability, Evidence)],
) {
    let claims = EmbeddingCapability::ALL
        .iter()
        .map(|&c| (c, provider.embedding_capability(model, c)));
    assert_backed(claims, model, fixtures, evidence);
}

/// Assert that every chat capability `provider` reports `Unsupported` for
/// `model` is refused: a probe request that uses it ends in a warning whose
/// subject names the capability's request field or content, or in an
/// [`Error::Unsupported`] with stage
/// [`BeforeSending`](allama::UnsupportedStage::BeforeSending) and nothing
/// sent. A warning about something else, a refusal at another stage (such
/// as [`Error::unsupported`]'s unspecified one) or after sending, and a
/// silent send all fail.
///
/// `probe` sends a request through the provider's `chat`, wired to a
/// transport that answers with a canned success response, and
/// `stream_probe` through its `chat_stream`, collecting the stream into a
/// response (see [streaming probes](crate::probe#streaming-probes)). Every
/// probe request goes through both. Capabilities that describe output rather
/// than a request feature (`Streaming`, `NativeStreaming`, `StreamUsage`,
/// `ImageOutput`, `ReasoningOutput`) have no probe and are skipped;
/// [`assert_stream_usage_matches_capability`] checks `StreamUsage` on the
/// output. So is `ParallelToolCalls`: no request can use it silently, since
/// `parallel_tool_calls: true` is permission, never a degradation, and
/// `false` is what a model without parallel calls does anyway.
///
/// `known_gaps` lists capabilities with an acknowledged defect: they are
/// skipped, but listing one that passes on both paths (or is not reported
/// `Unsupported`) fails, so fixed gaps leave the list.
///
/// # Panics
///
/// On any capability used silently, or a stale known gap.
pub async fn assert_unsupported_chat_capabilities_refused<F, Fut, G, GFut>(
    provider: &impl ChatProvider,
    model: &str,
    known_gaps: &[ChatCapability],
    mut probe: F,
    mut stream_probe: G,
) where
    F: FnMut(ChatRequest) -> Fut,
    Fut: Future<Output = ProbeOutcome<ChatResponse>>,
    G: FnMut(ChatRequest) -> GFut,
    GFut: Future<Output = ProbeOutcome<ChatResponse>>,
{
    let mut report = Report::default();
    for &capability in ChatCapability::ALL {
        let known_gap = known_gaps.contains(&capability);
        let request = chat_probe_request(model, capability);
        let (true, Some(request)) = (
            provider.chat_capability(model, capability) == CapabilitySupport::Unsupported,
            request,
        ) else {
            report.record(capability, None, known_gap);
            continue;
        };
        let names = chat_names(capability);
        let chat = probe(request.clone()).await;
        let stream = stream_probe(request).await;
        let problems: Vec<String> = [
            on_path(
                CHAT,
                refusal_problem(capability, &chat, names, |r| &r.warnings),
            ),
            on_path(
                CHAT_STREAM,
                refusal_problem(capability, &stream, names, |r| &r.warnings),
            ),
        ]
        .into_iter()
        .flatten()
        .collect();
        let problem = (!problems.is_empty()).then(|| problems.join("\n  "));
        report.record(capability, problem, known_gap);
    }
    report.finish("negative chat capability audit");
}

/// Assert that `provider`'s [`StreamUsage`](ChatCapability::StreamUsage)
/// answer for `model` matches its output: the response `stream_probe`
/// collects from `chat_stream` carries usage when the answer is
/// `Supported`, and none when it is `Unsupported`. `Unknown` promises
/// neither, so it is not checked.
///
/// `stream_probe` is the probe of
/// [`assert_unsupported_chat_capabilities_refused`]: it answers with a
/// stream the provider's API sends for the model, ideally a captured one,
/// since the check is only as good as that stream.
///
/// # Panics
///
/// When the stream fails, or its usage contradicts the answer.
pub async fn assert_stream_usage_matches_capability<G, GFut>(
    provider: &impl ChatProvider,
    model: &str,
    mut stream_probe: G,
) where
    G: FnMut(ChatRequest) -> GFut,
    GFut: Future<Output = ProbeOutcome<ChatResponse>>,
{
    let answer = provider.chat_capability(model, ChatCapability::StreamUsage);
    let outcome = stream_probe(ChatRequest::new(model).user("Say hello.")).await;
    let response = match outcome.result {
        Ok(response) => response,
        Err(error) => panic!("{model}: the stream probe failed: {error}"),
    };
    match answer {
        CapabilitySupport::Supported => assert!(
            response.usage.is_some(),
            "{model}: StreamUsage is reported Supported, but the collected stream has no usage"
        ),
        CapabilitySupport::Unsupported => assert!(
            response.usage.is_none(),
            "{model}: StreamUsage is reported Unsupported, but the collected stream has usage \
             {:?}",
            response.usage
        ),
        _ => {}
    }
}

/// [`assert_unsupported_chat_capabilities_refused`] for embedding
/// capabilities: `BatchInput` is probed with two inputs, `OutputDimensions`
/// with `dimensions`, `ImageInput` with an image input, `CompositeInput`
/// with an input of a text and an image part, and `InputType` with
/// `input_type: Query`. A warning refuses `dimensions` and `input_type`
/// when it names the field, and the inputs when it names an input.
///
/// # Panics
///
/// On any capability used silently, or a stale known gap.
pub async fn assert_unsupported_embedding_capabilities_refused<F, Fut>(
    provider: &impl EmbeddingProvider,
    model: &str,
    known_gaps: &[EmbeddingCapability],
    mut probe: F,
) where
    F: FnMut(EmbeddingRequest) -> Fut,
    Fut: Future<Output = ProbeOutcome<EmbeddingResponse>>,
{
    let mut report = Report::default();
    for &capability in EmbeddingCapability::ALL {
        let known_gap = known_gaps.contains(&capability);
        let request = embedding_probe_request(model, capability);
        let (true, Some(request)) = (
            provider.embedding_capability(model, capability) == CapabilitySupport::Unsupported,
            request,
        ) else {
            report.record(capability, None, known_gap);
            continue;
        };
        let outcome = probe(request).await;
        let problem = refusal_problem(capability, &outcome, embedding_names(capability), |r| {
            &r.warnings
        });
        report.record(capability, problem, known_gap);
    }
    report.finish("negative embedding capability audit");
}

/// A minimal request that uses `capability`, or `None` when the capability
/// describes output and no request can exercise it.
fn chat_probe_request(model: &str, capability: ChatCapability) -> Option<ChatRequest> {
    let base = ChatRequest::new(model);
    let tool = || {
        ToolDefinition::new(
            "lookup",
            json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}),
        )
        .description("Look something up.")
    };
    let image = || Image::base64("image/png", crate::e2e::RED_PNG_BASE64);
    let request = match capability {
        ChatCapability::ToolCalls => base.user("Look up rust.").tools([tool()]),
        ChatCapability::ForcedToolChoice => base
            .user("Look up rust.")
            .tools([tool()])
            .tool_choice(ToolChoice::Required),
        ChatCapability::ImageInput => base.message(Message::user_parts(vec![
            ContentPart::text("Describe the image."),
            ContentPart::Image(image()),
        ])),
        ChatCapability::ImageDetail => base.message(Message::user_parts(vec![
            ContentPart::text("Describe the image."),
            ContentPart::Image(image().with_detail(ImageDetail::High)),
        ])),
        ChatCapability::ImageReplay => base
            .user("Draw a red square.")
            .message(Message::assistant_blocks(vec![
                ContentBlock::Image(image()),
            ]))
            .user("Make it blue."),
        ChatCapability::FileInput => base.message(Message::user_parts(vec![
            ContentPart::text("Summarize the file."),
            ContentPart::file_base64("application/pdf", crate::e2e::PINEAPPLE_PDF_BASE64),
        ])),
        ChatCapability::AudioInput => base.message(Message::user_parts(vec![
            ContentPart::text("Transcribe the audio."),
            ContentPart::file_base64("audio/wav", "UklGRiQAAABXQVZF"),
        ])),
        ChatCapability::VideoInput => base.message(Message::user_parts(vec![
            ContentPart::text("Describe the video."),
            ContentPart::file_base64("video/mp4", "AAAAIGZ0eXBpc29t"),
        ])),
        ChatCapability::StructuredOutput => base.user("Say hello.").response_format(
            ResponseFormat::json_schema(json!({
                "type": "object",
                "properties": {"greeting": {"type": "string"}},
                "required": ["greeting"]
            }))
            .name("greeting"),
        ),
        ChatCapability::ReasoningReplay => base
            .user("What is 2 + 2?")
            .message(Message::assistant_blocks(vec![
                ContentBlock::reasoning("Two plus two is four."),
                ContentBlock::text("4"),
            ]))
            .user("And 3 + 3?"),
        ChatCapability::ReasoningConfig => base.user("Say hello.").reasoning(ReasoningEffort::High),
        ChatCapability::PromptCache => base
            .user("Say hello.")
            .cache(PromptCache::new().key("allama-cache-probe")),
        _ => return None,
    };
    Some(request)
}

/// A 1x1 PNG, for image probes.
const PROBE_PNG: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";

fn embedding_probe_request(
    model: &str,
    capability: EmbeddingCapability,
) -> Option<EmbeddingRequest> {
    let base = EmbeddingRequest::new(model);
    match capability {
        EmbeddingCapability::BatchInput => Some(base.inputs(["first", "second"])),
        EmbeddingCapability::OutputDimensions => Some(base.input("hello").dimensions(64)),
        EmbeddingCapability::ImageInput => {
            Some(base.input(EmbeddingInput::image_base64("image/png", PROBE_PNG)))
        }
        EmbeddingCapability::CompositeInput => Some(base.input(vec![
            EmbeddingPart::text("A small image"),
            EmbeddingPart::image_base64("image/png", PROBE_PNG),
        ])),
        EmbeddingCapability::InputType => {
            Some(base.input("hello").input_type(EmbeddingInputType::Query))
        }
        _ => None,
    }
}

/// Which warnings name what a chat capability's probe request uses.
fn chat_names(capability: ChatCapability) -> Names {
    match capability {
        ChatCapability::ToolCalls => Names::Field(RequestField::Tools),
        ChatCapability::ParallelToolCalls => Names::Field(RequestField::ParallelToolCalls),
        ChatCapability::ForcedToolChoice => Names::ToolChoice,
        ChatCapability::StructuredOutput => Names::Field(RequestField::ResponseFormat),
        ChatCapability::ReasoningConfig => Names::Field(RequestField::Reasoning),
        ChatCapability::PromptCache => Names::Field(RequestField::Cache),
        // The replayed assistant turn.
        ChatCapability::ImageReplay | ChatCapability::ReasoningReplay => Names::Message(1),
        // The user's parts: image input, its detail, file input.
        _ => Names::Message(0),
    }
}

/// Which warnings name what an embedding capability's probe request uses.
fn embedding_names(capability: EmbeddingCapability) -> Names {
    match capability {
        EmbeddingCapability::OutputDimensions => Names::Field(RequestField::Dimensions),
        EmbeddingCapability::InputType => Names::Field(RequestField::InputType),
        _ => Names::EmbeddingInput,
    }
}

/// `None` when a probe of an unsupported capability was refused: a warning
/// that names what it uses, or `Error::Unsupported` before sending.
fn refusal_problem<T>(
    capability: impl Debug,
    outcome: &ProbeOutcome<T>,
    names: Names,
    warnings: impl Fn(&T) -> &Vec<Warning>,
) -> Option<String> {
    match &outcome.result {
        Err(Error::Unsupported { .. }) => misshapen_refusal(outcome)
            .map(|problem| format!("{capability:?} is reported Unsupported, but it was {problem}")),
        Ok(response) if warnings(response).iter().any(|w| names.matches(&w.subject)) => None,
        Ok(response) if !warnings(response).is_empty() => Some(format!(
            "{capability:?} is reported Unsupported, but a request using it was sent with \
             warnings that do not name it ({names}): {:?}",
            warnings(response)
        )),
        Ok(_) => Some(format!(
            "{capability:?} is reported Unsupported, but a request using it was sent \
             without a warning"
        )),
        Err(_) => Some(format!(
            "{capability:?} is reported Unsupported; expected a warning or Unsupported, got {}",
            describe(&outcome.result)
        )),
    }
}

fn assert_backed<C: Copy + PartialEq + Debug>(
    claims: impl Iterator<Item = (C, CapabilitySupport)>,
    model: &str,
    fixtures: &FixtureDir,
    evidence: &[(C, Evidence)],
) {
    let mut problems = Vec::new();
    for (capability, support) in claims {
        let backed = evidence.iter().any(|(c, _)| *c == capability);
        match (support, backed) {
            (CapabilitySupport::Supported, false) => problems.push(format!(
                "{capability:?} is claimed Supported for `{model}` without evidence"
            )),
            (CapabilitySupport::Supported, true) | (_, false) => {}
            (other, true) => problems.push(format!(
                "{capability:?} has evidence but is reported {other:?} for `{model}`"
            )),
        }
    }
    let crate_root = fixtures.as_path().parent().unwrap_or(fixtures.as_path());
    let mut sources = String::new();
    for dir in ["src", "tests", "examples"] {
        collect_sources(&crate_root.join(dir), &mut sources);
    }
    for (capability, item) in evidence {
        match item {
            Evidence::Fixture(name) if !fixtures.as_path().join(name).is_file() => {
                problems.push(format!(
                    "{capability:?}: fixture `{name}` does not exist in {}",
                    fixtures.as_path().display()
                ));
            }
            Evidence::Test(name) | Evidence::E2e(name)
                if !sources.contains(&format!("fn {name}(")) =>
            {
                problems.push(format!(
                    "{capability:?}: test `{name}` does not exist in {}",
                    crate_root.display()
                ));
            }
            _ => {}
        }
    }
    assert!(
        problems.is_empty(),
        "capability audit failed:\n  {}",
        problems.join("\n  ")
    );
}

fn collect_sources(dir: &Path, out: &mut String) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs")
            && let Ok(text) = std::fs::read_to_string(&path)
        {
            out.push_str(&text);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use allama::mock::{MockEmbeddingProvider, MockProvider};

    // The crate root doubles as the "fixture directory" here, so the "crate
    // root" `assert_backed` derives from it is the directory holding the
    // workspace's crates, which has no `src`, `tests` or `examples`; test
    // names then resolve against nothing unless a test points at this crate
    // explicitly (`src_fixtures`).
    fn fixtures() -> FixtureDir {
        FixtureDir::new(env!("CARGO_MANIFEST_DIR"))
    }

    fn src_fixtures() -> FixtureDir {
        FixtureDir::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"))
    }

    fn dimensions(support: CapabilitySupport) -> MockEmbeddingProvider {
        MockEmbeddingProvider::empty()
            .with_embedding_capability(EmbeddingCapability::OutputDimensions, support)
    }

    #[test]
    fn named_tests_must_exist() {
        assert_embedding_capabilities_backed(
            &dimensions(CapabilitySupport::Supported),
            "m",
            &src_fixtures(),
            &[(
                EmbeddingCapability::OutputDimensions,
                Evidence::Test("named_tests_must_exist"),
            )],
        );
    }

    #[test]
    fn named_tests_may_live_in_examples() {
        // An example built with `test = true` runs its tests under `cargo test`.
        let root = tempfile::TempDir::new().unwrap();
        std::fs::create_dir(root.path().join("examples")).unwrap();
        std::fs::write(
            root.path().join("examples/adapter.rs"),
            "#[test]\nfn dimensions_round_trip() {}\n",
        )
        .unwrap();
        assert_embedding_capabilities_backed(
            &dimensions(CapabilitySupport::Supported),
            "m",
            &FixtureDir::new(root.path().join("fixtures")),
            &[(
                EmbeddingCapability::OutputDimensions,
                Evidence::Test("dimensions_round_trip"),
            )],
        );
    }

    #[test]
    #[should_panic(expected = "test `no_such_test` does not exist")]
    fn fails_on_unknown_test_name() {
        assert_embedding_capabilities_backed(
            &dimensions(CapabilitySupport::Supported),
            "m",
            &src_fixtures(),
            &[(
                EmbeddingCapability::OutputDimensions,
                Evidence::E2e("no_such_test"),
            )],
        );
    }

    #[test]
    fn passes_when_every_claim_is_backed() {
        assert_embedding_capabilities_backed(
            &dimensions(CapabilitySupport::Supported),
            "m",
            &fixtures(),
            &[(
                EmbeddingCapability::OutputDimensions,
                Evidence::Fixture("Cargo.toml"),
            )],
        );
    }

    #[test]
    #[should_panic(expected = "without evidence")]
    fn fails_on_unbacked_claim() {
        assert_embedding_capabilities_backed(
            &dimensions(CapabilitySupport::Supported),
            "m",
            &fixtures(),
            &[],
        );
    }

    #[test]
    #[should_panic(expected = "has evidence but is reported Unsupported")]
    fn fails_on_stale_evidence() {
        assert_embedding_capabilities_backed(
            &dimensions(CapabilitySupport::Unsupported),
            "m",
            &fixtures(),
            &[(
                EmbeddingCapability::OutputDimensions,
                Evidence::Fixture("Cargo.toml"),
            )],
        );
    }

    #[test]
    #[should_panic(expected = "does not exist")]
    fn fails_on_missing_fixture() {
        assert_embedding_capabilities_backed(
            &dimensions(CapabilitySupport::Supported),
            "m",
            &fixtures(),
            &[(
                EmbeddingCapability::OutputDimensions,
                Evidence::Fixture("nope.json"),
            )],
        );
    }

    /// A fake adapter that sends everything and answers with `warnings`.
    fn sends_everything(
        warnings: Vec<Warning>,
    ) -> impl FnMut(ChatRequest) -> std::future::Ready<ProbeOutcome<ChatResponse>> {
        move |_request| {
            let mut response = ChatResponse::new(vec![ContentBlock::text("ok")]);
            response.warnings.clone_from(&warnings);
            std::future::ready(ProbeOutcome::new(Ok(response), vec![json!({})]))
        }
    }

    fn rejects_tools(request: ChatRequest) -> std::future::Ready<ProbeOutcome<ChatResponse>> {
        let outcome = if !request.tools.is_empty() {
            let refusal =
                allama::adapter::Degradations::new("fake", allama::UnsupportedPolicy::Warn)
                    .unsupported(RequestField::Tools, "no tools");
            ProbeOutcome::new(Err(refusal), Vec::new())
        } else {
            ProbeOutcome::new(
                Ok(ChatResponse::new(vec![ContentBlock::text("ok")])),
                vec![json!({})],
            )
        };
        std::future::ready(outcome)
    }

    fn without_tools() -> MockProvider {
        MockProvider::build(|b| b.text("x"))
            .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Unsupported)
    }

    #[tokio::test]
    async fn unsupported_capability_refused_with_error_passes() {
        assert_unsupported_chat_capabilities_refused(
            &without_tools(),
            "m",
            &[],
            rejects_tools,
            rejects_tools,
        )
        .await;
    }

    #[tokio::test]
    async fn unsupported_capability_with_warning_passes() {
        let warning = Warning::unsupported(RequestField::Tools, "dropped");
        assert_unsupported_chat_capabilities_refused(
            &without_tools(),
            "m",
            &[],
            sends_everything(vec![warning.clone()]),
            sends_everything(vec![warning]),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "ToolCalls is reported Unsupported, but a request using it was sent")]
    async fn silent_send_of_unsupported_capability_fails() {
        assert_unsupported_chat_capabilities_refused(
            &without_tools(),
            "m",
            &[],
            sends_everything(Vec::new()),
            sends_everything(Vec::new()),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(
        expected = "chat_stream: ToolCalls is reported Unsupported, but a request using it was sent"
    )]
    async fn silent_send_only_when_streaming_fails() {
        assert_unsupported_chat_capabilities_refused(
            &without_tools(),
            "m",
            &[],
            rejects_tools,
            sends_everything(Vec::new()),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "ToolCalls is listed as a known gap but passes")]
    async fn stale_capability_gap_fails() {
        assert_unsupported_chat_capabilities_refused(
            &without_tools(),
            "m",
            &[ChatCapability::ToolCalls],
            rejects_tools,
            rejects_tools,
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "sent with warnings that do not name it (field `tools`)")]
    async fn a_warning_about_something_else_is_no_refusal() {
        let warning = Warning::unsupported(RequestField::Seed, "no seed");
        assert_unsupported_chat_capabilities_refused(
            &without_tools(),
            "m",
            &[],
            sends_everything(vec![warning.clone()]),
            sends_everything(vec![warning]),
        )
        .await;
    }

    /// `Error::unsupported` says nothing about whether the request was
    /// sent, so it is no refusal before sending.
    #[tokio::test]
    #[should_panic(
        expected = "ToolCalls is reported Unsupported, but it was refused with stage \
                               unspecified"
    )]
    async fn a_refusal_at_an_unspecified_stage_fails() {
        let refuses = |_request| {
            std::future::ready(ProbeOutcome::<ChatResponse>::new(
                Err(Error::unsupported("no tools")),
                Vec::new(),
            ))
        };
        assert_unsupported_chat_capabilities_refused(&without_tools(), "m", &[], refuses, refuses)
            .await;
    }

    #[tokio::test]
    #[should_panic(expected = "only after sending")]
    async fn a_refusal_after_sending_fails() {
        let refuses = |_request| {
            let refusal =
                allama::adapter::Degradations::new("fake", allama::UnsupportedPolicy::Warn)
                    .unsupported(RequestField::Tools, "no tools");
            std::future::ready(ProbeOutcome::<ChatResponse>::new(
                Err(refusal),
                vec![json!({})],
            ))
        };
        assert_unsupported_chat_capabilities_refused(&without_tools(), "m", &[], refuses, refuses)
            .await;
    }

    #[tokio::test]
    async fn output_only_capabilities_are_skipped() {
        let provider = MockProvider::build(|b| b.text("x"))
            .with_chat_capability(ChatCapability::ImageOutput, CapabilitySupport::Unsupported);
        assert_unsupported_chat_capabilities_refused(
            &provider,
            "m",
            &[],
            sends_everything(Vec::new()),
            sends_everything(Vec::new()),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "OutputDimensions is reported Unsupported, but a request using it")]
    async fn embedding_dimensions_sent_silently_fails() {
        let provider = MockEmbeddingProvider::empty().with_embedding_capability(
            EmbeddingCapability::OutputDimensions,
            CapabilitySupport::Unsupported,
        );
        assert_unsupported_embedding_capabilities_refused(&provider, "m", &[], |_request| async {
            ProbeOutcome::new(Ok(EmbeddingResponse::new(vec![vec![0.0]])), vec![json!({})])
        })
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "InputType is reported Unsupported, but a request using it")]
    async fn embedding_input_type_sent_silently_fails() {
        let provider = MockEmbeddingProvider::empty().with_embedding_capability(
            EmbeddingCapability::InputType,
            CapabilitySupport::Unsupported,
        );
        assert_unsupported_embedding_capabilities_refused(&provider, "m", &[], |_request| async {
            ProbeOutcome::new(Ok(EmbeddingResponse::new(vec![vec![0.0]])), vec![json!({})])
        })
        .await;
    }

    #[tokio::test]
    async fn embedding_input_type_dropped_by_preflight_passes() {
        let provider = MockEmbeddingProvider::empty().with_embedding_capability(
            EmbeddingCapability::InputType,
            CapabilitySupport::Unsupported,
        );
        assert_unsupported_embedding_capabilities_refused(&provider, "m", &[], |request| {
            let provider = provider.clone();
            async move {
                let mut d = allama::adapter::Degradations::new(
                    "mock",
                    request.unsupported_policy.unwrap_or_default(),
                );
                let mut checked = std::borrow::Cow::Borrowed(&request);
                let result = allama::adapter::embedding::preflight(&provider, &mut checked, &mut d)
                    .map(|()| {
                        let mut response = EmbeddingResponse::new(vec![vec![0.0]]);
                        response.warnings = d.into_warnings();
                        response
                    });
                ProbeOutcome::new(result, vec![json!({})])
            }
        })
        .await;
    }
}
