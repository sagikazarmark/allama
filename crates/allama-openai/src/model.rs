//! What OpenAI accepts per model, keyed by (model family, endpoint).
//!
//! Capability answers describe the endpoint, and the model only where the
//! answer is positive (ADR 0015). A model id is classified in three tiers:
//!
//! 1. a specialty marker settles it first: search, audio, realtime,
//!    transcription and speech variants do not inherit chat defaults, and
//!    answer `Unknown`;
//! 2. a parseable id in a known family (gpt-4o, gpt-4.1, gpt-5 and its
//!    point releases, GPT-6, the o-series) gets the family's positive
//!    answers, so a new point release (`gpt-5.7`) lands in its family; a
//!    fine-tuned id (`ft:gpt-4.1-mini:acme::1`) is its base model;
//! 3. anything else gets `Unknown` for every model-dependent capability.
//!
//! Nothing is refused, dropped or clamped away because of the model id:
//! which endpoint serves a model, which models reason, which take tools on
//! Chat Completions, and which reject sampling while they reason are model
//! claims, and OpenAI's rejection answers them. Their evidence is in
//! `docs/catalog/openai.md`; a caller re-adds such rules with the providers' `with_model_rules`. What
//! stays is wire format: the effort levels a known model takes, which the
//! Responses encoder maps a requested level to (ADR 0004).
//!
//! Sources, checked 2026-10-03, are cited where each fact is used. Model
//! pages are `https://developers.openai.com/api/docs/models/<id>`.

use allama::{CapabilitySupport, ChatCapability, ReasoningEffort};
use allama_openai_compat::{ChatModelRules, ResponsesModelRules};

/// Effort levels of the original GPT-5 models (verified live:
/// `gpt-5-nano` rejects `none` and `xhigh`).
const GPT5_LEVELS: &[ReasoningEffort] = &[
    ReasoningEffort::Minimal,
    ReasoningEffort::Low,
    ReasoningEffort::Medium,
    ReasoningEffort::High,
];

/// Effort levels of the o-series reasoning models.
const O_SERIES_LEVELS: &[ReasoningEffort] = &[
    ReasoningEffort::Low,
    ReasoningEffort::Medium,
    ReasoningEffort::High,
];

/// Effort levels of GPT-6 Astra: "reasoning.effort supports low, medium,
/// high, xhigh, and max", and "GPT-6 Astra does not support none reasoning
/// effort" (model page; <https://developers.openai.com/api/docs/guides/reasoning>).
/// `max` has no portable effort.
const ASTRA_LEVELS: &[ReasoningEffort] = &[
    ReasoningEffort::Low,
    ReasoningEffort::Medium,
    ReasoningEffort::High,
    ReasoningEffort::XHigh,
];

/// The OpenAI API a request goes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Endpoint {
    ChatCompletions,
    Responses,
}

/// A model family, from its id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    /// gpt-4o and gpt-4.1 (any size).
    Gpt4,
    /// gpt-5 (`minor` 0) and its point releases (`gpt-5.1`, ...).
    Gpt5 { minor: u32, size: Size },
    /// The chat snapshots (`gpt-5-chat-latest`), which are not the
    /// reasoning models of their family.
    Gpt5Chat,
    /// GPT-6: Astra, Sol, Luna, and their point releases.
    Gpt6 { minor: u32, tier: Gpt6Tier },
    /// o1, o3, o4-mini and later o-series models. `mini` marks the small
    /// models, whose input modalities differ (o3-mini takes no images,
    /// o4-mini does), so they claim none.
    OSeries { mini: bool },
    /// o1-mini, which does not share the o-series' answers: every
    /// model-dependent capability is `Unknown`.
    O1Mini,
}

/// The size of an original GPT-5 model, which decides its effort levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Size {
    /// `gpt-5`, `gpt-5-mini`, `gpt-5-nano`.
    Standard,
    /// Anything else in the family (a variant without verified levels).
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gpt6Tier {
    Astra,
    Sol,
    Luna,
}

/// What OpenAI accepts for one model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ModelProfile {
    family: Option<Family>,
    /// A search, audio, realtime, transcription or speech variant.
    specialty: bool,
    /// The caller's `ReasoningConfig` override, which the Responses request
    /// rules follow as preflight does.
    reasoning_override: Option<CapabilitySupport>,
}

impl ModelProfile {
    pub(crate) fn for_model(model: &str) -> Self {
        let name = base_model(model);
        Self {
            family: family(&name),
            specialty: is_specialty(&name),
            reasoning_override: None,
        }
    }

    /// The same profile with the caller's reasoning answer (`None` keeps
    /// the built-in one), for the Responses request rules.
    pub(crate) fn with_overrides(
        self,
        resolver: impl Fn(ChatCapability) -> Option<CapabilitySupport>,
    ) -> Self {
        Self {
            reasoning_override: resolver(ChatCapability::ReasoningConfig),
            ..self
        }
    }

    /// Whether the model reasons: `Supported` for the reasoning families,
    /// `Unknown` for anything else.
    pub(crate) fn reasoning(&self) -> CapabilitySupport {
        if let Some(answer) = self.reasoning_override {
            return answer;
        }
        match self.family {
            Some(Family::Gpt5 { .. } | Family::Gpt6 { .. } | Family::OSeries { .. })
                if !self.specialty =>
            {
                CapabilitySupport::Supported
            }
            _ => CapabilitySupport::Unknown,
        }
    }

    /// The effort levels the model accepts, when documented. Unrecognized
    /// ids, and point releases without documented levels, get none: their
    /// effort is sent unchanged.
    fn effort_levels(&self) -> Option<&'static [ReasoningEffort]> {
        if self.specialty {
            return None;
        }
        match self.family? {
            Family::Gpt5 {
                minor: 0,
                size: Size::Standard,
            } => Some(GPT5_LEVELS),
            Family::OSeries { .. } => Some(O_SERIES_LEVELS),
            Family::Gpt6 {
                minor: 0,
                tier: Gpt6Tier::Astra,
            } => Some(ASTRA_LEVELS),
            _ => None,
        }
    }

    /// The answer for `capability` on `endpoint`.
    pub(crate) fn chat_capability(
        &self,
        endpoint: Endpoint,
        capability: ChatCapability,
    ) -> CapabilitySupport {
        use CapabilitySupport::{Supported, Unknown, Unsupported};
        // Endpoint facts.
        match capability {
            ChatCapability::Streaming
            | ChatCapability::NativeStreaming
            | ChatCapability::PromptCache => return Supported,
            // Chat Completions streams ask for usage
            // (`stream_options.include_usage`), sent in a final chunk:
            // https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream_options
            // (captured in `fixtures/recorded/stream_text.sse`); Responses'
            // `response.completed` carries the response with its usage:
            // https://platform.openai.com/docs/api-reference/responses-streaming/response/completed
            // (captured in `fixtures/responses/text_stream.sse`).
            ChatCapability::StreamUsage => return Supported,
            // Generated images arrive as hosted-tool `Other` items on
            // Responses, and Chat Completions returns none: neither is a
            // portable image block.
            ChatCapability::ImageOutput | ChatCapability::ImageReplay => return Unsupported,
            // Chat Completions returns no reasoning content.
            ChatCapability::ReasoningOutput | ChatCapability::ReasoningReplay
                if endpoint == Endpoint::ChatCompletions =>
            {
                return Unsupported;
            }
            // The Responses encoder refuses every audio part: the API takes
            // audio on Chat Completions only.
            ChatCapability::AudioInput if endpoint == Endpoint::Responses => return Unsupported,
            _ => {}
        }
        if self.specialty {
            return Unknown;
        }
        let Some(family) = self.family else {
            return Unknown;
        };
        let tools = matches!(
            capability,
            ChatCapability::ToolCalls
                | ChatCapability::ParallelToolCalls
                | ChatCapability::ForcedToolChoice
        );
        match capability {
            ChatCapability::ReasoningConfig
            | ChatCapability::ReasoningOutput
            | ChatCapability::ReasoningReplay => self.reasoning(),
            _ if family == Family::O1Mini => Unknown,
            // GPT-6 takes tool calls on Chat Completions depending on the
            // tier and the reasoning effort ("GPT-6 Astra and GPT-6.1 Sol
            // support Chat Completions, but tool calling requires
            // Responses. GPT-6 Sol and GPT-6 Luna support function calling
            // in Chat Completions only with reasoning_effort: "none"",
            // <https://developers.openai.com/api/docs/guides/latest-model>),
            // so nothing is claimed there.
            _ if tools => match (endpoint, family) {
                (Endpoint::ChatCompletions, Family::Gpt6 { .. }) => Unknown,
                _ => Supported,
            },
            ChatCapability::ImageInput
            | ChatCapability::ImageDetail
            | ChatCapability::FileInput
                if matches!(family, Family::OSeries { mini: true }) =>
            {
                // o3-mini takes text only; o4-mini takes images. No claim
                // for the small models (ADR 0015: positive answers must be
                // true).
                Unknown
            }
            ChatCapability::ImageInput
            | ChatCapability::ImageDetail
            | ChatCapability::FileInput
            | ChatCapability::StructuredOutput => {
                // Documented for gpt-4o, gpt-4.1, gpt-5 and the o-series;
                // not verified for GPT-6.
                if matches!(family, Family::Gpt6 { .. }) {
                    Unknown
                } else {
                    Supported
                }
            }
            _ => Unknown,
        }
    }

    /// The request rules the Chat Completions encoder applies to this
    /// model: the documented effort levels, which are wire format, not model
    /// claims; nothing else.
    pub(crate) fn chat_rules(self) -> ChatModelRules {
        match self.effort_levels() {
            Some(levels) => ChatModelRules::new().effort_levels(levels),
            None => ChatModelRules::new(),
        }
    }

    /// The request rules the Responses encoder applies to this model: the
    /// reasoning answer (`Unsupported` only from a caller's override) and
    /// the documented effort levels.
    pub(crate) fn rules(self) -> ResponsesModelRules {
        let rules = ResponsesModelRules::new().reasoning(self.reasoning());
        match self.effort_levels() {
            Some(levels) => rules.effort_levels(levels),
            None => rules,
        }
    }
}

/// The id of the model behind `model`: lowercased, and a fine-tuned id
/// (`ft:gpt-4.1-nano-2025-04-14:openai::BTz2REMH`, the shape
/// <https://developers.openai.com/api/docs/guides/supervised-fine-tuning>
/// shows) reduced to its base model.
fn base_model(model: &str) -> String {
    let model = model.trim().to_ascii_lowercase();
    match model.strip_prefix("ft:") {
        Some(rest) => rest.split(':').next().unwrap_or(rest).to_owned(),
        None => model,
    }
}

/// Search (`-search-preview`), audio (`-audio-preview`), realtime,
/// transcription and speech models, which do not inherit chat defaults.
fn is_specialty(name: &str) -> bool {
    [
        "-search-preview",
        "-audio-preview",
        "realtime",
        "transcribe",
        "-tts",
    ]
    .iter()
    .any(|marker| name.contains(marker))
}

/// `5.1` → `(5, 1)`, `5` → `(5, 0)`.
fn version(token: &str) -> Option<(u32, u32)> {
    let mut parts = token.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = match parts.next() {
        Some(minor) => minor.parse().ok()?,
        None => 0,
    };
    parts.next().is_none().then_some((major, minor))
}

fn family(name: &str) -> Option<Family> {
    if name.starts_with("gpt-4o") {
        return Some(Family::Gpt4);
    }
    if let Some(rest) = name.strip_prefix("gpt-") {
        let mut tokens = rest.split('-');
        let (major, minor) = version(tokens.next()?)?;
        let rest: Vec<&str> = tokens.collect();
        return match (major, minor) {
            (4, 1) => Some(Family::Gpt4),
            (5, _) if rest.contains(&"chat") => Some(Family::Gpt5Chat),
            (5, minor) => {
                let base = strip_snapshot_date(&rest);
                let size = if matches!(base, [] | ["mini"] | ["nano"]) {
                    Size::Standard
                } else {
                    Size::Other
                };
                Some(Family::Gpt5 { minor, size })
            }
            (6, minor) => {
                let tier = match rest.first().copied()? {
                    "astra" => Gpt6Tier::Astra,
                    "sol" => Gpt6Tier::Sol,
                    "luna" => Gpt6Tier::Luna,
                    _ => return None,
                };
                Some(Family::Gpt6 { minor, tier })
            }
            _ => None,
        };
    }
    let mut tokens = name.split('-');
    let head = tokens.next()?;
    let digits = head.strip_prefix('o')?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if digits == "1" && tokens.next() == Some("mini") {
        return Some(Family::O1Mini);
    }
    Some(Family::OSeries {
        mini: tokens.any(|token| token == "mini"),
    })
}

/// `["mini", "2025", "08", "07"]` → `["mini"]`.
fn strip_snapshot_date<'a>(tokens: &'a [&'a str]) -> &'a [&'a str] {
    match tokens {
        [head @ .., year, month, day]
            if year.len() == 4
                && month.len() == 2
                && day.len() == 2
                && [year, month, day]
                    .iter()
                    .all(|t| t.bytes().all(|b| b.is_ascii_digit())) =>
        {
            head
        }
        _ => tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use CapabilitySupport::{Supported, Unknown, Unsupported};
    use Endpoint::{ChatCompletions, Responses};

    fn rules(model: &str) -> ResponsesModelRules {
        ModelProfile::for_model(model).rules()
    }

    fn answer(model: &str, endpoint: Endpoint, capability: ChatCapability) -> CapabilitySupport {
        ModelProfile::for_model(model).chat_capability(endpoint, capability)
    }

    #[test]
    fn families() {
        let p = rules("gpt-5-nano-2025-08-07");
        assert_eq!(p.reasoning, Supported);
        assert_eq!(p.effort_levels.as_deref(), Some(GPT5_LEVELS));

        let p = rules("gpt-5.1");
        assert_eq!(p.reasoning, Supported);
        assert_eq!(p.effort_levels, None);

        assert_eq!(
            rules("o4-mini").effort_levels.as_deref(),
            Some(O_SERIES_LEVELS)
        );
        assert_eq!(rules("my-proxy-model"), ResponsesModelRules::new());
    }

    /// The small o-series models claim no image or file input
    /// (o3-mini takes text only); the others keep the family's answers. The
    /// Chat Completions rules carry the same effort levels as Responses.
    #[test]
    fn small_o_series_models_claim_no_image_input() {
        for model in ["o3-mini", "o4-mini", "o3-mini-2025-01-31"] {
            assert_eq!(
                answer(model, ChatCompletions, ChatCapability::ImageInput),
                Unknown,
                "{model}"
            );
            assert_eq!(
                answer(model, Responses, ChatCapability::FileInput),
                Unknown,
                "{model}"
            );
            assert_eq!(
                answer(model, ChatCompletions, ChatCapability::StructuredOutput),
                Supported,
                "{model}"
            );
        }
        assert_eq!(
            answer("o3", ChatCompletions, ChatCapability::ImageInput),
            Supported
        );
        assert_eq!(
            ModelProfile::for_model("o3-mini")
                .chat_rules()
                .effort_levels
                .as_deref(),
            Some(O_SERIES_LEVELS)
        );
    }

    /// Nothing about sampling, tools or a default effort is assumed: the
    /// rules a caller does not set are the defaults, which send the request
    /// as asked (ADR 0015; the removed rules are in `docs/catalog/openai.md`).
    #[test]
    fn rules_assume_nothing_negative() {
        for model in [
            "gpt-5",
            "gpt-5.1",
            "o3",
            "o1-mini",
            "gpt-6-sol",
            "gpt-6-luna",
            "gpt-6-astra",
            "gpt-4o",
        ] {
            let p = rules(model);
            assert_ne!(p.reasoning, Unsupported, "{model}");
            assert!(!p.rejects_sampling(Some("high")), "{model}");
            assert!(!p.rejects_sampling(None), "{model}");
            assert_eq!(p.default_effort, None, "{model}");
        }
    }

    /// A model outside the reasoning families is not said not to reason:
    /// its reasoning settings are sent, and OpenAI decides.
    #[test]
    fn reasoning_is_never_refused_by_the_model() {
        for model in [
            "gpt-4o",
            "gpt-4.1-nano",
            "gpt-5-chat-latest",
            "o1-mini",
            "ft:gpt-4o-mini:acme::1",
        ] {
            for endpoint in [ChatCompletions, Responses] {
                assert_eq!(
                    answer(model, endpoint, ChatCapability::ReasoningConfig),
                    Unknown,
                    "{model}"
                );
            }
            assert_eq!(rules(model).reasoning, Unknown, "{model}");
        }
    }

    /// Version parsing places a new point release in its family, and a
    /// fine-tuned id is its base model.
    #[test]
    fn versions_and_fine_tunes() {
        assert_eq!(
            family("gpt-5.7"),
            Some(Family::Gpt5 {
                minor: 7,
                size: Size::Standard
            })
        );
        assert_eq!(
            family(&base_model("ft:gpt-4.1-nano-2025-04-14:openai::BTz2REMH")),
            Some(Family::Gpt4)
        );
        assert_eq!(
            answer(
                "ft:gpt-4.1-mini:acme::1",
                ChatCompletions,
                ChatCapability::StructuredOutput
            ),
            Supported
        );
        assert_eq!(family("gpt-6.2-luna").map(|_| ()), Some(()));
        for unknown in [
            "gpt-6-nova",
            "gpt-x",
            "o",
            "omni-moderation-latest",
            "davinci-002",
        ] {
            assert_eq!(family(unknown), None, "{unknown}");
        }
    }

    /// Which endpoint serves a model is model data (ADR 0015): a model OpenAI
    /// serves on Responses only is not refused on Chat Completions, and
    /// o1-mini is not refused on Responses. No `pro` token or `codex` or
    /// `deep-research` substring settles anything.
    #[test]
    fn no_model_is_refused_for_its_endpoint() {
        for model in [
            "o1-pro",
            "o3-pro",
            "gpt-5-pro",
            "gpt-5-codex",
            "codex-mini-latest",
            "o3-deep-research",
            "o4-mini-deep-research",
            "my-pro-model",
            "o1-mini",
        ] {
            for endpoint in [ChatCompletions, Responses] {
                assert_eq!(
                    answer(model, endpoint, ChatCapability::Streaming),
                    Supported,
                    "{model}"
                );
                assert_ne!(
                    answer(model, endpoint, ChatCapability::ToolCalls),
                    Unsupported,
                    "{model}"
                );
            }
        }
    }

    /// o1-mini does not share the o-series' answers, and is not refused
    /// anything either: every model-dependent capability is `Unknown`.
    #[test]
    fn o1_mini_is_unknown() {
        for capability in [
            ChatCapability::ToolCalls,
            ChatCapability::ForcedToolChoice,
            ChatCapability::StructuredOutput,
            ChatCapability::ImageInput,
            ChatCapability::FileInput,
            ChatCapability::ReasoningConfig,
        ] {
            for model in ["o1-mini", "o1-mini-2024-09-12"] {
                assert_eq!(answer(model, ChatCompletions, capability), Unknown);
            }
        }
        assert_eq!(rules("o1-mini").effort_levels, None);
        assert_eq!(
            answer("o1", ChatCompletions, ChatCapability::ImageInput),
            Supported
        );
        assert_eq!(
            answer("o1", ChatCompletions, ChatCapability::ReasoningConfig),
            Supported
        );
    }

    /// Specialty variants do not inherit chat defaults, and are refused
    /// nothing: every model-dependent capability is `Unknown`.
    #[test]
    fn specialty_variants_are_unknown() {
        for model in [
            "gpt-4o-search-preview",
            "gpt-4o-audio-preview",
            "gpt-realtime",
            "gpt-4o-transcribe",
            "gpt-4o-mini-tts",
        ] {
            for capability in [
                ChatCapability::ToolCalls,
                ChatCapability::ForcedToolChoice,
                ChatCapability::StructuredOutput,
                ChatCapability::ImageInput,
                ChatCapability::ReasoningConfig,
            ] {
                assert_eq!(
                    answer(model, ChatCompletions, capability),
                    Unknown,
                    "{model}"
                );
            }
        }
    }

    /// GPT-6 tool calls on Chat Completions depend on the tier and the
    /// effort, so they are `Unknown` there (source on `chat_capability`);
    /// Responses takes them.
    #[test]
    fn gpt_6() {
        for model in ["gpt-6-astra", "gpt-6.1-sol", "gpt-6-sol", "gpt-6-luna"] {
            for capability in [ChatCapability::ToolCalls, ChatCapability::ForcedToolChoice] {
                assert_eq!(answer(model, ChatCompletions, capability), Unknown);
                assert_eq!(answer(model, Responses, capability), Supported);
            }
        }
        assert_eq!(
            rules("gpt-6-astra").effort_levels.as_deref(),
            Some(ASTRA_LEVELS)
        );
        // A point release without documented levels sends its effort as is.
        assert_eq!(rules("gpt-6.1-astra").effort_levels, None);
        assert_eq!(
            answer("gpt-6-luna", Responses, ChatCapability::ImageInput),
            Unknown
        );
    }

    #[test]
    fn unknown_ids_are_unknown() {
        for capability in [
            ChatCapability::ToolCalls,
            ChatCapability::ForcedToolChoice,
            ChatCapability::StructuredOutput,
            ChatCapability::ReasoningConfig,
        ] {
            assert_eq!(
                answer("my-proxy-model", ChatCompletions, capability),
                Unknown
            );
            assert_eq!(answer("my-proxy-model", Responses, capability), Unknown);
        }
        assert_eq!(
            answer("my-proxy-model", ChatCompletions, ChatCapability::Streaming),
            Supported
        );
    }

    /// The Responses encoder refuses every audio
    /// part, so audio input is `Unsupported` there for every model, the
    /// audio models and ids the adapter does not know included. Chat
    /// Completions takes audio, and which model does is model data.
    #[test]
    fn audio_input_is_unsupported_on_responses() {
        for model in [
            "gpt-5-mini",
            "gpt-4o",
            "gpt-4o-audio-preview",
            "my-proxy-model",
        ] {
            assert_eq!(
                answer(model, Responses, ChatCapability::AudioInput),
                Unsupported,
                "{model}"
            );
            assert_eq!(
                answer(model, ChatCompletions, ChatCapability::AudioInput),
                Unknown,
                "{model}"
            );
        }
    }

    #[test]
    fn snapshot_dates() {
        assert_eq!(strip_snapshot_date(&["mini", "2025", "08", "07"]), ["mini"]);
        assert!(strip_snapshot_date(&["2025", "08", "07"]).is_empty());
        assert_eq!(strip_snapshot_date(&["codex"]), ["codex"]);
    }

    #[test]
    fn a_reasoning_override_reaches_the_rules() {
        let with = |answer| {
            ModelProfile::for_model("gpt-4.1")
                .with_overrides(|c| (c == ChatCapability::ReasoningConfig).then_some(answer))
                .rules()
                .reasoning
        };
        assert_eq!(with(Supported), Supported);
        assert_eq!(with(Unsupported), Unsupported);
    }
}
