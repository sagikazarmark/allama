//! Model-family knowledge used to pick the right wire shape per model.
//!
//! Anthropic's request surface differs across model generations (thinking
//! configuration, effort levels). The adapter derives a [`ModelProfile`]
//! from the model id and encodes portable request fields accordingly. An id
//! parses as `claude-{family}-{major}[-{minor}][-{date}]` (or the legacy
//! `claude-{major}[-{minor}]-{family}`), behind an optional `anthropic.`
//! (with a Bedrock inference profile's region, `us.anthropic.`) or path
//! prefix and before an optional `@date` (see the models overview,
//! <https://platform.claude.com/docs/en/about-claude/models/overview>).
//!
//! The profile picks **wire formats** (budget or adaptive thinking, how
//! thinking is turned off, the effort levels, interleaving) and answers
//! **positive** capability claims. It never refuses, drops or rewrites
//! what a caller asks for because of the model id (ADR 0015): a setting a
//! model rejects is sent, and the API's refusal maps to
//! `Error::Unsupported { stage: Provider }`. A capability the id does not
//! settle answers `Unknown`. An id that does not parse (a typo, a gateway
//! alias) gets the newest wire format and `Unknown` for every
//! model-dependent capability.
//!
//! A caller (or a model catalog) settles what the id does not through the
//! provider's capability resolver ([`Provider::with_chat_capabilities`])
//! and its model rules ([`Provider::with_model_rules`]).
//!
//! [`Provider::with_chat_capabilities`]: crate::Provider::with_chat_capabilities
//! [`Provider::with_model_rules`]: crate::Provider::with_model_rules

use allama::{CapabilitySupport, ChatCapability, ReasoningEffort};

/// Wire-format rules for one model, which replace what the adapter derives
/// from the model id.
///
/// Install them with [`Provider::with_model_rules`](crate::Provider::with_model_rules).
/// A field left unset keeps the adapter's own choice.
///
/// ```
/// use allama_anthropic::{EffortLevel, ModelRules, Provider};
///
/// # fn example(provider: Provider) -> Provider {
/// // A model that has gained `xhigh` since this release.
/// provider.with_model_rules(|model: &str| {
///     (model == "claude-opus-4-6").then(|| {
///         ModelRules::new().effort_levels([
///             EffortLevel::Low,
///             EffortLevel::Medium,
///             EffortLevel::High,
///             EffortLevel::XHigh,
///             EffortLevel::Max,
///         ])
///     })
/// })
/// # }
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModelRules {
    /// The `output_config.effort` levels the model takes. A reasoning
    /// effort is sent as the nearest of them, with an `Approximated`
    /// warning when it is not sent as asked.
    pub effort_levels: Option<Vec<EffortLevel>>,
}

impl ModelRules {
    /// Rules that keep every adapter choice.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the `output_config.effort` levels the model takes.
    #[must_use]
    pub fn effort_levels(mut self, levels: impl IntoIterator<Item = EffortLevel>) -> Self {
        self.effort_levels = Some(levels.into_iter().collect());
        self
    }
}

/// An `output_config.effort` level, in Anthropic's vocabulary
/// (<https://platform.claude.com/docs/en/build-with-claude/effort>).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum EffortLevel {
    /// `low`.
    Low,
    /// `medium`.
    Medium,
    /// `high`.
    High,
    /// `xhigh`.
    XHigh,
    /// `max`.
    Max,
}

impl EffortLevel {
    /// The wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }

    const fn bit(self) -> u8 {
        match self {
            Self::Low => 1,
            Self::Medium => 1 << 1,
            Self::High => 1 << 2,
            Self::XHigh => 1 << 3,
            Self::Max => 1 << 4,
        }
    }
}

/// Behavioral profile of an Anthropic model, derived from its id, with the
/// caller's capability answers and model rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ModelProfile {
    known: Option<(Family, u32, u32)>,
    overrides: Overrides,
}

impl ModelProfile {
    pub(crate) fn for_model(model: &str) -> Self {
        Self {
            known: parse(model),
            overrides: Overrides::default(),
        }
    }

    /// The same profile with the answers `resolver` gives (`None` keeps the
    /// built-in answer), so a capability override reaches the encoder: a
    /// structured output or forced tool choice answered `Unsupported` is
    /// refused, and a reasoning config answered `Unsupported` is dropped,
    /// as preflight does.
    pub(crate) fn with_overrides(
        mut self,
        resolver: impl Fn(ChatCapability) -> Option<CapabilitySupport>,
    ) -> Self {
        self.overrides.structured_output = resolver(ChatCapability::StructuredOutput);
        self.overrides.thinking = resolver(ChatCapability::ReasoningConfig);
        self.overrides.forced_tool_choice = resolver(ChatCapability::ForcedToolChoice);
        self
    }

    /// The same profile with the caller's model rules.
    pub(crate) fn with_rules(mut self, rules: &ModelRules) -> Self {
        if let Some(levels) = &rules.effort_levels {
            self.overrides.effort_levels = Some(EffortLevels::of(levels));
        }
        self
    }

    fn at_least(&self, family: Family, major: u32, minor: u32) -> bool {
        matches!(self.known, Some((f, ma, mi)) if f == family && (ma, mi) >= (major, minor))
    }

    fn is(&self, family: Family) -> bool {
        matches!(self.known, Some((f, _, _)) if f == family)
    }

    fn frontier(&self) -> bool {
        self.is(Family::Fable) || self.is(Family::Mythos)
    }

    /// Whether the id was recognized at all.
    pub(crate) fn is_known(&self) -> bool {
        self.known.is_some()
    }

    /// Native structured outputs (`output_config.format`): `Supported` from
    /// Claude Haiku 4.5, Sonnet 4.5 and Opus 4.1 on, and on Fable and
    /// Mythos; `Unknown` otherwise (the API decides).
    pub(crate) fn structured_output(&self) -> CapabilitySupport {
        if let Some(answer) = self.overrides.structured_output {
            return answer;
        }
        let supported = self.at_least(Family::Haiku, 4, 5)
            || self.at_least(Family::Sonnet, 4, 5)
            || self.at_least(Family::Opus, 4, 1)
            || self.frontier();
        if supported {
            CapabilitySupport::Supported
        } else {
            CapabilitySupport::Unknown
        }
    }

    /// Extended thinking (any style): `Supported` from Claude 4 on, on
    /// Claude Sonnet 3.7 and on Mythos Preview; `Unknown` otherwise.
    pub(crate) fn thinking(&self) -> CapabilitySupport {
        if let Some(answer) = self.overrides.thinking {
            return answer;
        }
        match self.known {
            Some((Family::Sonnet, 3, 7) | (Family::MythosPreview, ..)) => {
                CapabilitySupport::Supported
            }
            Some((_, major, _)) if major >= 4 => CapabilitySupport::Supported,
            _ => CapabilitySupport::Unknown,
        }
    }

    /// Forced tool choice (`tool_choice` `any`/`tool`): `Supported` for
    /// the generations documented to take it, the ones before Claude Opus
    /// and Sonnet 5.5 and Fable and Mythos 5.1; `Unknown` otherwise.
    pub(crate) fn forced_tool_choice(&self) -> CapabilitySupport {
        if let Some(answer) = self.overrides.forced_tool_choice {
            return answer;
        }
        let documented = self.is_known()
            && !(self.at_least(Family::Opus, 5, 5)
                || self.at_least(Family::Sonnet, 5, 5)
                || self.at_least(Family::Fable, 5, 1)
                || self.at_least(Family::Mythos, 5, 1));
        if documented {
            CapabilitySupport::Supported
        } else {
            CapabilitySupport::Unknown
        }
    }

    /// How an effort is sent. The budget form is for an enumerated list of
    /// legacy generations: Claude Sonnet 3.7, the Claude 4.0 and 4.1
    /// models, Opus 4.5, and Sonnet and Haiku 4.5
    /// (<https://platform.claude.com/docs/en/build-with-claude/extended-thinking>),
    /// and the Claude 3 models before them, which predate adaptive thinking.
    /// Every other id gets the adaptive form, the newest documented one
    /// (ADR 0015): later versions of a known family, versions this release
    /// has not seen, and ids that do not parse, whose effort is sent
    /// unchanged. It is a wire shape, not a claim: an unparsed id's
    /// capabilities stay `Unknown`.
    pub(crate) fn thinking_style(&self) -> ThinkingStyle {
        match self.known {
            Some(
                (Family::Haiku | Family::Sonnet | Family::Opus, 3, _)
                | (Family::Haiku | Family::Sonnet | Family::Opus, 4, 0 | 1 | 5),
            ) => ThinkingStyle::Budget,
            _ => ThinkingStyle::Adaptive,
        }
    }

    /// Whether manual thinking (`{type: "enabled"}`) interleaves with tool
    /// calls under the `interleaved-thinking-2025-05-14` beta: Claude 4
    /// models through Opus 4.5 and Sonnet 4.6, but not Haiku 4.5 (the
    /// header is ignored) or Opus 4.6 (only its adaptive mode interleaves).
    ///
    /// Source: <https://platform.claude.com/docs/en/build-with-claude/extended-thinking#interleaved-thinking-in-manual-mode>.
    pub(crate) fn interleaves_manual_thinking(&self) -> bool {
        matches!(
            self.known,
            Some((Family::Opus, 4, 0..=5) | (Family::Sonnet, 4, 0..=6))
        )
    }

    /// How a reasoning effort of `None` is sent: budget-style models are
    /// off unless enabled; Claude Sonnet 5.5 takes `between_tools`
    /// (<https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting>);
    /// every other adaptive model, unknown ids included, gets the adaptive
    /// format's off value, `disabled`, which a model that cannot turn
    /// thinking off refuses.
    pub(crate) fn disable_thinking(&self) -> DisableThinking {
        if self.thinking_style() == ThinkingStyle::Budget {
            DisableThinking::Omit
        } else if self.at_least(Family::Sonnet, 5, 5) {
            DisableThinking::BetweenTools
        } else {
            DisableThinking::Disabled
        }
    }

    /// Effort levels accepted by `output_config.effort`, or `None` when the
    /// model takes no effort (budget-style models). The caller's model
    /// rules win; an unknown id gets every level, so an effort is sent
    /// unchanged.
    pub(crate) fn effort_levels(&self) -> Option<EffortLevels> {
        if let Some(levels) = self.overrides.effort_levels {
            return Some(levels);
        }
        match self.known {
            None => Some(EffortLevels::FULL),
            Some((Family::Sonnet | Family::Opus, 4, 6)) => Some(EffortLevels::NO_XHIGH),
            Some(_) if self.thinking_style() == ThinkingStyle::Adaptive => Some(EffortLevels::FULL),
            Some(_) => None,
        }
    }

    /// Whether the model thinks when the request leaves `thinking` out:
    /// Claude Opus and Sonnet 5 and later, Haiku 5.5 and later, Fable, and
    /// the Mythos models
    /// (<https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking>).
    /// A positive claim only: `false` covers the models documented to run
    /// without thinking (Opus 4.6 to 4.8, Sonnet 4.6, Haiku 4.5, the
    /// budget-style generations) and every id that does not parse.
    pub(crate) fn thinks_by_default(&self) -> bool {
        self.at_least(Family::Opus, 5, 0)
            || self.at_least(Family::Sonnet, 5, 0)
            || self.at_least(Family::Haiku, 5, 5)
            || self.frontier()
            || self.is(Family::MythosPreview)
    }

    /// Whether `thinking.display` is accepted (adaptive-thinking models).
    pub(crate) fn supports_thinking_display(&self) -> bool {
        self.thinking_style() == ThinkingStyle::Adaptive
    }
}

/// How extended thinking is configured for a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ThinkingStyle {
    /// `{type: "enabled", budget_tokens}`; off unless enabled.
    Budget,
    /// `{type: "adaptive"}` plus `output_config.effort`.
    Adaptive,
}

/// How a model turns thinking off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DisableThinking {
    /// Thinking is off unless enabled: omit the `thinking` field.
    Omit,
    /// Send `{type: "disabled"}`.
    Disabled,
    /// Send `{type: "between_tools"}` (Claude Sonnet 5.5).
    BetweenTools,
}

/// A set of [`EffortLevel`]s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EffortLevels(u8);

impl EffortLevels {
    const FULL: Self = Self::of_const(&[
        EffortLevel::Low,
        EffortLevel::Medium,
        EffortLevel::High,
        EffortLevel::XHigh,
        EffortLevel::Max,
    ]);
    /// The 4.6 generation has no `xhigh`.
    const NO_XHIGH: Self = Self::of_const(&[
        EffortLevel::Low,
        EffortLevel::Medium,
        EffortLevel::High,
        EffortLevel::Max,
    ]);

    const fn of_const(levels: &[EffortLevel]) -> Self {
        let mut bits = 0;
        let mut i = 0;
        while i < levels.len() {
            bits |= levels[i].bit();
            i += 1;
        }
        Self(bits)
    }

    fn of(levels: &[EffortLevel]) -> Self {
        Self::of_const(levels)
    }

    fn contains(self, level: EffortLevel) -> bool {
        self.0 & level.bit() != 0
    }
}

/// Capability answers and rules a caller gave for the model, which the
/// encoder follows as preflight does.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Overrides {
    structured_output: Option<CapabilitySupport>,
    thinking: Option<CapabilitySupport>,
    forced_tool_choice: Option<CapabilitySupport>,
    effort_levels: Option<EffortLevels>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    Haiku,
    Sonnet,
    Opus,
    Fable,
    Mythos,
    /// Claude Mythos Preview (`claude-mythos-preview`), unversioned: both
    /// thinking modes.
    MythosPreview,
}

/// Map a portable effort to an Anthropic effort level the model takes.
/// Returns the level and whether the mapping was lossy.
pub(crate) fn map_effort(effort: ReasoningEffort, levels: EffortLevels) -> (EffortLevel, bool) {
    let wanted = match effort {
        ReasoningEffort::Minimal | ReasoningEffort::Low => EffortLevel::Low,
        ReasoningEffort::Medium => EffortLevel::Medium,
        ReasoningEffort::High => EffortLevel::High,
        _ => EffortLevel::XHigh,
    };
    let lossy_by_name = matches!(effort, ReasoningEffort::Minimal);
    if levels.contains(wanted) {
        return (wanted, lossy_by_name);
    }
    // Without `xhigh` (the 4.6 generation), the next level up is `max`.
    if wanted == EffortLevel::XHigh && levels.contains(EffortLevel::Max) {
        return (EffortLevel::Max, true);
    }
    // Otherwise the nearest level below what was asked for, then above.
    let order = [
        EffortLevel::Low,
        EffortLevel::Medium,
        EffortLevel::High,
        EffortLevel::XHigh,
        EffortLevel::Max,
    ];
    let at = order.iter().position(|l| *l == wanted).unwrap_or(0);
    let nearest = order[..at]
        .iter()
        .rev()
        .chain(&order[at + 1..])
        .copied()
        .find(|l| levels.contains(*l))
        .unwrap_or(wanted);
    (nearest, true)
}

/// Thinking budget a reasoning effort asks for on budget-style models.
/// Without an effort no budget is sent (the model's default, thinking off).
pub(crate) fn budget_for_effort(effort: ReasoningEffort) -> u32 {
    match effort {
        ReasoningEffort::Minimal => 1024,
        ReasoningEffort::Low => 2048,
        ReasoningEffort::High => 8192,
        ReasoningEffort::XHigh => 16384,
        _ => 4096,
    }
}

fn parse(model: &str) -> Option<(Family, u32, u32)> {
    let model = model.rsplit('/').next().unwrap_or(model);
    // Bedrock ids and inference profiles: `anthropic.claude-…`,
    // `us.anthropic.claude-…-v1:0`, `global.anthropic.claude-…`.
    let model = model
        .split_once("anthropic.")
        .map_or(model, |(_, rest)| rest);
    let rest = model.strip_prefix("claude-")?;
    let tokens: Vec<&str> = rest.split(['-', '@', '[']).collect();
    let family = |t: &str| match t {
        "haiku" => Some(Family::Haiku),
        "sonnet" => Some(Family::Sonnet),
        "opus" => Some(Family::Opus),
        "fable" => Some(Family::Fable),
        "mythos" => Some(Family::Mythos),
        _ => None,
    };
    // Version tokens are short numbers; date suffixes (20250514) are not.
    let version = |t: &str| (t.len() <= 2).then(|| t.parse::<u32>().ok()).flatten();

    if tokens == ["mythos", "preview"] {
        return Some((Family::MythosPreview, 0, 0));
    }
    if let Some(f) = tokens.first().copied().and_then(family) {
        // claude-sonnet-4-5-20250929, claude-opus-4-20250514, claude-fable-5-1
        let major = tokens.get(1).copied().and_then(version)?;
        let minor = tokens.get(2).copied().and_then(version).unwrap_or(0);
        return Some((f, major, minor));
    }
    // Legacy naming: claude-3-7-sonnet-20250219, claude-3-5-haiku-latest
    let major = tokens.first().copied().and_then(version)?;
    let (minor, family_at) = match tokens.get(1).copied().and_then(version) {
        Some(minor) => (minor, 2),
        None => (0, 1),
    };
    let f = tokens.get(family_at).copied().and_then(family)?;
    Some((f, major, minor))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_model_ids() {
        assert_eq!(
            parse("claude-haiku-4-5-20251001"),
            Some((Family::Haiku, 4, 5))
        );
        assert_eq!(
            parse("claude-sonnet-4-20250514"),
            Some((Family::Sonnet, 4, 0))
        );
        assert_eq!(parse("claude-sonnet-5-5"), Some((Family::Sonnet, 5, 5)));
        assert_eq!(parse("claude-fable-5-1"), Some((Family::Fable, 5, 1)));
        assert_eq!(parse("claude-opus-5"), Some((Family::Opus, 5, 0)));
        assert_eq!(
            parse("claude-3-7-sonnet-20250219"),
            Some((Family::Sonnet, 3, 7))
        );
        assert_eq!(
            parse("claude-3-5-haiku-latest"),
            Some((Family::Haiku, 3, 5))
        );
        assert_eq!(
            parse("claude-opus-4-5@20251101"),
            Some((Family::Opus, 4, 5))
        );
        assert_eq!(parse("gpt-5"), None);
        assert_eq!(parse("claude-unknown"), None);
    }

    /// Bedrock inference profile ids parse, so a budget-style model
    /// does not get the adaptive form.
    #[test]
    fn parses_bedrock_inference_profile_ids() {
        for id in [
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "global.anthropic.claude-haiku-4-5-20251001-v1:0",
            "anthropic.claude-haiku-4-5-20251001-v1:0",
            "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0",
        ] {
            assert_eq!(parse(id), Some((Family::Haiku, 4, 5)), "{id}");
            assert_eq!(
                ModelProfile::for_model(id).thinking_style(),
                ThinkingStyle::Budget,
                "{id}"
            );
        }
    }

    /// Structured output is claimed where it is documented, and never
    /// denied on the id (ADR 0015): older generations are `Unknown`.
    #[test]
    fn structured_output_support() {
        let s = |m: &str| ModelProfile::for_model(m).structured_output();
        assert_eq!(s("claude-haiku-4-5"), CapabilitySupport::Supported);
        assert_eq!(s("claude-sonnet-5-5"), CapabilitySupport::Supported);
        assert_eq!(s("claude-opus-4-1"), CapabilitySupport::Supported);
        assert_eq!(s("claude-sonnet-4-20250514"), CapabilitySupport::Unknown);
        assert_eq!(s("claude-3-5-haiku-latest"), CapabilitySupport::Unknown);
        assert_eq!(s("claude-mythos-preview"), CapabilitySupport::Unknown);
        assert_eq!(s("my-proxy-model"), CapabilitySupport::Unknown);
    }

    #[test]
    fn thinking_support() {
        let t = |m: &str| ModelProfile::for_model(m).thinking();
        assert_eq!(t("claude-sonnet-4-5"), CapabilitySupport::Supported);
        assert_eq!(
            t("claude-3-7-sonnet-20250219"),
            CapabilitySupport::Supported
        );
        assert_eq!(t("claude-mythos-preview"), CapabilitySupport::Supported);
        assert_eq!(t("claude-3-5-haiku-latest"), CapabilitySupport::Unknown);
        assert_eq!(t("my-proxy-model"), CapabilitySupport::Unknown);
    }

    #[test]
    fn manual_thinking_interleaving() {
        // https://platform.claude.com/docs/en/build-with-claude/extended-thinking#interleaved-thinking-in-manual-mode
        let i = |m: &str| ModelProfile::for_model(m).interleaves_manual_thinking();
        assert!(i("claude-sonnet-4-20250514"));
        assert!(i("claude-opus-4-1-20250805"));
        assert!(i("claude-opus-4-5"));
        assert!(i("claude-sonnet-4-5-20250929"));
        assert!(i("claude-sonnet-4-6"));
        assert!(!i("claude-opus-4-6"));
        assert!(!i("claude-haiku-4-5-20251001"));
        assert!(!i("claude-3-7-sonnet-20250219"));
        assert!(!i("claude-opus-4-7"));
        assert!(!i("my-proxy-model"));
    }

    /// Forced tool choice is claimed for the documented generations and
    /// `Unknown` for the rest: never `Unsupported` on the id.
    #[test]
    fn forced_tool_choice_support() {
        let f = |m: &str| ModelProfile::for_model(m).forced_tool_choice();
        assert_eq!(f("claude-haiku-4-5-20251001"), CapabilitySupport::Supported);
        assert_eq!(f("claude-opus-5"), CapabilitySupport::Supported);
        assert_eq!(f("claude-fable-5"), CapabilitySupport::Supported);
        for id in [
            "claude-sonnet-5-5",
            "claude-opus-5-5",
            "claude-fable-5-1",
            "claude-mythos-5-1",
            "my-proxy-model",
        ] {
            assert_eq!(f(id), CapabilitySupport::Unknown, "{id}");
        }
    }

    #[test]
    fn thinking_styles() {
        let p = ModelProfile::for_model;
        assert_eq!(
            p("claude-haiku-4-5").thinking_style(),
            ThinkingStyle::Budget
        );
        assert_eq!(
            p("claude-sonnet-4-6").thinking_style(),
            ThinkingStyle::Adaptive
        );
        assert_eq!(
            p("claude-sonnet-5-5").disable_thinking(),
            DisableThinking::BetweenTools
        );
        // Models that cannot turn thinking off get the adaptive off value
        // too, and the API's refusal is reported.
        for id in [
            "claude-opus-5-5",
            "claude-fable-5-1",
            "claude-mythos-preview",
            "claude-opus-4-8",
            "my-proxy-model",
        ] {
            assert_eq!(p(id).disable_thinking(), DisableThinking::Disabled, "{id}");
        }
        assert_eq!(
            p("claude-haiku-4-5").disable_thinking(),
            DisableThinking::Omit
        );
        assert_eq!(p("claude-haiku-4-5").effort_levels(), None);
        assert!(
            !p("claude-sonnet-4-6")
                .effort_levels()
                .unwrap()
                .contains(EffortLevel::XHigh)
        );
    }

    /// Only the enumerated legacy generations take the budget form; every
    /// other recognized id, a later version of a known family included, is
    /// adaptive.
    #[test]
    fn budget_thinking_is_for_the_legacy_generations_only() {
        for id in [
            "claude-3-7-sonnet-20250219",
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-opus-4-1-20250805",
            "claude-opus-4-5-20251101",
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
            "claude-3-5-haiku-latest",
        ] {
            let profile = ModelProfile::for_model(id);
            assert_eq!(profile.thinking_style(), ThinkingStyle::Budget, "{id}");
            assert_eq!(profile.effort_levels(), None, "{id}");
            assert_eq!(profile.disable_thinking(), DisableThinking::Omit, "{id}");
        }
        for id in [
            "claude-sonnet-4-6",
            "claude-opus-4-6",
            "claude-opus-4-8",
            "claude-sonnet-5",
            "claude-opus-5-5",
            "claude-haiku-5-5",
            "claude-fable-5-1",
            "claude-mythos-5-1",
            "claude-mythos-preview",
        ] {
            assert_eq!(
                ModelProfile::for_model(id).thinking_style(),
                ThinkingStyle::Adaptive,
                "{id}"
            );
        }
    }

    /// Claude Haiku 5.5 takes adaptive thinking only, with all five effort
    /// levels, and thinks when the request leaves `thinking` out
    /// (<https://platform.claude.com/docs/en/build-with-claude/effort>).
    #[test]
    fn haiku_5_5_is_adaptive_and_thinks_by_default() {
        for id in ["claude-haiku-5-5", "anthropic.claude-haiku-5-5"] {
            let profile = ModelProfile::for_model(id);
            assert_eq!(profile.thinking_style(), ThinkingStyle::Adaptive, "{id}");
            assert_eq!(profile.effort_levels(), Some(EffortLevels::FULL), "{id}");
            assert!(profile.thinks_by_default(), "{id}");
            assert!(profile.supports_thinking_display(), "{id}");
            assert_eq!(profile.disable_thinking(), DisableThinking::Disabled);
            assert_eq!(
                map_effort(ReasoningEffort::Low, profile.effort_levels().unwrap()),
                (EffortLevel::Low, false)
            );
        }
        assert!(!ModelProfile::for_model("claude-haiku-4-5").thinks_by_default());
    }

    /// A version of a known family this release has not seen gets the
    /// newest wire format (ADR 0015), not the legacy one.
    #[test]
    fn unknown_future_versions_get_the_newest_format() {
        for id in [
            "claude-haiku-6",
            "claude-haiku-5-7",
            "claude-sonnet-6",
            "claude-opus-6-1",
            "claude-fable-6",
            "claude-mythos-6",
        ] {
            let profile = ModelProfile::for_model(id);
            assert!(profile.is_known(), "{id}");
            assert_eq!(profile.thinking_style(), ThinkingStyle::Adaptive, "{id}");
            assert_eq!(profile.effort_levels(), Some(EffortLevels::FULL), "{id}");
            assert!(profile.thinks_by_default(), "{id}");
            assert_eq!(profile.thinking(), CapabilitySupport::Supported, "{id}");
        }
    }

    #[test]
    fn effort_mapping() {
        let full = ModelProfile::for_model("claude-opus-4-8")
            .effort_levels()
            .unwrap();
        assert_eq!(
            map_effort(ReasoningEffort::High, full),
            (EffortLevel::High, false)
        );
        assert_eq!(
            map_effort(ReasoningEffort::XHigh, full),
            (EffortLevel::XHigh, false)
        );
        assert_eq!(
            map_effort(ReasoningEffort::Minimal, full),
            (EffortLevel::Low, true)
        );
        let no_xhigh = ModelProfile::for_model("claude-opus-4-6")
            .effort_levels()
            .unwrap();
        assert_eq!(
            map_effort(ReasoningEffort::XHigh, no_xhigh),
            (EffortLevel::Max, true)
        );
        let up_to_high = EffortLevels::of(&[EffortLevel::Low, EffortLevel::High]);
        assert_eq!(
            map_effort(ReasoningEffort::XHigh, up_to_high),
            (EffortLevel::High, true)
        );
        assert_eq!(
            map_effort(ReasoningEffort::Medium, up_to_high),
            (EffortLevel::Low, true)
        );
    }

    /// Ids that do not parse are not assumed to be any generation: their
    /// model-dependent answers are `Unknown`, and they get the newest wire
    /// format with every effort level.
    #[test]
    fn unparseable_ids_are_unknown() {
        for id in [
            "claude-typo",
            "claude-sonnet",
            "claude-sonnet-x",
            "fast-alias",
        ] {
            let profile = ModelProfile::for_model(id);
            assert!(!profile.is_known(), "{id}");
            assert_eq!(profile.thinking(), CapabilitySupport::Unknown, "{id}");
            assert_eq!(profile.structured_output(), CapabilitySupport::Unknown);
            assert_eq!(profile.forced_tool_choice(), CapabilitySupport::Unknown);
            assert_eq!(profile.thinking_style(), ThinkingStyle::Adaptive);
            assert_eq!(profile.effort_levels(), Some(EffortLevels::FULL));
        }
        // The documented id forms all parse to their family.
        for (id, family) in [
            ("claude-opus-4-1-20250805", (Family::Opus, 4, 1)),
            (
                "anthropic.claude-haiku-4-5-20251001-v1:0",
                (Family::Haiku, 4, 5),
            ),
            ("claude-sonnet-4-5@20250929", (Family::Sonnet, 4, 5)),
            ("models/claude-mythos-5-1", (Family::Mythos, 5, 1)),
        ] {
            assert_eq!(parse(id), Some(family), "{id}");
        }
    }

    #[test]
    fn overrides_reach_the_profile() {
        let profile = ModelProfile::for_model("claude-3-5-haiku-latest").with_overrides(|c| {
            matches!(
                c,
                ChatCapability::StructuredOutput | ChatCapability::ReasoningConfig
            )
            .then_some(CapabilitySupport::Supported)
        });
        assert_eq!(profile.structured_output(), CapabilitySupport::Supported);
        assert_eq!(profile.thinking(), CapabilitySupport::Supported);
        let profile = ModelProfile::for_model("claude-opus-5-5").with_overrides(|c| {
            (c == ChatCapability::ForcedToolChoice).then_some(CapabilitySupport::Supported)
        });
        assert_eq!(profile.forced_tool_choice(), CapabilitySupport::Supported);
        let profile = ModelProfile::for_model("claude-opus-4-1").with_overrides(|c| {
            (c == ChatCapability::ForcedToolChoice).then_some(CapabilitySupport::Unsupported)
        });
        assert_eq!(profile.forced_tool_choice(), CapabilitySupport::Unsupported);
    }

    /// Model rules replace the effort levels the id implies, on known and
    /// unknown ids alike; unset fields keep the adapter's choice.
    #[test]
    fn model_rules_reach_the_effort_levels() {
        let with_xhigh = ModelRules::new().effort_levels([
            EffortLevel::Low,
            EffortLevel::Medium,
            EffortLevel::High,
            EffortLevel::XHigh,
            EffortLevel::Max,
        ]);
        let profile = ModelProfile::for_model("claude-opus-4-6").with_rules(&with_xhigh);
        assert_eq!(
            map_effort(ReasoningEffort::XHigh, profile.effort_levels().unwrap()),
            (EffortLevel::XHigh, false)
        );
        let narrow = ModelRules::new().effort_levels([EffortLevel::Low, EffortLevel::High]);
        let profile = ModelProfile::for_model("my-proxy-model").with_rules(&narrow);
        assert_eq!(
            profile.effort_levels(),
            Some(EffortLevels::of(&[EffortLevel::Low, EffortLevel::High]))
        );
        let unset = ModelProfile::for_model("claude-opus-4-6").with_rules(&ModelRules::new());
        assert_eq!(unset.effort_levels(), Some(EffortLevels::NO_XHIGH));
    }

    /// The per-model table of
    /// <https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting>:
    /// Claude Mythos Preview takes adaptive and extended thinking.
    #[test]
    fn mythos_preview_takes_both_thinking_modes() {
        let p = ModelProfile::for_model("claude-mythos-preview");
        assert!(p.is_known());
        assert_eq!(p.thinking(), CapabilitySupport::Supported);
        assert_eq!(p.thinking_style(), ThinkingStyle::Adaptive);
        assert_eq!(p.forced_tool_choice(), CapabilitySupport::Supported);
        assert_eq!(p.structured_output(), CapabilitySupport::Unknown);
    }
}
