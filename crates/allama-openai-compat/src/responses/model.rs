//! Per-model request rules for the Responses encoder.

use std::borrow::Cow;

use allama::{CapabilitySupport, ReasoningEffort};

/// What the Responses encoder needs to know about one model.
///
/// Dialect facts (what the *endpoint* accepts) live in
/// [`ResponsesQuirks`](crate::ResponsesQuirks); these are facts about the
/// *model* behind it, which the endpoint would otherwise reject with an
/// HTTP 400. The default, [`ResponsesModelRules::new`], assumes nothing:
/// the caller's reasoning and sampling settings are sent as requested.
///
/// Supply rules per model id with
/// [`ResponsesProviderBuilder::model_rules`](crate::ResponsesProviderBuilder::model_rules):
///
/// ```
/// use allama::{CapabilitySupport, ReasoningEffort};
/// use allama_openai_compat::{ResponsesModelRules, ResponsesProvider};
///
/// let provider = ResponsesProvider::builder()
///     .base_url("http://localhost:8000/v1")
///     .model_rules(|model: &str| {
///         if model.starts_with("gpt-oss") {
///             ResponsesModelRules::new()
///                 .reasoning(CapabilitySupport::Supported)
///                 .effort_levels(&[ReasoningEffort::Low, ReasoningEffort::Medium, ReasoningEffort::High])
///         } else {
///             ResponsesModelRules::new()
///         }
///     })
///     .build()?;
/// # let _ = provider; Ok::<(), allama::Error>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ResponsesModelRules {
    /// Whether the model reasons. When `Unsupported`, reasoning settings are
    /// reported as unsupported hints instead of being sent (an effort of
    /// `None` needs no report). `Supported` and `Unknown` send them.
    pub reasoning: CapabilitySupport,
    /// Reasoning effort levels the model accepts. A requested level outside
    /// the list is sent as the nearest listed level, with an
    /// `Approximated` warning. `None` sends every level unchanged.
    pub effort_levels: Option<Cow<'static, [ReasoningEffort]>>,
    /// When the model accepts `temperature`, `top_p`, penalties and
    /// `top_logprobs`.
    pub sampling: SamplingSupport,
    /// The effort the model reasons at when the request sets none, when
    /// known. With [`SamplingSupport::WithoutReasoning`], a request without
    /// an effort is treated as reasoning when this is an effort other than
    /// `none`.
    pub default_effort: Option<ReasoningEffort>,
}

impl ResponsesModelRules {
    /// Rules that assume nothing: reasoning `Unknown`, every effort level
    /// passed through, sampling parameters accepted.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether the model reasons.
    #[must_use]
    pub fn reasoning(mut self, reasoning: CapabilitySupport) -> Self {
        self.reasoning = reasoning;
        self
    }

    /// Set the reasoning effort levels the model accepts.
    #[must_use]
    pub fn effort_levels(mut self, levels: impl Into<Cow<'static, [ReasoningEffort]>>) -> Self {
        self.effort_levels = Some(levels.into());
        self
    }

    /// Set when the model accepts sampling parameters.
    #[must_use]
    pub fn sampling(mut self, sampling: SamplingSupport) -> Self {
        self.sampling = sampling;
        self
    }

    /// Set the effort the model reasons at when the request sets none.
    #[must_use]
    pub fn default_effort(mut self, effort: ReasoningEffort) -> Self {
        self.default_effort = Some(effort);
        self
    }

    /// Whether sampling parameters are rejected when the request sends
    /// `effort` (the raw `reasoning.effort` value, if any): never, always,
    /// or when the model reasons, at `effort` or, without one, at its
    /// [`default_effort`](Self::default_effort).
    #[must_use]
    pub fn rejects_sampling(&self, effort: Option<&str>) -> bool {
        match self.sampling {
            SamplingSupport::Accepted => false,
            SamplingSupport::Rejected => true,
            SamplingSupport::WithoutReasoning => effort
                .or(self.default_effort.map(ReasoningEffort::as_str))
                .is_some_and(|e| e != "none"),
        }
    }
}

/// When a model accepts sampling parameters (`temperature`, `top_p`,
/// `frequency_penalty`, `presence_penalty`).
///
/// Parameters the model would reject are dropped with an `Unsupported`
/// warning, or fail under `UnsupportedPolicy::Error`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum SamplingSupport {
    /// Always (or unknown: pass through).
    #[default]
    Accepted,
    /// Never: the model always reasons and rejects them (OpenAI GPT-5,
    /// o-series).
    Rejected,
    /// Only while reasoning is off: rejected when a reasoning effort other
    /// than `none` is sent (OpenAI GPT-5.1 and newer).
    WithoutReasoning,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_rules() {
        let rules = ResponsesModelRules::new();
        assert_eq!(rules.reasoning, CapabilitySupport::Unknown);
        assert!(!rules.rejects_sampling(Some("high")));
        let rejected = ResponsesModelRules::new().sampling(SamplingSupport::Rejected);
        assert!(rejected.rejects_sampling(None));
        let without = ResponsesModelRules::new().sampling(SamplingSupport::WithoutReasoning);
        assert!(!without.rejects_sampling(None));
        assert!(!without.rejects_sampling(Some("none")));
        assert!(without.rejects_sampling(Some("low")));
        let medium = without.default_effort(ReasoningEffort::Medium);
        assert!(medium.rejects_sampling(None));
        assert!(!medium.rejects_sampling(Some("none")));
    }
}
