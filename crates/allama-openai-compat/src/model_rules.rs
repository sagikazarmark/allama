//! Per-model request rules for the Chat Completions encoder.

use std::borrow::Cow;

use allama::adapter::Degradations;
use allama::{
    CapabilitySupport, ExtraMap, ReasoningEffort, Result, Subject, UnsupportedPolicy, Warning,
    WarningKind,
};

use crate::{SamplingSupport, SchemaConstraint, ToolCallIdFormat};

/// What the Chat Completions encoder needs to know about one model: the
/// counterpart of [`ResponsesModelRules`](crate::ResponsesModelRules).
///
/// Dialect facts (what the *endpoint* accepts) live in
/// [`CompatQuirks`](crate::CompatQuirks); these are facts about the *model*
/// behind it, which the endpoint would otherwise reject with an HTTP 400.
/// The default, [`ChatModelRules::new`], assumes nothing: the caller's
/// sampling settings and tools are sent as requested.
///
/// Whether reasoning is on is resolved from the request's reasoning effort,
/// or, when it sets none, from the model's
/// [`default_effort`](Self::default_effort). Supply rules per model id with
/// [`ProviderBuilder::model_rules`](crate::ProviderBuilder::model_rules):
///
/// ```
/// use allama::ReasoningEffort;
/// use allama_openai_compat::{ChatModelRules, Provider, SamplingSupport};
///
/// let provider = Provider::builder()
///     .base_url("http://localhost:8000/v1")
///     .model_rules(|model: &str| {
///         if model.starts_with("my-reasoner") {
///             ChatModelRules::new()
///                 .sampling(SamplingSupport::WithoutReasoning)
///                 .default_effort(ReasoningEffort::Medium)
///         } else {
///             ChatModelRules::new()
///         }
///     })
///     .build()?;
/// # let _ = provider; Ok::<(), allama::Error>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ChatModelRules {
    /// Reasoning effort levels the model accepts. A requested level outside
    /// the list is sent as the nearest listed level, with an
    /// `Approximated` warning. `None` sends the requested level as it is.
    pub effort_levels: Option<Cow<'static, [ReasoningEffort]>>,
    /// The effort the model reasons at when the request sets none, when
    /// known. `None` means unknown: a request without an effort is treated
    /// as not reasoning.
    pub default_effort: Option<ReasoningEffort>,
    /// When the model accepts `temperature`, `top_p`, penalties, `logprobs`
    /// and `top_logprobs`. Those it would reject are dropped as unsupported
    /// hints (an error under `UnsupportedPolicy::Error`).
    pub sampling: SamplingSupport,
    /// When the model takes tool definitions on this endpoint. A request
    /// with tools the model would reject is refused before sending.
    pub tool_calls: ToolCallSupport,
    /// How this model constrains a `json_schema` response format, when it
    /// differs from the endpoint's
    /// [`response_format_constraint`](crate::CompatQuirks::response_format_constraint):
    /// an endpoint whose strict mode covers only some of its models. `None`
    /// keeps the endpoint's.
    pub response_format_constraint: Option<SchemaConstraint>,
    /// The model's answer for
    /// [`StructuredOutput`](allama::ChatCapability::StructuredOutput), when
    /// it differs from the endpoint's: a model the endpoint serves
    /// elsewhere, where a JSON Schema response format is not honored. It takes
    /// precedence over the provider's declared capabilities, and a
    /// capability resolver takes precedence over it. `None` keeps the
    /// provider's answer.
    pub structured_output: Option<CapabilitySupport>,
    /// The tool call ids this model takes, when they differ from the
    /// endpoint's [`tool_call_id_format`](crate::CompatQuirks::tool_call_id_format):
    /// an endpoint that runs Mistral models through `mistral_common` next
    /// to others (Workers AI) takes only nine letters and digits for those.
    /// `None` keeps the endpoint's.
    pub tool_call_id_format: Option<ToolCallIdFormat>,
}

impl ChatModelRules {
    /// Rules that assume nothing: no default effort, sampling parameters and
    /// tools accepted.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the reasoning effort levels the model accepts.
    #[must_use]
    pub fn effort_levels(mut self, levels: impl Into<Cow<'static, [ReasoningEffort]>>) -> Self {
        self.effort_levels = Some(levels.into());
        self
    }

    /// Set the effort the model reasons at when the request sets none.
    #[must_use]
    pub fn default_effort(mut self, effort: ReasoningEffort) -> Self {
        self.default_effort = Some(effort);
        self
    }

    /// Set when the model accepts sampling parameters.
    #[must_use]
    pub fn sampling(mut self, sampling: SamplingSupport) -> Self {
        self.sampling = sampling;
        self
    }

    /// Set when the model takes tool definitions.
    #[must_use]
    pub fn tool_calls(mut self, tool_calls: ToolCallSupport) -> Self {
        self.tool_calls = tool_calls;
        self
    }

    /// Set how the model constrains a `json_schema` response format.
    #[must_use]
    pub fn response_format_constraint(mut self, constraint: SchemaConstraint) -> Self {
        self.response_format_constraint = Some(constraint);
        self
    }

    /// Set the model's answer for
    /// [`StructuredOutput`](allama::ChatCapability::StructuredOutput).
    #[must_use]
    pub fn structured_output(mut self, support: CapabilitySupport) -> Self {
        self.structured_output = Some(support);
        self
    }

    /// Set the tool call ids the model takes.
    #[must_use]
    pub fn tool_call_id_format(mut self, format: ToolCallIdFormat) -> Self {
        self.tool_call_id_format = Some(format);
        self
    }

    /// Whether the model reasons for a request that sends `effort` (the raw
    /// `reasoning_effort` value, if any): an effort other than `none`, or,
    /// without one, a default effort other than `none`.
    #[must_use]
    pub fn reasons(&self, effort: Option<&str>) -> bool {
        effort
            .or(self.default_effort.map(ReasoningEffort::as_str))
            .is_some_and(|effort| effort != "none")
    }

    /// Whether sampling parameters are rejected for a request that sends
    /// `effort`.
    #[must_use]
    pub fn rejects_sampling(&self, effort: Option<&str>) -> bool {
        match self.sampling {
            SamplingSupport::Rejected => true,
            SamplingSupport::WithoutReasoning => self.reasons(effort),
            _ => false,
        }
    }

    /// Whether tool definitions are rejected for a request that sends
    /// `effort`.
    #[must_use]
    pub fn rejects_tools(&self, effort: Option<&str>) -> bool {
        match self.tool_calls {
            ToolCallSupport::WithoutReasoning => self.reasons(effort),
            _ => false,
        }
    }
}

/// When a model takes tool definitions on Chat Completions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolCallSupport {
    /// Always (or unknown: pass through).
    #[default]
    Accepted,
    /// Only while reasoning is off (effort `none`), such as OpenAI's GPT-6
    /// Sol and Luna on Chat Completions.
    WithoutReasoning,
}

/// Drop the log-probability fields of `extra` a reasoning model rejects
/// (`logprobs`, `top_logprobs`), as unsupported hints.
pub(crate) fn drop_logprobs(
    extra: &mut ExtraMap,
    keys: &[&'static str],
    model: &str,
    d: &mut Degradations,
) -> Result<()> {
    for &key in keys {
        if !extra.contains_key(key) {
            continue;
        }
        let subject = Subject::Other {
            name: format!("`{key}`"),
        };
        let message = format!(
            "`{model}` does not accept `{key}` with reasoning enabled; `{key}` was not sent"
        );
        if d.policy() == UnsupportedPolicy::Error {
            d.hint(subject, message)?;
        } else {
            let code = match key {
                "logprobs" => "unsupported.logprobs",
                _ => "unsupported.top_logprobs",
            };
            d.warn(Warning::new(WarningKind::Unsupported, subject, message).with_code(code));
        }
        extra.remove(key);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_is_resolved_from_the_request_or_the_default() {
        let rules = ChatModelRules::new();
        assert!(!rules.reasons(None));
        assert!(rules.reasons(Some("low")));
        assert!(!rules.rejects_sampling(Some("high")));

        let rules = ChatModelRules::new()
            .sampling(SamplingSupport::WithoutReasoning)
            .tool_calls(ToolCallSupport::WithoutReasoning)
            .default_effort(ReasoningEffort::Medium);
        assert!(rules.rejects_sampling(None));
        assert!(!rules.rejects_sampling(Some("none")));
        assert!(rules.rejects_tools(None));
        assert!(!rules.rejects_tools(Some("none")));

        let off_by_default = rules.clone().default_effort(ReasoningEffort::None);
        assert!(!off_by_default.rejects_sampling(None));
        assert!(off_by_default.rejects_sampling(Some("low")));

        let rejected = ChatModelRules::new().sampling(SamplingSupport::Rejected);
        assert!(rejected.rejects_sampling(Some("none")));
    }
}
