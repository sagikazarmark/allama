use serde::{Deserialize, Serialize};

use crate::{CacheBreakpoint, ProviderData};

/// The definition of a tool the model may call: its name, description, and
/// argument schema. It declares the tool; running it is up to the caller.
///
/// Keep the tool's name in one constant, used both in the definition and
/// where you dispatch the model's calls, so the two cannot drift:
///
/// ```
/// use allama::{ToolCall, ToolDefinition};
/// use serde_json::json;
///
/// const LOOKUP_ORDER: &str = "lookup_order";
///
/// let tool = ToolDefinition::new(
///     LOOKUP_ORDER,
///     json!({"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}),
/// )
/// .description("Look up an order by id");
///
/// fn run(call: &ToolCall) -> String {
///     match call.name.as_str() {
///         LOOKUP_ORDER => format!("order {}: shipped", call.arguments),
///         other => format!("unknown tool `{other}`"),
///     }
/// }
///
/// let call = ToolCall::new("call_1", LOOKUP_ORDER, r#"{"id":"A1"}"#);
/// assert!(run(&call).starts_with("order"));
/// # let _ = tool;
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolDefinition {
    /// Tool name.
    pub name: String,
    /// What the tool does, shown to the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema of the arguments (an object schema). Adapters rewrite it
    /// into each provider's schema dialect.
    pub parameters: serde_json::Value,
    /// Request strict schema adherence for the arguments, where supported.
    ///
    /// `None` leaves it to the provider. [`new`](Self::new) and
    /// [`from_schema`][Self::from_schema] leave it unset; set it with
    /// [`strict`](Self::strict).
    ///
    #[cfg_attr(
        not(feature = "schema"),
        doc = "[Self::from_schema]: crate#cargo-features"
    )]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    /// Marks the end of a cacheable prefix (after this tool definition).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheBreakpoint>,
    /// Provider-specific data.
    #[serde(default, skip_serializing_if = "ProviderData::is_empty")]
    pub provider_data: ProviderData,
}

impl ToolDefinition {
    /// Create a tool definition from a JSON Schema for its arguments.
    ///
    /// [`strict`](Self::strict) is left unset, which leaves it to the
    /// provider (usually not strict). Add `.strict(true)` to ask for
    /// arguments constrained to the schema, and parse them with
    /// [`ToolCall::parse_arguments_with`](crate::ToolCall::parse_arguments_with).
    #[must_use]
    pub fn new(name: impl Into<String>, parameters: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            description: None,
            parameters,
            strict: None,
            cache: None,
            provider_data: ProviderData::new(),
        }
    }

    /// Create a tool definition whose argument schema is generated from `T`.
    ///
    /// `T` implements [`crate::schemars::JsonSchema`]: derive it through the
    /// [`allama::schemars`](crate::schemars) re-export to stay on the same
    /// `schemars` version.
    ///
    /// As with [`new`](Self::new), [`strict`](Self::strict) is left unset.
    /// It is not strict by default because a strict tool on an endpoint
    /// without strict tools is a warning on every call (or a refusal under
    /// [`UnsupportedPolicy::Error`](crate::UnsupportedPolicy::Error)), and
    /// OpenAI rewrites strict schemas (every property required, optional
    /// ones nullable). Ask for it with `.strict(true)` and parse the
    /// arguments with
    /// [`ToolCall::parse_arguments_with`](crate::ToolCall::parse_arguments_with).
    #[cfg(feature = "schema")]
    #[must_use]
    pub fn from_schema<T: schemars::JsonSchema>(name: impl Into<String>) -> Self {
        Self::new(name, crate::schema::schema_for::<T>())
    }

    /// Set the description.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Request strict schema adherence: `true` asks the provider to
    /// constrain the arguments to the schema, `false` says constraining is
    /// not required.
    ///
    /// Where strict tools are not supported, a strict tool is sent without
    /// strict and reported as an unsupported hint (see
    /// [`UnsupportedPolicy`](crate::UnsupportedPolicy)).
    #[must_use]
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }

    /// Mark this tool as the end of a cacheable prefix.
    #[must_use]
    pub fn with_cache(mut self, breakpoint: CacheBreakpoint) -> Self {
        self.cache = Some(breakpoint);
        self
    }
}

/// How the model should choose tools.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolChoice {
    /// The model decides whether to call tools.
    Auto,
    /// The model must not call tools; vendor APIs call this `none`.
    ///
    /// It is `Disabled`, not `None`, because the field it sets is an
    /// `Option<ToolChoice>`, whose `None` leaves the choice to the provider.
    #[doc(alias = "none")]
    Disabled,
    /// The model must call at least one tool.
    ///
    /// Reset it (to `Auto` or `None`) before the follow-up request that
    /// carries the tool results, or the model is forced to call a tool
    /// again instead of answering.
    Required,
    /// The model must call the named tool.
    Specific {
        /// Tool name.
        name: String,
    },
}

impl ToolChoice {
    /// The [kind name](Self::kind_name) of every kind of choice, in
    /// declaration order (for audits such as the conformance honesty
    /// contract). Grows when kinds are added.
    pub const KIND_NAMES: &'static [&'static str] = &["auto", "disabled", "required", "specific"];

    /// The stable snake_case name of the choice's kind, as serialized.
    #[must_use]
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Disabled => "disabled",
            Self::Required => "required",
            Self::Specific { .. } => "specific",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn every_kind_is_listed() {
        let choices = [
            ToolChoice::Auto,
            ToolChoice::Disabled,
            ToolChoice::Required,
            ToolChoice::Specific { name: "t".into() },
        ];
        let names: Vec<_> = choices.iter().map(ToolChoice::kind_name).collect();
        assert_eq!(names, ToolChoice::KIND_NAMES);
        for choice in choices {
            let value = serde_json::to_value(&choice).unwrap();
            let serialized = value
                .as_str()
                .map(str::to_owned)
                .or_else(|| value.as_object()?.keys().next().cloned())
                .unwrap();
            assert_eq!(serialized, choice.kind_name());
        }
    }

    #[test]
    fn tool_definition_round_trip_and_minimal_shape() {
        let tool = ToolDefinition::new("search", json!({"type": "object"}))
            .description("Search")
            .strict(true);
        let value = serde_json::to_value(&tool).unwrap();
        assert_eq!(value["strict"], true);
        assert_eq!(
            serde_json::from_value::<ToolDefinition>(value).unwrap(),
            tool
        );

        let bare = serde_json::to_value(ToolDefinition::new("bare", json!({}))).unwrap();
        assert_eq!(bare, json!({"name": "bare", "parameters": {}}));
    }

    #[test]
    fn tool_choice_round_trip() {
        for choice in [
            ToolChoice::Auto,
            ToolChoice::Disabled,
            ToolChoice::Required,
            ToolChoice::Specific { name: "x".into() },
        ] {
            let value = serde_json::to_value(&choice).unwrap();
            assert_eq!(serde_json::from_value::<ToolChoice>(value).unwrap(), choice);
        }
    }
}
