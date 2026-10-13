//! What an extraction can know before it sends ([`ExtractionPlan`]), the
//! enforcement floor, and how enforcement is reported in terms of the
//! caller's type.

use std::borrow::Cow;

use schemars::JsonSchema;
use serde_json::Value;

use super::{
    AutoFallback, EXTRACTION_TOOL, ExtractError, ExtractFailure, ExtractionMetadata,
    ExtractionMode, Extractor, SchemaEnforcement, counted_field, derive_request, is_about,
    requested_mode, schema_name, select_mode,
};
use crate::{
    ChatProvider, ChatRequest, Error, Message, RequestField, Result, ValidationCoverage, Warning,
    WarningKind,
};

/// What an extraction of `T` would do, found without sending anything
/// ([`Extractor::plan`]).
///
/// ```
/// # #[cfg(feature = "mock")]
/// # fn example() -> allama::Result<()> {
/// use allama::ChatRequest;
/// use allama::extract::{ExtractionMode, Extractor, SchemaEnforcement};
/// use allama::mock::MockProvider;
/// use allama::schemars::{self, JsonSchema};
///
/// #[derive(serde::Deserialize, JsonSchema)]
/// struct Person {
///     name: String,
/// }
///
/// let provider = MockProvider::with_text("{}");
/// let request = ChatRequest::new("m").user("Ada").with_option(ExtractionMode::Native);
/// let plan = Extractor::new().plan::<Person>(&provider, &request)?;
/// assert_eq!(plan.mode, ExtractionMode::Native);
/// assert_eq!(plan.enforcement, SchemaEnforcement::Exact);
/// # Ok(()) }
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ExtractionPlan {
    /// The mode the extraction would use: `ForcedTool` after an
    /// [automatic fallback](super#automatic-fallback-to-forced-tool-mode).
    pub mode: ExtractionMode,
    /// The enforcement the first pass would report, from the provider's
    /// validation. With [`ValidationCoverage::Capabilities`] the adapter's
    /// encoding did not run, so this is an upper bound: an adapter reports
    /// most degradations while encoding.
    pub enforcement: SchemaEnforcement,
    /// The warnings that decide [`enforcement`](Self::enforcement), in terms
    /// of the caller's type, as [`ExtractionMetadata::degradations`] would
    /// hold them.
    pub degradations: Vec<Warning>,
    /// The refusal that would switch an `Auto` extraction to forced-tool
    /// mode, when it would.
    pub auto_fallback: Option<AutoFallback>,
    /// How much of the call the provider's validation covered.
    pub coverage: ValidationCoverage,
}

impl Extractor {
    /// Plan an extraction of `T` from `request` without sending it: the mode
    /// it would use, the enforcement its first pass would report, and how
    /// much of that the provider could check.
    ///
    /// It runs the provider's validation
    /// ([`validate_chat_request`](ChatProvider::validate_chat_request)) on a
    /// schema-only copy of the extraction request: the same model, options,
    /// tools, response format and policy, with one placeholder message in
    /// place of the conversation, so the plan does not depend on what the
    /// conversation holds. Every first-party adapter validates with a dry
    /// run of the call ([`ValidationCoverage::Encoding`]).
    ///
    /// # Errors
    ///
    /// What the call would fail with before sending: a refusal under
    /// [`UnsupportedPolicy::Error`](crate::UnsupportedPolicy::Error), in
    /// terms of `T`, or an [`Error::Extract`] when no mode can be used.
    pub fn plan<T: JsonSchema>(
        &self,
        provider: &(impl ChatProvider + ?Sized),
        request: &ChatRequest,
    ) -> Result<ExtractionPlan> {
        Target::new::<T>(provider.provider_name(), &request.model).plan(provider, request, false)
    }

    /// Check the floor before sending, when the provider's validation is a
    /// dry run of the call.
    pub(super) fn check_floor_before_sending<P: ChatProvider + ?Sized>(
        &self,
        provider: &P,
        request: &ChatRequest,
        target: &Target,
        stream: bool,
    ) -> Result<()> {
        let Some(floor) = self.min_enforcement else {
            return Ok(());
        };
        let plan = target.plan(provider, request, stream)?;
        if plan.coverage != ValidationCoverage::Encoding || plan.enforcement.satisfies(floor) {
            return Ok(());
        }
        let mut metadata = ExtractionMetadata::new(plan.mode);
        metadata.auto_fallback = plan.auto_fallback;
        Err(ExtractFailure::new(
            ExtractError::EnforcementBelow {
                required: floor,
                achieved: plan.enforcement,
                reason: reason(&plan.degradations),
            },
            metadata,
        )
        .into())
    }

    /// The failure for a pass whose enforcement is below the floor.
    pub(super) fn below_floor(&self, metadata: &ExtractionMetadata) -> Option<ExtractFailure> {
        let floor = self.min_enforcement?;
        (!metadata.enforcement.satisfies(floor)).then(|| {
            ExtractFailure::new(
                ExtractError::EnforcementBelow {
                    required: floor,
                    achieved: metadata.enforcement,
                    reason: reason(&metadata.degradations),
                },
                metadata.clone(),
            )
        })
    }

    /// Whether a pass with `metadata`'s enforcement is checked for
    /// undeclared keys: every pass that is not `Exact`, unless allowed.
    pub(super) fn rejects_undeclared(&self, metadata: &ExtractionMetadata) -> bool {
        !self.allow_undeclared_keys && metadata.enforcement != SchemaEnforcement::Exact
    }
}

/// Why an enforcement is what it is: the counted degradations' messages.
fn reason(degradations: &[Warning]) -> String {
    if degradations.is_empty() {
        return "no pass was made".to_owned();
    }
    degradations
        .iter()
        .map(|w| w.message.as_str())
        .collect::<Vec<_>>()
        .join("; ")
}

/// What the degradations say the provider lacks, for the enforcement
/// warning: the limitation of each message (before its first `"; "`), so
/// the warning keeps one shape (`<limitation>; <outcome>`).
fn limitations(degradations: &[Warning]) -> String {
    if degradations.is_empty() {
        return "no pass was made".to_owned();
    }
    degradations
        .iter()
        .map(|w| {
            w.message
                .split_once("; ")
                .map_or(w.message.as_str(), |(l, _)| l)
        })
        .collect::<Vec<_>>()
        .join(", and ")
}

/// The type an extraction targets: its schema and how messages name it.
#[derive(Debug, Clone)]
pub(super) struct Target {
    /// The provider name.
    pub(super) provider: String,
    /// The model.
    pub(super) model: String,
    /// How messages name the type: the schema's `title` (the Rust type
    /// name, for a derived schema).
    pub(super) label: String,
    /// The schema name sent with the response format.
    pub(super) name: String,
    /// The schema of `T`.
    pub(super) schema: Value,
}

impl Target {
    pub(super) fn new<T: JsonSchema>(provider: &str, model: &str) -> Self {
        let schema = crate::schema::schema_for::<T>();
        let label = schema
            .get("title")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| short_type_name(std::any::type_name::<T>()));
        Self {
            provider: provider.to_owned(),
            model: model.to_owned(),
            label,
            name: schema_name::<T>(&schema),
            schema,
        }
    }

    /// The request a pass in `metadata`'s mode sends.
    pub(super) fn derive(
        &self,
        request: &ChatRequest,
        metadata: &ExtractionMetadata,
    ) -> Result<ChatRequest> {
        derive_request(
            request,
            metadata.mode,
            &self.provider,
            &self.name,
            self.schema.clone(),
        )
        .map_err(|error| ExtractFailure::new(error, metadata.clone()).into())
    }

    /// The error a call that failed returns: a refusal before sending in
    /// terms of the caller's type, and in `Auto` mode after a fallback, the
    /// native refusal as the primary cause. Any other error is returned as
    /// itself.
    pub(super) fn pre_send_error(&self, error: Error, metadata: &ExtractionMetadata) -> Error {
        if error.unsupported_stage() != Some(crate::UnsupportedStage::BeforeSending) {
            return metadata.attach_to(error);
        }
        match &metadata.auto_fallback {
            Some(fallback) if metadata.passes == 0 => {
                let forced = if error.message_origin() == crate::MessageOrigin::Authored {
                    format!(
                        "was refused too ({})",
                        refusal_in_caller_terms(&error.message(), &self.label)
                    )
                } else {
                    "was refused too".to_owned()
                };
                metadata.attach_to(self.auto_refused(fallback, &forced))
            }
            _ => metadata.attach_to(self.in_caller_terms(error)),
        }
    }

    /// An `Unsupported` refusal reworded in terms of the caller's type.
    fn in_caller_terms(&self, mut error: Error) -> Error {
        if let Error::Unsupported { message, .. } = &mut error {
            *message = refusal_in_caller_terms(message, &self.label);
        }
        error
    }

    /// An `Auto` extraction whose native output was refused (`fallback`, the
    /// primary cause, so the error names the response format) and whose
    /// forced-tool attempt `forced`.
    fn auto_refused(&self, fallback: &AutoFallback, forced: &str) -> Error {
        let native = if fallback.message.is_empty() {
            String::new()
        } else {
            format!(
                " ({})",
                refusal_in_caller_terms(&fallback.message, &self.label)
            )
        };
        Error::unsupported_before_sending(
            format!(
                "`{}` cannot be extracted from `{}` with its output constrained: native output was refused{native}, and forced-tool mode {forced}",
                self.label, self.model
            ),
            Some(crate::Subject::Field {
                field: RequestField::ResponseFormat,
            }),
        )
        .authored()
        .with_provider(self.provider.clone())
    }

    /// Emit the enforcement of a pass that is not `Exact` as a warning
    /// event in terms of the caller's type, coded per type so it is logged
    /// at `WARN` once per provider, model and type.
    pub(super) fn report_enforcement(&self, metadata: &ExtractionMetadata) {
        let (kind, how) = match metadata.enforcement {
            SchemaEnforcement::Exact => return,
            SchemaEnforcement::Approximate => (WarningKind::Approximated, "approximately"),
            _ => (WarningKind::Unsupported, "without constraining the output"),
        };
        let Some(field) = counted_field(metadata.mode) else {
            return;
        };
        let mut warning = Warning::new(
            kind,
            crate::Subject::Field { field },
            format!(
                "{}; `{}` is extracted from `{}` {how} (`{}` mode)",
                limitations(&metadata.degradations),
                self.label,
                self.model,
                metadata.mode,
            ),
        )
        .with_provider(self.provider.clone());
        warning.code = Some(Cow::Owned(format!(
            "extraction.{}.{}",
            metadata.enforcement, self.label
        )));
        crate::warning::emit(&self.provider, &self.model, &mut warning);
    }

    /// See [`Extractor::plan`].
    pub(super) fn plan<P: ChatProvider + ?Sized>(
        &self,
        provider: &P,
        request: &ChatRequest,
        stream: bool,
    ) -> Result<ExtractionPlan> {
        let requested = requested_mode(request);
        let (mode, guessed) = select_mode(provider, request);
        let mut metadata = ExtractionMetadata::new(mode);
        let mut request = request.clone();
        request.system.clear();
        request.messages = vec![Message::user("placeholder")];
        loop {
            let derived = self.derive(&request, &metadata)?;
            let validation = if stream {
                provider.validate_chat_stream_request(&derived)
            } else {
                provider.validate_chat_request(&derived)
            };
            match validation {
                Ok(validation) => {
                    let field = counted_field(metadata.mode);
                    let mut degradations: Vec<Warning> = Vec::new();
                    for warning in validation
                        .warnings()
                        .iter()
                        .filter(|w| field.is_some_and(|field| is_about(w, field)))
                    {
                        let warning = in_caller_terms(warning, &self.label);
                        if !degradations.contains(&warning) {
                            degradations.push(warning);
                        }
                    }
                    return Ok(ExtractionPlan {
                        mode: metadata.mode,
                        enforcement: SchemaEnforcement::from_degradations(&degradations),
                        degradations,
                        auto_fallback: metadata.auto_fallback,
                        coverage: validation.coverage(),
                    });
                }
                Err(error) => {
                    let Some(fallback) = metadata.auto_fallback_for(requested, guessed, &error)
                    else {
                        return Err(self.pre_send_error(error, &metadata));
                    };
                    metadata.fall_back(fallback);
                }
            }
        }
    }
}

/// `warning` reworded in terms of the caller's type, `label`.
pub(super) fn in_caller_terms(warning: &Warning, label: &str) -> Warning {
    let mut warning = warning.clone();
    warning.message = message_in_caller_terms(&warning.message, label);
    warning
}

/// A refusal's message in terms of the caller's type. A refusal leaves a
/// hint's outcome out (`Degradations`), so it never says the schema "was
/// sent without `strict`".
fn refusal_in_caller_terms(message: &str, label: &str) -> String {
    message_in_caller_terms(message, label)
}

/// `message` with the synthetic tool and the response schema named as the
/// caller's type, and JSON pointers into the schema written as field paths
/// (`/properties/attributes` as `Order.attributes`).
fn message_in_caller_terms(message: &str, label: &str) -> String {
    let schema = format!("the schema of `{label}`");
    let message = message
        .replace(&format!("tool `{EXTRACTION_TOOL}`"), &schema)
        .replace("the response schema", &schema)
        .replace(&format!("`{EXTRACTION_TOOL}`"), &format!("`{label}`"));
    let mut out = String::with_capacity(message.len());
    let mut rest = message.as_str();
    while let Some(start) = rest.find("`/") {
        let Some(len) = rest[start + 1..].find('`') else {
            break;
        };
        let pointer = &rest[start + 1..start + 1 + len];
        out.push_str(&rest[..start]);
        match field_path(label, pointer) {
            Some(path) => out.push_str(&format!("`{path}`")),
            None => out.push_str(&format!("`{pointer}`")),
        }
        rest = &rest[start + len + 2..];
    }
    out.push_str(rest);
    out
}

/// The field path a JSON pointer into the schema of `label` names, such as
/// `Order.items[].price`; `None` when it goes through a keyword that has no
/// field-path form.
fn field_path(label: &str, pointer: &str) -> Option<String> {
    let mut base = label.to_owned();
    let mut path = String::new();
    let mut segments = pointer
        .split('/')
        .skip(1)
        .filter(|s| !s.is_empty())
        .map(|s| s.replace("~1", "/").replace("~0", "~"));
    while let Some(segment) = segments.next() {
        match segment.as_str() {
            "properties" => path.push_str(&format!(".{}", segments.next()?)),
            "items" | "additionalItems" => path.push_str("[]"),
            "prefixItems" => path.push_str(&format!("[{}]", segments.next()?)),
            "additionalProperties" => path.push_str("[..]"),
            "$defs" | "definitions" => {
                base = segments.next()?;
                path.clear();
            }
            "anyOf" | "oneOf" | "allOf" => {
                segments.next()?;
            }
            _ => return None,
        }
    }
    Some(format!("{base}{path}"))
}

/// The last path segment of a type name, generics included:
/// `Wrapped<Inner>` for `my::Wrapped<my::Inner>`.
fn short_type_name(name: &str) -> String {
    let outer = name.split('<').next().unwrap_or(name);
    let short = outer.rsplit("::").next().unwrap_or(outer);
    match name.find('<') {
        Some(at) => format!("{short}{}", &name[at..]),
        None => short.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_name_the_callers_type() {
        assert_eq!(
            message_in_caller_terms(
                "tool `submit_structured_output` is not strict-representable: the subschema at \
                 `/properties/attributes` is a map; it was sent without `strict`",
                "Order"
            ),
            "the schema of `Order` is not strict-representable: the subschema at \
             `Order.attributes` is a map; it was sent without `strict`"
        );
        assert_eq!(
            refusal_in_caller_terms(
                "cannot send `response_format`: the response schema is not strict-representable: \
                 the subschema at `/$defs/Item/properties/tags/items` uses `not`",
                "Order"
            ),
            "cannot send `response_format`: the schema of `Order` is not strict-representable: \
             the subschema at `Item.tags[]` uses `not`"
        );
        // Pointers through other keywords, and unterminated ticks, stay.
        assert_eq!(
            message_in_caller_terms("at `/patternProperties/x` and `/", "T"),
            "at `/patternProperties/x` and `/"
        );
        assert_eq!(field_path("T", "/"), Some("T".into()));
        assert_eq!(
            field_path("T", "/properties/a~1b/anyOf/0/properties/c"),
            Some("T.a/b.c".into())
        );
    }

    #[test]
    fn short_type_names_keep_generics() {
        assert_eq!(short_type_name("my::Person"), "Person");
        assert_eq!(short_type_name("a::Wrapped<b::Inner>"), "Wrapped<b::Inner>");
    }
}
