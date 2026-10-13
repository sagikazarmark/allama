//! Strict JSON Schema dialects of OpenAI-grammar endpoints.
//!
//! A tool or response format marked `strict` asks the endpoint to constrain
//! the model's output to the schema with a grammar. Grammar engines accept a
//! subset of JSON Schema, and each has its own rules. [`strict_schema`]
//! rewrites a portable schema into one [`SchemaDialect`], composing the
//! neutral passes in [`allama::adapter::schema`], and tells two kinds of failure
//! apart ([`StrictSchemaError`]):
//!
//! - the schema means something the dialect cannot express, such as a map
//!   (`HashMap<String, T>`): [`StrictSchemaError::NotRepresentable`];
//! - the schema is malformed (an unresolved `$ref`):
//!   [`StrictSchemaError::Invalid`].
//!
//! What to send then is not this dialect's decision.
//! [`allama::adapter::schema::apply_strict_policy`] takes the dialect and applies
//! ADR 0003's policy: a tool or response format goes out non-strict with an
//! `Unsupported` warning, or fails under `UnsupportedPolicy::Error`, and a
//! malformed schema always fails. The bundled encoders call it that way.
//!
//! The module depends only on `allama` and `serde_json`, so adapters with
//! their own envelope (a native Mistral adapter, a gateway) can reuse it.
//!
//! ```
//! use allama::adapter::schema::{Degradable, StrictSchemaError, apply_strict_policy};
//! use allama::adapter::Degradations;
//! use allama::{RequestField, UnsupportedPolicy};
//! use allama_openai_compat::schema::{SchemaDialect, strict_schema};
//! use serde_json::json;
//!
//! let mut degradations = Degradations::new("demo", UnsupportedPolicy::Warn);
//! let person = json!({
//!     "type": "object",
//!     "properties": {"name": {"type": "string"}, "age": {"type": "integer", "format": "uint8"}},
//!     "required": ["name"]
//! });
//! let strict = strict_schema(&person, RequestField::ResponseFormat, SchemaDialect::OpenAiStrict, &mut degradations)?;
//! assert_eq!(strict["additionalProperties"], false);
//! assert_eq!(strict["required"], json!(["age", "name"]));
//!
//! let labels = json!({"type": "object", "additionalProperties": {"type": "string"}});
//! let err = strict_schema(&labels, RequestField::ResponseFormat, SchemaDialect::OpenAiStrict, &mut degradations)
//!     .unwrap_err();
//! assert!(matches!(err, StrictSchemaError::NotRepresentable { .. }));
//!
//! // Through the policy, the map goes out without strict, with a warning.
//! let outcome = apply_strict_policy(
//!     &labels,
//!     RequestField::ResponseFormat,
//!     "the response schema",
//!     &mut degradations,
//!     |schema, d| strict_schema(schema, RequestField::ResponseFormat, SchemaDialect::OpenAiStrict, d),
//!     Degradable::Yes,
//! )?;
//! assert!(!outcome.strict);
//! assert_eq!(outcome.schema, labels);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::collections::BTreeSet;

use allama::adapter::Degradations;
use allama::adapter::schema::{Degradable, StrictOutcome, StrictSchemaError, apply_strict_policy};
use allama::{RequestField, Subject};
use serde_json::{Map, Value};

use crate::SchemaConstraint;

/// A strict-mode JSON Schema dialect: the rules one grammar engine applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SchemaDialect {
    /// OpenAI Structured Outputs strict mode, as described in the
    /// "Supported schemas" section of OpenAI's Structured Outputs guide.
    ///
    /// - Every object is closed (`additionalProperties: false`) and lists
    ///   all its properties as required. An optional property with a
    ///   non-null `default` (a `#[serde(default)]` field) stays
    ///   non-nullable, so the model sends a value serde accepts; other
    ///   optional properties become nullable, with `null` added to their
    ///   `enum`. Remove those `null`s from the output with
    ///   [`allama::adapter::schema::normalize_strict_nulls`] before deserializing;
    ///   extraction and `ToolCall::parse_arguments_with` do.
    /// - `oneOf` is rewritten as `anyOf`. When no value can match two of its
    ///   branches ([`allama::adapter::schema::branches_disjoint`]), as in every enum
    ///   schemars writes, the two mean the same and the rewrite is silent;
    ///   otherwise it widens the schema and is reported as `Approximated`.
    /// - `format` keeps `date-time`, `time`, `date`, `duration`, `email`,
    ///   `hostname`, `ipv4`, `ipv6` and `uuid`. Formats on numbers
    ///   (`uint32`, `double`, …) are annotations and are dropped silently;
    ///   other formats are removed with an `Approximated` warning.
    /// - Single-element `allOf`s are flattened, a `$ref` with sibling
    ///   keywords is merged with its target into one object
    ///   ([`allama::adapter::schema::merge_sibling_refs`]; no `allOf` is sent), and
    ///   `default` and `$schema` are removed. A `$ref` whose target
    ///   conflicts with its siblings is not representable.
    /// - Maps (`additionalProperties` other than `false`,
    ///   `patternProperties`, `propertyNames`), unconstrained subschemas
    ///   (`true`, `{}`, `serde_json::Value`) and a root that is not an
    ///   object are not representable.
    /// - The composition keywords OpenAI documents as unsupported
    ///   ([`rejected_keywords`](Self::rejected_keywords): `allOf`, `not`,
    ///   `dependentRequired`, `dependentSchemas`, `if`/`then`/`else`) are
    ///   not representable either. Removing them would widen the schema
    ///   silently, and sending them is rejected.
    OpenAiStrict,
    /// [`OpenAiStrict`](Self::OpenAiStrict) for fine-tuned (`ft:`) models,
    /// which also reject `pattern`, `format`, numeric bounds, string length
    /// and array length keywords. They are removed with an `Approximated`
    /// warning. Strict mode on these models does not hold when the model
    /// calls several tools in one turn
    /// ([`parallel_tool_calls_break_strict`](Self::parallel_tool_calls_break_strict)).
    OpenAiStrictFineTuned,
    /// Azure OpenAI structured outputs, which apply to every deployment
    /// (the "JSON Schema support and limitations" section of Azure's
    /// structured outputs guide,
    /// <https://learn.microsoft.com/azure/foundry/openai/how-to/structured-outputs>,
    /// as of 2026-08-24):
    ///
    /// - the [`OpenAiStrict`](Self::OpenAiStrict) rules;
    /// - the fine-tuned removals plus Azure's own unsupported keywords:
    ///   `minLength`, `maxLength`, `pattern`, `format`, `minimum`,
    ///   `maximum`, `multipleOf`, `unevaluatedProperties`, `minProperties`,
    ///   `maxProperties`, `minItems`, `maxItems`, `uniqueItems`, `contains`,
    ///   `unevaluatedItems`, `minContains` and `maxContains`, removed with an
    ///   `Approximated` warning. `patternProperties` and `propertyNames`
    ///   are on Azure's list too, but they describe a map, which no OpenAI
    ///   dialect can express, so they stay not representable;
    /// - at most 100 object properties in the schema and five levels of
    ///   object nesting (the root object is the first level, and a
    ///   recursive reference is not unrolled); a larger schema is not
    ///   representable;
    /// - strict tools do not hold under parallel tool calls
    ///   ([`parallel_tool_calls_break_strict`](Self::parallel_tool_calls_break_strict)).
    AzureStrict,
}

/// Rewrite `schema` into `dialect`'s strict form.
///
/// Keywords the dialect cannot keep are removed or rewritten and reported
/// through `degradations` as `Approximated` warnings on `field`; warnings
/// are only recorded when the rewrite succeeds, as
/// [`apply_strict_policy`] requires of
/// a dialect. The input is not modified, so on
/// [`StrictSchemaError::NotRepresentable`] the caller still holds the
/// original schema to send without strict.
///
/// Pass it to `apply_strict_policy` to decide what to send:
/// `|schema, d| strict_schema(schema, field, dialect, d)`.
///
/// # Errors
///
/// - [`StrictSchemaError::NotRepresentable`] when the schema uses a
///   construct strict mode cannot express. Strict mode is a hint, so this
///   may degrade under `UnsupportedPolicy::Warn`.
/// - [`StrictSchemaError::Invalid`] when the schema is malformed.
pub fn strict_schema(
    schema: &Value,
    field: RequestField,
    dialect: SchemaDialect,
    degradations: &mut Degradations,
) -> Result<Value, StrictSchemaError> {
    openai_strict(schema, field, dialect, degradations)
}

impl SchemaDialect {
    /// The keywords that make a schema not representable in this dialect,
    /// wherever they appear: [`strict_schema`] fails with
    /// [`StrictSchemaError::NotRepresentable`] naming the keyword, and the
    /// adapter sends the schema without strict (or fails under
    /// `UnsupportedPolicy::Error`).
    ///
    /// These are the composition keywords OpenAI's Structured Outputs guide
    /// lists as unsupported. One case is the exception: a single-element
    /// `allOf` means the same as its element and is flattened
    /// ([`allama::adapter::schema::flatten_single_all_of`]) before the check, so only
    /// an `allOf` that survives flattening (several elements, or one that
    /// conflicts with its siblings) is rejected.
    ///
    /// ```
    /// use allama_openai_compat::schema::SchemaDialect;
    ///
    /// assert!(SchemaDialect::OpenAiStrict.rejected_keywords().contains(&"not"));
    /// ```
    #[must_use]
    pub fn rejected_keywords(self) -> &'static [&'static str] {
        match self {
            Self::OpenAiStrict | Self::OpenAiStrictFineTuned | Self::AzureStrict => {
                OPENAI_REJECTED_KEYWORDS
            }
        }
    }

    /// Whether strict tools stop being enforced when the model calls
    /// several tools in one turn: true for fine-tuned OpenAI models and
    /// Azure OpenAI, whose guides say to set `parallel_tool_calls: false`
    /// with strict tools.
    ///
    /// The bundled encoders act on it when a tool goes out strict: with no
    /// caller value they send `parallel_tool_calls: false` (if the endpoint
    /// accepts the parameter at all, see
    /// [`CompatQuirks::supports_parallel_tool_calls`](crate::CompatQuirks::supports_parallel_tool_calls)),
    /// and an explicit `true` is sent with an `Unsupported` warning on the
    /// tools (an error under `UnsupportedPolicy::Error`).
    ///
    /// ```
    /// use allama_openai_compat::schema::SchemaDialect;
    ///
    /// assert!(!SchemaDialect::OpenAiStrict.parallel_tool_calls_break_strict());
    /// assert!(SchemaDialect::AzureStrict.parallel_tool_calls_break_strict());
    /// ```
    #[must_use]
    pub fn parallel_tool_calls_break_strict(self) -> bool {
        matches!(self, Self::OpenAiStrictFineTuned | Self::AzureStrict)
    }

    /// The dialect to use for `model` on an endpoint that speaks `self`:
    /// fine-tuned OpenAI models (`ft:…`) get
    /// [`OpenAiStrictFineTuned`](Self::OpenAiStrictFineTuned).
    ///
    /// `ft:` is OpenAI's naming convention for fine-tuned models
    /// (`ft:gpt-4.1-nano-2025-04-14:openai::BTz2REMH`,
    /// <https://developers.openai.com/api/docs/guides/supervised-fine-tuning>),
    /// not a portable one: other vendors' fine-tuned models (Mistral's `ft:`
    /// ids) keep their keywords. It is a schema dialect chosen by the id's
    /// shape, a wire format, so it stays when model claims do not (ADR 0015).
    /// The bundled encoders consult it only when the endpoint's quirks set
    /// [`fine_tuned_dialect`](crate::CompatQuirks::fine_tuned_dialect).
    ///
    /// ```
    /// use allama_openai_compat::schema::SchemaDialect;
    ///
    /// let dialect = SchemaDialect::OpenAiStrict;
    /// assert_eq!(dialect.for_model("gpt-4.1"), SchemaDialect::OpenAiStrict);
    /// assert_eq!(
    ///     dialect.for_model("ft:gpt-4.1-mini:acme::abc123"),
    ///     SchemaDialect::OpenAiStrictFineTuned
    /// );
    /// ```
    #[must_use]
    pub fn for_model(self, model: &str) -> Self {
        match self {
            Self::OpenAiStrict if model.starts_with("ft:") => Self::OpenAiStrictFineTuned,
            other => other,
        }
    }
}

/// Formats OpenAI strict mode accepts.
const OPENAI_FORMATS: &[&str] = &[
    "date-time",
    "time",
    "date",
    "duration",
    "email",
    "hostname",
    "ipv4",
    "ipv6",
    "uuid",
];

/// Keywords Azure OpenAI rejects in strict schemas, beyond the fine-tuned
/// list ("Unsupported type-specific keywords", Azure's structured outputs
/// guide, 2026-08-24; `format` is handled with the other formats, and
/// `patternProperties`/`propertyNames` by the map check).
const AZURE_UNSUPPORTED: &[&str] = &[
    "minLength",
    "maxLength",
    "pattern",
    "minimum",
    "maximum",
    "multipleOf",
    "unevaluatedProperties",
    "minProperties",
    "maxProperties",
    "minItems",
    "maxItems",
    "uniqueItems",
    "contains",
    "unevaluatedItems",
    "minContains",
    "maxContains",
];

/// Azure's limit on object properties in one schema.
const AZURE_MAX_PROPERTIES: usize = 100;

/// Azure's limit on object nesting, counting the root object.
const AZURE_MAX_NESTING: usize = 5;

/// Composition keywords OpenAI strict mode rejects ("Some type-specific
/// keywords are not yet supported", Structured Outputs guide). `if`, `then`
/// and `else` are reported together.
const OPENAI_REJECTED_KEYWORDS: &[&str] = &[
    "allOf",
    "not",
    "dependentRequired",
    "dependentSchemas",
    "if",
    "then",
    "else",
];

/// The conditional keywords, reported as one construct.
const CONDITIONAL_KEYWORDS: &[&str] = &["if", "then", "else"];

/// Keywords fine-tuned OpenAI models reject in strict schemas (`format` is
/// handled with the other formats).
const FINE_TUNED_UNSUPPORTED: &[&str] = &[
    "pattern",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
];

/// Keywords that make a subschema constrain its value; a subschema with
/// none of them accepts anything.
/// (The rejected composition keywords are checked first.)
const CONSTRAINING_KEYWORDS: &[&str] = &[
    "type",
    "enum",
    "const",
    "anyOf",
    "oneOf",
    "$ref",
    "properties",
    "items",
    "prefixItems",
];

/// Keywords whose value is one subschema.
const SUBSCHEMA_KEYWORDS: &[&str] = &["items", "contains", "additionalItems", "unevaluatedItems"];

/// Keywords whose value is an array of subschemas.
const SUBSCHEMA_LIST_KEYWORDS: &[&str] = &["anyOf", "oneOf", "prefixItems"];

/// Keywords whose value maps names to subschemas.
const SUBSCHEMA_MAP_KEYWORDS: &[&str] = &["properties", "$defs", "definitions"];

/// The strict dialect the bundled encoders pass to
/// [`apply_strict_policy`] for an
/// endpoint whose strict mode speaks `dialect`.
///
/// With no dialect (`None`) the schema is sent strict as written, to a
/// grammar engine whose rules allama does not know, with an `Approximated`
/// warning saying so: the engine may not enforce every keyword. A dialect
/// the endpoint is not `verified` to enforce as written
/// ([`CompatQuirks::dialect_verified`](crate::CompatQuirks::dialect_verified))
/// is applied, and reported `Approximated` the same way.
pub(crate) fn dialect_pass(
    dialect: Option<SchemaDialect>,
    verified: bool,
    field: RequestField,
    subject: &str,
) -> impl FnOnce(&Value, &mut Degradations) -> Result<Value, StrictSchemaError> + '_ {
    move |schema, degradations| match dialect {
        Some(dialect) if verified => strict_schema(schema, field, dialect, degradations),
        Some(dialect) => {
            let strict = strict_schema(schema, field, dialect, degradations)?;
            degradations.approximated(
                Subject::Field { field },
                format!(
                    "this endpoint is not known to enforce {} exactly; {subject} was sent rewritten for it",
                    dialect.label()
                ),
            );
            Ok(strict)
        }
        None => {
            degradations.approximated(
                Subject::Field { field },
                format!(
                    "this endpoint's grammar engine may not enforce every keyword; {subject} was sent unchecked"
                ),
            );
            Ok(schema.clone())
        }
    }
}

/// A schema in one position of a Chat Completions request, as encoded by
/// [`encode_constrained`].
pub(crate) struct EncodedSchema {
    /// The schema to send.
    pub(crate) schema: Value,
    /// The `strict` flag to send, if any.
    pub(crate) strict: Option<bool>,
    /// Whether a strict schema went out with decoding constrained to it.
    pub(crate) constrained: bool,
}

/// Encode a schema the caller may have marked `strict`, for a position the
/// endpoint constrains as `constraint` says
/// ([`CompatQuirks::response_format_constraint`](crate::CompatQuirks::response_format_constraint),
/// [`CompatQuirks::tool_constraint`](crate::CompatQuirks::tool_constraint)).
///
/// A strict schema goes through
/// [`apply_strict_policy`] with the
/// endpoint's [dialect](dialect_pass) wherever the endpoint constrains, and
/// is reported `Unsupported` (`never`, the message) where it does not. The
/// `strict` flag is sent only to an endpoint that takes it
/// ([`SchemaConstraint::WhenStrict`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_constrained(
    schema: &Value,
    strict: Option<bool>,
    constraint: SchemaConstraint,
    field: RequestField,
    subject: &str,
    never: impl FnOnce() -> String,
    (dialect, verified): (Option<SchemaDialect>, bool),
    degradations: &mut Degradations,
) -> allama::Result<EncodedSchema> {
    let flag = |strict: Option<bool>| strict.filter(|_| constraint == SchemaConstraint::WhenStrict);
    if strict != Some(true) {
        return Ok(EncodedSchema {
            schema: schema.clone(),
            strict: flag(strict),
            constrained: false,
        });
    }
    if constraint == SchemaConstraint::Never {
        degradations.unsupported_hint(field, never())?;
        return Ok(EncodedSchema {
            schema: schema.clone(),
            strict: None,
            constrained: false,
        });
    }
    let StrictOutcome { schema, strict, .. } = apply_strict_policy(
        schema,
        field,
        subject,
        degradations,
        dialect_pass(dialect, verified, field, subject),
        Degradable::Yes,
    )?;
    Ok(EncodedSchema {
        schema,
        strict: flag(Some(strict)),
        constrained: strict,
    })
}

/// Refuse `schema`, about to be sent as `subject`, when its root is not an
/// object, for an endpoint that takes only object roots
/// ([`CompatQuirks::object_schema_root`](crate::CompatQuirks::object_schema_root)).
///
/// OpenAI's rule, verified live on 2026-10-13 for response formats and
/// function parameters, on Chat Completions and Responses, strict or not:
/// the root must be `"type": "object"`, or a local `$ref` to such a schema,
/// or the empty schema `{}`. Any other root (a string enum, a union, a
/// lone `allOf`, `["object", "null"]`, no `type`) is rejected with HTTP 400,
/// so it is refused before sending. A `$ref` this function cannot resolve
/// is left to the endpoint.
pub(crate) fn require_object_root(
    schema: &Value,
    field: RequestField,
    subject: &str,
    degradations: &Degradations,
) -> allama::Result<()> {
    if object_root(schema) {
        return Ok(());
    }
    Err(degradations.unsupported(
        field,
        format!(
            "this endpoint takes only an object schema (`\"type\": \"object\"`) at the root of {subject}, strict or not; wrap the value in an object"
        ),
    ))
}

/// Whether `schema`'s root is one [`require_object_root`] accepts.
fn object_root(schema: &Value) -> bool {
    let Value::Object(map) = schema else {
        return false;
    };
    if map.is_empty() {
        return true;
    }
    let root = match map.get("$ref").and_then(Value::as_str) {
        Some(reference) => match reference
            .strip_prefix('#')
            .and_then(|pointer| schema.pointer(pointer))
        {
            Some(target) => target,
            None => return true,
        },
        None => schema,
    };
    root.get("type").and_then(Value::as_str) == Some("object")
}

/// The caller's `parallel_tool_calls`, gated by whether the endpoint has
/// the field (`supports_parallel_tool_calls`).
///
/// `true` is permission, not a demand: without the field the model may
/// still call several tools at once, so it is dropped silently and never
/// refused. `false` asks for something the endpoint cannot be told: a hint,
/// refused under `UnsupportedPolicy::Error`.
pub(crate) fn gate_parallel_tool_calls(
    requested: Option<bool>,
    supports_parallel_tool_calls: bool,
    degradations: &mut Degradations,
) -> allama::Result<Option<bool>> {
    match requested {
        Some(value) if supports_parallel_tool_calls => Ok(Some(value)),
        Some(false) => {
            degradations.unsupported_hint(
                RequestField::ParallelToolCalls,
                "this endpoint has no `parallel_tool_calls` field; `false` was not sent, so the model may still call several tools at once",
            )?;
            Ok(None)
        }
        Some(true) | None => Ok(None),
    }
}

/// The `parallel_tool_calls` to send once the tools are encoded, for a
/// dialect whose strict mode does not hold under parallel calls
/// ([`SchemaDialect::parallel_tool_calls_break_strict`]).
///
/// When a tool went out strict:
///
/// - with no caller value, parallel calls are turned off (`Some(false)`),
///   unless the endpoint does not accept the parameter
///   (`supports_parallel_tool_calls`) or the reasoning effort is `minimal`,
///   under which Azure's reasoning deployments make no parallel calls and
///   reject the parameter: those are gated like an explicit value;
/// - an explicit `true` is kept, with an `Unsupported` warning on the
///   tools: strict was asked for and is not honoured. It fails under
///   `UnsupportedPolicy::Error`.
///
/// `requested` is the value already gated by `supports_parallel_tool_calls`
/// ([`gate_parallel_tool_calls`]).
pub(crate) fn serial_calls_for_strict(
    requested: Option<bool>,
    dialect: Option<SchemaDialect>,
    strict_tool_sent: bool,
    supports_parallel_tool_calls: bool,
    reasoning_effort: Option<&str>,
    degradations: &mut Degradations,
) -> allama::Result<Option<bool>> {
    let Some(dialect) =
        dialect.filter(|d| strict_tool_sent && d.parallel_tool_calls_break_strict())
    else {
        return Ok(requested);
    };
    match requested {
        None if supports_parallel_tool_calls && reasoning_effort != Some("minimal") => {
            Ok(Some(false))
        }
        Some(true) => {
            let on = match dialect {
                SchemaDialect::AzureStrict => "Azure OpenAI",
                _ => "fine-tuned OpenAI models",
            };
            degradations.unsupported_hint(
                RequestField::Tools,
                format!(
                    "strict tools are not enforced under `parallel_tool_calls: true` on {on}; the tools were sent with `strict` unenforced; leave `parallel_tool_calls` unset or set it to `false`"
                ),
            )?;
            Ok(Some(true))
        }
        other => Ok(other),
    }
}

fn openai_strict(
    schema: &Value,
    field: RequestField,
    dialect: SchemaDialect,
    degradations: &mut Degradations,
) -> Result<Value, StrictSchemaError> {
    let mut out = schema.clone();
    let Value::Object(root) = &mut out else {
        return Err(StrictSchemaError::not_representable(
            "the schema root must be an object schema",
        ));
    };
    let label = dialect.label();
    // Annotation only; carries no output semantics.
    root.remove("$schema");
    allama::adapter::schema::flatten_single_all_of(&mut out);
    // Before any rewrite, so the targets merged are the caller's.
    let mut out = allama::adapter::schema::merge_sibling_refs(&out)?.into_strict()?;
    let document = out.clone();
    let mut rewrite = OpenAiRewrite {
        document: &document,
        dialect,
        one_of: false,
        formats: BTreeSet::new(),
        removed: BTreeSet::new(),
    };
    rewrite.node(&mut out, "")?;
    check_openai_root(&out, label)?;
    if dialect == SchemaDialect::AzureStrict {
        check_azure_limits(&out)?;
    }

    let subject = Subject::Field { field };
    if rewrite.one_of {
        degradations.approximated(
            subject.clone(),
            format!(
                "{label} has no `oneOf`, and `anyOf` also admits values that match several variants; `oneOf` was sent as `anyOf`"
            ),
        );
    }
    if !rewrite.formats.is_empty() {
        degradations.approximated(
            subject.clone(),
            format!(
                "{label} does not support the string formats {}; the schema was sent without them",
                join(&rewrite.formats)
            ),
        );
    }
    if !rewrite.removed.is_empty() {
        let who = match dialect {
            SchemaDialect::OpenAiStrictFineTuned => {
                "fine-tuned OpenAI models in strict mode do not support"
            }
            _ => "Azure OpenAI strict mode does not support",
        };
        degradations.approximated(
            subject,
            format!(
                "{who} the JSON Schema keywords {}; the schema was sent without them",
                join(&rewrite.removed)
            ),
        );
    }
    Ok(out)
}

impl SchemaDialect {
    /// How warnings and errors name the dialect.
    fn label(self) -> &'static str {
        match self {
            Self::OpenAiStrict | Self::OpenAiStrictFineTuned => "OpenAI strict mode",
            Self::AzureStrict => "Azure OpenAI strict mode",
        }
    }

    /// The keywords the dialect removes, with an `Approximated` warning.
    fn removed_keywords(self) -> &'static [&'static [&'static str]] {
        match self {
            Self::OpenAiStrict => &[],
            Self::OpenAiStrictFineTuned => &[FINE_TUNED_UNSUPPORTED],
            Self::AzureStrict => &[FINE_TUNED_UNSUPPORTED, AZURE_UNSUPPORTED],
        }
    }

    /// Whether the dialect removes every string `format`.
    fn removes_formats(self) -> bool {
        matches!(self, Self::OpenAiStrictFineTuned | Self::AzureStrict)
    }
}

struct OpenAiRewrite<'a> {
    /// The document `$ref`s resolve against (after `allOf` flattening and
    /// the sibling `$ref` merge).
    document: &'a Value,
    dialect: SchemaDialect,
    one_of: bool,
    formats: BTreeSet<String>,
    /// Keywords removed by [`SchemaDialect::removed_keywords`], plus
    /// `format` where the dialect removes every format.
    removed: BTreeSet<String>,
}

impl OpenAiRewrite<'_> {
    fn node(&mut self, node: &mut Value, path: &str) -> Result<(), StrictSchemaError> {
        let Value::Object(map) = node else {
            return Err(StrictSchemaError::not_representable(format!(
                "the subschema at `{}` accepts any value (`true`, `false` or `{{}}`, e.g. `serde_json::Value`)",
                display_path(path)
            )));
        };
        // `default` is the signal `close_object` reads on the parent, which
        // has run by now. Strict mode requires every property, so a default
        // never applies, and OpenAI's guide does not list the keyword
        // (OpenAI rejected it until 2026-10; it is accepted now).
        map.remove("default");
        let label = self.dialect.label();
        check_rejected(map, path, label)?;
        check_map(map, path, label)?;
        if !map
            .keys()
            .any(|k| CONSTRAINING_KEYWORDS.contains(&k.as_str()))
        {
            return Err(StrictSchemaError::not_representable(format!(
                "the subschema at `{}` accepts any value (`{{}}`, e.g. `serde_json::Value`)",
                display_path(path)
            )));
        }
        if let Some(one_of) = map.remove("oneOf") {
            // Read before the branches are rewritten: closing them would
            // make object branches look disjoint when the caller's are not.
            let disjoint = one_of.as_array().is_some_and(|branches| {
                allama::adapter::schema::branches_disjoint_in(self.document, branches)
            });
            if !disjoint {
                self.one_of = true;
            }
            map.insert("anyOf".into(), one_of);
        }
        self.format(map);
        for keyword in self.dialect.removed_keywords().iter().copied().flatten() {
            if map.remove(*keyword).is_some() {
                self.removed.insert((*keyword).to_owned());
            }
        }
        if is_object(map) {
            self.close_object(map);
        }

        for (key, value) in map.iter_mut() {
            let key = key.as_str();
            if SUBSCHEMA_MAP_KEYWORDS.contains(&key) {
                if let Value::Object(entries) = value {
                    for (name, entry) in entries.iter_mut() {
                        self.node(entry, &format!("{path}/{key}/{}", escape(name)))?;
                    }
                }
            } else if SUBSCHEMA_LIST_KEYWORDS.contains(&key) {
                if let Value::Array(entries) = value {
                    for (index, entry) in entries.iter_mut().enumerate() {
                        self.node(entry, &format!("{path}/{key}/{index}"))?;
                    }
                }
            } else if SUBSCHEMA_KEYWORDS.contains(&key) {
                match value {
                    // Draft-07 tuple form.
                    Value::Array(entries) => {
                        for (index, entry) in entries.iter_mut().enumerate() {
                            self.node(entry, &format!("{path}/{key}/{index}"))?;
                        }
                    }
                    entry => self.node(entry, &format!("{path}/{key}"))?,
                }
            }
        }
        Ok(())
    }

    /// Close an object: `additionalProperties: false` and every property
    /// required. A property the caller left optional must still be
    /// deserializable when the model fills it in:
    ///
    /// - one that already admits `null` stays as it is (`Option<T>`);
    /// - one with a non-null `default` (schemars writes one for a
    ///   `#[serde(default)]` field whose type is `Serialize`) stays
    ///   non-nullable, so the model sends a value serde accepts (`[]`,
    ///   `0`) rather than a `null` it rejects. The keyword itself is removed
    ///   later;
    /// - any other is made nullable. The `null`s the model then sends are
    ///   removed from the output by
    ///   [`allama::adapter::schema::normalize_strict_nulls`].
    fn close_object(&self, map: &mut Map<String, Value>) {
        let required: BTreeSet<String> = map
            .get("required")
            .and_then(Value::as_array)
            .map(|r| {
                r.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        if let Some(Value::Object(properties)) = map.get_mut("properties") {
            for (name, property) in properties.iter_mut() {
                let has_default = property
                    .get("default")
                    .is_some_and(|default| !default.is_null());
                if !required.contains(name) && !has_default && !self.admits_null(property, 0) {
                    make_nullable(property);
                }
            }
            let all: Vec<Value> = properties.keys().cloned().map(Value::String).collect();
            map.insert("required".into(), Value::Array(all));
        }
        map.insert("additionalProperties".into(), Value::Bool(false));
    }

    /// Whether `schema` admits `null`: every keyword that constrains the
    /// type allows it. `true` where it cannot tell.
    fn admits_null(&self, schema: &Value, depth: usize) -> bool {
        let Value::Object(map) = schema else {
            return !matches!(schema, Value::Bool(false));
        };
        // A guard against reference cycles.
        if depth > 32 {
            return true;
        }
        let type_ok = match map.get("type") {
            None => true,
            Some(Value::String(name)) => name == "null",
            Some(Value::Array(names)) => names.iter().any(|name| name == "null"),
            Some(_) => false,
        };
        let enum_ok = map
            .get("enum")
            .and_then(Value::as_array)
            .is_none_or(|values| values.contains(&Value::Null));
        let const_ok = map.get("const").is_none_or(Value::is_null);
        let ref_ok = match map.get("$ref") {
            Some(Value::String(reference)) => resolve(self.document, reference)
                .is_none_or(|target| self.admits_null(target, depth + 1)),
            _ => true,
        };
        let all_of_ok = map
            .get("allOf")
            .and_then(Value::as_array)
            .is_none_or(|members| members.iter().all(|m| self.admits_null(m, depth + 1)));
        let any_of_ok = ["anyOf", "oneOf"].iter().all(|keyword| {
            map.get(*keyword)
                .and_then(Value::as_array)
                .is_none_or(|members| members.iter().any(|m| self.admits_null(m, depth + 1)))
        });
        type_ok && enum_ok && const_ok && ref_ok && all_of_ok && any_of_ok
    }

    fn format(&mut self, map: &mut Map<String, Value>) {
        let Some(Value::String(format)) = map.get("format") else {
            return;
        };
        // `uint32`, `double`, …: annotations on numbers; bounds stay.
        if is_numeric(map) {
            map.remove("format");
            return;
        }
        if self.dialect.removes_formats() {
            map.remove("format");
            self.removed.insert("format".into());
            return;
        }
        if !OPENAI_FORMATS.contains(&format.as_str()) {
            self.formats.insert(format.clone());
            map.remove("format");
        }
    }
}

/// The composition keywords strict mode rejects
/// ([`SchemaDialect::rejected_keywords`]) make the schema not representable.
/// Dropping one would widen the schema without a trace, and sending it is
/// a 400.
fn check_rejected(
    map: &Map<String, Value>,
    path: &str,
    label: &str,
) -> Result<(), StrictSchemaError> {
    let path = display_path(path);
    let conditional: Vec<&str> = CONDITIONAL_KEYWORDS
        .iter()
        .copied()
        .filter(|k| map.contains_key(*k))
        .collect();
    if !conditional.is_empty() {
        let keywords = conditional
            .iter()
            .map(|k| format!("`{k}`"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(StrictSchemaError::not_representable(format!(
            "the subschema at `{path}` is conditional ({keywords}), and {label} does not support `if`/`then`/`else`"
        )));
    }
    if let Some(Value::Array(members)) = map.get("allOf") {
        let what = if members.len() == 1 {
            "a single-element `allOf` that conflicts with its sibling keywords".to_owned()
        } else {
            format!("`allOf` (an intersection of {} schemas)", members.len())
        };
        return Err(StrictSchemaError::not_representable(format!(
            "the subschema at `{path}` uses {what}, and {label} does not support `allOf`"
        )));
    }
    let keyword = OPENAI_REJECTED_KEYWORDS
        .iter()
        .find(|k| map.contains_key(**k) && !CONDITIONAL_KEYWORDS.contains(*k));
    match keyword {
        Some(keyword) => Err(StrictSchemaError::not_representable(format!(
            "the subschema at `{path}` uses `{keyword}`, which {label} does not support"
        ))),
        None => Ok(()),
    }
}

/// Maps have no strict representation: OpenAI requires
/// `additionalProperties: false` on every object, and a schema-valued
/// `additionalProperties` sent alongside it is rejected.
fn check_map(map: &Map<String, Value>, path: &str, label: &str) -> Result<(), StrictSchemaError> {
    let keyword = match map.get("additionalProperties") {
        None | Some(Value::Bool(false)) => None,
        Some(_) => Some("additionalProperties"),
    }
    .or_else(|| {
        map.contains_key("patternProperties")
            .then_some("patternProperties")
    })
    .or_else(|| map.contains_key("propertyNames").then_some("propertyNames"));
    match keyword {
        Some(keyword) => Err(StrictSchemaError::not_representable(format!(
            "the object at `{}` is a map (`{keyword}`), which {label} cannot express",
            display_path(path)
        ))),
        None => Ok(()),
    }
}

fn check_openai_root(root: &Value, label: &str) -> Result<(), StrictSchemaError> {
    let Value::Object(map) = root else {
        return Ok(());
    };
    if map.contains_key("anyOf") {
        return Err(StrictSchemaError::not_representable(format!(
            "the schema root is a union (`anyOf`), and {label} requires an object root"
        )));
    }
    if map.get("type").and_then(Value::as_str) != Some("object") {
        return Err(StrictSchemaError::not_representable(format!(
            "the schema root is not an object, and {label} requires an object root"
        )));
    }
    Ok(())
}

/// Azure's size limits: object properties in the whole schema (each
/// definition counted once, as written), and object nesting from the root
/// (a `$ref` is followed, a recursive one is not unrolled).
fn check_azure_limits(schema: &Value) -> Result<(), StrictSchemaError> {
    let properties = count_properties(schema);
    if properties > AZURE_MAX_PROPERTIES {
        return Err(StrictSchemaError::not_representable(format!(
            "the schema has {properties} object properties, and Azure OpenAI strict mode allows at most {AZURE_MAX_PROPERTIES}"
        )));
    }
    let nesting = object_nesting(schema, schema, &mut vec!["#"]);
    if nesting > AZURE_MAX_NESTING {
        return Err(StrictSchemaError::not_representable(format!(
            "the schema nests objects {nesting} levels deep, and Azure OpenAI strict mode allows at most {AZURE_MAX_NESTING}"
        )));
    }
    Ok(())
}

/// The number of entries in every `properties` map of the schema.
fn count_properties(node: &Value) -> usize {
    match node {
        Value::Object(map) => map
            .iter()
            .map(|(key, value)| match (key.as_str(), value) {
                ("properties", Value::Object(properties)) => {
                    properties.len() + properties.values().map(count_properties).sum::<usize>()
                }
                (key, _) if DATA_KEYWORDS.contains(&key) => 0,
                _ => count_properties(value),
            })
            .sum(),
        Value::Array(items) => items.iter().map(count_properties).sum(),
        _ => 0,
    }
}

/// How many object schemas deep `node` nests, itself included. Definitions
/// count where they are referenced.
fn object_nesting<'a>(node: &'a Value, document: &'a Value, stack: &mut Vec<&'a str>) -> usize {
    let Value::Object(map) = node else {
        return 0;
    };
    let mut deepest = 0;
    for (key, value) in map {
        let depth = match (key.as_str(), value) {
            ("$ref", Value::String(reference)) => {
                if stack.contains(&reference.as_str()) {
                    0
                } else if let Some(target) = resolve(document, reference) {
                    stack.push(reference);
                    let depth = object_nesting(target, document, stack);
                    stack.pop();
                    depth
                } else {
                    0
                }
            }
            ("properties", Value::Object(properties)) => properties
                .values()
                .map(|p| object_nesting(p, document, stack))
                .max()
                .unwrap_or(0),
            (k, Value::Array(entries)) if SUBSCHEMA_LIST_KEYWORDS.contains(&k) => entries
                .iter()
                .map(|e| object_nesting(e, document, stack))
                .max()
                .unwrap_or(0),
            (k, entry) if SUBSCHEMA_KEYWORDS.contains(&k) => object_nesting(entry, document, stack),
            _ => 0,
        };
        deepest = deepest.max(depth);
    }
    usize::from(is_object(map)) + deepest
}

/// Keywords whose values are data, not subschemas.
const DATA_KEYWORDS: &[&str] = &["enum", "const", "default", "examples"];

fn is_object(map: &Map<String, Value>) -> bool {
    match map.get("type") {
        Some(Value::String(t)) => t == "object",
        Some(Value::Array(types)) => types.iter().any(|t| t == "object"),
        _ => map.contains_key("properties"),
    }
}

fn is_numeric(map: &Map<String, Value>) -> bool {
    let numeric = |t: &Value| t == "integer" || t == "number";
    match map.get("type") {
        Some(t @ Value::String(_)) => numeric(t),
        Some(Value::Array(types)) => {
            types.iter().any(numeric) && !types.iter().any(|t| t == "string")
        }
        _ => false,
    }
}

/// Keywords that combine or point to other schemas.
const COMPOSITE_KEYWORDS: &[&str] = &["$ref", "anyOf", "oneOf", "allOf", "not"];

/// Annotations kept on the outside when a property is wrapped in `anyOf`.
const WRAPPER_ANNOTATIONS: &[&str] = &["title", "description"];

/// Make `property` admit `null`, the way strict mode accepts:
///
/// - a plain typed schema gets `null` in its `type`, and in its `enum`, so
///   the schema does not advertise a `null` it then forbids (a `const`
///   becomes a two-value `enum`);
/// - a bare `anyOf`/`oneOf` gets a `{"type": "null"}` variant;
/// - anything else (a `$ref`, `allOf`, a mix) is wrapped:
///   `{"anyOf": [<property>, {"type": "null"}]}`, with its `title` and
///   `description` kept outside.
fn make_nullable(property: &mut Value) {
    let Value::Object(map) = property else {
        return;
    };
    let composite = COMPOSITE_KEYWORDS.iter().any(|k| map.contains_key(*k));
    let has_type = map.contains_key("type");
    let has_enum = map.contains_key("enum");
    let has_const = map.contains_key("const");
    if !composite && (has_type || has_enum || has_const) && !(has_enum && has_const) {
        if let Some(ty) = map.get_mut("type") {
            match ty {
                Value::String(name) if name != "null" => {
                    let name = std::mem::take(name);
                    *ty = Value::Array(vec![Value::String(name), "null".into()]);
                }
                Value::Array(names) if !names.iter().any(|n| n == "null") => {
                    names.push("null".into());
                }
                _ => {}
            }
        }
        if let Some(Value::Array(values)) = map.get_mut("enum")
            && !values.contains(&Value::Null)
        {
            values.push(Value::Null);
        }
        if let Some(value) = map.remove("const") {
            map.insert("enum".into(), Value::Array(vec![value, Value::Null]));
        }
        return;
    }
    let union = ["anyOf", "oneOf"]
        .into_iter()
        .filter(|k| map.contains_key(*k))
        .collect::<Vec<_>>();
    let bare_union = union.len() == 1
        && !has_type
        && !has_enum
        && !has_const
        && !map.contains_key("$ref")
        && !map.contains_key("allOf")
        && !map.contains_key("not");
    if bare_union && let Some(Value::Array(variants)) = map.get_mut(union[0]) {
        variants.push(serde_json::json!({"type": "null"}));
        return;
    }
    let mut inner = std::mem::take(map);
    for key in WRAPPER_ANNOTATIONS {
        if let Some(value) = inner.remove(*key) {
            map.insert((*key).to_owned(), value);
        }
    }
    map.insert(
        "anyOf".into(),
        Value::Array(vec![
            Value::Object(inner),
            serde_json::json!({"type": "null"}),
        ]),
    );
}

fn resolve<'a>(root: &'a Value, reference: &str) -> Option<&'a Value> {
    if reference == "#" {
        return Some(root);
    }
    let path = reference.strip_prefix("#/")?;
    path.split('/').try_fold(root, |node, segment| {
        let segment = segment.replace("~1", "/").replace("~0", "~");
        node.get(segment.as_str())
    })
}

fn display_path(path: &str) -> &str {
    if path.is_empty() { "/" } else { path }
}

fn escape(name: &str) -> String {
    name.replace('~', "~0").replace('/', "~1")
}

fn join(items: &BTreeSet<String>) -> String {
    items
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests;
