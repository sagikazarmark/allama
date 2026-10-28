//! The strict-schema policy: [`apply_strict_policy`].

use std::fmt;

use serde_json::Value;

use super::SchemaError;
use crate::{Degradations, Error, RequestField, Result};

/// Encode a schema the caller marked strict, through a grammar's `dialect`,
/// and decide what goes out when the dialect cannot express it.
///
/// Strict mode asks the provider to constrain the model's output to the
/// schema with a grammar. Each grammar engine accepts its own subset of JSON
/// Schema, so the adapter passes in the rewrite for its engine as `dialect`
/// (`allama_openai_compat::schema::strict_schema` for OpenAI-grammar
/// endpoints, `allama_anthropic::schema::strict_schema` for Anthropic, or a
/// third party's own). What happens when the dialect fails is not the
/// provider's choice; this function applies it the same way for every
/// grammar (ADR 0003):
///
/// - the dialect succeeds: its schema goes out strict, with the
///   `Approximated` warnings it recorded in `degradations`;
/// - [`StrictSchemaError::NotRepresentable`] on a [`Degradable::Yes`]
///   field: strict mode is a hint there, so the caller's schema goes out
///   non-strict, minus its `$schema` annotation, with an `Unsupported`
///   warning on `field` ([`Degradations::unsupported_hint`]). Under
///   [`UnsupportedPolicy::Error`](crate::UnsupportedPolicy::Error) the call
///   fails with [`Error::Unsupported`] instead;
/// - [`StrictSchemaError::NotRepresentable`] on a
///   [`Degradable::NoUnconstrainedForm`] field: there is nothing non-strict
///   to send, so it fails under every policy, through
///   [`Degradations::unsupported`], which names `field` in
///   [`Error::unsupported_field`];
/// - [`StrictSchemaError::Invalid`]: a malformed schema is always an
///   [`Error::InvalidRequest`].
///
/// `subject` names the schema's owner in messages: ``"tool `lookup`"``,
/// `"the response schema"`.
///
/// This is where extraction's
/// [`NotEnforced`][crate::chat::extract::SchemaEnforcement::NotEnforced] comes
/// from: the `Unsupported` warning it records is what extraction counts,
/// and the error it returns under `UnsupportedPolicy::Error` is the
/// pre-send failure.
///
/// # The dialect
///
/// `dialect` receives the caller's schema and the request's
/// `degradations`, and returns the schema to send strict. It must record
/// its `Approximated` warnings (what it removed or widened) only when it
/// succeeds: on failure the schema goes out non-strict, unrewritten, and a
/// warning about a rewrite that was never sent would be false. Rewrite into
/// a copy and record at the end, or collect into a scratch
/// [`Degradations`] and [`extend`](Extend::extend) the request's with it on
/// success. A dialect may also refuse for reasons outside one schema, such
/// as a per-request limit the adapter counted beforehand.
///
/// ```
/// use allama::adapter::schema::{Degradable, StrictOutcome, StrictSchemaError, apply_strict_policy};
/// use allama::adapter::Degradations;
/// use allama::{RequestField, UnsupportedPolicy, WarningKind};
/// use serde_json::{Value, json};
///
/// // A grammar that closes objects and cannot express maps.
/// fn dialect(schema: &Value, d: &mut Degradations) -> Result<Value, StrictSchemaError> {
///     if schema.get("additionalProperties").is_some_and(Value::is_object) {
///         return Err(StrictSchemaError::not_representable("maps are not supported"));
///     }
///     let mut out = schema.clone();
///     out["additionalProperties"] = json!(false);
///     Ok(out)
/// }
///
/// let mut d = Degradations::new("demo", UnsupportedPolicy::Warn);
/// let person = json!({"type": "object", "properties": {"name": {"type": "string"}}});
/// let StrictOutcome { schema, strict, .. } =
///     apply_strict_policy(&person, RequestField::Tools, "tool `save`", &mut d, dialect, Degradable::Yes)?;
/// assert!(strict);
/// assert_eq!(schema["additionalProperties"], false);
///
/// let labels = json!({"type": "object", "additionalProperties": {"type": "string"}});
/// let outcome =
///     apply_strict_policy(&labels, RequestField::Tools, "tool `tag`", &mut d, dialect, Degradable::Yes)?;
/// assert!(!outcome.strict);
/// assert_eq!(outcome.schema, labels);
/// assert_eq!(d.warnings()[0].kind, WarningKind::Unsupported);
///
/// let err = apply_strict_policy(
///     &labels,
///     RequestField::ResponseFormat,
///     "the response schema",
///     &mut d,
///     dialect,
///     Degradable::NoUnconstrainedForm,
/// )
/// .unwrap_err();
/// assert_eq!(err.unsupported_field(), Some(RequestField::ResponseFormat));
/// # Ok::<(), allama::Error>(())
/// ```
///
/// # Errors
///
/// - [`Error::Unsupported`] when the dialect cannot express the schema and
///   the field cannot degrade, or the policy is
///   [`UnsupportedPolicy::Error`](crate::UnsupportedPolicy::Error);
/// - [`Error::InvalidRequest`] when the schema is malformed.
///
#[cfg_attr(
    not(feature = "chat-extract"),
    doc = "[crate::chat::extract::SchemaEnforcement::NotEnforced]: crate#cargo-features"
)]
pub fn apply_strict_policy(
    schema: &Value,
    field: RequestField,
    subject: &str,
    degradations: &mut Degradations,
    dialect: impl FnOnce(&Value, &mut Degradations) -> std::result::Result<Value, StrictSchemaError>,
    degradable: Degradable,
) -> Result<StrictOutcome> {
    match dialect(schema, degradations) {
        Ok(schema) => Ok(StrictOutcome {
            schema,
            strict: true,
        }),
        Err(StrictSchemaError::NotRepresentable { reason }) => match degradable {
            Degradable::Yes => {
                degradations.unsupported_hint(
                    field,
                    format!("{subject} is not strict-representable: {reason}; it was sent without `strict`"),
                )?;
                Ok(StrictOutcome {
                    schema: without_schema_uri(schema),
                    strict: false,
                })
            }
            Degradable::NoUnconstrainedForm => Err(degradations.unsupported(
                field,
                format!("{subject} needs strict mode, which cannot express it: {reason}"),
            )),
        },
        Err(error @ StrictSchemaError::Invalid(_)) => {
            Err(Error::invalid_request(format!("{field}: {subject}: {error}")).authored())
        }
    }
}

/// What [`apply_strict_policy`] decided to send.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct StrictOutcome {
    /// The schema to send: the dialect's rewrite when `strict`, otherwise
    /// the caller's schema minus `$schema`.
    pub schema: Value,
    /// Whether to send it as strict.
    pub strict: bool,
}

/// Whether a field can go out non-strict when its grammar cannot express
/// the schema: the input [`apply_strict_policy`] reads to decide between
/// degrading and failing.
///
/// A closed two-state set, like [`CapabilitySupport`](crate::CapabilitySupport),
/// so it is not `#[non_exhaustive]`: a field either has an unconstrained
/// form or it does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Degradable {
    /// The field has an unconstrained form, so strict mode is a hint: a tool
    /// can be sent without `strict`, and so can most response formats.
    Yes,
    /// The field has no unconstrained form, such as Anthropic's
    /// `output_config.format`, which is always grammar-constrained. A
    /// schema the grammar cannot express fails under every policy.
    NoUnconstrainedForm,
}

/// Why a strict dialect could not produce a strict schema.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum StrictSchemaError {
    /// The schema is valid, but the grammar cannot express it: a construct
    /// such as a map, an unconstrained subschema, recursion or a `$ref` that
    /// conflicts with its siblings, or a size over the grammar's limits.
    /// Where the field allows it, the schema can still be sent without
    /// strict.
    ///
    /// Build it with [`not_representable`](Self::not_representable).
    #[non_exhaustive]
    NotRepresentable {
        /// What cannot be expressed, and where.
        reason: String,
    },
    /// The schema is malformed, for example a `$ref` that does not resolve.
    Invalid(SchemaError),
}

impl StrictSchemaError {
    /// A [`NotRepresentable`](Self::NotRepresentable) error: the grammar
    /// cannot express the schema, for `reason` (what, and where).
    #[must_use]
    pub fn not_representable(reason: impl Into<String>) -> Self {
        Self::NotRepresentable {
            reason: reason.into(),
        }
    }
}

impl fmt::Display for StrictSchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotRepresentable { reason } => f.write_str(reason),
            Self::Invalid(error) => write!(f, "invalid JSON schema: {error}"),
        }
    }
}

impl std::error::Error for StrictSchemaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NotRepresentable { .. } => None,
            Self::Invalid(error) => Some(error),
        }
    }
}

impl From<SchemaError> for StrictSchemaError {
    /// A schema pass's error means the schema is malformed.
    fn from(error: SchemaError) -> Self {
        Self::Invalid(error)
    }
}

fn without_schema_uri(schema: &Value) -> Value {
    let mut out = schema.clone();
    if let Value::Object(map) = &mut out {
        map.remove("$schema");
    }
    out
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{Subject, UnsupportedPolicy, WarningKind};

    fn map() -> Value {
        json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": {"type": "string"}
        })
    }

    /// A dialect that rewrites with an `Approximated` warning on success.
    fn approximating(
        schema: &Value,
        d: &mut Degradations,
    ) -> std::result::Result<Value, StrictSchemaError> {
        let mut out = schema.clone();
        out["additionalProperties"] = json!(false);
        d.approximated(
            Subject::Field {
                field: RequestField::Tools,
            },
            "closed",
        );
        Ok(out)
    }

    fn not_representable(
        _: &Value,
        _: &mut Degradations,
    ) -> std::result::Result<Value, StrictSchemaError> {
        Err(StrictSchemaError::not_representable(
            "maps are not supported",
        ))
    }

    fn invalid(_: &Value, _: &mut Degradations) -> std::result::Result<Value, StrictSchemaError> {
        Err(SchemaError::UnresolvedRef("#/nope".into()).into())
    }

    const POLICIES: [UnsupportedPolicy; 2] = [UnsupportedPolicy::Warn, UnsupportedPolicy::Error];

    #[test]
    fn a_representable_schema_goes_out_strict_with_the_dialect_warnings() {
        for policy in POLICIES {
            let mut d = Degradations::new("demo", policy);
            let schema = json!({"type": "object"});
            let outcome = apply_strict_policy(
                &schema,
                RequestField::Tools,
                "tool `f`",
                &mut d,
                approximating,
                Degradable::Yes,
            )
            .unwrap();
            assert_eq!(
                outcome,
                StrictOutcome {
                    schema: json!({"type": "object", "additionalProperties": false}),
                    strict: true
                }
            );
            let warnings = d.into_warnings();
            assert_eq!(warnings.len(), 1, "{policy:?}");
            assert_eq!(warnings[0].kind, WarningKind::Approximated);
        }
    }

    #[test]
    fn a_degradable_field_goes_out_non_strict_under_warn() {
        let mut d = Degradations::new("demo", UnsupportedPolicy::Warn);
        let outcome = apply_strict_policy(
            &map(),
            RequestField::Tools,
            "tool `f`",
            &mut d,
            not_representable,
            Degradable::Yes,
        )
        .unwrap();
        assert!(!outcome.strict);
        // The caller's schema, minus `$schema`.
        assert_eq!(
            outcome.schema,
            json!({"type": "object", "additionalProperties": {"type": "string"}})
        );
        let warnings = d.into_warnings();
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].kind, WarningKind::Unsupported);
        assert_eq!(
            warnings[0].subject,
            Subject::Field {
                field: RequestField::Tools
            }
        );
        assert_eq!(
            warnings[0].message,
            "tool `f` is not strict-representable: maps are not supported; it was sent without \
             `strict`"
        );
    }

    #[test]
    fn a_degradable_field_fails_under_error() {
        let mut d = Degradations::new("demo", UnsupportedPolicy::Error);
        let err = apply_strict_policy(
            &map(),
            RequestField::Tools,
            "tool `f`",
            &mut d,
            not_representable,
            Degradable::Yes,
        )
        .unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
        assert_eq!(err.unsupported_field(), Some(RequestField::Tools));
        assert!(d.warnings().is_empty());
    }

    #[test]
    fn a_field_with_no_unconstrained_form_fails_under_every_policy() {
        for policy in POLICIES {
            let mut d = Degradations::new("demo", policy);
            let err = apply_strict_policy(
                &map(),
                RequestField::ResponseFormat,
                "the response schema",
                &mut d,
                not_representable,
                Degradable::NoUnconstrainedForm,
            )
            .unwrap_err();
            assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
            assert_eq!(
                err.unsupported_field(),
                Some(RequestField::ResponseFormat),
                "{policy:?}"
            );
            assert!(err.to_string().contains("maps are not supported"), "{err}");
            assert!(d.warnings().is_empty(), "{policy:?}");
        }
    }

    #[test]
    fn a_malformed_schema_fails_under_every_policy_and_field() {
        for policy in POLICIES {
            for degradable in [Degradable::Yes, Degradable::NoUnconstrainedForm] {
                let mut d = Degradations::new("demo", policy);
                let err = apply_strict_policy(
                    &map(),
                    RequestField::Tools,
                    "tool `f`",
                    &mut d,
                    invalid,
                    degradable,
                )
                .unwrap_err();
                assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
                assert!(err.to_string().contains("#/nope"), "{err}");
                assert_eq!(err.unsupported_field(), None);
                assert!(d.warnings().is_empty());
            }
        }
    }

    #[test]
    fn the_dialect_records_into_the_request_degradations() {
        // A scratch helper replayed on success, as a dialect that counts
        // per-request limits first would do.
        let mut scratch = Degradations::new("demo", UnsupportedPolicy::Warn);
        scratch.approximated(
            Subject::Field {
                field: RequestField::Tools,
            },
            "removed `pattern`",
        );
        let held = scratch.into_warnings();
        let mut d = Degradations::new("demo", UnsupportedPolicy::Warn);
        let outcome = apply_strict_policy(
            &json!({"type": "object"}),
            RequestField::Tools,
            "tool `f`",
            &mut d,
            |schema, d| {
                d.extend(held);
                Ok(schema.clone())
            },
            Degradable::Yes,
        )
        .unwrap();
        assert!(outcome.strict);
        assert_eq!(d.warnings()[0].message, "removed `pattern`");
    }

    #[test]
    fn an_unmerged_ref_is_not_representable_and_a_merge_error_is_invalid() {
        let conflict = json!({
            "type": "object",
            "properties": {"v": {"$ref": "#/$defs/S", "type": "integer"}},
            "$defs": {"S": {"type": "string"}}
        });
        let merged = super::super::merge_sibling_refs(&conflict).unwrap();
        assert!(matches!(
            merged.into_strict(),
            Err(StrictSchemaError::NotRepresentable { reason }) if reason.contains("/properties/v")
        ));

        let broken = json!({"properties": {"a": {"$ref": "#/nope", "title": "A"}}});
        let err: StrictSchemaError = super::super::merge_sibling_refs(&broken)
            .unwrap_err()
            .into();
        assert_eq!(
            err,
            StrictSchemaError::Invalid(SchemaError::UnresolvedRef("#/nope".into()))
        );
        assert_eq!(
            err.to_string(),
            "invalid JSON schema: unresolved schema reference `#/nope`"
        );
    }
}
