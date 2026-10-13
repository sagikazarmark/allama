//! JSON Schema helpers shared by provider adapters.
//!
//! Tool parameters and `ResponseFormat::JsonSchema` carry portable JSON
//! Schema. Providers accept different dialects (OpenAI strict mode, Gemini's
//! subset, …), so adapters rewrite schemas late, with these reusable passes:
//!
//! - [`inline_refs`]: replace local `$ref`s with their `$defs` targets;
//! - [`strip_keywords`]: remove keywords a provider rejects, reporting what
//!   was removed so the adapter can emit an approximation warning;
//! - [`flatten_single_all_of`]: merge single-element `allOf`s into their
//!   parent;
//! - [`merge_sibling_refs`]: flatten a `$ref` that has sibling keywords
//!   into one object schema, which grammar engines read consistently;
//! - [`branches_disjoint`]: prove that no value matches two union branches,
//!   so a `oneOf` can become an `anyOf` without widening it;
//! - [`is_recursive`]: detect recursive references.
//!
//! These passes are neutral. A provider's strict dialect lives in the
//! provider crate and composes them:
//! `allama_openai_compat::schema::strict_schema` for OpenAI-grammar
//! endpoints, `allama_anthropic::schema::strict_schema` for Anthropic. The
//! policy for a schema a dialect cannot express is not the provider's:
//! [`apply_strict_policy`] runs any dialect, the adapter's own included,
//! and decides between sending strict, degrading to non-strict with an
//! `Unsupported` warning, and failing (ADR 0003).
//!
//! One helper works on output rather than on schemas:
//! [`normalize_strict_nulls`] removes the `null`s a strict dialect let the
//! model send for optional properties, guided by the caller's own schema, so
//! the output deserializes. Extraction and
//! [`ToolCall::parse_arguments_with`](crate::ToolCall::parse_arguments_with)
//! apply it.
//!
//! With the `schema` feature, [`schema_for`] generates a schema from a Rust
//! type with `schemars`, re-exported as [`allama::schemars`](crate::schemars).

use std::cell::Cell;
use std::collections::BTreeSet;

use serde_json::{Map, Value};

mod disjoint;
mod merge;
mod strict;
#[cfg(feature = "extract")]
mod undeclared;

pub use disjoint::{branches_disjoint, branches_disjoint_in};
pub use merge::{SiblingRefMerge, UnmergedRef, merge_sibling_refs};
pub use strict::{Degradable, StrictOutcome, StrictSchemaError, apply_strict_policy};

/// Generate the JSON Schema for `T`.
#[cfg(feature = "schema")]
#[must_use]
pub fn schema_for<T: schemars::JsonSchema>() -> Value {
    serde_json::to_value(schemars::schema_for!(T)).unwrap_or_else(|_| Value::Object(Map::new()))
}

/// Errors from schema passes.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SchemaError {
    /// A `$ref` points outside the document or to a missing definition.
    UnresolvedRef(String),
    /// The schema references itself, so it cannot be fully inlined.
    Recursive(String),
}

impl std::fmt::Display for SchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnresolvedRef(r) => write!(f, "unresolved schema reference `{r}`"),
            Self::Recursive(r) => write!(f, "recursive schema reference `{r}` cannot be inlined"),
        }
    }
}

impl std::error::Error for SchemaError {}

/// Replace every local `$ref` (`#/$defs/..`, `#/definitions/..`) with a copy
/// of its target and drop the definition tables.
///
/// # Errors
///
/// Returns [`SchemaError::Recursive`] for self-referencing schemas and
/// [`SchemaError::UnresolvedRef`] for missing or non-local targets.
pub fn inline_refs(schema: &Value) -> Result<Value, SchemaError> {
    let mut out = inline(schema, schema, &mut Vec::new())?;
    if let Value::Object(map) = &mut out {
        map.remove("$defs");
        map.remove("definitions");
    }
    Ok(out)
}

fn inline(node: &Value, root: &Value, stack: &mut Vec<String>) -> Result<Value, SchemaError> {
    match node {
        Value::Object(map) => {
            if let Some(Value::String(reference)) = map.get("$ref") {
                if stack.contains(reference) {
                    return Err(SchemaError::Recursive(reference.clone()));
                }
                let target = resolve(root, reference)
                    .ok_or_else(|| SchemaError::UnresolvedRef(reference.clone()))?;
                stack.push(reference.clone());
                let mut resolved = inline(target, root, stack)?;
                stack.pop();
                // Sibling keywords next to `$ref` (e.g. `description`) win;
                // `properties` and `required` are unioned, so the target's
                // own properties survive (see `merge_sibling_refs` for the
                // checked merge).
                if let Value::Object(resolved_map) = &mut resolved {
                    for (key, value) in map {
                        if key == "$ref" {
                            continue;
                        }
                        let value = inline(value, root, stack)?;
                        match (key.as_str(), resolved_map.get_mut(key), value) {
                            ("properties", Some(Value::Object(ours)), Value::Object(theirs)) => {
                                ours.extend(theirs);
                            }
                            ("required", Some(Value::Array(ours)), Value::Array(theirs)) => {
                                for name in theirs {
                                    if !ours.contains(&name) {
                                        ours.push(name);
                                    }
                                }
                            }
                            (_, _, value) => {
                                resolved_map.insert(key.clone(), value);
                            }
                        }
                    }
                }
                return Ok(resolved);
            }
            let mut out = Map::with_capacity(map.len());
            for (key, value) in map {
                if key == "$defs" || key == "definitions" {
                    continue;
                }
                out.insert(key.clone(), inline(value, root, stack)?);
            }
            Ok(Value::Object(out))
        }
        Value::Array(items) => items
            .iter()
            .map(|item| inline(item, root, stack))
            .collect::<Result<_, _>>()
            .map(Value::Array),
        other => Ok(other.clone()),
    }
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

/// Returns `true` when the schema contains a reference cycle.
#[must_use]
pub fn is_recursive(schema: &Value) -> bool {
    matches!(inline_refs(schema), Err(SchemaError::Recursive(_)))
}

/// Remove `keywords` everywhere in the schema (property names are left
/// alone). Returns the set of keywords that were actually removed.
pub fn strip_keywords(schema: &mut Value, keywords: &[&str]) -> BTreeSet<String> {
    let mut removed = BTreeSet::new();
    strip(schema, keywords, &mut removed, false);
    removed
}

fn strip(node: &mut Value, keywords: &[&str], removed: &mut BTreeSet<String>, in_properties: bool) {
    match node {
        Value::Object(map) => {
            if !in_properties {
                for keyword in keywords {
                    if map.remove(*keyword).is_some() {
                        removed.insert((*keyword).to_owned());
                    }
                }
            }
            for (key, value) in map.iter_mut() {
                let child_is_property_map =
                    !in_properties && (key == "properties" || key == "patternProperties");
                strip(value, keywords, removed, child_is_property_map);
            }
        }
        Value::Array(items) => {
            for item in items {
                strip(item, keywords, removed, false);
            }
        }
        _ => {}
    }
}

/// Merge every single-element `allOf` into the schema that holds it.
///
/// `{"allOf": [{"$ref": "#/$defs/A"}], "description": "an A"}` becomes
/// `{"$ref": "#/$defs/A", "description": "an A"}`. A single-element `allOf`
/// means the same as its element, but some dialects reject `allOf`
/// altogether. A keyword present on both sides is kept when the values agree
/// or when it is an annotation (`title`, `description`, …; the outer value
/// wins); any other conflict leaves that `allOf` in place. Values of data
/// keywords (`enum`, `const`, `default`, `examples`) and property names are
/// left alone.
pub fn flatten_single_all_of(schema: &mut Value) {
    flatten(schema, false);
}

fn flatten(node: &mut Value, in_properties: bool) {
    match node {
        Value::Object(map) => {
            if !in_properties {
                while merge_single_all_of(map) {}
            }
            for (key, value) in map.iter_mut() {
                if !in_properties && DATA_KEYWORDS.contains(&key.as_str()) {
                    continue;
                }
                let child_is_property_map =
                    !in_properties && (key == "properties" || key == "patternProperties");
                flatten(value, child_is_property_map);
            }
        }
        Value::Array(items) => {
            for item in items {
                flatten(item, false);
            }
        }
        _ => {}
    }
}

/// Keywords whose values are data, not subschemas.
const DATA_KEYWORDS: &[&str] = &["enum", "const", "default", "examples"];

/// Annotation keywords: they describe a value without constraining it.
///
/// A rewrite that removes or replaces only these changes nothing a value
/// must satisfy.
pub const ANNOTATION_KEYWORDS: &[&str] = &[
    "title",
    "description",
    "$comment",
    "examples",
    "default",
    "deprecated",
    "readOnly",
    "writeOnly",
];

/// Merge `allOf: [X]` into `map`. Returns `true` when it merged.
fn merge_single_all_of(map: &mut Map<String, Value>) -> bool {
    let Some(Value::Array(all_of)) = map.get("allOf") else {
        return false;
    };
    let [only] = all_of.as_slice() else {
        return false;
    };
    match only {
        Value::Bool(true) => {}
        Value::Object(inner) => {
            let conflict = inner.iter().any(|(key, value)| {
                map.get(key).is_some_and(|outer| {
                    outer != value && !ANNOTATION_KEYWORDS.contains(&key.as_str())
                })
            });
            if conflict {
                return false;
            }
            let inner = inner.clone();
            for (key, value) in inner {
                map.entry(key).or_insert(value);
            }
        }
        _ => return false,
    }
    map.remove("allOf");
    true
}

/// Remove the `null`s in `value` that stand in for optional properties the
/// model left out, guided by `original_schema`.
///
/// Strict modes that require every property to be listed (OpenAI's, for
/// one) express an optional property by making it nullable, so the model
/// sends `"tags": null` instead of leaving `tags` out. serde rejects that
/// for a `#[serde(default)] tags: Vec<String>`. Removing the property lets
/// serde apply the default, as if the model had left it out.
///
/// `original_schema` is the schema the caller wrote (for example
/// `schema_for::<T>()`), not the provider's strict rewrite of it, so the
/// rule is the caller's own and holds whichever provider produced `value`.
/// A `null` is removed only for an object key that, across the schemas
/// that apply to the object (its own, its `$ref` target, its `allOf`
/// members, and the `anyOf`/`oneOf` variants the object can match):
///
/// - is in `properties`;
/// - is in no `required`;
/// - has a schema that admits no `null`.
///
/// Every other `null` stays, and serde reports it if `T` rejects it:
///
/// - a `null` the schema admits: `Option<T>`, `Option<Option<T>>`,
///   `serde_json::Value`, `{"type": ["string", "null"]}`, an `enum` that
///   lists `null`;
/// - the value of a required property (`name: String` reports "invalid
///   type: null", not "missing field");
/// - a map entry, matched by `additionalProperties` or `patternProperties`
///   rather than `properties`: a `HashMap<String, String>` entry is real
///   data, never a stand-in;
/// - wherever the schema leaves the answer open (an unresolved `$ref`,
///   variants that disagree), an array element, and the root value.
///
/// The walk descends through `properties`, `additionalProperties`, `items`
/// and `prefixItems`, so nested objects (map values included) are
/// normalized by the same rule.
///
/// The pass is idempotent and harmless on output that was not produced
/// under strict mode.
///
/// ```
/// use allama::adapter::schema::normalize_strict_nulls;
/// use serde_json::json;
///
/// // `#[serde(default)] tags: Vec<String>` and `nickname: Option<String>`.
/// let schema = json!({
///     "type": "object",
///     "properties": {
///         "tags": {"type": "array", "items": {"type": "string"}},
///         "nickname": {"type": ["string", "null"]}
///     }
/// });
/// let mut output = json!({"tags": null, "nickname": null});
/// normalize_strict_nulls(&schema, &mut output);
/// assert_eq!(output, json!({"nickname": null}));
/// ```
pub fn normalize_strict_nulls(original_schema: &Value, value: &mut Value) {
    let nulls = Nulls::new(original_schema);
    let applicable = nulls.expand(original_schema, value, 0);
    nulls.walk(&applicable, value);
}

/// Deserialize `raw` into `T`, after [`normalize_strict_nulls`] against
/// `original_schema` when there is one.
///
/// When valid JSON does not fit `T`, the failure carries the path to the
/// value that did not ([`DeserializeFailure::path`]).
pub(crate) fn from_str_normalized<T: serde::de::DeserializeOwned>(
    raw: &str,
    original_schema: Option<&Value>,
) -> Result<T, DeserializeFailure> {
    let error = match parse_normalized(raw, original_schema) {
        Ok(value) => return Ok(value),
        Err(error) => error,
    };
    // The path costs a second pass, so it is found only after a failure,
    // and only for output that is valid JSON.
    let path = match error.classify() {
        serde_json::error::Category::Data => field_path::<T>(raw, original_schema),
        _ => None,
    };
    Err(DeserializeFailure {
        error,
        path,
        undeclared: false,
    })
}

/// [`from_str_normalized`], rejecting first a key `original_schema` does
/// not declare ([`undeclared::find_undeclared_key`]) when
/// `reject_undeclared` is set. The rejection reads like serde's own
/// (``unknown field `e_mail`, expected one of `email`, `name` ``), with the
/// path to the key.
#[cfg(feature = "extract")]
pub(crate) fn from_str_checked<T: serde::de::DeserializeOwned>(
    raw: &str,
    original_schema: Option<&Value>,
    reject_undeclared: bool,
) -> Result<T, DeserializeFailure> {
    let Some(schema) = original_schema.filter(|_| reject_undeclared) else {
        return from_str_normalized(raw, original_schema);
    };
    let mut value: Value = serde_json::from_str(raw).map_err(|error| DeserializeFailure {
        error,
        path: None,
        undeclared: false,
    })?;
    normalize_strict_nulls(schema, &mut value);
    let Some(found) = undeclared::find_undeclared_key(schema, &value) else {
        return from_str_normalized(raw, original_schema);
    };
    let expected = match found.declared.as_slice() {
        [] => "there are no fields".to_owned(),
        [only] => format!("expected `{only}`"),
        keys => format!(
            "expected one of {}",
            keys.iter()
                .map(|k| format!("`{k}`"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    Err(DeserializeFailure {
        error: serde::de::Error::custom(format!("unknown field `{}`, {expected}", found.key)),
        path: Some(found.path),
        undeclared: true,
    })
}

/// Output that could not be deserialized into the target type.
#[derive(Debug)]
pub(crate) struct DeserializeFailure {
    /// The serde error.
    pub(crate) error: serde_json::Error,
    /// The path to the value that did not fit the target type, such as
    /// `items[2].name`; see [`field_path`].
    pub(crate) path: Option<String>,
    /// The failure is a key the schema does not declare
    /// ([`from_str_checked`]).
    #[cfg_attr(not(feature = "extract"), allow(dead_code))]
    pub(crate) undeclared: bool,
}

fn parse_normalized<T: serde::de::DeserializeOwned>(
    raw: &str,
    original_schema: Option<&Value>,
) -> serde_json::Result<T> {
    let Some(schema) = original_schema else {
        return serde_json::from_str(raw);
    };
    let mut value: Value = serde_json::from_str(raw)?;
    normalize_strict_nulls(schema, &mut value);
    serde_json::from_value(value)
}

/// The path to the value in `raw` that does not fit `T`, found by
/// deserializing again with `serde_path_to_error`.
///
/// `None` when the failure is at the root, and without the `schema`
/// feature. The path is that of the first error serde meets, and stops at
/// an `untagged` enum or a `flatten`ed field: serde buffers their content,
/// so the path names the enum or the struct holding the `flatten`, not the
/// value inside.
#[cfg_attr(
    not(feature = "schema"),
    expect(clippy::extra_unused_type_parameters, reason = "used with `schema`")
)]
fn field_path<T: serde::de::DeserializeOwned>(
    raw: &str,
    original_schema: Option<&Value>,
) -> Option<String> {
    #[cfg(feature = "schema")]
    {
        let mut value: Value = serde_json::from_str(raw).ok()?;
        if let Some(schema) = original_schema {
            normalize_strict_nulls(schema, &mut value);
        }
        let error = serde_path_to_error::deserialize::<_, T>(value).err()?;
        let path = error.path();
        path.iter().next().is_some().then(|| path.to_string())
    }
    #[cfg(not(feature = "schema"))]
    {
        let _ = (raw, original_schema);
        None
    }
}

/// The schema `true`: admits every value.
static ANY_VALUE: Value = Value::Bool(true);

/// How deep `$ref`, `allOf` and `anyOf` may nest at one point of a value
/// before the walk stops following them (a guard against reference cycles
/// that never reach a value, such as `A: {"$ref": "#/$defs/A"}`).
const MAX_SCHEMA_DEPTH: usize = 32;

/// How many subschema expansions one [`normalize_strict_nulls`] call may do
/// in total. Recursive unions whose variants share a field (an untagged
/// `enum Tree { A { x: Option<Box<Tree>> }, B { x: Option<Box<Tree>> } }`)
/// otherwise double the work with each level of nesting, so model output
/// could stall the caller. Past the budget the rest of the value is walked
/// against the schema `true`, which keeps its nulls: the safe direction,
/// since a kept `null` still fits an `Option`.
const MAX_EXPANSIONS: usize = 20_000;

/// The subschemas that apply to one point of a value.
enum Applicable<'s> {
    /// One schema's own keywords (`properties`, `items`, `type`, …).
    Schema(&'s Value),
    /// The value must match every member (a schema with its `$ref` and
    /// `allOf`).
    All(Vec<Applicable<'s>>),
    /// The value matches at least one member (`anyOf`, `oneOf`).
    Any(Vec<Applicable<'s>>),
}

impl<'s> Applicable<'s> {
    /// What applies to the property `key` of an object here. Not expanded.
    fn property(&self, key: &str) -> Self {
        match self {
            Self::Schema(schema) => Self::Schema(own_property(schema, key)),
            Self::All(members) => Self::All(members.iter().map(|m| m.property(key)).collect()),
            Self::Any(members) => Self::Any(members.iter().map(|m| m.property(key)).collect()),
        }
    }

    /// What applies to element `index` of an array here. Not expanded.
    fn item(&self, index: usize) -> Self {
        match self {
            Self::Schema(schema) => Self::Schema(own_item(schema, index)),
            Self::All(members) => Self::All(members.iter().map(|m| m.item(index)).collect()),
            Self::Any(members) => Self::Any(members.iter().map(|m| m.item(index)).collect()),
        }
    }
}

/// What the applicable schemas say about one property of an object.
struct KeyFacts {
    /// The key is in `properties`.
    declared: bool,
    /// The key is in `required`.
    required: bool,
    /// The property's schema admits `null`.
    admits_null: bool,
}

struct Nulls<'s> {
    /// The document `$ref`s resolve against.
    root: &'s Value,
    /// Expansions left ([`MAX_EXPANSIONS`]).
    budget: Cell<usize>,
}

impl<'s> Nulls<'s> {
    fn new(root: &'s Value) -> Self {
        Self {
            root,
            budget: Cell::new(MAX_EXPANSIONS),
        }
    }

    fn walk(&self, applicable: &Applicable<'s>, value: &mut Value) {
        match value {
            Value::Object(map) => {
                map.retain(|key, child| !child.is_null() || !self.is_stand_in(applicable, key));
                for (key, child) in map.iter_mut() {
                    if child.is_object() || child.is_array() {
                        let expanded = self.expand_all(&applicable.property(key), child);
                        self.walk(&expanded, child);
                    }
                }
            }
            Value::Array(items) => {
                for (index, item) in items.iter_mut().enumerate() {
                    if item.is_object() || item.is_array() {
                        let expanded = self.expand_all(&applicable.item(index), item);
                        self.walk(&expanded, item);
                    }
                }
            }
            _ => {}
        }
    }

    /// Expand every schema in `applicable` against `value`. Once the
    /// budget is spent, the schema `true` instead, so the rest of the walk
    /// is linear in the value.
    fn expand_all(&self, applicable: &Applicable<'s>, value: &Value) -> Applicable<'s> {
        let expanded = self.expand_members(applicable, value);
        if self.budget.get() == 0 {
            Applicable::Schema(&ANY_VALUE)
        } else {
            expanded
        }
    }

    fn expand_members(&self, applicable: &Applicable<'s>, value: &Value) -> Applicable<'s> {
        match applicable {
            Applicable::Schema(schema) => self.expand(schema, value, 0),
            Applicable::All(members) => Applicable::All(
                members
                    .iter()
                    .map(|m| self.expand_members(m, value))
                    .collect(),
            ),
            Applicable::Any(members) => Applicable::Any(
                members
                    .iter()
                    .map(|m| self.expand_members(m, value))
                    .collect(),
            ),
        }
    }

    /// `schema` with its `$ref`, `allOf` and the `anyOf`/`oneOf` variants
    /// `value` may match.
    fn expand(&self, schema: &'s Value, value: &Value, depth: usize) -> Applicable<'s> {
        let Value::Object(map) = schema else {
            return Applicable::Schema(schema);
        };
        let budget = self.budget.get();
        if depth > MAX_SCHEMA_DEPTH || budget == 0 {
            return Applicable::Schema(&ANY_VALUE);
        }
        self.budget.set(budget - 1);
        let mut all = vec![Applicable::Schema(schema)];
        if let Some(Value::String(reference)) = map.get("$ref")
            && let Some(target) = resolve(self.root, reference)
        {
            all.push(self.expand(target, value, depth + 1));
        }
        if let Some(Value::Array(members)) = map.get("allOf") {
            all.extend(members.iter().map(|m| self.expand(m, value, depth + 1)));
        }
        for keyword in ["anyOf", "oneOf"] {
            let Some(Value::Array(members)) = map.get(keyword) else {
                continue;
            };
            let mut matching: Vec<&Value> = members
                .iter()
                .filter(|m| self.may_match(m, value, depth + 1))
                .collect();
            // No variant fits: keep them all, so a null stays unless every
            // variant forbids it.
            if matching.is_empty() {
                matching = members.iter().collect();
            }
            all.push(Applicable::Any(
                matching
                    .into_iter()
                    .map(|m| self.expand(m, value, depth + 1))
                    .collect(),
            ));
        }
        if all.len() == 1 {
            all.pop().unwrap_or(Applicable::Schema(&ANY_VALUE))
        } else {
            Applicable::All(all)
        }
    }

    /// Whether a `null` under `key` of an object that `applicable` describes
    /// stands in for a left-out property: across the applicable schemas,
    /// `key` is in `properties`, in no `required`, and its schema admits no
    /// `null`.
    fn is_stand_in(&self, applicable: &Applicable<'s>, key: &str) -> bool {
        let facts = self.key_facts(applicable, key);
        facts.declared && !facts.required && !facts.admits_null
    }

    /// What the schemas in `applicable` say about the property `key`.
    ///
    /// Every member of [`Applicable::All`] applies, so the key is declared
    /// or required when any member says so, and admits `null` only when
    /// every member does. The value matches some member of
    /// [`Applicable::Any`], without saying which, so there the key counts as
    /// declared only when every member declares it, and as required or
    /// nullable when any member makes it so.
    fn key_facts(&self, applicable: &Applicable<'s>, key: &str) -> KeyFacts {
        match applicable {
            Applicable::Schema(schema) => KeyFacts {
                declared: schema
                    .get("properties")
                    .and_then(|properties| properties.get(key))
                    .is_some(),
                required: schema
                    .get("required")
                    .and_then(Value::as_array)
                    .is_some_and(|required| required.iter().any(|k| k == key)),
                admits_null: self.schema_admits_null(own_property(schema, key), 0),
            },
            Applicable::All(members) => {
                let facts: Vec<_> = members.iter().map(|m| self.key_facts(m, key)).collect();
                KeyFacts {
                    declared: facts.iter().any(|f| f.declared),
                    required: facts.iter().any(|f| f.required),
                    admits_null: facts.iter().all(|f| f.admits_null),
                }
            }
            Applicable::Any(members) => {
                let facts: Vec<_> = members.iter().map(|m| self.key_facts(m, key)).collect();
                KeyFacts {
                    declared: !facts.is_empty() && facts.iter().all(|f| f.declared),
                    required: facts.iter().any(|f| f.required),
                    admits_null: facts.iter().any(|f| f.admits_null),
                }
            }
        }
    }

    /// Whether `schema` admits `null`. `true` where it cannot tell.
    fn schema_admits_null(&self, schema: &Value, depth: usize) -> bool {
        let Value::Object(map) = schema else {
            return !matches!(schema, Value::Bool(false));
        };
        if depth > MAX_SCHEMA_DEPTH {
            return true;
        }
        if let Some(ty) = map.get("type")
            && !type_matches(ty, &Value::Null)
        {
            return false;
        }
        if let Some(Value::Array(values)) = map.get("enum")
            && !values.contains(&Value::Null)
        {
            return false;
        }
        if map.get("const").is_some_and(|c| !c.is_null()) {
            return false;
        }
        if let Some(Value::String(reference)) = map.get("$ref")
            && let Some(target) = resolve(self.root, reference)
            && !self.schema_admits_null(target, depth + 1)
        {
            return false;
        }
        if let Some(Value::Array(members)) = map.get("allOf")
            && !members
                .iter()
                .all(|m| self.schema_admits_null(m, depth + 1))
        {
            return false;
        }
        for keyword in ["anyOf", "oneOf"] {
            if let Some(Value::Array(members)) = map.get(keyword)
                && !members
                    .iter()
                    .any(|m| self.schema_admits_null(m, depth + 1))
            {
                return false;
            }
        }
        true
    }

    /// Whether `value` may match `schema`: a cheap check of the keywords
    /// that tell union variants apart (`type`, `const`, `enum`, `required`,
    /// closed objects, and `const`/`enum` discriminator properties). `true`
    /// where it cannot tell.
    fn may_match(&self, schema: &Value, value: &Value, depth: usize) -> bool {
        let Value::Object(map) = schema else {
            return !matches!(schema, Value::Bool(false));
        };
        if depth > MAX_SCHEMA_DEPTH {
            return true;
        }
        if map.get("type").is_some_and(|ty| !type_matches(ty, value)) {
            return false;
        }
        if let Some(Value::Array(values)) = map.get("enum")
            && !values.contains(value)
        {
            return false;
        }
        if map.get("const").is_some_and(|c| c != value) {
            return false;
        }
        if let Value::Object(object) = value
            && !object_may_match(map, object)
        {
            return false;
        }
        if let Some(Value::String(reference)) = map.get("$ref")
            && let Some(target) = resolve(self.root, reference)
            && !self.may_match(target, value, depth + 1)
        {
            return false;
        }
        if let Some(Value::Array(members)) = map.get("allOf")
            && !members.iter().all(|m| self.may_match(m, value, depth + 1))
        {
            return false;
        }
        for keyword in ["anyOf", "oneOf"] {
            if let Some(Value::Array(members)) = map.get(keyword)
                && !members.iter().any(|m| self.may_match(m, value, depth + 1))
            {
                return false;
            }
        }
        true
    }
}

fn object_may_match(schema: &Map<String, Value>, object: &Map<String, Value>) -> bool {
    if let Some(Value::Array(required)) = schema.get("required")
        && required
            .iter()
            .filter_map(Value::as_str)
            .any(|key| !object.contains_key(key))
    {
        return false;
    }
    let properties = schema.get("properties").and_then(Value::as_object);
    if schema.get("additionalProperties") == Some(&Value::Bool(false))
        && !schema.contains_key("patternProperties")
        && object
            .keys()
            .any(|key| !properties.is_some_and(|p| p.contains_key(key)))
    {
        return false;
    }
    let Some(properties) = properties else {
        return true;
    };
    // Discriminators: a property pinned by `const` or `enum`. A `null` may
    // be a strict-mode stand-in for an absent property, so it decides
    // nothing.
    properties.iter().all(|(key, property)| {
        let Some(actual) = object.get(key).filter(|v| !v.is_null()) else {
            return true;
        };
        let const_ok = property.get("const").is_none_or(|c| c == actual);
        let enum_ok = property
            .get("enum")
            .and_then(Value::as_array)
            .is_none_or(|values| values.contains(actual));
        const_ok && enum_ok
    })
}

/// Whether `value` has one of the JSON types `ty` names.
fn type_matches(ty: &Value, value: &Value) -> bool {
    let matches = |name: &Value| match name.as_str() {
        Some("null") => value.is_null(),
        Some("boolean") => value.is_boolean(),
        Some("string") => value.is_string(),
        Some("number") => value.is_number(),
        Some("integer") => {
            value.is_i64() || value.is_u64() || value.as_f64().is_some_and(|n| n.fract() == 0.0)
        }
        Some("object") => value.is_object(),
        Some("array") => value.is_array(),
        // An unknown type name: cannot tell.
        _ => true,
    };
    match ty {
        Value::Array(names) => names.iter().any(matches),
        name => matches(name),
    }
}

/// The schema that `schema` itself gives the property `key`.
fn own_property<'s>(schema: &'s Value, key: &str) -> &'s Value {
    let Value::Object(map) = schema else {
        return &ANY_VALUE;
    };
    if let Some(property) = map
        .get("properties")
        .and_then(|properties| properties.get(key))
    {
        return property;
    }
    match map.get("additionalProperties") {
        Some(additional @ Value::Object(_)) => additional,
        // `false`, `true`, absent, or a key `patternProperties` may match:
        // nothing to go on.
        _ => &ANY_VALUE,
    }
}

/// The schema that `schema` itself gives array element `index`.
fn own_item(schema: &Value, index: usize) -> &Value {
    let Value::Object(map) = schema else {
        return &ANY_VALUE;
    };
    if let Some(Value::Array(prefix)) = map.get("prefixItems") {
        if let Some(item) = prefix.get(index) {
            return item;
        }
        return map.get("items").unwrap_or(&ANY_VALUE);
    }
    match map.get("items") {
        // Draft-07 tuple form.
        Some(Value::Array(tuple)) => tuple
            .get(index)
            .or_else(|| map.get("additionalItems"))
            .unwrap_or(&ANY_VALUE),
        Some(items) => items,
        None => &ANY_VALUE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn inline_refs_resolves_defs_and_keeps_siblings() {
        let schema = json!({
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/A", "description": "an A"}},
            "$defs": {"A": {"type": "object", "properties": {"b": {"$ref": "#/$defs/B"}}}, "B": {"type": "string"}}
        });
        let inlined = inline_refs(&schema).unwrap();
        assert_eq!(
            inlined,
            json!({
                "type": "object",
                "properties": {"a": {"type": "object", "description": "an A", "properties": {"b": {"type": "string"}}}}
            })
        );
    }

    #[test]
    fn inline_refs_unions_sibling_properties() {
        let schema = json!({
            "$ref": "#/$defs/Inner",
            "properties": {"kind": {"const": "New"}},
            "required": ["kind"],
            "$defs": {"Inner": {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}}
        });
        assert_eq!(
            inline_refs(&schema).unwrap(),
            json!({
                "type": "object",
                "properties": {"a": {"type": "integer"}, "kind": {"const": "New"}},
                "required": ["a", "kind"]
            })
        );
    }

    #[test]
    fn recursion_and_missing_refs_are_detected() {
        let recursive = json!({"$defs": {"N": {"properties": {"next": {"$ref": "#/$defs/N"}}}}, "$ref": "#/$defs/N"});
        assert!(is_recursive(&recursive));
        assert_eq!(
            inline_refs(&json!({"$ref": "#/$defs/missing"})),
            Err(SchemaError::UnresolvedRef("#/$defs/missing".into()))
        );
    }

    #[test]
    fn strip_keywords_skips_property_names() {
        let mut schema = json!({
            "type": "object",
            "properties": {"format": {"type": "string", "format": "email", "pattern": "x"}},
            "additionalProperties": false
        });
        let removed = strip_keywords(&mut schema, &["format", "pattern", "additionalProperties"]);
        assert_eq!(
            removed.into_iter().collect::<Vec<_>>(),
            vec!["additionalProperties", "format", "pattern"]
        );
        assert_eq!(
            schema,
            json!({"type": "object", "properties": {"format": {"type": "string"}}})
        );
    }

    #[test]
    fn single_all_of_is_flattened() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "a": {"allOf": [{"$ref": "#/$defs/A"}], "description": "an A"},
                "b": {"allOf": [{"type": "string", "description": "inner"}], "description": "outer"},
                "allOf": {"type": "string"}
            },
            "$defs": {"A": {"type": "integer"}}
        });
        flatten_single_all_of(&mut schema);
        assert_eq!(
            schema["properties"]["a"],
            json!({"$ref": "#/$defs/A", "description": "an A"})
        );
        assert_eq!(
            schema["properties"]["b"],
            json!({"type": "string", "description": "outer"})
        );
        // A property *named* `allOf` is not a keyword.
        assert_eq!(schema["properties"]["allOf"], json!({"type": "string"}));
    }

    #[test]
    fn conflicting_or_multi_element_all_of_is_kept() {
        let conflicting = json!({"type": "string", "allOf": [{"type": "integer"}]});
        let multi = json!({"allOf": [{"type": "string"}, {"minLength": 1}]});
        let data = json!({"default": {"allOf": [{"type": "string"}]}});
        for original in [conflicting, multi, data] {
            let mut schema = original.clone();
            flatten_single_all_of(&mut schema);
            assert_eq!(schema, original);
        }
    }

    fn normalized(schema: &Value, mut value: Value) -> Value {
        normalize_strict_nulls(schema, &mut value);
        value
    }

    #[test]
    fn nulls_are_removed_where_the_schema_forbids_them() {
        let schema = json!({
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
                "count": {"type": "integer"},
                "nickname": {"type": ["string", "null"]},
                "status": {"type": "string", "enum": ["open", "closed"]},
                "level": {"type": ["string", "null"], "enum": ["low", "high"]}
            }
        });
        assert_eq!(
            normalized(
                &schema,
                json!({"tags": null, "count": null, "nickname": null, "status": null, "level": null})
            ),
            json!({"nickname": null})
        );
        // Non-null values are left alone.
        let full =
            json!({"tags": ["a"], "count": 1, "nickname": "x", "status": "open", "level": "low"});
        assert_eq!(normalized(&schema, full.clone()), full);
    }

    #[test]
    fn nulls_the_schema_admits_are_kept() {
        let schema = json!({
            "type": "object",
            "properties": {
                "option": {"type": ["string", "null"]},
                "option_ref": {"anyOf": [{"$ref": "#/$defs/Color"}, {"type": "null"}]},
                "nullable_enum": {"enum": ["a", null]},
                "null_const": {"const": null},
                "any_value": true,
                "described": {"description": "anything"},
                "ref_to_nullable": {"$ref": "#/$defs/Maybe"},
                "all_of_nullable": {"allOf": [{"type": ["integer", "null"]}, {"minimum": 0}]},
                "unresolved": {"$ref": "#/$defs/Missing"},
                "not": {"not": {"type": "string"}}
            },
            "$defs": {
                "Color": {"type": "string", "enum": ["red"]},
                "Maybe": {"type": ["object", "null"], "properties": {}}
            }
        });
        let value = json!({
            "option": null, "option_ref": null, "nullable_enum": null, "null_const": null,
            "any_value": null, "described": null, "ref_to_nullable": null,
            "all_of_nullable": null, "unresolved": null, "not": null,
            "undeclared": null
        });
        assert_eq!(normalized(&schema, value.clone()), value);
    }

    #[test]
    fn nested_objects_arrays_and_defs_are_walked() {
        let schema = json!({
            "type": "object",
            "properties": {
                "home": {"$ref": "#/$defs/Address", "description": "Where they live"},
                "visits": {"type": "array", "items": {"$ref": "#/$defs/Address"}},
                "pair": {"type": "array", "prefixItems": [{"$ref": "#/$defs/Address"}, {"type": "object", "properties": {"n": {"type": "integer"}}}]},
                "by_name": {"type": "object", "additionalProperties": {"$ref": "#/$defs/Address"}},
                "via_all_of": {"allOf": [{"$ref": "#/$defs/Address"}]}
            },
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "lines": {"type": "array", "items": {"type": "string"}},
                        "zip": {"type": ["string", "null"]}
                    }
                }
            }
        });
        let address = json!({"lines": null, "zip": null});
        let value = json!({
            "home": address,
            "visits": [address, null],
            "pair": [address, {"n": null}],
            "by_name": {"x": address},
            "via_all_of": address
        });
        let clean = json!({"zip": null});
        assert_eq!(
            normalized(&schema, value),
            json!({
                "home": clean,
                // A null element stays: only properties are removed.
                "visits": [clean, null],
                "pair": [clean, {}],
                "by_name": {"x": clean},
                "via_all_of": clean
            })
        );
    }

    #[test]
    fn union_variants_are_told_apart() {
        // An internally tagged enum: the null in `r` is removed only when
        // the variant the value matches forbids it.
        let schema = json!({
            "type": "object",
            "properties": {"shape": {"oneOf": [
                {"type": "object", "properties": {"kind": {"const": "circle"}, "r": {"type": "number"}}, "required": ["kind"]},
                {"type": "object", "properties": {"kind": {"const": "blob"}, "r": {"type": ["number", "null"]}}, "required": ["kind"]}
            ]}}
        });
        assert_eq!(
            normalized(&schema, json!({"shape": {"kind": "circle", "r": null}})),
            json!({"shape": {"kind": "circle"}})
        );
        assert_eq!(
            normalized(&schema, json!({"shape": {"kind": "blob", "r": null}})),
            json!({"shape": {"kind": "blob", "r": null}})
        );
        // No variant fits: the null stays unless every variant forbids it.
        assert_eq!(
            normalized(&schema, json!({"shape": {"kind": "square", "r": null}})),
            json!({"shape": {"kind": "square", "r": null}})
        );
        // An externally tagged enum, told apart by `required`.
        let schema = json!({
            "anyOf": [
                {"type": "object", "properties": {"A": {"type": "object", "properties": {"x": {"type": "string"}}}}, "required": ["A"], "additionalProperties": false},
                {"type": "object", "properties": {"B": {"type": "object", "properties": {"x": {"type": ["string", "null"]}}}}, "required": ["B"], "additionalProperties": false}
            ]
        });
        assert_eq!(
            normalized(&schema, json!({"A": {"x": null}})),
            json!({"A": {}})
        );
        assert_eq!(
            normalized(&schema, json!({"B": {"x": null}})),
            json!({"B": {"x": null}})
        );
    }

    #[test]
    fn map_entries_keep_their_nulls() {
        // A map entry is data, never a stand-in for a left-out property,
        // whatever its value schema admits.
        let schema = json!({
            "type": "object",
            "properties": {
                "labels": {"type": "object", "additionalProperties": {"type": "string"}},
                "by_pattern": {"type": "object", "patternProperties": {"^x": {"type": "string"}}},
                "mixed": {
                    "type": "object",
                    "properties": {"known": {"type": "integer"}},
                    "additionalProperties": {"type": "integer"}
                }
            }
        });
        let value = json!({
            "labels": {"a": null, "b": "x"},
            "by_pattern": {"x1": null},
            "mixed": {"known": null, "extra": null}
        });
        assert_eq!(
            normalized(&schema, value),
            json!({
                "labels": {"a": null, "b": "x"},
                "by_pattern": {"x1": null},
                // `known` is declared and optional; `extra` is a map entry.
                "mixed": {"extra": null}
            })
        );
        // A map at the root.
        let map = json!({"type": "object", "additionalProperties": {"type": "integer"}});
        assert_eq!(normalized(&map, json!({"k": null})), json!({"k": null}));
    }

    #[test]
    fn required_properties_keep_their_nulls() {
        // `required` counts wherever it applies: the object's own schema, an
        // `allOf` member, a `$ref` target, or the matching variant.
        let schema = json!({
            "type": "object",
            "properties": {
                "own": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "nick": {"type": "string"}},
                    "required": ["name"]
                },
                "via_all_of": {
                    "allOf": [
                        {"properties": {"name": {"type": "string"}, "nick": {"type": "string"}}},
                        {"required": ["name"]}
                    ]
                },
                "via_ref": {"$ref": "#/$defs/Named", "properties": {"nick": {"type": "string"}}},
                "via_variant": {"anyOf": [
                    {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
                    {"type": "string"}
                ]}
            },
            "$defs": {
                "Named": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
            }
        });
        let pair = json!({"name": null, "nick": null});
        let value = json!({
            "own": pair, "via_all_of": pair, "via_ref": pair,
            "via_variant": {"name": null}
        });
        let kept = json!({"name": null});
        assert_eq!(
            normalized(&schema, value),
            json!({"own": kept, "via_all_of": kept, "via_ref": kept, "via_variant": kept})
        );
    }

    #[test]
    fn ambiguous_variants_keep_a_key_one_of_them_does_not_declare() {
        // Both variants fit `{"x": null}`. In the second, `x` is an extra
        // key rather than an optional property, so the null stays.
        let schema = json!({"anyOf": [
            {"type": "object", "properties": {"x": {"type": "string"}}},
            {"type": "object", "properties": {"y": {"type": "string"}}}
        ]});
        assert_eq!(normalized(&schema, json!({"x": null})), json!({"x": null}));
        // Declared and optional in both: removed.
        let schema = json!({"anyOf": [
            {"type": "object", "properties": {"x": {"type": "string"}}},
            {"type": "object", "properties": {"x": {"type": "integer"}, "y": {}}}
        ]});
        assert_eq!(normalized(&schema, json!({"x": null})), json!({}));
        // Required in one: kept.
        let schema = json!({"anyOf": [
            {"type": "object", "properties": {"x": {"type": "string"}}},
            {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
        ]});
        assert_eq!(normalized(&schema, json!({"x": null})), json!({"x": null}));
    }

    #[test]
    fn all_of_forbids_null_when_any_member_does() {
        let schema = json!({
            "type": "object",
            "allOf": [
                {"properties": {"a": {"type": ["string", "null"]}}},
                {"properties": {"a": {"type": "string"}}}
            ]
        });
        assert_eq!(normalized(&schema, json!({"a": null})), json!({}));
    }

    #[test]
    fn recursive_and_cyclic_schemas_terminate() {
        let tree = json!({
            "$ref": "#/$defs/Node",
            "$defs": {"Node": {
                "type": "object",
                "properties": {
                    "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
                    "label": {"type": "string"}
                }
            }}
        });
        assert_eq!(
            normalized(
                &tree,
                json!({"label": null, "children": [{"label": null, "children": [{"children": null}]}]})
            ),
            json!({"children": [{"children": [{}]}]})
        );
        let cycle = json!({
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/A"}},
            "$defs": {"A": {"$ref": "#/$defs/B"}, "B": {"$ref": "#/$defs/A"}}
        });
        let value = json!({"a": null});
        assert_eq!(normalized(&cycle, value.clone()), value);
        assert_eq!(
            normalized(&cycle, json!({"a": {"b": null}})),
            json!({"a": {"b": null}})
        );
    }

    #[test]
    fn normalization_is_idempotent_and_ignores_scalars() {
        let schema = json!({"type": "object", "properties": {"n": {"type": "integer"}}});
        let once = normalized(&schema, json!({"n": null}));
        assert_eq!(normalized(&schema, once.clone()), once);
        assert_eq!(normalized(&schema, json!(null)), json!(null));
        assert_eq!(normalized(&schema, json!("text")), json!("text"));
    }

    #[test]
    fn integer_and_number_types_match_variants() {
        let schema = json!({
            "type": "object",
            "properties": {"v": {"anyOf": [
                {"type": "integer"},
                {"type": "object", "properties": {"x": {"type": "string"}}}
            ]}}
        });
        assert_eq!(normalized(&schema, json!({"v": 3})), json!({"v": 3}));
        assert_eq!(
            normalized(&schema, json!({"v": {"x": null}})),
            json!({"v": {}})
        );
        assert!(type_matches(&json!("integer"), &json!(2.0)));
        assert!(!type_matches(&json!("integer"), &json!(2.5)));
        assert!(type_matches(&json!(["string", "number"]), &json!(2.5)));
    }

    #[cfg(feature = "schema")]
    #[test]
    fn rust_types_keep_their_null_semantics() {
        use serde::Deserialize;

        #[derive(Debug, Default, PartialEq, Deserialize, schemars::JsonSchema)]
        struct Fields {
            #[serde(default)]
            tags: Vec<String>,
            option: Option<String>,
            #[serde(default)]
            double: Option<Option<String>>,
            payload: serde_json::Value,
        }
        let schema = schema_for::<Fields>();
        let raw = r#"{"tags": null, "option": null, "double": null, "payload": null}"#;
        let fields: Fields = from_str_normalized(raw, Some(&schema)).unwrap();
        assert_eq!(fields, Fields::default());
        // `serde_json::Value` keeps its null.
        let mut value: Value = serde_json::from_str(raw).unwrap();
        normalize_strict_nulls(&schema, &mut value);
        assert_eq!(
            value,
            json!({"option": null, "double": null, "payload": null})
        );
        // Without a schema, nothing is normalized.
        assert!(from_str_normalized::<Fields>(raw, None).is_err());
    }

    #[cfg(feature = "schema")]
    #[test]
    fn maps_round_trip_and_real_errors_reach_serde() {
        use std::collections::HashMap;

        use serde::Deserialize;

        #[derive(Debug, PartialEq, Deserialize, schemars::JsonSchema)]
        struct Maps {
            optional: HashMap<String, Option<String>>,
            #[serde(default)]
            plain: HashMap<String, String>,
        }
        let schema = schema_for::<Maps>();
        let maps: Maps =
            from_str_normalized(r#"{"optional": {"a": null, "b": "x"}}"#, Some(&schema)).unwrap();
        assert_eq!(
            maps.optional,
            HashMap::from([
                ("a".to_owned(), None),
                ("b".to_owned(), Some("x".to_owned()))
            ])
        );
        // The entry is not removed, so serde reports it.
        let error =
            from_str_normalized::<Maps>(r#"{"optional": {}, "plain": {"a": null}}"#, Some(&schema))
                .unwrap_err()
                .error;
        assert!(error.to_string().contains("invalid type: null"), "{error}");
    }

    #[cfg(feature = "schema")]
    #[test]
    fn a_required_null_reports_its_type() {
        use serde::Deserialize;

        #[derive(Debug, Deserialize, schemars::JsonSchema)]
        #[allow(dead_code)]
        struct Person {
            name: String,
            #[serde(default)]
            tags: Vec<String>,
        }
        let schema = schema_for::<Person>();
        let failure =
            from_str_normalized::<Person>(r#"{"name": null, "tags": null}"#, Some(&schema))
                .unwrap_err();
        let error = failure.error;
        assert!(
            error.to_string().starts_with("invalid type: null"),
            "{error}"
        );
        assert_eq!(failure.path.as_deref(), Some("name"));
    }

    #[cfg(feature = "schema")]
    #[test]
    fn failure_paths_stop_at_untagged_enums_and_flattened_fields() {
        use serde::Deserialize;

        #[derive(Debug, Deserialize, schemars::JsonSchema)]
        #[allow(dead_code)]
        struct Item {
            price: u32,
        }
        #[derive(Debug, Deserialize, schemars::JsonSchema)]
        #[serde(untagged)]
        #[allow(dead_code)]
        enum Entry {
            Item(Item),
            Note(String),
        }
        #[derive(Debug, Deserialize, schemars::JsonSchema)]
        #[allow(dead_code)]
        struct Extra {
            count: u32,
        }
        #[derive(Debug, Deserialize, schemars::JsonSchema)]
        #[allow(dead_code)]
        struct Order {
            entries: Vec<Entry>,
            items: Vec<Item>,
            #[serde(flatten)]
            extra: Extra,
        }
        let schema = schema_for::<Order>();
        let path = |raw: &str| {
            from_str_normalized::<Order>(raw, Some(&schema))
                .unwrap_err()
                .path
        };
        // A plain field is named down to the value.
        assert_eq!(
            path(r#"{"entries": [], "items": [{"price": "1"}], "count": 1}"#).as_deref(),
            Some("items[0].price")
        );
        // Inside an untagged enum, the path names the enum.
        assert_eq!(
            path(r#"{"entries": [{"price": "1"}], "items": [], "count": 1}"#).as_deref(),
            Some("entries[0]")
        );
        // A flattened field's error is reported at the struct holding it.
        assert_eq!(path(r#"{"entries": [], "items": [], "count": "1"}"#), None);
    }

    #[cfg(feature = "schema")]
    #[test]
    fn internally_tagged_newtype_variants_are_judged_with_their_target() {
        use serde::Deserialize;

        #[derive(Debug, PartialEq, Deserialize, schemars::JsonSchema)]
        struct Inner {
            a: String,
            #[serde(default)]
            b: Vec<String>,
        }
        #[derive(Debug, PartialEq, Deserialize, schemars::JsonSchema)]
        #[serde(tag = "kind")]
        enum Shape {
            New(Inner),
            Other { c: u8 },
        }
        // schemars puts the tag next to a `$ref` to the inner struct.
        let schema = schema_for::<Shape>();
        let mut value = json!({"kind": "New", "a": null, "b": null});
        normalize_strict_nulls(&schema, &mut value);
        assert_eq!(value, json!({"kind": "New", "a": null}));

        let shape: Shape =
            from_str_normalized(r#"{"kind": "New", "a": "x", "b": null}"#, Some(&schema)).unwrap();
        assert_eq!(
            shape,
            Shape::New(Inner {
                a: "x".into(),
                b: Vec::new()
            })
        );
        let error =
            from_str_normalized::<Shape>(r#"{"kind": "New", "a": null, "b": null}"#, Some(&schema))
                .unwrap_err()
                .error;
        assert!(error.to_string().contains("invalid type: null"), "{error}");
    }

    #[cfg(feature = "schema")]
    #[test]
    fn schema_for_generates_object_schema() {
        #[derive(schemars::JsonSchema)]
        #[allow(dead_code)]
        struct Args {
            city: String,
        }
        let schema = schema_for::<Args>();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["city"].is_object());
    }

    /// Regression: a recursive untagged union whose
    /// variants share a field doubled the work per nesting level (depth 18
    /// took over a second, depth 30 about an hour).
    #[test]
    fn recursive_unions_with_shared_fields_normalize_in_bounded_time() {
        // schemars' shape for `#[serde(untagged)] enum Tree {
        //     A { x: Option<Box<Tree>>, a: Option<u8> },
        //     B { x: Option<Box<Tree>>, b: Option<u8> } }`.
        let schema = json!({
            "$ref": "#/$defs/Tree",
            "$defs": {"Tree": {"anyOf": [
                {"type": "object", "properties": {
                    "x": {"anyOf": [{"$ref": "#/$defs/Tree"}, {"type": "null"}]},
                    "a": {"type": ["integer", "null"]}
                }},
                {"type": "object", "properties": {
                    "x": {"anyOf": [{"$ref": "#/$defs/Tree"}, {"type": "null"}]},
                    "b": {"type": ["integer", "null"]}
                }}
            ]}}
        });
        let depth = 60;
        let raw = format!(
            "{}{{\"a\": null}}{}",
            "{\"x\":".repeat(depth),
            "}".repeat(depth)
        );
        let mut value: Value = serde_json::from_str(&raw).unwrap();
        let started = std::time::Instant::now();
        normalize_strict_nulls(&schema, &mut value);
        #[cfg(feature = "extract")]
        assert!(undeclared::find_undeclared_key(&schema, &value).is_none());
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "took {:?}",
            started.elapsed()
        );
    }
}
