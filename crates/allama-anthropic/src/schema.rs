//! Anthropic's strict JSON Schema dialect, for structured outputs
//! (`output_config.format`) and strict tools.
//!
//! [`strict_schema`] rewrites a portable schema into what Anthropic's grammar
//! accepts, following the JSON Schema limitations in Anthropic's structured
//! outputs guide:
//!
//! - every object is closed (`additionalProperties: false`); an object that
//!   was open (`additionalProperties: true`) is closed with an
//!   `Approximated` warning;
//! - `$ref`/`$defs`, `anyOf`, `enum` and `const` are kept. A `$ref` with
//!   sibling keywords is merged with its target
//!   ([`allama::adapter::schema::merge_sibling_refs`]): Anthropic rejects `allOf`
//!   beside a `$ref`, and closing both halves separately would make them
//!   contradict. A `oneOf` becomes an `anyOf`, silently when its branches are
//!   disjoint ([`allama::adapter::schema::branches_disjoint`]);
//! - numeric and string-length constraints, `uniqueItems`, `minItems` above
//!   1 and the other unsupported keywords are removed with an
//!   `Approximated` warning;
//! - `format` on an `integer` or `number` (schemars' `uint32`, `double`, …)
//!   is an annotation and is dropped silently. A string format outside
//!   `date-time`, `time`, `date`, `duration`, `email`, `hostname`, `uri`,
//!   `ipv4`, `ipv6` and `uuid` is removed with an `Approximated` warning.
//!   This matches `transform_schema` in Anthropic's Python SDK (v1.11.0,
//!   2026-09-30), which keeps only those string formats;
//! - recursion and maps (`additionalProperties` with a schema) are not
//!   representable ([`StrictSchemaError::NotRepresentable`]), and a `$ref`
//!   that does not resolve is malformed ([`StrictSchemaError::Invalid`]).
//!
//! What to send then is decided by
//! [`allama::adapter::schema::apply_strict_policy`], with this function as its
//! dialect. A strict tool is [`Degradable::Yes`](allama::adapter::schema::Degradable::Yes):
//! it goes out non-strict under `UnsupportedPolicy::Warn`. `output_config.format`
//! has no unconstrained mode, so it is
//! [`Degradable::NoUnconstrainedForm`](allama::adapter::schema::Degradable::NoUnconstrainedForm)
//! and fails under every policy.
//!
//! Anthropic also limits the strict schemas of one request as a whole (at
//! most 20 strict tools, 24 optional parameters and 16 union-typed
//! parameters across the strict tools and the output format). The provider
//! counts those per request, not here, before it applies the policy: it
//! degrades strict tools from the last one until the request fits.
//!
//! The module depends only on `allama` and `serde_json`, so another adapter
//! for Anthropic's Messages API (on Vertex AI, say) can reuse it.
//!
//! ```
//! use allama::adapter::schema::StrictSchemaError;
//! use allama::adapter::Degradations;
//! use allama::{RequestField, UnsupportedPolicy};
//! use allama_anthropic::schema::strict_schema;
//! use serde_json::json;
//!
//! let mut degradations = Degradations::new("anthropic", UnsupportedPolicy::Warn);
//! let person = json!({
//!     "type": "object",
//!     "properties": {"name": {"type": "string"}, "age": {"type": "integer", "format": "uint8"}},
//!     "required": ["name"]
//! });
//! let strict = strict_schema(&person, RequestField::ResponseFormat, &mut degradations)?;
//! assert_eq!(strict["additionalProperties"], false);
//! assert_eq!(strict["properties"]["age"], json!({"type": "integer"}));
//! assert!(degradations.warnings().is_empty());
//! # Ok::<(), StrictSchemaError>(())
//! ```

use std::collections::BTreeSet;

use allama::adapter::Degradations;
use allama::adapter::schema::StrictSchemaError;
use allama::{RequestField, Subject};
use serde_json::{Map, Value};

/// Rewrite `schema` into Anthropic's strict form.
///
/// Keywords Anthropic cannot keep are removed or rewritten and reported
/// through `degradations` as `Approximated` warnings on `field`; warnings
/// are only recorded when the rewrite succeeds, as
/// [`apply_strict_policy`](allama::adapter::schema::apply_strict_policy) requires of
/// a dialect. The input is not modified, so on
/// [`StrictSchemaError::NotRepresentable`] the caller still holds the
/// original schema to send without strict.
///
/// The per-request limits (strict tools, optional and union-typed
/// parameters) are not checked here; they depend on every strict schema in
/// the request.
///
/// # Errors
///
/// - [`StrictSchemaError::NotRepresentable`] when the schema uses a
///   construct Anthropic strict mode cannot express (recursion, a map, a
///   `$ref` that conflicts with its siblings).
/// - [`StrictSchemaError::Invalid`] when a `$ref` beside sibling keywords
///   does not resolve.
pub fn strict_schema(
    schema: &Value,
    field: RequestField,
    degradations: &mut Degradations,
) -> Result<Value, StrictSchemaError> {
    if allama::adapter::schema::is_recursive(schema) {
        return Err(StrictSchemaError::not_representable(
            "recursive JSON schemas are not supported by Anthropic structured outputs",
        ));
    }
    let mut out = schema.clone();
    if let Value::Object(root) = &mut out {
        // Annotation only; carries no output semantics.
        root.remove("$schema");
    }
    // Before any rewrite, so the targets merged are the caller's.
    let mut out = allama::adapter::schema::merge_sibling_refs(&out)?.into_strict()?;
    let document = out.clone();
    let mut removed: BTreeSet<String> =
        allama::adapter::schema::strip_keywords(&mut out, UNSUPPORTED_KEYWORDS);
    let mut rewrite = Rewrite {
        document: &document,
        removed: &mut removed,
        formats: BTreeSet::new(),
        opened: false,
        error: None,
    };
    rewrite.schema(&mut out);
    if let Some(message) = rewrite.error {
        return Err(StrictSchemaError::not_representable(message));
    }
    let opened = rewrite.opened;
    let formats = rewrite.formats;
    let subject = Subject::Field { field };
    if !removed.is_empty() {
        let list: Vec<&str> = removed.iter().map(String::as_str).collect();
        degradations.approximated(
            subject.clone(),
            format!(
                "Anthropic does not support the JSON Schema keywords {}; the schema was sent without them",
                list.join(", ")
            ),
        );
    }
    if !formats.is_empty() {
        let list: Vec<&str> = formats.iter().map(String::as_str).collect();
        degradations.approximated(
            subject.clone(),
            format!(
                "Anthropic does not support the string formats {}; the schema was sent without them",
                list.join(", ")
            ),
        );
    }
    if opened {
        degradations.approximated(
            subject,
            "Anthropic requires closed objects; objects with `additionalProperties: true` were sent as `additionalProperties: false`",
        );
    }
    Ok(out)
}

/// At most this many tools with `strict: true` per request.
pub(crate) const MAX_STRICT_TOOLS: usize = 20;

/// At most this many optional parameters across a request's strict schemas.
pub(crate) const MAX_OPTIONAL_PARAMETERS: usize = 24;

/// At most this many union-typed parameters (`anyOf` or a `type` array)
/// across a request's strict schemas.
pub(crate) const MAX_UNION_PARAMETERS: usize = 16;

/// What one strict schema adds to the per-request limits.
///
/// Every object property counts, nested ones included, which is
/// conservative where Anthropic counts only top-level parameters. A `$ref`
/// counts its target at each use, as the merged schema does for a `$ref`
/// with siblings: that is what the grammar compiles. Definitions are not
/// counted on their own.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Complexity {
    /// Properties not listed in their object's `required`.
    pub(crate) optional: usize,
    /// Properties whose schema is a union: `anyOf`, `oneOf`, or a `type`
    /// array with more than one type.
    pub(crate) unions: usize,
}

impl Complexity {
    /// Count `schema`, a strict schema (no recursion).
    pub(crate) fn of(schema: &Value) -> Self {
        let mut complexity = Self::default();
        complexity.walk(schema, schema, 0);
        complexity
    }

    /// The limits `self` exceeds as the whole request, with `strict_tools`
    /// strict tools, or `None` when it fits.
    pub(crate) fn overrun(self, strict_tools: usize) -> Option<String> {
        let mut over = Vec::new();
        if strict_tools > MAX_STRICT_TOOLS {
            over.push(format!(
                "{strict_tools} strict tools (at most {MAX_STRICT_TOOLS})"
            ));
        }
        if self.optional > MAX_OPTIONAL_PARAMETERS {
            over.push(format!(
                "{} optional parameters (at most {MAX_OPTIONAL_PARAMETERS})",
                self.optional
            ));
        }
        if self.unions > MAX_UNION_PARAMETERS {
            over.push(format!(
                "{} union-typed parameters (at most {MAX_UNION_PARAMETERS})",
                self.unions
            ));
        }
        (!over.is_empty()).then(|| over.join(", "))
    }

    fn walk(&mut self, node: &Value, document: &Value, depth: usize) {
        // A guard; strict schemas are not recursive.
        if depth > MAX_DEPTH {
            return;
        }
        match node {
            Value::Object(map) => {
                if let Some(Value::String(reference)) = map.get("$ref")
                    && let Some(target) = resolve(document, reference)
                {
                    self.walk(target, document, depth + 1);
                }
                if let Some(Value::Object(properties)) = map.get("properties") {
                    let required: BTreeSet<&str> = map
                        .get("required")
                        .and_then(Value::as_array)
                        .map(|r| r.iter().filter_map(Value::as_str).collect())
                        .unwrap_or_default();
                    for (name, property) in properties {
                        if !required.contains(name.as_str()) {
                            self.optional += 1;
                        }
                        if is_union(property, document, depth) {
                            self.unions += 1;
                        }
                        self.walk(property, document, depth + 1);
                    }
                }
                for (key, value) in map {
                    if !matches!(
                        key.as_str(),
                        "$ref" | "properties" | "$defs" | "definitions"
                    ) && !DATA_KEYWORDS.contains(&key.as_str())
                    {
                        self.walk(value, document, depth + 1);
                    }
                }
            }
            Value::Array(items) => {
                for item in items {
                    self.walk(item, document, depth + 1);
                }
            }
            _ => {}
        }
    }
}

impl std::ops::Add for Complexity {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            optional: self.optional + other.optional,
            unions: self.unions + other.unions,
        }
    }
}

impl std::ops::Sub for Complexity {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            optional: self.optional - other.optional,
            unions: self.unions - other.unions,
        }
    }
}

/// Keywords Anthropic rejects in strict schemas.
const UNSUPPORTED_KEYWORDS: &[&str] = &[
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
    "minContains",
    "maxContains",
];

/// String formats Anthropic strict mode supports (structured outputs guide;
/// `SupportedStringFormats` in the Python SDK's `transform_schema`, v1.11.0).
const SUPPORTED_FORMATS: &[&str] = &[
    "date-time",
    "time",
    "date",
    "duration",
    "email",
    "hostname",
    "uri",
    "ipv4",
    "ipv6",
    "uuid",
];

/// Keywords whose values are data, not subschemas.
const DATA_KEYWORDS: &[&str] = &["enum", "const", "default", "examples", "required"];

/// Keywords whose values are maps of name to subschema.
const SCHEMA_MAP_KEYWORDS: &[&str] = &["properties", "patternProperties", "$defs", "definitions"];

/// How deep [`Complexity`] follows a schema.
const MAX_DEPTH: usize = 64;

struct Rewrite<'a> {
    /// The document `$ref`s resolve against: the merged schema, before any
    /// keyword was removed or object closed.
    document: &'a Value,
    removed: &'a mut BTreeSet<String>,
    /// String formats removed.
    formats: BTreeSet<String>,
    opened: bool,
    error: Option<String>,
}

impl Rewrite<'_> {
    fn schema(&mut self, node: &mut Value) {
        match node {
            Value::Object(map) => {
                self.object(map);
                for (key, value) in map.iter_mut() {
                    if DATA_KEYWORDS.contains(&key.as_str()) {
                        continue;
                    }
                    if SCHEMA_MAP_KEYWORDS.contains(&key.as_str()) {
                        if let Value::Object(entries) = value {
                            entries.values_mut().for_each(|v| self.schema(v));
                        }
                    } else {
                        self.schema(value);
                    }
                }
            }
            Value::Array(items) => items.iter_mut().for_each(|v| self.schema(v)),
            _ => {}
        }
    }

    fn object(&mut self, map: &mut Map<String, Value>) {
        if let Some(one_of) = map.remove("oneOf") {
            // `anyOf` admits the same values, plus values that match several
            // branches; with disjoint branches there are none.
            let disjoint = one_of.as_array().is_some_and(|branches| {
                allama::adapter::schema::branches_disjoint_in(self.document, branches)
            });
            if !disjoint {
                self.removed.insert("oneOf (rewritten as anyOf)".into());
            }
            map.insert("anyOf".into(), one_of);
        }
        self.format(map);
        if map
            .get("minItems")
            .and_then(Value::as_u64)
            .is_some_and(|n| n > 1)
        {
            map.remove("minItems");
            self.removed.insert("minItems".into());
        }
        let is_object = match map.get("type") {
            Some(Value::String(t)) => t == "object",
            Some(Value::Array(types)) => types.iter().any(|t| t == "object"),
            _ => map.contains_key("properties"),
        };
        if !is_object {
            return;
        }
        match map.get("additionalProperties") {
            None | Some(Value::Bool(false)) => {}
            Some(Value::Bool(true)) => self.opened = true,
            Some(_) => {
                self.error.get_or_insert_with(|| {
                    "map-typed objects (`additionalProperties` with a schema) are not supported by Anthropic structured outputs".into()
                });
                return;
            }
        }
        map.insert("additionalProperties".into(), Value::Bool(false));
    }

    /// Drop a numeric `format` (an annotation), and remove a string format
    /// Anthropic does not support.
    fn format(&mut self, map: &mut Map<String, Value>) {
        let Some(format) = map.get("format") else {
            return;
        };
        if is_numeric(map) {
            map.remove("format");
            return;
        }
        let supported = format
            .as_str()
            .is_some_and(|format| SUPPORTED_FORMATS.contains(&format));
        if !supported {
            let name = match format {
                Value::String(name) => name.clone(),
                other => other.to_string(),
            };
            self.formats.insert(name);
            map.remove("format");
        }
    }
}

/// Whether the schema's `type` is `integer` or `number` (or a `type` array of
/// those, with or without `null`).
fn is_numeric(map: &Map<String, Value>) -> bool {
    let numeric = |t: &Value| t == "integer" || t == "number";
    match map.get("type") {
        Some(t @ Value::String(_)) => numeric(t),
        Some(Value::Array(types)) => {
            types.iter().any(numeric) && types.iter().all(|t| numeric(t) || t == "null")
        }
        _ => false,
    }
}

/// Whether `schema` (through its `$ref`s) is a union.
fn is_union(schema: &Value, document: &Value, depth: usize) -> bool {
    let Value::Object(map) = schema else {
        return false;
    };
    if depth > MAX_DEPTH {
        return false;
    }
    if map.contains_key("anyOf") || map.contains_key("oneOf") {
        return true;
    }
    if let Some(Value::Array(types)) = map.get("type")
        && types.len() > 1
    {
        return true;
    }
    match map.get("$ref") {
        Some(Value::String(reference)) => {
            resolve(document, reference).is_some_and(|target| is_union(target, document, depth + 1))
        }
        _ => false,
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

#[cfg(test)]
mod tests {
    use super::*;
    use allama::{UnsupportedPolicy, WarningKind};
    use serde_json::json;

    fn run(schema: Value) -> (Result<Value, StrictSchemaError>, Vec<allama::Warning>) {
        let mut d = Degradations::new("anthropic", UnsupportedPolicy::Warn);
        let out = strict_schema(&schema, RequestField::ResponseFormat, &mut d);
        (out, d.into_warnings())
    }

    #[test]
    fn closes_objects_and_keeps_refs() {
        let (out, warnings) = run(json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Person",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {"$ref": "#/$defs/Address"}
            },
            "required": ["name"],
            "$defs": {
                "Address": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        }));
        let out = out.unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(out["additionalProperties"], false);
        assert_eq!(out["$defs"]["Address"]["additionalProperties"], false);
        assert_eq!(out["properties"]["address"]["$ref"], "#/$defs/Address");
        assert!(out.get("$schema").is_none());
        assert_eq!(out["title"], "Person");
        // Optional properties stay optional (unlike OpenAI strict mode).
        assert_eq!(out["required"], json!(["name"]));
    }

    #[test]
    fn strips_constraints_with_warning() {
        let (out, warnings) = run(json!({
            "type": "object",
            "properties": {
                "age": {"type": "integer", "minimum": 0, "maximum": 150},
                "tags": {"type": "array", "items": {"type": "string", "maxLength": 3}, "minItems": 2},
                "kind": {"oneOf": [{"const": "a"}, {"const": "b"}]},
                "minimum": {"type": "string"}
            }
        }));
        let out = out.unwrap();
        assert!(out["properties"]["age"].get("minimum").is_none());
        assert!(out["properties"]["tags"].get("minItems").is_none());
        assert!(out["properties"]["kind"].get("anyOf").is_some());
        // A property *named* like a keyword is preserved.
        assert!(out["properties"].get("minimum").is_some());
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].kind, WarningKind::Approximated);
        assert!(
            warnings[0].message.contains("maximum"),
            "{}",
            warnings[0].message
        );
    }

    #[test]
    fn disjoint_one_of_is_rewritten_silently() {
        let (out, warnings) = run(json!({
            "type": "object",
            "properties": {
                "kind": {"oneOf": [{"type": "string", "const": "a"}, {"type": "string", "const": "b"}]},
                "n": {"oneOf": [{"type": "integer"}, {"type": "number"}]}
            }
        }));
        let out = out.unwrap();
        assert!(out["properties"]["kind"].get("anyOf").is_some());
        assert!(out["properties"]["n"].get("anyOf").is_some());
        // Only the overlapping `n` is reported.
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].message.contains("oneOf"), "{}", warnings[0]);
    }

    #[test]
    fn refs_with_siblings_are_merged() {
        let (out, warnings) = run(json!({
            "type": "object",
            "properties": {"shape": {"oneOf": [
                {"type": "object", "properties": {"kind": {"type": "string", "const": "Unit"}}, "required": ["kind"]},
                {"$ref": "#/$defs/Inner", "type": "object",
                 "properties": {"kind": {"type": "string", "const": "New"}}, "required": ["kind"]}
            ]}},
            "$defs": {"Inner": {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}}
        }));
        let out = out.unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(
            out["properties"]["shape"]["anyOf"][1],
            json!({
                "type": "object",
                "properties": {"a": {"type": "integer"}, "kind": {"type": "string", "const": "New"}},
                "required": ["a", "kind"],
                "additionalProperties": false
            })
        );
        assert!(!out.to_string().contains("allOf"));

        let (out, warnings) = run(json!({
            "type": "object",
            "properties": {"v": {"$ref": "#/$defs/S", "type": "integer"}},
            "$defs": {"S": {"type": "string"}}
        }));
        assert!(
            matches!(&out, Err(StrictSchemaError::NotRepresentable { reason: m, .. }) if m.contains("/properties/v")),
            "{out:?}"
        );
        assert!(warnings.is_empty());
    }

    #[test]
    fn an_unresolved_ref_beside_siblings_is_invalid() {
        let (out, warnings) = run(json!({
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/Missing", "title": "A"}}
        }));
        assert_eq!(
            out,
            Err(StrictSchemaError::Invalid(
                allama::adapter::schema::SchemaError::UnresolvedRef("#/$defs/Missing".into())
            ))
        );
        assert!(warnings.is_empty());
    }

    #[test]
    fn data_keywords_are_not_rewritten() {
        let (out, _) = run(json!({
            "type": "object",
            "properties": {"x": {"type": "object", "properties": {}, "default": {"type": "object"}}}
        }));
        let out = out.unwrap();
        assert!(
            out["properties"]["x"]["default"]
                .get("additionalProperties")
                .is_none()
        );
    }

    #[test]
    fn recursive_and_map_schemas_are_unsupported() {
        let (out, _) = run(json!({
            "type": "object",
            "properties": {"children": {"type": "array", "items": {"$ref": "#"}}}
        }));
        assert!(
            matches!(&out, Err(StrictSchemaError::NotRepresentable { reason: m, .. }) if m.contains("recursive")),
            "{out:?}"
        );

        let (out, warnings) = run(json!({
            "type": "object",
            "properties": {"age": {"type": "integer", "minimum": 0}},
            "additionalProperties": {"type": "string"}
        }));
        assert!(
            matches!(&out, Err(StrictSchemaError::NotRepresentable { reason: m, .. }) if m.contains("map-typed")),
            "{out:?}"
        );
        assert!(warnings.is_empty(), "no warnings on failure: {warnings:?}");
    }

    #[test]
    fn numeric_formats_are_dropped_silently() {
        let (out, warnings) = run(json!({
            "type": "object",
            "properties": {
                "count": {"type": "integer", "format": "uint32"},
                "ratio": {"type": "number", "format": "double"},
                "maybe": {"type": ["integer", "null"], "format": "int64"}
            }
        }));
        let out = out.unwrap();
        assert_eq!(out["properties"]["count"], json!({"type": "integer"}));
        assert_eq!(out["properties"]["ratio"], json!({"type": "number"}));
        assert_eq!(
            out["properties"]["maybe"],
            json!({"type": ["integer", "null"]})
        );
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    #[test]
    fn unsupported_string_formats_are_removed_with_a_warning() {
        let (out, warnings) = run(json!({
            "type": "object",
            "properties": {
                "a": {"type": "string", "format": "uri"},
                "b": {"type": "string", "format": "regex"},
                "c": {"type": "string", "format": "uri-reference"},
                "d": {"type": ["string", "null"], "format": "uuid"},
                "e": {"type": "string", "format": "date-time"}
            }
        }));
        let out = out.unwrap();
        assert_eq!(out["properties"]["a"]["format"], "uri");
        assert_eq!(out["properties"]["d"]["format"], "uuid");
        assert_eq!(out["properties"]["e"]["format"], "date-time");
        assert!(out["properties"]["b"].get("format").is_none());
        assert!(out["properties"]["c"].get("format").is_none());
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert_eq!(warnings[0].kind, WarningKind::Approximated);
        assert!(
            warnings[0]
                .message
                .contains("regex, uri-reference; the schema was sent without them"),
            "{}",
            warnings[0]
        );
    }

    #[test]
    fn complexity_counts_optional_and_union_parameters() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "nick": {"type": ["string", "null"]},
                "kind": {"anyOf": [{"const": "a"}, {"const": "b"}]},
                "single": {"type": ["string"]},
                "nested": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}, "y": {"type": ["number", "null"]}},
                    "required": ["x"]
                },
                "list": {"type": "array", "items": {"$ref": "#/$defs/Item"}}
            },
            "required": ["name", "kind", "list"],
            "$defs": {"Item": {
                "type": "object",
                "properties": {"z": {"$ref": "#/$defs/Choice"}}
            }, "Choice": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
               "Unused": {"type": "object", "properties": {"u": {"type": "string"}}}}
        });
        // Optional: nick, single, nested, y, z. Unions: nick, kind, y, z
        // (through its `$ref`). `Unused` is not counted.
        assert_eq!(
            Complexity::of(&schema),
            Complexity {
                optional: 5,
                unions: 4
            }
        );
    }

    #[test]
    fn complexity_counts_a_reference_at_each_use() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a": {"$ref": "#/$defs/Pair"},
                "b": {"$ref": "#/$defs/Pair"}
            },
            "required": ["a", "b"],
            "$defs": {"Pair": {
                "type": "object",
                "properties": {"l": {"type": ["string", "null"]}, "r": {"type": ["string", "null"]}}
            }}
        });
        assert_eq!(
            Complexity::of(&schema),
            Complexity {
                optional: 4,
                unions: 4
            }
        );
    }

    #[test]
    fn overrun_names_every_limit_exceeded() {
        let fits = Complexity {
            optional: 24,
            unions: 16,
        };
        assert_eq!(fits.overrun(20), None);
        let over = Complexity {
            optional: 25,
            unions: 17,
        };
        let message = over.overrun(21).unwrap();
        assert!(
            message.contains("21 strict tools (at most 20)"),
            "{message}"
        );
        assert!(
            message.contains("25 optional parameters (at most 24)"),
            "{message}"
        );
        assert!(
            message.contains("17 union-typed parameters (at most 16)"),
            "{message}"
        );
    }

    #[test]
    fn open_objects_are_closed_with_warning() {
        let (out, warnings) = run(json!({"type": "object", "additionalProperties": true}));
        assert_eq!(out.unwrap()["additionalProperties"], false);
        assert_eq!(warnings.len(), 1);
    }
}
