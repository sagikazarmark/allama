//! The response schema Gemini receives, and what it may not enforce.
//!
//! Gemini constrains structured output to `GenerationConfig.responseJsonSchema`,
//! which accepts a documented subset of JSON Schema and has no strict mode
//! that rejects the rest. Constructs outside the subset are sent anyway, so
//! [`response_schema`] reports them in one `Approximated` warning rather
//! than claiming the schema was enforced as written. Before that it applies
//! two rewrites that keep the schema's meaning:
//!
//! - a `$ref` with sibling keywords is merged with its target
//!   ([`allama::adapter::schema::merge_sibling_refs`]): the API reference forbids
//!   siblings next to `$ref`. A node that cannot be merged is sent as it is,
//!   and reported;
//! - a string or number `const: x` becomes `enum: [x]`, which accepts
//!   exactly the same values and is in the subset. schemars writes tag
//!   discriminators and documented unit variants as such `const`s.

use std::collections::BTreeSet;

use allama::adapter::schema::{ANNOTATION_KEYWORDS, branches_disjoint_in, merge_sibling_refs};
use serde_json::Value;

/// The keywords `GenerationConfig.responseJsonSchema` supports, from the
/// Gemini API reference (<https://ai.google.dev/api/generate-content>,
/// read 2026-10-03).
const SUPPORTED_KEYWORDS: &[&str] = &[
    "$id",
    "$defs",
    "$ref",
    "$anchor",
    "type",
    "format",
    "title",
    "description",
    "enum",
    "items",
    "prefixItems",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "anyOf",
    "oneOf",
    "properties",
    "additionalProperties",
    "required",
];

/// Keywords never counted although outside the subset: they carry no
/// constraint.
const NEVER_COUNTED: &[&str] = &["$schema"];

/// Keywords whose value maps names to subschemas.
const SCHEMA_MAP_KEYWORDS: &[&str] = &[
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
];

/// Keywords whose value is data, not a subschema.
const DATA_KEYWORDS: &[&str] = &["enum", "const", "default", "examples", "required"];

/// The schema to send as `responseJsonSchema`, and the message of the
/// `Approximated` warning naming every construct Gemini may not enforce,
/// when there is one.
pub(crate) fn response_schema(schema: &Value) -> (Value, Option<String>) {
    let mut constructs = BTreeSet::new();
    let mut out = match merge_sibling_refs(schema) {
        Ok(merged) => {
            for unmerged in merged.unmerged {
                constructs.insert(format!(
                    "a `$ref` with sibling keywords at `{}` (sent unmerged: {})",
                    unmerged.path, unmerged.reason
                ));
            }
            merged.schema
        }
        // A malformed reference: Gemini reports it; nothing to merge.
        Err(_) => schema.clone(),
    };
    scalar_consts_as_enums(&mut out);
    let mut audit = Audit {
        document: &out,
        constructs,
    };
    audit.node(&out);
    audit.recursion(&out, &mut vec!["#".to_owned()], None);
    let constructs = audit.constructs;
    let warning = (!constructs.is_empty()).then(|| {
        format!(
            "Gemini may not enforce constructs outside its documented `responseJsonSchema` subset ({}); the response schema was sent with them",
            constructs.into_iter().collect::<Vec<_>>().join(", ")
        )
    });
    (out, warning)
}

/// Rewrite every string or number `const` as a one-value `enum`.
fn scalar_consts_as_enums(node: &mut Value) {
    match node {
        Value::Object(map) => {
            let scalar = map
                .get("const")
                .is_some_and(|c| c.is_string() || c.is_number());
            if scalar
                && !map.contains_key("enum")
                && let Some(value) = map.remove("const")
            {
                map.insert("enum".into(), Value::Array(vec![value]));
            }
            for (key, value) in map.iter_mut() {
                if DATA_KEYWORDS.contains(&key.as_str()) {
                    continue;
                }
                if SCHEMA_MAP_KEYWORDS.contains(&key.as_str())
                    && let Value::Object(entries) = value
                {
                    entries.values_mut().for_each(scalar_consts_as_enums);
                } else {
                    scalar_consts_as_enums(value);
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(scalar_consts_as_enums),
        _ => {}
    }
}

struct Audit<'a> {
    /// The schema as it will be sent, for `$ref`s.
    document: &'a Value,
    constructs: BTreeSet<String>,
}

impl Audit<'_> {
    /// Count the constructs of `schema` and its subschemas.
    fn node(&mut self, schema: &Value) {
        let map = match schema {
            Value::Object(map) => map,
            Value::Bool(true) => {
                self.constructs.insert(
                    "an unconstrained `true` subschema (such as `serde_json::Value`)".into(),
                );
                return;
            }
            _ => return,
        };
        for (key, value) in map {
            let key = key.as_str();
            if !SUPPORTED_KEYWORDS.contains(&key)
                && !ANNOTATION_KEYWORDS.contains(&key)
                && !NEVER_COUNTED.contains(&key)
            {
                self.constructs.insert(match key {
                    "const" => "`const` with a value that is not a string or number".into(),
                    other => format!("`{other}`"),
                });
            }
            match key {
                "enum" => {
                    if value.as_array().is_some_and(|values| {
                        values.iter().any(|v| !v.is_string() && !v.is_number())
                    }) {
                        self.constructs
                            .insert("`enum` with values that are not strings or numbers".into());
                    }
                }
                "oneOf" => {
                    if let Value::Array(branches) = value
                        && !branches_disjoint_in(self.document, branches)
                    {
                        self.constructs.insert(
                            "`oneOf` with overlapping branches (Gemini reads it as `anyOf`)".into(),
                        );
                    }
                }
                _ => {}
            }
            if DATA_KEYWORDS.contains(&key) {
                continue;
            }
            if SCHEMA_MAP_KEYWORDS.contains(&key) {
                if let Value::Object(entries) = value {
                    entries.values().for_each(|entry| self.node(entry));
                }
            } else if let Value::Array(items) = value {
                items.iter().for_each(|item| self.node(item));
            } else {
                self.node(value);
            }
        }
    }

    /// Count recursion reached through required properties only.
    ///
    /// `stack` holds the references being followed (the root is `#`);
    /// `optional_at` is the stack length at the last step into a property
    /// that is not required (or a map entry) on the path to `schema`.
    fn recursion(&mut self, schema: &Value, stack: &mut Vec<String>, optional_at: Option<usize>) {
        let Value::Object(map) = schema else {
            return;
        };
        if let Some(Value::String(reference)) = map.get("$ref") {
            if let Some(position) = stack.iter().position(|r| r == reference) {
                // The cycle passed through an optional property when one was
                // stepped into after the reference was entered.
                if optional_at.is_none_or(|at| at <= position) {
                    self.constructs.insert(format!(
                        "recursion through required properties (`{reference}`)"
                    ));
                }
                return;
            }
            if let Some(target) = resolve(self.document, reference) {
                stack.push(reference.clone());
                self.recursion(target, stack, optional_at);
                stack.pop();
            }
        }
        let required: Vec<&str> = map
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        for (key, value) in map {
            match key.as_str() {
                "$defs" | "definitions" | "$ref" => {}
                key if DATA_KEYWORDS.contains(&key) => {}
                "properties" => {
                    if let Value::Object(properties) = value {
                        for (name, property) in properties {
                            let optional = if required.contains(&name.as_str()) {
                                optional_at
                            } else {
                                Some(stack.len())
                            };
                            self.recursion(property, stack, optional);
                        }
                    }
                }
                key if SCHEMA_MAP_KEYWORDS.contains(&key) => {
                    // Keys these match are never required by name.
                    if let Value::Object(entries) = value {
                        for entry in entries.values() {
                            self.recursion(entry, stack, Some(stack.len()));
                        }
                    }
                }
                "additionalProperties" => self.recursion(value, stack, Some(stack.len())),
                _ => match value {
                    Value::Array(items) => {
                        for item in items {
                            self.recursion(item, stack, optional_at);
                        }
                    }
                    other => self.recursion(other, stack, optional_at),
                },
            }
        }
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
    use serde_json::json;

    fn audit(schema: Value) -> (Value, Option<String>) {
        response_schema(&schema)
    }

    #[test]
    fn supported_schemas_are_sent_without_a_warning() {
        let schema = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Person",
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Full name", "format": "email"},
                "age": {"type": "integer", "format": "uint8", "minimum": 0},
                "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3, "default": []},
                "kind": {"type": "string", "enum": ["a", "b"]},
                "maybe": {"anyOf": [{"$ref": "#/$defs/Inner"}, {"type": "null"}]}
            },
            "required": ["name"],
            "$defs": {"Inner": {"type": "object", "properties": {}, "additionalProperties": false}}
        });
        let (out, warning) = audit(schema.clone());
        assert_eq!(out, schema);
        assert_eq!(warning, None);
    }

    #[test]
    fn scalar_consts_become_enums() {
        let (out, warning) = audit(json!({
            "type": "object",
            "properties": {
                "kind": {"type": "string", "const": "New"},
                "n": {"const": 3},
                "obj": {"const": {"a": 1}}
            }
        }));
        assert_eq!(
            out["properties"]["kind"],
            json!({"type": "string", "enum": ["New"]})
        );
        assert_eq!(out["properties"]["n"], json!({"enum": [3]}));
        assert_eq!(out["properties"]["obj"], json!({"const": {"a": 1}}));
        let warning = warning.unwrap();
        assert!(warning.contains("`const` with a value"), "{warning}");
    }

    #[test]
    fn constructs_outside_the_subset_are_named_once() {
        let (_, warning) = audit(json!({
            "type": "object",
            "properties": {
                "code": {"type": "string", "pattern": "^[A-Z]+$", "minLength": 1},
                "other": {"type": "string", "pattern": "x"},
                "any": true,
                "flag": {"enum": [true, false]},
                "both": {"allOf": [{"type": "string"}, {"minLength": 2}]}
            }
        }));
        let warning = warning.unwrap();
        for needle in [
            "`pattern`",
            "`minLength`",
            "`allOf`",
            "unconstrained `true`",
            "`enum` with values",
        ] {
            assert!(warning.contains(needle), "{needle}: {warning}");
        }
        assert_eq!(warning.matches("`pattern`").count(), 1, "{warning}");
    }

    #[test]
    fn one_of_counts_only_when_branches_overlap() {
        let (out, warning) = audit(json!({
            "oneOf": [{"type": "string", "const": "A"}, {"type": "string", "const": "B"}]
        }));
        assert_eq!(warning, None);
        assert_eq!(out["oneOf"][0], json!({"type": "string", "enum": ["A"]}));
        let (_, warning) = audit(json!({"oneOf": [{"type": "integer"}, {"type": "number"}]}));
        assert!(warning.unwrap().contains("`oneOf`"));
    }

    #[test]
    fn recursion_counts_only_through_required_properties() {
        // Optional: allowed.
        let (_, warning) = audit(json!({
            "$ref": "#/$defs/Node",
            "$defs": {"Node": {"type": "object", "properties": {
                "next": {"$ref": "#/$defs/Node"},
                "label": {"type": "string"}
            }, "required": ["label"]}}
        }));
        assert_eq!(warning, None);
        // Required at every step: counted.
        let (_, warning) = audit(json!({
            "type": "object",
            "properties": {"head": {"$ref": "#/$defs/Node"}},
            "$defs": {"Node": {"type": "object", "properties": {
                "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}
            }, "required": ["children"]}}
        }));
        assert!(warning.unwrap().contains("recursion"));
        // The root itself.
        let (_, warning) = audit(json!({
            "type": "object",
            "properties": {"next": {"$ref": "#"}},
            "required": ["next"]
        }));
        assert!(warning.unwrap().contains("recursion"));
    }

    #[test]
    fn sibling_refs_are_merged_or_reported() {
        let (out, warning) = audit(json!({
            "type": "object",
            "properties": {"inner": {"$ref": "#/$defs/Inner", "description": "doc"}},
            "$defs": {"Inner": {"type": "object", "properties": {"a": {"type": "integer"}}}}
        }));
        assert_eq!(warning, None);
        assert_eq!(
            out["properties"]["inner"],
            json!({"type": "object", "description": "doc", "properties": {"a": {"type": "integer"}}})
        );
        let schema = json!({
            "type": "object",
            "properties": {"v": {"$ref": "#/$defs/S", "type": "integer"}},
            "$defs": {"S": {"type": "string"}}
        });
        let (out, warning) = audit(schema.clone());
        assert_eq!(out, schema, "sent unmerged");
        let warning = warning.unwrap();
        assert!(warning.contains("/properties/v"), "{warning}");
    }
}
