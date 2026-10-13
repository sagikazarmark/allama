//! Union branch disjointness: [`branches_disjoint`].

use serde_json::{Map, Value};

use super::resolve;

/// Whether no value can match two of `branches`, so `oneOf` over them
/// accepts exactly what `anyOf` accepts.
///
/// A dialect whose grammar has `anyOf` but no `oneOf` can then rewrite one
/// into the other without changing the schema's meaning. The check is
/// conservative: `true` means proven disjoint, `false` means "could not
/// prove it". Two branches are disjoint when any of these holds:
///
/// 1. their JSON types do not overlap. Types come from `type` (every member
///    of a `type` array counts, including `null`; `integer` overlaps
///    `number`), or, when `type` is absent, from the values of `const` or
///    `enum`;
/// 2. their `const`/`enum` value sets do not intersect;
/// 3. a property both require has disjoint schemas by rule 1 or 2, such as
///    the tag of an internally or adjacently tagged enum;
/// 4. one is an object that requires a key the other, a closed object
///    (`additionalProperties: false`), does not declare, as in an
///    externally tagged enum.
///
/// Run it after `$ref` inlining ([`inline_refs`](super::inline_refs)) and
/// after [`merge_sibling_refs`](super::merge_sibling_refs): a branch, or a
/// property the rules read, that is still a `$ref` proves nothing. A dialect
/// that keeps references (for recursion) uses [`branches_disjoint_in`].
///
/// ```
/// use allama::adapter::schema::branches_disjoint;
/// use serde_json::json;
///
/// // A documented unit enum, as schemars writes it.
/// assert!(branches_disjoint(&[
///     json!({"type": "string", "const": "A", "description": "first"}),
///     json!({"type": "string", "const": "B", "description": "second"}),
/// ]));
/// // `2` matches both.
/// assert!(!branches_disjoint(&[json!({"type": "integer"}), json!({"type": "number"})]));
/// ```
#[must_use]
pub fn branches_disjoint(branches: &[Value]) -> bool {
    branches_disjoint_in(&Value::Null, branches)
}

/// [`branches_disjoint`] for branches that may still hold local `$ref`s,
/// resolved against `document` (the schema the branches belong to) where
/// the rules read: a branch itself and the properties rule 3 compares.
#[must_use]
pub fn branches_disjoint_in(document: &Value, branches: &[Value]) -> bool {
    let branches: Vec<&Value> = branches.iter().map(|b| deref(document, b)).collect();
    branches.iter().enumerate().all(|(index, a)| {
        branches[index + 1..]
            .iter()
            .all(|b| pair_disjoint(document, a, b))
    })
}

fn pair_disjoint(document: &Value, a: &Value, b: &Value) -> bool {
    if matches!(a, Value::Bool(false)) || matches!(b, Value::Bool(false)) {
        return true;
    }
    let (Value::Object(a), Value::Object(b)) = (a, b) else {
        return false;
    };
    values_disjoint(a, b)
        || types_disjoint(a, b)
        || required_property_disjoint(document, a, b)
        || closed_excludes(a, b)
        || closed_excludes(b, a)
}

/// Rule 1.
fn types_disjoint(a: &Map<String, Value>, b: &Map<String, Value>) -> bool {
    let (Some(a), Some(b)) = (types(a), types(b)) else {
        return false;
    };
    !a.iter().any(|x| b.iter().any(|y| types_overlap(x, y)))
}

/// Rule 2.
fn values_disjoint(a: &Map<String, Value>, b: &Map<String, Value>) -> bool {
    let (Some(a), Some(b)) = (values(a), values(b)) else {
        return false;
    };
    !a.iter().any(|x| b.iter().any(|y| json_eq(x, y)))
}

/// Rule 3.
fn required_property_disjoint(
    document: &Value,
    a: &Map<String, Value>,
    b: &Map<String, Value>,
) -> bool {
    let required_b: Vec<&str> = required(b).collect();
    required(a)
        .filter(|key| required_b.contains(key))
        .any(|key| {
            let (Some(pa), Some(pb)) = (property(document, a, key), property(document, b, key))
            else {
                return false;
            };
            match (pa, pb) {
                (Value::Object(pa), Value::Object(pb)) => {
                    values_disjoint(pa, pb) || types_disjoint(pa, pb)
                }
                (Value::Bool(false), _) | (_, Value::Bool(false)) => true,
                _ => false,
            }
        })
}

/// Rule 4: `a` admits only objects and requires a key that `closed` forbids.
fn closed_excludes(a: &Map<String, Value>, closed: &Map<String, Value>) -> bool {
    let object_only = a.get("type").and_then(Value::as_str) == Some("object");
    let is_closed = closed.get("additionalProperties") == Some(&Value::Bool(false))
        && !closed.contains_key("patternProperties");
    if !object_only || !is_closed {
        return false;
    }
    let declared = closed.get("properties").and_then(Value::as_object);
    required(a).any(|key| !declared.is_some_and(|d| d.contains_key(key)))
}

/// The JSON types a schema admits, when it says.
fn types(schema: &Map<String, Value>) -> Option<Vec<&str>> {
    match schema.get("type") {
        Some(Value::String(name)) => return Some(vec![name.as_str()]),
        Some(Value::Array(names)) => return names.iter().map(Value::as_str).collect(),
        Some(_) => return None,
        None => {}
    }
    values(schema).map(|values| values.iter().map(|v| value_type(v)).collect())
}

/// The values a schema admits, when it lists them (`const` or `enum`).
fn values(schema: &Map<String, Value>) -> Option<Vec<&Value>> {
    if let Some(value) = schema.get("const") {
        return Some(vec![value]);
    }
    match schema.get("enum") {
        Some(Value::Array(values)) => Some(values.iter().collect()),
        _ => None,
    }
}

fn value_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) if n.is_i64() || n.is_u64() => "integer",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn types_overlap(a: &str, b: &str) -> bool {
    a == b || matches!((a, b), ("integer", "number") | ("number", "integer"))
}

/// JSON Schema equality: numbers compare by value (`1` equals `1.0`).
fn json_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => match (x.as_i64(), y.as_i64()) {
            (Some(x), Some(y)) => x == y,
            _ => match (x.as_u64(), y.as_u64()) {
                (Some(x), Some(y)) => x == y,
                _ => x.as_f64() == y.as_f64(),
            },
        },
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(x, y)| json_eq(x, y))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter()
                    .all(|(key, value)| y.get(key).is_some_and(|other| json_eq(value, other)))
        }
        _ => a == b,
    }
}

fn required(schema: &Map<String, Value>) -> impl Iterator<Item = &str> {
    schema
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
}

fn property<'a>(
    document: &'a Value,
    schema: &'a Map<String, Value>,
    key: &str,
) -> Option<&'a Value> {
    schema
        .get("properties")
        .and_then(|p| p.get(key))
        .map(|p| deref(document, p))
}

/// `schema`, or the target of its bare `$ref` in `document`.
fn deref<'a>(document: &'a Value, schema: &'a Value) -> &'a Value {
    let mut current = schema;
    // A bound against reference cycles that never reach a schema.
    for _ in 0..32 {
        let Value::Object(map) = current else {
            return current;
        };
        match map.get("$ref") {
            Some(Value::String(reference)) if map.len() == 1 => {
                match resolve(document, reference) {
                    Some(target) => current = target,
                    None => return current,
                }
            }
            _ => return current,
        }
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn types_tell_branches_apart() {
        assert!(branches_disjoint(&[
            json!({"type": "string"}),
            json!({"type": "object"}),
            json!({"type": ["array", "null"]}),
        ]));
        assert!(!branches_disjoint(&[
            json!({"type": "integer"}),
            json!({"type": "number"}),
        ]));
        assert!(!branches_disjoint(&[
            json!({"type": ["string", "null"]}),
            json!({"type": "null"}),
        ]));
        // No `type`: unknown, so not proven.
        assert!(!branches_disjoint(&[
            json!({"properties": {}}),
            json!({"type": "string"})
        ]));
        // Types from values.
        assert!(branches_disjoint(&[
            json!({"const": 1}),
            json!({"type": "string"})
        ]));
    }

    #[test]
    fn const_and_enum_sets_tell_branches_apart() {
        assert!(branches_disjoint(&[
            json!({"type": "string", "enum": ["Unit"]}),
            json!({"type": "string", "const": "Other", "description": "documented"}),
        ]));
        assert!(!branches_disjoint(&[
            json!({"enum": ["a", "b"]}),
            json!({"enum": ["b", "c"]}),
        ]));
        // `1` and `1.0` are the same JSON Schema value.
        assert!(!branches_disjoint(&[
            json!({"const": 1}),
            json!({"const": 1.0})
        ]));
    }

    #[test]
    fn required_tags_tell_branches_apart() {
        let tagged = |tag: &str| {
            json!({
                "type": "object",
                "properties": {"kind": {"type": "string", "const": tag}, "x": {"type": "integer"}},
                "required": ["kind"]
            })
        };
        assert!(branches_disjoint(&[tagged("A"), tagged("B"), tagged("C")]));
        assert!(!branches_disjoint(&[tagged("A"), tagged("A")]));
        // The tag must be required by both: an optional tag may be absent.
        let mut optional = tagged("B");
        optional["required"] = json!([]);
        assert!(!branches_disjoint(&[tagged("A"), optional]));
    }

    #[test]
    fn closed_objects_tell_branches_apart() {
        let variant = |key: &str| {
            json!({
                "type": "object",
                "properties": {key: {"type": "object"}},
                "required": [key],
                "additionalProperties": false
            })
        };
        assert!(branches_disjoint(&[variant("New"), variant("Struct")]));
        // An open object may carry the other's key.
        let mut open = variant("Struct");
        open.as_object_mut().unwrap().remove("additionalProperties");
        let mut open_too = variant("New");
        open_too
            .as_object_mut()
            .unwrap()
            .remove("additionalProperties");
        assert!(!branches_disjoint(&[open_too, open]));
        // `patternProperties` may admit the key.
        let mut patterned = variant("Struct");
        patterned["patternProperties"] = json!({"^N": {}});
        let mut requires_new = variant("New");
        requires_new
            .as_object_mut()
            .unwrap()
            .remove("additionalProperties");
        assert!(!branches_disjoint(&[requires_new, patterned]));
    }

    #[test]
    fn boolean_branches() {
        assert!(branches_disjoint(&[json!(false), json!(true)]));
        assert!(!branches_disjoint(&[
            json!(true),
            json!({"type": "string"})
        ]));
        assert!(branches_disjoint(&[json!({"type": "string"})]));
        assert!(branches_disjoint(&[]));
    }

    #[test]
    fn refs_are_resolved_against_the_document() {
        let document = json!({"$defs": {
            "A": {"type": "object", "properties": {"t": {"$ref": "#/$defs/TagA"}}, "required": ["t"]},
            "B": {"type": "object", "properties": {"t": {"const": "b"}}, "required": ["t"]},
            "TagA": {"const": "a"}
        }});
        let branches = [json!({"$ref": "#/$defs/A"}), json!({"$ref": "#/$defs/B"})];
        assert!(branches_disjoint_in(&document, &branches));
        // Without the document, references prove nothing.
        assert!(!branches_disjoint(&branches));
    }
}
