//! Undeclared keys: object keys a value carries that the target type's
//! schema does not declare, for extraction ([`find_undeclared_key`]).

use std::collections::BTreeSet;

use serde_json::Value;

use super::{Applicable, Nulls};

/// A key the schema does not declare, at the first object that has one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UndeclaredKey {
    /// The key.
    pub(crate) key: String,
    /// The path to it, such as `contact.e_mail` or `items[2].colour`.
    pub(crate) path: String,
    /// The keys the schemas that apply there declare, sorted.
    pub(crate) declared: Vec<String>,
}

/// The first key of `value` (in document order, outer objects first) that
/// `original_schema` does not declare.
///
/// The walk is the one [`normalize_strict_nulls`](super::normalize_strict_nulls)
/// uses (`$ref`, `allOf`, and the `anyOf`/`oneOf` branches the value may
/// match), with the union rule inverted: a key is undeclared only when no
/// schema that may apply declares it. Where the walk cannot pick one
/// branch, the key is checked against all the branches that may match (all
/// of them when none does), never skipped.
///
/// Absent `additionalProperties` counts as `false` here, which is stricter
/// than JSON Schema. A key is never undeclared where the schema opens the
/// object: `additionalProperties` other than `false` (maps, and structs
/// with a `#[serde(flatten)]` map), `patternProperties`, or a schema that
/// says nothing about keys (`true`, `{}`, `{"type": "object"}`, a
/// `serde_json::Value`).
pub(crate) fn find_undeclared_key(original_schema: &Value, value: &Value) -> Option<UndeclaredKey> {
    let nulls = Nulls::new(original_schema);
    let applicable = nulls.expand(original_schema, value, 0);
    find(&nulls, &applicable, value, "")
}

fn find<'s>(
    nulls: &Nulls<'s>,
    applicable: &Applicable<'s>,
    value: &Value,
    path: &str,
) -> Option<UndeclaredKey> {
    match value {
        Value::Object(map) => {
            for key in map.keys() {
                if status(applicable, key) == Status::Undeclared {
                    let mut declared = BTreeSet::new();
                    declared_keys(applicable, &mut declared);
                    return Some(UndeclaredKey {
                        key: key.clone(),
                        path: join(path, key),
                        declared: declared.into_iter().map(str::to_owned).collect(),
                    });
                }
            }
            map.iter()
                .filter(|(_, child)| child.is_object() || child.is_array())
                .find_map(|(key, child)| {
                    let expanded = nulls.expand_all(&applicable.property(key), child);
                    find(nulls, &expanded, child, &join(path, key))
                })
        }
        Value::Array(items) => items
            .iter()
            .enumerate()
            .filter(|(_, item)| item.is_object() || item.is_array())
            .find_map(|(index, item)| {
                let expanded = nulls.expand_all(&applicable.item(index), item);
                find(nulls, &expanded, item, &format!("{path}[{index}]"))
            }),
        _ => None,
    }
}

fn join(path: &str, key: &str) -> String {
    if path.is_empty() {
        key.to_owned()
    } else {
        format!("{path}.{key}")
    }
}

/// What the schemas at one object say about one of its keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    /// A schema declares it in `properties`.
    Declared,
    /// A schema opens the object to any key (`additionalProperties`,
    /// `patternProperties`, `true`).
    OpenExplicitly,
    /// A schema says nothing about keys (`{}`, `{"type": "object"}`).
    OpenImplicitly,
    /// A schema lists its keys and this is not one of them.
    Undeclared,
    /// The schema does not apply to keys: it only carries `$ref`, `allOf`
    /// or a union, or its `type` excludes objects.
    Silent,
}

fn status(applicable: &Applicable<'_>, key: &str) -> Status {
    match applicable {
        Applicable::Schema(schema) => schema_status(schema, key),
        // Every member applies: a key one of them declares or opens is
        // allowed.
        Applicable::All(members) => {
            let statuses: Vec<_> = members.iter().map(|m| status(m, key)).collect();
            [
                Status::Declared,
                Status::OpenExplicitly,
                Status::Undeclared,
                Status::OpenImplicitly,
            ]
            .into_iter()
            .find(|wanted| statuses.contains(wanted))
            .unwrap_or(Status::Silent)
        }
        // Some branch applies, without saying which: a key is undeclared
        // only when no branch declares or opens it.
        Applicable::Any(members) => {
            let statuses: Vec<_> = members.iter().map(|m| status(m, key)).collect();
            [
                Status::Declared,
                Status::OpenExplicitly,
                Status::OpenImplicitly,
                Status::Undeclared,
            ]
            .into_iter()
            .find(|wanted| statuses.contains(wanted))
            .unwrap_or(Status::Silent)
        }
    }
}

fn schema_status(schema: &Value, key: &str) -> Status {
    let Value::Object(map) = schema else {
        return match schema {
            Value::Bool(true) => Status::OpenExplicitly,
            _ => Status::Silent,
        };
    };
    if map
        .get("type")
        .is_some_and(|ty| !super::type_matches(ty, &Value::Object(Default::default())))
    {
        return Status::Silent;
    }
    let properties = map.get("properties").and_then(Value::as_object);
    if properties.is_some_and(|p| p.contains_key(key)) {
        return Status::Declared;
    }
    if map.contains_key("patternProperties") {
        return Status::OpenExplicitly;
    }
    match map.get("additionalProperties") {
        Some(Value::Bool(false)) => return Status::Undeclared,
        Some(_) => return Status::OpenExplicitly,
        None => {}
    }
    if properties.is_some() {
        return Status::Undeclared;
    }
    if ["$ref", "allOf", "anyOf", "oneOf"]
        .iter()
        .any(|keyword| map.contains_key(*keyword))
    {
        return Status::Silent;
    }
    Status::OpenImplicitly
}

/// The keys the schemas in `applicable` declare.
fn declared_keys<'s>(applicable: &Applicable<'s>, out: &mut BTreeSet<&'s str>) {
    match applicable {
        Applicable::Schema(schema) => {
            if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
                out.extend(properties.keys().map(String::as_str));
            }
        }
        Applicable::All(members) | Applicable::Any(members) => {
            for member in members {
                declared_keys(member, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn undeclared(schema: &Value, value: Value) -> Option<(String, String)> {
        find_undeclared_key(schema, &value).map(|found| (found.key, found.path))
    }

    #[test]
    fn a_misnamed_key_is_undeclared() {
        let schema = json!({
            "type": "object",
            "properties": {"email": {"type": ["string", "null"]}, "name": {"type": "string"}},
            "required": ["name"]
        });
        let found = find_undeclared_key(&schema, &json!({"name": "Ada", "e_mail": "a@b"})).unwrap();
        assert_eq!(found.key, "e_mail");
        assert_eq!(found.path, "e_mail");
        assert_eq!(found.declared, ["email", "name"]);
        assert_eq!(
            undeclared(&schema, json!({"name": "Ada", "email": null})),
            None
        );
    }

    #[test]
    fn nested_objects_arrays_and_refs_are_walked() {
        let schema = json!({
            "type": "object",
            "properties": {
                "contact": {"$ref": "#/$defs/Contact"},
                "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}}
            },
            "$defs": {
                "Contact": {"type": "object", "properties": {"email": {"type": "string"}}},
                "Item": {"type": "object", "properties": {"colour": {"type": "string"}}}
            }
        });
        assert_eq!(
            undeclared(&schema, json!({"contact": {"e_mail": "x"}})),
            Some(("e_mail".into(), "contact.e_mail".into()))
        );
        assert_eq!(
            undeclared(
                &schema,
                json!({"items": [{"colour": "red"}, {"color": "blue"}]})
            ),
            Some(("color".into(), "items[1].color".into()))
        );
        // A key nested at the wrong level.
        assert_eq!(
            undeclared(&schema, json!({"email": "x", "contact": {}})),
            Some(("email".into(), "email".into()))
        );
    }

    #[test]
    fn maps_flattened_maps_and_open_schemas_are_exempt() {
        let map = json!({
            "type": "object",
            "properties": {"labels": {"type": "object", "additionalProperties": {"type": "string"}}}
        });
        assert_eq!(undeclared(&map, json!({"labels": {"anything": "x"}})), None);
        let flattened = json!({
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "additionalProperties": true
        });
        assert_eq!(undeclared(&flattened, json!({"id": "1", "extra": 2})), None);
        let pattern = json!({"type": "object", "properties": {}, "patternProperties": {"^x": {}}});
        assert_eq!(undeclared(&pattern, json!({"y": 1})), None);
        for open in [json!(true), json!({}), json!({"type": "object"})] {
            let schema = json!({"type": "object", "properties": {"data": open}});
            assert_eq!(
                undeclared(&schema, json!({"data": {"k": 1}})),
                None,
                "{schema}"
            );
        }
        // Map values are still checked against their schema.
        let nested = json!({
            "type": "object",
            "additionalProperties": {"type": "object", "properties": {"a": {}}}
        });
        assert_eq!(
            undeclared(&nested, json!({"k": {"b": 1}})),
            Some(("b".into(), "k.b".into()))
        );
    }

    #[test]
    fn a_closed_object_rejects_like_deny_unknown_fields() {
        let schema = json!({
            "type": "object",
            "properties": {"a": {}},
            "additionalProperties": false
        });
        assert_eq!(
            undeclared(&schema, json!({"b": 1})),
            Some(("b".into(), "b".into()))
        );
    }

    /// The union rule is inverted: with every branch optional, the value
    /// may match each, and a key is undeclared only when none declares it.
    #[test]
    fn untagged_unions_check_against_every_branch_that_may_apply() {
        let schema = json!({
            "anyOf": [
                {"type": "object", "properties": {"email": {"type": "string"}}},
                {"type": "object", "properties": {"phone": {"type": "string"}}}
            ]
        });
        assert_eq!(undeclared(&schema, json!({"phone": "1"})), None);
        assert_eq!(undeclared(&schema, json!({"email": "a"})), None);
        let found = find_undeclared_key(&schema, &json!({"e_mail": "a"})).unwrap();
        assert_eq!(found.key, "e_mail");
        assert_eq!(found.declared, ["email", "phone"]);
    }

    #[test]
    fn an_optional_struct_does_not_open_through_its_null_branch() {
        let schema = json!({
            "type": "object",
            "properties": {
                "contact": {"anyOf": [{"$ref": "#/$defs/Contact"}, {"type": "null"}]}
            },
            "$defs": {
                "Contact": {
                    "type": "object",
                    "properties": {"email": {"type": "string"}},
                    "required": ["email"]
                }
            }
        });
        assert_eq!(
            undeclared(&schema, json!({"contact": {"e_mail": "x"}})),
            Some(("e_mail".into(), "contact.e_mail".into()))
        );
    }

    #[test]
    fn all_of_members_and_flattened_enums_declare_their_keys() {
        let schema = json!({
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "oneOf": [
                {"properties": {"kind": {"const": "a"}, "a_field": {}}, "required": ["kind"]},
                {"properties": {"kind": {"const": "b"}, "b_field": {}}, "required": ["kind"]}
            ]
        });
        assert_eq!(
            undeclared(&schema, json!({"id": "1", "kind": "a", "a_field": 1})),
            None
        );
        assert_eq!(
            undeclared(&schema, json!({"id": "1", "kind": "a", "b_field": 1})),
            Some(("b_field".into(), "b_field".into()))
        );
        let all_of = json!({
            "allOf": [
                {"type": "object", "properties": {"a": {}}},
                {"type": "object", "properties": {"b": {}}}
            ]
        });
        assert_eq!(undeclared(&all_of, json!({"a": 1, "b": 2})), None);
        assert_eq!(
            undeclared(&all_of, json!({"c": 3})),
            Some(("c".into(), "c".into()))
        );
    }

    #[test]
    fn scalars_and_cycles_terminate() {
        assert_eq!(undeclared(&json!({"type": "string"}), json!("x")), None);
        let cyclic = json!({"$ref": "#/$defs/A", "$defs": {"A": {"$ref": "#/$defs/A"}}});
        assert_eq!(undeclared(&cyclic, json!({"k": 1})), None);
    }
}
