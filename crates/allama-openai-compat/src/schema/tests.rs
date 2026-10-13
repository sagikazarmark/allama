use std::collections::{BTreeMap, HashMap};

use allama::adapter::schema::{
    Degradable, SchemaError, StrictOutcome, apply_strict_policy, normalize_strict_nulls,
};
use allama::{Error, UnsupportedPolicy, Warning, WarningKind};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::*;

fn run(schema: &Value) -> (Result<Value, StrictSchemaError>, Vec<Warning>) {
    run_with(schema, SchemaDialect::OpenAiStrict)
}

fn run_with(
    schema: &Value,
    dialect: SchemaDialect,
) -> (Result<Value, StrictSchemaError>, Vec<Warning>) {
    let mut d = Degradations::new("demo", UnsupportedPolicy::Warn);
    let out = strict_schema(schema, RequestField::ResponseFormat, dialect, &mut d);
    (out, d.into_warnings())
}

fn strict(schema: &Value) -> Value {
    let (out, warnings) = run(schema);
    let out = out.unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    out
}

#[track_caller]
fn assert_not_representable(schema: &Value, needle: &str) {
    let (out, warnings) = run(schema);
    match out {
        Err(StrictSchemaError::NotRepresentable { reason, .. }) => {
            assert!(reason.contains(needle), "{reason}");
        }
        other => panic!("expected NotRepresentable, got {other:?}"),
    }
    assert!(warnings.is_empty(), "no warnings on failure: {warnings:?}");
}

#[test]
fn closes_objects_and_requires_every_property() {
    let out = strict(&json!({
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "integer"},
            "nested": {"type": "object", "properties": {"c": {"type": "boolean"}}}
        },
        "required": ["a", "nested"]
    }));
    assert_eq!(out["additionalProperties"], false);
    assert_eq!(out["required"], json!(["a", "b", "nested"]));
    assert_eq!(out["properties"]["a"]["type"], "string");
    assert_eq!(out["properties"]["b"]["type"], json!(["integer", "null"]));
    let nested = &out["properties"]["nested"];
    assert_eq!(nested["additionalProperties"], false);
    assert_eq!(nested["required"], json!(["c"]));
}

#[test]
fn the_input_is_left_unchanged() {
    let schema = json!({"type": "object", "properties": {"a": {"oneOf": [{"type": "string"}]}}});
    let before = schema.clone();
    let _ = run(&schema);
    assert_eq!(schema, before);
}

#[test]
fn schema_uri_is_removed() {
    let out = strict(&json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Person",
        "type": "object",
        "properties": {}
    }));
    assert!(out.get("$schema").is_none());
    assert_eq!(out["title"], "Person");
}

#[test]
fn disjoint_one_of_becomes_any_of_silently() {
    let (out, warnings) = run(&json!({
        "type": "object",
        "properties": {
            "kind": {"oneOf": [{"const": "a", "description": "A"}, {"const": "b"}]},
            "oneOf": {"type": "string"}
        },
        "required": ["kind", "oneOf"]
    }));
    let out = out.unwrap();
    assert_eq!(
        out["properties"]["kind"],
        json!({"anyOf": [{"const": "a", "description": "A"}, {"const": "b"}]})
    );
    // A property *named* `oneOf` is not a keyword.
    assert_eq!(out["properties"]["oneOf"], json!({"type": "string"}));
    assert!(warnings.is_empty(), "{warnings:?}");
}

#[test]
fn overlapping_one_of_becomes_any_of_with_a_warning() {
    let (out, warnings) = run(&json!({
        "type": "object",
        "properties": {"n": {"oneOf": [{"type": "integer"}, {"type": "number", "minimum": 0}]}},
        "required": ["n"]
    }));
    assert_eq!(
        out.unwrap()["properties"]["n"],
        json!({"anyOf": [{"type": "integer"}, {"type": "number", "minimum": 0}]})
    );
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].kind, WarningKind::Approximated);
    assert!(warnings[0].message.contains("anyOf"), "{}", warnings[0]);
}

#[test]
fn disjointness_follows_references_and_ignores_closing() {
    // Two open objects with different keys overlap ({"a": .., "b": ..}
    // matches both), even though closing them would make them disjoint.
    let (_, warnings) = run(&json!({
        "type": "object",
        "properties": {"v": {"oneOf": [
            {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
            {"type": "object", "properties": {"b": {"type": "string"}}, "required": ["b"]}
        ]}},
        "required": ["v"]
    }));
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    // Tags behind references are read through them.
    let (_, warnings) = run(&json!({
        "type": "object",
        "properties": {"v": {"oneOf": [{"$ref": "#/$defs/A"}, {"$ref": "#/$defs/B"}]}},
        "required": ["v"],
        "$defs": {
            "A": {"type": "object", "properties": {"t": {"const": "a"}}, "required": ["t"]},
            "B": {"type": "object", "properties": {"t": {"const": "b"}}, "required": ["t"]}
        }
    }));
    assert!(warnings.is_empty(), "{warnings:?}");
}

#[test]
fn numeric_formats_are_dropped_silently_and_bounds_kept() {
    let out = strict(&json!({
        "type": "object",
        "properties": {
            "count": {"type": "integer", "format": "uint32", "minimum": 0},
            "ratio": {"type": ["number", "null"], "format": "double"}
        },
        "required": ["count", "ratio"]
    }));
    assert_eq!(
        out["properties"]["count"],
        json!({"type": "integer", "minimum": 0})
    );
    assert_eq!(
        out["properties"]["ratio"],
        json!({"type": ["number", "null"]})
    );
}

#[test]
fn supported_formats_are_kept_and_others_removed_with_a_warning() {
    let (out, warnings) = run(&json!({
        "type": "object",
        "properties": {
            "at": {"type": "string", "format": "date-time"},
            "id": {"type": "string", "format": "uuid"},
            "home": {"type": "string", "format": "uri"},
            "ip": {"type": "string", "format": "ipv4"}
        },
        "required": ["at", "id", "home", "ip"]
    }));
    let out = out.unwrap();
    assert_eq!(out["properties"]["at"]["format"], "date-time");
    assert_eq!(out["properties"]["id"]["format"], "uuid");
    assert_eq!(out["properties"]["ip"]["format"], "ipv4");
    assert!(out["properties"]["home"].get("format").is_none());
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].kind, WarningKind::Approximated);
    assert!(warnings[0].message.contains("uri"), "{}", warnings[0]);
}

#[test]
fn ref_with_siblings_is_inlined_and_bare_refs_kept() {
    let out = strict(&json!({
        "type": "object",
        "properties": {
            "home": {"$ref": "#/$defs/Address", "description": "Where they live"},
            "work": {"$ref": "#/$defs/Address"}
        },
        "required": ["home", "work"],
        "$defs": {
            "Address": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        }
    }));
    assert_eq!(
        out["properties"]["home"],
        json!({
            "type": "object",
            "description": "Where they live",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": false
        })
    );
    assert_eq!(
        out["properties"]["work"],
        json!({"$ref": "#/$defs/Address"})
    );
    assert_eq!(out["$defs"]["Address"]["additionalProperties"], false);
}

#[test]
fn recursive_refs_survive_sibling_inlining() {
    let out = strict(&json!({
        "type": "object",
        "properties": {"root": {"$ref": "#/$defs/Node", "description": "The root"}},
        "required": ["root"],
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}},
                "required": ["children"]
            }
        }
    }));
    let root = &out["properties"]["root"];
    assert_eq!(root["description"], "The root");
    assert_eq!(
        root["properties"]["children"]["items"],
        json!({"$ref": "#/$defs/Node"})
    );
}

#[test]
fn internally_tagged_newtype_variants_are_merged_without_all_of() {
    let (out, warnings) = run(&json!({
        "type": "object",
        "properties": {"shape": {"$ref": "#/$defs/Shape"}},
        "required": ["shape"],
        "$defs": {
            "Inner": {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
            "Shape": {"oneOf": [
                {"type": "object", "properties": {"kind": {"type": "string", "const": "Unit"}}, "required": ["kind"]},
                {"$ref": "#/$defs/Inner", "type": "object",
                 "properties": {"kind": {"type": "string", "const": "New"}}, "required": ["kind"]}
            ]}
        }
    }));
    let out = out.unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    let new = &out["$defs"]["Shape"]["anyOf"][1];
    assert_eq!(
        new,
        &json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}, "kind": {"type": "string", "const": "New"}},
            "required": ["a", "kind"],
            "additionalProperties": false
        })
    );
    assert!(!out.to_string().contains("allOf"), "{out}");
}

#[test]
fn a_ref_that_conflicts_with_its_siblings_is_not_representable() {
    assert_not_representable(
        &json!({
            "type": "object",
            "properties": {"v": {"$ref": "#/$defs/Closed", "properties": {"extra": {"type": "string"}}}},
            "$defs": {"Closed": {"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": false}}
        }),
        "/properties/v",
    );
    assert_not_representable(
        &json!({
            "type": "object",
            "properties": {"v": {"$ref": "#/$defs/S", "type": "integer"}},
            "$defs": {"S": {"type": "string"}}
        }),
        "`type` differs",
    );
}

#[test]
fn single_all_of_is_flattened() {
    let out = strict(&json!({
        "type": "object",
        "properties": {
            "status": {"allOf": [{"$ref": "#/$defs/Status"}], "description": "Current status"}
        },
        "required": ["status"],
        "$defs": {"Status": {"type": "string", "enum": ["open", "closed"]}}
    }));
    assert_eq!(
        out["properties"]["status"],
        json!({"type": "string", "enum": ["open", "closed"], "description": "Current status"})
    );
}

#[test]
fn multi_element_all_of_is_not_representable() {
    // The members are not closed and sent: the whole schema is refused.
    assert_not_representable(
        &json!({
            "type": "object",
            "properties": {"a": {"allOf": [
                {"type": "object", "properties": {"x": {"type": "string"}}},
                {"type": "object", "properties": {"y": {"type": "string"}}}
            ]}}
        }),
        "`allOf`",
    );
    assert_not_representable(
        &json!({"allOf": [{"type": "object"}, {"type": "object"}]}),
        "intersection of 2 schemas",
    );
    // Inside a definition, too.
    assert_not_representable(
        &json!({
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/A"}},
            "$defs": {"A": {"allOf": [{"type": "string"}, {"minLength": 1}]}}
        }),
        "/$defs/A",
    );
    // A single-element `allOf` that cannot be flattened (it conflicts with
    // its siblings) is refused as well: the dialect never emits `allOf`.
    assert_not_representable(
        &json!({
            "type": "object",
            "properties": {"a": {"type": "string", "allOf": [{"type": "integer"}]}}
        }),
        "single-element `allOf`",
    );
}

#[test]
fn not_is_not_representable() {
    assert_not_representable(
        &json!({
            "type": "object",
            "properties": {"a": {"type": "string", "not": {"const": "x"}}}
        }),
        "`not`",
    );
}

#[test]
fn dependent_required_is_not_representable() {
    assert_not_representable(
        &json!({
            "type": "object",
            "properties": {"card": {"type": "string"}, "cvc": {"type": "string"}},
            "dependentRequired": {"card": ["cvc"]}
        }),
        "`dependentRequired`",
    );
}

#[test]
fn dependent_schemas_is_not_representable() {
    assert_not_representable(
        &json!({
            "type": "object",
            "properties": {"card": {"type": "string"}},
            "dependentSchemas": {"card": {"required": ["cvc"]}}
        }),
        "`dependentSchemas`",
    );
}

#[test]
fn conditionals_are_not_representable_for_the_right_reason() {
    for (schema, keywords) in [
        (
            json!({
                "type": "object",
                "properties": {"kind": {"type": "string"}},
                "if": {"properties": {"kind": {"const": "a"}}},
                "then": {"required": ["kind"]}
            }),
            &["`if`", "`then`"][..],
        ),
        // A bare `then` or `else` is reported, not "accepts any value".
        (
            json!({
                "type": "object",
                "properties": {"a": {"then": {"type": "string"}}}
            }),
            &["`then`"][..],
        ),
        (
            json!({
                "type": "object",
                "properties": {"a": {"type": "string", "else": {"minLength": 1}}}
            }),
            &["`else`"][..],
        ),
    ] {
        let (out, warnings) = run(&schema);
        let Err(StrictSchemaError::NotRepresentable { reason, .. }) = out else {
            panic!("{schema}: {out:?}");
        };
        assert!(reason.contains("conditional"), "{reason}");
        assert!(!reason.contains("accepts any value"), "{reason}");
        for keyword in keywords {
            assert!(reason.contains(keyword), "{keyword}: {reason}");
        }
        assert!(warnings.is_empty(), "{warnings:?}");
    }
}

#[test]
fn rejected_keywords_are_published() {
    for dialect in [
        SchemaDialect::OpenAiStrict,
        SchemaDialect::OpenAiStrictFineTuned,
    ] {
        let rejected = dialect.rejected_keywords();
        for keyword in [
            "allOf",
            "not",
            "dependentRequired",
            "dependentSchemas",
            "if",
            "then",
            "else",
        ] {
            assert!(rejected.contains(&keyword), "{dialect:?}: {keyword}");
            // Every listed keyword is refused wherever it appears.
            let schema = json!({
                "type": "object",
                "properties": {"a": {"type": "string", keyword: {"type": "string"}}}
            });
            let (out, _) = run_with(&schema, dialect);
            match out {
                Err(StrictSchemaError::NotRepresentable { reason, .. }) => {
                    assert!(reason.contains(&format!("`{keyword}`")), "{reason}");
                }
                other => panic!("{dialect:?} {keyword}: {other:?}"),
            }
        }
    }
}

#[test]
fn defaults_are_removed() {
    let out = strict(&json!({
        "type": "object",
        "default": {"note": null, "count": 3},
        "properties": {
            "note": {"type": ["string", "null"], "default": null},
            "count": {"type": "integer", "default": 3},
            "home": {"$ref": "#/$defs/Address", "default": {"city": "Paris"}},
            "work": {"$ref": "#/$defs/Address", "default": {"city": "Paris"}, "description": "Office"}
        },
        "required": ["note", "count"],
        "$defs": {"Address": {
            "type": "object",
            "default": {"city": ""},
            "properties": {"city": {"type": "string", "default": ""}}
        }}
    }));
    assert!(out.get("default").is_none());
    let properties = &out["properties"];
    assert_eq!(properties["note"], json!({"type": ["string", "null"]}));
    assert_eq!(properties["count"], json!({"type": "integer"}));
    // A reference with a default beside it is merged with its target, and
    // loses the default.
    assert_eq!(
        properties["home"],
        json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": false
        })
    );
    // A merged target loses its default too.
    assert!(
        properties["work"].get("default").is_none(),
        "{}",
        properties["work"]
    );
    assert!(
        properties["work"]["properties"]["city"]
            .get("default")
            .is_none()
    );
    let address = &out["$defs"]["Address"];
    assert!(address.get("default").is_none());
    assert!(address["properties"]["city"].get("default").is_none());
}

#[test]
fn optional_properties_with_a_default_stay_non_nullable() {
    let out = strict(&json!({
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"type": "string"}, "default": []},
            "count": {"type": "integer", "default": 0},
            "home": {"$ref": "#/$defs/Address", "default": {"city": ""}}
        },
        "$defs": {"Address": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
    }));
    assert_eq!(out["required"], json!(["count", "home", "tags"]));
    assert_eq!(
        out["properties"]["tags"],
        json!({"type": "array", "items": {"type": "string"}})
    );
    assert_eq!(out["properties"]["count"], json!({"type": "integer"}));
    // Merged with its target; the default kept it non-nullable.
    assert_eq!(
        out["properties"]["home"],
        json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": false
        })
    );
}

#[test]
fn optional_properties_without_a_default_become_nullable() {
    let (out, warnings) = run(&json!({
        "type": "object",
        "properties": {
            "plain": {"type": "string"},
            "typed_enum": {"type": "string", "enum": ["low", "high"]},
            "nullable_type_enum": {"type": ["string", "null"], "enum": ["low", "high"]},
            "bare_enum": {"enum": ["a", "b"]},
            "typed_const": {"type": "string", "const": "fixed"},
            "bare_ref": {"$ref": "#/$defs/Limits"},
            "described_ref": {"$ref": "#/$defs/Limits", "description": "Caps", "title": "Limits"},
            "union": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            "tagged": {"oneOf": [{"const": "a"}, {"const": "b"}]},
            "mixed": {"type": "string", "anyOf": [{"type": "string", "minLength": 1}, {"const": ""}]}
        },
        "$defs": {"Limits": {"type": "object", "properties": {"max": {"type": "integer"}}, "required": ["max"]}}
    }));
    let out = out.unwrap();
    let properties = &out["properties"];
    assert_eq!(properties["plain"], json!({"type": ["string", "null"]}));
    // The enum lists the null the type now admits.
    assert_eq!(
        properties["typed_enum"],
        json!({"type": ["string", "null"], "enum": ["low", "high", null]})
    );
    assert_eq!(
        properties["nullable_type_enum"],
        json!({"type": ["string", "null"], "enum": ["low", "high", null]})
    );
    assert_eq!(properties["bare_enum"], json!({"enum": ["a", "b", null]}));
    assert_eq!(
        properties["typed_const"],
        json!({"type": ["string", "null"], "enum": ["fixed", null]})
    );
    assert_eq!(
        properties["bare_ref"],
        json!({"anyOf": [{"$ref": "#/$defs/Limits"}, {"type": "null"}]})
    );
    // Merged with its target, then made nullable.
    assert_eq!(
        properties["described_ref"],
        json!({
            "type": ["object", "null"],
            "title": "Limits",
            "description": "Caps",
            "properties": {"max": {"type": "integer"}},
            "required": ["max"],
            "additionalProperties": false
        })
    );
    assert_eq!(
        properties["union"],
        json!({"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}]})
    );
    // The `oneOf` is disjoint, so its rewrite is silent.
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        properties["tagged"],
        json!({"anyOf": [{"const": "a"}, {"const": "b"}, {"type": "null"}]})
    );
    assert_eq!(
        properties["mixed"],
        json!({"anyOf": [
            {"type": "string", "anyOf": [{"type": "string", "minLength": 1}, {"const": ""}]},
            {"type": "null"}
        ]})
    );
}

#[test]
fn optional_properties_that_admit_null_are_unchanged() {
    let schema = json!({
        "type": "object",
        "properties": {
            "option": {"type": ["string", "null"]},
            "option_ref": {"anyOf": [{"$ref": "#/$defs/Color"}, {"type": "null"}]},
            "nullable_enum": {"enum": ["a", null]},
            "ref_to_nullable": {"$ref": "#/$defs/Maybe"},
            "with_null_default": {"type": ["integer", "null"], "default": null}
        },
        "$defs": {
            "Color": {"type": "string", "enum": ["red"]},
            "Maybe": {"type": ["string", "null"]}
        }
    });
    let out = strict(&schema);
    let properties = &out["properties"];
    for name in ["option", "option_ref", "nullable_enum", "ref_to_nullable"] {
        assert_eq!(properties[name], schema["properties"][name], "{name}");
    }
    assert_eq!(
        properties["with_null_default"],
        json!({"type": ["integer", "null"]})
    );
}

/// An output strict mode admits, with `null` wherever `schema` allows one:
/// the worst case for deserialization. Every object lists every property;
/// an array holds one element.
fn null_maximal(schema: &Value, document: &Value, depth: usize) -> Value {
    let map = schema
        .as_object()
        .unwrap_or_else(|| panic!("strict subschemas are objects: {schema}"));
    if let Some(Value::String(reference)) = map.get("$ref") {
        let target = resolve(document, reference).expect("resolvable reference");
        return null_maximal(target, document, depth + 1);
    }
    if let Some(Value::Array(variants)) = map.get("anyOf") {
        if variants.contains(&json!({"type": "null"})) {
            return Value::Null;
        }
        return null_maximal(&variants[0], document, depth + 1);
    }
    if let Some(Value::Array(values)) = map.get("enum") {
        return if values.contains(&Value::Null) {
            Value::Null
        } else {
            values[0].clone()
        };
    }
    if let Some(value) = map.get("const") {
        return value.clone();
    }
    let types: Vec<&str> = match map.get("type") {
        Some(Value::String(name)) => vec![name.as_str()],
        Some(Value::Array(names)) => names.iter().filter_map(Value::as_str).collect(),
        _ => panic!("untyped strict subschema: {schema}"),
    };
    if types.contains(&"null") {
        return Value::Null;
    }
    match types[0] {
        "object" => {
            let properties = map.get("properties").and_then(Value::as_object);
            assert_eq!(map["additionalProperties"], false, "{schema}");
            let keys: Vec<Value> = properties
                .map(|p| p.keys().cloned().map(Value::String).collect())
                .unwrap_or_default();
            if properties.is_some() {
                assert_eq!(
                    map["required"],
                    Value::Array(keys),
                    "every property is required"
                );
            }
            Value::Object(
                properties
                    .into_iter()
                    .flatten()
                    .map(|(name, property)| {
                        (name.clone(), null_maximal(property, document, depth + 1))
                    })
                    .collect(),
            )
        }
        "array" if depth > 8 => json!([]),
        "array" => json!([null_maximal(&map["items"], document, depth + 1)]),
        "string" => json!("x"),
        "integer" => json!(1),
        "number" => json!(1.5),
        "boolean" => json!(true),
        other => panic!("unexpected type {other}"),
    }
}

/// Generate `T`'s schema, rewrite it for strict mode, and deserialize the
/// null-maximal output after normalizing it against the original schema.
/// Returns the value and the strict schema.
#[track_caller]
fn round_trip<T: JsonSchema + DeserializeOwned>() -> (T, Value) {
    let original = allama::adapter::schema::schema_for::<T>();
    let (strict, _) = run(&original);
    let strict = strict.unwrap_or_else(|e| panic!("{e}: {original}"));
    let mut output = null_maximal(&strict, &strict, 0);
    normalize_strict_nulls(&original, &mut output);
    let value = serde_json::from_value(output.clone())
        .unwrap_or_else(|e| panic!("{e}\noutput: {output}\nstrict: {strict}"));
    (value, strict)
}

#[derive(Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
struct Address {
    city: String,
    #[serde(default)]
    lines: Vec<String>,
    zip: Option<String>,
}

/// Not `Serialize`: schemars writes no `default` for `#[serde(default)]`
/// fields of this type.
#[derive(Debug, Default, PartialEq, Deserialize, JsonSchema)]
struct Limits {
    max: u32,
    #[serde(default)]
    labels: Vec<String>,
}

/// Not `Serialize`, and inlined: an `enum` on the property itself.
#[derive(Debug, Default, PartialEq, Deserialize, JsonSchema)]
#[schemars(inline)]
enum Priority {
    #[default]
    Low,
    High,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
enum Color {
    Red,
    Green,
}

#[derive(Debug, PartialEq, Deserialize, JsonSchema)]
#[serde(tag = "kind")]
enum Shape {
    Circle {
        r: f64,
        #[serde(default)]
        tags: Vec<String>,
        label: Option<String>,
    },
    Square {
        side: f64,
    },
}

#[derive(Debug, PartialEq, Deserialize, JsonSchema)]
struct Everything {
    #[serde(default)]
    tags: Vec<String>,
    option: Option<String>,
    #[serde(default)]
    double: Option<Option<String>>,
    color: Option<Color>,
    #[serde(default)]
    priority: Priority,
    home: Address,
    work: Option<Address>,
    #[serde(default)]
    limits: Limits,
    #[serde(default)]
    history: Vec<Limits>,
    shape: Shape,
}

#[test]
fn strict_output_round_trips_through_serde() {
    let (value, strict) = round_trip::<Everything>();
    let properties = &strict["properties"];

    // `#[serde(default)] Vec<T>` with a `Serialize` default: required and
    // not nullable, so the model sends a list.
    assert_eq!(properties["tags"]["type"], "array");
    assert_eq!(value.tags, ["x"]);
    // `Option<T>`, `Option<Option<T>>` and an optional `$ref` enum.
    assert_eq!(value.option, None);
    assert_eq!(value.double, None);
    assert_eq!(value.color, None);
    // An enum field made nullable lists `null` in its `enum`.
    assert_eq!(properties["priority"]["type"], json!(["string", "null"]));
    assert_eq!(properties["priority"]["enum"], json!(["Low", "High", null]));
    assert_eq!(value.priority, Priority::Low);
    // Nested `$defs`.
    assert_eq!(
        value.home,
        Address {
            city: "x".into(),
            lines: vec!["x".into()],
            zip: None
        }
    );
    assert_eq!(value.work, None);
    // `#[serde(default)]` fields whose type is not `Serialize`: nullable in
    // the strict schema, and normalized back to their default.
    assert_eq!(
        properties["limits"],
        json!({"anyOf": [{"$ref": "#/$defs/Limits"}, {"type": "null"}]})
    );
    assert_eq!(value.limits, Limits::default());
    assert_eq!(properties["history"]["type"], json!(["array", "null"]));
    assert!(value.history.is_empty());
    // A tagged union variant.
    assert_eq!(
        value.shape,
        Shape::Circle {
            r: 1.5,
            tags: vec!["x".into()],
            label: None
        }
    );
}

#[test]
fn container_level_defaults_round_trip() {
    #[derive(Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
    #[serde(default)]
    struct Settings {
        retries: u32,
        tags: Vec<String>,
        name: Option<String>,
        home: Address,
    }
    let (value, strict) = round_trip::<Settings>();
    assert_eq!(
        strict["properties"]["retries"],
        json!({"type": "integer", "minimum": 0})
    );
    assert_eq!(value.retries, 1);
    assert_eq!(value.name, None);

    // schemars writes a `default` only for fields whose type is
    // `Serialize`; `limits` gets none, so it is nullable and normalization
    // restores the container's default for it.
    #[derive(Debug, Default, PartialEq, Deserialize, JsonSchema)]
    #[serde(default)]
    struct Lenient {
        retries: u32,
        name: Option<String>,
        limits: Limits,
    }
    let (value, strict) = round_trip::<Lenient>();
    assert_eq!(
        strict["properties"]["limits"],
        json!({"anyOf": [{"$ref": "#/$defs/Limits"}, {"type": "null"}]})
    );
    assert_eq!(
        value,
        Lenient {
            retries: 1,
            name: None,
            limits: Limits::default()
        }
    );
}

/// Stays in the suite whichever `default` rule the dialect uses: a model
/// that sends `null` for a `#[serde(default)] Vec` (a non-strict endpoint,
/// or a dialect that makes it nullable) still round-trips.
#[test]
fn serde_default_vec_round_trips_even_when_the_model_sends_null() {
    #[derive(Debug, PartialEq, Deserialize, JsonSchema)]
    struct Tagged {
        #[serde(default)]
        tags: Vec<String>,
    }
    let (value, _) = round_trip::<Tagged>();
    assert_eq!(value.tags, ["x"]);

    let original = allama::adapter::schema::schema_for::<Tagged>();
    let mut output = json!({"tags": null});
    assert!(serde_json::from_value::<Tagged>(output.clone()).is_err());
    normalize_strict_nulls(&original, &mut output);
    let value: Tagged = serde_json::from_value(output).unwrap();
    assert!(value.tags.is_empty());
}

/// `serde_json::Value` has no strict form, so such a schema goes out
/// non-strict; its `null` is a value and is kept.
#[test]
fn serde_json_value_keeps_its_null() {
    #[derive(Debug, PartialEq, Deserialize, JsonSchema)]
    struct Untyped {
        payload: Value,
        #[serde(default)]
        tags: Vec<String>,
    }
    let original = allama::adapter::schema::schema_for::<Untyped>();
    let (out, _) = run(&original);
    assert!(matches!(
        out,
        Err(StrictSchemaError::NotRepresentable { .. })
    ));
    let mut output = json!({"payload": null, "tags": null});
    normalize_strict_nulls(&original, &mut output);
    assert_eq!(output, json!({"payload": null}));
    let value: Untyped = serde_json::from_value(output).unwrap();
    assert_eq!(value.payload, Value::Null);
}

/// The normalizer removes exactly the `null`s the strict rewrite added: on
/// a null-maximal output, every `null` the original schema admits is kept.
#[test]
fn normalization_keeps_every_null_the_original_admits() {
    let original = allama::adapter::schema::schema_for::<Everything>();
    let (strict, _) = run(&original);
    let strict = strict.unwrap();
    let mut output = null_maximal(&strict, &strict, 0);
    normalize_strict_nulls(&original, &mut output);
    for name in ["option", "double", "color", "work"] {
        assert_eq!(output[name], Value::Null, "{name}: {output}");
    }
    for name in ["priority", "limits", "history"] {
        assert!(output.get(name).is_none(), "{name}: {output}");
    }
    assert_eq!(output["home"]["zip"], Value::Null);
    assert_eq!(output["shape"]["label"], Value::Null);
}

#[test]
fn non_object_roots_are_not_representable() {
    assert_not_representable(
        &json!({"type": "array", "items": {"type": "string"}}),
        "not an object",
    );
    assert_not_representable(
        &json!({"type": "string", "enum": ["a", "b"]}),
        "not an object",
    );
    assert_not_representable(
        &json!({"oneOf": [{"type": "object", "properties": {}}, {"type": "string"}]}),
        "union",
    );
    assert_not_representable(&json!(true), "root");
}

#[test]
fn a_root_ref_is_resolved() {
    let out = strict(&json!({
        "$ref": "#/$defs/Root",
        "$defs": {"Root": {"type": "object", "properties": {"a": {"type": "string"}}}}
    }));
    assert_eq!(out["type"], "object");
    assert_eq!(out["required"], json!(["a"]));
    assert_eq!(out["additionalProperties"], false);
}

#[test]
fn fine_tuned_models_lose_constraint_keywords() {
    let schema = json!({
        "type": "object",
        "properties": {
            "code": {"type": "string", "pattern": "^[A-Z]+$", "minLength": 2, "maxLength": 8, "format": "email"},
            "count": {"type": "integer", "minimum": 0, "maximum": 9, "multipleOf": 3},
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4}
        },
        "required": ["code", "count", "tags"]
    });
    let (out, warnings) = run_with(&schema, SchemaDialect::OpenAiStrictFineTuned);
    let out = out.unwrap();
    assert_eq!(out["properties"]["code"], json!({"type": "string"}));
    assert_eq!(out["properties"]["count"], json!({"type": "integer"}));
    assert_eq!(
        out["properties"]["tags"],
        json!({"type": "array", "items": {"type": "string"}})
    );
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    for keyword in [
        "format",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "multipleOf",
        "pattern",
    ] {
        assert!(warnings[0].message.contains(keyword), "{keyword}");
    }

    // The base dialect keeps them.
    let base = strict(&schema);
    assert_eq!(base["properties"]["code"]["pattern"], "^[A-Z]+$");
    assert_eq!(base["properties"]["count"]["maximum"], 9);
}

#[test]
fn azure_removes_its_unsupported_keywords() {
    let schema = json!({
        "type": "object",
        "properties": {
            "code": {"type": "string", "pattern": "^[A-Z]+$", "minLength": 2, "maxLength": 8, "format": "email"},
            "count": {"type": "integer", "format": "uint32", "minimum": 0, "maximum": 9, "multipleOf": 3},
            "tags": {
                "type": "array", "items": {"type": "string"},
                "minItems": 1, "maxItems": 4, "uniqueItems": true,
                "contains": {"const": "a"}, "minContains": 1, "maxContains": 2,
                "unevaluatedItems": false
            },
            "box": {
                "type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"],
                "minProperties": 1, "maxProperties": 3, "unevaluatedProperties": false
            }
        },
        "required": ["code", "count", "tags", "box"]
    });
    let (out, warnings) = run_with(&schema, SchemaDialect::AzureStrict);
    let out = out.unwrap();
    assert_eq!(out["properties"]["code"], json!({"type": "string"}));
    assert_eq!(out["properties"]["count"], json!({"type": "integer"}));
    assert_eq!(
        out["properties"]["tags"],
        json!({"type": "array", "items": {"type": "string"}})
    );
    assert_eq!(
        out["properties"]["box"],
        json!({
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
            "additionalProperties": false
        })
    );
    assert_eq!(out["additionalProperties"], false);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].kind, WarningKind::Approximated);
    assert!(warnings[0].message.contains("Azure"), "{}", warnings[0]);
    for keyword in [
        "contains",
        "format",
        "maxContains",
        "maxItems",
        "maxLength",
        "maxProperties",
        "maximum",
        "minContains",
        "minItems",
        "minLength",
        "minProperties",
        "minimum",
        "multipleOf",
        "pattern",
        "unevaluatedItems",
        "unevaluatedProperties",
        "uniqueItems",
    ] {
        assert!(warnings[0].message.contains(keyword), "{keyword}");
    }

    // A schema with nothing to remove goes through silently, and numeric
    // formats stay annotations dropped without a warning.
    let (out, warnings) = run_with(
        &json!({"type": "object", "properties": {"n": {"type": "number", "format": "double"}}}),
        SchemaDialect::AzureStrict,
    );
    assert_eq!(
        out.unwrap()["properties"]["n"],
        json!({"type": ["number", "null"]})
    );
    assert!(warnings.is_empty(), "{warnings:?}");
}

#[test]
fn azure_keeps_the_openai_rules() {
    let dialect = SchemaDialect::AzureStrict;
    assert_eq!(
        dialect.rejected_keywords(),
        SchemaDialect::OpenAiStrict.rejected_keywords()
    );
    for (schema, needle) in [
        // Maps, even through keywords on Azure's list.
        (
            json!({"type": "object", "patternProperties": {"^x-": {"type": "string"}}}),
            "patternProperties",
        ),
        (
            json!({"type": "object", "propertyNames": {"pattern": "^[a-z]+$"}}),
            "propertyNames",
        ),
        (
            json!({"type": "object", "additionalProperties": {"type": "string"}}),
            "additionalProperties",
        ),
        (
            json!({"type": "array", "items": {"type": "string"}}),
            "not an object",
        ),
        (
            json!({"type": "object", "properties": {"a": {"type": "string", "not": {"const": ""}}}}),
            "`not`",
        ),
    ] {
        let (out, warnings) = run_with(&schema, dialect);
        match out {
            Err(StrictSchemaError::NotRepresentable { reason, .. }) => {
                assert!(reason.contains(needle), "{reason}");
                assert!(reason.contains("Azure OpenAI strict mode"), "{reason}");
            }
            other => panic!("{schema}: {other:?}"),
        }
        assert!(warnings.is_empty());
    }
}

/// An object nesting `levels` objects deep, root included.
fn nested(levels: usize) -> Value {
    let mut schema = json!({"type": "object", "properties": {"leaf": {"type": "string"}}});
    for _ in 1..levels {
        schema = json!({"type": "object", "properties": {"child": schema}});
    }
    schema
}

#[test]
fn azure_limits_properties_and_nesting() {
    let dialect = SchemaDialect::AzureStrict;
    let wide = |n: usize| {
        let properties: Map<String, Value> = (0..n)
            .map(|i| (format!("p{i}"), json!({"type": "string"})))
            .collect();
        json!({"type": "object", "properties": properties})
    };
    assert!(run_with(&wide(100), dialect).0.is_ok());
    let (out, _) = run_with(&wide(101), dialect);
    assert!(
        matches!(&out, Err(StrictSchemaError::NotRepresentable { reason, .. }) if reason.contains("101 object properties")),
        "{out:?}"
    );
    // The base dialect has no such limit.
    assert!(run(&wide(101)).0.is_ok());

    // Properties in definitions count too.
    let mut split = wide(60);
    split["properties"]["d"] = json!({"$ref": "#/$defs/D"});
    split["$defs"] = json!({"D": wide(41)});
    let (out, _) = run_with(&split, dialect);
    assert!(
        matches!(&out, Err(StrictSchemaError::NotRepresentable { reason, .. }) if reason.contains("102 object properties")),
        "{out:?}"
    );

    assert!(run_with(&nested(5), dialect).0.is_ok());
    let (out, _) = run_with(&nested(6), dialect);
    assert!(
        matches!(&out, Err(StrictSchemaError::NotRepresentable { reason, .. }) if reason.contains("6 levels")),
        "{out:?}"
    );
    // Nesting counts through references and array items.
    let (out, _) = run_with(
        &json!({
            "type": "object",
            "properties": {"list": {"type": "array", "items": {"$ref": "#/$defs/Deep"}}},
            "$defs": {"Deep": nested(5)}
        }),
        dialect,
    );
    assert!(
        matches!(&out, Err(StrictSchemaError::NotRepresentable { reason, .. }) if reason.contains("6 levels")),
        "{out:?}"
    );
    // A recursive reference is not unrolled.
    let recursive = json!({
        "type": "object",
        "properties": {"children": {"type": "array", "items": {"$ref": "#"}}}
    });
    assert!(run_with(&recursive, dialect).0.is_ok());
}

#[test]
fn parallel_tool_calls_break_strict_on_fine_tuned_and_azure() {
    assert!(!SchemaDialect::OpenAiStrict.parallel_tool_calls_break_strict());
    assert!(SchemaDialect::OpenAiStrictFineTuned.parallel_tool_calls_break_strict());
    assert!(SchemaDialect::AzureStrict.parallel_tool_calls_break_strict());
    // Azure applies to every deployment: model names are not consulted.
    assert_eq!(
        SchemaDialect::AzureStrict.for_model("ft:gpt-4o-mini:org::id"),
        SchemaDialect::AzureStrict
    );
}

#[test]
fn the_fine_tuned_rule_is_opt_in() {
    let model = "ft:open-mistral-7b:acme:20240101:abc123";
    let mistral = crate::CompatQuirks::new("openai-compat:mistral");
    assert_eq!(
        mistral.schema_dialect_for(model),
        Some(SchemaDialect::OpenAiStrict)
    );
    let openai = crate::CompatQuirks::openai();
    assert_eq!(
        openai.schema_dialect_for(model),
        Some(SchemaDialect::OpenAiStrictFineTuned)
    );
    assert_eq!(
        openai.schema_dialect_for("gpt-4.1"),
        Some(SchemaDialect::OpenAiStrict)
    );
    assert_eq!(
        crate::ResponsesQuirks::new("openai-compat:demo").schema_dialect_for(model),
        Some(SchemaDialect::OpenAiStrict)
    );
    assert_eq!(
        crate::ResponsesQuirks::openai().schema_dialect_for(model),
        Some(SchemaDialect::OpenAiStrictFineTuned)
    );
}

#[test]
fn dialect_follows_fine_tuned_model_names() {
    assert_eq!(
        SchemaDialect::OpenAiStrict.for_model("ft:gpt-4o-mini:org::id"),
        SchemaDialect::OpenAiStrictFineTuned
    );
    assert_eq!(
        SchemaDialect::OpenAiStrict.for_model("gpt-4o-mini"),
        SchemaDialect::OpenAiStrict
    );
}

#[test]
fn maps_are_not_representable_in_every_shape() {
    // Schema-valued `additionalProperties` (a required map).
    assert_not_representable(
        &json!({
            "type": "object",
            "properties": {"labels": {"type": "object", "additionalProperties": {"type": "string"}}},
            "required": ["labels"]
        }),
        "/properties/labels",
    );
    // An optional map.
    assert_not_representable(
        &json!({
            "type": "object",
            "properties": {"labels": {"type": ["object", "null"], "additionalProperties": {"type": "string"}}}
        }),
        "additionalProperties",
    );
    // A map flattened alongside properties.
    assert_not_representable(
        &json!({
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "additionalProperties": {"type": "integer"}
        }),
        "additionalProperties",
    );
    assert_not_representable(
        &json!({"type": "object", "properties": {"id": {"type": "string"}}, "additionalProperties": true}),
        "additionalProperties",
    );
    assert_not_representable(
        &json!({"type": "object", "patternProperties": {"^x-": {"type": "string"}}}),
        "patternProperties",
    );
    assert_not_representable(
        &json!({"type": "object", "propertyNames": {"pattern": "^[a-z]+$"}}),
        "propertyNames",
    );
    // `serde_json::Value`: an unconstrained subschema.
    assert_not_representable(
        &json!({"type": "object", "properties": {"any": true}}),
        "/properties/any",
    );
    assert_not_representable(
        &json!({"type": "object", "properties": {"any": {"description": "Anything"}}}),
        "accepts any value",
    );
    assert_not_representable(
        &json!({"type": "object", "properties": {"list": {"type": "array", "items": {}}}}),
        "/properties/list/items",
    );
}

#[test]
fn maps_from_rust_types_are_detected() {
    #[derive(JsonSchema)]
    #[allow(dead_code)]
    struct Required {
        labels: HashMap<String, String>,
    }
    #[derive(JsonSchema)]
    #[allow(dead_code)]
    struct Optional {
        labels: Option<BTreeMap<String, u32>>,
    }
    #[derive(JsonSchema)]
    #[allow(dead_code)]
    struct Flattened {
        id: String,
        #[serde(flatten)]
        rest: HashMap<String, String>,
    }
    #[derive(JsonSchema)]
    #[allow(dead_code)]
    struct Untyped {
        payload: serde_json::Value,
    }
    #[derive(JsonSchema)]
    #[allow(dead_code)]
    struct Plain {
        name: String,
        tags: Vec<String>,
        nickname: Option<String>,
    }

    for schema in [
        allama::adapter::schema::schema_for::<Required>(),
        allama::adapter::schema::schema_for::<Optional>(),
        allama::adapter::schema::schema_for::<Flattened>(),
        allama::adapter::schema::schema_for::<Untyped>(),
        allama::adapter::schema::schema_for::<Vec<Plain>>(),
    ] {
        let (out, _) = run(&schema);
        assert!(
            matches!(out, Err(StrictSchemaError::NotRepresentable { .. })),
            "{schema}: {out:?}"
        );
    }
    let out = strict(&allama::adapter::schema::schema_for::<Plain>());
    assert_eq!(out["required"], json!(["name", "nickname", "tags"]));
}

#[test]
fn unresolved_sibling_refs_are_invalid() {
    let (out, _) = run(&json!({
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/Missing", "description": "gone"}}
    }));
    assert_eq!(
        out,
        Err(StrictSchemaError::Invalid(SchemaError::UnresolvedRef(
            "#/$defs/Missing".into()
        )))
    );
}

/// The bundled encoders' call: the dialect through the core policy.
fn encode(
    schema: &Value,
    dialect: Option<SchemaDialect>,
    d: &mut Degradations,
) -> allama::Result<StrictOutcome> {
    apply_strict_policy(
        schema,
        RequestField::Tools,
        "tool `f`",
        d,
        dialect_pass(dialect, true, RequestField::Tools, "tool `f`"),
        Degradable::Yes,
    )
}

#[test]
fn the_policy_degrades_or_fails_per_policy() {
    let map = json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": {"type": "string"}
    });
    let mut d = Degradations::new("demo", UnsupportedPolicy::Warn);
    let outcome = encode(&map, Some(SchemaDialect::OpenAiStrict), &mut d).unwrap();
    assert!(!outcome.strict);
    assert_eq!(
        outcome.schema,
        json!({"type": "object", "additionalProperties": {"type": "string"}})
    );
    let warnings = d.into_warnings();
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].kind, WarningKind::Unsupported);
    assert!(
        warnings[0]
            .message
            .ends_with("; it was sent without `strict`"),
        "{}",
        warnings[0]
    );

    let mut d = Degradations::new("demo", UnsupportedPolicy::Error);
    let err = encode(&map, Some(SchemaDialect::OpenAiStrict), &mut d).unwrap_err();
    assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");

    // A malformed schema fails under either policy.
    let broken = json!({"type": "object", "properties": {"a": {"$ref": "#/nope", "title": "A"}}});
    let mut d = Degradations::new("demo", UnsupportedPolicy::Warn);
    let err = encode(&broken, Some(SchemaDialect::OpenAiStrict), &mut d).unwrap_err();
    assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");

    // Without a dialect, the schema is sent as written, with a warning.
    let mut d = Degradations::new("demo", UnsupportedPolicy::Error);
    let outcome = encode(&map, None, &mut d).unwrap();
    assert!(outcome.strict);
    assert_eq!(outcome.schema, map);
    let warnings = d.into_warnings();
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].kind, WarningKind::Approximated);
}

#[test]
fn approximations_are_recorded_only_when_the_schema_goes_out_strict() {
    // A removed format beside a map: the rewrite fails, so the format
    // warning is not kept for a schema sent non-strict.
    let schema = json!({
        "type": "object",
        "properties": {
            "site": {"type": "string", "format": "uri"},
            "labels": {"type": "object", "additionalProperties": {"type": "string"}}
        }
    });
    let mut d = Degradations::new("demo", UnsupportedPolicy::Warn);
    let outcome = encode(&schema, Some(SchemaDialect::OpenAiStrict), &mut d).unwrap();
    assert!(!outcome.strict);
    assert_eq!(
        d.warnings().iter().map(|w| w.kind).collect::<Vec<_>>(),
        vec![WarningKind::Unsupported]
    );
}

/// OpenAI's root rule, case by case as probed live on 2026-10-13 (response
/// format and function parameters alike, strict or not).
#[test]
fn only_object_roots_pass_the_object_root_check() {
    let d = Degradations::new("demo", UnsupportedPolicy::Warn);
    let accepted = [
        json!({"type": "object"}),
        json!({"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "L", "type": "object", "properties": {}}),
        json!({}),
        json!({"$ref": "#/$defs/A", "$defs": {"A": {"type": "object", "properties": {"a": {"type": "string"}}}}}),
        // Unresolvable: the endpoint decides.
        json!({"$ref": "https://example.com/schema.json"}),
    ];
    for schema in accepted {
        require_object_root(
            &schema,
            RequestField::ResponseFormat,
            "the response schema",
            &d,
        )
        .unwrap_or_else(|err| panic!("{schema}: {err:?}"));
    }
    let rejected = [
        json!({"type": "string", "enum": ["billing", "bug"]}),
        json!({"properties": {"a": {"type": "string"}}}),
        json!({"description": "x"}),
        json!({"anyOf": [{"type": "object"}, {"type": "object"}]}),
        json!({"allOf": [{"type": "object"}]}),
        json!({"type": ["object", "null"]}),
        json!({"$ref": "#/$defs/A", "$defs": {"A": {"type": "string", "enum": ["a", "b"]}}}),
        json!(true),
    ];
    for schema in rejected {
        let err = require_object_root(&schema, RequestField::Tools, "tool `l`", &d).unwrap_err();
        assert_eq!(
            err.unsupported_field(),
            Some(RequestField::Tools),
            "{schema}"
        );
        assert_eq!(
            err.unsupported_stage(),
            Some(allama::UnsupportedStage::BeforeSending),
            "{schema}"
        );
        assert!(err.message().contains("tool `l`"), "{err}");
    }
}
