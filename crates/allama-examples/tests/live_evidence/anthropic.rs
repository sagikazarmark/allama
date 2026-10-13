//! Anthropic strict schemas (`output_config.format` and strict tools).
//!
//! - Numeric `format`s (`uint32`, `double`, …), which the strict pass drops
//!   as annotations: does Anthropic accept them as written? The adapter's
//!   side runs in `allama-anthropic`'s `anthropic_strict_numeric_fields`.
//! - How Anthropic counts optional properties against its per-request
//!   limit of 24: top-level only, nested ones too, and a `$ref` once per
//!   definition or once per use. The adapter counts every property, nested
//!   ones and each `$ref` use included, which is conservative.
//! - A `$ref` with sibling keywords, as written and after
//!   `merge_sibling_refs` (the strict pass's output).

use allama::RequestField;
use allama_anthropic::schema::strict_schema;
use allama_http::{HttpRequest, ReqwestTransport};
use schemars::JsonSchema;
use serde_json::{Map, Value, json};

use super::{dialect_record, env, env_or, parse_output, post_json, record, scratch, send};

const CHECK: &str = "anthropic";

/// The limit Anthropic documents for optional parameters across a
/// request's strict schemas.
const OPTIONAL_LIMIT: usize = 24;

fn request(body: &Value) -> HttpRequest {
    post_json(
        "https://api.anthropic.com/v1/messages",
        ("x-api-key", env("ANTHROPIC_API_KEY")),
        body,
    )
    .header("anthropic-version", "2023-06-01")
}

fn model() -> String {
    env_or("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
}

/// A request with `schema` as the strict output format.
fn output_format_request(prompt: &str, schema: &Value) -> Value {
    json!({
        "model": model(),
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {"format": {"type": "json_schema", "schema": schema}},
    })
}

/// A request that forces one strict tool, `probe`, with `schema` as its
/// input schema.
fn strict_tool_request(prompt: &str, schema: &Value) -> Value {
    json!({
        "model": model(),
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [{
            "name": "probe",
            "description": "Record the values.",
            "input_schema": schema,
            "strict": true,
        }],
        "tool_choice": {"type": "tool", "name": "probe"},
    })
}

/// The model's answer: the text of an output format answer, or the input of
/// the first tool call.
fn answer(body: Option<Value>) -> Value {
    let Some(content) = body.and_then(|b| b.get("content").cloned()) else {
        return Value::Null;
    };
    let blocks = content.as_array().cloned().unwrap_or_default();
    if let Some(input) = blocks
        .iter()
        .find(|b| b["type"] == "tool_use")
        .map(|b| b["input"].clone())
    {
        return input;
    }
    let text: String = blocks
        .iter()
        .filter(|b| b["type"] == "text")
        .filter_map(|b| b["text"].as_str())
        .collect();
    parse_output((!text.is_empty()).then_some(text))
}

async fn probe(transport: &ReqwestTransport, case: &str, body: Value, notes: Value) {
    let exchange = send(transport, request(&body)).await;
    record(
        CHECK,
        case,
        &json!({
            "model": model(),
            "sent": body,
            "exchange": exchange.record(),
            "output": answer(exchange.json()),
            "notes": notes,
        }),
    );
}

#[tokio::test]
#[ignore]
async fn anthropic_numeric_formats() {
    let transport = ReqwestTransport::new();
    let formats = [
        ("integer", "int32"),
        ("integer", "int64"),
        ("integer", "uint8"),
        ("integer", "uint32"),
        ("integer", "uint64"),
        ("number", "float"),
        ("number", "double"),
    ];
    for (kind, format) in formats {
        let schema = json!({
            "type": "object",
            "properties": {"value": {"type": kind, "format": format}},
            "required": ["value"],
            "additionalProperties": false
        });
        probe(
            &transport,
            &format!("numeric_format_{format}_output_format"),
            output_format_request("Set value to 3.", &schema),
            json!({"format": format}),
        )
        .await;
    }
    // Every format at once, in a strict tool.
    let properties: Map<String, Value> = formats
        .iter()
        .map(|(kind, format)| (format.to_string(), json!({"type": kind, "format": format})))
        .collect();
    let required: Vec<&str> = formats.iter().map(|(_, format)| *format).collect();
    let schema = json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false
    });
    probe(
        &transport,
        "numeric_formats_strict_tool",
        strict_tool_request("Set every value to 3.", &schema),
        json!("every numeric format in one strict tool"),
    )
    .await;
}

/// A closed object with `required` required and `optional` optional string
/// properties, plus `extra` required properties.
fn object(required: usize, optional: usize, extra: &[(&str, Value)]) -> Value {
    let mut properties = Map::new();
    let mut names = Vec::new();
    for i in 0..required {
        properties.insert(format!("r{i}"), json!({"type": "string"}));
        names.push(format!("r{i}"));
    }
    for i in 0..optional {
        properties.insert(format!("o{i}"), json!({"type": "string"}));
    }
    for (name, schema) in extra {
        properties.insert((*name).to_owned(), schema.clone());
        names.push((*name).to_owned());
    }
    json!({
        "type": "object",
        "properties": properties,
        "required": names,
        "additionalProperties": false
    })
}

#[tokio::test]
#[ignore]
async fn anthropic_nested_optional_counting() {
    let transport = ReqwestTransport::new();
    let prompt = "Call the tool with r0 set to \"a\"; leave the optional values out.";
    let item = object(0, 13, &[]);
    let mut ref_twice = object(
        1,
        0,
        &[
            ("first", json!({"$ref": "#/$defs/Item"})),
            ("second", json!({"$ref": "#/$defs/Item"})),
        ],
    );
    ref_twice["$defs"] = json!({"Item": item});
    let cases = [
        (
            "top_level_at_limit",
            object(1, OPTIONAL_LIMIT, &[]),
            json!({"top_level_optional": OPTIONAL_LIMIT, "nested_optional": 0}),
        ),
        (
            "top_level_over_limit",
            object(1, OPTIONAL_LIMIT + 1, &[]),
            json!({"top_level_optional": OPTIONAL_LIMIT + 1, "nested_optional": 0}),
        ),
        (
            "nested_over_limit_in_total",
            object(1, 4, &[("nested", object(0, 21, &[]))]),
            json!({"top_level_optional": 4, "nested_optional": 21}),
        ),
        (
            "nested_only_over_limit",
            object(1, 0, &[("nested", object(0, OPTIONAL_LIMIT + 1, &[]))]),
            json!({"top_level_optional": 0, "nested_optional": OPTIONAL_LIMIT + 1}),
        ),
        (
            "two_levels_over_limit",
            object(
                1,
                0,
                &[("outer", object(0, 10, &[("inner", object(0, 15, &[]))]))],
            ),
            json!({"top_level_optional": 0, "nested_optional": 25, "levels": 2}),
        ),
        (
            "ref_used_twice",
            ref_twice,
            json!({
                "top_level_optional": 0,
                "optional_per_definition": 13,
                "optional_per_use": 26,
            }),
        ),
    ];
    for (case, schema, counts) in cases {
        probe(
            &transport,
            &format!("optional_counting_{case}"),
            strict_tool_request(prompt, &schema),
            json!({"limit": OPTIONAL_LIMIT, "counts": counts}),
        )
        .await;
    }
}

/// schemars writes a `$ref` with sibling keywords for an internally tagged
/// newtype variant: the tag property beside a `$ref` to the struct.
#[derive(JsonSchema)]
#[allow(dead_code)]
struct Order {
    /// How the order is paid.
    payment: Payment,
}

#[derive(JsonSchema)]
#[serde(tag = "type")]
#[allow(dead_code)]
enum Payment {
    Card(Card),
    Cash(Cash),
}

#[derive(JsonSchema)]
#[allow(dead_code)]
struct Card {
    number: String,
}

#[derive(JsonSchema)]
#[allow(dead_code)]
struct Cash {
    amount: f64,
}

#[tokio::test]
#[ignore]
async fn anthropic_ref_with_siblings() {
    let transport = ReqwestTransport::new();
    let prompt = "The owner is Ada.";
    // As written, closed by hand: a `$ref` with a `description` beside it.
    let as_written = json!({
        "type": "object",
        "properties": {"owner": {"$ref": "#/$defs/Person", "description": "Who owns it"}},
        "required": ["owner"],
        "additionalProperties": false,
        "$defs": {"Person": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": false
        }}
    });
    probe(
        &transport,
        "ref_with_siblings_as_written_output_format",
        output_format_request(prompt, &as_written),
        json!("a `$ref` with a sibling `description`, not merged"),
    )
    .await;
    probe(
        &transport,
        "ref_with_siblings_as_written_strict_tool",
        strict_tool_request(prompt, &as_written),
        json!("a `$ref` with a sibling `description`, not merged"),
    )
    .await;

    // After the merge: what the strict pass sends for the same schema, and
    // for schemars' tagged newtype variants.
    let tagged = serde_json::to_value(schemars::schema_for!(Order)).unwrap();
    for (name, input, prompt) in [
        ("described_ref", as_written.clone(), prompt),
        (
            "tagged_newtype_variants",
            tagged,
            "Pay by card, number 4242.",
        ),
    ] {
        for (field, label) in [
            (RequestField::ResponseFormat, "output_format"),
            (RequestField::Tools, "strict_tool"),
        ] {
            let mut degradations = scratch("anthropic");
            let result = strict_schema(&input, field, &mut degradations);
            let dialect = dialect_record(&result, &degradations);
            let case = format!("ref_with_siblings_merged_{name}_{label}");
            let Ok(schema) = result else {
                record(
                    CHECK,
                    &case,
                    &json!({"input_schema": input, "dialect": dialect}),
                );
                continue;
            };
            let body = match field {
                RequestField::ResponseFormat => output_format_request(prompt, &schema),
                _ => strict_tool_request(prompt, &schema),
            };
            probe(
                &transport,
                &case,
                body,
                json!({"input_schema": input, "dialect": dialect}),
            )
            .await;
        }
    }
}

/// The tagged newtype case sends what it claims: a `$ref` with sibling
/// keywords, which the strict pass merges into one object.
#[test]
fn the_tagged_newtype_variants_put_siblings_beside_a_ref() {
    let schema = serde_json::to_value(schemars::schema_for!(Order)).unwrap();
    let text = schema.to_string();
    let variant = schema["$defs"]["Payment"]["oneOf"][0].clone();
    assert!(variant["$ref"].is_string(), "{text}");
    assert!(variant["properties"]["type"].is_object(), "{text}");

    let mut degradations = scratch("anthropic");
    let merged = strict_schema(&schema, RequestField::ResponseFormat, &mut degradations).unwrap();
    assert!(!merged.to_string().contains("allOf"), "{merged}");
}
