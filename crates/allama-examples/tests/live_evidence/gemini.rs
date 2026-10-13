//! Gemini `responseJsonSchema` (the adapter may report schemas enforced
//! that Gemini does not honour).
//!
//! - `oneOf` and `const`: is a `oneOf` of constants, and a tagged union of
//!   objects, enforced? A scalar `const` alone?
//! - Boolean subschemas (`true`, `false`).
//! - `$schema` at the root.
//!
//! Each schema is sent as written, so the record shows Gemini's own
//! behaviour, not the adapter's rewrite. The prompts ask for a value the
//! schema forbids, so an accepted schema also shows whether it is enforced.

use allama_http::ReqwestTransport;
use serde_json::{Value, json};

use super::{OutputCheck, env, env_or, parse_output, post_json, record, send};

const CHECK: &str = "gemini";

async fn probe(
    transport: &ReqwestTransport,
    case: &str,
    prompt: &str,
    schema: Value,
    honoured: OutputCheck,
) {
    let model = env_or("GEMINI_MODEL", "gemini-3.5-flash-lite");
    let body = json!({
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": schema,
        },
    });
    let request = post_json(
        &format!("https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"),
        ("x-goog-api-key", env("GEMINI_API_KEY")),
        &body,
    );
    let exchange = send(transport, request).await;
    // The answer's text parts, without thought summaries.
    let text = exchange.json().and_then(|body| {
        let parts = body
            .pointer("/candidates/0/content/parts")?
            .as_array()?
            .clone();
        let text: String = parts
            .iter()
            .filter(|part| part.get("thought").and_then(Value::as_bool) != Some(true))
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect();
        (!text.is_empty()).then_some(text)
    });
    let output = parse_output(text);
    let kept = (!output.is_null()).then(|| honoured(&output));
    record(
        CHECK,
        case,
        &json!({
            "model": model,
            "prompt": prompt,
            "sent_schema": schema,
            "exchange": exchange.record(),
            "output": output,
            "honoured": kept,
        }),
    );
}

#[tokio::test]
#[ignore]
async fn gemini_one_of_and_const() {
    let transport = ReqwestTransport::new();
    probe(
        &transport,
        "one_of_constants",
        "Set shape to \"triangle\".",
        json!({
            "type": "object",
            "properties": {"shape": {"oneOf": [{"const": "circle"}, {"const": "square"}]}},
            "required": ["shape"]
        }),
        |o| {
            matches!(
                o.get("shape").and_then(Value::as_str),
                Some("circle" | "square")
            )
        },
    )
    .await;
    probe(
        &transport,
        "one_of_tagged_objects",
        "Record a scroll event with type \"scroll\" and delta 5.",
        json!({
            "type": "object",
            "properties": {"event": {"oneOf": [
                {"type": "object", "properties": {"type": {"const": "click"}, "x": {"type": "integer"}}, "required": ["type", "x"]},
                {"type": "object", "properties": {"type": {"const": "key"}, "key": {"type": "string"}}, "required": ["type", "key"]}
            ]}},
            "required": ["event"]
        }),
        |o| {
            matches!(
                o.pointer("/event/type").and_then(Value::as_str),
                Some("click" | "key")
            )
        },
    )
    .await;
    probe(
        &transport,
        "const_alone",
        "Set kind to \"square\".",
        json!({
            "type": "object",
            "properties": {"kind": {"type": "string", "const": "circle"}},
            "required": ["kind"]
        }),
        |o| o.get("kind").and_then(Value::as_str) == Some("circle"),
    )
    .await;
}

#[tokio::test]
#[ignore]
async fn gemini_boolean_subschemas() {
    let transport = ReqwestTransport::new();
    probe(
        &transport,
        "boolean_true_property",
        "Set name to \"Ada\" and extra to {\"note\": \"anything\"}.",
        json!({
            "type": "object",
            "properties": {"name": {"type": "string"}, "extra": true},
            "required": ["name", "extra"]
        }),
        |o| o.get("name").is_some(),
    )
    .await;
    probe(
        &transport,
        "boolean_false_property",
        "Set name to \"Ada\" and forbidden to 1.",
        json!({
            "type": "object",
            "properties": {"name": {"type": "string"}, "forbidden": false},
            "required": ["name"]
        }),
        |o| o.get("forbidden").is_none(),
    )
    .await;
    probe(
        &transport,
        "boolean_true_items",
        "Set values to [1, \"two\", true].",
        json!({
            "type": "object",
            "properties": {"values": {"type": "array", "items": true}},
            "required": ["values"]
        }),
        |o| o.get("values").is_some_and(Value::is_array),
    )
    .await;
}

#[tokio::test]
#[ignore]
async fn gemini_dollar_schema() {
    let transport = ReqwestTransport::new();
    for (case, dialect) in [
        (
            "dollar_schema_2020_12",
            "https://json-schema.org/draft/2020-12/schema",
        ),
        (
            "dollar_schema_draft_07",
            "http://json-schema.org/draft-07/schema#",
        ),
    ] {
        probe(
            &transport,
            case,
            "Set name to \"Ada\".",
            json!({
                "$schema": dialect,
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            }),
            |o| o.get("name").and_then(Value::as_str).is_some(),
        )
        .await;
    }
}
