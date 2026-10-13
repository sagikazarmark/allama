//! OpenAI strict schemas (Chat Completions, `response_format`).
//!
//! - `properties` with a sibling `anyOf`: the `OpenAiStrict` dialect closes
//!   the parent object and every branch, which JSON Schema reads as
//!   unsatisfiable where the branches leave out the parent's properties.
//!   Is it accepted, and what does the model produce?
//! - An object root with properties plus a sibling `oneOf`, schemars'
//!   output for a `#[serde(flatten)]` tagged enum.
//! - Keywords OpenAI's Structured Outputs guide does not mention: are they
//!   accepted, rejected, or ignored?

use allama::RequestField;
use allama_http::{HttpRequest, ReqwestTransport};
use allama_openai_compat::schema::{SchemaDialect, strict_schema};
use schemars::JsonSchema;
use serde_json::{Value, json};

use super::{
    Exchange, dialect_record, env, env_or, honoured, keyword_case, parse_output, post_json, record,
    scratch, send,
};

const CHECK: &str = "openai";

/// The keywords OpenAI's guide neither lists as supported nor as
/// unsupported.
pub const UNDOCUMENTED_KEYWORDS: &[&str] = &[
    "uniqueItems",
    "contains",
    "prefixItems",
    "minProperties",
    "maxProperties",
    "unevaluatedProperties",
    "unevaluatedItems",
];

/// A Chat Completions request asking for `schema`, strict.
pub fn strict_request(model: &str, prompt: &str, schema: &Value) -> Value {
    json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 400,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "probe", "strict": true, "schema": schema},
        },
    })
}

/// The assistant text of a Chat Completions answer.
pub fn answer_text(exchange: &Exchange) -> Option<String> {
    exchange.json().and_then(|body| {
        body.pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .map(str::to_owned)
    })
}

fn request(body: &Value) -> HttpRequest {
    post_json(
        "https://api.openai.com/v1/chat/completions",
        ("authorization", format!("Bearer {}", env("OPENAI_API_KEY"))),
        body,
    )
}

/// Send `schema` strict and record the answer as `case`.
async fn probe(
    transport: &ReqwestTransport,
    case: &str,
    prompt: &str,
    schema: &Value,
    extra: Value,
) {
    let model = env_or("OPENAI_MODEL", "gpt-4.1-nano");
    let exchange = send(transport, request(&strict_request(&model, prompt, schema))).await;
    let output = parse_output(answer_text(&exchange));
    record(
        CHECK,
        case,
        &json!({
            "model": model,
            "prompt": prompt,
            "sent_schema": schema,
            "exchange": exchange.record(),
            "output": output,
            "notes": extra,
        }),
    );
}

/// Run the `OpenAiStrict` dialect on `input`, record what it made of it,
/// and send its output when it produced one.
async fn probe_allama(transport: &ReqwestTransport, case: &str, prompt: &str, input: &Value) {
    let mut degradations = scratch("openai");
    let result = strict_schema(
        input,
        RequestField::ResponseFormat,
        SchemaDialect::OpenAiStrict,
        &mut degradations,
    );
    let dialect = dialect_record(&result, &degradations);
    match result {
        Ok(schema) => {
            probe(
                transport,
                case,
                prompt,
                &schema,
                json!({"input_schema": input, "dialect": dialect}),
            )
            .await;
        }
        Err(_) => record(
            CHECK,
            case,
            &json!({"input_schema": input, "dialect": dialect}),
        ),
    }
}

#[tokio::test]
#[ignore]
async fn openai_properties_with_sibling_any_of() {
    let transport = ReqwestTransport::new();
    let prompt = "Describe a circle of radius 2 named \"wheel\".";
    // As the caller wrote it: shared properties next to the alternatives.
    let input = json!({
        "type": "object",
        "properties": {"shape": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "anyOf": [
                {"type": "object", "properties": {"radius": {"type": "number"}}, "required": ["radius"]},
                {"type": "object", "properties": {"side": {"type": "number"}}, "required": ["side"]}
            ]
        }},
        "required": ["shape"]
    });
    probe_allama(
        &transport,
        "properties_with_sibling_any_of_allama",
        prompt,
        &input,
    )
    .await;
    // Closed at both levels, by hand: the parent and every branch.
    let closed = json!({
        "type": "object",
        "properties": {"shape": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": false,
            "anyOf": [
                {"type": "object", "properties": {"radius": {"type": "number"}}, "required": ["radius"], "additionalProperties": false},
                {"type": "object", "properties": {"side": {"type": "number"}}, "required": ["side"], "additionalProperties": false}
            ]
        }},
        "required": ["shape"],
        "additionalProperties": false
    });
    probe(
        &transport,
        "properties_with_sibling_any_of_closed",
        prompt,
        &closed,
        json!("the parent and both branches closed by hand"),
    )
    .await;
    // The branches repeat the parent's properties, so they can be closed
    // without contradicting it.
    let repeated = json!({
        "type": "object",
        "properties": {"shape": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": false,
            "anyOf": [
                {"type": "object", "properties": {"name": {"type": "string"}, "radius": {"type": "number"}}, "required": ["name", "radius"], "additionalProperties": false},
                {"type": "object", "properties": {"name": {"type": "string"}, "side": {"type": "number"}}, "required": ["name", "side"], "additionalProperties": false}
            ]
        }},
        "required": ["shape"],
        "additionalProperties": false
    });
    probe(
        &transport,
        "properties_with_sibling_any_of_repeated",
        prompt,
        &repeated,
        json!("each branch repeats the parent's properties"),
    )
    .await;
}

/// schemars' output for a struct with a flattened, internally tagged enum:
/// the struct's properties beside a `oneOf` of the variants.
#[derive(JsonSchema)]
#[allow(dead_code)]
struct Event {
    id: String,
    #[serde(flatten)]
    kind: EventKind,
}

#[derive(JsonSchema)]
#[serde(tag = "type")]
#[allow(dead_code)]
enum EventKind {
    Click { x: i64, y: i64 },
    Key { key: String },
}

#[tokio::test]
#[ignore]
async fn openai_object_root_with_sibling_one_of() {
    let transport = ReqwestTransport::new();
    let prompt = "Record a click event with id \"e1\" at x=3, y=4.";
    let input = serde_json::to_value(schemars::schema_for!(Event)).unwrap();
    probe_allama(
        &transport,
        "object_root_with_sibling_one_of_allama",
        prompt,
        &input,
    )
    .await;
    // Closed by hand, keeping `oneOf` as written.
    let closed = json!({
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
        "additionalProperties": false,
        "oneOf": [
            {"type": "object", "properties": {"id": {"type": "string"}, "type": {"type": "string", "const": "Click"}, "x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["id", "type", "x", "y"], "additionalProperties": false},
            {"type": "object", "properties": {"id": {"type": "string"}, "type": {"type": "string", "const": "Key"}, "key": {"type": "string"}}, "required": ["id", "type", "key"], "additionalProperties": false}
        ]
    });
    probe(
        &transport,
        "object_root_with_sibling_one_of_closed",
        prompt,
        &closed,
        json!({"input_schema": input, "note": "closed by hand, `oneOf` kept"}),
    )
    .await;
}

#[tokio::test]
#[ignore]
async fn openai_undocumented_keywords() {
    let transport = ReqwestTransport::new();
    for keyword in UNDOCUMENTED_KEYWORDS {
        let case = keyword_case(keyword);
        let mut degradations = scratch("openai");
        let result = strict_schema(
            &case.schema,
            RequestField::ResponseFormat,
            SchemaDialect::OpenAiStrict,
            &mut degradations,
        );
        let model = env_or("OPENAI_MODEL", "gpt-4.1-nano");
        let exchange = send(
            &transport,
            request(&strict_request(&model, case.prompt, &case.schema)),
        )
        .await;
        let output = parse_output(answer_text(&exchange));
        record(
            CHECK,
            &format!("keyword_{}", case.keyword),
            &json!({
                "model": model,
                "keyword": case.keyword,
                "prompt": case.prompt,
                "sent_schema": case.schema,
                "exchange": exchange.record(),
                "output": output,
                "honoured": honoured(&case, &output),
                "allama_dialect": dialect_record(&result, &degradations),
            }),
        );
    }
}

/// The flattened enum case sends what it claims: properties beside a
/// `oneOf` at the root.
#[test]
fn the_flattened_enum_puts_one_of_beside_properties() {
    let schema = serde_json::to_value(schemars::schema_for!(Event)).unwrap();
    assert!(schema["properties"]["id"].is_object(), "{schema}");
    assert!(schema["oneOf"].is_array(), "{schema}");
}
