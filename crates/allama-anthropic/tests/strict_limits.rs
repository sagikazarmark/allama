//! Anthropic's documented strict-schema limits, through the real adapter:
//! numeric `format`s are dropped silently and unsupported string formats
//! removed with a warning; per request, at most 20 strict tools, 24 optional
//! parameters and 16 union-typed parameters across the strict tools and the
//! output format. Over a limit, strict tools degrade from the last one, and
//! an output format that alone exceeds a limit fails under every policy.

// The fixture types are only deserialized into, never read.
#![allow(dead_code)]

use allama::prelude::*;
use allama::schemars::JsonSchema;
use allama::{RequestField, Subject, UnsupportedPolicy, Warning, WarningKind};
use allama_anthropic::Provider;
use allama_http::{ReplayResponse, ReplayTransport};
use serde_json::{Map, Value, json};

const MODEL: &str = "claude-haiku-4-5-20251001";

fn ok() -> ReplayResponse {
    ReplayResponse::json(
        200,
        json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": MODEL,
            "content": [{"type": "text", "text": "{}"}],
            "stop_reason": "end_turn", "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })
        .to_string(),
    )
}

async fn send(request: &ChatRequest) -> (Result<ChatResponse>, Vec<Value>) {
    let transport = ReplayTransport::new([ok()]);
    let provider = Provider::builder()
        .api_key("k")
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let result = provider.chat(request).await;
    let bodies = transport
        .requests()
        .iter()
        .filter_map(|r| r.body.as_ref())
        .map(|body| serde_json::from_slice(body).expect("JSON request body"))
        .collect();
    (result, bodies)
}

fn schema<T: JsonSchema>() -> Value {
    allama::adapter::schema::schema_for::<T>()
}

fn output(schema: Value) -> ResponseFormat {
    ResponseFormat::json_schema(schema).name("out")
}

fn field_warnings(warnings: &[Warning], field: RequestField) -> Vec<&Warning> {
    warnings
        .iter()
        .filter(|w| w.subject == Subject::Field { field })
        .collect()
}

/// An object with `n` optional, union-typed properties, as schemars writes
/// `Option<String>` fields.
fn options(prefix: &str, n: usize) -> Value {
    let properties: Map<String, Value> = (0..n)
        .map(|i| (format!("{prefix}{i}"), json!({"type": ["string", "null"]})))
        .collect();
    json!({"type": "object", "properties": properties})
}

#[derive(JsonSchema)]
#[schemars(crate = "allama::schemars")]
struct Measurement {
    count: u32,
    ratio: f64,
    total: i64,
}

#[tokio::test]
async fn numeric_formats_are_dropped_silently() {
    let request = ChatRequest::new(MODEL)
        .user("Measure.")
        .response_format(output(schema::<Measurement>()));
    let (result, sent) = send(&request).await;
    let response = result.unwrap();
    let properties = &sent[0]["output_config"]["format"]["schema"]["properties"];
    for name in ["count", "ratio", "total"] {
        assert!(
            properties[name].get("format").is_none(),
            "{name}: {properties}"
        );
    }
    assert_eq!(properties["ratio"], json!({"type": "number"}));
    // `u32`'s `minimum: 0` is a numeric constraint Anthropic rejects; the
    // dropped `format`s are not reported.
    for warning in &response.warnings {
        assert!(!warning.message.contains("format"), "{warning}");
        assert!(!warning.message.contains("uint32"), "{warning}");
    }
}

#[tokio::test]
async fn unsupported_string_formats_are_stripped_with_a_warning() {
    let schema = json!({
        "type": "object",
        "properties": {
            "homepage": {"type": "string", "format": "uri"},
            "pattern": {"type": "string", "format": "regex"},
            "seen": {"type": ["string", "null"], "format": "date-time"}
        },
        "required": ["homepage", "pattern"]
    });
    let request = ChatRequest::new(MODEL)
        .user("Describe.")
        .response_format(output(schema.clone()))
        .tools([ToolDefinition::new("describe", schema).strict(true)]);
    let (result, sent) = send(&request).await;
    let response = result.unwrap();
    for sent_schema in [
        &sent[0]["output_config"]["format"]["schema"],
        &sent[0]["tools"][0]["input_schema"],
    ] {
        let properties = &sent_schema["properties"];
        assert_eq!(properties["homepage"]["format"], "uri");
        assert_eq!(properties["seen"]["format"], "date-time");
        assert_eq!(properties["pattern"], json!({"type": "string"}));
    }
    assert_eq!(sent[0]["tools"][0]["strict"], true);
    for field in [RequestField::ResponseFormat, RequestField::Tools] {
        let warnings = field_warnings(&response.warnings, field);
        assert_eq!(warnings.len(), 1, "{field}: {warnings:?}");
        assert_eq!(warnings[0].kind, WarningKind::Approximated);
        assert!(warnings[0].message.contains("regex"), "{}", warnings[0]);
        assert!(!warnings[0].message.contains("uri"), "{}", warnings[0]);
    }
}

#[tokio::test]
async fn seventeen_option_fields_across_two_tools_degrade_the_last_tool() {
    let request = ChatRequest::new(MODEL).user("Fill in.").tools([
        ToolDefinition::new("first", options("a", 9)).strict(true),
        ToolDefinition::new("second", options("b", 8)).strict(true),
    ]);
    let (result, sent) = send(&request).await;
    let response = result.unwrap();
    let tools = &sent[0]["tools"];
    assert_eq!(tools[0]["strict"], true);
    assert_eq!(tools[0]["input_schema"]["additionalProperties"], false);
    // The last tool goes out as the caller wrote it.
    assert!(tools[1].get("strict").is_none(), "{}", tools[1]);
    assert_eq!(tools[1]["input_schema"], options("b", 8));

    let warnings = field_warnings(&response.warnings, RequestField::Tools);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].kind, WarningKind::Unsupported);
    assert!(warnings[0].message.contains("`second`"), "{}", warnings[0]);
    assert!(
        warnings[0]
            .message
            .contains("17 union-typed parameters (at most 16)"),
        "{}",
        warnings[0]
    );

    // Under `Error`, any overrun fails before sending.
    let strict = request.unsupported_policy(UnsupportedPolicy::Error);
    let (result, sent) = send(&strict).await;
    let error = result.unwrap_err();
    assert!(matches!(error, Error::Unsupported { .. }), "{error:?}");
    assert!(sent.is_empty(), "nothing is sent");
}

#[tokio::test]
async fn optional_parameters_are_limited_to_twenty_four() {
    // Required unions do not count as optional, so only the optional limit
    // is reached: 13 + 12 optional, non-union properties.
    let plain = |prefix: &str, n: usize| {
        let properties: Map<String, Value> = (0..n)
            .map(|i| (format!("{prefix}{i}"), json!({"type": "string"})))
            .collect();
        json!({"type": "object", "properties": properties})
    };
    let request = ChatRequest::new(MODEL).user("Fill in.").tools([
        ToolDefinition::new("first", plain("a", 13)).strict(true),
        ToolDefinition::new("second", plain("b", 12)).strict(true),
    ]);
    let (result, sent) = send(&request).await;
    let response = result.unwrap();
    assert_eq!(sent[0]["tools"][0]["strict"], true);
    assert!(sent[0]["tools"][1].get("strict").is_none());
    let warnings = field_warnings(&response.warnings, RequestField::Tools);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(
        warnings[0]
            .message
            .contains("25 optional parameters (at most 24)"),
        "{}",
        warnings[0]
    );
}

#[tokio::test]
async fn twenty_one_strict_tools_degrade_the_last() {
    let tools: Vec<ToolDefinition> = (0..21)
        .map(|i| {
            ToolDefinition::new(
                format!("tool_{i}"),
                json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}),
            )
            .strict(true)
        })
        .collect();
    let request = ChatRequest::new(MODEL).user("Pick one.").tools(tools);
    let (result, sent) = send(&request).await;
    let response = result.unwrap();
    let sent_tools = sent[0]["tools"].as_array().unwrap();
    assert_eq!(sent_tools.len(), 21);
    for tool in &sent_tools[..20] {
        assert_eq!(tool["strict"], true, "{tool}");
    }
    assert!(sent_tools[20].get("strict").is_none(), "{}", sent_tools[20]);
    let warnings = field_warnings(&response.warnings, RequestField::Tools);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].kind, WarningKind::Unsupported);
    assert!(warnings[0].message.contains("`tool_20`"), "{}", warnings[0]);
    assert!(
        warnings[0].message.contains("21 strict tools (at most 20)"),
        "{}",
        warnings[0]
    );
}

#[tokio::test]
async fn an_output_format_over_a_limit_fails_under_every_policy() {
    let request = ChatRequest::new(MODEL)
        .user("Fill in.")
        .response_format(output(options("o", 17)));
    for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
        let (result, sent) = send(&request.clone().unsupported_policy(policy)).await;
        let error = result.unwrap_err();
        assert_eq!(
            error.unsupported_field(),
            Some(RequestField::ResponseFormat),
            "{policy:?}: {error:?}"
        );
        assert!(error.to_string().contains("17 union-typed"), "{error}");
        assert!(sent.is_empty(), "{policy:?}: nothing is sent");
    }
}

#[tokio::test]
async fn the_output_format_takes_its_share_before_the_tools() {
    // The output format fits alone (10 unions) but leaves room for 6 more:
    // the tool with 7 degrades, the output format stays strict.
    let request = ChatRequest::new(MODEL)
        .user("Fill in.")
        .response_format(output(options("o", 10)))
        .tools([ToolDefinition::new("extra", options("t", 7)).strict(true)]);
    let (result, sent) = send(&request).await;
    let response = result.unwrap();
    assert_eq!(
        sent[0]["output_config"]["format"]["schema"]["additionalProperties"],
        false
    );
    assert!(sent[0]["tools"][0].get("strict").is_none());
    let warnings = field_warnings(&response.warnings, RequestField::Tools);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].message.contains("`extra`"), "{}", warnings[0]);
}

#[derive(JsonSchema)]
#[schemars(crate = "allama::schemars")]
struct Inner {
    a: u32,
    b: String,
}

#[derive(JsonSchema)]
#[schemars(crate = "allama::schemars")]
#[serde(tag = "kind")]
enum Shape {
    Unit,
    New(Inner),
}

#[derive(JsonSchema)]
#[schemars(crate = "allama::schemars")]
struct Drawing {
    shape: Shape,
}

#[tokio::test]
async fn an_internally_tagged_newtype_variant_keeps_its_fields() {
    let request = ChatRequest::new(MODEL)
        .user("Draw.")
        .response_format(output(schema::<Drawing>()))
        .tools([ToolDefinition::new("draw", schema::<Drawing>()).strict(true)]);
    let (result, sent) = send(&request).await;
    result.unwrap();
    for sent_schema in [
        &sent[0]["output_config"]["format"]["schema"],
        &sent[0]["tools"][0]["input_schema"],
    ] {
        let text = sent_schema.to_string();
        assert!(!text.contains("allOf"), "{text}");
        let variants = sent_schema["$defs"]["Shape"]["anyOf"]
            .as_array()
            .unwrap_or_else(|| panic!("{text}"));
        let new = variants
            .iter()
            .find(|v| v["properties"]["kind"]["const"] == "New")
            .unwrap_or_else(|| panic!("{text}"));
        assert!(new.get("$ref").is_none(), "{new}");
        for field in ["a", "b", "kind"] {
            assert!(new["properties"].get(field).is_some(), "{field}: {new}");
        }
        assert_eq!(new["additionalProperties"], false);
    }
    assert_eq!(sent[0]["tools"][0]["strict"], true);
}

/// Five optional, union-typed fields.
#[derive(JsonSchema)]
#[schemars(crate = "allama::schemars")]
struct Contact {
    email: Option<String>,
    phone: Option<String>,
    fax: Option<String>,
    street: Option<String>,
    city: Option<String>,
}

#[derive(JsonSchema)]
#[schemars(crate = "allama::schemars")]
struct Parties {
    /// Who sends it.
    sender: Contact,
    /// Who receives it.
    recipient: Contact,
}

#[tokio::test]
async fn a_struct_used_in_two_places_counts_once_per_use_site() {
    // The merge flattens `Contact` into each documented use site, so the
    // grammar sees ten unions, not five; with seven more on the tool the
    // request has 17 and the tool degrades.
    let request = ChatRequest::new(MODEL)
        .user("Who?")
        .response_format(output(schema::<Parties>()))
        .tools([ToolDefinition::new("extra", options("t", 7)).strict(true)]);
    let (result, sent) = send(&request).await;
    let response = result.unwrap();
    let out = &sent[0]["output_config"]["format"]["schema"];
    for site in ["sender", "recipient"] {
        assert_eq!(
            out["properties"][site]["properties"]["email"]["type"],
            json!(["string", "null"])
        );
    }
    assert!(sent[0]["tools"][0].get("strict").is_none());
    let warnings = field_warnings(&response.warnings, RequestField::Tools);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(
        warnings[0].message.contains("17 union-typed parameters"),
        "{}",
        warnings[0]
    );

    // Six more fit.
    let request = ChatRequest::new(MODEL)
        .user("Who?")
        .response_format(output(schema::<Parties>()))
        .tools([ToolDefinition::new("extra", options("t", 6)).strict(true)]);
    let (result, sent) = send(&request).await;
    assert!(result.unwrap().warnings.is_empty());
    assert_eq!(sent[0]["tools"][0]["strict"], true);
}
