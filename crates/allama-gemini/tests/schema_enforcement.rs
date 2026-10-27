//! Extraction reports what was enforced (`SchemaEnforcement`) through the
//! Gemini adapter's native structured output. Gemini has no strict mode, so
//! constructs outside the documented `responseJsonSchema` subset are sent
//! and reported `Approximate`; the enum shapes schemars writes stay `Exact`
//! after the sibling `$ref` merge and the scalar `const` rewrite.

// The fixture types are only deserialized into, never read.
#![allow(dead_code)]

use allama::chat::extract::{
    ChatExtractExt, ChatExtractionMetadata, ChatExtractionMode, SchemaEnforcement,
};
use allama::prelude::*;
use allama::schemars::JsonSchema;
use allama::{RequestField, Subject, WarningKind};
use allama_gemini::Provider;
use allama_http::{ReplayResponse, ReplayTransport};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

const MODEL: &str = "gemini-2.5-flash";

#[derive(Clone, Copy, Debug)]
enum Target {
    Gemini,
}

impl Target {
    fn model(self) -> &'static str {
        MODEL
    }

    fn provider(self, transport: &ReplayTransport) -> Provider {
        Provider::builder()
            .api_key("k")
            .http_transport(transport.clone())
            .build()
            .unwrap()
    }

    fn text(self, text: &str) -> ReplayResponse {
        ReplayResponse::json(
            200,
            json!({
                "candidates": [{"content": {"role": "model", "parts": [{"text": text}]},
                                "finishReason": "STOP", "index": 0}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
                "modelVersion": MODEL
            })
            .to_string(),
        )
    }

    fn tool_call(self, arguments: &str) -> ReplayResponse {
        let args: Value = serde_json::from_str(arguments).unwrap();
        ReplayResponse::json(
            200,
            json!({
                "candidates": [{"content": {"role": "model", "parts": [
                                    {"functionCall": {"name": "submit_structured_output", "args": args}}]},
                                "finishReason": "STOP", "index": 0}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
                "modelVersion": MODEL
            })
            .to_string(),
        )
    }
}

// The enum shapes schemars writes, each wrapped in an object (OpenAI strict
// mode requires an object root). Every shape must extract `Exact`.

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "allama::schemars")]
struct Inner {
    inner_field: i64,
    note: Option<String>,
}

/// A documented unit enum.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "allama::schemars")]
enum Documented {
    /// The first.
    First,
    /// The second.
    Second,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "allama::schemars")]
enum External {
    Unit,
    New(Inner),
    Struct { x: i64 },
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "allama::schemars")]
#[serde(tag = "kind")]
enum Internal {
    Unit,
    New(Inner),
    Struct { x: i64 },
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "allama::schemars")]
#[serde(tag = "t", content = "c")]
enum Adjacent {
    Unit,
    New(Inner),
    Struct { x: i64 },
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "allama::schemars")]
enum Mixed {
    Unit,
    /// A documented unit variant.
    Other,
    Struct {
        x: i64,
    },
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "allama::schemars")]
#[serde(untagged)]
enum Untagged {
    Number(i64),
    Text(String),
}

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(crate = "allama::schemars")]
struct Wrapped<T> {
    v: T,
}

/// A number whose schema is a hand-written `oneOf` over overlapping
/// `integer` and `number` branches: `2` matches both.
#[derive(Debug, Deserialize)]
struct Overlapping(f64);

impl JsonSchema for Overlapping {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Overlapping".into()
    }

    fn json_schema(_: &mut allama::schemars::SchemaGenerator) -> allama::schemars::Schema {
        allama::schemars::json_schema!({"oneOf": [{"type": "integer"}, {"type": "number"}]})
    }
}

/// Every shape with an answer that deserializes into it.
fn exact_shapes() -> Vec<(&'static str, Value)> {
    vec![
        ("documented unit enum", json!({"v": "First"})),
        (
            "externally tagged",
            json!({"v": {"New": {"inner_field": 1, "note": null}}}),
        ),
        (
            "internally tagged",
            json!({"v": {"kind": "New", "inner_field": 1, "note": null}}),
        ),
        (
            "adjacently tagged",
            json!({"v": {"t": "New", "c": {"inner_field": 1, "note": null}}}),
        ),
        ("mixed unit and struct", json!({"v": {"Struct": {"x": 2}}})),
        ("untagged", json!({"v": "text"})),
    ]
}

/// Whether some object schema under `value` declares every one of `keys`
/// as a property.
fn declares(value: &Value, keys: &[&str]) -> bool {
    match value {
        Value::Object(map) => {
            map.get("properties")
                .and_then(Value::as_object)
                .is_some_and(|p| keys.iter().all(|k| p.contains_key(*k)))
                || map.values().any(|v| declares(v, keys))
        }
        Value::Array(items) => items.iter().any(|v| declares(v, keys)),
        _ => false,
    }
}

/// Whether a `$ref` with sibling keywords appears under `value`.
fn ref_with_siblings(value: &Value) -> bool {
    match value {
        Value::Object(map) => {
            (map.contains_key("$ref") && map.len() > 1) || map.values().any(ref_with_siblings)
        }
        Value::Array(items) => items.iter().any(ref_with_siblings),
        _ => false,
    }
}

fn bodies(transport: &ReplayTransport) -> Vec<Value> {
    transport
        .requests()
        .iter()
        .filter_map(|r| r.body.as_ref())
        .map(|body| serde_json::from_slice(body).expect("JSON request body"))
        .collect()
}

/// Extract the shape named `name` from `answer` and return its metadata and
/// the body sent.
async fn run_shape(
    target: Target,
    mode: ChatExtractionMode,
    name: &str,
    answer: Value,
) -> (ChatExtractionMetadata, Value) {
    match name {
        "documented unit enum" => target.extract::<Wrapped<Documented>>(mode, answer).await,
        "externally tagged" => target.extract::<Wrapped<External>>(mode, answer).await,
        "internally tagged" => target.extract::<Wrapped<Internal>>(mode, answer).await,
        "adjacently tagged" => target.extract::<Wrapped<Adjacent>>(mode, answer).await,
        "mixed unit and struct" => target.extract::<Wrapped<Mixed>>(mode, answer).await,
        "untagged" => target.extract::<Wrapped<Untagged>>(mode, answer).await,
        other => panic!("unknown shape {other}"),
    }
}

impl Target {
    async fn extract<T>(
        self,
        mode: ChatExtractionMode,
        answer: Value,
    ) -> (ChatExtractionMetadata, Value)
    where
        T: JsonSchema + DeserializeOwned + Send + std::fmt::Debug,
    {
        let response = match mode {
            ChatExtractionMode::Native => self.text(&answer.to_string()),
            _ => self.tool_call(&answer.to_string()),
        };
        let transport = ReplayTransport::new([response]);
        let request = ChatRequest::new(self.model())
            .user("Extract it.")
            .with_option(mode);
        let extracted = self
            .provider(&transport)
            .extract::<T>(&request)
            .await
            .unwrap_or_else(|err| panic!("{self:?} {mode:?}: {err}"));
        let mut sent = bodies(&transport);
        assert_eq!(sent.len(), 1);
        (extracted.metadata, sent.remove(0))
    }
}

#[tokio::test]
async fn schemars_enum_shapes_are_exact() {
    for (name, answer) in exact_shapes() {
        let (metadata, sent) =
            run_shape(Target::Gemini, ChatExtractionMode::Native, name, answer).await;
        assert_eq!(
            metadata.enforcement,
            SchemaEnforcement::Exact,
            "{name}: {:?}",
            metadata.degradations
        );
        let schema = &sent["generationConfig"]["responseJsonSchema"];
        assert!(!ref_with_siblings(schema), "{name}: {schema}");
        assert!(!schema.to_string().contains("allOf"), "{name}: {schema}");
        assert!(
            !schema.to_string().contains("\"const\""),
            "{name}: {schema}"
        );
    }
}

/// The internally tagged newtype variant is sent merged: the inner fields
/// beside the tag, the tag as a one-value `enum`, no `$ref` siblings.
#[tokio::test]
async fn an_internally_tagged_newtype_is_sent_merged() {
    let (metadata, sent) = Target::Gemini
        .extract::<Wrapped<Internal>>(
            ChatExtractionMode::Native,
            json!({"v": {"kind": "New", "inner_field": 1, "note": null}}),
        )
        .await;
    assert_eq!(metadata.enforcement, SchemaEnforcement::Exact);
    let schema = &sent["generationConfig"]["responseJsonSchema"];
    let variants = schema["$defs"]["Internal"]["oneOf"].as_array().unwrap();
    let new = variants
        .iter()
        .find(|v| v["properties"]["kind"]["enum"] == json!(["New"]))
        .unwrap_or_else(|| panic!("{schema}"));
    assert!(new.get("$ref").is_none(), "{new}");
    assert!(new["properties"].get("inner_field").is_some(), "{new}");
    assert_eq!(
        new["properties"]["kind"],
        json!({"type": "string", "enum": ["New"]})
    );
}

/// A boolean subschema (`serde_json::Value`) and an object-valued `const`
/// are outside the subset: sent, and reported `Approximate`.
#[tokio::test]
async fn constructs_outside_the_subset_are_approximate() {
    #[derive(Debug, Deserialize, JsonSchema)]
    #[schemars(crate = "allama::schemars")]
    struct Payload {
        data: Value,
    }

    #[derive(Debug, Deserialize)]
    struct Fixed(Value);

    impl JsonSchema for Fixed {
        fn schema_name() -> std::borrow::Cow<'static, str> {
            "Fixed".into()
        }

        fn json_schema(_: &mut allama::schemars::SchemaGenerator) -> allama::schemars::Schema {
            allama::schemars::json_schema!({"const": {"a": 1}})
        }
    }

    let (metadata, sent) = Target::Gemini
        .extract::<Payload>(ChatExtractionMode::Native, json!({"data": {"any": true}}))
        .await;
    assert_approximated(&metadata, "unconstrained `true`");
    assert_eq!(
        sent["generationConfig"]["responseJsonSchema"]["properties"]["data"],
        json!(true)
    );

    let (metadata, _) = Target::Gemini
        .extract::<Wrapped<Fixed>>(ChatExtractionMode::Native, json!({"v": {"a": 1}}))
        .await;
    assert_approximated(&metadata, "`const`");

    let (metadata, _) = Target::Gemini
        .extract::<Wrapped<Overlapping>>(ChatExtractionMode::Native, json!({"v": 2}))
        .await;
    assert_approximated(&metadata, "`oneOf`");
}

#[track_caller]
fn assert_approximated(metadata: &ChatExtractionMetadata, needle: &str) {
    assert_eq!(
        metadata.enforcement,
        SchemaEnforcement::Approximate,
        "{metadata:?}"
    );
    let [warning] = metadata.degradations.as_slice() else {
        panic!("{:?}", metadata.degradations);
    };
    assert_eq!(warning.kind, WarningKind::Approximated);
    assert_eq!(
        warning.subject,
        Subject::Field {
            field: RequestField::ResponseFormat
        }
    );
    assert!(
        warning.message.contains(needle),
        "{needle}: {}",
        warning.message
    );
}

/// Gemini has no strict tools, so forced-tool extraction is `NotEnforced`.
#[tokio::test]
async fn forced_tool_extraction_is_not_enforced() {
    let (metadata, _) = Target::Gemini
        .extract::<Wrapped<Documented>>(ChatExtractionMode::ForcedTool, json!({"v": "First"}))
        .await;
    assert_eq!(metadata.enforcement, SchemaEnforcement::NotEnforced);
    assert_eq!(metadata.degradations[0].kind, WarningKind::Unsupported);
}
