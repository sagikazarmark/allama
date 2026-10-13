//! Extraction reports what was enforced (`SchemaEnforcement`) through the
//! Anthropic adapter, in both modes: the enum shapes schemars writes are
//! `Exact`, a `oneOf` with overlapping branches is `Approximate`, and a
//! merged `$ref` never reaches the wire as `allOf`.

// The fixture types are only deserialized into, never read.
#![allow(dead_code)]

use allama::extract::{ExtractExt, ExtractionMetadata, ExtractionMode, SchemaEnforcement};
use allama::prelude::*;
use allama::schemars::JsonSchema;
use allama::{RequestField, Subject, WarningKind};
use allama_anthropic::Provider;
use allama_http::{ReplayResponse, ReplayTransport};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

const MODEL: &str = "claude-haiku-4-5-20251001";

#[derive(Clone, Copy, Debug)]
enum Target {
    Anthropic,
}

const TARGETS: [Target; 1] = [Target::Anthropic];
const MODES: [ExtractionMode; 2] = [ExtractionMode::Native, ExtractionMode::ForcedTool];

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
        message(json!([{"type": "text", "text": text}]), "end_turn")
    }

    fn tool_call(self, arguments: &str) -> ReplayResponse {
        let input: Value = serde_json::from_str(arguments).unwrap();
        message(
            json!([{"type": "tool_use", "id": "toolu_1", "name": "submit_structured_output", "input": input}]),
            "tool_use",
        )
    }
}

fn message(content: Value, stop_reason: &str) -> ReplayResponse {
    ReplayResponse::json(
        200,
        json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": MODEL,
            "content": content, "stop_reason": stop_reason, "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })
        .to_string(),
    )
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
    mode: ExtractionMode,
    name: &str,
    answer: Value,
) -> (ExtractionMetadata, Value) {
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
    async fn extract<T>(self, mode: ExtractionMode, answer: Value) -> (ExtractionMetadata, Value)
    where
        T: JsonSchema + DeserializeOwned + Send + std::fmt::Debug,
    {
        let response = match mode {
            ExtractionMode::Native => self.text(&answer.to_string()),
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
async fn schemars_enum_shapes_are_exact_in_both_modes() {
    for target in TARGETS {
        for mode in MODES {
            for (name, answer) in exact_shapes() {
                let (metadata, sent) = run_shape(target, mode, name, answer).await;
                assert_eq!(
                    metadata.enforcement,
                    SchemaEnforcement::Exact,
                    "{target:?} {mode:?} {name}: {:?}",
                    metadata.degradations
                );
                assert!(
                    metadata.degradations.is_empty(),
                    "{target:?} {mode:?} {name}"
                );
                assert!(
                    !sent.to_string().contains("allOf"),
                    "{target:?} {mode:?} {name}: {sent}"
                );
                if name == "internally tagged" {
                    // The newtype variant carries the inner struct's fields.
                    assert!(
                        declares(&sent, &["kind", "inner_field", "note"]),
                        "{target:?} {mode:?}: {sent}"
                    );
                }
            }
        }
    }
}

#[tokio::test]
async fn an_overlapping_one_of_is_approximate_in_both_modes() {
    for target in TARGETS {
        for mode in MODES {
            let (metadata, _) = target
                .extract::<Wrapped<Overlapping>>(mode, json!({"v": 2}))
                .await;
            assert_eq!(
                metadata.enforcement,
                SchemaEnforcement::Approximate,
                "{target:?} {mode:?}"
            );
            let field = match mode {
                ExtractionMode::Native => RequestField::ResponseFormat,
                _ => RequestField::Tools,
            };
            let [warning] = metadata.degradations.as_slice() else {
                panic!("{target:?} {mode:?}: {:?}", metadata.degradations);
            };
            assert_eq!(warning.kind, WarningKind::Approximated);
            assert_eq!(warning.subject, Subject::Field { field });
            assert!(warning.message.contains("oneOf"), "{}", warning.message);
        }
    }
}
