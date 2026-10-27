//! OpenAI strict-schema dialect through both OpenAI providers: strict tools
//! and response formats are rewritten into what strict mode accepts, and a
//! schema strict mode cannot express goes out non-strict with a warning
//! (`UnsupportedPolicy::Warn`) or fails before sending (`Error`).

use std::collections::HashMap;

use allama::chat::extract::{ChatExtractExt, ChatExtracted, ChatExtractionMode, SchemaEnforcement};
use allama::prelude::*;
use allama::schemars::JsonSchema;
use allama::{DynChatProvider, RequestField, Subject, UnsupportedPolicy, Warning, WarningKind};
use allama_http::{ReplayResponse, ReplayTransport};
use allama_openai::{Provider, ResponsesProvider};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

const MODEL: &str = "gpt-4.1";
const FINE_TUNED: &str = "ft:gpt-4.1-mini:acme::abc123";

fn chat_ok() -> ReplayResponse {
    ReplayResponse::json(
        200,
        json!({
            "id": "chatcmpl-1", "object": "chat.completion", "model": MODEL,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "{}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })
        .to_string(),
    )
}

fn responses_ok() -> ReplayResponse {
    ReplayResponse::json(
        200,
        json!({
            "id": "resp_1", "object": "response", "status": "completed", "model": MODEL,
            "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "{}", "annotations": []}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        })
        .to_string(),
    )
}

#[derive(Clone, Copy, Debug)]
enum Api {
    ChatCompletions,
    Responses,
}

const APIS: [Api; 2] = [Api::ChatCompletions, Api::Responses];

impl Api {
    /// Send `request`; return the result and the bodies that reached the
    /// wire.
    async fn send(self, request: &ChatRequest) -> (Result<ChatResponse>, Vec<Value>) {
        let ok = match self {
            Self::ChatCompletions => chat_ok(),
            Self::Responses => responses_ok(),
        };
        let transport = ReplayTransport::new([ok]);
        let result = self.provider(&transport).chat(request).await;
        (result, bodies(&transport))
    }

    /// Run a forced-tool extraction of `T` answered with a call to the
    /// synthetic tool with `arguments`; return the result and the bodies
    /// that reached the wire.
    async fn extract<T>(
        self,
        request: &ChatRequest,
        arguments: Value,
    ) -> (Result<ChatExtracted<T>>, Vec<Value>)
    where
        T: JsonSchema + DeserializeOwned + Send,
    {
        let transport = ReplayTransport::new([self.tool_call(arguments)]);
        let result = self.provider(&transport).extract::<T>(request).await;
        (result, bodies(&transport))
    }

    fn provider(self, transport: &ReplayTransport) -> DynChatProvider {
        match self {
            Self::ChatCompletions => DynChatProvider::new(
                Provider::builder()
                    .api_key("k")
                    .http_transport(transport.clone())
                    .build()
                    .unwrap(),
            ),
            Self::Responses => DynChatProvider::new(
                ResponsesProvider::builder()
                    .api_key("k")
                    .http_transport(transport.clone())
                    .build()
                    .unwrap(),
            ),
        }
    }

    /// A response that calls `submit_structured_output` with `arguments`.
    fn tool_call(self, arguments: Value) -> ReplayResponse {
        let arguments = arguments.to_string();
        let body = match self {
            Self::ChatCompletions => json!({
                "id": "chatcmpl-1", "object": "chat.completion", "model": MODEL,
                "choices": [{"index": 0, "finish_reason": "tool_calls",
                             "message": {"role": "assistant", "content": null,
                                         "tool_calls": [{"id": "call_1", "type": "function",
                                                         "function": {"name": "submit_structured_output",
                                                                      "arguments": arguments}}]}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            }),
            Self::Responses => json!({
                "id": "resp_1", "object": "response", "status": "completed", "model": MODEL,
                "output": [{"type": "function_call", "id": "fc_1", "call_id": "call_1",
                            "name": "submit_structured_output", "arguments": arguments,
                            "status": "completed"}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            }),
        };
        ReplayResponse::json(200, body.to_string())
    }

    /// The tool's `(parameters, strict)` as sent.
    fn tool(self, body: &Value) -> (&Value, &Value) {
        match self {
            Self::ChatCompletions => {
                let function = &body["tools"][0]["function"];
                (&function["parameters"], &function["strict"])
            }
            Self::Responses => {
                let tool = &body["tools"][0];
                (&tool["parameters"], &tool["strict"])
            }
        }
    }

    /// The response format's `(schema, strict)` as sent.
    fn response_format(self, body: &Value) -> (&Value, &Value) {
        let format = match self {
            Self::ChatCompletions => &body["response_format"]["json_schema"],
            Self::Responses => &body["text"]["format"],
        };
        (&format["schema"], &format["strict"])
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

/// A schema using every construct the dialect rewrites, shaped like
/// schemars output.
fn ticket_schema() -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Ticket",
        "type": "object",
        "properties": {
            "kind": {"oneOf": [
                {"const": "bug", "description": "Something is broken"},
                {"const": "feature", "description": "Something is missing"}
            ]},
            "count": {"type": "integer", "format": "uint32", "minimum": 0},
            "ratio": {"type": "number", "format": "double"},
            "link": {"type": "string", "format": "uri"},
            "opened": {"type": "string", "format": "date-time"},
            "owner": {"$ref": "#/$defs/Person", "description": "Who owns it"},
            "reviewer": {"$ref": "#/$defs/Person"},
            "status": {"allOf": [{"$ref": "#/$defs/Status"}], "description": "Current status"},
            "note": {"type": ["string", "null"], "default": null}
        },
        "required": ["kind", "count", "ratio", "link", "opened", "owner", "reviewer", "status", "note"],
        "$defs": {
            "Person": {
                "type": "object",
                "properties": {"name": {"type": "string", "pattern": "^[A-Z]"}},
                "required": ["name"]
            },
            "Status": {"type": "string", "enum": ["open", "closed"]}
        }
    })
}

/// What OpenAI strict mode receives for [`ticket_schema`].
fn strict_ticket_schema() -> Value {
    json!({
        "title": "Ticket",
        "type": "object",
        "properties": {
            "kind": {"anyOf": [
                {"const": "bug", "description": "Something is broken"},
                {"const": "feature", "description": "Something is missing"}
            ]},
            "count": {"type": "integer", "minimum": 0},
            "ratio": {"type": "number"},
            "link": {"type": "string"},
            "opened": {"type": "string", "format": "date-time"},
            "owner": {
                "type": "object",
                "description": "Who owns it",
                "properties": {"name": {"type": "string", "pattern": "^[A-Z]"}},
                "required": ["name"],
                "additionalProperties": false
            },
            "reviewer": {"$ref": "#/$defs/Person"},
            "status": {"type": "string", "enum": ["open", "closed"], "description": "Current status"},
            "note": {"type": ["string", "null"]}
        },
        "required": ["count", "kind", "link", "note", "opened", "owner", "ratio", "reviewer", "status"],
        "additionalProperties": false,
        "$defs": {
            "Person": {
                "type": "object",
                "properties": {"name": {"type": "string", "pattern": "^[A-Z]"}},
                "required": ["name"],
                "additionalProperties": false
            },
            "Status": {"type": "string", "enum": ["open", "closed"]}
        }
    })
}

fn ticket_request(model: &str) -> ChatRequest {
    ChatRequest::new(model)
        .user("File a ticket.")
        .tools([ToolDefinition::new("file_ticket", ticket_schema()).strict(true)])
        .response_format(ResponseFormat::json_schema(ticket_schema()).name("ticket"))
}

fn messages(warnings: &[Warning], field: RequestField) -> Vec<&str> {
    warnings
        .iter()
        .filter(|w| w.subject == Subject::Field { field })
        .map(|w| {
            assert_eq!(w.kind, WarningKind::Approximated, "{w}");
            w.message.as_str()
        })
        .collect()
}

#[tokio::test]
async fn strict_schemas_are_rewritten_into_the_openai_dialect() {
    for api in APIS {
        let (result, sent) = api.send(&ticket_request(MODEL)).await;
        let response = result.unwrap();
        let (parameters, strict) = api.tool(&sent[0]);
        assert_eq!(parameters, &strict_ticket_schema(), "{api:?} tool");
        assert_eq!(strict, true, "{api:?}");
        let (schema, strict) = api.response_format(&sent[0]);
        assert_eq!(schema, &strict_ticket_schema(), "{api:?} response format");
        assert_eq!(strict, true, "{api:?}");

        // The `uri` format, once per field; numeric formats,
        // `default: null`, `$schema`, `allOf`, `$ref` siblings and the
        // disjoint `oneOf` are rewritten without changing what the model
        // may produce.
        for field in [RequestField::Tools, RequestField::ResponseFormat] {
            let messages = messages(&response.warnings, field);
            assert_eq!(messages.len(), 1, "{api:?} {field}: {messages:?}");
            assert!(messages[0].contains("uri"), "{}", messages[0]);
        }
        assert_eq!(response.warnings.len(), 2, "{:?}", response.warnings);
    }
}

/// `ResponseFormat::json_schema` asks for strict mode on both APIs: the
/// schema goes out rewritten, with `strict: true`. `non_strict` sends it as
/// written with `strict: false`, and reports nothing.
#[tokio::test]
async fn json_schema_is_strict_unless_relaxed() {
    for api in APIS {
        let strict = ChatRequest::new(MODEL)
            .user("Open a ticket")
            .response_format(ResponseFormat::json_schema(ticket_schema()));
        let (result, sent) = api.send(&strict).await;
        let response = result.unwrap();
        let (schema, flag) = api.response_format(&sent[0]);
        assert_eq!(schema, &strict_ticket_schema(), "{api:?}");
        assert_eq!(flag, true, "{api:?}");
        assert_eq!(
            messages(&response.warnings, RequestField::ResponseFormat).len(),
            1,
            "{api:?}: the `uri` format"
        );

        let relaxed = ChatRequest::new(MODEL)
            .user("Open a ticket")
            .response_format(ResponseFormat::json_schema(ticket_schema()).non_strict());
        let (result, sent) = api.send(&relaxed).await;
        let response = result.unwrap();
        let (schema, flag) = api.response_format(&sent[0]);
        assert_eq!(schema, &ticket_schema(), "{api:?}");
        assert_eq!(flag, false, "{api:?}");
        assert!(
            response.warnings.is_empty(),
            "{api:?}: {:?}",
            response.warnings
        );
    }
}

#[tokio::test]
async fn fine_tuned_models_lose_constraint_keywords() {
    for api in APIS {
        let (result, sent) = api.send(&ticket_request(FINE_TUNED)).await;
        let response = result.unwrap();
        let (parameters, strict) = api.tool(&sent[0]);
        assert_eq!(strict, true, "{api:?}");
        let properties = &parameters["properties"];
        assert_eq!(properties["count"], json!({"type": "integer"}));
        assert_eq!(properties["opened"], json!({"type": "string"}));
        assert_eq!(
            parameters["$defs"]["Person"]["properties"]["name"],
            json!({"type": "string"})
        );
        let (schema, _) = api.response_format(&sent[0]);
        assert_eq!(schema, parameters);

        for field in [RequestField::Tools, RequestField::ResponseFormat] {
            let messages = messages(&response.warnings, field);
            let fine_tuned = messages
                .iter()
                .find(|m| m.contains("fine-tuned"))
                .unwrap_or_else(|| panic!("{api:?} {field}: {messages:?}"));
            for keyword in ["format", "minimum", "pattern"] {
                assert!(fine_tuned.contains(keyword), "{fine_tuned}");
            }
        }
    }
}

/// `extract::<Vec<T>>` or an enum: strict mode needs an object root.
fn list_schema() -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "array",
        "items": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    })
}

fn without_schema_uri(mut schema: Value) -> Value {
    schema.as_object_mut().unwrap().remove("$schema");
    schema
}

/// OpenAI rejects a non-object root with HTTP 400, strict or not
/// (verified live, 2026-10-13), so it is refused before sending under every
/// policy, with the response format named so extraction can fall back.
#[tokio::test]
async fn a_non_object_root_is_refused_before_sending() {
    let request = ChatRequest::new(MODEL)
        .user("List the people.")
        .response_format(ResponseFormat::json_schema(list_schema()).name("people"));
    let relaxed = ChatRequest::new(MODEL)
        .user("List the people.")
        .response_format(
            ResponseFormat::json_schema(list_schema())
                .name("people")
                .non_strict(),
        );
    for api in APIS {
        for request in [
            request.clone(),
            relaxed.clone(),
            request.clone().unsupported_policy(UnsupportedPolicy::Error),
        ] {
            let (result, sent) = api.send(&request).await;
            let err = result.unwrap_err();
            assert_eq!(
                err.unsupported_field(),
                Some(RequestField::ResponseFormat),
                "{api:?}: {err:?}"
            );
            assert!(err.message().contains("object"), "{err}");
            assert!(sent.is_empty(), "{api:?}: nothing is sent");
        }
    }
}

fn labels_schema() -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "title": {"type": "string", "format": "uri"},
            "labels": {"type": "object", "additionalProperties": {"type": "string"}}
        },
        "required": ["title", "labels"]
    })
}

#[tokio::test]
async fn a_strict_map_tool_goes_out_non_strict_with_its_original_parameters() {
    let request = ChatRequest::new(MODEL)
        .user("Label this.")
        .tools([ToolDefinition::new("label", labels_schema()).strict(true)]);
    for api in APIS {
        let (result, sent) = api.send(&request).await;
        let response = result.unwrap();
        let (parameters, strict) = api.tool(&sent[0]);
        // Not the partly rewritten schema: the `uri` format is still there.
        assert_eq!(parameters, &without_schema_uri(labels_schema()), "{api:?}");
        assert_eq!(strict, false, "{api:?}");
        assert_eq!(response.warnings.len(), 1, "{:?}", response.warnings);
        let warning = &response.warnings[0];
        assert_eq!(warning.kind, WarningKind::Unsupported);
        assert_eq!(
            warning.subject,
            Subject::Field {
                field: RequestField::Tools
            }
        );
        assert!(
            warning
                .message
                .starts_with("tool `label` is not strict-representable"),
            "{warning}"
        );
        assert!(warning.message.contains("/properties/labels"), "{warning}");

        let strict = request.clone().unsupported_policy(UnsupportedPolicy::Error);
        let (result, sent) = api.send(&strict).await;
        assert!(
            matches!(result, Err(Error::Unsupported { .. })),
            "{api:?}: {result:?}"
        );
        assert!(sent.is_empty(), "{api:?}: nothing is sent");
    }
}

/// A schema with a composition keyword OpenAI strict mode rejects: an
/// `allOf` of two objects and a `not`.
fn composed_schema() -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "contact": {"allOf": [
                {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
                {"type": "object", "properties": {"email": {"type": "string", "format": "email"}}}
            ]},
            "code": {"type": "string", "not": {"const": "0000"}}
        },
        "required": ["contact", "code"]
    })
}

#[tokio::test]
async fn a_rejected_composition_keyword_goes_out_non_strict_with_the_original_schema() {
    let request = ChatRequest::new(MODEL)
        .user("Who is it?")
        .tools([ToolDefinition::new("contact", composed_schema()).strict(true)])
        .response_format(ResponseFormat::json_schema(composed_schema()).name("contact"));
    for api in APIS {
        let (result, sent) = api.send(&request).await;
        let response = result.unwrap();
        // The caller's schema, minus `$schema`: the `allOf` members are not
        // closed, and nothing else was rewritten.
        let (parameters, strict) = api.tool(&sent[0]);
        assert_eq!(
            parameters,
            &without_schema_uri(composed_schema()),
            "{api:?}"
        );
        assert_eq!(strict, false, "{api:?}");
        let (schema, strict) = api.response_format(&sent[0]);
        assert_eq!(schema, &without_schema_uri(composed_schema()), "{api:?}");
        assert_eq!(strict, false, "{api:?}");

        assert_eq!(response.warnings.len(), 2, "{:?}", response.warnings);
        for (warning, field) in response
            .warnings
            .iter()
            .zip([RequestField::Tools, RequestField::ResponseFormat])
        {
            assert_eq!(warning.kind, WarningKind::Unsupported, "{warning}");
            assert_eq!(warning.subject, Subject::Field { field }, "{warning}");
            assert!(
                warning.message.contains("`not`") || warning.message.contains("`allOf`"),
                "{warning}"
            );
        }

        let strict = request.clone().unsupported_policy(UnsupportedPolicy::Error);
        let (result, sent) = api.send(&strict).await;
        assert!(
            matches!(result, Err(Error::Unsupported { .. })),
            "{api:?}: {result:?}"
        );
        assert!(sent.is_empty(), "{api:?}: nothing is sent");
    }
}

#[tokio::test]
async fn a_malformed_strict_schema_fails_under_either_policy() {
    let broken = json!({
        "type": "object",
        "properties": {"owner": {"$ref": "#/$defs/Missing", "description": "Who"}}
    });
    let request = ChatRequest::new(MODEL)
        .user("q")
        .tools([ToolDefinition::new("f", broken).strict(true)]);
    for api in APIS {
        let (result, sent) = api.send(&request).await;
        assert!(
            matches!(result, Err(Error::InvalidRequest { .. })),
            "{api:?}: {result:?}"
        );
        assert!(sent.is_empty(), "{api:?}: nothing is sent");
    }
}

#[derive(Debug, JsonSchema, Deserialize)]
#[schemars(crate = "allama::schemars")]
struct Person {
    name: String,
    nickname: Option<String>,
}

fn person_request(model: &str) -> ChatRequest {
    ChatRequest::new(model)
        .user("Ada Lovelace wrote the first program.")
        .with_option(ChatExtractionMode::ForcedTool)
}

/// The synthetic extraction tool is strict, rewritten into the dialect, and
/// the extraction reports the schema enforced.
#[tokio::test]
async fn forced_tool_extraction_sends_a_strict_tool() {
    for api in APIS {
        let (result, sent) = api
            .extract::<Person>(
                &person_request(MODEL),
                json!({"name": "Ada", "nickname": null}),
            )
            .await;
        let extracted = result.unwrap();
        assert_eq!(extracted.value.name, "Ada");
        assert_eq!(extracted.value.nickname, None);
        assert_eq!(
            extracted.metadata.enforcement,
            SchemaEnforcement::Exact,
            "{api:?}"
        );
        assert!(extracted.metadata.degradations.is_empty(), "{api:?}");
        let (parameters, strict) = api.tool(&sent[0]);
        assert_eq!(strict, &json!(true), "{api:?}");
        assert_eq!(
            parameters["required"],
            json!(["name", "nickname"]),
            "{api:?}"
        );
        assert!(sent[0].get("parallel_tool_calls").is_none(), "{api:?}");
    }
}

/// Fine-tuned models lose strict mode with parallel tool calls, so the
/// adapter turns them off for the strict synthetic tool (extraction itself
/// leaves `parallel_tool_calls` alone).
#[tokio::test]
async fn forced_tool_extraction_on_fine_tuned_models_disables_parallel_tool_calls() {
    for api in APIS {
        let (result, sent) = api
            .extract::<Person>(
                &person_request(FINE_TUNED),
                json!({"name": "Ada", "nickname": null}),
            )
            .await;
        assert_eq!(
            result.unwrap().metadata.enforcement,
            SchemaEnforcement::Exact,
            "{api:?}"
        );
        assert_eq!(sent[0]["parallel_tool_calls"], json!(false), "{api:?}");
        assert_eq!(api.tool(&sent[0]).1, &json!(true), "{api:?}");
    }
}

/// A target the dialect cannot express (a map) is extracted non-strict
/// under `Warn`, with the degradation in the metadata, and fails before
/// sending under `Error`.
#[tokio::test]
async fn forced_tool_extraction_into_a_map_reports_the_degradation() {
    #[derive(Debug, JsonSchema, Deserialize)]
    #[schemars(crate = "allama::schemars")]
    struct Labels {
        labels: HashMap<String, String>,
    }

    for api in APIS {
        let (result, sent) = api
            .extract::<Labels>(
                &person_request(MODEL),
                json!({"labels": {"area": "billing"}}),
            )
            .await;
        let extracted = result.unwrap();
        assert_eq!(extracted.value.labels["area"], "billing");
        assert_eq!(
            extracted.metadata.enforcement,
            SchemaEnforcement::NotEnforced,
            "{api:?}"
        );
        let [degradation] = extracted.metadata.degradations.as_slice() else {
            panic!("{api:?}: {:?}", extracted.metadata.degradations);
        };
        assert_eq!(degradation.kind, WarningKind::Unsupported, "{api:?}");
        assert_eq!(
            degradation.subject,
            Subject::Field {
                field: RequestField::Tools
            }
        );
        assert_ne!(api.tool(&sent[0]).1, &json!(true), "{api:?}");

        let strict = person_request(MODEL).unsupported_policy(UnsupportedPolicy::Error);
        let (result, sent) = api.extract::<Labels>(&strict, json!({"labels": {}})).await;
        assert!(
            matches!(result, Err(Error::Unsupported { .. })),
            "{api:?}: {result:?}"
        );
        assert!(sent.is_empty(), "{api:?}: nothing is sent");
    }
}
