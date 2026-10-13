//! Strict nullability round-trips through serde, through both OpenAI
//! providers: optional properties without a default go out nullable, the
//! `null`s the model sends for them are removed before deserializing, and
//! `#[serde(default)]` fields with a default stay non-nullable. Covers both
//! extraction modes and strict tool arguments (`parse_arguments_with`).

use allama::extract::{ExtractExt, ExtractionMode};
use allama::prelude::*;
use allama::schemars::JsonSchema;
use allama_http::{ReplayResponse, ReplayTransport};
use allama_openai::{Provider, ResponsesProvider};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const MODEL: &str = "gpt-4.1";

#[derive(Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[schemars(crate = "allama::schemars")]
struct Address {
    city: String,
    #[serde(default)]
    lines: Vec<String>,
}

/// Not `Serialize`: schemars writes no `default` for it, so it goes out
/// nullable.
#[derive(Debug, Default, PartialEq, Deserialize, JsonSchema)]
#[schemars(crate = "allama::schemars")]
struct Limits {
    max: u32,
}

#[derive(Debug, PartialEq, Deserialize, JsonSchema)]
#[schemars(crate = "allama::schemars")]
struct Ticket {
    title: String,
    #[serde(default)]
    tags: Vec<String>,
    assignee: Option<String>,
    #[serde(default)]
    limits: Limits,
    #[serde(default)]
    watchers: Vec<Limits>,
    home: Option<Address>,
}

/// What a strict-mode model may send: `null` for every optional property
/// without a default, a value for the ones with one.
fn strict_output() -> Value {
    json!({
        "title": "Charged twice",
        "tags": ["billing"],
        "assignee": null,
        "limits": null,
        "watchers": null,
        "home": null
    })
}

fn expected() -> Ticket {
    Ticket {
        title: "Charged twice".into(),
        tags: vec!["billing".into()],
        assignee: None,
        limits: Limits::default(),
        watchers: Vec::new(),
        home: None,
    }
}

#[derive(Clone, Copy, Debug)]
enum Api {
    ChatCompletions,
    Responses,
}

const APIS: [Api; 2] = [Api::ChatCompletions, Api::Responses];

impl Api {
    fn text(self, text: &str) -> ReplayResponse {
        let body = match self {
            Self::ChatCompletions => json!({
                "id": "chatcmpl-1", "object": "chat.completion", "model": MODEL,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            }),
            Self::Responses => json!({
                "id": "resp_1", "object": "response", "status": "completed", "model": MODEL,
                "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                            "content": [{"type": "output_text", "text": text, "annotations": []}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            }),
        };
        ReplayResponse::json(200, body.to_string())
    }

    fn tool_call(self, name: &str, arguments: &Value) -> ReplayResponse {
        let arguments = arguments.to_string();
        let body = match self {
            Self::ChatCompletions => json!({
                "id": "chatcmpl-1", "object": "chat.completion", "model": MODEL,
                "choices": [{"index": 0, "finish_reason": "tool_calls",
                             "message": {"role": "assistant", "content": null, "tool_calls": [
                                 {"id": "call_1", "type": "function",
                                  "function": {"name": name, "arguments": arguments}}
                             ]}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            }),
            Self::Responses => json!({
                "id": "resp_1", "object": "response", "status": "completed", "model": MODEL,
                "output": [{"type": "function_call", "id": "fc_1", "call_id": "call_1",
                            "name": name, "arguments": arguments, "status": "completed"}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            }),
        };
        ReplayResponse::json(200, body.to_string())
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

    /// The strict schema sent for the response format.
    fn response_format_schema(self, body: &Value) -> Value {
        match self {
            Self::ChatCompletions => body["response_format"]["json_schema"]["schema"].clone(),
            Self::Responses => body["text"]["format"]["schema"].clone(),
        }
    }

    /// The tool's `(parameters, strict)` as sent.
    fn tool(self, body: &Value) -> (Value, Value) {
        match self {
            Self::ChatCompletions => {
                let function = &body["tools"][0]["function"];
                (function["parameters"].clone(), function["strict"].clone())
            }
            Self::Responses => {
                let tool = &body["tools"][0];
                (tool["parameters"].clone(), tool["strict"].clone())
            }
        }
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

#[track_caller]
fn assert_strict_nullability(schema: &Value) {
    let properties = &schema["properties"];
    assert_eq!(
        schema["required"],
        json!(["assignee", "home", "limits", "tags", "title", "watchers"])
    );
    // A `#[serde(default)]` field with a `Serialize` default stays
    // non-nullable, and the `default` keyword is gone.
    assert_eq!(
        properties["tags"],
        json!({"type": "array", "items": {"type": "string"}})
    );
    // Without a default: nullable.
    assert_eq!(
        properties["limits"],
        json!({"anyOf": [{"$ref": "#/$defs/Limits"}, {"type": "null"}]})
    );
    assert_eq!(properties["watchers"]["type"], json!(["array", "null"]));
}

#[tokio::test]
async fn native_extraction_round_trips_strict_nulls() {
    for api in APIS {
        let transport = ReplayTransport::new([api.text(&strict_output().to_string())]);
        let request = ChatRequest::new(MODEL)
            .user("I was charged twice.")
            .with_option(ExtractionMode::Native);
        let extracted = api
            .provider(&transport)
            .extract::<Ticket>(&request)
            .await
            .unwrap_or_else(|e| panic!("{api:?}: {e}"));
        assert_eq!(extracted.value, expected(), "{api:?}");
        let sent = bodies(&transport);
        assert_strict_nullability(&api.response_format_schema(&sent[0]));
    }
}

#[tokio::test]
async fn forced_tool_extraction_round_trips_strict_nulls() {
    for api in APIS {
        let transport =
            ReplayTransport::new([api.tool_call("submit_structured_output", &strict_output())]);
        let request = ChatRequest::new(MODEL)
            .user("I was charged twice.")
            .with_option(ExtractionMode::ForcedTool);
        let extracted = api
            .provider(&transport)
            .extract::<Ticket>(&request)
            .await
            .unwrap_or_else(|e| panic!("{api:?}: {e}"));
        assert_eq!(extracted.value, expected(), "{api:?}");
    }
}

#[tokio::test]
async fn strict_tool_arguments_round_trip_with_parse_arguments_with() {
    let tool = ToolDefinition::from_schema::<Ticket>("file_ticket").strict(true);
    for api in APIS {
        let transport = ReplayTransport::new([api.tool_call("file_ticket", &strict_output())]);
        let request = ChatRequest::new(MODEL)
            .user("I was charged twice.")
            .tools([tool.clone()]);
        let response = api.provider(&transport).chat(&request).await.unwrap();
        let call = response.tool_calls().next().expect("a tool call");
        assert!(
            call.parse_arguments::<Ticket>().is_err(),
            "{api:?}: plain parsing rejects the strict nulls"
        );
        let args: Ticket = call
            .parse_arguments_with(&tool)
            .unwrap_or_else(|e| panic!("{api:?}: {e}"));
        assert_eq!(args, expected(), "{api:?}");

        let sent = bodies(&transport);
        let (parameters, strict) = api.tool(&sent[0]);
        assert_eq!(strict, true, "{api:?}");
        assert_strict_nullability(&parameters);
    }
}
