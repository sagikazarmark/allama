//! Strict tools whose schema Anthropic strict mode cannot express (maps,
//! recursion), or that go to a model a resolver answers without structured
//! output (and so without strict tools), degrade to
//! non-strict tools under `UnsupportedPolicy::Warn` and fail under
//! `UnsupportedPolicy::Error`, through the real adapter. Extraction reports
//! the degradation in its metadata.

use std::collections::HashMap;

use allama::chat::extract::{ChatExtractExt, ChatExtractionMode, SchemaEnforcement};
use allama::prelude::*;
use allama::schemars::JsonSchema;
use allama::{RequestField, Subject, UnsupportedPolicy, Warning, WarningKind};
use allama_anthropic::Provider;
use allama_http::{ReplayResponse, ReplayTransport};
use serde::Deserialize;
use serde_json::{Value, json};

const MODEL: &str = "claude-haiku-4-5-20251001";

fn provider(transport: &ReplayTransport) -> Provider {
    Provider::builder()
        .api_key("k")
        .http_transport(transport.clone())
        .build()
        .unwrap()
}

fn tool_use(name: &str, input: Value) -> ReplayResponse {
    ReplayResponse::json(
        200,
        json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": MODEL,
            "content": [{"type": "tool_use", "id": "toolu_1", "name": name, "input": input}],
            "stop_reason": "tool_use", "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })
        .to_string(),
    )
}

fn stream_ok() -> ReplayResponse {
    ReplayResponse::sse(concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-haiku-4-5-20251001\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":1}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    ))
}

fn bodies(transport: &ReplayTransport) -> Vec<Value> {
    transport
        .requests()
        .iter()
        .filter_map(|r| r.body.as_ref())
        .map(|body| serde_json::from_slice(body).expect("JSON request body"))
        .collect()
}

/// The schema schemars generates for a `HashMap<String, String>` field.
fn labels_schema() -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Labels",
        "type": "object",
        "properties": {
            "labels": {"type": "object", "additionalProperties": {"type": "string"}}
        },
        "required": ["labels"]
    })
}

fn labels_request() -> ChatRequest {
    ChatRequest::new(MODEL)
        .user("Label this ticket.")
        .tools([ToolDefinition::new("label", labels_schema()).strict(true)])
}

fn assert_sent_non_strict(body: &Value) {
    let tool = &body["tools"][0];
    assert!(tool.get("strict").is_none(), "{tool}");
    let mut expected = labels_schema();
    expected.as_object_mut().unwrap().remove("$schema");
    assert_eq!(tool["input_schema"], expected);
}

fn assert_strict_dropped(warnings: &[Warning]) {
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].kind, WarningKind::Unsupported);
    assert_eq!(
        warnings[0].subject,
        Subject::Field {
            field: RequestField::Tools
        }
    );
    assert!(warnings[0].message.contains("map-typed"), "{}", warnings[0]);
}

#[tokio::test]
async fn a_strict_map_tool_goes_out_non_strict_with_a_warning() {
    let transport = ReplayTransport::new([
        tool_use("label", json!({"labels": {"area": "billing"}})),
        stream_ok(),
    ]);
    let provider = provider(&transport);

    let response = provider.chat(&labels_request()).await.unwrap();
    assert_strict_dropped(&response.warnings);

    let streamed = provider
        .chat_stream(&labels_request())
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();
    assert_strict_dropped(&streamed.warnings);

    let sent = bodies(&transport);
    assert_eq!(sent.len(), 2);
    for body in &sent {
        assert_sent_non_strict(body);
    }
}

#[tokio::test]
async fn a_strict_map_tool_fails_under_the_error_policy_before_sending() {
    let transport = ReplayTransport::new([]);
    let request = labels_request().unsupported_policy(UnsupportedPolicy::Error);
    let err = provider(&transport).chat(&request).await.unwrap_err();
    assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
    let err = provider(&transport)
        .chat_stream(&request)
        .await
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
    assert!(transport.requests().is_empty(), "nothing is sent");
}

#[derive(Debug, JsonSchema, Deserialize)]
#[schemars(crate = "allama::schemars")]
struct Ticket {
    title: String,
    labels: HashMap<String, String>,
}

#[derive(Debug, JsonSchema, Deserialize)]
#[schemars(crate = "allama::schemars")]
struct Person {
    name: String,
}

fn extraction_request(model: &str) -> ChatRequest {
    ChatRequest::new(model)
        .user("I was charged twice this month.")
        .with_option(ChatExtractionMode::ForcedTool)
}

/// Forced-tool extraction into a struct with a `HashMap` keeps working under
/// `Warn`. The synthetic tool is strict, so the map makes it go out
/// non-strict with an `Unsupported` `Tools` warning, which the extraction
/// metadata reports.
#[tokio::test]
async fn forced_tool_extraction_into_a_map_works_under_warn() {
    let transport = ReplayTransport::new([tool_use(
        "submit_structured_output",
        json!({"title": "Charged twice", "labels": {"area": "billing"}}),
    )]);
    let extracted = provider(&transport)
        .extract::<Ticket>(&extraction_request(MODEL))
        .await
        .unwrap();
    assert_eq!(extracted.value.title, "Charged twice");
    assert_eq!(extracted.value.labels["area"], "billing");
    assert_strict_dropped(&extracted.response.warnings);
    assert_eq!(
        extracted.metadata.enforcement,
        SchemaEnforcement::NotEnforced
    );
    // The record names the caller's type, not the synthetic tool.
    let degradations = &extracted.metadata.degradations;
    assert_eq!(degradations.len(), 1, "{degradations:?}");
    assert_eq!(
        degradations[0].subject,
        extracted.response.warnings[0].subject
    );
    assert!(
        degradations[0]
            .message
            .starts_with("the schema of `Ticket` is not strict-representable: map-typed"),
        "{}",
        degradations[0].message
    );

    let sent = bodies(&transport);
    let tool = &sent[0]["tools"][0];
    assert!(tool.get("strict").is_none(), "{tool}");
    assert_eq!(
        tool["input_schema"]["properties"]["labels"]["additionalProperties"],
        json!({"type": "string"})
    );
}

/// Under `Error`, the same extraction fails before sending: enforced or
/// fail.
#[tokio::test]
async fn forced_tool_extraction_into_a_map_fails_under_error() {
    let transport = ReplayTransport::new([]);
    let request = extraction_request(MODEL).unsupported_policy(UnsupportedPolicy::Error);
    let err = provider(&transport)
        .extract::<Ticket>(&request)
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
    assert!(transport.requests().is_empty(), "nothing is sent");
}

/// A strict-capable model enforces the synthetic tool's schema.
#[tokio::test]
async fn forced_tool_extraction_is_strict_on_models_with_strict_tools() {
    let transport =
        ReplayTransport::new([tool_use("submit_structured_output", json!({"name": "Ada"}))]);
    let extracted = provider(&transport)
        .extract::<Person>(&extraction_request(MODEL))
        .await
        .unwrap();
    assert_eq!(extracted.metadata.enforcement, SchemaEnforcement::Exact);
    assert!(extracted.metadata.degradations.is_empty());
    assert_eq!(bodies(&transport)[0]["tools"][0]["strict"], true);
}

/// A model the caller's resolver answers without structured output (and so
/// without strict tools) gets the synthetic tool non-strict, and says so:
/// the extraction reports the degradation. The built-in answers never deny
/// it on the id (ADR 0015).
#[tokio::test]
async fn forced_tool_extraction_on_a_model_without_strict_tools_is_not_enforced() {
    const NO_STRICT: &str = "claude-sonnet-4-20250514";
    let without_strict = |transport: &ReplayTransport| {
        provider(transport).with_chat_capabilities(|_: &str, capability| {
            (capability == allama::ChatCapability::StructuredOutput)
                .then_some(allama::CapabilitySupport::Unsupported)
        })
    };
    let transport =
        ReplayTransport::new([tool_use("submit_structured_output", json!({"name": "Ada"}))]);
    let extracted = without_strict(&transport)
        .extract::<Person>(&extraction_request(NO_STRICT))
        .await
        .unwrap();
    assert_eq!(extracted.value.name, "Ada");
    assert_eq!(
        extracted.metadata.enforcement,
        SchemaEnforcement::NotEnforced
    );
    let [degradation] = extracted.metadata.degradations.as_slice() else {
        panic!("expected one degradation: {:?}", extracted.metadata);
    };
    assert_eq!(degradation.kind, WarningKind::Unsupported);
    assert_eq!(
        degradation.subject,
        Subject::Field {
            field: RequestField::Tools
        }
    );
    assert!(
        degradation
            .message
            .contains("does not support strict tools"),
        "{degradation}"
    );
    assert!(bodies(&transport)[0]["tools"][0].get("strict").is_none());

    let transport = ReplayTransport::new([]);
    let request = extraction_request(NO_STRICT).unsupported_policy(UnsupportedPolicy::Error);
    let err = without_strict(&transport)
        .extract::<Person>(&request)
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
    assert!(transport.requests().is_empty(), "nothing is sent");

    // Without a resolver, the same model is sent strict.
    let transport =
        ReplayTransport::new([tool_use("submit_structured_output", json!({"name": "Ada"}))]);
    let extracted = provider(&transport)
        .extract::<Person>(&extraction_request(NO_STRICT))
        .await
        .unwrap();
    assert_eq!(extracted.metadata.enforcement, SchemaEnforcement::Exact);
    assert_eq!(bodies(&transport)[0]["tools"][0]["strict"], true);
}

/// A JSON Schema response format for `Ticket`, as native extraction sends
/// it.
fn ticket_format_request() -> ChatRequest {
    ChatRequest::new(MODEL)
        .user("I was charged twice this month.")
        .response_format(
            ResponseFormat::json_schema(allama::adapter::schema::schema_for::<Ticket>())
                .name("ticket"),
        )
}

/// `output_config.format` has no unconstrained form, so a map in the schema
/// fails under every policy, before sending, naming the response format.
#[tokio::test]
async fn a_native_map_format_is_refused_before_sending_under_every_policy() {
    for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
        let transport = ReplayTransport::new([]);
        let request = ticket_format_request().unsupported_policy(policy);
        let err = provider(&transport).chat(&request).await.unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
        assert_eq!(
            err.unsupported_field(),
            Some(RequestField::ResponseFormat),
            "{policy:?}"
        );
        assert!(err.to_string().contains("map-typed"), "{err}");
        let err = provider(&transport)
            .chat_stream(&request)
            .await
            .map(|_| ())
            .unwrap_err();
        assert_eq!(
            err.unsupported_field(),
            Some(RequestField::ResponseFormat),
            "{policy:?}"
        );
        assert!(transport.requests().is_empty(), "nothing is sent");
    }
}

fn auto_extraction_request() -> ChatRequest {
    ChatRequest::new(MODEL).user("I was charged twice this month.")
}

/// The tool-use stream of a forced-tool extraction into `Ticket`.
fn ticket_tool_stream() -> ReplayResponse {
    ReplayResponse::sse(concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-haiku-4-5-20251001\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"submit_structured_output\",\"input\":{}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"title\\\": \\\"Charged twice\\\", \"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"labels\\\": {\\\"area\\\": \\\"billing\\\"}}\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":12}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    ))
}

/// What the forced-tool attempt of an `Auto` extraction into `Ticket`
/// reports: the switch, and the strict synthetic tool sent non-strict.
fn assert_fell_back_unenforced(metadata: &allama::chat::extract::ChatExtractionMetadata) {
    assert_eq!(metadata.mode, ChatExtractionMode::ForcedTool);
    let fallback = metadata.auto_fallback.as_ref().expect("auto_fallback");
    assert_eq!(fallback.field, RequestField::ResponseFormat);
    assert!(fallback.message.contains("map-typed"), "{fallback:?}");
    assert_eq!(metadata.passes, 1);
    assert_eq!(metadata.enforcement, SchemaEnforcement::NotEnforced);
    assert_strict_dropped(&metadata.degradations);
}

/// The only request sent is the forced-tool one: the native attempt was
/// refused before sending.
fn assert_only_the_forced_tool_attempt_was_sent(transport: &ReplayTransport) {
    let [body] = bodies(transport).try_into().expect("one request");
    assert!(body.get("output_config").is_none(), "{body}");
    assert_eq!(body["tools"][0]["name"], "submit_structured_output");
    assert!(body["tools"][0].get("strict").is_none(), "{body}");
    assert_eq!(body["tool_choice"]["name"], "submit_structured_output");
}

/// `Auto` picks native output on a model with structured outputs, and
/// Anthropic refuses the map before sending; extraction then succeeds in
/// forced-tool mode under `Warn`, without enforcement.
#[tokio::test]
async fn auto_extraction_into_a_map_falls_back_to_forced_tool_under_warn() {
    let transport = ReplayTransport::new([tool_use(
        "submit_structured_output",
        json!({"title": "Charged twice", "labels": {"area": "billing"}}),
    )]);
    let extracted = provider(&transport)
        .extract::<Ticket>(&auto_extraction_request())
        .await
        .unwrap();
    assert_eq!(extracted.value.labels["area"], "billing");
    assert_fell_back_unenforced(&extracted.metadata);
    assert_only_the_forced_tool_attempt_was_sent(&transport);
}

#[tokio::test]
async fn streamed_auto_extraction_into_a_map_falls_back_to_forced_tool_under_warn() {
    let transport = ReplayTransport::new([ticket_tool_stream()]);
    let mut stream = provider(&transport)
        .extract_stream::<Ticket>(&auto_extraction_request())
        .await
        .unwrap();
    assert_eq!(stream.mode(), ChatExtractionMode::ForcedTool);
    while let Some(event) = futures_util::StreamExt::next(&mut stream).await {
        event.unwrap();
    }
    let extracted = stream.finish().unwrap();
    assert_eq!(extracted.value.title, "Charged twice");
    assert_eq!(extracted.value.labels["area"], "billing");
    assert_fell_back_unenforced(&extracted.metadata);
    assert_only_the_forced_tool_attempt_was_sent(&transport);
}

/// Under `Error` the forced-tool attempt cannot enforce the map either, so
/// the extraction fails before sending anything, and records the switch.
/// The native refusal is the primary cause, and the message is about the
/// caller's type.
#[tokio::test]
async fn auto_extraction_into_a_map_fails_under_error_after_the_fallback() {
    let transport = ReplayTransport::new([]);
    let request = auto_extraction_request().unsupported_policy(UnsupportedPolicy::Error);
    let err = provider(&transport)
        .extract::<Ticket>(&request)
        .await
        .unwrap_err();
    assert_eq!(
        err.unsupported_field(),
        Some(RequestField::ResponseFormat),
        "{err:?}"
    );
    let message = err.message();
    assert!(
        message.starts_with(&format!(
            "`Ticket` cannot be extracted from `{MODEL}` with its output constrained: native output was refused"
        )),
        "{message}"
    );
    assert!(
        message.contains("forced-tool mode was refused too"),
        "{message}"
    );
    assert!(!message.contains("submit_structured_output"), "{message}");
    assert!(!message.contains("was sent"), "{message}");
    let record = allama::chat::extract::ChatExtractionMetadata::from_error(&err).unwrap();
    assert_eq!(record.mode, ChatExtractionMode::ForcedTool);
    assert_eq!(
        record.auto_fallback.map(|f| f.field),
        Some(RequestField::ResponseFormat)
    );
    assert!(transport.requests().is_empty(), "nothing is sent");
}
