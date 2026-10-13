//! Which strict dialect each endpoint gets, and the parallel tool call rule
//! that goes with it: OpenAI's `ft:` convention applies only where the
//! quirks opt in (`fine_tuned_dialect`), Azure has its own dialect on every
//! deployment, and dialects whose strict mode breaks under parallel calls
//! turn them off when a strict tool is sent.

use allama::prelude::*;
use allama::{RequestField, Subject, UnsupportedPolicy, Warning, WarningKind};
use allama_http::{ReplayResponse, ReplayTransport};
use allama_openai_compat::providers::{Azure, Mistral};
use allama_openai_compat::{
    CompatQuirks, Provider, ProviderBuilder, ResponsesProvider, ResponsesQuirks,
};
use serde_json::{Value, json};

fn chat_ok() -> ReplayResponse {
    ReplayResponse::json(
        200,
        json!({
            "id": "c1", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })
        .to_string(),
    )
}

fn responses_ok() -> ReplayResponse {
    ReplayResponse::json(
        200,
        json!({
            "id": "resp_1", "object": "response", "status": "completed", "model": "m",
            "output": [{"type": "message", "id": "msg_1", "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "ok", "annotations": []}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        })
        .to_string(),
    )
}

fn bodies(transport: &ReplayTransport) -> Vec<Value> {
    transport
        .requests()
        .iter()
        .filter_map(|r| r.body.as_ref())
        .map(|body| serde_json::from_slice(body).expect("JSON request body"))
        .collect()
}

/// A tool schema with keywords the fine-tuned and Azure dialects remove.
fn constrained() -> Value {
    json!({
        "type": "object",
        "properties": {
            "code": {"type": "string", "pattern": "^[A-Z]+$", "maxLength": 8},
            "when": {"type": "string", "format": "date-time"},
            "count": {"type": "integer", "minimum": 0, "maximum": 9},
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "uniqueItems": true}
        },
        "required": ["code", "when", "count", "tags"]
    })
}

fn tool_request(model: &str) -> ChatRequest {
    ChatRequest::new(model)
        .user("File it.")
        .tools([ToolDefinition::new("file", constrained()).strict(true)])
}

async fn send_chat(
    builder: ProviderBuilder,
    request: &ChatRequest,
) -> (Result<ChatResponse>, Vec<Value>) {
    let transport = ReplayTransport::new([chat_ok()]);
    let provider: Provider = builder.http_transport(transport.clone()).build().unwrap();
    let result = provider.chat(request).await;
    (result, bodies(&transport))
}

async fn send_responses(
    quirks: ResponsesQuirks,
    request: &ChatRequest,
) -> (Result<ChatResponse>, Vec<Value>) {
    let transport = ReplayTransport::new([responses_ok()]);
    let provider = ResponsesProvider::builder()
        .base_url("http://upstream/v1")
        .quirks(quirks)
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let result = provider.chat(request).await;
    (result, bodies(&transport))
}

fn openai_chat() -> ProviderBuilder {
    Provider::builder()
        .base_url("http://upstream/v1")
        .quirks(CompatQuirks::openai())
}

/// Builds a provider for one endpoint (builders are not `Clone`).
type Builder = fn() -> ProviderBuilder;

fn azure() -> ProviderBuilder {
    Azure::builder("res", "k").unwrap()
}

fn tools_warnings(warnings: &[Warning]) -> Vec<&Warning> {
    warnings
        .iter()
        .filter(|w| {
            w.subject
                == Subject::Field {
                    field: RequestField::Tools,
                }
        })
        .collect()
}

#[tokio::test]
async fn mistral_fine_tuned_models_keep_their_keywords() {
    let request = tool_request("ft:open-mistral-7b:acme:20240101:abc123");
    let (result, sent) = send_chat(Mistral::builder("k").unwrap(), &request).await;
    let response = result.unwrap();
    let function = &sent[0]["tools"][0]["function"];
    assert_eq!(function["strict"], true);
    let properties = &function["parameters"]["properties"];
    assert_eq!(properties["code"]["pattern"], "^[A-Z]+$");
    assert_eq!(properties["code"]["maxLength"], 8);
    assert_eq!(properties["when"]["format"], "date-time");
    // Only OpenAI's own dialects turn parallel calls off.
    assert!(sent[0].get("parallel_tool_calls").is_none(), "{}", sent[0]);
    // Mistral's enforcement of the dialect is not verified.
    let warnings = tools_warnings(&response.warnings);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].message.contains("not known to enforce"));
    assert!(!warnings[0].message.contains("removed"));
}

#[tokio::test]
async fn the_fine_tuned_rule_follows_the_responses_quirks() {
    let request = tool_request("ft:gpt-4.1-mini:acme::abc123");

    // A generic Responses endpoint: no `ft:` convention.
    // Its enforcement of the dialect is not verified, and nothing is removed.
    let (result, sent) = send_responses(ResponsesQuirks::new("openai-compat:demo"), &request).await;
    let response = result.unwrap();
    let warnings = tools_warnings(&response.warnings);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].message.contains("not known to enforce"));
    assert_eq!(
        sent[0]["tools"][0]["parameters"]["properties"]["code"]["pattern"],
        "^[A-Z]+$"
    );
    assert!(sent[0].get("parallel_tool_calls").is_none());

    // OpenAI's dialect: the fine-tuned removals and serial calls.
    let (result, sent) = send_responses(ResponsesQuirks::openai(), &request).await;
    let response = result.unwrap();
    let properties = &sent[0]["tools"][0]["parameters"]["properties"];
    assert_eq!(properties["code"], json!({"type": "string"}));
    assert_eq!(sent[0]["parallel_tool_calls"], false);
    let warnings = tools_warnings(&response.warnings);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(
        warnings[0].message.contains("fine-tuned"),
        "{}",
        warnings[0]
    );
}

#[tokio::test]
async fn azure_strips_its_keywords_and_turns_parallel_calls_off() {
    let request = tool_request("my-gpt-4o-deployment");
    let (result, sent) = send_chat(Azure::builder("res", "k").unwrap(), &request).await;
    let response = result.unwrap();
    let function = &sent[0]["tools"][0]["function"];
    assert_eq!(function["strict"], true);
    assert_eq!(
        function["parameters"]["properties"],
        json!({
            "code": {"type": "string"},
            "when": {"type": "string"},
            "count": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string"}}
        })
    );
    assert_eq!(sent[0]["parallel_tool_calls"], false);

    let warnings = tools_warnings(&response.warnings);
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].kind, WarningKind::Approximated);
    assert!(warnings[0].message.contains("Azure"), "{}", warnings[0]);
    for keyword in [
        "format",
        "maxLength",
        "maximum",
        "minItems",
        "minimum",
        "pattern",
        "uniqueItems",
    ] {
        assert!(warnings[0].message.contains(keyword), "{keyword}");
    }
}

#[tokio::test]
async fn azure_sends_no_automatic_false_where_parallel_calls_are_gated() {
    let request = tool_request("my-o3-deployment");

    // A deployment configured without `parallel_tool_calls`.
    let mut quirks = Azure::quirks();
    quirks.supports_parallel_tool_calls = false;
    let builder = Azure::builder("res", "k").unwrap().quirks(quirks);
    let (result, sent) = send_chat(builder, &request).await;
    assert!(
        result
            .unwrap()
            .warnings
            .iter()
            .all(|w| w.kind == WarningKind::Approximated)
    );
    assert!(sent[0].get("parallel_tool_calls").is_none(), "{}", sent[0]);

    // `reasoning_effort: minimal`, under which Azure's reasoning models make
    // no parallel calls.
    let minimal = request.clone().reasoning(ReasoningEffort::Minimal);
    let (result, sent) = send_chat(Azure::builder("res", "k").unwrap(), &minimal).await;
    result.unwrap();
    assert_eq!(sent[0]["reasoning_effort"], "minimal");
    assert!(sent[0].get("parallel_tool_calls").is_none(), "{}", sent[0]);
}

#[tokio::test]
async fn no_automatic_false_without_a_strict_tool() {
    let request = ChatRequest::new("my-gpt-4o-deployment")
        .user("q")
        .tools([ToolDefinition::new("file", constrained())]);
    let (result, sent) = send_chat(Azure::builder("res", "k").unwrap(), &request).await;
    result.unwrap();
    assert!(sent[0].get("parallel_tool_calls").is_none());
    // Strict that degraded (a map) does not count either.
    let map = json!({"type": "object", "additionalProperties": {"type": "string"}});
    let request = ChatRequest::new("my-gpt-4o-deployment")
        .user("q")
        .tools([ToolDefinition::new("labels", map).strict(true)]);
    let (result, sent) = send_chat(Azure::builder("res", "k").unwrap(), &request).await;
    result.unwrap();
    assert_eq!(sent[0]["tools"][0]["function"]["strict"], false);
    assert!(sent[0].get("parallel_tool_calls").is_none());
}

#[tokio::test]
async fn an_explicit_true_survives_on_openai_and_warns_where_strict_breaks() {
    // Base OpenAI models: honoured as asked, with no warning.
    let request = tool_request("gpt-4.1").parallel_tool_calls(true);
    let (result, sent) = send_chat(openai_chat(), &request).await;
    let response = result.unwrap();
    assert_eq!(sent[0]["parallel_tool_calls"], true);
    assert!(response.warnings.is_empty(), "{:?}", response.warnings);

    // Fine-tuned OpenAI models and Azure: sent, with `Unsupported` on the
    // tools (strict is not enforced), and refused under `Error`.
    let builders: [(&str, Builder, &str); 2] = [
        ("openai ft", openai_chat, "ft:gpt-4.1-mini:acme::abc123"),
        ("azure", azure, "deployment"),
    ];
    for (name, builder, model) in builders {
        let request = tool_request(model).parallel_tool_calls(true);
        let (result, sent) = send_chat(builder(), &request).await;
        let response = result.unwrap();
        assert_eq!(sent[0]["parallel_tool_calls"], true, "{name}");
        let unsupported: Vec<_> = tools_warnings(&response.warnings)
            .into_iter()
            .filter(|w| w.kind == WarningKind::Unsupported)
            .collect();
        assert_eq!(unsupported.len(), 1, "{name}: {:?}", response.warnings);
        assert!(
            unsupported[0].message.contains("parallel_tool_calls"),
            "{}",
            unsupported[0]
        );

        let strict = request.unsupported_policy(UnsupportedPolicy::Error);
        let (result, sent) = send_chat(builder(), &strict).await;
        assert!(
            matches!(result, Err(Error::Unsupported { .. })),
            "{name}: {result:?}"
        );
        assert!(sent.is_empty(), "{name}: nothing is sent");
    }

    // The same on Responses.
    let request = tool_request("ft:gpt-4.1-mini:acme::abc123").parallel_tool_calls(true);
    let (result, sent) = send_responses(ResponsesQuirks::openai(), &request).await;
    let response = result.unwrap();
    assert_eq!(sent[0]["parallel_tool_calls"], true);
    assert!(
        response.warnings.iter().any(
            |w| w.kind == WarningKind::Unsupported && w.message.contains("parallel_tool_calls")
        ),
        "{:?}",
        response.warnings
    );
    let request = tool_request("gpt-4.1").parallel_tool_calls(true);
    let (result, sent) = send_responses(ResponsesQuirks::openai(), &request).await;
    assert!(result.unwrap().warnings.is_empty());
    assert_eq!(sent[0]["parallel_tool_calls"], true);
}

#[tokio::test]
async fn an_explicit_false_is_sent_unchanged() {
    for (builder, model) in [
        (openai_chat(), "ft:gpt-4.1-mini:acme::abc123"),
        (Azure::builder("res", "k").unwrap(), "deployment"),
    ] {
        let request = tool_request(model).parallel_tool_calls(false);
        let (result, sent) = send_chat(builder, &request).await;
        assert!(
            result
                .unwrap()
                .warnings
                .iter()
                .all(|w| w.kind == WarningKind::Approximated)
        );
        assert_eq!(sent[0]["parallel_tool_calls"], false);
    }
}
