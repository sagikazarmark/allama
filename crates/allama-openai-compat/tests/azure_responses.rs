//! The Azure preset's Responses API provider: its address, key, dialect,
//! and a conversation moving between Azure's two chat APIs.

use allama::prelude::*;
use allama::{ContentBlock, ProviderIdentity, RequestField, Subject, WarningKind};
use allama_http::http::{HeaderValue, StatusCode};
use allama_http::{Credential, HttpRequestOptions, ReplayResponse, ReplayTransport};
use allama_openai_compat::ResponsesProvider;
use allama_openai_compat::providers::Azure;
use serde_json::{Value, json};

fn responses_text() -> ReplayResponse {
    ReplayResponse::json_value(
        StatusCode::OK,
        &json!({
            "id": "resp_1", "object": "response", "status": "completed", "model": "gpt-5-mini",
            "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "Hi.", "annotations": []}]}],
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
        }),
    )
    .header("apim-request-id", HeaderValue::from_static("apim-1"))
}

/// A reasoning turn that ends in a function call, with the encrypted
/// reasoning and item ids only the Responses API takes back.
fn responses_tool_call() -> ReplayResponse {
    ReplayResponse::json_value(
        StatusCode::OK,
        &json!({
            "id": "resp_2", "object": "response", "status": "completed", "model": "gpt-5-mini",
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "gAAA-opaque"},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_weather",
                 "arguments": "{\"city\":\"Paris\"}", "status": "completed"}
            ],
            "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
        }),
    )
}

fn chat_text() -> ReplayResponse {
    ReplayResponse::json(
        StatusCode::OK,
        r#"{"id":"c1","model":"gpt-5-mini","choices":[{"index":0,"message":{"role":"assistant","content":"Sunny."},"finish_reason":"stop"}]}"#,
    )
}

fn azure_responses(transport: &ReplayTransport) -> ResponsesProvider {
    Azure::responses_builder("res", "provider-key")
        .unwrap()
        .http_transport(transport.clone())
        .build()
        .unwrap()
}

fn header_values(transport: &ReplayTransport, index: usize, name: &str) -> Vec<String> {
    transport.requests()[index]
        .headers
        .get_all(name)
        .iter()
        .map(|value| value.to_str().unwrap().to_owned())
        .collect()
}

fn body(transport: &ReplayTransport, index: usize) -> Value {
    serde_json::from_slice(transport.requests()[index].body.as_ref()).unwrap()
}

fn weather_request(deployment: &str) -> ChatRequest {
    ChatRequest::new(deployment)
        .tools([ToolDefinition::new(
            "get_weather",
            json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
        )])
        .user("Weather in Paris?")
}

#[tokio::test]
async fn reaches_the_resource_with_its_key() {
    let transport = ReplayTransport::new([responses_text()]);
    let provider = azure_responses(&transport);
    assert_eq!(provider.provider_name(), Azure::PROVIDER_NAME);
    assert_eq!(
        provider.responses_url(),
        "https://res.openai.azure.com/openai/v1/responses"
    );
    assert_eq!(provider.quirks(), &Azure::responses_quirks());
    assert!(provider.reads_namespace(Azure::NAMESPACE));
    assert!(!provider.reads_namespace("openai"));

    let response = provider
        .chat(&ChatRequest::new("my-deployment").user("Hi"))
        .await
        .unwrap();
    assert_eq!(response.text().as_deref(), Some("Hi."));
    assert_eq!(response.provider.as_deref(), Some("azure"));
    assert_eq!(response.request_id.as_deref(), Some("apim-1"));

    let sent = transport.requests();
    assert_eq!(
        sent[0].uri.to_string(),
        "https://res.openai.azure.com/openai/v1/responses"
    );
    assert_eq!(header_values(&transport, 0, "api-key"), ["provider-key"]);
    assert!(header_values(&transport, 0, "authorization").is_empty());
    let sent = body(&transport, 0);
    assert_eq!(sent["model"], "my-deployment");
    assert_eq!(sent["store"], false);
}

#[test]
fn addresses_an_endpoint_and_checks_the_resource() {
    let transport = ReplayTransport::default;
    for (endpoint, expected) in [
        (
            "https://proj.services.ai.azure.com/",
            "https://proj.services.ai.azure.com/openai/v1/responses",
        ),
        (
            "https://r.openai.azure.com/openai/v1",
            "https://r.openai.azure.com/openai/v1/responses",
        ),
    ] {
        let provider = Azure::responses_builder_with_endpoint(endpoint, "k")
            .unwrap()
            .http_transport(transport())
            .build()
            .unwrap();
        assert_eq!(provider.responses_url(), expected);
    }

    // The same checks as the Chat Completions builders: nothing can move
    // the key to another host.
    for resource in ["attacker.example/x?", "a.b", "-lead", "x"] {
        let err = Azure::responses_builder(resource, "KEY").unwrap_err();
        assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    }
    assert!(Azure::responses_builder("res", " ").is_err());
    assert!(
        Azure::responses_builder_with_endpoint("https://x.example/?api-version=1", "k").is_err()
    );
}

/// A tenant's key goes in `api-key`, replacing a configured Entra ID token,
/// as on the Chat Completions provider.
#[tokio::test]
async fn a_tenant_key_replaces_a_configured_entra_token() {
    let transport = ReplayTransport::new([responses_text(), responses_text()]);
    let provider = Azure::responses_builder("res", "unused")
        .unwrap()
        .bearer_token("entra-token")
        .http_transport(transport.clone())
        .build()
        .unwrap();

    let tenant = ChatRequest::new("d")
        .user("Hi")
        .with_option(HttpRequestOptions::new().credential("azure", Credential::api_key("TENANT")));
    provider.chat(&tenant).await.unwrap();
    assert_eq!(header_values(&transport, 0, "api-key"), ["TENANT"]);
    assert!(header_values(&transport, 0, "authorization").is_empty());

    provider
        .chat(&ChatRequest::new("d").user("Hi"))
        .await
        .unwrap();
    assert_eq!(
        header_values(&transport, 1, "authorization"),
        ["Bearer entra-token"]
    );
}

/// Azure's strict dialect applies on every deployment, as on Chat
/// Completions: unsupported keywords removed, parallel calls turned off.
#[tokio::test]
async fn strict_tools_use_the_azure_dialect() {
    let transport = ReplayTransport::new([responses_text()]);
    let request = ChatRequest::new("my-deployment")
        .user("File it.")
        .tools([ToolDefinition::new(
            "file",
            json!({
                "type": "object",
                "properties": {"code": {"type": "string", "pattern": "^[A-Z]+$", "maxLength": 8}},
                "required": ["code"]
            }),
        )
        .strict(true)]);
    let response = azure_responses(&transport).chat(&request).await.unwrap();

    let sent = body(&transport, 0);
    assert_eq!(sent["tools"][0]["strict"], true);
    assert_eq!(
        sent["tools"][0]["parameters"]["properties"]["code"],
        json!({"type": "string"})
    );
    assert_eq!(sent["parallel_tool_calls"], false);
    let tools: Vec<_> = response
        .warnings
        .iter()
        .filter(|w| {
            w.subject
                == Subject::Field {
                    field: RequestField::Tools,
                }
        })
        .collect();
    assert_eq!(tools.len(), 1, "{tools:?}");
    assert_eq!(tools[0].kind, WarningKind::Approximated);
    assert!(tools[0].message.contains("Azure"), "{}", tools[0]);
}

/// Both APIs share the preset's namespace, so a conversation can move
/// between them: Chat Completions replays the tool call and drops the
/// Responses-only reasoning, and Responses reads the Chat Completions turn.
#[tokio::test]
async fn a_conversation_moves_between_the_two_apis() {
    let transport = ReplayTransport::new([responses_tool_call(), chat_text(), responses_text()]);
    let responses = azure_responses(&transport);
    let chat = Azure::builder("res", "provider-key")
        .unwrap()
        .http_transport(transport.clone())
        .build()
        .unwrap();

    let mut request = weather_request("my-deployment");
    let first = responses.chat(&request).await.unwrap();
    assert!(matches!(first.content[0], ContentBlock::Reasoning(_)));
    request.push_message(first.to_assistant_message());
    request.push_tool_result(first.first_tool_call().unwrap(), "Sunny");

    let second = chat.chat(&request).await.unwrap();
    assert_eq!(second.text().as_deref(), Some("Sunny."));
    let sent = body(&transport, 1);
    assert_eq!(
        sent["messages"][1]["tool_calls"][0]["id"], "call_1",
        "{sent}"
    );
    assert_eq!(sent["messages"][2]["tool_call_id"], "call_1", "{sent}");
    assert!(!sent.to_string().contains("gAAA-opaque"), "{sent}");

    request.push_message(second.to_assistant_message());
    request.push_message(Message::user("Thanks."));
    let third = responses.chat(&request).await.unwrap();
    assert!(third.warnings.is_empty(), "{:?}", third.warnings);
    let sent = body(&transport, 2);
    let input = sent["input"].as_array().unwrap();
    assert_eq!(input[1]["type"], "reasoning", "{sent}");
    assert_eq!(input[1]["encrypted_content"], "gAAA-opaque");
    assert_eq!(input[2]["type"], "function_call");
    assert_eq!(input[3]["type"], "function_call_output");
    assert_eq!(input[4]["role"], "assistant");
}
