//! Cross-provider history: a Claude conversation with thinking continues
//! on Gemini 3 through `FallbackProvider`. Anthropic's opaque state must not leak into the
//! Gemini request, the loss must be reported as warnings, and Gemini 3's
//! signature requirement must be satisfied.
//!
//! The history keeps the Claude model id; `MapModelProvider` gives the Gemini
//! side its own.

use std::borrow::Cow;
use std::path::PathBuf;

use allama::wrappers::{FallbackProvider, MapModelProvider};
use allama::{
    ChatProvider, ChatRequest, ContentBlock, Message, OtherContent, ReasoningEffort,
    ToolDefinition, WarningKind,
};
use allama_conformance::{MockHttpResponse, TestHttpServer};
use allama_http::http::StatusCode;
use allama_openai::CustomToolCall;
use serde_json::{Value, json};

fn weather_tool() -> ToolDefinition {
    ToolDefinition::new(
        "get_weather",
        json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
    )
    .description("Get the current weather for a city.")
}

/// The Gemini model that takes over any Claude conversation.
fn gemini_model(_claude_model: &str) -> Cow<'_, str> {
    "gemini-3.5-flash".into()
}

fn recorded_anthropic_response() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../allama-anthropic/fixtures/response_raw.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[tokio::test]
async fn claude_thinking_history_continues_on_gemini3_via_fallback() {
    // Turn 1 on Anthropic (recorded: thinking + tool_use on claude-haiku-4-5).
    let recorded = recorded_anthropic_response();
    let signature = recorded["content"][0]["signature"]
        .as_str()
        .unwrap()
        .to_owned();
    let anthropic_server = TestHttpServer::spawn([
        MockHttpResponse::json_value(StatusCode::OK, &recorded),
        MockHttpResponse::json_value(
            StatusCode::from_u16(529).unwrap(),
            &json!({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}),
        ),
    ])
    .await;
    let anthropic =
        allama_anthropic::Provider::new_with_base_url("test-key", anthropic_server.url()).unwrap();
    let request = ChatRequest::new("claude-haiku-4-5-20251001")
        .user("What's the weather in Paris? Use the tool.")
        .tools([weather_tool()])
        .reasoning(ReasoningEffort::Low);
    let first = anthropic.chat(&request).await.unwrap();
    assert!(matches!(first.content[0], ContentBlock::Reasoning(_)));
    let call = first.tool_calls().next().unwrap().clone();

    let mut history = request.clone();
    history.push_message(first.to_assistant_message());
    history.push_message(Message::tool_result(
        call.id.clone(),
        call.name.clone(),
        "18°C, clear",
    ));

    // Turn 2: Anthropic is overloaded, Gemini 3 answers.
    let gemini_server = TestHttpServer::spawn([MockHttpResponse::json_value(
        StatusCode::OK,
        &json!({
            "responseId": "resp-2", "modelVersion": "gemini-3.5-flash",
            "candidates": [{"content": {"role": "model", "parts": [{"text": "It is 18°C and clear in Paris."}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 40, "candidatesTokenCount": 9, "totalTokenCount": 49}
        }),
    )])
    .await;
    let gemini =
        allama_gemini::Provider::new_with_base_url("test-key", gemini_server.url()).unwrap();
    let response = FallbackProvider::new(anthropic, MapModelProvider::new(gemini, gemini_model))
        .chat(&history)
        .await
        .unwrap();

    assert_eq!(response.provider.as_deref(), Some("gemini"));
    // Each side received a model id from its own catalog.
    let anthropic_requests = anthropic_server.recorded_requests().await;
    assert_eq!(
        anthropic_requests[1].body_json()["model"],
        "claude-haiku-4-5-20251001"
    );
    let gemini_path = &gemini_server.recorded_requests().await[0].path;
    assert!(
        gemini_path.contains("models/gemini-3.5-flash:"),
        "{gemini_path}"
    );
    assert_eq!(
        response.text().as_deref(),
        Some("It is 18°C and clear in Paris.")
    );
    let kinds: Vec<WarningKind> = response.warnings.iter().map(|w| w.kind).collect();
    assert!(
        kinds.contains(&WarningKind::Dropped),
        "{:?}",
        response.warnings
    );
    assert!(
        kinds.contains(&WarningKind::Approximated),
        "{:?}",
        response.warnings
    );

    let sent = gemini_server.recorded_requests().await[0].body_json();
    let text = sent.to_string();
    assert!(
        !text.contains(&signature),
        "Anthropic signature leaked to Gemini"
    );
    let model_parts = &sent["contents"][1]["parts"];
    let function_call = model_parts
        .as_array()
        .unwrap()
        .iter()
        .find(|part| part.get("functionCall").is_some())
        .expect("function call replayed");
    assert_eq!(function_call["functionCall"]["name"], "get_weather");
    assert_eq!(
        function_call["thoughtSignature"],
        "skip_thought_signature_validator"
    );
    assert_eq!(
        sent["contents"][2]["parts"][0]["functionResponse"]["name"],
        "get_weather"
    );
}

/// Live variant: `cargo test -p allama-examples --test cross_provider -- --ignored`
/// with ANTHROPIC_API_KEY and GEMINI_API_KEY.
#[tokio::test]
#[ignore]
async fn live_claude_thinking_history_continues_on_gemini3_via_fallback() {
    let anthropic = allama_anthropic::Provider::from_env().unwrap();
    let request = ChatRequest::new("claude-haiku-4-5-20251001")
        .user("What's the weather in Paris? Use the tool.")
        .tools([weather_tool()])
        .reasoning(ReasoningEffort::Low)
        .max_output_tokens(2048);
    let first = anthropic.chat(&request).await.unwrap();
    assert!(
        first
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::Reasoning(_))),
        "expected thinking: {:?}",
        first.content
    );
    let call = first.tool_calls().next().expect("tool call").clone();

    let mut history = request.clone();
    history.push_message(first.to_assistant_message());
    history.push_message(Message::tool_result(
        call.id.clone(),
        call.name.clone(),
        "18°C, clear",
    ));

    // The primary is unreachable, standing in for an Anthropic outage.
    let unreachable =
        allama_anthropic::Provider::new_with_base_url("unused", "http://127.0.0.1:9").unwrap();
    let gemini = allama_gemini::Provider::from_env().unwrap();
    let response = FallbackProvider::new(unreachable, MapModelProvider::new(gemini, gemini_model))
        .chat(&history)
        .await
        .unwrap();
    assert_eq!(response.provider.as_deref(), Some("gemini"));
    assert!(
        response.text().is_some_and(|t| t.contains("18")),
        "{:?}",
        response.text()
    );
    assert!(
        response
            .warnings
            .iter()
            .any(|w| w.kind == WarningKind::Dropped)
    );
}

/// A Responses turn that called a custom tool (as the Responses adapter
/// decodes it, with its item `id` and `status`), and the caller's answer.
fn custom_tool_history() -> ChatRequest {
    let data = json!({
        "id": "ctc_1", "call_id": "call_custom_1", "name": "sql",
        "input": "SELECT count(*) FROM users", "status": "completed",
    });
    let call = OtherContent::new("custom_tool_call", data.as_object().cloned().unwrap())
        .with_origin("openai");
    let answer = CustomToolCall::from_content(&call)
        .unwrap()
        .output("42 users");
    ChatRequest::new("gpt-5.1")
        .user("How many users signed up?")
        .message(Message::assistant_blocks(vec![
            ContentBlock::text("Counting them."),
            ContentBlock::Other(call),
        ]))
        .message(Message::user(vec![answer]))
}

/// A Responses provider whose endpoint is unavailable.
async fn unavailable_responses() -> (TestHttpServer, allama_openai::ResponsesProvider) {
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(
        StatusCode::SERVICE_UNAVAILABLE,
        &json!({"error": {"message": "The server is overloaded.", "type": "server_error"}}),
    )])
    .await;
    let provider =
        allama_openai::ResponsesProvider::new_with_base_url("test-key", server.url()).unwrap();
    (server, provider)
}

/// The Anthropic model that takes over any conversation.
fn claude_model(_model: &str) -> Cow<'_, str> {
    "claude-haiku-4-5-20251001".into()
}

/// The Ollama model that takes over any conversation.
fn ollama_model(_model: &str) -> Cow<'_, str> {
    "llama3.2".into()
}

fn chat_completion(text: &str) -> Value {
    json!({
        "id": "chatcmpl-1", "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}
    })
}

/// Fails over from Responses to `fallback` over the custom tool history,
/// checks that `validate_chat_request` agrees, and returns the response and
/// the body the fallback side received.
async fn fail_over_custom_tool_turn<F>(
    fallback: F,
    server: &TestHttpServer,
) -> (allama::ChatResponse, Value)
where
    F: ChatProvider + 'static,
{
    let history = custom_tool_history();
    let (responses_server, responses) = unavailable_responses().await;
    assert!(responses.validate_chat_request(&history).is_ok());
    assert!(fallback.validate_chat_request(&history).is_ok());
    let response = FallbackProvider::new(responses, fallback)
        .chat(&history)
        .await
        .unwrap();
    let primary = &responses_server.recorded_requests().await[0].body_json();
    assert_eq!(
        primary["input"][3]["type"], "custom_tool_call_output",
        "{primary:#}"
    );
    let sent = server.recorded_requests().await[0].body_json();
    (response, sent)
}

/// The custom call and its answer are another provider's content there:
/// both are dropped with warnings, and the conversation ends on the
/// question.
fn assert_dropped_pair(response: &allama::ChatResponse, sent: &Value) {
    let dropped = response
        .warnings
        .iter()
        .filter(|w| w.kind == WarningKind::Dropped)
        .count();
    assert!(dropped >= 2, "{:?}", response.warnings);
    let text = sent.to_string();
    for leaked in ["call_custom_1", "SELECT count", "42 users"] {
        assert!(
            !text.contains(leaked),
            "{leaked} reached the wire: {sent:#}"
        );
    }
}

#[tokio::test]
async fn a_custom_tool_turn_fails_over_from_responses_to_anthropic() {
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(
        StatusCode::OK,
        &json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-4-5-20251001",
            "content": [{"type": "text", "text": "About 42."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 9, "output_tokens": 3}
        }),
    )])
    .await;
    let anthropic =
        allama_anthropic::Provider::new_with_base_url("test-key", server.url()).unwrap();
    let fallback = MapModelProvider::new(anthropic, claude_model);
    let (response, sent) = fail_over_custom_tool_turn(fallback, &server).await;
    assert_eq!(response.provider.as_deref(), Some("anthropic"));
    assert_dropped_pair(&response, &sent);
    assert_eq!(
        sent["messages"],
        json!([{"role": "user", "content": [{"type": "text", "text": "How many users signed up?"}]}])
    );
}

#[tokio::test]
async fn a_custom_tool_turn_fails_over_from_responses_to_gemini() {
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(
        StatusCode::OK,
        &json!({
            "responseId": "resp-1", "modelVersion": "gemini-3.5-flash",
            "candidates": [{"content": {"role": "model", "parts": [{"text": "About 42."}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 3, "totalTokenCount": 12}
        }),
    )])
    .await;
    let gemini = allama_gemini::Provider::new_with_base_url("test-key", server.url()).unwrap();
    let fallback = MapModelProvider::new(gemini, gemini_model);
    let (response, sent) = fail_over_custom_tool_turn(fallback, &server).await;
    assert_eq!(response.provider.as_deref(), Some("gemini"));
    assert_dropped_pair(&response, &sent);
    assert_eq!(
        sent["contents"],
        json!([{"role": "user", "parts": [{"text": "How many users signed up?"}]}])
    );
}

#[tokio::test]
async fn a_custom_tool_turn_fails_over_from_responses_to_a_compatible_endpoint() {
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(
        StatusCode::OK,
        &chat_completion("About 42."),
    )])
    .await;
    let ollama = allama_openai_compat::providers::Ollama::builder(server.url())
        .unwrap()
        .build()
        .unwrap();
    let fallback = MapModelProvider::new(ollama, ollama_model);
    let (response, sent) = fail_over_custom_tool_turn(fallback, &server).await;
    assert_eq!(response.provider.as_deref(), Some("ollama"));
    assert_dropped_pair(&response, &sent);
    assert_eq!(
        sent["messages"],
        json!([{"role": "user", "content": "How many users signed up?"}])
    );
}

/// Chat Completions shares the custom call representation: the call is
/// replayed as a `custom` tool call and the answer lifted into a `tool`
/// message, with nothing dropped.
#[tokio::test]
async fn a_custom_tool_turn_round_trips_from_responses_to_chat_completions() {
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(
        StatusCode::OK,
        &chat_completion("42 users."),
    )])
    .await;
    let chat = allama_openai::Provider::new_with_base_url("test-key", server.url()).unwrap();
    let (response, sent) = fail_over_custom_tool_turn(chat, &server).await;
    assert_eq!(response.provider.as_deref(), Some("openai"));
    // Only the failover is reported: nothing was dropped.
    let kinds: Vec<_> = response.warnings.iter().map(|w| w.kind).collect();
    assert_eq!(kinds, [WarningKind::Fallback], "{:?}", response.warnings);
    assert_eq!(
        sent["messages"],
        json!([
            {"role": "user", "content": "How many users signed up?"},
            {"role": "assistant", "content": "Counting them.", "tool_calls": [
                {"id": "call_custom_1", "type": "custom",
                 "custom": {"name": "sql", "input": "SELECT count(*) FROM users"}}
            ]},
            {"role": "tool", "content": "42 users", "tool_call_id": "call_custom_1"}
        ])
    );
}

/// The other direction: a custom call decoded from Chat Completions (no
/// item `id` or `status`) continues on Responses.
#[tokio::test]
async fn a_chat_completions_custom_call_continues_on_responses() {
    let chat_server = TestHttpServer::spawn([MockHttpResponse::json_value(
        StatusCode::OK,
        &json!({
            "id": "chatcmpl-1", "model": "gpt-5.1",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": null,
                "tool_calls": [{"id": "call_c", "type": "custom",
                                "custom": {"name": "sql", "input": "SELECT 1"}}]
            }}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}
        }),
    )])
    .await;
    let chat = allama_openai::Provider::new_with_base_url("test-key", chat_server.url()).unwrap();
    let request = ChatRequest::new("gpt-5.1").user("How many users?");
    let first = chat.chat(&request).await.unwrap();
    assert_eq!(first.finish_reason, Some(allama::FinishReason::ToolCalls));
    let pending = allama_openai::ClientToolCall::pending(&first.content);
    let [allama_openai::ClientToolCall::Custom(call)] = pending.as_slice() else {
        panic!("expected one custom call: {:?}", first.content)
    };
    let mut history = request.clone();
    history.push_message(first.to_assistant_message());
    history.push_message(Message::user(vec![call.output("1")]));

    let responses_server = TestHttpServer::spawn([MockHttpResponse::json_value(
        StatusCode::OK,
        &json!({
            "id": "resp_1", "object": "response", "status": "completed", "model": "gpt-5.1",
            "output": [{"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "One.", "annotations": []}]}],
            "usage": {"input_tokens": 9, "output_tokens": 2, "total_tokens": 11}
        }),
    )])
    .await;
    let responses =
        allama_openai::ResponsesProvider::new_with_base_url("test-key", responses_server.url())
            .unwrap();
    let response = responses.chat(&history).await.unwrap();
    assert!(response.warnings.is_empty(), "{:?}", response.warnings);
    let sent = responses_server.recorded_requests().await[0].body_json();
    assert_eq!(
        sent["input"].as_array().unwrap()[1..],
        [
            json!({"type": "custom_tool_call", "call_id": "call_c", "name": "sql", "input": "SELECT 1"}),
            json!({"type": "custom_tool_call_output", "call_id": "call_c", "output": "1"})
        ]
    );
}

/// Lane I (cross-provider tool-call ids): Gemini 2.5 sends function calls
/// without ids, so the adapter makes them up. A conversation with those
/// calls in its history must continue on Anthropic, which refuses any id
/// outside `^[a-zA-Z0-9_-]+$` (verified live: `gemini:abc:call:0` got a
/// 400), and on Mistral, which takes only nine letters and digits; each
/// call stays paired with its result.
#[tokio::test]
async fn a_gemini_tool_turn_with_made_up_ids_continues_on_anthropic_and_mistral() {
    // Turn 1 on Gemini 2.5 (recorded), with a second, parallel call.
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../allama-gemini/fixtures/response_raw.json");
    let mut recorded: Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let parts = recorded["candidates"][0]["content"]["parts"]
        .as_array_mut()
        .unwrap();
    parts.push(json!({"functionCall": {"name": "search", "args": {"q": "tokio docs"}}}));
    let gemini_server = TestHttpServer::spawn([
        MockHttpResponse::json_value(StatusCode::OK, &recorded),
        MockHttpResponse::json_value(
            StatusCode::SERVICE_UNAVAILABLE,
            &json!({"error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}}),
        ),
    ])
    .await;
    let gemini =
        allama_gemini::Provider::new_with_base_url("test-key", gemini_server.url()).unwrap();
    let request = ChatRequest::new("gemini-2.5-pro").user("Find the docs.");
    let first = gemini.chat(&request).await.unwrap();
    let calls: Vec<_> = first.tool_calls().cloned().collect();
    assert_eq!(calls.len(), 2);
    let mut history = request.clone();
    history.push_message(first.to_assistant_message());
    // Answered out of order, as `history::validate` allows.
    for call in calls.iter().rev() {
        history.push_message(Message::tool_result(
            &call.id,
            &call.name,
            format!("{} results", call.id),
        ));
    }

    // Turn 2: Gemini is overloaded, Anthropic answers.
    let anthropic_server = TestHttpServer::spawn([MockHttpResponse::json_value(
        StatusCode::OK,
        &json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-4-5-20251001",
            "content": [{"type": "text", "text": "Here are the docs."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 9, "output_tokens": 3}
        }),
    )])
    .await;
    let anthropic =
        allama_anthropic::Provider::new_with_base_url("test-key", anthropic_server.url()).unwrap();
    let response = FallbackProvider::new(gemini, MapModelProvider::new(anthropic, claude_model))
        .chat(&history)
        .await
        .unwrap();
    assert_eq!(response.provider.as_deref(), Some("anthropic"));
    let sent = anthropic_server.recorded_requests().await[0].body_json();
    let messages = sent["messages"].as_array().unwrap();
    let uses: Vec<&str> = messages[1]["content"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|block| block["type"] == "tool_use")
        .map(|block| block["id"].as_str().unwrap())
        .collect();
    let results: Vec<(&str, &str)> = messages[2]["content"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|block| block["type"] == "tool_result")
        .map(|block| {
            (
                block["tool_use_id"].as_str().unwrap(),
                block["content"].as_str().unwrap(),
            )
        })
        .collect();
    assert_eq!(uses.len(), 2, "{sent:#}");
    for id in &uses {
        assert!(
            !id.is_empty()
                && id
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
            "Anthropic refuses the tool_use id `{id}`"
        );
    }
    assert_paired(&calls, &uses, &results);

    // The same history on Mistral.
    let mistral_server = TestHttpServer::spawn([MockHttpResponse::json_value(
        StatusCode::OK,
        &chat_completion("Here."),
    )])
    .await;
    let mistral = allama_openai_compat::providers::Mistral::builder("test-key")
        .unwrap()
        .base_url(mistral_server.url())
        .build()
        .unwrap();
    let mut on_mistral = history.clone();
    on_mistral.model = "mistral-small-latest".into();
    mistral.chat(&on_mistral).await.unwrap();
    let sent = mistral_server.recorded_requests().await[0].body_json();
    let messages = sent["messages"].as_array().unwrap();
    let uses: Vec<&str> = messages[1]["tool_calls"]
        .as_array()
        .unwrap()
        .iter()
        .map(|call| call["id"].as_str().unwrap())
        .collect();
    let results: Vec<(&str, &str)> = messages[2..]
        .iter()
        .filter(|message| message["role"] == "tool")
        .map(|message| {
            (
                message["tool_call_id"].as_str().unwrap(),
                message["content"].as_str().unwrap(),
            )
        })
        .collect();
    for id in &uses {
        assert!(
            id.len() == 9 && id.chars().all(|c| c.is_ascii_alphanumeric()),
            "Mistral refuses the tool call id `{id}`"
        );
    }
    assert_paired(&calls, &uses, &results);
}

/// Each sent call id is answered by the result written for that call, and
/// distinct calls keep distinct ids.
fn assert_paired(calls: &[allama::ToolCall], uses: &[&str], results: &[(&str, &str)]) {
    assert_eq!(uses.len(), calls.len());
    assert_ne!(uses[0], uses[1], "two calls share an id");
    for (call, sent_id) in calls.iter().zip(uses) {
        let answer = results
            .iter()
            .find(|(id, _)| id == sent_id)
            .unwrap_or_else(|| panic!("no result answers `{sent_id}`: {results:?}"));
        assert_eq!(answer.1, format!("{} results", call.id));
    }
}
