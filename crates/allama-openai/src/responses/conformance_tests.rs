//! Fixture-based conformance tests for the Responses provider.
//!
//! Files under `fixtures/responses/` without an `_expected` suffix are
//! response bodies recorded from the live API (`scripts/capture-fixture.sh
//! openai-responses`): `text.json` and `incomplete.json` (gpt-4.1-nano),
//! `reasoning_tool.json` and `reasoning_tool_stream.sse` (gpt-5-nano, low
//! effort, summaries), `text_stream.sse` and `parallel_tools_stream.sse`
//! (gpt-4.1-nano), `hosted_tool.json` (gpt-4.1-mini with web search), and
//! the error bodies. `client_tools.json` is written by hand from the API
//! reference (custom, local shell and shell calls in one turn; OpenAI no
//! longer serves the local shell tool, so it cannot be recorded). Set `ALLAMA_UPDATE_FIXTURES=1` to rewrite the expected
//! files, then review every value.

use std::path::PathBuf;

use allama::{
    ChatProvider, ChatRequest, ChatStream, ContentBlock, Error, FinishReason, PromptCache,
    ReasoningConfig, ReasoningEffort, ReasoningOutput, ResponseFormat, SystemPrompt, ToolChoice,
    ToolDefinition,
};
use allama_conformance::{
    FixtureDir, MockHttpResponse, TestHttpServer, assert_json_fixture_eq,
    assert_response_fixture_eq, assert_stream_fixture_eq, assert_usage_contract, load_json_fixture,
    load_text_fixture,
};
use serde_json::{Value, json};

use super::{ResponsesProvider, ResponsesRequestOptions};
use crate::SystemOptions;
use allama_http::http::{HeaderValue, StatusCode};

fn fixtures() -> FixtureDir {
    FixtureDir::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"))
}

fn provider(server: &TestHttpServer) -> ResponsesProvider {
    ResponsesProvider::new_with_base_url("test-key", server.url()).unwrap()
}

/// A server returning a recorded body, then `text.json` for a follow-up
/// turn whose request body the test inspects.
async fn replay_server(name: &str) -> TestHttpServer {
    TestHttpServer::spawn([recorded(name), recorded("responses/text.json")]).await
}

fn recorded(name: &str) -> MockHttpResponse {
    MockHttpResponse::text(StatusCode::OK, load_text_fixture(&fixtures(), name))
        .header("content-type", HeaderValue::from_static("application/json"))
        .header("x-request-id", HeaderValue::from_static("req_recorded"))
}

/// Like [`assert_stream_fixture_eq`], but also (re)writes the events
/// fixture under `ALLAMA_UPDATE_FIXTURES`.
async fn assert_stream_fixtures(stream: ChatStream, events_name: &str, response_name: &str) {
    use futures_util::StreamExt;
    let events: Vec<allama::Result<allama::StreamEvent>> = stream.collect().await;
    if allama_conformance::update_fixtures() {
        let values: Vec<Value> = events
            .iter()
            .map(|item| match item {
                Ok(event) => serde_json::to_value(event).unwrap(),
                Err(err) => json!({"error": serde_json::to_value(err.as_log()).unwrap()}),
            })
            .collect();
        let path = fixtures().as_path().join(events_name);
        std::fs::write(path, serde_json::to_string_pretty(&values).unwrap() + "\n").unwrap();
    }
    let replay: ChatStream = Box::pin(futures_util::stream::iter(events));
    assert_stream_fixture_eq(replay, &fixtures(), events_name, response_name).await;
}

fn weather_tool() -> ToolDefinition {
    ToolDefinition::new(
        "get_weather",
        json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
    )
    .description("Get the current weather for a city")
    .strict(true)
}

fn weather_request(model: &str) -> ChatRequest {
    ChatRequest::new(model)
        .tools([weather_tool()])
        .user("What is the weather in Paris? Use the tool, then answer in one sentence.")
}

#[tokio::test]
async fn request_shape_matches_fixture() {
    let server = TestHttpServer::spawn([recorded("responses/text.json")]).await;
    let provider = ResponsesProvider::builder()
        .api_key("test-key")
        .base_url(server.url())
        .organization("org_1")
        .project("proj_1")
        .build()
        .unwrap();
    let request = ChatRequest::new("gpt-5-mini")
        .system(SystemPrompt::new("You are precise.").with_option(SystemOptions::developer()))
        .user("Find rust docs")
        .max_output_tokens(512)
        .parallel_tool_calls(false)
        .tools([weather_tool()])
        .tool_choice(ToolChoice::Auto)
        .response_format(ResponseFormat::json_schema(json!({"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]})).name("answer"))
        .reasoning(
            ReasoningConfig::new()
                .effort(ReasoningEffort::Medium)
                .output(ReasoningOutput::Summary),
        )
        .cache(PromptCache::new().key("session-1"))
        .with_option(
            ResponsesRequestOptions::new()
                .service_tier("flex")
                .metadata(serde_json::Map::from_iter([(
                    "trace_id".into(),
                    json!("trace-123"),
                )]))
                .safety_identifier("user-hash"),
        );
    let response = provider.chat(&request).await.unwrap();
    assert!(response.warnings.is_empty(), "{:?}", response.warnings);

    let sent = &server.recorded_requests().await[0];
    assert_eq!(sent.path, "/responses");
    assert_eq!(sent.header("authorization"), Some("Bearer test-key"));
    assert_eq!(sent.header("openai-organization"), Some("org_1"));
    assert_eq!(sent.header("openai-project"), Some("proj_1"));
    let body = sent.body_json();
    assert_eq!(body["store"], false);
    assert!(body.get("previous_response_id").is_none());
    assert_json_fixture_eq(&body, &fixtures(), "responses/request.json");
}

#[tokio::test]
async fn recorded_text_response() {
    let server = TestHttpServer::spawn([recorded("responses/text.json")]).await;
    let response = provider(&server)
        .chat(
            &ChatRequest::new("gpt-4.1-nano")
                .system("You are concise.")
                .user("Say hello in three words."),
        )
        .await
        .unwrap();
    assert_response_fixture_eq(&response, &fixtures(), "responses/text_expected.json");
    assert_eq!(response.finish_reason, Some(FinishReason::Stop));
    assert_eq!(response.finish_reason_raw.as_deref(), Some("completed"));
    assert_eq!(response.provider.as_deref(), Some("openai"));
    assert_eq!(response.request_id.as_deref(), Some("req_recorded"));
    assert_usage_contract(response.usage.as_ref().unwrap());
}

/// The recorded reasoning item replays statelessly: encrypted content and
/// item ids go back exactly as received, after a history round trip.
#[tokio::test]
async fn recorded_reasoning_tool_response_replays_statelessly() {
    let raw = load_json_fixture(&fixtures(), "responses/reasoning_tool.json");
    let server = replay_server("responses/reasoning_tool.json").await;
    let provider = provider(&server);
    let mut request = weather_request("gpt-5-nano").reasoning(
        ReasoningConfig::new()
            .effort(ReasoningEffort::Low)
            .output(ReasoningOutput::Summary),
    );
    let response = provider.chat(&request).await.unwrap();
    assert_response_fixture_eq(
        &response,
        &fixtures(),
        "responses/reasoning_tool_expected.json",
    );
    assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
    assert!(response.reasoning_text().is_some_and(|t| !t.is_empty()));
    assert_usage_contract(response.usage.as_ref().unwrap());

    request.push_message(response.to_assistant_message());
    let call = response.first_tool_call().unwrap().clone();
    request.push_tool_result(&call, "Sunny, 18 degrees Celsius.");

    // Persist and reload the conversation between turns.
    let saved = serde_json::to_string(&allama::history::History::from_request(&request)).unwrap();
    let restored: allama::history::History = serde_json::from_str(&saved).unwrap();
    let mut replay = restored.into_request("gpt-5-nano");
    replay.tools = request.tools.clone();
    replay.reasoning = request.reasoning;

    let next = provider.chat(&replay).await.unwrap();
    assert!(next.warnings.is_empty(), "{:?}", next.warnings);
    let body = server.recorded_requests().await[1].body_json();
    let input = body["input"].as_array().unwrap();
    let reasoning = &raw["output"][0];
    let function_call = &raw["output"][1];
    assert_eq!(input[1]["type"], "reasoning");
    assert_eq!(input[1]["id"], reasoning["id"]);
    assert_eq!(
        input[1]["encrypted_content"],
        reasoning["encrypted_content"]
    );
    assert_eq!(input[1]["summary"], reasoning["summary"]);
    assert_eq!(
        input[2],
        json!({
            "type": "function_call",
            "id": function_call["id"],
            "call_id": function_call["call_id"],
            "name": "get_weather",
            "arguments": function_call["arguments"],
        })
    );
    assert_eq!(
        input[3],
        json!({"type": "function_call_output", "call_id": function_call["call_id"], "output": "Sunny, 18 degrees Celsius."})
    );
}

#[tokio::test]
async fn recorded_incomplete_response() {
    let server = TestHttpServer::spawn([recorded("responses/incomplete.json")]).await;
    let response = provider(&server)
        .chat(
            &ChatRequest::new("gpt-4.1-nano")
                .user("Write a long story about a dragon.")
                .max_output_tokens(16),
        )
        .await
        .unwrap();
    assert_eq!(response.finish_reason, Some(FinishReason::Length));
    assert_eq!(
        response.finish_reason_raw.as_deref(),
        Some("max_output_tokens")
    );
    assert!(response.text().is_some_and(|t| !t.is_empty()));
    assert_eq!(response.usage.as_ref().unwrap().output_tokens, Some(16));
}

#[tokio::test]
async fn recorded_hosted_tool_items_round_trip_verbatim() {
    let raw = load_json_fixture(&fixtures(), "responses/hosted_tool.json");
    // Recorded without a request id.
    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(StatusCode::OK, &raw),
        recorded("responses/text.json"),
    ])
    .await;
    let provider = provider(&server);
    let response = provider
        .chat(&ChatRequest::new("gpt-4.1-mini").user("Search the web"))
        .await
        .unwrap();
    assert_response_fixture_eq(
        &response,
        &fixtures(),
        "responses/hosted_tool_expected.json",
    );
    let ContentBlock::ProviderToolCall(call) = &response.content[0] else {
        panic!("expected a provider tool call, got {:?}", response.content)
    };
    assert_eq!(call.name, "web_search");
    assert_eq!(call.origin.as_deref(), Some("openai"));
    let ContentBlock::ProviderToolResult(result) = &response.content[1] else {
        panic!(
            "expected a provider tool result, got {:?}",
            response.content
        )
    };
    assert_eq!(result.call_id, call.id);
    assert!(
        response.tool_calls().next().is_none(),
        "nothing for the caller to run"
    );

    let request = ChatRequest::new("gpt-4.1-mini")
        .user("Search the web")
        .message(response.to_assistant_message())
        .user("Repeat the URL only.");
    let next = provider.chat(&request).await.unwrap();
    assert!(next.warnings.is_empty(), "{:?}", next.warnings);
    let body = server.recorded_requests().await[1].body_json();
    assert_eq!(body["input"][1], raw["output"][0]);
}

/// Client-executed calls that are not function calls stay `Other` content;
/// the typed views read them, and their answers go back as top-level input
/// items next to the replayed calls (ADR 0014).
#[tokio::test]
async fn client_tool_calls_are_answered_with_typed_outputs() {
    use crate::{ClientToolCall, ShellCommandOutput};

    let raw = load_json_fixture(&fixtures(), "responses/client_tools.json");
    let server = TestHttpServer::spawn([
        MockHttpResponse::json_value(StatusCode::OK, &raw),
        recorded("responses/text.json"),
    ])
    .await;
    let provider = provider(&server);
    let mut request =
        ChatRequest::new("gpt-5.1").user("How many users signed up today? Check the logs too.");
    let response = provider.chat(&request).await.unwrap();
    assert_response_fixture_eq(
        &response,
        &fixtures(),
        "responses/client_tools_expected.json",
    );
    assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
    assert!(response.tool_calls().next().is_none(), "not function calls");

    let pending = ClientToolCall::pending(&response.content);
    assert_eq!(pending.len(), 3, "{pending:?}");
    let mut answers = Vec::new();
    for call in pending {
        answers.push(match call {
            ClientToolCall::Custom(call) => {
                assert_eq!(call.name, "sql");
                assert!(call.input.starts_with("SELECT count(*)"));
                call.output("42")
            }
            ClientToolCall::LocalShell(call) => {
                assert_eq!(call.command, ["tail", "-n", "20", "/var/log/signup.log"]);
                assert_eq!(call.working_directory.as_deref(), Some("/srv/app"));
                assert_eq!(call.timeout_ms, Some(10_000));
                call.output("signup ok\n")
            }
            ClientToolCall::Shell(call) => {
                assert_eq!(call.commands, ["wc -l /var/log/signup.log"]);
                call.output([ShellCommandOutput::exited(
                    "20 /var/log/signup.log\n",
                    "",
                    0,
                )])
            }
            other => panic!("unexpected call {other:?}"),
        });
    }
    request.push_message(response.to_assistant_message());
    request.push_message(allama::Message::user(answers));
    assert!(
        allama::history::validate(&request.messages, allama::history::ValidationMode::Sendable)
            .is_ok()
    );

    let next = provider.chat(&request).await.unwrap();
    assert!(next.warnings.is_empty(), "{:?}", next.warnings);
    let body = server.recorded_requests().await[1].body_json();
    let input = body["input"].as_array().unwrap();
    // The user message, the replayed turn (reasoning and three calls,
    // verbatim), and one output item per call.
    assert_eq!(input.len(), 8, "{input:#?}");
    for (sent, received) in input[2..5]
        .iter()
        .zip(&raw["output"].as_array().unwrap()[1..])
    {
        assert_eq!(sent, received);
    }
    assert_eq!(
        &input[5..],
        &[
            json!({"type": "custom_tool_call_output", "call_id": "call_Rk3XbQ1mZc7Vn2Pq8Ld5Ws0Y", "output": "42"}),
            json!({"type": "local_shell_call_output", "id": "call_Hn8TcW4yJf2Kd6Rm1Qa9Ze3B", "output": "signup ok\n"}),
            json!({"type": "shell_call_output", "call_id": "call_Vb5NsX7pLk1Gh4Ty2Ec8Rj6M", "max_output_length": 4096, "output": [
                {"stdout": "20 /var/log/signup.log\n", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}
            ]}),
        ]
    );
}

#[tokio::test]
async fn recorded_url_citations_become_portable() {
    let raw = load_json_fixture(&fixtures(), "responses/citations.json");
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(StatusCode::OK, &raw)]).await;
    let response = provider(&server)
        .chat(&ChatRequest::new("gpt-4.1-mini").user("Latest Rust release?"))
        .await
        .unwrap();
    assert_response_fixture_eq(&response, &fixtures(), "responses/citations_expected.json");
    assert_cites_rust_blog(&response);
}

/// Responses offsets are code points too: emoji and accented letters precede
/// the citation in this recording (gpt-4.1-mini with web search).
#[tokio::test]
async fn recorded_citation_offsets_are_code_points() {
    let raw = load_json_fixture(&fixtures(), "responses/citations_emoji.json");
    let server = TestHttpServer::spawn([MockHttpResponse::json_value(StatusCode::OK, &raw)]).await;
    let response = provider(&server)
        .chat(&ChatRequest::new("gpt-4.1-mini").user("Latest Rust release?"))
        .await
        .unwrap();
    let text = response
        .content
        .iter()
        .find_map(|block| match block {
            ContentBlock::Text(text) if !text.citations.is_empty() => Some(text),
            _ => None,
        })
        .expect("a cited text block");
    assert!(text.text.starts_with('🦀'));
    assert_eq!(
        text.cited_span(&text.citations[0]),
        Some(
            "([blog.rust-lang.org](https://blog.rust-lang.org/2026/10/01/Rust-1.99.0/?utm_source=openai))"
        )
    );
}

#[tokio::test]
async fn recorded_url_citations_stream() {
    let server = TestHttpServer::spawn([MockHttpResponse::sse(load_text_fixture(
        &fixtures(),
        "responses/citations_stream.sse",
    ))])
    .await;
    let stream = provider(&server)
        .chat_stream(&ChatRequest::new("gpt-4.1-mini").user("Latest Rust release?"))
        .await
        .unwrap();
    assert_stream_fixtures(
        stream,
        "responses/citations_stream_events.json",
        "responses/citations_stream_expected.json",
    )
    .await;
    let expected: allama::ChatResponse = serde_json::from_value(load_json_fixture(
        &fixtures(),
        "responses/citations_stream_expected.json",
    ))
    .unwrap();
    assert_cites_rust_blog(&expected);
}

/// Both recordings cite the Rust blog announcement for the parenthesized
/// link at the end of the answer.
fn assert_cites_rust_blog(response: &allama::ChatResponse) {
    let text = response
        .content
        .iter()
        .find_map(|block| match block {
            ContentBlock::Text(text) if !text.citations.is_empty() => Some(text),
            _ => None,
        })
        .expect("a cited text block");
    let citation = &text.citations[0];
    let allama::CitationSource::Url { url, title, .. } = &citation.source else {
        panic!("expected a URL source, got {:?}", citation.source)
    };
    assert!(url.starts_with("https://blog.rust-lang.org/"), "{url}");
    assert!(title.as_deref().is_some_and(|t| t.contains("Rust")));
    let cited = text.cited_span(citation).expect("span on the text");
    assert!(cited.starts_with("([blog.rust-lang.org]"), "{cited}");
    assert_eq!(response.citations().count(), 1);
}

#[tokio::test]
async fn recorded_text_stream() {
    let server = TestHttpServer::spawn([MockHttpResponse::sse(load_text_fixture(
        &fixtures(),
        "responses/text_stream.sse",
    ))
    .header("x-request-id", HeaderValue::from_static("req_stream"))])
    .await;
    let stream = provider(&server)
        .chat_stream(&ChatRequest::new("gpt-4.1-nano").user("Count from 1 to 12 in words."))
        .await
        .unwrap();
    assert_stream_fixtures(
        stream,
        "responses/text_stream_events.json",
        "responses/text_stream_expected.json",
    )
    .await;
    let body = server.recorded_requests().await[0].body_json();
    assert_eq!(body["stream"], true);
    assert_eq!(body["store"], false);
}

#[tokio::test]
async fn recorded_parallel_tools_stream() {
    let server = TestHttpServer::spawn([MockHttpResponse::sse(load_text_fixture(
        &fixtures(),
        "responses/parallel_tools_stream.sse",
    ))])
    .await;
    let stream = provider(&server)
        .chat_stream(
            &ChatRequest::new("gpt-4.1-nano")
                .tools([weather_tool()])
                .parallel_tool_calls(true)
                .user("What is the weather in Paris and in Tokyo? Call the tool once per city."),
        )
        .await
        .unwrap();
    assert_stream_fixtures(
        stream,
        "responses/parallel_tools_stream_events.json",
        "responses/parallel_tools_stream_expected.json",
    )
    .await;
}

#[tokio::test]
async fn recorded_reasoning_tool_stream() {
    let server = TestHttpServer::spawn([
        MockHttpResponse::sse(load_text_fixture(
            &fixtures(),
            "responses/reasoning_tool_stream.sse",
        )),
        recorded("responses/text.json"),
    ])
    .await;
    let request = weather_request("gpt-5-nano").reasoning(
        ReasoningConfig::new()
            .effort(ReasoningEffort::Low)
            .output(ReasoningOutput::Summary),
    );
    let response = allama::ChatStreamExt::collect_response(
        provider(&server).chat_stream(&request).await.unwrap(),
    )
    .await
    .unwrap();
    assert_response_fixture_eq(
        &response,
        &fixtures(),
        "responses/reasoning_tool_stream_expected.json",
    );
    let ContentBlock::Reasoning(reasoning) = &response.content[0] else {
        panic!("expected reasoning first, got {:?}", response.content)
    };
    assert!(!reasoning.text.is_empty());
    assert!(
        reasoning
            .provider_data
            .get_raw("openai", "encrypted_content")
            .is_some_and(|v| v.as_str().is_some_and(|s| s.starts_with("gAAAA")))
    );
    assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
    assert_usage_contract(response.usage.as_ref().unwrap());

    // The streamed turn replays like the unary one.
    let mut next = request.clone();
    next.push_message(response.to_assistant_message());
    next.push_tool_result(response.first_tool_call().unwrap(), "Sunny");
    let replayed = provider(&server).chat(&next).await.unwrap();
    assert!(replayed.warnings.is_empty(), "{:?}", replayed.warnings);
    let body = server.recorded_requests().await[1].body_json();
    assert_eq!(body["input"][1]["type"], "reasoning");
    assert_eq!(body["input"][2]["type"], "function_call");
}

#[tokio::test]
async fn recorded_errors_map_to_typed_errors() {
    let error = |status: u16, name: &str| {
        MockHttpResponse::text(
            StatusCode::from_u16(status).unwrap(),
            load_text_fixture(&fixtures(), name),
        )
        .header("content-type", HeaderValue::from_static("application/json"))
        .header("x-request-id", HeaderValue::from_static("req_err"))
    };
    let server = TestHttpServer::spawn([
        error(401, "responses/error_auth.json"),
        error(400, "responses/error_model_not_found.json"),
    ])
    .await;
    let provider = provider(&server);
    let request = ChatRequest::new("gpt-4.1-nano").user("hi");
    let auth = provider.chat(&request).await.unwrap_err();
    assert!(matches!(auth, Error::Auth { .. }), "{auth:?}");
    assert_eq!(auth.request_id(), Some("req_err"));
    assert_eq!(auth.context().unwrap().provider.as_deref(), Some("openai"));
    let missing = provider.chat(&request).await.unwrap_err();
    assert!(
        matches!(missing, Error::ModelNotFound { .. }),
        "{missing:?}"
    );
    assert_eq!(missing.status(), Some(400));
}

/// Every recorded stream assembles to the same content as decoding the
/// response object carried by its terminal event.
#[tokio::test]
async fn recorded_streams_match_their_terminal_response() {
    let dir = fixtures().as_path().join("responses");
    let mut checked = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|ext| ext != "sse") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let terminal = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|data| serde_json::from_str::<Value>(data).ok())
            .find(|event| {
                matches!(
                    event["type"].as_str(),
                    Some("response.completed" | "response.incomplete")
                )
            })
            .unwrap_or_else(|| panic!("{} has no terminal event", path.display()));
        let server = TestHttpServer::spawn([
            MockHttpResponse::json_value(StatusCode::OK, &terminal["response"]),
            MockHttpResponse::sse(text.clone()),
        ])
        .await;
        let provider = provider(&server);
        let request = ChatRequest::new("m").user("hi");
        let unary = provider.chat(&request).await.unwrap();
        let streamed =
            allama::ChatStreamExt::collect_response(provider.chat_stream(&request).await.unwrap())
                .await
                .unwrap();
        // OpenAI encrypts the same reasoning differently in
        // `output_item.done` and the terminal response, so compare
        // `encrypted_content` by presence only.
        assert_eq!(
            redact_encrypted(streamed.content),
            redact_encrypted(unary.content),
            "{}",
            path.display()
        );
        assert_eq!(
            streamed.finish_reason,
            unary.finish_reason,
            "{}",
            path.display()
        );
        checked += 1;
    }
    assert!(
        checked >= 3,
        "expected recorded stream fixtures, found {checked}"
    );
}

fn redact_encrypted(mut content: Vec<ContentBlock>) -> Vec<ContentBlock> {
    for block in &mut content {
        if let ContentBlock::Reasoning(reasoning) = block
            && let Some(Value::String(encrypted)) = reasoning
                .provider_data
                .namespace_mut("openai")
                .get_mut("encrypted_content")
        {
            *encrypted = "<encrypted>".into();
        }
    }
    content
}
