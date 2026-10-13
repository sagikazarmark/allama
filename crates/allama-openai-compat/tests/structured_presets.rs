//! What each preset says about structured output: its three
//! constraint quirks, its capability answers, and the enforcement an
//! extraction reports through it in each mode.
//!
//! Evidence for the Ollama and vLLM answers:
//!
//! - Ollama constrains a `json_schema` response format with llama.cpp's
//!   grammar, and "Ollama's Cloud currently does not support structured
//!   outputs" (<https://docs.ollama.com/capabilities/structured-outputs>);
//!   `tool_choice` is listed as not supported
//!   (<https://docs.ollama.com/api/openai-compatibility>).
//! - vLLM: "Structured outputs are supported by default in the
//!   OpenAI-Compatible Server"
//!   (<https://docs.vllm.ai/en/latest/features/structured_outputs.html>);
//!   a tool's argument schema "is pinned only when that tool sets
//!   `strict: true`", and named and `required` tool choice use structured
//!   outputs (<https://docs.vllm.ai/en/latest/features/tool_calling.html>).

// The fixture type is only deserialized into.
#![allow(dead_code)]

use allama::extract::{ExtractError, ExtractExt, ExtractionMode, SchemaEnforcement};
use allama::prelude::*;
use allama::{
    CapabilitySupport, ChatCapability, RequestField, UnsupportedPolicy, UnsupportedStage,
    WarningKind,
};
use allama_http::{ReplayResponse, ReplayTransport};
use allama_openai_compat::providers::{
    Azure, Cloudflare, DeepInfra, DeepSeek, Fireworks, Groq, Mistral, Ollama, OpenRouter, Together,
    Vllm, XAi,
};
use allama_openai_compat::{Provider, ProviderBuilder, ResponsesProvider, SchemaConstraint};
use serde_json::{Value, json};

#[derive(Debug, serde::Deserialize, allama::schemars::JsonSchema)]
#[schemars(crate = "allama::schemars")]
struct Person {
    name: String,
}

fn builder(name: &str) -> ProviderBuilder {
    match name {
        "generic" => Provider::builder().base_url("http://upstream/v1"),
        "groq" => Groq::builder("k").unwrap(),
        "together" => Together::builder("k").unwrap(),
        "fireworks" => Fireworks::builder("k").unwrap(),
        "deepinfra" => DeepInfra::builder("k").unwrap(),
        "openrouter" => OpenRouter::builder("k").unwrap(),
        "deepseek" => DeepSeek::builder("k").unwrap(),
        "xai" => XAi::builder("k").unwrap(),
        "mistral" => Mistral::builder("k").unwrap(),
        "ollama" => Ollama::builder("http://localhost:11434").unwrap(),
        "ollama-cloud" => Ollama::builder("https://ollama.com").unwrap(),
        "vllm" => Vllm::builder("http://localhost:8000/v1", None).unwrap(),
        "azure" => Azure::builder("resource", "k").unwrap(),
        "cloudflare" => Cloudflare::builder("0123456789abcdef0123456789abcdef", "k").unwrap(),
        other => panic!("unknown preset {other}"),
    }
}

fn provider(name: &str, transport: &ReplayTransport) -> Provider {
    builder(name)
        .http_transport(transport.clone())
        .build()
        .unwrap()
}

fn text() -> ReplayResponse {
    ReplayResponse::json(
        200,
        json!({
            "id": "c1", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "{\"name\":\"Ada\"}"}}]
        })
        .to_string(),
    )
}

fn tool_call() -> ReplayResponse {
    ReplayResponse::json(
        200,
        json!({
            "id": "c1", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "finish_reason": "tool_calls",
                         "message": {"role": "assistant", "content": null,
                                     "tool_calls": [{"id": "call_1", "type": "function",
                                                     "function": {"name": "submit_structured_output",
                                                                  "arguments": "{\"name\":\"Ada\"}"}}]}}]
        })
        .to_string(),
    )
}

fn body(transport: &ReplayTransport) -> Value {
    serde_json::from_str(transport.requests()[0].body_text().unwrap()).unwrap()
}

/// The three constraint quirks, and whether the dialect is verified, per
/// preset: `(response format, tools, forced choice constrains, verified)`.
#[test]
fn presets_declare_how_they_constrain_schemas() {
    use SchemaConstraint::{Always, Never, WhenStrict};
    for (name, expected) in [
        ("generic", (WhenStrict, WhenStrict, false, false)),
        ("groq", (WhenStrict, WhenStrict, false, true)),
        ("together", (Never, Never, false, false)),
        ("fireworks", (Never, Never, false, false)),
        ("deepinfra", (Never, Never, false, false)),
        ("openrouter", (WhenStrict, WhenStrict, false, false)),
        ("deepseek", (Never, Never, false, false)),
        ("xai", (WhenStrict, WhenStrict, false, true)),
        ("mistral", (WhenStrict, WhenStrict, false, false)),
        ("ollama", (Always, Never, false, false)),
        ("vllm", (Always, WhenStrict, true, false)),
        ("azure", (WhenStrict, WhenStrict, false, true)),
        ("cloudflare", (WhenStrict, WhenStrict, false, false)),
    ] {
        let provider = provider(name, &ReplayTransport::default());
        let q = provider.quirks();
        assert_eq!(
            (
                q.response_format_constraint,
                q.tool_constraint,
                q.forced_tool_choice_constrained,
                q.dialect_verified
            ),
            expected,
            "{name}"
        );
    }
    assert_eq!(Ollama::quirks().schema_dialect, None);
    assert_eq!(Vllm::quirks().schema_dialect, None);
}

/// The refusal of a forced tool choice names the endpoint that answered,
/// not the model: capability answers describe the endpoint (ADR 0015).
#[test]
fn ollama_refuses_a_forced_tool_choice_by_its_own_name() {
    let transport = ReplayTransport::default();
    let request = ChatRequest::new("llama3.2:3b")
        .user("Look up rust.")
        .tools([ToolDefinition::new("lookup", json!({"type": "object"}))])
        .tool_choice(ToolChoice::Required);
    let err = provider("ollama", &transport)
        .validate_chat_request(&request)
        .unwrap_err();
    assert_eq!(
        err.message(),
        "cannot send `tool_choice`: forced tool choice is not supported by `ollama`"
    );
    assert!(transport.requests().is_empty());
}

/// Ollama answers structured output by host, and never takes a forced tool
/// choice; vLLM answers structured output `Supported`.
#[test]
fn ollama_and_vllm_answer_structured_output_truthfully() {
    let answer = |name: &str, capability| {
        provider(name, &ReplayTransport::default()).chat_capability("m", capability)
    };
    assert_eq!(
        answer("ollama", ChatCapability::StructuredOutput),
        CapabilitySupport::Supported
    );
    assert_eq!(
        answer("ollama-cloud", ChatCapability::StructuredOutput),
        CapabilitySupport::Unsupported
    );
    for name in ["ollama", "ollama-cloud"] {
        assert_eq!(
            answer(name, ChatCapability::ForcedToolChoice),
            CapabilitySupport::Unsupported,
            "{name}"
        );
    }
    assert_eq!(
        answer("vllm", ChatCapability::StructuredOutput),
        CapabilitySupport::Supported
    );
    for host in [
        "https://ollama.com",
        "https://ollama.com:443",
        "https://OLLAMA.com/v1",
        "https://api.ollama.com",
    ] {
        let provider = Ollama::builder(host)
            .unwrap()
            .http_transport(ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(
            provider.chat_capability("m", ChatCapability::StructuredOutput),
            CapabilitySupport::Unsupported,
            "{host}"
        );
    }
    for host in ["http://notollama.com:11434", "http://gpu-box:11434"] {
        let provider = Ollama::builder(host)
            .unwrap()
            .http_transport(ReplayTransport::default())
            .build()
            .unwrap();
        assert_eq!(
            provider.chat_capability("m", ChatCapability::StructuredOutput),
            CapabilitySupport::Supported,
            "{host}"
        );
    }
}

/// What an extraction reports through each preset, in each explicit mode:
/// the enforcement, or `None` where the call fails before sending.
#[tokio::test]
async fn extraction_enforcement_per_preset() {
    use SchemaEnforcement::{Approximate, Exact, NotEnforced};
    for (name, model, native, forced) in [
        // A custom endpoint is never `Exact`: its enforcement is unverified.
        ("generic", "m", Some(Approximate), Some(Approximate)),
        // Groq documents strict mode for the endpoint
        // (https://console.groq.com/docs/structured-outputs). Which models
        // take it is model data (ADR 0015), so every model reports it.
        ("groq", "openai/gpt-oss-120b", Some(Exact), Some(Exact)),
        ("groq", "llama-3.3-70b-versatile", Some(Exact), Some(Exact)),
        ("xai", "m", Some(Exact), Some(Exact)),
        ("azure", "m", Some(Exact), Some(Exact)),
        ("openrouter", "m", Some(Approximate), Some(Approximate)),
        ("mistral", "m", Some(Approximate), Some(Approximate)),
        ("cloudflare", "m", Some(Approximate), Some(Approximate)),
        ("together", "m", Some(NotEnforced), Some(NotEnforced)),
        ("fireworks", "m", Some(NotEnforced), Some(NotEnforced)),
        ("deepinfra", "m", Some(NotEnforced), Some(NotEnforced)),
        // JSON mode only: a schema response format is refused.
        ("deepseek", "m", None, Some(NotEnforced)),
        // `format` is grammar-constrained; a forced choice is not taken.
        ("ollama", "m", Some(Approximate), None),
        // Ollama Cloud does not constrain the response format.
        ("ollama-cloud", "m", None, None),
        ("vllm", "m", Some(Approximate), Some(Approximate)),
    ] {
        for (mode, expected, answer) in [
            (ExtractionMode::Native, native, text()),
            (ExtractionMode::ForcedTool, forced, tool_call()),
        ] {
            let request = ChatRequest::new(model).user("Ada").with_option(mode);
            let transport = ReplayTransport::new([answer]);
            let result = provider(name, &transport).extract::<Person>(&request).await;
            match expected {
                Some(enforcement) => {
                    let extracted = result.unwrap_or_else(|err| panic!("{name} {mode:?}: {err}"));
                    assert_eq!(
                        extracted.metadata.enforcement, enforcement,
                        "{name} {mode:?}: {:?}",
                        extracted.metadata.degradations
                    );
                }
                None => {
                    let err = result.err().unwrap_or_else(|| panic!("{name} {mode:?}"));
                    assert!(matches!(err, Error::Unsupported { .. }), "{name}: {err}");
                    assert!(transport.requests().is_empty(), "{name} {mode:?}");
                }
            }
        }
    }
}

/// `Auto` picks native output on Ollama and vLLM, which answer structured
/// output `Supported`; on Ollama Cloud nothing is left to pick.
#[tokio::test]
async fn auto_picks_native_output_where_it_is_constrained() {
    for name in ["ollama", "vllm"] {
        let transport = ReplayTransport::new([text()]);
        let request = ChatRequest::new("m").user("Ada");
        let extracted = provider(name, &transport)
            .extract::<Person>(&request)
            .await
            .unwrap();
        assert_eq!(extracted.metadata.mode, ExtractionMode::Native, "{name}");
        assert_eq!(
            extracted.metadata.enforcement,
            SchemaEnforcement::Approximate,
            "{name}"
        );
    }

    let transport = ReplayTransport::new([]);
    let err = provider("ollama-cloud", &transport)
        .extract::<Person>(&ChatRequest::new("m").user("Ada"))
        .await
        .unwrap_err();
    let Error::Extract { failure, .. } = err else {
        panic!("expected an extraction failure: {err}");
    };
    assert!(matches!(failure.error(), ExtractError::Unsupported { .. }));
    assert!(transport.requests().is_empty());
}

/// On the wire: an endpoint that always constrains the response format
/// gets no `strict` flag, with an `Approximated` warning; vLLM gets the
/// strict flag on a strict tool, Ollama a strict tool without it and an
/// `Unsupported` warning.
#[tokio::test]
async fn constraint_quirks_shape_the_wire_and_the_warnings() {
    let schema =
        json!({"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]});
    let native = ChatRequest::new("m")
        .user("Ada")
        .response_format(ResponseFormat::json_schema(schema.clone()).name("person"));
    let tools = ChatRequest::new("m")
        .user("Ada")
        .tools([ToolDefinition::new("save", schema.clone()).strict(true)]);

    for name in ["ollama", "vllm"] {
        let transport = ReplayTransport::new([text()]);
        let response = provider(name, &transport).chat(&native).await.unwrap();
        let sent = body(&transport);
        assert_eq!(sent["response_format"]["json_schema"]["schema"], schema);
        assert!(
            sent["response_format"]["json_schema"]
                .get("strict")
                .is_none(),
            "{name}"
        );
        let kinds: Vec<_> = response.warnings.iter().map(|w| w.kind).collect();
        assert_eq!(kinds, [WarningKind::Approximated], "{name}");

        // `strict: false` asks for nothing, and nothing is reported.
        let mut relaxed = native.clone();
        relaxed.response_format = Some(ResponseFormat::json_schema(schema.clone()).non_strict());
        let transport = ReplayTransport::new([text()]);
        let response = provider(name, &transport).chat(&relaxed).await.unwrap();
        assert!(response.warnings.is_empty(), "{name}");
        assert!(
            body(&transport)["response_format"]["json_schema"]
                .get("strict")
                .is_none()
        );
    }

    let transport = ReplayTransport::new([text()]);
    let response = provider("vllm", &transport).chat(&tools).await.unwrap();
    assert_eq!(body(&transport)["tools"][0]["function"]["strict"], true);
    let kinds: Vec<_> = response.warnings.iter().map(|w| w.kind).collect();
    assert_eq!(kinds, [WarningKind::Approximated]);

    let transport = ReplayTransport::new([text()]);
    let response = provider("ollama", &transport).chat(&tools).await.unwrap();
    assert!(
        body(&transport)["tools"][0]["function"]
            .get("strict")
            .is_none()
    );
    let kinds: Vec<_> = response.warnings.iter().map(|w| w.kind).collect();
    assert_eq!(kinds, [WarningKind::Unsupported]);
    let strict = tools.unsupported_policy(UnsupportedPolicy::Error);
    let transport = ReplayTransport::new([]);
    let err = provider("ollama", &transport)
        .chat(&strict)
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported { .. }), "{err}");
    assert!(transport.requests().is_empty());
}

/// What a hosted endpoint serves depends on the model it routes to, so no
/// hosted preset claims tool calls or structured output for every model
/// (ADR 0015): both are `Unknown`, sent as asked, and the endpoint decides.
/// Cloudflare's json-mode and function-calling pages list the models that
/// take them
/// (<https://developers.cloudflare.com/workers-ai/features/json-mode/>,
/// <https://developers.cloudflare.com/workers-ai/features/function-calling/>).
/// DeepSeek takes no JSON Schema for any model, so it answers structured
/// output `Unsupported` (below).
#[test]
fn hosted_presets_leave_model_dependent_features_unknown() {
    for name in [
        "groq",
        "together",
        "fireworks",
        "deepinfra",
        "openrouter",
        "deepseek",
        "xai",
        "mistral",
        "azure",
        "cloudflare",
    ] {
        let provider = provider(name, &ReplayTransport::default());
        assert_eq!(
            provider.chat_capability("m", ChatCapability::ToolCalls),
            CapabilitySupport::Unknown,
            "{name}"
        );
        if name != "deepseek" {
            assert_eq!(
                provider.chat_capability("m", ChatCapability::StructuredOutput),
                CapabilitySupport::Unknown,
                "{name}"
            );
        }
        assert_eq!(
            provider.chat_capability("m", ChatCapability::Streaming),
            CapabilitySupport::Supported,
            "{name}"
        );
    }
}

/// Structured output means JSON Schema response
/// formats only. DeepSeek is JSON mode only, so it answers `Unsupported`
/// for every model: a schema is refused before sending, and JSON mode,
/// which no capability covers, is sent.
#[tokio::test]
async fn deepseek_answers_structured_output_for_json_schema_only() {
    let transport = ReplayTransport::new([text()]);
    let deepseek = provider("deepseek", &transport);
    for model in ["deepseek-chat", "m"] {
        assert_eq!(
            deepseek.chat_capability(model, ChatCapability::StructuredOutput),
            CapabilitySupport::Unsupported,
            "{model}"
        );
    }

    let schema = ChatRequest::new("deepseek-chat")
        .user("Reply in JSON.")
        .response_format(ResponseFormat::json_schema(json!({"type": "object"})));
    let err = deepseek.chat(&schema).await.unwrap_err();
    assert_eq!(err.unsupported_field(), Some(RequestField::ResponseFormat));
    assert_eq!(
        err.unsupported_stage(),
        Some(UnsupportedStage::BeforeSending)
    );
    assert!(transport.requests().is_empty());

    let json_mode = ChatRequest::new("deepseek-chat")
        .user("Reply in JSON.")
        .response_format(ResponseFormat::Json);
    let validation = deepseek.validate_chat_request(&json_mode).unwrap();
    assert!(validation.relied_on_unknown().is_empty());
    deepseek.chat(&json_mode).await.unwrap();
    let sent: Value = serde_json::from_str(transport.requests()[0].body_text().unwrap()).unwrap();
    assert_eq!(sent["response_format"], json!({"type": "json_object"}));
}

/// With structured output `Unknown` on Cloudflare's `/ai/v1`, `Auto`
/// guesses native output and falls back to forced-tool mode when the
/// endpoint refuses the response format. A preset-wide `Supported`
/// claim used to turn that fallback off.
#[tokio::test]
async fn cloudflare_auto_extraction_falls_back_on_a_refused_response_format() {
    let refusal = ReplayResponse::json(
        400,
        json!({"error": {
            "message": "Unsupported parameter: 'response_format' is not supported with this model.",
            "type": "invalid_request_error",
            "param": "response_format",
            "code": "unsupported_parameter"
        }})
        .to_string(),
    );
    let transport = ReplayTransport::new([refusal, tool_call()]);
    let extracted = provider("cloudflare", &transport)
        .extract::<Person>(&ChatRequest::new("m").user("Ada"))
        .await
        .unwrap();
    assert_eq!(extracted.value.name, "Ada");
    assert_eq!(extracted.metadata.mode, ExtractionMode::ForcedTool);
    assert!(extracted.metadata.auto_fallback.is_some());
    assert_eq!(transport.requests().len(), 2);
}

/// The Responses API refuses a response format with `code: null`; the
/// message prefix stands in for the code, so `Auto` falls back from native
/// output to forced-tool mode as it does on Chat Completions.
#[tokio::test]
async fn responses_auto_extraction_falls_back_on_a_codeless_refusal() {
    let refusal = ReplayResponse::json(
        400,
        json!({"error": {
            "message": "Unsupported parameter: 'text.format' is not supported with this model.",
            "type": "invalid_request_error",
            "param": "text.format",
            "code": null
        }})
        .to_string(),
    );
    let call = json!({"type": "function_call", "id": "fc_1", "call_id": "call_1",
        "name": "submit_structured_output", "arguments": "{\"name\":\"Ada\"}", "status": "completed"});
    let tool_call = ReplayResponse::json(
        200,
        json!({"id": "resp_1", "object": "response", "status": "completed", "model": "m",
               "output": [call]})
        .to_string(),
    );
    let transport = ReplayTransport::new([refusal, tool_call]);
    let provider = ResponsesProvider::builder()
        .base_url("http://upstream/v1")
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let extracted = provider
        .extract::<Person>(&ChatRequest::new("m").user("Ada"))
        .await
        .unwrap();
    assert_eq!(extracted.value.name, "Ada");
    assert_eq!(extracted.metadata.mode, ExtractionMode::ForcedTool);
    assert!(extracted.metadata.auto_fallback.is_some());
    let requests = transport.requests();
    assert_eq!(requests.len(), 2);
    let first: Value = serde_json::from_slice(requests[0].body.as_ref().unwrap()).unwrap();
    assert_eq!(first["text"]["format"]["type"], "json_schema");
}
