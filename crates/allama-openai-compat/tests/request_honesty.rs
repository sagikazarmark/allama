//! Every contract for the generic provider and every hosted preset, through
//! the aggregates (`assert_chat_adapter`, `assert_embedding_adapter`), and
//! the request honesty checks that need a configuration of their own
//! (declared capabilities, overridden answers, quirks).

use allama::prelude::*;
use allama::{
    CapabilitySupport, ChatCapability, EmbeddingCapability, EmbeddingInputType, EmbeddingResponse,
    RequestField, Subject, UnsupportedPolicy, ValidationCoverage,
};
use allama_conformance::capabilities::{
    assert_stream_usage_matches_capability, assert_unsupported_chat_capabilities_refused,
};
use allama_conformance::probe::{
    ProbeOutcome, VectorOrder, assert_app_data_never_sent, assert_batch_limit_enforced,
    assert_request_fields_accounted, assert_strict_not_honoured_warns,
    assert_unreplayable_reasoning_dropped,
};
use allama_conformance::stream_end::StreamEnds;
use allama_conformance::{
    AdapterUnderTest, Contract, EmbeddingAdapterUnderTest, ResponseCapture, TransportOutcome,
    assert_chat_adapter, assert_embedding_adapter, assert_embedding_validation_agrees,
    assert_validation_agrees, drift_requests,
};
use allama_http::http::{HeaderValue, StatusCode};
use allama_http::{ReplayResponse, ReplayTransport};
use allama_openai_compat::providers::{
    Azure, Cloudflare, DeepInfra, DeepSeek, Fireworks, Groq, Mistral, Ollama, OpenRouter, Together,
    Vllm, XAi,
};
use allama_openai_compat::{
    CompatQuirks, Provider, ProviderBuilder, ReasoningReplay, ResponsesProvider,
    ResponsesProviderBuilder, ResponsesQuirks, SchemaConstraint, ToolCallIdFormat,
};
use futures_util::StreamExt;
use serde_json::{Value, json};
use support::envelope_with_upstream;

mod support;

fn chat_ok() -> ReplayResponse {
    ReplayResponse::json(
        StatusCode::OK,
        json!({
            "id": "c1", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })
        .to_string(),
    )
}

fn chat_sse() -> ReplayResponse {
    let chunk = json!({
        "id": "c1", "object": "chat.completion.chunk", "model": "m",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}]
    });
    ReplayResponse::sse(format!("data: {chunk}\n\ndata: [DONE]\n\n"))
}

fn responses_ok() -> ReplayResponse {
    ReplayResponse::json(
        StatusCode::OK,
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

fn responses_sse() -> ReplayResponse {
    let events = [
        json!({"type": "response.created",
               "response": {"id": "resp_1", "model": "m", "status": "in_progress", "output": []}}),
        json!({"type": "response.completed",
               "response": {"id": "resp_1", "model": "m", "status": "completed", "output": [],
                            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}}),
    ];
    ReplayResponse::sse(
        events
            .iter()
            .map(|e| format!("event: {}\ndata: {e}\n\n", e["type"].as_str().unwrap()))
            .collect::<String>(),
    )
}

fn embeddings_ok(count: usize) -> ReplayResponse {
    let data: Vec<Value> = (0..count)
        .map(|i| json!({"index": i, "embedding": [0.1, 0.2]}))
        .collect();
    ReplayResponse::json(
        StatusCode::OK,
        json!({"data": data, "model": "m"}).to_string(),
    )
}

fn bodies(transport: &ReplayTransport) -> Vec<Value> {
    transport
        .requests()
        .iter()
        .filter_map(|r| Some(r.body.as_ref()).filter(|body| !body.is_empty()))
        .map(|body| serde_json::from_slice(body).expect("JSON request body"))
        .collect()
}

fn generic() -> ProviderBuilder {
    Provider::builder()
        .base_url("http://upstream/v1")
        .http_transport(ReplayTransport::default())
}

/// The generic dialect and every preset.
const DIALECTS: &[&str] = &[
    "generic",
    "groq",
    "together",
    "fireworks",
    "deepinfra",
    "openrouter",
    "deepseek",
    "xai",
    "mistral",
    "ollama",
    "vllm",
    "azure",
    "cloudflare",
];

/// The builder for `name`, over an empty replay transport (tests that send
/// replace it), so nothing here needs `reqwest`.
fn dialect(name: &str) -> ProviderBuilder {
    dialect_builder(name).http_transport(ReplayTransport::default())
}

fn dialect_builder(name: &str) -> ProviderBuilder {
    match name {
        "generic" => generic(),
        "groq" => Groq::builder("k").unwrap(),
        "together" => Together::builder("k").unwrap(),
        "fireworks" => Fireworks::builder("k").unwrap(),
        "deepinfra" => DeepInfra::builder("k").unwrap(),
        "openrouter" => OpenRouter::builder("k").unwrap(),
        "deepseek" => DeepSeek::builder("k").unwrap(),
        "xai" => XAi::builder("k").unwrap(),
        "mistral" => Mistral::builder("k").unwrap(),
        "ollama" => Ollama::builder("http://localhost:11434").unwrap(),
        "vllm" => Vllm::builder("http://localhost:8000/v1", None).unwrap(),
        "azure" => Azure::builder("resource", "k").unwrap(),
        "cloudflare" => Cloudflare::builder("0123456789abcdef0123456789abcdef", "k").unwrap(),
        other => panic!("unknown dialect {other}"),
    }
}

async fn chat_probe(builder: ProviderBuilder, request: ChatRequest) -> ProbeOutcome<ChatResponse> {
    let transport = ReplayTransport::new([chat_ok()]);
    let provider = builder.http_transport(transport.clone()).build().unwrap();
    let result = provider.chat(&request).await;
    ProbeOutcome::new(result, bodies(&transport))
}

async fn chat_stream_probe(
    builder: ProviderBuilder,
    request: ChatRequest,
) -> ProbeOutcome<ChatResponse> {
    let transport = ReplayTransport::new([chat_sse()]);
    let provider = builder.http_transport(transport.clone()).build().unwrap();
    let result = match provider.chat_stream(&request).await {
        Ok(stream) => stream.collect_response().await,
        Err(err) => Err(err),
    };
    ProbeOutcome::new(result, bodies(&transport))
}

/// A streamed extraction on Together that fails mid-way, after the usage
/// chunk: the error's extraction record keeps the billed usage and the
/// strict degradation, and counts no pass.
#[tokio::test]
async fn streamed_extraction_failing_mid_way_keeps_usage_and_degradations() {
    use allama::chat::extract::{
        ChatExtractExt, ChatExtractionMetadata, ChatExtractionMode, SchemaEnforcement,
    };

    #[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
    #[allow(dead_code)]
    struct Person {
        name: String,
    }

    let text = json!({
        "id": "c1", "object": "chat.completion.chunk", "model": "m",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "{\"name\":"}}]
    });
    let usage = json!({
        "id": "c1", "object": "chat.completion.chunk", "model": "m", "choices": [],
        "usage": {"prompt_tokens": 900, "completion_tokens": 3, "total_tokens": 903}
    });
    let error = json!({"error": {"message": "upstream overloaded", "type": "server_error"}});
    let transport = ReplayTransport::new([ReplayResponse::sse(format!(
        "data: {text}\n\ndata: {usage}\n\ndata: {error}\n\n"
    ))]);
    let provider = dialect("together")
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let request = ChatRequest::new("m")
        .user("Ada")
        .with_option(ChatExtractionMode::Native);
    let mut stream = provider.extract_stream::<Person>(&request).await.unwrap();
    while stream.next_event().await.is_some() {}
    let err = stream.finish().unwrap_err();
    assert!(!matches!(err, Error::Extract { .. }), "{err:?}");

    let record = ChatExtractionMetadata::from_error(&err).unwrap();
    assert_eq!(record.passes, 0);
    assert_eq!(
        record.usage.as_ref().and_then(|u| u.input_tokens),
        Some(900)
    );
    assert_eq!(record.degradations.len(), 1, "{:?}", record.degradations);
    assert_eq!(
        record.degradations[0].subject,
        Subject::Field {
            field: RequestField::ResponseFormat
        }
    );
    assert_eq!(record.enforcement, SchemaEnforcement::NotEnforced);
}

/// Extraction on a preset without strict support runs under `Warn` and says
/// the schema was not enforced; under `Error` it fails before sending.
#[tokio::test]
async fn extraction_on_presets_without_strict_is_not_enforced() {
    use allama::chat::extract::{ChatExtractExt, ChatExtractionMode, SchemaEnforcement};

    #[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
    struct Person {
        name: String,
    }

    let tool_call = || {
        ReplayResponse::json(
            StatusCode::OK,
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
    };
    let text = || {
        ReplayResponse::json(
            StatusCode::OK,
            json!({
                "id": "c1", "object": "chat.completion", "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "{\"name\":\"Ada\"}"}}]
            })
            .to_string(),
        )
    };
    for name in ["together", "fireworks", "deepinfra"] {
        for (mode, field, answer) in [
            (
                ChatExtractionMode::Native,
                RequestField::ResponseFormat,
                text(),
            ),
            (
                ChatExtractionMode::ForcedTool,
                RequestField::Tools,
                tool_call(),
            ),
        ] {
            let request = ChatRequest::new("m").user("Ada").with_option(mode);
            let transport = ReplayTransport::new([answer]);
            let provider = dialect(name)
                .http_transport(transport.clone())
                .build()
                .unwrap();
            if name == "ollama" && mode == ChatExtractionMode::ForcedTool {
                // Ollama ignores `tool_choice`, so forcing the tool is
                // refused before sending.
                let err = provider.extract::<Person>(&request).await.unwrap_err();
                assert_eq!(err.unsupported_field(), Some(RequestField::ToolChoice));
                assert!(transport.requests().is_empty());
                continue;
            }
            let extracted = provider.extract::<Person>(&request).await.unwrap();
            assert_eq!(extracted.value.name, "Ada");
            assert_eq!(
                extracted.metadata.enforcement,
                SchemaEnforcement::NotEnforced,
                "{name} {mode:?}"
            );
            assert!(
                extracted
                    .metadata
                    .degradations
                    .iter()
                    .all(|w| w.subject == Subject::Field { field }),
                "{name} {mode:?}: {:?}",
                extracted.metadata.degradations
            );
            assert_eq!(extracted.metadata.degradations.len(), 1);

            let transport = ReplayTransport::new([]);
            let provider = dialect(name)
                .http_transport(transport.clone())
                .build()
                .unwrap();
            let strict = request.unsupported_policy(UnsupportedPolicy::Error);
            let err = provider.extract::<Person>(&strict).await.unwrap_err();
            assert!(matches!(err, Error::Unsupported { .. }), "{name}: {err}");
            assert!(transport.requests().is_empty());
        }
    }
}

fn responses_without_strict(transport: &ReplayTransport) -> ResponsesProvider {
    let mut quirks = ResponsesQuirks::new("openai-compat:demo");
    quirks.supports_strict = false;
    ResponsesProvider::builder()
        .base_url("http://upstream/v1")
        .quirks(quirks)
        .http_transport(transport.clone())
        .build()
        .unwrap()
}

#[tokio::test]
async fn responses_without_strict_never_drops_it_silently() {
    assert_strict_not_honoured_warns(
        "m",
        async |request| {
            let transport = ReplayTransport::new([responses_ok()]);
            let result = responses_without_strict(&transport).chat(&request).await;
            ProbeOutcome::new(result, bodies(&transport))
        },
        async |request| {
            let transport = ReplayTransport::new([responses_sse()]);
            let result = match responses_without_strict(&transport)
                .chat_stream(&request)
                .await
            {
                Ok(stream) => stream.collect_response().await,
                Err(err) => Err(err),
            };
            ProbeOutcome::new(result, bodies(&transport))
        },
    )
    .await;
}

fn responses() -> ResponsesProviderBuilder {
    ResponsesProvider::builder()
        .base_url("http://upstream/v1")
        .http_transport(ReplayTransport::default())
}

async fn responses_probe(
    builder: ResponsesProviderBuilder,
    request: ChatRequest,
) -> ProbeOutcome<ChatResponse> {
    let transport = ReplayTransport::new([responses_ok()]);
    let provider = builder.http_transport(transport.clone()).build().unwrap();
    let result = provider.chat(&request).await;
    ProbeOutcome::new(result, bodies(&transport))
}

async fn responses_stream_probe(
    builder: ResponsesProviderBuilder,
    request: ChatRequest,
) -> ProbeOutcome<ChatResponse> {
    let transport = ReplayTransport::new([responses_sse()]);
    let provider = builder.http_transport(transport.clone()).build().unwrap();
    let result = match provider.chat_stream(&request).await {
        Ok(stream) => stream.collect_response().await,
        Err(err) => Err(err),
    };
    ProbeOutcome::new(result, bodies(&transport))
}

/// `StreamUsage` follows the `stream_usage` quirk. Without
/// `stream_options.include_usage` a Chat Completions stream has no usage
/// chunk
/// (<https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream_options>),
/// so a dialect that does not send it answers `Unsupported`. One that sends
/// it answers `Unknown`: whether the server honors it is the server's
/// claim, which a declaration makes.
#[tokio::test]
async fn stream_usage_follows_the_quirk() {
    let answer = |builder: ProviderBuilder| {
        builder
            .build()
            .unwrap()
            .chat_capability("m", ChatCapability::StreamUsage)
    };
    let without_usage = || {
        let mut quirks = CompatQuirks::new("openai-compat:vendor");
        quirks.stream_usage = false;
        generic().quirks(quirks)
    };

    for name in DIALECTS {
        assert_eq!(answer(dialect(name)), CapabilitySupport::Unknown, "{name}");
    }
    assert_eq!(answer(without_usage()), CapabilitySupport::Unsupported);
    assert_stream_usage_matches_capability(&without_usage().build().unwrap(), "m", |request| {
        chat_stream_probe(without_usage(), request)
    })
    .await;
    let sent = chat_stream_probe(without_usage(), ChatRequest::new("m").user("q")).await;
    assert_eq!(sent.sent[0].get("stream_options"), None, "{:?}", sent.sent);

    // A declaration wins over the quirk.
    let declared = |support| without_usage().chat_capability(ChatCapability::StreamUsage, support);
    assert_eq!(
        answer(declared(CapabilitySupport::Supported)),
        CapabilitySupport::Supported
    );
    let usage_chunk = json!({
        "id": "c1", "object": "chat.completion.chunk", "model": "m", "choices": [],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    });
    let with_usage = || {
        let chunk = json!({
            "id": "c1", "object": "chat.completion.chunk", "model": "m",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}]
        });
        generic()
            .chat_capability(ChatCapability::StreamUsage, CapabilitySupport::Supported)
            .http_transport(ReplayTransport::new([ReplayResponse::sse(format!(
                "data: {chunk}\n\ndata: {usage_chunk}\n\ndata: [DONE]\n\n"
            ))]))
    };
    assert_stream_usage_matches_capability(&with_usage().build().unwrap(), "m", |request| {
        let provider = with_usage().build().unwrap();
        async move {
            let result = match provider.chat_stream(&request).await {
                Ok(stream) => stream.collect_response().await,
                Err(err) => Err(err),
            };
            ProbeOutcome::new(result, Vec::new())
        }
    })
    .await;
}

/// A provider that declares every capability `Unsupported` must refuse
/// every request feature, whatever its dialect sends.
#[tokio::test]
async fn declared_unsupported_capabilities_are_refused() {
    let all_unsupported = || {
        ChatCapability::ALL
            .iter()
            .fold(generic(), |builder, &capability| {
                builder.chat_capability(capability, CapabilitySupport::Unsupported)
            })
    };
    let provider = all_unsupported().build().unwrap();
    assert_unsupported_chat_capabilities_refused(
        &provider,
        "m",
        &[],
        |request| chat_probe(all_unsupported(), request),
        |request| chat_stream_probe(all_unsupported(), request),
    )
    .await;
}

#[tokio::test]
async fn responses_declared_unsupported_capabilities_are_refused() {
    let all_unsupported = || {
        ChatCapability::ALL
            .iter()
            .fold(responses(), |builder, &capability| {
                builder.chat_capability(capability, CapabilitySupport::Unsupported)
            })
    };
    let provider = all_unsupported().build().unwrap();
    assert_unsupported_chat_capabilities_refused(
        &provider,
        "m",
        &[],
        |request| responses_probe(all_unsupported(), request),
        |request| responses_stream_probe(all_unsupported(), request),
    )
    .await;
}

fn embedding_provider() -> ProviderBuilder {
    generic()
        .max_embedding_inputs(3)
        .embedding_capability(
            EmbeddingCapability::OutputDimensions,
            CapabilitySupport::Unsupported,
        )
        .embedding_capability(
            EmbeddingCapability::BatchInput,
            CapabilitySupport::Unsupported,
        )
}

async fn embed_probe(request: EmbeddingRequest) -> ProbeOutcome<EmbeddingResponse> {
    let transport = ReplayTransport::new([embeddings_ok(request.inputs.len())]);
    let provider = embedding_provider()
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let result = provider.embed(&request).await;
    ProbeOutcome::new(result, bodies(&transport))
}

#[tokio::test]
async fn declared_embedding_batch_limit_is_enforced() {
    let provider = generic().max_embedding_inputs(3).build().unwrap();
    assert_batch_limit_enforced(&provider, "m", |request| async move {
        let transport = ReplayTransport::new([embeddings_ok(request.inputs.len())]);
        let provider = generic()
            .max_embedding_inputs(3)
            .http_transport(transport.clone())
            .build()
            .unwrap();
        let result = provider.embed(&request).await;
        ProbeOutcome::new(result, bodies(&transport))
    })
    .await;
}

async fn dialect_embed_probe(
    name: &'static str,
    request: EmbeddingRequest,
) -> ProbeOutcome<EmbeddingResponse> {
    let transport = ReplayTransport::new([embeddings_ok(request.inputs.len())]);
    let provider = dialect(name)
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let result = provider.embed(&request).await;
    ProbeOutcome::new(result, bodies(&transport))
}

/// A model known to take the role as a text prefix is named with its
/// prefix; the text is never rewritten. nomic-embed-text documents
/// `search_query: ` / `search_document: `
/// (<https://huggingface.co/nomic-ai/nomic-embed-text-v1.5>).
#[tokio::test]
async fn input_type_warning_names_a_known_role_prefix() {
    let request = EmbeddingRequest::new("nomic-embed-text")
        .input("allama-input-type-probe")
        .input_type(EmbeddingInputType::Document);
    let outcome = dialect_embed_probe("ollama", request).await;
    let response = outcome.result.unwrap();
    assert_eq!(response.warnings.len(), 1, "{:?}", response.warnings);
    assert_eq!(
        response.warnings[0].message,
        "`nomic-embed-text` takes the role as a text prefix, not a field; the `document` input \
         type was not sent; start each document with `search_document: `"
    );
    assert_eq!(outcome.sent[0]["input"], json!(["allama-input-type-probe"]));
}

/// Both compat providers run `allama::adapter::chat::preflight` against their
/// declared capabilities in `chat` and `chat_stream`, and agree with
/// `validate_chat_request`.
#[tokio::test]
async fn preflight_applies_declared_capabilities() {
    let unsupported = [ChatCapability::ToolCalls, ChatCapability::ReasoningConfig];

    check_preflight(
        |t| {
            unsupported
                .into_iter()
                .fold(generic(), |b, c| {
                    b.chat_capability(c, CapabilitySupport::Unsupported)
                })
                .http_transport(t.clone())
                .build()
                .unwrap()
        },
        [chat_ok, chat_sse],
        "/reasoning_effort",
    )
    .await;
    check_preflight(
        |t| {
            unsupported
                .into_iter()
                .fold(
                    ResponsesProvider::builder().base_url("http://upstream/v1"),
                    |b, c| b.chat_capability(c, CapabilitySupport::Unsupported),
                )
                .http_transport(t.clone())
                .build()
                .unwrap()
        },
        [responses_ok, responses_sse],
        "/reasoning",
    )
    .await;
}

/// The shared hint and strict-policy contracts hold for both compat
/// providers, on `chat` and on `chat_stream`.
#[tokio::test]
async fn unsupported_hints_warn_and_strict_policy_rejects_on_both_paths() {
    use allama_conformance::contract::{
        assert_strict_policy_rejects, assert_strict_policy_rejects_stream,
        assert_unsupported_hint_warns, assert_unsupported_hint_warns_stream,
    };

    let request = ChatRequest::new("m")
        .user("hi")
        .reasoning(ReasoningEffort::High);

    let transport = ReplayTransport::new([chat_ok(), chat_sse()]);
    let provider = generic()
        .chat_capability(
            ChatCapability::ReasoningConfig,
            CapabilitySupport::Unsupported,
        )
        .http_transport(transport.clone())
        .build()
        .unwrap();
    assert_unsupported_hint_warns(&provider, &request).await;
    assert_strict_policy_rejects(&provider, &request).await;
    assert_unsupported_hint_warns_stream(&provider, &request).await;
    assert_strict_policy_rejects_stream(&provider, &request).await;
    assert_eq!(transport.requests().len(), 2);

    let transport = ReplayTransport::new([responses_ok(), responses_sse()]);
    let provider = ResponsesProvider::builder()
        .base_url("http://upstream/v1")
        .chat_capability(
            ChatCapability::ReasoningConfig,
            CapabilitySupport::Unsupported,
        )
        .http_transport(transport.clone())
        .build()
        .unwrap();
    assert_unsupported_hint_warns(&provider, &request).await;
    assert_strict_policy_rejects(&provider, &request).await;
    assert_unsupported_hint_warns_stream(&provider, &request).await;
    assert_strict_policy_rejects_stream(&provider, &request).await;
    assert_eq!(transport.requests().len(), 2);
}

async fn check_preflight<P: ChatProvider<Stream = allama::ChatStream>>(
    provider: impl Fn(&ReplayTransport) -> P,
    [ok, sse]: [fn() -> ReplayResponse; 2],
    reasoning_pointer: &str,
) {
    use allama::{RequestField, Subject, UnsupportedPolicy, WarningKind};

    let is_reasoning_warning = |w: &allama::Warning| {
        w.kind == WarningKind::Unsupported
            && w.subject
                == Subject::Field {
                    field: RequestField::Reasoning,
                }
    };

    // An unsupported hint under `Warn`: validation passes, `chat` serves the
    // request with a warning, and the hint never reaches the wire.
    let hint = ChatRequest::new("m")
        .user("hi")
        .reasoning(ReasoningEffort::High);
    let transport = ReplayTransport::new([ok()]);
    let lenient = provider(&transport);
    let validation = lenient.validate_chat_request(&hint).unwrap();
    let response = lenient.chat(&hint).await.unwrap();
    assert_eq!(validation.warnings(), response.warnings);
    assert!(
        response.warnings.iter().any(is_reasoning_warning),
        "{:?}",
        response.warnings
    );
    let body = &bodies(&transport)[0];
    assert!(body.pointer(reasoning_pointer).is_none(), "{body}");

    // The stream reports the warning too.
    let transport = ReplayTransport::new([sse()]);
    let events: Vec<_> = provider(&transport)
        .chat_stream(&hint)
        .await
        .unwrap()
        .collect()
        .await;
    assert!(
        events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::Warning { warning, .. }) if is_reasoning_warning(warning)
        )),
        "{events:?}"
    );

    // The same hint under `Error`, and a semantic feature under any policy:
    // validation, `chat`, and `chat_stream` all refuse before sending.
    let strict = hint.unsupported_policy(UnsupportedPolicy::Error);
    let tools = ChatRequest::new("m")
        .user("hi")
        .tools([ToolDefinition::new("t", json!({"type": "object"}))]);
    for request in [strict, tools] {
        let transport = ReplayTransport::new([]);
        let provider = provider(&transport);
        let err = provider.validate_chat_request(&request).unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }), "{err}");
        let err = provider.chat(&request).await.unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }), "{err}");
        assert!(matches!(
            provider.chat_stream(&request).await,
            Err(Error::Unsupported { .. })
        ));
        assert!(transport.requests().is_empty());
    }
}

/// A provider that declares tool calls `Unsupported` passes the request
/// audits. The application data probe leaves the tool turn out for it and
/// still checks the system prompt and every other message.
/// Ollama ignores `tool_choice`: `Disabled` is sent as no tools, which is
/// exactly "the model must not call tools", so it records no warning, even
/// under `UnsupportedPolicy::Error`, on both paths.
#[tokio::test]
async fn disabled_tool_choice_leaves_the_tools_out_silently_where_it_is_ignored() {
    let request = ChatRequest::new("m")
        .user("hi")
        .tools([ToolDefinition::new("lookup", json!({"type": "object"}))])
        .tool_choice(ToolChoice::Disabled)
        .unsupported_policy(UnsupportedPolicy::Error);
    for outcome in [
        chat_probe(dialect("ollama"), request.clone()).await,
        chat_stream_probe(dialect("ollama"), request.clone()).await,
    ] {
        let response = outcome.result.expect("the request is honored");
        assert!(response.warnings.is_empty(), "{:?}", response.warnings);
        let [body] = outcome.sent.as_slice() else {
            panic!("one request: {:?}", outcome.sent);
        };
        assert!(body.get("tools").is_none(), "{body}");
        assert!(body.get("tool_choice").is_none(), "{body}");
    }
}

#[tokio::test]
async fn a_provider_without_tool_calls_passes_the_request_audits() {
    let without_tools =
        || generic().chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Unsupported);
    let provider = without_tools().build().unwrap();
    assert_request_fields_accounted(
        &provider,
        "m",
        &[],
        |request| chat_probe(without_tools(), request),
        |request| chat_stream_probe(without_tools(), request),
    )
    .await;
    assert_app_data_never_sent(
        &provider,
        "m",
        |request| chat_probe(without_tools(), request),
        |request| chat_stream_probe(without_tools(), request),
    )
    .await;
}

/// Reasoning never reaches the wire for a model that cannot replay it, in
/// any dialect (including those that replay `reasoning_content`) and in the
/// Responses provider with replay disabled; the warnings name the caller's
/// blocks.
#[tokio::test]
async fn unreplayable_reasoning_is_never_sent_by_any_dialect() {
    for &name in DIALECTS {
        println!("dialect: {name}");
        let no_replay = || {
            dialect(name).chat_capability(
                ChatCapability::ReasoningReplay,
                CapabilitySupport::Unsupported,
            )
        };
        assert_unreplayable_reasoning_dropped(
            &no_replay().build().unwrap(),
            "m",
            |request| chat_probe(no_replay(), request),
            |request| chat_stream_probe(no_replay(), request),
        )
        .await;
    }
    let disabled = || {
        let mut quirks = ResponsesQuirks::new("openai-compat:demo");
        quirks.reasoning_replay = ReasoningReplay::Disabled;
        responses().quirks(quirks)
    };
    assert_unreplayable_reasoning_dropped(
        &disabled().build().unwrap(),
        "m",
        |request| responses_probe(disabled(), request),
        |request| responses_stream_probe(disabled(), request),
    )
    .await;
}

/// A provider that takes its key from each request: embedding validation
/// refuses a request that carries none, as `embed` does, before sending.
#[tokio::test]
async fn embedding_validation_refuses_a_missing_scoped_credential() {
    let scoped = || generic().credential_scope("tenant");
    let request = EmbeddingRequest::new("m").input("Hi");
    let err = scoped()
        .build()
        .unwrap()
        .validate_embedding_request(&request)
        .unwrap_err();
    assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    assert_embedding_validation_agrees(
        &scoped().build().unwrap(),
        |request| async move {
            let transport = ReplayTransport::new([embeddings_ok(request.inputs.len())]);
            let provider = scoped().http_transport(transport.clone()).build().unwrap();
            let result = provider.embed(&request).await;
            ProbeOutcome::new(result, bodies(&transport))
        },
        &[request],
    )
    .await;
}

/// An endpoint that only streams: its unary validation refuses what its
/// unary call refuses.
#[tokio::test]
async fn responses_validation_agrees_with_the_call() {
    let stream_only = || {
        let mut quirks = ResponsesQuirks::new("openai-compat:stream-only");
        quirks.streaming = allama_openai_compat::StreamingSupport::StreamOnly;
        responses().quirks(quirks)
    };
    {
        let builder = stream_only;
        let provider = builder().build().unwrap();
        let validation = provider
            .validate_chat_stream_request(&ChatRequest::new("m").user("Hi"))
            .unwrap();
        assert_eq!(validation.coverage(), ValidationCoverage::Encoding);
        assert_validation_agrees(
            &provider,
            |request| responses_probe(builder(), request),
            |request| responses_stream_probe(builder(), request),
            &drift_requests("m"),
        )
        .await;
    }
    let unary = stream_only()
        .build()
        .unwrap()
        .validate_chat_request(&ChatRequest::new("m").user("Hi"));
    assert!(matches!(unary, Err(Error::Unsupported { .. })), "{unary:?}");
}

/// A provider that takes its key from each request: validation refuses a
/// request that carries none, as the call does, before sending.
#[tokio::test]
async fn validation_refuses_a_missing_scoped_credential() {
    let scoped = || generic().credential_scope("tenant");
    let request = ChatRequest::new("m").user("Hi");
    let err = scoped()
        .build()
        .unwrap()
        .validate_chat_request(&request)
        .unwrap_err();
    assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    assert_validation_agrees(
        &scoped().build().unwrap(),
        |request| chat_probe(scoped(), request),
        |request| chat_stream_probe(scoped(), request),
        &[request],
    )
    .await;
}

/// Chat Completions' error envelope for each kind, for every dialect.
fn chat_envelope(
    kind: allama_conformance::stream_errors::ErrorKindProbe,
    upstream: &str,
) -> Option<String> {
    envelope_with_upstream(kind, "insufficient_quota", upstream)
}

/// A chunk of text before an in-band error, so the stream has started.
fn started_chat_stream() -> String {
    let chunk = json!({
        "id": "c1", "object": "chat.completion.chunk", "model": "m",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}]
    });
    format!("data: {chunk}\n\n")
}

/// `response.created`, so a Responses stream has started.
fn started_responses_stream() -> String {
    let event = json!({
        "type": "response.created",
        "response": {"id": "resp_1", "object": "response", "status": "in_progress", "model": "m", "output": []},
        "sequence_number": 0
    });
    format!("event: response.created\ndata: {event}\n\n")
}

/// The chat call of `name` with transport options attached.
async fn dialect_send(
    name: &'static str,
    options: allama::RequestOptions,
    streaming: bool,
) -> TransportOutcome {
    let transport = ReplayTransport::default();
    let provider = dialect(name)
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let mut request = ChatRequest::new("m").user("Hi");
    request.options = options;
    let result = if streaming {
        provider.chat_stream(&request).await.map(drop)
    } else {
        provider.chat(&request).await.map(drop)
    };
    TransportOutcome::new(result, transport.requests())
}

/// A unary body and a stream body of the same response, from
/// `fixtures/agree/<name>.json` and `fixtures/agree/<name>.sse`.
fn capture(name: &str) -> ResponseCapture {
    let read = |extension: &str| {
        let path = format!(
            "{}/fixtures/agree/{name}.{extension}",
            env!("CARGO_MANIFEST_DIR")
        );
        std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {path}: {err}"))
    };
    ResponseCapture::new(read("json"), read("sse"))
}

/// The parallel tool-call ids in `agree_tool_calls` (Chat Completions).
const CHAT_TOOL_CALL_IDS: [&str; 2] = [
    "call_Sc2XmUdB2yP3h3MwRNQMcNBT",
    "call_cNeHvSaJ4H4XiLBXG5axpNqZ",
];

/// `agree_tool_calls` with the tool call ids an endpoint of `format`
/// issues: OpenAI's where it takes them, otherwise ids of its own shape
/// (Mistral's nine letters and digits), since an endpoint's own ids must
/// reach the next request unchanged.
fn tool_call_capture(format: ToolCallIdFormat) -> ResponseCapture {
    let mut capture = capture("agree_tool_calls");
    let mut ids = Vec::new();
    for id in CHAT_TOOL_CALL_IDS {
        let own = format.conform(id).into_owned();
        capture.unary = capture.unary.replace(id, &own);
        capture.stream = capture.stream.replace(id, &own);
        ids.push(own);
    }
    capture.wire_tool_call_ids(ids)
}

/// The parallel tool-call ids in `responses_tool_calls` (`call_id`s).
const RESPONSES_TOOL_CALL_IDS: [&str; 2] = ["call_agree_paris", "call_agree_tokyo"];

/// `chat` or `chat_stream` of a generic provider (Responses when
/// `responses`) built with its default `reqwest` client against
/// `base_url`, for the redirect contract.
#[cfg(feature = "reqwest")]
async fn redirect_call(base_url: String, responses: bool, streaming: bool) -> allama::Result<()> {
    let request = ChatRequest::new("m").user("hi");
    if responses {
        let provider = ResponsesProvider::builder()
            .base_url(base_url)
            .bearer_token("k")
            .build()?;
        if streaming {
            provider.chat_stream(&request).await.map(drop)
        } else {
            provider.chat(&request).await.map(drop)
        }
    } else {
        let provider = Provider::builder()
            .base_url(base_url)
            .bearer_token("k")
            .build()?;
        if streaming {
            provider.chat_stream(&request).await.map(drop)
        } else {
            provider.chat(&request).await.map(drop)
        }
    }
}

/// A `200` whose body is `body`, with a request id in the header each
/// dialect reads (Azure's is `apim-request-id`).
fn malformed(body: String) -> ReplayTransport {
    ReplayTransport::new([ReplayResponse::new(StatusCode::OK, body)
        .header("x-request-id", HeaderValue::from_static("req_1"))
        .header("apim-request-id", HeaderValue::from_static("req_1"))])
}

/// Every chat contract, for the generic dialect and every preset. The
/// presets that constrain no schema are found from their quirks, so a new
/// one is covered without editing this test. Unreplayable reasoning needs a model
/// that cannot replay it, which `m` is not; it runs with an override below.
#[tokio::test]
async fn every_dialect_passes_every_chat_contract() {
    let mut without_strict = Vec::new();
    for &name in DIALECTS {
        println!("dialect: {name}");
        let provider = dialect(name).build().unwrap();
        let validation = provider
            .validate_chat_request(&ChatRequest::new("m").user("Hi"))
            .unwrap();
        assert_eq!(validation.coverage(), ValidationCoverage::Encoding);
        let quirks = provider.quirks().clone();
        let started = started_chat_stream();
        let mut under_test = AdapterUnderTest::new(
            ["m"],
            quirks.namespace.clone(),
            |request| chat_probe(dialect(name), request),
            |request| chat_stream_probe(dialect(name), request),
        )
        .error_mapping(allama_conformance::error_kinds::http_error_mapping(
            move |status, headers, body| quirks.map_http_error(status, headers, body),
        ))
        .error_envelope(chat_envelope)
        .stream_body({
            let started = started.clone();
            move |envelope| format!("{started}data: {envelope}\n\n")
        })
        .stream_replay(move |body| async move {
            dialect(name)
                .http_transport(ReplayTransport::new([ReplayResponse::sse(body)]))
                .build()
                .unwrap()
                .chat_stream(&ChatRequest::new("m").user("hi"))
                .await
        })
        // `[DONE]` is the marker; no quirk changes this.
        .stream_ends(
            StreamEnds::new(started.clone())
                .terminated_without_finish(format!("{started}data: [DONE]\n\n")),
        )
        .sse_captures([format!("{started}data: [DONE]\n\n")])
        .transport_options(
            provider.credential_slot().clone(),
            move |options| dialect_send(name, options, false),
            move |options| dialect_send(name, options, true),
        )
        .malformed_response(
            move |body| async move {
                dialect(name)
                    .http_transport(malformed(body))
                    .build()
                    .unwrap()
                    .chat(&ChatRequest::new("m").user("hi"))
                    .await
            },
            Some("req_1"),
        )
        .unary_replay(move |body| async move {
            dialect(name)
                .http_transport(ReplayTransport::new([ReplayResponse::json(
                    StatusCode::OK,
                    body,
                )]))
                .build()
                .unwrap()
                .chat(&ChatRequest::new("m").user("hi"))
                .await
        })
        .response_captures([capture("agree_text")])
        .tool_call_capture(tool_call_capture(provider.quirks().tool_call_id_format))
        .length_capture(capture("agree_length"))
        .skip([Contract::UnreplayableReasoningDropped]);
        // Every preset shares the generic provider's client: the generic
        // dialect checks redirects for all of them.
        #[cfg(feature = "reqwest")]
        if name == "generic" {
            under_test = under_test.redirect_not_followed(|base_url, streaming| {
                redirect_call(base_url, false, streaming)
            });
        } else {
            under_test = under_test.skip([Contract::RedirectNotFollowed]);
        }
        // Without `reqwest` there is no HTTP client of the adapter's own.
        #[cfg(not(feature = "reqwest"))]
        {
            under_test = under_test.skip([Contract::RedirectNotFollowed]);
        }
        // A preset that constrains neither a response format nor a tool
        // sends a strict schema non-strict, with a warning.
        let quirks = provider.quirks();
        under_test = if quirks.response_format_constraint == SchemaConstraint::Never
            && quirks.tool_constraint == SchemaConstraint::Never
        {
            without_strict.push(name);
            under_test.strict_not_honoured(["m"])
        } else {
            under_test.skip([Contract::StrictNotHonouredWarns])
        };
        assert_chat_adapter(&provider, &under_test).await;
    }
    assert_eq!(
        without_strict,
        ["together", "fireworks", "deepinfra", "deepseek"]
    );
}

/// Every chat contract, for the Responses provider. An `error` event
/// carries its error inline; `response.completed` is the terminal marker.
#[tokio::test]
async fn responses_pass_every_chat_contract() {
    let provider = responses().build().unwrap();
    let validation = provider
        .validate_chat_stream_request(&ChatRequest::new("m").user("Hi"))
        .unwrap();
    assert_eq!(validation.coverage(), ValidationCoverage::Encoding);
    let namespace = generic().build().unwrap().quirks().namespace.clone();
    let started = started_responses_stream();
    let completed = json!({"type": "response.completed", "sequence_number": 2,
        "response": {"id": "resp_1", "model": "m", "output": []}});
    let under_test = AdapterUnderTest::new(
        ["m"],
        namespace,
        |request| responses_probe(responses(), request),
        |request| responses_stream_probe(responses(), request),
    )
    .error_mapping(allama_conformance::error_kinds::http_error_mapping(
        |status, headers, body| CompatQuirks::default().map_http_error(status, headers, body),
    ))
    .error_envelope(chat_envelope)
    .stream_body({
        let started = started.clone();
        move |envelope| {
            let mut error = serde_json::from_str::<Value>(envelope).unwrap()["error"].clone();
            error["type"] = json!("error");
            error["sequence_number"] = json!(1);
            format!("{started}event: error\ndata: {error}\n\n")
        }
    })
    .stream_replay(|body| async move {
        responses()
            .http_transport(ReplayTransport::new([ReplayResponse::sse(body)]))
            .build()
            .unwrap()
            .chat_stream(&ChatRequest::new("m").user("hi"))
            .await
    })
    .stream_ends(
        StreamEnds::new(started.clone()).terminated_without_finish(format!(
            "{started}event: response.completed\ndata: {completed}\n\n"
        )),
    )
    .sse_captures([format!(
        "{started}event: response.completed\ndata: {completed}\n\n"
    )])
    .transport_options(
        provider.credential_slot().clone(),
        |options| async move {
            let transport = ReplayTransport::default();
            let mut request = ChatRequest::new("m").user("Hi");
            request.options = options;
            let result = responses()
                .http_transport(transport.clone())
                .build()
                .unwrap()
                .chat(&request)
                .await;
            TransportOutcome::new(result, transport.requests())
        },
        |options| async move {
            let transport = ReplayTransport::default();
            let mut request = ChatRequest::new("m").user("Hi");
            request.options = options;
            let result = responses()
                .http_transport(transport.clone())
                .build()
                .unwrap()
                .chat_stream(&request)
                .await
                .map(drop);
            TransportOutcome::new(result, transport.requests())
        },
    )
    .malformed_response(
        |body| async move {
            responses()
                .http_transport(malformed(body))
                .build()
                .unwrap()
                .chat(&ChatRequest::new("m").user("hi"))
                .await
        },
        Some("req_1"),
    )
    .unary_replay(|body| async move {
        responses()
            .http_transport(ReplayTransport::new([ReplayResponse::json(
                StatusCode::OK,
                body,
            )]))
            .build()
            .unwrap()
            .chat(&ChatRequest::new("m").user("hi"))
            .await
    })
    .response_captures([capture("responses_text")])
    .tool_call_capture(capture("responses_tool_calls").wire_tool_call_ids(RESPONSES_TOOL_CALL_IDS))
    .length_capture(capture("responses_length"))
    // The Responses wire format has strict mode; replay is checked with an
    // override below.
    .skip([
        Contract::StrictNotHonouredWarns,
        Contract::UnreplayableReasoningDropped,
    ]);
    #[cfg(feature = "reqwest")]
    let under_test = under_test
        .redirect_not_followed(|base_url, streaming| redirect_call(base_url, true, streaming));
    // Without `reqwest` there is no HTTP client of the adapter's own.
    #[cfg(not(feature = "reqwest"))]
    let under_test = under_test.skip([Contract::RedirectNotFollowed]);
    assert_chat_adapter(&provider, &under_test).await;
}

/// Every embedding contract, for the generic dialect and every preset. A
/// dialect without a batch limit, or with an input-type field (DeepInfra,
/// OpenRouter), has no model for those contracts; the declared provider
/// below has a limit.
#[tokio::test]
async fn every_dialect_passes_every_embedding_contract() {
    let mut mapped = Vec::new();
    for &name in DIALECTS {
        println!("dialect: {name}");
        let provider = dialect(name).build().unwrap();
        let validation = provider
            .validate_embedding_request(&EmbeddingRequest::new("m").input("Hi"))
            .unwrap();
        assert_eq!(validation.coverage(), ValidationCoverage::Encoding);
        let quirks = provider.quirks().clone();
        let mut skip = Vec::new();
        if provider.embedding_limits("m").max_inputs_per_call.is_none() {
            skip.push(Contract::BatchLimitEnforced);
        }
        if quirks.embedding_input_type.is_some() {
            mapped.push(name);
            skip.push(Contract::InputTypeNotHonouredWarns);
        }
        let under_test =
            EmbeddingAdapterUnderTest::new(["m"], quirks.namespace.clone(), move |request| {
                dialect_embed_probe(name, request)
            })
            .answering(VectorOrder::Indexed, move |request, vectors| async move {
                let data: Vec<Value> = vectors
                    .iter()
                    .map(|vector| json!({"index": vector.index, "embedding": vector.values}))
                    .collect();
                let body = json!({"data": data, "model": "m"});
                dialect(name)
                    .http_transport(ReplayTransport::new([ReplayResponse::json(
                        StatusCode::OK,
                        body.to_string(),
                    )]))
                    .build()
                    .unwrap()
                    .embed(&request)
                    .await
            })
            .error_mapping(allama_conformance::error_kinds::http_error_mapping(
                move |status, headers, body| quirks.map_http_error(status, headers, body),
            ))
            .error_envelope(chat_envelope)
            .transport_options(
                provider.credential_slot().clone(),
                move |options| async move {
                    let transport = ReplayTransport::default();
                    let mut request = EmbeddingRequest::new("m").inputs(["hello"]);
                    request.options = options;
                    let result = dialect(name)
                        .http_transport(transport.clone())
                        .build()
                        .unwrap()
                        .embed(&request)
                        .await;
                    TransportOutcome::new(result, transport.requests())
                },
            )
            .skip(skip);
        assert_embedding_adapter(&provider, &under_test).await;
    }
    assert_eq!(mapped, ["deepinfra", "openrouter"]);
}

/// The embedding contracts against declared answers: no batches and no
/// output dimensions.
#[tokio::test]
async fn declared_embedding_answers_pass_every_embedding_contract() {
    let provider = embedding_provider().build().unwrap();
    let under_test =
        EmbeddingAdapterUnderTest::new(["m"], provider.quirks().namespace.clone(), embed_probe)
            .error_mapping(allama_conformance::error_kinds::http_error_mapping(
                |status, headers, body| {
                    CompatQuirks::default().map_http_error(status, headers, body)
                },
            ))
            .error_envelope(chat_envelope)
            // Its batches are refused before the limit or an input's
            // provider data is checked (both are checked on the dialects, and
            // the limit alone above), and the answering transport is the
            // dialects'.
            .skip([
                Contract::BatchLimitEnforced,
                Contract::InputProviderDataChecked,
                Contract::EmbeddingResponseChecked,
                Contract::EmbeddingCount,
                Contract::TransportOptionsApplied,
            ]);
    assert_embedding_adapter(&provider, &under_test).await;
}
