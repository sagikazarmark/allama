//! Request honesty contracts (C2, C3, C6) for both OpenAI providers: what a
//! request asks for reaches the wire, or the caller is told.

use allama::prelude::*;
use allama::{
    CapabilitySupport, EmbeddingCapability, EmbeddingInputType, EmbeddingProviderExt,
    EmbeddingResponse, RequestField, Subject, UnsupportedPolicy,
};
use allama_conformance::probe::ProbeOutcome;
use allama_conformance::{assert_embedding_validation_agrees, assert_validation_agrees};
use allama_http::{ReplayResponse, ReplayTransport};
use allama_openai::{Provider, ResponsesProvider};
use futures_util::StreamExt;
use serde_json::{Value, json};

fn chat_ok() -> ReplayResponse {
    ReplayResponse::json(
        200,
        json!({
            "id": "chatcmpl-1", "object": "chat.completion", "model": "gpt-4.1",
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
            "id": "resp_1", "object": "response", "status": "completed", "model": "gpt-4.1",
            "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "ok", "annotations": []}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        })
        .to_string(),
    )
}

fn embeddings_ok(count: usize) -> ReplayResponse {
    let data: Vec<Value> = (0..count)
        .map(|i| json!({"object": "embedding", "index": i, "embedding": [0.1, 0.2]}))
        .collect();
    ReplayResponse::json(
        200,
        json!({"object": "list", "model": "text-embedding-3-small", "data": data,
               "usage": {"prompt_tokens": count, "total_tokens": count}})
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

fn provider(transport: &ReplayTransport) -> Provider {
    Provider::builder()
        .api_key("k")
        .http_transport(transport.clone())
        .build()
        .unwrap()
}

fn responses_provider(transport: &ReplayTransport) -> ResponsesProvider {
    ResponsesProvider::builder()
        .api_key("k")
        .http_transport(transport.clone())
        .build()
        .unwrap()
}

async fn chat_probe(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
    let transport = ReplayTransport::new([chat_ok()]);
    let result = provider(&transport).chat(&request).await;
    ProbeOutcome::new(result, bodies(&transport))
}

async fn embed_probe(request: EmbeddingRequest) -> ProbeOutcome<EmbeddingResponse> {
    let transport = ReplayTransport::new([embeddings_ok(request.inputs.len())]);
    let result = provider(&transport).embed(&request).await;
    ProbeOutcome::new(result, bodies(&transport))
}

/// Chat Completions never returns reasoning, so reasoning in the history
/// came from elsewhere: it is dropped with a warning, streamed or not.
#[tokio::test]
async fn replayed_reasoning_is_dropped_with_a_warning() {
    let request = ChatRequest::new("gpt-5-mini")
        .user("2 + 2?")
        .message(allama::Message::assistant_blocks(vec![
            allama::ContentBlock::reasoning("Two plus two is four."),
            allama::ContentBlock::text("4"),
        ]))
        .user("3 + 3?");
    let outcome = chat_probe(request.clone()).await;
    let response = outcome.result.unwrap();
    assert_eq!(response.warnings.len(), 1, "{:?}", response.warnings);
    assert_eq!(response.warnings[0].kind, allama::WarningKind::Dropped);
    assert_eq!(
        outcome.sent[0]["messages"][1],
        json!({"role": "assistant", "content": "4"})
    );

    let sse = "data: {\"id\":\"c1\",\"model\":\"gpt-5-mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"6\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
    let transport = ReplayTransport::new([ReplayResponse::sse(sse)]);
    let streamed = provider(&transport)
        .chat_stream(&request)
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();
    assert_eq!(streamed.warnings, response.warnings);
}

/// The batch limit names the way out: `embed_chunked` splits at the limit.
#[tokio::test]
async fn oversized_batch_fails_before_sending_and_chunks_on_request() {
    let transport = ReplayTransport::new([embeddings_ok(2048), embeddings_ok(1)]);
    let provider = provider(&transport);
    let inputs: Vec<String> = (0..2049).map(|i| format!("doc {i}")).collect();
    let request = EmbeddingRequest::new("text-embedding-3-small").inputs(inputs);

    let err = provider.embed(&request).await.unwrap_err();
    assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
    assert!(err.message().contains("2048"), "{err}");
    assert!(err.message().contains("embed_chunked"), "{err}");
    assert!(transport.requests().is_empty());

    let response = provider.embed_chunked(&request, 2048).await.unwrap();
    assert_eq!(response.embeddings.len(), 2049);
    let sizes: Vec<usize> = bodies(&transport)
        .iter()
        .map(|b| b["input"].as_array().unwrap().len())
        .collect();
    assert_eq!(sizes, [2048, 1]);
}

/// A warning every chunk repeats is reported once for the whole batch.
#[tokio::test]
async fn chunked_embedding_reports_a_repeated_warning_once() {
    let transport =
        ReplayTransport::new([embeddings_ok(2048), embeddings_ok(2048), embeddings_ok(1)]);
    let provider = provider(&transport);
    let inputs: Vec<String> = (0..4097).map(|i| format!("doc {i}")).collect();
    // OpenAI has no query/document field, so each chunk of a model that is
    // not known to be symmetric warns.
    let request = EmbeddingRequest::new("text-embedding-4-large")
        .inputs(inputs)
        .input_type(EmbeddingInputType::Query);

    let response = provider.embed_chunked(&request, 2048).await.unwrap();
    assert_eq!(response.embeddings.len(), 4097);
    assert_eq!(transport.requests().len(), 3);
    assert_eq!(response.warnings.len(), 1, "{:?}", response.warnings);
    assert_eq!(
        response.warnings[0].subject,
        Subject::Field {
            field: RequestField::InputType
        }
    );
}

/// `dimensions` is sent for every model (ADR 0015): this crate no longer
/// refuses it for `text-embedding-ada-002`, and OpenAI's rejection is the
/// answer. The claim is in `docs/catalog/openai.md`; no rejection text is
/// documented, so it stays an `InvalidRequest`.
#[tokio::test]
async fn dimensions_are_sent_for_every_model() {
    let transport = ReplayTransport::new([embeddings_ok(1)]);
    let provider = provider(&transport);
    assert_eq!(
        provider.embedding_capability(
            "text-embedding-ada-002",
            EmbeddingCapability::OutputDimensions
        ),
        CapabilitySupport::Unknown
    );
    let request = EmbeddingRequest::new("text-embedding-ada-002")
        .input("x")
        .dimensions(2);
    let response = provider.embed(&request).await.unwrap();
    assert!(response.warnings.is_empty(), "{:?}", response.warnings);
    assert_eq!(response.provider.as_deref(), Some("openai"));
    assert_eq!(bodies(&transport)[0]["dimensions"], 2);
}

/// Models this crate does not know (fine-tunes, future releases) are not
/// rejected: `dimensions` is sent and the API decides.
#[tokio::test]
async fn dimensions_on_an_unknown_model_is_sent() {
    let transport = ReplayTransport::new([embeddings_ok(1)]);
    let provider = provider(&transport);
    assert_eq!(
        provider.embedding_capability(
            "text-embedding-4-large",
            EmbeddingCapability::OutputDimensions
        ),
        CapabilitySupport::Unknown
    );
    // The canned vectors have 2 values, the size the request asks for.
    let request = EmbeddingRequest::new("text-embedding-4-large")
        .input("x")
        .dimensions(2);
    provider.embed(&request).await.unwrap();
    assert_eq!(bodies(&transport)[0]["dimensions"], 2);
}

/// The symmetric models embed queries and documents alike, so the input
/// type is honored by sending nothing: no warning, under every policy, and
/// the request sent is the one without it. Evidence for the `InputType`
/// claim: the embeddings API takes no role
/// (<https://platform.openai.com/docs/api-reference/embeddings/create>), and
/// its guide embeds queries with the same call as documents
/// (<https://platform.openai.com/docs/guides/embeddings#use-cases>).
#[tokio::test]
async fn symmetric_models_take_any_input_type() {
    for model in [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ] {
        let base = || EmbeddingRequest::new(model).input("hello");
        let baseline = embed_probe(base()).await;
        for input_type in [EmbeddingInputType::Query, EmbeddingInputType::Document] {
            for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
                let request = base().input_type(input_type).unsupported_policy(policy);
                let validation = provider(&ReplayTransport::new([]))
                    .validate_embedding_request(&request)
                    .unwrap();
                assert!(validation.warnings().is_empty(), "{model} {input_type}");
                let outcome = embed_probe(request).await;
                let response = outcome.result.unwrap();
                assert!(response.warnings.is_empty(), "{:?}", response.warnings);
                assert_eq!(
                    outcome.sent, baseline.sent,
                    "{model} {input_type} {policy:?}"
                );
            }
        }
    }
}

/// `chat` and `chat_stream` of both providers run `allama::adapter::chat::preflight`
/// against the provider's own answers, including overrides the encoder knows
/// nothing about, and agree with `validate_chat_request`.
#[tokio::test]
async fn preflight_applies_capability_answers() {
    fn overrides(_: &str, capability: allama::ChatCapability) -> Option<CapabilitySupport> {
        use allama::ChatCapability;
        matches!(
            capability,
            ChatCapability::ToolCalls | ChatCapability::ReasoningConfig
        )
        .then_some(CapabilitySupport::Unsupported)
    }
    check_preflight(
        |t| provider(t).with_chat_capabilities(overrides),
        chat_ok,
        "/reasoning_effort",
    )
    .await;
    check_preflight(
        |t| responses_provider(t).with_chat_capabilities(overrides),
        responses_ok,
        "/reasoning",
    )
    .await;
}

async fn check_preflight<P: ChatProvider<Stream = allama::ChatStream>>(
    provider: impl Fn(&ReplayTransport) -> P,
    ok: fn() -> ReplayResponse,
    reasoning_pointer: &str,
) {
    use allama::{Error, RequestField, Subject, UnsupportedPolicy, WarningKind};

    let is_reasoning_warning = |w: &allama::Warning| {
        w.kind == WarningKind::Unsupported
            && w.subject
                == Subject::Field {
                    field: RequestField::Reasoning,
                }
    };

    // An unsupported hint under `Warn`: validation passes, `chat` serves the
    // request with a warning, and the hint never reaches the wire.
    let hint = ChatRequest::new("gpt-5")
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

    // The stream reports the warning before anything the upstream sends.
    let transport = ReplayTransport::new([ReplayResponse::sse("data: [DONE]\n\n")]);
    let first = provider(&transport)
        .chat_stream(&hint)
        .await
        .unwrap()
        .next()
        .await;
    assert!(
        matches!(
            &first,
            Some(Ok(StreamEvent::Warning { warning, .. })) if is_reasoning_warning(warning)
        ),
        "{first:?}"
    );

    // The same hint under `Error`, and a semantic feature under any policy:
    // validation, `chat`, and `chat_stream` all refuse before sending.
    let strict = hint.unsupported_policy(UnsupportedPolicy::Error);
    let tools = ChatRequest::new("gpt-5")
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

/// A provider that takes its key from each request: validation refuses a
/// request that carries none, as the call does, before sending.
#[tokio::test]
async fn validation_refuses_a_missing_scoped_credential() {
    fn scoped(transport: &ReplayTransport) -> Provider {
        Provider::builder()
            .credential_scope("tenant")
            .http_transport(transport.clone())
            .build()
            .unwrap()
    }
    let request = ChatRequest::new("gpt-4.1").user("Hi");
    let err = scoped(&ReplayTransport::new([]))
        .validate_chat_request(&request)
        .unwrap_err();
    assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    assert_validation_agrees(
        &scoped(&ReplayTransport::new([])),
        |request| async move {
            let transport = ReplayTransport::new([chat_ok()]);
            let result = scoped(&transport).chat(&request).await;
            ProbeOutcome::new(result, bodies(&transport))
        },
        |request| async move {
            let transport = ReplayTransport::new([ReplayResponse::sse(include_str!(
                "../fixtures/recorded/stream_text.sse"
            ))]);
            let result = match scoped(&transport).chat_stream(&request).await {
                Ok(stream) => stream.collect_response().await,
                Err(err) => Err(err),
            };
            ProbeOutcome::new(result, bodies(&transport))
        },
        &[request],
    )
    .await;
}

/// A provider that takes its key from each request: embedding validation
/// refuses a request that carries none, as `embed` does, before sending.
#[tokio::test]
async fn embedding_validation_refuses_a_missing_scoped_credential() {
    fn scoped(transport: &ReplayTransport) -> Provider {
        Provider::builder()
            .credential_scope("tenant")
            .http_transport(transport.clone())
            .build()
            .unwrap()
    }
    let request = EmbeddingRequest::new("text-embedding-3-small").input("Hi");
    let err = scoped(&ReplayTransport::new([]))
        .validate_embedding_request(&request)
        .unwrap_err();
    assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    assert_embedding_validation_agrees(
        &scoped(&ReplayTransport::new([])),
        |request| async move {
            let transport = ReplayTransport::new([embeddings_ok(request.inputs.len())]);
            let result = scoped(&transport).embed(&request).await;
            ProbeOutcome::new(result, bodies(&transport))
        },
        &[request],
    )
    .await;
}

/// A reasoning effort is sent on both APIs whatever the model (ADR 0015):
/// whether gpt-4.1 reasons is not claimed, and OpenAI's rejection is the
/// answer. An adapter-level override that says the model does not reason
/// reaches the request rules as well as preflight: the effort is then
/// dropped with a warning.
#[tokio::test]
async fn reasoning_follows_the_profile_and_adapter_overrides() {
    let request = ChatRequest::new("gpt-4.1")
        .user("q")
        .reasoning(ReasoningEffort::High);
    let transport = ReplayTransport::new([chat_ok()]);
    let response = provider(&transport).chat(&request).await.unwrap();
    assert!(response.warnings.is_empty(), "{:?}", response.warnings);
    assert_eq!(bodies(&transport)[0]["reasoning_effort"], "high");

    let transport = ReplayTransport::new([responses_ok()]);
    let response = responses_provider(&transport).chat(&request).await.unwrap();
    assert!(response.warnings.is_empty(), "{:?}", response.warnings);
    assert_eq!(bodies(&transport)[0]["reasoning"]["effort"], "high");

    let downgrade = |_: &str, capability| {
        (capability == allama::ChatCapability::ReasoningConfig)
            .then_some(allama::CapabilitySupport::Unsupported)
    };
    let transport = ReplayTransport::new([chat_ok()]);
    let response = provider(&transport)
        .with_chat_capabilities(downgrade)
        .chat(&request)
        .await
        .unwrap();
    assert_eq!(response.warnings.len(), 1, "{:?}", response.warnings);
    assert!(bodies(&transport)[0].get("reasoning_effort").is_none());

    let transport = ReplayTransport::new([responses_ok()]);
    let response = responses_provider(&transport)
        .with_chat_capabilities(downgrade)
        .chat(&request)
        .await
        .unwrap();
    assert_eq!(response.warnings.len(), 1, "{:?}", response.warnings);
    assert!(bodies(&transport)[0].get("reasoning").is_none());
}

/// Sampling with reasoning (ADR 0015): `temperature`, `top_p` and
/// `logprobs` are sent as asked, whatever the model and its effort, and so
/// are tools on GPT-6 over Chat Completions. The rules that dropped or
/// refused them are in `docs/catalog/openai.md`; a caller re-adds them with
/// `with_model_rules`. OpenAI's rejection ("When reasoning effort is not
/// none, remove temperature, top_p, and top_logprobs. For Chat Completions,
/// also remove logprobs",
/// <https://developers.openai.com/api/docs/guides/latest-model>) maps to a
/// provider-stage refusal of the field.
#[tokio::test]
async fn sampling_and_tools_are_sent_whatever_the_model() {
    let sampled = |model: &str| {
        ChatRequest::new(model)
            .user("q")
            .temperature(0.2)
            .top_p(0.9)
            .with_option(allama_openai::ChatRequestOptions::new().extra("logprobs", true))
    };
    for request in [
        sampled("gpt-6-sol"),
        sampled("gpt-5"),
        sampled("o3").reasoning(ReasoningEffort::High),
        sampled("gpt-5.1").reasoning(ReasoningEffort::Low),
    ] {
        let strict = request.clone().unsupported_policy(UnsupportedPolicy::Error);
        assert!(
            provider(&ReplayTransport::new([]))
                .validate_chat_request(&strict)
                .is_ok(),
            "{}",
            request.model
        );
        let transport = ReplayTransport::new([chat_ok()]);
        let response = provider(&transport).chat(&request).await.unwrap();
        assert!(response.warnings.is_empty(), "{:?}", response.warnings);
        let body = &bodies(&transport)[0];
        assert_eq!(body["logprobs"], true, "{}", request.model);
        assert!(body["temperature"].is_number() && body["top_p"].is_number());

        let transport = ReplayTransport::new([responses_ok()]);
        let response = responses_provider(&transport).chat(&request).await.unwrap();
        let kinds: Vec<_> = response
            .warnings
            .iter()
            .map(|w| w.subject.clone())
            .collect();
        // Only the Chat Completions options are reported: Responses reads
        // its own namespace.
        assert_eq!(kinds.len(), 1, "{:?}", response.warnings);
        let body = &bodies(&transport)[0];
        assert!(body["temperature"].is_number() && body["top_p"].is_number());
    }

    for model in ["gpt-6-sol", "gpt-6-astra", "gpt-6.1-sol"] {
        let tools = ChatRequest::new(model)
            .user("q")
            .tools(vec![ToolDefinition::new("t", json!({"type": "object"}))]);
        let transport = ReplayTransport::new([chat_ok()]);
        provider(&transport).chat(&tools).await.unwrap();
        assert_eq!(bodies(&transport)[0]["tools"][0]["function"]["name"], "t");
    }

    // OpenAI's refusal names the field.
    let refusal = ReplayResponse::json(
        400,
        json!({"error": {
            "message": "Unsupported parameter: 'temperature' is not supported with this model.",
            "type": "invalid_request_error",
            "param": "temperature",
            "code": "unsupported_parameter"
        }})
        .to_string(),
    );
    let transport = ReplayTransport::new([refusal]);
    let err = provider(&transport)
        .chat(&sampled("gpt-6-sol"))
        .await
        .unwrap_err();
    assert_eq!(err.unsupported_field(), Some(RequestField::Temperature));
    assert_eq!(
        err.unsupported_stage(),
        Some(allama::UnsupportedStage::Provider)
    );

    // Responses sends the same refusal with `code: null`: the message
    // prefix stands in for the code.
    let refusal = ReplayResponse::json(
        400,
        json!({"error": {
            "message": "Unsupported parameter: 'temperature' is not supported with this model.",
            "type": "invalid_request_error",
            "param": "temperature",
            "code": null
        }})
        .to_string(),
    );
    let transport = ReplayTransport::new([refusal]);
    let err = responses_provider(&transport)
        .chat(&sampled("gpt-6-sol"))
        .await
        .unwrap_err();
    assert_eq!(err.unsupported_field(), Some(RequestField::Temperature));
    assert_eq!(
        err.unsupported_stage(),
        Some(allama::UnsupportedStage::Provider)
    );

    // A caller's model rules re-add the drop, on both APIs.
    let transport = ReplayTransport::new([chat_ok()]);
    let response = provider(&transport)
        .with_model_rules(|model: &str| {
            (model == "gpt-6-sol").then(|| {
                allama_openai::ChatModelRules::new()
                    .sampling(allama_openai::SamplingSupport::WithoutReasoning)
                    .default_effort(ReasoningEffort::Medium)
            })
        })
        .chat(&sampled("gpt-6-sol"))
        .await
        .unwrap();
    assert_eq!(response.warnings.len(), 3, "{:?}", response.warnings);
    let body = &bodies(&transport)[0];
    assert!(body.get("temperature").is_none() && body.get("logprobs").is_none());

    let transport = ReplayTransport::new([responses_ok()]);
    responses_provider(&transport)
        .with_model_rules(|_: &str| {
            Some(
                allama_openai::ResponsesModelRules::new()
                    .sampling(allama_openai::SamplingSupport::Rejected),
            )
        })
        .chat(&sampled("gpt-6-sol"))
        .await
        .unwrap();
    assert!(bodies(&transport)[0].get("temperature").is_none());
}

/// Effort levels are wire format (ADR 0004): a level a documented model
/// does not take is sent as the nearest one it does, with a warning
/// (`gpt-5-nano` rejects `xhigh`, verified live). Any other model gets its
/// effort unchanged, and a caller's rules replace the levels.
#[tokio::test]
async fn effort_levels_are_overridable_and_unknown_ids_pass_through() {
    let request = |model: &str| {
        ChatRequest::new(model)
            .user("q")
            .reasoning(ReasoningEffort::XHigh)
    };
    let sent = |provider: ResponsesProvider, request: ChatRequest| async move {
        let response = provider.chat(&request).await.unwrap();
        (response.warnings.len(), response)
    };

    let transport = ReplayTransport::new([responses_ok()]);
    let (warnings, _) = sent(responses_provider(&transport), request("gpt-5-nano")).await;
    assert_eq!(warnings, 1);
    assert_eq!(bodies(&transport)[0]["reasoning"]["effort"], "high");

    for model in ["gpt-5.4", "gpt-6.1-astra", "my-proxy-model"] {
        let transport = ReplayTransport::new([responses_ok()]);
        let (warnings, _) = sent(responses_provider(&transport), request(model)).await;
        assert_eq!(warnings, 0, "{model}");
        assert_eq!(
            bodies(&transport)[0]["reasoning"]["effort"],
            "xhigh",
            "{model}"
        );
    }

    let transport = ReplayTransport::new([responses_ok()]);
    let provider = responses_provider(&transport).with_model_rules(|model: &str| {
        (model == "gpt-5-nano").then(allama_openai::ResponsesModelRules::new)
    });
    let (warnings, _) = sent(provider, request("gpt-5-nano")).await;
    assert_eq!(warnings, 0);
    assert_eq!(bodies(&transport)[0]["reasoning"]["effort"], "xhigh");
}

/// Which endpoint serves a model is model data (ADR 0015): a model OpenAI
/// serves on Responses only (`o3-pro`, `gpt-5-codex`, `o3-deep-research`
/// and any id with a `pro` token) is sent on Chat Completions as asked,
/// tools and schema included, and o1-mini is sent on Responses. OpenAI's
/// rejection is the answer; the evidence that used to refuse them is in
/// `docs/catalog/openai.md`.
#[tokio::test]
async fn no_model_is_refused_for_its_endpoint() {
    for model in ["o3-pro", "gpt-5-codex", "o3-deep-research", "acme-pro-1"] {
        let request = ChatRequest::new(model)
            .user("q")
            .tools(vec![ToolDefinition::new("t", json!({"type": "object"}))])
            .response_format(ResponseFormat::json_schema(json!({
                "type": "object", "properties": {}, "required": [],
                "additionalProperties": false
            })));
        let transport = ReplayTransport::new([chat_ok()]);
        let response = provider(&transport).chat(&request).await.unwrap();
        assert!(
            response.warnings.is_empty(),
            "{model}: {:?}",
            response.warnings
        );
        let body = &bodies(&transport)[0];
        assert_eq!(body["model"], model);
        assert_eq!(body["tools"][0]["function"]["name"], "t", "{model}");
        assert_eq!(body["response_format"]["type"], "json_schema", "{model}");
    }

    let transport = ReplayTransport::new([responses_ok()]);
    responses_provider(&transport)
        .chat(&ChatRequest::new("o1-mini").user("q"))
        .await
        .unwrap();
    assert_eq!(bodies(&transport)[0]["model"], "o1-mini");
}
