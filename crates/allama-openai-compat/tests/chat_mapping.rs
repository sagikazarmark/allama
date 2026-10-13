//! Chat Completions request mapping: the output-only reasoning setting,
//! effort levels, `top_k`, `safety_identifier`, stored files, tool
//! definition provider data, and what the presets send.

use allama::{
    CacheRetention, CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ContentBlock,
    ContentPart, File, Image, MediaSource, Message, PromptCache, ProviderData, Reasoning,
    ReasoningConfig, ReasoningEffort, ReasoningOutput, ToolDefinition, UnsupportedPolicy,
    WarningKind,
};
use allama_http::{ReplayResponse, ReplayTransport};
use allama_openai_compat::providers::{DeepSeek, Mistral, OpenRouter};
use allama_openai_compat::{ChatModelRules, ChatRequestOptions, CompatQuirks, Provider};
use serde_json::{Value, json};

const NAMESPACE: &str = "openai-compat:vendor";

fn ok_body() -> ReplayResponse {
    ReplayResponse::json(
        200,
        r#"{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#,
    )
}

fn provider(quirks: CompatQuirks) -> (Provider, ReplayTransport) {
    let transport = ReplayTransport::new([ok_body()]);
    let provider = Provider::builder()
        .base_url("http://upstream/v1")
        .provider_name("vendor")
        .quirks(quirks)
        .model_rules(|model: &str| {
            if model == "levels" {
                ChatModelRules::new().effort_levels(&[ReasoningEffort::Low, ReasoningEffort::High])
            } else {
                ChatModelRules::new()
            }
        })
        .http_transport(transport.clone())
        .build()
        .unwrap();
    (provider, transport)
}

fn provider_for_quirks() -> (Provider, ReplayTransport) {
    provider(CompatQuirks::new(NAMESPACE))
}

async fn body(provider: &Provider, transport: &ReplayTransport, request: &ChatRequest) -> Value {
    provider.chat(request).await.unwrap();
    serde_json::from_str(transport.requests().last().unwrap().body_text().unwrap()).unwrap()
}

/// An output setting alone sends nothing and turns nothing on; it is
/// reported, `Omitted` included, since Chat Completions has no such
/// setting.
#[tokio::test]
async fn reasoning_output_alone_sends_nothing_and_warns() {
    for output in [
        ReasoningOutput::Omitted,
        ReasoningOutput::Summary,
        ReasoningOutput::Full,
    ] {
        let (provider, transport) = provider(CompatQuirks::new(NAMESPACE));
        let request = ChatRequest::new("m")
            .user("q")
            .reasoning(ReasoningConfig::new().output(output));
        let validation = provider.validate_chat_request(&request).unwrap();
        assert_eq!(
            validation
                .warnings()
                .iter()
                .map(|w| w.kind)
                .collect::<Vec<_>>(),
            [WarningKind::Unsupported],
            "{output:?}"
        );
        let body = body(&provider, &transport, &request).await;
        assert!(body.get("reasoning_effort").is_none(), "{body}");
    }
}

/// A model's effort levels map a requested level to the nearest
/// one it takes, with an `Approximated` warning.
#[tokio::test]
async fn effort_levels_map_to_the_nearest_level() {
    let (provider, transport) = provider(CompatQuirks::new(NAMESPACE));
    let request = ChatRequest::new("levels")
        .user("q")
        .reasoning(ReasoningConfig::new().effort(ReasoningEffort::XHigh));
    let validation = provider.validate_chat_request(&request).unwrap();
    assert_eq!(validation.warnings()[0].kind, WarningKind::Approximated);
    assert_eq!(
        body(&provider, &transport, &request).await["reasoning_effort"],
        "high"
    );
    // A model without levels gets the level as it is.
    let (plain, transport) = provider_for_quirks();
    let request = ChatRequest::new("m")
        .user("q")
        .reasoning(ReasoningConfig::new().effort(ReasoningEffort::Minimal));
    assert_eq!(
        body(&plain, &transport, &request).await["reasoning_effort"],
        "minimal"
    );
}

/// `top_k` goes to servers that take it, and is reported elsewhere.
#[tokio::test]
async fn top_k_is_sent_where_the_endpoint_takes_it() {
    let request = ChatRequest::new("m").user("q").top_k(40);
    let mut quirks = CompatQuirks::new(NAMESPACE);
    quirks.supports_top_k = true;
    let (with, transport) = provider(quirks);
    assert!(
        with.validate_chat_request(&request)
            .unwrap()
            .warnings()
            .is_empty()
    );
    assert_eq!(body(&with, &transport, &request).await["top_k"], 40);

    let (without, transport) = provider(CompatQuirks::new(NAMESPACE));
    let validation = without.validate_chat_request(&request).unwrap();
    assert_eq!(validation.warnings()[0].kind, WarningKind::Unsupported);
    assert!(
        body(&without, &transport, &request)
            .await
            .get("top_k")
            .is_none()
    );
    assert!(
        without
            .validate_chat_request(&request.clone().unsupported_policy(UnsupportedPolicy::Error))
            .is_err()
    );

    for builder in [
        OpenRouter::builder("k").unwrap(),
        allama_openai_compat::providers::Together::builder("k").unwrap(),
        allama_openai_compat::providers::Vllm::builder("http://localhost:8000/v1", None).unwrap(),
    ] {
        let provider = builder
            .http_transport(ReplayTransport::default())
            .build()
            .unwrap();
        assert!(provider.quirks().supports_top_k);
    }
}

/// `safety_identifier` is a typed option.
#[tokio::test]
async fn safety_identifier_is_sent() {
    let (provider, transport) = provider(CompatQuirks::new(NAMESPACE));
    let request = ChatRequest::new("m")
        .user("q")
        .with_option(ChatRequestOptions::new().safety_identifier("user-hash"));
    assert_eq!(
        body(&provider, &transport, &request).await["safety_identifier"],
        "user-hash"
    );
}

/// A document the endpoint stores goes by its file id; a stored image
/// has no Chat Completions form.
#[tokio::test]
async fn stored_documents_are_sent_by_id() {
    let (provider, transport) = provider(CompatQuirks::new(NAMESPACE));
    let document = File::new(
        "application/pdf",
        MediaSource::provider_file(NAMESPACE, "file-1"),
    )
    .with_filename("report.pdf");
    let request = ChatRequest::new("m").message(Message::user(vec![
        ContentPart::File(document),
        ContentPart::text("Summarize."),
    ]));
    let body = body(&provider, &transport, &request).await;
    assert_eq!(
        body["messages"][0]["content"][0],
        json!({"type": "file", "file": {"file_id": "file-1", "filename": "report.pdf"}})
    );

    let image = ChatRequest::new("m").message(Message::user(vec![ContentPart::Image(Image::new(
        MediaSource::provider_file(NAMESPACE, "file-2"),
    ))]));
    let err = provider.validate_chat_request(&image).unwrap_err();
    assert!(matches!(err, allama::Error::Unsupported { .. }), "{err:?}");
}

/// A tool definition's own provider data has no place in a Chat
/// Completions function definition: it is reported, not silently ignored.
#[tokio::test]
async fn tool_definition_provider_data_is_reported() {
    let (provider, transport) = provider(CompatQuirks::new(NAMESPACE));
    let mut tool = ToolDefinition::new("f", json!({"type": "object"}));
    tool.provider_data = ProviderData::new().with_raw(NAMESPACE, "defer_loading", json!(true));
    let request = ChatRequest::new("m").user("q").tools([tool]);
    let validation = provider.validate_chat_request(&request).unwrap();
    assert_eq!(
        validation
            .warnings()
            .iter()
            .map(|w| w.kind)
            .collect::<Vec<_>>(),
        [WarningKind::Dropped]
    );
    let body = body(&provider, &transport, &request).await;
    assert!(body["tools"][0]["function"].get("defer_loading").is_none());
}

/// DeepSeek takes its `reasoning_content` back (thinking-mode tool calls
/// need it); Mistral is sent no `stream_options`, and its
/// stream usage is not claimed absent.
#[tokio::test]
async fn presets_replay_and_stream_as_their_endpoints_need() {
    let transport = ReplayTransport::new([ok_body()]);
    let deepseek = DeepSeek::builder("k")
        .unwrap()
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let reasoning = Reasoning::new("think").with_provider_data(ProviderData::new().with_raw(
        DeepSeek::NAMESPACE,
        "reasoning_field",
        json!("reasoning_content"),
    ));
    let mut request = ChatRequest::new("deepseek-chat").user("q");
    request.push_message(Message::assistant_blocks(vec![
        ContentBlock::Reasoning(reasoning),
        ContentBlock::tool_call("call_1", "f", "{}"),
    ]));
    request.push_message(Message::tool_result("call_1", "f", "done"));
    let response = deepseek.chat(&request).await.unwrap();
    assert!(response.warnings.is_empty(), "{:?}", response.warnings);
    let body: Value = serde_json::from_str(transport.requests()[0].body_text().unwrap()).unwrap();
    assert_eq!(body["messages"][1]["reasoning_content"], "think");

    let transport = ReplayTransport::new([ReplayResponse::sse(
        "data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
    )]);
    let mistral = Mistral::builder("k")
        .unwrap()
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let _ = mistral
        .chat_stream(&ChatRequest::new("mistral-small-latest").user("q"))
        .await
        .unwrap();
    let body: Value = serde_json::from_str(transport.requests()[0].body_text().unwrap()).unwrap();
    assert!(body.get("stream_options").is_none(), "{body}");
    assert_eq!(
        mistral.chat_capability("mistral-small-latest", ChatCapability::StreamUsage),
        CapabilitySupport::Unknown
    );
}

/// Mistral's chat API documents
/// `reasoning_effort`, `top_k` and `prompt_cache_key`, and no
/// `prompt_cache_retention` (<https://docs.mistral.ai/api/endpoint/chat>).
/// The preset sends the three, with no warning, and drops only a
/// retention. Pinned offline: no live Mistral call has checked it.
#[tokio::test]
async fn mistral_sends_reasoning_effort_top_k_and_the_cache_key() {
    let transport = ReplayTransport::new([ok_body(), ok_body(), ok_body()]);
    let mistral = Mistral::builder("k")
        .unwrap()
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let request = ChatRequest::new("mistral-medium-3-5")
        .user("q")
        .reasoning(ReasoningEffort::High)
        .top_k(40)
        .cache(PromptCache::new().key("tenant-1"))
        .unsupported_policy(UnsupportedPolicy::Error);
    let response = mistral.chat(&request).await.unwrap();
    assert!(response.warnings.is_empty(), "{:?}", response.warnings);
    let body: Value = serde_json::from_str(transport.requests()[0].body_text().unwrap()).unwrap();
    assert_eq!(body["reasoning_effort"], "high");
    assert_eq!(body["top_k"], 40);
    assert_eq!(body["prompt_cache_key"], "tenant-1");
    assert!(body.get("prompt_cache_retention").is_none(), "{body}");
    // Which models reason is model data: the effort is sent, and the
    // model decides.
    assert_eq!(
        mistral.chat_capability("mistral-medium-3-5", ChatCapability::ReasoningConfig),
        CapabilitySupport::Unknown
    );

    // A retention is not in the schema: the key is still sent.
    let request = ChatRequest::new("mistral-small-latest").user("q").cache(
        PromptCache::new()
            .key("tenant-1")
            .retention(CacheRetention::Long),
    );
    let response = mistral.chat(&request).await.unwrap();
    let body: Value = serde_json::from_str(transport.requests()[1].body_text().unwrap()).unwrap();
    assert_eq!(body["prompt_cache_key"], "tenant-1");
    assert!(body.get("prompt_cache_retention").is_none(), "{body}");
    let [warning] = response.warnings.as_slice() else {
        panic!("one warning: {:?}", response.warnings);
    };
    assert_eq!(warning.kind, WarningKind::Unsupported);
    assert_eq!(warning.code.as_deref(), Some("unsupported.cache"));
    assert_eq!(
        warning.message,
        "this endpoint takes no prompt cache retention; the retention was not sent"
    );

    // `zai-glm-5-3` takes `low`, `high` and `max`, "not `none`"
    // (https://docs.mistral.ai/capabilities/reasoning).
    let request = ChatRequest::new("zai-glm-5-3")
        .user("q")
        .reasoning(ReasoningEffort::None);
    let response = mistral.chat(&request).await.unwrap();
    let body: Value = serde_json::from_str(transport.requests()[2].body_text().unwrap()).unwrap();
    assert_eq!(body["reasoning_effort"], "low");
    assert_eq!(response.warnings.len(), 1, "{:?}", response.warnings);
    assert_eq!(response.warnings[0].kind, WarningKind::Approximated);
}

/// The cache key and the retention are separate quirks: an endpoint with
/// the retention only sends it and drops the key.
#[tokio::test]
async fn the_cache_key_and_retention_are_separate_quirks() {
    let mut quirks = CompatQuirks::new(NAMESPACE);
    quirks.supports_prompt_cache_retention = true;
    let (provider, transport) = provider(quirks);
    let request = ChatRequest::new("m").user("q").cache(
        PromptCache::new()
            .key("tenant-1")
            .retention(CacheRetention::Long),
    );
    let response = provider.chat(&request).await.unwrap();
    let body: Value = serde_json::from_str(transport.requests()[0].body_text().unwrap()).unwrap();
    assert_eq!(body["prompt_cache_retention"], "24h");
    assert!(body.get("prompt_cache_key").is_none(), "{body}");
    assert_eq!(
        response.warnings[0].message,
        "this endpoint takes no prompt cache key; the key was not sent"
    );
}

/// JSON options under a preset's key, its
/// provider name, are not its options (namespace `openai-compat`): chat,
/// embedding and Responses requests report them dropped instead of
/// ignoring them.
#[test]
fn options_under_the_provider_name_are_reported_dropped() {
    use allama::{EmbeddingProvider, EmbeddingRequest, Subject};
    use allama_openai_compat::ResponsesProvider;
    use allama_openai_compat::providers::Groq;

    let code = Some("dropped.provider_name_options");
    let groq = Groq::builder("k")
        .unwrap()
        .http_transport(ReplayTransport::default())
        .build()
        .unwrap();
    let request = ChatRequest::new("m")
        .user("q")
        .with_options_json("groq", json!({"service_tier": "flex"}));
    let validation = groq.validate_chat_request(&request).unwrap();
    let [warning] = validation.warnings() else {
        panic!("one warning: {:?}", validation.warnings());
    };
    assert_eq!(warning.kind, WarningKind::Dropped);
    assert_eq!(warning.code.as_deref(), code);
    assert_eq!(
        warning.subject,
        Subject::Option {
            namespace: "groq".into()
        }
    );
    assert_eq!(
        warning.message,
        "this provider reads options under `openai-compat`, not under its name `groq`; they were dropped; pass them under `openai-compat`, or as `ChatRequestOptions`"
    );
    // The namespace it reads: no warning.
    let read = ChatRequest::new("m")
        .user("q")
        .with_options_json("openai-compat", json!({"service_tier": "flex"}));
    let validation = groq.validate_chat_request(&read).unwrap();
    assert!(
        validation.warnings().is_empty(),
        "{:?}",
        validation.warnings()
    );

    let embedding = EmbeddingRequest::new("m")
        .inputs(["a"])
        .with_options_json("groq", json!({"user": "u"}));
    let validation = groq.validate_embedding_request(&embedding).unwrap();
    assert_eq!(
        validation.warnings().len(),
        1,
        "{:?}",
        validation.warnings()
    );
    assert_eq!(validation.warnings()[0].code.as_deref(), code);

    // A generic provider named for its options namespace reads them.
    let generic = Provider::builder()
        .base_url("http://upstream/v1")
        .http_transport(ReplayTransport::default())
        .build()
        .unwrap();
    let validation = generic.validate_chat_request(&read).unwrap();
    assert!(
        validation.warnings().is_empty(),
        "{:?}",
        validation.warnings()
    );

    let responses = ResponsesProvider::builder()
        .base_url("http://upstream/v1")
        .provider_name("vllm")
        .http_transport(ReplayTransport::default())
        .build()
        .unwrap();
    let request = ChatRequest::new("m")
        .user("q")
        .with_options_json("vllm", json!({"truncation": "auto"}));
    let validation = responses.validate_chat_request(&request).unwrap();
    let [warning] = validation.warnings() else {
        panic!("one warning: {:?}", validation.warnings());
    };
    assert_eq!(warning.code.as_deref(), code);
    assert!(
        warning
            .message
            .contains("pass them under `openai-compat-responses`, or as `ResponsesRequestOptions`"),
        "{warning:?}"
    );
}
