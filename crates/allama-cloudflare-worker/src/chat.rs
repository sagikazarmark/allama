//! `ChatProvider` implementation for Cloudflare Workers AI via the AI binding.

use allama::adapter::chat::Prepared;
use allama::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, ChatStream,
    ChatValidation, ProviderIdentity, Result, ValidationCoverage,
};

use crate::binding::{run_stream, run_value};
use crate::streaming::byte_stream_to_chat_stream;
use crate::wire;
use crate::{PROVIDER_NAME, Provider};

impl ProviderIdentity for Provider {
    fn provider_name(&self) -> &str {
        PROVIDER_NAME
    }

    fn reads_namespace(&self, namespace: &str) -> bool {
        namespace == crate::NAMESPACE
    }

    fn downcast_any(&self, id: std::any::TypeId) -> Option<&(dyn std::any::Any + Send + Sync)> {
        allama::expose(self, id)
    }
}

impl ChatProvider for Provider {
    type Stream = ChatStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let options = request.resolve_option::<crate::ChatRequestOptions>()?;
        let mut prepared = allama::adapter::chat::prepare(self, request)?;
        let body = encode(self, &mut prepared, false)?;
        let (model, messages) = (
            prepared.request().model.clone(),
            prepared.request().messages.len(),
        );
        prepared.mark_sent();
        let output = run_value(&self.ai, &model, &body, &options.run_options).await?;
        let raw = options.raw_response.then(|| output.clone());
        let mut response = wire::decode_response(output, messages)?;
        if let Some(raw) = raw {
            response.provider_data.insert(&crate::RawResponse(raw))?;
        }
        Ok(prepared.finish(response))
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        let run_options = request
            .resolve_option::<crate::ChatRequestOptions>()?
            .run_options;
        let mut prepared = allama::adapter::chat::prepare(self, request)?;
        let body = encode(self, &mut prepared, true)?;
        let model = prepared.request().model.clone();
        prepared.mark_sent();
        let byte_stream = run_stream(&self.ai, &model, &body, &run_options).await?;
        Ok(prepared.finish_stream(byte_stream_to_chat_stream(
            byte_stream,
            Vec::new(),
            self.max_stream_event_size,
        )))
    }

    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        self.chat_capability_resolver
            .as_ref()
            .and_then(|resolver| resolver.chat_capability(model, capability))
            .unwrap_or_else(|| builtin_chat_capability(model, capability))
    }

    /// A dry run of `chat`: the preflight, options and encoding, without
    /// running the model ([`ValidationCoverage::Encoding`]).
    fn validate_chat_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        validate(self, request, false)
    }

    /// A dry run of `chat_stream` ([`ValidationCoverage::Encoding`]).
    fn validate_chat_stream_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        validate(self, request, true)
    }
}

/// Everything before the binding runs, shared by the call and its dry run:
/// the body the binding receives (the adapter's wire request), with the
/// encoder's warnings reported through `prepared`.
fn encode<P: ChatProvider + ?Sized>(
    provider: &P,
    prepared: &mut Prepared<'_>,
    stream: bool,
) -> Result<wire::ChatRequest> {
    if stream {
        prepared.preflight_stream(provider);
    }
    let (request, degradations) = prepared.parts_mut();
    let (body, mapping_warnings) = wire::encode_request(request, stream)?;
    // Into the same helper, so warnings name the caller's blocks.
    degradations.extend(mapping_warnings);
    Ok(body)
}

/// A dry run of `chat` or `chat_stream`.
fn validate<P: ChatProvider + ?Sized>(
    provider: &P,
    request: &ChatRequest,
    stream: bool,
) -> Result<ChatValidation> {
    if !stream {
        request.resolve_option::<crate::ChatRequestOptions>()?;
    }
    let mut prepared = allama::adapter::chat::prepare_dry_run(provider, request)?;
    encode(provider, &mut prepared, stream)?;
    Ok(prepared.into_validation(ValidationCoverage::Encoding))
}

/// Capability answers for the binding adapter: what the adapter cannot
/// encode is `Unsupported` for every model, streaming is `Supported`, and
/// the rest is `Unknown`, for the model to decide.
///
/// Which models call tools or take JSON Mode is model data, which belongs
/// in a catalog, not in the adapter (ADR 0015): both answer `Unknown`, as on
/// `allama-openai-compat`'s Cloudflare preset, so the two transports agree.
/// Install a resolver with
/// [`Provider::with_chat_capabilities`] to claim them for the models you
/// use. Each model's input schema says what it takes: vision models take
/// image parts, reasoning models a `reasoning_effort`. They are not refused
/// for other models; a model that does not take them answers with an
/// error.
fn builtin_chat_capability(_model: &str, capability: ChatCapability) -> CapabilitySupport {
    match capability {
        ChatCapability::Streaming | ChatCapability::NativeStreaming => CapabilitySupport::Supported,
        // The adapter sends no files, assistant images or replayed
        // reasoning, and no way to require a tool call.
        ChatCapability::ImageOutput
        | ChatCapability::ImageReplay
        | ChatCapability::FileInput
        | ChatCapability::AudioInput
        | ChatCapability::VideoInput
        | ChatCapability::ReasoningReplay
        | ChatCapability::ForcedToolChoice => CapabilitySupport::Unsupported,
        // Image parts (with their detail) and the reasoning effort are
        // sent; the model decides. Several tool calls in one turn are
        // decoded, though no parallel-call control is sent.
        ChatCapability::ParallelToolCalls
        | ChatCapability::ImageInput
        | ChatCapability::ImageDetail
        | ChatCapability::ReasoningConfig => CapabilitySupport::Unknown,
        // Current runtimes end a stream with a usage chunk
        // (`fixtures/stream.sse`), but older runtimes and some models end
        // without one (`fixtures/stream_legacy.sse`), and the binding's docs
        // do not promise it.
        ChatCapability::StreamUsage => CapabilitySupport::Unknown,
        // Reasoning output depends on the model (e.g. Qwen3 returns
        // `reasoning_content`); prefix caching is automatic for select models.
        _ => CapabilitySupport::Unknown,
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) mod tests {
    use allama::{ChatStreamExt, ContentBlock};
    use allama_conformance::capabilities::assert_stream_usage_matches_capability;

    use allama_conformance::probe::ProbeOutcome;

    use super::*;

    /// Stands in for [`Provider`], whose binding exists only in the Workers
    /// runtime: the same capability answers, call path and validation, with
    /// the model's answer canned ("ok", or a recorded stream) and the bodies
    /// the binding would receive captured.
    #[derive(Default)]
    pub(crate) struct Binding {
        sent: std::sync::Mutex<Vec<serde_json::Value>>,
        /// The stream the model answers with, instead of the recording.
        stream: Option<String>,
    }

    impl Binding {
        /// A binding whose model streams `sse`.
        pub(crate) fn streaming(sse: impl Into<String>) -> Self {
            Self {
                stream: Some(sse.into()),
                ..Self::default()
            }
        }

        /// The bodies the binding received, in order.
        pub(crate) fn sent(&self) -> Vec<serde_json::Value> {
            self.sent.lock().unwrap().clone()
        }

        fn capture(&self, body: &wire::ChatRequest) {
            let body = serde_json::to_value(body).expect("the body serializes");
            self.sent.lock().unwrap().push(body);
        }
    }

    impl ProviderIdentity for Binding {
        fn provider_name(&self) -> &str {
            PROVIDER_NAME
        }

        fn reads_namespace(&self, namespace: &str) -> bool {
            namespace == crate::NAMESPACE
        }
    }

    impl ChatProvider for Binding {
        type Stream = ChatStream;

        async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
            request.resolve_option::<crate::ChatRequestOptions>()?;
            let mut prepared = allama::adapter::chat::prepare(self, request)?;
            let body = encode(self, &mut prepared, false)?;
            prepared.mark_sent();
            self.capture(&body);
            Ok(prepared.finish(ChatResponse::new(vec![ContentBlock::text("ok")])))
        }

        async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
            let mut prepared = allama::adapter::chat::prepare(self, request)?;
            let body = encode(self, &mut prepared, true)?;
            prepared.mark_sent();
            self.capture(&body);
            let sse = self
                .stream
                .as_deref()
                .unwrap_or(include_str!("../fixtures/stream.sse"));
            Ok(prepared.finish_stream(crate::streaming::stream_from_sse_text(sse, Vec::new())))
        }

        fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
            builtin_chat_capability(model, capability)
        }

        fn validate_chat_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
            validate(self, request, false)
        }

        fn validate_chat_stream_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
            validate(self, request, true)
        }
    }

    /// What `Provider::chat` sends to the binding, answered with "ok".
    pub(crate) async fn probe(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
        let binding = Binding::default();
        let result = binding.chat(&request).await;
        ProbeOutcome::new(result, binding.sent())
    }

    /// What `Provider::chat_stream` sends to the binding, answered with a
    /// recorded stream.
    pub(crate) async fn stream_probe(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
        let binding = Binding::default();
        let result = match binding.chat_stream(&request).await {
            Ok(stream) => stream.collect_response().await,
            Err(err) => Err(err),
        };
        ProbeOutcome::new(result, binding.sent())
    }

    /// `StreamUsage` is `Unknown`: a current stream ends with a usage
    /// chunk, but the legacy shape ends without one, so neither answer
    /// holds for every model. The contract checks nothing for `Unknown`,
    /// so the two streams are checked here.
    #[tokio::test]
    async fn stream_usage_is_unknown() {
        assert_eq!(
            Binding::default().chat_capability(
                "@cf/meta/llama-3.1-8b-instruct",
                ChatCapability::StreamUsage
            ),
            CapabilitySupport::Unknown
        );
        assert_stream_usage_matches_capability(
            &Binding::default(),
            "@cf/meta/llama-3.1-8b-instruct",
            stream_probe,
        )
        .await;
        let usage = |sse: &'static str| async move {
            crate::streaming::stream_from_sse_text(sse, Vec::new())
                .collect_response()
                .await
                .unwrap()
                .usage
        };
        assert!(
            usage(include_str!("../fixtures/stream.sse"))
                .await
                .is_some()
        );
        assert!(
            usage(include_str!("../fixtures/stream_legacy.sse"))
                .await
                .is_none()
        );
    }

    /// Workers decodes several tool calls in one turn, and `true` is
    /// permission: the capability is `Unknown`, and a chat with `true`
    /// warns nothing.
    #[tokio::test]
    async fn parallel_tool_calls_are_unknown_and_true_warns_nothing() {
        let model = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
        assert_eq!(
            builtin_chat_capability(model, ChatCapability::ParallelToolCalls),
            CapabilitySupport::Unknown
        );
        let request = ChatRequest::new(model)
            .user("Look up rust.")
            .tools([allama::ToolDefinition::new(
                "lookup",
                serde_json::json!({"type": "object", "properties": {}}),
            )])
            .parallel_tool_calls(true)
            .unsupported_policy(allama::UnsupportedPolicy::Error);
        let binding = Binding::default();
        let response = binding.chat(&request).await.unwrap();
        assert!(response.warnings.is_empty(), "{:?}", response.warnings);
        assert!(binding.sent()[0].get("parallel_tool_calls").is_none());
    }

    #[test]
    fn provider_identity_is_cloudflare() {
        // The wire identity is part of the public contract: dashboards and
        // OTEL backends keyed on `gen_ai.provider.name` break if it shifts.
        assert_eq!(PROVIDER_NAME, "cloudflare");
    }

    /// Tool calls and structured output are model data, `Unknown`
    /// for every model, as on the compat crate's Cloudflare preset.
    #[test]
    fn builtin_capabilities() {
        let answer = builtin_chat_capability;
        for model in [
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            "@cf/acme/new-model",
        ] {
            for capability in [ChatCapability::ToolCalls, ChatCapability::StructuredOutput] {
                assert_eq!(answer(model, capability), CapabilitySupport::Unknown);
            }
            assert_eq!(
                answer(model, ChatCapability::FileInput),
                CapabilitySupport::Unsupported
            );
            assert_eq!(
                answer(model, ChatCapability::ReasoningOutput),
                CapabilitySupport::Unknown
            );
        }
    }

    /// What an extraction plan reports through Workers AI, for the typical
    /// schemas in each mode, without sending anything.
    #[test]
    fn extraction_plans() {
        let binding = Binding::default();
        // JSON Mode cannot guarantee a schema, and the binding takes no
        // forced tool choice.
        let expected = [
            "Native Person: not_enforced",
            "Native Order: not_enforced",
            "Native Tagged: not_enforced",
            "ForcedTool Person: refused tool_choice",
            "ForcedTool Order: refused tool_choice",
            "ForcedTool Tagged: refused tool_choice",
        ];
        assert_eq!(
            typical::plans(&binding, "@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
            expected
        );
        assert!(binding.sent().is_empty());
    }

    // The fixture types are only planned for, never built.
    #[allow(dead_code)]
    mod typical {
        use std::collections::HashMap;

        use allama::chat::extract::{ChatExtractionMode, ChatExtractor};
        use allama::schemars::JsonSchema;
        use allama::{ChatProvider, ChatRequest, ValidationCoverage};
        use serde::Deserialize;

        /// Strings, one of them optional.
        #[derive(Deserialize, JsonSchema)]
        #[schemars(crate = "allama::schemars")]
        pub struct Person {
            name: String,
            nickname: Option<String>,
        }

        /// An unsigned integer (schemars writes `minimum: 0`).
        #[derive(Deserialize, JsonSchema)]
        #[schemars(crate = "allama::schemars")]
        pub struct Order {
            id: String,
            quantity: u32,
        }

        /// A map.
        #[derive(Deserialize, JsonSchema)]
        #[schemars(crate = "allama::schemars")]
        pub struct Tagged {
            title: String,
            labels: HashMap<String, String>,
        }

        /// The plan of each typical schema in each explicit mode, as
        /// `"<mode> <type>: <enforcement>"`, or `refused <field>` when the plan
        /// fails before sending. Every plan is a dry run (`Encoding`).
        pub fn plans<P: ChatProvider + ?Sized>(provider: &P, model: &str) -> Vec<String> {
            let mut rows = Vec::new();
            for mode in [ChatExtractionMode::Native, ChatExtractionMode::ForcedTool] {
                let request = ChatRequest::new(model).user("x").with_option(mode);
                let extractor = ChatExtractor::new();
                for (name, plan) in [
                    ("Person", extractor.plan::<Person>(provider, &request)),
                    ("Order", extractor.plan::<Order>(provider, &request)),
                    ("Tagged", extractor.plan::<Tagged>(provider, &request)),
                ] {
                    let cell = match plan {
                        Ok(plan) => {
                            assert_eq!(plan.coverage, ValidationCoverage::Encoding, "{name}");
                            plan.enforcement.to_string()
                        }
                        Err(err) => format!(
                            "refused {}",
                            err.unsupported_field()
                                .map_or("-".to_owned(), |f| f.to_string())
                        ),
                    };
                    rows.push(format!("{mode:?} {name}: {cell}"));
                }
            }
            rows
        }
    }
}
