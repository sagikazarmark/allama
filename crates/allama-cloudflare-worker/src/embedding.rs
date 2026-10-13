//! `EmbeddingProvider` implementation for Cloudflare Workers AI.
//!
//! Calls the AI binding's `run()` for text-embedding models such as
//! `@cf/baai/bge-base-en-v1.5`. Wire types mirror the native Workers AI
//! embedding schema: `{ "text": [...] }` in, `{ "shape": [N, D], "data": [...] }`
//! out.

use allama::adapter::Degradations;
use allama::adapter::embedding::Prepared;
use allama::{
    CapabilitySupport, EmbeddingCapability, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse,
    EmbeddingValidation, Error, RequestField, Result, ValidationCoverage,
};
use serde::{Deserialize, Serialize};

use crate::binding::run_json;
use crate::options::refuse_owned_fields;
use crate::{EmbeddingRequestOptions, Provider};
use crate::{NAMESPACE, PROVIDER_NAME};

impl EmbeddingProvider for Provider {
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        let run_options = request
            .resolve_option::<EmbeddingRequestOptions>()?
            .run_options;
        let mut prepared = allama::adapter::embedding::prepare(self, request)?;
        let body = encode(&mut prepared)?;
        let model = prepared.request().model.clone();
        prepared.mark_sent();
        let response: EmbedResponse = run_json(&self.ai, &model, &body, &run_options).await?;
        decode(prepared, response)
    }

    /// A dry run of `embed`: the preflight and encoding, without running
    /// the model ([`ValidationCoverage::Encoding`]).
    fn validate_embedding_request(
        &self,
        request: &EmbeddingRequest,
    ) -> Result<EmbeddingValidation> {
        validate(self, request)
    }

    fn embedding_capability(
        &self,
        _model: &str,
        capability: EmbeddingCapability,
    ) -> CapabilitySupport {
        builtin_embedding_capability(capability)
    }
}

/// Model-independent embedding capability answers for the binding adapter.
fn builtin_embedding_capability(capability: EmbeddingCapability) -> CapabilitySupport {
    match capability {
        EmbeddingCapability::BatchInput => CapabilitySupport::Supported,
        // The native embedding schema is `{ "text": [...] }`.
        EmbeddingCapability::OutputDimensions
        | EmbeddingCapability::ImageInput
        | EmbeddingCapability::CompositeInput
        | EmbeddingCapability::InputType => CapabilitySupport::Unsupported,
        _ => CapabilitySupport::Unknown,
    }
}

/// Everything before the binding runs, shared by the call and its dry run:
/// the body the binding receives.
fn encode(prepared: &mut Prepared<'_>) -> Result<EmbedRequest> {
    let (request, degradations) = prepared.parts_mut();
    crate::options::report_options_under_name(
        &request.options,
        "EmbeddingRequestOptions",
        degradations,
    );
    EmbedRequest::try_from(prepared.request())
}

/// The binding's output as the response to `prepared`, checked
/// ([`allama::adapter::embedding::check_response`]), with its warnings.
fn decode(prepared: Prepared<'_>, response: EmbedResponse) -> Result<EmbeddingResponse> {
    let request = prepared.request();
    let response = response.into_response(request.inputs.len())?;
    allama::adapter::embedding::check_response(request, &response)
        .map_err(|err| err.with_provider(PROVIDER_NAME))?;
    Ok(prepared.finish(response))
}

/// A dry run of `embed`.
fn validate<P: EmbeddingProvider + ?Sized>(
    provider: &P,
    request: &EmbeddingRequest,
) -> Result<EmbeddingValidation> {
    let mut prepared = allama::adapter::embedding::prepare_dry_run(provider, request)?;
    encode(&mut prepared)?;
    Ok(prepared.into_validation(ValidationCoverage::Encoding))
}

/// Workers AI embedding request body.
#[derive(Debug, Serialize)]
pub(crate) struct EmbedRequest {
    pub text: Vec<String>,
    /// `EmbeddingRequestOptions::extra`, checked against
    /// [`EMBED_INPUT_FIELDS`].
    #[serde(flatten)]
    pub extra: allama::ExtraMap,
}

/// The input fields of [`EmbedRequest`]: the adapter owns them, so
/// `EmbeddingRequestOptions::extra` cannot set them.
pub(crate) const EMBED_INPUT_FIELDS: &[&str] = &["text"];

/// Workers AI embedding response body.
///
/// The native API returns `{ "shape": [batch, dimensions], "data": [...] }`.
#[derive(Debug, Deserialize)]
pub(crate) struct EmbedResponse {
    pub data: Vec<Vec<f32>>,
}

impl TryFrom<&EmbeddingRequest> for EmbedRequest {
    type Error = Error;

    fn try_from(request: &EmbeddingRequest) -> Result<Self> {
        let degradations = Degradations::new(
            PROVIDER_NAME,
            request.unsupported_policy.unwrap_or_default(),
        );
        if request.dimensions.is_some() {
            return Err(degradations.unsupported(
                RequestField::Dimensions,
                "Workers AI embedding models do not support output dimension selection",
            ));
        }
        let options = request
            .resolve_option::<EmbeddingRequestOptions>()
            .map_err(|err| err.with_provider(PROVIDER_NAME))?;
        refuse_owned_fields(
            &options.extra,
            EMBED_INPUT_FIELDS,
            "EmbeddingRequestOptions::extra",
        )?;
        let text = request.text_inputs(PROVIDER_NAME)?;
        // Workers AI embedding models read no per-input settings.
        request.check_provider_data(PROVIDER_NAME, NAMESPACE, &[])?;
        Ok(Self {
            text: text.into_iter().map(str::to_owned).collect(),
            extra: options.extra,
        })
    }
}

impl EmbedResponse {
    fn into_response(self, expected: usize) -> Result<EmbeddingResponse> {
        if self.data.len() != expected {
            return Err(Error::unexpected_response(format!(
                "Workers AI returned {} embeddings for {expected} inputs",
                self.data.len()
            ))
            .authored()
            .with_provider(PROVIDER_NAME));
        }
        Ok(EmbeddingResponse::new(self.data))
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use allama::ProviderIdentity;
    use allama_conformance::probe::ProbeOutcome;

    use super::*;

    /// Stands in for [`Provider`], whose binding exists only in the Workers
    /// runtime: the same capability answers, call path and validation, with
    /// the model's output canned (one `[0.1, 0.2]` per input unless `answer`
    /// says otherwise) and the bodies the binding would receive captured.
    #[derive(Default)]
    pub(crate) struct Binding {
        sent: std::sync::Mutex<Vec<serde_json::Value>>,
        answer: Option<Vec<Vec<f32>>>,
    }

    impl Binding {
        /// A binding whose model outputs `answer`.
        pub(crate) fn answering(answer: Vec<Vec<f32>>) -> Self {
            Self {
                answer: Some(answer),
                ..Self::default()
            }
        }

        /// The bodies the binding received, in order.
        pub(crate) fn sent(&self) -> Vec<serde_json::Value> {
            self.sent.lock().unwrap().clone()
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

    impl EmbeddingProvider for Binding {
        async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
            let mut prepared = allama::adapter::embedding::prepare(self, request)?;
            let body = encode(&mut prepared)?;
            prepared.mark_sent();
            let captured = serde_json::to_value(&body).expect("the body serializes");
            self.sent.lock().unwrap().push(captured);
            let data = self
                .answer
                .clone()
                .unwrap_or_else(|| vec![vec![0.1, 0.2]; body.text.len()]);
            decode(prepared, EmbedResponse { data })
        }

        fn embedding_capability(
            &self,
            _model: &str,
            capability: EmbeddingCapability,
        ) -> CapabilitySupport {
            builtin_embedding_capability(capability)
        }

        fn validate_embedding_request(
            &self,
            request: &EmbeddingRequest,
        ) -> Result<EmbeddingValidation> {
            validate(self, request)
        }
    }

    /// What `Provider::embed` sends to the binding, answered with one vector
    /// per input.
    pub(crate) async fn probe(request: EmbeddingRequest) -> ProbeOutcome<EmbeddingResponse> {
        let binding = Binding::default();
        let result = binding.embed(&request).await;
        ProbeOutcome::new(result, binding.sent())
    }

    #[tokio::test]
    async fn extra_fields_reach_the_input() {
        let request = EmbeddingRequest::new("@cf/baai/bge-base-en-v1.5")
            .input("Hi")
            .with_option(EmbeddingRequestOptions::new().extra("pooling", "cls"));
        let binding = Binding::default();
        binding.embed(&request).await.unwrap();
        assert_eq!(
            binding.sent(),
            [serde_json::json!({"text": ["Hi"], "pooling": "cls"})]
        );

        let request = EmbeddingRequest::new("@cf/baai/bge-base-en-v1.5")
            .input("Hi")
            .with_option(EmbeddingRequestOptions::new().extra("TEXT", ["x"]));
        let err = binding.embed(&request).await.unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
    }

    /// Every embedding contract, through the binding stand-in. The binding
    /// has no HTTP response or headers, so the error mapping and transport
    /// contracts do not apply.
    #[tokio::test]
    async fn the_adapter_passes_every_contract() {
        use allama_conformance::capabilities::UNKNOWN_MODEL;
        use allama_conformance::probe::VectorOrder;
        use allama_conformance::{Contract, EmbeddingAdapterUnderTest, assert_embedding_adapter};

        const MODELS: &[&str] = &["@cf/baai/bge-base-en-v1.5", "@cf/baai/bge-m3"];
        for model in MODELS.iter().chain(&[UNKNOWN_MODEL]) {
            let validation = Binding::default()
                .validate_embedding_request(&EmbeddingRequest::new(*model).input("Hi"))
                .unwrap();
            assert_eq!(validation.coverage(), ValidationCoverage::Encoding);
        }
        // A batch over a model's own limit (100 texts on the BGE English
        // v1.5 models) is sent, and the model decides (ADR 0015).
        let over_the_limit = MODELS.iter().map(|model| {
            let inputs: Vec<String> = (0..=100).map(|i| i.to_string()).collect();
            EmbeddingRequest::new(*model).inputs(inputs)
        });
        let under_test =
            EmbeddingAdapterUnderTest::new(MODELS.iter().copied(), crate::NAMESPACE, probe)
                .validation_requests(over_the_limit)
                .answering(VectorOrder::InputOrder, |request, vectors| async move {
                    // The binding decodes JSON numbers into `f32` as serde does.
                    #[allow(clippy::cast_possible_truncation)]
                    let answer = vectors
                        .iter()
                        .map(|vector| vector.values.iter().map(|&value| value as f32).collect())
                        .collect();
                    Binding::answering(answer).embed(&request).await
                })
                .skip([
                    // No model has a batch limit without a catalog (ADR 0015).
                    Contract::BatchLimitEnforced,
                    Contract::ErrorKindsMapped,
                    Contract::ErrorLogExcludesBody,
                    Contract::RetrySignalsSurfaced,
                    Contract::TransportOptionsApplied,
                ]);
        assert_embedding_adapter(&Binding::default(), &under_test).await;
    }

    /// The response names the provider that served it, as `decode`
    /// finishes it for the real binding too.
    #[tokio::test]
    async fn the_response_names_the_provider() {
        let request = EmbeddingRequest::new("@cf/baai/bge-base-en-v1.5").input("a");
        let response = probe(request).await.result.unwrap();
        assert_eq!(response.provider.as_deref(), Some(PROVIDER_NAME));
    }

    /// The binding takes no input type: the warning names the prefix the
    /// model's card documents, or says the model may expect one (BGE-M3
    /// needs none, <https://huggingface.co/BAAI/bge-m3>).
    #[tokio::test]
    async fn input_type_warnings_name_the_model_prefix() {
        use allama::EmbeddingInputType;

        for (model, input_type, message) in [
            (
                "@cf/baai/bge-base-en-v1.5",
                EmbeddingInputType::Query,
                "`@cf/baai/bge-base-en-v1.5` takes the role as a text prefix, not a field; the \
                 `query` input type was not sent; start each query with `Represent this \
                 sentence for searching relevant passages: `",
            ),
            (
                "@cf/qwen/qwen3-embedding-0.6b",
                EmbeddingInputType::Document,
                "`@cf/qwen/qwen3-embedding-0.6b` takes the role as a text prefix, not a field; \
                 the `document` input type was not sent; documents take no prefix (queries \
                 start with `Instruct: {task}\\nQuery: `)",
            ),
            (
                "@cf/baai/bge-m3",
                EmbeddingInputType::Query,
                "`cloudflare` takes no query/document input type (`@cf/baai/bge-m3` may expect a \
                 role prefix in the text); the `query` input type was not sent",
            ),
        ] {
            let request = EmbeddingRequest::new(model)
                .input("a")
                .input_type(input_type);
            let outcome = probe(request).await;
            let warnings = outcome.result.unwrap().warnings;
            assert_eq!(warnings.len(), 1, "{warnings:?}");
            assert_eq!(warnings[0].message, message);
        }
    }

    #[test]
    fn request_conversion_forwards_inputs() {
        let request = EmbeddingRequest::new("@cf/baai/bge-base-en-v1.5")
            .input("hello")
            .input("world");

        let cf = EmbedRequest::try_from(&request).unwrap();
        assert_eq!(cf.text, vec!["hello".to_string(), "world".to_string()]);
    }

    #[test]
    fn request_rejects_dimension_override_even_when_lenient() {
        let request = EmbeddingRequest::new("@cf/baai/bge-base-en-v1.5")
            .input("hi")
            .dimensions(256);
        let err = EmbedRequest::try_from(&request).unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }));
        assert!(err.to_string().contains("dimensions"));
        assert_eq!(
            err.context().unwrap().provider.as_deref(),
            Some("cloudflare")
        );
    }

    #[test]
    fn request_rejects_empty_inputs() {
        let request = EmbeddingRequest::new("@cf/baai/bge-base-en-v1.5");
        let err = EmbedRequest::try_from(&request).unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }));
    }

    #[test]
    fn request_rejects_inputs_that_are_not_one_text() {
        let request = EmbeddingRequest::new("@cf/baai/bge-base-en-v1.5").input(vec![
            allama::EmbeddingPart::text("a"),
            allama::EmbeddingPart::text("b"),
        ]);
        let err = EmbedRequest::try_from(&request).unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }), "{err}");
        assert_eq!(
            err.context().unwrap().provider.as_deref(),
            Some("cloudflare")
        );
    }

    #[test]
    fn request_rejects_own_namespace_input_data_and_ignores_others() {
        use allama::{EmbeddingInput, ProviderData, Text};
        let data =
            |namespace: &str| ProviderData::new().with_raw(namespace, "k", serde_json::json!(1));
        let own = EmbeddingRequest::new("@cf/baai/bge-base-en-v1.5")
            .input(EmbeddingInput::text("a").with_provider_data(data(NAMESPACE)));
        let err = EmbedRequest::try_from(&own).unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err}");

        let tolerated = EmbeddingRequest::new("@cf/baai/bge-base-en-v1.5")
            .input(EmbeddingInput::text("a").with_provider_data(data("openai")))
            .input(EmbeddingInput::from(
                Text::new("b").with_provider_data(data(NAMESPACE)),
            ));
        let cf = EmbedRequest::try_from(&tolerated).unwrap();
        assert_eq!(cf.text, vec!["a".to_string(), "b".to_string()]);
    }

    /// No batch is refused on the model id (ADR 0015): the BGE English
    /// v1.5 models document 100 texts per call, and a larger batch is sent
    /// for the model to refuse. `embedding_limits` claims none.
    #[test]
    fn batches_are_sent_whatever_the_model() {
        let inputs: Vec<String> = (0..=100).map(|i| i.to_string()).collect();
        for model in ["@cf/baai/bge-base-en-v1.5", "@cf/baai/bge-m3"] {
            let request = EmbeddingRequest::new(model).inputs(inputs.clone());
            assert_eq!(EmbedRequest::try_from(&request).unwrap().text.len(), 101);
            assert_eq!(
                Binding::default().embedding_limits(model),
                allama::EmbeddingLimits::new()
            );
        }
    }

    #[test]
    fn response_conversion_preserves_vectors() {
        let wire: EmbedResponse = serde_json::from_value(serde_json::json!({
            "shape": [2, 3],
            "data": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "pooling": "mean"
        }))
        .unwrap();
        let response = wire.into_response(2).unwrap();
        assert_eq!(response.embeddings.len(), 2);
        assert_eq!(response.embeddings[0], vec![0.1, 0.2, 0.3]);
        assert_eq!(response.embeddings[1], vec![0.4, 0.5, 0.6]);
        assert!(response.warnings.is_empty());
    }

    #[test]
    fn response_count_mismatch_is_unexpected() {
        let wire = EmbedResponse {
            data: vec![vec![0.1]],
        };
        let err = wire.into_response(2).unwrap_err();
        assert!(matches!(err, Error::UnexpectedResponse { .. }));
    }
}
