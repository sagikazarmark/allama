//! Embeddings through `batchEmbedContents`.
//!
//! One request embeds up to [`MAX_INPUTS_PER_CALL`] inputs (the API rejects
//! larger batches: "at most 100 requests can be in one batch", verified live
//! 2026-10-02). Each input becomes one content whose parts Gemini combines
//! into one vector. `gemini-embedding-2` embeds text and images, at most
//! [`MAX_IMAGES_PER_INPUT`] images per input ("at most 6 image parts per input
//! instance", verified live 2026-10-03). Token usage is reported when the API
//! returns it.
//!
//! The portable [`input_type`](EmbeddingRequest::input_type) maps
//! to `taskType` (`RETRIEVAL_QUERY` or `RETRIEVAL_DOCUMENT`), for every
//! model. `gemini-embedding-2` takes no task type ("include the task as an
//! instruction in your prompt"); the adapter never rewrites the text.
//!
//! Nothing is refused or dropped because of the model (ADR 0015): images,
//! composite inputs and the task type are sent to every model, and a model
//! that lacks one refuses upstream.

use allama::adapter::Degradations;
use allama::adapter::embedding::Prepared;
use allama::{
    CapabilitySupport, EmbeddingCapability, EmbeddingInput, EmbeddingInputType, EmbeddingLimits,
    EmbeddingPart, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, EmbeddingValidation,
    Error, ProviderDataEntry, RequestField, Result, Subject, Usage, ValidationCoverage, Warning,
};
use allama_http::HttpRequest;
use serde::{Deserialize, Serialize};

use crate::error::SuccessBody;
use crate::model::resource_path;
use crate::options::refuse_owned_fields;
use crate::request::image_part;
use crate::wire::PartData;
use crate::{EmbeddingRequestOptions, EmbeddingTitle, NAMESPACE, PROVIDER_NAME, Provider};

/// Documented `batchEmbedContents` batch limit.
pub(crate) const MAX_INPUTS_PER_CALL: usize = 100;

/// Image parts `gemini-embedding-2` accepts in one input.
pub(crate) const MAX_IMAGES_PER_INPUT: usize = 6;

impl Provider {
    /// Everything before sending an embedding request, shared by the call
    /// and its dry run: encode the prepared request (reporting through
    /// `prepared`) and build the request to `batchEmbedContents` with the
    /// caller's transport options.
    fn encode_embedding(
        &self,
        prepared: &mut Prepared<'_>,
        options: &EmbeddingRequestOptions,
    ) -> Result<HttpRequest> {
        let (request, degradations) = prepared.parts_mut();
        let (body, mapping_warnings) = to_api_request(request, options)?;
        degradations.extend(mapping_warnings);
        let url = format!(
            "{}/{}:batchEmbedContents",
            self.inner.base_url,
            resource_path(&request.model)?
        );
        self.build_post(url, serde_json::to_string(&body)?, &request.options)
    }
}

impl EmbeddingProvider for Provider {
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        let options = request.resolve_option::<EmbeddingRequestOptions>()?;
        let mut prepared = allama::adapter::embedding::prepare(self, request)?;
        let http = self.encode_embedding(&mut prepared, &options)?;
        prepared.mark_sent();
        let response = self.send(http, false).await?;
        let body: BatchEmbedContentsResponse =
            SuccessBody::read(response, self.inner.max_response_size)
                .await?
                .decode()?;
        let request = prepared.request();
        let response = from_api_response(body, &request.model, request.inputs.len())?;
        allama::adapter::embedding::check_response(request, &response)
            .map_err(|err| err.with_provider(PROVIDER_NAME))?;
        Ok(prepared.finish(response))
    }

    /// A dry run of `embed`: the preflight, options, encoding and transport
    /// options, without sending ([`ValidationCoverage::Encoding`]).
    fn validate_embedding_request(
        &self,
        request: &EmbeddingRequest,
    ) -> Result<EmbeddingValidation> {
        let options = request.resolve_option::<EmbeddingRequestOptions>()?;
        let mut prepared = allama::adapter::embedding::prepare_dry_run(self, request)?;
        self.encode_embedding(&mut prepared, &options)?;
        Ok(prepared.into_validation(ValidationCoverage::Encoding))
    }

    /// Batches and output dimensions are `Supported` for every model.
    /// `gemini-embedding-2` embeds images and combines parts. The input type
    /// (`taskType`) is `Supported` for `gemini-embedding-001` and
    /// `text-embedding-*`. Everything else is `Unknown`: what a model lacks
    /// is model data, not answered `Unsupported` (ADR 0015).
    fn embedding_capability(
        &self,
        model: &str,
        capability: EmbeddingCapability,
    ) -> CapabilitySupport {
        model_capability(model, capability)
    }

    fn embedding_limits(&self, _model: &str) -> EmbeddingLimits {
        EmbeddingLimits::new().max_inputs_per_call(MAX_INPUTS_PER_CALL)
    }
}

fn model_capability(model: &str, capability: EmbeddingCapability) -> CapabilitySupport {
    let id = model.strip_prefix("models/").unwrap_or(model);
    let multimodal = id.starts_with("gemini-embedding-2");
    let task_types = id.starts_with("gemini-embedding-001") || id.starts_with("text-embedding-");
    match capability {
        EmbeddingCapability::BatchInput | EmbeddingCapability::OutputDimensions => {
            CapabilitySupport::Supported
        }
        EmbeddingCapability::ImageInput | EmbeddingCapability::CompositeInput if multimodal => {
            CapabilitySupport::Supported
        }
        EmbeddingCapability::InputType if task_types => CapabilitySupport::Supported,
        _ => CapabilitySupport::Unknown,
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct BatchEmbedContentsRequest {
    pub requests: Vec<EmbedContentRequest>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct EmbedContentRequest {
    pub model: String,
    pub content: EmbedContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dimensionality: Option<u32>,
    /// `EmbeddingRequestOptions::extra`, checked against
    /// [`EMBED_CONTENT_FIELDS`].
    #[serde(flatten)]
    pub extra: allama::ExtraMap,
}

/// The fields of [`EmbedContentRequest`]: the adapter owns them, so
/// `EmbeddingRequestOptions::extra` cannot set them.
pub(crate) const EMBED_CONTENT_FIELDS: &[&str] = &[
    "content",
    "model",
    "outputDimensionality",
    "taskType",
    "title",
];

#[derive(Debug, Serialize)]
pub(crate) struct EmbedContent {
    pub parts: Vec<PartData>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BatchEmbedContentsResponse {
    #[serde(default)]
    pub embeddings: Vec<EmbeddingValues>,
    #[serde(default)]
    pub usage_metadata: Option<EmbedUsageMetadata>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct EmbeddingValues {
    pub values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct EmbedUsageMetadata {
    #[serde(default)]
    pub prompt_token_count: Option<u64>,
}

/// Map a request to the wire. Capability checks
/// ([`allama::adapter::embedding::preflight`]) run before this; it
/// checks what the wire needs: inputs, provider data, and limits.
pub(crate) fn to_api_request(
    request: &EmbeddingRequest,
    options: &EmbeddingRequestOptions,
) -> Result<(BatchEmbedContentsRequest, Vec<Warning>)> {
    if request.inputs.is_empty() {
        return Err(
            Error::invalid_request("embedding request must have at least one input")
                .authored()
                .with_provider(PROVIDER_NAME),
        );
    }
    request.check_provider_data(PROVIDER_NAME, NAMESPACE, &[EmbeddingTitle::KEY])?;
    if request.inputs.len() > MAX_INPUTS_PER_CALL {
        return Err(Error::invalid_request(format!(
            "Gemini embeds at most {MAX_INPUTS_PER_CALL} inputs per call, got {}; split the \
             batch (see EmbeddingProvider::embedding_limits)",
            request.inputs.len()
        ))
        .authored()
        .with_provider(PROVIDER_NAME));
    }
    refuse_owned_fields(
        &options.extra,
        EMBED_CONTENT_FIELDS,
        "EmbeddingRequestOptions::extra",
    )?;
    let mut d = Degradations::new(
        PROVIDER_NAME,
        request.unsupported_policy.unwrap_or_default(),
    );
    crate::options::report_options_under_name(&request.options, "EmbeddingRequestOptions", &mut d);
    let task_type = task_type(request, options, &mut d)?;
    let model = resource_path(&request.model)?;
    let requests = request
        .inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            Ok(EmbedContentRequest {
                model: model.clone(),
                content: EmbedContent {
                    parts: input_parts(index, input, &mut d)?,
                },
                task_type: task_type.clone(),
                title: input_title(index, input, options)?,
                output_dimensionality: request.dimensions,
                extra: options.extra.clone(),
            })
        })
        .collect::<Result<_>>()?;
    Ok((BatchEmbedContentsRequest { requests }, d.into_warnings()))
}

/// The `taskType`: the [`EmbeddingRequestOptions::task_type`] option when
/// set, otherwise the portable input type.
fn task_type(
    request: &EmbeddingRequest,
    options: &EmbeddingRequestOptions,
    d: &mut Degradations,
) -> Result<Option<String>> {
    let portable = match request.input_type {
        None => None,
        Some(EmbeddingInputType::Query) => Some("RETRIEVAL_QUERY"),
        Some(EmbeddingInputType::Document) => Some("RETRIEVAL_DOCUMENT"),
        Some(other) => {
            d.unsupported_hint(
                RequestField::InputType,
                format!("Gemini has no task type for the `{other}` input type; it was not sent"),
            )?;
            None
        }
    };
    Ok(match (&options.task_type, portable) {
        (Some(raw), Some(portable)) => {
            if raw != portable {
                d.approximated(
                    Subject::Field {
                        field: RequestField::InputType,
                    },
                    format!(
                        "the `task_type` option overrides the portable input type; `{raw}` was \
                         used instead of `{portable}`"
                    ),
                );
            }
            Some(raw.clone())
        }
        (raw, portable) => raw.clone().or(portable.map(str::to_owned)),
    })
}

fn input_parts(
    index: usize,
    input: &EmbeddingInput,
    d: &mut Degradations,
) -> Result<Vec<PartData>> {
    if input.parts().is_empty() {
        return Err(
            Error::invalid_request(format!("embedding input {index} has no parts"))
                .authored()
                .with_provider(PROVIDER_NAME),
        );
    }
    let images = input
        .parts()
        .iter()
        .filter(|part| part.as_image().is_some())
        .count();
    if images > MAX_IMAGES_PER_INPUT {
        return Err(Error::invalid_request(format!(
            "Gemini embeds at most {MAX_IMAGES_PER_INPUT} images per input; input {index} has \
             {images}"
        ))
        .authored()
        .with_provider(PROVIDER_NAME));
    }
    input
        .parts()
        .iter()
        .enumerate()
        .map(|(part_index, part)| match part {
            EmbeddingPart::Text(text) => Ok(PartData::Text(text.text.clone())),
            EmbeddingPart::Image(image) => {
                if image.detail.is_some() {
                    d.hint(
                        Subject::EmbeddingInput {
                            input: index,
                            part: Some(part_index),
                        },
                        "Gemini embeddings have no image detail setting; the detail was not sent",
                    )?;
                }
                let subject = Subject::EmbeddingInput {
                    input: index,
                    part: Some(part_index),
                };
                image_part(image, subject, d)
            }
            other => Err(d.reject(
                Subject::EmbeddingInput {
                    input: index,
                    part: Some(part_index),
                },
                format!("Gemini cannot embed `{}` parts", other.kind()),
            )),
        })
        .collect()
}

/// A per-input title overrides the request-level default.
fn input_title(
    index: usize,
    input: &EmbeddingInput,
    options: &EmbeddingRequestOptions,
) -> Result<Option<String>> {
    let title = input.provider_data.get::<EmbeddingTitle>().map_err(|err| {
        Error::invalid_request(format!(
            "embedding input {index} has an invalid `{NAMESPACE}` title: {err}"
        ))
        .authored()
        .with_provider(PROVIDER_NAME)
    })?;
    Ok(title.map(|title| title.0).or_else(|| options.title.clone()))
}

pub(crate) fn from_api_response(
    response: BatchEmbedContentsResponse,
    model: &str,
    expected: usize,
) -> Result<EmbeddingResponse> {
    if response.embeddings.len() != expected {
        return Err(Error::unexpected_response(format!(
            "Gemini returned {} embeddings for {expected} inputs",
            response.embeddings.len()
        ))
        .authored()
        .with_provider(PROVIDER_NAME));
    }
    let embeddings = response.embeddings.into_iter().map(|e| e.values).collect();
    let mut converted = EmbeddingResponse::new(embeddings)
        .model(model)
        .provider(PROVIDER_NAME);
    if let Some(tokens) = response
        .usage_metadata
        .and_then(|usage| usage.prompt_token_count)
    {
        converted = converted.usage(Usage::new().input_tokens(tokens));
    }
    Ok(converted)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Embedding options under `gemini` are reported dropped, as chat
    /// options are.
    #[test]
    fn options_under_the_provider_name_are_reported_dropped() {
        let request = EmbeddingRequest::new("gemini-embedding-001")
            .inputs(["a"])
            .with_options_json(
                PROVIDER_NAME,
                serde_json::json!({"task_type": "CLUSTERING"}),
            );
        let options = request.resolve_option::<EmbeddingRequestOptions>().unwrap();
        let (body, warnings) = to_api_request(&request, &options).unwrap();
        let body = serde_json::to_value(body).unwrap();
        assert!(body["requests"][0].get("taskType").is_none(), "{body}");
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert_eq!(
            warnings[0].code.as_deref(),
            Some("dropped.provider_name_options")
        );
        assert!(
            warnings[0].message.contains("`EmbeddingRequestOptions`"),
            "{warnings:?}"
        );
    }

    #[test]
    fn extra_fields_go_into_every_input_request() {
        let request = EmbeddingRequest::new("gemini-embedding-001").inputs(["a", "b"]);
        let options = EmbeddingRequestOptions::new().extra("autoTruncate", false);
        let json = serde_json::to_value(to_api_request(&request, &options).unwrap().0).unwrap();
        for entry in json["requests"].as_array().unwrap() {
            assert_eq!(entry["autoTruncate"], false);
        }

        for key in EMBED_CONTENT_FIELDS.iter().copied().chain(["task_type"]) {
            let options = EmbeddingRequestOptions::new().extra(key, "x");
            let err = to_api_request(&request, &options).unwrap_err();
            assert!(
                matches!(err, Error::InvalidRequest { .. }),
                "{key}: {err:?}"
            );
        }
    }

    #[test]
    fn maps_inputs_options_and_dimensions() {
        let request = EmbeddingRequest::new("gemini-embedding-001")
            .inputs(["a", "b"])
            .dimensions(256);
        let options = EmbeddingRequestOptions::new()
            .task_type("RETRIEVAL_DOCUMENT")
            .title("Doc");
        let json = serde_json::to_value(to_api_request(&request, &options).unwrap().0).unwrap();
        let requests = json["requests"].as_array().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0]["model"], "models/gemini-embedding-001");
        assert_eq!(requests[0]["taskType"], "RETRIEVAL_DOCUMENT");
        assert_eq!(requests[0]["title"], "Doc");
        assert_eq!(requests[0]["outputDimensionality"], 256);
        assert_eq!(requests[1]["content"]["parts"][0]["text"], "b");
    }

    fn titled(text: &str, title: &str) -> allama::EmbeddingInput {
        allama::EmbeddingInput::text(text).with_provider_data(EmbeddingTitle::new(title).into())
    }

    fn data(namespace: &str, key: &str, value: serde_json::Value) -> allama::ProviderData {
        allama::ProviderData::new().with_raw(namespace, key, value)
    }

    #[test]
    fn per_input_title_overrides_the_request_default_for_its_entry_only() {
        let request = EmbeddingRequest::new("gemini-embedding-001")
            .input(titled("a", "First"))
            .input("b");
        let options = EmbeddingRequestOptions {
            title: Some("Default".into()),
            ..Default::default()
        };
        let api = to_api_request(&request, &options).unwrap().0;
        assert_eq!(api.requests[0].title.as_deref(), Some("First"));
        assert_eq!(api.requests[1].title.as_deref(), Some("Default"));

        let api = to_api_request(&request, &EmbeddingRequestOptions::default())
            .unwrap()
            .0;
        assert_eq!(api.requests[0].title.as_deref(), Some("First"));
        assert_eq!(api.requests[1].title, None);
    }

    #[test]
    fn title_on_a_part_is_rejected() {
        let request = EmbeddingRequest::new("m").input(allama::EmbeddingInput::from(
            allama::Text::new("a").with_provider_data(data(NAMESPACE, "title", "T".into())),
        ));
        let err = to_api_request(&request, &EmbeddingRequestOptions::default()).unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err}");
        assert!(err.message().contains("belongs on the input"), "{err}");
    }

    #[test]
    fn malformed_or_unknown_input_data_is_rejected() {
        let malformed = EmbeddingRequest::new("m").input(
            allama::EmbeddingInput::text("a").with_provider_data(data(
                NAMESPACE,
                "title",
                5.into(),
            )),
        );
        let err = to_api_request(&malformed, &EmbeddingRequestOptions::default()).unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err}");
        assert_eq!(
            err.context().unwrap().provider.as_deref(),
            Some(PROVIDER_NAME)
        );

        let typo = EmbeddingRequest::new("m").input(
            allama::EmbeddingInput::text("a").with_provider_data(data(
                NAMESPACE,
                "titel",
                "T".into(),
            )),
        );
        let err = to_api_request(&typo, &EmbeddingRequestOptions::default()).unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err}");
        assert!(err.message().contains("titel"), "{err}");
    }

    #[test]
    fn part_replay_state_and_foreign_data_are_ignored() {
        let request = EmbeddingRequest::new("m")
            .input(allama::EmbeddingInput::from(
                allama::Text::new("a").with_provider_data(data(
                    NAMESPACE,
                    "thought_signature",
                    "sig".into(),
                )),
            ))
            .input(allama::EmbeddingInput::text("b").with_provider_data(data(
                "openai",
                "anything",
                1.into(),
            )));
        let api = to_api_request(&request, &EmbeddingRequestOptions::default())
            .unwrap()
            .0;
        assert_eq!(api.requests.len(), 2);
        assert!(matches!(&api.requests[0].content.parts[0], PartData::Text(text) if text == "a"));
    }

    /// The text models' lack of image and composite input is model data
    /// (ADR 0015): the input is sent, and the API decides.
    #[test]
    fn text_models_get_images_and_several_parts_sent() {
        let provider = crate::tests::test_provider();
        for input in [
            allama::EmbeddingInput::image_base64("image/png", "AAAA"),
            allama::EmbeddingInput::from(vec![
                allama::EmbeddingPart::text("a"),
                allama::EmbeddingPart::text("b"),
            ]),
        ] {
            let request = EmbeddingRequest::new("gemini-embedding-001").input(input);
            let validation = provider.validate_embedding_request(&request).unwrap();
            assert!(validation.warnings().is_empty(), "{validation:?}");
            let (api, _) = to_api_request(&request, &EmbeddingRequestOptions::default()).unwrap();
            assert!(!api.requests[0].content.parts.is_empty());
        }
    }

    #[test]
    fn multimodal_inputs_map_to_content_parts() {
        let request = EmbeddingRequest::new("gemini-embedding-2")
            .input(vec![
                allama::EmbeddingPart::text("A red square"),
                allama::EmbeddingPart::image_base64("image/png", "AAAA"),
            ])
            .input(allama::EmbeddingInput::from(
                allama::Image::url("https://example.com/photo").with_media_type("image/jpeg"),
            ));
        let validation = crate::tests::test_provider()
            .validate_embedding_request(&request)
            .unwrap();
        assert!(validation.warnings().is_empty());
        let (api, warnings) =
            to_api_request(&request, &EmbeddingRequestOptions::default()).unwrap();
        assert!(warnings.is_empty());
        let json = serde_json::to_value(&api).unwrap();
        assert_eq!(
            json["requests"][0]["content"]["parts"],
            serde_json::json!([
                {"text": "A red square"},
                {"inlineData": {"mimeType": "image/png", "data": "AAAA"}}
            ])
        );
        assert_eq!(
            json["requests"][1]["content"]["parts"],
            serde_json::json!([
                {"fileData": {"mimeType": "image/jpeg", "fileUri": "https://example.com/photo"}}
            ])
        );
    }

    #[test]
    fn image_detail_is_a_hint() {
        let request =
            EmbeddingRequest::new("gemini-embedding-2").input(allama::EmbeddingInput::from(
                allama::Image::base64("image/png", "AAAA").with_detail(allama::ImageDetail::High),
            ));
        let (_, warnings) = to_api_request(&request, &EmbeddingRequestOptions::default()).unwrap();
        assert_eq!(warnings.len(), 1);
        assert_eq!(
            warnings[0].subject,
            allama::Subject::EmbeddingInput {
                input: 0,
                part: Some(0)
            }
        );
        let strict = request.unsupported_policy(allama::UnsupportedPolicy::Error);
        let err = to_api_request(&strict, &EmbeddingRequestOptions::default()).unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }), "{err}");
    }

    #[test]
    fn image_limits_and_media_types_are_checked_before_sending() {
        let too_many = EmbeddingRequest::new("gemini-embedding-2").input(
            (0..=MAX_IMAGES_PER_INPUT)
                .map(|_| allama::EmbeddingPart::image_base64("image/png", "AAAA"))
                .collect::<Vec<_>>(),
        );
        let err = to_api_request(&too_many, &EmbeddingRequestOptions::default()).unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err}");
        assert!(err.message().contains("at most 6 images"), "{err}");

        let unknown_type = EmbeddingRequest::new("gemini-embedding-2").input(
            allama::EmbeddingInput::image_url("https://example.com/photo"),
        );
        let err = to_api_request(&unknown_type, &EmbeddingRequestOptions::default()).unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }), "{err}");

        let conflicting =
            EmbeddingRequest::new("gemini-embedding-2").input(allama::EmbeddingInput::from(
                allama::Image::base64("image/png", "AAAA").with_media_type("image/jpeg"),
            ));
        let err = to_api_request(&conflicting, &EmbeddingRequestOptions::default()).unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err}");

        let empty = EmbeddingRequest::new("gemini-embedding-2")
            .input(allama::EmbeddingInput::from_parts([]));
        let err = to_api_request(&empty, &EmbeddingRequestOptions::default()).unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err}");
    }

    #[test]
    fn capabilities_follow_the_model() {
        use EmbeddingCapability::{CompositeInput, ImageInput};
        for model in ["gemini-embedding-2", "models/gemini-embedding-2-preview"] {
            assert_eq!(
                model_capability(model, ImageInput),
                CapabilitySupport::Supported
            );
            assert_eq!(
                model_capability(model, CompositeInput),
                CapabilitySupport::Supported
            );
        }
        for model in [
            "gemini-embedding-001",
            "text-embedding-004",
            "embedding-001",
        ] {
            assert_eq!(
                model_capability(model, ImageInput),
                CapabilitySupport::Unknown
            );
            assert_eq!(
                model_capability(model, CompositeInput),
                CapabilitySupport::Unknown
            );
        }
        assert_eq!(
            model_capability("custom", ImageInput),
            CapabilitySupport::Unknown
        );
    }

    #[test]
    fn input_type_maps_to_the_retrieval_task_types() {
        for (input_type, task_type) in [
            (EmbeddingInputType::Query, "RETRIEVAL_QUERY"),
            (EmbeddingInputType::Document, "RETRIEVAL_DOCUMENT"),
        ] {
            let request = EmbeddingRequest::new("gemini-embedding-001")
                .input("a")
                .input_type(input_type);
            let (api, warnings) =
                to_api_request(&request, &EmbeddingRequestOptions::default()).unwrap();
            assert!(warnings.is_empty(), "{warnings:?}");
            assert_eq!(api.requests[0].task_type.as_deref(), Some(task_type));
        }
    }

    #[test]
    fn the_task_type_option_overrides_the_input_type() {
        let request = EmbeddingRequest::new("gemini-embedding-001")
            .input("a")
            .input_type(EmbeddingInputType::Query);
        let options = EmbeddingRequestOptions {
            task_type: Some("CLASSIFICATION".into()),
            ..Default::default()
        };
        let (api, warnings) = to_api_request(&request, &options).unwrap();
        assert_eq!(api.requests[0].task_type.as_deref(), Some("CLASSIFICATION"));
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert_eq!(warnings[0].kind, allama::WarningKind::Approximated);
        assert_eq!(
            warnings[0].subject,
            Subject::Field {
                field: RequestField::InputType
            }
        );

        // Agreeing values are no override.
        let agreeing = EmbeddingRequestOptions {
            task_type: Some("RETRIEVAL_QUERY".into()),
            ..Default::default()
        };
        let (_, warnings) = to_api_request(&request, &agreeing).unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    #[test]
    fn input_type_support_follows_the_model() {
        use EmbeddingCapability::InputType;
        for model in [
            "gemini-embedding-001",
            "models/gemini-embedding-001",
            "text-embedding-004",
        ] {
            assert_eq!(
                model_capability(model, InputType),
                CapabilitySupport::Supported,
                "{model}"
            );
        }
        for model in [
            "gemini-embedding-2",
            "gemini-embedding-2-preview",
            "embedding-001",
            "custom",
        ] {
            assert_eq!(
                model_capability(model, InputType),
                CapabilitySupport::Unknown,
                "{model}"
            );
        }
    }

    /// `gemini-embedding-2` takes no task type ("The `task_type` parameter
    /// does not apply to gemini-embedding-2",
    /// <https://ai.google.dev/gemini-api/docs/embeddings>), but that is model
    /// data (ADR 0015): the portable input type is sent as `taskType`, under
    /// every policy, and the API decides.
    #[test]
    fn input_type_on_gemini_embedding_2_is_sent() {
        let provider = crate::tests::test_provider();
        let request = EmbeddingRequest::new("gemini-embedding-2")
            .input("a")
            .input_type(EmbeddingInputType::Query);
        for request in [
            request.clone(),
            request.unsupported_policy(allama::UnsupportedPolicy::Error),
        ] {
            let validation = provider.validate_embedding_request(&request).unwrap();
            assert!(validation.warnings().is_empty(), "{validation:?}");
            let (api, _) = to_api_request(&request, &EmbeddingRequestOptions::default()).unwrap();
            assert_eq!(
                api.requests[0].task_type.as_deref(),
                Some("RETRIEVAL_QUERY")
            );
        }
    }

    #[test]
    fn response_count_mismatch_is_unexpected_and_usage_is_mapped() {
        let short = BatchEmbedContentsResponse {
            embeddings: vec![EmbeddingValues { values: vec![1.0] }],
            usage_metadata: None,
        };
        let err = from_api_response(short, "m", 2).unwrap_err();
        assert!(matches!(err, Error::UnexpectedResponse { .. }), "{err}");

        let wire: BatchEmbedContentsResponse = serde_json::from_value(serde_json::json!({
            "embeddings": [{"values": [1.0]}],
            "usageMetadata": {"promptTokenCount": 520}
        }))
        .unwrap();
        let converted = from_api_response(wire, "m", 1).unwrap();
        assert_eq!(converted.usage.unwrap().input_tokens, Some(520));
    }

    #[test]
    fn keeps_explicit_models_prefix() {
        let request = EmbeddingRequest::new("models/gemini-embedding-001").input("a");
        let api = to_api_request(&request, &EmbeddingRequestOptions::default())
            .unwrap()
            .0;
        assert_eq!(api.requests[0].model, "models/gemini-embedding-001");
    }

    #[test]
    fn rejects_empty_and_oversized_batches() {
        let err = to_api_request(
            &EmbeddingRequest::new("m"),
            &EmbeddingRequestOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }));

        let inputs: Vec<String> = (0..=MAX_INPUTS_PER_CALL).map(|i| i.to_string()).collect();
        let err = to_api_request(
            &EmbeddingRequest::new("m").inputs(inputs),
            &EmbeddingRequestOptions::default(),
        )
        .unwrap_err();
        assert!(err.message().contains("at most 100"));
    }

    #[test]
    fn response_preserves_order_and_model() {
        let wire = BatchEmbedContentsResponse {
            embeddings: vec![
                EmbeddingValues {
                    values: vec![1.0, 2.0],
                },
                EmbeddingValues {
                    values: vec![3.0, 4.0],
                },
            ],
            usage_metadata: None,
        };
        let converted = from_api_response(wire, "gemini-embedding-001", 2).unwrap();
        assert_eq!(converted.embeddings, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(converted.model.as_deref(), Some("gemini-embedding-001"));
        assert!(converted.usage.is_none());
    }
}
