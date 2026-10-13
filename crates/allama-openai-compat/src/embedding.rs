//! Embeddings wire types and conversions.

use std::borrow::Cow;

use allama::adapter::Degradations;
use allama::{
    EmbeddingInputType, EmbeddingRequest, EmbeddingResponse, Error, ExtraMap, RequestField, Result,
    Subject, Usage, Warning,
};
use serde::{Deserialize, Serialize};

use crate::CompatQuirks;
use crate::options::refuse_owned_fields;

/// An embeddings request body.
#[derive(Debug, Serialize)]
#[non_exhaustive]
pub(crate) struct EmbeddingsRequest {
    /// Model id.
    pub model: String,
    /// Inputs.
    pub input: Vec<String>,
    /// Output dimensions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    /// End-user id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Endpoint-specific fields, checked against [`EMBEDDINGS_FIELDS`].
    #[serde(flatten, skip_serializing_if = "ExtraMap::is_empty")]
    pub extra: ExtraMap,
}

/// The body fields of [`EmbeddingsRequest`]: the adapter owns them, so
/// `EmbeddingRequestOptions::extra` cannot set them. The input type's field
/// ([`CompatQuirks::embedding_input_type`]) is not among them: `extra` may
/// override the portable input type, with a warning.
pub(crate) const EMBEDDINGS_FIELDS: &[&str] = &["dimensions", "input", "model", "user"];

/// Embedding fields with no portable equivalent. Provider crates convert
/// their own options type into this.
///
/// Start from [`EmbeddingWireOptions::default`] and set the fields you need:
///
/// ```
/// use allama_openai_compat::EmbeddingWireOptions;
///
/// let mut options = EmbeddingWireOptions::default();
/// options.user = Some("user-123".into());
/// # let _ = options;
/// ```
#[doc(hidden)]
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct EmbeddingWireOptions {
    /// End-user id.
    pub user: Option<String>,
    /// Endpoint-specific fields.
    pub extra: ExtraMap,
}

/// An embeddings response body.
#[derive(Debug, Deserialize)]
#[non_exhaustive]
pub(crate) struct EmbeddingsResponse {
    /// Vectors.
    pub data: Vec<EmbeddingData>,
    /// Model.
    #[serde(default)]
    pub model: Option<String>,
    /// Usage.
    #[serde(default)]
    pub usage: Option<EmbeddingsUsage>,
}

/// One vector.
#[derive(Debug, Deserialize)]
#[non_exhaustive]
pub(crate) struct EmbeddingData {
    /// Vector.
    pub embedding: Vec<f32>,
    /// Input index.
    #[serde(default)]
    pub index: u32,
}

/// Embeddings usage.
#[derive(Debug, Deserialize)]
#[non_exhaustive]
pub(crate) struct EmbeddingsUsage {
    /// Prompt tokens.
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    /// Total tokens.
    #[serde(default)]
    pub total_tokens: Option<u64>,
}

/// Convert a portable embedding request.
///
/// This maps fields only; [`allama::adapter::embedding::preflight`] holds the checks
/// that need the provider's answers. The
/// [`input_type`](EmbeddingRequest::input_type) goes on the field
/// [`CompatQuirks::embedding_input_type`] names; without one it is dropped
/// with a warning (preflight has normally removed it already, from the
/// provider's `Unsupported` answer).
///
/// # Errors
///
/// Fails for a request without inputs and for inputs that are not exactly one
/// text part (see [`EmbeddingRequest::text_inputs`]): the OpenAI embeddings
/// API embeds one text per input. Under
/// [`UnsupportedPolicy::Error`](allama::UnsupportedPolicy::Error), an input
/// type the endpoint has no field for fails with [`Error::Unsupported`].
pub(crate) fn to_embeddings_request(
    request: &EmbeddingRequest,
    options: &EmbeddingWireOptions,
    quirks: &CompatQuirks,
    provider: impl Into<Cow<'static, str>>,
) -> Result<(EmbeddingsRequest, Vec<Warning>)> {
    let provider = provider.into();
    refuse_owned_fields(
        &options.extra,
        EMBEDDINGS_FIELDS,
        "EmbeddingRequestOptions::extra",
        &provider,
    )?;
    let input = request
        .text_inputs(&provider)?
        .into_iter()
        .map(str::to_owned)
        .collect();
    let mut degradations =
        Degradations::new(provider, request.unsupported_policy.unwrap_or_default());
    let mut extra = options.extra.clone();
    if let Some(input_type) = request.input_type {
        encode_input_type(input_type, quirks, &mut extra, &mut degradations)?;
    }
    Ok((
        EmbeddingsRequest {
            model: request.model.clone(),
            input,
            dimensions: request.dimensions,
            user: options.user.clone(),
            extra,
        },
        degradations.into_warnings(),
    ))
}

/// Put `input_type` on the endpoint's field, unless an `extra` field of the
/// same name overrides it.
fn encode_input_type(
    input_type: EmbeddingInputType,
    quirks: &CompatQuirks,
    extra: &mut ExtraMap,
    d: &mut Degradations,
) -> Result<()> {
    let not_supported = |d: &mut Degradations, message: String| {
        d.unsupported_hint(RequestField::InputType, message)
    };
    let Some(field) = &quirks.embedding_input_type else {
        return not_supported(
            d,
            format!(
                "this endpoint takes no query/document input type (the model may expect a role \
                 prefix in the text); the `{input_type}` input type was not sent"
            ),
        );
    };
    let value = match input_type {
        EmbeddingInputType::Query => &field.query,
        EmbeddingInputType::Document => &field.document,
        other => {
            return not_supported(
                d,
                format!("this endpoint has no value for the `{other}` input type; it was not sent"),
            );
        }
    };
    if let Some(raw) = extra.get(field.name.as_ref()) {
        if raw.as_str() != Some(value.as_ref()) {
            d.approximated(
                Subject::Field {
                    field: RequestField::InputType,
                },
                format!(
                    "the `{}` extra field overrides the portable input type; its value {raw} was \
                     used instead of `{value}`",
                    field.name
                ),
            );
        }
        return Ok(());
    }
    extra.insert(
        field.name.clone().into_owned(),
        serde_json::Value::String(value.clone().into_owned()),
    );
    Ok(())
}

/// Convert an embeddings response, putting the vectors in input order by
/// their `index`.
///
/// # Errors
///
/// Fails with [`Error::UnexpectedResponse`] unless the indices are exactly
/// `0..expected`: one vector per input, with no duplicate, missing, or
/// out-of-range index.
pub(crate) fn from_embeddings_response(
    response: EmbeddingsResponse,
    expected: usize,
    warnings: Vec<Warning>,
) -> Result<EmbeddingResponse> {
    let mut ordered: Vec<(u32, Vec<f32>)> = response
        .data
        .into_iter()
        .map(|d| (d.index, d.embedding))
        .collect();
    ordered.sort_by_key(|(index, _)| *index);
    if let Some((position, (index, _))) =
        ordered.iter().enumerate().find(|(position, (index, _))| {
            *position >= expected || usize::try_from(*index).ok() != Some(*position)
        })
    {
        let problem = match position.checked_sub(1).map(|previous| ordered[previous].0) {
            Some(previous) if previous == *index => format!("duplicate index {index}"),
            _ if usize::try_from(*index).is_ok_and(|index| index >= expected) => {
                format!("index {index} is out of range")
            }
            _ => format!("index {position} is missing"),
        };
        return Err(Error::unexpected_response(format!(
            "expected one embedding for each of {expected} inputs: {problem}"
        ))
        .authored());
    }
    if ordered.len() != expected {
        return Err(Error::unexpected_response(format!(
            "expected {expected} embeddings, received {}",
            ordered.len()
        ))
        .authored());
    }
    let mut converted = EmbeddingResponse::new(ordered.into_iter().map(|(_, v)| v).collect());
    converted.model = response.model;
    converted.usage = response.usage.map(|u| {
        let mut usage = Usage::new();
        usage.input_tokens = u.prompt_tokens;
        usage.total_tokens = u.total_tokens.or(u.prompt_tokens);
        usage
    });
    converted.warnings = warnings;
    Ok(converted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn orders_vectors_and_checks_count() {
        let response: EmbeddingsResponse = serde_json::from_value(json!({
            "data": [{"embedding": [2.0], "index": 1}, {"embedding": [1.0], "index": 0}],
            "model": "m",
            "usage": {"prompt_tokens": 4, "total_tokens": 4}
        }))
        .unwrap();
        let converted = from_embeddings_response(response, 2, Vec::new()).unwrap();
        assert_eq!(converted.embeddings, vec![vec![1.0], vec![2.0]]);
        assert_eq!(converted.usage.unwrap().input_tokens, Some(4));

        let short: EmbeddingsResponse =
            serde_json::from_value(json!({"data": [{"embedding": [1.0], "index": 0}]})).unwrap();
        assert!(from_embeddings_response(short, 2, Vec::new()).is_err());
    }

    fn with_indices(indices: &[u32]) -> EmbeddingsResponse {
        let data: Vec<_> = indices
            .iter()
            .map(|&index| json!({"embedding": [index as f32], "index": index}))
            .collect();
        serde_json::from_value(json!({"data": data})).unwrap()
    }

    #[test]
    fn rejects_duplicate_missing_and_out_of_range_indices() {
        for (indices, problem) in [
            (&[0, 1, 1][..], "duplicate index 1"),
            (&[0, 0, 1, 2][..], "duplicate index 0"),
            (&[0, 2][..], "index 1 is missing"),
            (&[1, 2][..], "index 0 is missing"),
            (&[0, 1, 3][..], "index 3 is out of range"),
            (&[0, 1, 2, 3][..], "index 3 is out of range"),
        ] {
            let err = from_embeddings_response(with_indices(indices), 3, Vec::new()).unwrap_err();
            assert!(
                matches!(err, Error::UnexpectedResponse { .. }),
                "{indices:?}: {err}"
            );
            assert!(err.message().contains(problem), "{indices:?}: {err}");
        }
        let short = from_embeddings_response(with_indices(&[0, 1]), 3, Vec::new()).unwrap_err();
        assert!(
            short
                .message()
                .contains("expected 3 embeddings, received 2"),
            "{short}"
        );
        let ordered = from_embeddings_response(with_indices(&[2, 0, 1]), 3, Vec::new()).unwrap();
        assert_eq!(ordered.embeddings, vec![vec![0.0], vec![1.0], vec![2.0]]);
    }

    #[test]
    fn inputs_map_to_texts_and_several_parts_are_rejected() {
        let (body, _) = to_embeddings_request(
            &EmbeddingRequest::new("m").inputs(["a", "b"]),
            &EmbeddingWireOptions::default(),
            &CompatQuirks::default(),
            "x",
        )
        .unwrap();
        assert_eq!(body.input, vec!["a".to_string(), "b".to_string()]);

        let combined = EmbeddingRequest::new("m").input(vec![
            allama::EmbeddingPart::text("a"),
            allama::EmbeddingPart::text("b"),
        ]);
        let err = to_embeddings_request(
            &combined,
            &EmbeddingWireOptions::default(),
            &CompatQuirks::default(),
            String::from("openai-compat:groq"),
        )
        .unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }), "{err}");
        assert_eq!(
            err.context().unwrap().provider.as_deref(),
            Some("openai-compat:groq")
        );
    }

    #[test]
    fn empty_inputs_are_rejected() {
        assert!(
            to_embeddings_request(
                &EmbeddingRequest::new("m"),
                &EmbeddingWireOptions::default(),
                &CompatQuirks::default(),
                "x"
            )
            .is_err()
        );
    }

    fn with_input_type_field() -> CompatQuirks {
        let mut quirks = CompatQuirks::new("openai-compat:test");
        quirks.embedding_input_type = Some(crate::EmbeddingInputTypeField::new(
            "input_type",
            "search_query",
            "search_document",
        ));
        quirks
    }

    #[test]
    fn input_type_goes_on_the_endpoint_field() {
        for (input_type, value) in [
            (EmbeddingInputType::Query, "search_query"),
            (EmbeddingInputType::Document, "search_document"),
        ] {
            let request = EmbeddingRequest::new("m").input("a").input_type(input_type);
            let (body, warnings) = to_embeddings_request(
                &request,
                &EmbeddingWireOptions::default(),
                &with_input_type_field(),
                "x",
            )
            .unwrap();
            assert!(warnings.is_empty(), "{warnings:?}");
            let body = serde_json::to_value(body).unwrap();
            assert_eq!(body["input_type"], value);
        }
    }

    #[test]
    fn input_type_without_a_field_is_a_hint() {
        let request = EmbeddingRequest::new("m")
            .input("a")
            .input_type(EmbeddingInputType::Query);
        let (body, warnings) = to_embeddings_request(
            &request,
            &EmbeddingWireOptions::default(),
            &CompatQuirks::default(),
            "x",
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(body).unwrap(),
            json!({"model": "m", "input": ["a"]})
        );
        assert_eq!(warnings.len(), 1);
        assert_eq!(
            warnings[0].subject,
            Subject::Field {
                field: RequestField::InputType
            }
        );

        assert!(
            warnings[0].message.ends_with(
                "(the model may expect a role prefix in the text); the `query` input type was not \
                 sent"
            ),
            "{warnings:?}"
        );

        let strict = request.unsupported_policy(allama::UnsupportedPolicy::Error);
        let err = to_embeddings_request(
            &strict,
            &EmbeddingWireOptions::default(),
            &CompatQuirks::default(),
            "x",
        )
        .unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }), "{err}");
        // Nothing was sent, so the message leaves the outcome out.
        assert_eq!(
            err.message(),
            "cannot send `input_type`: this endpoint takes no query/document input type (the \
             model may expect a role prefix in the text)"
        );
    }

    #[test]
    fn extra_cannot_set_owned_fields() {
        let request = EmbeddingRequest::new("m").input("a");
        for key in EMBEDDINGS_FIELDS.iter().copied().chain(["Model"]) {
            let mut options = EmbeddingWireOptions::default();
            options.extra.insert(key.into(), json!("x"));
            let err = to_embeddings_request(&request, &options, &CompatQuirks::default(), "demo")
                .unwrap_err();
            assert!(
                matches!(err, Error::InvalidRequest { .. }),
                "{key}: {err:?}"
            );
        }
    }

    #[test]
    fn an_extra_field_overrides_the_input_type() {
        let request = EmbeddingRequest::new("m")
            .input("a")
            .input_type(EmbeddingInputType::Query);
        let mut options = EmbeddingWireOptions::default();
        options
            .extra
            .insert("input_type".into(), json!("classification"));
        let (body, warnings) =
            to_embeddings_request(&request, &options, &with_input_type_field(), "x").unwrap();
        assert_eq!(
            serde_json::to_value(body).unwrap()["input_type"],
            "classification"
        );
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert_eq!(warnings[0].kind, allama::WarningKind::Approximated);

        // The same value is no override.
        options
            .extra
            .insert("input_type".into(), json!("search_query"));
        let (_, warnings) =
            to_embeddings_request(&request, &options, &with_input_type_field(), "x").unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
    }
}
