use std::borrow::Cow;

use super::role_prefix::input_type_message;
use crate::{
    Degradations, EmbeddingCapability, EmbeddingPart, EmbeddingProvider, EmbeddingRequest, Error,
    RequestField, Result, Subject,
};

/// Check `request` against what `provider` reports for the request's model,
/// before it is encoded and sent.
///
/// - The batch must not be empty and must fit
///   [`embedding_limits`](EmbeddingProvider::embedding_limits); a
///   larger batch fails with [`Error::InvalidRequest`] (split it with
///   [`EmbeddingProviderExt::embed_chunked`](crate::EmbeddingProviderExt::embed_chunked)).
/// - Capabilities reported [`Unsupported`](crate::CapabilitySupport::Unsupported)
///   fail with [`Error::Unsupported`] under every
///   [`UnsupportedPolicy`](crate::UnsupportedPolicy): several inputs without
///   [`BatchInput`](EmbeddingCapability::BatchInput), `dimensions` without
///   [`OutputDimensions`](EmbeddingCapability::OutputDimensions), image parts
///   without [`ImageInput`](EmbeddingCapability::ImageInput), and several
///   parts in one input without
///   [`CompositeInput`](EmbeddingCapability::CompositeInput). Each of these
///   is semantic (ADR 0003, ADR 0012): dropping it would silently change the
///   vectors.
///
/// - The [`input_type`](EmbeddingRequest::input_type) is a hint (ADR 0003):
///   without [`InputType`](EmbeddingCapability::InputType) it is removed
///   from `request` with an [`Unsupported`](crate::WarningKind::Unsupported)
///   warning on [`RequestField::InputType`], or fails under
///   [`UnsupportedPolicy::Error`](crate::UnsupportedPolicy::Error). For a
///   model known to take the role as a text prefix (nomic, E5, BGE v1.5,
///   Qwen3-Embedding, `gemini-embedding-2`, …) the message names the prefix
///   its card documents; otherwise it says the model may expect one. The
///   text is never rewritten.
///
/// [`Unknown`](crate::CapabilitySupport::Unknown) never blocks a request (models
/// newer than the provider crate, fine-tunes) and never removes anything;
/// it is recorded in `degradations` so a validation can report it
/// ([`relied_on_unknown`](crate::Validation::relied_on_unknown)).
///
/// The signature matches [`chat::preflight`](crate::adapter::chat::preflight):
/// `degradations` carries the policy and collects warnings, and `request`
/// stays borrowed unless something is removed. Adapters reach it through
/// [`embedding::prepare`](crate::adapter::embedding::prepare), which runs it first in
/// `embed`; the default
/// [`validate_embedding_request`](EmbeddingProvider::validate_embedding_request)
/// runs it on a dry run.
///
/// # Errors
///
/// As above, for the first problem found; under
/// [`UnsupportedPolicy::Error`](crate::UnsupportedPolicy::Error), also for an
/// unsupported hint.
pub fn preflight<P: EmbeddingProvider + ?Sized>(
    provider: &P,
    request: &mut Cow<'_, EmbeddingRequest>,
    degradations: &mut Degradations,
) -> Result<()> {
    let name = provider.provider_name();
    let model = request.model.as_str();
    let count = request.inputs.len();
    let answer = |capability| provider.embedding_capability(model, capability);
    let d = degradations;
    // Records an `Unknown` answer, so validation can report it.
    let refuses = |d: &mut Degradations, capability, subject: Subject| {
        d.refuses(capability, answer(capability), || subject)
    };

    if count == 0 {
        return Err(
            Error::invalid_request("embedding request must have at least one input")
                .authored()
                .with_provider(name.to_owned()),
        );
    }
    if let Some(max) = provider.embedding_limits(model).max_inputs_per_call
        && count > max
    {
        return Err(Error::invalid_request(format!(
            "`{name}` embeds at most {max} inputs per request for `{model}`, got {count}; \
             split the batch with EmbeddingProviderExt::embed_chunked"
        ))
        .authored()
        .with_provider(name.to_owned()));
    }
    if count > 1
        && refuses(
            d,
            EmbeddingCapability::BatchInput,
            Subject::EmbeddingInput {
                input: 1,
                part: None,
            },
        )
    {
        return Err(d.reject(
            Subject::EmbeddingInput {
                input: 1,
                part: None,
            },
            format!("embedding {count} inputs in one request is not supported by `{name}`"),
        ));
    }
    if request.dimensions.is_some()
        && refuses(
            d,
            EmbeddingCapability::OutputDimensions,
            Subject::Field {
                field: RequestField::Dimensions,
            },
        )
    {
        return Err(d.unsupported(
            RequestField::Dimensions,
            format!("choosing the output dimensions is not supported by `{name}`"),
        ));
    }
    for (index, input) in request.inputs.iter().enumerate() {
        for (position, part) in input.parts().iter().enumerate() {
            // Exhaustive on purpose: each new part kind needs its capability.
            match part {
                EmbeddingPart::Text(_) => {}
                EmbeddingPart::Image(_) => {
                    let subject = Subject::EmbeddingInput {
                        input: index,
                        part: Some(position),
                    };
                    if refuses(d, EmbeddingCapability::ImageInput, subject.clone()) {
                        return Err(
                            d.reject(subject, format!("image input is not supported by `{name}`"))
                        );
                    }
                }
            }
        }
        if input.parts().len() > 1
            && refuses(
                d,
                EmbeddingCapability::CompositeInput,
                Subject::EmbeddingInput {
                    input: index,
                    part: None,
                },
            )
        {
            return Err(d.reject(
                Subject::EmbeddingInput {
                    input: index,
                    part: None,
                },
                format!(
                    "combining {} parts into one vector is not supported by `{name}`; pass \
                     separate inputs to embed them separately",
                    input.parts().len()
                ),
            ));
        }
    }

    // Hints: removed with a warning, or an error under a strict policy.
    if let Some(input_type) = request.input_type
        && refuses(
            d,
            EmbeddingCapability::InputType,
            Subject::Field {
                field: RequestField::InputType,
            },
        )
    {
        let message = input_type_message(name, model, input_type);
        d.unsupported_hint(RequestField::InputType, message)?;
        request.to_mut().input_type = None;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CapabilitySupport;
    use crate::mock::MockEmbeddingProvider;
    use crate::{EmbeddingInput, EmbeddingInputType, Subject, UnsupportedPolicy, WarningKind};

    fn run(provider: &MockEmbeddingProvider, request: &EmbeddingRequest) -> Result<()> {
        let mut d = Degradations::new("mock", request.unsupported_policy.unwrap_or_default());
        let mut checked = Cow::Borrowed(request);
        preflight(provider, &mut checked, &mut d)?;
        assert!(matches!(checked, Cow::Borrowed(_)));
        assert!(d.warnings().is_empty());
        Ok(())
    }

    fn without(capability: EmbeddingCapability) -> MockEmbeddingProvider {
        MockEmbeddingProvider::empty()
            .with_embedding_capability(capability, CapabilitySupport::Unsupported)
    }

    fn image() -> EmbeddingPart {
        EmbeddingPart::image_url("https://example.com/a.png")
    }

    #[test]
    fn refusals_are_authored() {
        let err = run(&MockEmbeddingProvider::empty(), &EmbeddingRequest::new("m")).unwrap_err();
        assert_eq!(
            err.as_log().to_string(),
            r#"kind=invalid_request provider=mock message="embedding request must have at least one input""#
        );
        let request = EmbeddingRequest::new("m").input("a").dimensions(8);
        let err = run(&without(EmbeddingCapability::OutputDimensions), &request).unwrap_err();
        assert!(
            err.as_log().to_string().contains("message="),
            "{}",
            err.as_log()
        );
    }

    #[test]
    fn unknown_capabilities_never_block() {
        let request = EmbeddingRequest::new("m")
            .input(EmbeddingInput::from(vec![
                EmbeddingPart::text("a"),
                image(),
            ]))
            .input("b")
            .dimensions(8)
            .unsupported_policy(UnsupportedPolicy::Error);
        run(&MockEmbeddingProvider::empty(), &request).unwrap();
    }

    #[test]
    fn batch_size_is_checked() {
        let provider = MockEmbeddingProvider::empty().with_max_inputs_per_call(2);
        run(&provider, &EmbeddingRequest::new("m").inputs(["a", "b"])).unwrap();

        let err = run(&provider, &EmbeddingRequest::new("m")).unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err}");

        let oversized = EmbeddingRequest::new("m").inputs(["a", "b", "c"]);
        let err = run(&provider, &oversized).unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err}");
        assert!(err.message().contains("embed_chunked"), "{err}");
        assert_eq!(err.context().unwrap().provider.as_deref(), Some("mock"));
    }

    #[test]
    fn unsupported_capabilities_fail_under_every_policy() {
        let cases = [
            (
                EmbeddingCapability::BatchInput,
                EmbeddingRequest::new("m").inputs(["a", "b"]),
                "2 inputs",
            ),
            (
                EmbeddingCapability::OutputDimensions,
                EmbeddingRequest::new("m").input("a").dimensions(8),
                "dimensions",
            ),
            (
                EmbeddingCapability::ImageInput,
                EmbeddingRequest::new("m").input(EmbeddingInput::from(vec![image()])),
                "image input",
            ),
            (
                EmbeddingCapability::CompositeInput,
                EmbeddingRequest::new("m").input(EmbeddingInput::from(vec![
                    EmbeddingPart::text("a"),
                    EmbeddingPart::text("b"),
                ])),
                "combining 2 parts",
            ),
        ];
        for (capability, request, what) in cases {
            for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
                let request = request.clone().unsupported_policy(policy);
                let err = run(&without(capability), &request).unwrap_err();
                assert!(matches!(err, Error::Unsupported { .. }), "{capability:?}");
                assert!(err.message().contains(what), "{capability:?}: {err}");
                assert_eq!(err.context().unwrap().provider.as_deref(), Some("mock"));
            }
        }
    }

    #[test]
    fn unsupported_input_type_is_removed_with_a_warning() {
        let provider = without(EmbeddingCapability::InputType);
        let request = EmbeddingRequest::new("m")
            .input("a")
            .input_type(EmbeddingInputType::Query);
        let mut d = Degradations::new("mock", request.unsupported_policy.unwrap_or_default());
        let mut checked = Cow::Borrowed(&request);
        preflight(&provider, &mut checked, &mut d).unwrap();
        assert_eq!(checked.input_type, None);
        assert_eq!(checked.inputs, request.inputs);
        let warnings = d.warnings();
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert_eq!(warnings[0].kind, WarningKind::Unsupported);
        assert_eq!(
            warnings[0].subject,
            Subject::Field {
                field: RequestField::InputType
            }
        );
        assert_eq!(warnings[0].provider.as_deref(), Some("mock"));
        assert_eq!(
            warnings[0].message,
            "`mock` takes no query/document input type (`m` may expect a role prefix in the \
             text); the `query` input type was not sent"
        );
    }

    #[test]
    fn unsupported_input_type_fails_under_the_error_policy() {
        let request = EmbeddingRequest::new("m")
            .input("a")
            .input_type(EmbeddingInputType::Document)
            .unsupported_policy(UnsupportedPolicy::Error);
        let err = run(&without(EmbeddingCapability::InputType), &request).unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }), "{err}");
        assert_eq!(err.context().unwrap().provider.as_deref(), Some("mock"));
        // Nothing was sent, so the message leaves the outcome out.
        assert_eq!(
            err.message(),
            "cannot send `input_type`: `mock` takes no query/document input type (`m` may \
             expect a role prefix in the text)"
        );
    }

    #[test]
    fn supported_or_unknown_input_type_is_kept() {
        let request = EmbeddingRequest::new("m")
            .input("a")
            .input_type(EmbeddingInputType::Query)
            .unsupported_policy(UnsupportedPolicy::Error);
        run(&MockEmbeddingProvider::empty(), &request).unwrap();
        let supported = MockEmbeddingProvider::empty().with_embedding_capability(
            EmbeddingCapability::InputType,
            CapabilitySupport::Supported,
        );
        run(&supported, &request).unwrap();
    }
}
