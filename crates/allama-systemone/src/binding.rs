//! The System One codec for transports that are not this crate's HTTP
//! client: `allama-cloudflare-worker` runs Clef decision models through the
//! Worker's AI binding, whose input and output are this crate's Workers AI
//! dialect without Cloudflare's HTTP envelope. Sharing the codec keeps one
//! mapping of the format (images, model selector, confidence) instead of
//! two that drift.
//!
//! This is not public API: it is `#[doc(hidden)]`, carries no stability
//! promise, and may change in any release. The crates that use it depend on
//! this one with an exact version requirement (`=`, set in the workspace
//! `Cargo.toml`), so a published pair can never be mixed.

use allama::adapter::Degradations;
use allama::{DecisionRequest, DecisionResponse, Error, ExtraMap, Result};
use serde::Serialize;

use crate::Confidence;
use crate::wire::{self, Dialect};

/// The input a Workers AI decision model (`@cf/cloudflare/clef`) runs
/// with, for a validated `request`, reporting through `d`: its model
/// selector, input, images and questions, with `extra` merged in.
///
/// Serialize it into the binding's input object: question and option order
/// is the order the model reads them in, so it must not go through a map
/// that sorts its keys (`serde_json::Value`).
///
/// # Errors
///
/// [`Error::InvalidRequest`] when `extra` sets a field the codec owns;
/// [`Error::Unsupported`] for an image that is not inline data.
pub fn encode_workers_ai<'a>(
    request: &'a DecisionRequest,
    extra: &'a ExtraMap,
    d: &mut Degradations,
) -> Result<impl Serialize + use<'a>> {
    wire::encode_parts(request, None, extra, Dialect::WorkersAi, d)
}

/// The model selector a Workers AI decision model reads from its input:
/// the last segment of its name (`clef` for `@cf/cloudflare/clef`).
#[must_use]
pub fn workers_ai_selector(model: &str) -> &str {
    wire::workers_ai_selector(model)
}

/// Decode a Workers AI decision model's output, the JSON text of the object
/// the binding returns, into the response to `request` from `provider`:
/// checked against the request ([`DecisionResponse::validate`]), with each
/// question's confidence statistic kept as [`Confidence`].
///
/// Answers keep the text's order, as on the HTTP transport.
///
/// # Errors
///
/// [`Error::UnexpectedResponse`] when `output` is not a System One
/// response, or does not answer `request`.
pub fn decode_workers_ai(
    request: &DecisionRequest,
    output: &str,
    provider: &str,
) -> Result<DecisionResponse> {
    let with_provider = |err: Error| err.with_provider(provider);
    let body: wire::WireResponse = serde_json::from_str(output).map_err(|err| {
        with_provider(
            Error::unexpected_response("the output is not a System One response")
                .authored()
                .with_upstream_message(err.to_string()),
        )
    })?;
    let wire::Decoded {
        mut response,
        confidence,
        // Gateway fields are OpenRouter's.
        gateway: _,
    } = wire::decode(body).map_err(with_provider)?;
    response.validate(request).map_err(with_provider)?;
    response.provider = Some(provider.to_owned().into());
    if !confidence.is_empty() {
        response
            .provider_data
            .insert(&Confidence(confidence))
            .map_err(with_provider)?;
    }
    Ok(response)
}

#[cfg(test)]
mod tests {
    use allama::decision::Question;
    use allama::{Image, UnsupportedPolicy};
    use serde_json::json;

    use super::*;

    #[test]
    fn encodes_the_workers_ai_input() {
        let request = DecisionRequest::new("@cf/cloudflare/clef", "x")
            .image(Image::base64("image/png", "aGk="))
            .question("q", Question::yes_no("?"));
        let extra: ExtraMap = [("seed".to_owned(), json!(1))].into_iter().collect();
        let mut d = Degradations::new("cloudflare", UnsupportedPolicy::Warn);
        let input = encode_workers_ai(&request, &extra, &mut d).unwrap();
        assert_eq!(
            serde_json::to_value(input).unwrap(),
            json!({
                "model": "clef",
                "state": "x",
                "questions": {"q": {"type": "noul", "instructions": "?"}},
                "images": [{"content_type": "image/png", "base64": "aGk="}],
                "seed": 1
            })
        );
    }

    #[test]
    fn decodes_in_document_order() {
        let request = DecisionRequest::new("@cf/cloudflare/clef", "x")
            .question(
                "team",
                Question::choice("?").options(["technical", "billing"]),
            )
            .question("b", Question::yes_no("?"));
        let output = r#"{"model":"clef","answers":{
            "team":{"type":"choice","choice":"technical","probabilities":{"technical":0.8,"billing":0.2},"confidence":0.3},
            "b":{"type":"noul","noul":0.4}},"usage":{"input_tokens":3,"output_tokens":0}}"#;
        let response = decode_workers_ai(&request, output, "cloudflare").unwrap();
        assert_eq!(response.provider.as_deref(), Some("cloudflare"));
        assert_eq!(response.answers[0].0, "team");
        assert_eq!(
            response.choice("team").unwrap().probabilities[0],
            ("technical".to_owned(), 0.8)
        );
        let confidence = response.provider_data.get::<Confidence>().unwrap().unwrap();
        assert_eq!(confidence.0["team"], 0.3);

        let err = decode_workers_ai(&request, r#"{"answers":{}}"#, "cloudflare").unwrap_err();
        assert!(matches!(err, Error::UnexpectedResponse { .. }), "{err:?}");
        assert_eq!(
            err.context().unwrap().provider.as_deref(),
            Some("cloudflare")
        );
        let err = decode_workers_ai(&request, "[]", "cloudflare").unwrap_err();
        assert!(matches!(err, Error::UnexpectedResponse { .. }), "{err:?}");
    }
}
