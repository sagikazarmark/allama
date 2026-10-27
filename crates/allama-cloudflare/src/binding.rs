//! The Clairic codec for transports that are not this crate's HTTP client:
//! `allama-cloudflare-worker` runs extraction models through the Worker's
//! AI binding, whose input and output are this crate's wire format without
//! Cloudflare's REST envelope. Sharing the codec keeps one mapping of the
//! format (fields, spans, statistics) instead of two that drift.
//!
//! This is not public API: it is `#[doc(hidden)]`, carries no stability
//! promise, and may change in any release. The crates that use it depend on
//! this one with an exact version requirement (`=`, set in the workspace
//! `Cargo.toml`), so a published pair can never be mixed.

use allama::adapter::Degradations;
use allama::{Error, ExtractionRequest, ExtractionResponse, Result};
use serde::Serialize;

use crate::ExtractionRequestOptions;
use crate::wire;

/// The input a Workers AI extraction model (`@cf/cloudflare/clairic`) runs
/// with, for a validated `request` with `options`, reporting through `d`.
///
/// Serialize it into the binding's input object: field order is the order
/// the model reads them in, so it must not go through a map that sorts its
/// keys (`serde_json::Value`).
///
/// # Errors
///
/// [`Error::InvalidRequest`] when `options.extra` sets a field the codec
/// owns or a null threshold names an unknown field; [`Error::Unsupported`]
/// for an image that is not inline data.
pub fn encode_workers_ai<'a>(
    request: &'a ExtractionRequest,
    options: &'a ExtractionRequestOptions,
    d: &mut Degradations,
) -> Result<impl Serialize + use<'a>> {
    wire::encode(request, options, d)
}

/// The model selector a Workers AI extraction model reads from its input:
/// the last segment of its name (`clairic` for `@cf/cloudflare/clairic`).
#[must_use]
pub fn workers_ai_selector(model: &str) -> &str {
    wire::selector(model)
}

pub use crate::builtin_extraction_capability;

/// Decode a Workers AI extraction model's output, the JSON text of the
/// object the binding returns, into the response to `request` from
/// `provider`: checked against the request
/// ([`ExtractionResponse::validate`]), with the model's statistics kept as
/// provider data ([`Confidence`](crate::Confidence),
/// [`Guesses`](crate::Guesses), [`Distributions`](crate::Distributions),
/// [`AnswerExtras`](crate::AnswerExtras), [`RunDetails`](crate::RunDetails)),
/// and the verbatim output as
/// [`RawResponse`](crate::RawResponse) when `options` ask for it.
///
/// # Errors
///
/// [`Error::UnexpectedResponse`] when `output` is not a Clairic response,
/// or does not answer `request`.
pub fn decode_workers_ai(
    request: &ExtractionRequest,
    output: &str,
    options: &ExtractionRequestOptions,
    provider: &'static str,
) -> Result<ExtractionResponse> {
    let body: wire::WireResponse = serde_json::from_str(output).map_err(|err| {
        Error::unexpected_response("the model's output is not a Clairic response")
            .authored()
            .with_upstream_message(err.to_string())
    })?;
    let decoded = wire::decode(body, request)?;
    let mut response = decoded.response;
    response.provider = Some(provider.into());
    crate::attach_provider_data(&mut response, decoded.statistics)?;
    if options.raw_response {
        let raw: serde_json::Value = serde_json::from_str(output).map_err(|err| {
            Error::unexpected_response("the model's output is not JSON")
                .authored()
                .with_upstream_message(err.to_string())
        })?;
        response.provider_data.insert(&crate::RawResponse(raw))?;
    }
    Ok(response)
}
