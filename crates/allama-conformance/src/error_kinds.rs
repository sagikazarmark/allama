//! Error kinds: an adapter's HTTP error mapping gives each failure the kind
//! retry, fallback and callers act on.
//!
//! [`assert_error_kinds_mapped`] maps the adapter's own error envelope for
//! each [`ErrorKindProbe`] at the statuses the kind is reported with, and
//! checks the kind, the status, and the stage of a provider's refusal. It is
//! the HTTP half of
//! [`assert_stream_errors_classified`](crate::assert_stream_errors_classified),
//! for an adapter that has no stream, or to check the mapping on its own.
//!
//! ```
//! use allama::Error;
//! use allama_conformance::error_kinds::{ErrorKindProbe, assert_error_kinds_mapped};
//! use serde_json::json;
//!
//! fn map_http_error(status: u16, body: &str) -> Error {
//!     let value: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
//!     let error = match (status, value["fault"]["code"].as_str()) {
//!         (429, _) => Error::rate_limited(format!("HTTP {status}")),
//!         (_, Some("prompt_too_long")) => Error::context_length_exceeded("HTTP 400", None),
//!         _ => Error::invalid_request(format!("HTTP {status}")),
//!     };
//!     error.authored().with_status(status)
//! }
//!
//! assert_error_kinds_mapped(
//!     |kind| {
//!         let code = match kind {
//!             ErrorKindProbe::RateLimited => "throttled",
//!             ErrorKindProbe::InvalidRequest => "bad_input",
//!             ErrorKindProbe::ContextLength => "prompt_too_long",
//!             // Kinds the adapter does not map, including later ones.
//!             _ => return None,
//!         };
//!         Some(json!({"fault": {"code": code, "reason": "probe"}}).to_string())
//!     },
//!     |status, _headers, body| map_http_error(status, &body),
//! );
//! ```

use allama::Error;

pub use crate::stream_errors::ErrorKindProbe;

/// Assert that the adapter's HTTP error mapping gives each error kind its
/// [`Error`] variant.
///
/// `envelope` gives the adapter's error envelope for each
/// [`ErrorKindProbe`], with its own vendor code, or `None` for a kind it
/// does not map, as for
/// [`assert_stream_errors_classified`](crate::assert_stream_errors_classified).
/// `map_http_error` is the adapter's HTTP error mapping. Each envelope is
/// mapped at each of the kind's [statuses](ErrorKindProbe::statuses), and
/// the error must:
///
/// - be of the kind ([`Error::kind_name`]), so retry and fallback policies
///   act on it as the kind says;
/// - carry the status, so a caller can tell a `409` from a `400`;
/// - for [`ErrorKindProbe::Unsupported`], have stage
///   [`Provider`](allama::UnsupportedStage::Provider): the provider refused
///   a request it received ([`Error::unsupported_by_provider`]).
///
/// `envelope` must map at least [`ErrorKindProbe::RateLimited`] and
/// [`ErrorKindProbe::InvalidRequest`], so an adapter that maps nothing
/// cannot pass.
///
/// # Panics
///
/// When `envelope` skips a required kind, or a mapped error breaks a rule.
pub fn assert_error_kinds_mapped<E, M>(envelope: E, mut map_http_error: M)
where
    E: Fn(ErrorKindProbe) -> Option<String>,
    M: FnMut(u16, &[(&str, &str)], String) -> Error,
{
    let headers: &[(&str, &str)] = &[("content-type", "application/json")];
    let mut problems = Vec::new();
    for &kind in ErrorKindProbe::ALL {
        let Some(envelope) = envelope(kind) else {
            if matches!(
                kind,
                ErrorKindProbe::RateLimited | ErrorKindProbe::InvalidRequest
            ) {
                problems.push(format!(
                    "{kind}: the envelope builder must map at least RateLimited and \
                     InvalidRequest, so the contract cannot pass vacuously"
                ));
            }
            continue;
        };
        for &status in kind.statuses() {
            let error = map_http_error(status, headers, envelope.clone());
            if let Some(problem) = misclassified(kind, status, &error) {
                problems.push(format!("{kind}: at HTTP {status}, {problem}: {envelope}"));
            }
        }
    }
    assert!(
        problems.is_empty(),
        "error_kinds_mapped failed:\n  {}",
        problems.join("\n  ")
    );
}

/// What is wrong with `error`, the mapping of `kind`'s envelope at
/// `status`.
fn misclassified(kind: ErrorKindProbe, status: u16, error: &Error) -> Option<String> {
    if error.kind_name() != kind.kind_name() {
        return Some(format!(
            "the envelope maps to {}, not {}",
            error.kind_name(),
            kind.kind_name()
        ));
    }
    if error.status() != Some(status) {
        return Some(format!(
            "the error carries status {:?}, not {status}",
            error.status()
        ));
    }
    let stage = error.unsupported_stage()?;
    (kind == ErrorKindProbe::Unsupported && stage != allama::UnsupportedStage::Provider).then(
        || {
            format!(
                "the provider's refusal has stage {stage}, not provider (map it with \
                 Error::unsupported_by_provider)"
            )
        },
    )
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;

    fn envelope(kind: ErrorKindProbe) -> Option<String> {
        let code = match kind {
            ErrorKindProbe::RateLimited => "throttled",
            ErrorKindProbe::InvalidRequest => "bad_input",
            ErrorKindProbe::Unsupported => "unsupported_field",
            _ => return None,
        };
        Some(json!({"fault": {"code": code}}).to_string())
    }

    fn map(status: u16, body: &str, refusal: Error) -> Error {
        let value: Value = serde_json::from_str(body).unwrap_or_default();
        let error = match value["fault"]["code"].as_str() {
            Some("throttled") => Error::rate_limited("HTTP 429"),
            Some("unsupported_field") => refusal,
            _ => Error::invalid_request("HTTP 400"),
        };
        error.with_status(status)
    }

    #[test]
    fn a_mapping_by_vendor_code_passes() {
        assert_error_kinds_mapped(envelope, |status, _, body| {
            map(
                status,
                &body,
                Error::unsupported_by_provider(None, "HTTP 400"),
            )
        });
    }

    #[test]
    #[should_panic(
        expected = "Unsupported: at HTTP 400, the provider's refusal has stage \
                               unspecified"
    )]
    fn a_provider_refusal_needs_its_stage() {
        assert_error_kinds_mapped(envelope, |status, _, body| {
            map(status, &body, Error::unsupported("HTTP 400"))
        });
    }

    #[test]
    #[should_panic(expected = "RateLimited: at HTTP 429, the envelope maps to invalid_request")]
    fn a_mapping_by_status_alone_fails() {
        assert_error_kinds_mapped(
            |kind| (kind == ErrorKindProbe::RateLimited).then(|| "{}".to_owned()),
            |status, _, _| Error::invalid_request("HTTP").with_status(status),
        );
    }

    #[test]
    #[should_panic(expected = "the error carries status None, not 400")]
    fn the_status_is_kept() {
        assert_error_kinds_mapped(envelope, |_, _, body| {
            let mut error = map(0, &body, Error::unsupported_by_provider(None, "HTTP 400"));
            // Drop the whole context, status included.
            if let Some(context) = error.context_mut() {
                *context = allama::ErrorContext::new();
            }
            error
        });
    }
}
