//! Error mapping for the Workers AI REST API.
//!
//! Errors come in Cloudflare's envelope, `{"errors": [{"code": ..,
//! "message": ..}], "success": false}`, with Workers AI's error codes
//! (<https://developers.cloudflare.com/workers-ai/platform/errors/>).

use allama::Error;
use allama_http::{ErrorEnvelope, Headers};
use serde_json::Value;

/// Map an HTTP error response.
///
/// The status and headers are read with [`allama_http::error_context`]
/// (status, request id, retry hint, server retry directive). An envelope's
/// message is the [upstream message](Error::upstream_message); the
/// constructor message is the adapter's own (`HTTP {status}`). A body that
/// is no envelope is kept only as the body.
#[must_use]
pub fn map_http_error(status: u16, headers: &Headers, body: &str) -> Error {
    let envelope = error_in_body(body);
    let code = envelope
        .as_ref()
        .and_then(|envelope| envelope.code.as_deref());
    let message = format!("HTTP {status}");
    let mapped = match workers_ai_code(status, code) {
        Some(error) => error,
        None => match status {
            // A structured 404 names a model the account cannot run.
            404 if envelope.is_some() => Error::model_not_found(message),
            _ => allama_http::status_error(status, message),
        },
    };
    let mapped = mapped
        .with_context(allama_http::error_context(status, headers))
        .authored();
    let mapped = match envelope {
        Some(envelope) => envelope.apply(mapped),
        None => mapped,
    };
    mapped.with_body_opt((!body.is_empty()).then_some(body))
}

/// What a Workers AI error code says beyond its status: a `429` that is a
/// daily allocation or capacity, a `400` or `404` that names the model, a
/// `413` that is the request size.
fn workers_ai_code(status: u16, code: Option<&str>) -> Option<Error> {
    let message = format!("HTTP {status}");
    Some(match (status, code?) {
        (429, "3036") => Error::quota_exhausted(message),
        (429, "3040") => Error::overloaded(message),
        // `7000` is the Cloudflare API's "No route for that URI": under
        // `/ai/run/` the path names no model.
        (400 | 404, "5007" | "3042" | "7000") => Error::model_not_found(message),
        (413, _) => Error::invalid_request(message),
        _ => return None,
    })
}

/// The error envelope in `body`, if it is one with a message: Cloudflare's
/// `errors` list, or one of the common envelopes
/// ([`allama_http::error_envelope`]).
pub(crate) fn error_in_body(body: &str) -> Option<ErrorEnvelope> {
    let value = serde_json::from_str::<Value>(body.trim()).ok()?;
    if let Some(errors) = value.get("errors").and_then(Value::as_array)
        && let Some(error) = errors.first()
    {
        let mut envelope = ErrorEnvelope::default();
        envelope.message = Some(text(error.get("message"))?);
        envelope.code = text(error.get("code"));
        return Some(envelope);
    }
    let envelope = allama_http::error_envelope_value(&value)?;
    envelope.message.as_ref()?;
    Some(envelope)
}

/// A string, or a number as its decimal text.
fn text(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(text) => Some(text.clone()),
        Value::Number(number) => Some(number.to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloudflare_codes_refine_the_status() {
        let headers = Headers::default();
        let body = r#"{"result":null,"success":false,"errors":[{"code":7000,"message":"No route for that URI"}],"messages":[]}"#;
        let err = map_http_error(400, &headers, body);
        assert!(matches!(err, Error::ModelNotFound { .. }), "{err:?}");
        assert_eq!(err.upstream_message(), Some("No route for that URI"));
        assert_eq!(err.provider_code(), Some("7000"));
        let body = r#"{"success":false,"errors":[{"code":3036,"message":"daily limit"}]}"#;
        assert!(matches!(
            map_http_error(429, &headers, body),
            Error::QuotaExhausted { .. }
        ));
        let body = r#"{"success":false,"errors":[{"code":3040,"message":"capacity"}]}"#;
        assert!(matches!(
            map_http_error(429, &headers, body),
            Error::Overloaded { .. }
        ));
        let body =
            r#"{"success":false,"errors":[{"code":10000,"message":"Authentication error"}]}"#;
        assert!(matches!(
            map_http_error(401, &headers, body),
            Error::Auth { .. }
        ));
        let body = r#"{"success":false,"errors":[{"code":5012,"message":"field name invalid"}]}"#;
        let err = map_http_error(422, &headers, body);
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
        // No envelope: the status alone, the body kept.
        let err = map_http_error(500, &headers, "boom");
        assert!(matches!(err, Error::Provider { .. }), "{err:?}");
        assert_eq!(err.upstream_message(), None);
    }
}
