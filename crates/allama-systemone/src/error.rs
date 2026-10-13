//! Error mapping for System One endpoints.
//!
//! Each server reports errors in its own envelope:
//!
//! - Ollama: `{"error": "message"}`
//! - TypeSafe (FastAPI): `{"detail": {"error_type": .., "message": ..}}`,
//!   `{"detail": [{"loc": .., "msg": .., "type": ..}]}` for validation, or
//!   `{"detail": "message"}`
//! - OpenRouter: `{"error": {"message": .., "code": ..}}`
//! - Vercel AI Gateway: `{"message": .., "error_type": ..}`
//! - Cloudflare: `{"errors": [{"message": .., "code": ..}], "success": false}`,
//!   with Workers AI's error codes
//!   (<https://developers.cloudflare.com/workers-ai/platform/errors/>)

use allama::Error;
use allama_http::{ErrorEnvelope, Headers};
use serde_json::Value;

/// Messages for a model the server does not know, reported with a status
/// other than 404 (TypeSafe answers 400 `api_usage_error`).
const UNKNOWN_MODEL: &[&str] = &["unknown model"];

/// Ollama's messages for a model or input it cannot serve with System One
/// (`server/routes.go`, `server/images.go`, `decision/`).
const UNSUPPORTED: &[&str] = &[
    "requires a local decision model",
    "not supported by system one",
    "does not support system one",
    "does not support decision",
    "does not support images",
    "does not support vision",
    "unsupported decision encoding",
];

/// Ollama's messages for input that does not fit the model's context; it
/// never truncates (`llm/llama_server_score.go`, `decision/clef.go`).
const CONTEXT_EXCEEDED: &[&str] = &[
    "input is never truncated",
    "context tokens for scoring",
    "exceeds the model context",
    "exceeds the loaded context",
];

/// Generic overflow phrases other servers use.
const CONTEXT_PHRASES: &[&str] = &["context length", "context window", "maximum context"];

/// Map an HTTP error response.
///
/// The status and headers are read with [`allama_http::error_context`]
/// (status, `x-request-id` or `request-id`, retry hint, server retry
/// directive); a server that names its request id header differently sets it
/// afterwards (`with_request_id`). An envelope's message is the
/// [upstream message](Error::upstream_message);
/// the constructor message is the adapter's own (`HTTP {status}`). A body
/// that is no envelope still informs the category, but is kept only as the
/// body.
#[must_use]
pub fn map_http_error(status: u16, headers: &Headers, body: &str) -> Error {
    map_http_error_with_hint(status, headers, body, None)
}

/// [`map_http_error`] with a preset's hint appended to the refusal for a
/// missing System One endpoint (a `404` that names no model).
pub(crate) fn map_http_error_with_hint(
    status: u16,
    headers: &Headers,
    body: &str,
    missing_endpoint_hint: Option<&str>,
) -> Error {
    let envelope = error_in_body(body);
    let structured = envelope.is_some();
    let text = envelope
        .as_ref()
        .and_then(|envelope| envelope.message.as_deref())
        .unwrap_or(body);
    let (error_type, code) = envelope.as_ref().map_or((None, None), |envelope| {
        (envelope.error_type.as_deref(), envelope.code.as_deref())
    });
    let lower = text.to_ascii_lowercase();
    let mentions = |needles: &[&str]| needles.iter().any(|needle| lower.contains(needle));
    let message = format!("HTTP {status}");

    let mapped = if let Some(error) = workers_ai_code(status, code) {
        error
    } else if mentions(UNSUPPORTED) {
        Error::unsupported_by_provider(None, message)
    } else if status == 404
        && (!structured
            || (error_type.is_none()
                && code.is_none()
                && text.trim().eq_ignore_ascii_case("not found")))
    {
        // A missing route, not a missing model: servers name unknown models
        // in their JSON envelope. Ollama before 0.35 answers a missing route
        // with plain text, FastAPI servers (TypeSafe, LiteLLM) with a bare
        // `{"detail": "Not Found"}`.
        let message = "the server has no System One endpoint here (HTTP 404)";
        Error::unsupported_by_provider(
            None,
            match missing_endpoint_hint {
                Some(hint) => format!("{message}; {hint}"),
                None => message.to_owned(),
            },
        )
    } else if matches!(status, 400 | 422) && mentions(UNKNOWN_MODEL) {
        Error::model_not_found(message)
    } else if matches!(status, 400 | 413 | 422)
        && !lower.contains("deadline")
        && (mentions(CONTEXT_EXCEEDED) || mentions(CONTEXT_PHRASES))
    {
        // Only for rejected requests: a timeout ("context deadline exceeded")
        // or a model name such as `long-context-window` is no overflow.
        Error::context_length_exceeded(message, None)
    } else {
        match status {
            // A structured 404 names a model the server does not know.
            404 => Error::model_not_found(message),
            // What the status means on its own, shared by every adapter.
            _ => allama_http::status_error(status, message),
        }
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
/// daily allocation or capacity, or a `400` or `404` that names the model.
/// The codes are Cloudflare's own, so only its statuses are read.
fn workers_ai_code(status: u16, code: Option<&str>) -> Option<Error> {
    let message = format!("HTTP {status}");
    Some(match (status, code?) {
        (429, "3036") => Error::quota_exhausted(message),
        (429, "3040") => Error::overloaded(message),
        // `7000` is the Cloudflare API's "No route for that URI": under
        // `/ai/run/` the path names no model (recorded 2026-10-13).
        (400 | 404, "5007" | "3042" | "7000") => Error::model_not_found(message),
        _ => return None,
    })
}

/// The error envelope in `body`, if it is one with a message. Gateways
/// sometimes send one with a success status.
///
/// The common envelopes (Ollama, OpenRouter, Vercel AI Gateway) are read
/// with [`allama_http::error_envelope`]; FastAPI's `detail` envelopes
/// (TypeSafe) and Cloudflare's `errors` list here.
pub(crate) fn error_in_body(body: &str) -> Option<ErrorEnvelope> {
    if let Some(mut envelope) = allama_http::error_envelope(body) {
        envelope.message.as_ref()?;
        // Vercel AI Gateway names its type `error_type`.
        if envelope.error_type.is_none() {
            envelope.error_type = serde_json::from_str::<Value>(body)
                .ok()
                .and_then(|value| text(value.get("error_type")));
        }
        return Some(envelope);
    }
    let value = serde_json::from_str::<Value>(body).ok()?;
    match value.get("detail") {
        Some(detail) => detail_envelope(detail),
        None => errors_envelope(value.get("errors")?),
    }
}

/// Cloudflare's `errors`: the first error with a message.
fn errors_envelope(errors: &Value) -> Option<ErrorEnvelope> {
    let error = errors.as_array()?.first()?;
    let mut envelope = ErrorEnvelope::default();
    envelope.message = Some(text(error.get("message"))?);
    envelope.code = text(error.get("code"));
    Some(envelope)
}

/// FastAPI's `detail`: a message, a typed object, or validation items.
fn detail_envelope(detail: &Value) -> Option<ErrorEnvelope> {
    let mut envelope = ErrorEnvelope::default();
    match detail {
        Value::String(message) => envelope.message = Some(message.clone()),
        Value::Object(object) => {
            envelope.message = Some(text(object.get("message"))?);
            envelope.error_type = text(object.get("error_type"));
        }
        Value::Array(items) => {
            let messages: Vec<String> = items.iter().filter_map(validation_item).collect();
            if messages.is_empty() {
                return None;
            }
            envelope.message = Some(messages.join("; "));
            envelope.error_type = Some("validation_error".to_owned());
            envelope.param = items.iter().find_map(location);
        }
        _ => return None,
    }
    Some(envelope)
}

/// `{"loc": ["body", "questions", "q"], "msg": "..."}` as `body.questions.q: ...`.
fn validation_item(item: &Value) -> Option<String> {
    let message = item.get("msg")?.as_str()?;
    Some(match location(item) {
        Some(location) => format!("{location}: {message}"),
        None => message.to_owned(),
    })
}

/// A validation item's `loc` as a dotted path.
fn location(item: &Value) -> Option<String> {
    let parts: Vec<String> = item
        .get("loc")?
        .as_array()?
        .iter()
        .filter_map(|part| text(Some(part)))
        .collect();
    (!parts.is_empty()).then(|| parts.join("."))
}

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
    use serde_json::json;

    fn headers(pairs: &[(&str, &str)]) -> Headers {
        pairs.iter().copied().collect()
    }

    fn http(status: u16, body: Value) -> Error {
        map_http_error(
            status,
            &headers(&[("x-request-id", "req_1")]),
            &body.to_string(),
        )
    }

    #[test]
    fn statuses_mean_what_they_mean_on_every_adapter() {
        for (status, kind) in [
            (408, "timeout"),
            (504, "timeout"),
            (524, "timeout"),
            (529, "overloaded"),
            (499, "provider"),
        ] {
            let err = map_http_error(status, &Headers::new(), "<html>gateway</html>");
            assert_eq!(err.kind_name(), kind, "HTTP {status}");
            assert_eq!(err.status(), Some(status));
        }
        assert!(!map_http_error(499, &Headers::new(), "").is_retryable());
    }

    #[test]
    fn ollama_errors() {
        let err = http(
            404,
            json!({"error": "model \"nimble\" not found, try pulling it first"}),
        );
        assert!(matches!(err, Error::ModelNotFound { .. }), "{err:?}");
        assert_eq!(err.request_id(), Some("req_1"));

        let err = http(
            400,
            json!({"error": "model \"qwen3\" does not support System One scoring; use a local decision model with a scoring-capable runner"}),
        );
        assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");

        let err = http(
            400,
            json!({"error": "criteria must contain 2–26 candidates"}),
        );
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");

        let err = http(
            413,
            json!({"error": "request body must not exceed 64 KiB without images"}),
        );
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");

        // Ollama's own messages (server/images.go, llm/llama_server_score.go,
        // decision/clef.go at ollama/ollama b0c1ca4).
        for message in [
            "llama3.2:latest does not support decision",
            "clef does not support vision",
            "unsupported decision encoding \"other\"",
            "System One requires a local decision model",
        ] {
            let err = http(400, json!({"error": message}));
            assert!(
                matches!(err, Error::Unsupported { .. }),
                "{message}: {err:?}"
            );
        }
        for message in [
            "prompt 0 has 5000 tokens; expected 1–4096 (input is never truncated)",
            "prompt 0 requires 4098 context tokens for scoring; model has 4096",
            "decision prompt exceeds the model context (input is never truncated)",
        ] {
            let err = http(400, json!({"error": message}));
            assert!(
                matches!(err, Error::ContextLengthExceeded { .. }),
                "{message}: {err:?}"
            );
        }

        let err = http(500, json!({"error": "scoring failed"}));
        assert!(matches!(err, Error::Provider { .. }), "{err:?}");
        assert!(err.is_retryable());
    }

    #[test]
    fn cloudflare_errors() {
        let envelope = |code: u32, message: &str| json!({"result": null, "success": false, "errors": [{"code": code, "message": message}], "messages": []});
        let err = http(
            429,
            envelope(3036, "You have used up your daily free allocation"),
        );
        assert!(matches!(err, Error::QuotaExhausted { .. }), "{err:?}");
        assert_eq!(
            err.context().unwrap().provider_code.as_deref(),
            Some("3036")
        );
        assert_eq!(
            err.upstream_message(),
            Some("You have used up your daily free allocation")
        );
        let err = http(429, envelope(3040, "Capacity temporarily exceeded"));
        assert!(matches!(err, Error::Overloaded { .. }), "{err:?}");
        let err = http(429, envelope(3001, "Rate limited"));
        assert!(matches!(err, Error::RateLimited { .. }), "{err:?}");
        let err = http(
            400,
            envelope(5007, "No such model @cf/cloudflare/nope or task"),
        );
        assert!(matches!(err, Error::ModelNotFound { .. }), "{err:?}");
        let err = http(400, envelope(7000, "No route for that URI"));
        assert!(matches!(err, Error::ModelNotFound { .. }), "{err:?}");
        let err = http(400, envelope(5006, "Bad input"));
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
        let err = http(
            401,
            json!({"success": false, "errors": [{"code": 10000, "message": "Authentication error"}]}),
        );
        assert!(matches!(err, Error::Auth { .. }), "{err:?}");
        // A code is Cloudflare's only on Cloudflare's statuses.
        let err = http(500, envelope(3040, "Internal error"));
        assert!(matches!(err, Error::Provider { .. }), "{err:?}");
    }

    #[test]
    fn typesafe_errors() {
        // Recorded from the live API (2026-10-02).
        let err = http(
            400,
            json!({"detail": {"error_type": "api_usage_error", "message": "Unknown model: no-such-model"}}),
        );
        assert!(matches!(err, Error::ModelNotFound { .. }), "{err:?}");
        assert_eq!(
            err.context().unwrap().provider_type.as_deref(),
            Some("api_usage_error")
        );
        let err = http(
            401,
            json!({"detail": {"error_type": "authentication_error", "message": "Cannot authenticate with the server. Please check your API key and try again."}}),
        );
        assert!(matches!(err, Error::Auth { .. }), "{err:?}");
        let err = http(
            400,
            json!({"detail": "Too many score levels. Must have at most 10 levels."}),
        );
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");

        let err = http(
            403,
            json!({"detail": {"error_type": "authentication_error", "message": "Must supply an API key! Check your request and try again."}}),
        );
        assert!(matches!(err, Error::Auth { .. }), "{err:?}");
        assert_eq!(
            err.context().unwrap().provider_type.as_deref(),
            Some("authentication_error")
        );

        let err = http(
            422,
            json!({"detail": [{"loc": ["body", "state"], "msg": "Field required", "type": "missing"}]}),
        );
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
        assert_eq!(err.message(), "HTTP 422");
        assert_eq!(err.upstream_message(), Some("body.state: Field required"));
        assert_eq!(err.context().unwrap().param.as_deref(), Some("body.state"));

        assert!(matches!(
            http(529, json!({"detail": "Overloaded"})),
            Error::Overloaded { .. }
        ));
        assert!(matches!(
            http(429, json!({"detail": "Slow down"})),
            Error::RateLimited { .. }
        ));
    }

    #[test]
    fn gateway_errors() {
        let err = http(
            402,
            json!({"error": {"message": "Insufficient credits", "code": 402}}),
        );
        assert!(matches!(err, Error::QuotaExhausted { .. }), "{err:?}");
        assert_eq!(err.context().unwrap().provider_code.as_deref(), Some("402"));

        let err = http(
            400,
            json!({"message": "bad question", "error_type": "invalid_request"}),
        );
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
        assert_eq!(err.upstream_message(), Some("bad question"));
        assert_eq!(
            err.context().unwrap().provider_type.as_deref(),
            Some("invalid_request")
        );
    }

    #[test]
    fn timeouts_and_model_names_are_not_overflow() {
        let err = http(
            500,
            json!({"error": "Post \"http://127.0.0.1:5555/completion\": context deadline exceeded"}),
        );
        assert!(matches!(err, Error::Provider { .. }), "{err:?}");
        assert!(err.is_retryable());
        let err = http(504, json!({"error": "context deadline exceeded"}));
        assert!(err.is_retryable(), "{err:?}");
        let err = http(400, json!({"error": "context deadline exceeded"}));
        assert!(
            !matches!(err, Error::ContextLengthExceeded { .. }),
            "{err:?}"
        );
        let err = http(
            404,
            json!({"error": "model 'qwen-long-context-window' not found"}),
        );
        assert!(matches!(err, Error::ModelNotFound { .. }), "{err:?}");
        let err = http(
            400,
            json!({"error": "input exceeds the maximum context length"}),
        );
        assert!(
            matches!(err, Error::ContextLengthExceeded { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn missing_endpoint_is_unsupported() {
        let err = http(404, json!({"detail": "Not Found"}));
        assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
        // Typed errors name a resource; they are not a missing route.
        let err = http(
            404,
            json!({"detail": {"error_type": "not_found_error", "message": "Not found"}}),
        );
        assert!(matches!(err, Error::ModelNotFound { .. }), "{err:?}");
        let err = map_http_error(404, &Headers::new(), "404 page not found");
        assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
        // The server answered, so the refusal is the provider's.
        assert_eq!(
            err.unsupported_stage(),
            Some(allama::UnsupportedStage::Provider)
        );
        assert!(err.message().contains("no System One endpoint"), "{err}");
        assert!(err.as_log().to_string().contains("no System One endpoint"));
        assert!(!err.as_log().to_string().contains("404 page"));
    }

    #[test]
    fn non_json_bodies_are_kept_out_of_the_message() {
        let err = map_http_error(502, &Headers::new(), "<html>my SSN is 123-45-6789</html>");
        assert_eq!(err.message(), "HTTP 502");
        assert_eq!(err.upstream_message(), None);
        assert_eq!(
            err.context().unwrap().body.as_deref(),
            Some("<html>my SSN is 123-45-6789</html>")
        );
        assert_eq!(
            map_http_error(503, &Headers::new(), "").message(),
            "HTTP 503"
        );
    }
}
