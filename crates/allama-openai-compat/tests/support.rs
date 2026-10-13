//! Helpers the conformance tests share.

#![allow(dead_code)]

use allama_conformance::stream_errors::ErrorKindProbe;
use serde_json::{Value, json};

/// The OpenAI error object for each probed kind, told apart by `code`.
/// `quota_code` is the code the endpoint reports quota exhaustion with.
pub fn error_object(kind: ErrorKindProbe, quota_code: &str) -> Option<Value> {
    let (message, error_type, code) = match kind {
        ErrorKindProbe::RateLimited => ("Rate limit reached", "requests", "rate_limit_exceeded"),
        ErrorKindProbe::Overloaded => (
            "The server is overloaded",
            "server_error",
            "server_is_overloaded",
        ),
        ErrorKindProbe::InvalidRequest => (
            "Invalid value for 'temperature'",
            "invalid_request_error",
            "invalid_value",
        ),
        ErrorKindProbe::Auth => (
            "Incorrect API key provided",
            "invalid_request_error",
            "invalid_api_key",
        ),
        ErrorKindProbe::QuotaExhausted => {
            ("You exceeded your current quota", quota_code, quota_code)
        }
        ErrorKindProbe::ContextLength => (
            "This model's maximum context length is 128000 tokens",
            "invalid_request_error",
            "context_length_exceeded",
        ),
        ErrorKindProbe::ModelNotFound => (
            "The model `m` does not exist",
            "invalid_request_error",
            "model_not_found",
        ),
        ErrorKindProbe::Timeout => ("The request timed out", "server_error", "timeout"),
        // A reasoning model refusing a sampling parameter: the provider's
        // own refusal (`UnsupportedStage::Provider`).
        ErrorKindProbe::Unsupported => {
            return Some(json!({
                "message": "Unsupported parameter: 'temperature' is not supported with this model.",
                "type": "invalid_request_error",
                "code": "unsupported_parameter",
                "param": "temperature"
            }));
        }
        _ => return None,
    };
    Some(json!({"message": message, "type": error_type, "code": code, "param": null}))
}

/// [`error_object`] in the OpenAI error envelope.
pub fn envelope(kind: ErrorKindProbe, quota_code: &str) -> Option<String> {
    envelope_with_upstream(kind, quota_code, "")
}

/// [`envelope`] with `upstream` text after the message, where the
/// classification reads it.
pub fn envelope_with_upstream(
    kind: ErrorKindProbe,
    quota_code: &str,
    upstream: &str,
) -> Option<String> {
    let mut error = error_object(kind, quota_code)?;
    let message = format!(
        "{} {upstream}",
        error["message"].as_str().unwrap_or_default()
    );
    error["message"] = json!(message.trim_end());
    Some(json!({"error": error}).to_string())
}
