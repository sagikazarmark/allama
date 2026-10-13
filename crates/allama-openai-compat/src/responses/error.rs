//! Error mapping for the Responses API.
//!
//! HTTP errors and errors reported inside a response or stream
//! (`response.failed`, `error` events, which carry a code but no HTTP status)
//! share the Chat Completions classification ([`crate::map_api_error`]).

use allama::{Error, StreamErrorKind};

use super::wire::ResponseError;
use crate::ApiError;

/// Map an error reported in a response body or stream event (no HTTP
/// status), with the classification shared with Chat Completions.
pub(crate) fn map_response_error(
    error: &ResponseError,
    provider: &str,
    request_id: Option<String>,
) -> Error {
    let error = ApiError {
        message: error.message.clone(),
        error_type: error.error_type.clone(),
        code: error.code.clone(),
        param: error.param.clone(),
    };
    crate::map_api_error(&error, None, request_id, None).with_provider(provider)
}

/// A stream that broke the Responses event protocol.
pub(crate) fn protocol(message: impl Into<String>) -> Error {
    Error::stream_protocol(message.into())
}

/// The SSE stream ended before a terminal event.
pub(crate) fn incomplete(provider: &str, request_id: Option<String>) -> Error {
    Error::stream(
        StreamErrorKind::Incomplete,
        "SSE stream ended before response.completed, response.incomplete, or response.failed",
    )
    .with_provider(provider)
    .with_request_id_opt(request_id.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(code: &str) -> Error {
        map_response_error(
            &ResponseError {
                code: Some(code.into()),
                message: format!("{code} happened"),
                ..ResponseError::default()
            },
            "demo",
            Some("req_1".into()),
        )
    }

    #[test]
    fn codes_map_to_typed_errors() {
        assert!(matches!(
            err("insufficient_quota"),
            Error::QuotaExhausted { .. }
        ));
        assert!(matches!(
            err("rate_limit_exceeded"),
            Error::RateLimited { .. }
        ));
        assert!(matches!(
            err("context_length_exceeded"),
            Error::ContextLengthExceeded { .. }
        ));
        assert!(matches!(
            err("invalid_prompt"),
            Error::InvalidRequest { .. }
        ));
        assert!(matches!(err("vector_store_timeout"), Error::Timeout { .. }));
        let server = err("server_error");
        assert!(matches!(server, Error::Provider { .. }));
        assert!(server.is_retryable());
        assert_eq!(server.request_id(), Some("req_1"));
        let context = server.context().unwrap();
        assert_eq!(context.provider_code.as_deref(), Some("server_error"));
        assert_eq!(context.provider.as_deref(), Some("demo"));
        assert!(!err("insufficient_quota").is_retryable());
    }
}
