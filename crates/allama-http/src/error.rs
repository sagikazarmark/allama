//! Shared helpers for mapping HTTP responses to [`allama::Error`].
//!
//! Every HTTP provider shares:
//!
//! - what a status means on its own ([`status_error`]), which each provider
//!   refines from its own error codes and body;
//! - reading the status line and headers into an [`ErrorContext`] (see
//!   [`error_context`]);
//! - reading the common JSON error envelopes, keeping the provider's text
//!   out of the redacted [`ErrorLog`](allama::ErrorLog) (see
//!   [`error_envelope`]);
//! - the error for a success response whose body cannot be decoded
//!   ([`malformed_response`]).

use allama::{Error, ErrorContext};
use serde_json::{Map, Value};

use crate::Headers;

/// The response headers that carry a request id, tried in order: OpenAI and
/// most compatible servers send `x-request-id`, Anthropic `request-id`.
const REQUEST_ID_HEADERS: [&str; 2] = ["x-request-id", "request-id"];

/// The diagnostic context of an HTTP error response, read from its status
/// and headers:
///
/// - [`status`](ErrorContext::status);
/// - [`request_id`](ErrorContext::request_id) from `x-request-id` or
///   `request-id`;
/// - [`retry_after`](ErrorContext::retry_after) from `Retry-After` or
///   `retry-after-ms` (see [`Headers::retry_after`]);
/// - [`server_retry_directive`](ErrorContext::server_retry_directive) from
///   `x-should-retry: true` or `false`; any other value is no directive.
///
/// Every bundled HTTP provider builds its error context here, so a header
/// rule added later reaches third-party providers that call it too. Attach
/// it with [`Error::with_context`], which merges: it sets the fields read
/// here, keeps the others, and never changes the
/// [message origin](Error::authored). A provider that names its request id
/// or retry hint header differently sets that field after attaching this
/// context (`with_request_id`, `with_retry_after`), so a standard header
/// does not overwrite it.
///
/// ```
/// use allama::Error;
/// use allama_http::{Headers, error_context, error_envelope, status_error};
///
/// fn map_http_error(status: u16, headers: &Headers, body: &str) -> Error {
///     let error = status_error(status, format!("HTTP {status}"));
///     error_envelope(body)
///         .unwrap_or_default()
///         .apply(error.with_context(error_context(status, headers)).authored())
///         .with_body(body.to_owned())
/// }
///
/// let headers: Headers = [
///     ("x-request-id", "req_1"),
///     ("retry-after", "2"),
///     ("x-should-retry", "false"),
/// ]
/// .into_iter()
/// .collect();
/// let err = map_http_error(429, &headers, r#"{"error": {"message": "slow down"}}"#);
/// assert_eq!(err.status(), Some(429));
/// assert_eq!(err.request_id(), Some("req_1"));
/// assert_eq!(err.retry_after(), Some(std::time::Duration::from_secs(2)));
/// assert_eq!(err.server_retry_directive(), Some(false));
/// assert_eq!(err.upstream_message(), Some("slow down"));
/// ```
#[must_use]
pub fn error_context(status: u16, headers: &Headers) -> ErrorContext {
    let mut context = ErrorContext::new();
    context.status = Some(status);
    context.request_id = REQUEST_ID_HEADERS
        .iter()
        .filter_map(|name| headers.get(name))
        .find_map(non_empty);
    context.retry_after = headers.retry_after("retry-after");
    context.server_retry_directive = headers.get("x-should-retry").and_then(retry_directive);
    context
}

/// The error an HTTP error status indicates on its own, before the body is
/// read.
///
/// | Status | Error |
/// |---|---|
/// | 3xx | [`Error::UnexpectedResponse`], not retryable (a redirect, which is never followed: see [`redirect_error`](crate::redirect_error)) |
/// | 401, 403 | [`Error::Auth`] |
/// | 402 | [`Error::QuotaExhausted`] |
/// | 408, 504, 524 | [`Error::Timeout`] |
/// | 429 | [`Error::RateLimited`] |
/// | 499 | [`Error::Provider`], not retryable |
/// | other 4xx | [`Error::InvalidRequest`] |
/// | 503, 529 | [`Error::Overloaded`] |
/// | 501, 505 | [`Error::Provider`], not retryable |
/// | anything else | [`Error::Provider`], retryable when 5xx |
///
/// Every bundled HTTP provider falls back to it after its own error codes
/// and body, so one status means the same thing on every provider. Refine
/// it from what the body says: a quota code on a `429` is
/// [`Error::QuotaExhausted`], a `404` that names the model is
/// [`Error::ModelNotFound`] (a bare `404` may be a wrong route, so it stays
/// [`Error::InvalidRequest`]), and a `400` about the prompt's length is
/// [`Error::ContextLengthExceeded`].
///
/// - `408` (the server gave up waiting for the request), `504` (a gateway
///   gave up waiting for the model) and `524` (Cloudflare's origin timeout)
///   are timeouts, which retry policies retry. A `504` or `524` can come
///   after the model ran, so the request may still be billed.
/// - `529` is Anthropic's overload status.
/// - `499` means the client closed the request (nginx; `CANCELLED` in
///   `google.rpc.Code`). It is a [`Error::Provider`] with the status, which
///   is not retryable: the cancellation came from the caller's side, so it
///   is neither retried nor reported as a rejected request.
/// - `501` (Not Implemented) and `505` (HTTP Version Not Supported) are
///   [`Error::Provider`]s that are not retryable: the server answers them
///   the same way every time. A `501` stays a provider error rather than
///   [`Error::Unsupported`], whose `Provider` stage comes only from an
///   adapter's own table of the replies it recognises (ADR 0003).
///
/// `message` is the adapter's own text, such as `HTTP {status}`. The error
/// carries the status and nothing else: add the rest of its context with
/// [`error_context`] and [`Error::with_context`], and mark the message
/// [`authored`](Error::authored).
///
/// ```
/// use allama::Error;
/// use allama_http::status_error;
///
/// assert!(matches!(status_error(504, "HTTP 504"), Error::Timeout { .. }));
/// assert!(matches!(status_error(529, "HTTP 529"), Error::Overloaded { .. }));
/// let cancelled = status_error(499, "HTTP 499");
/// assert!(matches!(cancelled, Error::Provider { .. }));
/// assert_eq!(cancelled.status(), Some(499));
/// assert!(!cancelled.is_retryable());
/// assert!(!status_error(501, "HTTP 501").is_retryable());
/// assert!(status_error(502, "HTTP 502").is_retryable());
/// assert!(matches!(status_error(307, "HTTP 307"), Error::UnexpectedResponse { .. }));
/// ```
#[must_use]
pub fn status_error(status: u16, message: impl Into<String>) -> Error {
    let message = message.into();
    let error = match status {
        300..=399 => Error::unexpected_response(message),
        401 | 403 => Error::auth(message),
        402 => Error::quota_exhausted(message),
        408 | 504 | 524 => Error::timeout(message),
        429 => Error::rate_limited(message),
        499 => Error::provider(message),
        400..=498 => Error::invalid_request(message),
        503 | 529 => Error::overloaded(message),
        _ => Error::provider(message),
    };
    error.with_status(status)
}

/// Recognise a JSON error envelope in an error response body or an in-band
/// error event.
///
/// Recognised shapes:
///
/// - OpenAI: `{"error": {"message", "type", "code", "param"}}`. Current vLLM
///   and Ollama's OpenAI-compatible `/v1` endpoints send it too (vLLM with a
///   numeric `code`), and Anthropic's `{"type": "error", "error": {"type",
///   "message"}}` reads the same way.
/// - Legacy vLLM, and other bare error objects:
///   `{"object": "error", "message", "type", "param", "code"}`.
/// - Native Ollama: `{"error": "message"}`.
///
/// Anything else, including an empty or non-JSON body, returns `None`.
///
/// The envelope's `message` is the provider's own text. Attach it with
/// [`ErrorEnvelope::apply`], which puts it in
/// [`Error::with_upstream_message`]. Never put it in the constructor message
/// of an [`authored`](Error::authored) error: upstream text may echo prompts
/// or input values. The constructor message is the adapter's own text; for
/// HTTP errors the bundled providers use `HTTP {status}`, with or without an
/// envelope:
///
/// ```
/// use allama::Error;
/// use allama_http::{error_envelope, status_error};
///
/// fn map_http_error(status: u16, body: &str) -> Error {
///     let envelope = error_envelope(body).unwrap_or_default();
///     envelope
///         .apply(status_error(status, format!("HTTP {status}")).authored())
///         .with_body(body.to_owned())
/// }
///
/// let err = map_http_error(
///     400,
///     r#"{"error": {"message": "bad value 'my SSN is 123-45-6789'", "param": "messages.[0].content"}}"#,
/// );
/// assert_eq!(
///     err.as_log().to_string(),
///     r#"kind=invalid_request status=400 param=messages.[0].content body_len=92 message="HTTP 400""#,
/// );
/// assert_eq!(err.upstream_message(), Some("bad value 'my SSN is 123-45-6789'"));
///
/// // An unrecognised body never reaches the message.
/// let err = map_http_error(502, "<html>my SSN is 123-45-6789</html>");
/// assert_eq!(err.to_string(), "provider error (502): HTTP 502");
/// ```
///
/// A caller that already parsed the body (an NDJSON line, an SSE event's
/// data) reads it with [`error_envelope_value`].
#[must_use]
pub fn error_envelope(body: &str) -> Option<ErrorEnvelope> {
    let value = serde_json::from_str::<Value>(body.trim()).ok()?;
    error_envelope_value(&value)
}

/// [`error_envelope`] of a body already parsed as JSON, for a caller that
/// parsed it to tell an error from its other frames (an NDJSON line, an SSE
/// event's data). It recognises the same shapes, listed on
/// [`error_envelope`].
///
/// ```
/// use allama_http::error_envelope_value;
/// use serde_json::json;
///
/// let line = json!({"error": "model 'llama9' not found"});
/// let envelope = error_envelope_value(&line).unwrap();
/// assert_eq!(envelope.message.as_deref(), Some("model 'llama9' not found"));
/// assert!(error_envelope_value(&json!({"message": {"role": "assistant"}})).is_none());
/// ```
#[must_use]
pub fn error_envelope_value(value: &Value) -> Option<ErrorEnvelope> {
    let Value::Object(object) = value else {
        return None;
    };
    match object.get("error") {
        Some(Value::Object(error)) => Some(ErrorEnvelope::from_object(error)),
        Some(Value::String(message)) => Some(ErrorEnvelope {
            message: non_empty(message),
            ..ErrorEnvelope::default()
        }),
        Some(_) => None,
        None if object.get("message").is_some_and(Value::is_string) => {
            Some(ErrorEnvelope::from_object(object))
        }
        None => None,
    }
}

/// The error for a success response whose body could not be decoded: an
/// HTML page from a proxy, a truncated body, or JSON of another shape.
///
/// It is an [`Error::UnexpectedResponse`], which is neither retried nor
/// failed over, with the `status`, the body (capped, and only its length
/// reaches the log) and the decoder's error as the
/// [upstream message](Error::upstream_message), since a decoder's text can
/// quote the body. Add the provider and the request id, so the failure can
/// be traced to a response:
///
/// ```
/// use allama::Error;
/// use allama_http::malformed_response;
///
/// let body = b"<html>502 Bad Gateway</html>";
/// let decoded = serde_json::from_slice::<serde_json::Value>(body);
/// let err = malformed_response(200, body, decoded.unwrap_err())
///     .with_provider("acme")
///     .with_request_id("req_1");
/// assert!(matches!(err, Error::UnexpectedResponse { .. }));
/// assert_eq!(err.status(), Some(200));
/// assert_eq!(err.request_id(), Some("req_1"));
/// assert_eq!(
///     err.as_log().to_string(),
///     r#"kind=unexpected_response provider=acme status=200 request_id=req_1 body_len=28 message="the response body could not be decoded""#,
/// );
/// ```
///
/// [`Error::Serialization`] is for failures of the caller's own
/// (de)serialization, never for what a provider sent.
#[must_use]
pub fn malformed_response(status: u16, body: &[u8], err: impl std::fmt::Display) -> Error {
    Error::unexpected_response("the response body could not be decoded")
        .authored()
        .with_status(status)
        .with_upstream_message(err.to_string())
        .with_body_opt(
            (!body.is_empty())
                .then(|| String::from_utf8_lossy(body))
                .as_deref(),
        )
}

/// The fields of a recognised JSON error envelope (see [`error_envelope`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ErrorEnvelope {
    /// The provider's error text. Upstream text: it belongs in
    /// [`Error::with_upstream_message`], never in an authored message.
    pub message: Option<String>,
    /// Provider error code. Numeric codes are kept as their decimal text.
    pub code: Option<String>,
    /// Provider error type or category.
    pub error_type: Option<String>,
    /// The path of the rejected request field, such as
    /// `messages.[1].content`.
    pub param: Option<String>,
}

impl ErrorEnvelope {
    /// Attach the envelope to `error`: the message as its
    /// [upstream message](Error::with_upstream_message), the code, type and
    /// rejected field as its provider code, provider type and
    /// [`param`](Error::with_param).
    #[must_use]
    pub fn apply(self, error: Error) -> Error {
        error
            .with_upstream_message_opt(self.message.as_deref())
            .with_provider_code_opt(self.code.as_deref())
            .with_provider_type_opt(self.error_type.as_deref())
            .with_param_opt(self.param.as_deref())
    }

    fn from_object(object: &Map<String, Value>) -> Self {
        Self {
            message: object
                .get("message")
                .and_then(Value::as_str)
                .and_then(non_empty),
            code: object.get("code").and_then(scalar_text),
            error_type: object.get("type").and_then(scalar_text),
            param: object.get("param").and_then(scalar_text),
        }
    }
}

/// A string or number as text; `None` for null, empty strings, and other
/// values.
fn scalar_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => non_empty(text),
        Value::Number(number) => Some(number.to_string()),
        _ => None,
    }
}

fn non_empty(text: &str) -> Option<String> {
    (!text.trim().is_empty()).then(|| text.to_owned())
}

/// An `x-should-retry` value: `true` or `false`, ignoring case and
/// surrounding whitespace.
fn retry_directive(value: &str) -> Option<bool> {
    let value = value.trim();
    if value.eq_ignore_ascii_case("true") {
        Some(true)
    } else if value.eq_ignore_ascii_case("false") {
        Some(false)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope(message: &str, code: Option<&str>, error_type: Option<&str>) -> ErrorEnvelope {
        ErrorEnvelope {
            message: Some(message.into()),
            code: code.map(Into::into),
            error_type: error_type.map(Into::into),
            param: None,
        }
    }

    fn headers(pairs: &[(&str, &str)]) -> Headers {
        pairs.iter().copied().collect()
    }

    #[test]
    fn error_context_reads_status_and_headers() {
        let context = error_context(
            409,
            &headers(&[
                ("X-Request-Id", "req_1"),
                ("Retry-After", "3"),
                ("X-Should-Retry", "true"),
            ]),
        );
        assert_eq!(context.status, Some(409));
        assert_eq!(context.request_id.as_deref(), Some("req_1"));
        assert_eq!(context.retry_after, Some(std::time::Duration::from_secs(3)));
        assert_eq!(context.server_retry_directive, Some(true));
        // Nothing else: classification and the body stay with the provider.
        let mut rest = context;
        rest.status = None;
        rest.request_id = None;
        rest.retry_after = None;
        rest.server_retry_directive = None;
        assert_eq!(rest, ErrorContext::new());

        let mut bare = error_context(500, &Headers::new());
        assert_eq!(bare.status, Some(500));
        bare.status = None;
        assert_eq!(bare, ErrorContext::new());
    }

    #[test]
    fn error_context_request_id_headers() {
        let anthropic = error_context(429, &headers(&[("request-id", "req_a")]));
        assert_eq!(anthropic.request_id.as_deref(), Some("req_a"));
        let both = error_context(
            429,
            &headers(&[("request-id", "req_a"), ("x-request-id", "req_x")]),
        );
        assert_eq!(both.request_id.as_deref(), Some("req_x"));
        let empty = error_context(
            429,
            &headers(&[("x-request-id", " "), ("request-id", "req_a")]),
        );
        assert_eq!(empty.request_id.as_deref(), Some("req_a"));
    }

    #[test]
    fn error_context_retry_directive_values() {
        for (value, expected) in [
            ("true", Some(true)),
            ("false", Some(false)),
            (" TRUE ", Some(true)),
            ("False", Some(false)),
            ("1", None),
            ("yes", None),
            ("", None),
        ] {
            let context = error_context(503, &headers(&[("x-should-retry", value)]));
            assert_eq!(context.server_retry_directive, expected, "{value:?}");
        }
        let context = error_context(429, &headers(&[("retry-after-ms", "1500")]));
        assert_eq!(
            context.retry_after,
            Some(std::time::Duration::from_millis(1500))
        );
    }

    #[test]
    fn status_error_kinds() {
        for (status, kind) in [
            (400, "invalid_request"),
            (401, "auth"),
            (402, "quota_exhausted"),
            (403, "auth"),
            (404, "invalid_request"),
            (408, "timeout"),
            (409, "invalid_request"),
            (413, "invalid_request"),
            (422, "invalid_request"),
            (429, "rate_limited"),
            (499, "provider"),
            (500, "provider"),
            (502, "provider"),
            (503, "overloaded"),
            (504, "timeout"),
            (524, "timeout"),
            (529, "overloaded"),
            (600, "provider"),
        ] {
            let err = status_error(status, format!("HTTP {status}"));
            assert_eq!(err.kind_name(), kind, "HTTP {status}");
            assert_eq!(err.status(), Some(status));
            assert_eq!(err.message(), format!("HTTP {status}"));
        }
    }

    #[test]
    fn a_closed_request_is_neither_retried_nor_blamed_on_the_request() {
        let err = status_error(499, "HTTP 499");
        assert!(matches!(err, Error::Provider { .. }), "{err:?}");
        assert!(!err.is_retryable());
        assert!(!err.is_transient());
    }

    #[test]
    fn timeouts_are_retryable() {
        for status in [408, 504, 524] {
            assert!(status_error(status, "x").is_retryable(), "{status}");
        }
    }

    #[test]
    fn malformed_responses_keep_the_body_out_of_the_log() {
        let body = b"<html>my SSN is 123-45-6789</html>";
        let err = malformed_response(200, body, "expected value at line 1 column 1");
        assert!(matches!(err, Error::UnexpectedResponse { .. }), "{err:?}");
        assert!(!err.is_retryable());
        assert_eq!(err.status(), Some(200));
        assert_eq!(
            err.upstream_message(),
            Some("expected value at line 1 column 1")
        );
        assert_eq!(
            err.context().unwrap().body.as_deref(),
            Some("<html>my SSN is 123-45-6789</html>")
        );
        assert!(!err.as_log().to_string().contains("SSN"));
        assert_eq!(
            malformed_response(200, b"", "EOF").context().unwrap().body,
            None
        );
    }

    #[test]
    fn openai_shape() {
        let parsed = error_envelope(
            r#"{"error": {"message": "Invalid value", "type": "invalid_request_error", "code": "invalid_value", "param": "messages.[1].content"}}"#,
        )
        .unwrap();
        assert_eq!(
            parsed,
            ErrorEnvelope {
                param: Some("messages.[1].content".into()),
                ..envelope(
                    "Invalid value",
                    Some("invalid_value"),
                    Some("invalid_request_error")
                )
            }
        );
    }

    #[test]
    fn current_vllm_and_ollama_v1_shape() {
        // vLLM `ErrorInfo`: the OpenAI shape with a numeric code.
        let vllm = error_envelope(
            r#"{"error": {"message": "max_tokens must be at least 1", "type": "BadRequestError", "param": null, "code": 400}}"#,
        )
        .unwrap();
        assert_eq!(
            vllm,
            envelope(
                "max_tokens must be at least 1",
                Some("400"),
                Some("BadRequestError")
            )
        );

        // Ollama `/v1` (`openai.NewError`).
        let ollama = error_envelope(
            r#"{"error": {"message": "model \"x\" not found", "type": "api_error", "param": null, "code": null}}"#,
        )
        .unwrap();
        assert_eq!(
            ollama,
            envelope("model \"x\" not found", None, Some("api_error"))
        );
    }

    #[test]
    fn legacy_vllm_shape() {
        let parsed = error_envelope(
            r#"{"object": "error", "message": "bad request", "type": "BadRequestError", "param": null, "code": 400}"#,
        )
        .unwrap();
        assert_eq!(
            parsed,
            envelope("bad request", Some("400"), Some("BadRequestError"))
        );
    }

    #[test]
    fn native_ollama_shape() {
        let parsed = error_envelope(r#"{"error": "model 'x' not found"}"#).unwrap();
        assert_eq!(parsed, envelope("model 'x' not found", None, None));
    }

    #[test]
    fn a_parsed_body_reads_as_its_text() {
        let bodies = [
            r#"{"error": {"message": "bad", "type": "invalid_request_error", "code": "x"}}"#,
            r#"{"object": "error", "message": "bad", "type": "BadRequestError", "code": 400}"#,
            r#"{"error": "model 'x' not found"}"#,
            r#"{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}"#,
            r#"{"message": {"role": "assistant", "content": "hi"}}"#,
            r#"{"error": 3}"#,
            r#"["error"]"#,
            r#""error""#,
        ];
        for body in bodies {
            let value: Value = serde_json::from_str(body).unwrap();
            assert_eq!(error_envelope_value(&value), error_envelope(body), "{body}");
        }
    }

    #[test]
    fn anthropic_shape_reads_as_openai() {
        let parsed = error_envelope(
            r#"{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}, "request_id": "req_1"}"#,
        )
        .unwrap();
        assert_eq!(
            parsed,
            envelope("Overloaded", None, Some("overloaded_error"))
        );
    }

    #[test]
    fn unknown_bodies_are_not_envelopes() {
        for body in [
            "",
            "   ",
            "<html>my SSN is 123-45-6789</html>",
            "upstream connect error",
            "[1, 2]",
            r#"{"detail": "Not Found"}"#,
            r#"{"error": 42}"#,
            r#"{"message": 42}"#,
        ] {
            assert_eq!(error_envelope(body), None, "{body}");
        }
    }

    #[test]
    fn empty_messages_are_absent() {
        let parsed =
            error_envelope(r#"{"error": {"message": "", "code": "server_error"}}"#).unwrap();
        assert_eq!(parsed.message, None);
        assert_eq!(parsed.code.as_deref(), Some("server_error"));
    }

    #[test]
    fn apply_keeps_upstream_text_out_of_the_log() {
        let err = error_envelope(
            r#"{"error": {"message": "SENTINEL", "type": "invalid_request_error", "code": "bad", "param": "tools[0].function.parameters"}}"#,
        )
        .unwrap()
        .apply(Error::invalid_request("HTTP 400").authored());
        assert_eq!(err.upstream_message(), Some("SENTINEL"));
        let context = err.context().unwrap();
        assert_eq!(context.provider_code.as_deref(), Some("bad"));
        assert_eq!(
            context.provider_type.as_deref(),
            Some("invalid_request_error")
        );
        assert_eq!(
            context.param.as_deref(),
            Some("tools[0].function.parameters")
        );
        assert_eq!(
            err.as_log().to_string(),
            r#"kind=invalid_request provider_code=bad provider_type=invalid_request_error param=tools[0].function.parameters message="HTTP 400""#
        );
    }
}
