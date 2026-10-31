//! Maps Anthropic HTTP errors, mid-stream `error` events, and transport
//! failures to typed [`allama::Error`]s.

use allama::{Error, RequestField, Subject};
use allama_http::TransportError;
#[cfg(test)]
use allama_http::TransportErrorKind;
use allama_http::http::{HeaderMap, StatusCode};

use crate::PROVIDER_NAME;
use crate::wire;

/// `error.details.error_code` values that mean billing or spend exhaustion.
const QUOTA_ERROR_CODES: &[&str] = &["enforced_spend_limit_reached"];

/// A request that failed before a response arrived.
pub(crate) fn map_transport_error(err: TransportError) -> Error {
    err.into_error().with_provider(PROVIDER_NAME)
}

/// Reading the response body failed.
pub(crate) fn map_body_read_error(err: TransportError, request_id: Option<String>) -> Error {
    // The transport's own mapping, as on every other adapter: a timeout
    // stays a timeout, a body past `max_response_size` is an unexpected
    // response, and any other failed read a retryable transport error.
    err.into_error()
        .with_provider(PROVIDER_NAME)
        .with_request_id_opt(request_id.as_deref())
}

/// A success response's body did not match the Messages API shape
/// ([`allama_http::malformed_response`]).
pub(crate) fn map_response_decode_error(
    status: StatusCode,
    body: &[u8],
    err: serde_json::Error,
    request_id: Option<String>,
) -> Error {
    allama_http::malformed_response(status, body, err)
        .with_provider(PROVIDER_NAME)
        .with_request_id_opt(request_id.as_deref())
}

/// Map a mid-stream `error` event.
///
/// Uses the same classification as HTTP errors, so retry and fallback
/// policies behave the same whenever the error arrives.
pub(crate) fn map_stream_api_error(
    error: &wire::ErrorResponse,
    request_id: Option<String>,
) -> Error {
    let api = &error.error;
    let code = api.details.as_ref().and_then(|d| d.error_code.clone());
    classify(None, &api.error_type, &api.message, code.as_deref(), None)
        .authored()
        .with_upstream_message(api.message.clone())
        .with_provider_type_opt((!api.error_type.is_empty()).then_some(api.error_type.as_str()))
        .with_provider_code_opt(code.as_deref())
        .with_provider(PROVIDER_NAME)
        .with_request_id_opt(error.request_id.as_deref().or(request_id.as_deref()))
}

/// Map an HTTP error response.
///
/// The status and headers are read with [`allama_http::error_context`], the
/// envelope with [`allama_http::error_envelope`]; its message is the
/// upstream message. A body that is no envelope still informs the category,
/// but is kept only as the body. `prefill` is the index of the assistant
/// message the caller sent last, which a refusal of a prefill names.
pub(crate) fn map_http_error(
    status: StatusCode,
    headers: &HeaderMap,
    body: &str,
    prefill: Option<usize>,
) -> Error {
    let context = allama_http::error_context(status, headers);
    let error = match allama_http::error_envelope(body) {
        Some(envelope) => {
            // Anthropic extras beside the common envelope.
            let json = serde_json::from_str::<serde_json::Value>(body).unwrap_or_default();
            let code = json
                .pointer("/error/details/error_code")
                .and_then(serde_json::Value::as_str);
            let body_request_id = json
                .get("request_id")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned);
            let error_type = envelope.error_type.clone().unwrap_or_default();
            let message = envelope.message.clone().unwrap_or_default();
            let code = code.map(str::to_owned);
            let request_id = context.request_id.clone().or(body_request_id);
            let error = classify(
                Some(status),
                &error_type,
                &message,
                code.as_deref(),
                prefill,
            );
            envelope
                .apply(error.with_context(context).authored())
                .with_provider_code_opt(code.as_deref())
                .with_request_id_opt(request_id.as_deref())
        }
        None => classify(Some(status), "", body, None, prefill)
            .with_context(context)
            .authored(),
    };
    error
        .with_provider(PROVIDER_NAME)
        .with_body_opt((!body.is_empty()).then_some(body))
}

/// Classify by Anthropic error type first, then by HTTP status, reading
/// `text` (the upstream message, or an unrecognised body) for the
/// conditions Anthropic reports only in prose. The result carries only the
/// constructor message; the caller adds the context and marks it
/// [authored](Error::authored).
fn classify(
    status: Option<StatusCode>,
    error_type: &str,
    text: &str,
    code: Option<&str>,
    prefill: Option<usize>,
) -> Error {
    let message = match status {
        Some(status) => format!("HTTP {}", status.as_u16()),
        None => "Anthropic stream reported an error".to_owned(),
    };

    if code.is_some_and(|c| QUOTA_ERROR_CODES.contains(&c))
        || error_type == "billing_error"
        || status == Some(StatusCode::PAYMENT_REQUIRED)
        || is_quota_message(text)
    {
        Error::quota_exhausted(message)
    } else {
        match (error_type, status) {
            ("authentication_error" | "permission_error", _) => Error::auth(message),
            ("rate_limit_error", _) => Error::rate_limited(message),
            ("overloaded_error", _) => Error::overloaded(message),
            ("timeout_error", _) => Error::timeout(message),
            ("not_found_error", _) => {
                refine_rejected(Error::invalid_request(message), text, prefill)
            }
            ("api_error", _) => Error::provider(message),
            ("invalid_request_error" | "request_too_large", _) => {
                refine_rejected(Error::invalid_request(message), text, prefill)
            }
            // What the status means on its own, shared by every adapter.
            (_, Some(status)) => {
                refine_rejected(allama_http::status_error(status, message), text, prefill)
            }
            (_, None) => Error::provider(message),
        }
    }
}

/// A rejected request, refined by the conditions Anthropic reports only in
/// prose: a context overflow, an unknown model, content filtering, a
/// refusal of a request field ([`upstream_refusal`]), or a refusal of the
/// assistant message the caller sent last ([`is_prefill_refusal`]), which
/// `prefill` indexes. Anything else is returned unchanged.
fn refine_rejected(error: Error, text: &str, prefill: Option<usize>) -> Error {
    let Error::InvalidRequest { message, .. } = &error else {
        return error;
    };
    let message = message.clone();
    if let Some(context_window) = context_length_exceeded(text) {
        Error::context_length_exceeded(message, context_window)
    } else if is_model_not_found_message(text) {
        Error::model_not_found(message)
    } else if text.to_ascii_lowercase().contains("content filtering") {
        Error::content_filtered(message)
    } else if let Some((field, refusal)) = upstream_refusal(text) {
        Error::unsupported_by_provider(
            Some(Subject::Field { field }),
            format!("{message}: {refusal}"),
        )
    } else if is_prefill_refusal(text) {
        Error::unsupported_by_provider(
            prefill.map(|message| Subject::Content {
                message,
                block: None,
            }),
            format!("{message}: {PREFILL_REFUSAL}"),
        )
    } else {
        error
    }
}

/// What a refused prefill means, and how to avoid it.
const PREFILL_REFUSAL: &str = "the model does not take an assistant message last (a prefill) with \
     this request; end the conversation with a user message";

/// Whether Anthropic refuses an assistant message sent last, a prefill.
/// Each wording is pinned by a fixture in this module's tests.
///
/// - "This model does not support assistant message prefill. The
///   conversation must end with a user message." (Claude Sonnet 4.6 and
///   later generations,
///   <https://platform.claude.com/docs/en/about-claude/models/migration-guide>).
/// - "Your API request included an `assistant` message in the final
///   position, which would pre-fill the `assistant` response. When using
///   output format, pre-filling the `assistant` response is not supported."
///   (a prefill with structured output on an earlier model,
///   <https://platform.claude.com/docs/en/build-with-claude/structured-outputs>;
///   other features, such as extended thinking, are worded the same way).
///
/// It is not a refusal of the response format: an extraction does not fall
/// back to another mode for it, which would send the prefill again.
fn is_prefill_refusal(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("does not support assistant message prefill")
        || (lower.contains("would pre-fill the `assistant` response")
            && lower.contains("not supported"))
}

/// The sampling parameters a model can refuse outright, by their wire name,
/// with the portable field each one carries and the authored refusal.
const DEPRECATED_SAMPLING: &[(&str, RequestField, &str)] = &[
    (
        "temperature",
        RequestField::Temperature,
        "the model does not take `temperature`",
    ),
    (
        "top_p",
        RequestField::TopP,
        "the model does not take `top_p`",
    ),
    (
        "top_k",
        RequestField::TopK,
        "the model does not take `top_k`",
    ),
];

/// The request field Anthropic refuses in a `400` it words this way, with an
/// authored message. Each pattern is pinned by a fixture in this module's
/// tests.
///
/// - "`temperature` is deprecated for this model.", the same for `top_p`
///   and `top_k` (Claude Opus 4.7 and 4.8 and every Claude 5 model, which
///   take no sampling parameter,
///   <https://platform.claude.com/docs/en/about-claude/models/migration-guide>):
///   the sampling field the message names.
/// - "`temperature` may only be set to 1 when thinking is enabled" and
///   "`temperature` and `top_p` cannot both be specified for this model"
///   (<https://docs.claude.com/en/docs/build-with-claude/extended-thinking#feature-compatibility>:
///   thinking constrains sampling): the sampling field.
/// - "Schema is too complex for compilation"
///   (<https://docs.claude.com/en/docs/build-with-claude/structured-outputs>):
///   the response format.
/// - "tool_choice: type \"tool\" and \"any\" are not supported for this
///   model." (a model that rejects forced tool use,
///   <https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide#forced-tool-use>):
///   the tool choice.
/// - "\"thinking.type.enabled\" is not supported for this model" (a fixed
///   budget on a model that takes adaptive thinking only), the same for
///   `"thinking.type.disabled"` (a model that cannot turn thinking off) and
///   `"thinking.type.between_tools"`, "adaptive thinking is not supported on
///   this model", and "To turn thinking off on this model, send
///   \"thinking\": {\"type\": \"between_tools\"} instead of {\"type\":
///   \"disabled\"}"
///   (<https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting>):
///   the reasoning configuration.
fn upstream_refusal(text: &str) -> Option<(RequestField, &'static str)> {
    let lower = text.to_ascii_lowercase();
    if let Some((_, field, refusal)) = DEPRECATED_SAMPLING.iter().find(|(parameter, ..)| {
        lower.contains(&format!("`{parameter}` is deprecated for this model"))
    }) {
        Some((*field, *refusal))
    } else if lower.contains("temperature") && lower.contains("may only be set to 1 when thinking")
    {
        Some((
            RequestField::Temperature,
            "the model takes `temperature` only at 1 while thinking is on",
        ))
    } else if lower.contains("temperature")
        && lower.contains("top_p")
        && lower.contains("cannot both be specified")
    {
        Some((
            RequestField::TopP,
            "the model does not take `temperature` and `top_p` together",
        ))
    } else if lower.contains("schema is too complex for compilation") {
        Some((
            RequestField::ResponseFormat,
            "the schema is too complex for the model's structured output",
        ))
    } else if lower.contains("tool_choice") && lower.contains("not supported for this model") {
        Some((
            RequestField::ToolChoice,
            "the model does not take a forced tool choice",
        ))
    } else if (lower.contains("\"thinking.type.") && lower.contains("not supported for this model"))
        || lower.contains("adaptive thinking is not supported on this model")
        || lower.contains("to turn thinking off on this model")
    {
        Some((
            RequestField::Reasoning,
            "the model does not take this thinking configuration",
        ))
    } else {
        None
    }
}

/// Billing exhaustion (credit balance) and workspace spend limits are not
/// fixed by retrying.
fn is_quota_message(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("credit balance is too low")
        || lower.contains("specified api usage limits")
        || lower.contains("spend limit")
}

/// `Some(max)` when the message reports a context overflow; `max` is the
/// documented limit when the message states it
/// (`prompt is too long: 208310 tokens > 200000 maximum`).
fn context_length_exceeded(message: &str) -> Option<Option<u64>> {
    let lower = message.to_ascii_lowercase();
    if !(lower.contains("prompt is too long")
        || lower.contains("too many tokens")
        || lower.contains("exceeds the context window")
        || lower.contains("context window"))
    {
        return None;
    }
    let max = lower
        .split_once('>')
        .and_then(|(_, rest)| rest.split_whitespace().next())
        .and_then(|n| n.parse::<u64>().ok());
    Some(max)
}

fn is_model_not_found_message(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.starts_with("model:")
        || (lower.contains("model")
            && (lower.contains("not found") || lower.contains("does not exist")))
}

#[cfg(test)]
pub(crate) fn conformance_map_timeout_transport_error() -> Error {
    map_transport_error(TransportError::new(
        TransportErrorKind::Timeout,
        "request timed out",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use allama_http::http::StatusCode;
    use serde_json::json;

    /// A response to a request that ends on a user message.
    fn map_http_error(status: u16, headers: &HeaderMap, body: &str) -> Error {
        super::map_http_error(StatusCode::from_u16(status).unwrap(), headers, body, None)
    }

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        pairs
            .iter()
            .map(|(name, value)| {
                (
                    allama_http::http::HeaderName::try_from(*name).unwrap(),
                    allama_http::http::HeaderValue::try_from(*value).unwrap(),
                )
            })
            .collect()
    }

    fn body(error_type: &str, message: &str) -> String {
        json!({"type": "error", "error": {"type": error_type, "message": message}}).to_string()
    }

    fn stream_error(error_type: &str, message: &str) -> Error {
        let parsed: wire::ErrorResponse = serde_json::from_str(&body(error_type, message)).unwrap();
        map_stream_api_error(&parsed, None)
    }

    /// Anthropic's refusals of a request field, as its API words them, are
    /// the provider's own refusal of that field, over HTTP and in a stream.
    #[test]
    fn refused_fields_are_unsupported_by_the_provider() {
        let cases = [
            (
                "`temperature` may only be set to 1 when thinking is enabled. Please consult our \
                 documentation at https://docs.claude.com/en/docs/build-with-claude/extended-thinking#important-considerations-when-using-extended-thinking",
                RequestField::Temperature,
            ),
            (
                "`temperature` and `top_p` cannot both be specified for this model. Please use \
                 only one.",
                RequestField::TopP,
            ),
            (
                "Schema is too complex for compilation.",
                RequestField::ResponseFormat,
            ),
            // https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide#forced-tool-use
            (
                "tool_choice: type \"tool\" and \"any\" are not supported for this model.",
                RequestField::ToolChoice,
            ),
            // https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting
            (
                "\"thinking.type.enabled\" is not supported for this model. Use \
                 \"thinking.type.adaptive\" and \"output_config.effort\" to control thinking \
                 behavior.",
                RequestField::Reasoning,
            ),
            (
                "\"thinking.type.disabled\" is not supported for this model. Use \
                 \"thinking.type.adaptive\" and \"output_config.effort\" to control thinking \
                 behavior.",
                RequestField::Reasoning,
            ),
            (
                "\"thinking.type.disabled\" is not supported for this model. Thinking defaults \
                 to adaptive mode when not specified; use \"thinking.type.enabled\" with \
                 \"budget_tokens\" for extended thinking.",
                RequestField::Reasoning,
            ),
            (
                "\"thinking.type.between_tools\" is not supported for this model.",
                RequestField::Reasoning,
            ),
            (
                "adaptive thinking is not supported on this model",
                RequestField::Reasoning,
            ),
            (
                "To turn thinking off on this model, send \"thinking\": {\"type\": \
                 \"between_tools\"} instead of {\"type\": \"disabled\"}. The model does not \
                 think before responding. The short updates it writes between tool calls come \
                 back as thinking blocks.",
                RequestField::Reasoning,
            ),
        ];
        for (text, field) in cases {
            let http = map_http_error(400, &HeaderMap::new(), &body("invalid_request_error", text));
            let in_band = stream_error("invalid_request_error", text);
            for err in [http, in_band] {
                assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
                assert_eq!(err.unsupported_field(), Some(field), "{text}");
                assert_eq!(
                    err.unsupported_stage(),
                    Some(allama::UnsupportedStage::Provider)
                );
            }
        }
        // Other rejections stay invalid requests.
        let other = map_http_error(
            400,
            &HeaderMap::new(),
            &body("invalid_request_error", "max_tokens: Field required"),
        );
        assert!(matches!(other, Error::InvalidRequest { .. }), "{other:?}");
    }

    /// Claude Opus 4.7 and 4.8 and every Claude 5 model refuse each sampling
    /// parameter outright, in the reply recorded from `claude-sonnet-5-5`,
    /// `claude-haiku-5-5`, `claude-opus-5-5` and the others. It is the
    /// provider's refusal of that field, so a fallback fails over and a
    /// caller can drop the field it names.
    #[test]
    fn a_deprecated_sampling_parameter_is_unsupported() {
        for (parameter, field) in [
            ("temperature", RequestField::Temperature),
            ("top_p", RequestField::TopP),
            ("top_k", RequestField::TopK),
        ] {
            let upstream = format!("`{parameter}` is deprecated for this model.");
            let recorded = json!({
                "type": "error",
                "error": {"type": "invalid_request_error", "message": upstream},
                "request_id": "req_011CfzPcSSaAQ7Wd46DMkbkW"
            })
            .to_string();
            let http = map_http_error(400, &HeaderMap::new(), &recorded);
            let in_band = stream_error("invalid_request_error", &upstream);
            for err in [http, in_band] {
                assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
                assert_eq!(err.unsupported_field(), Some(field), "{parameter}");
                assert_eq!(
                    err.unsupported_stage(),
                    Some(allama::UnsupportedStage::Provider)
                );
                assert_eq!(err.upstream_message(), Some(upstream.as_str()));
                assert!(
                    err.message()
                        .ends_with(&format!("the model does not take `{parameter}`")),
                    "{err:?}"
                );
            }
        }
        // Another parameter worded the same way is not a sampling refusal.
        let other = map_http_error(
            400,
            &HeaderMap::new(),
            &body(
                "invalid_request_error",
                "`stop_sequences` is deprecated for this model.",
            ),
        );
        assert!(matches!(other, Error::InvalidRequest { .. }), "{other:?}");
    }

    /// Anthropic's refusals of a prefill, as its API words them (captured
    /// from `claude-sonnet-4-6`, and from `claude-haiku-4-5` with an output
    /// format), are the provider's own refusal of the assistant message the
    /// caller sent last, not of a request field.
    #[test]
    fn a_refused_prefill_names_the_last_message() {
        for text in [
            "This model does not support assistant message prefill. The conversation must end \
             with a user message.",
            "Your API request included an `assistant` message in the final position, which \
             would pre-fill the `assistant` response. When using output format, pre-filling the \
             `assistant` response is not supported.",
        ] {
            let err = super::map_http_error(
                StatusCode::BAD_REQUEST,
                &HeaderMap::new(),
                &body("invalid_request_error", text),
                Some(3),
            );
            assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
            assert_eq!(
                err.unsupported_subject(),
                Some(&Subject::Content {
                    message: 3,
                    block: None
                }),
                "{text}"
            );
            assert_eq!(err.unsupported_field(), None);
            assert_eq!(
                err.unsupported_stage(),
                Some(allama::UnsupportedStage::Provider)
            );
            assert_eq!(err.upstream_message(), Some(text));
            assert!(
                err.to_string()
                    .contains("end the conversation with a user message"),
                "{err}"
            );
            // In a stream, no message is named.
            let in_band = stream_error("invalid_request_error", text);
            assert!(matches!(in_band, Error::Unsupported { .. }), "{in_band:?}");
            assert_eq!(in_band.unsupported_subject(), None);
        }
    }

    #[test]
    fn maps_auth_statuses() {
        let err = map_http_error(
            401,
            &HeaderMap::new(),
            &body("authentication_error", "invalid x-api-key"),
        );
        assert!(matches!(&err, Error::Auth { message, .. } if message == "HTTP 401"));
        assert_eq!(err.upstream_message(), Some("invalid x-api-key"));
        assert_eq!(err.status(), Some(401));
        assert_eq!(
            err.context().unwrap().provider.as_deref(),
            Some("anthropic")
        );
        assert_eq!(
            err.context().unwrap().provider_type.as_deref(),
            Some("authentication_error")
        );
        let err = map_http_error(
            403,
            &HeaderMap::new(),
            &body("permission_error", "forbidden"),
        );
        assert!(matches!(err, Error::Auth { .. }));
    }

    #[test]
    fn maps_429_with_request_id_and_retry_after() {
        let err = map_http_error(
            429,
            &headers(&[("request-id", "req_123"), ("retry-after", "3")]),
            &body("rate_limit_error", "too many requests"),
        );
        assert!(matches!(err, Error::RateLimited { .. }));
        assert_eq!(err.request_id(), Some("req_123"));
        assert_eq!(err.retry_after(), Some(std::time::Duration::from_secs(3)));
        assert!(err.is_retryable());
    }

    #[test]
    fn request_id_falls_back_to_body() {
        let body = json!({
            "type": "error",
            "error": {"type": "api_error", "message": "boom"},
            "request_id": "req_body"
        })
        .to_string();
        let err = map_http_error(500, &HeaderMap::new(), &body);
        assert!(matches!(err, Error::Provider { .. }));
        assert_eq!(err.request_id(), Some("req_body"));
        assert!(err.is_retryable());
    }

    #[test]
    fn spend_limit_429_is_quota_exhausted() {
        let body = json!({
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": "Your organization has reached its spend limit.",
                "details": {"error_code": "enforced_spend_limit_reached"}
            }
        })
        .to_string();
        let err = map_http_error(429, &headers(&[("retry-after", "60")]), &body);
        assert!(matches!(err, Error::QuotaExhausted { .. }), "{err:?}");
        assert!(!err.is_retryable());
        assert_eq!(
            err.context().unwrap().provider_code.as_deref(),
            Some("enforced_spend_limit_reached")
        );
    }

    #[test]
    fn billing_and_user_spend_limits_are_quota_exhausted() {
        let err = map_http_error(
            400,
            &HeaderMap::new(),
            &body(
                "invalid_request_error",
                "Your credit balance is too low to access the Anthropic API.",
            ),
        );
        assert!(matches!(err, Error::QuotaExhausted { .. }), "{err:?}");
        let err = map_http_error(
            400,
            &HeaderMap::new(),
            &body(
                "invalid_request_error",
                "You have reached your specified API usage limits. You will regain access on 2026-11-01.",
            ),
        );
        assert!(matches!(err, Error::QuotaExhausted { .. }), "{err:?}");
        let err = map_http_error(
            402,
            &HeaderMap::new(),
            &body("billing_error", "payment required"),
        );
        assert!(matches!(err, Error::QuotaExhausted { .. }), "{err:?}");
    }

    #[test]
    fn maps_overloaded() {
        for status in [503, 529] {
            let err = map_http_error(
                status,
                &HeaderMap::new(),
                &body("overloaded_error", "Overloaded"),
            );
            assert!(matches!(err, Error::Overloaded { .. }));
        }
        let err = map_http_error(503, &HeaderMap::new(), "Service Unavailable");
        assert!(matches!(&err, Error::Overloaded { message, .. } if message == "HTTP 503"));
        assert_eq!(err.upstream_message(), None);
    }

    #[test]
    fn maps_context_length_with_limit() {
        let err = map_http_error(
            400,
            &HeaderMap::new(),
            &body(
                "invalid_request_error",
                "prompt is too long: 208310 tokens > 200000 maximum",
            ),
        );
        assert!(
            matches!(
                err,
                Error::ContextLengthExceeded {
                    context_window: Some(200000),
                    ..
                }
            ),
            "{err:?}"
        );
        let err = map_http_error(
            400,
            &HeaderMap::new(),
            &body(
                "invalid_request_error",
                "too many tokens: your request had 200000",
            ),
        );
        assert!(matches!(
            err,
            Error::ContextLengthExceeded {
                context_window: None,
                ..
            }
        ));
    }

    #[test]
    fn maps_content_filtering_and_invalid_request() {
        let err = map_http_error(
            400,
            &HeaderMap::new(),
            &body(
                "invalid_request_error",
                "Your request was blocked due to content filtering",
            ),
        );
        assert!(matches!(err, Error::ContentFiltered { .. }));
        let err = map_http_error(
            400,
            &HeaderMap::new(),
            &body(
                "invalid_request_error",
                "max_tokens must be a positive integer",
            ),
        );
        assert!(matches!(err, Error::InvalidRequest { .. }));
    }

    #[test]
    fn maps_model_not_found() {
        let err = map_http_error(
            404,
            &HeaderMap::new(),
            &body("not_found_error", "model: claude-nope"),
        );
        assert!(matches!(err, Error::ModelNotFound { .. }), "{err:?}");
        let err = map_http_error(404, &HeaderMap::new(), "Not Found");
        assert!(matches!(err, Error::InvalidRequest { .. }));
    }

    #[test]
    fn maps_server_errors() {
        let err = map_http_error(502, &HeaderMap::new(), &body("api_error", "bad gateway"));
        assert!(matches!(err, Error::Provider { .. }));
        assert_eq!(err.status(), Some(502));
        let err = map_http_error(600, &HeaderMap::new(), "weird");
        assert!(matches!(err, Error::Provider { .. }));
        assert_eq!(err.context().unwrap().body.as_deref(), Some("weird"));
    }

    #[test]
    fn statuses_mean_what_they_mean_on_every_adapter() {
        // Without an Anthropic error type, the status decides.
        for (status, kind) in [
            (408, "timeout"),
            (504, "timeout"),
            (524, "timeout"),
            (529, "overloaded"),
            (499, "provider"),
        ] {
            let err = map_http_error(status, &HeaderMap::new(), "<html>gateway</html>");
            assert_eq!(err.kind_name(), kind, "HTTP {status}");
            assert_eq!(err.status(), Some(status));
        }
        assert!(!map_http_error(499, &HeaderMap::new(), "").is_retryable());
        // Anthropic's own 504 and 529 (https://docs.claude.com/en/api/errors).
        let err = map_http_error(504, &HeaderMap::new(), &body("timeout_error", "timed out"));
        assert!(matches!(err, Error::Timeout { .. }), "{err:?}");
        let err = map_http_error(
            529,
            &HeaderMap::new(),
            &body("overloaded_error", "Overloaded"),
        );
        assert!(matches!(err, Error::Overloaded { .. }), "{err:?}");
    }

    #[test]
    fn stream_error_events_map_to_typed_errors() {
        assert!(matches!(
            stream_error("overloaded_error", "Overloaded"),
            Error::Overloaded { .. }
        ));
        assert!(matches!(
            stream_error("rate_limit_error", "slow"),
            Error::RateLimited { .. }
        ));
        assert!(matches!(
            stream_error("authentication_error", "bad"),
            Error::Auth { .. }
        ));
        assert!(matches!(
            stream_error("api_error", "boom"),
            Error::Provider { .. }
        ));
        assert!(matches!(
            stream_error("invalid_request_error", "prompt is too long: 300000 tokens"),
            Error::ContextLengthExceeded { .. }
        ));
        let unknown = stream_error("totally_new_error", "new");
        assert!(matches!(unknown, Error::Provider { .. }), "{unknown:?}");
        assert!(unknown.is_retryable());
    }

    #[test]
    fn upstream_text_stays_out_of_the_log() {
        let err = map_http_error(
            400,
            &headers(&[("request-id", "req_1")]),
            &body("invalid_request_error", "bad value: my SSN is 123-45-6789"),
        );
        assert_eq!(
            err.as_log().to_string(),
            r#"kind=invalid_request provider=anthropic status=400 provider_type=invalid_request_error request_id=req_1 body_len=102 message="HTTP 400""#
        );
        assert_eq!(
            err.to_string(),
            "anthropic: invalid request: HTTP 400: bad value: my SSN is 123-45-6789"
        );

        let streamed = stream_error("overloaded_error", "Overloaded SENTINEL");
        assert!(!streamed.as_log().to_string().contains("SENTINEL"));
        assert_eq!(streamed.upstream_message(), Some("Overloaded SENTINEL"));
    }

    #[test]
    fn transport_errors() {
        let err = TransportError::new(TransportErrorKind::Timeout, "after 120s");
        assert!(matches!(map_transport_error(err), Error::Timeout { .. }));
        let err = map_transport_error(TransportError::new(
            TransportErrorKind::Connect,
            "connection reset",
        ));
        assert!(matches!(err, Error::Transport { .. }));
        assert!(err.is_retryable());
    }
}
