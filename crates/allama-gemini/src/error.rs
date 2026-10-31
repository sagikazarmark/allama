//! Mapping of Gemini API failures to [`allama::Error`].
//!
//! Gemini errors are `google.rpc.Status` objects. Classification uses, in
//! order: the HTTP status, the `ErrorInfo.reason` detail, the RPC status
//! string, and finally message heuristics for conditions Google reports only
//! in prose (context length).
//!
//! - `402` means prepaid credit is depleted: [`Error::QuotaExhausted`]
//!   (never retried).
//! - `429 RESOURCE_EXHAUSTED` is normally a rate limit: [`Error::RateLimited`]
//!   with the `google.rpc.RetryInfo.retryDelay` hint (falling back to the
//!   `Retry-After` header). A `google.rpc.QuotaFailure` violation of a daily
//!   quota (a `quotaId` with `PerDay`, such as the free tier's
//!   `GenerateRequestsPerDayPerProjectPerModel-FreeTier`) is
//!   [`Error::QuotaExhausted`], even beside a retry hint: retrying within
//!   the day does not help. So is a message that clearly names a spend cap
//!   or depleted credits, with no retry hint.
//! - `400` with `ErrorInfo.reason == "API_KEY_INVALID"` is an authentication
//!   failure.
//! - A missing or expired explicit cache (`cachedContents/…`) or Files API
//!   file is [`Error::InvalidRequest`] naming the resource, although Gemini
//!   answers `403 PERMISSION_DENIED` (or `404`): it is neither a credential
//!   problem nor a missing model.
//! - A `thinkingLevel` the model does not take ("Thinking level MINIMAL is
//!   not supported for this model") is [`Error::Unsupported`] on the
//!   reasoning field, stage `Provider`.
//! - `DEADLINE_EXCEEDED` is [`Error::Timeout`], whatever the HTTP status.
//! - Anything else is what the status means on its own
//!   ([`allama_http::status_error`]), as on every adapter.
//!
//! Gemini sends no request-id header, so errors carry a request id only when
//! a proxy adds `x-request-id` or `request-id` (read by
//! [`allama_http::error_context`], like the retry hint and the server retry
//! directive).

use std::time::Duration;

use allama::{Error, ErrorContext, RequestField, Result, Subject};
use allama_http::{Bytes, HttpResponse, TransportError};
use serde::de::DeserializeOwned;

use crate::PROVIDER_NAME;
use crate::wire::{GeminiError, GeminiErrorResponse};
use allama_http::http::{HeaderMap, StatusCode};

/// Map a non-success HTTP response to an [`Error`].
///
/// The status and headers are read with [`allama_http::error_context`]. A
/// body that is no `google.rpc.Status` envelope still informs the category,
/// but is kept only as the body; the message is `HTTP {status}`.
pub(crate) fn map_http_error(status: StatusCode, headers: &HeaderMap, body: &str) -> Error {
    let context = allama_http::error_context(status, headers);
    let status = status.as_u16();
    match serde_json::from_str::<GeminiErrorResponse>(body) {
        Ok(parsed) => map_api_error(Some(status), &parsed.error, context),
        Err(_) => classify(
            status,
            None,
            None,
            body,
            Signals::default(),
            format!("HTTP {status}"),
        )
        .with_context(context)
        .authored()
        .with_body_opt((!body.is_empty()).then_some(body))
        .with_provider(PROVIDER_NAME),
    }
}

/// Map a `google.rpc.Status` (from an HTTP body or an in-stream `error`
/// event) to an [`Error`].
///
/// `context` is the HTTP response's (empty for an in-stream event). The
/// status message is the [upstream message](Error::upstream_message),
/// `google.rpc.BadRequest` field violations name the rejected field
/// ([`ErrorContext::param`]), and a `google.rpc.RetryInfo` delay takes
/// precedence over the `Retry-After` header.
pub(crate) fn map_api_error(
    http_status: Option<u16>,
    error: &GeminiError,
    context: ErrorContext,
) -> Error {
    let status = http_status.or(error.code).unwrap_or(500);
    let reason = error_info_reason(error);
    let retry_delay = retry_info_delay(error);
    let message = match http_status {
        Some(status) => format!("HTTP {status}"),
        None => format!("Gemini API error ({status})"),
    };
    let body = serde_json::json!({
        "error": {
            "code": error.code,
            "message": error.message,
            "status": error.status,
            "details": error.details,
        }
    })
    .to_string();
    let signals = Signals {
        retry_info: retry_delay.is_some(),
        daily_quota: violates_daily_quota(error),
    };
    classify(
        status,
        error.status.as_deref(),
        reason.as_deref(),
        &error.message,
        signals,
        message,
    )
    .with_context(context)
    .authored()
    .with_status(status)
    .with_retry_after_opt(retry_delay)
    .with_upstream_message(error.message.clone())
    .with_param_opt(field_violations(error).as_deref())
    .with_provider_code_opt(reason.as_deref())
    .with_provider_type_opt(error.status.as_deref())
    .with_body(body)
    .with_provider(PROVIDER_NAME)
}

/// What the `google.rpc.Status` details say beyond the status and reason.
#[derive(Debug, Clone, Copy, Default)]
struct Signals {
    /// A `google.rpc.RetryInfo` delay is present.
    retry_info: bool,
    /// A `google.rpc.QuotaFailure` violation is of a daily quota.
    daily_quota: bool,
}

/// Pick the category; `text` is the upstream message (or an unrecognised
/// body), read for conditions Google reports only in prose. The result
/// carries only `message`; the caller adds the context and marks it
/// [authored](Error::authored).
fn classify(
    status: u16,
    rpc_status: Option<&str>,
    reason: Option<&str>,
    text: &str,
    signals: Signals,
    message: String,
) -> Error {
    let lower = text.to_ascii_lowercase();
    if status == 402 {
        Error::quota_exhausted(message)
    } else if let Some(resource) = missing_resource(status, rpc_status, &lower) {
        // Checked before the status: a 403 here is no credential problem,
        // and a 404 no missing model.
        Error::invalid_request(format!("{message}: {resource}"))
    } else if matches!(reason, Some("API_KEY_INVALID" | "API_KEY_EXPIRED"))
        || matches!(rpc_status, Some("UNAUTHENTICATED" | "PERMISSION_DENIED"))
        || matches!(status, 401 | 403)
    {
        Error::auth(message)
    } else if status == 429 || rpc_status == Some("RESOURCE_EXHAUSTED") {
        if signals.daily_quota || (!signals.retry_info && mentions_spend_cap(&lower)) {
            Error::quota_exhausted(message)
        } else {
            Error::rate_limited(message)
        }
    } else if status == 404 || rpc_status == Some("NOT_FOUND") {
        Error::model_not_found(message)
    } else if rpc_status == Some("DEADLINE_EXCEEDED") {
        Error::timeout(message)
    } else if status == 400 || rpc_status == Some("INVALID_ARGUMENT") {
        if is_context_length_message(&lower) {
            Error::context_length_exceeded(message, parse_input_token_limit(&lower))
        } else if lower.starts_with("thinking level") && lower.contains(NOT_SUPPORTED_FOR_MODEL) {
            Error::unsupported_by_provider(
                Some(Subject::Field {
                    field: RequestField::Reasoning,
                }),
                format!("{message}: the model does not take this thinking level"),
            )
        } else if let Some((field, feature)) = not_enabled(&lower) {
            Error::unsupported_by_provider(
                Some(Subject::Field { field }),
                format!("{message}: {feature} is not enabled for this model"),
            )
        } else if lower.contains(PENALTY_NOT_ENABLED) {
            // The reply does not say which penalty; the chat call names the
            // one the request set (`name_refused_penalty`).
            Error::unsupported_by_provider(
                Some(Subject::Field {
                    field: RequestField::FrequencyPenalty,
                }),
                format!("{message}: penalties are not enabled for this model"),
            )
        } else {
            Error::invalid_request(message)
        }
    } else {
        // What the status means on its own, shared by every adapter. An
        // in-stream `code` past what HTTP allows is a provider error.
        match StatusCode::from_u16(status) {
            Ok(status) => allama_http::status_error(status, message),
            Err(_) => Error::provider(message).with_status(status),
        }
    }
}

/// The request field Gemini refuses in a `400` worded "`<feature>` is not
/// enabled for models/`<model>`", as for the Gemma models
/// ("Function calling is not enabled for models/gemma-3-27b-it"), with the
/// feature named. Each pattern is pinned by a fixture in this module's
/// tests; any other feature stays an `InvalidRequest`.
fn not_enabled(lower: &str) -> Option<(RequestField, &'static str)> {
    let (feature, _) = lower.split_once(" is not enabled for models/")?;
    match feature.trim() {
        "function calling" => Some((RequestField::Tools, "function calling")),
        "developer instruction" => Some((RequestField::System, "a system instruction")),
        "json mode" => Some((RequestField::ResponseFormat, "JSON mode")),
        _ => None,
    }
}

/// A request resource Gemini cannot find: an explicit context cache or a
/// Files API file, both of which expire. Gemini answers `403
/// PERMISSION_DENIED` ("CachedContent not found (or permission denied)",
/// "You do not have permission to access the File … or it may not exist",
/// both verified live 2026-10-07) or `404`, which would otherwise read as a
/// credential problem or a missing model.
fn missing_resource(status: u16, rpc_status: Option<&str>, lower: &str) -> Option<&'static str> {
    let not_found = matches!(status, 403 | 404)
        || matches!(rpc_status, Some("PERMISSION_DENIED" | "NOT_FOUND"));
    if !not_found {
        return None;
    }
    if lower.contains("cachedcontent") || lower.contains("cached content") {
        Some(
            "the cached content does not exist or has expired; create it again and update `ChatRequestOptions::cached_content`",
        )
    } else if lower.contains("access the file") || lower.contains("file not found") {
        Some(
            "the Files API file does not exist or has expired (files are kept for 48 hours); upload it again",
        )
    } else {
        None
    }
}

/// Gemini's refusal of a `thinkingLevel` a model does not take, lowercased
/// ("Thinking level MINIMAL is not supported for this model. Please retry
/// with other thinking level.", verified live on Gemini 3.1 Pro
/// 2026-10-07).
const NOT_SUPPORTED_FOR_MODEL: &str = "is not supported for this model";

/// Gemini's refusal of `presencePenalty` or `frequencyPenalty` on a model
/// that does not enable them, lowercased ("Penalty is not enabled for this
/// model", verified live; no captured fixture yet).
const PENALTY_NOT_ENABLED: &str = "penalty is not enabled for this model";

/// Name the penalty Gemini refused: its reply does not say which, so an
/// [`Error::Unsupported`] for the frequency penalty names the presence
/// penalty when that is the only one `request` set.
pub(crate) fn name_refused_penalty(mut err: Error, request: &allama::ChatRequest) -> Error {
    if let Error::Unsupported {
        subject: Some(Subject::Field { field }),
        ..
    } = &mut err
        && *field == RequestField::FrequencyPenalty
        && request.frequency_penalty.is_none()
        && request.presence_penalty.is_some()
    {
        *field = RequestField::PresencePenalty;
    }
    err
}

/// Whether a `google.rpc.QuotaFailure` violation is of a daily quota: its
/// `quotaId` names a per-day window (`...PerDay...`). Every violation is
/// checked: Gemini lists the per-minute quotas beside the daily one.
fn violates_daily_quota(error: &GeminiError) -> bool {
    error
        .details
        .iter()
        .filter(|detail| {
            detail
                .get("@type")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|type_url| type_url.ends_with("google.rpc.QuotaFailure"))
        })
        .filter_map(|detail| detail.get("violations")?.as_array())
        .flatten()
        .filter_map(|violation| violation.get("quotaId")?.as_str())
        .any(|quota_id| quota_id.contains("PerDay"))
}

/// The fields named by `google.rpc.BadRequest.fieldViolations`, joined with
/// `, `.
fn field_violations(error: &GeminiError) -> Option<String> {
    let fields: Vec<&str> = error
        .details
        .iter()
        .filter(|detail| {
            detail
                .get("@type")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|type_url| type_url.ends_with("google.rpc.BadRequest"))
        })
        .filter_map(|detail| detail.get("fieldViolations")?.as_array())
        .flatten()
        .filter_map(|violation| violation.get("field")?.as_str())
        .filter(|field| !field.is_empty())
        .collect();
    (!fields.is_empty()).then(|| fields.join(", "))
}

fn mentions_spend_cap(lower: &str) -> bool {
    [
        "spend cap",
        "spending cap",
        "spend limit",
        "spending limit",
        "prepay",
        "prepaid",
        "credit balance",
        "credits are depleted",
        "out of credits",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn is_context_length_message(lower: &str) -> bool {
    (lower.contains("token") && (lower.contains("exceeds") || lower.contains("too long")))
        || lower.contains("context length")
        || lower.contains("context window")
}

/// Parse `... maximum number of tokens allowed (1048576)` style limits:
/// Gemini reports the model's input-token limit.
fn parse_input_token_limit(lower: &str) -> Option<u64> {
    let idx = lower.find("allowed")?;
    let rest = &lower[idx..];
    let start = rest.find(|c: char| c.is_ascii_digit())?;
    let digits: String = rest[start..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

fn error_info_reason(error: &GeminiError) -> Option<String> {
    error.details.iter().find_map(|detail| {
        let type_url = detail.get("@type")?.as_str()?;
        type_url
            .ends_with("google.rpc.ErrorInfo")
            .then(|| detail.get("reason")?.as_str().map(str::to_owned))
            .flatten()
    })
}

/// Parse `google.rpc.RetryInfo.retryDelay` (a protobuf `Duration` in JSON
/// form, e.g. `"23s"` or `"1.5s"`).
fn retry_info_delay(error: &GeminiError) -> Option<Duration> {
    error.details.iter().find_map(|detail| {
        let type_url = detail.get("@type")?.as_str()?;
        if !type_url.ends_with("google.rpc.RetryInfo") {
            return None;
        }
        parse_proto_duration(detail.get("retryDelay")?.as_str()?)
    })
}

/// `None` for a malformed, negative or non-finite value, and for one too
/// large for a [`Duration`] (such as `"1e20s"`), so a hostile or buggy reply
/// cannot panic the caller.
fn parse_proto_duration(value: &str) -> Option<Duration> {
    let secs: f64 = value.trim().strip_suffix('s')?.parse().ok()?;
    Duration::try_from_secs_f64(secs).ok()
}

/// A transport failure before a response arrived.
pub(crate) fn map_transport_error(err: TransportError) -> Error {
    err.into_error().with_provider(PROVIDER_NAME)
}

/// A success response's body, read whole.
pub(crate) struct SuccessBody {
    status: StatusCode,
    /// Gemini sends no request id; a proxy's `x-request-id` or `request-id`
    /// is kept, as on error responses.
    request_id: Option<String>,
    bytes: Bytes,
}

impl SuccessBody {
    /// Read `response`'s body, up to `limit` bytes. A failed read is the
    /// transport's error ([`TransportError::into_error`]), a timeout and a
    /// body past `limit` included.
    pub(crate) async fn read(response: HttpResponse, limit: usize) -> Result<Self> {
        let status = response.status;
        let request_id = allama_http::error_context(status, &response.headers).request_id;
        let bytes = response.bytes_limited(limit).await.map_err(|err| {
            err.into_error()
                .with_provider(PROVIDER_NAME)
                .with_request_id_opt(request_id.as_deref())
        })?;
        Ok(Self {
            status,
            request_id,
            bytes,
        })
    }

    /// The request id from the response headers, when a proxy sent one.
    pub(crate) fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    /// Decode the body as a `T`. A body that is not one is
    /// [`allama_http::malformed_response`].
    pub(crate) fn decode<T: DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_slice(&self.bytes).map_err(|err| {
            allama_http::malformed_response(self.status, &self.bytes, err)
                .with_provider(PROVIDER_NAME)
                .with_request_id_opt(self.request_id.as_deref())
        })
    }
}

#[cfg(test)]
pub(crate) fn conformance_map_http_error(
    status: u16,
    body: &str,
    retry_after: Option<Duration>,
) -> Error {
    let mut headers = HeaderMap::new();
    if let Some(retry_after) = retry_after {
        headers.insert("retry-after", retry_after.as_secs().into());
    }
    map_http_error(StatusCode::from_u16(status).unwrap(), &headers, body)
}

#[cfg(test)]
pub(crate) fn conformance_map_timeout_transport_error() -> Error {
    map_transport_error(TransportError::new(
        allama_http::TransportErrorKind::Timeout,
        "request timed out",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use allama_http::TransportErrorKind;
    use allama_http::http::StatusCode;
    use serde_json::json;

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

    fn body(code: u16, message: &str, status: &str) -> String {
        json!({"error": {"code": code, "message": message, "status": status}}).to_string()
    }

    #[test]
    fn a_missing_cache_404_is_no_missing_model() {
        let err = map_http_error(
            StatusCode::NOT_FOUND,
            &HeaderMap::new(),
            &body(
                404,
                "CachedContent not found: cachedContents/abc",
                "NOT_FOUND",
            ),
        );
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
        // A real permission problem stays an auth error.
        let err = map_http_error(
            StatusCode::FORBIDDEN,
            &HeaderMap::new(),
            &body(
                403,
                "Method doesn't allow unregistered callers.",
                "PERMISSION_DENIED",
            ),
        );
        assert!(matches!(err, Error::Auth { .. }), "{err:?}");
    }

    #[test]
    fn invalid_api_key_400_maps_to_auth() {
        // Shape recorded from the live API on 2026-10-02.
        let body = json!({"error": {
            "code": 400,
            "message": "API key not valid. Please pass a valid API key.",
            "status": "INVALID_ARGUMENT",
            "details": [{
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": "API_KEY_INVALID",
                "domain": "googleapis.com",
                "metadata": {"service": "generativelanguage.googleapis.com"}
            }]
        }})
        .to_string();
        let err = map_http_error(StatusCode::BAD_REQUEST, &HeaderMap::new(), &body);
        assert!(matches!(err, Error::Auth { .. }), "{err:?}");
        let context = err.context().unwrap();
        assert_eq!(context.provider_code.as_deref(), Some("API_KEY_INVALID"));
        assert_eq!(context.provider_type.as_deref(), Some("INVALID_ARGUMENT"));
        assert_eq!(context.provider.as_deref(), Some("gemini"));
        assert_eq!(err.status(), Some(400));
        assert!(!err.is_retryable());
    }

    #[test]
    fn unauthenticated_and_permission_denied_map_to_auth() {
        for (code, status) in [(401, "UNAUTHENTICATED"), (403, "PERMISSION_DENIED")] {
            let err = map_http_error(
                StatusCode::from_u16(code).unwrap(),
                &HeaderMap::new(),
                &body(code, "nope", status),
            );
            assert!(matches!(err, Error::Auth { .. }), "{err:?}");
        }
    }

    #[test]
    fn status_402_is_quota_exhausted() {
        let err = map_http_error(
            StatusCode::PAYMENT_REQUIRED,
            &HeaderMap::new(),
            &body(
                402,
                "Your prepayment credits are depleted.",
                "RESOURCE_EXHAUSTED",
            ),
        );
        assert!(matches!(err, Error::QuotaExhausted { .. }), "{err:?}");
        assert!(!err.is_retryable());
    }

    #[test]
    fn status_429_is_rate_limited_with_retry_info() {
        let body = json!({"error": {
            "code": 429,
            "message": "You exceeded your current quota, please check your plan and billing details.",
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": []},
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "23.5s"}
            ]
        }})
        .to_string();
        let err = map_http_error(
            StatusCode::TOO_MANY_REQUESTS,
            &headers(&[("retry-after", "1")]),
            &body,
        );
        assert!(matches!(err, Error::RateLimited { .. }), "{err:?}");
        assert_eq!(err.retry_after(), Some(Duration::from_secs_f64(23.5)));
        assert!(err.is_retryable());
    }

    #[test]
    fn status_429_falls_back_to_header_retry_after() {
        let err = map_http_error(
            StatusCode::TOO_MANY_REQUESTS,
            &headers(&[("retry-after", "2")]),
            &body(429, "Resource exhausted", "RESOURCE_EXHAUSTED"),
        );
        assert!(matches!(err, Error::RateLimited { .. }));
        assert_eq!(err.retry_after(), Some(Duration::from_secs(2)));
    }

    #[test]
    fn status_429_spend_cap_is_quota_exhausted() {
        let err = map_http_error(
            StatusCode::TOO_MANY_REQUESTS,
            &HeaderMap::new(),
            &body(
                429,
                "Your project has exceeded its monthly spending cap.",
                "RESOURCE_EXHAUSTED",
            ),
        );
        assert!(matches!(err, Error::QuotaExhausted { .. }), "{err:?}");
    }

    /// A free-tier daily quota, as reported in
    /// <https://github.com/google-gemini/gemini-cli/issues/8437>: the daily
    /// violation sits beside per-minute ones, with a retry hint.
    fn daily_quota_body(violations: &[&str]) -> String {
        let violations: Vec<_> = violations
            .iter()
            .map(|quota_id| {
                json!({
                    "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                    "quotaId": quota_id,
                    "quotaDimensions": {"location": "global", "model": "gemini-2.5-flash"}
                })
            })
            .collect();
        json!({"error": {
            "code": 429,
            "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits.",
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": violations},
                {"@type": "type.googleapis.com/google.rpc.Help", "links": [{
                    "description": "Learn more about Gemini API quotas",
                    "url": "https://ai.google.dev/gemini-api/docs/rate-limits"
                }]},
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "55s"}
            ]
        }})
        .to_string()
    }

    #[test]
    fn a_daily_quota_is_exhausted_even_with_a_retry_hint() {
        let body = daily_quota_body(&[
            "GenerateContentInputTokensPerModelPerMinute-FreeTier",
            "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
            "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
        ]);
        let err = map_http_error(StatusCode::TOO_MANY_REQUESTS, &HeaderMap::new(), &body);
        assert!(matches!(err, Error::QuotaExhausted { .. }), "{err:?}");
        assert!(!err.is_retryable());
        // The hint is kept for callers that wait it out anyway.
        assert_eq!(err.retry_after(), Some(Duration::from_secs(55)));

        // Per-minute quotas alone are a rate limit.
        let body = daily_quota_body(&[
            "GenerateContentInputTokensPerModelPerMinute-FreeTier",
            "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
        ]);
        let err = map_http_error(StatusCode::TOO_MANY_REQUESTS, &HeaderMap::new(), &body);
        assert!(matches!(err, Error::RateLimited { .. }), "{err:?}");

        // In a stream too.
        let parsed: GeminiErrorResponse = serde_json::from_str(&daily_quota_body(&[
            "GenerateRequestsPerDayPerProjectPerModel",
        ]))
        .unwrap();
        let err = map_api_error(None, &parsed.error, ErrorContext::new());
        assert!(matches!(err, Error::QuotaExhausted { .. }), "{err:?}");
    }

    #[test]
    fn deadline_exceeded_is_a_timeout() {
        // `DEADLINE_EXCEEDED` is Gemini's 504
        // (https://ai.google.dev/gemini-api/docs/troubleshooting).
        for status in [504, 408, 500] {
            let err = map_http_error(
                StatusCode::from_u16(status).unwrap(),
                &HeaderMap::new(),
                &body(
                    status,
                    "Deadline expired before operation could complete.",
                    "DEADLINE_EXCEEDED",
                ),
            );
            assert!(matches!(err, Error::Timeout { .. }), "{status}: {err:?}");
        }
        let error = GeminiError {
            code: Some(504),
            message: "Deadline expired".into(),
            status: Some("DEADLINE_EXCEEDED".into()),
            details: Vec::new(),
        };
        let err = map_api_error(None, &error, ErrorContext::new());
        assert!(matches!(err, Error::Timeout { .. }), "{err:?}");
    }

    #[test]
    fn statuses_mean_what_they_mean_on_every_adapter() {
        for (status, kind) in [
            (408, "timeout"),
            (504, "timeout"),
            (524, "timeout"),
            (499, "provider"),
        ] {
            let err = map_http_error(
                StatusCode::from_u16(status).unwrap(),
                &HeaderMap::new(),
                "<html>gateway</html>",
            );
            assert_eq!(err.kind_name(), kind, "HTTP {status}");
        }
        // `CANCELLED` is HTTP 499 in `google.rpc.Code`
        // (https://github.com/googleapis/googleapis/blob/master/google/rpc/code.proto).
        let err = map_http_error(
            StatusCode::from_u16(499).unwrap(),
            &HeaderMap::new(),
            &body(499, "cancelled", "CANCELLED"),
        );
        assert!(matches!(err, Error::Provider { .. }), "{err:?}");
        assert!(!err.is_retryable());
    }

    #[test]
    fn not_found_maps_to_model_not_found() {
        let err = map_http_error(
            StatusCode::NOT_FOUND,
            &HeaderMap::new(),
            &body(
                404,
                "models/gemini-9-nope is not found for API version v1beta",
                "NOT_FOUND",
            ),
        );
        assert!(matches!(err, Error::ModelNotFound { .. }));
    }

    #[test]
    fn context_length_is_detected_with_limit() {
        let err = map_http_error(
            StatusCode::BAD_REQUEST,
            &HeaderMap::new(),
            &body(
                400,
                "The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).",
                "INVALID_ARGUMENT",
            ),
        );
        match err {
            Error::ContextLengthExceeded { context_window, .. } => {
                assert_eq!(context_window, Some(1_048_576));
            }
            other => panic!("expected ContextLengthExceeded, got {other:?}"),
        }
    }

    /// Gemini refuses a feature a model lacks with "`<feature>` is not
    /// enabled for models/`<model>`" (the replies the API gives Gemma 3):
    /// the provider's own refusal of that field.
    #[test]
    fn features_not_enabled_are_unsupported_by_the_provider() {
        let cases = [
            (
                "Function calling is not enabled for models/gemma-3-27b-it",
                RequestField::Tools,
            ),
            (
                "Developer instruction is not enabled for models/gemma-3-27b-it",
                RequestField::System,
            ),
            (
                "JSON mode is not enabled for models/gemma-3-1b-it",
                RequestField::ResponseFormat,
            ),
        ];
        for (text, field) in cases {
            let err = map_http_error(
                StatusCode::BAD_REQUEST,
                &HeaderMap::new(),
                &body(400, text, "INVALID_ARGUMENT"),
            );
            assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
            assert_eq!(err.unsupported_field(), Some(field), "{text}");
            assert_eq!(
                err.unsupported_stage(),
                Some(allama::UnsupportedStage::Provider)
            );
            assert_eq!(err.upstream_message(), Some(text));
        }
        // A feature outside the table stays an invalid request.
        let err = map_http_error(
            StatusCode::BAD_REQUEST,
            &HeaderMap::new(),
            &body(
                400,
                "Audio input is not enabled for models/gemma-3-27b-it",
                "INVALID_ARGUMENT",
            ),
        );
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
    }

    /// A model that does not enable penalties refuses them with "Penalty is
    /// not enabled for this model" (verified live, as the adapter's penalty
    /// drop recorded before ADR 0015): the
    /// provider's refusal of the penalty the request set.
    #[test]
    fn penalties_not_enabled_are_unsupported_by_the_provider() {
        let text = "Penalty is not enabled for this model";
        let err = map_http_error(
            StatusCode::BAD_REQUEST,
            &HeaderMap::new(),
            &body(400, text, "INVALID_ARGUMENT"),
        );
        assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
        assert_eq!(
            err.unsupported_stage(),
            Some(allama::UnsupportedStage::Provider)
        );
        assert_eq!(err.upstream_message(), Some(text));
        assert_eq!(
            err.unsupported_field(),
            Some(RequestField::FrequencyPenalty)
        );

        let request = |frequency: Option<f64>, presence: Option<f64>| {
            let mut request = allama::ChatRequest::new("gemini-2.5-flash").user("q");
            request.frequency_penalty = frequency;
            request.presence_penalty = presence;
            request
        };
        for (request, field) in [
            (request(Some(0.5), None), RequestField::FrequencyPenalty),
            (request(None, Some(0.5)), RequestField::PresencePenalty),
            (
                request(Some(0.5), Some(0.5)),
                RequestField::FrequencyPenalty,
            ),
        ] {
            let named = name_refused_penalty(err.clone(), &request);
            assert_eq!(named.unsupported_field(), Some(field));
            assert_eq!(named.upstream_message(), Some(text));
        }
    }

    #[test]
    fn other_400_is_invalid_request() {
        let err = map_http_error(
            StatusCode::BAD_REQUEST,
            &HeaderMap::new(),
            &body(
                400,
                "You can only set only one of thinking budget and thinking level.",
                "INVALID_ARGUMENT",
            ),
        );
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
    }

    #[test]
    fn server_errors_map_by_status() {
        let err = map_http_error(
            StatusCode::SERVICE_UNAVAILABLE,
            &HeaderMap::new(),
            &body(503, "overloaded", "UNAVAILABLE"),
        );
        assert!(matches!(err, Error::Overloaded { .. }));
        let err = map_http_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &HeaderMap::new(),
            &body(500, "internal", "INTERNAL"),
        );
        assert!(matches!(err, Error::Provider { .. }));
        assert!(err.is_retryable());
        let err = map_http_error(
            StatusCode::GATEWAY_TIMEOUT,
            &HeaderMap::new(),
            &body(504, "deadline", "DEADLINE_EXCEEDED"),
        );
        assert!(matches!(err, Error::Timeout { .. }));
    }

    #[test]
    fn unstructured_body_is_kept() {
        let err = map_http_error(StatusCode::BAD_GATEWAY, &HeaderMap::new(), "Bad Gateway");
        assert!(matches!(err, Error::Provider { .. }));
        assert_eq!(err.message(), "HTTP 502");
        assert_eq!(err.upstream_message(), None);
        assert_eq!(err.context().unwrap().body.as_deref(), Some("Bad Gateway"));
    }

    #[test]
    fn field_violations_name_the_rejected_field() {
        let body = json!({"error": {
            "code": 400,
            "message": "Invalid value at 'contents[0].parts[1]': my SSN is 123-45-6789",
            "status": "INVALID_ARGUMENT",
            "details": [{
                "@type": "type.googleapis.com/google.rpc.BadRequest",
                "fieldViolations": [
                    {"field": "contents[0].parts[1]", "description": "bad"},
                    {"field": "generation_config.seed", "description": "bad"},
                ],
            }],
        }})
        .to_string();
        let err = map_http_error(StatusCode::BAD_REQUEST, &HeaderMap::new(), &body);
        assert_eq!(
            err.context().unwrap().param.as_deref(),
            Some("contents[0].parts[1], generation_config.seed")
        );
        // The joined list is no bounded token: the redacted log says a field
        // was named, not which.
        let line = err.as_log().to_string();
        assert!(line.contains("param=<redacted>"), "{line}");
        assert!(line.contains("message=\"HTTP 400\""), "{line}");
        assert!(!line.contains("123-45-6789"), "{line}");
        assert!(err.to_string().contains("123-45-6789"));

        // One field is.
        let body = body.replace(
            r#",{"description":"bad","field":"generation_config.seed"}"#,
            "",
        );
        let err = map_http_error(StatusCode::BAD_REQUEST, &HeaderMap::new(), &body);
        assert_eq!(
            err.context().unwrap().param.as_deref(),
            Some("contents[0].parts[1]")
        );
        let line = err.as_log().to_string();
        assert!(line.contains("param=contents[0].parts[1]"), "{line}");
    }

    #[test]
    fn in_stream_error_uses_rpc_code() {
        let error = GeminiError {
            code: Some(503),
            message: "The model is overloaded.".into(),
            status: Some("UNAVAILABLE".into()),
            details: Vec::new(),
        };
        let err = map_api_error(None, &error, ErrorContext::new());
        assert!(matches!(err, Error::Overloaded { .. }), "{err:?}");
    }

    #[test]
    fn proto_durations_parse() {
        assert_eq!(parse_proto_duration("23s"), Some(Duration::from_secs(23)));
        assert_eq!(
            parse_proto_duration("0.5s"),
            Some(Duration::from_millis(500))
        );
        assert_eq!(parse_proto_duration("-1s"), None);
        assert_eq!(parse_proto_duration("23"), None);
        // Regression: these used to panic.
        assert_eq!(parse_proto_duration("1e20s"), None);
        assert_eq!(parse_proto_duration("NaNs"), None);
        assert_eq!(parse_proto_duration("infs"), None);
    }

    #[test]
    fn transport_errors_keep_their_kind() {
        let err = map_transport_error(TransportError::new(TransportErrorKind::Timeout, "x"));
        assert!(matches!(err, Error::Timeout { .. }));
        let err = map_transport_error(TransportError::new(
            TransportErrorKind::Connect,
            "connection refused",
        ));
        assert!(matches!(err, Error::Transport { .. }));
        assert_eq!(
            err.context().unwrap().provider.as_deref(),
            Some(PROVIDER_NAME)
        );
    }
}
