//! Error mapping for OpenAI-compatible endpoints.

use std::borrow::Cow;
use std::time::Duration;

use allama::{Error, RequestField, Result, Subject};
use allama_http::{Bytes, Headers, HttpResponse};
use serde::de::DeserializeOwned;

use crate::wire::ApiError;

/// Billing and quota signals that retrying cannot fix.
const QUOTA_CODES: &[&str] = &[
    "insufficient_quota",
    "credit_balance_exhausted",
    "billing_hard_limit_reached",
    "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded",
    "organization_usage_limit_exceeded",
];

/// The in-band error codes of a Responses response the input caused: an
/// image or file it could not read
/// (<https://platform.openai.com/docs/api-reference/responses/object>,
/// `error.code`). Retrying sends the same input, so they are invalid
/// requests, not retryable provider errors.
const INPUT_ERROR_CODES: &[&str] = &[
    "empty_image_file",
    "failed_to_download_image",
    "image_file_not_found",
    "image_file_too_large",
    "image_parse_error",
    "image_too_large",
    "image_too_small",
    "unsupported_image_media_type",
];

/// The request parameters the Chat Completions and Responses encoders write
/// for a portable request field, by the name an OpenAI error's `param`
/// gives them. A refusal of one of these is the provider refusing what the
/// caller asked for; any other `param` (such as `max_tokens`, which the
/// dialect chose) is the adapter's mistake and stays an `InvalidRequest`.
const WIRE_FIELDS: &[(&str, RequestField)] = &[
    ("temperature", RequestField::Temperature),
    ("top_p", RequestField::TopP),
    ("stop", RequestField::Stop),
    ("frequency_penalty", RequestField::FrequencyPenalty),
    ("presence_penalty", RequestField::PresencePenalty),
    ("seed", RequestField::Seed),
    ("random_seed", RequestField::Seed),
    ("tools", RequestField::Tools),
    ("tool_choice", RequestField::ToolChoice),
    ("parallel_tool_calls", RequestField::ParallelToolCalls),
    ("response_format", RequestField::ResponseFormat),
    ("text.format", RequestField::ResponseFormat),
    ("reasoning_effort", RequestField::Reasoning),
    ("reasoning", RequestField::Reasoning),
    ("reasoning.effort", RequestField::Reasoning),
    ("reasoning.summary", RequestField::Reasoning),
    ("prompt_cache_key", RequestField::Cache),
    ("prompt_cache_retention", RequestField::Cache),
];

/// Request parameters with no portable field that the caller still asked
/// for, through the options' `extra`: a refusal of one names it as
/// [`Subject::Other`], the subject the model rules' drop warning uses.
/// "When reasoning effort is not none, remove temperature, top_p, and
/// top_logprobs. For Chat Completions, also remove logprobs"
/// (<https://developers.openai.com/api/docs/guides/latest-model>).
const EXTRA_PARAMS: &[&str] = &["logprobs", "top_logprobs"];

/// A refusal of something the caller asked for, as the provider words it:
/// the subject and an authored message, or `None` when the error is not one
/// the adapter recognizes. Each pattern is pinned by a fixture in this
/// module's tests.
///
/// - OpenAI's `unsupported_parameter` and `unsupported_value` codes, with a
///   `param` in [`WIRE_FIELDS`] or [`EXTRA_PARAMS`]. The value case keeps
///   its meaning: the value is refused, not the parameter.
/// - The same refusals with `code: null`, as the Responses API sends them:
///   the message prefix (`Unsupported parameter:` or `Unsupported value:`)
///   stands in for the code, when the type is absent or an invalid request.
///   A code, when present, always decides.
/// - Ollama's "`<model>` does not support tools".
/// - Ollama's "Multimodal data provided, but model does not support
///   multimodal requests.", which it nests as a JSON string inside the
///   message, so it is matched anywhere in it. Ollama names neither the
///   message nor the part, so the refusal has no subject.
/// - OpenAI Chat Completions' refusal of a content part the model does not
///   take (audio for a model without audio input): code `invalid_value`,
///   `param` `messages[N]`, and "Content blocks are expected to be either
///   …". The refusal has no subject: `N` counts the wire messages, which
///   start with the system prompts, not the caller's.
/// - GitHub Copilot's `invalid_reasoning_effort` (no `param`): a level the
///   model does not offer ("reasoning_effort \"none\" is not supported by
///   model …"), or any effort for a model without one ("… does not support
///   reasoning effort"), which refuses the parameter itself.
/// - OpenAI Chat Completions' refusal of function tools with a reasoning
///   effort on a model that takes them together only on the Responses API
///   (`gpt-6-astra`): `code: null`, `param` `reasoning_effort`, and
///   "Function tools with reasoning_effort are not supported for …". It
///   names the reasoning effort, not the tools, since the remedy it gives on
///   this endpoint is setting the effort to none.
fn upstream_refusal(error: &ApiError, status: Option<u16>) -> Option<(Option<Subject>, String)> {
    if status.is_some_and(|status| !(400..500).contains(&status)) {
        return None;
    }
    let field = |field| Some(Subject::Field { field });
    let what = match error.code.as_deref() {
        Some("unsupported_parameter") => "",
        Some("unsupported_value") => "this value of ",
        _ if error
            .message
            .trim_end_matches('.')
            .ends_with("does not support tools") =>
        {
            return Some((
                field(RequestField::Tools),
                "the model does not support tools".to_owned(),
            ));
        }
        _ if error.message.contains("does not support multimodal") => {
            return Some((
                None,
                "the model does not support multimodal input".to_owned(),
            ));
        }
        Some("invalid_value") if refuses_content_part(error) => {
            return Some((
                None,
                "the model does not support a content part that was sent".to_owned(),
            ));
        }
        Some("invalid_reasoning_effort") => {
            let what = if error.message.contains("does not support reasoning effort") {
                ""
            } else {
                "this value of "
            };
            return Some((
                field(RequestField::Reasoning),
                format!("the model does not support {what}`reasoning_effort`"),
            ));
        }
        None if refuses_tools_with_reasoning(error) => {
            return Some((
                field(RequestField::Reasoning),
                "the model does not support `reasoning_effort` with function tools on this \
                 endpoint"
                    .to_owned(),
            ));
        }
        None => codeless_refusal(error)?,
        _ => return None,
    };
    let param = error.param.as_deref()?;
    let (wire, subject) = match WIRE_FIELDS.iter().find(|(wire, _)| *wire == param) {
        Some((wire, refused)) => (*wire, field(*refused)),
        None => {
            let wire = EXTRA_PARAMS.iter().copied().find(|wire| *wire == param)?;
            let name = format!("`{wire}`");
            (wire, Some(Subject::Other { name }))
        }
    };
    Some((
        subject,
        format!("the model does not support {what}`{wire}`"),
    ))
}

/// Whether an `invalid_value` error refuses a content part: its `param`
/// names a message (`messages[1]`), and its message says which content
/// blocks the model takes. The same code on any other parameter
/// (`temperature`) is a malformed value, not a refusal.
fn refuses_content_part(error: &ApiError) -> bool {
    let names_message = error.param.as_deref().is_some_and(|param| {
        param
            .strip_prefix("messages[")
            .and_then(|rest| rest.strip_suffix(']'))
            .is_some_and(|index| !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()))
    });
    names_message && error.message.contains("Content blocks are expected")
}

/// Whether an error refuses function tools together with a reasoning
/// effort: its `param` is `reasoning_effort`, and its message says the two
/// are not supported together. The same `param` with any other wording is
/// not this refusal.
fn refuses_tools_with_reasoning(error: &ApiError) -> bool {
    error.param.as_deref() == Some("reasoning_effort")
        && error
            .message
            .contains("Function tools with reasoning_effort are not supported")
}

/// What a refusal without a code refuses, read from its message: `""` for
/// "Unsupported parameter: …", `"this value of "` for "Unsupported value:
/// …", or `None` when the message is neither or the type is not an invalid
/// request. The Responses API sends `{"message": "Unsupported parameter:
/// 'temperature' is not supported with this model.", "type":
/// "invalid_request_error", "param": "temperature", "code": null}`; a
/// stream's `error` event and `response.failed` carry no type.
fn codeless_refusal(error: &ApiError) -> Option<&'static str> {
    if !matches!(
        error.error_type.as_deref(),
        None | Some("invalid_request_error" | "invalid_request")
    ) {
        return None;
    }
    let message = error.message.trim();
    if message.starts_with("Unsupported parameter:") {
        Some("")
    } else if message.starts_with("Unsupported value:") {
        Some("this value of ")
    } else {
        None
    }
}

/// An operation the server refuses outright, as it words it: an authored
/// message, or `None` when the error is not one the adapter recognizes.
/// Pinned by a fixture in this module's tests.
///
/// - llama.cpp's server answers an embeddings request it was not started
///   for with `501` and "This server does not support embeddings. Start it
///   with `--embeddings`"; Ollama passes it through for a chat-only model.
///   Without this, the `501` would be a generic provider error.
/// - GitHub Copilot answers `unsupported_api_for_model` for a model it
///   serves only on another of its APIs ("model \"gpt-5.6-luna\" is not
///   accessible via the /chat/completions endpoint"): the model exists, but
///   not on this endpoint.
fn unsupported_operation(error: &ApiError, status: Option<u16>) -> Option<&'static str> {
    if error.code.as_deref() == Some("unsupported_api_for_model") {
        return Some("the model is not served on this API");
    }
    (status == Some(501)
        && error
            .message
            .trim_start()
            .starts_with("This server does not support embeddings"))
    .then_some("the server does not support embeddings")
}

/// Read a success response's body, up to `limit` bytes. A failed read is
/// the transport's error
/// ([`TransportError::into_error`](allama_http::TransportError::into_error)),
/// a timeout and a body past `limit` included, attributed to `provider` and
/// `request_id`.
pub(crate) async fn read_body(
    response: HttpResponse,
    limit: usize,
    provider: &str,
    request_id: Option<&str>,
) -> Result<(u16, Bytes)> {
    let status = response.status;
    let bytes = response.bytes_limited(limit).await.map_err(|err| {
        err.into_error()
            .with_provider(provider)
            .with_request_id_opt(request_id)
    })?;
    Ok((status, bytes))
}

/// Decode a success body [`read_body`] read. A body that is not a `T` is
/// [`allama_http::malformed_response`], attributed to `provider` and
/// `request_id`.
pub(crate) fn decode_body<T: DeserializeOwned>(
    status: u16,
    bytes: &[u8],
    provider: &str,
    request_id: Option<&str>,
) -> Result<T> {
    serde_json::from_slice(bytes).map_err(|err| {
        allama_http::malformed_response(status, bytes, err)
            .with_provider(provider)
            .with_request_id_opt(request_id)
    })
}

/// Map a structured API error, optionally with an HTTP status.
///
/// Without a status (errors reported inside a stream or response body),
/// the code and type alone decide the category. Shared by the Chat
/// Completions and Responses adapters.
///
/// The constructor message is [authored](Error::authored): `HTTP {status}`,
/// or `provider reported an error` without a status. The error's own
/// message is the [upstream message](Error::upstream_message), and its
/// `param` is the rejected field ([`ErrorContext::param`](allama::ErrorContext::param)).
///
/// This is the built-in classification. To also honor a dialect's
/// [`quota_error_codes`](crate::CompatQuirks::quota_error_codes), use
/// [`CompatQuirks::map_api_error`](crate::CompatQuirks::map_api_error).
#[must_use]
pub(crate) fn map_api_error(
    error: &ApiError,
    status: Option<u16>,
    request_id: Option<String>,
    retry_after: Option<Duration>,
) -> Error {
    classify_api_error(error, status, request_id, retry_after, &[])
}

/// [`map_api_error`] with extra codes (or types) that mean quota or billing
/// exhaustion.
pub(crate) fn classify_api_error(
    error: &ApiError,
    status: Option<u16>,
    request_id: Option<String>,
    retry_after: Option<Duration>,
    quota_codes: &[Cow<'static, str>],
) -> Error {
    let mapped = classify(error, status, quota_codes).authored();
    let mapped = match status {
        Some(status) => mapped.with_status(status),
        None => mapped,
    };
    with_api_error(mapped, error)
        .with_request_id_opt(request_id.as_deref())
        .with_retry_after_opt(retry_after)
}

/// Attach what the API error says beyond its category: its message as the
/// upstream message, the rejected field, the code and the type.
fn with_api_error(mapped: Error, error: &ApiError) -> Error {
    mapped
        .with_upstream_message(error.message.clone())
        .with_param_opt(error.param.as_deref())
        .with_provider_code_opt(error.code.as_deref())
        .with_provider_type_opt(error.error_type.as_deref())
}

/// Pick the category from the code, type, status and `error.message`; the
/// result carries only the constructor message. The caller adds the context
/// and marks it [authored](Error::authored).
fn classify(error: &ApiError, status: Option<u16>, quota_codes: &[Cow<'static, str>]) -> Error {
    let message = match status {
        Some(status) => format!("HTTP {status}"),
        None => "provider reported an error".to_owned(),
    };
    let text = error.message.as_str();
    let code = error.code.as_deref().unwrap_or_default();
    // An in-band error whose code is an HTTP status (OpenRouter's
    // `{"error": {"code": 429}}`, vLLM's `"code": 400`) is classified by it.
    let status = status.or_else(|| {
        code.parse::<u16>()
            .ok()
            .filter(|code| (400..600).contains(code))
    });
    let error_type = error.error_type.as_deref().unwrap_or_default();
    let lower = text.to_ascii_lowercase();

    // Codes and types are matched together: providers put the same signal
    // in either field.
    let is = |values: &[&str]| values.contains(&code) || values.contains(&error_type);
    let is_extra_quota = quota_codes
        .iter()
        .any(|value| value == code || value == error_type);
    if is(QUOTA_CODES) || is_extra_quota {
        Error::quota_exhausted(message)
    } else if let Some((subject, refusal)) = upstream_refusal(error, status) {
        Error::unsupported_by_provider(subject, format!("{message}: {refusal}"))
    } else if let Some(refusal) = unsupported_operation(error, status) {
        Error::unsupported_by_provider(None, format!("{message}: {refusal}"))
    } else if is(&["context_length_exceeded", "string_above_max_length"]) {
        Error::context_length_exceeded(message, context_limit(text))
    } else if is(&[
        "content_filter",
        "content_policy_violation",
        "image_content_policy_violation",
    ]) {
        Error::content_filtered(message)
    // GitHub Copilot answers `model_not_supported` (400, `param: model`) for
    // a model the token's plan does not serve (seen live, 2026-10-14).
    } else if is(&["model_not_found", "model_not_supported"]) {
        Error::model_not_found(message)
    } else if is(&[
        "invalid_api_key",
        "invalid_authentication",
        "authentication_error",
    ]) {
        Error::auth(message)
    } else if is(&[
        "rate_limit_exceeded",
        "rate_limit_error",
        "too_many_requests",
    ]) {
        Error::rate_limited(message)
    } else if is(&["overloaded", "server_is_overloaded", "service_unavailable"]) {
        Error::overloaded(message)
    } else {
        match status {
            Some(404 | 410) if lower.contains("model") => Error::model_not_found(message),
            Some(400)
                if lower.contains("maximum context length")
                    || lower.contains("context window")
                    || lower.contains("too many tokens") =>
            {
                Error::context_length_exceeded(message, context_limit(text))
            }
            // What the status means on its own, shared by every adapter.
            Some(status) => allama_http::status_error(status, message),
            // In-band errors (no HTTP status): streams and response bodies.
            None if is(&["timeout", "vector_store_timeout"]) => Error::timeout(message),
            None if is(INPUT_ERROR_CODES) => Error::invalid_request(message),
            // `invalid_request` is the Open Responses spelling.
            None if code.starts_with("invalid_")
                || matches!(error_type, "invalid_request_error" | "invalid_request") =>
            {
                Error::invalid_request(message)
            }
            // `server_error` and anything unrecognized: retryable.
            None => Error::provider(message),
        }
    }
}

/// Map an HTTP error response.
///
/// The status and headers are read with [`allama_http::error_context`]
/// (status, `x-request-id` or `request-id`, retry hint, server retry
/// directive); an endpoint that names its request id or retry hint header
/// differently sets those afterwards (`with_request_id`, `with_retry_after`).
/// The body is read with [`allama_http::error_envelope`], plus Cloudflare's
/// `{"errors": [..]}` envelope. An envelope's message becomes the
/// [upstream message](Error::upstream_message); a body that is no envelope
/// is kept only in the (capped) [`body`](allama::ErrorContext::body), and
/// the message is `HTTP {status}`.
///
/// This is the built-in classification. To also honor a dialect's
/// [`quota_error_codes`](crate::CompatQuirks::quota_error_codes), use
/// [`CompatQuirks::map_http_error`](crate::CompatQuirks::map_http_error).
#[must_use]
pub(crate) fn map_http_error(status: u16, headers: &Headers, body: &str) -> Error {
    classify_http_error(status, headers, body, &[])
}

/// [`map_http_error`] with extra codes (or types) that mean quota or billing
/// exhaustion.
pub(crate) fn classify_http_error(
    status: u16,
    headers: &Headers,
    body: &str,
    quota_codes: &[Cow<'static, str>],
) -> Error {
    let context = allama_http::error_context(status, headers);
    let mapped = match envelope_error(body) {
        Some(error) => with_api_error(
            classify(&error, Some(status), quota_codes)
                .with_context(context)
                .authored(),
            &error,
        ),
        // The body still informs the category (an HTML 404 page that names
        // the model), but never the message.
        None => classify(&ApiError::new(body), Some(status), quota_codes)
            .with_context(context)
            .authored(),
    };
    mapped.with_body_opt((!body.is_empty()).then_some(body))
}

/// The error in a body that is a known error envelope: one
/// [`allama_http::error_envelope`] reads, Cloudflare's `{"errors": [{…}]}`,
/// or either wrapped in an array (Gemini's OpenAI-compatible endpoint sends
/// `[{"error": {…}}]`).
pub(crate) fn envelope_error(body: &str) -> Option<ApiError> {
    let value: serde_json::Value = serde_json::from_str(body.trim()).ok()?;
    let value = match value {
        serde_json::Value::Array(mut items) if items.len() == 1 => items.pop()?,
        value => value,
    };
    allama_http::error_envelope_value(&value)
        .map(ApiError::from)
        .or_else(|| {
            // Cloudflare's envelope: `{"errors": [{"message": .., "code": ..}]}`.
            serde_json::from_value::<ErrorListResponse>(value)
                .ok()?
                .errors
                .into_iter()
                .next()
        })
}

/// The error a success body reports instead of a result: an error envelope
/// with no `choices` (OpenRouter answers some upstream failures with HTTP
/// 200 and `{"error": {"code": 502, …}}`). `None` for any other body.
pub(crate) fn success_body_error(bytes: &[u8]) -> Option<ApiError> {
    let value: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    crate::streaming::in_band_error(&value)
}

#[derive(serde::Deserialize)]
struct ErrorListResponse {
    errors: Vec<ApiError>,
}

/// The context window a context-overflow message states, as in OpenAI's
/// "This model's maximum context length is 128000 tokens. ..." (also used by
/// vLLM and other compatible servers).
fn context_limit(message: &str) -> Option<u64> {
    const PHRASE: &str = "maximum context length is";
    let lower = message.to_ascii_lowercase();
    let rest = lower[lower.find(PHRASE)? + PHRASE.len()..].trim_start();
    let digits: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == ',')
        .filter(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn headers(pairs: &[(&str, &str)]) -> Headers {
        pairs.iter().copied().collect()
    }

    fn http(status: u16, body: serde_json::Value) -> Error {
        map_http_error(
            status,
            &headers(&[("x-request-id", "req")]),
            &body.to_string(),
        )
    }

    #[test]
    fn quota_and_billing_are_not_retryable() {
        for body in [
            json!({"error": {"message": "quota", "type": "insufficient_quota", "code": "insufficient_quota"}}),
            json!({"error": {"message": "credits", "type": "invalid_request_error", "code": "credit_balance_exhausted"}}),
            json!({"error": {"message": "spend", "code": "project_spend_limit_exceeded"}}),
        ] {
            let err = http(429, body);
            assert!(matches!(err, Error::QuotaExhausted { .. }), "{err:?}");
            assert!(!err.is_retryable());
        }
        assert!(matches!(http(402, json!({})), Error::QuotaExhausted { .. }));
    }

    #[test]
    fn dialects_add_quota_codes() {
        let body = json!({"error": {"message": "monthly quota used", "code": "quota_exceeded"}})
            .to_string();
        // Unknown to the built-in classification: a retryable rate limit.
        let default = map_http_error(429, &Headers::new(), &body);
        assert!(matches!(default, Error::RateLimited { .. }), "{default:?}");

        let mut quirks = crate::CompatQuirks::new("openai-compat:vendor");
        quirks.quota_error_codes.push("quota_exceeded".into());
        let mapped = quirks.map_http_error(429, &headers(&[("x-request-id", "req")]), &body);
        assert!(matches!(mapped, Error::QuotaExhausted { .. }), "{mapped:?}");
        assert!(!mapped.is_retryable());
        assert_eq!(mapped.request_id(), Some("req"));
        assert_eq!(mapped.status(), Some(429));
        assert!(mapped.context().unwrap().body.is_some());

        // The type field counts too, and in-band errors (no status) as well.
        let in_band = quirks.map_api_error(
            &ApiError::new("over quota").with_type("quota_exceeded"),
            None,
            None,
            None,
        );
        assert!(
            matches!(in_band, Error::QuotaExhausted { .. }),
            "{in_band:?}"
        );
    }

    #[test]
    fn open_responses_error_types_without_status() {
        let typed = |error_type: &str| {
            map_api_error(&ApiError::new("x").with_type(error_type), None, None, None)
        };
        let invalid = typed("invalid_request");
        assert!(
            matches!(invalid, Error::InvalidRequest { .. }),
            "{invalid:?}"
        );
        assert!(!invalid.is_retryable());
        assert!(matches!(
            typed("too_many_requests"),
            Error::RateLimited { .. }
        ));
    }

    #[test]
    fn cloudflare_error_envelope_and_retired_models() {
        let err = http(
            410,
            json!({"errors": [{"message": "AiError: Model has been deprecated: @cf/x", "code": 5028}], "success": false}),
        );
        assert!(matches!(err, Error::ModelNotFound { .. }), "{err:?}");
        assert_eq!(err.message(), "HTTP 410");
        assert_eq!(
            err.upstream_message(),
            Some("AiError: Model has been deprecated: @cf/x")
        );
        assert_eq!(
            err.context().unwrap().provider_code.as_deref(),
            Some("5028")
        );
    }

    #[test]
    fn context_length_reports_the_stated_limit() {
        let openai = http(
            400,
            json!({"error": {
                "message": "This model's maximum context length is 128000 tokens. However, your messages resulted in 130512 tokens. Please reduce the length of the messages.",
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
            }}),
        );
        assert!(
            matches!(
                openai,
                Error::ContextLengthExceeded {
                    context_window: Some(128_000),
                    ..
                }
            ),
            "{openai:?}"
        );

        // vLLM and other compatible servers send the message without a code.
        let vllm = http(
            400,
            json!({"error": {
                "message": "This model's maximum context length is 4,096 tokens. However, you requested 5000 tokens (4000 in the messages, 1000 in the completion).",
                "type": "BadRequestError",
            }}),
        );
        assert!(
            matches!(
                vllm,
                Error::ContextLengthExceeded {
                    context_window: Some(4096),
                    ..
                }
            ),
            "{vllm:?}"
        );

        let unstated = http(
            400,
            json!({"error": {"message": "too long", "code": "context_length_exceeded"}}),
        );
        assert!(
            matches!(
                unstated,
                Error::ContextLengthExceeded {
                    context_window: None,
                    ..
                }
            ),
            "{unstated:?}"
        );
    }

    #[test]
    fn context_limit_parsing() {
        assert_eq!(
            context_limit("This model's maximum context length is 8192 tokens."),
            Some(8192)
        );
        assert_eq!(
            context_limit("maximum context length is 1,048,576 tokens"),
            Some(1_048_576)
        );
        assert_eq!(context_limit("maximum context length exceeded"), None);
        assert_eq!(context_limit("you requested 5000 tokens"), None);
    }

    #[test]
    fn status_and_code_mapping() {
        assert!(matches!(
            http(401, json!({"error": {"message": "bad"}})),
            Error::Auth { .. }
        ));
        assert!(matches!(
            http(
                400,
                json!({"error": {"message": "x", "code": "context_length_exceeded"}})
            ),
            Error::ContextLengthExceeded { .. }
        ));
        assert!(matches!(
            http(
                404,
                json!({"error": {"message": "The model `x` does not exist", "code": "model_not_found"}})
            ),
            Error::ModelNotFound { .. }
        ));
        // GitHub Copilot, for a model the token's plan does not serve: the
        // code, type, param and status of a live reply (2026-10-14); the
        // message is a stand-in.
        let unserved = http(
            400,
            json!({"error": {"message": "The requested model is not supported.",
                             "code": "model_not_supported", "param": "model",
                             "type": "invalid_request_error"}}),
        );
        assert!(
            matches!(unserved, Error::ModelNotFound { .. }),
            "{unserved:?}"
        );
        assert_eq!(unserved.status(), Some(400));
        let rate = http(
            429,
            json!({"error": {"message": "slow down", "code": "rate_limit_exceeded"}}),
        );
        assert!(matches!(rate, Error::RateLimited { .. }));
        assert_eq!(rate.request_id(), Some("req"));
        assert_eq!(rate.status(), Some(429));
        assert_eq!(
            rate.context().unwrap().provider_code.as_deref(),
            Some("rate_limit_exceeded")
        );
        assert!(matches!(http(503, json!({})), Error::Overloaded { .. }));
        let server = http(500, json!({"error": {"message": "boom"}}));
        assert!(matches!(server, Error::Provider { .. }));
        assert!(server.is_retryable());
        assert!(server.context().unwrap().body.is_some());
    }

    #[test]
    fn statuses_mean_what_they_mean_on_every_adapter() {
        // Gateway and proxy statuses arrive without an envelope.
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
        // A request the client closed is neither retried nor a rejected
        // request, whatever the envelope says.
        let closed = http(499, json!({"error": {"message": "client closed request"}}));
        assert!(matches!(closed, Error::Provider { .. }), "{closed:?}");
        assert!(!closed.is_retryable());
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
        assert_eq!(err.to_string(), "provider error (502): HTTP 502");
        assert_eq!(
            err.as_log().to_string(),
            "kind=provider status=502 body_len=34 message=\"HTTP 502\""
        );

        // The body still informs the category.
        let err = map_http_error(404, &Headers::new(), "<p>model gpt-x not found</p>");
        assert!(matches!(err, Error::ModelNotFound { .. }), "{err:?}");
    }

    #[test]
    fn envelope_messages_are_upstream_text() {
        let err = http(
            400,
            json!({"error": {
                "message": "Invalid value: 'my SSN is 123-45-6789'",
                "type": "invalid_request_error",
                "code": "invalid_value",
                "param": "messages.[1].content",
            }}),
        );
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
        assert_eq!(err.message(), "HTTP 400");
        assert_eq!(err.message_origin(), allama::MessageOrigin::Authored);
        assert_eq!(
            err.upstream_message(),
            Some("Invalid value: 'my SSN is 123-45-6789'")
        );
        assert_eq!(
            err.as_log().to_string(),
            "kind=invalid_request status=400 provider_code=invalid_value \
             provider_type=invalid_request_error param=messages.[1].content request_id=req \
             body_len=147 message=\"HTTP 400\""
        );
        assert_eq!(
            err.to_string(),
            "invalid request: HTTP 400: Invalid value: 'my SSN is 123-45-6789'"
        );

        // Legacy vLLM and native Ollama envelopes too.
        let vllm = http(
            400,
            json!({"object": "error", "message": "bad SENTINEL", "type": "BadRequestError", "code": 400}),
        );
        assert_eq!(vllm.upstream_message(), Some("bad SENTINEL"));
        assert_eq!(
            vllm.context().unwrap().provider_code.as_deref(),
            Some("400")
        );
        let ollama = http(404, json!({"error": "model 'x' not found"}));
        assert!(matches!(ollama, Error::ModelNotFound { .. }), "{ollama:?}");
        assert_eq!(ollama.upstream_message(), Some("model 'x' not found"));
    }

    #[test]
    fn error_logs_exclude_upstream_text() {
        use allama_conformance::error_kinds::ErrorKindProbe;

        // Cloudflare's `errors` list is the envelope the common shapes miss.
        let cloudflare = |kind: ErrorKindProbe, upstream: &str| {
            let (message, code) = match kind {
                ErrorKindProbe::InvalidRequest => ("No route for that URI", 7003),
                ErrorKindProbe::ModelNotFound => {
                    ("AiError: Model has been deprecated: @cf/x", 5028)
                }
                _ => return None,
            };
            let message = format!("{message} {upstream}");
            Some(json!({"errors": [{"message": message.trim_end(), "code": code}]}).to_string())
        };
        allama_conformance::assert_error_log_excludes_body(cloudflare, |status, headers, body| {
            map_http_error(status, &headers.iter().copied().collect(), &body)
        });
    }

    fn openai_error(message: &str, param: &str, code: &str) -> String {
        json!({"error": {
            "message": message,
            "type": "invalid_request_error",
            "param": param,
            "code": code
        }})
        .to_string()
    }

    /// OpenAI refuses parameters a reasoning model does not take with
    /// `unsupported_parameter` or `unsupported_value` and the parameter in
    /// `param` (the reply shapes of
    /// <https://platform.openai.com/docs/guides/reasoning>'s unsupported
    /// parameters, as the API returns them for `o3-mini` and `gpt-5`).
    #[test]
    fn refused_parameters_are_unsupported_by_the_provider() {
        let parameter = map_http_error(
            400,
            &Headers::new(),
            &openai_error(
                "Unsupported parameter: 'temperature' is not supported with this model.",
                "temperature",
                "unsupported_parameter",
            ),
        );
        assert!(
            matches!(parameter, Error::Unsupported { .. }),
            "{parameter:?}"
        );
        assert_eq!(
            parameter.unsupported_stage(),
            Some(allama::UnsupportedStage::Provider)
        );
        assert_eq!(
            parameter.unsupported_field(),
            Some(RequestField::Temperature)
        );
        assert_eq!(parameter.status(), Some(400));
        assert_eq!(parameter.param(), Some("temperature"));
        assert_eq!(
            parameter.message(),
            "HTTP 400: the model does not support `temperature`"
        );

        let value = map_http_error(
            400,
            &Headers::new(),
            &openai_error(
                "Unsupported value: 'temperature' does not support 0.2 with this model. Only the \
                 default (1) value is supported.",
                "temperature",
                "unsupported_value",
            ),
        );
        assert_eq!(value.unsupported_field(), Some(RequestField::Temperature));
        assert_eq!(
            value.message(),
            "HTTP 400: the model does not support this value of `temperature`"
        );

        // Responses spells nested parameters with dots.
        let effort = map_http_error(
            400,
            &Headers::new(),
            &openai_error(
                "Unsupported parameter: 'reasoning.effort' is not supported with this model.",
                "reasoning.effort",
                "unsupported_parameter",
            ),
        );
        assert_eq!(effort.unsupported_field(), Some(RequestField::Reasoning));

        // Ollama refuses tools for a model without tool support.
        let ollama = map_http_error(
            400,
            &Headers::new(),
            &json!({"error": "registry.ollama.ai/library/gemma2:latest does not support tools"})
                .to_string(),
        );
        assert_eq!(ollama.unsupported_field(), Some(RequestField::Tools));
        assert_eq!(
            ollama.unsupported_stage(),
            Some(allama::UnsupportedStage::Provider)
        );
    }

    /// GitHub Copilot's refusals, as recorded by Pydantic AI
    /// (`tests/models/cassettes/test_github_copilot/`, 2026-09-07): a
    /// reasoning effort the model does not offer, an effort for a model
    /// without one, and a model served only on the Responses API. Copilot
    /// sends them as `text/plain`, with no `param` or `type`.
    #[test]
    fn github_copilot_refusals_are_unsupported() {
        use allama::UnsupportedStage::Provider;

        let level = map_http_error(
            400,
            &Headers::new(),
            r#"{"error":{"message":"reasoning_effort \"none\" is not supported by model claude-sonnet-5; supported values: [low medium high xhigh max]","code":"invalid_reasoning_effort"}}"#,
        );
        assert!(matches!(level, Error::Unsupported { .. }), "{level:?}");
        assert_eq!(level.unsupported_stage(), Some(Provider));
        assert_eq!(level.unsupported_field(), Some(RequestField::Reasoning));
        assert_eq!(
            level.message(),
            "HTTP 400: the model does not support this value of `reasoning_effort`"
        );

        let none = map_http_error(
            400,
            &Headers::new(),
            r#"{"error":{"message":"reasoning_effort \"medium\" was provided, but model claude-haiku-4.5 does not support reasoning effort","code":"invalid_reasoning_effort"}}"#,
        );
        assert_eq!(none.unsupported_field(), Some(RequestField::Reasoning));
        assert_eq!(
            none.message(),
            "HTTP 400: the model does not support `reasoning_effort`"
        );

        let api = map_http_error(
            400,
            &Headers::new(),
            r#"{"error":{"message":"model \"gpt-5.6-luna\" is not accessible via the /chat/completions endpoint","code":"unsupported_api_for_model"}}"#,
        );
        assert!(matches!(api, Error::Unsupported { .. }), "{api:?}");
        assert_eq!(api.unsupported_stage(), Some(Provider));
        assert_eq!(api.unsupported_subject(), None);
        assert!(
            api.upstream_message()
                .is_some_and(|m| m.contains("/chat/completions")),
            "{api:?}"
        );
    }

    /// Ollama refuses an image for a text-only
    /// model with its own error nested as a JSON string inside the message
    /// (recorded from Ollama 0.40.0 with `qwen3:0.6b` and a PNG
    /// `image_url` part on `/v1/chat/completions`). The refusal is the
    /// provider's, so code can branch on it, with no subject: Ollama names
    /// neither the message nor the part.
    #[test]
    fn ollama_multimodal_refusal_is_unsupported() {
        let body = r#"{"error":{"message":"{\"error\":{\"code\":400,\"message\":\"Multimodal data provided, but model does not support multimodal requests.\",\"type\":\"invalid_request_error\"}}","type":"invalid_request_error","param":null,"code":null}}"#;
        let err = map_http_error(400, &Headers::new(), body);
        assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
        assert_eq!(
            err.unsupported_stage(),
            Some(allama::UnsupportedStage::Provider)
        );
        assert_eq!(err.unsupported_subject(), None);
        assert_eq!(err.status(), Some(400));
        assert_eq!(
            err.message(),
            "HTTP 400: the model does not support multimodal input"
        );
        assert!(
            err.upstream_message()
                .is_some_and(|m| m.contains("Multimodal data provided")),
            "{err:?}"
        );

        // Another 400 with no recognisable text stays an invalid request
        // (the PDF case: "invalid message format").
        let other = map_http_error(
            400,
            &Headers::new(),
            r#"{"error":{"message":"invalid message format","type":"invalid_request_error","param":null,"code":null}}"#,
        );
        assert!(matches!(other, Error::InvalidRequest { .. }), "{other:?}");
    }

    /// The parameters a model rule used to drop while a model reasons
    /// (ADR 0015 removed the rules, so the request carries them): OpenAI
    /// refuses each with `unsupported_parameter` or `unsupported_value` and
    /// the parameter in `param`, and each maps to a refusal of what the
    /// caller asked for. "When reasoning effort is not none, remove
    /// temperature, top_p, and top_logprobs. For Chat Completions, also
    /// remove logprobs"
    /// (<https://developers.openai.com/api/docs/guides/latest-model>);
    /// `reasoning_effort` on a model that does not reason, and `tools` where
    /// a model takes none on Chat Completions, use the same reply shape.
    #[test]
    fn parameters_a_reasoning_model_refuses_name_their_subject() {
        let refused = |param: &str, code: &str| {
            map_http_error(
                400,
                &Headers::new(),
                &openai_error(
                    &format!("Unsupported parameter: '{param}' is not supported with this model."),
                    param,
                    code,
                ),
            )
        };
        for (param, field) in [
            ("temperature", RequestField::Temperature),
            ("top_p", RequestField::TopP),
            ("frequency_penalty", RequestField::FrequencyPenalty),
            ("presence_penalty", RequestField::PresencePenalty),
            ("reasoning_effort", RequestField::Reasoning),
            ("tools", RequestField::Tools),
            ("response_format", RequestField::ResponseFormat),
        ] {
            for code in ["unsupported_parameter", "unsupported_value"] {
                let err = refused(param, code);
                assert_eq!(err.unsupported_field(), Some(field), "{param} {code}");
                assert_eq!(
                    err.unsupported_stage(),
                    Some(allama::UnsupportedStage::Provider),
                    "{param}"
                );
            }
        }
        for param in ["logprobs", "top_logprobs"] {
            let err = refused(param, "unsupported_parameter");
            assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
            assert_eq!(
                err.unsupported_stage(),
                Some(allama::UnsupportedStage::Provider)
            );
            assert_eq!(
                err.unsupported_subject(),
                Some(&Subject::Other {
                    name: format!("`{param}`")
                })
            );
            assert_eq!(
                err.message(),
                format!("HTTP 400: the model does not support `{param}`")
            );
        }
    }

    /// What does not resolve stays an `InvalidRequest`: a parameter the
    /// dialect chose (`max_tokens`, where the model wants
    /// `max_completion_tokens`), one outside the table, a nested message
    /// path (whose index is the wire's, not the caller's), and a refusal
    /// code without a `param`.
    #[test]
    fn unresolved_refusals_stay_invalid_requests() {
        let cases = [
            openai_error(
                "Unsupported parameter: 'max_tokens' is not supported with this model. Use \
                 'max_completion_tokens' instead.",
                "max_tokens",
                "unsupported_parameter",
            ),
            openai_error(
                "Unsupported parameter: 'user_data'.",
                "user_data",
                "unsupported_parameter",
            ),
            openai_error(
                "Unsupported value: messages[1].content[0].type.",
                "messages.[1].content.[0].type",
                "unsupported_value",
            ),
            json!({"error": {"message": "Unsupported value.", "type": "invalid_request_error",
                             "code": "unsupported_value"}})
            .to_string(),
        ];
        for body in cases {
            let err = map_http_error(400, &Headers::new(), &body);
            assert!(
                matches!(err, Error::InvalidRequest { .. }),
                "{body}: {err:?}"
            );
        }
        // A server error is never a refusal.
        let err = map_http_error(
            500,
            &Headers::new(),
            &openai_error("x", "temperature", "unsupported_parameter"),
        );
        assert!(!matches!(err, Error::Unsupported { .. }), "{err:?}");
    }

    /// The Responses API refuses a sampling parameter on a reasoning model
    /// with `code: null` (captured live for `temperature`): the message
    /// prefix stands in for the code.
    #[test]
    fn refusals_without_a_code_are_read_from_the_message() {
        let codeless = |message: &str, error_type: Option<&str>, param: Option<&str>| {
            json!({"error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": null
            }})
            .to_string()
        };
        let parameter = "Unsupported parameter: 'temperature' is not supported with this model.";

        let typed = map_http_error(
            400,
            &Headers::new(),
            &codeless(
                parameter,
                Some("invalid_request_error"),
                Some("temperature"),
            ),
        );
        assert!(matches!(typed, Error::Unsupported { .. }), "{typed:?}");
        assert_eq!(typed.unsupported_field(), Some(RequestField::Temperature));
        assert_eq!(
            typed.unsupported_stage(),
            Some(allama::UnsupportedStage::Provider)
        );
        assert_eq!(
            typed.message(),
            "HTTP 400: the model does not support `temperature`"
        );
        assert_eq!(typed.upstream_message(), Some(parameter));
        assert_eq!(typed.param(), Some("temperature"));
        assert_eq!(typed.context().unwrap().provider_code, None);

        // A stream's `error` event and `response.failed` carry no type; the
        // Open Responses spelling of the type counts too.
        for error_type in [None, Some("invalid_request")] {
            let err = map_http_error(
                400,
                &Headers::new(),
                &codeless(parameter, error_type, Some("temperature")),
            );
            assert_eq!(
                err.unsupported_field(),
                Some(RequestField::Temperature),
                "{error_type:?}"
            );
        }
        let in_band = map_api_error(
            &ApiError::new(format!("  {parameter}")).with_param("top_p"),
            None,
            None,
            None,
        );
        assert_eq!(in_band.unsupported_field(), Some(RequestField::TopP));

        // "Unsupported value:" keeps its meaning.
        let value = map_http_error(
            400,
            &Headers::new(),
            &codeless(
                "Unsupported value: 'reasoning.effort' does not support 'minimal' with this \
                 model.",
                Some("invalid_request_error"),
                Some("reasoning.effort"),
            ),
        );
        assert_eq!(value.unsupported_field(), Some(RequestField::Reasoning));
        assert_eq!(
            value.message(),
            "HTTP 400: the model does not support this value of `reasoning.effort`"
        );

        // Without a param, with one the table does not resolve, with another
        // type or with another message, the reply stays an invalid request.
        for body in [
            codeless(parameter, Some("invalid_request_error"), None),
            codeless(
                "Unsupported parameter: 'max_tokens' is not supported with this model.",
                Some("invalid_request_error"),
                Some("max_tokens"),
            ),
            codeless(parameter, Some("server_error"), Some("temperature")),
            codeless(
                "'temperature' is out of range.",
                Some("invalid_request_error"),
                Some("temperature"),
            ),
        ] {
            let err = map_http_error(400, &Headers::new(), &body);
            assert!(
                matches!(err, Error::InvalidRequest { .. }),
                "{body}: {err:?}"
            );
        }

        // A code, when present, decides; a server error is never a refusal.
        let coded = map_http_error(
            400,
            &Headers::new(),
            &openai_error(parameter, "temperature", "invalid_value"),
        );
        assert!(matches!(coded, Error::InvalidRequest { .. }), "{coded:?}");
        let server = map_http_error(
            500,
            &Headers::new(),
            &codeless(
                parameter,
                Some("invalid_request_error"),
                Some("temperature"),
            ),
        );
        assert!(matches!(server, Error::Provider { .. }), "{server:?}");
    }

    /// OpenAI Chat Completions refuses a part the
    /// model does not take (recorded from `gpt-5-mini` with a WAV
    /// `input_audio` part) with a structured `invalid_value` naming the wire
    /// message. The wire index counts the system prompt (`messages[1]`
    /// with one), so the refusal has no subject.
    #[test]
    fn openai_content_part_refusal_is_unsupported() {
        for index in [0, 1] {
            let body = json!({"error": {
                "message": format!("Invalid 'messages[{index}]'. Content blocks are expected to be either text or image_url type."),
                "type": "invalid_request_error",
                "param": format!("messages[{index}]"),
                "code": "invalid_value"
            }})
            .to_string();
            let err = map_http_error(400, &Headers::new(), &body);
            assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
            assert_eq!(
                err.unsupported_stage(),
                Some(allama::UnsupportedStage::Provider)
            );
            assert_eq!(err.unsupported_subject(), None);
            assert_eq!(err.status(), Some(400));
            assert_eq!(err.param(), Some(format!("messages[{index}]").as_str()));
            assert_eq!(
                err.message(),
                "HTTP 400: the model does not support a content part that was sent"
            );
            assert!(
                err.upstream_message()
                    .is_some_and(|m| m.contains("Content blocks are expected")),
                "{err:?}"
            );
        }

        // `invalid_value` on a message with other wording, or the wording
        // on a parameter that is not a message, stays an invalid request.
        for (message, param) in [
            ("Invalid 'messages[0]'. Expected an object.", "messages[0]"),
            (
                "Content blocks are expected to be either text or image_url type.",
                "messages",
            ),
            (
                "Content blocks are expected to be either text or image_url type.",
                "messages[0].content",
            ),
        ] {
            let body = openai_error(message, param, "invalid_value");
            let err = map_http_error(400, &Headers::new(), &body);
            assert!(
                matches!(err, Error::InvalidRequest { .. }),
                "{param}: {err:?}"
            );
        }
    }

    /// OpenAI Chat Completions refuses function tools with a reasoning
    /// effort on `gpt-6-astra` (recorded 2026-10-13, `code: null`). The reply
    /// tells the caller to set the effort to none, so it is the provider's
    /// refusal of the reasoning field.
    #[test]
    fn tools_with_a_reasoning_effort_refusal_names_the_reasoning() {
        let upstream = "Function tools with reasoning_effort are not supported for gpt-6-astra \
                        in /v1/chat/completions. To use function tools, use /v1/responses or set \
                        reasoning_effort to 'none'.";
        let recorded = json!({"error": {
            "message": upstream,
            "type": "invalid_request_error",
            "param": "reasoning_effort",
            "code": null
        }})
        .to_string();
        let err = map_http_error(400, &Headers::new(), &recorded);
        assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
        assert_eq!(err.unsupported_field(), Some(RequestField::Reasoning));
        assert_eq!(
            err.unsupported_stage(),
            Some(allama::UnsupportedStage::Provider)
        );
        assert_eq!(err.param(), Some("reasoning_effort"));
        assert_eq!(err.upstream_message(), Some(upstream));
        assert_eq!(
            err.message(),
            "HTTP 400: the model does not support `reasoning_effort` with function tools on \
             this endpoint"
        );

        // Another refusal worded differently on the same parameter keeps its
        // own reading, and the wording on another parameter is not this one.
        let other = map_http_error(
            400,
            &Headers::new(),
            &json!({"error": {
                "message": "Invalid value for reasoning_effort.",
                "type": "invalid_request_error",
                "param": "reasoning_effort",
                "code": null
            }})
            .to_string(),
        );
        assert!(matches!(other, Error::InvalidRequest { .. }), "{other:?}");
        let elsewhere = map_http_error(
            400,
            &Headers::new(),
            &json!({"error": {
                "message": upstream,
                "type": "invalid_request_error",
                "param": "tools",
                "code": null
            }})
            .to_string(),
        );
        assert!(
            matches!(elsewhere, Error::InvalidRequest { .. }),
            "{elsewhere:?}"
        );
    }

    /// llama.cpp's server refuses embeddings it was not started for with a
    /// `501`, and Ollama passes the reply through for a chat-only model
    /// (llama.cpp's `format_error_response` with `ERROR_TYPE_NOT_SUPPORTED`,
    /// in its own envelope and as Ollama's OpenAI layer rewraps it). It is
    /// the provider's refusal, not a retryable server error, and its text
    /// stays out of the log.
    #[test]
    fn a_server_without_embeddings_refuses_them() {
        let message = "This server does not support embeddings. Start it with `--embeddings`";
        for body in [
            json!({"error": {"code": 501, "message": message, "type": "not_supported_error"}}),
            json!({"error": {"message": message, "type": "api_error", "param": null, "code": null}}),
        ] {
            let err = http(501, body.clone());
            assert!(matches!(err, Error::Unsupported { .. }), "{body}: {err:?}");
            assert_eq!(
                err.unsupported_stage(),
                Some(allama::UnsupportedStage::Provider)
            );
            assert_eq!(err.unsupported_subject(), None);
            assert_eq!(
                err.message(),
                "HTTP 501: the server does not support embeddings"
            );
            assert_eq!(err.status(), Some(501));
            assert_eq!(err.request_id(), Some("req"));
            assert_eq!(err.upstream_message(), Some(message));
            assert!(!err.is_retryable());
            let log = err.as_log().to_string();
            assert!(!log.contains("--embeddings"), "{log}");
        }

        // Any other 501 is a provider error that is not retried, and the
        // message alone, without the status, refuses nothing.
        let other = http(501, json!({"error": {"message": "Not Implemented"}}));
        assert!(matches!(other, Error::Provider { .. }), "{other:?}");
        assert!(!other.is_retryable());
        let in_band = map_api_error(&ApiError::new(message), None, None, None);
        assert!(matches!(in_band, Error::Provider { .. }), "{in_band:?}");
    }

    /// An in-band refusal (no status) maps like the HTTP one.
    #[test]
    fn in_band_refusals_map_like_http_ones() {
        let error = ApiError::new("Unsupported parameter: 'seed'.")
            .with_param("seed")
            .with_code("unsupported_parameter");
        let err = map_api_error(&error, None, None, None);
        assert_eq!(err.unsupported_field(), Some(RequestField::Seed));
        assert_eq!(
            err.unsupported_stage(),
            Some(allama::UnsupportedStage::Provider)
        );
    }

    /// A vendor's own quota code changes the classification, never the
    /// retry signals. (The dialects' mappings run the contract through the
    /// adapter aggregate.)
    #[test]
    fn retry_signals_are_surfaced_with_vendor_quota_codes() {
        let mut quirks = crate::CompatQuirks::new("openai-compat:vendor");
        quirks.quota_error_codes.push("quota_exceeded".into());
        allama_conformance::assert_retry_signals_surfaced(None, |status, headers, body| {
            quirks.map_http_error(status, &headers.iter().copied().collect(), &body)
        });
    }
}
