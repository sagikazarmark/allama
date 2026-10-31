//! Error mapping from `worker::Error` to `allama::Error`.
//!
//! The `worker::Ai` binding exposes no HTTP status. Classification uses, in
//! order:
//!
//! 1. the structured `worker::Error` variants (`RateLimitExceeded`,
//!    `DailyLimitExceeded`, `InternalError`);
//! 2. the documented Workers AI internal error code
//!    (<https://developers.cloudflare.com/workers-ai/platform/errors/>), read
//!    from the JS error's `code` property or parsed from the `"<code>: "`
//!    prefix the binding puts in messages (e.g.
//!    `"InferenceUpstreamError: 3040: Capacity temporarily exceeded"`). A
//!    code means what its documented HTTP status means on every adapter
//!    ([`allama_http::status_error`]), unless the code says more (a `429`
//!    that is capacity or a daily allocation, a `400` or `404` that names
//!    the model);
//! 3. message substrings ([`MESSAGE_RULES`]);
//! 4. as a last resort, an [`Error::UnexpectedResponse`]: an error the
//!    adapter does not recognise is not assumed to pass on a retry or on
//!    another provider (most unrecognised binding errors are about the
//!    input).
//!
//! The binding's message is upstream text: it becomes the error's
//! [upstream message](Error::upstream_message), and the constructor message
//! is the adapter's own (`Workers AI error`, with the code when known).

use allama::Error;

use crate::PROVIDER_NAME;

/// Convert a `worker::Error` to an `allama::Error`.
pub(crate) fn map_worker_error(err: worker::Error) -> Error {
    let message = err.to_string();
    let error = match &err {
        worker::Error::RateLimitExceeded(_) => {
            Error::rate_limited("Workers AI rate limit exceeded").authored()
        }
        worker::Error::DailyLimitExceeded(_) => {
            Error::quota_exhausted("Workers AI daily limit exceeded").authored()
        }
        worker::Error::InternalError(_) => allama_http::status_error(
            allama_http::http::StatusCode::INTERNAL_SERVER_ERROR,
            "Workers AI internal error",
        )
        .authored(),
        worker::Error::UnknownJsError { name, code, .. } => {
            classify(code.as_deref(), &message).with_provider_type_opt(name.as_deref())
        }
        _ => classify(None, &message),
    };
    error
        .with_upstream_message(message)
        .with_provider(PROVIDER_NAME)
}

/// The error for a Workers AI binding the Worker's `Env` does not provide:
/// a deployment misconfiguration, not a provider failure.
///
/// The runtime's text is foreign: it is the upstream message, never part
/// of the authored message.
pub(crate) fn missing_binding(binding: &str, reason: &str) -> Error {
    Error::configuration(format!(
        "Workers AI binding `{binding}` is not available; declare it under `[ai]` in wrangler.toml"
    ))
    .authored()
    .with_upstream_message(reason.to_owned())
    .with_provider(PROVIDER_NAME)
}

/// Classify an error from its optional provider `code` and its upstream
/// message, which becomes the error's upstream message.
pub(crate) fn classify(code: Option<&str>, message: &str) -> Error {
    let lower = message.to_ascii_lowercase();
    let error = if let Some(known) = code
        .and_then(|code| code.trim().parse::<u32>().ok())
        .and_then(KnownCode::from_code)
        .or_else(|| KnownCode::from_message(message))
    {
        known.to_error(&lower)
    } else {
        MESSAGE_RULES
            .iter()
            .find(|(_, needles)| needles.iter().any(|needle| lower.contains(needle)))
            .map_or_else(
                || Error::unexpected_response("unrecognised Workers AI error"),
                |(kind, _)| kind.to_error("Workers AI error".to_owned(), &lower),
            )
    };
    error.authored().with_upstream_message(message.to_owned())
}

/// A documented Workers AI internal error code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct KnownCode {
    code: u32,
    /// The HTTP status Workers AI documents for the code.
    status: u16,
    /// The category the code refines the status to; `None` when the code
    /// means what the status means.
    refined: Option<Kind>,
}

/// Workers AI error codes, their documented HTTP status, and the category
/// the code refines it to, when the status alone would say less.
const KNOWN_CODES: &[KnownCode] = &[
    // "No such model `${model}` or task"
    KnownCode::refined(5007, 400, Kind::ModelNotFound),
    // "Bad input": the input failed the model's schema ("oneOf at '/' not
    // met", recorded 2026-10-07)
    KnownCode::new(5006, 400),
    // "Invalid data type for base64 input"
    KnownCode::new(5004, 400),
    // "Finetune is missing required files"
    KnownCode::new(3039, 400),
    // "Request is missing headers or body"
    KnownCode::new(3003, 400),
    // "The account is not allowed to access this model"
    KnownCode::new(5018, 403),
    // "User has not agreed to Llama3.2 model terms"
    KnownCode::new(5016, 403),
    // "Service unavailable for account" (account blocked)
    KnownCode::new(3023, 403),
    // "The account is not allowed to access this model"
    KnownCode::new(3041, 403),
    // "This model requires a Workers Paid plan"
    KnownCode::new(5035, 403),
    // "Request trying to use deprecated SDK version"
    KnownCode::new(5019, 405),
    // "The model does not support LoRa inference"
    KnownCode::new(5005, 405),
    // "The model name is invalid"
    KnownCode::refined(3042, 404, Kind::ModelNotFound),
    // "Request is too large"
    KnownCode::new(3006, 413),
    // "Request timeout"
    KnownCode::new(3007, 408),
    // "Request was aborted"
    KnownCode::new(3008, 408),
    // "You have used up your daily free allocation of 10,000 neurons"
    KnownCode::refined(3036, 429, Kind::QuotaExhausted),
    // "Capacity temporarily exceeded, please try again"
    KnownCode::refined(3040, 429, Kind::Overloaded),
];

impl KnownCode {
    /// A code that means what its status means.
    const fn new(code: u32, status: u16) -> Self {
        Self {
            code,
            status,
            refined: None,
        }
    }

    /// A code that says more than its status.
    const fn refined(code: u32, status: u16, kind: Kind) -> Self {
        Self {
            code,
            status,
            refined: Some(kind),
        }
    }

    fn from_code(code: u32) -> Option<Self> {
        KNOWN_CODES.iter().copied().find(|known| known.code == code)
    }

    /// Find the first `<digits>:` group in `message` that is a known code, so
    /// unrelated numbers (request ids, token counts) are not misread.
    fn from_message(message: &str) -> Option<Self> {
        let bytes = message.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if !bytes[i].is_ascii_digit() || (i > 0 && bytes[i - 1].is_ascii_alphanumeric()) {
                i += 1;
                continue;
            }
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            let digits = &message[start..i];
            let mut j = i;
            while j < bytes.len() && bytes[j] == b' ' {
                j += 1;
            }
            if j < bytes.len()
                && bytes[j] == b':'
                && let Some(known) = digits.parse().ok().and_then(Self::from_code)
            {
                return Some(known);
            }
        }
        None
    }

    fn to_error(self, lower: &str) -> Error {
        let message = format!("Workers AI error {}", self.code);
        let error = match self.refined {
            Some(kind) => kind.to_error(message, lower),
            None => match allama_http::http::StatusCode::from_u16(self.status) {
                Ok(status) => allama_http::status_error(status, message),
                Err(_) => Error::provider(message),
            },
        };
        error
            .with_status(self.status)
            .with_provider_code(self.code.to_string())
    }
}

/// Error categories this adapter distinguishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Auth,
    QuotaExhausted,
    RateLimited,
    Overloaded,
    Timeout,
    ModelNotFound,
    ContextLength,
    ContentFiltered,
    InvalidRequest,
}

impl Kind {
    /// The error with the authored `message`; `lower` is the lowercased
    /// upstream message.
    fn to_error(self, message: String, lower: &str) -> Error {
        match self {
            Kind::Auth => Error::auth(message),
            Kind::QuotaExhausted => Error::quota_exhausted(message),
            Kind::RateLimited => Error::rate_limited(message),
            Kind::Overloaded => Error::overloaded(message),
            Kind::Timeout => Error::timeout(message),
            Kind::ModelNotFound => Error::model_not_found(message),
            Kind::ContextLength => Error::context_length_exceeded(message, context_limit(lower)),
            Kind::ContentFiltered => Error::content_filtered(message),
            Kind::InvalidRequest => Error::invalid_request(message),
        }
    }
}

/// Substring fallbacks for messages without a known code, checked in order.
///
/// Order matters: quota before rate limits ("daily limit exceeded"), and
/// context length before generic "too many" phrasing.
const MESSAGE_RULES: &[(Kind, &[&str])] = &[
    (
        Kind::QuotaExhausted,
        &[
            "daily free allocation",
            "daily limit exceeded",
            "neurons",
            "quota exceeded",
            "quota exhausted",
            "insufficient quota",
            "billing",
        ],
    ),
    (
        Kind::ContextLength,
        &[
            "context length",
            "context window",
            "maximum context",
            "prompt is too long",
            "too many tokens",
            "input is too long",
        ],
    ),
    (
        Kind::RateLimited,
        &["rate limit", "too many requests", "ratelimit"],
    ),
    (
        Kind::Auth,
        &[
            "unauthorized",
            "unauthenticated",
            "authentication",
            "not authorized",
            "forbidden",
            "permission denied",
            "invalid api key",
            "not allowed to access",
        ],
    ),
    (
        Kind::Overloaded,
        &[
            "capacity",
            "overloaded",
            "service unavailable",
            "temporarily unavailable",
        ],
    ),
    (Kind::Timeout, &["timeout", "timed out"]),
    (
        Kind::InvalidRequest,
        &["bad input", "invalid input", "type mismatch", "not met"],
    ),
    (
        Kind::ModelNotFound,
        &[
            "no such model",
            "model not found",
            "unknown model",
            "model name is invalid",
        ],
    ),
    (
        Kind::ContentFiltered,
        &[
            "content filter",
            "content_filter",
            "unsafe content",
            "flagged",
            "safety system",
        ],
    ),
];

/// Best-effort extraction of the token limit from messages such as
/// "maximum context length is 32768 tokens".
fn context_limit(lower: &str) -> Option<u64> {
    let tail = &lower[lower.find("context")?..];
    let start = tail.find(|c: char| c.is_ascii_digit())?;
    let digits: String = tail[start..]
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == ',')
        .filter(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    fn rust_error(message: &str) -> worker::Error {
        worker::Error::RustError(message.to_string())
    }

    /// (message, expected kind, expected status, expected provider code)
    const TABLE: &[(&str, &str, Option<u16>, Option<&str>)] = &[
        // Documented codes in the binding's "<code>: <message>" form.
        (
            "InferenceUpstreamError: 3040: Capacity temporarily exceeded, please try again.",
            "overloaded",
            Some(429),
            Some("3040"),
        ),
        (
            "3036: You have used up your daily free allocation of 10,000 neurons, please upgrade to Cloudflare's Workers Paid plan",
            "quota_exhausted",
            Some(429),
            Some("3036"),
        ),
        (
            "AiError: 5007: No such model @cf/meta/missing or task",
            "model_not_found",
            Some(400),
            Some("5007"),
        ),
        (
            "3042: The model name is invalid",
            "model_not_found",
            Some(404),
            Some("3042"),
        ),
        (
            "5018: The account is not allowed to access this model",
            "auth",
            Some(403),
            Some("5018"),
        ),
        (
            "5035: This model requires a Workers Paid plan",
            "auth",
            Some(403),
            Some("5035"),
        ),
        ("3007: Request timeout", "timeout", Some(408), Some("3007")),
        (
            "3008: Request was aborted",
            "timeout",
            Some(408),
            Some("3008"),
        ),
        (
            "3006: Request is too large",
            "invalid_request",
            Some(413),
            Some("3006"),
        ),
        (
            "5004: Invalid data type for base64 input: number",
            "invalid_request",
            Some(400),
            Some("5004"),
        ),
        // An unrelated leading number is not misread as a code.
        (
            "request 1234: 3040: Capacity temporarily exceeded",
            "overloaded",
            Some(429),
            Some("3040"),
        ),
        // Substring fallbacks.
        ("Unauthorized request", "auth", None, None),
        ("Rate limit exceeded", "rate_limited", None, None),
        (
            "Daily limit exceeded for account",
            "quota_exhausted",
            None,
            None,
        ),
        ("Worker request timed out", "timeout", None, None),
        ("Service unavailable", "overloaded", None, None),
        (
            "Model not found: @cf/meta/missing",
            "model_not_found",
            None,
            None,
        ),
        (
            "This model's maximum context length is 32768 tokens",
            "context_length_exceeded",
            None,
            None,
        ),
        (
            "Safety system blocked this response",
            "content_filtered",
            None,
            None,
        ),
        // Schema failures, recorded from Workers AI on 2026-10-07.
        (
            "AiError: Bad input: Error: oneOf at '/' not met, 0 matches: required properties at '/' are 'prompt', Type mismatch of '/messages', 'array' not in 'string', required properties at '/' are 'requests' (0c909b91-7bc0-40b7-a66d-e95bce998ebe)",
            "invalid_request",
            None,
            None,
        ),
        (
            "AiError: 5006: Bad input: Error: oneOf at '/' not met, 2 matches: Type mismatch of '/messages/0/content', 'array' not in 'string'",
            "invalid_request",
            Some(400),
            Some("5006"),
        ),
        ("Invalid input", "invalid_request", None, None),
        // An error the adapter does not recognise is not assumed
        // to pass on a retry.
        (
            "unexpected worker failure",
            "unexpected_response",
            None,
            None,
        ),
        // "author" must not be read as an auth failure.
        (
            "Write as the author would",
            "unexpected_response",
            None,
            None,
        ),
    ];

    #[test]
    fn error_mapping_table() {
        for (message, kind, status, code) in TABLE {
            let err = map_worker_error(rust_error(message));
            assert_eq!(err.kind_name(), *kind, "kind for {message:?}");
            assert_eq!(err.status(), *status, "status for {message:?}");
            let context = err.context().expect("provider errors carry context");
            assert_eq!(
                context.provider_code.as_deref(),
                *code,
                "code for {message:?}"
            );
            assert_eq!(context.provider.as_deref(), Some("cloudflare"));
            assert_eq!(err.upstream_message(), Some(*message));
            assert!(
                !err.as_log().to_string().contains(message),
                "{message:?} reached the log"
            );
        }
    }

    #[test]
    fn structured_worker_variants_take_precedence() {
        let err = map_worker_error(worker::Error::RateLimitExceeded("slow down".into()));
        assert!(matches!(err, Error::RateLimited { .. }));
        assert!(err.is_retryable());

        let err = map_worker_error(worker::Error::DailyLimitExceeded(
            "come back tomorrow".into(),
        ));
        assert!(matches!(err, Error::QuotaExhausted { .. }));
        assert!(!err.is_retryable());

        let err = map_worker_error(worker::Error::InternalError("boom".into()));
        assert!(matches!(err, Error::Provider { .. }));
        assert_eq!(
            err.as_log().to_string(),
            r#"kind=provider provider=cloudflare status=500 message="Workers AI internal error""#
        );
        assert_eq!(err.status(), Some(500));
        assert!(err.is_retryable());
    }

    #[test]
    fn js_string_errors_are_classified() {
        let err = map_worker_error(worker::Error::JsError("3040: Capacity exceeded".into()));
        assert!(matches!(err, Error::Overloaded { .. }));
    }

    #[test]
    fn explicit_code_wins_over_message() {
        let err = classify(Some("3036"), "something went wrong");
        assert!(matches!(err, Error::QuotaExhausted { .. }));
        assert_eq!(err.status(), Some(429));

        // Unknown codes fall through to the message.
        let err = classify(Some("E_SOMETHING"), "Rate limit exceeded");
        assert!(matches!(err, Error::RateLimited { .. }));
    }

    #[test]
    fn unrecognised_and_input_errors_are_not_retryable() {
        assert!(!classify(None, "something odd").is_retryable());
        assert!(!classify(Some("5006"), "Bad input").is_retryable());
    }

    #[test]
    fn quota_is_not_retryable_but_capacity_is() {
        assert!(!classify(None, "3036: daily free allocation").is_retryable());
        assert!(classify(None, "3040: Capacity temporarily exceeded").is_retryable());
    }

    #[test]
    fn context_length_extracts_limit() {
        let err = classify(None, "This model's maximum context length is 32,768 tokens");
        assert!(matches!(
            err,
            Error::ContextLengthExceeded {
                context_window: Some(32768),
                ..
            }
        ));
        let err = classify(None, "prompt is too long");
        assert!(matches!(
            err,
            Error::ContextLengthExceeded {
                context_window: None,
                ..
            }
        ));
    }

    #[test]
    fn missing_binding_is_a_configuration_error() {
        let err = missing_binding("AI", "Binding `AI` is undefined.");
        assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
        assert!(!err.is_retryable());
        assert!(err.message().contains("`AI`"), "{}", err.message());
        assert!(err.as_log().to_string().contains("`AI`"));
        // The runtime's text stays out of the redacted log.
        assert_eq!(err.upstream_message(), Some("Binding `AI` is undefined."));
        assert!(!err.as_log().to_string().contains("undefined"));
        assert!(err.to_string().contains("undefined"));
        assert_eq!(
            err.context().unwrap().provider.as_deref(),
            Some(PROVIDER_NAME)
        );
    }
}
