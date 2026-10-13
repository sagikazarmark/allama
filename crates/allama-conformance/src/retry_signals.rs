//! The retry signals contract: [`assert_retry_signals_surfaced`].

use std::time::Duration;

use allama::Error;

/// The retry hint the contract sends, in seconds.
const RETRY_AFTER_SECS: u64 = 7;

/// Assert that `map_http_error` surfaces the server's retry signals on the
/// errors it returns:
/// [`Error::server_retry_directive`] from `x-should-retry` and
/// [`Error::retry_after`] from `Retry-After`.
///
/// A server tells the client whether and when to retry in headers: the retry
/// hint (`Retry-After`, `retry-after-ms`) and the retry directive
/// (`x-should-retry: true|false`). Some APIs also carry the hint in the
/// error body (Google's `RetryInfo` detail, for example). The default
/// [`RetryPolicy`](allama::wrappers::RetryPolicy) acts on both, but only if
/// the adapter's error mapping puts them on the error.
///
/// `map_http_error` receives an HTTP status, the response headers as
/// `(name, value)` pairs, and the body, as for
/// [`assert_error_log_excludes_body`](crate::assert_error_log_excludes_body).
/// It is called for a conflict, a rate limit, and server errors, with an
/// empty body, a JSON error envelope, and an HTML page, and with:
///
/// - `x-should-retry: true`: the directive must be `Some(true)`;
/// - `x-should-retry: false` and `Retry-After: 7`: the directive must be
///   `Some(false)` and the retry hint 7 s;
/// - no directive, or a value other than `true` or `false`: the directive
///   must be `None`.
///
/// An HTTP provider passes by building its error context with
/// `allama_http::error_context(status, &headers)`. A mapping that reads the
/// headers itself must match it:
///
/// ```
/// use std::time::Duration;
///
/// use allama::Error;
/// use allama_conformance::assert_retry_signals_surfaced;
///
/// fn map_http_error(status: u16, headers: &[(&str, &str)]) -> Error {
///     let header = |name: &str| {
///         headers
///             .iter()
///             .find(|(key, _)| key.eq_ignore_ascii_case(name))
///             .map(|(_, value)| *value)
///     };
///     Error::provider(format!("HTTP {status}"))
///         .authored()
///         .with_status(status)
///         .with_retry_after_opt(header("retry-after").and_then(|s| s.parse().ok()).map(Duration::from_secs))
///         .with_server_retry_directive_opt(match header("x-should-retry") {
///             Some("true") => Some(true),
///             Some("false") => Some(false),
///             _ => None,
///         })
/// }
///
/// // No retry hint in the body: pass `Some(&encoder)` when your API has one.
/// assert_retry_signals_surfaced(None, |status, headers, _body| map_http_error(status, headers));
/// ```
///
/// `body_hint` writes an error body that carries a retry hint, as the
/// adapter's API sends it (Google's `RetryInfo` detail with
/// `"retryDelay": "7s"`, for example), for an API that carries the hint in
/// the body. When it is given, its body for 7 s is mapped at `429` and
/// `503` with no retry headers, and the retry hint must be 7 s. Pass `None`
/// when the API carries hints in headers only.
///
/// # Panics
///
/// When a mapped error misses a signal, or reports a directive the server
/// did not send.
pub fn assert_retry_signals_surfaced(
    body_hint: Option<&dyn Fn(Duration) -> String>,
    mut map_http_error: impl FnMut(u16, &[(&str, &str)], String) -> Error,
) {
    const HINT: Duration = Duration::from_secs(RETRY_AFTER_SECS);
    let retry_after = RETRY_AFTER_SECS.to_string();
    let cases = [
        Case {
            name: "x-should-retry: true",
            headers: vec![("x-should-retry", "true")],
            directive: Some(true),
            hint: None,
        },
        Case {
            name: "x-should-retry: false with Retry-After",
            headers: vec![("x-should-retry", "false"), ("retry-after", &retry_after)],
            directive: Some(false),
            hint: Some(HINT),
        },
        Case {
            name: "no directive",
            headers: vec![],
            directive: None,
            hint: None,
        },
        Case {
            name: "x-should-retry: maybe",
            headers: vec![("x-should-retry", "maybe")],
            directive: None,
            hint: None,
        },
    ];

    for status in [409, 429, 500, 503] {
        for body in bodies() {
            for case in &cases {
                let name = case.name;
                let mut headers = vec![("content-type", "application/json")];
                headers.extend_from_slice(&case.headers);
                let error = map_http_error(status, &headers, body.clone());
                assert_eq!(
                    error.server_retry_directive(),
                    case.directive,
                    "retry signals ({name}): wrong server retry directive for HTTP {status} \
                     body {body:?}: {error:?}"
                );
                if let Some(hint) = case.hint {
                    assert_eq!(
                        error.retry_after(),
                        Some(hint),
                        "retry signals ({name}): the retry hint was not surfaced for HTTP \
                         {status} body {body:?}: {error:?}"
                    );
                }
            }
        }
    }

    let Some(body_hint) = body_hint else {
        return;
    };
    let body = body_hint(HINT);
    for status in [429, 503] {
        let error = map_http_error(
            status,
            &[("content-type", "application/json")],
            body.clone(),
        );
        assert_eq!(
            error.retry_after(),
            Some(HINT),
            "retry signals (hint in the body): the retry hint was not surfaced for HTTP \
             {status} body {body:?}: {error:?}"
        );
    }
}

/// One set of response headers and what the mapped error must report.
struct Case<'a> {
    name: &'static str,
    headers: Vec<(&'static str, &'a str)>,
    directive: Option<bool>,
    hint: Option<Duration>,
}

fn bodies() -> [String; 3] {
    [
        String::new(),
        r#"{"error": {"message": "request failed", "type": "error"}}"#.to_owned(),
        "<html><body>Service unavailable</body></html>".to_owned(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reads headers through `allama_http`, and a `retryDelay` from the body
    /// when `body_hints` is set.
    fn map(body_hints: bool) -> impl FnMut(u16, &[(&str, &str)], String) -> Error {
        move |status, headers, body| {
            let headers: allama_http::Headers = headers.iter().copied().collect();
            let mut error = Error::provider(format!("HTTP {status}"))
                .authored()
                .with_context(allama_http::error_context(status, &headers));
            if body_hints
                && error.retry_after().is_none()
                && let Some(seconds) = body
                    .split("\"retryDelay\": \"")
                    .nth(1)
                    .and_then(|rest| rest.split('s').next())
                    .and_then(|seconds| seconds.parse().ok())
            {
                error = error.with_retry_after(Duration::from_secs(seconds));
            }
            error
        }
    }

    fn retry_info(hint: Duration) -> String {
        format!(
            r#"{{"error": {{"details": [{{"retryDelay": "{}s"}}]}}}}"#,
            hint.as_secs()
        )
    }

    #[test]
    fn header_signals_pass() {
        assert_retry_signals_surfaced(None, map(false));
    }

    #[test]
    fn a_hint_in_the_body_passes_when_read() {
        assert_retry_signals_surfaced(Some(&retry_info), map(true));
    }

    #[test]
    #[should_panic(expected = "retry signals (hint in the body)")]
    fn a_hint_in_the_body_must_be_read() {
        assert_retry_signals_surfaced(Some(&retry_info), map(false));
    }
}
