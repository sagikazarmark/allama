//! Error redaction (C7): upstream text never reaches the redacted log.
//!
//! [`assert_error_log_excludes_body`] feeds a provider's error mapping a
//! sentinel string inside response bodies and checks that it never appears
//! in [`Error::as_log`] (nor, for the error envelope's type, code and
//! `param`, in `Debug`). The probe is transport-agnostic: it takes the
//! mapping as a plain function of status, headers and body, so expose your
//! error mapping as a pure function (the provider guide recommends it) and
//! pass it in. It also takes the adapter's own error envelope for each
//! [`ErrorKindProbe`], the builder the error kind contracts use, and sends
//! the sentinel through every kind, so every classification branch the
//! adapter's envelope reaches is checked.
//!
//! ```
//! use allama::Error;
//! use allama_conformance::assert_error_log_excludes_body;
//! use allama_conformance::error_kinds::ErrorKindProbe;
//! use serde_json::json;
//!
//! // A mapping that follows the rules: authored message, upstream text in
//! // its own slot, the body only in `body`.
//! fn map_http_error(status: u16, body: &str) -> Error {
//!     let json = serde_json::from_str::<serde_json::Value>(body).ok();
//!     let upstream = json.as_ref().and_then(|v| v["error"]["message"].as_str());
//!     let error = match status {
//!         400..=499 => Error::invalid_request(format!("HTTP {status}")),
//!         _ => Error::provider(format!("HTTP {status}")),
//!     };
//!     error
//!         .authored()
//!         .with_status(status)
//!         .with_upstream_message_opt(upstream)
//!         .with_body(body.to_owned())
//! }
//!
//! // The adapter's envelope for each kind it maps, with the upstream text
//! // appended to its wording.
//! fn envelope(kind: ErrorKindProbe, upstream: &str) -> Option<String> {
//!     let message = match kind {
//!         ErrorKindProbe::RateLimited => "Slow down.",
//!         ErrorKindProbe::InvalidRequest => "Invalid value for 'temperature'.",
//!         // Kinds the adapter does not map, including later ones.
//!         _ => return None,
//!     };
//!     let message = format!("{message} {upstream}");
//!     Some(json!({"error": {"message": message.trim_end()}}).to_string())
//! }
//!
//! assert_error_log_excludes_body(envelope, |status, _headers, body| {
//!     map_http_error(status, &body)
//! });
//! ```

use allama::Error;

use crate::stream_errors::ErrorKindProbe;

/// The string fed through the mapping. It must never reach the redacted log.
pub const SENTINEL: &str = "allama-sentinel-7Hq2 my SSN is 123-45-6789";

/// The string fed as an error envelope's type, code and `param` (case
/// (d)). Its spaces and punctuation are what no code has, so the redacted
/// log's token filter must reject it; it must never reach the redacted log
/// or `Debug`.
pub const FIELD_SENTINEL: &str = "allama-sentinel-3Fw9 <b>echoed input</b>, isn't it?";

/// Assert that `map_http_error` keeps upstream text out of the redacted
/// [`as_log`](Error::as_log) view.
///
/// `envelope` is the adapter's error envelope for each [`ErrorKindProbe`],
/// as its API returns it, or `None` for a kind the adapter does not map:
/// the builder the error kind contracts take, given the upstream text as
/// its second argument. It appends that text to its realistic wording
/// where the adapter's classification reads the message. It must map at
/// least one kind.
///
/// `map_http_error` receives an HTTP status, the response headers as
/// `(name, value)` pairs, and the body, and returns the provider's error. It
/// is called with [`SENTINEL`]:
///
/// - **(a)** in bodies that are no error envelope (HTML, plain text, other
///   JSON). The sentinel must not appear in `as_log()`, nor in
///   `as_log().with_message(..)`, which never includes text from an
///   unrecognised body;
/// - **(b)** as the message of the common error envelopes (OpenAI, legacy
///   vLLM, Anthropic, Gemini, Ollama, FastAPI);
/// - **(c)** inside messages that steer classification (unsupported input,
///   context overflow, unknown model, quota, content filtering), in the
///   common envelopes, so a branch that maps an upstream 400 to
///   [`Error::Unsupported`] or [`Error::InvalidRequest`] and wrongly marks
///   the upstream text [authored](Error::authored) is caught. At least one
///   of these must map to `Unsupported` or `InvalidRequest`;
/// - **(d)** as the error type, code and `param` of the common envelopes
///   that have them (OpenAI, legacy vLLM, Anthropic, Gemini's status,
///   reason and field violations, FastAPI's error type), with a plain
///   message. [`FIELD_SENTINEL`] must not appear in `as_log()`, its JSON,
///   or the error's `Debug`: an adapter keeps these fields in
///   [`ErrorContext`](allama::ErrorContext)'s `provider_type`,
///   `provider_code` and `param`, which both filter, and out of its
///   messages, which `Debug` shows even when they are not authored;
/// - **(e)** as the upstream text of the adapter's own envelope, for every
///   kind `envelope` maps: `envelope(kind, SENTINEL)` must carry the
///   sentinel, its mapping at the kind's [canonical
///   status](ErrorKindProbe::status) must be the same variant as the
///   mapping of `envelope(kind, "")`, and the sentinel must not appear in
///   `as_log()`. So every branch the adapter's envelope reaches is
///   checked, including those that read only its own envelope, and
///   upstream text never changes how an error is classified.
///
/// # Panics
///
/// When a sentinel appears in the redacted view of any mapped error (or
/// [`FIELD_SENTINEL`] in its `Debug`), when no case (c) error is
/// `Unsupported` or `InvalidRequest`, when `envelope` maps no kind, or
/// when a case (e) envelope drops the upstream text or maps to another
/// variant with it.
pub fn assert_error_log_excludes_body<E, M>(envelope: E, mut map_http_error: M)
where
    E: Fn(ErrorKindProbe, &str) -> Option<String>,
    M: FnMut(u16, &[(&str, &str)], String) -> Error,
{
    let headers: &[(&str, &str)] = &[
        ("content-type", "application/json"),
        ("x-request-id", "req_redaction_probe"),
        ("request-id", "req_redaction_probe"),
    ];
    let mut check = |case: &str, status: u16, body: String, opt_in_allowed: bool| {
        let error = map_http_error(status, headers, body.clone());
        assert_redacted(&error, case, status, &body, opt_in_allowed);
        error
    };

    for status in [400, 401, 404, 429, 500, 502, 503] {
        for body in unrecognised_bodies() {
            check("(a) unrecognised body", status, body, false);
        }
        for body in common_envelopes(SENTINEL) {
            check("(b) envelope message", status, body, true);
        }
    }

    let mut refused = false;
    for status in [400, 404, 413, 422] {
        for message in steering_messages() {
            for body in common_envelopes(&message) {
                let error = check("(c) classified message", status, body, true);
                refused |= matches!(
                    error,
                    Error::Unsupported { .. } | Error::InvalidRequest { .. }
                );
            }
        }
        for message in steering_messages() {
            check("(c) classified unrecognised body", status, message, false);
        }
    }
    assert!(
        refused,
        "error redaction (c): no upstream 4xx mapped to Unsupported or InvalidRequest"
    );

    let mut mapped = false;
    for &kind in ErrorKindProbe::ALL {
        let Some(plain) = envelope(kind, "") else {
            continue;
        };
        mapped = true;
        let status = kind.status();
        let Some(body) = envelope(kind, SENTINEL) else {
            panic!(
                "error redaction (e) adapter envelope: {kind} maps without upstream text \
                 but not with it; return the same envelope with the text appended"
            );
        };
        assert!(
            body.contains(SENTINEL),
            "error redaction (e) adapter envelope: the {kind} envelope drops the upstream \
             text; append it to the wording your classification reads: {body}"
        );
        let expected = check("(e) adapter envelope", status, plain, true);
        let error = check("(e) adapter envelope", status, body.clone(), true);
        assert!(
            error.kind_name() == expected.kind_name(),
            "error redaction (e) adapter envelope: with upstream text, the {kind} envelope \
             maps to {} at HTTP {status}, not {} as without it: {body}",
            error.kind_name(),
            expected.kind_name(),
        );
    }
    assert!(
        mapped,
        "error redaction (e): the envelope builder maps no kind, so the adapter's own \
         envelope is never checked"
    );

    for status in [400, 404, 429, 500] {
        for body in field_envelopes(FIELD_SENTINEL) {
            let error = map_http_error(status, headers, body.clone());
            assert_fields_redacted(&error, status, &body);
        }
    }
}

fn assert_fields_redacted(error: &Error, status: u16, body: &str) {
    let log = error.as_log();
    let line = log.to_string();
    let json = serde_json::to_string(&log).expect("serialize error log view");
    let debug = format!("{error:?}");
    for (view, text) in [("Display", &line), ("JSON", &json), ("Debug", &debug)] {
        assert!(
            !text.contains(FIELD_SENTINEL),
            "error redaction (d) envelope type, code and param: upstream text reached \
             {view} for HTTP {status} body {body:?}: {text}"
        );
    }
}

fn assert_redacted(error: &Error, case: &str, status: u16, body: &str, opt_in_allowed: bool) {
    let log = error.as_log();
    let line = log.to_string();
    let json = serde_json::to_string(&log).expect("serialize error log view");
    for (view, text) in [("Display", &line), ("JSON", &json)] {
        assert!(
            !text.contains(SENTINEL),
            "error redaction {case}: upstream text reached the redacted log ({view}) \
             for HTTP {status} body {body:?}: {text}"
        );
    }
    if !opt_in_allowed {
        let opted_in = log.with_message(usize::MAX).to_string();
        assert!(
            !opted_in.contains(SENTINEL),
            "error redaction {case}: text from an unrecognised body reached \
             `as_log().with_message(..)` for HTTP {status} body {body:?}: {opted_in}"
        );
    }
}

fn unrecognised_bodies() -> Vec<String> {
    vec![
        format!("<html><body><h1>502 Bad Gateway</h1><p>{SENTINEL}</p></body></html>"),
        SENTINEL.to_owned(),
        serde_json::json!({ "unexpected": SENTINEL }).to_string(),
    ]
}

/// `message` in each common error envelope.
fn common_envelopes(message: &str) -> Vec<String> {
    use serde_json::json;

    [
        json!({"error": {"message": message, "type": "invalid_request_error", "code": "invalid_value", "param": "messages.[0].content"}}),
        json!({"type": "error", "error": {"type": "invalid_request_error", "message": message}}),
        json!({"error": {"code": 400, "message": message, "status": "INVALID_ARGUMENT"}}),
        json!({"object": "error", "message": message, "type": "BadRequestError", "code": 400}),
        json!({"error": message}),
        json!({"detail": message}),
        json!({"detail": {"message": message, "error_type": "api_usage_error"}}),
    ]
    .iter()
    .map(ToString::to_string)
    .collect()
}

/// `value` as the type, code and `param` of each common envelope that has
/// them, with a plain message.
fn field_envelopes(value: &str) -> Vec<String> {
    use serde_json::json;

    let message = "the request was rejected";
    [
        json!({"error": {"message": message, "type": value, "code": value, "param": value}}),
        json!({"type": "error", "error": {"type": value, "message": message}}),
        json!({"error": {"code": 400, "message": message, "status": value, "details": [
            {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": value},
            {"@type": "type.googleapis.com/google.rpc.BadRequest",
             "fieldViolations": [{"field": value, "description": message}]},
        ]}}),
        json!({"object": "error", "message": message, "type": value, "code": value}),
        json!({"detail": {"message": message, "error_type": value}}),
    ]
    .iter()
    .map(ToString::to_string)
    .collect()
}

/// Messages that steer common classification heuristics, each carrying the
/// sentinel.
fn steering_messages() -> Vec<String> {
    [
        "model does not support images",
        "not supported by system one",
        "This model's maximum context length is 4096 tokens",
        "prompt is too long: 300000 tokens > 200000 maximum",
        "input token count exceeds the maximum number of tokens allowed (1048576)",
        "unknown model",
        "model not found",
        "Your credit balance is too low",
        "blocked by content filtering",
        "Invalid value",
    ]
    .iter()
    .map(|phrase| format!("{phrase}: {SENTINEL}"))
    .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// A minimal envelope reader (the conformance crate does not depend on
    /// `allama-http`).
    fn upstream(body: &str) -> Option<String> {
        let value: serde_json::Value = serde_json::from_str(body).ok()?;
        value["error"]["message"]
            .as_str()
            .or_else(|| value["error"].as_str())
            .or_else(|| value["message"].as_str())
            .or_else(|| value["detail"].as_str())
            .or_else(|| value["detail"]["message"].as_str())
            .map(str::to_owned)
    }

    /// An OpenAI-shaped envelope for a few kinds, with the upstream text
    /// appended to its wording.
    fn envelope(kind: ErrorKindProbe, upstream: &str) -> Option<String> {
        let (error_type, message) = match kind {
            ErrorKindProbe::RateLimited => ("rate_limit_error", "Slow down."),
            ErrorKindProbe::InvalidRequest => ("invalid_request_error", "Invalid value."),
            ErrorKindProbe::Unsupported => ("invalid_request_error", "Images: does not support."),
            _ => return None,
        };
        let message = format!("{message} {upstream}");
        Some(json!({"error": {"type": error_type, "message": message.trim_end()}}).to_string())
    }

    fn good(status: u16, body: &str) -> Error {
        let error = if status == 400 && body.contains("does not support") {
            Error::unsupported(format!("HTTP {status}"))
        } else if (400..500).contains(&status) {
            Error::invalid_request(format!("HTTP {status}"))
        } else {
            Error::provider(format!("HTTP {status}"))
        };
        error
            .authored()
            .with_status(status)
            .with_upstream_message_opt(upstream(body).as_deref())
            .with_body(body.to_owned())
    }

    #[test]
    fn a_redacting_mapping_passes() {
        assert_error_log_excludes_body(envelope, |status, _, body| good(status, &body));
    }

    #[test]
    #[should_panic(expected = "(a) unrecognised body")]
    fn authored_bodies_fail() {
        assert_error_log_excludes_body(envelope, |status, _, body| {
            Error::provider(body).authored().with_status(status)
        });
    }

    #[test]
    #[should_panic(expected = "(b) envelope message")]
    fn authored_envelope_messages_fail() {
        assert_error_log_excludes_body(envelope, |status, _, body| match upstream(&body) {
            Some(message) => Error::provider(message).authored().with_status(status),
            None => good(status, &body),
        });
    }

    #[test]
    #[should_panic(expected = "(c) classified message")]
    fn authored_classified_messages_fail() {
        assert_error_log_excludes_body(envelope, |status, _, body| match upstream(&body) {
            Some(message) if message.contains("context length") => {
                Error::context_length_exceeded(message, None).authored()
            }
            _ => good(status, &body),
        });
    }

    #[test]
    #[should_panic(expected = "with_message")]
    fn unrecognised_bodies_as_upstream_text_fail() {
        assert_error_log_excludes_body(envelope, |status, _, body| {
            good(status, &body).with_upstream_message(body)
        });
    }

    /// Nimbus's own error envelope, told apart by its code, with the
    /// upstream text appended to its reason.
    fn nimbus_envelope(kind: ErrorKindProbe, upstream: &str) -> Option<String> {
        let (code, reason) = match kind {
            ErrorKindProbe::RateLimited => ("throttled", "Too many requests."),
            ErrorKindProbe::Overloaded => ("busy", "Try again later."),
            ErrorKindProbe::InvalidRequest => ("bad_input", "Bad input."),
            _ => return None,
        };
        let reason = format!("{reason} {upstream}");
        Some(json!({"fault": {"code": code, "reason": reason.trim_end()}}).to_string())
    }

    /// Nimbus's code and reason, as a minimal reader finds them.
    fn fault(body: &str) -> Option<(String, String)> {
        let value: serde_json::Value = serde_json::from_str(body).ok()?;
        let code = value["fault"]["code"].as_str()?.to_owned();
        let reason = value["fault"]["reason"].as_str()?.to_owned();
        Some((code, reason))
    }

    /// Nimbus's mapping: the variant from the code, the reason in its slot.
    fn nimbus(status: u16, code: &str, reason: String, body: String) -> Error {
        let error = match code {
            "throttled" => Error::rate_limited(format!("HTTP {status}")),
            "busy" => Error::overloaded(format!("HTTP {status}")),
            _ => Error::invalid_request(format!("HTTP {status}")),
        };
        error
            .authored()
            .with_status(status)
            .with_upstream_message(reason)
            .with_body(body)
    }

    #[test]
    fn a_redacting_mapping_of_a_custom_envelope_passes() {
        assert_error_log_excludes_body(nimbus_envelope, |status, _, body| match fault(&body) {
            Some((code, reason)) => nimbus(status, &code, reason, body),
            None => good(status, &body),
        });
    }

    #[test]
    #[should_panic(expected = "(e) adapter envelope: upstream text reached")]
    fn a_custom_envelope_leak_is_caught() {
        // No common envelope reaches the branch that reads Nimbus's reason.
        assert_error_log_excludes_body(nimbus_envelope, |status, _, body| match fault(&body) {
            Some((_, reason)) => Error::invalid_request(reason)
                .authored()
                .with_status(status),
            None => good(status, &body),
        });
    }

    #[test]
    #[should_panic(expected = "(e) adapter envelope: upstream text reached")]
    fn a_leak_on_one_kind_only_fails() {
        // The branch for `busy` alone authors the reason.
        assert_error_log_excludes_body(nimbus_envelope, |status, _, body| match fault(&body) {
            Some((code, reason)) if code == "busy" => {
                Error::overloaded(reason).authored().with_status(status)
            }
            Some((code, reason)) => nimbus(status, &code, reason, body),
            None => good(status, &body),
        });
    }

    #[test]
    #[should_panic(expected = "maps to provider at HTTP 400, not invalid_request as without it")]
    fn upstream_text_that_changes_the_variant_fails() {
        // A classifier that matches the whole reason.
        assert_error_log_excludes_body(nimbus_envelope, |status, _, body| match fault(&body) {
            Some((code, reason)) if code == "bad_input" && reason != "Bad input." => {
                Error::provider(format!("HTTP {status}"))
                    .authored()
                    .with_status(status)
            }
            Some((code, reason)) => nimbus(status, &code, reason, body),
            None => good(status, &body),
        });
    }

    #[test]
    #[should_panic(expected = "the RateLimited envelope drops the upstream text")]
    fn an_envelope_that_drops_the_upstream_text_fails() {
        assert_error_log_excludes_body(
            |kind, _| nimbus_envelope(kind, ""),
            |status, _, body| good(status, &body),
        );
    }

    #[test]
    #[should_panic(expected = "the envelope builder maps no kind")]
    fn an_envelope_that_maps_nothing_fails() {
        assert_error_log_excludes_body(|_, _| None, |status, _, body| good(status, &body));
    }

    /// The envelope's error type, as a minimal reader finds it.
    fn error_type(body: &str) -> Option<String> {
        let value: serde_json::Value = serde_json::from_str(body).ok()?;
        value["error"]["type"]
            .as_str()
            .or_else(|| value["type"].as_str().filter(|t| *t != "error"))
            .map(str::to_owned)
    }

    #[test]
    fn a_mapping_that_keeps_the_type_in_its_slot_passes() {
        assert_error_log_excludes_body(envelope, |status, _, body| {
            good(status, &body).with_provider_type_opt(error_type(&body).as_deref())
        });
    }

    #[test]
    #[should_panic(expected = "(d) envelope type, code and param")]
    fn an_authored_message_naming_the_type_fails() {
        assert_error_log_excludes_body(envelope, |status, _, body| match error_type(&body) {
            Some(kind) => Error::invalid_request(format!("HTTP {status}: {kind}"))
                .authored()
                .with_status(status),
            None => good(status, &body),
        });
    }

    /// `Debug` shows the constructor message, authored or not.
    #[test]
    #[should_panic(expected = "reached Debug")]
    fn a_message_naming_the_type_fails_unauthored() {
        assert_error_log_excludes_body(envelope, |status, _, body| match error_type(&body) {
            Some(kind) => {
                Error::invalid_request(format!("HTTP {status}: {kind}")).with_status(status)
            }
            None => good(status, &body),
        });
    }

    #[test]
    #[should_panic(expected = "no upstream 4xx mapped")]
    fn mappings_must_refuse_something() {
        assert_error_log_excludes_body(envelope, |status, _, body| {
            Error::provider(format!("HTTP {status}"))
                .authored()
                .with_body(body)
        });
    }
}
