//! In-band stream errors: an error reported inside a stream is classified
//! like the same error reported over HTTP.
//!
//! A streaming response can fail after it started: the server answers
//! `200` and then sends an error frame (Anthropic's `event: error`, an
//! OpenAI-style `{"error": {..}}` chunk, a Gemini error object). Such a frame
//! carries no HTTP status, so the adapter classifies it from the vendor code
//! alone. Retry and fallback policies act on the error kind, so a rate limit
//! must be [`Error::RateLimited`] whether it arrives as a `429` or mid-stream.
//! [`assert_stream_errors_classified`] checks that it is, for every
//! [`ErrorKindProbe`] the adapter maps.
//!
//! ```
//! use allama::{ChatStream, Error, StreamEvent};
//! use allama_conformance::stream_errors::{ErrorKindProbe, assert_stream_errors_classified};
//! use serde_json::json;
//!
//! /// The adapter's classification, shared by its HTTP and stream paths.
//! fn classify(code: &str) -> Error {
//!     match code {
//!         "throttled" => Error::rate_limited("rate limited"),
//!         _ => Error::invalid_request("rejected"),
//!     }
//! }
//!
//! fn code(body: &str) -> String {
//!     let value: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
//!     value["fault"]["code"].as_str().unwrap_or_default().to_owned()
//! }
//!
//! # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
//! assert_stream_errors_classified(
//!     // The adapter's envelope, with its own vendor code for each kind.
//!     |kind| {
//!         let code = match kind {
//!             ErrorKindProbe::RateLimited => "throttled",
//!             ErrorKindProbe::InvalidRequest => "bad_input",
//!             // Kinds the adapter does not map, including later ones.
//!             _ => return None,
//!         };
//!         Some(json!({"fault": {"code": code, "reason": "probe"}}).to_string())
//!     },
//!     // How the envelope arrives inside a stream.
//!     |envelope| format!("event: fault\ndata: {envelope}\n\n"),
//!     // The adapter's HTTP error mapping.
//!     |_status, _headers, body| classify(&code(&body)),
//!     // A stream decoded from the body, as the adapter's provider yields it.
//!     |body| async move {
//!         let data = body.lines().find_map(|line| line.strip_prefix("data: ")).unwrap();
//!         let items: Vec<allama::Result<StreamEvent>> = vec![Err(classify(&code(data)))];
//!         Ok::<ChatStream, Error>(Box::pin(futures_util::stream::iter(items)))
//!     },
//! )
//! .await;
//! # });
//! ```

use std::fmt;
use std::future::Future;

use allama::{Error, StreamEvent};
use futures_core::Stream;
use futures_util::StreamExt;

/// Assert that an error reported inside a stream has the kind the adapter's
/// HTTP error mapping gives the same error.
///
/// The adapter supplies:
///
/// - `envelope`: for each [`ErrorKindProbe`], its error envelope for that
///   kind, with its own vendor code (OpenAI's `rate_limit_exceeded` and
///   `insufficient_quota` are both `429`s, and an in-band frame has no
///   status at all, so the code is what tells kinds apart). It returns
///   `None` for a kind the adapter does not map, including kinds added to
///   [`ErrorKindProbe`] later;
/// - `stream_body`: wraps an envelope as the body of a streaming response
///   (an SSE `error` event, an NDJSON line, ...);
/// - `map_http_error`: the adapter's HTTP error mapping, as for
///   [`assert_error_log_excludes_body`](crate::assert_error_log_excludes_body);
/// - `probe`: sends a request whose transport answers `200` with the given
///   body, and returns the adapter's stream (`chat_stream`'s result).
///
/// For each kind `envelope` maps, the contract maps the envelope with
/// `map_http_error` at each of the kind's [statuses](ErrorKindProbe::statuses),
/// each of which must give that kind, and then decodes `stream_body(envelope)`
/// through `probe`, whose first error must have the same kind. An error from
/// `chat_stream` itself counts as the first error.
///
/// `envelope` must map at least [`ErrorKindProbe::RateLimited`] and
/// [`ErrorKindProbe::InvalidRequest`], so an adapter that maps nothing cannot
/// pass.
///
/// # Panics
///
/// When `envelope` skips a required kind, when an envelope does not map to
/// its kind over HTTP, or when a stream ends without an error or with an
/// error of another kind.
pub async fn assert_stream_errors_classified<E, B, M, F, Fut, S>(
    envelope: E,
    stream_body: B,
    mut map_http_error: M,
    mut probe: F,
) where
    E: Fn(ErrorKindProbe) -> Option<String>,
    B: Fn(&str) -> String,
    M: FnMut(u16, &[(&str, &str)], String) -> Error,
    F: FnMut(String) -> Fut,
    Fut: Future<Output = allama::Result<S>>,
    S: Stream<Item = allama::Result<StreamEvent>>,
{
    let headers: &[(&str, &str)] = &[("content-type", "application/json")];
    let mut problems = Vec::new();
    let mut skipped = Vec::new();
    for &kind in ErrorKindProbe::ALL {
        let Some(envelope) = envelope(kind) else {
            skipped.push(kind);
            continue;
        };
        let mut http = None;
        for &status in kind.statuses() {
            let mapped = map_http_error(status, headers, envelope.clone());
            if mapped.kind_name() != kind.kind_name() {
                problems.push(format!(
                    "{kind}: map_http_error maps the envelope at HTTP {status} to {}, not {}: \
                     {envelope}",
                    mapped.kind_name(),
                    kind.kind_name()
                ));
            } else if let Some(problem) = misstaged(kind, &mapped) {
                problems.push(format!("{kind}: at HTTP {status}, {problem}: {envelope}"));
            }
            http.get_or_insert((status, mapped));
        }
        let Some((status, http)) = http.filter(|(_, http)| http.kind_name() == kind.kind_name())
        else {
            continue;
        };
        let body = stream_body(&envelope);
        match first_error(probe(body.clone()).await).await {
            Some(err) if err.kind_name() == http.kind_name() => {
                if let Some(problem) = misstaged(kind, &err) {
                    problems.push(format!(
                        "{kind}: in the stream, {problem}\n--- stream ---\n{body}"
                    ));
                }
            }
            Some(err) => problems.push(format!(
                "{kind}: the in-band error is {}, but HTTP {status} maps the same envelope to \
                 {}: {err:?}\n--- stream ---\n{body}",
                err.kind_name(),
                http.kind_name()
            )),
            None => problems.push(format!(
                "{kind}: the stream ended without an error\n--- stream ---\n{body}"
            )),
        }
    }
    for required in [ErrorKindProbe::RateLimited, ErrorKindProbe::InvalidRequest] {
        if skipped.contains(&required) {
            problems.push(format!(
                "{required}: the envelope builder must map at least RateLimited and \
                 InvalidRequest, so the contract cannot pass vacuously"
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "stream_errors_classified failed:\n  {}",
        problems.join("\n  ")
    );
}

/// An error kind [`assert_stream_errors_classified`] probes.
///
/// New kinds may be added; an envelope builder returns `None` for kinds it
/// does not map, so match with a wildcard arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKindProbe {
    /// A rate limit: [`Error::RateLimited`], canonically HTTP `429`.
    RateLimited,
    /// A temporarily overloaded service: [`Error::Overloaded`], canonically
    /// HTTP `503`.
    Overloaded,
    /// A rejected request: [`Error::InvalidRequest`], canonically HTTP `400`.
    InvalidRequest,
    /// Missing or invalid credentials: [`Error::Auth`], canonically HTTP
    /// `401`.
    Auth,
    /// Exhausted quota or billing: [`Error::QuotaExhausted`], canonically
    /// HTTP `429` with the vendor's quota code in the envelope.
    QuotaExhausted,
    /// A prompt over the context window: [`Error::ContextLengthExceeded`],
    /// canonically HTTP `400`.
    ContextLength,
    /// A feature the provider says it does not support for the request:
    /// [`Error::Unsupported`] with stage
    /// [`Provider`](allama::UnsupportedStage::Provider) (built with
    /// [`Error::unsupported_by_provider`]), canonically HTTP `400` with the
    /// vendor's code or wording for it in the envelope.
    Unsupported,
    /// A model the provider does not know: [`Error::ModelNotFound`],
    /// canonically HTTP `404`.
    ModelNotFound,
    /// The provider gave up before the model answered: [`Error::Timeout`],
    /// canonically HTTP `504`, and checked at `408` too.
    Timeout,
}

impl ErrorKindProbe {
    /// Every kind, in the order the contract probes them.
    pub const ALL: &'static [Self] = &[
        Self::RateLimited,
        Self::Overloaded,
        Self::InvalidRequest,
        Self::Auth,
        Self::QuotaExhausted,
        Self::ContextLength,
        Self::Unsupported,
        Self::ModelNotFound,
        Self::Timeout,
    ];

    /// The HTTP status this kind is canonically reported with.
    #[must_use]
    pub const fn status(self) -> u16 {
        self.statuses()[0]
    }

    /// The HTTP statuses at which [`assert_stream_errors_classified`] maps
    /// this kind's envelope, the [canonical one](Self::status) first.
    ///
    /// [`Timeout`](Self::Timeout) is checked at `504` (a gateway gave up
    /// waiting for the model) and `408` (the server gave up waiting for the
    /// request); every other kind at its canonical status alone.
    #[must_use]
    pub const fn statuses(self) -> &'static [u16] {
        match self {
            Self::RateLimited | Self::QuotaExhausted => &[429],
            Self::Overloaded => &[503],
            Self::InvalidRequest | Self::ContextLength | Self::Unsupported => &[400],
            Self::Auth => &[401],
            Self::ModelNotFound => &[404],
            Self::Timeout => &[504, 408],
        }
    }

    /// The [`Error::kind_name`] of the error this kind maps to.
    #[must_use]
    pub const fn kind_name(self) -> &'static str {
        match self {
            Self::RateLimited => "rate_limited",
            Self::Overloaded => "overloaded",
            Self::InvalidRequest => "invalid_request",
            Self::Auth => "auth",
            Self::QuotaExhausted => "quota_exhausted",
            Self::ContextLength => "context_length_exceeded",
            Self::Unsupported => "unsupported",
            Self::ModelNotFound => "model_not_found",
            Self::Timeout => "timeout",
        }
    }
}

impl fmt::Display for ErrorKindProbe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// What is wrong with the stage of an [`Unsupported`](ErrorKindProbe::Unsupported)
/// error: the provider refused a request it received, so the stage must be
/// [`Provider`](allama::UnsupportedStage::Provider).
fn misstaged(kind: ErrorKindProbe, err: &Error) -> Option<String> {
    let stage = err.unsupported_stage()?;
    (kind == ErrorKindProbe::Unsupported && stage != allama::UnsupportedStage::Provider).then(
        || {
            format!(
                "the provider's refusal has stage {stage}, not provider (map it with \
                 Error::unsupported_by_provider)"
            )
        },
    )
}

/// The first error of a stream, or the error that prevented it.
async fn first_error<S>(stream: allama::Result<S>) -> Option<Error>
where
    S: Stream<Item = allama::Result<StreamEvent>>,
{
    let stream = match stream {
        Ok(stream) => stream,
        Err(err) => return Some(err),
    };
    let mut stream = std::pin::pin!(stream);
    while let Some(item) = stream.next().await {
        if let Err(err) = item {
            return Some(err);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use allama::ChatStream;
    use serde_json::{Value, json};

    use super::*;

    /// What the fake Nimbus adapter gets wrong.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Fault {
        None,
        /// Classifies in-band faults by status only: with none, everything
        /// is a provider error.
        StreamIgnoresCode,
        /// Ends the stream normally at a fault frame.
        StreamEndsEarly,
    }

    /// Nimbus's vendor code for each kind it maps. It has no overload code.
    fn nimbus_code(kind: ErrorKindProbe) -> Option<&'static str> {
        Some(match kind {
            ErrorKindProbe::RateLimited => "throttled",
            ErrorKindProbe::InvalidRequest => "bad_input",
            ErrorKindProbe::Auth => "bad_key",
            ErrorKindProbe::QuotaExhausted => "out_of_credit",
            ErrorKindProbe::ContextLength => "prompt_too_long",
            ErrorKindProbe::ModelNotFound => "no_such_model",
            ErrorKindProbe::Timeout => "deadline",
            _ => return None,
        })
    }

    fn nimbus_envelope(kind: ErrorKindProbe) -> Option<String> {
        let code = nimbus_code(kind)?;
        Some(json!({"fault": {"code": code, "reason": "probe"}}).to_string())
    }

    fn nimbus_stream_body(envelope: &str) -> String {
        format!("event: delta\ndata: {{\"text\":\"Hel\"}}\n\nevent: fault\ndata: {envelope}\n\n")
    }

    /// Nimbus's classification: the vendor code first, then the status.
    fn classify(status: Option<u16>, code: &str) -> Error {
        match (code, status) {
            ("throttled", _) => Error::rate_limited("rate limited"),
            ("out_of_credit", _) => Error::quota_exhausted("out of credit"),
            ("bad_key", _) | (_, Some(401)) => Error::auth("bad key"),
            ("prompt_too_long", _) => Error::context_length_exceeded("too long", None),
            ("no_such_model", _) => Error::model_not_found("no such model"),
            ("deadline", _) => Error::timeout("deadline"),
            ("bad_input", _) | (_, Some(400..=499)) => Error::invalid_request("rejected"),
            _ => Error::provider("fault"),
        }
    }

    fn fault_code(data: &str) -> Option<String> {
        let value: Value = serde_json::from_str(data).ok()?;
        value["fault"]["code"].as_str().map(str::to_owned)
    }

    fn map_http_error(status: u16, _headers: &[(&str, &str)], body: String) -> Error {
        classify(Some(status), &fault_code(&body).unwrap_or_default())
    }

    /// Decode a Nimbus stream: a text delta per `delta` frame, and an error
    /// per `fault` frame.
    fn nimbus_stream(body: &str, fault: Fault) -> ChatStream {
        let mut items: Vec<allama::Result<StreamEvent>> = Vec::new();
        for frame in body.split("\n\n").filter(|frame| !frame.is_empty()) {
            let data = frame
                .lines()
                .find_map(|line| line.strip_prefix("data: "))
                .unwrap_or_default();
            if frame.starts_with("event: fault") {
                match fault {
                    Fault::None => {
                        items.push(Err(classify(None, &fault_code(data).unwrap_or_default())))
                    }
                    Fault::StreamIgnoresCode => items.push(Err(classify(None, ""))),
                    Fault::StreamEndsEarly => break,
                }
            } else {
                items.push(Ok(StreamEvent::response_start(None, None)));
            }
        }
        Box::pin(futures_util::stream::iter(items))
    }

    fn probe(fault: Fault) -> impl FnMut(String) -> std::future::Ready<allama::Result<ChatStream>> {
        move |body| std::future::ready(Ok(nimbus_stream(&body, fault)))
    }

    #[tokio::test]
    async fn a_nimbus_fault_stream_is_classified() {
        assert_stream_errors_classified(
            nimbus_envelope,
            nimbus_stream_body,
            map_http_error,
            probe(Fault::None),
        )
        .await;
    }

    #[tokio::test]
    async fn an_error_at_stream_setup_counts() {
        assert_stream_errors_classified(
            nimbus_envelope,
            nimbus_stream_body,
            map_http_error,
            |body: String| {
                let err = classify(
                    None,
                    &fault_code(&body[body.rfind("data: ").unwrap() + 6..]).unwrap_or_default(),
                );
                std::future::ready(Err::<ChatStream, _>(err))
            },
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(
        expected = "RateLimited: the in-band error is provider, but HTTP 429 maps the same \
                    envelope to rate_limited"
    )]
    async fn a_stream_classified_by_status_fails() {
        assert_stream_errors_classified(
            nimbus_envelope,
            nimbus_stream_body,
            map_http_error,
            probe(Fault::StreamIgnoresCode),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "InvalidRequest: the stream ended without an error")]
    async fn a_stream_dropping_the_fault_fails() {
        assert_stream_errors_classified(
            nimbus_envelope,
            nimbus_stream_body,
            map_http_error,
            probe(Fault::StreamEndsEarly),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(
        expected = "QuotaExhausted: map_http_error maps the envelope at HTTP 429 to \
                               rate_limited, not quota_exhausted"
    )]
    async fn an_envelope_the_http_mapping_misclassifies_fails() {
        // The builder sends the rate-limit code for quota too.
        assert_stream_errors_classified(
            |kind| match kind {
                ErrorKindProbe::QuotaExhausted => nimbus_envelope(ErrorKindProbe::RateLimited),
                kind => nimbus_envelope(kind),
            },
            nimbus_stream_body,
            map_http_error,
            probe(Fault::None),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "the envelope builder must map at least RateLimited and \
                               InvalidRequest")]
    async fn a_builder_that_maps_nothing_fails() {
        assert_stream_errors_classified(
            |_| None,
            nimbus_stream_body,
            map_http_error,
            probe(Fault::StreamIgnoresCode),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "InvalidRequest: the envelope builder must map at least")]
    async fn a_builder_that_skips_invalid_request_fails() {
        assert_stream_errors_classified(
            |kind| match kind {
                ErrorKindProbe::InvalidRequest => None,
                kind => nimbus_envelope(kind),
            },
            nimbus_stream_body,
            map_http_error,
            probe(Fault::None),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "Timeout: map_http_error maps the envelope at HTTP 408 to \
                               invalid_request, not timeout")]
    async fn a_timeout_the_http_mapping_misclassifies_at_408_fails() {
        // Classifies by status first, so a 408 is a rejected request.
        assert_stream_errors_classified(
            nimbus_envelope,
            nimbus_stream_body,
            |status, headers, body| match status {
                408 => Error::invalid_request("rejected"),
                _ => map_http_error(status, headers, body),
            },
            probe(Fault::None),
        )
        .await;
    }

    #[test]
    fn canonical_statuses() {
        let statuses: Vec<_> = ErrorKindProbe::ALL.iter().map(|k| k.status()).collect();
        assert_eq!(statuses, [429, 503, 400, 401, 429, 400, 400, 404, 504]);
        assert_eq!(ErrorKindProbe::Timeout.statuses(), [504, 408]);
    }
}
