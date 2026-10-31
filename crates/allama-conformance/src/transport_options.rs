//! The transport options contract: [`assert_transport_options_applied`].

use std::future::Future;

use allama::{Error, RequestOptions};
use allama_http::http::{HeaderName, HeaderValue};
use allama_http::{
    Credential, CredentialPlacement, CredentialSlot, HttpRequest, HttpRequestOptions,
    OPTIONS_NAMESPACE,
};
use serde_json::json;

/// A header the contract adds.
const ADDED: (&str, &str) = ("x-allama-conformance", "applied");

/// A header the contract replaces: every JSON adapter sets `content-type`.
const REPLACED: (&str, &str) = ("Content-Type", "application/json; charset=utf-8");

/// The adapter's own per-request key.
const OWN_SECRET: &str = "allama-conformance-own-SECRET";

/// A header credential addressed to the adapter's scope.
const OWN_HEADER: (&str, &str) = (
    "x-allama-conformance-credential",
    "allama-conformance-header-SECRET",
);

/// A scope no adapter has.
const FOREIGN_SCOPE: &str = "allama-conformance-foreign-scope";

/// Credentials addressed to [`FOREIGN_SCOPE`].
const FOREIGN_SECRET: &str = "allama-conformance-foreign-SECRET";
const FOREIGN_HEADER: (&str, &str) = (
    "x-allama-conformance-foreign",
    "allama-conformance-foreign-header-SECRET",
);

/// The value of a plain credential header, which must be refused.
const PLAIN_SECRET: &str = "allama-conformance-plain-SECRET";

/// What one call through an adapter did: its result and the HTTP requests
/// that reached the transport.
#[derive(Debug)]
#[non_exhaustive]
pub struct TransportOutcome {
    /// The call's result, without its value.
    pub result: allama::Result<()>,
    /// The HTTP requests the adapter sent, in order (for example
    /// `allama_http::ReplayTransport::requests`). Empty when it failed
    /// before sending.
    pub sent: Vec<HttpRequest>,
}

impl TransportOutcome {
    /// An outcome from a call's result and the requests it sent.
    #[must_use]
    pub fn new<T>(result: allama::Result<T>, sent: Vec<HttpRequest>) -> Self {
        Self {
            result: result.map(|_| ()),
            sent,
        }
    }
}

/// Assert that an HTTP adapter applies the caller's transport options to the
/// HTTP requests it sends, with `slot`, its credential slot.
///
/// Callers attach `allama_http::HttpRequestOptions` to a request to send
/// extra HTTP headers with it (gateway tags) and per-request credentials,
/// each addressed to a credential scope. Every HTTP adapter applies them by
/// calling `allama_http::HttpRequest::apply_options` with the options of the
/// request it serves and its own `CredentialSlot`, once its own headers are
/// set.
///
/// `send` receives [`RequestOptions`] holding transport options, attaches
/// them to a request, makes one call through an adapter built with a
/// configured credential (whether the call succeeds does not matter), and
/// returns a [`TransportOutcome`]. Run it once per call path the adapter
/// has (unary, streaming, embedding, decision). It checks:
///
/// - **Headers.** `x-allama-conformance: applied`, which the adapter does
///   not set, is added, and `content-type: application/json;
///   charset=utf-8` replaces the adapter's own `content-type`. They come in
///   the JSON form under the `allama-http` options namespace, unless `slot`
///   is [required](CredentialSlot::required), when they come typed beside
///   a credential.
/// - **The adapter's own credential.** A `Credential::api_key` addressed
///   to `slot.scope` is sent once, in the slot's placement, replacing the
///   configured credential and every header in `slot.replaces`; a
///   `Credential::header` addressed to the scope is sent once. A
///   [`Custom`](CredentialPlacement::Custom) placement is the adapter's to
///   check.
/// - **Foreign credentials.** Credentials addressed to another scope never
///   appear in a header, the URL or the body.
/// - **Refusals.** A plain `authorization` header, the slot's own header,
///   and each header it replaces are refused with
///   [`Error::Configuration`] before anything is sent. A required slot
///   refuses a request with no credential for its scope the same way.
/// - **Redaction.** No secret appears in the `Debug` output or the
///   serialized form of the options (what a `RecordingProvider` exchange
///   stores), in the `Debug` output of a sent request, or in the text of an
///   error.
///
/// ```
/// use allama::{ChatRequest, RequestOptions};
/// use allama_conformance::{TransportOutcome, assert_transport_options_applied};
/// use allama_http::http::{HeaderValue, Uri};
/// use allama_http::{CredentialPlacement, CredentialSlot, HttpRequest};
///
/// # struct Adapter { slot: CredentialSlot }
/// # impl Adapter {
/// #     async fn send(&self, request: &ChatRequest) -> TransportOutcome {
/// #         let http = HttpRequest::post(Uri::from_static("https://api.example.com/v1/chat"), "{}")
/// #             .header("content-type", HeaderValue::from_static("application/json"))
/// #             .header("authorization", HeaderValue::from_static("Bearer configured-key"))
/// #             .apply_options(&request.options, &self.slot);
/// #         match http {
/// #             Ok(http) => TransportOutcome::new(Ok(()), vec![http]),
/// #             Err(err) => TransportOutcome::new(Err::<(), _>(err), Vec::new()),
/// #         }
/// #     }
/// # }
/// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
/// let slot = CredentialSlot::new("example", CredentialPlacement::Bearer);
/// let adapter = Adapter { slot: slot.clone() };
/// assert_transport_options_applied(&slot, |options: RequestOptions| async {
///     // Build the adapter over a recording transport (for example
///     // `allama_http::ReplayTransport`) with a configured key, make one call
///     // with `options` attached, and return its result with the HTTP
///     // requests it sent.
///     let mut request = ChatRequest::new("m").user("Hi");
///     request.options = options;
///     adapter.send(&request).await
/// })
/// .await;
/// # });
/// ```
///
/// # Panics
///
/// When any check fails.
pub async fn assert_transport_options_applied<F, Fut>(slot: &CredentialSlot, send: F)
where
    F: Fn(RequestOptions) -> Fut,
    Fut: Future<Output = TransportOutcome>,
{
    assert_headers_applied(slot, &send).await;
    assert_credentials_addressed(slot, &send).await;
    assert_plain_credentials_refused(slot, &send).await;
    if slot.required {
        let outcome = send(RequestOptions::new()).await;
        assert_refused(
            &outcome,
            "a request without a credential for a required slot",
        );
    }
}

/// The headers are added or replace the adapter's own.
async fn assert_headers_applied<F, Fut>(slot: &CredentialSlot, send: &F)
where
    F: Fn(RequestOptions) -> Fut,
    Fut: Future<Output = TransportOutcome>,
{
    let mut options = RequestOptions::new();
    if slot.required {
        options.insert(
            HttpRequestOptions::new()
                .header(ADDED.0, HeaderValue::from_static(ADDED.1))
                .header(REPLACED.0, HeaderValue::from_static(REPLACED.1))
                .credential(slot.scope.clone(), Credential::api_key(OWN_SECRET)),
        );
    } else {
        options.insert_json(
            OPTIONS_NAMESPACE,
            json!({"headers": {ADDED.0: ADDED.1, REPLACED.0: REPLACED.1}}),
        );
    }
    let outcome = send(options).await;
    let request = last_sent(&outcome, "transport options with headers");
    for (name, value) in [ADDED, REPLACED] {
        assert_eq!(
            values(request, name),
            [value],
            "the `{name}` header from the transport options must be sent exactly once, \
             replacing the adapter's own; the adapter sent {:?}",
            header_names(request)
        );
    }
}

/// The scope's credentials are placed, and other scopes' never sent.
async fn assert_credentials_addressed<F, Fut>(slot: &CredentialSlot, send: &F)
where
    F: Fn(RequestOptions) -> Fut,
    Fut: Future<Output = TransportOutcome>,
{
    let transport = HttpRequestOptions::new()
        .credential(FOREIGN_SCOPE, Credential::api_key(FOREIGN_SECRET))
        .credential(
            FOREIGN_SCOPE,
            Credential::header(FOREIGN_HEADER.0, FOREIGN_HEADER.1),
        )
        .credential(slot.scope.clone(), Credential::api_key(OWN_SECRET))
        .credential(
            slot.scope.clone(),
            Credential::header(OWN_HEADER.0, OWN_HEADER.1),
        );
    let mut options = RequestOptions::new();
    options.insert(transport);
    assert_options_redacted(&options);

    let outcome = send(options).await;
    assert_error_redacted(&outcome);
    last_sent(&outcome, "credentials addressed to the adapter's scope");
    for request in &outcome.sent {
        let names = header_names(request);
        match (&slot.placement, slot.header_name()) {
            (CredentialPlacement::Custom, _) | (_, None) => {}
            (placement, Some(header)) => {
                let expected = match placement {
                    CredentialPlacement::Bearer => format!("Bearer {OWN_SECRET}"),
                    _ => OWN_SECRET.to_owned(),
                };
                assert_eq!(
                    values(request, header),
                    [expected.as_str()],
                    "an api key addressed to credential scope `{}` must be sent once in `{header}`, \
                     replacing the configured credential; the adapter sent {names:?}",
                    slot.scope
                );
                for replaced in &slot.replaces {
                    assert!(
                        values(request, replaced).is_empty(),
                        "the scope's api key must replace the configured `{replaced}` header; \
                         the adapter sent {names:?}"
                    );
                }
            }
        }
        assert_eq!(
            values(request, OWN_HEADER.0),
            [OWN_HEADER.1],
            "a header credential addressed to credential scope `{}` must be sent once; the \
             adapter sent {names:?}",
            slot.scope
        );
        let body = &request.body;
        for secret in [FOREIGN_SECRET, FOREIGN_HEADER.1] {
            let leaked = request
                .headers
                .values()
                .any(|value| contains(value.as_bytes(), secret.as_bytes()))
                || request.uri.to_string().contains(secret)
                || contains(body, secret.as_bytes());
            assert!(
                !leaked,
                "a credential addressed to another credential scope must never be sent; the \
                 adapter sent it to {}",
                allama_http::config::redact_url(&request.uri.to_string())
            );
        }
        assert!(
            values(request, FOREIGN_HEADER.0).is_empty(),
            "a header credential addressed to another credential scope must never be sent"
        );
        let debug = format!("{request:?}");
        for secret in [OWN_SECRET, OWN_HEADER.1, FOREIGN_SECRET, FOREIGN_HEADER.1] {
            assert!(
                !debug.contains(secret),
                "the `Debug` output of a sent request must redact credentials: {debug}"
            );
        }
    }
}

/// Plain headers that would carry a credential are refused before sending.
async fn assert_plain_credentials_refused<F, Fut>(slot: &CredentialSlot, send: &F)
where
    F: Fn(RequestOptions) -> Fut,
    Fut: Future<Output = TransportOutcome>,
{
    let mut names = vec!["authorization"];
    names.extend(slot.header_name());
    names.extend(slot.replaces.iter().map(AsRef::as_ref));
    for name in names {
        let name = HeaderName::try_from(name).expect("a slot's header is a valid header name");
        let mut transport =
            HttpRequestOptions::new().header(name.clone(), HeaderValue::from_static(PLAIN_SECRET));
        if slot.required {
            transport = transport.credential(slot.scope.clone(), Credential::api_key(OWN_SECRET));
        }
        let mut options = RequestOptions::new();
        options.insert(transport);
        let outcome = send(options).await;
        assert_refused(&outcome, &format!("the plain credential header `{name}`"));
    }
}

/// `outcome` failed with a configuration error before anything was sent.
fn assert_refused(outcome: &TransportOutcome, what: &str) {
    assert!(
        outcome.sent.is_empty(),
        "{what} must be refused before anything is sent, but the adapter sent {} request(s)",
        outcome.sent.len()
    );
    match &outcome.result {
        Err(Error::Configuration { .. }) => {}
        Err(other) => panic!(
            "{what} must be refused with `Error::Configuration`, not {}",
            other.as_log()
        ),
        Ok(()) => panic!("{what} must be refused with `Error::Configuration`"),
    }
    assert_error_redacted(outcome);
}

/// No secret in the options' `Debug` output or serialized form.
fn assert_options_redacted(options: &RequestOptions) {
    let debug = format!("{options:?} {:?}", options.get::<HttpRequestOptions>());
    let serialized = serde_json::to_string(options).expect("options serialize");
    for secret in SECRETS {
        assert!(
            !debug.contains(secret),
            "the `Debug` output of transport options must redact credentials: {debug}"
        );
        assert!(
            !serialized.contains(secret),
            "serialized transport options must omit credentials: {serialized}"
        );
    }
}

/// No secret in the text of the outcome's error.
fn assert_error_redacted(outcome: &TransportOutcome) {
    let Err(error) = &outcome.result else {
        return;
    };
    let text = format!(
        "{error} {error:?} {} {}",
        error.as_log(),
        error.upstream_message().unwrap_or_default()
    );
    for secret in SECRETS {
        assert!(
            !text.contains(secret),
            "error text must never contain a credential: {}",
            error.as_log()
        );
    }
}

/// Every secret the contract sends.
const SECRETS: [&str; 5] = [
    OWN_SECRET,
    OWN_HEADER.1,
    FOREIGN_SECRET,
    FOREIGN_HEADER.1,
    PLAIN_SECRET,
];

/// The last request `outcome` sent.
fn last_sent<'a>(outcome: &'a TransportOutcome, what: &str) -> &'a HttpRequest {
    outcome.sent.last().unwrap_or_else(|| {
        let error = outcome
            .result
            .as_ref()
            .err()
            .map(|err| err.as_log().to_string());
        panic!("the adapter sent no request with {what}; it returned {error:?}")
    })
}

/// The values of `name` in `request`, ignoring case, decoded lossily.
fn values(request: &HttpRequest, name: &str) -> Vec<String> {
    request
        .headers
        .get_all(name)
        .iter()
        .map(|value| String::from_utf8_lossy(value.as_bytes()).into_owned())
        .collect()
}

/// The header names of `request`, for messages that must not show values.
fn header_names(request: &HttpRequest) -> Vec<&str> {
    request.headers.keys().map(HeaderName::as_str).collect()
}

/// Whether `haystack` contains `needle`.
fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use allama_http::http::Uri;

    use super::*;

    fn post() -> HttpRequest {
        HttpRequest::post(Uri::from_static("https://api.example.com/v1"), "{}")
            .header("content-type", HeaderValue::from_static("application/json"))
    }

    fn value(value: &str) -> HeaderValue {
        HeaderValue::try_from(value).unwrap()
    }

    /// A conforming adapter: configured headers, then `apply_options`.
    fn adapter(
        slot: CredentialSlot,
    ) -> impl Fn(RequestOptions) -> std::future::Ready<TransportOutcome> {
        move |options| {
            let mut request = post();
            request = match slot.header_name() {
                Some("authorization") => {
                    request.header("authorization", value("Bearer configured"))
                }
                Some(name) => {
                    request.header(HeaderName::try_from(name).unwrap(), value("configured"))
                }
                None => request,
            };
            for replaced in &slot.replaces {
                request = request.header(
                    HeaderName::try_from(replaced.as_ref()).unwrap(),
                    value("configured"),
                );
            }
            std::future::ready(match request.apply_options(&options, &slot) {
                Ok(request) => TransportOutcome::new(Ok(()), vec![request]),
                Err(err) => TransportOutcome::new(Err::<(), _>(err), Vec::new()),
            })
        }
    }

    fn bearer() -> CredentialSlot {
        CredentialSlot::new("example", CredentialPlacement::Bearer)
    }

    #[tokio::test]
    async fn an_adapter_that_applies_the_options_passes() {
        assert_transport_options_applied(&bearer(), adapter(bearer())).await;
        let header = CredentialSlot::new("nimbus", CredentialPlacement::Header("x-nimbus".into()))
            .replaces("authorization");
        assert_transport_options_applied(&header, adapter(header.clone())).await;
        let required = bearer().required(true);
        assert_transport_options_applied(&required, adapter(required.clone())).await;
        let custom = CredentialSlot::new("signed", CredentialPlacement::Custom);
        assert_transport_options_applied(&custom, adapter(custom.clone())).await;
    }

    #[tokio::test]
    #[should_panic(expected = "x-allama-conformance")]
    async fn an_adapter_that_ignores_the_options_fails() {
        assert_transport_options_applied(&bearer(), |_options: RequestOptions| {
            let request = post();
            std::future::ready(TransportOutcome::new(Ok(()), vec![request]))
        })
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "Content-Type")]
    async fn an_adapter_that_appends_instead_of_replacing_fails() {
        assert_transport_options_applied(&bearer(), |options: RequestOptions| {
            let json = options.json(OPTIONS_NAMESPACE).cloned().unwrap_or_default();
            let mut request = post();
            for (name, value) in json["headers"].as_object().into_iter().flatten() {
                request = request.header(
                    HeaderName::try_from(name.as_str()).unwrap(),
                    self::value(value.as_str().unwrap()),
                );
            }
            std::future::ready(TransportOutcome::new(Ok(()), vec![request]))
        })
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "must be sent once in `authorization`")]
    async fn an_adapter_with_another_scope_fails() {
        assert_transport_options_applied(
            &bearer(),
            adapter(CredentialSlot::new("other", CredentialPlacement::Bearer)),
        )
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "another credential scope must never be sent")]
    async fn an_adapter_that_sends_every_credential_fails() {
        assert_transport_options_applied(&bearer(), |options: RequestOptions| {
            let slot = bearer();
            let mut request = post().header("authorization", value("Bearer configured"));
            let foreign = options
                .get::<HttpRequestOptions>()
                .and_then(|transport| transport.credential_for(FOREIGN_SCOPE))
                .map(|credential| credential.expose_secret().to_owned());
            let outcome = match request.clone().apply_options(&options, &slot) {
                Ok(applied) => {
                    request = applied;
                    if let Some(secret) = foreign {
                        request = request.header("x-leak", value(&secret));
                    }
                    TransportOutcome::new(Ok(()), vec![request])
                }
                Err(err) => TransportOutcome::new(Err::<(), _>(err), Vec::new()),
            };
            std::future::ready(outcome)
        })
        .await;
    }

    #[tokio::test]
    #[should_panic(expected = "must be refused before anything is sent")]
    async fn an_adapter_that_sends_plain_credential_headers_fails() {
        // An adapter that replaces headers itself instead of calling
        // `apply_options`.
        assert_transport_options_applied(&bearer(), |options: RequestOptions| {
            let mut request = post().header("authorization", value("Bearer configured"));
            if let Some(transport) = options.resolve::<HttpRequestOptions>().unwrap() {
                for (name, value) in &transport.headers {
                    request.headers.remove(name);
                    request = request.header(name.clone(), value.clone());
                }
                if let Some(credential) = transport.credential_for("example") {
                    request.headers.remove("authorization");
                    request = request.sensitive_header(
                        "authorization",
                        value(&format!("Bearer {}", credential.expose_secret())),
                    );
                }
                for (scope, credential) in transport.credentials() {
                    if let (allama_http::CredentialKind::Header(name), "example") =
                        (credential.kind(), scope)
                    {
                        request = request.sensitive_header(
                            HeaderName::try_from(name.as_str()).unwrap(),
                            value(credential.expose_secret()),
                        );
                    }
                }
            }
            std::future::ready(TransportOutcome::new(Ok(()), vec![request]))
        })
        .await;
    }
}
