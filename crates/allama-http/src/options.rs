//! Per-request transport options.

use std::collections::BTreeMap;
use std::fmt;

use allama::ProviderOptions;
use http::header::IntoHeaderName;
use http::{HeaderMap, HeaderName, HeaderValue};
use serde::de::{Error as _, IgnoredAny};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{Credential, CredentialKind, RedactedHeaders, is_credential_header};
#[cfg(doc)]
use crate::{CredentialSlot, is_sensitive_header};

/// The options namespace of [`HttpRequestOptions`]: `"allama-http"`.
///
/// It belongs to no provider: every HTTP adapter reads it through
/// [`HttpRequest::apply_options`](crate::HttpRequest::apply_options), and no
/// provider mints state under it.
pub const OPTIONS_NAMESPACE: &str = "allama-http";

/// Transport options for one request: HTTP headers and credentials to send
/// with it.
///
/// Attach them to a [`ChatRequest`](allama::ChatRequest),
/// [`EmbeddingRequest`](allama::EmbeddingRequest) or
/// [`DecisionRequest`](allama::DecisionRequest) like any options. Every
/// bundled HTTP adapter applies them with
/// [`HttpRequest::apply_options`](crate::HttpRequest::apply_options), and so
/// should third-party ones. Use them for gateway tags (tenant, route,
/// trace, a request id), and for per-tenant credentials:
///
/// ```
/// use allama::ChatRequest;
/// use allama_http::http::HeaderValue;
/// use allama_http::{Credential, HttpRequestOptions};
///
/// let request = ChatRequest::new("gpt-4o-mini").user("Hi").with_option(
///     HttpRequestOptions::new()
///         .header("x-tenant-id", HeaderValue::from_static("acme"))
///         .credential("openai", Credential::api_key("sk-tenant-key")),
/// );
/// # let _ = request;
/// ```
///
/// # From a wrapper
///
/// Header and credential injection happen at the transport, and these
/// options are how a wrapper reaches it: a wrapper sees the portable
/// request, never the HTTP request (ADR 0013). A wrapper that tags every
/// call, with a request id header for example, adds its header to the
/// options the request already carries, through
/// [`ChatRequest::option_mut`](allama::ChatRequest::option_mut), so the
/// caller's headers and credentials survive; a fresh `HttpRequestOptions`
/// given to `with_option` would replace them.
///
/// ```
/// use allama::ChatRequest;
/// use allama_http::http::HeaderValue;
/// use allama_http::{Credential, HttpRequestOptions};
///
/// fn tag(request: &ChatRequest, request_id: HeaderValue) -> ChatRequest {
///     let mut request = request.clone();
///     match request.option_mut::<HttpRequestOptions>() {
///         Some(options) => {
///             options.headers.insert("x-request-id", request_id);
///         }
///         None => {
///             let options = HttpRequestOptions::new().header("x-request-id", request_id);
///             request = request.with_option(options);
///         }
///     }
///     request
/// }
///
/// let request = ChatRequest::new("m").user("Hi").with_option(
///     HttpRequestOptions::new().credential("openai", Credential::api_key("sk-tenant")),
/// );
/// let tagged = tag(&request, HeaderValue::from_static("req-1"));
/// let options = tagged.option::<HttpRequestOptions>().unwrap();
/// assert_eq!(options.headers["x-request-id"], "req-1");
/// assert!(options.credential_for("openai").is_some(), "the caller's key survives");
/// ```
///
/// # Headers
///
/// A header given here replaces every header of the same name (ignoring
/// case) that the adapter set; headers with other names are added. A plain
/// header never carries a credential: one named on the exact credential list
/// ([`is_credential_header`]), or one the adapter's [`CredentialSlot`]
/// writes or replaces, is refused before anything is sent.
///
/// # Credentials
///
/// [`credential`](Self::credential) addresses a [`Credential`] to a
/// **credential scope**: the account it belongs to. Only the adapter whose
/// [`CredentialSlot`] has that scope sends it, so in a fallback chain each
/// provider gets its own key and no other. Bundled adapters default their
/// scope to their provider name (`openai`, `anthropic`, `groq`, ...), each
/// crate publishes it as `DEFAULT_CREDENTIAL_SCOPE`, and their builders
/// change it with `credential_scope`.
///
/// - The scope's [`Credential::api_key`] replaces the key the adapter was
///   configured with. An adapter configured with a key still sends its own
///   key when no credential is addressed to its scope, so a mistyped scope
///   falls back to the configured key; an adapter built without a key
///   refuses such a request instead.
/// - A [`Credential::header`] is added to requests for its scope, replacing
///   a configured header of the same name.
///
/// Two accounts of one provider in a fallback chain need two scopes: by
/// default they share one, and both would send the same key.
///
/// Which hops are protected: plain headers are refused by each adapter as it
/// sends, so a plain header that only one adapter owns (`x-nimbus-key`)
/// still reaches the providers tried before it in a fallback chain. Only
/// the exact credential list protects every hop. Send third-party keys as a
/// [`Credential::header`] addressed to their scope.
///
/// # Secrets stay in the process
///
/// - `Debug` output redacts credentials and credential headers (the names
///   [`is_sensitive_header`] matches, and values marked
///   [sensitive](HeaderValue::set_sensitive)), as for
///   [`HttpRequest`](crate::HttpRequest).
/// - Serialization omits credentials and the headers on the exact
///   credential list ([`is_credential_header`]), so a serialized request, a
///   `RecordingProvider` exchange, or a fixture never holds them. A request
///   rebuilt from one carries no credentials. Two requests that differ only
///   in credentials compare equal.
/// - The JSON form (`with_options_json("allama-http", ..)`) rejects them
///   with [`Error::InvalidRequest`](allama::Error::InvalidRequest): JSON
///   options are exported and printed as given. Set credentials with the
///   typed value.
///
/// Every other header is a plain header, exported and accepted in the JSON
/// form as given, even one whose name only looks like a credential
/// (`Idempotency-Key`, `x-gateway-token`): an idempotency key must survive
/// a recording and a replay. `Debug` still redacts those by name. A secret
/// belongs in a [`Credential::header`], never in a plain header.
///
/// Adapters never put header values on tracing spans.
///
/// # JSON form
///
/// `{"headers": {"x-tenant-id": "acme"}}` under [`OPTIONS_NAMESPACE`].
/// Header names are exported in lowercase, and a header given more than
/// once is exported once, with its values joined by `", "`. A name or value
/// that is not a valid header is rejected with
/// [`Error::InvalidRequest`](allama::Error::InvalidRequest).
#[derive(Clone, Default)]
#[non_exhaustive]
pub struct HttpRequestOptions {
    /// Headers to send. Each name replaces the adapter's headers of the
    /// same name.
    pub headers: HeaderMap,
    /// Credentials by scope, in order: at most one placed by the adapter
    /// per scope, and one header credential per scope and header name.
    credentials: Vec<(String, Credential)>,
}

impl HttpRequestOptions {
    /// No options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a header to send, after the values of the same name already
    /// set.
    ///
    /// The name is a [`HeaderName`] or a `&'static str`, which panics when
    /// it is not a valid header name, as in [`HeaderMap::append`]. A header
    /// that carries a credential is refused when the request is sent; use
    /// [`credential`](Self::credential) for those.
    #[must_use]
    pub fn header(mut self, name: impl IntoHeaderName, value: HeaderValue) -> Self {
        self.headers.append(name, value);
        self
    }

    /// Address `credential` to the credential scope `scope`.
    ///
    /// It replaces the scope's earlier credential of the same kind: its
    /// [`Credential::api_key`], or its [`Credential::header`] of the same
    /// name.
    #[must_use]
    pub fn credential(mut self, scope: impl Into<String>, credential: Credential) -> Self {
        let scope = scope.into();
        self.credentials.retain(|(existing_scope, existing)| {
            *existing_scope != scope || !same_slot(existing, &credential)
        });
        self.credentials.push((scope, credential));
        self
    }

    /// The credential addressed to `scope` that the adapter places itself:
    /// its [`Credential::api_key`] (header credentials are added by
    /// [`HttpRequest::apply_options`](crate::HttpRequest::apply_options)).
    ///
    /// An adapter with a [`CredentialPlacement::Custom`] slot reads its
    /// credential here, after `apply_options`.
    ///
    /// [`CredentialPlacement::Custom`]: crate::CredentialPlacement::Custom
    #[must_use]
    pub fn credential_for(&self, scope: &str) -> Option<&Credential> {
        self.credentials
            .iter()
            .find(|(existing, credential)| existing == scope && credential.is_placed_by_adapter())
            .map(|(_, credential)| credential)
    }

    /// Every credential, with the scope it is addressed to, in order.
    pub fn credentials(&self) -> impl Iterator<Item = (&str, &Credential)> {
        self.credentials
            .iter()
            .map(|(scope, credential)| (scope.as_str(), credential))
    }

    /// The header credentials addressed to `scope`, in order.
    pub(crate) fn header_credentials_for<'a>(
        &'a self,
        scope: &'a str,
    ) -> impl Iterator<Item = (&'a str, &'a str)> {
        self.credentials
            .iter()
            .filter(move |(existing, _)| existing == scope)
            .filter_map(|(_, credential)| match credential.kind() {
                CredentialKind::Header(name) => Some((name.as_str(), credential.expose_secret())),
                _ => None,
            })
    }
}

impl fmt::Debug for HttpRequestOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpRequestOptions")
            .field("headers", &RedactedHeaders(&self.headers))
            .field("credentials", &self.credentials)
            .finish()
    }
}

impl ProviderOptions for HttpRequestOptions {
    const NAMESPACE: &'static str = OPTIONS_NAMESPACE;
}

/// Writes the plain headers; credentials and the headers on the exact
/// credential list are omitted.
impl Serialize for HttpRequestOptions {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut headers: BTreeMap<&str, String> = BTreeMap::new();
        let exported = self
            .headers
            .iter()
            .filter(|(name, _)| !is_credential_header(name.as_str()));
        for (name, value) in exported {
            let value = String::from_utf8_lossy(value.as_bytes());
            headers
                .entry(name.as_str())
                .and_modify(|joined| {
                    joined.push_str(", ");
                    joined.push_str(&value);
                })
                .or_insert_with(|| value.into_owned());
        }
        Wire {
            headers,
            credentials: None,
        }
        .serialize(serializer)
    }
}

/// Reads the JSON form, which takes no credentials.
impl<'de> Deserialize<'de> for HttpRequestOptions {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = Wire::<String>::deserialize(deserializer)?;
        if wire.credentials.is_some() {
            return Err(D::Error::custom(
                "credentials cannot be set in the JSON form, which is exported and printed \
                 as given; attach them with the typed `HttpRequestOptions::credential`",
            ));
        }
        if let Some(name) = wire.headers.keys().find(|name| is_credential_header(name)) {
            return Err(D::Error::custom(format!(
                "header `{name}` carries a credential and JSON options are exported as \
                 given; send credentials with the typed `HttpRequestOptions::credential`, \
                 whose export omits them"
            )));
        }
        let mut headers = HeaderMap::new();
        for (name, value) in wire.headers {
            let Ok(header) = HeaderName::try_from(name.as_str()) else {
                return Err(D::Error::custom(format!(
                    "`{name}` is not a valid HTTP header name"
                )));
            };
            let Ok(value) = HeaderValue::try_from(value) else {
                return Err(D::Error::custom(format!(
                    "the value of header `{name}` contains characters an HTTP header cannot carry"
                )));
            };
            headers.append(header, value);
        }
        Ok(Self {
            headers,
            credentials: Vec::new(),
        })
    }
}

/// The JSON form of [`HttpRequestOptions`].
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire<K: Ord> {
    #[serde(default = "BTreeMap::new", skip_serializing_if = "BTreeMap::is_empty")]
    headers: BTreeMap<K, String>,
    /// Present only to refuse credentials with a clear message.
    #[serde(default, skip_serializing)]
    credentials: Option<IgnoredAny>,
}

/// Whether `new` takes `existing`'s place in a scope: both placed by the
/// adapter, or both headers of the same name.
fn same_slot(existing: &Credential, new: &Credential) -> bool {
    match (existing.kind(), new.kind()) {
        (CredentialKind::Header(a), CredentialKind::Header(b)) => a.eq_ignore_ascii_case(b),
        (CredentialKind::Header(_), _) | (_, CredentialKind::Header(_)) => false,
        _ => true,
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use allama::{ChatRequest, RequestOptions};
    use serde_json::json;

    use super::*;

    fn value(value: &'static str) -> HeaderValue {
        HeaderValue::from_static(value)
    }

    #[test]
    fn export_omits_credentials_and_joins_repeated_headers() {
        let options = HttpRequestOptions::new()
            .header("Authorization", value("Bearer SECRET1"))
            .header("x-api-key", value("SECRET2"))
            .header("x-tag", value("a"))
            .header("X-Tag", value("b"))
            .header("x-tenant-id", value("acme"));
        assert_eq!(
            serde_json::to_value(&options).unwrap(),
            json!({"headers": {"x-tag": "a, b", "x-tenant-id": "acme"}})
        );
        assert_eq!(
            serde_json::to_value(HttpRequestOptions::new().header("x-api-key", value("SECRET")))
                .unwrap(),
            json!({})
        );
    }

    #[test]
    fn json_form_round_trips_and_rejects_credentials() {
        let mut options = RequestOptions::new();
        options.insert_json(
            OPTIONS_NAMESPACE,
            json!({"headers": {"x-tenant-id": "acme"}}),
        );
        let resolved = options.resolve::<HttpRequestOptions>().unwrap().unwrap();
        assert_eq!(resolved.headers["x-tenant-id"], "acme");

        options.insert_json(
            OPTIONS_NAMESPACE,
            json!({"headers": {"Authorization": "Bearer SECRET"}}),
        );
        let err = options.resolve::<HttpRequestOptions>().unwrap_err();
        assert!(matches!(err, allama::Error::InvalidRequest { .. }), "{err}");
        assert!(err.to_string().contains("`Authorization`"), "{err}");
        assert!(!err.to_string().contains("SECRET"), "{err}");

        options.insert_json(OPTIONS_NAMESPACE, json!({"header": {}}));
        assert!(options.resolve::<HttpRequestOptions>().is_err());
        options.insert_json(OPTIONS_NAMESPACE, json!({}));
        assert!(
            options
                .resolve::<HttpRequestOptions>()
                .unwrap()
                .unwrap()
                .headers
                .is_empty()
        );
    }

    /// The JSON form is untrusted input: a name or value no header can
    /// carry is the caller's invalid request, and the value is never
    /// quoted.
    #[test]
    fn json_form_rejects_invalid_headers() {
        for headers in [
            json!({"bad name": "v"}),
            json!({"x-ok": "SECRET\r\nx-injected: 1"}),
        ] {
            let mut options = RequestOptions::new();
            options.insert_json(OPTIONS_NAMESPACE, json!({ "headers": headers }));
            let err = options.resolve::<HttpRequestOptions>().unwrap_err();
            assert!(matches!(err, allama::Error::InvalidRequest { .. }), "{err}");
            assert!(!format!("{err} {err:?}").contains("SECRET"), "{err}");
        }
    }

    #[test]
    fn json_form_rejects_credentials() {
        let mut options = RequestOptions::new();
        options.insert_json(
            OPTIONS_NAMESPACE,
            json!({"credentials": [{"scope": "openai", "kind": "api_key", "secret": "SECRET"}]}),
        );
        let err = options.resolve::<HttpRequestOptions>().unwrap_err();
        assert!(matches!(err, allama::Error::InvalidRequest { .. }), "{err}");
        assert!(err.to_string().contains("credentials"), "{err}");
        assert!(!err.to_string().contains("SECRET"), "{err}");

        options.insert_json(OPTIONS_NAMESPACE, json!({"headers": {"x-api-key": "k"}}));
        assert!(options.resolve::<HttpRequestOptions>().is_err());
    }

    /// A plain header whose name merely looks like a credential is
    /// accepted in the JSON form and survives an export, so an idempotency
    /// key is never dropped silently; `Debug` still redacts it.
    #[test]
    fn plain_headers_that_look_like_credentials_round_trip() {
        let mut options = RequestOptions::new();
        options.insert_json(
            OPTIONS_NAMESPACE,
            json!({"headers": {"Idempotency-Key": "idem-1", "x-author": "a"}}),
        );
        let resolved = options.resolve::<HttpRequestOptions>().unwrap().unwrap();
        assert_eq!(resolved.headers["idempotency-key"], "idem-1");
        let typed = HttpRequestOptions::new()
            .header("Idempotency-Key", value("idem-1"))
            .header("x-author", value("a"))
            .header("Authorization", value("Bearer SECRET"));
        assert_eq!(
            serde_json::to_value(&typed).unwrap(),
            json!({"headers": {"idempotency-key": "idem-1", "x-author": "a"}})
        );
        let debug = format!("{typed:?}");
        assert!(
            !debug.contains("idem-1") && !debug.contains("SECRET"),
            "{debug}"
        );
    }

    #[test]
    fn credentials_stay_out_of_debug_and_serialized_requests() {
        let request = ChatRequest::new("m").user("Hi").with_option(
            HttpRequestOptions::new()
                .header("Idempotency-Key", value("idem-0"))
                .credential("openai", Credential::api_key("SECRET1"))
                .credential(
                    "openai",
                    Credential::header("cf-aig-authorization", "Bearer SECRET2"),
                )
                .header("x-tenant-id", value("acme")),
        );
        let debug = format!("{request:?}");
        assert!(!debug.contains("SECRET"), "{debug}");
        let options_debug = format!("{:?}", request.options.get::<HttpRequestOptions>());
        assert!(!options_debug.contains("SECRET"), "{options_debug}");
        assert!(options_debug.contains("acme"), "{options_debug}");
        assert!(
            options_debug.contains("cf-aig-authorization"),
            "{options_debug}"
        );

        let serialized = serde_json::to_string(&request).unwrap();
        assert!(!serialized.contains("SECRET"), "{serialized}");
        assert!(!serialized.contains("credentials"), "{serialized}");
        assert!(serialized.contains(
            r#""allama-http":{"headers":{"idempotency-key":"idem-0","x-tenant-id":"acme"}}"#
        ));

        let rebuilt: ChatRequest = serde_json::from_str(&serialized).unwrap();
        let resolved = rebuilt
            .options
            .resolve::<HttpRequestOptions>()
            .unwrap()
            .unwrap();
        assert_eq!(resolved.headers["Idempotency-Key"], "idem-0");
        assert_eq!(resolved.headers["x-tenant-id"], "acme");
        assert_eq!(resolved.credentials().count(), 0);
        assert_eq!(rebuilt, request);
    }
}
