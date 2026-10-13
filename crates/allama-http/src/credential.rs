//! Per-request credentials: what a caller sends, and where an adapter puts
//! it.
//!
//! A [`Credential`] is a typed secret a caller attaches to one request with
//! [`HttpRequestOptions::credential`](crate::HttpRequestOptions::credential),
//! addressed to a **credential scope**: the account it belongs to. A
//! [`CredentialSlot`] is the adapter's side: its scope, and where its
//! credential goes on the wire. [`HttpRequest::apply_options`] matches the
//! two, so a key meant for one account never reaches another provider, even
//! when a fallback sends the same request to both.

use std::borrow::Cow;
use std::fmt;

use serde::de::Error as _;
use serde::ser::SerializeStruct as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(doc)]
use crate::HttpRequest;

/// A credential for one request: a typed, redacted secret.
///
/// Attach it with
/// [`HttpRequestOptions::credential`](crate::HttpRequestOptions::credential),
/// addressed to a credential scope. Only the adapter whose
/// [`CredentialSlot`] has that scope sends it, placed where that adapter's
/// API expects it:
///
/// - [`Credential::api_key`] is the provider's own key. It replaces the key
///   the provider was configured with.
/// - [`Credential::header`] is any other authentication header, such as a
///   gateway token (`cf-aig-authorization`, `ocp-apim-subscription-key`) or
///   a third-party adapter's own header. The secret is the whole header
///   value.
///
/// ```
/// use allama::ChatRequest;
/// use allama_http::{Credential, HttpRequestOptions};
///
/// let request = ChatRequest::new("gpt-5-mini").user("Hi").with_option(
///     HttpRequestOptions::new()
///         .credential("openai", Credential::api_key("sk-tenant-key"))
///         .credential("openai", Credential::header("cf-aig-authorization", "Bearer gw-token")),
/// );
/// assert!(!format!("{request:?}").contains("sk-tenant-key"));
/// ```
///
/// # Secrets stay in the process
///
/// `Debug` shows the kind and never the secret, and so does every error
/// allama-http writes about a credential. Serializing a credential writes
/// its kind and omits the secret, while deserializing one reads the secret:
/// the serde impl is asymmetric on purpose, so a credential can come from a
/// configuration file but is never written back out. The JSON form of
/// [`HttpRequestOptions`](crate::HttpRequestOptions) does not take
/// credentials, because JSON options are exported and printed as given;
/// deserialize the credential and attach it typed.
///
/// ```
/// use allama_http::Credential;
///
/// let credential: Credential =
///     serde_json::from_str(r#"{"kind": "api_key", "secret": "sk-SECRET"}"#).unwrap();
/// assert_eq!(credential.expose_secret(), "sk-SECRET");
/// assert_eq!(serde_json::to_string(&credential).unwrap(), r#"{"kind":"api_key"}"#);
/// ```
///
/// # No equality, no zeroing
///
/// A credential is not `PartialEq`: comparing
/// secrets is not something a request option needs, and a derived
/// comparison would short-circuit on the first differing byte. Compare
/// [`kind`](Self::kind)s, or compare [`expose_secret`](Self::expose_secret)
/// yourself, in constant time if the comparison guards anything.
/// [`HttpRequestOptions`](crate::HttpRequestOptions), which holds
/// credentials, is not `PartialEq` either.
///
/// ```compile_fail
/// use allama_http::Credential;
///
/// let _ = Credential::api_key("a") == Credential::api_key("b");
/// ```
///
/// The secret is a plain `String`: it is not zeroed when dropped. Copies of
/// it already live in the request headers, the HTTP client and the TLS
/// buffers, which a `Credential` cannot reach, so zeroing only its own copy
/// would promise more than it delivers. Zeroing can be added later without
/// changing this API.
#[derive(Clone)]
pub struct Credential {
    kind: CredentialKind,
    secret: String,
}

impl Credential {
    /// The provider's own API key.
    ///
    /// The adapter places it as its API expects (a bearer token, an
    /// `x-api-key` header, ...), replacing the key it was configured with.
    #[must_use]
    pub fn api_key(secret: impl Into<String>) -> Self {
        Self {
            kind: CredentialKind::ApiKey,
            secret: secret.into(),
        }
    }

    /// An authentication header named `name`, whose whole value is
    /// `secret`.
    ///
    /// For gateway tokens and third-party adapters' own headers. A header
    /// credential replaces every header of the same name the adapter set,
    /// so it is sent once. Include any scheme in the secret: Cloudflare's
    /// gateway token is `Bearer <token>`.
    #[must_use]
    pub fn header(name: impl Into<String>, secret: impl Into<String>) -> Self {
        Self {
            kind: CredentialKind::Header(name.into()),
            secret: secret.into(),
        }
    }

    /// What kind of credential this is.
    #[must_use]
    pub fn kind(&self) -> &CredentialKind {
        &self.kind
    }

    /// The secret, for an adapter that places the credential itself
    /// ([`CredentialPlacement::Custom`]), for example to sign a request.
    ///
    /// Never log or record it.
    #[must_use]
    pub fn expose_secret(&self) -> &str {
        &self.secret
    }

    /// The error for a credential an adapter cannot place: an authored
    /// [`allama::Error::Configuration`] naming the kind and the `scope`,
    /// never the secret.
    ///
    /// An adapter that places its credential itself returns it for a kind
    /// it does not understand, addressed to its scope, instead of falling
    /// back to its configured key, which would bill the wrong account.
    #[must_use]
    pub fn unsupported(&self, scope: &str) -> allama::Error {
        allama::Error::configuration(format!(
            "a {} credential addressed to credential scope `{scope}` cannot be sent by this \
             provider",
            self.kind.name()
        ))
        .authored()
    }

    /// Whether this is the scope's own credential, which the adapter
    /// places, rather than a header it adds as given.
    pub(crate) fn is_placed_by_adapter(&self) -> bool {
        !matches!(self.kind, CredentialKind::Header(_))
    }
}

impl fmt::Debug for Credential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Credential")
            .field("kind", &self.kind)
            .field("secret", &format_args!("<redacted>"))
            .finish()
    }
}

/// Writes the kind, never the secret (see [Secrets stay in the
/// process](Credential#secrets-stay-in-the-process)).
impl Serialize for Credential {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match &self.kind {
            CredentialKind::ApiKey => {
                let mut state = serializer.serialize_struct("Credential", 1)?;
                state.serialize_field("kind", "api_key")?;
                state.end()
            }
            CredentialKind::Header(name) => {
                let mut state = serializer.serialize_struct("Credential", 2)?;
                state.serialize_field("kind", "header")?;
                state.serialize_field("name", name)?;
                state.end()
            }
        }
    }
}

/// Reads `{"kind": "api_key", "secret": ".."}` or
/// `{"kind": "header", "name": "..", "secret": ".."}`. A serialized
/// credential has no secret, so it does not deserialize back.
impl<'de> Deserialize<'de> for Credential {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(rename_all = "snake_case")]
        enum Kind {
            ApiKey,
            Header,
        }

        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            kind: Kind,
            #[serde(default)]
            name: Option<String>,
            #[serde(default)]
            secret: Option<String>,
        }

        let wire = Wire::deserialize(deserializer)?;
        let Some(secret) = wire.secret else {
            return Err(D::Error::custom(
                "a credential needs its `secret`; serialized credentials omit it",
            ));
        };
        match (wire.kind, wire.name) {
            (Kind::ApiKey, None) => Ok(Self::api_key(secret)),
            (Kind::ApiKey, Some(_)) => Err(D::Error::custom(
                "an `api_key` credential has no `name`; the adapter places it",
            )),
            (Kind::Header, Some(name)) => Ok(Self::header(name, secret)),
            (Kind::Header, None) => Err(D::Error::custom("a `header` credential needs a `name`")),
        }
    }
}

/// The kind of a [`Credential`].
///
/// More kinds may follow (a bearer token, request signing keys). An adapter
/// that places credentials itself refuses a kind it does not know with
/// [`Credential::unsupported`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CredentialKind {
    /// The provider's own API key ([`Credential::api_key`]).
    ApiKey,
    /// An authentication header with this name ([`Credential::header`]).
    Header(String),
}

impl CredentialKind {
    /// A short name for errors.
    fn name(&self) -> &'static str {
        match self {
            Self::ApiKey => "`api_key`",
            Self::Header(_) => "`header`",
        }
    }
}

/// Where an adapter takes its per-request credential from, and where it
/// puts it: the adapter's side of a [`Credential`].
///
/// Every HTTP adapter declares one and passes it to
/// [`HttpRequest::apply_options`], which then:
///
/// - places the [`Credential::api_key`] addressed to [`scope`](Self::scope)
///   as [`placement`](Self::placement) says, removing the configured
///   credential and every header in [`replaces`](Self::replaces);
/// - adds the [`Credential::header`] values addressed to the scope;
/// - ignores credentials addressed to any other scope;
/// - refuses plain headers that would set a credential (see
///   [`HttpRequest::apply_options`]).
///
/// The scope names an account, not a wire format: it is independent of the
/// provider name and the namespaces, so telemetry identity never changes
/// with it. Bundled adapters default it to their provider name and publish
/// that default as `DEFAULT_CREDENTIAL_SCOPE`; their builders change it with
/// `credential_scope`. Two accounts of one provider in a fallback chain need
/// two scopes, and providers behind one gateway share one.
///
/// ```
/// use allama_http::{CredentialPlacement, CredentialSlot};
///
/// // An API that takes its key in `x-nimbus-key`, and also accepts an
/// // `Authorization` header that a tenant key must replace.
/// let slot = CredentialSlot::new("nimbus", CredentialPlacement::Header("x-nimbus-key".into()))
///     .replaces("authorization");
/// assert_eq!(slot.header_name(), Some("x-nimbus-key"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct CredentialSlot {
    /// The credential scope: credentials addressed to it are this adapter's.
    pub scope: Cow<'static, str>,
    /// Where the scope's [`Credential::api_key`] goes.
    pub placement: CredentialPlacement,
    /// Configured authentication headers the credential replaces, besides
    /// the placement's own header: for example Entra ID's `authorization`
    /// on an endpoint whose keys go in `api-key`. Plain headers with these
    /// names are refused, like the placement's own header.
    pub replaces: Vec<Cow<'static, str>>,
    /// Whether every request must carry a credential for the scope: set
    /// when the adapter has no configured credential (bring your own key).
    /// A request without one fails with an authored
    /// [`allama::Error::Configuration`] before anything is sent.
    pub required: bool,
}

impl CredentialSlot {
    /// A slot for `scope`, placed as `placement`, replacing nothing else and
    /// not required.
    #[must_use]
    pub fn new(scope: impl Into<Cow<'static, str>>, placement: CredentialPlacement) -> Self {
        Self {
            scope: scope.into(),
            placement,
            replaces: Vec::new(),
            required: false,
        }
    }

    /// Also replace the configured header `name` (see
    /// [`replaces`](Self::replaces)).
    #[must_use]
    pub fn replaces(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        self.replaces.push(name.into());
        self
    }

    /// Require a credential for the scope on every request (see
    /// [`required`](Self::required)).
    #[must_use]
    pub fn required(mut self, required: bool) -> Self {
        self.required = required;
        self
    }

    /// The header the placement writes: `authorization` for
    /// [`Bearer`](CredentialPlacement::Bearer), the name for
    /// [`Header`](CredentialPlacement::Header), and `None` for
    /// [`Custom`](CredentialPlacement::Custom).
    #[must_use]
    pub fn header_name(&self) -> Option<&str> {
        match &self.placement {
            CredentialPlacement::Bearer => Some("authorization"),
            CredentialPlacement::Header(name) => Some(name),
            CredentialPlacement::Custom => None,
        }
    }

    /// Whether `name` is a header this slot owns: its placement's header or
    /// one it replaces.
    pub(crate) fn owns_header(&self, name: &str) -> bool {
        self.header_name()
            .is_some_and(|own| own.eq_ignore_ascii_case(name))
            || self
                .replaces
                .iter()
                .any(|replaced| replaced.eq_ignore_ascii_case(name))
    }
}

/// Where an adapter puts its scope's [`Credential::api_key`].
///
/// There is no query-string placement: with `http-tracing`, the request URL,
/// query included, is recorded on spans. An adapter whose API takes its key
/// in the query uses [`Custom`](Self::Custom) and owns the redaction.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CredentialPlacement {
    /// `Authorization: Bearer <secret>`.
    Bearer,
    /// `<name>: <secret>`, such as `x-api-key`.
    Header(Cow<'static, str>),
    /// The adapter places the credential itself, for example to sign the
    /// request. [`HttpRequest::apply_options`] leaves the scope's
    /// [`Credential::api_key`] alone; the adapter reads it with
    /// [`HttpRequestOptions::credential_for`](crate::HttpRequestOptions::credential_for)
    /// and applies it after `apply_options`, which adds the plain headers a
    /// signature covers. It refuses a kind it cannot place with
    /// [`Credential::unsupported`].
    Custom,
}

/// Returns `true` when `name` is a header that carries a credential by
/// definition, so a plain header with that name is refused (see
/// [`HttpRequest::apply_options`]).
///
/// The exact list, ignoring case: `authorization`, `proxy-authorization`,
/// `x-api-key`, `api-key`, `x-goog-api-key`, `cookie`,
/// `cf-aig-authorization`, `ocp-apim-subscription-key`,
/// `x-amz-security-token`. Send these as a [`Credential`].
///
/// Redaction is broader: [`is_sensitive_header`](crate::is_sensitive_header)
/// also matches names that merely contain `key`, `token`, `secret` or
/// `auth`, which are redacted but not refused, so `Idempotency-Key` passes.
///
/// ```
/// use allama_http::is_credential_header;
///
/// assert!(is_credential_header("Authorization"));
/// assert!(is_credential_header("cf-aig-authorization"));
/// assert!(!is_credential_header("Idempotency-Key"));
/// ```
#[must_use]
pub fn is_credential_header(name: &str) -> bool {
    CREDENTIAL_HEADERS
        .iter()
        .any(|credential| credential.eq_ignore_ascii_case(name))
}

/// The headers [`is_credential_header`] matches.
pub(crate) const CREDENTIAL_HEADERS: [&str; 9] = [
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "x-goog-api-key",
    "cookie",
    "cf-aig-authorization",
    "ocp-apim-subscription-key",
    "x-amz-security-token",
];

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn debug_and_serialization_never_show_the_secret() {
        for credential in [
            Credential::api_key("sk-SECRET"),
            Credential::header("cf-aig-authorization", "Bearer SECRET"),
        ] {
            let debug = format!("{credential:?}");
            assert!(!debug.contains("SECRET"), "{debug}");
            assert!(debug.contains("<redacted>"), "{debug}");
            let json = serde_json::to_string(&credential).unwrap();
            assert!(!json.contains("SECRET"), "{json}");
            let error = credential.unsupported("tenant").to_string();
            assert!(!error.contains("SECRET"), "{error}");
            assert!(error.contains("`tenant`"), "{error}");
        }
        assert_eq!(
            serde_json::to_value(Credential::header("x-gw", "SECRET")).unwrap(),
            json!({"kind": "header", "name": "x-gw"})
        );
    }

    #[test]
    fn deserialization_reads_the_secret_and_refuses_a_serialized_credential() {
        let key: Credential =
            serde_json::from_value(json!({"kind": "api_key", "secret": "k"})).unwrap();
        assert_eq!(key.kind(), &CredentialKind::ApiKey);
        assert_eq!(key.expose_secret(), "k");
        let header: Credential =
            serde_json::from_value(json!({"kind": "header", "name": "x-gw", "secret": "v"}))
                .unwrap();
        assert_eq!(header.kind(), &CredentialKind::Header("x-gw".into()));
        assert_eq!(header.expose_secret(), "v");

        let exported = serde_json::to_value(&key).unwrap();
        let err = serde_json::from_value::<Credential>(exported).unwrap_err();
        assert!(err.to_string().contains("`secret`"), "{err}");
        for invalid in [
            json!({"kind": "header", "secret": "v"}),
            json!({"kind": "api_key", "name": "x", "secret": "v"}),
            json!({"kind": "bearer", "secret": "v"}),
            json!({"kind": "api_key", "secret": "v", "scope": "s"}),
        ] {
            assert!(
                serde_json::from_value::<Credential>(invalid.clone()).is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn slots_own_their_placement_header_and_replaced_headers() {
        let bearer = CredentialSlot::new("openai", CredentialPlacement::Bearer);
        assert_eq!(bearer.header_name(), Some("authorization"));
        assert!(bearer.owns_header("Authorization"));
        assert!(!bearer.owns_header("api-key"));
        assert!(!bearer.required);

        let azure = CredentialSlot::new("azure", CredentialPlacement::Header("api-key".into()))
            .replaces("authorization")
            .required(true);
        assert!(azure.owns_header("API-KEY"));
        assert!(azure.owns_header("authorization"));
        assert!(azure.required);

        let custom = CredentialSlot::new("signed", CredentialPlacement::Custom);
        assert_eq!(custom.header_name(), None);
        assert!(!custom.owns_header("authorization"));
    }

    #[test]
    fn the_credential_list_is_exact() {
        for name in CREDENTIAL_HEADERS {
            assert!(is_credential_header(&name.to_ascii_uppercase()), "{name}");
            assert!(crate::is_sensitive_header(name), "{name}");
        }
        for name in [
            "idempotency-key",
            "x-nimbus-key",
            "x-gateway-token",
            "set-cookie",
        ] {
            assert!(!is_credential_header(name), "{name}");
        }
    }
}
