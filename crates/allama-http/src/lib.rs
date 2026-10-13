#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]
//! The HTTP transport seam of allama's HTTP provider crates.
//!
//! Provider crates (`allama-anthropic`, `allama-gemini`,
//! `allama-openai-compat`, `allama-openai`, `allama-systemone`) build an
//! [`HttpRequest`], send it through an [`HttpTransport`], and decode the [`HttpResponse`]. Their
//! request mapping, error classification and decoding never touch an HTTP
//! client directly, so the transport can be swapped:
//!
//! - [`ReqwestTransport`] (feature `reqwest`, default) is what providers use
//!   unless told otherwise.
//! - [`ReplayTransport`] serves canned responses in memory and records the
//!   requests: transport-level fixtures that exercise the whole adapter
//!   without a server, on any target.
//! - Custom transports cover other HTTP stacks, proxies, or recording.
//!
//! This is a seam for provider crates, not middleware for applications: it
//! carries bytes, and the portable `allama` types stay above it. Beside the
//! seam, it holds the helpers every provider shares, first-party or not,
//! starting with the HTTP error helpers:
//!
//! - [`status_error`] is what an error status means on its own, the
//!   fallback every provider refines from its error codes and body, so one
//!   status means the same on every provider.
//! - [`error_context`] reads an error response's status and headers
//!   (request id, retry hint, server retry directive) into an
//!   [`ErrorContext`](allama::ErrorContext).
//! - [`error_envelope`] reads the common JSON error envelopes and keeps the
//!   provider's text out of the redacted
//!   [`ErrorLog`](allama::ErrorLog) (see [`ErrorEnvelope::apply`]);
//!   [`error_envelope_value`] reads one already parsed.
//! - [`malformed_response`] is the error for a success response whose body
//!   cannot be decoded.
//! - [`Headers::retry_after`] reads retry hints.
//! - [`TransportError::into_error`] maps a request that got no response,
//!   and [`TransportError::into_stream_error`] a streamed response whose
//!   body could not be read to the end.
//!
//! And the streaming helpers of every streaming adapter. A framer turns a
//! response body into frames, in linear time and under a size cap (the
//! provider builder's `max_stream_event_size`):
//! [`SseStream`](sse::SseStream) into SSE events, in [`sse`], and
//! [`NdjsonStream`](ndjson::NdjsonStream) into NDJSON lines, in [`ndjson`].
//! Whatever the framing, [`FramedChatStream`](framing::FramedChatStream)
//! drives an adapter's [`ChatFrameDecoder`](framing::ChatFrameDecoder) over
//! the frames ([`SseChatStream`](sse::SseChatStream) is its SSE form), and
//! [`VendorFrames`](framing::VendorFrames) surfaces the frames an adapter
//! does not model, in [`framing`].
//!
//! It also defines the caller's per-request transport options,
//! [`HttpRequestOptions`] (headers and [`Credential`]s, under the shared
//! [`OPTIONS_NAMESPACE`]), which every adapter applies with
//! [`HttpRequest::apply_options`] and its own [`CredentialSlot`]: a
//! credential is addressed to a credential scope, and only the adapter with
//! that scope sends it.
//!
//! # Trace context propagation
//!
//! `allama-http` sends no trace context of its own, and depends on no
//! OpenTelemetry crate. To continue the application's trace in a gateway
//! that joins distributed traces (LiteLLM proxy, vLLM, an instrumented
//! internal gateway), wrap the transport in an [`HttpTransport`] that
//! injects the current span's W3C `traceparent` into
//! [`HttpRequest::headers`] before it sends, with the application's own
//! OpenTelemetry version. Adapters call [`HttpRequest::apply_options`]
//! before [`send`](HttpTransport::send), so the wrapper sees the final
//! headers and can keep a `traceparent` the caller set. The
//! [observability guide]
//! has the recipe, for provider builders and for the provider registry.
//!
//! # Timeouts
//!
//! Every layer bounds a different phase, and each one can be turned off:
//!
//! | Bound | Where | Default | Disable |
//! | --- | --- | --- | --- |
//! | Connect (TCP + TLS) | [`ReqwestTransport::new`] | [`DEFAULT_CONNECT_TIMEOUT`] (not on `wasm32`) | pass your own client to [`ReqwestTransport::from_client`] |
//! | Whole non-streaming exchange | [`HttpRequest::timeout`], set by the provider | [`DEFAULT_REQUEST_TIMEOUT`] in the bundled crates | the provider builder's `timeout` |
//! | Stream headers, gap between stream chunks | [`StreamTimeouts`], applied by the provider | [`StreamTimeouts::default`] | the provider builder's `stream_timeouts` |
//!
//! `allama::wrappers::TimeoutProvider` sits above all of them and bounds
//! decoded calls and stream events, with the same `Error::Timeout` shape.
//! Whichever bound is shorter fires first, so use `TimeoutProvider` for
//! tight, workload-specific limits and leave these defaults as the
//! dead-connection safety net.
//!
//! # Secrets
//!
//! The `Debug` output of [`Headers`] (and so of [`HttpRequest`] and
//! [`HttpResponse`]) redacts credential headers (see
//! [`is_sensitive_header`]) and every credential
//! [`HttpRequest::apply_options`] placed. Transport errors drop the URL's
//! user info, query and fragment, and the bundled providers' `Debug`
//! output shows base URLs through [`config::redact_url`].
//!
//! # Network defaults
//!
//! What a broken or hostile endpoint (a tenant-configured gateway, a
//! compromised proxy) can do to the caller is bounded:
//!
//! - **No redirects.** [`ReqwestTransport::new`] never follows one, and the
//!   bundled providers report a `3xx` as an
//!   [`UnexpectedResponse`](allama::Error::UnexpectedResponse) naming the
//!   target host ([`redirect_error`]): following it would send the key and
//!   the prompt to a host the application never configured. A client you
//!   pass to [`ReqwestTransport::from_client`] keeps its own redirect
//!   policy; on `wasm32`, the host `fetch`'s applies.
//! - **Bounded bodies.** Success bodies are read with
//!   [`HttpResponse::bytes_limited`] under the builder's
//!   `max_response_size` ([`DEFAULT_MAX_RESPONSE_SIZE`]), error bodies with
//!   [`HttpResponse::map_error`] (at most [`MAX_ERROR_BODY_SIZE`], within
//!   [`ERROR_BODY_TIMEOUT`]), and stream events under
//!   `max_stream_event_size`. A whole stream has no byte cap: its length is
//!   the answer's, which the request bounds with `max_output_tokens`, and its
//!   duration is bounded by the stream timeouts ([`StreamTimeouts`], and
//!   `TimeoutProvider::stream_total` in allama). A byte cap would cut off a
//!   long answer that is legitimate, mid-stream. To bound what one call may
//!   cost, count its usage, or its bytes, in a wrapper around the stream.
//! - **Checked configuration.** [`config`] refuses malformed base URLs,
//!   keys no header can carry, and credential headers among a builder's
//!   static headers, at `build()`.
//!
//! # Versioning
//!
//! `allama-http` names these third-party crates in its public API, so
//! moving any of them to a new major (minor, before 1.0) version is a
//! breaking release of `allama-http`:
//!
//! - `bytes` 1: [`Bytes`], re-exported, carries request and response
//!   bodies;
//! - `futures-core` 0.3: the [`Body`] stream and the framers' stream
//!   bounds;
//! - `serde` and `serde_json` 1: [`HttpRequestOptions`]' JSON form,
//!   [`ReplayResponse::json_value`] and the JSON the error and framing
//!   helpers read;
//! - `reqwest` 0.13 (feature `reqwest`): [`ReqwestTransport::from_client`];
//! - `reqwest-middleware` 0.5 (feature `reqwest-middleware`):
//!   `ReqwestTransport::from_middleware`.
//!
//! The provider crates take a client only as an [`HttpTransport`]
//! (`http_transport(ReqwestTransport::from_client(..))`), so `reqwest`
//! appears in no other public signature.
//!
//! The allama crates are released together, at one version, for now. So a
//! breaking release of `allama-http` is today a breaking release of every
//! allama crate. That is how 0.1 is released, not a promise about every
//! later release. The reason is [`HttpRequestOptions`]: options are found
//! by their type, so with `allama-http` at two versions over one `allama`,
//! an adapter built on one version reads a different type than the one
//! attached with the other. The call fails with
//! [`Error::InvalidRequest`](allama::Error::InvalidRequest), naming the
//! `allama-http` namespace and both types, instead of sending the
//! configured key without the tenant's credential and headers; with one
//! version for every crate, the same mix is a compile error.
//!
//! ```
//! use allama_http::{HttpRequest, HttpTransport, ReplayResponse, ReplayTransport};
//!
//! # async fn example() -> Result<(), allama_http::TransportError> {
//! let transport = ReplayTransport::new([ReplayResponse::json(200, r#"{"ok":true}"#)]);
//! let response = transport
//!     .send(HttpRequest::post("https://api.example.com/v1/chat", "{}"))
//!     .await?;
//! assert_eq!(response.status, 200);
//! assert_eq!(transport.requests()[0].url, "https://api.example.com/v1/chat");
//! # Ok(()) }
//! ```
//!
#![doc = concat!(
    "[observability guide]: https://github.com/sagikazarmark/allama/blob/v",
    env!("CARGO_PKG_VERSION"),
    "/docs/guides/observability/README.md#trace-context-propagation",
)]
#![cfg_attr(
    not(feature = "reqwest"),
    doc = "[`ReqwestTransport`]: crate\n[`ReqwestTransport::new`]: crate\n[`DEFAULT_CONNECT_TIMEOUT`]: crate\n[`ReqwestTransport::from_client`]: crate"
)]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub use bytes::Bytes;
use futures_core::Stream;
use futures_util::StreamExt;

pub mod config;
mod credential;
mod error;
pub mod framing;
pub mod ndjson;
mod options;
mod replay;
#[cfg(feature = "reqwest")]
mod reqwest;
mod retry_after;
pub mod sse;
mod timeout;

pub use credential::{
    Credential, CredentialKind, CredentialPlacement, CredentialSlot, is_credential_header,
};
pub use error::{
    ErrorEnvelope, error_context, error_envelope, error_envelope_value, malformed_response,
    status_error,
};
pub use options::{HttpRequestOptions, OPTIONS_NAMESPACE};
pub use replay::{ReplayResponse, ReplayTransport};
#[cfg(feature = "reqwest")]
pub use reqwest::{DEFAULT_CONNECT_TIMEOUT, ReqwestTransport};
pub use retry_after::parse_retry_after;
pub use timeout::StreamTimeouts;

/// A boxed future returned by [`HttpTransport::send`].
pub type SendFuture<'a> =
    Pin<Box<dyn Future<Output = Result<HttpResponse, TransportError>> + Send + 'a>>;

/// A response body: a stream of byte chunks.
pub type Body = Pin<Box<dyn Stream<Item = Result<Bytes, TransportError>> + Send>>;

/// Sends HTTP requests for a provider.
///
/// Implementations return the response as soon as its status and headers are
/// known; the body streams afterwards (SSE responses are read incrementally).
/// A response with an error status is still `Ok`: classifying HTTP errors is
/// the provider's job. `Err` means no response arrived.
pub trait HttpTransport: Send + Sync + 'static {
    /// Send `request`.
    fn send(&self, request: HttpRequest) -> SendFuture<'_>;
}

impl<T: HttpTransport + ?Sized> HttpTransport for Arc<T> {
    fn send(&self, request: HttpRequest) -> SendFuture<'_> {
        T::send(self, request)
    }
}

impl<T: HttpTransport + ?Sized> HttpTransport for Box<T> {
    fn send(&self, request: HttpRequest) -> SendFuture<'_> {
        T::send(self, request)
    }
}

/// HTTP method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Method {
    /// `GET`.
    Get,
    /// `POST`.
    Post,
}

impl Method {
    /// The method name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
        }
    }
}

/// An HTTP request built by a provider.
///
/// `Debug` output redacts credential headers (see [`Headers`]).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct HttpRequest {
    /// Method.
    pub method: Method,
    /// Absolute URL.
    pub url: String,
    /// Headers, in order.
    pub headers: Headers,
    /// Body, if any.
    pub body: Option<Bytes>,
    /// Bound on the whole exchange, when the provider sets one. Streaming
    /// requests usually leave it unset.
    pub timeout: Option<Duration>,
}

impl HttpRequest {
    /// A `GET` request.
    #[must_use]
    pub fn get(url: impl Into<String>) -> Self {
        Self {
            method: Method::Get,
            url: url.into(),
            headers: Headers::new(),
            body: None,
            timeout: None,
        }
    }

    /// A `POST` request with `body`.
    #[must_use]
    pub fn post(url: impl Into<String>, body: impl Into<Bytes>) -> Self {
        Self {
            method: Method::Post,
            url: url.into(),
            headers: Headers::new(),
            body: Some(body.into()),
            timeout: None,
        }
    }

    /// Add a header.
    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name, value);
        self
    }

    /// Set (or clear) the timeout.
    #[must_use]
    pub fn timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = timeout;
        self
    }

    /// The body as UTF-8 text, if it is valid UTF-8.
    #[must_use]
    pub fn body_text(&self) -> Option<&str> {
        std::str::from_utf8(self.body.as_deref()?).ok()
    }

    /// Add a header whose value is a secret: `Debug` output redacts it
    /// whatever its name. For an adapter that places a credential itself
    /// ([`CredentialPlacement::Custom`]).
    #[must_use]
    pub fn sensitive_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert_sensitive(name, value);
        self
    }

    /// Set each of `headers`, replacing the request's headers of the same
    /// name (ignoring case) and adding the others, in order. For the
    /// headers a provider sends with every request: the bundled providers
    /// set their [`DEFAULT_USER_AGENT`] and the builder's own headers
    /// (`header(name, value)`, validated by [`config::static_headers`])
    /// with it, after their protocol headers and before
    /// [`apply_options`](Self::apply_options), so a request's own headers
    /// still win.
    ///
    /// ```
    /// use allama_http::{Headers, HttpRequest};
    ///
    /// let defaults: Headers = [("user-agent", "my-app/1.0"), ("x-route", "eu")].into_iter().collect();
    /// let request = HttpRequest::post("https://api.example.com/v1/chat", "{}")
    ///     .header("User-Agent", "allama/0.1.0")
    ///     .with_headers(&defaults);
    /// assert_eq!(request.headers.get("user-agent"), Some("my-app/1.0"));
    /// assert_eq!(request.headers.iter().count(), 2);
    /// ```
    #[must_use]
    pub fn with_headers(mut self, headers: &Headers) -> Self {
        for (name, _) in headers.iter() {
            self.headers.remove(name);
        }
        for header in &headers.0 {
            self.headers.0.push(header.clone());
        }
        self
    }

    /// Apply the caller's transport options for this request: the
    /// [`HttpRequestOptions`] in `options`, typed or in their JSON form,
    /// with the credentials addressed to `slot`'s scope.
    ///
    /// Call it once the request carries every header the adapter sets,
    /// including its configured credential, and before sending it. Every
    /// bundled HTTP adapter calls it with the options of the request it
    /// serves and its own [`CredentialSlot`], so callers get the same
    /// behaviour from first- and third-party adapters. In order, it:
    ///
    /// 1. refuses, before anything is sent, a plain header that would carry
    ///    a credential (see Errors);
    /// 2. replaces the request's headers with the plain headers of the same
    ///    name, and adds the others;
    /// 3. places the [`Credential::api_key`] addressed to the slot's scope
    ///    as the slot's [placement](CredentialSlot::placement) says,
    ///    removing the configured credential and every header the slot
    ///    [replaces](CredentialSlot::replaces). A
    ///    [`Custom`](CredentialPlacement::Custom) slot is left alone: the
    ///    adapter signs or places its credential after this call;
    /// 4. adds the [`Credential::header`] values addressed to the scope,
    ///    each replacing the headers of the same name.
    ///
    /// Credentials addressed to other scopes are never sent. Placed
    /// credentials are redacted from `Debug` whatever the header name.
    ///
    /// ```
    /// use allama::ChatRequest;
    /// use allama_http::{Credential, CredentialPlacement, CredentialSlot, HttpRequest, HttpRequestOptions};
    ///
    /// let slot = CredentialSlot::new("anthropic", CredentialPlacement::Header("x-api-key".into()));
    /// let request = ChatRequest::new("m").user("Hi").with_option(
    ///     HttpRequestOptions::new()
    ///         .credential("anthropic", Credential::api_key("tenant-key"))
    ///         .credential("openai", Credential::api_key("other-account"))
    ///         .header("x-tenant-id", "acme"),
    /// );
    /// let http = HttpRequest::post("https://api.example.com/v1/messages", "{}")
    ///     .header("x-api-key", "provider-key")
    ///     .apply_options(&request.options, &slot)?;
    /// assert_eq!(http.headers.get("x-api-key"), Some("tenant-key"));
    /// assert_eq!(http.headers.get("x-tenant-id"), Some("acme"));
    /// assert!(!format!("{http:?}").contains("other-account"));
    /// # Ok::<(), allama::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - [`allama::Error::InvalidRequest`] when the JSON form of the options
    ///   is malformed, names a header on the exact credential list
    ///   ([`is_credential_header`]), or sets credentials.
    /// - An authored [`allama::Error::Configuration`], before anything is
    ///   sent, when a plain header is named on the exact credential list
    ///   ([`is_credential_header`]) or is one the slot writes or replaces,
    ///   or when the slot is [required](CredentialSlot::required) and no
    ///   credential is addressed to its scope. These are the application's
    ///   bugs, not the end user's, so they are not retried, never fall over
    ///   to another provider, and map to a server error. Error text names
    ///   headers and scopes, never secrets.
    pub fn apply_options(
        mut self,
        options: &allama::RequestOptions,
        slot: &CredentialSlot,
    ) -> allama::Result<Self> {
        let transport = options.resolve::<HttpRequestOptions>()?;
        let transport = transport.as_deref();
        if let Some(transport) = transport {
            refuse_credential_headers(transport, slot)?;
        }
        let credential = transport.and_then(|transport| transport.credential_for(&slot.scope));
        if slot.required && credential.is_none() {
            return Err(allama::Error::configuration(format!(
                "no credential is addressed to credential scope `{}`, and the provider has no \
                 configured credential: attach one with `HttpRequestOptions::credential`",
                slot.scope
            ))
            .authored());
        }
        let Some(transport) = transport else {
            return Ok(self);
        };
        for (name, _) in transport.headers.iter() {
            self.headers.remove(name);
        }
        for (name, value) in transport.headers.iter() {
            self.headers.insert(name, value);
        }
        if let Some(credential) = credential {
            self.place(credential, slot)?;
        }
        for (name, secret) in transport.header_credentials_for(&slot.scope) {
            self.headers.remove(name);
            self.headers.insert_sensitive(name, secret);
        }
        Ok(self)
    }

    /// Place the scope's own `credential` as `slot` says.
    fn place(&mut self, credential: &Credential, slot: &CredentialSlot) -> allama::Result<()> {
        let (name, value) = match (&slot.placement, credential.kind()) {
            (CredentialPlacement::Custom, _) => return Ok(()),
            (CredentialPlacement::Bearer, CredentialKind::ApiKey) => (
                "authorization",
                format!("Bearer {}", credential.expose_secret()),
            ),
            (CredentialPlacement::Header(name), CredentialKind::ApiKey) => {
                (name.as_ref(), credential.expose_secret().to_owned())
            }
            _ => return Err(credential.unsupported(&slot.scope)),
        };
        self.headers.remove(name);
        for replaced in &slot.replaces {
            self.headers.remove(replaced);
        }
        self.headers.insert_sensitive(name, value);
        Ok(())
    }
}

/// Refuse the plain headers in `transport` that would carry a credential:
/// those on the exact credential list, and those `slot` writes or replaces.
fn refuse_credential_headers(
    transport: &HttpRequestOptions,
    slot: &CredentialSlot,
) -> allama::Result<()> {
    for (name, _) in transport.headers.iter() {
        if is_credential_header(name) {
            return Err(allama::Error::configuration(format!(
                "the plain header `{name}` carries a credential: send it with \
                 `HttpRequestOptions::credential`, addressed to a credential scope"
            ))
            .authored());
        }
        if slot.owns_header(name) {
            return Err(allama::Error::configuration(format!(
                "the plain header `{name}` is where credential scope `{}` puts its credential: \
                 send it with `HttpRequestOptions::credential`",
                slot.scope
            ))
            .authored());
        }
    }
    Ok(())
}

/// HTTP headers, matched case-insensitively and kept in order.
///
/// `Debug` output redacts the values of credential headers (see
/// [`is_sensitive_header`]), so requests and responses can be logged.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct Headers(Vec<Header>);

impl Headers {
    /// No headers.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a header.
    pub fn insert(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.0.push(Header {
            name: name.into(),
            value: value.into(),
            sensitive: false,
        });
    }

    /// Append a header whose value is a secret, which `Debug` output
    /// redacts whatever the header's name.
    pub fn insert_sensitive(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.0.push(Header {
            name: name.into(),
            value: value.into(),
            sensitive: true,
        });
    }

    /// Remove every value of `name`, ignoring case.
    pub fn remove(&mut self, name: &str) {
        self.0
            .retain(|header| !header.name.eq_ignore_ascii_case(name));
    }

    /// The first value of `name`, ignoring case.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&str> {
        self.0
            .iter()
            .find(|header| header.name.eq_ignore_ascii_case(name))
            .map(|header| header.value.as_str())
    }

    /// Iterate over `(name, value)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.0
            .iter()
            .map(|header| (header.name.as_str(), header.value.as_str()))
    }

    /// How long the server asks the client to wait before retrying: the
    /// `name` header (usually `retry-after`) or `retry-after-ms`, parsed
    /// with [`parse_retry_after`] relative to the current time.
    ///
    /// Unlike passing `SystemTime::now()` yourself, this also works on
    /// `wasm32-unknown-unknown`.
    #[must_use]
    pub fn retry_after(&self, name: &str) -> Option<Duration> {
        parse_retry_after(self.get(name), self.get("retry-after-ms"), now())
    }
}

impl fmt::Debug for Headers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.0.iter().map(|header| (&header.name, header)))
            .finish()
    }
}

/// One header of [`Headers`].
#[derive(Clone, PartialEq, Eq)]
struct Header {
    name: String,
    value: String,
    /// Set for a placed credential: redacted whatever the name.
    sensitive: bool,
}

impl Header {
    /// Whether `Debug` output redacts the value.
    fn is_sensitive(&self) -> bool {
        self.sensitive || is_sensitive_header(&self.name)
    }
}

/// The value as `Debug` shows it: redacted when the header is sensitive.
impl fmt::Debug for Header {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_sensitive() {
            f.write_str("<redacted>")
        } else {
            fmt::Debug::fmt(&self.value, f)
        }
    }
}

/// Returns `true` when a header named `name` likely carries a credential,
/// so its value must not be logged.
///
/// Matches, ignoring case, the exact credential list
/// ([`is_credential_header`]), `set-cookie`, and any name containing `key`,
/// `token`, `secret` or `auth`. Erring towards redaction is deliberate: a
/// rate-limit header hidden from a debug log costs less than a leaked key.
/// This rule only redacts: a plain header it matches but the exact list
/// does not, such as `Idempotency-Key`, is still sent.
///
/// ```
/// use allama_http::is_sensitive_header;
///
/// assert!(is_sensitive_header("Authorization"));
/// assert!(is_sensitive_header("x-my-gateway-token"));
/// assert!(!is_sensitive_header("x-request-id"));
/// ```
#[must_use]
pub fn is_sensitive_header(name: &str) -> bool {
    const FRAGMENTS: [&str; 4] = ["key", "token", "secret", "auth"];
    let name = name.to_ascii_lowercase();
    is_credential_header(&name)
        || name == "set-cookie"
        || FRAGMENTS.iter().any(|part| name.contains(part))
}

/// The current time as a `std` `SystemTime`, read through `web-time`:
/// `SystemTime::now` panics on `wasm32-unknown-unknown`.
fn now() -> SystemTime {
    let since_epoch = web_time::SystemTime::now()
        .duration_since(web_time::UNIX_EPOCH)
        .unwrap_or_default();
    UNIX_EPOCH + since_epoch
}

impl<K: Into<String>, V: Into<String>> FromIterator<(K, V)> for Headers {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut headers = Self::new();
        for (name, value) in iter {
            headers.insert(name, value);
        }
        headers
    }
}

/// An HTTP response: status and headers, then a streamed body.
#[non_exhaustive]
pub struct HttpResponse {
    /// Status code.
    pub status: u16,
    /// Headers.
    pub headers: Headers,
    body: Body,
}

impl fmt::Debug for HttpResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpResponse")
            .field("status", &self.status)
            .field("headers", &self.headers)
            .finish_non_exhaustive()
    }
}

impl HttpResponse {
    /// A response with a streamed body.
    #[must_use]
    pub fn new(status: u16, headers: Headers, body: Body) -> Self {
        Self {
            status,
            headers,
            body,
        }
    }

    /// A response with the whole body at once.
    #[must_use]
    pub fn from_bytes(status: u16, headers: Headers, body: impl Into<Bytes>) -> Self {
        let body = body.into();
        Self::new(
            status,
            headers,
            Box::pin(futures_util::stream::once(async move { Ok(body) })),
        )
    }

    /// Returns `true` for 2xx statuses.
    #[must_use]
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// Read the whole body, however large.
    ///
    /// For a response from a server you do not control, read with
    /// [`bytes_limited`](Self::bytes_limited) instead: a broken or hostile
    /// server can send a body that does not fit in memory.
    ///
    /// # Errors
    ///
    /// The transport failed while the body was read.
    pub async fn bytes(self) -> Result<Bytes, TransportError> {
        self.bytes_limited(usize::MAX).await
    }

    /// Read the whole body as text (invalid UTF-8 is replaced), however
    /// large. See [`bytes`](Self::bytes); prefer
    /// [`text_limited`](Self::text_limited).
    ///
    /// # Errors
    ///
    /// The transport failed while the body was read.
    pub async fn text(self) -> Result<String, TransportError> {
        self.text_limited(usize::MAX).await
    }

    /// Read the whole body, failing as soon as it passes `limit` bytes. The
    /// bundled providers read success bodies with it, under the builder's
    /// `max_response_size` (default [`DEFAULT_MAX_RESPONSE_SIZE`]).
    ///
    /// The buffer never grows past `limit`: the chunk that would take it
    /// past is refused before it is copied, and the rest of the body is
    /// never read.
    ///
    /// ```
    /// use allama_http::{Headers, HttpResponse, TransportErrorKind};
    ///
    /// # futures_util::FutureExt::now_or_never(async {
    /// let response = HttpResponse::from_bytes(200, Headers::new(), "0123456789");
    /// let err = response.bytes_limited(4).await.unwrap_err();
    /// assert_eq!(err.kind(), TransportErrorKind::TooLarge);
    /// assert!(matches!(err.into_error(), allama::Error::UnexpectedResponse { .. }));
    /// # }).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// - A [`TransportErrorKind::TooLarge`] error once the body passes
    ///   `limit`, with an [authored](TransportError::authored) message. It
    ///   maps to [`allama::Error::UnexpectedResponse`], which is not
    ///   retryable: asking again returns the same body.
    /// - The transport's error when reading the body failed.
    pub async fn bytes_limited(self, limit: usize) -> Result<Bytes, TransportError> {
        let mut body = self.body;
        let mut buffer = Vec::new();
        while let Some(chunk) = body.next().await {
            let chunk = chunk?;
            if chunk.len() > limit.saturating_sub(buffer.len()) {
                return Err(TransportError::new(
                    TransportErrorKind::TooLarge,
                    format!("the response body is larger than the limit of {limit} bytes"),
                )
                .authored());
            }
            if buffer.is_empty() && body_is_done(&body) {
                return Ok(chunk);
            }
            buffer.extend_from_slice(&chunk);
        }
        Ok(buffer.into())
    }

    /// Read the whole body as text (invalid UTF-8 is replaced), failing as
    /// soon as it passes `limit` bytes, as
    /// [`bytes_limited`](Self::bytes_limited) does.
    ///
    /// # Errors
    ///
    /// See [`bytes_limited`](Self::bytes_limited).
    pub async fn text_limited(self, limit: usize) -> Result<String, TransportError> {
        let bytes = self.bytes_limited(limit).await?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// The error for an error response (one that is not a success): reads
    /// the start of its body for diagnostics and maps the status, headers
    /// and body text with `map`, the adapter's own classification.
    ///
    /// Every bundled HTTP provider maps its error responses here, so these
    /// rules hold on all of them:
    ///
    /// - **The body read is bounded.** At most [`MAX_ERROR_BODY_SIZE`]
    ///   bytes are read, within [`ERROR_BODY_TIMEOUT`]; the rest of the
    ///   body is never read. An error keeps only the first
    ///   [`ErrorContext::MAX_BODY_BYTES`](allama::ErrorContext::MAX_BODY_BYTES)
    ///   of it anyway, and a streaming request has no total deadline, so an
    ///   unbounded read of a body that never ends would hang the call and
    ///   grow without limit.
    /// - **A failed read is recorded, not hidden.** `map` gets the text read
    ///   before the failure, and the failure (a transport error, or the time
    ///   bound) is added to the error's
    ///   [upstream message](allama::Error::upstream_message).
    /// - **A redirect is never followed.** A `3xx` is the
    ///   [`redirect_error`] for the response, and `map` is not called: an
    ///   LLM API never answers a request with a redirect, and following one
    ///   would send the request, credentials included, somewhere the
    ///   application never configured.
    ///
    /// ```
    /// use allama::Error;
    /// use allama_http::{HttpResponse, Headers, error_context, status_error};
    ///
    /// # futures_util::FutureExt::now_or_never(async {
    /// let map = |status: u16, headers: &Headers, body: &str| {
    ///     status_error(status, format!("HTTP {status}"))
    ///         .with_context(error_context(status, headers))
    ///         .authored()
    ///         .with_body(body.to_owned())
    /// };
    ///
    /// let response = HttpResponse::from_bytes(429, Headers::new(), "slow down");
    /// let err = response.map_error(map).await;
    /// assert!(matches!(err, Error::RateLimited { .. }));
    /// assert_eq!(err.context().and_then(|c| c.capped_body()), Some("slow down"));
    ///
    /// let headers = [("location", "https://elsewhere.example/v1/chat")].into_iter().collect();
    /// let redirect = HttpResponse::from_bytes(307, headers, "");
    /// let err = redirect.map_error(map).await;
    /// assert!(matches!(err, Error::UnexpectedResponse { .. }));
    /// assert!(err.to_string().contains("elsewhere.example"));
    /// # }).unwrap();
    /// ```
    pub async fn map_error(
        self,
        map: impl FnOnce(u16, &Headers, &str) -> allama::Error,
    ) -> allama::Error {
        if (300..400).contains(&self.status) {
            return redirect_error(self.status, &self.headers);
        }
        let (body, failure) =
            read_error_body(self.body, MAX_ERROR_BODY_SIZE, ERROR_BODY_TIMEOUT).await;
        let error = map(self.status, &self.headers, &String::from_utf8_lossy(&body));
        let Some(failure) = failure else {
            return error;
        };
        let note = format!("reading the error body failed: {failure}");
        let message = match error.upstream_message() {
            Some(message) => format!("{message} ({note})"),
            None => note,
        };
        error.with_upstream_message(message)
    }

    /// The body as a stream of chunks (for SSE).
    #[must_use]
    pub fn into_body(self) -> Body {
        self.body
    }
}

/// The bound on a whole non-streaming call that the bundled providers set
/// on every unary request ([`HttpRequest::timeout`]) unless the builder's
/// `timeout` is set: 120 s, whatever the transport. Only `timeout(None)`
/// removes it.
///
/// An adapter of your own sets it too: a transport adds no total deadline
/// of its own, so without it a server that accepts the connection and never
/// answers holds a unary call forever. Streaming requests take
/// [`StreamTimeouts`] instead.
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// The default limit on a success response's body that the bundled
/// providers read whole (every non-streaming call): 256 MiB. Change it with
/// the provider builder's `max_response_size`.
///
/// It is a bound against a broken or hostile server, not a size any API
/// sends: a chat response is kilobytes, but a full batch of large embeddings
/// in JSON (2,048 inputs of 3,072 dimensions) is over 128 MiB. Streamed
/// responses are bounded per event instead (`max_stream_event_size`).
pub const DEFAULT_MAX_RESPONSE_SIZE: usize = 256 * 1024 * 1024;

/// The most of an error response's body [`HttpResponse::map_error`] reads:
/// 64 KiB. An error keeps only
/// [`ErrorContext::MAX_BODY_BYTES`](allama::ErrorContext::MAX_BODY_BYTES)
/// of it.
pub const MAX_ERROR_BODY_SIZE: usize = 64 * 1024;

/// How long [`HttpResponse::map_error`] reads an error response's body:
/// 30 s. A streaming request has no total deadline, only bounds on the wait
/// for each chunk, so a body that keeps trickling would otherwise hold the
/// call forever.
pub const ERROR_BODY_TIMEOUT: Duration = Duration::from_secs(30);

/// The `User-Agent` the bundled providers send unless the builder's
/// `header("user-agent", ..)` replaces it: `allama/<version>`.
pub const DEFAULT_USER_AGENT: &str = concat!("allama/", env!("CARGO_PKG_VERSION"));

/// The error for a redirect (`3xx`) response, which the bundled providers
/// never follow: an [`allama::Error::UnexpectedResponse`] with the status,
/// the response's [`error_context`], and an
/// [authored](allama::Error::authored) message naming the host the
/// `Location` header points to (never its path or query, which may carry
/// tokens). It is not retryable: the server answers the same way again.
///
/// [`ReqwestTransport::new`] does not follow redirects, so the bundled
/// providers see the redirect itself and map it here, through
/// [`HttpResponse::map_error`].
///
/// ```
/// use allama::Error;
/// use allama_http::{Headers, redirect_error};
///
/// let headers: Headers = [("location", "http://169.254.169.254/latest?token=x")].into_iter().collect();
/// let err = redirect_error(307, &headers);
/// assert!(matches!(err, Error::UnexpectedResponse { .. }));
/// assert_eq!(err.status(), Some(307));
/// assert!(!err.is_retryable());
/// assert_eq!(
///     err.to_string(),
///     "unexpected response: HTTP 307: the server redirected the request to `169.254.169.254`; \
///      redirects are not followed",
/// );
/// ```
#[cfg_attr(not(feature = "reqwest"), doc = "[`ReqwestTransport::new`]: crate")]
#[must_use]
pub fn redirect_error(status: u16, headers: &Headers) -> allama::Error {
    let target = match headers.get("location").map(config::url_host) {
        Some(Some(host)) => format!("to `{host}`"),
        Some(None) => "within the same host".to_owned(),
        None => "without a `Location`".to_owned(),
    };
    allama::Error::unexpected_response(format!(
        "HTTP {status}: the server redirected the request {target}; redirects are not followed"
    ))
    .with_context(error_context(status, headers))
    .authored()
}

/// Whether a body stream is known to have ended, so a single chunk can be
/// returned without a copy.
fn body_is_done(body: &Body) -> bool {
    body.size_hint() == (0, Some(0))
}

/// Read at most `limit` bytes of an error body within `timeout`: what was
/// read, and why reading stopped early when it failed.
async fn read_error_body(
    mut body: Body,
    limit: usize,
    timeout: Duration,
) -> (Vec<u8>, Option<TransportError>) {
    let mut buffer = Vec::new();
    let failure = {
        let read = std::pin::pin!(async {
            while let Some(chunk) = body.next().await {
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(error) => return Some(error),
                };
                let room = limit - buffer.len();
                buffer.extend_from_slice(&chunk[..chunk.len().min(room)]);
                if buffer.len() == limit {
                    break;
                }
            }
            None
        });
        let deadline = futures_timer::Delay::new(timeout);
        match futures_util::future::select(read, deadline).await {
            futures_util::future::Either::Left((failure, _)) => failure,
            futures_util::future::Either::Right(((), _)) => Some(
                TransportError::new(
                    TransportErrorKind::Timeout,
                    format!("the error body did not end within {timeout:?}"),
                )
                .authored(),
            ),
        }
    };
    (buffer, failure)
}

/// Why a request got no (complete) response.
#[derive(Debug)]
pub struct TransportError {
    kind: TransportErrorKind,
    message: String,
    authored: bool,
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

/// Kinds of [`TransportError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TransportErrorKind {
    /// The request or the body read timed out.
    Timeout,
    /// The request could not be built (invalid URL or header); nothing was
    /// sent.
    InvalidRequest,
    /// No response was received: connection refused, DNS or TLS failure,
    /// connection reset.
    ///
    /// The request may or may not have reached the provider: a connection
    /// reset after the request was written cannot be told apart from one
    /// that was never sent. Do not assume it was not processed or billed.
    Connect,
    /// The response started, but reading its body failed.
    Body,
    /// The response body was larger than the reader's limit
    /// ([`HttpResponse::bytes_limited`]). Unlike the other kinds, the
    /// response arrived whole enough to tell: it maps to
    /// [`allama::Error::UnexpectedResponse`], which is not retryable.
    TooLarge,
    /// Any other failure.
    Other,
}

impl TransportError {
    /// Create an error.
    #[must_use]
    pub fn new(kind: TransportErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            authored: false,
            source: None,
        }
    }

    /// Mark the message as written by the transport's own code, with no
    /// text from a foreign error, the request, or the response in it, so
    /// [`into_error`](Self::into_error) keeps it as the
    /// [authored](allama::Error::authored) message.
    ///
    /// The same contract as [`allama::Error::authored`]: never call it on a
    /// message that embeds an HTTP client's error text, which can contain
    /// the request URL.
    #[must_use]
    pub fn authored(mut self) -> Self {
        self.authored = true;
        self
    }

    /// Attach the underlying error. [`into_error`](Self::into_error) and
    /// [`into_stream_error`](Self::into_stream_error) keep it as the portable
    /// error's [`source`](std::error::Error::source), so a caller can
    /// downcast it (a TLS failure from a DNS one). Its text is printed
    /// wherever the error's chain is, so attach only errors whose `Display`
    /// holds no secret, such as a URL with a key in its query.
    #[must_use]
    pub fn with_source(mut self, source: impl std::error::Error + Send + Sync + 'static) -> Self {
        self.source = Some(Box::new(source));
        self
    }

    /// What went wrong.
    #[must_use]
    pub fn kind(&self) -> TransportErrorKind {
        self.kind
    }

    /// The portable error for a failure before or while receiving a unary
    /// response: [`allama::Error::Timeout`], [`allama::Error::InvalidRequest`],
    /// or [`allama::Error::Transport`].
    ///
    /// The constructor message is an [authored](allama::Error::authored)
    /// description of the [kind](Self::kind). The transport's own text is
    /// the [upstream message](allama::Error::with_upstream_message): an HTTP
    /// client's error can contain the request URL, and some APIs put the key
    /// in the query string, so it stays out of the redacted
    /// [`ErrorLog`](allama::ErrorLog). A message marked
    /// [`authored`](Self::authored) is the constructor message instead.
    ///
    /// ```
    /// use allama_http::{TransportError, TransportErrorKind};
    ///
    /// let err = TransportError::new(
    ///     TransportErrorKind::Connect,
    ///     "error sending request for url (https://api.example.com/v1?key=SECRET)",
    /// )
    /// .into_error();
    /// assert_eq!(err.as_log().to_string(), r#"kind=transport message="the request got no response""#);
    /// assert!(err.to_string().contains("api.example.com"));
    /// ```
    #[must_use]
    pub fn into_error(mut self) -> allama::Error {
        let source = self.source.take();
        if self.authored {
            let error = match self.kind {
                TransportErrorKind::Timeout => allama::Error::timeout(self.message),
                TransportErrorKind::InvalidRequest => allama::Error::invalid_request(self.message),
                TransportErrorKind::TooLarge => allama::Error::unexpected_response(self.message),
                _ => allama::Error::transport(self.message),
            };
            return with_source(error.authored(), source);
        }
        let error = match self.kind {
            TransportErrorKind::Timeout => allama::Error::timeout("the request timed out"),
            TransportErrorKind::InvalidRequest => {
                allama::Error::invalid_request("the request could not be built")
            }
            TransportErrorKind::Connect => allama::Error::transport("the request got no response"),
            TransportErrorKind::Body => {
                allama::Error::transport("reading the response body failed")
            }
            TransportErrorKind::TooLarge => {
                allama::Error::unexpected_response("the response body is larger than the limit")
            }
            TransportErrorKind::Other => allama::Error::transport("the transport failed"),
        };
        with_source(error.authored().with_upstream_message(self.message), source)
    }

    /// The portable error for a failure while reading a streamed response,
    /// after it started: [`allama::Error::Timeout`] for a timeout (a stream
    /// timeout stays a timeout, so retry policies treat it as one), and a
    /// [`StreamErrorKind::Transport`](allama::StreamErrorKind::Transport)
    /// stream error, which is retryable, for anything else.
    ///
    /// As in [`into_error`](Self::into_error), the constructor message is
    /// [authored](allama::Error::authored) and the transport's own text is
    /// the [upstream message](allama::Error::with_upstream_message), unless
    /// the message is marked [`authored`](Self::authored).
    /// [`SseError::into_error`](sse::SseError::into_error) maps a failed body
    /// read with it.
    ///
    /// ```
    /// use allama::{Error, StreamErrorKind};
    /// use allama_http::{TransportError, TransportErrorKind};
    ///
    /// let err = TransportError::new(TransportErrorKind::Body, "connection reset by peer")
    ///     .into_stream_error();
    /// assert!(matches!(err, Error::Stream { kind: StreamErrorKind::Transport, .. }));
    /// assert!(err.is_retryable());
    /// assert_eq!(err.upstream_message(), Some("connection reset by peer"));
    ///
    /// let err = TransportError::new(TransportErrorKind::Timeout, "idle for 30s").into_stream_error();
    /// assert!(matches!(err, Error::Timeout { .. }));
    /// ```
    #[must_use]
    pub fn into_stream_error(mut self) -> allama::Error {
        let source = self.source.take();
        if self.authored {
            let error = match self.kind {
                TransportErrorKind::Timeout => allama::Error::timeout(self.message),
                TransportErrorKind::TooLarge => allama::Error::stream_protocol(self.message),
                _ => allama::Error::stream(allama::StreamErrorKind::Transport, self.message),
            };
            return with_source(error.authored(), source);
        }
        let error = match self.kind {
            TransportErrorKind::Timeout => allama::Error::timeout("the stream timed out"),
            TransportErrorKind::TooLarge => {
                allama::Error::stream_protocol("the stream is larger than the limit")
            }
            _ => allama::Error::stream(
                allama::StreamErrorKind::Transport,
                "reading the stream failed",
            ),
        };
        with_source(error.authored().with_upstream_message(self.message), source)
    }
}

/// Keep a [`TransportError`]'s attached cause as the portable error's
/// [`source`](std::error::Error::source).
fn with_source(
    error: allama::Error,
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
) -> allama::Error {
    match source {
        Some(source) => error.with_source(source),
        None => error,
    }
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            TransportErrorKind::Timeout => write!(f, "request timed out: {}", self.message),
            _ => f.write_str(&self.message),
        }
    }
}

impl std::error::Error for TransportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_deref()
            .map(|e| e as &(dyn std::error::Error + 'static))
    }
}

impl From<TransportError> for allama::Error {
    fn from(err: TransportError) -> Self {
        err.into_error()
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn headers_are_case_insensitive() {
        let headers: Headers = [("Request-Id", "req_1"), ("retry-after", "2")]
            .into_iter()
            .collect();
        assert_eq!(headers.get("request-id"), Some("req_1"));
        assert_eq!(headers.get("RETRY-AFTER"), Some("2"));
        assert_eq!(headers.get("missing"), None);
    }

    #[test]
    fn debug_output_redacts_credentials() {
        let request = HttpRequest::post("https://x", "{}")
            .header("Authorization", "Bearer sk-live-SECRET1")
            .header("x-api-key", "SECRET2")
            .header("x-goog-api-key", "SECRET3")
            .header("Proxy-Authorization", "Basic SECRET4")
            .header("cookie", "session=SECRET5")
            .header("x-gateway-token", "SECRET6")
            .header("X-Client-Secret", "SECRET7")
            .header("api-key", "SECRET8")
            .header("x-request-id", "req_1");
        let debug = format!("{request:?}");
        assert!(!debug.contains("SECRET"), "{debug}");
        assert!(debug.contains(r#""x-request-id": "req_1""#), "{debug}");
        assert!(debug.contains(r#""Authorization": <redacted>"#), "{debug}");

        let response = HttpResponse::from_bytes(
            200,
            [
                ("set-cookie", "session=SECRET9"),
                ("content-type", "text/plain"),
            ]
            .into_iter()
            .collect(),
            "",
        );
        let debug = format!("{response:?}");
        assert!(!debug.contains("SECRET"), "{debug}");
        assert!(debug.contains("text/plain"), "{debug}");

        let signed =
            HttpRequest::post("https://x", "{}").sensitive_header("x-signature", "SECRET10");
        assert_eq!(signed.headers.get("x-signature"), Some("SECRET10"));
        assert!(!format!("{signed:?}").contains("SECRET"));
    }

    #[test]
    fn sensitive_headers_are_matched_by_name_and_fragment() {
        for name in [
            "AUTHORIZATION",
            "cookie",
            "Set-Cookie",
            "x-auth",
            "openai-api-key",
        ] {
            assert!(is_sensitive_header(name), "{name}");
        }
        for name in ["content-type", "x-request-id", "retry-after", "x-tenant-id"] {
            assert!(!is_sensitive_header(name), "{name}");
        }
    }

    #[test]
    fn retry_after_reads_the_named_header_or_milliseconds() {
        let headers: Headers = [("x-retry", "3")].into_iter().collect();
        assert_eq!(headers.retry_after("x-retry"), Some(Duration::from_secs(3)));
        assert_eq!(headers.retry_after("retry-after"), None);

        let headers: Headers = [("retry-after-ms", "1500")].into_iter().collect();
        assert_eq!(
            headers.retry_after("retry-after"),
            Some(Duration::from_millis(1500))
        );

        let past: Headers = [("retry-after", "Sun, 06 Nov 1994 08:49:37 GMT")]
            .into_iter()
            .collect();
        assert_eq!(past.retry_after("retry-after"), Some(Duration::ZERO));
    }

    #[test]
    fn requests_build_fluently() {
        let request = HttpRequest::post("https://x", "{}")
            .header("authorization", "Bearer k")
            .timeout(Some(Duration::from_secs(5)));
        assert_eq!(request.method.as_str(), "POST");
        assert_eq!(request.headers.get("Authorization"), Some("Bearer k"));
        assert_eq!(request.body_text(), Some("{}"));
        assert!(HttpRequest::get("https://x").body.is_none());
    }

    fn bearer_slot() -> CredentialSlot {
        CredentialSlot::new("openai", CredentialPlacement::Bearer)
    }

    fn options(transport: HttpRequestOptions) -> allama::RequestOptions {
        let mut options = allama::RequestOptions::new();
        options.insert(transport);
        options
    }

    fn values<'a>(request: &'a HttpRequest, name: &str) -> Vec<&'a str> {
        request
            .headers
            .iter()
            .filter(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value)
            .collect()
    }

    #[test]
    fn request_options_replace_same_named_headers_and_add_others() {
        let base = || {
            HttpRequest::post("https://x", "{}")
                .header("Authorization", "Bearer provider")
                .header("x-tag", "one")
                .header("x-tag", "two")
                .header("content-type", "application/json")
        };

        let untouched = base()
            .apply_options(&allama::RequestOptions::new(), &bearer_slot())
            .unwrap();
        assert_eq!(untouched, base());

        let applied = base()
            .apply_options(
                &options(
                    HttpRequestOptions::new()
                        .header("X-TAG", "three")
                        .header("x-tenant-id", "acme")
                        .header("Idempotency-Key", "idem-1"),
                ),
                &bearer_slot(),
            )
            .unwrap();
        let headers: Vec<_> = applied.headers.iter().collect();
        assert_eq!(
            headers,
            [
                ("Authorization", "Bearer provider"),
                ("content-type", "application/json"),
                ("X-TAG", "three"),
                ("x-tenant-id", "acme"),
                ("Idempotency-Key", "idem-1"),
            ]
        );

        let mut invalid = allama::RequestOptions::new();
        invalid.insert_json(
            OPTIONS_NAMESPACE,
            serde_json::json!({"headers": {"x-api-key": "k"}}),
        );
        assert!(matches!(
            base().apply_options(&invalid, &bearer_slot()).unwrap_err(),
            allama::Error::InvalidRequest { .. }
        ));
    }

    #[test]
    fn the_scope_credential_replaces_the_configured_one_in_its_slot() {
        let transport = HttpRequestOptions::new()
            .credential("openai", Credential::api_key("TENANT-SECRET"))
            .credential("anthropic", Credential::api_key("FOREIGN-SECRET"));

        let bearer = HttpRequest::post("https://x", "{}")
            .header("authorization", "Bearer provider")
            .apply_options(&options(transport.clone()), &bearer_slot())
            .unwrap();
        assert_eq!(values(&bearer, "authorization"), ["Bearer TENANT-SECRET"]);

        let azure = CredentialSlot::new("openai", CredentialPlacement::Header("api-key".into()))
            .replaces("authorization");
        let header = HttpRequest::post("https://x", "{}")
            .header("authorization", "Bearer entra-token")
            .header("API-KEY", "configured")
            .apply_options(&options(transport.clone()), &azure)
            .unwrap();
        assert_eq!(values(&header, "api-key"), ["TENANT-SECRET"]);
        assert!(values(&header, "authorization").is_empty());

        for request in [&bearer, &header] {
            let debug = format!("{request:?}");
            assert!(!debug.contains("SECRET"), "{debug}");
            assert!(
                !request.headers.iter().any(|(_, v)| v.contains("FOREIGN")),
                "{request:?}"
            );
        }

        // A custom placement is the adapter's to sign; the configured
        // headers stay until it does.
        let custom = CredentialSlot::new("openai", CredentialPlacement::Custom);
        let signed = HttpRequest::post("https://x", "{}")
            .header("authorization", "configured")
            .apply_options(&options(transport.clone()), &custom)
            .unwrap();
        assert_eq!(values(&signed, "authorization"), ["configured"]);
        assert_eq!(
            transport
                .credential_for("openai")
                .map(Credential::expose_secret),
            Some("TENANT-SECRET")
        );
    }

    #[test]
    fn header_credentials_reach_only_their_scope_and_replace_same_named_headers() {
        let transport = HttpRequestOptions::new()
            .credential(
                "openai",
                Credential::header("cf-aig-authorization", "Bearer GATEWAY-SECRET"),
            )
            .credential(
                "anthropic",
                Credential::header("x-other-gw", "OTHER-SECRET"),
            );
        let applied = HttpRequest::post("https://x", "{}")
            .header("authorization", "Bearer provider")
            .header("cf-aig-authorization", "Bearer configured-gateway")
            .apply_options(&options(transport), &bearer_slot())
            .unwrap();
        assert_eq!(
            values(&applied, "cf-aig-authorization"),
            ["Bearer GATEWAY-SECRET"]
        );
        assert_eq!(values(&applied, "authorization"), ["Bearer provider"]);
        assert!(values(&applied, "x-other-gw").is_empty());
        let debug = format!("{applied:?}");
        assert!(!debug.contains("SECRET"), "{debug}");

        // Redacted whatever the name, once placed.
        let custom_name = HttpRequest::post("https://x", "{}")
            .apply_options(
                &options(
                    HttpRequestOptions::new()
                        .credential("openai", Credential::header("x-route", "ROUTE-SECRET")),
                ),
                &bearer_slot(),
            )
            .unwrap();
        assert_eq!(values(&custom_name, "x-route"), ["ROUTE-SECRET"]);
        assert!(!format!("{custom_name:?}").contains("ROUTE-SECRET"));
    }

    #[test]
    fn plain_credential_headers_are_refused_as_configuration_errors() {
        let nimbus = CredentialSlot::new(
            "nimbus",
            CredentialPlacement::Header("x-nimbus-session".into()),
        )
        .replaces("x-nimbus-legacy");
        for (name, slot) in [
            ("Authorization", &nimbus),
            ("x-api-key", &nimbus),
            ("cf-aig-authorization", &nimbus),
            ("ocp-apim-subscription-key", &nimbus),
            ("x-amz-security-token", &nimbus),
            ("cookie", &nimbus),
            ("X-Nimbus-Session", &nimbus),
            ("x-nimbus-legacy", &nimbus),
        ] {
            let err = HttpRequest::post("https://x", "{}")
                .apply_options(
                    &options(HttpRequestOptions::new().header(name, "PLAIN-SECRET")),
                    slot,
                )
                .unwrap_err();
            assert!(
                matches!(err, allama::Error::Configuration { .. }),
                "{err:?}"
            );
            assert!(!err.is_retryable(), "{name}");
            let text = format!("{err} {err:?} {}", err.as_log());
            assert!(!text.contains("PLAIN-SECRET"), "{text}");
            assert!(text.contains(name), "{text}");
        }
        // Another adapter's own header is not this one's to refuse.
        assert!(
            HttpRequest::post("https://x", "{}")
                .apply_options(
                    &options(HttpRequestOptions::new().header("x-nimbus-session", "v")),
                    &bearer_slot(),
                )
                .is_ok()
        );
    }

    #[test]
    fn a_required_slot_refuses_a_request_without_its_credential() {
        let slot = bearer_slot().required(true);
        for transport in [
            None,
            Some(HttpRequestOptions::new().credential("other", Credential::api_key("k"))),
            Some(
                HttpRequestOptions::new()
                    .credential("openai", Credential::header("cf-aig-authorization", "g")),
            ),
        ] {
            let options = transport.map(options).unwrap_or_default();
            let err = HttpRequest::post("https://x", "{}")
                .apply_options(&options, &slot)
                .unwrap_err();
            assert!(
                matches!(err, allama::Error::Configuration { .. }),
                "{err:?}"
            );
            assert!(err.to_string().contains("`openai`"), "{err}");
        }
        let sent = HttpRequest::post("https://x", "{}")
            .apply_options(
                &options(HttpRequestOptions::new().credential("openai", Credential::api_key("k"))),
                &slot,
            )
            .unwrap();
        assert_eq!(sent.headers.get("authorization"), Some("Bearer k"));
    }

    #[test]
    fn a_later_credential_replaces_the_same_kind_in_its_scope() {
        let transport = HttpRequestOptions::new()
            .credential("openai", Credential::api_key("first"))
            .credential("openai", Credential::header("x-gw", "gw-1"))
            .credential("OPENAI", Credential::api_key("other-scope"))
            .credential("openai", Credential::api_key("second"))
            .credential("openai", Credential::header("X-GW", "gw-2"));
        assert_eq!(
            transport
                .credential_for("openai")
                .map(Credential::expose_secret),
            Some("second")
        );
        let scopes: Vec<_> = transport
            .credentials()
            .map(|(scope, credential)| (scope, credential.expose_secret()))
            .collect();
        assert_eq!(
            scopes,
            [
                ("OPENAI", "other-scope"),
                ("openai", "second"),
                ("openai", "gw-2")
            ]
        );
    }

    #[tokio::test]
    async fn bodies_are_read_whole_or_streamed() {
        let response = HttpResponse::new(
            200,
            Headers::new(),
            Box::pin(futures_util::stream::iter([
                Ok(Bytes::from_static(b"he")),
                Ok(Bytes::from_static(b"llo")),
            ])),
        );
        assert!(response.is_success());
        assert_eq!(response.text().await.unwrap(), "hello");

        let failing = HttpResponse::new(
            200,
            Headers::new(),
            Box::pin(futures_util::stream::iter([
                Ok(Bytes::from_static(b"he")),
                Err(TransportError::new(TransportErrorKind::Body, "reset")),
            ])),
        );
        assert_eq!(
            failing.bytes().await.unwrap_err().kind(),
            TransportErrorKind::Body
        );
    }

    fn chunks(parts: Vec<Result<Bytes, TransportError>>) -> Body {
        Box::pin(futures_util::stream::iter(parts))
    }

    #[tokio::test]
    async fn limited_reads_stop_before_passing_the_limit() {
        let body = || {
            chunks(vec![
                Ok(Bytes::from_static(b"0123")),
                Ok(Bytes::from_static(b"4567")),
                Ok(Bytes::from_static(b"89")),
            ])
        };
        let read = |limit| HttpResponse::new(200, Headers::new(), body()).bytes_limited(limit);
        assert_eq!(read(10).await.unwrap(), "0123456789");
        assert_eq!(read(usize::MAX).await.unwrap(), "0123456789");
        for limit in [0, 3, 4, 9] {
            let err = read(limit).await.unwrap_err();
            assert_eq!(err.kind(), TransportErrorKind::TooLarge, "{limit}");
            let err = err.into_error();
            assert!(
                matches!(err, allama::Error::UnexpectedResponse { .. }),
                "{err:?}"
            );
            assert!(!err.is_retryable());
            assert!(
                err.as_log()
                    .to_string()
                    .contains(&format!("limit of {limit} bytes")),
                "{}",
                err.as_log()
            );
        }
        let text = HttpResponse::from_bytes(200, Headers::new(), "héllo")
            .text_limited(6)
            .await
            .unwrap();
        assert_eq!(text, "héllo");
        assert!(matches!(
            TransportError::new(TransportErrorKind::TooLarge, "x").into_stream_error(),
            allama::Error::Stream {
                kind: allama::StreamErrorKind::Protocol,
                ..
            }
        ));
    }

    fn map_http(status: u16, _: &Headers, body: &str) -> allama::Error {
        status_error(status, format!("HTTP {status}"))
            .authored()
            .with_body(body.to_owned())
    }

    /// An error body that never ends is read up to the cap and no
    /// further, so the call ends and memory stays bounded.
    #[tokio::test]
    async fn error_bodies_are_read_up_to_the_cap() {
        let endless: Body = Box::pin(futures_util::stream::repeat_with(|| {
            Ok(Bytes::from(vec![b'x'; 16 * 1024]))
        }));
        let err = HttpResponse::new(503, Headers::new(), endless)
            .map_error(map_http)
            .await;
        assert!(matches!(err, allama::Error::Overloaded { .. }), "{err:?}");
        assert_eq!(err.upstream_message(), None);
        let (body, failure) = read_error_body(
            Box::pin(futures_util::stream::repeat_with(|| {
                Ok(Bytes::from_static(b"abc"))
            })),
            MAX_ERROR_BODY_SIZE,
            ERROR_BODY_TIMEOUT,
        )
        .await;
        assert_eq!(body.len(), MAX_ERROR_BODY_SIZE);
        assert!(failure.is_none());
    }

    /// A body that stalls is given up on after the time bound, keeping
    /// what arrived, and a failed read is recorded on the error rather than
    /// turned into an empty body.
    #[tokio::test]
    async fn error_body_failures_are_recorded() {
        let stalled: Body = Box::pin(
            futures_util::stream::iter([Ok(Bytes::from_static(b"{\"err"))])
                .chain(futures_util::stream::pending()),
        );
        let (body, failure) =
            read_error_body(stalled, MAX_ERROR_BODY_SIZE, Duration::from_millis(20)).await;
        assert_eq!(body, b"{\"err");
        assert_eq!(failure.unwrap().kind(), TransportErrorKind::Timeout);

        let reset = HttpResponse::new(
            500,
            Headers::new(),
            chunks(vec![
                Ok(Bytes::from_static(b"partial")),
                Err(TransportError::new(
                    TransportErrorKind::Body,
                    "connection reset",
                )),
            ]),
        );
        let err = reset.map_error(map_http).await;
        assert_eq!(
            err.upstream_message(),
            Some("reading the error body failed: connection reset")
        );
        assert_eq!(err.context().and_then(|c| c.capped_body()), Some("partial"));

        let enveloped = HttpResponse::new(
            500,
            Headers::new(),
            chunks(vec![Err(TransportError::new(
                TransportErrorKind::Body,
                "reset",
            ))]),
        )
        .map_error(|status, headers, body| {
            map_http(status, headers, body).with_upstream_message("upstream")
        })
        .await;
        assert_eq!(
            enveloped.upstream_message(),
            Some("upstream (reading the error body failed: reset)")
        );
    }

    #[tokio::test]
    async fn redirects_are_unexpected_responses_naming_the_host() {
        let mut called = false;
        let headers: Headers = [
            (
                "location",
                "https://u:pw@Other.Example:8443/v1/x?token=SECRET",
            ),
            ("x-request-id", "req_1"),
        ]
        .into_iter()
        .collect();
        let err = HttpResponse::from_bytes(308, headers, "")
            .map_error(|status, headers, body| {
                called = true;
                map_http(status, headers, body)
            })
            .await;
        assert!(!called);
        assert!(
            matches!(err, allama::Error::UnexpectedResponse { .. }),
            "{err:?}"
        );
        assert_eq!(err.status(), Some(308));
        assert_eq!(err.request_id(), Some("req_1"));
        let log = err.as_log().to_string();
        assert!(log.contains("`other.example`"), "{log}");
        for text in [log, err.to_string(), format!("{err:?}")] {
            assert!(!text.contains("SECRET") && !text.contains("pw"), "{text}");
        }
        let relative = redirect_error(301, &[("location", "/v2/chat")].into_iter().collect());
        assert!(
            relative.to_string().contains("within the same host"),
            "{relative}"
        );
        let bare = redirect_error(302, &Headers::new());
        assert!(bare.to_string().contains("without a `Location`"), "{bare}");
    }

    #[test]
    fn with_headers_replaces_same_named_headers() {
        let defaults: Headers = [("User-Agent", "a/1"), ("x-route", "eu")]
            .into_iter()
            .collect();
        let request = HttpRequest::post("https://x", "{}")
            .header("content-type", "application/json")
            .header("user-agent", "old")
            .sensitive_header("x-signed", "SECRET")
            .with_headers(&defaults);
        assert_eq!(
            request.headers.iter().collect::<Vec<_>>(),
            [
                ("content-type", "application/json"),
                ("x-signed", "SECRET"),
                ("User-Agent", "a/1"),
                ("x-route", "eu"),
            ]
        );
        assert!(!format!("{request:?}").contains("SECRET"));
    }

    #[test]
    fn errors_map_to_portable_errors() {
        let timeout = TransportError::new(TransportErrorKind::Timeout, "after 5s");
        assert_eq!(timeout.to_string(), "request timed out: after 5s");
        assert!(matches!(
            allama::Error::from(timeout),
            allama::Error::Timeout { .. }
        ));
        assert!(matches!(
            TransportError::new(TransportErrorKind::InvalidRequest, "bad url").into_error(),
            allama::Error::InvalidRequest { .. }
        ));
        assert!(matches!(
            TransportError::new(TransportErrorKind::Connect, "refused").into_error(),
            allama::Error::Transport { .. }
        ));
    }

    #[test]
    fn transport_text_stays_out_of_the_log() {
        for kind in [
            TransportErrorKind::Timeout,
            TransportErrorKind::InvalidRequest,
            TransportErrorKind::Connect,
            TransportErrorKind::Body,
            TransportErrorKind::Other,
        ] {
            let err = TransportError::new(kind, "https://x.example/v1?key=SECRET").into_error();
            let line = err.as_log().to_string();
            assert!(!line.contains("SECRET"), "{line}");
            assert!(line.contains("message="), "{line}");
            assert_eq!(
                err.upstream_message(),
                Some("https://x.example/v1?key=SECRET")
            );
            assert!(err.to_string().contains("SECRET"));
        }
    }
}

/// Run on `wasm32` with `wasm-bindgen-test`
/// (`cargo test --target wasm32-unknown-unknown -p allama-http`).
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests {
    use wasm_bindgen_test::wasm_bindgen_test;

    use super::*;

    #[wasm_bindgen_test]
    fn retry_after_reads_the_clock() {
        let past: Headers = [("retry-after", "Sun, 06 Nov 1994 08:49:37 GMT")]
            .into_iter()
            .collect();
        assert_eq!(past.retry_after("retry-after"), Some(Duration::ZERO));

        let future: Headers = [("retry-after", "Fri, 31 Dec 9999 23:59:59 GMT")]
            .into_iter()
            .collect();
        assert!(future.retry_after("retry-after").unwrap() > Duration::ZERO);
    }
}
