//! What [`ProviderBuilder`](crate::ProviderBuilder) and
//! [`ResponsesProviderBuilder`](crate::ResponsesProviderBuilder) share: the
//! settings they collect, the setters that collect them, and how the
//! settings configure the compat builder underneath. The two public builders stay separate types, each with its
//! own `build`.

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Duration;

use allama::{Error, Result};
use allama_http::config::redact_url;
use allama_http::{HttpTransport, StreamTimeouts};

/// The settings of an OpenAI builder.
#[derive(Default)]
pub(crate) struct Settings {
    pub(crate) api_key: Option<String>,
    pub(crate) credential_scope: Option<Cow<'static, str>>,
    pub(crate) base_url: Option<String>,
    pub(crate) organization: Option<String>,
    pub(crate) project: Option<String>,
    pub(crate) timeout: Option<Option<Duration>>,
    pub(crate) stream_timeouts: Option<StreamTimeouts>,
    pub(crate) max_stream_event_size: Option<usize>,
    pub(crate) max_response_size: Option<usize>,
    pub(crate) headers: Vec<(String, String)>,
    pub(crate) http: Option<Arc<dyn HttpTransport>>,
}

impl Settings {
    /// `Debug` for the builder `name`, without the API key, header values,
    /// or credentials in the base URL.
    pub(crate) fn debug(&self, name: &str, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(name)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("credential_scope", &self.credential_scope)
            .field("base_url", &self.base_url.as_deref().map(redact_url))
            .field("organization", &self.organization)
            .field("project", &self.project)
            .field(
                "headers",
                &self
                    .headers
                    .iter()
                    .map(|(name, _)| name)
                    .collect::<Vec<_>>(),
            )
            .field("http_transport", &self.http.is_some())
            .field("timeout", &self.timeout)
            .field("stream_timeouts", &self.stream_timeouts)
            .field("max_stream_event_size", &self.max_stream_event_size)
            .field("max_response_size", &self.max_response_size)
            .finish()
    }
}

/// The setters both builders share, writing to `self.settings`. The
/// builders write `credential_scope` and `max_stream_event_size`
/// themselves, since their docs differ.
macro_rules! shared_setters {
    () => {
        /// API key. Optional when a [`credential_scope`](Self::credential_scope)
        /// is set. Surrounding whitespace is removed.
        #[must_use]
        pub fn api_key(mut self, key: impl Into<String>) -> Self {
            self.settings.api_key = Some(key.into());
            self
        }

        /// Base URL (default `https://api.openai.com/v1`).
        #[must_use]
        pub fn base_url(mut self, url: impl Into<String>) -> Self {
            self.settings.base_url = Some(url.into());
            self
        }

        /// Organization id (`OpenAI-Organization` header).
        #[must_use]
        pub fn organization(mut self, organization: impl Into<String>) -> Self {
            self.settings.organization = Some(organization.into());
            self
        }

        /// Project id (`OpenAI-Project` header).
        #[must_use]
        pub fn project(mut self, project: impl Into<String>) -> Self {
            self.settings.project = Some(project.into());
            self
        }

        /// The timeout for a whole non-streaming call; `None` sets none.
        ///
        /// Left unset, it is [`DEFAULT_REQUEST_TIMEOUT`](allama_http::DEFAULT_REQUEST_TIMEOUT)
        /// (120 s), whatever the transport: a client of your own keeps the
        /// bound unless you pass `None` here. Set, it applies in place of a
        /// client's own total timeout.
        ///
        /// Streams have no total timeout:
        /// [`stream_timeouts`](Self::stream_timeouts) bounds them.
        #[must_use]
        pub fn timeout(mut self, timeout: Option<std::time::Duration>) -> Self {
            self.settings.timeout = Some(timeout);
            self
        }

        /// Bounds on waiting for a stream's headers and between its chunks
        /// (default [`StreamTimeouts::default`](crate::StreamTimeouts::default);
        /// [`StreamTimeouts::unbounded`](crate::StreamTimeouts::unbounded)
        /// disables them).
        #[must_use]
        pub fn stream_timeouts(mut self, timeouts: crate::StreamTimeouts) -> Self {
            self.settings.stream_timeouts = Some(timeouts);
            self
        }

        /// The largest non-streaming response body, in bytes, the provider
        /// reads (default [`allama_http::DEFAULT_MAX_RESPONSE_SIZE`], 256 MiB).
        /// A larger body fails the call with an unexpected-response error,
        /// which is not retryable, without being read whole.
        #[must_use]
        pub fn max_response_size(mut self, bytes: usize) -> Self {
            self.settings.max_response_size = Some(bytes);
            self
        }

        /// Send the header `name: value` with every request, replacing a
        /// header of the same name the provider sets (later calls replace
        /// earlier ones). For headers that belong to this provider: a gateway's
        /// routing header, a tenant tag. `header("user-agent", ..)` replaces the
        /// default [`allama_http::DEFAULT_USER_AGENT`]. A request's own
        /// `allama_http::HttpRequestOptions` headers still replace these.
        ///
        /// Credential headers are refused at [`build`](Self::build): set the
        /// key with [`api_key`](Self::api_key), or send one per request as an
        /// `allama_http::Credential`.
        #[must_use]
        pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
            self.settings.headers.push((name.into(), value.into()));
            self
        }

        /// Send requests through `transport` instead of the default `reqwest`
        /// client: an `allama_http::ReqwestTransport` over your own client (a
        /// proxy, other TLS settings), or an [`allama_http::ReplayTransport`]
        /// in tests. A client of your own keeps its redirect policy: build it
        /// with `.redirect(reqwest::redirect::Policy::none())` to keep the
        /// default transport's refusal of redirects.
        ///
        /// The [`timeout`](Self::timeout), 120 s unless set, goes to the
        /// transport with each non-streaming request, whatever the transport.
        #[must_use]
        pub fn http_transport(mut self, transport: impl allama_http::HttpTransport) -> Self {
            self.settings.http = Some(std::sync::Arc::new(transport));
            self
        }
    };
}

/// Configure a compat builder (`allama_openai_compat::ProviderBuilder` or
/// `ResponsesProviderBuilder`, which have the same setters) with `settings`
/// and OpenAI's identity, returning it. Fails, through `?`, when the API key
/// is missing or malformed.
macro_rules! configure {
    ($builder:expr, $settings:expr) => {{
        let settings: $crate::builder::Settings = $settings;
        let api_key =
            $crate::builder::api_key(settings.api_key, settings.credential_scope.is_some())?;
        let mut builder = $builder
            .base_url(
                settings
                    .base_url
                    .unwrap_or_else(|| $crate::DEFAULT_BASE_URL.into()),
            )
            .provider_name($crate::PROVIDER_NAME)
            .credential_placement(allama_http::CredentialPlacement::Bearer)
            .credential_scope(
                settings
                    .credential_scope
                    .unwrap_or(std::borrow::Cow::Borrowed($crate::DEFAULT_CREDENTIAL_SCOPE)),
            )
            .timeout(
                settings
                    .timeout
                    .unwrap_or(Some(allama_http::DEFAULT_REQUEST_TIMEOUT)),
            );
        if let Some(api_key) = api_key {
            builder = builder.bearer_token(api_key);
        }
        if let Some(org) = settings.organization.filter(|v| !v.trim().is_empty()) {
            builder = builder.organization_header("openai-organization", org);
        }
        if let Some(project) = settings.project.filter(|v| !v.trim().is_empty()) {
            builder = builder.project_header("openai-project", project);
        }
        if let Some(timeouts) = settings.stream_timeouts {
            builder = builder.stream_timeouts(timeouts);
        }
        if let Some(bytes) = settings.max_stream_event_size {
            builder = builder.max_stream_event_size(bytes);
        }
        if let Some(bytes) = settings.max_response_size {
            builder = builder.max_response_size(bytes);
        }
        for (name, value) in settings.headers {
            builder = builder.header(name, value);
        }
        if let Some(http) = settings.http {
            builder = builder.http_transport(http);
        }
        builder
    }};
}

pub(crate) use {configure, shared_setters};

/// The configured API key, trimmed and checked
/// ([`allama_http::config::secret`]): required unless a credential scope
/// is set.
pub(crate) fn api_key(key: Option<String>, scoped: bool) -> Result<Option<String>> {
    match key {
        Some(key) => allama_http::config::secret("api_key", &key).map(Some),
        None if scoped => Ok(None),
        None => Err(Error::configuration("api_key is required").authored()),
    }
}
