use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{ExtraMap, RequestField, Subject};

type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Convenience alias for `std::result::Result<T, E>` with [`Error`] as the
/// default error type.
///
/// The error parameter stays overridable, so importing the alias (for example
/// through `use allama::prelude::*`) does not break code that names its own
/// error type:
///
/// ```
/// use allama::prelude::*;
///
/// #[derive(Debug)]
/// struct MyError;
///
/// fn parse(input: &str) -> Result<u32, MyError> {
///     input.parse().map_err(|_| MyError)
/// }
///
/// fn provider_call() -> Result<u32> {
///     Err(Error::invalid_request("demo"))
/// }
///
/// assert_eq!(parse("7").unwrap(), 7);
/// assert!(provider_call().is_err());
/// ```
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Diagnostic context shared by provider-originated errors.
///
/// Fields are best-effort: providers fill what their transport exposes. The
/// HTTP-shaped fields (`status`, `body`) exist because they are the most
/// useful debugging data for HTTP providers; they do not imply every provider
/// is HTTP-based.
///
/// # Message provenance
///
/// An error's text can come from two places, and the redacted
/// [`ErrorLog`] treats them differently:
///
/// - The **constructor message** (the `message` passed to
///   [`Error::invalid_request`] and friends) is shown in [`ErrorLog`] only
///   when [`message_origin`](Self::message_origin) is
///   [`MessageOrigin::Authored`], set with [`Error::authored`].
/// - The **upstream message** ([`upstream_message`](Self::upstream_message),
///   set with [`Error::with_upstream_message`]) is the provider's own text.
///   It may echo prompts or input values, so [`ErrorLog`] leaves it out
///   unless asked for with [`ErrorLog::with_message`].
///
/// `Display` is not redacted and shows both. `Debug` shows the constructor
/// message and the upstream message's length, never its text
/// (`upstream_message: <withheld N bytes; print with {err}>`), filters the
/// provider code, type and `param` as [`ErrorLog`] does, and leaves out the
/// fields that are not set.
#[derive(Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ErrorContext {
    /// Identity of the provider that produced the error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// HTTP status code, when the failure came from an HTTP response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    /// Provider-assigned request ID for support tickets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// Suggested delay before retrying, when reported.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "option_duration_secs_f64"
    )]
    pub retry_after: Option<Duration>,
    /// The server's retry directive, when it sent one: `Some(true)` asks the
    /// client to retry, `Some(false)` not to (HTTP `x-should-retry`).
    ///
    /// The default [`RetryPolicy`](crate::wrappers::RetryPolicy) classifier
    /// follows it ahead of the error kind and status; custom classifiers
    /// read it with [`Error::server_retry_directive`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_retry_directive: Option<bool>,
    /// Provider error code (e.g. `insufficient_quota`, `RESOURCE_EXHAUSTED`).
    ///
    /// Upstream text: [`ErrorLog`] shows it only when it is a bounded token
    /// (see [`ErrorLog`]'s filter).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_code: Option<String>,
    /// Provider error type or category, when distinct from the code.
    ///
    /// Upstream text: [`ErrorLog`] shows it only when it is a bounded token
    /// (see [`ErrorLog`]'s filter).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_type: Option<String>,
    /// The path of the request field the provider rejected, when it names
    /// one (OpenAI's `param`, such as `messages.[1].content`; Gemini's
    /// `BadRequest` field violations).
    ///
    /// A field path, never a field value, is expected; [`ErrorLog`] shows it
    /// only when it is a bounded token (see [`ErrorLog`]'s filter).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    /// Where the constructor message came from. Only an
    /// [`Authored`](MessageOrigin::Authored) message appears in
    /// [`ErrorLog`].
    #[serde(default, skip_serializing_if = "MessageOrigin::is_unknown")]
    pub message_origin: MessageOrigin,
    /// The provider's own error text, such as the `message` of a JSON error
    /// envelope. Excluded from [`ErrorLog`] unless asked for with
    /// [`ErrorLog::with_message`], because it may echo prompts or input
    /// values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream_message: Option<String>,
    /// Raw response body, capped at [`MAX_BODY_BYTES`](Self::MAX_BODY_BYTES).
    /// Excluded from [`ErrorLog`] because it may contain prompts, model
    /// output, or credentials.
    ///
    /// [`Error::with_body`] applies the cap. A body written to this field
    /// directly is capped again when serialized and in `Debug`, so serde
    /// round trips of a longer body are lossy.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_capped_body"
    )]
    pub body: Option<String>,
    /// Additional error details, keyed by namespace: provider-specific
    /// details from the adapter, and records that allama itself attaches
    /// under an `allama.*` key. Keys under `allama.` are reserved for
    /// allama.
    ///
    /// Extraction attaches its pass record (mode, passes, summed usage) to
    /// provider errors under `allama.extraction`; read it with
    /// `ExtractionMetadata::from_error`.
    /// [`embed_chunked`](crate::EmbeddingProviderExt::embed_chunked)
    /// attaches its progress (inputs completed, summed usage) under
    /// `allama.embedding`; read it with
    /// [`ChunkedEmbeddingProgress::from_error`](crate::ChunkedEmbeddingProgress::from_error).
    /// [`ErrorLog`] includes those records, which hold no input or output,
    /// and leaves every other key out.
    #[serde(default, skip_serializing_if = "ExtraMap::is_empty")]
    pub metadata: ExtraMap,
    /// The error tried before this one, when this error ended a failover:
    /// read it with [`Error::fallback_from`], set it with
    /// [`Error::with_fallback_from`]. Serialized as `fallback_from`, not
    /// compared, and logged by [`ErrorLog`] as a nested redacted log.
    #[serde(default, skip_serializing_if = "FallbackFrom::is_none")]
    fallback_from: FallbackFrom,
    /// The underlying cause, returned from the error's
    /// [`source`](std::error::Error::source). Not serialized, not compared,
    /// and not shown by `Debug` or [`ErrorLog`]. Set it with
    /// [`Error::with_source`].
    #[serde(skip)]
    source: ErrorSource,
}

/// The cause an [`ErrorContext`] keeps: shared so the error stays `Clone`,
/// and equal to any other so the context's equality reads its fields.
#[derive(Clone, Default)]
struct ErrorSource(Option<Arc<dyn std::error::Error + Send + Sync + 'static>>);

impl PartialEq for ErrorSource {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for ErrorSource {}

/// The error an [`ErrorContext`] records as tried before it
/// ([`Error::fallback_from`]): shared so the error stays `Clone`, and equal
/// to any other, like [`ErrorSource`], so the context keeps `PartialEq` and
/// `Eq` although [`Error`] has neither.
#[derive(Clone, Default)]
struct FallbackFrom(Option<Arc<Error>>);

impl FallbackFrom {
    fn is_none(&self) -> bool {
        self.0.is_none()
    }
}

impl PartialEq for FallbackFrom {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for FallbackFrom {}

impl Serialize for FallbackFrom {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.0.as_deref().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FallbackFrom {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        Ok(Self(
            Option::<Error>::deserialize(deserializer)?.map(Arc::new),
        ))
    }
}

/// The [`ErrorContext::metadata`] key of an extraction's pass record.
pub(crate) const EXTRACTION_METADATA_KEY: &str = "allama.extraction";

/// The [`ErrorContext::metadata`] key of a chunked embedding's progress.
pub(crate) const EMBEDDING_METADATA_KEY: &str = "allama.embedding";

fn context_is_empty(context: &ErrorContext) -> bool {
    context.is_empty()
}

impl ErrorContext {
    /// The most of a response body an error keeps, in bytes. Longer bodies
    /// are truncated at a character boundary.
    pub const MAX_BODY_BYTES: usize = 8 * 1024;

    /// Create an empty context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The body, capped at [`MAX_BODY_BYTES`](Self::MAX_BODY_BYTES).
    #[must_use]
    pub fn capped_body(&self) -> Option<&str> {
        self.body.as_deref().map(cap_body)
    }

    fn is_empty(&self) -> bool {
        self == &Self::default() && self.source.0.is_none() && self.fallback_from.is_none()
    }

    /// The underlying cause, when one was attached with
    /// [`Error::with_source`].
    #[must_use]
    pub fn source(&self) -> Option<&(dyn std::error::Error + Send + Sync + 'static)> {
        self.source.0.as_deref()
    }

    /// [`Error::with_context`]: overwrite with every `Some` field of
    /// `other` (the earlier errors of a failover too), merge metadata, and
    /// never touch the message origin.
    fn merge(&mut self, other: ErrorContext) {
        // Destructured so a new field needs a merge rule to compile.
        let ErrorContext {
            provider,
            status,
            request_id,
            retry_after,
            server_retry_directive,
            provider_code,
            provider_type,
            param,
            message_origin: _,
            upstream_message,
            body,
            metadata,
            fallback_from,
            source,
        } = other;
        overwrite(&mut self.provider, provider);
        overwrite(&mut self.status, status);
        overwrite(&mut self.request_id, request_id);
        overwrite(&mut self.retry_after, retry_after);
        overwrite(&mut self.server_retry_directive, server_retry_directive);
        overwrite(&mut self.provider_code, provider_code);
        overwrite(&mut self.provider_type, provider_type);
        overwrite(&mut self.param, param);
        overwrite(&mut self.upstream_message, upstream_message);
        overwrite(
            &mut self.body,
            body.map(|mut body| {
                body.truncate(cap_body(&body).len());
                body
            }),
        );
        self.metadata.extend(metadata);
        if !fallback_from.is_none() {
            self.fallback_from = fallback_from;
        }
        if source.0.is_some() {
            self.source = source;
        }
    }

    /// Add the fields that are set to `debug`, redacted as its `Debug`
    /// describes. [`Error`]'s `Debug` shows them inline.
    fn debug_fields(&self, debug: &mut fmt::DebugStruct<'_, '_>) {
        // Destructured so a new field needs a `Debug` rule to compile.
        let Self {
            provider,
            status,
            request_id,
            retry_after,
            server_retry_directive,
            provider_code,
            provider_type,
            param,
            message_origin,
            upstream_message,
            body: _,
            metadata,
            fallback_from,
            // Its text may hold upstream text or a URL.
            source: _,
        } = self;
        if let Some(provider) = provider {
            debug.field("provider", provider);
        }
        if let Some(status) = status {
            debug.field("status", status);
        }
        for (name, value) in [
            ("provider_code", provider_code),
            ("provider_type", provider_type),
            ("param", param),
        ] {
            if let Some(value) = value {
                debug.field(name, &log_token(value));
            }
        }
        if let Some(request_id) = request_id {
            debug.field("request_id", request_id);
        }
        if let Some(retry_after) = retry_after {
            debug.field("retry_after", retry_after);
        }
        if let Some(directive) = server_retry_directive {
            debug.field("server_retry_directive", directive);
        }
        if !message_origin.is_unknown() {
            debug.field("message_origin", message_origin);
        }
        if let Some(upstream) = upstream_message {
            debug.field("upstream_message", &Withheld::shown_by_display(upstream));
        }
        if let Some(body) = self.capped_body() {
            debug.field("body_len", &body.len());
        }
        if !metadata.is_empty() {
            debug.field("metadata", &MetadataKeys(metadata));
        }
        if let Some(earlier) = &fallback_from.0 {
            // Redacted like this error: `Error`'s own `Debug`.
            debug.field("fallback_from", earlier);
        }
    }
}

/// Shows the fields that are set, the body's and the upstream message's
/// lengths, never their text, the provider code, type and `param` only when
/// they are bounded tokens (see [`ErrorLog`]'s filter), and the metadata
/// keys without their values.
impl fmt::Debug for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("ErrorContext");
        self.debug_fields(&mut debug);
        debug.finish()
    }
}

/// `Debug` text that stands in for upstream text, such as a provider's
/// message or a model's output: its length, never the text.
pub(crate) struct Withheld {
    bytes: usize,
    shown_by_display: bool,
}

impl Withheld {
    /// Text that the error's `Display` shows.
    pub(crate) fn shown_by_display(text: &str) -> Self {
        Self {
            bytes: text.len(),
            shown_by_display: true,
        }
    }

    /// Text that the error's `Display` does not show either.
    #[cfg(feature = "extract")]
    pub(crate) fn new(text: &str) -> Self {
        Self {
            bytes: text.len(),
            shown_by_display: false,
        }
    }
}

impl fmt::Debug for Withheld {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<withheld {} bytes", self.bytes)?;
        if self.shown_by_display {
            f.write_str("; print with {err}")?;
        }
        f.write_str(">")
    }
}

/// `Debug` text for [`ErrorContext::metadata`]: its keys, not their values.
struct MetadataKeys<'a>(&'a ExtraMap);

impl fmt::Debug for MetadataKeys<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.0.keys()).finish()
    }
}

/// Where an [`Error::Unsupported`] refusal happened (see
/// [`Error::unsupported_stage`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum UnsupportedStage {
    /// The adapter refused the request before sending it: nothing reached
    /// the provider. Every refusal [`Degradations`](crate::adapter::Degradations)
    /// builds has this stage.
    BeforeSending,
    /// The provider refused the request it received, and the adapter mapped
    /// that reply ([`Error::unsupported_by_provider`]).
    Provider,
    /// Not stated ([`Error::unsupported`]): the request may have been sent.
    ///
    /// Also what an error stored by a newer allama with a stage this
    /// version does not know reads as.
    #[default]
    #[serde(other)]
    Unspecified,
}

impl UnsupportedStage {
    /// Stable snake_case name, as serialized.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BeforeSending => "before_sending",
            Self::Provider => "provider",
            Self::Unspecified => "unspecified",
        }
    }

    fn is_unspecified(&self) -> bool {
        *self == Self::Unspecified
    }
}

impl fmt::Display for UnsupportedStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Where an error's constructor message came from.
///
/// See [`ErrorContext`]'s message provenance section.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MessageOrigin {
    /// Not known to be free of upstream text: redacted from [`ErrorLog`].
    ///
    /// The default, including for errors applications and tests build with
    /// the public constructors.
    #[default]
    Unknown,
    /// Written by library or adapter code, with no upstream text in it:
    /// shown in [`ErrorLog`]. Set with [`Error::authored`].
    Authored,
}

impl MessageOrigin {
    fn is_unknown(&self) -> bool {
        *self == Self::Unknown
    }
}

/// Kinds of streaming failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum StreamErrorKind {
    /// The stream ended before its terminal event (truncated). Retryable.
    Incomplete,
    /// The transport failed mid-stream (connection reset, read error). Retryable.
    Transport,
    /// A stream event could not be decoded.
    Decode,
    /// Stream events violated the event protocol (ordering, conflicts).
    Protocol,
    /// Any other streaming failure.
    Other,
}

impl StreamErrorKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Incomplete => "incomplete",
            Self::Transport => "transport",
            Self::Decode => "decode",
            Self::Protocol => "protocol",
            Self::Other => "other",
        }
    }
}

/// Unified error type for all model operations.
///
/// Provider-originated variants carry an [`ErrorContext`]. Variants are
/// `#[non_exhaustive]`: construct them with the lowercase constructors
/// ([`Error::timeout`], [`Error::rate_limited`], ...) and the `with_*`
/// builders, and match them with `..`.
///
/// # Building errors
///
/// One rule covers the context builders: **`with_*` takes the value**, and
/// each field an upstream reply may lack also has a **`with_*_opt`** twin
/// that takes an `Option` and keeps the current value on `None`, so adapter
/// code passes parsed headers straight through. Text twins take an
/// `Option<&str>`: a header passes as it is, an owned field with
/// `.as_deref()`, and a bare `None` compiles. The provider and the status
/// have no `_opt` form: the adapter always knows them.
///
/// ```
/// use std::time::Duration;
///
/// use allama::Error;
///
/// // Application and test code: values.
/// let err = Error::rate_limited("HTTP 429")
///     .with_status(429)
///     .with_retry_after(Duration::from_secs(30))
///     .with_request_id("req_1");
/// assert_eq!(err.retry_after(), Some(Duration::from_secs(30)));
///
/// // Adapter code: whatever the response had.
/// let header: Option<&str> = None;
/// let err = err.with_request_id_opt(header);
/// assert_eq!(err.request_id(), Some("req_1"));
/// ```
///
/// # Handling errors
///
/// The variants are the error kinds: match on them to decide what to do.
/// For logs and metrics, [`kind_name`](Self::kind_name) gives the variant's
/// stable snake_case name, and [`as_log`](Self::as_log) a redacted view that
/// keeps prompts and provider text out.
///
/// [`Error::Unsupported`] is the application's mistake or the end user's,
/// by what it refused ([`unsupported_subject`](Self::unsupported_subject)):
/// a [`Subject::Content`] is a part of the conversation, such as an image
/// the user attached for a text-only model, so it answers like a bad
/// request; anything else is a feature the application chose, a
/// configuration fault.
///
/// A [`FallbackProvider`](crate::wrappers::FallbackProvider) that fails on
/// every side returns the last side's error as itself: same variant, status
/// and kind, so the same `match` handles a failed failover. The errors the
/// failover tried before it are kept on it, newest first: walk them with
/// [`fallback_from`](Self::fallback_from). [`as_log`](Self::as_log) logs
/// them too, and `Display` shows only the error itself.
///
/// ```
/// use allama::adapter::Degradations;
/// use allama::{Error, RequestField, Subject, UnsupportedPolicy};
///
/// fn http_status(err: &Error) -> u16 {
///     // One label per kind, safe for a metric or a log field.
///     let kind = err.kind_name();
///     // The redacted log includes the errors a failover tried first.
///     eprintln!("chat failed: kind={kind} {}", err.as_log());
///
///     match err {
///         // Content the end user sent that the model cannot take, such as
///         // an image for a text-only model: their request.
///         Error::Unsupported { .. }
///             if matches!(err.unsupported_subject(), Some(Subject::Content { .. })) =>
///         {
///             400
///         }
///         // A setup mistake: a missing key, a model the provider does not
///         // have, or a feature the application asks for that the model
///         // lacks.
///         Error::Configuration { .. } | Error::ModelNotFound { .. } | Error::Unsupported { .. } => 500,
///         Error::ContextLengthExceeded { .. } => 413,
///         Error::ContentFiltered { .. } => 422,
///         Error::InvalidRequest { .. } => 400,
///         Error::RateLimited { .. } => 429,
///         // Refused by the application's own policy, such as a budget.
///         Error::Rejected { .. } => 403,
///         _ if err.is_retryable() => 503,
///         // Rejected credentials, an exhausted quota, other provider failures.
///         _ => 502,
///     }
/// }
///
/// let err = Error::rate_limited("HTTP 429").with_status(429);
/// assert_eq!(err.kind_name(), "rate_limited");
/// assert_eq!(http_status(&err), 429);
///
/// // What a failed failover returns: the backup's error, with the
/// // primary's behind it.
/// let failover = err.with_fallback_from(Error::overloaded("HTTP 529"));
/// assert_eq!(failover.kind_name(), "rate_limited");
/// assert_eq!(http_status(&failover), 429);
/// assert!(matches!(failover.fallback_from(), Some(Error::Overloaded { .. })));
///
/// // `Unsupported` maps by what was refused.
/// let degradations = Degradations::new("demo", UnsupportedPolicy::Warn);
/// let tools = degradations.unsupported(RequestField::Tools, "no tool calls");
/// assert_eq!(http_status(&tools), 500);
/// let image = Subject::Content { message: 0, block: Some(1) };
/// assert_eq!(http_status(&degradations.reject(image, "no image input")), 400);
/// ```
///
/// # Printing errors
///
/// `Display` (`{err}`) shows everything, including the provider's own
/// text ([`upstream_message`](Self::upstream_message)), which may echo
/// prompts or input values. `Debug` (`{err:?}`) is compact: the variant,
/// its message, and the context fields that are set. It withholds the
/// provider's text, the response body, and the model's output in an
/// [`Error::Extract`], and shows only their lengths:
///
/// ```text
/// InvalidRequest { message: "HTTP 400", provider: "openai", status: 400, upstream_message: <withheld 69 bytes; print with {err}> }
/// ```
///
/// So `tracing::error!(?err)` keeps prompts out of logs.
/// [`as_log`](Self::as_log) is the redacted view for structured logs.
///
/// A `main` that returns `allama::Result<()>` or `Result<(), Box<dyn
/// std::error::Error>>` prints a failure with this `Debug`, so the
/// provider's explanation is withheld there. (`anyhow::Result` shows it:
/// anyhow's `Debug` prints the error's `Display` and its causes.) Print it
/// with `Display` to see it:
///
/// ```no_run
/// # fn run() -> allama::Result<()> { Ok(()) }
/// fn main() {
///     if let Err(err) = run() {
///         eprintln!("error: {err}");
///         std::process::exit(1);
///     }
/// }
/// ```
///
/// An error serializes, for logs and job queues. Its names are pinned (the
/// kind, and the persisted names it holds), while its shape is
/// same-version (see
/// [Serialization and stability](crate#serialization-and-stability)).
///
#[cfg_attr(
    not(feature = "extract"),
    doc = "[`Error::Extract`]: crate#cargo-features"
)]
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Error {
    /// The request timed out: the client's own timeout, or a server or
    /// gateway that gave up (HTTP `408`, `504`, or Cloudflare's `524`).
    ///
    /// Retryable. A gateway timeout can come after the model ran, so the
    /// request may still have been processed and billed.
    #[non_exhaustive]
    Timeout {
        /// Human-readable description.
        message: String,
        /// Diagnostic context.
        #[serde(default, skip_serializing_if = "context_is_empty")]
        context: Box<ErrorContext>,
    },

    /// No response was received: connection refused, DNS or TLS failure, or
    /// the connection dropped before a response arrived.
    ///
    /// The request may or may not have reached the provider: a connection
    /// reset after the request was written cannot be told apart from one
    /// that was never sent. Do not assume it was not processed or billed.
    ///
    /// Retryable, so a retry may repeat a request the provider already
    /// processed. Distinct from [`Error::Provider`], which means the provider
    /// answered with a failure, and from [`Error::Timeout`].
    #[non_exhaustive]
    Transport {
        /// Human-readable description.
        message: String,
        /// Diagnostic context.
        #[serde(default, skip_serializing_if = "context_is_empty")]
        context: Box<ErrorContext>,
    },

    /// Authentication or authorization failed.
    #[non_exhaustive]
    Auth {
        /// Human-readable description.
        message: String,
        /// Diagnostic context.
        #[serde(default, skip_serializing_if = "context_is_empty")]
        context: Box<ErrorContext>,
    },

    /// The client is misconfigured: a missing credential, an unknown provider
    /// key, a malformed model address, or a missing runtime binding.
    ///
    /// Raised locally, before any request reaches a provider. Not retryable
    /// and not eligible for fallback: the same configuration fails the same
    /// way every time, and failing over would hide the mistake. Distinct from
    /// [`Error::Auth`], which means the provider rejected the credentials it
    /// was given.
    #[non_exhaustive]
    Configuration {
        /// Human-readable description.
        message: String,
        /// Diagnostic context.
        #[serde(default, skip_serializing_if = "context_is_empty")]
        context: Box<ErrorContext>,
    },

    /// A wrapper or the application refused the call before any provider
    /// saw it: a tenant budget, a policy or moderation pre-check, a guard on
    /// the request's content.
    ///
    /// Not retryable and never fails over: the same call is refused the same
    /// way, and no provider is to blame. Distinct from
    /// [`Error::InvalidRequest`] (the request is malformed) and
    /// [`Error::QuotaExhausted`] (a provider's account is out of credit).
    /// Build it with [`Error::rejected`], and use
    /// [`with_provider_code`](Error::with_provider_code) for a reason code
    /// the application branches on.
    ///
    /// Do not set a provider on it with
    /// [`with_provider`](Error::with_provider): the provider is the one that
    /// served the call, and a refused call had none. `Display` would then
    /// read `<provider>: rejected: ...`, and a span's
    /// `allama.response.provider` would blame an upstream that was never
    /// called. Put the provider or model the policy was about in the
    /// message, or in your own error type behind
    /// [`with_source`](Error::with_source).
    ///
    /// ```
    /// use allama::Error;
    ///
    /// let err = Error::rejected("tenant budget exhausted")
    ///     .authored()
    ///     .with_provider_code("tenant_budget_exhausted");
    /// assert_eq!(err.kind_name(), "rejected");
    /// assert!(!err.is_retryable());
    /// assert!(!allama::wrappers::should_fallback(&err));
    /// ```
    #[non_exhaustive]
    Rejected {
        /// Human-readable description.
        message: String,
        /// Diagnostic context.
        #[serde(default, skip_serializing_if = "context_is_empty")]
        context: Box<ErrorContext>,
    },

    /// Rate limited: back off and retry the same provider.
    #[non_exhaustive]
    RateLimited {
        /// Human-readable description.
        message: String,
        /// Diagnostic context, including `retry_after` when reported.
        #[serde(default, skip_serializing_if = "context_is_empty")]
        context: Box<ErrorContext>,
    },

    /// Quota, credit, or spend limit exhausted. Retrying will not help until
    /// the account changes.
    #[non_exhaustive]
    QuotaExhausted {
        /// Human-readable description.
        message: String,
        /// Diagnostic context.
        #[serde(default, skip_serializing_if = "context_is_empty")]
        context: Box<ErrorContext>,
    },

    /// The provider is overloaded or temporarily unavailable. Distinct from
    /// rate limiting: consider failing over to another provider.
    #[non_exhaustive]
    Overloaded {
        /// Human-readable description.
        message: String,
        /// Diagnostic context.
        #[serde(default, skip_serializing_if = "context_is_empty")]
        context: Box<ErrorContext>,
    },

    /// A provider failure outside a more specific category.
    ///
    /// `context.status` is `Some` for HTTP error responses and `None` for
    /// failures a provider reported without one, such as an error inside a
    /// stream. Retryable without a status or with a 5xx one, except `501`
    /// (Not Implemented) and `505` (HTTP Version Not Supported): the server
    /// answers those the same way every time. A `499` (the client closed the
    /// request) is one, so it is neither retried nor reported as a rejected
    /// request.
    #[non_exhaustive]
    Provider {
        /// Human-readable description.
        message: String,
        /// Diagnostic context.
        #[serde(default, skip_serializing_if = "context_is_empty")]
        context: Box<ErrorContext>,
    },

    /// The requested feature is not supported by this provider or model.
    ///
    /// It says what was refused ([`subject`](Error::unsupported_subject))
    /// and where ([`stage`](Error::unsupported_stage)). Adapters refuse
    /// before sending through [`Degradations`](crate::adapter::Degradations)
    /// (stage [`BeforeSending`](UnsupportedStage::BeforeSending)), and map a
    /// provider's own refusal with [`Error::unsupported_by_provider`]
    /// (stage [`Provider`](UnsupportedStage::Provider)).
    /// [`Error::unsupported`] states neither.
    #[non_exhaustive]
    Unsupported {
        /// Human-readable description.
        message: String,
        /// What was refused: a request field, or a content block or part,
        /// when the refusal names one. Read it with
        /// [`unsupported_subject`](Error::unsupported_subject) or
        /// [`unsupported_field`](Error::unsupported_field).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        subject: Option<Subject>,
        /// Where the refusal happened. Read it with
        /// [`unsupported_stage`](Error::unsupported_stage).
        #[serde(default, skip_serializing_if = "UnsupportedStage::is_unspecified")]
        stage: UnsupportedStage,
        /// Diagnostic context.
        #[serde(default, skip_serializing_if = "context_is_empty")]
        context: Box<ErrorContext>,
    },

    /// The request was invalid.
    #[non_exhaustive]
    InvalidRequest {
        /// Human-readable description.
        message: String,
        /// Diagnostic context.
        #[serde(default, skip_serializing_if = "context_is_empty")]
        context: Box<ErrorContext>,
    },

    /// The requested model does not exist for this provider.
    #[non_exhaustive]
    ModelNotFound {
        /// Human-readable description.
        message: String,
        /// Diagnostic context.
        #[serde(default, skip_serializing_if = "context_is_empty")]
        context: Box<ErrorContext>,
    },

    /// The provider returned a response the adapter could not interpret,
    /// such as a success response whose body is not JSON (a proxy's HTML
    /// page) or not of the expected shape.
    ///
    /// Carries the provider, the status and the request id when the
    /// response had them. Not retryable and not eligible for fallback.
    #[non_exhaustive]
    UnexpectedResponse {
        /// Human-readable description.
        message: String,
        /// Diagnostic context.
        #[serde(default, skip_serializing_if = "context_is_empty")]
        context: Box<ErrorContext>,
    },

    /// The input exceeds the model's context window. Truncate or summarize
    /// the conversation and retry.
    ///
    /// The same limit reached while the model writes is not an error: the
    /// response arrives with
    /// [`FinishReason::ContextWindow`](crate::FinishReason::ContextWindow)
    /// (or `Length`, on providers that do not tell the two apart), and the
    /// remedy is the same.
    #[non_exhaustive]
    ContextLengthExceeded {
        /// Human-readable description.
        message: String,
        /// The context or input-token limit the provider reported, if it
        /// reported one. Most providers report the model's context window;
        /// Gemini reports its input-token limit.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        context_window: Option<u64>,
        /// Diagnostic context.
        #[serde(default, skip_serializing_if = "context_is_empty")]
        context: Box<ErrorContext>,
    },

    /// Content was blocked by provider safety systems.
    #[non_exhaustive]
    ContentFiltered {
        /// Human-readable description.
        message: String,
        /// Diagnostic context.
        #[serde(default, skip_serializing_if = "context_is_empty")]
        context: Box<ErrorContext>,
    },

    /// A streaming failure.
    #[non_exhaustive]
    Stream {
        /// What went wrong.
        kind: StreamErrorKind,
        /// Human-readable description.
        message: String,
        /// Diagnostic context.
        #[serde(default, skip_serializing_if = "context_is_empty")]
        context: Box<ErrorContext>,
    },

    /// Serialization or deserialization of the caller's own data failed,
    /// such as a typed option or provider data entry.
    ///
    /// What a provider sent that cannot be decoded is
    /// [`Error::UnexpectedResponse`] instead. Build it with
    /// [`Error::serialization`].
    ///
    /// It carries no [`ErrorContext`], so it cannot record the errors a
    /// failover tried before it (see
    /// [`with_fallback_from`](Error::with_fallback_from)).
    #[non_exhaustive]
    Serialization {
        /// The underlying failure.
        error: SerializationError,
    },

    /// Structured extraction produced no value: no mode could be used, the
    /// request conflicted with the selected mode, or the output could not be
    /// used.
    ///
    /// The [`ExtractFailure`](crate::extract::ExtractFailure) carries why,
    /// the last provider response, and the pass record (calls made, summed
    /// usage). Not retryable and not eligible for fallback. Provider errors
    /// during an extraction are returned as themselves, not wrapped here;
    /// their context carries the pass record instead
    /// ([`ExtractionMetadata::from_error`](crate::extract::ExtractionMetadata::from_error)).
    ///
    /// Build it with [`Error::extract`]. Every variant is
    /// `#[non_exhaustive]`, so a struct expression does not compile
    /// (`E0639`) outside this crate. To return the error after inspecting
    /// it, match on `&result`, or bind it whole with
    /// `err @ Error::Extract { .. }`; to rebuild one you took apart, use
    /// `Error::extract(*failure)`.
    #[cfg(feature = "extract")]
    #[non_exhaustive]
    Extract {
        /// The failed extraction.
        failure: Box<crate::extract::ExtractFailure>,
    },
}

macro_rules! constructors {
    ($($(#[$doc:meta])* $fn_name:ident => $variant:ident),* $(,)?) => {
        $(
            $(#[$doc])*
            ///
            /// The message's origin is [`MessageOrigin::Unknown`], so
            /// [`ErrorLog`] leaves it out. Mark text your code wrote, with no
            /// upstream text in it, with [`authored`](Self::authored).
            #[must_use]
            pub fn $fn_name(message: impl Into<String>) -> Self {
                Self::$variant {
                    message: message.into(),
                    context: Box::default(),
                }
            }
        )*
    };
}

impl Error {
    constructors! {
        /// Build an [`Error::Timeout`].
        timeout => Timeout,
        /// Build an [`Error::Transport`].
        transport => Transport,
        /// Build an [`Error::Auth`].
        auth => Auth,
        /// Build an [`Error::Configuration`].
        configuration => Configuration,
        /// Build an [`Error::Rejected`]: a wrapper or the application
        /// refused the call before any provider saw it.
        rejected => Rejected,
        /// Build an [`Error::RateLimited`].
        rate_limited => RateLimited,
        /// Build an [`Error::QuotaExhausted`].
        quota_exhausted => QuotaExhausted,
        /// Build an [`Error::Overloaded`].
        overloaded => Overloaded,
        /// Build an [`Error::Provider`].
        ///
        /// For failures a provider reported. A wrapper or middleware must
        /// not use it for its own failures: a `Provider` error without a
        /// status is retryable and eligible for fallback, so a retry or
        /// fallback wrapper around it would repeat a call that cannot
        /// succeed. Use the variant that says what went wrong instead, such
        /// as [`Error::configuration`] for a wrapper set up wrong.
        provider => Provider,
        /// Build an [`Error::InvalidRequest`].
        invalid_request => InvalidRequest,
        /// Build an [`Error::ModelNotFound`].
        model_not_found => ModelNotFound,
        /// Build an [`Error::UnexpectedResponse`].
        unexpected_response => UnexpectedResponse,
        /// Build an [`Error::ContentFiltered`].
        content_filtered => ContentFiltered,
    }

    /// Build an [`Error::Unsupported`] that names no subject, at an
    /// [unspecified](UnsupportedStage::Unspecified) stage.
    ///
    /// The stage is unspecified, so a caller must assume the request may
    /// have been sent. Adapters refuse before sending through
    /// [`Degradations`](crate::adapter::Degradations) instead, which records the
    /// subject and the [`BeforeSending`](UnsupportedStage::BeforeSending)
    /// stage, and map an upstream refusal with
    /// [`unsupported_by_provider`](Self::unsupported_by_provider).
    ///
    /// The message's origin is [`MessageOrigin::Unknown`], so [`ErrorLog`]
    /// leaves it out. Mark text your code wrote, with no upstream text in
    /// it, with [`authored`](Self::authored).
    #[must_use]
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported {
            message: message.into(),
            subject: None,
            stage: UnsupportedStage::Unspecified,
            context: Box::default(),
        }
    }

    /// Build an [`Error::Unsupported`] for a refusal the provider itself
    /// returned: stage [`Provider`](UnsupportedStage::Provider), refusing
    /// `subject` when the adapter can name it.
    ///
    /// For upstream error mapping. Map only a reply the adapter recognizes
    /// as a refusal of something the caller asked for (a request field it
    /// resolves through its own table of wire paths, with a fixture that
    /// pins the reply); a parameter the adapter itself got wrong is an
    /// [`InvalidRequest`](Error::InvalidRequest). The message's origin is
    /// [`MessageOrigin::Unknown`], as for [`unsupported`](Self::unsupported).
    ///
    /// ```
    /// use allama::{Error, RequestField, Subject, UnsupportedStage};
    ///
    /// let err = Error::unsupported_by_provider(
    ///     Some(Subject::Field { field: RequestField::Temperature }),
    ///     "the model does not support `temperature`",
    /// )
    /// .with_status(400);
    /// assert_eq!(err.unsupported_stage(), Some(UnsupportedStage::Provider));
    /// assert_eq!(err.unsupported_field(), Some(RequestField::Temperature));
    /// ```
    #[must_use]
    pub fn unsupported_by_provider(subject: Option<Subject>, message: impl Into<String>) -> Self {
        Self::Unsupported {
            message: message.into(),
            subject,
            stage: UnsupportedStage::Provider,
            context: Box::default(),
        }
    }

    /// An [`Error::Unsupported`] refusing `subject` before sending, for
    /// [`Degradations`](crate::adapter::Degradations).
    pub(crate) fn unsupported_before_sending(
        message: impl Into<String>,
        subject: Option<Subject>,
    ) -> Self {
        Self::Unsupported {
            message: message.into(),
            subject,
            stage: UnsupportedStage::BeforeSending,
            context: Box::default(),
        }
    }

    /// Build an [`Error::ContextLengthExceeded`].
    ///
    /// The message's origin is [`MessageOrigin::Unknown`], as for
    /// [`invalid_request`](Self::invalid_request).
    #[must_use]
    pub fn context_length_exceeded(
        message: impl Into<String>,
        context_window: Option<u64>,
    ) -> Self {
        Self::ContextLengthExceeded {
            message: message.into(),
            context_window,
            context: Box::default(),
        }
    }

    /// Build an [`Error::Stream`].
    ///
    /// The message's origin is [`MessageOrigin::Unknown`], as for
    /// [`invalid_request`](Self::invalid_request).
    #[must_use]
    pub fn stream(kind: StreamErrorKind, message: impl Into<String>) -> Self {
        Self::Stream {
            kind,
            message: message.into(),
            context: Box::default(),
        }
    }

    /// Build an [`Error::Stream`] for event-protocol violations.
    #[must_use]
    pub fn stream_protocol(message: impl Into<String>) -> Self {
        Self::stream(StreamErrorKind::Protocol, message)
    }

    /// Build an [`Error::Stream`] for undecodable stream data.
    #[must_use]
    pub fn stream_decode(message: impl Into<String>) -> Self {
        Self::stream(StreamErrorKind::Decode, message)
    }

    /// Build an [`Error::Serialization`] from any compatible source.
    #[must_use]
    pub fn serialization(err: impl Into<SerializationError>) -> Self {
        Self::Serialization { error: err.into() }
    }

    /// Build an [`Error::Extract`].
    #[cfg(feature = "extract")]
    #[must_use]
    pub fn extract(failure: crate::extract::ExtractFailure) -> Self {
        Self::Extract {
            failure: Box::new(failure),
        }
    }

    /// The error tried before this one, when this error ended a failover.
    ///
    /// A [`FallbackProvider`](crate::wrappers::FallbackProvider) whose sides
    /// all fail returns the last side's error, and keeps the earlier ones
    /// on it, newest first: this is the side tried just before, and calling
    /// `fallback_from` on the result walks back to the first primary. The
    /// returned error's own variant, status and kind are the last side's,
    /// so code that matches on the error needs no special case for a
    /// failover; read the chain to report or log what failed first.
    ///
    /// `None` for an error that ended no failover, and for the variants
    /// without a context ([`Error::Serialization`], `Error::Extract`):
    /// when the last side fails with one of them, the earlier errors are
    /// dropped and the failover's `allama::fallback` event is their only
    /// record.
    ///
    /// ```
    /// use allama::Error;
    ///
    /// // A primary, then two backups: what the chain of failovers returns.
    /// let err = Error::rate_limited("HTTP 429")
    ///     .with_status(429)
    ///     .with_fallback_from(Error::overloaded("secondary"))
    ///     .with_fallback_from(Error::timeout("primary"));
    /// assert!(matches!(err, Error::RateLimited { .. }));
    /// assert_eq!(err.status(), Some(429));
    ///
    /// let tried: Vec<_> = std::iter::successors(err.fallback_from(), |e| e.fallback_from())
    ///     .map(Error::kind_name)
    ///     .collect();
    /// assert_eq!(tried, ["overloaded", "timeout"]);
    /// ```
    #[must_use]
    pub fn fallback_from(&self) -> Option<&Error> {
        self.context()?.fallback_from.0.as_deref()
    }

    /// Record `earlier` as an error tried before this one, at the tail of
    /// the [`fallback_from`](Self::fallback_from) chain: after every error
    /// this one already records, since it came before them.
    ///
    /// This is how [`FallbackProvider`](crate::wrappers::FallbackProvider)
    /// builds the error it returns, and how a third-party failover or
    /// hedging wrapper builds the same shape: return the last error it got,
    /// with each earlier one attached, newest first. `earlier`'s own chain
    /// comes with it. Nesting composes: when the last side is itself a
    /// failover whose error already records its own primary, the outer
    /// primary goes after it.
    ///
    /// Has no effect on variants without a context ([`Error::Serialization`],
    /// `Error::Extract`), and the chain ends at an error that has none.
    ///
    /// ```
    /// use allama::Error;
    ///
    /// let inner = Error::rate_limited("backup").with_fallback_from(Error::overloaded("secondary"));
    /// let outer = inner.with_fallback_from(Error::timeout("primary"));
    /// assert!(matches!(outer.fallback_from(), Some(Error::Overloaded { .. })));
    /// assert!(matches!(
    ///     outer.fallback_from().and_then(Error::fallback_from),
    ///     Some(Error::Timeout { .. })
    /// ));
    /// ```
    #[must_use]
    pub fn with_fallback_from(mut self, earlier: Error) -> Self {
        self.attach_fallback_from(earlier);
        self
    }

    fn attach_fallback_from(&mut self, earlier: Error) {
        let Some(context) = self.context_mut() else {
            return;
        };
        match &mut context.fallback_from.0 {
            Some(next) => Arc::make_mut(next).attach_fallback_from(earlier),
            slot @ None => *slot = Some(Arc::new(earlier)),
        }
    }

    /// Borrow the diagnostic context, when this variant carries one.
    #[must_use]
    pub fn context(&self) -> Option<&ErrorContext> {
        match self {
            Error::Timeout { context, .. }
            | Error::Transport { context, .. }
            | Error::Auth { context, .. }
            | Error::Configuration { context, .. }
            | Error::Rejected { context, .. }
            | Error::RateLimited { context, .. }
            | Error::QuotaExhausted { context, .. }
            | Error::Overloaded { context, .. }
            | Error::Provider { context, .. }
            | Error::Unsupported { context, .. }
            | Error::InvalidRequest { context, .. }
            | Error::ModelNotFound { context, .. }
            | Error::UnexpectedResponse { context, .. }
            | Error::ContextLengthExceeded { context, .. }
            | Error::ContentFiltered { context, .. }
            | Error::Stream { context, .. } => Some(context.as_ref()),
            Error::Serialization { .. } => None,
            #[cfg(feature = "extract")]
            Error::Extract { .. } => None,
        }
    }

    /// Mutably borrow the diagnostic context, when this variant carries one.
    ///
    /// Use it to change or clear one field, or to replace the whole context
    /// (`*context = ErrorContext::new()`, which also drops the
    /// [`fallback_from`](Self::fallback_from) chain), which
    /// [`with_context`](Self::with_context) never does because it merges:
    ///
    /// ```
    /// use allama::{Error, ErrorContext};
    ///
    /// let mut err = Error::overloaded("fallback").with_request_id("req_1");
    /// if let Some(context) = err.context_mut() {
    ///     context.request_id = Some("req_123".into());
    /// }
    /// assert_eq!(err.request_id(), Some("req_123"));
    ///
    /// if let Some(context) = err.context_mut() {
    ///     *context = ErrorContext::new();
    /// }
    /// assert_eq!(err.request_id(), None);
    /// ```
    pub fn context_mut(&mut self) -> Option<&mut ErrorContext> {
        match self {
            Error::Timeout { context, .. }
            | Error::Transport { context, .. }
            | Error::Auth { context, .. }
            | Error::Configuration { context, .. }
            | Error::Rejected { context, .. }
            | Error::RateLimited { context, .. }
            | Error::QuotaExhausted { context, .. }
            | Error::Overloaded { context, .. }
            | Error::Provider { context, .. }
            | Error::Unsupported { context, .. }
            | Error::InvalidRequest { context, .. }
            | Error::ModelNotFound { context, .. }
            | Error::UnexpectedResponse { context, .. }
            | Error::ContextLengthExceeded { context, .. }
            | Error::ContentFiltered { context, .. }
            | Error::Stream { context, .. } => Some(context.as_mut()),
            Error::Serialization { .. } => None,
            #[cfg(feature = "extract")]
            Error::Extract { .. } => None,
        }
    }

    fn map_context(mut self, f: impl FnOnce(&mut ErrorContext)) -> Self {
        if let Some(context) = self.context_mut() {
            f(context);
        }
        self
    }

    /// Merge `context` into the diagnostic context.
    ///
    /// - Each `Some` field of `context` overwrites the error's value; a
    ///   `None` field keeps it. So the order of `with_context` and the
    ///   single-field builders (`with_status`, `with_provider`, …) does not
    ///   matter, except where both set the same field: the later call wins.
    /// - [`metadata`](ErrorContext::metadata) merges by key, and the keys of
    ///   `context` win.
    /// - The body is capped at [`ErrorContext::MAX_BODY_BYTES`].
    /// - [`message_origin`](ErrorContext::message_origin) is never changed.
    ///   Only [`authored`](Self::authored) marks the constructor message as
    ///   written by the adapter, so a context built elsewhere cannot let
    ///   upstream text into [`ErrorLog`], and merging after `authored()`
    ///   keeps the mark.
    ///
    /// To clear a field or replace the whole context, write through
    /// [`context_mut`](Self::context_mut).
    ///
    /// ```
    /// use allama::{Error, ErrorContext, MessageOrigin};
    ///
    /// let mut headers = ErrorContext::new();
    /// headers.status = Some(429);
    /// headers.request_id = Some("req_1".into());
    ///
    /// let err = Error::rate_limited("HTTP 429")
    ///     .authored()
    ///     .with_provider("acme")
    ///     .with_context(headers);
    /// assert_eq!(err.status(), Some(429));
    /// assert_eq!(err.request_id(), Some("req_1"));
    /// assert_eq!(err.context().unwrap().provider.as_deref(), Some("acme"));
    /// assert_eq!(err.message_origin(), MessageOrigin::Authored);
    /// ```
    #[must_use]
    pub fn with_context(self, context: ErrorContext) -> Self {
        self.map_context(|c| c.merge(context))
    }

    /// Set the provider identity.
    #[must_use]
    pub fn with_provider(self, provider: impl Into<String>) -> Self {
        let provider = provider.into();
        self.map_context(|c| c.provider = Some(provider))
    }

    /// Set the HTTP status code.
    #[must_use]
    pub fn with_status(self, status: u16) -> Self {
        self.map_context(|c| c.status = Some(status))
    }

    /// Set the provider request ID.
    #[must_use]
    pub fn with_request_id(self, request_id: impl Into<String>) -> Self {
        let request_id = request_id.into();
        self.map_context(|c| c.request_id = Some(request_id))
    }

    /// Set the provider request ID when there is one; `None` keeps the
    /// current value. The `_opt` form of
    /// [`with_request_id`](Self::with_request_id).
    #[must_use]
    pub fn with_request_id_opt(self, request_id: Option<&str>) -> Self {
        match request_id {
            Some(request_id) => self.with_request_id(request_id),
            None => self,
        }
    }

    /// Set the retry-after hint.
    #[must_use]
    pub fn with_retry_after(self, retry_after: Duration) -> Self {
        self.with_retry_after_opt(Some(retry_after))
    }

    /// Set the retry-after hint when there is one; `None` keeps the current
    /// value. The `_opt` form of [`with_retry_after`](Self::with_retry_after).
    #[must_use]
    pub fn with_retry_after_opt(self, retry_after: Option<Duration>) -> Self {
        self.map_context(|c| overwrite(&mut c.retry_after, retry_after))
    }

    /// Set the server's retry directive (see
    /// [`ErrorContext::server_retry_directive`]).
    #[must_use]
    pub fn with_server_retry_directive(self, directive: bool) -> Self {
        self.with_server_retry_directive_opt(Some(directive))
    }

    /// Set the server's retry directive when it sent one; `None` keeps the
    /// current value. The `_opt` form of
    /// [`with_server_retry_directive`](Self::with_server_retry_directive).
    #[must_use]
    pub fn with_server_retry_directive_opt(self, directive: Option<bool>) -> Self {
        self.map_context(|c| overwrite(&mut c.server_retry_directive, directive))
    }

    /// Set the provider error code (see [`ErrorContext::provider_code`]).
    #[must_use]
    pub fn with_provider_code(self, code: impl Into<String>) -> Self {
        let code = code.into();
        self.map_context(|c| c.provider_code = Some(code))
    }

    /// Set the provider error code when there is one; `None` keeps the
    /// current value. The `_opt` form of
    /// [`with_provider_code`](Self::with_provider_code).
    #[must_use]
    pub fn with_provider_code_opt(self, code: Option<&str>) -> Self {
        match code {
            Some(code) => self.with_provider_code(code),
            None => self,
        }
    }

    /// Set the provider error type (see [`ErrorContext::provider_type`]).
    #[must_use]
    pub fn with_provider_type(self, provider_type: impl Into<String>) -> Self {
        let provider_type = provider_type.into();
        self.map_context(|c| c.provider_type = Some(provider_type))
    }

    /// Set the provider error type when there is one; `None` keeps the
    /// current value. The `_opt` form of
    /// [`with_provider_type`](Self::with_provider_type).
    #[must_use]
    pub fn with_provider_type_opt(self, provider_type: Option<&str>) -> Self {
        match provider_type {
            Some(provider_type) => self.with_provider_type(provider_type),
            None => self,
        }
    }

    /// Set the raw response body, capped at [`ErrorContext::MAX_BODY_BYTES`].
    #[must_use]
    pub fn with_body(self, body: impl Into<String>) -> Self {
        let mut body = body.into();
        body.truncate(cap_body(&body).len());
        self.map_context(|c| c.body = Some(body))
    }

    /// Set the raw response body when there is one, capped at
    /// [`ErrorContext::MAX_BODY_BYTES`]; `None` keeps the current value.
    /// The `_opt` form of [`with_body`](Self::with_body).
    #[must_use]
    pub fn with_body_opt(self, body: Option<&str>) -> Self {
        match body {
            Some(body) => self.with_body(cap_body(body)),
            None => self,
        }
    }

    /// Set the provider's own error text, such as the `message` of a JSON
    /// error envelope. Empty text is ignored.
    ///
    /// Upstream text may echo prompts or input values, so the redacted
    /// [`ErrorLog`] leaves it out unless asked for with
    /// [`ErrorLog::with_message`]. `Display` shows it after the constructor
    /// message.
    ///
    /// ```
    /// use allama::Error;
    ///
    /// let err = Error::invalid_request("HTTP 400")
    ///     .authored()
    ///     .with_status(400)
    ///     .with_upstream_message("bad value 'my SSN is 123-45-6789'");
    /// assert_eq!(err.as_log().to_string(), r#"kind=invalid_request status=400 message="HTTP 400""#);
    /// assert!(err.to_string().contains("123-45-6789"));
    /// ```
    #[must_use]
    pub fn with_upstream_message(self, message: impl Into<String>) -> Self {
        let message = message.into();
        if message.trim().is_empty() {
            return self;
        }
        self.map_context(|c| c.upstream_message = Some(message))
    }

    /// Set the provider's own error text when there is one; `None` and
    /// empty text keep the current value. The `_opt` form of
    /// [`with_upstream_message`](Self::with_upstream_message).
    #[must_use]
    pub fn with_upstream_message_opt(self, message: Option<&str>) -> Self {
        match message {
            Some(message) => self.with_upstream_message(message),
            None => self,
        }
    }

    /// Set the path of the rejected request field (see
    /// [`ErrorContext::param`]).
    #[must_use]
    pub fn with_param(self, param: impl Into<String>) -> Self {
        let param = param.into();
        self.map_context(|c| c.param = Some(param))
    }

    /// Set the path of the rejected request field when the provider names
    /// one; `None` keeps the current value. The `_opt` form of
    /// [`with_param`](Self::with_param).
    #[must_use]
    pub fn with_param_opt(self, param: Option<&str>) -> Self {
        match param {
            Some(param) => self.with_param(param),
            None => self,
        }
    }

    /// Attach the underlying cause, returned from
    /// [`source`](std::error::Error::source): a typed error a caller may
    /// downcast, such as an I/O error behind a transport failure or a
    /// policy error behind an [`Error::Rejected`].
    ///
    /// The source is kept in memory only: it is not serialized, compared,
    /// or shown by `Debug`, `Display` or [`ErrorLog`]. `Display` does not
    /// show it, so a report chain (`anyhow`'s `{:#}`) prints it once,
    /// after the error: keep its text out of the message. Its text may
    /// hold whatever the cause's own `Display` says, such as a URL, so
    /// attach only causes that are safe to print where the chain is
    /// printed. Has no effect on variants without a context
    /// ([`Error::Serialization`], [`Error::Extract`]).
    ///
    /// ```
    /// use std::error::Error as _;
    ///
    /// use allama::Error;
    ///
    /// #[derive(Debug)]
    /// struct OverBudget {
    ///     tenant: u32,
    /// }
    ///
    /// impl std::fmt::Display for OverBudget {
    ///     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    ///         write!(f, "tenant {} is over its monthly budget", self.tenant)
    ///     }
    /// }
    ///
    /// impl std::error::Error for OverBudget {}
    ///
    /// let err = Error::rejected("budget guard").with_source(OverBudget { tenant: 7 });
    /// let cause = err.source().unwrap().downcast_ref::<OverBudget>().unwrap();
    /// assert_eq!(cause.tenant, 7);
    /// assert_eq!(err.to_string(), "rejected: budget guard");
    /// ```
    #[cfg_attr(
        not(feature = "extract"),
        doc = "[`Error::Extract`]: crate#cargo-features"
    )]
    #[must_use]
    pub fn with_source(
        self,
        source: impl Into<Box<dyn std::error::Error + Send + Sync + 'static>>,
    ) -> Self {
        let source = ErrorSource(Some(Arc::from(source.into())));
        self.map_context(|c| c.source = source)
    }

    /// Mark the constructor message as written by library or adapter code,
    /// so the redacted [`ErrorLog`] shows it.
    ///
    /// Never call this on a message that contains upstream text: a
    /// provider's error message, a response body, or the text of a foreign
    /// error (an HTTP client's error can contain the request URL, and some
    /// APIs put the key in the query string). Put provider text in
    /// [`with_upstream_message`](Self::with_upstream_message) instead.
    ///
    /// ```
    /// use allama::Error;
    ///
    /// let upstream = "Invalid value: 'my SSN is 123-45-6789'";
    ///
    /// // Bad: the upstream text would reach the redacted log.
    /// let leaky = Error::invalid_request(upstream).authored();
    /// assert!(leaky.as_log().to_string().contains("123-45-6789"));
    ///
    /// // Good: authored text in the message, upstream text in its own slot.
    /// let err = Error::invalid_request("HTTP 400")
    ///     .authored()
    ///     .with_upstream_message(upstream);
    /// assert!(!err.as_log().to_string().contains("123-45-6789"));
    /// ```
    ///
    /// Has no effect on variants without a context
    /// ([`Error::Serialization`], [`Error::Extract`]).
    ///
    #[cfg_attr(
        not(feature = "extract"),
        doc = "[`Error::Extract`]: crate#cargo-features"
    )]
    #[must_use]
    pub fn authored(self) -> Self {
        self.map_context(|c| c.message_origin = MessageOrigin::Authored)
    }

    /// Put `prefix` and `": "` in front of the message, keeping everything
    /// else: the kind, the context and the message's origin.
    ///
    /// This is how a wrapper or application adds what it was doing to an
    /// error it passes on. Re-wrapping instead (`Error::provider(format!(
    /// "summarize: {err}"))`) loses the kind and the context, and with them
    /// the retry hint and the status: a `Provider` error without a status
    /// is retried and fails over.
    ///
    /// - The prefix is shown wherever the message is. When the message is
    ///   [`Authored`](MessageOrigin::Authored), that includes the redacted
    ///   [`ErrorLog`], so use text your code wrote: never a prompt, an input
    ///   value or upstream text.
    /// - An error that ended a failover takes the prefix itself; the
    ///   errors in its [`fallback_from`](Self::fallback_from) chain keep
    ///   their messages.
    /// - [`Error::Serialization`]: the prefix goes in front of the
    ///   underlying failure's text; the failure itself stays reachable
    ///   through [`SerializationError::get_ref`].
    /// - [`Error::Extract`] is returned unchanged: its failure is
    ///   structured and carries no free-form message.
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use allama::{Error, MessageOrigin};
    ///
    /// let err = Error::rate_limited("HTTP 429")
    ///     .authored()
    ///     .with_status(429)
    ///     .with_retry_after(Duration::from_secs(2));
    ///
    /// let err = err.with_message_prefix("summarize step");
    /// assert!(matches!(err, Error::RateLimited { .. }));
    /// assert_eq!(err.message(), "summarize step: HTTP 429");
    /// assert_eq!(err.message_origin(), MessageOrigin::Authored);
    /// assert_eq!(err.retry_after(), Some(Duration::from_secs(2)));
    /// assert_eq!(
    ///     err.as_log().to_string(),
    ///     r#"kind=rate_limited status=429 message="summarize step: HTTP 429""#,
    /// );
    /// ```
    #[cfg_attr(
        not(feature = "extract"),
        doc = "[`Error::Extract`]: crate#cargo-features"
    )]
    #[must_use]
    pub fn with_message_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix_message(&prefix.into());
        self
    }

    fn prefix_message(&mut self, prefix: &str) {
        match self {
            Error::Timeout { message, .. }
            | Error::Transport { message, .. }
            | Error::Auth { message, .. }
            | Error::Configuration { message, .. }
            | Error::Rejected { message, .. }
            | Error::RateLimited { message, .. }
            | Error::QuotaExhausted { message, .. }
            | Error::Overloaded { message, .. }
            | Error::Provider { message, .. }
            | Error::Unsupported { message, .. }
            | Error::InvalidRequest { message, .. }
            | Error::ModelNotFound { message, .. }
            | Error::UnexpectedResponse { message, .. }
            | Error::ContextLengthExceeded { message, .. }
            | Error::ContentFiltered { message, .. }
            | Error::Stream { message, .. } => *message = format!("{prefix}: {message}"),
            Error::Serialization { error } => error.prefix(prefix),
            #[cfg(feature = "extract")]
            Error::Extract { .. } => {}
        }
    }

    /// The constructor message, without the category prefix and without the
    /// [`upstream_message`](Self::upstream_message).
    ///
    /// Not redacted: use [`as_log`](Self::as_log) for logs and telemetry.
    #[must_use]
    pub fn message(&self) -> String {
        match self {
            Error::Timeout { message, .. }
            | Error::Transport { message, .. }
            | Error::Auth { message, .. }
            | Error::Configuration { message, .. }
            | Error::Rejected { message, .. }
            | Error::RateLimited { message, .. }
            | Error::QuotaExhausted { message, .. }
            | Error::Overloaded { message, .. }
            | Error::Provider { message, .. }
            | Error::Unsupported { message, .. }
            | Error::InvalidRequest { message, .. }
            | Error::ModelNotFound { message, .. }
            | Error::UnexpectedResponse { message, .. }
            | Error::ContextLengthExceeded { message, .. }
            | Error::ContentFiltered { message, .. }
            | Error::Stream { message, .. } => message.clone(),
            Error::Serialization { error } => error.to_string(),
            #[cfg(feature = "extract")]
            Error::Extract { failure } => failure.to_string(),
        }
    }

    /// HTTP status code, when known.
    #[must_use]
    pub fn status(&self) -> Option<u16> {
        self.context()?.status
    }

    /// Provider request ID, when known.
    #[must_use]
    pub fn request_id(&self) -> Option<&str> {
        self.context()?.request_id.as_deref()
    }

    /// Retry-after hint, when known.
    #[must_use]
    pub fn retry_after(&self) -> Option<Duration> {
        self.context()?.retry_after
    }

    /// The server's retry directive, when it sent one: `Some(true)` asks the
    /// client to retry, `Some(false)` not to (see
    /// [`ErrorContext::server_retry_directive`]).
    ///
    /// The default [`RetryPolicy`](crate::wrappers::RetryPolicy) follows it;
    /// [`is_retryable`](Self::is_retryable) does not, because it classifies
    /// by kind and status alone. A custom classifier decides for itself:
    ///
    /// ```
    /// use allama::Error;
    /// use allama::wrappers::RetryPolicy;
    ///
    /// // Retry what the server asks for, and auth failures (a rotated key).
    /// let policy = RetryPolicy::new().retry_if(|error: &Error| {
    ///     error
    ///         .server_retry_directive()
    ///         .unwrap_or_else(|| error.is_retryable() || error.is_auth())
    /// });
    /// let conflict = Error::invalid_request("HTTP 409")
    ///     .with_status(409)
    ///     .with_server_retry_directive(true);
    /// assert!(policy.should_retry(&conflict));
    /// ```
    #[must_use]
    pub fn server_retry_directive(&self) -> Option<bool> {
        self.context()?.server_retry_directive
    }

    /// The provider's own error text, when known (see
    /// [`with_upstream_message`](Self::with_upstream_message)).
    #[must_use]
    pub fn upstream_message(&self) -> Option<&str> {
        self.context()?.upstream_message.as_deref()
    }

    /// The path of the request field the provider rejected, when it named
    /// one (see [`ErrorContext::param`]).
    ///
    /// ```
    /// use allama::Error;
    ///
    /// let err = Error::invalid_request("HTTP 400")
    ///     .with_param("messages.[1].content")
    ///     .with_provider_code("invalid_value");
    /// assert_eq!(err.param(), Some("messages.[1].content"));
    /// assert_eq!(err.provider_code(), Some("invalid_value"));
    /// ```
    #[must_use]
    pub fn param(&self) -> Option<&str> {
        self.context()?.param.as_deref()
    }

    /// The provider's machine-readable error code, when it sent one (see
    /// [`ErrorContext::provider_code`]).
    #[must_use]
    pub fn provider_code(&self) -> Option<&str> {
        self.context()?.provider_code.as_deref()
    }

    /// What an [`Error::Unsupported`] refused, when the refusal names it: a
    /// request field ([`Subject::Field`]), or a content block or part
    /// ([`Subject::Content`], with the caller's indices).
    ///
    /// `None` for other variants and for refusals that name nothing.
    ///
    /// ```
    /// use allama::adapter::Degradations;
    /// use allama::{Error, Subject, UnsupportedPolicy};
    ///
    /// let degradations = Degradations::new("demo", UnsupportedPolicy::Warn);
    /// let image = Subject::Content { message: 0, block: Some(1) };
    /// let err = degradations.reject(image.clone(), "no image input");
    /// assert_eq!(err.unsupported_subject(), Some(&image));
    /// ```
    #[must_use]
    pub fn unsupported_subject(&self) -> Option<&Subject> {
        match self {
            Error::Unsupported { subject, .. } => subject.as_ref(),
            _ => None,
        }
    }

    /// The portable request field an [`Error::Unsupported`] refused: its
    /// [subject](Self::unsupported_subject) when that is a
    /// [`Subject::Field`].
    ///
    /// The field says what was refused, not whether anything was sent; read
    /// that from [`unsupported_stage`](Self::unsupported_stage).
    ///
    /// ```
    /// use allama::adapter::Degradations;
    /// use allama::{Error, RequestField, UnsupportedPolicy};
    ///
    /// let degradations = Degradations::new("demo", UnsupportedPolicy::Warn);
    /// let err = degradations.unsupported(RequestField::ResponseFormat, "no JSON Schema output");
    /// assert_eq!(err.unsupported_field(), Some(RequestField::ResponseFormat));
    /// assert_eq!(Error::unsupported("HTTP 400").unsupported_field(), None);
    /// ```
    #[must_use]
    pub fn unsupported_field(&self) -> Option<RequestField> {
        match self.unsupported_subject()? {
            Subject::Field { field } => Some(*field),
            _ => None,
        }
    }

    /// Where an [`Error::Unsupported`] was refused; `None` for other
    /// variants.
    ///
    /// [`BeforeSending`](UnsupportedStage::BeforeSending) means nothing
    /// reached the provider, so nothing was billed and failing over or
    /// changing the request costs nothing extra.
    /// [`Provider`](UnsupportedStage::Provider) means the provider refused
    /// the request it received. [`Unspecified`](UnsupportedStage::Unspecified)
    /// promises neither: treat the request as possibly sent.
    ///
    /// ```
    /// use allama::adapter::Degradations;
    /// use allama::{Error, RequestField, UnsupportedPolicy, UnsupportedStage};
    ///
    /// let degradations = Degradations::new("demo", UnsupportedPolicy::Warn);
    /// let err = degradations.unsupported(RequestField::Seed, "no seed");
    /// assert_eq!(err.unsupported_stage(), Some(UnsupportedStage::BeforeSending));
    /// assert_eq!(
    ///     Error::unsupported("HTTP 400").unsupported_stage(),
    ///     Some(UnsupportedStage::Unspecified)
    /// );
    /// assert_eq!(Error::timeout("slow").unsupported_stage(), None);
    /// ```
    #[must_use]
    pub fn unsupported_stage(&self) -> Option<UnsupportedStage> {
        match self {
            Error::Unsupported { stage, .. } => Some(*stage),
            _ => None,
        }
    }

    /// Where the constructor message came from. [`MessageOrigin::Unknown`]
    /// for variants without a context.
    #[must_use]
    pub fn message_origin(&self) -> MessageOrigin {
        self.context()
            .map_or(MessageOrigin::Unknown, |c| c.message_origin)
    }

    /// Stable snake_case name of the error kind (the variant), for logs,
    /// metrics and telemetry: `"rate_limited"` for [`Error::RateLimited`].
    ///
    /// To branch on the kind, match on the variants instead (see
    /// [`Error`]'s "Handling errors" section).
    #[doc(alias = "kind")]
    #[must_use]
    pub fn kind_name(&self) -> &'static str {
        match self {
            Error::Timeout { .. } => "timeout",
            Error::Transport { .. } => "transport",
            Error::Auth { .. } => "auth",
            Error::Configuration { .. } => "configuration",
            Error::Rejected { .. } => "rejected",
            Error::RateLimited { .. } => "rate_limited",
            Error::QuotaExhausted { .. } => "quota_exhausted",
            Error::Overloaded { .. } => "overloaded",
            Error::Provider { .. } => "provider",
            Error::Serialization { .. } => "serialization",
            Error::Unsupported { .. } => "unsupported",
            Error::InvalidRequest { .. } => "invalid_request",
            Error::ModelNotFound { .. } => "model_not_found",
            Error::UnexpectedResponse { .. } => "unexpected_response",
            Error::ContextLengthExceeded { .. } => "context_length_exceeded",
            Error::ContentFiltered { .. } => "content_filtered",
            Error::Stream { .. } => "stream",
            #[cfg(feature = "extract")]
            Error::Extract { .. } => "extract",
        }
    }

    /// Returns `true` if the error is likely transient and worth retrying
    /// against the same provider.
    ///
    /// Covers rate limits, overload, timeouts, transport failures, provider
    /// failures without a status or with a 5xx status other than `501` and
    /// `505`, and incomplete or transport-failed streams. Quota exhaustion,
    /// local refusals ([`Error::Rejected`]) and configuration errors are
    /// never retryable, and neither is a
    /// `501` (Not Implemented) or `505` (HTTP Version Not Supported): the
    /// server answers those the same way every time.
    ///
    /// Classifies by kind and status alone: it never reads the
    /// [server retry directive](Self::server_retry_directive), which the
    /// default [`RetryPolicy`](crate::wrappers::RetryPolicy) follows first.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            Error::RateLimited { .. }
            | Error::Overloaded { .. }
            | Error::Timeout { .. }
            | Error::Transport { .. } => true,
            Error::Provider { context, .. } => context
                .status
                .is_none_or(|s| s >= 500 && !matches!(s, 501 | 505)),
            Error::Stream { kind, .. } => {
                matches!(
                    kind,
                    StreamErrorKind::Incomplete | StreamErrorKind::Transport
                )
            }
            _ => false,
        }
    }

    /// Returns `true` for canonically transient errors: timeouts, rate
    /// limits, and overload. Stricter than [`is_retryable`](Self::is_retryable):
    /// transport failures are excluded because a wrong base URL fails the
    /// same way every time.
    #[must_use]
    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            Error::Timeout { .. } | Error::RateLimited { .. } | Error::Overloaded { .. }
        )
    }

    /// Returns `true` for authentication or authorization failures reported
    /// by a provider.
    ///
    /// A credential that is missing locally is an [`Error::Configuration`],
    /// not an authentication failure, and returns `false`.
    #[must_use]
    pub fn is_auth(&self) -> bool {
        matches!(self, Error::Auth { .. })
    }

    /// Borrow a redacted logging view of this error.
    #[must_use]
    pub fn as_log(&self) -> ErrorLog<'_> {
        ErrorLog::new(self)
    }
}

/// Not redacted: shows the constructor message and the
/// [`upstream_message`](Error::upstream_message), which may echo prompts or
/// input values. Use [`Error::as_log`] for logs and telemetry.
///
/// Starts with the provider when the context names one
/// ([`ErrorContext::provider`]), as in `openai: authentication error: HTTP
/// 401: Incorrect API key provided.`, so an error says where it came from
/// behind a fallback or a router. An error that ended a failover shows
/// itself only, not the errors in its
/// [`fallback_from`](Error::fallback_from) chain; [`Error::as_log`] logs
/// those.
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(provider) = self.context().and_then(|c| c.provider.as_deref()) {
            write!(f, "{provider}: ")?;
        }
        match self {
            Error::Timeout { message, .. } => write!(f, "timeout: {message}"),
            Error::Transport { message, .. } => write!(f, "transport error: {message}"),
            Error::Auth { message, .. } => write!(f, "authentication error: {message}"),
            Error::Configuration { message, .. } => {
                write!(f, "configuration error: {message}")
            }
            Error::Rejected { message, .. } => write!(f, "rejected: {message}"),
            Error::RateLimited { message, .. } => write!(f, "rate limited: {message}"),
            Error::QuotaExhausted { message, .. } => write!(f, "quota exhausted: {message}"),
            Error::Overloaded { message, .. } => write!(f, "overloaded: {message}"),
            Error::Provider { message, context } => match context.status {
                Some(status) => write!(f, "provider error ({status}): {message}"),
                None => write!(f, "provider error: {message}"),
            },
            Error::Serialization { error } => write!(f, "serialization error: {error}"),
            Error::Unsupported { message, .. } => write!(f, "unsupported: {message}"),
            Error::InvalidRequest { message, .. } => write!(f, "invalid request: {message}"),
            Error::ModelNotFound { message, .. } => write!(f, "model not found: {message}"),
            Error::UnexpectedResponse { message, .. } => {
                write!(f, "unexpected response: {message}")
            }
            Error::ContextLengthExceeded { message, .. } => {
                write!(f, "context length exceeded: {message}")
            }
            Error::ContentFiltered { message, .. } => write!(f, "content filtered: {message}"),
            Error::Stream { kind, message, .. } => {
                write!(f, "stream error ({}): {message}", kind.as_str())
            }
            #[cfg(feature = "extract")]
            Error::Extract { failure } => write!(f, "extraction error: {failure}"),
        }?;
        if let Some(upstream) = self.upstream_message() {
            write!(f, ": {upstream}")?;
        }
        Ok(())
    }
}

/// Redacted and compact: shows the variant, its constructor message, and
/// the context fields that are set, inline. The upstream message, a
/// serialization failure's text and an extraction's model output show as
/// their length (`<withheld 21 bytes; print with {err}>`), and the response
/// body as `body_len`; see [`ErrorContext`]'s `Debug`.
impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (name, message, context) = match self {
            Error::Timeout { message, context } => ("Timeout", message, context),
            Error::Transport { message, context } => ("Transport", message, context),
            Error::Auth { message, context } => ("Auth", message, context),
            Error::Configuration { message, context } => ("Configuration", message, context),
            Error::Rejected { message, context } => ("Rejected", message, context),
            Error::RateLimited { message, context } => ("RateLimited", message, context),
            Error::QuotaExhausted { message, context } => ("QuotaExhausted", message, context),
            Error::Overloaded { message, context } => ("Overloaded", message, context),
            Error::Provider { message, context } => ("Provider", message, context),
            Error::Unsupported {
                message, context, ..
            } => ("Unsupported", message, context),
            Error::InvalidRequest { message, context } => ("InvalidRequest", message, context),
            Error::ModelNotFound { message, context } => ("ModelNotFound", message, context),
            Error::UnexpectedResponse { message, context } => {
                ("UnexpectedResponse", message, context)
            }
            Error::ContextLengthExceeded {
                message, context, ..
            } => ("ContextLengthExceeded", message, context),
            Error::ContentFiltered { message, context } => ("ContentFiltered", message, context),
            Error::Stream {
                message, context, ..
            } => ("Stream", message, context),
            // A decode failure may quote the value it could not decode.
            Error::Serialization { error } => {
                return f
                    .debug_struct("Serialization")
                    .field("error", &Withheld::shown_by_display(&error.to_string()))
                    .finish();
            }
            #[cfg(feature = "extract")]
            Error::Extract { failure } => {
                return f.debug_struct("Extract").field("failure", failure).finish();
            }
        };
        let mut debug = f.debug_struct(name);
        debug.field("message", message);
        match self {
            Error::Unsupported { subject, stage, .. } => {
                if let Some(subject) = subject {
                    debug.field("subject", subject);
                }
                if !stage.is_unspecified() {
                    debug.field("stage", stage);
                }
            }
            Error::ContextLengthExceeded {
                context_window: Some(context_window),
                ..
            } => {
                debug.field("context_window", context_window);
            }
            Error::Stream { kind, .. } => {
                debug.field("kind", kind);
            }
            _ => {}
        }
        context.debug_fields(&mut debug);
        debug.finish()
    }
}

/// The source chain never repeats what `Display` shows. `Display` shows the
/// whole error, including the failure an [`Error::Serialization`] or
/// [`Error::Extract`] holds, so their `source()` is the cause *behind* what
/// is shown: the serialization failure's source, and so on. For the other
/// variants it is the cause attached with [`Error::with_source`]. The
/// errors a failover tried first ([`Error::fallback_from`]) are not causes,
/// so they are never a `source()`.
///
/// So `anyhow`'s `{:#}` and other report chains print each text once.
///
#[cfg_attr(
    not(feature = "extract"),
    doc = "[`Error::Extract`]: crate#cargo-features"
)]
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Serialization { error } => error.source(),
            #[cfg(feature = "extract")]
            Self::Extract { failure } => failure.source(),
            _ => {
                let source = self.context()?.source()?;
                Some(source)
            }
        }
    }
}

/// A `serde_json` failure as [`Error::Serialization`]: the caller's own
/// data failed to (de)serialize, such as a typed option.
///
/// This is what `?` on a `serde_json` call gives. It carries no
/// [`ErrorContext`], so the provider, status and request id are lost, and a
/// failover's earlier errors with them. An adapter that decodes what a
/// provider sent maps a failure to [`Error::UnexpectedResponse`] instead
/// (`allama_http::malformed_response` builds it with the status and body
/// size), never with `?`.
impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Self::serialization(err)
    }
}

/// Wrapper for serialization failures surfaced through [`Error`].
///
/// Keeps [`Error`] cloneable and serializable while preserving the original
/// error as the source when available.
#[derive(Debug)]
pub struct SerializationError(BoxError);

impl Clone for SerializationError {
    fn clone(&self) -> Self {
        self.to_string().into()
    }
}

impl Serialize for SerializationError {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for SerializationError {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        Ok(String::deserialize(deserializer)?.into())
    }
}

impl fmt::Display for SerializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Transparent: `Display` is the underlying failure's text, so `source()`
/// is that failure's own source. Reach the failure itself with
/// [`get_ref`](Self::get_ref).
impl std::error::Error for SerializationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0.source()
    }
}

impl From<serde_json::Error> for SerializationError {
    fn from(err: serde_json::Error) -> Self {
        Self(Box::new(err))
    }
}

impl From<String> for SerializationError {
    fn from(message: String) -> Self {
        Self(Box::new(StringError(message)))
    }
}

impl From<&str> for SerializationError {
    fn from(message: &str) -> Self {
        message.to_owned().into()
    }
}

impl SerializationError {
    /// The underlying failure, such as a [`serde_json::Error`], to downcast.
    ///
    /// ```
    /// use allama::SerializationError;
    ///
    /// let failure = serde_json::from_str::<u8>("x").unwrap_err();
    /// let error = SerializationError::from(failure);
    /// assert!(error.get_ref().downcast_ref::<serde_json::Error>().is_some());
    /// ```
    #[must_use]
    pub fn get_ref(&self) -> &(dyn std::error::Error + Send + Sync + 'static) {
        match self.0.downcast_ref::<PrefixedError>() {
            Some(prefixed) => prefixed.root(),
            None => self.0.as_ref(),
        }
    }

    /// Put `prefix` in front of the failure's text, keeping the failure as
    /// the source.
    pub(crate) fn prefix(&mut self, prefix: &str) {
        let source = std::mem::replace(&mut self.0, Box::new(StringError(String::new())));
        self.0 = Box::new(PrefixedError {
            prefix: prefix.to_owned(),
            source,
        });
    }
}

#[derive(Debug)]
struct StringError(String);

impl fmt::Display for StringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for StringError {}

/// A failure with context in front of its text, from
/// [`Error::with_message_prefix`].
#[derive(Debug)]
struct PrefixedError {
    prefix: String,
    source: BoxError,
}

impl fmt::Display for PrefixedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.prefix, self.source)
    }
}

/// Transparent, like [`SerializationError`]: `Display` shows the failure.
impl std::error::Error for PrefixedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.source()
    }
}

impl PrefixedError {
    /// The failure under every prefix.
    fn root(&self) -> &(dyn std::error::Error + Send + Sync + 'static) {
        match self.source.downcast_ref::<PrefixedError>() {
            Some(inner) => inner.root(),
            None => self.source.as_ref(),
        }
    }
}

/// A name that is not one of a configuration enum's values, from its
/// [`FromStr`](std::str::FromStr) impl (`"high".parse::<ReasoningEffort>()`).
///
/// ```
/// use allama::ReasoningEffort;
///
/// assert_eq!("xhigh".parse::<ReasoningEffort>()?, ReasoningEffort::XHigh);
///
/// let err = "hgih".parse::<ReasoningEffort>().unwrap_err();
/// assert_eq!(
///     err.to_string(),
///     r#"unknown reasoning effort "hgih"; expected one of: none, minimal, low, medium, high, xhigh"#
/// );
/// # Ok::<(), allama::ParseEnumError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ParseEnumError {
    /// The type, in words: `"reasoning effort"`.
    pub type_name: &'static str,
    /// The text that did not parse.
    pub value: String,
    /// Every accepted name, in declaration order.
    pub expected: &'static [&'static str],
}

impl ParseEnumError {
    pub(crate) fn new(
        type_name: &'static str,
        value: &str,
        expected: &'static [&'static str],
    ) -> Self {
        Self {
            type_name,
            value: value.to_owned(),
            expected,
        }
    }
}

impl fmt::Display for ParseEnumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown {} {:?}; expected one of: {}",
            self.type_name,
            self.value,
            self.expected.join(", ")
        )
    }
}

impl std::error::Error for ParseEnumError {}

/// Redacted logging and telemetry view of an [`Error`].
///
/// Unlike `Display`, `Debug` and serde serialization of `Error`, this leaves
/// out everything that may contain prompts, model output, or credentials.
/// It includes:
///
/// - the kind, provider, status, the request ID, the retry hint, and the
///   server retry directive;
/// - the provider code and type and the rejected request field
///   ([`ErrorContext::param`]), when each is a bounded token: 1 to 128
///   characters out of `A-Z a-z 0-9 _ . : [ ] -`. Any other value is shown
///   as `<redacted>`, so the log still says the provider sent one;
/// - the body's length, never its text;
/// - the constructor message only when it is
///   [`Authored`](MessageOrigin::Authored) (see [`Error::authored`]);
/// - the pass record an extraction attached under `allama.extraction` in
///   [`ErrorContext::metadata`]: usage, not input, so cost logs keep the
///   tokens an extraction spent before a provider error. The serde form has
///   the whole record; `Display` shows its mode, passes, and input and
///   output tokens;
/// - likewise the progress a chunked embedding attached under
///   `allama.embedding`: `Display` shows the inputs completed and the input
///   tokens.
///
/// It never includes the body or any other metadata, and includes the
/// [`upstream_message`](ErrorContext::upstream_message) only when asked for
/// with [`with_message`](Self::with_message). The errors a failover tried
/// before this one ([`Error::fallback_from`]) are logged under these rules
/// as a nested `fallback_from` log, newest first.
///
/// The token filter keeps prose and markup out of the code, type and
/// `param` fields, where a provider might echo a sentence of input. It
/// cannot tell a code from an input value shaped like one: a provider that
/// puts a single word, a number or an identifier from the request into one
/// of these fields gets it logged.
///
/// Serialize it for structured logs, or format it with `Display` for a
/// single `key=value` line:
///
/// ```
/// use allama::Error;
///
/// let err = Error::rate_limited("HTTP 429")
///     .authored()
///     .with_provider("openai")
///     .with_status(429)
///     .with_request_id("req_1")
///     .with_upstream_message("slow down")
///     .with_body("{\"error\":\"...\"}");
/// assert_eq!(
///     err.as_log().to_string(),
///     r#"kind=rate_limited provider=openai status=429 request_id=req_1 body_len=15 message="HTTP 429""#,
/// );
/// // Opt in to the upstream text, for example at debug level.
/// assert_eq!(
///     err.as_log().with_message(4).to_string(),
///     r#"kind=rate_limited provider=openai status=429 request_id=req_1 body_len=15 message="HTTP 429" upstream_message="slow""#,
/// );
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ErrorLog<'a> {
    error: &'a Error,
    message_chars: Option<usize>,
}

impl<'a> ErrorLog<'a> {
    /// Wrap an error in its redacted logging view.
    #[must_use]
    pub const fn new(error: &'a Error) -> Self {
        Self {
            error,
            message_chars: None,
        }
    }

    /// Also include the [`upstream_message`](ErrorContext::upstream_message),
    /// truncated to `max_chars` characters.
    ///
    /// Upstream text may echo prompts or input values, so keep this for
    /// debug-level logs. It never includes the response body, so text from a
    /// body the adapter did not recognise stays out.
    #[must_use]
    pub const fn with_message(mut self, max_chars: usize) -> Self {
        self.message_chars = Some(max_chars);
        self
    }

    /// The error referenced by this view.
    #[must_use]
    pub const fn error(self) -> &'a Error {
        self.error
    }

    /// The same view of another error, for nested errors.
    const fn of(self, error: &'a Error) -> Self {
        Self { error, ..self }
    }

    /// The constructor message, when it may be logged.
    fn message(self) -> Option<String> {
        match self.error {
            Error::Serialization { .. } => None,
            #[cfg(feature = "extract")]
            Error::Extract { failure } => Some(failure.error().log_message()),
            error => (error.message_origin() == MessageOrigin::Authored).then(|| error.message()),
        }
    }

    /// The truncated upstream message, when asked for.
    fn upstream_message(self) -> Option<&'a str> {
        let max_chars = self.message_chars?;
        let upstream = self.error.upstream_message()?;
        Some(
            upstream
                .char_indices()
                .nth(max_chars)
                .map_or(upstream, |(end, _)| &upstream[..end]),
        )
    }
}

impl Serialize for ErrorLog<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let error = self.error;
        let mut map = serde_json::Map::new();
        map.insert("type".into(), error.kind_name().into());

        insert_opt(&mut map, "message", self.message());
        insert_opt(&mut map, "upstream_message", self.upstream_message());
        if let Error::ContextLengthExceeded {
            context_window: Some(window),
            ..
        } = error
        {
            map.insert("context_window".into(), (*window).into());
        }
        if let Error::Stream { kind, .. } = error {
            map.insert("stream_kind".into(), kind.as_str().into());
        }
        if let Some(context) = error.context() {
            insert_opt(&mut map, "provider", context.provider.clone());
            insert_opt(&mut map, "status", context.status);
            insert_opt(&mut map, "request_id", context.request_id.clone());
            insert_opt(
                &mut map,
                "retry_after_secs",
                context.retry_after.map(|d| d.as_secs_f64()),
            );
            insert_opt(
                &mut map,
                "server_retry_directive",
                context.server_retry_directive,
            );
            for (key, value) in [
                ("provider_code", &context.provider_code),
                ("provider_type", &context.provider_type),
                ("param", &context.param),
            ] {
                insert_opt(&mut map, key, value.as_deref().map(log_token));
            }
            if let Some(body) = context.capped_body() {
                map.insert("body_present".into(), true.into());
                map.insert("body_len".into(), body.len().into());
            }
            for key in [EXTRACTION_METADATA_KEY, EMBEDDING_METADATA_KEY] {
                if let Some(record) = context.metadata.get(key) {
                    map.insert(key.into(), record.clone());
                }
            }
        }
        if let Some(earlier) = error.fallback_from() {
            map.insert(
                "fallback_from".into(),
                serde_json::to_value(self.of(earlier)).unwrap_or_default(),
            );
        }

        serde_json::Value::Object(map).serialize(serializer)
    }
}

impl fmt::Display for ErrorLog<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let error = self.error;
        write!(f, "kind={}", error.kind_name())?;
        if let Error::Stream { kind, .. } = error {
            write!(f, " stream_kind={}", kind.as_str())?;
        }
        if let Some(context) = error.context() {
            write_log_field(f, "provider", context.provider.as_deref())?;
            if let Some(status) = context.status {
                write!(f, " status={status}")?;
            }
            for (key, value) in [
                ("provider_code", &context.provider_code),
                ("provider_type", &context.provider_type),
                ("param", &context.param),
            ] {
                write_log_field(f, key, value.as_deref().map(log_token))?;
            }
            write_log_field(f, "request_id", context.request_id.as_deref())?;
            if let Some(body) = context.capped_body() {
                write!(f, " body_len={}", body.len())?;
            }
            if let Some(extraction) = context.metadata.get(EXTRACTION_METADATA_KEY) {
                write_extraction_fields(f, extraction)?;
            }
            if let Some(embedding) = context.metadata.get(EMBEDDING_METADATA_KEY) {
                write_embedding_fields(f, embedding)?;
            }
        }
        if let Some(message) = self.message() {
            write!(f, " message={message:?}")?;
        }
        if let Some(upstream) = self.upstream_message() {
            write!(f, " upstream_message={upstream:?}")?;
        }
        if let Some(earlier) = error.fallback_from() {
            write!(f, " fallback_from=[{}]", self.of(earlier))?;
        }
        Ok(())
    }
}

/// What [`ErrorLog`] shows of an upstream code, type or `param` that is
/// not a bounded token.
const REDACTED: &str = "<redacted>";

/// The longest code, type or `param` [`ErrorLog`] shows.
const MAX_LOG_TOKEN_LEN: usize = 128;

/// `value` when it is a bounded token (`[A-Za-z0-9_.:\[\]-]{1,128}`), or
/// [`REDACTED`]: upstream codes, types and field paths, never prose.
fn log_token(value: &str) -> &str {
    let token = (1..=MAX_LOG_TOKEN_LEN).contains(&value.len())
        && value.bytes().all(|b| {
            b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b':' | b'[' | b']' | b'-')
        });
    if token { value } else { REDACTED }
}

/// Write ` key=value` when `value` is known.
fn write_log_field(f: &mut fmt::Formatter<'_>, key: &str, value: Option<&str>) -> fmt::Result {
    match value {
        Some(value) => {
            write!(f, " {key}=")?;
            write_log_value(f, value)
        }
        None => Ok(()),
    }
}

/// Write the cost fields of an extraction's pass record: its mode, passes,
/// and summed input and output tokens.
fn write_extraction_fields(f: &mut fmt::Formatter<'_>, record: &serde_json::Value) -> fmt::Result {
    let key = EXTRACTION_METADATA_KEY;
    write_log_field(f, &format!("{key}.mode"), record["mode"].as_str())?;
    for (name, value) in [
        ("passes", &record["passes"]),
        ("input_tokens", &record["usage"]["input_tokens"]),
        ("output_tokens", &record["usage"]["output_tokens"]),
    ] {
        if let Some(value) = value.as_u64() {
            write!(f, " {key}.{name}={value}")?;
        }
    }
    Ok(())
}

/// Write the cost fields of a chunked embedding's progress: the inputs
/// completed and the summed input tokens.
fn write_embedding_fields(f: &mut fmt::Formatter<'_>, record: &serde_json::Value) -> fmt::Result {
    let key = EMBEDDING_METADATA_KEY;
    for (name, value) in [
        ("completed_inputs", &record["completed_inputs"]),
        ("input_tokens", &record["usage"]["input_tokens"]),
    ] {
        if let Some(value) = value.as_u64() {
            write!(f, " {key}.{name}={value}")?;
        }
    }
    Ok(())
}

/// Write a value bare when it is a simple token, quoted and escaped otherwise,
/// so the rendering stays on one line.
fn write_log_value(f: &mut fmt::Formatter<'_>, value: &str) -> fmt::Result {
    let bare = !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_graphic() && c != '"' && c != '=');
    if bare {
        f.write_str(value)
    } else {
        write!(f, "{value:?}")
    }
}

/// `body` truncated to [`ErrorContext::MAX_BODY_BYTES`] at a character
/// boundary.
fn cap_body(body: &str) -> &str {
    &body[..body.floor_char_boundary(ErrorContext::MAX_BODY_BYTES)]
}

fn serialize_capped_body<S: Serializer>(
    body: &Option<String>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    body.as_deref().map(cap_body).serialize(serializer)
}

/// Set `slot` when `value` is `Some`; keep it otherwise.
fn overwrite<T>(slot: &mut Option<T>, value: Option<T>) {
    if value.is_some() {
        *slot = value;
    }
}

fn insert_opt<T: Into<serde_json::Value>>(
    map: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    value: Option<T>,
) {
    if let Some(value) = value {
        map.insert(key.into(), value.into());
    }
}

mod option_duration_secs_f64 {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(
        value: &Option<Duration>,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        value.map(|d| d.as_secs_f64()).serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Option<Duration>, D::Error> {
        // `try_from`: a stored value too large for a `Duration` (such as
        // `1e20`) reads as no hint instead of panicking.
        Ok(Option::<f64>::deserialize(deserializer)?
            .and_then(|secs| Duration::try_from_secs_f64(secs).ok()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_names_the_provider_of_each_error() {
        let auth = Error::auth("HTTP 401")
            .with_provider("openai")
            .with_upstream_message("Incorrect API key provided.");
        assert_eq!(
            auth.to_string(),
            "openai: authentication error: HTTP 401: Incorrect API key provided."
        );
        assert_eq!(
            Error::rate_limited("HTTP 429").to_string(),
            "rate limited: HTTP 429"
        );
        // An error that ended a failover shows itself only.
        let failover =
            auth.with_fallback_from(Error::rate_limited("HTTP 429").with_provider("anthropic"));
        assert_eq!(
            failover.to_string(),
            "openai: authentication error: HTTP 401: Incorrect API key provided."
        );
    }

    #[test]
    fn transport_errors_are_retryable_but_not_transient() {
        let err = Error::transport("connection refused").with_provider("demo");
        assert_eq!(err.kind_name(), "transport");
        assert!(err.is_retryable());
        assert!(!err.is_transient());
        assert_eq!(err.to_string(), "demo: transport error: connection refused");
        let round_trip: Error =
            serde_json::from_value(serde_json::to_value(&err).unwrap()).unwrap();
        assert!(matches!(round_trip, Error::Transport { .. }));
        assert_eq!(
            round_trip.context().unwrap().provider.as_deref(),
            Some("demo")
        );
    }

    #[test]
    fn display_and_retryability() {
        let cases = [
            (Error::timeout("slow"), "timeout: slow", true),
            (
                Error::auth("bad key"),
                "authentication error: bad key",
                false,
            ),
            (
                Error::configuration("OPENAI_API_KEY not set"),
                "configuration error: OPENAI_API_KEY not set",
                false,
            ),
            (Error::rate_limited("busy"), "rate limited: busy", true),
            (
                Error::quota_exhausted("no credit"),
                "quota exhausted: no credit",
                false,
            ),
            (Error::overloaded("down"), "overloaded: down", true),
            (Error::provider("reset"), "provider error: reset", true),
            (
                Error::provider("boom").with_status(500),
                "provider error (500): boom",
                true,
            ),
            (
                Error::provider("bad").with_status(400),
                "provider error (400): bad",
                false,
            ),
            (Error::unsupported("x"), "unsupported: x", false),
            (Error::invalid_request("x"), "invalid request: x", false),
            (Error::model_not_found("x"), "model not found: x", false),
            (
                Error::unexpected_response("x"),
                "unexpected response: x",
                false,
            ),
            (Error::content_filtered("x"), "content filtered: x", false),
            (
                Error::stream(StreamErrorKind::Incomplete, "cut"),
                "stream error (incomplete): cut",
                true,
            ),
            (
                Error::stream(StreamErrorKind::Transport, "reset"),
                "stream error (transport): reset",
                true,
            ),
            (
                Error::stream_decode("bad json"),
                "stream error (decode): bad json",
                false,
            ),
        ];
        for (err, display, retryable) in cases {
            assert_eq!(err.to_string(), display);
            assert_eq!(err.is_retryable(), retryable, "{display}");
        }
    }

    /// A 5xx is retried, except the two statuses a server answers the same
    /// way every time: `501` (Ollama's "does not support embeddings" for a
    /// chat-only model) and `505`.
    #[test]
    fn server_errors_are_retryable_except_501_and_505() {
        for (status, retryable) in [(500, true), (502, true), (501, false), (505, false)] {
            let err = Error::provider(format!("HTTP {status}")).with_status(status);
            assert_eq!(err.is_retryable(), retryable, "HTTP {status}");
        }
    }

    #[test]
    fn builders_fill_context() {
        let err = Error::rate_limited("slow down")
            .with_status(429)
            .with_request_id("req_1")
            .with_retry_after(Duration::from_secs(2))
            .with_provider_code("rate_limit_exceeded")
            .with_provider("openai");
        assert_eq!(err.status(), Some(429));
        assert_eq!(err.request_id(), Some("req_1"));
        assert_eq!(err.retry_after(), Some(Duration::from_secs(2)));
        assert_eq!(
            err.context().unwrap().provider_code.as_deref(),
            Some("rate_limit_exceeded")
        );
    }

    #[test]
    fn server_retry_directive_is_context() {
        let err = Error::rate_limited("HTTP 429").with_server_retry_directive(false);
        assert_eq!(err.server_retry_directive(), Some(false));
        // `None` keeps what is there, like the other `with_*` builders.
        let err = err.with_server_retry_directive_opt(None);
        assert_eq!(err.server_retry_directive(), Some(false));
        // It does not change the kind-and-status classification.
        assert!(err.is_retryable());
        assert_eq!(Error::provider("x").server_retry_directive(), None);

        let round_trip: Error =
            serde_json::from_str(&serde_json::to_string(&err).unwrap()).unwrap();
        assert_eq!(round_trip.server_retry_directive(), Some(false));
        let log = serde_json::to_value(err.as_log()).unwrap();
        assert_eq!(log["server_retry_directive"], false);
        assert!(format!("{:?}", err.context().unwrap()).contains("server_retry_directive"));
        // Older serialized errors have no directive.
        let old: Error =
            serde_json::from_str(r#"{"type":"rate_limited","data":{"message":"m"}}"#).unwrap();
        assert_eq!(old.server_retry_directive(), None);
    }

    /// The kinds along an error's `fallback_from` chain, newest first.
    fn tried(error: &Error) -> Vec<&'static str> {
        std::iter::successors(error.fallback_from(), |e| e.fallback_from())
            .map(Error::kind_name)
            .collect()
    }

    #[test]
    fn an_error_that_ended_a_failover_is_itself_and_logs_the_chain() {
        let err = Error::rate_limited("fallback")
            .with_request_id("r")
            .with_fallback_from(Error::timeout("primary").with_provider("a"));
        assert!(matches!(err, Error::RateLimited { .. }));
        assert_eq!(err.kind_name(), "rate_limited");
        assert!(err.is_retryable());
        assert_eq!(err.request_id(), Some("r"));
        assert_eq!(tried(&err), ["timeout"]);
        let log = serde_json::to_value(err.as_log()).unwrap();
        assert_eq!(log["type"], "rate_limited");
        assert_eq!(log["request_id"], "r");
        assert_eq!(log["fallback_from"]["type"], "timeout");
        assert_eq!(log["fallback_from"]["provider"], "a");
        assert!(log["fallback_from"].get("fallback_from").is_none());
    }

    #[test]
    fn with_fallback_from_attaches_at_the_tail() {
        let last = || {
            Error::invalid_request("HTTP 400")
                .with_status(400)
                .with_param("messages.[1].content")
                .with_provider_code("invalid_value")
        };
        assert_eq!(last().fallback_from().map(Error::kind_name), None);

        // Each call adds an older error after the ones already recorded.
        let chain = last()
            .with_fallback_from(Error::rate_limited("c"))
            .with_fallback_from(Error::overloaded("b").with_fallback_from(Error::timeout("a")));
        assert_eq!(chain.kind_name(), "invalid_request");
        assert_eq!(tried(&chain), ["rate_limited", "overloaded", "timeout"]);
        // The context accessors read the error itself.
        assert_eq!(chain.param(), Some("messages.[1].content"));
        assert_eq!(chain.provider_code(), Some("invalid_value"));
        assert_eq!(chain.status(), Some(400));
        assert!(!chain.is_retryable());

        // A clone shares the chain, and attaching to it leaves the
        // original alone.
        let shared = chain.clone();
        let longer = shared.with_fallback_from(Error::auth("first"));
        assert_eq!(tried(&chain), ["rate_limited", "overloaded", "timeout"]);
        assert_eq!(
            tried(&longer),
            ["rate_limited", "overloaded", "timeout", "auth"]
        );

        // Without a context, there is no chain to keep.
        let plain = Error::serialization("bad json").with_fallback_from(Error::timeout("a"));
        assert_eq!(plain.param(), None);
        assert_eq!(plain.provider_code(), None);
        assert!(plain.fallback_from().is_none());
        // A chain ends at an error without a context.
        let ends = Error::timeout("b")
            .with_fallback_from(Error::serialization("bad json"))
            .with_fallback_from(Error::overloaded("a"));
        assert_eq!(tried(&ends), ["serialization"]);
    }

    #[test]
    fn the_chain_is_not_compared_and_merges_only_when_set() {
        let plain = Error::rate_limited("HTTP 429");
        let chained = plain.clone().with_fallback_from(Error::timeout("a"));
        // `ErrorContext` keeps `Eq`: the chain is not part of it.
        assert_eq!(plain.context(), chained.context());
        assert!(!chained.context().unwrap().is_empty());

        // Merging a context without a chain keeps the chain…
        let kept = chained.clone().with_context(ErrorContext::new());
        assert_eq!(tried(&kept), ["timeout"]);
        // …and one with a chain replaces it.
        let other = Error::overloaded("x").with_fallback_from(Error::auth("b"));
        let replaced = chained.with_context(other.context().unwrap().clone());
        assert_eq!(tried(&replaced), ["auth"]);
    }

    #[test]
    fn the_chain_survives_serde_under_fallback_from() {
        let err = Error::rate_limited("HTTP 429")
            .with_status(429)
            .with_fallback_from(
                Error::overloaded("HTTP 529")
                    .with_provider("anthropic")
                    .with_fallback_from(Error::timeout("slow")),
            );
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "type": "rate_limited",
                "data": {
                    "message": "HTTP 429",
                    "context": {
                        "status": 429,
                        "fallback_from": {
                            "type": "overloaded",
                            "data": {
                                "message": "HTTP 529",
                                "context": {
                                    "provider": "anthropic",
                                    "fallback_from": {
                                        "type": "timeout",
                                        "data": {"message": "slow"}
                                    }
                                }
                            }
                        }
                    }
                }
            })
        );
        let back: Error = serde_json::from_value(json.clone()).unwrap();
        assert!(matches!(back, Error::RateLimited { .. }));
        assert_eq!(tried(&back), ["overloaded", "timeout"]);
        assert_eq!(
            back.fallback_from()
                .unwrap()
                .context()
                .unwrap()
                .provider
                .as_deref(),
            Some("anthropic")
        );
        assert_eq!(serde_json::to_value(&back).unwrap(), json);
        // A chain alone keeps the context in the serialized form.
        let only_chain = Error::timeout("b").with_fallback_from(Error::timeout("a"));
        let json = serde_json::to_value(&only_chain).unwrap();
        assert_eq!(json["data"]["context"]["fallback_from"]["type"], "timeout");
    }

    #[test]
    fn with_context_merges_and_never_touches_the_message_origin() {
        let mut context = ErrorContext::new();
        context.status = Some(429);
        context.request_id = Some("req_new".into());
        context
            .metadata
            .insert("b".into(), serde_json::json!("new"));
        context.metadata.insert("c".into(), serde_json::json!(3));
        context.message_origin = MessageOrigin::Authored;

        let mut existing = ErrorContext::new();
        existing.provider = Some("acme".into());
        existing.request_id = Some("req_old".into());
        existing.upstream_message = Some("slow down".into());
        existing.metadata.insert("a".into(), serde_json::json!(1));
        existing
            .metadata
            .insert("b".into(), serde_json::json!("old"));
        let mut err = Error::rate_limited("HTTP 429");
        *err.context_mut().unwrap() = existing;

        let merged = err.with_context(context.clone());
        let c = merged.context().unwrap();
        // `None` keeps, `Some` overwrites.
        assert_eq!(c.provider.as_deref(), Some("acme"));
        assert_eq!(c.upstream_message.as_deref(), Some("slow down"));
        assert_eq!(c.status, Some(429));
        assert_eq!(c.request_id.as_deref(), Some("req_new"));
        // Metadata merges by key; the argument wins.
        assert_eq!(c.metadata["a"], serde_json::json!(1));
        assert_eq!(c.metadata["b"], serde_json::json!("new"));
        assert_eq!(c.metadata["c"], serde_json::json!(3));
        // An authored origin in the argument is not promoted…
        assert_eq!(merged.message_origin(), MessageOrigin::Unknown);
        // …and an unknown one does not demote.
        let kept = Error::rate_limited("HTTP 429")
            .authored()
            .with_context(ErrorContext::new());
        assert_eq!(kept.message_origin(), MessageOrigin::Authored);

        // Slots set before or after the call give the same result.
        let before = Error::rate_limited("HTTP 429")
            .with_provider("acme")
            .authored()
            .with_context(context.clone());
        let after = Error::rate_limited("HTTP 429")
            .with_context(context)
            .with_provider("acme")
            .authored();
        assert_eq!(before.context(), after.context());
    }

    #[test]
    fn context_mut_changes_the_error_itself_not_the_chain() {
        let mut chain = Error::rate_limited("c")
            .with_fallback_from(Error::overloaded("b"))
            .with_fallback_from(Error::timeout("a"));
        chain.context_mut().unwrap().provider_code = Some("rate_limit_exceeded".into());
        assert_eq!(chain.provider_code(), Some("rate_limit_exceeded"));
        assert_eq!(chain.fallback_from().unwrap().provider_code(), None);

        // Replacing the whole context drops the chain.
        *chain.context_mut().unwrap() = ErrorContext::new();
        assert!(chain.fallback_from().is_none());
    }

    #[test]
    fn log_redacts_body() {
        let err = Error::provider("bad").with_status(400).with_body("secret");
        let log = serde_json::to_value(err.as_log()).unwrap();
        assert_eq!(log["body_present"], true);
        assert_eq!(log["body_len"], 6);
        assert!(log.get("body").is_none());
    }

    #[test]
    fn log_display_is_one_redacted_line() {
        let err = Error::provider("bad\nrequest")
            .authored()
            .with_provider("openai-compat:groq")
            .with_status(400)
            .with_provider_code("invalid_value")
            .with_param("messages.[1].content")
            .with_request_id("req 1")
            .with_body("secret prompt");
        let line = err.as_log().to_string();
        assert_eq!(
            line,
            r#"kind=provider provider=openai-compat:groq status=400 provider_code=invalid_value param=messages.[1].content request_id="req 1" body_len=13 message="bad\nrequest""#
        );
        assert!(!line.contains("secret"));
        assert!(!line.contains('\n'));

        assert_eq!(
            Error::configuration("OPENAI_API_KEY not set")
                .authored()
                .as_log()
                .to_string(),
            r#"kind=configuration message="OPENAI_API_KEY not set""#
        );
    }

    #[test]
    fn log_omits_messages_of_unknown_origin() {
        let err = Error::invalid_request("echoed SENTINEL")
            .with_status(400)
            .with_body("SENTINEL");
        assert_eq!(err.message_origin(), MessageOrigin::Unknown);
        let line = err.as_log().with_message(100).to_string();
        assert_eq!(line, "kind=invalid_request status=400 body_len=8");
        let json = serde_json::to_value(err.as_log().with_message(100)).unwrap();
        assert!(json.get("message").is_none(), "{json}");
        assert!(!json.to_string().contains("SENTINEL"), "{json}");
    }

    #[test]
    fn upstream_message_is_opt_in_and_truncated_at_a_char_boundary() {
        let err = Error::invalid_request("HTTP 400")
            .authored()
            .with_status(400)
            .with_upstream_message("héllo wörld");
        assert_eq!(err.upstream_message(), Some("héllo wörld"));

        let redacted = err.as_log().to_string();
        assert!(!redacted.contains("héllo"), "{redacted}");
        assert!(
            serde_json::to_value(err.as_log())
                .unwrap()
                .get("upstream_message")
                .is_none()
        );

        assert_eq!(
            err.as_log().with_message(2).to_string(),
            r#"kind=invalid_request status=400 message="HTTP 400" upstream_message="hé""#
        );
        let json = serde_json::to_value(err.as_log().with_message(100)).unwrap();
        assert_eq!(json["upstream_message"], "héllo wörld");
        assert_eq!(json["message"], "HTTP 400");
    }

    #[test]
    fn display_is_not_redacted() {
        let err = Error::invalid_request("HTTP 400")
            .authored()
            .with_upstream_message("bad value SENTINEL");
        assert_eq!(
            err.to_string(),
            "invalid request: HTTP 400: bad value SENTINEL"
        );
        assert_eq!(err.message(), "HTTP 400");

        // Empty upstream text is ignored.
        let err = Error::timeout("slow").with_upstream_message("  ");
        assert_eq!(err.upstream_message(), None);
        assert_eq!(err.to_string(), "timeout: slow");

        // A failover's earlier errors are not shown.
        let failover = Error::rate_limited("f")
            .with_upstream_message("slow down")
            .with_fallback_from(Error::overloaded("p").with_upstream_message("busy"));
        assert_eq!(failover.to_string(), "rate limited: f: slow down");
    }

    #[test]
    fn provenance_round_trips_and_unknown_is_not_serialized() {
        let unknown = serde_json::to_value(Error::timeout("x").with_status(408)).unwrap();
        assert_eq!(
            unknown,
            serde_json::json!({"type": "timeout", "data": {"message": "x", "context": {"status": 408}}})
        );

        let err = Error::invalid_request("HTTP 400")
            .authored()
            .with_param("tools[0].function.parameters")
            .with_upstream_message("bad schema");
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["data"]["context"]["message_origin"], "authored");
        let back: Error = serde_json::from_value(json).unwrap();
        assert_eq!(back.message_origin(), MessageOrigin::Authored);
        assert_eq!(back.upstream_message(), Some("bad schema"));
        assert_eq!(
            back.context().unwrap().param.as_deref(),
            Some("tools[0].function.parameters")
        );
    }

    #[test]
    fn body_is_capped_in_with_body_serialization_and_debug() {
        // A two-byte character straddles the cap.
        let long = format!(
            "{}é{}",
            "a".repeat(ErrorContext::MAX_BODY_BYTES - 1),
            "b".repeat(100)
        );
        let err = Error::provider("x").with_body(long.clone());
        let body = err.context().unwrap().body.as_deref().unwrap();
        assert_eq!(body.len(), ErrorContext::MAX_BODY_BYTES - 1);
        assert!(body.chars().all(|c| c == 'a'));
        assert_eq!(
            serde_json::to_value(err.as_log()).unwrap()["body_len"],
            ErrorContext::MAX_BODY_BYTES - 1
        );

        let mut context = ErrorContext::new();
        context.body = Some(long.clone());
        assert_eq!(
            Error::provider("x")
                .with_context(context.clone())
                .context()
                .unwrap()
                .body
                .as_deref()
                .unwrap()
                .len(),
            ErrorContext::MAX_BODY_BYTES - 1
        );

        // Written directly: capped when serialized and in `Debug`.
        let mut direct = Error::provider("x");
        direct.context_mut().unwrap().body = Some(long.clone());
        let json = serde_json::to_value(&direct).unwrap();
        assert_eq!(
            json["data"]["context"]["body"].as_str().unwrap().len(),
            ErrorContext::MAX_BODY_BYTES - 1
        );
        let debug = format!("{direct:?}");
        assert!(!debug.contains("aaaa"), "{debug}");
        assert!(
            debug.contains(&format!("body_len: {}", ErrorContext::MAX_BODY_BYTES - 1)),
            "{debug}"
        );
    }

    #[test]
    fn debug_shows_the_body_length_only() {
        let err = Error::provider("x").with_body("my SSN is 123-45-6789");
        let debug = format!("{err:?}");
        assert!(!debug.contains("123-45-6789"), "{debug}");
        assert!(debug.contains("body_len: 21"), "{debug}");
    }

    #[test]
    fn debug_shows_the_upstream_message_length_only() {
        let err = Error::invalid_request("HTTP 400")
            .authored()
            .with_upstream_message("my SSN is 123-45-6789");
        let debug = format!("{err:?}");
        assert!(!debug.contains("123-45-6789"), "{debug}");
        assert!(
            debug.contains("upstream_message: <withheld 21 bytes; print with {err}>"),
            "{debug}"
        );
        let none = format!("{:?}", Error::invalid_request("HTTP 400"));
        assert!(!none.contains("upstream_message"), "{none}");
        // `Display` is not redacted.
        assert!(err.to_string().contains("123-45-6789"));
    }

    #[test]
    fn debug_is_compact_and_omits_unset_fields() {
        let err = Error::invalid_request("HTTP 400")
            .with_provider("openai")
            .with_status(400)
            .with_provider_code("invalid_value")
            .with_param("messages.[1].content")
            .with_request_id("req_1")
            .with_upstream_message("my SSN is 123-45-6789");
        assert_eq!(
            format!("{err:?}"),
            r#"InvalidRequest { message: "HTTP 400", provider: "openai", status: 400, provider_code: "invalid_value", param: "messages.[1].content", request_id: "req_1", upstream_message: <withheld 21 bytes; print with {err}> }"#
        );
        assert_eq!(
            format!("{:?}", Error::timeout("deadline")),
            r#"Timeout { message: "deadline" }"#
        );
        assert_eq!(format!("{:?}", ErrorContext::new()), "ErrorContext");

        let retry = Error::rate_limited("HTTP 429")
            .authored()
            .with_retry_after(Duration::from_millis(1500))
            .with_server_retry_directive(true);
        assert_eq!(
            format!("{retry:?}"),
            r#"RateLimited { message: "HTTP 429", retry_after: 1.5s, server_retry_directive: true, message_origin: Authored }"#
        );
        assert_eq!(
            format!("{:?}", retry.context().unwrap()),
            "ErrorContext { retry_after: 1.5s, server_retry_directive: true, message_origin: Authored }"
        );
    }

    #[test]
    fn debug_shows_variant_fields_and_metadata_keys() {
        let unsupported = Error::unsupported_by_provider(
            Some(Subject::Field {
                field: RequestField::Seed,
            }),
            "no seed",
        );
        assert_eq!(
            format!("{unsupported:?}"),
            r#"Unsupported { message: "no seed", subject: Field { field: Seed }, stage: Provider }"#
        );
        assert_eq!(
            format!("{:?}", Error::unsupported("no seed")),
            r#"Unsupported { message: "no seed" }"#
        );
        assert_eq!(
            format!(
                "{:?}",
                Error::context_length_exceeded("too long", Some(8192))
            ),
            r#"ContextLengthExceeded { message: "too long", context_window: 8192 }"#
        );
        assert_eq!(
            format!("{:?}", Error::stream_protocol("bad order")),
            r#"Stream { message: "bad order", kind: Protocol }"#
        );

        let mut err = Error::stream(StreamErrorKind::Incomplete, "cut off");
        err.context_mut()
            .unwrap()
            .metadata
            .insert("vendor.position".into(), serde_json::json!("SENTINEL"));
        let debug = format!("{err:?}");
        assert!(
            debug.contains(r#"metadata: ["vendor.position"]"#),
            "{debug}"
        );
        assert!(!debug.contains("SENTINEL"), "{debug}");
    }

    #[test]
    fn pretty_debug_puts_each_field_on_its_own_line() {
        let err = Error::provider("HTTP 500")
            .with_status(500)
            .with_upstream_message("SENTINEL");
        assert_eq!(
            format!("{err:#?}"),
            "Provider {\n    message: \"HTTP 500\",\n    status: 500,\n    \
             upstream_message: <withheld 8 bytes; print with {err}>,\n}"
        );
    }

    #[test]
    fn debug_never_shows_upstream_text_through_a_fallback() {
        let leaf = |message: &str| {
            Error::overloaded("HTTP 529")
                .with_upstream_message(format!("UPSTREAM {message}"))
                .with_body(format!("BODY {message}"))
        };
        let err = leaf("three")
            .with_fallback_from(leaf("two"))
            .with_fallback_from(leaf("one"))
            .with_fallback_from(Error::serialization(
                "invalid type: string \"SERDE\", expected u64",
            ));
        for debug in [
            format!("{err:?}"),
            format!("{err:#?}"),
            format!("{:?}", err.fallback_from().unwrap()),
            format!("{:?}", err.context().unwrap()),
        ] {
            for secret in ["UPSTREAM", "BODY", "SERDE"] {
                assert!(!debug.contains(secret), "{secret}: {debug}");
            }
        }
        let debug = format!("{err:?}");
        assert!(
            debug.starts_with(
                "Overloaded { message: \"HTTP 529\", upstream_message: <withheld 14 bytes; \
                 print with {err}>, body_len: 10, fallback_from: Overloaded {"
            ),
            "{debug}"
        );
        assert!(
            debug.contains("Serialization { error: <withheld 42 bytes; print with {err}> }"),
            "{debug}"
        );
        // `Display` is not redacted, and shows the error itself only.
        assert!(err.to_string().contains("UPSTREAM three"));
        assert!(!err.to_string().contains("UPSTREAM two"));
    }

    #[test]
    fn log_shows_codes_types_and_params_only_as_bounded_tokens() {
        let tokens = Error::invalid_request("HTTP 400")
            .authored()
            .with_provider_code("context_length_exceeded")
            .with_provider_type("invalid_request_error")
            .with_param("tools[0].function.parameters");
        assert_eq!(
            tokens.as_log().to_string(),
            "kind=invalid_request provider_code=context_length_exceeded \
             provider_type=invalid_request_error param=tools[0].function.parameters \
             message=\"HTTP 400\""
        );

        // Prose, markup, an empty value or one past 128 characters: the log
        // says a value existed, not what it was.
        for value in [
            "my SSN is 123-45-6789",
            "<b>bad</b>",
            "a,b",
            "",
            &"a".repeat(129),
        ] {
            let err = Error::invalid_request("HTTP 400")
                .authored()
                .with_provider_code(value.to_owned())
                .with_provider_type(value.to_owned())
                .with_param(value.to_owned());
            assert_eq!(
                err.as_log().to_string(),
                "kind=invalid_request provider_code=<redacted> provider_type=<redacted> \
                 param=<redacted> message=\"HTTP 400\"",
                "{value:?}"
            );
            let json = serde_json::to_value(err.as_log()).unwrap();
            for key in ["provider_code", "provider_type", "param"] {
                assert_eq!(json[key], REDACTED, "{value:?}");
            }
            let debug = format!("{err:?}");
            assert!(debug.contains(r#"provider_type: "<redacted>""#), "{debug}");
            if !value.is_empty() {
                assert!(!debug.contains(value), "{debug}");
            }
        }
        // 128 characters still pass.
        let long = "a".repeat(128);
        let err = Error::provider("x").with_provider_code(long.clone());
        assert!(err.as_log().to_string().contains(&long));
    }

    /// The filter's limit, pinned: it keeps prose out, but an input value
    /// shaped like a code (one word, a number, an identifier) passes.
    #[test]
    fn log_cannot_tell_a_code_from_a_value_shaped_like_one() {
        for echoed in ["123-45-6789", "alice", "user_42:secret"] {
            let err = Error::invalid_request("HTTP 400")
                .authored()
                .with_param(echoed);
            assert!(err.as_log().to_string().contains(echoed), "{echoed}");
        }
    }

    #[test]
    fn log_renders_the_fallback_chain_as_nested_redacted_logs() {
        let err = Error::rate_limited("f")
            .with_status(429)
            .with_upstream_message("upstream f")
            .with_fallback_from(
                Error::overloaded("q")
                    .authored()
                    .with_body("BODY q")
                    .with_upstream_message("upstream q"),
            )
            .with_fallback_from(
                Error::timeout("p")
                    .authored()
                    .with_provider("a")
                    .with_upstream_message("upstream p"),
            );
        assert_eq!(
            err.as_log().to_string(),
            r#"kind=rate_limited status=429 fallback_from=[kind=overloaded body_len=6 message="q" fallback_from=[kind=timeout provider=a message="p"]]"#
        );
        assert_eq!(
            err.as_log().with_message(8).to_string(),
            r#"kind=rate_limited status=429 upstream_message="upstream" fallback_from=[kind=overloaded body_len=6 message="q" upstream_message="upstream" fallback_from=[kind=timeout provider=a message="p" upstream_message="upstream"]]"#
        );
        let json = serde_json::to_value(err.as_log()).unwrap();
        assert_eq!(json["type"], "rate_limited");
        assert!(json.get("message").is_none(), "{json}");
        assert!(json.get("upstream_message").is_none(), "{json}");
        let q = &json["fallback_from"];
        assert_eq!(q["type"], "overloaded");
        assert_eq!(q["message"], "q");
        assert_eq!(q["body_len"], 6);
        assert!(q.get("body").is_none(), "{json}");
        assert!(q.get("upstream_message").is_none(), "{json}");
        assert_eq!(q["fallback_from"]["type"], "timeout");
        assert_eq!(q["fallback_from"]["message"], "p");
        assert_eq!(q["fallback_from"]["provider"], "a");
        assert!(!json.to_string().contains("BODY"), "{json}");
    }

    #[test]
    fn serialization_messages_are_not_logged() {
        let err = Error::serialization("invalid type: string \"SENTINEL\", expected u64");
        assert_eq!(err.as_log().to_string(), "kind=serialization");
    }

    #[cfg(feature = "extract")]
    #[test]
    fn extraction_logs_omit_parse_detail() {
        use crate::extract::{ExtractError, ExtractFailure, ExtractionMetadata, ExtractionMode};

        let parse = ExtractError::Parse {
            mode: ExtractionMode::Native,
            provider: "demo".into(),
            raw: "{\"ssn\": \"SENTINEL\"}".into(),
            message: "\"SENTINEL\" is not of type integer".into(),
            path: Some("SENTINEL_KEY".into()),
        };
        let metadata = ExtractionMetadata::new(ExtractionMode::Native);
        let err = Error::extract(ExtractFailure::new(parse, metadata));
        assert_eq!(
            err.as_log().to_string(),
            r#"kind=extract message="demo: failed to parse the output using mode Native""#
        );
        assert!(err.to_string().contains("SENTINEL"));
        assert!(err.to_string().contains("SENTINEL_KEY"));
    }

    /// An extraction error's chain prints its
    /// failure once, with the provider prefix other errors use.
    #[cfg(feature = "extract")]
    #[test]
    fn an_extraction_error_chain_prints_the_failure_once() {
        use crate::extract::{ExtractError, ExtractFailure, ExtractionMetadata, ExtractionMode};

        let failure = ExtractError::MissingToolCall {
            provider: "demo".into(),
        };
        let metadata = ExtractionMetadata::new(ExtractionMode::ForcedTool);
        let err = Error::extract(ExtractFailure::new(failure, metadata));
        assert_eq!(
            chain(&err),
            [
                "extraction error: demo: the model did not call the extraction tool 'submit_structured_output'"
            ]
        );
    }

    #[cfg(feature = "extract")]
    #[test]
    fn extraction_debug_never_shows_the_models_text() {
        use crate::extract::{
            ExtractError, ExtractFailure, ExtractionMetadata, ExtractionMode, RefusalSource,
        };
        use crate::{ChatResponse, ContentBlock, FinishReason};

        let response = ChatResponse::new(vec![ContentBlock::text("RESPONSE {\"ssn\": 1}")]);
        let parse = ExtractError::Parse {
            mode: ExtractionMode::Native,
            provider: "demo".into(),
            raw: "{\"ssn\": \"SENTINEL\"}".into(),
            message: "\"SENTINEL\" is not of type integer".into(),
            path: Some("SENTINEL_KEY".into()),
        };
        let refused = ExtractError::Refused {
            mode: ExtractionMode::Native,
            provider: "demo".into(),
            finish_reason: Some(FinishReason::Refusal),
            text: Some("I won't share SENTINEL".into()),
            source: RefusalSource::Model,
        };
        let metadata = ExtractionMetadata::new(ExtractionMode::Native);
        let parse_err =
            Error::extract(ExtractFailure::new(parse, metadata.clone()).with_response(response));
        let refused_err = Error::extract(ExtractFailure::new(refused, metadata));
        // Also in a failover's chain.
        let nested = Error::timeout("t").with_fallback_from(parse_err.clone());
        for err in [&parse_err, &refused_err, &nested] {
            for debug in [format!("{err:?}"), format!("{err:#?}")] {
                for secret in ["SENTINEL", "RESPONSE"] {
                    assert!(!debug.contains(secret), "{secret}: {debug}");
                }
            }
        }

        assert_eq!(
            format!("{parse_err:?}"),
            "Extract { failure: ExtractFailure { error: Parse { mode: Native, \
             provider: \"demo\", raw: <withheld 19 bytes>, message: <withheld 33 \
             bytes; print with {err}>, path: <withheld 12 bytes; print with {err}> }, \
             response: <withheld>, metadata: ExtractionMetadata { mode: Native, \
             passes: 0, enforcement: NotEnforced } } }"
        );
        let debug = format!("{refused_err:?}");
        assert!(
            debug.contains(
                "Refused { mode: Native, provider: \"demo\", finish_reason: Refusal, \
                 text: <withheld 22 bytes; print with {err}>, source: Model }"
            ),
            "{debug}"
        );
        assert!(!debug.contains("response"), "{debug}");
    }

    #[cfg(feature = "extract")]
    #[test]
    fn extraction_failures_serialize_under_failure() {
        use crate::extract::{ExtractError, ExtractFailure, ExtractionMetadata, ExtractionMode};

        let failure = ExtractFailure::new(
            ExtractError::ToolConflict {
                provider: "demo".into(),
            },
            ExtractionMetadata::new(ExtractionMode::ForcedTool),
        );
        let json = serde_json::to_value(Error::extract(failure)).unwrap();
        assert_eq!(json["type"], "extract");
        assert!(json["data"]["failure"].is_object(), "{json}");
        assert!(json["data"].get("error").is_none(), "{json}");
        let back: Error = serde_json::from_value(json).unwrap();
        let Error::Extract { failure, .. } = back else {
            panic!("an extraction failure expected: {back:?}");
        };
        assert!(matches!(failure.error(), ExtractError::ToolConflict { .. }));
    }

    #[test]
    fn message_prefix_keeps_kind_context_and_origin() {
        let context = |err: Error| {
            err.with_provider("demo")
                .with_status(503)
                .with_request_id("req_1")
                .with_retry_after(Duration::from_secs(3))
                .with_upstream_message("upstream")
        };
        let errors = [
            Error::timeout("m"),
            Error::transport("m"),
            Error::auth("m"),
            Error::configuration("m"),
            Error::rate_limited("m"),
            Error::quota_exhausted("m"),
            Error::overloaded("m"),
            Error::provider("m"),
            Error::unsupported("m"),
            Error::invalid_request("m"),
            Error::model_not_found("m"),
            Error::unexpected_response("m"),
            Error::context_length_exceeded("m", Some(8)),
            Error::content_filtered("m"),
            Error::stream(StreamErrorKind::Incomplete, "m"),
        ];
        for err in errors {
            for authored in [false, true] {
                let err = context(if authored {
                    err.clone().authored()
                } else {
                    err.clone()
                });
                let prefixed = err.clone().with_message_prefix("step");
                assert_eq!(prefixed.kind_name(), err.kind_name());
                assert_eq!(prefixed.message(), "step: m");
                assert_eq!(prefixed.message_origin(), err.message_origin());
                assert_eq!(prefixed.is_retryable(), err.is_retryable());
                assert_eq!(prefixed.upstream_message(), Some("upstream"));
                assert_eq!(prefixed.context(), err.context());
                let log = prefixed.as_log().to_string();
                assert_eq!(log.contains("step: m"), authored, "{log}");
            }
        }
    }

    #[test]
    fn message_prefix_goes_on_the_error_not_its_fallback_chain() {
        let err = Error::rate_limited("HTTP 429")
            .authored()
            .with_status(429)
            .with_fallback_from(Error::overloaded("q"))
            .with_fallback_from(Error::timeout("p").authored())
            .with_message_prefix("step");
        assert!(matches!(err, Error::RateLimited { .. }));
        assert_eq!(err.message(), "step: HTTP 429");
        assert_eq!(err.status(), Some(429));
        let primary = err.fallback_from().and_then(Error::fallback_from).unwrap();
        assert_eq!(primary.message(), "p");
        assert_eq!(
            err.as_log().to_string(),
            r#"kind=rate_limited status=429 message="step: HTTP 429" fallback_from=[kind=overloaded fallback_from=[kind=timeout message="p"]]"#
        );
    }

    #[test]
    fn message_prefix_on_serialization_keeps_the_failure() {
        let source = serde_json::from_str::<u64>("\"x\"").unwrap_err();
        let source_text = source.to_string();
        let err = Error::from(source).with_message_prefix("decode option");
        assert!(matches!(err, Error::Serialization { .. }));
        assert_eq!(err.message(), format!("decode option: {source_text}"));
        assert_eq!(
            err.to_string(),
            format!("serialization error: decode option: {source_text}")
        );
        assert_eq!(err.as_log().to_string(), "kind=serialization");

        // `Display` shows the failure, so the chain does not repeat it; the
        // failure itself stays reachable.
        assert!(std::error::Error::source(&err).is_none());
        let Error::Serialization { error } = &err else {
            unreachable!()
        };
        assert!(
            error
                .get_ref()
                .downcast_ref::<serde_json::Error>()
                .is_some()
        );

        let back: Error = serde_json::from_value(serde_json::to_value(&err).unwrap()).unwrap();
        assert_eq!(back.message(), err.message());
    }

    #[cfg(feature = "extract")]
    #[test]
    fn message_prefix_leaves_extract_unchanged() {
        use crate::extract::{ExtractError, ExtractFailure, ExtractionMetadata, ExtractionMode};

        let failure = ExtractFailure::new(
            ExtractError::ToolConflict {
                provider: "demo".into(),
            },
            ExtractionMetadata::new(ExtractionMode::Native),
        );
        let err = Error::extract(failure);
        let prefixed = err.clone().with_message_prefix("step");
        assert!(matches!(prefixed, Error::Extract { .. }));
        assert_eq!(prefixed.to_string(), err.to_string());
    }

    #[test]
    fn configuration_errors_are_terminal_and_not_auth() {
        let err = Error::configuration("ANTHROPIC_API_KEY not set").with_provider("anthropic");
        assert_eq!(err.kind_name(), "configuration");
        assert!(!err.is_retryable());
        assert!(!err.is_transient());
        assert!(!err.is_auth());
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["type"], "configuration");
        let back: Error = serde_json::from_value(json).unwrap();
        assert!(matches!(back, Error::Configuration { .. }));
        assert_eq!(
            back.context().unwrap().provider.as_deref(),
            Some("anthropic")
        );
        let log = serde_json::to_value(err.as_log()).unwrap();
        assert_eq!(log["type"], "configuration");
        assert_eq!(log["provider"], "anthropic");
    }

    #[test]
    fn serde_round_trip() {
        let err = Error::context_length_exceeded("too long", Some(128_000)).with_request_id("r");
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["data"]["context_window"], 128_000, "{json}");
        let log = serde_json::to_value(err.as_log()).unwrap();
        assert_eq!(log["context_window"], 128_000, "{log}");
        let back: Error = serde_json::from_value(json).unwrap();
        assert!(matches!(
            back,
            Error::ContextLengthExceeded {
                context_window: Some(128_000),
                ..
            }
        ));
        assert_eq!(back.request_id(), Some("r"));
    }

    /// Regression: a stored hint too large for a
    /// `Duration` used to panic while loading.
    #[test]
    fn deserializing_a_huge_retry_after_never_panics() {
        for value in ["1e20", "1.9e19", "-1", "1e308"] {
            let json = format!(
                r#"{{"type":"rate_limited","data":{{"message":"x","context":{{"retry_after":{value}}}}}}}"#
            );
            let err: Error = serde_json::from_str(&json).unwrap();
            assert_eq!(err.retry_after(), None, "{value}");
        }
        let err: Error = serde_json::from_str(
            r#"{"type":"rate_limited","data":{"message":"x","context":{"retry_after":1.5}}}"#,
        )
        .unwrap();
        assert_eq!(err.retry_after(), Some(Duration::from_millis(1500)));
    }

    /// The error's `Display` followed by each `source()`, as report chains
    /// (`anyhow`'s `{:#}`) print it.
    fn chain(error: &(dyn std::error::Error + 'static)) -> Vec<String> {
        let mut lines = vec![error.to_string()];
        let mut source = error.source();
        while let Some(cause) = source {
            lines.push(cause.to_string());
            source = cause.source();
        }
        lines
    }

    /// Regression: the fallback side, a
    /// serialization failure and a prefixed one were printed twice by a
    /// report chain: in `Display` and again as `source()`.
    #[test]
    fn report_chains_print_each_error_once() {
        // A failover's earlier errors are not causes.
        let fallback = Error::rate_limited("HTTP 429")
            .with_provider("openai")
            .with_fallback_from(Error::overloaded("HTTP 529").with_provider("anthropic"));
        assert_eq!(chain(&fallback), ["openai: rate limited: HTTP 429"]);

        let json_failure = serde_json::from_str::<u8>("x").unwrap_err();
        let serialization = Error::from(json_failure).with_message_prefix("tool arguments");
        let lines = chain(&serialization);
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!(lines[0].starts_with("serialization error: tool arguments: expected value"));
        let Error::Serialization { error } = &serialization else {
            unreachable!()
        };
        assert!(
            error
                .get_ref()
                .downcast_ref::<serde_json::Error>()
                .is_some()
        );
    }

    #[derive(Debug)]
    struct Cause(&'static str);

    impl fmt::Display for Cause {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.0)
        }
    }

    impl std::error::Error for Cause {}

    /// A typed cause survives as `source()`, and
    /// `Display`, `Debug`, the log view and serde leave it out.
    #[test]
    fn an_attached_source_is_the_source_and_nothing_else() {
        use std::error::Error as _;

        let err = Error::transport("the request got no response")
            .authored()
            .with_source(Cause("dns: no such host"));
        let cause = err.source().unwrap();
        assert_eq!(
            cause.downcast_ref::<Cause>().unwrap().0,
            "dns: no such host"
        );
        assert_eq!(
            chain(&err),
            [
                "transport error: the request got no response",
                "dns: no such host"
            ]
        );
        assert!(!format!("{err:?}").contains("dns"));
        assert!(!err.as_log().to_string().contains("dns"));
        assert!(!serde_json::to_string(&err).unwrap().contains("dns"));

        // Clones share it; equality and `with_context` ignore or keep it.
        let clone = err.clone();
        assert!(clone.source().is_some());
        let merged = Error::transport("x").with_context(clone.context().unwrap().clone());
        assert!(merged.source().is_some());
        assert_eq!(
            clone.context().unwrap(),
            Error::transport("x").authored().context().unwrap()
        );

        // An error that ended a failover keeps its own source, and the
        // errors before it are not one.
        let fallback = err.clone().with_fallback_from(Error::timeout("p"));
        assert!(fallback.source().unwrap().downcast_ref::<Cause>().is_some());
        let before = Error::timeout("p").with_fallback_from(err);
        assert!(before.source().is_none());
    }

    /// A local refusal has its own kind, is not
    /// retryable, and round-trips.
    #[test]
    fn rejected_is_a_local_refusal() {
        let err = Error::rejected("tenant budget exhausted")
            .authored()
            .with_provider_code("tenant_budget_exhausted");
        assert_eq!(err.kind_name(), "rejected");
        assert!(!err.is_retryable());
        assert!(!err.is_transient());
        assert_eq!(err.to_string(), "rejected: tenant budget exhausted");
        assert_eq!(
            err.as_log().to_string(),
            r#"kind=rejected provider_code=tenant_budget_exhausted message="tenant budget exhausted""#
        );
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["type"], "rejected");
        let back: Error = serde_json::from_value(json).unwrap();
        assert!(matches!(back, Error::Rejected { .. }));
        assert_eq!(back.provider_code(), Some("tenant_budget_exhausted"));
    }

    /// Context builders take values; their `_opt` twins take options.
    #[test]
    fn context_builders_take_values_and_opt_twins_take_options() {
        let err = Error::rate_limited("HTTP 429")
            .with_retry_after(Duration::from_secs(30))
            .with_request_id("req_1")
            .with_server_retry_directive(true)
            .with_provider_code("rate_limit_exceeded")
            .with_provider_type("requests")
            .with_param("model")
            .with_upstream_message("slow down")
            .with_body("{}");
        let kept = err
            .clone()
            .with_retry_after_opt(None)
            .with_request_id_opt(None)
            .with_server_retry_directive_opt(None)
            .with_provider_code_opt(None)
            .with_provider_type_opt(None)
            .with_param_opt(None)
            .with_upstream_message_opt(Some("  "))
            .with_upstream_message("")
            .with_body_opt(None);
        assert_eq!(kept.context(), err.context());
        let header: Option<&str> = Some("req_2");
        assert_eq!(err.with_request_id_opt(header).request_id(), Some("req_2"));
    }
}
