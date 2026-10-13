use std::fmt;
use std::sync::Arc;

use allama::wrappers::{RetryPolicy, TimeoutPolicy};
use allama::{ChatCapabilityResolver, Error, UnsupportedPolicy};

use crate::registry::Operation;

/// How a [`Registry`](crate::Registry) lookup builds a provider: what the
/// constructor configures on the adapter, and the wrappers the lookup puts
/// around it.
///
/// The `_with` lookups take settings from the caller; the others use
/// [`ProviderSettings::default()`].
///
/// # The default stack
///
/// Every lookup returns the provider wrapped in, outermost first:
///
/// 1. a [`RetryingProvider`](allama::wrappers::RetryingProvider) with
///    [`retry`](Self::retry), **on by default** with
///    [`RetryPolicy::default()`];
/// 2. a [`TimeoutProvider`](allama::wrappers::TimeoutProvider) that bounds
///    each attempt, with [`attempt_timeout`](Self::attempt_timeout), off by
///    default, since the adapters already apply transport timeouts;
/// 3. an [`UnsupportedPolicyProvider`](allama::wrappers::UnsupportedPolicyProvider)
///    with [`unsupported_policy`](Self::unsupported_policy), off by default
///    (chat and embeddings; decision requests have no such policy).
///
/// Built-in and registered constructors get the same stack, and the result
/// is still a `Dyn*Provider`. `allama` itself stays bare: an adapter or
/// wrapper you build yourself retries nothing unless you wrap it.
/// [`chat::wrap`](crate::chat::wrap),
/// [`embedding::wrap`](crate::embedding::wrap) and
/// [`decision::wrap`](crate::decision::wrap) put this stack around a
/// provider you built, with these settings. They refuse
/// [`credential_scope`](Self::credential_scope), and for chat
/// [`chat_capabilities`](Self::chat_capabilities), as a lookup of a
/// constructor registered without settings does, since only the adapter's
/// builder can apply them.
///
/// **Tracing is not a setting.** Its span belongs outside everything that
/// shapes the call: around a deadline, so the deadline's expiry is a
/// failed call with `error.type = timeout` rather than a cancelled one, and
/// around a fallback pair, so one span covers the failover. A setting would
/// put it inside both, on each side. Wrap the provider, or the stack you
/// build from it, in
/// [`TracingProvider`][tracingprovider]
/// (`allama`'s `tracing` feature) last:
///
/// ```text
/// TracingProvider(TimeoutProvider(FallbackProvider(lookup, lookup)))
/// ```
///
/// ## Retries
///
/// The default policy makes up to 3 attempts. Between them it backs off
/// from 250 ms up to 5 s, with jitter, or waits out the provider's retry
/// hint (`Retry-After`) of up to 120 s, and gives up at once on a longer
/// hint. It follows the server's retry directive (`x-should-retry`), and
/// each provider a lookup builds owns a retry budget, so retries stay
/// bounded while a provider is down. See
/// [`RetryPolicy`](allama::wrappers::RetryPolicy) for every rule.
///
/// The defaults suit batch work that would rather wait than fail: a unary
/// call can take about 10 minutes before its error reaches you, and the
/// wait for a stream's first output has no bound at all (see
/// [Worst-case latency](#worst-case-latency)).
///
/// **Interactive servers and fallback pairs** set a hint ceiling of a few
/// seconds, so a long rate limit fails over, or reaches the client as
/// `429` with its `Retry-After`, at once. Sides built with the default
/// settings wait out a `Retry-After` of up to 120 s before a
/// [`FallbackProvider`](allama::wrappers::FallbackProvider) sees the error:
///
/// ```
/// use std::time::Duration;
///
/// use allama::wrappers::{RetryPolicy, TimeoutPolicy};
/// use allama_providers::ProviderSettings;
///
/// let interactive = ProviderSettings::new()
///     .retry(Some(RetryPolicy::new().max_retry_after(Duration::from_secs(5))))
///     // A stalled attempt is cut off and retried, unary or streamed.
///     .attempt_timeout(Some(
///         TimeoutPolicy::new()
///             .request(Duration::from_secs(30))
///             .stream_setup(Duration::from_secs(15))
///             .stream_idle(Duration::from_secs(30)),
///     ));
/// # let _ = interactive;
/// ```
///
/// **Do not nest retries.** A provider from a lookup or a `wrap` already
/// retries. Wrapping it in another `RetryingProvider` multiplies the
/// attempts (3 × 3) and the waits. Configure the policy here, or turn it off with
/// `retry(None)` and wrap the provider yourself.
///
/// **`409 Conflict` is not retried**, unlike in the official OpenAI and
/// Anthropic SDKs, which retry it as a lock timeout. allama classifies
/// `409` as an invalid request, and a provider that wants it retried says
/// so with `x-should-retry: true`, which the default policy follows. An
/// application that wants the SDKs' behavior adds it to the classifier:
///
/// ```
/// use allama::Error;
/// use allama::wrappers::RetryPolicy;
/// use allama_providers::ProviderSettings;
///
/// let settings = ProviderSettings::new().retry(Some(RetryPolicy::new().retry_if(|error: &Error| {
///     error.server_retry_directive().unwrap_or(error.is_retryable() || error.status() == Some(409))
/// })));
/// # let _ = settings;
/// ```
///
/// # Worst-case latency
///
/// Several timeouts bound a call, in different layers and phases. With the
/// default settings only the adapter's transport timeouts apply:
///
/// | Knob | Layer | Bounds | Default | Retried |
/// |---|---|---|---|---|
/// | adapter builder `timeout` | transport, one HTTP exchange | a whole unary call (`chat`, `embed`, `decide`) | 120 s for the built-in providers, whatever the transport; none for the `ollama` (chat, embeddings and decisions) and `vllm` presets | yes |
/// | adapter builder `stream_timeouts`, setup | transport, one HTTP exchange | a stream until its response headers | 120 s; none for the `ollama` and `vllm` presets | yes |
/// | adapter builder `stream_timeouts`, idle | transport, one HTTP exchange | each wait for the next body chunk; keep-alive pings count as chunks | 300 s; none for the `ollama` and `vllm` presets | before the commit point |
/// | the default transport's connect timeout | transport, one connection | TCP and TLS setup | 10 s, every preset included | yes |
/// | [`attempt_timeout`](Self::attempt_timeout) | `TimeoutProvider` inside the retries | one attempt, by `TimeoutPolicy` bound: `request`, `stream_setup` (until the adapter's stream opens; unset, the `request` bound), `stream_idle` (each wait for the next decoded event), `stream_total` | off | yes |
/// | a deadline: your own [`TimeoutProvider`](allama::wrappers::TimeoutProvider) around the provider | outside retries and fallback | the whole call: `request` for a unary call and for the wait for a stream's first output across every attempt, backoff and failover (`stream_setup` sets a different bound for the stream) | none | no: it is the caller's bound |
///
/// "Retried" means an expiry is `Error::Timeout`, which the default
/// [`RetryPolicy`](allama::wrappers::RetryPolicy) retries, and a
/// [`FallbackProvider`](allama::wrappers::FallbackProvider) fails over on.
/// Retry and fallback act only before a stream's
/// [commit point](allama::chat::stream#the-commit-point), its first model output;
/// after it, an expiry ends the stream.
///
/// The worst cases with the default settings, before an error reaches you:
///
/// - **The `ollama` and `vllm` presets: unbounded.** A local server may
///   load the model before it answers, so they set no request or stream
///   bound; only the 10 s connect timeout applies, so an unreachable
///   host fails fast but a hung one blocks the call. Bound them with an
///   [`attempt_timeout`](Self::attempt_timeout) or a deadline.
/// - **A unary call: about 10 minutes.** 3 attempts of up to 120 s each,
///   and two waits for a `Retry-After` of up to 120 s.
/// - **A stream before its commit point: unbounded.** Each attempt waits up
///   to 120 s for the headers, then up to 300 s for each body chunk, and
///   nothing bounds the wait for the first output as a whole. An attempt
///   that goes silent fails after about 7 minutes, so three take about
///   25 minutes with the two waits. An endpoint that keeps sending
///   keep-alive pings or other events without output while the model
///   thinks resets the idle bound every time, so the attempt never times
///   out.
/// - **Behind a fallback**, the sides' worst cases add up.
///
/// The [retry budget](allama::wrappers::RetryBudget) may cut retries short;
/// these are upper bounds. To bound a call, put a deadline around the
/// provider (or around the fallback pair), inside any `TracingProvider`:
/// `request`, which bounds a unary call and a stream's wait for its first
/// output alike, never `stream_total`, which would cut off a long answer
/// after it started. An
/// [`attempt_timeout`](Self::attempt_timeout) does not bound the call, but
/// it cuts a stalled attempt short so the retry comes sooner:
///
/// ```
/// use std::time::Duration;
///
/// use allama::mock::MockProvider;
/// use allama::wrappers::{TimeoutPolicy, TimeoutProvider};
/// use allama_providers::ProviderSettings;
///
/// # fn main() -> allama::Result<()> {
/// let settings = ProviderSettings::new().attempt_timeout(Some(
///     TimeoutPolicy::new()
///         .stream_setup(Duration::from_secs(15))
///         .stream_idle(Duration::from_secs(30)),
/// ));
/// # let adapter = MockProvider::empty();
/// let provider = allama_providers::chat::wrap(adapter, &settings)?;
/// // The deadline: the first output within 60 s, across every attempt.
/// let deadline = TimeoutPolicy::new().request(Duration::from_secs(60));
/// let provider = TimeoutProvider::new(provider, deadline);
/// # let _ = provider; Ok(()) }
/// ```
///
/// See [attempt timeout vs deadline](allama::wrappers::TimeoutPolicy#attempt-timeout-vs-deadline).
///
/// # What constructors apply
///
/// The wrappers are the lookup's. A constructor registered with
/// `register_chat_with` (or its embedding and decision twins) receives the
/// settings only the adapter can apply as [`AdapterSettings`]:
/// [`credential_scope`](Self::credential_scope), and for chat
/// [`chat_capabilities`](Self::chat_capabilities). It takes each one it
/// applies, and a lookup whose constructor leaves one untaken fails for its
/// key, as does a lookup of a constructor registered without settings (see
/// [Settings](crate::Registry#settings)).
///
#[doc = concat!(
    "[tracingprovider]: https://docs.rs/allama/",
    env!("CARGO_PKG_VERSION"),
    "/allama/wrappers/struct.TracingProvider.html",
)]
#[derive(Clone)]
#[non_exhaustive]
pub struct ProviderSettings {
    /// The credential scope to build the provider with: per-request
    /// credentials (`allama_http::HttpRequestOptions::credential`) addressed
    /// to it are the provider's. `None` keeps the provider's default scope,
    /// its provider name.
    ///
    /// The built-in providers still read their key from the environment, so
    /// a request without a credential for the scope is sent with that key.
    /// For bring-your-own-key without one, register a constructor that
    /// takes the scope ([`AdapterSettings::take_credential_scope`]) and
    /// builds the provider with it and no key.
    pub credential_scope: Option<String>,
    /// Capability answers that override the chat adapter's own, for model
    /// ids it cannot classify (a fine-tuned model, a deployment name, a
    /// gateway alias). The constructor hands it to the adapter's
    /// `with_chat_capabilities`, so it feeds the adapter's own checks and
    /// may upgrade an answer as well as downgrade it. `None` keeps the
    /// adapter's answers.
    ///
    /// There is no embedding or decision counterpart yet, here or as a
    /// wrapper: an embedding or decision provider built from the registry
    /// keeps its adapter's answers, `Unknown` ones included (Gemini answers
    /// [`EmbeddingCapability::InputType`] `Unknown` for `gemini-embedding-2`
    /// and for model ids it does not know), and a request that relies on
    /// one is sent with the answer recorded as `Unknown`.
    ///
    /// [`EmbeddingCapability::InputType`]: allama::EmbeddingCapability::InputType
    pub chat_capabilities: Option<Arc<dyn ChatCapabilityResolver>>,
    /// The unsupported-feature policy of chat and embedding requests that
    /// set none (see `UnsupportedPolicyProvider`). `None` leaves them to
    /// the provider's default, [`UnsupportedPolicy::Warn`].
    pub unsupported_policy: Option<UnsupportedPolicy>,
    /// A bound on each attempt, inside the retries, so a stalled attempt is
    /// retried. `None`, the default, leaves attempts to the adapter's
    /// transport timeouts.
    pub attempt_timeout: Option<TimeoutPolicy>,
    /// How transient errors are retried. The default is
    /// [`RetryPolicy::default()`]; `None` turns retries off.
    pub retry: Option<RetryPolicy>,
}

impl Default for ProviderSettings {
    fn default() -> Self {
        Self {
            credential_scope: None,
            chat_capabilities: None,
            unsupported_policy: None,
            attempt_timeout: None,
            retry: Some(RetryPolicy::default()),
        }
    }
}

impl ProviderSettings {
    /// The defaults: each provider's own configuration, with retries.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build with the credential scope `scope`.
    #[must_use]
    pub fn credential_scope(mut self, scope: impl Into<String>) -> Self {
        self.credential_scope = Some(scope.into());
        self
    }

    /// Override the chat adapter's capability answers with `resolver`
    /// (see [`chat_capabilities`](Self#structfield.chat_capabilities)).
    ///
    /// ```
    /// use allama::{CapabilitySupport, ChatCapability};
    /// use allama_providers::ProviderSettings;
    ///
    /// // A fine-tuned model that keeps its base model's vision.
    /// let settings = ProviderSettings::new().chat_capabilities(|model: &str, capability| {
    ///     (model.starts_with("ft:") && capability == ChatCapability::ImageInput)
    ///         .then_some(CapabilitySupport::Supported)
    /// });
    /// # let _ = settings;
    /// ```
    #[must_use]
    pub fn chat_capabilities(mut self, resolver: impl ChatCapabilityResolver) -> Self {
        self.chat_capabilities = Some(Arc::new(resolver));
        self
    }

    /// Give chat and embedding requests that set no unsupported-feature
    /// policy `policy`. A request that sets its own keeps it.
    #[must_use]
    pub fn unsupported_policy(mut self, policy: UnsupportedPolicy) -> Self {
        self.unsupported_policy = Some(policy);
        self
    }

    /// Bound each attempt with `policy`, inside the retries; `None` (the
    /// default) leaves attempts to the adapter's transport timeouts.
    #[must_use]
    pub fn attempt_timeout(mut self, policy: Option<TimeoutPolicy>) -> Self {
        self.attempt_timeout = policy;
        self
    }

    /// Retry transient errors with `policy`; `None` turns retries off.
    #[must_use]
    pub fn retry(mut self, policy: Option<RetryPolicy>) -> Self {
        self.retry = policy;
        self
    }
}

impl ProviderSettings {
    /// The first setting for `operation` that only the adapter's
    /// constructor can apply, if any is set. A lookup of a constructor
    /// registered without settings and [`wrap`](crate::chat::wrap) both
    /// refuse it.
    pub(crate) fn constructor_setting(&self, operation: Operation) -> Option<ConstructorSetting> {
        AdapterSettings::new(self, operation).untaken()
    }
}

impl fmt::Debug for ProviderSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderSettings")
            .field("credential_scope", &self.credential_scope)
            .field(
                "chat_capabilities",
                &self.chat_capabilities.as_ref().map(|_| "<resolver>"),
            )
            .field("unsupported_policy", &self.unsupported_policy)
            .field("attempt_timeout", &self.attempt_timeout)
            .field("retry", &self.retry)
            .finish()
    }
}

/// The settings of a lookup that only the adapter can apply, handed to a
/// constructor registered with
/// [`Registry::register_chat_with`](crate::Registry::register_chat_with)
/// (or its embedding and decision twins).
///
/// The constructor takes each setting and applies it to the adapter it
/// builds. When it returns, a setting the lookup set and the constructor
/// did not take fails the lookup with [`Error::Configuration`], as a lookup
/// of a constructor registered without settings does: an adapter built
/// without the tenant's credential scope would send the configured key and
/// bill the wrong account.
///
/// The other [`ProviderSettings`] (retries, the attempt timeout, the
/// unsupported-feature policy) are not here: the lookup applies them around
/// whatever the constructor builds.
///
/// ```
/// # #[cfg(feature = "openai-compat")] {
/// use allama_openai_compat::providers::Vllm;
/// use allama_providers::{AdapterSettings, ProviderSettings, Registry};
///
/// # fn main() -> allama::Result<()> {
/// let mut registry = Registry::new();
/// registry.register_chat_with("local", |settings: &mut AdapterSettings| {
///     let mut builder = Vllm::builder("http://gpu-box:8000/v1", None)?;
///     if let Some(scope) = settings.take_credential_scope() {
///         builder = builder.credential_scope(scope);
///     }
///     let provider = builder.build()?;
///     Ok(match settings.take_chat_capabilities() {
///         Some(resolver) => provider.with_chat_capabilities(resolver),
///         None => provider,
///     })
/// });
///
/// let settings = ProviderSettings::new().credential_scope("tenant-a");
/// let (provider, model) = registry.chat_from_model_with("local:qwen3-32b", &settings)?;
/// # let _ = (provider, model); Ok(()) }
/// # }
/// ```
pub struct AdapterSettings {
    credential_scope: Option<String>,
    chat_capabilities: Option<Arc<dyn ChatCapabilityResolver>>,
}

impl AdapterSettings {
    /// Take the credential scope to build the adapter with, for its builder's
    /// `credential_scope` or the adapter's `with_credential_scope`. `None`
    /// keeps the adapter's default scope, its provider name.
    pub fn take_credential_scope(&mut self) -> Option<String> {
        self.credential_scope.take()
    }

    /// Take the capability answers that override the chat adapter's own,
    /// for its `with_chat_capabilities`. `None` keeps the adapter's answers,
    /// and so does every embedding and decision lookup, where chat answers
    /// change nothing.
    pub fn take_chat_capabilities(&mut self) -> Option<Arc<dyn ChatCapabilityResolver>> {
        self.chat_capabilities.take()
    }
}

impl AdapterSettings {
    /// The settings of `settings` that a constructor for `operation` must
    /// apply.
    pub(crate) fn new(settings: &ProviderSettings, operation: Operation) -> Self {
        Self {
            credential_scope: settings.credential_scope.clone(),
            chat_capabilities: if operation == Operation::Chat {
                settings.chat_capabilities.clone()
            } else {
                None
            },
        }
    }

    /// The first setting still here, if any.
    pub(crate) fn untaken(&self) -> Option<ConstructorSetting> {
        if self.credential_scope.is_some() {
            Some(ConstructorSetting {
                name: "a credential scope",
                take_method: "take_credential_scope",
                adapter_method: "with_credential_scope",
            })
        } else if self.chat_capabilities.is_some() {
            Some(ConstructorSetting {
                name: "chat capability overrides",
                take_method: "take_chat_capabilities",
                adapter_method: "with_chat_capabilities",
            })
        } else {
            None
        }
    }
}

impl fmt::Debug for AdapterSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AdapterSettings")
            .field("credential_scope", &self.credential_scope)
            .field(
                "chat_capabilities",
                &self.chat_capabilities.as_ref().map(|_| "<resolver>"),
            )
            .finish()
    }
}

/// A setting only the adapter's constructor can apply (see
/// [What constructors apply](ProviderSettings#what-constructors-apply)).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ConstructorSetting {
    /// The setting, as the error names it: "a credential scope".
    pub(crate) name: &'static str,
    /// The [`AdapterSettings`] method that takes it: `take_credential_scope`.
    pub(crate) take_method: &'static str,
    /// The adapter method that applies it: `with_credential_scope`.
    pub(crate) adapter_method: &'static str,
}

impl ConstructorSetting {
    /// The configuration error for a provider that was built without the
    /// setting: "{subject}, so it cannot be built with {setting}; {remedy}".
    pub(crate) fn refuse(self, subject: &str, remedy: &str) -> Error {
        Error::configuration(format!(
            "{subject}, so it cannot be built with {}; {remedy}",
            self.name
        ))
        .authored()
    }

    /// The configuration error for a constructor registered for `key` with
    /// settings that returned a provider without taking the setting.
    pub(crate) fn untaken_by(self, key: &str) -> Error {
        Error::configuration(format!(
            "the constructor of provider `{key}` did not take {}, so the provider was built \
             without it; take it with `AdapterSettings::{}` and apply it with the adapter's `{}`",
            self.name, self.take_method, self.adapter_method
        ))
        .authored()
    }
}
