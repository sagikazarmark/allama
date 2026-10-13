//! The retry hint contract (C5): [`assert_retry_respects_hint`].

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use allama::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, ChatStream, Error,
    ProviderIdentity, Result, StreamEvent,
};
use futures_util::StreamExt;
use futures_util::stream::iter;

/// Assert that a chat retry wrapper never calls the provider again before
/// its retry hint has passed, on both `chat` and `chat_stream`.
///
/// When a provider answers "rate limited, retry after 30 s", calling it
/// again sooner only spends an attempt on a request it already said it would
/// reject, and can extend the rate limit. A retry wrapper must either wait
/// at least the whole hint (adding jitter or backoff on top is fine) or give
/// up and return the error with the hint intact, so a fallback or the caller
/// can act on it.
///
/// `wrap` builds the wrapper around a fresh [`RetryHintProbe`], a provider
/// whose first call fails with [`Error::RateLimited`] and a
/// [`RetryHintProbe::RETRY_AFTER`] hint and whose later calls succeed; it is
/// called once per path. The contract watches the wrapper for
/// [`RetryHintProbe::OBSERVATION`] and checks that:
///
/// - the probe is called exactly once (waiting longer than the observation
///   is fine; calling again early is not);
/// - an error that comes back is the rate-limit error with its hint.
///
/// Configure the wrapper so the hint exceeds its longest wait. A wrapper
/// that shortens the hint to a wait longer than the observation is not
/// caught. Run it under a Tokio runtime with timers enabled (for example
/// `#[tokio::test]`).
///
/// ```
/// use std::time::Duration;
///
/// use allama::wrappers::{RetryPolicy, RetryingProvider};
/// use allama_conformance::assert_retry_respects_hint;
///
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// assert_retry_respects_hint(|probe| {
///     RetryingProvider::new(probe, RetryPolicy::new().max_retry_after(Duration::from_millis(50)))
/// })
/// .await;
/// # }
/// ```
///
/// # Panics
///
/// Panics when the probe is called again within the observation window, or
/// when the wrapper returns an error other than the probe's rate-limit error
/// with its hint.
pub async fn assert_retry_respects_hint<W: ChatProvider>(wrap: impl Fn(RetryHintProbe) -> W) {
    let probe = RetryHintProbe::new();
    let wrapper = wrap(probe.clone());
    let request = ChatRequest::new(RetryHintProbe::MODEL).user("hi");
    let outcome = tokio::time::timeout(RetryHintProbe::OBSERVATION, wrapper.chat(&request)).await;
    check(&probe, "chat", outcome.ok().and_then(Result::err));

    let probe = RetryHintProbe::new();
    let wrapper = wrap(probe.clone());
    let outcome = tokio::time::timeout(RetryHintProbe::OBSERVATION, async {
        let mut stream = Box::pin(wrapper.chat_stream(&request).await?);
        while let Some(event) = stream.next().await {
            event?;
        }
        Ok(())
    })
    .await;
    check(&probe, "chat_stream", outcome.ok().and_then(Result::err));
}

fn check(probe: &RetryHintProbe, method: &str, error: Option<Error>) {
    let calls = probe.call_count();
    assert_eq!(
        calls,
        1,
        "{method}: the provider asked to retry after {:?} but was called {calls} times within {:?}",
        RetryHintProbe::RETRY_AFTER,
        RetryHintProbe::OBSERVATION,
    );
    if let Some(error) = error {
        assert_eq!(
            error.kind_name(),
            "rate_limited",
            "{method}: the wrapper replaced the rate-limit error: {error}"
        );
        assert_eq!(
            error.retry_after(),
            Some(RetryHintProbe::RETRY_AFTER),
            "{method}: the wrapper dropped or changed the retry hint"
        );
    }
}

/// The provider [`assert_retry_respects_hint`] wraps.
///
/// Its first call (`chat` or `chat_stream`) fails with
/// [`Error::RateLimited`] and a [`RETRY_AFTER`](Self::RETRY_AFTER) hint;
/// every later call succeeds. Clones share the call count.
#[derive(Debug, Clone, Default)]
pub struct RetryHintProbe {
    calls: Arc<AtomicUsize>,
}

impl RetryHintProbe {
    /// The probe's provider name.
    pub const PROVIDER_NAME: &str = "retry-hint-probe";

    /// The model the contract requests.
    pub const MODEL: &str = "retry-hint-probe-model";

    /// The retry hint on the probe's rate-limit error.
    pub const RETRY_AFTER: Duration = Duration::from_secs(30);

    /// How long the contract watches the wrapper per call.
    pub const OBSERVATION: Duration = Duration::from_secs(1);

    /// A probe that has not been called yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// How many times the probe has been called.
    #[must_use]
    pub fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn call(&self) -> Result<()> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            Err(Error::rate_limited("retry-hint probe: slow down")
                .with_retry_after(Self::RETRY_AFTER))
        } else {
            Ok(())
        }
    }
}

impl ProviderIdentity for RetryHintProbe {
    fn provider_name(&self) -> &str {
        Self::PROVIDER_NAME
    }

    fn reads_namespace(&self, _namespace: &str) -> bool {
        false
    }
}

impl ChatProvider for RetryHintProbe {
    type Stream = ChatStream;

    async fn chat(&self, _request: &ChatRequest) -> Result<ChatResponse> {
        self.call()?;
        Ok(ChatResponse::default())
    }

    async fn chat_stream(&self, _request: &ChatRequest) -> Result<Self::Stream> {
        self.call()?;
        let start = StreamEvent::response_start(None, None);
        Ok(Box::pin(iter([Ok(start)])))
    }

    fn chat_capability(&self, _model: &str, _capability: ChatCapability) -> CapabilitySupport {
        CapabilitySupport::Unknown
    }
}
