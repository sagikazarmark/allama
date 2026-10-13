//! Retry wrappers respect provider retry hints (C5).

use std::time::Duration;

use allama::mock::MockProvider;
use allama::wrappers::{FallbackProvider, RetryPolicy, RetryingProvider};
use allama::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, ProviderIdentity,
    Result,
};
use allama_conformance::{RetryHintProbe, assert_retry_respects_hint};

#[tokio::test]
async fn retrying_provider_gives_up_on_a_hint_beyond_its_ceiling() {
    assert_retry_respects_hint(|probe| {
        RetryingProvider::new(
            probe,
            RetryPolicy::new().max_retry_after(Duration::from_millis(50)),
        )
    })
    .await;
}

/// A wrapper that waits longer than the observation also passes: it is
/// still honouring the hint when the contract stops watching. The default
/// policy waits out the 30 s hint.
#[tokio::test]
async fn retrying_provider_may_wait_out_the_hint() {
    assert_retry_respects_hint(|probe| {
        RetryingProvider::new(
            probe,
            RetryPolicy::new().max_retry_after(RetryHintProbe::RETRY_AFTER),
        )
    })
    .await;
    assert_retry_respects_hint(|probe| RetryingProvider::new(probe, RetryPolicy::default())).await;
}

/// Giving up hands the rate limit to the fallback, which serves the call
/// without the primary being called again.
#[tokio::test]
async fn fallback_takes_over_from_a_retry_that_gives_up() {
    assert_retry_respects_hint(|probe| {
        FallbackProvider::new(
            RetryingProvider::new(
                probe,
                RetryPolicy::new().max_retry_after(Duration::from_millis(50)),
            ),
            MockProvider::with_text("backup"),
        )
    })
    .await;
}

/// The contract catches a wrapper that shortens the hint to its own delay.
#[tokio::test]
#[should_panic(expected = "called 2 times")]
async fn contract_catches_a_wrapper_that_clamps_the_hint() {
    assert_retry_respects_hint(Clamping).await;
}

/// Retries once after at most 10 ms, whatever the hint says.
struct Clamping<P>(P);

impl<P: ProviderIdentity> ProviderIdentity for Clamping<P> {
    fn provider_name(&self) -> &str {
        self.0.provider_name()
    }

    fn reads_namespace(&self, namespace: &str) -> bool {
        self.0.reads_namespace(namespace)
    }
}

impl<P: ChatProvider> ChatProvider for Clamping<P> {
    type Stream = P::Stream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        match self.0.chat(request).await {
            Err(error) if error.is_retryable() => {
                let wait = error.retry_after().unwrap_or_default();
                tokio::time::sleep(wait.min(Duration::from_millis(10))).await;
                self.0.chat(request).await
            }
            other => other,
        }
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
        self.0.chat_stream(request).await
    }

    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        self.0.chat_capability(model, capability)
    }
}
