//! Retry transient errors against the same provider; the retry budget each
//! wrapper owns bounds retries across calls.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use allama::mock::{FunctionProvider, MockProvider};
use allama::prelude::*;
use allama::wrappers::{RetryBudget, RetryPolicy, RetryingProvider};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = MockProvider::build(|b| {
        b.error(Error::timeout("first attempt timed out"))
            .text("Recovered on retry")
    });
    let retrying = RetryingProvider::new(
        provider.clone(),
        RetryPolicy::new()
            .max_attempts(2)
            .base_delay(Duration::ZERO)
            .max_delay(Duration::ZERO),
    );

    let response = retrying
        .chat(&ChatRequest::new("demo-model").user("Say hello"))
        .await?;
    println!("retry result: {}", response.text_or_empty());
    println!("calls: {}", provider.call_count());

    // A full outage: every call fails. Without a budget, each of the 100
    // calls would make 3 attempts (300 upstream calls). The budget the
    // wrapper owns by default affords 500 / 14 = 35 retries in all, then
    // calls fail fast. Build the wrapper once: one built per request would
    // start with a full budget every time.
    let upstream = Arc::new(AtomicUsize::new(0));
    let counter = upstream.clone();
    let down = FunctionProvider::new(move |_| {
        counter.fetch_add(1, Ordering::Relaxed);
        Err(Error::overloaded("upstream down"))
    });
    let retrying = RetryingProvider::new(
        down,
        RetryPolicy::new()
            .base_delay(Duration::ZERO)
            .max_delay(Duration::ZERO),
    );
    for _ in 0..100 {
        let _ = retrying
            .chat(&ChatRequest::new("demo-model").user("Say hello"))
            .await;
    }
    println!(
        "outage: 100 calls, {} upstream calls, {} budget tokens left",
        upstream.load(Ordering::Relaxed),
        retrying.budget().map_or(0, RetryBudget::available)
    );
    Ok(())
}
