//! Fail over to a second provider and inspect which one served the request,
//! then see what a failover that fails on every side returns.
//!
//! The backup has its own model catalog, so `MapModelProvider` maps the
//! caller's model id to one the backup knows. The backup model has no
//! reasoning controls: validation still accepts the request, because the
//! backup serves it with a warning instead of refusing it.

use std::time::Duration;

use allama::mock::MockProvider;
use allama::prelude::*;
use allama::wrappers::{FallbackAttempts, FallbackProvider, MapModelProvider};
use allama::{CapabilitySupport, ChatCapability};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let primary = MockProvider::build(|b| {
        b.error(
            Error::overloaded("primary overloaded")
                .with_retry_after(Duration::from_millis(25))
                .with_request_id("req_primary_1"),
        )
        .provider_name("primary-mock")
    });
    let backup = MockProvider::build(|b| {
        b.text("Served by fallback provider")
            .provider_name("fallback-mock")
    })
    .with_chat_capability(
        ChatCapability::ReasoningConfig,
        CapabilitySupport::Unsupported,
    );
    let provider = FallbackProvider::new(
        primary.clone(),
        MapModelProvider::new(backup.clone(), |model: &str| match model {
            "demo-model" => "backup-model".into(),
            other => other.into(),
        }),
    );

    let request = ChatRequest::new("demo-model")
        .user("Say hello")
        .reasoning(ReasoningEffort::Low);
    // Validation follows the call: the primary accepts the request, so the
    // pair reports the primary's validation, and the backup's own dry run
    // apart (`fallback()`), each about the model it would receive. The
    // backup's missing reasoning controls are a hint it would drop with a
    // warning (`Warn`, the default); under `UnsupportedPolicy::Error` it
    // could not serve, and its dry run would say so.
    let validation = provider.validate_chat_request(&request)?;
    if let Some(backup) = validation.fallback() {
        match &backup.result {
            Ok(backup_validation) => println!(
                "backup {} would serve, with {} warning(s)",
                backup.provider,
                backup_validation.warnings().len()
            ),
            Err(err) => println!("backup {} could not serve: {err}", backup.provider),
        }
    }
    let response = provider.chat(&request).await?;
    println!("text: {}", response.text_or_empty());
    println!("served by: {:?}", response.provider);
    for warning in &response.warnings {
        println!("warning: {warning}");
    }
    if let Some(attempts) = response.provider_data.get::<FallbackAttempts>()? {
        for attempt in &attempts {
            println!(
                "failed attempt: {} ({})",
                attempt.provider, attempt.error_kind
            );
        }
    }
    println!(
        "calls: primary={}, fallback={}",
        primary.call_count(),
        backup.call_count()
    );
    println!(
        "models sent: primary={:?}, fallback={:?}",
        primary.last_request().map(|r| r.model),
        backup.last_request().map(|r| r.model)
    );

    // When every side fails, the pair returns the backup's error as itself,
    // so the same `match` handles it as any other error. The errors tried
    // before it are kept on it, newest first.
    let failing = FallbackProvider::new(
        MockProvider::with_error(Error::overloaded("HTTP 529").with_status(529))
            .with_provider_name("primary-mock"),
        MockProvider::with_error(Error::rate_limited("HTTP 429").with_status(429))
            .with_provider_name("fallback-mock"),
    );
    let Err(error) = failing.chat(&request).await else {
        unreachable!("both sides fail");
    };
    let status = match &error {
        Error::RateLimited { .. } => 429,
        _ if error.is_retryable() => 503,
        _ => 502,
    };
    println!(
        "failed failover: kind={} -> HTTP {status}",
        error.kind_name()
    );
    let mut earlier = error.fallback_from();
    while let Some(tried) = earlier {
        println!("  fell back from: {}", tried.as_log());
        earlier = tried.fallback_from();
    }
    println!("redacted log: {}", error.as_log());
    Ok(())
}
