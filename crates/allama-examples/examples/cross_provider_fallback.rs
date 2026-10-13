//! Cross-provider fallback: [`FallbackProvider`] tries the primary and fails
//! over to a secondary on errors that happen before any model output and
//! that another provider may well not have: outages, overload, rate limits,
//! exhausted quotas, timeouts, and requests the primary cannot serve. The
//! request's or the setup's own problems (an invalid request, rejected
//! credentials, an unknown model, a content filter) surface directly from
//! the primary; `fallback_if` opts in to more (see `should_fallback`).
//!
//! The primary here is OpenAI pointed at an unreachable address, standing in
//! for an outage, so the failover path runs every time; the recorded attempt
//! reports it as a `transport` error. The served response
//! names the provider that answered and records the failed attempts in its
//! provider data ([`FallbackAttempts`]).
//!
//! The request names the model the caller wants (an Anthropic model here).
//! [`MapModelProvider`] gives the OpenAI side its own model id, so each
//! provider receives a model from its own catalog, and validation and
//! capability answers describe the model each side would actually receive.
//!
//! Requires `ANTHROPIC_API_KEY`.

use std::borrow::Cow;
use std::process::ExitCode;

use allama::prelude::*;
use allama::wrappers::{FallbackAttempts, FallbackProvider, MapModelProvider};
use allama_anthropic::Provider as Anthropic;
use allama_openai::Provider as OpenAI;

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<()> {
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "In one sentence: why is Rust good for servers?".into());
    let model = std::env::var("ANTHROPIC_MODEL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "claude-haiku-4-5-20251001".into());

    let unreachable = OpenAI::new_with_base_url("unused", "http://127.0.0.1:9/v1")?;
    let provider = FallbackProvider::new(
        MapModelProvider::new(unreachable, openai_model),
        Anthropic::from_env()?,
    );

    let response = provider
        .chat(&ChatRequest::new(&model).user(prompt))
        .await?;
    eprintln!("served by: {}", response.provider.as_deref().unwrap_or("?"));
    if let Some(attempts) = response.provider_data.get::<FallbackAttempts>()? {
        for attempt in &attempts {
            eprintln!(
                "failed attempt: provider={} kind={} message={}",
                attempt.provider, attempt.error_kind, attempt.error_message
            );
        }
    }
    println!("{}", response.text_or_empty());

    // A non-retryable error from the primary is returned as-is: a different
    // provider would not fix a bad credential.
    let bad_key = FallbackProvider::new(
        MapModelProvider::new(OpenAI::new("sk-invalid")?, openai_model),
        Anthropic::from_env()?,
    );
    match bad_key.chat(&ChatRequest::new(&model).user("Hello")).await {
        Ok(_) => eprintln!("unexpected: invalid key was accepted"),
        Err(err) => eprintln!(
            "non-retryable primary error: {} (retryable={})",
            err.kind_name(),
            err.is_retryable()
        ),
    }

    Ok(())
}

/// The OpenAI model to send whatever Anthropic model the caller asked for.
fn openai_model(_anthropic_model: &str) -> Cow<'_, str> {
    "gpt-5-mini".into()
}
