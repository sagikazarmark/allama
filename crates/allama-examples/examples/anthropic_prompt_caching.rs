//! Anthropic prompt caching with a cached preamble + volatile tail.
//!
//! Prompt caching intent is portable in `allama`: a
//! [`CacheBreakpoint`](allama::CacheBreakpoint) marks the end of a stable
//! prefix (on a system prompt, tool, or message). Anthropic maps it to
//! `cache_control` (with `ttl: "1h"` for `CacheRetention::Long`); providers
//! with automatic caching treat it as satisfied.
//!
//! The example sends the same request twice: the first call writes the
//! cache, the second reads it. Usage reports both as subsets of
//! `input_tokens` (`cache_write_input_tokens`, `cache_read_input_tokens`).
//!
//! Putting the stable, long preamble first (cached) and the volatile
//! per-request tail afterwards (not cached) lets Anthropic reuse the
//! cached prefix while still honoring fresh instructions on every call.
//!
//! Run:
//!     ANTHROPIC_API_KEY=... cargo run -p allama-examples --example anthropic_prompt_caching
//!
//! Without `ANTHROPIC_API_KEY` set, the example exits with a friendly
//! message rather than panicking.

use std::process::ExitCode;

use allama::CacheBreakpoint;
use allama::prelude::*;
use allama_anthropic::Provider as Anthropic;

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
    if std::env::var("ANTHROPIC_API_KEY")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .is_none()
    {
        eprintln!("Set ANTHROPIC_API_KEY to run this example.");
        return Ok(());
    }

    // Construct the Anthropic provider from the standard env vars
    // (`ANTHROPIC_API_KEY`, optional `ANTHROPIC_BASE_URL`).
    let provider = Anthropic::from_env()?;

    let model = std::env::var("ANTHROPIC_MODEL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "claude-haiku-4-5-20251001".to_string());

    // A cached preamble — typically long and stable across many
    // requests. In real usage this would be substantially larger
    // (Anthropic only caches segments that exceed a minimum token
    // threshold), but the shape of the request is identical.
    // Anthropic only caches prefixes above a minimum size (4096 tokens on
    // Haiku 4.5), so the rules are repeated here to stand in for a real
    // style guide.
    let rules = "You are a senior Rust reviewer. Follow these style rules: \
        prefer explicit error handling, avoid unwrap() outside tests, \
        keep public functions documented, and reason about ownership \
        before suggesting borrow changes. \
        Always cite the exact line a comment refers to and keep feedback \
        grounded in the diff rather than speculation. ";
    let cached_preamble = rules.repeat(80);

    // A small, per-request volatile tail — only meaningful for this one
    // turn, so it comes after the breakpoint.
    let volatile_tail = format!(
        "Context for this review: PR #{pr} touches concurrency primitives.",
        pr = 42,
    );

    // Two system prompts: the breakpoint closes the stable preamble; the
    // volatile tail follows it and is not cached.
    let request = ChatRequest::new(model)
        .system(SystemPrompt::new(cached_preamble).with_cache(CacheBreakpoint::new()))
        .system(SystemPrompt::new(volatile_tail))
        .user("In one sentence: what is the most important rule above?")
        .max_output_tokens(200);

    for attempt in ["first (cache write)", "second (cache read)"] {
        let response = provider.chat(&request).await?;
        for warning in &response.warnings {
            eprintln!("warning: {warning}");
        }
        if let Some(usage) = &response.usage {
            eprintln!(
                "{attempt}: input={:?} cache_write={:?} cache_read={:?}",
                usage.input_tokens, usage.cache_write_input_tokens, usage.cache_read_input_tokens
            );
        }
        println!("{}", response.text_or_empty());
    }
    Ok(())
}
