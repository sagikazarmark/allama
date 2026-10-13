//! Real rate-limit answers from OpenAI and Anthropic, as evidence for the
//! retry policy: which of `retry-after`, `retry-after-ms`
//! and `x-should-retry` a 429 (or Anthropic's 529) carries, and what
//! `allama_http::error_context` reads from them.
//!
//! Each check records one successful answer (for its rate-limit headers),
//! then sends cheap one-token requests, `ALLAMA_EVIDENCE_CONCURRENCY` at a
//! time, until the API answers 429 or `ALLAMA_EVIDENCE_BURST` requests have
//! been sent. It records the first answer of each error status, and a
//! summary. A 529 (overloaded) cannot be provoked; it is recorded when it
//! happens. Raise the burst for an account with high limits.

use std::collections::{BTreeMap, BTreeSet};

use allama_http::{HttpRequest, ReqwestTransport};
use futures_util::StreamExt;
use serde_json::json;

use super::{env, env_or, post_json, record, send, try_send};

const CHECK: &str = "rate_limits";

async fn capture(provider: &str, make: impl Fn() -> HttpRequest) {
    let transport = ReqwestTransport::new();
    let first = send(&transport, make()).await;
    record(
        CHECK,
        &format!("{provider}_first_answer"),
        &json!({"exchange": first.record()}),
    );

    let burst: usize = env_or("ALLAMA_EVIDENCE_BURST", "300")
        .parse()
        .expect("ALLAMA_EVIDENCE_BURST is a number");
    let concurrency: usize = env_or("ALLAMA_EVIDENCE_CONCURRENCY", "50")
        .parse()
        .expect("ALLAMA_EVIDENCE_CONCURRENCY is a number");
    let mut statuses: BTreeMap<u16, usize> = BTreeMap::new();
    let mut captured: BTreeSet<u16> = BTreeSet::new();
    let mut transport_errors = 0;
    let mut sent = 0;
    let mut answers = futures_util::stream::iter(0..burst)
        .map(|_| try_send(&transport, make()))
        .buffer_unordered(concurrency.max(1));
    while let Some(answer) = answers.next().await {
        sent += 1;
        let Ok(exchange) = answer else {
            transport_errors += 1;
            continue;
        };
        *statuses.entry(exchange.status).or_default() += 1;
        if !exchange.accepted() && captured.insert(exchange.status) {
            record(
                CHECK,
                &format!("{provider}_status_{}", exchange.status),
                &json!({"after_requests": sent, "exchange": exchange.record()}),
            );
        }
        if captured.contains(&429) {
            break;
        }
    }
    let statuses: BTreeMap<String, usize> = statuses
        .into_iter()
        .map(|(status, count)| (status.to_string(), count))
        .collect();
    record(
        CHECK,
        &format!("{provider}_summary"),
        &json!({
            "burst": burst,
            "concurrency": concurrency,
            "answers_read": sent,
            "statuses": statuses,
            "transport_errors": transport_errors,
            "rate_limited": captured.contains(&429),
        }),
    );
}

#[tokio::test]
#[ignore]
async fn rate_limit_openai() {
    let model = env_or("OPENAI_MODEL", "gpt-4.1-nano");
    let key = env("OPENAI_API_KEY");
    let body = json!({
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_completion_tokens": 1,
    });
    capture("openai", || {
        post_json(
            "https://api.openai.com/v1/chat/completions",
            ("authorization", format!("Bearer {key}")),
            &body,
        )
    })
    .await;
}

#[tokio::test]
#[ignore]
async fn rate_limit_anthropic() {
    let model = env_or("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001");
    let key = env("ANTHROPIC_API_KEY");
    let body = json!({
        "model": model,
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "Hi"}],
    });
    capture("anthropic", || {
        post_json(
            "https://api.anthropic.com/v1/messages",
            ("x-api-key", key.clone()),
            &body,
        )
        .header("anthropic-version", "2023-06-01")
    })
    .await;
}
