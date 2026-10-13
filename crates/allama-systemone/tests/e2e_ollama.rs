//! Live checks against a local Ollama 0.35+ with a decision model. Ignored
//! by default:
//!
//! ```text
//! ollama pull nimble
//! cargo test -p allama-systemone --test e2e_ollama -- --ignored
//! ```
//!
//! `OLLAMA_HOST` selects the server and `OLLAMA_DECISION_MODEL` (default
//! `nimble`) the model. On an Ollama older than 0.35, which has no System
//! One endpoint, the tests skip with a message instead of failing.

// Sends through the default `reqwest` transport.
#![cfg(feature = "reqwest")]

mod support;

use allama_systemone::Provider;
use allama_systemone::providers::Ollama;

/// The provider and model to test, or `None` (with a message) when the
/// server is older than Ollama 0.35, the first release with System One.
async fn provider() -> Option<(Provider, String)> {
    let model = std::env::var("OLLAMA_DECISION_MODEL").unwrap_or_else(|_| "nimble".into());
    let provider = Ollama::from_env().unwrap();
    let url = provider.url();
    let root = url.strip_suffix("/v1/systemone").unwrap_or(&url);
    let version: serde_json::Value = reqwest::get(format!("{root}/api/version"))
        .await
        .and_then(reqwest::Response::error_for_status)
        .expect("Ollama answers /api/version")
        .json()
        .await
        .expect("Ollama's version is JSON");
    let version = version["version"].as_str().unwrap_or_default();
    if older_than_0_35(version) {
        eprintln!("skipped: Ollama {version} has no System One endpoint (0.35 or later needed)");
        return None;
    }
    Some((provider, model))
}

/// Whether an Ollama version is below 0.35. A development build (`0.0.0`,
/// or a version that does not parse) is taken as recent.
fn older_than_0_35(version: &str) -> bool {
    let mut parts = version
        .split(['.', '-', '+'])
        .map(|part| part.parse::<u64>().ok());
    match (parts.next().flatten(), parts.next().flatten()) {
        (Some(0), Some(0)) => false,
        (Some(major), Some(minor)) => (major, minor) < (0, 35),
        _ => false,
    }
}

#[test]
fn versions_below_0_35_are_skipped() {
    assert!(older_than_0_35("0.34.0"));
    assert!(older_than_0_35("0.9.6"));
    assert!(!older_than_0_35("0.35.0"));
    assert!(!older_than_0_35("0.35.0-rc1"));
    assert!(!older_than_0_35("1.0.0"));
    assert!(!older_than_0_35("0.0.0-dev"));
    assert!(!older_than_0_35("main"));
}

#[tokio::test]
#[ignore]
async fn untyped() {
    let Some((provider, model)) = provider().await else {
        return;
    };
    support::untyped(&provider, &model).await;
}

#[tokio::test]
#[ignore]
async fn typed() {
    let Some((provider, model)) = provider().await else {
        return;
    };
    support::typed(&provider, &model).await;
}

#[tokio::test]
#[ignore]
async fn unknown_model() {
    let Some((provider, _)) = provider().await else {
        return;
    };
    support::unknown_model(&provider).await;
}
