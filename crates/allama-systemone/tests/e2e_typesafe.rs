//! Live checks against TypeSafe's hosted API. Ignored by default:
//!
//! ```text
//! TYPESAFE_API_KEY=... cargo test -p allama-systemone --test e2e_typesafe -- --ignored
//! ```
//!
//! `TYPESAFE_MODEL` (default `jev-latest`) selects the model.

// Sends through the default `reqwest` transport.
#![cfg(feature = "reqwest")]

mod support;

use allama_systemone::Provider;
use allama_systemone::providers::TypeSafe;

fn provider() -> (Provider, String) {
    let model = std::env::var("TYPESAFE_MODEL").unwrap_or_else(|_| "jev-latest".into());
    (TypeSafe::from_env().unwrap(), model)
}

#[tokio::test]
#[ignore]
async fn untyped() {
    let (provider, model) = provider();
    support::untyped(&provider, &model).await;
}

#[tokio::test]
#[ignore]
async fn typed() {
    let (provider, model) = provider();
    support::typed(&provider, &model).await;
}

#[tokio::test]
#[ignore]
async fn unknown_model() {
    let (provider, _) = provider();
    support::unknown_model(&provider).await;
}
