//! The presets' `from_env` constructors. A test binary of its own, since
//! the tests change the process environment.
#![cfg(feature = "reqwest")]

use std::sync::Mutex;

use allama::Error;
use allama_openai_compat::providers::{Azure, Vllm};

static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Set (or remove) environment variables for one test, restoring them when
/// dropped.
struct Env(Vec<(&'static str, Option<String>)>);

impl Env {
    fn set(vars: &[(&'static str, Option<&str>)]) -> Self {
        let saved = vars
            .iter()
            .map(|(name, _)| (*name, std::env::var(name).ok()))
            .collect();
        for (name, value) in vars {
            set(name, *value);
        }
        Self(saved)
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        for (name, value) in &self.0 {
            set(name, value.as_deref());
        }
    }
}

fn set(name: &str, value: Option<&str>) {
    // SAFETY: every test of this binary holds `ENV_LOCK` while it changes
    // or reads the environment.
    unsafe {
        match value {
            Some(value) => std::env::set_var(name, value),
            None => std::env::remove_var(name),
        }
    }
}

/// `AZURE_OPENAI_ENDPOINT` (Azure's SDKs read it) wins over
/// `AZURE_OPENAI_RESOURCE`, and a malformed one is named.
#[test]
fn azure_reads_the_endpoint_before_the_resource() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _env = Env::set(&[
        ("AZURE_OPENAI_API_KEY", Some("key")),
        (
            "AZURE_OPENAI_ENDPOINT",
            Some("https://proj.services.ai.azure.com/"),
        ),
        ("AZURE_OPENAI_RESOURCE", Some("ignored")),
    ]);
    assert_eq!(
        Azure::from_env().unwrap().base_url(),
        "https://proj.services.ai.azure.com/openai/v1"
    );

    set("AZURE_OPENAI_ENDPOINT", None);
    assert_eq!(
        Azure::from_env().unwrap().base_url(),
        "https://ignored.openai.azure.com/openai/v1"
    );

    set("AZURE_OPENAI_ENDPOINT", Some("proj.services.ai.azure.com"));
    let err = Azure::from_env().unwrap_err();
    assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    assert!(err.to_string().contains("AZURE_OPENAI_ENDPOINT"), "{err}");

    set("AZURE_OPENAI_ENDPOINT", None);
    set("AZURE_OPENAI_RESOURCE", Some("  "));
    let err = Azure::from_env().unwrap_err();
    assert!(err.to_string().contains("AZURE_OPENAI_RESOURCE"), "{err}");
}

/// A blank `VLLM_BASE_URL` is unset; a malformed one is named.
#[test]
fn vllm_names_a_bad_base_url() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _env = Env::set(&[("VLLM_BASE_URL", Some(" ")), ("VLLM_API_KEY", None)]);
    assert_eq!(
        Vllm::from_env().unwrap().base_url(),
        "http://localhost:8000/v1"
    );

    set("VLLM_BASE_URL", Some("localhost:8000"));
    let err = Vllm::from_env().unwrap_err();
    assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    assert!(err.to_string().contains("VLLM_BASE_URL"), "{err}");
}
