//! The Responses API keys: `openai-responses` builds what `openai` builds,
//! and `azure-responses` the Azure preset's Responses provider. A test
//! binary of its own, since it sets the providers' environment variables.
#![cfg(all(feature = "openai", feature = "openai-compat"))]

use allama::{ProviderIdentity, ProviderIdentityExt};
use allama_openai_compat::providers::Azure;
use allama_providers::ProviderSettings;

#[test]
fn responses_keys_build_responses_providers() {
    // SAFETY: the only test of this binary, so nothing reads the
    // environment concurrently.
    unsafe {
        std::env::set_var("OPENAI_API_KEY", "openai-key");
        std::env::remove_var("OPENAI_BASE_URL");
        std::env::set_var("AZURE_OPENAI_API_KEY", "azure-key");
        std::env::set_var("AZURE_OPENAI_RESOURCE", "res");
        std::env::remove_var("AZURE_OPENAI_ENDPOINT");
    }

    for address in ["openai:gpt-5.5", "openai-responses:gpt-5.5"] {
        let (provider, model) = allama_providers::chat::from_model(address).unwrap();
        assert_eq!(model, "gpt-5.5");
        assert_eq!(provider.provider_name(), allama_openai::PROVIDER_NAME);
        assert!(
            provider
                .downcast_ref::<allama_openai::ResponsesProvider>()
                .is_some(),
            "{address}"
        );
    }

    let (provider, model) =
        allama_providers::chat::from_model("azure-responses:my-deployment").unwrap();
    assert_eq!(model, "my-deployment");
    assert_eq!(provider.provider_name(), Azure::PROVIDER_NAME);
    let responses = provider
        .downcast_ref::<allama_openai_compat::ResponsesProvider>()
        .expect("the Azure Responses provider");
    assert_eq!(
        responses.responses_url(),
        "https://res.openai.azure.com/openai/v1/responses"
    );
    assert_eq!(
        responses.credential_slot().scope,
        Azure::DEFAULT_CREDENTIAL_SCOPE
    );
    // `azure` keeps Chat Completions.
    let (provider, _) = allama_providers::chat::from_model("azure:my-deployment").unwrap();
    assert!(
        provider
            .downcast_ref::<allama_openai_compat::Provider>()
            .is_some()
    );

    // A scoped lookup reaches the Responses provider too.
    let settings = ProviderSettings::new().credential_scope("tenant-a");
    let (provider, _) =
        allama_providers::chat::from_model_with("azure-responses:d", &settings).unwrap();
    let responses = provider
        .downcast_ref::<allama_openai_compat::ResponsesProvider>()
        .unwrap();
    assert_eq!(responses.credential_slot().scope, "tenant-a");

    // The API-named keys offer chat only.
    for key in ["openai-responses", "azure-responses"] {
        let err = allama_providers::embedding::from_provider(key).unwrap_err();
        assert!(
            err.message()
                .starts_with(&format!("provider `{key}` does not offer embeddings")),
            "{}",
            err.message()
        );
    }
}
