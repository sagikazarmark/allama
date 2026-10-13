//! The built-in providers and the features that enable them.

use std::sync::Arc;

use crate::registry::Operation::{self, Chat, Decision, Embedding};
use crate::{AdapterSettings, Registry};

/// Every built-in provider, whether or not its feature is on. A key may
/// appear more than once when different crates provide its operations.
const BUILTINS: &[Builtin] = &[
    Builtin::new(
        "anthropic",
        &[Chat],
        "anthropic",
        cfg!(feature = "anthropic"),
    ),
    Builtin::new(
        "gemini",
        &[Chat, Embedding],
        "gemini",
        cfg!(feature = "gemini"),
    ),
    Builtin::new(
        "openai",
        &[Chat, Embedding],
        "openai",
        cfg!(feature = "openai"),
    ),
    Builtin::new("openai-chat", &[Chat], "openai", cfg!(feature = "openai")),
    Builtin::compat("azure"),
    Builtin::compat("cloudflare"),
    Builtin::compat("deepinfra"),
    Builtin::compat("deepseek"),
    Builtin::compat("fireworks"),
    Builtin::compat("groq"),
    Builtin::compat("mistral"),
    Builtin::compat("ollama"),
    Builtin::compat("openrouter"),
    Builtin::compat("together"),
    Builtin::compat("vllm"),
    Builtin::compat("xai"),
    Builtin::systemone("cloudflare"),
    Builtin::systemone("digitalocean"),
    Builtin::systemone("litellm"),
    Builtin::systemone("ollama"),
    Builtin::systemone("openrouter"),
    Builtin::systemone("typesafe"),
];

/// Register every built-in provider enabled by crate features.
pub(crate) fn register(registry: &mut Registry) {
    #[cfg(feature = "anthropic")]
    registry.register_chat_with(
        "anthropic",
        configured(
            allama_anthropic::Provider::from_env,
            allama_anthropic::Provider::with_credential_scope,
            allama_anthropic::Provider::with_chat_capabilities,
        ),
    );

    #[cfg(feature = "gemini")]
    {
        let gemini = || {
            configured(
                allama_gemini::Provider::from_env,
                allama_gemini::Provider::with_credential_scope,
                allama_gemini::Provider::with_chat_capabilities,
            )
        };
        registry
            .register_chat_with("gemini", gemini())
            .register_embedding_with("gemini", gemini());
    }

    #[cfg(feature = "openai")]
    {
        let chat = || {
            configured(
                allama_openai::Provider::from_env,
                allama_openai::Provider::with_credential_scope,
                allama_openai::Provider::with_chat_capabilities,
            )
        };
        registry
            .register_chat_with(
                "openai",
                configured(
                    allama_openai::ResponsesProvider::from_env,
                    allama_openai::ResponsesProvider::with_credential_scope,
                    allama_openai::ResponsesProvider::with_chat_capabilities,
                ),
            )
            .register_embedding_with("openai", chat())
            .register_chat_with("openai-chat", chat());
    }

    #[cfg(feature = "openai-compat")]
    {
        use allama_openai_compat::providers::{
            Azure, Cloudflare, DeepInfra, DeepSeek, Fireworks, Groq, Mistral, Ollama, OpenRouter,
            Together, Vllm, XAi,
        };

        type FromEnv = fn() -> allama::Result<allama_openai_compat::Provider>;

        let presets: [(&str, FromEnv); 12] = [
            ("azure", Azure::from_env),
            ("cloudflare", Cloudflare::from_env),
            ("deepinfra", DeepInfra::from_env),
            ("deepseek", DeepSeek::from_env),
            ("fireworks", Fireworks::from_env),
            ("groq", Groq::from_env),
            ("mistral", Mistral::from_env),
            ("ollama", Ollama::from_env),
            ("openrouter", OpenRouter::from_env),
            ("together", Together::from_env),
            ("vllm", Vllm::from_env),
            ("xai", XAi::from_env),
        ];
        for (key, from_env) in presets {
            let preset = || {
                configured(
                    from_env,
                    allama_openai_compat::Provider::with_credential_scope,
                    allama_openai_compat::Provider::with_chat_capabilities,
                )
            };
            registry
                .register_chat_with(key, preset())
                .register_embedding_with(key, preset());
        }
    }

    #[cfg(feature = "systemone")]
    {
        use allama_systemone::providers::{
            Cloudflare, DigitalOcean, LiteLlm, Ollama, OpenRouter, TypeSafe,
        };

        type FromEnv = fn() -> allama::Result<allama_systemone::Provider>;

        let presets: [(&str, FromEnv); 6] = [
            ("cloudflare", Cloudflare::from_env),
            ("digitalocean", DigitalOcean::from_env),
            ("litellm", LiteLlm::from_env),
            ("ollama", Ollama::from_env),
            ("openrouter", OpenRouter::from_env),
            ("typesafe", TypeSafe::from_env),
        ];
        for (key, from_env) in presets {
            registry.register_decision_with(
                key,
                scoped(from_env, allama_systemone::Provider::with_credential_scope),
            );
        }
    }

    let _ = registry;
}

/// A constructor that builds with `from_env`, then applies the lookup's
/// credential scope, if any, with `with_scope`.
#[allow(dead_code)]
fn scoped<P>(
    from_env: fn() -> allama::Result<P>,
    with_scope: fn(P, String) -> P,
) -> impl Fn(&mut AdapterSettings) -> allama::Result<P> + Send + Sync + 'static
where
    P: 'static,
{
    move |settings| {
        let provider = from_env()?;
        Ok(match settings.take_credential_scope() {
            Some(scope) => with_scope(provider, scope),
            None => provider,
        })
    }
}

/// [`scoped`], then the lookup's chat capability overrides, if any, applied
/// with `with_capabilities`. An embedding lookup has none to apply.
#[allow(dead_code)]
fn configured<P>(
    from_env: fn() -> allama::Result<P>,
    with_scope: fn(P, String) -> P,
    with_capabilities: fn(P, Arc<dyn allama::ChatCapabilityResolver>) -> P,
) -> impl Fn(&mut AdapterSettings) -> allama::Result<P> + Send + Sync + 'static
where
    P: 'static,
{
    let scoped = scoped(from_env, with_scope);
    move |settings| {
        let provider = scoped(settings)?;
        Ok(match settings.take_chat_capabilities() {
            Some(resolver) => with_capabilities(provider, resolver),
            None => provider,
        })
    }
}

/// The feature that would enable `operation` for `key`, when a built-in
/// provider offers it behind a feature that is off.
pub(crate) fn disabled_feature(key: &str, operation: Operation) -> Option<&'static str> {
    BUILTINS
        .iter()
        .find(|b| b.key == key && b.operations.contains(&operation) && !b.enabled)
        .map(|b| b.feature)
}

struct Builtin {
    key: &'static str,
    operations: &'static [Operation],
    feature: &'static str,
    enabled: bool,
}

impl Builtin {
    const fn new(
        key: &'static str,
        operations: &'static [Operation],
        feature: &'static str,
        enabled: bool,
    ) -> Self {
        Self {
            key,
            operations,
            feature,
            enabled,
        }
    }

    const fn compat(key: &'static str) -> Self {
        Self::new(
            key,
            &[Chat, Embedding],
            "openai-compat",
            cfg!(feature = "openai-compat"),
        )
    }

    const fn systemone(key: &'static str) -> Self {
        Self::new(key, &[Decision], "systemone", cfg!(feature = "systemone"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_operations_match_enabled_features() {
        let registry = Registry::builtin();
        for builtin in BUILTINS {
            for &operation in builtin.operations {
                assert_eq!(
                    registry.supports(builtin.key, operation),
                    builtin.enabled,
                    "{} {operation}",
                    builtin.key
                );
            }
        }
        for key in registry.keys() {
            assert!(
                BUILTINS.iter().any(|b| b.key == key),
                "{key} is registered but not listed"
            );
        }
    }

    #[test]
    fn every_builtin_key_is_a_valid_provider_key() {
        for builtin in BUILTINS {
            assert!(
                crate::address::is_valid_key(builtin.key),
                "{} breaks the key grammar",
                builtin.key
            );
        }
    }

    #[cfg(feature = "openai-compat")]
    #[test]
    fn model_address_reaches_a_builtin_provider() {
        use allama::ProviderIdentity;

        // Ollama needs no credentials.
        let (provider, model) = crate::chat::from_model("ollama:llama3:8b").unwrap();
        assert_eq!(provider.provider_name(), "ollama");
        assert_eq!(model, "llama3:8b");

        let (provider, _) = crate::embedding::from_model("ollama:nomic-embed-text").unwrap();
        assert_eq!(provider.provider_name(), "ollama");
    }

    #[cfg(all(feature = "openai-compat", feature = "systemone"))]
    #[test]
    fn a_scoped_lookup_builds_a_builtin_with_the_scope() {
        use allama::{ProviderIdentity, ProviderIdentityExt};

        let scope = |provider: &dyn ProviderIdentity| {
            let slot = provider
                .downcast_ref::<allama_openai_compat::Provider>()
                .map(|p| p.credential_slot().clone())
                .or_else(|| {
                    provider
                        .downcast_ref::<allama_systemone::Provider>()
                        .map(|p| p.credential_slot().clone())
                })
                .expect("a built-in provider");
            (slot.scope.into_owned(), slot.required)
        };
        let settings = crate::ProviderSettings::new().credential_scope("tenant-a");

        // Ollama needs no credentials, so it builds without any.
        let (provider, model) = crate::chat::from_model_with("ollama:llama3", &settings).unwrap();
        assert_eq!(model, "llama3");
        // The identity is unchanged; with no configured key and a chosen
        // scope, every request must bring its own key.
        assert_eq!(provider.provider_name(), "ollama");
        assert_eq!(scope(&provider), ("tenant-a".to_owned(), true));
        let (provider, _) = crate::embedding::from_model_with("ollama:e", &settings).unwrap();
        assert_eq!(scope(&provider), ("tenant-a".to_owned(), true));
        let (provider, _) = crate::decision::from_model_with("ollama:d", &settings).unwrap();
        assert_eq!(scope(&provider), ("tenant-a".to_owned(), true));

        // The default scope is the provider name.
        let (provider, _) = crate::chat::from_model("ollama:llama3").unwrap();
        assert_eq!(scope(&provider), ("ollama".to_owned(), false));
        assert_eq!(
            allama_openai_compat::providers::Ollama::DEFAULT_CREDENTIAL_SCOPE,
            "ollama"
        );
    }

    /// Capability overrides reach the adapter, so they can upgrade an
    /// answer it gives as `Unsupported` (the Ollama preset ignores
    /// `tool_choice`).
    #[cfg(feature = "openai-compat")]
    #[test]
    fn chat_capabilities_reach_a_builtin_adapter() {
        use allama::{CapabilitySupport, ChatCapability, ChatProvider};

        let (provider, model) = crate::chat::from_model("ollama:llama3").unwrap();
        assert_eq!(
            provider.chat_capability(&model, ChatCapability::ForcedToolChoice),
            CapabilitySupport::Unsupported
        );

        let settings =
            crate::ProviderSettings::new().chat_capabilities(|model: &str, capability| {
                (model == "llama3" && capability == ChatCapability::ForcedToolChoice)
                    .then_some(CapabilitySupport::Supported)
            });
        let (provider, model) = crate::chat::from_model_with("ollama:llama3", &settings).unwrap();
        assert_eq!(
            provider.chat_capability(&model, ChatCapability::ForcedToolChoice),
            CapabilitySupport::Supported
        );
    }

    #[cfg(feature = "anthropic")]
    #[test]
    fn anthropic_has_no_embeddings() {
        let err = crate::embedding::from_provider("anthropic").unwrap_err();
        assert!(
            matches!(err, allama::Error::Configuration { .. }),
            "{err:?}"
        );
        assert!(
            err.message()
                .starts_with("provider `anthropic` does not offer embeddings (it offers chat); "),
            "{}",
            err.message()
        );
    }

    #[cfg(feature = "systemone")]
    #[test]
    fn model_address_reaches_a_builtin_decision_provider() {
        use allama::ProviderIdentity;

        let (provider, model) = crate::decision::from_model("ollama:nimble").unwrap();
        assert_eq!(provider.provider_name(), "ollama");
        assert_eq!(model, "nimble");
    }

    #[cfg(feature = "systemone")]
    #[test]
    fn decision_only_provider_has_no_chat() {
        let err = crate::chat::from_provider("typesafe").unwrap_err();
        assert!(
            matches!(err, allama::Error::Configuration { .. }),
            "{err:?}"
        );
        assert!(
            err.message()
                .starts_with("provider `typesafe` does not offer chat (it offers decisions); "),
            "{}",
            err.message()
        );
        // Only the keys that offer chat are listed.
        assert!(!err.message().contains("litellm"), "{}", err.message());
    }

    #[cfg(all(feature = "systemone", not(feature = "openai-compat")))]
    #[test]
    fn shared_key_names_the_feature_of_the_missing_operation() {
        let err = crate::chat::from_provider("ollama").unwrap_err();
        assert!(
            err.message()
                .contains("requires the `openai-compat` feature"),
            "{}",
            err.message()
        );
    }

    /// Every built-in provider's `gen_ai.provider.name` on tracing spans: the
    /// well-known value when the conventions define one, else its provider
    /// name unchanged.
    #[cfg(all(
        feature = "anthropic",
        feature = "gemini",
        feature = "openai",
        feature = "openai-compat",
        feature = "systemone"
    ))]
    #[test]
    fn provider_names_map_to_semconv_well_known_values() {
        use allama::wrappers::otel_genai_provider_name;
        use allama_openai_compat::providers as compat;
        use allama_systemone::providers as systemone;

        // `gen_ai.provider.name` well-known values, OpenTelemetry semantic
        // conventions v1.41.0 (`docs/registry/attributes/gen-ai.md`).
        const WELL_KNOWN: &[&str] = &[
            "anthropic",
            "aws.bedrock",
            "azure.ai.inference",
            "azure.ai.openai",
            "cohere",
            "deepseek",
            "gcp.gemini",
            "gcp.gen_ai",
            "gcp.vertex_ai",
            "groq",
            "ibm.watsonx.ai",
            "mistral_ai",
            "openai",
            "perplexity",
            "x_ai",
        ];

        // (registry key, provider name, `gen_ai.provider.name`); `None` is
        // pass-through.
        let cases: &[(&str, &str, Option<&str>)] = &[
            (
                "anthropic",
                allama_anthropic::PROVIDER_NAME,
                Some("anthropic"),
            ),
            ("gemini", allama_gemini::PROVIDER_NAME, Some("gcp.gemini")),
            ("openai", allama_openai::PROVIDER_NAME, Some("openai")),
            ("openai-chat", allama_openai::PROVIDER_NAME, Some("openai")),
            (
                "azure",
                compat::Azure::PROVIDER_NAME,
                Some("azure.ai.openai"),
            ),
            ("cloudflare", compat::Cloudflare::PROVIDER_NAME, None),
            ("deepinfra", compat::DeepInfra::PROVIDER_NAME, None),
            (
                "deepseek",
                compat::DeepSeek::PROVIDER_NAME,
                Some("deepseek"),
            ),
            ("fireworks", compat::Fireworks::PROVIDER_NAME, None),
            ("groq", compat::Groq::PROVIDER_NAME, Some("groq")),
            (
                "mistral",
                compat::Mistral::PROVIDER_NAME,
                Some("mistral_ai"),
            ),
            ("ollama", compat::Ollama::PROVIDER_NAME, None),
            ("openrouter", compat::OpenRouter::PROVIDER_NAME, None),
            ("together", compat::Together::PROVIDER_NAME, None),
            ("vllm", compat::Vllm::PROVIDER_NAME, None),
            ("xai", compat::XAi::PROVIDER_NAME, Some("x_ai")),
            ("cloudflare", systemone::Cloudflare::PROVIDER_NAME, None),
            ("digitalocean", systemone::DigitalOcean::PROVIDER_NAME, None),
            ("litellm", systemone::LiteLlm::PROVIDER_NAME, None),
            ("ollama", systemone::Ollama::PROVIDER_NAME, None),
            ("openrouter", systemone::OpenRouter::PROVIDER_NAME, None),
            ("typesafe", systemone::TypeSafe::PROVIDER_NAME, None),
            // The crates' defaults without a preset.
            ("", allama_openai_compat::PROVIDER_NAME, None),
            ("", allama_systemone::PROVIDER_NAME, None),
        ];

        for &(key, name, expected) in cases {
            let mapped = otel_genai_provider_name(name);
            match expected {
                Some(value) => {
                    assert!(WELL_KNOWN.contains(&value), "{value} is not well-known");
                    assert_eq!(mapped, value, "{key} ({name})");
                }
                None => {
                    assert_eq!(mapped, name, "{key} ({name}) should pass through");
                    assert!(
                        !WELL_KNOWN.contains(&name),
                        "{key} ({name}) is itself a well-known value"
                    );
                }
            }
        }
        for builtin in BUILTINS {
            assert!(
                cases.iter().any(|(key, ..)| *key == builtin.key),
                "{} has no row",
                builtin.key
            );
        }
    }

    #[cfg(not(feature = "anthropic"))]
    #[test]
    fn disabled_builtin_names_its_feature() {
        let err = crate::chat::from_provider("anthropic").unwrap_err();
        assert!(
            matches!(err, allama::Error::Configuration { .. }),
            "{err:?}"
        );
        assert!(
            err.message().contains("requires the `anthropic` feature"),
            "{}",
            err.message()
        );
    }
}
