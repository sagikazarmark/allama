//! The wrappers a [`Registry`](crate::Registry) lookup or a `wrap` puts
//! around a provider, from its [`ProviderSettings`].
//!
//! Outermost first: `Retry(Timeout(UnsupportedPolicy(provider)))`, each layer
//! only when its setting is on. Every branch builds its concrete stack, then
//! erases it once, so a stream is boxed once however many layers it passes.

use allama::wrappers::{RetryingProvider, TimeoutProvider, UnsupportedPolicyProvider};
use allama::{
    ChatProvider, DecisionProvider, DynChatProvider, DynDecisionProvider, DynEmbeddingProvider,
    EmbeddingProvider, Result,
};

use crate::ProviderSettings;
use crate::registry::Operation;

/// Refuse `settings` that a provider passed to `wrap` for `operation`
/// cannot have been built with, as a lookup of a constructor registered
/// without settings does.
pub(crate) fn refuse_unapplied(settings: &ProviderSettings, operation: Operation) -> Result<()> {
    match settings.constructor_setting(operation) {
        Some(setting) => Err(setting.refuse(
            &format!(
                "a provider passed to `{}::wrap` is already built",
                operation.provider_noun()
            ),
            &format!(
                "build it with the adapter's `{}` and leave the setting unset",
                setting.adapter_method
            ),
        )),
        None => Ok(()),
    }
}

/// The chat stack around `provider`.
pub(crate) fn chat<P>(provider: P, settings: &ProviderSettings) -> DynChatProvider
where
    P: ChatProvider + 'static,
{
    match settings.unsupported_policy {
        Some(policy) => chat_attempt(UnsupportedPolicyProvider::new(provider, policy), settings),
        None => chat_attempt(provider, settings),
    }
}

/// The embedding stack around `provider`.
pub(crate) fn embedding<P>(provider: P, settings: &ProviderSettings) -> DynEmbeddingProvider
where
    P: EmbeddingProvider + 'static,
{
    match settings.unsupported_policy {
        Some(policy) => {
            embedding_attempt(UnsupportedPolicyProvider::new(provider, policy), settings)
        }
        None => embedding_attempt(provider, settings),
    }
}

/// The decision stack around `provider`. Decision requests have no
/// unsupported-feature policy, so that setting does not apply.
pub(crate) fn decision<P>(provider: P, settings: &ProviderSettings) -> DynDecisionProvider
where
    P: DecisionProvider + 'static,
{
    match settings.attempt_timeout {
        Some(policy) => decision_retry(TimeoutProvider::new(provider, policy), settings),
        None => decision_retry(provider, settings),
    }
}

fn chat_attempt<P>(provider: P, settings: &ProviderSettings) -> DynChatProvider
where
    P: ChatProvider + 'static,
{
    match settings.attempt_timeout {
        Some(policy) => chat_retry(TimeoutProvider::new(provider, policy), settings),
        None => chat_retry(provider, settings),
    }
}

fn chat_retry<P>(provider: P, settings: &ProviderSettings) -> DynChatProvider
where
    P: ChatProvider + 'static,
{
    match &settings.retry {
        Some(policy) => DynChatProvider::new(RetryingProvider::new(provider, policy.clone())),
        None => DynChatProvider::new(provider),
    }
}

fn embedding_attempt<P>(provider: P, settings: &ProviderSettings) -> DynEmbeddingProvider
where
    P: EmbeddingProvider + 'static,
{
    match settings.attempt_timeout {
        Some(policy) => embedding_retry(TimeoutProvider::new(provider, policy), settings),
        None => embedding_retry(provider, settings),
    }
}

fn embedding_retry<P>(provider: P, settings: &ProviderSettings) -> DynEmbeddingProvider
where
    P: EmbeddingProvider + 'static,
{
    match &settings.retry {
        Some(policy) => DynEmbeddingProvider::new(RetryingProvider::new(provider, policy.clone())),
        None => DynEmbeddingProvider::new(provider),
    }
}

fn decision_retry<P>(provider: P, settings: &ProviderSettings) -> DynDecisionProvider
where
    P: DecisionProvider + 'static,
{
    match &settings.retry {
        Some(policy) => DynDecisionProvider::new(RetryingProvider::new(provider, policy.clone())),
        None => DynDecisionProvider::new(provider),
    }
}

#[cfg(test)]
mod tests {
    use allama::mock::{MockDecisionProvider, MockEmbeddingProvider, MockProvider};
    use allama::{CapabilitySupport, ChatCapability, ChatRequest, EmbeddingRequest, Error};

    use crate::{ProviderSettings, Registry, chat, decision, embedding};

    fn quick_retries() -> ProviderSettings {
        ProviderSettings::new().retry(Some(
            allama::wrappers::RetryPolicy::new()
                .base_delay(std::time::Duration::from_millis(1))
                .jitter(false),
        ))
    }

    /// What `wrap` and a lookup of a constructor registered without
    /// settings return for `settings`, per operation: the error message, or
    /// `None` when they build.
    fn refusals(settings: &ProviderSettings) -> Vec<(Option<String>, Option<String>)> {
        let mut registry = Registry::new();
        registry
            .register_chat("mock", || Ok(MockProvider::empty()))
            .register_embedding("mock", || Ok(MockEmbeddingProvider::with_vectors(vec![])))
            .register_decision("mock", || Ok(MockDecisionProvider::empty()));
        let message = |error: Error| {
            assert!(matches!(error, Error::Configuration { .. }), "{error:?}");
            error.message().to_owned()
        };
        vec![
            (
                chat::wrap(MockProvider::empty(), settings)
                    .err()
                    .map(message),
                registry
                    .chat_from_provider_with("mock", settings)
                    .err()
                    .map(message),
            ),
            (
                embedding::wrap(MockEmbeddingProvider::with_vectors(vec![]), settings)
                    .err()
                    .map(message),
                registry
                    .embedding_from_provider_with("mock", settings)
                    .err()
                    .map(message),
            ),
            (
                decision::wrap(MockDecisionProvider::empty(), settings)
                    .err()
                    .map(message),
                registry
                    .decision_from_provider_with("mock", settings)
                    .err()
                    .map(message),
            ),
        ]
    }

    /// The part of the message both refusals share.
    fn reason(message: &str) -> &str {
        let start = message.find("so it cannot be built with").expect(message);
        let end = message[start..].find(';').expect(message);
        &message[start..start + end]
    }

    #[test]
    fn wrap_refuses_the_settings_a_lookup_refuses() {
        let scope = ProviderSettings::new().credential_scope("tenant-a");
        for (wrapped, looked_up) in refusals(&scope) {
            let (wrapped, looked_up) = (wrapped.unwrap(), looked_up.unwrap());
            assert_eq!(
                reason(&wrapped),
                "so it cannot be built with a credential scope"
            );
            assert_eq!(reason(&wrapped), reason(&looked_up));
            assert!(wrapped.contains("`with_credential_scope`"), "{wrapped}");
        }

        let capabilities = ProviderSettings::new()
            .chat_capabilities(|_: &str, _: ChatCapability| Some(CapabilitySupport::Supported));
        let [chat, embedding, decision]: [_; 3] = refusals(&capabilities).try_into().unwrap();
        let (wrapped, looked_up) = (chat.0.unwrap(), chat.1.unwrap());
        assert_eq!(
            reason(&wrapped),
            "so it cannot be built with chat capability overrides"
        );
        assert_eq!(reason(&wrapped), reason(&looked_up));
        assert_eq!(
            wrapped,
            "a provider passed to `chat::wrap` is already built, so it cannot be built with \
             chat capability overrides; build it with the adapter's `with_chat_capabilities` \
             and leave the setting unset"
        );
        // Chat answers do not concern the other operations.
        assert_eq!(embedding, (None, None));
        assert_eq!(decision, (None, None));

        // Nor do the wrapper settings.
        for refusal in refusals(&quick_retries()) {
            assert_eq!(refusal, (None, None));
        }
    }

    #[tokio::test]
    async fn a_wrapped_provider_retries() {
        use allama::decision::{DecisionRequest, Question};
        use allama::{ChatProvider, DecisionProvider, EmbeddingProvider};

        let last_error = || Error::invalid_request("final");
        let mock = MockProvider::build(|b| b.error(Error::overloaded("busy")).error(last_error()));
        let provider = chat::wrap(mock.clone(), &quick_retries()).unwrap();
        let err = provider
            .chat(&ChatRequest::new("m").user("hi"))
            .await
            .unwrap_err();
        assert_eq!(err.message(), "final");
        assert_eq!(mock.call_count(), 2, "the overload was retried");

        let mock = MockEmbeddingProvider::new([Err(Error::overloaded("busy")), Err(last_error())]);
        let provider = embedding::wrap(mock.clone(), &quick_retries()).unwrap();
        let err = provider
            .embed(&EmbeddingRequest::new("e").input("text"))
            .await
            .unwrap_err();
        assert_eq!(err.message(), "final");
        assert_eq!(mock.requests().len(), 2);

        let mock = MockDecisionProvider::new([Err(Error::overloaded("busy")), Err(last_error())]);
        let provider = decision::wrap(mock.clone(), &quick_retries()).unwrap();
        let request = DecisionRequest::new("d", "x").question("q", Question::yes_no("?"));
        let err = provider.decide(&request).await.unwrap_err();
        assert_eq!(err.message(), "final");
        assert_eq!(mock.requests().len(), 2);
    }
}
