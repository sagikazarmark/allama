//! Chat providers from the built-in registry.

use allama::{ChatProvider, DynChatProvider, Result};

use crate::ProviderSettings;
use crate::registry::{Operation, builtin};
use crate::stack;

/// Build the chat provider for a provider key, such as `"anthropic"`,
/// configured from the environment.
///
/// # Errors
///
/// Fails with [`Error::Configuration`](allama::Error::Configuration) for an
/// unknown key, one whose feature is off, or a provider without chat (such
/// as `typesafe`); the message lists the keys that offer chat. Fails with
/// the provider's error when its environment is incomplete (for example
/// [`Error::Configuration`](allama::Error::Configuration) for a missing API
/// key).
pub fn from_provider(key: &str) -> Result<DynChatProvider> {
    builtin().chat_from_provider(key)
}

/// Build the chat provider for a `provider:model` address, such as
/// `"openai:gpt-5.5"`, and return it with the model id.
///
/// # Retries by default
///
/// The provider comes wrapped in a
/// [`RetryingProvider`](allama::wrappers::RetryingProvider) with
/// [`RetryPolicy::default()`](allama::wrappers::RetryPolicy): up to 3
/// attempts, backoff from 250 ms to 5 s with jitter, and the provider's
/// `Retry-After` waited out **up to 120 s** (a longer one gives up at
/// once). A call can take minutes before its error reaches you (see
/// [Worst-case latency](ProviderSettings#worst-case-latency)).
/// [`from_provider`] is the same.
///
/// - **Interactive servers and fallback pairs** should not wait that long:
///   sides built with the defaults wait out a `Retry-After` of up to 120 s
///   before a [`FallbackProvider`](allama::wrappers::FallbackProvider) sees
///   the error. Use [`from_model_with`] with
///   `ProviderSettings::new().retry(Some(RetryPolicy::new().max_retry_after(Duration::from_secs(5))))`.
/// - **Do not nest retries**: wrapping the result in another
///   `RetryingProvider` multiplies attempts and waits. Configure retries
///   with [`ProviderSettings::retry`], or turn them off with `retry(None)`.
///
/// See [The default stack](ProviderSettings#the-default-stack).
///
/// # Custom providers
///
/// This function uses the built-in registry, which cannot be changed. For
/// a provider key of your own, register a constructor in a
/// [`Registry`](crate::Registry) and look the address up there.
///
/// # Errors
///
/// Fails with [`Error::Configuration`](allama::Error::Configuration) when
/// the address is not a [`ModelAddress`](crate::ModelAddress) (whitespace,
/// an uppercase key or a missing `:` included), and as [`from_provider`]
/// otherwise.
pub fn from_model(address: &str) -> Result<(DynChatProvider, String)> {
    builtin().chat_from_model(address)
}

/// Build the chat provider for a provider key with `settings`, such as a
/// credential scope (see [`ProviderSettings`]).
///
/// # Errors
///
/// As [`from_provider`].
pub fn from_provider_with(key: &str, settings: &ProviderSettings) -> Result<DynChatProvider> {
    builtin().chat_from_provider_with(key, settings)
}

/// Build the chat provider for a `provider:model` address with `settings`,
/// such as a credential scope (see [`ProviderSettings`]), and return it
/// with the model id.
///
/// # Errors
///
/// As [`from_model`].
pub fn from_model_with(
    address: &str,
    settings: &ProviderSettings,
) -> Result<(DynChatProvider, String)> {
    builtin().chat_from_model_with(address, settings)
}

/// Wrap a chat provider you built yourself in the stack `settings`
/// describe: retries, the per-attempt timeout and the unsupported-feature
/// policy, the same wrappers a lookup puts around a registered provider
/// (see [The default stack](ProviderSettings#the-default-stack)).
///
/// The adapter is already built, so `wrap` cannot apply the settings only a
/// constructor can. Set the scope on the adapter instead: `credential_scope`
/// on its builder, or `with_credential_scope` on a built provider; and the
/// capability overrides with its `with_chat_capabilities`. The result
/// already retries with [`ProviderSettings::retry`]; do not wrap it in
/// another [`RetryingProvider`](allama::wrappers::RetryingProvider).
///
/// ```no_run
/// # #[cfg(feature = "openai-compat")] {
/// use allama_openai_compat::providers::Vllm;
/// use allama_providers::ProviderSettings;
///
/// # fn example() -> allama::Result<()> {
/// let adapter = Vllm::new("http://gpu-box:8000/v1", None)?;
/// let provider = allama_providers::chat::wrap(adapter, &ProviderSettings::new())?;
/// # let _ = provider; Ok(()) }
/// # }
/// ```
///
/// # Errors
///
/// Fails with [`Error::Configuration`](allama::Error::Configuration) when
/// `settings` set a credential scope or chat capability overrides, which a
/// lookup of a constructor registered without settings refuses too.
pub fn wrap<P>(provider: P, settings: &ProviderSettings) -> Result<DynChatProvider>
where
    P: ChatProvider + 'static,
{
    stack::refuse_unapplied(settings, Operation::Chat)?;
    Ok(stack::chat(provider, settings))
}
