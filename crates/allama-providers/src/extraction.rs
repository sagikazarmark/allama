//! Extraction providers from the built-in registry.
//!
//! Unstable, like [`ExtractionProvider`] (ADR 0018): these lookups may
//! change in any minor release, an exception to the promise a built-in key
//! otherwise makes (ADR 0016).

use allama::{DynExtractionProvider, ExtractionProvider, Result};

use crate::ProviderSettings;
use crate::registry::{Operation, builtin};
use crate::stack;

/// Build the extraction provider for a provider key, such as
/// `"cloudflare"`, configured from the environment.
///
/// # Errors
///
/// Fails with [`Error::Configuration`](allama::Error::Configuration) for an
/// unknown key, one whose feature is off, or a provider without extraction
/// models (such as `anthropic`); the message lists the keys that offer
/// extractions. Fails with the provider's error when its environment is
/// incomplete.
pub fn from_provider(key: &str) -> Result<DynExtractionProvider> {
    builtin().extraction_from_provider(key)
}

/// Build the extraction provider for a `provider:model` address, such as
/// `"cloudflare:@cf/cloudflare/clairic"`, and return it with the model id.
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
pub fn from_model(address: &str) -> Result<(DynExtractionProvider, String)> {
    builtin().extraction_from_model(address)
}

/// Build the extraction provider for a provider key with `settings`, such as a
/// credential scope (see [`ProviderSettings`]).
///
/// # Errors
///
/// As [`from_provider`].
pub fn from_provider_with(key: &str, settings: &ProviderSettings) -> Result<DynExtractionProvider> {
    builtin().extraction_from_provider_with(key, settings)
}

/// Build the extraction provider for a `provider:model` address with `settings`,
/// such as a credential scope (see [`ProviderSettings`]), and return it
/// with the model id.
///
/// # Errors
///
/// As [`from_model`].
pub fn from_model_with(
    address: &str,
    settings: &ProviderSettings,
) -> Result<(DynExtractionProvider, String)> {
    builtin().extraction_from_model_with(address, settings)
}

/// Wrap an extraction provider you built yourself in the stack `settings`
/// describe: retries and the per-attempt timeout (extraction requests have no
/// unsupported-feature policy, so that setting is ignored, as in a lookup),
/// the same wrappers a lookup puts around a registered provider (see [The
/// default stack](ProviderSettings#the-default-stack)).
///
/// The adapter is already built, so `wrap` cannot apply the settings only a
/// constructor can. Set the scope on the adapter instead: `credential_scope`
/// on its builder, or `with_credential_scope` on a built provider.
/// The result already retries with [`ProviderSettings::retry`]; do not wrap
/// it in another [`RetryingProvider`](allama::wrappers::RetryingProvider).
///
/// # Errors
///
/// Fails with [`Error::Configuration`](allama::Error::Configuration) when
/// `settings` set a credential scope, which a lookup of a constructor
/// registered without settings refuses too.
pub fn wrap<P>(provider: P, settings: &ProviderSettings) -> Result<DynExtractionProvider>
where
    P: ExtractionProvider + 'static,
{
    stack::refuse_unapplied(settings, Operation::Extraction)?;
    Ok(stack::extraction(provider, settings))
}
