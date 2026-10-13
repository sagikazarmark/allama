use std::any::{Any, TypeId};
use std::sync::Arc;

/// Shared provider identity surface for chat, embedding, and decision
/// providers: the provider's name, and typed lookup of the objects it exposes.
///
/// [`provider_name`](Self::provider_name) returns the provider identity
/// string used for logs, tracing, errors, fixtures, and
/// [`ChatResponse::provider`](crate::ChatResponse::provider) (for example
/// `"anthropic"` or `"openai"`). It is an open string surface rather than a
/// closed enum so wrappers and custom providers can preserve their existing
/// naming, and it borrows from the provider so names can come from
/// configuration. Callers should not treat it as a stable capability-dispatch
/// mechanism.
///
/// [`downcast_any`](Self::downcast_any) reaches typed objects a provider
/// exposes (most often the concrete provider itself, for its
/// provider-specific methods) through wrappers and type erasure. Call it
/// through [`ProviderIdentityExt::downcast_ref`].
///
/// # Provider name and namespace
///
/// A provider has two identifiers, and they need not be equal:
///
/// - the **provider name** ([`provider_name`](Self::provider_name)) says who
///   served a call, for people and telemetry;
/// - the **namespace** ([`ProviderOptions::NAMESPACE`](crate::ProviderOptions::NAMESPACE),
///   [`ProviderDataEntry::NAMESPACE`](crate::ProviderDataEntry::NAMESPACE))
///   keys the provider's typed options and the opaque state it stores in
///   [`ProviderData`](crate::ProviderData). The namespace is the origin of
///   that state: a provider replays only state under its own namespace.
///
/// Gemini is the example where they differ: its provider name is `gemini`
/// and its namespace is `google-ai`. Several OpenAI-compatible endpoints
/// share options under one namespace while each reports its own name. Every
/// built-in provider crate exports both as `PROVIDER_NAME` and `NAMESPACE`.
///
/// [`reads_namespace`](Self::reads_namespace) answers which namespaces a
/// provider reads, so code that holds only a provider (one built from a
/// model address, behind wrappers) can keep the state it uses, with
/// [`retain_provider_state_for`](crate::history::retain_provider_state_for).
///
/// # Implementing
///
/// - [`provider_name`](Self::provider_name) is required: a provider names
///   itself, and a wrapper forwards the name of the provider it wraps
///   (`self.inner.provider_name()`).
/// - [`reads_namespace`](Self::reads_namespace) is required: a provider
///   answers for its own namespace, and a wrapper forwards the question
///   (`self.inner.reads_namespace(namespace)`).
/// - A provider exposes itself, and optionally objects it owns, with
///   [`expose`].
/// - A wrapper forwards the query to every provider it wraps:
///   `self.inner.downcast_any(id)`. A wrapper that does not forward hides
///   the providers it wraps.
///
/// Wrappers are transparent: they forward every question about the provider
/// (its name, the namespaces it reads, [`downcast_any`](Self::downcast_any),
/// its capabilities, its limits, and request validation) and do not answer
/// for themselves. The methods without a sensible wrapper default (the name,
/// the namespaces and the capability answers) are required, so a wrapper
/// cannot forget them silently; the rest have defaults that are right for
/// providers and wrong for wrappers.
/// `allama_conformance::assert_wrapper` (and its per-trait variants)
/// checks a wrapper against all of them, and that its calls pass the
/// provider's answers through unchanged.
pub trait ProviderIdentity: Send + Sync {
    /// Returns the provider name: who serves calls made through this
    /// provider.
    fn provider_name(&self) -> &str;

    /// Whether this provider reads the [provider data](crate::ProviderData)
    /// under `namespace`.
    ///
    /// A namespace names the only provider that may read an entry: the
    /// origin of the state a provider replays, and the addressee of
    /// settings a caller attaches to a request item. A provider answers
    /// `true` for the namespace it mints and reads state under (Gemini for
    /// `google-ai`, an OpenAI-compatible preset for `openai-compat:groq`,
    /// both OpenAI endpoints for `openai`) and `false` for every other
    /// namespace, including options namespaces that hold no provider data
    /// (`openai-responses`, `allama-http`) and the `allama` namespace, which
    /// allama itself reads.
    ///
    /// It is a question, not an identifier: a wrapper answers for the
    /// providers that may serve its calls, which one namespace cannot
    /// name. A wrapper forwards it, and a
    /// [`FallbackProvider`](crate::wrappers::FallbackProvider) reads what
    /// either side reads, since either may serve the next turn.
    ///
    /// Required, like [`provider_name`](Self::provider_name): no default is
    /// right for both a provider and a wrapper, and a wrong answer silently
    /// removes state a provider needs from a conversation
    /// ([`retain_provider_state_for`](crate::history::retain_provider_state_for)).
    fn reads_namespace(&self, namespace: &str) -> bool;

    /// Returns the object of type `id` this provider exposes, if any.
    ///
    /// Concrete providers usually expose themselves (see [`expose`]);
    /// wrappers forward the query to the providers they wrap. The default
    /// exposes nothing.
    fn downcast_any(&self, id: TypeId) -> Option<&(dyn Any + Send + Sync)> {
        let _ = id;
        None
    }
}

/// Typed access to [`ProviderIdentity::downcast_any`].
pub trait ProviderIdentityExt: ProviderIdentity {
    /// Borrow the `T` this provider, or a provider it wraps, exposes.
    ///
    /// Like `anyhow::Error::downcast_ref`, this looks through layers: it
    /// reaches a concrete provider (and its provider-specific methods)
    /// through wrappers and [`DynChatProvider`](crate::DynChatProvider):
    ///
    /// ```
    /// # #[cfg(feature = "mock")] {
    /// use allama::mock::MockProvider;
    /// use allama::wrappers::{RetryPolicy, RetryingProvider};
    /// use allama::{DynChatProvider, ProviderIdentityExt};
    ///
    /// let retrying = RetryingProvider::new(MockProvider::empty(), RetryPolicy::default());
    /// let provider = DynChatProvider::new(retrying);
    /// assert!(provider.downcast_ref::<MockProvider>().is_some());
    /// # }
    /// ```
    ///
    /// - **Calls through the result skip the wrappers.** The reference points
    ///   at the inner provider, so its methods run without the retries,
    ///   timeouts, tracing, or recording around it.
    /// - **The first match wins.** A [`FallbackProvider`] answers from its
    ///   primary first; when both sides have the same type, reach each one
    ///   with [`primary`] and [`fallback`].
    /// - **Discovery is for code handed an already-wrapped provider.** An
    ///   application that builds the stack can keep a clone (or an [`Arc`])
    ///   of the concrete provider before wrapping it.
    ///
    /// [`FallbackProvider`]: crate::wrappers::FallbackProvider
    /// [`primary`]: crate::wrappers::FallbackProvider::primary
    /// [`fallback`]: crate::wrappers::FallbackProvider::fallback
    fn downcast_ref<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.downcast_any(TypeId::of::<T>())?.downcast_ref()
    }
}

impl<P: ProviderIdentity + ?Sized> ProviderIdentityExt for P {}

/// Answer a [`downcast_any`](ProviderIdentity::downcast_any) query with
/// `value` when `id` is its type.
///
/// A concrete provider exposes itself, and optionally objects it owns:
///
/// ```
/// use std::any::{Any, TypeId};
/// use allama::{ProviderIdentity, ProviderIdentityExt, expose};
///
/// struct MyProvider {
///     config: MyConfig,
/// }
///
/// struct MyConfig {
///     region: &'static str,
/// }
///
/// impl ProviderIdentity for MyProvider {
///     fn provider_name(&self) -> &str {
///         "mine"
///     }
///
///     fn reads_namespace(&self, namespace: &str) -> bool {
///         namespace == "mine"
///     }
///
///     fn downcast_any(&self, id: TypeId) -> Option<&(dyn Any + Send + Sync)> {
///         expose(self, id).or_else(|| expose(&self.config, id))
///     }
/// }
///
/// let provider = MyProvider { config: MyConfig { region: "eu" } };
/// assert_eq!(provider.downcast_ref::<MyConfig>().unwrap().region, "eu");
/// ```
///
/// A wrapper forwards the query instead: `self.inner.downcast_any(id)`. To
/// make a wrapper discoverable as well, answer
/// `expose(self, id).or_else(|| self.inner.downcast_any(id))`, which needs
/// `P: 'static` on the `impl`. This is discouraged: a caller that looks a
/// wrapper up depends on how the stack was built, which a wrapper should
/// not make observable.
pub fn expose<T: Any + Send + Sync>(value: &T, id: TypeId) -> Option<&(dyn Any + Send + Sync)> {
    (id == TypeId::of::<T>()).then_some(value as &(dyn Any + Send + Sync))
}

impl<T> ProviderIdentity for &T
where
    T: ProviderIdentity + ?Sized,
{
    fn provider_name(&self) -> &str {
        T::provider_name(*self)
    }

    fn reads_namespace(&self, namespace: &str) -> bool {
        T::reads_namespace(*self, namespace)
    }

    fn downcast_any(&self, id: TypeId) -> Option<&(dyn Any + Send + Sync)> {
        T::downcast_any(*self, id)
    }
}

impl<T> ProviderIdentity for Box<T>
where
    T: ProviderIdentity + ?Sized,
{
    fn provider_name(&self) -> &str {
        T::provider_name(self.as_ref())
    }

    fn reads_namespace(&self, namespace: &str) -> bool {
        T::reads_namespace(self.as_ref(), namespace)
    }

    fn downcast_any(&self, id: TypeId) -> Option<&(dyn Any + Send + Sync)> {
        T::downcast_any(self.as_ref(), id)
    }
}

impl<T> ProviderIdentity for Arc<T>
where
    T: ProviderIdentity + ?Sized,
{
    fn provider_name(&self) -> &str {
        T::provider_name(self.as_ref())
    }

    fn reads_namespace(&self, namespace: &str) -> bool {
        T::reads_namespace(self.as_ref(), namespace)
    }

    fn downcast_any(&self, id: TypeId) -> Option<&(dyn Any + Send + Sync)> {
        T::downcast_any(self.as_ref(), id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Named;

    impl ProviderIdentity for Named {
        fn provider_name(&self) -> &str {
            "named"
        }

        fn reads_namespace(&self, namespace: &str) -> bool {
            namespace == "named-state"
        }
    }

    struct Exposed {
        value: u32,
    }

    impl ProviderIdentity for Exposed {
        fn provider_name(&self) -> &str {
            "exposed"
        }

        fn reads_namespace(&self, _namespace: &str) -> bool {
            false
        }

        fn downcast_any(&self, id: TypeId) -> Option<&(dyn Any + Send + Sync)> {
            expose(self, id).or_else(|| expose(&self.value, id))
        }
    }

    #[test]
    fn concrete_impl_returns_configured_name() {
        assert_eq!(Named.provider_name(), "named");
    }

    #[test]
    fn ref_forwarding_delegates() {
        let named = Named;
        let borrowed: &Named = &named;
        assert_eq!(borrowed.provider_name(), "named");
    }

    #[test]
    fn box_forwarding_delegates() {
        let boxed: Box<dyn ProviderIdentity> = Box::new(Named);
        assert_eq!(boxed.provider_name(), "named");
    }

    #[test]
    fn arc_forwarding_delegates() {
        let arced: Arc<dyn ProviderIdentity> = Arc::new(Named);
        assert_eq!(arced.provider_name(), "named");
    }

    #[test]
    fn pointers_forward_reads_namespace() {
        let named = Named;
        let borrowed: &Named = &named;
        let boxed: Box<dyn ProviderIdentity> = Box::new(Named);
        let arced: Arc<dyn ProviderIdentity> = Arc::new(Named);
        for provider in [borrowed as &dyn ProviderIdentity, &boxed, &arced] {
            assert!(provider.reads_namespace("named-state"));
            assert!(!provider.reads_namespace("named"));
        }
    }

    #[test]
    fn exposed_objects_are_found_through_pointers() {
        let exposed = Exposed { value: 7 };
        assert_eq!(exposed.downcast_ref::<u32>(), Some(&7));
        assert!(exposed.downcast_ref::<Exposed>().is_some());
        assert!(exposed.downcast_ref::<String>().is_none());
        let borrowed = &exposed;
        assert!(
            <&Exposed as ProviderIdentity>::downcast_any(&borrowed, TypeId::of::<u32>()).is_some()
        );

        let arced: Arc<dyn ProviderIdentity> = Arc::new(Exposed { value: 1 });
        assert_eq!(arced.downcast_ref::<u32>(), Some(&1));
        let boxed: Box<dyn ProviderIdentity> = Box::new(Exposed { value: 2 });
        assert_eq!(boxed.downcast_ref::<u32>(), Some(&2));

        assert!(Named.downcast_ref::<Named>().is_none());
    }
}
