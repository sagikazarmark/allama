//! Forwarding of the questions a transparent wrapper passes on.
//!
//! The macros are exported at the crate root under hidden names and
//! re-exported from [`wrappers`](crate::wrappers) under their public ones,
//! so they are documented in one place.

/// Forward [`ProviderIdentity`](crate::ProviderIdentity) to the provider in
/// a field: the name, the namespaces it reads, and
/// [`downcast_any`](crate::ProviderIdentity::downcast_any).
///
/// Write it as the whole body of the `impl`, naming the field that holds the
/// wrapped provider (`inner`, or a path such as `inner.0`). A method added
/// to the trait later reaches the wrapper when it is recompiled against the
/// new allama, instead of falling back to a default that is wrong for a
/// wrapper.
///
/// ```
/// use allama::ProviderIdentity;
/// use allama::wrappers::forward_provider_identity;
///
/// struct Audited<P> {
///     inner: P,
/// }
///
/// impl<P: ProviderIdentity> ProviderIdentity for Audited<P> {
///     forward_provider_identity!(inner);
/// }
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! __forward_provider_identity {
    ($($field:tt).+) => {
        fn provider_name(&self) -> &str {
            $crate::ProviderIdentity::provider_name(&self.$($field).+)
        }

        fn reads_namespace(&self, namespace: &str) -> bool {
            $crate::ProviderIdentity::reads_namespace(&self.$($field).+, namespace)
        }

        fn downcast_any(
            &self,
            id: ::std::any::TypeId,
        ) -> ::std::option::Option<&(dyn ::std::any::Any + Send + Sync)> {
            $crate::ProviderIdentity::downcast_any(&self.$($field).+, id)
        }
    };
}

/// Forward every question about a chat provider to the provider in a field:
/// [`chat_capability`](crate::ChatProvider::chat_capability),
/// [`validate_chat_request`](crate::ChatProvider::validate_chat_request) and
/// [`validate_chat_stream_request`](crate::ChatProvider::validate_chat_stream_request).
///
/// A wrapper writes the operations (`type Stream`, `chat`, `chat_stream`)
/// and this macro for the rest. A question added to
/// [`ChatProvider`](crate::ChatProvider) later (a capability, a limit, a
/// validator) is added here in the same release, so a wrapper that uses the
/// macro forwards it by being recompiled. A wrapper that answers one of
/// these questions differently (it maps the model, or overrides a
/// capability) writes all of them by hand instead.
///
/// ```
/// use allama::wrappers::{forward_chat_questions, forward_provider_identity};
/// use allama::{ChatProvider, ChatRequest, ChatResponse, ProviderIdentity, Result};
///
/// /// Counts calls.
/// struct Counted<P> {
///     inner: P,
///     calls: std::sync::atomic::AtomicUsize,
/// }
///
/// impl<P: ChatProvider> ChatProvider for Counted<P> {
///     type Stream = P::Stream;
///
///     async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
///         self.calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
///         self.inner.chat(request).await
///     }
///
///     async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
///         self.calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
///         self.inner.chat_stream(request).await
///     }
///
///     forward_chat_questions!(inner);
/// }
///
/// impl<P: ProviderIdentity> ProviderIdentity for Counted<P> {
///     forward_provider_identity!(inner);
/// }
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! __forward_chat_questions {
    ($($field:tt).+) => {
        fn chat_capability(
            &self,
            model: &str,
            capability: $crate::ChatCapability,
        ) -> $crate::CapabilitySupport {
            $crate::ChatProvider::chat_capability(&self.$($field).+, model, capability)
        }

        fn validate_chat_request(
            &self,
            request: &$crate::ChatRequest,
        ) -> $crate::Result<$crate::ChatValidation> {
            $crate::ChatProvider::validate_chat_request(&self.$($field).+, request)
        }

        fn validate_chat_stream_request(
            &self,
            request: &$crate::ChatRequest,
        ) -> $crate::Result<$crate::ChatValidation> {
            $crate::ChatProvider::validate_chat_stream_request(&self.$($field).+, request)
        }
    };
}

/// Forward every question about an embedding provider to the provider in a
/// field: [`embedding_capability`](crate::EmbeddingProvider::embedding_capability),
/// [`embedding_limits`](crate::EmbeddingProvider::embedding_limits) and
/// [`validate_embedding_request`](crate::EmbeddingProvider::validate_embedding_request).
///
/// The embedding counterpart of
/// [`forward_chat_questions!`](crate::wrappers::forward_chat_questions): a
/// wrapper writes `embed` and this macro.
///
/// ```
/// use allama::wrappers::{forward_embedding_questions, forward_provider_identity};
/// use allama::{EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, ProviderIdentity, Result};
///
/// struct Logged<P> {
///     inner: P,
/// }
///
/// impl<P: EmbeddingProvider> EmbeddingProvider for Logged<P> {
///     async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
///         eprintln!("embedding {} inputs", request.inputs.len());
///         self.inner.embed(request).await
///     }
///
///     forward_embedding_questions!(inner);
/// }
///
/// impl<P: ProviderIdentity> ProviderIdentity for Logged<P> {
///     forward_provider_identity!(inner);
/// }
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! __forward_embedding_questions {
    ($($field:tt).+) => {
        fn embedding_capability(
            &self,
            model: &str,
            capability: $crate::EmbeddingCapability,
        ) -> $crate::CapabilitySupport {
            $crate::EmbeddingProvider::embedding_capability(&self.$($field).+, model, capability)
        }

        fn embedding_limits(&self, model: &str) -> $crate::EmbeddingLimits {
            $crate::EmbeddingProvider::embedding_limits(&self.$($field).+, model)
        }

        fn validate_embedding_request(
            &self,
            request: &$crate::EmbeddingRequest,
        ) -> $crate::Result<$crate::EmbeddingValidation> {
            $crate::EmbeddingProvider::validate_embedding_request(&self.$($field).+, request)
        }
    };
}

/// Forward every question about a decision provider to the provider in a
/// field: [`decision_capability`](crate::DecisionProvider::decision_capability),
/// [`decision_limits`](crate::DecisionProvider::decision_limits) and
/// [`validate_decision_request`](crate::DecisionProvider::validate_decision_request).
///
/// The decision counterpart of
/// [`forward_chat_questions!`](crate::wrappers::forward_chat_questions): a
/// wrapper writes `decide` and this macro.
///
/// ```
/// use allama::wrappers::{forward_decision_questions, forward_provider_identity};
/// use allama::{DecisionProvider, DecisionRequest, DecisionResponse, ProviderIdentity, Result};
///
/// struct Logged<P> {
///     inner: P,
/// }
///
/// impl<P: DecisionProvider> DecisionProvider for Logged<P> {
///     async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse> {
///         eprintln!("deciding {} questions", request.questions.len());
///         self.inner.decide(request).await
///     }
///
///     forward_decision_questions!(inner);
/// }
///
/// impl<P: ProviderIdentity> ProviderIdentity for Logged<P> {
///     forward_provider_identity!(inner);
/// }
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! __forward_decision_questions {
    ($($field:tt).+) => {
        fn decision_capability(
            &self,
            model: &str,
            capability: $crate::DecisionCapability,
        ) -> $crate::CapabilitySupport {
            $crate::DecisionProvider::decision_capability(&self.$($field).+, model, capability)
        }

        fn decision_limits(&self, model: &str) -> $crate::DecisionLimits {
            $crate::DecisionProvider::decision_limits(&self.$($field).+, model)
        }

        fn validate_decision_request(
            &self,
            request: &$crate::DecisionRequest,
        ) -> $crate::Result<$crate::DecisionValidation> {
            $crate::DecisionProvider::validate_decision_request(&self.$($field).+, request)
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::mock::{MockEmbeddingProvider, MockProvider};
    use crate::{
        CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse,
        EmbeddingLimits, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, ProviderIdentity,
        ProviderIdentityExt, Result,
    };

    struct Tuple<P>(P);

    impl<P: ChatProvider> ChatProvider for Tuple<P> {
        type Stream = P::Stream;

        async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
            self.0.chat(request).await
        }

        async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
            self.0.chat_stream(request).await
        }

        crate::wrappers::forward_chat_questions!(0);
    }

    impl<P: EmbeddingProvider> EmbeddingProvider for Tuple<P> {
        async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
            self.0.embed(request).await
        }

        crate::wrappers::forward_embedding_questions!(0);
    }

    impl<P: ProviderIdentity> ProviderIdentity for Tuple<P> {
        crate::wrappers::forward_provider_identity!(0);
    }

    struct Nested<P> {
        layer: Tuple<P>,
    }

    impl<P: ProviderIdentity> ProviderIdentity for Nested<P> {
        crate::wrappers::forward_provider_identity!(layer.0);
    }

    #[test]
    fn forwards_through_fields_and_paths() {
        let chat = Tuple(
            MockProvider::empty()
                .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported),
        );
        assert_eq!(
            chat.chat_capability("m", ChatCapability::ToolCalls),
            CapabilitySupport::Supported
        );
        assert_eq!(chat.provider_name(), "mock");
        assert!(chat.downcast_ref::<MockProvider>().is_some());

        let embedding = Tuple(MockEmbeddingProvider::empty().with_max_inputs_per_call(7));
        assert_eq!(
            embedding.embedding_limits("m"),
            EmbeddingLimits::new().max_inputs_per_call(7)
        );

        let nested = Nested {
            layer: Tuple(MockProvider::empty()),
        };
        assert!(nested.reads_namespace("mock"));
        assert!(nested.downcast_ref::<MockProvider>().is_some());
    }
}
