use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::typed::{Decided, Decision};
use super::{DecisionInput, DecisionLimits, DecisionRequest, DecisionResponse, QuestionKind};
use crate::macros::enum_names;
use crate::{CapabilitySupport, DecisionValidation, ProviderIdentity, Result, ValidationCoverage};

/// Portable decision features a provider or model may support.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionCapability {
    /// Yes/no questions.
    YesNo,
    /// Choice questions.
    Choice,
    /// Score questions.
    Score,
    /// Images read together with the input
    /// ([`DecisionRequest::images`](crate::DecisionRequest::images)).
    ImageInput,
}

enum_names!(DecisionCapability, "decision capability", {
    YesNo => "yes_no",
    Choice => "choice",
    Score => "score",
    ImageInput => "image_input",
});

impl DecisionCapability {
    /// Every capability, in declaration order.
    pub const ALL: &[Self] = &[Self::YesNo, Self::Choice, Self::Score, Self::ImageInput];
}

impl From<QuestionKind> for DecisionCapability {
    fn from(kind: QuestionKind) -> Self {
        match kind {
            QuestionKind::YesNo => Self::YesNo,
            QuestionKind::Choice => Self::Choice,
            QuestionKind::Score => Self::Score,
        }
    }
}

/// Providers of decision models: models that answer typed questions about an
/// input with probabilities instead of generating text.
///
/// See [`DecisionRequest`] for the questions and [`DecisionResponse`] for the
/// answers. Implementations check a request with
/// [`decision::prepare`](crate::adapter::decision::prepare) before sending anything,
/// and call [`DecisionResponse::validate`] on what they decode.
pub trait DecisionProvider: ProviderIdentity {
    /// Answer every question of `request`.
    fn decide(
        &self,
        request: &DecisionRequest,
    ) -> impl Future<Output = Result<DecisionResponse>> + Send;

    /// Whether the provider supports `capability` for `model`.
    ///
    /// Required: validation acts on the answer. A provider that does not know
    /// answers [`CapabilitySupport::Unknown`]; a wrapper forwards the answer
    /// of the provider it wraps.
    fn decision_capability(&self, model: &str, capability: DecisionCapability)
    -> CapabilitySupport;

    /// The request limits of `model`. The default reports none. Wrappers
    /// forward it.
    fn decision_limits(&self, _model: &str) -> DecisionLimits {
        DecisionLimits::default()
    }

    /// Validate a request without sending it: a dry run of
    /// [`decide`](Self::decide).
    ///
    /// The default runs [`decision::prepare_dry_run`](crate::adapter::decision::prepare_dry_run)
    /// and reports [`ValidationCoverage::Capabilities`]: the request's
    /// structure and the model's limits ([`DecisionRequest::validate`]), and
    /// each question kind against
    /// [`decision_capability`](Self::decision_capability). A kind answered
    /// `Unknown` is allowed through and reported in
    /// [`relied_on_unknown`](crate::Validation::relied_on_unknown).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`](crate::Error::InvalidRequest) for
    /// structural problems or exceeded limits, and
    /// [`Error::Unsupported`](crate::Error::Unsupported) for a question kind the model
    /// explicitly does not support.
    fn validate_decision_request(&self, request: &DecisionRequest) -> Result<DecisionValidation> {
        let prepared = crate::adapter::decision::prepare_dry_run(self, request)?;
        Ok(prepared.into_validation(ValidationCoverage::Capabilities))
    }
}

macro_rules! forward_decision_provider {
    ($ty:ty, |$s:ident| $inner:expr) => {
        impl<T> DecisionProvider for $ty
        where
            T: DecisionProvider + ?Sized,
        {
            async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse> {
                let $s = self;
                let inner: &T = $inner;
                inner.decide(request).await
            }

            fn decision_capability(
                &self,
                model: &str,
                capability: DecisionCapability,
            ) -> CapabilitySupport {
                let $s = self;
                let inner: &T = $inner;
                inner.decision_capability(model, capability)
            }

            fn decision_limits(&self, model: &str) -> DecisionLimits {
                let $s = self;
                let inner: &T = $inner;
                inner.decision_limits(model)
            }

            fn validate_decision_request(
                &self,
                request: &DecisionRequest,
            ) -> Result<DecisionValidation> {
                let $s = self;
                let inner: &T = $inner;
                inner.validate_decision_request(request)
            }
        }
    };
}

forward_decision_provider!(&T, |s| *s);
forward_decision_provider!(Box<T>, |s| s.as_ref());
forward_decision_provider!(Arc<T>, |s| s.as_ref());

/// A type-erased decision provider.
#[derive(Clone)]
pub struct DynDecisionProvider(Arc<dyn DecisionProviderErased>);

impl DynDecisionProvider {
    /// Erase a provider.
    #[must_use]
    pub fn new<T: DecisionProvider + 'static>(provider: T) -> Self {
        Self(Arc::new(provider))
    }
}

impl<T: DecisionProvider + 'static> From<Arc<T>> for DynDecisionProvider {
    fn from(provider: Arc<T>) -> Self {
        Self(provider)
    }
}

impl std::fmt::Debug for DynDecisionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynDecisionProvider")
            .field("provider", &self.0.provider_name())
            .finish()
    }
}

impl ProviderIdentity for DynDecisionProvider {
    fn provider_name(&self) -> &str {
        self.0.provider_name()
    }

    fn reads_namespace(&self, namespace: &str) -> bool {
        self.0.reads_namespace(namespace)
    }

    fn downcast_any(&self, id: std::any::TypeId) -> Option<&(dyn std::any::Any + Send + Sync)> {
        self.0.downcast_any(id)
    }
}

impl DecisionProvider for DynDecisionProvider {
    async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse> {
        self.0.decide_erased(request).await
    }

    fn decision_capability(
        &self,
        model: &str,
        capability: DecisionCapability,
    ) -> CapabilitySupport {
        self.0.decision_capability_erased(model, capability)
    }

    fn decision_limits(&self, model: &str) -> DecisionLimits {
        self.0.decision_limits_erased(model)
    }

    fn validate_decision_request(&self, request: &DecisionRequest) -> Result<DecisionValidation> {
        self.0.validate_erased(request)
    }
}

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

trait DecisionProviderErased: ProviderIdentity {
    fn decide_erased<'a>(
        &'a self,
        request: &'a DecisionRequest,
    ) -> BoxFuture<'a, Result<DecisionResponse>>;
    fn decision_capability_erased(
        &self,
        model: &str,
        capability: DecisionCapability,
    ) -> CapabilitySupport;
    fn decision_limits_erased(&self, model: &str) -> DecisionLimits;
    fn validate_erased(&self, request: &DecisionRequest) -> Result<DecisionValidation>;
}

impl<T: DecisionProvider> DecisionProviderErased for T {
    fn decide_erased<'a>(
        &'a self,
        request: &'a DecisionRequest,
    ) -> BoxFuture<'a, Result<DecisionResponse>> {
        Box::pin(DecisionProvider::decide(self, request))
    }

    fn decision_capability_erased(
        &self,
        model: &str,
        capability: DecisionCapability,
    ) -> CapabilitySupport {
        DecisionProvider::decision_capability(self, model, capability)
    }

    fn decision_limits_erased(&self, model: &str) -> DecisionLimits {
        DecisionProvider::decision_limits(self, model)
    }

    fn validate_erased(&self, request: &DecisionRequest) -> Result<DecisionValidation> {
        DecisionProvider::validate_decision_request(self, request)
    }
}

/// Convenience methods for every [`DecisionProvider`].
pub trait DecisionProviderExt: DecisionProvider {
    /// Ask the questions of a typed [`Decision`] about `input`, check the
    /// response against them, and decode the answers.
    ///
    /// The returned [`Decided`] keeps the response for usage, the request
    /// id, and provider data.
    ///
    /// ```
    /// # #[cfg(feature = "mock")] {
    /// use allama::decision::{Answer, Decision, DecisionField, Question, YesNoAnswer, answer_as};
    /// use allama::mock::MockDecisionProvider;
    /// use allama::{DecisionProviderExt, DecisionResponse};
    ///
    /// // `#[derive(Decision)]` (feature `derive`) writes this impl.
    /// struct Triage {
    ///     spam: YesNoAnswer,
    /// }
    ///
    /// impl Decision for Triage {
    ///     fn questions() -> Vec<(String, Question)> {
    ///         vec![("spam".into(), YesNoAnswer::question("Is this spam?"))]
    ///     }
    ///
    ///     fn from_response(response: &DecisionResponse) -> allama::Result<Self> {
    ///         Ok(Self { spam: answer_as(response, "spam")? })
    ///     }
    /// }
    ///
    /// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
    /// let provider = MockDecisionProvider::with_response(
    ///     DecisionResponse::new([("spam", Answer::from(YesNoAnswer::new(0.1)))]).model("nimble"),
    /// );
    /// let decided = provider.decide_as::<Triage>("nimble", "Hello!").await?;
    /// assert!(!decided.value.spam.is_yes());
    /// assert_eq!(decided.response.model.as_deref(), Some("nimble"));
    /// # Ok::<(), allama::Error>(()) }).unwrap();
    /// # }
    /// ```
    fn decide_as<T: Decision>(
        &self,
        model: &str,
        input: impl Into<DecisionInput>,
    ) -> impl Future<Output = Result<Decided<T>>> + Send {
        let request = T::request(model, input);
        async move {
            let response = self.decide(&request).await?;
            response.validate(&request)?;
            Ok(Decided {
                value: T::from_response(&response)?,
                response,
            })
        }
    }
}

impl<T: DecisionProvider + ?Sized> DecisionProviderExt for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use crate::decision::{Answer, Question, YesNoAnswer};
    use crate::mock::MockDecisionProvider;

    #[tokio::test]
    async fn dyn_provider_forwards() {
        let provider = DynDecisionProvider::new(
            MockDecisionProvider::with_response(DecisionResponse::new([(
                "spam",
                Answer::from(YesNoAnswer::new(0.9)),
            )]))
            .with_decision_limits(DecisionLimits::new().max_questions(4)),
        );
        let request = DecisionRequest::new("m", "x").question("spam", Question::yes_no("?"));
        let response = provider.decide(&request).await.unwrap();
        assert!(response.yes_no("spam").unwrap().is_yes());
        assert_eq!(provider.decision_limits("m").max_questions, Some(4));
        assert_eq!(provider.provider_name(), "mock");
    }

    #[tokio::test]
    async fn wrappers_retry_and_fall_back() {
        use crate::wrappers::{FallbackAttempts, FallbackProvider, RetryPolicy, RetryingProvider};

        let answer = || {
            Ok(DecisionResponse::new([(
                "spam",
                Answer::from(YesNoAnswer::new(0.1)),
            )]))
        };
        let request = DecisionRequest::new("m", "x").question("spam", Question::yes_no("?"));

        let flaky = MockDecisionProvider::new([Err(Error::overloaded("busy")), answer()]);
        let retrying = RetryingProvider::new(
            flaky.clone(),
            RetryPolicy::new().base_delay(std::time::Duration::ZERO),
        );
        retrying.decide(&request).await.unwrap();
        assert_eq!(flaky.requests().len(), 2);

        let fallback = FallbackProvider::new(
            MockDecisionProvider::with_error(Error::overloaded("busy"))
                .with_provider_name("primary"),
            MockDecisionProvider::new([answer()]),
        );
        let response = fallback.decide(&request).await.unwrap();
        let attempts = response
            .provider_data
            .get::<FallbackAttempts>()
            .unwrap()
            .unwrap();
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts.as_slice()[0].provider, "primary");
        assert_eq!(attempts.as_slice()[0].error_kind, "overloaded");

        // Errors a retry or another provider cannot fix do not fail over.
        let backup = MockDecisionProvider::new([answer()]);
        let fallback = FallbackProvider::new(
            MockDecisionProvider::with_error(Error::invalid_request("bad question")),
            backup.clone(),
        );
        let err = fallback.decide(&request).await.unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
        assert!(backup.requests().is_empty());
    }

    #[test]
    fn validation_rejects_unsupported_kinds_and_limits() {
        let provider = MockDecisionProvider::empty()
            .with_decision_capability(DecisionCapability::Score, CapabilitySupport::Unsupported)
            .with_decision_limits(DecisionLimits::new().max_choice_options(2));
        let score =
            DecisionRequest::new("m", "x").question("q", Question::score("?").levels(["a", "b"]));
        assert!(matches!(
            provider.validate_decision_request(&score),
            Err(Error::Unsupported { .. })
        ));
        let choice = DecisionRequest::new("m", "x").question(
            "q",
            Question::choice("?")
                .bare_option("a")
                .bare_option("b")
                .bare_option("c"),
        );
        assert!(matches!(
            provider.validate_decision_request(&choice),
            Err(Error::InvalidRequest { .. })
        ));
    }
}
