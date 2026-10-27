use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::{ExtractionLimits, ExtractionRequest, ExtractionResponse, FieldKind};
use crate::macros::enum_names;
use crate::{
    CapabilitySupport, ExtractionValidation, ProviderIdentity, Result, ValidationCoverage,
};

/// Providers of extraction models: models that answer typed fields about
/// an input with values instead of generating text.
///
/// See [`ExtractionRequest`] for the fields and [`ExtractionResponse`] for
/// the answers. Implementations check a request with
/// [`extraction::prepare`](crate::adapter::extraction::prepare) before
/// sending anything, and call [`ExtractionResponse::validate`] on what they
/// decode.
///
/// # Stability
///
/// Unstable (ADR 0018): grounded in one provider, this trait and the types
/// it takes may change in any minor release until a second provider
/// implements it.
pub trait ExtractionProvider: ProviderIdentity {
    /// Answer every field of `request`.
    fn extract_values(
        &self,
        request: &ExtractionRequest,
    ) -> impl Future<Output = Result<ExtractionResponse>> + Send;

    /// Whether the provider supports `capability` for `model`.
    ///
    /// Required: validation acts on the answer. A provider that does not know
    /// answers [`CapabilitySupport::Unknown`]; a wrapper forwards the answer
    /// of the provider it wraps.
    fn extraction_capability(
        &self,
        model: &str,
        capability: ExtractionCapability,
    ) -> CapabilitySupport;

    /// The request limits of `model`. The default reports none. Wrappers
    /// forward it.
    fn extraction_limits(&self, _model: &str) -> ExtractionLimits {
        ExtractionLimits::default()
    }

    /// Validate a request without sending it: a dry run of
    /// [`extract_values`](Self::extract_values).
    ///
    /// The default runs
    /// [`extraction::prepare_dry_run`](crate::adapter::extraction::prepare_dry_run)
    /// and reports [`ValidationCoverage::Capabilities`]: the request's
    /// structure and the model's limits ([`ExtractionRequest::validate`]),
    /// and each field kind against
    /// [`extraction_capability`](Self::extraction_capability). A kind
    /// answered `Unknown` is allowed through and reported in
    /// [`relied_on_unknown`](crate::Validation::relied_on_unknown).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`](crate::Error::InvalidRequest) for
    /// structural problems or exceeded limits, and
    /// [`Error::Unsupported`](crate::Error::Unsupported) for a field kind
    /// the model explicitly does not support.
    fn validate_extraction_request(
        &self,
        request: &ExtractionRequest,
    ) -> Result<ExtractionValidation> {
        let prepared = crate::adapter::extraction::prepare_dry_run(self, request)?;
        Ok(prepared.into_validation(ValidationCoverage::Capabilities))
    }
}

macro_rules! forward_extraction_provider {
    ($ty:ty, |$s:ident| $inner:expr) => {
        impl<T> ExtractionProvider for $ty
        where
            T: ExtractionProvider + ?Sized,
        {
            async fn extract_values(
                &self,
                request: &ExtractionRequest,
            ) -> Result<ExtractionResponse> {
                let $s = self;
                let inner: &T = $inner;
                inner.extract_values(request).await
            }

            fn extraction_capability(
                &self,
                model: &str,
                capability: ExtractionCapability,
            ) -> CapabilitySupport {
                let $s = self;
                let inner: &T = $inner;
                inner.extraction_capability(model, capability)
            }

            fn extraction_limits(&self, model: &str) -> ExtractionLimits {
                let $s = self;
                let inner: &T = $inner;
                inner.extraction_limits(model)
            }

            fn validate_extraction_request(
                &self,
                request: &ExtractionRequest,
            ) -> Result<ExtractionValidation> {
                let $s = self;
                let inner: &T = $inner;
                inner.validate_extraction_request(request)
            }
        }
    };
}

forward_extraction_provider!(&T, |s| *s);
forward_extraction_provider!(Box<T>, |s| s.as_ref());
forward_extraction_provider!(Arc<T>, |s| s.as_ref());

/// Portable extraction features a provider or model may support: one per
/// [`FieldKind`], and images read together with the input.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionCapability {
    /// Text fields.
    Text,
    /// Number fields.
    Number,
    /// Date fields.
    Date,
    /// Time fields.
    Time,
    /// Datetime fields.
    #[serde(rename = "datetime")]
    DateTime,
    /// Duration fields.
    Duration,
    /// List fields.
    List,
    /// Estimate fields.
    Estimate,
    /// Forecast fields.
    Forecast,
    /// Images read together with the input
    /// ([`ExtractionRequest::images`](crate::ExtractionRequest::images)).
    ImageInput,
}

enum_names!(ExtractionCapability, "extraction capability", {
    Text => "text",
    Number => "number",
    Date => "date",
    Time => "time",
    DateTime => "datetime",
    Duration => "duration",
    List => "list",
    Estimate => "estimate",
    Forecast => "forecast",
    ImageInput => "image_input",
});

impl ExtractionCapability {
    /// Every capability, in declaration order.
    pub const ALL: &[Self] = &[
        Self::Text,
        Self::Number,
        Self::Date,
        Self::Time,
        Self::DateTime,
        Self::Duration,
        Self::List,
        Self::Estimate,
        Self::Forecast,
        Self::ImageInput,
    ];
}

impl From<FieldKind> for ExtractionCapability {
    fn from(kind: FieldKind) -> Self {
        match kind {
            FieldKind::Text => Self::Text,
            FieldKind::Number => Self::Number,
            FieldKind::Date => Self::Date,
            FieldKind::Time => Self::Time,
            FieldKind::DateTime => Self::DateTime,
            FieldKind::Duration => Self::Duration,
            FieldKind::List => Self::List,
            FieldKind::Estimate => Self::Estimate,
            FieldKind::Forecast => Self::Forecast,
        }
    }
}

/// A type-erased extraction provider.
#[derive(Clone)]
pub struct DynExtractionProvider(Arc<dyn ExtractionProviderErased>);

impl DynExtractionProvider {
    /// Erase a provider.
    #[must_use]
    pub fn new<T: ExtractionProvider + 'static>(provider: T) -> Self {
        Self(Arc::new(provider))
    }
}

impl<T: ExtractionProvider + 'static> From<Arc<T>> for DynExtractionProvider {
    fn from(provider: Arc<T>) -> Self {
        Self(provider)
    }
}

impl std::fmt::Debug for DynExtractionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynExtractionProvider")
            .field("provider", &self.0.provider_name())
            .finish()
    }
}

impl ProviderIdentity for DynExtractionProvider {
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

impl ExtractionProvider for DynExtractionProvider {
    async fn extract_values(&self, request: &ExtractionRequest) -> Result<ExtractionResponse> {
        self.0.extract_values_erased(request).await
    }

    fn extraction_capability(
        &self,
        model: &str,
        capability: ExtractionCapability,
    ) -> CapabilitySupport {
        self.0.extraction_capability_erased(model, capability)
    }

    fn extraction_limits(&self, model: &str) -> ExtractionLimits {
        self.0.extraction_limits_erased(model)
    }

    fn validate_extraction_request(
        &self,
        request: &ExtractionRequest,
    ) -> Result<ExtractionValidation> {
        self.0.validate_erased(request)
    }
}

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

trait ExtractionProviderErased: ProviderIdentity {
    fn extract_values_erased<'a>(
        &'a self,
        request: &'a ExtractionRequest,
    ) -> BoxFuture<'a, Result<ExtractionResponse>>;
    fn extraction_capability_erased(
        &self,
        model: &str,
        capability: ExtractionCapability,
    ) -> CapabilitySupport;
    fn extraction_limits_erased(&self, model: &str) -> ExtractionLimits;
    fn validate_erased(&self, request: &ExtractionRequest) -> Result<ExtractionValidation>;
}

impl<T: ExtractionProvider> ExtractionProviderErased for T {
    fn extract_values_erased<'a>(
        &'a self,
        request: &'a ExtractionRequest,
    ) -> BoxFuture<'a, Result<ExtractionResponse>> {
        Box::pin(ExtractionProvider::extract_values(self, request))
    }

    fn extraction_capability_erased(
        &self,
        model: &str,
        capability: ExtractionCapability,
    ) -> CapabilitySupport {
        ExtractionProvider::extraction_capability(self, model, capability)
    }

    fn extraction_limits_erased(&self, model: &str) -> ExtractionLimits {
        ExtractionProvider::extraction_limits(self, model)
    }

    fn validate_erased(&self, request: &ExtractionRequest) -> Result<ExtractionValidation> {
        ExtractionProvider::validate_extraction_request(self, request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use crate::extraction::{Answer, Field, NumberAnswer};
    use crate::mock::MockExtractionProvider;

    fn request() -> ExtractionRequest {
        ExtractionRequest::new("m", "x").field("n", Field::number())
    }

    fn answer() -> Result<ExtractionResponse> {
        Ok(ExtractionResponse::new([(
            "n",
            Answer::from(NumberAnswer::new(Some(1.0))),
        )]))
    }

    #[tokio::test]
    async fn dyn_provider_forwards() {
        let provider = DynExtractionProvider::new(
            MockExtractionProvider::new([answer()])
                .with_extraction_limits(ExtractionLimits::new().max_fields(4)),
        );
        let response = provider.extract_values(&request()).await.unwrap();
        assert_eq!(response.number("n").unwrap().value, Some(1.0));
        assert_eq!(provider.extraction_limits("m").max_fields, Some(4));
        assert_eq!(provider.provider_name(), "mock");
        assert_eq!(
            ExtractionCapability::from(FieldKind::DateTime).to_string(),
            "datetime"
        );
        assert_eq!(ExtractionCapability::ALL.len(), FieldKind::ALL.len() + 1);
    }

    #[tokio::test]
    async fn wrappers_retry_map_and_fall_back() {
        use crate::wrappers::{
            FallbackAttempts, FallbackProvider, MapModelProvider, RetryPolicy, RetryingProvider,
            TimeoutProvider,
        };

        let flaky = MockExtractionProvider::new([Err(Error::overloaded("busy")), answer()]);
        let retrying = RetryingProvider::new(
            flaky.clone(),
            RetryPolicy::new().base_delay(std::time::Duration::ZERO),
        );
        retrying.extract_values(&request()).await.unwrap();
        assert_eq!(flaky.requests().len(), 2);

        let mock = MockExtractionProvider::new([answer()])
            .with_extraction_limits(ExtractionLimits::new().max_fields(3));
        let mapped = MapModelProvider::new(mock.clone(), |_: &str| "real".into());
        let response = mapped.extract_values(&request()).await.unwrap();
        assert_eq!(response.number("n").unwrap().value, Some(1.0));
        assert_eq!(mock.requests()[0].model, "real");
        assert_eq!(mapped.extraction_limits("alias").max_fields, Some(3));

        let fallback = FallbackProvider::new(
            MockExtractionProvider::with_error(Error::overloaded("busy"))
                .with_provider_name("primary"),
            MockExtractionProvider::new([answer()]),
        );
        let response = TimeoutProvider::new(fallback, Default::default())
            .extract_values(&request())
            .await
            .unwrap();
        let attempts = response
            .provider_data
            .get::<FallbackAttempts>()
            .unwrap()
            .unwrap();
        assert_eq!(attempts.as_slice()[0].provider, "primary");

        // A malformed answer does not fail over: it is this provider's bug.
        let backup = MockExtractionProvider::new([answer()]);
        let fallback = FallbackProvider::new(
            MockExtractionProvider::with_error(Error::invalid_request("bad field")),
            backup.clone(),
        );
        let err = fallback.extract_values(&request()).await.unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
        assert!(backup.requests().is_empty());
    }

    #[test]
    fn validation_rejects_unsupported_kinds_and_limits() {
        let provider = MockExtractionProvider::empty()
            .with_extraction_capability(
                ExtractionCapability::Number,
                CapabilitySupport::Unsupported,
            )
            .with_extraction_limits(ExtractionLimits::new().max_fields(1));
        assert!(matches!(
            provider.validate_extraction_request(&request()),
            Err(Error::Unsupported { .. })
        ));
        let two = ExtractionRequest::new("m", "x")
            .field("a", Field::text())
            .field("b", Field::text());
        assert!(matches!(
            provider.validate_extraction_request(&two),
            Err(Error::InvalidRequest { .. })
        ));
        let one = ExtractionRequest::new("m", "x").field("a", Field::text());
        let validation = provider.validate_extraction_request(&one).unwrap();
        assert_eq!(validation.relied_on_unknown().len(), 1);
    }
}
