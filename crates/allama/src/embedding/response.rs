use std::borrow::Cow;

use serde::{Deserialize, Serialize};

use crate::{ProviderData, Subject, Usage, Warning, WarningKind};

/// A batch embedding response.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EmbeddingResponse {
    /// One vector per input, in input order.
    pub embeddings: Vec<Vec<f32>>,
    /// Model that produced the vectors, as reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Token usage, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Identity of the provider that actually served the request.
    /// [`embedding::Prepared::finish`](crate::adapter::embedding::Prepared::finish)
    /// sets it to the adapter's name when the response has none, and
    /// wrappers such as fallback leave the serving provider's value in
    /// place.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<Cow<'static, str>>,
    /// Non-fatal warnings.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<Warning>,
    /// Provider-specific response data.
    #[serde(default, skip_serializing_if = "ProviderData::is_empty")]
    pub provider_data: ProviderData,
}

impl EmbeddingResponse {
    /// Create a response.
    #[must_use]
    pub fn new(embeddings: Vec<Vec<f32>>) -> Self {
        Self {
            embeddings,
            ..Self::default()
        }
    }

    /// Set the model.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set usage.
    #[must_use]
    pub fn usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Set the serving provider.
    #[must_use]
    pub fn provider(mut self, provider: impl Into<Cow<'static, str>>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Append a warning.
    #[must_use]
    pub fn warning(mut self, warning: Warning) -> Self {
        self.warnings.push(warning);
        self
    }

    /// Set provider data.
    #[must_use]
    pub fn provider_data(mut self, data: ProviderData) -> Self {
        self.provider_data = data;
        self
    }

    /// Append another response's vectors, usage, and warnings (for chunked
    /// requests). A warning this response already holds is not repeated, so
    /// a warning every chunk reports (an unsupported input type, say) is
    /// reported once.
    ///
    /// The serving [`provider`](Self::provider) stays this response's, the
    /// first chunk's. When `other` was served by another provider (a
    /// fallback took over partway through), the response gets a
    /// [`WarningKind::Fallback`] warning naming both, since vectors from
    /// different providers' models are rarely comparable.
    pub fn extend(&mut self, other: EmbeddingResponse) {
        self.embeddings.extend(other.embeddings);
        if self.model.is_none() {
            self.model = other.model;
        }
        let mixed = match (&self.provider, &other.provider) {
            (Some(first), Some(other)) if first != other => Some(Warning::new(
                WarningKind::Fallback,
                Subject::Response,
                format!(
                    "vectors from different providers may not be comparable; the chunks were \
                     served by `{first}` and `{other}`"
                ),
            )),
            _ => None,
        };
        if self.provider.is_none() {
            self.provider = other.provider;
        }
        self.usage = match (self.usage.take(), other.usage) {
            (Some(a), Some(b)) => Some(a + b),
            (a, b) => a.or(b),
        };
        for warning in other.warnings {
            if !self.warnings.contains(&warning) {
                self.warnings.push(warning);
            }
        }
        if let Some(warning) = mixed
            && !self.warnings.contains(&warning)
        {
            self.warnings.push(warning);
        }
        self.provider_data.merge(other.provider_data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RequestField;

    #[test]
    fn extend_reports_a_repeated_warning_once() {
        let hint = || Warning::unsupported(RequestField::InputType, "no input type");
        let other = Warning::unsupported(RequestField::Dimensions, "other");
        let mut combined = EmbeddingResponse::new(vec![vec![1.0]]).warning(hint());
        combined.extend(
            EmbeddingResponse::new(vec![vec![2.0]])
                .warning(hint())
                .warning(other.clone()),
        );
        combined.extend(EmbeddingResponse::new(vec![vec![3.0]]).warning(hint()));
        assert_eq!(combined.embeddings, vec![vec![1.0], vec![2.0], vec![3.0]]);
        assert_eq!(combined.warnings, vec![hint(), other]);
    }

    #[test]
    fn extend_keeps_the_first_provider_and_warns_when_another_served() {
        let mut combined = EmbeddingResponse::new(vec![vec![1.0]]).provider("primary");
        combined.extend(EmbeddingResponse::new(vec![vec![2.0]]).provider("primary"));
        assert!(combined.warnings.is_empty(), "{:?}", combined.warnings);

        combined.extend(EmbeddingResponse::new(vec![vec![3.0]]).provider("backup"));
        combined.extend(EmbeddingResponse::new(vec![vec![4.0]]).provider("backup"));
        assert_eq!(combined.provider.as_deref(), Some("primary"));
        assert_eq!(combined.warnings.len(), 1, "{:?}", combined.warnings);
        let warning = &combined.warnings[0];
        assert_eq!(warning.kind, WarningKind::Fallback);
        assert_eq!(warning.subject, Subject::Response);
        assert!(
            warning.message.contains("`primary` and `backup`"),
            "{}",
            warning.message
        );

        // A response without a provider takes the next one's, and an
        // unattributed chunk is not a mix.
        let mut combined = EmbeddingResponse::new(vec![vec![1.0]]);
        combined.extend(EmbeddingResponse::new(vec![vec![2.0]]).provider("primary"));
        combined.extend(EmbeddingResponse::new(vec![vec![3.0]]));
        assert_eq!(combined.provider.as_deref(), Some("primary"));
        assert!(combined.warnings.is_empty(), "{:?}", combined.warnings);
    }
}
