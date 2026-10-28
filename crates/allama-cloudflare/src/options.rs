use std::collections::BTreeMap;

use allama::{ExtraMap, ProviderOptions};
use serde::{Deserialize, Serialize};

use crate::NAMESPACE;

/// Options of an extraction request on Workers AI (namespace
/// [`NAMESPACE`], `cloudflare-workers-ai`).
///
/// Clairic's tuning that is not part of the portable model (ADR 0018):
/// where its null decision falls per field, and the extra statistics it
/// returns on request.
///
/// ```
/// use allama::ExtractionRequest;
/// use allama::extraction::Field;
/// use allama_cloudflare::ExtractionRequestOptions;
///
/// let request = ExtractionRequest::new("@cf/cloudflare/clairic", "Total due: $1,080")
///     .field("total", Field::number().decimals(2))
///     .with_option(
///         ExtractionRequestOptions::new()
///             .null_threshold("total", 0.3)
///             .include_quantiles(),
///     );
/// # let _ = request;
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ExtractionRequestOptions {
    /// Per field, the null probability above which its value is null
    /// (Clairic's default is 0.5). A name the request has no field for is
    /// an [`Error::InvalidRequest`](allama::Error::InvalidRequest).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub null_thresholds: BTreeMap<String, f64>,
    /// Ask for the 99 percentiles of every estimate, kept in provider data
    /// as [`Distributions`](crate::Distributions).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub include_quantiles: bool,
    /// Ask for the text the model's spans index into, kept in provider
    /// data as [`RunDetails::state_text`](crate::RunDetails::state_text).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub return_state_text: bool,
    /// Keep the verbatim response body in provider data as
    /// [`RawResponse`](crate::RawResponse).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub raw_response: bool,
    /// Request body fields the adapter does not model, merged into the body
    /// as given (in the JSON form, the nested `extra` object). A field the
    /// adapter owns (`model`, `state`, `fields`, `images`, `reference_date`,
    /// `options`) is refused.
    #[serde(default, skip_serializing_if = "ExtraMap::is_empty")]
    pub extra: ExtraMap,
    /// The options argument of a Worker's binding call (`env.AI.run(model,
    /// input, options)`), such as `{"gateway": {"id": "my-gateway"}}` to
    /// route the call through an AI Gateway, read by
    /// `allama-cloudflare-worker`. The REST API has no such argument: this
    /// provider drops a non-empty value with a warning.
    #[serde(default, skip_serializing_if = "ExtraMap::is_empty")]
    pub run_options: ExtraMap,
}

impl ExtractionRequestOptions {
    /// No options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the null threshold of the field named `field`.
    #[must_use]
    pub fn null_threshold(mut self, field: impl Into<String>, threshold: f64) -> Self {
        self.null_thresholds.insert(field.into(), threshold);
        self
    }

    /// Ask for every estimate's percentiles.
    #[must_use]
    pub fn include_quantiles(mut self) -> Self {
        self.include_quantiles = true;
        self
    }

    /// Ask for the text the spans index into.
    #[must_use]
    pub fn return_state_text(mut self) -> Self {
        self.return_state_text = true;
        self
    }

    /// Keep the verbatim response body.
    #[must_use]
    pub fn raw_response(mut self) -> Self {
        self.raw_response = true;
        self
    }

    /// Set a request body field the adapter does not model.
    #[must_use]
    pub fn extra(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra.insert(key.into(), value);
        self
    }

    /// Set an option of a Worker's binding call (see
    /// [`run_options`](Self::run_options)).
    #[must_use]
    pub fn run_option(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.run_options.insert(key.into(), value);
        self
    }
}

impl ProviderOptions for ExtractionRequestOptions {
    const NAMESPACE: &'static str = NAMESPACE;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_json() {
        let options = ExtractionRequestOptions::new()
            .null_threshold("total", 0.3)
            .include_quantiles()
            .extra("experimental", serde_json::json!(true));
        let json = serde_json::to_value(&options).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "null_thresholds": {"total": 0.3},
                "include_quantiles": true,
                "extra": {"experimental": true}
            })
        );
        let back: ExtractionRequestOptions = serde_json::from_value(json).unwrap();
        assert_eq!(back, options);
        assert_eq!(
            serde_json::to_value(ExtractionRequestOptions::new()).unwrap(),
            serde_json::json!({})
        );
    }
}
