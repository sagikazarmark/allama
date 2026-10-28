use std::collections::HashSet;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::{Date, Field};
use crate::decision::named;
use crate::{Error, Image, ProviderOptions, RequestOptions, Result};

/// Named, typed fields about one input, answered by an extraction model.
///
/// Every field parameter is semantic: a provider either sends it as given
/// or refuses the field. There are no hints to degrade, so, like a decision
/// request, an extraction request carries no
/// [`UnsupportedPolicy`](crate::UnsupportedPolicy).
///
/// ```
/// use allama::ExtractionRequest;
/// use allama::extraction::{Field, ListItemKind};
///
/// let request = ExtractionRequest::new(
///     "@cf/cloudflare/clairic",
///     "Order A-88412 arrived on Sept 28 with a cracked screen. I paid $649.00 plus $52.10 tax.",
/// )
/// .field("order_id", Field::text().instructions("Order number"))
/// .field("delivered_on", Field::date().instructions("Delivery date"))
/// .field("amount_paid", Field::number().decimals(2).unit("USD").instructions("Total paid including tax"))
/// .field("damage", Field::list(ListItemKind::Text).instructions("Each kind of damage described"))
/// .reference_date("2026-10-09".parse()?);
/// assert_eq!(request.fields.len(), 4);
/// # Ok::<(), allama::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ExtractionRequest {
    /// Model identifier.
    pub model: String,
    /// The text the fields are about. Text answers are spans of it. It may
    /// be empty only when there are images.
    pub input: String,
    /// Images the model reads together with the input, in order, for
    /// models that report
    /// [`ExtractionCapability::ImageInput`](crate::ExtractionCapability::ImageInput).
    ///
    /// [`Image::detail`] is a hint no extraction API honors: adapters warn
    /// about it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<Image>,
    /// Fields by unique name, in order.
    #[serde(with = "named")]
    pub fields: Vec<(String, Field)>,
    /// The date that anchors dates the input writes without a year.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_date: Option<Date>,
    /// Provider-specific options.
    #[serde(default, skip_serializing_if = "RequestOptions::is_empty")]
    pub options: RequestOptions,
}

impl ExtractionRequest {
    /// A request for `model` about `input`, with no fields yet.
    #[must_use]
    pub fn new(model: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            input: input.into(),
            images: Vec::new(),
            fields: Vec::new(),
            reference_date: None,
            options: RequestOptions::new(),
        }
    }

    /// Append an image the model reads together with the input.
    #[must_use]
    pub fn image(mut self, image: impl Into<Image>) -> Self {
        self.images.push(image.into());
        self
    }

    /// Append a field named `name`. Names must be unique.
    #[must_use]
    pub fn field(mut self, name: impl Into<String>, field: impl Into<Field>) -> Self {
        self.fields.push((name.into(), field.into()));
        self
    }

    /// Append several named fields.
    #[must_use]
    pub fn fields<I, N, F>(mut self, fields: I) -> Self
    where
        I: IntoIterator<Item = (N, F)>,
        N: Into<String>,
        F: Into<Field>,
    {
        self.fields.extend(
            fields
                .into_iter()
                .map(|(name, field)| (name.into(), field.into())),
        );
        self
    }

    /// Anchor dates written without a year to `date`.
    #[must_use]
    pub fn reference_date(mut self, date: Date) -> Self {
        self.reference_date = Some(date);
        self
    }

    /// The field named `name`, if any.
    #[must_use]
    pub fn find_field(&self, name: &str) -> Option<&Field> {
        named::get(&self.fields, name)
    }

    /// Check the request's structure and `limits` before sending it.
    ///
    /// Providers call this through
    /// [`validate_extraction_request`](crate::ExtractionProvider::validate_extraction_request)
    /// before any I/O.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`] when:
    ///
    /// - the model is blank, or the input is blank and there are no images;
    /// - there are no fields, or a field name or instructions are blank, or
    ///   a name repeats;
    /// - an estimate's or forecast's bounds are not finite with `min` below
    ///   `max`, an estimate's `step` is not positive, or a forecast's
    ///   `horizon` is zero;
    /// - the request exceeds `limits`, its images included.
    pub fn validate(&self, limits: &ExtractionLimits) -> Result<()> {
        if self.model.trim().is_empty() {
            return Err(invalid("model must not be blank"));
        }
        if self.input.trim().is_empty() && self.images.is_empty() {
            return Err(invalid("input must not be blank unless images are given"));
        }
        if self.fields.is_empty() {
            return Err(invalid("at least one field is required"));
        }
        check_max(self.fields.len(), limits.max_fields, || "fields".into())?;
        check_max(self.images.len(), limits.max_images, || "images".into())?;

        let mut names = HashSet::new();
        for (name, field) in &self.fields {
            if name.trim().is_empty() {
                return Err(invalid("field names must not be blank"));
            }
            if !names.insert(name.as_str()) {
                return Err(invalid(format!("duplicate field name {name:?}")));
            }
            let fail = |problem: String| invalid(format!("field {name:?}: {problem}"));
            if field
                .instructions()
                .is_some_and(|text| text.trim().is_empty())
            {
                return Err(fail("instructions must not be blank".into()));
            }
            match field {
                Field::Estimate(estimate) => {
                    check_bounds(estimate.min, estimate.max, estimate.scale).map_err(fail)?;
                    if let Some(step) = estimate.step
                        && !(step.is_finite() && step > 0.0)
                    {
                        return Err(fail(format!("step {step} is not positive")));
                    }
                }
                Field::Forecast(forecast) => {
                    check_bounds(forecast.min, forecast.max, forecast.scale).map_err(fail)?;
                    if forecast.horizon == 0 {
                        return Err(fail("horizon must be at least 1".into()));
                    }
                    check_max(forecast.horizon, limits.max_forecast_horizon, || {
                        format!("forecast steps in field {name:?}")
                    })?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Attach provider options of the exact type.
    #[must_use]
    pub fn with_option<T: ProviderOptions + Serialize>(mut self, value: T) -> Self {
        self.options.insert(value);
        self
    }

    /// Attach an application value that is not a provider option: data a
    /// wrapper of yours reads back with [`option`](Self::option). Replaces
    /// any earlier value of that type. See
    /// [`RequestOptions::insert_opaque`](crate::RequestOptions::insert_opaque).
    #[must_use]
    pub fn with_opaque_option<T: Clone + Send + Sync + 'static>(mut self, value: T) -> Self {
        self.options.insert_opaque(value);
        self
    }

    /// Attach provider options in their namespaced JSON form.
    #[must_use]
    pub fn with_options_json(
        mut self,
        namespace: impl Into<String>,
        value: serde_json::Value,
    ) -> Self {
        self.options.insert_json(namespace, value);
        self
    }

    /// Borrow provider options of the exact type.
    #[must_use]
    pub fn option<T: 'static>(&self) -> Option<&T> {
        self.options.get::<T>()
    }

    /// Resolve provider options of type `T`, falling back to the default.
    ///
    /// # Errors
    ///
    /// See [`RequestOptions::resolve`].
    pub fn resolve_option<T>(&self) -> Result<T>
    where
        T: ProviderOptions + DeserializeOwned + Default,
    {
        Ok(self.options.resolve_or_default::<T>()?.into_owned())
    }

    /// Mutably borrow provider options of the exact type, when attached.
    pub fn option_mut<T: Clone + Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        self.options.get_mut::<T>()
    }

    /// Resolve provider options of type `T` for changing in place.
    ///
    /// # Errors
    ///
    /// See [`RequestOptions::resolve_mut`].
    pub fn resolve_option_mut<T>(&mut self) -> Result<&mut T>
    where
        T: ProviderOptions + Serialize + DeserializeOwned + Default,
    {
        self.options.resolve_mut::<T>()
    }
}

/// Request limits of an extraction model, as reported by
/// [`ExtractionProvider::extraction_limits`](crate::ExtractionProvider::extraction_limits).
///
/// `None` means the provider reports no limit. Rules of one endpoint that
/// are not counts (a field-name grammar, image sizes) are the adapter's
/// own checks, not limits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ExtractionLimits {
    /// Maximum fields per request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_fields: Option<usize>,
    /// Maximum images per request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_images: Option<usize>,
    /// Maximum steps ahead of a forecast field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_forecast_horizon: Option<usize>,
}

impl ExtractionLimits {
    /// No known limits.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum fields per request.
    #[must_use]
    pub fn max_fields(mut self, max: usize) -> Self {
        self.max_fields = Some(max);
        self
    }

    /// Set the maximum images per request.
    #[must_use]
    pub fn max_images(mut self, max: usize) -> Self {
        self.max_images = Some(max);
        self
    }

    /// Set the maximum steps ahead of a forecast.
    #[must_use]
    pub fn max_forecast_horizon(mut self, max: usize) -> Self {
        self.max_forecast_horizon = Some(max);
        self
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::invalid_request(message).authored()
}

fn check_bounds(
    min: f64,
    max: f64,
    scale: super::EstimateScale,
) -> std::result::Result<(), String> {
    if !(min.is_finite() && max.is_finite()) {
        return Err(format!("bounds {min}..{max} must be finite"));
    }
    if min >= max {
        return Err(format!("min {min} must be below max {max}"));
    }
    if scale == super::EstimateScale::Log && min <= 0.0 {
        return Err(format!("min {min} must be positive on a log scale"));
    }
    Ok(())
}

fn check_max(count: usize, max: Option<usize>, what: impl FnOnce() -> String) -> Result<()> {
    match max {
        Some(max) if count > max => Err(invalid(format!(
            "too many {}: {count}, the model accepts at most {max}",
            what()
        ))),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extraction::{DurationUnit, ListItemKind};

    fn problem(request: &ExtractionRequest, limits: &ExtractionLimits) -> String {
        match request.validate(limits).unwrap_err() {
            Error::InvalidRequest { message, .. } => message,
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn a_complete_request_is_valid() {
        let request = ExtractionRequest::new("m", "text")
            .field("a", Field::text())
            .field("b", Field::duration(DurationUnit::Days))
            .field("c", Field::list(ListItemKind::Number).decimals(2))
            .field("d", Field::estimate(1.0, 2.0).step(0.1))
            .field("e", Field::forecast(0.0, 1.0).horizon(3));
        request.validate(&ExtractionLimits::default()).unwrap();
        assert!(request.find_field("d").is_some());
        assert!(request.find_field("z").is_none());
    }

    #[test]
    fn structure_is_checked() {
        let limits = ExtractionLimits::default();
        let base = || ExtractionRequest::new("m", "text");
        assert!(
            problem(
                &ExtractionRequest::new(" ", "x").field("a", Field::text()),
                &limits
            )
            .contains("model")
        );
        assert!(
            problem(
                &ExtractionRequest::new("m", " ").field("a", Field::text()),
                &limits
            )
            .contains("input")
        );
        // Images alone carry the input.
        ExtractionRequest::new("m", "")
            .image(Image::base64("image/png", "aGk="))
            .field("a", Field::text())
            .validate(&limits)
            .unwrap();
        assert!(problem(&base(), &limits).contains("at least one field"));
        assert!(problem(&base().field(" ", Field::text()), &limits).contains("blank"));
        assert!(
            problem(
                &base().field("a", Field::text()).field("a", Field::number()),
                &limits
            )
            .contains("duplicate")
        );
        assert!(
            problem(
                &base().field("a", Field::text().instructions("  ")),
                &limits
            )
            .contains("instructions")
        );
        assert!(problem(&base().field("a", Field::estimate(2.0, 1.0)), &limits).contains("min"));
        assert!(
            problem(&base().field("a", Field::estimate(f64::NAN, 1.0)), &limits).contains("finite")
        );
        assert!(
            problem(
                &base().field("a", Field::estimate(0.0, 1.0).log_scale()),
                &limits
            )
            .contains("log scale")
        );
        assert!(
            problem(
                &base().field("a", Field::estimate(0.0, 1.0).step(0.0)),
                &limits
            )
            .contains("step")
        );
        assert!(
            problem(
                &base().field("a", Field::forecast(0.0, 1.0).horizon(0)),
                &limits
            )
            .contains("horizon")
        );
    }

    #[test]
    fn limits_are_checked() {
        let request = ExtractionRequest::new("m", "text")
            .image(Image::base64("image/png", "aGk="))
            .field("a", Field::text())
            .field("b", Field::forecast(0.0, 1.0).horizon(5));
        assert!(problem(&request, &ExtractionLimits::new().max_fields(1)).contains("fields"));
        assert!(problem(&request, &ExtractionLimits::new().max_images(0)).contains("images"));
        assert!(
            problem(&request, &ExtractionLimits::new().max_forecast_horizon(4))
                .contains("forecast steps")
        );
        request
            .validate(
                &ExtractionLimits::new()
                    .max_fields(2)
                    .max_images(1)
                    .max_forecast_horizon(5),
            )
            .unwrap();
    }

    #[test]
    fn serializes_fields_as_an_ordered_object() {
        let request = ExtractionRequest::new("m", "text")
            .field("zeta", Field::text())
            .field("alpha", Field::number())
            .reference_date("2026-10-09".parse().unwrap());
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["reference_date"], "2026-10-09");
        let text = serde_json::to_string(&request).unwrap();
        assert!(
            text.find("zeta").unwrap() < text.find("alpha").unwrap(),
            "{text}"
        );
        let back: ExtractionRequest = serde_json::from_str(&text).unwrap();
        assert_eq!(back, request);
    }
}
