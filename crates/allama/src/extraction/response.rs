use std::borrow::Cow;
use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::{
    Date, DateTime, DurationUnit, ExtractionRequest, Field, FieldKind, ListItemKind, TextSpan, Time,
};
use crate::decision::named;
use crate::{Error, ProviderData, Result, Usage, Warning};

/// Answers to an [`ExtractionRequest`], by field name.
///
/// ```
/// use allama::ExtractionResponse;
/// use allama::extraction::{Answer, NumberAnswer, TextAnswer, TextSpan};
///
/// let response = ExtractionResponse::new([
///     ("order_id", Answer::from(TextAnswer::new(Some("A-88412")).span(TextSpan::new(6, 13)))),
///     ("total", Answer::from(NumberAnswer::new(Some(701.1)).null_probability(1.6e-5))),
///     ("tracking", Answer::from(TextAnswer::new(None::<String>).null_probability(1.0))),
/// ]);
/// assert_eq!(response.text("order_id").unwrap().value.as_deref(), Some("A-88412"));
/// assert_eq!(response.number("total").unwrap().value, Some(701.1));
/// assert!(response.text("tracking").unwrap().is_null());
/// ```
///
/// Answers hold what the provider reported: values are never clamped to a
/// field's bounds, and a null is the provider's determination, whatever
/// its null probability.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ExtractionResponse {
    /// One answer per field, keyed by field name, in the order the provider
    /// reported them.
    #[serde(with = "named")]
    pub answers: Vec<(String, Answer)>,
    /// Provider-assigned response identifier, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Model that answered, as reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Provider request identifier (from transport headers), for support.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// Identity of the provider that actually served the request. Wrappers
    /// such as fallback leave the serving provider's value in place.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<Cow<'static, str>>,
    /// Token usage, when reported. Extraction models bill input only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Non-fatal warnings, such as a span the adapter could not place.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<Warning>,
    /// Provider-specific response data: a provider's own statistics about
    /// each answer (confidence, a guess for a null), keyed by field name.
    #[serde(default, skip_serializing_if = "ProviderData::is_empty")]
    pub provider_data: ProviderData,
}

impl ExtractionResponse {
    /// A response with `answers`.
    #[must_use]
    pub fn new<I, N>(answers: I) -> Self
    where
        I: IntoIterator<Item = (N, Answer)>,
        N: Into<String>,
    {
        Self {
            answers: answers
                .into_iter()
                .map(|(name, answer)| (name.into(), answer))
                .collect(),
            ..Self::default()
        }
    }

    /// Set the model.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the response id.
    #[must_use]
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set the request id.
    #[must_use]
    pub fn request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    /// Set the serving provider's identity.
    #[must_use]
    pub fn provider(mut self, provider: impl Into<Cow<'static, str>>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Set usage.
    #[must_use]
    pub fn usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
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

    /// The answer to the field named `name`.
    #[must_use]
    pub fn answer(&self, name: &str) -> Option<&Answer> {
        named::get(&self.answers, name)
    }

    /// The answer to the text field named `name`.
    #[must_use]
    pub fn text(&self, name: &str) -> Option<&TextAnswer> {
        self.answer(name)?.as_text()
    }

    /// The answer to the number field named `name`.
    #[must_use]
    pub fn number(&self, name: &str) -> Option<&NumberAnswer> {
        self.answer(name)?.as_number()
    }

    /// The answer to the date field named `name`.
    #[must_use]
    pub fn date(&self, name: &str) -> Option<&DateAnswer> {
        self.answer(name)?.as_date()
    }

    /// The answer to the time field named `name`.
    #[must_use]
    pub fn time(&self, name: &str) -> Option<&TimeAnswer> {
        self.answer(name)?.as_time()
    }

    /// The answer to the datetime field named `name`.
    #[must_use]
    pub fn datetime(&self, name: &str) -> Option<&DateTimeAnswer> {
        self.answer(name)?.as_datetime()
    }

    /// The answer to the duration field named `name`.
    #[must_use]
    pub fn duration(&self, name: &str) -> Option<&DurationAnswer> {
        self.answer(name)?.as_duration()
    }

    /// The answer to the list field named `name`.
    #[must_use]
    pub fn list(&self, name: &str) -> Option<&ListAnswer> {
        self.answer(name)?.as_list()
    }

    /// The answer to the estimate field named `name`.
    #[must_use]
    pub fn estimate(&self, name: &str) -> Option<&EstimateAnswer> {
        self.answer(name)?.as_estimate()
    }

    /// The answer to the forecast field named `name`.
    #[must_use]
    pub fn forecast(&self, name: &str) -> Option<&ForecastAnswer> {
        self.answer(name)?.as_forecast()
    }

    /// Check that this response answers `request`.
    ///
    /// Every field must be answered exactly once by an answer of its kind,
    /// and nothing else may be answered. A null value carries no span;
    /// spans lie inside the request's input on `char` boundaries; list
    /// items are of the field's item kind; a duration is in the field's
    /// unit; probabilities lie in `[0, 1]`; an estimate's value, median and
    /// interval lie inside the field's bounds, up to rounding, with the
    /// interval's low end at or below its high end; a forecast has one
    /// step per `horizon`, each checked as an estimate.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnexpectedResponse`] naming the first offending
    /// field.
    pub fn validate(&self, request: &ExtractionRequest) -> Result<()> {
        let mut seen = HashSet::new();
        for (name, answer) in &self.answers {
            if !seen.insert(name.as_str()) {
                return Err(unexpected(name, "answered more than once"));
            }
            let Some(field) = request.find_field(name) else {
                return Err(unexpected(name, "was not asked"));
            };
            if answer.kind() != field.kind() {
                return Err(unexpected(
                    name,
                    format!("is a {} field, answered as {}", field.kind(), answer.kind()),
                ));
            }
            match (field, answer) {
                (Field::Text(_), Answer::Text(answer)) => {
                    check_null_probability(name, answer.null_probability)?;
                    if answer.value.is_none() && answer.span.is_some() {
                        return Err(unexpected(name, "is null but has a span"));
                    }
                    if let Some(span) = answer.span {
                        check_span(name, span, &request.input)?;
                    }
                }
                (Field::Number(_), Answer::Number(answer)) => {
                    check_null_probability(name, answer.null_probability)?;
                    if let Some(value) = answer.value {
                        check_finite(name, "value", value)?;
                    }
                }
                (Field::Date(_), Answer::Date(answer)) => {
                    check_null_probability(name, answer.null_probability)?;
                }
                (Field::Time(_), Answer::Time(answer)) => {
                    check_null_probability(name, answer.null_probability)?;
                }
                (Field::DateTime(_), Answer::DateTime(answer)) => {
                    check_null_probability(name, answer.null_probability)?;
                }
                (Field::Duration(field), Answer::Duration(answer)) => {
                    check_null_probability(name, answer.null_probability)?;
                    if let Some(value) = answer.value {
                        check_finite(name, "value", value)?;
                    }
                    if answer.unit != field.unit {
                        return Err(unexpected(
                            name,
                            format!("is asked in {}, answered in {}", field.unit, answer.unit),
                        ));
                    }
                }
                (Field::List(field), Answer::List(answer)) => {
                    for (index, item) in answer.items.iter().enumerate() {
                        if item.value.kind() != field.items {
                            return Err(unexpected(
                                name,
                                format!(
                                    "lists {} items, but item {index} is {}",
                                    field.items,
                                    item.value.kind()
                                ),
                            ));
                        }
                        if let ListValue::Number(value) = item.value {
                            check_finite(name, &format!("item {index}"), value)?;
                        }
                        if let Some(span) = item.span {
                            check_span(name, span, &request.input)?;
                        }
                    }
                }
                (Field::Estimate(field), Answer::Estimate(answer)) => {
                    check_estimate(name, answer, field.min, field.max)?;
                }
                (Field::Forecast(field), Answer::Forecast(answer)) => {
                    if answer.steps.len() != field.horizon {
                        return Err(unexpected(
                            name,
                            format!(
                                "forecasts {} steps for a horizon of {}",
                                answer.steps.len(),
                                field.horizon
                            ),
                        ));
                    }
                    for step in &answer.steps {
                        check_estimate(name, step, field.min, field.max)?;
                    }
                }
                _ => unreachable!("kinds were compared above"),
            }
        }
        for (name, _) in &request.fields {
            if !seen.contains(name.as_str()) {
                return Err(unexpected(name, "was not answered"));
            }
        }
        Ok(())
    }
}

/// The answer to one [`Field`], of the same kind.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Answer {
    /// Answer to a text field.
    Text(TextAnswer),
    /// Answer to a number field.
    Number(NumberAnswer),
    /// Answer to a date field.
    Date(DateAnswer),
    /// Answer to a time field.
    Time(TimeAnswer),
    /// Answer to a datetime field.
    #[serde(rename = "datetime")]
    DateTime(DateTimeAnswer),
    /// Answer to a duration field.
    Duration(DurationAnswer),
    /// Answer to a list field.
    List(ListAnswer),
    /// Answer to an estimate field.
    Estimate(EstimateAnswer),
    /// Answer to a forecast field.
    Forecast(ForecastAnswer),
}

macro_rules! answer_accessors {
    ($($variant:ident($ty:ident) => $as:ident),+ $(,)?) => {
        impl Answer {
            /// The kind of field this answers.
            #[must_use]
            pub fn kind(&self) -> FieldKind {
                match self {
                    $(Self::$variant(_) => FieldKind::$variant,)+
                }
            }

            $(
                #[doc = concat!("The ", stringify!($variant), " answer, if this is one.")]
                #[must_use]
                pub fn $as(&self) -> Option<&$ty> {
                    match self {
                        Self::$variant(answer) => Some(answer),
                        _ => None,
                    }
                }
            )+
        }

        $(impl From<$ty> for Answer {
            fn from(answer: $ty) -> Self {
                Self::$variant(answer)
            }
        })+
    };
}

answer_accessors!(
    Text(TextAnswer) => as_text,
    Number(NumberAnswer) => as_number,
    Date(DateAnswer) => as_date,
    Time(TimeAnswer) => as_time,
    DateTime(DateTimeAnswer) => as_datetime,
    Duration(DurationAnswer) => as_duration,
    List(ListAnswer) => as_list,
    Estimate(EstimateAnswer) => as_estimate,
    Forecast(ForecastAnswer) => as_forecast,
);

/// A span of the input, or null.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TextAnswer {
    /// The text, copied from the input, or `None` when the input does not
    /// determine it.
    pub value: Option<String>,
    /// The probability that the input does not determine the value, as the
    /// model reports it; `None` when the provider reports none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub null_probability: Option<f64>,
    /// Where the value was copied from, when the adapter could place it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<TextSpan>,
}

/// A number, or null.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct NumberAnswer {
    /// The number, or `None` when the input does not determine it.
    pub value: Option<f64>,
    /// The probability that the input does not determine the value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub null_probability: Option<f64>,
}

/// A calendar date, or null.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DateAnswer {
    /// The date, or `None` when the input does not determine it.
    pub value: Option<Date>,
    /// The probability that the input does not determine the value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub null_probability: Option<f64>,
}

/// A time of day, or null.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TimeAnswer {
    /// The time, or `None` when the input does not determine it.
    pub value: Option<Time>,
    /// The probability that the input does not determine the value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub null_probability: Option<f64>,
}

/// A date and time, or null.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DateTimeAnswer {
    /// The datetime, or `None` when the input does not determine it.
    pub value: Option<DateTime>,
    /// The probability that the input does not determine the value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub null_probability: Option<f64>,
}

/// A length of time in the field's unit, or null.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DurationAnswer {
    /// The duration in `unit`s, or `None` when the input does not
    /// determine it.
    pub value: Option<f64>,
    /// The unit, the field's.
    pub unit: DurationUnit,
    /// The probability that the input does not determine the value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub null_probability: Option<f64>,
}

/// Every matching value, in input order. An empty list is an answer: the
/// input has none.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ListAnswer {
    /// The items.
    pub items: Vec<ListItem>,
}

/// One item of a [`ListAnswer`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ListItem {
    /// The value.
    pub value: ListValue,
    /// Where a text value was copied from, when the adapter could place
    /// it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<TextSpan>,
}

/// The value of a list item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ListValue {
    /// A span of the input.
    Text(String),
    /// A number.
    Number(f64),
    /// A calendar date.
    Date(Date),
}

/// A distribution over a value, summarised.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EstimateAnswer {
    /// The point estimate the provider reports.
    pub value: f64,
    /// The median of the distribution, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub median: Option<f64>,
    /// An interval of the distribution, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval: Option<PredictionInterval>,
}

/// A central interval of a distribution: the value lies between `low` and
/// `high` with `probability`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PredictionInterval {
    /// The low end.
    pub low: f64,
    /// The high end.
    pub high: f64,
    /// The probability mass between the ends: 0.8 for an 80% interval.
    pub probability: f64,
}

/// Future values of a series, one estimate per step ahead.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ForecastAnswer {
    /// One estimate per step ahead, nearest first.
    pub steps: Vec<EstimateAnswer>,
}

macro_rules! scalar_answer {
    ($ty:ident, $value:ty, $what:literal) => {
        impl $ty {
            #[doc = concat!("An answer with ", $what, ", or null.")]
            #[must_use]
            pub fn new(value: Option<impl Into<$value>>) -> Self {
                let mut answer = Self::empty();
                answer.value = value.map(Into::into);
                answer
            }

            /// Set the probability that the input does not determine the value.
            #[must_use]
            pub fn null_probability(mut self, probability: f64) -> Self {
                self.null_probability = Some(probability);
                self
            }

            /// Whether the input does not determine the value.
            #[must_use]
            pub fn is_null(&self) -> bool {
                self.value.is_none()
            }
        }
    };
}

impl TextAnswer {
    fn empty() -> Self {
        Self {
            value: None,
            null_probability: None,
            span: None,
        }
    }

    /// Set where the value was copied from.
    #[must_use]
    pub fn span(mut self, span: TextSpan) -> Self {
        self.span = Some(span);
        self
    }
}

impl NumberAnswer {
    fn empty() -> Self {
        Self {
            value: None,
            null_probability: None,
        }
    }
}

impl DateAnswer {
    fn empty() -> Self {
        Self {
            value: None,
            null_probability: None,
        }
    }
}

impl TimeAnswer {
    fn empty() -> Self {
        Self {
            value: None,
            null_probability: None,
        }
    }
}

impl DateTimeAnswer {
    fn empty() -> Self {
        Self {
            value: None,
            null_probability: None,
        }
    }
}

scalar_answer!(TextAnswer, String, "the text");
scalar_answer!(NumberAnswer, f64, "the number");
scalar_answer!(DateAnswer, Date, "the date");
scalar_answer!(TimeAnswer, Time, "the time");
scalar_answer!(DateTimeAnswer, DateTime, "the datetime");

impl DurationAnswer {
    /// An answer in `unit`s, or null.
    #[must_use]
    pub fn new(value: Option<f64>, unit: DurationUnit) -> Self {
        Self {
            value,
            unit,
            null_probability: None,
        }
    }

    /// Set the probability that the input does not determine the value.
    #[must_use]
    pub fn null_probability(mut self, probability: f64) -> Self {
        self.null_probability = Some(probability);
        self
    }

    /// Whether the input does not determine the value.
    #[must_use]
    pub fn is_null(&self) -> bool {
        self.value.is_none()
    }
}

impl ListAnswer {
    /// A list of `items`.
    #[must_use]
    pub fn new(items: impl IntoIterator<Item = ListItem>) -> Self {
        Self {
            items: items.into_iter().collect(),
        }
    }

    /// The text of every text item.
    pub fn texts(&self) -> impl Iterator<Item = &str> {
        self.items.iter().filter_map(|item| match &item.value {
            ListValue::Text(text) => Some(text.as_str()),
            _ => None,
        })
    }

    /// Every number item.
    pub fn numbers(&self) -> impl Iterator<Item = f64> + '_ {
        self.items.iter().filter_map(|item| match item.value {
            ListValue::Number(value) => Some(value),
            _ => None,
        })
    }

    /// Every date item.
    pub fn dates(&self) -> impl Iterator<Item = &Date> {
        self.items.iter().filter_map(|item| match &item.value {
            ListValue::Date(date) => Some(date),
            _ => None,
        })
    }
}

impl ListItem {
    /// A text item.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            value: ListValue::Text(text.into()),
            span: None,
        }
    }

    /// A number item.
    #[must_use]
    pub fn number(value: f64) -> Self {
        Self {
            value: ListValue::Number(value),
            span: None,
        }
    }

    /// A date item.
    #[must_use]
    pub fn date(date: Date) -> Self {
        Self {
            value: ListValue::Date(date),
            span: None,
        }
    }

    /// Set where the item was copied from.
    #[must_use]
    pub fn span(mut self, span: TextSpan) -> Self {
        self.span = Some(span);
        self
    }
}

impl ListValue {
    /// The kind of item.
    #[must_use]
    pub fn kind(&self) -> ListItemKind {
        match self {
            Self::Text(_) => ListItemKind::Text,
            Self::Number(_) => ListItemKind::Number,
            Self::Date(_) => ListItemKind::Date,
        }
    }
}

impl EstimateAnswer {
    /// An estimate with a point value.
    #[must_use]
    pub fn new(value: f64) -> Self {
        Self {
            value,
            median: None,
            interval: None,
        }
    }

    /// Set the median.
    #[must_use]
    pub fn median(mut self, median: f64) -> Self {
        self.median = Some(median);
        self
    }

    /// Set the interval that holds the value with `probability`.
    #[must_use]
    pub fn interval(mut self, low: f64, high: f64, probability: f64) -> Self {
        self.interval = Some(PredictionInterval {
            low,
            high,
            probability,
        });
        self
    }
}

impl PredictionInterval {
    /// An interval from `low` to `high` holding `probability` of the mass.
    #[must_use]
    pub fn new(low: f64, high: f64, probability: f64) -> Self {
        Self {
            low,
            high,
            probability,
        }
    }

    /// The width of the interval.
    #[must_use]
    pub fn width(&self) -> f64 {
        self.high - self.low
    }
}

impl ForecastAnswer {
    /// A forecast of `steps`, nearest first.
    #[must_use]
    pub fn new(steps: impl IntoIterator<Item = EstimateAnswer>) -> Self {
        Self {
            steps: steps.into_iter().collect(),
        }
    }

    /// The point value of every step, nearest first.
    pub fn values(&self) -> impl Iterator<Item = f64> + '_ {
        self.steps.iter().map(|step| step.value)
    }
}

/// Rounding slack accepted on probabilities and bounds.
const TOLERANCE: f64 = 1e-6;

fn check_null_probability(name: &str, probability: Option<f64>) -> Result<()> {
    match probability {
        Some(p) if !(p.is_finite() && (-TOLERANCE..=1.0 + TOLERANCE).contains(&p)) => Err(
            unexpected(name, format!("reports null probability {p} outside 0..=1")),
        ),
        _ => Ok(()),
    }
}

fn check_finite(name: &str, what: &str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(unexpected(
            name,
            format!("reports a non-finite {what} {value}"),
        ))
    }
}

fn check_span(name: &str, span: TextSpan, input: &str) -> Result<()> {
    if span.end < span.start || span.text(input).is_none() {
        return Err(unexpected(
            name,
            format!(
                "reports span {}..{} outside the input or off a char boundary",
                span.start, span.end
            ),
        ));
    }
    Ok(())
}

fn check_estimate(name: &str, answer: &EstimateAnswer, min: f64, max: f64) -> Result<()> {
    let within = |what: &str, value: f64| -> Result<()> {
        check_finite(name, what, value)?;
        if (min - TOLERANCE..=max + TOLERANCE).contains(&value) {
            Ok(())
        } else {
            Err(unexpected(
                name,
                format!("reports {what} {value} outside its bounds {min}..={max}"),
            ))
        }
    };
    within("value", answer.value)?;
    if let Some(median) = answer.median {
        within("median", median)?;
    }
    if let Some(interval) = answer.interval {
        within("interval low", interval.low)?;
        within("interval high", interval.high)?;
        if interval.low > interval.high {
            return Err(unexpected(
                name,
                format!(
                    "reports an interval whose low {} is above its high {}",
                    interval.low, interval.high
                ),
            ));
        }
        let p = interval.probability;
        if !(p.is_finite() && (-TOLERANCE..=1.0 + TOLERANCE).contains(&p)) {
            return Err(unexpected(
                name,
                format!("reports interval probability {p} outside 0..=1"),
            ));
        }
    }
    Ok(())
}

fn unexpected(name: &str, problem: impl std::fmt::Display) -> Error {
    Error::unexpected_response(format!("extraction field {name:?} {problem}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extraction::ListItemKind;
    use serde_json::json;

    const INPUT: &str = "Order A-88412 arrived Sept 28; café door dented.";

    fn request() -> ExtractionRequest {
        ExtractionRequest::new("m", INPUT)
            .field("order_id", Field::text())
            .field("total", Field::number())
            .field("delivered", Field::date())
            .field("late_by", Field::duration(DurationUnit::Days))
            .field("damage", Field::list(ListItemKind::Text))
            .field("repair", Field::estimate(10.0, 5000.0))
            .field("demand", Field::forecast(0.0, 100.0).horizon(2))
    }

    fn response() -> ExtractionResponse {
        ExtractionResponse::new([
            (
                "order_id",
                Answer::from(
                    TextAnswer::new(Some("A-88412"))
                        .span(TextSpan::new(6, 13))
                        .null_probability(3.4e-6),
                ),
            ),
            ("total", NumberAnswer::new(Some(701.1)).into()),
            (
                "delivered",
                DateAnswer::new(Some("2026-09-28".parse::<Date>().unwrap())).into(),
            ),
            (
                "late_by",
                DurationAnswer::new(Some(11.0), DurationUnit::Days).into(),
            ),
            (
                "damage",
                ListAnswer::new([ListItem::text("dented").span(TextSpan::new(42, 48))]).into(),
            ),
            (
                "repair",
                EstimateAnswer::new(180.0)
                    .median(175.0)
                    .interval(120.0, 260.0, 0.8)
                    .into(),
            ),
            (
                "demand",
                ForecastAnswer::new([EstimateAnswer::new(50.0), EstimateAnswer::new(60.0)]).into(),
            ),
        ])
    }

    fn problem(response: ExtractionResponse) -> String {
        match response.validate(&request()).unwrap_err() {
            Error::UnexpectedResponse { message, .. } => message,
            other => panic!("expected UnexpectedResponse, got {other:?}"),
        }
    }

    fn with(name: &str, answer: impl Into<Answer>) -> ExtractionResponse {
        let mut response = response();
        let answer = answer.into();
        for (key, slot) in &mut response.answers {
            if key == name {
                *slot = answer.clone();
            }
        }
        response
    }

    #[test]
    fn matching_response_is_valid_and_accessible() {
        let response = response();
        response.validate(&request()).unwrap();
        assert_eq!(
            response.text("order_id").unwrap().span.unwrap().text(INPUT),
            Some("A-88412")
        );
        assert_eq!(response.number("total").unwrap().value, Some(701.1));
        assert_eq!(
            response
                .date("delivered")
                .unwrap()
                .value
                .as_ref()
                .unwrap()
                .as_str(),
            "2026-09-28"
        );
        assert_eq!(response.duration("late_by").unwrap().value, Some(11.0));
        assert_eq!(
            response.list("damage").unwrap().texts().collect::<Vec<_>>(),
            ["dented"]
        );
        assert_eq!(
            response
                .estimate("repair")
                .unwrap()
                .interval
                .unwrap()
                .width(),
            140.0
        );
        assert_eq!(
            response
                .forecast("demand")
                .unwrap()
                .values()
                .collect::<Vec<_>>(),
            [50.0, 60.0]
        );
        assert!(response.time("order_id").is_none());
        assert!(response.answer("missing").is_none());
    }

    #[test]
    fn every_field_is_answered_once_by_its_kind() {
        let mut extra = response();
        extra
            .answers
            .push(("extra".into(), NumberAnswer::new(None::<f64>).into()));
        assert!(problem(extra).contains("was not asked"));
        let mut twice = response();
        twice
            .answers
            .push(("total".into(), NumberAnswer::new(None::<f64>).into()));
        assert!(problem(twice).contains("more than once"));
        let mut missing = response();
        missing.answers.retain(|(name, _)| name != "total");
        assert!(problem(missing).contains("was not answered"));
        assert!(problem(with("total", TextAnswer::new(Some("x")))).contains("answered as text"));
    }

    #[test]
    fn scalar_answers_are_checked() {
        assert!(
            problem(with(
                "total",
                NumberAnswer::new(Some(1.0)).null_probability(1.5)
            ))
            .contains("null probability")
        );
        assert!(problem(with("total", NumberAnswer::new(Some(f64::NAN)))).contains("non-finite"));
        assert!(
            problem(with(
                "order_id",
                TextAnswer::new(None::<String>).span(TextSpan::new(0, 1))
            ))
            .contains("null but has a span")
        );
        // Off a char boundary: inside `é`.
        assert!(
            problem(with(
                "order_id",
                TextAnswer::new(Some("x")).span(TextSpan::new(35, 36))
            ))
            .contains("char boundary")
        );
        assert!(
            problem(with(
                "order_id",
                TextAnswer::new(Some("x")).span(TextSpan::new(0, 999))
            ))
            .contains("outside the input")
        );
        assert!(
            problem(with(
                "late_by",
                DurationAnswer::new(Some(1.0), DurationUnit::Hours)
            ))
            .contains("asked in days, answered in hours")
        );
    }

    #[test]
    fn list_items_match_the_field() {
        assert!(
            problem(with("damage", ListAnswer::new([ListItem::number(1.0)])))
                .contains("item 0 is number")
        );
        // An empty list is an answer.
        with("damage", ListAnswer::default())
            .validate(&request())
            .unwrap();
    }

    #[test]
    fn estimates_stay_inside_their_bounds() {
        assert!(
            problem(with("repair", EstimateAnswer::new(5001.0))).contains("outside its bounds")
        );
        assert!(problem(with("repair", EstimateAnswer::new(100.0).median(9.0))).contains("median"));
        assert!(
            problem(with(
                "repair",
                EstimateAnswer::new(100.0).interval(300.0, 200.0, 0.8)
            ))
            .contains("low 300 is above its high 200")
        );
        assert!(
            problem(with(
                "repair",
                EstimateAnswer::new(100.0).interval(100.0, 200.0, 1.2)
            ))
            .contains("interval probability")
        );
        // Rounding slack.
        with("repair", EstimateAnswer::new(5000.0000001))
            .validate(&request())
            .unwrap();
        assert!(
            problem(with(
                "demand",
                ForecastAnswer::new([EstimateAnswer::new(1.0)])
            ))
            .contains("forecasts 1 steps for a horizon of 2")
        );
        assert!(
            problem(with(
                "demand",
                ForecastAnswer::new([EstimateAnswer::new(1.0), EstimateAnswer::new(101.0)])
            ))
            .contains("outside its bounds")
        );
    }

    #[test]
    fn serializes_tagged_by_kind() {
        let answer: Answer = TextAnswer::new(Some("A")).span(TextSpan::new(0, 1)).into();
        assert_eq!(
            serde_json::to_value(&answer).unwrap(),
            json!({"type": "text", "value": "A", "span": {"start": 0, "end": 1}})
        );
        let answer: Answer = TextAnswer::new(None::<String>).null_probability(1.0).into();
        assert_eq!(
            serde_json::to_value(&answer).unwrap(),
            json!({"type": "text", "value": null, "null_probability": 1.0})
        );
        let answer: Answer = ListAnswer::new([ListItem::text("a"), ListItem::number(2.0)]).into();
        assert_eq!(
            serde_json::to_value(&answer).unwrap(),
            json!({"type": "list", "items": [
                {"value": {"type": "text", "value": "a"}},
                {"value": {"type": "number", "value": 2.0}}
            ]})
        );
        let answer: Answer =
            DateTimeAnswer::new(Some("2026-07-01T13:34:00".parse::<DateTime>().unwrap())).into();
        assert_eq!(serde_json::to_value(&answer).unwrap()["type"], "datetime");
        let text = serde_json::to_string(&response()).unwrap();
        let back: ExtractionResponse = serde_json::from_str(&text).unwrap();
        assert_eq!(back, response());
    }
}
