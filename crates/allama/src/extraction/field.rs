use std::fmt;

use serde::{Deserialize, Serialize};

use crate::macros::enum_names;

/// A named, typed request for one value about an extraction input.
///
/// Every field is answered against the whole input, in one pass; answers
/// are not passed to later fields. Build one with the constructors, which
/// take what the kind requires, and the builders, which add instructions
/// and constraints:
///
/// ```
/// use allama::extraction::{DurationUnit, Field, ListItemKind};
///
/// let order_id = Field::text().instructions("Order number");
/// let amount = Field::number().decimals(2).unit("USD").instructions("Total paid including tax");
/// let late_by = Field::duration(DurationUnit::Days).integer();
/// let damage = Field::list(ListItemKind::Text).instructions("Each kind of damage described");
/// let repair = Field::estimate(10.0, 5000.0).log_scale().unit("USD");
/// let demand = Field::forecast(0.0, 1_000_000.0).horizon(7);
/// # let _ = (order_id, amount, late_by, damage, repair, demand);
/// ```
///
/// Every parameter changes what the model computes (the unit it answers
/// in, the precision it rounds to, the grid and scale it estimates on), so
/// none is a hint: a provider that cannot send a parameter as given refuses
/// the field before sending.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Field {
    /// A span of the input, such as a name or an identifier.
    Text(TextField),
    /// A number read from the input or computed from it.
    Number(NumberField),
    /// A calendar date.
    Date(DateField),
    /// A time of day.
    Time(TimeField),
    /// A date and time.
    #[serde(rename = "datetime")]
    DateTime(DateTimeField),
    /// A length of time in a unit of your choice.
    Duration(DurationField),
    /// Every matching value, in input order.
    List(ListField),
    /// A distribution over a value the input gives evidence for, inside
    /// declared bounds.
    Estimate(EstimateField),
    /// Future values of a series, one estimate per step ahead.
    Forecast(ForecastField),
}

impl Field {
    /// A text field: the answer is a span of the input.
    #[must_use]
    pub fn text() -> TextField {
        TextField::new()
    }

    /// A number field.
    #[must_use]
    pub fn number() -> NumberField {
        NumberField::new()
    }

    /// A calendar date field.
    #[must_use]
    pub fn date() -> DateField {
        DateField::new()
    }

    /// A time-of-day field.
    #[must_use]
    pub fn time() -> TimeField {
        TimeField::new()
    }

    /// A date-and-time field.
    #[must_use]
    pub fn datetime() -> DateTimeField {
        DateTimeField::new()
    }

    /// A duration field answered in `unit`.
    #[must_use]
    pub fn duration(unit: DurationUnit) -> DurationField {
        DurationField::new(unit)
    }

    /// A list field whose items are of `items` kind.
    #[must_use]
    pub fn list(items: ListItemKind) -> ListField {
        ListField::new(items)
    }

    /// An estimate field bounded by `min` and `max`.
    #[must_use]
    pub fn estimate(min: f64, max: f64) -> EstimateField {
        EstimateField::new(min, max)
    }

    /// A forecast field bounded by `min` and `max`, of one step ahead
    /// until [`horizon`](ForecastField::horizon) says otherwise.
    #[must_use]
    pub fn forecast(min: f64, max: f64) -> ForecastField {
        ForecastField::new(min, max)
    }

    /// The kind of field.
    #[must_use]
    pub fn kind(&self) -> FieldKind {
        match self {
            Self::Text(_) => FieldKind::Text,
            Self::Number(_) => FieldKind::Number,
            Self::Date(_) => FieldKind::Date,
            Self::Time(_) => FieldKind::Time,
            Self::DateTime(_) => FieldKind::DateTime,
            Self::Duration(_) => FieldKind::Duration,
            Self::List(_) => FieldKind::List,
            Self::Estimate(_) => FieldKind::Estimate,
            Self::Forecast(_) => FieldKind::Forecast,
        }
    }

    /// What the field should contain, when given.
    #[must_use]
    pub fn instructions(&self) -> Option<&str> {
        match self {
            Self::Text(f) => f.instructions.as_deref(),
            Self::Number(f) => f.instructions.as_deref(),
            Self::Date(f) => f.instructions.as_deref(),
            Self::Time(f) => f.instructions.as_deref(),
            Self::DateTime(f) => f.instructions.as_deref(),
            Self::Duration(f) => f.instructions.as_deref(),
            Self::List(f) => f.instructions.as_deref(),
            Self::Estimate(f) => f.instructions.as_deref(),
            Self::Forecast(f) => f.instructions.as_deref(),
        }
    }

    /// The bounds of an estimate or forecast field, `(min, max)`.
    #[must_use]
    pub fn bounds(&self) -> Option<(f64, f64)> {
        match self {
            Self::Estimate(f) => Some((f.min, f.max)),
            Self::Forecast(f) => Some((f.min, f.max)),
            _ => None,
        }
    }
}

macro_rules! field_from {
    ($($variant:ident($ty:ident)),+ $(,)?) => {
        $(impl From<$ty> for Field {
            fn from(field: $ty) -> Self {
                Self::$variant(field)
            }
        })+
    };
}

field_from!(
    Text(TextField),
    Number(NumberField),
    Date(DateField),
    Time(TimeField),
    DateTime(DateTimeField),
    Duration(DurationField),
    List(ListField),
    Estimate(EstimateField),
    Forecast(ForecastField),
);

/// The kind of a [`Field`] or [`Answer`](super::Answer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FieldKind {
    /// A text span.
    Text,
    /// A number.
    Number,
    /// A calendar date.
    Date,
    /// A time of day.
    Time,
    /// A date and time.
    #[serde(rename = "datetime")]
    DateTime,
    /// A duration.
    Duration,
    /// A list.
    List,
    /// An estimate.
    Estimate,
    /// A forecast.
    Forecast,
}

enum_names!(FieldKind, "field kind", {
    Text => "text",
    Number => "number",
    Date => "date",
    Time => "time",
    DateTime => "datetime",
    Duration => "duration",
    List => "list",
    Estimate => "estimate",
    Forecast => "forecast",
});

impl FieldKind {
    /// Every kind, in declaration order.
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
    ];
}

/// Asks for a span of the input.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TextField {
    /// What the field should contain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

/// Asks for a number, read or computed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct NumberField {
    /// What the field should contain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// Whether the answer is a whole number.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub integer: bool,
    /// Decimal places the answer is rounded to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decimals: Option<u8>,
    /// The unit the answer is in, as a label the model reads (`USD`, `kg`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

/// Asks for a calendar date.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DateField {
    /// What the field should contain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

/// Asks for a time of day.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TimeField {
    /// What the field should contain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

/// Asks for a date and time.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DateTimeField {
    /// What the field should contain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

/// Asks for a length of time in a unit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DurationField {
    /// What the field should contain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// The unit the answer is in.
    pub unit: DurationUnit,
    /// Whether the answer is a whole number of units.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub integer: bool,
    /// Decimal places the answer is rounded to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decimals: Option<u8>,
}

/// Asks for every matching value, in input order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ListField {
    /// What each item should contain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// The kind of each item.
    pub items: ListItemKind,
    /// Decimal places number items are rounded to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decimals: Option<u8>,
}

/// Asks for a distribution over a value, inside bounds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EstimateField {
    /// What is being estimated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// The least value the answer can take.
    pub min: f64,
    /// The greatest value the answer can take.
    pub max: f64,
    /// The grid the distribution is laid out on, in units of the value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<f64>,
    /// Whether the grid is linear or logarithmic between the bounds.
    #[serde(default, skip_serializing_if = "EstimateScale::is_linear")]
    pub scale: EstimateScale,
    /// The unit the value is in, as a label the model reads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

/// Asks for future values of a series, one estimate per step ahead.
///
/// The history is in the input, as the caller has it; the field declares
/// how many steps ahead to estimate, the length of a step, and the bounds
/// every step's estimate stays inside.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ForecastField {
    /// What is being forecast.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// How many steps ahead to estimate, at least 1.
    pub horizon: usize,
    /// The length of one step.
    #[serde(default, skip_serializing_if = "ForecastPeriod::is_step")]
    pub period: ForecastPeriod,
    /// The least value a step's estimate can take.
    pub min: f64,
    /// The greatest value a step's estimate can take.
    pub max: f64,
    /// Whether the grid is linear or logarithmic between the bounds.
    #[serde(default, skip_serializing_if = "EstimateScale::is_linear")]
    pub scale: EstimateScale,
}

/// The unit of a duration field's answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum DurationUnit {
    /// Seconds.
    Seconds,
    /// Minutes.
    Minutes,
    /// Hours.
    Hours,
    /// Days.
    Days,
    /// Weeks.
    Weeks,
    /// Calendar months.
    Months,
    /// Calendar years.
    Years,
}

enum_names!(DurationUnit, "duration unit", {
    Seconds => "seconds",
    Minutes => "minutes",
    Hours => "hours",
    Days => "days",
    Weeks => "weeks",
    Months => "months",
    Years => "years",
});

/// The kind of each item of a list field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ListItemKind {
    /// Spans of the input.
    Text,
    /// Numbers.
    Number,
    /// Calendar dates.
    Date,
}

enum_names!(ListItemKind, "list item kind", {
    Text => "text",
    Number => "number",
    Date => "date",
});

/// How an estimate's grid is spaced between its bounds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum EstimateScale {
    /// Evenly spaced.
    #[default]
    Linear,
    /// Spaced by ratio, for quantities that span orders of magnitude, such
    /// as prices. The bounds must be positive.
    Log,
}

enum_names!(EstimateScale, "estimate scale", {
    Linear => "linear",
    Log => "log",
});

impl EstimateScale {
    fn is_linear(&self) -> bool {
        *self == Self::Linear
    }
}

/// The length of one forecast step.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ForecastPeriod {
    /// One entry of the series, whatever it spans.
    #[default]
    Step,
    /// A minute.
    Minute,
    /// An hour.
    Hour,
    /// A day.
    Day,
    /// A week.
    Week,
    /// A calendar month.
    Month,
    /// A calendar quarter.
    Quarter,
    /// A calendar year.
    Year,
}

enum_names!(ForecastPeriod, "forecast period", {
    Step => "step",
    Minute => "minute",
    Hour => "hour",
    Day => "day",
    Week => "week",
    Month => "month",
    Quarter => "quarter",
    Year => "year",
});

impl ForecastPeriod {
    fn is_step(&self) -> bool {
        *self == Self::Step
    }
}

macro_rules! instructions_builder {
    ($ty:ident) => {
        impl $ty {
            /// Say what the field should contain.
            #[must_use]
            pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
                self.instructions = Some(instructions.into());
                self
            }
        }
    };
}

instructions_builder!(TextField);
instructions_builder!(NumberField);
instructions_builder!(DateField);
instructions_builder!(TimeField);
instructions_builder!(DateTimeField);
instructions_builder!(DurationField);
instructions_builder!(ListField);
instructions_builder!(EstimateField);
instructions_builder!(ForecastField);

impl TextField {
    /// A text field without instructions.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl NumberField {
    /// A number field without constraints.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Require a whole number.
    #[must_use]
    pub fn integer(mut self) -> Self {
        self.integer = true;
        self
    }

    /// Round the answer to `decimals` places.
    #[must_use]
    pub fn decimals(mut self, decimals: u8) -> Self {
        self.decimals = Some(decimals);
        self
    }

    /// Name the unit the answer is in.
    #[must_use]
    pub fn unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }
}

impl DateField {
    /// A date field without instructions.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl TimeField {
    /// A time field without instructions.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl DateTimeField {
    /// A datetime field without instructions.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl DurationField {
    /// A duration field answered in `unit`.
    #[must_use]
    pub fn new(unit: DurationUnit) -> Self {
        Self {
            instructions: None,
            unit,
            integer: false,
            decimals: None,
        }
    }

    /// Require a whole number of units.
    #[must_use]
    pub fn integer(mut self) -> Self {
        self.integer = true;
        self
    }

    /// Round the answer to `decimals` places.
    #[must_use]
    pub fn decimals(mut self, decimals: u8) -> Self {
        self.decimals = Some(decimals);
        self
    }
}

impl ListField {
    /// A list field of `items`.
    #[must_use]
    pub fn new(items: ListItemKind) -> Self {
        Self {
            instructions: None,
            items,
            decimals: None,
        }
    }

    /// Round number items to `decimals` places.
    #[must_use]
    pub fn decimals(mut self, decimals: u8) -> Self {
        self.decimals = Some(decimals);
        self
    }
}

impl EstimateField {
    /// An estimate between `min` and `max`, on a linear grid.
    #[must_use]
    pub fn new(min: f64, max: f64) -> Self {
        Self {
            instructions: None,
            min,
            max,
            step: None,
            scale: EstimateScale::Linear,
            unit: None,
        }
    }

    /// Lay the grid out in steps of `step`.
    #[must_use]
    pub fn step(mut self, step: f64) -> Self {
        self.step = Some(step);
        self
    }

    /// Space the grid logarithmically.
    #[must_use]
    pub fn log_scale(mut self) -> Self {
        self.scale = EstimateScale::Log;
        self
    }

    /// Set the grid's scale.
    #[must_use]
    pub fn scale(mut self, scale: EstimateScale) -> Self {
        self.scale = scale;
        self
    }

    /// Name the unit the value is in.
    #[must_use]
    pub fn unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }
}

impl ForecastField {
    /// A forecast of one step ahead, between `min` and `max`.
    #[must_use]
    pub fn new(min: f64, max: f64) -> Self {
        Self {
            instructions: None,
            horizon: 1,
            period: ForecastPeriod::Step,
            min,
            max,
            scale: EstimateScale::Linear,
        }
    }

    /// Estimate `horizon` steps ahead.
    #[must_use]
    pub fn horizon(mut self, horizon: usize) -> Self {
        self.horizon = horizon;
        self
    }

    /// Set the length of one step.
    #[must_use]
    pub fn period(mut self, period: ForecastPeriod) -> Self {
        self.period = period;
        self
    }

    /// Space the grid logarithmically.
    #[must_use]
    pub fn log_scale(mut self) -> Self {
        self.scale = EstimateScale::Log;
        self
    }

    /// Set the grid's scale.
    #[must_use]
    pub fn scale(mut self, scale: EstimateScale) -> Self {
        self.scale = scale;
        self
    }
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} field", self.kind())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn serializes_tagged_by_kind() {
        let field: Field = Field::number().decimals(2).unit("USD").into();
        assert_eq!(
            serde_json::to_value(&field).unwrap(),
            json!({"type": "number", "decimals": 2, "unit": "USD"})
        );
        let field: Field = Field::datetime().instructions("When").into();
        assert_eq!(
            serde_json::to_value(&field).unwrap(),
            json!({"type": "datetime", "instructions": "When"})
        );
        let field: Field = Field::estimate(1.0, 10.0).log_scale().step(0.5).into();
        assert_eq!(
            serde_json::to_value(&field).unwrap(),
            json!({"type": "estimate", "min": 1.0, "max": 10.0, "step": 0.5, "scale": "log"})
        );
        let field: Field = Field::forecast(0.0, 5.0)
            .horizon(3)
            .period(ForecastPeriod::Day)
            .into();
        assert_eq!(
            serde_json::to_value(&field).unwrap(),
            json!({"type": "forecast", "horizon": 3, "period": "day", "min": 0.0, "max": 5.0})
        );
        let back: Field = serde_json::from_value(serde_json::to_value(&field).unwrap()).unwrap();
        assert_eq!(back, field);
    }

    #[test]
    fn kinds_and_names() {
        let field: Field = Field::duration(DurationUnit::Days).integer().into();
        assert_eq!(field.kind(), FieldKind::Duration);
        assert_eq!(field.instructions(), None);
        assert_eq!(field.bounds(), None);
        assert_eq!(FieldKind::DateTime.to_string(), "datetime");
        assert_eq!(
            "datetime".parse::<FieldKind>().unwrap(),
            FieldKind::DateTime
        );
        assert_eq!(
            serde_json::to_value(FieldKind::DateTime).unwrap(),
            json!("datetime")
        );
        assert_eq!(
            Field::from(Field::estimate(0.0, 1.0)).bounds(),
            Some((0.0, 1.0))
        );
        assert_eq!(FieldKind::ALL.len(), 9);
    }
}
