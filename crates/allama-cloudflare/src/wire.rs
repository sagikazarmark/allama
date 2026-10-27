//! Clairic's wire format: the request body a Workers AI extraction model
//! runs with, and the output it returns.
//!
//! Field and answer order is the order the model reads and reports them
//! in, so neither goes through a map that sorts its keys.

use std::collections::BTreeMap;

use allama::adapter::Degradations;
use allama::extraction::{
    Answer, Date, DateAnswer, DateTime, DateTimeAnswer, DurationAnswer, EstimateAnswer,
    EstimateScale, Field, ForecastAnswer, ForecastPeriod, ListAnswer, ListItem, ListItemKind,
    NumberAnswer, TextAnswer, TextSpan, Time, TimeAnswer,
};
use allama::{
    Error, ExtraMap, ExtractionRequest, ExtractionResponse, MediaSource, Result, Subject, Usage,
    Warning, WarningKind,
};
use serde::ser::{SerializeMap, SerializeStruct};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;
use serde_json::value::RawValue;

use crate::options::ExtractionRequestOptions;

/// Fields of the body the codec owns: `extra` may not set them.
const RESERVED: &[&str] = &[
    "model",
    "state",
    "fields",
    "images",
    "reference_date",
    "options",
];

/// The request body: `model`, `state`, `fields`, and the rest when set.
#[derive(Debug)]
pub(crate) struct WireRequest<'a> {
    model: &'a str,
    request: &'a ExtractionRequest,
    images: Vec<WireImage<'a>>,
    null_thresholds: &'a BTreeMap<String, f64>,
    options: Option<WireOptions>,
    extra: &'a ExtraMap,
}

#[derive(Debug, Serialize)]
struct WireImage<'a> {
    content_type: &'a str,
    base64: &'a str,
}

#[derive(Debug, Serialize)]
struct WireOptions {
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    include_quantiles: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    return_state_text: bool,
}

/// Encode a validated `request` for `model`'s selector, reporting through
/// `d`.
///
/// # Errors
///
/// [`Error::InvalidRequest`] when `extra` sets a field the codec owns, or
/// a null threshold names a field the request does not have;
/// [`Error::Unsupported`] for an image that is not inline data.
pub(crate) fn encode<'a>(
    request: &'a ExtractionRequest,
    options: &'a ExtractionRequestOptions,
    d: &mut Degradations,
) -> Result<WireRequest<'a>> {
    if let Some(key) = options
        .extra
        .keys()
        .find(|key| RESERVED.iter().any(|owned| key.eq_ignore_ascii_case(owned)))
    {
        return Err(Error::invalid_request(format!(
            "`extra.{key}` is set by the adapter; use the request's own field"
        ))
        .authored());
    }
    check_schema(request)?;
    for name in options.null_thresholds.keys() {
        if request.find_field(name).is_none() {
            return Err(Error::invalid_request(format!(
                "null threshold for unknown field {name:?}"
            ))
            .authored());
        }
    }
    let images = encode_images(request, d)?;
    let wire_options =
        (options.include_quantiles || options.return_state_text).then_some(WireOptions {
            include_quantiles: options.include_quantiles,
            return_state_text: options.return_state_text,
        });
    Ok(WireRequest {
        model: selector(&request.model),
        request,
        images,
        null_thresholds: &options.null_thresholds,
        options: wire_options,
        extra: &options.extra,
    })
}

/// Clairic's input schema, checked before sending so that a request the
/// model would refuse never costs a call (ADR 0018, decision 10): field
/// names of at most 64 of `[A-Za-z0-9_.-]` starting with a letter or `_`,
/// `decimals` up to 6, images of at most 4 MiB each and 8 MiB together.
fn check_schema(request: &ExtractionRequest) -> Result<()> {
    for (name, field) in &request.fields {
        let mut chars = name.chars();
        let starts_well = chars
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
        let rest_well = name.len() <= 64
            && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'));
        if !(starts_well && rest_well) {
            return Err(Error::invalid_request(format!(
                "field name {name:?} is not one Workers AI accepts: at most 64 of `[A-Za-z0-9_.-]`, starting with a letter or `_`"
            ))
            .authored());
        }
        let decimals = match field {
            Field::Number(number) => number.decimals,
            Field::Duration(duration) => duration.decimals,
            Field::List(list) => list.decimals,
            _ => None,
        };
        if let Some(decimals) = decimals
            && decimals > MAX_DECIMALS
        {
            return Err(Error::invalid_request(format!(
                "field {name:?} asks for {decimals} decimals; Workers AI accepts at most {MAX_DECIMALS}"
            ))
            .authored());
        }
    }
    let mut total = 0;
    for (index, image) in request.images.iter().enumerate() {
        // Base64 encodes 3 bytes in 4 characters; close enough to the cap.
        let decoded = image.source.data().map_or(0, |data| data.len() / 4 * 3);
        if decoded > MAX_IMAGE_BYTES {
            return Err(Error::invalid_request(format!(
                "image {index} is about {} MiB decoded; Workers AI accepts at most 4 MiB per image",
                decoded >> 20
            ))
            .authored());
        }
        total += decoded;
    }
    if total > MAX_IMAGES_BYTES {
        return Err(Error::invalid_request(format!(
            "the images are about {} MiB decoded together; Workers AI accepts at most 8 MiB",
            total >> 20
        ))
        .authored());
    }
    Ok(())
}

const MAX_DECIMALS: u8 = 6;
const MAX_IMAGE_BYTES: usize = 4 << 20;
const MAX_IMAGES_BYTES: usize = 8 << 20;

/// The model selector a Workers AI extraction model reads from its input:
/// the last segment of its name (`clairic` for `@cf/cloudflare/clairic`).
pub(crate) fn selector(model: &str) -> &str {
    model.rsplit('/').next().unwrap_or(model)
}

fn encode_images<'a>(
    request: &'a ExtractionRequest,
    d: &mut Degradations,
) -> Result<Vec<WireImage<'a>>> {
    let mut images = Vec::with_capacity(request.images.len());
    for (index, image) in request.images.iter().enumerate() {
        let subject = || Subject::Other {
            name: format!("image {index}"),
        };
        let MediaSource::Base64 {
            media_type, data, ..
        } = &image.source
        else {
            return Err(d.reject(
                subject(),
                "Workers AI takes images as inline data only: send the image's bytes (`Image::bytes`), not a URL or a provider file",
            ));
        };
        if image.detail.is_some() {
            d.hint(
                subject(),
                "Workers AI has no image detail setting; the image was sent without it",
            )?;
        }
        images.push(WireImage {
            content_type: image.resolved_media_type()?.unwrap_or(media_type),
            base64: data.as_str(),
        });
    }
    Ok(images)
}

impl Serialize for WireRequest<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut body = serializer.serialize_map(None)?;
        body.serialize_entry("model", self.model)?;
        body.serialize_entry("state", &self.request.input)?;
        body.serialize_entry(
            "fields",
            &WireFields {
                fields: &self.request.fields,
                null_thresholds: self.null_thresholds,
            },
        )?;
        if !self.images.is_empty() {
            body.serialize_entry("images", &self.images)?;
        }
        if let Some(date) = &self.request.reference_date {
            body.serialize_entry("reference_date", date)?;
        }
        if let Some(options) = &self.options {
            body.serialize_entry("options", options)?;
        }
        for (key, value) in self.extra {
            body.serialize_entry(key, value)?;
        }
        body.end()
    }
}

struct WireFields<'a> {
    fields: &'a [(String, Field)],
    null_thresholds: &'a BTreeMap<String, f64>,
}

impl Serialize for WireFields<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.fields.len()))?;
        for (name, field) in self.fields {
            map.serialize_entry(
                name,
                &WireField {
                    field,
                    null_threshold: self.null_thresholds.get(name).copied(),
                },
            )?;
        }
        map.end()
    }
}

struct WireField<'a> {
    field: &'a Field,
    null_threshold: Option<f64>,
}

impl Serialize for WireField<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("Field", 8)?;
        s.serialize_field("type", self.field.kind().as_str())?;
        if let Some(instructions) = self.field.instructions() {
            s.serialize_field("instructions", instructions)?;
        }
        match self.field {
            Field::Number(number) => {
                if number.integer {
                    s.serialize_field("integer", &true)?;
                }
                if let Some(decimals) = number.decimals {
                    s.serialize_field("decimals", &decimals)?;
                }
                if let Some(unit) = &number.unit {
                    s.serialize_field("unit", unit)?;
                }
            }
            Field::Duration(duration) => {
                s.serialize_field("unit", duration.unit.as_str())?;
                if duration.integer {
                    s.serialize_field("integer", &true)?;
                }
                if let Some(decimals) = duration.decimals {
                    s.serialize_field("decimals", &decimals)?;
                }
            }
            Field::List(list) => {
                s.serialize_field("items", list.items.as_str())?;
                if let Some(decimals) = list.decimals {
                    s.serialize_field("decimals", &decimals)?;
                }
            }
            Field::Estimate(estimate) => {
                s.serialize_field("min", &estimate.min)?;
                s.serialize_field("max", &estimate.max)?;
                if let Some(step) = estimate.step {
                    s.serialize_field("step", &step)?;
                }
                if estimate.scale != EstimateScale::Linear {
                    s.serialize_field("scale", estimate.scale.as_str())?;
                }
                if let Some(unit) = &estimate.unit {
                    s.serialize_field("unit", unit)?;
                }
            }
            Field::Forecast(forecast) => {
                s.serialize_field("horizon", &forecast.horizon)?;
                if forecast.period != ForecastPeriod::Step {
                    s.serialize_field("interval", forecast.period.as_str())?;
                }
                s.serialize_field("min", &forecast.min)?;
                s.serialize_field("max", &forecast.max)?;
                if forecast.scale != EstimateScale::Linear {
                    s.serialize_field("scale", forecast.scale.as_str())?;
                }
            }
            _ => {}
        }
        if let Some(threshold) = self.null_threshold {
            s.serialize_field("null_threshold", &threshold)?;
        }
        s.end()
    }
}

/// Cloudflare's REST envelope: `{"result": .., "success": .., "errors":
/// [..], "messages": [..]}`. Its errors are read as an error envelope.
#[derive(Debug, Deserialize)]
pub(crate) struct Envelope {
    pub(crate) result: Box<RawValue>,
}

/// The model's output. Answers stay raw until their field is known, so a
/// malformed answer is reported by name.
#[derive(Debug, Deserialize)]
pub(crate) struct WireResponse {
    #[serde(default)]
    model: Option<String>,
    #[serde(deserialize_with = "deserialize_entries")]
    answers: Vec<(String, Box<RawValue>)>,
    #[serde(default)]
    truncated: bool,
    #[serde(default)]
    state_text: Option<String>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Debug, Deserialize)]
struct WireUsage {
    #[serde(default)]
    state_tokens: Option<u64>,
    #[serde(default)]
    image_tokens: Option<u64>,
    #[serde(default)]
    schema_tokens: Option<u64>,
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
    #[serde(default)]
    passes: Option<u64>,
}

/// One answer, with every key Clairic may report; which apply depends on
/// the field's kind.
#[derive(Debug, Default, Deserialize)]
struct WireAnswer {
    #[serde(rename = "type", default)]
    kind: Option<String>,
    #[serde(default)]
    value: Value,
    #[serde(default)]
    guess: Option<Value>,
    #[serde(default)]
    p_null: Option<f64>,
    #[serde(default)]
    confidence: Option<f64>,
    #[serde(default)]
    unit: Option<String>,
    #[serde(default)]
    span: Option<Vec<usize>>,
    #[serde(default)]
    spans: Option<Vec<Vec<usize>>>,
    #[serde(default)]
    mean: Option<Value>,
    #[serde(default)]
    median: Option<Value>,
    #[serde(default)]
    p10: Option<Value>,
    #[serde(default)]
    p90: Option<Value>,
    #[serde(default)]
    p_below: Option<Value>,
    #[serde(default)]
    p_above: Option<Value>,
    #[serde(default)]
    quantiles: Option<Vec<Value>>,
    #[serde(default)]
    steps: Option<Vec<WireAnswer>>,
    /// Keys this adapter does not model, kept as provider data.
    #[serde(flatten)]
    other: ExtraMap,
}

/// What [`decode`] produces: the response, and the provider's statistics
/// for provider data.
#[derive(Debug)]
pub(crate) struct Decoded {
    pub(crate) response: ExtractionResponse,
    pub(crate) statistics: Statistics,
}

/// The provider's per-answer statistics and the run's details.
#[derive(Debug)]
pub(crate) struct Statistics {
    pub(crate) confidence: BTreeMap<String, f64>,
    pub(crate) guesses: BTreeMap<String, Value>,
    pub(crate) distributions: BTreeMap<String, Vec<crate::Distribution>>,
    pub(crate) extras: BTreeMap<String, ExtraMap>,
    pub(crate) run: crate::RunDetails,
}

/// Decode the model's output for `request`, checked against it.
///
/// # Errors
///
/// [`Error::UnexpectedResponse`] when an answer is malformed, or the
/// response does not answer the request.
pub(crate) fn decode(body: WireResponse, request: &ExtractionRequest) -> Result<Decoded> {
    let mut confidence = BTreeMap::new();
    let mut guesses = BTreeMap::new();
    let mut distributions = BTreeMap::new();
    let mut extras = BTreeMap::new();
    let mut warnings = Vec::new();
    let mut answers = Vec::with_capacity(body.answers.len());
    for (name, raw) in body.answers {
        // A name the request has no field for is the model's text: it goes
        // in the upstream message, out of the redacted view.
        let Some(field) = request.find_field(&name) else {
            return Err(Error::unexpected_response(
                "extraction answers a field that was not asked",
            )
            .authored()
            .with_upstream_message(format!("field {name:?}")));
        };
        let wire: WireAnswer = serde_json::from_str(raw.get()).map_err(|err| {
            unexpected(&name, "is malformed").with_upstream_message(err.to_string())
        })?;
        if let Some(kind) = &wire.kind
            && kind != field.kind().as_str()
        {
            return Err(unexpected(
                &name,
                format!("is a {} field, answered as another kind", field.kind()),
            )
            .with_upstream_message(format!("kind {kind:?}")));
        }
        if let Some(value) = wire.confidence {
            confidence.insert(name.clone(), value);
        }
        if let Some(guess) = &wire.guess {
            guesses.insert(name.clone(), guess.clone());
        }
        if !wire.other.is_empty() {
            extras.insert(name.clone(), wire.other.clone());
        }
        let answer = decode_answer(
            &name,
            field,
            &wire,
            &request.input,
            body.state_text.as_deref(),
            &mut warnings,
        )?;
        match (field, &wire) {
            (Field::Estimate(_), wire) => {
                distributions.insert(name.clone(), vec![distribution(wire)]);
            }
            (Field::Forecast(_), wire) => {
                distributions.insert(name.clone(), forecast_distributions(wire));
            }
            _ => {}
        }
        answers.push((name, answer));
    }
    let mut response = ExtractionResponse::new(answers);
    response.model = body.model;
    let tokens = body.usage.as_ref().map(|usage| crate::TokenCounts {
        state_tokens: usage.state_tokens,
        image_tokens: usage.image_tokens,
        schema_tokens: usage.schema_tokens,
    });
    let passes = body.usage.as_ref().and_then(|usage| usage.passes);
    response.usage = body.usage.map(usage);
    if body.truncated {
        warnings.push(
            Warning::new(
                WarningKind::Approximated,
                Subject::Other {
                    name: "input".into(),
                },
                "the model truncated the input to fit its context; the fields were answered from the truncated input",
            )
            .with_code("approximated.input_truncated"),
        );
    }
    response.warnings = warnings;
    response.validate(request)?;
    Ok(Decoded {
        response,
        statistics: Statistics {
            confidence,
            guesses,
            distributions,
            extras,
            run: crate::RunDetails {
                truncated: body.truncated,
                passes,
                state_text: body.state_text,
                tokens,
            },
        },
    })
}

fn usage(wire: WireUsage) -> Usage {
    let mut usage = Usage::new();
    if let Some(input) = wire.input_tokens {
        usage = usage.input_tokens(input);
    }
    if let Some(output) = wire.output_tokens {
        usage = usage.output_tokens(output);
    }
    // Breakdowns must fit the contract: a subset of the input total.
    let fits = |part: Option<u64>| part.filter(|part| wire.input_tokens.is_none_or(|t| *part <= t));
    if let Some(tokens) = fits(wire.state_tokens) {
        usage = usage.breakdown("text_input_tokens", tokens);
    }
    if let Some(tokens) = fits(wire.image_tokens) {
        usage = usage.breakdown("image_input_tokens", tokens);
    }
    if let Some(tokens) = fits(wire.schema_tokens) {
        usage = usage.breakdown("schema_input_tokens", tokens);
    }
    usage
}

fn decode_answer(
    name: &str,
    field: &Field,
    wire: &WireAnswer,
    input: &str,
    state_text: Option<&str>,
    warnings: &mut Vec<Warning>,
) -> Result<Answer> {
    Ok(match field {
        Field::Text(_) => {
            let value = optional(name, &wire.value, Value::as_str)?.map(str::to_owned);
            let mut answer = TextAnswer::new(value.clone());
            if let Some(p) = wire.p_null {
                answer = answer.null_probability(p);
            }
            if let (Some(value), Some(span)) = (&value, &wire.span)
                && let Some(span) = place(name, span, value, input, state_text, warnings)
            {
                answer = answer.span(span);
            }
            answer.into()
        }
        Field::Number(_) => {
            let mut answer = NumberAnswer::new(optional(name, &wire.value, Value::as_f64)?);
            if let Some(p) = wire.p_null {
                answer = answer.null_probability(p);
            }
            answer.into()
        }
        Field::Date(_) => {
            let value = optional(name, &wire.value, Value::as_str)?
                .map(|text| {
                    Date::new(text).map_err(|err| {
                        unexpected(name, "has a value that is not a date")
                            .with_upstream_message(err.message())
                    })
                })
                .transpose()?;
            let mut answer = DateAnswer::new(value);
            if let Some(p) = wire.p_null {
                answer = answer.null_probability(p);
            }
            answer.into()
        }
        Field::Time(_) => {
            let value = optional(name, &wire.value, Value::as_str)?
                .map(|text| {
                    Time::new(text).map_err(|err| {
                        unexpected(name, "has a value that is not a time")
                            .with_upstream_message(err.message())
                    })
                })
                .transpose()?;
            let mut answer = TimeAnswer::new(value);
            if let Some(p) = wire.p_null {
                answer = answer.null_probability(p);
            }
            answer.into()
        }
        Field::DateTime(_) => {
            let value = optional(name, &wire.value, Value::as_str)?
                .map(|text| {
                    DateTime::new(text).map_err(|err| {
                        unexpected(name, "has a value that is not a datetime")
                            .with_upstream_message(err.message())
                    })
                })
                .transpose()?;
            let mut answer = DateTimeAnswer::new(value);
            if let Some(p) = wire.p_null {
                answer = answer.null_probability(p);
            }
            answer.into()
        }
        Field::Duration(duration) => {
            if let Some(unit) = &wire.unit
                && unit != duration.unit.as_str()
            {
                return Err(unexpected(
                    name,
                    format!("is asked in {}, answered in another unit", duration.unit),
                )
                .with_upstream_message(format!("unit {unit:?}")));
            }
            let mut answer =
                DurationAnswer::new(optional(name, &wire.value, Value::as_f64)?, duration.unit);
            if let Some(p) = wire.p_null {
                answer = answer.null_probability(p);
            }
            answer.into()
        }
        Field::List(list) => {
            let values = match &wire.value {
                Value::Null => Vec::new(),
                Value::Array(values) => values.clone(),
                _ => return Err(unexpected(name, "lists something other than an array")),
            };
            let mut items = Vec::with_capacity(values.len());
            for (index, value) in values.iter().enumerate() {
                let mut item = match list.items {
                    ListItemKind::Text => ListItem::text(
                        value
                            .as_str()
                            .ok_or_else(|| unexpected(name, format!("item {index} is not text")))?,
                    ),
                    ListItemKind::Number => ListItem::number(value.as_f64().ok_or_else(|| {
                        unexpected(name, format!("item {index} is not a number"))
                    })?),
                    ListItemKind::Date => ListItem::date(
                        value
                            .as_str()
                            .ok_or_else(|| unexpected(name, format!("item {index} is not a date")))
                            .and_then(|text| {
                                Date::new(text).map_err(|err| {
                                    unexpected(name, format!("item {index} is not a date"))
                                        .with_upstream_message(err.message())
                                })
                            })?,
                    ),
                    _ => {
                        return Err(unexpected(
                            name,
                            "lists items of a kind this adapter does not know",
                        ));
                    }
                };
                if let Some(span) = wire.spans.as_ref().and_then(|spans| spans.get(index)) {
                    let placed = match value.as_str().filter(|_| list.items == ListItemKind::Text) {
                        Some(text) => place(name, span, text, input, state_text, warnings),
                        None => place_unverified(name, span, input, state_text, warnings),
                    };
                    if let Some(span) = placed {
                        item = item.span(span);
                    }
                }
                items.push(item);
            }
            ListAnswer::new(items).into()
        }
        Field::Estimate(_) => estimate(name, wire)?.into(),
        Field::Forecast(_) => {
            let steps = match &wire.steps {
                Some(steps) => steps
                    .iter()
                    .map(|step| estimate(name, step))
                    .collect::<Result<Vec<_>>>()?,
                // The blog's shape: parallel arrays, one entry per step.
                None => {
                    let values = wire.value.as_array().ok_or_else(|| {
                        unexpected(name, "forecasts neither `steps` nor an array of values")
                    })?;
                    let at = |series: &Option<Value>, index: usize| {
                        series
                            .as_ref()
                            .and_then(Value::as_array)
                            .and_then(|array| array.get(index))
                            .and_then(Value::as_f64)
                    };
                    values
                        .iter()
                        .enumerate()
                        .map(|(index, value)| {
                            let value = value.as_f64().ok_or_else(|| {
                                unexpected(name, format!("step {index} is not a number"))
                            })?;
                            let mut step = EstimateAnswer::new(value);
                            if let Some(median) = at(&wire.median, index) {
                                step = step.median(median);
                            }
                            if let (Some(low), Some(high)) =
                                (at(&wire.p10, index), at(&wire.p90, index))
                            {
                                step = step.interval(low, high, 0.8);
                            }
                            Ok(step)
                        })
                        .collect::<Result<Vec<_>>>()?
                }
            };
            ForecastAnswer::new(steps).into()
        }
        _ => return Err(unexpected(name, "has a kind this adapter does not know")),
    })
}

fn estimate(name: &str, wire: &WireAnswer) -> Result<EstimateAnswer> {
    let value = wire
        .value
        .as_f64()
        .ok_or_else(|| unexpected(name, "estimates something other than a number"))?;
    let mut answer = EstimateAnswer::new(value);
    if let Some(median) = wire.median.as_ref().and_then(Value::as_f64) {
        answer = answer.median(median);
    }
    if let (Some(low), Some(high)) = (
        wire.p10.as_ref().and_then(Value::as_f64),
        wire.p90.as_ref().and_then(Value::as_f64),
    ) {
        answer = answer.interval(low, high, 0.8);
    }
    Ok(answer)
}

fn distribution(wire: &WireAnswer) -> crate::Distribution {
    crate::Distribution {
        mean: wire.mean.as_ref().and_then(Value::as_f64),
        p_below: wire.p_below.as_ref().and_then(Value::as_f64),
        p_above: wire.p_above.as_ref().and_then(Value::as_f64),
        quantiles: wire
            .quantiles
            .as_ref()
            .map(|quantiles| quantiles.iter().filter_map(Value::as_f64).collect()),
    }
}

fn forecast_distributions(wire: &WireAnswer) -> Vec<crate::Distribution> {
    match &wire.steps {
        Some(steps) => steps.iter().map(distribution).collect(),
        None => {
            let series = |series: &Option<Value>| -> Vec<Option<f64>> {
                series
                    .as_ref()
                    .and_then(Value::as_array)
                    .map(|array| array.iter().map(Value::as_f64).collect())
                    .unwrap_or_default()
            };
            let (mean, below, above) = (
                series(&wire.mean),
                series(&wire.p_below),
                series(&wire.p_above),
            );
            let steps = wire.value.as_array().map_or(0, Vec::len);
            (0..steps)
                .map(|index| crate::Distribution {
                    mean: mean.get(index).copied().flatten(),
                    p_below: below.get(index).copied().flatten(),
                    p_above: above.get(index).copied().flatten(),
                    quantiles: None,
                })
                .collect()
        }
    }
}

/// A scalar `value`, `None` when null, read with `as`.
fn optional<'v, T>(
    name: &str,
    value: &'v Value,
    as_type: impl FnOnce(&'v Value) -> Option<T>,
) -> Result<Option<T>> {
    match value {
        Value::Null => Ok(None),
        other => as_type(other)
            .map(Some)
            .ok_or_else(|| unexpected(name, "has a value of the wrong type")),
    }
}

/// Place the model's span for `value` into `input` (ADR 0018, decision 7).
///
/// The offsets are read against the model's copy of the input when it
/// returned one (`state_text`), else against `input`; as `char` offsets,
/// as byte offsets, or as UTF-16 units, whichever makes the span equal the
/// value; else the value's unique occurrence in `input`, or the one nearest
/// the reported offset; else `None`, with a warning.
fn place(
    name: &str,
    span: &[usize],
    value: &str,
    input: &str,
    state_text: Option<&str>,
    warnings: &mut Vec<Warning>,
) -> Option<TextSpan> {
    let (start, end) = offsets(name, span, warnings)?;
    // The text the offsets index: the model's copy when it differs.
    let indexed = state_text.filter(|text| *text != input).unwrap_or(input);
    let indexed_is_input = std::ptr::eq(indexed, input);
    let candidates = [
        char_offsets_to_bytes(indexed, start, end),
        Some(TextSpan::new(start, end)),
        utf16_offsets_to_bytes(indexed, start, end),
    ];
    let placed = candidates
        .into_iter()
        .flatten()
        .find(|candidate| candidate.text(indexed) == Some(value));
    if let Some(span) = placed
        && indexed_is_input
    {
        return Some(span);
    }
    // Search `input`: a unique occurrence, or the one nearest the reported
    // start (in `input`'s bytes when the offsets were its own, else scaled
    // from the model's copy).
    let reported = match placed {
        Some(span) if !indexed.is_empty() => scale(span.start, input.len(), indexed.len()),
        _ => char_offsets_to_bytes(input, start, start).map_or(start, |span| span.start),
    };
    let mut best: Option<TextSpan> = None;
    for (at, _) in input.match_indices(value) {
        let candidate = TextSpan::new(at, at + value.len());
        best = match best {
            Some(found) if found.start.abs_diff(reported) <= at.abs_diff(reported) => Some(found),
            _ => Some(candidate),
        };
    }
    if best.is_none() {
        warnings.push(span_warning(name, "does not match the value in the input"));
    }
    best
}

/// Place the model's span for a value that has no text to verify against
/// (a number or date list item). The offsets are read in the unit the
/// model documents, `char`s, and kept when they land in `input` (ADR 0018,
/// decision 7: a span on a char boundary, with no value to compare).
/// Offsets into the model's own copy of the input cannot be carried over
/// without a text to find, so they are dropped with a warning.
fn place_unverified(
    name: &str,
    span: &[usize],
    input: &str,
    state_text: Option<&str>,
    warnings: &mut Vec<Warning>,
) -> Option<TextSpan> {
    let (start, end) = offsets(name, span, warnings)?;
    if state_text.is_some_and(|text| text != input) {
        warnings.push(span_warning(
            name,
            "indexes the model's copy of the input, which differs from the request's",
        ));
        return None;
    }
    let placed = char_offsets_to_bytes(input, start, end);
    if placed.is_none() {
        warnings.push(span_warning(name, "is outside the input"));
    }
    placed
}

/// The `[start, end]` pair of a reported span, or `None` with a warning.
fn offsets(name: &str, span: &[usize], warnings: &mut Vec<Warning>) -> Option<(usize, usize)> {
    match span {
        [start, end] if end >= start => Some((*start, *end)),
        _ => {
            warnings.push(span_warning(name, "is not a pair of offsets"));
            None
        }
    }
}

/// `offset * to / from`, without overflowing on 32-bit targets (a Worker
/// is `wasm32`).
fn scale(offset: usize, to: usize, from: usize) -> usize {
    usize::try_from(offset as u128 * to as u128 / from as u128).unwrap_or(usize::MAX)
}

fn span_warning(name: &str, problem: &str) -> Warning {
    Warning::new(
        WarningKind::Dropped,
        Subject::Other {
            name: format!("span of field {name:?}"),
        },
        format!("the span the model reported for field {name:?} {problem}; it was dropped"),
    )
    .with_code("dropped.span")
}

/// Byte span of the `char`s `start..end` of `input`.
fn char_offsets_to_bytes(input: &str, start: usize, end: usize) -> Option<TextSpan> {
    let byte_at = |chars: usize| -> Option<usize> {
        let mut count = 0;
        for (byte, _) in input.char_indices() {
            if count == chars {
                return Some(byte);
            }
            count += 1;
        }
        (count == chars).then_some(input.len())
    };
    Some(TextSpan::new(byte_at(start)?, byte_at(end)?))
}

/// Byte span of the UTF-16 units `start..end` of `input`.
fn utf16_offsets_to_bytes(input: &str, start: usize, end: usize) -> Option<TextSpan> {
    let byte_at = |units: usize| -> Option<usize> {
        let mut count = 0;
        for (byte, ch) in input.char_indices() {
            if count == units {
                return Some(byte);
            }
            if count > units {
                // Inside a surrogate pair.
                return None;
            }
            count += ch.len_utf16();
        }
        (count == units).then_some(input.len())
    };
    Some(TextSpan::new(byte_at(start)?, byte_at(end)?))
}

/// An authored `UnexpectedResponse` about the request's field `name`:
/// `problem` is this crate's text, and `name` the caller's (a response key
/// is matched to a request field before it is named).
fn unexpected(name: &str, problem: impl std::fmt::Display) -> Error {
    Error::unexpected_response(format!("extraction field {name:?} {problem}")).authored()
}

/// Deserialize a JSON object as an ordered list of raw entries.
fn deserialize_entries<'de, D>(
    deserializer: D,
) -> std::result::Result<Vec<(String, Box<RawValue>)>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct Entries;

    impl<'de> serde::de::Visitor<'de> for Entries {
        type Value = Vec<(String, Box<RawValue>)>;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("an object of answers")
        }

        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut access: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut entries = Vec::with_capacity(access.size_hint().unwrap_or(0).min(4096));
            while let Some(entry) = access.next_entry()? {
                entries.push(entry);
            }
            Ok(entries)
        }
    }

    deserializer.deserialize_map(Entries)
}

#[cfg(test)]
mod tests {
    use allama::extraction::{DurationUnit, Field, ListItemKind};
    use allama::{Image, ImageDetail, MessageOrigin, UnsupportedPolicy};
    use serde_json::json;

    use super::*;

    fn degradations() -> Degradations {
        Degradations::new("cloudflare".to_owned(), UnsupportedPolicy::Warn)
    }

    fn request() -> ExtractionRequest {
        ExtractionRequest::new(
            "@cf/cloudflare/clairic",
            "Hi, order A-88412 arrived on Sept 28, 2026 with a cracked screen and a dented corner. I paid $649.00 plus $52.10 tax.",
        )
        .field("order_id", Field::text().instructions("Order number"))
        .field("delivered_on", Field::date().instructions("Delivery date"))
        .field(
            "amount_paid",
            Field::number().decimals(2).unit("USD").instructions("Total paid including tax"),
        )
        .field(
            "days_since_delivery",
            Field::duration(DurationUnit::Days).integer(),
        )
        .field("damage", Field::list(ListItemKind::Text))
        .field("tracking_number", Field::text())
        .field("repair_cost", Field::estimate(10.0, 5000.0).log_scale().step(1.0).unit("USD"))
        .field("travelers", Field::forecast(500_000.0, 4_000_000.0).horizon(2).period(ForecastPeriod::Day))
    }

    #[test]
    fn encodes_the_documented_request_shape() {
        let request = request()
            .image(Image::base64("image/png", "aGk="))
            .reference_date("2026-10-09".parse().unwrap());
        let options = ExtractionRequestOptions::new()
            .include_quantiles()
            .null_threshold("order_id", 0.3);
        let body =
            serde_json::to_value(encode(&request, &options, &mut degradations()).unwrap()).unwrap();
        assert_eq!(body["model"], "clairic");
        assert_eq!(body["state"], request.input);
        assert_eq!(
            body["fields"]["order_id"],
            json!({"type": "text", "instructions": "Order number", "null_threshold": 0.3})
        );
        assert_eq!(
            body["fields"]["amount_paid"],
            json!({"type": "number", "instructions": "Total paid including tax", "decimals": 2, "unit": "USD"})
        );
        assert_eq!(
            body["fields"]["days_since_delivery"],
            json!({"type": "duration", "unit": "days", "integer": true})
        );
        assert_eq!(
            body["fields"]["damage"],
            json!({"type": "list", "items": "text"})
        );
        assert_eq!(
            body["fields"]["repair_cost"],
            json!({"type": "estimate", "min": 10.0, "max": 5000.0, "step": 1.0, "scale": "log", "unit": "USD"})
        );
        assert_eq!(
            body["fields"]["travelers"],
            json!({"type": "forecast", "horizon": 2, "interval": "day", "min": 500000.0, "max": 4000000.0})
        );
        assert_eq!(
            body["images"],
            json!([{"content_type": "image/png", "base64": "aGk="}])
        );
        assert_eq!(body["reference_date"], "2026-10-09");
        assert_eq!(body["options"], json!({"include_quantiles": true}));
        // Field order is the request's.
        let text = serde_json::to_string(&encode(&request, &options, &mut degradations()).unwrap())
            .unwrap();
        assert!(text.find("order_id").unwrap() < text.find("travelers").unwrap());
    }

    #[test]
    fn refuses_what_it_owns_and_unknown_thresholds() {
        let options = ExtractionRequestOptions::new().extra("state", json!("x"));
        let err = encode(&request(), &options, &mut degradations()).unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
        let options = ExtractionRequestOptions::new().null_threshold("nope", 0.5);
        let err = encode(&request(), &options, &mut degradations()).unwrap_err();
        assert!(err.message().contains("unknown field"), "{err}");
        let url = request().image(Image::url("https://example.com/a.png"));
        let err = encode(
            &url,
            &ExtractionRequestOptions::default(),
            &mut degradations(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
        let detail =
            request().image(Image::base64("image/png", "aGk=").with_detail(ImageDetail::High));
        let mut d = degradations();
        encode(&detail, &ExtractionRequestOptions::default(), &mut d).unwrap();
        assert_eq!(d.warnings().len(), 1);
    }

    /// The blog post's example response for the support email.
    fn blog_response() -> WireResponse {
        serde_json::from_value(json!({
            "model": "clairic",
            "answers": {
                "order_id": {"type": "text", "p_null": 3.4e-6, "span": [10, 17], "guess": "A-88412", "confidence": 0.999957, "value": "A-88412"},
                "delivered_on": {"type": "date", "p_null": 6.5e-8, "guess": "2026-09-28", "confidence": 0.999999, "value": "2026-09-28"},
                "amount_paid": {"type": "number", "p_null": 1.6e-5, "guess": 701.1, "confidence": 0.999944, "value": 701.1},
                "days_since_delivery": {"type": "duration", "p_null": 6.1e-9, "unit": "days", "guess": 11, "confidence": 0.999999, "value": 11},
                "damage": {"type": "list", "value": ["cracked screen", "dented corner"], "spans": [[50, 64], [71, 84]]},
                "tracking_number": {"type": "text", "p_null": 1, "span": [106, 112], "guess": "$52.10", "confidence": 0.416016, "value": null},
                "repair_cost": {"type": "estimate", "value": 180, "mean": 190.5, "median": 175, "p10": 120, "p90": 260, "p_below": 0.0, "p_above": 0.0},
                "travelers": {"type": "forecast", "value": [2416117, 2400000], "median": [2419283, 2410000], "p10": [2270774, 2200000], "p90": [2556769, 2600000]}
            },
            "truncated": false,
            "usage": {"state_tokens": 61, "image_tokens": 0, "schema_tokens": 472, "input_tokens": 655, "output_tokens": 0, "passes": 3}
        }))
        .unwrap()
    }

    #[test]
    fn decodes_the_documented_response() {
        let request = request();
        let decoded = decode(blog_response(), &request).unwrap();
        let response = &decoded.response;
        let order_id = response.text("order_id").unwrap();
        assert_eq!(order_id.value.as_deref(), Some("A-88412"));
        assert_eq!(order_id.null_probability, Some(3.4e-6));
        // The blog's offsets are for its own text; here the value is found.
        assert_eq!(order_id.span.unwrap().text(&request.input), Some("A-88412"));
        assert_eq!(
            response
                .date("delivered_on")
                .unwrap()
                .value
                .as_ref()
                .unwrap()
                .as_str(),
            "2026-09-28"
        );
        assert_eq!(response.number("amount_paid").unwrap().value, Some(701.1));
        let days = response.duration("days_since_delivery").unwrap();
        assert_eq!((days.value, days.unit), (Some(11.0), DurationUnit::Days));
        let damage = response.list("damage").unwrap();
        assert_eq!(
            damage.texts().collect::<Vec<_>>(),
            ["cracked screen", "dented corner"]
        );
        assert_eq!(
            damage.items[0].span.unwrap().text(&request.input),
            Some("cracked screen")
        );
        let tracking = response.text("tracking_number").unwrap();
        assert!(tracking.is_null());
        assert_eq!(tracking.span, None, "a null carries no span");
        let repair = response.estimate("repair_cost").unwrap();
        assert_eq!((repair.value, repair.median), (180.0, Some(175.0)));
        assert_eq!(
            repair.interval.unwrap(),
            allama::extraction::PredictionInterval::new(120.0, 260.0, 0.8)
        );
        let travelers = response.forecast("travelers").unwrap();
        assert_eq!(
            travelers.values().collect::<Vec<_>>(),
            [2_416_117.0, 2_400_000.0]
        );
        assert_eq!(travelers.steps[1].interval.unwrap().high, 2_600_000.0);
        assert_eq!(response.model.as_deref(), Some("clairic"));
        let usage = response.usage.as_ref().unwrap();
        assert_eq!(usage.input_tokens, Some(655));
        assert_eq!(usage.output_tokens, Some(0));
        assert_eq!(usage.breakdowns.get("schema_input_tokens"), Some(&472));
        assert_eq!(usage.breakdowns.get("text_input_tokens"), Some(&61));
        assert!(usage.contract_violation().is_none(), "{usage:?}");
        assert_eq!(decoded.statistics.confidence["order_id"], 0.999957);
        assert_eq!(
            decoded.statistics.guesses["tracking_number"],
            json!("$52.10")
        );
        assert_eq!(
            decoded.statistics.distributions["repair_cost"][0].mean,
            Some(190.5)
        );
        assert_eq!(decoded.statistics.distributions["travelers"].len(), 2);
        assert_eq!(decoded.statistics.run.passes, Some(3));
        assert!(!decoded.statistics.run.truncated);
        assert_eq!(
            decoded
                .statistics
                .run
                .tokens
                .as_ref()
                .unwrap()
                .schema_tokens,
            Some(472)
        );
        assert!(
            decoded.statistics.extras.is_empty(),
            "{:?}",
            decoded.statistics.extras
        );
        assert!(response.warnings.is_empty(), "{:?}", response.warnings);
    }

    #[test]
    fn unmodelled_answer_keys_are_kept() {
        let request = ExtractionRequest::new("m", "a receipt").field("t", Field::text());
        let body: WireResponse = serde_json::from_value(json!({
            "answers": {"t": {"type": "text", "value": "receipt", "span": [2, 9], "source": "image", "spell_confidence": 0.5}}
        }))
        .unwrap();
        let decoded = decode(body, &request).unwrap();
        assert_eq!(decoded.statistics.extras["t"]["source"], "image");
        assert_eq!(decoded.statistics.extras["t"].len(), 2);
    }

    /// A number or date item has no text to verify a span against: its
    /// offsets are kept as `char`s into the input, and dropped when they
    /// index the model's own copy of the input or fall outside it. A null
    /// list is an empty one.
    #[test]
    fn number_and_date_items_keep_unverified_spans() {
        let input = "Paid €52.10 on 2026-09-28 and €7 on Oct 1.";
        let request = ExtractionRequest::new("m", input)
            .field("amounts", Field::list(ListItemKind::Number))
            .field("dates", Field::list(ListItemKind::Date))
            .field("none", Field::list(ListItemKind::Text));
        let euros = input.find("€52.10").unwrap();
        let chars = |byte: usize| input[..byte].chars().count();
        let body = |state_text: Option<&str>| -> WireResponse {
            serde_json::from_value(json!({
                "answers": {
                    "amounts": {"type": "list", "value": [52.1, 7], "spans": [[chars(euros), chars(euros) + 6], [900, 901]]},
                    "dates": {"type": "list", "value": ["2026-09-28", "2026-10-01"], "spans": [[15, 25], [36, 41]]},
                    "none": {"type": "list", "value": null},
                },
                "state_text": state_text,
            }))
            .unwrap()
        };
        let decoded = decode(body(None), &request).unwrap();
        let amounts = decoded.response.list("amounts").unwrap();
        assert_eq!(amounts.items[0].span.unwrap().text(input), Some("€52.10"));
        assert_eq!(amounts.items[1].span, None, "outside the input");
        let dates = decoded.response.list("dates").unwrap();
        assert_eq!(dates.items[0].span.unwrap().text(input), Some("2026-09-28"));
        assert_eq!(dates.items[1].span.unwrap().text(input), Some("Oct 1"));
        assert!(decoded.response.list("none").unwrap().items.is_empty());
        let codes: Vec<_> = decoded
            .response
            .warnings
            .iter()
            .map(|w| w.code.as_deref())
            .collect();
        assert_eq!(codes, [Some("dropped.span")]);
        // The model's own copy of the input: nothing to find the items by.
        let decoded = decode(body(Some("paid 52.10 on 2026-09-28")), &request).unwrap();
        let spans: Vec<_> = ["amounts", "dates"]
            .into_iter()
            .flat_map(|name| decoded.response.list(name).unwrap().items.iter())
            .map(|item| item.span)
            .collect();
        assert_eq!(spans, [None; 4]);
        assert_eq!(decoded.response.warnings.len(), 4);
    }

    #[test]
    fn forecast_steps_shape_is_read_too() {
        let request =
            ExtractionRequest::new("m", "series").field("f", Field::forecast(0.0, 10.0).horizon(2));
        let body: WireResponse = serde_json::from_value(json!({
            "model": "clairic",
            "answers": {"f": {"type": "forecast", "value": [1, 2], "steps": [
                {"value": 1, "median": 1.5, "p10": 0.5, "p90": 2.5, "mean": 1.2},
                {"value": 2, "median": 2.5, "p10": 1.5, "p90": 3.5, "mean": 2.2}
            ]}},
            "truncated": true,
            "usage": {"input_tokens": 10, "output_tokens": 0}
        }))
        .unwrap();
        let decoded = decode(body, &request).unwrap();
        let steps = &decoded.response.forecast("f").unwrap().steps;
        assert_eq!(steps[1].median, Some(2.5));
        assert_eq!(decoded.statistics.distributions["f"][1].mean, Some(2.2));
        assert!(decoded.statistics.run.truncated);
        assert_eq!(
            decoded.response.warnings[0].code.as_deref(),
            Some("approximated.input_truncated")
        );
    }

    #[test]
    fn spans_are_placed_whatever_the_offset_unit() {
        // `é` is 2 bytes and 1 char; `𝄞` is 4 bytes, 1 char, 2 UTF-16 units.
        let input = "café 𝄞 order A-1 and A-1 again";
        let mut warnings = Vec::new();
        // Char offsets.
        let char_start = input[..input.find("A-1").unwrap()].chars().count();
        let span = place(
            "f",
            &[char_start, char_start + 3],
            "A-1",
            input,
            None,
            &mut warnings,
        )
        .unwrap();
        assert_eq!(span.text(input), Some("A-1"));
        assert_eq!(span.start, input.find("A-1").unwrap());
        // Byte offsets.
        let byte_start = input.find("A-1").unwrap();
        let span = place(
            "f",
            &[byte_start, byte_start + 3],
            "A-1",
            input,
            None,
            &mut warnings,
        )
        .unwrap();
        assert_eq!(span.start, byte_start);
        // UTF-16 units: one more than chars because of 𝄞.
        let span = place(
            "f",
            &[char_start + 1, char_start + 4],
            "A-1",
            input,
            None,
            &mut warnings,
        )
        .unwrap();
        assert_eq!(span.text(input), Some("A-1"));
        // Offsets that match nothing: the nearest occurrence to the reported start.
        let second = input.rfind("A-1").unwrap();
        let near_second = input[..second].chars().count() + 1;
        let span = place(
            "f",
            &[near_second, near_second + 2],
            "A-1",
            input,
            None,
            &mut warnings,
        )
        .unwrap();
        assert_eq!(span.start, second);
        assert!(warnings.is_empty(), "{warnings:?}");
        // A value absent from the input: dropped with a warning.
        assert_eq!(place("f", &[0, 3], "zzz", input, None, &mut warnings), None);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].code.as_deref(), Some("dropped.span"));
        // Not a pair.
        assert_eq!(place("f", &[1], "A-1", input, None, &mut warnings), None);
        assert_eq!(warnings.len(), 2);
        // The model's own copy of the input: offsets index it, the span is
        // then found in the caller's input.
        let normalised = "cafe  order A-1 and A-1 again";
        let at = normalised.find("A-1").unwrap();
        let span = place(
            "f",
            &[at, at + 3],
            "A-1",
            input,
            Some(normalised),
            &mut warnings,
        )
        .unwrap();
        assert_eq!(span.start, input.find("A-1").unwrap());
        assert_eq!(warnings.len(), 2);
    }

    #[test]
    fn clairics_schema_is_checked_before_sending() {
        let options = ExtractionRequestOptions::default();
        let long = "x".repeat(65);
        let bad_names = ["1st", "with space", long.as_str(), "a/b"];
        for name in bad_names {
            let request = ExtractionRequest::new("m", "x").field(name, Field::text());
            let err = encode(&request, &options, &mut degradations()).unwrap_err();
            assert!(
                matches!(err, Error::InvalidRequest { .. }),
                "{name}: {err:?}"
            );
        }
        ExtractionRequest::new("m", "x")
            .field("_ok.name-1", Field::text())
            .validate(&Default::default())
            .unwrap();
        let request = ExtractionRequest::new("m", "x").field("n", Field::number().decimals(7));
        let err = encode(&request, &options, &mut degradations()).unwrap_err();
        assert!(err.message().contains("decimals"), "{err}");
        let big = "A".repeat((4 << 20) / 3 * 4 + 8);
        let request = ExtractionRequest::new("m", "x")
            .image(Image::base64("image/png", big))
            .field("n", Field::number());
        let err = encode(&request, &options, &mut degradations()).unwrap_err();
        assert!(err.message().contains("4 MiB"), "{err}");
    }

    #[test]
    fn every_reserved_key_is_refused_whatever_its_case() {
        for key in RESERVED {
            for key in [key.to_string(), key.to_ascii_uppercase()] {
                let options = ExtractionRequestOptions::new().extra(key.clone(), json!(1));
                let err = encode(&request(), &options, &mut degradations()).unwrap_err();
                assert!(
                    matches!(err, Error::InvalidRequest { .. }),
                    "{key}: {err:?}"
                );
            }
        }
    }

    /// A field added to the body must be reserved too, or `extra` could
    /// send it twice (ADR 0006).
    #[test]
    fn every_body_field_is_reserved() {
        let request = request()
            .image(Image::bytes("image/png", b"\x89PNG"))
            .reference_date("2026-10-09".parse().unwrap());
        let options = ExtractionRequestOptions::new()
            .null_threshold("order_id", 0.3)
            .include_quantiles()
            .return_state_text();
        let body = encode(&request, &options, &mut degradations()).unwrap();
        let body = serde_json::to_value(body).unwrap();
        for key in body.as_object().unwrap().keys() {
            assert!(RESERVED.contains(&key.as_str()), "`{key}` is not reserved");
        }
    }

    #[test]
    fn malformed_answers_are_named() {
        let request = ExtractionRequest::new("m", "x").field("n", Field::number());
        let body: WireResponse = serde_json::from_value(json!({
            "answers": {"n": {"type": "number", "value": "seven"}}
        }))
        .unwrap();
        let err = decode(body, &request).unwrap_err();
        assert!(matches!(err, Error::UnexpectedResponse { .. }), "{err:?}");
        assert!(err.message().contains("\"n\""), "{err}");
        let body: WireResponse = serde_json::from_value(json!({
            "answers": {"n": {"type": "text", "value": "seven"}}
        }))
        .unwrap();
        let err = decode(body, &request).unwrap_err();
        // The response's text is the upstream message, out of the authored
        // one.
        assert!(err.message().contains("answered as another kind"), "{err}");
        assert!(!err.message().contains("text"), "{err}");
        assert_eq!(err.message_origin(), MessageOrigin::Authored);
        assert_eq!(err.upstream_message(), Some("kind \"text\""));
        let body: WireResponse = serde_json::from_value(json!({
            "answers": {"other": {"type": "number", "value": 1}}
        }))
        .unwrap();
        let err = decode(body, &request).unwrap_err();
        assert!(err.message().contains("was not asked"), "{err}");
        assert!(!err.message().contains("other"), "{err}");
        assert_eq!(err.message_origin(), MessageOrigin::Authored);
        assert_eq!(err.upstream_message(), Some("field \"other\""));
    }

    #[test]
    fn selector_is_the_last_segment() {
        assert_eq!(selector("@cf/cloudflare/clairic"), "clairic");
        assert_eq!(selector("clairic"), "clairic");
    }
}
