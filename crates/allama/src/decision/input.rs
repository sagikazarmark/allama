use std::fmt;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::json::Json;
use crate::macros::text_conversions;
use crate::{Error, Result};

/// What a decision model reads before answering: text, or a JSON object or
/// array.
///
/// JSON objects keep the key order the caller produced: the model reads
/// fields in that order. Their keys must be unique. The serde form is the
/// bare text or JSON, readable from any self-describing format.
///
/// ```
/// use allama::decision::DecisionInput;
///
/// let text = DecisionInput::from("I was charged twice.");
/// assert_eq!(text.as_text(), Some("I was charged twice."));
///
/// #[derive(serde::Serialize)]
/// struct Ticket { subject: &'static str, body: &'static str }
///
/// let json = DecisionInput::json(&Ticket { subject: "Refund", body: "Charged twice" }).unwrap();
/// assert_eq!(
///     json.to_json_string().as_deref(),
///     Some(r#"{"subject":"Refund","body":"Charged twice"}"#)
/// );
/// ```
#[derive(Clone, PartialEq)]
pub struct DecisionInput(Repr);

#[derive(Clone, PartialEq)]
enum Repr {
    Text(String),
    /// Always an object or an array.
    Json(Json),
}

impl DecisionInput {
    /// Text input.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self(Repr::Text(text.into()))
    }

    /// JSON input serialized from `value`, keeping its key order.
    ///
    /// The order is the one `value` serializes in: a struct's field order, or
    /// a map's iteration order. A `serde_json::Value` (for example from
    /// `json!`) has already sorted its keys unless `serde_json`'s
    /// `preserve_order` feature is enabled; serialize a struct, or deserialize
    /// the input from JSON text, to control the order. A `Value` also
    /// converts with [`From`], which cannot fail.
    ///
    /// A value that serializes to a JSON string becomes text input.
    ///
    /// # Errors
    ///
    /// Fails when `value` cannot be serialized, or serializes to a number,
    /// a boolean, or `null`: decision models read text and structured data,
    /// not scalars.
    pub fn json<T: Serialize + ?Sized>(value: &T) -> Result<Self> {
        let text = serde_json::to_string(value)?;
        let json: Json = serde_json::from_str(&text)?;
        Self::from_json(json).map_err(Error::invalid_request)
    }

    /// The text, when this is text input.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match &self.0 {
            Repr::Text(text) => Some(text),
            Repr::Json(_) => None,
        }
    }

    /// Whether this is JSON input.
    #[must_use]
    pub fn is_json(&self) -> bool {
        matches!(self.0, Repr::Json(_))
    }

    /// The JSON input as compact JSON text, keys in order.
    #[must_use]
    pub fn to_json_string(&self) -> Option<String> {
        match &self.0 {
            Repr::Text(_) => None,
            Repr::Json(json) => serde_json::to_string(json).ok(),
        }
    }

    /// Whether the input carries nothing to decide on: whitespace-only text.
    /// Empty JSON objects and arrays are not blank.
    #[must_use]
    pub fn is_blank(&self) -> bool {
        match &self.0 {
            Repr::Text(text) => text.trim().is_empty(),
            Repr::Json(_) => false,
        }
    }

    fn from_json(json: Json) -> std::result::Result<Self, String> {
        match json {
            Json::String(text) => Ok(Self::text(text)),
            json @ (Json::Object(_) | Json::Array(_)) => Ok(Self(Repr::Json(json))),
            other => Err(format!(
                "decision input must be text, a JSON object, or a JSON array, found {}",
                serde_json::to_string(&other).unwrap_or_default()
            )),
        }
    }
}

text_conversions!(DecisionInput, Self::text);

/// The same input as [`DecisionInput::json`], without the `Result`: a
/// `Value` always serializes.
///
/// Objects and arrays become JSON input and strings become text input. The
/// keys of a `Value` object are already sorted unless `serde_json`'s
/// `preserve_order` feature is enabled; serialize a struct with
/// [`DecisionInput::json`] to control the order.
///
/// `json` rejects numbers, booleans, and `null`. This conversion cannot
/// fail, so it makes numbers and booleans text input holding their JSON
/// text instead: `json!(42)` becomes the text `42`. `null` means there is
/// no input, so it becomes blank text, which
/// [`DecisionRequest::validate`](crate::DecisionRequest::validate) rejects
/// rather than sending the model the word `null`.
///
/// ```
/// use allama::DecisionRequest;
/// use allama::decision::{DecisionInput, Question};
/// use serde_json::json;
///
/// let input = DecisionInput::from(json!({"ticket": "I was charged twice."}));
/// assert_eq!(
///     input.to_json_string().as_deref(),
///     Some(r#"{"ticket":"I was charged twice."}"#)
/// );
///
/// let request = DecisionRequest::new("nimble", json!({"ticket": "I was charged twice."}))
///     .question("refund", Question::yes_no("Is the customer requesting a refund?"));
/// assert_eq!(request.input, input);
/// ```
impl From<serde_json::Value> for DecisionInput {
    fn from(value: serde_json::Value) -> Self {
        use serde_json::Value;
        match value {
            Value::String(text) => Self::text(text),
            Value::Object(_) | Value::Array(_) => Self(Repr::Json(Json::from(value))),
            Value::Null => Self::text(String::new()),
            scalar @ (Value::Bool(_) | Value::Number(_)) => Self::text(scalar.to_string()),
        }
    }
}

impl fmt::Debug for DecisionInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Repr::Text(text) => f.debug_tuple("Text").field(text).finish(),
            Repr::Json(_) => f
                .debug_tuple("Json")
                .field(&self.to_json_string().unwrap_or_default())
                .finish(),
        }
    }
}

impl Serialize for DecisionInput {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        match &self.0 {
            Repr::Text(text) => serializer.serialize_str(text),
            Repr::Json(json) => json.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for DecisionInput {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        Self::from_json(Json::deserialize(deserializer)?).map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn json_keeps_serialized_key_order() {
        #[derive(Serialize)]
        struct Ticket {
            zeta: u8,
            alpha: u8,
        }
        let input = DecisionInput::json(&Ticket { zeta: 1, alpha: 2 }).unwrap();
        assert!(input.is_json());
        assert_eq!(
            input.to_json_string().as_deref(),
            Some(r#"{"zeta":1,"alpha":2}"#)
        );
        assert_eq!(
            serde_json::to_string(&input).unwrap(),
            r#"{"zeta":1,"alpha":2}"#
        );
    }

    #[test]
    fn json_strings_become_text_and_scalars_are_rejected() {
        assert_eq!(
            DecisionInput::json("hello").unwrap(),
            DecisionInput::text("hello")
        );
        for scalar in [json!(1), json!(true), json!(null)] {
            let err = DecisionInput::json(&scalar).unwrap_err();
            assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
        }
    }

    #[test]
    fn from_value_matches_json() {
        for value in [
            json!({"zeta": 1, "alpha": {"b": [1, 2.5, null], "a": true}}),
            json!({}),
            json!([{"b": 1, "a": 2}, "x", -3]),
            json!([]),
            json!("hello"),
        ] {
            assert_eq!(
                DecisionInput::from(value.clone()),
                DecisionInput::json(&value).unwrap(),
                "{value}"
            );
        }
    }

    #[test]
    fn from_null_is_blank_text() {
        let input = DecisionInput::from(json!(null));
        assert_eq!(input, DecisionInput::text(""));
        assert!(input.is_blank());
    }

    #[test]
    fn from_value_makes_scalars_text() {
        assert_eq!(DecisionInput::from(json!(42)), DecisionInput::text("42"));
        assert_eq!(
            DecisionInput::from(json!(-2.5)),
            DecisionInput::text("-2.5")
        );
        assert_eq!(
            DecisionInput::from(json!(true)),
            DecisionInput::text("true")
        );
        assert!(!DecisionInput::from(json!(false)).is_blank());
        assert!(!DecisionInput::from(json!(0)).is_blank());
    }

    #[test]
    fn round_trips_through_serde_in_order() {
        let text = r#"{"b":[1,-2,2.5,null,true],"a":{"y":"z","x":"w"}}"#;
        let input: DecisionInput = serde_json::from_str(text).unwrap();
        assert_eq!(serde_json::to_string(&input).unwrap(), text);
        assert_eq!(
            serde_json::from_str::<DecisionInput>(r#""hi""#).unwrap(),
            DecisionInput::text("hi")
        );
        assert!(serde_json::from_str::<DecisionInput>("42").is_err());
        // Formatting differences do not matter, order does.
        assert_eq!(
            serde_json::from_str::<DecisionInput>(r#"{ "a" : 1 }"#).unwrap(),
            serde_json::from_str::<DecisionInput>(r#"{"a":1}"#).unwrap()
        );
    }

    #[test]
    fn deserializes_through_buffered_serde_paths() {
        #[derive(Deserialize)]
        #[serde(tag = "kind")]
        enum Envelope {
            Decide { input: DecisionInput },
        }
        let Envelope::Decide { input } =
            serde_json::from_str(r#"{"kind":"Decide","input":{"z":1,"a":2}}"#).unwrap();
        assert_eq!(input.to_json_string().as_deref(), Some(r#"{"z":1,"a":2}"#));
    }

    #[test]
    fn blank_is_whitespace_only_text() {
        assert!(DecisionInput::text("  \n").is_blank());
        assert!(!DecisionInput::json(&json!({})).unwrap().is_blank());
    }
}
