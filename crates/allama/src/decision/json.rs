//! An order-preserving JSON tree.
//!
//! `serde_json::Value` sorts object keys unless `preserve_order` is enabled
//! for the whole dependency graph, and `RawValue` only works with
//! `serde_json`'s own deserializer. Decision inputs need both: the model
//! reads keys in the order the caller wrote them, and requests must
//! deserialize from any self-describing format.

use std::collections::HashSet;
use std::fmt;

use serde::de::{Error as _, MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Number;

/// The key `serde_json` uses to pass a number through `deserialize_any`
/// when its `arbitrary_precision` feature is enabled anywhere in the
/// dependency graph.
const ARBITRARY_PRECISION_NUMBER: &str = "$serde_json::private::Number";

/// A JSON value whose objects keep their key order. Object keys are unique.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Serialize for Json {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Null => serializer.serialize_unit(),
            Self::Bool(value) => serializer.serialize_bool(*value),
            Self::Number(value) => value.serialize(serializer),
            Self::String(value) => serializer.serialize_str(value),
            Self::Array(items) => {
                let mut seq = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            Self::Object(entries) => {
                let mut map = serializer.serialize_map(Some(entries.len()))?;
                for (key, value) in entries {
                    map.serialize_entry(key, value)?;
                }
                map.end()
            }
        }
    }
}

impl From<serde_json::Value> for Json {
    fn from(value: serde_json::Value) -> Self {
        use serde_json::Value;
        match value {
            Value::Null => Self::Null,
            Value::Bool(value) => Self::Bool(value),
            Value::Number(value) => Self::Number(value),
            Value::String(value) => Self::String(value),
            Value::Array(items) => Self::Array(items.into_iter().map(Self::from).collect()),
            // A `Map`'s keys are unique, in its iteration order.
            Value::Object(map) => Self::Object(
                map.into_iter()
                    .map(|(key, value)| (key, Self::from(value)))
                    .collect(),
            ),
        }
    }
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(JsonVisitor)
    }
}

struct JsonVisitor;

impl<'de> Visitor<'de> for JsonVisitor {
    type Value = Json;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a JSON value")
    }

    fn visit_unit<E>(self) -> Result<Json, E> {
        Ok(Json::Null)
    }

    fn visit_none<E>(self) -> Result<Json, E> {
        Ok(Json::Null)
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Json, D::Error> {
        Json::deserialize(deserializer)
    }

    fn visit_bool<E>(self, value: bool) -> Result<Json, E> {
        Ok(Json::Bool(value))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Json, E> {
        Ok(Json::Number(value.into()))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Json, E> {
        // `Number` stores non-negative integers as unsigned, so `1` compares
        // equal whichever way a format reports it.
        Ok(Json::Number(value.into()))
    }

    fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Json, E> {
        Number::from_f64(value)
            .map(Json::Number)
            .ok_or_else(|| E::custom(format!("{value} is not a JSON number")))
    }

    fn visit_str<E>(self, value: &str) -> Result<Json, E> {
        Ok(Json::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Json, E> {
        Ok(Json::String(value))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut access: A) -> Result<Json, A::Error> {
        let mut items = Vec::with_capacity(super::named::cautious_capacity(access.size_hint()));
        while let Some(item) = access.next_element()? {
            items.push(item);
        }
        Ok(Json::Array(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> Result<Json, A::Error> {
        let mut entries: Vec<(String, Json)> =
            Vec::with_capacity(super::named::cautious_capacity(access.size_hint()));
        let mut keys = HashSet::new();
        while let Some(key) = access.next_key::<String>()? {
            if entries.is_empty() && key == ARBITRARY_PRECISION_NUMBER {
                let digits: String = access.next_value()?;
                return digits
                    .parse::<Number>()
                    .map(Json::Number)
                    .map_err(A::Error::custom);
            }
            if !keys.insert(key.clone()) {
                return Err(A::Error::custom(format!("duplicate object key {key:?}")));
            }
            let value = access.next_value()?;
            entries.push((key, value));
        }
        Ok(Json::Object(entries))
    }
}

#[cfg(test)]
mod tests {
    use serde::de::IntoDeserializer;
    use serde::de::value::{Error, MapDeserializer, SeqDeserializer};

    use super::*;

    #[test]
    fn arbitrary_precision_numbers_are_numbers() {
        // What `serde_json` hands `deserialize_any` with the feature enabled.
        let map =
            MapDeserializer::<_, Error>::new([(ARBITRARY_PRECISION_NUMBER, "2.5")].into_iter());
        let json = Json::deserialize(map.into_deserializer()).unwrap();
        assert_eq!(json, Json::Number(Number::from_f64(2.5).unwrap()));
        assert_eq!(serde_json::to_string(&json).unwrap(), "2.5");
    }

    #[test]
    fn integers_compare_equal_across_signedness() {
        let signed =
            Json::deserialize(SeqDeserializer::<_, Error>::new([1_i64].into_iter())).unwrap();
        let unsigned: Json = serde_json::from_str("[1]").unwrap();
        assert_eq!(signed, unsigned);
    }

    #[test]
    fn duplicate_keys_are_rejected() {
        let err = serde_json::from_str::<Json>(r#"{"a":1,"a":2}"#).unwrap_err();
        assert!(
            err.to_string().contains("duplicate object key \"a\""),
            "{err}"
        );
    }

    #[test]
    fn converts_from_values_like_parsing_their_text() {
        let value = serde_json::json!({
            "b": [1, -2, 2.5, null, true, "s"],
            "a": {"y": "z", "x": {}}
        });
        let parsed: Json = serde_json::from_str(&value.to_string()).unwrap();
        assert_eq!(Json::from(value), parsed);
    }

    #[test]
    fn numbers_keep_their_value() {
        let text = r#"[18446744073709551615,-9223372036854775808,-0.0,1e300,0.1]"#;
        let json: Json = serde_json::from_str(text).unwrap();
        assert_eq!(
            serde_json::to_string(&json).unwrap(),
            "[18446744073709551615,-9223372036854775808,-0.0,1e+300,0.1]"
        );
    }

    /// An iterator that declares `len` items and yields none, as a
    /// length-prefixed format reports a declared length.
    struct Declared<T>(usize, std::marker::PhantomData<T>);

    impl<T> Iterator for Declared<T> {
        type Item = T;
        fn next(&mut self) -> Option<T> {
            None
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            (self.0, Some(self.0))
        }
    }

    /// Regression: a declared length of 2^58 used
    /// to panic with "capacity overflow", and 2^34 tried to allocate ~1 TB.
    #[test]
    fn a_declared_length_does_not_preallocate() {
        let huge = 1_usize << 58;
        let seq = SeqDeserializer::<_, Error>::new(Declared::<String>(huge, Default::default()));
        assert_eq!(Json::deserialize(seq).unwrap(), Json::Array(Vec::new()));
        let map = MapDeserializer::<_, Error>::new(Declared::<(String, String)>(
            huge,
            Default::default(),
        ));
        assert_eq!(Json::deserialize(map).unwrap(), Json::Object(Vec::new()));
        let map = MapDeserializer::<_, Error>::new(Declared::<(String, String)>(
            huge,
            Default::default(),
        ));
        let entries: Vec<(String, String)> = super::super::named::deserialize(map).unwrap();
        assert!(entries.is_empty());
    }
}
