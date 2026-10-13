//! Serde for ordered `(name, value)` lists as JSON objects, keeping order.
//!
//! `serde_json`'s map type sorts keys unless `preserve_order` is enabled
//! workspace-wide; decision questions and probabilities are ordered, so they
//! are stored as lists and (de)serialized as objects here.

use std::fmt;
use std::marker::PhantomData;

use serde::de::{MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub(crate) fn serialize<S, T>(entries: &[(String, T)], serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    T: Serialize,
{
    let mut map = serializer.serialize_map(Some(entries.len()))?;
    for (name, value) in entries {
        map.serialize_entry(name, value)?;
    }
    map.end()
}

pub(crate) fn deserialize<'de, D, T>(deserializer: D) -> Result<Vec<(String, T)>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct Entries<T>(PhantomData<T>);

    impl<'de, T: Deserialize<'de>> Visitor<'de> for Entries<T> {
        type Value = Vec<(String, T)>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a map")
        }

        fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> Result<Self::Value, A::Error> {
            let mut entries = Vec::with_capacity(cautious_capacity(access.size_hint()));
            while let Some(entry) = access.next_entry()? {
                entries.push(entry);
            }
            Ok(entries)
        }
    }

    deserializer.deserialize_map(Entries(PhantomData))
}

/// A capacity to preallocate from a deserializer's `size_hint`, capped.
///
/// A length-prefixed format (MessagePack, CBOR) reports the length the input
/// declares, so trusting it lets one small message ask for terabytes or
/// overflow the capacity. serde's own collections cap the hint the same way.
pub(crate) fn cautious_capacity(hint: Option<usize>) -> usize {
    const MAX_PREALLOCATED: usize = 4096;
    hint.unwrap_or(0).min(MAX_PREALLOCATED)
}

/// Find the value named `name`.
pub(crate) fn get<'a, T>(entries: &'a [(String, T)], name: &str) -> Option<&'a T> {
    entries
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value)
}
