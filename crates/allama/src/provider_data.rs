//! Provider-scoped data attached to content, messages, tools, and responses.
//!
//! [`ProviderData`] is the persistent escape hatch for state that only one
//! provider understands: reasoning signatures, encrypted reasoning items,
//! response item ids, provider-specific annotations, and so on. Entries are
//! grouped by a **namespace** such as `"anthropic"`, `"google-ai"`, or
//! `"openai-compat:groq"`.
//!
//! The namespace names the only provider that may read an entry: an adapter
//! reads and replays only its own namespace. For state a provider minted
//! (signatures, item ids) that provider is the entry's *origin*; for
//! settings a caller attaches to a request item (a tool definition, an
//! [`EmbeddingInput`](crate::EmbeddingInput)) it is the entry's *addressee*.
//! Either way, data in one provider's namespace is never sent to another one,
//! which is what makes conversation history safe to move between providers
//! (for example through [`FallbackProvider`](crate::wrappers::FallbackProvider)).
//!
//! An empty namespace is the same as an absent one: equality, hashing,
//! [`ProviderData::is_empty`], and serialization all ignore it.
//!
//! Values are stored as JSON so they survive serialization of conversation
//! history. Typed access is available through [`ProviderDataEntry`], which
//! encodes and decodes on access.
//!
//! # Raw responses
//!
//! Providers can return the response they decoded, verbatim, for debugging and
//! for fields the portable model does not cover. This is a convention, not a
//! portable field: each provider offers a `raw_response` request option and,
//! when it is set, stores its raw response in [`ChatResponse::provider_data`](crate::ChatResponse::provider_data)
//! under its own namespace and the [`RAW_RESPONSE_KEY`] key. What the raw
//! response is (the JSON body of an HTTP API, a binding's output object) is
//! up to the provider; built-in providers export a typed `RawResponse` entry.
//! Raw responses are captured for unary calls. Streams surface unmapped data
//! as events instead: response-level data (vendor fields and stream frames
//! the adapter does not model) as
//! [`ResponseProviderData`](crate::StreamEvent::ResponseProviderData), which
//! collectors merge into the response, and block-level progress as
//! [`ProviderDelta`](crate::StreamEvent::ProviderDelta) events.
//!
//! # Decline explanations
//!
//! A provider that declines on the model's behalf (a refusal stop reason,
//! a safety finish) may explain why in its own shape: Anthropic's
//! `stop_details.explanation`, Gemini's `finishMessage`. That data stays
//! the provider's, under its namespace. As a convention, not a portable
//! field, an adapter also stores the explanation's text, verbatim, under
//! its namespace and the [`DECLINE_EXPLANATION_KEY`] key, so code that
//! reports a decline (extraction's
//! [`ChatExtractError::Refused`][crate::chat::extract::ChatExtractError::Refused]) can
//! quote it without knowing the provider. A refusal the model writes
//! itself is content, not provider data.
//!
#![cfg_attr(
    not(feature = "chat-extract"),
    doc = "[crate::chat::extract::ChatExtractError::Refused]: crate#cargo-features"
)]

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{Error, ExtraMap, Result};

/// Key of a provider's raw response inside its namespace (see
/// [Raw responses](self#raw-responses)).
pub const RAW_RESPONSE_KEY: &str = "raw_response";

/// Key of a provider's decline explanation inside its namespace: a JSON
/// string, the provider's own text (see
/// [Decline explanations](self#decline-explanations)).
pub const DECLINE_EXPLANATION_KEY: &str = "decline_explanation";

/// The namespace allama itself writes under, such as
/// [`FallbackAttempts`](crate::wrappers::FallbackAttempts).
///
/// No provider reads it and no adapter sends it, so it is foreign to none:
/// adapters do not report it as state they could not replay (see
/// [`ProviderData::foreign_namespaces`]), and
/// [`retain_provider_state`](crate::history::retain_provider_state) always
/// keeps it.
pub const ALLAMA_NAMESPACE: &str = "allama";

/// Namespaced, serializable provider-specific data.
///
/// See the [module documentation](self) for the namespace and replay rules.
///
/// An empty namespace is the same as an absent one: equality, hashing,
/// [`is_empty`](Self::is_empty), and serialization all ignore it.
#[derive(Debug, Clone, Default)]
pub struct ProviderData(BTreeMap<String, ExtraMap>);

/// A typed entry stored inside [`ProviderData`].
///
/// Implement this for small provider-specific structs that adapters attach to
/// content or responses.
///
/// ```
/// use allama::{ProviderData, ProviderDataEntry};
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Debug, PartialEq, Serialize, Deserialize)]
/// struct Fingerprint { system_fingerprint: String }
///
/// impl ProviderDataEntry for Fingerprint {
///     const NAMESPACE: &'static str = "openai";
///     const KEY: &'static str = "fingerprint";
/// }
///
/// let mut data = ProviderData::new();
/// data.insert(&Fingerprint { system_fingerprint: "fp_1".into() })?;
/// let fingerprint = data.get::<Fingerprint>()?.expect("entry is present");
/// assert_eq!(fingerprint.system_fingerprint, "fp_1");
/// # Ok::<(), allama::Error>(())
/// ```
pub trait ProviderDataEntry: Serialize + DeserializeOwned {
    /// Namespace (origin) the entry belongs to, e.g. `"anthropic"`.
    const NAMESPACE: &'static str;
    /// Key of the entry inside its namespace.
    const KEY: &'static str;
}

impl PartialEq for ProviderData {
    fn eq(&self, other: &Self) -> bool {
        self.namespaces().eq(other.namespaces())
    }
}

impl Eq for ProviderData {}

impl Hash for ProviderData {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Length-prefixed, like `BTreeMap`'s own `Hash`. `serde_json` hashes
        // maps in sorted key order, so the result does not depend on
        // insertion order (even with its `preserve_order` feature).
        state.write_usize(self.namespaces().count());
        for namespace in self.namespaces() {
            namespace.hash(state);
        }
    }
}

impl Serialize for ProviderData {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.collect_map(self.namespaces())
    }
}

impl<'de> Deserialize<'de> for ProviderData {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let mut namespaces = BTreeMap::<String, ExtraMap>::deserialize(deserializer)?;
        namespaces.retain(|_, entries| !entries.is_empty());
        Ok(Self(namespaces))
    }
}

impl ProviderData {
    /// Create an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` when no namespace holds any data.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.values().all(ExtraMap::is_empty)
    }

    /// Borrow all entries of one namespace.
    #[must_use]
    pub fn namespace(&self, namespace: &str) -> Option<&ExtraMap> {
        self.0.get(namespace).filter(|entries| !entries.is_empty())
    }

    /// Mutably borrow one namespace, creating it when missing.
    pub fn namespace_mut(&mut self, namespace: impl Into<String>) -> &mut ExtraMap {
        self.0.entry(namespace.into()).or_default()
    }

    /// Iterate over non-empty namespaces.
    pub fn namespaces(&self) -> impl Iterator<Item = (&str, &ExtraMap)> {
        self.0
            .iter()
            .filter(|(_, entries)| !entries.is_empty())
            .map(|(namespace, entries)| (namespace.as_str(), entries))
    }

    /// Iterate over the non-empty namespaces of other providers, from the
    /// point of view of the adapter that owns `namespace`: every namespace
    /// except `namespace` and [`ALLAMA_NAMESPACE`].
    ///
    /// These are the namespaces an adapter reports as not replayed.
    pub fn foreign_namespaces<'a>(&'a self, namespace: &'a str) -> impl Iterator<Item = &'a str> {
        self.namespaces()
            .map(|(name, _)| name)
            .filter(move |name| *name != namespace && *name != ALLAMA_NAMESPACE)
    }

    /// Borrow a single raw value.
    #[must_use]
    pub fn get_raw(&self, namespace: &str, key: &str) -> Option<&serde_json::Value> {
        self.0.get(namespace)?.get(key)
    }

    /// Insert a single raw value, replacing any previous value.
    pub fn insert_raw(
        &mut self,
        namespace: impl Into<String>,
        key: impl Into<String>,
        value: serde_json::Value,
    ) {
        self.namespace_mut(namespace).insert(key.into(), value);
    }

    /// Builder form of [`insert_raw`](Self::insert_raw).
    #[must_use]
    pub fn with_raw(
        mut self,
        namespace: impl Into<String>,
        key: impl Into<String>,
        value: serde_json::Value,
    ) -> Self {
        self.insert_raw(namespace, key, value);
        self
    }

    /// Decode a typed entry. Returns `Ok(None)` when absent.
    ///
    /// # Errors
    ///
    /// Returns an error when the stored JSON does not match `T`.
    pub fn get<T: ProviderDataEntry>(&self) -> Result<Option<T>> {
        self.get_raw(T::NAMESPACE, T::KEY)
            .map(|value| T::deserialize(value).map_err(Error::from))
            .transpose()
    }

    /// Encode and store a typed entry, replacing any previous value.
    ///
    /// # Errors
    ///
    /// Returns an error when `T` fails to serialize.
    pub fn insert<T: ProviderDataEntry>(&mut self, value: &T) -> Result<()> {
        let value = serde_json::to_value(value)?;
        self.insert_raw(T::NAMESPACE, T::KEY, value);
        Ok(())
    }

    /// Remove a single raw value.
    pub fn remove_raw(&mut self, namespace: &str, key: &str) -> Option<serde_json::Value> {
        let entries = self.0.get_mut(namespace)?;
        let removed = entries.remove(key);
        if entries.is_empty() {
            self.0.remove(namespace);
        }
        removed
    }

    /// Remove a whole namespace. Returns `None` when the namespace is absent
    /// or empty, consistent with [`namespace`](Self::namespace).
    pub fn remove_namespace(&mut self, namespace: &str) -> Option<ExtraMap> {
        self.0
            .remove(namespace)
            .filter(|entries| !entries.is_empty())
    }

    /// Merge `other` into `self`, key by key. Values from `other` win.
    pub fn merge(&mut self, other: ProviderData) {
        for (namespace, entries) in other.0 {
            self.namespace_mut(namespace).extend(entries);
        }
    }

    /// Keep only the namespaces for which `keep` returns `true`.
    pub fn retain_namespaces(&mut self, mut keep: impl FnMut(&str) -> bool) {
        self.0.retain(|namespace, _| keep(namespace));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Signature {
        value: String,
    }

    impl ProviderDataEntry for Signature {
        const NAMESPACE: &'static str = "anthropic";
        const KEY: &'static str = "signature";
    }

    #[test]
    fn typed_round_trip() {
        let mut data = ProviderData::new();
        data.insert(&Signature {
            value: "abc".into(),
        })
        .unwrap();
        assert_eq!(
            data.get::<Signature>().unwrap(),
            Some(Signature {
                value: "abc".into()
            })
        );
        assert_eq!(
            serde_json::to_value(&data).unwrap(),
            json!({"anthropic": {"signature": {"value": "abc"}}})
        );
    }

    #[test]
    fn foreign_namespaces_skip_the_own_and_core_namespaces() {
        let mut data = ProviderData::new()
            .with_raw("google-ai", "thought_signature", json!("s"))
            .with_raw(ALLAMA_NAMESPACE, "note", json!(1))
            .with_raw("anthropic", "signature", json!("a"));
        data.namespace_mut("openai");
        assert_eq!(
            data.foreign_namespaces("google-ai").collect::<Vec<_>>(),
            ["anthropic"]
        );
    }

    #[test]
    fn empty_namespaces_are_ignored() {
        let mut data = ProviderData::new();
        data.namespace_mut("google-ai");
        assert!(data.is_empty());
        assert!(data.namespace("google-ai").is_none());
        assert_eq!(data.namespaces().count(), 0);
        assert_eq!(data, ProviderData::new());
        assert_eq!(hash(&data), hash(&ProviderData::new()));
        assert_eq!(serde_json::to_value(&data).unwrap(), json!({}));
        assert_eq!(data.remove_namespace("google-ai"), None);
    }

    #[test]
    fn empty_namespace_left_by_namespace_mut_is_absent() {
        let mut data = ProviderData::new().with_raw("openai", "a", json!(1));
        data.namespace_mut("google-ai").insert("k".into(), json!(1));
        data.namespace_mut("google-ai").remove("k");
        let expected = ProviderData::new().with_raw("openai", "a", json!(1));
        assert_eq!(data, expected);
        assert_eq!(hash(&data), hash(&expected));
        assert_eq!(
            serde_json::to_value(&data).unwrap(),
            json!({"openai": {"a": 1}})
        );
    }

    #[test]
    fn deserialize_drops_empty_namespaces() {
        let decoded: ProviderData =
            serde_json::from_value(json!({"openai": {"a": 1}, "anthropic": {}})).unwrap();
        assert_eq!(
            decoded,
            ProviderData::new().with_raw("openai", "a", json!(1))
        );
        assert_eq!(decoded.0.len(), 1);
    }

    #[test]
    fn hash_ignores_insertion_order() {
        let mut a = serde_json::Map::new();
        a.insert("x".into(), json!(1));
        a.insert("y".into(), json!(2));
        let mut b = serde_json::Map::new();
        b.insert("y".into(), json!(2));
        b.insert("x".into(), json!(1));
        let a = ProviderData::new().with_raw("ns", "k", serde_json::Value::Object(a));
        let b = ProviderData::new().with_raw("ns", "k", serde_json::Value::Object(b));
        assert_eq!(a, b);
        assert_eq!(hash(&a), hash(&b));
    }

    fn hash<T: Hash>(value: &T) -> u64 {
        let mut hasher = std::hash::DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn merge_overwrites_per_key() {
        let mut a = ProviderData::new()
            .with_raw("openai", "a", json!(1))
            .with_raw("openai", "b", json!(2));
        a.merge(ProviderData::new().with_raw("openai", "b", json!(3)));
        assert_eq!(a.get_raw("openai", "a"), Some(&json!(1)));
        assert_eq!(a.get_raw("openai", "b"), Some(&json!(3)));
    }

    #[test]
    fn remove_raw_drops_empty_namespace() {
        let mut data = ProviderData::new().with_raw("openai", "a", json!(1));
        assert_eq!(data.remove_raw("openai", "a"), Some(json!(1)));
        assert!(data.is_empty());
    }

    #[test]
    fn typed_decode_errors_surface() {
        let data = ProviderData::new().with_raw("anthropic", "signature", json!(42));
        assert!(data.get::<Signature>().is_err());
    }

    #[test]
    fn absent_typed_entry_is_ok_none() {
        let data = ProviderData::new().with_raw("anthropic", "other", json!(42));
        assert_eq!(data.get::<Signature>().unwrap(), None);
    }
}
