//! Provider-specific request options.
//!
//! Each provider declares its own options type and implements
//! [`ProviderOptions`] for it. Callers attach options to a request in one of
//! two forms:
//!
//! - **the exact type**, e.g. `request.with_option(openai::ChatRequestOptions { .. })`,
//!   checked at compile time;
//! - **namespaced JSON**, e.g. `request.with_options_json("openai", json!({ .. }))`,
//!   useful for configuration files or code that does not depend on the
//!   provider crate.
//!
//! Adapters read both forms with [`RequestOptions::resolve`]: a typed value
//! wins; otherwise the JSON for the type's namespace is deserialized into it.
//! JSON is never forwarded to the wire as-is.
//!
//! Options are exportable: [`RequestOptions::to_json`] serializes typed and
//! JSON entries alike, so a serialized [`ChatRequest`](crate::ChatRequest)
//! keeps its provider options. A typed value that fails to serialize (a map
//! with non-string keys, say) fails the export, and so the serialization
//! of the request, instead of being left out. Values that are not plain data
//! can be attached with [`RequestOptions::insert_opaque`]; they are skipped
//! on export.
//!
//! # Composition
//!
//! A bag holds at most one value per type and one JSON object per
//! namespace. Inserting again **replaces** the earlier value; nothing is
//! merged implicitly, so the request always shows exactly what an adapter
//! will read. A typed value and JSON for the same namespace may coexist; the
//! typed value wins when resolved.
//!
//! Each namespace maps to one options type. Typed values of two types under
//! one namespace (an embedding options value attached to a chat request, or
//! the same type from two versions of the crate that defines it) fail
//! [`resolve`](RequestOptions::resolve) and
//! [`resolve_mut`](RequestOptions::resolve_mut) with
//! [`Error::InvalidRequest`] naming the namespace and both types, instead of
//! one of them being ignored.
//!
//! To combine settings from several places (defaults, configuration, a
//! wrapper), modify the value in place with
//! [`resolve_mut`](RequestOptions::resolve_mut): it starts from the typed
//! value, else from the JSON form, else from the default, so what the
//! caller set is kept:
//!
//! ```
//! # use allama::{ProviderOptions, RequestOptions};
//! # #[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
//! # struct MyOptions { top_k: Option<u32>, user: Option<String> }
//! # impl ProviderOptions for MyOptions { const NAMESPACE: &'static str = "my"; }
//! let mut options = RequestOptions::new();
//! options.insert_json("my", serde_json::json!({"top_k": 20}));
//! options.resolve_mut::<MyOptions>()?.user = Some("user_123".into());
//! let resolved = options.get::<MyOptions>().unwrap();
//! assert_eq!(resolved.top_k, Some(20));
//! assert_eq!(resolved.user.as_deref(), Some("user_123"));
//! # Ok::<(), allama::Error>(())
//! ```
//!
//! [`get_mut`](RequestOptions::get_mut) changes a typed value that is
//! there, and only that.

use std::any::{Any, TypeId};
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{Error, ExtraMap, Result, SerializationError};

type ErasedAny = dyn Any + Send + Sync;

/// Implemented by provider-declared option types.
///
/// The namespace is the options namespace: the key the options are read
/// under, and so the key of their JSON form (for example `"openai"` or
/// `"anthropic"`). It usually equals the provider's
/// [`ProviderData`](crate::ProviderData) namespace, but may differ: every
/// OpenAI-compatible preset reads its options under the shared
/// `"openai-compat"` namespace while it stores its state under
/// `"openai-compat:<name>"`, so one options value configures every preset.
/// A few options namespaces belong to no provider: `"allama.chat.extraction"` is
/// read by extraction, and `"allama-http"` by the transport code of every
/// HTTP adapter. Within one options bag, each namespace maps to a single
/// options type: [`RequestOptions::resolve`] fails when a typed value of
/// another type was attached under the namespace it reads.
pub trait ProviderOptions: Clone + Send + Sync + 'static {
    /// Namespace of the JSON form, e.g. `"openai"`.
    const NAMESPACE: &'static str;
}

/// Provider options attached to a request, keyed by type or namespace.
///
/// Callers attach options as the exact type ([`insert`](Self::insert)) or
/// as namespaced JSON ([`insert_json`](Self::insert_json)); adapters read
/// both with [`resolve`](Self::resolve). Exportable entries round-trip
/// through [`to_json`](Self::to_json) and [`from_json`](Self::from_json). A provider ignores options it does not recognize, so one
/// request can carry options for several providers.
///
/// Each namespace maps to one options type. Typed values of two types
/// under one namespace (an embedding options value attached to a chat
/// request, or the same type from two versions of the crate that defines
/// it) fail [`resolve`](Self::resolve) and
/// [`resolve_mut`](Self::resolve_mut) with [`Error::InvalidRequest`],
/// naming the namespace and both types, instead of one being ignored.
#[derive(Clone, Default)]
pub struct RequestOptions {
    typed: HashMap<TypeId, TypedEntry>,
    json: ExtraMap,
}

#[derive(Clone)]
struct TypedEntry {
    value: Arc<ErasedAny>,
    export: Option<Export>,
}

#[derive(Clone, Copy)]
struct Export {
    namespace: &'static str,
    type_name: &'static str,
    to_json: fn(&ErasedAny) -> serde_json::Result<serde_json::Value>,
}

impl RequestOptions {
    /// Create an empty options bag.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or replace) provider options of the exact type.
    ///
    /// The value is exported by [`to_json`](Self::to_json).
    pub fn insert<T>(&mut self, value: T)
    where
        T: ProviderOptions + Serialize,
    {
        fn to_json<T: Serialize + 'static>(
            value: &ErasedAny,
        ) -> serde_json::Result<serde_json::Value> {
            let value = value
                .downcast_ref::<T>()
                .expect("an entry holds a value of the type it was inserted as");
            serde_json::to_value(value)
        }

        self.typed.insert(
            TypeId::of::<T>(),
            TypedEntry {
                value: Arc::new(value),
                export: Some(Export {
                    namespace: T::NAMESPACE,
                    type_name: std::any::type_name::<T>(),
                    to_json: to_json::<T>,
                }),
            },
        );
    }

    /// Insert (or replace) a value that is not plain data.
    ///
    /// Opaque values are readable with [`get`](Self::get) but are skipped by
    /// [`to_json`](Self::to_json) and ignored by equality.
    pub fn insert_opaque<T>(&mut self, value: T)
    where
        T: Clone + Send + Sync + 'static,
    {
        self.typed.insert(
            TypeId::of::<T>(),
            TypedEntry {
                value: Arc::new(value),
                export: None,
            },
        );
    }

    /// Insert (or replace) options for `namespace` in their JSON form.
    pub fn insert_json(&mut self, namespace: impl Into<String>, value: serde_json::Value) {
        self.json.insert(namespace.into(), value);
    }

    /// Borrow a typed value by exact type. JSON entries are not consulted;
    /// use [`resolve`](Self::resolve) for that.
    #[must_use]
    pub fn get<T: 'static>(&self) -> Option<&T> {
        self.typed.get(&TypeId::of::<T>())?.value.downcast_ref()
    }

    /// Mutably borrow a typed value by exact type, cloning it if shared.
    ///
    /// JSON entries are not consulted; to change the options whichever
    /// form they were given in, use [`resolve_mut`](Self::resolve_mut).
    pub fn get_mut<T: Clone + Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        let entry = self.typed.get_mut(&TypeId::of::<T>())?;
        if Arc::get_mut(&mut entry.value).is_none() {
            let cloned: T = entry.value.downcast_ref::<T>()?.clone();
            entry.value = Arc::new(cloned);
        }
        Arc::get_mut(&mut entry.value)?.downcast_mut()
    }

    /// Remove and return a typed value by exact type.
    pub fn remove<T: Clone + Send + Sync + 'static>(&mut self) -> Option<T> {
        let entry = self.typed.remove(&TypeId::of::<T>())?;
        entry.value.downcast_ref::<T>().cloned()
    }

    /// Returns `true` when a typed value of `T` is present.
    #[must_use]
    pub fn contains<T: 'static>(&self) -> bool {
        self.typed.contains_key(&TypeId::of::<T>())
    }

    /// Borrow the JSON form stored for `namespace`, if any.
    #[must_use]
    pub fn json(&self, namespace: &str) -> Option<&serde_json::Value> {
        self.json.get(namespace)
    }

    /// Remove the JSON form stored for `namespace`.
    pub fn remove_json(&mut self, namespace: &str) -> Option<serde_json::Value> {
        self.json.remove(namespace)
    }

    /// Resolve the options of type `T` from either input form.
    ///
    /// A typed value wins. Otherwise the JSON stored under `T::NAMESPACE` is
    /// deserialized into `T`. Returns `Ok(None)` when neither is present.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`] when the JSON form does not match `T`
    /// (for example an unknown field when `T` uses `deny_unknown_fields`),
    /// or when a typed value of another type was attached under
    /// `T::NAMESPACE`: each namespace maps to one options type.
    pub fn resolve<T>(&self) -> Result<Option<Cow<'_, T>>>
    where
        T: ProviderOptions + DeserializeOwned,
    {
        self.check_namespace::<T>()?;
        if let Some(value) = self.get::<T>() {
            return Ok(Some(Cow::Borrowed(value)));
        }
        match self.json.get(T::NAMESPACE) {
            None => Ok(None),
            Some(value) => serde_json::from_value(value.clone())
                .map(|value| Some(Cow::Owned(value)))
                .map_err(|err| {
                    Error::invalid_request(format!("invalid `{}` options: {err}", T::NAMESPACE))
                }),
        }
    }

    /// Like [`resolve`](Self::resolve), falling back to `T::default()`.
    ///
    /// # Errors
    ///
    /// See [`resolve`](Self::resolve).
    pub fn resolve_or_default<T>(&self) -> Result<Cow<'_, T>>
    where
        T: ProviderOptions + DeserializeOwned + Default,
    {
        Ok(self
            .resolve::<T>()?
            .unwrap_or_else(|| Cow::Owned(T::default())))
    }

    /// Resolve the options of type `T` for changing in place: the typed
    /// value, else the JSON form deserialized into `T`, else
    /// `T::default()`.
    ///
    /// The result is stored as the typed value and replaces the JSON form,
    /// so what an adapter resolves afterwards is what you changed. Use it
    /// to add to the caller's options (a header, a tag) without replacing
    /// them, which [`insert`](Self::insert) would do.
    ///
    /// ```
    /// # use allama::{ProviderOptions, RequestOptions};
    /// # #[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
    /// # struct MyOptions { tags: Vec<String> }
    /// # impl ProviderOptions for MyOptions { const NAMESPACE: &'static str = "my"; }
    /// let mut options = RequestOptions::new();
    /// options.insert(MyOptions { tags: vec!["caller".into()] });
    /// options.resolve_mut::<MyOptions>()?.tags.push("wrapper".into());
    /// assert_eq!(options.get::<MyOptions>().unwrap().tags, ["caller", "wrapper"]);
    /// # Ok::<(), allama::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`] when the JSON form does not match
    /// `T`, or when a typed value of another type was attached under
    /// `T::NAMESPACE`, as [`resolve`](Self::resolve) does; the options are
    /// left unchanged.
    pub fn resolve_mut<T>(&mut self) -> Result<&mut T>
    where
        T: ProviderOptions + Serialize + DeserializeOwned + Default,
    {
        self.check_namespace::<T>()?;
        if !self.contains::<T>() {
            let value = match self.resolve::<T>()? {
                Some(value) => value.into_owned(),
                None => T::default(),
            };
            self.json.remove(T::NAMESPACE);
            self.insert(value);
        }
        Ok(self
            .get_mut::<T>()
            .expect("the typed value was just found or inserted"))
    }

    /// Export all exportable options as `{namespace: json}`.
    ///
    /// Typed values are serialized under their namespace and take precedence
    /// over a JSON entry for the same namespace. Opaque values are skipped.
    ///
    /// # Errors
    ///
    /// Returns the error of a typed value that fails to serialize (a map
    /// with non-string keys, say), prefixed with its namespace. It is
    /// never left out of the export silently.
    pub fn to_json(&self) -> std::result::Result<ExtraMap, SerializationError> {
        let mut out = self.json.clone();
        for entry in self.typed.values() {
            if let Some(export) = entry.export {
                let value = (export.to_json)(entry.value.as_ref()).map_err(|err| {
                    let mut err = SerializationError::from(err);
                    err.prefix(&format!("`{}` options", export.namespace));
                    err
                })?;
                out.insert(export.namespace.to_owned(), value);
            }
        }
        Ok(out)
    }

    /// Rebuild an options bag from an export. Every entry comes back in its
    /// JSON form and is resolved lazily by the adapter that understands it.
    #[must_use]
    pub fn from_json(json: ExtraMap) -> Self {
        Self {
            typed: HashMap::new(),
            json,
        }
    }

    /// Returns `true` when the bag holds no options.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.typed.is_empty() && self.json.is_empty()
    }

    /// Number of stored entries (typed and JSON).
    #[must_use]
    pub fn len(&self) -> usize {
        self.typed.len() + self.json.len()
    }

    /// Fail when a typed value of a type other than `T` was attached under
    /// `T::NAMESPACE`: reading only `T` would ignore it silently.
    fn check_namespace<T: ProviderOptions>(&self) -> Result<()> {
        let expected = TypeId::of::<T>();
        let clash = self.typed.iter().find_map(|(id, entry)| {
            entry
                .export
                .filter(|export| *id != expected && export.namespace == T::NAMESPACE)
        });
        match clash {
            None => Ok(()),
            Some(export) => Err(Error::invalid_request(format!(
                "`{}` options were attached as `{}`, but this call reads them as `{}`; \
                 attach one options type per namespace",
                T::NAMESPACE,
                export.type_name,
                std::any::type_name::<T>(),
            ))),
        }
    }
}

impl fmt::Debug for RequestOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RequestOptions")
            .field("typed", &self.typed.len())
            .field("json", &self.json)
            .finish()
    }
}

impl PartialEq for RequestOptions {
    /// Compares the exported form; opaque values are ignored.
    ///
    /// Bags compare unequal when either holds a typed value that fails to
    /// serialize, even to themselves: their exported forms are unknown.
    fn eq(&self, other: &Self) -> bool {
        match (self.to_json(), other.to_json()) {
            (Ok(this), Ok(other)) => this == other,
            _ => false,
        }
    }
}

/// Serializes the [exported form](RequestOptions::to_json), and fails with
/// its error.
impl Serialize for RequestOptions {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.to_json()
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for RequestOptions {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        ExtraMap::deserialize(deserializer).map(Self::from_json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct DemoOptions {
        tier: Option<String>,
        effort: Option<String>,
    }

    impl ProviderOptions for DemoOptions {
        const NAMESPACE: &'static str = "demo";
    }

    #[derive(Clone)]
    struct Handle(#[allow(dead_code)] u8);

    #[test]
    fn typed_value_resolves() {
        let mut options = RequestOptions::new();
        options.insert(DemoOptions {
            tier: Some("flex".into()),
            ..Default::default()
        });
        let resolved = options.resolve::<DemoOptions>().unwrap().unwrap();
        assert_eq!(resolved.tier.as_deref(), Some("flex"));
    }

    #[test]
    fn resolve_mut_starts_from_what_the_caller_set() {
        // The typed value is changed in place.
        let mut options = RequestOptions::new();
        options.insert(DemoOptions {
            tier: Some("flex".into()),
            ..Default::default()
        });
        options.resolve_mut::<DemoOptions>().unwrap().effort = Some("low".into());
        let typed = options.get::<DemoOptions>().unwrap();
        assert_eq!(
            (typed.tier.as_deref(), typed.effort.as_deref()),
            (Some("flex"), Some("low"))
        );

        // The JSON form is resolved, and replaced by the typed value.
        let mut options = RequestOptions::new();
        options.insert_json("demo", json!({"tier": "flex"}));
        options.resolve_mut::<DemoOptions>().unwrap().effort = Some("low".into());
        assert_eq!(options.json("demo"), None);
        assert_eq!(
            options.to_json().unwrap()["demo"],
            json!({"tier": "flex", "effort": "low"})
        );

        // Neither: the default.
        let mut options = RequestOptions::new();
        options.resolve_mut::<DemoOptions>().unwrap().effort = Some("low".into());
        assert_eq!(options.len(), 1);
        assert_eq!(options.get::<DemoOptions>().unwrap().tier, None);
    }

    #[test]
    fn resolve_mut_leaves_invalid_json_in_place() {
        let mut options = RequestOptions::new();
        options.insert_json("demo", json!({"unknown": true}));
        let err = options.resolve_mut::<DemoOptions>().unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
        assert!(options.json("demo").is_some());
        assert!(!options.contains::<DemoOptions>());
    }

    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
    struct OtherDemoOptions {
        tier: Option<String>,
    }

    impl ProviderOptions for OtherDemoOptions {
        const NAMESPACE: &'static str = "demo";
    }

    #[test]
    fn two_types_in_one_namespace_fail_to_resolve() {
        let mut options = RequestOptions::new();
        options.insert(OtherDemoOptions {
            tier: Some("flex".into()),
        });

        let err = options.resolve::<DemoOptions>().unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
        let message = err.to_string();
        assert!(message.contains("`demo` options"), "{message}");
        assert!(message.contains("OtherDemoOptions"), "{message}");
        assert!(message.contains("DemoOptions`"), "{message}");

        // Also when `T` itself is there: one of the two would be ignored.
        options.insert(DemoOptions::default());
        assert!(matches!(
            options.resolve::<DemoOptions>(),
            Err(Error::InvalidRequest { .. })
        ));
        assert!(matches!(
            options.resolve_or_default::<DemoOptions>(),
            Err(Error::InvalidRequest { .. })
        ));

        // resolve_mut fails too, and leaves the options as they were.
        let mut options = RequestOptions::new();
        options.insert(OtherDemoOptions::default());
        options.insert_json("demo", json!({"tier": "flex"}));
        let err = options.resolve_mut::<DemoOptions>().unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
        assert!(!options.contains::<DemoOptions>());
        assert!(options.json("demo").is_some());

        // An opaque value has no namespace, and JSON alone is not a type.
        let mut options = RequestOptions::new();
        options.insert_opaque(Handle(1));
        options.insert_json("demo", json!({"tier": "flex"}));
        assert!(options.resolve::<DemoOptions>().unwrap().is_some());
    }

    #[test]
    fn json_form_resolves_into_type() {
        let mut options = RequestOptions::new();
        options.insert_json("demo", json!({"effort": "high"}));
        let resolved = options.resolve::<DemoOptions>().unwrap().unwrap();
        assert_eq!(resolved.effort.as_deref(), Some("high"));
    }

    #[test]
    fn unknown_json_field_is_invalid_request() {
        let mut options = RequestOptions::new();
        options.insert_json("demo", json!({"efort": "high"}));
        let err = options.resolve::<DemoOptions>().unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
    }

    #[test]
    fn typed_wins_over_json() {
        let mut options = RequestOptions::new();
        options.insert_json("demo", json!({"tier": "default"}));
        options.insert(DemoOptions {
            tier: Some("flex".into()),
            ..Default::default()
        });
        assert_eq!(
            options
                .resolve::<DemoOptions>()
                .unwrap()
                .unwrap()
                .tier
                .as_deref(),
            Some("flex")
        );
        assert_eq!(options.to_json().unwrap()["demo"]["tier"], "flex");
    }

    #[test]
    fn export_round_trip_is_lossless_for_data() {
        let mut options = RequestOptions::new();
        options.insert(DemoOptions {
            tier: Some("flex".into()),
            ..Default::default()
        });
        options.insert_json("other", json!({"top_k": 5}));
        options.insert_opaque(Handle(1));

        let restored: RequestOptions =
            serde_json::from_value(serde_json::to_value(&options).unwrap()).unwrap();
        assert_eq!(restored, options);
        assert!(restored.get::<Handle>().is_none());
        assert_eq!(
            restored
                .resolve::<DemoOptions>()
                .unwrap()
                .unwrap()
                .tier
                .as_deref(),
            Some("flex")
        );
        assert_eq!(restored.json("other"), Some(&json!({"top_k": 5})));
    }

    /// A typed value whose serialization fails is never dropped silently:
    /// the export and serialization fail, and equality is false.
    #[test]
    fn a_typed_value_that_fails_to_serialize_is_an_error() {
        #[derive(Debug, Clone, Default, Serialize)]
        struct Keyed {
            // serde_json takes only string map keys.
            by_pair: HashMap<(u8, u8), u8>,
        }
        impl ProviderOptions for Keyed {
            const NAMESPACE: &'static str = "keyed";
        }

        let mut failing = RequestOptions::new();
        failing.insert(Keyed {
            by_pair: HashMap::from([((1, 2), 3)]),
        });
        let err = failing.to_json().unwrap_err();
        assert!(err.to_string().starts_with("`keyed` options: "), "{err}");
        assert!(err.get_ref().downcast_ref::<serde_json::Error>().is_some());
        let err = serde_json::to_value(&failing).unwrap_err();
        assert!(err.to_string().contains("`keyed` options"), "{err}");

        assert_ne!(failing, RequestOptions::new());
        assert_ne!(RequestOptions::new(), failing);
        assert_ne!(failing, failing.clone());

        // A typed value that serializes is still compared by its JSON.
        let mut working = RequestOptions::new();
        working.insert(Keyed::default());
        assert_eq!(working, working.clone());
        let mut opaque = working.clone();
        opaque.insert_opaque(Handle(1));
        assert_eq!(opaque, working, "opaque values stay out of equality");
    }

    #[test]
    fn get_mut_and_remove() {
        let mut options = RequestOptions::new();
        options.insert(DemoOptions::default());
        let cloned = options.clone();
        options.get_mut::<DemoOptions>().unwrap().tier = Some("x".into());
        assert!(cloned.get::<DemoOptions>().unwrap().tier.is_none());
        assert_eq!(
            options.remove::<DemoOptions>().unwrap().tier.as_deref(),
            Some("x")
        );
        assert!(!options.contains::<DemoOptions>());
    }
}
