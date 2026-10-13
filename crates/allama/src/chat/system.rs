//! System instructions.
//!
//! Instructions are carried separately from conversation messages because
//! providers place them differently (a top-level field, a leading message, or
//! a developer message). Each prompt can carry a cache breakpoint,
//! provider options (for example an OpenAI option selecting `developer`
//! placement), and application data.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::chat::message::{get_app_data, insert_app_data};
use crate::macros::text_conversions;
use crate::{CacheBreakpoint, ExtraMap, ProviderOptions, RequestOptions, Result};

/// One system instruction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SystemPrompt {
    /// Instruction text.
    pub content: String,
    /// Marks the end of a cacheable prefix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheBreakpoint>,
    /// Provider options for this prompt. This is request configuration, not
    /// conversation history.
    #[serde(default, skip_serializing_if = "RequestOptions::is_empty")]
    pub options: RequestOptions,
    /// Application data: never read by adapters or sent to a provider (see
    /// [Application data](crate::history#application-data)).
    #[serde(default, skip_serializing_if = "ExtraMap::is_empty")]
    pub app_data: ExtraMap,
}

impl SystemPrompt {
    /// Create a system prompt.
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            cache: None,
            options: RequestOptions::new(),
            app_data: ExtraMap::new(),
        }
    }

    /// Mark this prompt as the end of a cacheable prefix.
    #[must_use]
    pub fn with_cache(mut self, breakpoint: CacheBreakpoint) -> Self {
        self.cache = Some(breakpoint);
        self
    }

    /// Attach provider options of the exact type.
    #[must_use]
    pub fn with_option<T: ProviderOptions + Serialize>(mut self, value: T) -> Self {
        self.options.insert(value);
        self
    }

    /// Set an [application data](crate::history#application-data) entry,
    /// replacing any previous value under `key`.
    #[must_use]
    pub fn with_app_data(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.app_data.insert(key.into(), value.into());
        self
    }

    /// Decode the application data entry under `key`. Returns `Ok(None)`
    /// when absent.
    ///
    /// # Errors
    ///
    /// Returns an error when the stored JSON does not match `T`.
    pub fn app_data_get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        get_app_data(&self.app_data, key)
    }

    /// Encode `value` and store it as the application data entry under
    /// `key`, replacing any previous value.
    ///
    /// # Errors
    ///
    /// Returns an error when `value` fails to serialize.
    pub fn app_data_insert<T: Serialize + ?Sized>(
        &mut self,
        key: impl Into<String>,
        value: &T,
    ) -> Result<()> {
        insert_app_data(&mut self.app_data, key, value)
    }

    /// Resolve provider options of type `T` (typed value or JSON form).
    ///
    /// # Errors
    ///
    /// See [`RequestOptions::resolve`].
    pub fn resolve_option<T: ProviderOptions + DeserializeOwned + Default>(&self) -> Result<T> {
        Ok(self.options.resolve_or_default::<T>()?.into_owned())
    }
}

text_conversions!(SystemPrompt, Self::new);

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn round_trip_keeps_cache_and_options() {
        let mut prompt = SystemPrompt::new("Be concise").with_cache(CacheBreakpoint::new());
        prompt
            .options
            .insert_json("openai", json!({"role": "developer"}));
        let value = serde_json::to_value(&prompt).unwrap();
        assert_eq!(value["options"]["openai"]["role"], "developer");
        let back: SystemPrompt = serde_json::from_value(value).unwrap();
        assert_eq!(back, prompt);
    }

    #[test]
    fn app_data_round_trips() {
        let mut prompt = SystemPrompt::new("Be concise").with_app_data("source", json!("persona"));
        prompt.app_data_insert("revision", &3_u32).unwrap();
        let value = serde_json::to_value(&prompt).unwrap();
        assert_eq!(
            value,
            json!({"content": "Be concise", "app_data": {"source": "persona", "revision": 3}})
        );
        let back: SystemPrompt = serde_json::from_value(value).unwrap();
        assert_eq!(back.app_data_get::<u32>("revision").unwrap(), Some(3));
        assert!(back.app_data_get::<u32>("source").is_err());
        assert_eq!(back, prompt);
    }

    #[test]
    fn minimal_serialization() {
        let value = serde_json::to_value(SystemPrompt::from("x")).unwrap();
        assert_eq!(value, json!({"content": "x"}));
    }
}
