use std::time::Duration;

use allama::{ExtraMap, ProviderOptions};
use serde::{Deserialize, Serialize};

use crate::NAMESPACE;

/// System One request fields with no portable equivalent (namespace
/// `"systemone"`).
///
/// The struct is `#[non_exhaustive]`: build it with [`new`](Self::new) and
/// the setters, or give the JSON form.
///
/// ```
/// use allama::DecisionRequest;
/// use allama_systemone::{DecisionRequestOptions, KeepAlive};
///
/// let request = DecisionRequest::new("nimble", "Is this spam?")
///     .with_option(DecisionRequestOptions::new().keep_alive(KeepAlive::from("10m")));
/// # let _ = request;
/// ```
///
/// ```compile_fail
/// # use allama_systemone::DecisionRequestOptions;
/// // Struct literals do not compile outside this crate.
/// let options = DecisionRequestOptions { keep_alive: None, ..Default::default() };
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct DecisionRequestOptions {
    /// How long Ollama keeps the model loaded after the request. Ollama
    /// only: another endpoint is not sent it, with a warning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<KeepAlive>,
    /// Raise TypeSafe's context limits for this request, from 64k tokens
    /// per request and 32k per question to 262k for both. TypeSafe calls it
    /// a preview: such requests are slower and less accurate, so set it only
    /// on the requests that need it. TypeSafe only: another endpoint is not
    /// sent it, with a warning.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub long_context: bool,
    /// Endpoint-specific body fields, merged into the request. A field the
    /// adapter owns (`model`, `state`, `questions`, `keep_alive`, `images`,
    /// `assets`;
    /// images are [`DecisionRequest::images`](allama::DecisionRequest::images))
    /// is refused with [`Error::InvalidRequest`](allama::Error::InvalidRequest).
    /// In the JSON form it is the nested `extra` object.
    #[serde(skip_serializing_if = "ExtraMap::is_empty")]
    pub extra: ExtraMap,
    /// Keep the verbatim response body in the response's provider data (see
    /// [`RawResponse`](crate::RawResponse)).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub raw_response: bool,
}

impl ProviderOptions for DecisionRequestOptions {
    const NAMESPACE: &'static str = NAMESPACE;
}

impl DecisionRequestOptions {
    /// Empty options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set how long Ollama keeps the model loaded after the request.
    #[must_use]
    pub fn keep_alive(mut self, keep_alive: impl Into<KeepAlive>) -> Self {
        self.keep_alive = Some(keep_alive.into());
        self
    }

    /// Raise TypeSafe's context limits (see [`long_context`](Self::long_context)).
    #[must_use]
    pub fn long_context(mut self, enabled: bool) -> Self {
        self.long_context = enabled;
        self
    }

    /// Set one body field this crate does not model (see [`extra`](Self::extra)).
    #[must_use]
    pub fn extra(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }

    /// Keep the verbatim response body (see [`RawResponse`](crate::RawResponse)).
    #[must_use]
    pub fn raw_response(mut self, enabled: bool) -> Self {
        self.raw_response = enabled;
        self
    }
}

/// How long Ollama keeps a model loaded after a request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum KeepAlive {
    /// A duration string such as `"5m"` or `"1h"`.
    Duration(String),
    /// Seconds. Zero unloads the model after the request; a negative value
    /// keeps it loaded.
    Seconds(f64),
}

impl KeepAlive {
    /// Keep the model loaded indefinitely.
    pub const FOREVER: Self = Self::Seconds(-1.0);
    /// Unload the model after the request.
    pub const UNLOAD: Self = Self::Seconds(0.0);
}

impl From<Duration> for KeepAlive {
    fn from(duration: Duration) -> Self {
        Self::Seconds(duration.as_secs_f64())
    }
}

impl From<&str> for KeepAlive {
    fn from(duration: &str) -> Self {
        Self::Duration(duration.to_owned())
    }
}

impl From<String> for KeepAlive {
    fn from(duration: String) -> Self {
        Self::Duration(duration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn keep_alive_forms() {
        assert_eq!(
            serde_json::to_value(KeepAlive::from("5m")).unwrap(),
            json!("5m")
        );
        assert_eq!(
            serde_json::to_value(KeepAlive::from(Duration::from_secs(30))).unwrap(),
            json!(30.0)
        );
        assert_eq!(
            serde_json::to_value(KeepAlive::FOREVER).unwrap(),
            json!(-1.0)
        );
    }

    #[test]
    fn options_resolve_from_json() {
        let request = allama::DecisionRequest::new("m", "x")
            .with_options_json("systemone", json!({"keep_alive": 0, "raw_response": true}));
        let options: DecisionRequestOptions = request.resolve_option().unwrap();
        assert_eq!(options.keep_alive, Some(KeepAlive::UNLOAD));
        assert!(options.raw_response);
    }
}
