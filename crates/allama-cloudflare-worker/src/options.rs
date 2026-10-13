//! Provider options for Cloudflare Workers AI requests.

use allama::adapter::Degradations;
use allama::{
    Error, ExtraMap, ProviderOptions, RequestOptions, Result, Subject, Warning, WarningKind,
};
use serde::{Deserialize, Serialize};

use crate::{NAMESPACE, PROVIDER_NAME};

/// Workers AI chat options that have no portable equivalent.
///
/// Attach the typed value with
/// [`ChatRequest::with_option`](allama::ChatRequest::with_option), or the JSON
/// form under the `"cloudflare-workers-ai"` namespace with
/// [`ChatRequest::with_options_json`](allama::ChatRequest::with_options_json).
/// Unknown JSON fields are rejected.
///
/// The struct is `#[non_exhaustive]`: build it with [`new`](Self::new) and
/// the setters. Model inputs this crate does not model (`lora`, `raw`, …) go
/// in [`extra`](Self::extra); options of the binding call itself (an AI
/// Gateway, …) go in [`run_options`](Self::run_options).
///
/// ```
/// use allama::ChatRequest;
/// use allama_cloudflare_worker::ChatRequestOptions;
/// use serde_json::json;
///
/// let request = ChatRequest::new("@cf/meta/llama-4-scout-17b-16e-instruct")
///     .user("Hello")
///     .top_k(40) // portable, not an option
///     .with_option(
///         ChatRequestOptions::new()
///             .repetition_penalty(1.1)
///             .run_option("gateway", json!({"id": "my-gateway"})),
///     );
/// # let _ = request;
/// ```
///
/// ```compile_fail
/// # use allama_cloudflare_worker::ChatRequestOptions;
/// // Struct literals do not compile outside this crate.
/// let options = ChatRequestOptions { repetition_penalty: Some(1.1), ..Default::default() };
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
#[non_exhaustive]
pub struct ChatRequestOptions {
    /// Penalty for repeated tokens; higher values discourage repetition
    /// (Workers AI documents `0.0..=2.0`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,

    /// Keep the binding's output object as a [`RawResponse`](crate::RawResponse)
    /// in the response's provider data. Unary calls only.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub raw_response: bool,

    /// Other model input fields (`lora`, `raw`, …), merged into the input
    /// object the binding receives. A field the adapter owns (`messages`,
    /// `max_tokens`, `stream`, `tools`, …) is refused with
    /// [`Error::InvalidRequest`]: set it through the portable request or a
    /// typed option. In the JSON form it is the nested `extra` object.
    #[serde(skip_serializing_if = "ExtraMap::is_empty")]
    pub extra: ExtraMap,

    /// The options argument of the binding call (`env.AI.run(model, input,
    /// options)`), such as `{"gateway": {"id": "my-gateway"}}` to route the
    /// call through an AI Gateway. Sent as given; empty sends none.
    #[serde(skip_serializing_if = "ExtraMap::is_empty")]
    pub run_options: ExtraMap,
}

impl ProviderOptions for ChatRequestOptions {
    const NAMESPACE: &'static str = crate::NAMESPACE;
}

impl ChatRequestOptions {
    /// Empty options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set `repetition_penalty`.
    #[must_use]
    pub fn repetition_penalty(mut self, penalty: f64) -> Self {
        self.repetition_penalty = Some(penalty);
        self
    }

    /// Keep the binding's output object (see [`RawResponse`](crate::RawResponse)).
    #[must_use]
    pub fn raw_response(mut self, enabled: bool) -> Self {
        self.raw_response = enabled;
        self
    }

    /// Set one model input field this crate does not model (see
    /// [`extra`](Self::extra)).
    #[must_use]
    pub fn extra(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }

    /// Set one option of the binding call (see
    /// [`run_options`](Self::run_options)).
    #[must_use]
    pub fn run_option(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.run_options.insert(key.into(), value.into());
        self
    }
}

/// Workers AI embedding options (namespace `"cloudflare-workers-ai"`).
///
/// ```
/// use allama::EmbeddingRequest;
/// use allama_cloudflare_worker::EmbeddingRequestOptions;
///
/// // The BGE models pool with `mean` unless told otherwise.
/// let request = EmbeddingRequest::new("@cf/baai/bge-base-en-v1.5")
///     .input("Hello")
///     .with_option(EmbeddingRequestOptions::new().extra("pooling", "cls"));
/// # let _ = request;
/// ```
///
/// ```compile_fail
/// # use allama_cloudflare_worker::EmbeddingRequestOptions;
/// // Struct literals do not compile outside this crate.
/// let options = EmbeddingRequestOptions { extra: Default::default(), ..Default::default() };
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
#[non_exhaustive]
pub struct EmbeddingRequestOptions {
    /// Other model input fields (`pooling`, `truncate_inputs`, …), merged
    /// into the input object the binding receives. The field the adapter
    /// owns (`text`) is refused with [`Error::InvalidRequest`]. In the JSON
    /// form it is the nested `extra` object.
    #[serde(skip_serializing_if = "ExtraMap::is_empty")]
    pub extra: ExtraMap,

    /// The options argument of the binding call (see
    /// [`ChatRequestOptions::run_options`]).
    #[serde(skip_serializing_if = "ExtraMap::is_empty")]
    pub run_options: ExtraMap,
}

impl ProviderOptions for EmbeddingRequestOptions {
    const NAMESPACE: &'static str = crate::NAMESPACE;
}

impl EmbeddingRequestOptions {
    /// Empty options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set one model input field this crate does not model (see
    /// [`extra`](Self::extra)).
    #[must_use]
    pub fn extra(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }

    /// Set one option of the binding call (see
    /// [`run_options`](Self::run_options)).
    #[must_use]
    pub fn run_option(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.run_options.insert(key.into(), value.into());
        self
    }
}

/// Workers AI decision options (namespace `"cloudflare-workers-ai"`), for
/// Clef decision models.
///
/// ```
/// use allama::DecisionRequest;
/// use allama::decision::Question;
/// use allama_cloudflare_worker::DecisionRequestOptions;
/// use serde_json::json;
///
/// let request = DecisionRequest::new("@cf/cloudflare/clef", "Checkout fails for everyone.")
///     .question("urgent", Question::yes_no("Is this urgent?"))
///     .with_option(
///         DecisionRequestOptions::new().run_option("gateway", json!({"id": "my-gateway"})),
///     );
/// # let _ = request;
/// ```
///
/// ```compile_fail
/// # use allama_cloudflare_worker::DecisionRequestOptions;
/// // Struct literals do not compile outside this crate.
/// let options = DecisionRequestOptions { raw_response: true, ..Default::default() };
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
#[non_exhaustive]
pub struct DecisionRequestOptions {
    /// Keep the binding's output object as a [`RawResponse`](crate::RawResponse)
    /// in the response's provider data.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub raw_response: bool,
    /// Other model input fields, merged into the input object the binding
    /// receives. A field the adapter owns (`model`, `state`, `questions`,
    /// `images`) is refused with [`Error::InvalidRequest`]. In the JSON form
    /// it is the nested `extra` object.
    #[serde(skip_serializing_if = "ExtraMap::is_empty")]
    pub extra: ExtraMap,
    /// The options argument of the binding call (see
    /// [`ChatRequestOptions::run_options`]).
    #[serde(skip_serializing_if = "ExtraMap::is_empty")]
    pub run_options: ExtraMap,
}

impl ProviderOptions for DecisionRequestOptions {
    const NAMESPACE: &'static str = crate::NAMESPACE;
}

impl DecisionRequestOptions {
    /// Empty options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Keep the binding's output object (see [`RawResponse`](crate::RawResponse)).
    #[must_use]
    pub fn raw_response(mut self, enabled: bool) -> Self {
        self.raw_response = enabled;
        self
    }

    /// Set one model input field this crate does not model (see
    /// [`extra`](Self::extra)).
    #[must_use]
    pub fn extra(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }

    /// Set one option of the binding call (see
    /// [`run_options`](Self::run_options)).
    #[must_use]
    pub fn run_option(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.run_options.insert(key.into(), value.into());
        self
    }
}

/// Report JSON options under the provider name, `cloudflare`, which this
/// adapter does not read: its options namespace is [`NAMESPACE`],
/// `cloudflare-workers-ai`. A configuration file keys options by the name
/// used everywhere else, and without the warning they would be ignored
/// silently. `typed` names the options type in the remedy.
pub(crate) fn report_options_under_name(
    options: &RequestOptions,
    typed: &str,
    d: &mut Degradations,
) {
    if options.json(PROVIDER_NAME).is_none() {
        return;
    }
    d.warn(
        Warning::new(
            WarningKind::Dropped,
            Subject::Option {
                namespace: PROVIDER_NAME.to_owned(),
            },
            format!(
                "Workers AI reads options under `{NAMESPACE}`, not under its name `{PROVIDER_NAME}`; they were dropped; pass them under `{NAMESPACE}`, or as `{typed}`"
            ),
        )
        .with_code("dropped.provider_name_options"),
    );
}

/// Refuse an `extra` map that sets a field the adapter owns: merged into
/// the input, it would send the field twice. The comparison ignores ASCII
/// case. `owner` names the map in the error.
pub(crate) fn refuse_owned_fields(extra: &ExtraMap, owned: &[&str], owner: &str) -> Result<()> {
    match extra
        .keys()
        .find(|key| owned.iter().any(|field| key.eq_ignore_ascii_case(field)))
    {
        Some(key) => Err(Error::invalid_request(format!(
            "`{key}` cannot be set through {owner}: the adapter owns it; use the portable request field or the typed option"
        ))
        .authored()
        .with_provider(PROVIDER_NAME)),
        None => Ok(()),
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use allama::ChatRequest;
    use serde_json::json;

    use super::*;

    #[test]
    fn resolves_typed_options() {
        let request =
            ChatRequest::new("m").with_option(ChatRequestOptions::new().repetition_penalty(1.2));
        let options = request.resolve_option::<ChatRequestOptions>().unwrap();
        assert_eq!(options.repetition_penalty, Some(1.2));
        assert!(!options.raw_response);
    }

    #[test]
    fn resolves_json_options() {
        let request = ChatRequest::new("m").with_options_json(
            "cloudflare-workers-ai",
            json!({
                "repetition_penalty": 1.5,
                "extra": {"lora": "my-adapter"},
                "run_options": {"gateway": {"id": "g"}}
            }),
        );
        let options = request.resolve_option::<ChatRequestOptions>().unwrap();
        assert_eq!(options.repetition_penalty, Some(1.5));
        assert_eq!(options.extra["lora"], "my-adapter");
        assert_eq!(options.run_options["gateway"], json!({"id": "g"}));
    }

    #[test]
    fn rejects_unknown_json_fields() {
        // `top_k` is portable (`ChatRequest::top_k`), not an option.
        for field in ["top_q", "top_k"] {
            let request =
                ChatRequest::new("m").with_options_json("cloudflare-workers-ai", json!({field: 1}));
            let err = request.resolve_option::<ChatRequestOptions>().unwrap_err();
            assert!(matches!(err, allama::Error::InvalidRequest { .. }));
            assert!(err.to_string().contains(field), "{err}");
        }
    }

    #[test]
    fn defaults_when_absent() {
        let request = ChatRequest::new("m");
        assert_eq!(
            request.resolve_option::<ChatRequestOptions>().unwrap(),
            ChatRequestOptions::default()
        );
    }

    #[test]
    fn typed_value_round_trips_through_request_serde() {
        let request =
            ChatRequest::new("m").with_option(ChatRequestOptions::new().repetition_penalty(0.5));
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(
            value["options"]["cloudflare-workers-ai"],
            json!({"repetition_penalty": 0.5})
        );
        let back: ChatRequest = serde_json::from_value(value).unwrap();
        assert_eq!(
            back.resolve_option::<ChatRequestOptions>()
                .unwrap()
                .repetition_penalty,
            Some(0.5)
        );
    }
}
