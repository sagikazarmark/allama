//! Gemini-specific request options.
//!
//! Both option types live in the `"google-ai"` namespace (the Gemini API's
//! origin key), so they can be attached either as typed values or as
//! namespaced JSON:
//!
//! ```
//! use allama::ChatRequest;
//! use allama_gemini::ChatRequestOptions;
//! use serde_json::json;
//!
//! let typed = ChatRequest::new("gemini-2.5-flash")
//!     .with_option(ChatRequestOptions::new().thinking_budget(-1));
//! let json = ChatRequest::new("gemini-2.5-flash")
//!     .with_options_json("google-ai", json!({"thinking_budget": -1}));
//! assert_eq!(
//!     typed.resolve_option::<ChatRequestOptions>().unwrap(),
//!     json.resolve_option::<ChatRequestOptions>().unwrap(),
//! );
//! ```
//!
//! The option types are `#[non_exhaustive]`: build them with `new()` and
//! the setters. Request fields this crate does not model go in their
//! `extra` maps, which refuse the fields the adapter owns.

use allama::adapter::Degradations;
use allama::{
    Error, ExtraMap, ProviderData, ProviderDataEntry, ProviderOptions, RequestOptions, Result,
    Subject, Warning, WarningKind,
};
use serde::{Deserialize, Serialize};

use crate::{NAMESPACE, PROVIDER_NAME};

/// Per-request options for Gemini chat calls.
///
/// ```
/// use allama::ChatRequest;
/// use allama_gemini::{ChatRequestOptions, SafetySetting};
///
/// let request = ChatRequest::new("gemini-2.5-flash").user("Hello").with_option(
///     ChatRequestOptions::new()
///         .safety_setting(SafetySetting::new(
///             "HARM_CATEGORY_DANGEROUS_CONTENT",
///             "BLOCK_ONLY_HIGH",
///         ))
///         .generation_config_extra("responseLogprobs", true)
///         .extra("labels", serde_json::json!({"team": "search"})),
/// );
/// # let _ = request;
/// ```
///
/// ```compile_fail
/// # use allama_gemini::ChatRequestOptions;
/// // Struct literals do not compile outside this crate: the struct grows
/// // with the API.
/// let options = ChatRequestOptions { thinking_budget: Some(0), ..Default::default() };
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
#[non_exhaustive]
pub struct ChatRequestOptions {
    /// `generationConfig.responseMimeType` override (for example
    /// `text/x.enum`). Prefer the portable `ChatRequest::response_format`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
    /// Name of an explicit context cache (`cachedContents/...`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_content: Option<String>,
    /// `thinkingConfig.thinkingBudget` in tokens. `-1` asks for a dynamic
    /// budget and `0` disables thinking where the model allows it.
    ///
    /// When set, it replaces the budget or level derived from
    /// `ChatRequest::reasoning.effort`; Gemini rejects requests carrying both
    /// a budget and a level, so the level is never sent alongside it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<i32>,
    /// Built-in tool definitions (`{"google_search": {}}`,
    /// `{"url_context": {}}`, `{"code_execution": {}}`, …), sent verbatim after
    /// the function declarations. Google Search grounding comes back as
    /// citations on the answer text (see the crate docs).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub builtin_tools: Vec<ExtraMap>,
    /// `safetySettings`: the blocking threshold per harm category. Gemini
    /// applies its defaults to the categories left out.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub safety_settings: Vec<SafetySetting>,
    /// Keep the verbatim `generateContent` body as a
    /// [`RawResponse`](crate::RawResponse) in the response's provider data.
    /// Unary calls only.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub raw_response: bool,
    /// Other `generationConfig` fields (`responseModalities`,
    /// `mediaResolution`, `responseLogprobs`, `speechConfig`, …), merged
    /// into the request's `generationConfig`. A field the adapter owns
    /// (`temperature`, `topK`, `thinkingConfig`, `responseMimeType`, …, in
    /// either spelling) is refused with
    /// [`Error::InvalidRequest`]: set it through the portable request or a
    /// typed option.
    #[serde(skip_serializing_if = "ExtraMap::is_empty")]
    pub generation_config_extra: ExtraMap,
    /// Other `generateContent` body fields (`labels`, …), merged into the
    /// top level of the request body. A field the adapter owns (`contents`,
    /// `tools`, `toolConfig`, `generationConfig`, `safetySettings`, …, in
    /// either spelling) is refused with [`Error::InvalidRequest`]. In the
    /// JSON form it is the nested `extra` object.
    #[serde(skip_serializing_if = "ExtraMap::is_empty")]
    pub extra: ExtraMap,
}

impl ProviderOptions for ChatRequestOptions {
    const NAMESPACE: &'static str = NAMESPACE;
}

impl ChatRequestOptions {
    /// Empty options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Override `generationConfig.responseMimeType`.
    #[must_use]
    pub fn response_mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.response_mime_type = Some(mime_type.into());
        self
    }

    /// Use the explicit context cache `name` (`cachedContents/...`).
    #[must_use]
    pub fn cached_content(mut self, name: impl Into<String>) -> Self {
        self.cached_content = Some(name.into());
        self
    }

    /// Set `thinkingConfig.thinkingBudget` (see
    /// [`thinking_budget`](Self::thinking_budget)).
    #[must_use]
    pub fn thinking_budget(mut self, budget: i32) -> Self {
        self.thinking_budget = Some(budget);
        self
    }

    /// Add a built-in tool definition (see
    /// [`builtin_tools`](Self::builtin_tools)).
    #[must_use]
    pub fn builtin_tool(mut self, definition: ExtraMap) -> Self {
        self.builtin_tools.push(definition);
        self
    }

    /// Add a safety setting.
    #[must_use]
    pub fn safety_setting(mut self, setting: SafetySetting) -> Self {
        self.safety_settings.push(setting);
        self
    }

    /// Keep the verbatim response body (see [`RawResponse`](crate::RawResponse)).
    #[must_use]
    pub fn raw_response(mut self, enabled: bool) -> Self {
        self.raw_response = enabled;
        self
    }

    /// Set one `generationConfig` field this crate does not model (see
    /// [`generation_config_extra`](Self::generation_config_extra)).
    #[must_use]
    pub fn generation_config_extra(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.generation_config_extra
            .insert(key.into(), value.into());
        self
    }

    /// Set one body field this crate does not model (see
    /// [`extra`](Self::extra)).
    #[must_use]
    pub fn extra(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }
}

/// One entry of Gemini's `safetySettings`: the probability of harm at which
/// the response is blocked, for one category.
///
/// Categories and thresholds are Gemini's own names, sent as given:
/// categories such as `HARM_CATEGORY_HARASSMENT`,
/// `HARM_CATEGORY_HATE_SPEECH`, `HARM_CATEGORY_SEXUALLY_EXPLICIT` and
/// `HARM_CATEGORY_DANGEROUS_CONTENT`; thresholds such as
/// `BLOCK_LOW_AND_ABOVE`, `BLOCK_MEDIUM_AND_ABOVE`, `BLOCK_ONLY_HIGH`,
/// `BLOCK_NONE` and `OFF`
/// (<https://ai.google.dev/gemini-api/docs/safety-settings>).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct SafetySetting {
    /// The harm category (`HARM_CATEGORY_…`).
    pub category: String,
    /// The blocking threshold (`BLOCK_…` or `OFF`).
    pub threshold: String,
}

impl SafetySetting {
    /// Block `category` at `threshold`.
    #[must_use]
    pub fn new(category: impl Into<String>, threshold: impl Into<String>) -> Self {
        Self {
            category: category.into(),
            threshold: threshold.into(),
        }
    }
}

/// Per-request options for Gemini embedding calls.
///
/// ```
/// use allama::EmbeddingRequest;
/// use allama_gemini::EmbeddingRequestOptions;
///
/// let request = EmbeddingRequest::new("gemini-embedding-001")
///     .input("Is this spam?")
///     .with_option(EmbeddingRequestOptions::new().task_type("CLASSIFICATION"));
/// # let _ = request;
/// ```
///
/// ```compile_fail
/// # use allama_gemini::EmbeddingRequestOptions;
/// // Struct literals do not compile outside this crate.
/// let options = EmbeddingRequestOptions { title: None, ..Default::default() };
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
#[non_exhaustive]
pub struct EmbeddingRequestOptions {
    /// Gemini `taskType` (for example `CLASSIFICATION`), forwarded verbatim.
    /// For retrieval prefer the portable
    /// [`EmbeddingRequest::input_type`](allama::EmbeddingRequest::input_type),
    /// which maps to `RETRIEVAL_QUERY` and `RETRIEVAL_DOCUMENT`; when both are
    /// set and differ, this option wins with an `Approximated` warning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_type: Option<String>,
    /// `title` applied to every input of the request. Gemini uses titles
    /// only for `gemini-embedding-001` with `RETRIEVAL_DOCUMENT`. A title
    /// describes one document, so prefer the per-input [`EmbeddingTitle`],
    /// which overrides this default for its input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Other `EmbedContentRequest` fields, merged into the request of every
    /// input. A field the adapter owns (`model`, `content`, `taskType`,
    /// `title`, `outputDimensionality`, in either spelling) is refused with
    /// [`Error::InvalidRequest`]. In the JSON form it is the nested `extra`
    /// object.
    #[serde(skip_serializing_if = "ExtraMap::is_empty")]
    pub extra: ExtraMap,
}

impl ProviderOptions for EmbeddingRequestOptions {
    const NAMESPACE: &'static str = NAMESPACE;
}

impl EmbeddingRequestOptions {
    /// Empty options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the Gemini `taskType` (see [`task_type`](Self::task_type)).
    #[must_use]
    pub fn task_type(mut self, task_type: impl Into<String>) -> Self {
        self.task_type = Some(task_type.into());
        self
    }

    /// Set the `title` of every input (see [`title`](Self::title)).
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set one request field this crate does not model (see
    /// [`extra`](Self::extra)).
    #[must_use]
    pub fn extra(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }
}

/// The `title` of one embedded document, attached to its
/// [`EmbeddingInput`](allama::EmbeddingInput).
///
/// Overrides [`EmbeddingRequestOptions::title`] for that input. Titles are sent
/// as given; Gemini uses them only for `gemini-embedding-001` with
/// `RETRIEVAL_DOCUMENT` (the portable `EmbeddingInputType::Document`). Attach it to the input, not to a part: a title on a
/// part fails with [`Error::InvalidRequest`](allama::Error::InvalidRequest).
///
/// ```
/// use allama::{EmbeddingInput, EmbeddingInputType, EmbeddingRequest};
/// use allama_gemini::EmbeddingTitle;
///
/// let docs = [("Rust", "Rust is a systems language."), ("Go", "Go is simple.")];
/// let request = EmbeddingRequest::new("gemini-embedding-001")
///     .inputs(docs.iter().map(|(title, body)| {
///         EmbeddingInput::text(*body).with_provider_data(EmbeddingTitle::new(*title).into())
///     }))
///     // Sent as `taskType: RETRIEVAL_DOCUMENT`.
///     .input_type(EmbeddingInputType::Document);
/// # let _ = request;
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EmbeddingTitle(pub String);

impl EmbeddingTitle {
    /// A document title.
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self(title.into())
    }
}

impl ProviderDataEntry for EmbeddingTitle {
    const NAMESPACE: &'static str = NAMESPACE;
    const KEY: &'static str = "title";
}

impl From<EmbeddingTitle> for ProviderData {
    fn from(title: EmbeddingTitle) -> Self {
        ProviderData::new().with_raw(
            NAMESPACE,
            EmbeddingTitle::KEY,
            serde_json::Value::String(title.0),
        )
    }
}

/// Report JSON options under the provider name, `gemini`, which Gemini
/// does not read: its options namespace is [`NAMESPACE`], `google-ai`. A
/// configuration file keys options by the name used everywhere else (the
/// model address, the provider name, the credential scope), and without the
/// warning a safety policy given there would be ignored silently. `typed`
/// names the options type in the remedy.
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
                "Gemini reads options under `{NAMESPACE}`, not under its name `{PROVIDER_NAME}`; they were dropped; pass them under `{NAMESPACE}`, or as `{typed}`"
            ),
        )
        .with_code("dropped.provider_name_options"),
    );
}

/// Refuse an `extra` map that sets a field the adapter owns: merged into
/// the body, it would send the field twice. Gemini's JSON reads both
/// `camelCase` and `snake_case` field names, so the comparison ignores case
/// and underscores. `owner` names the map in the error.
pub(crate) fn refuse_owned_fields(extra: &ExtraMap, owned: &[&str], owner: &str) -> Result<()> {
    let normalize = |key: &str| key.replace('_', "").to_ascii_lowercase();
    let owned: Vec<String> = owned.iter().map(|key| normalize(key)).collect();
    let Some(key) = extra.keys().find(|key| owned.contains(&normalize(key))) else {
        return Ok(());
    };
    let message = if normalize(key) == "candidatecount" {
        // There is no portable field or typed option to point at: a
        // response holds one candidate, which is also Gemini's default.
        format!(
            "`{key}` cannot be set through {owner}: allama reads one candidate per response, which is Gemini's default; leave it out"
        )
    } else {
        format!(
            "`{key}` cannot be set through {owner}: the adapter owns it; use the portable request field or the typed option"
        )
    };
    Err(Error::invalid_request(message)
        .authored()
        .with_provider(PROVIDER_NAME))
}

#[cfg(test)]
mod tests {
    use super::*;
    use allama::{ChatRequest, EmbeddingRequest};
    use serde_json::json;

    #[test]
    fn json_options_resolve_like_typed_options() {
        let typed = ChatRequest::new("m").with_option(ChatRequestOptions {
            cached_content: Some("cachedContents/x".into()),
            thinking_budget: Some(-1),
            ..Default::default()
        });
        let json = ChatRequest::new("m").with_options_json(
            NAMESPACE,
            json!({"cached_content": "cachedContents/x", "thinking_budget": -1}),
        );
        assert_eq!(
            typed.resolve_option::<ChatRequestOptions>().unwrap(),
            json.resolve_option::<ChatRequestOptions>().unwrap()
        );
    }

    #[test]
    fn unknown_option_fields_are_rejected() {
        // `top_k` is portable (`ChatRequest::top_k`), not an option.
        for field in ["topK", "top_k"] {
            let request = ChatRequest::new("m").with_options_json(NAMESPACE, json!({field: 3}));
            let err = request.resolve_option::<ChatRequestOptions>().unwrap_err();
            assert!(
                matches!(err, allama::Error::InvalidRequest { .. }),
                "{field}"
            );
        }
    }

    #[test]
    fn options_survive_request_serialization() {
        let request = ChatRequest::new("m").with_option(ChatRequestOptions {
            cached_content: Some("cachedContents/x".into()),
            ..Default::default()
        });
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(
            value["options"]["google-ai"]["cached_content"],
            "cachedContents/x"
        );
        let back: ChatRequest = serde_json::from_value(value).unwrap();
        assert_eq!(
            back.resolve_option::<ChatRequestOptions>()
                .unwrap()
                .cached_content
                .as_deref(),
            Some("cachedContents/x")
        );
    }

    #[test]
    fn embedding_options_resolve_from_json() {
        let request = EmbeddingRequest::new("m")
            .with_options_json(NAMESPACE, json!({"task_type": "RETRIEVAL_QUERY"}));
        let options = request.resolve_option::<EmbeddingRequestOptions>().unwrap();
        assert_eq!(options.task_type.as_deref(), Some("RETRIEVAL_QUERY"));
    }
}
