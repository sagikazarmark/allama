use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::macros::enum_names;
use crate::{
    EmbeddingInput, EmbeddingPart, Error, ProviderOptions, RequestOptions, Result, Subject,
    UnsupportedPolicy,
};

/// A batch embedding request.
///
/// Holds one [`EmbeddingInput`] per vector to return. A request with no inputs
/// is invalid: providers fail it with [`Error::InvalidRequest`], so a wrapper
/// that filters inputs (for example to serve cache hits) must not call the
/// inner provider when nothing is left.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EmbeddingRequest {
    /// Model identifier.
    pub model: String,
    /// Inputs to embed, in order. The response holds one vector per input.
    /// Text always works; other modalities depend on
    /// [`embedding_capability`](crate::EmbeddingProvider::embedding_capability).
    pub inputs: Vec<EmbeddingInput>,
    /// Requested output dimensionality. Semantic: providers that cannot honor
    /// it fail rather than return vectors of another size.
    ///
    /// Providers differ in whether a reduced vector is unit length: some
    /// normalize it, Gemini normalizes only full-size vectors. Normalize the
    /// vectors yourself before comparing them by dot product.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    /// What the inputs are for in asymmetric retrieval: search queries or
    /// the documents they search. A hint (ADR 0003): a provider that cannot
    /// honor it drops it with a warning, or fails under
    /// [`UnsupportedPolicy::Error`]; see
    /// [`EmbeddingCapability::InputType`](crate::EmbeddingCapability::InputType).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_type: Option<EmbeddingInputType>,
    /// How adapters treat portable hints they cannot honor.
    ///
    /// `None` inherits: a wrapper that sets a provider-level default
    /// ([`UnsupportedPolicyProvider`](crate::wrappers::UnsupportedPolicyProvider))
    /// fills it in, and the provider treats a policy still unset as
    /// [`UnsupportedPolicy::Warn`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unsupported_policy: Option<UnsupportedPolicy>,
    /// Provider-specific options.
    #[serde(default, skip_serializing_if = "RequestOptions::is_empty")]
    pub options: RequestOptions,
}

impl EmbeddingRequest {
    /// Create a request for `model`.
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            inputs: Vec::new(),
            dimensions: None,
            input_type: None,
            unsupported_policy: None,
            options: RequestOptions::new(),
        }
    }

    /// Replace the inputs, one vector each.
    ///
    /// Strings convert to text inputs, so `.inputs(["a", "b"])` embeds two
    /// texts. A `Vec<EmbeddingPart>` converts to ONE input that combines its
    /// parts into a single vector.
    #[must_use]
    pub fn inputs<I, S>(mut self, inputs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<EmbeddingInput>,
    {
        self.inputs = inputs.into_iter().map(Into::into).collect();
        self
    }

    /// Append an input. A `Vec<EmbeddingPart>` is one combined input, not
    /// several inputs.
    #[must_use]
    pub fn input(mut self, input: impl Into<EmbeddingInput>) -> Self {
        self.inputs.push(input.into());
        self
    }

    /// Borrow every input as text, for adapters of models that embed one text
    /// per input.
    ///
    /// `provider` attributes the error. Content is never dropped or joined: an
    /// input that is not exactly one text part fails under every
    /// [`UnsupportedPolicy`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`] for a request without inputs or an
    /// input without parts, and [`Error::Unsupported`] for an input with
    /// several parts. Inputs are checked in order.
    pub fn text_inputs(&self, provider: &str) -> Result<Vec<&str>> {
        if self.inputs.is_empty() {
            return Err(
                Error::invalid_request("embedding request must have at least one input")
                    .authored()
                    .with_provider(provider),
            );
        }
        // Refusals before sending, attributed to `provider`.
        let refusals = crate::Degradations::new(provider.to_owned(), UnsupportedPolicy::Warn);
        self.inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                // Part kinds first, then the count. The match is exhaustive on
                // purpose: a new part kind must be reported here ("embeds text
                // only") before it compiles.
                for (position, part) in input.parts().iter().enumerate() {
                    match part {
                        EmbeddingPart::Text(_) => {}
                        EmbeddingPart::Image(_) => {
                            return Err(refusals.reject(
                                Subject::EmbeddingInput {
                                    input: index,
                                    part: Some(position),
                                },
                                format!("model `{}` embeds text only", self.model),
                            ));
                        }
                    }
                }
                match input.parts() {
                    [] => Err(Error::invalid_request(format!(
                        "embedding input {index} has no parts"
                    ))
                    .authored()
                    .with_provider(provider)),
                    [EmbeddingPart::Text(text)] => Ok(text.text.as_str()),
                    parts => Err(refusals.reject(
                        Subject::EmbeddingInput {
                            input: index,
                            part: None,
                        },
                        format!(
                            "model `{}` embeds one part per input, and this input has {} parts; \
                             pass separate inputs with `.inputs([..])` to embed them separately",
                            self.model,
                            parts.len()
                        ),
                    )),
                }
            })
            .collect::<Result<_>>()
    }

    /// Check the provider data of every input for `namespace`, for adapters.
    ///
    /// `provider` attributes the error; `namespace` is the provider data
    /// namespace the adapter reads (for example `allama_gemini::NAMESPACE`);
    /// `input_keys` are the keys it reads from
    /// [`EmbeddingInput::provider_data`]. Only `namespace` is inspected, and
    /// empty namespaces count as absent.
    ///
    /// - Only callers set input provider data, so a key in `namespace` that is
    ///   not in `input_keys` is a mistake: a typo would otherwise silently
    ///   change the vectors.
    /// - A key from `input_keys` in a part's provider data is at the wrong
    ///   level. Other keys on parts are ignored, because parts taken from chat
    ///   output legitimately carry the provider's replay state.
    ///
    /// Decoding the values of known keys is left to the adapter, which owns
    /// their types.
    ///
    /// ```
    /// # use allama::{EmbeddingInput, EmbeddingRequest, ProviderData};
    /// # use serde_json::json;
    /// let request = EmbeddingRequest::new("model").input(
    ///     EmbeddingInput::text("doc")
    ///         .with_provider_data(ProviderData::new().with_raw("demo", "titel", json!("x"))),
    /// );
    /// // provider name (for the error), then the namespace to inspect.
    /// let err = request
    ///     .check_provider_data("demo-provider", "demo", &["title"])
    ///     .unwrap_err();
    /// assert_eq!(
    ///     err.to_string(),
    ///     "demo-provider: invalid request: embedding input 0 has unknown `demo` provider data key `titel`"
    /// );
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`] naming the input (and part) of the
    /// first violation.
    pub fn check_provider_data(
        &self,
        provider: &str,
        namespace: &str,
        input_keys: &[&str],
    ) -> Result<()> {
        let fail = |message: String| {
            Err(Error::invalid_request(message)
                .authored()
                .with_provider(provider))
        };
        for (index, input) in self.inputs.iter().enumerate() {
            if let Some(entries) = input.provider_data.namespace(namespace)
                && let Some(key) = entries
                    .keys()
                    .find(|key| !input_keys.contains(&key.as_str()))
            {
                return fail(format!(
                    "embedding input {index} has unknown `{namespace}` provider data key `{key}`"
                ));
            }
            for (part_index, part) in input.parts().iter().enumerate() {
                // Exhaustive on purpose: each new part kind's provider data
                // must be checked here.
                let data = match part {
                    EmbeddingPart::Text(text) => &text.provider_data,
                    EmbeddingPart::Image(image) => &image.provider_data,
                };
                if let Some(entries) = data.namespace(namespace)
                    && let Some(key) = entries
                        .keys()
                        .find(|key| input_keys.contains(&key.as_str()))
                {
                    return fail(format!(
                        "embedding input {index}, part {part_index} has `{namespace}` key \
                         `{key}`, which belongs on the input"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Set the output dimensionality.
    #[must_use]
    pub fn dimensions(mut self, dimensions: u32) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    /// Say what the inputs are for: search queries or the documents they
    /// search.
    ///
    /// ```
    /// use allama::{EmbeddingInputType, EmbeddingRequest};
    ///
    /// let index = EmbeddingRequest::new("model")
    ///     .inputs(["Rust is a systems language.", "Go is simple."])
    ///     .input_type(EmbeddingInputType::Document);
    /// let search = EmbeddingRequest::new("model")
    ///     .input("Which language is simple?")
    ///     .input_type(EmbeddingInputType::Query);
    /// # let _ = (index, search);
    /// ```
    #[must_use]
    pub fn input_type(mut self, input_type: EmbeddingInputType) -> Self {
        self.input_type = Some(input_type);
        self
    }

    /// Set the unsupported-hint policy.
    #[must_use]
    pub fn unsupported_policy(mut self, policy: UnsupportedPolicy) -> Self {
        self.unsupported_policy = Some(policy);
        self
    }

    /// Attach provider options of the exact type.
    #[must_use]
    pub fn with_option<T: ProviderOptions + Serialize>(mut self, value: T) -> Self {
        self.options.insert(value);
        self
    }

    /// Attach an application value that is not a provider option: data a
    /// wrapper of yours reads back with [`option`](Self::option), such as a
    /// tenant id or a conversation id. Replaces any earlier value of that
    /// type.
    ///
    /// The value needs no namespace and no serde: adapters never send it,
    /// and it is left out of the request's serialized form and its
    /// equality (an *opaque* option, [`RequestOptions::insert_opaque`](crate::RequestOptions::insert_opaque)).
    #[must_use]
    pub fn with_opaque_option<T: Clone + Send + Sync + 'static>(mut self, value: T) -> Self {
        self.options.insert_opaque(value);
        self
    }

    /// Attach provider options in their namespaced JSON form.
    #[must_use]
    pub fn with_options_json(
        mut self,
        namespace: impl Into<String>,
        value: serde_json::Value,
    ) -> Self {
        self.options.insert_json(namespace, value);
        self
    }

    /// Borrow provider options of the exact type.
    #[must_use]
    pub fn option<T: 'static>(&self) -> Option<&T> {
        self.options.get::<T>()
    }

    /// Resolve provider options of type `T`, falling back to the default.
    ///
    /// # Errors
    ///
    /// See [`RequestOptions::resolve`].
    pub fn resolve_option<T>(&self) -> Result<T>
    where
        T: ProviderOptions + DeserializeOwned + Default,
    {
        Ok(self.options.resolve_or_default::<T>()?.into_owned())
    }

    /// Mutably borrow provider options of the exact type, when attached.
    ///
    /// Like [`option`](Self::option), the JSON form is not consulted; to
    /// change the options whichever form they were given in, use
    /// [`resolve_option_mut`](Self::resolve_option_mut).
    pub fn option_mut<T: Clone + Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        self.options.get_mut::<T>()
    }

    /// Resolve provider options of type `T` for changing in place: the
    /// typed value, else the JSON form, else `T::default()`, stored as the
    /// typed value. Use it to add to the caller's options without
    /// replacing them, as [`with_option`](Self::with_option) would.
    ///
    /// # Errors
    ///
    /// See [`RequestOptions::resolve_mut`].
    pub fn resolve_option_mut<T>(&mut self) -> Result<&mut T>
    where
        T: ProviderOptions + Serialize + DeserializeOwned + Default,
    {
        self.options.resolve_mut::<T>()
    }
}

/// What embedding inputs are for in asymmetric retrieval.
///
/// Retrieval models can embed a search query and the documents it searches
/// differently. Only this distinction is portable; other tasks
/// (classification, clustering, similarity) stay provider options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum EmbeddingInputType {
    /// A search query, embedded to retrieve documents.
    Query,
    /// A document, embedded to be retrieved by queries.
    Document,
}

enum_names!(EmbeddingInputType, "embedding input type", {
    Query => "query",
    Document => "document",
});

/// Request limits of an embedding model, as reported by
/// [`EmbeddingProvider::embedding_limits`](crate::EmbeddingProvider::embedding_limits).
///
/// `None` means the provider reports no limit. New limits (tokens per
/// input, tokens per batch) become fields, so a wrapper that forwards
/// `embedding_limits` forwards them too.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EmbeddingLimits {
    /// Maximum inputs per request.
    /// [`embed_chunked`](crate::EmbeddingProviderExt::embed_chunked) splits
    /// larger batches into requests of at most this many inputs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_inputs_per_call: Option<usize>,
}

impl EmbeddingLimits {
    /// No known limits.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum inputs per request.
    #[must_use]
    pub fn max_inputs_per_call(mut self, max: usize) -> Self {
        self.max_inputs_per_call = Some(max);
        self
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    #[test]
    fn options_change_in_place() {
        #[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
        struct Tags(Vec<String>);

        impl ProviderOptions for Tags {
            const NAMESPACE: &'static str = "tags";
        }

        let mut request = EmbeddingRequest::new("m").with_options_json("tags", json!(["caller"]));
        assert!(request.option_mut::<Tags>().is_none());
        request
            .resolve_option_mut::<Tags>()
            .unwrap()
            .0
            .push("wrapper".into());
        request.option_mut::<Tags>().unwrap().0.push("more".into());
        assert_eq!(
            request.option::<Tags>().unwrap().0,
            ["caller", "wrapper", "more"]
        );
    }

    use super::*;
    use crate::{ProviderData, Text};

    fn provider_of(err: &Error) -> Option<&str> {
        err.context()?.provider.as_deref()
    }

    fn with_data(input: EmbeddingInput, namespace: &str, key: &str) -> EmbeddingInput {
        input.with_provider_data(ProviderData::new().with_raw(namespace, key, json!("v")))
    }

    #[test]
    fn every_input_form_compiles_through_the_builders() {
        let owned = vec![String::from("a"), String::from("b")];
        let expected = EmbeddingRequest::new("m").inputs(["a", "b"]);
        assert_eq!(EmbeddingRequest::new("m").inputs(vec!["a", "b"]), expected);
        assert_eq!(EmbeddingRequest::new("m").inputs(owned.clone()), expected);
        assert_eq!(EmbeddingRequest::new("m").inputs(&owned), expected);
        assert_eq!(EmbeddingRequest::new("m").inputs(owned.iter()), expected);
        assert_eq!(
            EmbeddingRequest::new("m").inputs(owned.iter().cloned()),
            expected
        );
        assert_eq!(
            EmbeddingRequest::new("m").inputs(owned.iter().map(String::as_str)),
            expected
        );
        assert_eq!(
            EmbeddingRequest::new("m").input(&owned[0]).input("b"),
            expected
        );
    }

    #[test]
    fn requests_round_trip_with_bare_string_inputs() {
        let request = EmbeddingRequest::new("m").inputs(["a", "b"]).dimensions(8);
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(
            value,
            json!({"model": "m", "inputs": ["a", "b"], "dimensions": 8})
        );
        assert_eq!(
            serde_json::from_value::<EmbeddingRequest>(value).unwrap(),
            request
        );
    }

    #[test]
    fn requests_with_combined_inputs_round_trip() {
        let request = EmbeddingRequest::new("m")
            .input("plain")
            .input(vec![EmbeddingPart::text("a"), EmbeddingPart::text("b")])
            .input(
                EmbeddingInput::text("doc").with_provider_data(ProviderData::new().with_raw(
                    "demo",
                    "title",
                    json!("T"),
                )),
            )
            .with_options_json("demo", json!({"task": "x"}));
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(
            value["inputs"],
            json!([
                "plain",
                {"parts": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
                {"parts": [{"type": "text", "text": "doc"}], "provider_data": {"demo": {"title": "T"}}}
            ])
        );
        assert_eq!(
            serde_json::from_value::<EmbeddingRequest>(value).unwrap(),
            request
        );
    }

    #[test]
    fn input_type_round_trips_and_is_omitted_when_unset() {
        let request = EmbeddingRequest::new("m")
            .input("a")
            .input_type(EmbeddingInputType::Query);
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(
            value,
            json!({"model": "m", "inputs": ["a"], "input_type": "query"})
        );
        assert_eq!(
            serde_json::from_value::<EmbeddingRequest>(value).unwrap(),
            request
        );
        let document = serde_json::to_value(EmbeddingInputType::Document).unwrap();
        assert_eq!(document, json!("document"));
        assert_eq!(EmbeddingInputType::Document.to_string(), "document");
        assert!(serde_json::from_value::<EmbeddingInputType>(json!("passage")).is_err());

        let plain = serde_json::to_value(EmbeddingRequest::new("m").input("a")).unwrap();
        assert!(plain.get("input_type").is_none());
    }

    #[test]
    fn text_inputs_preserves_order() {
        let request = EmbeddingRequest::new("m").inputs(["a", "b", "c"]);
        assert_eq!(request.text_inputs("demo").unwrap(), vec!["a", "b", "c"]);
    }

    #[test]
    fn text_inputs_rejects_an_empty_request() {
        let err = EmbeddingRequest::new("m").text_inputs("demo").unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }));
        assert_eq!(provider_of(&err), Some("demo"));
    }

    #[test]
    fn text_inputs_rejects_an_input_without_parts() {
        let request = EmbeddingRequest::new("m")
            .input("a")
            .input(EmbeddingInput::from_parts([]));
        let err = request.text_inputs("demo").unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }));
        assert_eq!(provider_of(&err), Some("demo"));
        assert_eq!(
            err.to_string(),
            "demo: invalid request: embedding input 1 has no parts"
        );
    }

    #[test]
    fn text_inputs_rejects_several_parts_under_every_policy() {
        for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
            let request = EmbeddingRequest::new("m")
                .input(vec![EmbeddingPart::text("a"), EmbeddingPart::text("b")])
                .unsupported_policy(policy);
            let err = request.text_inputs("demo").unwrap_err();
            assert!(matches!(err, Error::Unsupported { .. }));
            assert_eq!(provider_of(&err), Some("demo"));
            assert_eq!(
                err.to_string(),
                "demo: unsupported: cannot send embedding input 0: model `m` embeds one part per input, \
                 and this input has 2 parts; pass separate inputs with `.inputs([..])` to embed \
                 them separately"
            );
        }
    }

    #[test]
    fn text_inputs_reports_images_before_part_counts() {
        for input in [
            EmbeddingInput::image_url("https://example.com/a.png"),
            EmbeddingInput::from(vec![
                EmbeddingPart::text("a"),
                EmbeddingPart::image_url("https://example.com/a.png"),
            ]),
        ] {
            let err = EmbeddingRequest::new("m")
                .input(input)
                .text_inputs("demo")
                .unwrap_err();
            assert!(matches!(err, Error::Unsupported { .. }));
            assert!(
                err.to_string().ends_with(": model `m` embeds text only"),
                "{err}"
            );
            assert!(matches!(
                err.unsupported_subject(),
                Some(Subject::EmbeddingInput {
                    input: 0,
                    part: Some(_)
                })
            ));
        }
    }

    #[test]
    fn text_inputs_accepts_a_runtime_provider_name() {
        let request = EmbeddingRequest::new("m")
            .input(vec![EmbeddingPart::text("a"), EmbeddingPart::text("b")]);
        let name = String::from("openai-compat:groq");
        let err = request.text_inputs(&name).unwrap_err();
        drop(name);
        assert_eq!(provider_of(&err), Some("openai-compat:groq"));
    }

    #[test]
    fn check_provider_data_accepts_declared_foreign_and_part_keys() {
        let request = EmbeddingRequest::new("m")
            .input(with_data(EmbeddingInput::text("a"), "demo", "title"))
            .input(with_data(EmbeddingInput::text("b"), "other", "anything"))
            .input(EmbeddingInput::from(Text::new("c").with_provider_data(
                ProviderData::new().with_raw("demo", "thought_signature", json!("s")),
            )));
        request
            .check_provider_data("p", "demo", &["title"])
            .unwrap();

        let mut empty = ProviderData::new();
        empty.namespace_mut("demo");
        EmbeddingRequest::new("m")
            .input(EmbeddingInput::text("a").with_provider_data(empty))
            .check_provider_data("p", "demo", &[])
            .unwrap();
    }

    #[test]
    fn check_provider_data_rejects_unknown_input_keys() {
        let request = EmbeddingRequest::new("m").input("a").input(with_data(
            EmbeddingInput::text("b"),
            "demo",
            "titel",
        ));
        let name = String::from("openai-compat:groq");
        let err = request
            .check_provider_data(&name, "demo", &["title"])
            .unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }));
        assert_eq!(provider_of(&err), Some("openai-compat:groq"));
        assert_eq!(
            err.to_string(),
            "openai-compat:groq: invalid request: embedding input 1 has unknown `demo` provider data \
             key `titel`"
        );

        // Adapters without embedding keys reject any key in their namespace.
        let err = EmbeddingRequest::new("m")
            .input(with_data(EmbeddingInput::text("a"), "demo", "title"))
            .check_provider_data("p", "demo", &[])
            .unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }));
    }

    #[test]
    fn check_provider_data_rejects_input_keys_on_parts() {
        let request = EmbeddingRequest::new("m").input(EmbeddingInput::from(
            Text::new("a").with_provider_data(ProviderData::new().with_raw(
                "demo",
                "title",
                json!("t"),
            )),
        ));
        let err = request
            .check_provider_data("p", "demo", &["title"])
            .unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }));
        assert_eq!(provider_of(&err), Some("p"));
        assert_eq!(
            err.to_string(),
            "p: invalid request: embedding input 0, part 0 has `demo` key `title`, which belongs on \
             the input"
        );
    }
}
