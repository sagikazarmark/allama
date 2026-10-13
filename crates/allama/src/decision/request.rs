use std::collections::HashSet;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::named;
use super::{DecisionInput, Question};
use crate::{Error, Image, ProviderOptions, RequestOptions, Result};

/// Named questions about one input, answered by a decision model.
///
/// Every field is semantic: a provider either sends it as given or fails.
/// There are no hints to degrade, so unlike chat and embedding requests a
/// decision request carries no [`UnsupportedPolicy`](crate::UnsupportedPolicy).
///
/// ```
/// use allama::DecisionRequest;
/// use allama::decision::Question;
///
/// let request = DecisionRequest::new("nimble", "I was charged twice. Please refund me.")
///     .question(
///         "label",
///         Question::choice("Which label fits this ticket?")
///             .option("billing", "Payments and refunds")
///             .option("bug", "Software errors"),
///     )
///     .question("refund", Question::yes_no("Is the customer requesting a refund?"));
/// assert_eq!(request.questions.len(), 2);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DecisionRequest {
    /// Model identifier.
    pub model: String,
    /// What the questions are about.
    pub input: DecisionInput,
    /// Images the model reads together with the input, in order, for
    /// models that report
    /// [`DecisionCapability::ImageInput`](crate::DecisionCapability::ImageInput).
    /// Every question is about the input and the images alike.
    ///
    /// [`Image::detail`] is a hint no decision API honors: adapters warn
    /// about it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<Image>,
    /// Questions by unique name, in order.
    #[serde(with = "named")]
    pub questions: Vec<(String, Question)>,
    /// Provider-specific options.
    #[serde(default, skip_serializing_if = "RequestOptions::is_empty")]
    pub options: RequestOptions,
}

impl DecisionRequest {
    /// A request for `model` about `input`, with no questions yet.
    #[must_use]
    pub fn new(model: impl Into<String>, input: impl Into<DecisionInput>) -> Self {
        Self {
            model: model.into(),
            input: input.into(),
            images: Vec::new(),
            questions: Vec::new(),
            options: RequestOptions::new(),
        }
    }

    /// Append an image the model reads together with the input.
    ///
    /// ```
    /// use allama::{DecisionRequest, Image};
    /// use allama::decision::Question;
    ///
    /// # let png_bytes: &[u8] = b"\x89PNG";
    /// let request = DecisionRequest::new("@cf/cloudflare/clef", "A product photo from a listing.")
    ///     .image(Image::bytes("image/png", png_bytes))
    ///     .question("damaged", Question::yes_no("Does the product look damaged?"));
    /// assert_eq!(request.images.len(), 1);
    /// ```
    #[must_use]
    pub fn image(mut self, image: impl Into<Image>) -> Self {
        self.images.push(image.into());
        self
    }

    /// Append a question named `name`. Names must be unique.
    #[must_use]
    pub fn question(mut self, name: impl Into<String>, question: impl Into<Question>) -> Self {
        self.questions.push((name.into(), question.into()));
        self
    }

    /// Append several named questions.
    #[must_use]
    pub fn questions<I, N, Q>(mut self, questions: I) -> Self
    where
        I: IntoIterator<Item = (N, Q)>,
        N: Into<String>,
        Q: Into<Question>,
    {
        self.questions.extend(
            questions
                .into_iter()
                .map(|(name, question)| (name.into(), question.into())),
        );
        self
    }

    /// The question named `name`, if any.
    #[must_use]
    pub fn find_question(&self, name: &str) -> Option<&Question> {
        named::get(&self.questions, name)
    }

    /// Check the request's structure and `limits` before sending it.
    ///
    /// Providers call this through
    /// [`validate_decision_request`](crate::DecisionProvider::validate_decision_request)
    /// before any I/O.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`] when:
    ///
    /// - the model, input, a question name, instructions, an option key, a
    ///   level label, or a description is blank;
    /// - question names, option keys, or level labels repeat within their
    ///   scope;
    /// - a choice has fewer than two options or a score fewer than two
    ///   levels;
    /// - the request exceeds `limits`, its images included.
    pub fn validate(&self, limits: &DecisionLimits) -> Result<()> {
        if self.model.trim().is_empty() {
            return Err(invalid("model must not be blank"));
        }
        if self.input.is_blank() {
            return Err(invalid("input must not be blank"));
        }
        if self.questions.is_empty() {
            return Err(invalid("at least one question is required"));
        }
        check_max(self.questions.len(), limits.max_questions, || {
            "questions".into()
        })?;
        check_max(self.images.len(), limits.max_images, || "images".into())?;

        let mut names = HashSet::new();
        for (name, question) in &self.questions {
            if is_blank(name) {
                return Err(invalid("question names must not be blank"));
            }
            if !names.insert(name.as_str()) {
                return Err(invalid(format!("duplicate question name {name:?}")));
            }
            let fail = |problem: String| invalid(format!("question {name:?}: {problem}"));
            if is_blank(question.instructions()) {
                return Err(fail("instructions must not be blank".into()));
            }
            match question {
                Question::YesNo(question) => {
                    if [&question.yes, &question.no]
                        .into_iter()
                        .flatten()
                        .any(|description| is_blank(description))
                    {
                        return Err(fail("outcome descriptions must not be blank".into()));
                    }
                }
                Question::Choice(choice) => {
                    check_entries(
                        choice
                            .options
                            .iter()
                            .map(|option| (option.key.as_str(), option.description.as_deref())),
                        "option",
                        "key",
                        "options",
                    )
                    .map_err(fail)?;
                    check_max(choice.options.len(), limits.max_choice_options, || {
                        format!("options in question {name:?}")
                    })?;
                }
                Question::Score(score) => {
                    check_entries(
                        score
                            .levels
                            .iter()
                            .map(|level| (level.label.as_str(), level.description.as_deref())),
                        "level",
                        "label",
                        "levels",
                    )
                    .map_err(fail)?;
                    check_max(score.levels.len(), limits.max_score_levels, || {
                        format!("levels in question {name:?}")
                    })?;
                }
            }
        }
        Ok(())
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

/// Request limits of a decision model, as reported by
/// [`DecisionProvider::decision_limits`](crate::DecisionProvider::decision_limits).
///
/// `None` means the provider reports no limit. Limits differ widely between
/// providers (for example 26 versus 255 choice options), which is why they
/// are queried rather than fixed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DecisionLimits {
    /// Maximum questions per request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_questions: Option<usize>,
    /// Maximum options per choice question.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_choice_options: Option<usize>,
    /// Maximum levels per score question.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_score_levels: Option<usize>,
    /// Maximum images per request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_images: Option<usize>,
}

impl DecisionLimits {
    /// No known limits.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum questions per request.
    #[must_use]
    pub fn max_questions(mut self, max: usize) -> Self {
        self.max_questions = Some(max);
        self
    }

    /// Set the maximum options per choice question.
    #[must_use]
    pub fn max_choice_options(mut self, max: usize) -> Self {
        self.max_choice_options = Some(max);
        self
    }

    /// Set the maximum levels per score question.
    #[must_use]
    pub fn max_score_levels(mut self, max: usize) -> Self {
        self.max_score_levels = Some(max);
        self
    }

    /// Set the maximum images per request.
    #[must_use]
    pub fn max_images(mut self, max: usize) -> Self {
        self.max_images = Some(max);
        self
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::invalid_request(message).authored()
}

fn is_blank(text: &str) -> bool {
    text.trim().is_empty()
}

/// Check the `(name, description)` entries of a choice (keys) or score
/// (labels): at least two, names non-blank and unique, descriptions non-blank
/// when present.
fn check_entries<'a>(
    entries: impl ExactSizeIterator<Item = (&'a str, Option<&'a str>)>,
    entry: &str,
    name: &str,
    plural: &str,
) -> std::result::Result<(), String> {
    let count = entries.len();
    if count < 2 {
        return Err(format!("at least two {plural} are required, found {count}"));
    }
    let mut keys = HashSet::new();
    for (key, description) in entries {
        if is_blank(key) {
            return Err(format!("{entry} {name}s must not be blank"));
        }
        if !keys.insert(key) {
            return Err(format!("duplicate {entry} {key:?}"));
        }
        if description.is_some_and(is_blank) {
            return Err(format!(
                "the description of {entry} {key:?} must not be blank"
            ));
        }
    }
    Ok(())
}

fn check_max(count: usize, max: Option<usize>, what: impl FnOnce() -> String) -> Result<()> {
    match max {
        Some(max) if count > max => Err(invalid(format!(
            "too many {}: {count}, the model accepts at most {max}",
            what()
        ))),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn options_change_in_place() {
        #[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
        struct Tags(Vec<String>);

        impl ProviderOptions for Tags {
            const NAMESPACE: &'static str = "tags";
        }

        let mut request = request().with_options_json("tags", json!(["caller"]));
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

    fn request() -> DecisionRequest {
        DecisionRequest::new("m", "ticket")
            .question(
                "label",
                Question::choice("Label?").bare_option("a").bare_option("b"),
            )
            .question("spam", Question::yes_no("Spam?"))
            .question(
                "urgency",
                Question::score("Urgency?").levels(["low", "high"]),
            )
    }

    fn message(request: DecisionRequest) -> String {
        match request.validate(&DecisionLimits::new()).unwrap_err() {
            Error::InvalidRequest { message, .. } => message,
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    fn one(question: impl Into<Question>) -> DecisionRequest {
        DecisionRequest::new("m", "x").question("q", question)
    }

    #[test]
    fn valid_request_passes() {
        request().validate(&DecisionLimits::new()).unwrap();
    }

    #[test]
    fn structure_is_checked() {
        assert!(message(DecisionRequest::new("m", "x")).contains("at least one question"));
        assert!(
            message(DecisionRequest::new("m", " ").question("q", Question::yes_no("?")))
                .contains("input")
        );
        // A `null` value is no input, not the text `null`.
        assert_eq!(
            message(DecisionRequest::new("m", json!(null)).question("q", Question::yes_no("?"))),
            "input must not be blank"
        );
        assert!(
            message(request().question("spam", Question::yes_no("Again?")))
                .contains("duplicate question")
        );
        assert!(
            message(one(Question::choice("?").bare_option("only")))
                .contains("at least two options")
        );
        assert!(
            message(one(Question::choice("?").bare_option("a").bare_option("a")))
                .contains("duplicate option \"a\"")
        );
        assert!(message(one(Question::yes_no(""))).contains("instructions"));
        assert!(
            message(one(Question::score("?")))
                .contains("at least two levels are required, found 0")
        );
        assert!(message(one(Question::score("?").levels(["x", "x"]))).contains("duplicate level"));
    }

    #[test]
    fn blank_descriptions_are_rejected() {
        assert!(
            message(one(Question::choice("?").option("a", " ").bare_option("b")))
                .contains("description of option \"a\"")
        );
        assert!(
            message(one(Question::score("?")
                .level("low", "")
                .bare_level("high")))
            .contains("description of level \"low\"")
        );
        assert!(message(one(Question::yes_no("?").yes("  "))).contains("outcome descriptions"));
    }

    #[test]
    fn limits_are_enforced() {
        let limits = DecisionLimits::new().max_questions(2);
        assert!(request().validate(&limits).is_err());

        let limits = DecisionLimits::new().max_choice_options(1);
        let err = request().validate(&limits).unwrap_err();
        assert!(
            err.message().contains("options in question \"label\""),
            "{err}"
        );

        let limits = DecisionLimits::new().max_score_levels(1);
        let err = request().validate(&limits).unwrap_err();
        assert!(err.message().contains("levels"), "{err}");

        let limits = DecisionLimits::new().max_images(1);
        let two = request()
            .image(Image::base64("image/png", "aGk="))
            .image(Image::base64("image/png", "aGk="));
        let err = two.validate(&limits).unwrap_err();
        assert!(err.message().contains("too many images: 2"), "{err}");
        two.validate(&DecisionLimits::new().max_images(2)).unwrap();
    }

    #[test]
    fn images_serialize_only_when_present() {
        assert!(
            serde_json::to_value(request())
                .unwrap()
                .get("images")
                .is_none()
        );
        let request = request().image(Image::base64("image/png", "aGk="));
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(
            value["images"],
            json!([{"source": {"type": "base64", "media_type": "image/png", "data": "aGk="}}])
        );
        let back: DecisionRequest = serde_json::from_value(value).unwrap();
        assert_eq!(back, request);
    }

    #[test]
    fn questions_serialize_in_order() {
        let text = serde_json::to_string(&request()).unwrap();
        let (label, spam, urgency) = (
            text.find("\"label\"").unwrap(),
            text.find("\"spam\"").unwrap(),
            text.find("\"urgency\"").unwrap(),
        );
        assert!(label < spam && spam < urgency);
        let back: DecisionRequest = serde_json::from_str(&text).unwrap();
        assert_eq!(back, request());
        assert_eq!(
            serde_json::to_value(request()).unwrap()["input"],
            json!("ticket")
        );
    }
}
