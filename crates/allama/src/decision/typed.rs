//! Typed decisions: questions described by Rust types, answers decoded into
//! them.

use std::fmt;
use std::marker::PhantomData;

use super::response::most_likely_level;
use super::{
    Answer, DecisionInput, DecisionRequest, DecisionResponse, Question, QuestionKind, YesNoAnswer,
};
use crate::{Error, Result};

/// A struct whose fields are questions, answered by one decision request.
///
/// Derive it with `#[derive(Decision)]` (feature `derive`):
///
/// ```
/// # #[cfg(feature = "derive")] {
/// use allama::decision::{Choice, ChoiceSet, Decision, Scale, Score, YesNoAnswer};
///
/// #[derive(ChoiceSet)]
/// #[decision(rename_all = "lowercase")]
/// enum Label {
///     #[decision(description = "Payments and refunds")]
///     Billing,
///     #[decision(description = "Software errors")]
///     Bug,
/// }
///
/// #[derive(Scale)]
/// enum Urgency {
///     /// No time pressure
///     Routine,
///     #[decision(description = "A customer is inconvenienced")]
///     Soon,
///     #[decision(description = "A critical service is unavailable")]
///     Immediate,
/// }
///
/// #[derive(Decision)]
/// struct Triage {
///     #[decision(instructions = "Which label fits this ticket?")]
///     label: Choice<Label>,
///     #[decision(
///         instructions = "Is the customer requesting a refund?",
///         yes = "The customer requests a refund"
///     )]
///     refund: YesNoAnswer,
///     /// How urgently does this ticket need a response?
///     urgency: Score<Urgency>,
/// }
///
/// let request = Triage::request("nimble", "I was charged twice.");
/// assert_eq!(request.questions.len(), 3);
/// # }
/// ```
///
/// Field types implement [`DecisionField`]: [`YesNoAnswer`], [`Choice<T>`],
/// [`Score<T>`], or an enum deriving [`ChoiceSet`] (which keeps only the
/// chosen option).
///
/// # Attributes
///
/// All attributes live under `#[decision(...)]`.
///
/// | Derive | On the type | On a variant or field |
/// |---|---|---|
/// | `Decision` | `crate = "path"` | `instructions = "..."` (required unless the field has a doc comment), `rename = "..."` (question name; default: the field name), `yes = "..."` / `no = "..."` (outcome descriptions; yes/no fields only) |
/// | `ChoiceSet` | `rename_all = "..."`, `crate = "path"` | `rename = "..."` (option key), `description = "..."` |
/// | `Scale` | `rename_all = "..."`, `crate = "path"` | `rename = "..."` (level label), `description = "..."` |
///
/// - Doc comments stand in for a missing `description` on a variant and a
///   missing `instructions` on a field. Only the first paragraph is used,
///   up to the first blank line: its lines are trimmed and joined with
///   spaces, and later paragraphs (details, examples, rustdoc links) stay
///   for Rust readers. An explicit attribute always wins, so use one when
///   the first paragraph is written for Rust readers rather than the model.
///   A field with neither is a compile error; a variant with neither has no
///   description.
/// - Keys and labels default to the variant name, verbatim. `rename_all`
///   accepts `lowercase`, `UPPERCASE`, `snake_case`, `kebab-case`,
///   `SCREAMING_SNAKE_CASE`, `camelCase`, and `PascalCase` (word boundaries
///   follow `heck`: `HTTPError` becomes `http_error`).
/// - `ChoiceSet` and `Scale` need at least two unit variants; levels go
///   lowest first. `Decision` needs at least one named field.
/// - `crate` names the path to `allama` when it is renamed or re-exported
///   (default `::allama`).
/// - Provider limits (option and level counts) are not checked at compile
///   time; they depend on the provider and model.
pub trait Decision: Sized {
    /// The questions, by name, in field order.
    fn questions() -> Vec<(String, Question)>;

    /// Decode the answers.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnexpectedResponse`] when an answer is missing or
    /// does not fit its field.
    fn from_response(response: &DecisionResponse) -> Result<Self>;

    /// A request asking these questions about `input`.
    #[must_use]
    fn request(model: impl Into<String>, input: impl Into<DecisionInput>) -> DecisionRequest {
        DecisionRequest::new(model, input).questions(Self::questions())
    }
}

/// A typed decision and the response it was decoded from, as returned by
/// [`DecisionProviderExt::decide_as`](super::DecisionProviderExt::decide_as).
///
/// The response keeps what the typed value does not: usage, the model and
/// request id, warnings, and provider data such as a provider's confidence
/// statistic.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Decided<T> {
    /// The decoded answers.
    pub value: T,
    /// The provider response they were decoded from.
    pub response: DecisionResponse,
}

/// A field of a [`Decision`]: one question and the decoding of its answer.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be a decision question",
    label = "not a decision field",
    note = "use `YesNoAnswer`, `Choice<T>`, `Score<T>`, or an enum deriving `ChoiceSet`"
)]
pub trait DecisionField: Sized {
    /// The kind of question this field asks.
    const KIND: QuestionKind;

    /// The question, with `instructions`.
    fn question(instructions: impl Into<String>) -> Question;

    /// Decode the answer.
    ///
    /// # Errors
    ///
    /// Fails when the answer has another kind or does not fit the type.
    fn from_answer(answer: &Answer) -> std::result::Result<Self, AnswerMismatch>;
}

/// Decode the answer to the question named `name` as `T`.
///
/// This is what `#[derive(Decision)]` calls for each field.
///
/// # Errors
///
/// Returns [`Error::UnexpectedResponse`] when the question was not answered
/// or the answer does not fit `T`.
pub fn answer_as<T: DecisionField>(response: &DecisionResponse, name: &str) -> Result<T> {
    let answer = response.answer(name).ok_or_else(|| {
        Error::unexpected_response(format!("decision question {name:?} was not answered"))
    })?;
    T::from_answer(answer).map_err(|mismatch| {
        Error::unexpected_response(format!("decision question {name:?}: {mismatch}"))
    })
}

/// Why an answer does not fit a [`DecisionField`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnswerMismatch(String);

impl AnswerMismatch {
    /// A mismatch described by `message`.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    fn kind(expected: QuestionKind, answer: &Answer) -> Self {
        Self(format!(
            "expected a {expected} answer, found a {} answer",
            answer.kind()
        ))
    }
}

impl fmt::Display for AnswerMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AnswerMismatch {}

impl DecisionField for YesNoAnswer {
    const KIND: QuestionKind = QuestionKind::YesNo;

    fn question(instructions: impl Into<String>) -> Question {
        Question::yes_no(instructions).into()
    }

    fn from_answer(answer: &Answer) -> std::result::Result<Self, AnswerMismatch> {
        match answer {
            Answer::YesNo(answer) => Ok(*answer),
            other => Err(AnswerMismatch::kind(Self::KIND, other)),
        }
    }
}

/// An enum of unit variants whose variants are the options of a choice
/// question. Derive it with `#[derive(ChoiceSet)]` (feature `derive`),
/// which also implements [`DecisionField`] for the enum (see [`Decision`]
/// for the attributes).
///
/// A choice set is a type, not a collection: its variants are the options
/// (the alternatives the model picks from, as in decision theory), and a
/// value is one option. The answer to a choice question over it is a
/// [`Choice<T>`].
///
/// A handwritten implementation must keep [`OPTIONS`](Self::OPTIONS),
/// [`from_index`](Self::from_index), and [`index`](Self::index) consistent.
#[doc(alias = "Choices", alias = "Options")]
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a set of choice options",
    note = "derive `ChoiceSet` for an enum of unit variants"
)]
pub trait ChoiceSet: Sized + 'static {
    /// `(key, description)` of every option, in variant order.
    const OPTIONS: &'static [(&'static str, Option<&'static str>)];

    /// The variant at `index` in [`OPTIONS`](Self::OPTIONS).
    fn from_index(index: usize) -> Option<Self>;

    /// This variant's index in [`OPTIONS`](Self::OPTIONS).
    fn index(&self) -> usize;

    /// This variant's option key.
    fn key(&self) -> &'static str {
        Self::OPTIONS.get(self.index()).map_or("", |(key, _)| key)
    }

    /// The variant whose option key is `key`.
    fn from_key(key: &str) -> Option<Self> {
        Self::OPTIONS
            .iter()
            .position(|(option, _)| *option == key)
            .and_then(Self::from_index)
    }
}

/// The answer to a choice question over `T`: the chosen option and the
/// probability of every option.
///
/// This is the field type, not a derive: derive [`ChoiceSet`] on the enum of
/// options, then use `Choice<Label>` as the field to keep the distribution,
/// or `Label` alone to keep only the chosen option.
///
/// The chosen option is the one the provider chose. Providers report it
/// alongside the probabilities, and allama does not renormalize them or
/// recompute the choice, so it may differ from the option with the highest
/// probability.
#[derive(Debug, Clone, PartialEq)]
pub struct Choice<T> {
    value: T,
    probabilities: Vec<f64>,
}

impl<T: ChoiceSet> Choice<T> {
    /// A choice of `value`, with the probability of each option. Options
    /// left out get probability 0, and a repeated option keeps its last
    /// probability. Useful to build expected values in tests.
    #[must_use]
    pub fn new(value: T, probabilities: impl IntoIterator<Item = (T, f64)>) -> Self {
        let mut slots = vec![0.0; T::OPTIONS.len()];
        for (option, probability) in probabilities {
            if let Some(slot) = slots.get_mut(option.index()) {
                *slot = probability;
            }
        }
        Self {
            value,
            probabilities: slots,
        }
    }

    /// The chosen option, as the provider reported it (see the type docs).
    #[must_use]
    pub fn value(&self) -> &T {
        &self.value
    }

    /// The chosen option, by value.
    #[must_use]
    pub fn into_value(self) -> T {
        self.value
    }

    /// The probability of `option`.
    #[must_use]
    pub fn probability(&self, option: &T) -> f64 {
        self.probabilities
            .get(option.index())
            .copied()
            .unwrap_or_default()
    }

    /// Every option with its probability, in variant order.
    pub fn probabilities(&self) -> impl Iterator<Item = (T, f64)> + '_ {
        self.probabilities
            .iter()
            .enumerate()
            .filter_map(|(index, &p)| Some((T::from_index(index)?, p)))
    }
}

impl<T: ChoiceSet> DecisionField for Choice<T> {
    const KIND: QuestionKind = QuestionKind::Choice;

    fn question(instructions: impl Into<String>) -> Question {
        T::OPTIONS
            .iter()
            .fold(
                Question::choice(instructions),
                |question, (key, description)| match description {
                    Some(description) => question.option(*key, *description),
                    None => question.bare_option(*key),
                },
            )
            .into()
    }

    fn from_answer(answer: &Answer) -> std::result::Result<Self, AnswerMismatch> {
        let Answer::Choice(answer) = answer else {
            return Err(AnswerMismatch::kind(Self::KIND, answer));
        };
        if let Some(missing) = (0..T::OPTIONS.len()).find(|&index| T::from_index(index).is_none()) {
            return Err(AnswerMismatch::new(format!(
                "`ChoiceSet::from_index` has no variant for option {missing}"
            )));
        }
        let value = T::from_key(&answer.chosen)
            .ok_or_else(|| AnswerMismatch::new(format!("unknown option {:?}", answer.chosen)))?;
        let mut slots = vec![None; T::OPTIONS.len()];
        for (key, probability) in &answer.probabilities {
            let slot = T::from_key(key)
                .and_then(|option| slots.get_mut(option.index()))
                .ok_or_else(|| {
                    AnswerMismatch::new(format!("probability for unknown option {key:?}"))
                })?;
            if slot.replace(*probability).is_some() {
                return Err(AnswerMismatch::new(format!(
                    "option {key:?} reported more than once"
                )));
            }
        }
        let probabilities = slots
            .into_iter()
            .zip(T::OPTIONS)
            .map(|(probability, (key, _))| {
                probability.ok_or_else(|| {
                    AnswerMismatch::new(format!("no probability for option {key:?}"))
                })
            })
            .collect::<std::result::Result<_, _>>()?;
        Ok(Self {
            value,
            probabilities,
        })
    }
}

/// An enum of unit variants whose variants are the levels of a score
/// question, lowest first. Derive it with `#[derive(Scale)]` (feature
/// `derive`; see [`Decision`] for the attributes).
///
/// Together the levels form an ordered scale: a variant's position is its
/// level number, from 0, so reordering the variants changes what every
/// score means. The answer to a score question over it is a [`Score<T>`].
///
/// The members fit together by level number. For a variant `x`:
/// `x.level()` is its number, `Self::from_level(x.level())` is `x` again,
/// `Self::LEVELS[x.level()]` is its `(label, description)`, and
/// [`x.label()`](Self::label) is that label.
///
/// A handwritten implementation must keep [`LEVELS`](Self::LEVELS),
/// [`from_level`](Self::from_level), and [`level`](Self::level) consistent.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a scale of levels",
    note = "derive `Scale` for an enum of unit variants, lowest level first"
)]
pub trait Scale: Sized + 'static {
    /// `(label, description)` of every level, lowest first.
    const LEVELS: &'static [(&'static str, Option<&'static str>)];

    /// The variant at `level`.
    fn from_level(level: usize) -> Option<Self>;

    /// This variant's level (0 is the lowest).
    fn level(&self) -> usize;

    /// This variant's label.
    fn label(&self) -> &'static str {
        Self::LEVELS
            .get(self.level())
            .map_or("", |(label, _)| label)
    }
}

/// The answer to a score question over `T`: the expected level and the
/// probability of every level.
///
/// This is the field type, not a derive: derive [`Scale`] on the enum of
/// levels, then use `Score<Urgency>` as the field.
pub struct Score<T> {
    score: f64,
    probabilities: Vec<f64>,
    scale: PhantomData<fn() -> T>,
}

impl<T: Scale> Score<T> {
    /// A score from the probability of each level; levels left out get
    /// probability 0, a repeated level keeps its last probability, and the
    /// score is their probability-weighted average. Useful to build expected
    /// values in tests; compare decoded scores with a tolerance, since
    /// providers round.
    #[must_use]
    pub fn new(probabilities: impl IntoIterator<Item = (T, f64)>) -> Self {
        let mut slots = vec![0.0; T::LEVELS.len()];
        for (level, probability) in probabilities {
            if let Some(slot) = slots.get_mut(level.level()) {
                *slot = probability;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let score = slots
            .iter()
            .enumerate()
            .map(|(level, p)| level as f64 * p)
            .sum();
        Self {
            score,
            probabilities: slots,
            scale: PhantomData,
        }
    }

    /// Probability-weighted average of the level numbers, from 0 to the number of
    /// levels minus 1, as the provider reported it. Not rounded to a level.
    #[must_use]
    pub fn score(&self) -> f64 {
        self.score
    }

    /// The single most likely level; ties go to the lower level.
    ///
    /// `None` when every probability is zero or any is NaN: such a
    /// distribution names no level, and reporting the lowest one would read
    /// as a confident answer that was never given.
    #[must_use]
    pub fn most_likely(&self) -> Option<T> {
        most_likely_level(&self.probabilities).map(Self::at)
    }

    /// The level nearest to [`score`](Self::score), clamped to the scale.
    ///
    /// The score is an average, so this can differ from
    /// [`most_likely`](Self::most_likely) and even name a level the model
    /// gave probability 0: a model split between the lowest and highest
    /// level averages to the middle. Check [`probability`](Self::probability)
    /// before acting on it.
    #[must_use]
    pub fn nearest(&self) -> T {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let level =
            (self.score.round().max(0.0) as usize).min(self.probabilities.len().saturating_sub(1));
        Self::at(level)
    }

    /// The probability of `level`.
    #[must_use]
    pub fn probability(&self, level: &T) -> f64 {
        self.probabilities
            .get(level.level())
            .copied()
            .unwrap_or_default()
    }

    /// Every level with its probability, lowest first.
    pub fn probabilities(&self) -> impl Iterator<Item = (T, f64)> + '_ {
        self.probabilities
            .iter()
            .enumerate()
            .filter_map(|(level, &p)| Some((T::from_level(level)?, p)))
    }

    fn at(level: usize) -> T {
        T::from_level(level).expect("`Scale::from_level` covers every index of `Scale::LEVELS`")
    }
}

impl<T: Scale> DecisionField for Score<T> {
    const KIND: QuestionKind = QuestionKind::Score;

    fn question(instructions: impl Into<String>) -> Question {
        T::LEVELS
            .iter()
            .fold(
                Question::score(instructions),
                |question, (label, description)| match description {
                    Some(description) => question.level(*label, *description),
                    None => question.bare_level(*label),
                },
            )
            .into()
    }

    fn from_answer(answer: &Answer) -> std::result::Result<Self, AnswerMismatch> {
        let Answer::Score(answer) = answer else {
            return Err(AnswerMismatch::kind(Self::KIND, answer));
        };
        if answer.probabilities.len() != T::LEVELS.len() {
            return Err(AnswerMismatch::new(format!(
                "expected {} level probabilities, found {}",
                T::LEVELS.len(),
                answer.probabilities.len()
            )));
        }
        if let Some(missing) = (0..T::LEVELS.len()).find(|&level| T::from_level(level).is_none()) {
            return Err(AnswerMismatch::new(format!(
                "`Scale::from_level` has no variant for level {missing}"
            )));
        }
        Ok(Self {
            score: answer.score,
            probabilities: answer.probabilities.clone(),
            scale: PhantomData,
        })
    }
}

// Written by hand: `T` is only a marker, so it needs none of these traits.
impl<T> fmt::Debug for Score<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Score")
            .field("score", &self.score)
            .field("probabilities", &self.probabilities)
            .finish()
    }
}

impl<T> Clone for Score<T> {
    fn clone(&self) -> Self {
        Self {
            score: self.score,
            probabilities: self.probabilities.clone(),
            scale: PhantomData,
        }
    }
}

impl<T> PartialEq for Score<T> {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.probabilities == other.probabilities
    }
}

/// The answer a provider sends for this choice: the chosen option's key and
/// every option's probability, by key. Decoding it as `Choice<T>` gives the
/// choice back.
///
/// It builds a canned answer from typed values, for a
/// `MockDecisionProvider` (feature `mock`) or a test of
/// a provider, without writing the option keys the derive generates.
impl<T: ChoiceSet> From<Choice<T>> for Answer {
    fn from(choice: Choice<T>) -> Self {
        let probabilities = T::OPTIONS
            .iter()
            .zip(choice.probabilities)
            .map(|((key, _), probability)| (*key, probability));
        Self::Choice(super::ChoiceAnswer::new(choice.value.key(), probabilities))
    }
}

/// The answer a provider sends for this score: the score and every level's
/// probability, lowest first. Decoding it as `Score<T>` gives the score
/// back.
///
/// Like `From<Choice<T>>`, it builds canned answers from typed values.
impl<T: Scale> From<Score<T>> for Answer {
    fn from(score: Score<T>) -> Self {
        Self::Score(super::ScoreAnswer::new(score.score, score.probabilities))
    }
}

/// Support for `#[derive(...)]` expansions. Not public API.
#[doc(hidden)]
pub mod __private {
    use crate::decision::Question;

    /// Attach yes/no outcome descriptions from field attributes.
    #[must_use]
    pub(crate) fn describe_yes_no(
        question: Question,
        yes: Option<&'static str>,
        no: Option<&'static str>,
    ) -> Question {
        match question {
            Question::YesNo(mut question) => {
                question.yes = yes.map(Into::into).or(question.yes);
                question.no = no.map(Into::into).or(question.no);
                Question::YesNo(question)
            }
            other => other,
        }
    }

    /// Requires `T: DecisionField`. The derive asserts this once per field,
    /// so a wrong field type is reported once, at the field.
    pub const fn assert_field<T: super::DecisionField>() {}

    /// A field of a derived [`Decision`](super::Decision).
    pub struct Field<T>(std::marker::PhantomData<T>);

    impl<T> std::fmt::Debug for Field<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("Field")
        }
    }

    impl<T: super::DecisionField> Field<T> {
        /// Whether the field asks a yes/no question.
        pub const IS_YES_NO: bool = matches!(T::KIND, crate::decision::QuestionKind::YesNo);

        /// The field's question.
        #[must_use]
        pub fn question(
            instructions: &'static str,
            yes: Option<&'static str>,
            no: Option<&'static str>,
        ) -> Question {
            describe_yes_no(T::question(instructions), yes, no)
        }

        /// Decode the field's answer.
        ///
        /// # Errors
        ///
        /// See [`answer_as`](super::answer_as).
        pub fn answer(response: &crate::DecisionResponse, name: &str) -> crate::Result<T> {
            super::answer_as(response, name)
        }
    }

    /// Resolves [`Field`]'s items when `T` is not a decision field, so the
    /// only error is [`assert_field`]'s. Never called: that error stops the
    /// build.
    pub trait FieldFallback<T> {
        /// Placeholder; see the trait docs.
        const IS_YES_NO: bool = true;

        /// Placeholder; see the trait docs.
        fn question(
            _instructions: &'static str,
            _yes: Option<&'static str>,
            _no: Option<&'static str>,
        ) -> Question {
            unreachable!("`assert_field` rejects this field type")
        }

        /// Placeholder; see the trait docs.
        ///
        /// # Errors
        ///
        /// Never returns.
        fn answer(_response: &crate::DecisionResponse, _name: &str) -> crate::Result<T> {
            unreachable!("`assert_field` rejects this field type")
        }
    }

    impl<T> FieldFallback<T> for Field<T> {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::{ChoiceAnswer, ScoreAnswer};

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Label {
        Billing,
        Bug,
    }

    impl ChoiceSet for Label {
        const OPTIONS: &'static [(&'static str, Option<&'static str>)] =
            &[("billing", Some("Payments")), ("bug", None)];

        fn from_index(index: usize) -> Option<Self> {
            [Self::Billing, Self::Bug].get(index).copied()
        }

        fn index(&self) -> usize {
            *self as usize
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Urgency {
        Low,
        Mid,
        High,
    }

    impl Scale for Urgency {
        const LEVELS: &'static [(&'static str, Option<&'static str>)] = &[
            ("Low", None),
            ("Mid", Some("Somewhat urgent")),
            ("High", None),
        ];

        fn from_level(level: usize) -> Option<Self> {
            [Self::Low, Self::Mid, Self::High].get(level).copied()
        }

        fn level(&self) -> usize {
            *self as usize
        }
    }

    struct Triage {
        label: Choice<Label>,
        spam: YesNoAnswer,
        urgency: Score<Urgency>,
    }

    impl Decision for Triage {
        fn questions() -> Vec<(String, Question)> {
            vec![
                ("label".into(), Choice::<Label>::question("Label?")),
                ("spam".into(), YesNoAnswer::question("Spam?")),
                ("urgency".into(), Score::<Urgency>::question("Urgency?")),
            ]
        }

        fn from_response(response: &DecisionResponse) -> Result<Self> {
            Ok(Self {
                label: answer_as(response, "label")?,
                spam: answer_as(response, "spam")?,
                urgency: answer_as(response, "urgency")?,
            })
        }
    }

    fn response() -> DecisionResponse {
        DecisionResponse::new([
            (
                "urgency",
                Answer::from(ScoreAnswer::new(1.0, [0.45, 0.1, 0.45])),
            ),
            (
                "label",
                ChoiceAnswer::new("bug", [("bug", 0.8), ("billing", 0.2)]).into(),
            ),
            ("spam", YesNoAnswer::new(0.3).into()),
        ])
    }

    #[test]
    fn builds_questions_from_types() {
        let request = Triage::request("m", "ticket");
        request.validate(&crate::DecisionLimits::new()).unwrap();
        let Question::Choice(label) = &request.questions[0].1 else {
            panic!("expected a choice");
        };
        assert_eq!(label.options[0].description.as_deref(), Some("Payments"));
        assert_eq!(label.options[1].description, None);
        let Question::Score(urgency) = &request.questions[2].1 else {
            panic!("expected a score");
        };
        let labels: Vec<_> = urgency.levels.iter().map(|l| l.label.as_str()).collect();
        assert_eq!(labels, ["Low", "Mid", "High"]);
        assert_eq!(
            urgency.levels[1].description.as_deref(),
            Some("Somewhat urgent")
        );
    }

    #[test]
    fn typed_answers_convert_to_wire_answers_and_back() {
        let label = Choice::new(Label::Bug, [(Label::Billing, 0.2), (Label::Bug, 0.8)]);
        let urgency = Score::new([(Urgency::Low, 0.25), (Urgency::High, 0.75)]);
        let response = DecisionResponse::new([
            ("label", Answer::from(label.clone())),
            ("spam", YesNoAnswer::new(0.3).into()),
            ("urgency", urgency.clone().into()),
        ]);
        assert_eq!(
            response.choice("label"),
            Some(&ChoiceAnswer::new("bug", [("billing", 0.2), ("bug", 0.8)]))
        );
        assert_eq!(
            response.score("urgency"),
            Some(&ScoreAnswer::new(1.5, [0.25, 0.0, 0.75]))
        );

        let triage = Triage::from_response(&response).unwrap();
        assert_eq!(triage.label, label);
        assert_eq!(triage.urgency, urgency);
    }

    #[test]
    fn decodes_answers_by_name() {
        let triage = Triage::from_response(&response()).unwrap();
        assert_eq!(*triage.label.value(), Label::Bug);
        assert_eq!(triage.label.probability(&Label::Billing), 0.2);
        assert!(!triage.spam.is_yes());
        assert_eq!(triage.urgency.most_likely(), Some(Urgency::Low));
        assert_eq!(triage.urgency.nearest(), Urgency::Mid);
        assert_eq!(Urgency::High.label(), "High");
    }

    #[test]
    fn reports_missing_and_mismatched_answers() {
        let mut missing = response();
        missing.answers.retain(|(name, _)| name != "spam");
        let err = Triage::from_response(&missing).err().unwrap();
        assert!(err.message().contains("\"spam\" was not answered"), "{err}");

        let mut wrong = response();
        wrong.answers[2].1 = ScoreAnswer::new(0.0, [1.0, 0.0, 0.0]).into();
        let err = Triage::from_response(&wrong).err().unwrap();
        assert!(err.message().contains("expected a yes/no answer"), "{err}");

        let mut unknown = response();
        unknown.answers[1].1 = ChoiceAnswer::new("other", [("bug", 1.0)]).into();
        let err = Triage::from_response(&unknown).err().unwrap();
        assert!(err.message().contains("unknown option \"other\""), "{err}");

        let mut repeated = response();
        repeated.answers[1].1 =
            ChoiceAnswer::new("bug", [("bug", 0.9), ("bug", 0.05), ("billing", 0.05)]).into();
        let err = Triage::from_response(&repeated).err().unwrap();
        assert!(err.message().contains("more than once"), "{err}");
    }

    #[test]
    fn typed_values_can_be_built_directly() {
        let choice = Choice::new(Label::Bug, [(Label::Bug, 0.7)]);
        assert_eq!(choice.probability(&Label::Billing), 0.0);
        let score = Score::new([(Urgency::Mid, 0.5), (Urgency::High, 0.5)]);
        assert!((score.score() - 1.5).abs() < 1e-9);
        assert_eq!(score.most_likely(), Some(Urgency::Mid));
        assert_eq!(score.clone(), score);
    }

    #[test]
    fn most_likely_names_a_level_only_when_the_distribution_does() {
        let argmax = Score::new([
            (Urgency::Low, 0.2),
            (Urgency::Mid, 0.1),
            (Urgency::High, 0.7),
        ]);
        assert_eq!(argmax.most_likely(), Some(Urgency::High));

        let tie = Score::new([
            (Urgency::Low, 0.1),
            (Urgency::Mid, 0.45),
            (Urgency::High, 0.45),
        ]);
        assert_eq!(tie.most_likely(), Some(Urgency::Mid));

        let all_zero = Score::<Urgency>::new([]);
        assert_eq!(all_zero.most_likely(), None);

        let nan = Score::new([(Urgency::Low, 0.5), (Urgency::High, f64::NAN)]);
        assert_eq!(nan.most_likely(), None);
    }

    #[test]
    fn describes_yes_no_outcomes() {
        let question = __private::describe_yes_no(YesNoAnswer::question("?"), Some("Yes!"), None);
        let Question::YesNo(question) = question else {
            panic!("expected yes/no");
        };
        assert_eq!(question.yes.as_deref(), Some("Yes!"));
        assert_eq!(question.no, None);
    }
}
