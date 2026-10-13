use std::borrow::Cow;
use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::named;
use super::{DecisionRequest, Question, QuestionKind};
use crate::{Error, ProviderData, Result, Usage, Warning};

/// Answers to a [`DecisionRequest`], by question name.
///
/// ```
/// use allama::DecisionResponse;
/// use allama::decision::{Answer, ChoiceAnswer, YesNoAnswer};
///
/// let response = DecisionResponse::new([
///     ("label", Answer::from(ChoiceAnswer::new("billing", [("billing", 0.97), ("bug", 0.03)]))),
///     ("refund", Answer::from(YesNoAnswer::new(0.99))),
/// ]);
/// assert_eq!(response.choice("label").unwrap().chosen, "billing");
/// assert!(response.yes_no("refund").unwrap().is_yes());
/// ```
///
/// Answers hold what the provider reported. Probabilities are not
/// renormalized, and a choice's [`chosen`](ChoiceAnswer::chosen) option is
/// the provider's, so it may differ from the most likely option in its
/// probabilities.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DecisionResponse {
    /// One answer per question, keyed by question name, in the order the
    /// provider reported them.
    #[serde(with = "named")]
    pub answers: Vec<(String, Answer)>,
    /// Provider-assigned response identifier, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Model that answered, as reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Provider request identifier (from transport headers), for support.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// Identity of the provider that actually served the request. Wrappers
    /// such as fallback leave the serving provider's value in place.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<Cow<'static, str>>,
    /// Token usage, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Non-fatal warnings.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<Warning>,
    /// Provider-specific response data (for example a provider's own
    /// confidence statistic).
    #[serde(default, skip_serializing_if = "ProviderData::is_empty")]
    pub provider_data: ProviderData,
}

impl DecisionResponse {
    /// A response with `answers`.
    #[must_use]
    pub fn new<I, N>(answers: I) -> Self
    where
        I: IntoIterator<Item = (N, Answer)>,
        N: Into<String>,
    {
        Self {
            answers: answers
                .into_iter()
                .map(|(name, answer)| (name.into(), answer))
                .collect(),
            ..Self::default()
        }
    }

    /// Set the model.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the response id.
    #[must_use]
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set the request id.
    #[must_use]
    pub fn request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    /// Set the serving provider's identity.
    #[must_use]
    pub fn provider(mut self, provider: impl Into<Cow<'static, str>>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Set usage.
    #[must_use]
    pub fn usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Append a warning.
    #[must_use]
    pub fn warning(mut self, warning: Warning) -> Self {
        self.warnings.push(warning);
        self
    }

    /// Set provider data.
    #[must_use]
    pub fn provider_data(mut self, data: ProviderData) -> Self {
        self.provider_data = data;
        self
    }

    /// The answer to the question named `name`.
    #[must_use]
    pub fn answer(&self, name: &str) -> Option<&Answer> {
        named::get(&self.answers, name)
    }

    /// The answer to the yes/no question named `name`.
    #[must_use]
    pub fn yes_no(&self, name: &str) -> Option<&YesNoAnswer> {
        self.answer(name)?.as_yes_no()
    }

    /// The answer to the choice question named `name`.
    #[must_use]
    pub fn choice(&self, name: &str) -> Option<&ChoiceAnswer> {
        self.answer(name)?.as_choice()
    }

    /// The answer to the score question named `name`.
    #[must_use]
    pub fn score(&self, name: &str) -> Option<&ScoreAnswer> {
        self.answer(name)?.as_score()
    }

    /// Check that this response answers `request`.
    ///
    /// Every question must be answered exactly once by an answer of its
    /// kind, and nothing else may be answered. A choice must be one of the
    /// question's options and carry exactly one probability per option; a
    /// score must carry one probability per level and lie on the scale.
    /// Probabilities must lie in `[0, 1]`. They are not required to sum to
    /// exactly 1: providers round them. A choice is not required to be the
    /// most likely option either, since providers report both.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnexpectedResponse`] naming the first offending
    /// question.
    pub fn validate(&self, request: &DecisionRequest) -> Result<()> {
        let mut seen = HashSet::new();
        for (name, answer) in &self.answers {
            if !seen.insert(name.as_str()) {
                return Err(unexpected(name, "answered more than once"));
            }
            let Some(question) = request.find_question(name) else {
                return Err(unexpected(name, "was not asked"));
            };
            if answer.kind() != question.kind() {
                return Err(unexpected(
                    name,
                    format!(
                        "is a {} question, answered as {}",
                        question.kind(),
                        answer.kind()
                    ),
                ));
            }
            match (question, answer) {
                (Question::YesNo(_), Answer::YesNo(answer)) => {
                    check_probability(name, answer.probability)?;
                }
                (Question::Choice(question), Answer::Choice(answer)) => {
                    if question.find_option(&answer.chosen).is_none() {
                        return Err(unexpected(
                            name,
                            format!("chose unknown option {:?}", answer.chosen),
                        ));
                    }
                    let mut keys = HashSet::new();
                    for (key, probability) in &answer.probabilities {
                        if question.find_option(key).is_none() {
                            return Err(unexpected(
                                name,
                                format!("reports a probability for unknown option {key:?}"),
                            ));
                        }
                        if !keys.insert(key.as_str()) {
                            return Err(unexpected(
                                name,
                                format!("reports option {key:?} more than once"),
                            ));
                        }
                        check_probability(name, *probability)?;
                    }
                    if let Some(option) = question
                        .options
                        .iter()
                        .find(|option| !keys.contains(option.key.as_str()))
                    {
                        return Err(unexpected(
                            name,
                            format!("reports no probability for option {:?}", option.key),
                        ));
                    }
                }
                (Question::Score(question), Answer::Score(answer)) => {
                    if question.levels.is_empty() {
                        return Err(unexpected(name, "asks a score question without levels"));
                    }
                    if answer.probabilities.len() != question.levels.len() {
                        return Err(unexpected(
                            name,
                            format!(
                                "reports {} level probabilities for {} levels",
                                answer.probabilities.len(),
                                question.levels.len()
                            ),
                        ));
                    }
                    for probability in &answer.probabilities {
                        check_probability(name, *probability)?;
                    }
                    #[allow(clippy::cast_precision_loss)]
                    let top = question.levels.len().saturating_sub(1) as f64;
                    if !(answer.score.is_finite()
                        && (-TOLERANCE..=top + TOLERANCE).contains(&answer.score))
                    {
                        return Err(unexpected(
                            name,
                            format!("score {} is outside 0..={top}", answer.score),
                        ));
                    }
                }
                _ => unreachable!("kinds were compared above"),
            }
        }
        for (name, _) in &request.questions {
            if !seen.contains(name.as_str()) {
                return Err(unexpected(name, "was not answered"));
            }
        }
        Ok(())
    }
}

/// The answer to one [`Question`], of the same kind.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Answer {
    /// Answer to a yes/no question.
    YesNo(YesNoAnswer),
    /// Answer to a choice question.
    Choice(ChoiceAnswer),
    /// Answer to a score question.
    Score(ScoreAnswer),
}

impl Answer {
    /// The kind of question this answers.
    #[must_use]
    pub fn kind(&self) -> QuestionKind {
        match self {
            Self::YesNo(_) => QuestionKind::YesNo,
            Self::Choice(_) => QuestionKind::Choice,
            Self::Score(_) => QuestionKind::Score,
        }
    }

    /// The yes/no answer, if this is one.
    #[must_use]
    pub fn as_yes_no(&self) -> Option<&YesNoAnswer> {
        match self {
            Self::YesNo(answer) => Some(answer),
            _ => None,
        }
    }

    /// The choice answer, if this is one.
    #[must_use]
    pub fn as_choice(&self) -> Option<&ChoiceAnswer> {
        match self {
            Self::Choice(answer) => Some(answer),
            _ => None,
        }
    }

    /// The score answer, if this is one.
    #[must_use]
    pub fn as_score(&self) -> Option<&ScoreAnswer> {
        match self {
            Self::Score(answer) => Some(answer),
            _ => None,
        }
    }
}

impl From<YesNoAnswer> for Answer {
    fn from(answer: YesNoAnswer) -> Self {
        Self::YesNo(answer)
    }
}

impl From<ChoiceAnswer> for Answer {
    fn from(answer: ChoiceAnswer) -> Self {
        Self::Choice(answer)
    }
}

impl From<ScoreAnswer> for Answer {
    fn from(answer: ScoreAnswer) -> Self {
        Self::Score(answer)
    }
}

/// The probability that the answer to a yes/no question is yes.
///
/// This is also the typed answer of a yes/no field in a
/// [`Decision`](super::Decision).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct YesNoAnswer {
    /// Probability of yes, from 0 to 1.
    pub probability: f64,
}

impl YesNoAnswer {
    /// An answer with the probability of yes.
    #[must_use]
    pub fn new(probability: f64) -> Self {
        Self { probability }
    }

    /// Whether yes is more likely than no.
    ///
    /// This always picks a side, however close the answer is to even; use
    /// [`decide`](Self::decide) to abstain on an uncertain one.
    #[must_use]
    pub fn is_yes(&self) -> bool {
        self.probability > 0.5
    }

    /// The answer at a confidence of `threshold`, or `None` to abstain.
    ///
    /// `Some(true)` when the probability of yes is at least `threshold`,
    /// `Some(false)` when the probability of no (1 minus the probability of
    /// yes) is at least `threshold`, and `None` when neither side is that
    /// sure, for example to route the input to a person instead. A NaN
    /// probability reaches no threshold, so it abstains too.
    ///
    /// `threshold` must be above 0.5, so that at most one side can reach it,
    /// and at most 1.
    ///
    /// ```
    /// use allama::decision::YesNoAnswer;
    ///
    /// assert_eq!(YesNoAnswer::new(0.97).decide(0.9), Some(true));
    /// assert_eq!(YesNoAnswer::new(0.04).decide(0.9), Some(false));
    /// // Leaning yes, but not by enough: abstain.
    /// assert_eq!(YesNoAnswer::new(0.7).decide(0.9), None);
    /// ```
    ///
    /// # Panics
    ///
    /// When `threshold` is not above 0.5 or is above 1 (or is NaN): at 0.5
    /// or below both sides could reach it, and above 1 neither ever does.
    #[must_use]
    pub fn decide(&self, threshold: f64) -> Option<bool> {
        assert!(
            threshold > 0.5 && threshold <= 1.0,
            "a yes/no threshold must be above 0.5 and at most 1, got {threshold}"
        );
        if self.probability >= threshold {
            Some(true)
        } else if 1.0 - self.probability >= threshold {
            Some(false)
        } else {
            None
        }
    }
}

/// The chosen option of a choice question and the distribution behind it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ChoiceAnswer {
    /// Key of the option the provider chose.
    ///
    /// Usually the most likely option, with ties following the provider's
    /// option order (not necessarily the request order). allama reports
    /// what the provider sent and does not renormalize the probabilities or
    /// recompute the choice, so `chosen` may differ from the option with
    /// the highest probability.
    pub chosen: String,
    /// Probability of each option, summing to 1 up to the provider's
    /// rounding, in the order the provider reported them.
    #[serde(with = "named")]
    pub probabilities: Vec<(String, f64)>,
}

impl ChoiceAnswer {
    /// An answer choosing `chosen`, with per-option probabilities.
    #[must_use]
    pub fn new<I, K>(chosen: impl Into<String>, probabilities: I) -> Self
    where
        I: IntoIterator<Item = (K, f64)>,
        K: Into<String>,
    {
        Self {
            chosen: chosen.into(),
            probabilities: probabilities
                .into_iter()
                .map(|(key, probability)| (key.into(), probability))
                .collect(),
        }
    }

    /// The probability of the option `key`.
    #[must_use]
    pub fn probability(&self, key: &str) -> Option<f64> {
        named::get(&self.probabilities, key).copied()
    }
}

/// The rating of a score question and the distribution behind it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ScoreAnswer {
    /// Probability-weighted average of the level numbers, from 0 to the
    /// number of levels minus 1, as the provider reported it. Not rounded to
    /// a level.
    pub score: f64,
    /// Probability of each level, lowest first.
    pub probabilities: Vec<f64>,
}

impl ScoreAnswer {
    /// An answer with the expected level and per-level probabilities.
    #[must_use]
    pub fn new(score: f64, probabilities: impl Into<Vec<f64>>) -> Self {
        Self {
            score,
            probabilities: probabilities.into(),
        }
    }

    /// The probability of `level` (0 is the lowest).
    #[must_use]
    pub fn probability(&self, level: usize) -> Option<f64> {
        self.probabilities.get(level).copied()
    }

    /// The single most likely level; ties go to the lower level.
    ///
    /// `None` when there are no probabilities, every probability is zero, or
    /// any is NaN: such a distribution names no level.
    #[must_use]
    pub fn most_likely(&self) -> Option<usize> {
        most_likely_level(&self.probabilities)
    }

    /// The level nearest to [`score`](Self::score), clamped to the scale;
    /// `None` when there are no levels.
    ///
    /// The score is an average, so this can name a level the model gave
    /// probability 0: a model split between the lowest and highest of five
    /// levels averages to the middle one. Use
    /// [`most_likely`](Self::most_likely) for the level the model favoured,
    /// and [`probabilities`](Self::probabilities) to tell a confident
    /// middle rating from a split one.
    ///
    /// ```
    /// use allama::decision::ScoreAnswer;
    ///
    /// // Split between "very low" and "very high".
    /// let answer = ScoreAnswer::new(2.0, [0.5, 0.0, 0.0, 0.0, 0.5]);
    /// assert_eq!(answer.nearest(), Some(2));
    /// assert_eq!(answer.probability(2), Some(0.0));
    /// assert_eq!(answer.most_likely(), Some(0));
    /// ```
    #[must_use]
    pub fn nearest(&self) -> Option<usize> {
        let top = self.probabilities.len().checked_sub(1)?;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let level = self.score.round().max(0.0) as usize;
        Some(level.min(top))
    }
}

/// Rounding slack accepted on probabilities and scores.
const TOLERANCE: f64 = 1e-6;

/// The index of the highest probability, ties to the lowest index; `None`
/// when the distribution is empty, all zero, or holds a NaN.
pub(super) fn most_likely_level(probabilities: &[f64]) -> Option<usize> {
    if probabilities.iter().any(|p| p.is_nan()) {
        return None;
    }
    let mut best: Option<(usize, f64)> = None;
    for (level, &p) in probabilities.iter().enumerate() {
        match best {
            Some((_, top)) if top >= p => {}
            _ => best = Some((level, p)),
        }
    }
    best.filter(|&(_, p)| p > 0.0).map(|(level, _)| level)
}

fn check_probability(name: &str, probability: f64) -> Result<()> {
    if probability.is_finite() && (-TOLERANCE..=1.0 + TOLERANCE).contains(&probability) {
        Ok(())
    } else {
        Err(unexpected(
            name,
            format!("reports probability {probability} outside 0..=1"),
        ))
    }
}

fn unexpected(name: &str, problem: impl std::fmt::Display) -> Error {
    Error::unexpected_response(format!("decision question {name:?} {problem}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::{Question, ScoreQuestion};
    use serde_json::json;

    fn request() -> DecisionRequest {
        DecisionRequest::new("m", "ticket")
            .question(
                "label",
                Question::choice("?").bare_option("a").bare_option("b"),
            )
            .question("spam", Question::yes_no("?"))
            .question(
                "urgency",
                Question::score("?").levels(["low", "mid", "high"]),
            )
    }

    fn response() -> DecisionResponse {
        DecisionResponse::new([
            (
                "label",
                Answer::from(ChoiceAnswer::new("a", [("b", 0.1), ("a", 0.9)])),
            ),
            ("spam", YesNoAnswer::new(0.2).into()),
            ("urgency", ScoreAnswer::new(1.2, [0.1, 0.6, 0.3]).into()),
        ])
    }

    fn problem(response: DecisionResponse) -> String {
        match response.validate(&request()).unwrap_err() {
            Error::UnexpectedResponse { message, .. } => message,
            other => panic!("expected UnexpectedResponse, got {other:?}"),
        }
    }

    #[test]
    fn matching_response_is_valid() {
        response().validate(&request()).unwrap();
    }

    #[test]
    fn mismatches_are_reported() {
        let mut missing = response();
        missing.answers.pop();
        assert!(problem(missing).contains("\"urgency\" was not answered"));

        let mut extra = response();
        extra
            .answers
            .push(("other".into(), YesNoAnswer::new(0.5).into()));
        assert!(problem(extra).contains("was not asked"));

        let mut wrong_kind = response();
        wrong_kind.answers[1].1 = ScoreAnswer::new(0.0, [1.0, 0.0]).into();
        assert!(problem(wrong_kind).contains("is a yes/no question, answered as score"));

        let mut unknown = response();
        unknown.answers[0].1 = ChoiceAnswer::new("c", [("a", 0.5), ("b", 0.5)]).into();
        assert!(problem(unknown).contains("unknown option \"c\""));

        let mut incomplete = response();
        incomplete.answers[0].1 = ChoiceAnswer::new("a", [("a", 1.0)]).into();
        assert!(problem(incomplete).contains("no probability for option \"b\""));

        let mut repeated = response();
        repeated.answers[0].1 =
            ChoiceAnswer::new("a", [("a", 0.9), ("a", 0.05), ("b", 0.05)]).into();
        assert!(problem(repeated).contains("option \"a\" more than once"));

        let mut levels = response();
        levels.answers[2].1 = ScoreAnswer::new(0.5, [0.5, 0.5]).into();
        assert!(problem(levels).contains("2 level probabilities for 3 levels"));

        let mut range = response();
        range.answers[1].1 = YesNoAnswer::new(1.5).into();
        assert!(problem(range).contains("outside 0..=1"));

        let mut nan = response();
        nan.answers[2].1 = ScoreAnswer::new(f64::NAN, [0.1, 0.6, 0.3]).into();
        assert!(problem(nan).contains("score NaN"));
    }

    #[test]
    fn unvalidated_empty_scale_does_not_underflow() {
        let request = DecisionRequest::new("m", "x").question("q", ScoreQuestion::new("?"));
        for score in [0.0, 3.0] {
            let response =
                DecisionResponse::new([("q", Answer::from(ScoreAnswer::new(score, [])))]);
            assert!(response.validate(&request).is_err(), "score {score}");
        }
    }

    #[test]
    fn accessors_and_helpers() {
        let response = response();
        assert_eq!(
            response.choice("label").unwrap().probability("a"),
            Some(0.9)
        );
        assert!(!response.yes_no("spam").unwrap().is_yes());
        let score = response.score("urgency").unwrap();
        assert_eq!(score.most_likely(), Some(1));
        assert_eq!(score.nearest(), Some(1));
        assert!(response.choice("spam").is_none());
        assert_eq!(ScoreAnswer::new(0.0, [0.5, 0.5]).most_likely(), Some(0));
        assert_eq!(ScoreAnswer::new(0.0, [0.0, 0.0]).most_likely(), None);
        assert_eq!(ScoreAnswer::new(0.0, [0.5, f64::NAN]).most_likely(), None);
        assert_eq!(ScoreAnswer::new(0.0, []).most_likely(), None);
        assert_eq!(ScoreAnswer::new(9.0, [0.5, 0.5]).nearest(), Some(1));
        assert_eq!(ScoreAnswer::new(0.0, []).nearest(), None);
    }

    #[test]
    fn nearest_can_name_a_level_with_no_probability() {
        let split = ScoreAnswer::new(2.0, [0.5, 0.0, 0.0, 0.0, 0.5]);
        assert_eq!(split.nearest(), Some(2));
        assert_eq!(split.probability(2), Some(0.0));
        assert_eq!(split.most_likely(), Some(0));
    }

    #[test]
    fn decide_answers_only_at_the_threshold() {
        let decide =
            |probability: f64, threshold: f64| YesNoAnswer::new(probability).decide(threshold);
        // Each side reaches the threshold inclusively.
        assert_eq!(decide(0.9, 0.9), Some(true));
        assert_eq!(decide(0.25, 0.75), Some(false));
        assert_eq!(decide(0.95, 0.9), Some(true));
        assert_eq!(decide(0.05, 0.9), Some(false));
        // Between the two thresholds: abstain, where `is_yes` picks a side.
        assert_eq!(decide(0.7, 0.9), None);
        assert_eq!(decide(0.3, 0.9), None);
        assert_eq!(decide(0.5, 0.51), None);
        assert!(YesNoAnswer::new(0.7).is_yes());
        // A threshold of 1 answers only certainty.
        assert_eq!(decide(1.0, 1.0), Some(true));
        assert_eq!(decide(0.0, 1.0), Some(false));
        assert_eq!(decide(0.999, 1.0), None);
        // Just above even, every answer but an exact tie picks a side.
        assert_eq!(decide(0.6, 0.55), Some(true));
        assert_eq!(decide(0.5, 0.500_001), None);
        // A NaN probability reaches no threshold.
        assert_eq!(decide(f64::NAN, 0.9), None);
    }

    #[test]
    fn decide_rejects_thresholds_that_cannot_separate_the_sides() {
        for threshold in [0.5, 0.2, 0.0, -1.0, 1.01, f64::NAN, f64::INFINITY] {
            let result = std::panic::catch_unwind(|| YesNoAnswer::new(0.5).decide(threshold));
            assert!(result.is_err(), "threshold {threshold}");
        }
    }

    #[test]
    fn serializes_answers_in_order() {
        let value = serde_json::to_value(response()).unwrap();
        assert_eq!(
            value["answers"]["label"],
            json!({"type": "choice", "chosen": "a", "probabilities": {"b": 0.1, "a": 0.9}})
        );
        let text = serde_json::to_string(&response()).unwrap();
        let back: DecisionResponse = serde_json::from_str(&text).unwrap();
        assert_eq!(back, response());
    }
}
