use std::fmt;

use serde::{Deserialize, Serialize};

/// A typed question about a decision input.
///
/// Every question is answered against the whole input; answers are not
/// passed to later questions. Build one with [`Question::yes_no`],
/// [`Question::choice`], [`Question::score`], or [`Question::locate`]:
///
/// ```
/// use allama::decision::Question;
///
/// let label = Question::choice("Which label fits this ticket?")
///     .option("billing", "Payments and refunds")
///     .option("bug", "Software errors");
/// let refund = Question::yes_no("Is the customer requesting a refund?")
///     .yes("The customer requests a refund");
/// let urgency = Question::score("How urgently does this ticket need a response?")
///     .level("Routine", "No time pressure")
///     .level("Soon", "A customer is inconvenienced")
///     .level("Immediate", "A critical service is unavailable");
/// # let _ = (label, refund, urgency);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Question {
    /// The probability that the answer is yes.
    YesNo(YesNoQuestion),
    /// Pick one of several named options.
    Choice(ChoiceQuestion),
    /// Rate on an ordered scale of levels.
    Score(ScoreQuestion),
    /// Find one target in the request's images (ADR 0018).
    Locate(LocateQuestion),
}

impl Question {
    /// A yes/no question.
    #[must_use]
    pub fn yes_no(instructions: impl Into<String>) -> YesNoQuestion {
        YesNoQuestion::new(instructions)
    }

    /// A pick-one question; add options with [`ChoiceQuestion::option`].
    #[must_use]
    pub fn choice(instructions: impl Into<String>) -> ChoiceQuestion {
        ChoiceQuestion::new(instructions)
    }

    /// A rating question; add levels, lowest first, with
    /// [`ScoreQuestion::level`].
    #[must_use]
    pub fn score(instructions: impl Into<String>) -> ScoreQuestion {
        ScoreQuestion::new(instructions)
    }

    /// A question that finds one target in the request's images, such as
    /// "the coffee mug in the photo", and answers with its bounding box.
    /// The request needs at least one image.
    #[must_use]
    pub fn locate(instructions: impl Into<String>) -> LocateQuestion {
        LocateQuestion::new(instructions)
    }

    /// The kind of question.
    #[must_use]
    pub fn kind(&self) -> QuestionKind {
        match self {
            Self::YesNo(_) => QuestionKind::YesNo,
            Self::Choice(_) => QuestionKind::Choice,
            Self::Score(_) => QuestionKind::Score,
            Self::Locate(_) => QuestionKind::Locate,
        }
    }

    /// What the question asks.
    #[must_use]
    pub fn instructions(&self) -> &str {
        match self {
            Self::YesNo(q) => &q.instructions,
            Self::Choice(q) => &q.instructions,
            Self::Score(q) => &q.instructions,
            Self::Locate(q) => &q.instructions,
        }
    }
}

impl From<YesNoQuestion> for Question {
    fn from(question: YesNoQuestion) -> Self {
        Self::YesNo(question)
    }
}

impl From<ChoiceQuestion> for Question {
    fn from(question: ChoiceQuestion) -> Self {
        Self::Choice(question)
    }
}

impl From<ScoreQuestion> for Question {
    fn from(question: ScoreQuestion) -> Self {
        Self::Score(question)
    }
}

impl From<LocateQuestion> for Question {
    fn from(question: LocateQuestion) -> Self {
        Self::Locate(question)
    }
}

/// The kind of a [`Question`] or [`Answer`](super::Answer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum QuestionKind {
    /// A yes/no question.
    YesNo,
    /// A pick-one question.
    Choice,
    /// A rating question.
    Score,
    /// A question that finds one target in the images.
    Locate,
}

impl QuestionKind {
    /// Every kind, in declaration order.
    pub const ALL: &[Self] = &[Self::YesNo, Self::Choice, Self::Score, Self::Locate];

    /// The kind's name, as used in diagnostics.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::YesNo => "yes/no",
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Locate => "locate",
        }
    }
}

impl fmt::Display for QuestionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Asks for the probability that the answer is yes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct YesNoQuestion {
    /// What the question asks.
    pub instructions: String,
    /// What a yes means, when the instructions alone are ambiguous.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub yes: Option<String>,
    /// What a no means, when the instructions alone are ambiguous.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub no: Option<String>,
}

impl YesNoQuestion {
    /// A yes/no question without outcome descriptions.
    #[must_use]
    pub fn new(instructions: impl Into<String>) -> Self {
        Self {
            instructions: instructions.into(),
            yes: None,
            no: None,
        }
    }

    /// Describe the yes outcome.
    #[must_use]
    pub fn yes(mut self, description: impl Into<String>) -> Self {
        self.yes = Some(description.into());
        self
    }

    /// Describe the no outcome.
    #[must_use]
    pub fn no(mut self, description: impl Into<String>) -> Self {
        self.no = Some(description.into());
        self
    }
}

/// Asks the model to pick one of several named options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ChoiceQuestion {
    /// What the question asks.
    pub instructions: String,
    /// The options, in order. Providers may break ties by option order.
    pub options: Vec<ChoiceOption>,
}

impl ChoiceQuestion {
    /// A pick-one question with no options yet.
    #[must_use]
    pub fn new(instructions: impl Into<String>) -> Self {
        Self {
            instructions: instructions.into(),
            options: Vec::new(),
        }
    }

    /// Append an option with a description.
    #[must_use]
    pub fn option(mut self, key: impl Into<String>, description: impl Into<String>) -> Self {
        self.options
            .push(ChoiceOption::new(key).description(description));
        self
    }

    /// Append an option whose key is self-explanatory.
    #[must_use]
    pub fn bare_option(mut self, key: impl Into<String>) -> Self {
        self.options.push(ChoiceOption::new(key));
        self
    }

    /// Append self-explanatory options.
    #[must_use]
    pub fn options<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.options.extend(keys.into_iter().map(ChoiceOption::new));
        self
    }

    /// The option with `key`, if any.
    #[must_use]
    pub fn find_option(&self, key: &str) -> Option<&ChoiceOption> {
        self.options.iter().find(|option| option.key == key)
    }
}

/// One option of a [`ChoiceQuestion`].
///
/// The model reads the key (and the description, when present), so a key is
/// prompt text, not just an identifier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ChoiceOption {
    /// The key the answer reports when this option is chosen.
    pub key: String,
    /// What the option means. Without one, the model reads the key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl ChoiceOption {
    /// An option without a description.
    #[must_use]
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            description: None,
        }
    }

    /// Set the description.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Asks the model to rate on an ordered scale.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ScoreQuestion {
    /// What the question asks.
    pub instructions: String,
    /// The levels, from lowest (level 0) to highest.
    pub levels: Vec<ScoreLevel>,
}

impl ScoreQuestion {
    /// A rating question with no levels yet.
    #[must_use]
    pub fn new(instructions: impl Into<String>) -> Self {
        Self {
            instructions: instructions.into(),
            levels: Vec::new(),
        }
    }

    /// Append the next-higher level with a description.
    #[must_use]
    pub fn level(mut self, label: impl Into<String>, description: impl Into<String>) -> Self {
        self.levels
            .push(ScoreLevel::new(label).description(description));
        self
    }

    /// Append the next-higher level whose label is self-explanatory.
    #[must_use]
    pub fn bare_level(mut self, label: impl Into<String>) -> Self {
        self.levels.push(ScoreLevel::new(label));
        self
    }

    /// Append self-explanatory levels, lowest first.
    #[must_use]
    pub fn levels<I, S>(mut self, labels: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.levels.extend(labels.into_iter().map(ScoreLevel::new));
        self
    }
}

/// One level of a [`ScoreQuestion`].
///
/// The model reads both the label and the description, so a label is prompt
/// text, not just an identifier: `Immediate` says more than `L3`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ScoreLevel {
    /// A short name for the level, unique within the question.
    pub label: String,
    /// What the level means. Without one, the model reads the label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl ScoreLevel {
    /// A level without a description.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            description: None,
        }
    }

    /// Set the description.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Asks where one target is in the request's images; answered with its
/// bounding box (ADR 0018).
///
/// The instructions name the target, and the image to search when the
/// request has several: "the coffee mug in the photo". A target that is not
/// there still gets a box, so ask a yes/no question alongside when it may
/// be absent.
///
/// ```
/// use allama::decision::Question;
/// use allama::{DecisionRequest, Image};
///
/// let request = DecisionRequest::new("jev-preview", "A kitchen counter.")
///     .image(Image::base64("image/jpeg", "/9j/4AAQ..."))
///     .question("visible", Question::yes_no("Is there a coffee mug in the image?"))
///     .question("mug", Question::locate("the coffee mug"));
/// # let _ = request;
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LocateQuestion {
    /// What to find, and in which image.
    pub instructions: String,
}

impl LocateQuestion {
    /// A question finding what `instructions` describe.
    #[must_use]
    pub fn new(instructions: impl Into<String>) -> Self {
        Self {
            instructions: instructions.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn serializes_tagged_by_kind() {
        let question: Question = Question::choice("Pick")
            .option("a", "First")
            .bare_option("b")
            .into();
        assert_eq!(
            serde_json::to_value(&question).unwrap(),
            json!({
                "type": "choice",
                "instructions": "Pick",
                "options": [{"key": "a", "description": "First"}, {"key": "b"}]
            })
        );
        let yes_no: Question = Question::yes_no("Spam?").yes("It is spam").into();
        assert_eq!(
            serde_json::to_value(&yes_no).unwrap(),
            json!({"type": "yes_no", "instructions": "Spam?", "yes": "It is spam"})
        );
        let back: Question =
            serde_json::from_value(serde_json::to_value(&yes_no).unwrap()).unwrap();
        assert_eq!(back, yes_no);
    }

    #[test]
    fn score_levels_have_labels_and_descriptions() {
        let score: Question = Question::score("Rate")
            .level("low", "Barely")
            .bare_level("high")
            .into();
        assert_eq!(
            serde_json::to_value(&score).unwrap(),
            json!({
                "type": "score",
                "instructions": "Rate",
                "levels": [{"label": "low", "description": "Barely"}, {"label": "high"}]
            })
        );
        assert_eq!(score.kind(), QuestionKind::Score);
        assert_eq!(score.instructions(), "Rate");
        assert_eq!(QuestionKind::YesNo.to_string(), "yes/no");
    }
}
