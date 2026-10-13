use allama::decision::{
    Answer, Choice, ChoiceAnswer, ChoiceSet, Decision, DecisionField, Question, QuestionKind,
    Scale, Score, ScoreAnswer, YesNoAnswer,
};
use allama::{DecisionLimits, DecisionResponse};
use serde_json::json;

#[derive(Debug, Clone, Copy, PartialEq, ChoiceSet)]
#[decision(rename_all = "snake_case")]
enum Label {
    #[decision(description = "Payments and refunds")]
    Billing,
    BugReport,
    #[decision(rename = "login", description = "Login and account access")]
    Account,
}

#[derive(Debug, Clone, Copy, PartialEq, Scale)]
enum Urgency {
    #[decision(description = "No time pressure")]
    Routine,
    Soon,
    #[decision(rename = "Critical", description = "A critical service is unavailable")]
    Immediate,
}

#[derive(Debug, Decision)]
struct Triage {
    #[decision(instructions = "Which label fits this ticket?")]
    label: Label,

    #[decision(
        rename = "label_details",
        instructions = "Which label fits this ticket?"
    )]
    label_full: Choice<Label>,

    #[decision(
        instructions = "Is the customer requesting a refund?",
        yes = "The customer requests a refund",
        no = "No refund is requested"
    )]
    refund: YesNoAnswer,

    #[decision(instructions = "Is this ticket spam?")]
    spam: YesNoAnswer,

    #[decision(instructions = "How urgently does this ticket need a response?")]
    urgency: Score<Urgency>,
}

#[derive(Debug, Decision)]
struct Generic<T: DecisionField> {
    #[decision(instructions = "Anything?", yes = "Something")]
    r#field: T,
    #[decision(instructions = "Spam?", yes = "Spam")]
    concrete: YesNoAnswer,
}

// Fields may depend on const and lifetime parameters.
#[derive(Debug, Decision)]
struct ConstGeneric<const N: usize> {
    #[decision(instructions = "Spam?")]
    spam: Sized<N>,
}

#[derive(Debug, Decision)]
struct Borrowing<'a> {
    #[decision(instructions = "Spam?")]
    spam: Borrowed<'a>,
}

/// A yes/no field type that depends on a const parameter.
#[derive(Debug)]
struct Sized<const N: usize>(YesNoAnswer);

/// A yes/no field type that depends on a lifetime.
#[derive(Debug)]
struct Borrowed<'a>(YesNoAnswer, std::marker::PhantomData<&'a ()>);

impl<const N: usize> DecisionField for Sized<N> {
    const KIND: QuestionKind = QuestionKind::YesNo;
    fn question(instructions: impl Into<String>) -> Question {
        YesNoAnswer::question(instructions)
    }
    fn from_answer(answer: &Answer) -> Result<Self, allama::decision::AnswerMismatch> {
        YesNoAnswer::from_answer(answer).map(Self)
    }
}

impl DecisionField for Borrowed<'_> {
    const KIND: QuestionKind = QuestionKind::YesNo;
    fn question(instructions: impl Into<String>) -> Question {
        YesNoAnswer::question(instructions)
    }
    fn from_answer(answer: &Answer) -> Result<Self, allama::decision::AnswerMismatch> {
        YesNoAnswer::from_answer(answer).map(|answer| Self(answer, std::marker::PhantomData))
    }
}

// The first paragraph of a doc comment stands in for `description` and
// `instructions`; an explicit attribute wins.
#[derive(Debug, Clone, Copy, PartialEq, ChoiceSet)]
#[decision(rename_all = "lowercase")]
enum Documented {
    /// Payments
    ///   and refunds.
    ///
    /// Rust readers only: see [`Documented::Attribute`].
    Doc,
    #[decision(description = "From the attribute")]
    Attribute,
    /// Shown to Rust readers only.
    #[decision(description = "The attribute wins")]
    Both,
    Neither,
}

#[derive(Debug, Clone, Copy, PartialEq, Scale)]
enum DocumentedLevel {
    /** No time
    pressure.

    ```
    // Rust readers only.
    ```
    */
    Low,
    /// Shown to Rust readers only.
    #[decision(description = "The attribute wins")]
    High,
}

#[derive(Debug, Decision)]
#[expect(dead_code, reason = "only the questions are checked")]
struct DocumentedTriage {
    /// Which label fits
    /// this ticket?
    ///
    /// Pick one.
    label: Documented,

    #[decision(instructions = "Is this ticket spam?")]
    spam: YesNoAnswer,

    /// Shown to Rust readers only.
    #[decision(instructions = "Is a refund requested?", yes = "A refund")]
    refund: YesNoAnswer,

    /// How urgent is it?
    urgency: Score<DocumentedLevel>,
}

mod shadowed {
    use allama::decision::{Decision, YesNoAnswer};

    // An item named `response` at the call site does not capture the
    // generated parameter.
    #[allow(dead_code, non_upper_case_globals)]
    const response: u8 = 0;

    #[derive(Decision)]
    pub struct Shadowed {
        #[decision(instructions = "Spam?")]
        pub spam: YesNoAnswer,
    }
}

// The derives accept a renamed or re-exported `allama`.
mod facade {
    pub use allama as renamed;
}

#[derive(Debug, PartialEq, ChoiceSet)]
#[decision(crate = "facade::renamed")]
enum Renamed {
    A,
    B,
}

#[derive(Decision)]
#[decision(crate = "facade::renamed")]
struct ThroughFacade {
    #[decision(instructions = "Which?")]
    pick: Renamed,
}

// The derive works on items produced by `macro_rules!`, whose tokens carry
// another hygiene context than the derive's own.
macro_rules! decision_struct {
    ($(#[$meta:meta])*) => {
        $(#[$meta])*
        struct FromMacro {
            #[decision(instructions = "Is this spam?")]
            spam: YesNoAnswer,
        }
    };
}

decision_struct!(#[derive(Decision)]);

fn label_answer() -> Answer {
    ChoiceAnswer::new(
        "bug_report",
        [("billing", 0.1), ("bug_report", 0.7), ("login", 0.2)],
    )
    .into()
}

fn response() -> DecisionResponse {
    DecisionResponse::new([
        ("label", label_answer()),
        ("label_details", label_answer()),
        ("refund", YesNoAnswer::new(0.25).into()),
        ("spam", YesNoAnswer::new(0.75).into()),
        ("urgency", ScoreAnswer::new(1.6, [0.1, 0.2, 0.7]).into()),
    ])
}

#[test]
fn choice_options() {
    assert_eq!(
        Label::OPTIONS,
        [
            ("billing", Some("Payments and refunds")),
            ("bug_report", None),
            ("login", Some("Login and account access")),
        ]
    );
    assert_eq!(Label::from_index(2), Some(Label::Account));
    assert_eq!(Label::from_index(3), None);
    assert_eq!(Label::Account.index(), 2);
    assert_eq!(Label::Account.key(), "login");
    assert_eq!(Label::from_key("bug_report"), Some(Label::BugReport));
    assert_eq!(Label::KIND, QuestionKind::Choice);
}

#[test]
fn levels() {
    assert_eq!(
        Urgency::LEVELS,
        [
            ("Routine", Some("No time pressure")),
            ("Soon", None),
            ("Critical", Some("A critical service is unavailable")),
        ]
    );
    assert_eq!(Urgency::from_level(1), Some(Urgency::Soon));
    assert_eq!(Urgency::Immediate.level(), 2);
    assert_eq!(Urgency::Immediate.label(), "Critical");
}

#[test]
fn request() {
    let request = Triage::request("nimble", "I was charged twice.");
    request.validate(&DecisionLimits::new()).unwrap();
    let options = json!([
        {"key": "billing", "description": "Payments and refunds"},
        {"key": "bug_report"},
        {"key": "login", "description": "Login and account access"}
    ]);
    assert_eq!(
        serde_json::to_value(&request).unwrap(),
        json!({
            "model": "nimble",
            "input": "I was charged twice.",
            "questions": {
                "label": {
                    "type": "choice",
                    "instructions": "Which label fits this ticket?",
                    "options": options
                },
                "label_details": {
                    "type": "choice",
                    "instructions": "Which label fits this ticket?",
                    "options": options
                },
                "refund": {
                    "type": "yes_no",
                    "instructions": "Is the customer requesting a refund?",
                    "yes": "The customer requests a refund",
                    "no": "No refund is requested"
                },
                "spam": {"type": "yes_no", "instructions": "Is this ticket spam?"},
                "urgency": {
                    "type": "score",
                    "instructions": "How urgently does this ticket need a response?",
                    "levels": [
                        {"label": "Routine", "description": "No time pressure"},
                        {"label": "Soon"},
                        {"label": "Critical", "description": "A critical service is unavailable"}
                    ]
                }
            }
        })
    );
    let names: Vec<_> = request
        .questions
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(
        names,
        ["label", "label_details", "refund", "spam", "urgency"]
    );
}

#[test]
fn decode() {
    let triage = Triage::from_response(&response()).unwrap();
    assert_eq!(triage.label, Label::BugReport);
    assert_eq!(*triage.label_full.value(), Label::BugReport);
    assert_eq!(triage.label_full.probability(&Label::Account), 0.2);
    assert_eq!(
        triage.label_full.probabilities().collect::<Vec<_>>(),
        [
            (Label::Billing, 0.1),
            (Label::BugReport, 0.7),
            (Label::Account, 0.2)
        ]
    );
    assert_eq!(triage.refund, YesNoAnswer::new(0.25));
    assert!(triage.spam.is_yes());
    assert_eq!(triage.urgency.score(), 1.6);
    assert_eq!(triage.urgency.most_likely(), Some(Urgency::Immediate));
    assert_eq!(triage.urgency.nearest(), Urgency::Immediate);
    assert_eq!(triage.urgency.probability(&Urgency::Soon), 0.2);
}

#[test]
fn decode_errors_name_the_question() {
    let mut missing = response();
    missing.answers.retain(|(name, _)| name != "refund");
    let err = Triage::from_response(&missing).unwrap_err();
    assert!(
        err.message().contains("\"refund\" was not answered"),
        "{err}"
    );

    let mut mismatch = response();
    mismatch.answers[4].1 = YesNoAnswer::new(0.5).into();
    let err = Triage::from_response(&mismatch).unwrap_err();
    assert!(
        err.message()
            .contains("\"urgency\": expected a score answer, found a yes/no answer"),
        "{err}"
    );

    let mut levels = response();
    levels.answers[4].1 = ScoreAnswer::new(0.5, [0.5, 0.5]).into();
    let err = Triage::from_response(&levels).unwrap_err();
    assert!(
        err.message().contains("expected 3 level probabilities"),
        "{err}"
    );
}

#[test]
fn generic_structs_and_raw_identifiers() {
    let questions = Generic::<YesNoAnswer>::questions();
    assert_eq!(questions.len(), 2);
    assert_eq!(questions[0].0, "field");
    let Question::YesNo(field) = &questions[0].1 else {
        panic!("expected a yes/no question");
    };
    assert_eq!(field.yes.as_deref(), Some("Something"));

    let decoded = Generic::<YesNoAnswer>::from_response(&DecisionResponse::new([
        ("field", Answer::from(YesNoAnswer::new(0.9))),
        ("concrete", YesNoAnswer::new(0.1).into()),
    ]))
    .unwrap();
    assert!(decoded.r#field.is_yes());
    assert!(!decoded.concrete.is_yes());
}

#[test]
fn const_and_lifetime_generics_and_shadowing() {
    let response = DecisionResponse::new([("spam", Answer::from(YesNoAnswer::new(0.8)))]);
    assert!(
        ConstGeneric::<3>::from_response(&response)
            .unwrap()
            .spam
            .0
            .is_yes()
    );
    assert!(Borrowing::from_response(&response).unwrap().spam.0.is_yes());
    assert!(
        shadowed::Shadowed::from_response(&response)
            .unwrap()
            .spam
            .is_yes()
    );
}

#[test]
fn renamed_crate_and_macro_generated_items() {
    assert_eq!(ThroughFacade::questions().len(), 1);
    let decoded = ThroughFacade::from_response(&DecisionResponse::new([(
        "pick",
        Answer::from(ChoiceAnswer::new("B", [("A", 0.4), ("B", 0.6)])),
    )]))
    .unwrap();
    assert_eq!(decoded.pick, Renamed::B);

    let decoded = FromMacro::from_response(&DecisionResponse::new([(
        "spam",
        Answer::from(YesNoAnswer::new(0.2)),
    )]))
    .unwrap();
    assert!(!decoded.spam.is_yes());
}

#[test]
fn doc_comments_fall_back() {
    assert_eq!(
        Documented::OPTIONS,
        [
            ("doc", Some("Payments and refunds.")),
            ("attribute", Some("From the attribute")),
            ("both", Some("The attribute wins")),
            ("neither", None),
        ]
    );
    assert_eq!(
        DocumentedLevel::LEVELS,
        [
            ("Low", Some("No time pressure.")),
            ("High", Some("The attribute wins")),
        ]
    );

    let instructions: Vec<_> = DocumentedTriage::questions()
        .into_iter()
        .map(|(name, question)| (name, question.instructions().to_owned()))
        .collect();
    let instructions: Vec<_> = instructions
        .iter()
        .map(|(name, instructions)| (name.as_str(), instructions.as_str()))
        .collect();
    assert_eq!(
        instructions,
        [
            ("label", "Which label fits this ticket?"),
            ("spam", "Is this ticket spam?"),
            ("refund", "Is a refund requested?"),
            ("urgency", "How urgent is it?"),
        ]
    );
}

/// Compile errors the macros emit themselves.
#[test]
fn ui() {
    trybuild::TestCases::new().compile_fail("tests/ui/*.rs");
}

/// Compile errors rustc emits about the generated code (trait bounds, const
/// evaluation). Their wording changes between compiler releases, and they
/// list `allama`'s `DecisionField` impls, so they run on request: see the
/// crate README.
#[test]
#[ignore = "rustc diagnostics recorded with rustc 1.98; run with `--ignored`"]
fn ui_rustc() {
    trybuild::TestCases::new().compile_fail("tests/ui-rustc/*.rs");
}
