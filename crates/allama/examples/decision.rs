//! Decision models: ask typed questions and read probabilities, untyped and
//! with `#[derive(Decision)]`.

use allama::DecisionResponse;
use allama::decision::{
    Answer, Choice, ChoiceAnswer, ChoiceSet, Decision, Question, Scale, Score, ScoreAnswer,
    YesNoAnswer,
};
use allama::mock::MockDecisionProvider;
use allama::prelude::*;

#[derive(Debug, ChoiceSet)]
#[decision(rename_all = "lowercase")]
enum Label {
    #[decision(description = "Payments and refunds")]
    Billing,
    #[decision(description = "Software errors")]
    Bug,
    #[decision(description = "Login and account access")]
    Account,
}

#[derive(Debug, Scale)]
enum Urgency {
    #[decision(description = "No time pressure")]
    Routine,
    #[decision(description = "A customer is inconvenienced")]
    Soon,
    #[decision(description = "A critical service is unavailable")]
    Immediate,
}

#[derive(Decision)]
struct Triage {
    #[decision(instructions = "Which label fits this ticket?")]
    label: Choice<Label>,
    #[decision(
        instructions = "Is the customer requesting a refund?",
        yes = "The customer requests a refund"
    )]
    refund: YesNoAnswer,
    #[decision(instructions = "How urgently does this ticket need a response?")]
    urgency: Score<Urgency>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // What a decision model would answer for this ticket.
    let answers = || {
        DecisionResponse::new([
            (
                "label",
                Answer::from(ChoiceAnswer::new(
                    "billing",
                    [("billing", 0.9781), ("bug", 0.0125), ("account", 0.0093)],
                )),
            ),
            ("refund", YesNoAnswer::new(0.9989).into()),
            (
                "urgency",
                ScoreAnswer::new(0.8308, [0.3, 0.5692, 0.1308]).into(),
            ),
        ])
        .model("demo-decision-model")
    };
    let provider = MockDecisionProvider::new([Ok(answers()), Ok(answers())]);
    let ticket = "I was charged twice. Please refund the extra payment.";

    // Untyped: build the questions and read answers by name.
    let request = DecisionRequest::new("demo-decision-model", ticket)
        .question(
            "label",
            Question::choice("Which label fits this ticket?")
                .option("billing", "Payments and refunds")
                .option("bug", "Software errors")
                .option("account", "Login and account access"),
        )
        .question(
            "refund",
            Question::yes_no("Is the customer requesting a refund?")
                .yes("The customer requests a refund"),
        )
        .question(
            "urgency",
            Question::score("How urgently does this ticket need a response?")
                .level("Routine", "No time pressure")
                .level("Soon", "A customer is inconvenienced")
                .level("Immediate", "A critical service is unavailable"),
        );
    let response = provider.decide(&request).await?;
    let label = response.choice("label").expect("label answered");
    println!(
        "label: {} (p = {:.2})",
        label.chosen,
        label.probability(&label.chosen).unwrap_or_default()
    );
    println!(
        "refund requested: {:.4}",
        response
            .yes_no("refund")
            .expect("refund answered")
            .probability
    );
    println!(
        "urgency score: {:.2} on 0..=2",
        response.score("urgency").expect("urgency answered").score
    );

    // Typed: the struct describes the questions and receives the answers;
    // the response stays available for usage and provider data.
    let decided = provider
        .decide_as::<Triage>("demo-decision-model", ticket)
        .await?;
    let triage = decided.value;
    println!(
        "typed: {:?} (p = {:.2}), refund = {}, most likely urgency = {:?}, nearest = {:?}",
        triage.label.value(),
        triage.label.probability(triage.label.value()),
        triage.refund.is_yes(),
        triage.urgency.most_likely(),
        triage.urgency.nearest(),
    );
    println!("answered by {:?}", decided.response.model);
    Ok(())
}
