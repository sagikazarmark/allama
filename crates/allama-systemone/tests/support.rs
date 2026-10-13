//! Live checks shared by the per-backend e2e tests.

#![allow(dead_code)]

use allama::decision::{Choice, ChoiceSet, Decision, Question, Scale, Score, YesNoAnswer};
use allama::{DecisionProvider, DecisionProviderExt, DecisionRequest};
use allama_systemone::{Confidence, Provider};

#[derive(Debug, PartialEq, ChoiceSet)]
#[decision(rename_all = "lowercase")]
pub enum Label {
    #[decision(description = "Payments and refunds")]
    Billing,
    #[decision(description = "Software errors")]
    Bug,
    #[decision(description = "Login and account access")]
    Account,
}

#[derive(Debug, PartialEq, Scale)]
pub enum Urgency {
    #[decision(description = "No time pressure")]
    Routine,
    #[decision(description = "A customer is inconvenienced")]
    Soon,
    #[decision(description = "A critical service is unavailable")]
    Immediate,
}

#[derive(Decision)]
pub struct Triage {
    #[decision(instructions = "Which label fits this ticket?")]
    pub label: Choice<Label>,
    #[decision(
        instructions = "Is the customer requesting a refund?",
        yes = "The customer requests a refund"
    )]
    pub refund: YesNoAnswer,
    #[decision(instructions = "How urgently does this ticket need a response?")]
    pub urgency: Score<Urgency>,
}

/// Untyped questions; the answers must be what any decision model says.
pub async fn untyped(provider: &Provider, model: &str) {
    let request = DecisionRequest::new(model, "Our checkout has returned 500 errors since 9am.")
        .question(
            "label",
            Question::choice("Which label fits this ticket?")
                .option("billing", "Payments and refunds")
                .option("bug", "Software errors")
                .option("account", "Login and account access"),
        )
        .question("spam", Question::yes_no("Is this message spam?"));
    let response = provider.decide(&request).await.unwrap();
    assert_eq!(response.choice("label").unwrap().chosen, "bug");
    assert!(!response.yes_no("spam").unwrap().is_yes());
    assert!(response.usage.as_ref().unwrap().input_tokens.unwrap() > 0);
    let confidence = response.provider_data.get::<Confidence>().unwrap().unwrap();
    assert!(confidence.0.contains_key("label"));
}

/// A derived decision, through `decide_as`.
pub async fn typed(provider: &Provider, model: &str) {
    let decided = provider
        .decide_as::<Triage>(
            model,
            "I was charged twice. Please refund the extra payment.",
        )
        .await
        .unwrap();
    assert_eq!(*decided.value.label.value(), Label::Billing);
    assert!(decided.value.refund.is_yes());
    let _ = decided.value.urgency.most_likely();
}

/// A model the provider does not know is `ModelNotFound`.
pub async fn unknown_model(provider: &Provider) {
    let request = DecisionRequest::new("no-such-decision-model", "Hello")
        .question("spam", Question::yes_no("Is this spam?"));
    let err = provider.decide(&request).await.unwrap_err();
    assert!(
        matches!(err, allama::Error::ModelNotFound { .. }),
        "{err:?}"
    );
}
