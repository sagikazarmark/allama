//! Decision models: typed questions answered with probabilities.
//!
//! A decision model reads an input and answers named questions with
//! probabilities instead of generating text: the probability that the answer
//! is yes ([`Question::yes_no`]), a pick among options
//! ([`Question::choice`]), or a rating on an ordered scale of levels
//! ([`Question::score`]). Providers implement [`DecisionProvider`].
//!
//! The portable model is what the decision protocols share. Provider
//! statistics such as a "confidence" summary are computed differently by
//! each provider and live in [`DecisionResponse::provider_data`]; request
//! limits are queried with [`DecisionProvider::decision_limits`] rather than
//! fixed.
//!
//! No decision model? Extract into an enum with a chat model
//! ([`extract`][crate::extract]): you get a label without probabilities; it
//! is not a [`DecisionProvider`].
//!
//! ```
//! # #[cfg(feature = "mock")] {
//! use allama::decision::{Answer, ChoiceAnswer, Question, ScoreAnswer, YesNoAnswer};
//! use allama::mock::MockDecisionProvider;
//! use allama::{DecisionProvider, DecisionRequest, DecisionResponse};
//!
//! # let provider = MockDecisionProvider::with_response(DecisionResponse::new([
//! #     ("label", Answer::from(ChoiceAnswer::new("billing", [("billing", 0.97), ("bug", 0.03)]))),
//! #     ("refund", YesNoAnswer::new(0.99).into()),
//! #     ("urgency", ScoreAnswer::new(0.8, [0.3, 0.6, 0.1]).into()),
//! # ]));
//! # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
//! let request = DecisionRequest::new("nimble", "I was charged twice. Please refund me.")
//!     .question(
//!         "label",
//!         Question::choice("Which label fits this ticket?")
//!             .option("billing", "Payments and refunds")
//!             .option("bug", "Software errors"),
//!     )
//!     .question("refund", Question::yes_no("Is the customer requesting a refund?"))
//!     .question(
//!         "urgency",
//!         Question::score("How urgently does this ticket need a response?")
//!             .levels(["Routine", "Soon", "Immediate"]),
//!     );
//!
//! let response = provider.decide(&request).await?;
//! assert_eq!(response.choice("label").unwrap().chosen, "billing");
//! assert!(response.yes_no("refund").unwrap().is_yes());
//! assert_eq!(response.score("urgency").unwrap().most_likely(), Some(1));
//! # Ok::<(), allama::Error>(()) }).unwrap();
//! # }
//! ```
//!
//! # Typed decisions
//!
//! [`Decision`] describes the questions with a struct and decodes the
//! answers into it; [`DecisionProviderExt::decide_as`] does both around one
//! call. With the `derive` feature, `#[derive(Decision, ChoiceSet,
//! Scale)]` write the impls (see [`Decision`] for an example and the
//! attributes).
//!
#![cfg_attr(
    not(feature = "extract"),
    doc = "[crate::extract]: crate#cargo-features"
)]

mod input;
mod json;
mod named;
pub(crate) mod prepared;
mod provider;
mod question;
mod request;
mod response;
mod typed;

#[cfg(any(test, feature = "mock"))]
pub(crate) mod mock;

pub use input::DecisionInput;
// Exported at the crate root, which is their canonical path.
#[doc(no_inline)]
pub use provider::{
    DecisionCapability, DecisionProvider, DecisionProviderExt, DynDecisionProvider,
};
pub use question::{
    ChoiceOption, ChoiceQuestion, Question, QuestionKind, ScoreLevel, ScoreQuestion, YesNoQuestion,
};
#[doc(no_inline)]
pub use request::{DecisionLimits, DecisionRequest};
#[doc(no_inline)]
pub use response::DecisionResponse;
pub use response::{Answer, ChoiceAnswer, ScoreAnswer, YesNoAnswer};
#[doc(hidden)]
pub use typed::__private;
pub use typed::{
    AnswerMismatch, Choice, ChoiceSet, Decided, Decision, DecisionField, Scale, Score, answer_as,
};

#[cfg(feature = "derive")]
pub use allama_derive::{ChoiceSet, Decision, Scale};
