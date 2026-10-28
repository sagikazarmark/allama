//! Extraction models: typed fields answered with values.
//!
//! An extraction model reads an input and answers named fields with typed
//! values instead of generating text: a span of the input
//! ([`Field::text`]), a number read or computed ([`Field::number`]), a date,
//! time or datetime, a duration, a list, an estimate (a distribution inside
//! bounds the caller declares) or a forecast (one estimate per step ahead
//! of a series). Each scalar answer is a value or null, with the
//! probability that the input does not determine it. Providers implement
//! [`ExtractionProvider`].
//!
//! The portable model is the typed fields, their answer constraints, and
//! the answers: values, null probabilities, spans into the input, and
//! interval summaries of estimates. Provider statistics such as a
//! "confidence" per answer, or the guess a model would have made for a
//! null, are computed differently by each provider and live in
//! [`ExtractionResponse::provider_data`]; request limits are queried with
//! [`ExtractionProvider::extraction_limits`] rather than fixed.
//!
//! This is not [chat extraction][crate::chat::extract], which drives a chat
//! model to produce a typed value: an extraction model generates no tokens,
//! answers every field in one pass, and reports where in the input each
//! text answer came from.
//!
//! # Stability
//!
//! **Unstable** (ADR 0018). The module is grounded in one provider, so it
//! ships without the second provider allama otherwise requires. Its types
//! and this trait may change in any minor release (`0.x` to `0.(x+1)`), with
//! a release note but no migration path, until a second provider
//! implements it; its serialized forms are same-version. Nothing here is a
//! persisted name.
//!
//! ```
//! # #[cfg(feature = "mock")] {
//! use allama::extraction::{Answer, Field, ListItem, ListAnswer, NumberAnswer, TextAnswer, TextSpan};
//! use allama::mock::MockExtractionProvider;
//! use allama::{ExtractionProvider, ExtractionRequest, ExtractionResponse};
//!
//! let email = "Order A-88412 arrived with a cracked screen. I paid $649.00 plus $52.10 tax.";
//! # let provider = MockExtractionProvider::with_response(ExtractionResponse::new([
//! #     ("order_id", Answer::from(TextAnswer::new(Some("A-88412")).span(TextSpan::new(6, 13)).null_probability(3.4e-6))),
//! #     ("amount_paid", Answer::from(NumberAnswer::new(Some(701.1)).null_probability(1.6e-5))),
//! #     ("damage", Answer::from(ListAnswer::new([ListItem::text("cracked screen").span(TextSpan::new(34, 48))]))),
//! #     ("tracking_number", Answer::from(TextAnswer::new(None::<String>).null_probability(1.0))),
//! # ]));
//! # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
//! let request = ExtractionRequest::new("@cf/cloudflare/clairic", email)
//!     .field("order_id", Field::text().instructions("Order number"))
//!     .field(
//!         "amount_paid",
//!         Field::number().decimals(2).unit("USD").instructions("Total paid including tax"),
//!     )
//!     .field("damage", Field::list(allama::extraction::ListItemKind::Text))
//!     .field("tracking_number", Field::text());
//!
//! let response = provider.extract_values(&request).await?;
//! let order_id = response.text("order_id").unwrap();
//! assert_eq!(order_id.value.as_deref(), Some("A-88412"));
//! assert_eq!(order_id.span.unwrap().text(&request.input), Some("A-88412"));
//! assert_eq!(response.number("amount_paid").unwrap().value, Some(701.1));
//! assert_eq!(response.list("damage").unwrap().texts().collect::<Vec<_>>(), ["cracked screen"]);
//! assert!(response.text("tracking_number").unwrap().is_null());
//! # Ok::<(), allama::Error>(()) }).unwrap();
//! # }
//! ```
#![cfg_attr(
    not(feature = "chat-extract"),
    doc = "[crate::chat::extract]: crate#cargo-features"
)]

mod field;
pub(crate) mod prepared;
mod provider;
mod request;
mod response;
mod value;

#[cfg(any(test, feature = "mock"))]
pub(crate) mod mock;

pub use field::{
    DateField, DateTimeField, DurationField, DurationUnit, EstimateField, EstimateScale, Field,
    FieldKind, ForecastField, ForecastPeriod, ListField, ListItemKind, NumberField, TextField,
    TimeField,
};
// Exported at the crate root, which is their canonical path.
#[doc(no_inline)]
pub use provider::{DynExtractionProvider, ExtractionCapability, ExtractionProvider};
#[doc(no_inline)]
pub use request::{ExtractionLimits, ExtractionRequest};
#[doc(no_inline)]
pub use response::ExtractionResponse;
pub use response::{
    Answer, DateAnswer, DateTimeAnswer, DurationAnswer, EstimateAnswer, ForecastAnswer, ListAnswer,
    ListItem, ListValue, NumberAnswer, PredictionInterval, TextAnswer, TimeAnswer,
};
pub use value::{Date, DateTime, TextSpan, Time};
