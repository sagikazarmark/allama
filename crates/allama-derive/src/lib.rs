#![forbid(unsafe_code)]
//! Derive macros for `allama`'s typed decisions.
//!
//! Use them through `allama` with the `derive` feature, which re-exports them
//! from `allama::decision` next to the traits they implement. The
//! `allama::decision::Decision` documentation describes the attributes.

mod attr;
mod choice_set;
mod decision;
mod scale;

use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

/// Derives `ChoiceSet` and `DecisionField` for an enum of at least two
/// unit variants: each variant is an option of a choice question.
///
/// ```ignore
/// #[derive(ChoiceSet)]
/// #[decision(rename_all = "lowercase")]
/// enum Label {
///     /// Payments and refunds
///     Billing,
///     #[decision(rename = "bug", description = "Software errors")]
///     Defect,
/// }
/// ```
///
/// - `#[decision(rename_all = "...")]` on the enum: `lowercase`,
///   `UPPERCASE`, `snake_case`, `kebab-case`, `SCREAMING_SNAKE_CASE`,
///   `camelCase`, or `PascalCase`. Without it, keys are the variant names.
/// - `#[decision(crate = "path")]` on the enum: the path to `allama`
///   (default `::allama`).
/// - `#[decision(rename = "...")]` on a variant: its option key.
/// - `#[decision(description = "...")]` on a variant: what the option means.
///   Without it, the first paragraph of the variant's doc comment is the
///   description (lines trimmed and joined with spaces); with neither, the
///   option has none.
#[proc_macro_derive(ChoiceSet, attributes(decision))]
pub fn derive_choice_set(input: TokenStream) -> TokenStream {
    expand(input, choice_set::expand)
}

/// Derives `Scale` for an enum of at least two unit variants: each variant
/// is a level of a score question, lowest first.
///
/// ```ignore
/// #[derive(Scale)]
/// enum Urgency {
///     #[decision(description = "No time pressure")]
///     Routine,
///     #[decision(description = "A customer is inconvenienced")]
///     Soon,
/// }
/// ```
///
/// Takes the same attributes as `ChoiceSet`: `rename_all` and `crate` on
/// the enum, `rename` (the level label; default: the variant name) and
/// `description` on a variant, which falls back to the first paragraph of
/// the variant's doc comment.
#[proc_macro_derive(Scale, attributes(decision))]
pub fn derive_scale(input: TokenStream) -> TokenStream {
    expand(input, scale::expand)
}

/// Derives `Decision` for a struct whose named fields are questions.
///
/// ```ignore
/// #[derive(Decision)]
/// struct Triage {
///     /// Which label fits this ticket?
///     label: Choice<Label>,
///     #[decision(
///         instructions = "Is the customer requesting a refund?",
///         yes = "The customer requests a refund"
///     )]
///     refund: YesNoAnswer,
/// }
/// ```
///
/// - `#[decision(instructions = "...")]` on a field: what the question asks.
///   Without it, the first paragraph of the field's doc comment is the
///   instructions (lines trimmed and joined with spaces); a field with
///   neither is an error.
/// - `#[decision(rename = "...")]` on a field: the question name (default:
///   the field name).
/// - `#[decision(yes = "...", no = "...")]` on a yes/no field: what each
///   outcome means.
/// - `#[decision(crate = "path")]` on the struct: the path to `allama`
///   (default `::allama`).
///
/// Field types must implement `DecisionField`: `YesNoAnswer`, `Choice<T>`,
/// `Score<T>`, or an enum deriving `ChoiceSet`.
#[proc_macro_derive(Decision, attributes(decision))]
pub fn derive_decision(input: TokenStream) -> TokenStream {
    expand(input, decision::expand)
}

fn expand(
    input: TokenStream,
    f: fn(DeriveInput) -> syn::Result<proc_macro2::TokenStream>,
) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    f(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
