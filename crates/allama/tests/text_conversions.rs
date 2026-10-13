//! Every text-carrying type accepts the same string sources: `&str`,
//! `String`, `&String`, `Box<str>` and `Cow<'_, str>`.

use std::borrow::Cow;
use std::fmt::Debug;

use allama::decision::DecisionInput;
use allama::{ContentPart, EmbeddingInput, SystemPrompt, ToolResultContent, UserContent};

/// The bound is the compile test: a type missing one conversion fails to
/// build here.
fn converts_from_every_string_source<T>(expected: T)
where
    T: Debug + PartialEq,
    T: for<'a> From<&'a str> + From<String> + for<'a> From<&'a String>,
    T: From<Box<str>> + for<'a> From<Cow<'a, str>>,
{
    let owned = String::from("hello");
    assert_eq!(T::from("hello"), expected);
    assert_eq!(T::from(owned.clone()), expected);
    assert_eq!(T::from(&owned), expected);
    assert_eq!(T::from(Box::<str>::from("hello")), expected);
    assert_eq!(T::from(Cow::Borrowed("hello")), expected);
    assert_eq!(T::from(Cow::<str>::Owned(owned)), expected);
}

#[test]
fn every_text_type_converts_from_every_string_source() {
    converts_from_every_string_source(UserContent::Text("hello".into()));
    converts_from_every_string_source(SystemPrompt::new("hello"));
    converts_from_every_string_source(ContentPart::text("hello"));
    converts_from_every_string_source(ToolResultContent::text("hello"));
    converts_from_every_string_source(DecisionInput::text("hello"));
    converts_from_every_string_source(EmbeddingInput::text("hello"));
}

#[test]
fn builders_take_a_borrowed_string() {
    let prompt = String::from("hello");
    let request = allama::ChatRequest::new("m").system(&prompt).user(&prompt);
    assert_eq!(request.messages.len(), 1);
}
