use std::fmt;
use std::str::FromStr;

use allama::{Error, Result};

/// A model address: a provider key and a model id, written
/// `provider:model` (`openai:gpt-5.5`).
///
/// The grammar is fixed, so an address in a configuration file resolves to
/// the same provider and model in every release:
///
/// - The **provider key** comes before the first `:`. It is lowercase
///   ASCII letters, digits and `-`, and starts with a letter or digit
///   (`openai`, `openai-chat`). A [`Registry`](crate::Registry) accepts
///   only such keys.
/// - The **model id** is everything after the first `:`, sent to the
///   provider as written, so it may hold further colons and slashes
///   (`ollama:qwen3:1.7b`, `openrouter:meta-llama/llama-3.3-70b-instruct`).
///   It is not empty and does not start with `:`.
/// - The address holds no whitespace or control characters. Nothing is
///   trimmed or case-folded: `" openai:gpt-5"` and `"OpenAI:gpt-5"` are
///   refused, never read as `openai:gpt-5`.
///
/// Parsing checks the syntax only. Whether a provider is registered for
/// the key is the lookup's question.
///
/// ```
/// use allama_providers::ModelAddress;
///
/// let address: ModelAddress = "ollama:qwen3:1.7b".parse()?;
/// assert_eq!(address.key(), "ollama");
/// assert_eq!(address.model(), "qwen3:1.7b");
/// assert_eq!(address.to_string(), "ollama:qwen3:1.7b");
///
/// assert!("openai: gpt-5".parse::<ModelAddress>().is_err());
/// assert!("OpenAI:gpt-5".parse::<ModelAddress>().is_err());
/// # Ok::<(), allama::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModelAddress {
    key: String,
    model: String,
}

impl ModelAddress {
    /// The address of `model` at the provider registered as `key`.
    ///
    /// # Errors
    ///
    /// Fails with [`Error::Configuration`] when `key` is not a valid
    /// provider key or `model` is not a valid model id (see the
    /// [grammar](ModelAddress)).
    pub fn new(key: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        let (key, model) = (key.into(), model.into());
        let address = format!("{key}:{model}");
        if let Err(reason) = check_key(&key).and_then(|()| check_model(&model)) {
            return Err(invalid(&address, &reason));
        }
        Ok(Self { key, model })
    }

    /// The provider key (`openai`).
    #[must_use]
    pub fn key(&self) -> &str {
        &self.key
    }

    /// The model id, as the provider receives it (`gpt-5.5`).
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The provider key and the model id.
    #[must_use]
    pub fn into_parts(self) -> (String, String) {
        (self.key, self.model)
    }
}

impl FromStr for ModelAddress {
    type Err = Error;

    /// Parse a `provider:model` address.
    ///
    /// # Errors
    ///
    /// Fails with [`Error::Configuration`] when `address` does not follow
    /// the [grammar](ModelAddress); the message says which rule it breaks.
    fn from_str(address: &str) -> Result<Self> {
        if let Some(c) = address
            .chars()
            .find(|c| c.is_whitespace() || c.is_control())
        {
            let what = if c.is_whitespace() {
                "whitespace"
            } else {
                "a control character"
            };
            return Err(invalid(address, &format!("it contains {what}")));
        }
        let Some((key, model)) = address.split_once(':') else {
            return Err(invalid(address, "it has no `:` between provider and model"));
        };
        match check_key(key).and_then(|()| check_model(model)) {
            Ok(()) => Ok(Self {
                key: key.to_owned(),
                model: model.to_owned(),
            }),
            Err(reason) => Err(invalid(address, &reason)),
        }
    }
}

impl fmt::Display for ModelAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.key, self.model)
    }
}

/// Whether `key` is a valid provider key: `[a-z0-9][a-z0-9-]*`.
pub(crate) fn is_valid_key(key: &str) -> bool {
    check_key(key).is_ok()
}

fn check_key(key: &str) -> std::result::Result<(), String> {
    let mut chars = key.chars();
    match chars.next() {
        None => Err("the provider key is empty".to_owned()),
        Some(first) if !(first.is_ascii_lowercase() || first.is_ascii_digit()) => Err(format!(
            "the provider key `{key}` does not start with a lowercase letter or a digit"
        )),
        Some(_) if chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-') => {
            Ok(())
        }
        Some(_) => Err(format!(
            "the provider key `{key}` holds characters other than lowercase letters, digits and `-`"
        )),
    }
}

fn check_model(model: &str) -> std::result::Result<(), String> {
    if model.is_empty() {
        Err("the model id is empty".to_owned())
    } else if model.starts_with(':') {
        Err("the model id starts with `:`".to_owned())
    } else if model.chars().any(|c| c.is_whitespace() || c.is_control()) {
        Err("the model id contains whitespace or a control character".to_owned())
    } else {
        Ok(())
    }
}

fn invalid(address: &str, reason: &str) -> Error {
    Error::configuration(format!(
        "invalid model address {address:?}: {reason}; expected `provider:model`, such as \
         `openai:gpt-5.5`"
    ))
    .authored()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(address: &str) -> Result<ModelAddress> {
        address.parse()
    }

    #[test]
    fn splits_at_the_first_colon() {
        let address = parse("ollama:qwen3:1.7b").unwrap();
        assert_eq!((address.key(), address.model()), ("ollama", "qwen3:1.7b"));
        let address = parse("openrouter:a/b").unwrap();
        assert_eq!((address.key(), address.model()), ("openrouter", "a/b"));
        let address = parse("cloudflare:@cf/meta/llama-3.1-8b-instruct").unwrap();
        assert_eq!(address.model(), "@cf/meta/llama-3.1-8b-instruct");
        let address = parse("openai-chat:gpt-5.5").unwrap();
        assert_eq!(address.key(), "openai-chat");
    }

    #[test]
    fn display_round_trips() {
        for text in [
            "openai:gpt-5.5",
            "ollama:qwen3:1.7b",
            "openrouter:a/b",
            "0x:m",
        ] {
            assert_eq!(parse(text).unwrap().to_string(), text);
        }
        let built = ModelAddress::new("ollama", "qwen3:1.7b").unwrap();
        assert_eq!(built, parse("ollama:qwen3:1.7b").unwrap());
        assert_eq!(
            built.into_parts(),
            ("ollama".to_owned(), "qwen3:1.7b".to_owned())
        );
    }

    #[test]
    fn whitespace_is_refused_not_trimmed() {
        for text in [
            "openai: gpt-5",
            "openai:gpt-5 ",
            "openai:gpt-5\n",
            " openai:gpt-5",
            "openai :gpt-5",
            "openai:gpt\t5",
            "openai:gpt-5\u{7f}",
        ] {
            let err = parse(text).unwrap_err();
            assert!(
                matches!(err, Error::Configuration { .. }),
                "{text:?}: {err:?}"
            );
            assert!(err.message().contains("provider:model"), "{text:?}: {err}");
        }
    }

    #[test]
    fn keys_are_lowercase_letters_digits_and_dashes() {
        for text in [
            "OpenAI:gpt-5",
            "-openai:gpt-5",
            "open_ai:gpt-5",
            "a/b:m",
            "é:m",
        ] {
            let err = parse(text).unwrap_err();
            assert!(err.message().contains("provider key"), "{text:?}: {err}");
        }
    }

    #[test]
    fn malformed_addresses_are_configuration_errors() {
        for text in [
            "gpt-5.5",
            ":gpt-5.5",
            "openai:",
            "ollama::",
            "ollama::qwen3",
            ":",
            "",
            "openai/gpt-5",
        ] {
            let err = parse(text).unwrap_err();
            assert!(
                matches!(err, Error::Configuration { .. }),
                "{text:?}: {err:?}"
            );
            assert!(err.message().contains("provider:model"), "{text:?}: {err}");
        }
    }

    #[test]
    fn new_checks_both_parts() {
        assert!(ModelAddress::new("OpenAI", "gpt-5").is_err());
        assert!(ModelAddress::new("openai", "").is_err());
        assert!(ModelAddress::new("openai", " gpt-5").is_err());
        assert!(ModelAddress::new("a:b", "m").is_err());
    }
}
