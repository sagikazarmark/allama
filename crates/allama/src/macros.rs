//! Internal macros shared by the core types.

/// The text conversions of a type built from one string: `From<&str>`,
/// `From<String>`, `From<&String>`, `From<Box<str>>` and
/// `From<Cow<'_, str>>`, so every text-carrying type accepts the same
/// sources.
///
/// `$build` turns the owned `String` into the type, e.g. `Self::Text` or
/// `Self::new`.
macro_rules! text_conversions {
    ($ty:ty, $build:expr) => {
        impl From<String> for $ty {
            fn from(text: String) -> Self {
                ($build)(text)
            }
        }

        impl From<&str> for $ty {
            fn from(text: &str) -> Self {
                Self::from(text.to_owned())
            }
        }

        impl From<&String> for $ty {
            fn from(text: &String) -> Self {
                Self::from(text.clone())
            }
        }

        impl From<Box<str>> for $ty {
            fn from(text: Box<str>) -> Self {
                Self::from(String::from(text))
            }
        }

        impl From<::std::borrow::Cow<'_, str>> for $ty {
            fn from(text: ::std::borrow::Cow<'_, str>) -> Self {
                Self::from(text.into_owned())
            }
        }
    };
}

pub(crate) use text_conversions;

/// The names of a fieldless configuration enum: `as_str`, `Display` and a
/// fallible `FromStr` from one list, so the three cannot disagree.
///
/// List every variant with the name serde gives it; the `match` in
/// `as_str` fails to compile when a variant is missing. `$what` names the
/// type in a [`ParseEnumError`](crate::ParseEnumError), such as
/// `"reasoning effort"`. Under `cfg(test)` it also defines `VARIANTS`,
/// every variant in list order, for round-trip tests.
macro_rules! enum_names {
    ($ty:ident, $what:literal, { $($variant:ident => $name:literal),+ $(,)? }) => {
        impl $ty {
            /// The stable snake_case name, as serialized.
            ///
            /// [`Display`](std::fmt::Display) writes it and
            /// [`FromStr`](std::str::FromStr) (`str::parse`) reads it back.
            #[must_use]
            pub fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $name,)+
                }
            }

            #[cfg(test)]
            pub(crate) const VARIANTS: &'static [Self] = &[$(Self::$variant),+];
        }

        impl ::std::fmt::Display for $ty {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl ::std::str::FromStr for $ty {
            type Err = $crate::ParseEnumError;

            /// Parse a name [`as_str`](Self::as_str) returns.
            fn from_str(value: &str) -> ::std::result::Result<Self, Self::Err> {
                match value {
                    $($name => Ok(Self::$variant),)+
                    _ => Err($crate::ParseEnumError::new($what, value, &[$($name),+])),
                }
            }
        }
    };
}

pub(crate) use enum_names;

#[cfg(test)]
mod tests {
    use std::fmt::Debug;
    use std::str::FromStr;

    use serde::Serialize;
    use serde::de::DeserializeOwned;

    use crate::{
        ChatCapability, EmbeddingInputType, ParseEnumError, ReasoningEffort, ReasoningOutput,
        UnsupportedPolicy,
    };

    /// For every variant, serde, `as_str`, `Display` and `parse` agree.
    fn names_agree<T>(variants: &[T], expected: &[&str])
    where
        T: Copy + Debug + PartialEq + Serialize + DeserializeOwned,
        T: std::fmt::Display + FromStr<Err = ParseEnumError>,
    {
        assert_eq!(variants.len(), expected.len());
        for (&variant, &name) in variants.iter().zip(expected) {
            assert_eq!(variant.to_string(), name);
            assert_eq!(serde_json::to_value(variant).unwrap(), name);
            assert_eq!(serde_json::from_value::<T>(name.into()).unwrap(), variant);
            assert_eq!(name.parse::<T>().unwrap(), variant);
        }
        let err = "not a name".parse::<T>().unwrap_err();
        assert_eq!(err.value, "not a name");
        assert_eq!(err.expected, expected);
    }

    #[test]
    fn every_config_enum_round_trips_through_its_name() {
        names_agree(
            ReasoningEffort::VARIANTS,
            &["none", "minimal", "low", "medium", "high", "xhigh"],
        );
        names_agree(ReasoningOutput::VARIANTS, &["omitted", "summary", "full"]);
        names_agree(UnsupportedPolicy::VARIANTS, &["warn", "error"]);
        names_agree(EmbeddingInputType::VARIANTS, &["query", "document"]);
        names_agree(
            ChatCapability::VARIANTS,
            &[
                "tool_calls",
                "parallel_tool_calls",
                "streaming",
                "native_streaming",
                "stream_usage",
                "image_input",
                "image_detail",
                "image_output",
                "image_replay",
                "file_input",
                "audio_input",
                "video_input",
                "structured_output",
                "reasoning_output",
                "reasoning_replay",
                "reasoning_config",
                "prompt_cache",
                "forced_tool_choice",
            ],
        );
        assert_eq!(ReasoningEffort::VARIANTS, ReasoningEffort::ALL);
        assert_eq!(ChatCapability::VARIANTS, ChatCapability::ALL);
    }

    /// The capability enums and the capability
    /// answer persist as snake_case names, like `ChatCapability`.
    #[test]
    fn every_capability_name_round_trips() {
        use crate::{CapabilitySupport, DecisionCapability, EmbeddingCapability};

        names_agree(
            EmbeddingCapability::VARIANTS,
            &[
                "batch_input",
                "output_dimensions",
                "image_input",
                "composite_input",
                "input_type",
            ],
        );
        names_agree(
            DecisionCapability::VARIANTS,
            &["yes_no", "choice", "score", "locate", "image_input"],
        );
        names_agree(
            CapabilitySupport::VARIANTS,
            &["supported", "unsupported", "unknown"],
        );
        assert_eq!(EmbeddingCapability::VARIANTS, EmbeddingCapability::ALL);
        assert_eq!(DecisionCapability::VARIANTS, DecisionCapability::ALL);
    }

    /// The extraction enums round-trip the same way; their names are not
    /// persisted names (the module is unstable, ADR 0018).
    #[test]
    fn every_extraction_model_enum_round_trips_through_its_name() {
        use crate::ExtractionCapability;
        use crate::extraction::{
            DurationUnit, EstimateScale, FieldKind, ForecastPeriod, ListItemKind,
        };

        let kinds = [
            "text", "number", "date", "time", "datetime", "duration", "list", "estimate",
            "forecast",
        ];
        names_agree(FieldKind::VARIANTS, &kinds);
        let mut capabilities = kinds.to_vec();
        capabilities.push("image_input");
        names_agree(ExtractionCapability::VARIANTS, &capabilities);
        names_agree(
            DurationUnit::VARIANTS,
            &[
                "seconds", "minutes", "hours", "days", "weeks", "months", "years",
            ],
        );
        names_agree(ListItemKind::VARIANTS, &["text", "number", "date"]);
        names_agree(EstimateScale::VARIANTS, &["linear", "log"]);
        names_agree(
            ForecastPeriod::VARIANTS,
            &[
                "step", "minute", "hour", "day", "week", "month", "quarter", "year",
            ],
        );
        assert_eq!(FieldKind::VARIANTS, FieldKind::ALL);
        assert_eq!(ExtractionCapability::VARIANTS, ExtractionCapability::ALL);
    }

    #[cfg(feature = "chat-extract")]
    #[test]
    fn every_extraction_enum_round_trips_through_its_name() {
        use crate::chat::extract::{ChatExtractionMode, SchemaEnforcement};

        names_agree(
            ChatExtractionMode::VARIANTS,
            &["auto", "native", "forced_tool"],
        );
        names_agree(
            SchemaEnforcement::VARIANTS,
            &["exact", "approximate", "not_enforced"],
        );
    }

    #[test]
    fn serde_and_from_str_accept_the_same_reasoning_effort_names() {
        assert_eq!(
            serde_json::to_string(&ReasoningEffort::XHigh).unwrap(),
            r#""xhigh""#
        );
        for name in ["xhigh", "x_high"] {
            assert_eq!(
                serde_json::from_str::<ReasoningEffort>(&format!("\"{name}\"")).ok(),
                name.parse::<ReasoningEffort>().ok(),
                "{name}"
            );
        }
    }

    #[test]
    fn a_parse_error_names_the_type_and_every_value() {
        let err = "hgih".parse::<ReasoningEffort>().unwrap_err();
        assert_eq!(
            err.to_string(),
            r#"unknown reasoning effort "hgih"; expected one of: none, minimal, low, medium, high, xhigh"#
        );
        assert_eq!(err.type_name, "reasoning effort");
    }
}
