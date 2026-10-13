//! Model-family and variant detection from Gemini model names.
//!
//! The family picks wire formats only (ADR 0015): the thinking controls
//! (`thinkingLevel` or `thinkingBudget`, and how an effort maps to them),
//! and whether an unsigned function call gets Google's placeholder
//! signature. Nothing is refused or dropped because of it. Unknown names
//! fall back to [`ModelFamily::Other`], which gets the newest documented
//! formats; every model-dependent capability of it is `Unknown`.
//!
//! A [`Variant`] is classified before the generation: an image model or a
//! specialty model of a known generation does not inherit its chat
//! defaults.

use allama::{Error, Result};

use crate::PROVIDER_NAME;

/// The Gemini generation a model name belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelFamily {
    /// Gemini 1.x and 2.0, which predate `thinkingLevel`: a reasoning
    /// config is sent as `thinkingBudget`, and the API decides.
    Legacy,
    /// Gemini 2.5: `thinkingBudget`, optional thought signatures.
    Gemini25 { pro: bool },
    /// Gemini 3 and newer (including the `-latest` aliases): `thinkingLevel`,
    /// mandatory function-call signatures.
    Gemini3 { pro: bool },
    /// Gemma 4 on the Gemini API (`gemma-4-31b-it`, `gemma-4-26b-a4b-it`):
    /// system instructions, function calling, image input, and thinking
    /// switched on (`HIGH`) or off (`MINIMAL`) through `thinkingLevel`
    /// (<https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api>).
    Gemma4,
    /// Anything else: tuned or experimental models, and earlier Gemma
    /// generations, which the Gemini API no longer documents. They get the
    /// newest formats (Gemini 3's).
    Other,
}

/// What kind of model a name denotes, apart from its generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Variant {
    /// A text (chat) model.
    Text,
    /// An image-generation model (`-image`, `nano-banana`).
    Image,
    /// A model for another purpose (`-tts`, `-live`, `gemini-live*`,
    /// `native-audio`, `computer-use`, `robotics`, `embedding`, `imagen`):
    /// nothing is assumed about it.
    Specialty,
}

/// What the encoder knows about the model a request names.
///
/// The provider builds it (`Provider::profile`), so the
/// encoder never parses the id itself; it is where a model-rules hook will
/// plug in. It holds wire-format inputs only: capability answers, the
/// resolver's included, act through preflight, and the encoder refuses or
/// drops nothing because of the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ModelProfile {
    pub(crate) family: ModelFamily,
}

impl ModelProfile {
    pub(crate) fn for_model(model: &str) -> Self {
        Self {
            family: ModelFamily::of(model),
        }
    }
}

impl Variant {
    pub(crate) fn of(model: &str) -> Self {
        let name = base_name(model);
        let has = |marker: &str| name.contains(marker);
        if name.starts_with("imagen")
            || name.starts_with("gemini-live")
            || [
                "embedding",
                "computer-use",
                "robotics",
                "-tts",
                "-live",
                "native-audio",
            ]
            .into_iter()
            .any(has)
        {
            Self::Specialty
        } else if has("-image") || has("nano-banana") {
            Self::Image
        } else {
            Self::Text
        }
    }
}

/// The resource path of `model` in a request URL: `models/<id>` for a bare
/// id or `models/<id>`, and `tunedModels/<id>` as it is.
///
/// The id is pasted into the URL path, and model ids often come from end
/// users (a model picker, a `provider:model` address), so anything else is
/// refused before sending: `<id>` must be letters, digits, `.`, `_` and
/// `-`, and not `.` or `..`. Otherwise `../../v1/tunedModels/x:delete?` would
/// reach another API path with the operator's key.
pub(crate) fn resource_path(model: &str) -> Result<String> {
    let (collection, id) = match model.split_once('/') {
        None => ("models", model),
        Some((collection @ ("models" | "tunedModels"), id)) => (collection, id),
        Some(_) => return Err(invalid_model_id()),
    };
    let valid = !id.is_empty()
        && id != "."
        && id != ".."
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'));
    if !valid {
        return Err(invalid_model_id());
    }
    Ok(format!("{collection}/{id}"))
}

fn invalid_model_id() -> Error {
    Error::invalid_request(
        "the model is not a Gemini model name: expected `<id>`, `models/<id>` or \
         `tunedModels/<id>`, where `<id>` has only letters, digits, `.`, `_` and `-`",
    )
    .authored()
    .with_provider(PROVIDER_NAME)
}

/// The model name without a `models/` (or other path) prefix, lowercased.
fn base_name(model: &str) -> String {
    model
        .rsplit('/')
        .next()
        .unwrap_or(model)
        .to_ascii_lowercase()
}

impl ModelFamily {
    pub(crate) fn of(model: &str) -> Self {
        let name = base_name(model);
        if name.starts_with("gemma-4-") {
            return Self::Gemma4;
        }
        let Some(rest) = name.strip_prefix("gemini-") else {
            return Self::Other;
        };
        let pro = name.contains("-pro");
        if rest.ends_with("-latest") {
            return Self::Gemini3 { pro };
        }
        let version = rest.split('-').next().unwrap_or_default();
        let mut numbers = version.split('.').map(str::parse::<u32>);
        let (Some(Ok(major)), minor) = (numbers.next(), numbers.next()) else {
            return Self::Other;
        };
        let minor = match minor {
            Some(Ok(minor)) => minor,
            Some(Err(_)) => return Self::Other,
            None => 0,
        };
        match (major, minor) {
            (3.., _) => Self::Gemini3 { pro },
            (2, 5..) => Self::Gemini25 { pro },
            _ => Self::Legacy,
        }
    }

    /// Whether the first function call of a model turn must carry a
    /// signature: Gemini 3, and the newest behavior for an unknown id.
    pub(crate) fn validates_call_signatures(self) -> bool {
        matches!(self, Self::Gemini3 { .. } | Self::Other)
    }

    /// Whether the family's thinking is documented (a `Supported` answer).
    pub(crate) fn documents_thinking(self) -> bool {
        matches!(
            self,
            Self::Gemini25 { .. } | Self::Gemini3 { .. } | Self::Gemma4
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Model ids become URL paths only when they are model names.
    #[test]
    fn resource_paths_accept_model_names_only() {
        for (model, path) in [
            ("gemini-2.5-flash", "models/gemini-2.5-flash"),
            ("models/gemini-3.5-flash", "models/gemini-3.5-flash"),
            ("tunedModels/my-model_1", "tunedModels/my-model_1"),
            ("gemini-flash-latest", "models/gemini-flash-latest"),
        ] {
            assert_eq!(resource_path(model).unwrap(), path);
        }
        for model in [
            "../../v1/tunedModels/x:delete?ignored=",
            "models/../files",
            "gemini-2.5-flash:generateContent?key=x",
            "gemini 2.5",
            "files/abc",
            "models/",
            "..",
            ".",
            "",
            "gemini-2.5-flash#x",
            "models/a/b",
        ] {
            let err = resource_path(model).unwrap_err();
            assert!(
                matches!(err, Error::InvalidRequest { .. }),
                "{model}: {err:?}"
            );
        }
    }

    #[test]
    fn classifies_model_names() {
        use ModelFamily::*;
        for (name, family) in [
            ("gemini-2.5-flash", Gemini25 { pro: false }),
            ("models/gemini-2.5-pro", Gemini25 { pro: true }),
            ("gemini-2.5-flash-lite", Gemini25 { pro: false }),
            ("gemini-2.0-flash", Legacy),
            ("gemini-1.5-pro", Legacy),
            ("gemini-3-flash-preview", Gemini3 { pro: false }),
            ("gemini-3.1-pro-preview", Gemini3 { pro: true }),
            ("gemini-3.5-flash-lite", Gemini3 { pro: false }),
            ("gemini-flash-latest", Gemini3 { pro: false }),
            ("gemini-pro-latest", Gemini3 { pro: true }),
            ("gemma-4-31b-it", Gemma4),
            ("models/gemma-4-26b-a4b-it", Gemma4),
            ("gemma-3-27b-it", Other),
            ("gemini-exp-1206", Other),
            ("tunedModels/my-model", Other),
        ] {
            assert_eq!(ModelFamily::of(name), family, "{name}");
        }
    }

    #[test]
    fn classifies_variants_before_generations() {
        use Variant::*;
        for (name, variant) in [
            ("gemini-2.5-flash-image", Image),
            ("gemini-3-pro-image", Image),
            ("gemini-3.1-flash-image", Image),
            ("nano-banana-pro-preview", Image),
            ("gemini-2.5-flash-preview-tts", Specialty),
            ("gemini-2.5-pro-preview-tts", Specialty),
            ("gemini-2.5-flash-native-audio-preview-12-2025", Specialty),
            ("gemini-3.1-flash-live-preview", Specialty),
            ("gemini-live-2.5-flash-preview", Specialty),
            ("gemini-2.5-computer-use-preview-10-2025", Specialty),
            ("gemini-robotics-er-1.5-preview", Specialty),
            ("gemini-embedding-001", Specialty),
            ("imagen-4.0-generate-001", Specialty),
            ("gemini-3.5-flash", Text),
            ("gemma-4-31b-it", Text),
        ] {
            assert_eq!(Variant::of(name), variant, "{name}");
        }
    }
}
