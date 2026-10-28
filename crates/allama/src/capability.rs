use std::fmt;

use serde::{Deserialize, Serialize};

use crate::macros::enum_names;
use crate::{ChatCapability, DecisionCapability, EmbeddingCapability, ExtractionCapability};

/// A portable capability enum: [`ChatCapability`], [`EmbeddingCapability`],
/// [`DecisionCapability`] or [`ExtractionCapability`].
///
/// It names the capabilities a [`Validation`](crate::Validation) reports as
/// [`relied_on_unknown`](crate::Validation::relied_on_unknown). It is
/// sealed: each operation has its own enum.
pub trait Capability: Copy + Eq + fmt::Debug + Send + Sync + 'static + sealed::Sealed {}

impl Capability for ChatCapability {}
impl Capability for EmbeddingCapability {}
impl Capability for DecisionCapability {}
impl Capability for ExtractionCapability {}

pub(crate) mod sealed {
    use crate::{ChatCapability, DecisionCapability, EmbeddingCapability, ExtractionCapability};

    /// A capability of any operation, so one record holds them all.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum AnyCapability {
        Chat(ChatCapability),
        Embedding(EmbeddingCapability),
        Decision(DecisionCapability),
        Extraction(ExtractionCapability),
    }

    pub trait Sealed: Sized {
        fn into_any(self) -> AnyCapability;
        fn from_any(any: AnyCapability) -> Option<Self>;
    }

    impl Sealed for ChatCapability {
        fn into_any(self) -> AnyCapability {
            AnyCapability::Chat(self)
        }

        fn from_any(any: AnyCapability) -> Option<Self> {
            match any {
                AnyCapability::Chat(capability) => Some(capability),
                _ => None,
            }
        }
    }

    impl Sealed for EmbeddingCapability {
        fn into_any(self) -> AnyCapability {
            AnyCapability::Embedding(self)
        }

        fn from_any(any: AnyCapability) -> Option<Self> {
            match any {
                AnyCapability::Embedding(capability) => Some(capability),
                _ => None,
            }
        }
    }

    impl Sealed for DecisionCapability {
        fn into_any(self) -> AnyCapability {
            AnyCapability::Decision(self)
        }

        fn from_any(any: AnyCapability) -> Option<Self> {
            match any {
                AnyCapability::Decision(capability) => Some(capability),
                _ => None,
            }
        }
    }

    impl Sealed for ExtractionCapability {
        fn into_any(self) -> AnyCapability {
            AnyCapability::Extraction(self)
        }

        fn from_any(any: AnyCapability) -> Option<Self> {
            match any {
                AnyCapability::Extraction(capability) => Some(capability),
                _ => None,
            }
        }
    }
}

/// Support state for a provider/model capability query.
///
/// The enum is intentionally a closed three-state set (`Supported`,
/// `Unsupported`, `Unknown`). It is not `#[non_exhaustive]` because the answer
/// to a capability query is either yes, no, or unknown — there is no fourth
/// state to add later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilitySupport {
    /// The provider explicitly supports this capability for the queried model.
    Supported,
    /// The provider explicitly does not support this capability for the queried model.
    Unsupported,
    /// The provider does not provide a definitive answer for this capability.
    ///
    /// Treat this as "unknown, ask carefully" rather than as support.
    #[default]
    Unknown,
}

enum_names!(CapabilitySupport, "capability support", {
    Supported => "supported",
    Unsupported => "unsupported",
    Unknown => "unknown",
});

impl CapabilitySupport {
    /// Returns `true` only when the provider explicitly reports support.
    ///
    /// [`Unknown`](Self::Unknown) is **not** treated as supported: code that
    /// gates behavior on capability reporting should opt in only when the
    /// provider is certain.
    #[must_use]
    pub const fn is_supported(self) -> bool {
        matches!(self, Self::Supported)
    }

    /// Returns `true` when the provider gave a definitive answer, regardless
    /// of whether that answer was positive or negative.
    ///
    /// Equivalent to `self != CapabilitySupport::Unknown`. Useful when a
    /// caller wants to distinguish "the provider told us" from "we have to
    /// guess."
    #[must_use]
    pub const fn is_known(self) -> bool {
        !matches!(self, Self::Unknown)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_supported_matches_only_supported() {
        assert!(CapabilitySupport::Supported.is_supported());
        assert!(!CapabilitySupport::Unsupported.is_supported());
        assert!(!CapabilitySupport::Unknown.is_supported());
    }

    #[test]
    fn is_known_matches_supported_and_unsupported() {
        assert!(CapabilitySupport::Supported.is_known());
        assert!(CapabilitySupport::Unsupported.is_known());
        assert!(!CapabilitySupport::Unknown.is_known());
    }

    #[test]
    fn default_is_unknown_and_not_known() {
        let value = CapabilitySupport::default();
        assert_eq!(value, CapabilitySupport::Unknown);
        assert!(!value.is_known());
        assert!(!value.is_supported());
    }
}
