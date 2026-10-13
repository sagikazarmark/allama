//! Portable prompt-caching intent.
//!
//! Caching is expressed as intent, not as a provider mechanism:
//!
//! - a [`CacheBreakpoint`] marks the end of a cacheable prefix on a system
//!   prompt, a message, or a tool definition;
//! - a request-level [`PromptCache`] carries a cache key, a preferred
//!   retention, and whether to cache the prompt prefix automatically.
//!
//! Providers with explicit breakpoints (Anthropic) map them to their own
//! markers. Providers with automatic prefix caching (OpenAI, Gemini) honor
//! the key and retention where supported and treat breakpoints as already
//! satisfied. Providers without caching ignore the hints with a warning.

use serde::{Deserialize, Serialize};

/// Preferred lifetime of a cache entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CacheRetention {
    /// The provider's short retention (minutes, e.g. Anthropic's 5m).
    Short,
    /// The provider's long retention (e.g. Anthropic 1h, OpenAI 24h).
    Long,
}

/// Marks the end of a cacheable prompt prefix.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CacheBreakpoint {
    /// Preferred retention; `None` uses the provider default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retention: Option<CacheRetention>,
}

impl CacheBreakpoint {
    /// A breakpoint with the provider's default retention.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A breakpoint with the given retention.
    #[must_use]
    pub fn with_retention(retention: CacheRetention) -> Self {
        Self {
            retention: Some(retention),
        }
    }
}

/// Request-level prompt-cache settings.
///
/// A key and a retention are hints for providers that take them.
///
/// # Automatic caching
///
/// Some providers cache every prompt prefix on their own (OpenAI, Gemini
/// implicit caching). Others cache only where asked, and bill cache writes
/// above the plain input price: Anthropic charges 1.25x for 5 minutes and 2x
/// for an hour. So automatic prefix caching is explicit intent, set with
/// [`PromptCache::automatic`]. A key or a retention alone never turns it on.
///
/// ```
/// use allama::{CacheRetention, ChatRequest, PromptCache};
///
/// // Routing only: groups requests on providers that take a cache key, and
/// // turns no caching on where caching is opt-in.
/// let routed = ChatRequest::new("model").cache(PromptCache::new().key("session-1"));
///
/// // Cache the whole prefix where the provider makes that opt-in, for an
/// // hour where it can. Providers that cache prefixes anyway already
/// // satisfy it.
/// let cached = ChatRequest::new("model")
///     .cache(PromptCache::automatic().retention(CacheRetention::Long));
/// # let _ = (routed, cached);
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PromptCache {
    /// Key that groups requests sharing a prefix (e.g. a session id).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    /// Preferred retention for automatically cached prefixes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retention: Option<CacheRetention>,
    /// Cache the prompt prefix automatically, without breakpoints.
    ///
    /// Where caching is opt-in and cache writes cost extra (Anthropic), this
    /// turns automatic prefix caching on. Where the provider caches prefixes
    /// on its own (OpenAI, Gemini), it is already satisfied and nothing is
    /// sent. A provider without prompt caching drops it with a warning, as
    /// any other cache hint.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub automatic: bool,
}

impl PromptCache {
    /// Empty settings: no key, no retention, no automatic caching.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Settings that ask for automatic prefix caching
    /// ([`automatic`](Self::automatic)).
    #[must_use]
    pub fn automatic() -> Self {
        Self {
            automatic: true,
            ..Self::default()
        }
    }

    /// Set the cache key.
    #[must_use]
    pub fn key(mut self, key: impl Into<String>) -> Self {
        self.key = Some(key.into());
        self
    }

    /// Set the retention.
    #[must_use]
    pub fn retention(mut self, retention: CacheRetention) -> Self {
        self.retention = Some(retention);
        self
    }
}
