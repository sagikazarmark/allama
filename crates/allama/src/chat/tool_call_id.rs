use std::borrow::Cow;

/// What a wire API accepts as a tool call id, and how an encoder turns an id
/// it does not accept into one it does.
///
/// A tool call id is only a correlation key: it pairs a call with its result
/// inside one request. A history continued on another provider carries the
/// ids the first provider issued (or its adapter made up), and APIs disagree
/// on what they accept. An encoder sends every id through
/// [`conform`](Self::conform), for the calls of assistant messages and the
/// results that answer them alike, so the pairing survives:
///
/// - an id the format accepts is sent unchanged;
/// - any other id becomes one derived from it by a stable hash. The same id
///   always maps to the same replacement, in every request and every
///   release, so a call and its result stay paired and a cached prompt
///   prefix stays the same; distinct ids map to distinct replacements, up
///   to a hash collision.
///
/// Nothing is lost, so no warning is due: the replacement names the same
/// call, and the model's next reply carries new ids of its own. An empty id
/// names no call and is left for the API to refuse.
///
/// ```
/// use allama::adapter::chat::ToolCallIdFormat;
///
/// // Anthropic accepts letters, digits, `_` and `-`.
/// let anthropic = ToolCallIdFormat::UrlSafe;
/// assert_eq!(anthropic.conform("toolu_01A"), "toolu_01A");
/// let mapped = anthropic.conform("vendor:resp/1:call:0");
/// assert!(anthropic.accepts(&mapped));
/// assert!(mapped.starts_with("vendor_resp_1_call_0_"));
///
/// // Mistral takes exactly nine letters and digits.
/// let mistral = ToolCallIdFormat::Alphanumeric(9);
/// assert_eq!(mistral.conform("D681PevKs"), "D681PevKs");
/// assert_eq!(mistral.conform("toolu_01A").len(), 9);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ToolCallIdFormat {
    /// Any id (Gemini's `functionCall.id`).
    #[default]
    Any,
    /// At most this many characters, any characters (OpenAI Chat
    /// Completions and Responses: 64). A longer id keeps its first
    /// characters and ends in a hash.
    MaxLength(usize),
    /// Letters, digits, `_` and `-` (Anthropic: `^[a-zA-Z0-9_-]+$`). Other
    /// characters become `_`, and the id ends in a hash; the replacement
    /// is at most 57 characters long.
    UrlSafe,
    /// Exactly this many ASCII letters and digits (Mistral, and servers
    /// running Mistral models through `mistral_common`: 9). The
    /// replacement is all hash.
    Alphanumeric(usize),
}

impl ToolCallIdFormat {
    /// Whether this format accepts `id` as it is.
    #[must_use]
    pub fn accepts(self, id: &str) -> bool {
        match self {
            Self::Any => true,
            Self::MaxLength(max) => id.chars().count() <= max,
            Self::UrlSafe => !id.is_empty() && id.chars().all(is_url_safe),
            Self::Alphanumeric(len) => {
                id.len() == len && id.chars().all(|c| c.is_ascii_alphanumeric())
            }
        }
    }

    /// `id` as this format accepts it: unchanged when it already is (or is
    /// empty), otherwise a deterministic replacement derived from it.
    #[must_use]
    pub fn conform(self, id: &str) -> Cow<'_, str> {
        if id.is_empty() || self.accepts(id) {
            return Cow::Borrowed(id);
        }
        let replacement = match self {
            Self::Any => return Cow::Borrowed(id),
            Self::MaxLength(max) if max > HASH_HEX_LEN + 1 => {
                let prefix: String = id.chars().take(max - HASH_HEX_LEN - 1).collect();
                format!("{prefix}_{:016x}", hash(id, 0))
            }
            Self::MaxLength(max) => alphanumeric_digest(id, max),
            Self::UrlSafe => {
                let prefix: String = id
                    .chars()
                    .take(URL_SAFE_PREFIX_LEN)
                    .map(|c| if is_url_safe(c) { c } else { '_' })
                    .collect();
                format!("{prefix}_{:016x}", hash(id, 0))
            }
            Self::Alphanumeric(len) => alphanumeric_digest(id, len),
        };
        Cow::Owned(replacement)
    }
}

/// Hex digits of the hash a replaced id ends in.
const HASH_HEX_LEN: usize = 16;

/// Characters of the original id a [`ToolCallIdFormat::UrlSafe`]
/// replacement keeps, so it also fits OpenAI's 64.
const URL_SAFE_PREFIX_LEN: usize = 40;

const BASE62: &[u8; 62] = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

fn is_url_safe(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

/// 64-bit FNV-1a of `round` then `id`: stable across platforms and
/// releases, unlike `std`'s hasher.
fn hash(id: &str, round: u64) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    round
        .to_le_bytes()
        .iter()
        .chain(id.as_bytes())
        .fold(OFFSET, |h, byte| (h ^ u64::from(*byte)).wrapping_mul(PRIME))
}

/// `len` base-62 digits derived from `id`, ten per hash round.
fn alphanumeric_digest(id: &str, len: usize) -> String {
    let mut out = String::with_capacity(len);
    let mut round = 0;
    while out.len() < len {
        let mut h = hash(id, round);
        for _ in 0..10 {
            if out.len() == len {
                break;
            }
            out.push(char::from(BASE62[(h % 62) as usize]));
            h /= 62;
        }
        round += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const FOREIGN: [&str; 6] = [
        "gemini:resp:call:0",
        "call_abc123",
        "toolu_01ABCdef",
        "D681PevKs",
        "a/b c+d",
        "ünïcödé-id",
    ];

    #[test]
    fn accepted_ids_pass_through() {
        assert!(matches!(
            ToolCallIdFormat::UrlSafe.conform("toolu_01-x"),
            Cow::Borrowed("toolu_01-x")
        ));
        assert!(matches!(
            ToolCallIdFormat::Alphanumeric(9).conform("D681PevKs"),
            Cow::Borrowed(_)
        ));
        assert!(matches!(
            ToolCallIdFormat::MaxLength(64).conform("gemini:resp:call:0"),
            Cow::Borrowed(_)
        ));
        assert_eq!(ToolCallIdFormat::Any.conform("a b:c"), "a b:c");
        // An empty id names no call; the API refuses it, not the encoder.
        assert_eq!(ToolCallIdFormat::UrlSafe.conform(""), "");
    }

    #[test]
    fn replacements_are_accepted_deterministic_and_distinct() {
        for format in [
            ToolCallIdFormat::UrlSafe,
            ToolCallIdFormat::Alphanumeric(9),
            ToolCallIdFormat::Alphanumeric(24),
            ToolCallIdFormat::MaxLength(12),
            ToolCallIdFormat::MaxLength(8),
        ] {
            let mut seen = std::collections::HashSet::new();
            for id in FOREIGN {
                let out = format.conform(id);
                assert!(format.accepts(&out), "{format:?} {id} -> {out}");
                assert_eq!(out, format.conform(id), "{format:?} {id}");
                assert!(
                    seen.insert(out.into_owned()),
                    "{format:?} collision on {id}"
                );
            }
        }
    }

    #[test]
    fn ids_differing_only_in_refused_characters_stay_distinct() {
        let format = ToolCallIdFormat::UrlSafe;
        let (a, b) = (format.conform("x:1"), format.conform("x/1"));
        assert!(a.starts_with("x_1_") && b.starts_with("x_1_"));
        assert_ne!(a, b);
    }

    #[test]
    fn replacements_stay_within_openais_limit() {
        let long = "x:".repeat(200);
        assert_eq!(ToolCallIdFormat::UrlSafe.conform(&long).len(), 57);
        let capped = ToolCallIdFormat::MaxLength(64).conform(&long);
        assert_eq!(capped.chars().count(), 64);
        assert!(capped.starts_with("x:x:"));
    }

    /// The replacements are part of the contract (a cached prompt prefix
    /// depends on them); a change here is a breaking change.
    #[test]
    fn replacements_are_stable() {
        assert_eq!(
            ToolCallIdFormat::UrlSafe.conform("gemini:r:call:0"),
            "gemini_r_call_0_9b6d3111a480fdea"
        );
        assert_eq!(
            ToolCallIdFormat::Alphanumeric(9).conform("gemini:r:call:0"),
            "kPfSHykzk"
        );
    }
}
