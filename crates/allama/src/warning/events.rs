//! The `tracing` side of [`emit`](super::emit): a target per kind, the
//! level a code decides, and the process-wide set of codes already logged.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::{LazyLock, Mutex, PoisonError};

use super::{Subject, Warning, WarningKind, attempt};

/// How many `(provider, model, kind, code)` keys the dedupe set keeps.
pub(super) const DEDUPE_CAPACITY: usize = 4096;

/// A warning event that has not gone out yet: emitted inside an attempt
/// scope, it waits for the attempt to commit. The level and the dedupe slot
/// are decided when it goes out ([`fire`]), so a discarded attempt uses up
/// no key.
pub(crate) struct Pending {
    provider: String,
    model: String,
    warning: Warning,
    /// The span current when the warning was emitted, which the event
    /// belongs to wherever it is flushed.
    span: tracing::Span,
}

/// Emit `warning`'s event, or hold it in the nearest open attempt scope.
pub(super) fn emit(provider: &str, model: &str, warning: &Warning) {
    if !enabled(warning.kind) {
        return;
    }
    let pending = Pending {
        provider: provider.to_owned(),
        model: model.to_owned(),
        warning: warning.clone(),
        span: tracing::Span::current(),
    };
    if let Some(pending) = attempt::hold(pending) {
        fire(&pending);
    }
}

/// Send `pending` to `tracing`, at the level its code decides.
pub(super) fn fire(pending: &Pending) {
    let level = match &pending.warning.code {
        None => Level::Warn,
        Some(code) => {
            if !warn_enabled(pending.warning.kind) {
                // Nobody reads the first-time `WARN`, so the key is not
                // spent on it.
                Level::Debug
            } else {
                let key = Key {
                    provider: pending.provider.clone(),
                    model: pending.model.clone(),
                    kind: pending.warning.kind,
                    code: code.to_string(),
                };
                let seen = DEDUPE
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .see(key);
                if seen.filled {
                    tracing::event!(
                        target: "allama::warning",
                        tracing::Level::WARN,
                        allama.warning.dedupe_capacity = DEDUPE_CAPACITY,
                        "warning dedupe set full; evicting the least recently seen codes, \
                         whose next warning goes out at WARN again",
                    );
                }
                if seen.first {
                    Level::FirstWarn
                } else {
                    Level::Debug
                }
            }
        }
    };
    dispatch(level, pending);
}

/// The level a warning event goes out at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Level {
    /// `WARN`, every time (no code).
    Warn,
    /// `WARN`, the first time for a coded warning's key.
    FirstWarn,
    /// `DEBUG`, a repeat of a coded warning.
    Debug,
}

/// The value of `allama.warning.repeats` on the first `WARN` of a coded
/// warning.
const REPEATS: &str = "further identical warnings at DEBUG";

/// One event macro per target and level, so every callsite is constant.
macro_rules! warning_event {
    ($target:literal, $level:ident, $pending:expr, $repeats:expr) => {
        tracing::event!(
            target: $target,
            parent: &$pending.span,
            tracing::Level::$level,
            allama.warning.provider = $pending.provider.as_str(),
            allama.warning.model = $pending.model.as_str(),
            allama.warning.kind = $pending.warning.kind.as_str(),
            allama.warning.subject = %SubjectLabel(&$pending.warning.subject),
            allama.warning.code = $pending.warning.code.as_deref(),
            allama.warning.cause_kind = $pending.warning.cause.as_ref().map(|c| c.kind.as_ref()),
            allama.warning.cause_stage = $pending
                .warning
                .cause
                .as_ref()
                .and_then(|c| c.stage)
                .map(crate::UnsupportedStage::as_str),
            allama.warning.repeats = $repeats,
            "{}",
            $pending.warning.message,
        )
    };
}

/// Expand `$body!` with the target of `$kind`.
macro_rules! by_target {
    ($kind:expr, $body:ident $(, $arg:expr)*) => {
        match $kind {
            WarningKind::Unsupported => $body!("allama::warning::unsupported" $(, $arg)*),
            WarningKind::Approximated => $body!("allama::warning::approximated" $(, $arg)*),
            WarningKind::Dropped => $body!("allama::warning::dropped" $(, $arg)*),
            WarningKind::Deprecated => $body!("allama::warning::deprecated" $(, $arg)*),
            WarningKind::Fallback => $body!("allama::warning::fallback" $(, $arg)*),
            WarningKind::Other => $body!("allama::warning::other" $(, $arg)*),
        }
    };
}

fn dispatch(level: Level, pending: &Pending) {
    macro_rules! at_level {
        ($target:literal, $pending:expr) => {
            match level {
                Level::Warn => warning_event!($target, WARN, $pending, None::<&str>),
                Level::FirstWarn => warning_event!($target, WARN, $pending, Some(REPEATS)),
                Level::Debug => warning_event!($target, DEBUG, $pending, None::<&str>),
            }
        };
    }
    by_target!(pending.warning.kind, at_level, pending);
}

/// Whether a subscriber takes warning events of `kind` at any level.
fn enabled(kind: WarningKind) -> bool {
    macro_rules! either {
        ($target:literal) => {
            tracing::enabled!(target: $target, tracing::Level::WARN)
                || tracing::enabled!(target: $target, tracing::Level::DEBUG)
        };
    }
    by_target!(kind, either)
}

/// Whether a subscriber takes warning events of `kind` at `WARN`.
fn warn_enabled(kind: WarningKind) -> bool {
    macro_rules! warn {
        ($target:literal) => {
            tracing::enabled!(target: $target, tracing::Level::WARN)
        };
    }
    by_target!(kind, warn)
}

/// The codes already logged at `WARN`, for the whole process.
static DEDUPE: LazyLock<Mutex<Dedupe>> = LazyLock::new(|| Mutex::new(Dedupe::new(DEDUPE_CAPACITY)));

/// What decides whether a coded warning recurs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Key {
    provider: String,
    model: String,
    kind: WarningKind,
    code: String,
}

/// A set of keys bounded by evicting the least recently seen: a `HashMap`
/// from key to its last-seen stamp, and the keys ordered by stamp.
#[derive(Debug)]
pub(super) struct Dedupe {
    capacity: usize,
    stamps: HashMap<Key, u64>,
    order: BTreeMap<u64, Key>,
    next: u64,
    /// Whether the set has filled before.
    filled: bool,
}

/// What [`Dedupe::see`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Seen {
    /// The key was not in the set.
    first: bool,
    /// The set filled for the first time and evicted a key to take this one.
    filled: bool,
}

impl Dedupe {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            stamps: HashMap::new(),
            order: BTreeMap::new(),
            next: 0,
            filled: false,
        }
    }

    /// Record that `key` was seen now.
    fn see(&mut self, key: Key) -> Seen {
        let stamp = self.next;
        self.next += 1;
        if let Some(previous) = self.stamps.get_mut(&key) {
            let old = std::mem::replace(previous, stamp);
            if let Some(key) = self.order.remove(&old) {
                self.order.insert(stamp, key);
            }
            return Seen {
                first: false,
                filled: false,
            };
        }
        let mut filled = false;
        if self.stamps.len() >= self.capacity
            && let Some((_, oldest)) = self.order.pop_first()
        {
            self.stamps.remove(&oldest);
            filled = !self.filled;
            self.filled = true;
        }
        self.stamps.insert(key.clone(), stamp);
        self.order.insert(stamp, key);
        Seen {
            first: true,
            filled,
        }
    }
}

/// A compact form of a subject for a log field: `field:seed`,
/// `content:1:0`, `response`.
struct SubjectLabel<'a>(&'a Subject);

impl fmt::Display for SubjectLabel<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Subject::Field { field } => write!(f, "field:{field}"),
            Subject::Content {
                message,
                block: Some(block),
            } => write!(f, "content:{message}:{block}"),
            Subject::Content {
                message,
                block: None,
            } => write!(f, "content:{message}"),
            Subject::EmbeddingInput {
                input,
                part: Some(part),
            } => write!(f, "embedding_input:{input}:{part}"),
            Subject::EmbeddingInput { input, part: None } => {
                write!(f, "embedding_input:{input}")
            }
            Subject::Messages { messages } => {
                f.write_str("messages:")?;
                crate::warning::write_indices(f, messages, ",")
            }
            Subject::System { index } => write!(f, "system:{index}"),
            Subject::Option { namespace } => write!(f, "option:{namespace}"),
            Subject::Response => f.write_str("response"),
            Subject::Other { name } => write!(f, "other:{name}"),
            Subject::Unknown => f.write_str("unknown"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: &str) -> Key {
        Key {
            provider: "p".into(),
            model: "m".into(),
            kind: WarningKind::Unsupported,
            code: code.into(),
        }
    }

    #[test]
    fn keys_are_seen_once() {
        let mut dedupe = Dedupe::new(8);
        assert!(dedupe.see(key("a")).first);
        assert!(!dedupe.see(key("a")).first);
        assert!(dedupe.see(key("b")).first);
        let other_model = Key {
            model: "n".into(),
            ..key("a")
        };
        assert!(dedupe.see(other_model).first);
    }

    /// The least recently seen key goes first, and the set says once that
    /// it filled.
    #[test]
    fn the_least_recently_seen_key_is_evicted_and_the_fill_is_reported_once() {
        let mut dedupe = Dedupe::new(2);
        assert_eq!(
            dedupe.see(key("a")),
            Seen {
                first: true,
                filled: false
            }
        );
        dedupe.see(key("b"));
        // `a` is seen again, so `b` is now the oldest.
        assert!(!dedupe.see(key("a")).first);
        assert_eq!(
            dedupe.see(key("c")),
            Seen {
                first: true,
                filled: true
            }
        );
        assert!(!dedupe.see(key("a")).first, "a was kept");
        // `b` was evicted: it is new again, and the fill is not reported twice.
        assert_eq!(
            dedupe.see(key("b")),
            Seen {
                first: true,
                filled: false
            }
        );
        assert_eq!(dedupe.stamps.len(), 2);
        assert_eq!(dedupe.order.len(), 2);
    }

    #[test]
    fn the_default_capacity_holds_4096_keys() {
        let mut dedupe = Dedupe::new(DEDUPE_CAPACITY);
        for i in 0..DEDUPE_CAPACITY {
            assert!(!dedupe.see(key(&i.to_string())).filled);
        }
        let seen = dedupe.see(key("one more"));
        assert!(seen.first && seen.filled);
        assert!(dedupe.see(key("0")).first, "the oldest was evicted");
        assert!(!dedupe.see(key("4095")).first);
    }
}
