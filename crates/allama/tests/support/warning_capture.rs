//! A subscriber layer that records the warning events (targets under
//! `allama::warning`) and retry events (`allama::retry`) a test emits.
//!
//! The deduplication state is process-wide, so every test gives its calls a
//! provider or model name of its own and reads only its own events.

#![allow(dead_code)]

use std::cell::RefCell;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Once};

use tracing::field::{Field, Visit};
use tracing::level_filters::LevelFilter;
use tracing::span::{Attributes, Id, Record};
use tracing::subscriber::Interest;
use tracing::{Metadata, Subscriber};

/// One captured event.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Event {
    pub target: String,
    pub level: String,
    pub message: String,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub code: Option<String>,
    pub repeats: bool,
}

/// The events captured so far, shared with the layer.
#[derive(Debug, Clone, Default)]
pub struct Events(Arc<Mutex<Vec<Event>>>);

impl Events {
    /// The warning events of calls to `model`.
    pub fn for_model(&self, model: &str) -> Vec<Event> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .filter(|event| event.model.as_deref() == Some(model))
            .cloned()
            .collect()
    }

    /// Every event under `target` (a prefix).
    pub fn under(&self, target: &str) -> Vec<Event> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .filter(|event| event.target.starts_with(target))
            .cloned()
            .collect()
    }
}

/// Run `future` on a current-thread runtime and return its output with
/// what it emitted.
pub fn capture<T>(future: impl Future<Output = T>) -> (T, Events) {
    capture_with(|_| future)
}

/// [`capture`], with the events readable while the future runs.
///
/// One global subscriber, installed once per test binary, hands each event
/// to the capture of the thread that emitted it. A scoped subscriber per
/// test (`with_default`) would race: `tracing` caches each callsite's
/// interest process-wide, and a callsite first hit on another thread while
/// a test's subscriber is being registered can be cached as uninteresting.
/// See [`install`] for the same race at install time.
pub fn capture_with<T, F>(future: impl FnOnce(Events) -> F) -> (T, Events)
where
    F: Future<Output = T>,
{
    install();
    let events = Events::default();
    let future = future(events.clone());
    SINK.with(|sink| *sink.borrow_mut() = Some(events.clone()));
    let output = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("a runtime")
        .block_on(future);
    SINK.with(|sink| sink.borrow_mut().take());
    (output, events)
}

thread_local! {
    /// The capture of the test running on this thread.
    static SINK: RefCell<Option<Events>> = const { RefCell::new(None) };
}

/// Make [`Global`] the global subscriber, once for the test binary.
///
/// Registering a subscriber raises the global level filter before the
/// subscriber becomes the global default, and a callsite another test hits
/// in between caches "never" against no subscriber, possibly after a
/// rebuild of the cache. So [`Global`] is installed disarmed, with a level
/// hint of `OFF` that keeps every callsite unregistered, and armed only once
/// it is the global default.
fn install() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        tracing::subscriber::set_global_default(Global)
            .expect("no other global subscriber in this test binary");
        ARMED.store(true, Ordering::SeqCst);
        // Raise the level filter, and compute every callsite's interest
        // against the global subscriber.
        tracing::callsite::rebuild_interest_cache();
    });
}

/// Whether [`Global`] is the global default yet.
static ARMED: AtomicBool = AtomicBool::new(false);

/// Interested in every callsite; takes the events of threads with a
/// capture, and no spans.
struct Global;

impl Subscriber for Global {
    fn register_callsite(&self, _: &'static Metadata<'static>) -> Interest {
        Interest::sometimes()
    }

    fn max_level_hint(&self) -> Option<LevelFilter> {
        if ARMED.load(Ordering::SeqCst) {
            None
        } else {
            Some(LevelFilter::OFF)
        }
    }

    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        !metadata.is_span()
            && captured(metadata.target())
            && SINK.with(|sink| sink.borrow().is_some())
    }

    fn new_span(&self, _: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _: &Id, _: &Record<'_>) {}

    fn record_follows_from(&self, _: &Id, _: &Id) {}

    fn event(&self, event: &tracing::Event<'_>) {
        let recorded = record(event);
        SINK.with(|sink| {
            if let Some(events) = sink.borrow().as_ref() {
                events.0.lock().unwrap().push(recorded);
            }
        });
    }

    fn enter(&self, _: &Id) {}

    fn exit(&self, _: &Id) {}
}

fn captured(target: &str) -> bool {
    target.starts_with("allama::warning")
        || target == "allama::retry"
        || target == "allama::fallback"
}

fn record(event: &tracing::Event<'_>) -> Event {
    let metadata = event.metadata();
    let mut recorded = Event {
        target: metadata.target().to_owned(),
        level: metadata.level().to_string(),
        ..Event::default()
    };
    event.record(&mut Fields(&mut recorded));
    recorded
}

/// Count the events at each level: `(warn, debug)`.
pub fn levels(events: &[Event]) -> (usize, usize) {
    let warn = events.iter().filter(|event| event.level == "WARN").count();
    let debug = events.iter().filter(|event| event.level == "DEBUG").count();
    (warn, debug)
}

struct Fields<'a>(&'a mut Event);

impl Visit for Fields<'_> {
    fn record_str(&mut self, field: &Field, value: &str) {
        match field.name() {
            "allama.warning.provider" => self.0.provider = Some(value.to_owned()),
            "allama.warning.model" => self.0.model = Some(value.to_owned()),
            "allama.warning.code" => self.0.code = Some(value.to_owned()),
            "allama.warning.repeats" => self.0.repeats = true,
            _ => {}
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.0.message = format!("{value:?}");
        }
    }
}
