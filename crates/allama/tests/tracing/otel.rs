//! Exporter harness: runs a future under a `tracing-opentelemetry` layer that
//! feeds an in-memory span exporter, so tests assert on the spans an
//! OpenTelemetry backend receives (span kind, status, attribute types), not
//! on raw `tracing` fields.
//!
//! The layer is configured to add nothing of its own (no code location,
//! thread, target, or busy/idle attributes), so every exported attribute was
//! recorded by allama.
//!
//! The harness fails on a span or event that carries the same attribute key
//! twice. `tracing-opentelemetry` appends a field recorded twice as a second
//! attribute, and backends disagree on which one wins (the OpenTelemetry
//! Collector reads the first), so allama records each key once.

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::sync::{Arc, Mutex};

use opentelemetry::trace::{Event, SpanKind, Status, TracerProvider as _};
use opentelemetry::{KeyValue, Value};
use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider, SpanData};
use tracing_subscriber::Registry;
use tracing_subscriber::layer::SubscriberExt;

/// Attributes `tracing-opentelemetry` adds to every span event on its own.
pub const LAYER_EVENT_ATTRIBUTES: &[&str] = &["level", "target"];

/// Spans exported while running a future with [`export`].
#[derive(Debug)]
pub struct Exported {
    spans: Vec<SpanData>,
    declared: BTreeMap<&'static str, BTreeSet<&'static str>>,
}

impl Exported {
    /// All exported spans, in the order they ended.
    pub fn spans(&self) -> impl Iterator<Item = ExportedSpan<'_>> {
        self.spans.iter().map(ExportedSpan)
    }

    /// The exported spans whose `gen_ai.operation.name` is `operation`.
    pub fn spans_for<'a>(
        &'a self,
        operation: &str,
    ) -> impl Iterator<Item = ExportedSpan<'a>> + use<'a> {
        let operation = operation.to_owned();
        self.spans()
            .filter(move |span| {
            matches!(span.get("gen_ai.operation.name"), Some(Value::String(op)) if op.as_str() == operation)
        })
    }

    /// The one exported span whose `gen_ai.operation.name` is `operation`.
    pub fn span(&self, operation: &str) -> ExportedSpan<'_> {
        let mut matching = self.spans_for(operation);
        let span = matching.next().unwrap_or_else(|| {
            panic!(
                "no `{operation}` span in {:?}",
                self.spans.iter().map(|s| &s.name).collect::<Vec<_>>()
            )
        });
        assert!(
            matching.next().is_none(),
            "more than one `{operation}` span"
        );
        span
    }

    /// The field names declared on `tracing` spans with this metadata name
    /// (`chat`, `embeddings`, `decision`), recorded or not. A `record` call
    /// on an undeclared name is silently dropped by `tracing`, so this is
    /// the full set of names a span can export.
    pub fn declared_fields(&self, span_name: &str) -> &BTreeSet<&'static str> {
        self.declared
            .get(span_name)
            .unwrap_or_else(|| panic!("no `{span_name}` span was created"))
    }
}

/// One exported span.
#[derive(Debug, Clone, Copy)]
pub struct ExportedSpan<'a>(&'a SpanData);

impl<'a> ExportedSpan<'a> {
    /// The exported span name (`otel.name`).
    pub fn name(&self) -> &'a str {
        &self.0.name
    }

    pub fn kind(&self) -> &'a SpanKind {
        &self.0.span_kind
    }

    pub fn status(&self) -> &'a Status {
        &self.0.status
    }

    pub fn attributes(&self) -> &'a [KeyValue] {
        &self.0.attributes
    }

    /// The attribute's exported value. [`export`] has already checked that
    /// each key appears once.
    pub fn get(&self, key: &str) -> Option<&'a Value> {
        find(&self.0.attributes, key)
    }

    /// The attribute's exported value; panics when it is missing.
    pub fn value(&self, key: &str) -> &'a Value {
        self.get(key).unwrap_or_else(|| {
            panic!(
                "no `{key}` on span `{}`; exported: {:?}",
                self.name(),
                self.attributes()
            )
        })
    }

    pub fn has(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    /// How long the span lasted, from its start to its end.
    pub fn duration(&self) -> std::time::Duration {
        self.0
            .end_time
            .duration_since(self.0.start_time)
            .unwrap_or_default()
    }

    pub fn events(&self) -> &'a [Event] {
        &self.0.events.events
    }
}

/// The attribute `key` of an event.
pub fn event_value<'a>(event: &'a Event, key: &str) -> Option<&'a Value> {
    find(&event.attributes, key)
}

/// Run `future` with an OpenTelemetry layer installed as the thread's default
/// subscriber, and collect the spans it exported.
pub async fn export<F: Future>(future: F) -> (F::Output, Exported) {
    let exporter = InMemorySpanExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    let declared = DeclaredFields::default();
    let subscriber = Registry::default()
        .with(
            tracing_opentelemetry::layer()
                .with_tracer(provider.tracer("allama-tests"))
                .with_location(false)
                .with_threads(false)
                .with_target(false)
                .with_tracked_inactivity(false),
        )
        .with(declared.clone());
    let dispatch = tracing::Dispatch::new(subscriber);
    let output = {
        let _guard = tracing::dispatcher::set_default(&dispatch);
        future.await
    };
    // Spans still held by `output` (an unconsumed stream) end when the caller
    // drops it, after this point, and are not exported.
    drop(dispatch);
    provider.force_flush().expect("flush spans");
    let spans = exporter.get_finished_spans().expect("exported spans");
    for span in &spans {
        assert_unique_keys(&span.attributes, &format!("span `{}`", span.name));
        for event in &span.events.events {
            assert_unique_keys(
                &event.attributes,
                &format!("event `{}` of span `{}`", event.name, span.name),
            );
        }
    }
    let declared = std::mem::take(&mut *declared.0.lock().unwrap());
    (output, Exported { spans, declared })
}

/// Records the declared field names of every new span, by metadata name.
#[derive(Clone, Default)]
struct DeclaredFields(Arc<Mutex<BTreeMap<&'static str, BTreeSet<&'static str>>>>);

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for DeclaredFields {
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        _id: &tracing::span::Id,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let metadata = attrs.metadata();
        self.0
            .lock()
            .unwrap()
            .entry(metadata.name())
            .or_default()
            .extend(metadata.fields().iter().map(|field| field.name()));
    }
}

/// Fail when `attributes` carry a key more than once.
fn assert_unique_keys(attributes: &[KeyValue], owner: &str) {
    let mut seen = BTreeSet::new();
    let duplicates: BTreeSet<_> = attributes
        .iter()
        .map(|kv| kv.key.as_str())
        .filter(|key| !seen.insert(*key))
        .collect();
    assert!(
        duplicates.is_empty(),
        "{owner} exported {duplicates:?} more than once: {attributes:?}"
    );
}

fn find<'a>(attributes: &'a [KeyValue], key: &str) -> Option<&'a Value> {
    attributes
        .iter()
        .find(|kv| kv.key.as_str() == key)
        .map(|kv| &kv.value)
}
