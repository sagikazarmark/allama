//! Span attribute names are public API: dashboards and alerts depend on
//! them. These tests pin every name allama emits against the checked-in
//! semantic conventions tables, and every `gen_ai.*` value against its
//! convention type, as exported by `tracing-opentelemetry`.

use std::collections::BTreeSet;

use allama::decision::{Answer, Question, YesNoAnswer};
use allama::mock::{
    ChatResponseBuilder, MockDecisionProvider, MockEmbeddingProvider, MockProvider,
    MockStreamingProvider,
};
use allama::wrappers::{
    FallbackProvider, RetryPolicy, RetryingProvider, TracingContentConfig, TracingProvider,
};
use allama::{
    ChatProvider, ChatRequest, ChatResponse, ChatStreamExt, DecisionProvider, DecisionRequest,
    DecisionResponse, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, Error, FinishReason,
    RequestField, ResponseFormat, Usage, Warning,
};
use opentelemetry::Value;
use opentelemetry::trace::{SpanKind, Status};

use super::otel::{Exported, LAYER_EVENT_ATTRIBUTES, export};
use super::semconv::{self, AttributeType};

/// The `otel.*` fields `tracing-opentelemetry` 0.32 and later read to set the
/// span name, kind, and status. The layer consumes them; they are never
/// exported as attributes.
const LAYER_FIELDS: &[&str] = &[
    "otel.name",
    "otel.kind",
    "otel.status_code",
    "otel.status_description",
];

/// The `tracing` span names `TracingProvider` creates.
const SPAN_NAMES: &[&str] = &["chat", "embeddings", "decision"];

#[tokio::test]
async fn emitted_names_are_semconv_span_attributes_or_allama() {
    let exported = exercise_every_operation().await;

    let mut violations = Vec::new();
    for span_name in SPAN_NAMES {
        for name in emitted_names(&exported, span_name) {
            if let Err(violation) = check_name(span_name, &name) {
                violations.push(violation);
            }
        }
    }
    assert!(
        violations.is_empty(),
        "attribute names outside the pinned conventions (v{}):\n{}",
        semconv::VERSION,
        violations.join("\n")
    );
}

#[tokio::test]
async fn exported_genai_values_have_semconv_types() {
    let exported = exercise_every_operation().await;

    let mut checked = 0;
    for span in exported.spans() {
        let Some(Value::String(operation)) = span.get("gen_ai.operation.name") else {
            continue;
        };
        let table = semconv::table_for(operation.as_str());
        for kv in span.attributes() {
            let Some(expected) = semconv::attribute_type(table, kv.key.as_str()) else {
                continue;
            };
            assert!(
                has_type(&kv.value, expected),
                "`{}` on `{}` exported as {:?}, the conventions say {expected:?}",
                kv.key,
                span.name(),
                kv.value,
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "no attributes were checked");
}

/// The fully populated calls must export every declared `gen_ai.*` field,
/// so the type check above covers all of them.
#[tokio::test]
async fn type_check_covers_every_declared_genai_field() {
    let exported = exercise_every_operation().await;

    for span_name in SPAN_NAMES {
        let missing: Vec<_> = exported
            .declared_fields(span_name)
            .iter()
            .filter(|name| {
                name.starts_with("gen_ai.")
                    && !exported.spans_for(span_name).any(|span| span.has(name))
            })
            .collect();
        assert!(
            missing.is_empty(),
            "`{span_name}` span never exported {missing:?}; populate them in \
             `exercise_every_operation`"
        );
    }
}

/// Every span is a `CLIENT` span, and a successful call leaves its status
/// unset.
#[tokio::test]
async fn successful_calls_export_client_spans_with_unset_status() {
    let exported = exercise_every_operation().await;

    for span_name in SPAN_NAMES {
        let mut spans = exported.spans_for(span_name).peekable();
        assert!(spans.peek().is_some(), "no `{span_name}` span");
        for span in spans {
            assert_eq!(span.kind(), &SpanKind::Client, "{}", span.name());
            assert_eq!(span.status(), &Status::Unset, "{}", span.name());
        }
    }
}

#[test]
fn name_check_follows_the_rule() {
    // Defined in the pinned version, so allowed before allama emits them.
    assert!(check_name("chat", "gen_ai.response.time_to_first_chunk").is_ok());
    assert!(check_name("chat", "gen_ai.request.stream").is_ok());
    assert!(check_name("chat", "error.type").is_ok());
    assert!(check_name("chat", "allama.response.provider").is_ok());
    assert!(check_name("chat", "otel.status_description").is_ok());
    // Not semconv span attributes (or not on this span).
    assert!(check_name("chat", "gen_ai.response.ttft").is_err());
    assert!(check_name("embeddings", "gen_ai.request.temperature").is_err());
    assert!(check_name("chat", "otel.status_message").is_err());
    assert!(check_name("chat", "response.provider").is_err());
}

/// Pins the `otel.*` field contract of the exporting layer: kind and status
/// come from these fields and are not exported as attributes. With
/// `tracing-opentelemetry` 0.31 or earlier, `otel.status_description` would
/// export as a plain attribute and this test fails.
#[tokio::test]
async fn layer_reads_otel_kind_and_status_fields() {
    let ((), exported) = export(async {
        let _span = tracing::info_span!(
            "probe",
            "otel.name" = "probe span",
            "otel.kind" = "client",
            "otel.status_code" = "ERROR",
            "otel.status_description" = "boom",
            "gen_ai.operation.name" = "probe",
        );
    })
    .await;
    let span = exported.span("probe");
    assert_eq!(span.name(), "probe span");
    assert_eq!(span.kind(), &SpanKind::Client);
    assert_eq!(span.status(), &Status::error("boom"));
    for field in LAYER_FIELDS {
        assert!(!span.has(field), "`{field}` exported as an attribute");
    }
}

/// The harness fails on a key exported twice, which is what a field
/// recorded twice exports.
#[tokio::test]
#[should_panic(expected = "more than once")]
async fn harness_fails_on_a_key_recorded_twice() {
    let ((), _) = export(async {
        let span = tracing::info_span!(
            "probe",
            "gen_ai.operation.name" = "probe",
            "gen_ai.provider.name" = "openai",
        );
        span.record("gen_ai.provider.name", "anthropic");
    })
    .await;
}

/// Every name allama can emit on a span: its declared fields, plus what was
/// exported on it and its events (warning events carry `allama.warning.*`),
/// minus what the layer adds to events.
fn emitted_names(exported: &Exported, span_name: &str) -> BTreeSet<String> {
    let mut names: BTreeSet<String> = exported
        .declared_fields(span_name)
        .iter()
        .map(|name| (*name).to_owned())
        .collect();
    for span in exported.spans_for(span_name) {
        names.extend(span.attributes().iter().map(|kv| kv.key.to_string()));
        for event in span.events() {
            names.extend(
                event
                    .attributes
                    .iter()
                    .map(|kv| kv.key.as_str())
                    .filter(|key| !LAYER_EVENT_ATTRIBUTES.contains(key))
                    .map(str::to_owned),
            );
        }
    }
    names
}

/// The naming rule: a name is a span attribute of the pinned conventions
/// table for this span, an `allama.*` name, or a field the exporting layer
/// consumes. A `gen_ai.*` name the table does not define belongs under
/// `allama.*`.
fn check_name(span_name: &str, name: &str) -> Result<(), String> {
    let table = semconv::table_for(span_name);
    if semconv::attribute_type(table, name).is_some()
        || name.starts_with("allama.")
        || LAYER_FIELDS.contains(&name)
    {
        return Ok(());
    }
    if name.starts_with("gen_ai.") {
        Err(format!(
            "`{name}` on `{span_name}` spans is not a span attribute in semconv v{}; \
             record it as `allama.*`",
            semconv::VERSION
        ))
    } else {
        Err(format!(
            "`{name}` on `{span_name}` spans is neither a semconv span attribute nor `allama.*`"
        ))
    }
}

/// Whether an exported value has the convention type. `tracing` fields
/// cannot carry arrays, so `string[]` attributes are exported as a JSON
/// array of strings; `any` attributes are JSON strings, as the conventions
/// allow.
fn has_type(value: &Value, expected: AttributeType) -> bool {
    match (expected, value) {
        (AttributeType::String, Value::String(_))
        | (AttributeType::Int, Value::I64(_))
        | (AttributeType::Double, Value::F64(_))
        | (AttributeType::Boolean, Value::Bool(_)) => true,
        (AttributeType::StringArray, Value::String(json)) => {
            serde_json::from_str::<Vec<String>>(json.as_str()).is_ok()
        }
        (AttributeType::Any, Value::String(json)) => {
            serde_json::from_str::<serde_json::Value>(json.as_str()).is_ok()
        }
        _ => false,
    }
}

/// Run one call of every operation through `TracingProvider`, with every
/// request setting and response field that allama records populated, and
/// chat and embedding calls that retry and fail over, so attempt events are
/// checked.
async fn exercise_every_operation() -> Exported {
    let conversation = |span: &tracing::Span, _: allama::wrappers::TracedRequest<'_>| {
        span.record("gen_ai.conversation.id", "conv-1");
    };
    let chat = TracingProvider::new(MockProvider::with_response(full_chat_response()))
        .with_content_capture(TracingContentConfig::all())
        .on_span(conversation);
    let resilient = TracingProvider::new(FallbackProvider::new(
        RetryingProvider::new(
            MockProvider::new([Error::overloaded("busy"), Error::overloaded("busy")]),
            RetryPolicy::new()
                .max_attempts(2)
                .base_delay(std::time::Duration::ZERO),
        ),
        MockProvider::with_text("backup").with_provider_name("backup"),
    ));
    let stream = TracingProvider::new(MockStreamingProvider::from_responses(
        [full_chat_response()],
    ));
    let embeddings = TracingProvider::new(MockEmbeddingProvider::new([Ok(
        EmbeddingResponse::new(vec![vec![0.1, 0.2]])
            .model("e-1")
            .usage(Usage::new().input_tokens(2))
            .warning(Warning::unsupported(RequestField::Seed, "demo")),
    )]));
    let resilient_embeddings = TracingProvider::new(FallbackProvider::new(
        RetryingProvider::new(
            MockEmbeddingProvider::new([
                Err(Error::overloaded("busy")),
                Err(Error::overloaded("busy")),
            ]),
            RetryPolicy::new()
                .max_attempts(2)
                .base_delay(std::time::Duration::ZERO),
        ),
        MockEmbeddingProvider::with_vectors(vec![vec![0.1]]).with_provider_name("backup"),
    ));
    let decision =
        TracingProvider::new(MockDecisionProvider::new([Ok(DecisionResponse::new([(
            "spam",
            Answer::from(YesNoAnswer::new(0.1)),
        )])
        .model("d-1")
        .id("resp-d")
        .request_id("req-d")
        .provider("mock")
        .usage(Usage::new().input_tokens(12).output_tokens(1))
        .warning(Warning::unsupported(RequestField::Seed, "demo")))]))
        .on_span(conversation);

    let request = ChatRequest::new("m")
        .system("Be brief.")
        .user("hi")
        .max_output_tokens(32)
        .temperature(0.5)
        .top_p(0.9)
        .top_k(40)
        .frequency_penalty(0.1)
        .presence_penalty(0.2)
        .stop(["END"])
        .seed(7)
        .response_format(ResponseFormat::Json);

    let ((), exported) = export(async {
        chat.chat(&request).await.unwrap();
        resilient.chat(&request).await.unwrap();
        stream
            .chat_stream(&request)
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        embeddings
            .embed(&EmbeddingRequest::new("e").input("x").dimensions(2))
            .await
            .unwrap();
        resilient_embeddings
            .embed(&EmbeddingRequest::new("e").input("x"))
            .await
            .unwrap();
        decision
            .decide(
                &DecisionRequest::new("d", "Win a cruise!")
                    .question("spam", Question::yes_no("Is this spam?")),
            )
            .await
            .unwrap();
    })
    .await;
    exported
}

fn full_chat_response() -> ChatResponse {
    ChatResponseBuilder::new()
        .text("hello")
        .finish_reason(FinishReason::Stop)
        .usage_value(
            Usage::new()
                .input_tokens(10)
                .output_tokens(5)
                .cache_read_input_tokens(4)
                .cache_write_input_tokens(3)
                .reasoning_tokens(2),
        )
        .model("m-1")
        .id("resp-1")
        .warning(Warning::unsupported(RequestField::Seed, "seed ignored"))
        .build()
}
