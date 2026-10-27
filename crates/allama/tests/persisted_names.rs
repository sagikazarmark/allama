//! Golden fixture for persisted names outside the `History` format: the
//! capability names a configuration or a catalog stores, the extraction
//! failure an `Error` serializes, the errors a failed failover keeps
//! (`fallback_from`), the usage counters, and the embedding limits.
//!
//! `tests/fixtures/persisted_names.json` changes only in the change that
//! alters one of these forms deliberately.

use allama::{
    CapabilitySupport, ChatCapability, DecisionCapability, EmbeddingCapability, EmbeddingLimits,
    Error, Usage,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

const FIXTURE: &str = include_str!("fixtures/persisted_names.json");

fn fixture() -> Value {
    serde_json::from_str(FIXTURE).unwrap()
}

/// `value` loads as `T` and writes back unchanged.
fn round_trips<T: Serialize + DeserializeOwned>(value: &Value) -> T {
    let typed: T = serde_json::from_value(value.clone())
        .unwrap_or_else(|err| panic!("{value} does not load: {err}"));
    assert_eq!(&serde_json::to_value(&typed).unwrap(), value);
    typed
}

#[test]
fn capability_names_are_snake_case() {
    let fixture = fixture();
    let chat: Vec<ChatCapability> = round_trips(&fixture["chat_capabilities"]);
    assert_eq!(chat[0], ChatCapability::ToolCalls);
    let embedding: Vec<EmbeddingCapability> = round_trips(&fixture["embedding_capabilities"]);
    assert_eq!(embedding, EmbeddingCapability::ALL);
    let decision: Vec<DecisionCapability> = round_trips(&fixture["decision_capabilities"]);
    assert_eq!(decision, DecisionCapability::ALL);
    let support: Vec<CapabilitySupport> = round_trips(&fixture["capability_support"]);
    assert_eq!(
        support,
        [
            CapabilitySupport::Supported,
            CapabilitySupport::Unsupported,
            CapabilitySupport::Unknown
        ]
    );
    for name in ["batch_input", "yes_no", "unknown"] {
        assert!(FIXTURE.contains(name));
    }
    assert_eq!(EmbeddingCapability::BatchInput.to_string(), "batch_input");
    assert_eq!(
        "yes_no".parse::<DecisionCapability>().unwrap(),
        DecisionCapability::YesNo
    );
}

#[test]
fn errors_use_one_tagging_convention() {
    let fixture = fixture();
    let errors: Vec<Error> = round_trips(&fixture["errors"]);
    assert!(matches!(errors[0], Error::Rejected { .. }));
    // A failed failover: the backup's error, with the primary's behind it.
    assert!(matches!(errors[1], Error::RateLimited { .. }));
    assert_eq!(errors[1].status(), Some(429));
    let primary = errors[1].fallback_from().unwrap();
    assert!(matches!(primary, Error::Overloaded { .. }));
    assert_eq!(primary.status(), Some(529));
    assert!(primary.fallback_from().is_none());
}

#[cfg(feature = "chat-extract")]
#[test]
fn extraction_errors_are_internally_tagged() {
    let fixture = fixture();
    let errors: Vec<Error> = round_trips(&fixture["extraction_errors"]);
    {
        use allama::chat::extract::ChatExtractError;

        let kinds: Vec<&str> = errors
            .iter()
            .map(|error| match error {
                Error::Extract { failure, .. } => match failure.error() {
                    ChatExtractError::Unsupported { .. } => "unsupported",
                    ChatExtractError::ToolConflict { .. } => "tool_conflict",
                    ChatExtractError::MissingStructuredText { .. } => "missing_structured_text",
                    ChatExtractError::MissingToolCall { .. } => "missing_tool_call",
                    ChatExtractError::UnexpectedToolCallCount { .. } => {
                        "unexpected_tool_call_count"
                    }
                    ChatExtractError::Truncated { .. } => "truncated",
                    ChatExtractError::Refused { .. } => "refused",
                    ChatExtractError::EnforcementBelow { .. } => "enforcement_below",
                    ChatExtractError::Parse { .. } => "parse",
                    _ => "unknown",
                },
                _ => "not an extraction error",
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "unsupported",
                "tool_conflict",
                "missing_structured_text",
                "missing_tool_call",
                "unexpected_tool_call_count",
                "truncated",
                "refused",
                "enforcement_below",
                "parse"
            ]
        );
        // A pass count above `u8::MAX`.
        let Error::Extract { failure, .. } = &errors[8] else {
            unreachable!()
        };
        assert_eq!(failure.metadata().passes, 300);
    }
}

#[test]
fn usage_and_limits_keep_their_field_names() {
    let fixture = fixture();
    let usage: Usage = round_trips(&fixture["usage"]);
    assert_eq!(usage.tool_requests["web_search"], 2);
    assert_eq!(usage.contract_violation(), None);
    let limits: EmbeddingLimits = round_trips(&fixture["embedding_limits"]);
    assert_eq!(limits.max_inputs_per_call, Some(2048));
}
