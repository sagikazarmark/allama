//! The process-wide dedupe set of warning events is bounded: when it first
//! fills, one event says so, and it never switches to `WARN` on every call.
//!
//! In a test binary of its own, because filling the set evicts every other
//! test's keys.

#![cfg(feature = "warning-events")]

#[path = "support/warning_capture.rs"]
mod support;

use allama::{Subject, Warning, WarningKind};
use support::{capture, levels};

#[test]
fn a_full_set_says_so_once_and_evicts_the_oldest_codes() {
    let emit = |model: &str| {
        let mut warning =
            Warning::new(WarningKind::Other, Subject::Response, "full").with_code("other.full");
        allama::warning::emit("p", model, &mut warning);
    };
    let ((), events) = capture(async {
        // 4,096 keys fill the set; the next one evicts the oldest.
        for key in 0..4_096 {
            emit(&format!("model-{key}"));
        }
        emit("model-4096");
        // The oldest key is new again, and the set is not announced twice.
        emit("model-0");
        for key in 4_097..5_000 {
            emit(&format!("model-{key}"));
        }
        // A key seen recently is still a repeat.
        emit("model-4999");
    });
    let full: Vec<_> = events
        .under("allama::warning")
        .into_iter()
        .filter(|event| event.target == "allama::warning")
        .collect();
    assert_eq!(full.len(), 1, "{full:?}");
    assert_eq!(full[0].level, "WARN");
    assert!(full[0].message.contains("dedupe set full"), "{full:?}");

    assert_eq!(levels(&events.for_model("model-0")), (2, 0));
    assert_eq!(levels(&events.for_model("model-4999")), (1, 1));
}
