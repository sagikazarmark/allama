//! Golden fixture for warning codes.
//!
//! A warning's [code](allama::Warning::code) is stable API: operators filter
//! and alert on it, and the warning events are deduplicated by it. This
//! pins the names of the warning kinds and request fields, the
//! `<kind>.<field>` codes `Degradations` derives from them, and the codes
//! core names as constants. The codes adapters write themselves are pinned
//! by the `warning_codes` test of `allama-examples`, which sees every crate.
//! The table in the `allama::warning` docs lists them all.
//!
//! `tests/fixtures/warning_codes.json` changes only in the change that adds
//! a kind or a field, or renames one deliberately.
//! Run the test with `ALLAMA_UPDATE_FIXTURES=1` to rewrite it.

use allama::adapter::Degradations;
use allama::adapter::chat::MISSING_FINISH_REASON;
use allama::{RequestField, Subject, UnsupportedPolicy, Warning, WarningKind};
use serde_json::{Value, json};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/warning_codes.json"
);

fn fixture() -> Value {
    let text = std::fs::read_to_string(FIXTURE).expect("the fixture exists");
    serde_json::from_str(&text).expect("the fixture is JSON")
}

/// The kinds the fixture names, each read back as a `WarningKind`.
fn kinds(fixture: &Value) -> Vec<WarningKind> {
    fixture["kinds"]
        .as_array()
        .expect("a list of kinds")
        .iter()
        .map(|name| {
            let kind: WarningKind = serde_json::from_value(name.clone()).unwrap();
            assert_eq!(kind.as_str(), name, "{name} reads back as itself");
            assert_eq!(kind.to_string(), kind.as_str());
            kind
        })
        .collect()
}

/// The code `Degradations` gives a warning of `kind` about `subject` that
/// has none.
fn derived(kind: WarningKind, subject: Subject) -> String {
    let mut degradations = Degradations::new("demo", UnsupportedPolicy::Warn);
    degradations.warn(Warning::new(kind, subject, "x"));
    let warning = &degradations.warnings()[0];
    warning.code.as_deref().expect("a derived code").to_owned()
}

/// The fixture as this build of allama writes it.
fn current(kinds: &[WarningKind]) -> Value {
    let fields: Vec<&str> = RequestField::ALL.iter().map(|f| f.as_str()).collect();
    let mut codes = Vec::new();
    for &kind in kinds {
        for &field in RequestField::ALL {
            let code = derived(kind, Subject::Field { field });
            assert_eq!(code, format!("{kind}.{field}"));
            codes.push(code);
        }
        // A system prompt is coded as the `system` field.
        assert_eq!(
            derived(kind, Subject::System { index: 0 }),
            format!("{kind}.system")
        );
    }
    json!({
        "kinds": kinds.iter().map(|kind| kind.as_str()).collect::<Vec<_>>(),
        "fields": fields,
        "derived": codes,
        "constants": {
            "allama::adapter::chat::MISSING_FINISH_REASON": MISSING_FINISH_REASON,
        },
    })
}

#[test]
fn warning_codes_are_stable() {
    let fixture = fixture();
    let current = current(&kinds(&fixture));
    if current == fixture {
        return;
    }
    if std::env::var("ALLAMA_UPDATE_FIXTURES").is_ok_and(|value| value.trim() == "1") {
        let text = serde_json::to_string_pretty(&current).unwrap();
        std::fs::write(FIXTURE, text + "\n").expect("the fixture is writable");
        return;
    }
    assert_eq!(
        current, fixture,
        "a warning code changed; if that is deliberate, rerun with \
         ALLAMA_UPDATE_FIXTURES=1"
    );
}

/// A content warning gets no derived code: whether it recurs with the
/// request's shape is for its producer to say.
#[test]
fn content_warnings_are_not_coded_by_default() {
    let mut degradations = Degradations::new("demo", UnsupportedPolicy::Warn);
    degradations.warn(Warning::new(
        WarningKind::Dropped,
        Subject::Content {
            message: 0,
            block: Some(0),
        },
        "x",
    ));
    assert_eq!(degradations.warnings()[0].code, None);
}
