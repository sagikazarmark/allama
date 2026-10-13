//! `ALLAMA_UPDATE_FIXTURES` creates a missing error fixture, which then
//! passes the comparison. One test only: it sets a process environment
//! variable.

use allama::Error;
use allama_conformance::{
    ErrorFixtureView, FixtureDir, assert_embedding_error_fixture_eq, assert_error_fixture_eq,
    load_json_fixture,
};

#[test]
fn update_mode_writes_error_fixtures_that_then_compare_equal() {
    let dir = tempfile::TempDir::new().unwrap();
    let fixtures = FixtureDir::new(dir.path().join("fixtures"));
    let err = Error::rate_limited("HTTP 429")
        .authored()
        .with_status(429)
        .with_upstream_message("slow down");

    // SAFETY: this binary runs a single test, so no other thread reads the
    // environment concurrently.
    unsafe { std::env::set_var("ALLAMA_UPDATE_FIXTURES", "1") };
    assert_error_fixture_eq(&err, &fixtures, "error_rate_limited.json");
    assert_embedding_error_fixture_eq(&err, &fixtures, "embed_error_rate_limited.json");
    // SAFETY: as above.
    unsafe { std::env::remove_var("ALLAMA_UPDATE_FIXTURES") };

    let expected = serde_json::to_value(ErrorFixtureView::new(&err)).unwrap();
    assert_eq!(
        load_json_fixture(&fixtures, "error_rate_limited.json"),
        expected
    );
    assert_eq!(
        load_json_fixture(&fixtures, "embed_error_rate_limited.json"),
        expected
    );
    assert_error_fixture_eq(&err, &fixtures, "error_rate_limited.json");
    assert_embedding_error_fixture_eq(&err, &fixtures, "embed_error_rate_limited.json");

    let changed = err.with_upstream_message("slower");
    let mismatch = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_error_fixture_eq(&changed, &fixtures, "error_rate_limited.json");
    }))
    .expect_err("a changed upstream message must not match the fixture");
    let message = mismatch
        .downcast_ref::<String>()
        .expect("assert_eq panics with a formatted message");
    assert!(
        message.contains("error fixture mismatch: error_rate_limited.json"),
        "{message}"
    );
    assert!(
        message.contains("slower"),
        "the panic shows the diff: {message}"
    );
}
