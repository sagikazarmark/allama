//! Two tables generated from `allama_conformance::probe::ACCOUNTED_FIELDS`:
//! the provider author guide's "Fields you must account for" checklist (the
//! probed fields), and the exempt fields in the rustdoc of
//! `ACCOUNTED_FIELDS`, which is for maintainers.
//!
//! This test fails when a table and the list disagree. Run it with
//! `ALLAMA_UPDATE_FIXTURES=1` to regenerate both. It lives in this
//! unpublished crate because the guide is outside every published package.

use std::path::{Path, PathBuf};

use allama_conformance::probe::{Account, Probe, accounted_fields};

#[test]
fn provider_guide_checklist_matches_the_accounted_fields() {
    sync(
        &repo().join("docs/guides/provider-authors.md"),
        "<!-- accounted-fields:start -->",
        "<!-- accounted-fields:end -->",
        &checklist(),
    );
}

#[test]
fn the_exempt_fields_rustdoc_matches_the_accounted_fields() {
    sync(
        &repo().join("crates/allama-conformance/src/probe/fields.rs"),
        "/// <!-- exempt-fields:start -->",
        "/// <!-- exempt-fields:end -->",
        &exempt_fields(),
    );
}

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Check that the text between `start` and `end` in `path` is `expected`,
/// or rewrite it under `ALLAMA_UPDATE_FIXTURES`.
fn sync(path: &Path, start: &str, end: &str, expected: &str) {
    let text = std::fs::read_to_string(path).expect("the file exists");
    let from = text.find(start).expect("the file has the start marker") + start.len();
    let to = text.find(end).expect("the file has the end marker");
    if text[from..to] == *expected {
        return;
    }
    if allama_conformance::update_fixtures() {
        let updated = format!("{}{expected}{}", &text[..from], &text[to..]);
        std::fs::write(path, updated).expect("the file is writable");
        return;
    }
    panic!(
        "the table in {} is out of date with ACCOUNTED_FIELDS; rerun with \
         ALLAMA_UPDATE_FIXTURES=1 to regenerate it",
        path.display()
    );
}

/// The guide's checklist: a legend of what each probe accepts, then the
/// probed fields grouped by probe.
fn checklist() -> String {
    let fields = accounted_fields();
    let mut legend: Vec<&str> = Vec::new();
    for probe in Probe::ALL {
        if !legend.contains(&probe.describe()) {
            legend.push(probe.describe());
        }
    }
    let mut out = String::from("\n\nA probed field is accounted for when it is:\n\n");
    for (key, description) in legend.iter().enumerate() {
        out.push_str(&format!("{}. {description}\n", key + 1));
    }
    out.push_str("\n| Probe | Request fields | Accounted for when it is |\n| --- | --- | --- |\n");
    for probe in Probe::ALL {
        let paths: Vec<String> = fields
            .iter()
            .filter(|field| matches!(&field.account, Account::Probed(p) if p.contains(probe)))
            .map(|field| format!("`{}`", field.path))
            .collect();
        let key = legend
            .iter()
            .position(|description| *description == probe.describe())
            .expect("every description is in the legend")
            + 1;
        out.push_str(&format!("| `{probe}` | {} | {key} |\n", paths.join("<br>")));
    }
    out.push('\n');
    out
}

/// The rustdoc table of exempt fields, as `///` lines.
fn exempt_fields() -> String {
    let mut out = String::from(
        "\n///\n/// | Exempt request field | Why C2 does not probe it |\n/// | --- | --- |\n",
    );
    for field in accounted_fields() {
        if let Account::Exempt(reason) = &field.account {
            out.push_str(&format!("/// | `{}` | {reason} |\n", field.path));
        }
    }
    out.push_str("///\n");
    out
}
