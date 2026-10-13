//! The warning codes every published crate writes, and the table in the
//! `allama::warning` docs that lists them.
//!
//! A warning's code is stable API: operators filter and alert on it. Core
//! derives `<kind>.<field>` codes, pinned by `allama`'s own `warning_codes`
//! test; adapters also write codes of their own (`dropped.foreign_reasoning`,
//! `approximated.tool_call_arguments`, …) as string literals. This test
//! finds those literals in each published crate's non-test sources and
//! compares them with `tests/fixtures/warning_codes.json`. It lives in this
//! unpublished crate because it reads every crate.
//!
//! The fixture changes only in the change that adds, renames or removes a
//! code; add the code to the table. Run with
//! `ALLAMA_UPDATE_FIXTURES=1` to rewrite the fixture.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::Value;

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

/// `allama`'s golden: the kinds, and the codes core derives.
fn core_golden() -> Value {
    let path = workspace().join("crates/allama/tests/fixtures/warning_codes.json");
    serde_json::from_str(&read(&path)).unwrap()
}

fn strings(value: &Value) -> Vec<String> {
    value
        .as_array()
        .expect("a list")
        .iter()
        .map(|item| item.as_str().expect("a string").to_owned())
        .collect()
}

/// Every `.rs` file under `dir`, except test files.
fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if path.is_dir() {
            if name != "tests" {
                sources(&path, out);
            }
        } else if name.ends_with(".rs") && name != "tests.rs" && !name.ends_with("_tests.rs") {
            out.push(path);
        }
    }
}

/// The lines of `text` outside comments and top-level test items
/// (`#[cfg(test)]`, `#[cfg(all(test, ..))]`).
fn code_lines(text: &str) -> Vec<&str> {
    let mut lines = text.lines();
    let mut out = Vec::new();
    while let Some(line) = lines.next() {
        if line.starts_with("#[cfg(") && line.contains("test") {
            // Skip the item: one line, or a block up to its closing brace.
            if let Some(item) = lines.next()
                && item.trim_end().ends_with('{')
            {
                for inner in lines.by_ref() {
                    if inner == "}" {
                        break;
                    }
                }
            }
            continue;
        }
        if !line.trim_start().starts_with("//") {
            out.push(line);
        }
    }
    out
}

/// The string literals in `line` that read `"<kind>.<name>"`.
fn literal_codes(line: &str, kinds: &[String], found: &mut BTreeSet<String>) {
    for (start, _) in line.match_indices('"') {
        let rest = &line[start + 1..];
        let Some(end) = rest.find('"') else { continue };
        let literal = &rest[..end];
        let Some((kind, name)) = literal.split_once('.') else {
            continue;
        };
        if kinds.iter().any(|k| k == kind)
            && !name.is_empty()
            && name.bytes().all(|b| b.is_ascii_lowercase() || b == b'_')
        {
            found.insert(literal.to_owned());
        }
    }
}

/// Each published crate's literal codes, without the ones core derives.
fn current() -> BTreeMap<String, Vec<String>> {
    let golden = core_golden();
    let kinds = strings(&golden["kinds"]);
    let derived: BTreeSet<String> = strings(&golden["derived"]).into_iter().collect();
    let mut crates = BTreeMap::new();
    for entry in std::fs::read_dir(workspace().join("crates")).unwrap() {
        let dir = entry.unwrap().path();
        if read(&dir.join("Cargo.toml")).contains("publish = false") {
            continue;
        }
        let mut files = Vec::new();
        sources(&dir.join("src"), &mut files);
        let mut found = BTreeSet::new();
        for file in files {
            for line in code_lines(&read(&file)) {
                literal_codes(line, &kinds, &mut found);
            }
        }
        let codes: Vec<String> = found.difference(&derived).cloned().collect();
        if !codes.is_empty() {
            let name = dir.file_name().unwrap().to_string_lossy().into_owned();
            crates.insert(name, codes);
        }
    }
    crates
}

#[test]
fn every_crate_writes_the_codes_the_fixture_lists() {
    let path = workspace().join("crates/allama-examples/tests/fixtures/warning_codes.json");
    let current = serde_json::to_value(current()).unwrap();
    let fixture: Value = serde_json::from_str(&read(&path)).unwrap();
    if current == fixture {
        return;
    }
    if allama_conformance::update_fixtures() {
        let text = serde_json::to_string_pretty(&current).unwrap();
        std::fs::write(&path, text + "\n").expect("the fixture is writable");
        return;
    }
    assert_eq!(
        current, fixture,
        "a crate writes a warning code the fixture does not list, or no longer writes one; \
         if that is deliberate, rerun with ALLAMA_UPDATE_FIXTURES=1, update the table in \
         the `allama::warning` docs"
    );
}

/// The `allama::warning` docs list every code: each literal code, each
/// code core names as a constant, and the `<kind>.<field>` rule.
#[test]
fn the_warning_docs_list_every_code() {
    let docs = read(&workspace().join("crates/allama/src/warning.rs"));
    let start = docs
        .find("//! # Warning codes")
        .expect("the warning docs have a codes section");
    let section = &docs[start..];
    let end = section[3..]
        .find("//! # ")
        .map_or(section.len(), |at| at + 3);
    let table = &section[..end];

    let golden = core_golden();
    let constants = golden["constants"].as_object().unwrap().values();
    let mut codes: BTreeSet<String> = constants
        .map(|code| code.as_str().unwrap().to_owned())
        .collect();
    codes.extend(current().into_values().flatten());
    let missing: Vec<&String> = codes
        .iter()
        .filter(|code| !table.contains(&format!("`{code}`")))
        .collect();
    assert!(missing.is_empty(), "missing from the table: {missing:?}");
    assert!(table.contains("`<kind>.<field>`"));
}
