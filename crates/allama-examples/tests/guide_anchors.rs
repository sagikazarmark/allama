//! Links into the applications guide name one of its headings.
//!
//! Every link to a section of `docs/guides/applications.md`, from the
//! Markdown files at the repository root, the Markdown and Rust files of
//! every crate and under `docs/`, and inside the guide itself, must match
//! the anchor GitHub gives one of the guide's headings. Renaming or removing
//! a heading then fails here instead of leaving a link that lands at the top
//! of the page.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use allama_examples::workspace::Workspace;

const GUIDE: &str = include_str!("../../../docs/guides/applications.md");

/// What precedes the anchor of a link into the guide from another file.
const GUIDE_LINK: &str = "applications.md#";

/// What precedes the anchor of a link inside the guide.
const LOCAL_LINK: &str = "](#";

#[test]
fn links_into_the_applications_guide_name_its_headings() {
    let headings = anchors(GUIDE);
    let workspace = Workspace::get();
    let root = &workspace.root;
    let mut files = Vec::new();
    for entry in fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|extension| extension == "md") {
            files.push(path);
        }
    }
    collect(&root.join("docs"), &mut files);
    for package in &workspace.packages {
        collect(&package.root, &mut files);
    }
    assert!(
        files.iter().any(|file| file.ends_with("README.md")),
        "the walk found the repository's files"
    );

    let mut broken = Vec::new();
    for file in &files {
        let text = fs::read_to_string(file).unwrap();
        for anchor in links(&text, GUIDE_LINK) {
            if !headings.contains(anchor) {
                let file = file.strip_prefix(root).unwrap_or(file);
                broken.push(format!("{}: {GUIDE_LINK}{anchor}", file.display()));
            }
        }
    }
    for anchor in links(GUIDE, LOCAL_LINK) {
        if !headings.contains(anchor) {
            broken.push(format!("applications.md: (#{anchor})"));
        }
    }
    assert!(
        broken.is_empty(),
        "links into the applications guide name no heading:\n{}",
        broken.join("\n")
    );
}

#[test]
fn anchors_follow_github_slugs() {
    let guide = "# Top\n\n## Starting a turn: `extend_messages` or `into_request`\n\n\
                 ```rust\n# fn main() {}\n```\n\n### Warnings\n\n## Warnings\n";
    let anchors = anchors(guide);
    assert!(anchors.contains("starting-a-turn-extend_messages-or-into_request"));
    assert!(anchors.contains("warnings"));
    assert!(anchors.contains("warnings-1"));
    // A `#` line inside a code block is not a heading.
    assert!(!anchors.contains("fn-main"));
}

/// The anchors GitHub gives the headings of `markdown`: lowercased, with
/// punctuation other than `-` and `_` removed and spaces turned into `-`.
/// A repeated anchor gets `-1`, `-2`, … appended.
fn anchors(markdown: &str) -> HashSet<String> {
    let mut anchors = HashSet::new();
    let mut in_code = false;
    for line in markdown.lines() {
        if line.starts_with("```") {
            in_code = !in_code;
            continue;
        }
        if in_code {
            continue;
        }
        let Some(heading) = line.strip_prefix('#') else {
            continue;
        };
        let Some(text) = heading.trim_start_matches('#').strip_prefix(' ') else {
            continue;
        };
        let slug: String = text
            .trim()
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | ' '))
            .map(|c| if c == ' ' { '-' } else { c })
            .collect();
        let mut anchor = slug.clone();
        let mut repeat = 0;
        while anchors.contains(&anchor) {
            repeat += 1;
            anchor = format!("{slug}-{repeat}");
        }
        anchors.insert(anchor);
    }
    anchors
}

/// The anchors of the links in `text` that follow `prefix`.
fn links<'a>(text: &'a str, prefix: &str) -> impl Iterator<Item = &'a str> {
    text.match_indices(prefix).filter_map(move |(at, _)| {
        let rest = &text[at + prefix.len()..];
        let end = rest
            .find(|c: char| !(c.is_alphanumeric() || matches!(c, '-' | '_')))
            .unwrap_or(rest.len());
        (end > 0).then(|| &rest[..end])
    })
}

/// The Markdown and Rust files under `dir`, skipping build output and
/// hidden directories.
fn collect(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy();
        if path.is_dir() {
            if !name.starts_with('.') && name != "target" {
                collect(&path, files);
            }
        } else if name.ends_with(".md") || name.ends_with(".rs") {
            files.push(path);
        }
    }
}
