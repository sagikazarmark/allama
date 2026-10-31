//! The guides quote the examples in their `rust,ignore` blocks.
//!
//! Every such block must still be in one of the quoted examples, line for
//! line (indentation aside), and every quoted example must be quoted at
//! least once. The guides' other code blocks are self-contained and run as
//! doc tests of this crate (`src/lib.rs`).
//!
//! The `allama` crate docs open with the README's quick start, which must
//! stay the same program.

/// The examples the applications guide quotes, by file name.
const APPLICATIONS_QUOTED: [(&str, &str); 3] = [
    ("axum_sse.rs", include_str!("../examples/axum_sse.rs")),
    ("chat_repl.rs", include_str!("../examples/chat_repl.rs")),
    (
        "tower_transport.rs",
        include_str!("../examples/tower_transport.rs"),
    ),
];

const APPLICATIONS_GUIDE: &str = include_str!("../../../docs/guides/applications.md");

/// The example the observability guide's "Trace context propagation"
/// section quotes. The guide's other `rust,ignore` blocks are fragments,
/// not quotes.
const PROPAGATION_QUOTED: [(&str, &str); 1] = [(
    "trace_context.rs",
    include_str!("../examples/trace_context.rs"),
)];

const OBSERVABILITY_GUIDE: &str = include_str!("../../../docs/guides/observability/README.md");

const README: &str = include_str!("../../../README.md");

/// The `allama` crate docs, whose quick start must be the README's: the
/// crate cannot depend on `allama-providers`, so its doc test stands in for
/// it, and the README runs the same program against the real crate.
const ALLAMA_CRATE_DOCS: &str = include_str!("../../allama/src/lib.rs");

#[test]
fn the_applications_guide_quotes_the_examples() {
    assert_quotes(quotes(APPLICATIONS_GUIDE), &APPLICATIONS_QUOTED);
}

#[test]
fn the_observability_guide_quotes_the_propagation_example() {
    let section = section(OBSERVABILITY_GUIDE, "## Trace context propagation\n");
    assert_quotes(quotes(section), &PROPAGATION_QUOTED);
}

#[test]
fn the_crate_docs_quick_start_is_the_readmes() {
    let readme = section(README, "## Quick start\n");
    let docs: String = ALLAMA_CRATE_DOCS
        .lines()
        .filter_map(|line| line.strip_prefix("//!"))
        .map(|line| line.strip_prefix(' ').unwrap_or(line))
        .collect::<Vec<_>>()
        .join("\n");
    let docs = section(&docs, "# Quick start\n");

    let readme_blocks: Vec<String> = rust_blocks(readme).take(2).collect();
    let docs_blocks: Vec<String> = rust_blocks(docs).take(2).collect();
    assert_eq!(
        readme_blocks.len(),
        2,
        "the README's quick start has a program and a stream"
    );
    assert_eq!(
        docs_blocks, readme_blocks,
        "the crate docs' quick start differs from the README's"
    );
}

/// The visible lines of the Rust code blocks in `text`, in order: rustdoc's
/// hidden `# ` lines are left out.
fn rust_blocks(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split("```").skip(1).step_by(2).filter_map(|block| {
        let (info, body) = block.split_once('\n')?;
        let rust = matches!(info, "" | "rust" | "no_run" | "rust,no_run");
        rust.then(|| {
            let visible = body
                .lines()
                .filter(|line| *line != "#" && !line.trim_start().starts_with("# "));
            visible.collect::<Vec<_>>().join("\n")
        })
    })
}

/// Every quote is in one of `quoted`, and every file in `quoted` is quoted.
fn assert_quotes<'a>(quotes: impl Iterator<Item = &'a str>, quoted: &[(&str, &str)]) {
    let sources: Vec<(&str, Vec<&str>)> = quoted
        .iter()
        .map(|(name, source)| (*name, source.lines().map(str::trim).collect()))
        .collect();
    let mut counts = vec![0; sources.len()];

    for quote in quotes {
        let lines: Vec<&str> = quote.lines().map(str::trim).collect();
        let found = sources
            .iter()
            .position(|(_, source)| source.windows(lines.len()).any(|window| window == lines));
        let Some(file) = found else {
            panic!("the guide quotes code that is in none of the quoted examples:\n{quote}");
        };
        counts[file] += 1;
    }

    for ((name, _), count) in sources.iter().zip(counts) {
        assert!(count > 0, "the guide quotes no code from {name}");
    }
}

/// The section of `guide` that starts at `heading`, up to the next
/// second-level heading.
fn section<'a>(guide: &'a str, heading: &str) -> &'a str {
    let start = guide.find(heading).expect("the guide has the section");
    let rest = &guide[start + heading.len()..];
    let end = rest.find("\n## ").unwrap_or(rest.len());
    &rest[..end]
}

/// The bodies of the `rust,ignore` blocks in `guide`.
fn quotes(guide: &str) -> impl Iterator<Item = &str> {
    guide.split("```rust,ignore\n").skip(1).map(|rest| {
        let end = rest.find("\n```").expect("the code block is closed");
        &rest[..end]
    })
}
