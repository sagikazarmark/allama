#![forbid(unsafe_code)]
//! Doc tests for the guides and the READMEs: the workspace README and the
//! guides live outside every published package, and each crate README
//! runs here, where the dependencies its examples use (a multi-threaded
//! Tokio runtime, `allama-providers`, mocks) are all available.
//!
//! The examples in `examples/` build their providers themselves, from a
//! model address, the way an application reads its model from
//! configuration; this library holds no setup for them.
//!
//! [`workspace`] lists the crates of the workspace for the tests that read
//! every crate.

pub mod workspace;

/// The applications guide (`docs/guides/applications.md`), whose
/// self-contained code blocks run as doc tests of this crate. Its
/// `rust,ignore` blocks quote the examples, and `tests/guide_quotes.rs`
/// checks the quotes.
#[cfg(doctest)]
#[doc = include_str!("../../../docs/guides/applications.md")]
pub struct ApplicationsGuide;

/// The observability guide (`docs/guides/observability/README.md`), whose
/// code blocks run as doc tests of this crate.
#[cfg(doctest)]
#[doc = include_str!("../../../docs/guides/observability/README.md")]
pub struct ObservabilityGuide;

/// The workspace README, whose Rust code blocks run as doc tests of this
/// crate.
#[cfg(doctest)]
#[doc = include_str!("../../../README.md")]
pub struct Readme;

/// The crate READMEs, whose Rust code blocks run as doc tests of this
/// crate, so every quick start compiles.
#[cfg(doctest)]
pub mod readmes {
    /// `crates/allama/README.md`.
    #[doc = include_str!("../../allama/README.md")]
    pub struct Allama;

    /// `crates/allama-anthropic/README.md`.
    #[doc = include_str!("../../allama-anthropic/README.md")]
    pub struct AllamaAnthropic;

    /// `crates/allama-cloudflare-worker/README.md`.
    #[doc = include_str!("../../allama-cloudflare-worker/README.md")]
    pub struct AllamaCloudflareWorker;

    /// `crates/allama-conformance/README.md`.
    #[doc = include_str!("../../allama-conformance/README.md")]
    pub struct AllamaConformance;

    /// `crates/allama-gemini/README.md`.
    #[doc = include_str!("../../allama-gemini/README.md")]
    pub struct AllamaGemini;

    /// `crates/allama-http/README.md`.
    #[doc = include_str!("../../allama-http/README.md")]
    pub struct AllamaHttp;

    /// `crates/allama-openai/README.md`.
    #[doc = include_str!("../../allama-openai/README.md")]
    pub struct AllamaOpenai;

    /// `crates/allama-openai-compat/README.md`.
    #[doc = include_str!("../../allama-openai-compat/README.md")]
    pub struct AllamaOpenaiCompat;

    /// `crates/allama-providers/README.md`.
    #[doc = include_str!("../../allama-providers/README.md")]
    pub struct AllamaProviders;

    /// `crates/allama-systemone/README.md`.
    #[doc = include_str!("../../allama-systemone/README.md")]
    pub struct AllamaSystemone;
}
