//! The provider-author toolkit: the steps every adapter shares around
//! encoding a request and decoding a response.
//!
//! Applications do not need this module. It is for code that implements
//! [`ChatProvider`](crate::ChatProvider),
//! [`EmbeddingProvider`](crate::EmbeddingProvider) or
//! [`DecisionProvider`](crate::DecisionProvider) against a wire API: a
//! provider crate, or an application's own adapter for an in-house
//! gateway. The [provider authors' guide] walks through it.
//!
//! - [`chat`]: [`prepare`](chat::prepare) and
//!   [`prepare_dry_run`](chat::prepare_dry_run), which check a request and
//!   carry its warnings through encoding to the response
//!   ([`Prepared`](chat::Prepared)), so the same encoding serves the call
//!   and its dry run (validation); [`preflight`](chat::preflight), the
//!   capability check `prepare` runs before encoding (ADR 0003);
//!   [`WireSupport`](chat::WireSupport) and
//!   [`Prepared::degrade_to`](chat::Prepared::degrade_to), which refuse or
//!   drop what a wire format cannot carry; [`retain_turns`](chat::retain_turns),
//!   the pass over the encoded turns; [`start_first`](chat::start_first),
//!   the rule for the events a stream yields before its `ResponseStart`;
//!   [`MISSING_FINISH_REASON`](chat::MISSING_FINISH_REASON); and
//!   [`ToolCallIdFormat`](chat::ToolCallIdFormat), which maps the tool call
//!   ids of a history from another provider to ones the wire accepts.
//! - [`embedding`] and [`decision`]: the same `prepare` step for the other
//!   operations, and [`check_response`](embedding::check_response) for
//!   embedding vectors.
//! - [`schema`]: the JSON Schema passes adapters compose into a provider's
//!   dialect, and the strict-schema policy.
//! - [`Degradations`]: the warnings and refusals a request collects while
//!   it is encoded.
//!
//! Stream decoders also use [`MetadataBuilder`](crate::chat::stream::MetadataBuilder)
//! and [`StreamCollector`](crate::chat::stream::StreamCollector) from
//! [`chat::stream`](crate::chat::stream), which applications use too.
//!
#![doc = concat!(
    "[provider authors' guide]: https://github.com/sagikazarmark/allama/blob/v",
    env!("CARGO_PKG_VERSION"),
    "/docs/guides/provider-authors.md",
)]

pub use crate::warning::degradations::Degradations;

/// The namespaces some state came from, for a warning or error message:
/// `` `openai`, `gemini` ``, or "an unknown origin" when there are none.
///
/// Adapters name the origin of the replayed state they drop
/// ([`Degradations::dropped_replay_state`]) with it, so every adapter
/// words it the same way.
///
/// ```
/// use allama::adapter::describe_origins;
///
/// assert_eq!(describe_origins(&["openai", "gemini"]), "`openai`, `gemini`");
/// // One optional origin, such as `OtherContent::origin`:
/// let origin: Option<&str> = None;
/// assert_eq!(describe_origins(origin.as_slice()), "an unknown origin");
/// ```
#[must_use]
pub fn describe_origins<S: AsRef<str>>(namespaces: &[S]) -> String {
    if namespaces.is_empty() {
        return "an unknown origin".to_owned();
    }
    namespaces
        .iter()
        .map(|namespace| format!("`{}`", namespace.as_ref()))
        .collect::<Vec<_>>()
        .join(", ")
}

pub mod chat {
    //! The chat steps adapters share around encoding a request.
    //!
    //! # Abnormal finishes
    //!
    //! A response that is not a complete answer still arrives as `Ok`, with
    //! the reason in [`ChatResponse::finish_reason`](crate::ChatResponse::finish_reason).
    //! A stream that ends with its terminal marker but no finish reason is
    //! `Ok` too, with a warning coded [`MISSING_FINISH_REASON`]; one that ends
    //! without its terminal marker fails with
    //! [`StreamErrorKind::Incomplete`](crate::StreamErrorKind::Incomplete).
    //!
    //! # Finish events
    //!
    //! With the default `warning-events` feature, a call that finishes at the
    //! output token limit, at the context window, at a content filter or with a
    //! refusal is also a [`tracing`](https://docs.rs/tracing) event at `INFO`,
    //! under the target `allama::finish` (see
    //! [Finish events](crate::chat#finish-events)). [`Prepared::finish`] and
    //! [`finish_stream`](Prepared::finish_stream) report it, once per call: the
    //! innermost layer built on [`prepare`] that sees the finish reports it. A
    //! dry run reports nothing.

    pub use crate::chat::preflight::preflight;
    pub use crate::chat::prepared::{
        MISSING_FINISH_REASON, Prepared, prepare, prepare_dry_run, start_first,
    };
    pub use crate::chat::tool_call_id::ToolCallIdFormat;
    pub use crate::chat::turns::{EncodedTurn, retain_turns};
    pub use crate::chat::wire::WireSupport;

    /// Every key path of a serialized [`ChatRequest`](crate::ChatRequest),
    /// each with the request field audit probe that checks it or the reason
    /// it is exempt.
    ///
    /// This is the source of truth for
    /// `allama_conformance::probe::ACCOUNTED_FIELDS`, which documents the
    /// format; it lives here so the test that derives it from `ChatRequest`
    /// stays inside this package. A new request field fails that test until
    /// it is listed.
    ///
    /// # Stability
    ///
    /// The constant and its line format are public API,
    /// because `allama-conformance` reads it under a caret requirement on
    /// this crate. Its lines change only in a minor release, which adds a
    /// line for each new request field; a line names only probes that the
    /// `allama-conformance` release of the same minor version knows. A patch
    /// release may reword an exemption's reason, nothing else. An older
    /// `allama-conformance` that meets a line it cannot read fails the
    /// request field audit and asks for an upgrade; it does not panic.
    pub const ACCOUNTED_FIELDS: &str = include_str!("chat/accounted_fields.txt");
}

pub mod embedding {
    //! The embedding steps adapters share around encoding a request and
    //! checking its response.

    pub use crate::embedding::check::check_response;
    pub use crate::embedding::preflight::preflight;
    pub use crate::embedding::prepared::{Prepared, prepare, prepare_dry_run};
}

pub mod decision {
    //! The decision steps adapters share around encoding a request.

    pub use crate::decision::prepared::{Prepared, prepare, prepare_dry_run};
}

pub mod schema {
    //! JSON Schema passes shared by provider adapters.
    //!
    //! Tool parameters and `ResponseFormat::JsonSchema` carry portable JSON
    //! Schema. Providers accept different dialects (OpenAI strict mode,
    //! Gemini's subset, …), so adapters rewrite schemas late, with reusable,
    //! neutral passes: [`inline_refs`], [`strip_keywords`],
    //! [`flatten_single_all_of`], [`merge_sibling_refs`],
    //! [`branches_disjoint`] and [`is_recursive`]. A provider's strict dialect
    //! lives in the provider crate and composes them;
    //! [`apply_strict_policy`] runs any dialect and decides between sending
    //! strict, degrading to non-strict with a warning, and failing (ADR 0003).
    //!
    //! [`normalize_strict_nulls`] works on output: it removes the `null`s a
    //! strict dialect let the model send for optional properties.
    //!
    //! With the `schema` feature, `schema_for` generates a schema from a Rust
    //! type with `schemars`. Applications reach the same through
    //! [`ToolDefinition::from_schema`][crate::ToolDefinition::from_schema] and
    //! [`ResponseFormat::from_schema`][crate::ResponseFormat::from_schema].
    //!
    #![cfg_attr(
        not(feature = "schema"),
        doc = "[crate::ToolDefinition::from_schema]: crate#cargo-features\n[crate::ResponseFormat::from_schema]: crate#cargo-features"
    )]

    pub use crate::schema::*;
}
