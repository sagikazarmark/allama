//! The GenAI span attribute tables allama is pinned to, checked in.
//!
//! Source: OpenTelemetry semantic conventions v1.41.0,
//! `docs/gen-ai/gen-ai-spans.md`, sections "Inference" and "Embeddings"
//! (<https://github.com/open-telemetry/semantic-conventions/blob/v1.41.0/docs/gen-ai/gen-ai-spans.md>).
//! The GenAI conventions now continue in
//! `open-telemetry/semantic-conventions-genai`.
//!
//! When moving to a newer version, copy the tables again in full, update
//! [`VERSION`], and update the version in the `TracingProvider` docs.

/// The semantic conventions version these tables came from.
pub const VERSION: &str = "1.41.0";

/// An attribute type, as the conventions name it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeType {
    String,
    Int,
    Double,
    Boolean,
    /// `string[]`.
    StringArray,
    /// `any`: structured content, which may be recorded as a JSON string.
    Any,
}

use AttributeType::{Any, Boolean, Double, Int, String, StringArray};

/// The inference span (`chat`, `generate_content`, `text_completion`).
pub const INFERENCE: &[(&str, AttributeType)] = &[
    ("gen_ai.operation.name", String),
    ("gen_ai.provider.name", String),
    ("error.type", String),
    ("gen_ai.conversation.id", String),
    ("gen_ai.output.type", String),
    ("gen_ai.request.choice.count", Int),
    ("gen_ai.request.model", String),
    ("gen_ai.request.seed", Int),
    ("gen_ai.request.stream", Boolean),
    ("server.port", Int),
    ("gen_ai.request.frequency_penalty", Double),
    ("gen_ai.request.max_tokens", Int),
    ("gen_ai.request.presence_penalty", Double),
    ("gen_ai.request.stop_sequences", StringArray),
    ("gen_ai.request.temperature", Double),
    ("gen_ai.request.top_k", Double),
    ("gen_ai.request.top_p", Double),
    ("gen_ai.response.finish_reasons", StringArray),
    ("gen_ai.response.id", String),
    ("gen_ai.response.model", String),
    // Seconds.
    ("gen_ai.response.time_to_first_chunk", Double),
    ("gen_ai.usage.cache_creation.input_tokens", Int),
    ("gen_ai.usage.cache_read.input_tokens", Int),
    ("gen_ai.usage.input_tokens", Int),
    ("gen_ai.usage.output_tokens", Int),
    ("gen_ai.usage.reasoning.output_tokens", Int),
    ("server.address", String),
    ("gen_ai.input.messages", Any),
    ("gen_ai.output.messages", Any),
    ("gen_ai.system_instructions", Any),
    ("gen_ai.tool.definitions", Any),
];

/// The embeddings span.
pub const EMBEDDINGS: &[(&str, AttributeType)] = &[
    ("gen_ai.operation.name", String),
    ("gen_ai.provider.name", String),
    ("error.type", String),
    ("gen_ai.request.model", String),
    ("server.port", Int),
    ("gen_ai.embeddings.dimension.count", Int),
    ("gen_ai.request.encoding_formats", StringArray),
    ("gen_ai.response.model", String),
    ("gen_ai.usage.input_tokens", Int),
    ("server.address", String),
];

/// The table an allama span is checked against, by its `tracing` span name.
/// Decision and extraction spans have no table of their own; both are
/// inference calls, so they use the inference table.
pub fn table_for(span_name: &str) -> &'static [(&'static str, AttributeType)] {
    match span_name {
        "chat" | "decision" | "extraction" => INFERENCE,
        "embeddings" => EMBEDDINGS,
        other => panic!("no semantic conventions table for `{other}` spans"),
    }
}

/// The type of `name` in `table`, if the table defines it.
pub fn attribute_type(table: &[(&str, AttributeType)], name: &str) -> Option<AttributeType> {
    table.iter().find(|(n, _)| *n == name).map(|(_, t)| *t)
}
