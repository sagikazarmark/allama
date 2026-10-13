//! Models that take the query/document role as a text prefix, for the
//! message of a dropped [`input_type`](crate::EmbeddingRequest::input_type).
//!
//! The library never rewrites the caller's text: prefixes are
//! model-specific, and adding one silently changes what is embedded. The
//! message tells the caller which prefix the model's card documents, so
//! they can add it themselves.

use crate::EmbeddingInputType;

/// The role prefixes a model's card documents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RolePrefix {
    /// What starts a query.
    query: &'static str,
    /// What starts a document; `None` when documents take no prefix.
    document: Option<&'static str>,
}

/// nomic-embed-text v1.5 and v2-moe:
/// <https://huggingface.co/nomic-ai/nomic-embed-text-v1.5>,
/// <https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe>.
const NOMIC: RolePrefix = RolePrefix {
    query: "search_query: ",
    document: Some("search_document: "),
};

/// E5 v2 and multilingual E5:
/// <https://huggingface.co/intfloat/e5-large-v2>,
/// <https://huggingface.co/intfloat/multilingual-e5-large>.
const E5: RolePrefix = RolePrefix {
    query: "query: ",
    document: Some("passage: "),
};

/// Instruction-tuned E5 and Qwen3-Embedding: an instruction on queries
/// only ("no need to add instruction for retrieval documents"):
/// <https://huggingface.co/intfloat/e5-mistral-7b-instruct>,
/// <https://huggingface.co/intfloat/multilingual-e5-large-instruct>,
/// <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B>.
const INSTRUCT: RolePrefix = RolePrefix {
    query: "Instruct: {task}\\nQuery: ",
    document: None,
};

/// BGE English v1.5, mxbai-embed-large and arctic-embed v1/v1.5, on
/// queries only ("instruction is not needed for passages"):
/// <https://huggingface.co/BAAI/bge-large-en-v1.5>,
/// <https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1>,
/// <https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5>.
const REPRESENT: RolePrefix = RolePrefix {
    query: "Represent this sentence for searching relevant passages: ",
    document: None,
};

/// arctic-embed v2.0, on queries only ("use the query prefix below (just
/// on the query)"):
/// <https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0>.
const ARCTIC_V2: RolePrefix = RolePrefix {
    query: "query: ",
    document: None,
};

/// Google's task prompts for `gemini-embedding-2` and EmbeddingGemma:
/// <https://ai.google.dev/gemini-api/docs/embeddings>,
/// <https://ai.google.dev/gemma/docs/embeddinggemma/model_card>.
const GOOGLE: RolePrefix = RolePrefix {
    query: "task: search result | query: ",
    document: Some("title: none | text: "),
};

/// The role prefixes of `model`, when it is a model known to take them.
///
/// The id is matched by its last path segment (`@cf/baai/…`, `BAAI/…`,
/// `models/…`), lowercased, without an Ollama tag (`:latest`). BGE-M3 needs
/// no instruction, so it is not a prefix model.
pub(crate) fn role_prefix(model: &str) -> Option<RolePrefix> {
    let id = model.rsplit('/').next().unwrap_or(model);
    let id = id.split(':').next().unwrap_or(id).to_ascii_lowercase();
    let has = |needle: &str| id.contains(needle);
    let prefix = if has("nomic-embed-text") {
        NOMIC
    } else if has("e5-mistral-7b-instruct")
        || has("multilingual-e5-large-instruct")
        || has("qwen3-embedding")
    {
        INSTRUCT
    } else if ["e5-small-v2", "e5-base-v2", "e5-large-v2"]
        .iter()
        .any(|name| has(name))
        || has("multilingual-e5-")
    {
        E5
    } else if has("arctic-embed") && (has("v2.0") || has("arctic-embed2")) {
        ARCTIC_V2
    } else if (has("bge-") && has("-en-v1.5")) || has("mxbai-embed-large") || has("arctic-embed") {
        REPRESENT
    } else if has("gemini-embedding-2") || has("embeddinggemma") {
        GOOGLE
    } else {
        return None;
    };
    Some(prefix)
}

/// The message of an `input_type` that `provider` takes no field for, with
/// `model`: the prefix the model's card documents for the role, as the
/// remedy, or, for a model not known to take one, that it may expect one.
///
/// Under a strict policy the refusal leaves the outcome out and keeps the
/// advice (see [`Degradations`](crate::adapter::Degradations#messages)).
pub(crate) fn input_type_message(
    provider: &str,
    model: &str,
    input_type: EmbeddingInputType,
) -> String {
    // Without a known prefix the message is about the capability answer,
    // which describes the endpoint (ADR 0015); a known prefix is a claim
    // about the model's card.
    let Some(prefix) = role_prefix(model) else {
        return format!(
            "`{provider}` takes no query/document input type (`{model}` may expect a role \
             prefix in the text); the `{input_type}` input type was not sent"
        );
    };
    // Exhaustive on purpose: a new input type needs its advice.
    let advice = match input_type {
        EmbeddingInputType::Query => {
            format!("start each query with `{}`", prefix.query)
        }
        EmbeddingInputType::Document => match prefix.document {
            Some(document) => format!("start each document with `{document}`"),
            None => format!(
                "documents take no prefix (queries start with `{}`)",
                prefix.query
            ),
        },
    };
    format!(
        "`{model}` takes the role as a text prefix, not a field; the `{input_type}` input type \
         was not sent; {advice}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every known prefix family, under the ids its providers use.
    #[test]
    fn prefix_models_are_recognized_by_their_ids() {
        for (models, prefix) in [
            (
                &[
                    "nomic-embed-text",
                    "nomic-embed-text:latest",
                    "nomic-ai/nomic-embed-text-v1.5",
                    "nomic-ai/nomic-embed-text-v2-moe",
                    "@cf/nomic-ai/nomic-embed-text-v1.5",
                ][..],
                NOMIC,
            ),
            (
                &[
                    "intfloat/e5-large-v2",
                    "e5-base-v2",
                    "intfloat/multilingual-e5-large",
                    "multilingual-e5-small",
                ][..],
                E5,
            ),
            (
                &[
                    "intfloat/e5-mistral-7b-instruct",
                    "intfloat/multilingual-e5-large-instruct",
                    "Qwen/Qwen3-Embedding-8B",
                    "@cf/qwen/qwen3-embedding-0.6b",
                    "qwen3-embedding:4b",
                ][..],
                INSTRUCT,
            ),
            (
                &[
                    "@cf/baai/bge-base-en-v1.5",
                    "BAAI/bge-large-en-v1.5",
                    "mxbai-embed-large",
                    "mixedbread-ai/mxbai-embed-large-v1",
                    "snowflake-arctic-embed",
                    "Snowflake/snowflake-arctic-embed-m-v1.5",
                ][..],
                REPRESENT,
            ),
            (
                &[
                    "Snowflake/snowflake-arctic-embed-l-v2.0",
                    "snowflake-arctic-embed2",
                ][..],
                ARCTIC_V2,
            ),
            (
                &[
                    "gemini-embedding-2",
                    "models/gemini-embedding-2-preview",
                    "embeddinggemma",
                    "google/embeddinggemma-300m",
                    "@cf/google/embeddinggemma-300m",
                ][..],
                GOOGLE,
            ),
        ] {
            for model in models {
                assert_eq!(role_prefix(model), Some(prefix), "{model}");
            }
        }
    }

    /// BGE-M3 needs no instruction (<https://huggingface.co/BAAI/bge-m3>),
    /// and the symmetric and field-based models take none either.
    #[test]
    fn other_models_take_no_known_prefix() {
        for model in [
            "@cf/baai/bge-m3",
            "BAAI/bge-m3",
            "bge-m3",
            "text-embedding-3-small",
            "gemini-embedding-001",
            "embedding-001",
            "BAAI/bge-small-zh-v1.5",
            "m",
        ] {
            assert_eq!(role_prefix(model), None, "{model}");
        }
    }

    #[test]
    fn messages_name_the_prefix_for_the_role() {
        use EmbeddingInputType::{Document, Query};
        for (model, input_type, message) in [
            (
                "nomic-embed-text",
                Query,
                "`nomic-embed-text` takes the role as a text prefix, not a field; the `query` \
                 input type was not sent; start each query with `search_query: `",
            ),
            (
                "nomic-embed-text",
                Document,
                "`nomic-embed-text` takes the role as a text prefix, not a field; the \
                 `document` input type was not sent; start each document with \
                 `search_document: `",
            ),
            (
                "@cf/baai/bge-base-en-v1.5",
                Document,
                "`@cf/baai/bge-base-en-v1.5` takes the role as a text prefix, not a field; the \
                 `document` input type was not sent; documents take no prefix (queries start \
                 with `Represent this sentence for searching relevant passages: `)",
            ),
            (
                "Qwen/Qwen3-Embedding-8B",
                Query,
                "`Qwen/Qwen3-Embedding-8B` takes the role as a text prefix, not a field; the \
                 `query` input type was not sent; start each query with \
                 `Instruct: {task}\\nQuery: `",
            ),
            (
                "gemini-embedding-2",
                Document,
                "`gemini-embedding-2` takes the role as a text prefix, not a field; the \
                 `document` input type was not sent; start each document with \
                 `title: none | text: `",
            ),
            (
                "m",
                Query,
                "`demo` takes no query/document input type (`m` may expect a role prefix in \
                 the text); the `query` input type was not sent",
            ),
        ] {
            assert_eq!(
                input_type_message("demo", model, input_type),
                message,
                "{model} {input_type}"
            );
        }
    }

    /// Under a strict policy the refusal keeps the advice, not the outcome.
    #[test]
    fn a_refusal_keeps_the_advice() {
        let message = input_type_message("demo", "nomic-embed-text", EmbeddingInputType::Query);
        let mut strict = crate::Degradations::new("demo", crate::UnsupportedPolicy::Error);
        let err = strict
            .unsupported_hint(crate::RequestField::InputType, message)
            .unwrap_err();
        assert_eq!(
            err.message(),
            "cannot send `input_type`: `nomic-embed-text` takes the role as a text prefix, not \
             a field; start each query with `search_query: `"
        );
    }
}
