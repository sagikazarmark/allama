//! What the golden history (`tests/fixtures/history_v1.json`) covers.
//!
//! The matches below are exhaustive, which only this crate can write: a new
//! message role, user or tool result content form, content block, content
//! part, media source, citation source, image detail or cache retention
//! fails to compile here. Give it a label, add an entry that holds it to the
//! golden history (and its copies in the provider crates), and add the label
//! to `golden_history_covers_every_kind`'s list. The format contract
//! itself is in `tests/format_contract.rs`.

use std::collections::BTreeSet;

use super::History;
use crate::{
    CacheRetention, CitationSource, ContentBlock, ContentPart, Image, ImageDetail, MediaSource,
    Message, OtherContent, ToolResultContent, UserContent,
};

const GOLDEN_HISTORY: &str = include_str!("../../tests/fixtures/history_v1.json");

/// The kinds a history holds, as `"group:kind"` labels.
fn kinds(history: &History) -> BTreeSet<String> {
    let mut kinds = BTreeSet::new();
    let mut add = |kind: String| {
        kinds.insert(kind);
    };
    for message in &history.messages {
        add(format!("message:{}", message.role()));
        if let Some(cache) = message.cache() {
            add(match cache.retention {
                None => "message:cache_default".into(),
                Some(CacheRetention::Short) => "message:cache_retention_short".into(),
                Some(CacheRetention::Long) => "message:cache_retention_long".into(),
            });
        }
        if !message.provider_data().is_empty() {
            add("message:provider_data".into());
        }
        if !message.app_data().is_empty() {
            add(format!("app_data:{}", message.role()));
        }
        match message {
            Message::User(user) => match &user.content {
                UserContent::Text(_) => add("user_content:text".into()),
                UserContent::Parts(parts) => {
                    add("user_content:parts".into());
                    part_kinds(parts, &mut add);
                }
            },
            Message::Assistant(assistant) => {
                for block in &assistant.content {
                    block_kinds(block, &mut add);
                }
            }
            Message::Tool(tool) => {
                if tool.is_error {
                    add("tool_content:error".into());
                }
                match &tool.content {
                    ToolResultContent::Text(_) => add("tool_content:text".into()),
                    ToolResultContent::Object(_) => add("tool_content:object".into()),
                    ToolResultContent::Parts(parts) => {
                        add("tool_content:parts".into());
                        part_kinds(parts, &mut add);
                    }
                }
            }
        }
    }
    kinds
}

fn part_kinds(parts: &[ContentPart], add: &mut impl FnMut(String)) {
    for part in parts {
        let (kind, provider_data) = match part {
            ContentPart::Text(text) => ("text", &text.provider_data),
            ContentPart::Image(image) => {
                image_kinds(image, add);
                ("image", &image.provider_data)
            }
            ContentPart::File(file) => {
                add(format!("file:{}", source_kind(&file.source)));
                if file.filename.is_some() {
                    add("file:filename".into());
                }
                ("file", &file.provider_data)
            }
            ContentPart::Other(other) => {
                other_kinds(other, add);
                ("other", &other.provider_data)
            }
        };
        add(format!("part:{kind}"));
        if !provider_data.is_empty() {
            add(format!("part_provider_data:{kind}"));
        }
    }
}

fn block_kinds(block: &ContentBlock, add: &mut impl FnMut(String)) {
    let kind = match block {
        ContentBlock::Text(text) => {
            for citation in &text.citations {
                add(format!("citation:{}", citation_kind(&citation.source)));
                if citation.span.is_some() {
                    add("citation:span".into());
                }
                if citation.cited_text.is_some() {
                    add("citation:cited_text".into());
                }
                if !citation.provider_data.is_empty() {
                    add("citation:provider_data".into());
                }
            }
            "text"
        }
        ContentBlock::Image(image) => {
            image_kinds(image, add);
            "image"
        }
        ContentBlock::ToolCall(call) => {
            if !call.has_valid_arguments() {
                add("tool_call:malformed_arguments".into());
            }
            "tool_call"
        }
        ContentBlock::Reasoning(reasoning) => {
            if reasoning.is_opaque() {
                add("reasoning:opaque".into());
            }
            "reasoning"
        }
        ContentBlock::Refusal(_) => "refusal",
        ContentBlock::ProviderToolCall(_) => "provider_tool_call",
        ContentBlock::ProviderToolResult(result) => {
            if result.is_error {
                add("provider_tool_result:error".into());
            }
            "provider_tool_result"
        }
        ContentBlock::Other(other) => {
            other_kinds(other, add);
            "other"
        }
    };
    add(format!("block:{kind}"));
    if !block.provider_data().is_empty() {
        add(format!("block_provider_data:{kind}"));
    }
}

fn other_kinds(other: &OtherContent, add: &mut impl FnMut(String)) {
    if other.origin.is_some() {
        add("other:origin".into());
    }
    if other.data.contains_key("thoughtSignature") {
        add("other:wire_gemini_thought_signature".into());
    }
    if ["type", "type_name", "origin", "provider_data"]
        .iter()
        .all(|key| other.data.contains_key(*key))
    {
        add("other:data_with_field_names".into());
    }
}

fn image_kinds(image: &Image, add: &mut impl FnMut(String)) {
    add(format!("image:{}", source_kind(&image.source)));
    if image.media_type.is_some() {
        add("image:media_type".into());
    }
    if let Some(detail) = image.detail {
        add(match detail {
            ImageDetail::Auto => "image:detail_auto".into(),
            ImageDetail::Low => "image:detail_low".into(),
            ImageDetail::High => "image:detail_high".into(),
        });
    }
}

fn source_kind(source: &MediaSource) -> &'static str {
    match source {
        MediaSource::Url { .. } => "url",
        MediaSource::Base64 { .. } => "base64",
        MediaSource::ProviderFile { .. } => "provider_file",
    }
}

fn citation_kind(source: &CitationSource) -> &'static str {
    match source {
        CitationSource::Url { .. } => "url",
        CitationSource::Document { .. } => "document",
        CitationSource::File { .. } => "file",
    }
}

#[test]
fn golden_history_covers_every_kind() {
    let expected = [
        "app_data:assistant",
        "app_data:tool",
        "app_data:user",
        "block:image",
        "block:other",
        "block:provider_tool_call",
        "block:provider_tool_result",
        "block:reasoning",
        "block:refusal",
        "block:text",
        "block:tool_call",
        "block_provider_data:image",
        "block_provider_data:other",
        "block_provider_data:provider_tool_call",
        "block_provider_data:provider_tool_result",
        "block_provider_data:reasoning",
        "block_provider_data:refusal",
        "block_provider_data:text",
        "block_provider_data:tool_call",
        "citation:cited_text",
        "citation:document",
        "citation:file",
        "citation:provider_data",
        "citation:span",
        "citation:url",
        "file:base64",
        "file:filename",
        "file:provider_file",
        "file:url",
        "image:base64",
        "image:detail_auto",
        "image:detail_high",
        "image:detail_low",
        "image:media_type",
        "image:url",
        "message:assistant",
        "message:cache_default",
        "message:cache_retention_long",
        "message:cache_retention_short",
        "message:provider_data",
        "message:tool",
        "message:user",
        "other:data_with_field_names",
        "other:origin",
        "other:wire_gemini_thought_signature",
        "part:file",
        "part:image",
        "part:other",
        "part:text",
        "part_provider_data:other",
        "provider_tool_result:error",
        "reasoning:opaque",
        "tool_call:malformed_arguments",
        "tool_content:error",
        "tool_content:object",
        "tool_content:parts",
        "tool_content:text",
        "user_content:parts",
        "user_content:text",
    ];
    let golden: History = serde_json::from_str(GOLDEN_HISTORY).expect("the golden history loads");
    let expected: BTreeSet<String> = expected.into_iter().map(str::to_owned).collect();
    assert_eq!(kinds(&golden), expected);
}
