//! Portable citations from OpenAI-style annotations.
//!
//! Both APIs annotate text with citations: the Responses API on each
//! `output_text` part (`{"type": "url_citation", "url": …}`), Chat
//! Completions on the message, with the fields nested under the type
//! (`{"type": "url_citation", "url_citation": {"url": …}}`). The raw
//! annotations stay in the text block's provider data; these are the
//! portable view (ADR 0007).

use std::ops::Range;

use allama::{Citation, CitationSource, ExtraMap, ProviderData};
use serde_json::Value;

/// Portable citations for `annotations` on `text`. Annotation types without
/// a citation meaning (`file_path`) and unknown types are skipped.
pub(crate) fn citations(annotations: Option<&Value>, text: &str, namespace: &str) -> Vec<Citation> {
    annotations
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|annotation| citation(annotation, text, namespace))
        .collect()
}

fn citation(annotation: &Value, text: &str, namespace: &str) -> Option<Citation> {
    let mut fields = annotation.as_object()?.clone();
    let kind = match fields.remove("type") {
        Some(Value::String(kind)) => kind,
        _ => return None,
    };
    // Chat Completions nests the fields under the type name.
    if let Some(Value::Object(nested)) = fields.remove(&kind) {
        fields.extend(nested);
    }
    let source = match kind.as_str() {
        "url_citation" => CitationSource::url(
            take_string(&mut fields, "url")?,
            take_string(&mut fields, "title"),
        ),
        "file_citation" | "container_file_citation" => CitationSource::file(
            take_string(&mut fields, "file_id")?,
            take_string(&mut fields, "filename"),
        ),
        _ => return None,
    };
    let mut citation = Citation::new(source);
    let start = take_index(&mut fields, "start_index");
    let end = take_index(&mut fields, "end_index");
    if let (Some(start), Some(end)) = (start, end)
        && let Some(span) = char_span(text, start, end)
    {
        citation = citation.with_span(span);
    }
    let mut provider_data = ProviderData::new().with_raw(namespace, "type", Value::String(kind));
    for (key, value) in fields.into_iter().filter(|(_, v)| !v.is_null()) {
        provider_data.insert_raw(namespace, key, value);
    }
    Some(citation.with_provider_data(provider_data))
}

/// The byte range of characters `start..end` of `text`, or `None` when the
/// range does not fit. OpenAI counts Unicode code points (verified on
/// recordings with emoji and accented text before the citation), which is
/// what `char_indices` counts.
fn char_span(text: &str, start: usize, end: usize) -> Option<Range<usize>> {
    if start > end {
        return None;
    }
    let mut offsets = text
        .char_indices()
        .map(|(offset, _)| offset)
        .chain(std::iter::once(text.len()));
    let start_byte = offsets.nth(start)?;
    let end_byte = if end == start {
        start_byte
    } else {
        offsets.nth(end - start - 1)?
    };
    Some(start_byte..end_byte)
}

fn take_string(fields: &mut ExtraMap, key: &str) -> Option<String> {
    match fields.remove(key) {
        Some(Value::String(value)) => Some(value),
        _ => None,
    }
}

fn take_index(fields: &mut ExtraMap, key: &str) -> Option<usize> {
    let value = fields.remove(key)?;
    let index = value.as_u64().and_then(|i| usize::try_from(i).ok());
    if index.is_none() {
        fields.insert(key.to_owned(), value);
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn character_offsets_become_byte_ranges() {
        assert_eq!(char_span("hello", 1, 3), Some(1..3));
        assert_eq!(char_span("hello", 0, 5), Some(0..5));
        assert_eq!(char_span("hello", 5, 5), Some(5..5));
        assert_eq!(char_span("héllo wörld", 6, 11), Some(7..13));
        assert_eq!(char_span("hello", 2, 9), None);
        assert_eq!(char_span("hello", 3, 1), None);
    }

    #[test]
    fn responses_and_chat_completions_shapes() {
        let text = "Rust 1.99 is out. (blog)";
        let responses = json!([
            {"type": "url_citation", "url": "https://blog.rust-lang.org", "title": "Rust Blog", "start_index": 18, "end_index": 24},
            {"type": "file_citation", "file_id": "file_1", "filename": "notes.md", "index": 3},
            {"type": "file_path", "file_id": "file_2", "index": 5}
        ]);
        let cited = citations(Some(&responses), text, "openai");
        assert_eq!(cited.len(), 2);
        assert_eq!(
            cited[0].source,
            CitationSource::url("https://blog.rust-lang.org", Some("Rust Blog".into()))
        );
        assert_eq!(&text[cited[0].span.clone().unwrap()], "(blog)");
        assert_eq!(
            cited[1].source,
            CitationSource::file("file_1", Some("notes.md".into()))
        );
        assert_eq!(cited[1].span, None);
        assert_eq!(
            cited[1].provider_data.get_raw("openai", "index"),
            Some(&json!(3))
        );

        let chat = json!([{"type": "url_citation", "url_citation": {"url": "https://x.dev", "title": "X", "start_index": 0, "end_index": 4}}]);
        let cited = citations(Some(&chat), text, "openai");
        assert_eq!(
            cited[0].source,
            CitationSource::url("https://x.dev", Some("X".into()))
        );
        assert_eq!(cited[0].span, Some(0..4));
        assert_eq!(
            cited[0].provider_data.get_raw("openai", "type"),
            Some(&json!("url_citation"))
        );

        let out_of_range = json!([{"type": "url_citation", "url": "https://x.dev", "start_index": 0, "end_index": 999}]);
        assert_eq!(citations(Some(&out_of_range), text, "openai")[0].span, None);
        assert!(citations(None, text, "openai").is_empty());
    }
}
