//! The audit that keeps [`ACCOUNTED_FIELDS`](crate::adapter::chat::ACCOUNTED_FIELDS) in step
//! with [`ChatRequest`], and the pin on the key paths of the persisted
//! [`History`] format.
//!
//! The samples below build every message, content part, content block,
//! media source and citation source kind with struct literals and
//! exhaustive matches, so a new field or variant fails to compile until it
//! is sampled here. Once sampled, it serializes to a key path that the list
//! must account for and, inside a message, one that `history_v1_paths.txt`
//! pins.

use std::collections::BTreeSet;

use serde_json::{Value, json};

use crate::history::{HISTORY_FORMAT_VERSION, History};
use crate::{
    AssistantMessage, CacheBreakpoint, CacheRetention, ChatRequest, Citation, CitationSource,
    ContentBlock, ContentPart, ExtraMap, File, Image, ImageDetail, MediaSource, Message,
    OtherContent, PromptCache, ProviderData, ProviderToolCall, ProviderToolResult, Reasoning,
    ReasoningConfig, ReasoningEffort, ReasoningOutput, Refusal, RequestOptions, ResponseFormat,
    SystemPrompt, Text, ToolCall, ToolChoice, ToolDefinition, ToolMessage, ToolResultContent,
    UnsupportedPolicy, UserContent, UserMessage,
};

/// One line of the list: a key path and how it is accounted for.
struct Entry<'a> {
    path: &'a str,
    account: &'a str,
}

fn entries() -> Vec<Entry<'static>> {
    crate::adapter::chat::ACCOUNTED_FIELDS
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let (path, account) = line
                .split_once(char::is_whitespace)
                .unwrap_or_else(|| panic!("`{line}` has no account"));
            Entry {
                path,
                account: account.trim(),
            }
        })
        .collect()
}

/// Every key path the samples serialize to, cut at listed paths.
///
/// Arrays add `[]` to the path and objects tagged with `role` or `type` add
/// the tag as a segment (`messages[].user.cache{}`), so each kind has its own
/// paths. A listed path ending in `[]` or `{}` accounts for the whole array
/// or object there; any other listed path accounts for a value that is not
/// an object or array. Returns the unlisted paths and the listed paths
/// reached.
fn walk(listed: &BTreeSet<&str>) -> (BTreeSet<String>, BTreeSet<String>) {
    let requests: Vec<Value> = samples()
        .iter()
        .map(|request| serde_json::to_value(request).expect("the sample serializes"))
        .collect();
    walk_values(&requests, &|path| listed.contains(path))
}

/// [`walk`] over any serialized values, cut at the paths `listed` accepts.
fn walk_values(
    values: &[Value],
    listed: &dyn Fn(&str) -> bool,
) -> (BTreeSet<String>, BTreeSet<String>) {
    let mut walk = Walk {
        listed,
        unlisted: BTreeSet::new(),
        reached: BTreeSet::new(),
    };
    for value in values {
        walk.visit(value, String::new());
    }
    (walk.unlisted, walk.reached)
}

struct Walk<'a> {
    listed: &'a dyn Fn(&str) -> bool,
    unlisted: BTreeSet<String>,
    reached: BTreeSet<String>,
}

impl Walk<'_> {
    fn visit(&mut self, value: &Value, path: String) {
        match value {
            Value::Object(map) if !map.is_empty() => {
                if self.covers(format!("{path}{{}}")) {
                    return;
                }
                let tag = ["role", "type"]
                    .into_iter()
                    .find_map(|key| Some((key, map.get(key)?.as_str()?)));
                let path = match tag {
                    Some((_, tag)) => join(&path, tag),
                    None => path,
                };
                if tag.is_some() && self.covers(format!("{path}{{}}")) {
                    return;
                }
                let mut fields = map
                    .iter()
                    .filter(|(key, _)| tag.is_none_or(|(tag_key, _)| tag_key != key.as_str()))
                    .peekable();
                if fields.peek().is_none() {
                    self.leaf(path);
                    return;
                }
                for (key, item) in fields {
                    self.visit(item, join(&path, key));
                }
            }
            Value::Array(items) if !items.is_empty() => {
                let path = format!("{path}[]");
                if self.covers(path.clone()) {
                    return;
                }
                for item in items {
                    self.visit(item, path.clone());
                }
            }
            _ => self.leaf(path),
        }
    }

    fn covers(&mut self, path: String) -> bool {
        let listed = (self.listed)(&path);
        if listed {
            self.reached.insert(path);
        }
        listed
    }

    fn leaf(&mut self, path: String) {
        if !self.covers(path.clone()) {
            self.unlisted.insert(path);
        }
    }
}

fn join(path: &str, segment: &str) -> String {
    if path.is_empty() {
        segment.to_owned()
    } else {
        format!("{path}.{segment}")
    }
}

/// Requests that, together, set every field of every request type and use
/// every message, part, and block kind.
fn samples() -> Vec<ChatRequest> {
    let full = full_request();
    let mut samples = vec![full.clone()];
    // The fields whose variants change the key paths, one variant each.
    for choice in tool_choices() {
        samples.push(ChatRequest {
            tool_choice: Some(choice),
            ..full.clone()
        });
    }
    for format in response_formats() {
        samples.push(ChatRequest {
            response_format: Some(format),
            ..full.clone()
        });
    }
    samples
}

/// A request with every field set.
fn full_request() -> ChatRequest {
    ChatRequest {
        model: "model".into(),
        system: vec![SystemPrompt {
            content: "system".into(),
            cache: Some(breakpoint()),
            options: options(),
            app_data: app_data(),
        }],
        messages: messages(),
        temperature: Some(0.5),
        max_output_tokens: Some(1),
        top_p: Some(0.5),
        top_k: Some(40),
        stop: vec!["stop".into()],
        frequency_penalty: Some(0.5),
        presence_penalty: Some(0.5),
        tools: vec![ToolDefinition {
            name: "tool".into(),
            description: Some("description".into()),
            parameters: json!({"type": "object"}),
            strict: Some(true),
            cache: Some(breakpoint()),
            provider_data: provider_data(),
        }],
        tool_choice: Some(ToolChoice::Auto),
        response_format: Some(ResponseFormat::Json),
        seed: Some(1),
        reasoning: Some(ReasoningConfig {
            effort: Some(ReasoningEffort::High),
            output: Some(ReasoningOutput::Summary),
        }),
        parallel_tool_calls: Some(true),
        cache: Some(PromptCache {
            key: Some("key".into()),
            retention: Some(CacheRetention::Long),
            automatic: true,
        }),
        unsupported_policy: Some(UnsupportedPolicy::Error),
        options: options(),
    }
}

/// One message of every role, with every content form.
fn messages() -> Vec<Message> {
    let user = |content| {
        Message::User(UserMessage {
            content,
            cache: Some(breakpoint()),
            provider_data: provider_data(),
            app_data: app_data(),
        })
    };
    let tool = |content| {
        Message::Tool(ToolMessage {
            tool_call_id: "call".into(),
            name: "tool".into(),
            content,
            is_error: true,
            cache: Some(breakpoint()),
            provider_data: provider_data(),
            app_data: app_data(),
        })
    };
    let messages = vec![
        user(UserContent::Text("text".into())),
        user(UserContent::Parts(parts())),
        Message::Assistant(AssistantMessage {
            content: blocks(),
            cache: Some(breakpoint()),
            provider_data: provider_data(),
            app_data: app_data(),
        }),
        tool(ToolResultContent::Text("text".into())),
        tool(ToolResultContent::Parts(parts())),
        tool(ToolResultContent::Object(
            json!({"result": "ok"}).as_object().unwrap().clone(),
        )),
    ];

    // Exhaustive: a new role or content form fails to compile here until it
    // is sampled above.
    let kinds: BTreeSet<_> = messages
        .iter()
        .map(|message| match message {
            Message::User(m) => match m.content {
                UserContent::Text(_) => "user text",
                UserContent::Parts(_) => "user parts",
            },
            Message::Assistant(_) => "assistant",
            Message::Tool(m) => match m.content {
                ToolResultContent::Text(_) => "tool text",
                ToolResultContent::Parts(_) => "tool parts",
                ToolResultContent::Object(_) => "tool object",
            },
        })
        .collect();
    assert_eq!(kinds.len(), messages.len(), "one sample per kind");
    messages
}

/// One part of every kind, with every media source.
fn parts() -> Vec<ContentPart> {
    let mut parts = vec![ContentPart::Text(text())];
    for source in sources() {
        parts.push(ContentPart::Image(image(source.clone())));
        parts.push(ContentPart::File(file(source)));
    }
    parts.push(ContentPart::Other(other()));

    let kinds: BTreeSet<_> = parts
        .iter()
        .map(|part| match part {
            ContentPart::Text(_) => "text".to_owned(),
            ContentPart::Image(image) => format!("image {}", source_kind(&image.source)),
            ContentPart::File(file) => format!("file {}", source_kind(&file.source)),
            ContentPart::Other(_) => "other".to_owned(),
        })
        .collect();
    assert_eq!(kinds.len(), parts.len(), "one sample per kind");
    parts
}

/// One block of every kind.
fn blocks() -> Vec<ContentBlock> {
    let blocks = vec![
        ContentBlock::Text(text()),
        ContentBlock::Image(image(MediaSource::Base64 {
            media_type: "image/png".into(),
            data: "AA==".into(),
        })),
        ContentBlock::ToolCall(ToolCall {
            id: "call".into(),
            name: "tool".into(),
            arguments: "{}".into(),
            provider_data: provider_data(),
        }),
        ContentBlock::Reasoning(Reasoning {
            text: "reasoning".into(),
            provider_data: provider_data(),
        }),
        ContentBlock::Refusal(Refusal {
            text: "refusal".into(),
            provider_data: provider_data(),
        }),
        ContentBlock::ProviderToolCall(ProviderToolCall {
            id: "call".into(),
            name: "web_search".into(),
            arguments: "{}".into(),
            origin: Some("origin".into()),
            provider_data: provider_data(),
        }),
        ContentBlock::ProviderToolResult(ProviderToolResult {
            call_id: "call".into(),
            output: json!({"results": []}),
            is_error: true,
            origin: Some("origin".into()),
            provider_data: provider_data(),
        }),
        ContentBlock::Other(other()),
    ];

    let kinds: BTreeSet<_> = blocks
        .iter()
        .map(|block| match block {
            ContentBlock::Text(_) => "text",
            ContentBlock::Image(_) => "image",
            ContentBlock::ToolCall(_) => "tool_call",
            ContentBlock::Reasoning(_) => "reasoning",
            ContentBlock::Refusal(_) => "refusal",
            ContentBlock::ProviderToolCall(_) => "provider_tool_call",
            ContentBlock::ProviderToolResult(_) => "provider_tool_result",
            ContentBlock::Other(_) => "other",
        })
        .collect();
    assert_eq!(kinds.len(), blocks.len(), "one sample per kind");
    blocks
}

fn tool_choices() -> Vec<ToolChoice> {
    let choices = vec![
        ToolChoice::Auto,
        ToolChoice::Disabled,
        ToolChoice::Required,
        ToolChoice::Specific {
            name: "tool".into(),
        },
    ];
    let kinds: BTreeSet<_> = choices
        .iter()
        .map(|choice| match choice {
            ToolChoice::Auto => "auto",
            ToolChoice::Disabled => "disabled",
            ToolChoice::Required => "required",
            ToolChoice::Specific { .. } => "specific",
        })
        .collect();
    assert_eq!(kinds.len(), choices.len(), "one sample per kind");
    choices
}

fn response_formats() -> Vec<ResponseFormat> {
    let formats = vec![
        ResponseFormat::Text,
        ResponseFormat::Json,
        ResponseFormat::JsonSchema {
            name: Some("name".into()),
            schema: json!({"type": "object"}),
            strict: Some(true),
        },
    ];
    let kinds: BTreeSet<_> = formats
        .iter()
        .map(|format| match format {
            ResponseFormat::Text => "text",
            ResponseFormat::Json => "json",
            ResponseFormat::JsonSchema { .. } => "json_schema",
        })
        .collect();
    assert_eq!(kinds.len(), formats.len(), "one sample per kind");
    formats
}

/// One media source of every kind.
fn sources() -> Vec<MediaSource> {
    let sources = vec![
        MediaSource::Url {
            url: "https://example.com/a".into(),
        },
        MediaSource::Base64 {
            media_type: "application/pdf".into(),
            data: "AA==".into(),
        },
        MediaSource::ProviderFile {
            origin: "namespace".into(),
            id: "file".into(),
        },
    ];
    let kinds: BTreeSet<_> = sources.iter().map(source_kind).collect();
    assert_eq!(kinds.len(), sources.len(), "one sample per kind");
    sources
}

fn source_kind(source: &MediaSource) -> &'static str {
    match source {
        MediaSource::Url { .. } => "url",
        MediaSource::Base64 { .. } => "base64",
        MediaSource::ProviderFile { .. } => "provider_file",
    }
}

/// Text with a citation of every source kind.
fn text() -> Text {
    let sources = vec![
        CitationSource::Url {
            url: "https://example.com".into(),
            title: Some("title".into()),
        },
        CitationSource::Document {
            index: Some(0),
            title: Some("title".into()),
        },
        CitationSource::File {
            file_id: "file".into(),
            filename: Some("a.pdf".into()),
        },
    ];
    let kinds: BTreeSet<_> = sources
        .iter()
        .map(|source| match source {
            CitationSource::Url { .. } => "url",
            CitationSource::Document { .. } => "document",
            CitationSource::File { .. } => "file",
        })
        .collect();
    assert_eq!(kinds.len(), sources.len(), "one sample per kind");
    Text {
        text: "text".into(),
        citations: sources
            .into_iter()
            .map(|source| Citation {
                source,
                span: Some(0..1),
                cited_text: Some("t".into()),
                provider_data: provider_data(),
            })
            .collect(),
        provider_data: provider_data(),
    }
}

fn image(source: MediaSource) -> Image {
    Image {
        source,
        media_type: Some("image/png".into()),
        detail: Some(ImageDetail::Low),
        provider_data: provider_data(),
    }
}

fn file(source: MediaSource) -> File {
    File {
        media_type: "application/pdf".into(),
        source,
        filename: Some("a.pdf".into()),
        provider_data: provider_data(),
    }
}

/// `Other` content: it serializes under the reserved `other` tag whatever
/// its wire type name.
fn other() -> OtherContent {
    OtherContent {
        type_name: "vendor_kind".into(),
        data: json!({"key": true}).as_object().unwrap().clone(),
        origin: Some("origin".into()),
        provider_data: provider_data(),
    }
}

fn breakpoint() -> CacheBreakpoint {
    CacheBreakpoint {
        retention: Some(CacheRetention::Short),
    }
}

fn provider_data() -> ProviderData {
    ProviderData::new().with_raw("namespace", "key", Value::Bool(true))
}

fn app_data() -> ExtraMap {
    json!({"key": true}).as_object().unwrap().clone()
}

fn options() -> RequestOptions {
    let mut options = RequestOptions::new();
    options.insert_json("namespace", json!({"key": true}));
    options
}

#[test]
fn every_serialized_request_key_is_accounted_for() {
    let entries = entries();
    let listed: BTreeSet<&str> = entries.iter().map(|e| e.path).collect();
    assert_eq!(listed.len(), entries.len(), "a path is listed twice");

    let (unlisted, reached) = walk(&listed);
    assert!(
        unlisted.is_empty(),
        "these `ChatRequest` key paths are neither probed nor exempt; add them \
         to crates/allama/src/chat/accounted_fields.txt with a probe or an \
         exemption:\n  {}",
        unlisted.into_iter().collect::<Vec<_>>().join("\n  ")
    );
    let stale: Vec<&str> = listed
        .iter()
        .copied()
        .filter(|path| !reached.contains(*path))
        .collect();
    assert!(
        stale.is_empty(),
        "these listed paths no longer appear in a serialized `ChatRequest`; \
         remove them:\n  {}",
        stale.join("\n  ")
    );
}

/// Where the persisted `History` format holds free-form JSON: the walk
/// stops there, since the keys inside belong to providers or the
/// application.
const OPAQUE_HISTORY_PATHS: [&str; 5] = [
    ".provider_data{}",
    ".app_data{}",
    ".other.data{}",
    ".provider_tool_result.output{}",
    ".tool.content{}",
];

/// Every key path of a serialized [`History`] holding every message,
/// content and source kind with every field set is pinned in
/// `history_v1_paths.txt`. A new field passes the golden fixture's round
/// trip, since the fixture does not set it, and a build that does not know
/// it drops it on the next save: that is why every change bumps the format
/// version.
#[test]
fn history_format_paths_are_pinned() {
    let history = serde_json::to_value(History::new(messages())).expect("the sample serializes");
    let (leaves, opaque) = walk_values(&[history], &|path| {
        OPAQUE_HISTORY_PATHS
            .iter()
            .any(|suffix| path.ends_with(suffix))
    });
    let paths: BTreeSet<&str> = leaves.iter().chain(&opaque).map(String::as_str).collect();
    let pinned: BTreeSet<&str> = include_str!("history_v1_paths.txt")
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect();
    let changes: Vec<String> = paths
        .difference(&pinned)
        .map(|path| format!("+ {path}"))
        .chain(pinned.difference(&paths).map(|path| format!("- {path}")))
        .collect();
    assert!(
        changes.is_empty(),
        "the persisted `History` format changed. After 0.1 every format change, \
         additive ones included, bumps `HISTORY_FORMAT_VERSION` (now {HISTORY_FORMAT_VERSION}) \
         and ships only in a minor release (\"Format evolution\" in the \
         `allama::history` docs). Add the change to the golden fixture \
         (tests/fixtures/history_v1.json) and update \
         crates/allama/src/chat/history_v1_paths.txt:\n  {}",
        changes.join("\n  ")
    );
}

#[test]
fn every_entry_is_probed_or_exempt_with_a_reason() {
    for Entry { path, account } in entries() {
        if let Some(probes) = account.strip_prefix("probe:") {
            let probes: Vec<&str> = probes.split(',').map(str::trim).collect();
            assert!(
                probes.iter().all(
                    |p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_lowercase() || b == b'_')
                ),
                "`{path}`: probe names are snake_case, comma-separated: {account}"
            );
        } else if let Some(reason) = account.strip_prefix("exempt:") {
            assert!(
                !reason.trim().is_empty(),
                "`{path}`: an exemption needs a reason"
            );
        } else {
            panic!("`{path}`: the account must start with `probe:` or `exempt:`: {account}");
        }
    }
}
