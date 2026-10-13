//! Maps Anthropic responses to portable [`ChatResponse`] values.
//!
//! Server and MCP tool blocks become [`ContentBlock::ProviderToolCall`] and
//! [`ContentBlock::ProviderToolResult`] with origin `"anthropic"`; their wire
//! type and other fields stay in provider data for replay. The block decoder
//! is open: types the portable model does not represent (container uploads,
//! …) become [`ContentBlock::Other`] with origin `"anthropic"`, so they
//! replay verbatim to Anthropic. Thinking signatures and redacted thinking live in
//! `Reasoning.provider_data["anthropic"]`. A `tool_use` block's fields
//! beyond `id`, `name` and `input` (`caller`, a computer toolset member's
//! `toolset_name`, …) live in `ToolCall.provider_data["anthropic"]`.

use allama::{
    ChatResponse, Citation, CitationSource, ContentBlock, Error, ExtraMap, FinishReason,
    OtherContent, ProviderData, ProviderToolCall, ProviderToolResult, Reasoning, Result, Text,
    ToolCall, Usage,
};
use serde_json::Value;

use crate::wire;
use crate::{NAMESPACE, PROVIDER_NAME};

/// Decode a non-streaming Messages API response.
pub(crate) fn decode_response(
    response: wire::CreateMessageResponse,
    request_id: Option<String>,
) -> Result<ChatResponse> {
    // An empty text block carries nothing, and the API refuses it on
    // replay; the stream collector drops it too.
    let content = response
        .content
        .into_iter()
        .map(decode_block)
        .filter(|block| !matches!(block, Ok(ContentBlock::Text(text)) if is_blank(text)))
        .collect::<Result<Vec<_>>>()?;

    let mut out = ChatResponse::new(content).provider(PROVIDER_NAME);
    if let Some(raw) = response.stop_reason {
        out.finish_reason = Some(finish_reason(&raw));
        out.finish_reason_raw = Some(raw);
    }
    // `usage` is optional on the wire (some gateways leave it out): report
    // none rather than an empty one.
    out.usage = Some(normalize_usage(&response.usage)).filter(|usage| !usage.is_empty());
    out.model = response.model;
    out.id = response.id;
    out.request_id = request_id;
    out.provider_data = response_provider_data(
        response.stop_sequence,
        response.stop_details,
        response.container,
        &response.usage,
    );
    unmodeled_fields(&mut out.provider_data, response.extra);
    Ok(out)
}

/// Decode one response content block.
pub(crate) fn decode_block(value: Value) -> Result<ContentBlock> {
    let Value::Object(mut map) = value else {
        return Err(unexpected("content block is not an object"));
    };
    let type_name = match map.remove("type") {
        Some(Value::String(t)) => t,
        _ => return Err(unexpected("content block has no string `type`")),
    };
    Ok(match type_name.as_str() {
        "text" => {
            let text = take_string(&mut map, "text")
                .ok_or_else(|| unexpected("text block has no `text`"))?;
            let mut provider_data = ProviderData::new();
            let mut portable = Vec::new();
            if let Some(citations) = map.remove("citations").filter(|c| !c.is_null()) {
                portable = citations
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(citation)
                    .collect();
                provider_data.insert_raw(NAMESPACE, "citations", citations);
            }
            ContentBlock::Text(
                Text::new(text)
                    .with_citations(portable)
                    .with_provider_data(provider_data),
            )
        }
        "thinking" => {
            let thinking = take_string(&mut map, "thinking").unwrap_or_default();
            ContentBlock::Reasoning(
                Reasoning::new(thinking)
                    .with_provider_data(signature_data(take_string(&mut map, "signature"))),
            )
        }
        "redacted_thinking" => {
            let data = take_string(&mut map, "data")
                .ok_or_else(|| unexpected("redacted_thinking block has no `data`"))?;
            ContentBlock::Reasoning(Reasoning::new("").with_provider_data(redacted_data(data)))
        }
        "tool_use" => {
            let id = take_string(&mut map, "id")
                .ok_or_else(|| unexpected("tool_use block has no `id`"))?;
            let name = take_string(&mut map, "name")
                .ok_or_else(|| unexpected("tool_use block has no `name`"))?;
            let input = map
                .remove("input")
                .unwrap_or_else(|| Value::Object(ExtraMap::new()));
            let arguments = serde_json::to_string(&input)?;
            ContentBlock::ToolCall(
                ToolCall::new(id, name, arguments).with_provider_data(tool_use_data(map)),
            )
        }
        kind if is_provider_tool_call(kind) => {
            let input = map
                .remove("input")
                .unwrap_or_else(|| Value::Object(ExtraMap::new()));
            ContentBlock::ProviderToolCall(provider_tool_call(type_name, map, &input)?)
        }
        kind if is_provider_tool_result(kind) => {
            ContentBlock::ProviderToolResult(provider_tool_result(type_name, map)?)
        }
        _ => ContentBlock::Other(OtherContent::new(type_name, map).with_origin(NAMESPACE)),
    })
}

/// Blocks for tools Anthropic runs itself: server tools (web search, web
/// fetch, code execution, tool search) and the MCP connector.
pub(crate) fn is_provider_tool_call(type_name: &str) -> bool {
    matches!(type_name, "server_tool_use" | "mcp_tool_use")
}

/// Results of provider-run tools: `web_search_tool_result`,
/// `code_execution_tool_result`, `mcp_tool_result`, and so on.
pub(crate) fn is_provider_tool_result(type_name: &str) -> bool {
    type_name.ends_with("_tool_result") && type_name != "tool_result"
}

/// A server or MCP tool call. The wire `type` and fields without a portable
/// meaning (`server_name`, `caller`, …) are kept in provider data for replay.
pub(crate) fn provider_tool_call(
    type_name: String,
    mut map: ExtraMap,
    input: &Value,
) -> Result<ProviderToolCall> {
    let id = take_string(&mut map, "id")
        .ok_or_else(|| unexpected(&format!("{type_name} block has no `id`")))?;
    let name = take_string(&mut map, "name")
        .ok_or_else(|| unexpected(&format!("{type_name} block has no `name`")))?;
    let arguments = serde_json::to_string(input)?;
    Ok(ProviderToolCall::new(id, name, arguments)
        .with_origin(NAMESPACE)
        .with_provider_data(wire_fields(type_name, map)))
}

/// A server or MCP tool result. `content` is the output, verbatim; error
/// results (an `*_error` content object, or `is_error` on MCP results) set
/// the error flag.
pub(crate) fn provider_tool_result(
    type_name: String,
    mut map: ExtraMap,
) -> Result<ProviderToolResult> {
    let call_id = take_string(&mut map, "tool_use_id")
        .ok_or_else(|| unexpected(&format!("{type_name} block has no `tool_use_id`")))?;
    let output = map.remove("content").unwrap_or(Value::Null);
    let is_error = map
        .get("is_error")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || output
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|t| t.ends_with("_error"));
    Ok(ProviderToolResult::new(call_id, output)
        .with_error(is_error)
        .with_origin(NAMESPACE)
        .with_provider_data(wire_fields(type_name, map)))
}

fn wire_fields(type_name: String, map: ExtraMap) -> ProviderData {
    let mut data = ProviderData::new().with_raw(NAMESPACE, "type", Value::String(type_name));
    for (key, value) in map {
        data.insert_raw(NAMESPACE, key, value);
    }
    data
}

/// Map an Anthropic `stop_reason`.
pub(crate) fn finish_reason(raw: &str) -> FinishReason {
    match raw {
        "end_turn" | "stop_sequence" => FinishReason::Stop,
        "max_tokens" => FinishReason::Length,
        "model_context_window_exceeded" => FinishReason::ContextWindow,
        "tool_use" => FinishReason::ToolCalls,
        "refusal" => FinishReason::Refusal,
        other => FinishReason::Other(other.to_owned()),
    }
}

/// Normalize Anthropic usage to the portable contract: `input_tokens`
/// includes cache reads and writes; cache counters are subsets. The
/// `server_tool_use` request counts (`web_search_requests`, …) are tool
/// requests, keyed by the tool (`web_search`). Arithmetic saturates.
pub(crate) fn normalize_usage(raw: &wire::Usage) -> Usage {
    let read = raw.cache_read_input_tokens;
    let write = raw.cache_creation_input_tokens;
    let mut usage = Usage::new();
    usage.input_tokens = raw.input_tokens.map(|i| {
        i.saturating_add(read.unwrap_or(0))
            .saturating_add(write.unwrap_or(0))
    });
    usage.output_tokens = raw.output_tokens;
    usage.cache_read_input_tokens = read;
    usage.cache_write_input_tokens = write;
    usage.reasoning_tokens = raw
        .output_tokens_details
        .as_ref()
        .and_then(|d| d.thinking_tokens);
    if let Some(creation) = &raw.cache_creation {
        for (key, value) in [
            (
                "cache_write_5m_input_tokens",
                creation.ephemeral_5m_input_tokens,
            ),
            (
                "cache_write_1h_input_tokens",
                creation.ephemeral_1h_input_tokens,
            ),
        ] {
            if let Some(value) = value.filter(|v| *v > 0) {
                usage.breakdowns.insert(key.into(), value);
            }
        }
    }
    for (key, value) in raw.server_tool_use.iter().flatten() {
        if let Some(value) = value.as_u64().filter(|v| *v > 0) {
            let tool = key.strip_suffix("_requests").unwrap_or(key);
            usage.tool_requests.insert(tool.to_owned(), value);
        }
    }
    usage.with_derived_total()
}

/// Response-level Anthropic data: matched stop sequence, refusal details,
/// container, service tier, and usage fields without a portable meaning.
pub(crate) fn response_provider_data(
    stop_sequence: Option<String>,
    stop_details: Option<Value>,
    container: Option<Value>,
    usage: &wire::Usage,
) -> ProviderData {
    let mut data = ProviderData::new();
    if let Some(stop_sequence) = stop_sequence {
        data.insert_raw(NAMESPACE, "stop_sequence", Value::String(stop_sequence));
    }
    if let Some(details) = stop_details.filter(|v| !v.is_null()) {
        // A refusal's `stop_details` explains it; its text is the decline
        // explanation every adapter stores under the same key.
        if let Some(explanation) = details.get("explanation").and_then(Value::as_str) {
            data.insert_raw(
                NAMESPACE,
                allama::provider_data::DECLINE_EXPLANATION_KEY,
                Value::String(explanation.to_owned()),
            );
        }
        data.insert_raw(NAMESPACE, "stop_details", details);
    }
    if let Some(container) = container.filter(|v| !v.is_null()) {
        data.insert_raw(NAMESPACE, "container", container);
    }
    if let Some(tier) = &usage.service_tier {
        data.insert_raw(NAMESPACE, "service_tier", Value::String(tier.clone()));
    }
    if !usage.extra.is_empty() {
        data.insert_raw(NAMESPACE, "usage", Value::Object(usage.extra.clone()));
    }
    data
}

/// Keep the message fields this adapter does not model (`diagnostics`, …)
/// as response provider data; `type` and `role` are fixed and say nothing.
pub(crate) fn unmodeled_fields(data: &mut ProviderData, mut fields: ExtraMap) {
    fields.remove("type");
    fields.remove("role");
    for (key, value) in fields.into_iter().filter(|(_, v)| !v.is_null()) {
        data.insert_raw(NAMESPACE, &key, value);
    }
}

/// The portable view of one Anthropic citation; `None` for location types
/// this adapter does not know. The raw citation stays in the text block's
/// provider data (`"citations"`), which is what gets replayed.
///
/// Every citation cites the whole text block it belongs to, so there is no
/// span. Fields without a portable meaning (`type`, character, page and
/// block ranges, `encrypted_index`, …) go in the citation's provider data.
pub(crate) fn citation(raw: &Value) -> Option<Citation> {
    let mut map = raw.as_object()?.clone();
    let kind = take_string(&mut map, "type")?;
    let source = match kind.as_str() {
        "web_search_result_location" => CitationSource::url(
            take_string(&mut map, "url")?,
            take_string(&mut map, "title"),
        ),
        "char_location" | "page_location" | "content_block_location" => {
            let index = map
                .remove("document_index")
                .and_then(|i| i.as_u64())
                .and_then(|i| usize::try_from(i).ok());
            CitationSource::document(index, take_string(&mut map, "document_title"))
        }
        "search_result_location" => {
            let title = take_string(&mut map, "title");
            match take_string(&mut map, "source") {
                Some(source) if source.starts_with("http://") || source.starts_with("https://") => {
                    CitationSource::url(source, title)
                }
                source => {
                    if let Some(source) = source {
                        map.insert("source".into(), Value::String(source));
                    }
                    CitationSource::document(None, title)
                }
            }
        }
        _ => return None,
    };
    let mut citation = Citation::new(source);
    if let Some(cited_text) = take_string(&mut map, "cited_text") {
        citation = citation.with_cited_text(cited_text);
    }
    let mut provider_data = ProviderData::new().with_raw(NAMESPACE, "type", Value::String(kind));
    for (key, value) in map.into_iter().filter(|(_, v)| !v.is_null()) {
        provider_data.insert_raw(NAMESPACE, key, value);
    }
    Some(citation.with_provider_data(provider_data))
}

pub(crate) fn signature_data(signature: Option<String>) -> ProviderData {
    let mut data = ProviderData::new();
    if let Some(signature) = signature.filter(|s| !s.is_empty()) {
        data.insert_raw(NAMESPACE, "signature", Value::String(signature));
    }
    data
}

pub(crate) fn redacted_data(data: String) -> ProviderData {
    ProviderData::new().with_raw(NAMESPACE, "redacted_data", Value::String(data))
}

/// The `tool_use` fields left after `type`, `id`, `name` and `input`, kept
/// for replay: a programmatic call's `caller` (a direct call needs no
/// state), a computer toolset member's `toolset_name`, and any key Anthropic
/// adds later. The encoder sends them back on the `tool_use` block and
/// repeats `toolset_name` on the matching `tool_result`.
pub(crate) fn tool_use_data(fields: ExtraMap) -> ProviderData {
    let mut data = ProviderData::new();
    for (key, value) in fields {
        let direct = key == "caller" && value.get("type").and_then(Value::as_str) == Some("direct");
        if !value.is_null() && !direct {
            data.insert_raw(NAMESPACE, key, value);
        }
    }
    data
}

/// A text block with nothing in it, which the stream collector drops too.
fn is_blank(text: &Text) -> bool {
    text.text.is_empty() && text.citations.is_empty() && text.provider_data.is_empty()
}

fn take_string(map: &mut ExtraMap, key: &str) -> Option<String> {
    match map.remove(key) {
        Some(Value::String(s)) => Some(s),
        _ => None,
    }
}

fn unexpected(message: &str) -> Error {
    Error::unexpected_response(format!("anthropic: {message}")).with_provider(PROVIDER_NAME)
}

#[cfg(test)]
pub(crate) fn conformance_from_api_response_value(value: Value) -> Result<ChatResponse> {
    let response: wire::CreateMessageResponse = serde_json::from_value(value)?;
    decode_response(response, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use allama_conformance::assert_usage_contract;
    use serde_json::json;

    fn decode(value: Value) -> ChatResponse {
        conformance_from_api_response_value(value).unwrap()
    }

    /// A unary response keeps no empty text, as the stream
    /// collector does, so replaying it never sends one.
    #[test]
    fn empty_text_is_dropped_as_the_stream_collector_does() {
        let response = decode(json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "content": [
                {"type": "text", "text": ""},
                {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }));
        assert_eq!(response.content.len(), 1);
        assert!(matches!(response.content[0], ContentBlock::ToolCall(_)));
    }

    /// Regression: a response without `usage`
    /// reported `Some` empty usage instead of none.
    #[test]
    fn missing_usage_is_none() {
        let response = decode(json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn"
        }));
        assert_eq!(response.usage, None);
        let response = decode(json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 1}
        }));
        assert_eq!(response.usage.unwrap().total_tokens, Some(4));
    }

    /// Response and usage fields this adapter does not model are
    /// kept, not dropped.
    #[test]
    fn unmodeled_response_and_usage_fields_are_kept() {
        let response = decode(json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "diagnostics": {"cache_miss_reason": "new_prefix"},
            "usage": {"input_tokens": 1, "output_tokens": 1, "inference_geo": "us"}
        }));
        let data = response.provider_data.namespace(NAMESPACE).unwrap();
        assert_eq!(
            data["diagnostics"],
            json!({"cache_miss_reason": "new_prefix"})
        );
        assert_eq!(data["usage"], json!({"inference_geo": "us"}));
        assert!(data.get("type").is_none() && data.get("role").is_none());
    }

    #[test]
    fn citation_locations_map_to_portable_sources() {
        let document = citation(&json!({
            "type": "page_location", "cited_text": "Revenue grew 4%.", "document_index": 1,
            "document_title": "Q3 report", "start_page_number": 2, "end_page_number": 3
        }))
        .unwrap();
        assert_eq!(
            document.source,
            CitationSource::document(Some(1), Some("Q3 report".into()))
        );
        assert_eq!(document.cited_text.as_deref(), Some("Revenue grew 4%."));
        assert_eq!(document.span, None);
        let data = document.provider_data.namespace(NAMESPACE).unwrap();
        assert_eq!(data["type"], "page_location");
        assert_eq!(data["start_page_number"], 2);

        let search = citation(&json!({
            "type": "search_result_location", "source": "https://docs.rs/serde", "title": "serde",
            "cited_text": "Serde is a framework", "search_result_index": 0,
            "start_block_index": 0, "end_block_index": 0
        }))
        .unwrap();
        assert_eq!(
            search.source,
            CitationSource::url("https://docs.rs/serde", Some("serde".into()))
        );
        let internal = citation(&json!({
            "type": "search_result_location", "source": "kb://42", "title": "Runbook",
            "cited_text": "Restart it."
        }))
        .unwrap();
        assert_eq!(
            internal.source,
            CitationSource::document(None, Some("Runbook".into()))
        );
        assert_eq!(
            internal.provider_data.get_raw(NAMESPACE, "source"),
            Some(&json!("kb://42"))
        );

        assert!(citation(&json!({"type": "future_location"})).is_none());
        assert!(citation(&json!({"type": "web_search_result_location"})).is_none());
    }

    #[test]
    fn decodes_text_thinking_and_tool_use() {
        let response = decode(json!({
            "id": "msg_1",
            "model": "claude-sonnet-4-5",
            "stop_reason": "tool_use",
            "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "sig_1"},
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "rust"}, "caller": {"type": "direct"}}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }));
        assert_eq!(response.provider.as_deref(), Some("anthropic"));
        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(response.finish_reason_raw.as_deref(), Some("tool_use"));
        let reasoning = response.content[0].as_reasoning().unwrap();
        assert_eq!(reasoning.text, "hmm");
        assert_eq!(
            reasoning.provider_data.get_raw("anthropic", "signature"),
            Some(&json!("sig_1"))
        );
        let call = response.first_tool_call().unwrap();
        assert_eq!(call.arguments, r#"{"q":"rust"}"#);
        assert!(
            call.provider_data.is_empty(),
            "direct caller carries no state"
        );
    }

    #[test]
    fn tool_use_keeps_unknown_fields_for_replay() {
        let response = decode(json!({
            "id": "msg_1",
            "model": "claude-opus-5-5",
            "stop_reason": "tool_use",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "left_click", "toolset_name": "computer", "input": {"coordinate": [512, 742]}},
                {"type": "tool_use", "id": "toolu_2", "name": "run", "input": {}, "caller": {"type": "code_execution_20260120", "tool_id": "srvtoolu_1"}, "future_field": [1], "absent": null}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }));
        let calls: Vec<_> = response.tool_calls().collect();
        assert_eq!(calls[0].name, "left_click");
        assert_eq!(calls[0].arguments, r#"{"coordinate":[512,742]}"#);
        assert_eq!(
            calls[0].provider_data,
            ProviderData::new().with_raw(NAMESPACE, "toolset_name", json!("computer"))
        );
        assert_eq!(
            calls[1].provider_data,
            ProviderData::new()
                .with_raw(
                    NAMESPACE,
                    "caller",
                    json!({"type": "code_execution_20260120", "tool_id": "srvtoolu_1"})
                )
                .with_raw(NAMESPACE, "future_field", json!([1])),
            "unknown fields are kept; null ones carry nothing"
        );
    }

    #[test]
    fn server_tool_blocks_and_unknown_blocks() {
        let response = decode(json!({
            "id": "msg_1",
            "model": "claude-sonnet-4-5",
            "stop_reason": "end_turn",
            "content": [
                {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "rust"}},
                {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [{"type": "web_search_result", "url": "https://rust-lang.org", "title": "Rust", "encrypted_content": "enc"}]},
                {"type": "container_upload", "file_id": "file_1"},
                {"type": "text", "text": "Rust is a language.", "citations": [{"type": "web_search_result_location", "url": "https://rust-lang.org", "cited_text": "Rust"}]}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "server_tool_use": {"web_search_requests": 1}}
        }));
        let ContentBlock::ProviderToolCall(call) = &response.content[0] else {
            panic!(
                "expected a provider tool call, got {:?}",
                response.content[0]
            );
        };
        assert_eq!(
            (call.id.as_str(), call.name.as_str()),
            ("srvtoolu_1", "web_search")
        );
        assert_eq!(call.origin.as_deref(), Some("anthropic"));
        assert_eq!(call.arguments, r#"{"query":"rust"}"#);
        assert_eq!(
            call.provider_data.get_raw("anthropic", "type"),
            Some(&json!("server_tool_use"))
        );
        let ContentBlock::ProviderToolResult(result) = &response.content[1] else {
            panic!(
                "expected a provider tool result, got {:?}",
                response.content[1]
            );
        };
        assert_eq!(result.call_id, "srvtoolu_1");
        assert!(!result.is_error);
        assert_eq!(result.output[0]["url"], "https://rust-lang.org");
        assert!(
            response.tool_calls().next().is_none(),
            "nothing for the caller to run"
        );
        assert!(
            matches!(&response.content[2], ContentBlock::Other(o) if o.type_name == "container_upload")
        );
        let ContentBlock::Text(text) = &response.content[3] else {
            panic!("expected text");
        };
        assert_eq!(
            text.provider_data
                .get_raw("anthropic", "citations")
                .unwrap()[0]["cited_text"],
            "Rust"
        );
        assert_eq!(
            response
                .usage
                .as_ref()
                .unwrap()
                .tool_requests
                .get("web_search"),
            Some(&1)
        );
    }

    #[test]
    fn redacted_thinking_is_reasoning_with_empty_text() {
        let response = decode(json!({
            "id": "msg_1",
            "model": "claude-sonnet-4-5",
            "stop_reason": "end_turn",
            "content": [
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "text", "text": "ok"}
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }));
        let reasoning = response.content[0].as_reasoning().unwrap();
        assert!(reasoning.is_opaque());
        assert_eq!(
            reasoning
                .provider_data
                .get_raw("anthropic", "redacted_data"),
            Some(&json!("opaque"))
        );
    }

    #[test]
    fn malformed_known_block_is_unexpected_response() {
        let err = conformance_from_api_response_value(json!({
            "content": [{"type": "text"}],
            "usage": {}
        }))
        .unwrap_err();
        assert!(matches!(err, Error::UnexpectedResponse { .. }), "{err:?}");
    }

    #[test]
    fn stop_reasons() {
        assert_eq!(finish_reason("end_turn"), FinishReason::Stop);
        assert_eq!(finish_reason("stop_sequence"), FinishReason::Stop);
        assert_eq!(finish_reason("max_tokens"), FinishReason::Length);
        // The context window is not the output cap.
        assert_eq!(
            finish_reason("model_context_window_exceeded"),
            FinishReason::ContextWindow
        );
        assert!(!FinishReason::ContextWindow.finished_normally());
        assert_eq!(finish_reason("tool_use"), FinishReason::ToolCalls);
        assert_eq!(finish_reason("refusal"), FinishReason::Refusal);
        assert_eq!(
            finish_reason("pause_turn"),
            FinishReason::Other("pause_turn".into())
        );
    }

    #[test]
    fn stop_sequence_and_refusal_details_are_kept() {
        let response = decode(json!({
            "id": "msg_1",
            "model": "claude-sonnet-4-5",
            "stop_reason": "stop_sequence",
            "stop_sequence": "END",
            "content": [{"type": "text", "text": "done"}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "service_tier": "standard"}
        }));
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(response.finish_reason_raw.as_deref(), Some("stop_sequence"));
        assert_eq!(
            response.provider_data.get_raw("anthropic", "stop_sequence"),
            Some(&json!("END"))
        );

        let refused = decode(json!({
            "id": "msg_2",
            "model": "claude-opus-5-5",
            "stop_reason": "refusal",
            "stop_details": {"type": "refusal", "category": "cyber", "explanation": "no"},
            "content": [],
            "usage": {"input_tokens": 1, "output_tokens": 0}
        }));
        assert_eq!(refused.finish_reason, Some(FinishReason::Refusal));
        assert_eq!(
            refused
                .provider_data
                .get_raw("anthropic", "stop_details")
                .unwrap()["category"],
            "cyber"
        );
    }

    #[test]
    fn usage_is_normalized_to_the_contract() {
        let usage = normalize_usage(&serde_json::from_value(json!({
            "input_tokens": 12,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 50,
            "cache_creation": {"ephemeral_5m_input_tokens": 20, "ephemeral_1h_input_tokens": 30},
            "output_tokens": 40,
            "output_tokens_details": {"thinking_tokens": 25}
        }))
        .unwrap());
        assert_eq!(usage.input_tokens, Some(162));
        assert_eq!(usage.cache_read_input_tokens, Some(100));
        assert_eq!(usage.cache_write_input_tokens, Some(50));
        assert_eq!(usage.output_tokens, Some(40));
        assert_eq!(usage.reasoning_tokens, Some(25));
        assert_eq!(usage.total_tokens, Some(202));
        assert_eq!(
            usage.breakdowns.get("cache_write_1h_input_tokens"),
            Some(&30)
        );
        assert_usage_contract(&usage);
    }
}
