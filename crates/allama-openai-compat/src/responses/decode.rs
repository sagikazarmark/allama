//! Maps Responses API output to portable [`ChatResponse`] content.
//!
//! Provider data is written under the dialect namespace
//! ([`ResponsesQuirks::namespace`](crate::ResponsesQuirks::namespace)):
//!
//! - `message` → [`Text`] per `output_text` part (URL and file citation
//!   annotations as portable citations; raw annotations and `phase` in
//!   provider data); `refusal` parts → [`Refusal`] (with the `phase`) and
//!   [`FinishReason::Refusal`].
//! - `function_call` → [`ToolCall`] with `call_id` as the id; the item id is
//!   kept as `item_id` for replay.
//! - `reasoning` → [`Reasoning`] with the summary text (or the raw reasoning
//!   content when there is no summary); `encrypted_content`, raw `content`,
//!   and the item id go in provider data for stateless replay.
//! - Hosted tool items (`web_search_call`, `code_interpreter_call`,
//!   `mcp_call`, …) → a [`ProviderToolCall`](allama::ProviderToolCall) and a
//!   [`ProviderToolResult`](allama::ProviderToolResult) with the namespace as
//!   origin (see `hosted`).
//! - Anything else → [`OtherContent`] with the namespace as origin, replayed
//!   verbatim. That includes client-executed items that are not function
//!   calls (`custom_tool_call`, `local_shell_call`, …; see `client`): the
//!   finish reason is `ToolCalls` while one waits for its output.

use allama::{
    ChatResponse, ContentBlock, Error, ExtraMap, FinishReason, OtherContent, ProviderData,
    Reasoning, Refusal, Result, Subject, Text, ToolCall, Usage, Warning, WarningKind,
};
use serde_json::Value;

use super::error::map_response_error;
use super::wire::{ResponseObject, ResponseUsage};
use crate::ResponseContext;

/// Who decoded content is attributed to: the provider identity (errors,
/// warnings, `ChatResponse::provider`) and the provider-data namespace.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Origin<'a> {
    pub provider: &'a str,
    pub namespace: &'a str,
}

/// Separator between reasoning summary parts.
pub(crate) const SUMMARY_SEPARATOR: &str = "\n\n";

/// Decode a complete (non-streaming) response.
pub(crate) fn decode_response(
    wire: ResponseObject,
    context: ResponseContext,
) -> Result<ChatResponse> {
    let ResponseContext {
        provider,
        namespace,
        request_id,
        warnings,
    } = context;
    let origin = Origin {
        provider: &provider,
        namespace: &namespace,
    };
    if wire.status.as_deref() == Some("failed") {
        let error = wire.error.unwrap_or_default();
        return Err(map_response_error(&error, &provider, request_id));
    }
    let mut response_warnings = warnings;
    let mut content = Vec::new();
    for item in &wire.output {
        content.extend(decode_item(item, origin, &mut response_warnings)?);
    }
    let flags = OutputFlags::of(&content);
    let (finish_reason, finish_reason_raw) = finish_reason(
        wire.status.as_deref(),
        wire.incomplete_details
            .as_ref()
            .and_then(|d| d.reason.as_deref()),
        flags,
    );

    let mut response = ChatResponse::new(content);
    response.finish_reason = finish_reason;
    response.finish_reason_raw = finish_reason_raw;
    response.usage = usage(wire.usage.as_ref(), wire.tool_usage.as_ref());
    response.model = wire.model;
    response.id = wire.id;
    response.provider = Some(provider.clone());
    response.provider_data =
        response_provider_data(wire.service_tier, wire.tool_usage, origin.namespace);
    response.request_id = request_id;
    response.warnings = response_warnings;
    Ok(response)
}

/// What the output contains, for the finish reason.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct OutputFlags {
    pub tool_calls: bool,
    pub refused: bool,
}

impl OutputFlags {
    pub(crate) fn of(content: &[ContentBlock]) -> Self {
        Self {
            // A client-executed call that is not a function call also waits
            // for the caller (ADR 0014).
            tool_calls: content
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolCall(_)))
                || crate::client_tools::has_pending(content),
            refused: content
                .iter()
                .any(|b| matches!(b, ContentBlock::Refusal(_))),
        }
    }
}

/// Decode one output item into content blocks.
pub(crate) fn decode_item(
    item: &Value,
    origin: Origin<'_>,
    warnings: &mut Vec<Warning>,
) -> Result<Vec<ContentBlock>> {
    let ns = origin.namespace;
    let unexpected = |message: &str| {
        Error::unexpected_response(message.to_owned()).with_provider(origin.provider)
    };
    let Value::Object(item) = item else {
        return Err(unexpected("output item is not an object"));
    };
    let Some(kind) = item.get("type").and_then(Value::as_str) else {
        return Err(unexpected("output item without a `type`"));
    };
    let blocks = match kind {
        "message" => {
            let phase = item.get("phase").filter(|p| !p.is_null());
            let parts = item.get("content").and_then(Value::as_array);
            let mut blocks = Vec::new();
            for part in parts.into_iter().flatten() {
                match part.get("type").and_then(Value::as_str) {
                    Some("output_text") => {
                        let text = string(part, "text").unwrap_or_default();
                        let annotations = part.get("annotations");
                        let citations = crate::citations::citations(annotations, &text, ns);
                        blocks.push(ContentBlock::Text(
                            Text::new(text)
                                .with_citations(citations)
                                .with_provider_data(text_data(Some(part), phase, ns)),
                        ));
                    }
                    Some("refusal") => {
                        let text = string(part, "refusal").unwrap_or_default();
                        blocks.push(ContentBlock::Refusal(
                            Refusal::new(text).with_provider_data(text_data(None, phase, ns)),
                        ));
                    }
                    other => {
                        warnings.push(dropped_part(other.unwrap_or("unknown"), origin.provider));
                    }
                }
            }
            blocks
        }
        "function_call" => {
            let call_id = string_field(item, "call_id")
                .ok_or_else(|| unexpected("function_call item without `call_id`"))?;
            let name = string_field(item, "name")
                .ok_or_else(|| unexpected("function_call item without `name`"))?;
            let arguments = string_field(item, "arguments").unwrap_or_default();
            vec![ContentBlock::ToolCall(
                ToolCall::new(call_id, name, arguments)
                    .with_provider_data(item_id_data(string_field(item, "id"), ns)),
            )]
        }
        "reasoning" => {
            let (text, source) = reasoning_text(item);
            vec![ContentBlock::Reasoning(
                Reasoning::new(text).with_provider_data(reasoning_data(item, source, ns)),
            )]
        }
        other => match super::hosted::decode(item, ns) {
            Some((call, result)) => vec![call.into(), result.into()],
            None => {
                let mut data = item.clone();
                data.remove("type");
                vec![ContentBlock::Other(
                    OtherContent::new(other, data).with_origin(ns),
                )]
            }
        },
    };
    Ok(blocks)
}

/// Text and refusal provider data: the content part's non-empty
/// `annotations` and `logprobs` (sent when the request includes
/// `message.output_text.logprobs`; kept as provider data),
/// and the message `phase`.
pub(crate) fn text_data(
    part: Option<&Value>,
    phase: Option<&Value>,
    namespace: &str,
) -> ProviderData {
    let mut data = ProviderData::new();
    for key in ["annotations", "logprobs"] {
        if let Some(value) = part
            .and_then(|part| part.get(key))
            .filter(|v| v.as_array().is_some_and(|a| !a.is_empty()))
        {
            data.insert_raw(namespace, key, value.clone());
        }
    }
    if let Some(phase) = phase {
        data.insert_raw(namespace, "phase", phase.clone());
    }
    data
}

/// Provider data carrying the output item id (function calls, reasoning).
pub(crate) fn item_id_data(item_id: Option<String>, namespace: &str) -> ProviderData {
    let mut data = ProviderData::new();
    if let Some(id) = item_id {
        data.insert_raw(namespace, "item_id", Value::String(id));
    }
    data
}

/// Where a reasoning block's text came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReasoningSource {
    /// `summary` parts (OpenAI).
    Summary,
    /// Raw `content` parts (`reasoning_text`), used when there is no summary
    /// (vLLM, llama.cpp, OpenRouter with open-weight models).
    Content,
}

/// Provider-data key marking reasoning text taken from raw `content`, so a
/// replay does not send it back as a summary.
pub(crate) const TEXT_SOURCE_KEY: &str = "text_source";

/// The reasoning block text: the summary, or the raw reasoning content when
/// there is no summary.
pub(crate) fn reasoning_text(item: &ExtraMap) -> (String, ReasoningSource) {
    let summary = summary_text(item.get("summary"));
    if !summary.is_empty() {
        return (summary, ReasoningSource::Summary);
    }
    let content = content_text(item.get("content"));
    if content.is_empty() {
        (summary, ReasoningSource::Summary)
    } else {
        (content, ReasoningSource::Content)
    }
}

/// Reasoning replay state: item id, `encrypted_content`, raw reasoning
/// `content` (returned by some models instead of summaries), and where the
/// text came from.
pub(crate) fn reasoning_data(
    item: &ExtraMap,
    source: ReasoningSource,
    namespace: &str,
) -> ProviderData {
    let mut data = ProviderData::new();
    if source == ReasoningSource::Content {
        data.insert_raw(namespace, TEXT_SOURCE_KEY, Value::String("content".into()));
    }
    if let Some(id) = string_field(item, "id") {
        data.insert_raw(namespace, "item_id", Value::String(id));
    }
    if let Some(encrypted) = string_field(item, "encrypted_content").filter(|e| !e.is_empty()) {
        data.insert_raw(namespace, "encrypted_content", Value::String(encrypted));
    }
    if let Some(content) = item
        .get("content")
        .filter(|c| c.as_array().is_some_and(|c| !c.is_empty()))
    {
        data.insert_raw(namespace, "content", content.clone());
    }
    data
}

/// Raw reasoning parts (`reasoning_text`, or `output_text` from LiteLLM) joined
/// with [`SUMMARY_SEPARATOR`].
fn content_text(content: Option<&Value>) -> String {
    content
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|part| {
            part.get("type")
                .and_then(Value::as_str)
                // LiteLLM sends raw reasoning as `output_text` parts.
                .is_none_or(|t| t == "reasoning_text" || t == "output_text")
        })
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(SUMMARY_SEPARATOR)
}

/// Summary parts joined with [`SUMMARY_SEPARATOR`].
pub(crate) fn summary_text(summary: Option<&Value>) -> String {
    summary
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join(SUMMARY_SEPARATOR)
}

/// Response provider data: the service tier, and the hosted tool usage
/// when it counts anything (image generation tokens are billed apart from
/// the response's tokens, so they are not usage breakdowns).
pub(crate) fn response_provider_data(
    service_tier: Option<String>,
    tool_usage: Option<Value>,
    namespace: &str,
) -> ProviderData {
    let mut data = ProviderData::new();
    if let Some(tier) = service_tier {
        data.insert_raw(namespace, "service_tier", Value::String(tier));
    }
    if let Some(tool_usage) = tool_usage.filter(counts_anything) {
        data.insert_raw(namespace, "tool_usage", tool_usage);
    }
    data
}

/// Whether a usage object holds a non-zero number anywhere.
fn counts_anything(value: &Value) -> bool {
    match value {
        Value::Number(number) => number.as_f64().is_some_and(|n| n != 0.0),
        Value::Object(object) => object.values().any(counts_anything),
        Value::Array(items) => items.iter().any(counts_anything),
        _ => false,
    }
}

/// The response's usage: the token counts, and the billable requests of
/// hosted tools (`tool_usage.<tool>.num_requests`) as tool requests. `None`
/// when neither is reported.
pub(crate) fn usage(usage: Option<&ResponseUsage>, tool_usage: Option<&Value>) -> Option<Usage> {
    let requests: Vec<(String, u64)> = tool_usage
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
        .filter_map(|(tool, counters)| {
            let requests = counters.get("num_requests")?.as_u64()?;
            (requests > 0).then(|| (tool.clone(), requests))
        })
        .collect();
    let mut normalized = match usage {
        Some(usage) => normalize_usage(usage),
        None if requests.is_empty() => return None,
        None => Usage::new(),
    };
    for (tool, requests) in requests {
        normalized = normalized.tool_requests(tool, requests);
    }
    Some(normalized)
}

/// Normalized finish reason and the raw value: the `incomplete_details`
/// reason for incomplete responses, otherwise the response `status`.
pub(crate) fn finish_reason(
    status: Option<&str>,
    incomplete_reason: Option<&str>,
    flags: OutputFlags,
) -> (Option<FinishReason>, Option<String>) {
    let raw = match status {
        Some("incomplete") => incomplete_reason.or(status),
        other => other,
    };
    let reason = if flags.refused {
        Some(FinishReason::Refusal)
    } else {
        match status {
            None => None,
            Some("completed") if flags.tool_calls => Some(FinishReason::ToolCalls),
            Some("completed") => Some(FinishReason::Stop),
            // A background response that has not finished yet.
            Some("queued" | "in_progress") => None,
            Some("incomplete") => Some(match incomplete_reason {
                Some("max_output_tokens") => FinishReason::Length,
                Some("content_filter") => FinishReason::ContentFilter,
                other => FinishReason::Other(other.unwrap_or("incomplete").to_owned()),
            }),
            Some(other) => Some(FinishReason::Other(other.to_owned())),
        }
    };
    (reason, raw.map(str::to_owned))
}

/// Map usage to the portable contract. The Responses API already reports
/// totals that include cached input and reasoning output.
pub(crate) fn normalize_usage(usage: &ResponseUsage) -> Usage {
    let input = usage.input_tokens_details.clone().unwrap_or_default();
    let output = usage.output_tokens_details.clone().unwrap_or_default();
    let mut normalized = Usage::new()
        .input_tokens(usage.input_tokens)
        .output_tokens(usage.output_tokens);
    normalized.total_tokens = Some(
        usage
            .total_tokens
            .unwrap_or(usage.input_tokens.saturating_add(usage.output_tokens)),
    );
    normalized.cache_read_input_tokens = input.cached_tokens;
    normalized.cache_write_input_tokens = input.cache_write_tokens;
    normalized.reasoning_tokens = output.reasoning_tokens;
    normalized
}

fn dropped_part(kind: &str, provider: &str) -> Warning {
    Warning::new(
        WarningKind::Dropped,
        Subject::Response,
        format!("the adapter cannot decode message content part `{kind}`; it was dropped"),
    )
    .with_provider(provider.to_owned())
}

fn string(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(str::to_owned)
}

fn string_field(map: &ExtraMap, key: &str) -> Option<String> {
    map.get(key).and_then(Value::as_str).map(str::to_owned)
}

#[cfg(test)]
pub(crate) const TEST_NAMESPACE: &str = "openai-compat:test";

/// The origin the codec unit tests decode and encode under.
#[cfg(test)]
pub(crate) const TEST_ORIGIN: Origin<'static> = Origin {
    provider: "test",
    namespace: TEST_NAMESPACE,
};

#[cfg(test)]
pub(crate) fn test_context(request_id: Option<&str>) -> ResponseContext {
    ResponseContext {
        provider: TEST_ORIGIN.provider.into(),
        namespace: TEST_NAMESPACE.into(),
        request_id: request_id.map(str::to_owned),
        warnings: Vec::new(),
    }
}

#[cfg(test)]
pub(crate) fn decode_value(raw: Value, request_id: Option<&str>) -> Result<ChatResponse> {
    let wire: ResponseObject = serde_json::from_value(raw).map_err(Error::serialization)?;
    decode_response(wire, test_context(request_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Regression: `input + output` overflowed when
    /// no total was reported.
    #[test]
    fn absurd_usage_counters_saturate() {
        let usage: ResponseUsage = serde_json::from_value(json!({
            "input_tokens": u64::MAX,
            "output_tokens": 5
        }))
        .unwrap();
        assert_eq!(normalize_usage(&usage).total_tokens, Some(u64::MAX));
    }

    #[test]
    fn finish_reasons() {
        let none = OutputFlags::default();
        let tools = OutputFlags {
            tool_calls: true,
            ..OutputFlags::default()
        };
        let refused = OutputFlags {
            refused: true,
            ..OutputFlags::default()
        };
        assert_eq!(
            finish_reason(Some("completed"), None, none),
            (Some(FinishReason::Stop), Some("completed".into()))
        );
        assert_eq!(
            finish_reason(Some("completed"), None, tools).0,
            Some(FinishReason::ToolCalls)
        );
        assert_eq!(
            finish_reason(Some("incomplete"), Some("max_output_tokens"), none),
            (Some(FinishReason::Length), Some("max_output_tokens".into()))
        );
        assert_eq!(
            finish_reason(Some("incomplete"), Some("content_filter"), none).0,
            Some(FinishReason::ContentFilter)
        );
        assert_eq!(
            finish_reason(Some("completed"), None, refused).0,
            Some(FinishReason::Refusal)
        );
        assert_eq!(
            finish_reason(Some("cancelled"), None, none).0,
            Some(FinishReason::Other("cancelled".into()))
        );
    }

    /// Regression: `output_text.logprobs` was dropped silently. It stays
    /// provider data, and the
    /// empty array sent when logprobs were not requested is left out.
    #[test]
    fn output_text_logprobs_are_provider_data() {
        let logprobs =
            json!([{"token": "Hi", "logprob": -0.01, "bytes": [72, 105], "top_logprobs": []}]);
        let response = decode_value(
            json!({
                "id": "resp_1", "status": "completed", "model": "gpt-5-mini",
                "output": [
                    {"type": "message", "id": "msg_1", "role": "assistant", "content": [
                        {"type": "output_text", "text": "Hi", "annotations": [], "logprobs": logprobs},
                        {"type": "output_text", "text": "!", "annotations": [], "logprobs": []}
                    ]}
                ]
            }),
            None,
        )
        .unwrap();
        let [ContentBlock::Text(first), ContentBlock::Text(second)] = response.content.as_slice()
        else {
            panic!("expected two text blocks: {:?}", response.content)
        };
        assert_eq!(
            first.provider_data.get_raw(TEST_NAMESPACE, "logprobs"),
            Some(&logprobs)
        );
        assert!(
            second.provider_data.is_empty(),
            "{:?}",
            second.provider_data
        );
    }

    #[test]
    fn refusal_annotations_phase_and_unknown_parts() {
        let response = decode_value(
            json!({
                "id": "resp_1", "status": "completed", "model": "gpt-5-mini",
                "output": [
                    {"type": "message", "id": "msg_1", "role": "assistant", "phase": "final_answer", "content": [
                        {"type": "output_text", "text": "See docs.", "annotations": [{"type": "url_citation", "url": "https://example.com"}]},
                        {"type": "refusal", "refusal": "I can't help with that."},
                        {"type": "output_audio", "data": "AAAA"}
                    ]}
                ],
                "usage": {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 4}, "output_tokens": 5, "output_tokens_details": {"reasoning_tokens": 2}, "total_tokens": 15}
            }),
            Some("req_9"),
        )
        .unwrap();
        assert_eq!(response.finish_reason, Some(FinishReason::Refusal));
        assert_eq!(response.finish_reason_raw.as_deref(), Some("completed"));
        let ContentBlock::Text(text) = &response.content[0] else {
            panic!("expected text")
        };
        assert_eq!(
            text.provider_data.get_raw(TEST_NAMESPACE, "annotations"),
            Some(&json!([{"type": "url_citation", "url": "https://example.com"}]))
        );
        assert_eq!(
            text.provider_data.get_raw(TEST_NAMESPACE, "phase"),
            Some(&json!("final_answer"))
        );
        let ContentBlock::Refusal(refusal) = &response.content[1] else {
            panic!("expected a refusal")
        };
        assert_eq!(refusal.text, "I can't help with that.");
        assert_eq!(
            refusal.provider_data.get_raw(TEST_NAMESPACE, "phase"),
            Some(&json!("final_answer"))
        );
        assert_eq!(response.first_refusal(), Some("I can't help with that."));
        assert_eq!(response.text().as_deref(), Some("See docs."));
        assert_eq!(response.warnings.len(), 1, "{:?}", response.warnings);
        assert_eq!(response.warnings[0].kind, WarningKind::Dropped);
        let usage = response.usage.unwrap();
        assert_eq!(usage.cache_read_input_tokens, Some(4));
        assert_eq!(usage.reasoning_tokens, Some(2));
        assert!(usage.contract_violation().is_none());
        assert_eq!(response.request_id.as_deref(), Some("req_9"));
        assert_eq!(response.provider.as_deref(), Some("test"));
    }

    #[test]
    fn failed_response_is_an_error() {
        let err = decode_value(
            json!({"status": "failed", "error": {"code": "server_error", "message": "boom"}, "output": []}),
            Some("req_f"),
        )
        .unwrap_err();
        assert!(matches!(err, Error::Provider { .. }));
        assert_eq!(err.request_id(), Some("req_f"));
    }

    #[test]
    fn malformed_items_fail() {
        assert!(
            decode_value(
                json!({"status": "completed", "output": [{"id": "x"}]}),
                None
            )
            .is_err()
        );
        assert!(
            decode_value(
                json!({"status": "completed", "output": [{"type": "function_call", "name": "f"}]}),
                None
            )
            .is_err()
        );
    }

    #[test]
    fn raw_reasoning_content_is_the_text_when_there_is_no_summary() {
        let decode = |item: serde_json::Value| {
            let mut warnings = Vec::new();
            let blocks = decode_item(&item, TEST_ORIGIN, &mut warnings).unwrap();
            let ContentBlock::Reasoning(reasoning) = blocks.into_iter().next().unwrap() else {
                panic!("expected reasoning");
            };
            reasoning
        };
        let raw = decode(
            serde_json::json!({"type": "reasoning", "id": "rs_1", "summary": [],
            "encrypted_content": "",
            "content": [{"type": "reasoning_text", "text": "a"}, {"type": "reasoning_text", "text": "b"}]}),
        );
        assert_eq!(raw.text, "a\n\nb");
        assert_eq!(
            raw.provider_data.get_raw(TEST_NAMESPACE, TEXT_SOURCE_KEY),
            Some(&serde_json::json!("content"))
        );
        assert!(
            raw.provider_data
                .get_raw(TEST_NAMESPACE, "encrypted_content")
                .is_none()
        );

        let summarized = decode(serde_json::json!({"type": "reasoning", "id": "rs_2",
            "summary": [{"type": "summary_text", "text": "s"}],
            "content": [{"type": "reasoning_text", "text": "raw"}]}));
        assert_eq!(summarized.text, "s");
        assert!(
            summarized
                .provider_data
                .get_raw(TEST_NAMESPACE, TEXT_SOURCE_KEY)
                .is_none()
        );
        assert!(
            summarized
                .provider_data
                .get_raw(TEST_NAMESPACE, "content")
                .is_some()
        );
    }
}
