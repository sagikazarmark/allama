//! Decoding contracts: what an adapter makes of the responses its API sends.
//!
//! The probe-based contracts answer every request with the same canned
//! text, so they never exercise tool-call or length decoding, and never
//! compare the two call paths. These contracts decode captured responses
//! instead, through the adapter's real decoders:
//!
//! - [`assert_unary_stream_agree`]: a response decoded from `chat` and the
//!   same response collected from `chat_stream` agree on content, finish
//!   reason, usage, ids, provider and provider data;
//! - [`assert_provider_stamped`]: both paths stamp
//!   [`ChatResponse::provider`];
//! - [`assert_tool_call_ids_round_trip`]: tool-call ids are kept as the wire
//!   sent them (or synthesized uniquely from `[A-Za-z0-9_-]`), finish with
//!   [`FinishReason::ToolCalls`], and reach the next request;
//! - [`assert_length_finish`]: a response cut off at the output cap finishes
//!   with [`FinishReason::Length`] on both paths;
//! - [`assert_vendor_fields_kept`] (provisional, outside the aggregate): a
//!   field the adapter does not model is kept at every level of a `chat`
//!   answer, not only at the top.
//!
//! Each takes a [`ResponseCapture`]: the body the API sends for `chat` and
//! the body it sends for `chat_stream`, for the same response (ideally
//! recorded), and two replays that decode a body as the adapter does: one
//! through `chat`, one through `chat_stream`. The aggregate
//! [`assert_chat_adapter`](crate::adapter::assert_chat_adapter) runs all but
//! [`assert_vendor_fields_kept`] from
//! [`unary_replay`](crate::adapter::AdapterUnderTest::unary_replay),
//! [`stream_replay`](crate::adapter::AdapterUnderTest::stream_replay) and the
//! capture setters.

use std::collections::HashSet;
use std::future::Future;

use allama::{
    ChatRequest, ChatResponse, ChatStreamExt, FinishReason, Message, Result, StreamEvent,
};
use futures_core::Stream;
use futures_util::StreamExt;
use serde_json::Value;

use crate::probe::ProbeOutcome;

/// A response as the adapter's API sends it: the body of a `chat` answer
/// (`unary`) and the body of a `chat_stream` answer (`stream`), for the same
/// response, so the two decode to the same [`ChatResponse`].
///
/// For [`assert_tool_call_ids_round_trip`], name the tool-call ids the wire
/// carries with [`wire_tool_call_ids`](Self::wire_tool_call_ids); leave them
/// out when the API sends none and the adapter synthesizes them.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ResponseCapture {
    /// The body of the `chat` answer.
    pub unary: String,
    /// The body of the `chat_stream` answer, in its wire framing.
    pub stream: String,
    /// The tool-call ids the wire carries, in order; `None` when the API
    /// sends none.
    pub wire_tool_call_ids: Option<Vec<String>>,
}

impl ResponseCapture {
    /// A capture of one response through both call paths.
    #[must_use]
    pub fn new(unary: impl Into<String>, stream: impl Into<String>) -> Self {
        Self {
            unary: unary.into(),
            stream: stream.into(),
            wire_tool_call_ids: None,
        }
    }

    /// The tool-call ids the wire carries, in order: decoded tool calls
    /// must keep them exactly.
    #[must_use]
    pub fn wire_tool_call_ids<S: Into<String>>(mut self, ids: impl IntoIterator<Item = S>) -> Self {
        self.wire_tool_call_ids = Some(ids.into_iter().map(Into::into).collect());
        self
    }
}

/// Assert that `capture` decodes to the same response through `chat` and
/// through `chat_stream`: the same content (blocks, their provider data and
/// citations), finish reason, usage, provider and response-level provider
/// data, and the same response id and model where both paths report one.
/// Returns both responses, `chat`'s first.
///
/// `unary` decodes a body through `chat` (a transport that answers `200`
/// with it); `stream` through `chat_stream`.
///
/// # Panics
///
/// When either path fails, or the responses differ.
pub async fn assert_unary_stream_agree<U, UFut, S, SFut, St>(
    capture: &ResponseCapture,
    unary: U,
    stream: S,
) -> (ChatResponse, ChatResponse)
where
    U: Fn(String) -> UFut,
    UFut: Future<Output = Result<ChatResponse>>,
    S: Fn(String) -> SFut,
    SFut: Future<Output = Result<St>>,
    St: Stream<Item = Result<StreamEvent>> + Send,
{
    let (from_chat, from_stream) = decode(capture, &unary, &stream, "unary_stream_agree").await;
    let differences = differences(&from_chat, &from_stream);
    assert!(
        differences.is_empty(),
        "unary_stream_agree: the same response decodes differently from `chat` and from \
         `chat_stream`; differing: {}\n  chat:        {from_chat:?}\n  chat_stream: {from_stream:?}",
        differences.join(", ")
    );
    (from_chat, from_stream)
}

/// Assert that `chat` and `chat_stream` stamp the serving provider:
/// [`ChatResponse::provider`] is `provider` on both paths, for a plain
/// request to `model`.
///
/// # Panics
///
/// When either path fails, or leaves the stamp out or names another
/// provider.
pub async fn assert_provider_stamped<F, Fut, G, GFut>(
    provider: &str,
    model: &str,
    probe: F,
    stream_probe: G,
) where
    F: Fn(ChatRequest) -> Fut,
    Fut: Future<Output = ProbeOutcome<ChatResponse>>,
    G: Fn(ChatRequest) -> GFut,
    GFut: Future<Output = ProbeOutcome<ChatResponse>>,
{
    let request = ChatRequest::new(model).user("Hi");
    for (path, outcome) in [
        ("chat", probe(request.clone()).await),
        ("chat_stream", stream_probe(request).await),
    ] {
        let response = outcome
            .result
            .unwrap_or_else(|err| panic!("provider_stamped: {path} failed: {err:?}"));
        assert_eq!(
            response.provider.as_deref(),
            Some(provider),
            "provider_stamped: {path} must stamp the serving provider on the response \
             (`MetadataBuilder::with_provider`, or let `Prepared::finish` / `finish_stream` \
             stamp it)"
        );
    }
}

/// Assert that tool calls decode with ids that survive a round trip.
///
/// `capture` is a response with tool calls (ideally two parallel calls).
/// On both paths:
///
/// - the response has tool calls and finishes with
///   [`FinishReason::ToolCalls`];
/// - every id is non-empty and unique within the response, and both paths
///   decode the same ids;
/// - with [`wire_tool_call_ids`](ResponseCapture::wire_tool_call_ids), the
///   ids are exactly the wire's (never replaced by `call_0` and the like);
///   without them, the synthesized ids use only `[A-Za-z0-9_-]`, which
///   every provider accepts, so the conversation can continue elsewhere
///   (Anthropic refuses any other character);
/// - the response, replayed as the assistant turn of the next request with
///   a result for each call, is sent through `probe` without error, and
///   with wire ids every id reaches the request body.
///
/// How an adapter makes up an id the wire lacks is its own: allama has no
/// shared helper, because the format is wire knowledge (an adapter may
/// recognize its made-up ids when they come back, to send the API none).
/// Scope it by the response id, or a random one, and keep to `[A-Za-z0-9_-]`.
/// The other direction, an id from another provider that this wire refuses,
/// is the encoder's to map, with
/// [`ToolCallIdFormat`](allama::adapter::chat::ToolCallIdFormat); the
/// capture's own ids must still reach the next request unchanged.
///
/// # Panics
///
/// When any of the above fails.
pub async fn assert_tool_call_ids_round_trip<U, UFut, S, SFut, St, F, Fut>(
    model: &str,
    capture: &ResponseCapture,
    unary: U,
    stream: S,
    probe: F,
) where
    U: Fn(String) -> UFut,
    UFut: Future<Output = Result<ChatResponse>>,
    S: Fn(String) -> SFut,
    SFut: Future<Output = Result<St>>,
    St: Stream<Item = Result<StreamEvent>> + Send,
    F: Fn(ChatRequest) -> Fut,
    Fut: Future<Output = ProbeOutcome<ChatResponse>>,
{
    let (from_chat, from_stream) = decode(capture, &unary, &stream, "tool_call_ids").await;
    let mut decoded = Vec::new();
    for (path, response) in [("chat", &from_chat), ("chat_stream", &from_stream)] {
        let ids: Vec<&str> = response.tool_calls().map(|call| call.id.as_str()).collect();
        assert!(
            !ids.is_empty(),
            "tool_call_ids: {path} decoded no tool call from the capture: {response:?}"
        );
        assert_eq!(
            response.finish_reason,
            Some(FinishReason::ToolCalls),
            "tool_call_ids: {path}: a response with tool calls finishes with ToolCalls"
        );
        assert!(
            ids.iter().all(|id| !id.trim().is_empty()),
            "tool_call_ids: {path} decoded an empty tool-call id: {ids:?}"
        );
        let unique: HashSet<&&str> = ids.iter().collect();
        assert_eq!(
            unique.len(),
            ids.len(),
            "tool_call_ids: {path} decoded duplicate tool-call ids {ids:?}; history validation \
             rejects them (`DuplicateToolCallId`)"
        );
        if let Some(wire) = &capture.wire_tool_call_ids {
            assert_eq!(
                ids, *wire,
                "tool_call_ids: {path} must keep the ids the wire carries"
            );
        } else if let Some(id) = ids.iter().find(|id| !portable_id(id)) {
            panic!(
                "tool_call_ids: {path} synthesized the id `{id}`; use only `[A-Za-z0-9_-]`, \
                 which every provider accepts after a failover (Anthropic refuses other \
                 characters)"
            );
        }
        decoded.push(ids);
    }
    assert_eq!(
        decoded[0], decoded[1],
        "tool_call_ids: chat and chat_stream decode different ids for the same response"
    );

    let mut request = ChatRequest::new(model)
        .user("Hi")
        .message(from_chat.to_assistant_message());
    for call in from_chat.tool_calls() {
        request = request.message(Message::tool_result(&call.id, &call.name, "ok"));
    }
    let outcome = probe(request).await;
    if let Err(err) = &outcome.result {
        panic!("tool_call_ids: replaying the tool calls with their results failed: {err:?}");
    }
    if capture.wire_tool_call_ids.is_some() {
        for id in &decoded[0] {
            let sent = outcome
                .sent_raw
                .iter()
                .any(|body| contains(body, id.as_bytes()));
            assert!(
                sent,
                "tool_call_ids: the replayed tool call and its result must carry the wire id \
                 `{id}` to the next request"
            );
        }
    }
}

/// Whether `id` uses only `[A-Za-z0-9_-]`, the characters every provider
/// accepts in a tool call id.
fn portable_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Assert that a response cut off at the output token cap (`capture`)
/// finishes with [`FinishReason::Length`] on `chat` and on `chat_stream`.
///
/// # Panics
///
/// When either path fails or reports another finish reason.
pub async fn assert_length_finish<U, UFut, S, SFut, St>(
    capture: &ResponseCapture,
    unary: U,
    stream: S,
) where
    U: Fn(String) -> UFut,
    UFut: Future<Output = Result<ChatResponse>>,
    S: Fn(String) -> SFut,
    SFut: Future<Output = Result<St>>,
    St: Stream<Item = Result<StreamEvent>> + Send,
{
    let (from_chat, from_stream) = decode(capture, &unary, &stream, "length_finish").await;
    for (path, response) in [("chat", from_chat), ("chat_stream", from_stream)] {
        assert_eq!(
            response.finish_reason,
            Some(FinishReason::Length),
            "length_finish: {path}: a response cut off at the output token cap finishes with \
             Length (raw `{:?}`)",
            response.finish_reason_raw
        );
    }
}

/// Decode `capture` through both paths, failing the contract `name` on an
/// error.
/// Assert that a vendor field the adapter does not model is kept at every
/// level of a `chat` answer, not only at the top: in a message, a choice, a
/// content block, a tool call, the usage.
///
/// The contract adds a field named [`VENDOR_FIELD`] to every JSON object
/// of `capture`'s `chat` body, each with its own marker value, decodes the
/// body once through `unary`, and asserts that every marker reaches the
/// decoded response: its provider data at any level, a warning, or the
/// content (a field inside tool-call arguments is part of the arguments).
/// The failure names the JSON pointer of each object whose field was
/// dropped.
///
/// # Provisional
///
/// No aggregate runs this contract, and no bundled adapter passes it yet.
/// Run it yourself to see what your adapter drops. What it checks
/// may change in any minor release, and nothing promises that it will join
/// [`assert_chat_adapter`](crate::adapter::assert_chat_adapter) (ADR 0017).
///
/// # Panics
///
/// When the body is not JSON, `chat` fails to decode it, or a field is
/// dropped.
pub async fn assert_vendor_fields_kept<U, UFut>(capture: &ResponseCapture, unary: U)
where
    U: Fn(String) -> UFut,
    UFut: Future<Output = Result<ChatResponse>>,
{
    let mut body: Value = serde_json::from_str(&capture.unary)
        .unwrap_or_else(|err| panic!("vendor_fields_kept: the `chat` body is not JSON: {err}"));
    let mut fields = Vec::new();
    add_vendor_fields(&mut body, String::new(), &mut fields);
    let response = unary(body.to_string()).await.unwrap_or_else(|err| {
        panic!(
            "vendor_fields_kept: `chat` failed to decode the capture with a vendor field in each \
             object (an unknown field must not break decoding): {err:?}"
        )
    });
    let decoded = serde_json::to_string(&response).expect("serialize the decoded response");
    let dropped: Vec<&str> = fields
        .iter()
        .filter(|(_, marker)| !decoded.contains(&format!("\"{marker}\"")))
        .map(|(pointer, _)| {
            if pointer.is_empty() {
                "(the body)"
            } else {
                pointer.as_str()
            }
        })
        .collect();
    assert!(
        dropped.is_empty(),
        "vendor_fields_kept: a `{VENDOR_FIELD}` field was dropped from the objects at {}; keep \
         unmodeled fields as provider data at the level they arrive (or warn), never drop them \
         silently\n  decoded: {decoded}",
        dropped.join(", ")
    );
}

/// The field [`assert_vendor_fields_kept`] adds to every object of a
/// response.
pub const VENDOR_FIELD: &str = "x_allama_vendor_field";

/// Add a [`VENDOR_FIELD`] with a unique marker to every object in `value`,
/// recording each object's JSON pointer and marker.
fn add_vendor_fields(value: &mut Value, pointer: String, fields: &mut Vec<(String, String)>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map.iter_mut() {
                let escaped = key.replace('~', "~0").replace('/', "~1");
                add_vendor_fields(child, format!("{pointer}/{escaped}"), fields);
            }
            let marker = format!("allama-conformance-vendor-field-{}", fields.len());
            map.insert(VENDOR_FIELD.to_owned(), Value::String(marker.clone()));
            fields.push((pointer, marker));
        }
        Value::Array(items) => {
            for (n, child) in items.iter_mut().enumerate() {
                add_vendor_fields(child, format!("{pointer}/{n}"), fields);
            }
        }
        _ => {}
    }
}

async fn decode<U, UFut, S, SFut, St>(
    capture: &ResponseCapture,
    unary: &U,
    stream: &S,
    name: &str,
) -> (ChatResponse, ChatResponse)
where
    U: Fn(String) -> UFut,
    UFut: Future<Output = Result<ChatResponse>>,
    S: Fn(String) -> SFut,
    SFut: Future<Output = Result<St>>,
    St: Stream<Item = Result<StreamEvent>> + Send,
{
    let from_chat = unary(capture.unary.clone())
        .await
        .unwrap_or_else(|err| panic!("{name}: `chat` failed to decode the capture: {err:?}"));
    let from_stream = match stream(capture.stream.clone()).await {
        Ok(events) => Box::pin(events).collect_response().await,
        Err(err) => Err(err),
    }
    .unwrap_or_else(|err| panic!("{name}: `chat_stream` failed to decode the capture: {err:?}"));
    (from_chat, from_stream)
}

/// The fields two decodings of one response disagree on.
fn differences(chat: &ChatResponse, stream: &ChatResponse) -> Vec<&'static str> {
    let mut differ = Vec::new();
    if chat.content != stream.content {
        differ.push("content");
    }
    if chat.finish_reason != stream.finish_reason {
        differ.push("finish_reason");
    }
    if chat.usage != stream.usage {
        differ.push("usage");
    }
    // A stream's wire may not carry the response id or the model (some
    // APIs send them only in the unary body): they differ only when both
    // paths report one.
    if chat.id.is_some() && stream.id.is_some() && chat.id != stream.id {
        differ.push("id");
    }
    if chat.model.is_some() && stream.model.is_some() && chat.model != stream.model {
        differ.push("model");
    }
    if chat.provider != stream.provider {
        differ.push("provider");
    }
    if chat.provider_data != stream.provider_data {
        differ.push("provider_data");
    }
    differ
}

/// Whether `haystack` contains `needle`.
fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty()
        && haystack
            .windows(needle.len())
            .any(|window| window == needle)
}

/// Collect a stream's events, for contracts that read them raw.
pub(crate) async fn events<St>(stream: St) -> Vec<Result<StreamEvent>>
where
    St: Stream<Item = Result<StreamEvent>>,
{
    Box::pin(stream).collect().await
}

#[cfg(test)]
mod tests;
