use std::borrow::Cow;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::chat::message::text_block_segments;
use crate::history::TurnEnvelope;
use crate::utils::concat_segments;
use crate::{
    AssistantMessage, Citation, ContentBlock, Message, ProviderData, ToolCall, Usage, Warning,
};

/// A provider-agnostic chat response.
///
/// Responses are serializable and comparable, so they can be logged,
/// recorded, and replayed.
///
/// A response remembers whether its finish was reported as a finish event
/// already (see [`Prepared::finish`](crate::adapter::chat::Prepared::finish)), so a
/// wrapper that passes it on reports it no second time. Like a
/// [`Warning`]'s emission mark, it is not serialized and is ignored by `==`;
/// a clone keeps it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ChatResponse {
    /// Ordered content blocks.
    pub content: Vec<ContentBlock>,
    /// Normalized reason the model stopped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    /// The provider's raw stop reason, verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason_raw: Option<String>,
    /// Token usage (see the [`Usage`] contract).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Model that produced the response, as reported by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Provider-assigned response identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Provider request identifier (from transport headers), for support.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// Identity of the provider that actually served the request. Wrappers
    /// such as fallback leave the serving provider's value in place.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<Cow<'static, str>>,
    /// Non-fatal warnings (ignored hints, approximations, dropped state).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<Warning>,
    /// Provider-specific response data.
    #[serde(default, skip_serializing_if = "ProviderData::is_empty")]
    pub provider_data: ProviderData,
    /// Whether an abnormal finish was reported as a finish event already.
    #[serde(skip)]
    pub(crate) finish_reported: bool,
}

impl PartialEq for ChatResponse {
    /// Compares everything but whether the finish was reported.
    fn eq(&self, other: &Self) -> bool {
        let Self {
            content,
            finish_reason,
            finish_reason_raw,
            usage,
            model,
            id,
            request_id,
            provider,
            warnings,
            provider_data,
            finish_reported: _,
        } = self;
        *content == other.content
            && *finish_reason == other.finish_reason
            && *finish_reason_raw == other.finish_reason_raw
            && *usage == other.usage
            && *model == other.model
            && *id == other.id
            && *request_id == other.request_id
            && *provider == other.provider
            && *warnings == other.warnings
            && *provider_data == other.provider_data
    }
}

impl Eq for ChatResponse {}

impl ChatResponse {
    /// Create a response with the given content.
    #[must_use]
    pub fn new(content: Vec<ContentBlock>) -> Self {
        Self {
            content,
            ..Self::default()
        }
    }

    /// Set the finish reason.
    #[must_use]
    pub fn finish_reason(mut self, finish_reason: FinishReason) -> Self {
        self.finish_reason = Some(finish_reason);
        self
    }

    /// Set the raw finish reason.
    #[must_use]
    pub fn finish_reason_raw(mut self, raw: impl Into<String>) -> Self {
        self.finish_reason_raw = Some(raw.into());
        self
    }

    /// Set usage.
    #[must_use]
    pub fn usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Set the model.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the response id.
    #[must_use]
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set the provider request id.
    #[must_use]
    pub fn request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    /// Set the serving provider.
    #[must_use]
    pub fn provider(mut self, provider: impl Into<Cow<'static, str>>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Append a warning.
    #[must_use]
    pub fn warning(mut self, warning: Warning) -> Self {
        self.warnings.push(warning);
        self
    }

    /// Set provider data.
    #[must_use]
    pub fn provider_data(mut self, provider_data: ProviderData) -> Self {
        self.provider_data = provider_data;
        self
    }

    /// Whether the model finished its turn: it ended it
    /// ([`FinishReason::Stop`]) or waits for tool results
    /// ([`FinishReason::ToolCalls`]).
    ///
    /// `false` for every abnormal finish (the output token limit, a content
    /// filter, a refusal, any other reason) and when the response reports
    /// no finish reason, so one call detects output that is not a complete
    /// answer. A streamed response reports none when the stream ended with
    /// its terminal marker but without a reason; it then also carries a
    /// warning.
    ///
    /// ```
    /// use allama::{ChatResponse, ContentBlock, FinishReason};
    ///
    /// let cut_off = ChatResponse::new(vec![ContentBlock::text("Once upon a")])
    ///     .finish_reason(FinishReason::Length);
    /// assert!(!cut_off.finished_normally());
    /// assert!(!ChatResponse::default().finished_normally());
    /// ```
    #[must_use]
    pub fn finished_normally(&self) -> bool {
        self.finish_reason
            .as_ref()
            .is_some_and(FinishReason::finished_normally)
    }

    /// Whether the model declined to answer: it wrote a
    /// [`Refusal`](crate::Refusal) block, or the response finished with
    /// [`FinishReason::Refusal`].
    ///
    /// A provider that declines without writing a refusal reports it only
    /// in the [`finish_reason`](Self::finish_reason), so this one call
    /// covers both. A content filter is a different finish
    /// ([`FinishReason::ContentFilter`]) and is not a refusal.
    ///
    /// ```
    /// use allama::{ChatResponse, ContentBlock, FinishReason};
    ///
    /// let written = ChatResponse::new(vec![ContentBlock::refusal("I can't help with that.")]);
    /// let declined = ChatResponse::default().finish_reason(FinishReason::Refusal);
    /// assert!(written.is_refusal());
    /// assert!(declined.is_refusal());
    /// assert!(!ChatResponse::new(vec![ContentBlock::text("Sure.")]).is_refusal());
    /// ```
    #[must_use]
    pub fn is_refusal(&self) -> bool {
        self.finish_reason == Some(FinishReason::Refusal) || self.first_refusal().is_some()
    }

    /// Iterate over the refusal segments: the text of each
    /// [`Refusal`](crate::Refusal) block, in order.
    pub fn refusal_segments(&self) -> impl Iterator<Item = &str> {
        self.content
            .iter()
            .filter_map(ContentBlock::as_refusal)
            .map(|refusal| refusal.text.as_str())
    }

    /// The text in which the model declined to answer: every
    /// [`Refusal`](crate::Refusal) block, concatenated, or `None` when the
    /// model wrote none.
    ///
    /// A provider that declines without writing a refusal reports it only
    /// in the [`finish_reason`](Self::finish_reason), with any details in
    /// its provider data: [`is_refusal`](Self::is_refusal) covers both.
    ///
    /// ```
    /// use allama::{ChatResponse, ContentBlock, FinishReason};
    ///
    /// let response = ChatResponse::new(vec![ContentBlock::refusal("I can't help with that.")])
    ///     .finish_reason(FinishReason::Refusal);
    ///
    /// assert_eq!(response.refusal_text().as_deref(), Some("I can't help with that."));
    /// assert_eq!(response.text(), None); // a refusal is not answer text
    /// ```
    #[must_use]
    pub fn refusal_text(&self) -> Option<String> {
        concat_segments(self.refusal_segments())
    }

    /// The first refusal segment; see [`refusal_text`](Self::refusal_text).
    #[must_use]
    pub fn first_refusal(&self) -> Option<&str> {
        self.refusal_segments().next()
    }

    /// Iterate over text segments: the text blocks only. Reasoning,
    /// refusals, tool calls, provider tool blocks, and `Other` content are
    /// left out.
    ///
    /// The first segment is `text_segments().next()`. To append the text to
    /// a buffer you already have, without the allocation
    /// [`text`](Self::text) makes, extend the buffer:
    ///
    /// ```
    /// use allama::{ChatResponse, ContentBlock};
    ///
    /// let response = ChatResponse::new(vec![
    ///     ContentBlock::text("Hello, "),
    ///     ContentBlock::reasoning("greet back"),
    ///     ContentBlock::text("world"),
    /// ]);
    /// assert_eq!(response.text_segments().next(), Some("Hello, "));
    ///
    /// let mut transcript = String::from("> ");
    /// transcript.extend(response.text_segments());
    /// assert_eq!(transcript, "> Hello, world");
    /// ```
    pub fn text_segments(&self) -> impl Iterator<Item = &str> {
        text_block_segments(&self.content)
    }

    /// Concatenated text, or `None` when there is no text block. Stored
    /// messages follow the same rules ([`Message::text`]).
    #[must_use]
    pub fn text(&self) -> Option<String> {
        concat_segments(self.text_segments())
    }

    /// Concatenated text, or an empty string.
    #[must_use]
    pub fn text_or_empty(&self) -> String {
        self.text().unwrap_or_default()
    }

    /// Iterate over the citations on text blocks, in order.
    pub fn citations(&self) -> impl Iterator<Item = &Citation> {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text(text) => Some(text.citations.iter()),
                _ => None,
            })
            .flatten()
    }

    /// Iterate over reasoning text segments. The first is
    /// `reasoning_segments().next()`.
    pub fn reasoning_segments(&self) -> impl Iterator<Item = &str> {
        self.content
            .iter()
            .filter_map(ContentBlock::as_reasoning)
            .map(|r| r.text.as_str())
    }

    /// Concatenated reasoning text.
    #[must_use]
    pub fn reasoning_text(&self) -> Option<String> {
        concat_segments(self.reasoning_segments())
    }

    /// Iterate over the client tool calls: the function calls the caller
    /// runs and answers with tool messages.
    ///
    /// Calls the provider ran itself
    /// ([`ContentBlock::ProviderToolCall`]) are not included. Neither are
    /// provider-specific client tools that are not function calls (a
    /// freeform custom tool, a shell or computer-use action), which arrive
    /// as [`ContentBlock::Other`]: their provider crate reads and answers
    /// them (see [Tool calls and results](crate::history#tool-calls-and-results)).
    pub fn tool_calls(&self) -> impl Iterator<Item = &ToolCall> {
        self.content.iter().filter_map(ContentBlock::as_tool_call)
    }

    /// The first tool call.
    #[must_use]
    pub fn first_tool_call(&self) -> Option<&ToolCall> {
        self.tool_calls().next()
    }

    /// Returns `true` when the response contains tool calls.
    #[must_use]
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls().next().is_some()
    }

    /// The response envelope: id, request id, model, serving provider,
    /// usage, finish reason, and warnings, for storing with the turn (see
    /// [The turn envelope](crate::history#the-turn-envelope)).
    #[must_use]
    pub fn envelope(&self) -> TurnEnvelope {
        TurnEnvelope {
            id: self.id.clone(),
            request_id: self.request_id.clone(),
            model: self.model.clone(),
            provider: self.provider.clone(),
            usage: self.usage.clone(),
            finish_reason: self.finish_reason.clone(),
            finish_reason_raw: self.finish_reason_raw.clone(),
            warnings: self.warnings.clone(),
        }
    }

    /// This response as an assistant message for the next turn.
    ///
    /// Only the content is carried over; [`envelope`](Self::envelope)
    /// returns the rest for storing alongside.
    #[must_use]
    pub fn to_assistant_message(&self) -> Message {
        Message::Assistant(AssistantMessage::new(self.content.clone()))
    }

    /// Convert into an assistant message for the next turn.
    #[must_use]
    pub fn into_assistant_message(self) -> Message {
        Message::Assistant(AssistantMessage::new(self.content))
    }
}

impl From<ChatResponse> for Message {
    fn from(response: ChatResponse) -> Self {
        response.into_assistant_message()
    }
}

impl From<&ChatResponse> for Message {
    fn from(response: &ChatResponse) -> Self {
        response.to_assistant_message()
    }
}

/// Normalized reason the model stopped generating.
///
/// The provider's raw value is kept in
/// [`ChatResponse::finish_reason_raw`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FinishReason {
    /// Natural end of the turn or a stop sequence.
    Stop,
    /// The output token cap was reached: raise
    /// [`max_output_tokens`](field@crate::ChatRequest::max_output_tokens),
    /// or ask for less reasoning.
    ///
    /// Not every provider tells the output cap from a full context window.
    /// OpenAI-compatible servers (vLLM, llama.cpp, …) also report `length`
    /// when the prompt and the output together filled the model's context
    /// window, so on those `Length` may mean [`ContextWindow`](Self::ContextWindow);
    /// a response that stops far below the cap you set is that case.
    Length,
    /// The model's context window filled up while it was writing: the input
    /// and the output together reached the most tokens the model can
    /// attend to. Raising the output cap does not help; shorten the input
    /// (truncate or summarize the conversation), as for
    /// [`Error::ContextLengthExceeded`](crate::Error::ContextLengthExceeded),
    /// which is the same limit reached before any output.
    ///
    /// Reported by providers that tell it from the output cap (Anthropic's
    /// `model_context_window_exceeded`); see [`Length`](Self::Length) for
    /// those that do not.
    ContextWindow,
    /// The model requested tool calls and waits for their results: function
    /// calls ([`ChatResponse::tool_calls`]), or provider-specific client
    /// tool calls that arrive as [`ContentBlock::Other`] (ADR 0014).
    ToolCalls,
    /// Output was blocked by a safety filter.
    ContentFilter,
    /// The model declined to answer.
    Refusal,
    /// Any other reason, verbatim.
    ///
    /// What to do about it depends on the provider and the reason, so the
    /// adapter's documentation says. For example, Anthropic's `pause_turn`
    /// (a long server tool turn paused) asks the application to append the
    /// response's [`to_assistant_message`](ChatResponse::to_assistant_message)
    /// and call again unchanged.
    Other(String),
}

impl FinishReason {
    /// Whether the model finished its turn: [`Stop`](Self::Stop) or
    /// [`ToolCalls`](Self::ToolCalls). Every other reason is an abnormal
    /// finish: the output is cut off (at the output cap or the context
    /// window), filtered, refused, or ended for a reason allama does not
    /// know.
    #[must_use]
    pub fn finished_normally(&self) -> bool {
        matches!(self, Self::Stop | Self::ToolCalls)
    }

    /// Stable string form.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ContextWindow => "context_window",
            Self::ToolCalls => "tool_calls",
            Self::ContentFilter => "content_filter",
            Self::Refusal => "refusal",
            Self::Other(reason) => reason,
        }
    }
}

impl fmt::Display for FinishReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for FinishReason {
    type Err = std::convert::Infallible;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Ok(match value {
            "stop" => Self::Stop,
            "length" => Self::Length,
            "context_window" => Self::ContextWindow,
            "tool_calls" => Self::ToolCalls,
            "content_filter" => Self::ContentFilter,
            "refusal" => Self::Refusal,
            other => Self::Other(other.to_owned()),
        })
    }
}

impl Serialize for FinishReason {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for FinishReason {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        Ok(value.parse().expect("infallible"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RequestField, ToolCall};
    use serde_json::json;

    /// A refusal is a refusal block or a `Refusal` finish; accessors follow
    /// the type's `first_*` / `*_text` naming.
    #[test]
    fn refusal_accessors() {
        let written = ChatResponse::new(vec![
            ContentBlock::refusal("I can't"),
            ContentBlock::text("answer"),
            ContentBlock::refusal(" help."),
        ]);
        assert!(written.is_refusal());
        assert_eq!(written.first_refusal(), Some("I can't"));
        assert_eq!(written.refusal_text().as_deref(), Some("I can't help."));
        assert_eq!(written.refusal_segments().count(), 2);

        let declined = ChatResponse::default().finish_reason(FinishReason::Refusal);
        assert!(declined.is_refusal());
        assert_eq!(declined.refusal_text(), None);

        let filtered = ChatResponse::default().finish_reason(FinishReason::ContentFilter);
        assert!(!filtered.is_refusal());
    }

    #[test]
    fn finish_reason_round_trip() {
        for (reason, s) in [
            (FinishReason::Stop, "stop"),
            (FinishReason::Length, "length"),
            (FinishReason::ContextWindow, "context_window"),
            (FinishReason::ToolCalls, "tool_calls"),
            (FinishReason::ContentFilter, "content_filter"),
            (FinishReason::Refusal, "refusal"),
            (FinishReason::Other("pause_turn".into()), "pause_turn"),
        ] {
            assert_eq!(serde_json::to_value(&reason).unwrap(), json!(s));
            assert_eq!(
                serde_json::from_value::<FinishReason>(json!(s)).unwrap(),
                reason
            );
        }
    }

    #[test]
    fn response_round_trip() {
        let response = ChatResponse::new(vec![
            ContentBlock::reasoning("think"),
            ContentBlock::text("Hello"),
            ContentBlock::ToolCall(ToolCall::new("c", "f", "{}")),
        ])
        .finish_reason(FinishReason::ToolCalls)
        .finish_reason_raw("tool_use")
        .usage(Usage::new().input_tokens(3).output_tokens(4))
        .model("m")
        .id("resp_1")
        .request_id("req_1")
        .provider("anthropic")
        .warning(Warning::unsupported(RequestField::Seed, "ignored"))
        .provider_data(ProviderData::new().with_raw("openai", "fp", json!("x")));
        let value = serde_json::to_value(&response).unwrap();
        assert_eq!(value["provider"], "anthropic");
        assert_eq!(value["finish_reason_raw"], "tool_use");
        assert_eq!(
            serde_json::from_value::<ChatResponse>(value).unwrap(),
            response
        );
    }

    #[test]
    fn envelope_copies_every_field_and_round_trips() {
        let warning = Warning::unsupported(RequestField::Seed, "ignored");
        let response = ChatResponse::new(vec![ContentBlock::text("Hello")])
            .finish_reason(FinishReason::Length)
            .finish_reason_raw("max_tokens")
            .usage(Usage::new().input_tokens(3).output_tokens(4))
            .model("m")
            .id("resp_1")
            .request_id("req_1")
            .provider("anthropic")
            .warning(warning.clone())
            .provider_data(ProviderData::new().with_raw("openai", "fp", json!("x")));

        let envelope = response.envelope();
        assert_eq!(envelope.id.as_deref(), Some("resp_1"));
        assert_eq!(envelope.request_id.as_deref(), Some("req_1"));
        assert_eq!(envelope.model.as_deref(), Some("m"));
        assert_eq!(envelope.provider.as_deref(), Some("anthropic"));
        assert_eq!(envelope.usage, response.usage);
        assert_eq!(envelope.finish_reason, Some(FinishReason::Length));
        assert_eq!(envelope.finish_reason_raw.as_deref(), Some("max_tokens"));
        assert_eq!(envelope.warnings, [warning]);

        let value = serde_json::to_value(&envelope).unwrap();
        assert_eq!(
            value,
            json!({
                "id": "resp_1",
                "request_id": "req_1",
                "model": "m",
                "provider": "anthropic",
                "usage": {"input_tokens": 3, "output_tokens": 4},
                "finish_reason": "length",
                "finish_reason_raw": "max_tokens",
                "warnings": [{"kind": "unsupported", "subject": {"type": "field", "field": "seed"},
                              "message": "ignored"}],
            })
        );
        assert_eq!(
            serde_json::from_value::<TurnEnvelope>(value).unwrap(),
            envelope
        );
    }

    #[test]
    fn empty_envelope_serializes_to_an_empty_object() {
        let envelope = ChatResponse::default().envelope();
        assert_eq!(envelope, TurnEnvelope::default());
        assert_eq!(serde_json::to_value(&envelope).unwrap(), json!({}));
        assert_eq!(
            serde_json::from_value::<TurnEnvelope>(json!({})).unwrap(),
            envelope
        );
    }

    #[test]
    fn helpers() {
        let response = ChatResponse::new(vec![
            ContentBlock::reasoning("a"),
            ContentBlock::text("x"),
            ContentBlock::text("y"),
            ContentBlock::tool_call("c", "f", "{}"),
        ]);
        assert_eq!(response.text().as_deref(), Some("xy"));
        assert_eq!(response.text_segments().next(), Some("x"));
        assert_eq!(response.reasoning_text().as_deref(), Some("a"));
        assert!(response.has_tool_calls());
        assert_eq!(response.first_tool_call().unwrap().id, "c");
        let message = response.to_assistant_message();
        assert_eq!(message.as_assistant().unwrap().content.len(), 4);
        assert!(ChatResponse::default().text().is_none());
    }
}
