use std::fmt;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::macros::text_conversions;
use crate::utils::{DISPLAY_MAX_BYTES, concat_segments, write_truncated, write_truncated_joined};
use crate::{CacheBreakpoint, ContentBlock, ContentPart, ExtraMap, ProviderData, ToolCall};

/// A message in a conversation.
///
/// The request-level system prompt is
/// [`ChatRequest::system`](crate::ChatRequest::system); positional system
/// messages are not modeled yet.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
#[non_exhaustive]
pub enum Message {
    /// User input.
    User(UserMessage),
    /// Model output from an earlier turn.
    Assistant(AssistantMessage),
    /// The result of a tool call.
    Tool(ToolMessage),
}

/// User input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct UserMessage {
    /// Text or multimodal content.
    pub content: UserContent,
    /// Marks the end of a cacheable prefix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheBreakpoint>,
    /// Provider-specific data.
    #[serde(default, skip_serializing_if = "ProviderData::is_empty")]
    pub provider_data: ProviderData,
    /// Application data: never read by adapters or sent to a provider (see
    /// [Application data](crate::history#application-data)).
    #[serde(default, skip_serializing_if = "ExtraMap::is_empty")]
    pub app_data: ExtraMap,
}

/// Model output from an earlier turn, replayed as history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AssistantMessage {
    /// Ordered content blocks.
    pub content: Vec<ContentBlock>,
    /// Marks the end of a cacheable prefix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheBreakpoint>,
    /// Provider-specific data (e.g. a response item id).
    #[serde(default, skip_serializing_if = "ProviderData::is_empty")]
    pub provider_data: ProviderData,
    /// Application data: never read by adapters or sent to a provider (see
    /// [Application data](crate::history#application-data)).
    #[serde(default, skip_serializing_if = "ExtraMap::is_empty")]
    pub app_data: ExtraMap,
}

/// The result of a tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolMessage {
    /// Identifier of the tool call this result answers.
    pub tool_call_id: String,
    /// Name of the tool. Some providers correlate results by name.
    pub name: String,
    /// Tool output.
    pub content: ToolResultContent,
    /// Whether the output describes a failure.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_error: bool,
    /// Marks the end of a cacheable prefix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheBreakpoint>,
    /// Provider-specific data.
    #[serde(default, skip_serializing_if = "ProviderData::is_empty")]
    pub provider_data: ProviderData,
    /// Application data: never read by adapters or sent to a provider (see
    /// [Application data](crate::history#application-data)).
    #[serde(default, skip_serializing_if = "ExtraMap::is_empty")]
    pub app_data: ExtraMap,
}

impl UserMessage {
    /// Create a user message.
    #[must_use]
    pub fn new(content: impl Into<UserContent>) -> Self {
        Self {
            content: content.into(),
            cache: None,
            provider_data: ProviderData::new(),
            app_data: ExtraMap::new(),
        }
    }

    /// Iterate over the text segments; see [`Message::text_segments`].
    pub fn text_segments(&self) -> impl Iterator<Item = &str> {
        self.content.text_segments()
    }

    /// Concatenated text, or `None` when there is none; see
    /// [`Message::text`].
    #[must_use]
    pub fn text(&self) -> Option<String> {
        self.content.text()
    }
}

impl AssistantMessage {
    /// Create an assistant message from blocks.
    #[must_use]
    pub fn new(content: Vec<ContentBlock>) -> Self {
        Self {
            content,
            cache: None,
            provider_data: ProviderData::new(),
            app_data: ExtraMap::new(),
        }
    }

    /// Iterate over tool calls in this message.
    pub fn tool_calls(&self) -> impl Iterator<Item = &ToolCall> {
        self.content.iter().filter_map(ContentBlock::as_tool_call)
    }

    /// Iterate over the text blocks; see [`Message::text_segments`].
    pub fn text_segments(&self) -> impl Iterator<Item = &str> {
        text_block_segments(&self.content)
    }

    /// Concatenated text blocks, or `None` when there is none; see
    /// [`Message::text`].
    #[must_use]
    pub fn text(&self) -> Option<String> {
        concat_segments(self.text_segments())
    }
}

impl ToolMessage {
    /// Create a successful tool result.
    #[must_use]
    pub fn new(
        tool_call_id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<ToolResultContent>,
    ) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            name: name.into(),
            content: content.into(),
            is_error: false,
            cache: None,
            provider_data: ProviderData::new(),
            app_data: ExtraMap::new(),
        }
    }
}

impl Message {
    /// A user message.
    #[must_use]
    pub fn user(content: impl Into<UserContent>) -> Self {
        Self::User(UserMessage::new(content))
    }

    /// A user message with multimodal parts.
    #[must_use]
    pub fn user_parts(parts: Vec<ContentPart>) -> Self {
        Self::user(parts)
    }

    /// An assistant message with a single text block.
    #[must_use]
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::Assistant(AssistantMessage::new(vec![ContentBlock::text(text)]))
    }

    /// An assistant message with the given blocks.
    #[must_use]
    pub fn assistant_blocks(content: Vec<ContentBlock>) -> Self {
        Self::Assistant(AssistantMessage::new(content))
    }

    /// A successful tool result.
    #[must_use]
    pub fn tool_result(
        tool_call_id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<ToolResultContent>,
    ) -> Self {
        Self::Tool(ToolMessage::new(tool_call_id, name, content))
    }

    /// A failed tool result.
    #[must_use]
    pub fn tool_error(
        tool_call_id: impl Into<String>,
        name: impl Into<String>,
        error: impl Into<ToolResultContent>,
    ) -> Self {
        let mut message = ToolMessage::new(tool_call_id, name, error);
        message.is_error = true;
        Self::Tool(message)
    }

    /// Mark this message as the end of a cacheable prefix.
    #[must_use]
    pub fn with_cache(mut self, breakpoint: CacheBreakpoint) -> Self {
        *self.cache_mut() = Some(breakpoint);
        self
    }

    /// The role name: `"user"`, `"assistant"`, or `"tool"`.
    #[must_use]
    pub fn role(&self) -> &'static str {
        match self {
            Self::User(_) => "user",
            Self::Assistant(_) => "assistant",
            Self::Tool(_) => "tool",
        }
    }

    /// Borrow a user message.
    #[must_use]
    pub fn as_user(&self) -> Option<&UserMessage> {
        match self {
            Self::User(m) => Some(m),
            _ => None,
        }
    }

    /// Borrow an assistant message.
    #[must_use]
    pub fn as_assistant(&self) -> Option<&AssistantMessage> {
        match self {
            Self::Assistant(m) => Some(m),
            _ => None,
        }
    }

    /// Borrow a tool message.
    #[must_use]
    pub fn as_tool(&self) -> Option<&ToolMessage> {
        match self {
            Self::Tool(m) => Some(m),
            _ => None,
        }
    }

    /// Iterate over the message's text segments: the text of a user
    /// message (its text parts, for multimodal content) or the text blocks
    /// of an assistant message.
    ///
    /// Reasoning, tool calls and their arguments, provider tool blocks,
    /// media, and `Other` content are left out, as in
    /// [`ChatResponse::text_segments`](crate::ChatResponse::text_segments).
    /// A tool message yields nothing, even when its result is text: tool
    /// results are plumbing, not conversation text. Read them from
    /// [`ToolMessage::content`] instead.
    pub fn text_segments(&self) -> impl Iterator<Item = &str> {
        let (user, blocks): (Option<&UserContent>, &[ContentBlock]) = match self {
            Self::User(m) => (Some(&m.content), &[]),
            Self::Assistant(m) => (None, &m.content),
            Self::Tool(_) => (None, &[]),
        };
        user.into_iter()
            .flat_map(UserContent::text_segments)
            .chain(text_block_segments(blocks))
    }

    /// Visit every piece of text the model reads in this message, mutably:
    /// for a redaction or rewriting pass that must not miss a shape.
    ///
    /// It visits user text (plain text and text parts), the text,
    /// reasoning and refusal blocks of an assistant message, and the text
    /// of a tool result (plain text and text parts). Allama owns the walk,
    /// so content shapes added later are covered without a change to the
    /// caller, unlike a `match` with a `_ => {}` arm.
    ///
    /// Structured data is not text and is not visited: tool call arguments,
    /// a tool result's JSON object, media, `Other` content and provider
    /// data. Rewrite those, if you must, by matching them.
    ///
    /// ```
    /// use allama::{ContentPart, Message};
    ///
    /// let mut message = Message::user_parts(vec![
    ///     ContentPart::text("Call me at 555-0100."),
    ///     ContentPart::image_url("https://example.com/a.png"),
    /// ]);
    /// message.for_each_text_mut(|text| *text = text.replace("555-0100", "[phone]"));
    /// assert_eq!(message.text().as_deref(), Some("Call me at [phone]."));
    /// ```
    pub fn for_each_text_mut(&mut self, mut f: impl FnMut(&mut String)) {
        let parts = |parts: &mut Vec<ContentPart>, f: &mut dyn FnMut(&mut String)| {
            for part in parts {
                // Exhaustive on purpose: a new part kind must be decided here.
                match part {
                    ContentPart::Text(text) => f(&mut text.text),
                    ContentPart::Image(_) | ContentPart::File(_) | ContentPart::Other(_) => {}
                }
            }
        };
        match self {
            Self::User(m) => match &mut m.content {
                UserContent::Text(text) => f(text),
                UserContent::Parts(p) => parts(p, &mut f),
            },
            Self::Assistant(m) => {
                for block in &mut m.content {
                    match block {
                        ContentBlock::Text(text) => f(&mut text.text),
                        ContentBlock::Reasoning(reasoning) => f(&mut reasoning.text),
                        ContentBlock::Refusal(refusal) => f(&mut refusal.text),
                        // Exhaustive on purpose: a new block kind must be
                        // decided here.
                        ContentBlock::Image(_)
                        | ContentBlock::ToolCall(_)
                        | ContentBlock::ProviderToolCall(_)
                        | ContentBlock::ProviderToolResult(_)
                        | ContentBlock::Other(_) => {}
                    }
                }
            }
            Self::Tool(m) => match &mut m.content {
                ToolResultContent::Text(text) => f(text),
                ToolResultContent::Parts(p) => parts(p, &mut f),
                ToolResultContent::Object(_) => {}
            },
        }
    }

    /// The message's concatenated text, or `None` when it has none (always
    /// for a tool message). The same rules as [`Message::text_segments`]
    /// and [`ChatResponse::text`](crate::ChatResponse::text) apply.
    ///
    /// Render a stored conversation for display:
    ///
    /// ```
    /// use allama::history::History;
    /// use allama::{ContentBlock, Message};
    ///
    /// let history = History::new(vec![
    ///     Message::user("Weather in Paris?"),
    ///     Message::assistant_blocks(vec![
    ///         ContentBlock::reasoning("The user wants the weather."),
    ///         ContentBlock::tool_call("call_1", "get_weather", r#"{"city":"Paris"}"#),
    ///     ]),
    ///     Message::tool_result("call_1", "get_weather", "Sunny, 24°C"),
    ///     Message::assistant("It is sunny and 24°C in Paris."),
    /// ]);
    /// let stored = serde_json::to_string(&history)?;
    ///
    /// let loaded: History = serde_json::from_str(&stored)?;
    /// let transcript: Vec<(&str, String)> = loaded
    ///     .messages
    ///     .iter()
    ///     .filter_map(|message| Some((message.role(), message.text()?)))
    ///     .collect();
    /// assert_eq!(
    ///     transcript,
    ///     [
    ///         ("user", "Weather in Paris?".to_owned()),
    ///         ("assistant", "It is sunny and 24°C in Paris.".to_owned()),
    ///     ],
    /// );
    /// # Ok::<(), serde_json::Error>(())
    /// ```
    #[must_use]
    pub fn text(&self) -> Option<String> {
        concat_segments(self.text_segments())
    }

    /// Whether the message holds nothing to send: an assistant message
    /// with no blocks, or a user message with no parts or only empty text.
    ///
    /// A tool message is never empty: an empty result is a valid answer.
    ///
    /// Adapters skip empty user and assistant messages without a warning,
    /// since nothing is lost (see
    /// [`chat::retain_turns`](crate::adapter::chat::retain_turns)). A message can be
    /// left empty on purpose, for example by
    /// [`retain_provider_state`](crate::history::retain_provider_state),
    /// which never removes messages so that their application data stays;
    /// a transcript UI or an agent loop can leave such messages out.
    ///
    /// ```
    /// use allama::Message;
    ///
    /// assert!(Message::user("").is_empty());
    /// assert!(Message::assistant_blocks(vec![]).is_empty());
    /// assert!(!Message::assistant("Hello").is_empty());
    /// assert!(!Message::tool_result("call_1", "lookup", "").is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::User(m) => match &m.content {
                UserContent::Text(text) => text.is_empty(),
                UserContent::Parts(parts) => parts
                    .iter()
                    .all(|part| matches!(part, ContentPart::Text(text) if text.text.is_empty())),
            },
            Self::Assistant(m) => m.content.is_empty(),
            Self::Tool(_) => false,
        }
    }

    /// The message's cache breakpoint.
    #[must_use]
    pub fn cache(&self) -> Option<CacheBreakpoint> {
        match self {
            Self::User(m) => m.cache,
            Self::Assistant(m) => m.cache,
            Self::Tool(m) => m.cache,
        }
    }

    fn cache_mut(&mut self) -> &mut Option<CacheBreakpoint> {
        match self {
            Self::User(m) => &mut m.cache,
            Self::Assistant(m) => &mut m.cache,
            Self::Tool(m) => &mut m.cache,
        }
    }

    /// Borrow the message's provider data.
    #[must_use]
    pub fn provider_data(&self) -> &ProviderData {
        match self {
            Self::User(m) => &m.provider_data,
            Self::Assistant(m) => &m.provider_data,
            Self::Tool(m) => &m.provider_data,
        }
    }

    /// Mutably borrow the message's provider data.
    pub fn provider_data_mut(&mut self) -> &mut ProviderData {
        match self {
            Self::User(m) => &mut m.provider_data,
            Self::Assistant(m) => &mut m.provider_data,
            Self::Tool(m) => &mut m.provider_data,
        }
    }

    /// Set an [application data](crate::history#application-data) entry,
    /// replacing any previous value under `key`.
    ///
    /// ```
    /// use allama::Message;
    /// use serde_json::json;
    ///
    /// let message = Message::user("Hi").with_app_data("myapp.message_id", "msg_01");
    /// assert_eq!(message.app_data()["myapp.message_id"], "msg_01");
    /// ```
    #[must_use]
    pub fn with_app_data(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.app_data_mut().insert(key.into(), value.into());
        self
    }

    /// Borrow the message's [application data](crate::history#application-data).
    #[must_use]
    pub fn app_data(&self) -> &ExtraMap {
        match self {
            Self::User(m) => &m.app_data,
            Self::Assistant(m) => &m.app_data,
            Self::Tool(m) => &m.app_data,
        }
    }

    /// Mutably borrow the message's
    /// [application data](crate::history#application-data).
    pub fn app_data_mut(&mut self) -> &mut ExtraMap {
        match self {
            Self::User(m) => &mut m.app_data,
            Self::Assistant(m) => &mut m.app_data,
            Self::Tool(m) => &mut m.app_data,
        }
    }

    /// Decode the application data entry under `key`. Returns `Ok(None)`
    /// when absent.
    ///
    /// ```
    /// use allama::Message;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Debug, PartialEq, Serialize, Deserialize)]
    /// struct Turn { id: u64 }
    ///
    /// let mut message = Message::assistant("Hello!");
    /// message.app_data_insert("myapp.turn", &Turn { id: 7 })?;
    /// assert_eq!(message.app_data_get::<Turn>("myapp.turn")?, Some(Turn { id: 7 }));
    /// # Ok::<(), allama::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when the stored JSON does not match `T`.
    pub fn app_data_get<T: DeserializeOwned>(&self, key: &str) -> crate::Result<Option<T>> {
        get_app_data(self.app_data(), key)
    }

    /// Encode `value` and store it as the application data entry under
    /// `key`, replacing any previous value.
    ///
    /// # Errors
    ///
    /// Returns an error when `value` fails to serialize.
    pub fn app_data_insert<T: Serialize + ?Sized>(
        &mut self,
        key: impl Into<String>,
        value: &T,
    ) -> crate::Result<()> {
        insert_app_data(self.app_data_mut(), key, value)
    }
}

impl From<UserMessage> for Message {
    fn from(value: UserMessage) -> Self {
        Self::User(value)
    }
}

impl From<AssistantMessage> for Message {
    fn from(value: AssistantMessage) -> Self {
        Self::Assistant(value)
    }
}

impl From<ToolMessage> for Message {
    fn from(value: ToolMessage) -> Self {
        Self::Tool(value)
    }
}

impl fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::User(m) => {
                f.write_str("user: ")?;
                match &m.content {
                    UserContent::Text(s) => write_truncated(f, s, DISPLAY_MAX_BYTES),
                    UserContent::Parts(parts) => write_truncated_joined(
                        f,
                        parts.iter().filter_map(ContentPart::as_text),
                        "[multimodal]",
                        DISPLAY_MAX_BYTES,
                    ),
                }
            }
            Self::Assistant(m) => {
                f.write_str("assistant: ")?;
                write_truncated_joined(
                    f,
                    m.content.iter().filter_map(ContentBlock::as_text),
                    "[non-text content]",
                    DISPLAY_MAX_BYTES,
                )
            }
            Self::Tool(m) => {
                f.write_str("tool: ")?;
                match &m.content {
                    ToolResultContent::Text(text) => write_truncated(f, text, DISPLAY_MAX_BYTES),
                    ToolResultContent::Parts(parts) => write_truncated_joined(
                        f,
                        parts.iter().filter_map(ContentPart::as_text),
                        "[multimodal]",
                        DISPLAY_MAX_BYTES,
                    ),
                    ToolResultContent::Object(object) => write_truncated(
                        f,
                        &serde_json::Value::Object(object.clone()).to_string(),
                        DISPLAY_MAX_BYTES,
                    ),
                }
            }
        }
    }
}

/// User message content: plain text or multimodal parts.
///
/// New shapes may be added, so a `match` outside this crate needs a
/// wildcard arm; [`to_parts`](Self::to_parts) and
/// [`text_segments`](Self::text_segments) read every shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum UserContent {
    /// Plain text.
    Text(String),
    /// Multimodal parts.
    Parts(Vec<ContentPart>),
}

impl UserContent {
    /// Iterate over the text segments: the plain text, or the text parts of
    /// multimodal content (media and `Other` parts are left out).
    pub fn text_segments(&self) -> impl Iterator<Item = &str> {
        let (text, parts): (Option<&str>, &[ContentPart]) = match self {
            Self::Text(text) => (Some(text), &[]),
            Self::Parts(parts) => (None, parts),
        };
        text.into_iter()
            .chain(parts.iter().filter_map(ContentPart::as_text))
    }

    /// Concatenated text, or `None` when there is none (multimodal content
    /// without text parts).
    #[must_use]
    pub fn text(&self) -> Option<String> {
        concat_segments(self.text_segments())
    }

    /// Iterate over the content as parts (text content yields one text part).
    #[must_use]
    pub fn to_parts(&self) -> Vec<ContentPart> {
        match self {
            Self::Text(text) => vec![ContentPart::text(text.clone())],
            Self::Parts(parts) => parts.clone(),
        }
    }
}

text_conversions!(UserContent, Self::Text);

impl From<Vec<ContentPart>> for UserContent {
    fn from(parts: Vec<ContentPart>) -> Self {
        Self::Parts(parts)
    }
}

/// Tool output: text, multimodal parts, or a structured JSON object.
///
/// Structured results use an object rather than an arbitrary JSON value:
/// providers that accept structured tool output (e.g. Gemini
/// `functionResponse`) require an object, and providers that only accept
/// text receive it serialized.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum ToolResultContent {
    /// Plain text.
    Text(String),
    /// Multimodal parts.
    Parts(Vec<ContentPart>),
    /// A structured JSON object.
    Object(ExtraMap),
}

impl ToolResultContent {
    /// Text output.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// Structured output from any serializable value. Non-object values are
    /// wrapped as `{"result": value}`.
    ///
    /// # Errors
    ///
    /// Returns an error when `value` fails to serialize.
    pub fn json<T: Serialize>(value: &T) -> crate::Result<Self> {
        Ok(match serde_json::to_value(value)? {
            serde_json::Value::Object(map) => Self::Object(map),
            other => {
                let mut map = ExtraMap::new();
                map.insert("result".into(), other);
                Self::Object(map)
            }
        })
    }

    /// Borrow text output.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            _ => None,
        }
    }

    /// Borrow multimodal output.
    #[must_use]
    pub fn as_parts(&self) -> Option<&[ContentPart]> {
        match self {
            Self::Parts(parts) => Some(parts),
            _ => None,
        }
    }

    /// Borrow structured output.
    #[must_use]
    pub fn as_object(&self) -> Option<&ExtraMap> {
        match self {
            Self::Object(object) => Some(object),
            _ => None,
        }
    }

    /// Render the output as text: text as-is, objects as JSON, parts as
    /// their text parts joined by a blank line (`"\n\n"`), so the boundary
    /// between two parts survives, as the bundled adapters send them.
    /// Returns `None` when parts contain non-text content.
    ///
    /// ```
    /// use allama::{ContentPart, ToolResultContent};
    ///
    /// let parts = ToolResultContent::Parts(vec![ContentPart::text("12 °C"), ContentPart::text("dry")]);
    /// assert_eq!(parts.to_text().as_deref(), Some("12 °C\n\ndry"));
    /// ```
    #[must_use]
    pub fn to_text(&self) -> Option<String> {
        match self {
            Self::Text(text) => Some(text.clone()),
            Self::Object(object) => Some(serde_json::Value::Object(object.clone()).to_string()),
            Self::Parts(parts) => parts
                .iter()
                .map(ContentPart::as_text)
                .collect::<Option<Vec<_>>>()
                .map(|texts| texts.join("\n\n")),
        }
    }
}

text_conversions!(ToolResultContent, Self::Text);

impl From<Vec<ContentPart>> for ToolResultContent {
    fn from(parts: Vec<ContentPart>) -> Self {
        Self::Parts(parts)
    }
}

impl From<ExtraMap> for ToolResultContent {
    fn from(object: ExtraMap) -> Self {
        Self::Object(object)
    }
}

/// The text of `blocks`: the text blocks only, in order. Shared by messages
/// and [`ChatResponse`](crate::ChatResponse) so their text accessors agree.
pub(crate) fn text_block_segments(blocks: &[ContentBlock]) -> impl Iterator<Item = &str> {
    blocks.iter().filter_map(ContentBlock::as_text)
}

/// Decode an application data entry (shared by messages and system prompts).
pub(crate) fn get_app_data<T: DeserializeOwned>(
    app_data: &ExtraMap,
    key: &str,
) -> crate::Result<Option<T>> {
    app_data
        .get(key)
        .map(|value| T::deserialize(value).map_err(crate::Error::from))
        .transpose()
}

/// Encode and store an application data entry (shared by messages and
/// system prompts).
pub(crate) fn insert_app_data<T: Serialize + ?Sized>(
    app_data: &mut ExtraMap,
    key: impl Into<String>,
    value: &T,
) -> crate::Result<()> {
    app_data.insert(key.into(), serde_json::to_value(value)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CacheRetention;
    use serde_json::json;

    /// Every text shape is visited, and structured
    /// data is left alone.
    #[test]
    fn for_each_text_mut_visits_every_text_and_nothing_else() {
        let mut object = crate::ExtraMap::new();
        object.insert("note".into(), json!("SECRET in an object"));
        let mut messages = vec![
            Message::user("SECRET user"),
            Message::user_parts(vec![
                ContentPart::text("SECRET part"),
                ContentPart::image_url("https://example.com/SECRET.png"),
            ]),
            Message::assistant_blocks(vec![
                ContentBlock::text("SECRET text"),
                ContentBlock::reasoning("SECRET reasoning"),
                ContentBlock::refusal("SECRET refusal"),
                ContentBlock::tool_call("call_1", "lookup", r#"{"q":"SECRET args"}"#),
            ]),
            Message::tool_result("call_1", "lookup", "SECRET tool text"),
            Message::tool_error(
                "call_2",
                "lookup",
                ToolResultContent::Parts(vec![ContentPart::text("SECRET tool part")]),
            ),
            Message::tool_result("call_3", "lookup", ToolResultContent::Object(object)),
        ];
        let mut visited = 0;
        for message in &mut messages {
            message.for_each_text_mut(|text| {
                visited += 1;
                *text = text.replace("SECRET", "[x]");
            });
        }
        assert_eq!(visited, 7);
        let json = serde_json::to_string(&messages).unwrap();
        for kept in ["SECRET.png", "SECRET args", "SECRET in an object"] {
            assert!(json.contains(kept), "{kept}: {json}");
        }
        assert_eq!(json.matches("SECRET").count(), 3, "{json}");
    }

    #[test]
    fn empty_messages_hold_nothing_to_send() {
        let empty = [
            Message::user(""),
            Message::user_parts(vec![]),
            Message::user_parts(vec![ContentPart::text(""), ContentPart::text("")]),
            Message::assistant_blocks(vec![]),
        ];
        for message in &empty {
            assert!(message.is_empty(), "{message:?}");
        }
        let not_empty = [
            Message::user(" "),
            Message::user_parts(vec![ContentPart::text(""), ContentPart::text("hi")]),
            Message::user_parts(vec![ContentPart::Other(crate::OtherContent::new(
                "x",
                ExtraMap::new(),
            ))]),
            // An assistant message is empty only without blocks.
            Message::assistant(""),
            Message::assistant_blocks(vec![ContentBlock::reasoning("")]),
            // An empty tool result is a valid answer.
            Message::tool_result("c1", "f", ""),
            Message::tool_result("c1", "f", Vec::<ContentPart>::new()),
        ];
        for message in &not_empty {
            assert!(!message.is_empty(), "{message:?}");
        }
    }

    #[test]
    fn messages_round_trip() {
        let messages = vec![
            Message::user("hi").with_cache(CacheBreakpoint::with_retention(CacheRetention::Long)),
            Message::assistant_blocks(vec![
                ContentBlock::reasoning("think"),
                ContentBlock::tool_call("c1", "f", "{}"),
            ]),
            Message::tool_error("c1", "f", "boom"),
            Message::tool_result(
                "c2",
                "g",
                ToolResultContent::json(&json!({"ok": true})).unwrap(),
            ),
        ];
        let value = serde_json::to_value(&messages).unwrap();
        assert_eq!(value[0]["role"], "user");
        assert_eq!(value[0]["cache"]["retention"], "long");
        assert_eq!(value[2]["is_error"], true);
        assert_eq!(value[3]["content"], json!({"ok": true}));
        assert_eq!(
            serde_json::from_value::<Vec<Message>>(value).unwrap(),
            messages
        );
    }

    #[test]
    fn unknown_role_is_rejected() {
        assert!(
            serde_json::from_value::<Message>(json!({"role": "developer", "content": "x"}))
                .is_err()
        );
    }

    #[test]
    fn tool_result_json_wraps_non_objects() {
        let content = ToolResultContent::json(&json!([1, 2])).unwrap();
        assert_eq!(content.as_object().unwrap()["result"], json!([1, 2]));
        assert_eq!(content.to_text().unwrap(), r#"{"result":[1,2]}"#);
    }

    /// Text parts keep their boundary, as the
    /// bundled adapters send them.
    #[test]
    fn tool_result_parts_render_with_a_blank_line_between() {
        let parts = ToolResultContent::Parts(vec![
            ContentPart::text("first"),
            ContentPart::text("second"),
        ]);
        assert_eq!(parts.to_text().as_deref(), Some("first\n\nsecond"));
        let with_image = ToolResultContent::Parts(vec![
            ContentPart::text("first"),
            ContentPart::image_url("https://example.com/a.png"),
        ]);
        assert_eq!(with_image.to_text(), None);
    }

    #[test]
    fn display_is_truncated_and_role_prefixed() {
        assert_eq!(Message::user("hello").to_string(), "user: hello");
        assert_eq!(
            Message::assistant_blocks(vec![ContentBlock::tool_call("c", "f", "{}")]).to_string(),
            "assistant: [non-text content]"
        );
    }

    #[test]
    fn app_data_is_serialized_only_when_set() {
        let value = serde_json::to_value(Message::user("hi")).unwrap();
        assert_eq!(value, json!({"role": "user", "content": "hi"}));

        let messages = vec![
            Message::user("hi").with_app_data("id", json!("m1")),
            Message::assistant("hello").with_app_data("id", json!("m2")),
            Message::tool_result("c1", "f", "ok").with_app_data("id", json!("m3")),
        ];
        let value = serde_json::to_value(&messages).unwrap();
        for (index, message) in value.as_array().unwrap().iter().enumerate() {
            assert_eq!(
                message["app_data"],
                json!({"id": format!("m{}", index + 1)})
            );
        }
        assert_eq!(
            serde_json::from_value::<Vec<Message>>(value).unwrap(),
            messages
        );
    }

    #[test]
    fn typed_app_data_round_trips() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct Turn {
            id: String,
            tokens: u32,
        }

        let mut message = Message::tool_result("c1", "f", "ok");
        assert_eq!(message.app_data_get::<Turn>("turn").unwrap(), None);
        let turn = Turn {
            id: "t1".into(),
            tokens: 42,
        };
        message.app_data_insert("turn", &turn).unwrap();
        assert_eq!(message.app_data_get::<Turn>("turn").unwrap(), Some(turn));
        assert_eq!(
            message.app_data()["turn"],
            json!({"id": "t1", "tokens": 42})
        );

        message
            .app_data_mut()
            .insert("turn".into(), json!("not a turn"));
        assert!(message.app_data_get::<Turn>("turn").is_err());
    }

    fn mixed_blocks() -> Vec<ContentBlock> {
        vec![
            ContentBlock::reasoning("thinking"),
            ContentBlock::text("Hello"),
            ContentBlock::tool_call("c1", "f", r#"{"text":"argument"}"#),
            crate::ProviderToolCall::new("p1", "web_search", r#"{"query":"q"}"#).into(),
            crate::OtherContent::new("custom", ExtraMap::new()).into(),
            ContentBlock::text(", world"),
        ]
    }

    #[test]
    fn assistant_text_keeps_only_text_blocks() {
        let message = AssistantMessage::new(mixed_blocks());
        assert_eq!(
            message.text_segments().collect::<Vec<_>>(),
            ["Hello", ", world"]
        );
        assert_eq!(message.text().as_deref(), Some("Hello, world"));

        let message = Message::from(message);
        assert_eq!(
            message.text_segments().collect::<Vec<_>>(),
            ["Hello", ", world"]
        );
        assert_eq!(message.text().as_deref(), Some("Hello, world"));
    }

    #[test]
    fn assistant_text_matches_response_text() {
        let response = crate::ChatResponse::new(mixed_blocks());
        let message = response.to_assistant_message();
        assert_eq!(message.text(), response.text());
        assert!(message.text_segments().eq(response.text_segments()));
    }

    #[test]
    fn assistant_without_text_is_none() {
        let message = AssistantMessage::new(vec![
            ContentBlock::reasoning("thinking"),
            ContentBlock::tool_call("c1", "f", "{}"),
        ]);
        assert_eq!(message.text_segments().count(), 0);
        assert_eq!(message.text(), None);
        assert_eq!(Message::from(message).text(), None);
        assert_eq!(Message::assistant_blocks(Vec::new()).text(), None);
    }

    #[test]
    fn user_text_keeps_only_text_parts() {
        let content = UserContent::Parts(vec![
            ContentPart::text("Look at "),
            ContentPart::image_url("https://example.com/cat.png"),
            ContentPart::text("this"),
        ]);
        assert_eq!(
            content.text_segments().collect::<Vec<_>>(),
            ["Look at ", "this"]
        );
        assert_eq!(content.text().as_deref(), Some("Look at this"));

        let message = UserMessage::new(content);
        assert_eq!(message.text().as_deref(), Some("Look at this"));
        assert_eq!(message.text_segments().count(), 2);
        let message = Message::from(message);
        assert_eq!(message.text().as_deref(), Some("Look at this"));
        assert_eq!(message.text_segments().count(), 2);

        let message = Message::user("plain");
        assert_eq!(message.text_segments().collect::<Vec<_>>(), ["plain"]);
        assert_eq!(message.text().as_deref(), Some("plain"));
        assert_eq!(
            message.as_user().unwrap().content.text().as_deref(),
            Some("plain")
        );
    }

    #[test]
    fn user_without_text_is_none() {
        let content =
            UserContent::Parts(vec![ContentPart::image_url("https://example.com/cat.png")]);
        assert_eq!(content.text_segments().count(), 0);
        assert_eq!(content.text(), None);
        let message = UserMessage::new(content);
        assert_eq!(message.text(), None);
        assert_eq!(Message::from(message).text(), None);
    }

    #[test]
    fn tool_message_text_is_none() {
        for message in [
            Message::tool_result("c1", "f", "Sunny"),
            Message::tool_error("c1", "f", "boom"),
            Message::tool_result("c1", "f", vec![ContentPart::text("parts")]),
        ] {
            assert_eq!(message.text_segments().count(), 0);
            assert_eq!(message.text(), None);
        }
    }

    #[test]
    fn accessors() {
        let m = Message::assistant_blocks(vec![
            ContentBlock::tool_call("c1", "f", "{}"),
            ContentBlock::text("x"),
        ]);
        assert_eq!(m.role(), "assistant");
        assert_eq!(m.as_assistant().unwrap().tool_calls().count(), 1);
        assert!(m.as_user().is_none());
    }
}
