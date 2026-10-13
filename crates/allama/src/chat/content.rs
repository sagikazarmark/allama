//! Content blocks (model output) and content parts (user and tool input).
//!
//! Each kind of content is a `#[non_exhaustive]` struct shared by
//! [`ContentBlock`] and [`ContentPart`] where the semantics overlap. Every
//! struct carries [`ProviderData`] for provider-specific state such as
//! reasoning signatures; see [`crate::provider_data`] for the replay rules.
//!
//! Serialization uses a flat `{"type": "...", ...}` shape, where `type` is
//! always one of allama's own kinds. `Other` content has the reserved type
//! `"other"`: its wire type name goes in `type_name` and its verbatim wire
//! fields under `data`, so they never collide with the keys or the kinds
//! allama owns (see [`OtherContent`]).

use std::fmt;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::macros::text_conversions;
pub use crate::media::{Base64Data, File, FileKind, Image, ImageDetail, MediaSource};
use crate::utils::truncate_display;
use crate::{Error, ExtraMap, ProviderData};

/// Text content.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, Hash)]
#[non_exhaustive]
pub struct Text {
    /// The text.
    pub text: String,
    /// Sources the text cites (model output only; see [`Citation`]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub citations: Vec<Citation>,
    /// Provider-specific data.
    #[serde(default, skip_serializing_if = "ProviderData::is_empty")]
    pub provider_data: ProviderData,
}

impl Text {
    /// Create text content.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            citations: Vec::new(),
            provider_data: ProviderData::new(),
        }
    }

    /// Attach citations.
    #[must_use]
    pub fn with_citations(mut self, citations: Vec<Citation>) -> Self {
        self.citations = citations;
        self
    }

    /// Attach provider data.
    #[must_use]
    pub fn with_provider_data(mut self, provider_data: ProviderData) -> Self {
        self.provider_data = provider_data;
        self
    }

    /// The text a citation supports: its span, or the whole text.
    #[must_use]
    pub fn cited_span(&self, citation: &Citation) -> Option<&str> {
        match &citation.span {
            Some(span) => self.text.get(span.clone()),
            None => Some(&self.text),
        }
    }
}

/// A link from generated text to a source that supports it.
///
/// Citations annotate [`Text`] blocks of model output (see ADR 0007). They
/// are read-only views: adapters replay the provider's own annotations, kept
/// in the block's provider data, rather than re-encoding this list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[non_exhaustive]
pub struct Citation {
    /// What is cited.
    pub source: CitationSource,
    /// The supported part of the text, as a byte range into
    /// [`Text::text`]. `None` means the whole block.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<std::ops::Range<usize>>,
    /// The quoted passage from the source, when the provider reports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cited_text: Option<String>,
    /// Provider-specific details (page numbers, indexes, confidence).
    #[serde(default, skip_serializing_if = "ProviderData::is_empty")]
    pub provider_data: ProviderData,
}

impl Citation {
    /// Cite `source` for the whole text block.
    #[must_use]
    pub fn new(source: CitationSource) -> Self {
        Self {
            source,
            span: None,
            cited_text: None,
            provider_data: ProviderData::new(),
        }
    }

    /// Limit the citation to a byte range of the text.
    #[must_use]
    pub fn with_span(mut self, span: std::ops::Range<usize>) -> Self {
        self.span = Some(span);
        self
    }

    /// Set the quoted source passage.
    #[must_use]
    pub fn with_cited_text(mut self, cited_text: impl Into<String>) -> Self {
        self.cited_text = Some(cited_text.into());
        self
    }

    /// Attach provider data.
    #[must_use]
    pub fn with_provider_data(mut self, provider_data: ProviderData) -> Self {
        self.provider_data = provider_data;
        self
    }
}

/// The source a [`Citation`] points to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum CitationSource {
    /// A web page.
    #[non_exhaustive]
    Url {
        /// Page URL.
        url: String,
        /// Page title.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
    /// A document supplied in the request.
    #[non_exhaustive]
    Document {
        /// Position among the request's documents, when reported.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        index: Option<usize>,
        /// Document title.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
    /// A file stored with the provider.
    #[non_exhaustive]
    File {
        /// Provider file id.
        file_id: String,
        /// File name.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
}

impl CitationSource {
    /// A web page.
    #[must_use]
    pub fn url(url: impl Into<String>, title: Option<String>) -> Self {
        Self::Url {
            url: url.into(),
            title,
        }
    }

    /// A request document.
    #[must_use]
    pub fn document(index: Option<usize>, title: Option<String>) -> Self {
        Self::Document { index, title }
    }

    /// A provider-stored file.
    #[must_use]
    pub fn file(file_id: impl Into<String>, filename: Option<String>) -> Self {
        Self::File {
            file_id: file_id.into(),
            filename,
        }
    }

    /// The source's title or file name, when known.
    #[must_use]
    pub fn title(&self) -> Option<&str> {
        match self {
            Self::Url { title, .. } | Self::Document { title, .. } => title.as_deref(),
            Self::File { filename, .. } => filename.as_deref(),
        }
    }
}

/// A tool call emitted by the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolCall {
    /// Tool call identifier (provider-issued or synthesized by the adapter).
    pub id: String,
    /// Name of the tool.
    pub name: String,
    /// Raw JSON arguments, kept verbatim even when malformed.
    pub arguments: String,
    /// Provider-specific data (e.g. a Gemini thought signature).
    #[serde(default, skip_serializing_if = "ProviderData::is_empty")]
    pub provider_data: ProviderData,
}

impl ToolCall {
    /// Create a tool call.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
            provider_data: ProviderData::new(),
        }
    }

    /// Attach provider data.
    #[must_use]
    pub fn with_provider_data(mut self, provider_data: ProviderData) -> Self {
        self.provider_data = provider_data;
        self
    }

    /// Deserialize the arguments into `T`.
    ///
    /// For a tool sent with [`strict`](crate::ToolDefinition::strict), use
    /// [`parse_arguments_with`](Self::parse_arguments_with): strict modes
    /// may send `null` for an optional property, which a
    /// `#[serde(default)]` field that is not an `Option` rejects.
    ///
    /// # Errors
    ///
    /// Returns a [`ToolArgumentsError`] holding the raw arguments when they
    /// are not valid JSON or do not match `T`.
    pub fn parse_arguments<T: DeserializeOwned>(&self) -> Result<T, ToolArgumentsError> {
        self.parse(None)
    }

    /// Deserialize the arguments into `T`, guided by the definition of the
    /// tool that was called.
    ///
    /// Strict modes that require every property (OpenAI's, for one) make
    /// optional properties nullable, so the model sends `"tags": null`
    /// where it would have left `tags` out. This first removes those
    /// stand-in `null`s, guided by `tool.parameters`, with
    /// [`normalize_strict_nulls`](crate::adapter::schema::normalize_strict_nulls),
    /// so `#[serde(default)]` fields get their default. A `null` the schema
    /// admits (`Option<T>`, `serde_json::Value`), a required property's
    /// `null` and a map entry's `null` are kept.
    ///
    /// It always normalizes, whether or not the tool went out strict (an
    /// adapter may have sent it non-strict with a warning): arguments that
    /// match `tool.parameters` are unaffected.
    ///
    /// ```
    /// use allama::{ToolCall, ToolDefinition};
    /// use serde_json::json;
    ///
    /// #[derive(serde::Deserialize)]
    /// struct Search {
    ///     query: String,
    ///     #[serde(default)]
    ///     tags: Vec<String>,
    /// }
    ///
    /// let tool = ToolDefinition::new(
    ///     "search",
    ///     json!({
    ///         "type": "object",
    ///         "properties": {
    ///             "query": {"type": "string"},
    ///             "tags": {"type": "array", "items": {"type": "string"}}
    ///         },
    ///         "required": ["query"]
    ///     }),
    /// )
    /// .strict(true);
    /// let call = ToolCall::new("call_1", "search", r#"{"query":"rust","tags":null}"#);
    /// assert!(call.parse_arguments::<Search>().is_err());
    /// let args: Search = call.parse_arguments_with(&tool)?;
    /// assert!(args.tags.is_empty());
    /// # Ok::<(), allama::ToolArgumentsError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a [`ToolArgumentsError`] holding the raw arguments when they
    /// are not valid JSON or do not match `T`.
    pub fn parse_arguments_with<T: DeserializeOwned>(
        &self,
        tool: &crate::ToolDefinition,
    ) -> Result<T, ToolArgumentsError> {
        self.parse(Some(&tool.parameters))
    }

    fn parse<T: DeserializeOwned>(
        &self,
        schema: Option<&serde_json::Value>,
    ) -> Result<T, ToolArgumentsError> {
        let raw = if self.arguments.trim().is_empty() {
            "{}"
        } else {
            &self.arguments
        };
        crate::schema::from_str_normalized(raw, schema).map_err(|failure| ToolArgumentsError {
            tool: self.name.clone(),
            raw: self.arguments.clone(),
            path: failure.path,
            source: failure.error,
        })
    }

    /// Returns `true` when the arguments are a valid JSON document.
    #[must_use]
    pub fn has_valid_arguments(&self) -> bool {
        self.arguments.trim().is_empty() || is_valid_json(&self.arguments)
    }
}

/// Tool-call arguments that could not be parsed.
#[derive(Debug)]
#[non_exhaustive]
pub struct ToolArgumentsError {
    /// Name of the tool.
    pub tool: String,
    /// Raw arguments as emitted by the model.
    pub raw: String,
    /// Where in the arguments the value that does not fit the target type
    /// is, such as `filters[0].field`, so a model asked to fix its call
    /// knows which field to change.
    ///
    /// Requires the `schema` feature, and is `None` without it, for
    /// arguments that are not valid JSON, and for a failure at the root.
    /// Only the first error is reported, and the path stops at an
    /// `untagged` enum or a `flatten`ed field (serde buffers their content,
    /// so the path names the enum or the struct holding the `flatten`).
    pub path: Option<String>,
    /// Parse error.
    pub source: serde_json::Error,
}

impl fmt::Display for ToolArgumentsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid arguments for tool `{}`", self.tool)?;
        if let Some(path) = &self.path {
            write!(f, " at `{path}`")?;
        }
        write!(
            f,
            ": {} (raw: {})",
            self.source,
            truncate_display(&self.raw, 200)
        )
    }
}

impl std::error::Error for ToolArgumentsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

impl From<ToolArgumentsError> for Error {
    fn from(err: ToolArgumentsError) -> Self {
        Error::serialization(err.to_string())
    }
}

/// Reasoning ("thinking") emitted by the model.
///
/// Opaque replay state (signatures, encrypted content) lives in
/// `provider_data` under the producing provider's namespace. A reasoning
/// block with empty text and provider data is a *redacted* block.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Reasoning {
    /// Reasoning text (may be a summary, or empty when redacted).
    pub text: String,
    /// Provider-specific data (signatures, encrypted content).
    #[serde(default, skip_serializing_if = "ProviderData::is_empty")]
    pub provider_data: ProviderData,
}

impl Reasoning {
    /// Create a reasoning block.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            provider_data: ProviderData::new(),
        }
    }

    /// Attach provider data.
    #[must_use]
    pub fn with_provider_data(mut self, provider_data: ProviderData) -> Self {
        self.provider_data = provider_data;
        self
    }

    /// Returns `true` when the block has no readable text, only opaque
    /// provider state to replay.
    ///
    /// This covers several provider behaviors: encrypted (redacted)
    /// reasoning, reasoning whose display was omitted
    /// (`ReasoningOutput::Omitted`), and signature-only parts. It says
    /// nothing about *why* the text is absent.
    #[must_use]
    pub fn is_opaque(&self) -> bool {
        self.text.is_empty() && !self.provider_data.is_empty()
    }
}

/// Text in which the model declines to answer, written by the model in
/// place of the answer.
///
/// It is not answer text: [`ChatResponse::text`](crate::ChatResponse::text)
/// leaves it out, a stream delivers it as
/// [`RefusalDelta`](crate::StreamEvent::RefusalDelta) rather than
/// `TextDelta`, and [`ChatResponse::refusal_text`](crate::ChatResponse::refusal_text)
/// reads it ([`is_refusal`](crate::ChatResponse::is_refusal) also covers a
/// refusal finish). Only providers that mark refusal text separately produce it
/// (OpenAI's `refusal` fields); a provider that declines without writing
/// a refusal reports it in the finish reason
/// ([`FinishReason::Refusal`](crate::FinishReason::Refusal) or
/// `ContentFilter`) and its response-level provider data.
///
/// Replayed to the provider that wrote it in its own form, and to any
/// other provider as assistant text.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Refusal {
    /// The refusal text.
    pub text: String,
    /// Provider-specific data.
    #[serde(default, skip_serializing_if = "ProviderData::is_empty")]
    pub provider_data: ProviderData,
}

impl Refusal {
    /// Create a refusal block.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            provider_data: ProviderData::new(),
        }
    }

    /// Attach provider data.
    #[must_use]
    pub fn with_provider_data(mut self, provider_data: ProviderData) -> Self {
        self.provider_data = provider_data;
        self
    }
}

/// A tool call the provider executes itself: web search, code execution,
/// a remote MCP server, and so on.
///
/// It is a separate block from [`ToolCall`] so that code running tools never
/// tries to execute it: the provider already did, and reports the outcome as
/// a [`ProviderToolResult`]. Provider tools are enabled through each
/// provider's request options. Both blocks carry the origin of the provider
/// that ran them and are replayed only to that provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ProviderToolCall {
    /// Call identifier, matched by [`ProviderToolResult::call_id`].
    pub id: String,
    /// Name of the provider tool, e.g. `"web_search"`.
    pub name: String,
    /// Raw JSON input, verbatim.
    pub arguments: String,
    /// Namespace of the provider that executed the call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
    /// Provider-specific data.
    #[serde(default, skip_serializing_if = "ProviderData::is_empty")]
    pub provider_data: ProviderData,
}

impl ProviderToolCall {
    /// Create a provider tool call.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
            origin: None,
            provider_data: ProviderData::new(),
        }
    }

    /// Set the origin namespace.
    #[must_use]
    pub fn with_origin(mut self, origin: impl Into<String>) -> Self {
        self.origin = Some(origin.into());
        self
    }

    /// Attach provider data.
    #[must_use]
    pub fn with_provider_data(mut self, provider_data: ProviderData) -> Self {
        self.provider_data = provider_data;
        self
    }
}

/// The outcome of a [`ProviderToolCall`].
///
/// The output keeps the provider's own shape (search results, execution
/// output, …): it is not portable, only its link to the call and its error
/// flag are.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ProviderToolResult {
    /// Id of the call this is the outcome of.
    pub call_id: String,
    /// The provider's result payload, verbatim.
    #[serde(default)]
    pub output: serde_json::Value,
    /// The tool failed; `output` describes the failure.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_error: bool,
    /// Namespace of the provider that executed the call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
    /// Provider-specific data.
    #[serde(default, skip_serializing_if = "ProviderData::is_empty")]
    pub provider_data: ProviderData,
}

impl ProviderToolResult {
    /// Create a provider tool result.
    #[must_use]
    pub fn new(call_id: impl Into<String>, output: serde_json::Value) -> Self {
        Self {
            call_id: call_id.into(),
            output,
            is_error: false,
            origin: None,
            provider_data: ProviderData::new(),
        }
    }

    /// Mark the result as a failure.
    #[must_use]
    pub fn with_error(mut self, is_error: bool) -> Self {
        self.is_error = is_error;
        self
    }

    /// Set the origin namespace.
    #[must_use]
    pub fn with_origin(mut self, origin: impl Into<String>) -> Self {
        self.origin = Some(origin.into());
        self
    }

    /// Attach provider data.
    #[must_use]
    pub fn with_provider_data(mut self, provider_data: ProviderData) -> Self {
        self.provider_data = provider_data;
        self
    }
}

/// Content the portable model does not represent, preserved verbatim.
///
/// `data` holds the wire fields exactly as the provider sent them; any key
/// is allowed there. State the adapter keeps *about* the content, such as a
/// signature it must send back with it, goes in namespaced `provider_data`
/// like on every other block and part.
///
/// Serializes as `{"type": "other", "type_name": "...", "origin": "...",
/// "data": {...}, "provider_data": {...}}`; `origin` and `provider_data` are
/// omitted when empty. `"other"` is a reserved kind, so a wire type name
/// never takes the place of one of allama's own kinds: an `Other` named
/// `text` or `refusal` round-trips as `Other`, and a kind allama adds later
/// cannot collide with a wire type stored today. Deserializing reads only
/// that shape: a field beside `type` other than those five is an error, not
/// wire data.
///
/// Content allama later models (audio output, for example) moves from
/// `Other` to its own variant in a minor release, so match `Other` only as
/// a fallback, not as the contract.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct OtherContent {
    /// Wire-level type discriminator, serialized as `type_name`.
    pub type_name: String,
    /// The wire fields, verbatim.
    pub data: ExtraMap,
    /// Namespace of the provider that produced the content. Adapters replay
    /// `Other` content only to a matching origin.
    pub origin: Option<String>,
    /// Provider-specific data.
    pub provider_data: ProviderData,
}

impl OtherContent {
    /// Create other content.
    #[must_use]
    pub fn new(type_name: impl Into<String>, data: ExtraMap) -> Self {
        Self {
            type_name: type_name.into(),
            data,
            origin: None,
            provider_data: ProviderData::new(),
        }
    }

    /// Set the origin namespace.
    #[must_use]
    pub fn with_origin(mut self, origin: impl Into<String>) -> Self {
        self.origin = Some(origin.into());
        self
    }

    /// Attach provider data.
    #[must_use]
    pub fn with_provider_data(mut self, provider_data: ProviderData) -> Self {
        self.provider_data = provider_data;
        self
    }
}

/// A block of model output.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ContentBlock {
    /// Text output.
    Text(Text),
    /// Image output.
    Image(Image),
    /// A tool call.
    ToolCall(ToolCall),
    /// Reasoning output.
    Reasoning(Reasoning),
    /// Text in which the model declines to answer; not answer text.
    Refusal(Refusal),
    /// A tool call the provider executed itself.
    ProviderToolCall(ProviderToolCall),
    /// The outcome of a provider-executed tool call.
    ProviderToolResult(ProviderToolResult),
    /// Content the portable model does not represent.
    Other(OtherContent),
}

/// A part of user or tool-result input.
///
/// Build parts with the constructors ([`text`](Self::text),
/// [`image_url`](Self::image_url), …) or convert a string, an [`Image`]
/// (`Image::url(..).into()`) or a [`File`]:
///
/// ```
/// use allama::{ChatRequest, ContentPart, Image};
///
/// let request = ChatRequest::new("demo-model").user(vec![
///     ContentPart::from("What is in this picture?"),
///     Image::url("https://example.com/cat.png").into(),
/// ]);
/// assert_eq!(request.messages.len(), 1);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ContentPart {
    /// Text input.
    Text(Text),
    /// Image input.
    Image(Image),
    /// File or document input.
    File(File),
    /// Input the portable model does not represent.
    Other(OtherContent),
}

impl ContentBlock {
    /// A text block.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(Text::new(text))
    }

    /// A reasoning block.
    #[must_use]
    pub fn reasoning(text: impl Into<String>) -> Self {
        Self::Reasoning(Reasoning::new(text))
    }

    /// A refusal block: text in which the model declined to answer.
    #[must_use]
    pub fn refusal(text: impl Into<String>) -> Self {
        Self::Refusal(Refusal::new(text))
    }

    /// A tool-call block.
    #[must_use]
    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self::ToolCall(ToolCall::new(id, name, arguments))
    }

    /// Borrow the text of a text block.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(&text.text),
            _ => None,
        }
    }

    /// Borrow an image block.
    #[must_use]
    pub fn as_image(&self) -> Option<&Image> {
        match self {
            Self::Image(image) => Some(image),
            _ => None,
        }
    }

    /// Borrow a tool-call block.
    #[must_use]
    pub fn as_tool_call(&self) -> Option<&ToolCall> {
        match self {
            Self::ToolCall(call) => Some(call),
            _ => None,
        }
    }

    /// Borrow a reasoning block.
    #[must_use]
    pub fn as_reasoning(&self) -> Option<&Reasoning> {
        match self {
            Self::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        }
    }

    /// Borrow a refusal block.
    #[must_use]
    pub fn as_refusal(&self) -> Option<&Refusal> {
        match self {
            Self::Refusal(refusal) => Some(refusal),
            _ => None,
        }
    }

    /// Borrow a provider tool call block.
    #[must_use]
    pub fn as_provider_tool_call(&self) -> Option<&ProviderToolCall> {
        match self {
            Self::ProviderToolCall(call) => Some(call),
            _ => None,
        }
    }

    /// Borrow a provider tool result block.
    #[must_use]
    pub fn as_provider_tool_result(&self) -> Option<&ProviderToolResult> {
        match self {
            Self::ProviderToolResult(result) => Some(result),
            _ => None,
        }
    }

    /// Borrow the block's provider data.
    #[must_use]
    pub fn provider_data(&self) -> &ProviderData {
        match self {
            Self::Text(c) => &c.provider_data,
            Self::Image(c) => &c.provider_data,
            Self::ToolCall(c) => &c.provider_data,
            Self::Reasoning(c) => &c.provider_data,
            Self::Refusal(c) => &c.provider_data,
            Self::ProviderToolCall(c) => &c.provider_data,
            Self::ProviderToolResult(c) => &c.provider_data,
            Self::Other(c) => &c.provider_data,
        }
    }

    /// Mutably borrow the block's provider data.
    pub fn provider_data_mut(&mut self) -> &mut ProviderData {
        match self {
            Self::Text(c) => &mut c.provider_data,
            Self::Image(c) => &mut c.provider_data,
            Self::ToolCall(c) => &mut c.provider_data,
            Self::Reasoning(c) => &mut c.provider_data,
            Self::Refusal(c) => &mut c.provider_data,
            Self::ProviderToolCall(c) => &mut c.provider_data,
            Self::ProviderToolResult(c) => &mut c.provider_data,
            Self::Other(c) => &mut c.provider_data,
        }
    }

    /// Wire-level type name of the block.
    #[must_use]
    pub fn type_name(&self) -> &str {
        match self {
            Self::Text(_) => "text",
            Self::Image(_) => "image",
            Self::ToolCall(_) => "tool_call",
            Self::Reasoning(_) => "reasoning",
            Self::Refusal(_) => "refusal",
            Self::ProviderToolCall(_) => "provider_tool_call",
            Self::ProviderToolResult(_) => "provider_tool_result",
            Self::Other(other) => &other.type_name,
        }
    }

    /// The portable type of the block, as a stream announces it in its
    /// `BlockStart` ([`StreamBlockType`](crate::StreamBlockType)).
    /// Every kind of block has one, so
    /// [`StreamBlockType::ALL`](crate::StreamBlockType::ALL) lists
    /// every kind of block.
    #[must_use]
    pub fn block_type(&self) -> crate::StreamBlockType {
        use crate::StreamBlockType;
        match self {
            Self::Text(_) => StreamBlockType::Text,
            Self::Image(_) => StreamBlockType::Image,
            Self::ToolCall(_) => StreamBlockType::ToolCall,
            Self::Reasoning(_) => StreamBlockType::Reasoning,
            Self::Refusal(_) => StreamBlockType::Refusal,
            Self::ProviderToolCall(_) => StreamBlockType::ProviderToolCall,
            Self::ProviderToolResult(_) => StreamBlockType::ProviderToolResult,
            Self::Other(_) => StreamBlockType::Other,
        }
    }

    /// The namespace of the provider that produced the block, for blocks
    /// without portable meaning (provider tool blocks, `Other`).
    #[must_use]
    pub fn origin(&self) -> Option<&str> {
        match self {
            Self::ProviderToolCall(c) => c.origin.as_deref(),
            Self::ProviderToolResult(c) => c.origin.as_deref(),
            Self::Other(c) => c.origin.as_deref(),
            _ => None,
        }
    }
}

impl ContentPart {
    /// The [type name](Self::type_name) of every portable kind of part, in
    /// declaration order (for audits such as the conformance honesty
    /// contract). `Other` parts carry their own wire type name and are left
    /// out. Grows when kinds are added.
    pub const TYPE_NAMES: &'static [&'static str] = &["text", "image", "file"];

    /// A text part.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(Text::new(text))
    }

    /// An image part from a URL.
    #[doc(alias = "image")]
    #[must_use]
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::Image(Image::url(url))
    }

    /// An image part from inline base64 data.
    #[must_use]
    pub fn image_base64(media_type: impl Into<String>, data: impl Into<Base64Data>) -> Self {
        Self::Image(Image::base64(media_type, data))
    }

    /// An image part from raw bytes, sent inline (base64-encoded).
    ///
    /// ```
    /// use allama::{ChatRequest, ContentPart, Message};
    ///
    /// # let png: Vec<u8> = vec![0x89, b'P', b'N', b'G'];
    /// // let png = std::fs::read("chart.png")?;
    /// let request = ChatRequest::new("demo-model").message(Message::user_parts(vec![
    ///     ContentPart::text("What does this chart show?"),
    ///     ContentPart::image_bytes("image/png", &png),
    /// ]));
    /// # let _ = request;
    /// ```
    #[must_use]
    pub fn image_bytes(media_type: impl Into<String>, data: impl AsRef<[u8]>) -> Self {
        Self::Image(Image::bytes(media_type, data))
    }

    /// A file part from inline base64 data.
    #[must_use]
    pub fn file_base64(media_type: impl Into<String>, data: impl Into<Base64Data>) -> Self {
        Self::File(File::base64(media_type, data))
    }

    /// A file part from raw bytes, sent inline (base64-encoded).
    #[must_use]
    pub fn file_bytes(media_type: impl Into<String>, data: impl AsRef<[u8]>) -> Self {
        Self::File(File::bytes(media_type, data))
    }

    /// A file part from a URL.
    #[must_use]
    pub fn file_url(media_type: impl Into<String>, url: impl Into<String>) -> Self {
        Self::File(File::url(media_type, url))
    }

    /// Borrow the text of a text part.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(&text.text),
            _ => None,
        }
    }

    /// Borrow an image part.
    #[must_use]
    pub fn as_image(&self) -> Option<&Image> {
        match self {
            Self::Image(image) => Some(image),
            _ => None,
        }
    }

    /// Borrow a file part.
    #[must_use]
    pub fn as_file(&self) -> Option<&File> {
        match self {
            Self::File(file) => Some(file),
            _ => None,
        }
    }

    /// Wire-level type name of the part.
    #[must_use]
    pub fn type_name(&self) -> &str {
        match self {
            Self::Text(_) => "text",
            Self::Image(_) => "image",
            Self::File(_) => "file",
            Self::Other(other) => &other.type_name,
        }
    }
}

impl From<Text> for ContentBlock {
    fn from(value: Text) -> Self {
        Self::Text(value)
    }
}

impl From<Image> for ContentBlock {
    fn from(value: Image) -> Self {
        Self::Image(value)
    }
}

impl From<ToolCall> for ContentBlock {
    fn from(value: ToolCall) -> Self {
        Self::ToolCall(value)
    }
}

impl From<Reasoning> for ContentBlock {
    fn from(value: Reasoning) -> Self {
        Self::Reasoning(value)
    }
}

impl From<Refusal> for ContentBlock {
    fn from(value: Refusal) -> Self {
        Self::Refusal(value)
    }
}

impl From<ProviderToolCall> for ContentBlock {
    fn from(value: ProviderToolCall) -> Self {
        Self::ProviderToolCall(value)
    }
}

impl From<ProviderToolResult> for ContentBlock {
    fn from(value: ProviderToolResult) -> Self {
        Self::ProviderToolResult(value)
    }
}

impl From<OtherContent> for ContentBlock {
    fn from(value: OtherContent) -> Self {
        Self::Other(value)
    }
}

impl From<Text> for ContentPart {
    fn from(value: Text) -> Self {
        Self::Text(value)
    }
}

impl From<Image> for ContentPart {
    fn from(value: Image) -> Self {
        Self::Image(value)
    }
}

impl From<File> for ContentPart {
    fn from(value: File) -> Self {
        Self::File(value)
    }
}

text_conversions!(ContentPart, Self::text);

fn describe_source(kind: &str, source: &MediaSource) -> String {
    match source {
        MediaSource::Url { url } => format!("{kind}:url({})", truncate_display(url, 50)),
        MediaSource::Base64 { media_type, .. } => format!("{kind}:base64({media_type})"),
        MediaSource::ProviderFile { origin, .. } => format!("{kind}:provider_file({origin})"),
    }
}

impl fmt::Display for ContentBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text(text) => f.write_str(&text.text),
            Self::Image(image) => f.write_str(&describe_source("image", &image.source)),
            Self::ToolCall(call) => write!(
                f,
                "tool_call:{}({})",
                call.name,
                truncate_display(&call.arguments, 50)
            ),
            Self::Reasoning(reasoning) => {
                write!(f, "[reasoning] {}", truncate_display(&reasoning.text, 50))
            }
            Self::Refusal(refusal) => {
                write!(f, "[refusal] {}", truncate_display(&refusal.text, 50))
            }
            Self::ProviderToolCall(call) => write!(
                f,
                "provider_tool_call:{}({})",
                call.name,
                truncate_display(&call.arguments, 50)
            ),
            Self::ProviderToolResult(result) => {
                write!(f, "provider_tool_result:{}", result.call_id)
            }
            Self::Other(other) => write!(f, "other:{}", other.type_name),
        }
    }
}

impl fmt::Display for ContentPart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text(text) => f.write_str(&text.text),
            Self::Image(image) => f.write_str(&describe_source("image", &image.source)),
            Self::File(file) => write!(f, "file({})", file.media_type),
            Self::Other(other) => write!(f, "other:{}", other.type_name),
        }
    }
}

// --- serde ------------------------------------------------------------------

/// The reserved `type` of [`OtherContent`]; its wire type name goes in
/// [`TYPE_NAME_KEY`].
const OTHER_TYPE: &str = "other";
const TYPE_NAME_KEY: &str = "type_name";
const ORIGIN_KEY: &str = "origin";
const DATA_KEY: &str = "data";
const PROVIDER_DATA_KEY: &str = "provider_data";

fn tagged<S: Serializer, T: Serialize>(
    serializer: S,
    type_name: &str,
    value: &T,
) -> Result<S::Ok, S::Error> {
    let mut map = match serde_json::to_value(value).map_err(serde::ser::Error::custom)? {
        serde_json::Value::Object(map) => map,
        _ => return Err(serde::ser::Error::custom("content must serialize as a map")),
    };
    map.insert("type".into(), type_name.into());
    map.serialize(serializer)
}

fn serialize_other<S: Serializer>(serializer: S, other: &OtherContent) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeMap;

    let mut map = serializer.serialize_map(None)?;
    map.serialize_entry("type", OTHER_TYPE)?;
    map.serialize_entry(TYPE_NAME_KEY, &other.type_name)?;
    if let Some(origin) = &other.origin {
        map.serialize_entry(ORIGIN_KEY, origin)?;
    }
    map.serialize_entry(DATA_KEY, &other.data)?;
    if !other.provider_data.is_empty() {
        map.serialize_entry(PROVIDER_DATA_KEY, &other.provider_data)?;
    }
    map.end()
}

fn take_type<E: serde::de::Error>(map: &mut ExtraMap) -> Result<String, E> {
    match map.remove("type") {
        Some(serde_json::Value::String(type_name)) => Ok(type_name),
        Some(_) => Err(E::custom("\"type\" must be a string")),
        None => Err(E::custom("missing \"type\" field")),
    }
}

fn from_map<T: DeserializeOwned, E: serde::de::Error>(map: ExtraMap) -> Result<T, E> {
    serde_json::from_value(serde_json::Value::Object(map)).map_err(E::custom)
}

/// Decode `Other` content from the fields left after `"type": "other"`.
///
/// Only the form [`OtherContent`] serializes to is read: the string
/// `type_name`, the wire fields nested under `data` (an object), plus an
/// optional string `origin` and optional namespaced `provider_data`. Any
/// other field, such as a wire field written beside `type`, is an error, so
/// wire data is never mistaken for the fields allama owns.
fn other_from_map<E: serde::de::Error>(map: ExtraMap) -> Result<OtherContent, E> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Nested {
        type_name: String,
        data: ExtraMap,
        #[serde(default)]
        origin: Option<String>,
        #[serde(default)]
        provider_data: ProviderData,
    }

    let type_name = match map.get(TYPE_NAME_KEY) {
        Some(serde_json::Value::String(type_name)) => format!(" `{type_name}`"),
        _ => String::new(),
    };
    let nested: Nested = serde_json::from_value(serde_json::Value::Object(map)).map_err(|err| {
        E::custom(format_args!(
            "`Other` content{type_name} must name its wire type in `type_name` and nest its wire fields under `data`: {err}"
        ))
    })?;
    Ok(OtherContent {
        type_name: nested.type_name,
        data: nested.data,
        origin: nested.origin,
        provider_data: nested.provider_data,
    })
}

/// The error for a `type` this version does not know: a kind added by a
/// newer allama, or `Other` content written before it had its own tag.
fn unknown_type<E: serde::de::Error>(what: &str, type_name: &str) -> E {
    E::custom(format_args!(
        "unknown {what} type `{type_name}` (written by a newer allama?)"
    ))
}

impl Serialize for ContentBlock {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Text(c) => tagged(serializer, "text", c),
            Self::Image(c) => tagged(serializer, "image", c),
            Self::ToolCall(c) => tagged(serializer, "tool_call", c),
            Self::Reasoning(c) => tagged(serializer, "reasoning", c),
            Self::Refusal(c) => tagged(serializer, "refusal", c),
            Self::ProviderToolCall(c) => tagged(serializer, "provider_tool_call", c),
            Self::ProviderToolResult(c) => tagged(serializer, "provider_tool_result", c),
            Self::Other(c) => serialize_other(serializer, c),
        }
    }
}

impl<'de> Deserialize<'de> for ContentBlock {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut map = ExtraMap::deserialize(deserializer)?;
        let type_name = take_type::<D::Error>(&mut map)?;
        Ok(match type_name.as_str() {
            "text" => Self::Text(from_map::<_, D::Error>(map)?),
            "image" => Self::Image(from_map::<_, D::Error>(map)?),
            "tool_call" => Self::ToolCall(from_map::<_, D::Error>(map)?),
            "reasoning" => Self::Reasoning(from_map::<_, D::Error>(map)?),
            "refusal" => Self::Refusal(from_map::<_, D::Error>(map)?),
            "provider_tool_call" => Self::ProviderToolCall(from_map::<_, D::Error>(map)?),
            "provider_tool_result" => Self::ProviderToolResult(from_map::<_, D::Error>(map)?),
            OTHER_TYPE => Self::Other(other_from_map::<D::Error>(map)?),
            _ => return Err(unknown_type::<D::Error>("content block", &type_name)),
        })
    }
}

impl Serialize for ContentPart {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Text(c) => tagged(serializer, "text", c),
            Self::Image(c) => tagged(serializer, "image", c),
            Self::File(c) => tagged(serializer, "file", c),
            Self::Other(c) => serialize_other(serializer, c),
        }
    }
}

impl<'de> Deserialize<'de> for ContentPart {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut map = ExtraMap::deserialize(deserializer)?;
        let type_name = take_type::<D::Error>(&mut map)?;
        Ok(match type_name.as_str() {
            "text" => Self::Text(from_map::<_, D::Error>(map)?),
            "image" => Self::Image(from_map::<_, D::Error>(map)?),
            "file" => Self::File(from_map::<_, D::Error>(map)?),
            OTHER_TYPE => Self::Other(other_from_map::<D::Error>(map)?),
            _ => return Err(unknown_type::<D::Error>("content part", &type_name)),
        })
    }
}

pub(crate) fn is_valid_json(input: &str) -> bool {
    let mut deserializer = serde_json::Deserializer::from_str(input);
    serde::de::IgnoredAny::deserialize(&mut deserializer).is_ok() && deserializer.end().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_call_round_trip_keeps_string_arguments() {
        let block = ContentBlock::tool_call("call_1", "search", r#"{"q":"rust"}"#);
        let value = serde_json::to_value(&block).unwrap();
        assert_eq!(
            value,
            json!({"type": "tool_call", "id": "call_1", "name": "search", "arguments": r#"{"q":"rust"}"#})
        );
        assert_eq!(
            serde_json::from_value::<ContentBlock>(value).unwrap(),
            block
        );
    }

    #[test]
    fn every_kind_is_listed() {
        // Exhaustive on purpose: a new variant fails to compile here, as a
        // reminder to list it.
        fn _exhaustive(part: &ContentPart) {
            match part {
                ContentPart::Text(_)
                | ContentPart::Image(_)
                | ContentPart::File(_)
                | ContentPart::Other(_) => {}
            }
        }
        let parts = [
            ContentPart::text("x"),
            ContentPart::image_url("https://example.com/a.png"),
            ContentPart::file_url("application/pdf", "https://example.com/a.pdf"),
        ];
        let names: Vec<_> = parts.iter().map(ContentPart::type_name).collect();
        assert_eq!(names, ContentPart::TYPE_NAMES);

        let blocks = [
            ContentBlock::text("x"),
            ContentBlock::Image(Image::url("https://example.com/a.png")),
            ContentBlock::reasoning("x"),
            ContentBlock::Refusal(Refusal::new("x")),
            ContentBlock::tool_call("c", "f", "{}"),
            ContentBlock::ProviderToolCall(ProviderToolCall::new("c", "f", "{}")),
            ContentBlock::ProviderToolResult(ProviderToolResult::new("c", json!(null))),
            ContentBlock::Other(OtherContent::new("x", ExtraMap::new())),
        ];
        let block_types: Vec<_> = blocks.iter().map(ContentBlock::block_type).collect();
        assert_eq!(block_types, crate::chat::stream::StreamBlockType::ALL);
    }

    #[test]
    fn refusal_round_trips_under_its_own_tag() {
        let block = ContentBlock::Refusal(Refusal::new("I can't help with that."));
        let value = serde_json::to_value(&block).unwrap();
        assert_eq!(
            value,
            json!({"type": "refusal", "text": "I can't help with that."})
        );
        assert_eq!(
            serde_json::from_value::<ContentBlock>(value).unwrap(),
            block
        );
        assert_eq!(block.as_text(), None);
        assert_eq!(block.type_name(), "refusal");
    }

    #[test]
    fn provider_data_round_trips() {
        let mut call = ToolCall::new("c", "f", "{}");
        call.provider_data
            .insert_raw("google-ai", "thought_signature", json!("sig"));
        let block = ContentBlock::ToolCall(call);
        let value = serde_json::to_value(&block).unwrap();
        assert_eq!(
            value["provider_data"]["google-ai"]["thought_signature"],
            "sig"
        );
        assert_eq!(
            serde_json::from_value::<ContentBlock>(value).unwrap(),
            block
        );
    }

    #[test]
    fn other_round_trips_with_origin() {
        let value = json!({"type": "other", "type_name": "server_tool_use", "origin": "anthropic", "data": {"id": "x"}});
        let block: ContentBlock = serde_json::from_value(value.clone()).unwrap();
        let ContentBlock::Other(other) = &block else {
            panic!("expected other");
        };
        assert_eq!(other.type_name, "server_tool_use");
        assert_eq!(other.origin.as_deref(), Some("anthropic"));
        assert_eq!(other.data["id"], "x");
        assert!(other.provider_data.is_empty());
        assert_eq!(serde_json::to_value(&block).unwrap(), value);
    }

    #[test]
    fn other_keeps_any_wire_key_in_data() {
        let data = json!({"type": "y", "type_name": "w", "origin": "z", "provider_data": {"k": 1}, "data": [1]});
        let other = OtherContent::new("x", serde_json::from_value(data.clone()).unwrap())
            .with_origin("anthropic")
            .with_provider_data(ProviderData::new().with_raw("anthropic", "sig", json!("s")));
        for value in [
            serde_json::to_value(ContentBlock::Other(other.clone())).unwrap(),
            serde_json::to_value(ContentPart::Other(other.clone())).unwrap(),
        ] {
            assert_eq!(
                value,
                json!({"type": "other", "type_name": "x", "origin": "anthropic", "data": data,
                       "provider_data": {"anthropic": {"sig": "s"}}})
            );
            assert_eq!(
                serde_json::from_value::<ContentBlock>(value.clone()).unwrap(),
                ContentBlock::Other(other.clone())
            );
            assert_eq!(
                serde_json::from_value::<ContentPart>(value).unwrap(),
                ContentPart::Other(other.clone())
            );
        }
    }

    /// `Other` content named like one of allama's own kinds (a Gemini raw
    /// `text` part, a vendor `refusal` block) stays `Other`.
    #[test]
    fn other_named_like_a_native_kind_round_trips() {
        let names = [
            "text",
            "image",
            "file",
            "tool_call",
            "reasoning",
            "refusal",
            "provider_tool_call",
            "provider_tool_result",
            "other",
        ];
        for name in names {
            let other =
                OtherContent::new(name, serde_json::from_value(json!({"text": "x"})).unwrap())
                    .with_origin("google-ai");
            let block = ContentBlock::Other(other.clone());
            let part = ContentPart::Other(other);
            let value = serde_json::to_value(&block).unwrap();
            assert_eq!(
                value,
                json!({"type": "other", "type_name": name, "origin": "google-ai", "data": {"text": "x"}})
            );
            assert_eq!(
                serde_json::from_value::<ContentBlock>(value.clone()).unwrap(),
                block
            );
            assert_eq!(serde_json::from_value::<ContentPart>(value).unwrap(), part);
        }
    }

    #[test]
    fn other_without_origin_or_provider_data_omits_them() {
        let block = ContentBlock::Other(OtherContent::new("x", ExtraMap::new()));
        let value = serde_json::to_value(&block).unwrap();
        assert_eq!(
            value,
            json!({"type": "other", "type_name": "x", "data": {}})
        );
        assert_eq!(
            serde_json::from_value::<ContentBlock>(value).unwrap(),
            block
        );
    }

    #[test]
    fn an_unknown_type_is_an_error() {
        // A kind this version does not know, such as one a newer allama
        // added, or `Other` content written before it had its own tag.
        for value in [
            json!({"type": "audio", "data": {"a": 1}}),
            json!({"type": "search_result", "origin": "anthropic", "data": {}}),
        ] {
            let block = serde_json::from_value::<ContentBlock>(value.clone()).unwrap_err();
            let part = serde_json::from_value::<ContentPart>(value.clone()).unwrap_err();
            let name = value["type"].as_str().unwrap();
            assert_eq!(
                block.to_string(),
                format!("unknown content block type `{name}` (written by a newer allama?)")
            );
            assert_eq!(
                part.to_string(),
                format!("unknown content part type `{name}` (written by a newer allama?)")
            );
        }
    }

    #[test]
    fn other_rejects_the_flat_form() {
        // Wire fields beside `type` (the form written before `Other` nested
        // them under `data`) are not read as wire data.
        for value in [
            json!({
                "type": "other",
                "type_name": "inline_data",
                "inlineData": {"mimeType": "audio/wav", "data": "AAAA"},
                "thoughtSignature": "sig",
                "origin": "google-ai",
            }),
            json!({"type": "other", "type_name": "x", "data": {"a": 1}, "extra": true}),
            json!({"type": "other", "type_name": "x"}),
            json!({"type": "other", "type_name": "x", "origin": "anthropic"}),
            json!({"type": "other", "data": {"a": 1}}),
        ] {
            let block = serde_json::from_value::<ContentBlock>(value.clone()).unwrap_err();
            let part = serde_json::from_value::<ContentPart>(value.clone()).unwrap_err();
            for err in [block, part] {
                assert!(
                    err.to_string()
                        .contains("must name its wire type in `type_name` and nest its wire fields under `data`"),
                    "{value}: {err}"
                );
            }
        }
    }

    #[test]
    fn other_rejects_malformed_owned_fields() {
        // Wire data that would once have been read flat, such as a string
        // `data`, a non-string `origin` or `type_name`, or provider data
        // that is not namespaced objects, is an error, never silently
        // reinterpreted.
        for value in [
            json!({"type": "other", "type_name": "redacted", "data": "opaque"}),
            json!({"type": "other", "type_name": "x", "data": {}, "origin": 7}),
            json!({"type": "other", "type_name": "x", "data": {"a": 1}, "provider_data": "nope"}),
            json!({"type": "other", "type_name": 7, "data": {}}),
        ] {
            let err = serde_json::from_value::<ContentBlock>(value.clone()).unwrap_err();
            assert!(
                err.to_string()
                    .contains("nest its wire fields under `data`"),
                "{value}: {err}"
            );
        }
    }

    #[test]
    fn other_reads_a_null_origin_as_none() {
        let block: ContentBlock = serde_json::from_value(
            json!({"type": "other", "type_name": "x", "data": {"a": 1}, "origin": null}),
        )
        .unwrap();
        let ContentBlock::Other(other) = block else {
            panic!("expected other");
        };
        assert_eq!(other.origin, None);
        assert_eq!(other.data["a"], 1);
    }

    #[test]
    fn every_block_has_provider_data() {
        let data = ProviderData::new().with_raw("ns", "k", json!(1));
        let mut block = ContentBlock::Other(OtherContent::new("x", ExtraMap::new()));
        *block.provider_data_mut() = data.clone();
        assert_eq!(block.provider_data(), &data);
    }

    #[test]
    fn parts_round_trip() {
        let parts = vec![
            ContentPart::text("hi"),
            ContentPart::Image(Image::url("https://x/y.png").with_detail(ImageDetail::High)),
            ContentPart::File(File::base64("application/pdf", "JVBERi0=").with_filename("a.pdf")),
        ];
        let value = serde_json::to_value(&parts).unwrap();
        assert_eq!(value[1]["detail"], "high");
        assert_eq!(value[2]["type"], "file");
        assert_eq!(
            serde_json::from_value::<Vec<ContentPart>>(value).unwrap(),
            parts
        );
    }

    #[test]
    fn parse_arguments_reports_raw_on_failure() {
        let call = ToolCall::new("c", "lookup", "{not json");
        let err = call.parse_arguments::<serde_json::Value>().unwrap_err();
        assert_eq!(err.raw, "{not json");
        assert!(err.to_string().contains("lookup"));
        assert!(!call.has_valid_arguments());
        assert!(ToolCall::new("c", "f", "").has_valid_arguments());
        assert_eq!(
            ToolCall::new("c", "f", "")
                .parse_arguments::<serde_json::Value>()
                .unwrap(),
            json!({})
        );
    }

    #[test]
    fn parse_arguments_with_removes_strict_nulls_the_schema_forbids() {
        #[derive(Debug, PartialEq, Deserialize)]
        struct Args {
            query: String,
            #[serde(default)]
            tags: Vec<String>,
            #[serde(default)]
            limit: u32,
            note: Option<String>,
        }
        let tool = crate::ToolDefinition::new(
            "search",
            json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer"},
                    "note": {"type": ["string", "null"]}
                },
                "required": ["query"]
            }),
        )
        .strict(true);
        let call = ToolCall::new(
            "c",
            "search",
            r#"{"query":"rust","tags":null,"limit":null,"note":null}"#,
        );
        assert!(call.parse_arguments::<Args>().is_err());
        assert_eq!(
            call.parse_arguments_with::<Args>(&tool).unwrap(),
            Args {
                query: "rust".into(),
                tags: Vec::new(),
                limit: 0,
                note: None,
            }
        );
        // Non-strict tools are normalized too, harmlessly.
        let lenient = crate::ToolDefinition::new("search", tool.parameters.clone());
        assert!(call.parse_arguments_with::<Args>(&lenient).is_ok());
        // Empty arguments still mean `{}`; errors keep the raw arguments.
        let empty = ToolCall::new("c", "search", " ");
        assert_eq!(
            empty
                .parse_arguments_with::<serde_json::Value>(&tool)
                .unwrap(),
            json!({})
        );
        let bad = ToolCall::new("c", "search", r#"{"query":null}"#);
        let err = bad.parse_arguments_with::<Args>(&tool).unwrap_err();
        assert_eq!(err.raw, r#"{"query":null}"#);
        assert_eq!(err.tool, "search");
    }

    #[cfg(feature = "schema")]
    #[test]
    fn argument_errors_name_the_field_path() {
        #[derive(Debug, Deserialize)]
        #[allow(dead_code)]
        struct Filter {
            field: String,
            value: u32,
        }
        #[derive(Debug, Deserialize)]
        #[allow(dead_code)]
        struct Args {
            #[serde(default)]
            filters: Vec<Filter>,
        }
        let tool = crate::ToolDefinition::new("search", json!({"type": "object"}));
        let wrong_type = ToolCall::new(
            "c",
            "search",
            r#"{"filters":[{"field":"a","value":1},{"field":"b","value":"x"}]}"#,
        );
        for err in [
            wrong_type.parse_arguments::<Args>().unwrap_err(),
            wrong_type.parse_arguments_with::<Args>(&tool).unwrap_err(),
        ] {
            assert_eq!(err.path.as_deref(), Some("filters[1].value"));
            assert!(
                err.to_string()
                    .starts_with("invalid arguments for tool `search` at `filters[1].value`: invalid type: string \"x\""),
                "{err}"
            );
        }
        // A missing field is reported where the object that lacks it is.
        let missing = ToolCall::new("c", "search", r#"{"filters":[{"field":"a"}]}"#);
        let err = missing.parse_arguments_with::<Args>(&tool).unwrap_err();
        assert_eq!(err.path.as_deref(), Some("filters[0]"));
        assert!(err.to_string().contains("missing field `value`"), "{err}");
        // No path for invalid JSON, or for a failure at the root.
        let syntax = ToolCall::new("c", "search", r#"{"filters":"#);
        assert_eq!(syntax.parse_arguments::<Args>().unwrap_err().path, None);
        let root = ToolCall::new("c", "search", r#""text""#);
        let err = root.parse_arguments::<Args>().unwrap_err();
        assert_eq!(err.path, None);
        assert!(
            err.to_string()
                .starts_with("invalid arguments for tool `search`: invalid type"),
            "{err}"
        );
    }

    #[cfg(not(feature = "schema"))]
    #[test]
    fn argument_errors_have_no_path_without_the_schema_feature() {
        #[derive(Debug, Deserialize)]
        #[allow(dead_code)]
        struct Args {
            limit: u32,
        }
        let call = ToolCall::new("c", "search", r#"{"limit":"x"}"#);
        assert_eq!(call.parse_arguments::<Args>().unwrap_err().path, None);
    }

    #[test]
    fn display_truncates() {
        let block = ContentBlock::tool_call("i", "search", "a".repeat(100));
        let shown = block.to_string();
        assert!(shown.starts_with("tool_call:search("));
        assert!(shown.ends_with("...)"));
        assert_eq!(
            ContentBlock::reasoning("think").to_string(),
            "[reasoning] think"
        );
    }

    #[test]
    fn citations_round_trip_and_resolve_spans() {
        let text = Text::new("Rust 1.99 is out.").with_citations(vec![
            Citation::new(CitationSource::url(
                "https://blog.rust-lang.org",
                Some("Rust Blog".into()),
            ))
            .with_span(0..9)
            .with_cited_text("Announcing Rust 1.99"),
            Citation::new(CitationSource::document(Some(0), None)),
            Citation::new(CitationSource::file("file_1", Some("notes.pdf".into())))
                .with_span(3..99),
        ]);
        let block = ContentBlock::Text(text.clone());
        let value = serde_json::to_value(&block).unwrap();
        assert_eq!(
            value["citations"][0],
            json!({"source": {"type": "url", "url": "https://blog.rust-lang.org", "title": "Rust Blog"}, "span": {"start": 0, "end": 9}, "cited_text": "Announcing Rust 1.99"})
        );
        assert_eq!(
            value["citations"][1]["source"],
            json!({"type": "document", "index": 0})
        );
        assert_eq!(
            serde_json::from_value::<ContentBlock>(value).unwrap(),
            block
        );

        assert_eq!(text.cited_span(&text.citations[0]), Some("Rust 1.99"));
        assert_eq!(
            text.cited_span(&text.citations[1]),
            Some("Rust 1.99 is out.")
        );
        assert_eq!(text.cited_span(&text.citations[2]), None);
        assert_eq!(text.citations[2].source.title(), Some("notes.pdf"));
        assert!(
            serde_json::to_value(ContentBlock::text("x"))
                .unwrap()
                .get("citations")
                .is_none()
        );
    }

    #[test]
    fn provider_tool_blocks_round_trip() {
        let call = ContentBlock::from(
            ProviderToolCall::new("srv_1", "web_search", r#"{"query":"rust"}"#)
                .with_origin("anthropic"),
        );
        let value = serde_json::to_value(&call).unwrap();
        assert_eq!(
            value,
            json!({"type": "provider_tool_call", "id": "srv_1", "name": "web_search", "arguments": r#"{"query":"rust"}"#, "origin": "anthropic"})
        );
        assert_eq!(serde_json::from_value::<ContentBlock>(value).unwrap(), call);

        let result = ContentBlock::from(
            ProviderToolResult::new("srv_1", json!({"error_code": "max_uses_exceeded"}))
                .with_error(true)
                .with_origin("anthropic"),
        );
        let value = serde_json::to_value(&result).unwrap();
        assert_eq!(value["is_error"], true);
        assert_eq!(
            serde_json::from_value::<ContentBlock>(value).unwrap(),
            result
        );
        assert_eq!(result.origin(), Some("anthropic"));
        assert!(result.as_provider_tool_result().is_some());
        assert_eq!(
            call.to_string(),
            r#"provider_tool_call:web_search({"query":"rust"})"#
        );
        assert!(
            serde_json::to_value(ContentBlock::from(ProviderToolResult::new(
                "x",
                json!(null)
            )))
            .unwrap()
            .get("is_error")
            .is_none()
        );
    }

    #[test]
    fn redacted_reasoning() {
        let r = Reasoning::new("").with_provider_data(ProviderData::new().with_raw(
            "anthropic",
            "redacted_data",
            json!("x"),
        ));
        assert!(r.is_opaque());
        assert!(!Reasoning::new("t").is_opaque());
    }
}
