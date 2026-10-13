use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::macros::enum_names;
use crate::{
    CacheBreakpoint, Message, PromptCache, ProviderOptions, RequestOptions, Result, SystemPrompt,
    ToolCall, ToolChoice, ToolDefinition, ToolMessage, ToolResultContent, UnsupportedPolicy,
    UserContent,
};

/// A provider-agnostic chat request.
///
/// Portable fields are public so applications, tests, and wrappers can
/// inspect and adjust requests in place. Provider-specific configuration goes
/// in [`options`](Self::options) (see [`RequestOptions`]).
///
/// Requests are serializable, including exportable provider options, so they
/// can be logged, recorded, and replayed.
///
/// The message and system builders ([`message`](Self::message),
/// [`extend_messages`](Self::extend_messages), [`user`](Self::user),
/// [`assistant`](Self::assistant), [`system`](Self::system)) append. Every
/// other builder replaces the field it sets, including the collection
/// setters [`stop`](Self::stop) and [`tools`](Self::tools). To replace the
/// messages, assign [`messages`](Self::messages); to continue a stored
/// conversation, start from
/// [`History::into_request`](crate::history::History::into_request).
///
/// ```
/// use allama::{ChatRequest, Message};
///
/// let earlier = vec![Message::user("Hi"), Message::assistant("Hello!")];
/// let request = ChatRequest::new("demo-model")
///     .system("You are concise.")
///     .extend_messages(earlier)
///     .user("Say hello again");
/// assert_eq!(request.messages.len(), 3);
/// ```
///
/// # Settings from optional configuration
///
/// The builders take a value. For a setting your configuration may leave
/// unset, assign the public field, which is an `Option`, so an unset value
/// keeps the provider default:
///
/// ```
/// use allama::{ChatRequest, ReasoningEffort};
///
/// struct Config {
///     temperature: Option<f64>,
///     max_tokens: Option<u32>,
///     effort: Option<ReasoningEffort>,
/// }
///
/// let cfg = Config { temperature: Some(0.2), max_tokens: None, effort: Some(ReasoningEffort::Low) };
///
/// let mut request = ChatRequest::new("demo-model").user("Hi");
/// request.temperature = cfg.temperature;
/// request.max_output_tokens = cfg.max_tokens;
/// request.reasoning = cfg.effort.map(Into::into);
/// assert_eq!(request.max_output_tokens, None);
/// ```
///
/// # Untrusted input
///
/// A request serializes whole, [`options`](Self::options) included, which
/// set provider HTTP headers and body fields. Never deserialize a
/// `ChatRequest` from an end user: build it on the server, and take only
/// the user's message (and, if the client holds the conversation, its
/// history through [`retain_portable_content`](crate::history::retain_portable_content);
/// see [The trust boundary](crate::history#the-trust-boundary)).
///
/// Its JSON is a same-version artifact: store conversations as a
/// [`History`](crate::history::History), which is versioned (see
/// [Serialization and stability](crate#serialization-and-stability)).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ChatRequest {
    /// Model identifier.
    pub model: String,
    /// System instructions, in order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub system: Vec<SystemPrompt>,
    /// Conversation messages, in order.
    #[serde(default)]
    pub messages: Vec<Message>,
    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Maximum number of output tokens.
    ///
    /// This is the provider's total output cap, and it includes reasoning:
    /// on providers whose cap covers hidden reasoning (OpenAI
    /// `max_completion_tokens`, Gemini, Anthropic `max_tokens`), reasoning
    /// tokens count against it. Adapters never raise a value you set: a
    /// reasoning hint that does not fit inside it is scaled down or dropped
    /// with a warning, and a provider option that contradicts it is an
    /// [`Error::InvalidRequest`](crate::Error::InvalidRequest).
    ///
    /// A reasoning model can spend part or all of the cap on reasoning, and
    /// then the answer is cut off or empty: the response finishes with
    /// [`FinishReason::Length`](crate::FinishReason::Length), and its
    /// [`Usage::reasoning_tokens`](crate::Usage::reasoning_tokens), when
    /// reported, says how much went to reasoning. Raise the cap, or ask for
    /// less reasoning with [`reasoning`](method@Self::reasoning) (a lower
    /// [`ReasoningEffort`], or [`ReasoningEffort::None`] where the model
    /// allows it).
    ///
    /// A finish at the model's context window is
    /// [`FinishReason::ContextWindow`](crate::FinishReason::ContextWindow),
    /// where a higher cap does not help (see
    /// [`FinishReason::Length`](crate::FinishReason::Length) for providers
    /// that report both as `Length`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Nucleus sampling probability mass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Top-k sampling: sample each token only from the `k` most likely
    /// ones.
    ///
    /// A hint, like the penalties: Anthropic, Gemini, Workers AI and many
    /// self-hosted servers take it, OpenAI does not, and an adapter that
    /// cannot send it drops it with an `Unsupported` warning (or refuses it
    /// under [`UnsupportedPolicy::Error`](crate::UnsupportedPolicy::Error)).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Stop sequences; empty sends none.
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "null_as_empty"
    )]
    pub stop: Vec<String>,
    /// Frequency penalty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    /// Presence penalty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// Tools the model may call; empty offers none.
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "null_as_empty"
    )]
    pub tools: Vec<ToolDefinition>,
    /// How the model should choose tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Requested output format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// Sampling seed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Reasoning configuration; `None` keeps the provider default.
    ///
    /// To turn reasoning off, set [`ReasoningEffort::None`] where the model
    /// allows it. That is not the same as leaving this or the effort unset
    /// (`Option::None`), which keeps the provider default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
    /// Allow several tool calls in one turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// Request-level prompt-cache settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<PromptCache>,
    /// How adapters treat portable hints they cannot honor.
    ///
    /// `None` inherits: a wrapper that sets a provider-level default
    /// ([`UnsupportedPolicyProvider`](crate::wrappers::UnsupportedPolicyProvider))
    /// fills it in, and the provider treats a policy still unset as
    /// [`UnsupportedPolicy::Warn`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unsupported_policy: Option<UnsupportedPolicy>,
    /// Provider-specific options.
    ///
    /// They reach the provider's wire format (headers, body fields), so they
    /// come from the application, never from an end user (see
    /// [Untrusted input](Self#untrusted-input)).
    #[serde(default, skip_serializing_if = "RequestOptions::is_empty")]
    pub options: RequestOptions,
}

impl ChatRequest {
    /// Create a request for `model`.
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            system: Vec::new(),
            messages: Vec::new(),
            temperature: None,
            max_output_tokens: None,
            top_p: None,
            top_k: None,
            stop: Vec::new(),
            frequency_penalty: None,
            presence_penalty: None,
            tools: Vec::new(),
            tool_choice: None,
            response_format: None,
            seed: None,
            reasoning: None,
            parallel_tool_calls: None,
            cache: None,
            unsupported_policy: None,
            options: RequestOptions::new(),
        }
    }

    /// Append a message.
    #[must_use]
    pub fn message(mut self, message: impl Into<Message>) -> Self {
        self.messages.push(message.into());
        self
    }

    /// Append messages, in order, after the existing ones.
    ///
    /// To replace the messages instead, assign the
    /// [`messages`](Self::messages) field.
    #[must_use]
    pub fn extend_messages<I: IntoIterator<Item = Message>>(mut self, messages: I) -> Self {
        self.messages.extend(messages);
        self
    }

    /// Append a system prompt.
    #[must_use]
    pub fn system(mut self, prompt: impl Into<SystemPrompt>) -> Self {
        self.system.push(prompt.into());
        self
    }

    /// Append a user message.
    #[must_use]
    pub fn user(self, content: impl Into<UserContent>) -> Self {
        self.message(Message::user(content))
    }

    /// Append an assistant text message.
    #[must_use]
    pub fn assistant(self, content: impl Into<String>) -> Self {
        self.message(Message::assistant(content))
    }

    /// Append a message in place.
    pub fn push_message(&mut self, message: impl Into<Message>) {
        self.messages.push(message.into());
    }

    /// Every cache breakpoint in the request, in prompt order: system
    /// prompts, tools, then messages.
    pub fn cache_breakpoints(&self) -> impl Iterator<Item = CacheBreakpoint> + '_ {
        self.system
            .iter()
            .filter_map(|prompt| prompt.cache)
            .chain(self.tools.iter().filter_map(|tool| tool.cache))
            .chain(self.messages.iter().filter_map(Message::cache))
    }

    /// Append a successful result for `call`.
    pub fn push_tool_result(&mut self, call: &ToolCall, content: impl Into<ToolResultContent>) {
        self.messages.push(Message::tool_result(
            call.id.clone(),
            call.name.clone(),
            content,
        ));
    }

    /// Append a failed result for `call`: the tool ran and failed, and the
    /// model reads why.
    ///
    /// Keep using it in portable code: each adapter sends the failure the
    /// way its endpoint can.
    ///
    /// - Anthropic: the result with its `is_error` flag set.
    /// - Gemini: the result under the documented `error` key.
    /// - Endpoints with no error flag (OpenAI Chat Completions and
    ///   Responses, the OpenAI-compatible adapters, Workers AI): the error
    ///   text as the result, and one `approximated.tool_result_error`
    ///   warning per request. The model still reads the text. To send an
    ///   encoding of your own to such an endpoint (`{"error": …}`, say),
    ///   encode it yourself and send it with
    ///   [`push_tool_result`](Self::push_tool_result).
    pub fn push_tool_error(&mut self, call: &ToolCall, error: impl Into<ToolResultContent>) {
        self.messages.push(Message::tool_error(
            call.id.clone(),
            call.name.clone(),
            error,
        ));
    }

    /// The tool results in the messages, in order: each
    /// [`Message::Tool`], whether successful or failed
    /// ([`is_error`](ToolMessage::is_error)).
    ///
    /// ```
    /// use allama::{ChatRequest, ToolCall};
    ///
    /// let call = ToolCall::new("call_1", "weather", r#"{"city":"Paris"}"#);
    /// let mut request = ChatRequest::new("demo-model").user("Weather in Paris?");
    /// request.push_tool_result(&call, "18 °C, clear");
    ///
    /// let result = request.tool_results().next().unwrap();
    /// assert_eq!(result.tool_call_id, "call_1");
    /// assert_eq!(result.content.as_text(), Some("18 °C, clear"));
    /// ```
    pub fn tool_results(&self) -> impl Iterator<Item = &ToolMessage> {
        self.messages.iter().filter_map(Message::as_tool)
    }

    /// Set the temperature, replacing any earlier value.
    #[must_use]
    pub fn temperature(mut self, value: f64) -> Self {
        self.temperature = Some(value);
        self
    }

    /// Set the output token cap, replacing any earlier value.
    ///
    /// The cap includes reasoning on providers that count it so, so a
    /// reasoning model may spend it before answering; see
    /// [`max_output_tokens`](field@Self::max_output_tokens) (the field).
    #[must_use]
    pub fn max_output_tokens(mut self, value: u32) -> Self {
        self.max_output_tokens = Some(value);
        self
    }

    /// Set `top_p`, replacing any earlier value.
    #[must_use]
    pub fn top_p(mut self, value: f64) -> Self {
        self.top_p = Some(value);
        self
    }

    /// Visit every piece of text the model reads in this request, mutably:
    /// the system prompts and every message's text (see
    /// [`Message::for_each_text_mut`] for what is visited and what is not).
    ///
    /// A redaction wrapper written with it covers content shapes added
    /// later without a change:
    ///
    /// ```
    /// use allama::{ChatRequest, ContentBlock, Message};
    ///
    /// let mut request = ChatRequest::new("m")
    ///     .system("The user's email is ada@example.com.")
    ///     .user("Write to ada@example.com.")
    ///     .message(Message::assistant_blocks(vec![ContentBlock::reasoning(
    ///         "Draft to ada@example.com.",
    ///     )]));
    /// request.for_each_text_mut(|text| *text = text.replace("ada@example.com", "[email]"));
    /// assert_eq!(request.system[0].content, "The user's email is [email].");
    /// assert_eq!(request.messages[0].text().as_deref(), Some("Write to [email]."));
    /// ```
    pub fn for_each_text_mut(&mut self, mut f: impl FnMut(&mut String)) {
        for prompt in &mut self.system {
            f(&mut prompt.content);
        }
        for message in &mut self.messages {
            message.for_each_text_mut(&mut f);
        }
    }

    /// Set `top_k`, replacing any earlier value (see
    /// [`top_k`](field@Self::top_k)).
    #[must_use]
    pub fn top_k(mut self, value: u32) -> Self {
        self.top_k = Some(value);
        self
    }

    /// Set the stop sequences, replacing any earlier ones.
    ///
    /// An empty list clears them.
    #[must_use]
    pub fn stop<I, S>(mut self, sequences: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.stop = sequences.into_iter().map(Into::into).collect();
        self
    }

    /// Set the frequency penalty, replacing any earlier value.
    #[must_use]
    pub fn frequency_penalty(mut self, value: f64) -> Self {
        self.frequency_penalty = Some(value);
        self
    }

    /// Set the presence penalty, replacing any earlier value.
    #[must_use]
    pub fn presence_penalty(mut self, value: f64) -> Self {
        self.presence_penalty = Some(value);
        self
    }

    /// Set the seed, replacing any earlier value.
    #[must_use]
    pub fn seed(mut self, value: u64) -> Self {
        self.seed = Some(value);
        self
    }

    /// Set the tools, replacing any earlier ones.
    ///
    /// An empty list clears them.
    #[must_use]
    pub fn tools<I: IntoIterator<Item = ToolDefinition>>(mut self, tools: I) -> Self {
        self.tools = tools.into_iter().collect();
        self
    }

    /// Set the tool choice, replacing any earlier value.
    #[must_use]
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Set the response format, replacing any earlier value.
    #[must_use]
    pub fn response_format(mut self, format: ResponseFormat) -> Self {
        self.response_format = Some(format);
        self
    }

    /// Set the reasoning configuration, replacing any earlier value.
    #[must_use]
    pub fn reasoning(mut self, config: impl Into<ReasoningConfig>) -> Self {
        self.reasoning = Some(config.into());
        self
    }

    /// Allow or forbid several tool calls per turn, replacing any earlier
    /// value.
    #[must_use]
    pub fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = Some(enabled);
        self
    }

    /// Set request-level prompt-cache settings, replacing any earlier ones.
    #[must_use]
    pub fn cache(mut self, cache: PromptCache) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Set the unsupported-hint policy, replacing the earlier one.
    #[must_use]
    pub fn unsupported_policy(mut self, policy: UnsupportedPolicy) -> Self {
        self.unsupported_policy = Some(policy);
        self
    }

    /// Attach provider options of the exact type, replacing any earlier
    /// options of that type.
    #[must_use]
    pub fn with_option<T: ProviderOptions + Serialize>(mut self, value: T) -> Self {
        self.options.insert(value);
        self
    }

    /// Attach an application value that is not a provider option: data a
    /// wrapper of yours reads back with [`option`](Self::option), such as a
    /// tenant id or a conversation id. Replaces any earlier value of that
    /// type.
    ///
    /// The value needs no namespace and no serde: adapters never send it,
    /// and it is left out of the request's serialized form and its
    /// equality (an *opaque* option, [`RequestOptions::insert_opaque`](crate::RequestOptions::insert_opaque)).
    #[must_use]
    pub fn with_opaque_option<T: Clone + Send + Sync + 'static>(mut self, value: T) -> Self {
        self.options.insert_opaque(value);
        self
    }

    /// Attach provider options in their namespaced JSON form, replacing any
    /// earlier JSON options for that namespace.
    #[must_use]
    pub fn with_options_json(
        mut self,
        namespace: impl Into<String>,
        value: serde_json::Value,
    ) -> Self {
        self.options.insert_json(namespace, value);
        self
    }

    /// Borrow provider options of the exact type.
    #[must_use]
    pub fn option<T: 'static>(&self) -> Option<&T> {
        self.options.get::<T>()
    }

    /// Resolve provider options of type `T` from either input form, falling
    /// back to `T::default()`.
    ///
    /// # Errors
    ///
    /// See [`RequestOptions::resolve`].
    pub fn resolve_option<T>(&self) -> Result<T>
    where
        T: ProviderOptions + DeserializeOwned + Default,
    {
        Ok(self.options.resolve_or_default::<T>()?.into_owned())
    }

    /// Mutably borrow provider options of the exact type, when attached.
    ///
    /// Like [`option`](Self::option), the JSON form is not consulted; to
    /// change the options whichever form they were given in, use
    /// [`resolve_option_mut`](Self::resolve_option_mut).
    pub fn option_mut<T: Clone + Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        self.options.get_mut::<T>()
    }

    /// Resolve provider options of type `T` for changing in place: the
    /// typed value, else the JSON form, else `T::default()`, stored as the
    /// typed value. Use it to add to the caller's options without
    /// replacing them, as [`with_option`](Self::with_option) would.
    ///
    /// ```
    /// use allama::{ChatRequest, ProviderOptions};
    ///
    /// #[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
    /// struct HeaderOptions {
    ///     headers: Vec<(String, String)>,
    /// }
    ///
    /// impl ProviderOptions for HeaderOptions {
    ///     const NAMESPACE: &'static str = "demo-http";
    /// }
    ///
    /// // The caller's options, given as JSON.
    /// let mut request = ChatRequest::new("demo-model")
    ///     .user("Hi")
    ///     .with_options_json("demo-http", serde_json::json!({"headers": [["x-tenant", "acme"]]}));
    ///
    /// // A wrapper adds its header and keeps the caller's.
    /// request
    ///     .resolve_option_mut::<HeaderOptions>()?
    ///     .headers
    ///     .push(("x-trace".into(), "abc".into()));
    /// assert_eq!(request.option::<HeaderOptions>().unwrap().headers.len(), 2);
    /// # Ok::<(), allama::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// See [`RequestOptions::resolve_mut`].
    pub fn resolve_option_mut<T>(&mut self) -> Result<&mut T>
    where
        T: ProviderOptions + Serialize + DeserializeOwned + Default,
    {
        self.options.resolve_mut::<T>()
    }
}

/// How much effort the model should spend reasoning.
///
/// Levels are ordered, so adapters can map an unsupported level to the
/// nearest supported one (reporting a warning).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ReasoningEffort {
    /// No reasoning: turn reasoning off where the model allows it.
    ///
    /// Not the same as leaving the effort unset (`Option::None` in
    /// [`ReasoningConfig::effort`]), which keeps the provider default.
    None,
    /// The smallest amount of reasoning.
    Minimal,
    /// Low effort.
    Low,
    /// Balanced effort.
    Medium,
    /// High effort.
    High,
    /// Extra-high effort; some models offer more through provider options.
    #[serde(rename = "xhigh")]
    XHigh,
}

enum_names!(ReasoningEffort, "reasoning effort", {
    None => "none",
    Minimal => "minimal",
    Low => "low",
    Medium => "medium",
    High => "high",
    XHigh => "xhigh",
});

impl ReasoningEffort {
    /// All levels, lowest first.
    ///
    /// A slice rather than an array, so adding a level is not a breaking
    /// change to this constant's type.
    pub const ALL: &'static [Self] = &[
        Self::None,
        Self::Minimal,
        Self::Low,
        Self::Medium,
        Self::High,
        Self::XHigh,
    ];

    /// The supported level closest to `self` (ties resolve upward).
    #[must_use]
    pub fn nearest(self, supported: &[ReasoningEffort]) -> Option<ReasoningEffort> {
        let rank = |e: ReasoningEffort| Self::ALL.iter().position(|x| *x == e).unwrap_or(0) as i32;
        supported
            .iter()
            .copied()
            .min_by_key(|candidate| ((rank(*candidate) - rank(self)).abs(), -rank(*candidate)))
    }
}

/// How much of the model's reasoning to return.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ReasoningOutput {
    /// Do not return reasoning content.
    Omitted,
    /// Return a summary of the reasoning.
    Summary,
    /// Return the full reasoning, where the provider allows it.
    Full,
}

enum_names!(ReasoningOutput, "reasoning output", {
    Omitted => "omitted",
    Summary => "summary",
    Full => "full",
});

/// Reasoning configuration.
///
/// Token budgets are provider-specific (see the Anthropic and Gemini option
/// types); the portable surface is effort plus output visibility.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ReasoningConfig {
    /// Effort level; `None` uses the provider default.
    ///
    /// [`ReasoningEffort::None`] is different: no reasoning, turning
    /// reasoning off where the model allows it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
    /// Output visibility; `None` uses the provider default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<ReasoningOutput>,
}

impl ReasoningConfig {
    /// Default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the effort.
    #[must_use]
    pub fn effort(mut self, effort: ReasoningEffort) -> Self {
        self.effort = Some(effort);
        self
    }

    /// Set the output visibility.
    #[must_use]
    pub fn output(mut self, output: ReasoningOutput) -> Self {
        self.output = Some(output);
        self
    }
}

impl From<ReasoningEffort> for ReasoningConfig {
    fn from(effort: ReasoningEffort) -> Self {
        Self::new().effort(effort)
    }
}

/// Requested output format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ResponseFormat {
    /// Plain text.
    Text,
    /// JSON mode: JSON output with no schema.
    ///
    /// What comes back depends on the provider, so prefer
    /// [`json_schema`](Self::json_schema) where the model takes one:
    ///
    /// - OpenAI's JSON mode (`json_object`, which the OpenAI-compatible
    ///   adapters and Workers AI send too) asks for a JSON object, and
    ///   OpenAI refuses the request unless the messages contain the word
    ///   "JSON".
    /// - Gemini returns any JSON value, an array included.
    /// - Anthropic has no JSON mode and refuses it with
    ///   [`Error::Unsupported`](crate::Error::Unsupported).
    ///
    /// No capability answers for it:
    /// [`StructuredOutput`](crate::ChatCapability::StructuredOutput) covers
    /// JSON Schema formats only, so preflight never refuses JSON mode, and
    /// an adapter whose API has none refuses it.
    Json,
    /// JSON constrained by a schema.
    ///
    /// Build it with [`ResponseFormat::json_schema`], which asks for strict
    /// mode, name it with [`name`](ResponseFormat::name), and relax it with
    /// [`non_strict`](ResponseFormat::non_strict). The variant is
    /// `#[non_exhaustive]`, so fields can be added; match it with `..`.
    #[non_exhaustive]
    JsonSchema {
        /// Optional schema name, for providers that use one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// JSON Schema of the output.
        schema: serde_json::Value,
        /// Whether constrained decoding is required: `Some(true)` asks the
        /// provider to constrain the output to the schema with a grammar
        /// (a [strict schema](crate::adapter::schema)), and a provider that cannot
        /// reports it (an `Unsupported` warning, or an error under
        /// [`UnsupportedPolicy::Error`](crate::UnsupportedPolicy::Error)).
        /// `Some(false)` or `None` does not require it; a provider may still
        /// constrain.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
}

impl ResponseFormat {
    /// A JSON Schema format in strict mode (`strict: Some(true)`): the
    /// provider is asked to constrain the output to `schema`, and says so
    /// when it cannot.
    ///
    /// Strict mode is where providers differ:
    ///
    /// - OpenAI (and OpenAI-compatible endpoints with its strict mode)
    ///   accepts a subset of JSON Schema and closes every object. The
    ///   adapter rewrites the schema into that subset: every property
    ///   becomes required, and an optional one becomes nullable, so the
    ///   model sends `null` where it would have left a field out
    ///   ([`Extractor`][crate::extract::Extractor] removes those `null`s
    ///   before deserializing). Each new schema also adds latency to its
    ///   first request, while OpenAI builds its grammar; later requests
    ///   with the same schema reuse it.
    /// - Anthropic and Gemini always constrain a schema response format, so
    ///   the setting changes nothing there.
    /// - A schema the provider's strict mode cannot express (a map, for
    ///   OpenAI) is sent non-strict with an `Unsupported` warning, or fails
    ///   under [`UnsupportedPolicy::Error`](crate::UnsupportedPolicy::Error).
    ///
    /// ```
    /// use allama::ResponseFormat;
    /// use serde_json::json;
    ///
    /// let format = ResponseFormat::json_schema(json!({"type": "object"}));
    /// assert!(matches!(format, ResponseFormat::JsonSchema { strict: Some(true), .. }));
    /// ```
    #[cfg_attr(
        not(feature = "extract"),
        doc = "[crate::extract::Extractor]: crate#cargo-features"
    )]
    #[must_use]
    pub fn json_schema(schema: serde_json::Value) -> Self {
        Self::JsonSchema {
            name: None,
            schema,
            strict: Some(true),
        }
    }

    /// A strict JSON Schema format for `T`'s schema (feature `schema`): the
    /// same as [`json_schema`](Self::json_schema) with the schema
    /// generated from `T`.
    ///
    /// `T` implements [`crate::schemars::JsonSchema`]: derive it through the
    /// [`allama::schemars`](crate::schemars) re-export to stay on the same
    /// `schemars` version. To get a typed value back, use extraction
    /// (`allama::extract`, feature `extract`), which also parses and checks
    /// the output.
    ///
    /// ```
    /// use allama::ResponseFormat;
    /// use allama::schemars::{self, JsonSchema};
    ///
    /// #[derive(JsonSchema)]
    /// struct Rating {
    ///     stars: u8,
    /// }
    ///
    /// let format = ResponseFormat::from_schema::<Rating>();
    /// let ResponseFormat::JsonSchema { schema, strict, .. } = format else {
    ///     unreachable!()
    /// };
    /// assert_eq!(strict, Some(true));
    /// assert!(schema["properties"]["stars"].is_object());
    /// ```
    #[cfg(feature = "schema")]
    #[must_use]
    pub fn from_schema<T: schemars::JsonSchema>() -> Self {
        Self::json_schema(crate::schema::schema_for::<T>())
    }

    /// The same format without requiring constrained decoding
    /// (`strict: Some(false)`): the schema is sent as written, and the
    /// provider may or may not constrain the output to it (Anthropic and
    /// Gemini always do). Nothing is reported either way. Other formats are
    /// returned unchanged.
    ///
    /// Use it for a schema strict mode cannot express, or to keep optional
    /// fields optional on OpenAI.
    ///
    /// ```
    /// use allama::ResponseFormat;
    /// use serde_json::json;
    ///
    /// let format = ResponseFormat::json_schema(json!({"type": "object"})).non_strict();
    /// assert!(matches!(format, ResponseFormat::JsonSchema { strict: Some(false), .. }));
    /// ```
    #[must_use]
    pub fn non_strict(self) -> Self {
        match self {
            Self::JsonSchema { name, schema, .. } => Self::JsonSchema {
                name,
                schema,
                strict: Some(false),
            },
            other => other,
        }
    }

    /// The same format with a schema name, for providers that use one
    /// (OpenAI sends it as `json_schema.name`; others ignore it). Other
    /// formats are returned unchanged.
    ///
    /// ```
    /// use allama::ResponseFormat;
    /// use serde_json::json;
    ///
    /// let format = ResponseFormat::json_schema(json!({"type": "object"})).name("rating");
    /// assert!(matches!(
    ///     format,
    ///     ResponseFormat::JsonSchema { name: Some(name), .. } if name == "rating"
    /// ));
    /// ```
    #[must_use]
    pub fn name(self, name: impl Into<String>) -> Self {
        match self {
            Self::JsonSchema { schema, strict, .. } => Self::JsonSchema {
                name: Some(name.into()),
                schema,
                strict,
            },
            other => other,
        }
    }
}

/// Deserialize a list that may be `null`, which reads as empty.
fn null_as_empty<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<Vec<T>>::deserialize(deserializer).map(Option::unwrap_or_default)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CacheRetention, ContentBlock};
    use serde_json::json;

    #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
    struct DemoOptions {
        tier: Option<String>,
    }

    impl ProviderOptions for DemoOptions {
        const NAMESPACE: &'static str = "demo";
    }

    #[test]
    fn options_change_in_place() {
        let mut request = ChatRequest::new("m");
        assert!(request.option_mut::<DemoOptions>().is_none());
        request.resolve_option_mut::<DemoOptions>().unwrap().tier = Some("flex".into());
        request.option_mut::<DemoOptions>().unwrap().tier = Some("priority".into());
        assert_eq!(
            request.option::<DemoOptions>().unwrap().tier.as_deref(),
            Some("priority")
        );
    }

    #[test]
    fn request_round_trips_with_options() {
        let request = ChatRequest::new("m")
            .system("Be concise")
            .user("hi")
            .max_output_tokens(100)
            .reasoning(ReasoningEffort::High)
            .cache(
                PromptCache::new()
                    .key("session")
                    .retention(CacheRetention::Long),
            )
            .unsupported_policy(UnsupportedPolicy::Error)
            .with_option(DemoOptions {
                tier: Some("flex".into()),
            });
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["options"]["demo"]["tier"], "flex");
        assert_eq!(value["reasoning"]["effort"], "high");
        assert_eq!(value["unsupported_policy"], "error");
        let back: ChatRequest = serde_json::from_value(value).unwrap();
        assert_eq!(back, request);
        assert_eq!(
            back.resolve_option::<DemoOptions>()
                .unwrap()
                .tier
                .as_deref(),
            Some("flex")
        );
    }

    #[test]
    fn minimal_request_shape() {
        let value = serde_json::to_value(ChatRequest::new("m")).unwrap();
        assert_eq!(value, json!({"model": "m", "messages": []}));
    }

    #[test]
    fn push_tool_result_uses_call_identity() {
        let call = ToolCall::new("c1", "lookup", "{}");
        let mut request =
            ChatRequest::new("m").message(Message::assistant_blocks(vec![ContentBlock::ToolCall(
                call.clone(),
            )]));
        request.push_tool_result(&call, "ok");
        request.push_tool_error(&call, "failed");
        let tool = request.messages[1].as_tool().unwrap();
        assert_eq!(
            (tool.tool_call_id.as_str(), tool.name.as_str()),
            ("c1", "lookup")
        );
        assert!(request.messages[2].as_tool().unwrap().is_error);
    }

    #[test]
    fn name_and_non_strict_adjust_only_a_schema_format() {
        let format = ResponseFormat::json_schema(json!({"type": "object"}))
            .name("rating")
            .non_strict();
        assert_eq!(
            format,
            ResponseFormat::JsonSchema {
                name: Some("rating".into()),
                schema: json!({"type": "object"}),
                strict: Some(false),
            }
        );
        for other in [ResponseFormat::Text, ResponseFormat::Json] {
            assert_eq!(other.clone().name("x"), other);
            assert_eq!(other.clone().non_strict(), other);
        }
    }

    #[test]
    fn empty_tools_and_stop_are_not_serialized() {
        let request = ChatRequest::new("m")
            .stop(["x"])
            .tools([ToolDefinition::new(
                "t",
                serde_json::json!({"type": "object"}),
            )])
            .stop(Vec::<String>::new())
            .tools(Vec::new());
        assert!(request.stop.is_empty() && request.tools.is_empty());
        assert_eq!(request, ChatRequest::new("m"));
        assert_eq!(
            serde_json::to_value(&request).unwrap(),
            serde_json::json!({"model": "m", "messages": []})
        );
    }

    /// Requests stored before `tools` and `stop` were plain lists wrote
    /// `null` or `[]` for an empty one; both still load as empty.
    #[test]
    fn null_and_empty_tools_and_stop_load_as_empty() {
        for value in [
            serde_json::json!({"model": "m", "tools": null, "stop": null}),
            serde_json::json!({"model": "m", "tools": [], "stop": []}),
            serde_json::json!({"model": "m"}),
        ] {
            let request: ChatRequest = serde_json::from_value(value.clone()).unwrap();
            assert_eq!(request, ChatRequest::new("m"), "{value}");
        }
    }

    #[test]
    fn extend_messages_appends() {
        let history = [Message::user("earlier"), Message::assistant("reply")];
        let request = ChatRequest::new("m")
            .user("first")
            .extend_messages(history)
            .user("last");
        assert_eq!(
            request.messages,
            [
                Message::user("first"),
                Message::user("earlier"),
                Message::assistant("reply"),
                Message::user("last"),
            ]
        );
    }

    #[test]
    fn system_appends_and_collection_setters_replace() {
        let request = ChatRequest::new("m")
            .system("one")
            .system("two")
            .stop(["a"])
            .stop(["b"])
            .tools([ToolDefinition::new("x", json!({"type": "object"}))])
            .tools([ToolDefinition::new("y", json!({"type": "object"}))]);
        assert_eq!(request.system.len(), 2);
        assert_eq!(request.stop, ["b"]);
        let names: Vec<_> = request.tools.iter().map(|t| &*t.name).collect();
        assert_eq!(names, ["y"]);
    }

    #[test]
    fn effort_nearest() {
        use ReasoningEffort::*;
        assert_eq!(XHigh.nearest(&[Low, Medium, High]), Some(High));
        assert_eq!(Minimal.nearest(&[Low, Medium, High]), Some(Low));
        assert_eq!(None.nearest(&[Low, High]), Some(Low));
        assert_eq!(Medium.nearest(&[Low, High]), Some(High));
        assert_eq!(Medium.nearest(&[]), Option::None);
    }

    #[test]
    fn effort_all_is_a_sorted_slice() {
        let all: &'static [ReasoningEffort] = ReasoningEffort::ALL;
        assert_eq!(all.len(), 6);
        assert!(all.windows(2).all(|w| w[0] < w[1]));
        let names: Vec<_> = ReasoningEffort::ALL.iter().map(|e| e.as_str()).collect();
        assert_eq!(names, ["none", "minimal", "low", "medium", "high", "xhigh"]);
    }

    /// An adapter that builds its body with `json!` sends sampling values as
    /// written: `0.2`, not `0.20000000298023224`.
    #[test]
    fn sampling_values_encode_with_json_as_written() {
        let request = ChatRequest::new("m")
            .temperature(0.2)
            .top_p(0.9)
            .frequency_penalty(0.1)
            .presence_penalty(-0.3);
        let mut body = json!({"model": request.model});
        body["temperature"] = json!(request.temperature);
        body["top_p"] = json!(request.top_p);
        body["frequency_penalty"] = json!(request.frequency_penalty);
        body["presence_penalty"] = json!(request.presence_penalty);
        let sent = body.to_string();
        for written in [
            r#""temperature":0.2"#,
            r#""top_p":0.9"#,
            r#""frequency_penalty":0.1"#,
            r#""presence_penalty":-0.3"#,
        ] {
            assert!(sent.contains(written), "{written} in {sent}");
        }
    }
}
