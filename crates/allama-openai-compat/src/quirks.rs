//! Dialect differences between OpenAI-compatible endpoints.

use std::borrow::Cow;

use allama::Error;
use allama::adapter::chat::ToolCallIdFormat;
use allama_http::Headers;

use crate::error::classify_http_error;
use crate::schema::SchemaDialect;

/// Which request field carries the output token cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MaxTokensField {
    /// `max_tokens` (legacy; most compatible endpoints).
    MaxTokens,
    /// `max_completion_tokens` (OpenAI; required by reasoning models).
    MaxCompletionTokens,
}

/// Which request field carries the sampling seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SeedField {
    /// `seed`.
    Seed,
    /// `random_seed` (Mistral).
    RandomSeed,
    /// The endpoint has no seed parameter.
    Unsupported,
}

/// How one OpenAI-compatible endpoint deviates from OpenAI Chat Completions.
///
/// Presets in [`crate::providers`] configure these; custom endpoints start
/// from [`CompatQuirks::default`] (a conservative OpenAI-like dialect).
/// Unsupported portable features are reported as warnings (or errors under
/// `UnsupportedPolicy::Error`) rather than sent and rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct CompatQuirks {
    /// Provider-data namespace (origin key) for state this endpoint mints,
    /// e.g. `"openai"` or `"openai-compat:groq"`.
    pub namespace: Cow<'static, str>,
    /// Field used for `ChatRequest::max_output_tokens`.
    pub max_tokens_field: MaxTokensField,
    /// Field used for `ChatRequest::seed`.
    pub seed_field: SeedField,
    /// Send `stream_options.include_usage` when streaming. When `false`,
    /// streams carry no usage, and the provider answers
    /// [`ChatCapability::StreamUsage`](allama::ChatCapability::StreamUsage)
    /// `Unsupported` unless it declares otherwise.
    pub stream_usage: bool,
    /// Whether a `json_schema` response format constrains decoding: only
    /// when sent with `strict: true` (OpenAI's strict mode, the default),
    /// always (Ollama, vLLM), or never. See [`SchemaConstraint`].
    pub response_format_constraint: SchemaConstraint,
    /// Whether a tool's parameters schema constrains the call's arguments:
    /// only when the tool is sent with `strict: true` (the default), always,
    /// or never. See [`SchemaConstraint`] and
    /// [`forced_tool_choice_constrained`](Self::forced_tool_choice_constrained).
    pub tool_constraint: SchemaConstraint,
    /// A forced `tool_choice` (`required`, or a named tool) constrains the
    /// forced call's arguments to the tool's schema, strict or not (vLLM's
    /// structured outputs). A strict tool the endpoint does not otherwise
    /// constrain ([`SchemaConstraint::Never`]) is then encoded as under
    /// [`SchemaConstraint::Always`] while the choice forces it, instead of
    /// being reported `Unsupported`. Defaults to `false`: OpenAI forces the
    /// call, and constrains its arguments only for a strict tool.
    pub forced_tool_choice_constrained: bool,
    /// The strict-mode dialect of the endpoint's grammar engine. Strict
    /// schemas are rewritten into it with
    /// [`strict_schema`](crate::schema::strict_schema); one it cannot
    /// express is sent without strict and reported as an unsupported hint
    /// (an error under `UnsupportedPolicy::Error`), by
    /// [`apply_strict_policy`](allama::adapter::schema::apply_strict_policy). `None` sends strict
    /// schemas as written, for engines with rules of their own, with an
    /// `Approximated` warning that the schema went out unchecked (extraction
    /// then reports `SchemaEnforcement::Approximate`). Defaults to
    /// [`SchemaDialect::OpenAiStrict`].
    pub schema_dialect: Option<SchemaDialect>,
    /// The endpoint documents that its grammar engine enforces
    /// [`schema_dialect`](Self::schema_dialect) as written. When `false`
    /// (the default for [`CompatQuirks::new`], a custom endpoint), a strict
    /// schema is still rewritten into the dialect, but with an
    /// `Approximated` warning that what the endpoint enforces is not
    /// verified, so extraction reports `SchemaEnforcement::Approximate`,
    /// never `Exact`. `true` for [`CompatQuirks::openai`] and for presets
    /// whose strict mode is documented.
    pub dialect_verified: bool,
    /// The endpoint takes only an object at the root of a JSON schema
    /// response format and of a function's parameters, strict or not. Any
    /// other root (an enum, a union, no `type`) is refused with
    /// `Error::Unsupported` before sending, which lets extraction fall back
    /// or fail clearly instead of meeting an HTTP 400. A local `$ref` to an
    /// object schema, and the empty schema `{}`, are object roots. True for
    /// [`CompatQuirks::openai`] (verified live, 2026-10-13); defaults to
    /// `false`, which sends any root.
    pub object_schema_root: bool,
    /// Follow OpenAI's fine-tuned model convention: a model id starting
    /// with `ft:` selects [`SchemaDialect::OpenAiStrictFineTuned`]
    /// ([`SchemaDialect::for_model`]), which removes more keywords and turns
    /// parallel tool calls off when a strict tool is sent. True only for
    /// [`CompatQuirks::openai`]; other vendors' `ft:` models (Mistral's)
    /// keep the configured dialect.
    pub fine_tuned_dialect: bool,
    /// Accepts `reasoning_effort`.
    pub supports_reasoning_effort: bool,
    /// Accepts `top_k`, which OpenAI's Chat Completions does not have but
    /// some servers take as an extension (vLLM, Together, Fireworks,
    /// DeepInfra, OpenRouter). When `false` (the default), a request's
    /// `top_k` is dropped as an unsupported hint.
    pub supports_top_k: bool,
    /// Accepts `parallel_tool_calls`. When `false`, the field is never sent:
    /// a caller's `true` is permission the model has anyway, so it is
    /// dropped silently; a caller's `false` is reported as an unsupported
    /// hint. The encoder then never sends the
    /// automatic `parallel_tool_calls: false` that dialects whose strict
    /// mode breaks under parallel calls otherwise get
    /// ([`SchemaDialect::parallel_tool_calls_break_strict`]). Set it to
    /// `false` for a deployment that rejects the parameter, such as an
    /// Azure reasoning deployment without parallel tool calls.
    pub supports_parallel_tool_calls: bool,
    /// Accepts the `developer` role for system instructions.
    pub supports_developer_role: bool,
    /// Honors `tool_choice`. When `false` (an endpoint that ignores it),
    /// `ToolChoice::Auto` (the default behavior) is omitted,
    /// `ToolChoice::Disabled` leaves the tools out of the request, which has
    /// the same effect on the turn, with an `Approximated` warning (the
    /// model no longer sees their definitions), and a forced choice fails
    /// with `Error::Unsupported`.
    pub supports_tool_choice: bool,
    /// Accepts `prompt_cache_key`. When `false` (the default), a request's
    /// cache key ([`PromptCache::key`](allama::PromptCache::key)) is
    /// dropped as an unsupported hint.
    pub supports_prompt_cache_key: bool,
    /// Accepts `prompt_cache_retention`, independently of the key: Mistral
    /// takes a key and no retention. When `false` (the default), a
    /// request's retention, or a cache breakpoint's long retention, is
    /// dropped as an unsupported hint.
    pub supports_prompt_cache_retention: bool,
    /// Accepts `json_schema` response formats. When `false`,
    /// `ResponseFormat::JsonSchema` fails with `Error::Unsupported` before
    /// sending, and the provider answers
    /// [`StructuredOutput`](allama::ChatCapability::StructuredOutput)
    /// `Unsupported` unless it declares otherwise; JSON mode
    /// (`ResponseFormat::Json`, sent as `json_object`) is still sent.
    pub supports_json_schema: bool,
    /// The assistant-message field that carries reasoning text back to the
    /// endpoint in later turns (some reasoning endpoints require it for
    /// multi-turn tool use), or [`ReasoningReplayField::Disabled`] (the
    /// default) to send none: the endpoint's own reasoning is then dropped
    /// silently, and other reasoning with a warning.
    pub reasoning_replay: ReasoningReplayField,
    /// Replays a [`Refusal`](allama::Refusal) block in the assistant
    /// message's `refusal` field, as OpenAI accepts it
    /// (<https://platform.openai.com/docs/api-reference/chat/create>, the
    /// assistant message's `refusal`). When `false`, a refusal is replayed
    /// as assistant text, so an endpoint that does not know the field
    /// never loses it.
    pub replay_refusal: bool,
    /// Drop all-zero usage chunks some endpoints emit after the real usage.
    pub drop_zero_usage_chunks: bool,
    /// Error codes (or error types) this endpoint uses for quota or billing
    /// exhaustion, on top of OpenAI's (`insufficient_quota`, …). Errors
    /// with these codes map to `Error::QuotaExhausted`, which retrying
    /// cannot fix, instead of a retryable rate limit.
    ///
    /// ```
    /// use allama_openai_compat::CompatQuirks;
    ///
    /// let mut quirks = CompatQuirks::new("openai-compat:vendor");
    /// quirks.quota_error_codes.push("quota_exceeded".into());
    /// # let _ = quirks;
    /// ```
    pub quota_error_codes: Vec<Cow<'static, str>>,
    /// The embeddings field that carries the portable query/document
    /// [`input_type`](allama::EmbeddingRequest::input_type), or
    /// `None` when the endpoint has none: the provider then reports
    /// [`EmbeddingCapability::InputType`](allama::EmbeddingCapability::InputType)
    /// `Unsupported` and the hint is dropped with a warning.
    ///
    /// ```
    /// use allama_openai_compat::{CompatQuirks, EmbeddingInputTypeField};
    ///
    /// // Voyage AI: `input_type` is `query` or `document`.
    /// let mut quirks = CompatQuirks::new("openai-compat:voyage");
    /// quirks.embedding_input_type =
    ///     Some(EmbeddingInputTypeField::new("input_type", "query", "document"));
    /// # let _ = quirks;
    /// ```
    pub embedding_input_type: Option<EmbeddingInputTypeField>,
    /// The tool call ids the endpoint accepts. Ids from another provider
    /// that it would refuse are mapped to ones it accepts, the same way in
    /// the assistant's `tool_calls` and in the `tool` messages that answer
    /// them, so a conversation continues here after a failover (see
    /// [`ToolCallIdFormat::conform`]). Defaults to
    /// [`ToolCallIdFormat::MaxLength`]`(64)`, OpenAI's limit (verified live:
    /// 65 characters are refused); the [`Mistral`](crate::providers::Mistral)
    /// preset uses [`ToolCallIdFormat::Alphanumeric`]`(9)`. A model that
    /// takes other ids than its endpoint sets its own in
    /// [`ChatModelRules::tool_call_id_format`](crate::ChatModelRules::tool_call_id_format)
    /// (the [`Cloudflare`](crate::providers::Cloudflare) preset does for
    /// Mistral models).
    ///
    /// Set `Alphanumeric(9)` for a server that runs Mistral models through
    /// `mistral_common` (vLLM with the Mistral tokenizer), which refuse
    /// every other id:
    ///
    /// ```
    /// use allama_openai_compat::{CompatQuirks, ToolCallIdFormat};
    ///
    /// let mut quirks = CompatQuirks::new("openai-compat:vllm");
    /// quirks.tool_call_id_format = ToolCallIdFormat::Alphanumeric(9);
    /// # let _ = quirks;
    /// ```
    pub tool_call_id_format: ToolCallIdFormat,
}

impl CompatQuirks {
    /// A conservative OpenAI-like dialect under `namespace`.
    ///
    /// Strict schemas are rewritten into OpenAI's dialect, but the
    /// endpoint's enforcement of it is not [verified](Self::dialect_verified),
    /// so a strict schema is reported `Approximated`.
    #[must_use]
    pub fn new(namespace: impl Into<Cow<'static, str>>) -> Self {
        Self {
            namespace: namespace.into(),
            max_tokens_field: MaxTokensField::MaxTokens,
            seed_field: SeedField::Seed,
            stream_usage: true,
            response_format_constraint: SchemaConstraint::WhenStrict,
            tool_constraint: SchemaConstraint::WhenStrict,
            forced_tool_choice_constrained: false,
            schema_dialect: Some(SchemaDialect::OpenAiStrict),
            dialect_verified: false,
            object_schema_root: false,
            fine_tuned_dialect: false,
            supports_reasoning_effort: true,
            supports_top_k: false,
            supports_parallel_tool_calls: true,
            supports_developer_role: false,
            supports_prompt_cache_key: false,
            supports_prompt_cache_retention: false,
            supports_json_schema: true,
            reasoning_replay: ReasoningReplayField::Disabled,
            replay_refusal: false,
            supports_tool_choice: true,
            drop_zero_usage_chunks: false,
            quota_error_codes: Vec::new(),
            embedding_input_type: None,
            tool_call_id_format: OPENAI_TOOL_CALL_IDS,
        }
    }

    /// The OpenAI dialect itself.
    #[must_use]
    pub fn openai() -> Self {
        Self {
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            fine_tuned_dialect: true,
            // https://platform.openai.com/docs/guides/structured-outputs
            dialect_verified: true,
            object_schema_root: true,
            supports_developer_role: true,
            supports_prompt_cache_key: true,
            supports_prompt_cache_retention: true,
            replay_refusal: true,
            ..Self::new("openai")
        }
    }

    /// The strict-schema dialect for `model`: [`schema_dialect`](Self::schema_dialect),
    /// with the fine-tuned variant for `ft:` models when
    /// [`fine_tuned_dialect`](Self::fine_tuned_dialect) is set.
    #[must_use]
    pub fn schema_dialect_for(&self, model: &str) -> Option<SchemaDialect> {
        effective_dialect(self.schema_dialect, self.fine_tuned_dialect, model)
    }

    /// Map a structured API error in this dialect: the built-in
    /// classification plus [`quota_error_codes`](Self::quota_error_codes).
    #[cfg(test)]
    #[must_use]
    pub(crate) fn map_api_error(
        &self,
        error: &crate::wire::ApiError,
        status: Option<u16>,
        request_id: Option<String>,
        retry_after: Option<std::time::Duration>,
    ) -> Error {
        crate::error::classify_api_error(
            error,
            status,
            request_id,
            retry_after,
            &self.quota_error_codes,
        )
    }

    /// Map an HTTP error response in this dialect: the built-in
    /// classification of OpenAI-style error envelopes (`{"error": {…}}`,
    /// with `code`, `type` and `param`) plus
    /// [`quota_error_codes`](Self::quota_error_codes).
    ///
    /// The provider's own classification; a provider crate built on this
    /// one, or a conformance test, maps a recorded failure with it.
    #[must_use]
    pub fn map_http_error(&self, status: u16, headers: &Headers, body: &str) -> Error {
        classify_http_error(status, headers, body, &self.quota_error_codes)
    }
}

impl Default for CompatQuirks {
    fn default() -> Self {
        Self::new("openai-compat")
    }
}

/// Whether an endpoint constrains decoding to a JSON Schema in one
/// position: the response format
/// ([`CompatQuirks::response_format_constraint`]) or a tool's parameters
/// ([`CompatQuirks::tool_constraint`]).
///
/// A schema the caller marks strict ([`allama::ResponseFormat::JsonSchema`]'s
/// `strict`, [`allama::ToolDefinition::strict`]) asks for constraining.
/// What the endpoint does decides what is sent and what is reported:
///
/// | | `strict` sent | strict schema reported |
/// |---|---|---|
/// | `WhenStrict` | the caller's value | as the [dialect](CompatQuirks::schema_dialect) rewrites it |
/// | `Always` | never | as the dialect rewrites it |
/// | `Never` | never | `Unsupported`, sent non-strict |
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SchemaConstraint {
    /// Decoding is constrained to the schema when it is sent with
    /// `strict: true`, and only then (OpenAI's strict mode).
    #[default]
    WhenStrict,
    /// Decoding is constrained to every schema in this position, and the
    /// endpoint takes no `strict` flag: Ollama's `format` and vLLM's
    /// structured outputs constrain every `json_schema` response format.
    Always,
    /// Decoding is not constrained to the schema. A strict schema goes out
    /// without `strict`, with an `Unsupported` warning (an error under
    /// `UnsupportedPolicy::Error`).
    Never,
}

/// How an embeddings endpoint spells the portable query/document
/// [`input_type`](allama::EmbeddingRequest::input_type): the request field
/// and its value for each role.
///
/// The field is added to the request body next to the standard fields. An
/// endpoint-specific `extra` field of the same name, set through the
/// request options, takes precedence with an `Approximated` warning.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct EmbeddingInputTypeField {
    /// The request field, for example `input_type`.
    pub name: Cow<'static, str>,
    /// The value for [`EmbeddingInputType::Query`](allama::EmbeddingInputType::Query).
    pub query: Cow<'static, str>,
    /// The value for [`EmbeddingInputType::Document`](allama::EmbeddingInputType::Document).
    pub document: Cow<'static, str>,
}

impl EmbeddingInputTypeField {
    /// The field `name` with the values for queries and documents.
    #[must_use]
    pub fn new(
        name: impl Into<Cow<'static, str>>,
        query: impl Into<Cow<'static, str>>,
        document: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self {
            name: name.into(),
            query: query.into(),
            document: document.into(),
        }
    }
}

/// How one endpoint deviates from the OpenAI Responses API.
///
/// The Responses counterpart of [`CompatQuirks`], kept separate because one
/// host can support different features on its two wire APIs (DeepSeek
/// honors `json_schema` on Responses but not on Chat Completions). Custom
/// endpoints start from [`ResponsesQuirks::new`], a conservative dialect.
/// Unsupported portable features are reported as warnings (or errors under
/// `UnsupportedPolicy::Error`) rather than sent and rejected.
///
/// What a *model* accepts (reasoning effort levels, sampling parameters
/// while reasoning) is not a dialect fact; see
/// [`ResponsesModelRules`](crate::ResponsesModelRules).
///
/// ```
/// use allama_openai_compat::ResponsesQuirks;
///
/// let mut quirks = ResponsesQuirks::new("openai-compat:vllm");
/// quirks.function_call_status_null = true;
/// # let _ = quirks;
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ResponsesQuirks {
    /// Provider-data namespace (origin key) for state this endpoint mints:
    /// reasoning replay state, item ids, annotations, and the origin of
    /// `Other` content. Only state under this namespace is replayed, so
    /// each endpoint needs its own, e.g. `"openai"` or
    /// `"openai-compat:vllm"`.
    pub namespace: Cow<'static, str>,
    /// Whether reasoning is replayed in later turns.
    pub reasoning_replay: ReasoningReplay,
    /// Which request modes the endpoint serves.
    pub streaming: StreamingSupport,
    /// Accepts `strict` on function tools and `json_schema` text formats.
    /// When `true`, `strict` is always sent explicitly (OpenAI treats tools
    /// that omit it as strict). When `false`, it is omitted, schemas are
    /// sent as written, and a strict tool or schema is reported as an
    /// unsupported hint.
    pub supports_strict: bool,
    /// The strict-mode dialect of the endpoint's grammar engine; see
    /// [`CompatQuirks::schema_dialect`]. Defaults to
    /// [`SchemaDialect::OpenAiStrict`].
    pub schema_dialect: Option<SchemaDialect>,
    /// The endpoint documents that its grammar engine enforces
    /// [`schema_dialect`](Self::schema_dialect) as written; see
    /// [`CompatQuirks::dialect_verified`]. `false` for
    /// [`ResponsesQuirks::new`], `true` for [`ResponsesQuirks::openai`].
    pub dialect_verified: bool,
    /// Follow OpenAI's fine-tuned model convention (`ft:` ids select
    /// [`SchemaDialect::OpenAiStrictFineTuned`]); see
    /// [`CompatQuirks::fine_tuned_dialect`]. True only for
    /// [`ResponsesQuirks::openai`].
    pub fine_tuned_dialect: bool,
    /// The endpoint takes only an object at the root of a `json_schema`
    /// text format and of a function's parameters; see
    /// [`CompatQuirks::object_schema_root`]. True for
    /// [`ResponsesQuirks::openai`], `false` for [`ResponsesQuirks::new`].
    pub object_schema_root: bool,
    /// Requires `"status": null` on replayed `function_call` items (older
    /// vLLM releases and NVIDIA NIM).
    pub function_call_status_null: bool,
    /// Accepts function calls interleaved with other assistant items. When
    /// `false`, the function calls of an assistant turn are replayed after
    /// its other items (DeepSeek).
    pub interleaved_function_calls: bool,
    /// Accepts `json_schema` text formats. When `false`,
    /// `ResponseFormat::JsonSchema` fails with `Error::Unsupported` before
    /// sending, and the provider answers
    /// [`StructuredOutput`](allama::ChatCapability::StructuredOutput)
    /// `Unsupported` unless it declares otherwise; JSON mode
    /// (`ResponseFormat::Json`, sent as `json_object`) is still sent.
    pub supports_json_schema: bool,
    /// Accepts `developer` input messages; otherwise developer system
    /// prompts are sent as `system` with a warning.
    pub supports_developer_role: bool,
    /// Accepts `parallel_tool_calls` (otherwise `true` is dropped silently,
    /// `false` is reported as an unsupported hint, and the automatic
    /// `false` for strict tools is not sent; see
    /// [`CompatQuirks::supports_parallel_tool_calls`]).
    pub supports_parallel_tool_calls: bool,
    /// Accepts `prompt_cache_key` / `prompt_cache_retention` (otherwise
    /// reported as an unsupported hint).
    pub supports_prompt_cache_key: bool,
    /// Accepts `tool_choice`. When `false`, `ToolChoice::Auto` (the
    /// default behavior) is omitted and any other choice fails with
    /// `Error::Unsupported`, since forcing or disabling tools changes what
    /// the model may do.
    pub supports_tool_choice: bool,
    /// The `call_id`s the endpoint accepts; see
    /// [`CompatQuirks::tool_call_id_format`]. Defaults to
    /// [`ToolCallIdFormat::MaxLength`]`(64)`, OpenAI's limit (verified live).
    pub tool_call_id_format: ToolCallIdFormat,
}

impl ResponsesQuirks {
    /// A conservative dialect under `namespace`: OpenAI's Responses wire
    /// format, without the `developer` role or prompt-cache keys.
    #[must_use]
    pub fn new(namespace: impl Into<Cow<'static, str>>) -> Self {
        Self {
            namespace: namespace.into(),
            reasoning_replay: ReasoningReplay::AsReceived,
            streaming: StreamingSupport::Both,
            supports_strict: true,
            schema_dialect: Some(SchemaDialect::OpenAiStrict),
            dialect_verified: false,
            fine_tuned_dialect: false,
            object_schema_root: false,
            function_call_status_null: false,
            interleaved_function_calls: true,
            supports_json_schema: true,
            supports_developer_role: false,
            supports_parallel_tool_calls: true,
            supports_prompt_cache_key: false,
            supports_tool_choice: true,
            tool_call_id_format: OPENAI_TOOL_CALL_IDS,
        }
    }

    /// The OpenAI dialect itself (namespace `"openai"`).
    #[must_use]
    pub fn openai() -> Self {
        Self {
            fine_tuned_dialect: true,
            // https://platform.openai.com/docs/guides/structured-outputs
            dialect_verified: true,
            object_schema_root: true,
            supports_developer_role: true,
            supports_prompt_cache_key: true,
            ..Self::new("openai")
        }
    }

    /// The strict-schema dialect for `model`; see
    /// [`CompatQuirks::schema_dialect_for`].
    #[must_use]
    pub fn schema_dialect_for(&self, model: &str) -> Option<SchemaDialect> {
        effective_dialect(self.schema_dialect, self.fine_tuned_dialect, model)
    }
}

impl Default for ResponsesQuirks {
    fn default() -> Self {
        Self::new("openai-compat")
    }
}

/// Whether reasoning items are replayed to the endpoint in later turns.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReasoningReplay {
    /// Replay the reasoning state the endpoint returned under this
    /// namespace: `encrypted_content` (OpenAI, OpenRouter), raw reasoning
    /// `content` (vLLM, llama.cpp), or both. Reasoning with neither cannot
    /// be replayed statelessly and is dropped with a warning.
    #[default]
    AsReceived,
    /// Never replay reasoning; every reasoning block is dropped with a
    /// warning.
    Disabled,
}

/// The Chat Completions assistant-message field that replays reasoning
/// text ([`CompatQuirks::reasoning_replay`]).
///
/// Endpoints disagree on the field: DeepSeek and vLLM take
/// `reasoning_content`, Ollama emits `reasoning`, OpenRouter takes its
/// structured `reasoning_details` back, and GitHub Copilot takes
/// `reasoning_text` with a `reasoning_opaque` signature. Reasoning from another provider is
/// never replayed, whichever field is set, and reasoning this endpoint
/// returned is dropped without a warning when no field is set.
///
/// ```
/// use allama_openai_compat::{CompatQuirks, ReasoningReplayField};
///
/// let mut quirks = CompatQuirks::new("openai-compat:vendor");
/// quirks.reasoning_replay = ReasoningReplayField::ReasoningContent;
/// # let _ = quirks;
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReasoningReplayField {
    /// No reasoning field is sent: reasoning the endpoint returned is
    /// dropped silently, and other reasoning with text with a warning.
    ///
    /// Either the endpoint takes no reasoning back, or it takes it in a
    /// form no field here sends. Mistral asks for its `thinking` content
    /// chunks back in the assistant content, so the `Mistral` preset, which
    /// keeps this default, drops Magistral's reasoning on replay.
    #[default]
    Disabled,
    /// `reasoning_content` (DeepSeek, vLLM, …).
    ReasoningContent,
    /// `reasoning` (Ollama, …).
    Reasoning,
    /// `reasoning_details` (OpenRouter): the structured reasoning the
    /// endpoint returned, encrypted reasoning and signatures included, sent
    /// back as it came. Reasoning without details (from elsewhere, or
    /// written by hand) is dropped with a warning.
    ReasoningDetails,
    /// `reasoning_text`, with the `reasoning_opaque` signature the
    /// endpoint returned (GitHub Copilot). The turn's reasoning text goes
    /// back joined, with its last signature: the endpoint sends a new one
    /// before each tool call, and its own editor clients replay only the
    /// latest.
    ReasoningText,
}

/// Which request modes an endpoint serves.
///
/// Requesting the other mode fails with `Error::Unsupported` before
/// anything is sent; nothing is emulated.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum StreamingSupport {
    /// Streaming and non-streaming requests.
    #[default]
    Both,
    /// Streaming requests only.
    StreamOnly,
    /// Non-streaming requests only.
    NonStreamingOnly,
}

impl StreamingSupport {
    /// Whether a request in this mode is served.
    #[must_use]
    pub fn allows(self, stream: bool) -> bool {
        match self {
            Self::Both => true,
            Self::StreamOnly => stream,
            Self::NonStreamingOnly => !stream,
        }
    }
}

/// OpenAI's tool call id rule, on Chat Completions (`tool_calls[].id`) and
/// Responses (`call_id`) alike: any characters, at most 64 (a longer id is
/// a 400 `string_above_max_length`, verified live on 2026-10-07).
const OPENAI_TOOL_CALL_IDS: ToolCallIdFormat = ToolCallIdFormat::MaxLength(64);

fn effective_dialect(
    dialect: Option<SchemaDialect>,
    fine_tuned: bool,
    model: &str,
) -> Option<SchemaDialect> {
    dialect.map(|dialect| {
        if fine_tuned {
            dialect.for_model(model)
        } else {
            dialect
        }
    })
}
