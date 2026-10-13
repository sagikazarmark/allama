//! What the request field audit probes, and the list that ties every
//! serialized [`ChatRequest`] key to a probe or an exemption.

use std::fmt;

use allama::{
    CacheBreakpoint, CacheRetention, CapabilitySupport, ChatCapability, ChatProvider, ChatRequest,
    ContentBlock, ContentPart, File, Image, ImageDetail, Message, OtherContent, PromptCache,
    ProviderData, ProviderToolCall, ProviderToolResult, Reasoning, ReasoningConfig,
    ReasoningEffort, ReasoningOutput, RequestField, Subject, SystemPrompt, Text, ToolDefinition,
    UserMessage,
};
use serde_json::{Value, json};

use crate::e2e::{PINEAPPLE_PDF_BASE64, RED_PNG_BASE64};

/// One check of the request field audit,
/// [`assert_request_fields_accounted`](super::assert_request_fields_accounted):
/// what [`ACCOUNTED_FIELDS`] names for a probed field, and what the audit's
/// `known_gaps` lists.
///
/// Each check sends one or more requests that set what it probes, through
/// the adapter's probe closure. Hints and metadata pass when they reach the
/// wire, are refused with [`Error::Unsupported`](allama::Error::Unsupported),
/// or are named by a warning; state from another provider passes when it
/// never reaches the wire. [`describe`](Self::describe) says what each check
/// expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Probe {
    /// `ChatRequest::temperature`.
    Temperature,
    /// `ChatRequest::top_p`.
    TopP,
    /// `ChatRequest::top_k`.
    TopK,
    /// `ChatRequest::max_output_tokens`.
    MaxOutputTokens,
    /// `ChatRequest::stop`.
    Stop,
    /// `ChatRequest::seed`.
    Seed,
    /// `ChatRequest::frequency_penalty`.
    FrequencyPenalty,
    /// `ChatRequest::presence_penalty`.
    PresencePenalty,
    /// `SystemPrompt::content`.
    SystemPrompt,
    /// `SystemPrompt::cache`.
    SystemCache,
    /// Cache breakpoints on user, assistant, and tool messages.
    MessageCache,
    /// `ToolDefinition::cache`.
    ToolCache,
    /// `ChatRequest::cache`: the prompt-cache key and retention.
    PromptCache,
    /// `Image::detail` on input images.
    ImageDetail,
    /// The source (the image itself) of input and replayed images.
    ImageSource,
    /// `ChatRequest::response_format` and its schema.
    ResponseFormat,
    /// `File::filename` on input files.
    FileName,
    /// `ToolDefinition::strict`.
    ToolStrict,
    /// `ToolMessage::is_error`.
    ToolResultError,
    /// `ChatRequest::parallel_tool_calls`.
    ParallelToolCalls,
    /// `ChatRequest::reasoning`: effort and output visibility.
    ReasoningConfig,
    /// Reasoning whose replay state belongs to another provider.
    ForeignReasoning,
    /// `Other` content from another provider, in parts and blocks.
    ForeignOtherContent,
    /// Provider-executed tool calls and results from another provider.
    ForeignProviderTool,
    /// Provider data in another provider's namespace, on every item that
    /// carries it.
    ForeignProviderData,
    /// Application data on system prompts and messages.
    AppData,
}

impl Probe {
    /// Every probe, in the order the audit runs them.
    pub const ALL: &'static [Probe] = &[
        Self::Temperature,
        Self::TopP,
        Self::TopK,
        Self::MaxOutputTokens,
        Self::Stop,
        Self::Seed,
        Self::FrequencyPenalty,
        Self::PresencePenalty,
        Self::SystemPrompt,
        Self::SystemCache,
        Self::MessageCache,
        Self::ToolCache,
        Self::PromptCache,
        Self::ImageDetail,
        Self::ImageSource,
        Self::ResponseFormat,
        Self::FileName,
        Self::ToolStrict,
        Self::ToolResultError,
        Self::ParallelToolCalls,
        Self::ReasoningConfig,
        Self::ForeignReasoning,
        Self::ForeignOtherContent,
        Self::ForeignProviderTool,
        Self::ForeignProviderData,
        Self::AppData,
    ];

    /// Stable snake_case name, as [`ACCOUNTED_FIELDS`] spells it.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Temperature => "temperature",
            Self::TopP => "top_p",
            Self::TopK => "top_k",
            Self::MaxOutputTokens => "max_output_tokens",
            Self::Stop => "stop",
            Self::Seed => "seed",
            Self::FrequencyPenalty => "frequency_penalty",
            Self::PresencePenalty => "presence_penalty",
            Self::SystemPrompt => "system_prompt",
            Self::SystemCache => "system_cache",
            Self::MessageCache => "message_cache",
            Self::ToolCache => "tool_cache",
            Self::PromptCache => "prompt_cache",
            Self::ImageDetail => "image_detail",
            Self::ImageSource => "image_source",
            Self::ResponseFormat => "response_format",
            Self::FileName => "file_name",
            Self::ToolStrict => "tool_strict",
            Self::ToolResultError => "tool_result_error",
            Self::ParallelToolCalls => "parallel_tool_calls",
            Self::ReasoningConfig => "reasoning_config",
            Self::ForeignReasoning => "foreign_reasoning",
            Self::ForeignOtherContent => "foreign_other_content",
            Self::ForeignProviderTool => "foreign_provider_tool",
            Self::ForeignProviderData => "foreign_provider_data",
            Self::AppData => "app_data",
        }
    }

    /// The probe named `name` ([`as_str`](Self::as_str)), if any.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|probe| probe.as_str() == name)
    }

    /// What the probe expects of an adapter, in one sentence.
    #[must_use]
    pub fn describe(self) -> &'static str {
        match self {
            Self::FileName | Self::ToolResultError => {
                "sent, dropped with a `Dropped` warning that names it, or handled as a hint"
            }
            Self::ForeignReasoning | Self::ForeignOtherContent | Self::ForeignProviderTool => {
                "never sent; dropped with a `Dropped` warning that names it"
            }
            Self::ForeignProviderData => "never sent",
            Self::AppData => "never sent, and changes nothing that is sent or reported",
            _ => {
                "sent (or approximated with a warning), refused with `Unsupported`, or dropped \
                 with a warning that names it, in which case `UnsupportedPolicy::Error` refuses \
                 it before sending"
            }
        }
    }
}

impl fmt::Display for Probe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Every key path of a serialized [`ChatRequest`], each with the
/// [`Probe`]s that check it or the reason the audit does not probe it.
///
/// One entry per line, `<path>  probe: <probe>[, <probe>...]` or
/// `<path>  exempt: <reason>`; `#` starts a comment. Paths follow the
/// serialized form: arrays add `[]`, and an object tagged with `role` or
/// `type` adds its tag as a segment, so `messages[].user.cache{}` is
/// `UserMessage::cache`. A path ending in `[]` or `{}` accounts for the whole
/// array or object there.
///
/// The list is derived from `ChatRequest` itself: a test in `allama`
/// serializes a request with every field and every message, part, and block
/// kind set, and fails on any key path the list does not account for. A new
/// portable request field therefore arrives with a probe here or an explicit
/// exemption. [`accounted_fields`] parses the list.
///
/// The provider author guide lists the probed fields ("Fields you must
/// account for"). The exempt fields, generated from the list by a test in
/// `allama-examples`:
///
/// <!-- exempt-fields:start -->
///
/// | Exempt request field | Why C2 does not probe it |
/// | --- | --- |
/// | `model` | selects the model; transports may carry it outside the body, and every probe sends it |
/// | `system[].options{}` | provider options, read only by the adapter that owns the namespace (ADR 0006) |
/// | `messages[].user.content` | conversation content; never dropped (ADR 0003), and request fixtures pin its encoding |
/// | `messages[].user.content[].text.text` | conversation content; never dropped (ADR 0003), and request fixtures pin its encoding |
/// | `messages[].user.content[].text.citations[]` | citations are a read-only view of model output; adapters replay their own annotations (ADR 0007) |
/// | `messages[].user.content[].image.media_type` | adapter input naming the media type of a URL image; adapters that need one read it |
/// | `messages[].user.content[].file.media_type` | file input; refused when `FileInput` is unsupported (C3), and request fixtures pin its encoding |
/// | `messages[].user.content[].file.source{}` | file input; refused when `FileInput` is unsupported (C3), and request fixtures pin its encoding |
/// | `messages[].user.content[].other.provider_data{}` | gated by namespace on content replayed to its origin; `probe::assert_other_provider_data_gated` checks it |
/// | `messages[].assistant.content[].text.text` | conversation content; never dropped (ADR 0003), and request fixtures pin its encoding |
/// | `messages[].assistant.content[].text.citations[]` | citations are a read-only view of model output; adapters replay their own annotations (ADR 0007) |
/// | `messages[].assistant.content[].image.media_type` | adapter input naming the media type of a URL image; adapters that need one read it |
/// | `messages[].assistant.content[].image.detail` | requested fidelity applies to input images; a replayed output image is sent as it was returned |
/// | `messages[].assistant.content[].tool_call.id` | tool-call replay; request fixtures and the tool-call contracts pin its encoding |
/// | `messages[].assistant.content[].tool_call.name` | tool-call replay; request fixtures and the tool-call contracts pin its encoding |
/// | `messages[].assistant.content[].tool_call.arguments` | tool-call replay; request fixtures and the tool-call contracts pin its encoding |
/// | `messages[].assistant.content[].refusal.text` | conversation content; replayed in the provider's refusal form or as assistant text, never dropped, and replay tests pin its encoding |
/// | `messages[].assistant.content[].other.provider_data{}` | gated by namespace on content replayed to its origin; `probe::assert_other_provider_data_gated` checks it |
/// | `messages[].tool.tool_call_id` | tool-result correlation; request fixtures and the tool-call contracts pin its encoding |
/// | `messages[].tool.name` | tool-result correlation; request fixtures and the tool-call contracts pin its encoding |
/// | `messages[].tool.content` | conversation content; never dropped (ADR 0003), and request fixtures pin its encoding |
/// | `messages[].tool.content{}` | conversation content; never dropped (ADR 0003), and request fixtures pin its encoding |
/// | `messages[].tool.content[].text.text` | conversation content; never dropped (ADR 0003), and request fixtures pin its encoding |
/// | `messages[].tool.content[].text.citations[]` | citations are a read-only view of model output; adapters replay their own annotations (ADR 0007) |
/// | `messages[].tool.content[].image.media_type` | adapter input naming the media type of a URL image; adapters that need one read it |
/// | `messages[].tool.content[].file.media_type` | file input; refused when `FileInput` is unsupported (C3), and request fixtures pin its encoding |
/// | `messages[].tool.content[].file.source{}` | file input; refused when `FileInput` is unsupported (C3), and request fixtures pin its encoding |
/// | `messages[].tool.content[].other.provider_data{}` | gated by namespace on content replayed to its origin; `probe::assert_other_provider_data_gated` checks it |
/// | `tools[].name` | tool definition; refused when `ToolCalls` is unsupported (C3), and request fixtures pin its encoding |
/// | `tools[].description` | tool definition; refused when `ToolCalls` is unsupported (C3), and request fixtures pin its encoding |
/// | `tools[].parameters{}` | tool definition; refused when `ToolCalls` is unsupported (C3), and request fixtures pin its encoding |
/// | `tool_choice` | tool choice; a forced choice is refused when `ForcedToolChoice` is unsupported (C3), and request fixtures pin it |
/// | `tool_choice.specific.name` | tool choice; a forced choice is refused when `ForcedToolChoice` is unsupported (C3), and request fixtures pin it |
/// | `response_format.json_schema.name` | a label for the schema, which some providers have no field for; the schema itself is probed |
/// | `response_format.json_schema.strict` | implied where structured output is always constrained; `probe::assert_strict_not_honoured_warns` checks configurations without strict mode |
/// | `cache.automatic` | an intent that providers caching prefixes on their own satisfy without sending anything, so no wire evidence can judge it; adapter tests cover the opt-in mapping (Anthropic top-level `cache_control`) |
/// | `unsupported_policy` | tells the adapter how to report what it cannot honor; never sent |
/// | `options{}` | provider options, read only by the adapter that owns the namespace (ADR 0006) |
///
/// <!-- exempt-fields:end -->
pub const ACCOUNTED_FIELDS: &str = allama::adapter::chat::ACCOUNTED_FIELDS;

/// The entries of [`ACCOUNTED_FIELDS`], in order.
///
/// An entry this crate cannot read is left out: one that names a probe it
/// does not know, or an account other than `probe:` and `exempt:`. That
/// means `allama` is newer than `allama-conformance`, and
/// [`assert_request_fields_accounted`](super::assert_request_fields_accounted)
/// fails on it, asking for an upgrade.
#[must_use]
pub fn accounted_fields() -> Vec<AccountedField> {
    parse_accounted_fields(ACCOUNTED_FIELDS).0
}

/// Parses an accounted-fields list: the entries this crate reads, in
/// order, and a description of each entry it cannot read.
pub(crate) fn parse_accounted_fields(list: &'static str) -> (Vec<AccountedField>, Vec<String>) {
    let mut fields = Vec::new();
    let mut unread = Vec::new();
    for line in list
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
    {
        let Some((path, account)) = line.split_once(char::is_whitespace) else {
            unread.push(format!("`{line}` has no account"));
            continue;
        };
        let account = account.trim();
        let account = if let Some(probes) = account.strip_prefix("probe:") {
            let names: Vec<&str> = probes.split(',').map(str::trim).collect();
            let unknown: Vec<&str> = names
                .iter()
                .copied()
                .filter(|name| Probe::from_name(name).is_none())
                .collect();
            if !unknown.is_empty() {
                unread.push(format!(
                    "`{path}` names the probe(s) {}, which this allama-conformance does not know",
                    unknown
                        .iter()
                        .map(|name| format!("`{name}`"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
                continue;
            }
            Account::Probed(names.into_iter().filter_map(Probe::from_name).collect())
        } else if let Some(reason) = account.strip_prefix("exempt:") {
            Account::Exempt(reason.trim())
        } else {
            unread.push(format!(
                "`{path}` has an account this allama-conformance does not read: `{account}`"
            ));
            continue;
        };
        fields.push(AccountedField { path, account });
    }
    (fields, unread)
}

/// One entry of [`ACCOUNTED_FIELDS`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct AccountedField {
    /// The key path in a serialized [`ChatRequest`].
    pub path: &'static str,
    /// How the audit accounts for it.
    pub account: Account,
}

/// How the request field audit accounts for a key path.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Account {
    /// These probes check it.
    Probed(Vec<Probe>),
    /// The audit does not probe it, for this reason.
    Exempt(&'static str),
}

// --- cases -------------------------------------------------------------------

/// How a probe is judged.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Rule {
    /// A hint (ADR 0003): sent, refused with `Unsupported`, or named by a
    /// warning, in which case `UnsupportedPolicy::Error` must refuse it
    /// before sending.
    Hint,
    /// As [`Rule::Hint`], but a `Dropped` warning that names it also
    /// accounts for it.
    Droppable,
    /// State from another provider: never sent. `warn` requires a `Dropped`
    /// warning that names it. `refusal` says when refusing the whole
    /// request with `Unsupported` is accepted instead.
    Withheld { warn: bool, refusal: Refusal },
    /// Data adapters never read: never sent, and the request is sent and
    /// answered exactly as this request without it is.
    Inert {
        without: fn(&str, Supports) -> ChatRequest,
    },
}

/// When a case accepts a refusal of the whole request with `Unsupported`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Refusal {
    /// Never: the adapter must serve the request.
    Never,
    /// Always: the request needs a capability (tools, image or file input)
    /// the model may lack.
    Allowed,
    /// Only when this request, the same without the probed value, is
    /// refused too, so the refusal is not about the value (a stream path
    /// without tool calls, for example).
    Unrelated(fn(&str, Supports) -> ChatRequest),
}

/// Evidence that a probed value reached the wire.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Evidence {
    /// The value appears in a request body.
    Value(WireValue),
    /// The request body differs from the one sent for this contrasting
    /// request, which sets the probed field to another value (or not at
    /// all).
    Change(fn(&str) -> ChatRequest),
}

/// A distinctive value: finding it anywhere in a request body means it
/// reached the wire.
#[derive(Clone, Copy, Debug)]
pub(crate) enum WireValue {
    Number(f64),
    Text(&'static str),
}

impl fmt::Display for WireValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Number(n) => write!(f, "{n}"),
            Self::Text(s) => write!(f, "{s:?}"),
        }
    }
}

/// One request a probe sends.
pub(crate) struct Case {
    pub(crate) probe: Probe,
    /// Which use of the probed field, when a probe has several.
    pub(crate) label: &'static str,
    /// The request for a model, shaped to what the provider supports.
    pub(crate) request: fn(&str, Supports) -> ChatRequest,
    pub(crate) evidence: Evidence,
    pub(crate) rule: Rule,
    /// Whether a warning is about what the case probes.
    pub(crate) names: fn(&Subject) -> bool,
}

impl Case {
    pub(crate) fn refusal(&self) -> Refusal {
        match self.rule {
            Rule::Hint | Rule::Droppable => Refusal::Allowed,
            Rule::Withheld { refusal, .. } => refusal,
            Rule::Inert { .. } => Refusal::Never,
        }
    }

    pub(crate) fn name(&self) -> String {
        if self.label.is_empty() {
            self.probe.to_string()
        } else {
            format!("{} ({})", self.probe, self.label)
        }
    }
}

/// What the provider under audit serves for the probed model, from its
/// capability answers. A case whose rule does not accept a refusal shapes
/// its request to it, so a provider is never failed for refusing what it
/// reports it cannot serve.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Supports {
    /// [`ChatCapability::ToolCalls`] is not
    /// [`Unsupported`](CapabilitySupport::Unsupported): the conversation may
    /// hold tool calls and their results.
    pub(crate) tool_calls: bool,
}

impl Supports {
    pub(crate) fn of(provider: &impl ChatProvider, model: &str) -> Self {
        Self {
            tool_calls: provider.chat_capability(model, ChatCapability::ToolCalls)
                != CapabilitySupport::Unsupported,
        }
    }
}

/// Namespace and origin that no adapter owns.
const FOREIGN: &str = super::FOREIGN_PROBE_NAMESPACE;

/// The tool call that [`tool_turn`] answers.
const CALL_ID: &str = "call_allama_probe";

const HINT: Rule = Rule::Hint;
const DROPPABLE: Rule = Rule::Droppable;
const WITHHELD: Rule = Rule::Withheld {
    warn: false,
    refusal: Refusal::Never,
};
const WITHHELD_REFUSABLE: Rule = Rule::Withheld {
    warn: false,
    refusal: Refusal::Allowed,
};
const WITHHELD_WARN: Rule = Rule::Withheld {
    warn: true,
    refusal: Refusal::Never,
};

pub(crate) const CASES: &[Case] = &[
    // Sampling and generation fields.
    Case {
        probe: Probe::Temperature,
        label: "",
        request: |m, _| base(m).temperature(0.37),
        evidence: Evidence::Value(WireValue::Number(0.37)),
        rule: HINT,
        names: |s| is_field(s, RequestField::Temperature),
    },
    Case {
        probe: Probe::TopP,
        label: "",
        request: |m, _| base(m).top_p(0.83),
        evidence: Evidence::Value(WireValue::Number(0.83)),
        rule: HINT,
        names: |s| is_field(s, RequestField::TopP),
    },
    Case {
        probe: Probe::TopK,
        label: "",
        request: |m, _| base(m).top_k(37),
        evidence: Evidence::Value(WireValue::Number(37.0)),
        rule: HINT,
        names: |s| is_field(s, RequestField::TopK),
    },
    Case {
        probe: Probe::MaxOutputTokens,
        label: "",
        request: |m, _| base(m).max_output_tokens(1234),
        evidence: Evidence::Value(WireValue::Number(1234.0)),
        rule: HINT,
        names: |s| is_field(s, RequestField::MaxOutputTokens),
    },
    Case {
        probe: Probe::Stop,
        label: "",
        request: |m, _| base(m).stop(["allama-stop-probe"]),
        evidence: Evidence::Value(WireValue::Text("allama-stop-probe")),
        rule: HINT,
        names: |s| is_field(s, RequestField::Stop),
    },
    Case {
        probe: Probe::Seed,
        label: "",
        request: |m, _| base(m).seed(424_242),
        evidence: Evidence::Value(WireValue::Number(424_242.0)),
        rule: HINT,
        names: |s| is_field(s, RequestField::Seed),
    },
    Case {
        probe: Probe::FrequencyPenalty,
        label: "",
        request: |m, _| base(m).frequency_penalty(0.21),
        evidence: Evidence::Value(WireValue::Number(0.21)),
        rule: HINT,
        names: |s| is_field(s, RequestField::FrequencyPenalty),
    },
    Case {
        probe: Probe::PresencePenalty,
        label: "",
        request: |m, _| base(m).presence_penalty(0.19),
        evidence: Evidence::Value(WireValue::Number(0.19)),
        rule: HINT,
        names: |s| is_field(s, RequestField::PresencePenalty),
    },
    // System prompts.
    Case {
        probe: Probe::SystemPrompt,
        label: "",
        request: |m, _| base(m).system("allama-probe-system: answer briefly."),
        evidence: Evidence::Value(WireValue::Text("allama-probe-system")),
        rule: HINT,
        names: |s| matches!(s, Subject::System { index: 0 }) || is_field(s, RequestField::System),
    },
    Case {
        probe: Probe::SystemCache,
        label: "",
        request: |m, _| base(m).system(SystemPrompt::new("Be brief.").with_cache(long())),
        evidence: Evidence::Change(|m| {
            base(m).system(SystemPrompt::new("Be brief.").with_cache(short()))
        }),
        rule: HINT,
        names: |s| matches!(s, Subject::System { index: 0 }) || is_field(s, RequestField::Cache),
    },
    // Messages.
    Case {
        probe: Probe::MessageCache,
        label: "user message",
        request: |m, _| ChatRequest::new(m).message(Message::user("Say hello.").with_cache(long())),
        evidence: Evidence::Change(|m| {
            ChatRequest::new(m).message(Message::user("Say hello.").with_cache(short()))
        }),
        rule: HINT,
        names: |s| is_message(s, 0) || is_field(s, RequestField::Cache),
    },
    Case {
        probe: Probe::MessageCache,
        label: "assistant message",
        request: |m, _| {
            ChatRequest::new(m)
                .user("Say hello.")
                .message(Message::assistant("Hello.").with_cache(long()))
                .user("Again.")
        },
        evidence: Evidence::Change(|m| {
            ChatRequest::new(m)
                .user("Say hello.")
                .message(Message::assistant("Hello.").with_cache(short()))
                .user("Again.")
        }),
        rule: HINT,
        names: |s| is_message(s, 1) || is_field(s, RequestField::Cache),
    },
    Case {
        probe: Probe::MessageCache,
        label: "tool message",
        request: |m, _| tool_turn(m, tool_result().with_cache(long())),
        evidence: Evidence::Change(|m| tool_turn(m, tool_result().with_cache(short()))),
        rule: HINT,
        names: |s| is_message(s, 2) || is_field(s, RequestField::Cache),
    },
    // Tools.
    Case {
        probe: Probe::ToolCache,
        label: "",
        request: |m, _| base(m).tools([lookup_tool().with_cache(long())]),
        evidence: Evidence::Change(|m| base(m).tools([lookup_tool().with_cache(short())])),
        rule: HINT,
        names: |s| is_field(s, RequestField::Cache) || is_field(s, RequestField::Tools),
    },
    Case {
        probe: Probe::ToolStrict,
        label: "",
        request: |m, _| base(m).tools([lookup_tool().strict(true)]),
        evidence: Evidence::Change(|m| base(m).tools([lookup_tool().strict(false)])),
        rule: HINT,
        names: |s| is_field(s, RequestField::Tools),
    },
    Case {
        probe: Probe::ToolResultError,
        label: "",
        request: |m, _| {
            tool_turn(
                m,
                Message::tool_error(CALL_ID, "lookup", "The lookup service is down."),
            )
        },
        evidence: Evidence::Change(|m| {
            tool_turn(
                m,
                Message::tool_result(CALL_ID, "lookup", "The lookup service is down."),
            )
        }),
        rule: DROPPABLE,
        names: |s| is_message(s, 2),
    },
    Case {
        probe: Probe::ParallelToolCalls,
        label: "",
        request: |m, _| base(m).tools([lookup_tool()]).parallel_tool_calls(true),
        evidence: Evidence::Change(|m| base(m).tools([lookup_tool()]).parallel_tool_calls(false)),
        rule: HINT,
        names: |s| is_field(s, RequestField::ParallelToolCalls),
    },
    // Prompt caching and reasoning.
    Case {
        probe: Probe::PromptCache,
        label: "key",
        request: |m, _| base(m).cache(PromptCache::new().key("allama-probe-cache-key")),
        evidence: Evidence::Value(WireValue::Text("allama-probe-cache-key")),
        rule: HINT,
        names: |s| is_field(s, RequestField::Cache),
    },
    Case {
        probe: Probe::PromptCache,
        label: "retention",
        request: |m, _| base(m).cache(PromptCache::new().retention(CacheRetention::Long)),
        evidence: Evidence::Change(|m| {
            base(m).cache(PromptCache::new().retention(CacheRetention::Short))
        }),
        rule: HINT,
        names: |s| is_field(s, RequestField::Cache),
    },
    Case {
        probe: Probe::ReasoningConfig,
        label: "effort",
        request: |m, _| base(m).reasoning(ReasoningEffort::High),
        evidence: Evidence::Change(|m| base(m).reasoning(ReasoningEffort::Low)),
        rule: HINT,
        names: |s| is_field(s, RequestField::Reasoning),
    },
    Case {
        probe: Probe::ReasoningConfig,
        label: "output",
        request: |m, _| base(m).reasoning(ReasoningConfig::new().output(ReasoningOutput::Full)),
        evidence: Evidence::Change(|m| {
            base(m).reasoning(ReasoningConfig::new().output(ReasoningOutput::Omitted))
        }),
        rule: HINT,
        names: |s| is_field(s, RequestField::Reasoning),
    },
    // Input media.
    Case {
        probe: Probe::ImageDetail,
        label: "user message",
        request: |m, _| image_request(m, Some(ImageDetail::High)),
        evidence: Evidence::Change(|m| image_request(m, Some(ImageDetail::Low))),
        rule: HINT,
        names: |s| is_message(s, 0),
    },
    Case {
        probe: Probe::ImageDetail,
        label: "tool result",
        request: |m, _| image_result(m, Some(ImageDetail::High)),
        evidence: Evidence::Change(|m| image_result(m, Some(ImageDetail::Low))),
        rule: HINT,
        names: |s| is_message(s, 2),
    },
    // Content the request asks the model to read (ADR 0003, principle 2:
    // whatever the capability answer, it is sent, warned about, or refused).
    Case {
        probe: Probe::ImageSource,
        label: "user message",
        request: |m, _| image_request(m, None),
        evidence: Evidence::Value(WireValue::Text(RED_PNG_BASE64)),
        rule: DROPPABLE,
        names: |s| is_message(s, 0),
    },
    Case {
        probe: Probe::ImageSource,
        label: "tool result",
        request: |m, _| image_result(m, None),
        evidence: Evidence::Value(WireValue::Text(RED_PNG_BASE64)),
        rule: DROPPABLE,
        names: |s| is_message(s, 2),
    },
    Case {
        probe: Probe::ImageSource,
        label: "assistant message",
        request: |m, _| {
            replay(
                m,
                vec![
                    ContentBlock::Image(image(None)),
                    ContentBlock::text("A red square."),
                ],
            )
        },
        evidence: Evidence::Value(WireValue::Text(RED_PNG_BASE64)),
        rule: DROPPABLE,
        names: |s| is_message(s, 1),
    },
    Case {
        probe: Probe::ResponseFormat,
        label: "",
        request: |m, _| {
            base(m).response_format(
                allama::ResponseFormat::json_schema(json!({
                    "type": "object",
                    "properties": {"allama_probe_property": {"type": "string"}},
                    "required": ["allama_probe_property"],
                    "additionalProperties": false
                }))
                .name("allama_probe_format"),
            )
        },
        evidence: Evidence::Value(WireValue::Text("allama_probe_property")),
        rule: DROPPABLE,
        names: |s| is_field(s, RequestField::ResponseFormat),
    },
    Case {
        probe: Probe::FileName,
        label: "user message",
        request: |m, _| {
            ChatRequest::new(m).message(Message::user_parts(vec![
                ContentPart::text("Summarize the file."),
                ContentPart::File(named_file()),
            ]))
        },
        evidence: Evidence::Value(WireValue::Text("allama-probe-file")),
        rule: DROPPABLE,
        names: |s| is_message(s, 0),
    },
    Case {
        probe: Probe::FileName,
        label: "tool result",
        request: |m, _| {
            tool_turn(
                m,
                Message::tool_result(CALL_ID, "lookup", vec![ContentPart::File(named_file())]),
            )
        },
        evidence: Evidence::Value(WireValue::Text("allama-probe-file")),
        rule: DROPPABLE,
        names: |s| is_message(s, 2),
    },
    // State from another provider.
    Case {
        probe: Probe::ForeignReasoning,
        label: "",
        request: |m, _| {
            replay(
                m,
                vec![
                    ContentBlock::Reasoning(
                        Reasoning::new("allama-probe-foreign reasoning")
                            .with_provider_data(foreign_data()),
                    ),
                    ContentBlock::text("4"),
                ],
            )
        },
        evidence: Evidence::Value(WireValue::Text(FOREIGN)),
        rule: WITHHELD_WARN,
        names: |s| is_message(s, 1),
    },
    Case {
        probe: Probe::ForeignOtherContent,
        label: "user part",
        request: |m, _| {
            ChatRequest::new(m).message(Message::user_parts(vec![
                ContentPart::text("Say hello."),
                ContentPart::Other(foreign_other()),
            ]))
        },
        evidence: Evidence::Value(WireValue::Text(FOREIGN)),
        rule: WITHHELD_WARN,
        names: |s| is_message(s, 0),
    },
    Case {
        probe: Probe::ForeignOtherContent,
        label: "tool result part",
        request: |m, supports| tool_result_parts(m, supports, true),
        evidence: Evidence::Value(WireValue::Text(FOREIGN)),
        // A refusal is accepted only when the same tool turn without the
        // part is refused too (an adapter path without tool calls).
        rule: Rule::Withheld {
            warn: true,
            refusal: Refusal::Unrelated(|m, supports| tool_result_parts(m, supports, false)),
        },
        names: |s| is_message(s, 2),
    },
    Case {
        probe: Probe::ForeignOtherContent,
        label: "assistant block",
        request: |m, _| {
            replay(
                m,
                vec![
                    ContentBlock::Other(foreign_other()),
                    ContentBlock::text("4"),
                ],
            )
        },
        evidence: Evidence::Value(WireValue::Text(FOREIGN)),
        rule: WITHHELD_WARN,
        names: |s| is_message(s, 1),
    },
    Case {
        probe: Probe::ForeignProviderTool,
        label: "",
        request: |m, _| {
            replay(
                m,
                vec![
                    ContentBlock::ProviderToolCall(
                        ProviderToolCall::new(
                            "srvtoolu_allama_probe",
                            "web_search",
                            r#"{"query": "allama-probe-foreign"}"#,
                        )
                        .with_origin(FOREIGN),
                    ),
                    ContentBlock::ProviderToolResult(
                        ProviderToolResult::new(
                            "srvtoolu_allama_probe",
                            json!({"results": ["allama-probe-foreign"]}),
                        )
                        .with_origin(FOREIGN),
                    ),
                    ContentBlock::text("4"),
                ],
            )
        },
        evidence: Evidence::Value(WireValue::Text(FOREIGN)),
        rule: WITHHELD_WARN,
        names: |s| is_message(s, 1),
    },
    Case {
        probe: Probe::ForeignProviderData,
        label: "messages",
        request: |m, _| {
            let mut user = UserMessage::new(vec![ContentPart::Text(
                Text::new("What is 2 + 2?").with_provider_data(foreign_data()),
            )]);
            user.provider_data = foreign_data();
            let mut assistant = Message::assistant_blocks(vec![
                ContentBlock::Text(Text::new("4").with_provider_data(foreign_data())),
                ContentBlock::Refusal(
                    allama::Refusal::new("I won't show the working.")
                        .with_provider_data(foreign_data()),
                ),
            ]);
            *assistant.provider_data_mut() = foreign_data();
            ChatRequest::new(m)
                .message(user)
                .message(assistant)
                .user("And 3 + 3?")
        },
        evidence: Evidence::Value(WireValue::Text(FOREIGN)),
        rule: WITHHELD,
        names: |_| false,
    },
    Case {
        probe: Probe::ForeignProviderData,
        label: "tool turn",
        request: |m, _| {
            let mut request = tool_turn(
                m,
                Message::tool_result(
                    CALL_ID,
                    "lookup",
                    vec![ContentPart::Text(
                        Text::new("Rust is a programming language.")
                            .with_provider_data(foreign_data()),
                    )],
                ),
            );
            for tool in &mut request.tools {
                tool.provider_data = foreign_data();
            }
            for message in &mut request.messages[1..] {
                *message.provider_data_mut() = foreign_data();
                if let Message::Assistant(assistant) = message {
                    for block in &mut assistant.content {
                        if let ContentBlock::ToolCall(call) = block {
                            call.provider_data = foreign_data();
                        }
                    }
                }
            }
            request
        },
        evidence: Evidence::Value(WireValue::Text(FOREIGN)),
        rule: WITHHELD_REFUSABLE,
        names: |_| false,
    },
    Case {
        probe: Probe::ForeignProviderData,
        label: "input media",
        request: |m, _| {
            ChatRequest::new(m).message(Message::user_parts(vec![
                ContentPart::text("Describe both."),
                ContentPart::Image(image(None).with_provider_data(foreign_data())),
                ContentPart::File(named_file().with_provider_data(foreign_data())),
            ]))
        },
        evidence: Evidence::Value(WireValue::Text(FOREIGN)),
        rule: WITHHELD_REFUSABLE,
        names: |_| false,
    },
    Case {
        probe: Probe::ForeignProviderData,
        label: "assistant image",
        request: |m, _| {
            replay(
                m,
                vec![ContentBlock::Image(
                    image(None).with_provider_data(foreign_data()),
                )],
            )
        },
        evidence: Evidence::Value(WireValue::Text(FOREIGN)),
        rule: WITHHELD_REFUSABLE,
        names: |_| false,
    },
    // Application data.
    Case {
        probe: Probe::AppData,
        label: "",
        request: |m, supports| app_data_conversation(m, supports, true),
        evidence: Evidence::Value(WireValue::Text(super::APP_DATA_PROBE)),
        rule: Rule::Inert {
            without: |m, supports| app_data_conversation(m, supports, false),
        },
        names: |_| false,
    },
];

fn base(model: &str) -> ChatRequest {
    ChatRequest::new(model).user("Say hello.")
}

/// A breakpoint asking for long retention.
///
/// A plain breakpoint may be satisfied without a wire form (automatic
/// prefix caching), so cache probes contrast long and short retention: an
/// adapter that keeps breakpoints sends them differently, or says it cannot.
fn long() -> CacheBreakpoint {
    CacheBreakpoint::with_retention(CacheRetention::Long)
}

fn short() -> CacheBreakpoint {
    CacheBreakpoint::with_retention(CacheRetention::Short)
}

fn is_field(subject: &Subject, field: RequestField) -> bool {
    matches!(subject, Subject::Field { field: f } if *f == field)
}

fn is_message(subject: &Subject, index: usize) -> bool {
    matches!(subject, Subject::Content { message, .. } if *message == index)
}

fn lookup_tool() -> ToolDefinition {
    ToolDefinition::new(
        "lookup",
        json!({
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
            "additionalProperties": false
        }),
    )
    .description("Look something up.")
}

fn tool_result() -> Message {
    Message::tool_result(CALL_ID, "lookup", "Rust is a programming language.")
}

/// A user turn, an assistant call to `lookup`, and `result` (message 2).
fn tool_turn(model: &str, result: Message) -> ChatRequest {
    ChatRequest::new(model)
        .user("Look up rust.")
        .message(Message::assistant_blocks(vec![ContentBlock::tool_call(
            CALL_ID,
            "lookup",
            r#"{"q": "rust"}"#,
        )]))
        .message(result)
        .tools([lookup_tool()])
}

/// A tool result (message 2) of text and, with `foreign`, another
/// provider's `Other` part. A provider without tool calls gets the same
/// parts in a user message at the same index.
fn tool_result_parts(model: &str, supports: Supports, foreign: bool) -> ChatRequest {
    let mut parts = vec![ContentPart::text("Rust is a programming language.")];
    if foreign {
        parts.push(ContentPart::Other(foreign_other()));
    }
    if supports.tool_calls {
        tool_turn(model, Message::tool_result(CALL_ID, "lookup", parts))
    } else {
        ChatRequest::new(model)
            .user("Look up rust.")
            .assistant("Rust is a programming language.")
            .message(Message::user_parts(parts))
    }
}

/// A question, an assistant answer made of `blocks` (message 1), and a
/// follow-up question.
fn replay(model: &str, blocks: Vec<ContentBlock>) -> ChatRequest {
    ChatRequest::new(model)
        .user("What is 2 + 2?")
        .message(Message::assistant_blocks(blocks))
        .user("And 3 + 3?")
}

fn image(detail: Option<ImageDetail>) -> Image {
    let image = Image::base64("image/png", RED_PNG_BASE64);
    match detail {
        Some(detail) => image.with_detail(detail),
        None => image,
    }
}

fn image_request(model: &str, detail: Option<ImageDetail>) -> ChatRequest {
    ChatRequest::new(model).message(Message::user_parts(vec![
        ContentPart::text("Describe the image."),
        ContentPart::Image(image(detail)),
    ]))
}

fn image_result(model: &str, detail: Option<ImageDetail>) -> ChatRequest {
    tool_turn(
        model,
        Message::tool_result(CALL_ID, "lookup", vec![ContentPart::Image(image(detail))]),
    )
}

fn named_file() -> File {
    File::base64("application/pdf", PINEAPPLE_PDF_BASE64).with_filename("allama-probe-file.pdf")
}

/// A system prompt and every message role (user text, user parts, an
/// assistant turn with a tool call, its tool result), with
/// [`APP_DATA_PROBE`](super::APP_DATA_PROBE) as application data on each
/// when `tagged`. The tool call and its result are left out when the
/// provider does not serve tool calls, which it would refuse.
pub(crate) fn app_data_conversation(model: &str, supports: Supports, tagged: bool) -> ChatRequest {
    use super::APP_DATA_PROBE;

    let message = |message: Message| {
        if tagged {
            message.with_app_data(APP_DATA_PROBE, APP_DATA_PROBE)
        } else {
            message
        }
    };
    let mut system = SystemPrompt::new("Be concise.");
    if tagged {
        system = system.with_app_data(APP_DATA_PROBE, APP_DATA_PROBE);
    }
    let mut request = ChatRequest::new(model)
        .system(system)
        .message(message(Message::user("What is the weather in Paris?")));
    if supports.tool_calls {
        request = request
            .message(message(Message::assistant_blocks(vec![
                ContentBlock::text("Checking."),
                ContentBlock::tool_call("call_1", "get_weather", r#"{"city":"Paris"}"#),
            ])))
            .message(message(Message::tool_result(
                "call_1",
                "get_weather",
                "Sunny",
            )));
    }
    request
        .message(message(Message::assistant("It is sunny.")))
        .message(message(Message::user(vec![
            ContentPart::text("Thanks."),
            ContentPart::text("And tomorrow?"),
        ])))
}

fn foreign_data() -> ProviderData {
    ProviderData::new().with_raw(FOREIGN, "state", Value::String(FOREIGN.into()))
}

fn foreign_other() -> OtherContent {
    let mut data = serde_json::Map::new();
    data.insert("allama_probe".into(), Value::String(FOREIGN.into()));
    OtherContent::new("allama_probe_content", data).with_origin(FOREIGN)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn the_list_names_known_probes_only() {
        let (fields, unread) = parse_accounted_fields(ACCOUNTED_FIELDS);
        assert!(!fields.is_empty());
        assert!(unread.is_empty(), "{unread:?}");
    }

    /// A list from a newer `allama` is read as far as this crate
    /// knows it, and what it cannot read is reported, never a panic.
    #[test]
    fn a_newer_list_is_read_as_far_as_it_is_known() {
        let (fields, unread) = parse_accounted_fields(
            "# a comment\n\
             temperature  probe: temperature\n\
             future_field  probe: temperature, future_probe\n\
             later_field  deferred: a kind of account from a later release\n\
             bare_line\n\
             model  exempt: selects the model\n",
        );
        assert_eq!(
            fields,
            [
                AccountedField {
                    path: "temperature",
                    account: Account::Probed(vec![Probe::Temperature]),
                },
                AccountedField {
                    path: "model",
                    account: Account::Exempt("selects the model"),
                },
            ]
        );
        assert_eq!(
            unread,
            [
                "`future_field` names the probe(s) `future_probe`, which this \
                 allama-conformance does not know",
                "`later_field` has an account this allama-conformance does not read: \
                 `deferred: a kind of account from a later release`",
                "`bare_line` has no account",
            ]
        );
    }

    #[test]
    fn every_probe_is_listed_and_has_a_case() {
        let listed: BTreeSet<&str> = accounted_fields()
            .iter()
            .filter_map(|field| match &field.account {
                Account::Probed(probes) => Some(probes.iter().map(|p| p.as_str())),
                Account::Exempt(_) => None,
            })
            .flatten()
            .collect();
        for probe in Probe::ALL {
            assert!(
                listed.contains(probe.as_str()),
                "`{probe}` checks no path in ACCOUNTED_FIELDS"
            );
            assert!(
                CASES.iter().any(|case| case.probe == *probe),
                "`{probe}` has no case"
            );
        }
    }

    #[test]
    fn probe_names_round_trip() {
        for probe in Probe::ALL {
            assert_eq!(Probe::from_name(probe.as_str()), Some(*probe));
        }
        assert_eq!(Probe::from_name("nope"), None);
    }
}
