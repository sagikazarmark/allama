//! Client-executed tool items that are not function calls (ADR 0014).
//!
//! The Responses API asks the client to run some tools whose calls are not
//! `function_call` items: custom tools (`custom_tool_call`, freeform text
//! input), the shell tools (`local_shell_call`, `shell_call`), computer use
//! (`computer_call`) and `apply_patch_call`. A hosted MCP tool that needs
//! approval asks the client too (`mcp_approval_request`), and waits for an
//! `mcp_approval_response`. Their input and output shapes
//! are specific to OpenAI, so they stay [`OtherContent`] with the dialect
//! namespace as origin. This module gives them typed views and builds their
//! answers, so a tool loop neither parses nor hand-writes the wire JSON.
//!
//! Custom tools exist in Chat Completions too (`tool_calls[]` entries with
//! `type: "custom"`). Both dialects share one representation, so a
//! conversation moves between them:
//!
//! - the call is an `Other` block of type `custom_tool_call` with
//!   `call_id`, `name` and `input` (Responses adds its item `id` and
//!   `status`, which Chat Completions does not have);
//! - the answer is an `Other` part of type `custom_tool_call_output` with
//!   `call_id` and `output`, in a user message after the assistant turn.
//!
//! The Responses encoder sends a client tool output of its own origin as a
//! top-level input item instead of message content. The Chat Completions
//! encoder lifts a custom tool answer into a `tool` message, and drops the
//! Responses-only calls and outputs (shell, computer use, `apply_patch`),
//! which it cannot express.
//!
//! **Pairing.** A client tool call that is in the request but that the
//! encoder does not replay takes its answer with it: the answer is dropped
//! with a [`Dropped`](allama::WarningKind::Dropped) warning, so a provider
//! never sees an output without its call. An answer whose call is not in
//! the request at all is sent by the Responses encoder, because with
//! server-side continuation (`previous_response_id`, `conversation`) the
//! call lives on the server. Chat Completions has no continuation, so its
//! encoder drops such an answer with a warning.

use std::collections::BTreeMap;

use allama::{ContentBlock, ContentPart, ExtraMap, Message, OtherContent};
use serde_json::{Value, json};

/// A client-executed OpenAI tool call that is not a function call, read
/// from an [`OtherContent`] block of the model's turn.
///
/// [`ChatResponse::tool_calls`](allama::ChatResponse::tool_calls) never
/// returns these: they are not portable tool calls. Find the ones that still
/// wait for an answer with [`ClientToolCall::pending`], run them, and send
/// each answer (from [`CustomToolCall::output`], [`LocalShellCall::output`]
/// or [`ShellCall::output`]) in a user message after the assistant turn.
///
/// Responses turns hold every kind; Chat Completions turns hold custom tool
/// calls only, in the same shape, so the same loop serves both dialects.
/// The Responses encoder sends an answer as an input item of its own; the
/// Chat Completions encoder lifts it into a `tool` message right after the
/// assistant turn.
///
/// ```
/// use allama::{ChatRequest, ChatResponse, ContentBlock, Message, OtherContent};
/// use allama_openai_compat::ClientToolCall;
/// use serde_json::json;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // A turn that called a freeform custom tool (as the Responses adapter
/// // decodes it).
/// let call = OtherContent::new(
///     "custom_tool_call",
///     serde_json::from_value(json!({
///         "id": "ctc_1", "call_id": "call_1", "name": "sql", "input": "SELECT 1",
///         "status": "completed",
///     }))?,
/// )
/// .with_origin("openai");
/// let response = ChatResponse::new(vec![ContentBlock::Other(call)]);
///
/// let mut answers = Vec::new();
/// for call in ClientToolCall::pending(&response.content) {
///     match call {
///         ClientToolCall::Custom(call) if call.name == "sql" => {
///             answers.push(call.output("1 row: 1"));
///         }
///         // Shell, computer use, unknown custom tools, …
///         other => return Err(format!("cannot run {other:?}").into()),
///     }
/// }
///
/// let mut request = ChatRequest::new("gpt-5").user("How many rows?");
/// request.push_message(response.to_assistant_message());
/// request.push_message(Message::user(answers));
/// # Ok(()) }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ClientToolCall {
    /// A custom tool call (`custom_tool_call`): freeform text input.
    Custom(CustomToolCall),
    /// A legacy local shell call (`local_shell_call`).
    LocalShell(LocalShellCall),
    /// A shell call in local mode (`shell_call`).
    Shell(ShellCall),
    /// A hosted MCP tool call that waits for the client's approval
    /// (`mcp_approval_request`).
    McpApproval(McpApprovalRequest),
    /// A client-executed item without a typed view here (`computer_call`,
    /// `apply_patch_call`), or a known one whose fields did not parse.
    /// Answer it with an `Other` part of the same origin whose type is the
    /// item type with an `_output` suffix.
    Other(OtherContent),
}

impl ClientToolCall {
    /// Read a client-executed tool call. `None` when `other` is not one.
    #[must_use]
    pub fn from_content(other: &OtherContent) -> Option<Self> {
        let call = match other.type_name.as_str() {
            CUSTOM => CustomToolCall::from_content(other).map(Self::Custom),
            LOCAL_SHELL => LocalShellCall::from_content(other).map(Self::LocalShell),
            SHELL => ShellCall::from_content(other).map(Self::Shell),
            MCP_APPROVAL_REQUEST => McpApprovalRequest::from_content(other).map(Self::McpApproval),
            kind if CALL_TYPES.contains(&kind) => None,
            _ => return None,
        };
        Some(call.unwrap_or_else(|| Self::Other(other.clone())))
    }

    /// The client-executed tool calls in `content` (a model turn) that have
    /// no output in the same turn, in order.
    ///
    /// An output in the same turn means the endpoint ran the call itself
    /// (OpenAI's hosted shell answers `shell_call` with `shell_call_output`
    /// items in the response).
    #[must_use]
    pub fn pending(content: &[ContentBlock]) -> Vec<Self> {
        pending_calls(content)
            .filter_map(Self::from_content)
            .collect()
    }

    /// The id that correlates the call with its output.
    #[must_use]
    pub fn call_id(&self) -> Option<&str> {
        match self {
            Self::Custom(call) => Some(&call.call_id),
            Self::LocalShell(call) => Some(&call.call_id),
            Self::Shell(call) => Some(&call.call_id),
            Self::McpApproval(request) => Some(&request.id),
            Self::Other(other) => call_id(other),
        }
    }
}

/// A call to a custom tool (`custom_tool_call`): the model sends freeform
/// text, constrained by the tool's grammar when it has one, instead of JSON
/// arguments.
///
/// Custom tools are declared in each dialect's own words: through
/// [`ResponsesRequestOptions::client_tools`](crate::ResponsesRequestOptions::client_tools)
/// (`{"type": "custom", "name": …, "format": …}`) and through
/// [`ChatRequestOptions::client_tools`](crate::ChatRequestOptions::client_tools)
/// (`{"type": "custom", "custom": {"name": …, "format": …}}`).
///
/// A call decoded from Chat Completions has no Responses item `id` or
/// `status`; the Responses encoder replays it without them.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct CustomToolCall {
    /// Id the output answers (`call_id`).
    pub call_id: String,
    /// Name of the custom tool.
    pub name: String,
    /// The model's freeform input.
    pub input: String,
    /// Origin namespace of the call; the output gets the same one.
    pub origin: Option<String>,
}

impl CustomToolCall {
    /// Read a `custom_tool_call` item. `None` for any other content, or
    /// when a required field is missing.
    #[must_use]
    pub fn from_content(other: &OtherContent) -> Option<Self> {
        if other.type_name != CUSTOM {
            return None;
        }
        Some(Self {
            call_id: string(&other.data, "call_id")?,
            name: string(&other.data, "name")?,
            input: string(&other.data, "input")?,
            origin: other.origin.clone(),
        })
    }

    /// The answer to this call (`custom_tool_call_output`), for a user
    /// message after the assistant turn.
    #[must_use]
    pub fn output(&self, output: impl Into<String>) -> ContentPart {
        output_part(
            CUSTOM_OUTPUT,
            json!({"call_id": self.call_id, "output": output.into()}),
            &self.origin,
        )
    }
}

/// A call to the legacy local shell tool (`local_shell_call`): run one
/// command, as an argument vector, on the client's machine.
///
/// OpenAI ended support for the local shell tool (with `codex-mini-latest`)
/// on 2026-02-12 in favor of the shell tool ([`ShellCall`]); other
/// Responses endpoints may still emit it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct LocalShellCall {
    /// Id the output answers (`call_id`).
    pub call_id: String,
    /// The command and its arguments (`action.command`).
    pub command: Vec<String>,
    /// Environment variables to set (`action.env`).
    pub env: BTreeMap<String, String>,
    /// Timeout in milliseconds (`action.timeout_ms`).
    pub timeout_ms: Option<u64>,
    /// User to run the command as (`action.user`).
    pub user: Option<String>,
    /// Working directory (`action.working_directory`).
    pub working_directory: Option<String>,
    /// Origin namespace of the call; the output gets the same one.
    pub origin: Option<String>,
}

impl LocalShellCall {
    /// Read a `local_shell_call` item. `None` for any other content, or
    /// when a required field is missing.
    #[must_use]
    pub fn from_content(other: &OtherContent) -> Option<Self> {
        if other.type_name != LOCAL_SHELL {
            return None;
        }
        let action = other.data.get("action")?.as_object()?;
        let command = action
            .get("command")?
            .as_array()?
            .iter()
            .map(|arg| arg.as_str().map(str::to_owned))
            .collect::<Option<Vec<_>>>()?;
        let env = match action.get("env") {
            None | Some(Value::Null) => BTreeMap::new(),
            Some(env) => serde_json::from_value(env.clone()).ok()?,
        };
        Some(Self {
            call_id: string(&other.data, "call_id")?,
            command,
            env,
            timeout_ms: action.get("timeout_ms").and_then(Value::as_u64),
            user: string(action, "user"),
            working_directory: string(action, "working_directory"),
            origin: other.origin.clone(),
        })
    }

    /// The answer to this call (`local_shell_call_output`), for a user
    /// message after the assistant turn. `output` is what the command
    /// printed, or an error message.
    #[must_use]
    pub fn output(&self, output: impl Into<String>) -> ContentPart {
        // The output item names the call in `id`, not `call_id`.
        output_part(
            LOCAL_SHELL_OUTPUT,
            json!({"id": self.call_id, "output": output.into()}),
            &self.origin,
        )
    }
}

/// A call to the shell tool in local mode (`shell_call`): run one or more
/// shell commands on the client's machine.
///
/// The same tool can run in an OpenAI-hosted container instead; the
/// response then carries the outputs itself, and
/// [`ClientToolCall::pending`] leaves the call out.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ShellCall {
    /// Id the output answers (`call_id`).
    pub call_id: String,
    /// Shell commands to run in order (`action.commands`).
    pub commands: Vec<String>,
    /// Wall-clock limit in milliseconds (`action.timeout_ms`).
    pub timeout_ms: Option<u64>,
    /// Most UTF-8 characters of combined stdout and stderr to capture
    /// (`action.max_output_length`); sent back with the output.
    pub max_output_length: Option<u64>,
    /// Origin namespace of the call; the output gets the same one.
    pub origin: Option<String>,
}

impl ShellCall {
    /// Read a `shell_call` item. `None` for any other content, or when a
    /// required field is missing.
    #[must_use]
    pub fn from_content(other: &OtherContent) -> Option<Self> {
        if other.type_name != SHELL {
            return None;
        }
        let action = other.data.get("action")?.as_object()?;
        let commands = action
            .get("commands")?
            .as_array()?
            .iter()
            .map(|command| command.as_str().map(str::to_owned))
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            call_id: string(&other.data, "call_id")?,
            commands,
            timeout_ms: action.get("timeout_ms").and_then(Value::as_u64),
            max_output_length: action.get("max_output_length").and_then(Value::as_u64),
            origin: other.origin.clone(),
        })
    }

    /// The answer to this call (`shell_call_output`): one entry per command
    /// that ran, for a user message after the assistant turn.
    #[must_use]
    pub fn output(&self, outputs: impl IntoIterator<Item = ShellCommandOutput>) -> ContentPart {
        let outputs: Vec<Value> = outputs.into_iter().map(ShellCommandOutput::wire).collect();
        let mut data = json!({"call_id": self.call_id, "output": outputs});
        if let Some(limit) = self.max_output_length {
            data["max_output_length"] = limit.into();
        }
        output_part(SHELL_OUTPUT, data, &self.origin)
    }
}

/// A hosted MCP tool call that waits for the client's approval
/// (`mcp_approval_request`): OpenAI's remote MCP tool asks before it calls
/// a server tool, unless the tool definition sets `require_approval` to
/// `never`.
///
/// Answer it with [`approve`](Self::approve) or [`deny`](Self::deny) in a
/// user message after the assistant turn; the Responses encoder sends the
/// answer as an `mcp_approval_response` input item, and the endpoint then
/// runs the call (or not) in the next response. Chat Completions has no
/// MCP tool, so its encoder drops both.
///
/// ```
/// use allama::{ChatResponse, ContentBlock, Message, OtherContent};
/// use allama_openai_compat::ClientToolCall;
/// use serde_json::json;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let request = OtherContent::new(
///     "mcp_approval_request",
///     serde_json::from_value(json!({
///         "id": "mcpr_1", "server_label": "deepwiki", "name": "ask_question",
///         "arguments": "{\"q\":\"x\"}",
///     }))?,
/// )
/// .with_origin("openai");
/// let response = ChatResponse::new(vec![ContentBlock::Other(request)]);
///
/// let mut answers = Vec::new();
/// for call in ClientToolCall::pending(&response.content) {
///     if let ClientToolCall::McpApproval(request) = call {
///         answers.push(if request.server_label == "deepwiki" {
///             request.approve()
///         } else {
///             request.deny()
///         });
///     }
/// }
/// let answer = Message::user(answers);
/// # let _ = answer; Ok(()) }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct McpApprovalRequest {
    /// Id of the request, which the answer names (`id`).
    pub id: String,
    /// Label of the MCP server (`server_label`).
    pub server_label: String,
    /// Name of the server tool to call (`name`).
    pub name: String,
    /// The call's arguments as a JSON string (`arguments`).
    pub arguments: String,
    /// Origin namespace of the request; the answer gets the same one.
    pub origin: Option<String>,
}

impl McpApprovalRequest {
    /// Read an `mcp_approval_request` item. `None` for any other content,
    /// or when a required field is missing.
    #[must_use]
    pub fn from_content(other: &OtherContent) -> Option<Self> {
        if other.type_name != MCP_APPROVAL_REQUEST {
            return None;
        }
        Some(Self {
            id: string(&other.data, "id")?,
            server_label: string(&other.data, "server_label")?,
            name: string(&other.data, "name")?,
            arguments: string(&other.data, "arguments").unwrap_or_default(),
            origin: other.origin.clone(),
        })
    }

    /// Approve the call (`mcp_approval_response` with `approve: true`).
    #[must_use]
    pub fn approve(&self) -> ContentPart {
        self.answer(true)
    }

    /// Deny the call (`mcp_approval_response` with `approve: false`).
    #[must_use]
    pub fn deny(&self) -> ContentPart {
        self.answer(false)
    }

    fn answer(&self, approve: bool) -> ContentPart {
        output_part(
            MCP_APPROVAL_RESPONSE,
            json!({"approval_request_id": self.id, "approve": approve}),
            &self.origin,
        )
    }
}

/// What one command of a [`ShellCall`] printed, and how it ended.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ShellCommandOutput {
    /// Captured standard output.
    pub stdout: String,
    /// Captured standard error.
    pub stderr: String,
    /// How the command ended.
    pub outcome: ShellOutcome,
}

impl ShellCommandOutput {
    /// A command that exited with `exit_code`.
    #[must_use]
    pub fn exited(stdout: impl Into<String>, stderr: impl Into<String>, exit_code: i32) -> Self {
        Self {
            stdout: stdout.into(),
            stderr: stderr.into(),
            outcome: ShellOutcome::Exit(exit_code),
        }
    }

    /// A command stopped at the call's timeout.
    #[must_use]
    pub fn timed_out(stdout: impl Into<String>, stderr: impl Into<String>) -> Self {
        Self {
            stdout: stdout.into(),
            stderr: stderr.into(),
            outcome: ShellOutcome::Timeout,
        }
    }

    fn wire(self) -> Value {
        let outcome = match self.outcome {
            ShellOutcome::Exit(code) => json!({"type": "exit", "exit_code": code}),
            ShellOutcome::Timeout => json!({"type": "timeout"}),
        };
        json!({"stdout": self.stdout, "stderr": self.stderr, "outcome": outcome})
    }
}

/// How a shell command ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ShellOutcome {
    /// It exited with this code.
    Exit(i32),
    /// It was stopped at the timeout.
    Timeout,
}

pub(crate) const CUSTOM: &str = "custom_tool_call";
pub(crate) const CUSTOM_OUTPUT: &str = "custom_tool_call_output";
const LOCAL_SHELL: &str = "local_shell_call";
const LOCAL_SHELL_OUTPUT: &str = "local_shell_call_output";
const SHELL: &str = "shell_call";
const SHELL_OUTPUT: &str = "shell_call_output";
const MCP_APPROVAL_REQUEST: &str = "mcp_approval_request";
const MCP_APPROVAL_RESPONSE: &str = "mcp_approval_response";

/// Output item types the client answers that are not function calls.
const CALL_TYPES: &[&str] = &[
    CUSTOM,
    LOCAL_SHELL,
    SHELL,
    "computer_call",
    "apply_patch_call",
    MCP_APPROVAL_REQUEST,
];

/// The answers to [`CALL_TYPES`], sent as top-level input items.
const OUTPUT_TYPES: &[&str] = &[
    CUSTOM_OUTPUT,
    LOCAL_SHELL_OUTPUT,
    SHELL_OUTPUT,
    "computer_call_output",
    "apply_patch_call_output",
    MCP_APPROVAL_RESPONSE,
];

/// The client tool calls a request holds that an encoder does not replay.
/// Their answers are dropped with them, so a provider never receives an
/// output whose call it was not sent.
///
/// An answer whose call is not in the request at all is not affected: with
/// server-side continuation, the call lives on the server.
pub(crate) struct Pairing<'a> {
    unreplayed: Vec<&'a str>,
}

impl<'a> Pairing<'a> {
    /// The calls of `messages` that `replayed` says the encoder does not
    /// send. A call id replayed anywhere in the request counts as replayed.
    pub(crate) fn new(messages: &'a [Message], replayed: impl Fn(&OtherContent) -> bool) -> Self {
        let mut unreplayed = Vec::new();
        let mut sent = Vec::new();
        let calls = messages
            .iter()
            .filter_map(Message::as_assistant)
            .flat_map(|message| others(&message.content))
            .filter(|other| is_call(other));
        for call in calls {
            let Some(id) = call_id(call) else {
                continue;
            };
            if replayed(call) {
                sent.push(id);
            } else {
                unreplayed.push(id);
            }
        }
        unreplayed.retain(|id| !sent.contains(id));
        Self { unreplayed }
    }

    /// Whether `answer` is the output of a call that is in the request but
    /// not replayed.
    pub(crate) fn is_unpaired(&self, answer: &OtherContent) -> bool {
        output_call_id(answer).is_some_and(|id| self.unreplayed.contains(&id))
    }
}

/// Whether `other` is a client-executed call that is not a function call.
pub(crate) fn is_call(other: &OtherContent) -> bool {
    CALL_TYPES.contains(&other.type_name.as_str())
}

/// Whether `other` answers a client-executed call.
pub(crate) fn is_output(other: &OtherContent) -> bool {
    OUTPUT_TYPES.contains(&other.type_name.as_str())
}

/// The text of a `custom_tool_call_output` answer, for a Chat Completions
/// `tool` message, and the types of the parts that have no text form.
///
/// The output is a string, or (Responses) a list of input parts: their
/// text parts are joined, and the others (images, files) are left out.
pub(crate) fn output_text(answer: &OtherContent) -> (String, Vec<String>) {
    match answer.data.get("output") {
        Some(Value::String(text)) => (text.clone(), Vec::new()),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            let mut skipped = Vec::new();
            for part in parts {
                let kind = part.get("type").and_then(Value::as_str).unwrap_or_default();
                match part.get("text").and_then(Value::as_str) {
                    Some(fragment) if matches!(kind, "input_text" | "text" | "output_text") => {
                        text.push_str(fragment);
                    }
                    _ => skipped.push(kind.to_owned()),
                }
            }
            (text, skipped)
        }
        None | Some(Value::Null) => (String::new(), Vec::new()),
        Some(other) => (other.to_string(), Vec::new()),
    }
}

/// Whether `content` holds a client-executed call that waits for an answer.
pub(crate) fn has_pending(content: &[ContentBlock]) -> bool {
    pending_calls(content).next().is_some()
}

/// The input item for a client tool output of `namespace`'s origin, which
/// goes at the top level of `input` rather than inside a message. `None`
/// for any other part.
pub(crate) fn output_item(other: &OtherContent, namespace: &str) -> Option<ExtraMap> {
    if other.origin.as_deref() != Some(namespace) || !is_output(other) {
        return None;
    }
    let mut item = other.data.clone();
    item.insert("type".into(), Value::String(other.type_name.clone()));
    Some(item)
}

/// The client-executed calls in `content` without an output in `content`.
fn pending_calls(content: &[ContentBlock]) -> impl Iterator<Item = &OtherContent> {
    let answered: Vec<&str> = others(content).filter_map(output_call_id).collect();
    others(content).filter(move |other| {
        is_call(other) && call_id(other).is_none_or(|id| !answered.contains(&id))
    })
}

fn others(content: &[ContentBlock]) -> impl Iterator<Item = &OtherContent> {
    content.iter().filter_map(|block| match block {
        ContentBlock::Other(other) => Some(other),
        _ => None,
    })
}

/// The id an answer names: `call_id`, or the item `id` of an
/// `mcp_approval_request`.
pub(crate) fn call_id(other: &OtherContent) -> Option<&str> {
    let key = match other.type_name.as_str() {
        MCP_APPROVAL_REQUEST => "id",
        _ => "call_id",
    };
    other.data.get(key).and_then(Value::as_str)
}

/// The call an output item answers: `call_id`, `id` for
/// `local_shell_call_output`, or `approval_request_id` for
/// `mcp_approval_response`.
pub(crate) fn output_call_id(other: &OtherContent) -> Option<&str> {
    let key = match other.type_name.as_str() {
        LOCAL_SHELL_OUTPUT => "id",
        MCP_APPROVAL_RESPONSE => "approval_request_id",
        kind if OUTPUT_TYPES.contains(&kind) => "call_id",
        _ => return None,
    };
    other.data.get(key).and_then(Value::as_str)
}

fn output_part(type_name: &str, data: Value, origin: &Option<String>) -> ContentPart {
    let Value::Object(data) = data else {
        unreachable!("output data is an object")
    };
    let mut part = OtherContent::new(type_name, data);
    part.origin.clone_from(origin);
    ContentPart::Other(part)
}

fn string(map: &ExtraMap, key: &str) -> Option<String> {
    map.get(key).and_then(Value::as_str).map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn other(type_name: &str, data: Value) -> OtherContent {
        OtherContent::new(type_name, serde_json::from_value(data).unwrap()).with_origin("openai")
    }

    fn part_json(part: &ContentPart) -> (String, Option<String>, Value) {
        let ContentPart::Other(other) = part else {
            panic!("expected an Other part, got {part:?}")
        };
        (
            other.type_name.clone(),
            other.origin.clone(),
            Value::Object(other.data.clone()),
        )
    }

    #[test]
    fn custom_tool_calls_read_and_answer() {
        let call = CustomToolCall::from_content(&other(
            "custom_tool_call",
            json!({"id": "ctc_1", "call_id": "call_1", "name": "sql", "input": "SELECT 1", "status": "completed"}),
        ))
        .unwrap();
        assert_eq!(call.call_id, "call_1");
        assert_eq!(call.name, "sql");
        assert_eq!(call.input, "SELECT 1");
        assert_eq!(
            part_json(&call.output("1")),
            (
                "custom_tool_call_output".into(),
                Some("openai".into()),
                json!({"call_id": "call_1", "output": "1"})
            )
        );
        assert!(
            CustomToolCall::from_content(&other("custom_tool_call", json!({"call_id": "c"})))
                .is_none()
        );
        assert!(CustomToolCall::from_content(&other("local_shell_call", json!({}))).is_none());
    }

    #[test]
    fn local_shell_calls_read_and_answer_by_id() {
        let call = LocalShellCall::from_content(&other(
            "local_shell_call",
            json!({"id": "lsh_1", "call_id": "call_2", "status": "completed", "action": {
                "type": "exec", "command": ["ls", "-la"], "env": {"LANG": "C"},
                "timeout_ms": 5000, "user": null, "working_directory": "/tmp"
            }}),
        ))
        .unwrap();
        assert_eq!(call.command, ["ls", "-la"]);
        assert_eq!(call.env.get("LANG").map(String::as_str), Some("C"));
        assert_eq!(call.timeout_ms, Some(5000));
        assert_eq!(call.user, None);
        assert_eq!(call.working_directory.as_deref(), Some("/tmp"));
        assert_eq!(
            part_json(&call.output("total 0")).2,
            json!({"id": "call_2", "output": "total 0"})
        );
        // A command that is not a list of strings does not parse.
        let bad = other(
            "local_shell_call",
            json!({"call_id": "c", "action": {"command": "ls"}}),
        );
        assert!(LocalShellCall::from_content(&bad).is_none());
        assert_eq!(
            ClientToolCall::from_content(&bad),
            Some(ClientToolCall::Other(bad))
        );
    }

    #[test]
    fn shell_calls_answer_per_command() {
        let call = ShellCall::from_content(&other(
            "shell_call",
            json!({"id": "sh_1", "call_id": "call_3", "status": "completed", "action": {
                "commands": ["ls", "sleep 99"], "timeout_ms": 1000, "max_output_length": 4096
            }}),
        ))
        .unwrap();
        assert_eq!(call.commands, ["ls", "sleep 99"]);
        let part = call.output([
            ShellCommandOutput::exited("a.txt\n", "", 0),
            ShellCommandOutput::timed_out("", ""),
        ]);
        assert_eq!(
            part_json(&part).2,
            json!({"call_id": "call_3", "max_output_length": 4096, "output": [
                {"stdout": "a.txt\n", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}},
                {"stdout": "", "stderr": "", "outcome": {"type": "timeout"}}
            ]})
        );
    }

    #[test]
    fn pending_skips_answered_calls_and_other_content() {
        let custom = other(
            "custom_tool_call",
            json!({"call_id": "c1", "name": "sql", "input": "x"}),
        );
        let computer = other("computer_call", json!({"call_id": "c2", "action": {}}));
        let hosted_shell = other(
            "shell_call",
            json!({"call_id": "c3", "action": {"commands": ["ls"]}}),
        );
        let hosted_output = other("shell_call_output", json!({"call_id": "c3", "output": []}));
        let content = vec![
            ContentBlock::text("Running."),
            ContentBlock::tool_call("f1", "lookup", "{}"),
            ContentBlock::Other(custom.clone()),
            ContentBlock::Other(computer.clone()),
            ContentBlock::Other(hosted_shell),
            ContentBlock::Other(hosted_output),
            ContentBlock::Other(other("reasoning_extra", json!({}))),
        ];
        let pending = ClientToolCall::pending(&content);
        assert_eq!(pending.len(), 2, "{pending:?}");
        assert!(matches!(&pending[0], ClientToolCall::Custom(call) if call.input == "x"));
        assert_eq!(pending[1], ClientToolCall::Other(computer));
        assert_eq!(pending[1].call_id(), Some("c2"));
        assert!(has_pending(&content));
        assert!(!has_pending(&content[..2]));
        assert!(ClientToolCall::from_content(&other("message", json!({}))).is_none());
    }

    /// An MCP approval request waits for the client, and its
    /// answer is an input item of its own.
    #[test]
    fn mcp_approval_requests_wait_and_are_answered() {
        let request = other(
            "mcp_approval_request",
            json!({"id": "mcpr_1", "server_label": "deepwiki", "name": "ask", "arguments": "{}"}),
        );
        let content = vec![ContentBlock::Other(request.clone())];
        assert!(has_pending(&content));
        let pending = ClientToolCall::pending(&content);
        let [ClientToolCall::McpApproval(pending)] = pending.as_slice() else {
            panic!("expected an approval request")
        };
        assert_eq!(pending.server_label, "deepwiki");
        let ContentPart::Other(answer) = pending.approve() else {
            panic!("expected an Other part")
        };
        assert_eq!(
            output_item(&answer, "openai"),
            Some(
                serde_json::from_value(json!({
                    "type": "mcp_approval_response", "approval_request_id": "mcpr_1", "approve": true
                }))
                .unwrap()
            )
        );
        let ContentPart::Other(denial) = pending.deny() else {
            panic!("expected an Other part")
        };
        assert_eq!(denial.data["approve"], false);
        // Answered in the same turn, it no longer waits.
        let answered = vec![ContentBlock::Other(request), ContentBlock::Other(answer)];
        assert!(!has_pending(&answered));
    }

    #[test]
    fn only_own_client_outputs_are_input_items() {
        let output = other(
            "custom_tool_call_output",
            json!({"call_id": "c1", "output": "ok"}),
        );
        assert_eq!(
            output_item(&output, "openai"),
            Some(
                serde_json::from_value(
                    json!({"type": "custom_tool_call_output", "call_id": "c1", "output": "ok"})
                )
                .unwrap()
            )
        );
        assert!(output_item(&output, "openrouter").is_none());
        assert!(output_item(&other("input_audio", json!({})), "openai").is_none());
    }
}
