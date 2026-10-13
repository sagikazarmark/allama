# ADR 0014: Client-executed tools that are not function calls

- Status: accepted

## Context

A function tool is declared with a JSON Schema, the model calls it with
JSON arguments (`ToolCall`), and the caller answers with a tool message.
Tools the provider runs itself are `ProviderToolCall`s (ADR 0008). A third
kind sits between them: tools the **client** runs whose calls are **not**
function calls.

- **OpenAI** has several, each with its own wire items: custom tools with
  freeform text input, optionally constrained by a Lark or regex grammar
  (Responses and Chat Completions); the local shell and shell tools; computer
  use; `apply_patch`. Each call and its answer have a distinct item type,
  and the local shell answer even names the call by `id` rather than
  `call_id`.
- **Anthropic** (bash, text editor, computer use) and **Gemini** (computer
  use) express their client tools as ordinary function calls with JSON
  input, answered by ordinary tool results. Only the tool *definition* is
  provider-specific, with one exception: a call from Anthropic's computer
  toolset (`computer_toolset_20260801`) names its toolset, and the result
  must repeat that name.

Freeform input exists only at OpenAI; Anthropic and Gemini take only JSON
objects as tool input. Shell and computer use share no vocabulary either:
Anthropic's bash keeps a session alive and runs one command string, OpenAI's
shells run an argv or a command list with per-command exit outcomes, and
computer-use action sets, coordinate systems and safety checks all differ.

## Decision

1. **No portable kind for these tools.** `ToolCall` keeps JSON arguments;
   there is no text input kind and no portable shell or computer-use call.
   A text input kind would be a one-provider concept in the core. Revisit
   when a second major provider ships freeform tool input, designing the
   neutral model first.
2. **Function-call-shaped client tools are ordinary `ToolCall`s.**
   Anthropic's bash, text editor and computer-use tools and Gemini's
   computer use are `ToolCall`s answered with tool messages, because that
   is their wire shape; only their definitions are provider options
   (Anthropic's `server_tools`).
3. **Anthropic's computer toolset is a provider-specific round trip on a
   portable call.** Its member calls stay `ToolCall`s answered by id, so a
   loop writes nothing provider-specific. The decoder keeps every
   `tool_use` field beyond `id`, `name` and `input` (`toolset_name`,
   `caller`, and fields Anthropic adds later) in the call's provider data
   under `anthropic` and sends them back on replay, and the encoder copies
   the call's `toolset_name` onto the `tool_result` that answers it.
   `server_tools` accepts a computer toolset definition without a `name`,
   since its members are named by the toolset.
4. **OpenAI's calls stay `Other` content, with typed views in the provider
   crate.** `allama_openai_compat::ClientToolCall` (re-exported by
   `allama_openai`) reads them: `Custom`, `LocalShell`, `Shell`,
   `McpApproval` (a hosted MCP call waiting for approval), and `Other` for
   items without a typed view (`computer_call`, `apply_patch_call`) or that
   do not parse. `ClientToolCall::pending` lists a turn's calls that have
   no output in the same turn (a hosted shell answers its own call). Each
   view builds its answer as an `Other` part with the call's origin, so a
   loop never hand-writes wire JSON or gets the `id`/`call_id` difference
   wrong. The views copy the call's origin, so they serve any
   Responses-compatible endpoint.
5. **Answers go in a user message after the assistant turn**, not a tool
   message. A tool message answers a `ToolCall` by id, and
   `history::validate` and every other adapter would treat it as an orphan
   result. The Responses encoder sends an own-origin answer part as a
   top-level input item, splitting the message around it.
6. **One custom tool call shape for both OpenAI APIs.** Responses and Chat
   Completions share the `openai` namespace, so they share one
   representation: an `Other` block `custom_tool_call` (`call_id`, `name`,
   `input`; Responses adds `id` and `status`) and an `Other` part
   `custom_tool_call_output` (`call_id`, `output`). The Chat Completions
   encoder replays the call as a `type: "custom"` tool call and lifts the
   answer into a `tool` message. It lifts only from the user message
   directly after the turn, and only answers no other content precedes, so
   lifting never reorders the conversation; a list output becomes text.
   This is shared because the namespace is shared, not because it is
   portable.
7. **Definitions stay in each API's own words**:
   `ResponsesRequestOptions::client_tools` and Chat Completions'
   `ChatRequestOptions::client_tools`, sent verbatim, so grammars and shell
   environments are never translated. Each API reads only its own options,
   so a conversation that fails over between them needs both.
8. **Loop safety.**
   - `tool_calls()` on responses, assistant messages and stream collectors
     returns function calls only, so a loop that knows only function tools
     never runs a custom or shell call as one.
   - Adapters report `FinishReason::ToolCalls` while a client-executed call
     waits, unary and streamed, even when `tool_calls()` is empty.
     Reporting `Stop` would let a loop end the turn with the call
     unanswered.
   - `history::validate` checks only `ToolCall`s. It reads no
     provider-specific content, so the provider reports an unanswered
     client call with its own error; teaching the core to check them would
     mean teaching it one provider's item types.
9. **Pairing and failover.** A call the encoder does not replay takes its
   answer with it, so a provider never sees an output without its call.
   - Chat Completions drops Responses-only calls and outputs (shell,
     computer use, `apply_patch`, MCP approvals, unknown types) with
     `Dropped` warnings, and drops an answer whose call is not in the
     request, never sending it as message content.
   - Responses sends an answer whose call is not in the request, because
     with server-side continuation (`previous_response_id`,
     `conversation`) the call lives on the server.
   - Every other adapter drops foreign client tool calls and answers with
     `Dropped` warnings under every policy (ADR 0002), and drops a reply
     the removals leave last, so the failover succeeds.

## Consequences

- An agent loop offering these tools on OpenAI asks
  `ClientToolCall::pending` after each turn, matches the kinds it can run,
  and fails loudly on the rest. The same loop serves both OpenAI APIs.
- A conversation holding these calls moves to another provider as it is;
  the target drops what it cannot send, with warnings.
- Fixtures written from the documented shapes and gated live tests
  (`responses_client_tools_round_trip`,
  `anthropic_computer_toolset_round_trip`) cover the round trips. Typed
  views for `computer_call` and `apply_patch_call` are left for later.

## Rejected alternatives

- **Synthetic `ToolCall`s with made-up JSON arguments.** Faking
  portability: it changes the wire shape on replay and loses the grammar
  and action semantics.
- **The Vercel AI SDK's mapping**, the smallest form of the above: a custom
  call becomes a tool call whose arguments are the freeform text as a JSON
  string, recognized on replay by finding its name among the custom tools
  declared in the same request.
  - `tool_calls()` would return it, so a function-only loop would run it
    with arguments that are not an object.
  - The history would not say which wire shape to replay. A history
    sent without the original tool definitions (a resumed checkpoint, a
    fallback target) would turn it into a call to a function never
    declared.
  - Anthropic and Gemini take only JSON objects, so failover would fail
    instead of dropping the call with a warning.
  - It adds no portable meaning: freeform input and its grammar remain
    OpenAI concepts dressed as portable ones.
