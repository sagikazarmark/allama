# ADR 0008: Provider-executed tool blocks

- Status: accepted

## Context

Providers run some tools themselves: Anthropic server tools (web search,
web fetch, code execution, tool search) and its MCP connector, OpenAI
Responses hosted tools (web search, file search, code interpreter, image
generation, remote MCP), and Gemini built-in tools such as code
execution. Callers need to see that such a tool ran, and code that runs
tools must never mistake one for a call it has to answer.

The wire shapes differ:

| Provider | Call | Outcome |
|---|---|---|
| Anthropic | `server_tool_use` / `mcp_tool_use` block with an id | a separate `*_tool_result` block |
| Gemini | `executableCode` part, no id | the following `codeExecutionResult` part |
| OpenAI Responses | one output item per call | the same item (`status`, `outputs`, `error`, …) |

## Decision

- Two content blocks, separate from `ToolCall`: `ProviderToolCall` (id,
  name, arguments) and `ProviderToolResult` (the call id, the output, and
  whether it failed). They are not `ToolCall`s with a flag:
  `ChatResponse::tool_calls` feeds tool loops, which must never see a
  call the provider already executed.
- **What is portable:** that a provider tool ran, its name, its input,
  which result belongs to which call, and whether it failed. The result
  `output` keeps the provider's shape verbatim; it is not normalized.
- Both blocks carry an origin and replay only to it (ADR 0002). Adapters
  keep the wire type in provider data and rebuild the wire form on
  replay: Anthropic re-emits both blocks; the Responses adapter splits an
  item into a call (input fields) and a result (everything else) and
  merges them back.
- Gemini's parts carry no ids, so the adapter links a result to the code
  part before it through a synthetic id; thought signatures stay in
  provider data and the turn replays verbatim.
- Streaming: `StreamBlockType::ProviderToolCall` streams its arguments
  like a tool call; `StreamBlockType::ProviderToolResult` is complete at
  start. Responses hosted-tool progress events become `ProviderDelta`s
  on the call block.
- Tool *definitions* stay provider options (Anthropic `server_tools`,
  Responses `hosted_tools`, Gemini `builtin_tools`), because what each
  provider offers and how it is configured does not align.
- Billable requests these tools make are counted in
  `Usage::tool_requests` (ADR 0005).
- Items the *caller* executes that are not function calls (computer use,
  local shell, custom tools) are not provider tools; ADR 0014 covers
  them.

## Consequences

- Upper layers can show "searched the web for …" without parsing
  provider payloads, and tool loops stay safe by construction.
- A caller who needs a tool's detailed results reads the verbatim
  `output` and knows its provider's shape.

## Rejected alternatives

- **`Other` content with an origin.** It replays correctly but gives
  callers nothing to read, and does not distinguish a provider-run tool
  from any other unknown block.
- **A flag on `ToolCall`.** Every tool loop would have to remember to
  filter on it, and forgetting would answer a call twice.
- **Normalizing result payloads** (search results, code output). The
  shapes differ per tool and per provider; a common shape would be fake
  portability.
