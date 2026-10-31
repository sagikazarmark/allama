# Writing an allama provider

This guide is for people who connect a new model API to `allama`: a hosted
service, a local server, a gateway, or an in-house endpoint. It covers which
building block to start from, the contract a provider keeps, streaming over
`allama-http`, and how to test it.

Start with the [quickstart](#quickstart): one page, the steps and the
concepts no adapter avoids, built on the shortest complete adapter. The
numbered sections after it are the reference: read them as your API
needs them.

Three runnable examples go with it.
[`text_gateway`](../../crates/allama-examples/examples/text_gateway.rs) is
the quickstart's adapter: a text-only gateway, unary and NDJSON streaming,
in one short file that passes every contract of the conformance aggregate.
[`provider_impl`](../../crates/allama/examples/provider_impl.rs)
shows the whole contract on one page with an in-memory "wire".
[`native_adapter`](../../crates/allama-examples/examples/native_adapter.rs)
is the native HTTP adapter the reference builds, for a fictional "Acme" API
with tools, reasoning and SSE: every code block of the reference is quoted
from it (a test checks the quotes), and its tests run the contracts of
[§6](#6-contracts-and-tests). API documentation is on docs.rs
([`allama`](https://docs.rs/allama), [`allama-http`](https://docs.rs/allama-http),
[`allama-conformance`](https://docs.rs/allama-conformance)), and terms are
defined in [`CONTEXT.md`](../../CONTEXT.md).

## Quickstart

First check [which starting point](#which-starting-point) fits: most
OpenAI-compatible endpoints need a preset or a dialect of about ten lines,
not an adapter. For an API with its own wire format, copy
[`text_gateway`](../../crates/allama-examples/examples/text_gateway.rs) and
change what differs. It is about 300 lines of adapter, a demo `main` and a
conformance test. Its code blocks below are quoted from it (a test checks
the quotes).

1. **Name it and declare what your wire format carries.** Export
   `PROVIDER_NAME` and `NAMESPACE` ([§2](#2-identity-and-capabilities)).
   Declare the request fields your wire format has a slot for, and the
   kinds of content it takes, as a
   [`WireSupport`](https://docs.rs/allama/latest/allama/adapter/chat/struct.WireSupport.html):

   ```rust,ignore
   const WIRE: WireSupport =
       WireSupport::text().fields([RequestField::System, RequestField::MaxOutputTokens]);
   ```

2. **Answer capabilities from it.** What the declaration cannot carry is
   `Unsupported` for every model; answer the rest (streaming, output) for
   your endpoint ([§2](#2-identity-and-capabilities)):

   ```rust,ignore
   fn chat_capability(&self, _model: &str, capability: ChatCapability) -> CapabilitySupport {
       WIRE.chat_capability(capability)
   ```

3. **Encode once, for the call and its dry run.** `chat` and `chat_stream`
   start with `adapter::chat::prepare`, and validation with
   `adapter::chat::prepare_dry_run`;
   both run the same `encode`, which first lets
   [`Prepared::degrade_to`](https://docs.rs/allama/latest/allama/adapter/chat/struct.Prepared.html#method.degrade_to)
   refuse or drop, with the right warnings, everything your wire format
   cannot carry ([Preflight first](#preflight-first),
   [Wire support](#wire-support-the-standard-degradation-pass)):

   ```rust,ignore
   fn encode(&self, prepared: &mut Prepared<'_>, stream: bool) -> Result<HttpRequest> {
       if stream {
           prepared.preflight_stream(self);
       }
       prepared.degrade_to(self, &WIRE)?;
       let (request, d) = prepared.parts_mut();
   ```

   Then encode what is left, let `retain_turns` skip empty turns
   ([Empty turns](#empty-turns)), and apply the caller's transport options
   with your credential slot (`apply_options`, [§5](#5-sending-and-streaming-with-allama-http)).
4. **Send, decode, and finish.** Call `mark_sent` right before sending,
   and bound the call: a unary request carries
   `allama_http::DEFAULT_REQUEST_TIMEOUT`, and a stream gets
   `StreamTimeouts` instead ([§5](#5-sending-and-streaming-with-allama-http)):

   ```rust,ignore
   let sent = match stream {
       true => StreamTimeouts::default().send(&*self.http, request).await,
       // A bound on the whole call: no transport adds one of its own.
       false => {
           let request = request.timeout(Some(DEFAULT_REQUEST_TIMEOUT));
           self.http.send(request).await
       }
   };
   ```

   Then decode the answer, and end with `finish`. A stream runs your
   `ChatFrameDecoder` over the body's frames (`NdjsonStream` or
   `SseStream`) with `FramedChatStream`, and ends with `finish_stream`
   ([§5](#5-sending-and-streaming-with-allama-http)).
5. **Map errors in one pure function** of status, headers and body, with
   the `allama-http` helpers ([§4](#4-errors)).
6. **Run the conformance aggregate**, `assert_chat_adapter`, against canned
   wire data on both paths ([§6](#6-contracts-and-tests)).

The concepts no adapter avoids:

| Concept | What it does for you | Reference |
| --- | --- | --- |
| [`Prepared`](https://docs.rs/allama/latest/allama/adapter/chat/struct.Prepared.html) (`prepare`, `prepare_dry_run`, `mark_sent`, `finish`, `finish_stream`) | runs preflight, carries the warnings to the response, makes validation a dry run | [Preflight first](#preflight-first) |
| [`WireSupport`](https://docs.rs/allama/latest/allama/adapter/chat/struct.WireSupport.html) and [`Degradations`](https://docs.rs/allama/latest/allama/adapter/struct.Degradations.html) | refuse, warn about or drop what you cannot send, never silently | [Wire support](#wire-support-the-standard-degradation-pass), [Degradations and warnings](#degradations-and-warnings) |
| [`EncodedTurn` and `retain_turns`](https://docs.rs/allama/latest/allama/adapter/chat/fn.retain_turns.html) | skip empty turns, and a reply left last by them | [Empty turns](#empty-turns) |
| [`ChatFrameDecoder`](https://docs.rs/allama-http/latest/allama_http/framing/trait.ChatFrameDecoder.html), [`FramedChatStream`](https://docs.rs/allama-http/latest/allama_http/framing/struct.FramedChatStream.html) and [`VendorFrames`](https://docs.rs/allama-http/latest/allama_http/framing/struct.VendorFrames.html) | the stream loop, its error rules, and frames you do not model | [§5](#5-sending-and-streaming-with-allama-http) |
| [`CredentialSlot`](https://docs.rs/allama-http/latest/allama_http/struct.CredentialSlot.html) and `HttpRequest::apply_options` | per-request headers and credentials | [§5](#5-sending-and-streaming-with-allama-http) |
| The error helpers ([`error_envelope`](https://docs.rs/allama-http/latest/allama_http/fn.error_envelope.html), `error_context`, `status_error`, `malformed_response`) | one error kind per upstream signal | [§4](#4-errors) |
| [`AdapterUnderTest`](https://docs.rs/allama-conformance/latest/allama_conformance/adapter/struct.AdapterUnderTest.html) and `assert_chat_adapter` | every contract from one test | [§6](#6-contracts-and-tests) |

Everything else (typed options, provider data, tools, reasoning, strict
schemas, embeddings, registration) is in the reference, only where your
API has it: the table in [Minimal path](#minimal-path) says which section
covers what.

## 1. Choose a starting point

### Minimal path

Check [which starting point](#which-starting-point) fits first: most
endpoints need a preset, or a dialect of about ten lines, not an adapter.
For a native adapter, read
[`provider_impl`](../../crates/allama/examples/provider_impl.rs) first,
the whole contract on one page, then
[`native_adapter`](../../crates/allama-examples/examples/native_adapter.rs)
section by section. Its banners mark each section
`// ── Required: … ──` or `// ── Only if your API has …: … ──`, with the
same split as this table:

| Every native adapter needs | Only if your API has… |
| --- | --- |
| [Identity and capabilities](#2-identity-and-capabilities): `PROVIDER_NAME`, `NAMESPACE`, `downcast_any`, and capability answers that describe your endpoint | per-model facts a caller may correct: a capability resolver, in [§2](#2-identity-and-capabilities) |
| [Preflight first](#preflight-first): `adapter::chat::prepare`, one `encode` shared with the dry run, then `finish` | request fields of its own: [typed options](#typed-options-provideroptions) |
| [Degradations and warnings](#degradations-and-warnings): every field is sent or reported, never skipped. Tools, a forced tool choice, a response format and media you cannot send are always refused, never warned about; a [`WireSupport`](#wire-support-the-standard-degradation-pass) declaration does the whole pass for a wire format that carries less than the request | reasoning: replay only your own state ([provider data](#provider-data-providerdata)) and fit it inside the caller's `max_output_tokens` ([Degradations and warnings](#degradations-and-warnings)) |
| [Non-exhaustive enums](#non-exhaustive-enums): the wildcard arm refuses | constrained decoding: [strict schemas](#strict-schemas) |
| [Empty turns](#empty-turns): let `retain_turns` decide what to send | refusal text or refusal stop reasons: [refusals](#refusals) |
| [Provider data](#provider-data-providerdata), its core rule: store state under your namespace, replay only your namespace, and drop the rest with a warning | an explanation of why it declined: `DECLINE_EXPLANATION_KEY`, in [provider data](#provider-data-providerdata); client tool items of its own (ADR 0014), in the same section |
| [Errors](#4-errors), with [message provenance](#message-provenance): your own messages, upstream text in its own slot | embeddings: the embedding entries of [§2](#2-identity-and-capabilities) |
| [Streaming](#5-sending-and-streaming-with-allama-http): a framer for your wire framing and the shared driver (`SseChatStream` for SSE), in [event order](#event-order) | a stream that is not SSE: [other wire framings](#other-wire-framings-ndjson-) (`NdjsonStream`, `FramedChatStream`); a key that is not a bearer token: the [credential slot](#5-sending-and-streaming-with-allama-http)'s `Header` and `Custom` placements, `replaces` and `required` |
| The adapter aggregate, `assert_chat_adapter`, passing on both paths ([§6](#6-contracts-and-tests)) | a live endpoint to record fixtures from: [`scripts/capture-fixture.sh`](../../scripts/capture-fixture.sh), in [§6](#6-contracts-and-tests) |

Registration ([§7](#7-register-with-allama-providers)) matters once
applications pick providers by model address.

### Which starting point

| The endpoint… | Start from | You write |
| --- | --- | --- |
| is already a preset (Groq, Together, Fireworks, DeepInfra, OpenRouter, DeepSeek, xAI, Mistral, Ollama, vLLM, Azure, Cloudflare, GitHub Copilot) | `allama_openai_compat::providers::<Preset>` | Nothing. Adjust with `<Preset>::builder(..)`. |
| speaks OpenAI Chat Completions, with its own URL, auth, and a few accepted or rejected fields | `allama_openai_compat::Provider` with a `CompatQuirks` dialect | A `CompatQuirks` value and a builder call, in your application or a small crate. |
| speaks the OpenAI Responses API | `allama_openai_compat::ResponsesProvider` with `ResponsesQuirks` (and `ResponsesModelRules` for per-model facts) | The same: a dialect and a builder call. |
| speaks Chat Completions but has request fields of its own | `allama_openai_compat::Provider`, or a thin crate that wraps it | Vendor fields in `allama_openai_compat::ChatRequestOptions::extra` (fields the adapter owns are refused), and your capability answers through the builder (`chat_capability`) or a resolver (`with_chat_capabilities`). A wrapper crate with an options type of its own maps it onto `ChatRequestOptions` on the request it forwards. The Chat Completions wire types are internal; the `WireOptions` hooks `allama-openai` builds on are `#[doc(hidden)]` and not a stable API. |
| has its own wire format (Anthropic Messages, Gemini, Bedrock, …) | a native adapter: implement `ChatProvider` (and `EmbeddingProvider` if it embeds) | Request encoding, response and stream decoding, and error mapping, sending through `allama-http`. |

Prefer the first row that fits. `CompatQuirks` covers which field carries
the output cap (`max_tokens` or `max_completion_tokens`), `seed`, strict
tool schemas, JSON Schema output, reasoning effort, the `developer` role,
prompt-cache keys, stream usage, parallel tool calls, `top_k`, the field
that replays reasoning (`ReasoningReplayField`), and vendor error codes that mean exhausted quota
(`quota_error_codes`). A preset is just a `CompatQuirks` plus a provider
identity and its environment variables; contribute one once it is verified
live against the real endpoint.

```rust
use allama_openai_compat::{CompatQuirks, Provider};

let mut quirks = CompatQuirks::new("openai-compat:internal");
quirks.supports_json_schema = false;
quirks.quota_error_codes.push("quota_exceeded".into());

let provider = Provider::builder()
    .base_url("https://llm.internal.example/v1")
    .bearer_token(token)
    .provider_name("internal")
    .quirks(quirks)
    .build()?;
```

The rest of this guide is about the last row, a native adapter.

Give every endpoint its own provider name and namespace. The namespace is
where the endpoint's opaque state lives, and state is only replayed to the
namespace that minted it.

## 2. Identity and capabilities

Every provider implements `ProviderIdentity` and the capability methods.
They are required: the compiler stops a provider (or wrapper) that forgets
them.

A provider has three identifiers. They are often the same string, but each
keys something different, so keep them apart:

| Identifier | What it keys | Who reads it | When it differs |
|---|---|---|---|
| Provider name (`provider_name()`, `PROVIDER_NAME`) | who served a call: `ChatResponse::provider`, errors, warnings, spans, fallback attribution | callers, logs and telemetry | the name a user knows the service by: Gemini reports `gemini`, Workers AI `cloudflare` |
| Namespace (`NAMESPACE`) | the provider data your adapter mints, and settings a caller addresses to you on a request item | your adapter alone; every other provider drops it as foreign state | where a wire format's state lives: Gemini stores state under `google-ai`, each OpenAI-compatible preset under `openai-compat:<name>` |
| Credential scope (`DEFAULT_CREDENTIAL_SCOPE`, builder `credential_scope`) | the account a per-request credential is addressed to | `HttpRequest::apply_options`, through your `CredentialSlot` | per account, not per wire format: two accounts of one provider need two scopes, and providers behind one gateway share one |

Changing the credential scope never changes the provider name or where
state lives. An options namespace (`ProviderOptions::NAMESPACE`) usually
equals the namespace; the bullets below cover when it does not.

- **`provider_name()`**: who serves calls, stamped on responses
  (`ChatResponse::provider`), errors, and warnings. Export it as a
  `PROVIDER_NAME` constant.
- **Namespace**: the key for your provider data, and usually for your
  options too. Export it as `NAMESPACE`, even when it equals the provider
  name. The options namespace may differ: every OpenAI-compatible preset
  reads options under `openai-compat` but stores state under
  `openai-compat:<name>`. Never read another provider's options namespace;
  the one shared exception is `allama-http`, the transport options read
  through `allama-http`'s own helper.
- **`reads_namespace(namespace)`**: `true` for the namespace your adapter
  reads provider data under (`namespace == NAMESPACE`), `false` for every
  other, options namespaces included. `retain_provider_state_for` keeps
  what it answers `true` for, so an application can move a conversation to
  your provider without knowing your namespace.
- **`downcast_any`**: return `allama::expose(self, id)`, so callers can
  reach your concrete type behind wrappers and `DynChatProvider` with
  `provider.downcast_ref::<YourProvider>()`.
- **`chat_capability(model, capability)`**: `Supported`, `Unsupported`, or
  `Unknown`, in three tiers (ADR 0003): a verified entry gets the table's
  answer; a parseable id in a known family, with no specialty marker
  (`-tts`, `-image`, `realtime`, ...), gets the family default; anything
  else gets `Unknown`. `Unsupported` describes your endpoint, not a model
  (ADR 0015): answer it only for what your API cannot carry for any model.
  What a model page says one model lacks is model data for the catalog, and
  so are which endpoints serve a model and which parameters a model rejects
  while it reasons: answer `Unknown`, send the request, and map the
  provider's refusal to `Error::unsupported_by_provider`. A caller re-adds
  such rules through your model-rules hook. Never match a family by a bare
  prefix (`o1` matches `o1-mini`). Every claim you
  add or remove, and every refusal or drop rule, needs a captured fixture
  or a cited provider doc in its test. The capability honesty contract
  (`allama_conformance::capabilities::assert_capabilities_honest`) probes
  every claim whatever you answer, so a `Supported` capability that is
  refused, warned about as not sent, or never reaches the wire fails it.
  One rule is deliberate: on `chat_stream`, a refusal before sending keeps
  a `Supported` claim. A capability answer has no path, and streaming can
  refuse what `chat` takes (Workers AI refuses JSON mode and tools when
  streaming), so the claim is judged on `chat`. Refuse before sending, and
  say so in `validate_chat_stream_request`; a warning that drops it on
  `chat_stream` still fails.
- **A capability resolver**: give your provider a
  `with_chat_capabilities(impl ChatCapabilityResolver)` method and consult
  the resolver first in `chat_capability`. Your `chat` reaches it through
  `adapter::chat::prepare(self, ..)`, so it can upgrade an answer as well as
  downgrade it; encode from the request, not from a second lookup of your
  own table, or an upgrade will not take effect.
  `allama::wrappers::CapabilityOverrideProvider` covers callers holding
  only a `DynChatProvider`, but it cannot upgrade your `Unsupported`.
- **`Unknown` never drops**: it means "send it and let the provider
  decide". Preflight leaves the feature in the request, so your encoder
  sends every field and content part or reports it through
  `Degradations`. A field your wire format cannot carry is `Unsupported`
  for every model, and an endpoint that silently ignores a field it
  receives (Ollama's `tool_choice`, a server with `drop_params`) answers
  `Unsupported` for it, or you encode an equivalent it honors.
- **`validate_chat_request` and `validate_chat_stream_request`**: a dry
  run of `chat` and `chat_stream`. They return a `ChatValidation`: the
  warnings the call would report, the capabilities the request relies on
  that you answered `Unknown` (`relied_on_unknown`), and how far the dry
  run went (`coverage`). The defaults check your capability answers only
  (`ValidationCoverage::Capabilities`); override both to run your encoding
  too and report `ValidationCoverage::Encoding` (see
  [§3](#preflight-first)).
- **`embedding_limits`** (embeddings): report the endpoint's limits, such
  as its batch limit (`EmbeddingLimits::new().max_inputs_per_call(2048)`),
  so `embed_chunked` can split large batches.
- **`validate_embedding_request`**: the default runs
  `allama::adapter::embedding::prepare_dry_run`, whose preflight rejects an empty
  batch, a batch over `EmbeddingLimits::max_inputs_per_call`, and anything that uses a
  capability you report `Unsupported` (images, composite inputs, batches,
  `dimensions`). Start `embed` with `allama::adapter::embedding::prepare` and end it
  with `finish`, as `chat` does, and override `validate_embedding_request`
  to run the same `encode` on `prepare_dry_run` and report `Encoding`
  coverage.
- **Embedding responses**: call
  `allama::adapter::embedding::check_response(request, &response)` on the vectors
  you decode, before `finish`, and attribute its error to yourself
  (`err.with_provider(..)`): a vector of the wrong size, of another size
  than the rest, empty, or holding a non-finite value is
  `Error::UnexpectedResponse`, never returned.
- **Embedding inputs** (ADR 0012): each `EmbeddingInput` is a list of parts
  combined into one vector. An endpoint that embeds one text per input calls
  `request.text_inputs(self.provider_name())?`, which refuses anything else
  instead of joining parts. Every adapter also calls
  `request.check_provider_data(self.provider_name(), NAMESPACE, &[..])?` with
  the keys it reads from inputs (`&[]` for none): unknown keys in your
  namespace on an input fail, keys on parts may be chat replay state and are
  left alone.
- **Embedding input type** (a hint, ADR 0003): map `EmbeddingRequest::input_type`
  (`Query`, `Document`) to your native field (Gemini `taskType`, an
  `input_type` field) and answer `EmbeddingCapability::InputType`. Answer
  `Unsupported` when you have no field (preflight then drops the hint with a
  warning, or fails under `UnsupportedPolicy::Error`), and never prefix the
  caller's text instead: the core preflight's message names the prefix a
  known prefix model's card documents (nomic, E5, BGE v1.5, Qwen3, …), or
  says the model may expect one. A model you positively identify as
  symmetric (it embeds queries and documents alike, as OpenAI's
  `text-embedding-3-*` do) honors the hint trivially: answer `Supported`,
  cite the evidence, and send nothing, under every policy. An
  OpenAI-compatible endpoint names its field with
  `CompatQuirks::embedding_input_type`. Run
  `probe::assert_input_type_not_honoured_warns` for models you report
  `Unsupported`.

Constructors that read the environment are called `from_env`. A missing
credential or base URL is `Error::configuration(..)`, never `Error::Auth`:
nothing reached the provider yet, and retrying or failing over will not fix
it.

## 3. Options, provider data, and warnings

These are the three escape hatches that keep the portable request honest.

### Typed options (`ProviderOptions`)

Request fields with no portable meaning go in an options type. Vendors add
request fields often, so make it `#[non_exhaustive]` with `new()` and a
setter per field: a field you add later is then not a breaking change.
Give it an `extra` map, so a caller can send a field your crate does not
model yet:

```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct ChatRequestOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_mode: Option<String>,
    /// Body fields this crate does not model yet, merged into the request.
    #[serde(skip_serializing_if = "ExtraMap::is_empty")]
    pub extra: ExtraMap,
}

impl ProviderOptions for ChatRequestOptions {
    const NAMESPACE: &'static str = NAMESPACE;
}
```

Read it while encoding the request with
[`ChatRequest::resolve_option`](https://docs.rs/allama/latest/allama/struct.ChatRequest.html#method.resolve_option),
which takes the typed value or its JSON form (`option` reads the typed
value alone; [`RequestOptions`](https://docs.rs/allama/latest/allama/struct.RequestOptions.html)
says how the two forms combine). `deny_unknown_fields` makes a misspelled
option fail instead of vanishing:

```rust
// In `chat`: accepts the typed value or the same data as namespaced JSON
// (`ChatRequest::with_options_json`), so options survive serialization.
let options: ChatRequestOptions = request.resolve_option()?;
```

Keep `deny_unknown_fields` with `extra`: in the JSON form, `extra` is a
nested object (`{"safety_mode": "strict", "extra": {"seed_offset": 3}}`),
not flattened into the options, so a misspelled option still fails instead
of reaching the wire as a body field. Merge `extra` into the body with
`#[serde(flatten)]`, and refuse a key your encoder owns, or it goes out
twice (ADR 0006):

```rust
// A field the adapter owns would go out twice: refuse it.
if let Some(key) = options
    .extra
    .keys()
    .find(|k| OWNED_FIELDS.contains(&k.as_str()))
{
    let message = format!("`{key}` cannot be set through ChatRequestOptions::extra");
    return Err(Error::invalid_request(message)
        .authored()
        .with_provider(PROVIDER_NAME));
}
```

Never add a vendor-only field to `ChatRequest`. If a feature might be
portable, design the neutral model first (an ADR in `docs/decisions`).

### Provider data (`ProviderData`)

Opaque state the provider needs back on the next turn (signatures, item
ids, encrypted reasoning) is stored on the block, message, or response
under **your namespace only**, and only your namespace is read back:

```rust
let state = ProviderData::new().with_raw(NAMESPACE, "signature", json!(sig));
ContentBlock::Reasoning(Reasoning::new(summary).with_provider_data(state))
```

When encoding, replay a block's state only from your namespace, and drop
the content you cannot replay with `degradations.dropped_replay_state(..)`,
a `Dropped` warning coded by its `ReplayState`:

```rust
// Replay only the reasoning this adapter minted.
ContentBlock::Reasoning(reasoning) => {
    match reasoning.provider_data.get_raw(NAMESPACE, "signature") {
        Some(signature) => wire.push(json!({
            "type": "reasoning",
            "text": reasoning.text,
            "signature": signature,
        })),
        None => {
            let from: Vec<&str> = reasoning
                .provider_data
                .foreign_namespaces(NAMESPACE)
                .collect();
            // Replay state stays in the conversation: a coded
            // drop logs one `WARN` per model, not one per turn.
            let (state, message) = if from.is_empty() {
                (
                    ReplayState::Reasoning,
                    "reasoning without an Acme signature cannot be replayed; it was dropped"
                        .to_owned(),
                )
            } else {
                (
                    ReplayState::ForeignReasoning,
                    format!(
                        "reasoning from {} can be replayed only to its origin; it was dropped",
                        describe_origins(&from)
                    ),
                )
            };
            degradations.dropped_replay_state(state, subject, message);
        }
    }
}
```

That rule is what makes a conversation safe to continue on another
provider: each adapter ignores state it did not mint. Report the state you
skip with `ProviderData::foreign_namespaces(NAMESPACE)`, which leaves out
your namespace and the core-owned `"allama"` one (`ALLAMA_NAMESPACE`), and
`allama::adapter::describe_origins` words the namespaces as the bundled
adapters do (`` `openai`, `gemini` ``, or "an unknown origin"). Name
the message (and block) in the warning's `Subject::Content`. The rule applies to
`OtherContent` too: wire fields you do not model go verbatim in its `data`
(any key is allowed there), and state you keep *about* the content, such
as a signature, goes in its `provider_data`. Replay `Other` content only
when its `origin` is your namespace, and gate its provider data like any
other block's. Drop another provider's `Other` content with
`dropped_replay_state(ReplayState::ForeignContent, ..)`, in user and tool
parts as in assistant blocks, and never refuse
the request over it: a failover would fail under every policy. An `Other`
part with no origin is the caller's own raw part; send it verbatim when
your wire can hold it. If several of your endpoints share one namespace
(OpenAI's Chat Completions and Responses both write `"openai"`), each
replays own-origin `Other` content only for the types it knows, and drops
the rest with a warning. When your API has client tool items of its own
(ADR 0014), drop an answer together with its call when you do not replay
the call. Never read a message's or system prompt's `app_data`: it
belongs to the application, so it is neither sent nor reported. Offer the
`raw_response` request option convention (`RAW_RESPONSE_KEY`) if callers may
need the verbatim response. When your API declines on the model's behalf
and explains why (a refusal stop reason with details, a safety finish with
a message), keep its decline data in your response-level provider data and
store the explanation's text, verbatim, under `DECLINE_EXPLANATION_KEY` in
your namespace too, so extraction's `ChatExtractError::Refused` quotes it. A
refusal the model writes is content, not provider data.

Tool call ids cross providers too: a history carries the ids the first
provider issued, or its adapter made up, and APIs disagree on what they
accept (Anthropic takes only `[A-Za-z0-9_-]`, OpenAI at most 64
characters, Mistral exactly nine letters and digits). If your API
restricts them, send every id through
`allama::adapter::chat::ToolCallIdFormat::conform`, in the assistant's
calls and in the results that answer them alike. An id your API accepts
goes out unchanged; any other becomes a stable id derived from it, so a
call and its result stay paired and a cached prefix stays the same. Report
nothing: the id is only a correlation key, and nothing is lost.

#### Keep reading every shape you wrote

Applications store conversations (`allama::history::History`), and the
provider data you write goes with them: the keys under your namespace and
the values' shapes. That makes them part of the stored format, but the
core's `HISTORY_FORMAT_VERSION` does not cover them, so no version bump
protects a history from a change you make. **Keep reading every shape you
ever wrote.** To rename a key or restructure a value, write the new shape
and keep reading the old one too, for as long as histories written with it
may exist. A test that replays a stored history through your encoder (as
`allama-anthropic`, `allama-gemini` and `allama-openai` replay the golden
history in their `tests/history_replay.rs`) is
what catches a change that strands them. The rules for the core format
are in the `allama::history` docs ("Format evolution").

#### Typed entries

Response-level data a caller may want to read (the region that served the
call, a system fingerprint) goes in the response's provider data, under
your namespace too. Give it a typed entry, a small struct that implements
`ProviderDataEntry` with your namespace and a key, so callers read it with
`response.provider_data.get::<AcmeRegion>()?` instead of a raw key, and
export it from your crate:

```rust
/// The Acme region that served a call: response-level provider data, read
/// with `response.provider_data.get::<AcmeRegion>()`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcmeRegion {
    /// The region, such as `eu-west`.
    pub region: String,
}

impl ProviderDataEntry for AcmeRegion {
    const NAMESPACE: &'static str = NAMESPACE;
    const KEY: &'static str = "region";
}

/// The response-level provider data of a call Acme served in `region`,
/// on the unary and the streaming path alike.
fn region_data(region: &str) -> Result<ProviderData> {
    let mut provider_data = ProviderData::new();
    provider_data.insert(&AcmeRegion {
        region: region.to_owned(),
    })?;
    Ok(provider_data)
}
```

Set it on the unary response:

```rust
if let Some(region) = &wire.region {
    response = response.provider_data(region_data(region)?);
}
```

and emit it in a stream with `StreamEvent::response_provider_data`, which
may come at any point of the stream, so `chat` and a collected
`chat_stream` give the same response:

```rust
// The collector merges it into the response's provider data.
if let Some(region) = json.get("region").and_then(Value::as_str) {
    out.push(region_data(region).map(StreamEvent::response_provider_data));
}
```

Only response-level data goes here. State you need back on the next turn
belongs on the block it is about, as above: only a response's content is
carried into the next turn (`ChatResponse::to_assistant_message`).

### Refusals

When your API marks text in which the model declines to answer (OpenAI's
`refusal`), decode it as a `ContentBlock::Refusal`, and in a stream as a
`StreamBlockType::Refusal` block of `RefusalDelta`s: never as text, so no
caller shows it as the answer. A decline without refusal text is a finish
reason (`FinishReason::Refusal` or `ContentFilter`) with the details in
response-level provider data. Encode a `Refusal` block explicitly: in your
refusal form if you have one, otherwise as assistant text, without a
warning, since nothing is lost (the `native_adapter` example does this).

### Empty turns

A message can reach your encoder with nothing to send: an assistant reply
whose blocks were all another provider's state, a user message emptied by
`retain_provider_state`, or one your own drops emptied. Never send it as an
empty wire message, and never warn about skipping it: nothing was lost, and
whatever you dropped from it was reported already. Encode each message,
then let `allama::adapter::chat::retain_turns` decide what to send:

```rust
let mut turns = Vec::with_capacity(request.messages.len());
for (index, message) in request.messages.iter().enumerate() {
    let wire = encode_message(index, message, degradations)?;
    let turn = EncodedTurn::new(index, message, wire.content.is_empty());
    turns.push((turn, wire));
}
// Skip empty turns, and drop a reply that the skips and drops left last.
let turns = retain_turns(turns, degradations, |(turn, _)| *turn)?;
```

`EncodedTurn::new` takes the message's index, the message, and whether
your encoding of it is empty; a message that is itself empty
(`Message::is_empty`) counts too, and a tool message never does, since an
empty result is a valid answer. `retain_turns` skips the empty user and
assistant turns, and drops an assistant turn that ends the conversation
only because the turns after it were skipped, with a `Dropped` warning
(under every policy). A reply the caller sent last on purpose, as a
prefill, stays. When nothing is left, it fails with `InvalidRequest`
before anything is sent. If your encoder splits one message into several
turns, or builds a turn from part of one, describe each turn with
`EncodedTurn::user`, `assistant` or `tool` and the index of the message it
came from, so warnings keep the caller's indices. Merge adjacent same-role
turns afterwards if your API requires alternation.

### Preflight first

`chat` and `chat_stream` start with `allama::adapter::chat::prepare`, which runs
`allama::adapter::chat::preflight`. It applies your capability answers (resolver
overrides included) and the request's `UnsupportedPolicy` (unset means
`Warn`, ADR 0003):

- a **semantic** feature you report `Unsupported` (tools, a forced tool
  choice, structured output, image or file input, assistant images) fails
  with `Error::Unsupported` under every policy. There is no warning path
  for them: a tool you cannot send changes what the model can do, so it is
  refused, never dropped with a warning (ADR 0003);
- an unsupported **hint** (reasoning configuration, prompt-cache settings
  and breakpoints, image detail) is removed from the request with an
  `Unsupported` warning, or fails under `UnsupportedPolicy::Error`;
- `parallel_tool_calls` is left alone: `true` is permission, never a
  degradation, and `false` is honored by a model that makes one call per
  turn. If your wire format has no such field, drop `true` silently and
  report `false` as an unsupported hint;
- reasoning blocks a model cannot replay are removed with a `Dropped`
  warning;
- `Unknown` never blocks and never removes anything; it is recorded, so
  validation can report what it could not know.

`prepare` returns a [`Prepared`](https://docs.rs/allama/latest/allama/adapter/chat/struct.Prepared.html)
request. Split your call path where the
transport sends: everything before it goes in one `encode` method, which
encodes `prepared.request()` and reports through
`prepared.degradations_mut()` (or both at once with `parts_mut`):

```rust
fn encode(&self, prepared: &mut Prepared<'_>, stream: bool) -> Result<HttpRequest> {
    if stream {
        // What a stream relies on, so its validation reports `Unknown`
        // answers.
        prepared.preflight_stream(self);
    }
    let (request, degradations) = prepared.parts_mut();
    let body = encode_body(request, stream, degradations)?;
```

The call prepares, encodes, marks the request sent immediately before the
transport sends it, and ends with `finish`, which joins the preflight and
encoding warnings to the response (`finish_stream` puts them right after a
stream's `ResponseStart`):

```rust
let mut prepared = allama::adapter::chat::prepare(self, request)?;
let http = self.encode(&mut prepared, false)?;
// Immediately before sending: from here on, an error (a 429) still
// reports the warnings as warning events.
prepared.mark_sent();
let response = self.send(http, false).await?;
```

Validation is the same encoding on a dry run, which never sends and never
emits warning events:

```rust
fn validate(&self, request: &ChatRequest, stream: bool) -> Result<ChatValidation> {
    let mut prepared = allama::adapter::chat::prepare_dry_run(self, request)?;
    self.encode(&mut prepared, stream)?;
    Ok(prepared.into_validation(ValidationCoverage::Encoding))
}
```

`finish` and `finish_stream` also emit each warning as a `tracing` event
(feature `warning-events`), once: a warning carries a mark once emitted,
so a layer that passes another provider's warnings through its own
`finish` emits only its own. `?` needs no ceremony: dropping a `Prepared`
on an error path never panics. If the request reached the transport
(`mark_sent`), its warnings are emitted then, because the degradations
happened even though the call failed; if it was refused before sending,
nothing is emitted, and the error is the only signal. A copied adapter
that forgets `mark_sent` loses only those error-path events.

Removing reasoning shifts the block indices of the messages that held it.
Preflight records the shift in the `Prepared` request's `Degradations`,
which translates the block index of every `Subject::Content` warning
recorded after it back to the caller's request. So report indices
relative to the request you received, the preflighted one, and never
translate them yourself. If your encoder collects warnings on its own,
record them into the same helper (`degradations.extend(encoder_warnings)`),
not next to it: warnings kept apart still name the preflighted blocks. An
adapter layered on another provider (a toolkit dialect, as `allama-openai`
is on `allama-openai-compat`) runs that provider's preflight too with
`prepared.preflight_with(&inner)`, and `finish` translates the warnings of
the response the inner provider returns.

The responsibilities are split:

- **`chat` and `chat_stream` MUST enforce refusals themselves.** Nothing
  calls `validate_chat_request` for them, and wrappers or callers may skip
  it. Preflight in `chat` is what makes a request you cannot serve fail
  before it is sent.
- **`validate_chat_request` and `validate_chat_stream_request` are for
  callers** that want to know before sending, for example at startup or to
  pick a provider. Run the call's own `encode` in them, so they agree with
  the call exactly: the same refusal, the same warnings.
  `allama_conformance::assert_validation_agrees` checks it (see
  [§6](#6-contracts-and-tests)). Validation never checks credentials or
  that the model exists; those need a real call.

`embed` does the same with `allama::adapter::embedding::prepare`, and its dry run
`validate_embedding_request` runs the same `encode`
(`allama_conformance::assert_embedding_validation_agrees` checks it);
`decide` does the same with `allama::adapter::decision::prepare`, and
`extract_values` with `allama::adapter::extraction::prepare`.

An extraction adapter (`ExtractionProvider`, unstable, [ADR
0018](../decisions/0018-extraction-models.md)) has two rules of its own.
Every field parameter is semantic: send it as given, or refuse the field
with `Degradations::reject` before sending; nothing is folded into the
instructions. And a text span is a byte range into the request's input
that equals the value: convert the provider's offsets (its unit, its copy
of the input), fall back to finding the value, and drop a span you cannot
place with a `Dropped` warning rather than report one that lies. Provider
statistics about an answer (a confidence, a guess for a null) are provider
data keyed by field name, as `allama-cloudflare` keeps Clairic's; call
`ExtractionResponse::validate` on what you decode.

### Wire support: the standard degradation pass

An API that takes less than the full portable request (a text-only
gateway, an endpoint without sampling controls) declares once what its
wire format carries, as a
[`WireSupport`](https://docs.rs/allama/latest/allama/adapter/chat/struct.WireSupport.html),
and calls
[`Prepared::degrade_to`](https://docs.rs/allama/latest/allama/adapter/chat/struct.Prepared.html#method.degrade_to)
first in `encode`. It applies the rules of the next section to everything
the declaration leaves out, so your encoder only encodes what it can send:

- it refuses before sending, under every policy, what must not be lost:
  tools and tool turns, a forced tool choice, a JSON response format, an
  output cap, system prompts, images and files, and an `Other` part
  without an origin (the caller's raw part);
- it removes hints with an `Unsupported` warning coded
  `unsupported.<field>`, or refuses them under `UnsupportedPolicy::Error`:
  `temperature`, `top_p`, `stop`, the penalties, `seed`, reasoning
  configuration, prompt caching, image detail, and
  `parallel_tool_calls: false`;
- it removes silently what is honoured anyway: `parallel_tool_calls:
  true`, an automatic tool choice, a text response format,
  `ReasoningEffort::None`, and a disabled tool choice, by leaving the tools
  out;
- it drops replay state with a coded warning (reasoning it cannot take
  back, another provider's `Other` content and provider tool blocks
  whatever you declare, your own when you do not declare
  `provider_content`), and records what it removed, so later warnings
  name the caller's blocks.

Answer capabilities from the same declaration
(`WIRE.chat_capability(capability)` is `Unsupported` for what it cannot
carry), so preflight and the encoder agree. The pass still runs after
preflight because a capability resolver may answer `Unknown` or
`Supported` for something the wire format cannot carry. What it leaves is
yours, under the same rules (refuse what is semantic, warn about a hint
you drop or change):

- the limits within a field you declared: how many `stop` sequences your
  API takes, the range of `temperature`, an output cap above the model's
  maximum;
- tool details once you declare tools: `ToolDefinition::strict` when your
  API has no strict mode (warn that it is not honoured; the
  `StrictNotHonouredWarns` contract checks it), and `ToolMessage::is_error`
  when your tool results carry no error flag (say so in the result text,
  with an `Approximated` warning);
- media details once you declare a kind: the media types, sizes and
  sources (URL, inline data, a provider file) your API takes;
- an assistant `Refusal` (send it as text when you have no refusal form);
- your own provider options and their `extra` fields;
- the wildcard arms of the [non-exhaustive enums](#non-exhaustive-enums). The
[`text_gateway`](../../crates/allama-examples/examples/text_gateway.rs)
example is built on it; `native_adapter` encodes field by field instead,
as the rest of this section shows.

### Degradations and warnings

Never drop a request field silently. Fields no capability covers go
through the same [`Degradations`](https://docs.rs/allama/latest/allama/adapter/struct.Degradations.html),
with the request's `UnsupportedPolicy`. Pick the call by what happens to
the request (ADR 0003). Tools are always the first row: a tool you cannot
send is never a hint:

| What happens to the request | Under `UnsupportedPolicy::Error` | Warning kind | Call |
| --- | --- | --- | --- |
| Its meaning would change (a response format, tools, a forced tool choice you cannot send) | always an error, under every policy | none: the request is refused | `unsupported(field, ..)`, `reject(subject, ..)` |
| A hint is not sent (`seed`, a penalty, a cache breakpoint) | refused before sending | `Unsupported` | `unsupported_hint(field, ..)`, `hint(subject, ..)` |
| It is sent as an approximation the model sees differently (malformed tool arguments sent as `{}`) | refused before sending | `Approximated` | `coded_lossy_hint(subject, .., code)`: history repeats it, so it is always coded |
| It is mapped to the closest equivalent, and the caller's intent is kept (an effort level rounded to the nearest one) | never refused | `Approximated` | `approximated(subject, ..)` |
| The same, for content history repeats on every turn (a tool error sent as a plain result where your API has no error flag) | never refused | `Approximated` | `coded_approximated(subject, .., code)`: one warning per code per request |
| Content or state is dropped (a file name, another provider's reasoning) | never refused | `Dropped` | `dropped(subject, ..)`, `dropped_replay_state(state, subject, ..)` for replay state |

Two worked rows:

| Situation | Call | Why |
| --- | --- | --- |
| Your API has no system role, so you send the system prompt as a user turn | `coded_lossy_hint(Subject::System { index }, "Acme has no system role; the system prompt was sent as a user turn", "approximated.system")` | the model sees the instructions as the user's words, so a strict caller must be able to refuse it |
| `ToolChoice::Disabled` on an endpoint that has no tool choice field, so you leave the tools out | nothing, under every policy | not offering the tools is exactly "the model must not call tools": the request is honored, not approximated. When omitting a field or content is exactly equivalent to what the caller asked for, record no warning |

In the Acme encoder:

```rust
if request.seed.is_some() {
    // A hint: dropped with a warning, or an error under UnsupportedPolicy::Error.
    degradations
        .unsupported_hint(RequestField::Seed, "Acme has no seed; `seed` was not sent")?;
}
if request.stop.len() > MAX_STOP_SEQUENCES {
    // Semantic: changing the meaning of the request is always an error.
    return Err(degradations.unsupported(RequestField::Stop, "at most 4 stop sequences"));
}
```

Every warning `Degradations` records about a request field or a system
prompt gets the code `<kind>.<field>` (`unsupported.seed`) unless it has
one, so its warning event recurs at `DEBUG` instead of `WARN` on every call.
A warning about content needs a code only when it recurs with the
request's shape. Replay state does: a conversation that moved to you holds
another provider's reasoning, provider data and `Other` content on every
turn, so record its drop with `degradations.dropped_replay_state(state,
subject, message)`. The `ReplayState` sets the code
(`dropped.foreign_reasoning`, `dropped.foreign_provider_data`,
`dropped.foreign_content`, `dropped.reasoning_replay` for reasoning you
cannot take back whoever wrote it, or `dropped.content_replay` for content
of your own namespace you cannot take back, such as a Responses item
replayed to Chat Completions), and for another provider's state the
message points the application to `retain_provider_state`. A limitation
the conversation repeats on every turn recurs too: if your wire format has
no tool error flag, send the error text as the result and record it with
`degradations.coded_approximated(subject, message, "approximated.tool_result_error")`,
which no policy refuses: the model still reads the error, only the flag is
lost. Tool call arguments that are not a JSON object, which you send as
`{}`, change what the model is asked to act on, so record them with
`degradations.coded_lossy_hint(subject, message, "approximated.tool_call_arguments")`,
which a strict policy refuses in any message.
Both helpers keep one warning per code per request: a later one with the
same code joins the first, listing the messages, so the warnings do not
grow with the conversation.
For anything else that recurs, build the warning with
`Warning::with_code` and record it with `degradations.warn(..)`. Leave a drop of content the application sent
uncoded, so it warns every time.

Never loosen a bound the caller set (ADR 0003, principle 4).
`max_output_tokens` includes reasoning, so fit reasoning into it, not it
around reasoning: a reasoning hint that does not fit (a thinking budget you
derived from the effort) is lowered with `degradations.approximated(..)`,
since the caller's bound holds and nothing is left to refuse, or dropped
with `degradations.unsupported_hint(..)` when nothing fits. Keep the
lowered value monotone in the cap: a larger cap never gives the hint less
room. A provider option that contradicts the cap is
`Error::invalid_request(..)`. Raise a cap only when it is your own default,
because the wire format requires one.

Then end the call with `finish`, which joins the preflight and encoding
warnings to the response and emits them as warning events
(`finish_stream` yields them as `StreamEvent::Warning` right after
`ResponseStart`, see [Event order](#event-order)):

```rust
Ok(prepared.finish(response))
```

Apply `Degradations` while mapping the request, before anything is sent.
Every refusal it returns says what it refused (`Error::unsupported_subject`:
the field of `unsupported(field, ..)` or of a hint, the subject you pass to
`reject`, translated to the caller's indices) and where:
`Error::unsupported_stage()` is `UnsupportedStage::BeforeSending`, which
callers read as "nothing was sent": extraction's `Auto` mode retries a
refused `ResponseFormat` in forced-tool mode, which is safe only because
the refused request never left. So never return a `Degradations` refusal
after the request reached the wire. The C2 audit
(`allama_conformance::probe::assert_request_fields_accounted`) fails a
`BeforeSending` refusal that comes after sending, and every probe contract
fails a refusal at another stage: an `Error::unsupported(..)` built in your
encoder says nothing about sending, so a router cannot fail over on it
safely. A refusal by warning names what it refuses: the capability
contracts accept only a warning whose subject is the capability's field or
content.

Refusals read `cannot send <subject>: <reason>` whatever their kind. A
hint's message is the warning it becomes under `Warn`. Under
`UnsupportedPolicy::Error` nothing was sent, so `Degradations` leaves the
outcome out of the refusal and keeps the limitation and the remedy:
``cannot send `seed`: Acme has no seed``. A message you only ever refuse
with (`unsupported`, `reject`) has no outcome: `<limitation>[; <remedy>]`.
The C2 audit fails a strict refusal that says the hint was ignored,
dropped or sent, so a remedy must not say so either.

When the provider itself refuses something the caller asked for (an HTTP
400 that names an unsupported parameter), map it with
`Error::unsupported_by_provider(Some(subject), ..)` (stage `Provider`),
and only through a table from your wire paths to `RequestField`s, for
reply patterns a captured fixture pins. A parameter you got wrong yourself
is an `InvalidRequest`. Give your error envelope builder (see
[§4](#4-errors)) an `ErrorKindProbe::Unsupported` reply, so the error kind
and in-band stream error contracts check its stage. `Error::unsupported(..)` states no stage, so callers must
assume the request may have been sent.

`allama_conformance::contract::{assert_unsupported_hint_warns,
assert_strict_policy_rejects}` check both sides on `chat`, and
`assert_unsupported_hint_warns_stream` and
`assert_strict_policy_rejects_stream` on `chat_stream`: the stream must
carry the `Unsupported` warning as a `StreamEvent::Warning`, and under
`UnsupportedPolicy::Error` it must fail before its first event. Run all four.

#### How a message reads

Every warning, hint and refusal message has one shape (ADR 0003, "how a
warning reads"):

- **`<limitation>; <outcome>[; <remedy>]`.** What you lack, what happened
  to the request, and, when there is one, what the caller can do:
  ``"Acme has no seed; `seed` was not sent"``. `"; "` separates the
  parts and appears nowhere else; inside a part use a comma, a colon or
  parentheses. Write the limitation so it reads true on its own, because
  a refusal gives it as its reason.
- **The outcome's verb follows the `WarningKind`.** `Unsupported`: "`x`
  was not sent" (or "… was sent without `x`" when `x` qualifies something
  you sent). `Approximated`: "… was sent as X" or "X was used instead".
  `Dropped`: "it was dropped" (or "`x` was dropped" for a part of the
  subject, such as a message's name). `Fallback`: "served by `X`".
- **Identifiers go in backticks:** request fields, literal values, error
  kinds, provider names, model ids, option namespaces, and the items a
  remedy names. Protocol names (Chat Completions, Anthropic) are not
  identifiers.
- **Don't repeat what the display adds.** A warning displays as
  `[provider] ` and, for content and embedding inputs, the location
  (`message 3, block 0: `), so the message names neither your instance
  nor the block. Name a protocol, endpoint or provider only where it is
  the reason ("Chat Completions has no tool error flag").
- **A warning about replay is coded.** What a model or provider wrote
  into the conversation, and a limitation the conversation's shape
  repeats, recur on every turn: record them coded (see above). A drop of
  content the application sent stays uncoded.

### Non-exhaustive enums

`Message`, `ContentPart`, `ContentBlock`, `ToolChoice`, `ToolResultContent`,
`ChatCapability` and most other portable enums are `#[non_exhaustive]`, so a
`match` on them needs a wildcard arm (without one the compiler stops with
`E0004: non-exhaustive patterns: '&_' not covered`). In request encoding,
make that arm refuse: a variant added after your adapter was written is
input you do not know how to send, and skipping it with `_ => {}` drops it
silently.

```rust
// `Message` is `#[non_exhaustive]`: refuse what this adapter does not know.
_ => return Err(degradations.reject(subject, "unsupported message type")),
```

`Degradations::reject` returns an authored `Error::Unsupported` naming the
subject (`None` when the refusal is about the request as a whole). In
`chat_capability`, the wildcard answers `CapabilitySupport::Unknown`
instead: you cannot vouch for a capability you have not heard of.

### Strict schemas

`strict` on a tool or a JSON Schema response format asks for constrained
decoding. Grammar engines accept a subset of JSON Schema, so rewrite the
caller's schema late, into your engine's dialect, and hand the dialect to
`allama::adapter::schema::apply_strict_policy`, which decides what goes out. For a
tool, call `apply_strict_policy(&tool.parameters, RequestField::Tools,
&subject, &mut degradations, |schema, d| my_grammar(schema, d),
Degradable::Yes)?`, where `subject` names the schema in messages
(``"tool `lookup`"``), and send the returned `StrictOutcome`'s `schema`
with `strict` set to its `strict`.

The dialect is any `FnOnce(&Value, &mut Degradations) -> Result<Value,
StrictSchemaError>`: your own grammar, built from the neutral passes in
`allama::adapter::schema` (`inline_refs`, `strip_keywords`, `flatten_single_all_of`,
`merge_sibling_refs`, `branches_disjoint`, `is_recursive`), or a bundled
one. If your endpoint speaks OpenAI's grammar, pass
`allama_openai_compat::schema::strict_schema` with
`SchemaDialect::OpenAiStrict` (`.for_model(model)` picks the fine-tuned
variant for OpenAI's `ft:` models; call it only if your endpoint follows
OpenAI's naming) or `SchemaDialect::AzureStrict`; for Anthropic's Messages
API on another platform, `allama_anthropic::schema::strict_schema`. Neither
module needs HTTP.

A dialect records its `Approximated` warnings only when it succeeds: on
failure the policy sends the caller's schema non-strict, and a warning
about a rewrite that was never sent would be false. Rewrite into a copy and
record at the end, or collect into a scratch `Degradations` and `extend`
the request's helper with it on success.

If `SchemaDialect::parallel_tool_calls_break_strict()` holds for your
dialect, turn parallel tool calls off yourself when a tool goes out strict
and the caller set no value, and report an explicit `true` as an
`Unsupported` warning on `RequestField::Tools`: extraction does not set
`parallel_tool_calls` for you.

Two passes keep common schemars output exact:

- Run `merge_sibling_refs` first, on the caller's schema, before anything
  closes objects. schemars puts sibling keywords next to a `$ref` (the tag
  beside an internally tagged newtype variant's struct, a `description`
  beside a documented field); the pass flattens each into one object and
  never emits `allOf`. `SiblingRefMerge::into_strict` turns a node it
  reports in `unmerged` into `StrictSchemaError::NotRepresentable`, and
  `?` turns its `SchemaError` into `StrictSchemaError::Invalid`.
- Rewrite `oneOf` as `anyOf` silently only when `branches_disjoint` (or
  `branches_disjoint_in`, if you keep `$ref`s) proves the branches
  disjoint, which holds for every enum shape schemars writes; otherwise
  report the widening as `Approximated`.

Refuse, rather than strip, a composition keyword your engine rejects
(OpenAI's `allOf`, `not` and `if`/`then`/`else`, for example): report it
as not representable, naming the keyword.
`SchemaDialect::rejected_keywords()` lists OpenAI's.
Report every rewrite that widens what the schema accepts as an
`Approximated` warning on the field, and keep silent only for rewrites
that change nothing the target type can deserialize. Extraction turns
those warnings into `SchemaEnforcement::Approximate`.

If your adapter sends schemas to a grammar engine unchecked, because you
have no dialect for it or the API has no strict mode that rejects what it
cannot enforce, say so: add one `Approximated` warning on the field. The
compat providers do this for an endpoint configured with
`schema_dialect: None` ("schema sent unchecked to this endpoint's grammar
engine"), and Gemini names each construct outside its documented
`responseJsonSchema` subset. Silence would make extraction report `Exact`.
The same holds for a dialect you apply but nobody has verified the
endpoint enforces: a custom compat endpoint (`dialect_verified: false`)
reports that it is "not known to enforce" the dialect exactly.

Keep three facts apart, because endpoints differ on each: whether the
response format constrains decoding (only when sent strict, always, or
never), whether a tool's `strict` is honoured, and whether a forced tool
choice constrains the forced call. The compat toolkit spells them
`response_format_constraint`, `tool_constraint` and
`forced_tool_choice_constrained`. An endpoint that always constrains
takes no `strict` flag; one that ignores the flag silently must be
configured `Never`, so a strict schema is reported `Unsupported`.

If the API limits strict schemas per request rather than per schema
(Anthropic caps strict tools, optional parameters and union-typed
parameters across a request), count the rewritten schemas before the
policy runs: run the dialect into a scratch `Degradations` for each
schema, count, and degrade strict tools from the last one until the
request fits by handing the policy a
`StrictSchemaError::not_representable(reason)` naming the limit. The Anthropic adapter does this.

The dialect tells two failures apart, and the policy (ADR 0003) acts on
them the same way for every grammar:

- `StrictSchemaError::NotRepresentable` (built with
  `StrictSchemaError::not_representable`): a schema the dialect cannot
  express (a map, a recursive type on an engine without recursion). On a
  field with a non-strict form (`Degradable::Yes`: tools, and response
  formats on APIs that have a non-strict mode) the caller's schema goes
  out non-strict, minus `$schema`, with an `Unsupported` warning, or the
  call fails under `UnsupportedPolicy::Error`. On a field with none
  (`Degradable::NoUnconstrainedForm`, such as Anthropic's
  `output_config.format`) it fails under every policy through
  `degradations.unsupported(field, ..)`, so `Error::unsupported_field` is
  set, the stage is `BeforeSending`, and extraction's `Auto` mode can fall
  back to a forced tool.
- `StrictSchemaError::Invalid`: a malformed schema (an unresolved `$ref`)
  is always an `Error::InvalidRequest`.

A partly rewritten schema is never sent without strict.

The same holds when the endpoint or model has no strict mode at all: report
the dropped `strict` as an unsupported hint on `RequestField::Tools` or
`RequestField::ResponseFormat`, never drop it silently. Extraction reads
those warnings into `ChatExtractionMetadata::enforcement`, counting only the
field its mode relies on (the response format in `Native` mode, the tools
in `ForcedTool` mode): an `Unsupported` warning means `NotEnforced`, any
other warning `Approximate`, none `Exact`. `UnsupportedPolicy::Error` is how
callers ask for "constrained decoding, possibly over an approximated schema,
or fail". List every configuration without strict mode in
`AdapterUnderTest::strict_not_honoured` (see [§6](#6-contracts-and-tests)),
which runs `allama_conformance::probe::assert_strict_not_honoured_warns`
for each with your `chat` and `chat_stream` probes. A model you answer
`ToolCalls` and `StructuredOutput` `Unsupported` for has no strict mode to
honour and needs no listing: the aggregate runs the contract for it anyway,
and an adapter whose every model answers so (a text-only one) leaves
`strict_not_honoured` out.

A dialect that requires every property must still let the model leave an
optional one out. Keep a property with a non-null `default` non-nullable
(schemars writes one for a `#[serde(default)]` field whose type is
`Serialize`), and make the others nullable, adding `null` to any `enum`.
Callers remove the `null`s the model then sends with
`allama::adapter::schema::normalize_strict_nulls`, guided by their original schema;
extraction and `ToolCall::parse_arguments_with` do it for them, whichever
adapter produced the output.

## 4. Errors

Map upstream failures to the variant that tells callers what to do; retry
and fallback act on the kind and on what the server said about retrying.
The table in `provider_impl` lists the usual mapping. Write the mapping as a
pure function of the status, headers and body, so the error contracts can
call it:

```rust
pub fn map_http_error(status: StatusCode, headers: &HeaderMap, body: &str) -> Error {
    let envelope = error_envelope(body);
    let (kind, code, param, text) = match &envelope {
        Some(envelope) => (
            envelope.error_type.as_deref(),
            envelope.code.as_deref(),
            envelope.param.as_deref(),
            envelope.message.as_deref().unwrap_or_default(),
        ),
        // Classify from an unrecognised body, but never quote it.
        None => (None, None, None, body),
    };
    let error = classify(Some(status), kind, code, param, text)
        .with_context(error_context(status, headers))
        .authored();
    let error = match envelope {
        Some(envelope) => envelope.apply(error),
        None => error,
    };
    error
        .with_provider(PROVIDER_NAME)
        .with_body_opt((!body.is_empty()).then_some(body))
}
```

Call it through `HttpResponse::map_error`, never by reading the body
yourself: `response.map_error(map_http_error).await` reads at most 64 KiB
of the body within 30 s (an error body that never ends would otherwise
hang a stream request and grow without bound), records a failed read in
the error's upstream message instead of passing an empty body, and turns a
redirect into an `UnexpectedResponse` naming its target host without
calling your function: a `3xx` is never followed. Read a success body with
`bytes_limited` or `text_limited`, under a `max_response_size` your
builder exposes (default `allama_http::DEFAULT_MAX_RESPONSE_SIZE`).

`classify` picks the variant from the vendor's error type and code, then
the status, and gives it a message the adapter writes itself, such as
`HTTP 429`, never the upstream text. For the status, fall back to
`allama_http::status_error(status, message)`, as every bundled provider
does, so a status means the same on every provider: `408`, `504` and `524`
are timeouts (a `504` may still be billed), `503` and `529` overload, and a
`499` (the client closed the request) is a `Provider` error that is neither
retried nor reported as a rejected request. Refine it from what the body
says, such as a `404` that names the model. It does not mark the message
`authored()`; `map_http_error` does, once, for every message `classify`
writes. `with_context` merges and never changes that mark, so the order of
the two does not matter.
Three details matter most:

- **Build the context with `allama_http::error_context(status, &headers)`.**
  It reads the status, the request id (`x-request-id` or `request-id`), the
  retry hint (`Retry-After` in seconds or as an HTTP date, and
  `retry-after-ms`), and the server retry directive (`x-should-retry: true`
  or `false`) into an `ErrorContext`. Every bundled HTTP provider uses it,
  so a header rule added later reaches your adapter too. Attach it with
  `with_context`, which merges: each field it read overwrites the error's,
  the rest is kept, and the message origin never changes:
  `Error::rate_limited(..).with_context(error_context(status, &headers)).authored()`,
  then the envelope and the body. To clear a field or replace the whole
  context, write through `Error::context_mut`. If your API names its
  request id or retry hint header differently, set it after attaching the
  context (`with_request_id`, `with_retry_after`), so a standard header
  does not overwrite it. Retry wrappers wait at least the hint, or give up
  and pass it on (a zero hint counts as none); the default retry policy
  follows the directive ahead of the kind and status.
- **Quota vs rate limit.** Exhausted credit is `Error::quota_exhausted`
  (not retryable), even when the endpoint returns 429. Check the vendor's
  error code.
- **Provider identity.** Attach `with_provider` so `Error::as_log()` gives
  operators a useful line.

Failures that got no response are `Error::Transport`;
`allama_http::TransportError::into_error` does that mapping. Such a request
may still have reached the provider: a connection reset after the request
was written looks the same as one that was never sent.

A success response whose body you cannot decode (a proxy's HTML page, a
truncated body) is `Error::UnexpectedResponse`, never
`Error::Serialization`, which is for the caller's own data. Build it with
`allama_http::malformed_response(status, body, err)`, which keeps the
decoder's text as upstream text and the body out of the log, and add the
provider and the request id so the response can be traced:

```rust
let wire: WireResponse = serde_json::from_str(body).map_err(|error| {
    malformed_response(status, body.as_bytes(), error)
        .with_provider(PROVIDER_NAME)
        .with_request_id_opt(request_id.as_deref())
})?;
```

`allama_conformance::contract::assert_malformed_response_contract` checks it
against a transport that answers `200` with `MALFORMED_BODY` (the adapter
aggregate's `.malformed_response(..)`).

A wrapper reports its own failures with the variant that says what went
wrong, never `Error::provider`: a `Provider` error without a status is
retryable and fails over, so a retry or fallback wrapper around yours
would repeat a call that cannot succeed.

### Message provenance

`Error::as_log()` is what operators log and what `FallbackAttempt` keeps,
so it must never carry a prompt. Provider text can: validation errors echo
input values, and gateways answer with HTML pages that quote the request.
The log therefore shows the constructor message only when you mark it
`authored()`, and keeps the provider's text in its own slot:

- **`authored()`** marks a message your code wrote, with no upstream text in
  it: `HTTP 400`, `prompt was blocked by Gemini: SAFETY`, a preflight
  refusal. Errors from `Degradations` are authored already.
- **`with_upstream_message(..)`** holds the provider's own message. `Display`
  shows it; `as_log()` shows it only under `as_log().with_message(n)`, which
  applications opt into at debug level.
- **`with_param(..)`**, **`with_provider_code(..)`** and
  **`with_provider_type(..)`** keep the rejected field when the provider
  names one (OpenAI's `param`, Gemini's field violations) and the
  provider's error code and type. The log shows each only as a bounded
  token (up to 128 characters out of `A-Z a-z 0-9 _ . : [ ] -`), and
  `<redacted>` otherwise, so prose or markup an endpoint puts there stays
  out. The filter cannot tell a code from an input value shaped like one,
  so put nothing else in these slots, and never copy them into the
  message, which `Debug` shows even when it is not authored.
- **`with_body(..)`** keeps the raw body, capped at 8 KiB. Only its length
  reaches the log.

Each of these setters takes the value. A field the reply may lack also has
an `_opt` twin that takes an `Option` and keeps the current value on
`None` (`with_request_id_opt(context.request_id.as_deref())`,
`with_provider_code_opt(envelope.code.as_deref())`), so parsed fields pass
straight through. Text twins take an `Option<&str>`, so a bare `None`
compiles. `MetadataBuilder` follows the same rule.

```rust
let upstream = "Invalid value: 'my SSN is 123-45-6789'";

// Bad: the upstream text reaches every log line and FallbackAttempt.
let bad = Error::invalid_request(format!("rejected: {upstream}")).authored();

// Good: authored message, upstream text in its own slot.
let good = Error::invalid_request(format!("HTTP {}", status.as_u16()))
    .with_context(error_context(status, &headers))
    .authored()
    .with_upstream_message(upstream.to_owned())
    .with_body(body.to_owned());
```

Read the common JSON envelopes (OpenAI, current and legacy vLLM, Ollama
native and `/v1`, Anthropic; the shapes are listed in the rustdoc of
[`error_envelope`](../../crates/allama-http/src/error.rs)) with
`allama_http::error_envelope(body)`, or `error_envelope_value(&value)` for
a stream line or event you parsed already, and attach what it found with
`ErrorEnvelope::apply`. It returns `None` for a
body that is no envelope: keep that body only in `with_body`, and use
`HTTP {status}` as the message. Classify from the upstream text if you need
to (a message that names the model), but never copy it into an authored
message. The text of a foreign error (an HTTP client's, a JSON decoder's)
is upstream text too: it can contain the request URL or quote the body.

Expose the mapping as a pure function of status, headers and body: the
error contracts call it directly, through the adapter aggregate of
[§6](#6-contracts-and-tests). They pass a plain status and header pairs,
so a mapping need not be HTTP's; wrap an HTTP mapping, which takes the
`http` types `map_error` passes, with
`allama_conformance::error_kinds::http_error_mapping`:

```rust
/// Acme's HTTP error mapping, as the contracts call it.
fn map(status: u16, headers: &[(&str, &str)], body: String) -> Error {
    http_error_mapping(map_http_error)(status, headers, body)
}
```

The error contracts also take your error envelope for each
`ErrorKindProbe` (`.error_envelope(..)`): the body your API sends for that
kind, with its own type, code and wording, or `None` for a kind you do not
map, including kinds added later. Its second argument is upstream text:
append it to the wording, where a classifier that reads the message would
find it. The kind contracts pass `""`; the redaction contract passes a
sentinel.

```rust
/// Acme's envelope for each kind, told apart by its type and code. The
/// upstream text goes after the message, where a classifier that reads
/// the wording would find it.
fn error_envelope(kind: ErrorKindProbe, upstream: &str) -> Option<String> {
    let (error_type, code, message) = match kind {
        ErrorKindProbe::RateLimited => ("rate_limit_error", None, "Slow down."),
        ErrorKindProbe::Overloaded => ("overloaded_error", None, "Acme is busy."),
        ErrorKindProbe::InvalidRequest => ("invalid_request_error", None, "Bad request."),
        ErrorKindProbe::Auth => ("authentication_error", None, "Invalid API key."),
        ErrorKindProbe::QuotaExhausted => (
            "rate_limit_error",
            Some("credit_exhausted"),
            "Out of credit.",
        ),
        ErrorKindProbe::ContextLength => ("context_length_exceeded", None, "Too long."),
        ErrorKindProbe::ModelNotFound => ("model_not_found", None, "No such model."),
        ErrorKindProbe::Unsupported => ("unsupported_parameter", None, "Not supported."),
        // A kind Acme does not map, such as one added later.
        _ => return None,
    };
    let message = format!("{message} {upstream}");
    let mut error = json!({"type": error_type, "message": message.trim_end()});
    if error_type == "unsupported_parameter" {
        error["param"] = json!("temperature");
    }
    if let Some(code) = code {
        error["code"] = json!(code);
    }
    Some(json!({ "error": error }).to_string())
}
```

With `.error_mapping(map)` and `.error_envelope(error_envelope)`, the
aggregate runs three contracts on them:

- `allama_conformance::assert_error_log_excludes_body` feeds a sentinel
  through unrecognised bodies, the common envelopes' messages, and
  messages that steer classification, and fails if the sentinel reaches
  `as_log()`. It also puts a sentinel with spaces and punctuation in the
  envelopes' error type, code and `param`, and fails if it reaches
  `as_log()` or `Debug`. Then it sends the sentinel through your envelope
  for every kind you map, at the kind's canonical status, so it reaches
  every branch of your classification, including those that read only
  your envelope. It fails if the sentinel reaches `as_log()`, if your
  envelope drops it, or if it changes the variant the same envelope maps
  to without it.
- `allama_conformance::assert_retry_signals_surfaced` checks that the
  server retry directive and the retry hint in the headers reach the
  error. When your API also carries the hint in the error body (Google's
  `RetryInfo`), give it an encoder for that body
  (`.retry_hint_body(..)`), and it checks that the hint is read from there
  too.
- `allama_conformance::assert_error_kinds_mapped` maps your envelope for
  each kind you map at the kind's statuses (`429`, `503`, `400`, `401`,
  `429` with your quota code, `400` for context length, `400` for an
  unsupported feature, `404` for an unknown model, and both `504` and
  `408` for a timeout), and fails unless the error has the kind, keeps
  the status, and, for a provider's refusal, has stage `Provider`. It
  requires at least `RateLimited` and `InvalidRequest`.

An error can also arrive inside a stream that has started: an `error` event
after the first deltas, with no HTTP status. Classify it from the vendor
code like the same error over HTTP, so retry and fallback act the same
whenever it arrives. `allama_conformance::assert_stream_errors_classified`
checks it: it wraps each envelope as a stream body (`.stream_body(..)`),
decodes it through your `chat_stream` over a transport that answers with
that body (`.stream_replay(..)`), and fails unless the in-band error has
the kind your HTTP mapping gives the same envelope:

```rust
.error_mapping(map)
.error_envelope(error_envelope)
.stream_body(move |envelope| format!("{first_delta}event: error\ndata: {envelope}\n\n"))
.stream_replay(|sse| async move {
    let transport = ReplayTransport::new([ReplayResponse::sse(sse)]);
    provider(&transport)
        .chat_stream(&ChatRequest::new(MODEL).user("Hi!"))
        .await
})
```

`assert_error_fixture_eq` compares the non-redacted `ErrorFixtureView`, so
fixtures still pin which message you author and which you pass upstream.

## 5. Sending and streaming with `allama-http`

HTTP providers build an `HttpRequest`, send it through an
`HttpTransport`, and decode the `HttpResponse`; they never call `reqwest`
directly. Expose `http_transport(..)` on your builder so tests (and users)
can swap the transport. Make `reqwest` an optional default feature, as the
bundled crates do, so an application that injects its own transport does
not compile it.

Building the request belongs to `encode` (a missing scoped credential is
an error a dry run must report), sending to the call:

```rust
HttpRequest::post(self.chat_uri.clone(), body)
    .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
    .header(AUTHORIZATION, self.authorization.clone())
    // Last: the caller's headers, and the credentials addressed to
    // your scope, which replace your configured key.
    .apply_options(&request.options, &self.credential_slot)
```

```rust
let response = if streaming {
    // No total deadline, but a setup and an idle bound.
    self.stream_timeouts.send(&*self.http, request).await
} else {
    self.http.send(request.timeout(self.request_timeout)).await
}
.map_err(TransportError::into_error)?;
```

- **Transport options.** Call `HttpRequest::apply_options` with the options
  of the request you serve (chat, embedding, decision, or extraction) and your
  credential slot, after setting every header of your own, including your
  configured key. It applies the caller's `allama_http::HttpRequestOptions`:
  a plain header replaces yours of the same name, and the credentials
  addressed to your scope replace your configured key. Never read the
  `allama-http` options namespace yourself, and never put header values on
  spans. `allama_conformance::assert_transport_options_applied` checks it
  with your slot on every call path: give the adapter aggregate the slot
  and a call per path (`.transport_options(slot, send, stream_send)`). A provider that sends no HTTP request
  ignores the option, like options for any other namespace, and says so in
  its docs.
- **Credential slot.** Declare an `allama_http::CredentialSlot`: your
  credential scope, where your key goes, and the configured auth headers it
  replaces. Build it once, in your builder:

  ```rust
  // Acme takes its key as a bearer token.
  let credential_slot =
      CredentialSlot::new(self.credential_scope, CredentialPlacement::Bearer);
  ```

  - **The scope** names the account a per-request credential belongs to.
    Default it to your provider name, publish that as
    `DEFAULT_CREDENTIAL_SCOPE`, and let users change it with a
    `credential_scope(..)` builder method, so two accounts of your provider
    can share a fallback chain. It is independent of your provider name and
    namespace: changing it never changes telemetry or provider data.
  - **The placement** is `Bearer` (`Authorization: Bearer <key>`),
    `Header(name)` (`x-api-key: <key>`, or your own header, such as
    `x-nimbus-key`), or `Custom`. With `Custom`, `apply_options` leaves your
    credential alone: read it with
    `HttpRequestOptions::credential_for(scope)` and
    `Credential::expose_secret()` after `apply_options`, which adds the
    plain headers a signature covers, and place it with
    `HttpRequest::sensitive_header` (redacted in `Debug`). Refuse a kind you
    cannot place with `Credential::unsupported(scope)`; never fall back to
    your configured key, which would bill the wrong account. There is no
    query-string placement: `http-tracing` records the URL on spans, so an
    API that takes its key in the query needs `Custom` and its own
    redaction.
  - **`replaces`** lists configured auth headers your key replaces besides
    its own, such as an `authorization` header on an endpoint whose keys go
    in `api-key`. A plain header your slot writes or replaces is refused
    with `Error::Configuration` before anything is sent, which protects your
    own header (`x-nimbus-key`). That refusal happens only in your adapter:
    in a fallback chain the providers tried before yours still get a plain
    header, so tell your users to send keys as `Credential::header`, and
    only the exact credential list (`is_credential_header`) protects every
    hop.
  - **`required`**: when your builder has no configured key because the
    user set a scope (bring your own key), mark the slot required, and
    `apply_options` refuses a request without a credential for the scope.
- **Request timeout.** Send every non-streaming request with a bound on
  the whole exchange (`HttpRequest::timeout`). Default it to
  `allama_http::DEFAULT_REQUEST_TIMEOUT` (120 s), whatever the transport,
  and offer a `timeout(Option<Duration>)` builder method where only
  `timeout(None)` removes it; a client of the user's own keeps the bound.
  No transport adds a total deadline of its own, so without it a server
  that accepts the connection and never answers holds the call forever,
  and the conformance aggregate does not catch that. Local servers that
  load a model on first use should default to none. Expiry is a retryable
  `Error::Timeout`.
- **Stream timeouts.** Default to `StreamTimeouts::default()` (120 s for
  headers, 300 s between chunks; generous on purpose, since reasoning models
  can think for minutes before the first token) and offer a
  `stream_timeouts(..)` builder method. Local servers that load a model on
  first use should default to `StreamTimeouts::unbounded()`. Expiry is a
  retryable `Error::Timeout`.
- **Stream event size.** `SseStream` holds at most
  `SseDecoder::DEFAULT_MAX_EVENT_SIZE` (64 MiB) of one line or one event's
  data, and fails with `SseError::EventTooLarge` past it. Offer a
  `max_stream_event_size(..)` builder method and pass it to
  `SseStream::max_event_size`. The error is a stream protocol error:
  retrying gets the same event, so it is not retryable.
  `SseError::InvalidUtf8` is a stream decode error for the same reason: the
  decoder fails on bytes that are not UTF-8 instead of replacing them, and
  asking again streams the same bytes. `SseError::into_error` maps both,
  with authored messages, and the driver below calls it for you.
- **Decoding.** Frame the body with `allama_http::sse::SseStream`, as the
  built-in crates do: it handles `CRLF` line endings, multi-line `data:`
  fields, comments and `retry:` lines, and yields each event's name and
  data. Write the decoder, and let `allama_http::sse::SseChatStream` drive
  it:

  ```rust
  let events =
      SseStream::new(response.into_body()).max_event_size(self.max_stream_event_size);
  let stream = SseChatStream::new(events, StreamDecoder::new(request_id));
  // The warnings and vendor frames follow the stream's `ResponseStart`.
  Ok(prepared.finish_stream(stream))
  ```

  The decoder implements `ChatFrameDecoder<SseEvent>`: `frame` for each
  SSE event, `end` when the body ends before your terminal event, `fail`
  when reading the body fails, and `is_done` once your terminal event (or
  an error) arrived:

  ```rust
  impl ChatFrameDecoder<SseEvent> for StreamDecoder {
      fn frame(&mut self, event: SseEvent) -> Vec<Result<StreamEvent>> {
          self.process(&event.event, &event.data)
      }

      /// The body ended before `done`.
      fn end(&mut self) -> Vec<Result<StreamEvent>> {
          vec![Err(self.incomplete())]
      }

      /// Reading the body failed: `error` is already classified (a stream
      /// timeout stays a timeout); say who served and which request it was.
      fn fail(&mut self, error: Error) -> Vec<Result<StreamEvent>> {
          self.finished = true;
          let error = error
              .with_provider(PROVIDER_NAME)
              .with_request_id_opt(self.request_id.as_deref());
          vec![Err(error)]
      }

      /// After `done` or an error, the stream ends without waiting for the
      /// body to close.
      fn is_done(&self) -> bool {
          self.finished
      }
  }
  ```

  The driver keeps the rules a hand-written loop gets wrong: the stream
  ends after its first error (a chat stream must; see `ChatStream`), `end`
  or `fail` runs at most once, the body is never polled after the stream
  ended, and a failed read is classified by its kind
  (`TransportError::into_stream_error`: a stream timeout is
  `Error::Timeout`, anything else a retryable transport stream error, with
  the transport's text as the upstream message). A body whose error is not
  a `TransportError` (a platform's own byte stream) goes through
  `SseChatStream::with_error_map`, or has its errors converted first, as
  the Workers AI adapter does. The driver is not tied to SSE: see
  [other wire framings](#other-wire-framings-ndjson-). Dispatch on the event name *and* data, and
  emit the normalized lifecycle (see [Event order](#event-order)). Build
  the `ResponseMetadata` event (usage, finish reason, request id) with
  `allama::chat::stream::MetadataBuilder`. `StreamCollector` turns the events
  into the same `ChatResponse` the unary call returns; test that they
  agree.

  A body that ends before the terminal event is an error
  (`StreamErrorKind::Incomplete`, from your `end`), never a quiet end. A
  body that reaches its terminal event without reporting a finish reason
  is complete: end it with `ResponseStop` and no finish reason, and never
  make one up.
  `Prepared::finish_stream` adds the warning that tells the caller (code
  `allama::adapter::chat::MISSING_FINISH_REASON`), and `finish` / `finish_stream`
  report a `Length`, `ContentFilter` or `Refusal` finish as an
  `allama::finish` event, so the adapter does nothing for either. A wire
  format with no terminal event of its own (Gemini's last chunk carries the
  finish reason) is `Incomplete` when the last chunk has no reason.
  `allama_conformance::assert_stream_ends_classified` checks both ends
  through your `chat_stream` (the adapter aggregate's `.stream_ends(..)`).
- **Vendor frames.** Endpoints and proxies add data your decoder does not
  model: named events the API does not define (`event: search_results`) or
  extra top-level chunk fields. Never decode them as your own events and
  never drop them: surface the data as
  `StreamEvent::response_provider_data(..)` under your namespace, with one
  warning per unknown event type:
  `Warning::new(WarningKind::Other, Subject::Response, ..).with_provider(PROVIDER_NAME)`.
  [`allama_http::framing::VendorFrames`](https://docs.rs/allama-http/latest/allama_http/framing/struct.VendorFrames.html)
  does both for named events
  (`surface(event_name, data)`). Each frame's event holds that frame's
  payload; call `finish()` just before your `ResponseStop`, and yield what
  it returns, so the collected response holds every payload of a type that
  repeated. A vendor frame never starts the response, but yield what
  `surface` returns as it comes, before your `ResponseStart` too:
  `Prepared::finish_stream` holds it until the start, so keep no buffer of
  your own:

  ```rust
  if !KNOWN_EVENTS.contains(&event) {
      // Before `ResponseStart` too: `finish_stream` holds them until it.
      let frames = self.vendor_frames.surface(event, data);
      out.extend(frames.into_iter().map(Ok));
      return out;
  }
  ```

  SSE comments and `retry:` lines are framing, not data. `event: error`
  frames are typed errors.
- **Provider-specific updates** that are not content (hosted-tool progress,
  for example) go out as `StreamEvent::provider_delta(..)`, which callers
  may observe but never need to collect.

### Other wire framings (NDJSON, …)

The driver does not depend on SSE.
`allama_http::framing::FramedChatStream` runs any
`ChatFrameDecoder<F>` (`frame`, `end`, `fail`, `is_done`) over a stream of
frames of type `F`, with the guarantees above; `SseChatStream` is its SSE
form (`FramedChatStream<SseStream<_>, _>`), driving a
`ChatFrameDecoder<SseEvent>`. For
newline-delimited JSON, frame the body with
`allama_http::ndjson::NdjsonStream`. It yields each line as a `String`,
handles `CRLF` endings (also split across chunks), skips blank keep-alive
lines, keeps a last line without its `LF`, and fails on a line past its
size cap (`max_line_size`: pass your builder's `max_stream_event_size`) or
one that is not UTF-8.

Acme also streams NDJSON, one line per event. Only the frame differs, so
its line decoder hands each line's event and data to the SSE decoder; an
API with lines of its own decodes them here:

```rust
fn frame(&mut self, line: String) -> Vec<Result<StreamEvent>> {
    let mut frame: Value = match serde_json::from_str(&line) {
        Ok(frame) => frame,
        Err(error) => {
            self.0.finished = true;
            let error = Error::stream_decode("invalid JSON in an Acme stream line")
                .authored()
                .with_upstream_message(error.to_string())
                .with_provider(PROVIDER_NAME);
            return vec![Err(error)];
        }
    };
    let event = frame["event"].as_str().unwrap_or_default().to_owned();
    let data = frame["data"].take().to_string();
    self.0.process(&event, &data)
}
```

```rust
let lines = NdjsonStream::new(response.into_body()).max_line_size(max_line_size);
FramedChatStream::new(lines, LineDecoder(StreamDecoder::new(request_id)))
```

`FramedChatStream::new` classifies a failure with
`NdjsonError::into_error`, as `SseChatStream::new` does with
`SseError::into_error`: a stream timeout stays a timeout, a line that is
not UTF-8 is a stream decode error, and a line past the cap a stream
protocol error. A line of a type you do not model is a vendor frame:
surface it with
[`VendorFrames::surface_payload(line_type, line)`](https://docs.rs/allama-http/latest/allama_http/framing/struct.VendorFrames.html#method.surface_payload).
When your lines carry no type, a line that is none of yours (no field you
decode) is the vendor frame: surface it under a name of your own
(`surface_payload("line", json)`), as `text_gateway` does. A framing
of your own (an event stream of binary messages, a WebSocket) needs only a
`Stream` of `Result<frame, error>`, whose error converts into an
`allama::Error` or goes through `FramedChatStream::with_error_map`.

Run contract C1 in your framing: give the adapter aggregate
`.stream_captures(framing, captures)` instead of `.sse_captures(..)`, with
`StreamFraming::ndjson` and a closure that writes an unknown line the way
your API names a line's type:

```rust
let ndjson =
    StreamFraming::ndjson(|event, data| json!({"event": event, "data": data}).to_string());
```

When your lines carry no type, invent none: write the injected frame's
payload as the line, so it is a line of fields your decoder does not know,
`StreamFraming::ndjson(|_, payload| payload.to_string())`.

Test that your stream decodes the same however the network splits it:
[`ReplayResponse::ndjson(body)`](https://docs.rs/allama-http/latest/allama_http/struct.ReplayResponse.html#method.ndjson)
delivers a stream one line per chunk,
`ReplayResponse::new(StatusCode::OK, body).split_every(1)` one byte at a time, and
`ReplayResponse::chunks(status, chunks)` in the chunks you choose.

### Event order

A stream is `ResponseStart`, then for each content block `BlockStart`, its
deltas and `BlockStop`, then `ResponseMetadata` (possibly several), then
`ResponseStop`. Keep this order: applications that read the raw events
rely on it (a UI that announces a tool call at its `BlockStart`,
`until_committed`). `StreamCollector` is lenient and does not check it: it
opens a block for a delta that arrives before its `BlockStart`, and takes
metadata before a `BlockStop`, so a stream that collects to the right
response may still break the order. The aggregate's `StreamEventOrder`
contract checks the raw events of your stream captures instead (see
[§6](#6-contracts-and-tests)).

- **Blocks may overlap.** A block may stay open while later blocks start
  and stream: the Gemini adapter keeps a text block open until its
  candidate ends, to attach grounding citations to it, so its `BlockStop`
  comes after later blocks' events. Every delta names its block's index;
  consumers route by it, never by the last `BlockStart`.

- **`ResponseStart` comes first.** It precedes every block event and every
  `ResponseMetadata` event.
- **`Warning` and `ResponseProviderData` may appear at any point**, before
  `ResponseStart` too. `Prepared::finish_stream` puts the request warnings
  right after `ResponseStart`, then the warnings and provider data your
  stream yielded before it, so callers see the response start first. When
  the first other event is not a start (an early error), they come right
  before it. A stream that does not go through `finish_stream`, such as a
  decoder's stream in a test or a benchmark, gets the same order from
  `allama::adapter::chat::start_first`.
- **No start frame on the wire?** Synthesize `ResponseStart` from the first
  chunk, as the Gemini adapter does. The Acme stream has none either; every
  chunk carries the response id and model:

  ```rust
  fn start(&mut self, json: &Value, out: &mut Vec<Result<StreamEvent>>) {
      if self.started {
          return;
      }
      self.started = true;
      let field = |key: &str| json.get(key).and_then(Value::as_str).map(str::to_owned);
      out.push(Ok(StreamEvent::response_start(
          field("response_id"),
          field("model"),
      )));
  }
  ```
- **A tool call without an id?** Make one up that no other call in the
  conversation can have: scope it by the response id, or by a random id
  when the response has none, as the Gemini and OpenAI-compatible decoders
  do (`call_<response id>_<n>`). An index alone (`call_0`) repeats in every
  turn, and Anthropic rejects a conversation that reuses an id;
  `history::validate` reports it as `DuplicateToolCallId`. Use only
  `[A-Za-z0-9_-]`, replacing other characters of the response id with
  `_`, so every provider accepts the id after a failover (Anthropic
  refuses any other character), and keep it short (OpenAI takes at most
  64). allama has no shared helper for this on purpose:
  the format is wire knowledge (an adapter may need to recognize its own
  made-up ids when they come back, to send the API none), so each decoder
  owns it, and the `ToolCallIdsRoundTrip` contract checks the result,
  alphabet included.
- **An in-band error may arrive before `ResponseStart`**: Anthropic sends
  `event: error` on an HTTP 200 stream, and the Responses API an `error`
  event. End the stream with that error, without a synthesized start, and
  classify it as you classify HTTP errors, so retry and fallback behave the
  same whenever it arrives.
- **Wrappers keep the order too** when they add events. For example
  `FallbackProvider` reports its failed attempts in a `ResponseMetadata`
  event right after the fallback's `ResponseStart`.

## 6. Contracts and tests

`allama-conformance` holds the shared bar. Add it as a dev-dependency
(`allama-conformance = "0.1"` from crates.io once published; until then a
path or git dependency on this repository). Its `server` feature, on by
default, provides [`TestHttpServer`](https://docs.rs/allama-conformance/latest/allama_conformance/struct.TestHttpServer.html);
a wrapper or a provider that does not
speak HTTP can turn it off with `default-features = false`.
`TestHttpServer` is a real local server, so the provider under test needs a
real HTTP client, `allama-http`'s `reqwest` transport, which `server` does
not bring in. The bundled crates get it from their own optional `reqwest`
feature and gate the server tests on it (`#![cfg(feature = "reqwest")]`), so
`--no-default-features` still builds. A crate without that feature, or one
that wants the server tests in every build, adds `allama-http` with its
`reqwest` feature as a dev-dependency (`allama-http = { version = "0.1",
features = ["reqwest"] }`) and hands the provider an
`allama_http::ReqwestTransport`. A provider crate typically has:

1. **Fixture tests** (`conformance_tests.rs`): request encoding, response
   decoding, stream decoding, and error mapping against recorded wire data
   in `fixtures/`, with `FixtureDir`, `assert_json_fixture_eq`,
   `assert_response_fixture_eq`, `assert_stream_fixture_eq`, and
   `assert_error_fixture_eq`. Regenerate expectations with
   `ALLAMA_UPDATE_FIXTURES=1`.

   Record real exchanges with
   [`scripts/capture-fixture.sh`](../../scripts/capture-fixture.sh). It
   needs only bash and curl, so copy it into your repository. Its `raw`
   mode works with any endpoint:

   ```sh
   scripts/capture-fixture.sh raw https://api.example.com/v1/chat \
     request.json fixtures/stream.sse \
     --header 'Authorization: Bearer $EXAMPLE_API_KEY'
   ```

   It POSTs the request body as-is, so put the streaming flag in the body
   for a streaming capture. `$NAME` in a header names an environment
   variable (or one in `.env` at the repository root). Keep the single
   quotes, so the secret stays out of shell history. The script writes only
   the response body, whatever the status, never the headers. It refuses to
   write a capture that contains a value it took from the environment.
   Review the file before committing it. The `openai`, `openai-responses`,
   `anthropic`, `gemini`, and `gemini-stream` modes are shortcuts for the
   bundled providers.
2. **Transport tests** (`tests/http_integration.rs`): the real provider
   against `TestHttpServer` (`MockHttpResponse::json` for a body as
   written, `::json_value` for a serialized value, `::sse`, `.header(..)`):
   URL, auth headers, unary and streaming decoding, and HTTP error mapping
   with real headers (a `Retry-After` on a 429, for example).
   [`ReplayTransport`](https://docs.rs/allama-http/latest/allama_http/struct.ReplayTransport.html)
   runs the adapter with no server at all, with the same constructors on
   [`ReplayResponse`](https://docs.rs/allama-http/latest/allama_http/struct.ReplayResponse.html)
   (and `ReplayResponse::ndjson` for an NDJSON stream);
   `ReplayResponse::then_stall()` tests
   the stream idle bound, and `.split_every(n)` delivers any body in
   `n`-byte chunks, so a test can split every line ending and UTF-8
   sequence.
3. **Behavioral contracts** from `allama_conformance::contract`: success,
   streaming, tool calls, structured output, the error kinds, and the
   unsupported-feature policy on both paths
   (`assert_structured_output_contract`, `assert_unsupported_hint_warns`,
   `assert_strict_policy_rejects`, and their `_stream` variants). The error
   kind contracts (`assert_auth_error_contract` and its siblings) call
   `chat` and then a stream probe, so a failure is classified the same on
   both; `collect_chat_stream` is the usual probe, and the transport then
   answers each failure twice:

   ```rust
   // Each failure answers `chat`, then `chat_stream`.
   let failing = |status: StatusCode, body: &'static str| {
       let answer = || ReplayResponse::json(status, body);
       provider(&ReplayTransport::new([answer(), answer()]))
   };
   let auth = failing(
       StatusCode::UNAUTHORIZED,
       r#"{"error":{"type":"authentication_error","message":"bad key"}}"#,
   );
   assert_auth_error_contract(&auth, &request, |r| collect_chat_stream(&auth, r)).await;
   ```
   The usage contract is at the crate root:
   `allama_conformance::assert_usage_contract` checks that the counts are
   consistent (cache reads and writes within the input, reasoning within
   the output, a total of at least input plus output). It cannot tell
   whether you normalized them: an input that should have had cached
   tokens added still passes while the cached count is the smaller one.
   So also record a cache-hit response (with reasoning, when your API
   reports it) and assert its normalized numbers by hand, citing the
   vendor's usage documentation for what its counts include:

   ```rust
   // A cache hit, normalized by hand from Acme's usage docs: its
   // `input_tokens` already counts the cached ones, and its
   // `output_tokens` the reasoning ones. The contract only checks that
   // the numbers are consistent.
   let usage = unary.usage.as_ref().unwrap();
   assert_eq!(usage.input_tokens, Some(12));
   assert_eq!(usage.cache_read_input_tokens, Some(4));
   assert_eq!(usage.output_tokens, Some(7));
   assert_eq!(usage.reasoning_tokens, Some(3));
   assert_eq!(usage.total_tokens, Some(19));
   assert_usage_contract(usage);
   ```
4. **The adapter aggregate**: `allama_conformance::assert_chat_adapter`
   runs every contract against your adapter in one test. The set grows
   only in minor releases of `allama-conformance`, never in a patch
   release. A new contract can join it as provisional (always, when it
   needs an input your suite does not give): it is reported as not run,
   or its failure is printed, until a later minor release makes it
   required. No contract is provisional in 0.1,
   and one contract stays outside the aggregate for now
   ([vendor fields below the top level](#vendor-fields-below-the-top-level-provisional)).
   Most of its contracts run the provider through
   *probes*: async closures that send a request through a fresh provider
   over a canned transport and return a `ProbeOutcome` with the result and
   the bodies sent. It takes two, a chat probe and a streaming probe, and
   runs each check on both paths, because `chat` and `chat_stream` can each
   drop a field or a warning on their own:

   ```rust
   async fn probe(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
       let transport = ReplayTransport::new([ReplayResponse::json(StatusCode::OK, OK_BODY)]);
       let result = provider(&transport).chat(&request).await;
       ProbeOutcome::new(result, bodies(&transport))
   }

   async fn stream_probe(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
       let transport = ReplayTransport::new([ReplayResponse::sse(OK_STREAM)]);
       // Collecting keeps every StreamEvent::Warning; an error from
       // chat_stream or any stream item becomes the outcome's error.
       let result = match provider(&transport).chat_stream(&request).await {
           Ok(stream) => stream.collect_response().await,
           Err(err) => Err(err),
       };
       ProbeOutcome::new(result, bodies(&transport))
   }
   ```

   The aggregate takes them with the other inputs; `map` and
   `error_envelope` are the error mapping and envelope of [§4](#4-errors),
   and `send` makes one call with transport options attached:

   ```rust
   #[tokio::test]
   async fn the_adapter_passes_every_contract() {
       // The capability answers shape the probe requests.
       let acme = provider(&ReplayTransport::new([]));
       // Validation is a dry run of the encoding, so it agrees both ways.
       let validation = acme
           .validate_chat_request(&ChatRequest::new(MODEL).user("Hi"))
           .unwrap();
       assert_eq!(validation.coverage(), ValidationCoverage::Encoding);

       // An in-band error arrives after the stream started.
       let first_delta = OK_STREAM.split_inclusive("\n\n").next().unwrap();
       // A stream cut short, and one that ends without a finish reason.
       let cut_short = OK_STREAM.split("event: done").next().unwrap();
       let without_finish = format!("{cut_short}event: done\ndata: {{}}\n\n");
       let slot = CredentialSlot::new(DEFAULT_CREDENTIAL_SCOPE, CredentialPlacement::Bearer);
       let under_test = AdapterUnderTest::new([MODEL], NAMESPACE, probe, stream_probe)
           // Acme has no strict mode.
           .strict_not_honoured([MODEL])
           .error_mapping(map)
           .error_envelope(error_envelope)
           .stream_body(move |envelope| format!("{first_delta}event: error\ndata: {envelope}\n\n"))
           .stream_replay(|sse| async move {
               let transport = ReplayTransport::new([ReplayResponse::sse(sse)]);
               provider(&transport)
                   .chat_stream(&ChatRequest::new(MODEL).user("Hi!"))
                   .await
           })
           .stream_ends(StreamEnds::new(cut_short).terminated_without_finish(without_finish))
           .sse_captures([OK_STREAM])
           .transport_options(
               slot,
               |options| send(options, false),
               |options| send(options, true),
           )
           .malformed_response(
               |body| async move {
                   let answer = ReplayResponse::new(StatusCode::OK, body)
                       .header("x-request-id", HeaderValue::from_static("req_1"));
                   provider(&ReplayTransport::new([answer]))
                       .chat(&ChatRequest::new(MODEL).user("Hi!"))
                       .await
               },
               Some("req_1"),
           )
           // The decoding contracts decode captured responses on both paths.
           .unary_replay(|body| async move {
               let transport = ReplayTransport::new([ReplayResponse::json(StatusCode::OK, body)]);
               provider(&transport)
                   .chat(&ChatRequest::new(MODEL).user("Hi!"))
                   .await
           })
           .response_captures([ResponseCapture::new(OK_BODY, OK_STREAM)])
           .tool_call_capture(
               ResponseCapture::new(TOOL_BODY, TOOL_STREAM).wire_tool_call_ids(["call_1"]),
           )
           .length_capture(ResponseCapture::new(LENGTH_BODY, LENGTH_STREAM))
           // Over the real HTTP client: a redirect is never followed.
           .redirect_not_followed(|base_url, streaming| async move {
               let acme = AcmeProvider::builder()
                   .api_key("test-key")
                   .base_url(base_url)
                   .build()?;
               let request = ChatRequest::new(MODEL).user("Hi!");
               match streaming {
                   true => acme.chat_stream(&request).await.map(drop),
                   false => acme.chat(&request).await.map(drop),
               }
           })
           // Acme's model replays reasoning, so no model drops it.
           .skip([Contract::UnreplayableReasoningDropped]);
       assert_chat_adapter(&acme, &under_test).await;
   }
   ```

   `TOOL_BODY` and `TOOL_STREAM` are one response with a tool call, as
   `chat` and as `chat_stream` receive it; `LENGTH_BODY` and
   `LENGTH_STREAM` one cut off at `max_tokens`. Record them from the API
   when you can.

   A probe whose transport is not JSON (protobuf, NDJSON) reports raw
   bodies with `ProbeOutcome::raw`; every probe contract searches them as
   bytes.

   `AdapterUnderTest` takes the models to probe (one of every kind your
   adapter tells apart), your namespace and the two probes; setters give
   the inputs some contracts need (the error mapping and envelope of
   [§4](#4-errors), a unary and a stream replay, captured responses, the
   stream ends and stream captures, the transport options of
   [§5](#5-sending-and-streaming-with-allama-http), a malformed response,
   a redirect call, requests of your own for validation). When a contract's input is
   missing, the aggregate fails before running anything and names it; a
   contract that does not apply to your adapter (no wire stream, no HTTP)
   goes in `.skip([Contract::..])`, so nothing is skipped silently. `.known_gaps(..)` lists acknowledged defects, and fails once
   one is fixed. Every contract also stays a public function, for a test
   that pins one behavior. `assert_embedding_adapter`,
   `assert_decision_adapter` and `assert_extraction_adapter` are the same
   for embeddings, decisions and extractions.

   The contracts it runs:
   - **Decoding**: `allama_conformance::decoding` decodes each captured
     response (`ResponseCapture`: its `chat` body and its `chat_stream`
     body) through both paths and checks that they agree on content,
     finish reason, usage, provider and provider data; that both paths
     stamp `ChatResponse::provider`; that tool-call ids are the wire's
     (or, where the API sends none, synthesized uniquely from
     `[A-Za-z0-9_-]`), finish with
     `ToolCalls` and reach the next request; and that a response cut off at
     the cap finishes with `Length`.
   - **Raw event order**: `allama_conformance::stream_order` checks the raw
     events of every stream capture against the documented order
     (`ResponseStart` first, a block's events between its `BlockStart` and
     `BlockStop`, every block stopped before the finish reason,
     `ResponseStop` last). The collector is lenient about the order;
     applications that read raw events are not.
   - **Validation coverage**: `validate_chat_request` must report
     `ValidationCoverage::Encoding`, a dry run of the encoding (see
     [§2](#2-identity-and-capabilities)). Skip it only if your adapter cannot encode
     without sending.
   - **Redirects**: `allama_conformance::redirect::assert_redirect_not_followed`
     points your provider, with its real HTTP client, at a server that
     redirects to another host, and checks that the other host sees
     nothing and the call fails with a non-retryable `UnexpectedResponse`.
   - **Vendor frames (C1)**:
     `allama_conformance::stream_frames::assert_unknown_stream_frames_surfaced`
     injects a frame of an unknown type, and frames that carry no data, into
     one of your stream captures, in its own framing, and checks that the
     stream is otherwise unchanged and the frame's data reaches the
     response. It also reframes the capture (`CRLF` line endings, SSE data
     split over several `data:` lines) and checks that the events stay
     exactly the same. `.sse_captures(..)` runs it over SSE
     (`StreamFraming::sse()`), and `.stream_captures(framing, ..)` in
     another framing (see
     [other wire framings](#other-wire-framings-ndjson-)).
   - **Request fields (C2)**:
     `allama_conformance::probe::assert_request_fields_accounted` checks
     that every portable request field is sent, warned about, or rejected,
     and that state from another provider and application data are never
     sent. It takes your provider as well as the probes, because some
     checks shape their requests to its capability answers. See
     [Fields you must account for](#fields-you-must-account-for).
   - **Validation agreement**: `allama_conformance::assert_validation_agrees`
     checks that `validate_chat_request` agrees with `chat` and
     `validate_chat_stream_request` with `chat_stream`: the same refusal
     (error kind and refused field, nothing sent) or the same warnings,
     and `relied_on_unknown` matching your `Unknown` answers. The aggregate
     runs `drift_requests(model)` (the cases where the two drift apart in
     practice) for each model and for a model you do not know; add your
     own with `.validation_requests(..)`: provider options that change
     encoding. Run it on its own for a provider built with a credential
     scope and no key (both must refuse with `Error::Configuration`). It
     is two-way only when your validation reports `Encoding` coverage, so
     also assert that it does. `assert_embedding_validation_agrees` with
     `embedding_drift_requests(model)` and
     `assert_decision_validation_agrees` with
     `decision_drift_requests(model)` and
     `assert_extraction_validation_agrees` with
     `extraction_drift_requests(model)` are the same contract for `embed`,
     `decide` and `extract_values`.
   - **Negative capability audit (C3)**:
     `allama_conformance::capabilities::assert_unsupported_chat_capabilities_refused`
     (and the embedding variant) checks that a request using a capability
     you report `Unsupported` is refused: a warning whose subject names the
     capability's field or content, or `Error::Unsupported` with stage
     `BeforeSending` and nothing sent. A warning about something else, or
     a refusal at another stage, fails. The chat audit takes both probes.
   - **Stream usage**:
     `allama_conformance::capabilities::assert_stream_usage_matches_capability`
     checks your `ChatCapability::StreamUsage` answer on the output: the
     response your stream probe collects has usage when you answer
     `Supported`, and none when you answer `Unsupported`. Answer
     `Supported` only when your API reports usage in every stream (a budget
     guard relies on it), and give the stream probe a captured stream.
   - **Capability honesty**:
     `allama_conformance::capabilities::assert_capabilities_honest(&provider,
     &HonestyUnderTest::new(models, probe, stream_probe))` probes every
     capability, request field, content part, content block and tool choice
     whatever you answer, for each listed model and for a model you cannot
     know (`UNKNOWN_MODEL`): content must reach the body, be warned about,
     or be refused before sending; a setting must change the body (against
     a contrasting value), be warned about, or be refused. It is the
     evidence for your `Supported` chat claims. For embeddings,
     `allama_conformance::capabilities::assert_embedding_capabilities_backed`
     checks that every `Supported` claim names a fixture or test that
     exists.
   - **Batch limits (C6)**:
     `allama_conformance::probe::assert_batch_limit_enforced` checks that
     an embedding batch over `EmbeddingLimits::max_inputs_per_call` is never sent as one
     request.
   - **Embedding vector count**:
     `allama_conformance::probe::assert_embedding_count` checks that a
     response with too few or too many vectors, or (with
     `VectorOrder::Indexed`) a duplicate, missing, or out-of-range index, is
     rejected with `Error::UnexpectedResponse` instead of being misaligned
     with the inputs. Its probe answers with the vectors the contract asks
     for.
   - **Embedding response checks**:
     `allama_conformance::probe::assert_embedding_response_checked` checks
     that vectors of the wrong size, of different sizes, empty, or holding
     a non-finite value are rejected with `Error::UnexpectedResponse`
     naming your provider. Its probe answers with the vectors the contract
     asks for, as JSON numbers.
   - **Embedding inputs**:
     `allama_conformance::probe::assert_multi_part_input_refused` (for
     endpoints that embed one text per input) and
     `allama_conformance::probe::assert_input_provider_data_checked` check
     that inputs are never joined and that misspelled input settings fail.
   - **Opaque state on `Other` content**:
     `allama_conformance::probe::assert_other_provider_data_gated` checks
     that provider data in another namespace on an `Other` block or part
     never reaches the wire, and that content you do not send is reported.
   - **Unreplayable reasoning**:
     `allama_conformance::probe::assert_unreplayable_reasoning_dropped`
     checks, for a model whose `ReasoningReplay` you report `Unsupported`,
     that reasoning never reaches the wire and that the `Dropped` warnings
     name the caller's blocks, including a block after the removed
     reasoning. The aggregate runs it for each listed model you answer so
     for, and needs one.
   - **Empty turns**:
     `allama_conformance::probe::assert_empty_turns_skipped` checks that
     an empty user or assistant message in the middle of a conversation
     sends exactly the bodies of the conversation without it, and that a
     reply left last by empty turns is dropped with a `Dropped` warning.
   - **Application data**:
     `allama_conformance::probe::assert_app_data_never_sent` checks that
     `app_data` on system prompts and messages never reaches the wire. Its
     conversation has every message role. It includes a tool call and its
     result unless your provider reports `ChatCapability::ToolCalls`
     `Unsupported` for the model, so an adapter without tool support can
     pass it.
   - **Transport options**:
     `allama_conformance::assert_transport_options_applied` attaches
     caller headers and credentials to calls and checks what is sent. Give
     `.transport_options(slot, send, stream_send)` the same
     `CredentialSlot` your builder declares (your default scope, and the
     placement and `replaces` of
     [§5](#5-sending-and-streaming-with-allama-http)), so the contract
     looks for a credential addressed to that scope where your key goes,
     and checks that one addressed to another scope is never sent. `send`
     and `stream_send` make one call through `chat` and `chat_stream` with
     the options attached, over a transport that answers both, and return
     the result with the requests sent:

     ```rust
     /// One call with transport options attached, on `chat` or
     /// `chat_stream`.
     async fn send(options: RequestOptions, streaming: bool) -> TransportOutcome {
         let transport = ReplayTransport::new([
             ReplayResponse::json(StatusCode::OK, OK_BODY),
             ReplayResponse::sse(OK_STREAM),
         ]);
         let mut request = ChatRequest::new(MODEL).user("Hi!");
         request.options = options;
         let provider = provider(&transport);
         let result = if streaming {
             provider.chat_stream(&request).await.map(drop)
         } else {
             provider.chat(&request).await.map(drop)
         };
         TransportOutcome::new(result, transport.requests())
     }
     ```

     Build the provider in `send` with a configured key, as a user would:
     the contract checks that the caller's credential replaces it and the
     headers in `replaces`, and that a plain header carrying a credential
     is refused before sending. For a provider built without a key (bring
     your own key), pass the slot it builds, `.required(true)`, and the
     contract also checks that a call with no credential for the scope is
     refused. With a `Custom` placement, where the credential goes is
     yours to test.

   The embedding contracts (batch limits, vector count, response checks,
   inputs) run through `assert_embedding_adapter`, whose probe calls
   `embed`, and whose `.answering(order, answer)` input sends a request
   over a transport that answers with the vectors a contract asks for.

   The tests of `native_adapter` run the aggregate against one adapter
   (all but the reasoning contract: its model replays reasoning); copy
   them along with the probes.
5. **Live tests**, `#[ignore]`d and run on a schedule, with the shared
   scenarios in `allama_conformance::e2e`.

If you ship a wrapper (middleware around another provider), two more
contracts apply. The runnable
[`middleware`](../../crates/allama-examples/examples/middleware.rs) example is a
template: an async request hook, response and stream observation, full
forwarding, and the C4 test. A wrapper sees the portable request, never
the HTTP request, so it sends a header (a request id, a tenant tag) or a
per-tenant key by adding it to the request's
[`HttpRequestOptions`](https://docs.rs/allama-http/latest/allama_http/struct.HttpRequestOptions.html),
keeping the options the caller set ("From a wrapper" in its rustdoc);
`MapRequestProvider` does that without a wrapper of your own. Latency,
outcomes and attempts come from `TracingProvider`'s span, not from a
timing wrapper.

- **Transparent wrapper (C4)**:
  [`assert_chat_wrapper`](https://docs.rs/allama-conformance/latest/allama_conformance/fn.assert_chat_wrapper.html)
  (or [`assert_wrapper`](https://docs.rs/allama-conformance/latest/allama_conformance/fn.assert_wrapper.html)
  for a wrapper over chat, embeddings and decisions, the embedding and
  decision variants, and `assert_extraction_wrapper` for the unstable
  extraction trait, which `assert_wrapper` leaves out) builds your wrapper
  around a
  [`TransparencyProbe`](https://docs.rs/allama-conformance/latest/allama_conformance/struct.TransparencyProbe.html)
  and checks two things. It forwards the provider
  name, typed lookup, capability answers, limits, and validation, both
  chat validators included, with their coverage. And its calls pass the
  data path through unchanged
  (`assert_transparent_chat_data_path`): the inner provider's warnings in
  order (you may add your own), provider data, usage and ids, every stream
  event (warnings, refusal deltas, metadata) in order, and every error with
  its kind, status, retry hint, refused subject and stage. A wrapper that
  rewrites model ids uses `assert_transparent_model_mapping` and
  `assert_transparent_chat_data_path` instead. A retry wrapper gets a
  policy without backoff in the test, since the probe's retryable errors
  come back on every attempt.

  Forward the questions with the macros in `allama::wrappers`, naming the
  field that holds the wrapped provider. A question added to a provider
  trait later is added to its macro in the same release, so your wrapper
  forwards it by being recompiled instead of silently answering with a
  default that is wrong for a wrapper (ADR 0013):

  ```text
  impl<P: ChatProvider> ChatProvider for MyWrapper<P> {
      type Stream = P::Stream;
      // `chat` and `chat_stream`: your behavior around the inner call.
      allama::wrappers::forward_chat_questions!(inner);
  }
  impl<P: ProviderIdentity> ProviderIdentity for MyWrapper<P> {
      allama::wrappers::forward_provider_identity!(inner);
  }
  // Likewise `forward_embedding_questions!`, `forward_decision_questions!`
  // and `forward_extraction_questions!`.
  ```

  A wrapper that answers a question itself (it maps the model or
  overrides a capability) writes that trait's questions by hand, and
  checks for new trait methods on every upgrade.

  A wrapper that adds warnings of its own builds on `allama::adapter::chat::prepare`
  and passes the inner response through its own `finish`: warnings the
  inner provider emitted already are not emitted again, so `finish` is safe
  at every layer, and you can call it freely. Add a warning event for
  anything else with `allama::warning::emit`. Give a warning that recurs
  with the request's shape a code (`Warning::with_code`), so its event goes
  out at `WARN` once per provider, model and code and at `DEBUG` after that;
  leave a warning about this call alone uncoded. A response cache or replay
  layer goes outside such a wrapper: a cached response's warnings are
  deserialized without their emitted mark, so a `finish` around the cache
  logs them again on every hit.

  Where your wrapper's annotations go:

  - **What it changed** is a coded warning of the matching kind. A
    redaction wrapper that rewrites the prompt adds an `Approximated`
    warning with code `approximated.redacted_content`, and `Dropped` for
    content it removed.
  - **Pure observations** (timings, cache hits, what a lookup found) go to
    `tracing` or your application's own channel, not the response.
  - **Never provider data**: not under `allama`, not under a provider's
    namespace, and not on blocks. `allama` is reserved for conventions
    allama publishes (ADR 0002), and a provider's namespace holds state
    its adapter reads back on replay. `FallbackAttempts`, which
    `FallbackProvider` records under `allama`, is the one reserved
    exception.
  - An annotation that is not a degradation has no portable place on the
    response yet; that is an open follow-up.
  - A wrapper that rewrites or annotates responses beyond adding warnings
    is not transparent (ADR 0010 §6): run the identity half
    (`assert_transparent_chat_wrapper`) and test what it changes yourself,
    instead of the data-path contract.
- **Retry hint (C5)**: a retry wrapper passes
  [`assert_retry_respects_hint`](https://docs.rs/allama-conformance/latest/allama_conformance/fn.assert_retry_respects_hint.html)
  (with `RetryHintProbe`): it never calls again before the hint, and it
  returns the rate-limit error with the hint when it gives up.

A wrapper passes the inner provider's errors on as they are. Do not
re-wrap them: `Error::provider(format!("summarize: {err}"))` loses the
kind, the status, the retry hint and the request id, and a `Provider` error
without a status is retried and fails over, so the layers around yours
would retry an `Auth` or `InvalidRequest` failure. To say what your wrapper
was doing, prefix the message instead:

```rust
use allama::Error;

let err = Error::rate_limited("HTTP 429").authored().with_status(429);
let err = err.with_message_prefix("summarize step");
assert!(matches!(err, Error::RateLimited { .. }));
assert_eq!(err.status(), Some(429));
assert_eq!(err.message(), "summarize step: HTTP 429");
```

`with_message_prefix` keeps the kind, the context and the message's origin.
On a failed failover it prefixes the returned error, not the errors it
fell back from (`Error::fallback_from`). The prefix is logged
wherever the message is, so it must be text your code wrote, never a prompt
or an input value. Errors of your wrapper's own are new errors, built with
the variant that says what went wrong (see [§4](#4-errors)).

A wrapper that tries several providers (failover, hedging) and fails on
all of them returns the last error it got as itself, with each earlier one
attached by `Error::with_fallback_from`, as `FallbackProvider` does.
Callers then match on the error they got, with no case for the wrapper,
walk the earlier ones with `Error::fallback_from`, and `as_log()` logs
them. Do not wrap the errors in one of your own: that changes the kind
that retries, fallback and the caller's `match` decide on.

A resilience wrapper of your own (a custom retry, hedging, a circuit
breaker) has two things `RetryingProvider` and `FallbackProvider` have
that it does not, in this release:

- **The resilience summary.** The built-in wrappers report their attempts,
  backoff, give-ups and failovers through `allama::wrappers::call_scope`,
  and `TracingProvider` records them once per call on its span
  (`allama.attempts`, `allama.retry.*`, `allama.fallback.*`,
  `allama.response.provider`). The only fact a crate outside allama
  reports is the server a request went to (`CallFact::server`, recorded by
  HTTP transports). Your wrapper's activity shows up as the attempts made
  under it. Do not record the summary's fields on the span yourself: the
  summary is recorded once per call, and a field recorded twice is
  exported twice.
- **Attempt scopes.** `RetryingProvider` runs each attempt in an attempt
  scope, so the warnings of an attempt it throws away are not logged. The
  scope is internal, so every attempt your wrapper makes logs its own
  warnings. A `RetryingProvider` inside your wrapper still holds back its
  own retried attempts.

Both can be made public later without a breaking change; open an issue if
your wrapper needs them. Until then, prefer composing the built-in
wrappers, and keep a custom one to what they cannot do.

A custom fallback wrapper that streams runs each attempt up to its commit
point (`until_committed`) before deciding to move on. To add events to the
committed prefix, as `FallbackProvider` adds its failed attempts, take the
returned `Prefixed` apart with `into_parts` and rebuild it with
`Prefixed::new`.

A custom retry wrapper that follows a `RetryPolicy` owns a retry budget the
way `RetryingProvider` does: take it from `RetryPolicy::make_budget()` once,
when the wrapper is built, and keep it (a new default budget, the policy's
shared one, or none). Call `RetryBudget::try_acquire` before each retry the
policy allows (`RetryPolicy::retry_delay`), keep the `RetryPermit` it
returns, and hand the last one to `RetryBudget::record_success` when the
call succeeds. Calling `make_budget` per request would give every call a
fresh, full budget and bound nothing.

A stream succeeds once it reaches its first output, so a streaming retry
wrapper keeps the permit in its stream state until then, as
`RetryingProvider` does: a field such as `permit: Option<RetryPermit>`
(`allama::wrappers::RetryPermit`, opaque), handed to `record_success` when
the first output arrives. The `RetryPermit` rustdoc has a runnable example.

The [crate README](../../crates/allama-conformance/README.md) lists every
contract with its entry point.

### Fields you must account for

Every key of a serialized `ChatRequest` is either probed by C2 or exempt
for a stated reason. The list is `allama_conformance::probe::ACCOUNTED_FIELDS`
(parsed by `accounted_fields()`), and a test in `allama` derives it from
`ChatRequest` itself, so a new portable field cannot reach your adapter
unnoticed: it arrives with a probe. Paths follow the serialized
form: arrays add `[]`, a message role or content type is a segment
(`messages[].user.cache{}` is `UserMessage::cache`), and a path ending in `[]`
or `{}` covers the whole array or object.

A probe with an acknowledged defect goes in `known_gaps`
(`&[Probe::MessageName]`); the audit fails once it passes, so the list
shrinks as you fix gaps. An `allama` newer than your `allama-conformance`
may list a field with a probe it does not know: C2 then runs the probes it
knows and fails, asking you to upgrade `allama-conformance` to the version
of `allama`. The fields C2 exempts, each with its reason, are
listed in the rustdoc of `ACCOUNTED_FIELDS`. This checklist of the probed
fields is generated from the list:

<!-- accounted-fields:start -->

A probed field is accounted for when it is:

1. sent (or approximated with a warning), refused with `Unsupported`, or dropped with a warning that names it, in which case `UnsupportedPolicy::Error` refuses it before sending
2. sent, dropped with a `Dropped` warning that names it, or handled as a hint
3. never sent; dropped with a `Dropped` warning that names it
4. never sent
5. never sent, and changes nothing that is sent or reported

| Probe | Request fields | Accounted for when it is |
| --- | --- | --- |
| `temperature` | `temperature` | 1 |
| `top_p` | `top_p` | 1 |
| `top_k` | `top_k` | 1 |
| `max_output_tokens` | `max_output_tokens` | 1 |
| `stop` | `stop[]` | 1 |
| `seed` | `seed` | 1 |
| `frequency_penalty` | `frequency_penalty` | 1 |
| `presence_penalty` | `presence_penalty` | 1 |
| `system_prompt` | `system[].content` | 1 |
| `system_cache` | `system[].cache{}` | 1 |
| `message_cache` | `messages[].user.cache{}`<br>`messages[].assistant.cache{}`<br>`messages[].tool.cache{}` | 1 |
| `tool_cache` | `tools[].cache{}` | 1 |
| `prompt_cache` | `cache.key`<br>`cache.retention` | 1 |
| `image_detail` | `messages[].user.content[].image.detail`<br>`messages[].tool.content[].image.detail` | 1 |
| `image_source` | `messages[].user.content[].image.source{}`<br>`messages[].assistant.content[].image.source{}`<br>`messages[].tool.content[].image.source{}` | 1 |
| `response_format` | `response_format`<br>`response_format.json_schema.schema{}` | 1 |
| `file_name` | `messages[].user.content[].file.filename`<br>`messages[].tool.content[].file.filename` | 2 |
| `tool_strict` | `tools[].strict` | 1 |
| `tool_result_error` | `messages[].tool.is_error` | 2 |
| `parallel_tool_calls` | `parallel_tool_calls` | 1 |
| `reasoning_config` | `reasoning.effort`<br>`reasoning.output` | 1 |
| `foreign_reasoning` | `messages[].assistant.content[].reasoning.text`<br>`messages[].assistant.content[].reasoning.provider_data{}` | 3 |
| `foreign_other_content` | `messages[].user.content[].other.type_name`<br>`messages[].user.content[].other.origin`<br>`messages[].user.content[].other.data{}`<br>`messages[].assistant.content[].other.type_name`<br>`messages[].assistant.content[].other.origin`<br>`messages[].assistant.content[].other.data{}`<br>`messages[].tool.content[].other.type_name`<br>`messages[].tool.content[].other.origin`<br>`messages[].tool.content[].other.data{}` | 3 |
| `foreign_provider_tool` | `messages[].assistant.content[].provider_tool_call.id`<br>`messages[].assistant.content[].provider_tool_call.name`<br>`messages[].assistant.content[].provider_tool_call.arguments`<br>`messages[].assistant.content[].provider_tool_call.origin`<br>`messages[].assistant.content[].provider_tool_call.provider_data{}`<br>`messages[].assistant.content[].provider_tool_result.call_id`<br>`messages[].assistant.content[].provider_tool_result.output{}`<br>`messages[].assistant.content[].provider_tool_result.is_error`<br>`messages[].assistant.content[].provider_tool_result.origin`<br>`messages[].assistant.content[].provider_tool_result.provider_data{}` | 3 |
| `foreign_provider_data` | `messages[].user.content[].text.provider_data{}`<br>`messages[].user.content[].image.provider_data{}`<br>`messages[].user.content[].file.provider_data{}`<br>`messages[].user.provider_data{}`<br>`messages[].assistant.content[].text.provider_data{}`<br>`messages[].assistant.content[].image.provider_data{}`<br>`messages[].assistant.content[].tool_call.provider_data{}`<br>`messages[].assistant.content[].reasoning.provider_data{}`<br>`messages[].assistant.content[].refusal.provider_data{}`<br>`messages[].assistant.content[].provider_tool_call.provider_data{}`<br>`messages[].assistant.content[].provider_tool_result.provider_data{}`<br>`messages[].assistant.provider_data{}`<br>`messages[].tool.content[].text.provider_data{}`<br>`messages[].tool.content[].image.provider_data{}`<br>`messages[].tool.content[].file.provider_data{}`<br>`messages[].tool.provider_data{}`<br>`tools[].provider_data{}` | 4 |
| `app_data` | `system[].app_data{}`<br>`messages[].user.app_data{}`<br>`messages[].assistant.app_data{}`<br>`messages[].tool.app_data{}` | 5 |

<!-- accounted-fields:end -->

### Vendor fields below the top level (provisional)

`allama_conformance::decoding::assert_vendor_fields_kept` checks that a
response field your adapter does not model is kept wherever it arrives,
not only at the top of the response. It takes one `ResponseCapture` and
the unary replay you give `.unary_replay(..)`, and:

1. adds a field named `x_allama_vendor_field` (`decoding::VENDOR_FIELD`),
   with a marker value of its own, to every JSON object of the capture's
   `chat` body (in a Chat Completions body: the body itself, each choice,
   its message, each tool call, the usage);
2. decodes that body once through `chat`;
3. expects every marker somewhere in the decoded response: in provider
   data at any level, in a warning, or in the content (a field inside a
   tool call's arguments is part of the arguments).

The failure names the JSON pointer of each object whose field was
dropped (`/choices/0/message`, `/usage`). Keeping a field means keeping
it as provider data under your namespace at the level it arrives
([provider data](#provider-data-providerdata)), or warning about it;
dropping it silently fails. It reads the `chat` body only: frames a
stream does not model are C1's
([vendor frames](#6-contracts-and-tests)).

The contract is provisional, and `assert_chat_adapter` does not run it.
No bundled adapter passes it yet, nor does the `text_gateway` template,
because each needs a catch-all at every level of its response types.
Run it on its own to see what your adapter drops. What it checks may
change in any minor release, and it joins the aggregate only once the
bundled adapters and the template pass it and a release says so
([ADR 0017](../decisions/0017-conformance-contract-lifecycle.md)).

## 7. Register with `allama-providers`

Applications that pick providers from configuration use model addresses
(`acme:acme-large`). Register your constructor under a key (no `:`):

```rust
use allama_providers::Registry;

let mut registry = Registry::builtin(); // or Registry::new() for none
registry.register_chat("acme", AcmeProvider::from_env);
```

`registry.chat_from_model("acme:acme-large")` then builds the provider and
returns the model id. Register an embedding, decision or extraction
constructor with `register_embedding`, `register_decision` or
`register_extraction` the same way. The lookup wraps your provider in the
same stack as a built-in one: retries by default, and the per-attempt
timeout and unsupported-feature policy the `ProviderSettings` ask for. So
do not retry inside the adapter, or retries nest.

A constructor registered this way takes no settings, so a lookup that asks
for a credential scope or chat capability overrides
(`chat_from_model_with`) fails for its key. Register it with
`register_chat_with` instead: the constructor receives
`&mut AdapterSettings`, which holds only the settings an adapter applies,
and takes each one it applies with `take_credential_scope()` and
`take_chat_capabilities()` (an `Arc<dyn ChatCapabilityResolver>`, which
your `with_chat_capabilities` takes as is). A setting the lookup set and
the constructor left untaken fails the lookup with `Error::Configuration`,
so a constructor cannot ignore a tenant's credential scope. The lookup
applies the rest of the `ProviderSettings` (retries, the attempt timeout,
the unsupported-feature policy) around what the constructor builds.

The constructor runs on each lookup, so keep it cheap, and make it fail
with `Error::Configuration` when its environment is incomplete; the registry
reports unknown keys and malformed addresses the same way.
