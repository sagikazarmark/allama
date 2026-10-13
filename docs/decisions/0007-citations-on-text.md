# ADR 0007: Citations are annotations on text

- Status: accepted

## Context

Anthropic, OpenAI (Responses and Chat Completions) and Gemini all report
sources for grounded answers, and recorded responses show the same shape:
a citation links a stretch of *generated text* to the *source* that
supports it. They differ in where the link lives and how positions are
counted:

| Provider | Where | Position in the generated text |
|---|---|---|
| Anthropic | `citations` on each `text` block | the whole block |
| OpenAI Responses | `annotations` on each `output_text` part | code-point offsets |
| OpenAI Chat Completions | `message.annotations` | code-point offsets |
| Gemini | `groundingMetadata` on the candidate, at its end | byte offsets into the answer |

Only Anthropic quotes the source text (`cited_text`).

## Decision

- **Citations are annotations on `Text`** (`Text::citations`), not content
  blocks. A separate source block would lose the link to the text it
  supports, and Anthropic replays citations as part of the text block.
- A `Citation` has a `source` (`CitationSource::Url`, `Document` for a
  document supplied in the request, or `File` for a file stored with the
  provider), an optional `span`, the quoted `cited_text` when reported,
  and `provider_data` for everything else (page numbers, encrypted
  indexes, confidence scores).
- **`span` is a byte range into the block's text**, so it slices a Rust
  string directly; `None` means the whole block. Adapters convert the
  provider's offsets and drop a span that does not land on the text
  rather than guess.
- Streaming adds `StreamEvent::Citation`, which the collector appends to
  the open text block. Like all block data it arrives before the block's
  `BlockStop`.
- **Citations are output only.** Adapters keep the raw provider
  annotations in the text block's provider data and replay those to
  their origin; the portable list is never re-encoded, so editing it does
  not change what is replayed.
- **The origin is the wire format, not just the namespace.** Chat
  Completions and Responses share the `openai` namespace but shape
  annotations differently. Chat Completions keeps its raw annotations
  under `chat_annotations`, and the Responses encoder replays only flat
  Responses-shaped `annotations`, dropping others with a warning. A
  conversation moving between the two APIs keeps its portable citations,
  not its raw annotations.
- Gemini grounding arrives only at the end of a candidate, so its adapter
  holds the answer blocks open until then and attaches each support to
  the block its offsets fall in. A support that matches no answer text is
  reported with a warning; the raw `groundingMetadata` stays in the
  response's provider data either way.
- Request-side grounding and search configuration stay provider options.

## Consequences

- Callers render citations without parsing provider payloads;
  `ChatResponse::citations` iterates over all of them.
- Offset conversion is adapter work that needs recorded evidence: OpenAI's
  code-point offsets are verified on recordings with characters outside
  the Basic Multilingual Plane before the citation, for both APIs.
- Gemini answer blocks end only with the candidate, so a stream consumer
  sees their `BlockStop` late.

## Rejected alternatives

- **A `ContentBlock::Source` block.** It separates a source from the text
  it supports and cannot be replayed as Anthropic expects.
- **Character offsets in `span`.** No provider agrees on the unit, and a
  byte range is what Rust string slicing takes.
