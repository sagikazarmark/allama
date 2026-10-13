# ADR 0012: Embedding inputs: one trait, ordered parts, modality per model

- Status: accepted

## Context

Embedding APIs take more than text, and they combine it differently:

- Gemini `gemini-embedding-2` combines the text, image, audio, video and
  PDF parts of one content into one vector.
- Cohere Embed v4 and Voyage multimodal take inputs that mix text and
  images.
- Jina, Bedrock Nova and Vertex `multimodalembedding@001` take one item of
  one modality per input.

Which modalities work depends on the model, not the provider type: Gemini
serves text-only `gemini-embedding-001` and multimodal
`gemini-embedding-2` from one endpoint. The input type is hard to change
later, so it had to be right before 1.0.

## Decision

1. **One trait.** `EmbeddingProvider` is the only embedding trait; there is
   no per-modality trait and no generic input type. Text always works;
   other modalities are reported per model through `embedding_capability`
   (`EmbeddingCapability::ImageInput`, `CompositeInput`), as chat reports
   `ChatCapability::ImageInput`. A split by modality would need a third
   trait for mixed inputs, and every trait multiplies the plumbing (dynamic
   type, forwarding, every wrapper, the registry).
2. **An input is an ordered list of parts combined into one vector.** A
   request holds one `EmbeddingInput` per vector it returns. How parts
   combine, whether order matters, and how many of each kind are allowed is
   model-defined, except that every model reporting `CompositeInput`
   accepts at least one text plus one image, in either order. The response
   stays one vector per input; multi-vector output (late interaction,
   per-modality vectors) is a different contract and would be a separate
   operation.
3. **Parts reuse chat's content types.** `EmbeddingPart::Text` holds chat's
   `Text` and `EmbeddingPart::Image` holds chat's `Image`, from the shared
   media types, so content moves between chat and embeddings unchanged.
   Fields that mean nothing here are not silently honored: `Text`
   citations describe model output (ADR 0007) and are cleared by the
   constructors, and `Image::detail` is a hint no embedding API honors.
   Unlike chat's `ContentPart`, `EmbeddingPart` is strict: an unknown part
   type fails to deserialize instead of becoming `Other`, because dropping
   a part changes the vector.
4. **Content is never dropped or joined.** Either would silently change the
   vector, so adapters check inputs before sending:
   - an empty request or an input with no parts is `InvalidRequest`;
   - a part kind the model does not embed, or several parts on a model
     that embeds one part per input, is `Unsupported` under every
     `UnsupportedPolicy` (ADR 0003): a capability gap a fallback may fail
     over on;
   - a combination a combining model does not allow is `InvalidRequest`
     naming the rule: a caller mistake;
   - empty text, or a capability reported `Unknown`, is sent and the
     provider decides;
   - a hint the model cannot honor (image detail, or the request's
     query/document input type) follows `UnsupportedPolicy`: dropped with a
     warning, or an error under `UnsupportedPolicy::Error`. The text is
     never rewritten with a role prefix.

   Part kinds are checked through exhaustive matches, so a new part kind
   does not compile until every check handles it.
5. **URLs are not fetched.** `MediaSource::Url` is an absolute URI; each
   adapter accepts the schemes its provider documents and rejects others
   with `Unsupported`. `Image::media_type` names the type of a URL image
   whose path does not reveal it; inline data carries its own, and a
   conflict is `InvalidRequest`.

### Provider data and the namespace rule

6. **The namespace names the only provider that may read an entry**: the
   origin for state a provider minted (ADR 0002), the addressee for
   settings a caller attaches to a request item, such as an embedding input
   or a tool definition. An empty namespace is the same as an absent one
   for equality, hashing and serialization. Credentials are addressed
   separately, to a credential scope (ADR 0009), which is independent of
   both the namespace and the provider name.
7. **Input and part provider data.** `EmbeddingInput::provider_data`
   carries caller settings for one provider's per-input wire entry (Gemini
   `EmbeddingTitle`); part-shaped settings go in the part's provider data.
   No key is read at both levels. In the adapter's own namespace:
   - only callers set input provider data, so an unknown key there is
     `InvalidRequest`, as ADR 0006 does for options, and so is an input key
     placed on a part;
   - unknown keys on a *part* are ignored. This is a deliberate exception:
     parts taken from chat output carry the provider's replay state (Gemini
     `thought_signature`). The cost is that a misspelled part key is
     ignored; typed part options in provider crates mitigate it.

   Other namespaces are ignored without a warning.
   `EmbeddingRequest::text_inputs` and `EmbeddingRequest::check_provider_data`
   implement these rules once for every adapter.
8. **Serialization is lossless for caller data.** An input that is one plain
   text part serializes as a bare string, so the common case stays compact;
   anything else is `{"parts": [..], "provider_data"?: {..}}`, and the
   object form rejects unknown keys. A field added later makes older
   readers fail on newer recordings, which is accepted: losing caller
   content silently is worse than a loud version-skew error.

### Responses are checked

9. **A response is checked, never repaired.**
   `allama::adapter::embedding::check_response` fails with
   `Error::UnexpectedResponse` for a vector count other than the input
   count, ragged or empty vectors, a size other than the requested
   `dimensions`, or a non-finite value. An index built from such vectors is
   silently wrong, and truncating, padding or normalizing would change the
   vectors just as joining parts would. Every first-party adapter calls it
   on what it decodes. `embed_chunked` also checks every chunk against the
   first chunk's vector size, since a fallback may switch models mid-batch.
   `allama_conformance::probe::assert_embedding_response_checked` is the
   contract.

## Consequences

- `EmbeddingRequest::inputs` holds `EmbeddingInput`s, and strings convert,
  so `.inputs(["a", "b"])` stays the common path.
- Inputs implement `Hash` and work as cache keys, provided parts are built
  through the constructors (a directly built `Text` part keeps citations,
  which count for equality).
- `embed_input` fails unless exactly one vector comes back.
- `EmbeddingCapability::BatchInput` and `EmbeddingLimits` stay distinct: a
  model that embeds one input per call is a capability gap (`Unsupported`);
  a batch over `max_inputs_per_call` is a caller mistake (`InvalidRequest`)
  that `embed_chunked` avoids.
- The design is grounded in more than one provider: Gemini
  `gemini-embedding-2` through its adapter, and test-only encoders for
  Voyage multimodal and Cohere Embed v4
  (`crates/allama/tests/embedding_input_grounding.rs`).
- Audio, video and document parts are added once a second provider
  supports each. Segmented media, token-id inputs, and quantized or
  multi-vector outputs are out of scope.
