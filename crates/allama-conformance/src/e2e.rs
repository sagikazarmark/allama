//! Live end-to-end scenarios every provider runs against its real API.
//!
//! Each scenario builds its own request for `model`, calls the provider,
//! and checks the portable contract on what comes back (non-empty text,
//! finish reasons, valid tool-call JSON, stream reconstruction, usage).
//! They spend tokens, so provider crates run them from `#[ignore]`d tests
//! gated on an API key:
//!
//! ```no_run
//! use allama_conformance::e2e;
//! # fn provider() -> allama::mock::MockProvider { allama::mock::MockProvider::with_text("ok") }
//!
//! #[tokio::test]
//! #[ignore = "calls the live API"]
//! async fn basic_chat() {
//!     e2e::basic_chat(&provider(), "my-model").await;
//! }
//! ```
//!
//! `extract` and `extract_enum_root` require the `extract` feature.

#[cfg(feature = "chat-extract")]
use allama::chat::extract::ChatExtractExt;
use allama::{
    CapabilitySupport, ChatProvider, ChatRequest, ChatStreamExt, ContentPart, EmbeddingCapability,
    EmbeddingInput, EmbeddingInputType, EmbeddingPart, EmbeddingProvider, EmbeddingRequest, Error,
    FinishReason, ReasoningConfig, ResponseFormat, StreamEvent, ToolDefinition,
};
use futures_util::StreamExt;
#[cfg(feature = "chat-extract")]
use schemars::JsonSchema;
#[cfg(feature = "chat-extract")]
use serde::Deserialize;
use serde_json::json;

/// Validate basic chat: non-empty text, Stop finish reason, usage present.
pub async fn basic_chat(provider: &impl ChatProvider, model: &str) {
    let request = ChatRequest::new(model).user("Say hello in one sentence.");

    let response = provider
        .chat(&request)
        .await
        .expect("basic_chat: chat request failed");

    let text = response.text().expect("basic_chat: response has no text");
    assert!(!text.is_empty(), "basic_chat: response text is empty");

    assert_eq!(
        response.finish_reason,
        Some(FinishReason::Stop),
        "basic_chat: expected Stop finish reason, got {:?}",
        response.finish_reason
    );

    let usage = response
        .usage
        .as_ref()
        .expect("basic_chat: usage is missing");
    assert!(
        usage.input_tokens.is_some_and(|t| t > 0),
        "basic_chat: input_tokens missing or zero"
    );
    assert!(
        usage.output_tokens.is_some_and(|t| t > 0),
        "basic_chat: output_tokens missing or zero"
    );
    crate::assert_usage_contract(usage);
    assert!(
        response.provider.is_some(),
        "basic_chat: adapters must stamp the serving provider"
    );
}

/// Validate streaming: events arrive, collector produces valid response with same structural checks.
pub async fn streaming(provider: &impl ChatProvider, model: &str) {
    let request = ChatRequest::new(model).user("Say hello in one sentence.");

    let stream = provider
        .chat_stream(&request)
        .await
        .expect("streaming: chat_stream request failed");

    let response = stream
        .collect_response()
        .await
        .expect("streaming: stream collection failed");

    let text = response.text().expect("streaming: response has no text");
    assert!(!text.is_empty(), "streaming: response text is empty");

    assert_eq!(
        response.finish_reason,
        Some(FinishReason::Stop),
        "streaming: expected Stop finish reason, got {:?}",
        response.finish_reason
    );

    let usage = response
        .usage
        .as_ref()
        .expect("streaming: usage is missing");
    assert!(
        usage.input_tokens.is_some_and(|t| t > 0),
        "streaming: input_tokens missing or zero"
    );
    assert!(
        usage.output_tokens.is_some_and(|t| t > 0),
        "streaming: output_tokens missing or zero"
    );
}

/// Validate system prompt: structural checks pass when a system message is included.
pub async fn system_prompt(provider: &impl ChatProvider, model: &str) {
    let request = ChatRequest::new(model)
        .system("You are a helpful assistant.")
        .user("Say hello in one sentence.");

    let response = provider
        .chat(&request)
        .await
        .expect("system_prompt: chat request failed");

    let text = response
        .text()
        .expect("system_prompt: response has no text");
    assert!(!text.is_empty(), "system_prompt: response text is empty");

    assert_eq!(
        response.finish_reason,
        Some(FinishReason::Stop),
        "system_prompt: expected Stop finish reason, got {:?}",
        response.finish_reason
    );
}

/// Validate multi-turn: a 2-turn conversation is accepted and produces a valid response.
pub async fn multi_turn(provider: &impl ChatProvider, model: &str) {
    let request = ChatRequest::new(model)
        .user("My name is Alice.")
        .assistant("Hello Alice! How can I help you today?")
        .user("What is my name?");

    let response = provider
        .chat(&request)
        .await
        .expect("multi_turn: chat request failed");

    let text = response.text().expect("multi_turn: response has no text");
    assert!(!text.is_empty(), "multi_turn: response text is empty");

    assert_eq!(
        response.finish_reason,
        Some(FinishReason::Stop),
        "multi_turn: expected Stop finish reason, got {:?}",
        response.finish_reason
    );
}

/// Validate tool calling: model returns tool calls with valid JSON arguments.
pub async fn tool_calling(provider: &impl ChatProvider, model: &str) {
    let tool = ToolDefinition::new(
        "get_weather",
        json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" }
            },
            "required": ["city"],
            "additionalProperties": false
        }),
    )
    .description("Get the current weather for a city.");

    let request = ChatRequest::new(model)
        .tools(vec![tool])
        .user("What is the weather in San Francisco?");

    let response = provider
        .chat(&request)
        .await
        .expect("tool_calling: chat request failed");

    assert!(
        response.has_tool_calls(),
        "tool_calling: expected tool calls in response"
    );

    for call in response.tool_calls() {
        assert!(
            !call.name.is_empty(),
            "tool_calling: tool call has empty name"
        );
        assert!(
            serde_json::from_str::<serde_json::Value>(&call.arguments).is_ok(),
            "tool_calling: tool call arguments are not valid JSON: {}",
            call.arguments
        );
    }

    assert_eq!(
        response.finish_reason,
        Some(FinishReason::ToolCalls),
        "tool_calling: expected ToolCalls finish reason, got {:?}",
        response.finish_reason
    );
}

/// Validate structured output: JSON schema response format produces parseable JSON.
pub async fn structured_output(provider: &impl ChatProvider, model: &str) {
    let schema = json!({
        "type": "object",
        "properties": {
            "greeting": { "type": "string" }
        },
        "required": ["greeting"],
        "additionalProperties": false
    });

    let request = ChatRequest::new(model)
        .response_format(ResponseFormat::json_schema(schema.clone()).name("greeting"))
        .user("Say hello in JSON format with a 'greeting' field.");

    let response = provider
        .chat(&request)
        .await
        .expect("structured_output: chat request failed");

    let text = response
        .text()
        .expect("structured_output: response has no text");

    let parsed: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|e| {
        panic!("structured_output: response is not valid JSON: {e}\nraw: {text}")
    });

    assert!(
        parsed.get("greeting").is_some(),
        "structured_output: JSON missing 'greeting' field: {parsed}"
    );
}

/// Validate typed extraction: uses native structured output when the provider
/// supports it, otherwise falls back to a forced tool call.
#[cfg(feature = "chat-extract")]
pub async fn extract<P: ChatProvider + Sync>(provider: &P, model: &str) {
    #[derive(JsonSchema, Deserialize)]
    struct Greeting {
        greeting: String,
    }

    let request = ChatRequest::new(model).user("Say hello in JSON format with a 'greeting' field.");

    let extracted = provider
        .extract::<Greeting>(&request)
        .await
        .expect("extract: extraction failed");

    assert!(
        !extracted.value.greeting.is_empty(),
        "extract: greeting field is empty"
    );
}

/// Extraction into a type whose schema root is not an object (a bare
/// string enum), which strict structured output may reject.
///
/// Passes when the extraction succeeds (warnings allowed) or is refused as
/// unsupported before sending; fails when the provider is sent a request it
/// rejects as invalid (an HTTP 400).
#[cfg(feature = "chat-extract")]
pub async fn extract_enum_root<P: ChatProvider + Sync>(provider: &P, model: &str) {
    #[derive(JsonSchema, Deserialize)]
    #[serde(rename_all = "lowercase")]
    #[allow(dead_code)]
    enum Label {
        Billing,
        Bug,
        Account,
    }

    let request = ChatRequest::new(model)
        .user("Label this support ticket: \"I was charged twice this month.\"");
    let result = provider.extract::<Label>(&request).await;
    let refused = |err: &Error| match err {
        Error::Unsupported { .. } => true,
        Error::Extract { failure, .. } => {
            matches!(
                failure.error(),
                allama::chat::extract::ChatExtractError::Unsupported { .. }
            )
        }
        _ => false,
    };
    assert_not_rejected("extract_enum_root", result.as_ref().err(), refused);
}

/// Temperature 0 on a reasoning model, which some reasoning models reject.
///
/// Passes when the call succeeds (an `Unsupported` warning allowed) or is
/// refused with [`Error::Unsupported`] before sending; fails when the
/// provider is sent a request it rejects as invalid (an HTTP 400).
pub async fn sampling_on_reasoning_model(provider: &impl ChatProvider, model: &str) {
    let request = ChatRequest::new(model)
        .user("Say hello in one sentence.")
        .temperature(0.0);
    let result = provider.chat(&request).await;
    let refused = |err: &Error| matches!(err, Error::Unsupported { .. });
    assert_not_rejected(
        "sampling_on_reasoning_model",
        result.as_ref().err(),
        refused,
    );
}

fn assert_not_rejected(scenario: &str, err: Option<&Error>, refused: impl Fn(&Error) -> bool) {
    let Some(err) = err else {
        return;
    };
    assert!(
        !matches!(err, Error::InvalidRequest { .. }),
        "{scenario}: the provider rejected the request as invalid; the adapter must \
         adapt it, warn, or fail with Unsupported before sending: {err}"
    );
    assert!(
        refused(err),
        "{scenario}: expected success or Unsupported, got {}: {err}",
        err.kind_name()
    );
}

/// Validate basic embedding: non-empty vector of f32, matches input count.
pub async fn basic_embed(provider: &impl EmbeddingProvider, model: &str) {
    let request = EmbeddingRequest::new(model).input("hello world");
    let response = provider.embed(&request).await.expect("embed failed");

    assert_eq!(response.embeddings.len(), 1, "expected 1 embedding");
    assert!(
        !response.embeddings[0].is_empty(),
        "expected non-empty vector"
    );
}

/// Validate batch embedding: output count matches input count and order.
pub async fn batch_embed(provider: &impl EmbeddingProvider, model: &str) {
    let inputs = ["first input", "second input", "third input"];
    let request = EmbeddingRequest::new(model).inputs(inputs);
    let response = provider.embed(&request).await.expect("embed failed");

    assert_eq!(
        response.embeddings.len(),
        inputs.len(),
        "batch output count should match input count"
    );
    let dim = response.embeddings[0].len();
    for vector in &response.embeddings {
        assert_eq!(
            vector.len(),
            dim,
            "all vectors should share the same length"
        );
    }
}

/// Validate optional `dimensions` request field when the provider supports it.
/// No-ops when the provider reports `Unsupported` or `Unknown`.
pub async fn dimensions(provider: &impl EmbeddingProvider, model: &str, requested: u32) {
    let support = provider.embedding_capability(model, EmbeddingCapability::OutputDimensions);
    if support != CapabilitySupport::Supported {
        return;
    }
    let request = EmbeddingRequest::new(model)
        .input("hello dimensions")
        .dimensions(requested);
    let response = provider.embed(&request).await.expect("embed failed");
    assert_eq!(response.embeddings.len(), 1);
    assert_eq!(
        response.embeddings[0].len(),
        requested as usize,
        "vector length should equal the requested dimensions"
    );
}

/// Validate the query/document input type when the provider
/// supports it: documents embedded with `Document` and a query embedded with
/// `Query` are served without warnings, and the query lands closer to the
/// document that answers it than to an unrelated one. No-ops unless
/// `InputType` is `Supported`.
pub async fn input_type_embed(provider: &impl EmbeddingProvider, model: &str) {
    if provider.embedding_capability(model, EmbeddingCapability::InputType)
        != CapabilitySupport::Supported
    {
        return;
    }
    let documents = EmbeddingRequest::new(model)
        .inputs([
            "The Eiffel Tower is a wrought-iron tower in Paris, France.",
            "Minestrone is an Italian soup made with vegetables and pasta.",
        ])
        .input_type(EmbeddingInputType::Document);
    let documents = provider.embed(&documents).await.expect("embed failed");
    assert!(
        documents.warnings.is_empty(),
        "a supported input type is served without warnings: {:?}",
        documents.warnings
    );
    let query = EmbeddingRequest::new(model)
        .input("Where is the Eiffel Tower?")
        .input_type(EmbeddingInputType::Query);
    let query = provider.embed(&query).await.expect("embed failed");
    assert!(query.warnings.is_empty(), "{:?}", query.warnings);
    let [answer, unrelated] = &documents.embeddings[..] else {
        panic!("expected one vector per document");
    };
    assert!(
        cosine(&query.embeddings[0], answer) > cosine(&query.embeddings[0], unrelated),
        "the query should be closer to the document that answers it"
    );
}

/// Validate image embeddings when the provider supports them: an image input
/// and a text input describing it land in one vector space, closer to each
/// other than to an unrelated text. No-ops unless `ImageInput` is `Supported`.
pub async fn image_embed(provider: &impl EmbeddingProvider, model: &str) {
    if provider.embedding_capability(model, EmbeddingCapability::ImageInput)
        != CapabilitySupport::Supported
    {
        return;
    }
    let request = EmbeddingRequest::new(model)
        .input(EmbeddingInput::image_base64("image/png", RED_PNG_BASE64))
        .input("a plain red square")
        .input("a recipe for vegetable soup");
    let response = provider.embed(&request).await.expect("embed failed");
    assert_eq!(response.embeddings.len(), 3, "one vector per input");
    let [image, related, unrelated] = &response.embeddings[..] else {
        unreachable!()
    };
    assert_eq!(image.len(), related.len(), "image and text share a space");
    assert!(
        cosine(image, related) > cosine(image, unrelated),
        "the image should be closer to its description than to an unrelated text"
    );
}

/// Validate composite inputs when the provider supports them: a text and an
/// image combine into one vector. No-ops unless `CompositeInput` is
/// `Supported`.
pub async fn composite_embed(provider: &impl EmbeddingProvider, model: &str) {
    if provider.embedding_capability(model, EmbeddingCapability::CompositeInput)
        != CapabilitySupport::Supported
    {
        return;
    }
    let request = EmbeddingRequest::new(model)
        .input(vec![
            EmbeddingPart::text("Product photo:"),
            EmbeddingPart::image_base64("image/png", RED_PNG_BASE64),
        ])
        .input("a plain red square");
    let response = provider.embed(&request).await.expect("embed failed");
    assert_eq!(
        response.embeddings.len(),
        2,
        "a composite input yields one vector"
    );
    assert_eq!(response.embeddings[0].len(), response.embeddings[1].len());
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

/// A 256x256 solid red PNG (OpenAI ignores very small images).
pub const RED_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAIAAADTED8xAAAB+0lEQVR42u3TQQkAAAjAwPUvrX8reHAJBmsK3pIAA4ABwABgADAAGAAMAAYAA4ABwABgADAAGAAMAAYAA4ABwABgADAAGAAMAAYAA4ABwABgADAAGAAMAAYAA4ABwABgADAAGAAMAAYAA4ABwABgADAAGAAMAAYAA4ABwAAYAAwABgADgAHAAGAAMAAYAAwABgADgAHAAGAAMAAYAAwABgADgAHAAGAAMAAYAAwABgADgAHAAGAAMAAYAAwABgADgAHAAGAAMAAYAAwABgADgAHAAGAAMAAYAAwABgAJMAAYAAwABgADgAHAAGAAMAAYAAwABgADgAHAAGAAMAAYAAwABgADgAHAAGAAMAAYAAwABgADgAHAAGAAMAAYAAwABgADgAHAAGAAMAAYAAwABgADgAHAAGAAMAAYAAyAASTAAGAAMAAYAAwABgADgAHAAGAAMAAYAAwABgADgAHAAGAAMAAYAAwABgADgAHAAGAAMAAYAAwABgADgAHAAGAAMAAYAAwABgADgAHAAGAAMAAYAAwABgADgAHAAGAAMAAGAAOAAcAAYAAwABgADAAGAAOAAcAAYAAwABgADAAGAAOAAcAAYAAwABgADAAGAAOAAcAAYAAwABgADAAGAAOAAcAAYAAwABgADAAGAAOAAcAAYAAwABgADADHAllMDvLkz2XNAAAAAElFTkSuQmCC";

/// A one-page PDF containing "The secret word is pineapple."
pub const PINEAPPLE_PDF_BASE64: &str = "JVBERi0xLjQKMSAwIG9iago8PCAvVHlwZSAvQ2F0YWxvZyAvUGFnZXMgMiAwIFIgPj4KZW5kb2JqCjIgMCBvYmoKPDwgL1R5cGUgL1BhZ2VzIC9LaWRzIFszIDAgUl0gL0NvdW50IDEgPj4KZW5kb2JqCjMgMCBvYmoKPDwgL1R5cGUgL1BhZ2UgL1BhcmVudCAyIDAgUiAvTWVkaWFCb3ggWzAgMCAzMDAgMTQ0XSAvQ29udGVudHMgNCAwIFIgL1Jlc291cmNlcyA8PCAvRm9udCA8PCAvRjEgNSAwIFIgPj4gPj4gPj4KZW5kb2JqCjQgMCBvYmoKPDwgL0xlbmd0aCA1OSA+PgpzdHJlYW0KQlQgL0YxIDE4IFRmIDIwIDcwIFRkIChUaGUgc2VjcmV0IHdvcmQgaXMgcGluZWFwcGxlLikgVGogRVQKZW5kc3RyZWFtCmVuZG9iago1IDAgb2JqCjw8IC9UeXBlIC9Gb250IC9TdWJ0eXBlIC9UeXBlMSAvQmFzZUZvbnQgL0hlbHZldGljYSA+PgplbmRvYmoKeHJlZgowIDYKMDAwMDAwMDAwMCA2NTUzNSBmIAowMDAwMDAwMDA5IDAwMDAwIG4gCjAwMDAwMDAwNTggMDAwMDAgbiAKMDAwMDAwMDExNSAwMDAwMCBuIAowMDAwMDAwMjQxIDAwMDAwIG4gCjAwMDAwMDAzNTAgMDAwMDAgbiAKdHJhaWxlcgo8PCAvU2l6ZSA2IC9Sb290IDEgMCBSID4+CnN0YXJ0eHJlZgo0MjAKJSVFT0YK";

fn weather_tool() -> ToolDefinition {
    ToolDefinition::new(
        "get_weather",
        json!({
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
            "additionalProperties": false
        }),
    )
    .description("Get the current weather for a city.")
}

/// A long streamed reply must arrive in several chunks and still collect.
/// Guards against decoders that assume cumulative chunks.
pub async fn streaming_multi_chunk(provider: &impl ChatProvider, model: &str) {
    let request = ChatRequest::new(model).user(
        "Write a short paragraph of about 120 words describing a walk through an autumn forest.",
    );
    let mut stream = Box::pin(
        provider
            .chat_stream(&request)
            .await
            .expect("streaming_multi_chunk: chat_stream failed"),
    );
    let mut collector = allama::chat::stream::StreamCollector::new();
    let mut text_deltas = 0usize;
    while let Some(event) = stream.next().await {
        let event = event.expect("streaming_multi_chunk: stream error");
        if matches!(event, StreamEvent::TextDelta { .. }) {
            text_deltas += 1;
        }
        collector
            .push(event)
            .expect("streaming_multi_chunk: protocol error");
    }
    let response = collector
        .finish()
        .expect("streaming_multi_chunk: incomplete");
    let text = response.text().unwrap_or_default().to_lowercase();
    assert!(
        text_deltas > 1,
        "streaming_multi_chunk: expected several text deltas, got {text_deltas}"
    );
    assert!(
        text.split_whitespace().count() >= 40,
        "streaming_multi_chunk: reply too short: {text}"
    );
}

/// Call a tool, replay the assistant turn with the result, and get a final
/// answer. With `reasoning`, this exercises replay of opaque reasoning state
/// (signatures) attached to the assistant turn.
pub async fn tool_round_trip(
    provider: &impl ChatProvider,
    model: &str,
    reasoning: Option<ReasoningConfig>,
) {
    let mut request = ChatRequest::new(model)
        .tools([weather_tool()])
        .user("What is the weather in Paris? Use the tool, then answer in one sentence.");
    if let Some(reasoning) = reasoning {
        request = request.reasoning(reasoning);
    }
    let first = provider
        .chat(&request)
        .await
        .expect("tool_round_trip: first turn failed");
    let calls: Vec<_> = first.tool_calls().cloned().collect();
    assert!(!calls.is_empty(), "tool_round_trip: expected a tool call");
    request.push_message(first.to_assistant_message());
    for call in &calls {
        request.push_tool_result(call, "Sunny, 18 degrees Celsius.");
    }
    let second = provider
        .chat(&request)
        .await
        .expect("tool_round_trip: replay turn failed");
    assert!(
        second.text().is_some_and(|t| !t.is_empty()),
        "tool_round_trip: expected a final text answer"
    );
}

/// Same as [`tool_round_trip`], but streaming both turns.
pub async fn streaming_tool_round_trip(
    provider: &impl ChatProvider,
    model: &str,
    reasoning: Option<ReasoningConfig>,
) {
    let mut request = ChatRequest::new(model)
        .tools([weather_tool()])
        .user("What is the weather in Paris? Use the tool, then answer in one sentence.");
    if let Some(reasoning) = reasoning {
        request = request.reasoning(reasoning);
    }
    let first = provider
        .chat_stream(&request)
        .await
        .expect("streaming_tool_round_trip: first turn failed")
        .collect_response()
        .await
        .expect("streaming_tool_round_trip: first turn collection failed");
    let calls: Vec<_> = first.tool_calls().cloned().collect();
    assert!(
        !calls.is_empty(),
        "streaming_tool_round_trip: expected a tool call"
    );
    request.push_message(first.to_assistant_message());
    for call in &calls {
        request.push_tool_result(call, "Sunny, 18 degrees Celsius.");
    }
    let second = provider
        .chat_stream(&request)
        .await
        .expect("streaming_tool_round_trip: replay failed")
        .collect_response()
        .await
        .expect("streaming_tool_round_trip: replay collection failed");
    assert!(second.text().is_some_and(|t| !t.is_empty()));
}

/// A provider configured with an invalid key fails with `Error::Auth`.
pub async fn error_bad_key(provider_with_bad_key: &impl ChatProvider, model: &str) {
    let err = provider_with_bad_key
        .chat(&ChatRequest::new(model).user("hi"))
        .await
        .expect_err("error_bad_key: expected an error");
    assert!(
        matches!(err, Error::Auth { .. }),
        "error_bad_key: expected Auth, got {err:?}"
    );
    assert!(!err.is_retryable());
}

/// The model can see an image.
pub async fn vision(provider: &impl ChatProvider, model: &str) {
    let request = ChatRequest::new(model).user(vec![
        ContentPart::text("What single color fills this image? Answer with one word."),
        ContentPart::image_base64("image/png", RED_PNG_BASE64),
    ]);
    let response = provider.chat(&request).await.expect("vision: chat failed");
    let text = response.text().unwrap_or_default().to_lowercase();
    assert!(text.contains("red"), "vision: unexpected answer: {text}");
}

/// The model can read a PDF document.
pub async fn file_input(provider: &impl ChatProvider, model: &str) {
    let request = ChatRequest::new(model).user(vec![
        ContentPart::text("What is the secret word in the document? Answer with one word."),
        ContentPart::file_base64("application/pdf", PINEAPPLE_PDF_BASE64),
    ]);
    let response = provider
        .chat(&request)
        .await
        .expect("file_input: chat failed");
    let text = response.text().unwrap_or_default().to_lowercase();
    assert!(
        text.contains("pineapple"),
        "file_input: unexpected answer: {text}"
    );
}

/// The model can call several tools in one turn.
pub async fn parallel_tools(provider: &impl ChatProvider, model: &str) {
    let request = ChatRequest::new(model)
        .tools([weather_tool()])
        .parallel_tool_calls(true)
        .user("What is the weather in Paris and in Tokyo? Call the tool once per city.");
    let response = provider
        .chat(&request)
        .await
        .expect("parallel_tools: chat failed");
    assert!(
        response.tool_calls().count() >= 2,
        "parallel_tools: expected at least two tool calls, got {:?}",
        response.content
    );
}
