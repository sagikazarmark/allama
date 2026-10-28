//! The extraction request, recovery, and extraction bookkeeping.
//!
//! An extraction keeps every field of the chat request it is given, and
//! refuses a request that carries tools of its own, in every mode. To
//! extract from the conversation alone, start a fresh request from its
//! messages (or clear `tools` and `tool_choice`), and ask for the value in
//! a final user message. Every extraction reports the calls it made and
//! their summed usage, on success and on failure.

use allama::chat::extract::{ChatExtractError, ChatExtracted, ChatExtractionMode, ChatExtractor};
use allama::mock::{ChatResponseBuilder, MockProvider, MockResponse};
use allama::prelude::*;
use allama::schemars::{self, JsonSchema};
use allama::{CapabilitySupport, ChatCapability, FinishReason, Usage};
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize, JsonSchema)]
struct Review {
    title: String,
    rating: u8,
}

fn provider() -> MockProvider {
    MockProvider::new([MockResponse::tool_call(
        "call_extract_1",
        "submit_structured_output",
        json!({ "title": "Heat", "rating": 10 }),
    )])
    .with_chat_capability(
        ChatCapability::StructuredOutput,
        CapabilitySupport::Unsupported,
    )
    .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // A finished conversation, which ends on the assistant's answer.
    let chat = ChatRequest::new("demo-model")
        .system("You are a support assistant.")
        .user("Summarize customer 42's review.")
        .assistant("Customer 42 says Heat deserves a 10 out of 10.")
        .tools([ToolDefinition::new(
            "lookup_customer",
            json!({"type": "object"}),
        )]);
    let instruction = "Extract the review from this conversation.";

    let request = chat.clone().user(instruction);
    match provider().extract::<Review>(&request).await {
        Err(Error::Extract { failure, .. })
            if matches!(failure.error(), ChatExtractError::ToolConflict { .. }) =>
        {
            println!("extraction refuses a request that carries its own tools");
        }
        other => println!("unexpected: {other:?}"),
    }

    // Start a fresh request from the conversation: the caller's tools (and
    // sampling settings) do not apply. To keep the rest of the request,
    // clear `tools` and `tool_choice` on a copy instead. The instruction is
    // the final user message: a conversation that ends on the assistant's
    // answer would send it as a prefill, which several providers refuse
    // with structured output.
    let conversation = ChatRequest::new(&chat.model)
        .extend_messages(chat.messages.clone())
        .user(instruction);
    let extracted: ChatExtracted<Review> = ChatExtractor::new()
        .extract(
            &provider(),
            &conversation
                .clone()
                .with_option(ChatExtractionMode::ForcedTool),
        )
        .await?;
    println!(
        "extraction from the conversation: {} ({}/10)",
        extracted.value.title, extracted.value.rating
    );

    // Opt-in recovery: repair a fenced reply locally, and ask once more if
    // the reply still does not match the schema (one extra, billed call).
    let flaky = MockProvider::build(|b| {
        b.response(
            ChatResponseBuilder::new()
                .text(r#"Here you go: {"title": "Heat"}"#)
                .usage(120, 12)
                .build(),
        )
        .response(
            ChatResponseBuilder::new()
                .text("```json\n{\"title\": \"Heat\", \"rating\": 10}\n```")
                .usage(160, 18)
                .build(),
        )
    });
    let recovered: ChatExtracted<Review> = ChatExtractor::new()
        .repair(true)
        .retries(1)
        .extract(
            &flaky,
            &conversation.clone().with_option(ChatExtractionMode::Native),
        )
        .await?;
    println!(
        "recovered extraction: {} ({}/10) after {} calls, repaired: {}, total tokens: {:?}",
        recovered.value.title,
        recovered.value.rating,
        recovered.metadata.passes,
        recovered.metadata.repaired,
        recovered.metadata.usage.as_ref().and_then(Usage::total),
    );

    // A failure keeps the last response and the cost of every pass. Output
    // cut off at the token limit is reported as truncated and not retried.
    let truncated = MockProvider::with_response(
        ChatResponseBuilder::new()
            .text(r#"{"title": "He"#)
            .finish_reason(FinishReason::Length)
            .usage(120, 8)
            .build(),
    );
    match ChatExtractor::new()
        .retries(2)
        .extract::<_, Review>(
            &truncated,
            &conversation.with_option(ChatExtractionMode::Native),
        )
        .await
    {
        Err(Error::Extract { failure, .. }) => println!(
            "failed: {failure} (calls: {}, total tokens: {:?}, partial output: {:?})",
            failure.metadata().passes,
            failure.metadata().usage.as_ref().and_then(Usage::total),
            failure.response().and_then(ChatResponse::text),
        ),
        other => println!("unexpected: {other:?}"),
    }

    // Parse a response you already have (a batch result, a recording).
    let stored = ChatResponseBuilder::new()
        .text(r#"{"title": "Heat", "rating": 10}"#)
        .build();
    let parsed: ChatExtracted<Review> =
        ChatExtractor::new().parse(stored, ChatExtractionMode::Native)?;
    println!("parsed stored response: {}", parsed.value.title);
    Ok(())
}
