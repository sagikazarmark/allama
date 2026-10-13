use allama::chat::stream::StreamCollector;
use allama::extract::Extracted;
use allama::prelude::*;
use allama::{DynChatProvider, DynEmbeddingProvider};
use allama::{Error, Result, ToolChoice, ToolDefinition};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use support::{KNOWN_PROVIDERS, KnownProvider, Loaded};

mod support;

#[derive(Debug, Deserialize, JsonSchema)]
struct LiveExtract {
    token: String,
    summary: String,
}

#[derive(Debug, Deserialize)]
struct LookupFactArgs {
    topic: String,
}

type ChatTarget = Loaded<DynChatProvider>;

fn load_live_provider(selection: &str, known: KnownProvider) -> Result<Option<ChatTarget>> {
    if !support::selection_includes(selection, known.key) {
        return Ok(None);
    }

    if !known.is_configured() {
        if support::selection_uses_configured_alias(selection) {
            eprintln!(
                "skipping {} live checks; {} is not set",
                known.key, known.credential_env
            );
            return Ok(None);
        }

        return Err(Error::invalid_request(format!(
            "{} selected by ALLAMA_LIVE_PROVIDER but {} is not set",
            known.key, known.credential_env
        )));
    }

    known.load_chat().map(Some)
}

fn lookup_fact_tool() -> ToolDefinition {
    ToolDefinition::new(
        "lookup_fact",
        json!({
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic to look up"
                }
            },
            "required": ["topic"],
            "additionalProperties": false
        }),
    )
    .description("Look up a short factual answer for the requested topic.")
}

async fn run_live_chat(target: &ChatTarget) -> Result<()> {
    let response = target
        .provider
        .chat(
            &ChatRequest::new(&target.model)
                .system("Reply in one short sentence and include the exact token live-chat-ok.")
                .user("Confirm that this provider smoke test is working."),
        )
        .await?;

    support::assert_contains_token(&response.text_or_empty(), "live-chat-ok");
    Ok(())
}

async fn run_live_stream(target: &ChatTarget) -> Result<()> {
    let request = ChatRequest::new(&target.model)
        .system("Reply in one short sentence and include the exact token live-stream-ok.")
        .user("Confirm that this streaming smoke test is working.");

    let mut stream = target.provider.chat_stream(&request).await?;
    let mut collector = StreamCollector::new();
    let mut saw_text_delta = false;

    while let Some(item) = stream.next_event().await {
        let event = item?;
        saw_text_delta |= matches!(event, StreamEvent::TextDelta { .. });
        collector.push(event)?;
    }

    let response = collector.finish()?;
    assert!(saw_text_delta, "expected at least one text delta");
    support::assert_contains_token(&response.text_or_empty(), "live-stream-ok");
    Ok(())
}

async fn run_live_tool_roundtrip(target: &ChatTarget) -> Result<()> {
    let mut request = ChatRequest::new(&target.model)
        .system(
            "When tools are available and required, call the lookup_fact tool first. After you receive the tool result, answer with that result in one short sentence.",
        )
        .user("What color is the sky?")
        .tools(vec![lookup_fact_tool()])
        .tool_choice(ToolChoice::Required);

    let first = target.provider.chat(&request).await?;
    assert!(first.has_tool_calls(), "expected at least one tool call");

    request.push_message(first.to_assistant_message());
    for call in first.tool_calls() {
        assert_eq!(call.name, "lookup_fact", "unexpected live tool name");

        let args: LookupFactArgs = call.parse_arguments()?;
        let normalized_topic = args.topic.trim().to_ascii_lowercase();
        assert!(
            normalized_topic == "sky" || normalized_topic == "the sky",
            "expected lookup_fact topic to target the sky, got: {}",
            args.topic
        );

        request.push_tool_result(call, "The sky is blue. live-tool-ok.");
    }

    let second = target.provider.chat(&request).await?;
    support::assert_contains_token(&second.text_or_empty(), "live-tool-ok");
    Ok(())
}

async fn run_live_embed(target: &Loaded<DynEmbeddingProvider>) -> Result<()> {
    let inputs = [
        "allama live embedding smoke input A".to_string(),
        "allama live embedding smoke input B".to_string(),
    ];
    let request = EmbeddingRequest::new(&target.model).inputs(inputs.iter().cloned());
    let response = target.provider.embed(&request).await?;

    assert_eq!(
        response.embeddings.len(),
        inputs.len(),
        "expected embedding count to match input count"
    );
    let dim = response.embeddings[0].len();
    assert!(dim > 0, "expected non-empty embedding vector");
    for vector in &response.embeddings {
        assert_eq!(
            vector.len(),
            dim,
            "all embedding vectors should share the same length"
        );
    }
    Ok(())
}

async fn run_live_extract(target: &ChatTarget) -> Result<()> {
    let request = ChatRequest::new(&target.model)
        .system(
            "Return structured data with token set to the exact string live-extract-ok and summary set to one short sentence.",
        )
        .user("Confirm that this structured extraction smoke test is working.");

    let extracted: Extracted<LiveExtract> = target.provider.extract(&request).await?;
    assert_eq!(extracted.value.token, "live-extract-ok");
    assert!(
        !extracted.value.summary.trim().is_empty(),
        "expected structured summary to be populated"
    );
    Ok(())
}

#[tokio::test]
async fn live_http_providers_smoke() -> Result<()> {
    let Some(selection) = support::live_selection() else {
        eprintln!(
            "skipping HTTP provider live checks; set ALLAMA_LIVE_PROVIDER=openai|openai-chat|anthropic|gemini|all"
        );
        return Ok(());
    };

    support::validate_live_selection(&selection)?;
    support::ensure_selection_has_runnable_target(&selection)?;

    let mut executed = 0usize;
    for known in KNOWN_PROVIDERS {
        let Some(target) = load_live_provider(&selection, known)? else {
            continue;
        };

        eprintln!(
            "running live HTTP smoke checks for model address={}",
            target.address
        );
        run_live_chat(&target).await?;
        run_live_stream(&target).await?;
        run_live_tool_roundtrip(&target).await?;
        run_live_extract(&target).await?;

        if let Some(embedding_target) = known.load_embedding()? {
            eprintln!(
                "running live embedding smoke check for model address={}",
                embedding_target.address
            );
            run_live_embed(&embedding_target).await?;
        }

        executed += 1;
    }

    if executed == 0 {
        eprintln!(
            "no configured HTTP providers matched ALLAMA_LIVE_PROVIDER={selection}; another selected target must satisfy the live gate"
        );
    }

    Ok(())
}
