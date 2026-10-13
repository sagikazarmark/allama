use std::hint::black_box;

use allama::{
    ChatProvider, ChatRequest, ChatStream, ContentBlock, ContentPart, Image, ImageDetail, Message,
    ToolChoice, ToolDefinition,
};
use allama_http::{ReplayResponse, ReplayTransport};
use allama_openai_compat::{CompatQuirks, Provider};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use futures_util::StreamExt;
use serde_json::json;

fn bench_request() -> ChatRequest {
    ChatRequest::new("gpt-4o-mini")
        .system("You are a structured assistant.")
        .user(vec![
            ContentPart::text("Summarize this image and text payload."),
            ContentPart::Image(Image::url("https://example.com/image.png").with_detail(ImageDetail::High)),
        ])
        .message(Message::assistant_blocks(vec![
            ContentBlock::text("I can help with that. ".repeat(8)),
            ContentBlock::tool_call(
                "call_1",
                "search_docs",
                json!({"query": "streaming API design", "limit": 5}).to_string(),
            ),
        ]))
        .message(Message::tool_result("call_1", "search_docs", "ok"))
        .temperature(0.2)
        .max_output_tokens(512)
        .tool_choice(ToolChoice::Auto)
        .tools([ToolDefinition::new(
            "search_docs",
            json!({"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}}),
        )])
}

fn sse_lines() -> Vec<String> {
    let mut lines = vec![
        r#"{"id":"c","model":"m","choices":[{"index":0,"delta":{"role":"assistant"}}]}"#.to_owned(),
    ];
    for i in 0..64 {
        lines.push(format!(
            r#"{{"id":"c","choices":[{{"index":0,"delta":{{"content":"token {i} "}}}}]}}"#
        ));
    }
    lines
        .push(r#"{"id":"c","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#.to_owned());
    lines.push("[DONE]".to_owned());
    lines
}

/// A provider whose transport replays each queued body, so the benches run
/// the public paths.
fn provider(transport: &ReplayTransport) -> Provider {
    Provider::builder()
        .base_url("http://bench.invalid/v1")
        .quirks(CompatQuirks::openai())
        .http_transport(transport.clone())
        .build()
        .expect("provider")
}

/// Open a stream over `body`: the request is encoded and the response
/// headers are read here, outside the measured decode.
fn open_stream(provider: &Provider, transport: &ReplayTransport, body: &str) -> ChatStream {
    transport.push(ReplayResponse::sse(body));
    futures_executor::block_on(provider.chat_stream(&ChatRequest::new("gpt-4o-mini").user("bench")))
        .expect("stream")
}

fn drain(mut stream: ChatStream) -> usize {
    futures_executor::block_on(async move {
        let mut n = 0;
        while let Some(event) = stream.next().await {
            event.expect("event");
            n += 1;
        }
        n
    })
}

fn bench(c: &mut Criterion) {
    let transport = ReplayTransport::default();
    let provider = provider(&transport);
    let request = bench_request();
    // The encoding dry run: everything `chat` does before sending.
    c.bench_function("chat_request_encoding", |b| {
        b.iter(|| provider.validate_chat_request(black_box(&request)).unwrap())
    });
    let body: String = sse_lines()
        .iter()
        .map(|line| format!("data: {line}\n\n"))
        .collect();
    c.bench_function("sse_decode", |b| {
        b.iter_batched(
            || open_stream(&provider, &transport, black_box(&body)),
            drain,
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
