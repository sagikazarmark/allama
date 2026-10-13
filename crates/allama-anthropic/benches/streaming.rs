use allama::{ChatProvider, ChatRequest, ChatStream, Result, StreamEvent};
use allama_http::{ReplayResponse, ReplayTransport};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use futures_util::StreamExt;
use std::hint::black_box;

fn anthropic_sse_text() -> &'static str {
    "event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_bench\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-20250514\",\"stop_reason\":null,\"usage\":{\"input_tokens\":128,\"output_tokens\":0}}}\n\
\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\
\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Analyzing the request carefully.\"}}\n\
\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig_bench\"}}\n\
\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\
\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\
\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello from Anthropic.\"}}\n\
\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":1}\n\
\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_bench\",\"name\":\"search_docs\",\"input\":{}}}\n\
\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"query\\\":\\\"bench\"}}\n\
\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"mark\\\"}\"}}\n\
\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":2}\n\
\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":64}}\n\
\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\
\n"
}

/// A provider whose transport replays each queued SSE body, so a bench
/// iteration decodes through the public `chat_stream` path.
fn provider(transport: &ReplayTransport) -> allama_anthropic::Provider {
    allama_anthropic::Provider::builder()
        .api_key("bench")
        .http_transport(transport.clone())
        .build()
        .expect("provider")
}

/// Open a stream over `text`: the request is encoded and the response
/// headers are read here, outside the measured decode.
fn open_stream(
    provider: &allama_anthropic::Provider,
    transport: &ReplayTransport,
    text: &str,
) -> ChatStream {
    transport.push(ReplayResponse::sse(text));
    futures_executor::block_on(
        provider.chat_stream(&ChatRequest::new("claude-sonnet-4-20250514").user("bench")),
    )
    .expect("stream")
}

fn drain(mut stream: ChatStream) -> Vec<Result<StreamEvent>> {
    futures_executor::block_on(async move {
        let mut out = Vec::new();
        while let Some(event) = stream.next().await {
            out.push(event);
        }
        out
    })
}

fn bench_anthropic_streaming(c: &mut Criterion) {
    let text = anthropic_sse_text();
    let transport = ReplayTransport::default();
    let provider = provider(&transport);
    assert!(
        drain(open_stream(&provider, &transport, text))
            .iter()
            .all(Result::is_ok)
    );
    c.bench_function("anthropic_stream_event_translation", |b| {
        b.iter_batched(
            || open_stream(&provider, &transport, black_box(text)),
            |stream| black_box(drain(stream)),
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_anthropic_streaming);
criterion_main!(benches);
