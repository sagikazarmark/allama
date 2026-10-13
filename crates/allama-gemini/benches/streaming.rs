use allama::{ChatProvider, ChatRequest, ChatStream, Result, StreamEvent};
use allama_http::{ReplayResponse, ReplayTransport};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use futures_util::StreamExt;
use std::hint::black_box;

/// Recorded delta chunks (thoughts, text, and a signed function call).
fn gemini_sse_text() -> String {
    let chunks = [
        include_str!("../fixtures/stream_thoughts.sse"),
        r#"data: {"responseId":"resp_bench","modelVersion":"gemini-3.5-flash","candidates":[{"content":{"parts":[{"functionCall":{"name":"search_docs","args":{"query":"bench","limit":3},"id":"call_1"},"thoughtSignature":"c2ln"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":120,"candidatesTokenCount":24,"totalTokenCount":144}}"#,
    ];
    let mut text = String::new();
    for chunk in chunks {
        // Keep only the last chunk's finish reason: drop STOP from the
        // recorded thought stream so the function call ends the response.
        text.push_str(&chunk.replace(r#""finishReason": "STOP","#, ""));
        text.push_str("\n\n");
    }
    text
}

/// A provider whose transport replays each queued SSE body, so a bench
/// iteration decodes through the public `chat_stream` path.
fn provider(transport: &ReplayTransport) -> allama_gemini::Provider {
    allama_gemini::Provider::builder()
        .api_key("bench")
        .http_transport(transport.clone())
        .build()
        .expect("provider")
}

/// Open a stream over `text`: the request is encoded and the response
/// headers are read here, outside the measured decode.
fn open_stream(
    provider: &allama_gemini::Provider,
    transport: &ReplayTransport,
    text: &str,
) -> ChatStream {
    transport.push(ReplayResponse::sse(text));
    futures_executor::block_on(
        provider.chat_stream(&ChatRequest::new("gemini-3.5-flash").user("bench")),
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

fn bench_gemini_streaming(c: &mut Criterion) {
    let text = gemini_sse_text();
    let transport = ReplayTransport::default();
    let provider = provider(&transport);
    assert!(
        drain(open_stream(&provider, &transport, &text))
            .iter()
            .all(Result::is_ok)
    );
    c.bench_function("gemini_stream_event_translation", |b| {
        b.iter_batched(
            || open_stream(&provider, &transport, black_box(&text)),
            |stream| black_box(drain(stream)),
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_gemini_streaming);
criterion_main!(benches);
