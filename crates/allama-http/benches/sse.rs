//! SSE decoding throughput, for reporting: a long line fed in small chunks
//! (where a decoder that rescans its pending line goes quadratic) and a
//! stream of small events.

use std::hint::black_box;

use allama_http::sse::SseDecoder;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

/// Decode `text` fed in `chunk`-byte chunks; the number of events.
fn decode(text: &[u8], chunk: usize) -> usize {
    let mut decoder = SseDecoder::new();
    let mut events = 0;
    for bytes in text.chunks(chunk) {
        decoder.push(bytes);
        while decoder.next_event().unwrap().is_some() {
            events += 1;
        }
    }
    events
}

fn bench_long_line(c: &mut Criterion) {
    // An 8 MiB `data:` line, as a large base64 image or a repeated
    // response produces.
    let text = format!("data: \"{}\"\n\n", "A".repeat(8 << 20)).into_bytes();
    let mut group = c.benchmark_group("sse_long_line");
    group.throughput(Throughput::Bytes(text.len() as u64));
    group.sample_size(10);
    for chunk in [8 << 10, 16 << 10] {
        group.bench_with_input(BenchmarkId::from_parameter(chunk), &chunk, |b, &chunk| {
            b.iter(|| black_box(decode(black_box(&text), chunk)));
        });
    }
    group.finish();
}

fn bench_small_events(c: &mut Criterion) {
    let text = "event: delta\ndata: {\"type\":\"text_delta\",\"text\":\"Hello\"}\n\n"
        .repeat(10_000)
        .into_bytes();
    let mut group = c.benchmark_group("sse_small_events");
    group.throughput(Throughput::Bytes(text.len() as u64));
    group.bench_function("16KiB_chunks", |b| {
        b.iter(|| black_box(decode(black_box(&text), 16 << 10)));
    });
    group.finish();
}

criterion_group!(benches, bench_long_line, bench_small_events);
criterion_main!(benches);
