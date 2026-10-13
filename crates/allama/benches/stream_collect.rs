use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

use allama::chat::stream::{StreamBlockType, StreamCollector};
use allama::{FinishReason, ProviderData, StreamEvent, Usage};

fn sample_events() -> Vec<StreamEvent> {
    let mut events = vec![StreamEvent::response_start(
        Some("resp_bench".into()),
        Some("bench-model".into()),
    )];

    for index in 0..24 {
        events.push(match index % 3 {
            0 => StreamEvent::tool_call_start(
                index,
                format!("call_{index}"),
                format!("tool_{index}"),
            ),
            1 => StreamEvent::block_start(index, StreamBlockType::Text),
            _ => StreamEvent::block_start(index, StreamBlockType::Reasoning),
        });
        match index % 3 {
            0 => events.push(StreamEvent::tool_call_delta(
                index,
                format!(r#"{{"value":{index},"payload":"{}"}}"#, "x".repeat(96)),
            )),
            1 => events.push(StreamEvent::text_delta(
                index,
                "Lorem ipsum dolor sit amet. ".repeat(6),
            )),
            _ => {
                events.push(StreamEvent::reasoning_delta(
                    index,
                    "Deliberating carefully. ".repeat(5),
                ));
                events.push(StreamEvent::block_provider_data(
                    index,
                    ProviderData::new().with_raw("bench", "signature", "sig".into()),
                ));
            }
        }
        events.push(StreamEvent::block_stop(index));
    }

    let mut metadata = StreamEvent::metadata();
    if let StreamEvent::ResponseMetadata {
        finish_reason,
        usage,
        ..
    } = &mut metadata
    {
        *finish_reason = Some(FinishReason::Stop);
        *usage = Some(
            Usage::new()
                .input_tokens(1024)
                .output_tokens(256)
                .total_tokens(1280),
        );
    }
    events.push(metadata);
    events.push(StreamEvent::ResponseStop);
    events
}

fn bench_stream_collect(c: &mut Criterion) {
    let events = sample_events();
    c.bench_function("stream_collector_collect_response", |b| {
        b.iter(|| {
            let mut collector = StreamCollector::new();
            for event in black_box(&events) {
                collector.push(event.clone()).unwrap();
            }
            black_box(collector.finish().unwrap())
        })
    });
}

criterion_group!(benches, bench_stream_collect);
criterion_main!(benches);
