//! Contract C1 against a minimal adapter built on `allama_http::sse`, and
//! against the framing bugs C1 must catch.
#![cfg(not(target_arch = "wasm32"))]

use allama::chat::stream::StreamBlockType;
use allama::{Error, FinishReason, StreamErrorKind, StreamEvent};
use allama_conformance::stream_frames::{StreamFraming, assert_unknown_stream_frames_surfaced};
use allama_http::sse::{SseDecoder, VendorFrames};
use serde_json::Value;

/// Commas outside strings, so the framing variants split the data lines.
const BASE: &str = concat!(
    ": keepalive\n\n",
    "data: {\"text\":\"Hello, \",\"n\":1}\n\n",
    "data: {\"text\":\"world\",\"n\":2}\n\n",
    "data: [DONE]\n\n",
);

/// Splits an SSE body into `(event, data)` pairs.
type Framer = fn(&str) -> Vec<(String, String)>;

#[tokio::test]
async fn an_adapter_on_the_sse_helpers_passes() {
    run(frame_with_helper).await;
}

#[tokio::test]
#[should_panic(expected = "CRLF line endings")]
async fn a_decoder_that_assumes_lf_fails() {
    run(frame_lf_only).await;
}

#[tokio::test]
#[should_panic(expected = "data split over several `data:` lines")]
async fn a_decoder_that_keeps_one_data_line_fails() {
    run(frame_last_data_line).await;
}

async fn run(framer: Framer) {
    assert_unknown_stream_frames_surfaced(BASE, &StreamFraming::sse(), |sse| async move {
        // `Prepared::finish_stream` holds what arrives before the start;
        // the decoder holds nothing.
        Ok(allama::adapter::chat::start_first(
            futures_util::stream::iter(decode(&framer(&sse))),
        ))
    })
    .await;
}

fn frame_with_helper(sse: &str) -> Vec<(String, String)> {
    let mut decoder = SseDecoder::new();
    decoder.push(sse.as_bytes());
    let mut frames = Vec::new();
    while let Some(event) = decoder.next_event().unwrap() {
        frames.push((event.event, event.data));
    }
    frames
}

/// Bug: splits on `\n\n` and `\n` only.
fn frame_lf_only(sse: &str) -> Vec<(String, String)> {
    sse.split("\n\n")
        .filter_map(|block| {
            let mut event = "message".to_owned();
            let mut data = None;
            for line in block.split('\n') {
                if let Some(name) = line.strip_prefix("event: ") {
                    event = name.to_owned();
                } else if let Some(value) = line.strip_prefix("data: ") {
                    data = Some(value.to_owned());
                }
            }
            data.map(|data| (event, data))
        })
        .collect()
}

/// Bug: a later `data:` line replaces the earlier ones.
fn frame_last_data_line(sse: &str) -> Vec<(String, String)> {
    frame_lf_only(&sse.replace("\r\n", "\n"))
}

/// A toy protocol: `{"text": ..}` chunks, then `[DONE]`.
fn decode(frames: &[(String, String)]) -> Vec<allama::Result<StreamEvent>> {
    let mut vendor_frames = VendorFrames::new("toy", "toy");
    let mut out = Vec::new();
    let mut started = false;
    for (event, data) in frames {
        if event != "message" {
            out.extend(vendor_frames.surface(event, data).into_iter().map(Ok));
            continue;
        }
        if data == "[DONE]" {
            out.push(Ok(StreamEvent::block_stop(0)));
            let mut metadata = StreamEvent::metadata();
            if let StreamEvent::ResponseMetadata { finish_reason, .. } = &mut metadata {
                *finish_reason = Some(FinishReason::Stop);
            }
            out.push(Ok(metadata));
            out.extend(vendor_frames.finish().map(Ok));
            out.push(Ok(StreamEvent::ResponseStop));
            return out;
        }
        let text = match serde_json::from_str::<Value>(data) {
            Ok(chunk) => chunk["text"].as_str().unwrap_or_default().to_owned(),
            Err(error) => {
                out.push(Err(Error::stream_decode(error.to_string())));
                return out;
            }
        };
        if !started {
            started = true;
            out.push(Ok(StreamEvent::response_start(Some("r1".into()), None)));
            out.push(Ok(StreamEvent::block_start(0, StreamBlockType::Text)));
        }
        out.push(Ok(StreamEvent::text_delta(0, text)));
    }
    out.push(Err(Error::stream(StreamErrorKind::Incomplete, "no [DONE]")));
    out
}
