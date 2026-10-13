//! Contract C1 in NDJSON against a minimal adapter built on
//! `allama_http::ndjson` and `allama_http::framing`, and against the
//! decoder bugs C1 must catch, with the body replayed a byte at a time.
#![cfg(not(target_arch = "wasm32"))]

use allama::chat::stream::StreamBlockType;
use allama::{Error, FinishReason, Result, StreamErrorKind, StreamEvent};
use allama_conformance::stream_frames::{StreamFraming, assert_unknown_stream_frames_surfaced};
use allama_http::framing::{ChatFrameDecoder, FramedChatStream, VendorFrames};
use allama_http::ndjson::NdjsonStream;
use allama_http::{HttpRequest, HttpTransport, ReplayResponse, ReplayTransport};
use serde_json::{Value, json};

/// Text lines, then a `done` line.
const BASE: &str = concat!(
    "{\"type\":\"text\",\"id\":\"r1\",\"delta\":\"Hello, \"}\n",
    "\n",
    "{\"type\":\"text\",\"delta\":\"world\"}\r\n",
    "{\"type\":\"done\",\"finish\":\"stop\"}\n",
);

/// What a decoder does with a line whose type it does not know.
#[derive(Clone, Copy)]
enum Unknown {
    Surface,
    Drop,
    DecodeAsText,
}

#[tokio::test]
async fn an_adapter_on_the_ndjson_helpers_passes() {
    run(Unknown::Surface).await;
}

#[tokio::test]
#[should_panic(expected = "was not surfaced")]
async fn a_decoder_that_drops_unknown_lines_fails() {
    run(Unknown::Drop).await;
}

#[tokio::test]
#[should_panic(expected = "ResponseStart changed")]
async fn a_decoder_that_reads_unknown_lines_as_its_own_fails() {
    run(Unknown::DecodeAsText).await;
}

async fn run(unknown: Unknown) {
    // A line's type is its `type` field.
    let framing = StreamFraming::ndjson(|frame_type, payload| {
        let mut line = payload.clone();
        line["type"] = json!(frame_type);
        line.to_string()
    });
    assert_unknown_stream_frames_surfaced(BASE, &framing, |body| async move {
        let transport = ReplayTransport::new([ReplayResponse::chunks(200, [body]).split_every(1)]);
        let response = transport
            .send(HttpRequest::get("https://acme.example/stream"))
            .await
            .unwrap();
        let lines = NdjsonStream::new(response.into_body());
        let decoder = Lines {
            unknown,
            vendor_frames: VendorFrames::new("toy", "toy"),
            started: false,
            done: false,
        };
        // `Prepared::finish_stream` holds what arrives before the start.
        Ok(allama::adapter::chat::start_first(FramedChatStream::new(
            lines, decoder,
        )))
    })
    .await;
}

/// A toy protocol: `text` lines, then `done`.
struct Lines {
    unknown: Unknown,
    vendor_frames: VendorFrames,
    started: bool,
    done: bool,
}

impl ChatFrameDecoder<String> for Lines {
    fn frame(&mut self, line: String) -> Vec<Result<StreamEvent>> {
        let json: Value = match serde_json::from_str(&line) {
            Ok(json) => json,
            Err(error) => return vec![Err(Error::stream_decode(error.to_string()))],
        };
        let mut out = Vec::new();
        let kind = json["type"].as_str().unwrap_or_default();
        let kind = match (kind, self.unknown) {
            ("text" | "done", _) | (_, Unknown::DecodeAsText) => kind,
            (_, Unknown::Drop) => return out,
            (other, Unknown::Surface) => {
                let frames = self.vendor_frames.surface_payload(other, json.clone());
                return frames.into_iter().map(Ok).collect();
            }
        };
        if !self.started {
            self.started = true;
            let id = json["id"].as_str().map(str::to_owned);
            out.push(Ok(StreamEvent::response_start(id, None)));
            out.push(Ok(StreamEvent::block_start(0, StreamBlockType::Text)));
        }
        if kind == "done" {
            self.done = true;
            out.push(Ok(StreamEvent::block_stop(0)));
            let mut metadata = StreamEvent::metadata();
            if let StreamEvent::ResponseMetadata { finish_reason, .. } = &mut metadata {
                *finish_reason = Some(FinishReason::Stop);
            }
            out.push(Ok(metadata));
            out.extend(self.vendor_frames.finish().map(Ok));
            out.push(Ok(StreamEvent::ResponseStop));
        } else {
            let text = json["delta"].as_str().unwrap_or("?");
            out.push(Ok(StreamEvent::text_delta(0, text)));
        }
        out
    }

    fn end(&mut self) -> Vec<Result<StreamEvent>> {
        vec![Err(Error::stream(StreamErrorKind::Incomplete, "no `done`"))]
    }

    fn is_done(&self) -> bool {
        self.done
    }
}
