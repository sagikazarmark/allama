use std::sync::{Arc, Mutex};

use allama::{ChatStream, Error, StreamErrorKind};
use futures_util::StreamExt;
use serde_json::json;

use super::*;
use crate::{TransportError, TransportErrorKind};

/// What the decoder was called with, shared with the test.
#[derive(Debug, Default)]
struct Calls {
    frames: Vec<String>,
    ends: usize,
    fails: Vec<Error>,
}

/// A decoder of string frames that yields the scripted items for each
/// frame, and records its calls.
struct Scripted {
    calls: Arc<Mutex<Calls>>,
    script: fn(&str) -> Vec<Result<StreamEvent>>,
    end: Vec<Result<StreamEvent>>,
    done_after: Option<&'static str>,
    done: bool,
}

impl Scripted {
    fn new(script: fn(&str) -> Vec<Result<StreamEvent>>) -> (Self, Arc<Mutex<Calls>>) {
        let calls = Arc::default();
        let decoder = Self {
            calls: Arc::clone(&calls),
            script,
            end: Vec::new(),
            done_after: None,
            done: false,
        };
        (decoder, calls)
    }
}

impl ChatFrameDecoder<&'static str> for Scripted {
    fn frame(&mut self, frame: &'static str) -> Vec<Result<StreamEvent>> {
        self.calls.lock().unwrap().frames.push(frame.to_owned());
        self.done |= self.done_after == Some(frame);
        (self.script)(frame)
    }

    fn end(&mut self) -> Vec<Result<StreamEvent>> {
        self.calls.lock().unwrap().ends += 1;
        std::mem::take(&mut self.end)
    }

    fn fail(&mut self, error: Error) -> Vec<Result<StreamEvent>> {
        self.calls.lock().unwrap().fails.push(error.clone());
        vec![Err(error)]
    }

    fn is_done(&self) -> bool {
        self.done
    }
}

fn delta(data: &str) -> Vec<Result<StreamEvent>> {
    vec![Ok(StreamEvent::text_delta(0, data))]
}

type Frame = std::result::Result<&'static str, TransportError>;

/// A stream of `frames`, which counts its polls.
fn frames(
    frames: Vec<Frame>,
    polls: &Arc<Mutex<usize>>,
) -> impl Stream<Item = Frame> + Send + Unpin + 'static {
    let polls = Arc::clone(polls);
    futures_util::stream::iter(frames).inspect(move |_| *polls.lock().unwrap() += 1)
}

async fn run(items: Vec<Frame>, decoder: Scripted) -> (Vec<Result<StreamEvent>>, usize) {
    let polls = Arc::default();
    let stream = FramedChatStream::with_error_map(
        frames(items, &polls),
        decoder,
        TransportError::into_stream_error,
    );
    let items: Vec<_> = stream.collect().await;
    let polls = *polls.lock().unwrap();
    (items, polls)
}

#[tokio::test]
async fn end_runs_once_when_the_body_ends() {
    let (mut decoder, calls) = Scripted::new(delta);
    decoder.end = vec![Ok(StreamEvent::ResponseStop)];
    let (items, _) = run(vec![Ok("a"), Ok("b")], decoder).await;
    assert_eq!(items.len(), 3);
    assert!(matches!(items[2], Ok(StreamEvent::ResponseStop)));
    let calls = calls.lock().unwrap();
    assert_eq!(calls.frames, ["a", "b"]);
    assert_eq!(calls.ends, 1);
    assert!(calls.fails.is_empty());
}

#[tokio::test]
async fn an_empty_end_terminates() {
    let (decoder, calls) = Scripted::new(delta);
    let (items, _) = run(vec![Ok("a")], decoder).await;
    assert_eq!(items.len(), 1);
    assert_eq!(calls.lock().unwrap().ends, 1);
}

#[tokio::test]
async fn fail_runs_once_when_the_body_fails_and_nothing_follows() {
    let (decoder, calls) = Scripted::new(delta);
    let (items, polls) = run(
        vec![
            Ok("a"),
            Err(TransportError::new(TransportErrorKind::Body, "reset")),
            Ok("b"),
        ],
        decoder,
    )
    .await;
    assert_eq!(items.len(), 2);
    let error = items[1].as_ref().unwrap_err();
    assert!(matches!(
        error,
        Error::Stream {
            kind: StreamErrorKind::Transport,
            ..
        }
    ));
    assert_eq!(error.upstream_message(), Some("reset"));
    assert_eq!(polls, 2, "the body is not polled after it failed");
    let calls = calls.lock().unwrap();
    assert_eq!(calls.fails.len(), 1);
    assert_eq!(calls.ends, 0);
}

#[tokio::test]
async fn the_error_map_classifies_a_failure() {
    let (decoder, _) = Scripted::new(delta);
    let (items, _) = run(
        vec![Err(TransportError::new(
            TransportErrorKind::Timeout,
            "idle",
        ))],
        decoder,
    )
    .await;
    assert!(matches!(items[..], [Err(Error::Timeout { .. })]));
}

#[tokio::test]
async fn the_stream_ends_after_a_decoder_error() {
    let (decoder, calls) = Scripted::new(|data| match data {
        "bad" => vec![
            Ok(StreamEvent::text_delta(0, "before")),
            Err(Error::stream_decode("bad")),
            Ok(StreamEvent::text_delta(0, "dropped")),
        ],
        data => delta(data),
    });
    let (items, polls) = run(vec![Ok("a"), Ok("bad"), Ok("c"), Ok("d")], decoder).await;
    assert_eq!(items.len(), 3, "{items:?}");
    assert!(items[2].is_err());
    assert_eq!(polls, 2, "the body is not polled after the error");
    let calls = calls.lock().unwrap();
    assert_eq!(calls.frames, ["a", "bad"]);
    assert_eq!(calls.ends, 0);
    assert!(calls.fails.is_empty());
}

#[tokio::test]
async fn an_error_from_end_or_fail_is_the_last_item() {
    let (mut decoder, _) = Scripted::new(delta);
    decoder.end = vec![
        Err(Error::stream(StreamErrorKind::Incomplete, "cut")),
        Ok(StreamEvent::ResponseStop),
    ];
    let (items, _) = run(vec![Ok("a")], decoder).await;
    assert_eq!(items.len(), 2, "{items:?}");
    assert!(items[1].is_err());
}

#[tokio::test]
async fn the_stream_ends_once_the_decoder_is_done() {
    let (mut decoder, calls) = Scripted::new(delta);
    decoder.done_after = Some("last");
    let (items, polls) = run(
        vec![
            Ok("a"),
            Ok("last"),
            Ok("ignored"),
            Err(TransportError::new(TransportErrorKind::Body, "never read")),
        ],
        decoder,
    )
    .await;
    assert_eq!(items.len(), 2);
    assert_eq!(polls, 2);
    let calls = calls.lock().unwrap();
    assert_eq!(calls.frames, ["a", "last"]);
    assert_eq!(calls.ends, 0);
    assert!(calls.fails.is_empty());
}

#[tokio::test]
async fn a_failure_that_converts_needs_no_map() {
    /// A framer error of its own, classified by its conversion.
    struct Broken;

    impl From<Broken> for Error {
        fn from(Broken: Broken) -> Self {
            Error::stream_protocol("broken frame").authored()
        }
    }

    struct Frames;

    impl ChatFrameDecoder<u8> for Frames {
        fn frame(&mut self, frame: u8) -> Vec<Result<StreamEvent>> {
            vec![Ok(StreamEvent::text_delta(0, frame.to_string()))]
        }

        fn end(&mut self) -> Vec<Result<StreamEvent>> {
            Vec::new()
        }
    }

    let frames = futures_util::stream::iter([Ok(1), Err(Broken), Ok(2)]);
    let items: Vec<_> = FramedChatStream::new(frames, Frames).collect().await;
    assert_eq!(items.len(), 2);
    assert!(matches!(
        items[1],
        Err(Error::Stream {
            kind: StreamErrorKind::Protocol,
            ..
        })
    ));
}

#[test]
fn the_stream_is_send() {
    let (decoder, _) = Scripted::new(delta);
    let polls = Arc::default();
    let stream = FramedChatStream::with_error_map(
        frames(Vec::new(), &polls),
        decoder,
        TransportError::into_stream_error,
    );
    let _: ChatStream = Box::pin(stream);
}

#[test]
fn vendor_frames_warn_once_per_event_type_and_accumulate() {
    let mut frames = VendorFrames::new("demo", "openai-compat:demo");
    let first = frames.surface("search_results", " {\"n\":1} ");
    let [
        StreamEvent::Warning { warning, .. },
        StreamEvent::ResponseProviderData { .. },
    ] = first.as_slice()
    else {
        panic!("{first:?}")
    };
    assert_eq!(warning.kind, WarningKind::Other);
    assert_eq!(warning.subject, Subject::Response);
    assert_eq!(warning.provider.as_deref(), Some("demo"));
    assert_eq!(
        warning.message,
        "unrecognized stream event `search_results`; its data is kept in the response's \
         provider data under `openai-compat:demo`"
    );

    // No new warning for the same type; each event holds its own payload.
    let second = frames.surface("search_results", "plain text");
    assert_eq!(
        second,
        vec![StreamEvent::response_provider_data(
            ProviderData::new().with_raw(
                "openai-compat:demo",
                "search_results",
                json!(["plain text"]),
            )
        )]
    );

    // Another type warns on its own.
    let other = frames.surface_payload("usage_report", json!({"tokens": 3}));
    assert_eq!(other.len(), 2);
    assert_eq!(
        other[1],
        StreamEvent::response_provider_data(ProviderData::new().with_raw(
            "openai-compat:demo",
            "usage_report",
            json!([{"tokens": 3}]),
        ))
    );

    // Empty data has nothing to keep.
    assert!(frames.surface("search_results", "  \n").is_empty());

    // The finish event holds the whole array of each repeated type.
    assert_eq!(
        frames.finish(),
        Some(StreamEvent::response_provider_data(
            ProviderData::new().with_raw(
                "openai-compat:demo",
                "search_results",
                json!([{"n": 1}, "plain text"]),
            )
        ))
    );
    assert_eq!(frames.finish(), None);
}

/// Each frame costs constant work, and a collector that merges the
/// events and the finish event holds every payload in order.
#[test]
fn vendor_frames_are_linear_and_collect_every_payload() {
    let mut frames = VendorFrames::new("demo", "demo");
    let mut collected = ProviderData::new();
    let mut held = 0;
    for n in 0..5_000 {
        for event in frames.surface_payload("tick", json!(n)) {
            if let StreamEvent::ResponseProviderData { provider_data, .. } = event {
                held += provider_data
                    .get_raw("demo", "tick")
                    .unwrap()
                    .as_array()
                    .unwrap()
                    .len();
                collected.merge(provider_data);
            }
        }
    }
    assert_eq!(held, 5_000, "every event holds one payload");
    let Some(StreamEvent::ResponseProviderData { provider_data, .. }) = frames.finish() else {
        panic!("no finish event");
    };
    collected.merge(provider_data);
    let all = collected
        .get_raw("demo", "tick")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(all.len(), 5_000);
    assert_eq!(all[4_999], json!(4_999));
}
