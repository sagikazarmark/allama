//! The documented order of raw stream events.
//!
//! [`StreamCollector`](allama::chat::stream::StreamCollector) is lenient:
//! a delta before its `BlockStart` opens the block, and metadata may come
//! early. Applications that read the raw events are not: a UI keyed on
//! `BlockStart`, the announcement of a tool call by name, a check for the
//! first output event. So adapters keep the documented order (see
//! [`StreamEvent`]), and [`assert_stream_event_order`] checks it on the raw
//! events:
//!
//! 1. `ResponseStart` comes first and once;
//! 2. a block's events (deltas, citations, block provider data, provider
//!    deltas) come after its `BlockStart` and before its `BlockStop`, each
//!    block starts and stops once, and a delta matches the block's type;
//! 3. every block stops before the `ResponseMetadata` that carries the
//!    finish reason;
//! 4. `ResponseStop` comes last and once.
//!
//! `Warning` and `ResponseProviderData` events may appear at any point.
//! Blocks may overlap: a block may stay open while later blocks stream.

use std::collections::HashMap;
use std::future::Future;

use allama::chat::stream::StreamBlockType;
use allama::{Result, StreamEvent};
use futures_core::Stream;

use crate::decoding::events;

/// Assert that `events`, a stream that ended without an error, keep the
/// documented order (see the [module docs](self)). `label` names the stream
/// in failures.
///
/// # Panics
///
/// At the first event out of order, naming it and its position.
pub fn assert_stream_event_order(label: &str, events: &[StreamEvent]) {
    if let Err(problem) = check(events) {
        panic!(
            "stream_event_order: {label}: {problem}\n  events: {:?}",
            events.iter().map(name).collect::<Vec<_>>()
        );
    }
}

/// Decode `body` through `replay` (as
/// [`stream_replay`](crate::adapter::AdapterUnderTest::stream_replay) does)
/// and assert that the raw events keep the documented order.
///
/// # Panics
///
/// When the stream fails to start or ends with an error, or an event is out
/// of order.
pub async fn assert_stream_order_kept<R, RFut, S>(label: &str, body: &str, replay: R)
where
    R: Fn(String) -> RFut,
    RFut: Future<Output = Result<S>>,
    S: Stream<Item = Result<StreamEvent>>,
{
    let stream = replay(body.to_owned())
        .await
        .unwrap_or_else(|err| panic!("stream_event_order: {label}: the stream failed: {err:?}"));
    let events: Vec<StreamEvent> = events(stream)
        .await
        .into_iter()
        .collect::<Result<_>>()
        .unwrap_or_else(|err| {
            panic!("stream_event_order: {label}: the stream ended with an error: {err:?}")
        });
    assert_stream_event_order(label, &events);
}

/// Where a block is in its life.
struct Block {
    block_type: StreamBlockType,
    stopped: bool,
}

fn check(events: &[StreamEvent]) -> std::result::Result<(), String> {
    let mut started = false;
    let mut stopped = false;
    let mut blocks: HashMap<usize, Block> = HashMap::new();
    for (position, event) in events.iter().enumerate() {
        let at = |what: String| Err(format!("event {position} ({}): {what}", name(event)));
        if stopped {
            return at("an event after ResponseStop".into());
        }
        let body = matches!(
            event,
            StreamEvent::Warning { .. } | StreamEvent::ResponseProviderData { .. }
        );
        if !started && !body && !matches!(event, StreamEvent::ResponseStart { .. }) {
            return at("ResponseStart must come first".into());
        }
        match event {
            StreamEvent::ResponseStart { .. } => {
                if started {
                    return at("a second ResponseStart".into());
                }
                started = true;
            }
            StreamEvent::BlockStart {
                index, block_type, ..
            } => {
                if blocks.contains_key(index) {
                    return at(format!("block {index} starts twice"));
                }
                blocks.insert(
                    *index,
                    Block {
                        block_type: *block_type,
                        stopped: false,
                    },
                );
            }
            StreamEvent::BlockStop { index, .. } => match blocks.get_mut(index) {
                None => return at(format!("block {index} stops before it starts")),
                Some(block) if block.stopped => return at(format!("block {index} stops twice")),
                Some(block) => block.stopped = true,
            },
            StreamEvent::ResponseMetadata {
                finish_reason: Some(_),
                ..
            } => {
                if let Some(index) = open(&blocks) {
                    return at(format!(
                        "the finish reason arrives while block {index} is open; stop every \
                         block first"
                    ));
                }
            }
            StreamEvent::ResponseStop => {
                if let Some(index) = open(&blocks) {
                    return at(format!("block {index} never stops"));
                }
                stopped = true;
            }
            _ => {
                if let Some((index, expected)) = block_event(event) {
                    match blocks.get(&index) {
                        None => {
                            return at(format!("an event for block {index} before its BlockStart"));
                        }
                        Some(block) if block.stopped => {
                            return at(format!("an event for block {index} after its BlockStop"));
                        }
                        Some(block) => {
                            if let Some(expected) = expected
                                && !expected.contains(&block.block_type)
                            {
                                return at(format!(
                                    "a delta that does not fit block {index} ({:?})",
                                    block.block_type
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    if !stopped {
        return Err("the stream ends without ResponseStop".into());
    }
    Ok(())
}

/// The lowest open block, if any.
fn open(blocks: &HashMap<usize, Block>) -> Option<usize> {
    blocks
        .iter()
        .filter(|(_, block)| !block.stopped)
        .map(|(index, _)| *index)
        .min()
}

/// The block an event belongs to, and the block types its delta fits
/// (`None`: any type).
fn block_event(event: &StreamEvent) -> Option<(usize, Option<&'static [StreamBlockType]>)> {
    use StreamBlockType as T;
    Some(match event {
        StreamEvent::TextDelta { index, .. } => (*index, Some(&[T::Text][..])),
        StreamEvent::ReasoningDelta { index, .. } => (*index, Some(&[T::Reasoning][..])),
        StreamEvent::RefusalDelta { index, .. } => (*index, Some(&[T::Refusal][..])),
        StreamEvent::ToolCallDelta { index, .. } => {
            (*index, Some(&[T::ToolCall, T::ProviderToolCall][..]))
        }
        StreamEvent::Citation { index, .. } => (*index, Some(&[T::Text][..])),
        StreamEvent::BlockProviderData { index, .. } | StreamEvent::ProviderDelta { index, .. } => {
            (*index, None)
        }
        _ => return None,
    })
}

/// An event's name and block index, for failure messages that must not
/// print model output.
fn name(event: &StreamEvent) -> String {
    let index = block_event(event).map(|(index, _)| index).or(match event {
        StreamEvent::BlockStart { index, .. } | StreamEvent::BlockStop { index, .. } => {
            Some(*index)
        }
        _ => None,
    });
    let variant = match event {
        StreamEvent::ResponseStart { .. } => "ResponseStart",
        StreamEvent::BlockStart { .. } => "BlockStart",
        StreamEvent::TextDelta { .. } => "TextDelta",
        StreamEvent::ReasoningDelta { .. } => "ReasoningDelta",
        StreamEvent::RefusalDelta { .. } => "RefusalDelta",
        StreamEvent::ToolCallDelta { .. } => "ToolCallDelta",
        StreamEvent::Citation { .. } => "Citation",
        StreamEvent::BlockProviderData { .. } => "BlockProviderData",
        StreamEvent::ResponseProviderData { .. } => "ResponseProviderData",
        StreamEvent::ProviderDelta { .. } => "ProviderDelta",
        StreamEvent::BlockStop { .. } => "BlockStop",
        StreamEvent::Warning { .. } => "Warning",
        StreamEvent::ResponseMetadata {
            finish_reason: Some(_),
            ..
        } => "ResponseMetadata(finish)",
        StreamEvent::ResponseMetadata { .. } => "ResponseMetadata",
        StreamEvent::ResponseStop => "ResponseStop",
        _ => "event",
    };
    match index {
        Some(index) => format!("{variant}[{index}]"),
        None => variant.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use allama::FinishReason;
    use allama::chat::stream::MetadataBuilder;

    use super::*;

    fn finish() -> StreamEvent {
        MetadataBuilder::new()
            .with_finish_reason(FinishReason::Stop)
            .build()
    }

    fn ordered() -> Vec<StreamEvent> {
        vec![
            StreamEvent::response_start(None, None),
            StreamEvent::block_start(0, StreamBlockType::Text),
            StreamEvent::text_delta(0, "Hello"),
            StreamEvent::block_stop(0),
            finish(),
            StreamEvent::ResponseStop,
        ]
    }

    #[test]
    fn the_documented_order_passes() {
        assert_stream_event_order("ordered", &ordered());
    }

    #[test]
    fn overlapping_blocks_and_early_metadata_pass() {
        let events = vec![
            StreamEvent::response_start(None, None),
            MetadataBuilder::new().with_model("m".to_owned()).build(),
            StreamEvent::block_start(0, StreamBlockType::Text),
            StreamEvent::text_delta(0, "a"),
            StreamEvent::block_start(1, StreamBlockType::ToolCall),
            StreamEvent::tool_call_delta(1, "{}"),
            StreamEvent::block_stop(1),
            StreamEvent::text_delta(0, "b"),
            StreamEvent::block_stop(0),
            finish(),
            StreamEvent::ResponseStop,
        ];
        assert_stream_event_order("overlap", &events);
    }

    /// The collector accepts this stream and the next one; the order check
    /// does not.
    #[test]
    #[should_panic(expected = "before its BlockStart")]
    fn a_delta_before_its_block_start_fails() {
        let mut events = ordered();
        events.swap(1, 2);
        assert_stream_event_order("delta first", &events);
    }

    #[test]
    #[should_panic(expected = "while block 0 is open")]
    fn a_finish_before_block_stop_fails() {
        let mut events = ordered();
        events.swap(3, 4);
        assert_stream_event_order("finish first", &events);
    }

    #[test]
    #[should_panic(expected = "ResponseStart must come first")]
    fn a_missing_response_start_fails() {
        let mut events = ordered();
        events.remove(0);
        assert_stream_event_order("no start", &events);
    }

    #[test]
    #[should_panic(expected = "ends without ResponseStop")]
    fn a_missing_response_stop_fails() {
        let mut events = ordered();
        events.pop();
        assert_stream_event_order("no stop", &events);
    }

    #[test]
    #[should_panic(expected = "does not fit block 0")]
    fn a_delta_of_the_wrong_type_fails() {
        let mut events = ordered();
        events[2] = StreamEvent::tool_call_delta(0, "{}");
        assert_stream_event_order("wrong delta", &events);
    }

    #[test]
    #[should_panic(expected = "after its BlockStop")]
    fn a_delta_after_block_stop_fails() {
        let mut events = ordered();
        events.insert(4, StreamEvent::text_delta(0, "late"));
        assert_stream_event_order("late delta", &events);
    }
}
