use futures_util::{FutureExt, StreamExt, stream};
use serde_json::Value;

use super::*;

/// `(event, data, id)` of each event of `bytes` fed in one chunk.
fn decode(bytes: &[u8]) -> Vec<(String, String, Option<String>)> {
    decode_chunks(&[bytes])
}

/// `(event, data, id)` of each event, feeding `chunks` one by one.
fn decode_chunks(chunks: &[&[u8]]) -> Vec<(String, String, Option<String>)> {
    let mut decoder = SseDecoder::new();
    let mut events = Vec::new();
    for chunk in chunks {
        decoder.push(chunk);
        while let Some(event) = decoder.next_event().unwrap() {
            events.push((event.event, event.data, event.id));
        }
    }
    events
}

/// Feed `bytes` one byte at a time.
fn decode_bytewise(bytes: &[u8]) -> Vec<(String, String, Option<String>)> {
    let chunks: Vec<&[u8]> = bytes.chunks(1).collect();
    decode_chunks(&chunks)
}

fn message(data: &str) -> (String, String, Option<String>) {
    ("message".into(), data.into(), None)
}

fn named(event: &str, data: &str) -> (String, String, Option<String>) {
    (event.into(), data.into(), None)
}

#[test]
fn line_endings_are_lf_cr_or_crlf() {
    let expected = vec![named("delta", "{\"a\":1}"), message("[DONE]")];
    for newline in ["\n", "\r", "\r\n"] {
        let text = format!(
            "event: delta{newline}data: {{\"a\":1}}{newline}{newline}data: [DONE]{newline}{newline}"
        );
        assert_eq!(decode(text.as_bytes()), expected, "{newline:?}");
        assert_eq!(decode_bytewise(text.as_bytes()), expected, "{newline:?}");
    }
}

#[test]
fn a_crlf_split_across_chunks_is_one_line_ending() {
    // A decoder that ends the line at `CR` and reads the `LF` of the next
    // chunk as a blank line dispatches early.
    let events = decode_chunks(&[b"event: delta\r", b"\ndata: x\r", b"\n\r", b"\n"]);
    assert_eq!(events, vec![named("delta", "x")]);
}

#[test]
fn multi_line_data_is_joined_with_lf() {
    let text = "data: {\r\ndata:   \"a\": 1\r\ndata: }\r\n\r\n";
    assert_eq!(decode(text.as_bytes()), vec![message("{\n  \"a\": 1\n}")]);
    assert_eq!(
        decode_bytewise(text.as_bytes()),
        vec![message("{\n  \"a\": 1\n}")]
    );
}

#[test]
fn comments_and_retry_lines_are_skipped() {
    let text =
        ": keepalive\n\nretry: 3000\n\n:\nevent: delta\n: inside an event\nretry: 10\ndata: x\n\n";
    assert_eq!(decode(text.as_bytes()), vec![named("delta", "x")]);
}

#[test]
fn an_event_without_data_is_not_dispatched_but_empty_data_is() {
    let text = "event: ping\n\ndata:\n\ndata\n\n";
    assert_eq!(decode(text.as_bytes()), vec![message(""), message("")]);
}

#[test]
fn the_event_type_resets_after_each_event() {
    let text = "event: delta\ndata: a\n\ndata: b\n\nevent: first\nevent: second\ndata: c\n\n";
    assert_eq!(
        decode(text.as_bytes()),
        vec![named("delta", "a"), message("b"), named("second", "c")]
    );
}

#[test]
fn field_values_lose_one_leading_space_only() {
    let text = "data:x\n\ndata:  x\n\ndata: x \n\nunknown: y\ndata: a:b\n\n";
    assert_eq!(
        decode(text.as_bytes()),
        vec![message("x"), message(" x"), message("x "), message("a:b")]
    );
}

#[test]
fn the_last_event_id_carries_over() {
    let text = "id: 1\ndata: a\n\ndata: b\n\nid: 2\0\ndata: c\n\nid\ndata: d\n\n";
    let ids: Vec<_> = decode(text.as_bytes())
        .into_iter()
        .map(|(_, _, id)| id)
        .collect();
    // An id with a NUL is ignored; an empty one clears the id.
    assert_eq!(
        ids,
        [Some("1".into()), Some("1".into()), Some("1".into()), None]
    );
}

#[test]
fn a_leading_byte_order_mark_is_skipped() {
    let text = b"\xEF\xBB\xBFdata: a\n\n\xEF\xBB\xBFdata: b\n\n";
    // Only at the start: later, it is part of the field name.
    assert_eq!(decode(text), vec![message("a")]);
    assert_eq!(decode_bytewise(text), vec![message("a")]);
}

#[test]
fn utf8_split_across_chunks_is_reassembled() {
    let text = "data: héllo 🦀\n\n".as_bytes();
    assert_eq!(decode_bytewise(text), vec![message("héllo 🦀")]);
}

#[test]
fn an_event_without_its_blank_line_is_discarded() {
    let events: Vec<_> = SseStream::new(stream::iter([Ok::<_, std::io::Error>(
        "data: a\n\ndata: b\n",
    )]))
    .collect()
    .now_or_never()
    .unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].as_ref().unwrap().data, "a");
}

#[test]
fn invalid_utf8_fails_the_stream_once() {
    let mut decoder = SseDecoder::new();
    decoder.push(b"data: \xFF\n\ndata: ok\n\n");
    assert!(decoder.next_event().is_err());
    // The decoder itself can go on; the stream stops.
    assert_eq!(decoder.next_event().unwrap().unwrap().data, "ok");

    let events: Vec<_> = SseStream::new(stream::iter([
        Ok::<_, std::io::Error>(&b"data: a\n\n"[..]),
        Ok(&b"data: \xFF\n\ndata: b\n\n"[..]),
    ]))
    .collect()
    .now_or_never()
    .unwrap();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].as_ref().unwrap().data, "a");
    let error = events[1].as_ref().unwrap_err();
    assert!(matches!(error, SseError::InvalidUtf8(_)));
    assert!(
        error
            .to_string()
            .starts_with("invalid UTF-8 in the SSE stream")
    );
}

/// Scanning is linear: a long line fed in small chunks is scanned once,
/// not once per chunk.
#[test]
fn a_line_fed_in_many_chunks_is_scanned_once() {
    let text = format!("data: {}\n\n", "x".repeat(1 << 20));
    let chunks: Vec<&[u8]> = text.as_bytes().chunks(1024).collect();
    let mut decoder = SseDecoder::new();
    let mut events = Vec::new();
    for chunk in &chunks {
        decoder.push(chunk);
        while let Some(event) = decoder.next_event().unwrap() {
            events.push(event);
        }
    }
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].data.len(), 1 << 20);
    // Each byte once; at most one line ending per chunk twice (found by
    // `push` scanning back, then by `next_event`). A rescan of the pending
    // line per chunk would be about `len * chunks / 2`.
    assert!(
        decoder.examined <= text.len() + chunks.len(),
        "{} bytes examined for {} bytes",
        decoder.examined,
        text.len()
    );
}

#[test]
fn many_small_events_in_one_chunk_are_scanned_once() {
    let text = "event: delta\r\ndata: {\"n\":1}\r\n\r\n".repeat(1000);
    let mut decoder = SseDecoder::new();
    decoder.push(text.as_bytes());
    let mut count = 0;
    while decoder.next_event().unwrap().is_some() {
        count += 1;
    }
    assert_eq!(count, 1000);
    assert!(decoder.examined <= text.len() + 1, "{}", decoder.examined);
}

#[test]
fn a_line_waiting_for_its_ending_past_the_cap_fails() {
    let mut decoder = SseDecoder::new().max_event_size(16);
    decoder.push(b"data: 0123456789");
    assert_eq!(decoder.next_event(), Ok(None));
    // 17 bytes without a line ending: refused before it is buffered.
    decoder.push(b"a");
    assert_eq!(
        decoder.buffer.len(),
        16,
        "refused, never grown past the cap"
    );
    assert_eq!(
        decoder.next_event(),
        Err(SseError::EventTooLarge { limit: 16 })
    );
    // Decoding cannot go on: later bytes are ignored, the error stays.
    decoder.push(b"\n\ndata: ok\n\n");
    assert_eq!(
        decoder.next_event(),
        Err(SseError::EventTooLarge { limit: 16 })
    );
}

#[test]
fn a_line_of_exactly_the_cap_passes() {
    // `data: ` and ten bytes: 16 bytes, its line ending excluded.
    let events = {
        let mut decoder = SseDecoder::new().max_event_size(16);
        let mut events = Vec::new();
        for chunk in [&b"data: 01234"[..], b"56789", b"\n", b"\n"] {
            decoder.push(chunk);
            while let Some(event) = decoder.next_event().unwrap() {
                events.push(event.data);
            }
        }
        events
    };
    assert_eq!(events, ["0123456789"]);
}

#[test]
fn a_long_line_inside_one_chunk_fails_too() {
    // The same rule whatever the chunking: the line is complete, but past
    // the cap.
    let mut decoder = SseDecoder::new().max_event_size(16);
    decoder.push(b": a comment longer than sixteen bytes\ndata: x\n\n");
    assert_eq!(
        decoder.next_event(),
        Err(SseError::EventTooLarge { limit: 16 })
    );
}

#[test]
fn data_accumulated_past_the_cap_fails() {
    // Every line fits; the joined data does not.
    let mut decoder = SseDecoder::new().max_event_size(16);
    decoder.push(b"data: 01234567\ndata: 0123456\n");
    assert_eq!(decoder.next_event(), Ok(None), "15 bytes of data: fits");
    decoder.push(b"data: 0\n");
    assert_eq!(
        decoder.next_event(),
        Err(SseError::EventTooLarge { limit: 16 })
    );
    assert!(decoder.event.data.capacity() <= 17);
}

#[test]
fn the_cap_bounds_the_buffer_while_it_grows() {
    let limit = 1000;
    let mut decoder = SseDecoder::new().max_event_size(limit);
    decoder.push(b"data: ");
    for _ in 0..(limit - 6) / 7 {
        decoder.push(b"0123456");
        assert_eq!(decoder.next_event(), Ok(None));
        // Doubling stops at the cap.
        assert!(
            decoder.buffer.capacity() <= limit,
            "{}",
            decoder.buffer.capacity()
        );
    }
}

#[test]
fn small_events_in_a_chunk_larger_than_the_cap_pass() {
    // The cap bounds lines and events, not chunks.
    let text = "data: 0123456789\n\n".repeat(100);
    let mut decoder = SseDecoder::new().max_event_size(16);
    decoder.push(text.as_bytes());
    let mut count = 0;
    while decoder.next_event().unwrap().is_some() {
        count += 1;
    }
    assert_eq!(count, 100);
}

#[test]
fn an_event_past_the_cap_ends_the_stream() {
    let events: Vec<_> = SseStream::new(stream::iter([
        Ok::<_, std::io::Error>(&b"data: a\n\ndata: 0123"[..]),
        Ok(&b"456789"[..]),
        Ok(&b"\n\ndata: b\n\n"[..]),
    ]))
    .max_event_size(8)
    .collect()
    .now_or_never()
    .unwrap();
    assert_eq!(events.len(), 2, "{events:?}");
    assert_eq!(events[0].as_ref().unwrap().data, "a");
    let error = events[1].as_ref().unwrap_err();
    assert!(matches!(error, SseError::EventTooLarge { limit: 8 }));
    assert_eq!(
        error.to_string(),
        "an SSE event passed the size cap of 8 bytes"
    );
    assert!(std::error::Error::source(error).is_none());
}

#[test]
fn the_default_cap_is_64_mib() {
    assert_eq!(SseDecoder::DEFAULT_MAX_EVENT_SIZE, 64 * 1024 * 1024);
    assert_eq!(
        SseDecoder::new().max_event_size,
        SseDecoder::DEFAULT_MAX_EVENT_SIZE
    );
    assert_eq!(
        SseDecoder::default().max_event_size,
        SseDecoder::DEFAULT_MAX_EVENT_SIZE
    );
}

#[test]
fn a_transport_error_ends_the_stream_after_the_buffered_events() {
    let events: Vec<_> = SseStream::new(stream::iter([
        Ok(&b"data: a\n\ndata: b\n\ndata: c"[..]),
        Err(crate::TransportError::new(
            crate::TransportErrorKind::Timeout,
            "stream idle",
        )),
        Ok(&b"\n\n"[..]),
    ]))
    .collect()
    .now_or_never()
    .unwrap();
    assert_eq!(events.len(), 3);
    assert_eq!(events[1].as_ref().unwrap().data, "b");
    let SseError::Transport(error) = events[2].as_ref().unwrap_err() else {
        panic!("{events:?}")
    };
    assert_eq!(error.kind(), crate::TransportErrorKind::Timeout);
    // Transparent: the body error's own message.
    assert_eq!(
        events[2].as_ref().unwrap_err().to_string(),
        error.to_string()
    );
}

/// The root directory of every crate in the workspace, from `cargo
/// metadata`, so the fixtures are found wherever the crates live; or `None`
/// when this crate has no sibling crates (a packaged `allama-http` is tested
/// on its own).
#[cfg(not(target_arch = "wasm32"))]
fn workspace_crates() -> Option<Vec<std::path::PathBuf>> {
    let output = std::process::Command::new(env!("CARGO"))
        .args([
            "metadata",
            "--no-deps",
            "--offline",
            "--format-version",
            "1",
        ])
        .args([
            "--manifest-path",
            concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"),
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let metadata: Value = serde_json::from_slice(&output.stdout).ok()?;
    let roots: Vec<std::path::PathBuf> = metadata["packages"]
        .as_array()?
        .iter()
        .filter_map(|package| package["manifest_path"].as_str())
        .filter_map(|manifest| std::path::Path::new(manifest).parent())
        .map(std::path::Path::to_path_buf)
        .collect();
    (roots.len() > 1).then_some(roots)
}

/// The SSE fixtures of every provider crate, as `(path, contents)`, or
/// `None` outside the workspace (a packaged `allama-http` has no sibling
/// crates).
#[cfg(not(target_arch = "wasm32"))]
fn sse_fixtures() -> Option<Vec<(std::path::PathBuf, String)>> {
    fn walk(dir: &std::path::Path, out: &mut Vec<(std::path::PathBuf, String)>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "sse") {
                let text = std::fs::read_to_string(&path).unwrap();
                out.push((path, text));
            }
        }
    }
    let Some(crates) = workspace_crates() else {
        eprintln!("skipped: the workspace SSE fixtures are not available");
        return None;
    };
    let mut fixtures = Vec::new();
    for root in &crates {
        walk(&root.join("fixtures"), &mut fixtures);
    }
    fixtures.sort();
    Some(fixtures)
}

/// A whole-text reading of an SSE body: an oracle for the incremental
/// decoder.
#[cfg(not(target_arch = "wasm32"))]
fn oracle(text: &str) -> Vec<(String, String, Option<String>)> {
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut events = Vec::new();
    // Text after the last blank line is an unterminated event.
    let complete = text.rfind("\n\n").map_or("", |end| &text[..end]);
    for block in complete.split("\n\n") {
        let mut event = String::new();
        let mut data: Vec<&str> = Vec::new();
        for line in block.lines().filter(|line| !line.starts_with(':')) {
            let (field, value) = line.split_once(':').unwrap_or((line, ""));
            let value = value.strip_prefix(' ').unwrap_or(value);
            match field {
                "event" => event = value.to_owned(),
                "data" => data.push(value),
                "id" => panic!("the fixtures set no ids"),
                _ => {}
            }
        }
        if !data.is_empty() {
            let event = if event.is_empty() { "message" } else { &event };
            events.push((event.to_owned(), data.join("\n"), None));
        }
    }
    events
}

/// JSON text split after each comma outside a string: joined with `\n`,
/// the same JSON value.
#[cfg(not(target_arch = "wasm32"))]
fn split_after_commas(json: &str) -> Vec<&str> {
    let (mut parts, mut start) = (Vec::new(), 0);
    let (mut in_string, mut escaped) = (false, false);
    for (i, c) in json.char_indices() {
        match c {
            _ if escaped => escaped = false,
            '\\' if in_string => escaped = true,
            '"' => in_string = !in_string,
            ',' if !in_string => {
                parts.push(&json[start..=i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&json[start..]);
    parts
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn provider_sse_fixtures_decode_the_same_under_any_framing() {
    let Some(fixtures) = sse_fixtures() else {
        return;
    };
    assert!(fixtures.len() >= 20, "found {} fixtures", fixtures.len());
    for (path, text) in fixtures {
        // Truncated fixtures may end in an unterminated event, or be one.
        let expected = oracle(&text);
        let label = path.display();
        assert_eq!(decode(text.as_bytes()), expected, "{label}");
        assert_eq!(
            decode_bytewise(text.as_bytes()),
            expected,
            "{label}: bytewise"
        );

        let lf = text.replace("\r\n", "\n");
        let crlf = lf.replace('\n', "\r\n");
        assert_eq!(decode(crlf.as_bytes()), expected, "{label}: CRLF");
        assert_eq!(
            decode_bytewise(crlf.as_bytes()),
            expected,
            "{label}: CRLF bytewise"
        );
        let cr = lf.replace('\n', "\r");
        assert_eq!(decode(cr.as_bytes()), expected, "{label}: CR");

        // Each JSON data line split over several `data:` lines decodes to the
        // same JSON.
        let split: String = lf
            .lines()
            .map(|line| match line.strip_prefix("data: ") {
                Some(data) if serde_json::from_str::<Value>(data).is_ok() => {
                    split_after_commas(data)
                        .into_iter()
                        .map(|part| format!("data: {part}\n"))
                        .collect()
                }
                _ => format!("{line}\n"),
            })
            .collect();
        let decoded = decode(split.as_bytes());
        assert_eq!(decoded.len(), expected.len(), "{label}: multi-line data");
        for ((event, data, _), (expected_event, expected_data, _)) in decoded.iter().zip(&expected)
        {
            assert_eq!(event, expected_event, "{label}: multi-line data");
            match serde_json::from_str::<Value>(expected_data) {
                Ok(json) => assert_eq!(
                    serde_json::from_str::<Value>(data).ok(),
                    Some(json),
                    "{label}: multi-line data"
                ),
                Err(_) => assert_eq!(data, expected_data, "{label}: multi-line data"),
            }
        }
    }
}

#[test]
fn a_constructed_event_equals_the_decoded_one() {
    let mut decoder = SseDecoder::new();
    decoder.push(b"event: delta\ndata: {\"a\":1}\n\n");
    let decoded = decoder.next_event().unwrap().unwrap();

    assert_eq!(decoded, SseEvent::new("delta", "{\"a\":1}"));
    assert_eq!(SseEvent::new("delta", String::new()).id, None);
}

#[test]
fn a_first_chunk_past_the_cap_fails() {
    let mut decoder = SseDecoder::new().max_event_size(8);
    decoder.push(b"data: 0123456789");
    assert_eq!(
        decoder.next_event(),
        Err(SseError::EventTooLarge { limit: 8 })
    );
    let body = stream::iter([Ok::<_, std::convert::Infallible>(&b"data: 0123456789"[..])]);
    let events: Vec<_> = SseStream::new(body)
        .max_event_size(8)
        .collect()
        .now_or_never()
        .unwrap();
    assert_eq!(events, [Err(SseError::EventTooLarge { limit: 8 })]);
}
