use futures_util::{FutureExt, StreamExt, stream};

use super::*;
use crate::{TransportError, TransportErrorKind};

/// The lines of `bytes` fed in one chunk, the body then ending.
fn decode(bytes: &[u8]) -> Vec<String> {
    decode_chunks(&[bytes])
}

/// The lines of `chunks` fed in order, the body then ending.
fn decode_chunks(chunks: &[&[u8]]) -> Vec<String> {
    let body = stream::iter(chunks.iter().map(|chunk| Ok::<_, Infallible>(*chunk)));
    NdjsonStream::new(body)
        .map(Result::unwrap)
        .collect()
        .now_or_never()
        .unwrap()
}

/// The lines of `bytes` fed in chunks of `size` bytes.
fn decode_every(bytes: &[u8], size: usize) -> Vec<String> {
    let chunks: Vec<&[u8]> = bytes.chunks(size).collect();
    decode_chunks(&chunks)
}

/// What every chunking must decode to: the non-blank lines, `CR`s of
/// `CRLF` endings stripped, with no BOM.
fn oracle(text: &str) -> Vec<String> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    text.split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .filter(|line| !line.trim_matches([' ', '\t', '\r']).is_empty())
        .map(str::to_owned)
        .collect()
}

#[test]
fn lines_end_with_lf_or_crlf() {
    assert_eq!(
        decode(b"{\"a\":1}\n{\"b\":2}\r\n"),
        ["{\"a\":1}", "{\"b\":2}"]
    );
}

#[test]
fn a_crlf_split_across_chunks_is_one_line_ending() {
    assert_eq!(
        decode_chunks(&[b"{\"a\":1}\r", b"\n{\"b\":2}\r", b"\n"]),
        ["{\"a\":1}", "{\"b\":2}"]
    );
}

#[test]
fn a_lone_cr_is_not_a_line_ending() {
    assert_eq!(decode(b"{\"a\":\r1}\n"), ["{\"a\":\r1}"]);
}

#[test]
fn blank_lines_are_skipped() {
    assert_eq!(
        decode(b"\n\r\n{\"a\":1}\n  \t\n\n{\"b\":2}\n\r\n"),
        ["{\"a\":1}", "{\"b\":2}"]
    );
}

#[test]
fn a_last_line_without_its_ending_is_a_line() {
    assert_eq!(decode(b"{\"a\":1}\n{\"b\":2}"), ["{\"a\":1}", "{\"b\":2}"]);
    assert_eq!(
        decode(b"{\"a\":1}\n{\"b\":2}\r"),
        ["{\"a\":1}", "{\"b\":2}"]
    );
    // A cut-off line is yielded as it is: the adapter's JSON rejects it.
    assert_eq!(decode(b"{\"a\":1}\n{\"b\":"), ["{\"a\":1}", "{\"b\":"]);
}

#[test]
fn a_blank_last_line_is_nothing() {
    assert_eq!(decode(b"{\"a\":1}\n  "), ["{\"a\":1}"]);
    assert!(decode(b"").is_empty());
}

#[test]
fn a_leading_byte_order_mark_is_skipped() {
    assert_eq!(decode(b"\xEF\xBB\xBF{\"a\":1}\n"), ["{\"a\":1}"]);
    assert_eq!(
        decode_chunks(&[b"\xEF", b"\xBB", b"\xBF{\"a\":1}"]),
        ["{\"a\":1}"]
    );
    // Only at the start of the stream.
    let mut decoder = NdjsonDecoder::new();
    decoder.push(b"1\n\xEF\xBB\xBF2\n");
    assert_eq!(decoder.next_line(), Ok(Some("1".to_owned())));
    assert_eq!(decoder.next_line(), Ok(Some("\u{feff}2".to_owned())));
}

#[test]
fn utf8_split_across_chunks_is_reassembled() {
    let text = "{\"text\":\"héllo wörld 👋\"}\n";
    for size in 1..=text.len() {
        assert_eq!(
            decode_every(text.as_bytes(), size),
            [text.trim_end()],
            "{size}"
        );
    }
}

#[test]
fn every_chunking_down_to_one_byte_decodes_the_same() {
    let corpus = [
        "{\"a\":1}\n{\"b\":[1,2]}\n",
        "\u{feff}{\"a\":1}\r\n\r\n{\"b\":\"x\\ny\"}\r\n",
        "\n\n  \n{\"done\":true}",
        "{\"a\":\"\r\"}\r\n{\"b\":2}\r",
        "{\"t\":\"日本語\"}\n\t\n{\"t\":\"ok\"}\n",
    ];
    for text in corpus {
        let expected = oracle(text);
        for size in 1..=text.len() {
            assert_eq!(
                decode_every(text.as_bytes(), size),
                expected,
                "{text:?} in chunks of {size}"
            );
        }
    }
}

#[test]
fn invalid_utf8_fails_the_stream_once() {
    let body = stream::iter([
        Ok::<_, Infallible>(&b"{\"a\":1}\n"[..]),
        Ok(&b"{\"a\":\"\xFF\"}\n{\"b\":2}\n"[..]),
    ]);
    let items: Vec<_> = NdjsonStream::new(body).collect().now_or_never().unwrap();
    assert_eq!(items.len(), 2, "{items:?}");
    assert_eq!(items[0], Ok("{\"a\":1}".to_owned()));
    assert!(matches!(items[1], Err(NdjsonError::InvalidUtf8(_))));

    // The decoder skips the line and can go on.
    let mut decoder = NdjsonDecoder::new();
    decoder.push(b"\xFF\n{\"b\":2}\n");
    assert!(matches!(
        decoder.next_line(),
        Err(NdjsonError::InvalidUtf8(_))
    ));
    assert_eq!(decoder.next_line(), Ok(Some("{\"b\":2}".to_owned())));

    // An invalid last line without its ending fails too.
    let mut decoder = NdjsonDecoder::new();
    decoder.push(b"\xFF");
    assert_eq!(decoder.next_line(), Ok(None));
    assert!(matches!(decoder.finish(), Err(NdjsonError::InvalidUtf8(_))));
}

#[test]
fn a_line_of_exactly_the_cap_passes() {
    let mut decoder = NdjsonDecoder::new().max_line_size(8);
    decoder.push(b"12345678\r");
    assert_eq!(decoder.next_line(), Ok(None));
    decoder.push(b"\n");
    assert_eq!(decoder.next_line(), Ok(Some("12345678".to_owned())));

    let mut decoder = NdjsonDecoder::new().max_line_size(8);
    decoder.push(b"12345678");
    assert_eq!(decoder.next_line(), Ok(None));
    assert_eq!(decoder.finish(), Ok(Some("12345678".to_owned())));
}

#[test]
fn a_line_waiting_for_its_ending_past_the_cap_fails() {
    let mut decoder = NdjsonDecoder::new().max_line_size(8);
    decoder.push(b"{\"a\":1}\n1234");
    decoder.push(b"5678");
    decoder.push(b"9");
    // The complete line first, then the failure, for good.
    assert_eq!(decoder.next_line(), Ok(Some("{\"a\":1}".to_owned())));
    assert_eq!(
        decoder.next_line(),
        Err(NdjsonError::LineTooLarge { limit: 8 })
    );
    decoder.push(b"\n{}\n");
    assert_eq!(
        decoder.next_line(),
        Err(NdjsonError::LineTooLarge { limit: 8 })
    );
    assert_eq!(
        decoder.finish(),
        Err(NdjsonError::LineTooLarge { limit: 8 })
    );
}

#[test]
fn the_cap_bounds_the_buffer_while_it_grows() {
    let mut decoder = NdjsonDecoder::new().max_line_size(8);
    for _ in 0..1000 {
        decoder.push(b"0123456789");
    }
    assert!(decoder.buffer.len() <= 10, "{}", decoder.buffer.len());
    assert_eq!(
        decoder.next_line(),
        Err(NdjsonError::LineTooLarge { limit: 8 })
    );
}

#[test]
fn a_long_line_inside_one_chunk_fails_too() {
    let mut decoder = NdjsonDecoder::new().max_line_size(8);
    decoder.push(b"123456789\n{}\n");
    assert_eq!(
        decoder.next_line(),
        Err(NdjsonError::LineTooLarge { limit: 8 })
    );
}

#[test]
fn an_unterminated_last_line_past_the_cap_fails() {
    let mut decoder = NdjsonDecoder::new().max_line_size(8);
    decoder.push(b"123456789");
    assert_eq!(
        decoder.next_line(),
        Err(NdjsonError::LineTooLarge { limit: 8 })
    );
}

#[test]
fn small_lines_in_a_chunk_larger_than_the_cap_pass() {
    let mut decoder = NdjsonDecoder::new().max_line_size(4);
    decoder.push(b"{}\n[]\n1\n\"ab\"\n");
    let mut lines = Vec::new();
    while let Some(line) = decoder.next_line().unwrap() {
        lines.push(line);
    }
    assert_eq!(lines, ["{}", "[]", "1", "\"ab\""]);
}

#[test]
fn a_line_past_the_cap_ends_the_stream() {
    let body = stream::iter([
        Ok::<_, Infallible>(&b"{}\n"[..]),
        Ok(&b"123456789"[..]),
        Ok(&b"\n{}\n"[..]),
    ]);
    let items: Vec<_> = NdjsonStream::new(body)
        .max_line_size(8)
        .collect()
        .now_or_never()
        .unwrap();
    assert_eq!(
        items,
        [
            Ok("{}".to_owned()),
            Err(NdjsonError::LineTooLarge { limit: 8 })
        ]
    );
}

#[test]
fn the_default_cap_is_64_mib() {
    assert_eq!(NdjsonDecoder::DEFAULT_MAX_LINE_SIZE, 64 * 1024 * 1024);
    let mut decoder = NdjsonDecoder::new();
    let line = vec![b'1'; 1024 * 1024];
    decoder.push(&line);
    decoder.push(b"\n");
    assert_eq!(
        decoder.next_line().unwrap().map(|line| line.len()),
        Some(1024 * 1024)
    );
}

#[test]
fn a_line_fed_in_many_chunks_is_scanned_once() {
    let mut decoder = NdjsonDecoder::new();
    let chunks = 2000;
    for _ in 0..chunks {
        decoder.push(b"0123456789");
        assert_eq!(decoder.next_line(), Ok(None));
    }
    decoder.push(b"\n");
    assert_eq!(
        decoder.next_line().unwrap().map(|line| line.len()),
        Some(chunks * 10)
    );
    let total = chunks * 10 + 1;
    assert!(
        decoder.examined <= 2 * total,
        "examined {} bytes of {total}",
        decoder.examined
    );
}

#[test]
fn many_small_lines_in_one_chunk_are_scanned_once() {
    let body = "{\"n\":1}\n".repeat(5000);
    let mut decoder = NdjsonDecoder::new();
    decoder.push(body.as_bytes());
    let mut count = 0;
    while decoder.next_line().unwrap().is_some() {
        count += 1;
    }
    assert_eq!(count, 5000);
    assert!(
        decoder.examined <= 2 * body.len(),
        "examined {} bytes of {}",
        decoder.examined,
        body.len()
    );
}

#[test]
fn a_transport_error_ends_the_stream_after_the_buffered_lines() {
    let body = stream::iter([
        Ok(&b"{\"a\":1}\n{\"b\":2}\n{\"c\":"[..]),
        Err(TransportError::new(TransportErrorKind::Timeout, "idle")),
        Ok(&b"3}\n"[..]),
    ]);
    let items: Vec<_> = NdjsonStream::new(body).collect().now_or_never().unwrap();
    assert_eq!(items.len(), 3, "{items:?}");
    let NdjsonError::Transport(failure) = items[2].as_ref().unwrap_err() else {
        panic!("{items:?}")
    };
    assert_eq!(failure.kind(), TransportErrorKind::Timeout);
    // Transparent: the body error's own message.
    assert_eq!(
        items[2].as_ref().unwrap_err().to_string(),
        failure.to_string()
    );
}

#[test]
fn failures_map_to_stream_errors() {
    use allama::{Error, StreamErrorKind};

    let timeout = NdjsonError::Transport(TransportError::new(TransportErrorKind::Timeout, "idle"));
    assert!(matches!(timeout.into_error(), Error::Timeout { .. }));

    let error: Error = NdjsonError::<TransportError>::LineTooLarge { limit: 8 }.into();
    assert!(matches!(
        error,
        Error::Stream {
            kind: StreamErrorKind::Protocol,
            ..
        }
    ));
    assert!(!error.is_retryable());
    assert_eq!(error.message_origin(), allama::MessageOrigin::Authored);
    assert_eq!(
        error.message(),
        "an NDJSON line passed the size cap of 8 bytes"
    );
}
