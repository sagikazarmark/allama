//! Formatting for failure reports.

/// The shortest run of base64 text [`elide_payloads`] shortens: long enough
/// that no id, hash or word in a report reaches it, short enough to catch
/// the probes' media payloads.
const MIN_PAYLOAD_CHARS: usize = 100;

/// `text` with every long run of base64 text (an image or file payload,
/// alone or in a `data:` URL) replaced by `<base64: N chars>`, so a failure
/// report that shows a request or an event stays readable.
pub(crate) fn elide_payloads(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut run = String::new();
    for c in text.chars() {
        if c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '=') {
            run.push(c);
        } else {
            flush(&mut out, &mut run);
            out.push(c);
        }
    }
    flush(&mut out, &mut run);
    out
}

fn flush(out: &mut String, run: &mut String) {
    if run.len() >= MIN_PAYLOAD_CHARS {
        out.push_str("<base64: ");
        out.push_str(&run.len().to_string());
        out.push_str(" chars>");
    } else {
        out.push_str(run);
    }
    run.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::e2e::RED_PNG_BASE64;

    #[test]
    fn long_base64_is_elided() {
        let event = format!(r#"{{"type":"image","data":"{RED_PNG_BASE64}"}}"#);
        assert_eq!(
            elide_payloads(&event),
            format!(
                r#"{{"type":"image","data":"<base64: {} chars>"}}"#,
                RED_PNG_BASE64.len()
            )
        );

        let url = format!("data:image/png;base64,{RED_PNG_BASE64}");
        assert_eq!(
            elide_payloads(&url),
            format!(
                "data:image/png;base64,<base64: {} chars>",
                RED_PNG_BASE64.len()
            )
        );
    }

    #[test]
    fn short_runs_and_prose_are_kept() {
        let text = "chat `gpt-4o`: field `response_format` (call_honesty, \
                    9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08)";
        assert_eq!(elide_payloads(text), text);
        assert_eq!(elide_payloads(""), "");
    }
}
