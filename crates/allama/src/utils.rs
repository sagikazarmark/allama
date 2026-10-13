use std::fmt;

/// Maximum bytes shown by `Display` impls for message content.
pub(crate) const DISPLAY_MAX_BYTES: usize = 50;

pub(crate) fn truncate_display(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_owned();
    }

    let end = s.floor_char_boundary(max_bytes);
    format!("{}...", &s[..end])
}

pub(crate) fn write_truncated(
    f: &mut fmt::Formatter<'_>,
    s: &str,
    max_bytes: usize,
) -> fmt::Result {
    if s.len() <= max_bytes {
        return f.write_str(s);
    }

    let end = s.floor_char_boundary(max_bytes);
    write!(f, "{}...", &s[..end])
}

pub(crate) fn write_truncated_joined<'a>(
    f: &mut fmt::Formatter<'_>,
    mut texts: impl Iterator<Item = &'a str>,
    empty: &str,
    max_bytes: usize,
) -> fmt::Result {
    let Some(first) = texts.next() else {
        return f.write_str(empty);
    };

    let mut remaining = max_bytes;
    let mut truncated = write_limited_fragment(f, first, &mut remaining)?;

    if !truncated {
        for text in texts {
            if write_limited_fragment(f, " ", &mut remaining)?
                || write_limited_fragment(f, text, &mut remaining)?
            {
                truncated = true;
                break;
            }
        }
    }

    if truncated {
        f.write_str("...")?;
    }

    Ok(())
}

pub(crate) fn concat_segments<'a>(mut segments: impl Iterator<Item = &'a str>) -> Option<String> {
    let first = segments.next()?;
    let mut joined = String::from(first);
    for segment in segments {
        joined.push_str(segment);
    }
    Some(joined)
}

fn write_limited_fragment(
    f: &mut fmt::Formatter<'_>,
    fragment: &str,
    remaining: &mut usize,
) -> Result<bool, fmt::Error> {
    if *remaining == 0 {
        return Ok(true);
    }

    if fragment.len() <= *remaining {
        f.write_str(fragment)?;
        *remaining -= fragment.len();
        return Ok(false);
    }

    let end = fragment.floor_char_boundary(*remaining);
    f.write_str(&fragment[..end])?;
    *remaining = 0;
    Ok(true)
}
