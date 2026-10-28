//! Value types of extraction answers: calendar dates, times and datetimes
//! as RFC 3339 text, and spans into the request's input.

use std::fmt;
use std::ops::Range;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{Error, Result};

/// A calendar date in RFC 3339 form, `YYYY-MM-DD`.
///
/// Validated on construction, so a `Date` always names a real day. It holds
/// text, not a date-time library's type: convert with your own crate.
///
/// ```
/// use allama::extraction::Date;
///
/// let date: Date = "2026-09-28".parse()?;
/// assert_eq!(date.as_str(), "2026-09-28");
/// assert!("2026-02-30".parse::<Date>().is_err());
/// # Ok::<(), allama::Error>(())
/// ```
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Date(String);

/// A time of day in RFC 3339 form: `HH:MM`, or `HH:MM:SS` with optional
/// fractional seconds, with an optional `Z` or `±HH:MM` offset.
///
/// ```
/// use allama::extraction::Time;
///
/// let time: Time = "13:34:00".parse()?;
/// assert_eq!(time.as_str(), "13:34:00");
/// assert!("25:00".parse::<Time>().is_err());
/// # Ok::<(), allama::Error>(())
/// ```
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Time(String);

/// A date and time in RFC 3339 form: a [`Date`], `T` (or a space), and a
/// [`Time`], such as `2026-07-01T13:34:00` or `2026-07-01T13:34:00Z`.
///
/// ```
/// use allama::extraction::DateTime;
///
/// let at: DateTime = "2026-07-01T13:34:00".parse()?;
/// assert_eq!(at.date().as_str(), "2026-07-01");
/// assert_eq!(at.time().as_str(), "13:34:00");
/// # Ok::<(), allama::Error>(())
/// ```
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DateTime(String);

/// Where a text answer was copied from: a byte range into
/// [`ExtractionRequest::input`](crate::ExtractionRequest::input), end
/// exclusive, on `char` boundaries.
///
/// ```
/// use allama::extraction::TextSpan;
///
/// let input = "Order A-88412 arrived.";
/// let span = TextSpan::new(6, 13);
/// assert_eq!(span.text(input), Some("A-88412"));
/// assert_eq!(span.range(), 6..13);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TextSpan {
    /// Byte offset of the first byte.
    pub start: usize,
    /// Byte offset one past the last byte.
    pub end: usize,
}

impl Date {
    /// Parse `YYYY-MM-DD`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidRequest`] when the text is not a calendar date.
    pub fn new(text: impl Into<String>) -> Result<Self> {
        let text = text.into();
        parse_date(&text).map_err(|problem| invalid("date", &text, problem))?;
        Ok(Self(text))
    }

    /// The date as written.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The year.
    #[must_use]
    pub fn year(&self) -> i32 {
        parse_date(&self.0).map_or(0, |(year, _, _)| year)
    }

    /// The month, 1 to 12.
    #[must_use]
    pub fn month(&self) -> u8 {
        parse_date(&self.0).map_or(0, |(_, month, _)| month)
    }

    /// The day of the month, 1 to 31.
    #[must_use]
    pub fn day(&self) -> u8 {
        parse_date(&self.0).map_or(0, |(_, _, day)| day)
    }
}

impl Time {
    /// Parse `HH:MM`, `HH:MM:SS` or `HH:MM:SS.fff`, with an optional offset.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidRequest`] when the text is not a time of day.
    pub fn new(text: impl Into<String>) -> Result<Self> {
        let text = text.into();
        parse_time(&text).map_err(|problem| invalid("time", &text, problem))?;
        Ok(Self(text))
    }

    /// The time as written.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl DateTime {
    /// Parse a date, `T` or a space, and a time.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidRequest`] when the text is not a date and time.
    pub fn new(text: impl Into<String>) -> Result<Self> {
        let text = text.into();
        split_datetime(&text).map_err(|problem| invalid("datetime", &text, problem))?;
        Ok(Self(text))
    }

    /// The datetime as written.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The date part.
    #[must_use]
    pub fn date(&self) -> Date {
        Date(self.0[..DATE_LEN].to_owned())
    }

    /// The time part, with its offset if any.
    #[must_use]
    pub fn time(&self) -> Time {
        Time(self.0[DATE_LEN + 1..].to_owned())
    }
}

impl TextSpan {
    /// A span from byte `start` to byte `end`, exclusive.
    #[must_use]
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// The span as a range, for slicing.
    #[must_use]
    pub const fn range(&self) -> Range<usize> {
        self.start..self.end
    }

    /// The text the span covers in `input`, or `None` when the span is out
    /// of bounds or not on `char` boundaries. A span from a validated
    /// response always covers its request's input.
    #[must_use]
    pub fn text<'a>(&self, input: &'a str) -> Option<&'a str> {
        input.get(self.range())
    }

    /// Whether the span covers no bytes.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

macro_rules! text_newtype {
    ($ty:ident) => {
        impl fmt::Debug for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_tuple(stringify!($ty)).field(&self.0).finish()
            }
        }

        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl FromStr for $ty {
            type Err = Error;

            fn from_str(text: &str) -> Result<Self> {
                Self::new(text)
            }
        }

        impl TryFrom<String> for $ty {
            type Error = Error;

            fn try_from(text: String) -> Result<Self> {
                Self::new(text)
            }
        }

        impl TryFrom<&str> for $ty {
            type Error = Error;

            fn try_from(text: &str) -> Result<Self> {
                Self::new(text)
            }
        }

        impl AsRef<str> for $ty {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }

        impl Serialize for $ty {
            fn serialize<S: Serializer>(
                &self,
                serializer: S,
            ) -> std::result::Result<S::Ok, S::Error> {
                serializer.serialize_str(&self.0)
            }
        }

        impl<'de> Deserialize<'de> for $ty {
            fn deserialize<D: Deserializer<'de>>(
                deserializer: D,
            ) -> std::result::Result<Self, D::Error> {
                let text = String::deserialize(deserializer)?;
                Self::new(text).map_err(serde::de::Error::custom)
            }
        }
    };
}

text_newtype!(Date);
text_newtype!(Time);
text_newtype!(DateTime);

const DATE_LEN: usize = "YYYY-MM-DD".len();

fn invalid(what: &str, text: &str, problem: &'static str) -> Error {
    Error::invalid_request(format!("{text:?} is not a {what}: {problem}")).authored()
}

/// `(year, month, day)` of a `YYYY-MM-DD` date.
fn parse_date(text: &str) -> std::result::Result<(i32, u8, u8), &'static str> {
    let bytes = text.as_bytes();
    if bytes.len() != DATE_LEN || bytes[4] != b'-' || bytes[7] != b'-' {
        return Err("expected YYYY-MM-DD");
    }
    let year = digits(&text[0..4]).ok_or("the year must be four digits")?;
    let month = digits(&text[5..7]).ok_or("the month must be two digits")?;
    let day = digits(&text[8..10]).ok_or("the day must be two digits")?;
    if !(1..=12).contains(&month) {
        return Err("the month must be 01 to 12");
    }
    // Four, two and two digits fit; the conversions cannot fail.
    let year = i32::try_from(year).unwrap_or(i32::MAX);
    let month = u8::try_from(month).unwrap_or(u8::MAX);
    let day = u8::try_from(day).unwrap_or(u8::MAX);
    if day == 0 || day > days_in_month(year, month) {
        return Err("the day is not in the month");
    }
    Ok((year, month, day))
}

fn days_in_month(year: i32, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        _ => 28,
    }
}

/// Check `HH:MM[:SS[.fff]][Z|±HH:MM]`.
fn parse_time(text: &str) -> std::result::Result<(), &'static str> {
    let (clock, offset) = split_offset(text);
    let mut parts = clock.splitn(3, ':');
    let hour = parts
        .next()
        .filter(|part| part.len() == 2)
        .and_then(digits)
        .ok_or("expected HH:MM")?;
    let minute = parts
        .next()
        .filter(|part| part.len() == 2)
        .and_then(digits)
        .ok_or("expected HH:MM")?;
    if hour > 23 || minute > 59 {
        return Err("the hour must be 00 to 23 and the minute 00 to 59");
    }
    if let Some(seconds) = parts.next() {
        let (whole, fraction) = seconds.split_once('.').unwrap_or((seconds, ""));
        let second = Some(whole)
            .filter(|part| part.len() == 2)
            .and_then(digits)
            .ok_or("the seconds must be two digits")?;
        // 60 allows a leap second, as RFC 3339 does.
        if second > 60 {
            return Err("the second must be 00 to 60");
        }
        if seconds.contains('.') && (fraction.is_empty() || digits(fraction).is_none()) {
            return Err("fractional seconds must be digits");
        }
    }
    if let Some(offset) = offset {
        check_offset(offset)?;
    }
    Ok(())
}

/// Split a trailing `Z` or `±HH:MM` offset off a time.
fn split_offset(text: &str) -> (&str, Option<&str>) {
    if let Some(clock) = text.strip_suffix('Z').or_else(|| text.strip_suffix('z')) {
        return (clock, Some("Z"));
    }
    // The offset's sign is the last `+` or `-`; a time has no other.
    match text.rfind(['+', '-']) {
        Some(at) => (&text[..at], Some(&text[at..])),
        None => (text, None),
    }
}

fn check_offset(offset: &str) -> std::result::Result<(), &'static str> {
    if offset == "Z" {
        return Ok(());
    }
    let digits_part = &offset[1..];
    let (hours, minutes) = digits_part
        .split_once(':')
        .ok_or("expected a Z or ±HH:MM offset")?;
    let hours = Some(hours)
        .filter(|part| part.len() == 2)
        .and_then(digits)
        .ok_or("expected a Z or ±HH:MM offset")?;
    let minutes = Some(minutes)
        .filter(|part| part.len() == 2)
        .and_then(digits)
        .ok_or("expected a Z or ±HH:MM offset")?;
    if hours > 23 || minutes > 59 {
        return Err("the offset must be within ±23:59");
    }
    Ok(())
}

fn split_datetime(text: &str) -> std::result::Result<(), &'static str> {
    if text.len() < DATE_LEN + 1 {
        return Err("expected a date, T, and a time");
    }
    if !text.is_char_boundary(DATE_LEN) || !text.is_char_boundary(DATE_LEN + 1) {
        return Err("expected a date, T, and a time");
    }
    parse_date(&text[..DATE_LEN])?;
    match &text[DATE_LEN..DATE_LEN + 1] {
        "T" | "t" | " " => {}
        _ => return Err("expected T between the date and the time"),
    }
    parse_time(&text[DATE_LEN + 1..])
}

/// The value of a run of ASCII digits, `None` when empty or not digits.
fn digits(text: &str) -> Option<u32> {
    if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_are_calendar_days() {
        let date = Date::new("2024-02-29").unwrap();
        assert_eq!((date.year(), date.month(), date.day()), (2024, 2, 29));
        for bad in [
            "2023-02-29",
            "2026-13-01",
            "2026-00-10",
            "2026-04-31",
            "26-04-01",
            "2026/04/01",
            "2026-4-1",
            "",
            "2026-04-0a",
        ] {
            let err = Date::new(bad).unwrap_err();
            assert!(
                matches!(err, Error::InvalidRequest { .. }),
                "{bad}: {err:?}"
            );
        }
        assert_eq!(Date::new("1900-02-28").unwrap().day(), 28);
        assert!(Date::new("1900-02-29").is_err());
        assert!(Date::new("2000-02-29").is_ok());
    }

    #[test]
    fn times_accept_rfc_3339_forms() {
        for good in [
            "13:34",
            "13:34:00",
            "23:59:60",
            "00:00:00.5",
            "13:34:00Z",
            "13:34:00+02:00",
            "13:34:00.123-05:30",
            "13:34+01:00",
        ] {
            Time::new(good).unwrap_or_else(|err| panic!("{good}: {err}"));
        }
        for bad in [
            "24:00",
            "13:60",
            "13:34:61",
            "1:34",
            "13:34:",
            "13:34:00.",
            "13:34:00+25:00",
            "13:34:00+0200",
            "13.34",
            "",
        ] {
            assert!(Time::new(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn datetimes_split_into_date_and_time() {
        let at = DateTime::new("2026-07-01T13:34:00Z").unwrap();
        assert_eq!(at.date().as_str(), "2026-07-01");
        assert_eq!(at.time().as_str(), "13:34:00Z");
        assert!(DateTime::new("2026-07-01 13:34").is_ok());
        for bad in [
            "2026-07-01",
            "2026-07-01T",
            "2026-07-01X13:34",
            "2026-07-01T25:00",
            "é",
        ] {
            assert!(DateTime::new(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn serde_is_the_bare_text_and_validates() {
        let date: Date = serde_json::from_str(r#""2026-09-28""#).unwrap();
        assert_eq!(serde_json::to_string(&date).unwrap(), r#""2026-09-28""#);
        assert!(serde_json::from_str::<Date>(r#""2026-09-31""#).is_err());
        assert_eq!(format!("{date:?}"), r#"Date("2026-09-28")"#);
        assert_eq!(date.to_string(), "2026-09-28");
    }

    #[test]
    fn spans_slice_on_char_boundaries() {
        let input = "Café A-1";
        let span = TextSpan::new(6, 9);
        assert_eq!(span.text(input), Some("A-1"));
        // Inside the two-byte `é`.
        assert_eq!(TextSpan::new(3, 4).text(input), None);
        assert_eq!(TextSpan::new(0, 99).text(input), None);
        assert!(TextSpan::new(3, 3).is_empty());
        assert_eq!(
            serde_json::to_value(span).unwrap(),
            serde_json::json!({"start": 6, "end": 9})
        );
    }
}
