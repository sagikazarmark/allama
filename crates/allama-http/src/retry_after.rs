//! Parsing of the `Retry-After` and `retry-after-ms` response headers.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Parse a retry-after hint from header values.
///
/// Understands:
///
/// - `retry-after-ms`: milliseconds, integer or fractional (sent by several
///   vendors and honored by their SDKs). It wins over `retry-after`.
/// - `retry-after` as delta-seconds, integer or fractional.
/// - `retry-after` as an HTTP-date relative to `now`, in any of the three
///   forms RFC 9110 asks recipients to accept: IMF-fixdate
///   (`Sun, 06 Nov 1994 08:49:37 GMT`), the obsolete RFC 850 form
///   (`Sunday, 06-Nov-94 08:49:37 GMT`) and asctime
///   (`Sun Nov  6 08:49:37 1994`).
///
/// A date at or before `now` gives `Duration::ZERO`, which
/// [`RetryPolicy`](allama::wrappers::RetryPolicy) treats as no hint.
///
/// Never panics. A value that does not fit a [`Duration`] (such as
/// `retry-after: 1e20`), a negative or non-finite number, and a date that is
/// malformed or out of range give `None`, so a hostile or buggy upstream
/// cannot crash the caller.
///
/// [`Headers::retry_after`](crate::Headers::retry_after) calls this with the
/// current time and also works on `wasm32-unknown-unknown`.
///
/// ```
/// use std::time::{Duration, UNIX_EPOCH};
///
/// use allama_http::parse_retry_after;
///
/// let now = UNIX_EPOCH + Duration::from_secs(784_111_777); // Sun, 06 Nov 1994 08:49:37 GMT
/// assert_eq!(parse_retry_after(Some("2"), None, now), Some(Duration::from_secs(2)));
/// assert_eq!(parse_retry_after(Some("2"), Some("250"), now), Some(Duration::from_millis(250)));
/// assert_eq!(
///     parse_retry_after(Some("Sun, 06 Nov 1994 08:49:47 GMT"), None, now),
///     Some(Duration::from_secs(10)),
/// );
/// assert_eq!(parse_retry_after(Some("1e20"), None, now), None);
/// ```
#[must_use]
pub fn parse_retry_after(
    retry_after: Option<&str>,
    retry_after_ms: Option<&str>,
    now: SystemTime,
) -> Option<Duration> {
    if let Some(ms) = retry_after_ms.and_then(parse_seconds_like) {
        return Duration::try_from_secs_f64(ms / 1000.0).ok();
    }
    let value = retry_after?.trim();
    if let Some(secs) = parse_seconds_like(value) {
        return Duration::try_from_secs_f64(secs).ok();
    }
    let at = parse_http_date(value, now)?;
    Some(at.duration_since(now).unwrap_or(Duration::ZERO))
}

/// A finite, non-negative number.
fn parse_seconds_like(value: &str) -> Option<f64> {
    value
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite() && *v >= 0.0)
}

/// Parse an HTTP-date in any of the three RFC 9110 forms.
fn parse_http_date(value: &str, now: SystemTime) -> Option<SystemTime> {
    let parts: Vec<&str> = value.split_whitespace().collect();
    let (day, month, year, time) = match parts.as_slice() {
        // IMF-fixdate: `Sun, 06 Nov 1994 08:49:37 GMT`.
        [weekday, day, month, year, time, "GMT"] if weekday.ends_with(',') => {
            (*day, *month, four_digit_year(year)?, *time)
        }
        // RFC 850: `Sunday, 06-Nov-94 08:49:37 GMT`.
        [weekday, date, time, "GMT"] if weekday.ends_with(',') => {
            let mut fields = date.split('-');
            let (day, month, year) = (fields.next()?, fields.next()?, fields.next()?);
            if fields.next().is_some() {
                return None;
            }
            (day, month, two_digit_year(year, now)?, *time)
        }
        // asctime: `Sun Nov  6 08:49:37 1994`.
        [_weekday, month, day, time, year] => (*day, *month, four_digit_year(year)?, *time),
        _ => return None,
    };
    let day: u64 = day.parse().ok()?;
    let month = month_number(month)?;
    let mut hms = time.split(':').map(|p| p.parse::<u64>().ok());
    let (h, m, s) = (hms.next()??, hms.next()??, hms.next()??);
    if hms.next().is_some() || h > 23 || m > 59 || s > 60 || !(1..=31).contains(&day) {
        return None;
    }
    let days = days_from_civil(year, month, day);
    let secs = u64::try_from(days)
        .ok()?
        .checked_mul(86_400)?
        .checked_add(h * 3600 + m * 60 + s)?;
    UNIX_EPOCH.checked_add(Duration::from_secs(secs))
}

/// A year written with exactly four digits, as IMF-fixdate and asctime
/// require. Bounding the digits also bounds the date arithmetic.
fn four_digit_year(value: &str) -> Option<i64> {
    (value.len() == 4 && value.bytes().all(|b| b.is_ascii_digit()))
        .then(|| value.parse().ok())
        .flatten()
}

/// An RFC 850 two-digit year: RFC 9110 reads one that would be more than 50
/// years after `now` as the most recent past year with those digits.
fn two_digit_year(value: &str, now: SystemTime) -> Option<i64> {
    if value.len() != 2 || !value.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let yy: i64 = value.parse().ok()?;
    let days_now = now.duration_since(UNIX_EPOCH).ok()?.as_secs() / 86_400;
    let current = year_from_days(i64::try_from(days_now).ok()?);
    let mut year = current - current.rem_euclid(100) + yy;
    if year > current + 50 {
        year -= 100;
    }
    Some(year)
}

fn month_number(name: &str) -> Option<i64> {
    Some(match name {
        "Jan" => 1,
        "Feb" => 2,
        "Mar" => 3,
        "Apr" => 4,
        "May" => 5,
        "Jun" => 6,
        "Jul" => 7,
        "Aug" => 8,
        "Sep" => 9,
        "Oct" => 10,
        "Nov" => 11,
        "Dec" => 12,
        _ => return None,
    })
}

/// Days since 1970-01-01 (Howard Hinnant's `days_from_civil`). The callers
/// bound `year` to four digits, so the arithmetic cannot overflow.
fn days_from_civil(year: i64, month: i64, day: u64) -> i64 {
    let (y, mo) = if month <= 2 {
        (year - 1, month + 9)
    } else {
        (year, month - 3)
    };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * mo + 2) / 5 + day as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The civil year of a day count since 1970-01-01 (Hinnant's
/// `civil_from_days`, year only).
fn year_from_days(days: i64) -> i64 {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let year = yoe + era * 400;
    if mp >= 10 { year + 1 } else { year }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sun, 06 Nov 1994 08:49:37 GMT.
    fn now() -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(784_111_777)
    }

    #[test]
    fn parses_delta_seconds_milliseconds_and_dates() {
        let now = now();
        assert_eq!(
            parse_retry_after(Some("3"), None, now),
            Some(Duration::from_secs(3))
        );
        assert_eq!(
            parse_retry_after(Some("1.5"), None, now),
            Some(Duration::from_millis(1500))
        );
        assert_eq!(
            parse_retry_after(Some("3"), Some("250"), now),
            Some(Duration::from_millis(250))
        );
        assert_eq!(
            parse_retry_after(Some("Sun, 06 Nov 1994 08:49:47 GMT"), None, now),
            Some(Duration::from_secs(10))
        );
        assert_eq!(
            parse_retry_after(Some("Sun, 06 Nov 1994 08:49:27 GMT"), None, now),
            Some(Duration::ZERO)
        );
        assert_eq!(
            parse_retry_after(Some("soon"), None, SystemTime::now()),
            None
        );
        assert_eq!(parse_retry_after(None, None, now), None);
    }

    #[test]
    fn parses_the_obsolete_date_forms() {
        let now = now();
        assert_eq!(
            parse_retry_after(Some("Sunday, 06-Nov-94 08:49:47 GMT"), None, now),
            Some(Duration::from_secs(10))
        );
        assert_eq!(
            parse_retry_after(Some("Sun Nov  6 08:49:47 1994"), None, now),
            Some(Duration::from_secs(10))
        );
        // A two-digit year is read near `now`, and one more than 50 years
        // ahead is in the past century.
        let new_year_2026 =
            UNIX_EPOCH + Duration::from_secs(days_from_civil(2026, 1, 1) as u64 * 86_400);
        assert_eq!(
            parse_retry_after(
                Some("Thursday, 01-Jan-26 00:00:10 GMT"),
                None,
                new_year_2026
            ),
            Some(Duration::from_secs(10))
        );
        assert_eq!(
            parse_retry_after(Some("Sunday, 06-Nov-94 08:49:47 GMT"), None, new_year_2026),
            Some(Duration::ZERO)
        );
    }

    /// Regression: these used to panic.
    #[test]
    fn huge_or_malformed_values_never_panic() {
        let now = now();
        for (retry_after, retry_after_ms) in [
            (Some("1e20"), None),
            (Some("99999999999999999999"), None),
            (None, Some("1e30")),
            (Some("1.9e19"), None),
            (Some("Sun, 06 Nov 300000000000 08:49:37 GMT"), None),
            (Some("Sun, 06 Nov 9223372036854775807 08:49:37 GMT"), None),
            (Some("Sun, 06 Nov -9223372036854775807 08:49:37 GMT"), None),
            (Some("Sun Nov  6 08:49:37 99999999999999"), None),
            (
                Some("Sunday, 06-Nov-9223372036854775807 08:49:37 GMT"),
                None,
            ),
            (Some("Sun, 99 Nov 1994 99:99:99 GMT"), None),
            (Some("Sun, 06 Nov 1994 08:49:37:00 GMT"), None),
            (Some("-1"), Some("-5")),
            (Some("NaN"), Some("inf")),
        ] {
            assert_eq!(
                parse_retry_after(retry_after, retry_after_ms, now),
                None,
                "{retry_after:?} / {retry_after_ms:?}"
            );
        }
        // Even an extreme `now` is safe.
        assert_eq!(
            parse_retry_after(Some("Sunday, 06-Nov-94 08:49:47 GMT"), None, UNIX_EPOCH),
            Some(Duration::from_secs(784_111_787))
        );
    }

    #[test]
    fn year_from_days_inverts_days_from_civil() {
        for year in [1970, 1994, 2000, 2024, 2100, 9999] {
            for (month, day) in [(1, 1), (2, 29), (12, 31)] {
                let days = days_from_civil(year, month, day);
                assert_eq!(year_from_days(days), year, "{year}-{month}-{day}");
            }
        }
    }
}
