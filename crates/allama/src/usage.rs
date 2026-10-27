use std::collections::BTreeMap;
use std::fmt;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Sub};

use serde::{Deserialize, Serialize};

/// Token usage reported by a provider.
///
/// # Contract
///
/// Every adapter normalizes provider counters to the same meaning:
///
/// - `input_tokens` is the **total** prompt size, *including* cache reads and
///   cache writes. `cache_read_input_tokens` and `cache_write_input_tokens`
///   are subsets of it.
/// - `output_tokens` is the **total** generated size, *including* reasoning
///   tokens. `reasoning_tokens` is a subset of it.
/// - `total_tokens` equals `input_tokens + output_tokens` when both are known
///   (providers that bill extra categories may report more, never less).
/// - [`breakdowns`](Self::breakdowns) holds only breakdowns of
///   `input_tokens` or `output_tokens`: tokens already counted there, under
///   the key names listed below. Adding them to input or output double
///   counts.
/// - [`tool_requests`](Self::tool_requests) holds **counts that are not
///   tokens**: the billable requests provider-executed tools made (a web
///   search, a web fetch), by tool. They are part of no token counter.
/// - `None` means the provider did not report the counter; `Some(0)` is a
///   reported zero.
///
/// ## Breakdown keys
///
/// A [`breakdowns`](Self::breakdowns) key names what it breaks down by its
/// suffix, `_input_tokens` or `_output_tokens`, so a consumer knows which
/// total it is part of; [`contract_violation`](Self::contract_violation)
/// rejects any other key, and a breakdown larger than its total. The
/// prefixes in use:
///
/// | Key | Part of | Meaning |
/// | --- | --- | --- |
/// | `{modality}_input_tokens` | input | Input tokens of one modality: `text`, `audio`, `image`, `video`, `document` |
/// | `{modality}_output_tokens` | output | Output tokens of one modality |
/// | `cache_write_5m_input_tokens`, `cache_write_1h_input_tokens` | input | Cache writes by time to live (subsets of `cache_write_input_tokens`) |
/// | `tool_use_input_tokens` | input | Input the provider added from provider-executed tools' results |
///
/// A provider counter that is neither a breakdown nor a tool request stays
/// in the response's provider data.
///
/// ## Tool request keys
///
/// A [`tool_requests`](Self::tool_requests) key is the tool in snake_case,
/// without a provider prefix or a `_requests` suffix: `web_search`,
/// `web_fetch`, `code_execution`, `file_search`.
///
/// # Summing and printing
///
/// Usage adds up with `+`, `+=` and [`Sum`]: a counter is the sum of the
/// reports that have it, and stays `None` only when none did. When either
/// side reported `total_tokens`, the sum's total adds each side's total,
/// derived from its input and output where it reported none, so
/// `total() >= input + output` holds for the sum too. Arithmetic saturates.
/// A response's usage is an `Option<Usage>`, so `+=` and [`Sum`] also take
/// an `Option<&Usage>`, where `None` (nothing reported) adds nothing.
/// `Display` writes one line for people, leaving out the counters the
/// provider did not report and the breakdowns; it is not a stable format,
/// so parse the fields, not the text.
///
/// ```
/// use allama::Usage;
///
/// let turns = [
///     Usage::new().input_tokens(1200).cache_read_input_tokens(800).output_tokens(340),
///     Usage::new().input_tokens(300).output_tokens(60).tool_requests("web_search", 2),
/// ];
///
/// let mut total = Usage::new();
/// for usage in &turns {
///     total += usage;
/// }
/// assert_eq!(total, turns.iter().sum::<Usage>());
/// assert_eq!(
///     total.to_string(),
///     "input 1500 (cache read 800), output 400, tool requests (web_search 2)"
/// );
///
/// // `ChatResponse::usage` is an `Option<Usage>`: add it as it is.
/// let responses = [Some(Usage::new().output_tokens(7)), None];
/// let mut spent = Usage::new();
/// for usage in &responses {
///     spent += usage.as_ref();
/// }
/// assert_eq!(spent, responses.iter().map(Option::as_ref).sum::<Usage>());
/// assert_eq!(spent.output_tokens, Some(7));
///
/// let reported: Option<Usage> = None;
/// let line = reported.as_ref().map(ToString::to_string);
/// assert_eq!(line, None);
/// assert_eq!(Usage::new().to_string(), "no usage reported");
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Usage {
    /// Total input tokens, including cache reads and writes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    /// Total output tokens, including reasoning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    /// Total tokens as reported or derived.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    /// Input tokens served from the prompt cache (subset of `input_tokens`).
    ///
    /// Local servers (Ollama, llama.cpp) report the reuse of their
    /// KV-cache prefix here: real reuse, but not a billing discount.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
    /// Input tokens written to the prompt cache (subset of `input_tokens`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_input_tokens: Option<u64>,
    /// Reasoning tokens (subset of `output_tokens`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
    /// Breakdowns of `input_tokens` or `output_tokens`, such as
    /// `audio_input_tokens`: tokens already counted there, under the
    /// [breakdown keys](Self#breakdown-keys).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub breakdowns: BTreeMap<String, u64>,
    /// Billable requests made by provider-executed tools, by tool, such as
    /// `web_search`: counts, not tokens (see the
    /// [tool request keys](Self#tool-request-keys)).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tool_requests: BTreeMap<String, u64>,
}

impl Usage {
    /// Create an empty usage report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set `input_tokens`.
    #[must_use]
    pub fn input_tokens(mut self, value: u64) -> Self {
        self.input_tokens = Some(value);
        self
    }

    /// Set `output_tokens`.
    #[must_use]
    pub fn output_tokens(mut self, value: u64) -> Self {
        self.output_tokens = Some(value);
        self
    }

    /// Set `total_tokens`.
    #[must_use]
    pub fn total_tokens(mut self, value: u64) -> Self {
        self.total_tokens = Some(value);
        self
    }

    /// Set `cache_read_input_tokens`.
    #[must_use]
    pub fn cache_read_input_tokens(mut self, value: u64) -> Self {
        self.cache_read_input_tokens = Some(value);
        self
    }

    /// Set `cache_write_input_tokens`.
    #[must_use]
    pub fn cache_write_input_tokens(mut self, value: u64) -> Self {
        self.cache_write_input_tokens = Some(value);
        self
    }

    /// Set `reasoning_tokens`.
    #[must_use]
    pub fn reasoning_tokens(mut self, value: u64) -> Self {
        self.reasoning_tokens = Some(value);
        self
    }

    /// Set a breakdown of `input_tokens` or `output_tokens`, under one of
    /// the [breakdown keys](Self#breakdown-keys).
    #[must_use]
    pub fn breakdown(mut self, key: impl Into<String>, value: u64) -> Self {
        self.breakdowns.insert(key.into(), value);
        self
    }

    /// Set the number of billable requests a provider-executed tool made,
    /// under its [tool request key](Self#tool-request-keys).
    #[must_use]
    pub fn tool_requests(mut self, tool: impl Into<String>, count: u64) -> Self {
        self.tool_requests.insert(tool.into(), count);
        self
    }

    /// Total tokens: the reported total, or `input + output` when both are known.
    #[must_use]
    pub fn total(&self) -> Option<u64> {
        self.total_tokens
            .or_else(|| Some(self.input_tokens?.saturating_add(self.output_tokens?)))
    }

    /// Fill `total_tokens` from `input + output` when it is missing.
    #[must_use]
    pub fn with_derived_total(mut self) -> Self {
        if self.total_tokens.is_none() {
            self.total_tokens = self.total();
        }
        self
    }

    /// Returns `true` when no counter is reported.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }

    /// Check the contract invariants. Returns a description of the first
    /// violation, if any. Used by conformance tests.
    ///
    /// Besides the subset rules, every [`breakdowns`](Self::breakdowns) key
    /// must end in `_input_tokens` or `_output_tokens` and be no larger than
    /// that total.
    #[must_use]
    pub fn contract_violation(&self) -> Option<String> {
        let input = self.input_tokens;
        for (name, part) in [
            ("cache_read_input_tokens", self.cache_read_input_tokens),
            ("cache_write_input_tokens", self.cache_write_input_tokens),
        ] {
            if let (Some(total), Some(part)) = (input, part)
                && part > total
            {
                return Some(format!("{name} ({part}) exceeds input_tokens ({total})"));
            }
        }
        if let (Some(read), Some(write), Some(total)) = (
            self.cache_read_input_tokens,
            self.cache_write_input_tokens,
            input,
        ) && read.saturating_add(write) > total
        {
            return Some(format!(
                "cache read + write ({}) exceeds input_tokens ({total})",
                read.saturating_add(write)
            ));
        }
        if let (Some(output), Some(reasoning)) = (self.output_tokens, self.reasoning_tokens)
            && reasoning > output
        {
            return Some(format!(
                "reasoning_tokens ({reasoning}) exceeds output_tokens ({output})"
            ));
        }
        if let (Some(total), Some(i), Some(o)) = (self.total_tokens, input, self.output_tokens)
            && total < i.saturating_add(o)
        {
            return Some(format!(
                "total_tokens ({total}) is less than input + output ({})",
                i.saturating_add(o)
            ));
        }
        for (key, &value) in &self.breakdowns {
            let (parent, total) = if key.ends_with("_input_tokens") {
                ("input_tokens", input)
            } else if key.ends_with("_output_tokens") {
                ("output_tokens", self.output_tokens)
            } else {
                return Some(format!(
                    "breakdown key `{key}` is not a breakdown of input or output tokens \
                     (it must end in `_input_tokens` or `_output_tokens`)"
                ));
            };
            if let Some(total) = total
                && value > total
            {
                return Some(format!("{key} ({value}) exceeds {parent} ({total})"));
            }
        }
        None
    }

    /// This report's total for summing: the reported total, but never less
    /// than input + output (unreported counters count as zero).
    fn total_floor(&self) -> u64 {
        let derived = self
            .input_tokens
            .unwrap_or(0)
            .saturating_add(self.output_tokens.unwrap_or(0));
        self.total_tokens.unwrap_or(0).max(derived)
    }
}

impl Add for Usage {
    type Output = Usage;

    fn add(mut self, rhs: Usage) -> Usage {
        self += &rhs;
        self
    }
}

impl Add<&Usage> for Usage {
    type Output = Usage;

    fn add(mut self, rhs: &Usage) -> Usage {
        self += rhs;
        self
    }
}

impl AddAssign for Usage {
    fn add_assign(&mut self, rhs: Usage) {
        *self += &rhs;
    }
}

impl AddAssign<&Usage> for Usage {
    fn add_assign(&mut self, rhs: &Usage) {
        // A side without a reported total contributes the one it implies,
        // so the sum keeps `total >= input + output`.
        if self.total_tokens.is_some() || rhs.total_tokens.is_some() {
            self.total_tokens = Some(self.total_floor().saturating_add(rhs.total_floor()));
        }
        self.input_tokens = add_options(self.input_tokens, rhs.input_tokens);
        self.output_tokens = add_options(self.output_tokens, rhs.output_tokens);
        self.cache_read_input_tokens =
            add_options(self.cache_read_input_tokens, rhs.cache_read_input_tokens);
        self.cache_write_input_tokens =
            add_options(self.cache_write_input_tokens, rhs.cache_write_input_tokens);
        self.reasoning_tokens = add_options(self.reasoning_tokens, rhs.reasoning_tokens);
        add_maps(&mut self.breakdowns, &rhs.breakdowns);
        add_maps(&mut self.tool_requests, &rhs.tool_requests);
    }
}

/// Adds a response's usage, which is `None` when the provider reported
/// none; `None` adds nothing.
impl AddAssign<Option<&Usage>> for Usage {
    fn add_assign(&mut self, rhs: Option<&Usage>) {
        if let Some(rhs) = rhs {
            *self += rhs;
        }
    }
}

/// Sums usage across responses, for example every pass of a multi-step
/// operation. Counters follow [`Add`]: a counter stays `None` only when no
/// summand reported it.
impl Sum for Usage {
    fn sum<I: Iterator<Item = Usage>>(iter: I) -> Usage {
        iter.fold(Usage::default(), |acc, usage| acc + usage)
    }
}

impl<'a> Sum<&'a Usage> for Usage {
    fn sum<I: Iterator<Item = &'a Usage>>(iter: I) -> Usage {
        iter.fold(Usage::default(), |acc, usage| acc + usage)
    }
}

/// Sums the usage of responses (`responses.iter().map(|r| r.usage.as_ref())`),
/// skipping those that reported none.
impl<'a> Sum<Option<&'a Usage>> for Usage {
    fn sum<I: Iterator<Item = Option<&'a Usage>>>(iter: I) -> Usage {
        iter.flatten().sum()
    }
}

impl Sub for Usage {
    type Output = Usage;

    fn sub(self, rhs: Usage) -> Usage {
        self - &rhs
    }
}

impl Sub<&Usage> for Usage {
    type Output = Usage;

    fn sub(self, rhs: &Usage) -> Usage {
        Usage {
            input_tokens: sub_options(self.input_tokens, rhs.input_tokens),
            output_tokens: sub_options(self.output_tokens, rhs.output_tokens),
            total_tokens: sub_options(self.total_tokens, rhs.total_tokens),
            cache_read_input_tokens: sub_options(
                self.cache_read_input_tokens,
                rhs.cache_read_input_tokens,
            ),
            cache_write_input_tokens: sub_options(
                self.cache_write_input_tokens,
                rhs.cache_write_input_tokens,
            ),
            reasoning_tokens: sub_options(self.reasoning_tokens, rhs.reasoning_tokens),
            breakdowns: sub_maps(&self.breakdowns, &rhs.breakdowns),
            tool_requests: sub_maps(&self.tool_requests, &rhs.tool_requests),
        }
    }
}

/// One line for people, such as `input 1200 (cache read 800, cache write
/// 0), output 340 (reasoning 120), total 1540, tool requests (web_search
/// 2)`; see the [type docs](Usage#summing-and-printing).
///
/// Counters the provider did not report are left out, `total` is the
/// reported [`total_tokens`](Usage::total_tokens) (never derived), and the
/// [`breakdowns`](Usage::breakdowns) are left out. An empty
/// report prints `no usage reported`.
impl fmt::Display for Usage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut groups = Vec::new();
        let mut group = |name: &str, value: Option<u64>, parts: &[(&str, Option<u64>)]| {
            let parts: Vec<String> = parts
                .iter()
                .filter_map(|(name, value)| Some(format!("{name} {}", (*value)?)))
                .collect();
            match (value, parts.is_empty()) {
                (Some(value), true) => groups.push(format!("{name} {value}")),
                (Some(value), false) => {
                    groups.push(format!("{name} {value} ({})", parts.join(", ")));
                }
                (None, _) => groups.extend(parts),
            }
        };
        group(
            "input",
            self.input_tokens,
            &[
                ("cache read", self.cache_read_input_tokens),
                ("cache write", self.cache_write_input_tokens),
            ],
        );
        group(
            "output",
            self.output_tokens,
            &[("reasoning", self.reasoning_tokens)],
        );
        group("total", self.total_tokens, &[]);
        if !self.tool_requests.is_empty() {
            let tools: Vec<String> = self
                .tool_requests
                .iter()
                .map(|(tool, count)| format!("{tool} {count}"))
                .collect();
            groups.push(format!("tool requests ({})", tools.join(", ")));
        }

        if groups.is_empty() {
            return f.write_str(if self.breakdowns.is_empty() {
                "no usage reported"
            } else {
                "only token breakdowns reported"
            });
        }
        f.write_str(&groups.join(", "))
    }
}

/// How much of the output went to reasoning, as in "400 of 400 output
/// tokens went to reasoning", when the reported usage shows any reasoning.
///
/// Only reported counters count: `None` when reasoning was not reported or
/// was zero.
#[cfg(any(feature = "chat-extract", feature = "warning-events"))]
pub(crate) fn reasoning_share(
    output_tokens: Option<u64>,
    reasoning_tokens: Option<u64>,
) -> Option<String> {
    let reasoning = reasoning_tokens.filter(|&tokens| tokens > 0)?;
    Some(match output_tokens {
        Some(output) => format!("{reasoning} of {output} output tokens went to reasoning"),
        None => format!("{reasoning} output tokens went to reasoning"),
    })
}

fn sub_options(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.saturating_sub(b)),
        (a, _) => a,
    }
}

fn add_maps(into: &mut BTreeMap<String, u64>, rhs: &BTreeMap<String, u64>) {
    for (key, value) in rhs {
        let entry = into.entry(key.clone()).or_default();
        *entry = entry.saturating_add(*value);
    }
}

fn sub_maps(lhs: &BTreeMap<String, u64>, rhs: &BTreeMap<String, u64>) -> BTreeMap<String, u64> {
    lhs.iter()
        .map(|(key, value)| {
            let rhs = rhs.get(key).copied().unwrap_or(0);
            (key.clone(), value.saturating_sub(rhs))
        })
        .collect()
}

fn add_options(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.saturating_add(b)),
        (a, None) => a,
        (None, b) => b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn breakdowns_serialize_as_breakdowns() {
        let usage = Usage::new().breakdown("audio_input_tokens", 3);
        let value = serde_json::to_value(&usage).unwrap();
        assert_eq!(
            value,
            serde_json::json!({"breakdowns": {"audio_input_tokens": 3}})
        );
        let back: Usage = serde_json::from_value(value).unwrap();
        assert_eq!(back, usage);
    }

    #[test]
    fn total_prefers_reported_total() {
        let u = Usage::new()
            .input_tokens(10)
            .output_tokens(20)
            .total_tokens(42);
        assert_eq!(u.total(), Some(42));
        assert_eq!(
            Usage::new().input_tokens(10).output_tokens(20).total(),
            Some(30)
        );
        assert_eq!(Usage::new().input_tokens(10).total(), None);
    }

    #[test]
    fn sum_over_owned_and_borrowed_usage() {
        let usages = vec![
            Usage::new().input_tokens(10).output_tokens(1),
            Usage::new()
                .input_tokens(5)
                .breakdown("audio_input_tokens", 2),
            Usage::new(),
        ];
        let borrowed: Usage = usages.iter().sum();
        assert_eq!(borrowed.input_tokens, Some(15));
        assert_eq!(borrowed.output_tokens, Some(1));
        assert_eq!(borrowed.total_tokens, None);
        assert_eq!(borrowed.breakdowns["audio_input_tokens"], 2);
        let owned: Usage = usages.into_iter().sum();
        assert_eq!(owned, borrowed);
        assert_eq!(std::iter::empty::<Usage>().sum::<Usage>(), Usage::default());
    }

    #[test]
    fn unreported_usage_adds_nothing() {
        let reports = [Some(Usage::new().input_tokens(3)), None];
        let mut total = Usage::new();
        for report in &reports {
            total += report.as_ref();
        }
        assert_eq!(total, Usage::new().input_tokens(3));
        assert_eq!(reports.iter().map(Option::as_ref).sum::<Usage>(), total);
        assert_eq!(
            std::iter::once(None::<&Usage>).sum::<Usage>(),
            Usage::default()
        );
    }

    #[test]
    fn add_keeps_reported_values_and_sums_breakdowns() {
        let a = Usage::new()
            .input_tokens(10)
            .breakdown("audio_input_tokens", 3);
        let b = Usage::new()
            .output_tokens(5)
            .breakdown("audio_input_tokens", 2);
        let sum = a + b;
        assert_eq!(sum.input_tokens, Some(10));
        assert_eq!(sum.output_tokens, Some(5));
        assert_eq!(sum.breakdowns["audio_input_tokens"], 5);
    }

    #[test]
    fn sub_saturates() {
        let a = Usage::new().input_tokens(10);
        let b = Usage::new().input_tokens(15);
        assert_eq!((a - b).input_tokens, Some(0));
    }

    #[test]
    fn display_shows_the_reported_counters() {
        let full = Usage::new()
            .input_tokens(1200)
            .cache_read_input_tokens(800)
            .cache_write_input_tokens(0)
            .output_tokens(340)
            .reasoning_tokens(120)
            .total_tokens(1540)
            .breakdown("audio_input_tokens", 7);
        assert_eq!(
            full.to_string(),
            "input 1200 (cache read 800, cache write 0), output 340 (reasoning 120), total 1540"
        );
        // `total` is the reported value, never derived.
        assert_eq!(
            Usage::new().input_tokens(10).output_tokens(5).to_string(),
            "input 10, output 5"
        );
        // A subset without its total still shows.
        assert_eq!(
            Usage::new().cache_read_input_tokens(3).to_string(),
            "cache read 3"
        );
        assert_eq!(Usage::new().to_string(), "no usage reported");
        assert_eq!(
            Usage::new().breakdown("audio_input_tokens", 1).to_string(),
            "only token breakdowns reported"
        );
        assert_eq!(
            Usage::new()
                .output_tokens(3)
                .tool_requests("web_search", 2)
                .tool_requests("web_fetch", 1)
                .to_string(),
            "output 3, tool requests (web_fetch 1, web_search 2)"
        );
    }

    #[test]
    #[cfg(any(feature = "chat-extract", feature = "warning-events"))]
    fn reasoning_share_is_said_only_from_reported_reasoning() {
        assert_eq!(
            reasoning_share(Some(400), Some(400)).as_deref(),
            Some("400 of 400 output tokens went to reasoning")
        );
        assert_eq!(
            reasoning_share(None, Some(120)).as_deref(),
            Some("120 output tokens went to reasoning")
        );
        assert_eq!(reasoning_share(Some(400), Some(0)), None);
        assert_eq!(reasoning_share(Some(400), None), None);
    }

    #[test]
    fn contract_checks() {
        assert!(
            Usage::new()
                .input_tokens(100)
                .cache_read_input_tokens(80)
                .output_tokens(10)
                .reasoning_tokens(4)
                .total_tokens(110)
                .contract_violation()
                .is_none()
        );
        assert!(
            Usage::new()
                .input_tokens(10)
                .cache_read_input_tokens(80)
                .contract_violation()
                .is_some()
        );
        assert!(
            Usage::new()
                .output_tokens(1)
                .reasoning_tokens(4)
                .contract_violation()
                .is_some()
        );
        assert!(
            Usage::new()
                .input_tokens(5)
                .output_tokens(5)
                .total_tokens(3)
                .contract_violation()
                .is_some()
        );
    }

    /// Regression: a reported total on one side and
    /// none on the other summed to the stale total.
    #[test]
    fn sums_with_mixed_totals_keep_the_total_contract() {
        let reported = Usage::new()
            .input_tokens(10)
            .output_tokens(5)
            .total_tokens(15);
        let derived = Usage::new().input_tokens(100).output_tokens(50);
        for sum in [
            reported.clone() + &derived,
            derived.clone() + &reported,
            [reported.clone(), derived.clone()].iter().sum(),
        ] {
            assert_eq!(sum.total_tokens, Some(165));
            assert_eq!(sum.total(), Some(165));
            assert_eq!(sum.contract_violation(), None);
        }
        // A reported total above input + output (an extra billed category)
        // is kept.
        let extra = Usage::new()
            .input_tokens(10)
            .output_tokens(5)
            .total_tokens(20);
        assert_eq!((extra + &derived).total_tokens, Some(170));
        // Neither side reported one: the sum reports none either.
        assert_eq!((derived.clone() + &derived).total_tokens, None);
        // A side that reported only a part still counts.
        let partial = Usage::new().input_tokens(7);
        let sum = reported + &partial;
        assert_eq!(sum.total_tokens, Some(22));
        assert_eq!(sum.contract_violation(), None);
    }

    /// Regression: absurd counters saturate
    /// instead of panicking in debug builds.
    #[test]
    fn arithmetic_and_checks_saturate() {
        let max = Usage::new()
            .input_tokens(u64::MAX)
            .output_tokens(u64::MAX)
            .cache_read_input_tokens(u64::MAX)
            .cache_write_input_tokens(u64::MAX)
            .total_tokens(1);
        assert!(max.contract_violation().is_some());
        let sum = max.clone() + &max;
        assert_eq!(sum.total_tokens, Some(u64::MAX));
        assert_eq!(sum.input_tokens, Some(u64::MAX));
    }

    #[test]
    fn breakdowns_must_name_their_total_and_fit_in_it() {
        let ok = Usage::new()
            .input_tokens(10)
            .output_tokens(5)
            .breakdown("audio_input_tokens", 4)
            .breakdown("image_output_tokens", 5)
            .tool_requests("web_search", 3);
        assert_eq!(ok.contract_violation(), None);
        let too_big = ok.clone().breakdown("audio_input_tokens", 11);
        assert_eq!(
            too_big.contract_violation().as_deref(),
            Some("audio_input_tokens (11) exceeds input_tokens (10)")
        );
        let not_a_breakdown = ok.breakdown("web_search_requests", 1);
        assert!(
            not_a_breakdown
                .contract_violation()
                .unwrap()
                .contains("`web_search_requests` is not a breakdown"),
        );
    }

    #[test]
    fn tool_requests_add_and_subtract_by_tool() {
        let a = Usage::new().tool_requests("web_search", 2);
        let b = Usage::new()
            .tool_requests("web_search", 1)
            .tool_requests("web_fetch", 4);
        let sum = a.clone() + &b;
        assert_eq!(sum.tool_requests["web_search"], 3);
        assert_eq!(sum.tool_requests["web_fetch"], 4);
        assert_eq!((sum - &a).tool_requests["web_search"], 1);
        let json = serde_json::to_value(&b).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"tool_requests": {"web_fetch": 4, "web_search": 1}})
        );
    }
}
