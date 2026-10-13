use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::Error;

/// A shared allowance of retries that bounds how much extra load retrying
/// adds while a provider keeps failing.
///
/// Without a budget, every caller retries every retryable error up to
/// [`max_attempts`](super::RetryPolicy::max_attempts), so a full outage
/// multiplies the load on the failing provider (three times with the
/// default policy). A budget is a bucket of tokens that every retry draws
/// from and successes refill, so once a provider has been failing for a
/// while, callers stop retrying and fail fast; first attempts are never
/// held back. Each [`RetryingProvider`](super::RetryingProvider) owns one by
/// default (see its [retry budget](super::RetryingProvider#retry-budget)
/// docs); share one across wrappers with
/// [`RetryPolicy::budget`](super::RetryPolicy::budget).
///
/// It follows the retry quota of the AWS SDKs, with the costs the AWS SDKs
/// and Tools Reference Guide documents for its standard retry strategy as
/// of 2026:
///
/// - the bucket starts full, at [`DEFAULT_CAPACITY`](Self::DEFAULT_CAPACITY)
///   (500) tokens, or the [`capacity`](Self::capacity) you set;
/// - a retry after a rate limit ([`Error::RateLimited`]) costs 5 tokens,
///   any other retry 14;
/// - when the bucket holds less than a retry costs, the retry is denied and
///   the error is returned unchanged;
/// - a call that succeeds on a retry refunds that retry's cost, and a call
///   that succeeds on its first attempt adds 1 token, never beyond the
///   capacity;
/// - the first attempt of a call never needs a token.
///
/// The costs and the refund are fixed: a drained budget recovers one
/// transient retry per 14 first-try successes. The budget applies after every other rule of the
/// policy: the classifier (server retry directive included), the attempt
/// limit, and the retry hint rule decide first, and a budget only ever
/// denies a retry they allowed.
///
/// Clones share their tokens, so one budget can span many wrappers. The
/// tokens live in memory: a budget bounds the retries of one process (one
/// isolate on Cloudflare Workers), not of a fleet. By default each provider
/// of a fallback has its own, so an outage of one does not use up the
/// other's retries; share one across both sides to bound retries for the
/// whole stack:
///
/// ```
/// # #[cfg(feature = "mock")] {
/// use allama::wrappers::{FallbackProvider, RetryBudget, RetryPolicy, RetryingProvider};
/// # use allama::mock::MockProvider;
/// # let (primary, backup) = (MockProvider::with_text("a"), MockProvider::with_text("b"));
/// # let (primary2, backup2) = (primary.clone(), backup.clone());
///
/// // A budget per provider: the default.
/// let separate = FallbackProvider::new(
///     RetryingProvider::new(primary, RetryPolicy::default()),
///     RetryingProvider::new(backup, RetryPolicy::default()),
/// );
///
/// // One budget for the whole stack.
/// let budget = RetryBudget::new();
/// let policy = RetryPolicy::new().budget(budget.clone());
/// let shared = FallbackProvider::new(
///     RetryingProvider::new(primary2, policy.clone()),
///     RetryingProvider::new(backup2, policy),
/// );
/// # let _ = (shared, separate); }
/// ```
///
/// Set the capacity before sharing a budget:
/// [`capacity`](Self::capacity) returns a new, full budget that shares
/// nothing with clones made earlier.
///
/// A custom retry wrapper follows the same rules: it takes its budget from
/// [`RetryPolicy::make_budget`](super::RetryPolicy::make_budget) once, when
/// it is built, calls [`try_acquire`](Self::try_acquire) before each retry
/// the policy allows, keeps the [`RetryPermit`] it returns, and hands it to
/// [`record_success`](Self::record_success) when the call succeeds.
#[derive(Clone)]
pub struct RetryBudget {
    shared: Arc<Shared>,
}

impl RetryBudget {
    /// The default number of tokens, and the number a new budget starts with.
    pub const DEFAULT_CAPACITY: u32 = 500;

    /// The default cost of a retry after any error except a rate limit.
    pub(crate) const DEFAULT_RETRY_COST: u32 = 14;

    /// The default cost of a retry after a rate limit
    /// ([`Error::RateLimited`]).
    pub(crate) const DEFAULT_THROTTLING_RETRY_COST: u32 = 5;

    /// The default number of tokens a call that succeeds on its first
    /// attempt adds.
    pub(crate) const DEFAULT_SUCCESS_REFUND: u32 = 1;

    /// A full budget with the AWS retry quota defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::from_config(Config {
            capacity: Self::DEFAULT_CAPACITY,
            retry_cost: Self::DEFAULT_RETRY_COST,
            throttling_retry_cost: Self::DEFAULT_THROTTLING_RETRY_COST,
            success_refund: Self::DEFAULT_SUCCESS_REFUND,
        })
    }

    /// Set the number of tokens the budget holds when full; a new budget
    /// starts full.
    #[must_use]
    pub fn capacity(self, capacity: u32) -> Self {
        self.reconfigure(|config| config.capacity = capacity)
    }

    /// Set the cost of a retry after any error except a rate limit.
    #[cfg(test)]
    pub(crate) fn retry_cost(self, cost: u32) -> Self {
        self.reconfigure(|config| config.retry_cost = cost)
    }

    /// Set the cost of a retry after a rate limit ([`Error::RateLimited`]).
    #[cfg(test)]
    pub(crate) fn throttling_retry_cost(self, cost: u32) -> Self {
        self.reconfigure(|config| config.throttling_retry_cost = cost)
    }

    /// Set the number of tokens a call that succeeds on its first attempt
    /// adds.
    #[cfg(test)]
    pub(crate) fn success_refund(self, refund: u32) -> Self {
        self.reconfigure(|config| config.success_refund = refund)
    }

    /// The tokens left.
    #[must_use]
    pub fn available(&self) -> u32 {
        self.shared.available.load(Ordering::Relaxed)
    }

    /// Withdraw the cost of retrying after `error`, or `None` when the
    /// budget holds less than that and the retry is denied.
    ///
    /// Call it only for a retry the policy already allows, and keep the
    /// permit until the call ends: pass the last one to
    /// [`record_success`](Self::record_success) if the call succeeds, and
    /// drop it otherwise. The [`RetryPermit`] is opaque: a custom retry
    /// wrapper only holds it, in its stream's state when a stream succeeds
    /// at its first output, and hands it back.
    #[must_use = "dropping the permit forfeits the refund a successful retry earns"]
    pub fn try_acquire(&self, error: &Error) -> Option<RetryPermit> {
        let cost = self.shared.config.cost(error);
        self.update_available(|available| available.checked_sub(cost))
            .then_some(RetryPermit { cost })
    }

    /// Record a call that succeeded: refund `last_retry`, the permit of the
    /// retry that succeeded, or, when the first attempt succeeded (`None`),
    /// add the success refund. The budget never exceeds its capacity.
    pub fn record_success(&self, last_retry: Option<RetryPermit>) {
        let refund = last_retry.map_or(self.shared.config.success_refund, |permit| permit.cost);
        let capacity = self.shared.config.capacity;
        // The closure always returns `Some`, so the update cannot fail.
        self.update_available(|available| Some(available.saturating_add(refund).min(capacity)));
    }

    /// Replace the available budget with `update`'s result, retrying when
    /// another call changed it in between; `false` when `update` returns
    /// `None` and the budget is left as it is. `fetch_update` does the
    /// same, but it is deprecated since Rust 1.99 in favour of `try_update`,
    /// which is unstable on the MSRV (1.92).
    fn update_available(&self, mut update: impl FnMut(u32) -> Option<u32>) -> bool {
        let mut available = self.shared.available.load(Ordering::Relaxed);
        while let Some(next) = update(available) {
            match self.shared.available.compare_exchange_weak(
                available,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(current) => available = current,
            }
        }
        false
    }

    fn from_config(config: Config) -> Self {
        Self {
            shared: Arc::new(Shared {
                available: AtomicU32::new(config.capacity),
                config,
            }),
        }
    }

    fn reconfigure(self, change: impl FnOnce(&mut Config)) -> Self {
        let mut config = self.shared.config;
        change(&mut config);
        Self::from_config(config)
    }
}

impl Default for RetryBudget {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for RetryBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let config = &self.shared.config;
        f.debug_struct("RetryBudget")
            .field("available", &self.available())
            .field("capacity", &config.capacity)
            .field("retry_cost", &config.retry_cost)
            .field("throttling_retry_cost", &config.throttling_retry_cost)
            .field("success_refund", &config.success_refund)
            .finish()
    }
}

/// Tokens withdrawn from a [`RetryBudget`] for one retry.
///
/// Returned by [`RetryBudget::try_acquire`]; hand the last one of a call to
/// [`RetryBudget::record_success`] when the call succeeds.
///
/// It is opaque, with nothing to read: a custom retry wrapper names it only
/// to hold it, such as in the state of a stream that counts as a success
/// once it reaches its first output:
///
/// ```
/// use allama::wrappers::{RetryBudget, RetryPermit};
///
/// /// How far a call through a custom retry wrapper has got.
/// struct Attempts {
///     number: usize,
///     /// The permit of the retry that started the current attempt.
///     permit: Option<RetryPermit>,
/// }
///
/// let budget = RetryBudget::new();
/// let mut attempts = Attempts { number: 1, permit: None };
/// // The first attempt failed, and the policy allows a retry.
/// if let Some(permit) = budget.try_acquire(&allama::Error::overloaded("busy")) {
///     attempts.number += 1;
///     attempts.permit = Some(permit);
/// }
/// // The retry reached its first output: refund what it cost.
/// budget.record_success(attempts.permit.take());
/// assert_eq!(budget.available(), RetryBudget::DEFAULT_CAPACITY);
/// ```
pub struct RetryPermit {
    cost: u32,
}

impl RetryPermit {
    /// The tokens this retry cost, refunded if it succeeds.
    #[cfg(test)]
    pub(crate) fn cost(&self) -> u32 {
        self.cost
    }
}

impl fmt::Debug for RetryPermit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RetryPermit").finish_non_exhaustive()
    }
}

struct Shared {
    config: Config,
    available: AtomicU32,
}

#[derive(Clone, Copy)]
struct Config {
    capacity: u32,
    retry_cost: u32,
    throttling_retry_cost: u32,
    success_refund: u32,
}

impl Config {
    /// The cost of a retry after `error`: rate limits, the errors expected
    /// to clear soonest, cost less.
    fn cost(&self, error: &Error) -> u32 {
        if matches!(error, Error::RateLimited { .. }) {
            self.throttling_retry_cost
        } else {
            self.retry_cost
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_full_with_aws_defaults() {
        let budget = RetryBudget::new();
        assert_eq!(budget.available(), 500);
        let permit = budget.try_acquire(&Error::overloaded("busy")).unwrap();
        assert_eq!(permit.cost(), 14);
        let permit = budget.try_acquire(&Error::rate_limited("slow")).unwrap();
        assert_eq!(permit.cost(), 5);
        assert_eq!(budget.available(), 500 - 14 - 5);
    }

    #[test]
    fn a_failover_ending_in_a_rate_limit_costs_a_throttling_retry() {
        let budget = RetryBudget::new();
        let error = Error::rate_limited("slow down").with_fallback_from(Error::timeout("slow"));
        assert_eq!(budget.try_acquire(&error).unwrap().cost(), 5);
    }

    #[test]
    fn denies_a_retry_it_cannot_afford() {
        let budget = RetryBudget::new().capacity(20);
        assert!(budget.try_acquire(&Error::timeout("a")).is_some());
        assert_eq!(budget.available(), 6);
        // 6 tokens: a transient retry (14) is denied, a throttling one (5) is not.
        assert!(budget.try_acquire(&Error::timeout("b")).is_none());
        assert_eq!(budget.available(), 6);
        assert!(budget.try_acquire(&Error::rate_limited("c")).is_some());
        assert_eq!(budget.available(), 1);
    }

    #[test]
    fn successes_refund_per_the_aws_rules() {
        let budget = RetryBudget::new().capacity(100);
        let permit = budget.try_acquire(&Error::timeout("x")).unwrap();
        assert_eq!(budget.available(), 86);
        // A successful retry refunds its cost.
        budget.record_success(Some(permit));
        assert_eq!(budget.available(), 100);

        let _ = budget.try_acquire(&Error::timeout("x")).unwrap();
        // A first-try success adds one token.
        budget.record_success(None);
        assert_eq!(budget.available(), 87);
        // Never beyond the capacity.
        let full = RetryBudget::new().capacity(10);
        full.record_success(None);
        assert_eq!(full.available(), 10);
        full.record_success(Some(RetryPermit { cost: 14 }));
        assert_eq!(full.available(), 10);
    }

    #[test]
    fn clones_share_tokens_and_setters_start_afresh() {
        let budget = RetryBudget::new();
        let clone = budget.clone();
        let _ = clone.try_acquire(&Error::timeout("x")).unwrap();
        assert_eq!(budget.available(), 486);

        let reconfigured = budget.clone().retry_cost(1);
        assert_eq!(reconfigured.available(), 500);
        let _ = reconfigured.try_acquire(&Error::timeout("x")).unwrap();
        assert_eq!(reconfigured.available(), 499);
        assert_eq!(budget.available(), 486);
    }

    #[test]
    fn costs_and_refund_are_configurable() {
        let budget = RetryBudget::new()
            .capacity(10)
            .retry_cost(3)
            .throttling_retry_cost(2)
            .success_refund(4);
        assert_eq!(budget.try_acquire(&Error::timeout("x")).unwrap().cost(), 3);
        assert_eq!(
            budget
                .try_acquire(&Error::rate_limited("x"))
                .unwrap()
                .cost(),
            2
        );
        assert_eq!(budget.available(), 5);
        budget.record_success(None);
        assert_eq!(budget.available(), 9);
    }

    #[test]
    fn concurrent_withdrawals_never_overdraw() {
        let budget = RetryBudget::new().capacity(14 * 10);
        let granted: usize = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let budget = budget.clone();
                    scope.spawn(move || {
                        (0..10)
                            .filter(|_| budget.try_acquire(&Error::timeout("x")).is_some())
                            .count()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).sum()
        });
        assert_eq!(granted, 10);
        assert_eq!(budget.available(), 0);
    }

    #[test]
    fn debug_shows_tokens_and_settings() {
        let debug = format!("{:?}", RetryBudget::new());
        assert!(debug.contains("available: 500"), "{debug}");
        assert!(debug.contains("retry_cost: 14"), "{debug}");
    }
}
