//! `RetryBudget` under `RetryingProvider`: each wrapper owns a budget by
//! default, retries during a full outage stay within it, successes refill
//! it per the AWS retry quota rules, first attempts are never held back,
//! and one budget can span several wrappers when shared explicitly.

#![cfg(feature = "mock")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use allama::decision::{Answer, Question, YesNoAnswer};
use allama::mock::{FunctionProvider, MockDecisionProvider, MockEmbeddingProvider, MockProvider};
use allama::wrappers::{FallbackProvider, RetryBudget, RetryPermit, RetryPolicy, RetryingProvider};
use allama::{
    ChatProvider, ChatRequest, ChatResponse, ChatStreamExt, DecisionProvider, DecisionRequest,
    DecisionResponse, EmbeddingProvider, EmbeddingRequest, Error,
};

fn request() -> ChatRequest {
    ChatRequest::new("m").user("hi")
}

/// The default policy without waiting.
fn policy() -> RetryPolicy {
    RetryPolicy::new()
        .base_delay(Duration::ZERO)
        .max_delay(Duration::ZERO)
}

/// A chat provider that always fails with `error`, and its call count.
fn outage(error: fn() -> Error) -> (FunctionProvider, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let provider = FunctionProvider::new(move |_| {
        counter.fetch_add(1, Ordering::SeqCst);
        Err(error())
    });
    (provider, calls)
}

/// A chat provider whose every call fails once, then succeeds on the retry.
fn flaky() -> (FunctionProvider, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let provider = FunctionProvider::new(move |_| {
        if counter.fetch_add(1, Ordering::SeqCst).is_multiple_of(2) {
            Err(Error::overloaded("busy"))
        } else {
            Ok(ChatResponse::default())
        }
    });
    (provider, calls)
}

const CALLS: usize = 100;

#[tokio::test]
async fn full_outage_stays_within_the_budget() {
    // Without a budget every call makes all three attempts.
    let (provider, upstream) = outage(|| Error::overloaded("down"));
    let wrapper = RetryingProvider::new(provider, policy().no_budget());
    for _ in 0..CALLS {
        assert!(wrapper.chat(&request()).await.is_err());
    }
    assert_eq!(upstream.load(Ordering::SeqCst), 3 * CALLS);

    // A default budget affords 500 / 14 = 35 transient retries in all.
    let budget = RetryBudget::new();
    let (provider, upstream) = outage(|| Error::overloaded("down"));
    let wrapper = RetryingProvider::new(provider, policy().budget(budget.clone()));
    for _ in 0..CALLS {
        let error = wrapper.chat(&request()).await.unwrap_err();
        assert_eq!(error.kind_name(), "overloaded");
    }
    // The documented costs: 14 tokens a retry, 5 after a rate limit.
    let retries = (RetryBudget::DEFAULT_CAPACITY / 14) as usize;
    assert_eq!(upstream.load(Ordering::SeqCst), CALLS + retries);
    assert_eq!(budget.available(), 500 % 14);

    // Rate limits are cheaper: 500 / 5 = 100 retries.
    let (provider, upstream) = outage(|| Error::rate_limited("slow down"));
    let wrapper = RetryingProvider::new(provider, policy().budget(RetryBudget::new()));
    for _ in 0..CALLS {
        assert!(wrapper.chat(&request()).await.is_err());
    }
    let retries = (RetryBudget::DEFAULT_CAPACITY / 5) as usize;
    assert_eq!(upstream.load(Ordering::SeqCst), CALLS + retries);
}

#[tokio::test]
async fn streams_draw_on_the_budget_too() {
    let (provider, upstream) = outage(|| Error::overloaded("down"));
    let wrapper = RetryingProvider::new(
        provider,
        policy().budget(RetryBudget::new().capacity(14 * 3)),
    );
    for _ in 0..CALLS {
        assert!(wrapper.chat_stream(&request()).await.is_err());
    }
    assert_eq!(upstream.load(Ordering::SeqCst), CALLS + 3);
}

#[tokio::test]
async fn first_attempts_are_never_blocked() {
    let budget = RetryBudget::new().capacity(0);
    let (provider, upstream) = outage(|| Error::overloaded("down"));
    let wrapper = RetryingProvider::new(provider, policy().budget(budget.clone()));
    for _ in 0..10 {
        assert!(wrapper.chat(&request()).await.is_err());
    }
    assert_eq!(upstream.load(Ordering::SeqCst), 10);

    let mock = MockProvider::with_text("ok");
    let wrapper = RetryingProvider::new(mock.clone(), policy().budget(budget));
    assert!(wrapper.chat(&request()).await.is_ok());
    assert_eq!(mock.call_count(), 1);
}

#[tokio::test]
async fn a_successful_retry_refunds_its_cost() {
    // Exactly one transient retry's worth: without the refund, the second
    // call's retry would be denied.
    let budget = RetryBudget::new().capacity(14);
    let (provider, upstream) = flaky();
    let wrapper = RetryingProvider::new(provider, policy().budget(budget.clone()));
    for _ in 0..CALLS {
        assert!(wrapper.chat(&request()).await.is_ok());
        assert_eq!(budget.available(), 14);
    }
    assert_eq!(upstream.load(Ordering::SeqCst), 2 * CALLS);
}

#[tokio::test]
async fn only_the_last_retry_is_refunded() {
    // Two failures, then success: both retries are paid for, only the
    // second is refunded.
    let budget = RetryBudget::new().capacity(100);
    let mock = MockProvider::build(|b| {
        b.error(Error::overloaded("one"))
            .error(Error::rate_limited("two"))
            .text("ok")
    });
    let wrapper = RetryingProvider::new(mock, policy().budget(budget.clone()));
    assert!(wrapper.chat(&request()).await.is_ok());
    assert_eq!(budget.available(), 100 - 14);
}

#[tokio::test]
async fn first_try_successes_refill_a_drained_budget() {
    let budget = RetryBudget::new().capacity(20);
    let (provider, _) = outage(|| Error::overloaded("down"));
    let failing = RetryingProvider::new(provider, policy().budget(budget.clone()));
    assert!(failing.chat(&request()).await.is_err());
    assert_eq!(budget.available(), 6);

    let healthy = RetryingProvider::new(
        FunctionProvider::new(|_| Ok(ChatResponse::default())),
        policy().budget(budget.clone()),
    );
    for _ in 0..8 {
        healthy.chat(&request()).await.unwrap();
    }
    assert_eq!(budget.available(), 14);
    // One more transient retry is affordable again.
    let (provider, upstream) = outage(|| Error::overloaded("down"));
    let failing = RetryingProvider::new(provider, policy().budget(budget.clone()));
    assert!(failing.chat(&request()).await.is_err());
    assert_eq!(upstream.load(Ordering::SeqCst), 2);
    // Never beyond the capacity.
    for _ in 0..100 {
        healthy.chat(&request()).await.unwrap();
    }
    assert_eq!(budget.available(), 20);
}

#[tokio::test]
async fn a_shared_budget_drains_jointly_under_a_fallback() {
    // Two transient retries for the whole stack.
    let budget = RetryBudget::new().capacity(28);
    let (primary, primary_calls) = outage(|| Error::overloaded("down"));
    let (backup, backup_calls) = outage(|| Error::overloaded("down"));
    let provider = FallbackProvider::new(
        RetryingProvider::new(primary, policy().budget(budget.clone())),
        RetryingProvider::new(backup, policy().budget(budget.clone())),
    );
    let error = provider.chat(&request()).await.unwrap_err();
    assert!(matches!(error, Error::Overloaded { .. }));
    assert!(matches!(
        error.fallback_from(),
        Some(Error::Overloaded { .. })
    ));
    // The primary spends both retries; the backup gets none.
    assert_eq!(primary_calls.load(Ordering::SeqCst), 3);
    assert_eq!(backup_calls.load(Ordering::SeqCst), 1);
    assert_eq!(budget.available(), 0);
}

#[tokio::test]
async fn separate_budgets_drain_independently_under_a_fallback() {
    let primary_budget = RetryBudget::new().capacity(28);
    let backup_budget = RetryBudget::new().capacity(28);
    let (primary, primary_calls) = outage(|| Error::overloaded("down"));
    let (backup, backup_calls) = outage(|| Error::overloaded("down"));
    let provider = FallbackProvider::new(
        RetryingProvider::new(primary, policy().budget(primary_budget.clone())),
        RetryingProvider::new(backup, policy().budget(backup_budget.clone())),
    );
    assert!(provider.chat(&request()).await.is_err());
    assert_eq!(primary_calls.load(Ordering::SeqCst), 3);
    assert_eq!(backup_calls.load(Ordering::SeqCst), 3);

    // The primary's budget is spent; the backup's is too, but only by the
    // backup's own retries.
    assert_eq!(primary_budget.available(), 0);
    assert_eq!(backup_budget.available(), 0);
    assert!(provider.chat(&request()).await.is_err());
    assert_eq!(primary_calls.load(Ordering::SeqCst), 4);
    assert_eq!(backup_calls.load(Ordering::SeqCst), 4);
}

#[tokio::test]
async fn a_server_retry_directive_never_overrides_a_denial() {
    let (provider, upstream) = outage(|| {
        Error::invalid_request("HTTP 409")
            .with_status(409)
            .with_server_retry_directive(true)
    });
    let wrapper = RetryingProvider::new(provider, policy().budget(RetryBudget::new().capacity(0)));
    let error = wrapper.chat(&request()).await.unwrap_err();
    assert_eq!(error.server_retry_directive(), Some(true));
    assert_eq!(upstream.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_denied_retry_returns_the_error_unchanged() {
    let mock = MockProvider::build(|b| {
        b.error(
            Error::rate_limited("slow down")
                .with_status(429)
                .with_request_id("req-1"),
        )
        .text("never")
    });
    let wrapper = RetryingProvider::new(
        mock.clone(),
        policy().budget(RetryBudget::new().capacity(4)),
    );
    let error = wrapper.chat(&request()).await.unwrap_err();
    assert_eq!(error.kind_name(), "rate_limited");
    assert_eq!(error.message(), "slow down");
    assert_eq!(error.status(), Some(429));
    assert_eq!(error.request_id(), Some("req-1"));
    assert_eq!(mock.call_count(), 1);
}

#[tokio::test]
async fn embeddings_and_decisions_share_the_chat_budget() {
    // One transient retry for every operation together.
    let budget = RetryBudget::new().capacity(14);

    let embedder = MockEmbeddingProvider::new([
        Err(Error::overloaded("busy")),
        Err(Error::overloaded("busy")),
        Err(Error::overloaded("busy")),
    ]);
    let wrapper = RetryingProvider::new(embedder.clone(), policy().budget(budget.clone()));
    assert!(
        wrapper
            .embed(&EmbeddingRequest::new("e").inputs(["a"]))
            .await
            .is_err()
    );
    assert_eq!(embedder.requests().len(), 2);
    assert_eq!(budget.available(), 0);

    // The chat call finds the budget spent.
    let (provider, upstream) = outage(|| Error::overloaded("down"));
    let chat = RetryingProvider::new(provider, policy().budget(budget.clone()));
    assert!(chat.chat(&request()).await.is_err());
    assert_eq!(upstream.load(Ordering::SeqCst), 1);

    // A decision that succeeds on its first attempt refunds one token.
    let decider = MockDecisionProvider::new([
        Err(Error::overloaded("busy")),
        Ok(DecisionResponse::new([(
            "q",
            Answer::from(YesNoAnswer::new(0.9)),
        )])),
    ]);
    let wrapper = RetryingProvider::new(decider.clone(), policy().budget(budget.clone()));
    let request = DecisionRequest::new("d", "x").question("q", Question::yes_no("?"));
    assert!(wrapper.decide(&request).await.is_err());
    assert_eq!(decider.requests().len(), 1);
    wrapper.decide(&request).await.unwrap();
    assert_eq!(budget.available(), 1);
}

#[tokio::test]
async fn a_stream_that_commits_counts_as_a_success() {
    let budget = RetryBudget::new().capacity(14);
    let mock = MockProvider::build(|b| b.error(Error::overloaded("busy")).text("ok"));
    let wrapper = RetryingProvider::new(mock, policy().budget(budget.clone()));
    let response = wrapper
        .chat_stream(&request())
        .await
        .unwrap()
        .collect_response()
        .await
        .unwrap();
    assert_eq!(response.text().as_deref(), Some("ok"));
    assert_eq!(budget.available(), 14);
}

#[tokio::test]
async fn each_wrapper_owns_a_budget_by_default() {
    let (provider, upstream) = outage(|| Error::overloaded("down"));
    let wrapper = RetryingProvider::new(provider, policy());
    for _ in 0..CALLS {
        assert!(wrapper.chat(&request()).await.is_err());
    }
    // The default budget affords 500 / 14 = 35 transient retries.
    assert_eq!(upstream.load(Ordering::SeqCst), CALLS + 35);
    assert_eq!(wrapper.budget().unwrap().available(), 500 % 14);
    assert!(
        RetryingProvider::new(MockProvider::with_text("ok"), RetryPolicy::default())
            .budget()
            .is_some()
    );
}

/// Two wrappers built with clones of one policy each own a budget: policy
/// clones share a budget only when it was set explicitly.
#[tokio::test]
async fn policy_clones_do_not_share_a_default_budget() {
    let policy = policy();
    let (first, _) = outage(|| Error::overloaded("down"));
    let (second, second_calls) = outage(|| Error::overloaded("down"));
    let first = RetryingProvider::new(first, policy.clone());
    let second = RetryingProvider::new(second, policy);
    for _ in 0..CALLS {
        assert!(first.chat(&request()).await.is_err());
    }
    assert_eq!(first.budget().unwrap().available(), 500 % 14);
    // The second wrapper's budget is untouched, so it still retries.
    assert_eq!(second.budget().unwrap().available(), 500);
    assert!(second.chat(&request()).await.is_err());
    assert_eq!(second_calls.load(Ordering::SeqCst), 3);
}

/// Clones of a wrapper share its budget, as clones of an AWS client share
/// its token bucket; a new wrapper around a clone of the inner provider
/// gets a budget of its own.
#[tokio::test]
async fn wrapper_clones_share_the_budget_until_given_a_policy() {
    let (provider, _) = outage(|| Error::overloaded("down"));
    let wrapper = RetryingProvider::new(provider, policy());
    let clone = wrapper.clone();
    assert!(clone.chat(&request()).await.is_err());
    assert_eq!(wrapper.budget().unwrap().available(), 500 - 2 * 14);

    let detached = RetryingProvider::new(wrapper.inner().clone(), policy());
    assert_eq!(detached.budget().unwrap().available(), 500);
    assert!(detached.chat(&request()).await.is_err());
    assert_eq!(wrapper.budget().unwrap().available(), 500 - 2 * 14);
    assert_eq!(detached.budget().unwrap().available(), 500 - 2 * 14);
    assert_eq!(clone.budget().unwrap().available(), 500 - 2 * 14);
}

#[tokio::test]
async fn no_budget_opts_out() {
    let (provider, upstream) = outage(|| Error::overloaded("down"));
    let wrapper = RetryingProvider::new(provider, policy().no_budget());
    assert!(wrapper.budget().is_none());
    for _ in 0..CALLS {
        assert!(wrapper.chat(&request()).await.is_err());
    }
    assert_eq!(upstream.load(Ordering::SeqCst), 3 * CALLS);
}

#[test]
fn make_budget_follows_the_setting() {
    // A new, full budget each time: a custom wrapper owns the one it made.
    let policy = RetryPolicy::new();
    let first = policy.make_budget().unwrap();
    let second = policy.make_budget().unwrap();
    let _ = first.try_acquire(&Error::overloaded("x")).unwrap();
    assert_eq!(first.available(), 500 - 14);
    assert_eq!(second.available(), 500);

    // The shared budget itself.
    let shared = RetryBudget::new();
    let policy = RetryPolicy::new().budget(shared.clone());
    let _ = policy
        .make_budget()
        .unwrap()
        .try_acquire(&Error::overloaded("x"))
        .unwrap();
    assert_eq!(shared.available(), 500 - 14);

    assert!(RetryPolicy::new().no_budget().make_budget().is_none());
}

/// A custom streaming retry wrapper keeps the permit of the retry that
/// started an attempt in its stream state, by name, until the stream
/// reaches its first output (ADR 0011: a stream counts as a success there).
#[test]
fn a_custom_wrapper_names_the_permit_in_its_state() {
    struct StreamState {
        permit: Option<RetryPermit>,
    }

    let budget = RetryPolicy::new().make_budget().unwrap();
    let mut state = StreamState { permit: None };
    state.permit = budget.try_acquire(&Error::overloaded("x"));
    assert!(state.permit.is_some());
    assert_eq!(budget.available(), 500 - 14);
    assert_eq!(
        format!("{:?}", state.permit.as_ref().unwrap()),
        "RetryPermit { .. }"
    );

    // The first output: the retry succeeded, and its cost comes back.
    budget.record_success(state.permit.take());
    assert_eq!(budget.available(), 500);
}

/// Caveat: a wrapper built per request gets a fresh budget each time, so
/// it never limits retries across calls.
#[tokio::test]
async fn a_wrapper_built_per_request_has_no_protection() {
    let (provider, upstream) = outage(|| Error::overloaded("down"));
    for _ in 0..CALLS {
        let wrapper = RetryingProvider::new(provider.clone(), policy());
        assert!(wrapper.chat(&request()).await.is_err());
    }
    assert_eq!(upstream.load(Ordering::SeqCst), 3 * CALLS);
}

/// Caveat: refill is slow at low volume. Once drained, a default budget
/// affords one more transient retry per 14 first-try successes.
#[tokio::test]
async fn refill_is_slow_at_low_volume() {
    let down = Arc::new(AtomicBool::new(true));
    let calls = Arc::new(AtomicUsize::new(0));
    let provider = {
        let (down, calls) = (Arc::clone(&down), Arc::clone(&calls));
        FunctionProvider::new(move |_| {
            calls.fetch_add(1, Ordering::SeqCst);
            if down.load(Ordering::SeqCst) {
                Err(Error::overloaded("down"))
            } else {
                Ok(ChatResponse::default())
            }
        })
    };
    let wrapper = RetryingProvider::new(provider, policy());
    let budget = wrapper.budget().unwrap().clone();
    // Upstream calls one failing call makes, after `successes` healthy ones.
    let failing_call_after = |successes: usize| {
        let (down, calls, wrapper) = (Arc::clone(&down), Arc::clone(&calls), &wrapper);
        async move {
            down.store(false, Ordering::SeqCst);
            for _ in 0..successes {
                wrapper.chat(&request()).await.unwrap();
            }
            down.store(true, Ordering::SeqCst);
            let before = calls.load(Ordering::SeqCst);
            assert!(wrapper.chat(&request()).await.is_err());
            calls.load(Ordering::SeqCst) - before
        }
    };

    // Drain it: 35 retries leave 10 tokens.
    for _ in 0..CALLS {
        assert!(wrapper.chat(&request()).await.is_err());
    }
    assert_eq!(budget.available(), 10);
    // Four successes make 14: one retry, which spends it all.
    assert_eq!(failing_call_after(4).await, 2);
    assert_eq!(budget.available(), 0);
    // Thirteen are not enough for the next one; the fourteenth is.
    assert_eq!(failing_call_after(13).await, 1);
    assert_eq!(failing_call_after(1).await, 2);
}
