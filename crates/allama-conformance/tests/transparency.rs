//! Every built-in wrapper and type-erased provider is transparent (C4): it
//! forwards identity, capabilities, limits, and validation, and passes the
//! provider's responses, streams and errors through unchanged.

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Duration;

use allama::mock::MockProvider;
use allama::wrappers::{
    CapabilityOverrideProvider, FallbackProvider, MapModelProvider, MapRequestProvider,
    ObserveProvider, RecordingProvider, RetryPolicy, RetryingProvider, TimeoutPolicy,
    TimeoutProvider, TracingProvider, UnsupportedPolicyProvider,
};
use allama::{
    ChatProvider, ChatRequest, DynChatProvider, DynDecisionProvider, DynEmbeddingProvider,
    DynExtractionProvider, ProviderIdentityExt, UnsupportedPolicy,
};
use allama_conformance::{
    TransparencyProbe, assert_chat_wrapper, assert_decision_wrapper, assert_embedding_wrapper,
    assert_extraction_wrapper, assert_transparent_chat_data_path, assert_transparent_model_mapping,
    assert_wrapper,
};

/// Retries without waiting: the data path's retryable errors come back on
/// every attempt.
fn retrying<P>(probe: P) -> RetryingProvider<P> {
    RetryingProvider::new(
        probe,
        RetryPolicy::new()
            .base_delay(Duration::ZERO)
            .max_delay(Duration::ZERO)
            .jitter(false),
    )
}

#[tokio::test]
async fn wrappers_are_transparent() {
    assert_wrapper(retrying).await;
    assert_wrapper(|probe| TimeoutProvider::new(probe, TimeoutPolicy::new())).await;
    assert_wrapper(TracingProvider::new).await;
    assert_wrapper(|probe| RecordingProvider::new(probe, |_| {})).await;
    assert_wrapper(|probe| ObserveProvider::new(probe).on_warning(|_, _| {})).await;
    assert_wrapper(|probe| UnsupportedPolicyProvider::new(probe, UnsupportedPolicy::Error)).await;
}

/// The built-in wrappers forward the unstable extraction operation too;
/// `assert_wrapper` leaves it to this check (ADR 0018).
#[tokio::test]
async fn wrappers_forward_extraction_transparently() {
    assert_extraction_wrapper(retrying).await;
    assert_extraction_wrapper(|probe| TimeoutProvider::new(probe, TimeoutPolicy::new())).await;
    assert_extraction_wrapper(TracingProvider::new).await;
    assert_extraction_wrapper(|probe| RecordingProvider::new(probe, |_| {})).await;
    assert_extraction_wrapper(|probe| ObserveProvider::new(probe).on_warning(|_, _| {})).await;
    assert_extraction_wrapper(|probe| {
        UnsupportedPolicyProvider::new(probe, UnsupportedPolicy::Error)
    })
    .await;
    assert_extraction_wrapper(|probe| MapRequestProvider::new(probe, |_| {})).await;
    assert_extraction_wrapper(|probe| MapModelProvider::new(probe, |model: &str| model.into()))
        .await;
    assert_extraction_wrapper(|probe| FallbackProvider::new(probe, probe)).await;
    assert_extraction_wrapper(Box::new).await;
    assert_extraction_wrapper(Arc::new).await;
    assert_extraction_wrapper(DynExtractionProvider::new).await;
    assert_extraction_wrapper(|probe| {
        TracingProvider::new(FallbackProvider::new(
            retrying(TimeoutProvider::new(probe, TimeoutPolicy::new())),
            retrying(probe),
        ))
    })
    .await;
}

/// The default retry policy wraps the probe transparently too: it gives up
/// on the probe's errors with the error itself. (The data path is checked
/// with a policy that does not wait.)
#[test]
fn the_default_retry_policy_is_transparent() {
    allama_conformance::assert_transparent_wrapper(|probe| {
        RetryingProvider::new(probe, RetryPolicy::default())
    });
}

/// A capability override that settles nothing is transparent; its own
/// check of each request passes the provider's data through. It wraps chat
/// providers only.
#[tokio::test]
async fn capability_overrides_are_transparent() {
    assert_chat_wrapper(|probe| CapabilityOverrideProvider::new(probe, |_: &str, _| None)).await;
}

/// `MapRequestProvider` validates the transformed request: it is transparent
/// when the transform keeps the request valid.
#[tokio::test]
async fn request_transforms_are_transparent() {
    assert_wrapper(|probe| MapRequestProvider::new(probe, |_| {})).await;
    assert_wrapper(|probe| {
        MapRequestProvider::new(probe, |request| {
            request.temperature.get_or_insert(0.2);
        })
    })
    .await;
}

/// Swaps the probe's models and maps the second to the one it rejects, so
/// every answer changes under the mapping.
fn shuffle(model: &str) -> Cow<'_, str> {
    let [a, b] = TransparencyProbe::MODELS;
    match model {
        m if m == a => b.into(),
        m if m == b => TransparencyProbe::REJECTED_MODEL.into(),
        _ => a.into(),
    }
}

/// `MapModelProvider` answers for the model it sends: transparent once the
/// mapping is applied, and plainly transparent when it maps nothing. Its
/// calls pass the provider's data through either way.
#[tokio::test]
async fn model_mapping_is_transparent_with_the_mapping_applied() {
    assert_transparent_model_mapping(|probe| MapModelProvider::new(probe, shuffle), shuffle);
    assert_transparent_model_mapping(
        |probe| MapModelProvider::new(probe, |_: &str| TransparencyProbe::MODELS[1].into()),
        |_| TransparencyProbe::MODELS[1].into(),
    );
    assert_transparent_chat_data_path(|probe| {
        MapModelProvider::new(probe, |_: &str| TransparencyProbe::MODELS[0].into())
    })
    .await;
    assert_wrapper(|probe| MapModelProvider::new(probe, |model: &str| model.into())).await;
}

#[tokio::test]
async fn pointers_are_transparent() {
    assert_wrapper(Box::new).await;
    assert_wrapper(Arc::new).await;
}

#[tokio::test]
async fn erased_providers_are_transparent() {
    assert_chat_wrapper(DynChatProvider::new).await;
    assert_embedding_wrapper(DynEmbeddingProvider::new).await;
    assert_decision_wrapper(DynDecisionProvider::new).await;
}

/// `FallbackProvider` combines two providers: it reports the primary's name,
/// validates against both sides, merges capabilities, and takes the smaller
/// limits. With the probe on both sides it is transparent, including when
/// both sides map models the same way; when both sides fail, its error ends
/// in the probe's.
#[tokio::test]
async fn fallback_is_transparent_over_the_same_provider() {
    assert_wrapper(|probe| FallbackProvider::new(probe, probe)).await;
    assert_transparent_model_mapping(
        |probe| {
            FallbackProvider::new(
                MapModelProvider::new(probe, shuffle),
                RetryingProvider::new(
                    MapModelProvider::new(probe, shuffle),
                    RetryPolicy::default(),
                ),
            )
        },
        shuffle,
    );
}

/// A fallback side that maps models is asked about the model it would
/// receive: the backup's dry run reports the rejection of its mapped model.
#[test]
fn fallback_validates_the_model_each_side_receives() {
    fn to_rejected(model: &str) -> Cow<'_, str> {
        if model == TransparencyProbe::MODELS[0] {
            TransparencyProbe::REJECTED_MODEL.into()
        } else {
            model.into()
        }
    }

    let [a, b] = TransparencyProbe::MODELS;
    let fallback = FallbackProvider::new(
        TransparencyProbe::new(),
        MapModelProvider::new(TransparencyProbe::new(), to_rejected),
    );
    let validation = fallback
        .validate_chat_request(&ChatRequest::new(a).user("hi"))
        .unwrap();
    let err = validation.fallback().unwrap().result.as_ref().unwrap_err();
    assert!(
        err.message().contains(TransparencyProbe::REJECTION),
        "{err}"
    );
    let validation = fallback
        .validate_chat_request(&ChatRequest::new(b).user("hi"))
        .unwrap();
    assert!(validation.fallback().unwrap().result.is_ok());
}

#[test]
fn fallback_exposes_both_sides_primary_first() {
    let other = MockProvider::empty();
    let primary = FallbackProvider::new(TransparencyProbe::new(), other.clone());
    assert!(primary.downcast_ref::<TransparencyProbe>().is_some());
    let secondary = FallbackProvider::new(other.clone(), TransparencyProbe::new());
    assert!(secondary.downcast_ref::<TransparencyProbe>().is_some());

    // Same type on both sides: the primary answers.
    let both = FallbackProvider::new(other.clone(), other);
    assert!(std::ptr::eq(
        both.downcast_ref::<MockProvider>().unwrap(),
        both.primary()
    ));
}

#[tokio::test]
async fn stacks_are_transparent() {
    assert_wrapper(|probe| {
        TracingProvider::new(FallbackProvider::new(
            retrying(TimeoutProvider::new(probe, TimeoutPolicy::new())),
            retrying(probe),
        ))
    })
    .await;
    assert_chat_wrapper(|probe| {
        DynChatProvider::new(retrying(TracingProvider::new(TimeoutProvider::new(
            probe,
            TimeoutPolicy::new(),
        ))))
    })
    .await;
}
