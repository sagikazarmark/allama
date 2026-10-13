//! The honest-degradation, error-kind, and structured-output contracts check
//! `chat_stream` as well as `chat`: a provider that behaves correctly in
//! `chat` but not when streaming fails the stream variants.

use std::borrow::Cow;

use allama::adapter::Degradations;
use allama::mock::{MockProvider, MockStreamEvent, MockStreamingProvider};
use allama::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, ChatStream,
    ChatStreamExt, ContentBlock, Error, ProviderIdentity, ReasoningEffort, Result, StreamEvent,
    UnsupportedPolicy,
};
use allama_conformance::capabilities::assert_unsupported_chat_capabilities_refused;
use allama_conformance::contract::{
    assert_auth_error_contract, assert_context_length_exceeded_contract,
    assert_overloaded_error_contract, assert_quota_exhausted_contract,
    assert_rate_limited_error_contract, assert_strict_policy_rejects,
    assert_strict_policy_rejects_stream, assert_structured_output_contract_stream,
    assert_unsupported_hint_warns, assert_unsupported_hint_warns_stream, collect_chat_stream,
};
use allama_conformance::probe::ProbeOutcome;
use serde_json::json;

/// What the stream path of [`Reasonless`] gets wrong. `None` is honest.
#[derive(Clone, Copy, PartialEq, Eq)]
enum StreamFault {
    None,
    /// Skips the capability check: the hint is dropped without a warning.
    DropsWarning,
    /// Checks capabilities as if the policy were always `Warn`.
    IgnoresPolicy,
    /// Streams output before failing under `UnsupportedPolicy::Error`.
    RefusesLate,
}

/// A provider whose model has no reasoning configuration. `chat` runs
/// `allama::adapter::chat::preflight`, so it warns under `Warn` and fails under
/// `Error`; `chat_stream` does the same unless `fault` breaks it.
struct Reasonless {
    fault: StreamFault,
}

impl ProviderIdentity for Reasonless {
    fn provider_name(&self) -> &str {
        "reasonless"
    }

    fn reads_namespace(&self, namespace: &str) -> bool {
        namespace == "reasonless"
    }
}

impl ChatProvider for Reasonless {
    type Stream = ChatStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let mut response = ChatResponse::new(vec![ContentBlock::text("ok")]);
        response.warnings =
            self.preflight(request, request.unsupported_policy.unwrap_or_default())?;
        Ok(response)
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        let strict = request.unsupported_policy == Some(UnsupportedPolicy::Error);
        let warnings = match self.fault {
            StreamFault::None => {
                self.preflight(request, request.unsupported_policy.unwrap_or_default())?
            }
            StreamFault::DropsWarning => Vec::new(),
            StreamFault::IgnoresPolicy => self.preflight(request, UnsupportedPolicy::Warn)?,
            StreamFault::RefusesLate if strict && request.reasoning.is_some() => {
                let text = ChatResponse::new(vec![ContentBlock::text("ok")])
                    .into_stream_events()
                    .take(2)
                    .map(Ok)
                    .chain([Err(Error::unsupported("reasoning is not supported"))]);
                return Ok(Box::pin(futures_util::stream::iter(
                    text.collect::<Vec<_>>(),
                )));
            }
            StreamFault::RefusesLate => {
                self.preflight(request, request.unsupported_policy.unwrap_or_default())?
            }
        };
        let events = warnings
            .into_iter()
            .map(StreamEvent::warning)
            .chain(ChatResponse::new(vec![ContentBlock::text("ok")]).into_stream_events())
            .map(Ok)
            .collect::<Vec<_>>();
        Ok(Box::pin(futures_util::stream::iter(events)))
    }

    fn chat_capability(&self, _: &str, capability: ChatCapability) -> CapabilitySupport {
        if capability == ChatCapability::ReasoningConfig {
            CapabilitySupport::Unsupported
        } else {
            CapabilitySupport::Supported
        }
    }
}

impl Reasonless {
    fn preflight(
        &self,
        request: &ChatRequest,
        policy: UnsupportedPolicy,
    ) -> Result<Vec<allama::Warning>> {
        let mut degradations = Degradations::new("reasonless", policy);
        let mut request = Cow::Borrowed(request);
        allama::adapter::chat::preflight(self, &mut request, &mut degradations)?;
        Ok(degradations.into_warnings())
    }
}

fn hint() -> ChatRequest {
    ChatRequest::new("m")
        .user("hi")
        .reasoning(ReasoningEffort::High)
}

#[tokio::test]
async fn an_honest_provider_passes_both_paths() {
    let provider = Reasonless {
        fault: StreamFault::None,
    };
    assert_unsupported_hint_warns(&provider, &hint()).await;
    assert_unsupported_hint_warns_stream(&provider, &hint()).await;
    assert_strict_policy_rejects(&provider, &hint()).await;
    assert_strict_policy_rejects_stream(&provider, &hint()).await;
}

#[tokio::test]
#[should_panic(expected = "unsupported_hint_stream: expected an Unsupported StreamEvent::Warning")]
async fn dropping_the_warning_only_when_streaming_fails() {
    let provider = Reasonless {
        fault: StreamFault::DropsWarning,
    };
    assert_unsupported_hint_warns(&provider, &hint()).await;
    assert_unsupported_hint_warns_stream(&provider, &hint()).await;
}

#[tokio::test]
#[should_panic(expected = "strict_policy_stream: expected the stream to fail with Unsupported")]
async fn ignoring_the_error_policy_only_when_streaming_fails() {
    let provider = Reasonless {
        fault: StreamFault::IgnoresPolicy,
    };
    assert_strict_policy_rejects(&provider, &hint()).await;
    assert_strict_policy_rejects_stream(&provider, &hint()).await;
}

#[tokio::test]
#[should_panic(
    expected = "strict_policy_stream: expected the stream to fail with Unsupported \
                           before any event"
)]
async fn refusing_after_output_fails() {
    let provider = Reasonless {
        fault: StreamFault::RefusesLate,
    };
    // Collected, the late refusal still ends in `Unsupported`.
    let collected = provider
        .chat_stream(&hint().unsupported_policy(UnsupportedPolicy::Error))
        .await
        .unwrap()
        .collect_response()
        .await;
    assert!(matches!(collected, Err(Error::Unsupported { .. })));
    assert_strict_policy_rejects_stream(&provider, &hint()).await;
}

async fn chat_probe(fault: StreamFault, request: ChatRequest) -> ProbeOutcome<ChatResponse> {
    let result = Reasonless { fault }.chat(&request).await;
    let sent = if result.is_ok() {
        vec![json!({"sent": true})]
    } else {
        Vec::new()
    };
    ProbeOutcome::new(result, sent)
}

async fn stream_probe(fault: StreamFault, request: ChatRequest) -> ProbeOutcome<ChatResponse> {
    let result = match (Reasonless { fault }).chat_stream(&request).await {
        Ok(stream) => stream.collect_response().await,
        Err(err) => Err(err),
    };
    let sent = if result.is_ok() {
        vec![json!({"sent": true})]
    } else {
        Vec::new()
    };
    ProbeOutcome::new(result, sent)
}

#[tokio::test]
async fn the_capability_audit_passes_an_honest_stream() {
    let fault = StreamFault::None;
    assert_unsupported_chat_capabilities_refused(
        &Reasonless { fault },
        "m",
        &[],
        |request| chat_probe(fault, request),
        |request| stream_probe(fault, request),
    )
    .await;
}

#[tokio::test]
#[should_panic(
    expected = "chat_stream: ReasoningConfig is reported Unsupported, but a request \
                           using it was sent without a warning"
)]
async fn the_capability_audit_catches_a_silent_stream() {
    let fault = StreamFault::DropsWarning;
    assert_unsupported_chat_capabilities_refused(
        &Reasonless { fault },
        "m",
        &[],
        |request| chat_probe(fault, request),
        |request| stream_probe(fault, request),
    )
    .await;
}

/// A provider whose every call fails with `error`.
fn failing(error: Error) -> MockProvider {
    MockProvider::with_error(error)
}

#[tokio::test]
async fn error_kinds_classified_on_both_paths_pass() {
    let request = ChatRequest::new("m").user("hi");
    let auth = || Error::auth("HTTP 401");
    assert_auth_error_contract(&failing(auth()), &request, |request| async move {
        collect_chat_stream(&failing(auth()), request).await
    })
    .await;
    let rate = || Error::rate_limited("HTTP 429");
    assert_rate_limited_error_contract(&failing(rate()), &request, |request| async move {
        collect_chat_stream(&failing(rate()), request).await
    })
    .await;
    let overloaded = || Error::overloaded("HTTP 503");
    assert_overloaded_error_contract(&failing(overloaded()), &request, |request| async move {
        collect_chat_stream(&failing(overloaded()), request).await
    })
    .await;
    let context = || Error::context_length_exceeded("HTTP 400", Some(8192));
    assert_context_length_exceeded_contract(&failing(context()), &request, |request| async move {
        collect_chat_stream(&failing(context()), request).await
    })
    .await;
    let quota = || Error::quota_exhausted("HTTP 429");
    assert_quota_exhausted_contract(&failing(quota()), &request, |request| async move {
        collect_chat_stream(&failing(quota()), request).await
    })
    .await;
}

#[tokio::test]
async fn an_error_kind_in_the_first_stream_item_passes() {
    let request = ChatRequest::new("m").user("hi");
    let stream = MockStreamingProvider::build(|b| {
        b.stream([MockStreamEvent::Error(Error::rate_limited("HTTP 429"))])
    });
    assert_rate_limited_error_contract(
        &failing(Error::rate_limited("HTTP 429")),
        &request,
        |request| async move { collect_chat_stream(&stream, request).await },
    )
    .await;
}

#[tokio::test]
#[should_panic(expected = "rate_limited (chat_stream): expected Error::RateLimited, got Provider")]
async fn an_error_kind_classified_differently_when_streaming_fails() {
    let request = ChatRequest::new("m").user("hi");
    assert_rate_limited_error_contract(
        &failing(Error::rate_limited("HTTP 429")),
        &request,
        |request| async move {
            collect_chat_stream(&failing(Error::provider("HTTP 429")), request).await
        },
    )
    .await;
}

#[tokio::test]
#[should_panic(expected = "auth_error (chat_stream): expected the call to fail with Error::Auth")]
async fn a_stream_that_succeeds_against_an_error_fails() {
    let request = ChatRequest::new("m").user("hi");
    assert_auth_error_contract(
        &failing(Error::auth("HTTP 401")),
        &request,
        |request| async move {
            collect_chat_stream(&MockStreamingProvider::with_text("ok"), request).await
        },
    )
    .await;
}

#[tokio::test]
async fn structured_output_streamed_as_json_passes() {
    let request = ChatRequest::new("m").user("hi");
    let provider = MockStreamingProvider::with_text(r#"{"city":"Paris"}"#);
    assert_structured_output_contract_stream(&provider, &request).await;
}

#[tokio::test]
#[should_panic(expected = "structured_output_stream: response text is not valid JSON")]
async fn structured_output_streamed_as_prose_fails() {
    let request = ChatRequest::new("m").user("hi");
    let provider = MockStreamingProvider::with_text("The city is Paris.");
    assert_structured_output_contract_stream(&provider, &request).await;
}
