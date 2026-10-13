//! Provider-agnostic behavioral contract helpers.
//!
//! Each helper takes any [`ChatProvider`] plus a pre-built [`ChatRequest`]
//! and verifies that the provider's response honors the portable contract:
//! the finish reason is the right variant, tool-call blocks carry valid
//! JSON arguments, streaming events reconstruct into a coherent response,
//! and so on.
//!
//! The helpers are provider-agnostic by construction. Provider crates wire
//! a `TestHttpServer` (feature `server`) with provider-specific
//! fixture responses, build their `Provider` against the server's URL,
//! and then call these helpers so the same behavioral contract is
//! enforced uniformly across the provider matrix.
//!
//! The error-kind contracts (`assert_auth_error_contract` and its siblings)
//! also take a stream probe, which sends the request with `chat_stream`, so
//! a failure is classified the same on both paths;
//! [`collect_chat_stream`] is the usual one. Errors reported inside a
//! stream that has started are checked by
//! [`assert_stream_errors_classified`](crate::assert_stream_errors_classified),
//! and a success response whose body cannot be decoded by
//! [`assert_malformed_response_contract`].
//!
//! For live-API checks that build the request internally, see
//! [`crate::e2e`].
//!
//! # Example
//!
//! ```no_run
//! use allama::prelude::*;
//! use allama_conformance::contract::assert_chat_success_contract;
//!
//! # async fn example<P: ChatProvider>(provider: &P) {
//! let request = ChatRequest::new("my-model").user("say hi");
//! // `provider` is wired to a transport (for example a `TestHttpServer`)
//! // that answers with the provider's success fixture.
//! assert_chat_success_contract(provider, &request).await;
//! # }
//! ```

use std::future::Future;

use allama::{
    ChatProvider, ChatRequest, ChatResponse, ChatStreamExt, Error, FinishReason, StreamEvent,
};
use futures_util::StreamExt;

/// Assert a successful non-streaming chat: response has text, Stop finish
/// reason, and non-empty content.
pub async fn assert_chat_success_contract(provider: &impl ChatProvider, request: &ChatRequest) {
    let response = provider
        .chat(request)
        .await
        .expect("chat_success: chat request failed");
    let text = response
        .text()
        .expect("chat_success: response has no text block");
    assert!(!text.is_empty(), "chat_success: response text is empty");
    assert!(
        response.provider.is_some(),
        "chat_success: adapters must stamp the serving provider"
    );
    if let Some(usage) = &response.usage {
        crate::assert_usage_contract(usage);
    }
    assert_eq!(
        response.finish_reason,
        Some(FinishReason::Stop),
        "chat_success: expected Stop finish reason, got {:?}",
        response.finish_reason
    );
}

/// Assert a successful streaming chat: events arrive, the collector
/// reconstructs a response with text and a Stop finish reason.
pub async fn assert_streaming_chat_contract(provider: &impl ChatProvider, request: &ChatRequest) {
    let stream = provider
        .chat_stream(request)
        .await
        .expect("streaming_chat: chat_stream request failed");
    let response = stream
        .collect_response()
        .await
        .expect("streaming_chat: stream collection failed");
    let text = response
        .text()
        .expect("streaming_chat: response has no text block");
    assert!(!text.is_empty(), "streaming_chat: response text is empty");
    assert_eq!(
        response.finish_reason,
        Some(FinishReason::Stop),
        "streaming_chat: expected Stop finish reason, got {:?}",
        response.finish_reason
    );
}

/// Assert a non-streaming tool-calling round trip: the response has tool
/// calls, each call's arguments parse as JSON, and the finish reason is
/// [`FinishReason::ToolCalls`].
pub async fn assert_tool_call_contract(provider: &impl ChatProvider, request: &ChatRequest) {
    let response = provider
        .chat(request)
        .await
        .expect("tool_call: chat request failed");
    assert!(
        response.has_tool_calls(),
        "tool_call: response has no tool calls"
    );
    assert_eq!(
        response.finish_reason,
        Some(FinishReason::ToolCalls),
        "tool_call: expected ToolCalls finish reason, got {:?}",
        response.finish_reason
    );
    for call in response.tool_calls() {
        assert!(!call.name.is_empty(), "tool_call: empty tool name");
        assert!(
            serde_json::from_str::<serde_json::Value>(&call.arguments).is_ok(),
            "tool_call: arguments not valid JSON for {}: {}",
            call.name,
            call.arguments
        );
    }
}

/// Assert a streaming tool-calling round trip: the stream reconstructs
/// into a response with tool calls, each call's arguments parse as JSON,
/// and the finish reason is [`FinishReason::ToolCalls`].
///
/// A reconstructed tool call is evidence that the stream emitted the
/// underlying `BlockStart(ToolCall)` / `ToolCallDelta` / `BlockStop`
/// sequence; the collector would otherwise reject the event shape.
pub async fn assert_streaming_tool_call_contract(
    provider: &impl ChatProvider,
    request: &ChatRequest,
) {
    let stream = provider
        .chat_stream(request)
        .await
        .expect("streaming_tool_call: chat_stream request failed");
    let response = stream
        .collect_response()
        .await
        .expect("streaming_tool_call: stream collection failed");
    assert!(
        response.has_tool_calls(),
        "streaming_tool_call: collected response has no tool calls"
    );
    assert_eq!(
        response.finish_reason,
        Some(FinishReason::ToolCalls),
        "streaming_tool_call: expected ToolCalls finish reason, got {:?}",
        response.finish_reason
    );
    for call in response.tool_calls() {
        assert!(
            !call.name.is_empty(),
            "streaming_tool_call: empty tool name"
        );
        assert!(
            serde_json::from_str::<serde_json::Value>(&call.arguments).is_ok(),
            "streaming_tool_call: arguments not valid JSON for {}: {}",
            call.name,
            call.arguments
        );
    }
}

/// Assert a structured-output response: the text is valid JSON.
///
/// [`assert_structured_output_contract_stream`] checks the same on
/// `chat_stream`; run both.
pub async fn assert_structured_output_contract(
    provider: &impl ChatProvider,
    request: &ChatRequest,
) {
    let response = provider
        .chat(request)
        .await
        .expect("structured_output: chat request failed");
    assert_json_text("structured_output", &response);
}

/// [`assert_structured_output_contract`] on `chat_stream`: the collected
/// response's text is valid JSON.
pub async fn assert_structured_output_contract_stream(
    provider: &impl ChatProvider,
    request: &ChatRequest,
) {
    let response = collect_chat_stream(provider, request.clone())
        .await
        .expect("structured_output_stream: chat_stream request failed");
    assert_json_text("structured_output_stream", &response);
}

/// Send `request` with `chat_stream` and collect the stream into a response:
/// the stream probe of the error-kind contracts for a provider whose
/// transport answers the streaming request too.
///
/// An error from `chat_stream` itself, or from any item of the stream, is
/// the result's error.
///
/// ```no_run
/// use allama::ChatRequest;
/// use allama_conformance::contract::{assert_auth_error_contract, collect_chat_stream};
///
/// # async fn example<P: allama::ChatProvider>(provider: &P) {
/// // `provider`'s transport answers both calls with an authentication error.
/// let request = ChatRequest::new("my-model").user("hi");
/// assert_auth_error_contract(provider, &request, |request| {
///     collect_chat_stream(provider, request)
/// })
/// .await;
/// # }
/// ```
pub async fn collect_chat_stream(
    provider: &impl ChatProvider,
    request: ChatRequest,
) -> allama::Result<ChatResponse> {
    provider
        .chat_stream(&request)
        .await?
        .collect_response()
        .await
}

/// Assert that a request against an authentication-failure fixture
/// surfaces as [`allama::Error::Auth`], on `chat` and on `chat_stream`.
///
/// `provider.chat(request)` runs first, then `stream_probe(request)`, which
/// sends the request with `chat_stream` through a provider answering with
/// the same failure and collects the stream ([`collect_chat_stream`] does
/// this for `provider`; its transport then answers twice). An error at
/// stream setup and one in the first stream item both count.
pub async fn assert_auth_error_contract<F, Fut>(
    provider: &impl ChatProvider,
    request: &ChatRequest,
    stream_probe: F,
) where
    F: FnOnce(ChatRequest) -> Fut,
    Fut: Future<Output = allama::Result<ChatResponse>>,
{
    let expected = ErrorKind {
        contract: "auth_error",
        name: "Auth",
        matches: |err| matches!(err, Error::Auth { .. }),
        retryable: false,
        transient: None,
    };
    expected.check(provider, request, stream_probe).await;
}

/// Assert that a request against a rate-limited-failure fixture surfaces
/// as [`allama::Error::RateLimited`] and classifies as retryable and
/// transient, on `chat` and on `chat_stream`.
///
/// `stream_probe` works as for [`assert_auth_error_contract`].
pub async fn assert_rate_limited_error_contract<F, Fut>(
    provider: &impl ChatProvider,
    request: &ChatRequest,
    stream_probe: F,
) where
    F: FnOnce(ChatRequest) -> Fut,
    Fut: Future<Output = allama::Result<ChatResponse>>,
{
    let expected = ErrorKind {
        contract: "rate_limited",
        name: "RateLimited",
        matches: |err| matches!(err, Error::RateLimited { .. }),
        retryable: true,
        transient: Some(true),
    };
    expected.check(provider, request, stream_probe).await;
}

/// Assert that a request against an overloaded-failure fixture surfaces
/// as [`allama::Error::Overloaded`] and classifies as retryable and
/// transient, on `chat` and on `chat_stream`.
///
/// `stream_probe` works as for [`assert_auth_error_contract`].
pub async fn assert_overloaded_error_contract<F, Fut>(
    provider: &impl ChatProvider,
    request: &ChatRequest,
    stream_probe: F,
) where
    F: FnOnce(ChatRequest) -> Fut,
    Fut: Future<Output = allama::Result<ChatResponse>>,
{
    let expected = ErrorKind {
        contract: "overloaded",
        name: "Overloaded",
        matches: |err| matches!(err, Error::Overloaded { .. }),
        retryable: true,
        transient: Some(true),
    };
    expected.check(provider, request, stream_probe).await;
}

/// Assert that a request against a context-length-exceeded fixture
/// surfaces as [`allama::Error::ContextLengthExceeded`], on `chat` and on
/// `chat_stream`.
///
/// `stream_probe` works as for [`assert_auth_error_contract`].
pub async fn assert_context_length_exceeded_contract<F, Fut>(
    provider: &impl ChatProvider,
    request: &ChatRequest,
    stream_probe: F,
) where
    F: FnOnce(ChatRequest) -> Fut,
    Fut: Future<Output = allama::Result<ChatResponse>>,
{
    let expected = ErrorKind {
        contract: "context_length",
        name: "ContextLengthExceeded",
        matches: |err| matches!(err, Error::ContextLengthExceeded { .. }),
        retryable: false,
        transient: None,
    };
    expected.check(provider, request, stream_probe).await;
}

/// The provider maps quota or billing exhaustion to a non-retryable
/// [`Error::QuotaExhausted`], on `chat` and on `chat_stream`.
///
/// `stream_probe` works as for [`assert_auth_error_contract`].
pub async fn assert_quota_exhausted_contract<F, Fut>(
    provider: &impl ChatProvider,
    request: &ChatRequest,
    stream_probe: F,
) where
    F: FnOnce(ChatRequest) -> Fut,
    Fut: Future<Output = allama::Result<ChatResponse>>,
{
    let expected = ErrorKind {
        contract: "quota_exhausted",
        name: "QuotaExhausted",
        matches: |err| matches!(err, Error::QuotaExhausted { .. }),
        retryable: false,
        transient: None,
    };
    expected.check(provider, request, stream_probe).await;
}

/// What a proxy or gateway answers in place of a provider's JSON: a body
/// for the transport behind [`assert_malformed_response_contract`].
pub const MALFORMED_BODY: &str = "<html><body><h1>502 Bad Gateway</h1>SENTINEL</body></html>";

/// Assert that a success response whose body the adapter cannot decode
/// surfaces as [`Error::UnexpectedResponse`], on `chat`.
///
/// `provider`'s transport answers `200` with a body that is no response of
/// the provider's ([`MALFORMED_BODY`], for example) and with `request_id` in
/// the header the adapter reads its request id from (`None` for a provider
/// with no request id). The error must:
///
/// - be neither retryable nor eligible for fallback
///   ([`should_fallback`](allama::wrappers::should_fallback)): the same
///   response comes back, and another provider would not fix it;
/// - name the provider ([`provider_name`](allama::ProviderIdentity::provider_name)),
///   the status `200`, and `request_id`, so the response can be traced;
/// - keep the body out of the redacted
///   [`as_log`](allama::Error::as_log) view (checked with
///   [`MALFORMED_BODY`]'s `SENTINEL`).
///
/// [`Error::Serialization`] is for the caller's own (de)serialization, never
/// for what a provider sent.
///
/// # Panics
///
/// When `chat` succeeds or its error breaks one of the rules above.
pub async fn assert_malformed_response_contract(
    provider: &impl ChatProvider,
    request: &ChatRequest,
    request_id: Option<&str>,
) {
    check_malformed_response(
        provider.provider_name(),
        provider.chat(request).await,
        request_id,
    );
}

/// The rules of [`assert_malformed_response_contract`], for `result`, what
/// `chat` of the provider named `provider_name` returned for a malformed
/// body.
pub(crate) fn check_malformed_response(
    provider_name: &str,
    result: allama::Result<ChatResponse>,
    request_id: Option<&str>,
) {
    const CONTRACT: &str = "malformed_response";
    let err = match result {
        Ok(response) => {
            panic!("{CONTRACT}: expected Error::UnexpectedResponse, got a response: {response:?}")
        }
        Err(err) => err,
    };
    assert!(
        matches!(err, Error::UnexpectedResponse { .. }),
        "{CONTRACT}: expected Error::UnexpectedResponse, got {err:?}"
    );
    assert!(
        !err.is_retryable(),
        "{CONTRACT}: a malformed body should not be is_retryable()"
    );
    assert!(
        !allama::wrappers::should_fallback(&err),
        "{CONTRACT}: a malformed body should not be eligible for fallback"
    );
    let context = err.context();
    assert_eq!(
        context.and_then(|context| context.provider.as_deref()),
        Some(provider_name),
        "{CONTRACT}: the error should name the provider: {err:?}"
    );
    assert_eq!(
        err.status(),
        Some(200),
        "{CONTRACT}: the error should carry the response's status: {err:?}"
    );
    assert_eq!(
        err.request_id(),
        request_id,
        "{CONTRACT}: the error should carry the response's request id: {err:?}"
    );
    let log = err.as_log().to_string();
    assert!(
        !log.contains("SENTINEL"),
        "{CONTRACT}: the body reached the redacted log: {log}"
    );
}

/// The provider reports dropped or approximated request features as
/// warnings rather than silently ignoring them, on `chat`.
///
/// `request` uses a feature the provider cannot honour exactly (a hint, in
/// the sense of ADR 0003). [`assert_unsupported_hint_warns_stream`] checks
/// the same on `chat_stream`; run both.
pub async fn assert_unsupported_hint_warns(provider: &impl ChatProvider, request: &ChatRequest) {
    let response = provider
        .chat(request)
        .await
        .expect("unsupported_hint: chat request failed");
    assert!(
        response
            .warnings
            .iter()
            .any(|w| w.kind == allama::WarningKind::Unsupported),
        "unsupported_hint: expected an Unsupported warning, got {:?}",
        response.warnings
    );
}

/// [`assert_unsupported_hint_warns`] on `chat_stream`: the stream completes
/// and carries an `Unsupported` [`StreamEvent::Warning`].
pub async fn assert_unsupported_hint_warns_stream(
    provider: &impl ChatProvider,
    request: &ChatRequest,
) {
    let stream = provider
        .chat_stream(request)
        .await
        .expect("unsupported_hint_stream: chat_stream request failed");
    let mut stream = std::pin::pin!(stream);
    let mut warnings = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamEvent::Warning { warning, .. }) => warnings.push(warning),
            Ok(_) => {}
            Err(err) => panic!("unsupported_hint_stream: stream failed: {err}"),
        }
    }
    assert!(
        warnings
            .iter()
            .any(|w| w.kind == allama::WarningKind::Unsupported),
        "unsupported_hint_stream: expected an Unsupported StreamEvent::Warning, got {warnings:?}"
    );
}

/// Under `UnsupportedPolicy::Error`, the request of
/// [`assert_unsupported_hint_warns`] fails with [`Error::Unsupported`] on
/// `chat`.
///
/// This contract sees only the result, not the transport. That nothing was
/// sent is checked by the probe-driven
/// [`assert_request_fields_accounted`](crate::probe::assert_request_fields_accounted),
/// which reruns every degraded hint under `UnsupportedPolicy::Error`.
/// [`assert_strict_policy_rejects_stream`] checks `chat_stream`; run both.
pub async fn assert_strict_policy_rejects(provider: &impl ChatProvider, request: &ChatRequest) {
    let strict = request
        .clone()
        .unsupported_policy(allama::UnsupportedPolicy::Error);
    let err = provider
        .chat(&strict)
        .await
        .expect_err("strict_policy: expected an Unsupported error");
    assert!(
        matches!(err, Error::Unsupported { .. }),
        "strict_policy: expected Error::Unsupported, got {err:?}"
    );
}

/// [`assert_strict_policy_rejects`] on `chat_stream`: `chat_stream` itself,
/// or the first item of its stream, fails with [`Error::Unsupported`], so
/// no output event arrives before the refusal.
pub async fn assert_strict_policy_rejects_stream(
    provider: &impl ChatProvider,
    request: &ChatRequest,
) {
    let strict = request
        .clone()
        .unsupported_policy(allama::UnsupportedPolicy::Error);
    let err = match provider.chat_stream(&strict).await {
        Err(err) => err,
        Ok(stream) => match std::pin::pin!(stream).next().await {
            Some(Err(err)) => err,
            Some(Ok(event)) => panic!(
                "strict_policy_stream: expected the stream to fail with Unsupported before any \
                 event, got {event:?}"
            ),
            None => {
                panic!("strict_policy_stream: expected an Unsupported error, got an empty stream")
            }
        },
    };
    assert!(
        matches!(err, Error::Unsupported { .. }),
        "strict_policy_stream: expected Error::Unsupported, got {err:?}"
    );
}

/// The error kind an error-kind contract expects, and how it classifies.
struct ErrorKind {
    contract: &'static str,
    name: &'static str,
    matches: fn(&Error) -> bool,
    retryable: bool,
    /// `None` when the contract does not pin it.
    transient: Option<bool>,
}

impl ErrorKind {
    /// Check `chat`, then `stream_probe`.
    async fn check<F, Fut>(
        &self,
        provider: &impl ChatProvider,
        request: &ChatRequest,
        stream_probe: F,
    ) where
        F: FnOnce(ChatRequest) -> Fut,
        Fut: Future<Output = allama::Result<ChatResponse>>,
    {
        self.check_result("chat", provider.chat(request).await);
        self.check_result("chat_stream", stream_probe(request.clone()).await);
    }

    fn check_result(&self, path: &str, result: allama::Result<ChatResponse>) {
        let Self { contract, name, .. } = self;
        let err = match result {
            Ok(_) => panic!("{contract} ({path}): expected the call to fail with Error::{name}"),
            Err(err) => err,
        };
        assert!(
            (self.matches)(&err),
            "{contract} ({path}): expected Error::{name}, got {err:?}"
        );
        let not = |yes: bool| if yes { "" } else { " not" };
        assert_eq!(
            err.is_retryable(),
            self.retryable,
            "{contract} ({path}): {name} should{} be is_retryable()",
            not(self.retryable)
        );
        if let Some(transient) = self.transient {
            assert_eq!(
                err.is_transient(),
                transient,
                "{contract} ({path}): {name} should{} be is_transient()",
                not(transient)
            );
        }
    }
}

/// Panic unless `response`'s text is valid JSON.
fn assert_json_text(contract: &str, response: &ChatResponse) {
    let text = response
        .text()
        .unwrap_or_else(|| panic!("{contract}: response has no text block"));
    serde_json::from_str::<serde_json::Value>(&text).unwrap_or_else(|e| {
        panic!("{contract}: response text is not valid JSON: {e}\nraw: {text}")
    });
}
