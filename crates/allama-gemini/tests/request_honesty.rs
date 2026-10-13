//! Request honesty beyond the contract aggregates (which run in
//! `src/conformance_tests.rs`): preflight follows overridden capability
//! answers, and validation refuses what the call refuses.

use allama::prelude::*;
use allama_conformance::assert_validation_agrees;
use allama_conformance::probe::ProbeOutcome;
use allama_gemini::Provider;
use allama_http::{ReplayResponse, ReplayTransport};
use futures_util::StreamExt;
use serde_json::{Value, json};

fn chat_ok() -> ReplayResponse {
    ReplayResponse::json(
        200,
        json!({
            "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]},
                            "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
            "modelVersion": "gemini-2.5-flash"
        })
        .to_string(),
    )
}

fn provider(transport: &ReplayTransport) -> Provider {
    Provider::builder()
        .api_key("k")
        .http_transport(transport.clone())
        .build()
        .unwrap()
}

fn bodies(transport: &ReplayTransport) -> Vec<Value> {
    transport
        .requests()
        .iter()
        .filter_map(|r| r.body.as_ref())
        .map(|body| serde_json::from_slice(body).expect("JSON request body"))
        .collect()
}

#[tokio::test]
async fn preflight_applies_capability_answers() {
    use allama::{
        CapabilitySupport, ChatCapability, Error, RequestField, Subject, UnsupportedPolicy,
        WarningKind,
    };

    let overrides = |_: &str, capability| {
        matches!(
            capability,
            ChatCapability::ToolCalls | ChatCapability::ReasoningConfig
        )
        .then_some(CapabilitySupport::Unsupported)
    };
    let model = "gemini-2.5-flash";
    let is_reasoning_warning = |w: &allama::Warning| {
        w.kind == WarningKind::Unsupported
            && w.subject
                == Subject::Field {
                    field: RequestField::Reasoning,
                }
    };

    // An unsupported hint under `Warn`: validation passes, `chat` serves the
    // request with a warning, and the hint never reaches the wire.
    let hint = ChatRequest::new(model)
        .user("hi")
        .reasoning(ReasoningEffort::High);
    let transport = ReplayTransport::new([chat_ok()]);
    let lenient = provider(&transport).with_chat_capabilities(overrides);
    let validation = lenient.validate_chat_request(&hint).unwrap();
    let response = lenient.chat(&hint).await.unwrap();
    assert_eq!(validation.warnings(), response.warnings);
    assert!(
        response.warnings.iter().any(is_reasoning_warning),
        "{:?}",
        response.warnings
    );
    let body = &bodies(&transport)[0];
    assert!(
        body.pointer("/generationConfig/thinkingConfig").is_none(),
        "{body}"
    );

    let transport =
        ReplayTransport::new([ReplayResponse::sse(include_str!("../fixtures/stream.sse"))]);
    let events: Vec<_> = provider(&transport)
        .with_chat_capabilities(overrides)
        .chat_stream(&hint)
        .await
        .unwrap()
        .collect()
        .await;
    assert!(
        events.iter().any(|e| matches!(
            e,
            Ok(StreamEvent::Warning { warning, .. }) if is_reasoning_warning(warning)
        )),
        "{events:?}"
    );

    // The same hint under `Error`, and a semantic feature under any policy:
    // validation, `chat`, and `chat_stream` all refuse before sending.
    let strict = hint.unsupported_policy(UnsupportedPolicy::Error);
    let tools = ChatRequest::new(model)
        .user("hi")
        .tools([ToolDefinition::new("t", json!({"type": "object"}))]);
    for request in [strict, tools] {
        let transport = ReplayTransport::new([]);
        let provider = provider(&transport).with_chat_capabilities(overrides);
        let err = provider.validate_chat_request(&request).unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }), "{err}");
        let err = provider.chat(&request).await.unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }), "{err}");
        assert!(matches!(
            provider.chat_stream(&request).await,
            Err(Error::Unsupported { .. })
        ));
        assert!(transport.requests().is_empty());
    }
}

/// A provider that takes its key from each request: validation refuses a
/// request that carries none, as the call does, before sending.
#[tokio::test]
async fn validation_refuses_a_missing_scoped_credential() {
    fn scoped(transport: &ReplayTransport) -> Provider {
        Provider::builder()
            .credential_scope("tenant")
            .http_transport(transport.clone())
            .build()
            .unwrap()
    }
    let request = ChatRequest::new("gemini-2.5-flash").user("Hi");
    let err = scoped(&ReplayTransport::new([]))
        .validate_chat_request(&request)
        .unwrap_err();
    assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    assert_validation_agrees(
        &scoped(&ReplayTransport::new([])),
        |request| async move {
            let transport = ReplayTransport::new([chat_ok()]);
            let result = scoped(&transport).chat(&request).await;
            ProbeOutcome::new(result, bodies(&transport))
        },
        |request| async move {
            let transport =
                ReplayTransport::new([ReplayResponse::sse(include_str!("../fixtures/stream.sse"))]);
            let result = match scoped(&transport).chat_stream(&request).await {
                Ok(stream) => stream.collect_response().await,
                Err(err) => Err(err),
            };
            ProbeOutcome::new(result, bodies(&transport))
        },
        &[request],
    )
    .await;
}

/// Nothing is refused or dropped because of the model id (ADR 0015): the
/// features a model page says a model lacks (TTS and image models' tools and
/// structured output, Gemini 2.0's thinking, multimodal tool results before
/// Gemini 3, penalties) are sent under the strict policy, with no warning
/// beyond an approximation, and so are they when a resolver answers `Supported` for them. The model
/// refuses upstream; the catalog (later) refuses before sending.
#[tokio::test]
async fn nothing_is_refused_because_of_the_model() {
    use allama::{CapabilitySupport, ContentPart, ToolResultContent, UnsupportedPolicy};

    let request = |model: &str| {
        ChatRequest::new(model)
            .user("Describe the chart.")
            .message(Message::assistant_blocks(vec![ContentBlock::tool_call(
                "c1", "chart", "{}",
            )]))
            .message(Message::tool_result(
                "c1",
                "chart",
                ToolResultContent::Parts(vec![ContentPart::image_base64("image/png", "iVBO")]),
            ))
            .tools([ToolDefinition::new("chart", json!({"type": "object"}))])
            .response_format(ResponseFormat::json_schema(json!({"type": "object"})))
            .reasoning(ReasoningEffort::Low)
            .frequency_penalty(0.5)
            .unsupported_policy(UnsupportedPolicy::Error)
    };
    let upgraded = |_: &str, _| Some(CapabilitySupport::Supported);
    for model in [
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-preview-tts",
        "gemini-2.5-flash-image",
        "gemini-3.1-flash-live-preview",
        "gemma-4-31b-it",
        "tunedModels/my-model",
    ] {
        for resolved in [false, true] {
            let transport = ReplayTransport::new([chat_ok()]);
            let mut gemini = provider(&transport);
            if resolved {
                gemini = gemini.with_chat_capabilities(upgraded);
            }
            let response = gemini.chat(&request(model)).await.unwrap();
            // Only the placeholder signature of the unsigned call (a wire
            // format) may be reported.
            assert!(
                response
                    .warnings
                    .iter()
                    .all(|w| w.kind == allama::WarningKind::Approximated),
                "{model}: {:?}",
                response.warnings
            );
            let body = &bodies(&transport)[0];
            for pointer in [
                "/tools/0/functionDeclarations/0",
                "/generationConfig/responseJsonSchema",
                "/generationConfig/thinkingConfig",
                "/generationConfig/frequencyPenalty",
                "/contents/2/parts/0/functionResponse/parts/0/inlineData",
            ] {
                assert!(body.pointer(pointer).is_some(), "{model} {pointer}: {body}");
            }
        }
    }
}
