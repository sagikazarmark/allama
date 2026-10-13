//! Request honesty beyond the contract aggregate (which runs in
//! `src/conformance_tests.rs`): preflight follows overridden capability
//! answers, and validation refuses what the call refuses, on `chat` and
//! `chat_stream`.

use allama::prelude::*;
use allama::{CapabilitySupport, ChatCapability};
use allama_anthropic::Provider;
use allama_conformance::assert_validation_agrees;
use allama_conformance::probe::ProbeOutcome;
use allama_http::{ReplayResponse, ReplayTransport};
use futures_util::StreamExt;
use serde_json::{Value, json};

fn ok() -> ReplayResponse {
    ReplayResponse::json(
        200,
        json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn", "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 1}
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
        .map(|body| serde_json::from_slice::<Value>(body).expect("JSON request body"))
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
    let model = "claude-sonnet-4-5";
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
    let transport = ReplayTransport::new([ok()]);
    let lenient = provider(&transport).with_chat_capabilities(overrides);
    let validation = lenient.validate_chat_request(&hint).unwrap();
    let response = lenient.chat(&hint).await.unwrap();
    assert_eq!(validation.warnings(), response.warnings);
    assert!(
        response.warnings.iter().any(is_reasoning_warning),
        "{:?}",
        response.warnings
    );
    let body: Value = serde_json::from_slice(transport.requests()[0].body.as_ref().unwrap())
        .expect("JSON request body");
    assert!(body.get("thinking").is_none(), "{body}");
    assert!(body.get("output_config").is_none(), "{body}");

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
        events.iter().any(
            |e| matches!(e, Ok(StreamEvent::Warning { warning, .. }) if is_reasoning_warning(warning))
        ),
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
    let request = ChatRequest::new("claude-sonnet-4-5").user("Hi");
    let err = scoped(&ReplayTransport::new([]))
        .validate_chat_request(&request)
        .unwrap_err();
    assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    assert_validation_agrees(
        &scoped(&ReplayTransport::new([])),
        |request| async move {
            let transport = ReplayTransport::new([ok()]);
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

/// Without a resolver, nothing is refused on the model id (ADR 0015):
/// structured output on an older generation and a forced tool choice on a
/// model documented to reject it are sent. An adapter-level capability
/// override feeds the encoder as well as preflight, in both directions: an
/// `Unsupported` answer refuses before sending, a `Supported` one sends.
#[tokio::test]
async fn adapter_overrides_reach_the_encoder() {
    let schema = ResponseFormat::json_schema(
        json!({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": false}),
    );
    let structured = ChatRequest::new("claude-3-5-haiku-latest")
        .user("q")
        .response_format(schema);
    let forced = ChatRequest::new("claude-opus-5-5")
        .user("q")
        .tools(vec![ToolDefinition::new("t", json!({"type": "object"}))])
        .tool_choice(ToolChoice::Required);
    for (request, capability) in [
        (&structured, ChatCapability::StructuredOutput),
        (&forced, ChatCapability::ForcedToolChoice),
    ] {
        let transport = ReplayTransport::new([ok(), ok()]);
        let plain = provider(&transport);
        assert!(
            plain
                .validate_chat_request(request)
                .unwrap()
                .warnings()
                .is_empty()
        );
        plain.chat(request).await.unwrap();

        let upgraded = provider(&transport).with_chat_capabilities(move |_: &str, asked| {
            (asked == capability).then_some(CapabilitySupport::Supported)
        });
        upgraded.chat(request).await.unwrap();
        assert_eq!(bodies(&transport).len(), 2, "{capability:?}");

        let refusing = provider(&transport).with_chat_capabilities(move |_: &str, asked| {
            (asked == capability).then_some(CapabilitySupport::Unsupported)
        });
        assert!(refusing.validate_chat_request(request).is_err());
        assert!(matches!(
            refusing.chat(request).await,
            Err(Error::Unsupported { .. })
        ));
        assert_eq!(bodies(&transport).len(), 2, "{capability:?}: not sent");
    }
    let sent = {
        let transport = ReplayTransport::new([ok()]);
        provider(&transport).chat(&structured).await.unwrap();
        bodies(&transport)
    };
    assert_eq!(sent[0]["output_config"]["format"]["type"], "json_schema");
    let sent = {
        let transport = ReplayTransport::new([ok()]);
        provider(&transport).chat(&forced).await.unwrap();
        bodies(&transport)
    };
    assert_eq!(sent[0]["tool_choice"]["type"], "any");
}

/// Model rules reach the encoder: effort levels the caller declares replace
/// the ones the adapter derives from the id, and validation agrees.
#[tokio::test]
async fn model_rules_reach_the_encoder() {
    use allama_anthropic::{EffortLevel, ModelRules};

    let request = ChatRequest::new("claude-opus-4-6")
        .user("q")
        .reasoning(ReasoningEffort::XHigh);
    let transport = ReplayTransport::new([ok(), ok()]);
    let builtin = provider(&transport);
    let validation = builtin.validate_chat_request(&request).unwrap();
    let response = builtin.chat(&request).await.unwrap();
    assert_eq!(validation.warnings(), response.warnings);
    assert_eq!(response.warnings.len(), 1, "{:?}", response.warnings);

    let ruled = provider(&transport).with_model_rules(|model: &str| {
        (model == "claude-opus-4-6").then(|| {
            ModelRules::new().effort_levels([
                EffortLevel::Low,
                EffortLevel::Medium,
                EffortLevel::High,
                EffortLevel::XHigh,
                EffortLevel::Max,
            ])
        })
    });
    let validation = ruled.validate_chat_request(&request).unwrap();
    let response = ruled.chat(&request).await.unwrap();
    assert!(validation.warnings().is_empty());
    assert!(response.warnings.is_empty(), "{:?}", response.warnings);

    let sent = bodies(&transport);
    assert_eq!(sent[0]["output_config"]["effort"], "max");
    assert_eq!(sent[1]["output_config"]["effort"], "xhigh");
}
