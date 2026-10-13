//! Tool call ids the adapter makes up for an endpoint that omits them stay
//! unique across a conversation, so the history passes
//! `history::validate(.., Sendable)` and providers that reject a reused id
//! (Anthropic) accept it after a failover.

use allama::history::{self, ValidationMode};
use allama::{ChatProvider, ChatRequest, ChatStreamExt, Message};
use allama_http::{ReplayResponse, ReplayTransport};
use allama_openai_compat::Provider;

/// A streamed turn with one function call and no tool call id, from a
/// response with `id` (or none).
fn id_less_turn(response_id: Option<&str>) -> String {
    let id = response_id.map_or_else(String::new, |id| format!(r#""id":"{id}","#));
    [
        format!(
            r#"{{{id}"model":"m","choices":[{{"index":0,"delta":{{"role":"assistant","tool_calls":[{{"index":0,"type":"function","function":{{"name":"get_weather","arguments":""}}}}]}}}}]}}"#
        ),
        format!(
            r#"{{{id}"choices":[{{"index":0,"delta":{{"tool_calls":[{{"index":0,"function":{{"arguments":"{{}}"}}}}]}}}}]}}"#
        ),
        format!(r#"{{{id}"choices":[{{"index":0,"delta":{{}},"finish_reason":"tool_calls"}}]}}"#),
        "[DONE]".to_owned(),
    ]
    .iter()
    .map(|line| format!("data: {line}\n\n"))
    .collect()
}

/// Stream two tool turns from `turns`, answering each call, and return the
/// conversation.
async fn two_turn_conversation(turns: [String; 2]) -> Vec<Message> {
    let provider = Provider::builder()
        .base_url("http://upstream/v1")
        .provider_name("id-less")
        .http_transport(ReplayTransport::new(turns.map(ReplayResponse::sse)))
        .build()
        .unwrap();
    let mut request = ChatRequest::new("m").user("Weather in Paris?");
    for (city, weather) in [("Tokyo", "Sunny"), ("Rome", "Rain")] {
        let response = provider
            .chat_stream(&request)
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        let call = response.first_tool_call().unwrap().clone();
        request.messages.push(response.to_assistant_message());
        request
            .messages
            .push(Message::tool_result(call.id, call.name, weather));
        request.messages.push(Message::user(format!("And {city}?")));
    }
    request.messages
}

fn call_ids(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|message| match message {
            Message::Assistant(turn) => Some(turn.tool_calls().map(|call| call.id.clone())),
            _ => None,
        })
        .flatten()
        .collect()
}

#[tokio::test]
async fn a_two_turn_id_less_conversation_is_sendable() {
    let messages =
        two_turn_conversation([id_less_turn(Some("r1")), id_less_turn(Some("r2"))]).await;
    assert_eq!(call_ids(&messages), ["call_r1_0", "call_r2_0"]);
    assert_eq!(
        history::validate(&messages, ValidationMode::Sendable),
        Ok(())
    );
}

#[tokio::test]
async fn a_two_turn_conversation_without_response_ids_is_sendable() {
    let messages = two_turn_conversation([id_less_turn(None), id_less_turn(None)]).await;
    let ids = call_ids(&messages);
    assert_eq!(ids.len(), 2);
    assert_ne!(ids[0], ids[1]);
    assert_eq!(
        history::validate(&messages, ValidationMode::Sendable),
        Ok(())
    );
}
