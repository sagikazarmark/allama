//! The decoding contracts against a fake adapter whose wire format is a
//! serialized `ChatResponse` (both paths), with injected bugs.

use std::borrow::Cow;

use allama::chat::stream::SingleResponseStream;
use allama::mock::ChatResponseBuilder;
use allama::{ChatResponse, ContentBlock, FinishReason, ProviderData};
use serde_json::json;

use super::*;

/// What a buggy decoder does to a response on one path.
type Bug = fn(&mut ChatResponse);

fn none(_: &mut ChatResponse) {}

fn decode_body(body: &str, bug: Bug) -> ChatResponse {
    let mut response: ChatResponse = serde_json::from_str(body).unwrap();
    bug(&mut response);
    response.provider.get_or_insert(Cow::Borrowed("fake"));
    response
}

/// `chat` of the fake adapter, with `bug` applied.
fn unary(bug: Bug) -> impl Fn(String) -> std::future::Ready<Result<ChatResponse>> {
    move |body| std::future::ready(Ok(decode_body(&body, bug)))
}

/// `chat_stream` of the fake adapter, with `bug` applied.
fn stream(bug: Bug) -> impl Fn(String) -> std::future::Ready<Result<SingleResponseStream>> {
    move |body| std::future::ready(Ok(decode_body(&body, bug).into_stream()))
}

fn capture(response: &ChatResponse) -> ResponseCapture {
    let body = serde_json::to_string(response).unwrap();
    ResponseCapture::new(body.clone(), body)
}

fn text_response() -> ChatResponse {
    let mut response = ChatResponseBuilder::new()
        .text("Hello!")
        .finish_reason(FinishReason::Stop)
        .usage(10, 3)
        .id("resp_1")
        .model("fake-1")
        .build();
    response.provider_data = ProviderData::new().with_raw("fake", "service_tier", json!("default"));
    response
}

fn tool_response() -> ChatResponse {
    ChatResponseBuilder::new()
        .tool_call("call_abc", "weather", json!({"city": "Paris"}))
        .tool_call("call_def", "time", json!({"zone": "CET"}))
        .finish_reason(FinishReason::ToolCalls)
        .id("resp_2")
        .build()
}

/// A probe that accepts anything and reports the request it would send.
async fn echo(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
    let body = serde_json::to_value(&request).unwrap();
    let mut response = text_response();
    response.provider = Some("fake".into());
    ProbeOutcome::new(Ok(response), vec![body])
}

#[tokio::test]
async fn an_agreeing_adapter_passes() {
    for response in [text_response(), tool_response()] {
        assert_unary_stream_agree(&capture(&response), unary(none), stream(none)).await;
    }
}

/// Injected bug: "Stream drops response provider data".
#[tokio::test]
#[should_panic(expected = "differing: provider_data")]
async fn a_stream_that_drops_provider_data_fails() {
    fn drop_data(response: &mut ChatResponse) {
        response.provider_data = ProviderData::new();
    }
    assert_unary_stream_agree(&capture(&text_response()), unary(none), stream(drop_data)).await;
}

/// Injected bug: "Usage dropped on one path".
#[tokio::test]
#[should_panic(expected = "differing: usage")]
async fn a_path_that_drops_usage_fails() {
    fn drop_usage(response: &mut ChatResponse) {
        response.usage = None;
    }
    assert_unary_stream_agree(&capture(&text_response()), unary(drop_usage), stream(none)).await;
}

#[tokio::test]
async fn stamped_paths_pass() {
    assert_provider_stamped("fake", "m", echo, echo).await;
}

/// A stream without the provider stamp.
#[tokio::test]
#[should_panic(expected = "chat_stream must stamp the serving provider")]
async fn an_unstamped_stream_fails() {
    async fn unstamped(_: ChatRequest) -> ProbeOutcome<ChatResponse> {
        let mut response = text_response();
        response.provider = None;
        ProbeOutcome::new(Ok(response), Vec::new())
    }
    async fn stamped(_: ChatRequest) -> ProbeOutcome<ChatResponse> {
        let mut response = text_response();
        response.provider = Some("fake".into());
        ProbeOutcome::new(Ok(response), Vec::new())
    }
    assert_provider_stamped("fake", "m", stamped, unstamped).await;
}

#[tokio::test]
async fn wire_tool_call_ids_that_round_trip_pass() {
    let capture = capture(&tool_response()).wire_tool_call_ids(["call_abc", "call_def"]);
    assert_tool_call_ids_round_trip("m", &capture, unary(none), stream(none), echo).await;
}

#[tokio::test]
async fn unique_synthesized_ids_pass() {
    fn synthesize(response: &mut ChatResponse) {
        for (n, block) in response.content.iter_mut().enumerate() {
            if let ContentBlock::ToolCall(call) = block {
                call.id = format!("fake_resp-2_call_{n}");
            }
        }
    }
    let capture = capture(&tool_response());
    assert_tool_call_ids_round_trip("m", &capture, unary(synthesize), stream(synthesize), echo)
        .await;
}

/// Lane I: Anthropic refuses `gemini:<scope>:call:<n>` ids (verified live),
/// so a synthesized id outside `[A-Za-z0-9_-]` breaks a failover.
#[tokio::test]
#[should_panic(expected = "synthesized the id `fake:resp_2:call:0`")]
async fn synthesized_ids_outside_the_portable_alphabet_fail() {
    fn synthesize(response: &mut ChatResponse) {
        for (n, block) in response.content.iter_mut().enumerate() {
            if let ContentBlock::ToolCall(call) = block {
                call.id = format!("fake:resp_2:call:{n}");
            }
        }
    }
    let capture = capture(&tool_response());
    assert_tool_call_ids_round_trip("m", &capture, unary(synthesize), stream(synthesize), echo)
        .await;
}

/// Wire ids are the provider's own: any characters are kept as they are.
#[tokio::test]
async fn wire_ids_outside_the_portable_alphabet_pass() {
    fn rename(response: &mut ChatResponse) {
        for block in &mut response.content {
            if let ContentBlock::ToolCall(call) = block {
                call.id = call.id.replace('_', ":");
            }
        }
    }
    let capture = capture(&tool_response()).wire_tool_call_ids(["call:abc", "call:def"]);
    assert_tool_call_ids_round_trip("m", &capture, unary(rename), stream(rename), echo).await;
}

/// Injected bug: "Tool-call ids replaced by `call_0`".
#[tokio::test]
#[should_panic(expected = "must keep the ids the wire carries")]
async fn ids_replaced_by_their_position_fail() {
    fn renumber(response: &mut ChatResponse) {
        let mut n = 0;
        for block in &mut response.content {
            if let ContentBlock::ToolCall(call) = block {
                call.id = format!("call_{n}");
                n += 1;
            }
        }
    }
    let capture = capture(&tool_response()).wire_tool_call_ids(["call_abc", "call_def"]);
    assert_tool_call_ids_round_trip("m", &capture, unary(renumber), stream(renumber), echo).await;
}

#[tokio::test]
#[should_panic(expected = "duplicate tool-call ids")]
async fn duplicate_synthesized_ids_fail() {
    fn constant(response: &mut ChatResponse) {
        for block in &mut response.content {
            if let ContentBlock::ToolCall(call) = block {
                call.id = "call_0".into();
            }
        }
    }
    let capture = capture(&tool_response());
    assert_tool_call_ids_round_trip("m", &capture, unary(constant), stream(constant), echo).await;
}

/// Injected bug: "Tool calls mapped to `Stop`".
#[tokio::test]
#[should_panic(expected = "finishes with ToolCalls")]
async fn tool_calls_finishing_with_stop_fail() {
    fn stop(response: &mut ChatResponse) {
        response.finish_reason = Some(FinishReason::Stop);
    }
    let capture = capture(&tool_response()).wire_tool_call_ids(["call_abc", "call_def"]);
    assert_tool_call_ids_round_trip("m", &capture, unary(none), stream(stop), echo).await;
}

#[tokio::test]
#[should_panic(expected = "must carry the wire id `call_def`")]
async fn a_replay_that_drops_an_id_fails() {
    async fn forgetful(request: ChatRequest) -> ProbeOutcome<ChatResponse> {
        let body = serde_json::to_string(&request)
            .unwrap()
            .replace("call_def", "call_x");
        ProbeOutcome::raw(Ok(text_response()), [body])
    }
    let capture = capture(&tool_response()).wire_tool_call_ids(["call_abc", "call_def"]);
    assert_tool_call_ids_round_trip("m", &capture, unary(none), stream(none), forgetful).await;
}

#[tokio::test]
async fn a_length_finish_passes() {
    let mut response = text_response();
    response.finish_reason = Some(FinishReason::Length);
    assert_length_finish(&capture(&response), unary(none), stream(none)).await;
}

/// Injected bug: "`cap` mapped to `Stop` instead of `Length`".
#[tokio::test]
#[should_panic(expected = "length_finish: chat:")]
async fn a_cap_mapped_to_stop_fails() {
    fn stop(response: &mut ChatResponse) {
        response.finish_reason = Some(FinishReason::Stop);
    }
    let mut response = text_response();
    response.finish_reason = Some(FinishReason::Length);
    assert_length_finish(&capture(&response), unary(stop), stream(none)).await;
}

/// A wire body with objects at three levels.
fn nested_capture() -> ResponseCapture {
    let body = json!({
        "text": "Hi",
        "meta": {"tier": "flex"},
        "parts": [{"kind": "text"}, {"kind": "text"}],
    })
    .to_string();
    ResponseCapture::new(body.clone(), body)
}

/// `chat` of a fake adapter that keeps the whole wire body as provider
/// data, less the unknown fields of `meta` when `drop_nested` is set.
fn nested_unary(drop_nested: bool) -> impl Fn(String) -> std::future::Ready<Result<ChatResponse>> {
    move |body| {
        let mut wire: serde_json::Value = serde_json::from_str(&body).unwrap();
        if drop_nested {
            wire["meta"] = json!({"tier": wire["meta"]["tier"]});
        }
        let mut response = ChatResponseBuilder::new().text("Hi").build();
        response.provider_data = ProviderData::new().with_raw("fake", "wire", wire);
        std::future::ready(Ok(response))
    }
}

#[tokio::test]
async fn vendor_fields_kept_at_every_level_pass() {
    assert_vendor_fields_kept(&nested_capture(), nested_unary(false)).await;
}

/// A field below the top level that is dropped is named by its JSON
/// pointer.
#[tokio::test]
#[should_panic(expected = "dropped from the objects at /meta;")]
async fn a_vendor_field_dropped_below_the_top_level_fails() {
    assert_vendor_fields_kept(&nested_capture(), nested_unary(true)).await;
}
