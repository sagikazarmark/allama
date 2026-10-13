use allama::{
    CacheBreakpoint, ContentBlock, ContentPart, File, Image, ImageDetail, Message, OtherContent,
    PromptCache, ProviderData, Reasoning, ReasoningConfig, ReasoningEffort, ReasoningOutput,
    ResponseFormat, SystemPrompt, Text, ToolCall, ToolChoice, ToolDefinition, ToolResultContent,
    UnsupportedPolicy, WarningKind,
};
use serde_json::{Value, json};

use super::*;

/// The encoder with the profile the provider builds for the request's
/// model.
fn encode_request(request: &ChatRequest) -> Result<EncodedRequest> {
    super::encode_request(request, ModelProfile::for_model(&request.model))
}

fn encode(request: &ChatRequest) -> (Value, Vec<Warning>) {
    let encoded = encode_request(request).unwrap();
    (
        serde_json::to_value(encoded.body).unwrap(),
        encoded.warnings,
    )
}

fn body(request: &ChatRequest) -> Value {
    encode(request).0
}

fn signed(signature: &str) -> ProviderData {
    ProviderData::new().with_raw(NAMESPACE, THOUGHT_SIGNATURE_KEY, json!(signature))
}

fn weather_call(id: &str, city: &str) -> ToolCall {
    ToolCall::new(id, "get_weather", json!({ "city": city }).to_string())
}

#[test]
fn simple_user_message() {
    let body = body(&ChatRequest::new("gemini-2.5-flash").user("Hello!"));
    assert_eq!(
        body,
        json!({"contents": [{"role": "user", "parts": [{"text": "Hello!"}]}]})
    );
}

/// `systemInstruction` takes several parts, so each prompt keeps
/// its boundary and nothing is approximated.
#[test]
fn several_system_prompts_are_parts_of_the_instruction() {
    let (body, warnings) = encode(
        &ChatRequest::new("gemini-2.5-flash")
            .system("One.")
            .system("")
            .system("Two.")
            .user("hi"),
    );
    assert_eq!(
        body["systemInstruction"],
        json!({"parts": [{"text": "One."}, {"text": "Two."}]})
    );
    assert!(warnings.is_empty(), "{warnings:?}");
}

#[test]
fn single_system_prompt_has_no_warning_and_cache_breakpoints_are_satisfied() {
    let (body, warnings) = encode(
        &ChatRequest::new("gemini-2.5-flash")
            .system(SystemPrompt::new("Be brief.").with_cache(CacheBreakpoint::new()))
            .message(Message::user("hi").with_cache(CacheBreakpoint::new())),
    );
    assert_eq!(body["systemInstruction"]["parts"][0]["text"], "Be brief.");
    assert!(warnings.is_empty(), "{warnings:?}");
}

#[test]
fn system_prompt_options_in_own_namespace_warn() {
    let mut prompt = SystemPrompt::new("x");
    prompt
        .options
        .insert_json(NAMESPACE, json!({"thinking_budget": 1}));
    let (_, warnings) = encode(
        &ChatRequest::new("gemini-2.5-flash")
            .system(prompt)
            .user("hi"),
    );
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].subject, Subject::System { index: 0 });
}

#[test]
fn consecutive_same_role_messages_are_merged() {
    let body = body(
        &ChatRequest::new("gemini-2.5-flash")
            .user("a")
            .user("b")
            .assistant("c")
            .user("d"),
    );
    let contents = body["contents"].as_array().unwrap();
    assert_eq!(contents.len(), 3);
    assert_eq!(contents[0]["parts"], json!([{"text": "a"}, {"text": "b"}]));
    assert_eq!(contents[1]["role"], "model");
}

#[test]
fn empty_turns_are_skipped_without_a_warning() {
    let baseline = ChatRequest::new("gemini-2.5-flash")
        .user("a")
        .assistant("b")
        .user("c");
    let foreign_only = Message::assistant_blocks(vec![ContentBlock::Other(
        OtherContent::new("web_search_call", Default::default()).with_origin("openai"),
    )]);
    let with_empty = ChatRequest::new("gemini-2.5-flash")
        .user("a")
        .message(Message::user(""))
        .message(Message::assistant_blocks(vec![]))
        .assistant("b")
        .message(foreign_only)
        .message(Message::user_parts(vec![]))
        .user("c");
    let (body, warnings) = encode(&with_empty);
    assert_eq!(body, self::body(&baseline));
    // Only the foreign block's drop is reported, at the caller's index.
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].subject, content_subject(4, Some(0)));
}

#[test]
fn a_reply_left_last_by_skipped_turns_is_dropped_under_every_policy() {
    for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
        let mut request = ChatRequest::new("gemini-2.5-flash")
            .user("a")
            .assistant("b")
            .message(Message::user(""));
        request.unsupported_policy = Some(policy);
        let (body, warnings) = encode(&request);
        assert_eq!(
            body["contents"],
            json!([{"role": "user", "parts": [{"text": "a"}]}])
        );
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert_eq!(warnings[0].kind, WarningKind::Dropped);
        assert_eq!(warnings[0].subject, content_subject(1, None));
    }

    // A prefill the caller sent last stays.
    let prefill = body(
        &ChatRequest::new("gemini-2.5-flash")
            .user("a")
            .assistant("{"),
    );
    assert_eq!(prefill["contents"][1]["role"], "model");
}

#[test]
fn an_empty_tool_result_is_never_skipped() {
    let request = ChatRequest::new("gemini-2.5-flash")
        .user("Weather?")
        .message(Message::assistant_blocks(vec![ContentBlock::ToolCall(
            weather_call("c1", "Paris"),
        )]))
        .message(Message::tool_result("c1", "get_weather", ""));
    let body = body(&request);
    assert_eq!(body["contents"].as_array().unwrap().len(), 3);
    assert!(body["contents"][2]["parts"][0]["functionResponse"].is_object());
}

#[test]
fn a_conversation_with_nothing_to_send_is_refused() {
    let Err(err) = encode_request(&ChatRequest::new("gemini-2.5-flash").message(Message::user("")))
    else {
        panic!("a conversation of empty turns is refused");
    };
    assert!(matches!(err, Error::InvalidRequest { .. }), "{err}");
}

#[test]
fn replays_signatures_on_the_part_they_arrived_with() {
    let request =
        ChatRequest::new("gemini-3.5-flash")
            .user("hi")
            .message(Message::assistant_blocks(vec![
                ContentBlock::Text(Text::new("Checking.").with_provider_data(signed("text-sig"))),
                ContentBlock::ToolCall(
                    weather_call("call_1", "Paris").with_provider_data(signed("call-sig")),
                ),
                ContentBlock::ToolCall(weather_call("call_2", "Tokyo")),
                ContentBlock::Text(Text::new("").with_provider_data(signed("tail-sig"))),
            ]));
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["contents"][1]["parts"],
        json!([
            {"text": "Checking.", "thoughtSignature": "text-sig"},
            {"functionCall": {"id": "call_1", "name": "get_weather", "args": {"city": "Paris"}}, "thoughtSignature": "call-sig"},
            {"functionCall": {"id": "call_2", "name": "get_weather", "args": {"city": "Tokyo"}}},
            {"text": "", "thoughtSignature": "tail-sig"}
        ])
    );
}

/// A refusal another provider wrote (OpenAI's `refusal`) has no Gemini
/// part: it is replayed as the assistant text it was, without a warning.
#[test]
fn refusals_are_replayed_as_text() {
    let request = ChatRequest::new("gemini-2.5-flash")
        .user("Help me.")
        .message(Message::assistant_blocks(vec![ContentBlock::Refusal(
            allama::Refusal::new("I can't help with that."),
        )]))
        .user("Why not?");
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["contents"][1],
        json!({"role": "model", "parts": [{"text": "I can't help with that."}]})
    );
}

#[test]
fn replays_assistant_images_with_their_signature() {
    let request = ChatRequest::new("gemini-3.5-flash-image")
        .user("Draw a red square.")
        .message(Message::assistant_blocks(vec![
            ContentBlock::text("Here it is."),
            ContentBlock::Image(
                Image::base64("image/png", "AAAA").with_provider_data(signed("img-sig")),
            ),
        ]))
        .user("Now make it blue.");
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["contents"][1]["parts"][1],
        json!({"inlineData": {"mimeType": "image/png", "data": "AAAA"}, "thoughtSignature": "img-sig"})
    );
}

#[test]
fn gemini3_unsigned_first_function_call_gets_dummy_signature() {
    let request = ChatRequest::new("gemini-3.5-flash")
        .user("weather?")
        .message(Message::assistant_blocks(vec![
            ContentBlock::text("Let me check."),
            ContentBlock::ToolCall(weather_call("toolu_1", "Paris")),
            ContentBlock::ToolCall(weather_call("toolu_2", "Tokyo")),
        ]))
        .message(Message::tool_result("toolu_1", "get_weather", "Sunny"))
        .message(Message::tool_result("toolu_2", "get_weather", "Rainy"));
    let (body, warnings) = encode(&request);
    let parts = &body["contents"][1]["parts"];
    assert!(parts[0].get("thoughtSignature").is_none());
    assert_eq!(parts[1]["thoughtSignature"], DUMMY_THOUGHT_SIGNATURE);
    assert!(parts[2].get("thoughtSignature").is_none());
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].kind, WarningKind::Approximated);
    // History repeats the call on every turn, so the warning is coded.
    assert_eq!(
        warnings[0].code.as_deref(),
        Some("approximated.thought_signature")
    );
    assert_eq!(
        warnings[0].subject,
        Subject::Content {
            message: 1,
            block: Some(1)
        }
    );
    assert_eq!(
        body["contents"][2]["parts"][0]["functionResponse"],
        json!({"id": "toolu_1", "name": "get_weather", "response": {"result": "Sunny"}})
    );
}

/// A model the adapter does not know gets the newest behavior: Gemini 3's
/// signature validation, so an unsigned call gets the placeholder.
#[test]
fn an_unknown_models_unsigned_function_call_gets_dummy_signature() {
    for model in ["tunedModels/my-model", "gemini-exp-1206"] {
        let request = ChatRequest::new(model)
            .user("weather?")
            .message(Message::assistant_blocks(vec![ContentBlock::ToolCall(
                weather_call("toolu_1", "Paris"),
            )]));
        let (body, warnings) = encode(&request);
        assert_eq!(
            body["contents"][1]["parts"][0]["thoughtSignature"], DUMMY_THOUGHT_SIGNATURE,
            "{model}"
        );
        assert_eq!(warnings.len(), 1, "{model}: {warnings:?}");
        assert_eq!(warnings[0].kind, WarningKind::Approximated);
    }
}

#[test]
fn gemini25_function_calls_get_no_dummy_signature() {
    let request = ChatRequest::new("gemini-2.5-flash")
        .user("weather?")
        .message(Message::assistant_blocks(vec![ContentBlock::ToolCall(
            weather_call("toolu_1", "Paris"),
        )]));
    let (body, warnings) = encode(&request);
    assert!(
        body["contents"][1]["parts"][0]
            .get("thoughtSignature")
            .is_none()
    );
    assert!(warnings.is_empty());
}

#[test]
fn foreign_signatures_are_not_replayed() {
    let foreign = ProviderData::new().with_raw("anthropic", "signature", json!("abc"));
    let request = ChatRequest::new("gemini-3.5-flash")
        .user("weather?")
        .message(Message::assistant_blocks(vec![
            ContentBlock::Reasoning(
                Reasoning::new("I should call the tool.").with_provider_data(foreign.clone()),
            ),
            ContentBlock::Reasoning(Reasoning::new("").with_provider_data(foreign.clone())),
            ContentBlock::ToolCall(weather_call("toolu_1", "Paris").with_provider_data(foreign)),
        ]));
    let (body, warnings) = encode(&request);
    let parts = body["contents"][1]["parts"].as_array().unwrap();
    assert_eq!(parts.len(), 1, "reasoning blocks are dropped: {parts:?}");
    assert_eq!(parts[0]["thoughtSignature"], DUMMY_THOUGHT_SIGNATURE);
    let dropped = warnings
        .iter()
        .filter(|w| w.kind == WarningKind::Dropped)
        .count();
    // The two reasoning blocks (one warning per code per request) plus
    // the foreign state on the tool call.
    assert_eq!(dropped, 2, "{warnings:?}");
    assert!(warnings.iter().any(|w| w.kind == WarningKind::Approximated));
}

#[test]
fn core_provider_data_is_not_reported_as_foreign() {
    let core = || signed("sig").with_raw("allama", "note", json!(1));
    let request = ChatRequest::new("gemini-2.5-flash")
        .user("weather?")
        .message(Message::assistant_blocks(vec![
            ContentBlock::Text(Text::new("Checking.").with_provider_data(core())),
            ContentBlock::ToolCall(weather_call("c1", "Paris").with_provider_data(core())),
        ]));
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(!body.to_string().contains("note"));
}

#[test]
fn own_thought_blocks_replay_as_thought_parts() {
    let thought = ProviderData::new()
        .with_raw(NAMESPACE, "thought", json!(true))
        .with_raw(NAMESPACE, THOUGHT_SIGNATURE_KEY, json!("s"));
    let request =
        ChatRequest::new("gemini-2.5-flash")
            .user("q")
            .message(Message::assistant_blocks(vec![
                ContentBlock::Reasoning(Reasoning::new("thinking").with_provider_data(thought)),
                ContentBlock::text("answer"),
            ]));
    let (body, warnings) = encode(&request);
    assert_eq!(
        body["contents"][1]["parts"],
        json!([
            {"text": "thinking", "thought": true, "thoughtSignature": "s"},
            {"text": "answer"}
        ])
    );
    assert!(warnings.is_empty());
}

#[test]
fn other_content_replays_only_from_own_origin() {
    let mut data = serde_json::Map::new();
    data.insert(
        "executableCode".into(),
        json!({"language": "PYTHON", "code": "print(1)"}),
    );
    let own = OtherContent::new("executableCode", data.clone()).with_origin(NAMESPACE);
    let foreign = OtherContent::new("server_tool_use", data).with_origin("anthropic");
    let request =
        ChatRequest::new("gemini-2.5-flash")
            .user("q")
            .message(Message::assistant_blocks(vec![
                ContentBlock::Other(own),
                ContentBlock::Other(foreign),
            ]));
    let (body, warnings) = encode(&request);
    assert_eq!(
        body["contents"][1]["parts"],
        json!([{"executableCode": {"language": "PYTHON", "code": "print(1)"}}])
    );
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].kind, WarningKind::Dropped);
}

#[test]
fn other_content_replays_both_signature_forms() {
    let audio = |extra: Value| {
        let mut data = serde_json::Map::new();
        data.insert(
            "inlineData".into(),
            json!({"mimeType": "audio/wav", "data": "AAAA"}),
        );
        if let Value::Object(extra) = extra {
            data.extend(extra);
        }
        OtherContent::new("inline_data", data).with_origin(NAMESPACE)
    };
    let foreign = ProviderData::new().with_raw("anthropic", "signature", json!("abc"));
    let request = ChatRequest::new("gemini-2.5-flash")
        .message(Message::user(vec![ContentPart::Other(
            audio(json!({})).with_provider_data(signed("part-sig")),
        )]))
        .message(Message::assistant_blocks(vec![
            // Decoded form: the signature in provider data.
            ContentBlock::Other(audio(json!({})).with_provider_data(signed("new-sig"))),
            // Wire form: the signature inside the verbatim part.
            ContentBlock::Other(audio(json!({"thoughtSignature": "wire-sig"}))),
            // Both: provider data wins, and the key is not sent twice.
            ContentBlock::Other(
                audio(json!({"thoughtSignature": "wire-sig"}))
                    .with_provider_data(signed("new-sig")),
            ),
            // Foreign provider data is never sent.
            ContentBlock::Other(audio(json!({})).with_provider_data(foreign)),
        ]));
    let (body, warnings) = encode(&request);
    let part = |signature: Option<&str>| {
        let mut part = json!({"inlineData": {"mimeType": "audio/wav", "data": "AAAA"}});
        if let Some(signature) = signature {
            part["thoughtSignature"] = json!(signature);
        }
        part
    };
    assert_eq!(
        body["contents"][0]["parts"],
        json!([part(Some("part-sig"))])
    );
    assert_eq!(
        body["contents"][1]["parts"],
        json!([
            part(Some("new-sig")),
            part(Some("wire-sig")),
            part(Some("new-sig")),
            part(None),
        ])
    );
    assert!(!body.to_string().contains("abc"));
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert_eq!(warnings[0].kind, WarningKind::Dropped);
    assert_eq!(
        warnings[0].subject,
        Subject::Content {
            message: 1,
            block: Some(3)
        }
    );
}

#[test]
fn invalid_tool_arguments_replay_as_empty_object_with_warning() {
    let request =
        ChatRequest::new("gemini-2.5-flash")
            .user("q")
            .message(Message::assistant_blocks(vec![ContentBlock::tool_call(
                "c1",
                "f",
                "{not json",
            )]));
    let (body, warnings) = encode(&request);
    assert_eq!(
        body["contents"][1]["parts"][0]["functionCall"]["args"],
        json!({})
    );
    assert_eq!(warnings.len(), 1);
    // A lossy approximation, replayed on every turn, so coded.
    assert_eq!(warnings[0].kind, WarningKind::Approximated);
    assert_eq!(
        warnings[0].code.as_deref(),
        Some("approximated.tool_call_arguments")
    );
    assert_eq!(
        warnings[0].message,
        "the arguments of tool call `f` are not a JSON object; they were sent as `{}`"
    );

    let strict = request.unsupported_policy(UnsupportedPolicy::Error);
    assert!(matches!(
        encode_request(&strict),
        Err(Error::Unsupported { .. })
    ));
}

#[test]
fn empty_tool_arguments_replay_as_empty_object_silently() {
    let request =
        ChatRequest::new("gemini-2.5-flash")
            .user("q")
            .message(Message::assistant_blocks(vec![ContentBlock::tool_call(
                "c1", "f", "",
            )]));
    let (body, warnings) = encode(&request);
    assert_eq!(
        body["contents"][1]["parts"][0]["functionCall"]["args"],
        json!({})
    );
    assert!(warnings.is_empty());
}

#[test]
fn tool_results_map_to_function_responses() {
    let mut object = serde_json::Map::new();
    object.insert("temp".into(), json!(18));
    let request = ChatRequest::new("gemini-2.5-flash")
        .user("q")
        .message(Message::assistant_blocks(vec![
            ContentBlock::tool_call("a", "f", "{}"),
            ContentBlock::tool_call("b", "f", "{}"),
            ContentBlock::tool_call("c", "f", "{}"),
            ContentBlock::tool_call("d", "f", "{}"),
        ]))
        .message(Message::tool_result("a", "f", "plain"))
        .message(Message::tool_error("b", "f", "boom"))
        .message(Message::tool_result(
            "c",
            "f",
            ToolResultContent::Object(object),
        ))
        .message(Message::tool_result(
            "d",
            "f",
            ToolResultContent::Parts(vec![ContentPart::text("x"), ContentPart::text("y")]),
        ));
    let body = body(&request);
    let responses: Vec<_> = body["contents"][2]["parts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["functionResponse"]["response"].clone())
        .collect();
    assert_eq!(
        responses,
        vec![
            json!({"result": "plain"}),
            json!({"error": "boom"}),
            json!({"temp": 18}),
            json!({"result": "x\n\ny"}),
        ]
    );
}

/// Text parts keep their boundaries, and media that came
/// before text is reported as moved after it.
#[test]
fn multimodal_tool_results_keep_text_boundaries_and_report_the_order() {
    let request = ChatRequest::new("gemini-2.5-flash")
        .user("q")
        .message(Message::assistant_blocks(vec![ContentBlock::tool_call(
            "c1", "f", "{}",
        )]))
        .message(Message::tool_result(
            "c1",
            "f",
            ToolResultContent::Parts(vec![
                ContentPart::text("before"),
                ContentPart::image_base64("image/png", "iVBO"),
                ContentPart::text("after"),
            ]),
        ));
    let (body, warnings) = encode(&request);
    let response = &body["contents"][2]["parts"][0]["functionResponse"];
    assert_eq!(response["response"], json!({"result": "before\n\nafter"}));
    assert_eq!(response["parts"].as_array().unwrap().len(), 1);
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].kind, WarningKind::Approximated);
    assert_eq!(
        warnings[0].code.as_deref(),
        Some("approximated.tool_result_order")
    );
}

/// History repeats both approximations on every
/// turn, so a five-turn tool conversation gets one warning per code, each
/// listing every message, not one per turn.
#[test]
fn a_tool_conversation_gets_one_warning_per_code() {
    let mut request = ChatRequest::new("gemini-3.5-flash");
    for turn in 0..5 {
        let id = format!("c{turn}");
        request = request
            .user(format!("Question {turn}"))
            .message(Message::assistant_blocks(vec![ContentBlock::tool_call(
                id.clone(),
                "f",
                "{}",
            )]))
            .message(Message::tool_result(
                id,
                "f",
                ToolResultContent::Parts(vec![
                    ContentPart::image_base64("image/png", "iVBO"),
                    ContentPart::text("answer"),
                ]),
            ));
    }
    let (body, warnings) = encode(&request);
    let calls: Vec<_> = body["contents"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|content| content["role"] == "model")
        .map(|content| &content["parts"][0]["thoughtSignature"])
        .collect();
    assert_eq!(calls, vec![DUMMY_THOUGHT_SIGNATURE; 5]);
    let coded: Vec<_> = warnings
        .iter()
        .map(|w| (w.kind, w.code.as_deref(), w.subject.clone()))
        .collect();
    assert_eq!(
        coded,
        vec![
            (
                WarningKind::Approximated,
                Some("approximated.tool_result_order"),
                Subject::Messages {
                    messages: vec![2, 5, 8, 11, 14]
                }
            ),
            (
                WarningKind::Approximated,
                Some("approximated.thought_signature"),
                Subject::Messages {
                    messages: vec![1, 4, 7, 10, 13]
                }
            ),
        ],
        "{warnings:?}"
    );
}

/// Multimodal function responses are sent to every model: one that lacks
/// them (Gemini 2.5 and older, Gemma 4) refuses upstream, since its lack is
/// model data (ADR 0015), recorded in `docs/catalog/gemini.md`.
#[test]
fn multimodal_tool_results_use_function_response_parts_on_every_model() {
    let result = ToolResultContent::Parts(vec![
        ContentPart::text("see image"),
        ContentPart::image_base64("image/png", "iVBO"),
    ]);
    let request = |model: &str| {
        ChatRequest::new(model)
            .user("q")
            .message(Message::assistant_blocks(vec![ContentBlock::tool_call(
                "c1",
                "get_image",
                "{}",
            )]))
            .message(Message::tool_result("c1", "get_image", result.clone()))
    };
    let expected = json!({
        "id": "c1",
        "name": "get_image",
        "response": {"result": "see image"},
        "parts": [{"inlineData": {"mimeType": "image/png", "data": "iVBO"}}]
    });
    let gemini3 = body(&request("gemini-3.5-flash-lite"));
    assert_eq!(
        gemini3["contents"][2]["parts"][0]["functionResponse"],
        expected
    );
    for model in [
        "tunedModels/my-model",
        "gemini-exp-1206",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemma-4-31b-it",
    ] {
        let unknown = body(&request(model));
        assert_eq!(
            unknown["contents"][2]["parts"][0]["functionResponse"], expected,
            "{model}"
        );
    }
}

#[test]
fn tool_name_is_recovered_from_history_when_missing() {
    let request = ChatRequest::new("gemini-2.5-flash")
        .user("q")
        .message(Message::assistant_blocks(vec![ContentBlock::tool_call(
            "c1", "lookup", "{}",
        )]))
        .message(Message::tool_result("c1", "", "ok"));
    let body = body(&request);
    assert_eq!(
        body["contents"][2]["parts"][0]["functionResponse"]["name"],
        "lookup"
    );
}

#[test]
fn synthetic_id_tool_results_are_sorted_by_call_index() {
    let request = ChatRequest::new("gemini-2.5-flash")
        .user("q")
        .message(Message::assistant_blocks(vec![
            ContentBlock::tool_call("gemini_r_call_0", "f", "{}"),
            ContentBlock::tool_call("gemini_r_call_1", "g", "{}"),
        ]))
        .message(Message::tool_result("gemini_r_call_1", "g", "second"))
        .message(Message::tool_result("gemini_r_call_0", "f", "first"));
    let body = body(&request);
    let names: Vec<_> = body["contents"][2]["parts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["functionResponse"]["name"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(names, vec!["f", "g"]);
}

#[test]
fn tools_send_json_schema_unchanged() {
    let schema = json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"pet": {"$ref": "#/$defs/Pet"}},
        "additionalProperties": false,
        "$defs": {"Pet": {"type": "object", "properties": {"name": {"type": "string"}}}}
    });
    let (body, warnings) = encode(
        &ChatRequest::new("gemini-2.5-flash")
            .user("q")
            .tools([ToolDefinition::new("add_pet", schema.clone()).description("Add")]),
    );
    assert_eq!(
        body["tools"],
        json!([{"functionDeclarations": [{"name": "add_pet", "description": "Add", "parametersJsonSchema": schema}]}])
    );
    assert!(warnings.is_empty());
}

#[test]
fn strict_tools_warn() {
    let (_, warnings) = encode(
        &ChatRequest::new("gemini-2.5-flash")
            .user("q")
            .tools([ToolDefinition::new("f", json!({"type": "object"})).strict(true)]),
    );
    assert_eq!(warnings.len(), 1);
    assert_eq!(
        warnings[0].subject,
        Subject::Field {
            field: RequestField::Tools
        }
    );
}

#[test]
fn tool_choice_modes() {
    for (choice, expected) in [
        (ToolChoice::Auto, json!({"mode": "AUTO"})),
        (ToolChoice::Disabled, json!({"mode": "NONE"})),
        (ToolChoice::Required, json!({"mode": "ANY"})),
        (
            ToolChoice::Specific { name: "f".into() },
            json!({"mode": "ANY", "allowedFunctionNames": ["f"]}),
        ),
    ] {
        let body = body(
            &ChatRequest::new("gemini-2.5-flash")
                .user("q")
                .tool_choice(choice),
        );
        assert_eq!(body["toolConfig"]["functionCallingConfig"], expected);
    }
}

/// Penalties are sent to every model, under every policy: a model that
/// does not enable them answers "Penalty is not enabled for this model",
/// which maps to `Unsupported` at the `Provider` stage (see the error
/// module). That no current model enables them is model data (ADR 0015).
#[test]
fn penalties_are_sent() {
    for model in [
        "gemini-2.5-flash",
        "gemini-3.5-flash",
        "tunedModels/my-model",
    ] {
        let request = ChatRequest::new(model)
            .user("q")
            .frequency_penalty(0.5)
            .presence_penalty(0.25);
        for request in [
            request.clone(),
            request.unsupported_policy(UnsupportedPolicy::Error),
        ] {
            let (body, warnings) = encode(&request);
            assert_eq!(
                body["generationConfig"],
                json!({"frequencyPenalty": 0.5, "presencePenalty": 0.25}),
                "{model}"
            );
            assert!(warnings.is_empty(), "{model}: {warnings:?}");
        }
    }
}

/// A seed is a hint (ADR 0003); one out of Gemini's range is not
/// sent, with a warning, and refused only under the strict policy.
#[test]
fn seed_is_mapped_and_range_checked() {
    let body = body(&ChatRequest::new("gemini-2.5-flash").user("q").seed(42));
    assert_eq!(body["generationConfig"]["seed"], 42);
    let request = ChatRequest::new("gemini-2.5-flash")
        .user("q")
        .seed(u64::MAX);
    let (body, warnings) = encode(&request);
    assert!(body.get("generationConfig").is_none());
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].kind, WarningKind::Unsupported);
    assert_eq!(
        warnings[0].subject,
        Subject::Field {
            field: RequestField::Seed
        }
    );
    let err = encode_request(&request.unsupported_policy(UnsupportedPolicy::Error))
        .err()
        .unwrap();
    assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
}

#[test]
fn response_formats() {
    let body_for = |format| {
        body(
            &ChatRequest::new("gemini-2.5-flash")
                .user("q")
                .response_format(format),
        )
    };
    assert!(
        body_for(ResponseFormat::Text)
            .get("generationConfig")
            .is_none()
    );
    assert_eq!(
        body_for(ResponseFormat::Json)["generationConfig"],
        json!({"responseMimeType": "application/json"})
    );
    let schema = json!({"type": "object", "$defs": {}, "properties": {}});
    assert_eq!(
        body_for(ResponseFormat::json_schema(schema.clone()))["generationConfig"],
        json!({"responseMimeType": "application/json", "responseJsonSchema": schema})
    );
}

/// Gemini always constrains `responseJsonSchema` and has no strict flag, so
/// a strict and a `non_strict` schema encode the same, without warnings.
#[test]
fn json_schema_is_constrained_strict_or_not() {
    let schema = json!({"type": "object", "properties": {"a": {"type": "string"}}});
    let encode_format = |format| {
        encode(
            &ChatRequest::new("gemini-2.5-flash")
                .user("q")
                .response_format(format),
        )
    };
    let (strict, strict_warnings) = encode_format(ResponseFormat::json_schema(schema.clone()));
    let (relaxed, relaxed_warnings) =
        encode_format(ResponseFormat::json_schema(schema.clone()).non_strict());
    assert_eq!(
        strict["generationConfig"],
        json!({"responseMimeType": "application/json", "responseJsonSchema": schema})
    );
    assert_eq!(strict["generationConfig"], relaxed["generationConfig"]);
    assert!(strict_warnings.is_empty(), "{strict_warnings:?}");
    assert!(relaxed_warnings.is_empty(), "{relaxed_warnings:?}");
}

/// Gemini accepts map schemas (`additionalProperties` with a schema), so
/// unlike OpenAI and Anthropic strict mode they reach the wire unchanged.
#[test]
fn map_schemas_are_sent_unchanged() {
    let schema = json!({
        "type": "object",
        "properties": {"labels": {"type": "object", "additionalProperties": {"type": "string"}}},
        "required": ["labels"]
    });
    let (body, warnings) = encode(
        &ChatRequest::new("gemini-2.5-flash")
            .user("q")
            .tools([ToolDefinition::new("label", schema.clone())])
            .response_format(ResponseFormat::json_schema(schema.clone())),
    );
    assert_eq!(
        body["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"],
        schema
    );
    assert_eq!(body["generationConfig"]["responseJsonSchema"], schema);
    assert!(warnings.is_empty(), "{warnings:?}");
}

#[test]
fn response_mime_type_option_overrides() {
    let body = body(
        &ChatRequest::new("gemini-2.5-flash")
            .user("q")
            .response_format(ResponseFormat::Json)
            .with_option(ChatRequestOptions {
                response_mime_type: Some("text/x.enum".into()),
                ..Default::default()
            }),
    );
    assert_eq!(body["generationConfig"]["responseMimeType"], "text/x.enum");
}

#[test]
fn candidate_count_is_refused_with_its_own_message() {
    for key in ["candidateCount", "candidate_count"] {
        let request = ChatRequest::new("gemini-2.5-flash")
            .user("q")
            .with_option(ChatRequestOptions::new().generation_config_extra(key, 2));
        let err = encode_request(&request).err().expect("refused");
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
        let message = err.to_string();
        assert!(
            message.contains(&format!("`{key}` cannot be set"))
                && message.contains("one candidate per response")
                && !message.contains("typed option"),
            "{message}"
        );
    }

    // The typed option is gone, and its JSON form is an unknown field.
    let request = ChatRequest::new("gemini-2.5-flash")
        .user("q")
        .with_options_json(crate::NAMESPACE, json!({"candidate_count": 1}));
    assert!(matches!(
        encode_request(&request),
        Err(Error::InvalidRequest { .. })
    ));
}

fn thinking(
    model: &str,
    reasoning: Option<ReasoningConfig>,
    budget: Option<i32>,
) -> (Value, Vec<Warning>) {
    let mut request = ChatRequest::new(model).user("q");
    if let Some(reasoning) = reasoning {
        request = request.reasoning(reasoning);
    }
    if budget.is_some() {
        request = request.with_option(ChatRequestOptions {
            thinking_budget: budget,
            ..Default::default()
        });
    }
    let (body, warnings) = encode(&request);
    (body["generationConfig"]["thinkingConfig"].clone(), warnings)
}

#[test]
fn gemini3_effort_maps_to_thinking_level() {
    for (effort, level, approximated) in [
        (ReasoningEffort::None, "MINIMAL", true),
        (ReasoningEffort::Minimal, "MINIMAL", false),
        (ReasoningEffort::Low, "LOW", false),
        (ReasoningEffort::Medium, "MEDIUM", false),
        (ReasoningEffort::High, "HIGH", false),
        (ReasoningEffort::XHigh, "HIGH", true),
    ] {
        let (config, warnings) = thinking(
            "gemini-3.5-flash",
            Some(ReasoningConfig::new().effort(effort)),
            None,
        );
        assert_eq!(config["thinkingLevel"], level, "{effort:?}");
        assert!(config.get("thinkingBudget").is_none());
        assert_eq!(
            !warnings.is_empty(),
            approximated,
            "{effort:?}: {warnings:?}"
        );
    }
}

#[test]
fn gemini3_pro_has_no_minimal_level() {
    for effort in [ReasoningEffort::None, ReasoningEffort::Minimal] {
        let (config, warnings) = thinking(
            "gemini-3.1-pro-preview",
            Some(ReasoningConfig::new().effort(effort)),
            None,
        );
        assert_eq!(config["thinkingLevel"], "LOW");
        assert_eq!(warnings.len(), 1);
    }
}

#[test]
fn gemini25_effort_maps_to_thinking_budget() {
    for (model, effort, budget, approximated) in [
        ("gemini-2.5-flash", ReasoningEffort::None, 0, false),
        ("gemini-2.5-pro", ReasoningEffort::None, 128, true),
        ("gemini-2.5-flash", ReasoningEffort::Minimal, 512, false),
        ("gemini-2.5-flash", ReasoningEffort::Low, 1024, false),
        ("gemini-2.5-flash", ReasoningEffort::Medium, 8192, false),
        ("gemini-2.5-flash", ReasoningEffort::High, 24576, false),
        ("gemini-2.5-pro", ReasoningEffort::XHigh, 32768, false),
        ("gemini-2.5-flash", ReasoningEffort::XHigh, 24576, true),
    ] {
        let (config, warnings) = thinking(model, Some(ReasoningConfig::new().effort(effort)), None);
        assert_eq!(config["thinkingBudget"], budget, "{model} {effort:?}");
        assert!(config.get("thinkingLevel").is_none());
        assert_eq!(!warnings.is_empty(), approximated, "{model} {effort:?}");
    }
}

#[test]
fn reasoning_output_controls_include_thoughts() {
    let (config, _) = thinking(
        "gemini-3.5-flash",
        Some(ReasoningConfig::new().effort(ReasoningEffort::Low)),
        None,
    );
    assert_eq!(config["includeThoughts"], true);
    let (config, _) = thinking(
        "gemini-3.5-flash",
        Some(ReasoningConfig::new().output(ReasoningOutput::Omitted)),
        None,
    );
    assert_eq!(config, json!({"includeThoughts": false}));
    let (config, warnings) = thinking(
        "gemini-3.5-flash",
        Some(ReasoningConfig::new().output(ReasoningOutput::Full)),
        None,
    );
    assert_eq!(config, json!({"includeThoughts": true}));
    assert_eq!(warnings[0].kind, WarningKind::Approximated);
    let (config, _) = thinking(
        "gemini-2.5-flash",
        Some(ReasoningConfig::new().effort(ReasoningEffort::None)),
        None,
    );
    assert_eq!(config, json!({"thinkingBudget": 0}));
}

#[test]
fn thinking_budget_option_overrides_effort_and_never_sends_a_level() {
    let (config, warnings) = thinking(
        "gemini-3.5-flash",
        Some(ReasoningConfig::new().effort(ReasoningEffort::High)),
        Some(-1),
    );
    assert_eq!(
        config,
        json!({"includeThoughts": true, "thinkingBudget": -1})
    );
    assert!(warnings.is_empty());
    let (config, _) = thinking("gemini-2.5-flash", None, Some(2048));
    assert_eq!(config, json!({"thinkingBudget": 2048}));
}

/// Gemini 1.x and 2.0 predate `thinkingLevel`: a reasoning config is sent
/// as `thinkingBudget`, from the 2.5 ladder without its limits, and never
/// dropped (ADR 0015). A model without thinking refuses upstream.
#[test]
fn older_generations_get_a_thinking_budget() {
    for (effort, budget) in [
        (ReasoningEffort::None, 0),
        (ReasoningEffort::Low, 1024),
        (ReasoningEffort::XHigh, 32768),
    ] {
        for model in ["gemini-2.0-flash", "gemini-1.5-pro"] {
            let (config, warnings) =
                thinking(model, Some(ReasoningConfig::new().effort(effort)), None);
            assert_eq!(config["thinkingBudget"], budget, "{model} {effort:?}");
            assert!(config.get("thinkingLevel").is_none());
            assert!(warnings.is_empty(), "{model} {effort:?}: {warnings:?}");
        }
    }
    let (config, _) = thinking("gemini-2.0-flash", None, Some(2048));
    assert_eq!(config, json!({"thinkingBudget": 2048}));
}

/// Gemma 4 thinks on or off through `thinkingLevel`: "setting the thinking
/// level to "high" for enabled or "minimal" for disabled"
/// (<https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api>).
#[test]
fn gemma_4_thinking_is_on_or_off() {
    let (config, warnings) = thinking(
        "gemma-4-31b-it",
        Some(ReasoningConfig::new().effort(ReasoningEffort::High)),
        None,
    );
    assert_eq!(
        config,
        json!({"includeThoughts": true, "thinkingLevel": "HIGH"})
    );
    assert!(warnings.is_empty());
    let (config, warnings) = thinking(
        "gemma-4-26b-a4b-it",
        Some(ReasoningConfig::new().effort(ReasoningEffort::Low)),
        None,
    );
    assert_eq!(config["thinkingLevel"], "HIGH");
    assert_eq!(warnings[0].kind, WarningKind::Approximated);
    let (config, _) = thinking(
        "gemma-4-31b-it",
        Some(ReasoningConfig::new().effort(ReasoningEffort::None)),
        None,
    );
    assert_eq!(config["thinkingLevel"], "MINIMAL");
}

#[test]
fn parallel_tool_calls_false_warns_true_is_native() {
    let (_, warnings) = encode(
        &ChatRequest::new("gemini-2.5-flash")
            .user("q")
            .parallel_tool_calls(true),
    );
    assert!(warnings.is_empty());
    let request = ChatRequest::new("gemini-2.5-flash")
        .user("q")
        .parallel_tool_calls(false);
    let (_, warnings) = encode(&request);
    assert_eq!(
        warnings[0].subject,
        Subject::Field {
            field: RequestField::ParallelToolCalls
        }
    );
    assert!(encode_request(&request.unsupported_policy(UnsupportedPolicy::Error)).is_err());
}

#[test]
fn prompt_cache_key_warns() {
    let (_, warnings) = encode(
        &ChatRequest::new("gemini-2.5-flash")
            .user("q")
            .cache(PromptCache::new().key("session")),
    );
    assert_eq!(
        warnings[0].subject,
        Subject::Field {
            field: RequestField::Cache
        }
    );
}

/// Implicit caching already caches prefixes automatically, so the
/// intent is satisfied with nothing to send and nothing to warn about.
#[test]
fn automatic_prompt_cache_is_satisfied_by_implicit_caching() {
    let request = ChatRequest::new("gemini-2.5-flash")
        .user("q")
        .cache(PromptCache::automatic());
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body,
        self::body(&ChatRequest::new("gemini-2.5-flash").user("q"))
    );
}

#[test]
fn file_parts_map_to_inline_and_file_data() {
    let request = ChatRequest::new("gemini-2.5-flash").user(vec![
        ContentPart::text("Summarize"),
        ContentPart::file_base64("application/pdf", "JVBE"),
        ContentPart::File(
            File::url("audio/mpeg", "https://example.com/a.mp3").with_filename("a.mp3"),
        ),
        ContentPart::file_base64("text/plain", "aGk="),
    ]);
    let (body, warnings) = encode(&request);
    assert_eq!(
        body["contents"][0]["parts"],
        json!([
            {"text": "Summarize"},
            {"inlineData": {"mimeType": "application/pdf", "data": "JVBE"}},
            {"fileData": {"mimeType": "audio/mpeg", "fileUri": "https://example.com/a.mp3"}},
            {"inlineData": {"mimeType": "text/plain", "data": "aGk="}}
        ])
    );
    assert_eq!(warnings.len(), 1, "filename is reported: {warnings:?}");
    assert_eq!(warnings[0].kind, WarningKind::Dropped);
}

#[test]
fn image_urls_need_a_known_mime_type() {
    let inferred = body(
        &ChatRequest::new("gemini-2.5-flash").user(vec![ContentPart::image_url(
            "https://example.com/cat.JPG?x=1",
        )]),
    );
    assert_eq!(
        inferred["contents"][0]["parts"][0],
        json!({"fileData": {"mimeType": "image/jpeg", "fileUri": "https://example.com/cat.JPG?x=1"}})
    );
    let err = encode_request(
        &ChatRequest::new("gemini-2.5-flash")
            .user(vec![ContentPart::image_url("https://example.com/image")]),
    )
    .err()
    .unwrap();
    assert!(matches!(err, Error::Unsupported { .. }));
    assert!(err.message().contains("Image::with_media_type"));

    let declared = body(
        &ChatRequest::new("gemini-2.5-flash").user(vec![ContentPart::Image(
            allama::Image::url("https://example.com/image").with_media_type("image/webp"),
        )]),
    );
    assert_eq!(
        declared["contents"][0]["parts"][0],
        json!({"fileData": {"mimeType": "image/webp", "fileUri": "https://example.com/image"}})
    );
}

/// Image detail is the part's `mediaResolution` (verified live on
/// 2026-10-07: the image token count follows the level); `Auto` leaves it
/// to the model.
#[test]
fn image_detail_maps_to_media_resolution() {
    let request = ChatRequest::new("gemini-3.5-flash").user(vec![
        ContentPart::Image(Image::base64("image/png", "iVBO").with_detail(ImageDetail::High)),
        ContentPart::Image(Image::base64("image/png", "iVBO").with_detail(ImageDetail::Low)),
        ContentPart::Image(Image::base64("image/png", "iVBO").with_detail(ImageDetail::Auto)),
    ]);
    let (body, warnings) = encode(&request);
    let parts = &body["contents"][0]["parts"];
    assert_eq!(
        parts[0]["mediaResolution"],
        json!({"level": "MEDIA_RESOLUTION_HIGH"})
    );
    assert_eq!(
        parts[1]["mediaResolution"],
        json!({"level": "MEDIA_RESOLUTION_LOW"})
    );
    assert!(parts[2].get("mediaResolution").is_none());
    assert!(warnings.is_empty(), "{warnings:?}");
}

/// A `google-ai` provider file (the Files API name) is a `fileData`
/// part with its Files API URI; the base URL does not change it.
#[test]
fn own_provider_files_map_to_file_data() {
    let request = ChatRequest::new("gemini-3.5-flash").user(vec![
        ContentPart::File(File::provider_file(
            "application/pdf",
            NAMESPACE,
            "files/abc-123",
        )),
        ContentPart::Image(Image::provider_file(NAMESPACE, "img-1")),
    ]);
    let (body, warnings) = encode(&request);
    let parts = &body["contents"][0]["parts"];
    assert_eq!(
        parts[0],
        json!({"fileData": {
            "mimeType": "application/pdf",
            "fileUri": "https://generativelanguage.googleapis.com/v1beta/files/abc-123"
        }})
    );
    assert_eq!(
        parts[1],
        json!({"fileData": {"fileUri": "https://generativelanguage.googleapis.com/v1beta/files/img-1"}})
    );
    assert!(warnings.is_empty(), "{warnings:?}");

    let foreign = ChatRequest::new("gemini-3.5-flash").user(vec![ContentPart::Image(
        Image::provider_file("openai", "file-1"),
    )]);
    assert!(matches!(
        encode_request(&foreign).err().unwrap(),
        Error::Unsupported { .. }
    ));
}

/// A draft image Gemini made while thinking is replayed as a
/// thought part.
#[test]
fn thought_images_replay_as_thought_parts() {
    let image = Image::base64("image/png", "iVBO")
        .with_provider_data(ProviderData::new().with_raw(NAMESPACE, "thought", json!(true)));
    let request = ChatRequest::new("gemini-3-pro-image")
        .user("draw")
        .message(Message::assistant_blocks(vec![
            ContentBlock::Image(image),
            ContentBlock::Image(Image::base64("image/png", "iVBO")),
        ]))
        .user("again");
    let body = body(&request);
    let parts = &body["contents"][1]["parts"];
    assert_eq!(parts[0]["thought"], true);
    assert!(parts[1].get("thought").is_none());
}

/// The output setting never turns thinking on; where thinking is off
/// it has no effect and is reported.
#[test]
fn reasoning_output_alone_keeps_the_default_thinking() {
    let (body, warnings) = encode(
        &ChatRequest::new("gemini-2.5-flash")
            .user("q")
            .reasoning(ReasoningConfig::new().output(ReasoningOutput::Omitted)),
    );
    assert_eq!(
        body["generationConfig"]["thinkingConfig"],
        json!({"includeThoughts": false})
    );
    assert!(warnings.is_empty(), "{warnings:?}");

    let off = ChatRequest::new("gemini-2.5-flash").user("q").reasoning(
        ReasoningConfig::new()
            .effort(ReasoningEffort::None)
            .output(ReasoningOutput::Summary),
    );
    let (body, warnings) = encode(&off);
    assert_eq!(
        body["generationConfig"]["thinkingConfig"],
        json!({"thinkingBudget": 0})
    );
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].kind, WarningKind::Unsupported);
}

#[test]
fn signature_round_trips_through_serialized_history() {
    // Decode a recorded Gemini 3 response, persist the assistant turn, and
    // replay it: the signature must come back on the same functionCall.
    let raw: Value = serde_json::from_str(include_str!(
        "../../fixtures/gemini3_tool_response_raw.json"
    ))
    .unwrap();
    let response =
        crate::decode::decode_response(serde_json::from_value(raw).unwrap(), Vec::new(), None)
            .unwrap();
    let persisted = serde_json::to_string(&response.to_assistant_message()).unwrap();
    let assistant: Message = serde_json::from_str(&persisted).unwrap();
    let mut request = ChatRequest::new("gemini-3.5-flash")
        .user("weather?")
        .message(assistant);
    for call in response.tool_calls() {
        request.push_tool_result(call, "Sunny");
    }
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty(), "{warnings:?}");
    let parts = &body["contents"][1]["parts"];
    assert_eq!(parts[0]["functionCall"]["id"], "call_234181");
    assert!(
        parts[0]["thoughtSignature"]
            .as_str()
            .unwrap()
            .starts_with("EpcDCpQD")
    );
    assert!(parts[1].get("thoughtSignature").is_none());
    assert_eq!(
        body["contents"][2]["parts"][1]["functionResponse"]["id"],
        "call_234182"
    );
}

/// An OpenAI custom tool call and its answer after a failover (ADR 0014):
/// both are another provider's content, dropped with warnings under every
/// policy; the function call of the same turn and its result are kept, so
/// the conversation ends on the function response. The caller's own raw
/// part (no origin) is sent verbatim.
#[test]
fn foreign_client_tool_content_is_dropped_and_raw_parts_are_sent() {
    let object = |value: Value| value.as_object().cloned().unwrap();
    let custom_call = ContentBlock::Other(
        OtherContent::new(
            "custom_tool_call",
            object(json!({"call_id": "call_2", "name": "sql", "input": "SELECT 1"})),
        )
        .with_origin("openai"),
    );
    let custom_answer = ContentPart::Other(
        OtherContent::new(
            "custom_tool_call_output",
            object(json!({"call_id": "call_2", "output": "1"})),
        )
        .with_origin("openai"),
    );
    for policy in [UnsupportedPolicy::Warn, UnsupportedPolicy::Error] {
        let mut request = ChatRequest::new("gemini-2.5-flash")
            .user("Weather in Paris, and count the users.")
            .message(Message::assistant_blocks(vec![
                ContentBlock::ToolCall(weather_call("c1", "Paris")),
                custom_call.clone(),
            ]))
            .message(Message::tool_result("c1", "get_weather", "Sunny"))
            .message(Message::user_parts(vec![custom_answer.clone()]));
        request.unsupported_policy = Some(policy);
        let (body, warnings) = encode(&request);
        let roles: Vec<&str> = body["contents"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, ["user", "model", "user"], "{policy:?}");
        assert!(
            body["contents"][2]["parts"][0]["functionResponse"].is_object(),
            "{body:#}"
        );
        assert!(!body.to_string().contains("SELECT 1"), "{body:#}");
        let kinds: Vec<WarningKind> = warnings.iter().map(|w| w.kind).collect();
        // Both drops share a code: one warning per request.
        assert_eq!(kinds, vec![WarningKind::Dropped], "{policy:?}");
    }

    let raw = OtherContent::new("text", object(json!({"text": "raw part"})));
    let (body, warnings) =
        encode(&ChatRequest::new("gemini-2.5-flash").user(vec![ContentPart::Other(raw)]));
    assert_eq!(body["contents"][0]["parts"], json!([{"text": "raw part"}]));
    assert!(warnings.is_empty(), "{warnings:?}");
}

/// A model the adapter does not know answers `Unknown` for reasoning, so
/// its reasoning config is sent in the current API's shape (Gemini 3's
/// `thinkingLevel`) for the API to decide, never dropped (ADR 0003,
/// principle 2). Gemini ≤ 2.0 is not dropped either (ADR 0015).
#[test]
fn an_unknown_models_reasoning_is_sent_not_dropped() {
    let request = ChatRequest::new("tunedModels/my-model")
        .user("Hi")
        .reasoning(ReasoningEffort::High);
    let (body, warnings) = encode(&request);
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        body["generationConfig"]["thinkingConfig"]["thinkingLevel"], "HIGH",
        "{body}"
    );

    let legacy = ChatRequest::new("gemini-2.0-flash")
        .user("Hi")
        .reasoning(ReasoningEffort::High);
    let (body, warnings) = encode(&legacy);
    assert_eq!(
        body["generationConfig"]["thinkingConfig"],
        json!({"includeThoughts": true, "thinkingBudget": 24576}),
        "{body}"
    );
    assert!(warnings.is_empty(), "{warnings:?}");
}

/// Replay state stays in a stored conversation, so a conversation that
/// moved here from another provider drops it on every turn: each drop is
/// coded, so its warning event goes out at `WARN` once per model, not once
/// per turn, and points to `retain_provider_state`.
#[test]
fn a_long_foreign_reasoning_history_drops_with_codes() {
    let mut request = ChatRequest::new("gemini-2.5-flash");
    for turn in 0..10 {
        request = request
            .user(format!("question {turn}"))
            .message(Message::assistant_blocks(vec![
                ContentBlock::Reasoning(Reasoning::new("thoughts").with_provider_data(
                    ProviderData::new().with_raw("openai", "signature", json!("opaque")),
                )),
                ContentBlock::text("answer"),
            ]));
    }
    let (_, warnings) = encode(&request.user("next"));
    // One warning per code per request, listing the ten messages.
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(matches!(&warnings[0].subject, Subject::Messages { messages } if messages.len() == 10));
    for warning in &warnings {
        assert_eq!(warning.code.as_deref(), Some("dropped.foreign_reasoning"));
        assert!(
            warning.message.contains("retain_provider_state"),
            "{}",
            warning.message
        );
    }
}

#[test]
fn safety_settings_and_extra_fields_reach_the_body() {
    let options = ChatRequestOptions::new()
        .safety_setting(crate::SafetySetting::new(
            "HARM_CATEGORY_DANGEROUS_CONTENT",
            "BLOCK_NONE",
        ))
        .generation_config_extra("responseModalities", json!(["TEXT"]))
        .extra("labels", json!({"team": "search"}));
    let (body, warnings) = encode(
        &ChatRequest::new("gemini-2.5-flash")
            .user("hi")
            .temperature(0.5)
            .with_option(options),
    );
    assert_eq!(
        body["safetySettings"],
        json!([{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}])
    );
    assert_eq!(
        body["generationConfig"],
        json!({"temperature": 0.5, "responseModalities": ["TEXT"]})
    );
    assert_eq!(body["labels"], json!({"team": "search"}));
    assert!(warnings.is_empty(), "{warnings:?}");

    // A generation config with only extra fields is still sent; the JSON
    // form nests the maps.
    let body = super::tests::body(&ChatRequest::new("gemini-2.5-flash").user("hi").with_options_json(
        NAMESPACE,
        json!({
            "safety_settings": [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"}],
            "generation_config_extra": {"mediaResolution": "MEDIA_RESOLUTION_LOW"},
            "extra": {"labels": {"a": "b"}}
        }),
    ));
    assert_eq!(
        body["generationConfig"],
        json!({"mediaResolution": "MEDIA_RESOLUTION_LOW"})
    );
    assert_eq!(body["safetySettings"][0]["threshold"], "OFF");
    assert_eq!(body["labels"], json!({"a": "b"}));
}

#[test]
fn extra_maps_cannot_set_fields_the_adapter_owns() {
    // Gemini reads both spellings, so `snake_case` and other cases count.
    let body_keys = RESERVED_FIELDS
        .iter()
        .map(|key| (*key).to_owned())
        .chain(["safety_settings".into(), "GENERATION_CONFIG".into()]);
    for key in body_keys {
        let request = ChatRequest::new("gemini-2.5-flash")
            .user("hi")
            .with_option(ChatRequestOptions::new().extra(key.clone(), "x"));
        let err = encode_request(&request).err().expect("refused");
        assert!(
            matches!(err, Error::InvalidRequest { .. }),
            "{key}: {err:?}"
        );
        assert!(
            err.to_string().contains("ChatRequestOptions::extra"),
            "{err}"
        );
    }
    let config_keys = GENERATION_CONFIG_FIELDS
        .iter()
        .map(|key| (*key).to_owned())
        .chain(["thinking_config".into(), "top_k".into()]);
    for key in config_keys {
        let request = ChatRequest::new("gemini-2.5-flash")
            .user("hi")
            .with_option(ChatRequestOptions::new().generation_config_extra(key.clone(), 1));
        let err = encode_request(&request).err().expect("refused");
        assert!(
            matches!(err, Error::InvalidRequest { .. }),
            "{key}: {err:?}"
        );
        assert!(
            err.to_string()
                .contains("ChatRequestOptions::generation_config_extra"),
            "{err}"
        );
    }
}

#[test]
fn every_body_and_generation_config_field_is_reserved() {
    // A field added to the body or the generation config must be reserved
    // too, or an extra map could send it twice.
    let request = ChatRequest::new("gemini-2.5-flash")
        .system("s")
        .user("hi")
        .temperature(0.5)
        .top_p(0.9)
        .top_k(3)
        .max_output_tokens(10)
        .stop(["x"])
        .seed(1)
        .presence_penalty(0.1)
        .frequency_penalty(0.1)
        .reasoning(ReasoningEffort::Low)
        .response_format(ResponseFormat::json_schema(json!({"type": "object"})))
        .tools([ToolDefinition::new("f", json!({"type": "object"}))])
        .tool_choice(ToolChoice::Auto)
        .with_option(
            ChatRequestOptions::new()
                .cached_content("cachedContents/x")
                .safety_setting(crate::SafetySetting::new("HARM_CATEGORY_HARASSMENT", "OFF")),
        );
    let body = body(&request);
    for key in body.as_object().unwrap().keys() {
        assert!(
            RESERVED_FIELDS.contains(&key.as_str()),
            "{key} is not reserved"
        );
    }
    for key in body["generationConfig"].as_object().unwrap().keys() {
        assert!(
            GENERATION_CONFIG_FIELDS.contains(&key.as_str()),
            "generationConfig.{key} is not reserved"
        );
    }
}

/// JSON options under `gemini`, the provider name, are not Gemini's
/// options (namespace `google-ai`): they are reported dropped, never sent
/// and never silently ignored.
#[test]
fn options_under_the_provider_name_are_reported_dropped() {
    let safety = json!({"safety_settings": [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}]});
    let request = ChatRequest::new("gemini-3-flash")
        .user("hi")
        .with_options_json("gemini", safety.clone())
        .unsupported_policy(UnsupportedPolicy::Error);
    let (body, warnings) = encode(&request);
    assert!(body.get("safetySettings").is_none(), "{body}");
    let [warning] = warnings.as_slice() else {
        panic!("one warning: {warnings:?}");
    };
    assert_eq!(warning.kind, WarningKind::Dropped);
    assert_eq!(
        warning.code.as_deref(),
        Some("dropped.provider_name_options")
    );
    assert_eq!(
        warning.subject,
        Subject::Option {
            namespace: "gemini".into()
        }
    );
    assert_eq!(
        warning.message,
        "Gemini reads options under `google-ai`, not under its name `gemini`; they were dropped; pass them under `google-ai`, or as `ChatRequestOptions`"
    );

    // Under `google-ai` they are sent, with no warning.
    let request = ChatRequest::new("gemini-3-flash")
        .user("hi")
        .with_options_json(NAMESPACE, safety);
    let (body, warnings) = encode(&request);
    assert_eq!(body["safetySettings"][0]["threshold"], "BLOCK_NONE");
    assert!(warnings.is_empty(), "{warnings:?}");
}
