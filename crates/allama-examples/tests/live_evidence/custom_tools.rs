//! Chat Completions custom tools (ADR 0014): the wire shape of a custom
//! tool call, unary and streamed. OpenAI does not document the streamed
//! shape; the decoder reads it as `fixtures/hand_written/` in
//! `allama-openai` assumes. The records keep each stream chunk, and the
//! raw body goes to `custom_tools/stream.sse` for use as a fixture.
//!
//! Whether a replayed custom call is accepted without its declaration is
//! `allama-openai`'s `chat_custom_call_replay_without_its_declaration`.

use allama_http::ReqwestTransport;
use serde_json::{Value, json};

use super::{env, env_or, post_json, record, record_raw, send};

const CHECK: &str = "custom_tools";

fn body(model: &str, stream: bool) -> Value {
    let mut body = json!({
        "model": model,
        "messages": [{"role": "user", "content": "Use the sql tool to count the rows of the users table."}],
        "tools": [{"type": "custom", "custom": {
            "name": "sql",
            "description": "Run a read-only SQLite query.",
            "format": {"type": "grammar", "grammar": {"syntax": "regex", "definition": "SELECT .+"}}
        }}],
        "tool_choice": "required",
    });
    if stream {
        body["stream"] = json!(true);
        body["stream_options"] = json!({"include_usage": true});
    }
    body
}

#[tokio::test]
#[ignore]
async fn custom_tool_wire_shape() {
    let transport = ReqwestTransport::new();
    let model = env_or("OPENAI_REASONING_MODEL", "gpt-5-nano");
    let auth = || ("authorization", format!("Bearer {}", env("OPENAI_API_KEY")));
    let url = "https://api.openai.com/v1/chat/completions";

    let unary = send(&transport, post_json(url, auth(), &body(&model, false))).await;
    record(
        CHECK,
        "unary",
        &json!({
            "model": model,
            "exchange": unary.record(),
            "message": unary.json().and_then(|b| b.pointer("/choices/0/message").cloned()),
        }),
    );

    let streamed = send(&transport, post_json(url, auth(), &body(&model, true))).await;
    record_raw(CHECK, "stream.sse", &streamed.body);
    // Every `data:` payload, in order; the chunks with a tool call delta
    // are the shape under test.
    let chunks: Vec<Value> = streamed
        .body
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim)
        .map(|data| serde_json::from_str(data).unwrap_or_else(|_| json!(data)))
        .collect();
    let tool_call_deltas: Vec<&Value> = chunks
        .iter()
        .filter_map(|chunk| chunk.pointer("/choices/0/delta/tool_calls"))
        .collect();
    let mut exchange = streamed.record();
    // The chunks are recorded below, one value each.
    exchange["body"] = json!("see `chunks`");
    record(
        CHECK,
        "streamed",
        &json!({
            "model": model,
            "exchange": exchange,
            "tool_call_deltas": tool_call_deltas,
            "chunks": chunks,
        }),
    );
}
