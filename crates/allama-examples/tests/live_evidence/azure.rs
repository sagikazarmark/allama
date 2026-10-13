//! Azure OpenAI (the v1 Chat Completions API).
//!
//! - The strict keyword list: `SchemaDialect::AzureStrict` removes the
//!   keywords Azure's structured outputs guide lists as unsupported. Which
//!   does Azure reject, and which does it accept or ignore?
//! - A reasoning deployment's handling of `parallel_tool_calls`, which the
//!   compat encoder sends as `false` beside a strict tool unless the caller
//!   set it (`SchemaDialect::parallel_tool_calls_break_strict`).

use allama_http::{HttpRequest, ReqwestTransport};
use serde_json::{Value, json};

use super::openai::{answer_text, strict_request};
use super::{env, honoured, keyword_case, parse_output, post_json, record, send};

const CHECK: &str = "azure";

/// The keywords `AzureStrict` removes or refuses, and `prefixItems`.
pub const KEYWORDS: &[&str] = &[
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "unevaluatedProperties",
    "minProperties",
    "maxProperties",
    "minItems",
    "maxItems",
    "uniqueItems",
    "contains",
    "unevaluatedItems",
    "minContains",
    "maxContains",
    "patternProperties",
    "propertyNames",
    "prefixItems",
];

fn request(body: &Value) -> HttpRequest {
    let resource = env("AZURE_OPENAI_RESOURCE");
    post_json(
        &format!("https://{resource}.openai.azure.com/openai/v1/chat/completions"),
        ("api-key", env("AZURE_OPENAI_API_KEY")),
        body,
    )
}

#[tokio::test]
#[ignore]
async fn azure_strict_keywords() {
    let transport = ReqwestTransport::new();
    let deployment = env("AZURE_OPENAI_DEPLOYMENT");
    for keyword in KEYWORDS {
        let case = keyword_case(keyword);
        let body = strict_request(&deployment, case.prompt, &case.schema);
        let exchange = send(&transport, request(&body)).await;
        let output = parse_output(answer_text(&exchange));
        record(
            CHECK,
            &format!("keyword_{}", case.keyword),
            &json!({
                "deployment": deployment,
                "keyword": case.keyword,
                "prompt": case.prompt,
                "sent_schema": case.schema,
                "exchange": exchange.record(),
                "output": output,
                "honoured": honoured(&case, &output),
            }),
        );
    }
}

#[tokio::test]
#[ignore]
async fn azure_reasoning_deployment_parallel_tool_calls() {
    let transport = ReqwestTransport::new();
    let deployment = env("AZURE_OPENAI_REASONING_DEPLOYMENT");
    let weather = |strict: bool| {
        json!({"type": "function", "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "strict": strict,
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": false
            }
        }})
    };
    for strict in [true, false] {
        for parallel in [None, Some(false), Some(true)] {
            let mut body = json!({
                "model": deployment,
                "messages": [{"role": "user", "content": "What is the weather in Lisbon and in Porto?"}],
                "max_completion_tokens": 2000,
                "tools": [weather(strict)],
            });
            if let Some(parallel) = parallel {
                body["parallel_tool_calls"] = json!(parallel);
            }
            let exchange = send(&transport, request(&body)).await;
            let tool_calls = exchange
                .json()
                .and_then(|b| b.pointer("/choices/0/message/tool_calls").cloned())
                .map_or(0, |calls| calls.as_array().map_or(0, Vec::len));
            let parallel_name = match parallel {
                None => "unset",
                Some(false) => "false",
                Some(true) => "true",
            };
            let strict_name = if strict { "strict" } else { "non_strict" };
            record(
                CHECK,
                &format!("reasoning_parallel_tool_calls_{parallel_name}_{strict_name}"),
                &json!({
                    "deployment": deployment,
                    "parallel_tool_calls": parallel,
                    "strict_tool": strict,
                    "exchange": exchange.record(),
                    "tool_calls_in_answer": tool_calls,
                }),
            );
        }
    }
}
