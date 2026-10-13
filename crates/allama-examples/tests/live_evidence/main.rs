//! Live evidence checks.
//!
//! These checks record how the real provider APIs behave, as evidence for
//! later decisions: the schema dialects, the Azure parallel tool call rule,
//! the Chat Completions custom tool stream, and the rate-limit headers the
//! retry policy reads. They assert nothing about the API's answer (only
//! that a request could be sent at all); each prints what it observed and,
//! when `ALLAMA_EVIDENCE_DIR` is set, writes it to
//! `$ALLAMA_EVIDENCE_DIR/<check>/<case>.json`.
//!
//! Most checks send hand-written requests through `allama-http`'s `reqwest`
//! transport, not through an adapter: the question is what the API does
//! with a construct, and an adapter would rewrite it first. Where the
//! question is about what allama sends, the case also runs the bundled
//! strict dialect and sends its output (the `allama` variants).
//!
//! Every check is `#[ignore]`d and none runs in CI, including the scheduled
//! e2e run. They spend real tokens, and the rate-limit checks deliberately
//! send bursts until the API throttles them. Run one provider's checks with
//! a name filter:
//!
//! ```sh
//! ALLAMA_EVIDENCE_DIR=target/evidence \
//!   cargo test -p allama-examples --test live_evidence -- --ignored --nocapture gemini_
//! ```
//!
//! | Filter | Checks | Environment |
//! | --- | --- | --- |
//! | `gemini_` | `oneOf`/`const`, boolean subschemas, `$schema` in `responseJsonSchema` | `GEMINI_API_KEY`; `GEMINI_MODEL` (default `gemini-3.5-flash-lite`) |
//! | `openai_` | `properties` with a sibling `anyOf`, an object root with a sibling `oneOf` (a flattened enum), keywords OpenAI's docs do not mention | `OPENAI_API_KEY`; `OPENAI_MODEL` (default `gpt-4.1-nano`) |
//! | `azure_` | the strict keyword list, and `parallel_tool_calls` on a reasoning deployment | `AZURE_OPENAI_RESOURCE`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT` (keywords), `AZURE_OPENAI_REASONING_DEPLOYMENT` (parallel tool calls) |
//! | `anthropic_` | numeric `format`s, nested optional property counting, a `$ref` with siblings before and after the merge | `ANTHROPIC_API_KEY`; `ANTHROPIC_MODEL` (default `claude-haiku-4-5-20251001`) |
//! | `custom_tool_` | the Chat Completions custom tool call, unary and streamed, as sent on the wire | `OPENAI_API_KEY`; `OPENAI_REASONING_MODEL` (default `gpt-5-nano`) |
//! | `rate_limit_` | real 429/529 headers (`retry-after`, `retry-after-ms`, `x-should-retry`) from OpenAI and Anthropic | the OpenAI and Anthropic variables above; `ALLAMA_EVIDENCE_BURST` (default 300 requests) and `ALLAMA_EVIDENCE_CONCURRENCY` (default 50) |
//!
//! Three more live evidence checks run through the adapters and live with their
//! provider's e2e tests, which run weekly:
//!
//! - `cargo test -p allama-anthropic --test e2e -- --ignored --nocapture
//!   anthropic_strict_numeric_fields`: numeric fields through extraction,
//!   in native and forced-tool mode.
//! - `cargo test -p allama-openai --test e2e -- --ignored --nocapture
//!   chat_custom`: `chat_custom_tool_round_trip` (the decoder's reading of
//!   the custom tool stream) and
//!   `chat_custom_call_replay_without_its_declaration` (whether a replayed
//!   custom call is accepted when the request does not declare the tool).
//!
//! The records keep only these response headers: the retry signals, the
//! rate-limit headers, request ids and the content type. Organization and
//! project headers are left out, and request headers (the credentials) are
//! never recorded. Review the files before committing any of them.

// Sends through `allama-http`'s `reqwest` transport (a dev-dependency).

mod anthropic;
mod azure;
mod custom_tools;
mod gemini;
mod openai;
mod rate_limits;

use std::path::PathBuf;
use std::time::Duration;

use allama::UnsupportedPolicy;
use allama::adapter::Degradations;
use allama::adapter::schema::StrictSchemaError;
use allama_http::{
    Headers, HttpRequest, HttpTransport, ReqwestTransport, TransportError, error_context,
};
use serde_json::{Value, json};

/// Record `observation` as `check/case`: print it, and write it to
/// `$ALLAMA_EVIDENCE_DIR/<check>/<case>.json` when the variable is set.
pub fn record(check: &str, case: &str, observation: &Value) {
    let text = serde_json::to_string_pretty(observation).expect("a JSON value serializes");
    println!("== {check}/{case} ==\n{text}\n");
    write_file(check, &format!("{}.json", slug(case)), &format!("{text}\n"));
}

/// Write `contents` verbatim to `$ALLAMA_EVIDENCE_DIR/<check>/<file>` when
/// the variable is set, for wire data worth keeping as it came (an SSE
/// body).
pub fn record_raw(check: &str, file: &str, contents: &str) {
    write_file(check, file, contents);
}

/// One HTTP exchange: what the API answered.
pub struct Exchange {
    pub status: u16,
    pub headers: Headers,
    pub body: String,
}

impl Exchange {
    /// Whether the API accepted the request (a 2xx status).
    pub fn accepted(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// The body as JSON, when it is JSON.
    pub fn json(&self) -> Option<Value> {
        serde_json::from_str(&self.body).ok()
    }

    /// The record of this exchange: the status, the kept response headers,
    /// the retry signals allama reads from them (`allama_http::error_context`),
    /// and the body (as JSON when it parses).
    pub fn record(&self) -> Value {
        let headers: serde_json::Map<String, Value> = self
            .headers
            .iter()
            .filter(|(name, _)| kept_header(name))
            .map(|(name, value)| (name.to_ascii_lowercase(), json!(value)))
            .collect();
        let context = error_context(self.status, &self.headers);
        json!({
            "status": self.status,
            "accepted": self.accepted(),
            "headers": headers,
            "allama_reads": {
                "retry_after_ms": context.retry_after.map(|d| d.as_millis() as u64),
                "server_retry_directive": context.server_retry_directive,
                "request_id": context.request_id,
            },
            "body": self.json().unwrap_or_else(|| json!(self.body)),
        })
    }
}

/// Send `request` and read the whole answer.
///
/// # Panics
///
/// When no response arrives (a transport failure): there is nothing to
/// record.
pub async fn send(transport: &ReqwestTransport, request: HttpRequest) -> Exchange {
    try_send(transport, request)
        .await
        .unwrap_or_else(|error| panic!("no answer: {error}"))
}

/// Send `request` and read the whole answer, or say why none arrived.
pub async fn try_send(
    transport: &ReqwestTransport,
    request: HttpRequest,
) -> Result<Exchange, TransportError> {
    let response = transport
        .send(request.timeout(Some(Duration::from_secs(180))))
        .await?;
    let status = response.status;
    let headers = response.headers.clone();
    let body = response.text().await?;
    Ok(Exchange {
        status,
        headers,
        body,
    })
}

/// A JSON `POST` with a secret header.
pub fn post_json(url: &str, secret: (&str, String), body: &Value) -> HttpRequest {
    HttpRequest::post(url, body.to_string())
        .header("content-type", "application/json")
        .sensitive_header(secret.0, secret.1)
}

/// The value of the environment variable `name`.
///
/// # Panics
///
/// When it is not set, like the e2e tests.
pub fn env(name: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| panic!("{name} must be set to run this evidence check"))
}

/// The value of the environment variable `name`, or `default`.
pub fn env_or(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_owned())
}

/// What a bundled strict dialect made of a schema: its output and
/// warnings, or why it could not express the schema.
pub fn dialect_record(
    result: &Result<Value, StrictSchemaError>,
    degradations: &Degradations,
) -> Value {
    match result {
        Ok(schema) => json!({
            "outcome": "rewritten",
            "schema": schema,
            "warnings": degradations
                .warnings()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
        }),
        Err(error) => json!({"outcome": "not sent strict", "error": error.to_string()}),
    }
}

/// A fresh `Degradations` for running a dialect outside an adapter.
pub fn scratch(provider: &'static str) -> Degradations {
    Degradations::new(provider, UnsupportedPolicy::Warn)
}

/// A strict object schema with one property, `value`, and `object_keywords`
/// merged into the root object.
pub fn one_property(property: Value, object_keywords: Value) -> Value {
    let mut schema = json!({
        "type": "object",
        "properties": {"value": property},
        "required": ["value"],
        "additionalProperties": false,
    });
    if let (Value::Object(root), Value::Object(extra)) = (&mut schema, object_keywords) {
        root.extend(extra);
    }
    schema
}

/// A JSON Schema keyword under test, placed where it applies.
pub struct KeywordCase {
    pub keyword: &'static str,
    pub schema: Value,
    /// A prompt that tempts the model to break the keyword, so an accepted
    /// keyword also shows whether it is enforced.
    pub prompt: &'static str,
    /// Whether the model's output kept the keyword, where that can be told
    /// from one answer.
    pub honoured: Option<OutputCheck>,
}

/// Whether a model's output kept a constraint.
pub type OutputCheck = fn(&Value) -> bool;

/// The keyword cases, by keyword. Unknown keywords panic.
pub fn keyword_case(keyword: &'static str) -> KeywordCase {
    let string = |extra: Value| {
        let mut property = json!({"type": "string"});
        if let (Value::Object(p), Value::Object(e)) = (&mut property, extra) {
            p.extend(e);
        }
        property
    };
    let integer = |extra: Value| {
        let mut property = json!({"type": "integer"});
        if let (Value::Object(p), Value::Object(e)) = (&mut property, extra) {
            p.extend(e);
        }
        property
    };
    let strings = |extra: Value| {
        let mut property = json!({"type": "array", "items": {"type": "string"}});
        if let (Value::Object(p), Value::Object(e)) = (&mut property, extra) {
            p.extend(e);
        }
        property
    };
    // A function, not a closure, so the checks below stay `fn` pointers.
    fn value(output: &Value) -> Value {
        output.get("value").cloned().unwrap_or(Value::Null)
    }
    let (schema, prompt, honoured): (Value, &'static str, Option<OutputCheck>) = match keyword {
        "minLength" => (
            one_property(string(json!({"minLength": 12})), json!({})),
            "Set value to \"hi\".",
            Some(|o| value(o).as_str().is_some_and(|s| s.chars().count() >= 12)),
        ),
        "maxLength" => (
            one_property(string(json!({"maxLength": 3})), json!({})),
            "Set value to \"a much longer sentence\".",
            Some(|o| value(o).as_str().is_some_and(|s| s.chars().count() <= 3)),
        ),
        "pattern" => (
            one_property(string(json!({"pattern": "^[a-z]+$"})), json!({})),
            "Set value to \"HELLO 123\".",
            Some(|o| {
                value(o)
                    .as_str()
                    .is_some_and(|s| !s.is_empty() && s.chars().all(|c| c.is_ascii_lowercase()))
            }),
        ),
        "format" => (
            one_property(string(json!({"format": "date-time"})), json!({})),
            "Set value to \"next Tuesday\".",
            None,
        ),
        "minimum" => (
            one_property(integer(json!({"minimum": 10})), json!({})),
            "Set value to 3.",
            Some(|o| value(o).as_i64().is_some_and(|n| n >= 10)),
        ),
        "maximum" => (
            one_property(integer(json!({"maximum": 10})), json!({})),
            "Set value to 99.",
            Some(|o| value(o).as_i64().is_some_and(|n| n <= 10)),
        ),
        "exclusiveMinimum" => (
            one_property(integer(json!({"exclusiveMinimum": 10})), json!({})),
            "Set value to 10.",
            Some(|o| value(o).as_i64().is_some_and(|n| n > 10)),
        ),
        "exclusiveMaximum" => (
            one_property(integer(json!({"exclusiveMaximum": 10})), json!({})),
            "Set value to 10.",
            Some(|o| value(o).as_i64().is_some_and(|n| n < 10)),
        ),
        "multipleOf" => (
            one_property(integer(json!({"multipleOf": 5})), json!({})),
            "Set value to 7.",
            Some(|o| value(o).as_i64().is_some_and(|n| n % 5 == 0)),
        ),
        "minItems" => (
            one_property(strings(json!({"minItems": 4})), json!({})),
            "Set value to [\"a\"].",
            Some(|o| value(o).as_array().is_some_and(|a| a.len() >= 4)),
        ),
        "maxItems" => (
            one_property(strings(json!({"maxItems": 2})), json!({})),
            "Set value to [\"a\", \"b\", \"c\", \"d\", \"e\"].",
            Some(|o| value(o).as_array().is_some_and(|a| a.len() <= 2)),
        ),
        "uniqueItems" => (
            one_property(strings(json!({"uniqueItems": true})), json!({})),
            "Set value to [\"a\", \"a\", \"b\", \"b\"], keeping the duplicates.",
            Some(|o| {
                value(o).as_array().is_some_and(|a| {
                    a.iter()
                        .enumerate()
                        .all(|(i, x)| a.iter().skip(i + 1).all(|y| y != x))
                })
            }),
        ),
        "contains" => (
            one_property(strings(json!({"contains": {"const": "seven"}})), json!({})),
            "Set value to [\"one\", \"two\"] and nothing else.",
            Some(|o| {
                value(o)
                    .as_array()
                    .is_some_and(|a| a.iter().any(|x| x == "seven"))
            }),
        ),
        "minContains" => (
            one_property(
                strings(json!({"contains": {"const": "x"}, "minContains": 2})),
                json!({}),
            ),
            "Set value to [\"x\", \"y\"].",
            Some(|o| {
                value(o)
                    .as_array()
                    .is_some_and(|a| a.iter().filter(|x| *x == "x").count() >= 2)
            }),
        ),
        "maxContains" => (
            one_property(
                strings(json!({"contains": {"const": "x"}, "maxContains": 1})),
                json!({}),
            ),
            "Set value to [\"x\", \"x\", \"x\"].",
            Some(|o| {
                value(o)
                    .as_array()
                    .is_some_and(|a| a.iter().filter(|x| *x == "x").count() <= 1)
            }),
        ),
        "prefixItems" => (
            one_property(
                json!({
                    "type": "array",
                    "prefixItems": [{"type": "string"}, {"type": "integer"}],
                    "items": {"type": "integer"},
                }),
                json!({}),
            ),
            "Set value to [1, 2, 3].",
            Some(|o| {
                value(o)
                    .as_array()
                    .and_then(|a| a.first())
                    .is_some_and(Value::is_string)
            }),
        ),
        "unevaluatedItems" => (
            one_property(strings(json!({"unevaluatedItems": false})), json!({})),
            "Set value to [\"a\", \"b\"].",
            None,
        ),
        "minProperties" => (
            one_property(json!({"type": "string"}), json!({"minProperties": 1})),
            "Set value to \"a\".",
            None,
        ),
        "maxProperties" => (
            one_property(json!({"type": "string"}), json!({"maxProperties": 1})),
            "Set value to \"a\".",
            None,
        ),
        "unevaluatedProperties" => (
            one_property(
                json!({"type": "string"}),
                json!({"unevaluatedProperties": false}),
            ),
            "Set value to \"a\".",
            None,
        ),
        "patternProperties" => (
            one_property(
                json!({"type": "string"}),
                json!({"patternProperties": {"^x_": {"type": "string"}}}),
            ),
            "Set value to \"a\".",
            None,
        ),
        "propertyNames" => (
            one_property(
                json!({"type": "string"}),
                json!({"propertyNames": {"pattern": "^[a-z]+$"}}),
            ),
            "Set value to \"a\".",
            None,
        ),
        other => panic!("no keyword case for `{other}`"),
    };
    KeywordCase {
        keyword,
        schema,
        prompt,
        honoured,
    }
}

/// Parse `text` as the model's JSON answer.
pub fn parse_output(text: Option<String>) -> Value {
    match text {
        Some(text) => serde_json::from_str(&text).unwrap_or(Value::String(text)),
        None => Value::Null,
    }
}

/// Whether `output` kept the keyword: `null` when the answer cannot tell.
pub fn honoured(case: &KeywordCase, output: &Value) -> Value {
    match case.honoured {
        Some(check) if !output.is_null() => json!(check(output)),
        _ => Value::Null,
    }
}

fn kept_header(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    matches!(
        name.as_str(),
        "retry-after"
            | "retry-after-ms"
            | "x-should-retry"
            | "request-id"
            | "x-request-id"
            | "apim-request-id"
            | "content-type"
    ) || name.starts_with("x-ratelimit-")
        || name.starts_with("anthropic-ratelimit-")
}

fn write_file(check: &str, file: &str, contents: &str) {
    let Some(dir) = std::env::var_os("ALLAMA_EVIDENCE_DIR") else {
        return;
    };
    let dir = PathBuf::from(dir).join(check);
    std::fs::create_dir_all(&dir)
        .unwrap_or_else(|error| panic!("creating {}: {error}", dir.display()));
    let path = dir.join(file);
    std::fs::write(&path, contents)
        .unwrap_or_else(|error| panic!("writing {}: {error}", path.display()));
}

fn slug(case: &str) -> String {
    case.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

// Offline self-tests of the recording helpers; these run in CI.

#[test]
fn every_listed_keyword_has_a_case() {
    for keyword in openai::UNDOCUMENTED_KEYWORDS.iter().chain(azure::KEYWORDS) {
        let case = keyword_case(keyword);
        assert_eq!(case.keyword, *keyword);
        assert!(
            case.schema.to_string().contains(&format!("\"{keyword}\"")),
            "{keyword}: {}",
            case.schema
        );
    }
}

#[test]
fn keyword_checks_tell_kept_from_broken() {
    let cases = [
        (
            "uniqueItems",
            json!({"value": ["a", "b"]}),
            json!({"value": ["a", "a"]}),
        ),
        (
            "contains",
            json!({"value": ["seven"]}),
            json!({"value": ["one"]}),
        ),
        (
            "prefixItems",
            json!({"value": ["a", 1]}),
            json!({"value": [1, 2]}),
        ),
        ("maximum", json!({"value": 10}), json!({"value": 99})),
        (
            "pattern",
            json!({"value": "abc"}),
            json!({"value": "ABC 1"}),
        ),
    ];
    for (keyword, kept, broken) in cases {
        let case = keyword_case(keyword);
        assert_eq!(honoured(&case, &kept), json!(true), "{keyword}");
        assert_eq!(honoured(&case, &broken), json!(false), "{keyword}");
        assert_eq!(honoured(&case, &Value::Null), Value::Null, "{keyword}");
    }
}

#[test]
fn records_keep_only_the_listed_headers() {
    let mut headers = Headers::new();
    headers.insert("Retry-After", "2");
    headers.insert("x-should-retry", "true");
    headers.insert("x-ratelimit-remaining-requests", "0");
    headers.insert("anthropic-ratelimit-tokens-reset", "2026-10-03T00:00:00Z");
    headers.insert("openai-organization", "acme-corp");
    headers.insert("openai-project", "proj_123");
    headers.insert("set-cookie", "session=secret");
    let exchange = Exchange {
        status: 429,
        headers,
        body: r#"{"error":{"type":"rate_limit_error"}}"#.into(),
    };
    let record = exchange.record();
    let kept: Vec<&str> = record["headers"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        kept,
        [
            "anthropic-ratelimit-tokens-reset",
            "retry-after",
            "x-ratelimit-remaining-requests",
            "x-should-retry",
        ]
    );
    assert_eq!(record["allama_reads"]["retry_after_ms"], json!(2000));
    assert_eq!(
        record["allama_reads"]["server_retry_directive"],
        json!(true)
    );
    assert_eq!(record["body"]["error"]["type"], json!("rate_limit_error"));
    assert_eq!(record["accepted"], json!(false));
}

#[test]
fn case_names_become_file_names() {
    assert_eq!(slug("keyword_minLength"), "keyword_minLength");
    assert_eq!(slug("a/b c"), "a_b_c");
}
