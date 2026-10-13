//! Hosted tool items (tools the endpoint runs itself) and the portable
//! provider tool blocks.
//!
//! The Responses API reports a hosted tool as one output item holding both
//! the call and its outcome (`web_search_call` with `action` and `status`,
//! `code_interpreter_call` with `code` and `outputs`, …). It becomes two
//! blocks: a [`ProviderToolCall`] whose arguments are the item's input fields,
//! and a [`ProviderToolResult`] whose output is everything else. Both record
//! the item `type` in provider data; replay merges them back into the item.

use allama::{ExtraMap, ProviderData, ProviderToolCall, ProviderToolResult};
use serde_json::Value;

/// Provider-data key holding the output item `type`.
const ITEM_TYPE: &str = "type";

/// Item types of hosted tools, with the portable tool name and the item
/// fields that are the call's input. `mcp_call` carries its own `name` and
/// `arguments`. Client-executed items (`function_call`, `computer_call`,
/// `local_shell_call`, `custom_tool_call`) are not hosted tools.
fn hosted(item_type: &str) -> Option<(Option<&'static str>, &'static [&'static str])> {
    Some(match item_type {
        "web_search_call" => (Some("web_search"), &["action"]),
        "file_search_call" => (Some("file_search"), &["queries"]),
        "code_interpreter_call" => (Some("code_interpreter"), &["code", "container_id"]),
        "image_generation_call" => (Some("image_generation"), &[]),
        "mcp_call" => (None, &["server_label"]),
        _ => return None,
    })
}

/// Returns `true` for output items of hosted tools.
pub(crate) fn is_hosted(item_type: &str) -> bool {
    hosted(item_type).is_some()
}

/// The portable tool name of a hosted item, as known when it is added.
pub(crate) fn tool_name(item: &ExtraMap) -> Option<String> {
    let item_type = item.get("type")?.as_str()?;
    match hosted(item_type)? {
        (Some(name), _) => Some(name.to_owned()),
        (None, _) => item.get("name")?.as_str().map(str::to_owned),
    }
}

/// Provider data recording the item type.
pub(crate) fn item_type_data(item_type: &str, namespace: &str) -> ProviderData {
    ProviderData::new().with_raw(namespace, ITEM_TYPE, Value::String(item_type.to_owned()))
}

/// Split a finished hosted item into its call and result blocks. `None` when
/// the item is not a hosted tool or has no id.
pub(crate) fn decode(
    item: &ExtraMap,
    namespace: &str,
) -> Option<(ProviderToolCall, ProviderToolResult)> {
    let mut fields = item.clone();
    let item_type = match fields.remove("type") {
        Some(Value::String(kind)) => kind,
        _ => return None,
    };
    let (name, input_fields) = hosted(&item_type)?;
    let id = match fields.remove("id") {
        Some(Value::String(id)) => id,
        _ => return None,
    };
    let (name, arguments) = match name {
        Some(name) => {
            let mut input = ExtraMap::new();
            for field in input_fields {
                if let Some(value) = fields.remove(*field) {
                    input.insert((*field).to_owned(), value);
                }
            }
            (name.to_owned(), Value::Object(input).to_string())
        }
        None => {
            let name = match fields.remove("name") {
                Some(Value::String(name)) => name,
                _ => return None,
            };
            let arguments = match fields.remove("arguments") {
                Some(Value::String(arguments)) => arguments,
                Some(other) => other.to_string(),
                None => "{}".to_owned(),
            };
            let mut data = item_type_data(&item_type, namespace);
            for field in input_fields {
                if let Some(value) = fields.remove(*field) {
                    data.insert_raw(namespace, *field, value);
                }
            }
            let call = ProviderToolCall::new(id.clone(), name, arguments)
                .with_origin(namespace)
                .with_provider_data(data);
            return Some((call, result(id, fields, &item_type, namespace)));
        }
    };
    let call = ProviderToolCall::new(id.clone(), name, arguments)
        .with_origin(namespace)
        .with_provider_data(item_type_data(&item_type, namespace));
    Some((call, result(id, fields, &item_type, namespace)))
}

fn result(id: String, output: ExtraMap, item_type: &str, namespace: &str) -> ProviderToolResult {
    let failed = output.get("status").and_then(Value::as_str) == Some("failed")
        || output.get("error").is_some_and(|e| !e.is_null());
    ProviderToolResult::new(id, Value::Object(output))
        .with_error(failed)
        .with_origin(namespace)
        .with_provider_data(item_type_data(item_type, namespace))
}

/// Rebuild the output item of a hosted call (and its result, when known).
/// `None` when the call is not from `namespace` or lacks its item type.
pub(crate) fn encode(
    call: &ProviderToolCall,
    result: Option<&ProviderToolResult>,
    namespace: &str,
) -> Option<ExtraMap> {
    if call.origin.as_deref() != Some(namespace) {
        return None;
    }
    let data = call.provider_data.namespace(namespace)?;
    let item_type = data.get(ITEM_TYPE)?.as_str()?;
    let (name, input_fields) = hosted(item_type)?;
    let mut item = match result.map(|r| &r.output) {
        Some(Value::Object(output)) => output.clone(),
        _ => ExtraMap::new(),
    };
    item.insert("type".into(), Value::String(item_type.to_owned()));
    item.insert("id".into(), Value::String(call.id.clone()));
    match name {
        Some(_) => {
            if let Ok(Value::Object(input)) = serde_json::from_str(&call.arguments) {
                item.extend(input);
            }
        }
        None => {
            item.insert("name".into(), Value::String(call.name.clone()));
            item.insert("arguments".into(), Value::String(call.arguments.clone()));
            for field in input_fields {
                if let Some(value) = data.get(*field) {
                    item.insert((*field).to_owned(), value.clone());
                }
            }
        }
    }
    Some(item)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn map(value: Value) -> ExtraMap {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn hosted_items_split_and_merge_back() {
        for item in [
            json!({"type": "web_search_call", "id": "ws_1", "status": "completed", "action": {"type": "search", "query": "rust"}}),
            json!({"type": "code_interpreter_call", "id": "ci_1", "status": "completed", "code": "print(1)", "container_id": "cntr_1", "outputs": [{"type": "logs", "logs": "1"}]}),
            json!({"type": "file_search_call", "id": "fs_1", "status": "completed", "queries": ["q"], "results": null}),
            json!({"type": "image_generation_call", "id": "ig_1", "status": "completed", "result": "iVBOR"}),
            json!({"type": "mcp_call", "id": "mcp_1", "name": "roll", "arguments": "{\"sides\":6}", "server_label": "dice", "output": "4", "error": null}),
        ] {
            let item = map(item);
            let (call, result) = decode(&item, "openai").unwrap();
            assert_eq!(call.id, result.call_id);
            assert_eq!(call.origin.as_deref(), Some("openai"));
            assert!(!result.is_error);
            assert_eq!(encode(&call, Some(&result), "openai"), Some(item));
        }
    }

    #[test]
    fn names_arguments_and_errors() {
        let (call, result) = decode(
            &map(json!({"type": "web_search_call", "id": "ws_1", "status": "failed", "action": {"type": "search", "query": "rust"}})),
            "openai",
        )
        .unwrap();
        assert_eq!(call.name, "web_search");
        assert_eq!(
            serde_json::from_str::<Value>(&call.arguments).unwrap(),
            json!({"action": {"type": "search", "query": "rust"}})
        );
        assert!(result.is_error);
        assert_eq!(result.output, json!({"status": "failed"}));

        let (call, result) = decode(
            &map(json!({"type": "mcp_call", "id": "mcp_1", "name": "roll", "arguments": "{}", "server_label": "dice", "error": "boom"})),
            "openai",
        )
        .unwrap();
        assert_eq!(call.name, "roll");
        assert!(result.is_error);

        assert!(
            decode(
                &map(json!({"type": "computer_call", "id": "c_1"})),
                "openai"
            )
            .is_none()
        );
        assert!(decode(&map(json!({"type": "web_search_call"})), "openai").is_none());
        assert_eq!(
            tool_name(&map(json!({"type": "mcp_call", "name": "roll"}))).as_deref(),
            Some("roll")
        );
    }

    #[test]
    fn foreign_calls_are_not_encoded() {
        let call = ProviderToolCall::new("ws_1", "web_search", "{}").with_origin("anthropic");
        assert!(encode(&call, None, "openai").is_none());
        let untyped = ProviderToolCall::new("ws_1", "web_search", "{}").with_origin("openai");
        assert!(encode(&untyped, None, "openai").is_none());
    }
}
