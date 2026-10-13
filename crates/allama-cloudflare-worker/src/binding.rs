//! Calls into the Workers AI binding (`env.AI`).
//!
//! The binding is reached through this crate's own `wasm_bindgen` import
//! ([`AiBinding`]), so the public API names no `worker` type: a `worker`
//! upgrade is not a breaking change of this crate. It also passes the
//! binding's third argument, the call options (an AI Gateway, …), which
//! `worker::Ai::run` cannot.
//!
//! Inputs are converted with `serde_wasm_bindgen`'s JSON-compatible
//! serializer: its default serializer turns JSON objects held in
//! `serde_json::Value` (tool parameter schemas, `response_format`) into ES
//! `Map`s that the binding does not read as plain objects.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use allama::{Error, ExtraMap, Result};
use send_wrapper::SendWrapper;
use serde::Serialize;
use serde::de::DeserializeOwned;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::PROVIDER_NAME;
use crate::error::map_worker_error;

#[wasm_bindgen]
extern "C" {
    /// The Workers AI binding object, as the runtime puts it in `env`.
    #[wasm_bindgen(extends = js_sys::Object)]
    #[derive(Debug, Clone)]
    pub(crate) type AiBinding;

    /// `AI.run(model, inputs, options)`, which returns a promise of the
    /// model's output (a `ReadableStream` with `stream: true`).
    #[wasm_bindgen(method, structural, catch, js_name = run)]
    fn run(
        this: &AiBinding,
        model: &str,
        inputs: &JsValue,
        options: &JsValue,
    ) -> std::result::Result<js_sys::Promise, JsValue>;
}

/// The binding, made `Send` for the single-threaded Workers runtime.
///
/// `SendWrapper` panics if the value is touched from another thread, which
/// a Worker never does.
pub(crate) struct Binding(SendWrapper<AiBinding>);

impl Binding {
    /// The binding object `value` (it is checked when a call runs).
    pub(crate) fn new(value: &JsValue) -> Self {
        Self(SendWrapper::new(value.clone().unchecked_into()))
    }

    /// Start `AI.run(model, input, options)`; the call options are left out
    /// when empty.
    fn start<T: Serialize>(
        &self,
        model: &str,
        input: &T,
        options: &ExtraMap,
    ) -> Result<SendFuture> {
        let input = to_js(input)?;
        let options = if options.is_empty() {
            JsValue::UNDEFINED
        } else {
            to_js(options)?
        };
        let promise = self
            .0
            .run(model, &input, &options)
            .map_err(|err| map_worker_error(worker::Error::from(err)))?;
        Ok(SendFuture(SendWrapper::new(JsFuture::from(promise))))
    }
}

/// Whether `value` looks like a Workers AI binding: an object with a `run`
/// method.
pub(crate) fn is_binding(value: &JsValue) -> bool {
    value.is_object()
        && js_sys::Reflect::get(value, &JsValue::from_str("run")).is_ok_and(|run| run.is_function())
}

/// Run `model` and return its JSON output.
pub(crate) async fn run_value<T: Serialize>(
    binding: &Binding,
    model: &str,
    input: &T,
    options: &ExtraMap,
) -> Result<serde_json::Value> {
    let output = binding.start(model, input, options)?.await?;
    serde_wasm_bindgen::from_value(output.take()).map_err(|err| {
        Error::unexpected_response(format!("unexpected Workers AI output for `{model}`"))
            .authored()
            .with_upstream_message(err.to_string())
            .with_provider(PROVIDER_NAME)
    })
}

/// Run `model` and return its output as JSON text (`JSON.stringify`), which
/// keeps the order of the output's keys: decoding into `serde_json::Value`
/// would sort them.
pub(crate) async fn run_text<T: Serialize>(
    binding: &Binding,
    model: &str,
    input: &T,
    options: &ExtraMap,
) -> Result<String> {
    let output = binding.start(model, input, options)?.await?;
    js_sys::JSON::stringify(&output.take())
        .ok()
        .and_then(|text| text.as_string())
        .ok_or_else(|| {
            Error::unexpected_response(format!("Workers AI output for `{model}` is not JSON"))
                .authored()
                .with_provider(PROVIDER_NAME)
        })
}

/// Run `model` and decode its JSON output into `U`.
pub(crate) async fn run_json<T, U>(
    binding: &Binding,
    model: &str,
    input: &T,
    options: &ExtraMap,
) -> Result<U>
where
    T: Serialize,
    U: DeserializeOwned,
{
    decode_output(model, run_value(binding, model, input, options).await?)
}

/// Decode a binding output into `U`.
pub(crate) fn decode_output<U: DeserializeOwned>(
    model: &str,
    output: serde_json::Value,
) -> Result<U> {
    // The binding reports no status or request id. The decoder's text can
    // quote the output, so it is upstream text.
    serde_json::from_value(output).map_err(|err| {
        Error::unexpected_response(format!("unexpected Workers AI output for `{model}`"))
            .authored()
            .with_upstream_message(err.to_string())
            .with_provider(PROVIDER_NAME)
    })
}

/// Run `model` and return its raw byte stream (used with `stream: true`).
pub(crate) async fn run_stream<T: Serialize>(
    binding: &Binding,
    model: &str,
    input: &T,
    options: &ExtraMap,
) -> Result<worker::ByteStream> {
    let output = binding.start(model, input, options)?.await?.take();
    match output.dyn_into::<worker::web_sys::ReadableStream>() {
        Ok(stream) => Ok(worker::ByteStream::from(stream)),
        // The model answered whole: it does not stream.
        Err(_) => Err(Error::unsupported_by_provider(
            None,
            format!("Workers AI returned no stream for `{model}`; the model does not stream"),
        )
        .authored()
        .with_provider(PROVIDER_NAME)),
    }
}

/// A binding call in flight, made `Send` like [`Binding`].
struct SendFuture(SendWrapper<JsFuture>);

impl Future for SendFuture {
    type Output = Result<SendWrapper<JsValue>>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut *self.get_mut().0).poll(cx).map(|output| {
            output
                .map(SendWrapper::new)
                .map_err(|err| map_worker_error(worker::Error::from(err)))
        })
    }
}

/// `input` as a plain JS value (see the module docs).
fn to_js<T: Serialize + ?Sized>(input: &T) -> Result<JsValue> {
    input
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(|err| Error::serialization(format!("failed to convert Workers AI input: {err}")))
}

/// Binding-level tests: run on `wasm32` with `wasm-bindgen-test`
/// (`cargo test --target wasm32-unknown-unknown -p allama-cloudflare-worker --lib`).
#[cfg(all(test, target_arch = "wasm32"))]
mod tests {
    use serde_json::{Value, json};
    use wasm_bindgen::JsCast;
    use wasm_bindgen_test::wasm_bindgen_test;

    use super::*;

    fn chat_input() -> Value {
        json!({
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [{"type": "function", "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
            }}],
            "response_format": {"type": "json_schema", "json_schema": {"type": "object"}},
            "max_tokens": 64
        })
    }

    fn get(value: &JsValue, key: &str) -> JsValue {
        js_sys::Reflect::get(value, &JsValue::from_str(key)).unwrap()
    }

    #[wasm_bindgen_test]
    fn inputs_become_plain_js_objects() {
        let input = to_js(&chat_input()).unwrap();
        let value: &JsValue = &input;
        assert!(!value.is_instance_of::<js_sys::Map>());
        let parameters = get(
            &get(&get(&get(value, "tools"), "0"), "function"),
            "parameters",
        );
        assert!(parameters.is_object());
        assert!(!parameters.is_instance_of::<js_sys::Map>());
        assert!(!get(value, "response_format").is_instance_of::<js_sys::Map>());

        let text: String = js_sys::JSON::stringify(value).unwrap().into();
        assert_eq!(serde_json::from_str::<Value>(&text).unwrap(), chat_input());
    }

    #[wasm_bindgen_test]
    fn the_default_serializer_would_produce_maps() {
        // Why `to_js` uses the JSON-compatible serializer: the default
        // `serde_wasm_bindgen` serializer turns JSON objects into ES `Map`s
        // that the binding does not read.
        let value = serde_wasm_bindgen::to_value(&chat_input()).unwrap();
        assert!(value.is_instance_of::<js_sys::Map>());
        let _ = value.dyn_into::<js_sys::Map>().unwrap();
    }

    #[wasm_bindgen_test]
    async fn calls_the_binding_with_the_input_and_the_call_options() {
        // A stand-in for `env.AI`: `run` echoes its arguments.
        let ai = js_sys::Function::new_with_args(
            "model, inputs, options",
            "return Promise.resolve({model, inputs, options: options === undefined ? null : options});",
        );
        let object = js_sys::Object::new();
        js_sys::Reflect::set(&object, &"run".into(), &ai).unwrap();
        let binding = Binding::new(&object);
        assert!(is_binding(&object));
        assert!(!is_binding(&JsValue::UNDEFINED));

        let options: ExtraMap = serde_json::from_value(json!({"gateway": {"id": "g"}})).unwrap();
        let output = run_value(&binding, "@cf/m", &json!({"a": {"b": 1}}), &options)
            .await
            .unwrap();
        assert_eq!(
            output,
            json!({"model": "@cf/m", "inputs": {"a": {"b": 1}}, "options": {"gateway": {"id": "g"}}})
        );
        let output = run_value(&binding, "@cf/m", &json!({}), &ExtraMap::new())
            .await
            .unwrap();
        assert_eq!(output["options"], Value::Null);
    }

    #[wasm_bindgen_test]
    fn from_env_checks_the_binding() {
        let env = js_sys::Object::new();
        let err = crate::Provider::from_env(&env, "AI").expect_err("no binding");
        assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
        assert!(err.message().contains("`AI`"), "{err}");

        let ai = js_sys::Object::new();
        let run = js_sys::Function::new_no_args("return Promise.resolve({});");
        js_sys::Reflect::set(&ai, &"run".into(), &run).unwrap();
        js_sys::Reflect::set(&env, &"AI".into(), &ai).unwrap();
        assert!(crate::Provider::from_env(&env, "AI").is_ok());
        // Not a binding: an object without `run`.
        js_sys::Reflect::set(&env, &"KV".into(), &js_sys::Object::new()).unwrap();
        assert!(crate::Provider::from_env(&env, "KV").is_err());
    }

    /// A binding whose `run` resolves to what `body` (a JS function body)
    /// returns.
    fn fake_provider(body: &str) -> crate::Provider {
        let run = js_sys::Function::new_with_args("model, inputs, options", body);
        let object = js_sys::Object::new();
        js_sys::Reflect::set(&object, &"run".into(), &run).unwrap();
        crate::Provider::new(&object)
    }

    fn hello() -> allama::ChatRequest {
        allama::ChatRequest::new("@cf/m").user("hi")
    }

    /// A model that answers whole when asked to
    /// stream does not stream, which is `Unsupported`, not a decode error.
    #[wasm_bindgen_test]
    async fn an_object_answer_to_a_stream_is_unsupported() {
        use allama::ChatProvider;

        let provider = fake_provider("return Promise.resolve({response: 'hello'});");
        let err = provider.chat_stream(&hello()).await.err().expect("refused");
        assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
        assert!(err.message().contains("does not stream"), "{err}");
    }

    /// `with_max_stream_event_size` bounds the
    /// events of the binding's stream.
    #[wasm_bindgen_test]
    async fn the_stream_event_size_is_capped() {
        use allama::{ChatProvider, ChatStreamExt};

        let streaming = r#"
            const sse = 'data: {"response":"' + 'x'.repeat(256) + '"}\n\n'
                + 'data: {"response":"","finish_reason":"stop"}\n\n'
                + 'data: [DONE]\n\n';
            return Promise.resolve(new ReadableStream({
                start(controller) {
                    controller.enqueue(new TextEncoder().encode(sse));
                    controller.close();
                }
            }));
        "#;
        let response = fake_provider(streaming)
            .chat_stream(&hello())
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap();
        assert_eq!(response.text(), Some("x".repeat(256)));

        let err = fake_provider(streaming)
            .with_max_stream_event_size(64)
            .chat_stream(&hello())
            .await
            .unwrap()
            .collect_response()
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Stream { .. }), "{err:?}");
        assert!(!err.is_retryable(), "{err:?}");
    }

    /// A decision runs the model with the Clef input, and its output is
    /// read in the order the model wrote it.
    #[wasm_bindgen_test]
    async fn decides_through_the_binding() {
        use allama::decision::Question;
        use allama::{DecisionProvider, DecisionRequest};

        let provider = fake_provider(
            r#"
            if (model !== '@cf/cloudflare/clef' || inputs.model !== 'clef'
                || options?.gateway?.id !== 'g') {
                return Promise.reject(new Error('5006: Bad input'));
            }
            const keys = Object.keys(inputs.questions.team.criteria).join(',');
            return Promise.resolve({
                model: 'clef',
                answers: {
                    team: {type: 'choice', choice: 'technical', probabilities: {technical: 0.8, billing: 0.2}, confidence: 0.3},
                    urgent: {type: 'noul', noul: keys === 'technical,billing' ? 0.9 : 0.1},
                },
                usage: {input_tokens: 3, output_tokens: 0},
            });
            "#,
        );
        let request = DecisionRequest::new("@cf/cloudflare/clef", "Checkout fails.")
            .question(
                "team",
                Question::choice("Which team?").options(["technical", "billing"]),
            )
            .question("urgent", Question::yes_no("Is it urgent?"))
            .with_option(
                crate::DecisionRequestOptions::new().run_option("gateway", json!({"id": "g"})),
            );
        let response = provider.decide(&request).await.unwrap();
        // The input kept the options' order.
        assert_eq!(response.yes_no("urgent").unwrap().probability, 0.9);
        // The output kept the model's order, not a sorted one.
        let team = response.choice("team").unwrap();
        assert_eq!(team.probabilities[0], ("technical".to_owned(), 0.8));
        assert_eq!(response.answers[0].0, "team");

        let err = fake_provider("return Promise.resolve(undefined);")
            .decide(&request)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::UnexpectedResponse { .. }), "{err:?}");
    }

    #[wasm_bindgen_test]
    async fn a_throwing_binding_is_an_error() {
        let ai = js_sys::Function::new_with_args(
            "",
            "throw new Error('3040: Capacity temporarily exceeded');",
        );
        let object = js_sys::Object::new();
        js_sys::Reflect::set(&object, &"run".into(), &ai).unwrap();
        let err = run_value(
            &Binding::new(&object),
            "@cf/m",
            &json!({}),
            &ExtraMap::new(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, Error::Overloaded { .. }), "{err:?}");
    }
}
