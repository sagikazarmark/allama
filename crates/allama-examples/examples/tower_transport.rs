//! # A tower stack as the transport
//!
//! An application with a `tower` stack (a concurrency limit, a rate limit,
//! its own instrumentation) puts it between its providers and the HTTP
//! client: [`TowerTransport`] makes any `tower` service over `http` types a
//! transport, which every provider builder takes through `http_transport`.
//! It needs `allama-http` with the `tower` feature.
//!
//! At the bottom of the stack goes the HTTP client:
//!
//! - [`ReqwestTransport::into_service`]: the default transport's client,
//!   with its guarantees (no redirects, URL redaction, failure
//!   classification). It needs the `reqwest` feature too.
//! - `tower-reqwest`'s `HttpClientService`, for a `reqwest` client the
//!   application builds itself, behind a `map_request` layer that converts
//!   the body. The client must not follow redirects:
//!   `reqwest::Client::new()` does, and would send the provider's key to the
//!   redirect's host.
//!
//! `TowerTransport` bounds every non-streaming call with the provider's
//! request timeout, and a streaming call keeps the provider's stream
//! timeouts, whatever the stack. A concurrency limit releases its permit
//! when the response headers arrive: it bounds requests being set up, not
//! open streams.
//!
//! The applications guide quotes it (`docs/guides/applications.md`, "A
//! tower stack as the transport"). It needs no credentials: the provider
//! talks to a stand-in OpenAI-compatible server it starts itself. Run it
//! with `cargo run -p allama-examples --example tower_transport`, and its
//! tests with `cargo test -p allama-examples --example tower_transport`.

use std::time::Duration;

use allama::prelude::*;
use allama_http::http;
use allama_http::{HttpTransport, RequestBody, ReqwestTransport, TowerTransport};
use allama_openai_compat::Provider;
use futures_util::StreamExt;
use tower::ServiceBuilder;
use tower_reqwest::HttpClientService;

/// The stack over the default transport's client: at most four requests
/// being set up at once.
fn transport() -> impl HttpTransport {
    let stack = ServiceBuilder::new()
        .concurrency_limit(4)
        .service(ReqwestTransport::new().into_service());
    TowerTransport::new(stack)
}

/// The same stack over a `reqwest` client of the application's own, through
/// `tower-reqwest`. The client follows no redirects, like the default one.
fn tower_reqwest_transport() -> reqwest::Result<impl HttpTransport> {
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(10))
        .build()?;
    let stack = ServiceBuilder::new()
        .concurrency_limit(4)
        .map_request(|request: http::Request<RequestBody>| request.map(reqwest::Body::wrap))
        .service(HttpClientService::new(client));
    Ok(TowerTransport::new(stack))
}

/// An OpenAI-compatible provider at `base_url`, over `transport`.
fn provider(base_url: &str, transport: impl HttpTransport) -> Result<Provider> {
    Provider::builder()
        .base_url(base_url)
        .bearer_token("local-key")
        .http_transport(transport)
        .build()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let base_url = local::spawn(local::Answer::Whole).await?;
    let providers = [
        ("into_service", provider(&base_url, transport())?),
        (
            "tower-reqwest",
            provider(&base_url, tower_reqwest_transport()?)?,
        ),
    ];
    for (stack, provider) in providers {
        let request = ChatRequest::new("local-model").user("Say hello");
        let response = provider.chat(&request).await?;
        println!("{stack}, chat: {}", response.text().unwrap_or_default());

        let mut stream = provider.chat_stream(&request).await?;
        let mut text = String::new();
        while let Some(event) = stream.next().await {
            if let StreamEvent::TextDelta { text: delta, .. } = event? {
                text.push_str(&delta);
            }
        }
        println!("{stack}, chat_stream: {text}");
    }
    Ok(())
}

/// A stand-in OpenAI-compatible server, so the example runs without
/// credentials.
mod local {
    use std::convert::Infallible;

    use axum::Json;
    use axum::body::Body;
    use axum::response::{IntoResponse, Response};
    use axum::routing::post;
    use futures_util::StreamExt;
    use serde_json::{Value, json};

    /// How the server answers a streaming request.
    #[derive(Clone, Copy)]
    pub enum Answer {
        /// The whole stream.
        Whole,
        /// Its first event, then nothing, with the connection open: for
        /// the tests of the stream idle bound.
        #[cfg_attr(not(test), allow(dead_code))]
        Stalled,
    }

    const EVENTS: [&str; 4] = [
        r#"{"id":"c1","object":"chat.completion.chunk","model":"local-model","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"},"finish_reason":null}]}"#,
        r#"{"id":"c1","object":"chat.completion.chunk","model":"local-model","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}"#,
        r#"{"id":"c1","object":"chat.completion.chunk","model":"local-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        "[DONE]",
    ];

    /// Serve on a local port; the base URL to configure.
    pub async fn spawn(answer: Answer) -> std::io::Result<String> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let base_url = format!("http://{}/v1", listener.local_addr()?);
        let app = axum::Router::new().route(
            "/v1/chat/completions",
            post(move |Json(request): Json<Value>| completions(request, answer)),
        );
        tokio::spawn(async move { axum::serve(listener, app).await });
        Ok(base_url)
    }

    async fn completions(request: Value, answer: Answer) -> Response {
        if request["stream"] != true {
            return Json(json!({
                "id": "c1",
                "object": "chat.completion",
                "model": "local-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }))
            .into_response();
        }
        let events = match answer {
            Answer::Whole => &EVENTS[..],
            Answer::Stalled => &EVENTS[..1],
        };
        let events = futures_util::stream::iter(
            events
                .iter()
                .map(|data| Ok::<_, Infallible>(format!("data: {data}\n\n"))),
        );
        let body = match answer {
            Answer::Whole => Body::from_stream(events),
            Answer::Stalled => Body::from_stream(events.chain(futures_util::stream::pending())),
        };
        ([("content-type", "text/event-stream")], body).into_response()
    }
}

#[cfg(test)]
mod tests {
    use allama::Error;
    use allama_conformance::redirect::assert_redirect_not_followed;
    use allama_conformance::{TransportOutcome, assert_transport_options_applied};
    use allama_http::{
        Bytes, HttpRequest, ReplayResponse, ReplayTransport, StreamTimeouts, TransportError,
    };
    use http_body_util::BodyExt;

    use super::*;

    /// Both stacks, as a provider's `http_transport`.
    fn stacks() -> [(&'static str, Box<dyn HttpTransport>); 2] {
        [
            ("into_service", Box::new(transport())),
            (
                "tower-reqwest",
                Box::new(tower_reqwest_transport().unwrap()),
            ),
        ]
    }

    async fn streamed_text(provider: &Provider) -> Result<String> {
        let mut stream = provider
            .chat_stream(&ChatRequest::new("local-model").user("Hi"))
            .await?;
        let mut text = String::new();
        while let Some(event) = stream.next().await {
            if let StreamEvent::TextDelta { text: delta, .. } = event? {
                text.push_str(&delta);
            }
        }
        Ok(text)
    }

    #[tokio::test]
    async fn both_stacks_chat_and_stream() {
        let base_url = local::spawn(local::Answer::Whole).await.unwrap();
        for (stack, transport) in stacks() {
            let provider = provider(&base_url, transport).unwrap();
            let response = provider
                .chat(&ChatRequest::new("local-model").user("Hi"))
                .await
                .unwrap();
            assert_eq!(response.text().as_deref(), Some("Hello"), "{stack}");
            assert_eq!(streamed_text(&provider).await.unwrap(), "Hello", "{stack}");
        }
    }

    /// The provider's stream timeouts still bound a stream that stalls.
    #[tokio::test]
    async fn a_stalled_stream_hits_the_idle_bound() {
        let base_url = local::spawn(local::Answer::Stalled).await.unwrap();
        for (stack, transport) in stacks() {
            let provider = Provider::builder()
                .base_url(&base_url)
                .bearer_token("local-key")
                .http_transport(transport)
                .stream_timeouts(StreamTimeouts::default().idle(Some(Duration::from_millis(200))))
                .build()
                .unwrap();
            let err = streamed_text(&provider).await.unwrap_err();
            assert!(matches!(err, Error::Timeout { .. }), "{stack}: {err:?}");
        }
    }

    /// Neither stack follows a redirect: the key and the prompt stay put.
    #[tokio::test]
    async fn redirects_are_not_followed() {
        for stack in 0..2 {
            assert_redirect_not_followed(|base_url, stream| async move {
                let transport = stacks().into_iter().nth(stack).unwrap().1;
                let provider = provider(&format!("{base_url}/v1"), transport)?;
                let request = ChatRequest::new("local-model").user("Hi");
                match stream {
                    false => provider.chat(&request).await.map(drop),
                    true => provider.chat_stream(&request).await.map(drop),
                }
            })
            .await;
        }
    }

    /// `replay` as a `tower` service: what reaches it is what a provider
    /// sent through the stack.
    fn replayed(replay: ReplayTransport) -> impl HttpTransport {
        let service = tower::service_fn(move |request: http::Request<RequestBody>| {
            let replay = replay.clone();
            async move {
                let (parts, body) = request.into_parts();
                let Ok(body) = body.collect().await;
                let request = HttpRequest::from(http::Request::from_parts(parts, body.to_bytes()));
                let response = replay.send(request).await?;
                let mut answer = http::Response::new(RequestBody::default());
                *answer.status_mut() = response.status;
                *answer.headers_mut() = response.headers.clone();
                let body = response.bytes().await?;
                Ok::<_, TransportError>(answer.map(|_| RequestBody::new(body)))
            }
        });
        TowerTransport::new(ServiceBuilder::new().concurrency_limit(4).service(service))
    }

    /// The caller's headers and credentials reach the stack as the adapter
    /// placed them.
    #[tokio::test]
    async fn transport_options_reach_the_stack() {
        let slot = provider("http://upstream/v1", ReplayTransport::default())
            .unwrap()
            .credential_slot()
            .clone();
        for stream in [false, true] {
            assert_transport_options_applied(&slot, |options| async move {
                let replay = ReplayTransport::default();
                let mut request = ChatRequest::new("local-model").user("Hi");
                request.options = options;
                let provider = provider("http://upstream/v1", replayed(replay.clone())).unwrap();
                let result = match stream {
                    false => provider.chat(&request).await.map(drop),
                    true => provider.chat_stream(&request).await.map(drop),
                };
                TransportOutcome::new(result, replay.requests())
            })
            .await;
        }
        // And a response comes back through the stack as it was sent.
        let replay = ReplayTransport::new([ReplayResponse::json(
            http::StatusCode::OK,
            Bytes::from_static(
                br#"{"id":"c1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}]}"#,
            ),
        )]);
        let response = provider("http://upstream/v1", replayed(replay))
            .unwrap()
            .chat(&ChatRequest::new("m").user("Hi"))
            .await
            .unwrap();
        assert_eq!(response.text().as_deref(), Some("Hi"));
    }
}
