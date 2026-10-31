//! # Calling an HTTP provider from a Cloudflare Worker
//!
//! A Worker that streams a chat completion from an OpenAI-compatible API,
//! over the Workers `fetch` API: `tower-worker-fetch`'s `FetchService`,
//! made a transport by `allama-http`'s [`TowerTransport`] (its `tower`
//! feature). The provider crate is built without `reqwest`.
//!
//! One thing has to be set for every request: the redirect mode. A request
//! built in Rust has `fetch`'s default, `follow`, and the Workers runtime
//! sends every header to the redirect's target, the provider's key
//! included. `transport` sets `Redirect::Manual`, so a `3xx` comes back to
//! the provider, which reports it.
//!
//! The rest comes with `TowerTransport`: the provider's request timeout
//! bounds every non-streaming call, and dropping the call cancels the fetch;
//! a streaming call keeps the provider's stream timeouts. `FetchService` is
//! `Send`, so no `SendWrapper` is needed.
//!
//! `POST /` streams the reply to the prompt in the request body (`GET /`
//! asks to say hello). The upstream is configured in `wrangler.toml` and
//! `.dev.vars`: `BASE_URL`, `MODEL` and the secret `API_KEY`. With no
//! `BASE_URL`, it is a stand-in OpenAI-compatible API this Worker serves
//! itself, so it runs without credentials. `devenv up` serves it under
//! `wrangler dev` (`worker-build` builds it):
//!
//! ```text
//! devenv up
//! curl http://127.0.0.1:8787/ -d "Say hello"
//! ```
//!
//! The `/test-*` routes check the transport against the stand-in (`devenv
//! test` serves the Worker and runs them with `just worker-smoke`): unary
//! and streaming chat, a service binding, the request timeout aborting the
//! fetch, the stream idle bound, and redirects not followed. The
//! applications guide quotes `transport` (`docs/guides/applications.md`,
//! "From a Cloudflare Worker").

use std::convert::Infallible;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use allama::prelude::*;
use allama_http::http::{self, Response, StatusCode};
use allama_http::{HttpTransport, RequestBody, StreamTimeouts, TowerTransport};
use allama_openai_compat::Provider;
use futures_util::{StreamExt, TryStreamExt};
use tower::ServiceBuilder;
use tower_worker_fetch::{FetchService, Redirect};
use worker::{Context, Date, Delay, Env, HttpRequest, event};

/// The Workers `fetch` API as a transport. Every request goes out with
/// `redirect: "manual"`: `fetch` follows redirects by default, and sends the
/// provider's key to the redirect's host.
fn transport(fetch: FetchService) -> impl HttpTransport {
    let stack = ServiceBuilder::new()
        .map_request(|mut request: http::Request<RequestBody>| {
            request.extensions_mut().insert(Redirect::Manual);
            request
        })
        .service(fetch);
    TowerTransport::new(stack)
}

/// The upstream API, from the Worker's environment.
struct Upstream {
    base_url: String,
    api_key: Option<String>,
    model: String,
}

impl Upstream {
    /// `BASE_URL`, `API_KEY` and `MODEL`; without `BASE_URL`, the stand-in
    /// served at `origin`.
    fn from_env(env: &Env, origin: &str) -> Self {
        let var = |name| {
            env.var(name)
                .map(|value| value.to_string())
                .ok()
                .filter(|value| !value.is_empty())
        };
        let api_key = env
            .secret("API_KEY")
            .map(|value| value.to_string())
            .ok()
            .filter(|value| !value.is_empty());
        match var("BASE_URL") {
            Some(base_url) => Self {
                base_url,
                api_key,
                model: var("MODEL").unwrap_or_default(),
            },
            None => stand_in::upstream(origin),
        }
    }

    /// An OpenAI-compatible provider for this upstream, over `fetch`.
    fn provider(&self, fetch: FetchService) -> Result<Provider> {
        let mut builder = Provider::builder()
            .base_url(&self.base_url)
            .http_transport(transport(fetch));
        if let Some(api_key) = &self.api_key {
            builder = builder.bearer_token(api_key);
        }
        builder.build()
    }
}

#[event(fetch)]
async fn fetch(
    request: HttpRequest,
    env: Env,
    _ctx: Context,
) -> worker::Result<Response<worker::Body>> {
    let origin = origin(&request)?;
    let path = request.uri().path().to_owned();
    match path.as_str() {
        "/" => chat(request, &env, &origin).await,

        "/test-chat" => smoke::report(smoke::chat(&origin).await),
        "/test-stream" => smoke::report(smoke::stream(&origin).await),
        "/test-binding" => smoke::report(smoke::binding(&origin, &env).await),
        "/test-timeout" => smoke::report(smoke::timeout(&origin, &env).await),
        "/test-stream-idle" => smoke::report(smoke::stream_idle(&origin).await),
        "/test-redirect" => smoke::report(smoke::redirect(&origin).await),

        path => match stand_in::Route::of(path) {
            Some(route) => stand_in::serve(route, request).await,
            None => text(StatusCode::NOT_FOUND, "not found"),
        },
    }
}

/// Stream the reply to the prompt in the request body as plain text.
async fn chat(
    request: HttpRequest,
    env: &Env,
    origin: &str,
) -> worker::Result<Response<worker::Body>> {
    let upstream = Upstream::from_env(env, origin);
    let prompt = String::from_utf8_lossy(&read_body(request).await?).into_owned();
    let prompt = if prompt.trim().is_empty() {
        "Say hello".to_owned()
    } else {
        prompt
    };

    let provider = match upstream.provider(FetchService::new()) {
        Ok(provider) => provider,
        Err(err) => return text(StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
    };
    let request = ChatRequest::new(&upstream.model).user(prompt);
    // Failures before the first event still have a status code; after it,
    // the response is cut short.
    let stream = match provider.chat_stream(&request).await {
        Ok(stream) => stream,
        Err(err) => return text(StatusCode::BAD_GATEWAY, err.to_string()),
    };
    let text = stream
        .try_filter_map(|event| async move {
            Ok(match event {
                StreamEvent::TextDelta { text, .. } => Some(text.into_bytes()),
                _ => None,
            })
        })
        .inspect_err(|err| worker::console_error!("the stream failed: {err}"));
    Ok(Response::builder()
        .header("content-type", "text/plain; charset=utf-8")
        .body(worker::Body::from_stream(text)?)?)
}

/// A stand-in OpenAI-compatible API, served by this Worker, so the example
/// and its checks run without credentials.
mod stand_in {
    use super::*;

    /// The key the checks send to the stand-in.
    pub const API_KEY: &str = "stand-in-key";

    /// How long `/slow` waits before it answers.
    pub const SLOW: Duration = Duration::from_millis(500);

    /// Bumped by `/slow` once it has waited and answered. The Worker reaches
    /// itself through a service binding in the same isolate under `wrangler
    /// dev`, so a check reads what the handler it called managed to do;
    /// cancelling the call cancels the handler.
    pub static SLOW_ANSWERED: AtomicU32 = AtomicU32::new(0);

    /// Bumped by `/leak` for each request that arrives with the key: a
    /// followed redirect.
    pub static KEYS_LEAKED: AtomicU32 = AtomicU32::new(0);

    /// The stand-in at `origin`.
    pub fn upstream(origin: &str) -> Upstream {
        at(origin, Route::Answer)
    }

    /// The stand-in at `origin`, answering as `route` does.
    pub fn at(origin: &str, route: Route) -> Upstream {
        Upstream {
            base_url: format!("{origin}/{}/v1", route.prefix()),
            api_key: Some(API_KEY.to_owned()),
            model: "stand-in-model".to_owned(),
        }
    }

    /// The stand-in's routes, each a base URL of its own.
    #[derive(Clone, Copy)]
    pub enum Route {
        /// Answers.
        Answer,
        /// Answers after [`SLOW`].
        Slow,
        /// Streams its first event, then pauses for ten seconds before each
        /// of the rest.
        Stalled,
        /// Redirects to `/leak`.
        Redirect,
        /// Answers, and counts a key that arrives in [`KEYS_LEAKED`].
        Leak,
    }

    impl Route {
        const ALL: [Self; 5] = [
            Self::Answer,
            Self::Slow,
            Self::Stalled,
            Self::Redirect,
            Self::Leak,
        ];

        fn prefix(self) -> &'static str {
            match self {
                Self::Answer => "stand-in",
                Self::Slow => "slow",
                Self::Stalled => "stalled",
                Self::Redirect => "redirect",
                Self::Leak => "leak",
            }
        }

        /// The route serving `path`, if any.
        pub fn of(path: &str) -> Option<Self> {
            Self::ALL
                .into_iter()
                .find(|route| path == format!("/{}/v1/chat/completions", route.prefix()))
        }
    }

    const EVENTS: [&str; 4] = [
        r#"{"id":"c1","object":"chat.completion.chunk","model":"stand-in-model","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"},"finish_reason":null}]}"#,
        r#"{"id":"c1","object":"chat.completion.chunk","model":"stand-in-model","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}"#,
        r#"{"id":"c1","object":"chat.completion.chunk","model":"stand-in-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        "[DONE]",
    ];

    pub async fn serve(
        route: Route,
        request: HttpRequest,
    ) -> worker::Result<Response<worker::Body>> {
        match route {
            Route::Redirect => {
                return Ok(Response::builder()
                    .status(StatusCode::SEE_OTHER)
                    .header(
                        "location",
                        format!("/{}/v1/chat/completions", Route::Leak.prefix()),
                    )
                    .body(worker::Body::empty())?);
            }
            Route::Leak if request.headers().contains_key("authorization") => {
                KEYS_LEAKED.fetch_add(1, Ordering::Relaxed);
            }
            Route::Slow => {
                Delay::from(SLOW).await;
                SLOW_ANSWERED.fetch_add(1, Ordering::Relaxed);
            }
            Route::Answer | Route::Stalled | Route::Leak => {}
        }

        // A followed `303` arrives as a `GET`, without the request body.
        let request: serde_json::Value =
            serde_json::from_slice(&read_body(request).await?).unwrap_or_default();
        if request["stream"] != true {
            return Ok(Response::builder()
                .header("content-type", "application/json")
                .body(body(serde_json::json!({
                    "id": "c1",
                    "object": "chat.completion",
                    "model": "stand-in-model",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                }))?)?);
        }

        // A stalled stream waits on a timer, not on nothing: the runtime
        // cancels a response that waits on nothing at all.
        let pause = move |index| match (route, index) {
            (Route::Stalled, 1..) => Duration::from_secs(10),
            _ => Duration::from_millis(10),
        };
        let events = futures_util::stream::iter(EVENTS.iter().enumerate()).then(
            move |(index, data)| async move {
                Delay::from(pause(index)).await;
                Ok::<_, Infallible>(format!("data: {data}\n\n").into_bytes())
            },
        );
        Ok(Response::builder()
            .header("content-type", "text/event-stream")
            .body(worker::Body::from_stream(events)?)?)
    }
}

/// The checks `just worker-smoke` runs: each `/test-*` route answers `200
/// {"ok":true}`, or `500 {"ok":false,"error":...}`.
mod smoke {
    use super::stand_in::{KEYS_LEAKED, Route, SLOW, SLOW_ANSWERED};
    use super::*;

    type BoxError = Box<dyn std::error::Error + Send + Sync>;
    type Outcome = std::result::Result<serde_json::Value, BoxError>;

    fn request() -> ChatRequest {
        ChatRequest::new("stand-in-model").user("Say hello")
    }

    /// The text of a streamed reply, and how many deltas it came in.
    async fn streamed(provider: &Provider) -> Result<(String, usize)> {
        let mut stream = provider.chat_stream(&request()).await?;
        let (mut text, mut deltas) = (String::new(), 0);
        while let Some(event) = stream.next().await {
            if let StreamEvent::TextDelta { text: delta, .. } = event? {
                text.push_str(&delta);
                deltas += 1;
            }
        }
        Ok((text, deltas))
    }

    pub async fn chat(origin: &str) -> Outcome {
        let provider = stand_in::upstream(origin).provider(FetchService::new())?;
        let response = provider.chat(&request()).await?;
        ensure(
            response.text().as_deref() == Some("Hello"),
            "the reply is the stand-in's",
        )?;
        Ok(serde_json::json!({ "ok": true }))
    }

    pub async fn stream(origin: &str) -> Outcome {
        let provider = stand_in::upstream(origin).provider(FetchService::new())?;
        let (text, deltas) = streamed(&provider).await?;
        ensure(text == "Hello", "the streamed reply is the stand-in's")?;
        ensure(deltas == 2, "the reply arrived in two deltas")?;
        Ok(serde_json::json!({ "ok": true, "deltas": deltas }))
    }

    /// Through a service binding (`SELF`, this Worker) instead of global
    /// `fetch`.
    pub async fn binding(origin: &str, env: &Env) -> Outcome {
        let provider =
            stand_in::upstream(origin).provider(FetchService::from(env.service("SELF")?))?;
        let response = provider.chat(&request()).await?;
        ensure(
            response.text().as_deref() == Some("Hello"),
            "the reply is the stand-in's",
        )?;
        Ok(serde_json::json!({ "ok": true }))
    }

    /// The provider's request timeout fails the call when it expires, and
    /// aborts the fetch: the upstream handler, reached through a service
    /// binding, is cancelled before it answers.
    pub async fn timeout(origin: &str, env: &Env) -> Outcome {
        let slow = |timeout| {
            let upstream = stand_in::at(origin, Route::Slow);
            Provider::builder()
                .base_url(&upstream.base_url)
                .http_transport(transport(FetchService::from(env.service("SELF")?)))
                .timeout(Some(timeout))
                .build()
                .map_err(BoxError::from)
        };

        // Control: with time to spare, the handler answers.
        let before = SLOW_ANSWERED.load(Ordering::Relaxed);
        slow(SLOW * 4)?.chat(&request()).await?;
        ensure(
            SLOW_ANSWERED.load(Ordering::Relaxed) == before + 1,
            "the slow handler answers a call that waits for it",
        )?;

        let before = SLOW_ANSWERED.load(Ordering::Relaxed);
        let started = Date::now().as_millis();
        let err = match slow(Duration::from_millis(100))?.chat(&request()).await {
            Ok(_) => return Err("the call succeeded past its timeout".into()),
            Err(err) => err,
        };
        let elapsed = Date::now().as_millis() - started;
        ensure(
            matches!(err, Error::Timeout { .. }),
            &format!("a timeout error, not {err}"),
        )?;
        ensure(
            elapsed < SLOW.as_millis() as u64,
            &format!("the call failed before the upstream answered (after {elapsed} ms)"),
        )?;

        Delay::from(SLOW * 2).await;
        ensure(
            SLOW_ANSWERED.load(Ordering::Relaxed) == before,
            "the fetch was aborted: the slow handler never answered",
        )?;
        Ok(serde_json::json!({ "ok": true, "elapsed_ms": elapsed }))
    }

    /// The provider's stream idle bound ends a stream that stalls.
    pub async fn stream_idle(origin: &str) -> Outcome {
        let upstream = stand_in::at(origin, Route::Stalled);
        let provider = Provider::builder()
            .base_url(&upstream.base_url)
            .http_transport(transport(FetchService::new()))
            .stream_timeouts(StreamTimeouts::default().idle(Some(Duration::from_millis(200))))
            .build()?;
        match streamed(&provider).await {
            Ok(_) => Err("the stalled stream ended".into()),
            Err(err) => {
                ensure(
                    matches!(err, Error::Timeout { .. }),
                    &format!("a timeout error, not {err}"),
                )?;
                Ok(serde_json::json!({ "ok": true }))
            }
        }
    }

    /// The transport does not follow a redirect, unary or streaming: the
    /// provider reports it, and the key stays put. A bare `FetchService`
    /// follows it, and the key reaches the redirect's target.
    pub async fn redirect(origin: &str) -> Outcome {
        let upstream = stand_in::at(origin, Route::Redirect);
        let before = KEYS_LEAKED.load(Ordering::Relaxed);
        let provider = upstream.provider(FetchService::new())?;
        for stream in [false, true] {
            let result = match stream {
                false => provider.chat(&request()).await.map(drop),
                true => provider.chat_stream(&request()).await.map(drop),
            };
            let Err(err) = result else {
                return Err(format!("the redirect was followed (stream: {stream})").into());
            };
            ensure(
                matches!(err, Error::UnexpectedResponse { .. })
                    && err.to_string().contains("redirect"),
                &format!("a redirect error, not {err} (stream: {stream})"),
            )?;
        }
        ensure(
            KEYS_LEAKED.load(Ordering::Relaxed) == before,
            "the key did not reach the redirect's target",
        )?;

        // Control: without `Redirect::Manual`, `fetch` follows.
        let bare = Provider::builder()
            .base_url(&upstream.base_url)
            .bearer_token(stand_in::API_KEY)
            .http_transport(TowerTransport::new(FetchService::new()))
            .build()?;
        bare.chat(&request()).await?;
        ensure(
            KEYS_LEAKED.load(Ordering::Relaxed) == before + 1,
            "a bare FetchService follows the redirect, key and all",
        )?;
        Ok(serde_json::json!({ "ok": true }))
    }

    fn ensure(condition: bool, what: &str) -> std::result::Result<(), BoxError> {
        match condition {
            true => Ok(()),
            false => Err(format!("check failed: {what}").into()),
        }
    }

    pub fn report(outcome: Outcome) -> worker::Result<Response<worker::Body>> {
        let (status, value) = match outcome {
            Ok(value) => (StatusCode::OK, value),
            Err(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                serde_json::json!({ "ok": false, "error": err.to_string() }),
            ),
        };
        Ok(Response::builder()
            .status(status)
            .header("content-type", "application/json")
            .body(body(value)?)?)
    }
}

/// `scheme://authority` of the incoming request: the stand-in's origin.
fn origin(request: &HttpRequest) -> worker::Result<String> {
    let uri = request.uri();
    match (uri.scheme_str(), uri.authority()) {
        (Some(scheme), Some(authority)) => Ok(format!("{scheme}://{authority}")),
        _ => Err(worker::Error::RustError(format!(
            "the request URI has no origin: {uri}"
        ))),
    }
}

async fn read_body(request: HttpRequest) -> worker::Result<Vec<u8>> {
    request
        .into_body()
        .try_fold(Vec::new(), |mut body, chunk| async move {
            body.extend_from_slice(&chunk);
            Ok(body)
        })
        .await
}

/// A response body of `content`.
fn body(content: impl ToString) -> worker::Result<worker::Body> {
    let content = content.to_string().into_bytes();
    worker::Body::from_stream(futures_util::stream::once(async move {
        Ok::<_, Infallible>(content)
    }))
}

fn text(status: StatusCode, content: impl ToString) -> worker::Result<Response<worker::Body>> {
    Ok(Response::builder()
        .status(status)
        .header("content-type", "text/plain; charset=utf-8")
        .body(body(content)?)?)
}
