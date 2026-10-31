//! The bundled transports tell the call scope which server they sent to,
//! so a `TracingProvider` span records `server.address` and `server.port`.

#![cfg(all(
    any(feature = "reqwest", feature = "tower"),
    not(target_arch = "wasm32")
))]

use std::fmt::Write as _;
use std::sync::{Arc, Mutex};

use allama::mock::MockProvider;
use allama::wrappers::TracingProvider;
use allama::{ChatProvider, ChatRequest, ChatResponse};
use allama_http::{HttpRequest, HttpTransport};
use http::Uri;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};

/// Records every field of every span and event as text.
#[derive(Clone, Default)]
struct Capture(Arc<Mutex<String>>);

impl Visit for Capture {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        let _ = writeln!(self.0.lock().unwrap(), "{}={value:?}", field.name());
    }
}

impl<S: Subscriber> Layer<S> for Capture {
    fn on_new_span(&self, attrs: &Attributes<'_>, _: &Id, _: Context<'_, S>) {
        attrs.record(&mut self.clone());
    }

    fn on_record(&self, _: &Id, values: &Record<'_>, _: Context<'_, S>) {
        values.record(&mut self.clone());
    }

    fn on_event(&self, event: &Event<'_>, _: Context<'_, S>) {
        event.record(&mut self.clone());
    }
}

/// Sends one request through the transport, then answers from the mock.
struct Sends {
    inner: MockProvider,
    transport: Box<dyn HttpTransport>,
    url: String,
}

impl ChatProvider for Sends {
    type Stream = <MockProvider as ChatProvider>::Stream;

    async fn chat(&self, request: &ChatRequest) -> allama::Result<ChatResponse> {
        let response = self
            .transport
            .send(HttpRequest::get(Uri::try_from(self.url.clone()).unwrap()))
            .await
            .map_err(allama_http::TransportError::into_error)?;
        assert_eq!(response.status, 200);
        self.inner.chat(request).await
    }

    async fn chat_stream(&self, request: &ChatRequest) -> allama::Result<Self::Stream> {
        self.inner.chat_stream(request).await
    }

    allama::wrappers::forward_chat_questions!(inner);
}

impl allama::ProviderIdentity for Sends {
    allama::wrappers::forward_provider_identity!(inner);
}

/// The span fields recorded for one call that sends to `url` through
/// `transport`.
async fn recorded_fields(transport: impl HttpTransport, url: String) -> String {
    let provider = TracingProvider::new(Sends {
        inner: MockProvider::with_text("ok"),
        transport: Box::new(transport),
        url,
    });
    let capture = Capture::default();
    let _guard =
        tracing::subscriber::set_default(tracing_subscriber::registry().with(capture.clone()));

    provider
        .chat(&ChatRequest::new("m").user("hi"))
        .await
        .unwrap();

    capture.0.lock().unwrap().clone()
}

#[cfg(feature = "reqwest")]
#[tokio::test]
async fn the_transport_records_the_server_it_sent_to() {
    use allama_conformance::{MockHttpResponse, TestHttpServer};
    use http::StatusCode;

    let server = TestHttpServer::spawn([MockHttpResponse::text(StatusCode::OK, "ok")]).await;
    let port = server.url().rsplit(':').next().unwrap().to_owned();
    let fields = recorded_fields(
        allama_http::ReqwestTransport::new(),
        format!("{}/v1/models", server.url()),
    )
    .await;
    assert!(fields.contains("server.address=\"127.0.0.1\""), "{fields}");
    assert!(fields.contains(&format!("server.port={port}")), "{fields}");
}

/// `TowerTransport` records the host lowercase, and the scheme's default
/// port when the URI names none, as the reqwest transport does.
#[cfg(feature = "tower")]
#[tokio::test]
async fn the_tower_transport_records_the_server_it_sent_to() {
    use allama_http::{Bytes, RequestBody, TowerTransport};

    let ok = || {
        TowerTransport::new(tower::service_fn(|_: http::Request<RequestBody>| async {
            Ok::<_, std::convert::Infallible>(http::Response::new(RequestBody::new(Bytes::new())))
        }))
    };
    for (url, address, port) in [
        ("https://API.Example.com/v1/models", "api.example.com", 443),
        ("http://api.example.com/v1/models", "api.example.com", 80),
        ("http://[::1]:8080/v1/models", "[::1]", 8080),
    ] {
        let fields = recorded_fields(ok(), url.to_owned()).await;
        assert!(
            fields.contains(&format!("server.address=\"{address}\"")),
            "{url}: {fields}"
        );
        assert!(
            fields.contains(&format!("server.port={port}")),
            "{url}: {fields}"
        );
    }
}
