//! The reqwest transport tells the call scope which server it sent to, so
//! a `TracingProvider` span records `server.address` and `server.port`.

#![cfg(all(feature = "reqwest", not(target_arch = "wasm32")))]

use std::fmt::Write as _;
use std::sync::{Arc, Mutex};

use allama::mock::MockProvider;
use allama::wrappers::TracingProvider;
use allama::{ChatProvider, ChatRequest, ChatResponse};
use allama_conformance::{MockHttpResponse, TestHttpServer};
use allama_http::{HttpRequest, HttpTransport, ReqwestTransport};
use http::{StatusCode, Uri};
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
    url: String,
}

impl ChatProvider for Sends {
    type Stream = <MockProvider as ChatProvider>::Stream;

    async fn chat(&self, request: &ChatRequest) -> allama::Result<ChatResponse> {
        let response = ReqwestTransport::new()
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

#[tokio::test]
async fn the_transport_records_the_server_it_sent_to() {
    let server = TestHttpServer::spawn([MockHttpResponse::text(StatusCode::OK, "ok")]).await;
    let port = server.url().rsplit(':').next().unwrap().to_owned();
    let provider = TracingProvider::new(Sends {
        inner: MockProvider::with_text("ok"),
        url: format!("{}/v1/models", server.url()),
    });
    let capture = Capture::default();
    let _guard =
        tracing::subscriber::set_default(tracing_subscriber::registry().with(capture.clone()));

    provider
        .chat(&ChatRequest::new("m").user("hi"))
        .await
        .unwrap();

    let fields = capture.0.lock().unwrap().clone();
    assert!(fields.contains("server.address=\"127.0.0.1\""), "{fields}");
    assert!(fields.contains(&format!("server.port={port}")), "{fields}");
}
