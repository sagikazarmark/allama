//! `http-tracing` spans never record the values of the caller's
//! per-request headers or credentials.

#![cfg(all(feature = "http-tracing", not(target_arch = "wasm32")))]

use std::fmt::Write as _;
use std::sync::{Arc, Mutex};

use allama::ChatRequest;
use allama_conformance::{MockHttpResponse, TestHttpServer};
use allama_http::{
    Credential, CredentialPlacement, CredentialSlot, HttpRequest, HttpRequestOptions,
    HttpTransport, ReqwestTransport,
};
use http::{HeaderValue, StatusCode, Uri};
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

#[tokio::test]
async fn spans_never_record_header_values() {
    let server = TestHttpServer::spawn([MockHttpResponse::text(StatusCode::OK, "ok")]).await;
    let capture = Capture::default();
    let _guard =
        tracing::subscriber::set_default(tracing_subscriber::registry().with(capture.clone()));

    let request = ChatRequest::new("m").with_option(
        HttpRequestOptions::new()
            .credential("gateway", Credential::api_key("TENANT-SECRET"))
            .credential("gateway", Credential::header("x-route-key", "ROUTE-SECRET"))
            .header("x-gateway-tag", HeaderValue::from_static("TAG-VALUE")),
    );
    let slot = CredentialSlot::new("gateway", CredentialPlacement::Bearer);
    let http = HttpRequest::post(
        Uri::try_from(format!("{}/v1/chat", server.url())).unwrap(),
        "{}",
    )
    .header(
        "authorization",
        HeaderValue::from_static("Bearer provider-key"),
    )
    .apply_options(&request.options, &slot)
    .unwrap();
    let response = ReqwestTransport::new().send(http).await.unwrap();
    assert_eq!(response.status, 200);

    let recorded = server.recorded_requests().await;
    assert_eq!(
        recorded[0].header("authorization"),
        Some("Bearer TENANT-SECRET")
    );
    assert_eq!(recorded[0].header("x-route-key"), Some("ROUTE-SECRET"));

    let fields = capture.0.lock().unwrap().clone();
    assert!(
        fields.contains("/v1/chat"),
        "no span was recorded: {fields}"
    );
    assert!(!fields.contains("TENANT-SECRET"), "{fields}");
    assert!(!fields.contains("ROUTE-SECRET"), "{fields}");
    assert!(!fields.contains("TAG-VALUE"), "{fields}");
}
