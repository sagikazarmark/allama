//! Transport options: the caller's per-request headers reach the wire on
//! every call path, and the credentials addressed to the provider's
//! credential scope, and no other, are sent in `x-api-key`.

use allama::RequestOptions;
use allama::prelude::*;
use allama::wrappers::{ChatExchange, RecordingProvider};
use allama_anthropic::{DEFAULT_CREDENTIAL_SCOPE, Provider};
use allama_conformance::{TransportOutcome, assert_transport_options_applied};
use allama_http::{
    Credential, CredentialPlacement, CredentialSlot, HttpRequestOptions, ReplayTransport,
};

fn provider(transport: &ReplayTransport) -> Provider {
    Provider::builder()
        .api_key("provider-key")
        .http_transport(transport.clone())
        .build()
        .unwrap()
}

/// The slot every Anthropic provider is expected to declare.
fn slot(scope: &'static str) -> CredentialSlot {
    CredentialSlot::new(scope, CredentialPlacement::Header("x-api-key".into()))
}

fn sent_values(transport: &ReplayTransport, name: &str) -> Vec<String> {
    let requests = transport.requests();
    let request = requests.last().expect("the adapter sent a request");
    request
        .headers
        .iter()
        .filter(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.to_owned())
        .collect()
}

fn request_with(options: RequestOptions) -> ChatRequest {
    let mut request = ChatRequest::new("claude-haiku-4-5").user("Hi");
    request.options = options;
    request
}

fn with_transport(options: HttpRequestOptions) -> ChatRequest {
    ChatRequest::new("claude-haiku-4-5")
        .user("Hi")
        .with_option(options)
}

#[tokio::test]
async fn a_byok_provider_applies_transport_options() {
    let byok = slot("tenant-anthropic").required(true);
    assert_transport_options_applied(&byok, |options| async {
        let transport = ReplayTransport::default();
        let provider = Provider::builder()
            .credential_scope("tenant-anthropic")
            .http_transport(transport.clone())
            .build()
            .unwrap();
        assert_eq!(
            provider.credential_slot(),
            &slot("tenant-anthropic").required(true)
        );
        let result = provider.chat(&request_with(options)).await;
        TransportOutcome::new(result, transport.requests())
    })
    .await;
}

#[test]
fn the_slot_defaults_to_the_provider_name() {
    let provider = provider(&ReplayTransport::default());
    assert_eq!(DEFAULT_CREDENTIAL_SCOPE, "anthropic");
    assert_eq!(provider.credential_slot(), &slot("anthropic"));
    let scoped = provider.with_credential_scope("second-account");
    assert_eq!(scoped.credential_slot(), &slot("second-account"));
}

#[tokio::test]
async fn a_tenant_key_replaces_the_provider_key() {
    let transport = ReplayTransport::default();
    let request = with_transport(
        HttpRequestOptions::new()
            .credential("anthropic", Credential::api_key("TENANT-SECRET"))
            .credential("openai", Credential::api_key("FOREIGN-SECRET"))
            .header("Idempotency-Key", "idem-1"),
    );

    let exchanges = std::sync::Arc::new(std::sync::Mutex::new(Vec::<ChatExchange>::new()));
    let sink = exchanges.clone();
    let recording = RecordingProvider::new(provider(&transport), move |exchange| {
        sink.lock().unwrap().push(exchange);
    });
    let _ = recording.chat(&request).await;

    assert_eq!(sent_values(&transport, "x-api-key"), ["TENANT-SECRET"]);
    assert_eq!(sent_values(&transport, "idempotency-key"), ["idem-1"]);
    let sent = format!("{:?}", transport.requests());
    assert!(!sent.contains("SECRET"), "{sent}");
    let request = &transport.requests()[0];
    assert!(
        !request
            .headers
            .iter()
            .any(|(_, value)| value.contains("FOREIGN")),
        "a credential for another scope was sent"
    );
    let exchanges = exchanges.lock().unwrap();
    let debug = format!("{exchanges:?}");
    assert!(!debug.contains("SECRET"), "{debug}");
    let json = serde_json::to_string(&*exchanges).unwrap();
    assert!(!json.contains("SECRET"), "{json}");
}

#[tokio::test]
async fn a_second_scope_sends_its_own_key_and_keeps_the_default_scope_out() {
    let options = HttpRequestOptions::new()
        .credential("anthropic", Credential::api_key("first-account"))
        .credential("anthropic-b", Credential::api_key("second-account"));
    let transport = ReplayTransport::default();
    let second = provider(&transport).with_credential_scope("anthropic-b");
    let _ = second.chat(&with_transport(options.clone())).await;
    assert_eq!(sent_values(&transport, "x-api-key"), ["second-account"]);

    // A scope with no credential falls back to the configured key.
    let transport = ReplayTransport::default();
    let _ = provider(&transport)
        .with_credential_scope("anthropic-c")
        .chat(&with_transport(options))
        .await;
    assert_eq!(sent_values(&transport, "x-api-key"), ["provider-key"]);
}

#[tokio::test]
async fn a_gateway_header_credential_reaches_only_its_scope() {
    let options = HttpRequestOptions::new()
        .credential(
            "anthropic",
            Credential::header("cf-aig-authorization", "Bearer GATEWAY-SECRET"),
        )
        .credential("openai", Credential::header("x-other-gateway", "OTHER"));
    let transport = ReplayTransport::default();
    let _ = provider(&transport).chat(&with_transport(options)).await;
    assert_eq!(
        sent_values(&transport, "cf-aig-authorization"),
        ["Bearer GATEWAY-SECRET"]
    );
    assert_eq!(sent_values(&transport, "x-api-key"), ["provider-key"]);
    assert!(sent_values(&transport, "x-other-gateway").is_empty());
}

#[tokio::test]
async fn a_plain_credential_header_is_refused_before_sending() {
    for name in ["x-api-key", "X-Api-Key", "authorization"] {
        let transport = ReplayTransport::default();
        let err = provider(&transport)
            .chat(&with_transport(
                HttpRequestOptions::new().header(name, "PLAIN-SECRET"),
            ))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
        assert_eq!(
            err.context().and_then(|c| c.provider.as_deref()),
            Some("anthropic")
        );
        assert!(!format!("{err} {err:?}").contains("PLAIN-SECRET"));
        assert!(transport.requests().is_empty(), "{name} was sent");
    }
}

#[tokio::test]
async fn a_byok_provider_needs_a_credential_for_its_scope() {
    assert!(
        Provider::builder().build().is_err(),
        "a key or a scope is required"
    );
    let transport = ReplayTransport::default();
    let provider = Provider::builder()
        .credential_scope("tenant")
        .http_transport(transport.clone())
        .build()
        .unwrap();

    let err = provider
        .chat(&with_transport(
            HttpRequestOptions::new().credential("anthropic", Credential::api_key("wrong-scope")),
        ))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    assert!(transport.requests().is_empty());

    let _ = provider
        .chat_stream(&with_transport(
            HttpRequestOptions::new().credential("tenant", Credential::api_key("TENANT-KEY")),
        ))
        .await;
    assert_eq!(sent_values(&transport, "x-api-key"), ["TENANT-KEY"]);
}
