//! Transport options: the caller's per-request headers reach the wire on
//! every call path, and the credentials addressed to the provider's
//! credential scope, and no other, are sent in `x-goog-api-key`.

use allama::RequestOptions;
use allama::prelude::*;
use allama_conformance::{TransportOutcome, assert_transport_options_applied};
use allama_gemini::{DEFAULT_CREDENTIAL_SCOPE, Provider};
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

/// The slot every Gemini provider is expected to declare.
fn slot(scope: &'static str) -> CredentialSlot {
    CredentialSlot::new(scope, CredentialPlacement::Header("x-goog-api-key".into()))
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

fn chat_request(options: RequestOptions) -> ChatRequest {
    let mut request = ChatRequest::new("gemini-2.5-flash").user("Hi");
    request.options = options;
    request
}

fn embedding_request(options: RequestOptions) -> EmbeddingRequest {
    let mut request = EmbeddingRequest::new("gemini-embedding-001").inputs(["hello"]);
    request.options = options;
    request
}

fn options(transport: HttpRequestOptions) -> RequestOptions {
    let mut options = RequestOptions::new();
    options.insert(transport);
    options
}

#[tokio::test]
async fn a_byok_provider_applies_transport_options_on_chat_and_embed() {
    let byok = slot("tenant-gemini").required(true);
    let build = |transport: &ReplayTransport| {
        Provider::builder()
            .credential_scope("tenant-gemini")
            .http_transport(transport.clone())
            .build()
            .unwrap()
    };
    assert_transport_options_applied(&byok, |options| async {
        let transport = ReplayTransport::default();
        let provider = build(&transport);
        assert_eq!(
            provider.credential_slot(),
            &slot("tenant-gemini").required(true)
        );
        let result = provider.chat(&chat_request(options)).await;
        TransportOutcome::new(result, transport.requests())
    })
    .await;
    assert_transport_options_applied(&byok, |options| async {
        let transport = ReplayTransport::default();
        let result = build(&transport).embed(&embedding_request(options)).await;
        TransportOutcome::new(result, transport.requests())
    })
    .await;
}

#[test]
fn the_slot_defaults_to_the_provider_name_not_the_namespace() {
    let provider = provider(&ReplayTransport::default());
    assert_eq!(DEFAULT_CREDENTIAL_SCOPE, allama_gemini::PROVIDER_NAME);
    assert_ne!(DEFAULT_CREDENTIAL_SCOPE, allama_gemini::NAMESPACE);
    assert_eq!(provider.credential_slot(), &slot("gemini"));
    let scoped = provider.with_credential_scope("second-account");
    assert_eq!(scoped.credential_slot(), &slot("second-account"));
}

#[tokio::test]
async fn a_tenant_key_replaces_the_provider_key_on_chat_and_embed() {
    let tenant = || {
        options(
            HttpRequestOptions::new()
                .credential("gemini", Credential::api_key("TENANT-SECRET"))
                .credential("google-ai", Credential::api_key("NAMESPACE-SECRET"))
                .credential("openai", Credential::api_key("FOREIGN-SECRET"))
                .header("Idempotency-Key", "idem-1"),
        )
    };
    let transport = ReplayTransport::default();
    let _ = provider(&transport).chat(&chat_request(tenant())).await;
    assert_eq!(sent_values(&transport, "x-goog-api-key"), ["TENANT-SECRET"]);
    assert_eq!(sent_values(&transport, "idempotency-key"), ["idem-1"]);

    let _ = provider(&transport)
        .embed(&embedding_request(tenant()))
        .await;
    assert_eq!(sent_values(&transport, "x-goog-api-key"), ["TENANT-SECRET"]);

    let debug = format!("{:?}", transport.requests());
    assert!(!debug.contains("SECRET"), "{debug}");
    for request in transport.requests() {
        assert!(
            !request
                .headers
                .iter()
                .any(|(_, value)| value.contains("NAMESPACE") || value.contains("FOREIGN")),
            "a credential for another scope was sent"
        );
    }
}

#[tokio::test]
async fn a_gateway_header_credential_reaches_only_its_scope() {
    let transport = ReplayTransport::default();
    let _ = provider(&transport)
        .chat(&chat_request(options(
            HttpRequestOptions::new()
                .credential("gemini", Credential::header("x-gateway-key", "GATEWAY"))
                .credential("anthropic", Credential::header("x-other-gateway", "OTHER")),
        )))
        .await;
    assert_eq!(sent_values(&transport, "x-gateway-key"), ["GATEWAY"]);
    assert_eq!(sent_values(&transport, "x-goog-api-key"), ["provider-key"]);
    assert!(sent_values(&transport, "x-other-gateway").is_empty());
}

#[tokio::test]
async fn a_plain_credential_header_is_refused_before_sending() {
    for name in ["x-goog-api-key", "X-Goog-Api-Key", "authorization"] {
        let transport = ReplayTransport::default();
        let err = provider(&transport)
            .embed(&embedding_request(options(
                HttpRequestOptions::new().header(name, "PLAIN-SECRET"),
            )))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
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
        .chat(&chat_request(RequestOptions::new()))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    assert!(transport.requests().is_empty());

    let _ = provider
        .embed(&embedding_request(options(
            HttpRequestOptions::new().credential("tenant", Credential::api_key("TENANT-KEY")),
        )))
        .await;
    assert_eq!(sent_values(&transport, "x-goog-api-key"), ["TENANT-KEY"]);
}
