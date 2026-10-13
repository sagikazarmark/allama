//! Transport options: the caller's per-request headers reach the wire on
//! every call path, and per-request credentials reach only the provider
//! whose credential scope they are addressed to.

use allama::prelude::*;
use allama_conformance::{TransportOutcome, assert_transport_options_applied};
use allama_http::{
    Credential, CredentialPlacement, CredentialSlot, HttpRequestOptions, ReplayTransport,
};
use allama_openai::{DEFAULT_CREDENTIAL_SCOPE, Provider, ResponsesProvider};

fn chat(transport: &ReplayTransport) -> Provider {
    Provider::builder()
        .api_key("provider-key")
        .http_transport(transport.clone())
        .build()
        .unwrap()
}

fn responses(transport: &ReplayTransport) -> ResponsesProvider {
    ResponsesProvider::builder()
        .api_key("provider-key")
        .http_transport(transport.clone())
        .build()
        .unwrap()
}

fn slot() -> CredentialSlot {
    CredentialSlot::new("openai", CredentialPlacement::Bearer)
}

fn chat_request(transport: HttpRequestOptions) -> ChatRequest {
    ChatRequest::new("gpt-4o-mini")
        .user("Hi")
        .with_option(transport)
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

#[test]
fn the_default_scope_is_the_provider_name() {
    assert_eq!(DEFAULT_CREDENTIAL_SCOPE, "openai");
    assert_eq!(chat(&ReplayTransport::default()).credential_slot(), &slot());
    assert_eq!(
        responses(&ReplayTransport::default()).credential_slot(),
        &slot()
    );
}

#[tokio::test]
async fn a_tenant_key_replaces_the_provider_key() {
    let tenant =
        HttpRequestOptions::new().credential("openai", Credential::api_key("TENANT-SECRET"));
    let transport = ReplayTransport::default();
    let _ = chat(&transport).chat(&chat_request(tenant.clone())).await;
    assert_eq!(
        sent_values(&transport, "Authorization"),
        ["Bearer TENANT-SECRET"]
    );
    let debug = format!("{:?}", transport.requests());
    assert!(!debug.contains("TENANT-SECRET"), "{debug}");

    let transport = ReplayTransport::default();
    let _ = responses(&transport).chat(&chat_request(tenant)).await;
    assert_eq!(
        sent_values(&transport, "Authorization"),
        ["Bearer TENANT-SECRET"]
    );
}

#[tokio::test]
async fn a_plain_authorization_header_is_refused_before_sending() {
    let plain = HttpRequestOptions::new().header("Authorization", "Bearer PLAIN-SECRET");
    let transport = ReplayTransport::default();
    let err = chat(&transport)
        .chat(&chat_request(plain.clone()))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    assert!(!err.is_retryable());
    assert!(!format!("{err} {err:?}").contains("PLAIN-SECRET"));
    assert!(transport.requests().is_empty());

    let transport = ReplayTransport::default();
    let err = responses(&transport)
        .chat(&chat_request(plain))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    assert!(transport.requests().is_empty());
}

#[tokio::test]
async fn two_accounts_with_two_scopes_each_send_their_own_key() {
    let account = |scope: &'static str, transport: &ReplayTransport| {
        Provider::builder()
            .api_key(format!("{scope}-configured"))
            .credential_scope(scope)
            .http_transport(transport.clone())
            .build()
            .unwrap()
    };
    let request = chat_request(
        HttpRequestOptions::new()
            .credential("tenant-a", Credential::api_key("key-a"))
            .credential("tenant-b", Credential::api_key("key-b")),
    );
    let (a, b) = (ReplayTransport::default(), ReplayTransport::default());
    let _ = account("tenant-a", &a).chat(&request).await;
    let _ = account("tenant-b", &b).chat(&request).await;
    assert_eq!(sent_values(&a, "authorization"), ["Bearer key-a"]);
    assert_eq!(sent_values(&b, "authorization"), ["Bearer key-b"]);

    // A default-scope provider sends neither, but its own configured key.
    let default = ReplayTransport::default();
    let _ = chat(&default).chat(&request).await;
    assert_eq!(
        sent_values(&default, "authorization"),
        ["Bearer provider-key"]
    );
}

#[tokio::test]
async fn a_gateway_header_reaches_only_its_scope() {
    let request = chat_request(
        HttpRequestOptions::new()
            .credential(
                "openai",
                Credential::header("cf-aig-authorization", "Bearer gw"),
            )
            .credential(
                "elsewhere",
                Credential::header("x-other-gateway", "OTHER-SECRET"),
            ),
    );
    let transport = ReplayTransport::default();
    let _ = chat(&transport).chat(&request).await;
    assert_eq!(
        sent_values(&transport, "cf-aig-authorization"),
        ["Bearer gw"]
    );
    assert_eq!(
        sent_values(&transport, "authorization"),
        ["Bearer provider-key"]
    );
    assert!(sent_values(&transport, "x-other-gateway").is_empty());

    let elsewhere = ReplayTransport::default();
    let _ = Provider::builder()
        .api_key("k")
        .credential_scope("tenant")
        .http_transport(elsewhere.clone())
        .build()
        .unwrap()
        .chat(&request)
        .await;
    assert!(sent_values(&elsewhere, "cf-aig-authorization").is_empty());
}

#[tokio::test]
async fn an_idempotency_key_passes() {
    let transport = ReplayTransport::default();
    let _ = responses(&transport)
        .chat(&chat_request(
            HttpRequestOptions::new().header("Idempotency-Key", "idem-1"),
        ))
        .await;
    assert_eq!(sent_values(&transport, "Idempotency-Key"), ["idem-1"]);
}

#[tokio::test]
async fn bring_your_own_key_needs_a_scope_and_a_credential() {
    let Err(err) = Provider::builder()
        .http_transport(ReplayTransport::default())
        .build()
    else {
        panic!("built without a key or a scope");
    };
    assert_eq!(err.message(), "api_key is required");

    let byok = |transport: &ReplayTransport| {
        Provider::builder()
            .credential_scope("tenant")
            .http_transport(transport.clone())
            .build()
            .unwrap()
    };
    let required = CredentialSlot::new("tenant", CredentialPlacement::Bearer).required(true);
    assert_eq!(
        byok(&ReplayTransport::default()).credential_slot(),
        &required
    );
    assert_transport_options_applied(&required, |options| async {
        let transport = ReplayTransport::default();
        let mut request = ChatRequest::new("gpt-4o-mini").user("Hi");
        request.options = options;
        let result = byok(&transport).chat(&request).await;
        TransportOutcome::new(result, transport.requests())
    })
    .await;

    let transport = ReplayTransport::default();
    let err = byok(&transport)
        .chat(&chat_request(HttpRequestOptions::new()))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    assert!(err.to_string().contains("`tenant`"), "{err}");
    assert!(transport.requests().is_empty());

    let transport = ReplayTransport::default();
    let _ = byok(&transport)
        .chat(&chat_request(
            HttpRequestOptions::new().credential("tenant", Credential::api_key("tenant-key")),
        ))
        .await;
    assert_eq!(
        sent_values(&transport, "authorization"),
        ["Bearer tenant-key"]
    );

    // The same for Responses, whose id-addressed calls take no options.
    let transport = ReplayTransport::default();
    let provider = ResponsesProvider::builder()
        .credential_scope("tenant")
        .http_transport(transport.clone())
        .build()
        .unwrap();
    assert_eq!(provider.credential_slot(), &required);
    let err = provider.retrieve_response("resp_1").await.unwrap_err();
    assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    assert!(transport.requests().is_empty());
}

#[tokio::test]
async fn with_credential_scope_readdresses_the_provider() {
    let transport = ReplayTransport::default();
    let provider = chat(&transport).with_credential_scope("tenant-b");
    assert_eq!(
        provider.credential_slot(),
        &CredentialSlot::new("tenant-b", CredentialPlacement::Bearer)
    );
    let _ = provider
        .chat(&chat_request(
            HttpRequestOptions::new()
                .credential("openai", Credential::api_key("a-key"))
                .credential("tenant-b", Credential::api_key("b-key")),
        ))
        .await;
    assert_eq!(sent_values(&transport, "authorization"), ["Bearer b-key"]);

    let responses = responses(&ReplayTransport::default()).with_credential_scope("tenant-r");
    assert_eq!(
        responses.credential_slot(),
        &CredentialSlot::new("tenant-r", CredentialPlacement::Bearer)
    );
}

#[test]
fn provider_debug_never_shows_credentials() {
    let provider = Provider::builder()
        .api_key("sk-SECRET")
        .credential_scope("tenant")
        .http_transport(ReplayTransport::default())
        .build()
        .unwrap();
    let responses = ResponsesProvider::builder()
        .api_key("sk-SECRET")
        .http_transport(ReplayTransport::default())
        .build()
        .unwrap();
    let debug = format!(
        "{provider:?} {:?} {responses:?} {:?}",
        provider.credential_slot(),
        responses.credential_slot()
    );
    assert!(!debug.contains("SECRET"), "{debug}");
}
