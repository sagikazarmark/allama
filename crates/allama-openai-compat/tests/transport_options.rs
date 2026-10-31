//! Transport options: the caller's per-request headers reach the wire on
//! every call path, and per-request credentials reach only the provider
//! whose credential scope they are addressed to.

use allama::prelude::*;
use allama_conformance::{TransportOutcome, assert_transport_options_applied};
use allama_http::http::HeaderValue;
use allama_http::{
    Credential, CredentialPlacement, CredentialSlot, HttpRequestOptions, ReplayTransport,
};
use allama_openai_compat::providers::{Azure, Groq, Ollama};
use allama_openai_compat::{Provider, ResponsesProvider};

fn compat(transport: &ReplayTransport) -> Provider {
    Provider::builder()
        .base_url("http://upstream/v1")
        .bearer_token("provider-key")
        .http_transport(transport.clone())
        .build()
        .unwrap()
}

fn responses(transport: &ReplayTransport) -> ResponsesProvider {
    ResponsesProvider::builder()
        .base_url("http://upstream/v1")
        .bearer_token("provider-key")
        .http_transport(transport.clone())
        .build()
        .unwrap()
}

fn chat_request(transport: HttpRequestOptions) -> ChatRequest {
    ChatRequest::new("m").user("Hi").with_option(transport)
}

fn sent_values(transport: &ReplayTransport, name: &str) -> Vec<String> {
    let requests = transport.requests();
    let request = requests.last().expect("the adapter sent a request");
    request
        .headers
        .get_all(name)
        .iter()
        .map(|value| value.to_str().unwrap().to_owned())
        .collect()
}

#[tokio::test]
async fn azure_places_a_tenant_key_in_api_key() {
    let slot = CredentialSlot::new("azure", CredentialPlacement::Header("api-key".into()));
    let azure = |transport: &ReplayTransport| {
        Azure::builder("res", "provider-key")
            .unwrap()
            .http_transport(transport.clone())
            .build()
            .unwrap()
    };
    assert_eq!(azure(&ReplayTransport::default()).credential_slot(), &slot);
    assert_transport_options_applied(&slot, |options| async {
        let transport = ReplayTransport::default();
        let mut request = ChatRequest::new("m").user("Hi");
        request.options = options;
        let result = azure(&transport).chat(&request).await;
        TransportOutcome::new(result, transport.requests())
    })
    .await;
}

#[tokio::test]
async fn azure_tenant_key_replaces_a_configured_entra_token() {
    let entra = |transport: &ReplayTransport| {
        Azure::builder("res", "unused")
            .unwrap()
            .bearer_token("entra-token")
            .http_transport(transport.clone())
            .build()
            .unwrap()
    };
    let slot = CredentialSlot::new("azure", CredentialPlacement::Header("api-key".into()))
        .replaces("authorization");
    assert_eq!(entra(&ReplayTransport::default()).credential_slot(), &slot);
    assert_transport_options_applied(&slot, |options| async {
        let transport = ReplayTransport::default();
        let mut request = ChatRequest::new("m").user("Hi");
        request.options = options;
        let result = entra(&transport).chat(&request).await;
        TransportOutcome::new(result, transport.requests())
    })
    .await;

    let transport = ReplayTransport::default();
    let _ = entra(&transport)
        .chat(&chat_request(
            HttpRequestOptions::new().credential("azure", Credential::api_key("TENANT-SECRET")),
        ))
        .await;
    assert_eq!(sent_values(&transport, "api-key"), ["TENANT-SECRET"]);
    assert!(sent_values(&transport, "authorization").is_empty());

    // Without a tenant key, the configured Entra token is sent.
    let transport = ReplayTransport::default();
    let _ = entra(&transport)
        .chat(&chat_request(HttpRequestOptions::new()))
        .await;
    assert_eq!(
        sent_values(&transport, "authorization"),
        ["Bearer entra-token"]
    );
}

#[tokio::test]
async fn a_tenant_key_replaces_the_provider_key() {
    let tenant =
        HttpRequestOptions::new().credential("openai-compat", Credential::api_key("TENANT-SECRET"));
    let transport = ReplayTransport::default();
    let _ = compat(&transport).chat(&chat_request(tenant.clone())).await;
    assert_eq!(
        sent_values(&transport, "authorization"),
        ["Bearer TENANT-SECRET"]
    );
    let debug = format!("{:?}", transport.requests());
    assert!(!debug.contains("TENANT-SECRET"), "{debug}");

    let transport = ReplayTransport::default();
    let _ = responses(&transport)
        .chat(&chat_request(tenant.clone()))
        .await;
    assert_eq!(
        sent_values(&transport, "authorization"),
        ["Bearer TENANT-SECRET"]
    );

    let transport = ReplayTransport::default();
    let request = EmbeddingRequest::new("m")
        .inputs(["hello"])
        .with_option(tenant);
    let _ = compat(&transport).embed(&request).await;
    assert_eq!(
        sent_values(&transport, "authorization"),
        ["Bearer TENANT-SECRET"]
    );
}

#[tokio::test]
async fn a_plain_credential_header_is_refused_before_sending() {
    let plain = HttpRequestOptions::new().header(
        "Authorization",
        HeaderValue::from_static("Bearer PLAIN-SECRET"),
    );
    let transport = ReplayTransport::default();
    let err = compat(&transport)
        .chat(&chat_request(plain.clone()))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    assert!(!err.to_string().contains("PLAIN-SECRET"), "{err}");
    assert!(transport.requests().is_empty());

    let transport = ReplayTransport::default();
    let Err(err) = responses(&transport)
        .chat_stream(&chat_request(plain))
        .await
    else {
        panic!("a plain credential header was sent");
    };
    assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    assert!(transport.requests().is_empty());
}

#[tokio::test]
async fn an_idempotency_key_passes() {
    let transport = ReplayTransport::default();
    let _ = compat(&transport)
        .chat(&chat_request(HttpRequestOptions::new().header(
            "Idempotency-Key",
            HeaderValue::from_static("idem-1"),
        )))
        .await;
    assert_eq!(sent_values(&transport, "idempotency-key"), ["idem-1"]);
    assert_eq!(
        sent_values(&transport, "authorization"),
        ["Bearer provider-key"]
    );
}

#[tokio::test]
async fn a_preset_receives_only_credentials_for_its_scope() {
    let options = HttpRequestOptions::new()
        .credential("openai-compat", Credential::api_key("OTHER-SECRET"))
        .credential(
            "openai-compat",
            Credential::header("cf-aig-authorization", "Bearer OTHER-GATEWAY"),
        )
        .credential(
            "groq",
            Credential::header("cf-aig-authorization", "Bearer gw"),
        );
    let transport = ReplayTransport::default();
    let groq = Groq::builder("groq-key")
        .unwrap()
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let _ = groq.chat(&chat_request(options)).await;
    assert_eq!(
        sent_values(&transport, "authorization"),
        ["Bearer groq-key"]
    );
    assert_eq!(
        sent_values(&transport, "cf-aig-authorization"),
        ["Bearer gw"]
    );
    let requests = format!(
        "{:?}",
        transport.requests()[0].headers.iter().collect::<Vec<_>>()
    );
    assert!(!requests.contains("OTHER"), "{requests}");
}

#[tokio::test]
async fn bring_your_own_key_requires_a_credential_for_the_scope() {
    let byok = |transport: &ReplayTransport| {
        Provider::builder()
            .base_url("http://upstream/v1")
            .credential_scope("tenant")
            .http_transport(transport.clone())
            .build()
            .unwrap()
    };
    let slot = CredentialSlot::new("tenant", CredentialPlacement::Bearer).required(true);
    assert_eq!(byok(&ReplayTransport::default()).credential_slot(), &slot);
    assert_transport_options_applied(&slot, |options| async {
        let transport = ReplayTransport::default();
        let mut request = ChatRequest::new("m").user("Hi");
        request.options = options;
        let result = byok(&transport).chat(&request).await;
        TransportOutcome::new(result, transport.requests())
    })
    .await;

    let transport = ReplayTransport::default();
    let err = byok(&transport)
        .chat(&chat_request(
            HttpRequestOptions::new().credential("other", Credential::api_key("k")),
        ))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
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
}

#[tokio::test]
async fn bring_your_own_key_retrieval_fails_without_options() {
    let transport = ReplayTransport::default();
    let provider = ResponsesProvider::builder()
        .base_url("http://upstream/v1")
        .credential_scope("tenant")
        .http_transport(transport.clone())
        .build()
        .unwrap();
    assert!(provider.credential_slot().required);
    let err = provider.retrieve_response("resp_1").await.unwrap_err();
    assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    let err = provider.cancel_response("resp_1").await.unwrap_err();
    assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    let Err(err) = provider.resume_stream("resp_1", None).await else {
        panic!("resumed without a credential");
    };
    assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    assert!(transport.requests().is_empty());
}

#[tokio::test]
async fn a_keyless_endpoint_sends_no_auth_and_requires_none() {
    let transport = ReplayTransport::default();
    let ollama = Ollama::builder("http://localhost:11434")
        .unwrap()
        .http_transport(transport.clone())
        .build()
        .unwrap();
    assert_eq!(
        ollama.credential_slot(),
        &CredentialSlot::new("ollama", CredentialPlacement::Bearer)
    );
    let _ = ollama.chat(&chat_request(HttpRequestOptions::new())).await;
    assert!(sent_values(&transport, "authorization").is_empty());

    // A key addressed to it is still placed.
    let transport = ReplayTransport::default();
    let ollama = Ollama::builder("http://localhost:11434")
        .unwrap()
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let _ = ollama
        .chat(&chat_request(
            HttpRequestOptions::new().credential("ollama", Credential::api_key("k")),
        ))
        .await;
    assert_eq!(sent_values(&transport, "authorization"), ["Bearer k"]);
}

#[tokio::test]
async fn with_credential_scope_readdresses_the_provider() {
    let transport = ReplayTransport::default();
    let provider = compat(&transport).with_credential_scope("tenant-b");
    assert_eq!(
        provider.credential_slot(),
        &CredentialSlot::new("tenant-b", CredentialPlacement::Bearer)
    );
    let _ = provider
        .chat(&chat_request(
            HttpRequestOptions::new()
                .credential("openai-compat", Credential::api_key("a-key"))
                .credential("tenant-b", Credential::api_key("b-key")),
        ))
        .await;
    assert_eq!(sent_values(&transport, "authorization"), ["Bearer b-key"]);

    // A keyless provider readdressed to a scope requires its credential.
    let keyless = Provider::builder()
        .base_url("http://upstream/v1")
        .http_transport(ReplayTransport::default())
        .build()
        .unwrap();
    assert!(!keyless.credential_slot().required);
    assert!(
        keyless
            .with_credential_scope("tenant")
            .credential_slot()
            .required
    );

    let responses = responses(&ReplayTransport::default()).with_credential_scope("tenant-r");
    assert_eq!(responses.credential_slot().scope, "tenant-r");
    assert!(!responses.credential_slot().required);
}

#[tokio::test]
async fn auth_headers_set_the_placement_and_explicit_placements_replace_them() {
    let custom = Provider::builder()
        .base_url("http://upstream/v1")
        .auth_header("x-nimbus-key", "configured")
        .provider_name("nimbus")
        .http_transport(ReplayTransport::default())
        .build()
        .unwrap();
    assert_eq!(
        custom.credential_slot(),
        &CredentialSlot::new("nimbus", CredentialPlacement::Header("x-nimbus-key".into()))
    );

    let transport = ReplayTransport::default();
    let moved = Provider::builder()
        .base_url("http://upstream/v1")
        .auth_header("x-nimbus-key", "configured")
        .credential_placement(CredentialPlacement::Bearer)
        .http_transport(transport.clone())
        .build()
        .unwrap();
    assert_eq!(
        moved.credential_slot(),
        &CredentialSlot::new("openai-compat", CredentialPlacement::Bearer).replaces("x-nimbus-key")
    );
    let _ = moved
        .chat(&chat_request(
            HttpRequestOptions::new().credential("openai-compat", Credential::api_key("k")),
        ))
        .await;
    assert_eq!(sent_values(&transport, "authorization"), ["Bearer k"]);
    assert!(sent_values(&transport, "x-nimbus-key").is_empty());
}

#[test]
fn provider_debug_never_shows_credentials() {
    let provider = Provider::builder()
        .base_url("http://upstream/v1")
        .bearer_token("sk-SECRET")
        .credential_scope("tenant")
        .http_transport(ReplayTransport::default())
        .build()
        .unwrap();
    let responses = ResponsesProvider::builder()
        .base_url("http://upstream/v1")
        .bearer_token("sk-SECRET")
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
