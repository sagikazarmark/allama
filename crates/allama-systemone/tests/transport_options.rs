//! Transport options: the caller's per-request headers reach the wire, and
//! the credentials addressed to the provider's credential scope, and no
//! other, are placed where its configured credential goes.

use allama::decision::Question;
use allama::{DecisionProvider, DecisionRequest, Error, RequestOptions};
use allama_conformance::{TransportOutcome, assert_transport_options_applied};
use allama_http::{
    Credential, CredentialPlacement, CredentialSlot, HttpRequestOptions, ReplayTransport,
};
use allama_systemone::Provider;
use allama_systemone::providers::{Cloudflare, LiteLlm, Ollama, TypeSafe};

fn typesafe(transport: &ReplayTransport) -> Provider {
    TypeSafe::builder("provider-key")
        .unwrap()
        .http_transport(transport.clone())
        .build()
        .unwrap()
}

fn request(options: RequestOptions) -> DecisionRequest {
    let mut request = DecisionRequest::new("nimble", "Is this a refund request?")
        .question("refund", Question::yes_no("Is it?"));
    request.options = options;
    request
}

fn options(transport: HttpRequestOptions) -> RequestOptions {
    let mut options = RequestOptions::new();
    options.insert(transport);
    options
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

#[tokio::test]
async fn a_custom_auth_header_is_the_placement() {
    let build = |transport: &ReplayTransport| {
        Provider::builder()
            .base_url("https://gateway.example")
            .auth_header("x-custom", "provider-key")
            .header("x-gateway-tag", "kept")
            .http_transport(transport.clone())
            .build()
            .unwrap()
    };
    let slot = CredentialSlot::new(
        allama_systemone::DEFAULT_CREDENTIAL_SCOPE,
        CredentialPlacement::Header("x-custom".into()),
    );
    assert_eq!(build(&ReplayTransport::default()).credential_slot(), &slot);
    assert_transport_options_applied(&slot, |options| async {
        let transport = ReplayTransport::default();
        let result = build(&transport).decide(&request(options)).await;
        TransportOutcome::new(result, transport.requests())
    })
    .await;

    let transport = ReplayTransport::default();
    let _ = build(&transport)
        .decide(&request(options(
            HttpRequestOptions::new().credential("systemone", Credential::api_key("TENANT-SECRET")),
        )))
        .await;
    assert_eq!(sent_values(&transport, "x-custom"), ["TENANT-SECRET"]);
    assert_eq!(sent_values(&transport, "x-gateway-tag"), ["kept"]);
    assert!(sent_values(&transport, "authorization").is_empty());
}

#[tokio::test]
async fn a_placement_other_than_the_configured_header_replaces_it() {
    let build = |transport: &ReplayTransport| {
        Provider::builder()
            .base_url("https://gateway.example")
            .bearer_token("entra-token")
            .credential_placement(CredentialPlacement::Header("api-key".into()))
            .http_transport(transport.clone())
            .build()
            .unwrap()
    };
    let slot = CredentialSlot::new("systemone", CredentialPlacement::Header("api-key".into()))
        .replaces("authorization");
    assert_eq!(build(&ReplayTransport::default()).credential_slot(), &slot);
    assert_transport_options_applied(&slot, |options| async {
        let transport = ReplayTransport::default();
        let result = build(&transport).decide(&request(options)).await;
        TransportOutcome::new(result, transport.requests())
    })
    .await;
}

#[tokio::test]
async fn a_tenant_key_replaces_the_provider_key() {
    let transport = ReplayTransport::default();
    let _ = typesafe(&transport)
        .decide(&request(options(
            HttpRequestOptions::new()
                .credential("typesafe", Credential::api_key("TENANT-SECRET"))
                .credential("openai", Credential::api_key("FOREIGN-SECRET"))
                .credential(
                    "typesafe",
                    Credential::header("cf-aig-authorization", "Bearer GATEWAY-SECRET"),
                )
                .credential("openai", Credential::header("x-other-gateway", "OTHER"))
                .header("Idempotency-Key", "idem-1"),
        )))
        .await;
    assert_eq!(
        sent_values(&transport, "authorization"),
        ["Bearer TENANT-SECRET"]
    );
    assert_eq!(
        sent_values(&transport, "cf-aig-authorization"),
        ["Bearer GATEWAY-SECRET"]
    );
    assert_eq!(sent_values(&transport, "idempotency-key"), ["idem-1"]);
    assert!(sent_values(&transport, "x-other-gateway").is_empty());
    let sent = &transport.requests()[0];
    assert!(
        !sent
            .headers
            .iter()
            .any(|(_, value)| value.contains("FOREIGN")),
        "a credential for another scope was sent"
    );
    let debug = format!("{:?}", transport.requests());
    assert!(!debug.contains("SECRET"), "{debug}");
}

#[tokio::test]
async fn a_plain_credential_header_is_refused_before_sending() {
    for name in ["authorization", "Authorization", "x-api-key"] {
        let transport = ReplayTransport::default();
        let err = typesafe(&transport)
            .decide(&request(options(
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
async fn presets_default_their_scope_to_their_provider_name() {
    let transport = ReplayTransport::default();
    assert_eq!(TypeSafe::DEFAULT_CREDENTIAL_SCOPE, TypeSafe::PROVIDER_NAME);
    assert_eq!(typesafe(&transport).credential_slot().scope, "typesafe");
    let scoped = typesafe(&transport).with_credential_scope("typesafe-b");
    assert_eq!(scoped.credential_slot().scope, "typesafe-b");
    assert!(!scoped.credential_slot().required);
    assert_eq!(Ollama::DEFAULT_CREDENTIAL_SCOPE, "ollama");
    // The scope `allama-openai-compat`'s Cloudflare preset uses, so one
    // per-request credential reaches the account for chat and decisions.
    assert_eq!(Cloudflare::DEFAULT_CREDENTIAL_SCOPE, "cloudflare");
    let cloudflare = Cloudflare::builder("0123456789abcdef0123456789abcdef", "t")
        .unwrap()
        .http_transport(transport.clone())
        .build()
        .unwrap();
    assert_eq!(cloudflare.credential_slot().scope, "cloudflare");
}

#[tokio::test]
async fn keyless_presets_send_no_auth_and_require_none() {
    let ollama = Ollama::builder("http://127.0.0.1:11434")
        .unwrap()
        .http_transport(ReplayTransport::default())
        .build()
        .unwrap();
    let litellm = LiteLlm::builder("http://127.0.0.1:4000", None)
        .unwrap()
        .http_transport(ReplayTransport::default())
        .build()
        .unwrap();
    for (provider, scope) in [(ollama, "ollama"), (litellm, "litellm")] {
        assert_eq!(
            provider.credential_slot(),
            &CredentialSlot::new(scope, CredentialPlacement::Bearer)
        );
    }

    let transport = ReplayTransport::default();
    let ollama = Ollama::builder("http://127.0.0.1:11434")
        .unwrap()
        .http_transport(transport.clone())
        .build()
        .unwrap();
    let _ = ollama.decide(&request(RequestOptions::new())).await;
    assert!(sent_values(&transport, "authorization").is_empty());

    // A keyless provider given a scope explicitly brings its own key.
    let byok = ollama.with_credential_scope("tenant");
    assert!(byok.credential_slot().required);
}

#[tokio::test]
async fn a_byok_provider_needs_a_credential_for_its_scope() {
    let slot = CredentialSlot::new("tenant", CredentialPlacement::Bearer).required(true);
    let build = |transport: &ReplayTransport| {
        Provider::builder()
            .base_url("https://gateway.example")
            .credential_scope("tenant")
            .http_transport(transport.clone())
            .build()
            .unwrap()
    };
    assert_transport_options_applied(&slot, |options| async {
        let transport = ReplayTransport::default();
        let provider = build(&transport);
        assert_eq!(provider.credential_slot(), &slot);
        let result = provider.decide(&request(options)).await;
        TransportOutcome::new(result, transport.requests())
    })
    .await;

    let transport = ReplayTransport::default();
    let err = build(&transport)
        .decide(&request(RequestOptions::new()))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    assert!(transport.requests().is_empty());
}
