//! Per-request credentials across providers: in a fallback chain, each
//! provider sends only the credentials addressed to its credential scope.

use allama::wrappers::{FallbackProvider, MapModelProvider};
use allama::{ChatProvider, ChatRequest, Error};
use allama_http::http::{HeaderValue, StatusCode};
use allama_http::{Credential, HttpRequest, HttpRequestOptions, ReplayResponse, ReplayTransport};

const ANTHROPIC_OK: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-haiku-4-5","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":1}}"#;

const OPENAI_OK: &str = r#"{"id":"c1","model":"gpt-5-mini","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}"#;

fn values<'a>(request: &'a HttpRequest, name: &str) -> Vec<&'a str> {
    request
        .headers
        .get_all(name)
        .iter()
        .map(|value| value.to_str().unwrap())
        .collect()
}

fn openai(transport: &ReplayTransport, key: &str) -> allama_openai::Provider {
    allama_openai::Provider::builder()
        .api_key(key)
        .http_transport(transport.clone())
        .build()
        .unwrap()
}

fn anthropic(transport: &ReplayTransport) -> allama_anthropic::Provider {
    allama_anthropic::Provider::builder()
        .api_key("anthropic-own-key")
        .http_transport(transport.clone())
        .build()
        .unwrap()
}

#[tokio::test]
async fn an_openai_scoped_key_never_reaches_the_anthropic_fallback() {
    let openai_transport = ReplayTransport::new([ReplayResponse::json(
        StatusCode::SERVICE_UNAVAILABLE,
        r#"{"error":{"message":"overloaded","type":"server_error"}}"#,
    )]);
    let anthropic_transport =
        ReplayTransport::new([ReplayResponse::json(StatusCode::OK, ANTHROPIC_OK)]);
    let provider = FallbackProvider::new(
        openai(&openai_transport, "openai-server-key"),
        MapModelProvider::new(anthropic(&anthropic_transport), |_: &str| {
            "claude-haiku-4-5".into()
        }),
    );

    let request = ChatRequest::new("gpt-5-mini").user("Hi").with_option(
        HttpRequestOptions::new().credential(
            allama_openai::DEFAULT_CREDENTIAL_SCOPE,
            Credential::api_key("TENANT-OPENAI-SECRET"),
        ),
    );
    let response = provider.chat(&request).await.unwrap();
    assert_eq!(response.text_or_empty(), "hi");

    let openai_sent = openai_transport.requests();
    assert_eq!(
        values(&openai_sent[0], "authorization"),
        ["Bearer TENANT-OPENAI-SECRET"]
    );

    let anthropic_sent = anthropic_transport.requests();
    assert_eq!(anthropic_sent.len(), 1);
    let sent = &anthropic_sent[0];
    assert_eq!(values(sent, "x-api-key"), ["anthropic-own-key"]);
    assert!(values(sent, "authorization").is_empty());
    assert!(
        !sent
            .headers
            .iter()
            .any(|(_, value)| value.to_str().unwrap().contains("TENANT-OPENAI-SECRET")),
        "the OpenAI key reached Anthropic"
    );
    assert!(!sent.body_text().unwrap().contains("TENANT-OPENAI-SECRET"));
}

#[tokio::test]
async fn two_openai_accounts_with_two_scopes_each_send_their_own_key() {
    let first_transport = ReplayTransport::new([ReplayResponse::json(
        StatusCode::TOO_MANY_REQUESTS,
        r#"{"error":{}}"#,
    )]);
    let second_transport = ReplayTransport::new([ReplayResponse::json(StatusCode::OK, OPENAI_OK)]);
    let first = allama_openai::Provider::builder()
        .api_key("server-key-a")
        .credential_scope("openai-account-a")
        .http_transport(first_transport.clone())
        .build()
        .unwrap();
    let second = allama_openai::Provider::builder()
        .api_key("server-key-b")
        .credential_scope("openai-account-b")
        .http_transport(second_transport.clone())
        .build()
        .unwrap();
    let provider = FallbackProvider::new(first, second);

    let request = ChatRequest::new("gpt-5-mini").user("Hi").with_option(
        HttpRequestOptions::new()
            .credential("openai-account-a", Credential::api_key("TENANT-A"))
            .credential("openai-account-b", Credential::api_key("TENANT-B")),
    );
    provider.chat(&request).await.unwrap();

    assert_eq!(
        values(&first_transport.requests()[0], "authorization"),
        ["Bearer TENANT-A"]
    );
    assert_eq!(
        values(&second_transport.requests()[0], "authorization"),
        ["Bearer TENANT-B"]
    );
}

#[tokio::test]
async fn a_gateway_token_reaches_only_its_scope() {
    // OpenAI behind a gateway that wants its own token; Anthropic direct.
    let openai_transport = ReplayTransport::new([ReplayResponse::json(
        StatusCode::INTERNAL_SERVER_ERROR,
        r#"{"error":{}}"#,
    )]);
    let anthropic_transport =
        ReplayTransport::new([ReplayResponse::json(StatusCode::OK, ANTHROPIC_OK)]);
    let provider = FallbackProvider::new(
        openai(&openai_transport, "openai-server-key"),
        MapModelProvider::new(anthropic(&anthropic_transport), |_: &str| {
            "claude-haiku-4-5".into()
        }),
    );
    let request = ChatRequest::new("gpt-5-mini").user("Hi").with_option(
        HttpRequestOptions::new().credential(
            "openai",
            Credential::header("cf-aig-authorization", "Bearer GATEWAY-SECRET"),
        ),
    );
    provider.chat(&request).await.unwrap();

    let openai_sent = &openai_transport.requests()[0];
    assert_eq!(
        values(openai_sent, "cf-aig-authorization"),
        ["Bearer GATEWAY-SECRET"]
    );
    assert_eq!(
        values(openai_sent, "authorization"),
        ["Bearer openai-server-key"]
    );
    let anthropic_sent = &anthropic_transport.requests()[0];
    assert!(values(anthropic_sent, "cf-aig-authorization").is_empty());
}

#[tokio::test]
async fn a_plain_credential_header_is_refused_and_never_failed_over() {
    let openai_transport = ReplayTransport::new([ReplayResponse::json(StatusCode::OK, OPENAI_OK)]);
    let anthropic_transport =
        ReplayTransport::new([ReplayResponse::json(StatusCode::OK, ANTHROPIC_OK)]);
    let provider = FallbackProvider::new(
        openai(&openai_transport, "openai-server-key"),
        MapModelProvider::new(anthropic(&anthropic_transport), |_: &str| {
            "claude-haiku-4-5".into()
        }),
    );
    let request =
        ChatRequest::new("gpt-5-mini")
            .user("Hi")
            .with_option(HttpRequestOptions::new().header(
                "authorization",
                HeaderValue::from_static("Bearer PLAIN-SECRET"),
            ));
    let err = provider.chat(&request).await.unwrap_err();
    assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    assert!(!err.to_string().contains("PLAIN-SECRET"));
    assert!(openai_transport.requests().is_empty());
    assert!(anthropic_transport.requests().is_empty());
}
