//! # Implementing a provider for `allama`
//!
//! A provider maps the portable request to its wire format, sends it, and
//! maps the reply back. This example shows every part of that contract with
//! an in-memory "wire":
//!
//! 1. **Identity and capabilities.** [`ProviderIdentity`]: a provider name
//!    (who serves calls) and `downcast_any` (expose the provider so callers
//!    can reach it behind wrappers), plus honest [`ChatCapability`] answers;
//!    `Unknown` when unsure. The name and the capability answers are
//!    required. The provider name and the namespace (options and provider
//!    data) are separate constants, even when equal.
//! 2. **Provider options.** Declare an options type implementing
//!    [`ProviderOptions`]; read it with `request.resolve_option::<T>()`,
//!    which accepts the exact type or namespaced JSON.
//! 3. **Preflight and degradations.** Call [`allama::adapter::chat::prepare`]
//!    first: it applies your capability answers and the request's
//!    [`UnsupportedPolicy`]. Then report what the capabilities do not cover
//!    through its [`Degradations`]: semantic fields you cannot honor are
//!    errors, hints are dropped with a [`Warning`] (or rejected under strict
//!    policy). Keep everything before sending in one `encode` function, so
//!    `validate_chat_request` can run it as a dry run, and end the call with
//!    `finish`, which joins the warnings to the response.
//! 4. **Provider data.** Store opaque replay state under your own namespace
//!    in [`ProviderData`] and only read your own namespace back. That keeps
//!    conversations safe to move between providers.
//! 5. **Response identity.** Stamp `provider` (served-by), `request_id`, the
//!    normalized and raw finish reason, and usage per the [`Usage`] contract.
//! 6. **Errors.** Map upstream failures to the right [`Error`] variant with
//!    its [`ErrorContext`](allama::ErrorContext) so retry and fallback react
//!    correctly:
//!
//! | Upstream signal | `allama::Error` |
//! |---|---|
//! | bad credentials (401, 403) | `Error::auth` |
//! | rate limited (429) | `Error::rate_limited` + `with_retry_after` |
//! | quota / billing exhausted (402, or a quota code) | `Error::quota_exhausted` (not retryable) |
//! | overloaded (503, 529) | `Error::overloaded` |
//! | timed out (408, 504, 524) | `Error::timeout` |
//! | client closed the request (499) | `Error::provider` + `with_status` (not retryable) |
//! | other 5xx | `Error::provider` + `with_status` (501 and 505 are not retryable) |
//! | other 4xx | `Error::invalid_request` |
//! | no response | `Error::transport` |
//! | unknown model | `Error::model_not_found` |
//! | prompt too long | `Error::context_length_exceeded` |
//! | safety block | `Error::content_filtered` |
//! | undecodable success body | `Error::unexpected_response` |
//!
//!    HTTP adapters get the status rows from `allama_http::status_error`
//!    and refine them from the body, and build the last row with
//!    `allama_http::malformed_response`.
//!
//!    Mark messages the adapter wrote with [`Error::authored`] so the
//!    redacted [`Error::as_log`] shows them, and put the provider's own
//!    error text in [`Error::with_upstream_message`], never in an authored
//!    message: it may echo the prompt.
//!
//! Provider crates add fixture-based tests with `allama-conformance`.
//!
//! The provider author guide, `docs/guides/provider-authors.md`, covers the
//! rest: when an OpenAI-compatible preset, a `CompatQuirks` dialect, or a
//! native adapter fits, streaming over `allama-http`, the conformance
//! contracts, and registering with `allama_providers::Registry`.

use allama::adapter::chat::Prepared;
use allama::prelude::*;
use allama::{
    CapabilitySupport, ChatCapability, ChatValidation, FinishReason, ProviderData,
    ProviderIdentity, ProviderOptions, Reasoning, RequestField, SingleResponseStream,
    UnsupportedPolicy, Usage, ValidationCoverage,
};
use serde::{Deserialize, Serialize};

/// Namespace of the echo provider's options and provider data.
const NAMESPACE: &str = "echo";

/// Who serves calls: stamped on responses, errors, and warnings.
const PROVIDER_NAME: &str = "echo";

/// Options understood by the echo provider.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EchoOptions {
    /// Repeat the last user message this many times.
    repeat: Option<u8>,
}

impl ProviderOptions for EchoOptions {
    const NAMESPACE: &'static str = NAMESPACE;
}

struct EchoProvider;

impl ProviderIdentity for EchoProvider {
    fn provider_name(&self) -> &str {
        PROVIDER_NAME
    }

    /// The state this provider writes lives under its namespace, and it reads
    /// no other.
    fn reads_namespace(&self, namespace: &str) -> bool {
        namespace == NAMESPACE
    }

    fn downcast_any(&self, id: std::any::TypeId) -> Option<&(dyn std::any::Any + Send + Sync)> {
        allama::expose(self, id)
    }
}

impl ChatProvider for EchoProvider {
    type Stream = SingleResponseStream;

    async fn chat(&self, request: &ChatRequest) -> allama::Result<ChatResponse> {
        // Preflight first: features `chat_capability` reports `Unsupported`
        // fail (structured output) or are removed with a warning (reasoning
        // configuration), per the request's policy.
        let mut prepared = allama::adapter::chat::prepare(self, request)?;
        let text = encode(&mut prepared)?;
        // Immediately before sending: a failure from here on still reports
        // the warnings as events.
        prepared.mark_sent();
        // The "server" answers: only it knows its models.
        let model = prepared.request().model.clone();
        if !model.starts_with("echo-") {
            return Err(Error::model_not_found(format!("unknown model `{model}`"))
                .authored()
                .with_provider(PROVIDER_NAME));
        }

        // Opaque state lives under our own namespace.
        let reasoning = Reasoning::new("echoing").with_provider_data(ProviderData::new().with_raw(
            NAMESPACE,
            "signature",
            serde_json::json!("sig-1"),
        ));

        let response = ChatResponse::new(vec![
            ContentBlock::Reasoning(reasoning),
            ContentBlock::text(text),
        ])
        .finish_reason(FinishReason::Stop)
        .finish_reason_raw("end")
        .usage(
            Usage::new()
                .input_tokens(3)
                .output_tokens(5)
                .reasoning_tokens(1)
                .with_derived_total(),
        )
        .model(model)
        .provider(PROVIDER_NAME)
        .request_id("req-local");
        // The request-side warnings join the response.
        Ok(prepared.finish(response))
    }

    async fn chat_stream(&self, request: &ChatRequest) -> allama::Result<Self::Stream> {
        Ok(self.chat(request).await?.into_stream())
    }

    fn chat_capability(&self, _model: &str, capability: ChatCapability) -> CapabilitySupport {
        match capability {
            ChatCapability::Streaming | ChatCapability::ReasoningOutput => {
                CapabilitySupport::Supported
            }
            ChatCapability::NativeStreaming
            | ChatCapability::StructuredOutput
            | ChatCapability::ReasoningConfig => CapabilitySupport::Unsupported,
            _ => CapabilitySupport::Unknown,
        }
    }

    /// A dry run: the same preflight and encoding as `chat`, without
    /// sending.
    fn validate_chat_request(&self, request: &ChatRequest) -> allama::Result<ChatValidation> {
        let mut prepared = allama::adapter::chat::prepare_dry_run(self, request)?;
        encode(&mut prepared)?;
        Ok(prepared.into_validation(ValidationCoverage::Encoding))
    }

    fn validate_chat_stream_request(
        &self,
        request: &ChatRequest,
    ) -> allama::Result<ChatValidation> {
        let mut prepared = allama::adapter::chat::prepare_dry_run(self, request)?;
        prepared.preflight_stream(self);
        encode(&mut prepared)?;
        Ok(prepared.into_validation(ValidationCoverage::Encoding))
    }
}

/// Everything before sending, shared by the call and its dry run: the
/// fields no capability covers, reported through the same helper, then the
/// "wire" request (here, the text to echo).
fn encode(prepared: &mut Prepared<'_>) -> allama::Result<String> {
    let (request, degradations) = prepared.parts_mut();
    if request.seed.is_some() {
        degradations
            .unsupported_hint(RequestField::Seed, "echo has no seed; `seed` was not sent")?;
    }
    // JSON mode has no capability (`StructuredOutput` covers JSON Schema
    // formats, which preflight refused above): an API without it refuses
    // it here.
    if matches!(request.response_format, Some(ResponseFormat::Json)) {
        return Err(degradations.unsupported(
            RequestField::ResponseFormat,
            "echo returns plain text; JSON mode is not supported",
        ));
    }
    let options: EchoOptions = request.resolve_option()?;
    let last_user = request
        .messages
        .iter()
        .rev()
        .find_map(|m| m.as_user())
        .map(|m| {
            m.content
                .to_parts()
                .iter()
                .filter_map(ContentPart::as_text)
                .collect::<String>()
        })
        .unwrap_or_default();
    Ok(last_user.repeat(usize::from(options.repeat.unwrap_or(1))))
}

#[tokio::main]
async fn main() -> allama::Result<()> {
    let provider = EchoProvider;

    let request = ChatRequest::new("echo-1")
        .user("hi ")
        .seed(7)
        .reasoning(ReasoningEffort::High)
        .with_options_json(NAMESPACE, serde_json::json!({"repeat": 2}));
    // Validation is a dry run of `chat`: the answer and the warnings `chat`
    // would give, without sending.
    let validation = provider.validate_chat_request(&request)?;
    println!("validation warnings: {}", validation.warnings().len());
    let response = provider.chat(&request).await?;
    println!("text: {}", response.text_or_empty());
    for warning in &response.warnings {
        println!("warning: {warning}");
    }

    let strict = request.clone().unsupported_policy(UnsupportedPolicy::Error);
    println!(
        "strict policy: {}",
        provider.chat(&strict).await.unwrap_err()
    );

    let json = ChatRequest::new("echo-1")
        .user("hi")
        .response_format(ResponseFormat::json_schema(
            serde_json::json!({"type": "object"}),
        ));
    println!(
        "semantic field: {}",
        provider.chat(&json).await.unwrap_err()
    );

    let unknown = ChatRequest::new("other").user("hi");
    println!(
        "error mapping: {}",
        provider.chat(&unknown).await.unwrap_err()
    );
    Ok(())
}
