use allama::adapter::chat::Prepared;
use allama::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, ChatStream,
    ChatValidation, Message, ProviderIdentity, Result, ValidationCoverage,
};

use allama_http::http::header::CONTENT_TYPE;
use allama_http::http::{HeaderMap, HeaderValue};
use allama_http::sse::{SseChatStream, SseStream};
use allama_http::{HttpRequest, HttpResponse, request_uri};

use crate::Provider;
use crate::encode::encode_request_for;
use crate::error::{
    map_body_read_error, map_http_error, map_response_decode_error, map_transport_error,
};
use crate::model::ModelProfile;
use crate::streaming::SseState;
use crate::{PROVIDER_NAME, decode};

const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic's request id header (`req_...`).
const REQUEST_ID_HEADER: &str = "request-id";

fn header(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .map(|value| String::from_utf8_lossy(value.as_bytes()).into_owned())
}

/// The index of the assistant message the caller sent last, a prefill.
fn prefill(request: &ChatRequest) -> Option<usize> {
    matches!(request.messages.last(), Some(Message::Assistant(_)))
        .then(|| request.messages.len() - 1)
}

impl Provider {
    /// Everything before sending, shared by the call and its dry run:
    /// encode the prepared request (reporting through `prepared`), and
    /// build the HTTP request with the caller's transport options.
    fn encode(&self, prepared: &mut Prepared<'_>, stream: bool) -> Result<HttpRequest> {
        if stream {
            prepared.preflight_stream(self);
        }
        let (request, degradations) = prepared.parts_mut();
        let profile = self.profile(&request.model);
        let encoded = encode_request_for(request, stream, profile)?;
        // Into the same helper, so warnings name the caller's blocks.
        degradations.extend(encoded.warnings);
        let body = serde_json::to_vec(&encoded.body)?;

        let uri = request_uri(format!("{}/v1/messages", self.inner.base_url))
            .map_err(|err| err.with_provider(PROVIDER_NAME))?;
        let mut http = HttpRequest::post(uri, body);
        if let Some(api_key) = &self.inner.api_key {
            http = http.header(crate::API_KEY_HEADER, api_key.clone());
        }
        http = http
            .header(
                "anthropic-version",
                HeaderValue::from_static(ANTHROPIC_VERSION),
            )
            .header(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if !encoded.betas.is_empty() {
            let betas = HeaderValue::try_from(encoded.betas.join(",")).map_err(|_| {
                allama::Error::invalid_request(
                    "an `anthropic-beta` flag holds characters an HTTP header cannot carry",
                )
                .authored()
                .with_provider(PROVIDER_NAME)
            })?;
            http = http.header("anthropic-beta", betas);
        }
        http.with_headers(&self.inner.headers)
            .apply_options(&request.options, &self.inner.credential_slot)
            .map_err(|err| err.with_provider(PROVIDER_NAME))
    }

    /// The model's profile with the caller's capability overrides and
    /// model rules, so an override reaches the encoder as well as
    /// preflight.
    fn profile(&self, model: &str) -> ModelProfile {
        let mut profile = ModelProfile::for_model(model);
        if let Some(resolver) = &self.inner.chat_capability_resolver {
            profile =
                profile.with_overrides(|capability| resolver.chat_capability(model, capability));
        }
        if let Some(rules) = self
            .inner
            .model_rules
            .as_ref()
            .and_then(|rules| rules(model))
        {
            profile = profile.with_rules(&rules);
        }
        profile
    }

    /// Send an encoded request; map non-success statuses, naming the
    /// caller's `prefill` in a refusal of it. Streams get no total timeout,
    /// but setup and idle bounds.
    async fn send(
        &self,
        request: HttpRequest,
        stream: bool,
        prefill: Option<usize>,
    ) -> Result<(HttpResponse, Option<String>)> {
        let http = &*self.inner.http;
        let sent = if stream {
            self.inner.stream_timeouts.send(http, request).await
        } else {
            http.send(request.timeout(self.inner.request_timeout)).await
        };
        let response = sent.map_err(map_transport_error)?;
        let request_id = header(&response.headers, REQUEST_ID_HEADER);
        if !response.is_success() {
            let error = response
                .map_error(|status, headers, body| map_http_error(status, headers, body, prefill))
                .await;
            return Err(error.with_provider(PROVIDER_NAME));
        }
        Ok((response, request_id))
    }

    /// A dry run of `chat` or `chat_stream`.
    fn validate(&self, request: &ChatRequest, stream: bool) -> Result<ChatValidation> {
        if !stream {
            request.resolve_option::<crate::ChatRequestOptions>()?;
        }
        let mut prepared = allama::adapter::chat::prepare_dry_run(self, request)?;
        self.encode(&mut prepared, stream)?;
        Ok(prepared.into_validation(ValidationCoverage::Encoding))
    }
}

impl ChatProvider for Provider {
    type Stream = ChatStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let keep_raw = request
            .resolve_option::<crate::ChatRequestOptions>()?
            .raw_response;
        let mut prepared = allama::adapter::chat::prepare(self, request)?;
        let http = self.encode(&mut prepared, false)?;
        prepared.mark_sent();
        let (sent, request_id) = self.send(http, false, prefill(request)).await?;
        let status = sent.status;
        let bytes = sent
            .bytes_limited(self.inner.max_response_size)
            .await
            .map_err(|e| map_body_read_error(e, request_id.clone()))?;
        let wire = serde_json::from_slice(&bytes)
            .map_err(|e| map_response_decode_error(status, &bytes, e, request_id.clone()))?;
        let mut response = decode::decode_response(wire, request_id.clone())?;
        if keep_raw {
            let raw = serde_json::from_slice(&bytes)
                .map_err(|e| map_response_decode_error(status, &bytes, e, request_id))?;
            response.provider_data.insert(&crate::RawResponse(raw))?;
        }
        Ok(prepared.finish(response))
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        let mut prepared = allama::adapter::chat::prepare(self, request)?;
        let http = self.encode(&mut prepared, true)?;
        prepared.mark_sent();
        let (sent, request_id) = self.send(http, true, prefill(request)).await?;
        let state = SseState::new(request_id, Vec::new());
        let events =
            SseStream::new(sent.into_body()).max_event_size(self.inner.max_stream_event_size);
        Ok(prepared.finish_stream(SseChatStream::new(events, state)))
    }

    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        self.inner
            .chat_capability_resolver
            .as_ref()
            .and_then(|resolver| resolver.chat_capability(model, capability))
            .unwrap_or_else(|| crate::builtin_chat_capability(model, capability))
    }

    /// A dry run of `chat`: the preflight, options, encoding and transport
    /// options, without sending ([`ValidationCoverage::Encoding`]).
    fn validate_chat_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        self.validate(request, false)
    }

    /// A dry run of `chat_stream` ([`ValidationCoverage::Encoding`]).
    fn validate_chat_stream_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        self.validate(request, true)
    }
}

impl ProviderIdentity for Provider {
    fn provider_name(&self) -> &str {
        PROVIDER_NAME
    }

    fn reads_namespace(&self, namespace: &str) -> bool {
        namespace == crate::NAMESPACE
    }

    fn downcast_any(&self, id: std::any::TypeId) -> Option<&(dyn std::any::Any + Send + Sync)> {
        allama::expose(self, id)
    }
}
