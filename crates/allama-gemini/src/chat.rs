use allama::adapter::chat::Prepared;
use allama::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, ChatStream,
    ChatValidation, ProviderIdentity, RequestOptions, Result, ValidationCoverage,
};
use allama_http::http::HeaderValue;
use allama_http::http::header::CONTENT_TYPE;
use allama_http::sse::SseStream;
use allama_http::{HttpRequest, HttpResponse, request_uri};

use crate::decode::decode_response;
use crate::error::{SuccessBody, map_http_error, map_transport_error, name_refused_penalty};
use crate::model::{ModelProfile, resource_path};
use crate::request::encode_request;
use crate::streaming::sse_to_stream;
use crate::wire::GenerateContentResponse;
use crate::{PROVIDER_NAME, Provider};

impl Provider {
    /// Everything before sending, shared by the call and its dry run:
    /// encode the prepared request (reporting through `prepared`), and
    /// build the request to `generateContent` or
    /// `streamGenerateContent?alt=sse` with the caller's transport options.
    fn encode(&self, prepared: &mut Prepared<'_>, stream: bool) -> Result<HttpRequest> {
        if stream {
            prepared.preflight_stream(self);
        }
        let (request, degradations) = prepared.parts_mut();
        let encoded = encode_request(request, self.profile(&request.model))?;
        // Into the same helper, so warnings name the caller's blocks.
        degradations.extend(encoded.warnings);
        let model = resource_path(&request.model)?;
        let url = if stream {
            format!(
                "{}/{model}:streamGenerateContent?alt=sse",
                self.inner.base_url
            )
        } else {
            format!("{}/{model}:generateContent", self.inner.base_url)
        };
        let body = serde_json::to_string(&encoded.body)?;
        self.build_post(url, body, &request.options)
    }

    /// What the encoder knows about `model`: the wire formats it takes.
    ///
    /// The encoder gets it from here, never from the id itself, so this is
    /// the one place a model-rules hook will reach. It refuses or drops
    /// nothing: the capability answers, a resolver's included, act through
    /// preflight alone.
    fn profile(&self, model: &str) -> ModelProfile {
        ModelProfile::for_model(model)
    }

    /// A JSON POST with the API key and the caller's transport `options`.
    pub(crate) fn build_post(
        &self,
        url: String,
        body: String,
        options: &RequestOptions,
    ) -> Result<HttpRequest> {
        let uri = request_uri(url).map_err(|err| err.with_provider(PROVIDER_NAME))?;
        let mut request = HttpRequest::post(uri, body);
        if let Some(api_key) = &self.inner.api_key {
            request = request.header(crate::API_KEY_HEADER, api_key.clone());
        }
        request
            .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
            .with_headers(&self.inner.headers)
            .apply_options(options, &self.inner.credential_slot)
            .map_err(|err| err.with_provider(PROVIDER_NAME))
    }

    /// Send a request; map transport failures and error statuses. Streams
    /// get no total timeout, but setup and idle bounds.
    pub(crate) async fn send(&self, request: HttpRequest, stream: bool) -> Result<HttpResponse> {
        let http = &*self.inner.http;
        let sent = if stream {
            self.inner.stream_timeouts.send(http, request).await
        } else {
            http.send(request.timeout(self.inner.request_timeout)).await
        };
        let response = sent.map_err(map_transport_error)?;
        if !response.is_success() {
            let error = response.map_error(map_http_error).await;
            return Err(error.with_provider(PROVIDER_NAME));
        }
        Ok(response)
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
        let response = self
            .send(http, false)
            .await
            .map_err(|err| name_refused_penalty(err, request))?;
        let body = SuccessBody::read(response, self.inner.max_response_size).await?;
        let mut decoded = decode_response(
            body.decode::<GenerateContentResponse>()?,
            Vec::new(),
            body.request_id().map(str::to_owned),
        )?;
        if keep_raw {
            let raw = body.decode()?;
            decoded.provider_data.insert(&crate::RawResponse(raw))?;
        }
        Ok(prepared.finish(decoded))
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        let mut prepared = allama::adapter::chat::prepare(self, request)?;
        let http = self.encode(&mut prepared, true)?;
        prepared.mark_sent();
        let response = self
            .send(http, true)
            .await
            .map_err(|err| name_refused_penalty(err, request))?;
        let request_id = allama_http::error_context(response.status, &response.headers).request_id;
        let events =
            SseStream::new(response.into_body()).max_event_size(self.inner.max_stream_event_size);
        Ok(prepared.finish_stream(sse_to_stream(events, Vec::new(), request_id)))
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
