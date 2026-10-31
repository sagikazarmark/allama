use allama::adapter::chat::Prepared;
use allama::adapter::embedding::Prepared as EmbeddingPrepared;
use allama::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, ChatStream,
    ChatValidation, EmbeddingCapability, EmbeddingLimits, EmbeddingProvider, EmbeddingRequest,
    EmbeddingResponse, EmbeddingValidation, ProviderIdentity, RequestOptions, Result,
    ValidationCoverage,
};

use allama_http::http::HeaderValue;
use allama_http::http::header::CONTENT_TYPE;
use allama_http::sse::{SseChatStream, SseStream};
use allama_http::{HttpRequest, HttpResponse, request_uri};

use crate::embedding::{
    EmbeddingWireOptions, EmbeddingsResponse, from_embeddings_response, to_embeddings_request,
};
use crate::error::{decode_body, read_body};
use crate::options::{ChatRequestOptions, EmbeddingRequestOptions, WireOptions};
use crate::wire::{ChatCompletionResponse, ResponseContext};
use crate::{Provider, SseState, extract_request_id, extract_retry_after, from_api_response};

impl Provider {
    /// The verbatim response body kept when the request set `raw_response`
    /// (it lives under this provider's namespace).
    #[must_use]
    pub fn raw_response<'a>(&self, response: &'a ChatResponse) -> Option<&'a serde_json::Value> {
        response.provider_data.get_raw(
            &self.inner.quirks.namespace,
            allama::provider_data::RAW_RESPONSE_KEY,
        )
    }

    /// Encode a prepared chat request into the HTTP request this provider
    /// sends: everything before sending, so a dry run of it is a validation
    /// with [`ValidationCoverage::Encoding`].
    ///
    /// It maps the request to the Chat Completions wire format under this
    /// provider's quirks, reporting what it degrades through `prepared`,
    /// and applies the caller's transport options (a missing scoped
    /// credential is an [`Error::Configuration`](allama::Error::Configuration)). It does not run a
    /// preflight: build `prepared` with [`allama::adapter::chat::prepare`] against
    /// the provider whose answers apply, and add this provider's own with
    /// [`Prepared::preflight_with`] when they differ.
    ///
    /// ```no_run
    /// use allama::{ChatRequest, ChatValidation, Result, ValidationCoverage};
    /// use allama_openai_compat::{Provider, WireOptions};
    ///
    /// fn validate(provider: &Provider, request: &ChatRequest) -> Result<ChatValidation> {
    ///     let mut prepared = allama::adapter::chat::prepare_dry_run(provider, request)?;
    ///     provider.encode_chat(&mut prepared, &WireOptions::default(), false)?;
    ///     Ok(prepared.into_validation(ValidationCoverage::Encoding))
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Request mapping errors (including refusals under the request's
    /// policy) and transport option errors.
    #[doc(hidden)]
    pub fn encode_chat(
        &self,
        prepared: &mut Prepared<'_>,
        options: &WireOptions,
        stream: bool,
    ) -> Result<HttpRequest> {
        let (request, degradations) = prepared.parts_mut();
        let rules = self.model_rules(&request.model);
        let (body, mapping_warnings) = crate::wire::encode_chat_completion(
            request,
            stream,
            options,
            &self.inner.quirks,
            &rules,
            self.inner.provider_name.clone(),
        )?;
        // Into the same helper, so warnings name the caller's blocks.
        degradations.extend(mapping_warnings);
        let url = self.inner.transport.chat_completions_url();
        self.build_post(&url, serde_json::to_string(&body)?, &request.options)
    }

    /// Send a request [`encode_chat`](Self::encode_chat) built (not
    /// streaming), decode the response, and join `prepared`'s warnings to
    /// it ([`Prepared::finish`]).
    ///
    /// # Errors
    ///
    /// Transport, HTTP, and decoding errors.
    #[doc(hidden)]
    pub async fn send_chat(
        &self,
        request: HttpRequest,
        mut prepared: Prepared<'_>,
        options: &WireOptions,
    ) -> Result<ChatResponse> {
        prepared.mark_sent();
        let response = self.send(request, false).await?;
        let context = self.response_context(&response);
        let namespace = context.namespace.clone();
        let (provider, request_id) = (self.provider_name(), context.request_id.clone());
        let (status, bytes) = read_body(
            response,
            self.inner.transport.max_response_size,
            provider,
            request_id.as_deref(),
        )
        .await?;
        // An error reported with a success status (OpenRouter) is that
        // error, classified as in-band errors are.
        if let Some(error) = crate::error::success_body_error(&bytes) {
            return Err(crate::error::classify_api_error(
                &error,
                None,
                request_id,
                None,
                &self.inner.quirks.quota_error_codes,
            )
            .with_provider(provider));
        }
        let body: ChatCompletionResponse =
            decode_body(status, &bytes, provider, request_id.as_deref())?;
        let mut decoded = from_api_response(body, context)?;
        if options.raw_response {
            let raw = decode_body(status, &bytes, provider, request_id.as_deref())?;
            decoded.provider_data.insert_raw(
                namespace,
                allama::provider_data::RAW_RESPONSE_KEY,
                raw,
            );
        }
        Ok(prepared.finish(decoded))
    }

    /// Send a streaming request [`encode_chat`](Self::encode_chat) built,
    /// and join `prepared`'s warnings to the stream
    /// ([`Prepared::finish_stream`]).
    ///
    /// # Errors
    ///
    /// Transport and HTTP errors.
    #[doc(hidden)]
    pub async fn send_chat_stream(
        &self,
        request: HttpRequest,
        mut prepared: Prepared<'_>,
    ) -> Result<ChatStream> {
        prepared.mark_sent();
        let response = self.send(request, true).await?;
        let context = self.response_context(&response);
        let events = SseStream::new(response.into_body())
            .max_event_size(self.inner.transport.max_stream_event_size);
        let state = SseState::new(context, &self.inner.quirks);
        Ok(prepared.finish_stream(SseChatStream::new(events, state)))
    }

    /// Embed with explicit wire options.
    ///
    /// The request is first checked against this provider's own answers
    /// ([`allama::adapter::embedding::prepare`]), then encoded
    /// ([`encode_embedding`](Self::encode_embedding)) and sent
    /// ([`send_embedding`](Self::send_embedding)). A provider crate that
    /// checks the request against its own answers too calls those two
    /// itself.
    ///
    /// # Errors
    ///
    /// Request mapping (including the pre-flight checks), transport, HTTP,
    /// and decoding errors.
    pub(crate) async fn embed_with_options(
        &self,
        request: &EmbeddingRequest,
        options: &EmbeddingWireOptions,
    ) -> Result<EmbeddingResponse> {
        let mut prepared = allama::adapter::embedding::prepare(self, request)?;
        let http = self.encode_own_embedding(&mut prepared, options)?;
        self.send_embedding(http, prepared).await
    }

    /// Encode an embedding request for this provider's own call or dry run:
    /// options under its name are reported.
    fn encode_own_embedding(
        &self,
        prepared: &mut EmbeddingPrepared<'_>,
        options: &EmbeddingWireOptions,
    ) -> Result<HttpRequest> {
        let (request, degradations) = prepared.parts_mut();
        crate::options::report_options_under_name(
            &request.options,
            self.provider_name(),
            crate::NAMESPACE,
            &[],
            "EmbeddingRequestOptions",
            degradations,
        );
        self.encode_embedding(prepared, options)
    }

    /// Everything before sending an embedding request, shared by the call
    /// and its dry run: encode the prepared request (reporting through
    /// `prepared`) and build the request to the embeddings URL with the
    /// caller's transport options.
    ///
    /// It runs no capability check of its own: a provider crate built on
    /// this toolkit runs [`allama::adapter::embedding::prepare`] with its own answers
    /// and, when the toolkit's answers should apply too,
    /// [`Prepared::preflight_with`](allama::adapter::embedding::Prepared::preflight_with)
    /// with this provider before calling it.
    ///
    /// # Errors
    ///
    /// Request mapping errors (see [`to_embeddings_request`]) and transport
    /// option errors.
    #[doc(hidden)]
    pub fn encode_embedding(
        &self,
        prepared: &mut EmbeddingPrepared<'_>,
        options: &EmbeddingWireOptions,
    ) -> Result<HttpRequest> {
        let (request, degradations) = prepared.parts_mut();
        // OpenAI-compatible embeddings read no per-input settings.
        request.check_provider_data(&self.inner.provider_name, &self.quirks().namespace, &[])?;
        let (body, mapping_warnings) = to_embeddings_request(
            request,
            options,
            self.quirks(),
            self.inner.provider_name.clone(),
        )?;
        // Into the same helper, so warnings name the caller's inputs.
        degradations.extend(mapping_warnings);
        let url = self.inner.transport.embeddings_url();
        self.build_post(&url, serde_json::to_string(&body)?, &request.options)
    }

    /// Send an embedding request [`encode_embedding`](Self::encode_embedding)
    /// built, decode the response, and join `prepared`'s warnings to it
    /// ([`Prepared::finish`](allama::adapter::embedding::Prepared::finish)).
    ///
    /// # Errors
    ///
    /// Transport, HTTP, and decoding errors, and a response that does not
    /// hold one usable vector per input
    /// ([`allama::adapter::embedding::check_response`]).
    #[doc(hidden)]
    pub async fn send_embedding(
        &self,
        request: HttpRequest,
        mut prepared: EmbeddingPrepared<'_>,
    ) -> Result<EmbeddingResponse> {
        prepared.mark_sent();
        let response = self.send(request, false).await?;
        let provider = self.provider_name();
        let request_id = self.response_context(&response).request_id;
        let (status, bytes) = read_body(
            response,
            self.inner.transport.max_response_size,
            provider,
            request_id.as_deref(),
        )
        .await?;
        let decoded: EmbeddingsResponse =
            decode_body(status, &bytes, provider, request_id.as_deref())?;
        let request = prepared.request();
        let response = from_embeddings_response(decoded, request.inputs.len(), Vec::new())
            .and_then(|response| {
                allama::adapter::embedding::check_response(request, &response)?;
                Ok(response)
            })
            .map_err(|err| {
                err.with_provider(self.inner.provider_name.clone())
                    .with_request_id_opt(request_id.as_deref())
            })?;
        Ok(prepared.finish(response))
    }

    /// Who a response is attributed to.
    fn response_context(&self, response: &HttpResponse) -> ResponseContext {
        ResponseContext {
            provider: self.inner.provider_name.clone(),
            namespace: self.inner.quirks.namespace.clone(),
            request_id: extract_request_id(
                &response.headers,
                &self.inner.transport.request_id_header_name,
            ),
            warnings: Vec::new(),
        }
    }

    /// A JSON POST with the configured headers and the caller's transport
    /// `options` applied.
    fn build_post(&self, url: &str, body: String, options: &RequestOptions) -> Result<HttpRequest> {
        let config = &self.inner.transport;
        let uri = request_uri(url).map_err(|err| err.with_provider(self.provider_name()))?;
        let mut request = HttpRequest::post(uri, body)
            .header(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Some((name, value)) = &config.auth {
            request = request.header(name.clone(), value.clone());
        }
        request
            .with_headers(&config.headers)
            .apply_options(options, &self.inner.credential_slot)
            .map_err(|err| err.with_provider(self.provider_name()))
    }

    /// Send a request; streams get no total timeout, but setup and idle
    /// bounds. Map error statuses.
    async fn send(&self, request: HttpRequest, stream: bool) -> Result<HttpResponse> {
        let config = &self.inner.transport;
        let http = &*self.inner.http;
        let sent = if stream {
            config.stream_timeouts.send(http, request).await
        } else {
            http.send(request.timeout(config.timeout)).await
        };
        let response = sent.map_err(|e| e.into_error().with_provider(self.provider_name()))?;
        if response.is_success() {
            return Ok(response);
        }
        let error = response
            .map_error(|status, headers, body| {
                // The configured headers take precedence over the common ones.
                self.inner
                    .quirks
                    .map_http_error(status, headers, body)
                    .with_request_id_opt(
                        extract_request_id(headers, &config.request_id_header_name).as_deref(),
                    )
                    .with_retry_after_opt(extract_retry_after(
                        headers,
                        &config.retry_after_header_name,
                    ))
            })
            .await;
        Err(error.with_provider(self.provider_name()))
    }

    /// Encode for this provider's own call or dry run: its stream
    /// capabilities are recorded for a stream.
    fn encode_own(
        &self,
        prepared: &mut Prepared<'_>,
        options: &WireOptions,
        stream: bool,
    ) -> Result<HttpRequest> {
        if stream {
            prepared.preflight_stream(self);
        }
        let (request, degradations) = prepared.parts_mut();
        crate::options::report_options_under_name(
            &request.options,
            self.provider_name(),
            crate::NAMESPACE,
            &[],
            "ChatRequestOptions",
            degradations,
        );
        self.encode_chat(prepared, options, stream)
    }

    /// A dry run of `chat` or `chat_stream`.
    fn validate(&self, request: &ChatRequest, stream: bool) -> Result<ChatValidation> {
        let options: ChatRequestOptions = request.resolve_option()?;
        let mut prepared = allama::adapter::chat::prepare_dry_run(self, request)?;
        self.encode_own(&mut prepared, &options.into(), stream)?;
        Ok(prepared.into_validation(ValidationCoverage::Encoding))
    }

    /// Capability answers implied by the quirks, for capabilities the
    /// provider declares nothing about.
    fn implied_chat_capability(&self, capability: ChatCapability) -> CapabilitySupport {
        match capability {
            // Without `stream_options.include_usage` a Chat Completions
            // stream carries no usage chunk:
            // https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream_options
            // With it, whether the endpoint honors it is its own claim.
            ChatCapability::StreamUsage if !self.inner.quirks.stream_usage => {
                CapabilitySupport::Unsupported
            }
            // The encoder refuses a `json_schema` response format the
            // endpoint does not take, whatever the model (JSON mode is no
            // part of the answer).
            ChatCapability::StructuredOutput if !self.inner.quirks.supports_json_schema => {
                CapabilitySupport::Unsupported
            }
            _ => CapabilitySupport::Unknown,
        }
    }

    /// The wire options of an embedding request.
    fn embedding_wire_options(request: &EmbeddingRequest) -> Result<EmbeddingWireOptions> {
        let options: EmbeddingRequestOptions = request.resolve_option()?;
        Ok(EmbeddingWireOptions {
            user: options.user,
            extra: options.extra,
        })
    }
}

impl ChatProvider for Provider {
    type Stream = ChatStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let options: WireOptions = request.resolve_option::<ChatRequestOptions>()?.into();
        let mut prepared = allama::adapter::chat::prepare(self, request)?;
        let http = self.encode_own(&mut prepared, &options, false)?;
        self.send_chat(http, prepared, &options).await
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<ChatStream> {
        let options: WireOptions = request.resolve_option::<ChatRequestOptions>()?.into();
        let mut prepared = allama::adapter::chat::prepare(self, request)?;
        let http = self.encode_own(&mut prepared, &options, true)?;
        self.send_chat_stream(http, prepared).await
    }

    /// A dry run of `chat`: the preflight, option resolution, encoding and
    /// transport options, without sending ([`ValidationCoverage::Encoding`]).
    fn validate_chat_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        self.validate(request, false)
    }

    /// A dry run of `chat_stream`, as [`validate_chat_request`](Self::validate_chat_request).
    fn validate_chat_stream_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        self.validate(request, true)
    }

    fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
        self.inner
            .chat_capability_resolver
            .as_ref()
            .and_then(|resolver| resolver.chat_capability(model, capability))
            .or_else(|| match capability {
                ChatCapability::StructuredOutput => self.model_rules(model).structured_output,
                _ => None,
            })
            .unwrap_or_else(|| {
                self.inner
                    .chat_capabilities
                    .get(&capability)
                    .copied()
                    .unwrap_or_else(|| self.implied_chat_capability(capability))
            })
    }
}

impl EmbeddingProvider for Provider {
    async fn embed(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        let options = Self::embedding_wire_options(request)?;
        self.embed_with_options(request, &options).await
    }

    /// A dry run of `embed`: the preflight, options, encoding and transport
    /// options, without sending ([`ValidationCoverage::Encoding`]).
    fn validate_embedding_request(
        &self,
        request: &EmbeddingRequest,
    ) -> Result<EmbeddingValidation> {
        let options = Self::embedding_wire_options(request)?;
        let mut prepared = allama::adapter::embedding::prepare_dry_run(self, request)?;
        self.encode_own_embedding(&mut prepared, &options)?;
        Ok(prepared.into_validation(ValidationCoverage::Encoding))
    }

    /// What the builder declared, except that the OpenAI-compatible
    /// embeddings wire carries one text per input: image and composite
    /// inputs are always `Unsupported`. The input type is `Unsupported`
    /// without a [`CompatQuirks::embedding_input_type`](crate::CompatQuirks::embedding_input_type)
    /// field to send it on.
    fn embedding_capability(
        &self,
        _model: &str,
        capability: EmbeddingCapability,
    ) -> CapabilitySupport {
        match capability {
            EmbeddingCapability::ImageInput | EmbeddingCapability::CompositeInput => {
                CapabilitySupport::Unsupported
            }
            EmbeddingCapability::InputType if self.quirks().embedding_input_type.is_none() => {
                CapabilitySupport::Unsupported
            }
            _ => self
                .inner
                .embedding_capabilities
                .get(&capability)
                .copied()
                .unwrap_or_default(),
        }
    }

    fn embedding_limits(&self, _model: &str) -> EmbeddingLimits {
        let mut limits = EmbeddingLimits::new();
        limits.max_inputs_per_call = self.inner.max_embedding_inputs;
        limits
    }
}

impl ProviderIdentity for Provider {
    fn provider_name(&self) -> &str {
        &self.inner.provider_name
    }

    fn reads_namespace(&self, namespace: &str) -> bool {
        namespace == self.inner.quirks.namespace
    }

    fn downcast_any(&self, id: std::any::TypeId) -> Option<&(dyn std::any::Any + Send + Sync)> {
        allama::expose(self, id)
    }
}
