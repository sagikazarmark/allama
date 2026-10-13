//! The shortest complete native adapter: a text-only gateway, unary and
//! NDJSON streaming, that passes every contract of the conformance
//! aggregate (`assert_chat_adapter`).
//!
//! The (fictional) wire format: `POST {base}/generate` with
//! `{"model", "messages": [{"role", "content"}], "max_tokens"?, "stream"}`.
//! An answer is `{"id", "model", "text", "finish_reason", "usage"}`; a
//! stream is `{"delta": ".."}` lines, then `{"done": true, "finish_reason",
//! "usage"}`; an error is `{"error": {"type", "message"}}`, in-band in a
//! stream too. [`WireSupport`] declares what the wire format carries, and
//! `Prepared::degrade_to` refuses or drops the rest with the right
//! warnings, so the encoder only encodes text. Start here; `native_adapter`
//! and the provider author guide cover what a richer API needs.
//!
//! Run the demo (canned answers, no network) with
//! `cargo run -p allama-examples --example text_gateway`, and its contract
//! tests with `cargo test -p allama-examples --example text_gateway`.

use std::sync::Arc;

use allama::adapter::chat::{self, EncodedTurn, Prepared, WireSupport, retain_turns};
use allama::chat::stream::MetadataBuilder;
use allama::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, ChatStream,
    ChatStreamExt, ChatValidation, ContentBlock, Error, FinishReason, Message, ProviderIdentity,
    RequestField, Result, StreamBlockType, StreamErrorKind, StreamEvent, Subject, Usage,
    ValidationCoverage,
};
use allama_http::framing::{ChatFrameDecoder, FramedChatStream, VendorFrames};
use allama_http::ndjson::NdjsonStream;
use allama_http::{
    CredentialPlacement, CredentialSlot, DEFAULT_MAX_RESPONSE_SIZE, DEFAULT_REQUEST_TIMEOUT,
    Headers, HttpRequest, HttpResponse, HttpTransport, ReplayResponse, ReplayTransport,
    StreamTimeouts, TransportError, error_context, error_envelope, malformed_response,
    status_error,
};
use serde::Deserialize;
use serde_json::{Value, json};

const PROVIDER_NAME: &str = "gateway";
const NAMESPACE: &str = "gateway";

/// Role-tagged text, system prompts and an output cap: everything else is
/// refused or dropped, with a warning, and answered `Unsupported`.
const WIRE: WireSupport =
    WireSupport::text().fields([RequestField::System, RequestField::MaxOutputTokens]);

#[derive(Clone)]
pub struct Gateway {
    http: Arc<dyn HttpTransport>,
    url: String,
    token: String,
    slot: CredentialSlot,
}

impl std::fmt::Debug for Gateway {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gateway")
            .field("url", &self.url)
            .field("token", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl Gateway {
    pub fn new(base_url: &str, token: &str, http: impl HttpTransport) -> Self {
        let url = format!("{}/generate", base_url.trim_end_matches('/'));
        let slot = CredentialSlot::new(PROVIDER_NAME, CredentialPlacement::Bearer);
        let (http, token) = (Arc::new(http), token.to_owned());
        Self {
            http,
            url,
            token,
            slot,
        }
    }

    /// Everything before sending, so validation (a dry run) runs it too.
    fn encode(&self, prepared: &mut Prepared<'_>, stream: bool) -> Result<HttpRequest> {
        if stream {
            prepared.preflight_stream(self);
        }
        prepared.degrade_to(self, &WIRE)?;
        let (request, d) = prepared.parts_mut();
        let mut turns = Vec::new();
        for (index, message) in request.messages.iter().enumerate() {
            let text = match message {
                Message::User(user) => user.content.text().unwrap_or_default(),
                // What `degrade_to` left of an assistant turn: text and refusals.
                Message::Assistant(assistant) => {
                    let mut text = String::new();
                    for (block, content) in assistant.content.iter().enumerate() {
                        match content {
                            ContentBlock::Text(t) => text.push_str(&t.text),
                            ContentBlock::Refusal(r) => text.push_str(&r.text),
                            _ => {
                                return Err(d.reject(at(index, Some(block)), "unsupported content"));
                            }
                        }
                    }
                    text
                }
                // `degrade_to` refused tool turns; `Message` is non-exhaustive.
                _ => return Err(d.reject(at(index, None), "unsupported message")),
            };
            let turn = EncodedTurn::new(index, message, text.is_empty());
            turns.push((turn, json!({"role": message.role(), "content": text})));
        }
        let system = request.system.iter();
        let system = system.map(|s| json!({"role": "system", "content": s.content}));
        let turns = retain_turns(turns, d, |(turn, _)| *turn)?;
        let messages: Vec<Value> = system.chain(turns.into_iter().map(|(_, m)| m)).collect();
        let mut body = json!({"model": request.model, "messages": messages, "stream": stream});
        if let Some(max) = request.max_output_tokens {
            body["max_tokens"] = json!(max);
        }
        HttpRequest::post(&self.url, body.to_string())
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {}", self.token))
            .apply_options(&request.options, &self.slot)
    }

    async fn send(&self, request: HttpRequest, stream: bool) -> Result<HttpResponse> {
        let sent = match stream {
            true => StreamTimeouts::default().send(&*self.http, request).await,
            // A bound on the whole call: no transport adds one of its own.
            false => {
                let request = request.timeout(Some(DEFAULT_REQUEST_TIMEOUT));
                self.http.send(request).await
            }
        };
        let response = sent.map_err(TransportError::into_error)?;
        if response.is_success() {
            return Ok(response);
        }
        Err(response.map_error(map_http_error).await)
    }

    fn validate(&self, request: &ChatRequest, stream: bool) -> Result<ChatValidation> {
        let mut prepared = chat::prepare_dry_run(self, request)?;
        self.encode(&mut prepared, stream)?;
        Ok(prepared.into_validation(ValidationCoverage::Encoding))
    }
}

impl ProviderIdentity for Gateway {
    fn provider_name(&self) -> &str {
        PROVIDER_NAME
    }

    fn reads_namespace(&self, namespace: &str) -> bool {
        namespace == NAMESPACE
    }
}

impl ChatProvider for Gateway {
    type Stream = ChatStream;

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let mut prepared = chat::prepare(self, request)?;
        let http = self.encode(&mut prepared, false)?;
        prepared.mark_sent();
        let response = self.send(http, false).await?;
        let (status, request_id) = (response.status, request_id(&response));
        let body = response
            .text_limited(DEFAULT_MAX_RESPONSE_SIZE)
            .await
            .map_err(TransportError::into_error)?;
        let answer: Answer = serde_json::from_str(&body).map_err(|error| {
            let error = malformed_response(status, body.as_bytes(), error);
            error
                .with_provider(PROVIDER_NAME)
                .with_request_id_opt(request_id.as_deref())
        })?;
        let mut response = ChatResponse::new(vec![ContentBlock::text(answer.text)]);
        (response.finish_reason, response.usage) =
            (finish(&answer.finish_reason), usage(answer.usage));
        (response.finish_reason_raw, response.id) = (answer.finish_reason, answer.id);
        (response.model, response.request_id) = (answer.model, request_id);
        Ok(prepared.finish(response.provider(PROVIDER_NAME)))
    }

    async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
        let mut prepared = chat::prepare(self, request)?;
        let http = self.encode(&mut prepared, true)?;
        prepared.mark_sent();
        let response = self.send(http, true).await?;
        let frames = VendorFrames::new(PROVIDER_NAME, NAMESPACE);
        let lines = Lines {
            request_id: request_id(&response),
            open: false,
            done: false,
            frames,
        };
        let body = NdjsonStream::new(response.into_body());
        Ok(prepared.finish_stream(FramedChatStream::new(body, lines)))
    }

    fn chat_capability(&self, _model: &str, capability: ChatCapability) -> CapabilitySupport {
        WIRE.chat_capability(capability)
            .unwrap_or(match capability {
                ChatCapability::ImageOutput | ChatCapability::ReasoningOutput => {
                    CapabilitySupport::Unsupported
                }
                _ => CapabilitySupport::Supported, // streaming, with usage
            })
    }

    fn validate_chat_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        self.validate(request, false)
    }

    fn validate_chat_stream_request(&self, request: &ChatRequest) -> Result<ChatValidation> {
        self.validate(request, true)
    }
}

fn at(message: usize, block: Option<usize>) -> Subject {
    Subject::Content { message, block }
}

fn request_id(response: &HttpResponse) -> Option<String> {
    error_context(response.status, &response.headers).request_id
}

#[derive(Deserialize)]
struct Answer {
    id: Option<String>,
    model: Option<String>,
    text: String,
    finish_reason: Option<String>,
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
struct WireUsage {
    input: Option<u64>,
    output: Option<u64>,
}

fn usage(wire: Option<WireUsage>) -> Option<Usage> {
    let wire = wire?;
    let mut usage = Usage::new();
    (usage.input_tokens, usage.output_tokens) = (wire.input, wire.output);
    Some(usage.with_derived_total())
}

fn finish(raw: &Option<String>) -> Option<FinishReason> {
    Some(match raw.as_deref()? {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        other => FinishReason::Other(other.to_owned()),
    })
}

/// The NDJSON decoder. The lines carry no type: a line that is not a
/// delta, the end or an error is a vendor frame, surfaced as provider data.
struct Lines {
    request_id: Option<String>,
    open: bool,
    done: bool,
    frames: VendorFrames,
}

impl ChatFrameDecoder<String> for Lines {
    fn frame(&mut self, line: String) -> Vec<Result<StreamEvent>> {
        let json: Value = match serde_json::from_str(&line) {
            Ok(json) => json,
            Err(e) => {
                let error = Error::stream_decode("invalid JSON line").authored();
                return self.fail(error.with_upstream_message(e.to_string()));
            }
        };
        if json.get("error").is_some() {
            return self.fail(map_error(None, &line));
        }
        let (delta, end) = (json["delta"].as_str(), json["done"] == true);
        let mut out = Vec::new();
        if !self.open && (delta.is_some() || end) {
            self.open = true;
            out.push(StreamEvent::response_start(None, None));
            out.push(StreamEvent::block_start(0, StreamBlockType::Text));
        }
        if let Some(delta) = delta {
            out.push(StreamEvent::text_delta(0, delta));
        } else if end {
            self.done = true;
            let raw = json["finish_reason"].as_str().map(str::to_owned);
            let metadata = MetadataBuilder::new()
                .with_finish_reason_opt(finish(&raw))
                .with_finish_reason_raw_opt(raw.as_deref())
                .with_usage_opt(usage(WireUsage::deserialize(&json["usage"]).ok()))
                .with_request_id_opt(self.request_id.as_deref());
            out.extend([StreamEvent::block_stop(0), metadata.build()]);
            // The whole array of a repeated vendor frame type, before the
            // stop (`VendorFrames::finish`).
            out.extend(self.frames.finish());
            out.push(StreamEvent::ResponseStop);
        } else {
            out.extend(self.frames.surface_payload("line", json));
        }
        out.into_iter().map(Ok).collect()
    }

    fn end(&mut self) -> Vec<Result<StreamEvent>> {
        let message = "the stream ended before `done`";
        self.fail(Error::stream(StreamErrorKind::Incomplete, message).authored())
    }

    fn fail(&mut self, error: Error) -> Vec<Result<StreamEvent>> {
        self.done = true;
        let error = error.with_provider(PROVIDER_NAME);
        vec![Err(error.with_request_id_opt(self.request_id.as_deref()))]
    }

    fn is_done(&self) -> bool {
        self.done
    }
}

/// The HTTP error mapping: a pure function, which the kit checks alone.
pub fn map_http_error(status: u16, headers: &Headers, body: &str) -> Error {
    let error = map_error(Some(status), body).with_context(error_context(status, headers));
    error.with_body_opt((!body.is_empty()).then_some(body))
}

/// An error envelope, from a response (`status`) or a stream line (`None`).
fn map_error(status: Option<u16>, body: &str) -> Error {
    let envelope = error_envelope(body);
    let message = status.map_or("the stream failed".to_owned(), |s| format!("HTTP {s}"));
    let error = match (
        envelope.as_ref().and_then(|e| e.error_type.as_deref()),
        status,
    ) {
        (Some("rate_limited"), _) => Error::rate_limited(message),
        (Some("bad_request"), _) => Error::invalid_request(message),
        (_, Some(status)) => status_error(status, message),
        (_, None) => Error::provider(message),
    };
    let error = error.authored().with_provider(PROVIDER_NAME);
    match envelope {
        Some(envelope) => envelope.apply(error),
        None => error,
    }
}

// ── Not part of the adapter: canned answers for the demo and the tests ──

const ANSWER: &str =
    r#"{"id":"r1","text":"Hello!","finish_reason":"stop","usage":{"input":5,"output":2}}"#;
const STREAM: &str = "{\"delta\":\"Hel\"}\n{\"delta\":\"lo!\"}\n{\"done\":true,\"finish_reason\":\"stop\",\"usage\":{\"input\":5,\"output\":2}}\n";

fn answer(stream: bool) -> ReplayTransport {
    ReplayTransport::new([match stream {
        false => ReplayResponse::json(200, ANSWER),
        true => ReplayResponse::ndjson(STREAM),
    }])
}

#[tokio::main]
async fn main() -> Result<()> {
    // `temperature` has no slot on this wire: it is dropped, with a warning.
    let request = ChatRequest::new("small").user("Say hello").temperature(0.2);
    let gateway = Gateway::new("https://gateway.test/v1", "key", answer(false));
    let response = gateway.chat(&request).await?;
    println!("{}; {}", response.text_or_empty(), response.warnings[0]);
    let gateway = Gateway::new("https://gateway.test/v1", "key", answer(true));
    let streamed = gateway
        .chat_stream(&request)
        .await?
        .collect_response()
        .await?;
    println!(
        "streamed: {}; {:?}",
        streamed.text_or_empty(),
        streamed.usage
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use allama::RequestOptions;
    use allama_conformance::probe::ProbeOutcome;
    use allama_conformance::stream_end::StreamEnds;
    use allama_conformance::stream_errors::ErrorKindProbe;
    use allama_conformance::stream_frames::StreamFraming;
    use allama_conformance::{
        AdapterUnderTest, ResponseCapture, TransportOutcome, assert_chat_adapter,
    };

    use super::*;

    fn gateway(transport: &ReplayTransport) -> Gateway {
        Gateway::new("https://gateway.test/v1", "key", transport.clone())
    }

    /// `chat`, or `chat_stream` collected, over `transport`.
    async fn call(
        transport: &ReplayTransport,
        request: &ChatRequest,
        stream: bool,
    ) -> Result<ChatResponse> {
        match stream {
            false => gateway(transport).chat(request).await,
            true => {
                gateway(transport)
                    .chat_stream(request)
                    .await?
                    .collect_response()
                    .await
            }
        }
    }

    async fn probe(request: ChatRequest, stream: bool) -> ProbeOutcome<ChatResponse> {
        let transport = answer(stream);
        let result = call(&transport, &request, stream).await;
        let bodies = transport
            .requests()
            .into_iter()
            .map(|r| serde_json::from_slice(&r.body.unwrap()).unwrap());
        ProbeOutcome::new(result, bodies.collect())
    }

    async fn send(options: RequestOptions, stream: bool) -> TransportOutcome {
        let (transport, mut request) = (answer(stream), ChatRequest::new("small").user("Hi"));
        request.options = options;
        let result = call(&transport, &request, stream).await.map(drop);
        TransportOutcome::new(result, transport.requests())
    }

    fn envelope(kind: ErrorKindProbe, upstream: &str) -> Option<String> {
        let kind = match kind {
            ErrorKindProbe::RateLimited => "rate_limited",
            ErrorKindProbe::InvalidRequest => "bad_request",
            _ => return None,
        };
        Some(json!({"error": {"type": kind, "message": format!("failed {upstream}")}}).to_string())
    }

    #[tokio::test]
    async fn the_gateway_passes_every_contract() {
        let request = || ChatRequest::new("small").user("Hi");
        let replay = move |body: String| async move {
            gateway(&ReplayTransport::new([ReplayResponse::ndjson(body)]))
                .chat_stream(&request())
                .await
        };
        let malformed = move |body: String| async move {
            let answer = ReplayResponse::json(200, body).header("x-request-id", "req_1");
            gateway(&ReplayTransport::new([answer]))
                .chat(&request())
                .await
        };
        let unary = move |body: String| async move {
            gateway(&ReplayTransport::new([ReplayResponse::json(200, body)]))
                .chat(&request())
                .await
        };
        // The same answer cut off at `max_tokens`, as the API sends it.
        let cut_off = ResponseCapture::new(
            r#"{"id":"r2","text":"Hel","finish_reason":"length","usage":{"input":5,"output":1}}"#,
            "{\"delta\":\"Hel\"}\n{\"done\":true,\"finish_reason\":\"length\",\"usage\":{\"input\":5,\"output\":1}}\n",
        );
        // Through the real HTTP client, for the redirect contract.
        let redirected = move |base_url: String, stream: bool| async move {
            let gateway = Gateway::new(&base_url, "key", allama_http::ReqwestTransport::new());
            match stream {
                false => gateway.chat(&request()).await.map(drop),
                true => gateway.chat_stream(&request()).await.map(drop),
            }
        };
        let first = STREAM.lines().next().unwrap().to_owned();
        let cut_short: String = STREAM.split_inclusive('\n').take(2).collect();
        let without_finish = format!("{cut_short}{{\"done\":true}}\n");
        let slot = CredentialSlot::new(PROVIDER_NAME, CredentialPlacement::Bearer);
        // Untyped lines: a vendor frame is a line of fields the decoder does not know.
        let framing = StreamFraming::ndjson(|_, payload| payload.to_string());
        let under_test = AdapterUnderTest::new(
            ["small"],
            NAMESPACE,
            |r| probe(r, false),
            |r| probe(r, true),
        )
        .error_mapping(|status, headers, body| {
            map_http_error(status, &headers.iter().copied().collect(), &body)
        })
        .error_envelope(envelope)
        .stream_body(move |envelope| format!("{first}\n{envelope}\n"))
        .stream_replay(replay)
        .stream_ends(StreamEnds::new(cut_short).terminated_without_finish(without_finish))
        .stream_captures(framing, [STREAM])
        .transport_options(slot, |o| send(o, false), |o| send(o, true))
        .malformed_response(malformed, Some("req_1"))
        .unary_replay(unary)
        .response_captures([ResponseCapture::new(ANSWER, STREAM)])
        .length_capture(cut_off)
        .redirect_not_followed(redirected);
        assert_chat_adapter(&gateway(&ReplayTransport::new([])), &under_test).await;
    }

    /// A unary call carries the whole-call bound, so a server that accepts
    /// and never answers cannot hold it forever; a stream has its own bounds.
    #[tokio::test]
    async fn a_unary_call_is_bounded_by_the_request_timeout() {
        for stream in [false, true] {
            let transport = answer(stream);
            let request = ChatRequest::new("small").user("Hi");
            call(&transport, &request, stream).await.unwrap();
            let expected = (!stream).then_some(DEFAULT_REQUEST_TIMEOUT);
            assert_eq!(
                transport.requests()[0].timeout,
                expected,
                "stream: {stream}"
            );
        }
    }

    /// The provider guide's quickstart (its `rust,ignore` blocks) quotes
    /// this file, line for line (indentation aside).
    #[test]
    fn the_provider_guide_quickstart_quotes_this_file() {
        let guide = include_str!("../../../docs/guides/provider-authors.md");
        let source: Vec<&str> = include_str!("text_gateway.rs")
            .lines()
            .map(str::trim)
            .collect();
        let mut lines = guide.lines().map(str::trim);
        let mut count = 0;
        while let Some(line) = lines.next() {
            if line == "```rust,ignore" {
                let quote: Vec<&str> = lines.by_ref().take_while(|line| *line != "```").collect();
                assert!(
                    source.windows(quote.len()).any(|w| w == quote),
                    "not in text_gateway.rs:\n{}",
                    quote.join("\n")
                );
                count += 1;
            }
        }
        assert!(
            count > 0,
            "the provider guide quotes no code from text_gateway.rs"
        );
    }
}
