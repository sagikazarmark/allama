//! The Chat Completions decoder for transports that are not this crate's
//! HTTP client: `allama-cloudflare-worker` runs Workers AI models through
//! the Worker's AI binding, whose output is (or is normalized into) the
//! Chat Completions shape. Sharing the decoder keeps one mapping of that
//! shape (tool calls, reasoning, finish reasons, vendor fields) instead of
//! two that drift.
//!
//! This is not public API: it is `#[doc(hidden)]`, carries no stability
//! promise, and may change in any release. The crates that use it depend on
//! this one with an exact version requirement (`=`, set in the workspace
//! `Cargo.toml`), so a published pair can never be mixed.

use std::borrow::Cow;

use allama::{ChatResponse, Error, Result, StreamEvent};

use crate::quirks::CompatQuirks;
use crate::streaming::SseState;
use crate::wire::{ChatCompletionResponse, ResponseContext, from_api_response};

/// Decode a Chat Completions response body for `provider`, whose state
/// lives under `namespace`.
///
/// # Errors
///
/// [`Error::UnexpectedResponse`] when `body` is not a Chat Completions
/// response or has no choices.
pub fn decode_response(
    body: serde_json::Value,
    provider: impl Into<Cow<'static, str>>,
    namespace: impl Into<Cow<'static, str>>,
) -> Result<ChatResponse> {
    let provider = provider.into();
    let response: ChatCompletionResponse = serde_json::from_value(body).map_err(|err| {
        Error::unexpected_response("the response is not a chat completion")
            .authored()
            .with_upstream_message(err.to_string())
            .with_provider(provider.as_ref())
    })?;
    from_api_response(response, context(provider, namespace.into()))
}

/// A Chat Completions stream decoder fed one chunk at a time: the decoder
/// of this crate's own streams.
pub struct ChunkDecoder {
    state: SseState,
}

impl std::fmt::Debug for ChunkDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkDecoder").finish_non_exhaustive()
    }
}

impl ChunkDecoder {
    /// A decoder for one response of `provider`, whose state lives under
    /// `namespace`. All-zero usage chunks are dropped when
    /// `drop_zero_usage` is set.
    #[must_use]
    pub fn new(
        provider: impl Into<Cow<'static, str>>,
        namespace: impl Into<Cow<'static, str>>,
        drop_zero_usage: bool,
    ) -> Self {
        let namespace = namespace.into();
        let mut quirks = CompatQuirks::new(namespace.clone());
        quirks.drop_zero_usage_chunks = drop_zero_usage;
        Self {
            state: SseState::new(context(provider.into(), namespace), &quirks),
        }
    }

    /// Decode one chunk's JSON text, or `[DONE]`.
    pub fn chunk(&mut self, data: &str) -> Vec<Result<StreamEvent>> {
        self.state.process("", data)
    }

    /// Decode a named SSE event: `error` is an in-band error, and any
    /// other name a vendor event, surfaced as provider data.
    pub fn event(&mut self, name: &str, data: &str) -> Vec<Result<StreamEvent>> {
        self.state.process(name, data)
    }

    /// The body ended: an `Incomplete` error unless the stream is done.
    pub fn end(&mut self) -> Vec<Result<StreamEvent>> {
        allama_http::framing::ChatFrameDecoder::end(&mut self.state)
    }

    /// The stream failed with `error`: the open blocks are closed first.
    pub fn fail(&mut self, error: Error) -> Vec<Result<StreamEvent>> {
        allama_http::framing::ChatFrameDecoder::fail(&mut self.state, error)
    }

    /// Whether the stream is done (after `[DONE]`, a decode error, or an
    /// error).
    #[must_use]
    pub fn is_done(&self) -> bool {
        allama_http::framing::ChatFrameDecoder::is_done(&self.state)
    }
}

fn context(provider: Cow<'static, str>, namespace: Cow<'static, str>) -> ResponseContext {
    ResponseContext {
        provider,
        namespace,
        request_id: None,
        warnings: Vec::new(),
    }
}
