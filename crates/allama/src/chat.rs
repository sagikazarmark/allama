//! Chat primitives: requests, responses, messages, content, and streaming.
//!
//! The chat types are exported at the crate root, which is their only path
//! (see [Paths](crate#paths)): [`ChatRequest`](crate::ChatRequest),
//! [`ChatResponse`](crate::ChatResponse), [`Message`](crate::Message),
//! [`ContentBlock`](crate::ContentBlock), [`ToolDefinition`](crate::ToolDefinition),
//! [`ChatProvider`](crate::ChatProvider), and the rest. This module holds
//! [`stream`], which defines the stream types (`StreamEvent`, `ChatStream`,
//! `ChatStreamExt`, …; import them from the root) beside the stream
//! collection and commit-point helpers.
//!
//! ```
//! use allama::{ChatRequest, Message};
//!
//! let request = ChatRequest::new("demo-model").user("Hi");
//! assert!(matches!(request.messages[0], Message::User(_)));
//! ```
//!
//! The steps adapters share around encoding a request (`prepare`,
//! `preflight`, `WireSupport`, …) are in
//! [`adapter::chat`](crate::adapter::chat).
//!
//! # Abnormal finishes
//!
//! A response that is not a complete answer still arrives as `Ok`, with
//! the reason in
//! [`ChatResponse::finish_reason`](crate::ChatResponse::finish_reason):
//! cut off at the output token limit
//! ([`FinishReason::Length`](crate::FinishReason::Length)) or at the
//! model's context window
//! ([`ContextWindow`](crate::FinishReason::ContextWindow)), stopped by a
//! content filter ([`ContentFilter`](crate::FinishReason::ContentFilter)),
//! declined by the model ([`Refusal`](crate::FinishReason::Refusal)), or
//! ended for a reason allama does not know
//! ([`Other`](crate::FinishReason::Other)). One call detects them all:
//! [`ChatResponse::finished_normally`](crate::ChatResponse::finished_normally),
//! which is also `false` when the response reports no finish reason.
//!
//! A stream that ends with its terminal marker but no finish reason is
//! `Ok` too, with a warning coded
//! [`MISSING_FINISH_REASON`](crate::adapter::chat::MISSING_FINISH_REASON); one that ends
//! without its terminal marker fails with
//! [`StreamErrorKind::Incomplete`](crate::StreamErrorKind::Incomplete).
//!
//! # Finish events
//!
//! With the default `warning-events` feature, a call that finishes at the
//! output token limit, at the context window, at a content filter or with a
//! refusal is also a
//! [`tracing`](https://docs.rs/tracing) event at `INFO`, under the target
//! `allama::finish`, so truncation in production shows up without reading
//! every response. Its fields are `allama.finish.provider`,
//! `allama.finish.model`, `allama.finish.reason` (the normalized reason)
//! and `allama.finish.reason_raw` (the provider's). At the output token
//! limit, the message also says how much of the output went to reasoning
//! when the response's usage reports it ("400 of 400 output tokens went to
//! reasoning"; a stream only when the usage arrives with the finish
//! reason). [`Prepared::finish`](crate::adapter::chat::Prepared::finish)
//! and [`finish_stream`](crate::adapter::chat::Prepared::finish_stream)
//! report it, once per call:
//! the innermost layer built on
//! [`prepare`](crate::adapter::chat::prepare) that sees the finish reports
//! it. A dry run reports nothing.

pub(crate) mod cache;
pub(crate) mod content;
#[cfg(feature = "chat-extract")]
pub mod extract;
pub(crate) mod message;
pub(crate) mod preflight;
pub(crate) mod prepared;
pub(crate) mod provider;
pub(crate) mod request;
pub(crate) mod response;
pub mod stream;
pub(crate) mod system;
pub(crate) mod tool;
pub(crate) mod tool_call_id;
pub(crate) mod transcript;
pub(crate) mod turns;
pub(crate) mod wire;

#[cfg(test)]
mod accounted;

#[cfg(any(test, feature = "mock"))]
pub(crate) mod mock;
