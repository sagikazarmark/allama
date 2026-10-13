//! Embedding primitives.
//!
//! The portable surface is intentionally narrow: a batch of inputs, an
//! optional output dimensionality, an optional input type (a search query or
//! a document), and ordered vectors. Other provider-specific
//! concerns (other task types, truncation, encoding) live in provider
//! options.
//!
//! The embedding types are exported at the crate root, which is their only
//! path (see [Paths](crate#paths)): [`EmbeddingProvider`](crate::EmbeddingProvider)
//! and [`EmbeddingProviderExt`](crate::EmbeddingProviderExt),
//! [`EmbeddingRequest`](crate::EmbeddingRequest) with its
//! [`EmbeddingInput`](crate::EmbeddingInput)s and
//! [`EmbeddingInputType`](crate::EmbeddingInputType),
//! [`EmbeddingResponse`](crate::EmbeddingResponse),
//! [`EmbeddingLimits`](crate::EmbeddingLimits) for batching, and
//! [`EmbeddingChunk`](crate::EmbeddingChunk) for chunked calls. The steps
//! adapters share around encoding a request and checking its vectors are in
//! [`adapter::embedding`](crate::adapter::embedding).

pub(crate) mod check;
pub(crate) mod chunks;
pub(crate) mod input;
pub(crate) mod preflight;
pub(crate) mod prepared;
pub(crate) mod progress;
pub(crate) mod provider;
pub(crate) mod request;
pub(crate) mod response;
mod role_prefix;

#[cfg(any(test, feature = "mock"))]
pub(crate) mod mock;
