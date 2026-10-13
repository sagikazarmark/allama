use serde::{Deserialize, Serialize};

use crate::error::EMBEDDING_METADATA_KEY;
use crate::{Error, Usage};

/// What a failed chunked embedding call
/// ([`embed_chunks`](crate::EmbeddingProviderExt::embed_chunks) or
/// [`embed_chunked`](crate::EmbeddingProviderExt::embed_chunked)) had done
/// before it failed: where to resume, and the usage billed. It never holds
/// vectors.
///
/// `embed_chunks` yields the chunks that succeeded before its error, and
/// `completed_inputs` is the offset to resume from.
/// `embed_chunked` is all-or-nothing: on failure it returns the error, never
/// the vectors of the chunks that succeeded. Both attach this record to the
/// error's [`ErrorContext::metadata`](crate::ErrorContext::metadata) under
/// `allama.embedding`, so the tokens already billed are not lost to cost
/// accounting; the redacted [`as_log`](Error::as_log) view keeps it. The
/// error itself is returned unchanged, so its kind,
/// [`is_retryable`](Error::is_retryable), and
/// [`retry_after`](Error::retry_after) still apply. Read it back with
/// [`from_error`](Self::from_error).
///
/// ```
/// # #[cfg(feature = "mock")] {
/// use allama::mock::MockEmbeddingProvider;
/// use allama::{
///     ChunkedEmbeddingProgress, EmbeddingProviderExt, EmbeddingRequest, EmbeddingResponse, Error,
///     Usage,
/// };
///
/// # async fn example() {
/// let provider = MockEmbeddingProvider::new([
///     Ok(EmbeddingResponse::new(vec![vec![0.1], vec![0.2]]).usage(Usage::new().input_tokens(8))),
///     Err(Error::rate_limited("slow down")),
/// ]);
/// let request = EmbeddingRequest::new("m").inputs(["a", "b", "c"]);
/// let err = provider.embed_chunked(&request, 2).await.unwrap_err();
/// assert!(err.is_retryable());
///
/// let progress = ChunkedEmbeddingProgress::from_error(&err).unwrap();
/// assert_eq!(progress.completed_inputs, 2);
/// assert_eq!(progress.usage.unwrap().input_tokens, Some(8));
/// # }
/// # }
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ChunkedEmbeddingProgress {
    /// The number of leading inputs whose chunks completed: their vectors
    /// arrived, one per input. Inputs from this index on were not embedded
    /// (or their chunk was rejected), so it is the offset to resume from:
    /// embed `request.inputs[completed_inputs..]`.
    pub completed_inputs: usize,
    /// The summed usage the provider reported for every chunk it answered,
    /// including a chunk rejected for its vector count. `None` when no
    /// answered chunk reported usage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

impl ChunkedEmbeddingProgress {
    /// The record `embed_chunked` attached to `error`, if any.
    ///
    /// `embed_chunked` attaches it to every error it returns, except a
    /// [`Serialization`](Error::Serialization) error, which has no context.
    /// For an error that ended a failover it is on that error's own context,
    /// not on the errors it [fell back from](Error::fallback_from).
    #[must_use]
    pub fn from_error(error: &Error) -> Option<Self> {
        let record = error.context()?.metadata.get(EMBEDDING_METADATA_KEY)?;
        Self::deserialize(record).ok()
    }

    /// Add the usage of an answered chunk.
    pub(crate) fn add_usage(&mut self, usage: Option<&Usage>) {
        if let Some(usage) = usage {
            self.usage = Some(self.usage.take().unwrap_or_default() + usage);
        }
    }

    /// Attach this record to an error that ended the call, for
    /// [`from_error`](Self::from_error).
    pub(crate) fn attach_to(&self, mut error: Error) -> Error {
        if let Some(context) = error.context_mut()
            && let Ok(record) = serde_json::to_value(self)
        {
            context
                .metadata
                .insert(EMBEDDING_METADATA_KEY.to_owned(), record);
        }
        error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_the_error_context() {
        let mut progress = ChunkedEmbeddingProgress::default();
        progress.add_usage(Some(&Usage::new().input_tokens(3)));
        progress.add_usage(None);
        progress.add_usage(Some(&Usage::new().input_tokens(4)));
        progress.completed_inputs = 5;
        let mut err = progress.attach_to(
            Error::overloaded("busy")
                .authored()
                .with_body("secret body"),
        );
        assert_eq!(ChunkedEmbeddingProgress::from_error(&err), Some(progress));
        let record = serde_json::json!({"completed_inputs": 5, "usage": {"input_tokens": 7}});
        assert_eq!(err.context().unwrap().metadata["allama.embedding"], record);
        assert!(matches!(err, Error::Overloaded { .. }));

        // The redacted log keeps the record (usage, not input), and still
        // leaves other metadata out.
        err.context_mut()
            .unwrap()
            .metadata
            .insert("vendor.detail".into(), serde_json::json!("may quote input"));
        let log = serde_json::to_value(err.as_log()).unwrap();
        assert_eq!(log["allama.embedding"], record);
        assert!(log.get("vendor.detail").is_none());
        assert_eq!(
            err.as_log().to_string(),
            "kind=overloaded body_len=11 allama.embedding.completed_inputs=5 \
             allama.embedding.input_tokens=7 message=\"busy\""
        );
    }

    #[test]
    fn rides_on_the_error_that_ended_a_failover() {
        let progress = ChunkedEmbeddingProgress {
            completed_inputs: 2,
            usage: None,
        };
        let err = progress
            .attach_to(Error::rate_limited("second").with_fallback_from(Error::timeout("first")));
        assert_eq!(ChunkedEmbeddingProgress::from_error(&err), Some(progress));
        assert_eq!(
            ChunkedEmbeddingProgress::from_error(err.fallback_from().unwrap()),
            None
        );
    }

    #[test]
    fn absent_without_a_record() {
        assert_eq!(
            ChunkedEmbeddingProgress::from_error(&Error::timeout("slow")),
            None
        );
    }
}
