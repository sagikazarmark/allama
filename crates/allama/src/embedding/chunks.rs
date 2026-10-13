use std::borrow::Cow;

use futures_core::Stream;

use super::check::check_vectors;
use crate::{
    ChunkedEmbeddingProgress, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse, Error,
    ProviderIdentity, Result, Subject,
};

/// One chunk of a request embedded by
/// [`embed_chunks`](crate::EmbeddingProviderExt::embed_chunks): the
/// response for the inputs from `offset` on, one vector per input.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct EmbeddingChunk {
    /// The index in `request.inputs` of the chunk's first input.
    pub offset: usize,
    /// The chunk's response: one vector per input of the chunk, in order.
    /// A warning about an input ([`Subject::EmbeddingInput`]) names its
    /// index in the whole request, not in the chunk.
    pub response: EmbeddingResponse,
}

impl EmbeddingChunk {
    /// The index in `request.inputs` after the chunk's last input: where
    /// the next chunk starts.
    #[must_use]
    pub fn end(&self) -> usize {
        self.offset + self.response.embeddings.len()
    }
}

/// The chunks of `request`, embedded one after another (see
/// [`embed_chunks`](crate::EmbeddingProviderExt::embed_chunks)).
pub(crate) fn chunks<'a, P>(
    provider: &'a P,
    request: &'a EmbeddingRequest,
    max_chunk_inputs: usize,
) -> impl Stream<Item = Result<EmbeddingChunk>> + Send + 'a
where
    P: EmbeddingProvider + Sync + ?Sized,
{
    let chunk_size = chunk_size(provider, request, max_chunk_inputs);
    let state = State {
        offset: 0,
        size: None,
        progress: ChunkedEmbeddingProgress::default(),
        done: false,
    };
    futures_util::stream::unfold(state, move |mut state| async move {
        let total = request.inputs.len();
        // The first chunk always runs, so a request without inputs fails as
        // the provider fails it.
        if state.done || (state.offset > 0 && state.offset >= total) {
            return None;
        }
        let offset = state.offset;
        let end = total.min(offset + chunk_size);
        let part = if offset == 0 && end == total {
            Cow::Borrowed(request)
        } else {
            Cow::Owned(slice(request, offset, end))
        };
        let result = match provider.embed(&part).await {
            Ok(response) => check_chunk(
                provider,
                &part,
                response,
                &mut state.size,
                &mut state.progress,
            ),
            Err(err) => Err(state.progress.attach_to(err)),
        };
        let item = result.map(|response| {
            state.offset = end;
            chunk_at(offset, response)
        });
        state.done = item.is_err() || end >= total;
        Some((item, state))
    })
}

/// The chunk size: `max_chunk_inputs`, at most the provider's limit, and
/// at least one.
fn chunk_size<P: EmbeddingProvider + ?Sized>(
    provider: &P,
    request: &EmbeddingRequest,
    max_chunk_inputs: usize,
) -> usize {
    provider
        .embedding_limits(&request.model)
        .max_inputs_per_call
        .map_or(max_chunk_inputs, |max| max.min(max_chunk_inputs))
        .max(1)
}

/// `response` as the chunk at `offset`, its input warnings renumbered to
/// the whole request.
fn chunk_at(offset: usize, mut response: EmbeddingResponse) -> EmbeddingChunk {
    for warning in &mut response.warnings {
        if let Subject::EmbeddingInput { input, .. } = &mut warning.subject {
            *input += offset;
        }
    }
    EmbeddingChunk { offset, response }
}

/// What [`chunks`] carries from one chunk to the next.
struct State {
    /// The index of the next chunk's first input.
    offset: usize,
    /// The size of the vectors so far.
    size: Option<usize>,
    progress: ChunkedEmbeddingProgress,
    done: bool,
}

/// `request` with only its inputs `start..end`.
fn slice(request: &EmbeddingRequest, start: usize, end: usize) -> EmbeddingRequest {
    // A literal on purpose: a new request field must be carried to chunks.
    EmbeddingRequest {
        model: request.model.clone(),
        inputs: request.inputs[start..end].to_vec(),
        dimensions: request.dimensions,
        input_type: request.input_type,
        unsupported_policy: request.unsupported_policy,
        options: request.options.clone(),
    }
}

/// Check that the chunk `part` (the request's inputs from
/// `progress.completed_inputs` on) came back with one vector each, all of
/// the `size` of the chunks before it ([`check_vectors`]), and record it in
/// `progress`.
fn check_chunk<P: ProviderIdentity + ?Sized>(
    provider: &P,
    part: &EmbeddingRequest,
    response: EmbeddingResponse,
    size: &mut Option<usize>,
    progress: &mut ChunkedEmbeddingProgress,
) -> Result<EmbeddingResponse> {
    progress.add_usage(response.usage.as_ref());
    let inputs = part.inputs.len();
    let fail = |err: Error, progress: &ChunkedEmbeddingProgress| {
        progress.attach_to(err.with_provider(provider.provider_name().to_owned()))
    };
    if response.embeddings.len() != inputs {
        let err = Error::unexpected_response(format!(
            "the response holds {} embeddings for {inputs} inputs (inputs {}..{}) of a chunked \
             embedding request",
            response.embeddings.len(),
            progress.completed_inputs,
            progress.completed_inputs + inputs,
        ))
        .authored();
        return Err(fail(err, progress));
    }
    *size = check_vectors(
        &part.model,
        part.dimensions,
        *size,
        progress.completed_inputs,
        &response.embeddings,
    )
    .map_err(|err| fail(err, progress))?;
    progress.completed_inputs += inputs;
    Ok(response)
}

#[cfg(test)]
mod tests {
    use futures_util::StreamExt;

    use super::*;
    use crate::mock::MockEmbeddingProvider;
    use crate::{EmbeddingProviderExt, Usage, Warning, WarningKind};

    fn answer(vectors: &[f32], tokens: u64) -> Result<EmbeddingResponse> {
        Ok(
            EmbeddingResponse::new(vectors.iter().map(|&v| vec![v, v]).collect())
                .usage(Usage::new().input_tokens(tokens)),
        )
    }

    async fn collect(
        provider: &MockEmbeddingProvider,
        request: &EmbeddingRequest,
        max: usize,
    ) -> Vec<Result<EmbeddingChunk>> {
        provider.embed_chunks(request, max).collect().await
    }

    /// A failure mid-way ends the stream after the chunks that succeeded;
    /// its progress names the offset to resume from, and resuming there
    /// embeds exactly the rest.
    #[tokio::test]
    async fn a_failure_mid_way_can_be_resumed() {
        let request = EmbeddingRequest::new("m").inputs(["a", "b", "c", "d", "e"]);
        let provider = MockEmbeddingProvider::new([
            answer(&[1.0, 2.0], 3),
            Err(Error::rate_limited("slow down")),
            answer(&[3.0, 4.0], 4),
        ]);
        let chunks = collect(&provider, &request, 2).await;
        assert_eq!(chunks.len(), 2, "the stream ends after the error");
        let first = chunks[0].as_ref().unwrap();
        assert_eq!((first.offset, first.end()), (0, 2));
        let err = chunks[1].as_ref().unwrap_err();
        assert!(matches!(err, Error::RateLimited { .. }), "{err}");
        let progress = ChunkedEmbeddingProgress::from_error(err).unwrap();
        assert_eq!(progress.completed_inputs, 2);
        assert_eq!(progress.usage.unwrap().input_tokens, Some(3));
        // Nothing was asked after the failure.
        assert_eq!(provider.requests().len(), 2);

        // Resume: the rest of the inputs, in a new request.
        let mut rest = request.clone();
        rest.inputs.drain(..progress.completed_inputs);
        provider.push_response(answer(&[5.0], 1));
        let resumed = collect(&provider, &rest, 2).await;
        let resumed: Vec<_> = resumed.into_iter().map(Result::unwrap).collect();
        assert_eq!(resumed.iter().map(|c| c.offset).collect::<Vec<_>>(), [0, 2]);
        let sent: Vec<_> = provider.requests()[2..]
            .iter()
            .map(|r| r.inputs.clone())
            .collect();
        assert_eq!(sent, [rest.inputs[..2].to_vec(), rest.inputs[2..].to_vec()]);
    }

    #[tokio::test]
    async fn chunks_carry_their_offset_and_the_request_fields() {
        let hint = Warning::unsupported(crate::RequestField::InputType, "no input type")
            .with_provider("x");
        let image = |input| {
            Warning::new(
                WarningKind::Approximated,
                Subject::EmbeddingInput {
                    input,
                    part: Some(0),
                },
                "image resized",
            )
        };
        let provider = MockEmbeddingProvider::new([
            answer(&[1.0, 2.0], 1).map(|r| r.warning(hint.clone())),
            answer(&[3.0], 1).map(|r| r.warning(hint.clone()).warning(image(0))),
        ])
        // The provider's limit wins over a larger chunk size.
        .with_max_inputs_per_call(2);
        let request = EmbeddingRequest::new("m")
            .inputs(["a", "b", "c"])
            .dimensions(2)
            .input_type(crate::EmbeddingInputType::Query);
        let chunks: Vec<_> = collect(&provider, &request, 10)
            .await
            .into_iter()
            .map(Result::unwrap)
            .collect();
        assert_eq!(chunks.len(), 2);
        assert_eq!((chunks[1].offset, chunks[1].end()), (2, 3));
        // Each chunk keeps its warnings; input warnings name the input in
        // the whole request.
        assert_eq!(chunks[1].response.warnings, vec![hint, image(2)]);
        for sent in provider.requests() {
            assert_eq!(sent.dimensions, Some(2));
            assert_eq!(sent.input_type, Some(crate::EmbeddingInputType::Query));
        }
    }

    #[tokio::test]
    async fn a_chunk_of_another_size_than_the_first_ends_the_stream() {
        let provider = MockEmbeddingProvider::new([
            answer(&[1.0, 2.0], 2),
            Ok(EmbeddingResponse::new(vec![vec![0.1, 0.2, 0.3]; 2])),
        ]);
        let request = EmbeddingRequest::new("m").inputs(["a", "b", "c", "d"]);
        let chunks = collect(&provider, &request, 2).await;
        assert!(chunks[0].is_ok());
        let err = chunks[1].as_ref().unwrap_err();
        assert!(matches!(err, Error::UnexpectedResponse { .. }), "{err}");
        assert_eq!(
            ChunkedEmbeddingProgress::from_error(err)
                .unwrap()
                .completed_inputs,
            2
        );
    }

    /// The provider fails a request without inputs, and the stream reports
    /// it rather than ending empty.
    #[tokio::test]
    async fn a_request_without_inputs_is_sent_once() {
        let provider =
            MockEmbeddingProvider::new([Err(Error::invalid_request("no inputs").authored())]);
        let chunks = collect(&provider, &EmbeddingRequest::new("m"), 2).await;
        assert_eq!(chunks.len(), 1);
        assert!(matches!(chunks[0], Err(Error::InvalidRequest { .. })));
    }

    /// A batch within one chunk is sent as the caller's request.
    #[tokio::test]
    async fn a_single_chunk_is_the_request_itself() {
        let provider = MockEmbeddingProvider::new([answer(&[1.0, 2.0], 1)]);
        let request = EmbeddingRequest::new("m").inputs(["a", "b"]);
        let chunks = collect(&provider, &request, 2).await;
        assert_eq!(chunks.len(), 1);
        assert_eq!(provider.requests(), [request]);
    }
}
